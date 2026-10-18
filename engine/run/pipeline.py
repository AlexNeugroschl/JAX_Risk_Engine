"""
`price_portfolio`'s pipeline: ORE's semantics end to end, for every configuration.

    Market + CamConfig --calibrate CAM--> CrossAssetModel --simulate--> ScenarioMarket
        --ValuationEngine (engine.pricing)--> t=0 NPVs and the NPV cube (base currency)
        --ExposureCalculator (engine.risk.counterparty.exposure)--> EPE, ENE, EE_B, EEE_B, EPE_B, EEPE_B, PFE

The model of each currency is `config.simulation.ir[ccy]`: ORE's LGM (the default) or the
Hull-White model (`HullWhiteConfig`), both in the cross-asset model with the exact
discretization and the LGM numeraire. The valuation reads only the scenario market, so every
engine prices under either model.

Greeks by `config.greeks.method`: ORE's bump-and-revalue sensitivities on today's market
(`engine.risk.greeks.bump`, the default) or automatic differentiation
(`engine.risk.greeks.ad`).

Every choice comes from the request's run configuration (`engine.run.config`):
`simulation` (the CAM, its model per currency, the grid), `pricing` (the engine per product),
`greeks` (the method and ORE's sensitivity settings), `precision` and the reporting currency.
What the pipeline does not implement is refused before any work (`validate_request`).

Precision (`engine.precision`, docs/planning/details/precision.md): `simulate` and
`value_portfolio` hold the cast points of the simulation, market and pricing stages; the
calibration, t=0 values and Greeks are float64; every reduction over paths loads one stored
column and the numeraire at float64 (`_exposures`). Only a requested `npv_cube` is assembled.
With `precision.paired_fraction > 0` the first paths are simulated and priced again at float64
throughout (`_paired_sample`, decision A-13): the exposure means are the two-level estimates,
and each figure's estimate is in the result's `PrecisionReport`, which every result carries.
The result's `warnings` name every figure whose precision has no passing row of the evidence
table under the run's pricing settings (`precision_warnings`, I-55, I-114; a float64 run has
none), and an AD run's Greeks that are not ORE's (`engine.risk.greeks.ad.ad_warnings`, I-78).
"""
import dataclasses
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence

import jax.numpy as jnp
import numpy as np

from engine.market_data.curves import ZeroCurve, discount
from engine.market_data.market import Market
from engine.market_simulation.config import build_cross_asset_model, simulate
from engine.precision import (
    STAGES, Precision, PrecisionReport, RunFigures, evidence_warnings, format_name, load, paired_error_warning,
    realized_format,
)
from engine.precision.estimate import paired_count
from engine.pricing.config import PricingConfig
from engine.pricing.context import simulation_market_today
from engine.pricing.cube import (
    reporting_fx_spots, trade_maturity, validate_trades, value_paths, value_portfolio, value_today,
)
from engine.risk.counterparty.exposure import (
    CORRECTED_FIGURES, ExposureProfile, PairedColumns, column_profiles, figure_names,
)
from engine.risk.market.var_es import ENGINE_RISK_MEASURE
from engine.run.trace import phase

if TYPE_CHECKING:
    from engine.run.request import PortfolioResult


def price_on_market(request) -> "PortfolioResult":
    """`price_portfolio`'s work (see the module docstring)."""
    from engine.run.request import PortfolioResult

    validate_request(request)
    market: Market = request.market
    run = request.config
    trades = list(request.trades)
    base = run.reporting_currency
    greeks = None
    exposure, trade_exposures, exercised = None, [], {}
    if request.scenario_risk:
        simulation = run.simulation
        with phase("calibration"):
            model = build_cross_asset_model(market, simulation)
        with phase("simulation"):
            scenarios = simulate(market, simulation, model, run.precision)
        with phase("pricing"):
            valuation = value_portfolio(trades, market, scenarios, base, run.pricing, simulation.swaption_vol_decay,
                                        run.precision)
        paired = paired_count(scenarios.num_paths, run.precision.paired_fraction, simulation.replicates)
        sample = None
        if paired:
            with phase("paired_sample"):
                sample = _paired_sample(trades, market, model, run, paired)
        today, exercised = valuation.today, valuation.exercised
        with phase("exposure"):
            start = value_today(trades, market, base, run.pricing,
                                simulation_market_today(market, simulation.curve_tenors))
            exposure, trade_exposures = _exposures(trades, start, valuation.columns, scenarios, market, base,
                                                   request.pfe_quantiles,
                                                   sample, simulation.replicates)
        # Assemble only an explicitly requested output, after the reduction working set
        # has been released. The float64 sample also stays columnar throughout.
        del sample
        cube = valuation.float64_cube() if request.include_npv_cube else jnp.zeros((0, 0, 0), dtype=jnp.float64)
        figures = {f"netting_set/{k}": v for k, v in exposure.estimates.items()}
        figures.update({f"trades/{t.trade_id}/{k}": v
                        for t, profile in zip(trades, trade_exposures) for k, v in profile.estimates.items()})
        realized = {**scenarios.simulation_formats, "market": realized_format(scenarios.path_arrays()),
                    **{f"values/{t.trade_id}": format_name(c) for t, c in zip(trades, valuation.columns)}}
        computed = [run.precision.simulation.compute, run.precision.market.compute,
                    *(run.precision.precision_for(t).compute for t in trades)]
        report = PrecisionReport.of(run.precision, trades, realized, valuation.columns, scenarios.num_paths,
                                    paired, figures, computed)
        report = dataclasses.replace(report, replicates=simulation.replicates,
                                     uncertainty_method="independent_scrambles" if simulation.replicates > 1 else "unavailable")
        warnings = precision_warnings(run.precision, trades, scenarios.num_paths, request.pfe_quantiles,
                                       pricing_settings(run.pricing, trades), simulation.replicates)
    else:
        with phase("base_npv"):
            today = value_today(trades, market, base, run.pricing)
        cube = jnp.zeros((0, 0, 0))
        report = PrecisionReport.of(run.precision, trades, {}, (), paths=0)
        warnings = []
    if request.compute_greeks:
        with phase("greeks"):
            greeks = _greeks(run.greeks.method)(trades, market, base, run.pricing, run.greeks.sensitivity)
        if run.greeks.method == "AD":
            from engine.risk.greeks.ad import ad_warnings
            warnings = warnings + ad_warnings(trades)
    return PortfolioResult(
        base_npv=float(np.sum(today)), npv_cube=cube, exposure=exposure, trade_exposures=trade_exposures,
        greeks=greeks, warnings=warnings, base_npv_per_trade=list(today),
        scenario_risk_available=request.scenario_risk, measure=ENGINE_RISK_MEASURE if request.scenario_risk else None,
        precision=report, exercised=exercised,
    )


def run_figures(quantiles: Sequence[float]) -> RunFigures:
    """The figures a portfolio run reports, as the evidence table names them (I-55), each
    trade's and the netting set's: the cube's mean per date ("NPV", the values a storage format
    of the cube holds) and the exposure figures with PFE at `quantiles` (`figure_names`), of
    which the paired sample corrects `CORRECTED_FIGURES`; every adjustable stage."""
    names = ("NPV",) + figure_names(quantiles)
    return RunFigures(stages=STAGES, aggregate="netting_set", trade=names, total=names, corrected=CORRECTED_FIGURES,
                      legend="NPV here is the paths' mean value per simulation date (the cube); today's NPV is "
                             "float64 at every precision.")


def pricing_settings(pricing: PricingConfig, trades: Sequence) -> Dict[str, str]:
    """The pricing settings of each product of `trades` that change its kernel's arithmetic, as
    the evidence table keys a portfolio run's rows (`PricingConfig.kernel_settings`, I-114): a
    Bermudan or American keeps on every path the model calibrated when it was built, as ORE's
    exposure simulation keeps it (A-24)."""
    return pricing.kernel_settings({t.product for t in trades}, recalibrated=False)


def precision_warnings(precision: Precision, trades: Sequence, paths: int, quantiles: Sequence[float],
                       settings: Mapping[str, str], replicates: int = 1) -> List[str]:
    """A portfolio run's precision warnings: every figure (`run_figures`) without a passing row
    in the evidence table (`engine.precision.evidence`, I-55) under the run's pricing `settings`
    (`pricing_settings`), and the label of the paired correction's unavailable or unvalidated
    sampling uncertainty (I-111)."""
    warnings = evidence_warnings(precision, trades, paths, run_figures(quantiles), settings=settings,
                                 replicates=replicates)
    label = paired_error_warning(precision, replicates)
    if label:
        warnings.append(label)
    return warnings


def _paired_sample(trades: Sequence, market: Market, model, run, paths: int) -> PairedColumns:
    """The paired float64 sample (decision A-13): the first `paths` paths simulated and priced
    again at float64 throughout, on the same calibrated model, as columns `[n, D]` and the
    numeraire `[n, D]`. A Sobol sequence's first points do not depend on the sample size, and
    every kernel is per path, so these are exactly the paths a float64 run would give."""
    simulation = run.simulation
    scenarios = simulate(market, dataclasses.replace(simulation, samples=paths), model)
    columns = value_paths(trades, market, scenarios, run.reporting_currency, run.pricing,
                          simulation.swaption_vol_decay)
    return columns, scenarios.numeraire


def _greeks(method: str):
    """The Greeks function of a method: ORE's bump-and-revalue or AD, same signature."""
    if method == "AD":
        from engine.risk.greeks.ad import portfolio_greeks
        return portfolio_greeks
    from engine.risk.greeks.bump import portfolio_sensitivities
    return portfolio_sensitivities


def validate_request(request) -> None:
    """Refuse, before any JAX work, a request the pipeline cannot price: a setting it does
    not implement: scenario risk without a simulation, or with a trade in a currency the
    simulation does not model; a trade the market cannot value or its engine refuses, a
    precision override naming no trade or product (`validate_trades`); a reporting currency the
    market lacks, or a trade with no FX spot into it (`reporting_fx_spots`). The configuration
    itself is validated when it is built. The HTTP route runs it synchronously so such a request
    is a 400, not a failed job."""
    run = request.config
    if request.scenario_risk and run.simulation is None:
        raise ValueError("scenario_risk needs config.simulation (a CamConfig); set scenario_risk=False for "
                         "today's NPVs and Greeks only")
    validate_trades(request.trades, request.market, run.pricing, run.precision)
    if request.scenario_risk:
        simulation = run.simulation
        run.precision.validate_sampling(simulation.samples, simulation.replicates, "simulation")
        for cfg in request.trades:
            if cfg.currency not in run.simulation.ir:
                raise ValueError(f"trade {cfg.trade_id!r} is in {cfg.currency}, which the simulation does not model; "
                                 f"give config.simulation.ir[{cfg.currency!r}] a model (and its FX volatility)")
    reporting_fx_spots(request.trades, request.market, run.reporting_currency, "base_currency")


def _exposures(trades: Sequence, start: List[float], columns, scenarios, market: Market, base: str, quantiles,
               paired: Optional[PairedColumns] = None, replicates: int = 1) -> "tuple[ExposureProfile, List[ExposureProfile]]":
    """Netting-set and per-trade profiles, deflated by the LGM numeraire, EE_B against the base
    currency's discount curve, time weights on the simulation dates; with the `paired` float64
    sample, their two-level estimates. Reductions over paths load `columns` individually and
    the stored numeraire at float64 (cast point 5).

    The profiles start from `start`, each trade's value on the simulation market of the as-of
    date (its curves sampled at the simulation tenors), as ORE's start from the cube's `T0`
    (`ValuationEngine::buildCube` prices t=0 on the `ScenarioSimMarket`). The result's
    `base_npv` stays the value on today's market, ORE's NPV analytic."""
    curve = ZeroCurve.from_config(market.currency(base).discount_curve)
    p0 = discount(curve, jnp.asarray(scenarios.times, dtype=jnp.float64))
    numeraire = load(scenarios.numeraire, jnp.float64)
    common = dict(numeraire=numeraire, discount=p0, times=scenarios.times, quantiles=quantiles,
                  dates=scenarios.dates, asof=market.asof)
    return column_profiles(columns, start, maturities=[trade_maturity(t) for t in trades], paired=paired,
                           replicates=replicates, **common)
