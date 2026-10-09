"""
`price_portfolio`'s pipeline: ORE's semantics end to end, for every configuration.

    Market + CamConfig --calibrate CAM--> CrossAssetModel --simulate--> ScenarioMarket
        --ValuationEngine (engine.valuation)--> t=0 NPVs and the NPV cube (base currency)
        --ExposureCalculator (engine.risk.exposure)--> EPE, ENE, EE_B, EEE_B, EPE_B, EEPE_B, PFE

The model of each currency is `config.simulation.ir[ccy]`: ORE's LGM (the default) or the
Hull-White model (`HullWhiteConfig`), both in the cross-asset model with the exact
discretization and the LGM numeraire. The valuation reads only the scenario market, so every
engine prices under either model.

Greeks by `config.greeks.method`: ORE's bump-and-revalue sensitivities on today's market
(`engine.risk.sensitivities`, the default) or automatic differentiation
(`engine.risk.greeks`).

Every choice comes from the request's run configuration (`engine.portfolio.config`):
`simulation` (the CAM, its model per currency, the grid), `pricing` (the engine per product),
`greeks` (the method and ORE's sensitivity settings), `precision` and the reporting currency.
What the pipeline does not implement is refused before any work (`validate_request`).

Precision (`engine.precision`, docs/planning/details/precision.md): `simulate` and
`value_portfolio` hold the cast points of the simulation, market and pricing stages; the
calibration, t=0 values and Greeks are float64; every reduction over paths loads the stored
cube and the numeraire at float64 first (`_exposures`), and so does the result's `npv_cube`.
With `precision.paired_fraction > 0` the first paths are simulated and priced again at float64
throughout (`_paired_sample`, decision A-13): the exposure means are the two-level estimates,
and each figure's estimate is in the result's `PrecisionReport`, which every result carries.
"""
import dataclasses
from typing import TYPE_CHECKING, List, Optional, Sequence

import jax.numpy as jnp
import numpy as np

from engine.market import Market
from engine.models.curves import ZeroCurve, discount
from engine.precision import PrecisionReport, format_name, load, paired_paths, realized_format
from engine.portfolio.profiling import phase
from engine.risk.exposure import ExposureProfile, PairedPaths, exposure_profile, netting_set_profile
from engine.risk.var_es import ENGINE_RISK_MEASURE
from engine.simulation.config import build_cross_asset_model, simulate
from engine.valuation.context import simulation_market_today
from engine.valuation.portfolio import trade_maturity, validate_trades, value_paths, value_portfolio, value_today

if TYPE_CHECKING:
    from engine.portfolio.request import PortfolioResult


def price_on_market(request) -> "PortfolioResult":
    """`price_portfolio`'s work (see the module docstring)."""
    from engine.portfolio.request import PortfolioResult

    validate_request(request)
    market: Market = request.market
    run = request.config
    trades = list(request.trades)
    base = run.reporting_currency
    greeks = None
    exposure, trade_exposures = None, []
    if request.scenario_risk:
        simulation = run.simulation
        with phase("calibration"):
            model = build_cross_asset_model(market, simulation)
        with phase("simulation"):
            scenarios = simulate(market, simulation, model, run.precision)
        with phase("pricing"):
            valuation = value_portfolio(trades, market, scenarios, base, run.pricing, simulation.swaption_vol_decay,
                                        run.precision)
        paired = paired_paths(scenarios.num_paths, run.precision.paired_fraction)
        sample = None
        if paired:
            with phase("paired_sample"):
                sample = _paired_sample(trades, market, model, run, paired)
        today, cube = valuation.today, load(valuation.cube, jnp.float64)
        with phase("exposure"):
            start = value_today(trades, market, base, run.pricing,
                                simulation_market_today(market, simulation.curve_tenors))
            exposure, trade_exposures = _exposures(trades, start, cube, scenarios, market, base, request.pfe_quantiles,
                                                   sample)
        figures = {f"netting_set/{k}": v for k, v in exposure.estimates.items()}
        figures.update({f"trades/{t.trade_id}/{k}": v
                        for t, profile in zip(trades, trade_exposures) for k, v in profile.estimates.items()})
        realized = {**scenarios.simulation_formats, "market": realized_format(scenarios.path_arrays()),
                    **{f"values/{t.trade_id}": format_name(c) for t, c in zip(trades, valuation.columns)}}
        report = PrecisionReport.of(run.precision, trades, realized, valuation.columns, scenarios.num_paths,
                                    paired, figures)
    else:
        with phase("base_npv"):
            today = value_today(trades, market, base, run.pricing)
        cube = jnp.zeros((0, 0, 0))
        report = PrecisionReport.of(run.precision, trades, {}, (), paths=0)
    if request.compute_greeks:
        with phase("greeks"):
            greeks = _greeks(run.greeks.method)(trades, market, base, run.pricing, run.greeks.sensitivity)
    return PortfolioResult(
        base_npv=float(np.sum(today)), npv_cube=cube, exposure=exposure, trade_exposures=trade_exposures,
        greeks=greeks, warnings=[], base_npv_per_trade=list(today), scenario_risk_available=request.scenario_risk,
        measure=ENGINE_RISK_MEASURE if request.scenario_risk else None, precision=report,
    )


def _paired_sample(trades: Sequence, market: Market, model, run, paths: int) -> PairedPaths:
    """The paired float64 sample (decision A-13): the first `paths` paths simulated and priced
    again at float64 throughout, on the same calibrated model, as the cube `[n, D, T]` and the
    numeraire `[n, D]`. A Sobol sequence's first points do not depend on the sample size, and
    every kernel is per path, so these are exactly the paths a float64 run would give."""
    simulation = run.simulation
    scenarios = simulate(market, dataclasses.replace(simulation, samples=paths), model)
    columns = value_paths(trades, market, scenarios, run.reporting_currency, run.pricing,
                          simulation.swaption_vol_decay)
    return jnp.stack(columns, axis=-1), scenarios.numeraire


def _greeks(method: str):
    """The Greeks function of a method: ORE's bump-and-revalue or AD, same signature."""
    if method == "AD":
        from engine.risk.greeks import portfolio_greeks
        return portfolio_greeks
    from engine.risk.sensitivities import portfolio_sensitivities
    return portfolio_sensitivities


def validate_request(request) -> None:
    """Refuse, before any JAX work, a request the pipeline cannot price: a setting it does
    not implement: scenario risk without a simulation, a trade the market cannot value or its
    engine refuses, a precision override naming no trade or product (`validate_trades`), or a
    reporting currency the market lacks (the configuration itself is validated when it is
    built). The HTTP route runs it synchronously
    so such a request is a 400, not a failed job."""
    run = request.config
    if request.scenario_risk and run.simulation is None:
        raise ValueError("scenario_risk needs config.simulation (a CamConfig); set scenario_risk=False for "
                         "today's NPVs and Greeks only")
    validate_trades(request.trades, request.market, run.pricing, run.precision)
    request.market.currency(run.reporting_currency)


def _exposures(trades: Sequence, start: List[float], cube, scenarios, market: Market, base: str, quantiles,
               paired: Optional[PairedPaths] = None) -> "tuple[ExposureProfile, List[ExposureProfile]]":
    """Netting-set and per-trade profiles, deflated by the LGM numeraire, EE_B against the base
    currency's discount curve, time weights on the simulation dates; with the `paired` float64
    sample, their two-level estimates. Reductions over paths: `cube` is float64 and the stored
    numeraire is loaded at float64 (cast point 5).

    The profiles start from `start`, each trade's value on the simulation market of the as-of
    date (its curves sampled at the simulation tenors), as ORE's start from the cube's `T0`
    (`ValuationEngine::buildCube` prices t=0 on the `ScenarioSimMarket`). The result's
    `base_npv` stays the value on today's market, ORE's NPV analytic."""
    curve = ZeroCurve.from_config(market.currency(base).discount_curve)
    p0 = discount(curve, jnp.asarray(scenarios.times, dtype=jnp.float64))
    numeraire = load(scenarios.numeraire, jnp.float64)
    common = dict(numeraire=numeraire, discount=p0, times=scenarios.times, quantiles=quantiles,
                  dates=scenarios.dates, asof=market.asof)
    netting_set = netting_set_profile(cube, start, paired=paired, **common)
    per_trade = [exposure_profile(cube[:, :, i], start[i], maturity=trade_maturity(trade),
                                  paired=None if paired is None else (paired[0][:, :, i], paired[1]), **common)
                 for i, trade in enumerate(trades)]
    return netting_set, per_trade
