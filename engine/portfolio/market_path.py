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
"""
from typing import TYPE_CHECKING, List, Sequence

import jax.numpy as jnp
import numpy as np

from engine.market import Market
from engine.models.curves import ZeroCurve, discount
from engine.precision import load
from engine.portfolio.profiling import phase
from engine.risk.exposure import ExposureProfile, exposure_profile, netting_set_profile
from engine.risk.var_es import ENGINE_RISK_MEASURE
from engine.simulation.config import build_cross_asset_model, simulate
from engine.valuation.portfolio import validate_trades, value_portfolio, value_today

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
        today, cube = valuation.today, load(valuation.cube, jnp.float64)
        with phase("exposure"):
            exposure, trade_exposures = _exposures(trades, today, cube, scenarios, market, base, request.pfe_quantiles)
    else:
        with phase("base_npv"):
            today = value_today(trades, market, base, run.pricing)
        cube = jnp.zeros((0, 0, 0))
    if request.compute_greeks:
        with phase("greeks"):
            greeks = _greeks(run.greeks.method)(trades, market, base, run.pricing, run.greeks.sensitivity)
    return PortfolioResult(
        base_npv=float(np.sum(today)), npv_cube=cube, exposure=exposure, trade_exposures=trade_exposures,
        greeks=greeks, warnings=[], base_npv_per_trade=list(today), scenario_risk_available=request.scenario_risk,
        measure=ENGINE_RISK_MEASURE if request.scenario_risk else None,
    )


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
    engine refuses (`validate_trades`), or a reporting currency the market lacks (the
    configuration itself is validated when it is built). The HTTP route runs it synchronously
    so such a request is a 400, not a failed job."""
    run = request.config
    if request.scenario_risk and run.simulation is None:
        raise ValueError("scenario_risk needs config.simulation (a CamConfig); set scenario_risk=False for "
                         "today's NPVs and Greeks only")
    validate_trades(request.trades, request.market, run.pricing)
    request.market.currency(run.reporting_currency)


def _exposures(trades: Sequence, today: List[float], cube, scenarios, market: Market, base: str,
               quantiles) -> "tuple[ExposureProfile, List[ExposureProfile]]":
    """Netting-set and per-trade profiles, deflated by the LGM numeraire, EE_B against the base
    currency's discount curve, time weights on the simulation dates. Reductions over paths:
    `cube` is float64 and the stored numeraire is loaded at float64 (cast point 5)."""
    curve = ZeroCurve.from_config(market.currency(base).discount_curve)
    p0 = discount(curve, jnp.asarray(scenarios.times, dtype=jnp.float64))
    numeraire = load(scenarios.numeraire, jnp.float64)
    common = dict(numeraire=numeraire, discount=p0, times=scenarios.times, quantiles=quantiles,
                  dates=scenarios.dates, asof=market.asof)
    netting_set = netting_set_profile(cube, today, **common)
    per_trade = [exposure_profile(cube[:, :, i], today[i], maturity=trade.maturity_date, **common)
                 for i, trade in enumerate(trades)]
    return netting_set, per_trade
