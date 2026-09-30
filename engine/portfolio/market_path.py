"""
`price_portfolio` on today's `Market`: ORE's semantics end to end (the ORE alignment plan's
target design). The default path.

    Market + CamConfig --calibrate CAM--> CrossAssetModel --simulate--> ScenarioMarket
        --ValuationEngine (engine.valuation)--> t=0 NPVs and the NPV cube (base currency)
        --ExposureCalculator (engine.risk.exposure)--> EPE, ENE, EE_B, EEE_B, EPE_B, EEPE_B, PFE

Greeks are ORE's bump-and-revalue sensitivities on today's market
(`engine.risk.sensitivities`).

Every choice comes from the request's run configuration (`engine.portfolio.config`):
`simulation` (the CAM, its model per currency, the grid), `pricing` (the engine per product),
`greeks` (the method and ORE's sensitivity settings), `precision` and the reporting currency.
Options this path does not implement are refused by `check_market_path`.

The Hull-White path (`engine.portfolio.request`, a `SimulationConfig` as the market) is the
other model: supported, not the default, with the known limitations registered as I-42 to
I-47, to be fixed within it when roadmap 1.3 makes it a model of the same configuration.
"""
from typing import TYPE_CHECKING, List, Sequence

import jax.numpy as jnp
import numpy as np

from engine.market import Market
from engine.models.curves import ZeroCurve, discount
from engine.portfolio.config import _dtype_of, check_market_path
from engine.portfolio.profiling import phase
from engine.risk.exposure import ExposureProfile, exposure_profile, netting_set_profile
from engine.risk.var_es import ENGINE_RISK_MEASURE
from engine.simulation.config import build_cross_asset_model, simulate
from engine.valuation.portfolio import validate_trades, value_portfolio, value_today

if TYPE_CHECKING:
    from engine.portfolio.request import PortfolioResult


def price_on_market(request) -> "PortfolioResult":
    """`price_portfolio` for a `PortfolioRequest` whose `market` is a `Market`."""
    from engine.portfolio.request import PortfolioResult

    validate_market_request(request)
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
            scenarios = simulate(market, simulation, model, dtype=_dtype_of(run.precision.simulation))
        with phase("pricing"):
            valuation = value_portfolio(trades, market, scenarios, base, run.pricing, simulation.swaption_vol_decay)
        today, cube = valuation.today, valuation.cube
        with phase("exposure"):
            exposure, trade_exposures = _exposures(trades, today, cube, scenarios, market, base, request.pfe_quantiles)
    else:
        with phase("base_npv"):
            today = value_today(trades, market, base, run.pricing)
        cube = jnp.zeros((0, 0, 0))
    if request.compute_greeks:
        from engine.risk.sensitivities import portfolio_sensitivities
        with phase("greeks"):
            greeks = portfolio_sensitivities(trades, market, base, run.pricing, run.greeks.sensitivity)
    return PortfolioResult(
        base_npv=float(np.sum(today)), npv_cube=cube, exposure=exposure, trade_exposures=trade_exposures,
        greeks=greeks, warnings=[], base_npv_per_trade=list(today), scenario_risk_available=request.scenario_risk,
        measure=ENGINE_RISK_MEASURE if request.scenario_risk else None,
    )


def validate_market_request(request) -> None:
    """Refuse, before any JAX work, a request the market path cannot price: an option of
    the run configuration it does not implement (`check_market_path`), scenario risk
    without a simulation, or a trade the market cannot value (`validate_trades`). The HTTP
    route runs it synchronously so such a request is a 400, not a failed job."""
    run = request.config
    check_market_path(run, request.trades, request.compute_greeks, request.calibration_targets)
    if request.scenario_risk and run.simulation is None:
        raise ValueError("scenario_risk needs config.simulation (a CamConfig); set scenario_risk=False for "
                         "today's NPVs and Greeks only")
    validate_trades(request.trades, request.market)
    request.market.currency(run.reporting_currency)


def _exposures(trades: Sequence, today: List[float], cube, scenarios, market: Market, base: str,
               quantiles) -> "tuple[ExposureProfile, List[ExposureProfile]]":
    """Netting-set and per-trade profiles, deflated by the LGM numeraire, EE_B against the base
    currency's discount curve, time weights on the simulation dates."""
    curve = ZeroCurve.from_config(market.currency(base).discount_curve, dtype=cube.dtype)
    p0 = discount(curve, jnp.asarray(scenarios.times, dtype=cube.dtype))
    numeraire = jnp.asarray(scenarios.numeraire, dtype=cube.dtype)
    common = dict(numeraire=numeraire, discount=p0, times=scenarios.times, quantiles=quantiles,
                  dates=scenarios.dates, asof=market.asof)
    netting_set = netting_set_profile(cube, today, **common)
    per_trade = [exposure_profile(cube[:, :, i], today[i], maturity=trade.maturity_date, **common)
                 for i, trade in enumerate(trades)]
    return netting_set, per_trade
