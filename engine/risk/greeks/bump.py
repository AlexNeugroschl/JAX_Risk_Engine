"""
ORE's sensitivity analysis on today's market: bump and revalue (plan T-18, T-19; gates V-2,
V-3), OREAnalytics/orea/engine/sensitivityanalysis.cpp and cube/sensitivitycube.cpp.

The sensitivity market is ORE's `ScenarioSimMarket` on the as-of date: every discount and
index curve sampled at the configured tenors (`asof + tenor`, discount factors log-linear
between them, flat forward beyond), the swaption volatilities as given. Each trade is priced
there with its t=0 engine, as a function of the market it reads
(`engine.risk.greeks.price_functions.trade_price_function` on the tenors' zero rates and the
quotes), so Bermudans recalibrate under every bump.

**One batched revaluation.** A trade's shifts are the scenarios of one revaluation of its price
function (`engine.risk.greeks.scenario_pnl.price_function_pnl`, the one market risk runs): per
curve an up and a down shift of each tenor, then one shift of each quote its engine reads,
each scenario's NPV change in float64 in difference form (`TradePriceFunction.pnl`), an option
recalibrated under every shift first as ORE's sensitivity analysis rebuilds its model, all in
one program per product and shape rather than one revaluation per shift on the host.

  * Delta / Gamma per curve and tenor: an absolute zero-rate shift of `curve_shift` at that
    tenor (`ShiftType::Absolute`, shift tenors = the curve tenors: the shifted point alone moves,
    by `exp(-shift t)` in its discount factor). ORE's default `ShiftScheme::Forward`:
        delta = NPV(up) - NPV(base),   gamma = NPV(up) - 2 NPV(base) + NPV(down),
    each difference a scenario's P&L: delta = pnl(up), gamma = pnl(up) + pnl(down).
  * Vega per swaption volatility quote: an absolute shift of `vol_shift`, forward difference.
  * Theta (`sensitivityanalysis.cpp`, theta block): the evaluation date moves to
    `thetaDate = asof + theta_days` (calendar days, I-38) and a new simulation market is built
    from today's market. Its curves are the base market's as a function of the time from their
    reference date: ORE's simulation-market curve holds its tenor points at the tenors' times
    from the as-of date, with the as-of curve's discount factors there, and takes its reference
    date from the evaluation date (`InterpolatedDiscountCurve2`, `setAdjustReferenceDate(false)`;
    the same rule as on a simulated path, I-84). The ATM volatility matrix, a simulated factor
    (Vega bumps it), floats with it: its option dates roll from its reference date, the first
    business day on or after `thetaDate` (`SwaptionVolatilityMatrix` on tenors, a calendar and no
    settlement days: `engine.pricing.context.floating_volatility_date`), where an option's
    variance starts, a Saturday's from the Monday. `FixingManager` backfills the fixings from the as-of
    date to `thetaDate` with the index's forecast on that market, trades are rebuilt there, and
        theta = NPV(thetaDate) - NPV(base).
    ORE adds `aggregateTradeFlow`'s flows paid in (asof, thetaDate], but reads them from the
    trade's cash flows on the Theta date without its past flows (`Trade::cashflows(...,
    includePastCashflows = false)`), where every flow of the roll has occurred
    (`CashFlow::hasOccurred(thetaDate)`, a flow on the Theta date too, `includeTodaysCashFlows`
    unset), so under ORE's settings the sum is empty: a coupon paid in the roll leaves Theta as
    it leaves the NPV. (The engine added those flows back until 2026-10-18, I-39, read from the
    source; ORE's analytic showed none, I-116.) Equal to ORE's sensitivity analytic: every trade
    of the shared portfolio (tests/test_sensitivity_ore_parity.py, I-51) and of the seeded sweeps
    (tests/test_ore_parity_sweep.py), rolls across weekends, coupons, fixings and exercises.

AD Greeks (`engine.risk.greeks.ad`, `GreeksConfig.method = "AD"`) are the other method (decision
A-5): the same keys, Delta/Gamma/Vega by differentiation, and this module's Theta.
"""
from dataclasses import dataclass
from typing import Callable, Dict, Sequence

import jax.numpy as jnp
import numpy as np
import ORE

from engine.instruments.schedules import SWAP_CALENDAR, ibor_index
from engine.market_data.curves import DiscountCurve, log_discount
from engine.market_data.day_counts import TIME_AXIS_DAY_COUNTER
from engine.market_data.market import VOL_DAY_COUNTER, Market
from engine.market_simulation.config import DEFAULT_CURVE_TENORS
from engine.market_simulation.sharding import host_array
from engine.precision import Arithmetic
from engine.pricing.config import PricingConfig
from engine.pricing.context import PricingContext, floating_volatility_date, simulation_market_today
from engine.pricing.cube import Trade, validate_trades, value_on
from engine.risk.greeks.price_functions import Moves, TradePriceFunction, trade_price_function
from engine.risk.greeks.scenario_pnl import price_function_pnl


@dataclass(frozen=True)
class SensitivityConfig:
    """ORE's `sensitivity.xml` and the sensitivity simulation market: curve tenors (also the
    shift tenors), absolute shift sizes and the Theta horizon. The swaption volatilities are a
    simulated factor of that market, as Vega needs them to be, so ORE's `ReactionToTimeDecay`
    (read for a volatility that is not simulated) has no setting here."""
    curve_tenors: Sequence[str] = DEFAULT_CURVE_TENORS
    curve_shift: float = 1e-4
    vol_shift: float = 1e-4
    theta_days: int = 1


def sensitivity_context(market: Market, config: SensitivityConfig) -> PricingContext:
    """The base sensitivity market (see the module docstring): ORE's simulation market on the
    as-of date at the sensitivity tenors."""
    return simulation_market_today(market, config.curve_tenors)


def theta_context(market: Market, config: SensitivityConfig) -> PricingContext:
    """The Theta market (see the module docstring): the sensitivity market's curves on the Theta
    date, the volatility matrix read from it, fixings backfilled."""
    base = sensitivity_context(market, config)
    asof, theta_date = market.asof, market.asof + config.theta_days
    fixings = {name: _backfilled_fixings(name, curve, asof, theta_date) for name, curve in base.index.items()}

    reference = floating_volatility_date(theta_date)
    offset = VOL_DAY_COUNTER.yearFraction(theta_date, reference)

    def volatility(currency, option_time, swap_length):
        return float(market.swaption_vols(currency).volatility(reference, max(option_time - offset, 0.0), swap_length))

    return PricingContext(theta_date, base.discount, base.index, volatility, fixings, offset)


def _backfilled_fixings(name: str, curve: DiscountCurve, asof: ORE.Date, theta_date: ORE.Date) -> Dict[ORE.Date, float]:
    """`FixingManager::update(thetaDate)` from the as-of date: every fixing date in
    [asof, thetaDate) gets the index's forecast for `adjust(thetaDate)` off the Theta market."""
    tenor_months = int(name.rsplit("-", 1)[1][:-1])
    ibor = ibor_index(tenor_months)
    fixing_date = SWAP_CALENDAR.adjust(theta_date, ORE.Following)
    d1 = ibor.valueDate(fixing_date)
    d2 = ibor.maturityDate(d1)
    t1, t2 = (TIME_AXIS_DAY_COUNTER.yearFraction(theta_date, d) for d in (d1, d2))
    forecast = (float(jnp.exp(log_discount(curve, t1) - log_discount(curve, t2))) - 1.0) / \
        ibor.dayCounter().yearFraction(d1, d2)
    return {asof + k: forecast for k in range((theta_date - asof))
            if ibor.isValidFixingDate(asof + k)}


def portfolio_sensitivities(trades: Sequence[Trade], market: Market, base_currency: str,
                            pricing: PricingConfig = PricingConfig(),
                            config: SensitivityConfig = SensitivityConfig()) -> Dict[int, Dict[str, np.ndarray]]:
    """Per trade (by request index), in the base currency:
    `delta:discount:<ccy>` / `gamma:discount:<ccy>` and `delta:index:<name>` /
    `gamma:index:<name>` `[K]` per curve tenor (the trade's own curves), `vega:<ccy>`
    `[option tenors, swap tenors]` for a trade whose engine reads the swaption volatilities
    (`reads_swaption_vols`), and `theta`."""
    return greeks_by_trade(trade_sensitivities, trades, market, base_currency, pricing, config)


#: `(cfg, market, pricing, config) -> {key: Greek}`: one trade's Delta, Gamma and Vega in its
#: currency, by one method (`trade_sensitivities`, `engine.risk.greeks.ad.trade_greeks`).
TradeGreeks = Callable[..., Dict[str, np.ndarray]]


def greeks_by_trade(trade_greeks: TradeGreeks, trades: Sequence[Trade], market: Market, base_currency: str,
                    pricing: PricingConfig, config: SensitivityConfig) -> Dict[int, Dict[str, np.ndarray]]:
    """Per trade (by request index), `trade_greeks`' Greeks and ORE's Theta (`trade_theta`) in
    the base currency, each trade in its own profiler phase: the portfolio loop of both
    methods."""
    # Not at module scope: importing engine.run runs its __init__, which imports engine.risk.
    from engine.run.trace import trade_greeks_phase

    validate_trades(trades, market, pricing)
    base_context, theta_ctx = sensitivity_context(market, config), theta_context(market, config)
    result: Dict[int, Dict[str, np.ndarray]] = {}
    for i, cfg in enumerate(trades):
        with trade_greeks_phase(i, cfg):
            fx = market.fx_spot(cfg.currency, base_currency)
            greeks = {key: value * fx for key, value in trade_greeks(cfg, market, pricing, config).items()}
            value = lambda context, _cfg=cfg: value_on(_cfg, context, pricing) * fx  # noqa: E731
            greeks["theta"] = trade_theta(value, value(base_context), theta_ctx)
            result[i] = greeks
    return result


def trade_sensitivities(cfg, market: Market, pricing: PricingConfig = PricingConfig(),
                        config: SensitivityConfig = SensitivityConfig()) -> Dict[str, np.ndarray]:
    """ORE's Delta and Gamma per tenor of each curve the trade reads and, where its engine reads
    the quotes, its Vega per quote, in the trade's currency: every shift a scenario of one
    revaluation of its price function on the sensitivity market (see the module docstring)."""
    fn = trade_price_function(cfg, market, pricing, tenors=config.curve_tenors)
    moves = bump_moves(fn, config)
    count = moves.shifts[0].shape[0]
    _, pnl = price_function_pnl(cfg, fn, fn.rates, moves, pricing, count, Arithmetic())
    pnl = host_array(pnl).astype(np.float64)
    greeks: Dict[str, np.ndarray] = {}
    row = 0
    for (kind, name), rates in zip(fn.curves, fn.rates):
        size = rates.shape[-1]
        up, down = pnl[row:row + size], pnl[row + size:row + 2 * size]
        greeks[f"delta:{kind}:{name}"] = up
        greeks[f"gamma:{kind}:{name}"] = up + down
        row += 2 * size
    if fn.quotes is not None:
        greeks[f"vega:{cfg.currency}"] = pnl[row:].reshape(fn.quotes.shape)
    return greeks


def bump_moves(fn: TradePriceFunction, config: SensitivityConfig) -> Moves:
    """The scenarios of a trade's shifts, in the node values of its price function: per curve
    (`fn.curves`, in order) an up shift of `curve_shift` at each tenor, then a down shift at each,
    then a shift of `vol_shift` of each quote (`fn.quotes`, row-major) where the engine reads
    them. Each scenario moves one node or one quote."""
    sizes = [rates.shape[-1] for rates in fn.rates]
    quotes = 0 if fn.quotes is None else fn.quotes.size
    count = 2 * sum(sizes) + quotes
    shifts, row = [], 0
    for size in sizes:
        shift = np.zeros((count, size))
        shift[row:row + size] = config.curve_shift * np.eye(size)
        shift[row + size:row + 2 * size] = -config.curve_shift * np.eye(size)
        shifts.append(shift)
        row += 2 * size
    vega = None
    if fn.quotes is not None:
        vega = np.zeros((count, quotes))
        vega[row:] = config.vol_shift * np.eye(quotes)
        vega = vega.reshape((count,) + fn.quotes.shape)
    return Moves(tuple(jnp.asarray(s) for s in shifts), quotes=None if vega is None else jnp.asarray(vega))


def trade_theta(value, base: float, theta: PricingContext) -> np.ndarray:
    """ORE's Theta (see the module docstring) in the base currency, from the trade's
    valuation on a context `value(context)` and its value `base` on the sensitivity market;
    shared by both Greeks methods (`greeks_by_trade`)."""
    return np.asarray(value(theta) - base)
