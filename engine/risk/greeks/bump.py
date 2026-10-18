"""
ORE's sensitivity analysis on today's market: bump and revalue (plan T-18, T-19; gates V-2,
V-3), OREAnalytics/orea/engine/sensitivityanalysis.cpp and cube/sensitivitycube.cpp.

The sensitivity market is ORE's `ScenarioSimMarket` on the as-of date: every discount and
index curve sampled at the configured tenors (`asof + tenor`, discount factors log-linear
between them, flat forward beyond), the swaption volatilities as given. Each trade is priced
there with its t=0 engine (`engine.pricing`), so Bermudans recalibrate under every bump.

  * Delta / Gamma per curve and tenor: an absolute zero-rate shift of `curve_shift` at that
    tenor (`ShiftType::Absolute`, shift tenors = the curve tenors: the shifted point alone moves,
    by `exp(-shift t)` in its discount factor). ORE's default `ShiftScheme::Forward`:
        delta = NPV(up) - NPV(base),   gamma = NPV(up) - 2 NPV(base) + NPV(down).
  * Vega per swaption volatility quote: an absolute shift of `vol_shift`, forward difference.
  * Theta (`sensitivityanalysis.cpp`, theta block): the evaluation date moves to
    `thetaDate = asof + theta_days` (calendar days, I-38) and a new simulation market is built
    from today's market. Its curves are the base market's as a function of the time from their
    reference date: ORE's simulation-market curve holds its tenor points at the tenors' times
    from the as-of date, with the as-of curve's discount factors there, and takes its reference
    date from the evaluation date (`InterpolatedDiscountCurve2`, `setAdjustReferenceDate(false)`;
    the same rule as on a simulated path, I-84). The ATM volatility matrix, a simulated factor
    (Vega bumps it), floats with it: its option dates roll from `thetaDate`
    (`SwaptionVolatilityMatrix` on tenors). `FixingManager` backfills the fixings from the as-of
    date to `thetaDate` with the index's forecast on that market, trades are rebuilt there, and
        theta = NPV(thetaDate) - NPV(base) + cash flows paid in (asof, thetaDate].
    Equal to ORE's sensitivity analytic, every trade of the shared portfolio
    (tests/test_sensitivity_ore_parity.py, I-51).

AD Greeks (`engine.risk.greeks.ad`, `GreeksConfig.method = "AD"`) are the other method (decision
A-5): the same keys, Delta/Gamma/Vega by differentiation, and this module's Theta.
"""
import dataclasses
from dataclasses import dataclass
from typing import Dict, Sequence

import jax.numpy as jnp
import numpy as np
import ORE

from engine.instruments.schedules import SWAP_CALENDAR, ibor_index
from engine.instruments.swap import SwapConfig, underlying_swap as _swap_underlying
from engine.instruments.treasury import BondConfig
from engine.market_data.curves import DiscountCurve, log_discount
from engine.market_data.day_counts import TIME_AXIS_DAY_COUNTER
from engine.market_data.market import Market, index_name
from engine.market_simulation.config import DEFAULT_CURVE_TENORS
from engine.pricing.config import PricingConfig
from engine.pricing.context import PricingContext, simulation_market_today
from engine.pricing.cube import Trade, bond_legs, reads_swaption_vols, validate_trades, value_on
from engine.pricing.legs import Legs, legs_of


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


def _shifted(curve: DiscountCurve, k: int, shift: float) -> DiscountCurve:
    """The tenor point k's zero rate shifted by `shift` (its log discount by `-shift t_k`)."""
    bump = jnp.zeros_like(curve.log_discounts).at[k].set(-shift * curve.times[k])
    return DiscountCurve(curve.times, curve.log_discounts + bump)


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

    def volatility(currency, option_time, swap_length):
        return float(market.swaption_vols(currency).volatility(theta_date, option_time, swap_length))

    return PricingContext(theta_date, base.discount, base.index, volatility, fixings)


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
    # Not at module scope: importing engine.run runs its __init__, which imports engine.risk.
    from engine.run.trace import trade_greeks_phase

    validate_trades(trades, market, pricing)
    base_context = sensitivity_context(market, config)
    theta_ctx = theta_context(market, config)
    result: Dict[int, Dict[str, np.ndarray]] = {}
    for i, cfg in enumerate(trades):
        with trade_greeks_phase(i, cfg):
            currency = cfg.currency
            fx = market.fx_spot(currency, base_currency)
            value = lambda context: value_on(cfg, context, pricing) * fx  # noqa: E731
            base = value(base_context)
            greeks: Dict[str, np.ndarray] = {}
            curves = [("discount", currency, base_context.discount)]
            if not isinstance(cfg, BondConfig):
                curves.append(("index", index_name(currency, cfg.index_tenor_months), base_context.index))
            for kind, key, table in curves:
                deltas, gammas = [], []
                for k in range(1, len(table[key].times)):
                    up = value(_with(base_context, kind, key, _shifted(table[key], k, config.curve_shift)))
                    down = value(_with(base_context, kind, key, _shifted(table[key], k, -config.curve_shift)))
                    deltas.append(up - base)
                    gammas.append(up - 2.0 * base + down)
                greeks[f"delta:{kind}:{key}"] = np.asarray(deltas)
                greeks[f"gamma:{kind}:{key}"] = np.asarray(gammas)
            if reads_swaption_vols(cfg, pricing):
                greeks[f"vega:{currency}"] = _vega(value, base, market, currency, base_context, config.vol_shift)
            greeks["theta"] = trade_theta(value, base, cfg, theta_ctx, fx)
            result[i] = greeks
    return result


def trade_theta(value, base: float, cfg: Trade, theta: PricingContext, fx: float) -> np.ndarray:
    """ORE's Theta (see the module docstring) in the base currency, from the trade's
    valuation on a context `value(context)` and its value `base` on the sensitivity market;
    shared by both Greeks methods (`engine.risk.greeks.ad` for AD)."""
    return np.asarray(value(theta) - base + _period_flows(cfg, theta) * fx)


def _with(context: PricingContext, kind: str, key: str, curve: DiscountCurve) -> PricingContext:
    table = dict(context.discount if kind == "discount" else context.index)
    table[key] = curve
    return dataclasses.replace(context, **{kind: table})


def _vega(value, base: float, market: Market, currency: str, context: PricingContext, shift: float) -> np.ndarray:
    surface = market.swaption_vols(currency)
    vega = np.zeros((len(surface.option_tenors), len(surface.swap_tenors)))
    for i in range(vega.shape[0]):
        for j in range(vega.shape[1]):
            vols = np.array(surface.vols)
            vols[i, j] += shift
            bumped = dataclasses.replace(surface, vols=tuple(map(tuple, vols)))

            def volatility(ccy, option_time, swap_length, _bumped=bumped):
                chosen = _bumped if ccy == currency else market.swaption_vols(ccy)
                return float(chosen.volatility(market.asof, option_time, swap_length))

            vega[i, j] = value(dataclasses.replace(context, volatility=volatility)) - base
    return vega


def _period_flows(cfg: Trade, theta: PricingContext) -> float:
    """Cash flows paid in (asof, thetaDate] (`aggregateTradeFlow`), in the trade's currency:
    swap coupons at their fixings, bond flows. Options pay nothing."""
    if isinstance(cfg, SwapConfig):
        legs = legs_of(_swap_underlying(cfg), cfg.payer, cfg.evaluation_date, cfg.fixings)
        return _flows_between(legs, cfg.evaluation_date, theta.date)
    if isinstance(cfg, BondConfig):
        return _flows_between(bond_legs(cfg), cfg.evaluation_date, theta.date)
    return 0.0


def _flows_between(legs: Legs, start: ORE.Date, end: ORE.Date) -> float:
    """Signed amounts of the flows paid in (start, end]; a floating coupon paying then fixed
    before `start`, so its history rate is known."""
    window = lambda serials: (serials > start.serialNumber()) & (serials <= end.serialNumber())  # noqa: E731
    fixed = float(np.sum(legs.fixed_amount[window(legs.fixed_pay_serial)]))
    paying = window(legs.float_pay_serial)
    floating = float(np.sum(legs.float_nominal[paying] * (np.nan_to_num(legs.history[paying]) + legs.float_spread)
                            * legs.float_accrual[paying]))
    net = floating - fixed
    return net if legs.payer else -net
