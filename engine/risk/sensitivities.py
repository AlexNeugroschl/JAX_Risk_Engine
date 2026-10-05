"""
ORE's sensitivity analysis on the market path: bump and revalue (plan T-18, T-19; gates V-2,
V-3), OREAnalytics/orea/engine/sensitivityanalysis.cpp and cube/sensitivitycube.cpp.

The sensitivity market is ORE's `ScenarioSimMarket` on the as-of date: every discount and
index curve sampled at the configured tenors (`asof + tenor`, discount factors log-linear
between them, flat forward beyond), the swaption volatilities as given. Each trade is priced
there with its t=0 engine (`engine.valuation`), so Bermudans recalibrate under every bump.

  * Delta / Gamma per curve and tenor: an absolute zero-rate shift of `curve_shift` at that
    tenor (`ShiftType::Absolute`, shift tenors = the curve tenors: the shifted point alone moves,
    by `exp(-shift t)` in its discount factor). ORE's default `ShiftScheme::Forward`:
        delta = NPV(up) - NPV(base),   gamma = NPV(up) - 2 NPV(base) + NPV(down).
  * Vega per swaption volatility quote: an absolute shift of `vol_shift`, forward difference.
  * Theta (`sensitivityanalysis.cpp`, theta block): the date moves to
    `thetaDate = asof + theta_days` (calendar days, I-38) and a new simulation market is built
    there from the original market: each curve's tenor points take the original curve's
    discount factor at `thetaDate + tenor`, with 1 at `thetaDate` (fixed in dates, not
    renormalised); swaption volatilities are the as-of surface seen from `thetaDate`
    (`DynamicSwaptionVolatilityMatrix`). `FixingManager` backfills the fixings from the as-of
    date to `thetaDate` with the index's forecast on that market, trades are rebuilt there, and
        theta = NPV(thetaDate) - NPV(base) + cash flows paid in (asof, thetaDate].

AD Greeks (`engine.risk.greeks`, `GreeksConfig.method = "AD"`) are the other method (decision
A-5): the same keys, Delta/Gamma/Vega by differentiation, and this module's Theta.
"""
import dataclasses
from dataclasses import dataclass
from typing import Dict, Sequence

import jax.numpy as jnp
import numpy as np
import ORE

from engine.instruments.swap import SwapConfig, _build_ore_swap as _swap_underlying
from engine.instruments.treasury import BondConfig
from engine.market import Market, SwaptionVolSurface, index_name
from engine.models.curves import DiscountCurve, ZeroCurve, log_discount
from engine.models.ore_builders import SWAP_CALENDAR, TIME_AXIS_DAY_COUNTER, ibor_index
from engine.simulation.config import DEFAULT_CURVE_TENORS
from engine.valuation.config import PricingConfig
from engine.valuation.context import PricingContext
from engine.valuation.european import volatility_on_path
from engine.valuation.legs import Legs, legs_of
from engine.valuation.portfolio import Trade, bond_legs, reads_swaption_vols, validate_trades, value_on


@dataclass(frozen=True)
class SensitivityConfig:
    """ORE's `sensitivity.xml` and the sensitivity simulation market: curve tenors (also the
    shift tenors), absolute shift sizes, Theta horizon, and the volatility decay on the Theta
    date."""
    curve_tenors: Sequence[str] = DEFAULT_CURVE_TENORS
    curve_shift: float = 1e-4
    vol_shift: float = 1e-4
    theta_days: int = 1
    swaption_vol_decay: str = "ForwardVariance"


def _sampled(curve: ZeroCurve, asof: ORE.Date, reference: ORE.Date, tenors: Sequence[str]) -> DiscountCurve:
    """ORE's sim-market curve on `reference`: points at `reference + tenor` carrying the
    original (as-of) curve's discount factors there, and 1 at `reference`."""
    dates = [reference + ORE.Period(t) for t in tenors]
    times = [0.0] + [TIME_AXIS_DAY_COUNTER.yearFraction(reference, d) for d in dates]
    values = [0.0] + [float(log_discount(curve, TIME_AXIS_DAY_COUNTER.yearFraction(asof, d))) for d in dates]
    return DiscountCurve(times=jnp.asarray(times), log_discounts=jnp.asarray(values))


def _shifted(curve: DiscountCurve, k: int, shift: float) -> DiscountCurve:
    """The tenor point k's zero rate shifted by `shift` (its log discount by `-shift t_k`)."""
    bump = jnp.zeros_like(curve.log_discounts).at[k].set(-shift * curve.times[k])
    return DiscountCurve(curve.times, curve.log_discounts + bump)


def _static_volatility(market: Market, surfaces: Dict[str, SwaptionVolSurface]):
    def volatility(currency, option_time, swap_length):
        return float(surfaces[currency].volatility(market.asof, option_time, swap_length))
    return volatility


def sensitivity_context(market: Market, config: SensitivityConfig) -> PricingContext:
    """The base sensitivity market (see the module docstring)."""
    discount, index = {}, {}
    for code, data in market.currencies.items():
        discount[code] = _sampled(ZeroCurve.from_config(data.discount_curve), market.asof, market.asof,
                                  config.curve_tenors)
        index.update({name: _sampled(ZeroCurve.from_config(c), market.asof, market.asof, config.curve_tenors)
                      for name, c in data.index_curves.items()})
    surfaces = {c: d.swaption_vols for c, d in market.currencies.items() if d.swaption_vols is not None}
    return PricingContext(market.asof, discount, index, _static_volatility(market, surfaces))


def theta_context(market: Market, config: SensitivityConfig) -> PricingContext:
    """The Theta market (see the module docstring), fixings backfilled."""
    asof, theta_date = market.asof, market.asof + config.theta_days
    discount, index, fixings = {}, {}, {}
    for code, data in market.currencies.items():
        discount[code] = _sampled(ZeroCurve.from_config(data.discount_curve), asof, theta_date, config.curve_tenors)
        for name, curve in data.index_curves.items():
            index[name] = _sampled(ZeroCurve.from_config(curve), asof, theta_date, config.curve_tenors)
            fixings[name] = _backfilled_fixings(name, index[name], asof, theta_date)

    def volatility(currency, option_time, swap_length):
        return volatility_on_path(market.swaption_vols(currency), asof, theta_date, option_time, swap_length,
                                  config.swaption_vol_decay)

    return PricingContext(theta_date, discount, index, volatility, fixings)


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
    # Not at module scope: importing engine.portfolio runs its __init__, which imports engine.risk.
    from engine.portfolio.profiling import trade_greeks_phase

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
    shared by both Greeks methods (`engine.risk.greeks` for AD)."""
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
