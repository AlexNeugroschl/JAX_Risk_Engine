"""
European swaptions with ORE's default engine: `EuropeanSwaptionEngineBuilder` ->
`BlackMultiLegOptionEngine` (QuantExt/qle/pricingengines/blackmultilegoptionengine.cpp) on the
market's normal swaption volatilities, today and on every simulated path (plan T-12).

For a physically settled option the engine keeps the coupons that pay after expiry and start
accruing on or after it, and prices a Bachelier option on the swap rate:

    annuity  = |fixed BPS| = sum_i N tau_i P(T_i)
    forward  = sum_k N F_k delta_k P(T_k) / annuity            (float leg without spread)
    strike   = K - spread * (sum_k N delta_k P(T_k)) / annuity (`fairRateFromNpvBps`)
    variance = vol(expiry, swap length)^2 * t(expiry), 0 on the expiry date
    NPV      = annuity * Bachelier(call for a payer, put for a receiver)

A cash-settled option uses ORE's default settlement method for it, `ParYieldCurve`
(OREData/ored/portfolio/swaption.cpp, `defaultSettlementMethod`): the annuity multiplying the
Bachelier price is the fixed leg discounted at the forward swap rate itself, from the earliest
accrual start (taken as the settlement date),

    annuity_cash = P(start) * sum_i N tau_i (1 + forward)^(-yf(start, T_i)),

yf on the fixed leg's day counter, annual compounding; forward and strike are unchanged.

The swap length is the vol surface's `swapLength(earliest accrual start, latest accrual end)`,
rounded to whole months and floored at one month. On a path the volatility is the t=0 surface
seen from the simulation date (`DynamicSwaptionVolatilityMatrix`, `volatility_on_path`).

This is the European's default engine for either simulation model (I-46); the configurable
alternative is QuantLib's Jamshidian engine on a Hull-White model (`engine.valuation.jamshidian`).
"""
import dataclasses
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import ORE
from jax.scipy.stats import norm

from engine.instruments.european_swaption import SwaptionConfig, _build_ore_swap
from engine.market import VOL_DAY_COUNTER, SwaptionVolSurface, index_name, swap_length_between
from engine.models.ore_builders import TIME_AXIS_DAY_COUNTER
from engine.simulation.scenario_market import ScenarioCurves
from engine.valuation.context import PricingContext
from engine.valuation.legs import (
    Legs, PathSchedule, coupon_rates, discount_from, legs_of, on_every_date, today_schedule,
)

#: `DynamicSwaptionVolatilityMatrix::volatilityImpl`'s floor on the forward variance rate.
FORWARD_VARIANCE_FLOOR = 1e-6


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EuropeanTerms:
    """What a European engine reads from the trade: the exercise-into coupons (as `Legs`,
    times from the as-of date), the expiry (a serial date, as `Legs` keeps its dates), the
    exercise-into swap's start and nominal, and where the volatility is read. A pytree, so
    the engines are jitted with it as an argument."""
    legs: Legs
    expiry_serial: int
    expiry_time: float        # model time from the as-of date
    swap_length: float        # the vol surface's rounded swap length
    start_time: float         # the earliest accrual start (model time from the as-of date)
    nominal: float
    #: Cash settlement (`ParYieldCurve`): each fixed coupon's year fraction from the start on
    #: the fixed leg's day counter. None when physically settled.
    par_yield: Optional[np.ndarray] = None

    def astype(self, dtype) -> "EuropeanTerms":
        """The same terms with every real-valued array in `dtype` (see `Legs.astype`)."""
        par_yield = None if self.par_yield is None else np.asarray(self.par_yield, dtype=dtype)
        return dataclasses.replace(self, legs=self.legs.astype(dtype), par_yield=par_yield)


def european_terms(cfg: SwaptionConfig, asof: ORE.Date, fixings=None) -> EuropeanTerms:
    """Keep the coupons with pay date > expiry and accrual start >= expiry (the engine's filter
    for exercise into whole periods). Times are measured from `asof`; `fixings` is history."""
    swap = _build_ore_swap(cfg)
    fixed = [ORE.as_fixed_rate_coupon(c) for c in swap.fixedLeg()]
    floating = [ORE.as_floating_rate_coupon(c) for c in swap.floatingLeg()]
    in_exercise = lambda c: c.date() > cfg.exercise_date and c.accrualStartDate() >= cfg.exercise_date  # noqa: E731
    first_fixed = next(i for i, c in enumerate(fixed) if in_exercise(c))
    first_float = next(i for i, c in enumerate(floating) if in_exercise(c))
    starts = [c.accrualStartDate() for c in fixed[first_fixed:] + floating[first_float:]]
    ends = [c.accrualEndDate() for c in fixed[first_fixed:] + floating[first_float:]]
    start = min(starts)
    par_yield = None
    if cfg.settlement == "Cash":
        par_yield = np.array([c.dayCounter().yearFraction(start, c.date()) for c in fixed[first_fixed:]])
    return EuropeanTerms(
        legs=legs_of(swap, cfg.payer, asof, fixings or {}, first_fixed, first_float),
        expiry_serial=cfg.exercise_date.serialNumber(),
        expiry_time=TIME_AXIS_DAY_COUNTER.yearFraction(asof, cfg.exercise_date),
        swap_length=max(swap_length_between(start, max(ends)), 1.0 / 12.0),
        start_time=TIME_AXIS_DAY_COUNTER.yearFraction(asof, start),
        nominal=fixed[first_fixed].nominal(),
        par_yield=par_yield,
    )


@jax.jit
def black_multileg_npv(terms: EuropeanTerms, disc, index, t, variance, projected=True, known_rates=0.0) -> jax.Array:
    """The engine's NPV on a valuation date at model time `t` (curves measured from it, any
    batch axes), for a normal `variance` to expiry; the cash annuity when `terms.par_yield`. Coupons pay `coupon_rates` (a coupon that
    fixed before the valuation date pays its fixing: ORE reads the coupon's amount)."""
    legs = terms.legs
    annuity = jnp.sum(legs.fixed_bps * discount_from(disc, t, legs.fixed_pay), axis=-1)
    float_df = discount_from(disc, t, legs.float_pay)
    rate = coupon_rates(legs, index, t, projected, known_rates)
    float_bps = jnp.sum(legs.float_nominal * legs.float_accrual * float_df, axis=-1)
    forward = jnp.sum(legs.float_nominal * rate * legs.float_accrual * float_df, axis=-1) / annuity
    strike = legs.fixed_rate - legs.float_spread * float_bps / annuity
    if terms.par_yield is not None:
        compounded = (1.0 + jnp.expand_dims(forward, -1)) ** (-terms.par_yield)
        annuity = discount_from(disc, t, terms.start_time) * jnp.sum(legs.fixed_bps * compounded, axis=-1)
    return annuity * bachelier(forward, strike, jnp.sqrt(variance), call=legs.payer)


def european_value(cfg: SwaptionConfig, context: PricingContext) -> jax.Array:
    """ORE's NPV on the context's date (0 once expired: `isExpired`, exercise on or before
    the date), the volatility read where the engine reads it."""
    if not cfg.exercise_date > context.date:
        return jnp.zeros(())
    extra = context.fixings.get(index_name(cfg.currency, cfg.index_tenor_months), {})
    terms = european_terms(cfg, context.date, extra)
    disc, index = context.curves(cfg.currency, index_name(cfg.currency, cfg.index_tenor_months))
    t = VOL_DAY_COUNTER.yearFraction(context.date, cfg.exercise_date)
    variance = context.volatility(cfg.currency, t, terms.swap_length) ** 2 * t
    known = today_schedule(terms.legs, context.date)
    return black_multileg_npv(terms, disc, index, 0.0, variance, known.projected, known.known_rates)


def european_cube(terms: EuropeanTerms, schedule: PathSchedule, times: np.ndarray, disc: ScenarioCurves,
                  index: ScenarioCurves, fixings: jax.Array, variances: np.ndarray) -> jax.Array:
    """`[S, D]` option NPVs on every path and date (0 from expiry on), in the curves' dtype;
    `variances` `[D]` from `variance_on_path`."""
    alive = np.asarray([v > 0.0 for v in variances])
    dtype = disc.log_discounts.dtype
    return _european_cube(terms.astype(dtype), schedule, times, disc, index, fixings,
                          np.asarray(variances, dtype=dtype), alive)


@jax.jit
def _european_cube(terms: EuropeanTerms, schedule: PathSchedule, times, disc: ScenarioCurves, index: ScenarioCurves,
                   fixings: jax.Array, variances, alive) -> jax.Array:
    def value(disc_j, idx_j, t_j, _fixed, _float, projected_j, known_j, variance_j, alive_j):
        npv = black_multileg_npv(terms, disc_j, idx_j, t_j, variance_j, projected_j, known_j)
        return jnp.where(alive_j, npv, 0.0)

    return on_every_date(value, terms.legs, schedule, times, disc, index, fixings, variances, alive)


def bachelier(forward, strike, std_dev, call: bool):
    """QuantLib's `bachelierBlackFormula` without the discount: w (F - K) N(w d) + s n(d),
    intrinsic when `std_dev` is 0."""
    w = 1.0 if call else -1.0
    safe = jnp.where(std_dev > 0.0, std_dev, 1.0)
    d = (forward - strike) / safe
    value = w * (forward - strike) * norm.cdf(w * d) + safe * norm.pdf(d)
    return jnp.where(std_dev > 0.0, value, jnp.maximum(w * (forward - strike), 0.0))


def volatility_on_path(surface: SwaptionVolSurface, asof: ORE.Date, date: ORE.Date, option_time: float,
                       swap_len: float, decay: str) -> float:
    """The volatility `DynamicSwaptionVolatilityMatrix` (settlement days 0, reference date
    moving with the evaluation date) gives on `date` for an option `option_time` years
    ahead: `ConstantVariance` reads the t=0 surface at the same time to expiry;
    `ForwardVariance` takes the forward variance between the date and the expiry,
        sqrt(max((V(tf + tau) - V(tf)) / tau, 1e-6)),  V(t) = vol(t)^2 t, tf = t(asof, date)."""
    if decay == "ConstantVariance":
        return float(surface.volatility(asof, option_time, swap_len))
    tf = VOL_DAY_COUNTER.yearFraction(asof, date)
    realised = float(surface.black_variance(asof, tf + option_time, swap_len))
    if tf > 0.0 and not np.isclose(tf, 0.0, rtol=0.0, atol=1e-14):
        realised -= float(surface.black_variance(asof, tf, swap_len))
    return float(np.sqrt(max(realised / option_time, FORWARD_VARIANCE_FLOOR)))


def variance_on_path(terms: EuropeanTerms, surface: SwaptionVolSurface, asof: ORE.Date, date: ORE.Date,
                     decay: str) -> float:
    """The normal variance to expiry the engine reads on `date` (0 once expired)."""
    if not terms.expiry_serial > date.serialNumber():
        return 0.0
    tau = VOL_DAY_COUNTER.yearFraction(date, ORE.Date(terms.expiry_serial))
    return volatility_on_path(surface, asof, date, tau, terms.swap_length, decay) ** 2 * tau
