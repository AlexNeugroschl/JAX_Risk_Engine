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
alternative is QuantLib's Jamshidian engine on a Hull-White model (`engine.pricing.jamshidian`).

In difference form (F-07): the engine reads the exercise-into swap through `leg_values` as a
payer swap's value V, its annuity A and floating basis points B (and, cash settled, the
discount factor at the start), each a level plus each path's deviation from matrix products in
the stage's compute format. The forward less the effective strike is `V / A`, so the moneyness
is a swap's value, whose legs cancel in the level; the Bachelier formula on it, and the cash
annuity, are evaluated in the stage's accumulate format, as a native low-precision kernel
evaluates them (`engine.precision.arithmetic`).
"""
import dataclasses
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import ORE

from engine.instruments.european_swaption import SwaptionConfig, underlying_swap
from engine.market_data.day_counts import TIME_AXIS_DAY_COUNTER
from engine.market_data.market import VOL_DAY_COUNTER, SwaptionVolSurface, index_name, swap_length_between
from engine.market_simulation.scenario_market import ScenarioCurves
from engine.models.bachelier import bachelier
from engine.precision import Arithmetic, add_level
from engine.pricing.context import PricingContext
from engine.pricing.legs import (
    Deviations, Legs, LegValues, PathFixings, PathSchedule, leg_values, legs_of, on_every_date, today_schedule,
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


def _exercise_into(cfg: SwaptionConfig):
    """The underlying swap, its fixed and floating coupons, and the index of each leg's first
    coupon the exercise enters: the first paying after the expiry and accruing from it on (the
    engine's filter for exercise into whole periods). An expiry inside a leg's last accrual
    period leaves that leg nothing to enter, which the engine cannot price: refused, naming the
    trade and the dates (I-103)."""
    swap = underlying_swap(cfg)
    fixed = [ORE.as_fixed_rate_coupon(c) for c in swap.fixedLeg()]
    floating = [ORE.as_floating_rate_coupon(c) for c in swap.floatingLeg()]

    def first(coupons, leg: str) -> int:
        for i, c in enumerate(coupons):
            if c.date() > cfg.exercise_date and c.accrualStartDate() >= cfg.exercise_date:
                return i
        last = coupons[-1]
        raise ValueError(f"trade {cfg.trade_id!r}: exercise date {cfg.exercise_date.ISO()} leaves no whole period "
                         f"of its {leg} leg to exercise into (the last accrues from "
                         f"{last.accrualStartDate().ISO()} to {last.accrualEndDate().ISO()}); a European enters the "
                         f"coupons accruing from its expiry on")

    return swap, fixed, floating, first(fixed, "fixed"), first(floating, "floating")


def check_exercise_into_whole_periods(cfg: SwaptionConfig) -> None:
    """Refuse, before any pricing, a European whose expiry leaves a leg no whole period to enter
    (`_exercise_into`)."""
    _exercise_into(cfg)


def european_terms(cfg: SwaptionConfig, asof: ORE.Date, fixings=None) -> EuropeanTerms:
    """Keep the coupons with pay date > expiry and accrual start >= expiry (the engine's filter
    for exercise into whole periods, `_exercise_into`). Times are measured from `asof`;
    `fixings` is history."""
    swap, fixed, floating, first_fixed, first_float = _exercise_into(cfg)
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


def european_legs(terms: EuropeanTerms, disc, index, t, projected=True, known_rates=0.0,
                  deviations: Optional[Deviations] = None, arithmetic: Optional[Arithmetic] = None):
    """The exercise-into swap's `LegValues` on a valuation date at model time `t` (curves
    measured from it), every coupon alive (the engine keeps those paying after the expiry), and
    the discount factor at the start when cash settled: `(level, deviation)` (`leg_values`), its
    value and annuity, and its floating basis points when cash settled (the forward the cash
    annuity compounds at reads them)."""
    legs = terms.legs
    alive_fixed, alive_float = jnp.ones(legs.fixed_pay.shape, dtype=bool), jnp.ones(legs.float_pay.shape, dtype=bool)
    cash = terms.par_yield is not None
    return leg_values(legs, disc, index, t, alive_fixed, alive_float, projected, known_rates, deviations,
                      terms.start_time if cash else None,
                      ("value", "annuity", "float_bps") if cash else ("value", "annuity"), arithmetic)


def bachelier_on_legs(terms: EuropeanTerms, values: LegValues, variance) -> jax.Array:
    """The engine's NPV from the exercise-into swap's values (`european_legs`, any batch axes)
    for a normal `variance` to expiry: the annuity times the Bachelier price of the forward less
    the effective strike, `values.value / values.annuity`; the cash annuity when
    `terms.par_yield`."""
    legs = terms.legs
    cast = lambda x: jnp.asarray(x, dtype=values.value.dtype)  # noqa: E731  (the terms in the values' format)
    moneyness = values.value / values.annuity
    annuity = values.annuity
    if terms.par_yield is not None:
        forward = moneyness + cast(legs.fixed_rate) - cast(legs.float_spread) * values.float_bps / values.annuity
        compounded = (1.0 + jnp.expand_dims(forward, -1)) ** (-cast(terms.par_yield))
        annuity = values.discounts[..., 0] * jnp.sum(cast(legs.fixed_bps) * compounded, axis=-1)
    return annuity * bachelier(moneyness, 0.0, jnp.sqrt(cast(variance)), call=legs.payer)


@jax.jit
def black_multileg_npv(terms: EuropeanTerms, disc, index, t, variance, projected=True, known_rates=0.0) -> jax.Array:
    """The engine's NPV on a valuation date at model time `t` (curves measured from it), for a
    normal `variance` to expiry; the cash annuity when `terms.par_yield`. Coupons pay their
    forecast, or `known_rates` where not `projected` (a coupon that fixed before the valuation
    date pays its fixing: ORE reads the coupon's amount)."""
    level, _ = european_legs(terms, disc, index, t, projected, known_rates)
    return bachelier_on_legs(terms, level, variance)


def european_value(cfg: SwaptionConfig, context: PricingContext) -> jax.Array:
    """ORE's NPV on the context's date (0 once expired: `isExpired`, exercise on or before
    the date), the volatility read where the engine reads it, its variance from the date the
    volatilities run from (`PricingContext.volatility_offset`, ORE's `blackVariance`)."""
    if not cfg.exercise_date > context.date:
        return jnp.zeros(())
    extra = context.fixings.get(index_name(cfg.currency, cfg.index_tenor_months), {})
    terms = european_terms(cfg, context.date, extra)
    disc, index = context.curves(cfg.currency, index_name(cfg.currency, cfg.index_tenor_months))
    t = VOL_DAY_COUNTER.yearFraction(context.date, cfg.exercise_date)
    variance = context.volatility(cfg.currency, t, terms.swap_length) ** 2 * max(t - context.volatility_offset, 0.0)
    known = today_schedule(terms.legs, context.date)
    return black_multileg_npv(terms, disc, index, 0.0, variance, known.projected, known.known_rates)


def european_cube(terms: EuropeanTerms, schedule: PathSchedule, times: np.ndarray, disc: ScenarioCurves,
                  index: ScenarioCurves, fixings: PathFixings, variances: np.ndarray,
                  arithmetic: Optional[Arithmetic] = None) -> jax.Array:
    """`[S, D]` option NPVs on every path and date (0 from expiry on), in `arithmetic` (the
    curves' own format, float64 or float32, by default); `variances` `[D]` from
    `variance_on_path`."""
    alive = np.asarray([v > 0.0 for v in variances])
    arithmetic = arithmetic or Arithmetic.of(disc.log_discounts.dtype)
    return _european_cube(terms, schedule, times, disc, index, fixings,
                          np.asarray(variances, dtype=arithmetic.dtype), alive, arithmetic)


@jax.jit
def _european_cube(terms: EuropeanTerms, schedule: PathSchedule, times, disc: ScenarioCurves, index: ScenarioCurves,
                   fixings: PathFixings, variances, alive, arithmetic: Arithmetic) -> jax.Array:
    def value(disc_j, idx_j, t_j, _fixed, _float, projected_j, known_j, deviations, variance_j, alive_j):
        level, moved = european_legs(terms, disc_j, idx_j, t_j, projected_j, known_j, deviations, arithmetic)
        npv = bachelier_on_legs(terms, on_path(level, moved), variance_j)
        return jnp.where(alive_j, npv, 0.0)

    return on_every_date(value, terms.legs, schedule, times, disc, index, fixings, variances, alive)


def on_path(level, deviation):
    """A level's values plus each path's deviation (`LegValues` or arrays alike; an output not
    asked for, None, stays None), in the deviation's format (`engine.precision.add_level`)."""
    return jax.tree_util.tree_map(add_level, level, deviation)


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
