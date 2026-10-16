"""
A vanilla swap's coupons as arrays, valued as ORE values them on any date: today on today's
curves, and on every simulation date and path on that path's scenario market.

    ORE: `DiscountingSwapEngine` with at-par Ibor coupons (`IborCouponPricer`), paid cashflows
    dropped by `CashFlow::hasOccurred` (includeReferenceDateEvents = false: a flow paid on the
    valuation date has occurred), and during a simulation `FixingManager::applyFixings`
    (OREAnalytics/orea/simulation/fixingmanager.cpp).

On valuation date d, a floating coupon with fixing date f is:

  * historical if f is before the as-of date: its fixing from the trade's history (ORE
    refuses a missing one; so does `schedules.known_fixing`);
  * fixed on the path if as-of <= f < d: when the simulation stepped from grid date d_{g-1}
    to d_g with d_{g-1} <= f < d_g (d_0 = as-of), `FixingManager` stored the index's own
    forecast for the fixing date `adjust(d_g)`, read off that path's index curve on d_g, for
    every fixing date in the step. The coupon pays that rate from then on (`path_fixings`);
  * projected if f is on or after d: the at-par forecast over its par period
    (`schedules.par_coupon_forecast_period`) off the index curve of date d.

`leg_values` values one date, in difference form (F-07; docs/planning/details/precision.md
§8.2): the level, every flow valued off a curve no path depends on (today's, or a scenario
date's `ScenarioCurves.level`), with ORE's arithmetic; and, given each path's deviation from it
(`Deviations`), the path's deviation from the level, the flows' amounts times each path's
discount factors as a matrix product in the stage's compute format, accumulated in its
accumulate format (`engine.precision.Arithmetic`), the interpolation before it a stencil in the
accumulate format:

    dL                        each pay time's log discount factor's deviation, the curve's
                              node deviations carried by its own two-node interpolation
                              (`log_discount_deviation`: ln P is linear in the node values), and
                              each coupon's forecast ratio's (its start's less its end's)
    dV = [e, h] @ C           e = expm1(dL) at every fixed payment, h = (F0 + m) e + dF (1 + e)
                              for every coupon, dF its rate's deviation; C the level's present
                              value of each fixed flow (for e) and of a unit rate on each coupon
                              (for h)

since (F + m) P - (F0 + m) P0 = P0 [(F0 + m) e + dF (1 + e)] = P0 h for a coupon at spread m,
and a P - a P0 = a P0 e for a fixed flow: every term is a sensitivity times a deviation, so the
near cancellation of the two legs happens in the level, at the accumulate format, and a product
in a format below float32 rounds deviations, not legs. The outputs are a payer swap's value,
the annuity and the floating basis points (`LegValues`), as many as an engine reads (a swap its
value, a European its value and annuity); `legs_npv` is a swap's value on one date at the level
alone, and `legs_cube` every path and simulation date at once. Today's price is `legs_npv` on
today's `ZeroCurve`s, which is ORE's t=0 value (tests/test_pricing.py checks both against ORE); a
market-risk scenario's P&L is the deviation of the same values under its shift of the pillar rates.

`Legs` and `PathSchedule` are pytrees and the pricers are jitted with them as arguments, so a
program is compiled once per shape (coupon counts, paths, dates), not per trade or date.
"""
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Mapping, NamedTuple, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import ORE

from engine.instruments.schedules import SWAP_CALENDAR, ibor_index, known_fixing, par_coupon_forecast_period
from engine.market_data.curves import DiscountCurve, log_discount, log_discount_deviation
from engine.market_data.day_counts import TIME_AXIS_DAY_COUNTER
from engine.market_simulation.scenario_market import ScenarioCurves
from engine.precision import Arithmetic, add_level, deviation_from


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class Legs:
    """Every coupon of one vanilla swap (paid or not): serial dates for the date rules, times
    from the as-of date for the curves. `payer` pays the fixed leg (static under `jax.jit`;
    every other field is traced)."""
    payer: bool = field(metadata=dict(static=True))
    fixed_pay_serial: np.ndarray
    fixed_pay: np.ndarray
    fixed_amount: np.ndarray        # nominal * rate * accrual
    fixed_bps: np.ndarray           # nominal * accrual (the annuity weight)
    fixed_rate: float
    float_pay_serial: np.ndarray
    float_pay: np.ndarray
    float_nominal: np.ndarray
    float_accrual: np.ndarray
    float_spread: float
    forecast_start: np.ndarray
    forecast_end: np.ndarray
    spanning: np.ndarray
    fixing_serial: np.ndarray
    history: np.ndarray             # historical fixing where one was supplied, else NaN

    def astype(self, dtype) -> "Legs":
        """The same coupons with every real-valued field in `dtype` (serial dates stay
        integers), so a float32 curve is not promoted to float64 by the legs."""
        real = np.dtype(dtype).type
        cast = {name: np.asarray(value, dtype=dtype) for name, value in vars(self).items()
                if isinstance(value, np.ndarray) and value.dtype.kind == "f"}
        return dataclasses.replace(self, **cast, fixed_rate=real(self.fixed_rate),
                                   float_spread=real(self.float_spread))


def legs_of(swap: ORE.VanillaSwap, payer: bool, asof: ORE.Date, fixings: Mapping[ORE.Date, float],
            first_fixed: int = 0, first_float: int = 0) -> Legs:
    """`Legs` of an ORE swap from coupon `first_fixed`/`first_float` on (the underlying an
    exercise enters, `Swaption::buildUnderlyingSwaps`)."""
    t_of = lambda d: TIME_AXIS_DAY_COUNTER.yearFraction(asof, d)  # noqa: E731
    fixed = [ORE.as_fixed_rate_coupon(c) for c in swap.fixedLeg()][first_fixed:]
    floating = [ORE.as_floating_rate_coupon(c) for c in swap.floatingLeg()][first_float:]
    periods = [par_coupon_forecast_period(c) for c in floating]
    spreads = {c.spread() for c in floating} or {0.0}
    rates = {c.rate() for c in fixed} or {0.0}
    if len(spreads) != 1 or len(rates) != 1:
        raise ValueError("a leg with a varying fixed rate or spread is not supported")
    return Legs(
        payer=payer,
        fixed_pay_serial=np.array([c.date().serialNumber() for c in fixed], dtype=np.int64),
        fixed_pay=np.array([t_of(c.date()) for c in fixed]),
        fixed_amount=np.array([c.amount() for c in fixed]),
        fixed_bps=np.array([c.nominal() * c.accrualPeriod() for c in fixed]),
        fixed_rate=float(rates.pop()),
        float_pay_serial=np.array([c.date().serialNumber() for c in floating], dtype=np.int64),
        float_pay=np.array([t_of(c.date()) for c in floating]),
        float_nominal=np.array([c.nominal() for c in floating]),
        float_accrual=np.array([c.accrualPeriod() for c in floating]),
        float_spread=float(spreads.pop()),
        forecast_start=np.array([t_of(p[0]) for p in periods]),
        forecast_end=np.array([t_of(p[1]) for p in periods]),
        spanning=np.array([p[2] for p in periods]),
        fixing_serial=np.array([c.fixingDate().serialNumber() for c in floating], dtype=np.int64),
        history=np.array([fixings.get(c.fixingDate(), np.nan) for c in floating], dtype=np.float64),
    )


def discount_from(curve, t, times) -> jax.Array:
    """P(t, T) for maturities `times` (from the as-of date) off a curve measured from model
    time `t`."""
    return jnp.exp(log_discount(curve, jnp.asarray(times) - t))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Deviations:
    """Each path's deviation, on one valuation date, of what a leg reads from the levels its
    value is measured from: the discount and index curves' node values `[S, K]`
    (`log_discount_deviation`: a scenario curve's log discount factors less its level's, a
    market-risk scenario's shift of the pillar rates) and each floating coupon's known rate
    `[S, Nc]` (a path fixing less the level's, 0 for a historical one)."""
    discount: jax.Array
    index: jax.Array
    fixings: jax.Array


#: The values `leg_values` can carry, in `LegValues` order.
OUTPUTS = ("value", "annuity", "float_bps")


class LegValues(NamedTuple):
    """What the engines read of a leg on one date, all alive flows: `value`, the floating leg
    less the fixed leg (a payer swap's NPV); `annuity`, the fixed basis points (each fixed
    coupon's nominal times accrual, discounted); `float_bps`, the floating ones; `discounts`
    `[..., X]`, the discount factors at extra times (`leg_values`' `discount_at`). An output
    `leg_values` was not asked for is None."""
    value: Optional[jax.Array]
    annuity: Optional[jax.Array]
    float_bps: Optional[jax.Array]
    discounts: jax.Array


def discount_deviations(curve, t, times, deviation):
    """The curve's discount factors at `times` (from the as-of date) on the date at model time
    `t`: `(level, e)` with each path's factor `level * (1 + e)`, the level off `curve` and `e`
    `[S, Q]` the `expm1` of the path's node `deviation` `[S, K]` carried to the times
    (`log_discount_deviation`); `e` is None without a deviation."""
    level = discount_from(curve, t, times)
    if deviation is None:
        return level, None
    return level, jnp.expm1(log_discount_deviation(curve, deviation, jnp.asarray(times) - t))


def date_arithmetic(arithmetic: Arithmetic, t) -> Arithmetic:
    """Independent rounding on an Actual365Fixed valuation date, with time as traced data.

    Round back to a calendar day: truncation aliases days 2 and 3 because 3 / 365 * 365
    is slightly below 3. Shared by the leg and Jamshidian products.
    """
    if arithmetic.key is None:
        return arithmetic
    return arithmetic.fold_in(jnp.asarray(jnp.rint(t * 365.0), dtype=jnp.uint32))


def leg_values(legs: Legs, disc, index, t, alive_fixed, alive_float, projected, known_rates,
               deviations: Optional[Deviations] = None, discount_at=None, outputs: Tuple[str, ...] = ("value",),
               arithmetic: Optional[Arithmetic] = None) -> Tuple[LegValues, Optional[LegValues]]:
    """`(level, deviation)` of the leg's `LegValues` `outputs` (a static subset of `OUTPUTS`)
    on a valuation date at model time `t` (see the module docstring): the level off the curves
    `disc` and `index` (either type, measured from the date), alive flows only, each floating
    coupon at its at-par forecast where `projected`, else at `known_rates`; and given
    `deviations`, each path's deviation from the level `[S]` (None without them), its product in
    `arithmetic` (the deviations' own format, float64 or float32, by default), in the
    deviations' format; the level is in the curves' and the legs' (float64 on the paths), so
    `engine.precision.add_level` joins them with one rounding per path. `discount_at`: extra
    times whose discount factors the result carries (`LegValues.discounts`).

    The product's operand has a column per fixed flow, `e`, and per coupon, `h = (F0 + m) e + dF
    (1 + e)`, its rate times discount factor's deviation per unit of the level's discount factor
    (`(F + m) P / P0 - (F0 + m)`), so a coupon's two terms take one column; the floating basis
    points add a column per coupon, `e`, only when asked for. Each block of columns is its own
    product, the products summed in the accumulate format."""
    extra = jnp.zeros((0,), dtype=legs.fixed_pay.dtype) if discount_at is None else jnp.atleast_1d(discount_at)
    nf, nc = legs.fixed_pay.shape[0], legs.float_pay.shape[0]
    pay = jnp.concatenate([jnp.asarray(legs.fixed_pay), jnp.asarray(legs.float_pay), extra])
    discounts, e = discount_deviations(disc, t, pay, None if deviations is None else deviations.discount)
    fixed_df, float_df, extra_df = discounts[..., :nf], discounts[..., nf:nf + nc], discounts[..., nf + nc:]
    ratio = discount_from(index, t, legs.forecast_start) / discount_from(index, t, legs.forecast_end)
    rate = jnp.where(projected, (ratio - 1.0) / legs.spanning, known_rates)
    fixed_pv = jnp.where(alive_fixed, legs.fixed_amount * fixed_df, 0.0)
    float_pv = jnp.where(alive_float, legs.float_nominal * (rate + legs.float_spread) * legs.float_accrual * float_df,
                         0.0)
    fixed_bps = jnp.where(alive_fixed, legs.fixed_bps * fixed_df, 0.0)
    float_bps = jnp.where(alive_float, legs.float_nominal * legs.float_accrual * float_df, 0.0)
    levels = {"value": jnp.sum(float_pv, axis=-1) - jnp.sum(fixed_pv, axis=-1),
              "annuity": jnp.sum(fixed_bps, axis=-1), "float_bps": jnp.sum(float_bps, axis=-1)}
    level = LegValues(*(levels[name] if name in outputs else None for name in OUTPUTS), extra_df)
    if deviations is None:
        return level, None
    dtype = deviations.discount.dtype
    forecast = jnp.asarray(ratio / legs.spanning, dtype=dtype) * jnp.expm1(
        log_discount_deviation(index, deviations.index, jnp.asarray(legs.forecast_start) - t)
        - log_discount_deviation(index, deviations.index, jnp.asarray(legs.forecast_end) - t))
    rate_deviation = jnp.where(projected, forecast, deviations.fixings)                     # [S, Nc]
    e_fixed, e_float, e_extra = e[..., :nf], e[..., nf:nf + nc], e[..., nf + nc:]
    coupons = (jnp.asarray(rate + legs.float_spread, dtype=dtype) * e_float
               + rate_deviation * (1.0 + e_float))                                         # h, [S, Nc]
    # One product per block of columns, summed in the accumulator: a block's operand is then
    # written once in the layout its product reads (measured 2026-10-10: concatenating the blocks
    # into one operand cost the float64 leg kernel a quarter of its time on a CPU).
    blocks = [(e_fixed, {"value": -fixed_pv, "annuity": fixed_bps}),       # fixed flows: e
              (coupons, {"value": float_bps}),                              # coupons: h, at N tau P0
              (e_float, {"float_bps": float_bps})]                          # coupons: e
    asked = [name for name in OUTPUTS if name in outputs]
    arithmetic = arithmetic or Arithmetic.of(e.dtype)
    moved = 0.0
    arithmetic = date_arithmetic(arithmetic, t)
    for stream, (operand, coefficients) in enumerate(blocks):
        if any(name in coefficients for name in asked):
            none = jnp.zeros_like(next(iter(coefficients.values())))
            moved = moved + arithmetic.matmul(operand, jnp.stack([coefficients.get(name, none) for name in asked], -1),
                                              shared="right", stream=stream)
    by_name = {name: moved[..., i] for i, name in enumerate(asked)}
    extra = jnp.asarray(extra_df, dtype=dtype) * e_extra
    return level, LegValues(*(by_name.get(name) for name in OUTPUTS), extra)


@jax.jit
def legs_npv(legs: Legs, disc, index, t, alive_fixed, alive_float, projected, known_rates) -> jax.Array:
    """A swap's NPV on one valuation date at model time `t` (curves measured from it), at the
    level of `leg_values`: alive cashflows only, floating coupons at their forecast or
    `known_rates`."""
    level, _ = leg_values(legs, disc, index, t, alive_fixed, alive_float, projected, known_rates)
    return level.value if legs.payer else -level.value


@dataclass(frozen=True)
class TodaySchedule:
    """The date rules of one `Legs` on the as-of date (`schedules.known_fixing`)."""
    alive_fixed: np.ndarray
    alive_float: np.ndarray
    projected: np.ndarray
    known_rates: np.ndarray


def today_schedule(legs: Legs, asof: ORE.Date) -> TodaySchedule:
    """Flows paid on or before `asof` have occurred; a fixing before `asof` (or on it, when
    supplied) is known, the rest are projected. A missing past fixing is refused."""
    serial = asof.serialNumber()
    known = np.array([_known(int(f), h, asof) for f, h in zip(legs.fixing_serial, legs.history)],
                     dtype=np.float64)
    alive_float = legs.float_pay_serial > serial
    _require_history(legs, alive_float & np.isnan(known) & (legs.fixing_serial < serial), asof)
    return TodaySchedule(legs.fixed_pay_serial > serial, alive_float, np.isnan(known), np.nan_to_num(known))


def today_npv(legs: Legs, asof: ORE.Date, disc, index) -> jax.Array:
    """ORE's t=0 NPV (`today_schedule`)."""
    today = today_schedule(legs, asof)
    return legs_npv(legs, disc, index, 0.0, today.alive_fixed, today.alive_float, today.projected,
                    today.known_rates)


def _known(fixing_serial: int, history: float, asof: ORE.Date) -> float:
    """The fixing ORE takes as known on `asof`, NaN if it forecasts (missing history before
    `asof` is refused by `_require_history`)."""
    date = ORE.Date(fixing_serial)
    fixings = {} if np.isnan(history) else {date: history}
    if date < asof and not fixings:
        return np.nan
    value = known_fixing(date, asof, fixings)
    return np.nan if value is None else value


def _require_history(legs: Legs, missing: np.ndarray, asof: ORE.Date) -> None:
    if np.any(missing):
        dates = [ORE.Date(int(s)) for s in legs.fixing_serial[missing]]
        known_fixing(dates[0], asof, {})  # raises MissingFixingError naming the first


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PathSchedule:
    """The date rules of one `Legs` on the simulation dates, worked out on the host."""
    alive_fixed: np.ndarray   # [D, Nf]
    alive_float: np.ndarray   # [D, Nc]
    projected: np.ndarray     # [D, Nc]
    path_fixed: np.ndarray    # [D, Nc]
    fixing_step: np.ndarray   # [Nc] grid index whose fixing a path-fixed coupon pays


def path_schedule(legs: Legs, asof: ORE.Date, dates: Sequence[ORE.Date]) -> PathSchedule:
    """`FixingManager`/`hasOccurred` on the grid (see the module docstring)."""
    serials = np.array([d.serialNumber() for d in dates], dtype=np.int64)
    after_asof = legs.fixing_serial >= asof.serialNumber()
    alive_float = legs.float_pay_serial[None, :] > serials[:, None]
    history_needed = alive_float[0] & ~after_asof & np.isnan(legs.history) if dates else np.zeros(0, bool)
    _require_history(legs, history_needed, asof)
    return PathSchedule(
        alive_fixed=legs.fixed_pay_serial[None, :] > serials[:, None],
        alive_float=alive_float,
        projected=legs.fixing_serial[None, :] >= serials[:, None],
        path_fixed=after_asof[None, :] & (legs.fixing_serial[None, :] < serials[:, None]),
        fixing_step=np.minimum(np.searchsorted(serials, legs.fixing_serial, side="right"), len(dates) - 1),
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PathFixings:
    """FixingManager's fixings of one index on every grid date (`path_fixings`): `level` `[D]`,
    the forecast off the index curve's level on each date (float64 on the paths), and `deviation`
    `[S, D]`, each path's from it, in the curve's format. `rates` is the fixing on each path, in
    the deviation's format (`engine.precision.add_level`)."""
    level: jax.Array
    deviation: jax.Array

    @property
    def rates(self) -> jax.Array:
        return add_level(self.level[None, :], self.deviation)


def on_every_date(value_fn, legs: Legs, schedule: PathSchedule, times: np.ndarray, disc: ScenarioCurves,
                  index: ScenarioCurves, fixings: Optional[PathFixings], *per_date) -> jax.Array:
    """`[S, D]`: `value_fn(disc_j, index_j, t_j, alive_fixed_j, alive_float_j, projected_j,
    known_rates_j, deviations_j, *per_date_j)` on every simulation date, all paths at once:
    the curves' levels on date j (`DiscountCurve`s, float64 as `ScenarioCurves` keeps them), the
    coupons' known rates at the level (the trade's history before the as-of date, the level of
    FixingManager's path fixings after it, `fixings`), and each path's `Deviations` from them, in
    the curves' format. Extra `per_date` arrays are indexed by
    date on their first axis. `fixings` may be None for a leg without floating coupons."""
    level_dtype, dtype = disc.level.dtype, disc.log_discounts.dtype
    history = jnp.nan_to_num(jnp.asarray(legs.history)).astype(level_dtype)
    path_fixed = jnp.asarray(schedule.path_fixed)                                              # [D, Nc]
    if fixings is None:
        fixings = PathFixings(jnp.zeros(len(times), dtype=level_dtype),
                              jnp.zeros(disc.log_discounts.shape[:2], dtype=dtype))
    step = jnp.asarray(schedule.fixing_step)
    known = jnp.where(path_fixed, fixings.level[step][None, :], history)                     # [D, Nc]
    known_deviation = jnp.where(path_fixed[None], fixings.deviation[:, step][:, None, :], 0.0)  # [S, D, Nc]

    def one_date(t, disc_times, disc_level, disc_deviation, idx_times, idx_level, idx_deviation, alive_fixed,
                 alive_float, projected, known_j, known_deviation_j, *extra):
        return value_fn(DiscountCurve(disc_times, disc_level), DiscountCurve(idx_times, idx_level), t, alive_fixed,
                        alive_float, projected, known_j,
                        Deviations(disc_deviation, idx_deviation, known_deviation_j), *extra)

    in_axes = (0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1) + (0,) * len(per_date)
    values = jax.vmap(one_date, in_axes=in_axes)(
        jnp.asarray(times, dtype=level_dtype), disc.tenor_times, disc.level,
        deviation_from(disc.level[None], disc.log_discounts), index.tenor_times, index.level,
        deviation_from(index.level[None], index.log_discounts), schedule.alive_fixed, schedule.alive_float,
        schedule.projected, known, known_deviation, *(jnp.asarray(a) for a in per_date))
    return values.T


def legs_cube(legs: Legs, schedule: PathSchedule, times: np.ndarray, disc: ScenarioCurves,
              index: ScenarioCurves, fixings: Optional[PathFixings] = None,
              arithmetic: Optional[Arithmetic] = None) -> jax.Array:
    """`[S, D]` NPVs on every path and simulation date, in `arithmetic` (the curves' own
    format, float64 or float32, by default), whose accumulate format the curves' values are in;
    the legs and the levels stay float64."""
    arithmetic = arithmetic or Arithmetic.of(disc.log_discounts.dtype)
    return _legs_cube(legs, schedule, times, disc, index, fixings, arithmetic)


@jax.jit
def _legs_cube(legs: Legs, schedule: PathSchedule, times, disc: ScenarioCurves, index: ScenarioCurves,
               fixings: Optional[PathFixings], arithmetic: Arithmetic) -> jax.Array:
    def value(disc_j, index_j, t, alive_fixed, alive_float, projected, known, deviations):
        level, moved = leg_values(legs, disc_j, index_j, t, alive_fixed, alive_float, projected, known, deviations,
                                  arithmetic=arithmetic)
        npv = add_level(level.value, moved.value)
        return npv if legs.payer else -npv

    return on_every_date(value, legs, schedule, times, disc, index, fixings)


def path_fixings(index_tenor_months: int, asof: ORE.Date, dates: Sequence[ORE.Date], times: np.ndarray,
                 index: ScenarioCurves) -> PathFixings:
    """On grid date d_g, the index's forecast for the fixing date `fixingCalendar.adjust(d_g)`
    off that path's index curve (`FixingManager::applyFixings`: `index->fixing(currentFixingDate)`,
    i.e. `IborIndex::forecastFixing` over the index's own value and maturity dates, not a
    coupon's par period), as a level and each path's deviation (`PathFixings`), in the
    curve's format."""
    ibor = ibor_index(index_tenor_months)
    starts, ends, fractions = [], [], []
    for d in dates:
        fixing_date = SWAP_CALENDAR.adjust(d, ORE.Following)
        d1 = ibor.valueDate(fixing_date)
        d2 = ibor.maturityDate(d1)
        starts.append(TIME_AXIS_DAY_COUNTER.yearFraction(asof, d1))
        ends.append(TIME_AXIS_DAY_COUNTER.yearFraction(asof, d2))
        fractions.append(ibor.dayCounter().yearFraction(d1, d2))
    dtype = index.level.dtype
    rel = lambda a: np.asarray(np.asarray(a) - times, dtype=dtype)  # noqa: E731
    return _path_fixings(index, rel(starts), rel(ends), np.asarray(fractions, dtype=dtype))


@jax.jit
def _path_fixings(index: ScenarioCurves, starts, ends, fractions) -> PathFixings:
    """The forecasts over [starts_j, ends_j] (times from date j) off each date's curve, in
    difference form: the level's, and each path's deviation from it."""
    def forecast(tenor_times, level, logs, t1, t2, fraction):
        curve = DiscountCurve(tenor_times, level)
        ratio = jnp.exp(log_discount(curve, t1) - log_discount(curve, t2))
        deviation = deviation_from(level, logs)
        moved = log_discount_deviation(curve, deviation, t1) - log_discount_deviation(curve, deviation, t2)
        return (ratio - 1.0) / fraction, jnp.asarray(ratio / fraction, dtype=deviation.dtype) * jnp.expm1(moved)

    level, deviation = jax.vmap(forecast, in_axes=(0, 0, 1, 0, 0, 0))(index.tenor_times, index.level,
                                                                       index.log_discounts, starts, ends, fractions)
    return PathFixings(level, deviation.T)


def index_fixings_by_name(requests: Dict[str, int], asof: ORE.Date, dates, times,
                          index_curves: Mapping[str, ScenarioCurves]) -> Dict[str, PathFixings]:
    """`path_fixings` for every index name -> tenor in `requests`."""
    return {name: path_fixings(tenor, asof, dates, times, index_curves[name]) for name, tenor in requests.items()}
