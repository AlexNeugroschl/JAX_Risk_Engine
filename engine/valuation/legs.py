"""
A vanilla swap's coupons as arrays, valued as ORE values them on any date: today on today's
curves, and on every simulation date and path on that path's scenario market.

    ORE: `DiscountingSwapEngine` with at-par Ibor coupons (`IborCouponPricer`), paid cashflows
    dropped by `CashFlow::hasOccurred` (includeReferenceDateEvents = false: a flow paid on the
    valuation date has occurred), and during a simulation `FixingManager::applyFixings`
    (OREAnalytics/orea/simulation/fixingmanager.cpp).

On valuation date d, a floating coupon with fixing date f is:

  * historical if f is before the as-of date: its fixing from the trade's history (ORE
    refuses a missing one; so does `ore_builders.known_fixing`);
  * fixed on the path if as-of <= f < d: when the simulation stepped from grid date d_{g-1}
    to d_g with d_{g-1} <= f < d_g (d_0 = as-of), `FixingManager` stored the index's own
    forecast for the fixing date `adjust(d_g)`, read off that path's index curve on d_g, for
    every fixing date in the step. The coupon pays that rate from then on (`path_fixings`);
  * projected if f is on or after d: the at-par forecast over its par period
    (`ore_builders.par_coupon_forecast_period`) off the index curve of date d.

`legs_npv` values one date; `legs_cube` is it vmapped over the simulation dates, every path
at once. Today's price is `legs_npv` on today's `ZeroCurve`s, which is ORE's t=0 value
(tests/test_valuation.py checks both against ORE).
"""
import dataclasses
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import ORE

from engine.models.curves import DiscountCurve, log_discount
from engine.models.ore_builders import (
    SWAP_CALENDAR, TIME_AXIS_DAY_COUNTER, ibor_index, known_fixing, par_coupon_forecast_period,
)
from engine.models.static_key import StaticKeyMixin
from engine.simulation.scenario_market import ScenarioCurves


@dataclass(frozen=True, eq=False)
class Legs(StaticKeyMixin):
    """Every coupon of one vanilla swap (paid or not): serial dates for the date rules, times
    from the as-of date for the curves. `payer` pays the fixed leg."""
    payer: bool
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


def coupon_rates(legs: Legs, index, t, projected, known_rates) -> jax.Array:
    """Each floating coupon's rate on a valuation date: the at-par forecast off `index` where
    `projected`, else `known_rates` (a historical or path fixing)."""
    forecast = (discount_from(index, t, legs.forecast_start)
                / discount_from(index, t, legs.forecast_end) - 1.0) / legs.spanning
    return jnp.where(projected, forecast, known_rates)


def legs_npv(legs: Legs, disc, index, t, alive_fixed, alive_float, projected, known_rates) -> jax.Array:
    """NPV on one valuation date at model time `t` (curves measured from it): alive
    cashflows only, floating coupons at `coupon_rates` (any batch axes of the curves)."""
    fixed_pv = jnp.sum(jnp.where(alive_fixed, legs.fixed_amount * discount_from(disc, t, legs.fixed_pay), 0.0),
                       axis=-1)
    rate = coupon_rates(legs, index, t, projected, known_rates)
    float_cf = (legs.float_nominal * (rate + legs.float_spread) * legs.float_accrual
                * discount_from(disc, t, legs.float_pay))
    float_pv = jnp.sum(jnp.where(alive_float, float_cf, 0.0), axis=-1)
    npv = float_pv - fixed_pv
    return npv if legs.payer else -npv


@dataclass(frozen=True)
class TodaySchedule:
    """The date rules of one `Legs` on the as-of date (`ore_builders.known_fixing`)."""
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


@dataclass(frozen=True)
class PathSchedule:
    """The date rules of one `Legs` on the simulation dates (host, static)."""
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


def on_every_date(value_fn, legs: Legs, schedule: PathSchedule, times: np.ndarray, disc: ScenarioCurves,
                  index: ScenarioCurves, fixings: jax.Array, *per_date) -> jax.Array:
    """`[S, D]`: `value_fn(disc_j, index_j, t_j, alive_fixed_j, alive_float_j, projected_j,
    known_rates_j, *per_date_j)` on every simulation date, all paths at once. The known rates
    are the trade's history before the as-of date and FixingManager's path fixings after it
    (`fixings` `[S, D]`, the index's `path_fixings`). Extra `per_date` arrays are indexed by
    date on their first axis."""
    history = jnp.asarray(np.nan_to_num(legs.history), dtype=fixings.dtype)
    known = jnp.where(jnp.asarray(schedule.path_fixed)[None], fixings[:, schedule.fixing_step][:, None, :],
                      history)                                                    # [S, D, Nc]

    def one_date(t, disc_times, disc_logs, idx_times, idx_logs, alive_fixed, alive_float, projected, known_j,
                 *extra):
        return value_fn(DiscountCurve(disc_times, disc_logs), DiscountCurve(idx_times, idx_logs), t,
                        alive_fixed, alive_float, projected, known_j, *extra)

    in_axes = (0, 0, 1, 0, 1, 0, 0, 0, 1) + (0,) * len(per_date)
    values = jax.vmap(one_date, in_axes=in_axes)(
        jnp.asarray(times, dtype=fixings.dtype), disc.tenor_times, disc.log_discounts,
        index.tenor_times, index.log_discounts, schedule.alive_fixed, schedule.alive_float,
        schedule.projected, known, *(jnp.asarray(a) for a in per_date))
    return values.T


def legs_cube(legs: Legs, schedule: PathSchedule, times: np.ndarray, disc: ScenarioCurves,
              index: ScenarioCurves, fixings: jax.Array) -> jax.Array:
    """`[S, D]` NPVs on every path and simulation date, in the curves' dtype."""
    legs = legs.astype(disc.log_discounts.dtype)
    return on_every_date(lambda *args: legs_npv(legs, *args), legs, schedule, times, disc, index, fixings)


def path_fixings(index_tenor_months: int, asof: ORE.Date, dates: Sequence[ORE.Date], times: np.ndarray,
                 index: ScenarioCurves) -> jax.Array:
    """`[S, D]`: on grid date d_g, the index's forecast for the fixing date
    `fixingCalendar.adjust(d_g)` off that path's index curve (`FixingManager::applyFixings`:
    `index->fixing(currentFixingDate)`, i.e. `IborIndex::forecastFixing` over the index's own
    value and maturity dates, not a coupon's par period)."""
    ibor = ibor_index(index_tenor_months)
    starts, ends, fractions = [], [], []
    for d in dates:
        fixing_date = SWAP_CALENDAR.adjust(d, ORE.Following)
        d1 = ibor.valueDate(fixing_date)
        d2 = ibor.maturityDate(d1)
        starts.append(TIME_AXIS_DAY_COUNTER.yearFraction(asof, d1))
        ends.append(TIME_AXIS_DAY_COUNTER.yearFraction(asof, d2))
        fractions.append(ibor.dayCounter().yearFraction(d1, d2))
    dtype = index.log_discounts.dtype
    rel = lambda a: jnp.asarray(np.asarray(a) - times, dtype=dtype)  # noqa: E731

    def forecast(tenor_times, logs, t1, t2, fraction):
        curve = DiscountCurve(tenor_times, logs)
        return (jnp.exp(log_discount(curve, t1) - log_discount(curve, t2)) - 1.0) / fraction

    per_date = jax.vmap(forecast, in_axes=(0, 1, 0, 0, 0))(
        index.tenor_times, index.log_discounts, rel(starts), rel(ends), jnp.asarray(fractions, dtype=dtype))
    return per_date.T


def index_fixings_by_name(requests: Dict[str, int], asof: ORE.Date, dates, times,
                          index_curves: Mapping[str, ScenarioCurves]) -> Dict[str, jax.Array]:
    """`path_fixings` for every index name -> tenor in `requests`."""
    return {name: path_fixings(tenor, asof, dates, times, index_curves[name]) for name, tenor in requests.items()}
