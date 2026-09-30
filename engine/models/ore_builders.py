"""
Builds ORE trade objects from trade configs and extracts their cashflow schedules.

Schedules, calendars and accrual fractions come from ORE (`MakeVanillaSwap`, coupon
`accrualPeriod()`), not reimplemented. Day counts are always set explicitly, never left to
`MakeVanillaSwap`'s per-index defaults.

Scope: generic term-Ibor swaps only. `build_vanilla_swap` produces a `SimIndex<N>M` index
on a TARGET calendar, with ACT/365 accrual by default. It cannot express an overnight
index (SOFR: ACT/360, compounded in arrears, US calendar, lookback/lockout). Such trades
must be refused, not priced here; see I-05 in docs/planning/known-issues.md.
"""
import math
from dataclasses import dataclass

import numpy as np
import ORE


# Actual/365Fixed plays two separate roles; they are named apart.
#
#   1. Simulation time axis: converts an ORE.Date to the year fraction that indexes the
#      simulated curve cube. Fixed at ACT/365; pricers look cashflow times up on the
#      cube's pillars (`engine.instruments.swap._maturity_indices`), so changing it would
#      silently misalign every pricer.
#   2. Instrument accrual: the day count a contract's coupons accrue on. Per booking;
#      the `accrual_day_count` argument of `build_vanilla_swap`.
#
#: Role 1. Other modules import this object rather than constructing their own
#: (identity checked by `tests/test_day_count_roles.py::TestTimeAxisIsOneObject`).
TIME_AXIS_DAY_COUNTER = ORE.Actual365Fixed()

#: Deprecated alias for `TIME_AXIS_DAY_COUNTER` (the time axis, not accrual).
DAY_COUNTER = TIME_AXIS_DAY_COUNTER


def time_from_reference(evaluation_date: ORE.Date, date: ORE.Date) -> float:
    """A date's position on the simulation time axis, as ORE's
    `timeFromReference(d)` on a curve with day counter `TIME_AXIS_DAY_COUNTER`. Equal
    dates map to identical floats, so exercise dates match accrual dates exactly."""
    return TIME_AXIS_DAY_COUNTER.yearFraction(evaluation_date, date)

#: The accrual day-count vocabulary lives in the leaf module `engine.day_count`, so that
#: `engine.integration` can use it without importing this module (see that module).
#: Re-exported here for existing callers.
from engine.day_count import (  # noqa: E402  (re-export, see above)
    DEFAULT_ACCRUAL_DAY_COUNT,
    SUPPORTED_ACCRUAL_DAY_COUNTS,
    UnsupportedDayCountError,
    resolve_accrual_day_count,
)


#: Fixing calendar and settlement lag of the generic index. `MakeVanillaSwap` also uses
#: them to place a tenor-quoted swap's start date.
SWAP_CALENDAR = ORE.TARGET()
SPOT_LAG_DAYS = 2


def validate_tenor(period_str: str, field_name: str) -> None:
    """`period_str` must parse as an `ORE.Period` ("5Y", "18M"); raises a `ValueError`
    naming the field, rather than failing later inside `MakeVanillaSwap`."""
    try:
        ORE.Period(period_str)
    except Exception as exc:
        raise ValueError(f"{field_name} is not a valid ORE.Period string: {period_str!r} ({exc})") from exc


def resolve_swap_dates(trade_date: ORE.Date, swap_tenor: str, forward_start: ORE.Period = None):
    """`(effective_date, maturity_date)` for a swap quoted as a tenor on `trade_date`,
    following `MakeVanillaSwap` (QuantLib/ql/instruments/makevanillaswap.cpp):

      spot      = calendar.advance(calendar.adjust(trade_date), 2 business days)
      effective = spot + forward_start, adjusted Following (Preceding if negative)
      maturity  = effective + swap_tenor, unadjusted (the schedule adjusts it)

    Called once at booking; the resulting dates are the trade. Checked against
    `MakeVanillaSwap`'s tenor path in tests/test_trade_dates.py."""
    forward_start = forward_start if forward_start is not None else ORE.Period(0, ORE.Days)
    spot = SWAP_CALENDAR.advance(SWAP_CALENDAR.adjust(trade_date), SPOT_LAG_DAYS, ORE.Days)
    effective = spot + forward_start
    if forward_start.length() > 0:
        effective = SWAP_CALENDAR.adjust(effective, ORE.Following)
    elif forward_start.length() < 0:
        effective = SWAP_CALENDAR.adjust(effective, ORE.Preceding)
    return effective, effective + ORE.Period(swap_tenor)


def book_swap_dates(cfg, swap_tenor, forward_start=None) -> None:
    """`__post_init__` step shared by every config holding a swap.

    A tenor is resolved to dates on the config's `evaluation_date`; the dates are then
    required and validated. Giving both a tenor and dates is an error. The tenor is an
    `InitVar`, so `dataclasses.replace` copies the dates and not the tenor, and a booked
    trade keeps its dates on any later evaluation date."""
    if swap_tenor is not None:
        validate_tenor(swap_tenor, "swap_tenor")
        if cfg.effective_date is not None or cfg.maturity_date is not None:
            raise ValueError("give either swap_tenor or effective_date/maturity_date, not both")
        cfg.effective_date, cfg.maturity_date = resolve_swap_dates(
            cfg.evaluation_date, swap_tenor, forward_start)
    elif forward_start is not None:
        raise ValueError("forward_start is only meaningful with swap_tenor")
    elif cfg.effective_date is None and cfg.maturity_date is None:
        raise ValueError("a swap needs its dates: give effective_date and maturity_date, or swap_tenor")
    for name in ("effective_date", "maturity_date"):
        if not isinstance(getattr(cfg, name), ORE.Date):
            raise TypeError(f"{name} must be an ORE.Date; got {getattr(cfg, name)!r}")
    if not cfg.effective_date < cfg.maturity_date:
        raise ValueError(
            f"effective_date ({cfg.effective_date}) must be before maturity_date ({cfg.maturity_date})")


def validate_fixings(fixings) -> None:
    """Historical index fixings are `{ORE.Date: finite rate}`."""
    for date, value in fixings.items():
        if not isinstance(date, ORE.Date):
            raise TypeError(f"fixings must be keyed by ORE.Date; got {date!r}")
        if not math.isfinite(value):
            raise ValueError(f"fixing on {date} must be finite; got {value}")


class MissingFixingError(ValueError):
    """A coupon fixed before the evaluation date and no fixing was supplied
    for it -- ORE's own "Missing ... fixing" failure."""


def known_fixing(fixing_date: ORE.Date, today: ORE.Date, fixings) -> "float | None":
    """The fixing ORE treats as known on `today`, or `None` if ORE forecasts it, as
    `InterestRateIndex::fixing` (QuantLib/ql/indexes/interestrateindex.cpp) under default
    settings:

      * after today: forecast (a supplied value is ignored);
      * today: the supplied fixing if present, else forecast;
      * before today: the supplied fixing, else `MissingFixingError`."""
    if fixing_date > today:
        return None
    if fixing_date in fixings:
        return float(fixings[fixing_date])
    if fixing_date == today:
        return None
    raise MissingFixingError(
        f"missing fixing for {fixing_date}: the coupon fixed before the evaluation date {today}; "
        f"supply it in the trade's fixings"
    )


def ibor_index(tenor_months: int, forwarding_curve: "ORE.YieldTermStructureHandle" = None) -> ORE.IborIndex:
    """The generic Ibor index every swap here references (`SimIndex<N>M`): `tenor_months`
    fixing period, 2 settlement days, TARGET, Modified Following, no end-of-month,
    ACT/365, forecasting off `forwarding_curve` (none by default: the engine reads only the
    schedule)."""
    return ORE.IborIndex(
        "SimIndex", ORE.Period(tenor_months, ORE.Months), SPOT_LAG_DAYS,
        ORE.USDCurrency(), SWAP_CALENDAR, ORE.ModifiedFollowing, False,
        TIME_AXIS_DAY_COUNTER, forwarding_curve or ORE.YieldTermStructureHandle(),
    )


def build_vanilla_swap(
    notional: float,
    fixed_rate: float,
    payer: bool,
    effective_date: ORE.Date,
    maturity_date: ORE.Date,
    index_tenor_months: int,
    floating_spread: float,
    accrual_day_count=None,
) -> ORE.VanillaSwap:
    """An `ORE.VanillaSwap` built with `MakeVanillaSwap` from explicit booked dates, so
    the schedule does not depend on the evaluation date.

    `accrual_day_count` sets both legs' accrual (a name from
    `SUPPORTED_ACCRUAL_DAY_COUNTS` or an `ORE.DayCounter`; default ACT/365). The index
    itself keeps `TIME_AXIS_DAY_COUNTER` and has no forwarding curve: the engine never
    reads ORE's forecasts, only the schedule and accrual fractions.
    """
    accrual = resolve_accrual_day_count(accrual_day_count)
    index = ibor_index(index_tenor_months)
    swap_type = ORE.VanillaSwap.Payer if payer else ORE.VanillaSwap.Receiver
    return ORE.MakeVanillaSwap(
        ORE.Period(0, ORE.Days), index, fixed_rate,
        effectiveDate=effective_date,
        terminationDate=maturity_date,
        nominal=notional,
        swapType=swap_type,
        floatingLegSpread=floating_spread,
        # Accrual role; everything else here is the time axis.
        fixedLegDayCount=accrual,
        floatingLegDayCount=accrual,
    )


def par_coupon_forecast_period(coupon) -> "tuple[ORE.Date, ORE.Date, float]":
    """`(start, end, spanning_time)` over which an at-par Ibor coupon forecasts its rate,
    as QuantLib's `IborCouponPricer::initializeCachedData` (ql/cashflows/couponpricer.cpp)
    for a coupon fixed in advance with at-par coupons (ORE's default):

        start = fixingCalendar.advance(fixingDate, fixingDays)          (fixingValueDate)
        end   = fixingCalendar.advance(fixingCalendar.advance(accrualEnd, -fixingDays),
                                       fixingDays), at least start + 1  (fixingEndDate)
        spanning_time = index day counter's yearFraction(start, end)

    The forecast is `(P(start)/P(end) - 1) / spanning_time` on the forwarding curve
    (`IborIndex::forecastFixing`), and the coupon pays it times the leg's own accrual
    fraction. Dividing by the accrual fraction instead is wrong whenever the leg and the
    index count days differently (I-36)."""
    index = coupon.index()
    calendar = index.fixingCalendar()
    fixing_days = coupon.fixingDays()
    start = calendar.advance(coupon.fixingDate(), index.fixingDays(), ORE.Days)
    next_fixing = calendar.advance(coupon.accrualEndDate(), -fixing_days, ORE.Days)
    end = max(calendar.advance(next_fixing, index.fixingDays(), ORE.Days), start + 1)
    return start, end, index.dayCounter().yearFraction(start, end)


@dataclass
class LegCashflows:
    """One leg's remaining cashflows on `today`, times in years from `today`. With
    fixings, `is_fixed`/`fixed_rates` mark coupons whose fixing is already known
    (`known_fixing`); the rest are projected over `forecast_start_times` to
    `forecast_end_times`, divided by `spanning_times` (`par_coupon_forecast_period`;
    floating legs only)."""
    payment_times: np.ndarray        # [N]
    accrual_start_times: np.ndarray  # [N]
    accrual_end_times: np.ndarray    # [N]
    accrual_fractions: np.ndarray    # [N]
    notional: float
    is_fixed: np.ndarray = None      # [N] bool
    fixed_rates: np.ndarray = None   # [N] the fixing where is_fixed, else 0
    forecast_start_times: np.ndarray = None  # [N] floating legs only
    forecast_end_times: np.ndarray = None    # [N]
    spanning_times: np.ndarray = None        # [N] index day count over the forecast period


def is_live(cashflow_date: ORE.Date, today: ORE.Date) -> bool:
    """Whether a cashflow is still live on `today`: QuantLib's `CashFlow::hasOccurred`
    under ORE defaults, where a cashflow paid on the evaluation date has occurred."""
    return cashflow_date > today


def _leg_cashflows(leg, as_coupon, today: ORE.Date, notional: float, fixings=None,
                   floating: bool = False) -> LegCashflows:
    t_of = lambda d: TIME_AXIS_DAY_COUNTER.yearFraction(today, d)  # noqa: E731
    payment_times, accrual_starts, accrual_ends, fractions = [], [], [], []
    is_fixed, fixed_rates = [], []
    forecast_starts, forecast_ends, spanning = [], [], []
    for cf in leg:
        c = as_coupon(cf)
        if not is_live(c.date(), today):
            continue
        payment_times.append(t_of(c.date()))
        accrual_starts.append(t_of(c.accrualStartDate()))
        accrual_ends.append(t_of(c.accrualEndDate()))
        fractions.append(c.accrualPeriod())
        if floating:
            start, end, span = par_coupon_forecast_period(c)
            forecast_starts.append(t_of(start))
            forecast_ends.append(t_of(end))
            spanning.append(span)
        if fixings is not None:
            rate = known_fixing(c.fixingDate(), today, fixings)
            is_fixed.append(rate is not None)
            fixed_rates.append(0.0 if rate is None else rate)
    as_array = lambda values: np.array(values, dtype=np.float64) if floating else None  # noqa: E731
    return LegCashflows(
        payment_times=np.array(payment_times),
        accrual_start_times=np.array(accrual_starts),
        accrual_end_times=np.array(accrual_ends),
        accrual_fractions=np.array(fractions),
        notional=notional,
        is_fixed=None if fixings is None else np.array(is_fixed, dtype=bool),
        fixed_rates=None if fixings is None else np.array(fixed_rates, dtype=np.float64),
        forecast_start_times=as_array(forecast_starts),
        forecast_end_times=as_array(forecast_ends),
        spanning_times=as_array(spanning),
    )


def fixed_leg_cashflows(swap: ORE.VanillaSwap, today: ORE.Date) -> LegCashflows:
    """Remaining fixed coupons' payment/accrual times and ORE `accrualPeriod()`s."""
    notional = swap.fixedNominals()[0] if swap.fixedNominals() else swap.nominal()
    return _leg_cashflows(swap.fixedLeg(), ORE.as_fixed_rate_coupon, today, notional)


def floating_leg_cashflows(swap: ORE.VanillaSwap, today: ORE.Date, fixings=None) -> LegCashflows:
    """As `fixed_leg_cashflows`, for the floating leg.

    With `fixings`, each coupon is marked known or projected as ORE decides
    (`known_fixing`); a coupon fixed before `today` without a fixing raises
    `MissingFixingError`. Without `fixings` only the schedule is read."""
    notional = swap.floatingNominals()[0] if swap.floatingNominals() else swap.nominal()
    return _leg_cashflows(swap.floatingLeg(), ORE.as_floating_rate_coupon, today, notional, fixings,
                          floating=True)
