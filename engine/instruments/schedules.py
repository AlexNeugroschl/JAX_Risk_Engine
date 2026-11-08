"""
Builds ORE trade objects from trade configs and extracts their cashflow schedules.

Schedules, calendars and accrual fractions come from ORE (`MakeVanillaSwap`, coupon
`accrualPeriod()`), not reimplemented. Day counts are always set explicitly, never left to
`MakeVanillaSwap`'s per-index defaults.

Scope: generic term-Ibor swaps, whose `build_vanilla_swap` index is `SimIndex<N>M` on a
TARGET calendar with ACT/365 accrual by default, and (TraderX fork, R3) overnight-index swaps
named by `overnight_index` (`OVERNIGHT_INDICES`, `build_overnight_swap`): both legs on the
index's day count and calendar, paid `payment_lag_days` after each period, the floating coupon
compounded in arrears with no lookback or lockout. On one forwarding curve a compounded
overnight coupon is exactly `P(start)/P(end) - 1` (the daily forwards telescope), which is the
at-par coupon of a term index whose tenor is the period and whose day count is the leg's, so
the coupon is that coupon and every pricer reads it unchanged.
"""
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import ORE

from engine.market_data.day_counts import TIME_AXIS_DAY_COUNTER, resolve_accrual_day_count


@contextmanager
def evaluation_date(date: ORE.Date):
    """ORE's global evaluation date set to `date` inside the block and restored after it.
    For ORE objects that read it while the engine extracts their data (a `SwaptionHelper`
    building its underlying asks for every fixing before it), so they see the date they are
    built on, not whatever a previous caller left there or the wall-clock default."""
    settings = ORE.Settings.instance()
    previous = settings.evaluationDate
    settings.evaluationDate = date
    try:
        yield
    finally:
        settings.evaluationDate = previous


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
    overnight_index: Optional[OvernightIndex] = None,
    today: Optional[ORE.Date] = None,
    overnight_fixings=None,
) -> ORE.VanillaSwap:
    """An `ORE.VanillaSwap` built with `MakeVanillaSwap` from explicit booked dates, so
    the schedule does not depend on the evaluation date.

    `accrual_day_count` sets both legs' accrual (a name from
    `SUPPORTED_ACCRUAL_DAY_COUNTS` or an `ORE.DayCounter`; default ACT/365). The index
    itself keeps `TIME_AXIS_DAY_COUNTER` and has no forwarding curve: the engine never
    reads ORE's forecasts, only the schedule and accrual fractions.
    """
    if overnight_index is not None:
        return build_overnight_swap(notional, fixed_rate, effective_date, maturity_date,
                                    index_tenor_months, floating_spread, overnight_index, today, overnight_fixings)
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


@dataclass(frozen=True)
class OvernightIndex:
    """An overnight index's swap conventions (TraderX fork, R3): its currency, the calendar its
    schedule rolls Modified Following on and its payments are lagged on, the day count of both
    legs and of the index, and the payment lag in business days."""
    currency: str
    calendar: object
    day_count: str
    payment_lag_days: int
    #: The calendar the index publishes on, whose business days its fixings compound over
    #: (ORE's own index's `fixingCalendar()`).
    fixing_calendar: object


#: The overnight indices an `overnight_index` swap may name, with the standard OIS conventions:
#: SOFR on the US government-bond calendar paid 2 days late, ESTR on TARGET paid 1 day late,
#: SONIA on the UK calendar paid on the period end, TONA on the Japanese calendar paid 2 days late.
OVERNIGHT_INDICES = {
    "USD-SOFR": OvernightIndex("USD", ORE.UnitedStates(ORE.UnitedStates.GovernmentBond), "ACT/360", 2,
                               ORE.Sofr().fixingCalendar()),
    "EUR-ESTR": OvernightIndex("EUR", ORE.TARGET(), "ACT/360", 1, ORE.Estr().fixingCalendar()),
    "GBP-SONIA": OvernightIndex("GBP", ORE.UnitedKingdom(), "ACT/365", 0, ORE.Sonia().fixingCalendar()),
    "JPY-TONA": OvernightIndex("JPY", ORE.Japan(), "ACT/365", 2, ORE.Tonar().fixingCalendar()),
}


def overnight_index(name: Optional[str], currency: str) -> Optional[OvernightIndex]:
    """The conventions of `name` (None for a term-index swap); an unknown index, or one in
    another currency than the trade's, is refused."""
    if name is None:
        return None
    if name not in OVERNIGHT_INDICES:
        raise ValueError(f"overnight_index {name!r} is not supported; this engine implements {sorted(OVERNIGHT_INDICES)}")
    index = OVERNIGHT_INDICES[name]
    if index.currency != currency:
        raise ValueError(f"overnight_index {name!r} is a {index.currency} index; the trade is in {currency}")
    return index


class OvernightSwap:
    """An overnight-index swap as the pricers read a swap: `fixedLeg()`, `floatingLeg()` (ORE
    coupons with their lagged payment dates) and `maturityDate()`, the latest accrual end, as
    `ORE.Swap::maturityDate` gives it."""

    def __init__(self, fixed, floating, fixings=None):
        self._fixed, self._floating = fixed, floating
        #: The term fixings of the coupons the realised compounding fixed
        #: (`build_overnight_swap`), which `engine.pricing.legs.legs_of` reads.
        self.fixings = fixings or {}

    def fixedLeg(self):
        return self._fixed

    def floatingLeg(self):
        return self._floating

    def maturityDate(self) -> ORE.Date:
        return max(ORE.as_coupon(c).accrualEndDate() for c in (*self._fixed, *self._floating))

    def fixedNominals(self):
        return [ORE.as_coupon(c).nominal() for c in self._fixed]

    def nominal(self) -> float:
        return self.fixedNominals()[0]


def build_overnight_swap(notional: float, fixed_rate: float, effective_date: ORE.Date, maturity_date: ORE.Date,
                         index_tenor_months: int, floating_spread: float, index: OvernightIndex,
                         today: Optional[ORE.Date] = None, overnight_fixings=None) -> OvernightSwap:
    """An overnight-index swap paying every `index_tenor_months` on both legs (see the module
    docstring). The floating coupons reference a term index over the period with no fixing lag,
    so they fix on their accrual start and forecast over exactly their accrual.

    With `today`, a coupon accruing on it (started, not yet paid) pays the compounding realised
    from `overnight_fixings` (`{ORE.Date: rate}`, the index's daily fixings) times the forecast
    from the first day not yet fixed, `split`:

        N (C x P(split)/P(end) - 1) = N (C - 1)  +  N C (P(split)/P(end) - 1)

    so it is replaced by two coupons paid on its payment date: one over `[start, split]` on
    `N` with the known rate `(C - 1) / accrual`, and one over `[split, end]` on `N C`, which
    forecasts as any other (`realised_compounding`). A spread is refused on such a coupon (its
    amount would need a spread on `N` over the whole period, which one leg spread cannot carry)."""
    calendar, day_count = index.calendar, resolve_accrual_day_count(index.day_count)
    period = ORE.Period(index_tenor_months, ORE.Months)
    schedule = ORE.Schedule(effective_date, maturity_date, period, calendar, ORE.ModifiedFollowing,
                            ORE.ModifiedFollowing, ORE.DateGeneration.Forward, False)
    term = ORE.IborIndex("SimON", period, 0, ORE.USDCurrency(), calendar, ORE.ModifiedFollowing, False, day_count)
    fixed = ORE.FixedRateLeg(schedule, day_count, [notional], [fixed_rate], ORE.ModifiedFollowing,
                             paymentCalendar=calendar, paymentLag=index.payment_lag_days)
    floating = ORE.IborLeg([notional], schedule, term, day_count, ORE.ModifiedFollowing, [0], [1.0],
                           [floating_spread], paymentCalendar=calendar, paymentLag=index.payment_lag_days,
                           withIndexedCoupons=False)
    if today is None:
        return OvernightSwap(fixed, floating)
    coupons, fixings = [], {}
    for c in map(ORE.as_floating_rate_coupon, floating):
        start, end = c.accrualStartDate(), c.accrualEndDate()
        if not start < today < c.date():
            coupons.append(c)
            continue
        split, factor = realised_compounding(index, start, end, today, overnight_fixings or {})
        if split < end and floating_spread != 0.0:
            raise ValueError(f"an {index.currency} overnight coupon accruing {start.ISO()} to {end.ISO()} is in "
                             f"progress on {today.ISO()}: a floating spread on it is not supported")
        known = ORE.IborCoupon(c.date(), notional, start, split, 0, term, 1.0, floating_spread, start, split, day_count)
        fixings[start] = (factor - 1.0) / day_count.yearFraction(start, split)
        coupons.append(known)
        if split < end:
            coupons.append(ORE.IborCoupon(c.date(), notional * factor, split, end, 0, term, 1.0, 0.0, split, end,
                                          day_count))
    return OvernightSwap(fixed, ORE.Leg(coupons), fixings)


def realised_compounding(index: OvernightIndex, start: ORE.Date, end: ORE.Date, today: ORE.Date,
                         fixings) -> "tuple[ORE.Date, float]":
    """`(split, C)`: the overnight compounding of `[start, end]` known on `today`, as QuantLib's
    `OvernightIndexedCoupon` compounds it with no lookback or lockout. Each business day `d` of
    the index's fixing calendar from `start` accrues its fixing to the next one (or `end`),
    `C *= 1 + fixing(d) x yearFraction(d, next)`, while `known_fixing` takes the fixing as known
    (before `today`: required, else `MissingFixingError`; on it: if supplied); `split` is the
    first day not fixed, `end` once every day is."""
    calendar, day_count = index.fixing_calendar, resolve_accrual_day_count(index.day_count)
    # ponytail: `start` taken as a fixing-calendar business day, as every OIS schedule date is.
    d, factor = start, 1.0
    while d < end:
        try:
            rate = known_fixing(d, today, fixings)
        except MissingFixingError:
            raise MissingFixingError(f"missing overnight fixing for {d}: the coupon accruing {start.ISO()} to "
                                     f"{end.ISO()} is in progress on {today}; supply every past day's fixing in "
                                     f"the trade's overnight_fixings") from None
        if rate is None:
            break
        following = min(calendar.advance(d, 1, ORE.Days), end)
        factor *= 1.0 + rate * day_count.yearFraction(d, following)
        d = following
    return d, factor


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


def is_live(cashflow_date: ORE.Date, today: ORE.Date) -> bool:
    """Whether a cashflow is still live on `today`: QuantLib's `CashFlow::hasOccurred`
    under ORE defaults, where a cashflow paid on the evaluation date has occurred."""
    return cashflow_date > today
