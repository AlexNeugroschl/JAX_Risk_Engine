"""
Shared ORE trade-building and cashflow-extraction helpers.

**The single source of truth for turning a trade config into a real ORE
object and its cashflow schedule.** Before this module existed, `_build_ore_
swap` was defined three separate times -- `engine/instruments/swap.py`,
`engine/instruments/european_swaption.py`, `engine/instruments/
bermudan_swaption.py` -- with identical bodies except that
`european_swaption.py`'s version additionally passed `forwardStart` to
`ORE.MakeVanillaSwap` (`bermudan_swaption.py`'s own docstring said so
explicitly: "identical pattern to swap._build_ore_swap /
european_swaption._build_ore_swap"). The per-leg cashflow-extraction loop
(iterate `swap.fixedLeg()`/`floatingLeg()`, pull `payment/accrualStart/
accrualEnd` dates and `accrualPeriod()` off each ORE coupon) was likewise
hand-copied four times across those same modules (fixed + floating legs in
`swap.py` and `bermudan_swaption.py`, fixed leg only in
`european_swaption.py`, which collapses the floating leg to a telescoping-
notional identity instead -- see `engine.instruments.european_swaption`'s
own module docstring for why that shortcut is valid there specifically).
This module replaces all of it with one implementation, used by every
instrument pricer.

Day counts are always set explicitly here, never left to
`MakeVanillaSwap`'s implicit per-index defaults (which differ unpredictably
by index/currency -- e.g. Euribor6M defaults to 30/360 fixed vs Act/360
float). Which day count, though, depends on WHICH OF TWO ROLES is being
filled -- see the TWO ROLES block below `import ORE`. The simulation time
axis is `TIME_AXIS_DAY_COUNTER` and is permanently ACT/365; a trade's own
coupon accrual is `build_vanilla_swap`'s `accrual_day_count` argument,
which defaults to ACT/365 so this module's long-standing behavior is
unchanged for every caller that does not ask for something else (W1.1).

**Scope warning -- this builder is GENERIC TERM-IBOR ONLY.** It produces a
`SimIndex<N>M` term index, ACT/365 on both legs, a TARGET calendar, and a
schedule derived from a TENOR STRING ("5Y") rather than explicit booked
dates. A real USD-SOFR contract is an OVERNIGHT index, ACT/360, daily
compounded in arrears, on a US calendar, with explicit effective/maturity
dates plus lookback/lockout/payment-lag terms this signature cannot even
express. Routing such a booking through here produces a confident, WRONG
number: ACT/360-vs-ACT/365 alone shifts every accrual factor by 1.389%
(~$1,906 on a $1mm 5Y fixed leg, roughly 46x a 1bp DV01), and no test in
this repository would catch it, because every test builds its inputs with
this same builder.

Registered as **I-05** in docs/known-issues.md. The ACT/365 choice above is
deliberate and should stay; a faithful SOFR path belongs in a SEPARATE
builder alongside this one (every swaption pricer depends on this one's
consistency with the simulation time axis), gated on the convention set
agreed in decisions D03/D04, with unsupported conventions REFUSED rather
than approximated here.
"""
from dataclasses import dataclass

import numpy as np
import ORE

# =============================================================================
# THE TWO ROLES Actual/365Fixed PLAYS HERE, AND WHY THEY MUST BE NAMED APART
#
# `Actual365Fixed` was used for two unrelated jobs under one name, which is
# what made "make the day count per-instrument" look like a one-line change
# when it is not (plan §W1.1).
#
#   1. SIMULATION TIME AXIS -- converting an ORE.Date into the year-fraction
#      that indexes `time_grid`, `maturities` and `hw_paths`. **This must
#      stay ACT/365 forever.** Every pricer's cashflow times are looked up
#      against the simulated curve cube's own axis (see
#      `engine.instruments.swap._maturity_indices`, which requires each
#      cashflow time to land EXACTLY on a simulation maturity pillar).
#      Changing this silently desynchronizes every pricer from the cube --
#      no error, just wrong discount factors.
#
#   2. INSTRUMENT ACCRUAL -- the day count a contract's coupons actually
#      accrue on. **This is a property of the booking, not of the engine**,
#      and must be per-instrument: the TraderX note is ACT/ACT (ICMA), a
#      USD-SOFR swap is ACT/360.
#
# `TIME_AXIS_DAY_COUNTER` is role 1 and is not configurable. Role 2 is the
# `accrual_day_count` argument on `build_vanilla_swap` below, defaulting to
# ACT/365 so every pre-existing caller is byte-identical.
#
# `DAY_COUNTER` remains as a deprecated alias for role 1 so no import
# breaks; prefer the explicit name in new code.
# =============================================================================
#: **The single source of truth for role 1.** `engine.instruments.
#: bermudan_swaption` and `engine.risk.greeks` import this object rather
#: than constructing their own; they used to do the latter, which left
#: three equal-but-distinct copies of a value whose entire point is that it
#: is fixed engine-wide, with only this one pinned by a test. Identity is
#: asserted by `tests/test_day_count_roles.py::TestTimeAxisIsOneObject`.
TIME_AXIS_DAY_COUNTER = ORE.Actual365Fixed()

#: Deprecated alias for `TIME_AXIS_DAY_COUNTER`. Kept so existing imports
#: keep working; it always meant the time axis, never instrument accrual.
DAY_COUNTER = TIME_AXIS_DAY_COUNTER

#: ---------------------------------------------------------------------
#: The accrual day-count vocabulary lives in `engine.day_count` and is
#: re-exported here so every existing caller and test keeps working
#: unchanged.
#:
#: **Why it moved (W1.3).** `engine/integration/note.py` needs the same
#: allowlist, and `engine/integration/` is forbidden from importing
#: `engine.models` -- this module is where `build_vanilla_swap` lives, the
#: exact object W0.4's refusal keeps unreachable (I-05). Borrowing the
#: table by importing this module would put that builder one attribute
#: access from the refusal boundary, so the table moved to a leaf module
#: that imports only ORE. See `engine.day_count` for the full rationale.
#:
#: The *time axis* role above deliberately did NOT move: it is a property
#: of this engine's simulated curve cube, not of any contract.
#: ---------------------------------------------------------------------
from engine.day_count import (  # noqa: E402  (re-export, see above)
    DEFAULT_ACCRUAL_DAY_COUNT,
    SUPPORTED_ACCRUAL_DAY_COUNTS,
    UnsupportedDayCountError,
    resolve_accrual_day_count,
)


def build_vanilla_swap(
    notional: float,
    fixed_rate: float,
    payer: bool,
    swap_tenor: str,
    index_tenor_months: int,
    floating_spread: float,
    evaluation_date: ORE.Date,
    forward_start: ORE.Period = None,
    accrual_day_count=None,
) -> ORE.VanillaSwap:
    """Builds a real `ORE.VanillaSwap` (schedules, day counts, conventions)
    via `ORE.MakeVanillaSwap` -- date generation and accrual math match ORE
    exactly rather than being reimplemented. `forward_start` (an
    `ORE.Period` the swap's first accrual is delayed by beyond the
    standard spot lag) defaults to no delay; only
    `engine.instruments.european_swaption` currently passes a non-default
    value (a swaption's own `forward_start`), but any instrument needing a
    forward-starting underlying can use it the same way.

    `accrual_day_count` is the **instrument accrual** role (see this
    module's TWO ROLES block): the day count this swap's coupons accrue on,
    by name from `SUPPORTED_ACCRUAL_DAY_COUNTS` or as an `ORE.DayCounter`.
    Defaults to ACT/365, which is what every caller got before this
    parameter existed, so existing behavior is byte-identical.

    Note what does NOT take it: the index's own day count and the dummy
    forward curve's, both of which stay `TIME_AXIS_DAY_COUNTER`. The index
    day count feeds ORE's forward-rate calculation, but this engine never
    reads ORE's forwards -- it reprices against the JAX-simulated cube
    whose axis is ACT/365 (see `floating_leg_cashflows`' docstring). Only
    the LEG accrual fractions, which `fixed_leg_cashflows` reads back out
    via `accrualPeriod()`, are the contract's own accrual.
    """
    ORE.Settings.instance().evaluationDate = evaluation_date
    accrual = resolve_accrual_day_count(accrual_day_count)
    dummy_forward_curve = ORE.YieldTermStructureHandle(
        ORE.FlatForward(evaluation_date, 0.0, TIME_AXIS_DAY_COUNTER)
    )
    index = ORE.IborIndex(
        "SimIndex", ORE.Period(index_tenor_months, ORE.Months), 2,
        ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False,
        TIME_AXIS_DAY_COUNTER, dummy_forward_curve,
    )
    swap_type = ORE.VanillaSwap.Payer if payer else ORE.VanillaSwap.Receiver
    kwargs = dict(
        nominal=notional,
        swapType=swap_type,
        floatingLegSpread=floating_spread,
        # The two accrual-role lines -- everything else in this function is
        # the time axis.
        fixedLegDayCount=accrual,
        floatingLegDayCount=accrual,
    )
    if forward_start is not None:
        kwargs["forwardStart"] = forward_start
    return ORE.MakeVanillaSwap(ORE.Period(swap_tenor), index, fixed_rate, **kwargs)


@dataclass
class LegCashflows:
    """One leg's cashflow schedule, as year-fractions from `today`."""
    payment_times: np.ndarray        # [N]
    accrual_start_times: np.ndarray  # [N]
    accrual_end_times: np.ndarray    # [N]
    accrual_fractions: np.ndarray    # [N]
    notional: float


def fixed_leg_cashflows(swap: ORE.VanillaSwap, today: ORE.Date) -> LegCashflows:
    """Extracts each fixed coupon's payment/accrual dates (as
    year-fractions from `today`) and ORE's own `accrualPeriod()` for each,
    from the real ORE-generated schedule -- no date/day-count math
    reimplemented here."""
    payment_times, accrual_starts, accrual_ends, fractions = [], [], [], []
    for cf in swap.fixedLeg():
        c = ORE.as_fixed_rate_coupon(cf)
        payment_times.append(TIME_AXIS_DAY_COUNTER.yearFraction(today, c.date()))
        accrual_starts.append(TIME_AXIS_DAY_COUNTER.yearFraction(today, c.accrualStartDate()))
        accrual_ends.append(TIME_AXIS_DAY_COUNTER.yearFraction(today, c.accrualEndDate()))
        fractions.append(c.accrualPeriod())
    return LegCashflows(
        payment_times=np.array(payment_times),
        accrual_start_times=np.array(accrual_starts),
        accrual_end_times=np.array(accrual_ends),
        accrual_fractions=np.array(fractions),
        notional=swap.fixedNominals()[0] if swap.fixedNominals() else swap.nominal(),
    )


def floating_leg_cashflows(swap: ORE.VanillaSwap, today: ORE.Date) -> LegCashflows:
    """Same extraction as `fixed_leg_cashflows`, for the floating leg's
    coupons. `accrual_start`/`accrual_end` times are what forward rates get
    computed from downstream -- ORE's own fixing is never read, since the
    whole point is repricing under JAX-simulated scenarios, not ORE's own
    (single, deterministic) fixing history."""
    payment_times, accrual_starts, accrual_ends, fractions = [], [], [], []
    for cf in swap.floatingLeg():
        c = ORE.as_floating_rate_coupon(cf)
        payment_times.append(TIME_AXIS_DAY_COUNTER.yearFraction(today, c.date()))
        accrual_starts.append(TIME_AXIS_DAY_COUNTER.yearFraction(today, c.accrualStartDate()))
        accrual_ends.append(TIME_AXIS_DAY_COUNTER.yearFraction(today, c.accrualEndDate()))
        fractions.append(c.accrualPeriod())
    return LegCashflows(
        payment_times=np.array(payment_times),
        accrual_start_times=np.array(accrual_starts),
        accrual_end_times=np.array(accrual_ends),
        accrual_fractions=np.array(fractions),
        notional=swap.floatingNominals()[0] if swap.floatingNominals() else swap.nominal(),
    )
