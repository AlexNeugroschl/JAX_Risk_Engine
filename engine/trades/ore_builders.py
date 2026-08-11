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

Every pricer still explicitly uses `Actual/365Fixed` (`DAY_COUNTER`) on
both legs, rather than relying on `MakeVanillaSwap`'s implicit per-index
day-count defaults (which differ unpredictably by index/currency -- e.g.
Euribor6M defaults to 30/360 fixed vs Act/360 float) -- this keeps the
day-count convention a single, deliberate, documented choice consistent
with the simulation's own year-fraction time axis, not an accident of
whatever `MakeVanillaSwap` happens to default to.
"""
from dataclasses import dataclass

import numpy as np
import ORE

DAY_COUNTER = ORE.Actual365Fixed()


def build_vanilla_swap(
    notional: float,
    fixed_rate: float,
    payer: bool,
    swap_tenor: str,
    index_tenor_months: int,
    floating_spread: float,
    evaluation_date: ORE.Date,
    forward_start: ORE.Period = None,
) -> ORE.VanillaSwap:
    """Builds a real `ORE.VanillaSwap` (schedules, day counts, conventions)
    via `ORE.MakeVanillaSwap` -- date generation and accrual math match ORE
    exactly rather than being reimplemented. `forward_start` (an
    `ORE.Period` the swap's first accrual is delayed by beyond the
    standard spot lag) defaults to no delay; only
    `engine.instruments.european_swaption` currently passes a non-default
    value (a swaption's own `forward_start`), but any instrument needing a
    forward-starting underlying can use it the same way."""
    ORE.Settings.instance().evaluationDate = evaluation_date
    dummy_forward_curve = ORE.YieldTermStructureHandle(
        ORE.FlatForward(evaluation_date, 0.0, DAY_COUNTER)
    )
    index = ORE.IborIndex(
        "SimIndex", ORE.Period(index_tenor_months, ORE.Months), 2,
        ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False,
        DAY_COUNTER, dummy_forward_curve,
    )
    swap_type = ORE.VanillaSwap.Payer if payer else ORE.VanillaSwap.Receiver
    kwargs = dict(
        nominal=notional,
        swapType=swap_type,
        floatingLegSpread=floating_spread,
        fixedLegDayCount=DAY_COUNTER,
        floatingLegDayCount=DAY_COUNTER,
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
        payment_times.append(DAY_COUNTER.yearFraction(today, c.date()))
        accrual_starts.append(DAY_COUNTER.yearFraction(today, c.accrualStartDate()))
        accrual_ends.append(DAY_COUNTER.yearFraction(today, c.accrualEndDate()))
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
        payment_times.append(DAY_COUNTER.yearFraction(today, c.date()))
        accrual_starts.append(DAY_COUNTER.yearFraction(today, c.accrualStartDate()))
        accrual_ends.append(DAY_COUNTER.yearFraction(today, c.accrualEndDate()))
        fractions.append(c.accrualPeriod())
    return LegCashflows(
        payment_times=np.array(payment_times),
        accrual_start_times=np.array(accrual_starts),
        accrual_end_times=np.array(accrual_ends),
        accrual_fractions=np.array(fractions),
        notional=swap.floatingNominals()[0] if swap.floatingNominals() else swap.nominal(),
    )
