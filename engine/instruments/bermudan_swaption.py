"""
Bermudan swaptions: the trade, and the exercise style it shares with American swaptions
(`engine.instruments.american_swaption`).

The trade names its currency, index, exercise and settlement; it carries no curve or model.
ORE prices Bermudans and Americans with `QuantExt::NumericLgmMultiLegOptionEngine` (Grid
solver), reproduced in JAX in `engine.pricing.lgm_grid`; its model's reversion, volatility and
grid settings come from the pricing configuration (`engine.pricing.config.LgmSwaptionEngineConfig`,
ORE's `LGMGridSwaptionEngineBuilder`), calibrated per trade (`engine.pricing.bermudan`).

Exercise is in dates, converted to times on the simulation's time axis as ORE derives
`optionTimes`. A Bermudan exercise inside an accrual period enters the next whole period, as
in ORE (`ExerciseStyle`). See docs/instruments/american-bermudan-swaptions.md.
"""
from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import ClassVar, Dict, List, Optional, Sequence

import ORE

from engine.instruments._validation import _validate_common_fields, _validate_identity, _validate_settlement
from engine.instruments.schedules import book_swap_dates, build_vanilla_swap, is_live, validate_fixings
from engine.market_data.day_counts import time_from_reference


class ExerciseStyle(Enum):
    """Which of ORE's two coupon-membership rules an option's exercise uses
    -- `NumericLgmMultiLegOptionEngineBase::buildCashflowInfo`
    (QuantExt/qle/pricingengines/numericlgmmultilegoptionengine.cpp):

      * BERMUDAN: a coupon belongs to the exercised-into swap while
        `t <= accrualStart`. An exercise inside a period enters the next
        WHOLE period ("bermudan exercise implies that we always exercise
        into whole periods").
      * AMERICAN: a coupon belongs while `t <= accrualEnd`, credited
        `couponRatio(t)` of its value ("american exercise implies that we
        can exercise into broken periods").
    """
    BERMUDAN = "bermudan"
    AMERICAN = "american"


@dataclass
class BermudanSwaptionConfig:
    """
    One Bermudan swaption: the option to enter a vanilla swap on any of a list of dates.

    trade_id / evaluation_date: as in `SwapConfig` (required, keyword only).
    currency / settlement: as in `SwaptionConfig`.
    exercise_dates: ascending `ORE.Date`s. Converted to times with the curve's day counter,
        as ORE derives `optionTimes`, so an exercise date and the accrual date it names map
        to the same float. Dates on or before `evaluation_date` are not exercise
        opportunities. A date inside an accrual period enters the next whole period, as in
        ORE. `exercisable_dates(cfg)` lists the underlying's accrual starts.
    effective_date / maturity_date / swap_tenor / fixings: as in `SwapConfig`. A fixing is
        needed only for a coupon fixed before `evaluation_date` that can still be entered.
    """
    notional: float
    fixed_rate: float
    payer: bool
    exercise_dates: Sequence[ORE.Date] = ()
    effective_date: Optional[ORE.Date] = None
    maturity_date: Optional[ORE.Date] = None
    swap_tenor: InitVar[Optional[str]] = None
    index_tenor_months: int = 6
    floating_spread: float = 0.0
    fixings: Dict[ORE.Date, float] = field(default_factory=dict)
    currency: str = "USD"
    settlement: str = "Physical"
    trade_id: str = field(kw_only=True)
    evaluation_date: ORE.Date = field(kw_only=True)
    #: The product name precision overrides are keyed by (`engine.precision.Precision.by_product`).
    product: ClassVar[str] = "bermudan_swaption"

    exercise_style = ExerciseStyle.BERMUDAN

    def option_times(self, exercise_time_steps_per_year: Optional[int] = None) -> List[float]:
        """ORE's Bermudan `optionTimes`: the time of every exercise date strictly after the
        evaluation date (`calculate()`, lines 487-493). The argument is an American's; a
        Bermudan ignores it."""
        return [time_from_reference(self.evaluation_date, d)
                for d in self.exercise_dates if d > self.evaluation_date]

    def is_expired(self) -> bool:
        """ORE's `Instrument::isExpired`: the last exercise date is on or
        before the evaluation date. An expired option is worth 0."""
        return not is_live(self.exercise_dates[-1], self.evaluation_date)

    def __post_init__(self, swap_tenor: Optional[str]) -> None:
        _validate_identity(self.trade_id, self.evaluation_date)
        _validate_common_fields(self.notional, self.fixed_rate, self.evaluation_date)
        book_swap_dates(self, swap_tenor)
        validate_fixings(self.fixings)
        _validate_settlement(self.settlement)
        if len(self.exercise_dates) == 0:
            raise ValueError("exercise_dates must be non-empty")
        if any(not isinstance(d, ORE.Date) for d in self.exercise_dates):
            raise TypeError(f"exercise_dates must be ORE.Date objects; got {list(self.exercise_dates)}")
        if any(b < a for a, b in zip(self.exercise_dates, self.exercise_dates[1:])):
            raise ValueError(f"exercise_dates must be sorted ascending; got {list(self.exercise_dates)}")


def underlying_swap(cfg) -> ORE.VanillaSwap:
    """The option's underlying, the ORE swap it exercises into (see
    `engine.instruments.schedules.build_vanilla_swap`); a Bermudan's or an American's."""
    return build_vanilla_swap(
        notional=cfg.notional, fixed_rate=cfg.fixed_rate, payer=cfg.payer,
        effective_date=cfg.effective_date, maturity_date=cfg.maturity_date,
        index_tenor_months=cfg.index_tenor_months, floating_spread=cfg.floating_spread,
    )


def exercisable_dates(cfg) -> List[ORE.Date]:
    """The underlying's own fixed accrual start dates, ascending -- the
    exercise dates of a standard coterminal Bermudan, where each exercise
    enters a whole remaining swap.

    They are a property of the ORE-generated schedule, so a caller cannot
    know them before the swap is built; this builds it and reads them off.
    Every element is a valid exercise date, including the last (it starts
    the final accrual period, strictly before final maturity)."""
    swap = underlying_swap(cfg)
    return [ORE.as_fixed_rate_coupon(cf).accrualStartDate() for cf in swap.fixedLeg()]


