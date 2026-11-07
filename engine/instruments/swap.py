"""
A vanilla fixed-vs-floating swap: the trade and its ORE schedule (`MakeVanillaSwap`).

The trade names its currency and Ibor index; it carries no curve or model. It is priced by the
valuation pipeline (`engine.pricing.legs`: ORE's `DiscountingSwapEngine` with at-par Ibor
coupons, paid cashflows dropped and path fixings as `FixingManager` stores them) on today's
market, a bumped one, and every simulated path, whichever model simulates it.

Seasoned trades are priced as ORE prices them: cashflows paid on or before the evaluation date
are dropped, and a coupon fixed before it pays its historical fixing from `fixings`, or is
refused if missing (`schedules.known_fixing`).
"""
from dataclasses import InitVar, dataclass, field
from typing import ClassVar, Dict, Optional

import ORE

from engine.instruments._validation import _validate_common_fields, _validate_identity
from engine.instruments.schedules import (
    book_swap_dates, build_vanilla_swap, overnight_index, refuse_in_progress_overnight, validate_fixings,
)
from engine.market_data.day_counts import (
    DEFAULT_ACCRUAL_DAY_COUNT, UnsupportedDayCountError, resolve_accrual_day_count,
)


@dataclass
class SwapConfig:
    """
    One vanilla fixed-vs-floating swap.

    trade_id: the trade's id (ORE's `<Trade id>`), unique in a portfolio and echoed on every
        result. Required, keyword only.
    evaluation_date: the date the trade is valued on. Required, keyword only; a portfolio's
        trades share it, and on a `Market` it must be the market's as-of date.
    currency: the trade's currency: its discount curve, and with `index_tenor_months` its
        forwarding curve (`market.index_name`).
    effective_date / maturity_date: the booked schedule's start and unadjusted end. They
        define the trade; `evaluation_date` only sets when it is priced.
    swap_tenor: booking shortcut ("5Y", "18M"), resolved once at construction to the
        dates of a spot-starting swap traded on `evaluation_date`. Give either it or both
        dates. Not stored, so `dataclasses.replace(cfg, evaluation_date=...)` keeps the
        trade.
    index_tenor_months: floating reset frequency in months.
    fixings: historical index fixings, `{ORE.Date: rate}`; needed only for a coupon that
        fixed before `evaluation_date` and has not yet paid.
    """
    notional: float
    fixed_rate: float
    payer: bool
    effective_date: Optional[ORE.Date] = None
    maturity_date: Optional[ORE.Date] = None
    swap_tenor: InitVar[Optional[str]] = None
    index_tenor_months: int = 6
    floating_spread: float = 0.0
    #: TraderX fork (R3): an overnight index (`engine.instruments.schedules.OVERNIGHT_INDICES`,
    #: e.g. "USD-SOFR") makes the swap an OIS on its conventions, compounded in arrears and
    #: paying every `index_tenor_months`; None is the generic term index.
    overnight_index: Optional[str] = None
    #: Coupon accrual day count, a property of the booking. Names outside
    #: `SUPPORTED_ACCRUAL_DAY_COUNTS` are refused. Not the simulation time axis (always
    #: ACT/365; see `engine.instruments.schedules`).
    accrual_day_count: str = DEFAULT_ACCRUAL_DAY_COUNT
    fixings: Dict[ORE.Date, float] = field(default_factory=dict)
    currency: str = "USD"
    trade_id: str = field(kw_only=True)
    evaluation_date: ORE.Date = field(kw_only=True)
    #: The product name precision overrides are keyed by (`engine.precision.Precision.by_product`).
    product: ClassVar[str] = "swap"

    def __post_init__(self, swap_tenor: Optional[str]) -> None:
        _validate_identity(self.trade_id, self.evaluation_date)
        _validate_common_fields(self.notional, self.fixed_rate, self.evaluation_date)
        book_swap_dates(self, swap_tenor)
        refuse_in_progress_overnight(self)
        validate_fixings(self.fixings)
        # Validate here, where the trade is identifiable, not later inside ORE.
        resolve_accrual_day_count(self.accrual_day_count)
        index = overnight_index(self.overnight_index, self.currency)
        if index is None and self.accrual_day_count == "ACT/360":
            raise UnsupportedDayCountError(
                "accrual day count 'ACT/360' is an overnight index's: name it in overnight_index (e.g. 'USD-SOFR'). "
                "A term-index swap accruing ACT/360 against the ACT/365 generic index is refused rather than priced")
        if index is not None and self.accrual_day_count not in (index.day_count, DEFAULT_ACCRUAL_DAY_COUNT):
            raise ValueError(f"accrual_day_count {self.accrual_day_count!r} contradicts overnight_index "
                             f"{self.overnight_index!r}, which accrues {index.day_count}")


def underlying_swap(cfg: SwapConfig) -> ORE.VanillaSwap:
    """The ORE trade (see `engine.instruments.schedules.build_vanilla_swap`)."""
    return build_vanilla_swap(
        notional=cfg.notional, fixed_rate=cfg.fixed_rate, payer=cfg.payer,
        effective_date=cfg.effective_date, maturity_date=cfg.maturity_date,
        index_tenor_months=cfg.index_tenor_months, floating_spread=cfg.floating_spread,
        accrual_day_count=cfg.accrual_day_count, overnight_index=overnight_index(cfg.overnight_index, cfg.currency),
    )
