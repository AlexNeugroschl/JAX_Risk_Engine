"""
A European swaption: the trade and its ORE underlying (`MakeVanillaSwap`).

The trade names its currency, index and settlement; it carries no curve or model. Its engine
is the run configuration's (`PricingConfig.european`): ORE's default Bachelier engine on the
market's normal swaption volatilities (`engine.valuation.european`), or Jamshidian on the
pricing configuration's Hull-White model (`engine.valuation.jamshidian`). Either engine prices
it today and on every simulated path; in a simulation it is wrapped as ORE wraps it
(`engine.valuation.options`), so an exercised physical option becomes its swap.

A config holds its booked `exercise_date` and underlying dates; on or after the exercise date
the option is expired and worth 0 (ORE's `Instrument::isExpired`).
"""
from dataclasses import InitVar, dataclass, field
from typing import Optional

import ORE

from engine.instruments._validation import _validate_common_fields, _validate_identity, _validate_settlement
from engine.models.ore_builders import SWAP_CALENDAR, book_swap_dates, build_vanilla_swap, is_live


@dataclass
class SwaptionConfig:
    """
    One European swaption: the option to enter a vanilla swap on `exercise_date`.

    trade_id / evaluation_date: as in `SwapConfig` (required, keyword only).
    currency: the trade's currency (its curves and swaption volatilities).
    exercise_date / effective_date / maturity_date: the booked expiry and the underlying's
        schedule. They define the trade; `evaluation_date` only sets when it is priced.
    index_tenor_months / floating_spread: as in `SwapConfig`. The Jamshidian engine refuses
        a non-zero spread (QuantLib's does).
    settlement: Physical (ORE's default) or Cash: what the option becomes on exercise in a
        simulation (ORE's `OptionWrapper`), and for cash ORE's `ParYieldCurve` annuity (the
        Jamshidian engine refuses it, as QuantLib's does).

    Booking by tenor instead (resolved once, on `evaluation_date`, not stored):
      * swap_tenor: the underlying's length, e.g. "5Y";
      * forward_start: ORE.Period by which the underlying starts after spot;
      * exercise_lag_days: business days from `evaluation_date + forward_start` to the
        exercise date (default 2, so without a forward start the exercise date is the
        underlying's spot start).
    """
    notional: float
    fixed_rate: float
    payer: bool
    exercise_date: Optional[ORE.Date] = None
    effective_date: Optional[ORE.Date] = None
    maturity_date: Optional[ORE.Date] = None
    swap_tenor: InitVar[Optional[str]] = None
    index_tenor_months: int = 6
    floating_spread: float = 0.0
    forward_start: InitVar[Optional[ORE.Period]] = None
    exercise_lag_days: InitVar[Optional[int]] = None
    currency: str = "USD"
    settlement: str = "Physical"
    trade_id: str = field(kw_only=True)
    evaluation_date: ORE.Date = field(kw_only=True)

    def __post_init__(self, swap_tenor, forward_start, exercise_lag_days) -> None:
        _validate_identity(self.trade_id, self.evaluation_date)
        _validate_common_fields(self.notional, self.fixed_rate, self.evaluation_date)
        if swap_tenor is not None:
            if self.exercise_date is not None:
                raise ValueError("give either swap_tenor or exercise_date/effective_date/maturity_date, not both")
            self.exercise_date = resolve_exercise_date(
                self.evaluation_date, forward_start, 2 if exercise_lag_days is None else exercise_lag_days)
        elif exercise_lag_days is not None:
            raise ValueError("exercise_lag_days is only meaningful with swap_tenor")
        book_swap_dates(self, swap_tenor, forward_start)
        if not isinstance(self.exercise_date, ORE.Date):
            raise TypeError(f"exercise_date must be an ORE.Date; got {self.exercise_date!r}")
        if not self.exercise_date < self.maturity_date:
            raise ValueError(
                f"exercise_date ({self.exercise_date}) must be before maturity_date ({self.maturity_date})")
        _validate_settlement(self.settlement)

    def is_expired(self) -> bool:
        """Exercise date on or before the evaluation date (ORE's `isExpired`)."""
        return not is_live(self.exercise_date, self.evaluation_date)


def resolve_exercise_date(trade_date: ORE.Date, forward_start, exercise_lag_days: int) -> ORE.Date:
    """Exercise date of a swaption booked by tenor: `exercise_lag_days` business days
    after `trade_date + forward_start`. Not measured back from the accrual start, which
    already includes the spot lag."""
    forward_start = forward_start if forward_start is not None else ORE.Period(0, ORE.Days)
    forward_start_date = SWAP_CALENDAR.advance(trade_date, forward_start)
    return SWAP_CALENDAR.advance(forward_start_date, exercise_lag_days, ORE.Days)


def _build_ore_swap(cfg: SwaptionConfig) -> ORE.VanillaSwap:
    """The ORE underlying swap (see `engine.models.ore_builders.build_vanilla_swap`)."""
    return build_vanilla_swap(
        notional=cfg.notional, fixed_rate=cfg.fixed_rate, payer=cfg.payer,
        effective_date=cfg.effective_date, maturity_date=cfg.maturity_date,
        index_tenor_months=cfg.index_tenor_months, floating_spread=cfg.floating_spread,
    )
