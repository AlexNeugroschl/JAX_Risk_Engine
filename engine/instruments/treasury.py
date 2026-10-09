"""
Treasury bills and notes: the trade and its remaining cashflows.

A bond names its currency; it carries no curve. It is priced by discounting every remaining
cashflow on its currency's discount curve, today and on every simulated path
(`engine.pricing.cube.bond_legs`): ORE's `DiscountingRiskyBondEngine` with no credit
curve and no security spread, which is how a Treasury without credit is set up (plan V-9).
NPV is dirty. The coupon schedule is supplied, not generated.

The TraderX path prices TraderX's bills and notes with its own pricers
(`engine.traderx.bill`, `engine.traderx.note`).
"""
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Sequence, Tuple

import ORE

from engine.instruments._validation import _validate_identity
from engine.market_data.day_counts import UnsupportedDayCountError, resolve_accrual_day_count


class BondPricingError(Exception):
    """A bond could not be priced. Separate from `ValueError` so pricing refusals are
    distinguishable from malformed requests."""


@dataclass(frozen=True)
class CouponPeriod:
    """One coupon period, supplied rather than generated (a regenerated schedule that
    disagreed with the booked one would reprice every coupon)."""
    start_date: ORE.Date
    end_date: ORE.Date
    payment_date: Optional[ORE.Date] = None

    def payment(self) -> ORE.Date:
        return self.payment_date if self.payment_date is not None else self.end_date


@dataclass(frozen=True)
class BondConfig:
    """A Treasury bill or note, discounted on its currency's curve.

    A bill is `coupon_schedule=()` with `coupon_rate=0.0`.

    `face_amount` is signed: a short position has a negative face and a negative NPV. Do
    not apply a separate sign on top.

    `trade_id` and `evaluation_date` are required keywords, as on every trade config.
    """
    face_amount: float
    maturity_date: ORE.Date
    #: Annual coupon rate as a decimal (0.04 == 4%), not a percent.
    coupon_rate: float = 0.0
    coupon_schedule: Tuple[CouponPeriod, ...] = ()
    redemption_fraction: float = 1.0
    #: Coupon accrual day count; refused if unsupported. Ignored without coupons.
    accrual_day_count: str = "ACT/ACT (ICMA)"
    currency: str = "USD"
    trade_id: str = field(kw_only=True)
    evaluation_date: ORE.Date = field(kw_only=True)
    #: The product name precision overrides are keyed by (`engine.precision.Precision.by_product`).
    product: ClassVar[str] = "bond"

    def __post_init__(self):
        _validate_identity(self.trade_id, self.evaluation_date)
        if self.maturity_date <= self.evaluation_date:
            raise BondPricingError(
                f"maturity_date {_iso(self.maturity_date)} is not after "
                f"evaluation_date {_iso(self.evaluation_date)}. A matured bond has "
                f"no remaining cashflow to discount; its value is a settlement "
                f"question, not a pricing one."
            )
        if self.redemption_fraction < 0:
            raise BondPricingError(
                f"redemption_fraction={self.redemption_fraction} is negative"
            )
        if self.coupon_schedule and self.coupon_rate == 0.0:
            raise BondPricingError(
                "coupon_schedule is non-empty but coupon_rate is 0.0. A schedule of "
                "zero coupons is a contradiction: either the instrument is a bill "
                "(empty schedule) or it pays a coupon. Refused rather than priced "
                "as a bill, because which of the two was meant changes the price."
            )
        if self.coupon_rate != 0.0 and not self.coupon_schedule:
            raise BondPricingError(
                f"coupon_rate={self.coupon_rate} is non-zero but no coupon_schedule "
                f"was supplied. A schedule is not derived from a rate here -- a "
                f"generated schedule that disagreed with the booked one would "
                f"silently reprice every coupon."
            )
        if self.coupon_schedule:
            # Resolve now so an unsupported convention fails at construction.
            try:
                resolve_accrual_day_count(self.accrual_day_count)
            except UnsupportedDayCountError as exc:
                raise BondPricingError(str(exc)) from None
            _validate_schedule(self.coupon_schedule, self.maturity_date)

    @property
    def is_bill(self) -> bool:
        """Whether this is the zero-coupon single-cashflow case."""
        return not self.coupon_schedule


def _validate_schedule(schedule: Sequence[CouponPeriod], maturity: ORE.Date) -> None:
    """Structure and contiguity of an explicit coupon schedule, as
    `engine.traderx.note._parse_schedule` checks it. Gaps and overlaps are refused."""
    for i, period in enumerate(schedule):
        if period.end_date <= period.start_date:
            raise BondPricingError(
                f"coupon_schedule[{i}] ends {_iso(period.end_date)} on or before it "
                f"starts {_iso(period.start_date)}; a period with no length cannot accrue."
            )
        if period.payment() < period.end_date:
            raise BondPricingError(
                f"coupon_schedule[{i}] pays {_iso(period.payment())} before its "
                f"accrual ends {_iso(period.end_date)}."
            )
    for i in range(1, len(schedule)):
        if schedule[i].start_date != schedule[i - 1].end_date:
            raise BondPricingError(
                f"coupon_schedule[{i}] starts {_iso(schedule[i].start_date)} but "
                f"coupon_schedule[{i-1}] ended {_iso(schedule[i-1].end_date)}. A gap "
                f"or overlap between coupon periods means the schedule does not "
                f"describe one continuous instrument, and is refused rather than bridged."
            )
    if schedule and schedule[-1].end_date != maturity:
        raise BondPricingError(
            f"the coupon schedule ends {_iso(schedule[-1].end_date)} but maturity_date "
            f"is {_iso(maturity)}. The redemption and the final coupon must fall on the "
            f"same date; a disagreement means the two describe different instruments."
        )


def _iso(date: ORE.Date) -> str:
    return f"{date.year():04d}-{date.month():02d}-{date.dayOfMonth():02d}"


def accrued_interest(cfg: BondConfig) -> float:
    """Accrued interest in currency (position-signed) from the coupon schedule; 0 for a
    bill or before the first period.

    This is the recomputed-schedule figure only. The integration boundary also reconciles
    against the exporter's published accrued fraction (`engine.traderx.note`); a
    direct caller has no exporter to reconcile against.
    """
    if not cfg.coupon_schedule:
        return 0.0
    day_count = resolve_accrual_day_count(cfg.accrual_day_count)
    for period in cfg.coupon_schedule:
        if period.start_date <= cfg.evaluation_date < period.end_date:
            fraction = cfg.coupon_rate * day_count.yearFraction(
                period.start_date, cfg.evaluation_date,
                period.start_date, period.end_date,
            )
            return fraction * cfg.face_amount
    return 0.0


def _remaining_cashflows(cfg: BondConfig) -> Tuple[Tuple[ORE.Date, float], ...]:
    """Remaining cashflows as `(payment_date, amount per unit face)`: coupons in schedule
    order, then the redemption. A coupon paid on or before the evaluation date is
    excluded. `engine.pricing.cube.bond_legs` discounts them."""
    flows = []
    if cfg.coupon_schedule:
        day_count = resolve_accrual_day_count(cfg.accrual_day_count)
        for period in cfg.coupon_schedule:
            payment = period.payment()
            if payment <= cfg.evaluation_date:
                continue
            accrual_fraction = day_count.yearFraction(
                period.start_date, period.end_date,
                period.start_date, period.end_date,
            )
            flows.append((payment, cfg.coupon_rate * accrual_fraction))
    flows.append((cfg.maturity_date, cfg.redemption_fraction))
    return tuple(flows)
