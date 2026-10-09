"""
Treasury bill pricer: a single cashflow, face redeemed at maturity.

    NPV = signed_face x redemptionFraction x exp(-r x t),   t = ACT/365(valuation, maturity)

`r` is the assumed profile's flat, continuously compounded rate. ACT/365 Fixed is this
boundary's discounting convention (the simulation time axis), separate from any accrual
convention; a bill has none.

Refused, never defaulted: a row whose terms do not say zero-coupon (`couponFrequency: NONE`
with no schedule; never inferred from the CSV), a matured bill (maturity on or before the
valuation date), and incomplete terms.
"""
import math
from dataclasses import dataclass
from typing import Optional

import ORE

from engine.traderx.market_inputs import AssumedProfile, CurveProvenance
from engine.traderx.terms import TermsEntry

#: Reason codes for refusals.
NOT_A_BILL = "NOT_A_BILL"
INSTRUMENT_MATURED = "INSTRUMENT_MATURED"
TERMS_INCOMPLETE = "TERMS_INCOMPLETE"

#: Discounting day count (see the module docstring).
DISCOUNT_DAY_COUNT = ORE.Actual365Fixed()

#: Pricing method, echoed in the result.
METHOD = "discounted-cashflow"


class BillPricingError(Exception):
    """A bill could not be priced: a reason code and detail, reported as that row's
    refusal so one row cannot fail the bundle."""

    def __init__(self, reason: str, detail: str):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


@dataclass(frozen=True)
class BillPrice:
    """One bill's present value, with the discount factor and year fraction used, so a
    consumer can reconcile it."""
    npv: float
    signed_face_amount: float
    redemption_fraction: float
    discount_factor: float
    year_fraction: float
    maturity_date: str
    valuation_date: str
    curve_provenance: CurveProvenance
    day_count: str = "ACT/365 (Fixed)"
    method: str = METHOD

    def to_payload(self) -> dict:
        """The `CalculationOutcome.ok` payload for this price."""
        return {
            "method": self.method,
            "signedFaceAmount": self.signed_face_amount,
            "redemptionFraction": self.redemption_fraction,
            "discountFactor": self.discount_factor,
            "yearFraction": self.year_fraction,
            "dayCount": self.day_count,
            "maturityDate": self.maturity_date,
            "valuationDate": self.valuation_date,
            "curveProvenance": self.curve_provenance.to_dict(),
        }


def is_bill(entry: Optional[TermsEntry]) -> bool:
    """Whether the terms describe a zero-coupon bullet instrument: `couponFrequency: NONE`
    and no `schedule` (a schedule would contradict the frequency)."""
    if entry is None:
        return False
    terms = entry.terms
    if str(terms.get("couponFrequency", "")).upper() != "NONE":
        return False
    return not terms.get("schedule")


def _parse_date(raw: Optional[str], field: str) -> ORE.Date:
    """ISO `YYYY-MM-DD` terms date -> `ORE.Date`."""
    if not raw:
        raise BillPricingError(
            TERMS_INCOMPLETE,
            f"{field} is absent from the instrument terms, so the cashflow "
            f"date is unknown. A bill cannot be priced without its maturity.",
        )
    try:
        year, month, day = (int(part) for part in str(raw).split("-"))
        return ORE.Date(day, month, year)
    except (ValueError, TypeError, RuntimeError) as exc:
        # ORE.Date raises RuntimeError for a well-formed but impossible date (2025-02-30);
        # catching it keeps one bad row from failing the whole bundle
        # (tests/test_traderx_bill.py::TestImpossibleCalendarDates).
        raise BillPricingError(
            TERMS_INCOMPLETE, f"{field}={raw!r} is not a valid ISO YYYY-MM-DD date ({exc})",
        ) from exc


def _redemption_fraction(entry: TermsEntry) -> float:
    """`redemptionFraction` from the terms: absent means par (1.0); present but
    unparseable raises."""
    raw = entry.terms.get("redemptionFraction")
    if raw is None:
        return 1.0
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise BillPricingError(
            TERMS_INCOMPLETE, f"redemptionFraction={raw!r} is not a number",
        ) from exc


def price_bill(
    entry: TermsEntry,
    signed_face_amount: float,
    valuation_date: ORE.Date,
    profile: AssumedProfile,
) -> BillPrice:
    """Price one bill position against `profile`'s flat curve.

    `signed_face_amount` carries the sign (a short gives a negative NPV); do not apply
    another. Raises `BillPricingError` rather than defaulting anything.
    """
    if not is_bill(entry):
        raise BillPricingError(
            NOT_A_BILL,
            "instrument terms do not describe a zero-coupon bullet "
            "instrument (couponFrequency is not NONE, or a coupon schedule "
            "is present), so the single-cashflow bill model does not apply.",
        )

    maturity = _parse_date(entry.terms.get("maturityDate"), "maturityDate")

    if maturity <= valuation_date:
        raise BillPricingError(
            INSTRUMENT_MATURED,
            f"maturityDate {_iso(maturity)} is not after the valuation date "
            f"{_iso(valuation_date)}. A matured or same-day-maturing bill has "
            f"no remaining cashflow to discount; its value is a settlement "
            f"question, not a pricing one.",
        )

    redemption = _redemption_fraction(entry)
    year_fraction = DISCOUNT_DAY_COUNT.yearFraction(valuation_date, maturity)
    # Continuously compounded, as the assumed profile states.
    discount_factor = math.exp(-profile.flat_rate * year_fraction)

    return BillPrice(
        npv=signed_face_amount * redemption * discount_factor,
        signed_face_amount=signed_face_amount,
        redemption_fraction=redemption,
        discount_factor=discount_factor,
        year_fraction=year_fraction,
        maturity_date=_iso(maturity),
        valuation_date=_iso(valuation_date),
        curve_provenance=profile.provenance(),
    )


def _iso(date: ORE.Date) -> str:
    """`ORE.Date` -> ISO `YYYY-MM-DD`."""
    return f"{date.year():04d}-{date.month():02d}-{date.dayOfMonth():02d}"
