"""
W1.2 -- the bill pricer. The first thing in this package that returns a
number instead of a refusal.

A Treasury bill is a single cashflow: face redeemed at maturity, nothing in
between. Its present value is

    NPV = signed_face x redemptionFraction x P(valuationDate, maturityDate)

and that is the whole model. No coupon schedule, no Monte Carlo, no
calibration, no optionality. This is deliberate ordering (plan §W1.2): the
bill exists to prove the *transport* -- bundle -> terms -> normalize ->
conventions -> price -> identified result -- carries a real number end to
end, while the pricing math is simple enough that any discrepancy is
unambiguously a plumbing bug rather than a modelling one.

---

**What this module will and will not do.**

It prices a zero-coupon, bullet-redemption instrument whose terms say so.
It does **not**:

- infer that something is a bill from a blank accrual column or a zero
  coupon in the CSV (`couponFrequency: NONE` in the *terms* is the only
  thing it keys on -- same rule as `engine.integration.normalize`);
- price a matured instrument, or one maturing on the valuation date;
- accept a curve it was not explicitly handed.

Each of those returns a refusal naming the reason. None of them falls back
to a default.

---

**Day count is a policy of this boundary, stated here rather than assumed.**

Discounting uses **ACT/365 Fixed**, matching the engine's simulation time
axis (`TIME_AXIS_DAY_COUNTER`) and the convention the W0.4 allowlist already
accepts. This is the *discounting* convention, and it is distinct from an
instrument's *accrual* convention -- the distinction W1.1 introduced. A bill
has no accrual to convert (`dayCount: NOT_APPLICABLE` in the fixture terms),
so only the discounting choice applies here, and a note (W1.3) will need the
accrual half that this instrument does not exercise.

**The assumed profiles are continuously compounded** (`flat-3pct-v1` is
"Flat 3% continuously-compounded zero curve"), so the discount factor is
`exp(-r*t)`. Using a simple or annually-compounded convention against the
same profile would silently shift every price -- at 3% over six months the
gap is ~$3.6 per $100k face, small enough to look like rounding and large
enough to be wrong.
"""
import math
from dataclasses import dataclass
from typing import Optional

import ORE

from engine.integration.market_inputs import AssumedProfile, CurveProvenance
from engine.integration.terms import TermsEntry

#: Reason codes. Each names a condition this pricer refuses rather than
#: approximates.
NOT_A_BILL = "NOT_A_BILL"
INSTRUMENT_MATURED = "INSTRUMENT_MATURED"
TERMS_INCOMPLETE = "TERMS_INCOMPLETE"

#: Discounting day count for this boundary -- see the module docstring.
#: ACT/365 Fixed, consistent with the engine's simulation time axis.
DISCOUNT_DAY_COUNT = ORE.Actual365Fixed()

#: The pricing method, echoed in the result so a consumer never has to infer
#: it from the shape of the payload.
METHOD = "discounted-cashflow"


class BillPricingError(Exception):
    """A bill could not be priced. Carries a reason code and a detail,
    shaped for a `CalculationOutcome` refusal rather than a stack trace --
    one unpriceable row must not fail the other 200."""

    def __init__(self, reason: str, detail: str):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


@dataclass(frozen=True)
class BillPrice:
    """One bill's present value and everything needed to check it.

    The intermediate values are carried deliberately: a consumer
    reconciling against its own model needs the discount factor and year
    fraction this engine actually used, not just the final number. A bare
    NPV is unreconcilable -- if it disagrees, nothing says whether the
    curve, the day count, or the face amount was the cause.
    """
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
    """Whether the *terms* describe a zero-coupon bullet instrument.

    Keyed on `couponFrequency: NONE`, exactly as
    `engine.integration.normalize._is_zero_coupon` is -- never on a zero
    `coupon` column or a blank accrual field. Those are the CSV's own
    shorthand and are ambiguous; the terms are the statement.

    A non-empty `schedule` disqualifies the row even if the frequency says
    NONE: the two would be contradicting each other, and this pricer is
    not the place to decide which one wins.
    """
    if entry is None:
        return False
    terms = entry.terms
    if str(terms.get("couponFrequency", "")).upper() != "NONE":
        return False
    return not terms.get("schedule")


def _parse_date(raw: Optional[str], field: str) -> ORE.Date:
    """Parses an ISO `YYYY-MM-DD` terms date into an `ORE.Date`."""
    if not raw:
        raise BillPricingError(
            TERMS_INCOMPLETE,
            f"{field} is absent from the instrument terms, so the cashflow "
            f"date is unknown. A bill cannot be priced without its maturity.",
        )
    try:
        year, month, day = (int(part) for part in str(raw).split("-"))
        return ORE.Date(day, month, year)
    except (ValueError, TypeError) as exc:
        raise BillPricingError(
            TERMS_INCOMPLETE, f"{field}={raw!r} is not an ISO YYYY-MM-DD date",
        ) from exc


def _redemption_fraction(entry: TermsEntry) -> float:
    """`redemptionFraction` from the terms, defaulting to par.

    Absent means par (1.0) -- the overwhelmingly common case, and the
    fixture states it explicitly anyway. A *present but unparseable* value
    raises instead, because that is a malformed artifact rather than an
    omission.
    """
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
    """Prices one bill position against `profile`'s curve.

    `signed_face_amount` carries the position's sign, so a short position
    returns a negative NPV in one step -- there is no separate `sign()`
    factor to apply, and applying one would make a short position positive
    (the double-sign bug TraderX flagged in their v3 §2).

    Raises `BillPricingError` for anything it will not price: a
    coupon-bearing instrument, a matured one, or incomplete terms. Never
    substitutes a default maturity, curve, or redemption.
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
    # Continuously compounded, matching the assumed profile's own stated
    # convention -- see the module docstring.
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
    """`ORE.Date` -> ISO `YYYY-MM-DD`, for echoing dates back in the same
    format the terms artifact used."""
    return f"{date.year():04d}-{date.month():02d}-{date.dayOfMonth():02d}"
