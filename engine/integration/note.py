"""
W1.3 -- the note pricer. The first instrument at this boundary with a
*schedule*.

A Treasury note is a strip of fixed coupons plus a bullet redemption:

    NPV = signed_face x [ sum_i c_i x P(t_i) + redemptionFraction x P(T) ]

where `c_i` is the coupon accrual fraction for period `i` under the
instrument's **own** accrual day count, and `P(.)` is the discount factor
off the explicitly requested curve.

This is the half of W1 the bill could not exercise. Three things are
genuinely new here, and each is a place a plausible wrong answer lives:

1. **A coupon schedule**, taken from the terms rather than derived from a
   tenor string. The terms artifact enumerates every period explicitly
   (`startDate`/`endDate`/`paymentDate`), and the periods are used as
   supplied -- this module never regenerates a schedule by stepping back
   from maturity, because a regenerated schedule that disagreed with the
   exporter's would silently reprice every coupon.

2. **A per-instrument accrual day count** -- the half of W1.1 the bill
   never used. The delivered note is `ACT/ACT (ICMA)`, which accrues
   against the *actual coupon period* rather than a calendar year, so a
   semiannual period is exactly 0.5 whatever its day length. Pricing it as
   ACT/365 instead would shift every coupon: on this fixture the June 2025
   period is 182 days, so ACT/365 gives 0.4986 against ICMA's 0.5 -- a
   0.27% error on each coupon, invisible in isolation and wrong in
   aggregate. The day count is **resolved from the terms and refused if
   unsupported**, never defaulted (`resolve_accrual_day_count`, W1.1).

3. **Accrued interest, from two independent paths that must agree.** See
   below -- this is the substantive judgement in the module.

---

**Accrued interest: the export is authoritative, the schedule is the
check.**

There are two ways to know this note's accrued interest, and the plan (§1,
"Accrued source label") settles which one wins:

  - `exported-fraction` -- `accruedInterestFraction` from the positions
    extract, rounded HALF_EVEN at 6 decimals by the exporter. **This is the
    published number and this is what the result reports**, because it is
    the value TraderX's own books carry and the one a reconciliation is
    against.
  - `recomputed-schedule` -- recomputed here from the coupon schedule and
    the accrual day count, unrounded.

`NotePrice` carries **both**, plus their difference, and the label saying
which one the reported value is. Reporting only one would lose the check;
reporting the recomputed one as *the* value would publish a number
TraderX's books do not contain -- exactly the correction owed in plan §1
(`+1,857.14` recomputed vs `1,857.10` exported on this fixture).

**They must never be compared as exact equals.** The exporter's own
rounding makes them differ by construction. The agreed tolerance is

    round(0.5 x 10^-fractionDecimals x |face|, 2) + 0.01

-- derived from the stated `accrualBasis` decimals, never a fixed
constant. `accrual_mismatch_tolerance` below implements exactly that, and
a difference beyond it is a **refusal**, not a warning: the two paths
disagreeing means the schedule this engine priced is not the schedule the
exporter accrued against, and every coupon it discounted is then suspect.

---

**Conventions, stated rather than assumed.** Discounting is ACT/365 Fixed
continuously compounded -- identical to the bill (`engine.integration.bill`)
and to the assumed profiles' own stated convention. **Accrual is the
instrument's**, from `terms.dayCount`. Those are two different day counts
doing two different jobs in the same formula, which is precisely the
distinction W1.1 exists to keep straight.

**Settlement is the session date.** The fixture states `settlementDays: 0`,
and TraderX's preamble says accrual "runs to sessionDate ITSELF, not to a
T+1 settlement date, because every other column here is as-of sessionDate
and this system carries no holiday calendar". A non-zero `settlementDays`
is **refused**, not silently ignored: it would put the exporter's accrual
and this engine's discounting on two different dates.
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import ORE

from engine.integration.bill import DISCOUNT_DAY_COUNT, _iso
from engine.integration.market_inputs import AssumedProfile, CurveProvenance
from engine.integration.terms import TermsEntry
from engine.day_count import (
    UnsupportedDayCountError,
    resolve_accrual_day_count,
)

#: Reason codes. Each names a condition this pricer refuses rather than
#: approximates.
NOT_A_NOTE = "NOT_A_NOTE"
INSTRUMENT_MATURED = "INSTRUMENT_MATURED"
TERMS_INCOMPLETE = "TERMS_INCOMPLETE"
DAY_COUNT_NOT_SUPPORTED = "DAY_COUNT_NOT_SUPPORTED"
SCHEDULE_INCONSISTENT = "SCHEDULE_INCONSISTENT"
ACCRUAL_MISMATCH = "ACCRUAL_MISMATCH"
SETTLEMENT_CONVENTION_NOT_SUPPORTED = "SETTLEMENT_CONVENTION_NOT_SUPPORTED"

#: The pricing method, echoed in the result so a consumer never has to
#: infer it from the shape of the payload.
METHOD = "discounted-cashflow"

#: `accrualSource` labels (plan §1). The label is always explicit, so a
#: consumer never has to guess which of the two paths produced the number
#: it is reconciling against.
ACCRUAL_EXPORTED = "exported-fraction"
ACCRUAL_RECOMPUTED = "recomputed-schedule"

#: Decimals the exporter rounds `accruedInterestFraction` to, stated in the
#: positions preamble ("HALF_EVEN at 6 decimals"). Used to *derive* the
#: reconciliation tolerance rather than to hard-code one -- see
#: `accrual_mismatch_tolerance`.
DEFAULT_FRACTION_DECIMALS = 6

#: Settlement lag this pricer implements. Zero only: the exporter accrues
#: to the session date itself (see the module docstring), so a booking
#: naming a settlement lag is refused rather than priced on a date the
#: accrued interest does not correspond to.
SUPPORTED_SETTLEMENT_DAYS = (0,)

#: Per-pillar bump for `rateSensitivity`, in absolute rate terms. 1bp,
#: stated as an explicit numeric in the payload (plan §1, "Sensitivities")
#: rather than implied by a field name.
RATE_BUMP = 1e-4

#: How `rateSensitivity` is produced. A bumped revaluation, not AD: this
#: pricer is closed-form arithmetic over a flat profile, and saying
#: `ad-first-order` would claim machinery that is not there.
SENSITIVITY_METHOD = "bumped-revaluation"


class NotePricingError(Exception):
    """A note could not be priced. Carries a reason code and a detail,
    shaped for a `CalculationOutcome` refusal rather than a stack trace --
    one unpriceable row must not fail the other 200."""

    def __init__(self, reason: str, detail: str):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


@dataclass(frozen=True)
class CouponFlow:
    """One scheduled coupon, with every input to its own value.

    Carried in full deliberately. A consumer reconciling a total that
    disagrees needs to see *which* coupon diverged and whether it was the
    accrual fraction or the discount factor -- a bare list of amounts
    cannot answer that, and the answer is what turns a mismatch into a
    fixable bug.
    """
    start_date: str
    end_date: str
    payment_date: str
    #: Period accrual under the *instrument's* day count (0.5 for a
    #: semiannual ICMA period, whatever its day length).
    accrual_fraction: float
    #: `coupon_rate x accrual_fraction`, per unit face.
    amount_per_unit_face: float
    #: Discount factor at the payment date, off the requested curve.
    discount_factor: float
    #: Time to payment under the *discounting* day count (ACT/365).
    year_fraction: float

    def to_dict(self) -> Dict:
        return {
            "startDate": self.start_date,
            "endDate": self.end_date,
            "paymentDate": self.payment_date,
            "accrualFraction": self.accrual_fraction,
            "amountPerUnitFace": self.amount_per_unit_face,
            "discountFactor": self.discount_factor,
            "yearFraction": self.year_fraction,
        }


@dataclass(frozen=True)
class AccruedReconciliation:
    """The two accrued-interest paths and their agreement.

    Exists as its own type because the pair is the point: either number
    alone is unreconcilable, and asserting they are equal is wrong by
    construction (the exporter rounds). See the module docstring.
    """
    #: The published value, per unit face: the exporter's rounded fraction.
    exported_fraction: Optional[float]
    #: Recomputed here from the schedule and the accrual day count,
    #: unrounded, per unit face.
    recomputed_fraction: float
    #: Which of the two `NotePrice.accrued_interest` reports.
    source: str
    #: Signed currency difference between the two monetary paths.
    difference: float
    #: The derived tolerance the difference was checked against.
    tolerance: float

    @property
    def agrees(self) -> bool:
        return abs(self.difference) <= self.tolerance

    def to_dict(self) -> Dict:
        return {
            "accrualSource": self.source,
            "exportedFraction": self.exported_fraction,
            "recomputedFraction": self.recomputed_fraction,
            "difference": self.difference,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True)
class NotePrice:
    """One note position's value and everything needed to check it.

    `dirty_npv` is the answer: the present value of every remaining
    cashflow, which is what a discounted-cashflow valuation produces and
    what `npv` reports. `clean_npv` is `dirty_npv - accrued_interest`,
    carried because the exported `closingMark` is a *clean* price
    (`priceBasis: clean-fraction-of-par`) and a consumer reconciling
    against the extract compares like with like.
    """
    #: Present value of all remaining cashflows, position-signed.
    dirty_npv: float
    #: `dirty_npv - accrued_interest`, position-signed.
    clean_npv: float
    #: Position-signed accrued interest in currency, from `accrual.source`.
    accrued_interest: float
    signed_face_amount: float
    coupon_rate: float
    redemption_fraction: float
    #: Present value of the redemption, per unit face.
    redemption_pv_per_unit_face: float
    #: Discount factor at maturity.
    redemption_discount_factor: float
    coupons: Tuple[CouponFlow, ...]
    accrual: AccruedReconciliation
    accrual_day_count: str
    maturity_date: str
    valuation_date: str
    curve_provenance: CurveProvenance
    discount_day_count: str = "ACT/365 (Fixed)"
    method: str = METHOD

    @property
    def npv(self) -> float:
        """The reported `npv`: the dirty (full) present value.

        Named explicitly rather than left implicit. A "bond NPV" that
        silently meant the clean value would be off by the accrued
        interest -- $1,857 on this fixture's $100k -- which is large
        enough to matter and small enough to look like a curve
        difference.
        """
        return self.dirty_npv

    def to_payload(self) -> Dict:
        """The `CalculationOutcome.ok` payload for this price."""
        return {
            "method": self.method,
            "priceType": "dirty",
            "cleanNpv": self.clean_npv,
            "accruedInterest": self.accrued_interest,
            "signedFaceAmount": self.signed_face_amount,
            "couponRate": self.coupon_rate,
            "redemptionFraction": self.redemption_fraction,
            "redemptionPvPerUnitFace": self.redemption_pv_per_unit_face,
            "redemptionDiscountFactor": self.redemption_discount_factor,
            "accrualDayCount": self.accrual_day_count,
            "discountDayCount": self.discount_day_count,
            "maturityDate": self.maturity_date,
            "valuationDate": self.valuation_date,
            "coupons": [c.to_dict() for c in self.coupons],
            "accrualReconciliation": self.accrual.to_dict(),
            "curveProvenance": self.curve_provenance.to_dict(),
        }


def is_note(entry: Optional[TermsEntry]) -> bool:
    """Whether the *terms* describe a coupon-bearing scheduled instrument.

    The exact complement of `engine.integration.bill.is_bill` over
    well-formed terms: a stated coupon frequency other than NONE, **and** a
    non-empty explicit schedule. Both are required -- a frequency with no
    schedule is not something this pricer will fill in for itself, and a
    schedule with `couponFrequency: NONE` is a contradiction this pricer is
    not the place to resolve.

    Keyed on the terms and never on the CSV's `coupon` column, for the same
    reason `engine.integration.normalize` is: the CSV's shorthand is
    ambiguous and the terms are the statement.
    """
    if entry is None:
        return False
    terms = entry.terms
    if str(terms.get("couponFrequency", "")).upper() in ("", "NONE"):
        return False
    return bool(terms.get("schedule"))


def accrual_mismatch_tolerance(
    signed_face_amount: float, fraction_decimals: int = DEFAULT_FRACTION_DECIMALS,
) -> float:
    """The agreed accrued-interest reconciliation tolerance (plan §1).

        0.5 x 10^-fractionDecimals x |face| + 0.01

    **Derived from the stated rounding, never a fixed constant.** The first
    term is the largest error the exporter's own HALF_EVEN rounding at
    `fractionDecimals` can introduce on this face amount; the `+0.01`
    absorbs cent-level rounding in the monetary comparison itself. Scaling
    with face is the whole point -- a constant tolerance is either useless
    on a $1 position or vacuous on a $1bn one.

    **The rounding bound is NOT itself rounded** (TraderX v4/v5). An earlier
    implementation computed `round(rounding_error, 2) + 0.01`, which
    truncates the very quantity it is meant to bound: at 124,000 face the
    true bound is 0.062 and rounding gives 0.06, so the tolerance came out
    0.07 instead of 0.072. That is *tighter* than agreed -- it can reject a
    reconciliation that is within the exporter's own stated rounding error,
    reporting a mismatch where none exists.
    """
    rounding_error = 0.5 * (10.0 ** -fraction_decimals) * abs(signed_face_amount)
    # Deliberately unrounded. See the docstring: rounding the bound makes it
    # narrower than the error it is supposed to admit.
    return rounding_error + 0.01


def _parse_date(raw: Optional[str], field: str) -> ORE.Date:
    """Parses an ISO `YYYY-MM-DD` terms date into an `ORE.Date`.

    **Deliberately not shared with `engine.integration.bill`**, though the
    arithmetic is identical. The bill's parser raises `BillPricingError`,
    and a note raising it would escape `_note_outcomes`'s
    `except NotePricingError` in the pipeline -- propagating out of
    `_build_item` and failing the *whole bundle* on one malformed date,
    when the contract is that one unpriceable row must not cost the other
    200 their results. The duplicated four lines are the cheaper price
    than that coupling; `tests/test_integration_note.py` pins both the
    exception type and the bundle-level consequence.
    """
    if not raw:
        raise NotePricingError(
            TERMS_INCOMPLETE,
            f"{field} is absent from the instrument terms. A coupon-bearing "
            f"note cannot be priced without it, and it will not be inferred.",
        )
    try:
        year, month, day = (int(part) for part in str(raw).split("-"))
        return ORE.Date(day, month, year)
    except (ValueError, TypeError, RuntimeError) as exc:
        # `RuntimeError` is not defensive breadth -- it is the exception
        # `ORE.Date` actually raises for a date that PARSES but cannot
        # exist ("2025-02-30" -> "day outside month (2) day-range [1,28]",
        # "2025-13-01" -> "month 13 outside ... range"). SWIG surfaces
        # QuantLib's C++ `std::runtime_error` that way, so the Python date
        # exceptions alone miss exactly the malformed-but-numeric case.
        # Without it the error escapes `_note_outcomes`' handler in the
        # pipeline and fails the WHOLE bundle on one bad row -- see
        # `tests/test_integration_note.py::TestImpossibleCalendarDates`.
        raise NotePricingError(
            TERMS_INCOMPLETE, f"{field}={raw!r} is not a valid ISO YYYY-MM-DD date ({exc})",
        ) from exc


def _terms_float(entry: TermsEntry, field: str, required: bool = True,
                 default: Optional[float] = None) -> Optional[float]:
    """Reads a numeric terms field. Absent is `default` (or a refusal when
    required); present-but-unparseable is always a refusal -- that is a
    malformed artifact rather than an omission."""
    raw = entry.terms.get(field)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        if required:
            raise NotePricingError(
                TERMS_INCOMPLETE,
                f"{field} is absent from the instrument terms. It is required to "
                f"price a coupon-bearing note and will not be defaulted.",
            )
        return default
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise NotePricingError(
            TERMS_INCOMPLETE, f"{field}={raw!r} is not a number",
        ) from exc


def _resolve_day_count(entry: TermsEntry):
    """The instrument's accrual day count, from its terms.

    **Refused if unsupported, never defaulted** -- the W1.1 rule. A note
    silently priced on ACT/365 when its terms say ACT/ACT (ICMA) returns a
    confident number that is wrong on every coupon.
    """
    name = entry.terms.get("dayCount")
    if not name:
        raise NotePricingError(
            TERMS_INCOMPLETE,
            "dayCount is absent from the instrument terms. A coupon accrual "
            "convention is a property of the booking and is refused rather than "
            "defaulted (plan §W1.1).",
        )
    try:
        return str(name), resolve_accrual_day_count(str(name))
    except UnsupportedDayCountError as exc:
        raise NotePricingError(DAY_COUNT_NOT_SUPPORTED, str(exc)) from None


def _check_settlement(entry: TermsEntry) -> None:
    """Refuses a settlement lag this pricer does not implement.

    See the module docstring: the exporter accrues to the session date
    itself, so a non-zero lag would place the accrued interest and the
    discounting on different dates. Ignoring the field would be exactly the
    silent approximation this boundary refuses.
    """
    raw = entry.terms.get("settlementDays")
    if raw is None:
        return
    try:
        days = int(raw)
    except (TypeError, ValueError) as exc:
        raise NotePricingError(
            TERMS_INCOMPLETE, f"settlementDays={raw!r} is not an integer",
        ) from exc
    if days not in SUPPORTED_SETTLEMENT_DAYS:
        raise NotePricingError(
            SETTLEMENT_CONVENTION_NOT_SUPPORTED,
            f"settlementDays={days} is not implemented; this pricer values as of "
            f"the session date itself (settlementDays in "
            f"{list(SUPPORTED_SETTLEMENT_DAYS)}), matching the accrual basis the "
            f"exporter states. A settlement lag is refused rather than ignored: "
            f"ignoring it would discount on one date and accrue to another.",
        )


def _parse_schedule(entry: TermsEntry) -> Tuple[Tuple[ORE.Date, ORE.Date, ORE.Date], ...]:
    """The explicit coupon schedule from the terms, as supplied.

    Validated for structure and contiguity but **never regenerated**. The
    exporter's schedule is the schedule the exported accrued interest was
    computed against; deriving our own by stepping back from maturity would
    reprice every coupon whenever the two disagreed -- and would do it
    silently, because a plausible schedule produces a plausible price.
    """
    raw_schedule = entry.terms.get("schedule")
    if not raw_schedule:
        raise NotePricingError(
            TERMS_INCOMPLETE,
            "the instrument terms carry no coupon schedule. A schedule is not "
            "derived from the coupon frequency here: a regenerated schedule that "
            "disagreed with the exporter's would silently reprice every coupon.",
        )

    periods: List[Tuple[ORE.Date, ORE.Date, ORE.Date]] = []
    for i, raw in enumerate(raw_schedule):
        if not isinstance(raw, dict):
            raise NotePricingError(
                TERMS_INCOMPLETE,
                f"schedule[{i}] is {type(raw).__name__}, expected an object with "
                f"startDate/endDate/paymentDate",
            )
        start = _parse_date(raw.get("startDate"), f"schedule[{i}].startDate")
        end = _parse_date(raw.get("endDate"), f"schedule[{i}].endDate")
        # `paymentDate` falls back to `endDate` only when genuinely absent
        # -- the fixture states it explicitly, and an unadjusted schedule
        # has them equal. A present-but-unparseable value still raises.
        payment = (
            _parse_date(raw["paymentDate"], f"schedule[{i}].paymentDate")
            if raw.get("paymentDate") else end
        )
        if end <= start:
            raise NotePricingError(
                SCHEDULE_INCONSISTENT,
                f"schedule[{i}] ends {_iso(end)} on or before it starts "
                f"{_iso(start)}; a coupon period with no length cannot accrue.",
            )
        if payment < end:
            raise NotePricingError(
                SCHEDULE_INCONSISTENT,
                f"schedule[{i}] pays {_iso(payment)} before its accrual ends "
                f"{_iso(end)}.",
            )
        periods.append((start, end, payment))

    for i in range(1, len(periods)):
        if periods[i][0] != periods[i - 1][1]:
            raise NotePricingError(
                SCHEDULE_INCONSISTENT,
                f"schedule[{i}] starts {_iso(periods[i][0])} but schedule[{i-1}] "
                f"ended {_iso(periods[i-1][1])}. A gap or overlap between coupon "
                f"periods means the schedule does not describe one continuous "
                f"instrument, and is refused rather than bridged.",
            )
    return tuple(periods)


def _discount(profile: AssumedProfile, valuation: ORE.Date, date: ORE.Date,
              rate_shift: float = 0.0) -> Tuple[float, float]:
    """`(discount_factor, year_fraction)` for one date off the profile.

    Continuously compounded over an ACT/365 year fraction -- identical to
    `engine.integration.bill`, and matching the assumed profiles' own
    stated convention. `rate_shift` is the `rateSensitivity` bump; it is a
    parallel shift because the profile is flat and has no other pillars to
    move independently.
    """
    year_fraction = DISCOUNT_DAY_COUNT.yearFraction(valuation, date)
    return math.exp(-(profile.flat_rate + rate_shift) * year_fraction), year_fraction


def _accrued_fraction_from_schedule(
    periods: Sequence[Tuple[ORE.Date, ORE.Date, ORE.Date]],
    valuation: ORE.Date, coupon_rate: float, day_count,
) -> float:
    """Accrued interest per unit face, recomputed from the schedule.

    The current period is the one containing the valuation date; accrual
    runs from its start to the valuation date under the instrument's own
    day count. Under ACT/ACT (ICMA) the reference period must be passed --
    that is what makes a semiannual period exactly 0.5 and is the whole
    reason the convention exists.

    Returns 0.0 before the first period starts: nothing has accrued yet,
    which is a structural fact and not a missing value.
    """
    for start, end, _payment in periods:
        if start <= valuation < end:
            return coupon_rate * day_count.yearFraction(start, valuation, start, end)
    if periods and valuation < periods[0][0]:
        return 0.0
    # On or after the last accrual end: every coupon has been paid, so
    # nothing is accruing. A matured note is refused before reaching here.
    return 0.0


def _reconcile_accrued(
    exported_fraction: Optional[float], recomputed_fraction: float,
    signed_face_amount: float, fraction_decimals: int,
) -> AccruedReconciliation:
    """Builds the two-path accrued reconciliation, or refuses.

    **Refuses on disagreement rather than warning.** The two paths
    disagreeing beyond the derived tolerance means this engine priced a
    different schedule from the one the exporter accrued against -- so
    every discounted coupon is suspect, not just the accrued figure. A
    warning attached to a published number would invite exactly the
    reconciliation the disagreement says is unsound.
    """
    tolerance = accrual_mismatch_tolerance(signed_face_amount, fraction_decimals)

    if exported_fraction is None:
        # Legitimate: the blank-accrual compatibility fixture. The
        # recomputed value is reported, and the label says so -- it is not
        # passed off as the exporter's.
        return AccruedReconciliation(
            exported_fraction=None, recomputed_fraction=recomputed_fraction,
            source=ACCRUAL_RECOMPUTED, difference=0.0, tolerance=tolerance,
        )

    difference = (exported_fraction - recomputed_fraction) * signed_face_amount
    reconciliation = AccruedReconciliation(
        exported_fraction=exported_fraction, recomputed_fraction=recomputed_fraction,
        source=ACCRUAL_EXPORTED, difference=difference, tolerance=tolerance,
    )
    if not reconciliation.agrees:
        raise NotePricingError(
            ACCRUAL_MISMATCH,
            f"exported accrued interest ({exported_fraction:.10f} of par) and the "
            f"value recomputed from the supplied schedule "
            f"({recomputed_fraction:.10f} of par) differ by "
            f"{difference:.4f} in currency, beyond the agreed tolerance of "
            f"{tolerance:.4f} derived from {fraction_decimals}-decimal rounding "
            f"on a face of {signed_face_amount:.2f}. The schedule this engine "
            f"would discount is therefore not the schedule the export accrued "
            f"against, so every coupon in it is suspect -- refused rather than "
            f"priced with a warning.",
        )
    return reconciliation


def price_note(
    entry: TermsEntry,
    signed_face_amount: float,
    valuation_date: ORE.Date,
    profile: AssumedProfile,
    exported_accrued_fraction: Optional[float] = None,
    fraction_decimals: int = DEFAULT_FRACTION_DECIMALS,
) -> NotePrice:
    """Prices one coupon-bearing Treasury note position against `profile`.

    `signed_face_amount` carries the position's sign, so a short position
    returns a negative NPV in one step -- there is no separate `sign()`
    factor, and applying one would make a short position positive (the
    double-sign bug TraderX flagged in their v3 §2).

    `exported_accrued_fraction` is the extract's `accruedInterestFraction`.
    When supplied it is **the reported accrued interest** and the recomputed
    schedule value becomes a cross-check; when absent (a legitimate state --
    the blank-accrual compatibility fixture) the recomputed value is
    reported instead, and `accrual.source` says so. Either way the label is
    explicit and both numbers travel in the payload.

    Raises `NotePricingError` for anything it will not price. Never
    substitutes a schedule, a day count, a curve, or an accrued value.
    """
    if not is_note(entry):
        raise NotePricingError(
            NOT_A_NOTE,
            "instrument terms do not describe a coupon-bearing scheduled "
            "instrument (couponFrequency is NONE or absent, or no explicit "
            "coupon schedule is supplied), so the fixed-coupon note model does "
            "not apply.",
        )

    maturity = _parse_date(entry.terms.get("maturityDate"), "maturityDate")
    if maturity <= valuation_date:
        raise NotePricingError(
            INSTRUMENT_MATURED,
            f"maturityDate {_iso(maturity)} is not after the valuation date "
            f"{_iso(valuation_date)}. A matured note has no remaining cashflow "
            f"to discount; its value is a settlement question, not a pricing one.",
        )

    _check_settlement(entry)
    day_count_name, day_count = _resolve_day_count(entry)
    periods = _parse_schedule(entry)

    if periods[-1][1] != maturity:
        raise NotePricingError(
            SCHEDULE_INCONSISTENT,
            f"the coupon schedule ends {_iso(periods[-1][1])} but maturityDate is "
            f"{_iso(maturity)}. The redemption and the final coupon must fall on "
            f"the same date; a disagreement means the two terms describe "
            f"different instruments.",
        )

    # `couponRatePercent` is an annual percent (4.0 means 4%), the same unit
    # as the CSV `coupon` column -- see engine.integration.normalize's table.
    coupon_rate = _terms_float(entry, "couponRatePercent") / 100.0
    redemption = _terms_float(entry, "redemptionFraction", required=False, default=1.0)

    coupons: List[CouponFlow] = []
    for start, end, payment in periods:
        if payment <= valuation_date:
            # Already paid. Not this position's cashflow, and including it
            # would double-count what the accrued figure already excludes.
            continue
        accrual_fraction = day_count.yearFraction(start, end, start, end)
        discount_factor, year_fraction = _discount(profile, valuation_date, payment)
        coupons.append(CouponFlow(
            start_date=_iso(start), end_date=_iso(end), payment_date=_iso(payment),
            accrual_fraction=accrual_fraction,
            amount_per_unit_face=coupon_rate * accrual_fraction,
            discount_factor=discount_factor, year_fraction=year_fraction,
        ))

    redemption_df, _ = _discount(profile, valuation_date, maturity)
    dirty_per_unit_face = (
        sum(c.amount_per_unit_face * c.discount_factor for c in coupons)
        + redemption * redemption_df
    )
    dirty_npv = signed_face_amount * dirty_per_unit_face

    recomputed_fraction = _accrued_fraction_from_schedule(
        periods, valuation_date, coupon_rate, day_count,
    )
    accrual = _reconcile_accrued(
        exported_accrued_fraction, recomputed_fraction,
        signed_face_amount, fraction_decimals,
    )
    reported_fraction = (
        accrual.exported_fraction if accrual.source == ACCRUAL_EXPORTED
        else accrual.recomputed_fraction
    )
    accrued_interest = reported_fraction * signed_face_amount

    return NotePrice(
        dirty_npv=dirty_npv,
        clean_npv=dirty_npv - accrued_interest,
        accrued_interest=accrued_interest,
        signed_face_amount=signed_face_amount,
        coupon_rate=coupon_rate,
        redemption_fraction=redemption,
        redemption_pv_per_unit_face=redemption * redemption_df,
        redemption_discount_factor=redemption_df,
        coupons=tuple(coupons),
        accrual=accrual,
        accrual_day_count=day_count_name,
        maturity_date=_iso(maturity),
        valuation_date=_iso(valuation_date),
        curve_provenance=profile.provenance(),
    )


def rate_sensitivity(
    entry: TermsEntry,
    signed_face_amount: float,
    valuation_date: ORE.Date,
    profile: AssumedProfile,
    exported_accrued_fraction: Optional[float] = None,
    bump: float = RATE_BUMP,
) -> float:
    """Change in dirty NPV for a `bump` parallel shift of the zero curve.

    A **bumped revaluation**, and labelled as such: the profile is a flat
    constant, so the only shift it can express is a parallel one, and
    calling this a per-pillar sensitivity would overstate what a
    single-rate curve can distinguish. The value is the one-sided
    difference `NPV(r + bump) - NPV(r)`, in currency, for the stated bump.

    Deliberately re-prices through the same code path rather than
    differentiating a formula: a sensitivity derived from an expression
    that has drifted from the pricer measures the expression, not the
    price.
    """
    base = price_note(
        entry, signed_face_amount, valuation_date, profile,
        exported_accrued_fraction,
    ).dirty_npv
    bumped_profile = AssumedProfile(
        profile_id=profile.profile_id, description=profile.description,
        flat_rate=profile.flat_rate + bump, times=profile.times,
    )
    bumped = price_note(
        entry, signed_face_amount, valuation_date, bumped_profile,
        exported_accrued_fraction,
    ).dirty_npv
    return bumped - base
