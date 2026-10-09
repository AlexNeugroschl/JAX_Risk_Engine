"""
Treasury note pricer: fixed coupons plus a bullet redemption.

    NPV = signed_face x [ sum_i c_i x P(t_i) + redemptionFraction x P(T) ]

`c_i` = coupon rate x period accrual under the instrument's own day count (from
`terms.dayCount`; refused if unsupported, never defaulted). ACT/ACT (ICMA) accrues against
the coupon period, so a semiannual period is exactly 0.5. P is the assumed profile's
flat, continuously compounded ACT/365 discount factor, as in `engine.traderx.bill`.
NPV is dirty.

The schedule is the terms' explicit periods, used as supplied (validated for structure and
contiguity, never regenerated). Coupons paid on or before the valuation date are dropped.

Accrued interest has two sources: the exported `accruedInterestFraction` (rounded HALF_EVEN
by the exporter), which is what is reported, and a recomputation from the schedule, which
is the check. Both travel in the payload with an explicit `accrualSource`. They are
compared within

    0.5 x 10^-fractionDecimals x |face| + 0.01

(`accrual_mismatch_tolerance`); a larger difference means the priced schedule is not the
one the exporter accrued against, and the note is refused.

Valuation and accrual are as of the session date: only `settlementDays: 0` is supported, and
any other lag is refused.

`rate_sensitivity` re-prices with the same declared `fractionDecimals` as `price_note`, so
the two outcomes reconcile accrued interest under one tolerance (I-40).
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import ORE

from engine.market_data.day_counts import (
    UnsupportedDayCountError,
    resolve_accrual_day_count,
)
from engine.traderx.bill import DISCOUNT_DAY_COUNT, _iso
from engine.traderx.market_inputs import AssumedProfile, CurveProvenance
from engine.traderx.terms import TermsEntry

#: Reason codes for refusals.
NOT_A_NOTE = "NOT_A_NOTE"
INSTRUMENT_MATURED = "INSTRUMENT_MATURED"
TERMS_INCOMPLETE = "TERMS_INCOMPLETE"
DAY_COUNT_NOT_SUPPORTED = "DAY_COUNT_NOT_SUPPORTED"
SCHEDULE_INCONSISTENT = "SCHEDULE_INCONSISTENT"
ACCRUAL_MISMATCH = "ACCRUAL_MISMATCH"
SETTLEMENT_CONVENTION_NOT_SUPPORTED = "SETTLEMENT_CONVENTION_NOT_SUPPORTED"

#: Pricing method, echoed in the result.
METHOD = "discounted-cashflow"

#: `accrualSource` labels: which path the reported accrued came from.
ACCRUAL_EXPORTED = "exported-fraction"
ACCRUAL_RECOMPUTED = "recomputed-schedule"

#: Decimals the exporter rounds `accruedInterestFraction` to (positions preamble: "HALF_EVEN
#: at 6 decimals"), used when the terms declare none.
DEFAULT_FRACTION_DECIMALS = 6

#: Supported settlement lag: zero only (the exporter accrues to the session date).
SUPPORTED_SETTLEMENT_DAYS = (0,)

#: Parallel bump for `rateSensitivity`, absolute (1bp), echoed in the payload.
RATE_BUMP = 1e-4

#: How `rateSensitivity` is produced: bumped revaluation (not AD).
SENSITIVITY_METHOD = "bumped-revaluation"


class NotePricingError(Exception):
    """A note could not be priced: reason code and detail, reported as that row's refusal
    so one row cannot fail the bundle."""

    def __init__(self, reason: str, detail: str):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


@dataclass(frozen=True)
class CouponFlow:
    """One scheduled coupon with every input to its value, so a reconciliation can find
    which coupon, and which input, diverged."""
    start_date: str
    end_date: str
    payment_date: str
    #: Period accrual under the instrument's day count.
    accrual_fraction: float
    #: `coupon_rate x accrual_fraction`, per unit face.
    amount_per_unit_face: float
    #: Discount factor at the payment date, off the requested curve.
    discount_factor: float
    #: Time to payment under the discounting day count (ACT/365).
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
    """The two accrued-interest values (exported and recomputed), their difference and the
    tolerance it was checked against."""
    #: The exported (rounded) fraction per unit face, or `None` if blank.
    exported_fraction: Optional[float]
    #: Recomputed from the schedule and day count, unrounded, per unit face.
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
    """One note position's value and the inputs to check it. `dirty_npv` is the reported
    `npv`; `clean_npv = dirty_npv - accrued_interest`, comparable with the exported clean
    `closingMark`."""
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
        """The reported `npv`: the dirty present value."""
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
    """Whether the terms describe a coupon-bearing scheduled instrument: a coupon frequency
    other than NONE and a non-empty explicit schedule (never inferred from the CSV)."""
    if entry is None:
        return False
    terms = entry.terms
    if str(terms.get("couponFrequency", "")).upper() in ("", "NONE"):
        return False
    return bool(terms.get("schedule"))


def accrual_mismatch_tolerance(
    signed_face_amount: float, fraction_decimals: int = DEFAULT_FRACTION_DECIMALS,
) -> float:
    """Accrued-interest reconciliation tolerance:

        0.5 x 10^-fractionDecimals x |face| + 0.01

    The first term is the largest error the exporter's rounding can introduce on this face;
    the 0.01 absorbs cent rounding. The bound itself is not rounded (rounding it would make
    the tolerance tighter than the rounding error it must admit).
    """
    rounding_error = 0.5 * (10.0 ** -fraction_decimals) * abs(signed_face_amount)
    # Not rounded (see the docstring).
    return rounding_error + 0.01


def _parse_date(raw: Optional[str], field: str) -> ORE.Date:
    """ISO `YYYY-MM-DD` terms date -> `ORE.Date`, raising `NotePricingError` (not the
    bill's error type, which the pipeline's note handler would not catch)."""
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
        # ORE.Date raises RuntimeError for a well-formed but impossible date (2025-02-30);
        # catching it keeps one bad row from failing the whole bundle
        # (tests/test_traderx_note.py::TestImpossibleCalendarDates).
        raise NotePricingError(
            TERMS_INCOMPLETE, f"{field}={raw!r} is not a valid ISO YYYY-MM-DD date ({exc})",
        ) from exc


def _terms_float(entry: TermsEntry, field: str, required: bool = True,
                 default: Optional[float] = None) -> Optional[float]:
    """A numeric terms field: absent gives `default` (or a refusal if required); present
    but unparseable is always a refusal."""
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
    """The instrument's accrual day count from its terms; refused if absent or
    unsupported."""
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
    """Refuse a non-zero `settlementDays`: accrual is to the session date, and a lag would
    put accrual and discounting on different dates."""
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
    """The terms' explicit coupon schedule as `(start, end, payment)`, validated for
    structure and contiguity but never regenerated."""
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
        # paymentDate defaults to endDate only when absent; an unparseable value raises.
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
    """`(discount_factor, year_fraction)`: continuous compounding over ACT/365, at the flat
    rate plus `rate_shift` (a parallel bump)."""
    year_fraction = DISCOUNT_DAY_COUNT.yearFraction(valuation, date)
    return math.exp(-(profile.flat_rate + rate_shift) * year_fraction), year_fraction


def _accrued_fraction_from_schedule(
    periods: Sequence[Tuple[ORE.Date, ORE.Date, ORE.Date]],
    valuation: ORE.Date, coupon_rate: float, day_count,
) -> float:
    """Accrued interest per unit face from the schedule: accrual from the current period's
    start to the valuation date under the instrument's day count (with the reference period,
    as ICMA requires). 0 before the first period."""
    for start, end, _payment in periods:
        if start <= valuation < end:
            return coupon_rate * day_count.yearFraction(start, valuation, start, end)
    if periods and valuation < periods[0][0]:
        return 0.0
    # After the last accrual end nothing accrues (a matured note is refused earlier).
    return 0.0


def _reconcile_accrued(
    exported_fraction: Optional[float], recomputed_fraction: float,
    signed_face_amount: float, fraction_decimals: int,
) -> AccruedReconciliation:
    """Build the two-path reconciliation, or raise `ACCRUAL_MISMATCH` when the difference
    exceeds the tolerance (a refusal, not a warning: every coupon is then suspect)."""
    tolerance = accrual_mismatch_tolerance(signed_face_amount, fraction_decimals)

    if exported_fraction is None:
        # Blank export: report the recomputed value, labelled as such.
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
    """Price one coupon-bearing Treasury note position against `profile`.

    `signed_face_amount` carries the sign (a short gives a negative NPV); do not apply
    another. `exported_accrued_fraction` is the extract's accrued fraction: reported when
    given, with the recomputed value as the check; when absent the recomputed value is
    reported. Raises `NotePricingError` rather than substituting anything.
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

    # `couponRatePercent` is an annual percent (4.0 = 4%).
    coupon_rate = _terms_float(entry, "couponRatePercent") / 100.0
    redemption = _terms_float(entry, "redemptionFraction", required=False, default=1.0)

    coupons: List[CouponFlow] = []
    for start, end, payment in periods:
        if payment <= valuation_date:
            # Already paid: not a remaining cashflow.
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
    fraction_decimals: int = DEFAULT_FRACTION_DECIMALS,
) -> float:
    """Dirty NPV change for a parallel `bump` of the flat profile: `NPV(r + bump) - NPV(r)`,
    by repricing through `price_note` with the same `fraction_decimals`."""
    base = price_note(
        entry, signed_face_amount, valuation_date, profile,
        exported_accrued_fraction, fraction_decimals,
    ).dirty_npv
    bumped_profile = AssumedProfile(
        profile_id=profile.profile_id, description=profile.description,
        flat_rate=profile.flat_rate + bump, times=profile.times,
    )
    bumped = price_note(
        entry, signed_face_amount, valuation_date, bumped_profile,
        exported_accrued_fraction, fraction_decimals,
    ).dirty_npv
    return bumped - base
