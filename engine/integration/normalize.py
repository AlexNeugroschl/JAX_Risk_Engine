"""
W0.3 -- unit normalization: source units in, engine units out.

Every conversion here is mechanical. The one piece of real judgement is the
zero-coupon rule below, and it is the reason this module exists as its own
layer rather than as a few inline `float()` calls.

| Source field              | Source unit              | Normalized                        |
|---------------------------|--------------------------|-----------------------------------|
| `coupon`                  | annual **percent** (4.0) | decimal (0.04)                    |
| `closingMark`             | clean, fraction of par   | kept as fraction, echoed          |
| `accruedInterestFraction` | fraction of par          | x face x sign -> signed currency  |
| `quantity`                | signed face              | `signed_face_amount`              |

`MAPPING_VERSION` is echoed in every result (plan §W0.3) so a consumer can
tell which revision of these rules produced a number. Change the rules,
change the version -- it is part of the workload key.

---

**The zero-coupon rule -- key on terms, never on the blank.**

A bill and a coupon-bearing note both show a **blank** `accruedInterestFraction`
in the extract. They mean opposite things:

| Terms say                         | Accrued field | Result                                    |
|-----------------------------------|---------------|-------------------------------------------|
| Zero-coupon (`couponFrequency: NONE`) | blank     | `0.0`, `provenance: structural-zero`      |
| Coupon-bearing                    | blank         | **`unavailable` + `ACCRUED_NOT_SUPPLIED`** |
| Coupon-bearing                    | present       | convert                                   |
| **No terms artifact**             | blank         | **`unavailable`** -- uninterpretable       |

The naive `blank -> 0.0` is correct for the bill and **silently wrong** for
the note, and a bill-only test suite never notices. TraderX's own exporter
preamble makes the same point from the other side: "Empty means no schedule
exists; a zero in the accrual column would mean one exists and nothing has
accrued, which is a different and false claim."

The fourth row is the subtle one. Without a terms artifact there is nothing
to key on -- the blank is *uninterpretable*, not zero. A v1 bundle therefore
yields `unavailable` even for what is in fact a bill, because the engine
cannot know that from a v1 bundle. Guessing from the `coupon` CSV column
would be inferring conventions from a blank, which is what this whole design
refuses to do.
"""
from dataclasses import dataclass
from typing import Optional

from engine.integration.terms import JoinedRow, TermsEntry

#: Revision of the rules in this module. Echoed in every result and part of
#: the workload key -- bump it whenever a conversion changes, so a cached
#: result computed under the old rules is never reused under the new ones.
MAPPING_VERSION = "traderx-adapter-v1"

#: `Quantity.status` values. A deliberately small vocabulary mirroring the
#: per-calculation statuses in `engine.integration.result`.
OK = "ok"
UNAVAILABLE = "unavailable"

#: `Quantity.provenance` for a value that is zero because the instrument's
#: structure makes it zero -- not because it was measured as zero, and not
#: because it was missing. A bill has no coupon schedule, so its accrued
#: interest is structurally zero.
STRUCTURAL_ZERO = "structural-zero"
CONVERTED = "converted"

#: Reason codes for an `unavailable` quantity.
ACCRUED_NOT_SUPPLIED = "ACCRUED_NOT_SUPPLIED"
NO_TERMS_ARTIFACT = "NO_TERMS_ARTIFACT"


class NormalizationError(ValueError):
    """A source field that is present but malformed -- a non-numeric
    quantity, an unparseable coupon.

    Distinct from a *missing* field, which produces an `unavailable`
    `Quantity` rather than an exception. Missing is a knowable state the
    contract has an answer for; malformed means the extract is broken.
    """


@dataclass(frozen=True)
class Quantity:
    """One normalized value, or the explicit absence of one.

    A bare `float` cannot represent "this is zero because the instrument
    has no coupon schedule" distinctly from "this was not supplied" -- and
    those two are exactly what must not be conflated. Hence this wrapper:
    `value` is meaningful only when `status == "ok"`, and `provenance` says
    *why* a zero is zero.
    """
    status: str
    value: Optional[float] = None
    provenance: Optional[str] = None
    reason: Optional[str] = None

    @property
    def is_ok(self) -> bool:
        return self.status == OK

    @classmethod
    def ok(cls, value: float, provenance: str) -> "Quantity":
        return cls(status=OK, value=value, provenance=provenance)

    @classmethod
    def unavailable(cls, reason: str) -> "Quantity":
        return cls(status=UNAVAILABLE, reason=reason)


@dataclass(frozen=True)
class NormalizedPosition:
    """A position row in engine units, with `mapping_version` attached."""
    account_id: str
    security: str
    instrument_type: str
    currency: str
    #: Signed face amount. The CSV `quantity` is already signed currency
    #: face (`quantityUnit=signed-currency-face`), so this is a parse, not a
    #: rescale. `faceDenomination` in the terms "does not rescale the CSV
    #: position quantity" (bundle-v2-and-terms.md) and is deliberately not
    #: applied here.
    signed_face_amount: float
    #: Clean price as a fraction of par, echoed in the source unit.
    observed_clean_price: Optional[float]
    #: Annual coupon as a decimal (source is percent).
    coupon_rate: Optional[float]
    #: Signed accrued interest in currency, or an explicit absence.
    accrued_interest: Quantity
    mapping_version: str = MAPPING_VERSION


def _parse_float(raw: Optional[str], field: str) -> Optional[float]:
    """Parses a CSV numeric field. A blank/absent field is `None` (the
    caller decides what that means); a present-but-unparseable one raises."""
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise NormalizationError(f"{field}={raw!r} is not a number") from exc


def _is_zero_coupon(entry: TermsEntry) -> bool:
    """Whether the *terms* say this instrument has no coupon schedule.

    Keyed on `couponFrequency`, the field that states it, and never on a
    blank accrual column or a `coupon` of 0 in the CSV -- see this module's
    docstring.
    """
    return str(entry.terms.get("couponFrequency", "")).upper() == "NONE"


def _normalize_accrued(
    raw_accrued: Optional[str],
    signed_face: float,
    entry: Optional[TermsEntry],
) -> Quantity:
    """The zero-coupon rule. See this module's docstring for the full table.

    `signed_face` carries the position's sign, so a short position's accrued
    interest comes out negative -- consistent with NPV (plan §1, "Accrued
    sign"). Multiplying by the signed face does the sign and the scaling in
    one step; there is no separate `sign()` factor to get wrong.
    """
    fraction = _parse_float(raw_accrued, "accruedInterestFraction")

    if fraction is not None:
        # Present: convert regardless of what the terms say. The exported
        # economics are authoritative over the reference supplement.
        return Quantity.ok(fraction * signed_face, CONVERTED)

    # Blank from here down -- and what it means depends entirely on terms.
    if entry is None:
        # Row 4: no terms artifact. The blank is uninterpretable, not zero.
        return Quantity.unavailable(NO_TERMS_ARTIFACT)

    if _is_zero_coupon(entry):
        # Row 1: the instrument has no coupon schedule. Zero is a fact about
        # its structure, not a measurement and not a default.
        return Quantity.ok(0.0, STRUCTURAL_ZERO)

    # Row 2: coupon-bearing with nothing supplied. The naive `0.0` here is
    # the silent error this whole module exists to prevent.
    return Quantity.unavailable(ACCRUED_NOT_SUPPLIED)


def normalize_position(joined: JoinedRow) -> NormalizedPosition:
    """Normalizes one joined position row into engine units.

    Raises `NormalizationError` if a required field is missing or malformed;
    returns `Quantity.unavailable(...)` for values the contract says may
    legitimately be absent.
    """
    if joined.source != "positions":
        raise NormalizationError(
            f"normalize_position expects a positions row, got source={joined.source!r}"
        )
    row = joined.row

    signed_face = _parse_float(row.get("quantity"), "quantity")
    if signed_face is None:
        raise NormalizationError("quantity is required on a position row but was blank")

    coupon_percent = _parse_float(row.get("coupon"), "coupon")

    return NormalizedPosition(
        account_id=row["accountId"],
        security=row["security"],
        instrument_type=row["instrumentType"],
        currency=row.get("currency", ""),
        signed_face_amount=signed_face,
        # `closingMark` is clean and stays a fraction of par -- echoed in its
        # source unit so a consumer can reconcile against the extract
        # (`priceBasis: clean-fraction-of-par`, confirmed by TraderX).
        observed_clean_price=_parse_float(row.get("closingMark"), "closingMark"),
        # Annual percent -> decimal. 4.0 means 4%.
        coupon_rate=None if coupon_percent is None else coupon_percent / 100.0,
        accrued_interest=_normalize_accrued(
            row.get("accruedInterestFraction"), signed_face, joined.entry
        ),
    )
