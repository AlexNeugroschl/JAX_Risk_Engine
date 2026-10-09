"""
Unit normalization: source units in, engine units out.

| Source field              | Source unit              | Normalized                        |
|---------------------------|--------------------------|-----------------------------------|
| `coupon`                  | annual percent (4.0)     | decimal (0.04)                    |
| `closingMark`             | clean, fraction of par   | kept as fraction, echoed          |
| `accruedInterestFraction` | fraction of par          | x signed face -> signed currency  |
| `quantity`                | signed face              | `signed_face_amount`              |

`MAPPING_VERSION` is echoed in every result and is part of the workload key; change it when
a rule changes.

A blank `accruedInterestFraction` means different things depending on the terms, so it is
interpreted from the terms, never from the blank itself:

| Terms say                             | Accrued field | Result                                 |
|---------------------------------------|---------------|----------------------------------------|
| Zero-coupon (`couponFrequency: NONE`) | blank         | `0.0`, `provenance: structural-zero`   |
| Coupon-bearing                        | blank         | `unavailable`, `ACCRUED_NOT_SUPPLIED`  |
| Coupon-bearing                        | present       | converted                              |
| No terms artifact (v1 bundle)         | blank         | `unavailable` (uninterpretable)        |

Treating blank as 0 is right for a bill and wrong for a note. Without terms even a bill is
`unavailable`: the engine does not infer conventions from the CSV `coupon` column.
"""
from dataclasses import dataclass
from typing import Optional

from engine.traderx.terms import (  # noqa: F401  (NO_TERMS_ARTIFACT is a re-export)
    NO_TERMS_ARTIFACT,
    JoinedRow,
    TermsEntry,
)

#: Revision of these rules; echoed in results and part of the workload key, so a result
#: cached under old rules is never reused under new ones.
MAPPING_VERSION = "traderx-adapter-v1"

#: `Quantity.status` values.
OK = "ok"
UNAVAILABLE = "unavailable"

#: `Quantity.provenance` for a zero that follows from the instrument's structure (a bill has
#: no coupons, so no accrued), as opposed to a measured or missing value.
STRUCTURAL_ZERO = "structural-zero"
CONVERTED = "converted"

#: Reason codes for an `unavailable` quantity. `NO_TERMS_ARTIFACT` is re-exported from
#: `engine.traderx.terms` so both layers use one value.
ACCRUED_NOT_SUPPLIED = "ACCRUED_NOT_SUPPLIED"


class NormalizationError(ValueError):
    """A present but malformed source field (non-numeric quantity or coupon). A missing
    field is not an error: it yields an `unavailable` `Quantity`."""


@dataclass(frozen=True)
class Quantity:
    """A normalized value or its explicit absence. `value` is meaningful only when
    `status == "ok"`; `provenance` says why a zero is zero."""
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
    #: Signed face amount. The CSV `quantity` is already signed currency face, so this is a
    #: parse; `faceDenomination` in the terms does not rescale it.
    signed_face_amount: float
    #: Clean price as a fraction of par, echoed in the source unit.
    observed_clean_price: Optional[float]
    #: Annual coupon as a decimal (source is percent).
    coupon_rate: Optional[float]
    #: Signed accrued interest in currency, or an explicit absence.
    accrued_interest: Quantity
    mapping_version: str = MAPPING_VERSION


def _parse_float(raw: Optional[str], field: str) -> Optional[float]:
    """A CSV number: `None` if blank or absent, raises if present but unparseable."""
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
    """Whether the terms say the instrument has no coupon schedule (`couponFrequency`)."""
    return str(entry.terms.get("couponFrequency", "")).upper() == "NONE"


def _normalize_accrued(
    raw_accrued: Optional[str],
    signed_face: float,
    entry: Optional[TermsEntry],
) -> Quantity:
    """Accrued interest per the zero-coupon rule (module docstring). Multiplying by the
    signed face applies the position's sign, so a short's accrued is negative."""
    fraction = _parse_float(raw_accrued, "accruedInterestFraction")

    if fraction is not None:
        # Present: convert; the exported figure wins over the terms.
        return Quantity.ok(fraction * signed_face, CONVERTED)

    # Blank: the meaning depends on the terms.
    if entry is None:
        # No terms artifact: uninterpretable.
        return Quantity.unavailable(NO_TERMS_ARTIFACT)

    if _is_zero_coupon(entry):
        # No coupon schedule: structurally zero.
        return Quantity.ok(0.0, STRUCTURAL_ZERO)

    # Coupon-bearing and not supplied.
    return Quantity.unavailable(ACCRUED_NOT_SUPPLIED)


def normalize_position(joined: JoinedRow) -> NormalizedPosition:
    """One joined position row in engine units. Raises `NormalizationError` for a
    missing or malformed required field; optional absent values become `unavailable`."""
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
        # Clean price as a fraction of par, echoed in the source unit.
        observed_clean_price=_parse_float(row.get("closingMark"), "closingMark"),
        # Annual percent -> decimal. 4.0 means 4%.
        coupon_rate=None if coupon_percent is None else coupon_percent / 100.0,
        accrued_interest=_normalize_accrued(
            row.get("accruedInterestFraction"), signed_face, joined.entry
        ),
    )
