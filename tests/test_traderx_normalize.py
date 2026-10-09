"""
Unit normalization and the zero-coupon rule (`engine.traderx.normalize`, W0.3).

`TestZeroCouponRule` pins the four cases of the blank-accrued rule. The key one is
`test_coupon_bearing_blank_is_not_zero`: a naive `blank -> 0.0` passes everything else in
this file.
"""
from pathlib import Path

import pytest

from engine.traderx.bundle import load_bundle
from engine.traderx.normalize import (
    ACCRUED_NOT_SUPPLIED,
    CONVERTED,
    MAPPING_VERSION,
    NO_TERMS_ARTIFACT,
    NormalizationError,
    NormalizedPosition,
    Quantity,
    STRUCTURAL_ZERO,
    normalize_position,
)
from engine.traderx.terms import JoinedRow, TermsEntry, join_terms

FIXTURES = Path(__file__).parent / "fixtures" / "traderx-eod"


def _positions(case: str, version: str = "v2"):
    """Every position row of a fixture, joined, keyed by account."""
    join = join_terms(load_bundle(FIXTURES / case / version))
    return {r.row["accountId"]: r for r in join.rows if r.source == "positions"}


def _entry(coupon_frequency: str) -> TermsEntry:
    """A minimal terms entry differing only in `couponFrequency`, the field the zero-coupon
    rule keys on."""
    return TermsEntry(
        instrument_type="TREASURY",
        terms={"couponFrequency": coupon_frequency, "currency": "USD"},
        missing_terms=(),
        provenance={"origin": "synthetic"},
        identity={"source": "positions", "security": "TEST"},
    )


def _row(accrued: str, quantity: str = "100000", entry=None) -> JoinedRow:
    return JoinedRow(
        source="positions",
        row={
            "accountId": "1", "security": "TEST", "instrumentType": "TREASURY",
            "currency": "USD", "quantity": quantity, "coupon": "4.0",
            "closingMark": "1.015000", "accruedInterestFraction": accrued,
        },
        entry=entry,
        unjoined_reason=None if entry else NO_TERMS_ARTIFACT,
    )


class TestUnitConversions:
    """Unit conversion."""

    def test_coupon_percent_becomes_decimal(self):
        """Source is annual PERCENT: 4.0 means 4%."""
        normalized = normalize_position(_positions("note")["22214"])
        assert normalized.coupon_rate == pytest.approx(0.04)

    def test_zero_coupon_bill_converts_to_zero_rate(self):
        normalized = normalize_position(_positions("bill")["22214"])
        assert normalized.coupon_rate == pytest.approx(0.0)

    def test_closing_mark_stays_a_fraction_of_par(self):
        """`closingMark` is clean and stays in its source unit, echoed as
        `observedCleanPrice`."""
        normalized = normalize_position(_positions("note")["22214"])
        assert normalized.observed_clean_price == pytest.approx(1.015)

    def test_quantity_is_signed_face_and_is_not_rescaled(self):
        """The quantity is already signed currency face; `faceDenomination` (100 here) does
        not rescale it (applying it would be a 100x error)."""
        positions = _positions("note")
        assert normalize_position(positions["22214"]).signed_face_amount == 100_000.0
        assert normalize_position(positions["42422"]).signed_face_amount == -100_000.0

    def test_mapping_version_is_present(self):
        """`mappingVersion` is echoed in every result."""
        normalized = normalize_position(_positions("note")["22214"])
        assert normalized.mapping_version == MAPPING_VERSION
        assert MAPPING_VERSION  # non-empty


class TestAccruedSignAndMagnitude:
    """Accrued is signed for the position (short negative), like NPV: long and short have
    opposite signs and equal magnitude."""

    def test_long_and_short_are_opposite_and_equal_in_magnitude(self):
        positions = _positions("note")
        long_accrued = normalize_position(positions["22214"]).accrued_interest
        short_accrued = normalize_position(positions["42422"]).accrued_interest

        assert long_accrued.is_ok and short_accrued.is_ok
        assert long_accrued.value == pytest.approx(-short_accrued.value)
        assert abs(long_accrued.value) == pytest.approx(abs(short_accrued.value))

    def test_long_is_positive_short_is_negative(self):
        positions = _positions("note")
        assert normalize_position(positions["22214"]).accrued_interest.value > 0
        assert normalize_position(positions["42422"]).accrued_interest.value < 0

    def test_magnitude_reconciles_to_the_exported_fraction(self):
        """0.018571 of par on 100,000 face (the adapter's own arithmetic; note pricing
        reconciles against 0.0185714286)."""
        accrued = normalize_position(_positions("note")["22214"]).accrued_interest
        assert accrued.value == pytest.approx(0.018571 * 100_000.0)


class TestZeroCouponRule:
    """The four cases: a blank accrued is 0.0 for a bill and silently wrong as 0.0 for a
    coupon-bearing note."""

    def test_row1_zero_coupon_blank_is_structural_zero(self):
        """`couponFrequency: NONE` + blank -> 0.0 with `provenance: structural-zero`."""
        accrued = normalize_position(_positions("bill")["22214"]).accrued_interest
        assert accrued.is_ok
        assert accrued.value == 0.0
        assert accrued.provenance == STRUCTURAL_ZERO

    def test_row2_coupon_bearing_blank_is_unavailable(self):
        """Coupon-bearing + blank -> `unavailable` + `ACCRUED_NOT_SUPPLIED`."""
        accrued = normalize_position(_row("", entry=_entry("6M"))).accrued_interest
        assert accrued.status == "unavailable"
        assert accrued.reason == ACCRUED_NOT_SUPPLIED

    def test_coupon_bearing_blank_is_not_zero(self):
        """Coupon-bearing + blank is `unavailable`, not 0.0 (fails against `blank -> 0.0`)."""
        accrued = normalize_position(_row("", entry=_entry("6M"))).accrued_interest
        assert accrued.value != 0.0
        assert accrued.value is None
        assert not accrued.is_ok

    def test_row3_coupon_bearing_present_converts(self):
        accrued = normalize_position(_row("0.018571", entry=_entry("6M"))).accrued_interest
        assert accrued.is_ok
        assert accrued.provenance == CONVERTED
        assert accrued.value == pytest.approx(1857.1)

    def test_row4_no_terms_artifact_blank_is_unavailable(self):
        """v1 (no terms) + blank -> `unavailable`: a bill cannot be told from a note. (The
        v1 bill; the v1 note supplies a value.)"""
        accrued = normalize_position(_positions("bill", "v1")["22214"]).accrued_interest
        assert accrued.status == "unavailable"
        assert accrued.reason == NO_TERMS_ARTIFACT
        assert accrued.value is None

    def test_v1_with_a_supplied_value_still_converts(self):
        """The rule is about blanks: a supplied fraction converts even without terms."""
        accrued = normalize_position(_positions("note", "v1")["22214"]).accrued_interest
        assert accrued.is_ok
        assert accrued.value == pytest.approx(1857.1)

    def test_rule_keys_on_terms_not_on_the_coupon_column(self):
        """A CSV `coupon` of 0 on a coupon-bearing terms entry is still `unavailable`:
        structure comes from the terms, not from a value in the row."""
        row = _row("", entry=_entry("6M"))
        row.row["coupon"] = "0"

        accrued = normalize_position(row).accrued_interest
        assert accrued.status == "unavailable"
        assert accrued.value != 0.0

    def test_structural_zero_is_distinguishable_from_a_converted_zero(self):
        """Both are 0.0; only `provenance` says why."""
        structural = normalize_position(_row("", entry=_entry("NONE"))).accrued_interest
        converted = normalize_position(_row("0.0", entry=_entry("6M"))).accrued_interest

        assert structural.value == converted.value == 0.0
        assert structural.provenance == STRUCTURAL_ZERO
        assert converted.provenance == CONVERTED


class TestMalformedInput:
    """A present-but-broken field is an error; a missing one is a status."""

    def test_non_numeric_quantity_raises(self):
        with pytest.raises(NormalizationError, match="quantity"):
            normalize_position(_row("0.01", quantity="not-a-number", entry=_entry("6M")))

    def test_blank_quantity_raises(self):
        """Unlike accrued, a quantity cannot legitimately be absent."""
        with pytest.raises(NormalizationError, match="quantity is required"):
            normalize_position(_row("0.01", quantity="", entry=_entry("6M")))

    def test_non_numeric_accrued_raises(self):
        row = _row("not-a-number", entry=_entry("6M"))
        with pytest.raises(NormalizationError, match="accruedInterestFraction"):
            normalize_position(row)

    def test_contracts_row_is_rejected(self):
        contracts_row = JoinedRow(source="contracts", row={}, entry=None)
        with pytest.raises(NormalizationError, match="expects a positions row"):
            normalize_position(contracts_row)


class TestQuantityInvariants:
    def test_unavailable_carries_no_value(self):
        """A non-ok `Quantity` carries no number."""
        q = Quantity.unavailable(ACCRUED_NOT_SUPPLIED)
        assert q.value is None
        assert not q.is_ok

    def test_ok_carries_a_provenance(self):
        q = Quantity.ok(1.0, CONVERTED)
        assert q.is_ok and q.provenance == CONVERTED


class TestRoundTripThroughTheResult:
    """`mappingVersion` round-trips."""

    def test_mapping_version_reaches_the_published_result(self):
        from engine.traderx import price_bundle
        result = price_bundle(FIXTURES / "note" / "v2").to_dict()

        assert result["mappingVersion"] == MAPPING_VERSION
        for item in result["items"]:
            assert item["mappingVersion"] == MAPPING_VERSION

    def test_accrued_echoes_its_source_units(self):
        from engine.traderx import price_bundle
        result = price_bundle(FIXTURES / "note" / "v2").to_dict()
        accrued = result["items"][0]["calculations"]["accruedInterest"]

        assert accrued["status"] == "ok"
        assert accrued["observedCleanPrice"] == pytest.approx(1.015)
        assert accrued["signedFaceAmount"] == pytest.approx(100_000.0)
        assert accrued["provenance"] == CONVERTED
