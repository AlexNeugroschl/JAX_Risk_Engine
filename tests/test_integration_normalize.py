"""
W0.3 -- unit normalization and the zero-coupon rule
(`docs/planning/traderx-integration-plan.md` §W0.3).

The centrepiece is `TestZeroCouponRule`, which pins all four rows of the
plan's table. `test_coupon_bearing_blank_is_not_zero` is the one that
matters most: it is the assertion a bill-only, long-only test suite would
never make, and the naive `blank -> 0.0` implementation passes everything
else in this file.
"""
from pathlib import Path

import pytest

from engine.integration.bundle import load_bundle
from engine.integration.normalize import (
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
from engine.integration.terms import JoinedRow, TermsEntry, join_terms

FIXTURES = Path(__file__).parent / "fixtures" / "traderx-eod"


def _positions(case: str, version: str = "v2"):
    """Every position row of a fixture, joined, keyed by account."""
    join = join_terms(load_bundle(FIXTURES / case / version))
    return {r.row["accountId"]: r for r in join.rows if r.source == "positions"}


def _entry(coupon_frequency: str) -> TermsEntry:
    """A minimal terms entry differing only in `couponFrequency` -- the
    single field the zero-coupon rule keys on."""
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
    """The mechanical half of the adapter."""

    def test_coupon_percent_becomes_decimal(self):
        """Source is annual PERCENT: 4.0 means 4%."""
        normalized = normalize_position(_positions("note")["22214"])
        assert normalized.coupon_rate == pytest.approx(0.04)

    def test_zero_coupon_bill_converts_to_zero_rate(self):
        normalized = normalize_position(_positions("bill")["22214"])
        assert normalized.coupon_rate == pytest.approx(0.0)

    def test_closing_mark_stays_a_fraction_of_par(self):
        """`closingMark` is clean and stays in its source unit, echoed as
        `observedCleanPrice` so a consumer can reconcile against the
        extract without re-deriving anything."""
        normalized = normalize_position(_positions("note")["22214"])
        assert normalized.observed_clean_price == pytest.approx(1.015)

    def test_quantity_is_signed_face_and_is_not_rescaled(self):
        """`quantityUnit=signed-currency-face`: already signed currency
        face. `faceDenomination` (100 in these fixtures) 'does not rescale
        the CSV position quantity' -- applying it would be a 100x error."""
        positions = _positions("note")
        assert normalize_position(positions["22214"]).signed_face_amount == 100_000.0
        assert normalize_position(positions["42422"]).signed_face_amount == -100_000.0

    def test_mapping_version_is_present(self):
        """Plan §W0.3: `mappingVersion` echoed in every result."""
        normalized = normalize_position(_positions("note")["22214"])
        assert normalized.mapping_version == MAPPING_VERSION
        assert MAPPING_VERSION  # non-empty


class TestAccruedSignAndMagnitude:
    """Plan §1 'Accrued sign': signed for the position (short -> negative),
    consistent with NPV. Named test: 'long **and short** note (signs
    opposite, equal magnitude)'."""

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
        """0.018571 of par on 100,000 face. (W1.3 moves the *pricing*-side
        reconciliation to 0.0185714286 into the engine; this is the
        adapter's own arithmetic.)"""
        accrued = normalize_position(_positions("note")["22214"]).accrued_interest
        assert accrued.value == pytest.approx(0.018571 * 100_000.0)


class TestZeroCouponRule:
    """Plan §W0.3's table, all four rows.

    'A bill and a coupon-bearing note both show a blank accrued field. The
    naive `blank -> 0.0` is correct for one and silently wrong for the
    other.'
    """

    def test_row1_zero_coupon_blank_is_structural_zero(self):
        """Terms say `couponFrequency: NONE` + blank -> 0.0 with
        `provenance: structural-zero`."""
        accrued = normalize_position(_positions("bill")["22214"]).accrued_interest
        assert accrued.is_ok
        assert accrued.value == 0.0
        assert accrued.provenance == STRUCTURAL_ZERO

    def test_row2_coupon_bearing_blank_is_unavailable(self):
        """Terms say coupon-bearing + blank -> `unavailable` +
        `ACCRUED_NOT_SUPPLIED`."""
        accrued = normalize_position(_row("", entry=_entry("6M"))).accrued_interest
        assert accrued.status == "unavailable"
        assert accrued.reason == ACCRUED_NOT_SUPPLIED

    def test_coupon_bearing_blank_is_not_zero(self):
        """**The assertion the plan calls out explicitly**: 'coupon-bearing
        with blank accrued -> `unavailable`, asserted NOT `0.0`'.

        This is the test that fails against the plausible-but-wrong
        implementation (`blank -> 0.0`). Every other test in this file
        passes against it.
        """
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
        """Plan's named test: 'v1 bundle -> `unavailable`'.

        Without terms the blank is *uninterpretable*, not zero -- the
        engine cannot tell a bill from a note. Note this is the v1 BILL:
        the v1 note supplies a value, so it converts normally.
        """
        accrued = normalize_position(_positions("bill", "v1")["22214"]).accrued_interest
        assert accrued.status == "unavailable"
        assert accrued.reason == NO_TERMS_ARTIFACT
        assert accrued.value is None

    def test_v1_with_a_supplied_value_still_converts(self):
        """The rule is about BLANKS. A v1 bundle that supplies the fraction
        is converted -- refusing it would discard exported data."""
        accrued = normalize_position(_positions("note", "v1")["22214"]).accrued_interest
        assert accrued.is_ok
        assert accrued.value == pytest.approx(1857.1)

    def test_rule_keys_on_terms_not_on_the_coupon_column(self):
        """A row whose CSV `coupon` is 0 but whose TERMS say coupon-bearing
        is `unavailable`, not a structural zero.

        Keying on the CSV column would be inferring structure from a
        value -- the exporter's own preamble makes the same point: 'a zero
        in the accrual column would mean one exists and nothing has
        accrued, which is a different and false claim.'
        """
        row = _row("", entry=_entry("6M"))
        row.row["coupon"] = "0"

        accrued = normalize_position(row).accrued_interest
        assert accrued.status == "unavailable"
        assert accrued.value != 0.0

    def test_structural_zero_is_distinguishable_from_a_converted_zero(self):
        """Both are `0.0`; only `provenance` says why. A consumer auditing
        'which zeros are real?' needs that."""
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
        """Unlike accrued interest, quantity has no 'legitimately absent'
        interpretation -- a position with no size is malformed."""
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
        """A `Quantity` that is not ok must not carry a number for a
        consumer to accidentally read."""
        q = Quantity.unavailable(ACCRUED_NOT_SUPPLIED)
        assert q.value is None
        assert not q.is_ok

    def test_ok_carries_a_provenance(self):
        q = Quantity.ok(1.0, CONVERTED)
        assert q.is_ok and q.provenance == CONVERTED


class TestRoundTripThroughTheResult:
    """Plan §W0.3's last named test: 'round-trip `mappingVersion`
    present'."""

    def test_mapping_version_reaches_the_published_result(self):
        from engine.integration import price_bundle
        result = price_bundle(FIXTURES / "note" / "v2").to_dict()

        assert result["mappingVersion"] == MAPPING_VERSION
        for item in result["items"]:
            assert item["mappingVersion"] == MAPPING_VERSION

    def test_accrued_echoes_its_source_units(self):
        from engine.integration import price_bundle
        result = price_bundle(FIXTURES / "note" / "v2").to_dict()
        accrued = result["items"][0]["calculations"]["accruedInterest"]

        assert accrued["status"] == "ok"
        assert accrued["observedCleanPrice"] == pytest.approx(1.015)
        assert accrued["signedFaceAmount"] == pytest.approx(100_000.0)
        assert accrued["provenance"] == CONVERTED
