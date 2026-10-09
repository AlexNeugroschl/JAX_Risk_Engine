"""
`accrualSource` on the standalone `accruedInterest` outcome (W1.6.3,
`docs/planning/details/traderx-integration.md`).

The standalone outcome uses the same vocabulary as the note's NPV payload
(`exported-fraction`), while a bill keeps `structural-zero`: its zero comes from having no
coupon schedule, not from an exported fraction that happened to be zero (the W0.3
distinction). `TestStructuralZeroSurvivesTheAlignment` guards that.
"""
from pathlib import Path

import pytest

from engine.traderx import price_bundle
from engine.traderx.normalize import CONVERTED, STRUCTURAL_ZERO
from engine.traderx.note import ACCRUAL_EXPORTED, ACCRUAL_RECOMPUTED
from engine.traderx.pipeline import _accrual_source

FIXTURES = Path(__file__).parent / "fixtures" / "traderx-eod"
MARKET = {"mode": "assumed-profile", "assumedProfileId": "flat-3pct-v1"}


def _accrued(case, version="v2", market=MARKET):
    result = price_bundle(FIXTURES / case / version, market).to_dict()
    return [item["calculations"]["accruedInterest"] for item in result["items"]]


@pytest.fixture
def blank_accrued_bundle(tmp_path):
    """A coupon-bearing note with a blank `accruedInterestFraction` (missing data, never
    0.0). The column is blanked and the manifest hash re-pinned, so normalization rather
    than the hash check is under test."""
    import hashlib
    import json
    import shutil

    root = tmp_path / "note-blank-accrued"
    shutil.copytree(FIXTURES / "note" / "v2", root)

    positions = root / "positions.csv"
    text = positions.read_bytes().decode("utf-8")
    lines = text.split("\n")
    header_idx = next(i for i, l in enumerate(lines) if l.startswith("accountId,"))
    columns = lines[header_idx].split(",")
    col = columns.index("accruedInterestFraction")
    for i in range(header_idx + 1, len(lines)):
        if not lines[i].strip():
            continue
        fields = lines[i].split(",")
        if len(fields) == len(columns):
            fields[col] = ""
            lines[i] = ",".join(fields)
    raw = "\n".join(lines).encode("utf-8")
    positions.write_bytes(raw)

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["artifacts"]["positions"]["sha256"] = hashlib.sha256(raw).hexdigest()
    manifest_path.write_bytes(
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )

    result = price_bundle(root, MARKET).to_dict()
    return [item["calculations"]["accruedInterest"] for item in result["items"]]


class TestLabelIsPresentAndAligned:
    def test_note_standalone_accrued_carries_accrual_source(self):
        for outcome in _accrued("note"):
            assert outcome["status"] == "ok"
            assert outcome["accrualSource"] == ACCRUAL_EXPORTED

    def test_note_standalone_label_matches_the_npv_payload(self):
        """The standalone label equals the NPV payload's."""
        result = price_bundle(FIXTURES / "note" / "v2", MARKET).to_dict()
        for item in result["items"]:
            standalone = item["calculations"]["accruedInterest"]["accrualSource"]
            reconciliation = item["calculations"]["npv"]["accrualReconciliation"]
            assert standalone == reconciliation["accrualSource"]

    def test_label_travels_on_both_long_and_short(self):
        outcomes = _accrued("note")
        assert len(outcomes) == 2
        values = [o["value"] for o in outcomes]
        assert values[0] == pytest.approx(-values[1])
        assert {o["accrualSource"] for o in outcomes} == {ACCRUAL_EXPORTED}


class TestStructuralZeroSurvivesTheAlignment:
    """A bill's accrued is a structural zero, not an exported fraction of zero."""

    def test_bill_accrued_is_labelled_structural_zero(self):
        for outcome in _accrued("bill"):
            assert outcome["value"] == 0.0
            assert outcome["accrualSource"] == STRUCTURAL_ZERO

    def test_bill_is_not_labelled_exported_fraction(self):
        """The collapse a naive implementation makes."""
        for outcome in _accrued("bill"):
            assert outcome["accrualSource"] != ACCRUAL_EXPORTED

    def test_bill_and_note_labels_are_different(self):
        """If these agree the distinction is lost."""
        bill = {o["accrualSource"] for o in _accrued("bill")}
        note = {o["accrualSource"] for o in _accrued("note")}
        assert bill.isdisjoint(note)

    def test_provenance_is_retained_alongside_the_new_label(self):
        """The label is added alongside `provenance`, which is kept."""
        for outcome in _accrued("bill"):
            assert outcome["provenance"] == STRUCTURAL_ZERO
        for outcome in _accrued("note"):
            assert outcome["provenance"] == CONVERTED


class TestMappingFunction:
    """`_accrual_source` directly, including cases the fixtures do not reach."""

    def test_converted_maps_to_exported_fraction(self):
        assert _accrual_source(CONVERTED) == ACCRUAL_EXPORTED

    def test_structural_zero_maps_to_itself(self):
        assert _accrual_source(STRUCTURAL_ZERO) == STRUCTURAL_ZERO

    def test_unknown_provenance_yields_no_label(self):
        """No label rather than an invented one."""
        assert _accrual_source("something-else") is None

    def test_none_provenance_yields_no_label(self):
        assert _accrual_source(None) is None


class TestUnavailableAccrualCarriesNoLabel:
    """A value-less calculation carries no source label. (The v1 note fixture supplies the
    fraction in its CSV and converts normally; the unavailable case needs a blank field.)"""

    def test_v1_bundle_with_a_supplied_fraction_still_converts(self):
        """With no terms artifact, a supplied fraction still converts; only a blank is
        uninterpretable."""
        for outcome in _accrued("note", version="v1"):
            assert outcome["status"] == "ok"
            assert outcome["accrualSource"] == ACCRUAL_EXPORTED

    def test_blank_accrued_is_unavailable_without_a_label(self, blank_accrued_bundle):
        for outcome in blank_accrued_bundle:
            assert outcome["status"] == "unavailable"
            assert "accrualSource" not in outcome

    def test_blank_accrued_outcome_carries_no_value(self, blank_accrued_bundle):
        """A non-ok outcome carries no number."""
        for outcome in blank_accrued_bundle:
            assert "value" not in outcome


class TestExistingBehaviourUnchanged:
    """The delivered numbers are unchanged."""

    def test_note_accrued_value_is_unchanged(self):
        """Â±1,857.10: the exported fraction 0.018571 on 100,000 face."""
        values = sorted(o["value"] for o in _accrued("note"))
        assert values[0] == pytest.approx(-1857.10, abs=1e-6)
        assert values[1] == pytest.approx(1857.10, abs=1e-6)

    def test_note_npv_is_unchanged(self):
        result = price_bundle(FIXTURES / "note" / "v2", MARKET).to_dict()
        values = sorted(i["calculations"]["npv"]["value"] for i in result["items"])
        assert values[1] == pytest.approx(103308.33, abs=0.01)

    def test_bill_npv_is_unchanged(self):
        result = price_bundle(FIXTURES / "bill" / "v2", MARKET).to_dict()
        values = sorted(i["calculations"]["npv"]["value"] for i in result["items"])
        assert values[1] == pytest.approx(98507.15, abs=0.01)
