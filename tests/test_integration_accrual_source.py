"""
W1.6.3 -- `accrualSource` on the standalone `accruedInterest` outcome
(`docs/planning/traderx-integration-plan.md` §W1.6).

**The problem this closes.** Before W1.6.3 a consumer reading the standalone
`accruedInterest` outcome saw `provenance: "converted"`, while the note's
NPV payload described the same fact as `accrualSource: "exported-fraction"`.
Two vocabularies for one thing, neither cross-referenced, so reconciling the
standalone value against the priced one required knowing that "converted"
and "exported-fraction" meant the same thing.

**The constraint that makes this non-trivial.** The plan is explicit:
"Keep `structural-zero` distinct from an exported-fraction conversion --
that distinction is W0.3's whole point and must survive the alignment."

A bill's zero is not an exported fraction that happened to be zero. It is
zero because the instrument has no coupon schedule. Folding it into
`exported-fraction` would erase exactly the distinction W0.3 exists to
preserve -- the one between a bill and a coupon-bearing note whose accrual
the exporter omitted. `TestStructuralZeroSurvivesTheAlignment` is therefore
the class that matters most here, and the naive implementation ("label
everything `exported-fraction`") fails it.
"""
from pathlib import Path

import pytest

from engine.integration import price_bundle
from engine.integration.normalize import CONVERTED, STRUCTURAL_ZERO
from engine.integration.note import ACCRUAL_EXPORTED, ACCRUAL_RECOMPUTED
from engine.integration.pipeline import _accrual_source

FIXTURES = Path(__file__).parent / "fixtures" / "traderx-eod"
MARKET = {"mode": "assumed-profile", "assumedProfileId": "flat-3pct-v1"}


def _accrued(case, version="v2", market=MARKET):
    result = price_bundle(FIXTURES / case / version, market).to_dict()
    return [item["calculations"]["accruedInterest"] for item in result["items"]]


@pytest.fixture
def blank_accrued_bundle(tmp_path):
    """A coupon-bearing note whose `accruedInterestFraction` is blank.

    This is the W0.3 case that must never become `0.0`: the terms say the
    instrument pays coupons, so a blank is missing data rather than a
    structural zero. Built by blanking the column and re-pinning the
    manifest hash, so the *normalization* is under test rather than the
    hash check.
    """
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
        """The alignment, asserted directly: the same fact described the
        same way in both places."""
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
    """**The distinction W0.3 exists to preserve.**

    A bill's accrued interest is zero *structurally*. Labelling it
    `exported-fraction` would say the exporter supplied a fraction that
    happened to be zero -- a different claim, and a false one.
    """

    def test_bill_accrued_is_labelled_structural_zero(self):
        for outcome in _accrued("bill"):
            assert outcome["value"] == 0.0
            assert outcome["accrualSource"] == STRUCTURAL_ZERO

    def test_bill_is_not_labelled_exported_fraction(self):
        """Asserted negatively and explicitly: this is the exact collapse
        the naive implementation makes."""
        for outcome in _accrued("bill"):
            assert outcome["accrualSource"] != ACCRUAL_EXPORTED

    def test_bill_and_note_labels_are_different(self):
        """If these two ever agree, the distinction has been lost."""
        bill = {o["accrualSource"] for o in _accrued("bill")}
        note = {o["accrualSource"] for o in _accrued("note")}
        assert bill.isdisjoint(note)

    def test_provenance_is_retained_alongside_the_new_label(self):
        """W1.6.3 adds a field; it does not replace one. A consumer pinned
        to `provenance` keeps working."""
        for outcome in _accrued("bill"):
            assert outcome["provenance"] == STRUCTURAL_ZERO
        for outcome in _accrued("note"):
            assert outcome["provenance"] == CONVERTED


class TestMappingFunction:
    """`_accrual_source` in isolation, including the cases the fixtures do
    not reach."""

    def test_converted_maps_to_exported_fraction(self):
        assert _accrual_source(CONVERTED) == ACCRUAL_EXPORTED

    def test_structural_zero_maps_to_itself(self):
        assert _accrual_source(STRUCTURAL_ZERO) == STRUCTURAL_ZERO

    def test_unknown_provenance_yields_no_label(self):
        """An absent label is honest; an invented one is not."""
        assert _accrual_source("something-else") is None

    def test_none_provenance_yields_no_label(self):
        assert _accrual_source(None) is None


class TestUnavailableAccrualCarriesNoLabel:
    """A calculation with no value must not carry a source label -- there is
    no source, and a label would imply one.

    Note the v1 *note* fixture is not this case: it supplies
    `accruedInterestFraction` in the CSV, so it converts normally even
    without a terms artifact. The `unavailable` state needs a genuinely
    blank field, which is what `_blank_accrued` below produces.
    """

    def test_v1_bundle_with_a_supplied_fraction_still_converts(self):
        """A v1 bundle has no terms artifact, but a *supplied* fraction is
        still interpretable -- the blank is what is uninterpretable, not the
        missing artifact on its own."""
        for outcome in _accrued("note", version="v1"):
            assert outcome["status"] == "ok"
            assert outcome["accrualSource"] == ACCRUAL_EXPORTED

    def test_blank_accrued_is_unavailable_without_a_label(self, blank_accrued_bundle):
        for outcome in blank_accrued_bundle:
            assert outcome["status"] == "unavailable"
            assert "accrualSource" not in outcome

    def test_blank_accrued_outcome_carries_no_value(self, blank_accrued_bundle):
        """A non-ok outcome must not carry a number -- attaching one invites
        consumers to read it."""
        for outcome in blank_accrued_bundle:
            assert "value" not in outcome


class TestExistingBehaviourUnchanged:
    """W1.6.3 is additive. The delivered numbers must not have moved."""

    def test_note_accrued_value_is_unchanged(self):
        """TraderX independently verified +/-1,857.10."""
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
