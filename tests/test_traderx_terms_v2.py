"""
`traderx.instrument-terms.v2` and its `accrualBasis` (`engine.traderx.terms`).

v2 is accepted, including the optional `accrualBasis`, and its `fractionDecimals` really
sets the reconciliation tolerance. Unrecognized values are refused rather than read with v1
meaning (an assumption, I-23). Delivered v1 results are unchanged.
"""
import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from engine.traderx.bundle import load_bundle
from engine.traderx.terms import (
    ACCRUAL_BASIS_SCHEMA_V1,
    MAX_FRACTION_DECIMALS,
    MIN_FRACTION_DECIMALS,
    SUPPORTED_ACCRUAL_BASIS_SCHEMAS,
    SUPPORTED_TERMS_SCHEMAS,
    TERMS_SCHEMA_V1,
    TERMS_SCHEMA_V2,
    AccrualBasis,
    TermsJoinError,
    join_terms,
)

FIXTURES = Path(__file__).parent / "fixtures" / "traderx-eod"

VALID_BASIS = {
    "schema": ACCRUAL_BASIS_SCHEMA_V1,
    "dateBasis": "SESSION_DATE",
    "settlementAdjustment": "NONE",
    "rounding": "HALF_EVEN",
    "fractionDecimals": 6,
}


@pytest.fixture
def terms_bundle(tmp_path):
    """Copy a fixture bundle, let a test rewrite its terms, and re-pin the manifest hash so
    the join (not the hash check) is under test."""

    def _make(case: str, mutate=None, version: str = "v2"):
        root = tmp_path / f"{case}-{version}-{abs(hash(str(mutate)))%10000}"
        shutil.copytree(FIXTURES / case / version, root)
        if mutate is not None:
            terms_path = root / "instrument-terms.json"
            terms = json.loads(terms_path.read_bytes())
            mutate(terms)
            raw = json.dumps(terms, indent=2, sort_keys=True).encode("utf-8") + b"\n"
            terms_path.write_bytes(raw)

            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest["artifacts"]["instrumentTerms"]["sha256"] = hashlib.sha256(raw).hexdigest()
            manifest["artifacts"]["instrumentTerms"]["entries"] = len(terms["entries"])
            manifest_path.write_bytes(
                json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
            )
        return load_bundle(root)

    return _make


def _to_v2(terms, basis=None):
    """Relabel a fixture's terms as v2, optionally with a basis."""
    terms["schema"] = TERMS_SCHEMA_V2
    if basis is not None:
        for entry in terms["entries"]:
            entry["accrualBasis"] = copy.deepcopy(basis)


class TestV2IsAccepted:
    """The v2 schema joins, with and without a basis."""

    def test_v2_schema_label_alone_joins(self, terms_bundle):
        bundle = terms_bundle("note", lambda t: _to_v2(t))
        join = join_terms(bundle)
        assert join.has_terms_artifact
        assert all(r.is_joined for r in join.rows)

    def test_v2_with_accrual_basis_joins(self, terms_bundle):
        bundle = terms_bundle("note", lambda t: _to_v2(t, VALID_BASIS))
        entry = join_terms(bundle).joined[0].entry
        assert entry.accrual_basis is not None
        assert entry.accrual_basis.fraction_decimals == 6
        assert entry.accrual_basis.date_basis == "SESSION_DATE"
        assert entry.accrual_basis.rounding == "HALF_EVEN"

    def test_v2_without_basis_is_legal_and_carries_none(self, terms_bundle):
        """No basis is legal ("not stated"), not an error and not defaulted."""
        bundle = terms_bundle("note", lambda t: _to_v2(t))
        assert join_terms(bundle).joined[0].entry.accrual_basis is None

    def test_terms_schema_is_recorded_on_the_entry(self, terms_bundle):
        """The entry records its terms schema (to tell v1 from v2 without a basis)."""
        v1 = terms_bundle("note")
        v2 = terms_bundle("note", lambda t: _to_v2(t))
        assert join_terms(v1).joined[0].entry.terms_schema == TERMS_SCHEMA_V1
        assert join_terms(v2).joined[0].entry.terms_schema == TERMS_SCHEMA_V2
        # Neither has a basis; only the schema field tells them apart.
        assert join_terms(v1).joined[0].entry.accrual_basis is None
        assert join_terms(v2).joined[0].entry.accrual_basis is None

    def test_both_schemas_are_advertised_as_supported(self):
        assert TERMS_SCHEMA_V1 in SUPPORTED_TERMS_SCHEMAS
        assert TERMS_SCHEMA_V2 in SUPPORTED_TERMS_SCHEMAS


class TestTermsVersionIsIndependentOfBundleVersion:
    """A v2 bundle may carry either terms version."""

    def test_v2_bundle_with_v1_terms_joins(self, terms_bundle):
        bundle = terms_bundle("note")  # v2 bundle, v1 terms as delivered
        assert bundle.schema.endswith("v2")
        assert join_terms(bundle).joined[0].entry.terms_schema == TERMS_SCHEMA_V1

    def test_v2_bundle_with_v2_terms_joins(self, terms_bundle):
        bundle = terms_bundle("note", lambda t: _to_v2(t, VALID_BASIS))
        assert bundle.schema.endswith("v2")
        assert join_terms(bundle).joined[0].entry.terms_schema == TERMS_SCHEMA_V2


class TestUnrecognizedValuesAreRefused:
    """Unrecognized values are refused (I-23); read with v1 meaning, `dateBasis` would
    reconcile accrued against the wrong date."""

    @pytest.mark.parametrize("field,value", [
        ("dateBasis", "TRADE_DATE"),
        ("dateBasis", "SETTLEMENT_DATE"),
        ("settlementAdjustment", "FOLLOWING"),
        ("settlementAdjustment", "T+1"),
        ("rounding", "HALF_UP"),
        ("rounding", "TRUNCATE"),
    ])
    def test_unknown_enum_value_is_refused(self, terms_bundle, field, value):
        basis = dict(VALID_BASIS, **{field: value})
        with pytest.raises(TermsJoinError) as exc:
            join_terms(terms_bundle("note", lambda t: _to_v2(t, basis)))
        assert field in str(exc.value)
        assert value in str(exc.value)

    def test_future_accrual_basis_schema_is_refused(self, terms_bundle):
        """An `accrual-basis.v2` is refused."""
        basis = dict(VALID_BASIS, schema="traderx.accrual-basis.v2")
        with pytest.raises(TermsJoinError, match="accrual-basis"):
            join_terms(terms_bundle("note", lambda t: _to_v2(t, basis)))

    def test_unknown_terms_schema_is_refused(self, terms_bundle):
        with pytest.raises(TermsJoinError, match="unsupported terms schema"):
            join_terms(terms_bundle(
                "note", lambda t: t.__setitem__("schema", "traderx.instrument-terms.v3"),
            ))

    @pytest.mark.parametrize("missing", [
        "dateBasis", "settlementAdjustment", "rounding", "fractionDecimals",
    ])
    def test_missing_required_basis_key_is_refused(self, terms_bundle, missing):
        basis = {k: v for k, v in VALID_BASIS.items() if k != missing}
        with pytest.raises(TermsJoinError, match=missing):
            join_terms(terms_bundle("note", lambda t: _to_v2(t, basis)))

    def test_basis_that_is_not_an_object_is_refused(self, terms_bundle):
        def mutate(t):
            t["schema"] = TERMS_SCHEMA_V2
            t["entries"][0]["accrualBasis"] = "SESSION_DATE"
        with pytest.raises(TermsJoinError, match="must be an object"):
            join_terms(terms_bundle("note", mutate))


class TestFractionDecimalsIsValidated:
    """`fractionDecimals` scales the tolerance, so it is validated where it is parsed."""

    @pytest.mark.parametrize("bad", [0, -1, 13, 100])
    def test_out_of_range_is_refused(self, terms_bundle, bad):
        basis = dict(VALID_BASIS, fractionDecimals=bad)
        with pytest.raises(TermsJoinError, match="fractionDecimals"):
            join_terms(terms_bundle("note", lambda t: _to_v2(t, basis)))

    @pytest.mark.parametrize("bad", ["6", 6.0, None, [6]])
    def test_non_integer_is_refused(self, terms_bundle, bad):
        basis = dict(VALID_BASIS, fractionDecimals=bad)
        with pytest.raises(TermsJoinError, match="fractionDecimals"):
            join_terms(terms_bundle("note", lambda t: _to_v2(t, basis)))

    def test_bool_is_refused_despite_being_an_int_subclass(self, terms_bundle):
        """`True` (an int subclass) is refused; it would read as 1 decimal."""
        basis = dict(VALID_BASIS, fractionDecimals=True)
        with pytest.raises(TermsJoinError, match="fractionDecimals"):
            join_terms(terms_bundle("note", lambda t: _to_v2(t, basis)))

    @pytest.mark.parametrize("good", [MIN_FRACTION_DECIMALS, 6, MAX_FRACTION_DECIMALS])
    def test_in_range_is_accepted(self, terms_bundle, good):
        basis = dict(VALID_BASIS, fractionDecimals=good)
        entry = join_terms(terms_bundle("note", lambda t: _to_v2(t, basis))).joined[0].entry
        assert entry.accrual_basis.fraction_decimals == good


class TestSelfContradictoryDocumentsAreRefused:
    """A v1-labelled artifact carrying a v2-only field contradicts its own version."""

    def test_v1_schema_carrying_an_accrual_basis_is_refused(self, terms_bundle):
        def mutate(t):
            # schema left at v1
            t["entries"][0]["accrualBasis"] = copy.deepcopy(VALID_BASIS)
        with pytest.raises(TermsJoinError, match="contradicts its own version"):
            join_terms(terms_bundle("note", mutate))


class TestV1BundlesAreUnchanged:
    """Delivered v1 results are unchanged."""

    def test_delivered_v1_terms_still_join(self, terms_bundle):
        join = join_terms(terms_bundle("note"))
        assert join.has_terms_artifact
        assert all(r.is_joined for r in join.rows)
        assert join.joined[0].entry.accrual_basis is None

    @pytest.mark.parametrize("case", ["note", "bill", "sofr", "equity"])
    def test_every_delivered_fixture_still_joins_unchanged(self, case):
        """Fixtures straight off disk still join unchanged."""
        join = join_terms(load_bundle(FIXTURES / case / "v2"))
        assert join.has_terms_artifact
        for row in join.joined:
            assert row.entry.terms_schema == TERMS_SCHEMA_V1
            assert row.entry.accrual_basis is None

    def test_missing_terms_still_carried_verbatim(self):
        """The SOFR fixture's 13 missing terms are still carried verbatim."""
        join = join_terms(load_bundle(FIXTURES / "sofr" / "v2"))
        entries = [r.entry for r in join.joined if r.entry.missing_terms]
        assert entries, "the SOFR fixture should carry missing terms"
        assert len(entries[0].missing_terms) == 13


class TestFractionDecimalsActuallyReachesTheTolerance:
    """`fractionDecimals` sets the tolerance through the pipeline (a parser that validated
    the block and ignored it once passed every other test).

    The note's two accrual paths differ by $0.04 (the exporter's rounding). The tolerance at
    100,000 face:

        3 decimals -> 50.01     6 decimals -> 0.06
        4 decimals ->  5.01     8 decimals -> 0.0105

    At 8 decimals $0.04 exceeds it, so the note is refused with `ACCRUAL_MISMATCH`.
    """

    MARKET = {"mode": "assumed-profile", "assumedProfileId": "flat-3pct-v1"}

    def _price(self, tmp_path, decimals):
        import hashlib
        import json
        import shutil

        from engine.traderx import price_bundle

        root = tmp_path / f"note-decimals-{decimals}"
        shutil.copytree(FIXTURES / "note" / "v2", root)

        terms_path = root / "instrument-terms.json"
        terms = json.loads(terms_path.read_bytes())
        _to_v2(terms, dict(VALID_BASIS, fractionDecimals=decimals))
        raw = json.dumps(terms, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        terms_path.write_bytes(raw)

        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["artifacts"]["instrumentTerms"]["sha256"] = hashlib.sha256(raw).hexdigest()
        manifest_path.write_bytes(
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )
        return price_bundle(root, self.MARKET).to_dict()

    def test_declared_precision_of_six_still_prices(self, tmp_path):
        """Control: 6 decimals (the default and the exporter's precision) prices."""
        result = self._price(tmp_path, 6)
        values = sorted(i["calculations"]["npv"]["value"] for i in result["items"])
        assert values[1] == pytest.approx(103308.33, abs=0.01)

    def test_tighter_declared_precision_refuses_the_reconciliation(self, tmp_path):
        """At 8 decimals the note is refused (fails against a pipeline ignoring the field)."""
        result = self._price(tmp_path, 8)
        for item in result["items"]:
            npv = item["calculations"]["npv"]
            assert npv["status"] != "ok", (
                "an 8-decimal declared precision makes the tolerance 0.0105, which "
                "the observed $0.04 difference exceeds -- pricing anyway means the "
                "declared precision was ignored"
            )
            assert npv["reason"] == "ACCRUAL_MISMATCH"

    def test_looser_declared_precision_still_prices(self, tmp_path):
        """Fewer decimals still price (so refusal is not triggered by any basis)."""
        result = self._price(tmp_path, 4)
        values = sorted(i["calculations"]["npv"]["value"] for i in result["items"])
        assert values[1] == pytest.approx(103308.33, abs=0.01)

    def test_the_tolerance_itself_moves_with_declared_precision(self):
        """The tolerance moves with the declared precision."""
        from engine.traderx.note import accrual_mismatch_tolerance

        assert accrual_mismatch_tolerance(100000.0, 6) == pytest.approx(0.06)
        assert accrual_mismatch_tolerance(100000.0, 8) < 0.04
        assert accrual_mismatch_tolerance(100000.0, 4) > 0.04


class TestAccrualBasisType:
    """The dataclass itself."""

    def test_round_trips_to_dict(self):
        basis = AccrualBasis(
            date_basis="SESSION_DATE", settlement_adjustment="NONE",
            rounding="HALF_EVEN", fraction_decimals=6,
        )
        assert basis.to_dict() == VALID_BASIS

    def test_is_frozen(self):
        basis = AccrualBasis("SESSION_DATE", "NONE", "HALF_EVEN", 6)
        with pytest.raises(Exception):
            basis.fraction_decimals = 8

    def test_only_v1_basis_schema_is_supported(self):
        assert SUPPORTED_ACCRUAL_BASIS_SCHEMAS == (ACCRUAL_BASIS_SCHEMA_V1,)
