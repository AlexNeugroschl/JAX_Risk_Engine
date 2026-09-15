"""
W0.2 -- the `instrument-terms.json` join
(`docs/planning/traderx-integration-plan.md` §W0.2).
"""
import copy
import json
import shutil
from pathlib import Path

import pytest

from engine.integration.bundle import load_bundle
from engine.integration.terms import (
    NO_TERMS_ARTIFACT,
    TERMS_ENTRY_NOT_FOUND,
    TermsJoinError,
    join_terms,
)

FIXTURES = Path(__file__).parent / "fixtures" / "traderx-eod"


@pytest.fixture
def terms_bundle(tmp_path):
    """Factory: copies a fixture bundle and lets a test rewrite its terms
    artifact, re-pinning the manifest hash so the *join* is under test
    rather than the hash check."""
    import hashlib

    def _make(case: str, mutate=None, version: str = "v2"):
        root = tmp_path / f"{case}-{version}"
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


class TestSecurityJoinAcrossAccounts:
    """Plan §W0.2 step 2 + its first named test: the same security across
    two accounts joins to ONE terms entry, with correct per-account signs.

    This is the shape that breaks a naive per-row terms lookup: the terms
    describe the instrument, the amounts and signs stay in the rows.
    """

    def test_one_entry_serves_both_accounts(self):
        join = join_terms(load_bundle(FIXTURES / "note" / "v2"))
        assert len(join.rows) == 2
        assert all(r.is_joined for r in join.rows)

        entries = {id(r.entry) for r in join.rows}
        assert len(entries) == 1, "both rows must join to the same terms entry object"

    def test_per_account_signs_are_opposite_and_preserved(self):
        join = join_terms(load_bundle(FIXTURES / "note" / "v2"))
        by_account = {r.row["accountId"]: r for r in join.rows}

        assert float(by_account["22214"].row["quantity"]) == 100_000.0
        assert float(by_account["42422"].row["quantity"]) == -100_000.0
        # Same instrument, same terms, opposite direction.
        assert by_account["22214"].entry is by_account["42422"].entry

    def test_every_joined_row_is_checked_not_only_the_first(self):
        """bundle-v2-and-terms.md: 'The validator checks each joined row,
        not only the first account.'"""
        join = join_terms(load_bundle(FIXTURES / "note" / "v2"))
        for row in join.rows:
            assert row.entry is not None
            assert row.entry.instrument_type == "TREASURY"


class TestContractJoin:
    """Contracts join by `contractId` + `clusterEpoch`."""

    def test_contract_joins_on_id_and_epoch(self):
        bundle = load_bundle(FIXTURES / "sofr" / "v2")
        join = join_terms(bundle)
        assert len(join.rows) == 1
        row = join.rows[0]
        assert row.is_joined
        assert row.row["contractId"] == "SW-3"
        assert row.entry.identity["clusterEpoch"] == bundle.cluster_epoch

    def test_wrong_epoch_is_rejected(self, terms_bundle):
        """A contract id is unique only within its epoch. An entry claiming
        a different one may describe a different contract entirely, so it
        is refused rather than joined."""
        def wrong_epoch(terms):
            terms["entries"][0]["identity"]["clusterEpoch"] = "some-other-epoch-v9"

        bundle = terms_bundle("sofr", wrong_epoch)
        with pytest.raises(TermsJoinError, match="but the bundle's epoch is"):
            join_terms(bundle)


class TestMissingTermsIsAuthoritative:
    """Plan §W0.2 step 3: `missingTerms` is an authoritative refusal input,
    carried **verbatim**, not a hint to be summarized or reordered."""

    def test_sofr_entry_carries_all_thirteen(self):
        join = join_terms(load_bundle(FIXTURES / "sofr" / "v2"))
        entry = join.rows[0].entry
        assert len(entry.missing_terms) == 13
        assert not entry.is_complete

    def test_carried_verbatim_and_in_supplied_order(self):
        """Order is preserved rather than sorted by this code: the exporter
        supplied a sorted list, and re-sorting would hide it if it ever
        stopped being sorted."""
        raw = json.loads((FIXTURES / "sofr" / "v2" / "instrument-terms.json").read_bytes())
        supplied = raw["entries"][0]["missingTerms"]

        join = join_terms(load_bundle(FIXTURES / "sofr" / "v2"))
        assert list(join.rows[0].entry.missing_terms) == supplied

    def test_complete_entries_report_complete(self):
        for case in ("bill", "note"):
            join = join_terms(load_bundle(FIXTURES / case / "v2"))
            for row in join.rows:
                assert row.entry.missing_terms == ()
                assert row.entry.is_complete

    def test_complete_does_not_mean_priceable(self):
        """bundle-v2-and-terms.md: an entry with no missing fields 'does not
        prove a model can represent it'. `is_complete` is a statement about
        the export, not about this engine."""
        join = join_terms(load_bundle(FIXTURES / "note" / "v2"))
        entry = join.rows[0].entry
        assert entry.is_complete
        # The priceability verdict belongs to W0.4, and is asked separately.
        from engine.integration.conventions import check_conventions
        assert check_conventions(join.rows[0]) is None  # conventions ok...
        # ...yet the pipeline still reports no NPV, because no pricer exists.
        from engine.integration import price_bundle
        result = price_bundle(FIXTURES / "note" / "v2")
        assert result.items[0].calculations["npv"].status == "unsupported"


class TestDuplicateAndUnjoinable:
    """Plan §W0.2 step 4."""

    def test_duplicate_identity_is_rejected(self, terms_bundle):
        def duplicate(terms):
            terms["entries"].append(copy.deepcopy(terms["entries"][0]))

        bundle = terms_bundle("note", duplicate)
        with pytest.raises(TermsJoinError, match="duplicates identity"):
            join_terms(bundle)

    def test_entry_matching_no_row_is_rejected(self, terms_bundle):
        """'Extra ... entries fail' -- an entry describing an instrument the
        bundle does not contain cannot be attributed to anything."""
        def extra(terms):
            spare = copy.deepcopy(terms["entries"][0])
            spare["identity"]["security"] = "UST-NOTE-NOT-IN-BUNDLE"
            terms["entries"].append(spare)

        bundle = terms_bundle("note", extra)
        with pytest.raises(TermsJoinError, match="matching no bundle row"):
            join_terms(bundle)

    def test_row_matching_no_entry_is_unjoined_not_raised(self, terms_bundle):
        """The asymmetry with the previous test, and it is deliberate: an
        unjoined ROW is still identifiable and gets an identified refusal,
        so it is carried. An unattributable ENTRY is not."""
        def rename(terms):
            terms["entries"][0]["identity"]["security"] = "UST-NOTE-SOMETHING-ELSE"

        bundle = terms_bundle("note", rename)
        # The renamed entry now matches nothing, which fails first.
        with pytest.raises(TermsJoinError):
            join_terms(bundle)

    def test_a_row_with_no_entry_is_refused_not_dropped(self, tmp_path):
        """The genuinely-unjoined row: one position has a terms entry, the
        other does not, and every entry still joins to something (so the
        "extra entries" check does not fire first).

        The row must survive as an *identified refusal* rather than vanish
        from the result -- a silently shrinking portfolio is the failure
        mode the coverage model exists to make impossible.
        """
        import hashlib

        root = tmp_path / "mixed"
        shutil.copytree(FIXTURES / "note" / "v2", root)
        path = root / "positions.csv"
        raw = path.read_bytes().replace(
            b"42422,UST-NOTE-20261215", b"42422,UST-NOTE-UNKNOWN"
        )
        path.write_bytes(raw)

        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["artifacts"]["positions"]["sha256"] = hashlib.sha256(raw).hexdigest()
        manifest_path.write_bytes(
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )

        join = join_terms(load_bundle(root))
        by_security = {r.row["security"]: r for r in join.rows}

        assert by_security["UST-NOTE-20261215"].is_joined
        assert not by_security["UST-NOTE-UNKNOWN"].is_joined
        assert by_security["UST-NOTE-UNKNOWN"].unjoined_reason == TERMS_ENTRY_NOT_FOUND

        from engine.integration import price_bundle
        result = price_bundle(root)
        assert len(result.items) == 2, "the unjoined row must not be dropped"

        outcomes = {
            item.identity.security: item.calculations["npv"] for item in result.items
        }
        assert outcomes["UST-NOTE-UNKNOWN"].reason == "TERMS_NOT_SUPPLIED"
        assert outcomes["UST-NOTE-20261215"].reason == "NO_PRICER_AT_THIS_STAGE"

    def test_unknown_identity_source_is_rejected(self, terms_bundle):
        def bad_source(terms):
            terms["entries"][0]["identity"]["source"] = "somewhere-else"

        bundle = terms_bundle("note", bad_source)
        with pytest.raises(TermsJoinError, match="unknown identity.source"):
            join_terms(bundle)

    def test_malformed_missing_terms_is_rejected(self, terms_bundle):
        def bad_missing(terms):
            terms["entries"][0]["missingTerms"] = "calendar"  # string, not list

        bundle = terms_bundle("note", bad_missing)
        with pytest.raises(TermsJoinError, match="must be a list of strings"):
            join_terms(bundle)

    def test_unsupported_terms_schema_is_rejected(self, terms_bundle):
        def bump(terms):
            terms["schema"] = "traderx.instrument-terms.v2"

        bundle = terms_bundle("note", bump)
        with pytest.raises(TermsJoinError, match="unsupported terms schema"):
            join_terms(bundle)


class TestV1HasNoTermsArtifact:
    """Plan §W0.2 step 5 and its named test: a v1 bundle makes every
    terms-dependent calculation `unsupported`."""

    @pytest.mark.parametrize("case", ("bill", "note", "sofr"))
    def test_every_row_is_unjoined_with_a_reason(self, case):
        join = join_terms(load_bundle(FIXTURES / case / "v1"))
        assert join.has_terms_artifact is False
        assert join.joined == ()
        for row in join.unjoined:
            assert row.entry is None
            assert row.unjoined_reason == NO_TERMS_ARTIFACT

    def test_v1_is_not_an_error(self):
        """v1 is a valid bundle version. It yields refusals, not
        exceptions -- conflating 'older version' with 'broken bundle' would
        make the coordinator retry something that cannot improve."""
        join = join_terms(load_bundle(FIXTURES / "note" / "v1"))
        assert len(join.rows) == 2  # the rows are all still there

    @pytest.mark.parametrize("case", ("bill", "note", "sofr"))
    def test_terms_dependent_calculations_are_unsupported(self, case):
        from engine.integration import price_bundle
        result = price_bundle(FIXTURES / case / "v1")
        assert result.items, "rows must still be reported, not dropped"
        for item in result.items:
            assert item.calculations["npv"].status == "unsupported"
            assert item.calculations["npv"].reason == "TERMS_NOT_SUPPLIED"

    def test_row_counts_match_v2(self):
        """The same population, refused rather than absent. A v1 bundle
        must not silently shrink the portfolio."""
        for case in ("bill", "note", "sofr"):
            v1 = join_terms(load_bundle(FIXTURES / case / "v1"))
            v2 = join_terms(load_bundle(FIXTURES / case / "v2"))
            assert len(v1.rows) == len(v2.rows)
