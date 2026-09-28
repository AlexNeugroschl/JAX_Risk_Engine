"""
Bundle ingestion and hash verification (`engine.integration.bundle`).

Fixtures are the delivered TraderX YU18 bundles, vendored under `tests/fixtures/traderx-eod/`
with their committed LF bytes. (A Windows checkout with `core.autocrlf=true` can
CRLF-translate the copies under `reference/traderX/`, which then fail verification;
`TestCrlfTranslation` checks that this is caught.)
"""
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from engine.integration.bundle import (
    Bundle,
    BundleIntegrityError,
    load_bundle,
)

FIXTURES = Path(__file__).parent / "fixtures" / "traderx-eod"
ALL_CASES = ("bill", "note", "sofr")


@pytest.fixture
def bundle_copy(tmp_path):
    """Factory: copy a fixture bundle into tmp_path so a test can corrupt it."""
    def _copy(case: str, version: str = "v2") -> Path:
        dest = tmp_path / f"{case}-{version}"
        shutil.copytree(FIXTURES / case / version, dest)
        return dest
    return _copy


def _rewrite_manifest(root: Path, mutate) -> None:
    """Apply `mutate` to the parsed manifest and write it back with LF endings (so the
    rewrite does not introduce CRLF)."""
    manifest = json.loads((root / "manifest.json").read_bytes())
    mutate(manifest)
    raw = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    (root / "manifest.json").write_bytes(raw)


class TestHappyPath:
    """Every delivered fixture loads and fully verifies."""

    @pytest.mark.parametrize("case", ALL_CASES)
    @pytest.mark.parametrize("version", ("v1", "v2"))
    def test_loads_and_verifies(self, case, version):
        bundle = load_bundle(FIXTURES / case / version)
        assert isinstance(bundle, Bundle)
        assert bundle.schema == f"traderx.eod-bundle.{version}"
        assert bundle.version == (2 if version == "v2" else 1)
        assert bundle.cluster_epoch == "synthetic-shared-examples-v1"
        assert bundle.session_date == "2025-06-02"

    @pytest.mark.parametrize("case", ALL_CASES)
    def test_v2_carries_terms_v1_does_not(self, case):
        """v2 carries a terms artifact; v1 has none."""
        assert load_bundle(FIXTURES / case / "v2").has_terms is True
        assert load_bundle(FIXTURES / case / "v1").has_terms is False
        assert load_bundle(FIXTURES / case / "v1").terms is None

    def test_row_counts_match_the_delivered_population(self):
        """bill/note carry two position rows (long and short) and no contracts; sofr the
        reverse."""
        note = load_bundle(FIXTURES / "note" / "v2")
        assert note.positions.row_count == 2
        assert note.contracts.row_count == 0

        sofr = load_bundle(FIXTURES / "sofr" / "v2")
        assert sofr.positions.row_count == 0
        assert sofr.contracts.row_count == 1

    def test_parses_rows_against_the_csv_header(self):
        note = load_bundle(FIXTURES / "note" / "v2")
        accounts = {row["accountId"] for row in note.positions.rows}
        assert accounts == {"22214", "42422"}
        for row in note.positions.rows:
            assert row["security"] == "UST-NOTE-20261215"
            assert row["instrumentType"] == "TREASURY"

    def test_preamble_is_parsed(self):
        note = load_bundle(FIXTURES / "note" / "v2")
        assert note.positions.preamble["rows"] == "2"
        assert note.positions.preamble["sessionDate"] == "2025-06-02"
        # The prose legends are kept too, for diagnostics.
        assert "treasuryZeroCoupon" in note.positions.preamble

    def test_manifest_sha256_is_the_byte_digest_not_the_bundle_id(self):
        """`manifest_sha256` is the digest of the manifest file; `bundleId` is TraderX's
        digest of the canonical manifest without bundleId. They differ."""
        bundle = load_bundle(FIXTURES / "note" / "v2")
        expected = hashlib.sha256((FIXTURES / "note" / "v2" / "manifest.json").read_bytes()).hexdigest()
        assert bundle.manifest_sha256 == expected
        assert bundle.manifest_sha256 != bundle.bundle_id


class TestEachArtifactAgainstItsOwnHash:
    """Each artifact is verified against its own sha256, never against `cutSha256`."""

    @pytest.mark.parametrize("artifact", ("positions.csv", "contracts.csv", "instrument-terms.json"))
    def test_tampered_byte_is_rejected(self, bundle_copy, artifact):
        root = bundle_copy("note", "v2")
        path = root / artifact
        raw = path.read_bytes()
        # Flip one byte in the last line, keeping length and structure.
        path.write_bytes(raw[:-2] + bytes([raw[-2] ^ 0x01]) + raw[-1:])

        with pytest.raises(BundleIntegrityError, match="hash mismatch"):
            load_bundle(root)

    def test_appending_whitespace_is_rejected(self, bundle_copy):
        """Even appended whitespace changes the terms artifact's hash."""
        root = bundle_copy("note", "v2")
        path = root / "instrument-terms.json"
        path.write_bytes(path.read_bytes() + b"\n")

        with pytest.raises(BundleIntegrityError, match="hash mismatch"):
            load_bundle(root)

    def test_artifact_is_not_checked_against_cut_sha256(self, bundle_copy):
        """Putting `cutSha256` in place of an artifact's own hash fails."""
        root = bundle_copy("note", "v2")
        cut_sha = json.loads((root / "manifest.json").read_bytes())["cut"]["cutSha256"]

        def swap_in_cut_sha(manifest):
            manifest["artifacts"]["positions"]["sha256"] = cut_sha
        _rewrite_manifest(root, swap_in_cut_sha)

        with pytest.raises(BundleIntegrityError, match="hash mismatch"):
            load_bundle(root)


class TestCrlfTranslation:
    """A CRLF-translated artifact fails, and the error names CRLF translation as the cause
    (a bare mismatch on every file is what confused TraderX's own verifier)."""

    @pytest.mark.parametrize("artifact", ("positions.csv", "instrument-terms.json"))
    def test_crlf_translated_artifact_fails(self, bundle_copy, artifact):
        root = bundle_copy("note", "v2")
        path = root / artifact
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))

        with pytest.raises(BundleIntegrityError) as exc:
            load_bundle(root)
        assert "hash mismatch" in str(exc.value)

    def test_failure_names_crlf_as_the_cause(self, bundle_copy):
        root = bundle_copy("note", "v2")
        path = root / "positions.csv"
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))

        with pytest.raises(BundleIntegrityError) as exc:
            load_bundle(root)
        message = str(exc.value)
        assert "CRLF" in message
        assert "autocrlf" in message
        assert ".gitattributes" in message
        assert "will not normalize" in message

    def test_loader_does_not_normalize_to_make_it_pass(self, bundle_copy):
        """The loader does not normalize line endings to make a CRLF file pass."""
        root = bundle_copy("note", "v2")
        for name in ("positions.csv", "contracts.csv", "instrument-terms.json"):
            p = root / name
            p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))

        with pytest.raises(BundleIntegrityError):
            load_bundle(root)

    def test_the_vendored_reference_checkout_is_actually_crlf_corrupted(self):
        """Why the fixtures are vendored: the reference checkout's copies are CRLF-translated
        here. Skips where that checkout is absent or has been fixed."""
        ref = (
            Path(__file__).parents[1] / "reference" / "traderX" / "specs"
            / "YU18-eod-risk-bundles" / "generation" / "runtime-overrides"
            / "eod-risk-bundles" / "tests" / "fixtures" / "shared" / "note" / "v2"
        )
        if not ref.is_dir():
            pytest.skip("reference/traderX checkout not present")
        if b"\r\n" not in (ref / "positions.csv").read_bytes():
            pytest.skip("reference checkout is not CRLF-translated (already fixed)")

        with pytest.raises(BundleIntegrityError) as exc:
            load_bundle(ref)
        assert "CRLF" in str(exc.value)

    def test_our_own_fixtures_are_committed_with_lf(self):
        """The vendored fixtures are LF (the `.gitattributes` guard)."""
        for case in ALL_CASES:
            for version in ("v1", "v2"):
                for path in (FIXTURES / case / version).iterdir():
                    assert b"\r\n" not in path.read_bytes(), (
                        f"{path} contains CRLF. Check .gitattributes "
                        f"(*.json -text / *.csv -text) and re-checkout."
                    )


class TestRowCountValidation:
    """Row counts."""

    def test_manifest_row_count_disagreeing_with_artifact_fails(self, bundle_copy):
        root = bundle_copy("note", "v2")

        def claim_three_rows(manifest):
            manifest["artifacts"]["positions"]["rows"] = 3
        _rewrite_manifest(root, claim_three_rows)

        with pytest.raises(BundleIntegrityError, match="declares rows=3"):
            load_bundle(root)

    def test_preamble_row_count_disagreeing_with_manifest_fails(self, bundle_copy):
        """A preamble count disagreeing with the manifest (a truncation that took the
        preamble with it) fails."""
        root = bundle_copy("note", "v2")
        path = root / "positions.csv"
        path.write_bytes(path.read_bytes().replace(b"# rows=2", b"# rows=9"))

        # Changing bytes fails the hash first; re-pin it to isolate the row-count check.
        def repin(manifest):
            manifest["artifacts"]["positions"]["sha256"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        _rewrite_manifest(root, repin)

        with pytest.raises(BundleIntegrityError, match="contradicts manifest"):
            load_bundle(root)

    def test_terms_entry_count_disagreeing_fails(self, bundle_copy):
        root = bundle_copy("note", "v2")

        def claim_two_entries(manifest):
            manifest["artifacts"]["instrumentTerms"]["entries"] = 2
        _rewrite_manifest(root, claim_two_entries)

        with pytest.raises(BundleIntegrityError, match="declares entries=2"):
            load_bundle(root)


class TestEmptyVersusMissing:
    """An empty contracts file (no OTC rows) is valid; a missing one is an integrity
    failure."""

    def test_empty_contracts_file_is_valid(self):
        """The bill and note bundles ship a contracts.csv with a preamble, a header and no
        rows."""
        bundle = load_bundle(FIXTURES / "note" / "v2")
        assert bundle.contracts.row_count == 0
        assert bundle.contracts.header  # header present
        assert bundle.contracts.preamble["contracts"] == "0"

    def test_empty_positions_file_is_valid(self):
        """The sofr bundle ships an empty positions file."""
        bundle = load_bundle(FIXTURES / "sofr" / "v2")
        assert bundle.positions.row_count == 0
        assert bundle.positions.header

    def test_missing_contracts_file_is_an_integrity_failure(self, bundle_copy):
        root = bundle_copy("note", "v2")
        (root / "contracts.csv").unlink()

        with pytest.raises(BundleIntegrityError, match="missing from the bundle"):
            load_bundle(root)

    def test_missing_terms_file_is_an_integrity_failure(self, bundle_copy):
        root = bundle_copy("note", "v2")
        (root / "instrument-terms.json").unlink()

        with pytest.raises(BundleIntegrityError, match="missing from the bundle"):
            load_bundle(root)

    def test_the_two_conditions_raise_distinguishable_messages(self, bundle_copy):
        """Empty loads; missing raises."""
        empty = load_bundle(FIXTURES / "note" / "v2")
        assert empty.contracts.row_count == 0

        root = bundle_copy("note", "v2")
        (root / "contracts.csv").unlink()
        with pytest.raises(BundleIntegrityError):
            load_bundle(root)


class TestManifestValidation:
    def test_unsupported_schema_is_rejected(self, bundle_copy):
        root = bundle_copy("note", "v2")

        def bump_schema(manifest):
            manifest["schema"] = "traderx.eod-bundle.v3"
        _rewrite_manifest(root, bump_schema)

        with pytest.raises(BundleIntegrityError, match="unsupported bundle schema"):
            load_bundle(root)

    def test_missing_manifest_is_rejected(self, tmp_path):
        with pytest.raises(BundleIntegrityError, match="missing"):
            load_bundle(tmp_path)

    def test_malformed_manifest_json_is_rejected(self, bundle_copy):
        root = bundle_copy("note", "v2")
        (root / "manifest.json").write_bytes(b"{not json")

        with pytest.raises(BundleIntegrityError, match="not valid UTF-8 JSON"):
            load_bundle(root)

    @pytest.mark.parametrize("key", ("bundleId", "clusterEpoch", "cut", "valuationTime"))
    def test_missing_required_manifest_key_is_rejected(self, bundle_copy, key):
        root = bundle_copy("note", "v2")
        _rewrite_manifest(root, lambda m: m.pop(key))

        with pytest.raises(BundleIntegrityError, match=f"missing required key '{key}'"):
            load_bundle(root)

    def test_v1_manifest_carrying_a_terms_artifact_is_rejected(self, bundle_copy):
        """A v1 manifest claiming a terms artifact is inconsistent and rejected."""
        root = bundle_copy("note", "v1")

        def add_terms(manifest):
            manifest["artifacts"]["instrumentTerms"] = {
                "path": "instrument-terms.json",
                "schema": "traderx.instrument-terms.v1",
                "sha256": "0" * 64,
                "entries": 1,
            }
        _rewrite_manifest(root, add_terms)

        with pytest.raises(BundleIntegrityError, match="v1 bundles have no terms artifact"):
            load_bundle(root)

    def test_v2_manifest_without_a_terms_artifact_is_rejected(self, bundle_copy):
        root = bundle_copy("note", "v2")
        _rewrite_manifest(root, lambda m: m["artifacts"].pop("instrumentTerms"))

        with pytest.raises(BundleIntegrityError, match="missing the required"):
            load_bundle(root)


class TestRequiredColumns:
    """Intact bytes with the wrong shape. `csv.DictReader` is silent about short and long
    rows, so without these checks the first symptom would be a bare KeyError far
    downstream."""

    def _repin(self, root: Path, name: str) -> None:
        path = root / f"{name}.csv"
        _rewrite_manifest(root, lambda m: m["artifacts"][name].__setitem__(
            "sha256", hashlib.sha256(path.read_bytes()).hexdigest()))

    def test_missing_header_column_is_rejected(self, bundle_copy):
        root = bundle_copy("note", "v2")
        path = root / "positions.csv"
        path.write_bytes(path.read_bytes().replace(b"accountId,security,", b"accountId,notSecurity,"))
        self._repin(root, "positions")

        with pytest.raises(BundleIntegrityError, match="missing required column"):
            load_bundle(root)

    def test_missing_contract_column_is_rejected(self, bundle_copy):
        root = bundle_copy("sofr", "v2")
        path = root / "contracts.csv"
        path.write_bytes(path.read_bytes().replace(b"contractId,accountId,", b"contractId,acct,"))
        self._repin(root, "contracts")

        with pytest.raises(BundleIntegrityError, match="missing required column"):
            load_bundle(root)

    def test_short_data_row_is_rejected(self, bundle_copy):
        root = bundle_copy("note", "v2")
        path = root / "positions.csv"
        lines = path.read_bytes().split(b"\n")
        lines[-2] = b"22214,UST-NOTE-20261215"  # truncated row
        path.write_bytes(b"\n".join(lines))
        self._repin(root, "positions")

        with pytest.raises(BundleIntegrityError, match="is short"):
            load_bundle(root)

    def test_long_data_row_is_rejected(self, bundle_copy):
        root = bundle_copy("note", "v2")
        path = root / "positions.csv"
        lines = path.read_bytes().split(b"\n")
        lines[-2] = lines[-2] + b",surplus"
        path.write_bytes(b"\n".join(lines))
        self._repin(root, "positions")

        with pytest.raises(BundleIntegrityError, match="more fields than the header"):
            load_bundle(root)

    def test_the_failure_is_not_a_bare_key_error(self, bundle_copy):
        """The error is the loader's own type and names the artifact and column."""
        from engine.integration import price_bundle
        root = bundle_copy("note", "v2")
        path = root / "positions.csv"
        path.write_bytes(path.read_bytes().replace(b"accountId,security,", b"accountId,notSecurity,"))
        self._repin(root, "positions")

        with pytest.raises(BundleIntegrityError) as exc:
            price_bundle(root)
        assert "positions.csv" in str(exc.value)
        assert "security" in str(exc.value)

    def test_valid_fixtures_still_pass(self):
        """The real artifacts still pass."""
        for case in ALL_CASES:
            for version in ("v1", "v2"):
                assert load_bundle(FIXTURES / case / version)


class TestPreambleAgreesWithManifest:
    """A manifest and an artifact describing different cuts are rejected."""

    def test_contradicting_cut_sha_is_rejected(self, bundle_copy):
        root = bundle_copy("note", "v2")
        path = root / "positions.csv"
        original = path.read_bytes()
        cut_sha = json.loads((root / "manifest.json").read_bytes())["cut"]["cutSha256"]
        path.write_bytes(original.replace(cut_sha.encode(), b"f" * 64))

        def repin(manifest):
            manifest["artifacts"]["positions"]["sha256"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        _rewrite_manifest(root, repin)

        with pytest.raises(BundleIntegrityError, match="contradicts manifest"):
            load_bundle(root)

    def test_contradicting_session_date_is_rejected(self, bundle_copy):
        root = bundle_copy("note", "v2")
        path = root / "positions.csv"
        path.write_bytes(path.read_bytes().replace(b"# sessionDate=2025-06-02", b"# sessionDate=2025-06-03"))

        def repin(manifest):
            manifest["artifacts"]["positions"]["sha256"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        _rewrite_manifest(root, repin)

        with pytest.raises(BundleIntegrityError, match="contradicts manifest"):
            load_bundle(root)
