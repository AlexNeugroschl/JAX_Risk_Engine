"""
EOD bundle ingestion and verification.

Reads a `traderx.eod-bundle.v1`/`.v2` bundle, verifies every artifact against its own
`sha256` (`manifest.artifacts.<name>.sha256`; `cut.cutSha256` identifies the consensus cut
and is a different thing), checks row counts and preambles against the manifest, and parses
the CSVs. Nothing downstream sees unverified bytes.

A contracts file with `rows=0` is valid (no OTC coverage); a missing artifact is an
integrity failure. v1 bundles have no terms artifact, so everything needing terms is refused
downstream.

Line endings: hashes are over committed bytes, and a Windows checkout with
`core.autocrlf=true` rewrites LF to CRLF in `.json`/`.csv`, failing every hash. Bytes are
read in binary and verified exactly as read, never normalized. On a mismatch,
`_verify_artifact` checks whether the LF-normalized bytes would match and, if so, names CRLF
translation and the `.gitattributes` fix in the error; the file is still rejected.
"""
import csv
import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

#: Manifest `schema` values this loader accepts.
SUPPORTED_BUNDLE_SCHEMAS = ("traderx.eod-bundle.v1", "traderx.eod-bundle.v2")

#: Artifacts every bundle must carry, whatever its version.
REQUIRED_ARTIFACTS = ("positions", "contracts")

#: Present only in v2; its absence in v2 or presence in v1 is a manifest inconsistency.
TERMS_ARTIFACT = "instrumentTerms"

#: CSV preamble fields that must agree with the manifest's `cut` block.
_CUT_PREAMBLE_FIELDS = {
    "consensusSequence": ("cut", "consensusSequence"),
    "sessionDate": ("cut", "sessionDate"),
    "priceSnapshotVersion": ("cut", "priceSnapshotVersion"),
    "cutSha256": ("cut", "cutSha256"),
}

#: Columns downstream code indexes by name, checked against the header at load time. A file
#: can hash correctly and still have the wrong shape, and `csv.DictReader` does not raise on
#: short or long rows.
_REQUIRED_POSITION_COLUMNS = ("accountId", "security", "quantity")
_REQUIRED_CONTRACT_COLUMNS = ("contractId", "accountId")


class BundleIntegrityError(Exception):
    """The bundle on disk is not the bundle the manifest describes (bad hash, missing
    artifact, row-count or preamble disagreement, unsupported schema, bad columns). One type:
    all of these are equally fatal."""


@dataclass(frozen=True)
class CsvArtifact:
    """One verified CSV artifact: preamble, header and rows (dicts, in file order; nothing
    downstream keys on the order)."""
    path: str
    sha256: str
    preamble: Dict[str, str]
    header: List[str]
    rows: List[Dict[str, str]]

    @property
    def row_count(self) -> int:
        return len(self.rows)


@dataclass(frozen=True)
class Bundle:
    """A fully verified bundle: every hash, row count and preamble matched."""
    root: Path
    manifest: Dict
    manifest_sha256: str
    positions: CsvArtifact
    contracts: CsvArtifact
    #: Parsed `instrument-terms.json`, or `None` for a v1 bundle (a terms artifact that
    #: fails to load raises instead).
    terms: Optional[Dict] = None
    terms_sha256: Optional[str] = None

    @property
    def schema(self) -> str:
        return self.manifest["schema"]

    @property
    def version(self) -> int:
        """2 if this bundle carries a terms artifact, else 1."""
        return 2 if self.schema.endswith(".v2") else 1

    @property
    def bundle_id(self) -> str:
        return self.manifest["bundleId"]

    @property
    def cluster_epoch(self) -> str:
        return self.manifest["clusterEpoch"]

    @property
    def session_date(self) -> str:
        return self.manifest["cut"]["sessionDate"]

    @property
    def valuation_time(self) -> str:
        return self.manifest["valuationTime"]

    @property
    def has_terms(self) -> bool:
        return self.terms is not None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_bytes(path: Path, label: str) -> bytes:
    """Read a file in binary (text mode would translate newlines before hashing). A missing
    file is a `BundleIntegrityError`."""
    try:
        return path.read_bytes()
    except FileNotFoundError as exc:
        raise BundleIntegrityError(
            f"{label}: required artifact is missing from the bundle: {path}. "
            f"A missing artifact is an integrity failure. (An artifact that "
            f"exists but declares rows=0 is valid and means zero coverage -- "
            f"these are different conditions and are not conflated.)"
        ) from exc


def _verify_artifact(raw: bytes, expected_sha256: str, label: str) -> None:
    """Verify `raw` against `expected_sha256` exactly as read. On mismatch, diagnose CRLF
    translation if the LF-normalized bytes would match; they never satisfy the check."""
    actual = _sha256(raw)
    if actual == expected_sha256:
        return

    detail = (
        f"{label}: artifact hash mismatch.\n"
        f"  expected {expected_sha256}\n"
        f"  actual   {actual}"
    )

    if b"\r\n" in raw:
        normalized = raw.replace(b"\r\n", b"\n")
        if _sha256(normalized) == expected_sha256:
            raise BundleIntegrityError(
                detail + "\n"
                "\n"
                "  The LF-normalized bytes DO match the expected hash: this file "
                "was CRLF-translated on checkout (git core.autocrlf=true), so the "
                "bytes on disk are not the committed bytes the manifest pins.\n"
                "\n"
                "  Fix the checkout, not the verifier -- add to .gitattributes:\n"
                "      *.json -text\n"
                "      *.csv  -text\n"
                "  then re-checkout the affected files "
                "(git rm --cached -r . && git reset --hard).\n"
                "\n"
                "  This loader will not normalize line endings for you: hashes are "
                "over committed bytes, and accepting rewritten bytes would defeat "
                "the purpose of verifying them."
            )

    raise BundleIntegrityError(
        detail + "\n"
        "\n  The bytes on disk are not the bytes this manifest pins. The bundle "
        "is immutable by contract, so this means corruption or tampering."
    )


def _parse_csv(raw: bytes, label: str) -> (Dict[str, str], List[str], List[Dict[str, str]]):
    """Split a TraderX CSV into its `# key=value` preamble, header and rows. Decoded as
    UTF-8 after hashing."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BundleIntegrityError(f"{label}: artifact is not valid UTF-8 ({exc})") from exc

    preamble: Dict[str, str] = {}
    body_lines: List[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            stripped = line[1:].strip()
            key, sep, value = stripped.partition("=")
            if sep:
                preamble[key.strip()] = value.strip()
            continue
        body_lines.append(line)

    # Drop trailing blank lines so a final newline is not an empty row.
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()

    if not body_lines:
        raise BundleIntegrityError(f"{label}: artifact has a preamble but no CSV header row")

    reader = csv.DictReader(io.StringIO("\n".join(body_lines)))
    header = list(reader.fieldnames or [])
    rows = [dict(row) for row in reader]
    return preamble, header, rows


def _check_preamble_against_manifest(artifact: CsvArtifact, manifest: Dict, label: str) -> None:
    """Every field the CSV preamble repeats from the manifest must agree; otherwise the
    directory mixes two exports."""
    for preamble_key, manifest_path in _CUT_PREAMBLE_FIELDS.items():
        if preamble_key not in artifact.preamble:
            continue  # not every artifact repeats every field
        expected = manifest
        for part in manifest_path:
            expected = expected.get(part, {}) if isinstance(expected, dict) else {}
        if not isinstance(expected, str):
            continue
        actual = artifact.preamble[preamble_key]
        if actual != expected:
            raise BundleIntegrityError(
                f"{label}: preamble {preamble_key}={actual!r} contradicts manifest "
                f"{'.'.join(manifest_path)}={expected!r}. The manifest and the artifact "
                f"describe different cuts."
            )


def _check_row_count(artifact: CsvArtifact, declared: int, label: str, preamble_key: str) -> None:
    """The manifest's row count, the preamble's count and the parsed rows must all agree
    (catching both a truncated file and a manifest for another export)."""
    if artifact.row_count != declared:
        raise BundleIntegrityError(
            f"{label}: manifest declares rows={declared} but the artifact contains "
            f"{artifact.row_count} data row(s)."
        )
    if preamble_key in artifact.preamble:
        try:
            preamble_count = int(artifact.preamble[preamble_key])
        except ValueError as exc:
            raise BundleIntegrityError(
                f"{label}: preamble {preamble_key}={artifact.preamble[preamble_key]!r} "
                f"is not an integer"
            ) from exc
        if preamble_count != declared:
            raise BundleIntegrityError(
                f"{label}: preamble {preamble_key}={preamble_count} contradicts manifest "
                f"rows={declared}."
            )


def _check_required_columns(artifact: CsvArtifact, required: tuple, label: str) -> None:
    """Required columns must be in the header, and no row may be short or carry surplus
    fields (both silent in `csv.DictReader`)."""
    missing = [column for column in required if column not in artifact.header]
    if missing:
        raise BundleIntegrityError(
            f"{label}: CSV header is missing required column(s) {missing}. "
            f"Present: {artifact.header}. The artifact hashed correctly, so this is "
            f"schema drift rather than corruption -- the bytes are intact but their "
            f"shape is not what this loader can consume."
        )

    for index, row in enumerate(artifact.rows):
        if None in row:
            raise BundleIntegrityError(
                f"{label}: data row {index} has more fields than the header declares "
                f"({len(artifact.header)}); surplus values {row[None]!r}."
            )
        blank = [column for column in required if row.get(column) is None]
        if blank:
            raise BundleIntegrityError(
                f"{label}: data row {index} is short -- no value for required "
                f"column(s) {blank}."
            )


def _load_csv_artifact(root: Path, manifest: Dict, name: str, preamble_count_key: str) -> CsvArtifact:
    artifacts = manifest.get("artifacts", {})
    if name not in artifacts:
        raise BundleIntegrityError(
            f"manifest.artifacts is missing required entry {name!r}. "
            f"Present: {sorted(artifacts)}"
        )
    entry = artifacts[name]
    for required_key in ("path", "sha256", "rows"):
        if required_key not in entry:
            raise BundleIntegrityError(
                f"manifest.artifacts.{name} is missing {required_key!r}"
            )

    label = f"{name} ({entry['path']})"
    path = root / entry["path"]
    raw = _read_bytes(path, label)
    _verify_artifact(raw, entry["sha256"], label)

    preamble, header, rows = _parse_csv(raw, label)
    artifact = CsvArtifact(
        path=entry["path"], sha256=entry["sha256"],
        preamble=preamble, header=header, rows=rows,
    )
    _check_row_count(artifact, int(entry["rows"]), label, preamble_count_key)
    _check_preamble_against_manifest(artifact, manifest, label)
    _check_required_columns(
        artifact,
        _REQUIRED_POSITION_COLUMNS if name == "positions" else _REQUIRED_CONTRACT_COLUMNS,
        label,
    )
    return artifact


def _load_terms_artifact(root: Path, manifest: Dict) -> (Optional[Dict], Optional[str]):
    """Load and verify `instrument-terms.json` for a v2 bundle (`(None, None)` for v1). The
    hash is checked before parsing, and the declared entry count against the parsed list."""
    artifacts = manifest.get("artifacts", {})
    schema = manifest["schema"]
    has_entry = TERMS_ARTIFACT in artifacts

    if schema.endswith(".v1"):
        if has_entry:
            raise BundleIntegrityError(
                f"manifest declares schema {schema} but carries an "
                f"{TERMS_ARTIFACT!r} artifact. v1 bundles have no terms artifact."
            )
        return None, None

    if not has_entry:
        raise BundleIntegrityError(
            f"manifest declares schema {schema} but is missing the required "
            f"{TERMS_ARTIFACT!r} artifact."
        )

    entry = artifacts[TERMS_ARTIFACT]
    for required_key in ("path", "sha256", "entries"):
        if required_key not in entry:
            raise BundleIntegrityError(
                f"manifest.artifacts.{TERMS_ARTIFACT} is missing {required_key!r}"
            )

    label = f"{TERMS_ARTIFACT} ({entry['path']})"
    raw = _read_bytes(root / entry["path"], label)
    _verify_artifact(raw, entry["sha256"], label)  # before parsing, deliberately

    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleIntegrityError(f"{label}: not valid UTF-8 JSON ({exc})") from exc

    declared = int(entry["entries"])
    actual = len(parsed.get("entries", []))
    if actual != declared:
        raise BundleIntegrityError(
            f"{label}: manifest declares entries={declared} but the artifact "
            f"contains {actual}."
        )
    return parsed, entry["sha256"]


def load_bundle(root) -> Bundle:
    """Read and fully verify the bundle at `root`; raise `BundleIntegrityError` on any
    discrepancy. There is no partially verified `Bundle`."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    raw_manifest = _read_bytes(manifest_path, "manifest (manifest.json)")

    try:
        manifest = json.loads(raw_manifest.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleIntegrityError(f"manifest.json is not valid UTF-8 JSON ({exc})") from exc

    schema = manifest.get("schema")
    if schema not in SUPPORTED_BUNDLE_SCHEMAS:
        raise BundleIntegrityError(
            f"unsupported bundle schema {schema!r}; this loader accepts "
            f"{list(SUPPORTED_BUNDLE_SCHEMAS)}"
        )

    for required_key in ("bundleId", "clusterEpoch", "cut", "valuationTime", "artifacts"):
        if required_key not in manifest:
            raise BundleIntegrityError(f"manifest.json is missing required key {required_key!r}")

    positions = _load_csv_artifact(root, manifest, "positions", "rows")
    contracts = _load_csv_artifact(root, manifest, "contracts", "contracts")
    terms, terms_sha256 = _load_terms_artifact(root, manifest)

    return Bundle(
        root=root,
        manifest=manifest,
        # Digest of the manifest file's exact bytes. Distinct from `bundleId`, which
        # TraderX computes over the canonical manifest without bundleId.
        manifest_sha256=_sha256(raw_manifest),
        positions=positions,
        contracts=contracts,
        terms=terms,
        terms_sha256=terms_sha256,
    )
