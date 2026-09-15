"""
W0.1 -- bundle ingestion and hash verification.

Reads a `traderx.eod-bundle.v1`/`.v2` bundle, verifies **every artifact
against its own `sha256`**, parses the CSV preambles, and rejects tampering.
Nothing downstream of this module ever sees bytes that failed verification.

**Each artifact is checked against its OWN hash.** `manifest.artifacts.
<name>.sha256` is the SHA-256 of that artifact's exact bytes. `cut.cutSha256`
is a different thing entirely -- it identifies the consensus cut the export
was taken from, and is repeated inside each CSV's preamble. Comparing a CSV
against `cutSha256` would be a category error that happens to typecheck;
this module never does it. See `_verify_artifact`.

**Empty is not missing.** A contracts file with `rows=0` and only a preamble
is a *valid* bundle carrying zero OTC coverage (both the bill and note
fixtures are exactly this). A contracts file that is absent from disk is an
integrity failure. Conflating them would turn "this cut had no swaps" into
"this bundle is broken", or worse, the reverse. `load_bundle` distinguishes
them explicitly.

**v1 has no terms artifact.** That is not a defect -- it is the older
version. It has a consequence the caller must handle rather than paper over:
every instrument needing reference terms is `unsupported` under v1 (see
`engine.integration.terms`, W0.2 step 5).

---

**CRLF -- the hazard this module is built around.**

Hashes are over *committed bytes*. A Windows checkout with
`core.autocrlf=true` (the default from Git for Windows' standard installer)
silently rewrites LF -> CRLF on checkout for anything it considers text,
which includes `.json` and `.csv` unless a `.gitattributes` says otherwise.
Every artifact hash then fails, for a reason nothing in the error message
points at. This is not hypothetical: it is live in this repository's own
`reference/traderX` checkout right now, and it already broke TraderX's own
verifier (plan §W0.1).

Two decisions follow, and they pull in opposite directions on purpose:

  1. **Bytes are read in binary and verified exactly as read.** This module
     never normalizes line endings before hashing. A file whose committed
     bytes were LF and whose on-disk bytes are CRLF is *not* the artifact
     the manifest pins, and saying otherwise would defeat the entire point
     of hashing. There is deliberately no `tolerate_crlf` escape hatch.

  2. **The failure explains itself.** On mismatch, `_verify_artifact`
     additionally tests whether the LF-normalized bytes *would* have
     matched. If they would, the raised error names CRLF translation as the
     cause and gives the `.gitattributes` fix. This is a pure diagnostic:
     the verification still fails, the bytes are still rejected, and the
     normalized digest is never substituted for the real one.

The second decision is what makes the first one survivable. Strictness
without a diagnostic is what produced the original confusion.
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

#: Artifact present only in v2. Its absence in v1 is normal; its absence in
#: v2, or its presence in v1, is a manifest inconsistency.
TERMS_ARTIFACT = "instrumentTerms"

#: Preamble fields each CSV repeats and that must agree with the manifest's
#: own `cut` block. A disagreement means the manifest and the artifact
#: describe different cuts -- an integrity failure, not a warning.
_CUT_PREAMBLE_FIELDS = {
    "consensusSequence": ("cut", "consensusSequence"),
    "sessionDate": ("cut", "sessionDate"),
    "priceSnapshotVersion": ("cut", "priceSnapshotVersion"),
    "cutSha256": ("cut", "cutSha256"),
}

#: Columns every downstream stage indexes by name. Checked against the CSV
#: HEADER at load time so schema drift fails here, naming the artifact and
#: the missing column, rather than surfacing later as a bare `KeyError`
#: from inside the join or the adapter.
#:
#: A file can hash correctly and still be unusable this way: the hash pins
#: the bytes, not their shape. `csv.DictReader` fills a short row's missing
#: column with `None` and buckets a long row's surplus under the `None`
#: key, so neither condition raises on its own -- the first symptom would
#: otherwise be an exception with no artifact, row, or column in it.
_REQUIRED_POSITION_COLUMNS = ("accountId", "security", "quantity")
_REQUIRED_CONTRACT_COLUMNS = ("contractId", "accountId")


class BundleIntegrityError(Exception):
    """Raised for any failure that means the bundle on disk is not the
    bundle the manifest describes: a bad hash, a missing required artifact,
    a row-count disagreement, an unsupported schema, or a preamble that
    contradicts the manifest.

    Deliberately one exception type rather than a hierarchy. Every one of
    these is equally fatal and equally non-recoverable -- there is no
    caller that should catch "bad hash" and proceed while still refusing
    "missing file". The message carries the detail; the type carries the
    verdict.
    """


@dataclass(frozen=True)
class CsvArtifact:
    """One verified CSV artifact: its preamble comments and its data rows.

    `rows` are `dict`s keyed by the CSV header, in file order. Order is
    preserved because the caller may need it for diagnostics -- but nothing
    downstream keys on it (plan working rule 7: identity never rides on
    array position).
    """
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
    """A fully verified EOD bundle. Reaching this object means every
    artifact hash matched, every declared row count matched, and every
    preamble agreed with the manifest."""
    root: Path
    manifest: Dict
    manifest_sha256: str
    positions: CsvArtifact
    contracts: CsvArtifact
    #: Parsed `instrument-terms.json`, or `None` for a v1 bundle. `None`
    #: means "this bundle version carries no terms", never "terms failed to
    #: load" -- a terms artifact that fails to load raises instead.
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
    """Reads a file in **binary**. Never `open(path)` without `'rb'` here:
    text mode applies universal-newline translation, which would silently
    change the bytes being hashed on exactly the platform where the CRLF
    hazard lives."""
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
    """Verifies `raw` against `expected_sha256` **exactly as read**.

    On mismatch, checks whether LF-normalized bytes would have matched and,
    if so, says so -- see this module's docstring. The normalized digest is
    used only to explain the failure; it never satisfies it.
    """
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
    """Splits a TraderX extract CSV into its `# key=value` preamble and its
    header + data rows.

    The preamble carries both machine-checkable fields (`rows=`,
    `cutSha256=`) and long prose legends explaining conventions. Both are
    `# key=value`; only the former are read here, but all are kept so a
    diagnostic can quote them.

    Decoded as UTF-8 **after** hashing, never before -- the hash is over
    bytes, and decoding is only for parsing.
    """
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

    # Drop trailing blank lines so a file ending in a newline doesn't read
    # as carrying an empty final row.
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()

    if not body_lines:
        raise BundleIntegrityError(f"{label}: artifact has a preamble but no CSV header row")

    reader = csv.DictReader(io.StringIO("\n".join(body_lines)))
    header = list(reader.fieldnames or [])
    rows = [dict(row) for row in reader]
    return preamble, header, rows


def _check_preamble_against_manifest(artifact: CsvArtifact, manifest: Dict, label: str) -> None:
    """Every field the CSV preamble repeats from the manifest must agree.

    A bundle whose manifest says one cut and whose artifact preamble says
    another is not a bundle with a cosmetic inconsistency -- it is two
    different exports in one directory, and pricing it would attribute one
    cut's numbers to another's identity.
    """
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
    """The manifest's declared row count, the preamble's own count, and the
    rows actually parsed must all agree.

    Checking the manifest alone would miss a truncated file whose preamble
    was truncated with it; checking the preamble alone would miss a
    manifest that describes a different export.
    """
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
    """Every column downstream indexes by name must exist in the header, and
    no row may carry surplus unnamed fields.

    Both conditions are silent in `csv.DictReader` (see
    `_REQUIRED_POSITION_COLUMNS`), so they are checked explicitly here --
    at the point where the artifact's name and the offending column can
    still be named in the message.
    """
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
    """Loads and hash-verifies `instrument-terms.json` for a v2 bundle.

    Returns `(None, None)` for v1. The manifest's declared `entries` count
    is checked against the parsed entry list for the same reason row counts
    are checked on the CSVs.

    Verifying the hash **before parsing** is W0.2 step 1: parsing
    unverified bytes would mean a tampered terms file gets a chance to
    influence behavior (even just via an exception path) before it is
    rejected.
    """
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
    """Reads and fully verifies the EOD bundle rooted at `root`.

    Raises `BundleIntegrityError` on any discrepancy between what the
    manifest says and what is on disk. Returns only bundles that verified
    completely -- there is no partially-verified `Bundle`.
    """
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
        # The manifest's own digest, over its exact bytes. Distinct from
        # `bundleId`, which TraderX computes over the canonical manifest
        # *without* bundleId -- this is the byte digest of the file as
        # delivered, which is what a workload key should pin.
        manifest_sha256=_sha256(raw_manifest),
        positions=positions,
        contracts=contracts,
        terms=terms,
        terms_sha256=terms_sha256,
    )
