"""
Durable, crash-safe store for the TraderX path's published results (I-08).

Publication protocol:

  1. write the manifest to a temporary path (fsync);
  2. read it back and verify its hash;
  3. atomically rename it into `attempts/` (the commit point);
  4. then advance the workload's pointer (best effort).

A crash before 3 leaves only unreachable temp bytes, so a partial result is never
discoverable. A crash between 3 and 4 leaves a complete result with a stale pointer:
lookup does not depend on the pointer, and falls back to a scan of manifests (advancing the
pointer as it goes). The manifest is the record; the pointer and `sequence` file are caches
whose loss costs a scan, never a result.

Each manifest records a `publicationSequence`, recovered from disk rather than from a
process-local counter, so "most recent" survives restarts (directory order and mtimes are
not reliable). Attempts are immutable once terminal; each has its own manifest.

A submission id's binding (the attempt, and so the workload, it names) is recorded when it is
made, before the caller is answered (I-57): a binding record is the only record of an id that
was served a cached result, or whose attempt is still running, since neither has a manifest
of its own. A manifest also carries its attempt's `submissionId`, so an id whose record is
missing (a store written before 2026-10-10) is still found, by a scan.

Layout under the store root::

    attempts/<attemptId>.json             one attempt's manifest (the commit point)
    pointers/<workloadKeyDigest>          the most recent successful attemptId and its sequence
    submissions/<submissionIdDigest>.json the attempt a submission id is bound to
    sequence                              high-water mark for publicationSequence
    tmp/                                  scratch for every write; never read by lookup

`tmp/` sits beside the others because `os.replace` is atomic only within one filesystem.
Workload keys (`sha256:<hex>`) and submission ids (caller text) are digested for filenames,
since `:` is not legal on Windows and an id may hold any character.

Thread-safe within a process. Across processes nothing is coordinated: two processes can
issue the same sequence, and ties resolve by attempt id, which is safe because two
successful attempts under one workload key are the same computation; and two processes that
bind one new submission id at the same moment both succeed, the later record replacing the
earlier. Within a process `engine.traderx.workload.AttemptStore` serializes bindings.
"""
import hashlib
import json
import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Dict, Iterator, List, Optional

#: On-disk layout version, recorded in every manifest; a reader treats an unfamiliar one
#: as absent rather than guessing.
PUBLICATION_SCHEMA = "jax.eod-publication.v1"

_ATTEMPTS_DIR = "attempts"
_POINTERS_DIR = "pointers"
_SUBMISSIONS_DIR = "submissions"
_TMP_DIR = "tmp"
_STORE_DIRS = (_ATTEMPTS_DIR, _POINTERS_DIR, _SUBMISSIONS_DIR, _TMP_DIR)

#: The fields of a submission binding, as `ResultStore.find_submission` returns it.
_BINDING_FIELDS = ("submissionId", "workloadKey", "attemptId")

#: Store-wide high-water mark for `publicationSequence` (a cache; see the module docstring).
_SEQUENCE_FILE = "sequence"


class PublicationError(RuntimeError):
    """A result could not be published, or what was written did not verify. Raised, never
    logged, because a caller that wrongly believes it published tells the coordinator to
    stop retrying."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(payload: Dict) -> bytes:
    """Canonical manifest bytes (sorted keys, no whitespace), so the hash does not depend
    on dict order."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _pointer_name(workload_key: str) -> str:
    """Filesystem-safe name for a workload key: its SHA-256 (the key's `:` is not legal on
    Windows)."""
    return _sha256(workload_key.encode("utf-8"))


def _binding_of(record: Dict) -> Optional[Dict]:
    """The binding fields of a binding record or manifest, or `None` if one is not a
    string (an unreadable record is absent, as an unreadable manifest is)."""
    binding = {name: record.get(name) for name in _BINDING_FIELDS}
    if not all(isinstance(value, str) for value in binding.values()):
        return None
    return binding


class ResultStore:
    """Filesystem-backed, crash-safe store for published EOD results. Thread-safe; the
    atomicity is that of `os.replace` within one filesystem (POSIX and Windows)."""

    def __init__(self, root) -> None:
        self.root = Path(root)
        self._lock = threading.Lock()
        #: Whether the sequence high-water mark has been reconciled with disk (done once,
        #: on first publish).
        self._sequence_recovered = False
        #: Guards read-then-increment of the sequence; separate from `_lock` (lookup) so
        #: publication and lookup do not wait on each other.
        self._sequence_lock = threading.Lock()
        #: Highest sequence issued by this object, including ones not yet on disk.
        self._issued_sequence = -1
        for sub in _STORE_DIRS:
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # -- paths ---------------------------------------------------------

    def _attempt_path(self, attempt_id: str) -> Path:
        return self.root / _ATTEMPTS_DIR / f"{attempt_id}.json"

    def _pointer_path(self, workload_key: str) -> Path:
        return self.root / _POINTERS_DIR / _pointer_name(workload_key)

    def _submission_path(self, submission_id: str) -> Path:
        return self.root / _SUBMISSIONS_DIR / f"{_sha256(submission_id.encode('utf-8'))}.json"

    def _write_atomically(self, destination: Path, payload: bytes) -> None:
        """Replace `destination` with `payload` through a fsynced file in `tmp/`, so a
        reader sees the old bytes or the new, never a torn write. Raises `OSError`, leaving
        no temporary file behind."""
        tmp_path = self.root / _TMP_DIR / f"{destination.name}.{uuid.uuid4().hex}"
        try:
            with open(tmp_path, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, destination)
        except OSError:
            tmp_path.unlink(missing_ok=True)
            raise

    # -- sequencing ----------------------------------------------------

    def _highest_sequence(self) -> int:
        """Highest publication sequence recorded in the `sequence` file, or -1."""
        try:
            return int(
                (self.root / _SEQUENCE_FILE).read_bytes().decode("utf-8").strip()
            )
        except (OSError, ValueError):
            return -1

    def _next_sequence(self) -> int:
        """Issue and record the next publication sequence. On first use the high-water mark
        is reconciled with every published manifest, so it survives a lost or stale
        `sequence` file. Not coordinated across processes (see the module docstring)."""
        with self._sequence_lock:
            # Under the lock: concurrent publications would otherwise read the same mark.
            # `_issued_sequence` covers numbers issued but not yet written.
            highest = max(self._highest_sequence(), self._issued_sequence)
            if not self._sequence_recovered:
                # Once per store object: catches a missing or stale high-water file
                # (restored backup, or a crash between manifest rename and this write).
                for manifest in self.iter_manifests():
                    sequence = manifest.get("publicationSequence")
                    if isinstance(sequence, int) and sequence > highest:
                        highest = sequence
                self._sequence_recovered = True

            nxt = highest + 1
            self._issued_sequence = nxt
            try:
                self._write_atomically(
                    self.root / _SEQUENCE_FILE, str(nxt).encode("utf-8")
                )
            except OSError:  # pragma: no cover - filesystem failure
                pass  # a cache: a lost write costs the scan above, never an order
            return nxt

    # -- publication ---------------------------------------------------

    def publish(
        self,
        *,
        attempt_id: str,
        workload_key: str,
        state: str,
        result: Optional[Dict] = None,
        reason: Optional[str] = None,
        submission_id: Optional[str] = None,
    ) -> Dict:
        """Publish one terminal attempt by the four-step protocol and return its manifest.

        The manifest is staged, read back and verified before the atomic rename (step 3,
        the commit point). Advancing the pointer (step 4) is best effort: the manifest is
        already durable and a scan will find it.
        """
        if state not in ("completed", "failed"):
            raise PublicationError(
                f"refusing to publish attempt {attempt_id} in state {state!r}: only "
                f"terminal attempts are published. A running attempt has no result "
                f"to commit, and publishing one would make an in-flight computation "
                f"discoverable as a finished answer."
            )

        manifest = {
            "publicationSchema": PUBLICATION_SCHEMA,
            "attemptId": attempt_id,
            "workloadKey": workload_key,
            "state": state,
            "submissionId": submission_id,
            "result": result,
            "reason": reason,
            # Commit order, inside the hashed bytes (see the module docstring).
            "publicationSequence": self._next_sequence(),
        }
        payload = _canonical_bytes(manifest)
        expected = _sha256(payload)

        # Step 1: stage in tmp/ (same filesystem as the destination).
        tmp_path = self.root / _TMP_DIR / f"{attempt_id}.{uuid.uuid4().hex}.json"
        with open(tmp_path, "wb") as handle:
            handle.write(payload)
            handle.flush()
            # fsync before the rename, so a power loss cannot publish a manifest whose data
            # never reached disk.
            os.fsync(handle.fileno())

        # Step 2: verify the bytes actually written.
        try:
            written = tmp_path.read_bytes()
        except OSError as exc:  # pragma: no cover - filesystem failure
            raise PublicationError(
                f"attempt {attempt_id}: could not read back the staged manifest at "
                f"{tmp_path}: {exc}"
            ) from exc

        actual = _sha256(written)
        if actual != expected:
            tmp_path.unlink(missing_ok=True)
            raise PublicationError(
                f"attempt {attempt_id}: staged manifest failed verification before "
                f"publication.\n"
                f"  expected {expected}\n"
                f"  actual   {actual}\n"
                f"Nothing was published. The bytes on disk are not the bytes that "
                f"were meant to be written, so promoting them would publish a "
                f"result no consumer can reproduce."
            )

        # Step 3: atomic publish (the commit point).
        destination = self._attempt_path(attempt_id)
        try:
            os.replace(tmp_path, destination)
        except OSError as exc:
            # Reported as PublicationError, so the route answers RESULT_NOT_PUBLISHED and
            # the coordinator retries.
            tmp_path.unlink(missing_ok=True)
            raise PublicationError(
                f"attempt {attempt_id}: the manifest verified but could not be "
                f"published to {destination}: {exc}. Nothing was committed, so the "
                f"result is not discoverable and the submission should be retried."
            ) from exc

        # Step 4: advance the pointer, for a successful attempt only; non-fatal.
        if state == "completed":
            self._advance_pointer(
                workload_key, attempt_id, sequence=manifest["publicationSequence"]
            )

        return manifest

    def _read_pointer(self, workload_key: str):
        """The attempt id and sequence a pointer names, or `(None, -1)` if absent or
        unparseable (the scan is always available)."""
        try:
            raw = self._pointer_path(workload_key).read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            return None, -1
        attempt_id, _, sequence_text = raw.strip().partition("\n")
        if not attempt_id:
            return None, -1
        try:
            sequence = int(sequence_text)
        except ValueError:
            sequence = -1
        return attempt_id, sequence

    def _advance_pointer(
        self, workload_key: str, attempt_id: str, sequence: Optional[int] = None
    ) -> None:
        """Point this workload key at `attempt_id`, forward only (never to a lower
        sequence), written atomically. Failures are swallowed: the manifest is the record."""
        if sequence is None:
            sequence = -1
        _, current = self._read_pointer(workload_key)
        if current > sequence:
            return
        try:
            self._write_atomically(
                self._pointer_path(workload_key),
                f"{attempt_id}\n{sequence}".encode("utf-8"),
            )
        except OSError:  # pragma: no cover - filesystem failure
            pass

    # -- submission bindings -------------------------------------------

    def bind_submission(
        self, *, submission_id: str, workload_key: str, attempt_id: str
    ) -> None:
        """Record that `submission_id` is bound to `attempt_id` of `workload_key`, replacing
        any earlier record of the id (the caller rebinds an id only to a new attempt of the
        same workload, when its attempt was lost running). Raises `PublicationError`: a
        binding that did not land cannot be honoured after a restart, so the caller must
        not answer as if it had."""
        record = {
            "publicationSchema": PUBLICATION_SCHEMA,
            "submissionId": submission_id,
            "workloadKey": workload_key,
            "attemptId": attempt_id,
        }
        destination = self._submission_path(submission_id)
        try:
            self._write_atomically(destination, _canonical_bytes(record))
        except OSError as exc:
            raise PublicationError(
                f"submission {submission_id!r}: its binding to attempt {attempt_id} could "
                f"not be recorded at {destination}: {exc}. Nothing was bound, so the "
                f"submission should be retried."
            ) from exc

    def find_submission(self, submission_id: str) -> Optional[Dict]:
        """The binding of `submission_id` (`submissionId`, `workloadKey`, `attemptId`), or
        `None` if the id was never bound: its record, else the newest published manifest
        carrying the id (see the module docstring)."""
        record = self._read_record(self._submission_path(submission_id))
        if record is not None and record.get("submissionId") == submission_id:
            binding = _binding_of(record)
            if binding is not None:
                return binding

        newest = None
        newest_sequence = -2
        for manifest in self.iter_manifests():
            if manifest.get("submissionId") != submission_id:
                continue
            sequence = manifest.get("publicationSequence")
            if not isinstance(sequence, int):
                sequence = -1
            if sequence > newest_sequence:
                newest, newest_sequence = manifest, sequence
        return None if newest is None else _binding_of(newest)

    # -- reading -------------------------------------------------------

    @staticmethod
    def _read_record(path: Path) -> Optional[Dict]:
        """A JSON record of this layout, or `None` if missing, unparseable or of an
        unfamiliar `publicationSchema` (so one unreadable file cannot break every
        lookup)."""
        try:
            record = json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(record, dict):
            return None
        if record.get("publicationSchema") != PUBLICATION_SCHEMA:
            return None
        return record

    def read_attempt(self, attempt_id: str) -> Optional[Dict]:
        """One published manifest, or `None` if missing or unreadable."""
        return self._read_record(self._attempt_path(attempt_id))

    def iter_manifests(self) -> Iterator[Dict]:
        """Every readable published manifest, in no particular order."""
        attempts_dir = self.root / _ATTEMPTS_DIR
        if not attempts_dir.is_dir():
            return
        for path in sorted(attempts_dir.glob("*.json")):
            manifest = self.read_attempt(path.stem)
            if manifest is not None:
                yield manifest

    def lookup(self, workload_key: str) -> Optional[Dict]:
        """The newest successful published attempt for `workload_key`, or `None`.

        The pointer is trusted only if it names a completed attempt of this workload whose
        sequence is the store-wide high-water mark (nothing published since); otherwise the
        manifests are scanned and the pointer advanced. Failed attempts are addressable by
        id but never served here.
        """
        with self._lock:
            attempt_id, pointed_sequence = self._read_pointer(workload_key)

            if attempt_id:
                manifest = self.read_attempt(attempt_id)
                # Trust the pointer only under the conditions in the docstring.
                if (
                    manifest is not None
                    and manifest.get("workloadKey") == workload_key
                    and manifest.get("state") == "completed"
                    and manifest.get("publicationSequence") == pointed_sequence
                    and pointed_sequence == self._highest_sequence()
                ):
                    return manifest

            found = self._scan_for_workload(workload_key)
            if found is not None:
                # The pointer was missing, torn or behind.
                self._advance_pointer(
                    workload_key,
                    found["attemptId"],
                    sequence=found.get("publicationSequence"),
                )
            return found

    def _scan_for_workload(self, workload_key: str) -> Optional[Dict]:
        """Newest successful manifest for this workload by `publicationSequence`; ties go
        to the larger attempt id (deterministic, and either is correct)."""
        candidates: List = []
        attempts_dir = self.root / _ATTEMPTS_DIR
        if not attempts_dir.is_dir():
            return None
        for path in attempts_dir.glob("*.json"):
            manifest = self.read_attempt(path.stem)
            if manifest is None:
                continue
            if manifest.get("workloadKey") != workload_key:
                continue
            if manifest.get("state") != "completed":
                continue
            sequence = manifest.get("publicationSequence")
            if not isinstance(sequence, int):
                # No usable sequence: still published; sorts oldest.
                sequence = -1
            candidates.append((sequence, manifest["attemptId"], manifest))

        if not candidates:
            return None
        candidates.sort(key=lambda entry: (entry[0], entry[1]))
        return candidates[-1][2]

    # -- test support --------------------------------------------------

    def clear(self) -> None:
        """Delete everything. Test support only; no route reaches it."""
        with self._lock:
            for sub in _STORE_DIRS:
                shutil.rmtree(self.root / sub, ignore_errors=True)
                (self.root / sub).mkdir(parents=True, exist_ok=True)
            (self.root / _SEQUENCE_FILE).unlink(missing_ok=True)
            self._sequence_recovered = False
            self._issued_sequence = -1


def default_store_root() -> Path:
    """`JAX_EOD_STORE_ROOT` if set, else `jax-eod-store` under the system temp directory
    (not the working directory, so the store does not depend on where the service
    started)."""
    configured = os.environ.get("JAX_EOD_STORE_ROOT")
    if configured:
        return Path(configured)
    return Path(tempfile.gettempdir()) / "jax-eod-store"
