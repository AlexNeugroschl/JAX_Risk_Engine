"""
W0.8 -- crash-safe result publication and the durable result store.

**This is the half of W0.8 that W1.6.4 did not build.** That task landed the
state machine: the workload key, idempotent submission, immutable terminal
attempts, and the four distinguishable lookup states. All of it lived in an
in-process dict, so a restart lost everything and the crash-safety design in
plan §W0.8 had nothing to stand on. This module is the store that design
needs, and the publication protocol that makes writing to it survivable.

---

**The protocol, and what each step buys** (plan §W0.8):

  1. Write artifacts to a **temporary path**
  2. **Verify hashes** of what was actually written
  3. **Atomically publish** the result manifest
  4. **Then** advance the workload pointer

The ordering is the whole design, and it is chosen so that *every* crash
window leaves a coherent store rather than a plausible-looking wrong one:

  - Crash before (3) -> orphaned bytes under a temp path no lookup reaches.
    A partial result is never discoverable, so it can never be served as a
    complete one.
  - Crash between (3) and (4) -> a complete, discoverable result whose
    pointer is stale. **This is the window TraderX found in v3** and it is
    why step 4 is not the commit point: see "the pointer is a cache" below.
  - Crash after (4) -> the ordinary complete case.

**Step 2 is not ceremony.** Hashing what was written, rather than what was
meant to be written, is what makes a truncated or partially-flushed file a
publication *failure* instead of a durable artifact that verifies against
nothing. A short write that nobody checks is exactly the "plausible wrong
number" this boundary exists to refuse -- it just arrives as bytes rather
than as a price.

---

**The manifest is the commit point; the pointer is a cache.**

A result is published the instant its manifest lands atomically at step 3.
Discoverability deliberately does **not** depend on the pointer file: if the
pointer is missing or behind, `lookup` falls back to a **scan** over
published manifests and advances the pointer as a side effect
(`_scan_for_workload`). So the crash window between (3) and (4) costs a
slower lookup, never a lost result.

That is the difference between a cache and a record, and it is worth being
precise about because the failure mode is asymmetric: trusting a stale
pointer returns `UNKNOWN_WORKLOAD` for work that **is** complete, which
invites a coordinator to resubmit an overnight batch it already has the
answer to. The scan is slower and always correct; the pointer is fast and
sometimes behind. Reading the cache first and the record second gives both.

---

**What is stored, and what is addressable.**

Attempts are keyed by `attemptId` and are **immutable once terminal** -- the
same guarantee `engine.integration.workload` makes in memory, now durable.
A second attempt never overwrites a first: each gets its own manifest path,
and the pointer names the most recent *successful* one.

**Publication order is recorded, not inferred.** Each manifest carries a
`publicationSequence`, because "the most recent successful attempt" is a
statement about commit order and both ways of recovering it after a restart
are wrong: directory order is arbitrary, and file mtime is the filesystem's
opinion at a resolution that varies by platform and that a copy or a restore
rewrites. The counter is recovered from disk rather than held in memory --
a process-local one restarts at zero, so after a bounce a newly published
attempt would claim to predate everything already stored, and lookup would
then serve a stale result while reporting it as the most recent.

Layout under the store root::

    attempts/<attemptId>.json        one attempt's manifest (the commit point)
    pointers/<workloadKeyDigest>     the most recent successful attemptId
    sequence                         high-water mark for publicationSequence
    tmp/<attemptId>.<uuid>.json      step-1 scratch; never read by lookup

`sequence` and the pointers are both caches: losing either costs a scan,
never a result.

`tmp/` is a sibling of `attempts/` **on purpose**: `os.replace` is only
atomic within a filesystem, and a temp directory elsewhere on the machine
can be on a different one. A cross-device rename raises rather than
silently copying, but only at publication time, on a machine that may
differ from the developer's -- so the layout removes the possibility rather
than relying on a test to notice it.

**The workload key is digested before use as a filename.** A key is
`sha256:<hex>`, and the colon is not a legal filename character on Windows
(it opens an alternate data stream). Digesting rather than escaping keeps
one rule on every platform.

---

**This is still not a distributed store, and the limit is specific.**
Within one process it is thread-safe: `_sequence_lock` makes the
read-then-increment of `publicationSequence` atomic, which matters because
FastAPI serves from a thread pool and an unguarded version was measured
issuing **2 distinct sequences across 30 concurrent publications**.

Across *processes* it does not coordinate. Two engines publishing at the
same instant can issue the same sequence, and concurrent publication of the
same attempt id would be last-writer-wins at step 3 (attempt ids are uuid4,
so that does not arise in practice). A sequence tie resolves deterministically
by attempt id, which is safe precisely because two successful attempts under
one workload key are the same computation by construction -- so either is a
correct answer.

What it does provide, and what I-08 asks for, is that a crash at any point
never yields a discoverable partial result, that a lost pointer never yields
a lost result, and that a restart never loses a completed one.
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

#: Bumped if the on-disk layout changes in a way an older reader would
#: misread. Recorded in every manifest so a future reader can refuse an
#: unfamiliar one rather than guess at it -- the same reason the bundle
#: loader pins `traderx.eod-bundle.v1`/`v2` instead of sniffing.
PUBLICATION_SCHEMA = "jax.eod-publication.v1"

_ATTEMPTS_DIR = "attempts"
_POINTERS_DIR = "pointers"
_TMP_DIR = "tmp"

#: Store-wide high-water mark for `publicationSequence`. Lets `lookup` check
#: a cached pointer's freshness with one read instead of a directory scan.
#: Advisory: losing it costs a scan, never a result.
_SEQUENCE_FILE = "sequence"


class PublicationError(RuntimeError):
    """A result could not be published, or what was published did not
    verify.

    Raised rather than logged: a caller that believes it published a result
    it did not is the one state this module must never produce, because the
    coordinator's next move is to stop retrying.
    """


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(payload: Dict) -> bytes:
    """Serializes a manifest canonically, for a stable hash.

    Sorted keys and no incidental whitespace, matching
    `engine.integration.workload.workload_key`. Two manifests differing only
    in key order must hash identically, or re-verifying a published artifact
    would depend on how the writer happened to order a dict.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _pointer_name(workload_key: str) -> str:
    """A filesystem-safe name for a workload key.

    `sha256:<hex>` contains a colon, which Windows reads as an alternate
    data stream separator -- so the key is digested rather than escaped.
    One rule on every platform beats a per-platform escape.
    """
    return _sha256(workload_key.encode("utf-8"))


class ResultStore:
    """A filesystem-backed, crash-safe store for published EOD results.

    **Thread-safe for its own bookkeeping**, in the same way and for the
    same reason as `engine.integration.workload.AttemptStore`: FastAPI
    serves from a thread pool, and the pointer read-then-advance in
    `lookup` is not atomic on its own.

    The durability guarantees are the filesystem's: `os.replace` is atomic
    within a filesystem on both POSIX and Windows, and every publication
    goes through it.
    """

    def __init__(self, root) -> None:
        self.root = Path(root)
        self._lock = threading.Lock()
        #: Whether this object has reconciled the sequence high-water mark
        #: against what is on disk. Done lazily, once, on first publish --
        #: see `_next_sequence`.
        self._sequence_recovered = False
        #: Guards read-then-increment of the sequence. Separate from
        #: `_lock` (which guards `lookup`'s pointer reconciliation) so a
        #: publication and a lookup never wait on each other.
        self._sequence_lock = threading.Lock()
        #: Highest sequence this object has issued, including ones whose
        #: manifest has not landed yet.
        self._issued_sequence = -1
        for sub in (_ATTEMPTS_DIR, _POINTERS_DIR, _TMP_DIR):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # -- paths ---------------------------------------------------------

    def _attempt_path(self, attempt_id: str) -> Path:
        return self.root / _ATTEMPTS_DIR / f"{attempt_id}.json"

    def _pointer_path(self, workload_key: str) -> Path:
        return self.root / _POINTERS_DIR / _pointer_name(workload_key)

    # -- sequencing ----------------------------------------------------

    def _highest_sequence(self) -> int:
        """The highest publication sequence this store has ever issued.

        Kept in its own tiny file so that `lookup` can ask "is my cached
        pointer current?" in **one read** rather than a directory scan. The
        file is a high-water mark, not a lock: it only ever moves forward,
        and losing it costs a scan (via the `-1` fallback), never a result.
        """
        try:
            return int(
                (self.root / _SEQUENCE_FILE).read_bytes().decode("utf-8").strip()
            )
        except (OSError, ValueError):
            return -1

    def _next_sequence(self) -> int:
        """The next publication sequence number, and record it.

        **Recovered from what is already published whenever the high-water
        file is missing or behind**, never from a counter in memory. A
        process-local counter restarts at zero, so after a bounce a newly
        published attempt would claim to predate everything already on
        disk -- and `lookup` would then serve a stale result as the most
        recent one. The scan runs at most once per process here, and never
        on the read path.

        Not a strong distributed guarantee: two processes publishing
        simultaneously can pick the same number. That is a tie, and ties
        resolve deterministically by attempt id in `_scan_for_workload` --
        which is safe here because two successful attempts under one
        workload key are the same computation by construction, so either is
        a correct answer.
        """
        with self._sequence_lock:
            # **Read-then-increment is only atomic under a lock.** The store
            # is shared across FastAPI's thread pool, so without this two
            # concurrent publications read the same high-water mark and
            # issue the same number. `_issued_sequence` covers the window
            # between issuing a number and its manifest landing on disk,
            # which the high-water file alone does not.
            highest = max(self._highest_sequence(), self._issued_sequence)
            if not self._sequence_recovered:
                # Runs at most once per store object, and only matters when
                # the high-water file is missing or behind what is on disk
                # -- a restored backup, or a crash between the manifest
                # rename and the high-water write. Every later publication
                # reads the file.
                for manifest in self.iter_manifests():
                    sequence = manifest.get("publicationSequence")
                    if isinstance(sequence, int) and sequence > highest:
                        highest = sequence
                self._sequence_recovered = True

            nxt = highest + 1
            self._issued_sequence = nxt
            tmp_path = self.root / _TMP_DIR / f"seq.{uuid.uuid4().hex}"
            try:
                with open(tmp_path, "wb") as handle:
                    handle.write(str(nxt).encode("utf-8"))
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_path, self.root / _SEQUENCE_FILE)
            except OSError:  # pragma: no cover - filesystem failure
                tmp_path.unlink(missing_ok=True)
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
        """Publishes one terminal attempt, following the four-step protocol.

        Returns the published manifest.

        **Steps 1 and 2 happen before anything is discoverable.** The
        manifest is written to `tmp/`, read back, and hashed. Only bytes
        that verify against what was intended get promoted at step 3.
        Reading back rather than trusting the write is the point: a
        truncated file that was never re-read is indistinguishable from a
        good one until someone tries to price against it.

        **Step 3 is the commit point.** `os.replace` either moves the whole
        file or does nothing; there is no state in which half a manifest is
        visible under `attempts/`.

        **Step 4 is best-effort by design.** A failure to advance the
        pointer is *not* a publication failure, because the manifest is
        already the durable record and `lookup`'s scan will find it. Raising
        here would tell the caller its published result was lost, which is
        false, and would invite exactly the duplicate recomputation the
        scan exists to prevent.
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
            # **Publication order, carried in the record itself.** "Most
            # recent successful attempt" is a statement about commit order,
            # and the alternatives for recovering it after a restart are
            # both wrong: directory order is arbitrary, and file mtime is
            # the *filesystem's* opinion, at a resolution that varies by
            # platform and that a backup or a copy rewrites. A monotonic
            # counter written into the bytes that are hashed cannot drift
            # from the thing it orders.
            "publicationSequence": self._next_sequence(),
        }
        payload = _canonical_bytes(manifest)
        expected = _sha256(payload)

        # Step 1 -- write to a temporary path. Same filesystem as the
        # destination (see the module docstring), so step 3's rename is a
        # rename and not a cross-device copy.
        tmp_path = self.root / _TMP_DIR / f"{attempt_id}.{uuid.uuid4().hex}.json"
        with open(tmp_path, "wb") as handle:
            handle.write(payload)
            handle.flush()
            # Force the bytes out of the OS cache before the rename. Without
            # this, a power loss can reorder the rename ahead of the data and
            # leave a published manifest pointing at an empty file -- the
            # "durable pointer to nothing" case, which is worse than no
            # publication at all because it is discoverable.
            os.fsync(handle.fileno())

        # Step 2 -- verify what was actually written, not what was meant to
        # be. A short write that nobody reads back is a durable artifact
        # that verifies against nothing.
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

        # Step 3 -- atomic publish. This is the commit point: after it, the
        # result is discoverable by scan whether or not step 4 runs.
        destination = self._attempt_path(attempt_id)
        try:
            os.replace(tmp_path, destination)
        except OSError as exc:
            # **Translated, not left raw.** A cross-device rename, a
            # permissions failure or a full disk here means the result was
            # not published, which is exactly what `PublicationError`
            # signals -- and the route turns that into a truthful
            # `RESULT_NOT_PUBLISHED` telling the coordinator to retry. A
            # bare `OSError` escapes that branch and surfaces as an opaque
            # 500, which says "something broke" rather than "your result
            # is not discoverable, submit again".
            tmp_path.unlink(missing_ok=True)
            raise PublicationError(
                f"attempt {attempt_id}: the manifest verified but could not be "
                f"published to {destination}: {exc}. Nothing was committed, so the "
                f"result is not discoverable and the submission should be retried."
            ) from exc

        # Step 4 -- advance the pointer. Deliberately after, deliberately
        # non-fatal, and only for a *successful* attempt: the pointer names
        # the result a lookup should serve, and a failed attempt is not one.
        if state == "completed":
            self._advance_pointer(
                workload_key, attempt_id, sequence=manifest["publicationSequence"]
            )

        return manifest

    def _read_pointer(self, workload_key: str):
        """The attempt id and sequence a pointer names, or `(None, -1)`.

        A pointer that does not parse reads as absent, for the same reason
        an unreadable manifest does: the scan is always available and
        always correct, so there is never a need to guess at bytes.
        """
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
        """Points this workload key at `attempt_id`, **forward only**.

        The pointer carries the sequence it names, so advancing can refuse
        to move backwards. That is what keeps it a *valid* cache rather
        than merely a fast one: without it, a late-arriving publication or
        a reconciling scan could point the cache at an attempt an earlier
        write had already superseded, and every subsequent lookup would
        serve the older result without ever scanning to notice.

        Written atomically -- a half-written pointer would name a
        nonexistent attempt, and `lookup` would then have to distinguish a
        torn pointer from a legitimately missing one. Cheaper to make it
        impossible.

        **Failures are swallowed on purpose.** See `publish`: the manifest
        is already the record, so a pointer that will not write costs a
        scan, not a result.
        """
        if sequence is None:
            sequence = -1
        _, current = self._read_pointer(workload_key)
        if current > sequence:
            return

        pointer = self._pointer_path(workload_key)
        tmp_path = self.root / _TMP_DIR / f"ptr.{uuid.uuid4().hex}"
        try:
            with open(tmp_path, "wb") as handle:
                handle.write(f"{attempt_id}\n{sequence}".encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, pointer)
        except OSError:  # pragma: no cover - filesystem failure
            tmp_path.unlink(missing_ok=True)

    # -- reading -------------------------------------------------------

    def read_attempt(self, attempt_id: str) -> Optional[Dict]:
        """One published attempt's manifest, or `None`.

        A manifest that does not parse, or that carries an unrecognized
        `publicationSchema`, is treated as **absent** rather than raising.
        The reasoning is the same as the bundle loader's refusal to
        normalize: this reader does not guess at bytes it does not
        understand. Returning `None` lets lookup fall through to the scan
        and, at worst, report the workload as unknown -- which is honest.
        Raising would let one unreadable file from a future writer take out
        every lookup in the store.
        """
        path = self._attempt_path(attempt_id)
        try:
            raw = path.read_bytes()
        except OSError:
            return None
        try:
            manifest = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(manifest, dict):
            return None
        if manifest.get("publicationSchema") != PUBLICATION_SCHEMA:
            return None
        return manifest

    def iter_manifests(self) -> Iterator[Dict]:
        """Every published manifest, in no particular order.

        Unreadable entries are skipped, per `read_attempt`.
        """
        attempts_dir = self.root / _ATTEMPTS_DIR
        if not attempts_dir.is_dir():
            return
        for path in sorted(attempts_dir.glob("*.json")):
            manifest = self.read_attempt(path.stem)
            if manifest is not None:
                yield manifest

    def lookup(self, workload_key: str) -> Optional[Dict]:
        """The published result for this workload key, or `None`.

        **Pointer first, scan second** -- and the scan is not a fallback for
        errors, it is the fallback for a pointer that is merely *behind*.
        That is the v3 gap: a crash between step 3 and step 4 leaves a
        complete result whose pointer never advanced, and a lookup that
        trusted the pointer alone would answer `UNKNOWN_WORKLOAD` for work
        that is finished and correct.

        **The scan advances the pointer as a side effect**, so the cost is
        paid once per crash rather than on every subsequent lookup.

        Returns the most recent *successful* attempt only. A failed attempt
        is published and addressable by id, but it is never what a workload
        lookup serves -- plan §W0.8.
        """
        with self._lock:
            attempt_id, pointed_sequence = self._read_pointer(workload_key)

            if attempt_id:
                manifest = self.read_attempt(attempt_id)
                # The pointer is only trusted when it names an attempt that
                # actually exists, belongs to this workload, and succeeded.
                # Anything else means the pointer is stale or wrong, and the
                # scan is authoritative.
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
                # Reconcile: the pointer was missing, torn, or behind.
                self._advance_pointer(
                    workload_key,
                    found["attemptId"],
                    sequence=found.get("publicationSequence"),
                )
            return found

    def _scan_for_workload(self, workload_key: str) -> Optional[Dict]:
        """Scans published manifests for this workload's newest success.

        **Ordered by `publicationSequence`**, the manifest's own record of
        commit order -- not by directory order, which is arbitrary, and not
        by file mtime, which is the filesystem's opinion at a resolution
        that varies by platform and that a copy or a restore rewrites.

        Ties keep the lexicographically larger attempt id purely for
        determinism: two successful attempts for one workload key are the
        same computation by construction, so either is a correct answer,
        and an *arbitrary* choice would make a test flaky without making a
        caller wrong.
        """
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
                # A manifest with no usable sequence still counts as
                # published -- it just sorts oldest, rather than being
                # dropped. Losing a real result over a missing ordering
                # field would be a worse failure than ordering it
                # conservatively.
                sequence = -1
            candidates.append((sequence, manifest["attemptId"], manifest))

        if not candidates:
            return None
        candidates.sort(key=lambda entry: (entry[0], entry[1]))
        return candidates[-1][2]

    def find_by_submission(self, submission_id: str) -> Optional[Dict]:
        """The published attempt carrying this `submissionId`, or `None`.

        Scans rather than indexing. This exists so idempotent retry survives
        a **restart**: without it, a coordinator retrying a lost response
        after the engine bounced would get a second attempt for work already
        completed, which is precisely what `submissionId` is supposed to
        prevent. In-memory, `AttemptStore` answers this from a dict; the
        scan is what makes the guarantee durable rather than
        process-lifetime.
        """
        newest = None
        newest_sequence = -2
        attempts_dir = self.root / _ATTEMPTS_DIR
        if not attempts_dir.is_dir():
            return None
        for path in attempts_dir.glob("*.json"):
            manifest = self.read_attempt(path.stem)
            if manifest is None or manifest.get("submissionId") != submission_id:
                continue
            sequence = manifest.get("publicationSequence")
            if not isinstance(sequence, int):
                sequence = -1
            if sequence > newest_sequence:
                newest, newest_sequence = manifest, sequence
        return newest

    # -- test support --------------------------------------------------

    def clear(self) -> None:
        """Drops everything. Test support only -- deliberately no route."""
        with self._lock:
            for sub in (_ATTEMPTS_DIR, _POINTERS_DIR, _TMP_DIR):
                shutil.rmtree(self.root / sub, ignore_errors=True)
                (self.root / sub).mkdir(parents=True, exist_ok=True)
            (self.root / _SEQUENCE_FILE).unlink(missing_ok=True)
            self._sequence_recovered = False
            self._issued_sequence = -1


def default_store_root() -> Path:
    """Where results are published when nothing configures it.

    `JAX_EOD_STORE_ROOT` if set, else a per-user directory under the system
    temp root. **Not the current working directory**, which would scatter
    stores wherever the service happened to be started from and make "did
    this restart see the same store?" depend on how it was launched.
    """
    configured = os.environ.get("JAX_EOD_STORE_ROOT")
    if configured:
        return Path(configured)
    return Path(tempfile.gettempdir()) / "jax-eod-store"
