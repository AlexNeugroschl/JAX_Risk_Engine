"""
Workload key and attempt store for EOD submissions.

The workload key is a canonical hash of what can change a result: bundle identity, market
inputs, calculation set, reporting currency, adapter/engine/schema versions (and, when given,
precision and a terms hash). Same key, same computation; results may be shared. A
`submissionId` identifies a request, not a computation, so it is not in the key: a retry
with the same id recovers the same attempt.

Lookup has four states and only "never submitted" is `UNKNOWN_WORKLOAD` (a 404); running,
failed and completed are reported as such, so accepted work is never mistaken for unknown.
Lookup prefers the most recent successful attempt. Attempts are immutable once terminal and
addressable by `attemptId`.

Storage is in memory, optionally backed by an `engine.traderx.publication.ResultStore`
that publishes terminal attempts and is consulted when memory has no answer (I-08). Running
attempts are never published, so after a restart a job that was running reads as unknown
and is resubmitted; the workload key makes the recomputation identical.
"""
import hashlib
import json
import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

#: Lookup states. Only `UNKNOWN_WORKLOAD` is a 404; the rest are 200s carrying the state.
STATE_RUNNING = "running"
STATE_FAILED = "failed"
STATE_COMPLETED = "completed"
UNKNOWN_WORKLOAD = "UNKNOWN_WORKLOAD"

#: Version of the key derivation, part of the key so old and new derivations never collide.
WORKLOAD_KEY_VERSION = "workload-key-v1"


class SubmissionIdConflict(ValueError):
    """A `submissionId` reused for a different workload key. Refused: returning the
    existing attempt would give a result computed from other inputs."""


def workload_key(
    *,
    bundle_id: str,
    cluster_epoch: str,
    session_date: str,
    market_inputs: Optional[Dict],
    calculations: Optional[List[str]],
    reporting_currency: Optional[str],
    mapping_version: str,
    engine_version: str,
    result_schema: str,
    precision: Optional[str] = None,
    terms_hash: Optional[str] = None,
) -> str:
    """`sha256:`-prefixed canonical hash of the inputs that can change the result.

    Serialized with sorted keys and no whitespace, and `calculations` sorted, so requests
    that differ only in ordering share a key.
    """
    payload = {
        "keyVersion": WORKLOAD_KEY_VERSION,
        "bundleId": bundle_id,
        "clusterEpoch": cluster_epoch,
        "sessionDate": session_date,
        # The market-input block as the caller gave it.
        "marketInputs": market_inputs,
        "calculations": sorted(calculations) if calculations else None,
        "reportingCurrency": reporting_currency,
        "mappingVersion": mapping_version,
        "engineVersion": engine_version,
        # A schema change can change the document even when the numbers do not.
        "resultSchema": result_schema,
        # A float32 and a float64 run are different computations. Not passed by the
        # current caller (engine.api.traderx_routes), and neither is terms_hash.
        "precision": precision,
        "termsHash": terms_hash,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


@dataclass
class Attempt:
    """One computation attempt. Immutable once terminal (`complete`/`fail` refuse to run
    twice). Publishing to the durable store happens inside the terminal transition, so a
    caller cannot forget it."""
    attempt_id: str
    workload_key: str
    submission_id: Optional[str]
    state: str = STATE_RUNNING
    result: Optional[Dict] = None
    reason: Optional[str] = None
    #: The durable store, if any. Excluded from equality and repr.
    _store: Optional[object] = field(default=None, compare=False, repr=False)

    @property
    def is_terminal(self) -> bool:
        return self.state in (STATE_COMPLETED, STATE_FAILED)

    def complete(self, result: Dict) -> None:
        if self.is_terminal:
            raise ValueError(
                f"attempt {self.attempt_id} is already {self.state}; attempts are "
                f"immutable once terminal so a retry cannot overwrite a recorded "
                f"outcome"
            )
        # Publish before transitioning: if publication raises, the attempt stays running
        # and retryable, rather than completed in memory with nothing on disk.
        self._publish(state=STATE_COMPLETED, result=result)
        self.state = STATE_COMPLETED
        self.result = result

    def fail(self, reason: str) -> None:
        """Record the attempt as failed, publishing it if there is a store.

        Unlike `complete`, the state changes even if publication fails (the error is
        re-raised): the attempt did fail, and leaving it running would make a coordinator
        wait forever. After a restart an unpublished failure reads as `UNKNOWN_WORKLOAD`.
        """
        if self.is_terminal:
            raise ValueError(
                f"attempt {self.attempt_id} is already {self.state}; attempts are "
                f"immutable once terminal"
            )
        try:
            self._publish(state=STATE_FAILED, reason=reason)
        finally:
            self.state = STATE_FAILED
            self.reason = reason

    def _publish(
        self,
        *,
        state: str,
        result: Optional[Dict] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Commit this terminal attempt to the durable store, if any. Runs before the state
        changes, so the outcome is passed in. A publication failure is raised, so the caller
        does not report unpublished work as done."""
        if self._store is None:
            return
        self._store.publish(
            attempt_id=self.attempt_id,
            workload_key=self.workload_key,
            state=state,
            result=result,
            reason=reason,
            submission_id=self.submission_id,
        )


def _attempt_from_manifest(manifest: Dict) -> Attempt:
    """Rebuild an `Attempt` from a published manifest, without a store (it is terminal
    and must never publish again)."""
    return Attempt(
        attempt_id=manifest["attemptId"],
        workload_key=manifest["workloadKey"],
        submission_id=manifest.get("submissionId"),
        state=manifest["state"],
        result=manifest.get("result"),
        reason=manifest.get("reason"),
    )


class AttemptStore:
    """Attempts indexed by workload key and submission id. Thread-safe (the API serves
    requests from a thread pool, and the submission-id check-then-create must be atomic).
    With a `ResultStore`, terminal attempts are published and reads fall back to the store
    (I-08)."""

    def __init__(self, store=None) -> None:
        self._lock = threading.Lock()
        self._attempts: Dict[str, Attempt] = {}
        #: workload key -> attempt ids, most recent last.
        self._by_workload: Dict[str, List[str]] = {}
        #: submission id -> attempt id, for idempotent retry recovery.
        self._by_submission: Dict[str, str] = {}
        #: Durable backing store, or None for memory-only.
        self._store = store

    def start(self, key: str, submission_id: Optional[str] = None) -> Attempt:
        """Create a running attempt, or return the existing one for a repeated
        `submission_id` (so a lost response can be retried safely). A submission id reused
        for a different workload key raises `SubmissionIdConflict`."""
        with self._lock:
            if submission_id is not None:
                existing = self._by_submission.get(submission_id)
                if existing is None and self._store is not None:
                    # Idempotency must survive a restart: check published attempts too.
                    published = self._store.find_by_submission(submission_id)
                    if published is not None:
                        attempt = self._rehydrate(published)
                        existing = attempt.attempt_id
                if existing is not None:
                    attempt = self._attempts[existing]
                    if attempt.workload_key != key:
                        raise SubmissionIdConflict(
                            f"submissionId {submission_id!r} was already used for "
                            f"workload {attempt.workload_key}, but this submission is "
                            f"for {key}. A submission id identifies one computation; "
                            f"reusing it for different inputs would return a result "
                            f"computed from those other inputs. Use a new "
                            f"submissionId for a different workload."
                        )
                    return attempt

            attempt = Attempt(
                attempt_id=str(uuid.uuid4()),
                workload_key=key,
                submission_id=submission_id,
                _store=self._store,
            )
            self._attempts[attempt.attempt_id] = attempt
            self._by_workload.setdefault(key, []).append(attempt.attempt_id)
            if submission_id is not None:
                self._by_submission[submission_id] = attempt.attempt_id
            return attempt

    def _rehydrate(self, manifest: Dict) -> Attempt:
        """Index a published attempt into memory (caller holds the lock). An attempt
        already in memory is returned as is, never replaced by the disk copy."""
        attempt_id = manifest["attemptId"]
        cached = self._attempts.get(attempt_id)
        if cached is not None:
            return cached
        attempt = _attempt_from_manifest(manifest)
        self._attempts[attempt_id] = attempt
        ids = self._by_workload.setdefault(attempt.workload_key, [])
        if attempt_id not in ids:
            ids.append(attempt_id)
        if attempt.submission_id is not None:
            self._by_submission.setdefault(attempt.submission_id, attempt_id)
        return attempt

    def get(self, attempt_id: str) -> Optional[Attempt]:
        """One attempt by id, from memory or the durable store (ids stay addressable
        across restarts)."""
        with self._lock:
            cached = self._attempts.get(attempt_id)
            if cached is not None:
                return cached
            if self._store is None:
                return None
            manifest = self._store.read_attempt(attempt_id)
            if manifest is None:
                return None
            return self._rehydrate(manifest)

    def lookup(self, key: str) -> Optional[Attempt]:
        """The attempt a lookup reports for `key`: the most recent completed one, else the
        most recent of any state.

        Memory first (it alone knows running attempts), then the store; consulting the
        store first could let a published result hide a running retry.
        """
        with self._lock:
            ids = self._by_workload.get(key) or []
            attempts = [self._attempts[i] for i in ids]
            completed = [a for a in attempts if a.state == STATE_COMPLETED]
            if completed:
                return completed[-1]

            if self._store is not None:
            # ResultStore.lookup serves only successful attempts and reconciles the
            # pointer with a manifest scan (the crash window between publish and pointer).
                published = self._store.lookup(key)
                if published is not None:
                    return self._rehydrate(published)

            if attempts:
                return attempts[-1]
            return None

    def clear(self) -> None:
        """Drop every attempt, in memory and in the durable store. For tests only (no route
        reaches it); clearing the store keeps tests isolated, since workload keys are
        deterministic."""
        with self._lock:
            self._attempts.clear()
            self._by_workload.clear()
            self._by_submission.clear()
            if self._store is not None:
                self._store.clear()
