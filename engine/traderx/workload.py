"""
Workload key and attempt store for EOD submissions.

The workload key is a canonical hash of what can change a result: bundle identity, market
inputs, calculation set, reporting currency, adapter/engine/schema versions (and, when given,
precision and a terms hash). Same key, same computation; results may be shared. A
`submissionId` identifies a request, not a computation, so it is not in the key: a retry
with the same id recovers the same attempt. An id binds to one attempt, and so to one
workload, before anything is returned, a cached result included, so it can never carry
another workload's result (I-57; `AttemptStore.submit`).

Lookup has four states and only "never submitted" is `UNKNOWN_WORKLOAD` (a 404); running,
failed and completed are reported as such, so accepted work is never mistaken for unknown.
Lookup prefers the most recent successful attempt. Attempts are immutable once terminal and
addressable by `attemptId`.

Storage is in memory, optionally backed by an `engine.traderx.publication.ResultStore`
that publishes terminal attempts and submission bindings and is consulted when memory has no
answer (I-08). Running attempts are never published, so after a restart a job that was
running reads as unknown and is resubmitted; the workload key makes the recomputation
identical, and its id, still bound to that workload, is bound to the new attempt.
"""
import hashlib
import json
import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple

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


class Submission(NamedTuple):
    """What `AttemptStore.submit` hands a caller: the attempt its submission is bound to, and
    whether that attempt existed before (a retry's, or a cached result). A reused attempt is
    reported as it stands, never executed: only the caller that created an attempt runs it."""
    attempt: Attempt
    reused: bool


class AttemptStore:
    """Attempts indexed by workload key and submission id. Thread-safe (the API serves
    requests from a thread pool, and a submission's check-then-bind must be atomic). With a
    `ResultStore`, terminal attempts and submission bindings are published and reads fall
    back to the store (I-08)."""

    def __init__(self, store=None) -> None:
        self._lock = threading.Lock()
        self._attempts: Dict[str, Attempt] = {}
        #: workload key -> attempt ids, most recent last.
        self._by_workload: Dict[str, List[str]] = {}
        #: submission id -> the attempt id it is bound to, for idempotent retry recovery.
        self._by_submission: Dict[str, str] = {}
        #: Durable backing store, or None for memory-only.
        self._store = store

    def submit(
        self,
        key: str,
        submission_id: Optional[str] = None,
        *,
        reuse_existing: bool = False,
    ) -> Submission:
        """Bind one submission of workload `key` to an attempt, atomically, in this order:

        1. A `submission_id` already bound is checked before anything is returned (I-57):
           bound to another workload it raises `SubmissionIdConflict`, even when either
           workload has a cached result; bound to this one, its attempt is returned in
           whatever state it is in, so a retry recovers what the lost response carried.
        2. With `reuse_existing`, the most recent completed attempt of `key` is returned,
           and the id bound to it.
        3. Otherwise a new running attempt, bound to the id, for the caller to execute.

        With a durable store the binding is recorded before it is returned, so it holds
        across a restart; a binding that cannot be recorded raises `PublicationError` and
        binds nothing. An id whose attempt was lost running (in a process that died) is
        rebound to a new attempt of the same workload.
        """
        with self._lock:
            if submission_id is not None:
                bound_key, bound = self._binding(submission_id)
                if bound_key is not None and bound_key != key:
                    raise SubmissionIdConflict(
                        f"submissionId {submission_id!r} was already used for workload "
                        f"{bound_key}, but this submission is for {key}. A submission id "
                        f"identifies one computation; reusing it for different inputs "
                        f"would return a result computed from those other inputs. Use a "
                        f"new submissionId for a different workload."
                    )
                if bound is not None:
                    return Submission(bound, reused=True)

            if reuse_existing:
                cached = self._latest_completed(key)
                if cached is not None:
                    self._bind(submission_id, cached)
                    return Submission(cached, reused=True)

            attempt = Attempt(
                attempt_id=str(uuid.uuid4()),
                workload_key=key,
                submission_id=submission_id,
                _store=self._store,
            )
            # Bound first: an attempt whose binding failed is never indexed.
            self._bind(submission_id, attempt)
            self._attempts[attempt.attempt_id] = attempt
            self._by_workload.setdefault(key, []).append(attempt.attempt_id)
            return Submission(attempt, reused=False)

    def start(self, key: str, submission_id: Optional[str] = None) -> Attempt:
        """`submit` without a cached result: a new running attempt, or the attempt a
        repeated `submission_id` is bound to."""
        return self.submit(key, submission_id).attempt

    def _binding(self, submission_id: str) -> Tuple[Optional[str], Optional[Attempt]]:
        """The workload key `submission_id` is bound to and its attempt (caller holds the
        lock): `(None, None)` if unbound, and `(key, None)` if the attempt is gone (bound
        by a process that died while it ran)."""
        attempt_id = self._by_submission.get(submission_id)
        if attempt_id is not None:
            attempt = self._attempts[attempt_id]
            return attempt.workload_key, attempt
        if self._store is None:
            return None, None
        binding = self._store.find_submission(submission_id)
        if binding is None:
            return None, None
        attempt = self._get(binding["attemptId"])
        if attempt is not None:
            self._by_submission[submission_id] = attempt.attempt_id
        return binding["workloadKey"], attempt

    def _bind(self, submission_id: Optional[str], attempt: Attempt) -> None:
        """Bind `submission_id` to `attempt`, durably first (caller holds the lock)."""
        if submission_id is None:
            return
        if self._store is not None:
            self._store.bind_submission(
                submission_id=submission_id,
                workload_key=attempt.workload_key,
                attempt_id=attempt.attempt_id,
            )
        self._by_submission[submission_id] = attempt.attempt_id

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
            return self._get(attempt_id)

    def _get(self, attempt_id: str) -> Optional[Attempt]:
        """`get`, for a caller that holds the lock."""
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
        most recent of any state in memory."""
        with self._lock:
            completed = self._latest_completed(key)
            if completed is not None:
                return completed
            ids = self._by_workload.get(key)
            return self._attempts[ids[-1]] if ids else None

    def _latest_completed(self, key: str) -> Optional[Attempt]:
        """The most recent completed attempt of `key`, or `None` (caller holds the lock).

        Memory first (it alone knows running attempts), then the store; consulting the
        store first could let a published result hide a running retry.
        """
        for attempt_id in reversed(self._by_workload.get(key) or []):
            attempt = self._attempts[attempt_id]
            if attempt.state == STATE_COMPLETED:
                return attempt
        if self._store is None:
            return None
        # ResultStore.lookup serves only successful attempts and reconciles the pointer
        # with a manifest scan (the crash window between publish and pointer).
        published = self._store.lookup(key)
        return None if published is None else self._rehydrate(published)

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
