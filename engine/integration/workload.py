"""
W1.6.4 / W0.8 -- the workload key and the durable attempt store.

**The workload key answers "have I already computed exactly this?"** It is a
canonical hash over every input that can change a number: the bundle's own
identity, the market inputs, the calculation set, the adapter and engine
versions. Two submissions with the same key are the same computation and may
share a result; two with different keys must never share one.

**Everything that changes a number is in the key, and nothing else is.**
That cuts both ways and both directions are bugs:

  - *Omitting* an input that matters means a cache hit returns a result
    computed against something else. The plan calls out precision
    specifically ("different precision -> different workload key"), because
    a float32 and a float64 run of identical inputs are different
    computations with different answers.
  - *Including* something that does not matter -- a submission id, a
    timestamp, a retry counter -- means the key changes on every retry and
    the cache never hits, which quietly turns a lookup into a recompute.

**Submission identity is deliberately NOT part of the key** (plan §W0.8,
v4 §4.3). A `submissionId` identifies *a request*; the workload key
identifies *a computation*. Retrying a lost submission response recovers the
same attempt because the submission id is remembered separately -- if it
were in the key, every retry would compute afresh, which is the opposite of
idempotent.

**Four lookup states, never a bare 404 for accepted work** (v4 §4.2). A 404
for both "never submitted" and "still running" invites a coordinator to
resubmit an overnight batch that is already in flight. So `UNKNOWN_WORKLOAD`
is distinguishable from `running`, `failed` and `completed`.

**Lookup returns the most recent *successful* attempt** -- never a failed,
partial, or in-flight one. Attempts are immutable once terminal and
permanently addressable by `attemptId`, so a second attempt never overwrites
a first.

**Storage is in-process, with an optional durable backing store.** By
default this module keeps attempts in memory only, and a restart loses them
-- the limitation `docs/known-issues.md` I-08 records. Given a
`engine.integration.publication.ResultStore`, terminal attempts are also
**published** through the four-step protocol in plan §W0.8, and lookup falls
back to the store's manifest scan when memory does not have the answer. The
state machine is identical either way; what the store changes is whether it
survives the process.

**Running state is still memory-only, and deliberately so.** Only *terminal*
attempts are published. A running attempt has no result to commit, and
writing one would make an in-flight computation discoverable as a finished
answer -- so after a restart a job that was running is reported as unknown
rather than as something it is not. That is an infrastructure event, not a
financial one: the coordinator resubmits and the workload key makes the
recomputation identical. What must never happen, and does not, is a partial
or stale result being served as a complete one.
"""
import hashlib
import json
import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

#: The four lookup states (plan §W0.8). `UNKNOWN_WORKLOAD` is the only one
#: that is a 404; the other three are 200s carrying a state, because the
#: work *was* accepted and saying "not found" would be false.
STATE_RUNNING = "running"
STATE_FAILED = "failed"
STATE_COMPLETED = "completed"
UNKNOWN_WORKLOAD = "UNKNOWN_WORKLOAD"

#: Version of the key derivation itself. Part of the key, so that changing
#: how keys are computed cannot make an old key collide with a new one --
#: the old and new derivations produce disjoint key spaces by construction.
WORKLOAD_KEY_VERSION = "workload-key-v1"


class SubmissionIdConflict(ValueError):
    """A `submissionId` was reused against a *different* workload key.

    Raised rather than honoured: returning the existing attempt would hand
    the caller a result computed from different inputs. See
    `AttemptStore.start`.
    """


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
    """Canonical hash of everything that can change the resulting numbers.

    **Canonical means order-independent and formatting-independent.** The
    payload is serialized with sorted keys and no incidental whitespace, so
    two requests that differ only in JSON key order produce the *same* key
    -- otherwise a cache would miss on a cosmetic difference and recompute
    an overnight batch.

    `calculations` is sorted before hashing for the same reason: asking for
    `["npv", "theta"]` and `["theta", "npv"]` is one computation.

    Returns a `sha256:`-prefixed hex digest. The prefix names the algorithm
    rather than leaving a consumer to infer it from the length -- the same
    convention the bundle manifests use.
    """
    payload = {
        "keyVersion": WORKLOAD_KEY_VERSION,
        "bundleId": bundle_id,
        "clusterEpoch": cluster_epoch,
        "sessionDate": session_date,
        # The resolved market-input block, not the raw request: two
        # requests naming the same profile differently still price against
        # the same curve, and two naming the same profile id resolve
        # identically.
        "marketInputs": market_inputs,
        "calculations": sorted(calculations) if calculations else None,
        "reportingCurrency": reporting_currency,
        "mappingVersion": mapping_version,
        "engineVersion": engine_version,
        # A schema change can change the document a consumer receives even
        # when every number is identical, so it separates the key space.
        "resultSchema": result_schema,
        # Plan §W0.8 names precision explicitly: a float32 and a float64
        # run of identical inputs are different computations.
        "precision": precision,
        "termsHash": terms_hash,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


@dataclass
class Attempt:
    """One computation attempt.

    Immutable once terminal: `complete()` and `fail()` refuse to run twice,
    so a second attempt can never overwrite a first's outcome (plan §W0.8).
    An attempt is permanently addressable by `attempt_id` regardless of how
    it ended -- a failed attempt is still a record of what was tried.

    **Publication happens on the terminal transition**, through the optional
    `_store`. It is done *inside* `complete()`/`fail()` rather than left to
    the caller because a caller that forgets produces an attempt that is
    complete in memory and absent from disk -- which looks exactly like a
    successful publication until a restart, and then looks exactly like work
    that was never submitted.
    """
    attempt_id: str
    workload_key: str
    submission_id: Optional[str]
    state: str = STATE_RUNNING
    result: Optional[Dict] = None
    reason: Optional[str] = None
    #: The durable store, if this attempt's `AttemptStore` has one. Excluded
    #: from equality and repr: it is plumbing, not part of the attempt's
    #: identity, and two attempts do not differ because they were published
    #: through different store objects.
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
        # **Publish first, transition second.** If publication raises, this
        # attempt must be left exactly as it was -- still running, still
        # retryable -- because an attempt marked terminal in memory with no
        # manifest on disk is a result the process reports as completed and
        # no restart can ever find. See `_publish`.
        self._publish(state=STATE_COMPLETED, result=result)
        self.state = STATE_COMPLETED
        self.result = result

    def fail(self, reason: str) -> None:
        """Records this attempt as failed, publishing it if there is a store.

        **Unlike `complete()`, the in-memory transition survives a
        publication failure.** The two paths differ because what is at
        stake differs. On the success path a failed publication must undo
        the transition: there is a real result, it is worth retrying for,
        and an attempt left `completed` with nothing on disk claims a
        durability it does not have. Here there is no result to save and
        nothing to retry -- the attempt *did* fail, that is a fact about
        this attempt regardless of whether the record of it landed, and
        leaving it `running` would report an in-flight job to a coordinator
        that would then wait forever for an answer that is never coming.

        So publication is attempted first and its failure is re-raised for
        the caller to decide about (`eod_routes._record_failure` swallows
        it, for reasons documented there), but the state is set either way.
        The degraded outcome is a failure that a *restart* cannot see --
        reported as `UNKNOWN_WORKLOAD`, which is honest -- rather than one
        this process misreports as still running.
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
        """Commits this terminal attempt to the durable store, if there is
        one.

        **The outcome is passed in rather than read off `self`,** because
        this runs *before* the attempt transitions. That ordering is the
        point: the manifest is the commit point for the in-memory attempt
        too, so a store failure leaves nothing half-applied.

        **A publication failure is raised, not swallowed.** The caller is
        about to tell a coordinator its work is done; if the record of that
        work did not land, the coordinator must find out now rather than
        discover it after a restart, when the same submission looks like it
        was never made. This is the one place where failing loudly costs a
        successful computation and is still right -- the computation is
        reproducible from the workload key, a false "published" is not
        recoverable at all.
        """
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
    """Rebuilds an `Attempt` from a published manifest.

    Deliberately **not** given the store: a rehydrated attempt is already
    terminal, so it must never publish again. `complete()`/`fail()` would
    refuse anyway, but leaving the store off makes the intent structural
    rather than dependent on that guard.
    """
    return Attempt(
        attempt_id=manifest["attemptId"],
        workload_key=manifest["workloadKey"],
        submission_id=manifest.get("submissionId"),
        state=manifest["state"],
        result=manifest.get("result"),
        reason=manifest.get("reason"),
    )


class AttemptStore:
    """Attempt store keyed by workload key and submission id.

    **Thread-safe**, because the FastAPI app serves requests from a thread
    pool: two concurrent submissions of the same `submissionId` must
    resolve to one attempt, and the check-then-create that guarantees it is
    not atomic on its own.

    **In-memory by default; durable when given a store.** With a
    `engine.integration.publication.ResultStore`, terminal attempts are
    published as they finish and every read falls back to the store when
    memory does not have the answer -- so a restart loses running state
    but never a completed result. See the module docstring and I-08.
    """

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
        """Creates a running attempt, or returns the existing one for a
        repeated `submission_id`.

        **A repeated submission id recovers the same attempt rather than
        starting a second** (v4 §4.3). That is what makes a lost response
        safe to retry: the coordinator gets back the attempt it already
        has, not a duplicate overnight computation.

        **But only when the workload matches.** A submission id reused
        against a *different* workload key is a caller error, and it is
        raised rather than honoured. Returning the existing attempt would
        hand back a result computed from **different inputs** -- the
        silently-wrong-number failure this whole boundary exists to
        prevent, reached through the idempotency path rather than through
        a pricer. Idempotency means "this exact request, again"; it cannot
        mean "whatever I sent last time under this name".
        """
        with self._lock:
            if submission_id is not None:
                existing = self._by_submission.get(submission_id)
                if existing is None and self._store is not None:
                    # **Idempotency has to survive a restart to be worth
                    # anything.** Without this lookup, a coordinator
                    # retrying a lost response after the engine bounced
                    # gets a *second* attempt for work that already
                    # completed -- the exact duplicate-overnight-batch this
                    # id exists to prevent, just reached through a restart
                    # instead of through a race.
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
        """Indexes a published attempt back into memory.

        Caller must hold `self._lock`. Idempotent: a manifest already in
        memory returns the in-memory attempt rather than replacing it, so a
        scan can never overwrite live state with a copy read off disk.
        """
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
        """One attempt by id, from memory or from the durable store.

        The store fallback is what keeps `attemptId` *permanently*
        addressable (plan §W0.8) rather than addressable for as long as the
        process happens to live.
        """
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
        """The attempt a lookup should report for this workload key.

        **Prefers the most recent *successful* attempt** (plan §W0.8). Only
        when there is no successful one does it fall back to reporting the
        most recent attempt's state -- so a later failure never hides an
        earlier success, and a running retry never masks a completed
        result the coordinator could already use.

        **The durable store is consulted before giving up**, not before
        memory. Memory is authoritative for *running* state, which is never
        published; the store is authoritative across restarts. Consulting
        the store first would let a completed attempt shadow a running
        retry that memory knows about, inverting the precedence above.
        """
        with self._lock:
            ids = self._by_workload.get(key) or []
            attempts = [self._attempts[i] for i in ids]
            completed = [a for a in attempts if a.state == STATE_COMPLETED]
            if completed:
                return completed[-1]

            if self._store is not None:
                # Only a *successful* published attempt is served here --
                # `ResultStore.lookup` enforces that, and it is also where
                # the pointer/scan reconciliation lives (the v3 crash
                # window between publish and pointer advance).
                published = self._store.lookup(key)
                if published is not None:
                    return self._rehydrate(published)

            if attempts:
                return attempts[-1]
            return None

    def clear(self) -> None:
        """Drops every attempt, in memory **and** in the durable store.

        Test support only -- there is deliberately no HTTP route that
        reaches this. Clearing the durable store too is what keeps tests
        isolated now that attempts outlive the process: a memory-only clear
        would leave published manifests on disk for the next test (and the
        next *run*) to find, and a workload key is deterministic, so that
        leak would look like a cache hit rather than like stale state.
        """
        with self._lock:
            self._attempts.clear()
            self._by_workload.clear()
            self._by_submission.clear()
            if self._store is not None:
                self._store.clear()
