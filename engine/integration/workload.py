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

**Storage is in-process, and that is a stated limitation, not an
oversight.** `docs/known-issues.md` I-08 records it: a restart loses
*running*-state knowledge. The recovery design that makes this survivable --
publication via a content-addressed manifest, with lookup falling back to a
scan rather than trusting a pointer -- is specified in plan §W0.8 and is not
built here, because there is no persistent artifact store to scan yet. What
*is* built is the state machine and the key, so the day a store arrives the
semantics do not change. A lost in-memory job is an infrastructure event;
this module's job is to make sure it is never a *financial* one, by never
serving a partial or stale result as a complete one.
"""
import hashlib
import json
import threading
import uuid
from dataclasses import dataclass
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
    """
    attempt_id: str
    workload_key: str
    submission_id: Optional[str]
    state: str = STATE_RUNNING
    result: Optional[Dict] = None
    reason: Optional[str] = None

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
        self.state = STATE_COMPLETED
        self.result = result

    def fail(self, reason: str) -> None:
        if self.is_terminal:
            raise ValueError(
                f"attempt {self.attempt_id} is already {self.state}; attempts are "
                f"immutable once terminal"
            )
        self.state = STATE_FAILED
        self.reason = reason


class AttemptStore:
    """In-process attempt store keyed by workload key and submission id.

    **Thread-safe**, because the FastAPI app serves requests from a thread
    pool: two concurrent submissions of the same `submissionId` must
    resolve to one attempt, and the check-then-create that guarantees it is
    not atomic on its own.

    See the module docstring for why this is in-process and what that does
    and does not cost (I-08).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._attempts: Dict[str, Attempt] = {}
        #: workload key -> attempt ids, most recent last.
        self._by_workload: Dict[str, List[str]] = {}
        #: submission id -> attempt id, for idempotent retry recovery.
        self._by_submission: Dict[str, str] = {}

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
            )
            self._attempts[attempt.attempt_id] = attempt
            self._by_workload.setdefault(key, []).append(attempt.attempt_id)
            if submission_id is not None:
                self._by_submission[submission_id] = attempt.attempt_id
            return attempt

    def get(self, attempt_id: str) -> Optional[Attempt]:
        with self._lock:
            return self._attempts.get(attempt_id)

    def lookup(self, key: str) -> Optional[Attempt]:
        """The attempt a lookup should report for this workload key.

        **Prefers the most recent *successful* attempt** (plan §W0.8). Only
        when there is no successful one does it fall back to reporting the
        most recent attempt's state -- so a later failure never hides an
        earlier success, and a running retry never masks a completed
        result the coordinator could already use.
        """
        with self._lock:
            ids = self._by_workload.get(key)
            if not ids:
                return None
            attempts = [self._attempts[i] for i in ids]
            completed = [a for a in attempts if a.state == STATE_COMPLETED]
            if completed:
                return completed[-1]
            return attempts[-1]

    def clear(self) -> None:
        """Drops every attempt. Test support only -- there is deliberately
        no HTTP route that reaches this."""
        with self._lock:
            self._attempts.clear()
            self._by_workload.clear()
            self._by_submission.clear()
