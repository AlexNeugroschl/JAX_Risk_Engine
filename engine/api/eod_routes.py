"""
W1.6.4 -- the EOD HTTP routes.

**This is where the integration boundary becomes reachable.** Until now
`engine.integration` was a well-tested library with no service in front of
it: `capabilities()` existed but was not routed (W0.9), and the durable
lookup (W0.8) had no endpoint at all. This module is the transport layer
that lands both, plus bundle submission.

**Direction of dependency: `engine.api` imports `engine.integration`, never
the reverse.** `engine/integration/` deliberately imports no FastAPI, no
Pydantic, no JAX and no pricer from the simulation path -- an invariant
enforced by `tests/test_integration_pipeline.py::TestPackageImportsNoSimulationPricer`.
Putting routes inside that package would break it, so they live here.

**The EOD result is returned as a plain dict, not a Pydantic model.** That
is a deliberate departure from `engine/api/schemas.py`'s wrap-every-
dataclass approach, for a specific reason: the result document's contract is
the published JSON Schema (W1.6.2), and re-describing it in Pydantic would
create a *second* definition that can drift from the first. One schema, one
source of truth, validated in tests against real documents. A Pydantic model
here would add a layer that can disagree with the contract this engine
publishes.

**Submission is synchronous, unlike `/portfolio/price`.** The async job
pattern there exists because a 4096-scenario Monte Carlo portfolio measured
~52 seconds. The EOD path is closed-form discounted cashflows over a handful
of rows -- the delivered fixtures price in well under a second -- so holding
the connection is honest rather than fragile. If a bundle large enough to
need async ever arrives, the attempt store already carries the state machine
to support it (`engine.integration.workload`), and the route can start
returning `202` without the lookup contract changing.

**Every refusal keeps its HTTP status truthful.** A bundle that fails hash
verification is a `422` (the request was well-formed, its content was not);
an unresolvable market-input request is a `400`; an unknown workload is the
only `404`. A refused *instrument* is not an HTTP error at all -- it is a
`200` carrying a result whose coverage block says what was refused, because
the refusal is the answer, not a failure to answer.
"""
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from engine.integration.bundle import BundleIntegrityError, load_bundle
from engine.integration.capabilities import ENGINE_VERSION, capabilities
from engine.integration.market_inputs import MarketInputsNotSupplied
from engine.integration.normalize import MAPPING_VERSION
from engine.integration.pipeline import price_bundle
from engine.integration.publication import (
    PublicationError,
    ResultStore,
    default_store_root,
)
from engine.integration.schema import (
    RESULT_SCHEMA_VERSION,
    capability_schema,
    result_schema,
)
from engine.integration.terms import TermsJoinError
from engine.integration.workload import (
    STATE_COMPLETED,
    STATE_FAILED,
    UNKNOWN_WORKLOAD,
    AttemptStore,
    SubmissionIdConflict,
    workload_key,
)

router = APIRouter(prefix="/eod", tags=["eod"])

#: The durable result store backing the attempt store (W0.8). Rooted at
#: `JAX_EOD_STORE_ROOT` when set -- see
#: `engine.integration.publication.default_store_root`.
#:
#: **Constructed at import, not per request.** A store object is a path and
#: a lock; making one per request would be harmless but would also make
#: "which store am I talking to?" a per-request question, and the whole
#: point of the durable store is that it is the *same* one across requests
#: and across restarts.
RESULT_STORE = ResultStore(default_store_root())

#: The process-wide attempt store, backed by the durable store above.
#: Running state remains in-process (it is never published, deliberately --
#: an in-flight computation is not a result); completed and failed attempts
#: survive a restart. See `engine.integration.workload` and I-08.
STORE = AttemptStore(store=RESULT_STORE)


class EodSubmissionSchema(BaseModel):
    """An EOD pricing submission.

    Only `bundlePath` is required. `marketInputs` is optional in the same
    way it is optional for `price_bundle`: omitting it is legal and means
    nothing gets priced, rather than something getting priced against an
    assumed curve. There is no default profile here, deliberately -- the
    fallback this contract refuses to have would have to be introduced
    right at this layer, so its absence is stated rather than implied.
    """
    bundlePath: str = Field(
        description="Filesystem path to the bundle directory to price.",
    )
    marketInputs: Optional[Dict] = Field(
        default=None,
        description=(
            "The W0.6 market-input block, e.g. "
            '{"mode": "assumed-profile", "assumedProfileId": "flat-3pct-v1"}. '
            "Omitting it is legal and prices nothing; no curve is ever "
            "substituted."
        ),
    )
    submissionId: Optional[str] = Field(
        default=None,
        description=(
            "Caller-generated idempotency key. Retrying a lost response with the "
            "same value recovers the SAME attempt rather than starting a second. "
            "A deliberate repeat run uses a new one."
        ),
    )
    reuseExistingResult: bool = Field(
        default=True,
        description=(
            "Whether a previously completed result for this workload may be "
            "served. False means 'do not serve me a cached result' -- it does "
            "NOT mean 'always start a new attempt every time you see this "
            "request', which would defeat submissionId idempotency."
        ),
    )
    calculations: Optional[List[str]] = Field(
        default=None,
        description="Requested calculation set; part of the workload key.",
    )
    reportingCurrency: Optional[str] = Field(default=None)


@router.get("/capabilities")
def get_capabilities() -> Dict:
    """The capability document (W0.9, finally routed in W1.6.4).

    Lets a coordinator determine **before submitting** whether a bundle is
    priceable, which is what makes "no silent exclusions" enforceable
    rather than aspirational. Derived from the allowlist on every call, so
    it cannot go stale relative to the engine that serves it.
    """
    return capabilities()


@router.get("/schemas/result")
def get_result_schema() -> Dict:
    """The machine-readable JSON Schema for the result document (W1.6.2).

    Served so a consumer's validator can fetch and pin it rather than
    reimplementing the shape from prose.
    """
    return result_schema()


@router.get("/schemas/capabilities")
def get_capability_schema() -> Dict:
    """The machine-readable JSON Schema for the capability document."""
    return capability_schema()


def _key_for(request: EodSubmissionSchema, bundle) -> str:
    """The workload key for this submission.

    Note what is absent: `submissionId` is **not** in the key. It
    identifies a request, not a computation -- see
    `engine.integration.workload`'s docstring.
    """
    return workload_key(
        bundle_id=bundle.bundle_id,
        cluster_epoch=bundle.cluster_epoch,
        session_date=bundle.session_date,
        market_inputs=request.marketInputs,
        calculations=request.calculations,
        reporting_currency=request.reportingCurrency,
        mapping_version=MAPPING_VERSION,
        engine_version=ENGINE_VERSION,
        result_schema=RESULT_SCHEMA_VERSION,
    )


def _record_failure(attempt, reason: str) -> None:
    """Marks an attempt failed without letting bookkeeping mask the cause.

    **A publication failure here is swallowed, and that is the opposite of
    the rule on the success path** -- deliberately. On success, an
    unpublished result must become a loud error, because the caller would
    otherwise be told work is discoverable when it is not. Here the caller
    is already receiving an error that names the real problem: letting a
    store write failure replace `TERMS_ARTIFACT_UNUSABLE` with a disk
    message would hide the thing they actually need to fix, and would
    change a `422` the coordinator must not retry into a `500` it will.

    The cost is bounded to durability: `Attempt.fail` sets the state
    whether or not publication succeeds, so *this* process still reports
    the attempt as `failed`. Only a lookup after a **restart** loses it,
    and it then reads `UNKNOWN_WORKLOAD` -- the weaker of the two states
    but not a wrong one. What must not happen, and does not, is the
    attempt being left `running`: that would tell a coordinator to wait for
    an answer that is never coming. The failure also reached the caller
    synchronously, which is where it matters.
    """
    try:
        attempt.fail(reason)
    except PublicationError:
        pass


@router.post("/price")
def submit_eod_price(request: EodSubmissionSchema) -> Dict:
    """Prices one EOD bundle and publishes the attempt.

    **Bundle integrity failures are `422`, not `500`.** The request was
    well-formed; its referenced content did not verify. That distinction
    matters to a coordinator deciding whether to retry (never, for a hash
    failure -- the bytes will not change) or to escalate.

    **A market-input failure fails the whole job**, because market data is
    the shared basis every price is measured against rather than a property
    of one instrument. It is a `400`: the caller asked for something
    unresolvable and can fix it.

    **An instrument the engine refuses is not an error.** It comes back
    `200` inside a result whose coverage block names the refusal. Returning
    an HTTP error for a refusal would make "we correctly declined to guess"
    indistinguishable from "we broke".
    """
    try:
        bundle = load_bundle(request.bundlePath)
    except BundleIntegrityError as exc:
        # **A missing bundle lands here too, deliberately.** W0.1 step 4
        # established that a *missing* artifact is an integrity failure
        # rather than an absence -- the distinction between an empty
        # contracts file (valid) and a missing one (the bundle does not
        # describe what it claims to). Re-classifying "no manifest" as a
        # 404 here would reintroduce at the transport layer the exact
        # conflation W0.1 exists to prevent, so it is not done.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"reason": "BUNDLE_INTEGRITY_FAILED", "detail": str(exc)},
        ) from exc

    key = _key_for(request, bundle)

    # Serve a cached result only when the caller allows it. `False` means
    # "don't serve me a cache", and is honoured without disabling
    # submissionId idempotency below.
    if request.reuseExistingResult:
        existing = STORE.lookup(key)
        if existing is not None and existing.state == STATE_COMPLETED:
            return {
                "workloadKey": key,
                "attemptId": existing.attempt_id,
                "state": STATE_COMPLETED,
                "reused": True,
                "result": existing.result,
            }

    try:
        attempt = STORE.start(key, submission_id=request.submissionId)
    except SubmissionIdConflict as exc:
        # 409, not 400: the request is well-formed and would be valid on its
        # own -- it conflicts with a submission id already bound to different
        # inputs. Honouring it would return a result computed from those
        # other inputs.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"reason": "SUBMISSION_ID_CONFLICT", "detail": str(exc)},
        ) from exc

    # A repeated submissionId recovers the attempt that already exists,
    # including one that already finished -- the retry-a-lost-response case.
    if attempt.is_terminal:
        return {
            "workloadKey": key,
            "attemptId": attempt.attempt_id,
            "state": attempt.state,
            "reused": True,
            "result": attempt.result,
            "reason": attempt.reason,
        }

    try:
        result = price_bundle(bundle, request.marketInputs)
    except MarketInputsNotSupplied as exc:
        _record_failure(attempt, f"MARKET_INPUTS_NOT_SUPPLIED: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"reason": "MARKET_INPUTS_NOT_SUPPLIED", "detail": str(exc)},
        ) from exc
    except TermsJoinError as exc:
        _record_failure(attempt, f"TERMS_ARTIFACT_UNUSABLE: {exc}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"reason": "TERMS_ARTIFACT_UNUSABLE", "detail": str(exc)},
        ) from exc
    except Exception as exc:
        # The attempt is recorded as failed before the error propagates, so
        # a later lookup reports `failed` with a reason rather than
        # `UNKNOWN_WORKLOAD` -- which would wrongly invite a resubmission.
        _record_failure(attempt, f"{type(exc).__name__}: {exc}")
        raise

    try:
        attempt.complete(result.to_dict())
    except PublicationError as exc:
        # **The computation succeeded and the record of it did not land.**
        # Reporting 200 here would tell the coordinator its result is
        # published and discoverable when a lookup will not find it -- so
        # the one thing it must not do is stop retrying. A 500 is right:
        # this is an infrastructure failure on my side, the request was
        # valid, and the workload key makes the retry the same computation.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "reason": "RESULT_NOT_PUBLISHED",
                "detail": (
                    f"the bundle priced successfully but the result could not be "
                    f"durably published, so it is not discoverable by lookup: {exc}"
                ),
            },
        ) from exc

    return {
        "workloadKey": key,
        "attemptId": attempt.attempt_id,
        "state": STATE_COMPLETED,
        "reused": False,
        "result": attempt.result,
    }


@router.get("/results/by-workload/{workload_key_value}")
def lookup_by_workload(workload_key_value: str) -> Dict:
    """Durable result lookup (W0.8), finally routed.

    **Four distinct states, and only one of them is a 404** (v4 §4.2):

    | State            | Response                                  |
    |------------------|-------------------------------------------|
    | Never submitted  | `404 UNKNOWN_WORKLOAD`                    |
    | Accepted, running| `200 {"state": "running", ...}`           |
    | Accepted, failed | `200 {"state": "failed", "reason": ...}`  |
    | Completed        | `200 {"state": "completed", "result": …}` |

    A bare 404 for both "unknown" and "running" is what invites a
    coordinator to launch a duplicate overnight batch against work already
    in flight.

    **Returns the most recent *successful* attempt**, never a failed,
    partial or in-flight one when a successful one exists.
    """
    attempt = STORE.lookup(workload_key_value)
    if attempt is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "reason": UNKNOWN_WORKLOAD,
                "detail": (
                    "no attempt has ever been submitted for this workload key. This "
                    "is distinct from an accepted job that is still running, which "
                    "returns 200 with state 'running'."
                ),
            },
        )

    payload = {
        "workloadKey": workload_key_value,
        "attemptId": attempt.attempt_id,
        "state": attempt.state,
    }
    if attempt.state == STATE_COMPLETED:
        payload["result"] = attempt.result
    elif attempt.state == STATE_FAILED:
        payload["reason"] = attempt.reason
    return payload


@router.get("/attempts/{attempt_id}")
def get_attempt(attempt_id: str) -> Dict:
    """One attempt by its id.

    Attempts are permanently addressable and immutable once terminal, so
    this answers "what did that specific run produce?" even after a later
    attempt superseded it.
    """
    attempt = STORE.get(attempt_id)
    if attempt is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"reason": "UNKNOWN_ATTEMPT", "detail": f"no attempt {attempt_id}"},
        )
    payload = {
        "attemptId": attempt.attempt_id,
        "workloadKey": attempt.workload_key,
        "state": attempt.state,
        "submissionId": attempt.submission_id,
    }
    if attempt.state == STATE_COMPLETED:
        payload["result"] = attempt.result
    elif attempt.state == STATE_FAILED:
        payload["reason"] = attempt.reason
    return payload
