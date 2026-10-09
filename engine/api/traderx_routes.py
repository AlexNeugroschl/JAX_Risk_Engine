"""
HTTP routes of the TraderX path (`/eod`): capabilities, result schemas, bundle
submission and durable result lookup.

`engine.api` imports `engine.traderx`, never the reverse: the integration package
imports no FastAPI, Pydantic, JAX or simulation pricer
(`tests/test_traderx_pipeline.py::TestPackageImportsNoSimulationPricer`).

The result is returned as a plain dict, not a Pydantic model: its contract is the published
JSON Schema, and a second definition could drift from it.

Submission is synchronous: EOD bundles are closed-form pricing of a few rows, well under a
second. `engine.traderx.workload`'s state machine would support an async variant.

Status codes: a bundle failing integrity checks (including a missing file) or an unusable
terms artifact is 422; unresolvable market inputs 400; a submission id bound to different
inputs 409; an unknown workload or attempt 404; a result that priced but could not be
published 500. A refused instrument is a 200 whose coverage block names the refusal.
"""
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from engine.traderx.bundle import BundleIntegrityError, load_bundle
from engine.traderx.capabilities import ENGINE_VERSION, capabilities
from engine.traderx.market_inputs import MarketInputsNotSupplied
from engine.traderx.normalize import MAPPING_VERSION
from engine.traderx.pipeline import price_bundle
from engine.traderx.publication import (
    PublicationError,
    ResultStore,
    default_store_root,
)
from engine.traderx.schema import (
    RESULT_SCHEMA_VERSION,
    capability_schema,
    result_schema,
)
from engine.traderx.terms import TermsJoinError
from engine.traderx.workload import (
    STATE_COMPLETED,
    STATE_FAILED,
    UNKNOWN_WORKLOAD,
    AttemptStore,
    SubmissionIdConflict,
    workload_key,
)

router = APIRouter(prefix="/eod", tags=["eod"])

#: Durable result store, rooted at `JAX_EOD_STORE_ROOT` when set
#: (`engine.traderx.publication.default_store_root`). One per process, so every request
#: and restart uses the same store.
RESULT_STORE = ResultStore(default_store_root())

#: Attempt store backed by the durable store. Running attempts are in-process only;
#: completed and failed ones survive a restart (I-08).
STORE = AttemptStore(store=RESULT_STORE)


class EodSubmissionSchema(BaseModel):
    """An EOD pricing submission. Only `bundlePath` is required. Omitting `marketInputs` is
    legal and prices nothing; no curve is ever substituted."""
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
    """The capability document, derived from the allowlist on every call, so a coordinator
    can check whether a bundle is priceable before submitting it."""
    return capabilities()


@router.get("/schemas/result")
def get_result_schema() -> Dict:
    """JSON Schema for the result document."""
    return result_schema()


@router.get("/schemas/capabilities")
def get_capability_schema() -> Dict:
    """The machine-readable JSON Schema for the capability document."""
    return capability_schema()


def _key_for(request: EodSubmissionSchema, bundle) -> str:
    """The workload key for this submission. `submissionId` identifies a request, not a
    computation, so it is not part of the key (see `engine.traderx.workload`)."""
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
    """Mark an attempt failed, swallowing a publication error so it cannot replace the real
    cause (and turn a non-retryable 422 into a retryable 500). The attempt is still marked
    failed in this process; only after a restart would it read as `UNKNOWN_WORKLOAD`."""
    try:
        attempt.fail(reason)
    except PublicationError:
        pass


@router.post("/price")
def submit_eod_price(request: EodSubmissionSchema) -> Dict:
    """Price one EOD bundle and publish the attempt (status codes: see the module
    docstring). A market-input failure fails the whole job, since market data is shared by
    every row."""
    try:
        bundle = load_bundle(request.bundlePath)
    except BundleIntegrityError as exc:
        # A missing bundle is an integrity failure (422), not a 404.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"reason": "BUNDLE_INTEGRITY_FAILED", "detail": str(exc)},
        ) from exc

    key = _key_for(request, bundle)

    # Serve a completed result unless the caller opted out; submissionId idempotency below
    # applies either way.
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
        # 409: the submission id is already bound to different inputs.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"reason": "SUBMISSION_ID_CONFLICT", "detail": str(exc)},
        ) from exc

    # A repeated submissionId returns the existing attempt, including a finished one.
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
        # Record the failure first, so a lookup says `failed` rather than inviting a
        # resubmission with `UNKNOWN_WORKLOAD`.
        _record_failure(attempt, f"{type(exc).__name__}: {exc}")
        raise

    try:
        attempt.complete(result.to_dict())
    except PublicationError as exc:
        # Priced but not published: a 200 would claim the result is discoverable, so the
        # coordinator must keep retrying. The workload key makes the retry the same job.
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
    """Durable result lookup. Only a never-submitted workload is a 404:

    | State            | Response                                  |
    |------------------|-------------------------------------------|
    | Never submitted  | `404 UNKNOWN_WORKLOAD`                    |
    | Accepted, running| `200 {"state": "running", ...}`           |
    | Accepted, failed | `200 {"state": "failed", "reason": ...}`  |
    | Completed        | `200 {"state": "completed", "result": â€¦}` |

    Returns the most recent successful attempt when one exists.
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
    """One attempt by id. Attempts are immutable once terminal and stay addressable after
    a later attempt supersedes them."""
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
