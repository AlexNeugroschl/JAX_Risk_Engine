"""
HTTP routes over `engine.run.run`; translation only, no pricing logic (docs/reference/http-api.md;
decisions A-2, A-14). One request for every analytic (I-87):

    POST /runs                       a run request (`RunRequestSchema`): validated, then queued; 202
                                     with its `run_id`
    GET  /runs/{run_id}[?wait=N]     its status, and its result once done; with `wait` (at most 30 s)
                                     the answer comes as soon as the run is finished or N s pass
    POST /runs/{run_id}/cancel       cancel it: at once if pending, at its next phase boundary if
                                     running (I-91)
    GET  /runs/{run_id}/artifacts/{name}/{chunk}
                                     one chunk of an array the request asked for by reference
    GET  /config/defaults            the whole `config` with every default, to copy and edit
    GET  /health, GET /version       the API process and its engine worker; the engine's identity

A submission is validated synchronously, without JAX work (`RunRequestSchema.check`: a refusal is
a 400 naming its field, and no run is created), then its body is written as received to the
durable job queue (`engine.api.job_queue`) and the route returns `202` with a `run_id`. The engine
worker (`engine.api.worker`, one process per host, kept alive by `engine.api.supervisor`) runs
queued runs one at a time and writes each result document back (decision A-14). With an
`idempotency_key`, the same key and body return the first `run_id`, and the same key with another
body is a `409` (I-89). A host whose engine worker serves another queue refuses the run with a
`503` before queueing it (I-106); a worker that cannot start leaves the run pending with the
reason in its status, never an error for a run that was queued (I-77).

The routes of 2026-10-07 (`POST /portfolio/price`, `POST /portfolio/market-risk`,
`POST /calibration/cam`, `POST /calibration/lgm`, `GET /jobs/...`) were removed on 2026-10-18,
there being no client (A-2's revision of 2026-10-07): each is an analytic of the run request.

Job store: the queue's SQLite file (`JAX_RISK_JOB_QUEUE`), which survives restarts and is
shared by every API process that opens it (docs/reference/http-api.md).
"""
import platform
import threading
import time
from typing import Optional

import jax
from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from starlette.concurrency import run_in_threadpool

from engine.api.job_queue import (
    CANCELLED, DONE, PENDING, RUN, TERMINAL, IdempotencyConflict, Job, JobQueue, default_queue_path, lock_owner,
    same_queue, worker_lock_path,
)
from engine.api.requests import RunRequestConfigSchema, RunRequestSchema
from engine.api.results import (
    EngineIdentitySchema, HealthSchema, RunStatusSchema, VersionSchema, WorkerHealthSchema, iso_time,
)
from engine.api.supervisor import EXTERNAL, SPAWN, WorkerSupervisor, host_conflict, worker_mode
from engine.run import engine_identity

router = APIRouter()

#: The longest `GET /runs/{run_id}?wait=N` holds a request open, and how often it looks.
MAX_WAIT_SECONDS = 30.0
_WAIT_POLL_SECONDS = 0.05

# The job queue, the host's worker lock, the worker mode and, in `spawn` mode, the supervisor of
# this process's worker; created on first use from the environment, or by `configure_jobs`.
_QUEUE: Optional[JobQueue] = None
_LOCK_PATH = None
_MODE: Optional[str] = None
_SUPERVISOR: Optional[WorkerSupervisor] = None
_CONFIGURING = threading.Lock()  # the first requests may arrive together, on the thread pool

#: What a refusal of the engine's own validation raises: a 400 naming the field.
_REFUSALS = (ValueError, KeyError, TypeError)


def configure_jobs(queue_path=None, mode: Optional[str] = None, lock_path=None) -> JobQueue:
    """Use the queue at `queue_path` (default: `JAX_RISK_JOB_QUEUE`, else the temp directory)
    with the worker `mode` (default: `JAX_RISK_WORKER`, else `spawn`) under the host's worker
    lock `lock_path` (default: `JAX_RISK_WORKER_LOCK`, else the temp directory), stopping a worker
    this process started for a previous queue. Called on first use; tests call it with a queue
    and a lock of their own."""
    global _QUEUE, _LOCK_PATH, _MODE, _SUPERVISOR
    shutdown_worker()
    _QUEUE = JobQueue(queue_path if queue_path is not None else default_queue_path())
    _LOCK_PATH = lock_path if lock_path is not None else worker_lock_path()
    _MODE = mode or worker_mode()
    _SUPERVISOR = WorkerSupervisor(_QUEUE, lock_path=_LOCK_PATH) if _MODE == SPAWN else None
    return _QUEUE


def job_queue() -> JobQueue:
    if _QUEUE is None:
        with _CONFIGURING:
            if _QUEUE is None:
                configure_jobs()
    return _QUEUE


def worker_supervisor() -> Optional[WorkerSupervisor]:
    """This process's worker supervisor; None in `external` mode."""
    job_queue()
    return _SUPERVISOR


def shutdown_worker() -> None:
    """Stop the worker this process started, if any (the app's shutdown, and tests)."""
    if _SUPERVISOR is not None:
        _SUPERVISOR.stop()


def _ensure_worker() -> None:
    """Start the worker if it is needed and not running. Never raises: a start that fails is the
    supervisor's to report on the pending run's status (I-77)."""
    supervisor = worker_supervisor()
    if supervisor is not None:
        supervisor.ensure_running()


def _worker_health() -> WorkerHealthSchema:
    """The engine worker as this process sees it (I-77, I-106)."""
    queue = job_queue()
    supervisor = worker_supervisor()
    if supervisor is not None:
        state = supervisor.status()
        return WorkerHealthSchema(mode=SPAWN, serving=state.serving, failed_starts=state.failed_starts,
                                  error=state.error)
    owner = lock_owner(_LOCK_PATH)
    return WorkerHealthSchema(mode=EXTERNAL, serving=owner is not None and same_queue(owner.get("queue", ""),
                                                                                       queue.path),
                              error=host_conflict(queue.path, _LOCK_PATH))


def _refused(exc: Exception, code: int = status.HTTP_400_BAD_REQUEST) -> HTTPException:
    message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
    return HTTPException(status_code=code, detail=str(message))


@router.get("/health", response_model=HealthSchema)
def health() -> HealthSchema:
    """The API process is up, and its engine worker's state: whether one serves this API's queue,
    and why not when that is known (a worker that cannot start, or the host's worker serving
    another queue). No engine work."""
    return HealthSchema(worker=_worker_health())


@router.get("/version", response_model=VersionSchema)
def version() -> VersionSchema:
    """The engine's identity (`engine.run.engine_identity`: its version, git revision, and the
    jax, jaxlib and ORE versions) and the JAX backend of this API process, which is not where runs
    execute: a served API's own JAX is on the CPU (`keep_jax_on_the_cpu`), and each run's result
    names its own engine, and its precision report the devices that ran it (I-12)."""
    try:
        backend = jax.default_backend()
    except Exception:
        backend = "unknown"
    return VersionSchema(engine=EngineIdentitySchema.from_dataclass(engine_identity()),
                         jax_backend=f"{backend} ({platform.system()})")


@router.get("/config/defaults", response_model=RunRequestConfigSchema)
def config_defaults() -> RunRequestConfigSchema:
    """The run request's whole `config` with every default (ORE's), to copy and edit. `models`,
    `simulation` and `market_risk` have none, being the inputs of the analytics that need them:
    their schemas are at `/docs`."""
    return RunRequestConfigSchema()


# --- runs ---------------------------------------------------------------------------------------

@router.post("/runs", status_code=status.HTTP_202_ACCEPTED)
async def submit_run(request: RunRequestSchema, http_request: Request) -> dict:
    """Run a request's analytics (`engine.run.run`): validate it synchronously (a refusal is a 400
    naming its field, and no run is created), then queue its body and return the `run_id`."""
    # The body as received; the worker parses it exactly as FastAPI parsed it here. FastAPI has
    # already read and cached it, so this does not read the stream again.
    body = await http_request.body()
    return {"run_id": await run_in_threadpool(_validate_and_queue, request, body)}


def _validate_and_queue(request: RunRequestSchema, body: bytes) -> str:
    try:
        request.check()
    except _REFUSALS as exc:
        raise _refused(exc) from exc
    queue = job_queue()
    conflict = host_conflict(queue.path, _LOCK_PATH)
    if conflict is not None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=conflict)
    key = request.idempotency_key
    try:
        run_id = queue.submit(body, RUN, idempotency_key=key, request_sha256=request.body_sha256() if key else None,
                              keep=request.keep)
    except IdempotencyConflict as exc:
        raise _refused(exc, status.HTTP_409_CONFLICT) from exc
    _ensure_worker()
    return run_id


@router.get("/runs/{run_id}", response_model=RunStatusSchema)
def get_run(run_id: str, wait: float = Query(0.0, ge=0.0, le=MAX_WAIT_SECONDS)):
    """The run's status: "pending" (queued; `error` says why no worker is running it, if one
    cannot), "running", "done" with the result, "failed" with the failure class and the worker's
    traceback, "interrupted" (the worker stopped during the run; submit it again) or "cancelled".
    With `wait`, the answer comes once the run is finished or `wait` seconds have passed. A done
    run's result document is sent as the worker stored it."""
    queue = job_queue()
    deadline = time.monotonic() + wait
    job = queue.summary(run_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown run_id: {run_id}")
    while job.status not in TERMINAL:
        _ensure_worker()  # a pending run needs a live worker
        if time.monotonic() >= deadline:
            error = _worker_health().error if job.status == PENDING else None
            return _status(job, error=error)
        time.sleep(_WAIT_POLL_SECONDS)
        job = queue.summary(run_id)
    if job.status != DONE:
        return _status(job, error=job.error)
    # The stored document is the result schema as JSON: spliced in, not parsed and serialized
    # again (a cube inline can run to megabytes).
    head = _status(job).model_dump_json(exclude={"result"})
    return Response(content=f'{head[:-1]},"result":{queue.get(run_id).result}}}', media_type="application/json")


def _status(job: Job, error: Optional[str] = None) -> RunStatusSchema:
    return RunStatusSchema(run_id=job.id, kind=job.kind, status=job.status, error=error,
                           failure_class=job.failure_class, cancel_requested=job.cancel_requested, keep=job.keep,
                           submitted=iso_time(job.submitted), started=iso_time(job.started),
                           finished=iso_time(job.finished))


@router.post("/runs/{run_id}/cancel", response_model=RunStatusSchema)
def cancel_run(run_id: str) -> RunStatusSchema:
    """Cancel the run (I-91): a pending run is `cancelled` at once; a running one stops at its
    next phase boundary (the calibration, the simulation, each trade's pricing, the Greeks, ...)
    and then reads `cancelled`, the engine worker going on to the next run with its compiled
    programs. A cancelled run is answered as it is; a finished one is a 409."""
    queue = job_queue()
    outcome = queue.cancel(run_id)
    if outcome is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown run_id: {run_id}")
    if outcome in TERMINAL and outcome != CANCELLED:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail=f"run {run_id} is {outcome}: there is nothing to cancel")
    job = queue.summary(run_id)
    return _status(job, error=job.error)


@router.get("/runs/{run_id}/artifacts/{name}/{chunk}")
def get_artifact_chunk(run_id: str, name: str, chunk: int) -> Response:
    """One chunk of a done run's array artifact, its raw bytes as the result's reference
    describes them (`engine.api.artifacts`): check its `sha256` before use."""
    data = job_queue().artifact(run_id, name, chunk)
    if data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"run {run_id} has no chunk {chunk} of an artifact {name!r}")
    return Response(content=data, media_type="application/octet-stream")
