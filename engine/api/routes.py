"""
HTTP routes over `engine.portfolio` and `engine.calibration`; translation only, no pricing
logic.

`POST /portfolio/price` takes the portfolio request (`engine.api.market_schemas`: today's
market, the trades, the run configuration with the model per currency, engines, Greeks and
precision), validates it synchronously (no JAX work), then writes the body as received to the
durable job queue (`engine.api.job_queue`) and returns `202` with a `job_id`; pricing a
portfolio with a simulation can take minutes, too long to hold a request open. The engine
worker (`engine.api.worker`, one process per host, kept alive by `engine.api.supervisor`)
prices queued jobs one at a time and writes each result document back.
`GET /portfolio/price/{job_id}` reads the job's row (roadmap 1.8, decision A-14).

`POST /v2/portfolio/price` takes the same request: the `/v2` is a historical name, not a
version (roadmap 4.1 retires it; compliance/decisions.md A-2). Until roadmap 1.3
`POST /portfolio/price` took the Hull-White model's own request shape; the Hull-White model is
now `"model": "HullWhite"` in the request's simulation, and the old shape is refused with a
422 naming its replacement.

Job store: the queue's SQLite file (`JAX_RISK_JOB_QUEUE`), which survives restarts and is
shared by every API process that opens it (docs/reference/http-api.md).
"""
import platform
import subprocess
import threading
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from fastapi import APIRouter, HTTPException, Request, Response, status
from starlette.concurrency import run_in_threadpool

from engine.api.job_queue import DONE, TERMINAL, JobQueue, default_queue_path
from engine.api.supervisor import SPAWN, WorkerSupervisor, worker_mode
from engine.portfolio.market_path import validate_request
from engine.calibration.basket import build_coterminal_basket
from engine.calibration.lgm import calibrate_lgm_sigma
from engine.models.hull_white import ZeroCurve as _HwZeroCurve

from engine.api.market_schemas import MarketPortfolioRequestSchema
from engine.api.schemas import (
    CalibrationRequestSchema, CalibrationResultSchema, HealthSchema, JobStatusSchema, VersionSchema,
    _parse_ore_date,
)

router = APIRouter()

# The job queue and, in `spawn` mode, the supervisor of this process's worker; created on first
# use from the environment, or by `configure_jobs`.
_QUEUE: Optional[JobQueue] = None
_SUPERVISOR: Optional[WorkerSupervisor] = None
_CONFIGURING = threading.Lock()  # the first requests may arrive together, on the thread pool


def configure_jobs(queue_path=None, mode: Optional[str] = None) -> JobQueue:
    """Use the queue at `queue_path` (default: `JAX_RISK_JOB_QUEUE`, else the temp directory)
    with the worker `mode` (default: `JAX_RISK_WORKER`, else `spawn`), stopping a worker this
    process started for a previous queue. Called on first use; tests call it with a queue of
    their own."""
    global _QUEUE, _SUPERVISOR
    shutdown_worker()
    _QUEUE = JobQueue(queue_path if queue_path is not None else default_queue_path())
    _SUPERVISOR = WorkerSupervisor(_QUEUE.path) if (mode or worker_mode()) == SPAWN else None
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
    supervisor = worker_supervisor()
    if supervisor is not None:
        supervisor.ensure_running()


@router.get("/health", response_model=HealthSchema)
def health() -> HealthSchema:
    """Liveness only; no engine work."""
    return HealthSchema()


@router.get("/version", response_model=VersionSchema)
def version() -> VersionSchema:
    """Engine version, JAX backend of this (API) process, and git commit if available. The
    backend is not where jobs run: each job's result names its own devices and backend in its
    `precision` report, built in the engine worker that ran it (roadmap 1.7, I-12)."""
    try:
        import importlib.metadata
        engine_version = importlib.metadata.version("jax-risk-engine")
    except Exception:
        engine_version = "unknown"

    try:
        backend = jax.default_backend()
    except Exception:
        backend = "unknown"

    git_commit = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            git_commit = result.stdout.strip()
    except Exception:
        pass

    return VersionSchema(engine_version=engine_version, jax_backend=f"{backend} ({platform.system()})", git_commit=git_commit)


@router.post("/portfolio/price", status_code=status.HTTP_202_ACCEPTED)
async def submit_portfolio_price(request: MarketPortfolioRequestSchema, http_request: Request) -> dict:
    """Validate the request synchronously (a failure is a 400, and no job is created),
    then queue its body and return the job's `job_id`."""
    return {"job_id": await _validate_and_submit(request, http_request)}


@router.post("/v2/portfolio/price", status_code=status.HTTP_202_ACCEPTED)
async def submit_market_portfolio_price(request: MarketPortfolioRequestSchema, http_request: Request) -> dict:
    """The same request at its historical name (see the module docstring)."""
    return {"job_id": await _validate_and_submit(request, http_request)}


async def _validate_and_submit(request: MarketPortfolioRequestSchema, http_request: Request) -> str:
    # The body as received; the worker parses it exactly as FastAPI parsed it here. FastAPI
    # has already read and cached it, so this does not read the stream again.
    body = await http_request.body()
    return await run_in_threadpool(_validate_and_queue, request, body)


def _validate_and_queue(request: MarketPortfolioRequestSchema, body: bytes) -> str:
    try:
        validate_request(request.to_dataclass())
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    job_id = job_queue().submit(body)
    _ensure_worker()
    return job_id


@router.get("/portfolio/price/{job_id}", response_model=JobStatusSchema)
def get_portfolio_price(job_id: str):
    """Job status: "pending" (queued), "running", "done" with the result, "failed" with the
    failure class and the worker's traceback, or "interrupted" (the worker stopped during the
    job; submit it again). A done job's result document is sent as the worker stored it."""
    queue = job_queue()
    current = queue.status(job_id)
    if current is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown job_id: {job_id}")
    if current not in TERMINAL:
        _ensure_worker()  # a pending job needs a live worker
        return JobStatusSchema(status=current)

    job = queue.get(job_id)
    if job.status == DONE:
        # The stored document is `PortfolioResultSchema`'s JSON: spliced in, not parsed and
        # serialized again (a cube can run to megabytes).
        return Response(content=_DONE_PREFIX + job.result + _DONE_SUFFIX, media_type="application/json")
    return JobStatusSchema(status=job.status, error=job.error, failure_class=job.failure_class)


# `JobStatusSchema(status="done", result=...)` as JSON, around the stored result document.
_DONE_PREFIX = '{"status":"done","result":'
_DONE_SUFFIX = ',"error":null,"failure_class":null}'


@router.post("/calibration/lgm", response_model=CalibrationResultSchema)
def calibrate(request: CalibrationRequestSchema) -> CalibrationResultSchema:
    """Build a co-terminal basket and calibrate a piecewise LGM `Sigma` to it
    (`build_coterminal_basket` + `calibrate_lgm_sigma`). Synchronous: calibration is a fast
    bootstrap."""
    try:
        eval_date = _parse_ore_date(request.evaluation_date)
        curve_dc = request.zero_curve.to_dataclass()
        curve_jax = _HwZeroCurve(
            pillar_times=jnp.asarray(curve_dc.times, dtype=jnp.float64),
            pillar_rates=jnp.asarray(curve_dc.rates, dtype=jnp.float64),
        )
        targets = build_coterminal_basket(
            exercise_times=request.exercise_times, final_maturity_time=request.final_maturity_time,
            notional=request.notional, payer=request.payer, market_vols=request.market_vols,
            zero_curve=curve_jax, evaluation_date=eval_date, index_tenor_months=request.index_tenor_months,
        )
        result = calibrate_lgm_sigma(targets, curve_jax, a=request.hw_a)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return CalibrationResultSchema(
        sigma_times=np.asarray(result.sigma.times).tolist(),
        sigma_values=np.asarray(result.sigma.values).tolist(),
        market_prices=np.asarray(result.market_prices).tolist(),
        model_prices=np.asarray(result.model_prices).tolist(),
        rmse=result.rmse,
    )
