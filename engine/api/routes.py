"""
HTTP routes over `engine.portfolio`, `engine.market_risk` and `engine.calibration`; translation
only, no pricing logic (docs/reference/http-api.md; decision A-2, roadmap 3.1).

Jobs, for work that can take minutes (a simulation, a revaluation under thousands of scenarios),
too long to hold a request open:

    POST /portfolio/price          the portfolio request (`MarketPortfolioRequestSchema`)
    POST /portfolio/market-risk    the market-risk request (`MarketRiskRequestSchema`)
    GET  /jobs/{job_id}            the job's status, and its result once done
    GET  /jobs/{job_id}/artifacts/{name}/{chunk}
                                   one chunk of an array the request asked for by reference

Each submission is validated synchronously (no JAX work; a refusal is a 400 and no job is
created), then its body is written as received to the durable job queue
(`engine.api.job_queue`) with its kind, and the route returns `202` with a `job_id`. The engine
worker (`engine.api.worker`, one process per host, kept alive by `engine.api.supervisor`) runs
queued jobs one at a time and writes each result document back (roadmap 1.8, decision A-14).

Synchronous, for small bootstraps that answer in about a second once compiled:

    POST /calibration/cam          the cross-asset model's calibration per currency
    POST /calibration/lgm          a Hagan bootstrap of an LGM to a caller-given co-terminal basket

Both run in the API process, whose JAX is on the CPU (`engine.api.app.keep_jax_on_the_cpu`):
calibration is float64 by decision (A-10), and a few helpers' bootstraps gain nothing from an
accelerator.

Roadmap 3.1 retired `POST /v2/portfolio/price` (the same request at a name that looked like a
version) and `GET /portfolio/price/{job_id}` (now `GET /jobs/{job_id}`, for every kind).

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

from engine.api.job_queue import DONE, MARKET_RISK, PORTFOLIO, TERMINAL, JobQueue, default_queue_path
from engine.api.supervisor import SPAWN, WorkerSupervisor, worker_mode
from engine.calibration.basket import build_coterminal_basket
from engine.calibration.cam import calibrate_cam
from engine.calibration.lgm import calibrate_lgm_sigma
from engine.models.hull_white import ZeroCurve as _HwZeroCurve

from engine.api.market_schemas import CamCalibrationRequestSchema, MarketPortfolioRequestSchema, MarketRiskRequestSchema
from engine.api.schemas import (
    CalibrationRequestSchema, CalibrationResultSchema, CamCalibrationResultSchema, CurrencyCalibrationSchema,
    HealthSchema, JobStatusSchema, VersionSchema, _parse_ore_date,
)

router = APIRouter()

# The job queue and, in `spawn` mode, the supervisor of this process's worker; created on first
# use from the environment, or by `configure_jobs`.
_QUEUE: Optional[JobQueue] = None
_SUPERVISOR: Optional[WorkerSupervisor] = None
_CONFIGURING = threading.Lock()  # the first requests may arrive together, on the thread pool

#: What a refusal of the engine's own validation raises: a 400 naming the field.
_REFUSALS = (ValueError, KeyError, TypeError)


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


def _refused(exc: Exception) -> HTTPException:
    message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(message))


@router.get("/health", response_model=HealthSchema)
def health() -> HealthSchema:
    """Liveness only; no engine work."""
    return HealthSchema()


@router.get("/version", response_model=VersionSchema)
def version() -> VersionSchema:
    """Engine version, JAX backend of this (API) process, and git commit if available. The
    backend is not where jobs run: a served API's own JAX is on the CPU (`keep_jax_on_the_cpu`),
    and each job's result names its own devices and backend in its `precision` report, built in
    the engine worker that ran it (roadmap 1.7 and 2.2, I-12)."""
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


# --- jobs ---------------------------------------------------------------------------------------

@router.post("/portfolio/price", status_code=status.HTTP_202_ACCEPTED)
async def submit_portfolio_price(request: MarketPortfolioRequestSchema, http_request: Request) -> dict:
    """Price a portfolio (`engine.portfolio.price_portfolio`): validate the request synchronously
    (a refusal is a 400, and no job is created), then queue its body and return the `job_id`."""
    return {"job_id": await _validate_and_submit(request, http_request, PORTFOLIO)}


@router.post("/portfolio/market-risk", status_code=status.HTTP_202_ACCEPTED)
async def submit_market_risk(request: MarketRiskRequestSchema, http_request: Request) -> dict:
    """VaR and Expected Shortfall of a portfolio by full revaluation under shock scenarios
    (`engine.market_risk.run_market_risk`): validated, then queued, as `POST /portfolio/price`."""
    return {"job_id": await _validate_and_submit(request, http_request, MARKET_RISK)}


async def _validate_and_submit(request, http_request: Request, kind: str) -> str:
    # The body as received; the worker parses it exactly as FastAPI parsed it here. FastAPI
    # has already read and cached it, so this does not read the stream again.
    body = await http_request.body()
    return await run_in_threadpool(_validate_and_queue, request, body, kind)


def _validate_and_queue(request, body: bytes, kind: str) -> str:
    try:
        request.check()
    except _REFUSALS as exc:
        raise _refused(exc) from exc
    job_id = job_queue().submit(body, kind)
    _ensure_worker()
    return job_id


@router.get("/jobs/{job_id}", response_model=JobStatusSchema)
def get_job(job_id: str):
    """Job status: "pending" (queued), "running", "done" with the result, "failed" with the
    failure class and the worker's traceback, or "interrupted" (the worker stopped during the
    job; submit it again). A done job's result document is sent as the worker stored it."""
    queue = job_queue()
    state = queue.state(job_id)
    if state is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown job_id: {job_id}")
    if state[1] not in TERMINAL:
        _ensure_worker()  # a pending job needs a live worker
        return JobStatusSchema(kind=state[0], status=state[1])
    job = queue.get(job_id)
    if job.status == DONE:
        # The stored document is the kind's result schema as JSON: spliced in, not parsed and
        # serialized again (a cube can run to megabytes).
        head = f'{{"kind":"{job.kind}","status":"done","result":'
        return Response(content=head + job.result + _DONE_SUFFIX, media_type="application/json")
    return JobStatusSchema(kind=job.kind, status=job.status, error=job.error, failure_class=job.failure_class)


# `JobStatusSchema(..., status="done", result=...)` as JSON, after the stored result document.
_DONE_SUFFIX = ',"error":null,"failure_class":null}'


@router.get("/jobs/{job_id}/artifacts/{name}/{chunk}")
def get_artifact_chunk(job_id: str, name: str, chunk: int) -> Response:
    """One chunk of a done job's array artifact, its raw bytes as the result's reference
    describes them (`engine.api.artifacts`): check its `sha256` before use."""
    data = job_queue().artifact(job_id, name, chunk)
    if data is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"job {job_id} has no chunk {chunk} of an artifact {name!r}")
    return Response(content=data, media_type="application/octet-stream")


# --- calibration --------------------------------------------------------------------------------

@router.post("/calibration/cam", response_model=CamCalibrationResultSchema)
def calibrate_cross_asset_model(request: CamCalibrationRequestSchema) -> CamCalibrationResultSchema:
    """Each currency's model bootstrapped to its calibration basket on today's market, as a
    portfolio run calibrates the cross-asset model before simulating
    (`engine.calibration.cam.calibrate_cam`). Synchronous."""
    try:
        market, models = request.to_dataclass()
        calibrations = calibrate_cam(market, models)
    except _REFUSALS as exc:
        raise _refused(exc) from exc
    return CamCalibrationResultSchema(currencies={
        currency: CurrencyCalibrationSchema.from_dataclass(calibration, models[currency])
        for currency, calibration in calibrations.items()})


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
        result = calibrate_lgm_sigma(targets, curve_jax, a=request.hw_a, solver=request.solver)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return CalibrationResultSchema(
        sigma_times=np.asarray(result.sigma.times).tolist(),
        sigma_values=np.asarray(result.sigma.values).tolist(),
        market_prices=np.asarray(result.market_prices).tolist(),
        model_prices=np.asarray(result.model_prices).tolist(),
        rmse=result.rmse,
    )
