"""
HTTP route handlers -- FastAPI's transport layer over `engine.portfolio.
price_portfolio`. No new pricing/orchestration logic lives here; every
route is a thin translation between an HTTP request/response and a call
into `engine.portfolio`/`engine.calibration`.

**Async job pattern for `/portfolio/price`, not a blocking sync response.**
A 4-trade, 4096-scenario portfolio measured ~52 seconds wall time end-to-end
(JAX JIT compilation + Monte Carlo simulation dominate) -- see
docs/reference/http-api.md for the full measurement and reasoning. A
synchronous HTTP response held open that long is fragile (client/proxy
timeouts, no progress visibility, no retry-without-recompute), so
`POST /portfolio/price` validates synchronously (fast -- Phase 1's
validators do no JAX work) and returns `202 Accepted` + a `job_id`
immediately, dispatching `price_portfolio` to `engine.portfolio.worker_pool`
(a `ProcessPoolExecutor`-backed, per-precision-tier pool -- see that
module's own docstring); `GET /portfolio/price/{job_id}` polls the
resulting `Future` for the result.

**Job store: in-process `job_id -> Future` table, still single-process.**
This dispatcher process itself still keeps job state in a plain in-process
dict -- no Redis/DB in this phase (see `docs/reference/http-api.md`'s "Job
store" section for the two distinct reasons a shared store might eventually
be needed: HTTP-scaling to multiple uvicorn workers, still deferred, versus
process-isolated precision/device concurrency, now solved one layer down by
`engine.portfolio.worker_pool` rather than by this dict). What changed from
the previous architecture is what a job_id maps to and where the actual
`price_portfolio` call runs: previously, a FastAPI `BackgroundTasks` thread
mutated `_JOBS[job_id]` directly, in-process, serialized against every other
concurrent job by `_PRICING_LOCK`; now, `_JOBS[job_id]` holds a
`concurrent.futures.Future` returned by `worker_pool.submit_pricing_job`,
whose actual work runs in a separate OS process, genuinely concurrently with
other jobs (including other precision tiers) -- polling just checks
`future.done()`/`future.result()` instead of a dict a background thread
mutated directly.
"""
import platform
import subprocess
import traceback
import uuid
from concurrent.futures import Future
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
from fastapi import APIRouter, HTTPException, status

from engine.portfolio import price_portfolio, validate_portfolio_against_simulation
from engine.portfolio.worker_pool import submit_pricing_job
from engine.simulation.market_model import validate_joint_covariance
from engine.calibration.basket import build_coterminal_basket
from engine.calibration.lgm import calibrate_lgm_sigma
from engine.models.hull_white import ZeroCurve as _HwZeroCurve

from engine.api.schemas import (
    CalibrationRequestSchema, CalibrationResultSchema, HealthSchema, JobStatusSchema,
    PortfolioRequestSchema, PortfolioResultSchema, VersionSchema, _parse_ore_date,
)

router = APIRouter()

# In-process job store: job_id -> concurrent.futures.Future[PortfolioResult],
# returned by engine.portfolio.worker_pool.submit_pricing_job. See module
# docstring's "Job store" section.
_JOBS: Dict[str, "Future"] = {}


@router.get("/health", response_model=HealthSchema)
def health() -> HealthSchema:
    """Liveness only -- confirms the process is up, no engine work."""
    return HealthSchema()


@router.get("/version", response_model=VersionSchema)
def version() -> VersionSchema:
    """Engine package version, JAX backend (CPU/GPU), and git commit if
    available (best-effort -- None if this isn't a git checkout or `git`
    isn't on PATH)."""
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
def submit_portfolio_price(request: PortfolioRequestSchema) -> dict:
    """Validates synchronously (cheap -- no JAX work) by attempting the
    dataclass conversion + Phase 1 validators up front, THEN dispatches the
    actual (expensive) `price_portfolio` call to
    `engine.portfolio.worker_pool.submit_pricing_job`, which routes it to
    the worker pool matching `request.precision.simulation` and returns a
    `Future` immediately -- the real pricing work runs in a separate OS
    process, not a thread in this one. A validation failure here is
    returned as a `4xx` immediately, before a job_id is ever created -- a
    request that will never succeed shouldn't occupy a job slot."""
    try:
        dataclass_request = request.to_dataclass()
        validate_joint_covariance(dataclass_request.market.joint_covariance)
        validate_portfolio_against_simulation(dataclass_request.market, dataclass_request.trades)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    job_id = str(uuid.uuid4())
    _JOBS[job_id] = submit_pricing_job(dataclass_request)
    return {"job_id": job_id}


@router.get("/portfolio/price/{job_id}", response_model=JobStatusSchema)
def get_portfolio_price(job_id: str) -> JobStatusSchema:
    """Polls `future.done()`/`future.result()` instead of reading a dict a
    background thread mutated directly -- see module docstring. `"pending"`
    covers both "genuinely queued behind this tier's pool" and "actively
    running in a worker" (the worker process can't cheaply report its own
    sub-states back to this dispatcher without a mechanism this phase
    doesn't build -- see docs/reference/http-api.md); `"done"`/`"failed"`
    are reported once `future.done()` is true, matching the previous
    architecture's exception-handling/error-message behavior exactly (the
    worker-side exception, raised again by `future.result()`, is formatted
    the same way `_run_pricing_job` used to format it in-process)."""
    future = _JOBS.get(job_id)
    if future is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown job_id: {job_id}")

    if not future.done():
        return JobStatusSchema(status="pending", result=None, error=None)

    try:
        result = future.result()
    except Exception as exc:
        return JobStatusSchema(status="failed", result=None, error=f"{exc}\n{traceback.format_exc()}")

    return JobStatusSchema(status="done", result=PortfolioResultSchema.from_dataclass(result), error=None)


@router.post("/calibration/lgm", response_model=CalibrationResultSchema)
def calibrate(request: CalibrationRequestSchema) -> CalibrationResultSchema:
    """Standalone calibration endpoint -- wraps `build_coterminal_basket` +
    `calibrate_lgm_sigma` for a caller who wants a fitted `Sigma` back
    before submitting a full portfolio request. Synchronous (calibration is
    a cheap bisection-based bootstrap, not a Monte Carlo simulation -- no
    async job pattern needed here)."""
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
