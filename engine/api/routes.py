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
immediately, running `price_portfolio` in a FastAPI `BackgroundTasks` task;
`GET /portfolio/price/{job_id}` polls for the result.

**Job store: in-process dict.** Matches this system's low-volume,
single-consumer scope (see docs/planning/roadmap-and-history.md's roadmap
entry and docs/reference/http-api.md). A multi-process deployment (more than
one uvicorn worker) would need a shared store (Redis, a DB table) instead --
out of scope for this phase; each worker process would otherwise have its
own, mutually invisible job dict.
"""
import platform
import subprocess
import traceback
import uuid
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from engine.portfolio import price_portfolio, validate_portfolio_against_simulation
from engine.simulation.market_model import validate_joint_covariance
from engine.calibration.basket import build_coterminal_basket
from engine.calibration.lgm import calibrate_lgm_sigma
from engine.models.hull_white import ZeroCurve as _HwZeroCurve

from engine.api.schemas import (
    CalibrationRequestSchema, CalibrationResultSchema, HealthSchema, JobStatusSchema,
    PortfolioRequestSchema, PortfolioResultSchema, VersionSchema, _parse_ore_date,
)

router = APIRouter()

# In-process job store: job_id -> {"status": ..., "result": PortfolioResultSchema | None, "error": str | None}.
# See module docstring's "Job store" section.
_JOBS: Dict[str, dict] = {}


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


def _run_pricing_job(job_id: str, request_schema: PortfolioRequestSchema) -> None:
    """Runs in FastAPI's BackgroundTasks executor (a thread) after the
    202 response has already been sent -- see module docstring."""
    try:
        _JOBS[job_id]["status"] = "running"
        request = request_schema.to_dataclass()
        result = price_portfolio(request)
        _JOBS[job_id]["status"] = "done"
        _JOBS[job_id]["result"] = PortfolioResultSchema.from_dataclass(result)
    except Exception as exc:
        _JOBS[job_id]["status"] = "failed"
        _JOBS[job_id]["error"] = f"{exc}\n{traceback.format_exc()}"


@router.post("/portfolio/price", status_code=status.HTTP_202_ACCEPTED)
def submit_portfolio_price(request: PortfolioRequestSchema, background_tasks: BackgroundTasks) -> dict:
    """Validates synchronously (cheap -- no JAX work) by attempting the
    dataclass conversion + Phase 1 validators up front, THEN schedules the
    actual (expensive) `price_portfolio` call as a background task. A
    validation failure here is returned as a `4xx` immediately, before a
    job_id is ever created -- a request that will never succeed shouldn't
    occupy a job slot."""
    try:
        dataclass_request = request.to_dataclass()
        validate_joint_covariance(dataclass_request.market.joint_covariance)
        validate_portfolio_against_simulation(dataclass_request.market, dataclass_request.trades)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {"status": "pending", "result": None, "error": None}
    background_tasks.add_task(_run_pricing_job, job_id, request)
    return {"job_id": job_id}


@router.get("/portfolio/price/{job_id}", response_model=JobStatusSchema)
def get_portfolio_price(job_id: str) -> JobStatusSchema:
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown job_id: {job_id}")
    return JobStatusSchema(status=job["status"], result=job["result"], error=job["error"])


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
