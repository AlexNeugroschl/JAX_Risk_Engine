"""
HTTP routes over `engine.portfolio` and `engine.calibration`; translation only, no pricing
logic.

`POST /portfolio/price` takes the portfolio request (`engine.api.market_schemas`: today's
market, the trades, the run configuration with the model per currency, engines, Greeks and
precision), validates it synchronously (no JAX work), then submits the job to
`engine.portfolio.worker_pool` and returns `202` with a `job_id`; pricing a portfolio with a
simulation can take minutes, too long to hold a request open. `GET /portfolio/price/{job_id}`
polls the job's `Future`.

`POST /v2/portfolio/price` takes the same request: the `/v2` is a historical name, not a
version (roadmap 4.1 retires it; compliance/decisions.md A-2). Until roadmap 1.3
`POST /portfolio/price` took the Hull-White model's own request shape; the Hull-White model is
now `"model": "HullWhite"` in the request's simulation, and the old shape is refused with a
422 naming its replacement.

Job store: an in-process `job_id -> Future` dict, lost on restart and not shared between
uvicorn workers (I-08; see docs/reference/http-api.md).
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

from engine.portfolio.market_path import validate_request
from engine.portfolio.worker_pool import submit_pricing_job
from engine.calibration.basket import build_coterminal_basket
from engine.calibration.lgm import calibrate_lgm_sigma
from engine.models.hull_white import ZeroCurve as _HwZeroCurve

from engine.api.market_schemas import MarketPortfolioRequestSchema
from engine.api.schemas import (
    CalibrationRequestSchema, CalibrationResultSchema, HealthSchema, JobStatusSchema,
    PortfolioResultSchema, VersionSchema, _parse_ore_date,
)

router = APIRouter()

# In-process job store: job_id -> Future[PortfolioResult] (see the module docstring).
_JOBS: Dict[str, "Future"] = {}


@router.get("/health", response_model=HealthSchema)
def health() -> HealthSchema:
    """Liveness only; no engine work."""
    return HealthSchema()


@router.get("/version", response_model=VersionSchema)
def version() -> VersionSchema:
    """Engine version, JAX backend of this (API) process, and git commit if available. The
    backend is not where jobs run: each job's result names its own devices and backend in its
    `precision` report, built in the worker that ran it (roadmap 1.7, I-12)."""
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
def submit_portfolio_price(request: MarketPortfolioRequestSchema) -> dict:
    """Validate the request synchronously (a failure is a 400, and no job is created),
    then submit it to the worker pool and return its `job_id`."""
    return {"job_id": _validate_and_submit(request)}


@router.post("/v2/portfolio/price", status_code=status.HTTP_202_ACCEPTED)
def submit_market_portfolio_price(request: MarketPortfolioRequestSchema) -> dict:
    """The same request at its historical name (see the module docstring)."""
    return {"job_id": _validate_and_submit(request)}


def _validate_and_submit(request: MarketPortfolioRequestSchema) -> str:
    try:
        dataclass_request = request.to_dataclass()
        validate_request(dataclass_request)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    job_id = str(uuid.uuid4())
    _JOBS[job_id] = submit_pricing_job(dataclass_request)
    return job_id


@router.get("/portfolio/price/{job_id}", response_model=JobStatusSchema)
def get_portfolio_price(job_id: str) -> JobStatusSchema:
    """Job status: "pending" (queued or running; not distinguished), "done" with the
    result, or "failed" with the worker's exception and traceback."""
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
