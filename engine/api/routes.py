"""
HTTP routes over `engine.portfolio` and `engine.calibration`; translation only, no pricing
logic.

`POST /portfolio/price` validates synchronously (no JAX work), then submits the job to
`engine.portfolio.worker_pool` and returns `202` with a `job_id`; pricing a 4-trade,
4096-scenario portfolio takes about a minute, too long to hold a request open.
`GET /portfolio/price/{job_id}` polls the job's `Future`.

`POST /v2/portfolio/price` is the same for the market path, ORE's pipeline and the default
model (`engine.api.market_schemas`, `engine.portfolio.market_path`); its jobs are polled at
the same `GET` route. `POST /portfolio/price` takes the Hull-White model's request. The two
routes are models, not versions: the `/v2` is a historical name. They are to become one
route taking one configurable request (compliance/decisions.md A-2).

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

from engine.portfolio.request import validate_hull_white_request
from engine.portfolio.market_path import validate_market_request
from engine.portfolio.worker_pool import submit_pricing_job
from engine.calibration.basket import build_coterminal_basket
from engine.calibration.lgm import calibrate_lgm_sigma
from engine.models.hull_white import ZeroCurve as _HwZeroCurve

from engine.api.market_schemas import MarketPortfolioRequestSchema
from engine.api.schemas import (
    CalibrationRequestSchema, CalibrationResultSchema, HealthSchema, JobStatusSchema,
    PortfolioRequestSchema, PortfolioResultSchema, VersionSchema, _parse_ore_date,
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
    """Engine version, JAX backend of this (dispatcher) process, and git commit if
    available. The backend is not the workers' device (I-12)."""
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
    """Validate the request synchronously (a failure is a 400, and no job is created),
    then submit it to the worker pool for its precision tier and return its `job_id`."""
    try:
        dataclass_request = request.to_dataclass()
        validate_hull_white_request(dataclass_request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return {"job_id": _submit(dataclass_request)}


@router.post("/v2/portfolio/price", status_code=status.HTTP_202_ACCEPTED)
def submit_market_portfolio_price(request: MarketPortfolioRequestSchema) -> dict:
    """The market path's request: validate synchronously (a failure is a 400), then
    submit as `POST /portfolio/price` does."""
    try:
        dataclass_request = request.to_dataclass()
        validate_market_request(dataclass_request)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return {"job_id": _submit(dataclass_request)}


def _submit(dataclass_request) -> str:
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
