"""
FastAPI app. Run with:

    .venv/Scripts/python.exe -m uvicorn engine.api.app:app --reload

The app starts its engine worker on the first portfolio job and stops it on shutdown
(`engine.api.supervisor`; `JAX_RISK_WORKER=external` leaves the worker to another supervisor).
The server's own JAX work runs on the CPU (`keep_jax_on_the_cpu`), so the accelerators belong
to the worker. See docs/reference/http-api.md.
"""
import warnings
from contextlib import asynccontextmanager

import jax
from fastapi import FastAPI

from engine.api.eod_routes import router as eod_router
from engine.api.routes import router, shutdown_worker


def keep_jax_on_the_cpu() -> None:
    """Run this process's JAX on the CPU (roadmap 2.2). The server prices nothing: its JAX work
    is `/version` and the synchronous `/calibration/cam` and `/calibration/lgm`, small float64
    bootstraps (decision A-10). A GPU client would
    take device memory beside the engine worker's, which prices every job (on a TPU host a
    second process cannot open the chips at all). The worker is a fresh interpreter started
    from the environment, not from this setting, so it still sees every device.

    JAX chooses its platforms when the process first uses a device, and keeps them: a process
    that already has (one embedding the app after its own JAX work) is left as it is, with a
    warning."""
    from jax._src import xla_bridge

    if xla_bridge.backends_are_initialized():
        if jax.default_backend() != "cpu":
            warnings.warn(f"the API process already uses JAX's {jax.default_backend()!r} backend, so it "
                          "holds device memory beside the engine worker's", RuntimeWarning, stacklevel=2)
        return
    jax.config.update("jax_platforms", "cpu")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    keep_jax_on_the_cpu()
    yield
    shutdown_worker()


def create_app() -> FastAPI:
    app = FastAPI(
        title="JAX Risk Engine API",
        description=(
            "HTTP API over the engine: price a portfolio and profile its exposure "
            "(engine.portfolio.price_portfolio), its market-risk VaR and ES "
            "(engine.market_risk.run_market_risk), and calibrate the cross-asset model "
            "(engine.calibration.cam), for interest-rate swaps, swaptions and Treasuries. "
            "See docs/reference/http-api.md for the full reference."
        ),
        version="0.1.0",
        lifespan=_lifespan,
    )
    app.include_router(router)
    # The TraderX EOD boundary, under `/eod`. A separate router: it has a different
    # contract (a JSON-Schema-published result document) and shares no state.
    app.include_router(eod_router)
    return app


app = create_app()
