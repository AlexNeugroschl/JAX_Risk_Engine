"""
FastAPI app. Run with:

    .venv/Scripts/python.exe -m uvicorn engine.api.app:app --reload

The app starts its engine worker on the first run and stops it on shutdown
(`engine.api.supervisor`; `JAX_RISK_WORKER=external` leaves the worker to another supervisor).
The server's own JAX work runs on the CPU (`keep_jax_on_the_cpu`), so the accelerators belong
to the worker. See docs/reference/http-api.md.
"""
import warnings
from contextlib import asynccontextmanager

import jax
from fastapi import FastAPI

from engine.api.routes import router, shutdown_worker
from engine.api.traderx_routes import router as traderx_router
from engine.run import engine_identity


def keep_jax_on_the_cpu() -> None:
    """Run this process's JAX on the CPU. The server prices nothing: its JAX work is building a
    run request's few arrays to check it (a piecewise volatility) and `/version`. A GPU client would
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
            "HTTP API over the engine: one run request (POST /runs) for every analytic, as ORE runs a "
            "portfolio with a list of analytics: today's values, exposure, sensitivities, market-risk "
            "VaR and ES, and the cross-asset model's calibration (engine.run.run), for interest-rate "
            "swaps, swaptions and Treasuries. See docs/reference/http-api.md for the full reference."
        ),
        version=engine_identity().version,
        lifespan=_lifespan,
    )
    app.include_router(router)
    # The TraderX path, under `/eod`. A separate router: it has a different
    # contract (a JSON-Schema-published result document) and shares no state.
    app.include_router(traderx_router)
    return app


app = create_app()
