"""
FastAPI app factory. Run directly with uvicorn:

    .venv/Scripts/python.exe -m uvicorn engine.api.app:app --reload

See docs/reference/http-api.md for the full endpoint reference and the
sync-vs-async job pattern's reasoning (measured ~52s wall time for a
4-trade, 4096-scenario portfolio -- see that doc for the actual numbers).
"""
from fastapi import FastAPI

from engine.api.eod_routes import router as eod_router
from engine.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="JAX Risk Engine API",
        description=(
            "HTTP API over engine.portfolio.price_portfolio -- simulate, "
            "validate, price, and profile exposure for a portfolio of "
            "interest-rate swaps, swaptions and Treasuries. See "
            "docs/reference/http-api.md for the full reference."
        ),
        version="0.1.0",
    )
    app.include_router(router)
    # W1.6.4: the TraderX EOD boundary, under its own `/eod` prefix. A
    # separate router rather than more handlers on the portfolio one --
    # the two speak different contracts (a JSON-Schema-published result
    # document versus Pydantic-wrapped engine dataclasses) and share no
    # state, so keeping them apart keeps either free to change.
    app.include_router(eod_router)
    return app


app = create_app()
