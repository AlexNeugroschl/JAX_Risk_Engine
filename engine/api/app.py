"""
FastAPI app. Run with:

    .venv/Scripts/python.exe -m uvicorn engine.api.app:app --reload

See docs/reference/http-api.md.
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
    # The TraderX EOD boundary, under `/eod`. A separate router: it has a different
    # contract (a JSON-Schema-published result document) and shares no state.
    app.include_router(eod_router)
    return app


app = create_app()
