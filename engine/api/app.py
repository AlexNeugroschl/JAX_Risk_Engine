"""
FastAPI app factory. Run directly with uvicorn:

    venv/Scripts/python.exe -m uvicorn engine.api.app:app --reload

See docs/reference/http-api.md for the full endpoint reference and the
sync-vs-async job pattern's reasoning (measured ~52s wall time for a
4-trade, 4096-scenario portfolio -- see that doc for the actual numbers).
"""
from fastapi import FastAPI

from engine.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="JAX Risk Engine API",
        description=(
            "HTTP API over engine.portfolio.price_portfolio -- simulate, "
            "validate, price, and aggregate risk for a portfolio of "
            "interest-rate swaps and swaptions. See "
            "docs/reference/http-api.md for the full reference."
        ),
        version="0.1.0",
    )
    app.include_router(router)
    return app


app = create_app()
