"""
`engine.run`: the top-level entry point (portfolio + market + run configuration in,
prices and risk out).

    request.py   PortfolioRequest, PortfolioResult and price_portfolio
    pipeline.py  the pipeline: calibration, simulation, pricing on every path, exposure, Greeks
    config.py    the run configuration (RunConfig and its components)
    trace.py     the pipeline's phase labels for profiler traces, and a written trace's summary

The request, the result and the configuration are re-exported here with the configuration types
of the other layers, so one import configures a run. No lower layer imports this package at
module scope (audit A-5; tests/test_import_layering.py), so the imports below are plain.
"""
from engine.market_simulation.config import CamConfig, HullWhiteConfig, LgmConfig  # noqa: F401
from engine.precision import Precision, StagePrecision  # noqa: F401
from engine.pricing.config import JamshidianEngineConfig, LgmSwaptionEngineConfig, PricingConfig  # noqa: F401
from engine.risk.greeks.bump import SensitivityConfig  # noqa: F401
from engine.run.config import GreeksConfig, RunConfig  # noqa: F401
from engine.run.request import (  # noqa: F401
    PortfolioRequest,
    PortfolioResult,
    TradeConfig,
    price_portfolio,
)

__all__ = [
    "CamConfig",
    "GreeksConfig",
    "HullWhiteConfig",
    "JamshidianEngineConfig",
    "LgmConfig",
    "LgmSwaptionEngineConfig",
    "PortfolioRequest",
    "PortfolioResult",
    "Precision",
    "PricingConfig",
    "RunConfig",
    "SensitivityConfig",
    "StagePrecision",
    "TradeConfig",
    "price_portfolio",
]
