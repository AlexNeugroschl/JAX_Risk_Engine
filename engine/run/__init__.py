"""
`engine.run`: the top-level entry point (portfolio + market + run configuration in,
prices and risk out).

    run_request.py  the run request (RunRequest, RunRequestConfig and its sections, RunResult)
                    and `run`, the one way in for every analytic (I-87)
    request.py      PortfolioRequest, PortfolioResult and price_portfolio, which `run` calls
    pipeline.py     the pipeline: calibration, simulation, pricing on every path, exposure, Greeks
    config.py       price_portfolio's run configuration (RunConfig and its components)
    identity.py     the engine's identity every run result carries (revision, library versions)
    trace.py        the phases of a run: labels for profiler traces, the boundaries a run can
                    stop at, and a written trace's summary

The requests, the results and the configurations are re-exported here with the configuration
types of the other layers, so one import configures a run. No lower layer imports this package at
module scope (audit A-5; tests/test_import_layering.py), so the imports below are plain.
"""
from engine.market_simulation.config import CamConfig, HullWhiteConfig, LgmConfig  # noqa: F401
from engine.precision import Precision, StagePrecision  # noqa: F401
from engine.pricing.config import JamshidianEngineConfig, LgmSwaptionEngineConfig, PricingConfig  # noqa: F401
from engine.risk.greeks.bump import SensitivityConfig  # noqa: F401
from engine.run.config import GreeksConfig, RunConfig  # noqa: F401
from engine.run.identity import EngineIdentity, engine_identity  # noqa: F401
from engine.run.request import (  # noqa: F401
    PortfolioRequest,
    PortfolioResult,
    TradeConfig,
    price_portfolio,
)
from engine.run.run_request import (  # noqa: F401
    ANALYTICS,
    ExposureConfig,
    MarketRiskConfig,
    ModelsConfig,
    RunRequest,
    RunRequestConfig,
    RunResult,
    SimulationParameters,
    run,
    validate_run,
)

__all__ = [
    "ANALYTICS",
    "CamConfig",
    "EngineIdentity",
    "ExposureConfig",
    "GreeksConfig",
    "HullWhiteConfig",
    "JamshidianEngineConfig",
    "LgmConfig",
    "LgmSwaptionEngineConfig",
    "MarketRiskConfig",
    "ModelsConfig",
    "PortfolioRequest",
    "PortfolioResult",
    "Precision",
    "PricingConfig",
    "RunConfig",
    "RunRequest",
    "RunRequestConfig",
    "RunResult",
    "SensitivityConfig",
    "SimulationParameters",
    "StagePrecision",
    "TradeConfig",
    "engine_identity",
    "price_portfolio",
    "run",
    "validate_run",
]
