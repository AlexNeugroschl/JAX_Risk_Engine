"""
`engine.portfolio`: the top-level entry point (portfolio + market + run configuration in,
prices and risk out).

`request.py` holds `PortfolioRequest`, `PortfolioResult` and `price_portfolio`;
`market_path.py` the pipeline; `config.py` the run configuration (`RunConfig` and its
components). All are re-exported here with the configuration types of the other layers, so one
import configures a run. Instrument modules do not import this package (audit A-5), so the
imports below are plain.
"""
from engine.portfolio.validation import _validate_common_fields, _validate_tenor  # noqa: F401
from engine.portfolio.config import GreeksConfig, RunConfig  # noqa: F401
from engine.portfolio.request import (  # noqa: F401
    PortfolioRequest,
    PortfolioResult,
    TradeConfig,
    price_portfolio,
)
from engine.precision import Precision, StagePrecision  # noqa: F401
from engine.risk.sensitivities import SensitivityConfig  # noqa: F401
from engine.simulation.config import CamConfig, HullWhiteConfig, LgmConfig  # noqa: F401
from engine.valuation.config import JamshidianEngineConfig, LgmSwaptionEngineConfig, PricingConfig  # noqa: F401

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
