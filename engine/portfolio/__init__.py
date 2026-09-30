"""
`engine.portfolio`: the top-level entry point (portfolio + market + run configuration in,
prices and risk out).

`request.py` holds `PortfolioRequest`, `PortfolioResult`, `price_portfolio` and friends;
`config.py` the run configuration (`RunConfig` and its components); `validation.py`
portfolio-level checks. All are re-exported here. Instrument modules do not import this
package (audit A-5), so the imports below are plain.
"""
from engine.portfolio.validation import _validate_common_fields, _validate_tenor  # noqa: F401
from engine.portfolio.config import (  # noqa: F401
    HULL_WHITE_CONFIG,
    GreeksConfig,
    PrecisionConfig,
    PricingPrecisionOverride,
    RiskPrecisionOverride,
    RunConfig,
)
from engine.portfolio.request import (  # noqa: F401
    PortfolioRequest,
    PortfolioResult,
    TradeConfig,
    derive_maturity_pillars,
    price_portfolio,
    validate_portfolio_against_simulation,
)
from engine.risk.sensitivities import SensitivityConfig  # noqa: F401
from engine.simulation.config import CamConfig, LgmConfig  # noqa: F401
from engine.valuation.config import LgmSwaptionEngineConfig, PricingConfig  # noqa: F401

__all__ = [
    "CamConfig",
    "GreeksConfig",
    "HULL_WHITE_CONFIG",
    "LgmConfig",
    "LgmSwaptionEngineConfig",
    "PortfolioRequest",
    "PortfolioResult",
    "PrecisionConfig",
    "PricingConfig",
    "PricingPrecisionOverride",
    "RiskPrecisionOverride",
    "RunConfig",
    "SensitivityConfig",
    "TradeConfig",
    "price_portfolio",
    "validate_portfolio_against_simulation",
    "derive_maturity_pillars",
]
