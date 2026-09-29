"""
`engine.portfolio`: the top-level entry point (portfolio + market + risk parameters in,
prices and risk out).

`request.py` holds `PortfolioRequest`, `PortfolioResult`, `price_portfolio` and friends;
`validation.py` holds portfolio-level checks. Both are re-exported here. Instrument modules
do not import this package (audit A-5), so the imports below are plain.
"""
from engine.portfolio.validation import _validate_common_fields, _validate_tenor  # noqa: F401
from engine.portfolio.request import (  # noqa: F401
    PortfolioRequest,
    PortfolioResult,
    PrecisionConfig,
    PricingPrecisionOverride,
    RiskPrecisionOverride,
    TradeConfig,
    derive_maturity_pillars,
    price_portfolio,
    validate_portfolio_against_simulation,
)

__all__ = [
    "PortfolioRequest",
    "PortfolioResult",
    "PrecisionConfig",
    "PricingPrecisionOverride",
    "RiskPrecisionOverride",
    "TradeConfig",
    "price_portfolio",
    "validate_portfolio_against_simulation",
    "derive_maturity_pillars",
]
