"""
`engine.portfolio`: the top-level entry point (portfolio + market + risk parameters in,
prices and risk out).

`request.py` holds `PortfolioRequest`, `PortfolioResult`, `price_portfolio` and friends;
`validation.py` holds the field validators every trade config uses. Both are re-exported
here.

`request`'s names are loaded lazily (PEP 562 `__getattr__`). Instrument modules import
`engine.portfolio.validation`, which runs this file first; importing `request` here eagerly
would re-import the half-initialized instrument module and fail.
"""
from engine.portfolio.validation import _validate_common_fields, _validate_tenor  # noqa: F401

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

_REQUEST_EXPORTS = frozenset(__all__)


def __getattr__(name: str):
    if name in _REQUEST_EXPORTS:
        from engine.portfolio import request as _request
        return getattr(_request, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
