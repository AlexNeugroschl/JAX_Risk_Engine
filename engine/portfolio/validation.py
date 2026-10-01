"""
Re-exports of the trade-field validators.

The trade-field validators live in `engine.instruments._validation` (audit A-5) so that
instrument modules do not import this package; they are re-exported here for existing callers.
Portfolio-level checks are the pipeline's (`engine.valuation.portfolio.validate_trades`: every
trade on the market's as-of date, its curves and volatilities present; `PortfolioRequest`:
unique trade ids).
"""
from engine.instruments._validation import _validate_common_fields  # noqa: F401  (re-export)
from engine.models.ore_builders import validate_tenor as _validate_tenor  # noqa: F401
