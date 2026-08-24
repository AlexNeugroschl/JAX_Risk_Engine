"""
`engine.portfolio` -- the engine's top-level entry point package: "a
portfolio of trades + market data + risk parameters" in, "prices + risk"
out.

Split into `request.py` (the actual `PortfolioRequest`/`PortfolioResult`/
`price_portfolio`/`validate_portfolio_against_simulation`/
`derive_maturity_pillars` implementation) and `validation.py` (leaf-level
field validators shared by every trade config's `__post_init__` -- kept
separate to avoid a circular import; see `validation.py`'s own docstring
for why). This `__init__.py` re-exports both modules' public surface so
every existing caller's `from engine.portfolio import ...` keeps working
unchanged now that `engine.portfolio` is a package rather than a single
module.

**Why `request`'s names are re-exported lazily (PEP 562 `__getattr__`)
below, rather than with a plain `from engine.portfolio.request import ...`
at module scope.** Every instrument module (`engine/instruments/swap.py`
etc.) imports `_validate_common_fields`/`_validate_tenor` from
`engine.portfolio.validation` at ITS OWN module scope. Importing any
submodule of a package -- including `engine.portfolio.validation` alone --
requires Python to first run this `__init__.py` to completion. If this file
eagerly imported `request` (which imports every instrument module, e.g.
`engine.instruments.swap`, to build `TradeConfig`), then whenever an
instrument module is the FIRST thing to import `engine.portfolio.validation`
(e.g. `demo.py` importing `engine.instruments.swap` before ever touching
`engine.portfolio` directly), the sequence would recurse: importing
`engine.instruments.swap` triggers this `__init__.py`, which eagerly imports
`request`, which imports `engine.instruments.swap` again -- except that
module is already mid-import (stuck at its own `validation` import) and
only partially initialized, so `SwapConfig` isn't defined on it yet ->
`ImportError`. Deferring `request`'s names behind `__getattr__` means
merely running this `__init__.py` (to satisfy the `validation` submodule
import) does NOT import `request`/the instrument modules at all; `request`
only imports on first actual access to one of the names below, by which
point `engine.instruments.swap` (or whichever module triggered this file)
has finished initializing.
"""
from engine.portfolio.validation import _validate_common_fields, _validate_tenor  # noqa: F401

__all__ = [
    "PortfolioRequest",
    "PortfolioResult",
    "PrecisionConfig",
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
