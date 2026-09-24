"""
Leaf-level field validators shared by every trade config's `__post_init__`
(`SwapConfig`, `SwaptionConfig`, `BermudanSwaptionConfig`,
`AmericanSwaptionConfig`).

**Why this is a separate module from `engine/portfolio/request.py`, despite
the plan calling for `engine.portfolio._validate_common_fields`.** Each
instrument module (`engine/instruments/swap.py`,
`european_swaption.py`, etc.) needs this helper at `__post_init__` time,
so it must import it. `engine/portfolio/request.py` itself needs to import
those SAME instrument modules' config classes (for `PortfolioRequest`'s
trade union and `derive_maturity_pillars`) -- so `engine/portfolio/
request.py` importing the instrument modules, and the instrument modules
importing `_validate_common_fields` back out of `engine/portfolio/
request.py`, is a genuine circular import, not just an ordering
inconvenience. This module breaks the cycle: it has zero dependency on any
instrument config, and `engine/portfolio/request.py` re-exports
`_validate_common_fields`/`_validate_tenor` from here (`from
engine.portfolio.validation import ...`) so
`engine.portfolio._validate_common_fields` still resolves exactly as the
plan names it -- this split is an implementation detail invisible to any
caller of `engine.portfolio`.

Deliberately scoped to reject malformed input, not impose business-rule
policy limits (e.g. no "no rate above 20%" check) -- notional/fixed_rate
sign and magnitude are otherwise unconstrained (zero and negative notional
are explicitly supported, see tests/test_swap.py::TestZeroNotional and
similar classes in the other instrument test files); only non-finite
(NaN/Inf) values are rejected here.
"""
import math

import numpy as np
import ORE


def _validate_common_fields(notional: float, fixed_rate: float, evaluation_date: ORE.Date) -> None:
    """Finite notional, finite fixed_rate -- called from every trade
    config's `__post_init__`. Zero and negative notional/fixed_rate are
    valid (see module docstring); only non-finite values are rejected."""
    if not math.isfinite(notional):
        raise ValueError(f"notional must be finite; got {notional}")
    if not math.isfinite(fixed_rate):
        raise ValueError(f"fixed_rate must be finite; got {fixed_rate}")


def _validate_hw_sigma(hw_sigma) -> None:
    """Every value of a flat `hw_sigma` or a piecewise `Sigma` must be
    finite. `None` is accepted: it is the "calibrate me" sentinel on the
    Bermudan/American configs, filled in by `price_portfolio`."""
    if hw_sigma is None:
        return
    values = hw_sigma.values if hasattr(hw_sigma, "values") else [hw_sigma]
    if not np.all(np.isfinite(np.asarray(values, dtype=np.float64))):
        raise ValueError(f"hw_sigma must be finite; got {hw_sigma}")


def _validate_tenor(period_str: str, field_name: str) -> None:
    """Confirms `period_str` parses as a valid `ORE.Period` (e.g. "5Y",
    "18M"), re-raising ORE's own parse failure as a `ValueError` naming the
    offending field -- rather than letting a malformed tenor string surface
    as an opaque error deep inside `ORE.MakeVanillaSwap` at pricing time."""
    try:
        ORE.Period(period_str)
    except Exception as exc:
        raise ValueError(f"{field_name} is not a valid ORE.Period string: {period_str!r} ({exc})") from exc


def validate_single_evaluation_date(trade_configs) -> None:
    """Every trade in one run must share one `evaluation_date`.

    Each pricer measures its cashflow and exercise times from its own
    trade's `evaluation_date`, while a run has exactly one t=0 -- the
    simulation's, or the market a shock is applied to. A trade dated
    differently would be priced on a shifted time axis: finite, plausible,
    and wrong."""
    dates = {}
    for i, cfg in enumerate(trade_configs):
        dates.setdefault(cfg.evaluation_date.ISO(), i)
    if len(dates) > 1:
        listed = ", ".join(f"{iso} (first at trade[{i}])" for iso, i in dates.items())
        raise ValueError(
            f"all trades in one portfolio must share one evaluation_date; got {listed}"
        )
