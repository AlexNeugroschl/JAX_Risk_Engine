"""
Field validators shared by every trade config's `__post_init__`.

A separate module from `request.py` to break an import cycle: `request` imports the
instrument configs, and the instruments import these validators. Only malformed input
(e.g. non-finite values) is rejected; zero and negative notionals and rates are allowed.
"""
import math

import numpy as np
import ORE


def _validate_common_fields(notional: float, fixed_rate: float, evaluation_date: ORE.Date) -> None:
    """Notional and fixed rate must be finite (zero and negative are allowed)."""
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
    """`period_str` must parse as an `ORE.Period` ("5Y", "18M"); raises a `ValueError`
    naming the field, rather than failing later inside `MakeVanillaSwap`."""
    try:
        ORE.Period(period_str)
    except Exception as exc:
        raise ValueError(f"{field_name} is not a valid ORE.Period string: {period_str!r} ({exc})") from exc


def validate_single_evaluation_date(trade_configs) -> None:
    """All trades in one run must share one `evaluation_date`: pricers measure times from
    their own trade's date, and a run has one t=0."""
    dates = {}
    for i, cfg in enumerate(trade_configs):
        dates.setdefault(cfg.evaluation_date.ISO(), i)
    if len(dates) > 1:
        listed = ", ".join(f"{iso} (first at trade[{i}])" for iso, i in dates.items())
        raise ValueError(
            f"all trades in one portfolio must share one evaluation_date; got {listed}"
        )
