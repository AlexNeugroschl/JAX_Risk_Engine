"""
Field validators shared by every trade config's `__post_init__`.

They live in the instruments layer so that instrument modules import nothing from
`engine.portfolio`, the layer above them (audit A-5; enforced by
tests/test_import_layering.py). Only malformed input (e.g. non-finite values) is rejected;
zero and negative notionals and rates are allowed.
"""
import math

import numpy as np
import ORE


def _validate_common_fields(notional: float, fixed_rate: float, evaluation_date: ORE.Date) -> None:
    """Notional and fixed rate must be finite (zero and negative are allowed), and the
    evaluation date must be an `ORE.Date`."""
    if not math.isfinite(notional):
        raise ValueError(f"notional must be finite; got {notional}")
    if not math.isfinite(fixed_rate):
        raise ValueError(f"fixed_rate must be finite; got {fixed_rate}")
    if not isinstance(evaluation_date, ORE.Date):
        raise TypeError(f"evaluation_date must be an ORE.Date; got {evaluation_date!r}")


def _validate_hw_sigma(hw_sigma) -> None:
    """Every value of a flat `hw_sigma` or a piecewise `Sigma` must be finite. `None` is
    accepted: it is the "calibrate me" sentinel on the Bermudan/American configs."""
    if hw_sigma is None:
        return
    values = hw_sigma.values if hasattr(hw_sigma, "values") else [hw_sigma]
    if not np.all(np.isfinite(np.asarray(values, dtype=np.float64))):
        raise ValueError(f"hw_sigma must be finite; got {hw_sigma}")


def _validate_hw_a(hw_a: float) -> None:
    """Hull-White mean reversion must be finite and strictly positive. QuantLib's
    `HullWhite` model (and so `ORE.JamshidianSwaptionEngine`) raises at `a = 0`; without this
    check the bond-option volatility divides by zero and every option silently prices at
    intrinsic value (I-41)."""
    if not math.isfinite(hw_a) or hw_a <= 0.0:
        raise ValueError(
            f"hw_a must be finite and > 0 for the Hull-White pricers (QuantLib's HullWhite "
            f"raises at a = 0); got {hw_a}")


#: ORE's option settlement types (`Settlement::Type`).
SETTLEMENT_TYPES = ("Physical", "Cash")


def _validate_settlement(settlement: str) -> None:
    """Physical or Cash, as ORE's `OptionData/Settlement`."""
    if settlement not in SETTLEMENT_TYPES:
        raise ValueError(f"settlement must be one of {SETTLEMENT_TYPES}; got {settlement!r}")
