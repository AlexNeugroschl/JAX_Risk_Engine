"""
Field validators shared by every trade config's `__post_init__`.

They live in the instruments layer so that instrument modules import nothing from
`engine.portfolio`, the layer above them (audit A-5; enforced by
tests/test_import_layering.py). Only malformed input (e.g. non-finite values) is rejected;
zero and negative notionals and rates are allowed.

Every trade config names itself and its valuation date: `trade_id`, ORE's
`<Trade id>`, and `evaluation_date`, both required keyword fields. A trade never reads ORE's
thread-local evaluation date (I-64), and results are never keyed by position alone (I-10).
"""
import math

import ORE


def _validate_identity(trade_id: str, evaluation_date: ORE.Date) -> None:
    """`trade_id` must be a non-empty string and `evaluation_date` an `ORE.Date`."""
    if not isinstance(trade_id, str) or not trade_id:
        raise ValueError(f"trade_id must be a non-empty string; got {trade_id!r}")
    if not isinstance(evaluation_date, ORE.Date):
        raise TypeError(f"trade {trade_id!r}: evaluation_date must be an ORE.Date; got {evaluation_date!r}")


def _validate_common_fields(notional: float, fixed_rate: float, evaluation_date: ORE.Date) -> None:
    """Notional and fixed rate must be finite (zero and negative are allowed), and the
    evaluation date must be an `ORE.Date`."""
    if not math.isfinite(notional):
        raise ValueError(f"notional must be finite; got {notional}")
    if not math.isfinite(fixed_rate):
        raise ValueError(f"fixed_rate must be finite; got {fixed_rate}")
    if not isinstance(evaluation_date, ORE.Date):
        raise TypeError(f"evaluation_date must be an ORE.Date; got {evaluation_date!r}")


#: ORE's option settlement types (`Settlement::Type`).
SETTLEMENT_TYPES = ("Physical", "Cash")


def _validate_settlement(settlement: str) -> None:
    """Physical or Cash, as ORE's `OptionData/Settlement`."""
    if settlement not in SETTLEMENT_TYPES:
        raise ValueError(f"settlement must be one of {SETTLEMENT_TYPES}; got {settlement!r}")
