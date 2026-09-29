"""
Portfolio-level validation, plus re-exports of the trade-field validators.

The trade-field validators moved to `engine.instruments._validation` (audit A-5) so that
instrument modules no longer import this package; they are re-exported here for existing
callers.
"""
from engine.instruments._validation import (  # noqa: F401  (re-exports)
    _validate_common_fields,
    _validate_hw_a,
    _validate_hw_sigma,
)
from engine.models.ore_builders import validate_tenor as _validate_tenor  # noqa: F401


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
