"""
`engine.precision`: the numeric precision of a run (docs/planning/details/precision.md).

    formats.py   the format table: name -> dtype, bits, mantissa bits, max, scaled, enabled step
    policy.py    StagePrecision, Precision and their validation; Precision.precision_for, the
                 pricing stage of one trade (overrides per product and per trade)
    storage.py   store / load, the only casts between stages

Depends on JAX and NumPy only: the pipeline imports it, it imports nothing from the pipeline.
"""
from engine.precision.formats import FORMAT_NAMES, FORMATS, Format, dtype_of, format_of, name_of
from engine.precision.policy import (
    OVERRIDES, RETIRED_SHAPE, STAGES, Overrides, Precision, StagePrecision, require_precision,
)
from engine.precision.storage import load, store

__all__ = [
    "FORMATS",
    "FORMAT_NAMES",
    "Format",
    "OVERRIDES",
    "Overrides",
    "Precision",
    "RETIRED_SHAPE",
    "STAGES",
    "StagePrecision",
    "dtype_of",
    "format_of",
    "load",
    "name_of",
    "require_precision",
    "store",
]
