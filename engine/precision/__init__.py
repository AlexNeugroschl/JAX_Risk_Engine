"""
`engine.precision`: the numeric precision of a run (docs/planning/details/precision.md).

    formats.py   the format table: name -> dtype, bits, mantissa bits, max, scaled, enabled step
    policy.py    StagePrecision, Precision and their validation; Precision.precision_for, the
                 pricing stage of one trade (overrides per product and per trade);
                 Precision.store, storage with the policy's rounding
    storage.py   store / load, the only casts between stages; Stored, a scaled format's values
                 and block scales (float16, bfloat16, FP8); nearest and stochastic rounding

Depends on JAX and NumPy only: the pipeline imports it, it imports nothing from the pipeline.
"""
from engine.precision.formats import FORMAT_NAMES, FORMATS, Format, dtype_of, format_of, name_of
from engine.precision.policy import (
    OVERRIDES, RETIRED_SHAPE, STAGES, Overrides, Precision, StagePrecision, require_precision,
)
from engine.precision.storage import BLOCK, ROUNDINGS, Stored, load, rounding_key, store

__all__ = [
    "BLOCK",
    "FORMATS",
    "FORMAT_NAMES",
    "Format",
    "OVERRIDES",
    "Overrides",
    "Precision",
    "RETIRED_SHAPE",
    "ROUNDINGS",
    "STAGES",
    "StagePrecision",
    "Stored",
    "dtype_of",
    "format_of",
    "load",
    "name_of",
    "require_precision",
    "rounding_key",
    "store",
]
