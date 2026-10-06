"""
`engine.precision`: the numeric precision of a run (docs/planning/details/precision.md).

    formats.py   the format table: name -> dtype, bits, mantissa bits, max, scaled, enabled step
    policy.py    StagePrecision, Precision and their validation; Precision.precision_for, the
                 pricing stage of one trade (overrides per product and per trade);
                 Precision.store, storage with the policy's rounding
    storage.py   store / load, the only casts between stages; Stored, a scaled format's values
                 and block scales (float16, bfloat16, FP8); nearest and stochastic rounding
    estimate.py  the paired float64 sample: its size, the two-level estimator of a mean,
                 a quantile measured on it (decision A-13)
    report.py    PrecisionReport, on every result: the policy as run, the realized formats,
                 the devices, and each figure's estimate
    products.py  matmul, every matrix product of the engine's JAX code, at the precision of
                 its operands' compute format (product_precision), never the device's default

Depends on JAX and NumPy only: the pipeline imports it, it imports nothing from the pipeline.
"""
from engine.precision.estimate import MeanEstimate, QuantileEstimate, paired_paths, paired_quantile, two_level_mean
from engine.precision.formats import FORMAT_NAMES, FORMATS, Format, dtype_of, format_of, name_of
from engine.precision.policy import (
    OVERRIDES, RETIRED_SHAPE, STAGES, Overrides, Precision, StagePrecision, require_precision,
)
from engine.precision.products import matmul, product_precision
from engine.precision.report import PrecisionReport, devices_of, format_name, realized_format
from engine.precision.storage import BLOCK, ROUNDINGS, Stored, load, rounding_key, store

__all__ = [
    "BLOCK",
    "FORMATS",
    "FORMAT_NAMES",
    "Format",
    "MeanEstimate",
    "OVERRIDES",
    "Overrides",
    "Precision",
    "PrecisionReport",
    "QuantileEstimate",
    "RETIRED_SHAPE",
    "ROUNDINGS",
    "STAGES",
    "StagePrecision",
    "Stored",
    "devices_of",
    "dtype_of",
    "format_name",
    "format_of",
    "load",
    "matmul",
    "name_of",
    "paired_paths",
    "paired_quantile",
    "product_precision",
    "realized_format",
    "require_precision",
    "rounding_key",
    "store",
    "two_level_mean",
]
