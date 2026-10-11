"""
`engine.precision`: the numeric precision of a run (docs/planning/details/precision.md).

    formats.py   the format table: name -> dtype, bits, mantissa bits, max, array or product
                 format, scaled, the kind of matrix product, the shared operand's terms (each
                 format's single pass `<name>_1x` a row of its own), enabled step
    arithmetic.py  Arithmetic, a stage's compute and accumulate formats as its kernels take
                 them (static), and their validation
    policy.py    StagePrecision, Precision and their validation; Precision.precision_for, the
                 pricing stage of one trade (overrides per product and per trade);
                 Precision.store, storage with the policy's rounding
    storage.py   store / load, the only casts between stages; Stored, a scaled format's values,
                 block scales and block offsets (float16, bfloat16, FP8); nearest and stochastic
                 rounding
    estimate.py  the paired float64 sample: its size, the two-level estimator of a mean,
                 a quantile measured on it (decision A-13)
    report.py    PrecisionReport, on every result: the policy as run, the realized formats,
                 the devices, and each figure's estimate
    products.py  matmul, every matrix product of the engine's JAX code, in a stage's compute
                 format and accumulated in its accumulate format, the operand every path shares
                 (SHARED: left or right) in two terms of a format below float32 (A-23), its
                 precision stated in the program (product_precision), never the device's
                 default; product_execution, native or emulated on a device
    grid.py      exact rounding onto a format's grid and power-of-two scales (storage and
                 products)

Depends on JAX and NumPy only: the pipeline imports it, it imports nothing from the pipeline.
"""
from engine.precision.arithmetic import FLOAT64, Arithmetic, add_level, deviation_from, level_terms
from engine.precision.estimate import MeanEstimate, QuantileEstimate, paired_paths, paired_quantile, two_level_mean
from engine.precision.formats import FORMAT_NAMES, FORMATS, Format, dtype_of, format_of, name_of
from engine.precision.policy import (
    OVERRIDES, RETIRED_SHAPE, STAGES, Overrides, Precision, StagePrecision, require_precision,
)
from engine.precision.products import SHARED, matmul, product_execution, product_precision
from engine.precision.report import PrecisionReport, devices_of, format_name, realized_format
from engine.precision.storage import BLOCK, OFFSETS, ROUNDINGS, Stored, load, rounding_key, store

__all__ = [
    "Arithmetic",
    "BLOCK",
    "FLOAT64",
    "FORMATS",
    "FORMAT_NAMES",
    "Format",
    "MeanEstimate",
    "OFFSETS",
    "OVERRIDES",
    "Overrides",
    "Precision",
    "PrecisionReport",
    "QuantileEstimate",
    "RETIRED_SHAPE",
    "ROUNDINGS",
    "SHARED",
    "STAGES",
    "StagePrecision",
    "Stored",
    "add_level",
    "deviation_from",
    "devices_of",
    "dtype_of",
    "format_name",
    "format_of",
    "level_terms",
    "load",
    "matmul",
    "name_of",
    "paired_paths",
    "paired_quantile",
    "product_execution",
    "product_precision",
    "realized_format",
    "require_precision",
    "rounding_key",
    "store",
    "two_level_mean",
]
