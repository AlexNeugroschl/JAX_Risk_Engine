"""
The format table: every number format the engine knows, by name. It is the only place that
maps a name to a dtype, a bit count or a range (docs/planning/details/precision.md §6.1);
validation, the HTTP schema and storage read it.

| Name | Bits | Mantissa bits | Max | Scaled when stored | Storage from | Compute from |
|---|---|---|---|---|---|---|
| float64 | 64 | 52 | 1.8e308 | no | 1.4 | 1.4 |
| float32 | 32 | 23 | 3.4e38 | no | 1.4 | 1.4 |
| float16 | 16 | 10 | 65,504 | yes | 1.6 | 2.8 |
| bfloat16 | 16 | 7 | 3.4e38 | yes | 1.6 | 2.8 |
| float8_e4m3fn | 8 | 3 | 448 | yes | 1.6 | 2.8 |
| float8_e5m2 | 8 | 2 | 57,344 | yes | 1.6 | 2.8 |

"Storage from" and "compute from" are the roadmap steps that enable each use; a format used
before its step is refused, naming the step (`engine.precision.policy`). FP4 joins at
step 6.3.
"""
from dataclasses import dataclass
from typing import Mapping, Optional

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Format:
    """One row of the table. `storage_step`/`compute_step`: the roadmap step that enables the
    format for storage/compute, `None` once it is enabled."""
    name: str
    dtype: np.dtype
    bits: int
    mantissa_bits: int
    max: float
    scaled: bool
    storage_step: Optional[str]
    compute_step: Optional[str]


def _row(name, bits, scaled, storage_step=None, compute_step=None) -> Format:
    dtype = jnp.dtype(getattr(jnp, name))
    info = jnp.finfo(dtype)
    return Format(name, dtype, bits, int(info.nmant), float(info.max), scaled, storage_step, compute_step)


#: Name -> format, widest first.
FORMATS: Mapping[str, Format] = {f.name: f for f in (
    _row("float64", 64, scaled=False),
    _row("float32", 32, scaled=False),
    _row("float16", 16, scaled=True, storage_step="1.6", compute_step="2.8"),
    _row("bfloat16", 16, scaled=True, storage_step="1.6", compute_step="2.8"),
    _row("float8_e4m3fn", 8, scaled=True, storage_step="1.6", compute_step="2.8"),
    _row("float8_e5m2", 8, scaled=True, storage_step="1.6", compute_step="2.8"),
)}

#: Every name in the table, widest first (the HTTP schema's choices).
FORMAT_NAMES = tuple(FORMATS)


def format_of(name: str) -> Format:
    """The table's row for `name`; a name outside the table is refused."""
    try:
        return FORMATS[name]
    except (KeyError, TypeError):
        raise ValueError(f"unknown number format {name!r}; the formats are {list(FORMAT_NAMES)}") from None


def dtype_of(name: str) -> np.dtype:
    """The dtype of a format name."""
    return format_of(name).dtype


def name_of(dtype) -> str:
    """The format name of a dtype (the inverse of `dtype_of`), for reporting realized dtypes."""
    dtype = jnp.dtype(dtype)
    for f in FORMATS.values():
        if f.dtype == dtype:
            return f.name
    raise ValueError(f"dtype {dtype} is not a number format of the table")
