"""
The format table: every number format the engine knows, by name. It is the only place that
maps a name to a dtype, a bit count, a range or a kind of matrix product
(docs/planning/details/precision.md §6.1); validation, the HTTP schema, storage, the matrix
products and the report read it.

| Name | Bits | Mantissa bits | Min normal exponent | Max | Array | Scaled when stored | Product | Storage | Compute |
|---|---|---|---|---|---|---|---|---|---|
| float64 | 64 | 52 | -1022 | 1.8e308 | yes | no | its dtype | yes | yes |
| float32 | 32 | 23 | -126 | 3.4e38 | yes | no | its dtype | yes | yes |
| float16 | 16 | 10 | -14 | 65,504 | yes | yes | its dtype | yes | yes |
| bfloat16 | 16 | 7 | -126 | 3.4e38 | yes | yes | its dtype | yes | yes |
| float8_e4m3fn | 8 | 3 | -6 | 448 | yes | yes | its dtype | yes | yes |
| float8_e5m2 | 8 | 2 | -14 | 57,344 | yes | yes | its dtype | yes | yes |
| tensorfloat32 | 32 | 10 | -126 | 3.4e38 | no | no | TensorFloat-32 | no | yes |
| bfloat16_3x | 32 | 16 | -126 | 3.4e38 | no | no | three bfloat16 passes | no | yes |

An **array format** has a dtype of its own: values can be stored and accumulated in it. A
**product format** (`tensorfloat32`, `bfloat16_3x`) is only a way of running a matrix product
on float32 operands, as an NVIDIA GPU's TensorFloat-32 units (operands rounded to a 10-bit
mantissa) and a TPU's three bfloat16 passes (each operand split into a bfloat16 and the
bfloat16 of its remainder, the product of the two remainders dropped: about 16 bits) run one;
its values are float32 arrays (`dtype`, `bits`) and `mantissa_bits` is its operands'
precision. A product format is a compute format only, never storage or accumulate.

"Storage" and "compute" say whether each use is enabled, or name the planning item
(docs/planning) whose work enables it; a format used before then is refused, naming the item
(`engine.precision.policy`). Every array format is enabled for storage, and every format for
compute since the kernel rewrite (F-07); FP4 joins with F-07's FP4 part. The mantissa bits, the
minimum normal exponent and the max define the grid a value is rounded to when stored in a
scaled format (`engine.precision.storage`) or multiplied in a matrix product
(`engine.precision.products`).
"""
from dataclasses import dataclass
from typing import Mapping, Optional

import jax.numpy as jnp
import numpy as np

#: How a matrix product in a format runs: on operands of the format's own dtype, as
#: TensorFloat-32, or as three bfloat16 passes (`engine.precision.products`).
PRODUCT_KINDS = ("dtype", "tensorfloat32", "bfloat16_3x")


@dataclass(frozen=True)
class Format:
    """One row of the table. `array`: whether the format has a dtype of its own (see the module
    docstring); `dtype` is float32 for a product format. `min_exponent`: the exponent of the
    smallest normal number (below it the spacing stays that of the subnormals). `product`: how a
    matrix product whose compute format this is runs (`PRODUCT_KINDS`).
    `storage_pending`/`compute_pending`: the planning item (docs/planning) whose work enables
    the format for storage/compute, `None` once it is enabled."""
    name: str
    dtype: np.dtype
    bits: int
    mantissa_bits: int
    min_exponent: int
    max: float
    array: bool
    scaled: bool
    product: str
    storage_pending: Optional[str]
    compute_pending: Optional[str]

    @property
    def below_float32(self) -> bool:
        """Whether its products are narrower than float32's: every format but float64 and
        float32. Such a product runs on matrix units that accumulate in float32."""
        return self.product != "dtype" or self.bits < 32


def _row(name, bits, scaled, storage_pending=None, compute_pending=None) -> Format:
    dtype = jnp.dtype(getattr(jnp, name))
    info = jnp.finfo(dtype)
    return Format(name, dtype, bits, int(info.nmant), int(info.minexp), float(info.max), True, scaled, "dtype",
                  storage_pending, compute_pending)


def _product_row(name, mantissa_bits) -> Format:
    """A product format: float32 values, operands of `mantissa_bits`."""
    info = jnp.finfo(jnp.float32)
    return Format(name, jnp.dtype(jnp.float32), 32, mantissa_bits, int(info.minexp), float(info.max), False, False,
                  name, None, None)


#: Name -> format: the array formats widest first, then the product formats.
FORMATS: Mapping[str, Format] = {f.name: f for f in (
    _row("float64", 64, scaled=False),
    _row("float32", 32, scaled=False),
    _row("float16", 16, scaled=True),
    _row("bfloat16", 16, scaled=True),
    _row("float8_e4m3fn", 8, scaled=True),
    _row("float8_e5m2", 8, scaled=True),
    _product_row("tensorfloat32", 10),
    _product_row("bfloat16_3x", 16),
)}

#: Every name in the table, in its order (the HTTP schema's choices).
FORMAT_NAMES = tuple(FORMATS)


def format_of(name: str) -> Format:
    """The table's row for `name`; a name outside the table is refused."""
    try:
        return FORMATS[name]
    except (KeyError, TypeError):
        raise ValueError(f"unknown number format {name!r}; the formats are {list(FORMAT_NAMES)}") from None


def dtype_of(name: str) -> np.dtype:
    """The dtype of a format name (float32 for a product format)."""
    return format_of(name).dtype


def name_of(dtype) -> str:
    """The array format of a dtype (the inverse of `dtype_of` on array formats), for reporting
    realized dtypes."""
    dtype = jnp.dtype(dtype)
    for f in FORMATS.values():
        if f.array and f.dtype == dtype:
            return f.name
    raise ValueError(f"dtype {dtype} is not a number format of the table")
