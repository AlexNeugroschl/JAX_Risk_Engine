"""
The format table: every number format the engine knows, by name. It is the only place that
maps a name to a dtype, a bit count, a range or a kind of matrix product
(docs/planning/details/precision.md §6.1); validation, the HTTP schema, storage, the matrix
products and the report read it.

| Name | Bits | Mantissa bits | Min normal exponent | Max | Array | Scaled when stored | Product | Shared terms | Shared operand held in | Storage | Compute |
|---|---|---|---|---|---|---|---|---|---|---|---|
| float64 | 64 | 52 | -1022 | 1.8e308 | yes | no | its dtype | 1 | its dtype | yes | yes |
| float32 | 32 | 23 | -126 | 3.4e38 | yes | no | its dtype | 1 | its dtype | yes | yes |
| float16 | 16 | 10 | -14 | 65,504 | yes | yes | its dtype | 2 | its dtype | yes | yes |
| bfloat16 | 16 | 7 | -126 | 3.4e38 | yes | yes | its dtype | 2 | its dtype | yes | yes |
| float8_e4m3fn | 8 | 3 | -6 | 448 | yes | yes | its dtype | 2 | its dtype | yes | yes |
| float8_e5m2 | 8 | 2 | -14 | 57,344 | yes | yes | its dtype | 2 | float8_e4m3fn | yes | yes |
| tensorfloat32 | 32 | 10 | -126 | 3.4e38 | no | no | TensorFloat-32 | 2 | its dtype | no | yes |
| bfloat16_3x | 32 | 16 | -126 | 3.4e38 | no | no | three bfloat16 passes | 1 | its dtype | no | yes |
| `<name>_1x` | as `<name>` | | | | no | no | as `<name>` | 1 | as `<name>` | no | yes |

An **array format** has a dtype of its own: values can be stored and accumulated in it. A
**product format** is only a way of running a matrix product, a compute format only, never
storage or accumulate: `tensorfloat32` and `bfloat16_3x` run one on float32 operands, as an
NVIDIA GPU's TensorFloat-32 units (operands rounded to a 10-bit mantissa) and a TPU's three
bfloat16 passes (each operand split into a bfloat16 and the bfloat16 of its remainder, the
product of the two remainders dropped: about 16 bits) run one; their values are float32 arrays
(`dtype`, `bits`) and `mantissa_bits` is their operands' precision.

**Shared terms** (decision A-23, I-96): a product below float32 multiplies a per-path operand by
one that every path shares (the Brownian bridge's matrix, the step moments, the market's
loadings, each flow's value at the level). Rounded once, the shared operand's error would be
common to every path, a bias that more paths do not remove; so every format below float32 whose
product is one pass carries it in two terms of the format (two passes, the second on what the
first leaves), and its single pass is a product format of its own, `<name>_1x` (`float16_1x`,
`bfloat16_1x`, `float8_e4m3fn_1x`, `float8_e5m2_1x`, `tensorfloat32_1x`): the row of `<name>`
under another name, neither stored nor accumulated in, with one term. `bfloat16_3x` carries each
operand in two bfloat16 terms already. `base` names the row whose grid, dtype and units a row's
products use: itself, or the format a single pass is of.

**The shared operand's dtype** (`shared_dtype`, I-97, 2026-10-19): FP8 matrix units take an e5m2
operand only beside an e4m3 one (an NVIDIA GPU's: cuBLASLt multiplies two e5m2 operands on its
other units, measured on an RTX 5060). So float8_e5m2's products hold the operand every path shares
in e4m3: each of its terms keeps e5m2's 2-bit mantissa, scaled to e4m3's top and rounded on
e4m3's subnormal spacing below e4m3's smallest normal number, a grid whose every value e4m3
holds exactly; the per-path operand is e5m2's own. The one difference from e5m2 throughout is
the shared operand's range: a value more than 2^-15 below its slice's largest is rounded on a
coarser grid, a change below 2^-15 of the slice's largest, far below e5m2's own rounding (2^-3).
Emulated or native, the products are these numbers (`engine.precision.products`).

"Storage" and "compute" say whether each use is enabled, or name the planning item
(docs/planning) whose work enables it; a format used before then is refused, naming the item
(`engine.precision.policy`). Every array format is enabled for storage, and every format for
compute since the kernel rewrite (F-07); FP4 joins with F-07's FP4 part. The mantissa bits, the
minimum normal exponent and the max define the grid a value is rounded to when stored in a
scaled format (`engine.precision.storage`) or multiplied in a matrix product
(`engine.precision.products`).
"""
from dataclasses import dataclass, replace
from typing import Mapping, Optional

import jax.numpy as jnp
import numpy as np

#: How a matrix product in a format runs: on operands of the format's own dtype, as
#: TensorFloat-32, or as three bfloat16 passes (`engine.precision.products`).
PRODUCT_KINDS = ("dtype", "tensorfloat32", "bfloat16_3x")


@dataclass(frozen=True)
class Format:
    """One row of the table. `array`: whether the format has a dtype of its own (see the module
    docstring); `dtype` is the dtype of its values, float32 for `tensorfloat32` and
    `bfloat16_3x`, its base's for a single pass. `min_exponent`: the exponent of the smallest
    normal number (below it the spacing stays that of the subnormals). `product`: how a matrix
    product whose compute format this is runs (`PRODUCT_KINDS`). `shared_terms`: how many terms
    of the format such a product carries the operand every path shares in; `base`: the row whose
    grid, dtype and units its products use (see the module docstring).
    `storage_pending`/`compute_pending`: the planning item (docs/planning) whose work enables
    the format for storage/compute, `None` once it is enabled. `shared_dtype`: the format whose
    range and dtype the operand every path shares is held in, `None` for its own (see the module
    docstring)."""
    name: str
    dtype: np.dtype
    bits: int
    mantissa_bits: int
    min_exponent: int
    max: float
    array: bool
    scaled: bool
    product: str
    shared_terms: int
    base: str
    storage_pending: Optional[str]
    compute_pending: Optional[str]
    shared_dtype: Optional[str] = None

    @property
    def below_float32(self) -> bool:
        """Whether its products are narrower than float32's: every format but float64 and
        float32. Such a product runs on matrix units that accumulate in float32."""
        return self.product != "dtype" or self.bits < 32


def _row(name, bits, scaled, storage_pending=None, compute_pending=None) -> Format:
    """An array format: its products carry the shared operand in two terms below float32."""
    dtype = jnp.dtype(getattr(jnp, name))
    info = jnp.finfo(dtype)
    return Format(name, dtype, bits, int(info.nmant), int(info.minexp), float(info.max), True, scaled, "dtype",
                  2 if bits < 32 else 1, name, storage_pending, compute_pending)


def _product_row(name, mantissa_bits, shared_terms) -> Format:
    """A product format on float32 operands of `mantissa_bits`."""
    info = jnp.finfo(jnp.float32)
    return Format(name, jnp.dtype(jnp.float32), 32, mantissa_bits, int(info.minexp), float(info.max), False, False,
                  name, shared_terms, name, None, None)


def _single_pass(row: Format) -> Format:
    """`<row>_1x`: `row`'s product in one pass, the shared operand rounded once (A-23)."""
    return replace(row, name=f"{row.name}_1x", array=False, scaled=False, shared_terms=1, base=row.name)


_TWO_TERMS = (_row("float16", 16, scaled=True), _row("bfloat16", 16, scaled=True),
              _row("float8_e4m3fn", 8, scaled=True),
              replace(_row("float8_e5m2", 8, scaled=True), shared_dtype="float8_e4m3fn"),
              _product_row("tensorfloat32", 10, shared_terms=2))

#: Name -> format: the array formats widest first, then the product formats, the single passes
#: last.
FORMATS: Mapping[str, Format] = {f.name: f for f in (
    _row("float64", 64, scaled=False),
    _row("float32", 32, scaled=False),
    *_TWO_TERMS,
    _product_row("bfloat16_3x", 16, shared_terms=1),
    *(_single_pass(row) for row in _TWO_TERMS),
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


def shared_format(fmt: Format) -> Format:
    """The grid and dtype a product in `fmt` holds the operand every path shares in: `fmt`'s own,
    or its mantissa on the range and in the dtype of `fmt.shared_dtype` (see the module
    docstring)."""
    if fmt.shared_dtype is None:
        return fmt
    held = format_of(fmt.shared_dtype)
    return replace(fmt, dtype=held.dtype, max=held.max, min_exponent=held.min_exponent)
