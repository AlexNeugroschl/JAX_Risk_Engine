"""
The arithmetic of a stage (F-07; docs/planning/details/precision.md §3.1, §8): the two
formats its kernels compute with, one implementation for every precision (decision A-16).

    compute     the format of its matrix products: their operands are rounded to it, the one
                every path shares in the explicitly configured residual terms, and the
                product runs as the format runs (`engine.precision.products`)
    accumulate  the format the products accumulate in and every other operation of the stage
                is evaluated in: sums, the scan's carry, a path-independent level plus a path's
                deviation, the payoff formulas, the special functions

Low precision is put where it pays (§2.3): the matrix units, which run FP8, bfloat16 and
TensorFloat-32 natively, and the stored arrays (storage). No device runs elementwise
arithmetic below float32 faster, so a kernel's other operations run at `accumulate`, as a
native low-precision kernel runs them (a matrix unit's float32 accumulator, the vector unit's
float32 arithmetic). The kernels are written in difference form so that what enters a product
is a path's deviation from a level that does not depend on the path, never the level itself.
A level is computed in float64 (it has no path axis, so this is cheap even where float64 is
emulated) and meets each path's deviation in two terms of the accumulate format
(`add_level`, `deviation_from`): `hi + (lo + deviation)`, `hi` the level on the format's grid and
`lo` what it leaves, so the one rounding of a path's value is its own, and no rounding of the
level is shared by every path, which would be a bias more paths do not remove (measured: a
float32 bond's mean moved 0.04 with the level rounded once, 2026-10-10) (§8.2).

Kernels take `Arithmetic` as a pytree: formats, product options, execution and rounding are static
metadata; the random key and global sample ids are traced data. Programs compile once
per numerical strategy, rounding policy and shape, never per seed, date or trade.
`Precision.arithmetic_for` binds a stage or trade rounding stream.

Refused, naming the field, by `arithmetic_refusal` (shared with `StagePrecision`):

  * a compute format not enabled yet (the format table names the item that enables it);
  * an accumulate format that is a product format (it has no array of its own), or narrower
    than float32 (no matrix unit accumulates narrower), or narrower than compute's values;
  * an accumulate format other than float32 under a compute format below float32: the matrix
    units of those formats accumulate in float32, and a policy that said otherwise would not be
    what a native run computes.
"""
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from engine.precision.formats import Format, format_of, name_of
from engine.precision.grid import round_to_grid
from engine.precision.products import matmul
from engine.precision.options import ProductOptions


def arithmetic_refusal(compute: Format, accumulate: Format) -> Optional[Tuple[str, str]]:
    """Why `compute` and `accumulate` cannot be a stage's arithmetic, as `(field, reason)`, or
    None (see the module docstring)."""
    if compute.compute_pending:
        return "compute", (f"compute in {compute.name} is not enabled yet ({compute.compute_pending}); the compute "
                           f"formats are those of the format table without a pending item")
    if not accumulate.array:
        return "accumulate", f"{accumulate.name} is a matrix-product format, not an array format to accumulate in"
    if accumulate.bits < 32:
        return "accumulate", "narrower than float32: no matrix unit accumulates narrower; float32 or float64"
    if accumulate.bits < compute.bits:
        return "accumulate", f"narrower than compute={compute.name!r}"
    if compute.below_float32 and accumulate.name != "float32":
        return "accumulate", (f"compute={compute.name!r} runs on matrix units that accumulate in float32; "
                              f"accumulate must be 'float32'")
    return None


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Arithmetic:
    """A stage's compute and accumulate formats, by name (see the module docstring).
    `Arithmetic()` is float64 throughout. Stochastic products need an explicit `key`;
    `Precision.arithmetic_for` binds it for pipelines. Keys and `sample_ids` are data,
    never static JIT arguments. `execution` requires native products by default;
    software or unqualified product experiments explicitly use 'allow_emulation'."""
    compute: str = "float64"
    accumulate: str = "float64"
    rounding: str = "nearest"
    key: object = None
    sample_ids: object = None
    execution: str = "native"
    product: ProductOptions = ProductOptions()

    def __post_init__(self):
        if not isinstance(self.product, ProductOptions):
            raise TypeError("Arithmetic.product must be ProductOptions")
        from engine.precision.execution import EXECUTION_MODES
        if self.execution not in EXECUTION_MODES:
            raise ValueError(f"Arithmetic.execution must be one of {EXECUTION_MODES}; got {self.execution!r}")
        rows = {}
        for field in ("compute", "accumulate"):
            try:
                rows[field] = format_of(getattr(self, field))
            except ValueError as exc:
                raise ValueError(f"Arithmetic.{field}: {exc}") from None
        refusal = arithmetic_refusal(rows["compute"], rows["accumulate"])
        if refusal is not None:
            field, reason = refusal
            raise ValueError(f"Arithmetic.{field}={getattr(self, field)!r}: {reason}")
        self.product.validate(self.compute)
        if self.rounding not in ("nearest", "stochastic"):
            raise ValueError("Arithmetic.rounding must be 'nearest' or 'stochastic'")
        if self.rounding == "stochastic" and (not rows["compute"].below_float32
                                               or rows["compute"].product == "bfloat16_3x"):
            raise ValueError("Arithmetic.rounding='stochastic' needs a single-grid compute format below float32")

    def tree_flatten(self):
        return (self.key, self.sample_ids), (self.compute, self.accumulate, self.rounding, self.execution, self.product)

    @classmethod
    def tree_unflatten(cls, meta, data):
        compute, accumulate, rounding, execution, product = meta
        return cls(compute, accumulate, rounding, *data, execution=execution, product=product)

    def fold_in(self, data):
        """Derive a date/step/product stream; the key remains traced data."""
        return self if self.key is None else replace(self, key=jax.random.fold_in(self.key, data))

    @classmethod
    def compensated(cls, compute="float64", accumulate="float64", rounding="nearest", **kwargs):
        """Explicit historical scaled/residual profile; execution permission is separate."""
        return cls(compute, accumulate, rounding, product=ProductOptions.compensated(compute), **kwargs)

    @classmethod
    def of(cls, dtype) -> "Arithmetic":
        """float64 or float32 throughout, by dtype: the arithmetic of a stage fixed at it."""
        name = name_of(dtype)
        return cls(name, name)

    @property
    def dtype(self):
        """The accumulate format's dtype: the dtype a kernel's inputs, levels and results are
        in."""
        return format_of(self.accumulate).dtype

    def matmul(self, a, b, *, shared: str, stream: int = 0, sample_axis: int = 0):
        """`a @ b` in this arithmetic, `shared` the operand every path shares, `"left"` or
        `"right"` (`engine.precision.products.matmul`; decision A-23)."""
        return matmul(a, b, self.fold_in(stream), shared, sample_axis=sample_axis)

    def cast(self, x):
        """`x` in the accumulate format."""
        from engine.precision.execution import validate_format
        from engine.precision.products import default_device
        validate_format(self.accumulate, "accumulate", default_device())
        return jnp.asarray(x, dtype=self.dtype)


#: float64 throughout: the default, and the fixed stages' (A-10).
FLOAT64 = Arithmetic()


def level_terms(level, dtype) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """`level` (path-independent, as wide as `dtype` or wider) as `(hi, lo)` in `dtype`, `hi`
    its value on `dtype`'s grid and `lo` the rest, rounded once; `lo` is None when `level` is
    no wider than `dtype` (it is exact there: a float64 run's levels). `hi` is rounded on the
    grid, not by a round trip through `dtype`, which XLA may remove."""
    level = jnp.asarray(level)
    from engine.precision.execution import validate_format
    from engine.precision.products import default_device
    validate_format(level.dtype.name, "level", default_device())
    validate_format(jnp.dtype(dtype).name, "level output", default_device())
    dtype = jnp.dtype(dtype)
    if jnp.finfo(level.dtype).bits <= jnp.finfo(dtype).bits:
        return level.astype(dtype), None
    fmt = format_of(name_of(dtype))
    high = round_to_grid(level, fmt.mantissa_bits, fmt.min_exponent)
    return high.astype(dtype), (level - high).astype(dtype)


def add_level(level, deviation):
    """A path-independent `level` plus each path's `deviation`, in the deviation's format, with
    one rounding per path (see the module docstring)."""
    high, low = level_terms(level, deviation.dtype)
    return high + deviation if low is None else high + (low + deviation)


def deviation_from(level, value):
    """Each path's `value` less a path-independent `level`, in the value's format (the inverse
    of `add_level`)."""
    high, low = level_terms(level, value.dtype)
    return value - high if low is None else (value - high) - low
