"""
Storage between stages: `store(x, format)` keeps an array in a format, `load(x, dtype)` reads
it back at the next stage's compute dtype (docs/planning/details/precision.md §6.2). Every
array that crosses a stage boundary goes through this pair, so the cast points are the only
places a format changes.

**float64 and float32** are stored as plain arrays of that dtype, rounded to nearest: storing
is a cast, and storing or loading at the dtype an array already has returns it unchanged, so
the float64 default moves no bit.

**The scaled formats** (float16, bfloat16, FP8) are stored as a `Stored`: the values in the
format, a block scale per `BLOCK` consecutive entries along the scenario axis (the axis the
scenario sharding splits, so blocks fall inside shards; a short last block when the scenario
count is not a multiple) and, by default, a block offset beside each scale. Each scale is a
power of two, kept as float32, that brings its block's largest finite magnitude as close to
the format's maximum as it goes without passing it. A power of two scales exactly (the idea
behind the OCP MX formats), so the scale adds no rounding of its own and loading multiplies it
back exactly.

**The block offset** (`offset="block"`, I-75): a block whose entries sit close together
against their level (a bond's cube column, near 1e6 with a spread near 1e4 across paths; a
long log discount factor) would spend the format's few mantissa bits on the level, and keep
its spread, the part that carries risk, to about one bit; rounded to nearest every entry of
the block then moves the same way, a bias that more paths do not remove. With an offset, a
concentrated block, one whose finite entries all have one sign and whose farthest from zero is
at most three times its nearest (its range at most twice its nearest magnitude), stores their
deviations from its midpoint, the mean of its largest and smallest finite entries rounded to
float32 and kept beside the scale. The scale is then set by the deviations, at most half the
block's range, so the format's bits go to the spread; and the deviations lie symmetrically
about zero, so rounding them to nearest moves as many up as down. A rounding error is at most a
spacing of the format at the largest deviation, an eighth of the range in the coarsest format
(e5m2), so in a concentrated block no entry changes sign: an option's value is never
negative, a discount factor never below zero. Every other block keeps an offset of 0 and is
stored as without one, bit for bit: one that holds zero or both signs has no level apart from
zero (its range is at least its largest magnitude), and one that reaches nearer zero than half
its range would gain less than a bit, at the cost of signs. The midpoint never makes the
largest deviation larger than the largest entry, so the rounding error bound is never worse
than without an offset, at 4 more bytes per block. The deviations are computed from the stored
offset in the input's dtype, so the offset adds no error but that dtype's rounding, far below
the format's step; loading adds it back with one rounding in the loaded dtype. `offset="none"`
stores the entries themselves, the layout the OCP MX formats and the FP8 matrix units read.

Rounding, in the scaled domain:

  * `nearest`: to the nearest value of the format, ties to even;
  * `stochastic`: down or up to the format's two neighbouring values, up with probability
    equal to the distance from the lower one in units of their spacing, so each value's
    rounding error has mean zero. The uniform draws come from a `key`, so a run reproduces.

Both round on the format's own grid, computed in the input's dtype
(`engine.precision.grid`, shared with the matrix products' operands): the spacing at `x` is
`2^(max(binade(x), min_exponent) - mantissa_bits)` (the subnormal spacing below the smallest
normal), and `x / spacing` is rounded to an integer. Every step is exact (powers of two), and
the cast to the format at the end is too, since the result lies on the format's grid. The
rounding is not left to the dtype conversion: XLA converts float64 to FP8 and float16 through
float32, rounding twice. NaN passes through, and so do infinities except in `float8_e4m3fn`,
which has none (they become NaN), and so do zeros, signed (a block holding one has offset 0).
Magnitudes beyond float32's range cannot be scaled into a format and overflow, as they would
stored in float32.
"""
import functools
import zlib
from dataclasses import dataclass
from typing import Optional, Union

import jax
import jax.numpy as jnp
from jax import lax

from engine.precision.formats import FORMATS, format_of
from engine.precision.grid import pow2 as _pow2
from engine.precision.grid import round_to_grid as _round_to_grid
from engine.precision.grid import scale_exponent as _scale_exponent

#: Entries per block scale along the scenario axis (an engineering default, precision.md §16).
BLOCK = 32

#: The rounding modes of the scaled formats.
ROUNDINGS = ("nearest", "stochastic")

#: The layouts of a scaled format's blocks: a float32 offset beside each block scale, the
#: deviations from it stored (`"block"`), or none, the entries themselves stored (`"none"`).
OFFSETS = ("block", "none")

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class Stored:
    """An array kept in a scaled format: `values` in the format, with the array's shape;
    `scales`, float32 powers of two, one per block of `BLOCK` entries along `axis` (the
    array's shape with `axis` of length ceil(n / BLOCK)); and `offsets`, float32, one beside
    each scale, or None (`offset="none"`). The value of an entry is
    `values * scales + offsets` of its block; `load` computes it. Compared by identity, as
    arrays have no single truth value."""
    values: jax.Array
    scales: jax.Array
    offsets: Optional[jax.Array]
    format: str
    axis: int

    def tree_flatten(self):
        return (self.values, self.scales, self.offsets), (self.format, self.axis)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)

    @property
    def shape(self):
        return self.values.shape

    @property
    def dtype(self):
        """The format's dtype: the realized storage dtype."""
        return self.values.dtype

    @property
    def offset(self) -> str:
        """The block layout, read from the arrays: `"block"` with offsets, else `"none"`."""
        return "none" if self.offsets is None else "block"

    @property
    def nbytes(self) -> int:
        """Bytes held, the scales and offsets included."""
        return self.values.nbytes + self.scales.nbytes + (0 if self.offsets is None else self.offsets.nbytes)


def rounding_key(seed: int, stream: str) -> jax.Array:
    """The key of one stored array's stochastic rounding: `seed`'s key folded with `stream`, a
    name that identifies the array in its run (`"shocks"`, `"values/<trade id>"`). Arrays of
    different names round independently, and an array rounds the same in every run."""
    return jax.random.fold_in(jax.random.PRNGKey(seed), zlib.crc32(stream.encode()))


def store(x, name: str, rounding: str = "nearest", key: Optional[jax.Array] = None,
          axis: int = 0, offset: str = "block", *, scaling: str = "power_of_two") -> Union[jax.Array, Stored]:
    """`x` kept in format `name`: a plain array at float64 and float32 (rounded to nearest;
    `rounding` and `offset` are for the scaled formats), else a `Stored` with block scales
    along `axis`, the scenario axis, and with `offset="block"` an offset beside each, rounded
    by `rounding` (`stochastic` needs `key`). With `scaling="none"` and
    `offset="none"`, low formats are plain quantized arrays without block metadata."""
    fmt = format_of(name)
    from engine.precision.execution import validate_format
    from engine.precision.products import default_device
    validate_format(name, "storage", default_device())
    x = jnp.asarray(x)
    if not fmt.scaled:
        return x.astype(fmt.dtype)
    if scaling not in ("power_of_two", "none"):
        raise ValueError("storage scaling must be 'power_of_two' or 'none'")
    if rounding not in ROUNDINGS:
        raise ValueError(f"rounding {rounding!r}: the roundings are {list(ROUNDINGS)}")
    if offset not in OFFSETS:
        raise ValueError(f"offset {offset!r}: the block layouts are {list(OFFSETS)}")
    if rounding == "stochastic" and key is None:
        raise ValueError("stochastic rounding needs a key (`rounding_key`)")
    if scaling == "none":
        if offset != "none":
            raise ValueError("plain storage requires offset='none'")
        return _plain_quantize(x, key if rounding == "stochastic" else None, name=name)
    if x.ndim == 0:
        raise ValueError(f"storing in {name} needs a scenario axis for its block scales; got a scalar")
    axis = axis % x.ndim
    values, scales, offsets = _quantize(x, key if rounding == "stochastic" else None, name=name, axis=axis,
                                        offset=offset == "block")
    return Stored(values, scales, offsets, name, axis)


@functools.partial(jax.jit, static_argnames=("name",))
def _plain_quantize(x, key, *, name):
    fmt = FORMATS[name]
    uniforms = None if key is None else jax.random.uniform(key, x.shape, x.dtype)
    return _round_to_grid(x, fmt.mantissa_bits, fmt.min_exponent, uniforms).astype(fmt.dtype)


def load(x, dtype) -> jax.Array:
    """A stored array (a plain array or a `Stored`) read at `dtype` (a compute dtype, or
    float64 for a reduction). A `Stored` without offsets loads exactly at float32 or wider;
    adding its offsets back rounds once in `dtype`."""
    dtype = jnp.dtype(dtype)
    from engine.precision.execution import validate_format
    from engine.precision.products import default_device
    validate_format(dtype.name, "load", default_device())
    if isinstance(x, Stored):
        return _dequantize(x.values, x.scales, x.offsets, axis=x.axis, dtype=dtype)
    return jnp.asarray(x).astype(dtype)


# ---------------------------------------------------------------------------
# The kernels: one compiled program per shape, dtype, format, axis, rounding and layout.
# ---------------------------------------------------------------------------
@functools.partial(jax.jit, static_argnames=("name", "axis", "offset"))
def _quantize(x, key, name: str, axis: int, offset: bool):
    fmt = FORMATS[name]
    blocks = _blocks(x, axis)                                            # [nb, BLOCK, *rest]
    finite = jnp.isfinite(blocks)                                        # the padding is NaN
    levels = _levels(blocks, finite) if offset else None                 # [nb, *rest], float32
    if offset:
        blocks = blocks - levels.astype(x.dtype)[:, None]
    amax = jnp.max(jnp.where(finite, jnp.abs(blocks), 0), axis=1)
    k = _scale_exponent(amax, fmt.max)                                   # [nb, *rest]
    scaled = blocks * _pow2(k, x.dtype)[:, None]
    uniforms = None if key is None else jax.random.uniform(key, scaled.shape, x.dtype)
    rounded = _round_to_grid(scaled, fmt.mantissa_bits, fmt.min_exponent, uniforms)
    n = x.shape[axis]
    values = jnp.moveaxis(rounded.reshape((-1,) + rounded.shape[2:])[:n], 0, axis).astype(fmt.dtype)
    per_block = lambda a: jnp.moveaxis(a, 0, axis)  # noqa: E731
    return values, per_block(_pow2(-k, jnp.float32)), None if levels is None else per_block(levels)


@functools.partial(jax.jit, static_argnames=("axis", "dtype"))
def _dequantize(values, scales, offsets, axis: int, dtype):
    def per_entry(blocks):
        return lax.slice_in_dim(jnp.repeat(blocks.astype(dtype), BLOCK, axis=axis), 0, values.shape[axis], axis=axis)

    deviations = values.astype(dtype) * per_entry(scales)
    if offsets is None:
        return deviations
    offsets = per_entry(offsets)
    return jnp.where(offsets == 0, deviations, deviations + offsets)  # keeps a block of offset 0 bit for bit, -0.0 too


def _blocks(x, axis: int):
    """`x` with `axis` moved first, padded with NaN to whole blocks (so the padding counts
    towards no block's scale or offset; it is cut off after rounding), split
    `[nb, BLOCK, ...]`."""
    front = jnp.moveaxis(x, axis, 0)
    n = front.shape[0]
    padded = jnp.pad(front, [(0, -n % BLOCK)] + [(0, 0)] * (front.ndim - 1), constant_values=jnp.nan)
    return padded.reshape((-1, BLOCK) + front.shape[1:])


def _levels(blocks, finite):
    """Each block's offset: for a concentrated block, one whose finite entries all have one sign
    and whose farthest from zero is at most three times its nearest (its range at most twice
    its nearest magnitude), the midpoint of its largest and smallest finite entries, rounded to
    float32; else 0. Each end is halved before the two are added, which is exact and cannot
    overflow. The midpoint is rounded on float32's grid in the blocks' dtype before the cast, so
    the deviations taken from it are taken from the offset kept: XLA may remove a float64 ->
    float32 -> float64 round trip inside a fusion (`xla_allow_excess_precision`), which on a GPU
    took a one-entry block's deviation from the unrounded midpoint (2026-10-11, I-99)."""
    top = jnp.max(jnp.where(finite, blocks, -jnp.inf), axis=1)
    bottom = jnp.min(jnp.where(finite, blocks, jnp.inf), axis=1)
    concentrated = jnp.any(finite, axis=1) & (((bottom > 0) & (top <= 3 * bottom))
                                              | ((top < 0) & (bottom >= 3 * top)))
    float32 = FORMATS["float32"]
    level = _round_to_grid(jnp.where(concentrated, 0.5 * top + 0.5 * bottom, 0), float32.mantissa_bits,
                           float32.min_exponent)
    return level.astype(jnp.float32)
