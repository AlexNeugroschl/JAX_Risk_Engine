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

Both round on the format's own grid, computed in the input's dtype: the spacing at `x` is
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
import math
import zlib
from dataclasses import dataclass
from typing import Optional, Union

import jax
import jax.numpy as jnp
from jax import lax

from engine.precision.formats import FORMATS, format_of

#: Entries per block scale along the scenario axis (an engineering default, precision.md §16).
BLOCK = 32

#: The rounding modes of the scaled formats.
ROUNDINGS = ("nearest", "stochastic")

#: The layouts of a scaled format's blocks: a float32 offset beside each block scale, the
#: deviations from it stored (`"block"`), or none, the entries themselves stored (`"none"`).
OFFSETS = ("block", "none")

#: The range of a scale's exponent: every scale, and its inverse, a normal float32.
_SCALE_EXPONENTS = (-126, 126)


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
          axis: int = 0, offset: str = "block") -> Union[jax.Array, Stored]:
    """`x` kept in format `name`: a plain array at float64 and float32 (rounded to nearest;
    `rounding` and `offset` are for the scaled formats), else a `Stored` with block scales
    along `axis`, the scenario axis, and with `offset="block"` an offset beside each, rounded
    by `rounding` (`stochastic` needs `key`)."""
    fmt = format_of(name)
    x = jnp.asarray(x)
    if not fmt.scaled:
        return x.astype(fmt.dtype)
    if rounding not in ROUNDINGS:
        raise ValueError(f"rounding {rounding!r}: the roundings are {list(ROUNDINGS)}")
    if offset not in OFFSETS:
        raise ValueError(f"offset {offset!r}: the block layouts are {list(OFFSETS)}")
    if rounding == "stochastic" and key is None:
        raise ValueError("stochastic rounding needs a key (`rounding_key`)")
    if x.ndim == 0:
        raise ValueError(f"storing in {name} needs a scenario axis for its block scales; got a scalar")
    axis = axis % x.ndim
    values, scales, offsets = _quantize(x, key if rounding == "stochastic" else None, name=name, axis=axis,
                                        offset=offset == "block")
    return Stored(values, scales, offsets, name, axis)


def load(x, dtype) -> jax.Array:
    """A stored array (a plain array or a `Stored`) read at `dtype` (a compute dtype, or
    float64 for a reduction). A `Stored` without offsets loads exactly at float32 or wider;
    adding its offsets back rounds once in `dtype`."""
    dtype = jnp.dtype(dtype)
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
    overflow."""
    top = jnp.max(jnp.where(finite, blocks, -jnp.inf), axis=1)
    bottom = jnp.min(jnp.where(finite, blocks, jnp.inf), axis=1)
    concentrated = jnp.any(finite, axis=1) & (((bottom > 0) & (top <= 3 * bottom))
                                              | ((top < 0) & (bottom >= 3 * top)))
    return jnp.where(concentrated, 0.5 * top + 0.5 * bottom, 0).astype(jnp.float32)


def _scale_exponent(amax, fmt_max: float):
    """The largest k with `amax * 2^k <= fmt_max`, within `_SCALE_EXPONENTS` (0 -> the top)."""
    max_mantissa, max_exponent = math.frexp(fmt_max)
    mantissa, exponent = jnp.frexp(amax)          # amax = mantissa 2^exponent, mantissa in [0.5, 1)
    k = max_exponent - exponent - (mantissa > max_mantissa).astype(exponent.dtype)
    return jnp.clip(k, *_SCALE_EXPONENTS)


def _round_to_grid(x, mantissa_bits: int, min_exponent: int, uniforms):
    """`x` rounded to the grid of a format with `mantissa_bits` and `min_exponent`, to nearest
    (ties to even) or, given `uniforms` in [0, 1), stochastically. Exact in `x`'s dtype: the
    spacing is a power of two, kept a normal number of that dtype (XLA's CPU flushes
    subnormals), which coarsens only bfloat16's subnormals under float32, values at least 2^119
    below their block's largest."""
    _, exponent = jnp.frexp(x)                    # |x| in [2^(exponent - 1), 2^exponent)
    spacing = jnp.maximum(exponent - 1, min_exponent) - mantissa_bits
    spacing = jnp.maximum(spacing, jnp.finfo(x.dtype).minexp)
    units = x * _pow2(-spacing, x.dtype)
    if uniforms is None:
        whole = jnp.round(units)
    else:
        lower = jnp.floor(units)
        whole = jnp.where(uniforms < units - lower, lower + 1, lower)  # keeps -0.0
    return jnp.where(jnp.isfinite(x), whole * _pow2(spacing, x.dtype), x)


def _pow2(exponent, dtype):
    """2^exponent in `dtype` (float32 or float64), exactly, for exponents of normal numbers:
    built from its bits."""
    info = jnp.finfo(dtype)
    bits = {32: jnp.int32, 64: jnp.int64}[info.bits]
    biased = (exponent.astype(bits) - (info.minexp - 1)) << info.nmant
    return lax.bitcast_convert_type(biased, dtype)
