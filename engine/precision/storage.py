"""
Storage between stages: `store(x, format)` keeps an array in a format, `load(x, dtype)` reads
it back at the next stage's compute dtype (docs/planning/details/precision.md §6.2). Every
array that crosses a stage boundary goes through this pair, so the cast points are the only
places a format changes.

**float64 and float32** are stored as plain arrays of that dtype, rounded to nearest: storing
is a cast, and storing or loading at the dtype an array already has returns it unchanged, so
the float64 default moves no bit.

**The scaled formats** (float16, bfloat16, FP8) are stored as a `Stored`: the values in the
format and a block scale per `BLOCK` consecutive entries along the scenario axis (the axis
the scenario sharding splits, so blocks fall inside shards; a short last block when the scenario count is
not a multiple). Each scale is a power of two, kept as float32, that brings its block's
largest finite magnitude as close to the format's maximum as it goes without passing it. A
power of two scales exactly (the idea behind the OCP MX formats), so the scale adds no
rounding of its own and loading multiplies it back exactly.

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
which has none (they become NaN). Zero passes through. Magnitudes beyond float32's range
cannot be scaled into a format and overflow, as they would stored in float32.
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

#: The range of a scale's exponent: every scale, and its inverse, a normal float32.
_SCALE_EXPONENTS = (-126, 126)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class Stored:
    """An array kept in a scaled format: `values` in the format, with the array's shape, and
    `scales`, float32 powers of two, one per block of `BLOCK` entries along `axis` (the
    array's shape with `axis` of length ceil(n / BLOCK)). The value of an entry is
    `values * scales` of its block; `load` computes it. Compared by identity, as arrays have
    no single truth value."""
    values: jax.Array
    scales: jax.Array
    format: str
    axis: int

    def tree_flatten(self):
        return (self.values, self.scales), (self.format, self.axis)

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
    def nbytes(self) -> int:
        """Bytes held, the scales included."""
        return self.values.nbytes + self.scales.nbytes


def rounding_key(seed: int, stream: str) -> jax.Array:
    """The key of one stored array's stochastic rounding: `seed`'s key folded with `stream`, a
    name that identifies the array in its run (`"shocks"`, `"values/<trade id>"`). Arrays of
    different names round independently, and an array rounds the same in every run."""
    return jax.random.fold_in(jax.random.PRNGKey(seed), zlib.crc32(stream.encode()))


def store(x, name: str, rounding: str = "nearest", key: Optional[jax.Array] = None,
          axis: int = 0) -> Union[jax.Array, Stored]:
    """`x` kept in format `name`: a plain array at float64 and float32 (rounded to nearest;
    `rounding` is for the scaled formats), else a `Stored` with block scales along `axis`, the
    scenario axis, rounded by `rounding` (`stochastic` needs `key`)."""
    fmt = format_of(name)
    x = jnp.asarray(x)
    if not fmt.scaled:
        return x.astype(fmt.dtype)
    if rounding not in ROUNDINGS:
        raise ValueError(f"rounding {rounding!r}: the roundings are {list(ROUNDINGS)}")
    if rounding == "stochastic" and key is None:
        raise ValueError("stochastic rounding needs a key (`rounding_key`)")
    if x.ndim == 0:
        raise ValueError(f"storing in {name} needs a scenario axis for its block scales; got a scalar")
    axis = axis % x.ndim
    values, scales = _quantize(x, key if rounding == "stochastic" else None, name=name, axis=axis)
    return Stored(values, scales, name, axis)


def load(x, dtype) -> jax.Array:
    """A stored array (a plain array or a `Stored`) read at `dtype` (a compute dtype, or
    float64 for a reduction). Exact for a `Stored` loaded at float32 or wider."""
    dtype = jnp.dtype(dtype)
    if isinstance(x, Stored):
        return _dequantize(x.values, x.scales, axis=x.axis, dtype=dtype)
    return jnp.asarray(x).astype(dtype)


# ---------------------------------------------------------------------------
# The kernels: one compiled program per shape, dtype, format, axis and rounding.
# ---------------------------------------------------------------------------
@functools.partial(jax.jit, static_argnames=("name", "axis"))
def _quantize(x, key, name: str, axis: int):
    fmt = FORMATS[name]
    blocks = _blocks(x, axis)                                            # [nb, BLOCK, *rest]
    amax = jnp.max(jnp.where(jnp.isfinite(blocks), jnp.abs(blocks), 0), axis=1)
    k = _scale_exponent(amax, fmt.max)                                   # [nb, *rest]
    scaled = blocks * _pow2(k, x.dtype)[:, None]
    uniforms = None if key is None else jax.random.uniform(key, scaled.shape, x.dtype)
    rounded = _round_to_grid(scaled, fmt.mantissa_bits, fmt.min_exponent, uniforms)
    n = x.shape[axis]
    values = jnp.moveaxis(rounded.reshape((-1,) + rounded.shape[2:])[:n], 0, axis).astype(fmt.dtype)
    return values, jnp.moveaxis(_pow2(-k, jnp.float32), 0, axis)


@functools.partial(jax.jit, static_argnames=("axis", "dtype"))
def _dequantize(values, scales, axis: int, dtype):
    per_entry = lax.slice_in_dim(jnp.repeat(scales.astype(dtype), BLOCK, axis=axis), 0, values.shape[axis], axis=axis)
    return values.astype(dtype) * per_entry


def _blocks(x, axis: int):
    """`x` with `axis` moved first, padded with zeros to whole blocks, split `[nb, BLOCK, ...]`."""
    front = jnp.moveaxis(x, axis, 0)
    n = front.shape[0]
    padded = jnp.pad(front, [(0, -n % BLOCK)] + [(0, 0)] * (front.ndim - 1))
    return padded.reshape((-1, BLOCK) + front.shape[1:])


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
