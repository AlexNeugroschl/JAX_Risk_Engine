"""
Exact rounding onto a number format's grid and power-of-two scaling, shared by storage
(`engine.precision.storage`, block scales) and matrix products (`engine.precision.products`,
operand scales).

Every step is exact in the dtype it runs in: scales are powers of two built from their bits,
and a value is rounded to an integer number of the format's spacings, itself a power of two.
The rounding is not left to a dtype conversion because XLA converts float64 to FP8 and float16
through float32, rounding twice (measured: 2 of 2M values differ from a correct rounding for
e4m3, 118 for float16); a value rounded here lies on the format's grid, so its cast to the
format is exact.
"""
import math

import jax.numpy as jnp
from jax import lax

#: The range of a scale's exponent: every scale, and its inverse, a normal float32.
SCALE_EXPONENTS = (-126, 126)


def _frexp_exponent(x):
    """IEEE exponent from 32-bit words, without JAX's signed-64-bit frexp expansion.

    libtpu 0.0.42 rejects that expansion with mixed s32/u32 bitwise operands.
    These operations also preserve the exponent of subnormal inputs without floating
    point normalization (which may flush them to zero) on CPU/CUDA. The October 26
    hardware rerun found that TPU software FP64 bitcasts expose different words;
    this IEEE-64 branch is incompatible with that backend and is now refused.
    See docs/hardware-testing/hardware-correctness-rerun.md. TPU has no native FP64.
    """
    if x.dtype == jnp.float64:
        _require_ieee_float64()
        words = lax.bitcast_convert_type(x, jnp.uint32)
        low, high = words[..., 0], words[..., 1] & jnp.uint32(0x7fffffff)
        field = high >> jnp.uint32(20)
        fraction = high & jnp.uint32(0xfffff)
        subnormal = jnp.where(fraction != 0, -1010 - lax.clz(fraction).astype(jnp.int32),
                             -1042 - lax.clz(low).astype(jnp.int32))
        exponent = jnp.where(field == 0, subnormal, field.astype(jnp.int32) - 1022)
        special = (field == 0x7ff) | ((high == 0) & (low == 0))
    elif x.dtype == jnp.float32:
        bits = lax.bitcast_convert_type(x, jnp.uint32) & jnp.uint32(0x7fffffff)
        field = bits >> jnp.uint32(23)
        subnormal = -117 - lax.clz(bits & jnp.uint32(0x7fffff)).astype(jnp.int32)
        exponent = jnp.where(field == 0, subnormal, field.astype(jnp.int32) - 126)
        special = (field == 0xff) | (bits == 0)
    else:
        raise TypeError("precision-grid inputs must be float32 or float64")
    return jnp.where(special, jnp.int32(0), exponent)


def scale_exponent(amax, fmt_max: float):
    """The largest k with `amax * 2^k <= fmt_max`, within `SCALE_EXPONENTS`."""
    k = math.frexp(fmt_max)[1] - _frexp_exponent(amax)
    # Clipping the comparison scale avoids an out-of-range power for extreme inputs.
    # Clip the result after subtracting: a k beyond either limit stays saturated.
    scaled = amax * pow2(jnp.clip(k, *SCALE_EXPONENTS), amax.dtype)
    return jnp.clip(k - (scaled > fmt_max).astype(k.dtype), *SCALE_EXPONENTS)


def round_to_grid(x, mantissa_bits: int, min_exponent: int, uniforms=None):
    """`x` rounded to the grid of a format with `mantissa_bits` and `min_exponent`, to nearest
    (ties to even) or, given `uniforms` in [0, 1), stochastically. Exact in `x`'s dtype: the
    spacing is a power of two, kept a normal number of that dtype (XLA's CPU flushes
    subnormals), which coarsens only bfloat16's subnormals under float32, values at least 2^119
    below their block's largest. Non-finite values pass through."""
    exponent = _frexp_exponent(x)                # |x| in [2^(exponent - 1), 2^exponent)
    spacing = jnp.maximum(exponent - 1, min_exponent) - mantissa_bits
    spacing = jnp.maximum(spacing, jnp.finfo(x.dtype).minexp)
    units = x * pow2(-spacing, x.dtype)
    if uniforms is None:
        whole = jnp.round(units)
    else:
        lower = jnp.floor(units)
        whole = jnp.where(uniforms < units - lower, lower + 1, lower)  # keeps -0.0
    return jnp.where(jnp.isfinite(x), whole * pow2(spacing, x.dtype), x)


def pow2(exponent, dtype):
    """2^exponent in `dtype` (float32 or float64), exactly, for exponents of normal numbers:
    built from its bits."""
    info = jnp.finfo(dtype)
    biased = (exponent.astype(jnp.int32) - (info.minexp - 1)).astype(jnp.uint32)
    if info.bits == 64:
        _require_ieee_float64()
        high = biased << jnp.uint32(20)
        return lax.bitcast_convert_type(jnp.stack((jnp.zeros_like(high), high), axis=-1), dtype)
    return lax.bitcast_convert_type(biased << jnp.uint32(info.nmant), dtype)


def _require_ieee_float64():
    from engine.precision.execution import validate_format
    from engine.precision.products import default_device
    validate_format("float64", "precision grid", default_device())
