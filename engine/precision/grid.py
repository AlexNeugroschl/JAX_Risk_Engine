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


def scale_exponent(amax, fmt_max: float):
    """The largest k with `amax * 2^k <= fmt_max`, within `SCALE_EXPONENTS` (0 -> the top)."""
    max_mantissa, max_exponent = math.frexp(fmt_max)
    mantissa, exponent = jnp.frexp(amax)          # amax = mantissa 2^exponent, mantissa in [0.5, 1)
    k = max_exponent - exponent - (mantissa > max_mantissa).astype(exponent.dtype)
    return jnp.clip(k, *SCALE_EXPONENTS)


def round_to_grid(x, mantissa_bits: int, min_exponent: int, uniforms=None):
    """`x` rounded to the grid of a format with `mantissa_bits` and `min_exponent`, to nearest
    (ties to even) or, given `uniforms` in [0, 1), stochastically. Exact in `x`'s dtype: the
    spacing is a power of two, kept a normal number of that dtype (XLA's CPU flushes
    subnormals), which coarsens only bfloat16's subnormals under float32, values at least 2^119
    below their block's largest. Non-finite values pass through."""
    _, exponent = jnp.frexp(x)                    # |x| in [2^(exponent - 1), 2^exponent)
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
    bits = {32: jnp.int32, 64: jnp.int64}[info.bits]
    biased = (exponent.astype(bits) - (info.minexp - 1)) << info.nmant
    return lax.bitcast_convert_type(biased, dtype)
