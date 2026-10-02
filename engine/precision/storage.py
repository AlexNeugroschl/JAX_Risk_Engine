"""
Storage between stages: `store(x, format)` keeps an array in a format, `load(x, dtype)` reads
it back at the next stage's compute dtype (docs/planning/details/precision.md §6.2). Every
array that crosses a stage boundary goes through this pair, so the cast points are the only
places a format changes.

float64 and float32 are stored as plain arrays of that dtype: storing is a cast, and storing
or loading at the dtype an array already has returns it unchanged, so the float64 default
moves no bit. The scaled formats (float16, bfloat16, FP8) need block scales along the
scenario axis and a rounding mode; roadmap step 1.6 adds them, with a stored form that carries
the scales.
"""
import jax
import jax.numpy as jnp

from engine.precision.formats import format_of


def store(x, name: str) -> jax.Array:
    """`x` kept in format `name` (rounded to nearest)."""
    fmt = format_of(name)
    if fmt.scaled:
        raise ValueError(f"storing in {name} needs block scales, enabled by roadmap step {fmt.storage_step or '1.6'}")
    return jnp.asarray(x).astype(fmt.dtype)


def load(x, dtype) -> jax.Array:
    """A stored array read at `dtype` (a compute dtype, or float64 for a reduction)."""
    return jnp.asarray(x).astype(dtype)
