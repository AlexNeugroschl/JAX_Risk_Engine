"""Explicit floating-point preparation at kernel boundaries; no device selection."""
from dataclasses import fields, is_dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np


def cast_floating(value, dtype, *, scalars=False):
    """Cast numerical data without changing schedules, integer indices or static choices.

    Used during preparation, never as an operation-by-operation fallback. Array dtypes
    remain JIT inputs, so FP32 and FP64 preparations cannot share a wrong specialization.
    """
    if isinstance(value, (jax.Array, jax.core.Tracer)):
        return value.astype(dtype) if jnp.issubdtype(value.dtype, jnp.floating) else value
    if isinstance(value, np.ndarray):
        return value.astype(dtype, copy=False) if np.issubdtype(value.dtype, np.floating) else value
    if scalars and isinstance(value, (float, np.floating)):
        return float(np.asarray(value, dtype=dtype))
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{f.name: cast_floating(getattr(value, f.name), dtype, scalars=scalars)
                                 for f in fields(value) if f.init})
    if isinstance(value, dict):
        return {k: cast_floating(v, dtype, scalars=scalars) for k, v in value.items()}
    if isinstance(value, tuple):
        values = [cast_floating(v, dtype, scalars=scalars) for v in value]
        return type(value)(*values) if hasattr(value, "_fields") else tuple(values)
    if isinstance(value, list):
        return [cast_floating(v, dtype, scalars=scalars) for v in value]
    return value
