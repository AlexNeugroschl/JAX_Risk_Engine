"""Sampling uncertainty without treating dependent observations as IID.

Reusable array operations only. Pricing owns which observations belong to a scramble;
market risk owns the statistic and historical block length. All reductions are float64.
"""
from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames="groups")
def delete_group(sample, index, groups):
    """Delete one contiguous equal-sized replicate, with the index as traced data."""
    by_group = sample.reshape((groups, -1) + sample.shape[1:])
    indices = jnp.arange(groups - 1, dtype=jnp.int32)
    indices = indices + (indices >= jnp.asarray(index, dtype=jnp.int32)).astype(jnp.int32)
    return by_group[indices].reshape((-1,) + sample.shape[1:])


@jax.jit
def jackknife_error(estimates):
    """Delete-one-replicate jackknife SE of a pooled nonlinear statistic."""
    estimates = jnp.asarray(estimates, dtype=jnp.float64)
    groups = estimates.shape[0]
    return jnp.sqrt((groups - 1) * jnp.mean((estimates - jnp.mean(estimates, axis=0)) ** 2, axis=0))


@partial(jax.jit, static_argnames=("size", "block", "replicates"))
def moving_block_indices(key, size: int, block: int, replicates: int):
    """Independent bootstrap resamples of overlapping, non-circular moving blocks.

    Resample consecutive windows, rather than individual horizons that share days. The
    block length is a modelling choice exposed in the request and echoed in the result.
    """
    count = (size + block - 1) // block
    starts = jax.random.randint(key, (replicates, count), 0, size - block + 1, dtype=jnp.int32)
    return (starts[:, :, None] + jnp.arange(block, dtype=jnp.int32)).reshape(replicates, -1)[:, :size]
