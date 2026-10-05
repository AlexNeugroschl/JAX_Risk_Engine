"""
The scenario axis across the devices this process owns (roadmap 3.8, I-61;
docs/planning/details/precision.md §11.3).

A run's scenario draws (the simulation's Sobol normals, market risk's shifts) are placed on a
one-axis `Mesh` of the host's devices, split along the scenario axis
(`shard_scenarios`). Everything after them is per scenario until a reduction, so XLA's
sharding propagation keeps the split through the path evolution, the scenario market, the
pricing and the stored cube, and gathers where a reduction over scenarios (exposure, VaR/ES)
needs every scenario. Nothing else changes: no kernel names a device, and the values on each
device are the one-device run's up to the rounding that a smaller batch shape moves (about an
ulp at float64, measured on the shared portfolio).

The count of devices is the largest one that divides the scenario count, at most the local
devices JAX sees, or `JAX_RISK_SCENARIO_DEVICES` if that is lower (read at each call; `1`
turns sharding off). On one device the draws are returned as they are, so a one-device run
is bit for bit what it was before sharding existed.

One host only: on a pod slice every host must run the same job (§11.3), which is still to be
built; `jax.local_devices()` keeps a process to its own host meanwhile.
"""
import functools
import os

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

#: Caps the devices a run's scenario axis is split across (default: every local device).
SCENARIO_DEVICES_ENV = "JAX_RISK_SCENARIO_DEVICES"
#: The mesh's one axis.
SCENARIO_AXIS = "scenarios"


def scenario_device_count(num_scenarios: int) -> int:
    """The devices `num_scenarios` scenarios are split across: the largest count, at most the
    local devices (and `JAX_RISK_SCENARIO_DEVICES`), that divides `num_scenarios`, so every
    device holds the same number of scenarios."""
    limit = len(jax.local_devices())
    configured = os.environ.get(SCENARIO_DEVICES_ENV)
    if configured:
        requested = int(configured)
        if requested < 1:
            raise ValueError(f"{SCENARIO_DEVICES_ENV} must be a positive device count, not {configured!r}")
        limit = min(limit, requested)
    return next(n for n in range(min(limit, max(num_scenarios, 1)), 0, -1) if num_scenarios % n == 0)


def shard_scenarios(array: jax.Array, axis: int) -> jax.Array:
    """`array` split along its scenario `axis` across `scenario_device_count` devices; on one
    device, `array` itself."""
    count = scenario_device_count(array.shape[axis])
    if count == 1:
        return array
    spec = [None] * array.ndim
    spec[axis] = SCENARIO_AXIS
    return jax.device_put(array, NamedSharding(_mesh(count), PartitionSpec(*spec)))


@functools.lru_cache(maxsize=None)
def _mesh(count: int) -> Mesh:
    """The first `count` local devices as a one-axis mesh, built once per count."""
    return Mesh(np.array(jax.local_devices()[:count]), (SCENARIO_AXIS,))
