"""
The scenario axis across the devices of a run (I-61; docs/planning/details/precision.md §11.3).

A run's scenario draws (the simulation's Sobol normals, market risk's shifts) are placed on a
one-axis `Mesh` of devices, split along the scenario axis (`shard_scenarios`). Everything after
them is per scenario until a reduction, so XLA's sharding propagation keeps the split through
the path evolution, the scenario market, the pricing and the stored cube, and gathers where a
reduction over scenarios (exposure, VaR/ES) needs every scenario. Nothing else changes: no
kernel names a device, and the values on each device are the one-device run's up to the
rounding that a smaller batch shape moves (about an ulp at float64, measured on the shared
portfolio).

**The devices.** Every device of every host the run's process sees (`jax.devices()`): on one
host its own, on a pod slice every host's, each host's worker running the same job
(`engine.api.hosts`). The count is the largest that divides the scenario count and, across
several hosts, takes the same number of devices from each (`scenario_device_count`), at most
`JAX_RISK_SCENARIO_DEVICES` if set (read at each call; `1` turns sharding off). On one device the
draws are returned as they are, so a one-device run is bit for bit what it was before sharding
existed; on several hosts with no such count every host runs the whole job on one device of its
own, the same numbers redundantly. A run on fewer devices than it has says so in its warnings
(`unused_devices_warning`).

**Reading an array.** An array split across hosts is not whole on any host: `host_array` reads
it whole on every host, gathering it from the others (a collective every host calls at the same
point, as every host runs the same job), and `whole_on_every_host` a whole result (the run
request's, `engine.run.run`, before its document is built).
"""
import dataclasses
import functools
import os
from typing import Optional

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

#: Caps the devices a run's scenario axis is split across (default: every device).
SCENARIO_DEVICES_ENV = "JAX_RISK_SCENARIO_DEVICES"
#: The mesh's one axis.
SCENARIO_AXIS = "scenarios"


def _device_limit() -> int:
    """The devices a run may split its scenarios across: every device of every host, capped by
    `JAX_RISK_SCENARIO_DEVICES`."""
    limit = jax.process_count() * len(jax.local_devices())
    configured = os.environ.get(SCENARIO_DEVICES_ENV)
    if configured:
        requested = int(configured)
        if requested < 1:
            raise ValueError(f"{SCENARIO_DEVICES_ENV} must be a positive device count, not {configured!r}")
        limit = min(limit, requested)
    return limit


def scenario_device_count(num_scenarios: int) -> int:
    """The devices `num_scenarios` scenarios are split across: the largest count, at most the
    run's devices (`_device_limit`), that divides `num_scenarios` and, on several hosts, is a
    multiple of their count, so every device holds the same number of scenarios and every host
    the same number of devices; 1 where none is (every host on one device of its own)."""
    hosts = jax.process_count()
    for count in range(min(_device_limit(), max(num_scenarios, 1)), 1, -1):
        if num_scenarios % count == 0 and count % hosts == 0:
            return count
    return 1


def unused_devices_warning(num_scenarios: int, what: str = "scenarios") -> Optional[str]:
    """The warning of a run of `num_scenarios` that splits them across fewer devices than it
    may (`scenario_device_count`), or None."""
    limit, used = _device_limit(), scenario_device_count(num_scenarios)
    if used >= limit:
        return None
    hosts = jax.process_count()
    across = f" and is a multiple of the {hosts} hosts" if hosts > 1 else ""
    return (f"the {num_scenarios} {what} ran on {used} of the run's {limit} devices: no larger device count divides "
            f"them{across}; a count with more divisors (a power of two) uses every device (I-61)")


def shard_scenarios(array: jax.Array, axis: int) -> jax.Array:
    """`array` split along its scenario `axis` across `scenario_device_count` devices; on one
    device, `array` itself."""
    count = scenario_device_count(array.shape[axis])
    if count == 1:
        return array
    spec = [None] * array.ndim
    spec[axis] = SCENARIO_AXIS
    return jax.device_put(array, NamedSharding(_mesh(count), PartitionSpec(*spec)))


def host_array(array) -> np.ndarray:
    """`array` whole on the host: `np.asarray`, or, for an array split across hosts, gathered
    from every host (`multihost_utils.process_allgather`, a collective of every host)."""
    if getattr(array, "is_fully_addressable", True):
        return np.asarray(array)
    from jax.experimental import multihost_utils

    return np.asarray(multihost_utils.process_allgather(array, tiled=True))


@functools.lru_cache(maxsize=None)
def _mesh(count: int) -> Mesh:
    """A one-axis mesh of `count` devices, the first `count / hosts` of each host's, host by host;
    built once per count."""
    per_host = count // jax.process_count()
    devices = [d for host in range(jax.process_count()) for d in jax.local_devices(process_index=host)[:per_host]]
    return Mesh(np.array(devices), (SCENARIO_AXIS,))


def whole_on_every_host(value):
    """`value` (a result: dataclasses, mappings and sequences of arrays) with every array split
    across hosts read whole (`host_array`), on every host; on one host, `value` itself. Every host
    calls it at the same point, with the same structure (each runs the same job)."""
    if jax.process_count() == 1:
        return value
    if isinstance(value, jax.Array):
        return value if value.is_fully_addressable else host_array(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(value, **{f.name: whole_on_every_host(getattr(value, f.name))
                                             for f in dataclasses.fields(value) if f.init})
    if isinstance(value, dict):
        return type(value)((k, whole_on_every_host(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)) and not hasattr(value, "_fields"):
        return type(value)(whole_on_every_host(v) for v in value)
    if isinstance(value, tuple):   # a NamedTuple
        return type(value)(*(whole_on_every_host(v) for v in value))
    return value
