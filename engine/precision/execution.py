"""Execution requirements, separate from numerical formats and dispatch labels.

Checks run before pricing/tracing. They never select another backend, change JAX's
process settings, or silently replace a format. A native dispatch is still not an
instruction-level qualification of every compiled shape.
"""
from dataclasses import dataclass, fields
from functools import lru_cache
from types import MappingProxyType, SimpleNamespace
from collections.abc import Mapping

import jax


EXECUTION_MODES = ("native", "allow_emulation")
BACKENDS = ("default", "cpu", "gpu", "tpu")


@dataclass(frozen=True)
class ValidatedExecution:
    """One job's immutable execution context, passed only to internal implementations.

    Created by the request's validation on the executing worker. Public standalone
    entry points create their own context. No validation success is cached globally.
    """
    policy: object
    device: object
    stages: tuple = ()
    internal_roles: tuple = ()

    def report_policy(self):
        """A fresh JSON-ready copy; runtime device/array observations are never cached."""
        result = report_policy(self.policy, self.device)
        result['active_stages'] = list(self.stages)
        result['active_internal_roles'] = list(self.internal_roles)
        selected = {*self.stages, *(f'internal.{name}' for name in self.internal_roles)}
        result['placement'] = {k: v for k, v in result['placement'].items() if k in selected}
        return result


def prepare_execution(policy, *, stages=(), trades=None, arrays=(), internal_roles=None):
    """Called once at the end of structural request validation, before numerical work."""
    stages = tuple(stages)
    roles = tuple(f.name for f in fields(policy.internal)) if internal_roles is None else tuple(internal_roles)
    device = policy.validate_execution(stages=stages, trades=trades, arrays=arrays, internal_roles=roles)
    return ValidatedExecution(policy, device, stages, roles)


def report_policy(policy, device):
    """Static metadata only: this cache cannot authorize work or bypass hardware checks."""
    from dataclasses import asdict
    from engine.precision.runtime import capture_compiler_runtime
    result = _plain(_resolved_metadata(policy, device.platform))
    result['compiler_runtime'] = asdict(capture_compiler_runtime())
    return result


@lru_cache(maxsize=128)
def _resolved_metadata(policy, backend):
    # resolved() uses only the platform for placement; no capability or runtime
    # flag checks are performed/cached here. Frozen values are safe across jobs.
    return _freeze(policy.resolved(SimpleNamespace(platform=backend)))


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class InternalPrecision:
    """Arithmetic of the non-scenario kernels; host input parsing is reported separately."""
    calibration: str = "float64"
    levels: str = "float64"
    initial_values: str = "float64"
    reductions: str = "float64"
    greeks: str = "float64"
    output: str = "float64"

    def __post_init__(self):
        for item in fields(self):
            if getattr(self, item.name) not in ("float32", "float64"):
                raise ValueError(f"InternalPrecision.{item.name}: choose float32 or float64")

    @classmethod
    def throughout(cls, name):
        return cls(**{item.name: name for item in fields(cls)})

    def dtype(self, role):
        return jax.numpy.dtype(getattr(self, role))


@dataclass(frozen=True)
class ExecutionPolicy:
    """Required backend and permission to use software product emulation.

    `default` resolves to the active JAX default device, and is recorded as a concrete
    backend in the report. A named backend must already be selected by the caller;
    a mismatch is an error, never a request to copy work to another device.
    `native` refuses engine-selected emulation. Qualification is report metadata.
    `allow_emulation` permits software routes; it is not a correctness certificate.
    TPU FP64 is refused in either mode because the engine's IEEE FP64 helpers are
    incompatible with TPU's software representation.
    """
    backend: str = "default"
    mode: str = "native"
    gpu_gemm: str = "inherit"

    def __post_init__(self):
        if self.gpu_gemm not in ("inherit", "default", "cublas"):
            raise ValueError("ExecutionPolicy.gpu_gemm must be 'inherit', 'default' or 'cublas'")
        if self.backend not in BACKENDS:
            raise ValueError(f"ExecutionPolicy.backend must be one of {BACKENDS}; got {self.backend!r}")
        if self.mode not in EXECUTION_MODES:
            raise ValueError(f"ExecutionPolicy.mode must be one of {EXECUTION_MODES}; got {self.mode!r}")

    def validate(self, formats=(), *, device=None):
        from engine.precision.products import default_device

        device = device or default_device()
        from engine.precision.runtime import validate_compiler_route
        validate_compiler_route(self.gpu_gemm, device.platform)
        if self.backend != "default" and self.backend != device.platform:
            raise ValueError(f"execution.backend={self.backend!r} requires that backend, but the active "
                             f"device is {device.platform}; select the backend before running (no automatic fallback)")
        for role, name in formats:
            validate_format(name, role, device)
        return device


def validate_format(name, role, device):
    if name == "float64":
        if not jax.config.x64_enabled:
            raise ValueError(f"{role}=float64 requires jax_enable_x64=True; implicit FP32 demotion is refused")
        if device.platform == "tpu":
            raise ValueError(f"{role}=float64 is unsupported on TPU: no native FP64 and the engine's IEEE FP64 "
                             "helpers are incompatible with TPU software FP64; no FP32 downgrade or CPU fallback")


def validate_product(compute, accumulate, mode="native", *, device=None, options=None):
    """Check the requested arithmetic before tracing a dot, including its accumulator."""
    from engine.precision.formats import format_of
    from engine.precision.products import default_device, product_execution

    if mode not in EXECUTION_MODES:
        raise ValueError(f"execution mode must be one of {EXECUTION_MODES}; got {mode!r}")
    device = device or default_device()
    from engine.precision.options import ProductOptions
    options = options or ProductOptions()
    options.validate(compute)
    validate_format(compute, "compute", device)
    validate_format(accumulate, "accumulate", device)
    dispatch = product_execution(compute, device, options=options)
    if options.subnormals != "native" or options.algorithm != "format":
        if device.platform != "gpu" or dispatch != "native":
            raise ValueError("native FP16 normalization and explicit FP8 algorithms require a capable GPU; no fallback")
    if mode == "native":
        if dispatch != "native":
            raise ValueError(f"compute={compute!r} requires native execution on {device.platform}; "
                             "this path is emulated. Explicitly use mode='allow_emulation' for research")


def validate_arrays(arrays, device):
    """Refuse pre-existing JAX arrays on another platform, before an implicit transfer."""
    for array in jax.tree_util.tree_leaves(arrays):
        if isinstance(array, jax.Array):
            platforms = {d.platform for d in array.devices()}
            if platforms != {device.platform}:
                raise ValueError(f"input arrays are on {sorted(platforms)}, execution is on {device.platform}; "
                                 "explicitly place inputs on the requested backend before running")
