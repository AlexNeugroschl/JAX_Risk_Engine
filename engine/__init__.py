"""
The JAX risk engine.

`jax_enable_x64` is on for every engine run: each stage takes its dtype explicitly (float64 by
default, float32 where the run configuration asks for it), which needs 64-bit types to exist.
Until roadmap 1.3 this was a side effect of importing the Hull-White simulation module, which
`engine.market` imported; it is set here, for the whole package, now that the module is gone.

This module does not import JAX itself, so `engine.integration` stays free of it (I-05): it
sets `JAX_ENABLE_X64`, which JAX reads when it is first imported, or updates the flag when JAX
is already loaded. Worker processes switch it on again in their initializer (I-71).

Three more defaults hold on accelerators (roadmap 2.2), each unless the environment or the
process already sets it, and none changes a CPU number:

- **No GPU preallocation** (`XLA_PYTHON_CLIENT_PREALLOCATE=false`). By default XLA takes 75%
  of a GPU's memory in every process that opens it, and an engine host runs several (the
  engine worker, test processes, a notebook). Each process holds what it has used instead.
- **Matrix products at their operands' precision** (`jax_default_matmul_precision` "highest").
  By default a float32 product runs in TensorFloat-32 on an NVIDIA GPU (a 10-bit mantissa) and
  in bfloat16 passes on a TPU, so a run whose policy computes in float32 would compute below
  it. A lower product precision is a compute format of its own (roadmap 3.7, 5.2), chosen by
  the policy, never by the device.
- **Deterministic GPU kernels** (`--xla_gpu_exclude_nondeterministic_ops=true` in `XLA_FLAGS`).
  A compiled program gives the same bits on every run: without it the AD Greeks' scatter-adds
  accumulate in whatever order the GPU's atomics land, and move by an ulp between runs.
  Compilation may still autotune, so two compiles of one program could in principle choose
  different kernels; a restarted worker reads its programs back from the disk cache, and
  `--xla_gpu_deterministic_ops=true` pins compilation too, at twice the compile time
  (measured on the demo's job, roadmap 2.2).

JAX reads the device variables when the process first opens a device, so they apply to any
process that imports `engine` before its first device operation.
"""
import os as _os
import sys as _sys

_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_XLA_FLAGS = _os.environ.get("XLA_FLAGS", "")
if "deterministic_ops" not in _XLA_FLAGS:  # either XLA determinism flag set by the operator wins
    _os.environ["XLA_FLAGS"] = f"{_XLA_FLAGS} --xla_gpu_exclude_nondeterministic_ops=true".strip()

if "jax" in _sys.modules:
    _jax = _sys.modules["jax"]
    _jax.config.update("jax_enable_x64", True)
    if _jax.config.jax_default_matmul_precision is None:
        _jax.config.update("jax_default_matmul_precision", "highest")
else:
    _os.environ["JAX_ENABLE_X64"] = "1"
    _os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
