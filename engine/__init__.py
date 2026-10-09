"""
The JAX risk engine.

`jax_enable_x64` is on for every engine run: each stage takes its dtype explicitly (float64 by
default, float32 where the run configuration asks for it), which needs 64-bit types to exist.
Until 2026-10-01 this was a side effect of importing the Hull-White simulation module, which
`engine.market_data.market` imported; it is set here, for the whole package, now that the module is gone.

This module does not import JAX itself, so `engine.traderx` stays free of it (I-05): it
sets `JAX_ENABLE_X64`, which JAX reads when it is first imported, or updates the flag when JAX
is already loaded. The engine worker gets it the same way, by importing `engine` (I-71).

It is the one process-wide setting importing `engine` makes (decision A-22). The
others belong to the process that owns them:

- **Matrix-product precision** is stated by each product (`engine.precision.matmul`), from its
  operands' compute format, so `jax_default_matmul_precision` is neither set nor read.
- **Deterministic GPU kernels** (`--xla_gpu_exclude_nondeterministic_ops=true` in `XLA_FLAGS`)
  are set by the engine worker for itself (`engine.api.worker.deterministic_kernels_environment`)
  and by the test suite; a library user who wants reproducible GPU bits sets the flag.
- **GPU preallocation** stays JAX's default (75% of a GPU per process) unless the environment
  says otherwise; the demos that start a server and the test suite turn it off, since their
  processes share one GPU.
"""
import os as _os
import sys as _sys

if "jax" in _sys.modules:
    _sys.modules["jax"].config.update("jax_enable_x64", True)
else:
    _os.environ["JAX_ENABLE_X64"] = "1"
