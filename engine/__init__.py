"""
The JAX risk engine.

`jax_enable_x64` is on for every engine run: each stage takes its dtype explicitly (float64 by
default, float32 where the run configuration asks for it), which needs 64-bit types to exist.
Until roadmap 1.3 this was a side effect of importing the Hull-White simulation module, which
`engine.market` imported; it is set here, for the whole package, now that the module is gone.

This module does not import JAX itself, so `engine.integration` stays free of it (I-05): it
sets `JAX_ENABLE_X64`, which JAX reads when it is first imported, or updates the flag when JAX
is already loaded. Worker processes switch it on again in their initializer (I-71).
"""
import os as _os
import sys as _sys

if "jax" in _sys.modules:
    _sys.modules["jax"].config.update("jax_enable_x64", True)
else:
    _os.environ["JAX_ENABLE_X64"] = "1"
