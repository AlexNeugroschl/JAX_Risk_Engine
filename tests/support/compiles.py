"""
Counting XLA compilations (tests/test_profiling_and_jit.py, tests/test_precision_report.py):
patches `jax._src.compiler.backend_compile_and_load`, the function an xprof trace records as
XLA compilation; cache hits do not call it.
"""
import collections
from contextlib import contextmanager

import jax


@contextmanager
def count_compiles():
    """Count XLA compilations in the block, as a `Counter` keyed by the program's MLIR
    `sym_name` (`jit_<fn>`, the name xprof shows). A hit in JAX's in-memory caches is not
    counted. The persistent (on-disk) cache the suite keeps (`tests/conftest.py`) is off
    inside the block: a program read back from disk would otherwise look like a cache hit."""
    import jax._src.compilation_cache as _compilation_cache
    import jax._src.compiler as _compiler

    counter: collections.Counter = collections.Counter()
    original = _compiler.backend_compile_and_load
    persistent = jax.config.jax_enable_compilation_cache
    jax.config.update("jax_enable_compilation_cache", False)
    _compilation_cache.reset_cache()

    def counting(backend, module, *args, **kwargs):
        name = "?"
        try:
            name = module.operation.attributes["sym_name"].value
        except Exception:
            pass
        counter[name] += 1
        return original(backend, module, *args, **kwargs)

    _compiler.backend_compile_and_load = counting
    try:
        yield counter
    finally:
        _compiler.backend_compile_and_load = original
        jax.config.update("jax_enable_compilation_cache", persistent)
        _compilation_cache.reset_cache()
