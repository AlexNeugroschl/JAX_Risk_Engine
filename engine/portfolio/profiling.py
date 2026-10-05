"""
Phase labels for profiler traces.

The engine worker traces with `python_tracer_level=0`, so trace events carry no Python source
location and a JAX dispatch cannot be attributed to calibration, pricing or a Greek. `phase`
labels regions two ways, since each covers what the other misses:

  - `jax.profiler.TraceAnnotation`: a host-timeline region, which captures eager dispatch
    and XLA compilation (most of this engine's cost).
  - `jax.named_scope`: names baked into ops traced inside it (HLO attribution); it labels
    nothing for already-compiled eager calls.

Both are cheap and inert without an active trace, so they stay in the pricing path. See
docs/concepts/profiling.md.
"""
from contextlib import contextmanager


@contextmanager
def phase(name: str):
    """Label a region of the pricing path as `name` in a profiler trace (both mechanisms;
    see the module docstring). JAX is imported here, not at module scope, so the engine
    worker's profiler hook imports nothing when `JAX_RISK_PROFILE_DIR` is unset."""
    import jax

    with jax.profiler.TraceAnnotation(name):
        with jax.named_scope(name):
            yield
