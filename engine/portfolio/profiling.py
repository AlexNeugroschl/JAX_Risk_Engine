"""
Phase labelling for profiler traces -- the one piece that makes an xprof
timeline of this engine readable.

**The problem this solves.** `engine.portfolio.worker_pool` runs its trace
with `python_tracer_level=0` (see `_profile_options`' own docstring for the
measured 97.1%-of-events / silent-1.6s-truncation reasoning that forced
that). The host tracer still records everything JAX and XLA do, so
compile-vs-dispatch-vs-execute stays fully separable -- but **no event in
the trace carries a Python source file or line** (confirmed directly: 0 of
926,463 events had `source_file`/`source_line`/`long_name` args). A
`PjitFunction(scan)` event names the JAX primitive, never the engine
function that issued it, so "is this dispatch calibration, or pricing, or a
Greek?" is unanswerable from the raw trace.

**Two mechanisms, deliberately both.** They label different things and
neither one alone is sufficient:

  - `jax.profiler.TraceAnnotation` is a HOST-side region. It wraps the
    wall-clock interval on the Python thread, so everything dispatched
    inside it -- including XLA compilation, which is host work -- lands
    under that label on the timeline. This is the one that matters for this
    engine, whose cost is dominated by compile and dispatch.
  - `jax.named_scope` labels ops as they are TRACED into a jaxpr, so the
    name is baked into the compiled HLO and shows up in `op_profile` /
    `hlo_stats` attribution. It only affects work traced inside the scope;
    an eagerly-dispatched call that was already compiled gets nothing from
    it, which is why `TraceAnnotation` is needed alongside.

Using only `named_scope` was tried first and measured: it produced **zero**
labelled events on the host timeline for every phase whose work was eager
dispatch rather than fresh tracing. Hence `phase()` below, which enters
both.

Cost is negligible (a string push/pop per region, no synchronization), and
when no trace is active both context managers are effectively free -- so
these annotations stay permanently in the pricing path rather than being
conditional on a profiling flag. See `docs/concepts/profiling.md`.
"""
from contextlib import contextmanager


@contextmanager
def phase(name: str):
    """Labels a region of the pricing path as `name` in a profiler trace.

    Enters BOTH `jax.profiler.TraceAnnotation` (host timeline) and
    `jax.named_scope` (compiled-HLO op names) -- see this module's docstring
    for why neither alone is enough.

    `jax` is imported lazily, inside the call, so that importing this module
    never pulls JAX in: `engine.portfolio.worker_pool` maintains a deliberate
    "no `import jax` at module scope" property (the profiler hook must be
    completely inert, down to not importing JAX, when
    `JAX_RISK_PROFILE_DIR` is unset).
    """
    import jax

    with jax.profiler.TraceAnnotation(name):
        with jax.named_scope(name):
            yield
