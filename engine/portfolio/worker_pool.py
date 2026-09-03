"""
Per-precision-tier worker-pool dispatch for `engine.portfolio.price_portfolio`
-- the multi-process replacement for "one process, `_PRICING_LOCK`,
everything time-sliced" as the *primary* concurrency mechanism (the lock
itself stays exactly as-is, one layer down in `engine.portfolio.request`, as
narrower defense-in-depth -- see that module's docstring).

**Why multi-process, not multi-thread or single-process device sharding.**
`jax_enable_x64` is a process-global JAX/XLA flag with no per-thread,
per-device, or per-mesh scoped alternative (confirmed at the JAX source
level: `jax/_src/config.py`'s own maintainers explicitly excluded this one
flag from the context-manager-scoping mechanism every other JAX config flag
gets). Two *threads* in one process wanting different precisions at the same
time cannot both be correct without serializing -- that's exactly what
`_PRICING_LOCK` does today. Two *processes*, each fixing its own
`jax_enable_x64` once at boot and never touching it again, have entirely
independent JAX/XLA runtimes and never race each other. That's the model
here: one `ProcessPoolExecutor` per precision tier (float32 / float64),
sized to `N` workers each. `N` float32-tier workers running concurrently
with `M` float64-tier workers gives genuine cross-precision, cross-device
concurrency; within a single tier's pool, jobs beyond that pool's worker
count still queue (expected pool exhaustion, not a bug -- see
`docs/reference/http-api.md`).

**Routing.** `PrecisionConfig` has three independent knobs (`simulation`,
`pricing`, `risk`); only `simulation` selects the worker pool, since it's the
one that actually drives `generate_paths`'s own `jax.config.update` call --
`pricing`/`risk` stay independently-settable dtypes *within* a job, honored
by `price_portfolio`'s own explicit casts once inside a worker of either
tier (see that module's re-enable-x64-after-generate_paths comment). This
module does not change `PrecisionConfig`'s/`PortfolioRequest`'s shape at
all -- `simulation` already is the routing selector.

**Backend-agnostic by construction, optimized for TPU.** On this CPU-only
dev machine there is exactly one `CpuDevice`, so every worker (of both
tiers) ends up sharing that one device -- device pinning below degenerates
to a no-op. The property this module proves on this machine is process
isolation of `jax_enable_x64` plus genuine overlapping wall-clock execution,
not device-level parallelism specifically (no CPU affinity is pinned).
Real deployment is a Cloud TPU VM, standardly single-host-multi-chip (e.g. a
v4-8 exposes 4 chips / 8 cores to one host process tree) -- see
`_worker_init`'s docstring for the environment-variable pinning shape this
leaves ready for that, deliberately not implemented against hardware that
cannot be tested here. Pool sizes below (`_DEFAULT_POOL_SIZE`) are a small,
arbitrary CPU-dev-machine default; real deployment would size each tier's
pool to (a share of) `len(jax.devices())` on the actual host.

**Windows note (also true, deliberately, on Linux/TPU).**
`ProcessPoolExecutor` on Windows always uses spawn, never fork -- every
worker process re-imports `engine.portfolio`/`jax`/`ORE` fresh from a clean
interpreter, which is exactly the "set `jax_enable_x64` once at worker boot,
before any pricing work runs" model this design already requires. Real TPU
deployment (Linux) could use fork in principle but *should* still use
spawn/forkserver deliberately -- forking a process that has already
initialized a JAX/XLA client is a known source of hangs per JAX's own
documentation. `_worker_init` below is a plain top-level, module-level
function (not a lambda/closure) specifically because spawn pickles the
`initializer` callable by reference -- a closure or lambda would fail to
pickle (or worse, silently pickle the wrong thing) when Windows spawns the
worker.

**Why request/result cross the process boundary via a thin JSON-ish
translation, not the raw dataclasses.** `PortfolioRequest.trades` entries
each carry a real `ORE.Date` (`evaluation_date`), and `SwaptionConfig`
additionally carries an `ORE.Period` (`forward_start`) -- both SWIG-bound
objects, confirmed NOT picklable (`TypeError: cannot pickle 'SwigPyObject'
object`), which `ProcessPoolExecutor.submit` requires for anything crossing
the process boundary. `engine/api/schemas.py` already solves exactly this
problem at the HTTP boundary (ISO date strings, ORE period strings); the
helpers below (`_freeze_trade`/`_thaw_trade`) reuse that identical
round-trip (`ORE.Date.ISO()`/`ORE.DateParser.parseISO`,
`str(ORE.Period)`/`ORE.Period(str)`) to make a `PortfolioRequest` picklable
for submission, and thaw it back to real ORE objects inside the worker
before calling `price_portfolio`. `PortfolioResult` (JAX/numpy arrays, plain
floats/dicts, no ORE types) pickles as-is with no translation needed for the
return trip -- confirmed directly.
"""
import os
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import replace
from typing import Optional

from engine.portfolio.request import PortfolioRequest, PortfolioResult

# A small, arbitrary default for this CPU-only dev machine -- real TPU
# deployment would size each tier's pool to (a share of) len(jax.devices())
# on the actual host (see module docstring's "Backend-agnostic" section).
# Kept small deliberately: each worker is a full Python + JAX/XLA process
# (spawn, not fork -- see module docstring), so pool startup itself has real
# wall-clock cost on a dev machine.
_DEFAULT_POOL_SIZE = 2

_POOLS: dict = {}  # precision_bits (32 or 64) -> ProcessPoolExecutor


def _worker_init(precision_bits: int) -> None:
    """`ProcessPoolExecutor(initializer=...)` target -- runs exactly ONCE,
    at worker-process boot, before this worker ever accepts a job, and never
    again for the worker's lifetime. Must stay a plain top-level function
    (see module docstring's Windows note): spawn pickles this by reference,
    so a closure/lambda can't be used here.

    Two things, deliberately in this order and never repeated:
    1. `jax.config.update("jax_enable_x64", ...)`, matching this worker's
       own fixed tier. Every job this worker ever processes shares this one
       setting -- that's what makes `_PRICING_LOCK` unnecessary *at this
       level*: a worker processes jobs strictly sequentially (a
       `ProcessPoolExecutor` worker has exactly one job in flight at a
       time, by construction), so no second thread in this process can ever
       flip the flag mid-job.
    2. Device pinning via process-launch-time environment configuration --
       e.g. `JAX_PLATFORMS`, or a `CUDA_VISIBLE_DEVICES`-equivalent for TPU
       (`TPU_VISIBLE_CHIPS`) -- deliberately set via `os.environ` BEFORE
       `import jax` runs in this (freshly spawned) process, not via
       `jax.device_put` after the fact: device visibility is most reliably
       controlled at process-launch time, and one process seeing one device
       removes any placement ambiguity. On this CPU-only dev machine there
       is exactly one `CpuDevice`, so this is a deliberate no-op (every
       worker of both tiers ends up sharing that one device) -- the shape
       is left ready for real TPU env-var wiring rather than implemented
       against hardware this repo cannot currently test against (see
       `docs/planning/roadmap-and-history.md`'s deferred-to-TPU-deployment
       list).
    """
    # Deliberately no-op today (single CpuDevice) -- real deployment would
    # set e.g. os.environ["TPU_VISIBLE_CHIPS"] = str(chip_index) here, BEFORE
    # `import jax` below, keyed off this worker's own index within its pool.
    os.environ.setdefault("JAX_PLATFORMS", "")

    import jax  # local import: must happen AFTER any device-pinning env vars above

    jax.config.update("jax_enable_x64", precision_bits == 64)


def _freeze_trade(cfg):
    """`ORE.Date`/`ORE.Period` -> plain `str`, so a trade config pickles
    across the process boundary. Generic over all four trade-config types
    via `dataclasses.replace` -- no per-type reconstruction needed, since
    every trade config already exposes `evaluation_date` and only
    `SwaptionConfig` additionally carries `forward_start`."""
    updates = {"evaluation_date": cfg.evaluation_date.ISO()}
    if hasattr(cfg, "forward_start"):
        updates["forward_start"] = str(cfg.forward_start)
    return replace(cfg, **updates)


def _thaw_trade(cfg):
    """Inverse of `_freeze_trade` -- runs inside the worker process, right
    before `price_portfolio` is called, turning the ISO/period strings back
    into real `ORE.Date`/`ORE.Period` objects."""
    import ORE

    updates = {"evaluation_date": ORE.DateParser.parseISO(cfg.evaluation_date)}
    if hasattr(cfg, "forward_start"):
        updates["forward_start"] = ORE.Period(cfg.forward_start)
    return replace(cfg, **updates)


def _run_pricing_job(frozen_request: PortfolioRequest) -> PortfolioResult:
    """Runs inside the worker process (submitted to the pool, not called
    directly). `_worker_init` has already set this worker's fixed
    `jax_enable_x64` tier by the time this ever runs. Thaws the frozen
    (string-dated) trades back into real ORE objects, then delegates to
    `price_portfolio` unchanged -- this module does not reimplement or
    wrap any pricing logic.

    Opt-in profiling: if `JAX_RISK_PROFILE_DIR` is set in this process's
    environment, the `price_portfolio` call is wrapped in `jax.profiler.trace`,
    writing an XProf/TensorBoard-profiler trace into
    `$JAX_RISK_PROFILE_DIR/pid-<pid>/` (one subdir per process, so it's safe
    whether this is called directly in one process or fans out across pool
    workers -- each writes its own). This traces the job as it actually runs
    in a fresh worker: XLA lowering/compilation included, not just steady-state
    execution. That is deliberate -- for this engine the compilation cost is a
    first-class thing to measure (the tree pricers and the calibration
    bisection lower a large number of `jit` programs), and larger portfolios
    are where the execution-vs-compilation ratio gets meaningful. View with
    `xprof --port 8791 <dir>`; Tools -> `op_profile` / `framework_op_stats`
    give the aggregate on-device-vs-on-host breakdown, `trace_viewer` the
    timeline. Unset (the default, including every test and the HTTP path in
    CI) -> completely inert: no `import jax` here, byte-identical to before
    this hook existed."""
    from engine.portfolio.request import price_portfolio

    def _run() -> PortfolioResult:
        trades = [_thaw_trade(cfg) for cfg in frozen_request.trades]
        request = replace(frozen_request, trades=trades)
        return price_portfolio(request)

    profile_dir = os.environ.get("JAX_RISK_PROFILE_DIR")
    if not profile_dir:
        return _run()

    import jax  # local: keep jax out of module import, mirroring _worker_init

    out_dir = os.path.join(profile_dir, f"pid-{os.getpid()}")
    with jax.profiler.trace(out_dir):
        result = _run()
        # The trace must not end before device execution does, or the timeline
        # is truncated (JAX profiling docs are explicit about this). npv_cube
        # is the dominant device-side tail; base_npv is already a plain Python
        # float by the time price_portfolio returns.
        jax.block_until_ready(result.npv_cube)
    return result


def _pool_for(precision_bits: int, pool_size: int = _DEFAULT_POOL_SIZE) -> ProcessPoolExecutor:
    """Lazily creates (once) and returns the `ProcessPoolExecutor` for the
    given tier (32 or 64). One pool per tier, created on first use and
    reused for the life of this process -- workers are never re-spawned
    per-job, only per-pool-lifetime (see `_worker_init`'s docstring)."""
    if precision_bits not in (32, 64):
        raise ValueError(f"precision_bits must be 32 or 64, got {precision_bits!r}")
    pool = _POOLS.get(precision_bits)
    if pool is None:
        pool = ProcessPoolExecutor(
            max_workers=pool_size,
            initializer=_worker_init,
            initargs=(precision_bits,),
        )
        _POOLS[precision_bits] = pool
    return pool


def submit_pricing_job(request: PortfolioRequest, pool_size: int = _DEFAULT_POOL_SIZE) -> "Future[PortfolioResult]":
    """Routes `request` to the worker pool matching
    `request.precision.simulation` (32 -> the float32-tier pool, 64 -> the
    float64-tier pool -- see module docstring's "Routing" section) and
    returns a `concurrent.futures.Future[PortfolioResult]` immediately;
    the actual `price_portfolio` call runs in a separate OS process,
    strictly sequentially with respect to any other job that same worker
    later picks up, but genuinely concurrently with jobs running in other
    workers/pools.

    `pool_size` only takes effect the first time a given tier's pool is
    created in this process (`ProcessPoolExecutor`'s worker count is fixed
    at pool construction) -- later calls with a different `pool_size` for a
    tier that's already running reuse the existing pool unchanged.
    """
    pool = _pool_for(request.precision.simulation, pool_size=pool_size)
    frozen_trades = [_freeze_trade(cfg) for cfg in request.trades]
    frozen_request = replace(request, trades=frozen_trades)
    return pool.submit(_run_pricing_job, frozen_request)


def shutdown_pools(wait: bool = True) -> None:
    """Shuts down every tier's pool this process has created so far (a
    no-op for any tier never used). Mainly for tests/interpreter shutdown --
    a long-running dispatcher process (e.g. `engine/api/routes.py`'s FastAPI
    app) is expected to keep its pools alive for its own lifetime."""
    for precision_bits in list(_POOLS.keys()):
        pool = _POOLS.pop(precision_bits)
        pool.shutdown(wait=wait)
