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
translation, not the raw dataclasses.** Trade configs carry real ORE objects
-- every one an `ORE.Date` (`evaluation_date`), Bermudan and American
swaptions their exercise dates, `SwaptionConfig` an `ORE.Period`
(`forward_start`) -- and SWIG-bound objects are NOT picklable (`TypeError:
cannot pickle 'SwigPyObject' object`), which `ProcessPoolExecutor.submit`
requires for anything crossing the process boundary. `_freeze_trade`
therefore turns each trade into a `_FrozenTrade` record (its class plus its
field values, ORE dates and periods written as ISO/period text exactly as
`engine/api/schemas.py` writes them at the HTTP boundary), and
`_thaw_trade` rebuilds the real config inside the worker before
`price_portfolio` runs. The record is deliberately NOT a half-converted
config: a config's own validation rejects a date field holding text, and it
should. `PortfolioResult` (JAX/numpy arrays, plain
floats/dicts, no ORE types) pickles as-is with no translation needed for the
return trip -- confirmed directly.
"""
import os
import time
import warnings
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, fields, replace
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


@dataclass(frozen=True)
class _OreValue:
    """An `ORE.Date` or `ORE.Period` written as text, so it pickles."""
    kind: str  # "date" | "period"
    text: str


@dataclass(frozen=True)
class _FrozenTrade:
    """A trade config in picklable form: its class and its field values."""
    cls: type
    values: dict


def _freeze_value(value):
    import ORE

    if isinstance(value, ORE.Date):
        return _OreValue("date", value.ISO())
    if isinstance(value, ORE.Period):
        return _OreValue("period", str(value))
    if isinstance(value, (list, tuple)):
        return type(value)(_freeze_value(v) for v in value)
    return value


def _thaw_value(value):
    import ORE

    if isinstance(value, _OreValue):
        return ORE.DateParser.parseISO(value.text) if value.kind == "date" else ORE.Period(value.text)
    if isinstance(value, (list, tuple)):
        return type(value)(_thaw_value(v) for v in value)
    return value


def _freeze_trade(cfg) -> _FrozenTrade:
    """A trade config -> a picklable `_FrozenTrade`: every `ORE.Date`/
    `ORE.Period` field, including one inside a list (a Bermudan's exercise
    dates), becomes text. Generic over every trade-config type -- nothing
    here names a field."""
    return _FrozenTrade(type(cfg), {f.name: _freeze_value(getattr(cfg, f.name)) for f in fields(cfg)})


def _thaw_trade(frozen: _FrozenTrade):
    """Inverse of `_freeze_trade` -- runs inside the worker process, right
    before `price_portfolio` is called, rebuilding the config (and so
    re-running its own validation) from real ORE objects."""
    return frozen.cls(**{name: _thaw_value(value) for name, value in frozen.values.items()})


def _profile_options(jax):
    """`jax.profiler.ProfileOptions` for `_run_pricing_job`'s trace, with
    the Python tracer OFF (see that function's docstring for the measured
    97.1%-of-events / silent-1.6s-truncation reasoning behind overriding
    JAX's own `python_tracer_level=1` default).

    `jax` is passed in rather than imported here so this module keeps its
    "no `import jax` at module scope" property -- the caller has already
    done the local import by the time it needs these options.

    `host_tracer_level` and `enable_hlo_proto` are left at JAX's defaults
    (2 and True): the HLO protos are what give xprof's `op_profile` view
    its per-op attribution, which is the main thing worth opening a trace
    for here, and they are not what made the trace unusable.

    **What the trace contains with `python_tracer_level=0`.** The host
    tracer still instruments everything JAX/XLA does on the host; what is
    dropped is ONLY the CPython interpreter's own call/return events. The
    resulting trace has five lanes, measured on the 4-trade demo portfolio
    (event counts and summed durations; durations exceed wall time because
    the lanes run concurrently and nest):

        /host:CPU (main)             301,105 events   107.2s
        tf_PjRtCompilerThreadPool    237,300 events    81.2s
        tf_xla-cpu-codegen           241,552 events    47.1s
        tf_XLAEigen                  132,769 events     2.9s
        tf_XLAPjRtCpuClient           13,737 events     1.4s

    Classifying that by what the event names mean:

        XLA compilation      ~118.6s  (backend_compile_and_load,
                                       CpuCompiler::RunBackend, Codegen,
                                       FusionCompiler::Compile, MLIR passes)
        pjit dispatch (host)  ~76.3s  (PjitFunction(scan/multiply/_interp/...))
        JAX tracing            ~3.1s  (trace_to_jaxpr_nounits)
        device execute         ~3.0s  (ThunkExecutor::Execute)

    So the compile-vs-execute and dispatch-vs-device questions this hook
    exists to answer are all still answerable -- in fact more clearly than
    before, since the whole job is now covered instead of its first 1.6s.

    **What is genuinely lost, and how it is bought back.** Per-`PjitFunction`
    events name the JAX primitive (`PjitFunction(scan)`,
    `PjitFunction(_interp)`), NOT the engine function that called it: no
    event in the trace carries a Python source file/line (confirmed
    directly -- 0 of 926,463 events have `source_file`/`source_line`/
    `long_name` args). Attributing a dispatch back to, say,
    `_cashflow_values_at_nodes` versus `_lgm_numeraire` is what
    `JAX_RISK_PROFILE_PYTHON_TRACER=1` buys, and the only thing it buys.

    PHASE-level attribution -- which is what one actually wants most of the
    time -- is already provided without it, by
    `engine.portfolio.profiling.phase`: `price_portfolio` wraps each stage
    (calibration / simulation / pricing / base_npv / risk / greeks, plus one
    region per trade inside greeks) in a `jax.profiler.TraceAnnotation` +
    `jax.named_scope` pair, which shows up as named regions on the timeline
    at negligible cost. See that module's docstring for why BOTH mechanisms
    are needed, and `docs/concepts/profiling.md` for what the resulting
    breakdown looks like. Only per-CALLSITE attribution within a phase still
    requires the Python tracer."""
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 1 if os.environ.get("JAX_RISK_PROFILE_PYTHON_TRACER") == "1" else 0
    return options


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
    this hook existed.

    **`python_tracer_level=0`, always** (see `_profile_options`). JAX's own
    default (`python_tracer_level=1`) instruments the CPython interpreter,
    not just this engine's own frames, which for a dispatch-heavy workload
    like this one buries the actual computation: measured on the 4-trade
    demo portfolio, 971,080 of the trace's 1,000,115 events (97.1%) were
    Python-interpreter frames -- `isinstance` x 90,235, `append` x 33,156,
    `len` x 21,073 -- from inside JAX's own dispatch machinery. Worse, the
    profiler's event buffer is a fixed ~1M-event cap with no backpressure
    and no truncation warning, so those interpreter frames saturated it
    during startup and the trace SILENTLY covered only the first 1.6s of a
    ~90s job (confirmed by the captured events' own timestamp span) --
    a partial trace that reports success and looks exactly like a complete
    one. Turning the Python tracer off is what makes the whole job fit:
    467MB -> 50MB, 82.7s -> 45.4s, and 2% -> 93% of the job's wall time
    actually covered by the trace, with `base_npv` identical.

    This does NOT make the trace "JAX-only": the host tracer still records
    all of JAX/XLA's own host-side work, so XLA compilation, pjit dispatch,
    JAX tracing and device execution each stay separately visible and
    separately attributable (see `_profile_options` for the per-lane
    breakdown and the measured compile-vs-dispatch-vs-execute split). What
    is dropped is CPython's own interpreter frames, and with them the
    ability to attribute a dispatch back to the specific ENGINE function
    that issued it. Set `JAX_RISK_PROFILE_PYTHON_TRACER=1` to opt back in
    when that per-callsite attribution is specifically what's needed --
    accepting the ~9x size, the ~2x slowdown, and near-certain silent
    truncation on any job this size or larger.

    **Warmup: `JAX_RISK_PROFILE_WARMUP=1`** (default off) runs the job once
    BEFORE opening the trace and discards it, so the traced run measures
    warm steady-state execution against already-populated compilation
    caches -- the conventional shape for a profiler capture, and the only
    way the timeline reflects execution rather than XLA lowering. Left OFF
    by default precisely because this hook's stated purpose above is to
    measure cold-start compilation too; turn it on when the question is
    "where does the *execution* time go," leave it off when the question is
    "what does this job cost from cold." Note it roughly doubles wall time
    (the job runs twice) and that JAX's compilation cache is process-global,
    so the discarded run's effect is exactly the cache population the
    traced run then benefits from."""
    from engine.portfolio.request import price_portfolio

    def _run() -> PortfolioResult:
        trades = [_thaw_trade(cfg) for cfg in frozen_request.trades]
        request = replace(frozen_request, trades=trades)
        return price_portfolio(request)

    profile_dir = os.environ.get("JAX_RISK_PROFILE_DIR")
    if not profile_dir:
        return _run()

    import jax  # local: keep jax out of module import, mirroring _worker_init

    if os.environ.get("JAX_RISK_PROFILE_WARMUP") == "1":
        # Discarded on purpose -- run once before the trace opens so the
        # traced run below hits warm compilation caches (see docstring).
        # block_until_ready for the same reason as the traced run: the
        # caches aren't fully populated until device execution finishes.
        jax.block_until_ready(_run().npv_cube)

    out_dir = os.path.join(profile_dir, f"pid-{os.getpid()}")
    started = time.time()
    with jax.profiler.trace(out_dir, profiler_options=_profile_options(jax)):
        result = _run()
        # The trace must not end before device execution does, or the timeline
        # is truncated (JAX profiling docs are explicit about this). npv_cube
        # is the dominant device-side tail; base_npv is already a plain Python
        # float by the time price_portfolio returns.
        jax.block_until_ready(result.npv_cube)
    _warn_if_trace_truncated(out_dir, time.time() - started)
    return result


# The profiler's event buffer is a fixed cap with no backpressure: once it
# fills, remaining events are dropped silently and the resulting trace looks
# exactly like a complete one. This is the threshold to start worrying at.
_TRACE_EVENT_CAP = 1_000_000
_TRACE_EVENT_WARN = 950_000
# Fraction of the job's wall time the captured events must span before the
# trace is considered to cover the run. Generous: the profiler legitimately
# starts a moment after, and stops a moment before, the wall-clock window
# measured around it, so a complete trace still falls somewhat short of 1.0.
_TRACE_COVERAGE_WARN = 0.5


def _warn_if_trace_truncated(out_dir: str, wall_seconds: float) -> None:
    """Warns when a just-written trace looks TRUNCATED rather than complete.

    **Why this is not optional bookkeeping.** A trace that overruns the
    profiler's ~1M-event buffer reports success, writes a well-formed file,
    and is byte-indistinguishable from a complete capture -- it simply stops
    partway through the job. That failure mode already bit this engine once
    (see `_run_pricing_job`'s docstring: the Python tracer silently capped a
    ~90s job's trace at its first 1.6s), and the only cheap way to catch it
    is to compare the captured events' own timestamp span against the wall
    time of the run they were supposed to cover.

    Emits a `UserWarning` rather than raising: a truncated trace is a
    degraded diagnostic, never a reason to fail a pricing job that has
    already computed its result correctly. Any error reading the trace back
    is likewise swallowed -- profiling must not be able to break pricing.
    """
    try:
        newest, total_bytes = None, 0
        for root, _dirs, files in os.walk(out_dir):
            for name in files:
                path = os.path.join(root, name)
                total_bytes += os.path.getsize(path)
                if name.endswith(".trace.json.gz"):
                    if newest is None or os.path.getmtime(path) > os.path.getmtime(newest):
                        newest = path
        if newest is None:
            return

        import gzip
        import json
        events = json.load(gzip.open(newest, "rt"))["traceEvents"]
        stamps = [e["ts"] for e in events if "ts" in e]
        span = (max(stamps) - min(stamps)) / 1e6 if stamps else 0.0

        if len(events) >= _TRACE_EVENT_WARN:
            warnings.warn(
                f"profiler trace in {out_dir!r} has {len(events):,} events, at or near "
                f"the profiler's ~{_TRACE_EVENT_CAP:,}-event buffer cap -- it is probably "
                f"TRUNCATED and silently covers only part of this job. Shrink the "
                f"portfolio, turn Greeks off, or unset JAX_RISK_PROFILE_PYTHON_TRACER.",
                UserWarning, stacklevel=2,
            )
        elif wall_seconds > 1.0 and span < _TRACE_COVERAGE_WARN * wall_seconds:
            warnings.warn(
                f"profiler trace in {out_dir!r} spans only {span:.1f}s of a "
                f"{wall_seconds:.1f}s job ({span / wall_seconds:.0%}) -- it is probably "
                f"truncated or was stopped early; treat the timeline as partial.",
                UserWarning, stacklevel=2,
            )
    except Exception:
        # A profiling self-check must never break, or even noisily interfere
        # with, a pricing job whose result is already computed and correct.
        pass


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
