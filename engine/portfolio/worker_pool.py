"""
Worker-process pools for `price_portfolio`, one per precision tier.

`jax_enable_x64` is process-global and cannot be scoped per thread, so jobs at different
precisions cannot share a process safely. Each tier (float32, float64) gets its own
`ProcessPoolExecutor`; each worker sets the flag once at start-up and runs one job at a
time. Jobs beyond a pool's size queue. `request.precision.simulation` selects the tier;
`pricing` and `risk` are applied inside the job by `price_portfolio`'s own casts.

Workers are always spawned, never forked: forking a process that has initialized JAX hangs
(I-33, on Linux). Spawn pickles the initializer by reference, so `_worker_init` is a
top-level function.

Trade configs hold `ORE.Date`/`ORE.Period` SWIG objects, which do not pickle.
`_freeze_trade` writes them as text in a `_FrozenTrade` record and `_thaw_trade` rebuilds
the config (re-running its validation) in the worker. `PortfolioResult` holds no ORE types
and pickles as is.

On a machine with one CPU device every worker shares it; there is no device pinning.
`_DEFAULT_POOL_SIZE` is a small development default.
"""
import multiprocessing
import os
import time
import warnings
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Optional

from engine.portfolio.request import PortfolioRequest, PortfolioResult

# Small development default; each worker is a full Python + JAX process.
_DEFAULT_POOL_SIZE = 2

_POOLS: dict = {}  # precision_bits (32 or 64) -> ProcessPoolExecutor


def _worker_init(precision_bits: int) -> None:
    """Pool initializer: runs once per worker, before its first job, and sets
    `jax_enable_x64` for this worker's tier. A worker runs one job at a time, so the flag
    never changes under a job.

    JAX is already imported when this runs (unpickling this function imports this
    module, which imports `engine.portfolio.request`), so device selection by environment
    variable would have to be done at process launch, not here. None is done today.
    """
    # No device pinning (see the docstring).
    os.environ.setdefault("JAX_PLATFORMS", "")

    import jax

    jax.config.update("jax_enable_x64", precision_bits == 64)


@dataclass(frozen=True)
class _OreValue:
    """An `ORE.Date` or `ORE.Period` written as text, so it pickles."""
    kind: str  # "date" | "period"
    text: str


@dataclass(frozen=True)
class _FrozenTrade:
    """A dataclass (a trade config, or one nested in it) in picklable form: its class and
    its field values."""
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
    if isinstance(value, dict):
        # Historical fixings are keyed by ORE.Date.
        return {_freeze_value(k): _freeze_value(v) for k, v in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        # Nested dataclasses (a bond's CouponPeriods) can hold ORE dates too.
        return _freeze_trade(value)
    return value


def _thaw_value(value):
    import ORE

    if isinstance(value, _OreValue):
        return ORE.DateParser.parseISO(value.text) if value.kind == "date" else ORE.Period(value.text)
    if isinstance(value, _FrozenTrade):
        return _thaw_trade(value)
    if isinstance(value, (list, tuple)):
        return type(value)(_thaw_value(v) for v in value)
    if isinstance(value, dict):
        return {_thaw_value(k): _thaw_value(v) for k, v in value.items()}
    return value


def _freeze_trade(cfg) -> _FrozenTrade:
    """Trade config -> picklable `_FrozenTrade`: every `ORE.Date`/`ORE.Period`, including
    inside lists, dicts and nested dataclasses, becomes text. Generic over config types."""
    return _FrozenTrade(type(cfg), {f.name: _freeze_value(getattr(cfg, f.name)) for f in fields(cfg)})


def _thaw_trade(frozen: _FrozenTrade):
    """Inverse of `_freeze_trade`, run in the worker; rebuilding the config re-runs its
    validation."""
    return frozen.cls(**{name: _thaw_value(value) for name, value in frozen.values.items()})


def _profile_options(jax):
    """`jax.profiler.ProfileOptions` for `_run_pricing_job`'s trace: Python tracer off
    unless `JAX_RISK_PROFILE_PYTHON_TRACER=1`; host tracer and HLO protos at JAX's
    defaults. `jax` is passed in so this module does not import it itself.

    With the Python tracer off, the host tracer still records XLA compilation, pjit
    dispatch, tracing and device execution, so compile vs dispatch vs execute remain
    separable (measured on the 4-trade demo: compilation ~119s, dispatch ~76s, tracing
    ~3s, execution ~3s, summed across concurrent lanes). What is lost is Python source
    attribution: no event names the engine function that dispatched it. Phase-level
    attribution comes from `engine.portfolio.profiling.phase` regardless; only
    per-callsite attribution needs the Python tracer.
    """
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 1 if os.environ.get("JAX_RISK_PROFILE_PYTHON_TRACER") == "1" else 0
    return options


def _run_pricing_job(frozen_request: PortfolioRequest) -> PortfolioResult:
    """Run one job in the worker: thaw the trades and call `price_portfolio`.

    Profiling (opt-in): with `JAX_RISK_PROFILE_DIR` set, the call runs under
    `jax.profiler.trace`, written to `$JAX_RISK_PROFILE_DIR/pid-<pid>/`, including
    compilation. View with `xprof --port 8791 <dir>`. Unset, nothing is traced.

    The Python tracer is off by default (see `_profile_options`). With it on, 97% of events
    were interpreter frames from JAX's dispatch machinery, and the profiler's ~1M-event
    buffer, which drops events silently, filled after the first 1.6s of a ~90s job. Off,
    the trace shrank from 467MB to 50MB and covered 93% of the job instead of 2%.

    `JAX_RISK_PROFILE_WARMUP=1` runs the job once untraced first, so the trace shows warm
    execution rather than compilation (and the job runs twice).
    """
    from engine.portfolio.request import price_portfolio

    def _run() -> PortfolioResult:
        trades = [_thaw_trade(cfg) for cfg in frozen_request.trades]
        request = replace(frozen_request, trades=trades)
        return price_portfolio(request)

    profile_dir = os.environ.get("JAX_RISK_PROFILE_DIR")
    if not profile_dir:
        return _run()

    import jax

    if os.environ.get("JAX_RISK_PROFILE_WARMUP") == "1":
        # Untraced warm-up run to populate the compilation caches.
        jax.block_until_ready(_run().npv_cube)

    out_dir = os.path.join(profile_dir, f"pid-{os.getpid()}")
    started = time.time()
    with jax.profiler.trace(out_dir, profiler_options=_profile_options(jax)):
        result = _run()
        # Wait for device execution before the trace closes, or the timeline is cut short.
        jax.block_until_ready(result.npv_cube)
    _warn_if_trace_truncated(out_dir, time.time() - started)
    return result


# The profiler's event buffer is capped and drops events silently once full.
_TRACE_EVENT_CAP = 1_000_000
_TRACE_EVENT_WARN = 950_000
# Minimum fraction of the job's wall time a complete trace should span (a complete trace
# still starts slightly after and ends slightly before the measured window).
_TRACE_COVERAGE_WARN = 0.5


def _warn_if_trace_truncated(out_dir: str, wall_seconds: float) -> None:
    """Warn if a trace looks truncated: near the event cap, or spanning less than half the
    job's wall time. A truncated trace is otherwise indistinguishable from a complete one.
    Warns rather than raises, and swallows any error reading the trace, so profiling
    cannot break a job."""
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
        # Profiling must not interfere with a completed job.
        pass


def _pool_for(precision_bits: int, pool_size: int = _DEFAULT_POOL_SIZE) -> ProcessPoolExecutor:
    """The pool for tier 32 or 64, created on first use and kept for this process's
    lifetime."""
    if precision_bits not in (32, 64):
        raise ValueError(f"precision_bits must be 32 or 64, got {precision_bits!r}")
    pool = _POOLS.get(precision_bits)
    if pool is None:
        # Spawn explicitly; fork hangs once the parent has run JAX (I-33).
        pool = ProcessPoolExecutor(
            max_workers=pool_size,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_worker_init,
            initargs=(precision_bits,),
        )
        _POOLS[precision_bits] = pool
    return pool


def submit_pricing_job(request: PortfolioRequest, pool_size: int = _DEFAULT_POOL_SIZE) -> "Future[PortfolioResult]":
    """Submit `request` to the pool for `request.precision.simulation` and return a
    `Future[PortfolioResult]`. `pool_size` applies only when that tier's pool is first
    created."""
    pool = _pool_for(request.precision.simulation, pool_size=pool_size)
    frozen_trades = [_freeze_trade(cfg) for cfg in request.trades]
    frozen_request = replace(request, trades=frozen_trades)
    return pool.submit(_run_pricing_job, frozen_request)


def shutdown_pools(wait: bool = True) -> None:
    """Shut down every pool this process created (mainly for tests); a long-running API
    process keeps its pools."""
    for precision_bits in list(_POOLS.keys()):
        pool = _POOLS.pop(precision_bits)
        pool.shutdown(wait=wait)
