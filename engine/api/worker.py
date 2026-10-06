"""
The engine worker: one single-threaded process per host that takes portfolio jobs from the
job queue (`engine.api.job_queue`), prices them and writes the results back (decision A-14,
roadmap 1.8; docs/planning/details/precision.md §11).

    jax-risk-worker [--queue PATH] [--parent-pid PID]

(or `python -c "from engine.api.worker import main; main()" ...`). The API starts and
supervises one itself unless `JAX_RISK_WORKER=external` (`engine.api.supervisor`).

A job is the HTTP body as the API received it. The worker parses it exactly as the route did
(`json.loads`, then `MarketPortfolioRequestSchema`), so nothing is pickled and no ORE object
crosses a process boundary, calls `price_portfolio` in this process, and stores the result
document `PortfolioResultSchema` serializes, the bytes a route would have sent. Jobs run one
at a time, in submission order. The process owns every device JAX sees on its host; no device
is pinned or shared with another engine process (the API keeps its own JAX on the CPU,
`engine.api.app.keep_jax_on_the_cpu`, and `engine` turns GPU preallocation off). XLA programs
stay compiled for the process's lifetime, so a repeated job shape compiles nothing (each job's
count is stored in its row). Across restarts, a worker started by `main` keeps JAX's
persistent compilation cache: in `JAX_COMPILATION_CACHE_DIR` if set (empty turns it off), else
in `xla-cache/` beside the queue file, every program cached (`compilation_cache_environment`).
A restarted worker then reads its programs back instead of compiling them again.

A failing job fails only its own row, with a failure class (`failure_class`) and the
traceback; the worker goes on to the next job. A worker that dies mid-job leaves the row
`running`; the next worker to take the queue marks it `interrupted` before claiming anything.

The queue's lock is taken before JAX is imported, so a redundant worker (another already
holds the queue) exits within milliseconds with `EXIT_QUEUE_OWNED`. With `--parent-pid` the
worker exits once that process is gone (checked between jobs), so a killed API or test run
leaves no engine process behind.

Profiling (opt-in): with `JAX_RISK_PROFILE_DIR` set, each job's `price_portfolio` runs under
`jax.profiler.trace`, written to `$JAX_RISK_PROFILE_DIR/pid-<pid>/` (see `_profiled`).
"""
import argparse
import dataclasses
import json
import os
import socket
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Callable, Optional

from engine.api.job_queue import (
    BAD_TERMS, INFRASTRUCTURE, MISSING_MARKET_DATA, NUMERICAL_FAILURE, UNSUPPORTED_PRODUCT, Job, JobQueue,
    WorkerLock, default_queue_path,
)

#: Exit status of a worker that found another worker serving its queue.
EXIT_QUEUE_OWNED = 3
#: How long a starting worker retries the lock: a supervisor's probe holds it for an instant
#: (`engine.api.job_queue.worker_lock_held`).
LOCK_WAIT_SECONDS = 2.0
#: Idle sleep between looks at the queue. A look costs about 5 us on the worker's kept
#: connection (`engine.api.job_queue`), so a 5 ms period is well under 1% of a core and adds
#: 2.5 ms to a job's latency on average.
POLL_SECONDS = 0.005

#: The persistent compilation cache's directory beside the queue file, when
#: `JAX_COMPILATION_CACHE_DIR` is unset.
COMPILATION_CACHE_DIRNAME = "xla-cache"

#: `request body -> result document`, the work of one job.
Pricer = Callable[[bytes], str]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="jax-risk-worker", description=__doc__.split("\n\n")[0])
    parser.add_argument("--queue", default=str(default_queue_path()),
                        help="the job queue's SQLite file (default: JAX_RISK_JOB_QUEUE or the temp directory)")
    parser.add_argument("--parent-pid", type=int, default=None,
                        help="exit once this process is gone (the supervising API)")
    parser.add_argument("--poll-seconds", type=float, default=POLL_SECONDS)
    args = parser.parse_args(argv)
    # Before JAX is imported (`serve` imports it after taking the lock), which reads these.
    for name, value in compilation_cache_environment(args.queue, os.environ).items():
        os.environ.setdefault(name, value)
    sys.exit(serve(args.queue, parent_pid=args.parent_pid, poll_seconds=args.poll_seconds))


def compilation_cache_environment(queue_path, environ) -> dict:
    """The environment that turns on JAX's persistent compilation cache for a worker on
    `queue_path`, for the variables `environ` does not set already: the cache beside the queue
    file, and every program cached, as the tests do (JAX's own default skips programs that
    compile in under a second; a job builds about a hundred small ones). A program is keyed on
    its HLO, the compile options and the jax/jaxlib versions, so a hit is the executable a
    fresh compile would build. An explicit `JAX_COMPILATION_CACHE_DIR` wins, and an empty one
    turns the cache off."""
    defaults = {
        "JAX_COMPILATION_CACHE_DIR": str(Path(queue_path).resolve().parent / COMPILATION_CACHE_DIRNAME),
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "0",
    }
    return {name: value for name, value in defaults.items() if name not in environ}


def serve(queue_path, *, price: Optional[Pricer] = None, parent_pid: Optional[int] = None,
          poll_seconds: float = POLL_SECONDS, drain: bool = False) -> int:
    """Be the queue's engine worker until the parent is gone (or, with `drain`, until no job
    is pending); the process exit status. `price` defaults to `price_job`."""
    lock = WorkerLock(queue_path)
    if not lock.acquire(wait_seconds=LOCK_WAIT_SECONDS):
        return EXIT_QUEUE_OWNED
    compiles = queue = None
    try:
        queue = JobQueue(queue_path)
        queue.interrupt_running()
        compiles = _CompileCounter()
        price = price or price_job
        parent = _ParentWatch(parent_pid)
        worker = f"{socket.gethostname()}:{os.getpid()}"
        while parent.alive():
            job = queue.claim(worker)
            if job is None:
                if drain:
                    return 0
                time.sleep(poll_seconds)
                continue
            run_job(queue, job, price, compiles)
        return 0
    finally:
        if compiles is not None:
            compiles.close()
        if queue is not None:
            queue.close()
        lock.release()


def run_job(queue: JobQueue, job: Job, price: Pricer, compiles: "_CompileCounter") -> None:
    """Price one claimed job and close its row: `done` with the result document, or `failed`
    with the failure class and traceback. Nothing raised by the job escapes; a result the
    queue cannot store (larger than SQLite's 1 GB limit) fails the job as `infrastructure`.
    Only a queue that cannot be written at all stops the worker."""
    before = compiles.count
    try:
        result = price(job.request)
    except Exception as exc:
        queue.fail(job.id, failure_class(exc), _describe(exc), compiles=compiles.count - before)
        return
    try:
        queue.finish(job.id, result, compiles=compiles.count - before)
    except Exception as exc:
        queue.fail(job.id, INFRASTRUCTURE, f"storing the result failed: {_describe(exc)}",
                   compiles=compiles.count - before)


def _describe(exc: BaseException) -> str:
    return "".join(traceback.format_exception(exc))


def price_job(request: bytes) -> str:
    """One job's work: the HTTP body -> `price_portfolio` -> the result document. Parsed as
    the route parsed it, so the dataclass request is the one the route validated, and
    serialized as the route would have, so the document is the bytes it would have sent."""
    from engine.api.market_schemas import MarketPortfolioRequestSchema
    from engine.api.schemas import PortfolioResultSchema
    from engine.portfolio import price_portfolio

    dataclass_request = MarketPortfolioRequestSchema.model_validate(json.loads(request)).to_dataclass()
    result = _profiled(lambda: price_portfolio(dataclass_request))
    return PortfolioResultSchema.from_dataclass(result).model_dump_json()


def failure_class(exc: BaseException) -> str:
    """The failure class of a job that raised `exc`, by the exception's type. The route has
    validated the request before queueing it, so a failure here is mostly a pricing one;
    a request that reaches the worker unvalidated is classified the same way."""
    from engine.day_count import UnsupportedDayCountError
    from engine.models.ore_builders import MissingFixingError

    if isinstance(exc, (KeyError, MissingFixingError)):  # a curve, currency or fixing not supplied
        return MISSING_MARKET_DATA
    if isinstance(exc, (NotImplementedError, UnsupportedDayCountError)):
        return UNSUPPORTED_PRODUCT
    if isinstance(exc, ArithmeticError):  # FloatingPointError, OverflowError, ZeroDivisionError
        return NUMERICAL_FAILURE
    if isinstance(exc, (ValueError, TypeError)):  # pydantic's ValidationError is a ValueError
        return BAD_TERMS
    return INFRASTRUCTURE  # MemoryError, XLA runtime errors, OSError, ...


class _CompileCounter:
    """XLA programs this process has had to build, compiled or read from the persistent cache:
    JAX's `backend_compile_duration` event, recorded on every miss of its in-memory caches
    (`jax._src.interpreters.pxla`). A repeated job shape adds nothing."""

    EVENT = "/jax/core/compile/backend_compile_duration"

    def __init__(self):
        import jax.monitoring

        self.count = 0
        jax.monitoring.register_event_duration_secs_listener(self)

    def __call__(self, event: str, duration: float, **kwargs) -> None:
        if event == self.EVENT:
            self.count += 1

    def close(self) -> None:
        import jax.monitoring

        jax.monitoring.unregister_event_duration_listener(self)


class _ParentWatch:
    """Whether the process `pid` is still running (always, without a `pid`). On Windows a
    handle opened now keeps the process object, so a reused pid cannot be mistaken for it;
    on POSIX an orphan is re-parented, so `getppid` changes."""

    def __init__(self, pid: Optional[int]):
        self._pid = pid
        self._handle = None
        if pid is not None and os.name == "nt":
            import ctypes

            synchronize = 0x00100000
            self._handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, pid)
            if not self._handle:  # already gone
                self._pid = -1

    def alive(self) -> bool:
        if self._pid is None:
            return True
        if self._pid == -1:
            return False
        if os.name == "nt":
            import ctypes

            wait_timeout = 0x102
            return ctypes.windll.kernel32.WaitForSingleObject(self._handle, 0) == wait_timeout
        return os.getppid() == self._pid


def _profile_options(jax):
    """`jax.profiler.ProfileOptions` for `_profiled`'s trace: Python tracer off unless
    `JAX_RISK_PROFILE_PYTHON_TRACER=1`; host tracer and HLO protos at JAX's defaults. `jax` is
    passed in so this module does not import it itself.

    With the Python tracer off, the host tracer still records XLA compilation, pjit dispatch,
    tracing and device execution, so compile vs dispatch vs execute remain separable (measured
    on the 4-trade demo: compilation ~119s, dispatch ~76s, tracing ~3s, execution ~3s, summed
    across concurrent lanes). What is lost is Python source attribution: no event names the
    engine function that dispatched it. Phase-level attribution comes from
    `engine.portfolio.profiling.phase` regardless; only per-callsite attribution needs the
    Python tracer.
    """
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 1 if os.environ.get("JAX_RISK_PROFILE_PYTHON_TRACER") == "1" else 0
    return options


def _profiled(run):
    """`run()`, under `jax.profiler.trace` when `JAX_RISK_PROFILE_DIR` is set (written to
    `$JAX_RISK_PROFILE_DIR/pid-<pid>/`, compilation included; view with
    `xprof --port 8791 <dir>`). Unset, nothing is traced.

    The Python tracer is off by default (see `_profile_options`). With it on, 97% of events
    were interpreter frames from JAX's dispatch machinery, and the trace's `.trace.json.gz`,
    which keeps only the ~1M events that start first, covered the first 1.6s of a ~90s job.
    Off, the trace shrank from 467MB to 50MB and covered 93% of the job instead of 2%.

    `JAX_RISK_PROFILE_WARMUP=1` runs the job once untraced first, so the trace shows warm
    execution rather than compilation (and the job runs twice).

    Beside each trace goes its summary, `<trace run>.summary.json` (`_record_trace`).
    """
    profile_dir = os.environ.get("JAX_RISK_PROFILE_DIR")
    if not profile_dir:
        return run()

    import jax

    warmup = None
    if os.environ.get("JAX_RISK_PROFILE_WARMUP") == "1":
        # Untraced warm-up run to populate the compilation caches.
        _, warmup = _measured(jax, run)

    out_dir = os.path.join(profile_dir, f"pid-{os.getpid()}")
    with jax.profiler.trace(out_dir, profiler_options=_profile_options(jax)):
        result, traced = _measured(jax, run)
    _record_trace(out_dir, traced, warmup)
    return result


def _measured(jax, run):
    """`run()`'s result once its device work is done (a trace closed before it would be cut
    short), and `{"wall_seconds", "compiles"}` of that run."""
    compiles = _CompileCounter()
    started = time.perf_counter()
    try:
        result = run()
        jax.block_until_ready(result.npv_cube)
    finally:
        compiles.close()
    return result, {"wall_seconds": time.perf_counter() - started, "compiles": compiles.count}


def _record_trace(out_dir: str, traced: dict, warmup: Optional[dict]) -> None:
    """Summarize the trace just written under `out_dir` (`engine.portfolio.profiling.summarize_trace`)
    into `out_dir/<trace run>.summary.json`, with the traced run's and the warm-up's wall time
    and compiles and the share of the traced run the trace covers, and warn if it is partial
    (`_trace_warning`). Swallows any error, so profiling cannot break a job whose result is
    already correct."""
    try:
        from engine.portfolio.profiling import latest_trace, summarize_trace

        path = latest_trace(out_dir)
        if path is None:
            return
        summary = summarize_trace(path)
        warning = _trace_warning(summary, traced["wall_seconds"])
        document = {"traced": traced, "warmup": warmup, "coverage": summary.coverage(traced["wall_seconds"]),
                    "warning": warning, **dataclasses.asdict(summary)}
        run_name = os.path.basename(os.path.dirname(path))
        with open(os.path.join(out_dir, f"{run_name}.summary.json"), "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=1)
        if warning:
            warnings.warn(warning, UserWarning, stacklevel=2)
    except Exception:
        # Profiling must not interfere with a completed job.
        pass


# Minimum fraction of the job's wall time a complete trace should span (a complete trace
# still starts slightly after and ends slightly before the measured window).
_TRACE_COVERAGE_WARN = 0.5


def _trace_warning(summary, wall_seconds: float) -> Optional[str]:
    """What is partial about the trace of `summary` (a `TraceSummary`), or None: the record
    itself, if it spans less than half the job's `wall_seconds` (stopped early); else its
    `.trace.json.gz` export, if the trace has more events than the export keeps."""
    from engine.portfolio.profiling import JSON_EXPORT_EVENT_CAP

    if wall_seconds > 1.0 and summary.coverage(wall_seconds) < _TRACE_COVERAGE_WARN:
        return (f"profiler trace {summary.path!r} spans only {summary.span_seconds:.1f}s of a "
                f"{wall_seconds:.1f}s job ({summary.coverage(wall_seconds):.0%}) -- it was probably "
                f"stopped early; treat the timeline as partial.")
    if not summary.json_export_complete:
        return (f"profiler trace {summary.path!r} has {summary.events:,} events; the .trace.json.gz "
                f"beside it keeps only the ~{JSON_EXPORT_EVENT_CAP:,} that start first, so a viewer "
                f"reading that file (Perfetto, chrome://tracing) sees part of the job. xprof reads the "
                f"whole trace.")
    return None
