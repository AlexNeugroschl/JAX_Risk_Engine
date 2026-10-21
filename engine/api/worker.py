"""
The engine worker: one single-threaded process per host that takes jobs from the job queue
(`engine.api.job_queue`), runs them and writes the results back (decision A-14;
docs/planning/details/precision.md §11).

    jax-risk-worker [--queue PATH] [--lock PATH] [--parent-pid PID]

(or `python -c "from engine.api.worker import main; main()" ...`). The API starts and
supervises one itself unless `JAX_RISK_WORKER=external` (`engine.api.supervisor`).

A job is a run request's HTTP body as the API received it. The worker parses the body exactly
as the route did (`engine.api.requests.RunRequestSchema`), so nothing is pickled and no ORE
object crosses a process boundary, runs it in this process (`engine.run.run`), and stores the
result document (`engine.api.results.run_result_document`), with the arrays the request asked
for by reference as chunked artifacts (`engine.api.artifacts`). Jobs run one at a time, in
submission order.

**Cancelling** (I-91): before a job starts and at each phase boundary of its run
(`engine.run.trace.at_phase_boundaries`: the calibration, the simulation, each trade's pricing,
the Greeks, ...) the worker looks whether a cancel has reached the job, and stops it there with
`JobCancelled`; the job reads `cancelled`, and the worker goes on to the next one with its
compiled programs. Between boundaries a job is never interrupted.

**Retention** (I-76): between jobs, at most every `PURGE_INTERVAL_SECONDS`, the worker deletes the
finished jobs older than the retention period (`engine.api.job_queue.retention_seconds`,
`JAX_RISK_RETENTION_DAYS`), but those submitted with `keep`. The process owns every device JAX sees on its host; no device
is pinned or shared with another engine process (the API keeps its own JAX on the CPU,
`engine.api.app.keep_jax_on_the_cpu`), so the worker keeps JAX's default of preallocating
75% of a GPU; set `XLA_PYTHON_CLIENT_PREALLOCATE=false` where other processes share the card
(the demos and the tests do). XLA programs stay compiled for the process's lifetime, so a
repeated job shape compiles nothing (each job's count is stored in its row). Across restarts,
a worker started by `main` keeps JAX's persistent compilation cache: in
`JAX_COMPILATION_CACHE_DIR` if set (empty turns it off), else in `xla-cache/` beside the queue
file, every program cached (`compilation_cache_environment`). A restarted worker then reads its
programs back instead of compiling them again.

A worker started by `main` also runs deterministic GPU kernels, so a compiled job gives the
same bits on every run (`deterministic_kernels_environment`, decision A-22): it appends
`--xla_gpu_exclude_nondeterministic_ops=true` to its `XLA_FLAGS`, unless they name either XLA
determinism flag. XLA has no per-computation switch, so it is set for the process whose
results the engine vouches for; it changes nothing on a CPU. A GPU's FP8 products run on its FP8
units only where the operator sends its matrix products to cuBLAS
(`XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`, I-97), which the worker inherits and the engine
reads (`engine.precision.products.cublas_gemms`); by default they are emulated, exactly, since
cuBLAS slowed the engine's float64 products on an RTX 5060.

**On a pod slice** (`--distributed`, I-61): one worker on every host, all running each job at
once, as SPMD requires (`engine.api.hosts`). Process 0 alone serves the queue and hands each
job to the others; each phase boundary's cancel is decided by process 0 for every host, so all
stop at the same one; a failure the request determines is reported by every host, and any other
stops the whole worker set, whose job the restarted process 0 runs again once. Each host holds
its own worker lock; start every host's worker externally (`JAX_RISK_WORKER=external`), the API
beside process 0.

A failing job fails only its own row, with a failure class (`failure_class`) and the
traceback; the worker goes on to the next job. A worker that dies mid-job leaves the row
`running`; the next worker to take the queue marks it `interrupted` before claiming anything.

The host's worker lock (`engine.api.job_queue.worker_lock_path`, I-106) is taken before JAX is
imported, so a redundant worker (another already serves the host) exits within milliseconds with
`EXIT_HOST_OWNED`; the lock records the queue its holder serves. With `--parent-pid` the worker
exits once that process is gone (checked between jobs), so a killed API or test run leaves no
engine process behind.

Profiling (opt-in): with `JAX_RISK_PROFILE_DIR` set, each job's `run` (or, with
`JAX_RISK_PROFILE_PHASE`, one phase of it) runs under the JAX profiler, written to
`$JAX_RISK_PROFILE_DIR/pid-<pid>/` (see `_profiled`).
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
from contextvars import ContextVar
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence

from engine.api.job_queue import (
    BAD_TERMS, INFRASTRUCTURE, MISSING_MARKET_DATA, NUMERICAL_FAILURE, RUN, UNSUPPORTED_PRODUCT, Job, JobQueue,
    WorkerLock, default_queue_path, retention_seconds, worker_lock_path,
)
from engine.api.hosts import (
    CANCELLED, DONE, FAILED, IDLE, JOB, LOCAL, OUTCOME_NAMES, STOP, Command, Hosts, distributed_hosts, fail_stop,
)
from engine.api.supervisor import EXIT_HOST_OWNED
#: How long a starting worker retries the lock: a supervisor's probe holds it for an instant
#: (`engine.api.job_queue.worker_lock_held`).
LOCK_WAIT_SECONDS = 2.0
#: Idle sleep between looks at the queue. A look costs about 5 us on the worker's kept
#: connection (`engine.api.job_queue`), so a 5 ms period is well under 1% of a core and adds
#: 2.5 ms to a job's latency on average.
POLL_SECONDS = 0.005
#: The least time between two retention purges (`JobQueue.purge`): a purge is one indexed delete.
PURGE_INTERVAL_SECONDS = 60.0
#: Idle sleep of a worker set of several hosts: each look at the queue is a broadcast to every
#: host (`engine.api.hosts.Hosts.share`), a few milliseconds across a network.
DISTRIBUTED_POLL_SECONDS = 0.05

#: The persistent compilation cache's directory beside the queue file, when
#: `JAX_COMPILATION_CACHE_DIR` is unset.
COMPILATION_CACHE_DIRNAME = "xla-cache"

#: XLA's flag that excludes GPU kernels whose results depend on scheduling (atomics). Without
#: it the AD Greeks' scatter-adds moved by an ulp between identical runs on an RTX 5060
#: (2026-10-06). `--xla_gpu_deterministic_ops=true` also pins autotuning, at twice the
#: compile time; an operator who sets either flag, true or false, is not overridden.
DETERMINISTIC_KERNELS_FLAG = "--xla_gpu_exclude_nondeterministic_ops=true"

@dataclasses.dataclass(frozen=True)
class JobResult:
    """What a job leaves in its row: the result document (JSON) and its artifacts (name ->
    chunks of bytes), stored together (`JobQueue.finish`)."""
    document: str
    artifacts: Mapping[str, Sequence[bytes]] = dataclasses.field(default_factory=dict)


#: `job -> JobResult`, the work of one job.
Pricer = Callable[[Job], JobResult]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="jax-risk-worker", description=__doc__.split("\n\n")[0])
    parser.add_argument("--queue", default=str(default_queue_path()),
                        help="the job queue's SQLite file (default: JAX_RISK_JOB_QUEUE or the temp directory)")
    parser.add_argument("--lock", default=str(worker_lock_path()),
                        help="the host's worker lock (default: JAX_RISK_WORKER_LOCK or the temp directory)")
    parser.add_argument("--parent-pid", type=int, default=None,
                        help="exit once this process is gone (the supervising API)")
    parser.add_argument("--distributed", action="store_true",
                        help="one worker of a set on every host of a pod slice (engine.api.hosts): process 0 "
                             "serves the queue and every host runs each job")
    parser.add_argument("--gpu-gemm", choices=("default", "cublas"), default=None,
                        help="GPU matrix-product route, set before JAX starts; cublas enables the native FP8 route")
    parser.add_argument("--poll-seconds", type=float, default=None,
                        help=f"idle wait between looks at the queue (default {POLL_SECONDS} s, "
                             f"{DISTRIBUTED_POLL_SECONDS} s with --distributed)")
    args = parser.parse_args(argv)
    poll = args.poll_seconds if args.poll_seconds is not None else \
        DISTRIBUTED_POLL_SECONDS if args.distributed else POLL_SECONDS
    # Before JAX is imported (`serve` imports it after taking the lock), which reads these.
    os.environ.update(compilation_cache_environment(args.queue, os.environ))
    os.environ.update(gemm_environment(args.gpu_gemm, os.environ))
    os.environ.update(deterministic_kernels_environment(os.environ))
    sys.exit(serve(args.queue, lock_path=args.lock, parent_pid=args.parent_pid, poll_seconds=poll,
                   distributed=args.distributed))


def gemm_environment(route, environ) -> dict:
    """Explicit process route without importing JAX or changing any unrelated XLA flag.
    None preserves inherited settings; a running backend cannot be switched per job."""
    if route is None:
        return {}
    if route not in ("default", "cublas"):
        raise ValueError(f"unknown GPU GEMM route: {route}")
    flags = [f for f in environ.get("XLA_FLAGS", "").split()
             if not f.startswith("--xla_gpu_enable_triton_gemm=")]
    if route == "cublas":
        flags.append("--xla_gpu_enable_triton_gemm=false")
    return {"XLA_FLAGS": " ".join(flags)}


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


def deterministic_kernels_environment(environ) -> dict:
    """`XLA_FLAGS` with `DETERMINISTIC_KERNELS_FLAG` appended to the flags `environ` has, or
    nothing when they already name a determinism flag (`--xla_gpu_deterministic_ops` or
    `--xla_gpu_exclude_nondeterministic_ops`, either value). For the engine worker and the test
    suite, before JAX opens a device; XLA reads the flags then."""
    flags = environ.get("XLA_FLAGS", "")
    if "deterministic_ops" in flags:
        return {}
    return {"XLA_FLAGS": f"{flags} {DETERMINISTIC_KERNELS_FLAG}".strip()}


def serve(queue_path, *, lock_path=None, price: Optional[Pricer] = None, parent_pid: Optional[int] = None,
          poll_seconds: float = POLL_SECONDS, drain: bool = False, retention: Optional[float] = None,
          environ: Optional[Mapping[str, str]] = None, distributed: bool = False) -> int:
    """Be the host's engine worker, serving the queue at `queue_path`, until the parent is gone
    (or, with `drain`, until no job is pending); the process exit status. `lock_path` defaults to
    `worker_lock_path(environ)`, `price` to `price_job`, and the retention period in seconds to
    `retention_seconds(environ)`.

    With `distributed`, one worker of a set, one on every host of a cluster (I-61,
    `engine.api.hosts`): once its host's lock is taken it joins the cluster; the primary (JAX's
    process 0) alone opens the queue, puts back the job its set was running when it stopped
    (`JobQueue.requeue_running`), purges finished jobs and claims each job, which every host of
    the set then runs (`run_job`); the primary's parent and `drain` decide when the set exits."""
    lock = WorkerLock(lock_path if lock_path is not None else worker_lock_path(environ))
    host, pid = socket.gethostname(), os.getpid()
    if not lock.acquire(wait_seconds=LOCK_WAIT_SECONDS,
                        owner={"queue": str(Path(queue_path).resolve()), "host": host, "pid": pid}):
        return EXIT_HOST_OWNED
    compiles = queue = None
    try:
        hosts = distributed_hosts(environ) if distributed else LOCAL
        if hosts.primary:
            queue = JobQueue(queue_path)
            if hosts.count > 1:
                queue.requeue_running()
            queue.interrupt_running()
        compiles = _CompileCounter()
        price = price or price_job
        claim = _Claims(queue, f"{host}:{pid}", _ParentWatch(parent_pid), drain,
                        retention if retention is not None else retention_seconds(environ)) if hosts.primary else None
        while True:
            command = hosts.share(claim.next() if hosts.primary else None)
            if command.kind == STOP:
                return 0
            if command.kind == IDLE:
                time.sleep(poll_seconds)
                continue
            run_job(queue, command.job, price, compiles, hosts)
    finally:
        if compiles is not None:
            compiles.close()
        if queue is not None:
            queue.close()
        lock.release()


class _Claims:
    """The primary's next `Command`: `STOP` once its parent is gone (or, with `drain`, once no job
    is pending), else the oldest pending job claimed for `worker`, or `IDLE`; finished jobs past
    the retention period `keep_for` (seconds, None: forever) purged at most every
    `PURGE_INTERVAL_SECONDS`."""

    def __init__(self, queue: JobQueue, worker: str, parent: "_ParentWatch", drain: bool, keep_for: Optional[float]):
        self.queue, self.worker, self.parent, self.drain, self.keep_for = queue, worker, parent, drain, keep_for
        self._next_purge = 0.0

    def next(self) -> Command:
        if not self.parent.alive():
            return Command(STOP)
        if self.keep_for is not None and time.monotonic() >= self._next_purge:
            self.queue.purge(time.time() - self.keep_for)
            self._next_purge = time.monotonic() + PURGE_INTERVAL_SECONDS
        job = self.queue.claim(self.worker)
        if job is None:
            return Command(STOP if self.drain else IDLE)
        return Command(JOB, job)


class JobCancelled(BaseException):
    """Raised at a phase boundary of a job a cancel has reached (I-91). A `BaseException`, as
    asyncio's `CancelledError` is, so that no `except Exception` on its way out of the engine
    turns it into a failure."""


#: The running job's cancellation check (`phase_boundary`), set by `run_job` around its pricer.
_CANCELLATION: ContextVar[Optional[Callable[[str], None]]] = ContextVar("cancellation", default=None)


def phase_boundary(name: str) -> None:
    """A phase boundary of the running job: raises `JobCancelled` if a cancel has reached it.
    `price_job` installs it at the run's phases (`engine.run.trace.at_phase_boundaries`); a
    test's pricer calls it directly."""
    check = _CANCELLATION.get()
    if check is not None:
        check(name)


def run_job(queue: Optional[JobQueue], job: Job, price: Pricer, compiles: "_CompileCounter",
            hosts: Hosts = LOCAL) -> None:
    """Price one claimed job and close its row: `done` with the result document, `failed`
    with the failure class and traceback, or `cancelled` at the phase boundary where a cancel
    stopped it. Nothing raised by the job escapes; a result the queue cannot store (larger than
    SQLite's 1 GB limit) fails the job as `infrastructure`. Only a queue that cannot be written
    at all stops the worker.

    On several hosts (`hosts`, the primary's `queue`; None on the others) every host runs the job:
    a cancel stops each at the same phase boundary, the primary's look at the flag shared
    (`Hosts.agree`); a failure of a class the request determines is reported by every host, any
    other stops the worker set (`engine.api.hosts.fail_stop`); the primary closes the row from
    every host's outcome, a job the hosts ended differently failed as `infrastructure`."""
    before = compiles.count

    def stop_if_cancelled(name: str) -> None:
        if hosts.agree(hosts.primary and queue.cancel_requested(job.id)):
            raise JobCancelled(name)

    token = _CANCELLATION.set(stop_if_cancelled)
    try:
        phase_boundary("start")
        result, outcome = price(job), DONE
    except JobCancelled as stop:
        stopped_at, outcome = stop.args[0], CANCELLED
    except Exception as exc:
        if hosts.count > 1 and failure_class(exc) == INFRASTRUCTURE:
            fail_stop(exc)
        failure, outcome = exc, FAILED
    finally:
        _CANCELLATION.reset(token)
    outcomes = hosts.outcomes(outcome)
    if not hosts.primary:
        return
    compiled = compiles.count - before
    if len(set(outcomes)) > 1:
        ended = ", ".join(f"host {i}: {OUTCOME_NAMES[o]}" for i, o in enumerate(outcomes))
        detail = f"; this host's: {_describe(failure)}" if outcome == FAILED else ""
        queue.fail(job.id, INFRASTRUCTURE, f"the hosts ended the job differently ({ended}){detail}", compiles=compiled)
    elif outcome == CANCELLED:
        queue.cancel_running(job.id, f"cancelled at the {stopped_at!r} phase boundary", compiles=compiled)
    elif outcome == FAILED:
        queue.fail(job.id, failure_class(failure), _describe(failure), compiles=compiled)
    else:
        try:
            queue.finish(job.id, result.document, compiles=compiled, artifacts=result.artifacts)
        except Exception as exc:
            queue.fail(job.id, INFRASTRUCTURE, f"storing the result failed: {_describe(exc)}", compiles=compiled)


def _describe(exc: BaseException) -> str:
    return "".join(traceback.format_exception(exc))


def price_job(job: Job) -> JobResult:
    """One job's work, by its kind (`JOB_KINDS`): the HTTP body -> the engine -> the result
    document. Parsed as the route parsed it, so the dataclass request is the one the route
    validated."""
    return JOB_KINDS[job.kind](job)


def _run_job(job: Job) -> JobResult:
    """A run request (`engine.run.run`), stopping at a phase boundary once cancelled."""
    from engine.api.requests import RunRequestSchema
    from engine.api.results import run_result_document
    from engine.run import run
    from engine.run.trace import at_phase_boundaries

    schema = RunRequestSchema.model_validate_json(job.request)
    request = schema.to_dataclass()
    with at_phase_boundaries(phase_boundary):
        result = _profiled(lambda: run(request), _arrays)
    document, artifacts = run_result_document(result, schema, job.id, f"/runs/{job.id}/artifacts")
    return JobResult(document.model_dump_json(), artifacts)


def _arrays(result) -> tuple:
    """The arrays a run is done once computed (`_profiled`'s `ready`): its cube and its P&L."""
    return (result.portfolio.npv_cube if result.portfolio is not None else None,
            result.market_risk.pnl if result.market_risk is not None else None)


#: The work of each job kind (`engine.api.job_queue.KINDS`).
JOB_KINDS: Dict[str, Pricer] = {RUN: _run_job}


def failure_class(exc: BaseException) -> str:
    """The failure class of a job that raised `exc`, by the exception's type. The route has
    validated the request before queueing it, so a failure here is mostly a pricing one;
    a request that reaches the worker unvalidated is classified the same way."""
    from engine.market_data.day_counts import UnsupportedDayCountError
    from engine.instruments.schedules import MissingFixingError

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
    `engine.run.trace.phase` regardless; only per-callsite attribution needs the
    Python tracer.
    """
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 1 if os.environ.get("JAX_RISK_PROFILE_PYTHON_TRACER") == "1" else 0
    return options


def _profiled(run, ready):
    """`run()`, under `jax.profiler.trace` when `JAX_RISK_PROFILE_DIR` is set (written to
    `$JAX_RISK_PROFILE_DIR/pid-<pid>/`, compilation included; view with
    `xprof --port 8791 <dir>`). Unset, nothing is traced. `ready(result)` is the array the job
    is done once computed (a portfolio's cube, a market-risk run's P&L).

    The Python tracer is off by default (see `_profile_options`). With it on, 97% of events
    were interpreter frames from JAX's dispatch machinery, and the trace's `.trace.json.gz`,
    which keeps only the ~1M events that start first, covered the first 1.6s of a ~90s job.
    Off, the trace shrank from 467MB to 50MB and covered 93% of the job instead of 2%.

    `JAX_RISK_PROFILE_WARMUP=1` runs the job once untraced first, so the trace shows warm
    execution rather than compilation (and the job runs twice).

    `JAX_RISK_PROFILE_PHASE=<phase>` traces only that phase of the job (a label of
    `engine.run.trace.PHASES`, or one trade's Greeks, `greeks/trade<i>/<type>`); the
    rest runs untraced. On a GPU the profiler slows every kernel launch, whatever it records
    (docs/concepts/profiling.md §2.0), so a phase is traced at its own cost alone.

    Beside each trace goes its summary, `<trace run>.summary.json` (`_record_trace`).
    """
    profile_dir = os.environ.get("JAX_RISK_PROFILE_DIR")
    if not profile_dir:
        return run()

    import jax

    warmup = None
    if os.environ.get("JAX_RISK_PROFILE_WARMUP") == "1":
        # Untraced warm-up run to populate the compilation caches.
        _, warmup = _measured(jax, run, ready)

    out_dir = os.path.join(profile_dir, f"pid-{os.getpid()}")
    traced_phase = os.environ.get("JAX_RISK_PROFILE_PHASE")
    if not traced_phase:
        with jax.profiler.trace(out_dir, profiler_options=_profile_options(jax)):
            result, traced = _measured(jax, run, ready)
        _record_trace(out_dir, traced, warmup)
        return result

    from engine.run.trace import traced_phase as window_on

    compiles = []

    def start():
        jax.profiler.start_trace(out_dir, profiler_options=_profile_options(jax))
        compiles.append(_CompileCounter())

    def stop():
        jax.profiler.stop_trace()
        compiles[0].close()

    with window_on(traced_phase, start, stop) as window:
        result, job = _measured(jax, run, ready)
    if window.wall_seconds is None:
        warnings.warn(f"JAX_RISK_PROFILE_PHASE={traced_phase!r}: the job has no such phase; nothing was traced "
                      f"(see engine.run.trace.PHASES)", UserWarning, stacklevel=2)
        return result
    traced = {"wall_seconds": window.wall_seconds, "compiles": compiles[0].count, "phase": traced_phase,
              "job_wall_seconds": job["wall_seconds"], "job_compiles": job["compiles"]}
    _record_trace(out_dir, traced, warmup)
    return result


def _measured(jax, run, ready):
    """`run()`'s result once its device work is done (a trace closed before it would be cut
    short), and `{"wall_seconds", "compiles"}` of that run."""
    compiles = _CompileCounter()
    started = time.perf_counter()
    try:
        result = run()
        jax.block_until_ready(ready(result))
    finally:
        compiles.close()
    return result, {"wall_seconds": time.perf_counter() - started, "compiles": compiles.count}


def _record_trace(out_dir: str, traced: dict, warmup: Optional[dict]) -> None:
    """Summarize the trace just written under `out_dir` (`engine.run.trace.summarize_trace`)
    into `out_dir/<trace run>.summary.json`, with the traced run's and the warm-up's wall time
    and compiles and the share of the traced run the trace covers, and warn if it is partial
    (`_trace_warning`). Swallows any error, so profiling cannot break a job whose result is
    already correct."""
    try:
        from engine.run.trace import latest_trace, summarize_trace

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
    from engine.run.trace import JSON_EXPORT_EVENT_CAP

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
