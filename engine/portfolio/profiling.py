"""
Phase labels for profiler traces, and the summary of a written trace.

The engine worker traces with `python_tracer_level=0`, so trace events carry no Python source
location and a JAX dispatch cannot be attributed to calibration, pricing or a Greek. `phase`
labels regions two ways, since each covers what the other misses:

  - `jax.profiler.TraceAnnotation`: a host-timeline region, which captures eager dispatch
    and XLA compilation (most of this engine's cost).
  - `jax.named_scope`: names baked into ops traced inside it (HLO attribution); it labels
    nothing for already-compiled eager calls.

Both are cheap and inert without an active trace, so they stay in the pricing path.

`summarize_trace` reads a written trace back (events, the time it spans, the time in each
phase), for the engine worker's trace checks and `demos/demo_profile_small.py`. It reads the
profiler's whole record, the `.xplane.pb` that xprof opens, not the `.trace.json.gz` beside it,
which keeps at most about `JSON_EXPORT_EVENT_CAP` events. See docs/concepts/profiling.md.

`traced_phase` narrows a trace to one phase: the profiler runs only while that
phase does. On a GPU the profiler costs per kernel launch whatever it records (measured: a
repeated demo job 5.8 s untraced, 33 s traced, under every CUPTI setting), so a phase's window
is the way to trace it at the cost of that phase alone.
"""
import dataclasses
import os
import time
from collections import Counter
from contextlib import contextmanager
from typing import Callable, Dict, Optional

#: The pipeline's phases (`engine.portfolio.market_path`), in the order they run.
PHASES = ("calibration", "simulation", "pricing", "paired_sample", "exposure", "base_npv", "greeks")
#: One region per trade inside "greeks", named `greeks/trade<index>/<config type>`.
GREEKS_TRADE_PREFIX = "greeks/"
#: The events the profiler's `.trace.json.gz` export keeps at most, the earliest-starting
#: (measured, jax 0.10: a 45 s, 1,500,332-event trace exported 1,000,116, nothing that started
#: after 23 s but the regions that began before it). xprof reads the `.xplane.pb`, which holds
#: every event.
JSON_EXPORT_EVENT_CAP = 1_000_000


@contextmanager
def phase(name: str):
    """Label a region of the pricing path as `name` in a profiler trace (both mechanisms;
    see the module docstring), and trace it when it is the `traced_phase`. JAX is imported
    here, not at module scope, so the engine worker's profiler hook imports nothing when
    `JAX_RISK_PROFILE_DIR` is unset."""
    import jax

    window = _window
    if window is not None and window.name == name and window.wall_seconds is None:
        with window.open():
            with jax.profiler.TraceAnnotation(name), jax.named_scope(name):
                yield
        return
    with jax.profiler.TraceAnnotation(name):
        with jax.named_scope(name):
            yield


def trade_greeks_phase(index: int, trade):
    """The `phase` of one trade's Greeks, by its index in the request and its config type."""
    return phase(f"{GREEKS_TRADE_PREFIX}trade{index}/{type(trade).__name__}")


@dataclasses.dataclass
class PhaseWindow:
    """One phase's traced window (`traced_phase`): `start()` and `stop()` run around the
    phase's first occurrence; `wall_seconds` is how long it took, None until it has run."""

    name: str
    start: Callable[[], None]
    stop: Callable[[], None]
    wall_seconds: Optional[float] = None

    @contextmanager
    def open(self):
        """The window around one run of the phase. The devices finish their queued work
        before it opens and before it closes (JAX dispatches asynchronously), so the window
        holds exactly the phase's device work, which an untraced run would leave to the
        next phase to wait for."""
        _settle()
        self.start()
        started = time.perf_counter()
        try:
            yield
            _settle()
        finally:
            self.wall_seconds = time.perf_counter() - started
            self.stop()


#: The window `traced_phase` has open, if any. Set only by the engine worker's profiler hook,
#: around one job in its single job thread.
_window: Optional[PhaseWindow] = None


@contextmanager
def traced_phase(name: str, start: Callable[[], None], stop: Callable[[], None]):
    """Within the block, `start()` when the phase `name` (a label of `PHASES`, or one
    trade's Greeks, `greeks/trade<i>/<type>`) is entered for the first time and `stop()` when
    it is left: the profiler hook's window. Yields the `PhaseWindow`, whose `wall_seconds`
    stays None if the phase never ran."""
    global _window
    if _window is not None:
        raise RuntimeError(f"a traced phase ({_window.name!r}) is already open")
    _window = PhaseWindow(name, start, stop)
    try:
        yield _window
    finally:
        _window = None


def _settle() -> None:
    """Wait until every device has finished the work queued so far."""
    import jax

    jax.block_until_ready(jax.live_arrays())


@dataclasses.dataclass(frozen=True)
class TraceSummary:
    """What a written trace holds. `phases` is seconds per phase label (a label entered
    more than once is summed); host time, so device work lands in the phase that waits for it.
    `threads` is events per thread name, the most first ("" is the thread that ran the job)."""

    path: str
    bytes: int
    events: int
    span_seconds: float
    phases: Dict[str, float]
    threads: Dict[str, int]

    def coverage(self, wall_seconds: float) -> float:
        """The share of a job of `wall_seconds` that the trace's events span."""
        return self.span_seconds / wall_seconds if wall_seconds > 0 else 1.0

    @property
    def json_export_complete(self) -> bool:
        """Whether the `.trace.json.gz` beside the trace holds all of it."""
        return self.events <= JSON_EXPORT_EVENT_CAP


def latest_trace(directory: str) -> Optional[str]:
    """The newest `*.xplane.pb` under `directory` (the profiler writes one per trace, in
    `plugins/profile/<timestamp>/`), or None."""
    traces = [os.path.join(root, name) for root, _dirs, files in os.walk(directory)
              for name in files if name.endswith(".xplane.pb")]
    return max(traces, key=os.path.getmtime) if traces else None


def summarize_trace(path: str) -> TraceSummary:
    """The `TraceSummary` of the trace file `path` (a `*.xplane.pb`). `bytes` counts every
    file of that trace (its directory), the `.trace.json.gz` export included."""
    from jax.profiler import ProfileData

    events, first, last = 0, float("inf"), float("-inf")
    phases: Dict[str, float] = {}
    threads: Counter = Counter()
    for plane in ProfileData.from_file(path).planes:
        for line in plane.lines:
            for event in line.events:
                events += 1
                threads[line.name] += 1
                first, last = min(first, event.start_ns), max(last, event.start_ns + event.duration_ns)
                name = event.name
                if name in PHASES or name.startswith(GREEKS_TRADE_PREFIX):
                    phases[name] = phases.get(name, 0.0) + event.duration_ns / 1e9
    directory = os.path.dirname(path)
    size = sum(os.path.getsize(os.path.join(directory, name)) for name in os.listdir(directory))
    return TraceSummary(path=path, bytes=size, events=events, span_seconds=(last - first) / 1e9 if events else 0.0,
                        phases=phases, threads=dict(threads.most_common()))
