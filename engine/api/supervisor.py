"""
Keeping one engine worker (`engine.api.worker`) alive behind the API (decision A-14).

`JAX_RISK_WORKER` chooses who runs the worker:

  - `spawn` (the default): the API process starts the worker as a child process on the
    first run and checks it on every submission and poll, restarting it if it has died.
    No thread watches it: a dead worker matters only when someone submits or polls, which is
    exactly when it is checked. The worker exits on its own when the API process is gone.
  - `external`: something else runs `jax-risk-worker` (systemd `Restart=always`, a container
    restart policy, a pod's process manager) and the API only reads and writes the queue.

**One worker per host** (I-106): a worker holds the host's worker lock
(`engine.api.job_queue.worker_lock_path`) and records the queue it serves in it. Several API
processes on one queue (uvicorn `--workers N`) share one worker: a supervisor starts its child
only if no process holds the lock, and a redundant child exits at once (`EXIT_HOST_OWNED`). A
queue whose host's worker serves another queue has no worker: `host_conflict` says so, and the
API refuses to queue a run on it.

**A worker that cannot start is reported** (I-77): the supervisor records each child's exit, and
a child that exits with an error having served no run is a failed start. After
`FAILED_STARTS_REPORTED` of them running, or a start that could not even launch the process, the
supervisor's `status` carries the reason and the tail of the worker's output (its stderr goes to
`<queue>.worker.log`), which a pending run's poll and `/health` report. A start that raises is
recorded, never raised: the run is queued and its id returned.

The child is started with `subprocess` (a fresh interpreter), never `fork`: forking a process
that has initialized JAX hangs (I-33).
"""
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from engine.api.job_queue import JobQueue, lock_owner, same_queue, worker_lock_held, worker_lock_path

SPAWN, EXTERNAL = "spawn", "external"

#: The exit status of a worker that found another worker serving its host (`engine.api.worker`).
EXIT_HOST_OWNED = 3
#: Failed starts running before a pending run's poll reports them: one can be a race with a
#: worker of another API process, two in a row are not.
FAILED_STARTS_REPORTED = 2
#: How much of the worker's output a report quotes, from its end.
LOG_TAIL_CHARACTERS = 4000


def worker_mode() -> str:
    """`JAX_RISK_WORKER`: `spawn` (default) or `external`."""
    mode = os.environ.get("JAX_RISK_WORKER", SPAWN)
    if mode not in (SPAWN, EXTERNAL):
        raise ValueError(f"JAX_RISK_WORKER={mode!r}: expected {SPAWN!r} or {EXTERNAL!r}")
    return mode


def worker_command(queue_path, lock_path) -> list:
    """The command line of a worker for `queue_path` under the lock `lock_path`, watching this
    process."""
    return [sys.executable, "-c", "from engine.api.worker import main; main()",
            "--queue", str(queue_path), "--lock", str(lock_path), "--parent-pid", str(os.getpid())]


def host_conflict(queue_path, lock_path) -> Optional[str]:
    """Why no worker can serve `queue_path`, if this host's worker (the holder of `lock_path`)
    serves another queue; None otherwise (I-106)."""
    return _conflict(lock_owner(lock_path), queue_path)


def _conflict(owner: Optional[dict], queue_path) -> Optional[str]:
    """`host_conflict` of the lock's recorded `owner`."""
    if owner is None or same_queue(owner.get("queue", ""), queue_path):
        return None
    return (f"this host's engine worker (pid {owner.get('pid')}) serves the job queue {owner.get('queue')}, not "
            f"this API's {queue_path}: one engine worker per host owns its devices (decision A-14). Point this API at "
            f"that queue (JAX_RISK_JOB_QUEUE), or give the worker of other devices its own lock "
            f"(JAX_RISK_WORKER_LOCK)")


@dataclass(frozen=True)
class WorkerStatus:
    """The worker as a supervisor sees it: whether a worker serves its queue, its failed starts
    running, and why no worker serves the queue when that is known (`error`)."""
    serving: bool
    failed_starts: int
    error: Optional[str]


class WorkerSupervisor:
    """Starts and restarts the engine worker of `queue` as a child of this process, under the
    host's lock (`lock_path`, by default `worker_lock_path()`). `command` overrides the worker's
    command line (tests run a stub pricer, or a worker that cannot start, through it)."""

    def __init__(self, queue: JobQueue, command: Optional[Sequence[str]] = None, lock_path=None):
        self.queue = queue
        self.lock_path = Path(lock_path) if lock_path is not None else worker_lock_path()
        self.log_path = Path(f"{queue.path}.worker.log")
        self._command = list(command) if command is not None else worker_command(queue.path, self.lock_path)
        self._process: Optional[subprocess.Popen] = None
        self._started = 0.0
        self._failed_starts = 0
        self._failure: Optional[str] = None
        self._launch_failed = False
        # Routes run on the server's thread pool; one check-and-start at a time.
        self._lock = threading.Lock()

    @property
    def process(self) -> Optional[subprocess.Popen]:
        """The child started last, alive or not; None before the first start."""
        return self._process

    def ensure_running(self) -> None:
        """Start the worker unless this process's child is alive or another process's worker
        holds the host's lock; a child that has exited is recorded first (`status`)."""
        with self._lock:
            if self._process is not None:
                if self._process.poll() is None:
                    return
                self._record_exit(self._process)
            if worker_lock_held(self.lock_path):
                return
            try:
                with open(self.log_path, "wb") as log:  # the child keeps its own handle
                    # No stdin: the worker reads none, and on Windows a child sharing a pipe its
                    # parent is blocked reading cannot even start.
                    self._process = subprocess.Popen(self._command, stdin=subprocess.DEVNULL, stderr=log)
                self._started = time.time()
                self._launch_failed = False
            except OSError as exc:
                self._process = None
                self._failed_starts += 1
                self._failure = f"the worker process could not be launched ({exc!r}): {self._command!r}"
                self._launch_failed = True

    def _record_exit(self, process: subprocess.Popen) -> None:
        """A child that exited with an error having served no run is a failed start; any other
        exit ends a run of them."""
        self._process = None
        status = process.returncode
        if status not in (0, EXIT_HOST_OWNED) and not self.queue.claimed_since(self._started):
            self._failed_starts += 1
            self._failure = f"it exited with status {status} without serving a run; its output ends:\n{self._log_tail()}"
        else:
            self._failed_starts, self._failure = 0, None

    def _log_tail(self) -> str:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "(no output)"
        return text[-LOG_TAIL_CHARACTERS:].strip() or "(no output)"

    def status(self) -> WorkerStatus:
        """Whether a worker serves the queue, and why not if that is known: the host's worker
        serves another queue (`host_conflict`), or the worker cannot start (I-77)."""
        with self._lock:
            if self._process is not None and self._process.poll() is not None:
                self._record_exit(self._process)
            failed, failure, launch_failed = self._failed_starts, self._failure, self._launch_failed
        owner = lock_owner(self.lock_path)
        conflict = _conflict(owner, self.queue.path)
        serving = owner is not None and conflict is None
        if conflict is not None:
            error = conflict
        elif not serving and (launch_failed or failed >= FAILED_STARTS_REPORTED):
            error = (f"the engine worker cannot start: {failed} failed start(s) running, the last because {failure}; "
                     f"runs wait in the queue until a worker starts (its output: {self.log_path})")
        else:
            error = None
        return WorkerStatus(serving=serving, failed_starts=failed, error=error)

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the child, if any (between or during runs; a run it was running is marked
        `interrupted` by the next worker)."""
        with self._lock:
            process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout)
