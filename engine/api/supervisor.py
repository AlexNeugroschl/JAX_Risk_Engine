"""
Keeping one engine worker (`engine.api.worker`) alive behind the API (roadmap 1.8).

`JAX_RISK_WORKER` chooses who runs the worker:

  - `spawn` (the default): the API process starts the worker as a child process on the
    first job and checks it on every job submission and poll, restarting it if it has died.
    No thread watches it: a dead worker matters only when someone submits or polls, which is
    exactly when it is checked. The worker exits on its own when the API process is gone.
  - `external`: something else runs `jax-risk-worker` (systemd `Restart=always`, a container
    restart policy, a pod's process manager) and the API only reads and writes the queue.

Several API processes on one queue (uvicorn `--workers N`) share one worker: a supervisor
starts its child only if no process holds the queue's worker lock, and a redundant child
exits at once (`EXIT_QUEUE_OWNED`), so at most one worker prices at a time.

The child is started with `subprocess` (a fresh interpreter), never `fork`: forking a process
that has initialized JAX hangs (I-33).
"""
import os
import subprocess
import sys
import threading
from typing import Optional, Sequence

from engine.api.job_queue import worker_lock_held

SPAWN, EXTERNAL = "spawn", "external"


def worker_mode() -> str:
    """`JAX_RISK_WORKER`: `spawn` (default) or `external`."""
    mode = os.environ.get("JAX_RISK_WORKER", SPAWN)
    if mode not in (SPAWN, EXTERNAL):
        raise ValueError(f"JAX_RISK_WORKER={mode!r}: expected {SPAWN!r} or {EXTERNAL!r}")
    return mode


def worker_command(queue_path) -> list:
    """The command line of a worker for `queue_path`, watching this process."""
    return [sys.executable, "-c", "from engine.api.worker import main; main()",
            "--queue", str(queue_path), "--parent-pid", str(os.getpid())]


class WorkerSupervisor:
    """Starts and restarts the engine worker of `queue_path` as a child of this process.
    `command` overrides the worker's command line (tests run a stub pricer through it)."""

    def __init__(self, queue_path, command: Optional[Sequence[str]] = None):
        self.queue_path = queue_path
        self._command = list(command) if command is not None else worker_command(queue_path)
        self._process: Optional[subprocess.Popen] = None
        # Routes run on the server's thread pool; one check-and-start at a time.
        self._lock = threading.Lock()

    @property
    def process(self) -> Optional[subprocess.Popen]:
        """The child started last, alive or not; None before the first start."""
        return self._process

    def ensure_running(self) -> None:
        """Start the worker unless this process's child is alive or another process's worker
        holds the queue."""
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return
            if worker_lock_held(self.queue_path):
                return
            # No stdin: the worker reads none, and on Windows a child sharing a pipe its parent
            # is blocked reading cannot even start.
            self._process = subprocess.Popen(self._command, stdin=subprocess.DEVNULL)

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the child, if any (between or during jobs; a job it was running is marked
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
