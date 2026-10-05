"""
The durable job queue between the HTTP API and the engine worker (decision A-14, roadmap 1.8;
docs/planning/details/precision.md §11).

One SQLite file, one row per portfolio job:

    pending --claim--> running --finish--> done
                               --fail----> failed (with a failure class)
                               --worker restarts--> interrupted

The API inserts the request body exactly as it received it (`submit`) and reads rows back
(`get`); the engine worker (`engine.api.worker`) claims the oldest pending row, prices it and
writes the result document or the failure. The file survives restarts of either process, and
every process that opens it (several uvicorn workers included) sees the same jobs.

One engine worker per queue: the worker holds `WorkerLock` on `<queue>.worker.lock` for its
lifetime. The operating system releases the lock when the process dies however it dies, so
a crashed worker never leaves a stale lock, and a worker that starts knows that any `running`
row belongs to a predecessor that is gone (`interrupt_running`).

Each thread keeps one connection (the API calls from a thread pool, and a connection may not
cross threads): measured on Windows, a connection opened per call made each commit cost 2-5 ms
(closing a WAL database's last connection checkpoints it), a kept one 0.3-1 ms with full
durability (`synchronous=FULL`), and a look at an empty queue 5 us, so the idle worker can
poll every few milliseconds. WAL mode lets the API read while the worker writes; `timeout`
waits out a concurrent writer.

No JAX, no ORE: the API process and a redundant worker (which exits on finding the lock held)
import this module without starting the engine.
"""
import os
import sqlite3
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PENDING, RUNNING, DONE, FAILED, INTERRUPTED = "pending", "running", "done", "failed", "interrupted"
STATUSES = (PENDING, RUNNING, DONE, FAILED, INTERRUPTED)
TERMINAL = (DONE, FAILED, INTERRUPTED)

#: Why a job failed (I-08). Chosen by the worker from the exception's type
#: (`engine.api.worker.failure_class`).
BAD_TERMS = "bad-terms"
MISSING_MARKET_DATA = "missing-market-data"
UNSUPPORTED_PRODUCT = "unsupported-product"
NUMERICAL_FAILURE = "numerical-failure"
INFRASTRUCTURE = "infrastructure"
FAILURE_CLASSES = (BAD_TERMS, MISSING_MARKET_DATA, UNSUPPORTED_PRODUCT, NUMERICAL_FAILURE, INFRASTRUCTURE)

INTERRUPTED_ERROR = ("the engine worker stopped while running this job (killed, crashed or restarted); "
                     "nothing was priced to completion: submit the request again")

# `UPDATE ... RETURNING` makes the claim one atomic statement.
_MIN_SQLITE = (3, 35, 0)
_SCHEMA_VERSION = 1


def _in(values) -> str:
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS jobs (
    seq           INTEGER PRIMARY KEY AUTOINCREMENT,  -- claim order
    id            TEXT    NOT NULL UNIQUE,
    request       BLOB    NOT NULL,                   -- the HTTP body as received
    status        TEXT    NOT NULL CHECK (status IN {_in(STATUSES)}),
    failure_class TEXT    CHECK (failure_class IS NULL OR failure_class IN {_in(FAILURE_CLASSES)}),
    error         TEXT,
    result        TEXT,                               -- the result document (JSON)
    worker        TEXT,                               -- host:pid of the worker that claimed it
    compiles      INTEGER,                            -- XLA programs the job had to build
    submitted     REAL    NOT NULL,
    started       REAL,
    finished      REAL
);
CREATE INDEX IF NOT EXISTS jobs_by_status ON jobs (status, seq);
"""


def default_queue_path() -> Path:
    """`JAX_RISK_JOB_QUEUE` if set, else `jax-risk-jobs/jobs.sqlite3` under the system temp
    directory (the EOD store's convention, `engine.integration.publication.default_store_root`:
    not the working directory, so the queue does not depend on where a process started)."""
    configured = os.environ.get("JAX_RISK_JOB_QUEUE")
    if configured:
        return Path(configured)
    return Path(tempfile.gettempdir()) / "jax-risk-jobs" / "jobs.sqlite3"


@dataclass(frozen=True)
class Job:
    """One row. `request` is the HTTP body; `result` the result document, set when `done`."""
    id: str
    status: str
    request: bytes
    failure_class: Optional[str] = None
    error: Optional[str] = None
    result: Optional[str] = None
    worker: Optional[str] = None
    compiles: Optional[int] = None
    submitted: Optional[float] = None
    started: Optional[float] = None
    finished: Optional[float] = None


_COLUMNS = "id, status, request, failure_class, error, result, worker, compiles, submitted, started, finished"


class JobQueue:
    """The queue file at `path`, created with its directory on first use."""

    def __init__(self, path, timeout: float = 30.0):
        if sqlite3.sqlite_version_info < _MIN_SQLITE:
            raise RuntimeError(f"the job queue needs SQLite >= {'.'.join(map(str, _MIN_SQLITE))}; "
                               f"this Python has {sqlite3.sqlite_version}")
        self.path = Path(path)
        self._timeout = timeout
        self._local = threading.local()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = self._connection()
        version = con.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, _SCHEMA_VERSION):
            raise RuntimeError(f"{self.path}: job queue schema version {version}, this engine reads "
                               f"{_SCHEMA_VERSION}; move the file aside")
        con.execute("PRAGMA journal_mode=WAL")  # persistent: stored in the file
        con.executescript(_SCHEMA)
        con.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")

    def _connection(self) -> sqlite3.Connection:
        """This thread's connection, opened on first use. Autocommit: every statement below is
        its own transaction."""
        con = getattr(self._local, "connection", None)
        if con is None:
            con = self._local.connection = sqlite3.connect(self.path, timeout=self._timeout, isolation_level=None)
        return con

    def _execute(self, sql: str, params=()) -> list:
        return self._connection().execute(sql, params).fetchall()

    def close(self) -> None:
        """Close this thread's connection (others close with their threads)."""
        con, self._local.connection = getattr(self._local, "connection", None), None
        if con is not None:
            con.close()

    def submit(self, request: bytes) -> str:
        """Queue `request` (the HTTP body); returns the new job's id."""
        job_id = str(uuid.uuid4())
        self._execute("INSERT INTO jobs (id, request, status, submitted) VALUES (?, ?, ?, ?)",
                      (job_id, bytes(request), PENDING, time.time()))
        return job_id

    def get(self, job_id: str) -> Optional[Job]:
        rows = self._execute(f"SELECT {_COLUMNS} FROM jobs WHERE id = ?", (job_id,))
        return Job(*rows[0]) if rows else None

    def status(self, job_id: str) -> Optional[str]:
        """The status alone, without reading the result document."""
        rows = self._execute("SELECT status FROM jobs WHERE id = ?", (job_id,))
        return rows[0][0] if rows else None

    def claim(self, worker: str) -> Optional[Job]:
        """Mark the oldest pending job `running` for `worker` and return it, or None if none is
        pending. One statement, so two claimers can never take the same row."""
        rows = self._execute(
            "UPDATE jobs SET status = ?, worker = ?, started = ? "
            "WHERE seq = (SELECT seq FROM jobs WHERE status = ? ORDER BY seq LIMIT 1) "
            f"RETURNING {_COLUMNS}",
            (RUNNING, worker, time.time(), PENDING))
        return Job(*rows[0]) if rows else None

    def finish(self, job_id: str, result: str, compiles: Optional[int] = None) -> None:
        self._close_running(job_id, DONE, result=result, compiles=compiles)

    def fail(self, job_id: str, failure_class: str, error: str, compiles: Optional[int] = None) -> None:
        if failure_class not in FAILURE_CLASSES:
            raise ValueError(f"failure class {failure_class!r} not one of {FAILURE_CLASSES}")
        self._close_running(job_id, FAILED, failure_class=failure_class, error=error, compiles=compiles)

    def _close_running(self, job_id: str, status: str, *, result=None, failure_class=None, error=None,
                       compiles=None) -> None:
        rows = self._execute(
            "UPDATE jobs SET status = ?, result = ?, failure_class = ?, error = ?, compiles = ?, finished = ? "
            "WHERE id = ? AND status = ? RETURNING id",
            (status, result, failure_class, error, compiles, time.time(), job_id, RUNNING))
        if not rows:
            raise RuntimeError(f"job {job_id} is not running; cannot mark it {status}")

    def interrupt_running(self) -> int:
        """Mark every `running` job `interrupted`; the number marked. Called by a worker that
        has just taken the queue's lock, when no other worker can be running a job."""
        return len(self._execute(
            "UPDATE jobs SET status = ?, error = ?, finished = ? WHERE status = ? RETURNING id",
            (INTERRUPTED, INTERRUPTED_ERROR, time.time(), RUNNING)))


class WorkerLock:
    """The exclusive lock that makes a process the queue's one engine worker: an OS file lock
    on `<queue>.worker.lock`, released by the OS when the holder exits. `fcntl.flock` on POSIX,
    `msvcrt.locking` on Windows; both refuse a second holder in another process."""

    def __init__(self, queue_path):
        self.path = Path(f"{queue_path}.worker.lock")
        self._file = None

    def acquire(self, wait_seconds: float = 0.0, retry_seconds: float = 0.05) -> bool:
        """Take the lock, retrying for up to `wait_seconds`; whether it was taken."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + wait_seconds
        file = open(self.path, "a+b")
        while True:
            try:
                _lock(file)
            except OSError:
                if time.monotonic() >= deadline:
                    file.close()
                    return False
                time.sleep(retry_seconds)
            else:
                self._file = file
                return True

    def release(self) -> None:
        file, self._file = self._file, None
        if file is not None:
            _unlock(file)
            file.close()


def worker_lock_held(queue_path) -> bool:
    """Whether some process holds the queue's worker lock, i.e. an engine worker serves it.
    Takes and drops the lock to find out, so a worker starting at that instant must retry
    (`WorkerLock.acquire(wait_seconds=...)`)."""
    probe = WorkerLock(queue_path)
    if probe.acquire():
        probe.release()
        return False
    return True


if os.name == "nt":
    import msvcrt

    def _lock(file) -> None:
        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(file) -> None:
        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _lock(file) -> None:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(file) -> None:
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)
