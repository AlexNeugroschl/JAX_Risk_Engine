"""
The durable job queue between the HTTP API and the engine worker (decision A-14;
docs/planning/details/precision.md §11).

One SQLite file, one row per job, of a kind (`KINDS`: a portfolio pricing or a market-risk run):

    pending --claim--> running --finish--> done
                               --fail----> failed (with a failure class)
                               --worker restarts--> interrupted

The API inserts the request body exactly as it received it, with its kind (`submit`), and reads
rows back (`get`); the engine worker (`engine.api.worker`) claims the oldest pending row, runs it
and writes the result document or the failure. A result may come with artifacts: arrays a request
asked for by reference rather than inline (decision A-17), each stored as numbered chunks of
bytes in the `artifacts` table, written in the same transaction as the result, so a `done` row
always has all of its chunks (`finish`, `artifact`). The file survives restarts of either
process, and every process that opens it (several uvicorn workers included) sees the same jobs.

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
from typing import Mapping, Optional, Sequence, Tuple

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

#: What a job runs: a portfolio pricing (`POST /portfolio/price`) or a market-risk run
#: (`POST /portfolio/market-risk`); the worker picks its pricer by it.
PORTFOLIO, MARKET_RISK = "portfolio", "market-risk"
KINDS = (PORTFOLIO, MARKET_RISK)

INTERRUPTED_ERROR = ("the engine worker stopped while running this job (killed, crashed or restarted); "
                     "nothing was priced to completion: submit the request again")

# `UPDATE ... RETURNING` makes the claim one atomic statement.
_MIN_SQLITE = (3, 35, 0)
_SCHEMA_VERSION = 2


def _in(values) -> str:
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS jobs (
    seq           INTEGER PRIMARY KEY AUTOINCREMENT,  -- claim order
    id            TEXT    NOT NULL UNIQUE,
    kind          TEXT    NOT NULL DEFAULT '{PORTFOLIO}' CHECK (kind IN {_in(KINDS)}),
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
CREATE TABLE IF NOT EXISTS artifacts (
    job_id        TEXT    NOT NULL REFERENCES jobs (id),
    name          TEXT    NOT NULL,                   -- e.g. npv_cube
    chunk         INTEGER NOT NULL,                   -- 0, 1, ... in order
    data          BLOB    NOT NULL,
    PRIMARY KEY (job_id, name, chunk)
);
"""

# Schema version 1 (2026-10-04) had portfolio jobs only and no artifacts: its rows are
# portfolio jobs (the column's default), and the artifacts table is created with the schema.
_MIGRATE_FROM_1 = f"ALTER TABLE jobs ADD COLUMN kind TEXT NOT NULL DEFAULT '{PORTFOLIO}' CHECK (kind IN {_in(KINDS)})"


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
    kind: str
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


_COLUMNS = "id, kind, status, request, failure_class, error, result, worker, compiles, submitted, started, finished"


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
        if version not in (0, 1, _SCHEMA_VERSION):
            raise RuntimeError(f"{self.path}: job queue schema version {version}, this engine reads "
                               f"{_SCHEMA_VERSION} (and migrates 1); move the file aside")
        con.execute("PRAGMA journal_mode=WAL")  # persistent: stored in the file
        if version == 1:
            con.execute(_MIGRATE_FROM_1)
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

    def submit(self, request: bytes, kind: str = PORTFOLIO) -> str:
        """Queue `request` (the HTTP body) as a job of `kind`; returns the new job's id."""
        if kind not in KINDS:
            raise ValueError(f"job kind {kind!r} not one of {KINDS}")
        job_id = str(uuid.uuid4())
        self._execute("INSERT INTO jobs (id, kind, request, status, submitted) VALUES (?, ?, ?, ?, ?)",
                      (job_id, kind, bytes(request), PENDING, time.time()))
        return job_id

    def get(self, job_id: str) -> Optional[Job]:
        rows = self._execute(f"SELECT {_COLUMNS} FROM jobs WHERE id = ?", (job_id,))
        return Job(*rows[0]) if rows else None

    def status(self, job_id: str) -> Optional[str]:
        """The status alone, without reading the request or the result document."""
        state = self.state(job_id)
        return state[1] if state else None

    def state(self, job_id: str) -> Optional[Tuple[str, str]]:
        """`(kind, status)`, without reading the request or the result document."""
        rows = self._execute("SELECT kind, status FROM jobs WHERE id = ?", (job_id,))
        return tuple(rows[0]) if rows else None

    def claim(self, worker: str) -> Optional[Job]:
        """Mark the oldest pending job `running` for `worker` and return it, or None if none is
        pending. One statement, so two claimers can never take the same row."""
        rows = self._execute(
            "UPDATE jobs SET status = ?, worker = ?, started = ? "
            "WHERE seq = (SELECT seq FROM jobs WHERE status = ? ORDER BY seq LIMIT 1) "
            f"RETURNING {_COLUMNS}",
            (RUNNING, worker, time.time(), PENDING))
        return Job(*rows[0]) if rows else None

    def finish(self, job_id: str, result: str, compiles: Optional[int] = None,
               artifacts: Optional[Mapping[str, Sequence[bytes]]] = None) -> None:
        """Mark the running job `done` with its result document and its artifacts (name ->
        chunks), in one transaction: a reader never sees the row done without every chunk."""
        con = self._connection()
        con.execute("BEGIN IMMEDIATE")
        try:
            for name, chunks in (artifacts or {}).items():
                con.executemany("INSERT INTO artifacts (job_id, name, chunk, data) VALUES (?, ?, ?, ?)",
                                [(job_id, name, i, bytes(data)) for i, data in enumerate(chunks)])
            self._close_running(job_id, DONE, result=result, compiles=compiles)
        except BaseException:
            con.execute("ROLLBACK")
            raise
        con.execute("COMMIT")

    def artifact(self, job_id: str, name: str, chunk: int) -> Optional[bytes]:
        """Chunk `chunk` of the job's artifact `name`, or None if there is no such chunk."""
        rows = self._execute("SELECT data FROM artifacts WHERE job_id = ? AND name = ? AND chunk = ?",
                             (job_id, name, chunk))
        return rows[0][0] if rows else None

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
