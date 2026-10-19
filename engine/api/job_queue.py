"""
The durable job queue between the HTTP API and the engine worker (decision A-14;
docs/planning/details/precision.md §11).

One SQLite file, one row per job: a run request (`RUN`) as `POST /runs` received it.

    pending --claim--> running --finish--> done
       |                       --fail----> failed (with a failure class)
       |                       --cancel at a phase boundary--> cancelled
       |                       --worker restarts--> interrupted
       +--cancel--> cancelled

The API inserts the request body exactly as it received it (`submit`), and reads rows back
(`get`); the engine worker (`engine.api.worker`) claims the oldest pending row, runs it and
writes the result document or the failure. A result may come with artifacts: arrays a request
asked for by reference rather than inline (decision A-17), each stored as numbered chunks of
bytes in the `artifacts` table, written in the same transaction as the result, so a `done` row
always has all of its chunks (`finish`, `artifact`). The file survives restarts of either
process, and every process that opens it (several uvicorn workers included) sees the same jobs.

**A retried submission is the same job** (I-89): a job submitted with an `idempotency_key`
records the key and the SHA-256 of its body. Submitting the key again returns the first job's
id when the body is the same and raises `IdempotencyConflict` when it is not; the key is unique
in the table, so two API processes racing on one key get one job.

**Cancelling** (I-91): a pending job is `cancelled` at once; a running one is flagged
(`cancel_requested`), and the worker stops it at its next phase boundary and marks it
`cancelled` (`cancel_running`).

**Retention** (I-76): `purge` deletes the terminal jobs that finished before a cutoff, with their
artifacts, but never a job submitted with `keep`; the worker applies the policy between jobs
(`retention_seconds`, from `JAX_RISK_RETENTION_DAYS`).

**The schema** is at version 3 (2026-10-18). Versions 1 and 2 (the per-route jobs of 2026-10-04
and 2026-10-07) are migrated in place in one `BEGIN IMMEDIATE` transaction (I-106), so a crash
or a second process mid-migration leaves the file at its old version, intact. Their rows keep
their kinds (`LEGACY_KINDS`): their results stay readable, and a job of theirs that had not run
is marked `interrupted` (its route was removed, so no worker runs it).

**One engine worker per host** (decision A-14, I-106): the worker holds `WorkerLock` on one path
per host (`worker_lock_path`: `JAX_RISK_WORKER_LOCK`, else a fixed path in the system temp
directory) for its lifetime, and records in it the queue it serves (`lock_owner`). The operating
system releases the lock when the process dies however it dies, so a crashed worker never leaves
a stale lock, and a worker that starts knows that any `running` row belongs to a predecessor that
is gone (`interrupt_running`). A queue whose host's worker serves another queue has no worker;
the API refuses to queue a run on it (`engine.api.routes`).

Each thread keeps one connection (the API calls from a thread pool, and a connection may not
cross threads): measured on Windows, a connection opened per call made each commit cost 2-5 ms
(closing a WAL database's last connection checkpoints it), a kept one 0.3-1 ms with full
durability (`synchronous=FULL`), and a look at an empty queue 5 us, so the idle worker can
poll every few milliseconds. WAL mode lets the API read while the worker writes; `timeout`
waits out a concurrent writer.

No JAX, no ORE: the API process and a redundant worker (which exits on finding the lock held)
import this module without starting the engine.
"""
import json
import os
import sqlite3
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

PENDING, RUNNING, DONE, FAILED, INTERRUPTED, CANCELLED = "pending", "running", "done", "failed", "interrupted", \
    "cancelled"
STATUSES = (PENDING, RUNNING, DONE, FAILED, INTERRUPTED, CANCELLED)
TERMINAL = (DONE, FAILED, INTERRUPTED, CANCELLED)

#: Why a job failed (I-08). Chosen by the worker from the exception's type
#: (`engine.api.worker.failure_class`).
BAD_TERMS = "bad-terms"
MISSING_MARKET_DATA = "missing-market-data"
UNSUPPORTED_PRODUCT = "unsupported-product"
NUMERICAL_FAILURE = "numerical-failure"
INFRASTRUCTURE = "infrastructure"
FAILURE_CLASSES = (BAD_TERMS, MISSING_MARKET_DATA, UNSUPPORTED_PRODUCT, NUMERICAL_FAILURE, INFRASTRUCTURE)

#: What a job runs: a run request (`POST /runs`, `engine.run.run`).
RUN = "run"
KINDS = (RUN,)
#: The kinds of the routes a run request replaced (2026-10-18), kept on the rows a version-2
#: queue held: nothing submits or runs them.
LEGACY_KINDS = ("portfolio", "market-risk")

INTERRUPTED_ERROR = ("the engine worker stopped while running this job (killed, crashed or restarted); "
                     "nothing was priced to completion: submit the request again")
RETIRED_ROUTE_ERROR = ("this job was submitted to a route that was removed on 2026-10-18 and never ran; "
                       "submit it as a run request (POST /runs)")
CANCELLED_PENDING = "cancelled before it started"

# `UPDATE ... RETURNING` makes the claim one atomic statement.
_MIN_SQLITE = (3, 35, 0)
SCHEMA_VERSION = 3


def _in(values) -> str:
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


_JOBS = f"""
CREATE TABLE {{name}} (
    seq              INTEGER PRIMARY KEY AUTOINCREMENT,  -- claim order
    id               TEXT    NOT NULL UNIQUE,
    kind             TEXT    NOT NULL CHECK (kind IN {_in(KINDS + LEGACY_KINDS)}),
    request          BLOB    NOT NULL,                   -- the HTTP body as received
    status           TEXT    NOT NULL CHECK (status IN {_in(STATUSES)}),
    failure_class    TEXT    CHECK (failure_class IS NULL OR failure_class IN {_in(FAILURE_CLASSES)}),
    error            TEXT,
    result           TEXT,                               -- the result document (JSON)
    worker           TEXT,                               -- host:pid of the worker that claimed it
    compiles         INTEGER,                            -- XLA programs the job had to build
    submitted        REAL    NOT NULL,
    started          REAL,
    finished         REAL,
    idempotency_key  TEXT    UNIQUE,                     -- the caller's key, if any (I-89)
    request_sha256   TEXT,                               -- the canonical body's hash, with a key
    cancel_requested INTEGER NOT NULL DEFAULT 0,         -- a cancel reached it while running (I-91)
    keep             INTEGER NOT NULL DEFAULT 0          -- exempt from retention (I-76)
)"""
_INDEXES = ("CREATE INDEX IF NOT EXISTS jobs_by_status ON jobs (status, seq)",
            "CREATE INDEX IF NOT EXISTS jobs_by_finished ON jobs (finished)")
_ARTIFACTS = """
CREATE TABLE IF NOT EXISTS artifacts (
    job_id        TEXT    NOT NULL REFERENCES jobs (id),
    name          TEXT    NOT NULL,                   -- e.g. npv_cube
    chunk         INTEGER NOT NULL,                   -- 0, 1, ... in order
    data          BLOB    NOT NULL,
    PRIMARY KEY (job_id, name, chunk)
)"""
#: The columns every version had; version 1's rows were portfolio jobs (it had no `kind`).
_V1_COLUMNS = "id, request, status, failure_class, error, result, worker, compiles, submitted, started, finished"


def migration_statements(version: int) -> Tuple[str, ...]:
    """The statements that bring a file at schema `version` (0: new) to `SCHEMA_VERSION`, run in
    one transaction by `JobQueue`. A version-1 or -2 file's table is rebuilt (SQLite cannot widen
    a CHECK in place): its rows copied with their kinds, a job of theirs that never ran marked
    `interrupted` (`RETIRED_ROUTE_ERROR`)."""
    if version == 0:
        return (_JOBS.format(name="jobs"), *_INDEXES, _ARTIFACTS)
    kind = "'portfolio'" if version == 1 else "kind"
    return (
        _JOBS.format(name="jobs_v3"),
        f"INSERT INTO jobs_v3 (seq, kind, {_V1_COLUMNS}) SELECT seq, {kind}, {_V1_COLUMNS} FROM jobs ORDER BY seq",
        "DROP INDEX IF EXISTS jobs_by_status",
        "DROP TABLE jobs",
        "ALTER TABLE jobs_v3 RENAME TO jobs",
        *_INDEXES,
        _ARTIFACTS,
        f"UPDATE jobs SET status = '{INTERRUPTED}', error = '{RETIRED_ROUTE_ERROR}', finished = submitted "
        f"WHERE status IN ('{PENDING}', '{RUNNING}')",
    )


def default_queue_path() -> Path:
    """`JAX_RISK_JOB_QUEUE` if set, else `jax-risk-jobs/jobs.sqlite3` under the system temp
    directory (the TraderX path's store convention, `engine.traderx.publication.default_store_root`:
    not the working directory, so the queue does not depend on where a process started)."""
    configured = os.environ.get("JAX_RISK_JOB_QUEUE")
    if configured:
        return Path(configured)
    return Path(tempfile.gettempdir()) / "jax-risk-jobs" / "jobs.sqlite3"


#: `JAX_RISK_RETENTION_DAYS`, unset: how long a finished job is kept (I-76; an owner may choose
#: another, see docs/reference/http-api.md).
DEFAULT_RETENTION_DAYS = 30.0
#: `JAX_RISK_RETENTION_DAYS=off`: finished jobs are kept forever.
RETENTION_OFF = "off"


def retention_seconds(environ: Optional[Mapping[str, str]] = None) -> Optional[float]:
    """How long a finished job is kept before the worker purges it, from `JAX_RISK_RETENTION_DAYS`
    (a positive number of days; `off` keeps every job; unset, `DEFAULT_RETENTION_DAYS`), or None
    for forever. A job submitted with `keep` is never purged."""
    value = (os.environ if environ is None else environ).get("JAX_RISK_RETENTION_DAYS", "").strip()
    if value.lower() == RETENTION_OFF:
        return None
    try:
        days = float(value) if value else DEFAULT_RETENTION_DAYS
    except ValueError:
        days = float("nan")
    if not days > 0.0 or days == float("inf"):
        raise ValueError(f"JAX_RISK_RETENTION_DAYS={value!r}: give a positive number of days, or {RETENTION_OFF!r}")
    return days * 86400.0


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
    cancel_requested: bool = False
    keep: bool = False


_COLUMN_NAMES = ("id", "kind", "status", "request", "failure_class", "error", "result", "worker", "compiles",
                 "submitted", "started", "finished", "cancel_requested", "keep")
_COLUMNS = ", ".join(_COLUMN_NAMES)
#: `_COLUMNS` without the two that can be large: the request and the result document.
_SUMMARY_COLUMNS = ", ".join("NULL" if c in ("request", "result") else c for c in _COLUMN_NAMES)


def _job(row) -> Job:
    *head, cancel_requested, keep = row
    return Job(*head, cancel_requested=bool(cancel_requested), keep=bool(keep))


class IdempotencyConflict(ValueError):
    """An `idempotency_key` submitted again with another body (I-89): the job it names is not
    the one asked for."""

    def __init__(self, key: str, job_id: str):
        super().__init__(f"idempotency_key {key!r} was submitted with another body (run {job_id}); use a new key for "
                         f"a new run")
        self.job_id = job_id


class JobQueue:
    """The queue file at `path`, created with its directory on first use, and migrated to
    `SCHEMA_VERSION` in one transaction."""

    def __init__(self, path, timeout: float = 30.0):
        if sqlite3.sqlite_version_info < _MIN_SQLITE:
            raise RuntimeError(f"the job queue needs SQLite >= {'.'.join(map(str, _MIN_SQLITE))}; "
                               f"this Python has {sqlite3.sqlite_version}")
        self.path = Path(path)
        self._timeout = timeout
        self._local = threading.local()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._use_wal(self._connection())
        with self._transaction() as con:
            # Read inside the transaction: a second process migrating at once waits for the first,
            # then finds the file at the current version.
            version = con.execute("PRAGMA user_version").fetchone()[0]
            if version != SCHEMA_VERSION:
                if version not in (0, 1, 2):
                    raise RuntimeError(f"{self.path}: job queue schema version {version}, this engine reads "
                                       f"{SCHEMA_VERSION} (and migrates 1 and 2); move the file aside")
                for statement in migration_statements(version):
                    con.execute(statement)
                con.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def _use_wal(self, con: sqlite3.Connection) -> None:
        """Put the file in WAL mode, which is persistent (stored in the file) and cannot be set
        inside a transaction. Switching takes an exclusive lock, and SQLite answers `database is
        locked` at once rather than waiting out the busy timeout when another process holds the
        file (found 2026-10-18: processes opening a new or old file together), so a file already
        in WAL is left as it is, and a switch is retried until the timeout."""
        deadline = time.monotonic() + self._timeout
        while con.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
            try:
                con.execute("PRAGMA journal_mode=WAL")
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc) or time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)

    def _connection(self) -> sqlite3.Connection:
        """This thread's connection, opened on first use. Autocommit: every statement below is
        its own transaction, but those of `_transaction`."""
        con = getattr(self._local, "connection", None)
        if con is None:
            con = self._local.connection = sqlite3.connect(self.path, timeout=self._timeout, isolation_level=None)
        return con

    @contextmanager
    def _transaction(self):
        """One `BEGIN IMMEDIATE` transaction on this thread's connection: committed if the block
        completes, rolled back if it raises."""
        con = self._connection()
        con.execute("BEGIN IMMEDIATE")
        try:
            yield con
        except BaseException:
            con.execute("ROLLBACK")
            raise
        con.execute("COMMIT")

    def _execute(self, sql: str, params=()) -> list:
        return self._connection().execute(sql, params).fetchall()

    def close(self) -> None:
        """Close this thread's connection (others close with their threads)."""
        con, self._local.connection = getattr(self._local, "connection", None), None
        if con is not None:
            con.close()

    def submit(self, request: bytes, kind: str = RUN, *, idempotency_key: Optional[str] = None,
               request_sha256: Optional[str] = None, keep: bool = False) -> str:
        """Queue `request` (the HTTP body) as a job of `kind`; the new job's id. With an
        `idempotency_key` (and the body's `request_sha256`), the id of the job already submitted
        with that key and body, or `IdempotencyConflict` for another body (I-89)."""
        if kind not in KINDS:
            raise ValueError(f"job kind {kind!r} not one of {KINDS}")
        if idempotency_key is not None and request_sha256 is None:
            raise ValueError("an idempotency_key needs the request's sha256")
        job_id = str(uuid.uuid4())
        with self._transaction() as con:
            if idempotency_key is not None:
                row = con.execute("SELECT id, request_sha256 FROM jobs WHERE idempotency_key = ?",
                                  (idempotency_key,)).fetchone()
                if row is not None:
                    if row[1] != request_sha256:
                        raise IdempotencyConflict(idempotency_key, row[0])
                    return row[0]
            con.execute("INSERT INTO jobs (id, kind, request, status, submitted, idempotency_key, request_sha256, keep) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (job_id, kind, bytes(request), PENDING, time.time(), idempotency_key, request_sha256,
                         int(bool(keep))))
        return job_id

    def get(self, job_id: str) -> Optional[Job]:
        rows = self._execute(f"SELECT {_COLUMNS} FROM jobs WHERE id = ?", (job_id,))
        return _job(rows[0]) if rows else None

    def status(self, job_id: str) -> Optional[str]:
        """The status alone, without reading the request or the result document."""
        rows = self._execute("SELECT status FROM jobs WHERE id = ?", (job_id,))
        return rows[0][0] if rows else None

    def summary(self, job_id: str) -> Optional[Job]:
        """The row without its request and result document (a cube can run to megabytes)."""
        rows = self._execute(f"SELECT {_SUMMARY_COLUMNS} FROM jobs WHERE id = ?", (job_id,))
        return _job(rows[0]) if rows else None

    def claim(self, worker: str) -> Optional[Job]:
        """Mark the oldest pending job `running` for `worker` and return it, or None if none is
        pending. One statement, so two claimers can never take the same row."""
        rows = self._execute(
            "UPDATE jobs SET status = ?, worker = ?, started = ? "
            "WHERE seq = (SELECT seq FROM jobs WHERE status = ? ORDER BY seq LIMIT 1) "
            f"RETURNING {_COLUMNS}",
            (RUNNING, worker, time.time(), PENDING))
        return _job(rows[0]) if rows else None

    def claimed_since(self, since: float) -> bool:
        """Whether a job was claimed at or after `since` (seconds since the epoch): the
        supervisor's test that a worker it started served something (I-77)."""
        return bool(self._execute("SELECT 1 FROM jobs WHERE started >= ? LIMIT 1", (since,)))

    def finish(self, job_id: str, result: str, compiles: Optional[int] = None,
               artifacts: Optional[Mapping[str, Sequence[bytes]]] = None) -> None:
        """Mark the running job `done` with its result document and its artifacts (name ->
        chunks), in one transaction: a reader never sees the row done without every chunk."""
        with self._transaction() as con:
            for name, chunks in (artifacts or {}).items():
                con.executemany("INSERT INTO artifacts (job_id, name, chunk, data) VALUES (?, ?, ?, ?)",
                                [(job_id, name, i, bytes(data)) for i, data in enumerate(chunks)])
            self._close_running(job_id, DONE, result=result, compiles=compiles)

    def artifact(self, job_id: str, name: str, chunk: int) -> Optional[bytes]:
        """Chunk `chunk` of the job's artifact `name`, or None if there is no such chunk."""
        rows = self._execute("SELECT data FROM artifacts WHERE job_id = ? AND name = ? AND chunk = ?",
                             (job_id, name, chunk))
        return rows[0][0] if rows else None

    def fail(self, job_id: str, failure_class: str, error: str, compiles: Optional[int] = None) -> None:
        if failure_class not in FAILURE_CLASSES:
            raise ValueError(f"failure class {failure_class!r} not one of {FAILURE_CLASSES}")
        self._close_running(job_id, FAILED, failure_class=failure_class, error=error, compiles=compiles)

    def cancel(self, job_id: str) -> Optional[str]:
        """Cancel the job (I-91): a pending one is `cancelled` at once; a running one is flagged
        for the worker to stop at its next phase boundary; a finished one is left as it is. The
        job's status after the call, or None for an unknown id."""
        with self._transaction() as con:
            row = con.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                return None
            if row[0] == PENDING:
                con.execute("UPDATE jobs SET status = ?, error = ?, finished = ? WHERE id = ?",
                            (CANCELLED, CANCELLED_PENDING, time.time(), job_id))
                return CANCELLED
            if row[0] == RUNNING:
                con.execute("UPDATE jobs SET cancel_requested = 1 WHERE id = ?", (job_id,))
            return row[0]

    def cancel_requested(self, job_id: str) -> bool:
        """Whether a cancel has reached the running job (the worker's check at each phase)."""
        rows = self._execute("SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,))
        return bool(rows and rows[0][0])

    def cancel_running(self, job_id: str, error: str, compiles: Optional[int] = None) -> None:
        """Mark the running job `cancelled`, stopped at a phase boundary (`error` says which)."""
        self._close_running(job_id, CANCELLED, error=error, compiles=compiles)

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
        has just taken the host's lock, when no other worker can be running a job."""
        return len(self._execute(
            "UPDATE jobs SET status = ?, error = ?, finished = ? WHERE status = ? RETURNING id",
            (INTERRUPTED, INTERRUPTED_ERROR, time.time(), RUNNING)))

    def purge(self, finished_before: float) -> int:
        """Delete every finished job that finished before `finished_before` (seconds since the
        epoch), with its artifacts, but those submitted with `keep` (I-76); the number deleted.
        One transaction, so no reader sees a job without its chunks. SQLite reuses the freed
        pages; the file itself does not shrink."""
        where = f"status IN {_in(TERMINAL)} AND keep = 0 AND finished < ?"
        with self._transaction() as con:
            con.execute(f"DELETE FROM artifacts WHERE job_id IN (SELECT id FROM jobs WHERE {where})",
                        (finished_before,))
            return con.execute(f"DELETE FROM jobs WHERE {where}", (finished_before,)).rowcount


# --- the host's one engine worker ---------------------------------------------------------------

def worker_lock_path(environ: Optional[Mapping[str, str]] = None) -> Path:
    """The engine worker's lock, one per host (I-106): `JAX_RISK_WORKER_LOCK` if set (another
    path for another set of devices, or for a test of its own), else `jax-risk-engine/worker.lock`
    in the system temp directory."""
    configured = (os.environ if environ is None else environ).get("JAX_RISK_WORKER_LOCK")
    if configured:
        return Path(configured)
    return Path(tempfile.gettempdir()) / "jax-risk-engine" / "worker.lock"


class WorkerLock:
    """The exclusive lock that makes a process its host's one engine worker: an OS file lock on
    byte 0 of the file at `path`, released by the OS when the holder exits. `fcntl.flock` on
    POSIX, `msvcrt.locking` on Windows; both refuse a second holder in another process. The
    holder writes after byte 0 who it is and the queue it serves (`lock_owner`)."""

    def __init__(self, path):
        self.path = Path(path)
        self._file = None

    def acquire(self, wait_seconds: float = 0.0, retry_seconds: float = 0.05, owner: Optional[dict] = None) -> bool:
        """Take the lock, retrying for up to `wait_seconds`; whether it was taken. `owner` is
        recorded for `lock_owner` once it is."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + wait_seconds
        file = os.fdopen(os.open(self.path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)), "r+b")
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
                if owner is not None:
                    file.seek(1)
                    file.truncate()
                    file.write(json.dumps(owner).encode("utf-8"))
                    file.flush()
                return True

    def release(self) -> None:
        file, self._file = self._file, None
        if file is not None:
            _unlock(file)
            file.close()


def worker_lock_held(path) -> bool:
    """Whether some process holds the worker lock at `path`, i.e. an engine worker serves this
    host. Takes and drops the lock to find out, so a worker starting at that instant must retry
    (`WorkerLock.acquire(wait_seconds=...)`)."""
    probe = WorkerLock(path)
    if probe.acquire():
        probe.release()
        return False
    return True


def lock_owner(path) -> Optional[dict]:
    """What the holder of the worker lock at `path` recorded (`{"queue", "host", "pid"}`), or
    None if no process holds it or it has not recorded itself yet."""
    if not worker_lock_held(path):
        return None
    try:
        with open(path, "rb") as file:
            file.seek(1)
            recorded = file.read()
        return json.loads(recorded.decode("utf-8")) if recorded else None
    except (OSError, ValueError):
        return None


def same_queue(a, b) -> bool:
    """Whether two queue paths name one file."""
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return False


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
