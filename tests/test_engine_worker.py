"""
The engine worker behind the HTTP API (decision A-14;
docs/planning/details/precision.md §11 and §13.8): the durable job queue
(`engine.api.job_queue`), the single-threaded worker (`engine.api.worker`) and its supervisor
(`engine.api.supervisor`).

  * The queue keeps every job across reopenings, claims in submission order, never hands one
    row to two claimers, and closes a row only from `running`. A version-1 or -2 file is
    migrated in one transaction: interrupted, it is left as it was (I-106).
  * A retried submission is the same job; the same key with another body is refused, also
    between two API processes on one file (I-89).
  * Cancelling: a pending job at once, a running one at its next phase boundary, the worker
    going on to the next job (I-91). Retention: finished jobs older than the period are purged
    with their artifacts, a kept one never (I-76).
  * One worker per host: the lock refuses a second holder, is free once its holder dies, and
    records the queue it serves; a queue whose host's worker serves another is refused (I-106).
  * A failing job fails only its own row, with a failure class, and the worker goes on; a
    worker killed mid-job leaves `interrupted`, and the next worker prices what is queued.
  * The supervisor restarts a dead worker, starts none while another process's worker serves
    the host, reports a worker that cannot start, and a worker whose parent is gone exits (I-77).
  * Real pricing: runs queued together give the bits they give run one after another, and the
    in-process run's; a second identical run compiles nothing; the worker runs the very request
    the route validated.

The process-level tests run the real loop around a stub pricer (`tests/support/worker_stubs.py`);
the HTTP contract over the worker is tests/test_api.py.
"""
import json
import sqlite3
import subprocess
import sys
import threading
import time
from typing import get_args

import numpy as np
import pytest

from engine.api import job_queue as jq
from engine.api import worker
from engine.api.job_queue import JobQueue, WorkerLock, lock_owner, worker_lock_held
from engine.api.supervisor import FAILED_STARTS_REPORTED, WorkerSupervisor
from tests.support.runs import models, request_of, run_body, simulation
from tests.support.worker_stubs import ROOT, echo_pricer, failing_worker_command, stub_worker_command


def _body(**fields) -> bytes:
    return json.dumps(fields).encode()


def _wait_for(predicate, timeout=60.0, every=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(every)
    raise AssertionError(f"timed out after {timeout}s waiting for {predicate}")


def _dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


@pytest.fixture
def queue(tmp_path) -> JobQueue:
    return JobQueue(tmp_path / "jobs.sqlite3")


@pytest.fixture
def lock(tmp_path):
    """A host lock of the test's own, so no test meets the session's engine worker."""
    return tmp_path / "worker.lock"


def _serve(queue, lock, **kwargs) -> int:
    return worker.serve(queue.path, lock_path=lock, **{"price": echo_pricer, "drain": True, **kwargs})


#: Version 2's table (2026-10-07), as that engine created it.
_V2_SCHEMA = """
    CREATE TABLE jobs (seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE,
        kind TEXT NOT NULL DEFAULT 'portfolio' CHECK (kind IN ('portfolio', 'market-risk')),
        request BLOB NOT NULL, status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'done', 'failed',
        'interrupted')), failure_class TEXT, error TEXT, result TEXT, worker TEXT, compiles INTEGER,
        submitted REAL NOT NULL, started REAL, finished REAL);
    CREATE INDEX jobs_by_status ON jobs (status, seq);
    CREATE TABLE artifacts (job_id TEXT NOT NULL REFERENCES jobs (id), name TEXT NOT NULL, chunk INTEGER NOT NULL,
        data BLOB NOT NULL, PRIMARY KEY (job_id, name, chunk));
    INSERT INTO jobs (id, kind, request, status, result, submitted, finished)
        VALUES ('done-risk', 'market-risk', x'7b7d', 'done', '{"risk": {}}', 0, 1);
    INSERT INTO jobs (id, kind, request, status, submitted) VALUES ('queued', 'portfolio', x'7b7d', 'pending', 2);
    INSERT INTO artifacts VALUES ('done-risk', 'pnl', 0, x'00');
    PRAGMA user_version=2;"""


def _version(path) -> int:
    with sqlite3.connect(path) as con:
        return con.execute("PRAGMA user_version").fetchone()[0]


class TestJobQueue:
    def test_a_submitted_job_is_pending_with_its_body_byte_for_byte(self, queue):
        body = b'{"portfolio": [],  "market": {"asof": "2026-07-30"}}'
        job = queue.get(queue.submit(body))
        assert (job.status, job.kind, job.request, job.result, job.failure_class) == (
            jq.PENDING, jq.RUN, body, None, None)
        assert queue.status(job.id) == jq.PENDING and not job.keep and not job.cancel_requested
        summary = queue.summary(job.id)
        assert summary.request is None and summary.result is None and summary.status == jq.PENDING

    def test_an_unknown_id_has_no_row(self, queue):
        assert queue.get("nope") is None and queue.status("nope") is None and queue.summary("nope") is None

    def test_jobs_are_claimed_in_submission_order(self, queue):
        ids = [queue.submit(_body(n=n)) for n in range(3)]
        claimed = [queue.claim("w") for _ in range(4)]
        assert [job.id for job in claimed[:3]] == ids and claimed[3] is None
        assert all(queue.get(i).status == jq.RUNNING and queue.get(i).worker == "w" for i in ids)

    def test_a_row_closes_only_from_running(self, queue):
        job_id = queue.submit(_body())
        with pytest.raises(RuntimeError, match="not running"):
            queue.finish(job_id, "{}")
        queue.claim("w")
        queue.finish(job_id, '{"x": 1}', compiles=4)
        done = queue.get(job_id)
        assert (done.status, done.result, done.compiles) == (jq.DONE, '{"x": 1}', 4) and done.finished
        with pytest.raises(RuntimeError, match="not running"):
            queue.fail(job_id, jq.BAD_TERMS, "late")

    def test_only_a_run_is_submitted(self, queue):
        with pytest.raises(ValueError, match="job kind"):
            queue.submit(_body(), "portfolio")

    def test_a_result_and_its_artifacts_are_written_together(self, queue):
        """`finish` stores the chunks in the transaction that marks the row done: a job that
        cannot be finished leaves no chunk behind."""
        job_id = queue.submit(_body())
        with pytest.raises(RuntimeError, match="not running"):
            queue.finish(job_id, "{}", artifacts={"npv_cube": [b"orphan"]})
        assert queue.artifact(job_id, "npv_cube", 0) is None
        queue.claim("w")
        queue.finish(job_id, "{}", artifacts={"npv_cube": [b"a", b"bc"], "pnl": [b"d"]})
        assert [queue.artifact(job_id, "npv_cube", i) for i in (0, 1, 2)] == [b"a", b"bc", None]
        assert queue.artifact(job_id, "pnl", 0) == b"d" and queue.artifact("other", "pnl", 0) is None

    def test_a_failure_names_one_of_the_classes(self, queue):
        job_id = queue.submit(_body())
        queue.claim("w")
        with pytest.raises(ValueError, match="failure class"):
            queue.fail(job_id, "oops", "x")
        queue.fail(job_id, jq.MISSING_MARKET_DATA, "no curve")
        failed = queue.get(job_id)
        assert (failed.status, failed.failure_class, failed.error) == (jq.FAILED, jq.MISSING_MARKET_DATA, "no curve")

    def test_interrupting_touches_only_running_rows(self, queue):
        pending, running = queue.submit(_body()), queue.submit(_body())
        queue.claim("w")  # takes `pending` (the older)
        assert queue.interrupt_running() == 1
        assert queue.get(pending).status == jq.INTERRUPTED and queue.get(pending).error == jq.INTERRUPTED_ERROR
        assert queue.get(running).status == jq.PENDING

    def test_a_claim_counts_its_attempt(self, queue):
        job_id = queue.submit(_body())
        assert queue.get(job_id).attempts == 0
        assert queue.claim("w").attempts == 1 and queue.get(job_id).attempts == 1

    def test_a_worker_set_puts_its_running_job_back_once(self, queue):
        """A worker set of several hosts stops whole when one host fails (I-61): its job is put
        back to run again, claimed at most `MAX_ATTEMPTS` times; one a cancel had reached is
        cancelled instead; the rest are interrupted as on one host."""
        once, twice, cancelled = (queue.submit(_body(n=n)) for n in range(3))
        for _ in range(3):
            queue.claim("w")
        queue.cancel(cancelled)
        assert queue.requeue_running() == 2
        assert queue.claim("w").id == once and queue.claim("w").id == twice
        assert queue.requeue_running() == 0 and queue.interrupt_running() == 2
        assert {queue.get(i).status for i in (once, twice)} == {jq.INTERRUPTED}
        assert queue.get(once).attempts == jq.MAX_ATTEMPTS
        stopped = queue.get(cancelled)
        assert (stopped.status, stopped.error) == (jq.CANCELLED, jq.CANCELLED_STOPPED)

    def test_jobs_survive_reopening_the_file(self, queue):
        """Durability (I-08): a restarted API or worker sees every job."""
        job_id = queue.submit(_body(a=1))
        again = JobQueue(queue.path)
        assert again.get(job_id).status == jq.PENDING and again.claim("w").id == job_id

    def test_concurrent_claimers_never_share_a_row(self, queue):
        ids = {queue.submit(_body(n=n)) for n in range(200)}
        claimed, guard = [], threading.Lock()

        def claim_all():
            other = JobQueue(queue.path)
            while (job := other.claim("w")) is not None:
                with guard:
                    claimed.append(job.id)

        threads = [threading.Thread(target=claim_all) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(claimed) == sorted(ids)

    def test_a_queue_of_another_schema_version_is_refused(self, queue):
        with sqlite3.connect(queue.path) as con:
            con.execute("PRAGMA user_version=99")
        with pytest.raises(RuntimeError, match="schema version 99"):
            JobQueue(queue.path)

    def test_the_http_schema_lists_the_queues_statuses_kinds_and_classes(self):
        """`RunStatusSchema` spells the queue's values out as literals; they must agree."""
        from engine.api.results import RunStatusSchema

        fields = RunStatusSchema.model_fields
        assert get_args(fields["status"].annotation) == jq.STATUSES
        assert get_args(fields["kind"].annotation) == jq.KINDS + jq.LEGACY_KINDS
        assert get_args(get_args(fields["failure_class"].annotation)[0]) == jq.FAILURE_CLASSES


class TestMigration:
    """Versions 1 to 3 become version 4 in place, in one transaction (I-106)."""

    def test_a_version_3_file_counts_the_attempts_of_its_started_jobs(self, tmp_path):
        """Version 3 (2026-10-18) had no `attempts`: a job that had started was claimed once."""
        path = tmp_path / "v3.sqlite3"
        old = JobQueue(path)
        started, queued = old.submit(_body(n=0)), old.submit(_body(n=1))
        old.claim("w")
        old.close()
        with sqlite3.connect(path) as con:
            con.executescript("ALTER TABLE jobs DROP COLUMN attempts; PRAGMA user_version=3;")
        migrated = JobQueue(path)
        assert _version(path) == jq.SCHEMA_VERSION
        assert (migrated.get(started).attempts, migrated.get(queued).attempts) == (1, 0)
        assert migrated.claim("w").attempts == 1

    def test_a_version_2_file_keeps_its_rows_and_its_unrun_job_is_interrupted(self, tmp_path):
        path = tmp_path / "v2.sqlite3"
        with sqlite3.connect(path) as con:
            con.executescript(_V2_SCHEMA)
        migrated = JobQueue(path)
        done, queued = migrated.get("done-risk"), migrated.get("queued")
        assert (done.kind, done.status, done.result) == ("market-risk", jq.DONE, '{"risk": {}}')
        assert migrated.artifact("done-risk", "pnl", 0) == b"\x00"
        assert (queued.kind, queued.status, queued.error) == ("portfolio", jq.INTERRUPTED, jq.RETIRED_ROUTE_ERROR)
        assert migrated.claim("w") is None and _version(path) == jq.SCHEMA_VERSION
        new = migrated.submit(_body(), idempotency_key="k", request_sha256="h", keep=True)
        assert migrated.get(new).keep and migrated.claim("w").id == new

    def test_a_version_1_file_becomes_portfolio_rows(self, tmp_path):
        """The version-1 file (2026-10-04): no `kind` and no artifacts table."""
        path = tmp_path / "v1.sqlite3"
        with sqlite3.connect(path) as con:
            con.executescript("""
                CREATE TABLE jobs (seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE,
                    request BLOB NOT NULL, status TEXT NOT NULL, failure_class TEXT, error TEXT, result TEXT,
                    worker TEXT, compiles INTEGER, submitted REAL NOT NULL, started REAL, finished REAL);
                INSERT INTO jobs (id, request, status, submitted) VALUES ('old', x'7b7d', 'pending', 0);
                PRAGMA user_version=1;""")
        migrated = JobQueue(path)
        assert (migrated.get("old").kind, migrated.get("old").status) == ("portfolio", jq.INTERRUPTED)
        job_id = migrated.submit(_body())
        migrated.claim("w")
        migrated.finish(job_id, "{}", artifacts={"npv_cube": [b"x"]})
        assert migrated.artifact(job_id, "npv_cube", 0) == b"x" and _version(path) == jq.SCHEMA_VERSION

    def test_an_interrupted_migration_leaves_the_file_as_it_was(self, tmp_path, monkeypatch):
        """Red before 2026-10-18: the migration ran in autocommit statements, so a crash after
        the `ALTER TABLE` left the column at the old version, and every later open failed on the
        duplicate column."""
        path = tmp_path / "v2.sqlite3"
        with sqlite3.connect(path) as con:
            con.executescript(_V2_SCHEMA)
        statements = jq.migration_statements
        monkeypatch.setattr(jq, "migration_statements",
                            lambda version: statements(version)[:-2] + ("SELECT no_such_column FROM jobs",))
        with pytest.raises(sqlite3.OperationalError, match="no_such_column"):
            JobQueue(path)
        assert _version(path) == 2
        with sqlite3.connect(path) as con:
            assert [r[1] for r in con.execute("PRAGMA table_info(jobs)")][-1] == "finished"
            assert con.execute("SELECT status FROM jobs WHERE id = 'queued'").fetchone() == ("pending",)
        monkeypatch.undo()
        assert JobQueue(path).get("queued").status == jq.INTERRUPTED and _version(path) == jq.SCHEMA_VERSION

    @pytest.mark.parametrize("old", [True, False], ids=["version-2", "new"])
    def test_processes_opening_a_file_at_once_migrate_it_once(self, tmp_path, old):
        """Red on the first version of this code (2026-10-18), and on the code before for a new
        file: switching the file to WAL takes an exclusive lock that SQLite does not wait for, so
        about one trial in thirteen of a dozen openers had one fail with `database is locked` (3
        of 40 for each kind, measured; 0 of 40 after `JobQueue._use_wal`). Thirty trials fail
        the old code with a probability of about 0.9."""
        for trial in range(30):
            path = tmp_path / f"queue-{trial}.sqlite3"
            if old:
                with sqlite3.connect(path) as con:
                    con.executescript(_V2_SCHEMA)
            errors = []

            def open_it():
                try:
                    JobQueue(path)
                except Exception as exc:  # noqa: BLE001 - collected for the assertion
                    errors.append(exc)

            threads = [threading.Thread(target=open_it) for _ in range(12)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert errors == [] and _version(path) == jq.SCHEMA_VERSION
            with sqlite3.connect(path) as con:
                assert con.execute("SELECT COUNT(*) FROM jobs").fetchone() == (2 if old else 0,)


class TestIdempotency:
    """I-89: a retried submission runs once."""

    def test_the_same_key_and_body_is_the_first_job(self, queue):
        first = queue.submit(_body(a=1), idempotency_key="k1", request_sha256="h1")
        assert queue.submit(_body(a=1), idempotency_key="k1", request_sha256="h1") == first
        assert queue.submit(_body(a=1), idempotency_key="k2", request_sha256="h1") != first
        assert queue.submit(_body(a=1)) != queue.submit(_body(a=1))  # without a key, every submission is a job

    def test_the_same_key_with_another_body_is_refused(self, queue):
        first = queue.submit(_body(a=1), idempotency_key="k", request_sha256="h1")
        with pytest.raises(jq.IdempotencyConflict, match="another body") as refused:
            queue.submit(_body(a=2), idempotency_key="k", request_sha256="h2")
        assert refused.value.job_id == first
        with pytest.raises(ValueError, match="sha256"):
            queue.submit(_body(), idempotency_key="k3")

    def test_two_api_processes_racing_on_one_key_get_one_job(self, queue):
        """Each thread opens the file itself, as each API process does."""
        ids, guard = [], threading.Lock()

        def submit():
            run_id = JobQueue(queue.path).submit(_body(a=1), idempotency_key="race", request_sha256="h")
            with guard:
                ids.append(run_id)

        threads = [threading.Thread(target=submit) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(set(ids)) == 1 and len(ids) == 8


class TestCancelAndRetention:
    def test_a_pending_job_is_cancelled_at_once(self, queue):
        job_id = queue.submit(_body())
        assert queue.cancel(job_id) == jq.CANCELLED
        job = queue.get(job_id)
        assert (job.status, job.error) == (jq.CANCELLED, jq.CANCELLED_PENDING) and job.finished
        assert queue.claim("w") is None and queue.cancel(job_id) == jq.CANCELLED

    def test_a_running_job_is_flagged_and_a_finished_one_left(self, queue):
        running, done = queue.submit(_body()), queue.submit(_body())
        queue.claim("w"), queue.claim("w")
        queue.finish(done, "{}")
        assert not queue.cancel_requested(running)
        assert queue.cancel(running) == jq.RUNNING and queue.cancel_requested(running)
        assert queue.get(running).cancel_requested and queue.cancel(done) == jq.DONE
        assert queue.cancel("nope") is None
        queue.cancel_running(running, "cancelled at the 'pricing' phase boundary")
        assert (queue.get(running).status, queue.get(running).error) == (
            jq.CANCELLED, "cancelled at the 'pricing' phase boundary")

    def test_a_purge_deletes_finished_jobs_before_the_cutoff_with_their_artifacts(self, queue):
        old, kept, recent = (queue.submit(_body(n=n), keep=(n == 1)) for n in range(3))
        for job_id in (old, kept, recent):
            assert queue.claim("w").id == job_id
            queue.finish(job_id, "{}", artifacts={"npv_cube": [b"x"]})
        running, pending = queue.submit(_body()), queue.submit(_body())
        assert queue.claim("w").id == running
        cutoff = time.time()
        with sqlite3.connect(queue.path) as con:
            con.execute("UPDATE jobs SET finished = finished + 3600 WHERE id = ?", (recent,))
        assert queue.purge(cutoff + 1.0) == 1
        assert queue.get(old) is None and queue.artifact(old, "npv_cube", 0) is None
        assert queue.get(kept).status == jq.DONE and queue.artifact(kept, "npv_cube", 0) == b"x"
        assert queue.get(recent).status == jq.DONE
        assert queue.get(pending).status == jq.PENDING and queue.get(running).status == jq.RUNNING

    @pytest.mark.parametrize("value, seconds", [(None, 30 * 86400.0), ("7", 7 * 86400.0), ("0.5", 43200.0),
                                                ("off", None), ("OFF", None)])
    def test_the_retention_period_is_read_from_the_environment(self, value, seconds):
        environ = {} if value is None else {"JAX_RISK_RETENTION_DAYS": value}
        assert jq.retention_seconds(environ) == seconds

    @pytest.mark.parametrize("value", ["0", "-1", "soon", "inf"])
    def test_a_retention_period_that_is_not_a_positive_number_is_refused(self, value):
        with pytest.raises(ValueError, match="JAX_RISK_RETENTION_DAYS"):
            jq.retention_seconds({"JAX_RISK_RETENTION_DAYS": value})


class TestWorkerLock:
    def test_one_holder_at_a_time_and_it_records_its_queue(self, lock):
        first, second = WorkerLock(lock), WorkerLock(lock)
        assert lock_owner(lock) is None
        assert first.acquire(owner={"queue": "q1", "pid": 1}) and worker_lock_held(lock)
        assert lock_owner(lock) == {"queue": "q1", "pid": 1}
        assert not second.acquire(wait_seconds=0.1)
        first.release()
        assert not worker_lock_held(lock) and lock_owner(lock) is None
        assert second.acquire(owner={"queue": "q2"}) and lock_owner(lock) == {"queue": "q2"}
        second.release()

    def test_the_lock_is_one_per_host_by_default(self, tmp_path):
        assert jq.worker_lock_path({}) == jq.worker_lock_path({}) and jq.worker_lock_path({}).name == "worker.lock"
        assert jq.worker_lock_path({"JAX_RISK_WORKER_LOCK": str(tmp_path / "gpu1.lock")}) == tmp_path / "gpu1.lock"

    @pytest.mark.slow
    def test_a_dead_holders_lock_is_free(self, lock):
        """The OS releases the lock however the holder dies, so a killed worker leaves none.

        Once it has died: on Windows a venv's `python.exe` is a launcher whose child, the real
        interpreter, holds the lock and ends a moment after the launcher is killed (the full
        suite under load saw the lock still held right after `wait()`)."""
        code = (f"from engine.api.job_queue import WorkerLock; import time; lock = WorkerLock({str(lock)!r}); "
                f"assert lock.acquire(); print('locked', flush=True); time.sleep(60)")
        holder = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
        try:
            assert holder.stdout.readline().strip() == "locked"
            assert worker_lock_held(lock)
        finally:
            holder.kill()
            holder.wait()
        deadline = time.monotonic() + 30.0
        while worker_lock_held(lock) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not worker_lock_held(lock)


class TestWorkerLoop:
    """`serve` in this process around stub pricers, until the queue is drained."""

    def test_every_job_gets_its_own_result_in_order(self, queue, lock):
        ids = [queue.submit(_body(n=n)) for n in range(3)]
        assert _serve(queue, lock) == 0
        for n, job_id in enumerate(ids):
            job = queue.get(job_id)
            assert job.status == jq.DONE and json.loads(job.result) == {"echo": {"n": n}, "kind": jq.RUN}
        assert set(worker.JOB_KINDS) == set(jq.KINDS)

    def test_a_jobs_artifacts_are_stored_with_its_result(self, queue, lock):
        job_id = queue.submit(_body())
        chunks = [b"\x00" * 8, b"\x01" * 8]
        _serve(queue, lock, price=lambda job: worker.JobResult("{}", {"npv_cube": chunks}))
        assert queue.get(job_id).status == jq.DONE
        assert [queue.artifact(job_id, "npv_cube", i) for i in range(3)] == chunks + [None]

    def test_a_failing_job_fails_only_its_own_row(self, queue, lock):
        """§13.8: with a failure class and the traceback; the worker goes on to the next job."""
        def pricer(job):
            if json.loads(job.request).get("bad"):
                raise KeyError("no market for currency 'GBP'")
            return echo_pricer(job)

        ok_before, bad, ok_after = (queue.submit(_body(bad=b)) for b in (False, True, False))
        _serve(queue, lock, price=pricer)
        failed = queue.get(bad)
        assert (failed.status, failed.failure_class) == (jq.FAILED, jq.MISSING_MARKET_DATA)
        assert "KeyError" in failed.error and "no market for currency 'GBP'" in failed.error
        assert "Traceback" in failed.error
        assert [queue.get(i).status for i in (ok_before, ok_after)] == [jq.DONE, jq.DONE]

    def test_a_job_cancelled_while_running_stops_at_its_next_phase_boundary(self, queue, lock):
        """I-91: the job reads `cancelled`, naming the boundary; the worker goes on. A pricer's
        own `except Exception` does not swallow the stop."""
        reached = []

        def pricer(job):
            if json.loads(job.request).get("cancel"):
                queue.cancel(job.id)  # as the cancel route would, while the job runs
                try:
                    worker.phase_boundary("pricing")
                except Exception:  # noqa: BLE001 - the engine's broad handlers must not catch it
                    reached.append("swallowed")
                reached.append("past the boundary")
            return echo_pricer(job)

        cancelled, after = queue.submit(_body(cancel=True)), queue.submit(_body(n=1))
        _serve(queue, lock, price=pricer)
        job = queue.get(cancelled)
        assert (job.status, job.error, job.result) == (jq.CANCELLED, "cancelled at the 'pricing' phase boundary", None)
        assert reached == [] and queue.get(after).status == jq.DONE

    def test_a_job_cancelled_before_it_starts_stops_at_the_start(self, queue, lock):
        """A cancel that reaches a job between its claim and its pricer."""
        job_id = queue.submit(_body())
        job = queue.claim("w")
        queue.cancel(job_id)
        counter = type("Counter", (), {"count": 0})()
        worker.run_job(queue, job, lambda job: pytest.fail("the pricer ran"), counter)
        assert (queue.get(job_id).status, queue.get(job_id).error) == (
            jq.CANCELLED, "cancelled at the 'start' phase boundary")

    def test_outside_a_job_a_phase_boundary_does_nothing(self):
        worker.phase_boundary("pricing")

    @pytest.mark.parametrize("make, expected", [
        (lambda: KeyError("curve"), jq.MISSING_MARKET_DATA),
        (lambda: __import__("engine.instruments.schedules", fromlist=["x"]).MissingFixingError("fixing"),
         jq.MISSING_MARKET_DATA),
        (lambda: NotImplementedError("product"), jq.UNSUPPORTED_PRODUCT),
        (lambda: __import__("engine.market_data.day_counts", fromlist=["x"]).UnsupportedDayCountError("30/360 German"),
         jq.UNSUPPORTED_PRODUCT),
        (lambda: FloatingPointError("overflow"), jq.NUMERICAL_FAILURE),
        (lambda: ZeroDivisionError("x"), jq.NUMERICAL_FAILURE),
        (lambda: ValueError("terms"), jq.BAD_TERMS),
        (lambda: TypeError("terms"), jq.BAD_TERMS),
        (lambda: MemoryError(), jq.INFRASTRUCTURE),
        (lambda: RuntimeError("XLA"), jq.INFRASTRUCTURE),
    ], ids=["KeyError", "MissingFixing", "NotImplemented", "UnsupportedDayCount", "FloatingPoint",
            "ZeroDivision", "ValueError", "TypeError", "MemoryError", "RuntimeError"])
    def test_the_failure_class_follows_the_exception(self, make, expected):
        assert worker.failure_class(make()) == expected

    def test_a_request_that_does_not_parse_is_bad_terms(self, queue, lock):
        """A body that reaches the worker without the route's validation (pydantic's
        `ValidationError` is a `ValueError`)."""
        job_id = queue.submit(_body(market=12345))
        _serve(queue, lock, price=None)  # the real pricer: fails in the parse, before JAX work
        assert queue.get(job_id).failure_class == jq.BAD_TERMS

    def test_a_result_the_queue_cannot_store_fails_as_infrastructure(self, queue, lock):
        job_id = queue.submit(_body())
        _serve(queue, lock, price=lambda job: worker.JobResult(object()))
        failed = queue.get(job_id)
        assert failed.failure_class == jq.INFRASTRUCTURE and "storing the result failed" in failed.error

    def test_a_starting_worker_interrupts_its_predecessors_running_job(self, queue, lock):
        orphan, queued = queue.submit(_body()), queue.submit(_body())
        queue.claim("dead-worker")
        _serve(queue, lock)
        assert queue.get(orphan).status == jq.INTERRUPTED
        assert queue.get(queued).status == jq.DONE

    def test_a_second_worker_on_the_host_exits_at_once(self, queue, lock, tmp_path, monkeypatch):
        """Whatever queue it serves (I-106): the lock is the host's, not the queue file's."""
        monkeypatch.setattr(worker, "LOCK_WAIT_SECONDS", 0.05)
        other = JobQueue(tmp_path / "other.sqlite3")
        job_id = other.submit(_body())
        holder = WorkerLock(lock)
        assert holder.acquire(owner={"queue": str(queue.path)})
        try:
            assert _serve(other, lock) == worker.EXIT_HOST_OWNED
        finally:
            holder.release()
        assert other.get(job_id).status == jq.PENDING

    def test_the_worker_records_the_queue_it_serves(self, queue, lock):
        seen = {}

        def pricer(job):
            seen.update(lock_owner(lock))
            return echo_pricer(job)

        queue.submit(_body())
        _serve(queue, lock, price=pricer)
        assert seen["queue"] == str(queue.path.resolve()) and seen["pid"] > 0 and seen["host"]

    def test_a_worker_whose_parent_is_gone_claims_nothing(self, queue, lock):
        job_id = queue.submit(_body())
        assert _serve(queue, lock, parent_pid=_dead_pid(), drain=False) == 0
        assert queue.get(job_id).status == jq.PENDING

    def test_the_worker_purges_finished_jobs_past_the_retention_period(self, queue, lock):
        """I-76: between jobs; a job submitted with `keep` stays, as does one inside the period."""
        old, kept = queue.submit(_body()), queue.submit(_body(), keep=True)
        _serve(queue, lock)
        with sqlite3.connect(queue.path) as con:
            con.execute("UPDATE jobs SET finished = finished - 86400 * 10")
        recent = queue.submit(_body())
        _serve(queue, lock, retention=86400.0 * 5)
        assert queue.get(old) is None and queue.get(kept).status == jq.DONE and queue.get(recent).status == jq.DONE
        _serve(queue, lock, environ={"JAX_RISK_RETENTION_DAYS": "off"})
        assert queue.get(kept) is not None

    def test_each_job_records_the_programs_it_compiled(self, queue, lock):
        import jax
        import jax.numpy as jnp

        x = jax.block_until_ready(jnp.ones(3))

        def pricer(job):
            n = json.loads(job.request)["n"]
            jax.jit(lambda v: v * 2.0 + n)(x)  # a new program each job: one compile
            return worker.JobResult("{}")

        ids = [queue.submit(_body(n=n)) for n in range(2)]
        _serve(queue, lock, price=pricer)
        assert [queue.get(i).compiles for i in ids] == [1, 1]


class TestCompilationCache:
    def test_unset_the_cache_sits_beside_the_queue_and_keeps_every_program(self, tmp_path):
        env = worker.compilation_cache_environment(tmp_path / "jobs.sqlite3", {})
        assert env == {"JAX_COMPILATION_CACHE_DIR": str((tmp_path / worker.COMPILATION_CACHE_DIRNAME).resolve()),
                       "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
                       "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "0"}

    def test_a_variable_already_set_wins_and_empty_turns_the_cache_off(self, tmp_path):
        env = worker.compilation_cache_environment(
            tmp_path / "jobs.sqlite3", {"JAX_COMPILATION_CACHE_DIR": "", "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "1"})
        assert env == {"JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "0"}


class TestSupervisor:
    """A worker that cannot start is reported (I-77)."""

    def test_a_worker_that_exits_at_once_is_reported_after_its_failed_starts(self, queue, lock):
        supervisor = WorkerSupervisor(queue, command=failing_worker_command(), lock_path=lock)
        for attempt in range(FAILED_STARTS_REPORTED):
            assert supervisor.status().error is None
            supervisor.ensure_running()
            supervisor.process.wait()
        state = supervisor.status()
        assert (state.serving, state.failed_starts) == (False, FAILED_STARTS_REPORTED)
        assert "the engine worker cannot start" in state.error and "exited with status 1" in state.error
        assert "the engine cannot be imported: a broken install" in state.error

    def test_a_worker_that_cannot_be_launched_is_reported_at_once_and_nothing_raises(self, queue, lock, tmp_path):
        supervisor = WorkerSupervisor(queue, command=[str(tmp_path / "no-such-python")], lock_path=lock)
        supervisor.ensure_running()
        state = supervisor.status()
        assert supervisor.process is None and state.failed_starts == 1
        assert "could not be launched" in state.error

    def test_a_worker_that_served_a_run_ends_the_failed_starts(self, queue, lock, monkeypatch):
        supervisor = WorkerSupervisor(queue, command=failing_worker_command(), lock_path=lock)
        supervisor.ensure_running()
        supervisor.process.wait()
        monkeypatch.setattr(queue, "claimed_since", lambda since: True)
        assert supervisor.status().failed_starts == 0


@pytest.mark.slow
class TestWorkerProcesses:
    """Real worker processes around the stub pricer."""

    def test_a_worker_killed_mid_job_leaves_it_interrupted(self, queue, lock):
        """§13.8: the killed worker's job is `interrupted`; a restarted worker picks up the
        next."""
        long_job, next_job = queue.submit(_body(sleep=120)), queue.submit(_body(n=1))
        first = subprocess.Popen(stub_worker_command(queue.path, lock))
        try:
            _wait_for(lambda: queue.status(long_job) == jq.RUNNING)
        finally:
            first.kill()
            first.wait()
        assert queue.status(long_job) == jq.RUNNING  # nobody has noticed yet
        _wait_for(lambda: not worker_lock_held(lock), timeout=30)
        second = subprocess.Popen(stub_worker_command(queue.path, lock))
        try:
            _wait_for(lambda: queue.status(next_job) == jq.DONE)
        finally:
            second.kill()
            second.wait()
        interrupted = queue.get(long_job)
        assert (interrupted.status, interrupted.error) == (jq.INTERRUPTED, jq.INTERRUPTED_ERROR)

    def test_a_running_job_is_cancelled_at_a_phase_boundary_and_the_worker_goes_on(self, queue, lock):
        cancelled, after = queue.submit(_body(phases=200, sleep=0.05)), queue.submit(_body(n=2))
        process = subprocess.Popen(stub_worker_command(queue.path, lock))
        try:
            _wait_for(lambda: queue.status(cancelled) == jq.RUNNING)
            assert queue.cancel(cancelled) == jq.RUNNING
            _wait_for(lambda: queue.status(after) == jq.DONE)
            assert process.poll() is None
        finally:
            process.kill()
            process.wait()
        job = queue.get(cancelled)
        assert job.status == jq.CANCELLED and "phase boundary" in job.error and "stub" in job.error

    def test_the_supervisor_restarts_a_dead_worker(self, queue, lock):
        supervisor = WorkerSupervisor(queue, command=stub_worker_command(queue.path, lock), lock_path=lock)
        try:
            supervisor.ensure_running()
            first = supervisor.process
            supervisor.ensure_running()
            assert supervisor.process is first  # alive: left alone
            first.kill()
            first.wait()
            _wait_for(lambda: not worker_lock_held(lock))
            supervisor.ensure_running()
            assert supervisor.process is not first and supervisor.process.poll() is None
            job_id = queue.submit(_body(n=7))
            _wait_for(lambda: queue.status(job_id) == jq.DONE)
            assert supervisor.status().serving
        finally:
            supervisor.stop()
        assert supervisor.process is None

    def test_supervisors_of_one_queue_share_one_worker(self, queue, lock):
        """Several API processes (uvicorn `--workers`) on one queue: a supervisor starts no
        worker while another process's worker holds the host's lock."""
        first = WorkerSupervisor(queue, command=stub_worker_command(queue.path, lock), lock_path=lock)
        second = WorkerSupervisor(queue, command=stub_worker_command(queue.path, lock), lock_path=lock)
        try:
            first.ensure_running()
            _wait_for(lambda: lock_owner(lock))
            second.ensure_running()
            assert second.process is None and second.status().serving
            first.stop()
            _wait_for(lambda: not worker_lock_held(lock))
            second.ensure_running()  # the host's worker is gone: this one takes over
            assert second.process is not None
            job_id = queue.submit(_body(n=1))
            _wait_for(lambda: queue.status(job_id) == jq.DONE)
        finally:
            first.stop()
            second.stop()

    def test_a_worker_exits_once_its_parent_is_gone(self, queue, lock):
        """No orphaned engine process outlives a killed API or test run."""
        launcher = ("import subprocess, os, time; "
                    "from tests.support.worker_stubs import stub_worker_command; "
                    f"subprocess.Popen(stub_worker_command({str(queue.path)!r}, {str(lock)!r}, parent_pid=os.getpid())); "
                    "time.sleep(120)")
        parent = subprocess.Popen([sys.executable, "-c", launcher], cwd=ROOT)
        try:
            _wait_for(lambda: worker_lock_held(lock))
        finally:
            parent.kill()
            parent.wait()
        _wait_for(lambda: not worker_lock_held(lock), timeout=30)


class TestRoutesOverTheQueue:
    """The HTTP routes read and write the queue (no worker in this process: `external` mode,
    and the stub worker run in place)."""

    @pytest.fixture
    def client(self, queue, lock, monkeypatch, test_client):
        from engine.api import routes

        monkeypatch.setattr(routes, "_QUEUE", queue)
        monkeypatch.setattr(routes, "_LOCK_PATH", lock)
        monkeypatch.setattr(routes, "_SUPERVISOR", None)  # external: the API starts no worker
        return test_client

    @staticmethod
    def _request(**fields):
        return {**run_body(["npv"], ["swap-payer"]), **fields}

    def _submit(self, client, **fields):
        r = client.post("/runs", json=self._request(**fields))
        assert r.status_code == 202, r.text
        return r.json()["run_id"]

    def test_the_route_queues_the_body_as_received(self, client, queue):
        run_id = self._submit(client)
        status = client.get(f"/runs/{run_id}").json()
        assert {k: status[k] for k in ("run_id", "kind", "status", "result", "failure_class", "cancel_requested",
                                       "keep")} == {"run_id": run_id, "kind": "run", "status": "pending",
                                                    "result": None, "failure_class": None,
                                                    "cancel_requested": False, "keep": False}
        assert status["submitted"].endswith("+00:00") and status["started"] is None
        assert json.loads(queue.get(run_id).request)["portfolio"][0]["trade_id"] == "swap-payer"

    def test_a_done_run_returns_the_stored_document(self, client, queue, lock):
        run_id = self._submit(client)
        _serve(queue, lock, price=lambda job: worker.JobResult('{"run_id": "x", "echo": true}'))
        data = client.get(f"/runs/{run_id}").json()
        assert data["kind"] == "run" and data["status"] == "done" and data["finished"]
        assert data["error"] is None and data["failure_class"] is None
        assert data["result"] == {"run_id": "x", "echo": True}

    def test_an_artifact_chunk_is_served_as_stored(self, client, queue, lock):
        run_id = self._submit(client)
        _serve(queue, lock, price=lambda job: worker.JobResult("{}", {"npv_cube": [b"\x00\x01", b"\x02"]}))
        response = client.get(f"/runs/{run_id}/artifacts/npv_cube/1")
        assert response.status_code == 200 and response.content == b"\x02"
        assert response.headers["content-type"] == "application/octet-stream"
        assert client.get(f"/runs/{run_id}/artifacts/npv_cube/2").status_code == 404
        assert client.get(f"/runs/{run_id}/artifacts/pnl/0").status_code == 404

    def test_a_failed_run_returns_its_class_and_traceback(self, client, queue, lock):
        run_id = self._submit(client)

        def pricer(job):
            raise NotImplementedError("no such product")

        _serve(queue, lock, price=pricer)
        data = client.get(f"/runs/{run_id}").json()
        assert (data["status"], data["failure_class"], data["result"]) == ("failed", jq.UNSUPPORTED_PRODUCT, None)
        assert "NotImplementedError: no such product" in data["error"]

    def test_an_interrupted_run_says_so(self, client, queue):
        run_id = self._submit(client)
        queue.claim("dead-worker")
        queue.interrupt_running()
        data = client.get(f"/runs/{run_id}").json()
        assert (data["status"], data["error"]) == ("interrupted", jq.INTERRUPTED_ERROR)

    def test_an_unknown_run_is_a_404(self, client):
        assert client.get("/runs/not-a-run").status_code == 404
        assert client.post("/runs/not-a-run/cancel").status_code == 404

    def test_a_retried_submission_is_the_same_run_and_another_body_a_409(self, client, queue):
        """I-89, red before 2026-10-18: every submission was a new job. The body's identity is its
        canonical form: a default spelt out is the same body."""
        first = self._submit(client, idempotency_key="order-42")
        assert self._submit(client, idempotency_key="order-42") == first
        spelt_out = self._request(idempotency_key="order-42", outputs={"npv_cube": "artifact"})
        assert client.post("/runs", json=spelt_out).json()["run_id"] == first
        conflict = client.post("/runs", json=self._request(idempotency_key="order-42", keep=True))
        assert conflict.status_code == 409 and first in conflict.json()["detail"]
        assert queue.claim("w").id == first and queue.claim("w") is None

    def test_a_kept_run_says_so(self, client):
        run_id = self._submit(client, keep=True)
        assert client.get(f"/runs/{run_id}").json()["keep"] is True

    def test_cancelling_a_pending_run_and_a_finished_one(self, client, queue, lock):
        """I-91: a pending run is cancelled at once; a finished one is a 409; cancelling again is
        answered as it is."""
        pending = self._submit(client)
        cancelled = client.post(f"/runs/{pending}/cancel")
        assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"
        assert client.post(f"/runs/{pending}/cancel").json()["status"] == "cancelled"
        assert client.get(f"/runs/{pending}").json()["error"] == jq.CANCELLED_PENDING
        done = self._submit(client)
        _serve(queue, lock)
        assert client.post(f"/runs/{done}/cancel").status_code == 409

    def test_cancelling_a_running_run_flags_it(self, client, queue):
        run_id = self._submit(client)
        queue.claim("w")
        response = client.post(f"/runs/{run_id}/cancel").json()
        assert (response["status"], response["cancel_requested"]) == ("running", True)

    def test_wait_returns_when_the_run_finishes(self, client, queue, lock):
        run_id = self._submit(client)
        threading.Timer(0.3, lambda: _serve(JobQueue(queue.path), lock)).start()
        started = time.monotonic()
        data = client.get(f"/runs/{run_id}", params={"wait": 20}).json()
        assert data["status"] == "done" and time.monotonic() - started < 15
        assert client.get(f"/runs/{run_id}", params={"wait": 31}).status_code == 422

    def test_a_host_whose_worker_serves_another_queue_refuses_the_run(self, client, queue, lock):
        """I-106: one worker per host; this API's queue would have none, so no run is queued."""
        holder = WorkerLock(lock)
        assert holder.acquire(owner={"queue": "/elsewhere/jobs.sqlite3", "pid": 4242})
        try:
            r = client.post("/runs", json=self._request())
            assert r.status_code == 503 and "/elsewhere/jobs.sqlite3" in r.json()["detail"]
            health = client.get("/health").json()["worker"]
            assert health["serving"] is False and "one engine worker per host" in health["error"]
        finally:
            holder.release()
        assert queue.claim("w") is None

    def test_a_worker_that_cannot_start_leaves_the_run_pending_with_the_reason(self, client, queue, lock,
                                                                               monkeypatch, tmp_path):
        """I-77 (and the project audit's AUD-37): the run is queued and its id returned, never an
        error for a run that was queued; its status, and `/health`, say why it waits."""
        from engine.api import routes

        supervisor = WorkerSupervisor(queue, command=[str(tmp_path / "no-such-python")], lock_path=lock)
        monkeypatch.setattr(routes, "_SUPERVISOR", supervisor)
        run_id = self._submit(client)
        status = client.get(f"/runs/{run_id}").json()
        assert status["status"] == "pending" and "the engine worker cannot start" in status["error"]
        assert "could not be launched" in client.get("/health").json()["worker"]["error"]


# --- real pricing ---------------------------------------------------------------------------

def _pricing_body(trades, samples=64, **config) -> dict:
    return run_body(["npv", "exposure"], trades, outputs={"npv_cube": "inline"},
                    **{"models": models("HullWhite"), "simulation": simulation(samples, months=(6, 12), seed=5),
                       **config})


def test_the_worker_runs_the_request_the_route_validated(monkeypatch):
    """The worker builds its dataclass request from the stored body exactly as the route did
    (until 2026-10-04 the request travelled pickled, its ORE dates frozen as text): every section
    of the configuration, the trades' dates and fixings and the market arrive equal."""
    import engine.run

    f32 = {"storage": "float32", "compute": "float32", "accumulate": "float32"}
    body = run_body(
        ["npv", "exposure", "sensitivities"],
        ["swap-receiver-seasoned-icma", "european-payer", "bermudan-payer-physical", "bond"],
        models=models("HullWhite"), simulation=simulation(64, months=(6, 12), seed=5),
        pricing={"european": "Jamshidian", "jamshidian": {"reversion": 0.03, "volatility": 0.01},
                 "bermudan": {"n_per_std": 12, "std_devs": 4.0, "bootstrap_tolerance": 1e-3}},
        sensitivity={"method": "AD", "curve_tenors": ["1Y", "2Y"]},
        precision={"simulation": f32, "pricing": {"storage": "float32"}, "by_trade": {"bond": f32}},
        exposure={"pfe_quantiles": [0.9, 0.99]})
    seen = {}

    class Stop(Exception):
        pass

    def capture(request):
        seen["request"] = request
        raise Stop

    monkeypatch.setattr(engine.run, "run", capture)
    with pytest.raises(Stop):
        worker.price_job(jq.Job(id="j", kind=jq.RUN, status=jq.RUNNING, request=json.dumps(body).encode()))
    route = request_of(body)
    assert seen["request"] == route
    assert seen["request"].portfolio[0].fixings == route.portfolio[0].fixings and route.portfolio[0].fixings


def test_a_run_without_exposure_returns_no_cube_and_no_artifact():
    """`npv` alone has no cube: the default `outputs.npv_cube` ("artifact") gives neither the cube
    nor a reference, and no chunk."""
    body = run_body(["npv"], ["swap-payer", "bond"])
    result = worker.price_job(jq.Job(id="j", kind=jq.RUN, status=jq.RUNNING, request=json.dumps(body).encode()))
    document = json.loads(result.document)
    assert result.artifacts == {} and document["exposure"] is None and document["run_id"] == "j"
    assert [row["trade_id"] for row in document["trades"]] == ["swap-payer", "bond"]


@pytest.mark.slow
class TestEngineWorkerPricing:
    """The real worker (`price_job`) in its own process, started by the supervisor."""

    @pytest.fixture
    def running(self, queue, lock):
        supervisor = WorkerSupervisor(queue, lock_path=lock)
        supervisor.ensure_running()
        yield queue
        supervisor.stop()

    @staticmethod
    def _result(queue, job_id, timeout=600):
        job = _wait_for(lambda: (j := queue.get(job_id)).status in jq.TERMINAL and j, timeout=timeout)
        assert job.status == jq.DONE, job.error
        return job

    def test_runs_queued_together_give_the_bits_of_runs_one_after_another(self, running):
        """§13.8; and every run equals the in-process run, at float64 and float32."""
        from tests.support.runs import run_in_process

        f32 = {"storage": "float32", "compute": "float32", "accumulate": "float32"}
        bodies = [_pricing_body(["swap-payer", "european-payer"]),
                  _pricing_body(["swap-payer", "european-payer"],
                                precision={"simulation": f32, "market": f32, "pricing": f32}),
                  _pricing_body(["bond", "swap-receiver-seasoned-icma"], samples=32)]
        together = [running.submit(json.dumps(b).encode()) for b in bodies]
        together = [json.loads(self._result(running, j).result) for j in together]
        alone = [json.loads(self._result(running, running.submit(json.dumps(b).encode())).result) for b in bodies]
        strip = lambda document: {k: v for k, v in document.items() if k != "run_id"}  # noqa: E731
        assert [strip(d) for d in together] == [strip(d) for d in alone]
        for body, result in zip(bodies, together):
            direct = run_in_process(body).portfolio
            np.testing.assert_array_equal(np.asarray(result["exposure"]["npv_cube"]["values"]),
                                          np.asarray(direct.npv_cube))
            assert result["npv"] == direct.base_npv
            np.testing.assert_array_equal(result["exposure"]["netting_set"]["epe"], np.asarray(direct.exposure.epe))
        assert together[0]["exposure"]["npv_cube"] != together[1]["exposure"]["npv_cube"], \
            "the float32 run was not priced in float32"

    def test_a_worker_keeps_its_programs_on_disk_beside_its_queue(self, queue, lock):
        """A worker started with no cache settings (`main`, as the supervisor starts it) keeps
        JAX's persistent compilation cache in `xla-cache/` beside its queue file, so a
        restarted worker reads its programs back (I-53)."""
        import os

        cache = queue.path.parent / worker.COMPILATION_CACHE_DIRNAME
        env = {k: v for k, v in os.environ.items() if not k.startswith(("JAX_COMPILATION_CACHE", "JAX_PERSISTENT_CACHE"))}
        process = subprocess.Popen(
            [sys.executable, "-c", "from engine.api.worker import main; main()", "--queue", str(queue.path),
             "--lock", str(lock)], cwd=ROOT, env=env, stdin=subprocess.DEVNULL)
        try:
            self._result(queue, queue.submit(json.dumps(_pricing_body(["swap-payer"], samples=8)).encode()))
        finally:
            process.kill()
            process.wait()
        assert cache.is_dir() and any(cache.iterdir()), "the worker compiled without a persistent cache"

    def test_a_second_identical_run_compiles_nothing(self, running):
        """§13.9: the worker keeps its programs, so a repeated run shape builds no program,
        compiled or read from the persistent cache (until 2026-10-04 every pool worker built its
        own); the run request's own work (preflight, scenarios) included."""
        body = json.dumps(_pricing_body(["swap-payer", "european-payer", "bermudan-payer-physical"],
                                        pricing={"bermudan": {"n_per_std": 12, "std_devs": 4.0}})).encode()
        first = self._result(running, running.submit(body))
        second = self._result(running, running.submit(body))
        assert first.compiles > 0, "the compile counter saw nothing: has JAX renamed its compile event?"
        assert second.compiles == 0
        assert json.loads(first.result)["exposure"] == json.loads(second.result)["exposure"]


@pytest.mark.slow
@pytest.mark.parametrize("boundary", [2, 4, 7])
def test_a_real_run_stops_at_the_phase_boundary_a_cancel_reaches(queue, boundary, monkeypatch):
    """I-91: the engine's phases are the boundaries (`engine.run.trace.phase`, each trade's
    pricing and Greeks among them): a cancel seen at the `boundary`-th stops the real run there,
    through every layer of the engine, and the job reads `cancelled` naming the phase. The same
    run afterwards completes, equal to the in-process run."""
    from engine.run.trace import GREEKS_TRADE_PREFIX, PHASES, PRICING_TRADE_PREFIX
    from tests.support.runs import run_in_process

    body = run_body(["npv", "exposure", "sensitivities"], ["swap-payer", "bermudan-payer-physical"],
                    outputs={"npv_cube": "inline"}, models=models(), simulation=simulation(32, months=(6, 12)),
                    pricing={"bermudan": {"n_per_std": 8, "std_devs": 4.0}})
    raw = json.dumps(body).encode()
    counter = type("Counter", (), {"count": 0})()
    looks = []
    monkeypatch.setattr(queue, "cancel_requested", lambda run_id: looks.append(run_id) or len(looks) >= boundary)
    cancelled = queue.submit(raw)
    worker.run_job(queue, queue.claim("w"), worker.price_job, counter)
    job = queue.get(cancelled)
    phase = job.error.split("'")[1]
    assert job.status == jq.CANCELLED and len(looks) == boundary, job.error
    assert phase == "start" if boundary == 1 else (phase in PHASES or phase.startswith(
        (PRICING_TRADE_PREFIX, GREEKS_TRADE_PREFIX))), phase
    monkeypatch.undo()
    done = queue.submit(raw)
    worker.run_job(queue, queue.claim("w"), worker.price_job, counter)
    result = json.loads(queue.get(done).result)
    np.testing.assert_array_equal(np.asarray(result["exposure"]["npv_cube"]["values"]),
                                  np.asarray(run_in_process(body).portfolio.npv_cube))
