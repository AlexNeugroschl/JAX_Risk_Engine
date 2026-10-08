"""
The engine worker behind the HTTP API (roadmap 1.8, decision A-14;
docs/planning/details/precision.md §11 and §13.8): the durable job queue
(`engine.api.job_queue`), the single-threaded worker (`engine.api.worker`) and its supervisor
(`engine.api.supervisor`).

  * The queue keeps every job across reopenings, claims in submission order, never hands one
    row to two claimers, and closes a row only from `running`.
  * One worker per queue: the lock refuses a second holder and is free once its holder dies.
  * A failing job fails only its own row, with a failure class, and the worker goes on; a
    worker killed mid-job leaves `interrupted`, and the next worker prices what is queued.
  * The supervisor restarts a dead worker, starts none while another process's worker serves
    the queue, and a worker whose parent is gone exits.
  * Real pricing: jobs queued together give the bits they give run one after another, and
    the direct `price_portfolio` call's; a second identical job compiles nothing; the worker
    prices the very request the route validated.

The process-level tests run the real loop around a stub pricer (`tests/support/worker_stubs.py`);
the HTTP contract over the worker is tests/test_api.py.
"""
import json
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from engine.api import job_queue as jq
from engine.api import worker
from engine.api.job_queue import JobQueue, WorkerLock, worker_lock_held
from engine.api.supervisor import WorkerSupervisor
from tests.support import portfolio as shared
from tests.support.worker_stubs import ROOT, echo_pricer, stub_worker_command


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


class TestJobQueue:
    def test_a_submitted_job_is_pending_with_its_body_byte_for_byte(self, queue):
        body = b'{"trades": [],  "market": {"asof": "2026-07-30"}}'
        job = queue.get(queue.submit(body))
        assert (job.status, job.request, job.result, job.failure_class) == (jq.PENDING, body, None, None)
        assert queue.status(job.id) == jq.PENDING

    def test_an_unknown_id_has_no_row(self, queue):
        assert queue.get("nope") is None and queue.status("nope") is None

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

    def test_a_job_keeps_its_kind_and_an_unknown_kind_is_refused(self, queue):
        job_id = queue.submit(_body(), jq.MARKET_RISK)
        assert queue.get(job_id).kind == jq.MARKET_RISK and queue.state(job_id) == (jq.MARKET_RISK, jq.PENDING)
        assert queue.get(queue.submit(_body())).kind == jq.PORTFOLIO
        with pytest.raises(ValueError, match="job kind"):
            queue.submit(_body(), "greeks")

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

    def test_a_queue_of_schema_version_1_is_migrated(self, tmp_path):
        """Roadmap 1.8's file: its jobs become portfolio jobs, and it gains the artifacts table."""
        import sqlite3

        path = tmp_path / "v1.sqlite3"
        with sqlite3.connect(path) as con:
            con.executescript("""
                CREATE TABLE jobs (seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE,
                    request BLOB NOT NULL, status TEXT NOT NULL, failure_class TEXT, error TEXT, result TEXT,
                    worker TEXT, compiles INTEGER, submitted REAL NOT NULL, started REAL, finished REAL);
                INSERT INTO jobs (id, request, status, submitted) VALUES ('old', x'7b7d', 'pending', 0);
                PRAGMA user_version=1;""")
        migrated = JobQueue(path)
        assert migrated.get("old").kind == jq.PORTFOLIO and migrated.claim("w").id == "old"
        migrated.finish("old", "{}", artifacts={"npv_cube": [b"x"]})
        assert migrated.artifact("old", "npv_cube", 0) == b"x"
        with sqlite3.connect(path) as con:
            assert con.execute("PRAGMA user_version").fetchone()[0] == 2

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

    def test_jobs_survive_reopening_the_file(self, queue):
        """Durability (I-08): a restarted API or worker sees every job."""
        job_id = queue.submit(_body(a=1))
        again = JobQueue(queue.path)
        assert again.get(job_id).status == jq.PENDING and again.claim("w").id == job_id

    def test_concurrent_claimers_never_share_a_row(self, queue):
        ids = {queue.submit(_body(n=n)) for n in range(200)}
        claimed, lock = [], threading.Lock()

        def claim_all():
            other = JobQueue(queue.path)
            while (job := other.claim("w")) is not None:
                with lock:
                    claimed.append(job.id)

        threads = [threading.Thread(target=claim_all) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(claimed) == sorted(ids)

    def test_a_queue_of_another_schema_version_is_refused(self, queue):
        import sqlite3

        with sqlite3.connect(queue.path) as con:
            con.execute("PRAGMA user_version=99")
        with pytest.raises(RuntimeError, match="schema version 99"):
            JobQueue(queue.path)

    def test_the_http_schema_lists_the_queues_statuses_and_classes(self):
        """`JobStatusSchema` spells the queue's values out as literals; they must agree."""
        from typing import get_args

        from engine.api.schemas import JobStatusSchema

        fields = JobStatusSchema.model_fields
        assert get_args(fields["status"].annotation) == jq.STATUSES
        assert get_args(fields["kind"].annotation) == jq.KINDS
        assert get_args(get_args(fields["failure_class"].annotation)[0]) == jq.FAILURE_CLASSES


class TestWorkerLock:
    def test_one_holder_at_a_time(self, queue):
        first, second = WorkerLock(queue.path), WorkerLock(queue.path)
        assert first.acquire() and worker_lock_held(queue.path)
        assert not second.acquire(wait_seconds=0.1)
        first.release()
        assert not worker_lock_held(queue.path) and second.acquire()
        second.release()

    @pytest.mark.slow
    def test_a_dead_holders_lock_is_free(self, queue):
        """The OS releases the lock however the holder dies, so a killed worker leaves none.

        Once it has died: on Windows a venv's `python.exe` is a launcher whose child, the real
        interpreter, holds the lock and ends a moment after the launcher is killed (the full
        suite under load saw the lock still held right after `wait()`)."""
        code = (f"from engine.api.job_queue import WorkerLock; import time; lock = WorkerLock({str(queue.path)!r}); "
                f"assert lock.acquire(); print('locked', flush=True); time.sleep(60)")
        holder = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
        try:
            assert holder.stdout.readline().strip() == "locked"
            assert worker_lock_held(queue.path)
        finally:
            holder.kill()
            holder.wait()
        deadline = time.monotonic() + 30.0
        while worker_lock_held(queue.path) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not worker_lock_held(queue.path)


class TestWorkerLoop:
    """`serve` in this process around stub pricers, until the queue is drained."""

    def test_every_job_gets_its_own_result_in_order(self, queue):
        ids = [queue.submit(_body(n=n)) for n in range(3)]
        assert worker.serve(queue.path, price=echo_pricer, drain=True) == 0
        for n, job_id in enumerate(ids):
            job = queue.get(job_id)
            assert job.status == jq.DONE and json.loads(job.result) == {"echo": {"n": n}, "kind": jq.PORTFOLIO}

    def test_each_job_is_run_by_its_kinds_pricer(self, queue):
        """A market-risk job reaches the worker as one (roadmap 3.1)."""
        ids = {kind: queue.submit(_body(), kind) for kind in jq.KINDS}
        worker.serve(queue.path, price=echo_pricer, drain=True)
        assert {kind: json.loads(queue.get(i).result)["kind"] for kind, i in ids.items()} == {k: k for k in jq.KINDS}
        assert set(worker.JOB_KINDS) == set(jq.KINDS)

    def test_a_jobs_artifacts_are_stored_with_its_result(self, queue):
        job_id = queue.submit(_body())
        chunks = [b"\x00" * 8, b"\x01" * 8]
        worker.serve(queue.path, price=lambda job: worker.JobResult("{}", {"npv_cube": chunks}), drain=True)
        assert queue.get(job_id).status == jq.DONE
        assert [queue.artifact(job_id, "npv_cube", i) for i in range(3)] == chunks + [None]

    def test_a_failing_job_fails_only_its_own_row(self, queue):
        """§13.8: with a failure class and the traceback; the worker goes on to the next job."""
        def pricer(job):
            if json.loads(job.request).get("bad"):
                raise KeyError("no market for currency 'GBP'")
            return echo_pricer(job)

        ok_before, bad, ok_after = (queue.submit(_body(bad=b)) for b in (False, True, False))
        worker.serve(queue.path, price=pricer, drain=True)
        failed = queue.get(bad)
        assert (failed.status, failed.failure_class) == (jq.FAILED, jq.MISSING_MARKET_DATA)
        assert "KeyError" in failed.error and "no market for currency 'GBP'" in failed.error
        assert "Traceback" in failed.error
        assert [queue.get(i).status for i in (ok_before, ok_after)] == [jq.DONE, jq.DONE]

    @pytest.mark.parametrize("make, expected", [
        (lambda: KeyError("curve"), jq.MISSING_MARKET_DATA),
        (lambda: __import__("engine.models.ore_builders", fromlist=["x"]).MissingFixingError("fixing"),
         jq.MISSING_MARKET_DATA),
        (lambda: NotImplementedError("product"), jq.UNSUPPORTED_PRODUCT),
        (lambda: __import__("engine.day_count", fromlist=["x"]).UnsupportedDayCountError("30/360 German"),
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

    def test_a_request_that_does_not_parse_is_bad_terms(self, queue):
        """A body that reaches the worker without the route's validation (pydantic's
        `ValidationError` is a `ValueError`)."""
        job_id = queue.submit(_body(market=12345))
        worker.serve(queue.path, drain=True)  # the real pricer: fails in the parse, before JAX work
        assert queue.get(job_id).failure_class == jq.BAD_TERMS

    def test_a_result_the_queue_cannot_store_fails_as_infrastructure(self, queue):
        job_id = queue.submit(_body())
        worker.serve(queue.path, price=lambda job: worker.JobResult(object()), drain=True)
        failed = queue.get(job_id)
        assert failed.failure_class == jq.INFRASTRUCTURE and "storing the result failed" in failed.error

    def test_a_starting_worker_interrupts_its_predecessors_running_job(self, queue):
        orphan, queued = queue.submit(_body()), queue.submit(_body())
        queue.claim("dead-worker")
        worker.serve(queue.path, price=echo_pricer, drain=True)
        assert queue.get(orphan).status == jq.INTERRUPTED
        assert queue.get(queued).status == jq.DONE

    def test_a_second_worker_on_the_queue_exits_at_once(self, queue, monkeypatch):
        monkeypatch.setattr(worker, "LOCK_WAIT_SECONDS", 0.05)
        job_id = queue.submit(_body())
        holder = WorkerLock(queue.path)
        assert holder.acquire()
        try:
            assert worker.serve(queue.path, price=echo_pricer, drain=True) == worker.EXIT_QUEUE_OWNED
        finally:
            holder.release()
        assert queue.get(job_id).status == jq.PENDING

    def test_a_worker_whose_parent_is_gone_claims_nothing(self, queue):
        job_id = queue.submit(_body())
        assert worker.serve(queue.path, price=echo_pricer, parent_pid=_dead_pid()) == 0
        assert queue.get(job_id).status == jq.PENDING

    def test_each_job_records_the_programs_it_compiled(self, queue):
        import jax
        import jax.numpy as jnp

        x = jax.block_until_ready(jnp.ones(3))

        def pricer(job):
            n = json.loads(job.request)["n"]
            jax.jit(lambda v: v * 2.0 + n)(x)  # a new program each job: one compile
            return worker.JobResult("{}")

        ids = [queue.submit(_body(n=n)) for n in range(2)]
        worker.serve(queue.path, price=pricer, drain=True)
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


@pytest.mark.slow
class TestWorkerProcesses:
    """Real worker processes around the stub pricer."""

    def test_a_worker_killed_mid_job_leaves_it_interrupted(self, queue):
        """§13.8: the killed worker's job is `interrupted`; a restarted worker picks up the
        next."""
        long_job, next_job = queue.submit(_body(sleep=120)), queue.submit(_body(n=1))
        first = subprocess.Popen(stub_worker_command(queue.path))
        try:
            _wait_for(lambda: queue.status(long_job) == jq.RUNNING)
        finally:
            first.kill()
            first.wait()
        assert queue.status(long_job) == jq.RUNNING  # nobody has noticed yet
        second = subprocess.Popen(stub_worker_command(queue.path))
        try:
            _wait_for(lambda: queue.status(next_job) == jq.DONE)
        finally:
            second.kill()
            second.wait()
        interrupted = queue.get(long_job)
        assert (interrupted.status, interrupted.error) == (jq.INTERRUPTED, jq.INTERRUPTED_ERROR)

    def test_the_supervisor_restarts_a_dead_worker(self, queue):
        supervisor = WorkerSupervisor(queue.path, command=stub_worker_command(queue.path))
        try:
            supervisor.ensure_running()
            first = supervisor.process
            supervisor.ensure_running()
            assert supervisor.process is first  # alive: left alone
            first.kill()
            first.wait()
            _wait_for(lambda: not worker_lock_held(queue.path))
            supervisor.ensure_running()
            assert supervisor.process is not first and supervisor.process.poll() is None
            job_id = queue.submit(_body(n=7))
            _wait_for(lambda: queue.status(job_id) == jq.DONE)
        finally:
            supervisor.stop()
        assert supervisor.process is None

    def test_supervisors_of_one_queue_share_one_worker(self, queue):
        """Several API processes (uvicorn `--workers`) on one queue: a supervisor starts no
        worker while another process's worker holds the queue."""
        first = WorkerSupervisor(queue.path, command=stub_worker_command(queue.path))
        second = WorkerSupervisor(queue.path, command=stub_worker_command(queue.path))
        try:
            first.ensure_running()
            _wait_for(lambda: worker_lock_held(queue.path))
            second.ensure_running()
            assert second.process is None
            first.stop()
            _wait_for(lambda: not worker_lock_held(queue.path))
            second.ensure_running()  # the queue's worker is gone: this one takes over
            assert second.process is not None
            job_id = queue.submit(_body(n=1))
            _wait_for(lambda: queue.status(job_id) == jq.DONE)
        finally:
            first.stop()
            second.stop()

    def test_a_worker_exits_once_its_parent_is_gone(self, queue):
        """No orphaned engine process outlives a killed API or test run."""
        launcher = ("import subprocess, os, time; "
                    "from tests.support.worker_stubs import stub_worker_command; "
                    f"subprocess.Popen(stub_worker_command({str(queue.path)!r}, parent_pid=os.getpid())); "
                    "time.sleep(120)")
        parent = subprocess.Popen([sys.executable, "-c", launcher], cwd=ROOT)
        try:
            _wait_for(lambda: worker_lock_held(queue.path))
        finally:
            parent.kill()
            parent.wait()
        _wait_for(lambda: not worker_lock_held(queue.path), timeout=30)


class TestRoutesOverTheQueue:
    """The HTTP routes read and write the queue (no worker in this process: `external` mode,
    and the stub worker run in place)."""

    @pytest.fixture
    def client(self, queue, monkeypatch, test_client):
        from engine.api import routes

        monkeypatch.setattr(routes, "_QUEUE", queue)
        monkeypatch.setattr(routes, "_SUPERVISOR", None)  # external: the API starts no worker
        return test_client

    def _submit(self, client, queue):
        body = {"market": shared.market_json(), "trades": [shared.trades_json()["swap-payer"]],
                "scenario_risk": False}
        r = client.post("/portfolio/price", json=body)
        assert r.status_code == 202, r.text
        return r.json()["job_id"]

    def test_the_route_queues_the_body_as_received(self, client, queue):
        job_id = self._submit(client, queue)
        assert client.get(f"/jobs/{job_id}").json() == {
            "kind": "portfolio", "status": "pending", "result": None, "error": None, "failure_class": None}
        assert json.loads(queue.get(job_id).request)["trades"][0]["trade_id"] == "swap-payer"

    def test_a_done_job_returns_the_stored_document(self, client, queue):
        job_id = self._submit(client, queue)
        worker.serve(queue.path, price=echo_pricer, drain=True)
        data = client.get(f"/jobs/{job_id}").json()
        assert data["kind"] == "portfolio" and data["status"] == "done"
        assert data["error"] is None and data["failure_class"] is None
        assert data["result"]["echo"]["trades"][0]["trade_id"] == "swap-payer"

    def test_an_artifact_chunk_is_served_as_stored(self, client, queue):
        job_id = self._submit(client, queue)
        worker.serve(queue.path, price=lambda job: worker.JobResult("{}", {"npv_cube": [b"\x00\x01", b"\x02"]}),
                     drain=True)
        response = client.get(f"/jobs/{job_id}/artifacts/npv_cube/1")
        assert response.status_code == 200 and response.content == b"\x02"
        assert response.headers["content-type"] == "application/octet-stream"
        assert client.get(f"/jobs/{job_id}/artifacts/npv_cube/2").status_code == 404
        assert client.get(f"/jobs/{job_id}/artifacts/pnl/0").status_code == 404

    def test_a_failed_job_returns_its_class_and_traceback(self, client, queue):
        job_id = self._submit(client, queue)

        def pricer(job):
            raise NotImplementedError("no such product")

        worker.serve(queue.path, price=pricer, drain=True)
        data = client.get(f"/jobs/{job_id}").json()
        assert (data["status"], data["failure_class"], data["result"]) == ("failed", jq.UNSUPPORTED_PRODUCT, None)
        assert "NotImplementedError: no such product" in data["error"]

    def test_an_interrupted_job_says_so(self, client, queue):
        job_id = self._submit(client, queue)
        queue.claim("dead-worker")
        queue.interrupt_running()
        data = client.get(f"/jobs/{job_id}").json()
        assert (data["status"], data["error"]) == ("interrupted", jq.INTERRUPTED_ERROR)


# --- real pricing ---------------------------------------------------------------------------

def _pricing_body(trades, samples=64, **overrides) -> dict:
    import ORE

    dates = [(shared.ASOF + ORE.Period(m, ORE.Months)).ISO() for m in (6, 12)]
    body = {"market": shared.market_json(), "trades": trades,
            "simulation": {"dates": dates, "base_currency": "USD", "samples": samples, "seed": 5,
                           "ir": {"USD": {"model": "HullWhite", "reversion": 0.03, "volatility": 0.01}}}}
    body.update(overrides)
    return body


def _trades(*names):
    return [shared.trades_json()[n] for n in names]


def _direct(body: dict):
    from engine.api.market_schemas import MarketPortfolioRequestSchema
    from engine.portfolio import price_portfolio

    return price_portfolio(MarketPortfolioRequestSchema.model_validate(body).to_dataclass())


def test_the_worker_prices_the_request_the_route_validated(monkeypatch):
    """The worker builds its dataclass request from the stored body exactly as the route did
    (until 1.8 the request travelled pickled, its ORE dates frozen as text): every part of the
    run configuration, the trades' dates and fixings and the market arrive equal."""
    import engine.portfolio
    from engine.api.market_schemas import MarketPortfolioRequestSchema

    f32 = {"storage": "float32", "compute": "float32", "accumulate": "float32"}
    body = _pricing_body(
        _trades("swap-receiver-seasoned-icma", "european-payer", "bermudan-payer-physical", "bond"),
        pricing={"european": "Jamshidian", "jamshidian": {"reversion": 0.03, "volatility": 0.01},
                 "bermudan": {"n_per_std": 12, "std_devs": 4.0}, "recalibrate": False},
        compute_greeks=True, greeks={"method": "AD", "sensitivity": {"curve_tenors": ["1Y", "2Y"]}},
        precision={"simulation": f32, "pricing": {"storage": "float32"}, "by_trade": {"bond": f32}},
        pfe_quantiles=[0.9, 0.99])
    raw = json.dumps(body).encode()
    seen = {}

    class Stop(Exception):
        pass

    def capture(request):
        seen["request"] = request
        raise Stop

    monkeypatch.setattr(engine.portfolio, "price_portfolio", capture)
    with pytest.raises(Stop):
        worker.price_job(jq.Job(id="j", kind=jq.PORTFOLIO, status=jq.RUNNING, request=raw))
    route = MarketPortfolioRequestSchema.model_validate(body).to_dataclass()
    assert seen["request"] == route
    assert seen["request"].trades[0].fixings == route.trades[0].fixings and route.trades[0].fixings


def test_a_cube_by_reference_without_scenario_risk_is_none_not_a_failure():
    """`cube_output: "artifact"` with `scenario_risk: false` has no cube to return: the job
    completes with neither the cube nor a reference (it once failed building a reference for a
    cube with no trade axis)."""
    body = {"market": shared.market_json(), "trades": _trades("swap-payer", "bond"), "scenario_risk": False,
            "cube_output": "artifact"}
    result = worker.price_job(jq.Job(id="j", kind=jq.PORTFOLIO, status=jq.RUNNING, request=json.dumps(body).encode()))
    document = json.loads(result.document)
    assert result.artifacts == {}
    assert document["npv_cube"] is None and document["npv_cube_artifact"] is None
    assert [row["trade_id"] for row in document["trades"]] == ["swap-payer", "bond"]


@pytest.mark.slow
class TestEngineWorkerPricing:
    """The real worker (`price_job`) in its own process, started by the supervisor."""

    @pytest.fixture
    def running(self, queue):
        supervisor = WorkerSupervisor(queue.path)
        supervisor.ensure_running()
        yield queue
        supervisor.stop()

    @staticmethod
    def _result(queue, job_id, timeout=600):
        job = _wait_for(lambda: (j := queue.get(job_id)).status in jq.TERMINAL and j, timeout=timeout)
        assert job.status == jq.DONE, job.error
        return job

    def test_jobs_queued_together_give_the_bits_of_jobs_run_one_after_another(self, running):
        """§13.8; and every job equals the direct call in this process, at float64 and
        float32."""
        f32 = {"storage": "float32", "compute": "float32", "accumulate": "float32"}
        bodies = [_pricing_body(_trades("swap-payer", "european-payer")),
                  _pricing_body(_trades("swap-payer", "european-payer"),
                                precision={"simulation": f32, "market": f32, "pricing": f32}),
                  _pricing_body(_trades("bond", "swap-receiver-seasoned-icma"), samples=32)]
        together = [running.submit(json.dumps(b).encode()) for b in bodies]
        together = [json.loads(self._result(running, j).result) for j in together]
        alone = [json.loads(self._result(running, running.submit(json.dumps(b).encode())).result) for b in bodies]
        assert together == alone
        for body, result in zip(bodies, together):
            direct = _direct(body)
            np.testing.assert_array_equal(np.asarray(result["npv_cube"]), np.asarray(direct.npv_cube))
            assert result["base_npv"] == direct.base_npv
            np.testing.assert_array_equal(np.asarray(result["exposure"]["epe"]), np.asarray(direct.exposure.epe))
        assert together[0]["npv_cube"] != together[1]["npv_cube"], "the float32 job was not priced in float32"

    def test_a_worker_keeps_its_programs_on_disk_beside_its_queue(self, queue):
        """A worker started with no cache settings (`main`, as the supervisor starts it) keeps
        JAX's persistent compilation cache in `xla-cache/` beside its queue file, so a
        restarted worker reads its programs back (roadmap 2.4, I-53)."""
        import os

        cache = queue.path.parent / worker.COMPILATION_CACHE_DIRNAME
        env = {k: v for k, v in os.environ.items() if not k.startswith(("JAX_COMPILATION_CACHE", "JAX_PERSISTENT_CACHE"))}
        process = subprocess.Popen(
            [sys.executable, "-c", "from engine.api.worker import main; main()", "--queue", str(queue.path)],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL)
        try:
            self._result(queue, queue.submit(json.dumps(_pricing_body(_trades("swap-payer"), samples=8)).encode()))
        finally:
            process.kill()
            process.wait()
        assert cache.is_dir() and any(cache.iterdir()), "the worker compiled without a persistent cache"

    def test_a_second_identical_job_compiles_nothing(self, running):
        """§13.9: the worker keeps its programs, so a repeated job shape builds no program,
        compiled or read from the persistent cache (until 1.8 every pool worker built its own)."""
        body = json.dumps(_pricing_body(_trades("swap-payer", "european-payer", "bermudan-payer-physical"),
                                        pricing={"bermudan": {"n_per_std": 12, "std_devs": 4.0}})).encode()
        first = self._result(running, running.submit(body))
        second = self._result(running, running.submit(body))
        assert first.compiles > 0, "the compile counter saw nothing: has JAX renamed its compile event?"
        assert second.compiles == 0
        assert first.result == second.result
