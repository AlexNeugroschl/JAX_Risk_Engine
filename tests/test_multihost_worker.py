"""
The engine worker on several hosts (I-61; `engine.api.hosts`): a worker set of two processes,
each its own host of a `jax.distributed` cluster on localhost (JAX's gloo CPU collectives), with
the stub pricer (tests/support/worker_stubs.py), serving one queue:

  * every host runs each job, the primary (process 0) alone claims it and stores its result;
  * a failure the request determines is reported by every host, the job failed with its class,
    and the set goes on to the next job;
  * a cancel stops every host at the same phase boundary;
  * a failure of one host alone stops the whole set (JAX's distributed service stops the other
    host), and the restarted set runs the job again, once;
  * with the engine's own pricer, two hosts of two devices each run a run request (every
    analytic, the scenarios split across the four devices) to the document one host gives.

The queue's side (the attempts, putting a job back) is tests/test_engine_worker.py.
"""
import json
import os
import socket
import subprocess
import sys
import time

import pytest

from engine.api import job_queue as jq
from engine.api.hosts import EXIT_HOST_FAILED
from engine.api.job_queue import JobQueue
from tests.support.worker_stubs import ROOT, distributed_worker_command

HOSTS = 2


pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not sys.platform.startswith("linux"), reason="JAX's CPU collectives (gloo) are "
                                                                              "built on Linux")]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _start(queue: JobQueue, tmp_path, real: bool = False, devices: int = 1):
    """The worker set's processes on `queue`, host 0 first, each host with `devices` CPU devices
    and the stub pricer (or the engine's, `real`)."""
    port = _free_port()
    processes = []
    for host in range(HOSTS):
        env = {**os.environ, "JAX_PLATFORMS": "cpu", "JAX_CPU_COLLECTIVES_IMPLEMENTATION": "gloo",
               "JAX_RISK_COORDINATOR": f"localhost:{port}", "JAX_RISK_NUM_PROCESSES": str(HOSTS),
               "JAX_RISK_PROCESS_ID": str(host), "JAX_ENABLE_COMPILATION_CACHE": "false",
               "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}"}
        log = open(tmp_path / f"host{host}.log", "ab")
        command = distributed_worker_command(queue.path, tmp_path / f"host{host}.lock", real)
        processes.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=log,
                                          stdin=subprocess.DEVNULL))
    return processes


def _wait(processes, tmp_path, timeout: float = 300.0):
    """Every process's exit status, once all have exited."""
    deadline = time.monotonic() + timeout
    try:
        return [p.wait(timeout=max(1.0, deadline - time.monotonic())) for p in processes]
    except subprocess.TimeoutExpired:
        for p in processes:
            p.kill()
        logs = "\n".join((tmp_path / f"host{h}.log").read_text(errors="replace")[-3000:] for h in range(HOSTS))
        pytest.fail(f"the worker set did not exit:\n{logs}")


def _run_set(queue: JobQueue, tmp_path, **options):
    return _wait(_start(queue, tmp_path, **options), tmp_path)


def _submit(queue: JobQueue, **body) -> str:
    return queue.submit(json.dumps(body).encode())


@pytest.fixture
def queue(tmp_path):
    q = JobQueue(tmp_path / "jobs.sqlite3")
    yield q
    q.close()


def test_every_host_runs_each_job_and_the_primary_stores_it(queue, tmp_path):
    ran = tmp_path / "ran"
    ran.mkdir()
    ids = [_submit(queue, n=n, ran_in=str(ran)) for n in range(2)]
    assert _run_set(queue, tmp_path) == [0] * HOSTS
    for n, job_id in enumerate(ids):
        job = queue.get(job_id)
        assert job.status == jq.DONE, job.error
        assert json.loads(job.result)["echo"]["n"] == n
        assert job.attempts == 1
        assert sorted(p.name for p in ran.glob(f"{job_id}.*")) == [f"{job_id}.{h}" for h in range(HOSTS)]


def test_a_failure_the_request_determines_is_reported_by_every_host(queue, tmp_path):
    failing = _submit(queue, fail="no such product")
    after = _submit(queue, n=1)
    assert _run_set(queue, tmp_path) == [0] * HOSTS
    job = queue.get(failing)
    assert (job.status, job.failure_class) == (jq.FAILED, jq.BAD_TERMS)
    assert "no such product" in job.error
    assert queue.get(after).status == jq.DONE


def test_a_cancel_stops_every_host_at_one_phase_boundary(queue, tmp_path):
    job_id = _submit(queue, phases=200, sleep=0.05)
    processes = _start(queue, tmp_path)
    deadline = time.monotonic() + 120
    while queue.status(job_id) != jq.RUNNING and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(0.5)
    assert queue.cancel(job_id) == jq.RUNNING
    assert _wait(processes, tmp_path) == [0] * HOSTS
    job = queue.get(job_id)
    assert job.status == jq.CANCELLED and "phase boundary" in job.error


def test_a_host_s_own_failure_stops_the_set_and_its_job_runs_again(queue, tmp_path):
    """Host 1's device fails on the job's first run: host 1 stops its process, which stops host 0,
    and the job is left running. The set started again puts it back, once, and runs it whole."""
    marker = tmp_path / "failed-once"
    job_id = _submit(queue, host_failure=[1, str(marker)])
    first = _run_set(queue, tmp_path)
    assert first[1] == EXIT_HOST_FAILED and first[0] != 0, first
    assert queue.get(job_id).status == jq.RUNNING and marker.exists()
    assert _run_set(queue, tmp_path) == [0] * HOSTS
    job = queue.get(job_id)
    assert job.status == jq.DONE, job.error
    assert job.attempts == 2


def test_a_run_request_on_two_hosts_is_the_one_hosts_run(queue, tmp_path):
    """Every analytic of a run request (NPV, exposure, sensitivities by bumps, market risk), the
    paths and scenarios split across both hosts' four devices, gives the result document of the
    same job priced in one process on one device, number for number to the rounding a smaller
    batch moves (every array split across hosts read whole for the document, I-61)."""
    import numpy as np

    from engine.api.job_queue import Job
    from engine.api.worker import price_job
    from engine.risk.market.factors import RateRiskFactors
    from tests.support import portfolio as shared
    from tests.support.runs import models, run_body, simulation

    labels = RateRiskFactors.from_market(shared.market()).labels()
    body = run_body(["npv", "exposure", "sensitivities", "market_risk"],
                    ["swap-payer", "european-payer", "bermudan-payer-physical", "bond"],
                    outputs={"npv_cube": "inline", "pnl": "inline"}, simulation=simulation(samples=64),
                    models=models(), pricing={"bermudan": {"n_per_std": 8, "std_devs": 4.0}},
                    market_risk={"scenarios": {"source": "monte-carlo", "factors": labels, "horizon_days": 10,
                                               "covariance": (np.eye(len(labels)) * 1e-6).tolist(),
                                               "num_scenarios": 64}, "quantiles": [0.95]})
    job_id = queue.submit(json.dumps(body).encode())
    assert _run_set(queue, tmp_path, real=True, devices=2) == [0] * HOSTS
    job = queue.get(job_id)
    assert job.status == jq.DONE, job.error
    two_hosts = json.loads(job.result)
    one_host = json.loads(price_job(Job(id=job_id, kind=jq.RUN, status=jq.RUNNING, request=job.request)).document)
    assert _numbers(two_hosts) == pytest.approx(_numbers(one_host), rel=1e-9, abs=1e-7)
    devices = two_hosts["exposure"]["precision"]["devices"]
    assert len(devices) == 4, devices


def _numbers(document, path=""):
    """Every number of a result document by its path, but the devices and timings that name where
    it ran."""
    if isinstance(document, dict):
        return {k: v for key, value in document.items() if key not in ("devices", "backend", "engine", "run_id")
                for k, v in _numbers(value, f"{path}/{key}").items()}
    if isinstance(document, list):
        return {k: v for i, value in enumerate(document) for k, v in _numbers(value, f"{path}/{i}").items()}
    if isinstance(document, (int, float)) and not isinstance(document, bool):
        return {path: float(document)}
    return {}
