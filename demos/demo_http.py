"""
The HTTP plumbing the API demos share (`demo_api.py`, `demo_structured.py`,
`demo_profile_small.py`): starting an `engine.api` server, submitting a portfolio job and
waiting for it. The demos import it as `demo_http` (run as scripts from demos/), the tests as
`demos.demo_http`; the engine never imports it.

A demo starts `uvicorn engine.api.app:app` on `API_BASE` as a subprocess, unless
`JAX_RISK_ENGINE_DEMO_SKIP_SERVER=1`, when it uses a server already running there. The server's
environment is the demo's own (`server_environment`) with GPU preallocation turned off unless
it says otherwise (`XLA_PYTHON_CLIENT_PREALLOCATE`, decision A-22): XLA's default takes 75% of
a GPU in every process that opens it, and a developer's machine runs the engine worker beside
whatever else uses the card (a notebook, the tests). A deployment with one engine worker per
GPU keeps JAX's default. The engine worker sets its other process settings itself
(deterministic GPU kernels, the compilation cache; `engine.api.worker`).
"""
import os
import subprocess
import sys
import time
from typing import Mapping

import httpx

API_BASE = "http://127.0.0.1:8000"
#: Whether the demo starts (and stops) its own server.
MANAGE_SERVER = os.environ.get("JAX_RISK_ENGINE_DEMO_SKIP_SERVER") != "1"
#: A job's statuses after which it changes no more (docs/reference/http-api.md).
FINISHED = ("done", "failed", "interrupted")


def server_environment(environ: Mapping[str, str], **variables: str) -> dict:
    """The environment a demo's server runs in: `environ` with the demo's own `variables`, and
    GPU preallocation off unless either sets it."""
    return {"XLA_PYTHON_CLIENT_PREALLOCATE": "false", **environ, **variables}


def wait_until_healthy(timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if httpx.get(f"{API_BASE}/health", timeout=2.0).status_code == 200:
                return
        except httpx.TransportError:
            # Not listening yet, or still importing JAX/ORE: expected during startup.
            pass
        time.sleep(0.5)
    raise RuntimeError(f"server at {API_BASE} did not become healthy within {timeout_s}s")


def start_server(**variables: str) -> subprocess.Popen:
    """Start uvicorn in `server_environment(os.environ, **variables)` and wait until it is
    healthy. The engine worker it starts inherits the environment."""
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "engine.api.app:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=server_environment(os.environ, **variables),
    )
    print(f"launched uvicorn (pid {process.pid}), waiting for {API_BASE}/health ...")
    try:
        wait_until_healthy()
    except Exception:
        process.terminate()
        raise
    print("server is up")
    return process


def stop_server(process: subprocess.Popen) -> None:
    process.terminate()
    process.wait(timeout=10)
    print(f"stopped uvicorn (pid {process.pid})")


def submit_and_wait(request_body: dict, poll_seconds: float = 2.0) -> dict:
    """Submit a portfolio job (docs/reference/http-api.md#why-async-not-sync), poll it until it
    finishes, and return its result; a job that failed or was interrupted raises."""
    submit = httpx.post(f"{API_BASE}/portfolio/price", json=request_body, timeout=30.0)
    if submit.status_code != 202:
        raise RuntimeError(f"submission failed ({submit.status_code}): {submit.text}")
    job_id = submit.json()["job_id"]
    print(f"job_id: {job_id} (202 Accepted -- pricing is running in the background)")

    start = time.time()
    while True:
        poll = httpx.get(f"{API_BASE}/portfolio/price/{job_id}", timeout=30.0)
        poll.raise_for_status()
        status = poll.json()
        print(f"  [{time.time() - start:6.1f}s] status: {status['status']}")
        if status["status"] in FINISHED:
            break
        time.sleep(poll_seconds)

    if status["status"] != "done":
        raise RuntimeError(f"pricing job {status['status']}:\n{status['error']}")
    return status["result"]
