"""
Shared fixtures: the demo scenarios' evaluation date, a minimal `PortfolioRequest` and an API
client. x64 is enabled here, before any test builds an array (importing `engine` enables it
too).

Two device settings, each unless the environment sets it (decision A-22; importing `engine`
sets neither): no GPU preallocation (`XLA_PYTHON_CLIENT_PREALLOCATE=false`), since every test
process (`-n`) and every engine worker the tests start opens the same GPU, and XLA's default
takes 75% of it per process; and deterministic GPU kernels, as the engine worker runs
(`engine.api.worker.deterministic_kernels_environment`), since the tests price in-process. Both
are environment variables, so the worker processes the tests start inherit them, and both are
read when a process first opens a device. Neither changes anything on a CPU.

XLA programs are cached on disk across runs (JAX's persistent compilation cache), in
`.jax_cache/` at the repository root unless `JAX_COMPILATION_CACHE_DIR` names another
directory. A program is keyed on its HLO, the compile options and the jax/jaxlib versions, so
a hit is the executable a fresh compile would build. Set through the environment as well, so
worker processes the tests start use it too. Delete the directory to measure cold compiles.

The engine worker's lock is one per host (`engine.api.job_queue.worker_lock_path`, I-106), so each
test process takes a lock of its own (`JAX_RISK_WORKER_LOCK`), as each has its own job queue:
otherwise the first process's worker would serve its queue alone, and every other process's
runs would be refused.
"""
import os
import sys
import tempfile
from pathlib import Path

from engine.api.worker import deterministic_kernels_environment

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.update(deterministic_kernels_environment(os.environ))
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path(__file__).resolve().parents[1] / ".jax_cache"))
# Cache every program: the suite compiles thousands of small ones, each cheaper to read back.
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")
# Assigned, not defaulted: xdist's processes inherit the controller's environment, and each needs
# its own.
os.environ["JAX_RISK_WORKER_LOCK"] = str(Path(tempfile.gettempdir()) / "jax-risk-engine-tests"
                                         / f"worker-{os.getpid()}.lock")

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", os.environ["JAX_COMPILATION_CACHE_DIR"])
jax.config.update("jax_persistent_cache_min_compile_time_secs",
                  float(os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"]))
jax.config.update("jax_persistent_cache_min_entry_size_bytes",
                  int(os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"]))

import ctypes
import dataclasses
import gc

import ORE
import pytest

from demos.demo_scenarios import EVAL_DATE, demo_market, demo_simulation
from engine.instruments.swap import SwapConfig
from engine.run import PortfolioRequest, RunConfig


@pytest.fixture(scope="session")
def eval_date():
    return EVAL_DATE


@pytest.fixture
def portfolio_request():
    """A minimal valid `PortfolioRequest`: one USD swap on the demo market, simulated by the
    Hull-White model on a short grid with few paths. Function-scoped so each test has its own
    copy to vary."""
    swap_cfg = SwapConfig(notional=1_000_000.0, fixed_rate=0.032, payer=True, swap_tenor="2Y",
                          evaluation_date=EVAL_DATE, trade_id="swap-2y")
    dates = tuple(EVAL_DATE + ORE.Period(m, ORE.Months) for m in (6, 12, 18, 24))
    simulation = demo_simulation("HullWhite", samples=64, dates=dates, currencies=("USD",))
    return PortfolioRequest(market=demo_market(("USD",)), trades=[swap_cfg],
                            config=RunConfig(simulation=simulation), pfe_quantiles=(0.95,))


def with_simulation(request: PortfolioRequest, **changes) -> PortfolioRequest:
    """Helper (not a fixture): `request` with its simulation's fields replaced."""
    simulation = dataclasses.replace(request.config.simulation, **changes)
    return dataclasses.replace(request, config=dataclasses.replace(request.config, simulation=simulation))


@pytest.fixture(scope="module", autouse=True)
def _clear_jax_caches_after_each_module():
    """Drop this process's compiled XLA programs after each module (I-27). JAX keeps every
    program for the process's lifetime, so a test process grows with each module it runs: four
    of them on CI's 4-vCPU, 16 GB runner (`-n auto`) ran out of memory, and the runner reported
    both jobs "canceled". The next module compiles what it needs again, or reads it back from
    the persistent cache above.

    On Linux that alone does not shrink the process: glibc keeps freed memory in its per-thread
    arenas (XLA compiles on many threads), so each process held its high-water mark, the heaviest
    module it had run, and four of them still reached ~19 GB. `malloc_trim` returns the freed
    pages to the OS, which keeps one process near 1 GB between modules. Windows' allocator
    returns them by itself."""
    yield
    jax.clear_caches()
    gc.collect()
    _return_freed_memory_to_the_os()


def _return_freed_memory_to_the_os():
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):  # not glibc (musl, ...): nothing to trim
            pass


@pytest.fixture(scope="module", autouse=True)
def _stop_the_engine_worker_after_each_module():
    """Stop the engine worker a module's HTTP jobs started, so the modules after it do not run
    beside an idle JAX process holding its compiled programs (the next job starts a new one).
    Until 2026-10-04 the pool's modules shut their pools down the same way."""
    yield
    routes = sys.modules.get("engine.api.routes")
    if routes is not None:
        routes.shutdown_worker()


@pytest.fixture(scope="session")
def test_client(tmp_path_factory):
    """In-process `TestClient` over `engine.api.app`, on a job queue of this session's own (each
    xdist process has its own, and so its own engine worker under its own lock). Session-scoped;
    the only state is the queue, and each submission gets a unique run_id. The worker, started by the first job,
    is stopped at the end of the session (it would exit with this process anyway)."""
    from fastapi.testclient import TestClient
    from engine.api import routes
    from engine.api.app import app
    routes.configure_jobs(tmp_path_factory.mktemp("jobs") / "jobs.sqlite3", mode="spawn")
    yield TestClient(app)
    routes.shutdown_worker()
