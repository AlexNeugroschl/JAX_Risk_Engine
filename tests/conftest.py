"""
Shared fixtures: the demo scenarios' evaluation date, a minimal `PortfolioRequest` and an API
client. x64 is enabled here, before any test builds an array (importing `engine` enables it
too).

XLA programs are cached on disk across runs (JAX's persistent compilation cache), in
`.jax_cache/` at the repository root unless `JAX_COMPILATION_CACHE_DIR` names another
directory. A program is keyed on its HLO, the compile options and the jax/jaxlib versions, so
a hit is the executable a fresh compile would build. Set through the environment as well, so
worker processes the tests start use it too. Delete the directory to measure cold compiles.
"""
import os
from pathlib import Path

os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path(__file__).resolve().parents[1] / ".jax_cache"))
# Cache every program: the suite compiles thousands of small ones, each cheaper to read back.
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", os.environ["JAX_COMPILATION_CACHE_DIR"])
jax.config.update("jax_persistent_cache_min_compile_time_secs",
                  float(os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"]))
jax.config.update("jax_persistent_cache_min_entry_size_bytes",
                  int(os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"]))

import dataclasses

import ORE
import pytest

from demos.demo_scenarios import EVAL_DATE, demo_market, demo_simulation
from engine.instruments.swap import SwapConfig
from engine.portfolio import PortfolioRequest, RunConfig


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


@pytest.fixture(scope="session")
def test_client():
    """In-process `TestClient` over `engine.api.app`. Session-scoped; the only state is the
    job store, and each submission gets a unique job_id."""
    from fastapi.testclient import TestClient
    from engine.api.app import app
    return TestClient(app)
