"""
Shared fixtures: the demo scenarios' evaluation date, a minimal `PortfolioRequest` and an API
client. x64 is enabled here, before any test builds an array (importing `engine` enables it
too).
"""
import jax
jax.config.update("jax_enable_x64", True)

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
