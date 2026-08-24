"""
Shared pytest fixtures for the engine test suite.

Every test file used to hand-roll its own copy of the demo scenario config
(and, for the swap/VaR-ES tests, its own ORE flat-curve builder) --
already duplicated across engine/instruments/swap.py,
engine/risk/var_es.py, and the test files
themselves, and drifting slightly out of sync between copies. This module
re-exports the canonical scenario builders from engine.simulation.demo_scenarios
as fixtures so every test file draws from one source.

x64 is enabled here at collection time (before any test constructs a
float64 array) so individual test files don't each need their own
`jax.config.update("jax_enable_x64", True)` at import time.
"""
import jax
jax.config.update("jax_enable_x64", True)

import dataclasses

import pytest

from engine.simulation.demo_scenarios import (
    EVAL_DATE,
    SWAP_DEMO_MATURITIES,
    cross_asset_demo_config,
    flat_yield_curves,
    single_currency_swap_demo_config,
)
from engine.simulation.market_model import EquityConfig, RatesConfig, SimulationConfig, ZeroCurveConfig
from engine.instruments.swap import SwapConfig
from engine.portfolio import PortfolioRequest


# session-scoped: every fixture below returns either an immutable value or a
# freshly-built, side-effect-free dataclass/function -- safe to share across
# every test in the session (tests that need a variant use with_scenarios()
# or dataclasses.replace() to derive their own copy rather than mutating
# the shared instance).


@pytest.fixture(scope="session")
def eval_date():
    return EVAL_DATE


@pytest.fixture(scope="session")
def swap_demo_maturities():
    return SWAP_DEMO_MATURITIES


@pytest.fixture(scope="session")
def cross_asset_config():
    """Two-equity, two-rate-factor scenario (see engine.simulation.demo_scenarios docstring)."""
    return cross_asset_demo_config()


@pytest.fixture(scope="session")
def swap_config():
    """Single-currency, two-correlated-rate-factor scenario sized for the
    swap/risk-statistics demos and their ORE cross-checks."""
    return single_currency_swap_demo_config()


@pytest.fixture(scope="session")
def make_flat_yield_curves():
    """Factory fixture: make_flat_yield_curves(disc_rate, fwd_rate) -> cube,
    so tests can request more than one (disc_rate, fwd_rate) pair."""
    return flat_yield_curves


def with_scenarios(config, scenarios: int):
    """Small helper (not a fixture) for tests that need the shared demo
    scenario at a different Monte Carlo sample size than the default."""
    return dataclasses.replace(config, scenarios=scenarios)


@pytest.fixture
def portfolio_request():
    """A minimal, valid PortfolioRequest (one swap, small scenario count)
    for tests of engine.portfolio.price_portfolio / engine/api that just
    need SOME well-formed request, not a specific portfolio shape --
    function-scoped (not session) since engine.portfolio.price_portfolio
    mutates nothing on the request itself, but tests commonly want their
    own independent copy to modify via dataclasses.replace."""
    zero_curve = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[0.03] * 6)
    swap_cfg = SwapConfig(
        notional=1_000_000.0, fixed_rate=0.032, payer=True,
        discount_curve_index=0, forward_curve_index=0,
        swap_tenor="2Y", evaluation_date=EVAL_DATE,
    )
    market = SimulationConfig(
        time_grid=[0.0, 0.5, 1.0, 1.5, 2.0],
        scenarios=64,
        equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[0.0]]),
        rates=RatesConfig(
            initial_rates=[0.03], theta=[0.03], mean_reversion=[0.03],
            initial_zero_curves=[zero_curve],
        ),
        joint_covariance=[[0.04, 0.0], [0.0, 0.0001]],
    )
    return PortfolioRequest(market=market, trades=[swap_cfg], percentiles=(0.95,))


@pytest.fixture(scope="session")
def test_client():
    """FastAPI TestClient over engine.api.app -- in-process, no running
    server needed (backed by httpx). Session-scoped: the app itself is
    stateless aside from the in-process job store, which tests should treat
    as append-only (unique job_ids per submission), so sharing one client
    across tests is safe and avoids re-constructing the FastAPI app
    per-test."""
    from fastapi.testclient import TestClient
    from engine.api.app import app
    return TestClient(app)
