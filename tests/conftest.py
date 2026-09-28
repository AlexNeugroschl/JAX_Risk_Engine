"""
Shared fixtures: the demo scenarios from `engine.simulation.demo_scenarios`, a minimal
`PortfolioRequest` and an API client. x64 is enabled here, before any test builds an array.
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


# Session-scoped fixtures return immutable values or fresh, side-effect-free objects; tests
# needing a variant derive one with with_scenarios() or dataclasses.replace().


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
    """Single-currency, two-correlated-rate-factor scenario for the swap and risk-statistics
    tests."""
    return single_currency_swap_demo_config()


@pytest.fixture(scope="session")
def make_flat_yield_curves():
    """Factory: make_flat_yield_curves(disc_rate, fwd_rate) -> cube."""
    return flat_yield_curves


def with_scenarios(config, scenarios: int):
    """Helper (not a fixture): the shared demo scenario at another sample size."""
    return dataclasses.replace(config, scenarios=scenarios)


@pytest.fixture
def portfolio_request():
    """A minimal valid `PortfolioRequest` (one swap, few scenarios). Function-scoped so each
    test has its own copy to vary."""
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
    return PortfolioRequest(market=market, trades=[swap_cfg], pfe_quantiles=(0.95,))


@pytest.fixture(scope="session")
def test_client():
    """In-process `TestClient` over `engine.api.app`. Session-scoped; the only state is the
    job store, and each submission gets a unique job_id."""
    from fastapi.testclient import TestClient
    from engine.api.app import app
    return TestClient(app)
