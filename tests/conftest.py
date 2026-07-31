"""
Shared pytest fixtures for the engine test suite.

Every test file used to hand-roll its own copy of the demo scenario config
(and, for the swap/risk-statistics tests, its own ORE flat-curve builder) --
already duplicated across engine/instruments/interest_rate_swap.py,
engine/aggregate_statistics/risk_statistics.py, and the test files
themselves, and drifting slightly out of sync between copies. This module
re-exports the canonical scenario builders from engine.scenarios as
fixtures so every test file draws from one source.

x64 is enabled here at collection time (before any test constructs a
float64 array) so individual test files don't each need their own
`jax.config.update("jax_enable_x64", True)` at import time.
"""
import jax
jax.config.update("jax_enable_x64", True)

import dataclasses

import pytest

from engine.scenarios import (
    EVAL_DATE,
    SWAP_DEMO_MATURITIES,
    cross_asset_demo_config,
    flat_yield_curves,
    single_currency_swap_demo_config,
)


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
    """Two-equity, two-rate-factor scenario (see engine.scenarios docstring)."""
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
