"""
ORE parity for the market-risk path, end to end: every shocked scenario is
repriced independently in ORE, and ORE's own `RiskStatistics` computes VaR
and ES from ORE's P&L vector.

    engine: scenarios -> engine.market_risk.run_market_risk -> VaR/ES
    ORE:    same shifted pillar rates -> ORE.ZeroCurve -> ORE instruments and
            engines -> P&L vector -> ORE.RiskStatistics -> VaR/ES

ORE instruments and engines:

    swap        MakeVanillaSwap + DiscountingSwapEngine, separate forwarding
                and discounting curves
    European    Swaption + JamshidianSwaptionEngine on ORE.HullWhite
    bond        FixedRateBond (ACT/ACT ISMA, no settlement lag) + DiscountingBondEngine
    Bermudan    NumericLgmMultiLegOptionEngine through an in-process OREApp
                (engine.validation.ore_lgm_oracle)

Measured agreement per scenario, which the tolerances below are set from:
swap ~1e-14 and bond ~1e-16 relative (the same arithmetic); Bermudan ~2e-13
(the same LGM grid); European ~3e-7, inside the ~2e-6 envelope the existing
Jamshidian parity tests document for sloped curves
(tests/test_european_swaption.py).

ORE's `ParametricVarCalculator` and `ExposureCalculator` have no constructor
in the Python bindings, so VaR/ES parity is established this way -- full
revaluation in ORE plus ORE's statistics -- rather than by calling an ORE VaR
analytic.
"""
import numpy as np
import ORE
import pytest

from engine.market_risk import (
    MarketRiskRequest,
    historical_scenarios,
    monte_carlo_scenarios,
    run_market_risk,
)
from engine.models.ore_builders import build_vanilla_swap
from engine.validation.ore_lgm_oracle import ore_lgm_swaption_npv
from tests import market_risk_support as m

FACTORS = m.factors()
OIS, IBOR = FACTORS.slice_of(0), FACTORS.slice_of(1)
BASE = FACTORS.base_rates()


def _ore_bermudan_npv(cfg, rates) -> float:
    swap = build_vanilla_swap(
        notional=cfg.notional, fixed_rate=cfg.fixed_rate, payer=cfg.payer, swap_tenor=cfg.swap_tenor,
        index_tenor_months=cfg.index_tenor_months, floating_spread=cfg.floating_spread,
        evaluation_date=cfg.evaluation_date,
    )
    return ore_lgm_swaption_npv(
        evaluation_date=m.TODAY, curve_times=m.PILLAR_TIMES, curve_rates=list(rates), swap=swap,
        notional=cfg.notional, fixed_rate=cfg.fixed_rate, payer=cfg.payer,
        floating_spread=cfg.floating_spread, index_tenor_months=cfg.index_tenor_months, style="Bermudan",
        exercise_dates=list(cfg.exercise_dates), hw_a=cfg.hw_a, hw_sigma=cfg.hw_sigma,
        n_per_std=cfg.n_per_std, std_devs=cfg.std_devs,
    ).npv


def _ore_npv(cfg, factor_vector) -> float:
    """ORE's price of one trade at a full factor vector."""
    name = type(cfg).__name__
    if name == "SwapConfig":
        return m.ore_swap_npv(cfg, factor_vector[OIS], factor_vector[IBOR])
    if name == "SwaptionConfig":
        return m.ore_european_npv(cfg, factor_vector[OIS])
    if name == "BondConfig":
        return m.ore_bond_npv(cfg, factor_vector[OIS])
    return _ore_bermudan_npv(cfg, factor_vector[OIS])


def _ore_pnl(trades, shifts) -> np.ndarray:
    """`[S, N]` P&L of every trade under every scenario, priced by ORE."""
    base = [_ore_npv(cfg, BASE) for cfg in trades]
    return np.asarray([[_ore_npv(cfg, BASE + row) - b for cfg, b in zip(trades, base)] for row in shifts])


def _ore_statistics(pnl: np.ndarray) -> "ORE.RiskStatistics":
    stats = ORE.RiskStatistics()
    stats.add(ORE.DoubleVector([float(v) for v in pnl]))
    return stats


# ---------------------------------------------------------------------------
# Per-scenario revaluation, instrument by instrument
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("make, scenarios, rtol", [
    (m.swap, 32, 1e-12),
    (m.bond, 32, 1e-12),
    (m.european, 32, 1e-6),
    (m.bermudan, 6, 1e-10),
], ids=["swap", "bond", "european", "bermudan"])
def test_every_scenario_revalues_as_ore_does(make, scenarios, rtol):
    cfg = make()
    shocks = monte_carlo_scenarios(FACTORS, m.covariance(), horizon_days=10, num_scenarios=scenarios, seed=11)
    result = run_market_risk(MarketRiskRequest([cfg], shocks))

    engine_values = result.base_npv_per_trade[0] + np.asarray(result.pnl[:, 0])
    ore_values = np.asarray([_ore_npv(cfg, BASE + row) for row in shocks.shifts])
    np.testing.assert_allclose(engine_values, ore_values, rtol=rtol, atol=1e-6)


def test_base_value_is_ores():
    trades = [m.swap(), m.european(), m.bond()]
    shocks = monte_carlo_scenarios(FACTORS, m.covariance(), horizon_days=10, num_scenarios=4, seed=1)
    result = run_market_risk(MarketRiskRequest(trades, shocks))
    np.testing.assert_allclose(
        result.base_npv_per_trade, [_ore_npv(cfg, BASE) for cfg in trades], rtol=1e-6,
    )


# ---------------------------------------------------------------------------
# VaR / ES: ORE's statistics on ORE's own P&L
# ---------------------------------------------------------------------------
QUANTILES = (0.99, 0.975, 0.95)


def _assert_statistics_match(result, ore_pnl, quantiles, rtol):
    stats = _ore_statistics(ore_pnl.sum(axis=1))
    for q in quantiles:
        label = f"{q * 100:g}"
        assert result.risk[f"VaR_{label}"] == pytest.approx(stats.valueAtRisk(q), rel=rtol)
        assert result.risk[f"ES_{label}"] == pytest.approx(stats.expectedShortfall(q), rel=rtol)


@pytest.fixture(scope="module")
def monte_carlo_run():
    trades = [m.swap(), m.european(), m.bond()]
    shocks = monte_carlo_scenarios(FACTORS, m.covariance(), horizon_days=10, num_scenarios=512, seed=7)
    return trades, shocks, run_market_risk(MarketRiskRequest(trades, shocks, quantiles=QUANTILES))


@pytest.fixture(scope="module")
def historical_run():
    """A synthetic history -- a correlated random walk of both curves'
    pillars -- run through `historical_scenarios`: the overlapping 10-day
    moves become the scenarios, and ORE reprices each one."""
    rng = np.random.default_rng(2026)
    daily = rng.multivariate_normal(np.zeros(FACTORS.size), m.covariance(horizon_days=1), size=300)
    history = BASE[None, :] + np.cumsum(daily, axis=0) - np.sum(daily, axis=0)[None, :]
    shocks = historical_scenarios(FACTORS, history, horizon_days=10)
    trades = [m.swap(), m.bond()]
    return trades, shocks, run_market_risk(MarketRiskRequest(trades, shocks, quantiles=QUANTILES))


class TestMonteCarloVarEsMatchesOre:
    def test_pnl_vector_matches(self, monte_carlo_run):
        trades, shocks, result = monte_carlo_run
        np.testing.assert_allclose(np.asarray(result.pnl), _ore_pnl(trades, shocks.shifts), rtol=1e-6, atol=0.05)

    def test_var_and_es_match(self, monte_carlo_run):
        trades, shocks, result = monte_carlo_run
        _assert_statistics_match(result, _ore_pnl(trades, shocks.shifts), QUANTILES, rtol=1e-6)


class TestHistoricalVarEsMatchesOre:
    def test_every_window_is_a_scenario(self, historical_run):
        _, shocks, result = historical_run
        assert result.num_scenarios == 300 - 10
        assert result.source == "historical"

    def test_var_and_es_match(self, historical_run):
        trades, shocks, result = historical_run
        _assert_statistics_match(result, _ore_pnl(trades, shocks.shifts), QUANTILES, rtol=1e-10)
