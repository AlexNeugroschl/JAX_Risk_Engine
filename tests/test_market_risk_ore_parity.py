"""
ORE parity for the market-risk path, end to end: every shocked scenario is
repriced independently in ORE, and ORE's own `RiskStatistics` computes VaR
and ES from ORE's P&L vector.

    engine: scenarios -> engine.risk.market.run_market_risk -> VaR/ES
    ORE:    same shifted pillar rates -> ORE.ZeroCurve -> ORE instruments and
            engines -> P&L vector -> ORE.RiskStatistics -> VaR/ES

ORE instruments and engines:

    swap        MakeVanillaSwap + DiscountingSwapEngine, separate forwarding
                and discounting curves
    European    Swaption + BachelierSwaptionEngine on ORE's SwaptionVolatilityMatrix
                (ORE's default European engine, plan 6.4); the Jamshidian engine
                (`PricingConfig.european = "Jamshidian"`) against
                JamshidianSwaptionEngine on ORE.HullWhite
    bond        FixedRateBond (ACT/ACT ISMA, no settlement lag) + DiscountingBondEngine
    Bermudan    NumericLgmMultiLegOptionEngine through an in-process OREApp
                (tests.support.ore_lgm_oracle), both at the same fixed LGM
                (`calibration="None"`), the index forwarding on its own curve

Measured agreement per scenario, which the tolerances below are set from:
swap ~1e-14 and bond ~1e-16 relative (the same arithmetic); Bermudan ~2e-13
(the same LGM grid); Bachelier European ~2e-14 (measured 1.6e-14, 2026-09-29); Jamshidian
European within QuantLib's Brent tolerance on its root (tests/test_jamshidian.py).

ORE's `ParametricVarCalculator` and `ExposureCalculator` have no constructor
in the Python bindings, so VaR/ES parity is established this way -- full
revaluation in ORE plus ORE's statistics -- rather than by calling an ORE VaR
analytic.
"""
import numpy as np
import ORE
import pytest

from engine.instruments.schedules import build_vanilla_swap
from engine.risk.market import (
    MarketRiskRequest,
    historical_scenarios,
    monte_carlo_scenarios,
    run_market_risk,
)
from tests.support.ore_lgm_oracle import ore_lgm_swaption_npv
from tests import market_risk_support as m

FACTORS = m.factors()
OIS = FACTORS.slice_of(FACTORS.index_of("discount:USD"))
IBOR = FACTORS.slice_of(FACTORS.index_of(f"index:{m.INDEX}"))
BASE = FACTORS.base_rates()


def _ore_bermudan_npv(cfg, disc_rates, fwd_rates) -> float:
    swap = build_vanilla_swap(
        notional=cfg.notional, fixed_rate=cfg.fixed_rate, payer=cfg.payer, effective_date=cfg.effective_date,
        maturity_date=cfg.maturity_date, index_tenor_months=cfg.index_tenor_months,
        floating_spread=cfg.floating_spread,
    )
    return ore_lgm_swaption_npv(
        evaluation_date=m.TODAY, curve_times=m.PILLAR_TIMES, curve_rates=list(disc_rates), swap=swap,
        notional=cfg.notional, fixed_rate=cfg.fixed_rate, payer=cfg.payer,
        floating_spread=cfg.floating_spread, index_tenor_months=cfg.index_tenor_months, style="Bermudan",
        exercise_dates=list(cfg.exercise_dates), hw_a=m.ENGINE.reversion, hw_sigma=m.ENGINE.volatility,
        n_per_std=m.ENGINE.n_per_std, std_devs=m.ENGINE.std_devs,
        exercise_time_steps_per_year=m.ENGINE.exercise_time_steps_per_year, index_curve_rates=list(fwd_rates),
    ).npv


def _ore_npv(cfg, factor_vector, european: str = "Bachelier") -> float:
    """ORE's price of one trade at a full factor vector."""
    disc, fwd = factor_vector[OIS], factor_vector[IBOR]
    name = type(cfg).__name__
    if name == "SwapConfig":
        return m.ore_swap_npv(cfg, disc, fwd)
    if name == "SwaptionConfig":
        return m.ore_european_npv(cfg, disc, fwd, european)
    if name == "BondConfig":
        return m.ore_bond_npv(cfg, disc)
    return _ore_bermudan_npv(cfg, disc, fwd)


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
@pytest.mark.parametrize("make, pricing, scenarios, rtol", [
    (m.swap, m.PRICING, 32, 1e-12),
    (m.bond, m.PRICING, 32, 1e-12),
    (m.european, m.PRICING, 32, 1e-10),
    (m.european, m.JAMSHIDIAN, 32, 5e-6),
    (m.bermudan, m.PRICING, 6, 1e-10),
], ids=["swap", "bond", "european", "european-jamshidian", "bermudan"])
def test_every_scenario_revalues_as_ore_does(make, pricing, scenarios, rtol):
    cfg = make()
    shocks = monte_carlo_scenarios(FACTORS, m.covariance(), horizon_days=10, num_scenarios=scenarios, seed=11)
    result = run_market_risk(MarketRiskRequest([cfg], m.market(), shocks, pricing))

    engine_values = result.base_npv_per_trade[0] + np.asarray(result.pnl[:, 0])
    ore_values = np.asarray([_ore_npv(cfg, BASE + row, pricing.european) for row in shocks.shifts])
    np.testing.assert_allclose(engine_values, ore_values, rtol=rtol, atol=1e-6)


def test_base_value_is_ores():
    trades = [m.swap(), m.european(), m.bond()]
    shocks = monte_carlo_scenarios(FACTORS, m.covariance(), horizon_days=10, num_scenarios=4, seed=1)
    result = run_market_risk(MarketRiskRequest(trades, m.market(), shocks))
    np.testing.assert_allclose(
        result.base_npv_per_trade, [_ore_npv(cfg, BASE) for cfg in trades], rtol=1e-10,
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
    return trades, shocks, run_market_risk(MarketRiskRequest(trades, m.market(), shocks, quantiles=QUANTILES))


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
    return trades, shocks, run_market_risk(MarketRiskRequest(trades, m.market(), shocks, quantiles=QUANTILES))


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
