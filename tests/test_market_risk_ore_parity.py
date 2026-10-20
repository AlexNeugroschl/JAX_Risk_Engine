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
                (`calibration="None"`), the index forwarding on its own curve; or, the
                Bermudan and the American, both calibrated (`Bootstrap`), ORE's LGM built
                and calibrated on each scenario's curves as its historical P&L recalibrates
                the trades' model builders under every scenario (I-112)

Measured agreement per scenario, which the tolerances below are set from:
swap ~1e-14 and bond ~1e-16 relative (the same arithmetic); Bermudan ~2e-13
(the same LGM grid); calibrated Bermudan and American recalibrated per scenario 6e-12 and 7e-11
(ORE's Levenberg-Marquardt calibration converges to its tolerance, the engine's roots to
rounding; measured 2026-10-17); Bachelier European ~2e-14 (measured 1.6e-14, 2026-09-29);
Jamshidian European within QuantLib's Brent tolerance on its root (tests/test_jamshidian.py).

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
    ShockScenarios,
    historical_scenarios,
    monte_carlo_scenarios,
    run_market_risk,
)
from engine.instruments.american_swaption import AmericanSwaptionConfig
from tests.support.ore_lgm_oracle import OreLgmEngine, ore_lgm_swaption_npv
from tests.support.ore_parity_matrix import cells
from tests import market_risk_support as m

FACTORS = m.factors()
OIS = FACTORS.slice_of(FACTORS.index_of("discount:USD"))
IBOR = FACTORS.slice_of(FACTORS.index_of(f"index:{m.INDEX}"))
BASE = FACTORS.base_rates()
QUANTILES = (0.99, 0.975, 0.95)


def _declared(trades, pricing, source, quantiles=(0.99, 0.975)):
    """The matrix cells of a run of `trades` under `pricing` on scenarios of `source`, the
    quantiles `quantiles` (a stand-in request with the run's settings; the shifts are drawn in
    the test)."""
    scenarios = ShockScenarios(FACTORS, np.zeros((2, FACTORS.size)), horizon_days=10, source=source)
    return cells("market_risk", market_risk=MarketRiskRequest(list(trades), m.market(), scenarios, pricing,
                                                              quantiles=quantiles))


def _ore_option_npv(cfg, disc_rates, fwd_rates, pricing) -> float:
    """ORE's NPV of a Bermudan/American on the curves, its LGM as `pricing` configures it:
    fixed, or calibrated to `m.VOLS` on these curves."""
    engine = pricing.american if isinstance(cfg, AmericanSwaptionConfig) else pricing.bermudan
    if isinstance(cfg, AmericanSwaptionConfig):
        style, exercises = "American", [cfg.first_exercise_date, cfg.last_exercise_date]
    else:
        style, exercises = "Bermudan", list(cfg.exercise_dates)
    swap = build_vanilla_swap(
        notional=cfg.notional, fixed_rate=cfg.fixed_rate, payer=cfg.payer, effective_date=cfg.effective_date,
        maturity_date=cfg.maturity_date, index_tenor_months=cfg.index_tenor_months,
        floating_spread=cfg.floating_spread,
    )
    return ore_lgm_swaption_npv(
        evaluation_date=m.TODAY, curve_times=m.PILLAR_TIMES, curve_rates=list(disc_rates), swap=swap,
        notional=cfg.notional, fixed_rate=cfg.fixed_rate, payer=cfg.payer,
        floating_spread=cfg.floating_spread, index_tenor_months=cfg.index_tenor_months, style=style,
        exercise_dates=exercises, hw_a=engine.reversion, hw_sigma=engine.volatility,
        n_per_std=engine.n_per_std, std_devs=engine.std_devs,
        exercise_time_steps_per_year=engine.exercise_time_steps_per_year, index_curve_rates=list(fwd_rates),
        shift_horizon=engine.shift_horizon, swaption_vols=m.VOLS, calibration=OreLgmEngine.of(engine).calibration,
    ).npv


def _ore_npv(cfg, factor_vector, pricing=m.PRICING) -> float:
    """ORE's price of one trade at a full factor vector."""
    disc, fwd = factor_vector[OIS], factor_vector[IBOR]
    name = type(cfg).__name__
    if name == "SwapConfig":
        return m.ore_swap_npv(cfg, disc, fwd)
    if name == "SwaptionConfig":
        return m.ore_european_npv(cfg, disc, fwd, pricing.european)
    if name == "BondConfig":
        return m.ore_bond_npv(cfg, disc)
    return _ore_option_npv(cfg, disc, fwd, pricing)


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
def _revaluation(make, pricing, scenarios, rtol, name):
    return pytest.param(make, pricing, scenarios, rtol, id=name, marks=pytest.mark.ore_parity(
        _declared([make()], pricing, "monte-carlo"), tolerance=f"{rtol:g} relative, each scenario's value"))


@pytest.mark.parametrize("make, pricing, scenarios, rtol", [
    _revaluation(m.swap, m.PRICING, 32, 1e-12, "swap"),
    _revaluation(m.bond, m.PRICING, 32, 1e-12, "bond"),
    _revaluation(m.european, m.PRICING, 32, 1e-10, "european"),
    _revaluation(m.european, m.JAMSHIDIAN, 32, 5e-6, "european-jamshidian"),
    _revaluation(m.bermudan, m.PRICING, 6, 1e-10, "bermudan"),
    _revaluation(m.bermudan, m.CALIBRATED, 6, 5e-10, "bermudan-recalibrated"),
    _revaluation(m.american, m.CALIBRATED, 6, 5e-10, "american-recalibrated"),
])
def test_every_scenario_revalues_as_ore_does(make, pricing, scenarios, rtol):
    cfg = make()
    shocks = monte_carlo_scenarios(FACTORS, m.covariance(), horizon_days=10, num_scenarios=scenarios, seed=11)
    result = run_market_risk(MarketRiskRequest([cfg], m.market(), shocks, pricing))

    engine_values = result.base_npv_per_trade[0] + np.asarray(result.pnl[:, 0])
    ore_values = np.asarray([_ore_npv(cfg, BASE + row, pricing) for row in shocks.shifts])
    np.testing.assert_allclose(engine_values, ore_values, rtol=rtol, atol=1e-6)


@pytest.mark.ore_parity(_declared([m.swap(), m.european(), m.bond()], m.PRICING, "monte-carlo"),
                        tolerance="1e-10 relative")
def test_base_value_is_ores():
    trades = [m.swap(), m.european(), m.bond()]
    shocks = monte_carlo_scenarios(FACTORS, m.covariance(), horizon_days=10, num_scenarios=4, seed=1)
    result = run_market_risk(MarketRiskRequest(trades, m.market(), shocks))
    np.testing.assert_allclose(
        result.base_npv_per_trade, [_ore_npv(cfg, BASE) for cfg in trades], rtol=1e-10,
    )


@pytest.mark.parametrize("name", ["comparison_single", "comparison_portfolio"])
def test_comparison_snapshots_preserve_ore_scenario_prices(name):
    """The reporting snapshot's PV reconstruction, on its actual coarse-grid workload.

    Reuse the independent ORE oracle above; no new pricing implementation in the benchmark.
    The statistical estimators and their ORE references remain tested below.
    """
    from engine.precision import Precision
    from tests.support.precision_campaigns import COMPARISON_CAMPAIGNS

    campaign = COMPARISON_CAMPAIGNS[name]()
    scenarios = campaign.scenarios(128, 11)
    result = campaign.run(scenarios, Precision())
    snapshot = campaign.snapshot(result)
    for cfg in campaign.trades:
        ore = np.asarray([_ore_npv(cfg, BASE + row, campaign.pricing) for row in scenarios.shifts[:3]])
        np.testing.assert_allclose(snapshot[f"trades/{cfg.trade_id}/scenario_PV"][:3], ore, rtol=1e-10, atol=1e-6)
    np.testing.assert_allclose(snapshot["portfolio/scenario_PV"],
                               sum(snapshot[f"trades/{t.trade_id}/scenario_PV"] for t in campaign.trades),
                               rtol=1e-13, atol=1e-8)


# ---------------------------------------------------------------------------
# VaR / ES: ORE's statistics on ORE's own P&L
# ---------------------------------------------------------------------------


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
    @pytest.mark.ore_parity(_declared([m.swap(), m.european(), m.bond()], m.PRICING, "monte-carlo", QUANTILES),
                            tolerance="P&L 1e-6 relative; VaR, ES 1e-6 relative to ORE's RiskStatistics")
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

    @pytest.mark.ore_parity(_declared([m.swap(), m.bond()], m.PRICING, "historical", QUANTILES),
                            tolerance="VaR, ES 1e-10 relative to ORE's RiskStatistics on ORE's P&L")
    def test_var_and_es_match(self, historical_run):
        trades, shocks, result = historical_run
        _assert_statistics_match(result, _ore_pnl(trades, shocks.shifts), QUANTILES, rtol=1e-10)

