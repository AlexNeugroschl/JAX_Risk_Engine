import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.aggregate_statistics.risk_statistics import (
    compute_risk_metrics,
    expected_shortfall,
    portfolio_pnl,
    value_at_risk,
)
from engine.instruments.interest_rate_swap import SwapConfig, price_swaps
from engine.scenarios import EVAL_DATE, SWAP_DEMO_MATURITIES

MATURITIES = np.array(SWAP_DEMO_MATURITIES)


def _ore_risk_stats(pnl: np.ndarray) -> "ORE.RiskStatistics":
    stats = ORE.RiskStatistics()
    for v in pnl:
        stats.add(float(v), 1.0)
    return stats


class TestValueAtRiskAgainstORE:
    def test_matches_ore_tie_free(self):
        pnl_np = np.arange(-999, 1, 1.0)  # 1000 tie-free values
        stats = _ore_risk_stats(pnl_np)
        pnl = jnp.asarray(pnl_np[:, None], dtype=jnp.float64)  # [Scenarios, TimeSteps=1]

        for p in [0.90, 0.95, 0.99]:
            var = float(value_at_risk(pnl, p)[0])
            np.testing.assert_allclose(var, stats.valueAtRisk(p), atol=1e-9)

    def test_clamps_to_zero_when_no_losses(self):
        pnl_np = np.arange(0.0, 100.0, 1.0)  # entirely non-negative
        stats = _ore_risk_stats(pnl_np)
        pnl = jnp.asarray(pnl_np[:, None], dtype=jnp.float64)

        for p in [0.90, 0.95, 0.99]:
            var = float(value_at_risk(pnl, p)[0])
            assert var == 0.0
            assert stats.valueAtRisk(p) == 0.0


class TestExpectedShortfallAgainstORE:
    def test_matches_ore_tie_free(self):
        pnl_np = np.arange(-999, 1, 1.0)
        stats = _ore_risk_stats(pnl_np)
        pnl = jnp.asarray(pnl_np[:, None], dtype=jnp.float64)

        for p in [0.90, 0.95, 0.99]:
            es = float(expected_shortfall(pnl, p)[0])
            np.testing.assert_allclose(es, stats.expectedShortfall(p), atol=1e-9)

    def test_matches_ore_with_ties_at_var_boundary(self):
        """Regression coverage: a positional slice of the sorted array
        (`sorted[0:idx]`) diverges from ORE's actual strict value-based
        filter (`pnl[pnl < -VaR]`) whenever the tail has ties at the VaR
        cutoff. This dataset is constructed so p=0.9/0.95 land exactly on
        a tied value, which is the case that would have caught a
        positional-slice bug."""
        pnl_np = np.array([-100.0] * 5 + [-50.0] * 10 + list(np.arange(-40, 60, 1.0)))
        stats = _ore_risk_stats(pnl_np)
        pnl = jnp.asarray(pnl_np[:, None], dtype=jnp.float64)

        for p in [0.90, 0.95]:
            var = float(value_at_risk(pnl, p)[0])
            es = float(expected_shortfall(pnl, p)[0])
            np.testing.assert_allclose(var, stats.valueAtRisk(p), atol=1e-9)
            np.testing.assert_allclose(es, stats.expectedShortfall(p), atol=1e-9)
            # the wrong (positional) formula would give 72.7 here, not 100.0
            assert es == pytest.approx(100.0)

    def test_empty_tail_matches_ore_raising(self):
        """When every observation worse-or-equal to VaR is exactly tied at
        the cutoff, the strict tail is empty. ORE raises RuntimeError in
        this situation; this module can't raise from traced JAX code, so it
        must return NaN instead -- verified here to be the same condition,
        not a silent wrong answer."""
        pnl_np = np.array([-100.0] * 5 + [-50.0] * 10 + list(np.arange(-40, 60, 1.0)))
        pnl = jnp.asarray(pnl_np[:, None], dtype=jnp.float64)

        es = float(expected_shortfall(pnl, 0.99)[0])
        assert jnp.isnan(es)

        stats = _ore_risk_stats(pnl_np)
        with pytest.raises(RuntimeError, match="no data below the target"):
            stats.expectedShortfall(0.99)


class TestRiskMetricsProperties:
    def test_es_at_least_var_when_tail_nonempty(self):
        rng = np.random.default_rng(0)
        pnl_np = rng.normal(loc=0.0, scale=100.0, size=(5000, 3))  # [Scenarios, TimeSteps]
        pnl = jnp.asarray(pnl_np, dtype=jnp.float64)

        for p in [0.95, 0.99]:
            var = value_at_risk(pnl, p)
            es = expected_shortfall(pnl, p)
            assert jnp.all(es >= var - 1e-9)

    def test_var_monotonic_in_percentile(self):
        rng = np.random.default_rng(1)
        pnl_np = rng.normal(loc=0.0, scale=100.0, size=(5000, 1))
        pnl = jnp.asarray(pnl_np, dtype=jnp.float64)

        var95 = float(value_at_risk(pnl, 0.95)[0])
        var99 = float(value_at_risk(pnl, 0.99)[0])
        assert var99 >= var95 >= 0.0

    def test_single_scenario_single_trade(self):
        npv_cube = jnp.asarray([[[100.0]]], dtype=jnp.float64)  # [1, 1, 1]
        pnl = portfolio_pnl(npv_cube, base_npv=90.0)
        assert pnl.shape == (1, 1)
        np.testing.assert_allclose(float(pnl[0, 0]), 10.0)

        var = value_at_risk(pnl, 0.99)
        assert var.shape == (1,)


class TestPortfolioPnlSumsTrades:
    def test_sums_across_trades_axis(self):
        npv_cube = jnp.asarray(
            [[[10.0, 20.0], [30.0, 40.0]], [[1.0, 2.0], [3.0, 4.0]]], dtype=jnp.float64
        )  # [Scenarios=2, TimeSteps=2, Trades=2]
        pnl = portfolio_pnl(npv_cube, base_npv=0.0)
        expected = jnp.asarray([[30.0, 70.0], [3.0, 7.0]], dtype=jnp.float64)
        np.testing.assert_allclose(np.asarray(pnl), np.asarray(expected))


class TestRobustAcrossInstrumentSources:
    """The module must make no instrument-specific assumptions -- only the
    [Scenarios, TimeSteps, Trades] shape contract. Verified against both a
    fabricated cube and the real swap pricer's output."""

    def test_synthetic_arbitrary_shaped_cube(self):
        rng = np.random.default_rng(2)
        npv_cube = jnp.asarray(rng.normal(1000.0, 50.0, size=(2000, 4, 5)), dtype=jnp.float64)
        metrics = compute_risk_metrics(npv_cube, base_npv=5000.0, percentiles=(0.95, 0.99))
        assert set(metrics.keys()) == {"VaR_95", "ES_95", "VaR_99", "ES_99"}
        for arr in metrics.values():
            assert arr.shape == (4,)

    def test_real_swap_pricer_cube(self, make_flat_yield_curves):
        # base: [1, 1, Maturities, 2] deterministic (zero-shock) curve cube.
        base = make_flat_yield_curves(disc_rate=0.030, fwd_rate=0.035)

        # Fabricate a tiny "Monte Carlo" cube by jittering the deterministic
        # curve across a handful of scenarios/steps -- exercises price_swaps'
        # real code path without needing a full market_simulations run.
        rng = np.random.default_rng(3)
        n_scenarios, n_steps = 200, 2
        jitter = 1.0 + rng.normal(0.0, 0.01, size=(n_scenarios, n_steps, len(MATURITIES), 2))
        yield_curves = jnp.asarray(np.asarray(base) * jitter, dtype=jnp.float64)

        cfg = SwapConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="2Y", evaluation_date=EVAL_DATE,
        )
        npv_cube = price_swaps(yield_curves, MATURITIES, [cfg])
        base_npv = float(price_swaps(base, MATURITIES, [cfg])[0, 0, 0])

        metrics = compute_risk_metrics(npv_cube, base_npv, percentiles=(0.95, 0.99))
        assert metrics["VaR_95"].shape == (n_steps,)
        assert jnp.all(metrics["VaR_99"] >= metrics["VaR_95"] - 1e-6)
