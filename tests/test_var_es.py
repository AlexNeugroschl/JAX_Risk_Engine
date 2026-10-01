import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.risk.var_es import (
    compute_risk_metrics,
    expected_shortfall,
    portfolio_pnl,
    value_at_risk,
)
from engine.portfolio import price_portfolio


def _ore_risk_stats(pnl: np.ndarray) -> "ORE.RiskStatistics":
    stats = ORE.RiskStatistics()
    for v in pnl:
        stats.add(float(v), 1.0)
    return stats


def _ore_risk_stats_bulk(pnl: np.ndarray) -> "ORE.RiskStatistics":
    """As `_ore_risk_stats`, via the bulk DoubleVector overload (faster; agrees with the
    scalar loop)."""
    stats = ORE.RiskStatistics()
    stats.add(ORE.DoubleVector(pnl.tolist()))
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
        """Regression: a positional slice `sorted[0:idx]` differs from ORE's value-based
        tail `pnl[pnl < -VaR]` when there are ties at the cutoff. p=0.9/0.95 land exactly on
        a tied value here."""
        pnl_np = np.array([-100.0] * 5 + [-50.0] * 10 + list(np.arange(-40, 60, 1.0)))
        stats = _ore_risk_stats(pnl_np)
        pnl = jnp.asarray(pnl_np[:, None], dtype=jnp.float64)

        for p in [0.90, 0.95]:
            var = float(value_at_risk(pnl, p)[0])
            es = float(expected_shortfall(pnl, p)[0])
            np.testing.assert_allclose(var, stats.valueAtRisk(p), atol=1e-9)
            np.testing.assert_allclose(es, stats.expectedShortfall(p), atol=1e-9)
            # The positional formula would give 72.7 here, not 100.0.
            assert es == pytest.approx(100.0)

    def test_empty_tail_matches_ore_raising(self):
        """All observations at or below VaR tied at the cutoff: the strict tail is empty. ORE
        raises; the engine returns NaN (traced code cannot raise) for the same condition."""
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
    """Only the [Scenarios, TimeSteps, Trades] shape matters: a synthetic cube and the real
    swap pricer's cube both work."""

    def test_synthetic_arbitrary_shaped_cube(self):
        rng = np.random.default_rng(2)
        npv_cube = jnp.asarray(rng.normal(1000.0, 50.0, size=(2000, 4, 5)), dtype=jnp.float64)
        metrics = compute_risk_metrics(npv_cube, base_npv=5000.0, percentiles=(0.95, 0.99))
        # The statistics are present (diagnostics keys are added alongside;
        # include_diagnostics=False below pins the bare set).
        assert {"VaR_95", "ES_95", "VaR_99", "ES_99"} <= set(metrics.keys())
        for arr in metrics.values():
            assert arr.shape == (4,)

    def test_synthetic_cube_without_diagnostics(self):
        """Without diagnostics, only VaR/ES keys."""
        rng = np.random.default_rng(2)
        npv_cube = jnp.asarray(rng.normal(1000.0, 50.0, size=(2000, 4, 5)), dtype=jnp.float64)
        metrics = compute_risk_metrics(
            npv_cube, base_npv=5000.0, percentiles=(0.95, 0.99), include_diagnostics=False,
        )
        assert set(metrics.keys()) == {"VaR_95", "ES_95", "VaR_99", "ES_99"}

    def test_a_portfolio_runs_cube(self, portfolio_request):
        """The cube `price_portfolio` produces (a swap simulated by the Hull-White model)."""
        result = price_portfolio(portfolio_request)
        metrics = compute_risk_metrics(result.npv_cube, result.base_npv, percentiles=(0.95, 0.99))
        assert metrics["VaR_95"].shape == (result.npv_cube.shape[1],)
        assert jnp.all(metrics["VaR_99"] >= metrics["VaR_95"] - 1e-6)


class TestValueAtRiskEdgeCases:
    def test_all_losses_matches_ore(self):
        """An all-loss sample matches ORE."""
        pnl_np = np.arange(-200, -100, 1.0)  # 100 values, all losses
        stats = _ore_risk_stats(pnl_np)
        pnl = jnp.asarray(pnl_np[:, None], dtype=jnp.float64)

        for p in [0.90, 0.95, 0.99]:
            var = float(value_at_risk(pnl, p)[0])
            es = float(expected_shortfall(pnl, p)[0])
            np.testing.assert_allclose(var, stats.valueAtRisk(p), atol=1e-9)
            np.testing.assert_allclose(es, stats.expectedShortfall(p), atol=1e-9)

    def test_single_scenario_matches_ore(self):
        """N=1: floor(N*(1-p)) is index 0 for any p (the clamp in `value_at_risk`)."""
        stats = ORE.RiskStatistics()
        stats.add(-50.0, 1.0)
        pnl = jnp.asarray([[-50.0]], dtype=jnp.float64)  # [Scenarios=1, TimeSteps=1]

        for p in [0.90, 0.95, 0.99]:
            var = float(value_at_risk(pnl, p)[0])
            np.testing.assert_allclose(var, stats.valueAtRisk(p), atol=1e-9)

    def test_single_scenario_expected_shortfall_is_nan(self):
        """N=1: the strict tail is always empty, so ES is NaN (ORE raises)."""
        stats = ORE.RiskStatistics()
        stats.add(-50.0, 1.0)
        with pytest.raises(RuntimeError, match="no data below the target"):
            stats.expectedShortfall(0.95)

        pnl = jnp.asarray([[-50.0]], dtype=jnp.float64)
        es = float(expected_shortfall(pnl, 0.95)[0])
        assert jnp.isnan(es)

    def test_boundary_percentile_0_9_matches_ore(self):
        """p=0.9 is the lower end of ORE's accepted range [0.9, 1.0) and matches."""
        pnl_np = np.arange(-9, 1, 1.0)  # N=10
        stats = _ore_risk_stats(pnl_np)
        pnl = jnp.asarray(pnl_np[:, None], dtype=jnp.float64)

        var = float(value_at_risk(pnl, 0.9)[0])
        np.testing.assert_allclose(var, stats.valueAtRisk(0.9), atol=1e-9)

    def test_ore_rejects_percentile_outside_0_9_to_1(self):
        """ORE raises outside [0.9, 1.0), the range this module is validated against; the
        module itself does not check it."""
        stats = ORE.RiskStatistics()
        stats.add(-50.0, 1.0)
        for p in [0.0, 0.5, 0.89, 1.0]:
            with pytest.raises(RuntimeError, match="out of range"):
                stats.valueAtRisk(p)

    def test_negative_base_npv(self):
        """A negative base NPV shifts the P&L correctly."""
        npv_cube = jnp.asarray([[[100.0]], [[50.0]], [[-20.0]]], dtype=jnp.float64)  # [S=3,T=1,Trades=1]
        pnl = portfolio_pnl(npv_cube, base_npv=-30.0)
        expected = jnp.asarray([[130.0], [80.0], [10.0]], dtype=jnp.float64)
        np.testing.assert_allclose(np.asarray(pnl), np.asarray(expected))

    def test_zero_variance_sample_all_identical_values(self):
        """Identical P&L in every scenario: VaR/ES collapse to that value's magnitude, as in
        ORE."""
        pnl_np = np.full(50, -75.0)
        stats = _ore_risk_stats(pnl_np)
        pnl = jnp.asarray(pnl_np[:, None], dtype=jnp.float64)

        var = float(value_at_risk(pnl, 0.95)[0])
        np.testing.assert_allclose(var, stats.valueAtRisk(0.95), atol=1e-9)
        np.testing.assert_allclose(var, 75.0)

        # The strict ES tail is empty here too (every value tied at VaR).
        es = float(expected_shortfall(pnl, 0.95)[0])
        assert jnp.isnan(es)
        with pytest.raises(RuntimeError, match="no data below the target"):
            stats.expectedShortfall(0.95)


class TestComputeRiskMetricsNaNPropagation:
    """`compute_risk_metrics` passes ES's NaN through to its output."""

    def test_nan_es_reaches_public_entry_point(self):
        # Every scenario tied at the same loss: ES's strict tail is empty.
        npv_cube = jnp.full((50, 1, 1), -75.0, dtype=jnp.float64)
        metrics = compute_risk_metrics(npv_cube, base_npv=0.0, percentiles=(0.95,))
        assert bool(jnp.isnan(metrics["ES_95"][0]))
        assert not bool(jnp.isnan(metrics["VaR_95"][0]))


class TestPercentileConventionSweepAgainstORE:
    """(num_scenarios, percentile) pairs where N*(1-p) is often non-integer, where the
    nearest-rank-below convention differs from `numpy.percentile`'s default interpolation;
    each checked against `ORE.RiskStatistics` on the same sample."""

    SAMPLE_SIZES = [2, 3, 7, 10, 13, 17, 31, 50, 97, 101, 250, 999, 1000]
    PERCENTILES = [0.9, 0.925, 0.95, 0.975, 0.99, 0.995, 0.999]

    @pytest.mark.parametrize("n", SAMPLE_SIZES)
    @pytest.mark.parametrize("p", PERCENTILES)
    def test_var_matches_ore_across_sizes_and_percentiles(self, n, p):
        rng = np.random.default_rng(1000 + n)
        pnl_np = rng.normal(loc=10.0, scale=250.0, size=n)
        stats = _ore_risk_stats_bulk(pnl_np)
        pnl = jnp.asarray(pnl_np[:, None], dtype=jnp.float64)

        var = float(value_at_risk(pnl, p)[0])
        np.testing.assert_allclose(var, stats.valueAtRisk(p), atol=1e-9)

        # Recompute ORE's index directly, to confirm the convention being exercised.
        idx = int(np.floor(n * (1.0 - p)))
        idx = min(max(idx, 0), n - 1)
        expected = max(-np.sort(pnl_np)[idx], 0.0)
        np.testing.assert_allclose(var, expected, atol=1e-9)

    @pytest.mark.parametrize("n", [10, 17, 50, 101, 250])
    @pytest.mark.parametrize("p", [0.9, 0.95, 0.99])
    def test_es_matches_ore_across_sizes_and_percentiles(self, n, p):
        rng = np.random.default_rng(2000 + n)
        pnl_np = rng.normal(loc=-5.0, scale=300.0, size=n)
        stats = _ore_risk_stats_bulk(pnl_np)
        pnl = jnp.asarray(pnl_np[:, None], dtype=jnp.float64)

        es_val = float(expected_shortfall(pnl, p)[0])
        try:
            ore_es = stats.expectedShortfall(p)
        except RuntimeError:
            # ORE raises on an empty strict tail; the module returns NaN.
            assert jnp.isnan(es_val)
            return
        np.testing.assert_allclose(es_val, ore_es, atol=1e-9)

    def test_var_diverges_from_numpy_default_percentile_at_chosen_point(self):
        """At a non-integer N*(1-p), the nearest-rank result differs from
        `numpy.percentile`'s default, so the sweep above discriminates between them."""
        n = 37
        p = 0.95  # N*(1-p) = 1.85, non-integer
        rng = np.random.default_rng(42)
        pnl_np = rng.normal(loc=0.0, scale=100.0, size=n)
        pnl = jnp.asarray(pnl_np[:, None], dtype=jnp.float64)

        var = float(value_at_risk(pnl, p)[0])
        stats = _ore_risk_stats_bulk(pnl_np)
        np.testing.assert_allclose(var, stats.valueAtRisk(p), atol=1e-9)

        # numpy's default (linear) percentile of the loss distribution, for comparison.
        numpy_quantile = np.percentile(pnl_np, (1.0 - p) * 100.0, method="linear")
        numpy_var = max(-numpy_quantile, 0.0)
        assert abs(var - numpy_var) > 1e-6, (
            "expected the nearest-rank-below and linear-interpolation "
            "conventions to disagree for this N/p combination"
        )


class TestTimeStepIndependence:
    """VaR/ES are computed per time step; changing one step's column affects no other."""

    def test_shuffling_one_timestep_does_not_affect_others(self):
        rng = np.random.default_rng(7)
        n_scenarios, n_steps = 500, 4
        pnl_np = rng.normal(0.0, 100.0, size=(n_scenarios, n_steps))
        pnl = jnp.asarray(pnl_np, dtype=jnp.float64)

        var_before = value_at_risk(pnl, 0.95)
        es_before = expected_shortfall(pnl, 0.95)

        shuffled = pnl_np.copy()
        rng.shuffle(shuffled[:, 1])  # permute only column 1's scenario order
        pnl_shuffled = jnp.asarray(shuffled, dtype=jnp.float64)

        var_after = value_at_risk(pnl_shuffled, 0.95)
        es_after = expected_shortfall(pnl_shuffled, 0.95)

        # Shuffling within a column leaves that column's VaR/ES unchanged (order statistics)
        # and every other column untouched.
        np.testing.assert_allclose(np.asarray(var_after), np.asarray(var_before), atol=1e-9)
        np.testing.assert_allclose(np.asarray(es_after), np.asarray(es_before), atol=1e-9)

    def test_changing_one_timestep_values_isolated_to_that_column(self):
        rng = np.random.default_rng(8)
        n_scenarios, n_steps = 300, 3
        pnl_np = rng.normal(0.0, 50.0, size=(n_scenarios, n_steps))
        pnl = jnp.asarray(pnl_np, dtype=jnp.float64)
        var_before = np.asarray(value_at_risk(pnl, 0.99))
        es_before = np.asarray(expected_shortfall(pnl, 0.99))

        # Replace step 2's column with much larger losses.
        mutated = pnl_np.copy()
        mutated[:, 2] = rng.normal(-5000.0, 2000.0, size=n_scenarios)
        pnl_mutated = jnp.asarray(mutated, dtype=jnp.float64)
        var_after = np.asarray(value_at_risk(pnl_mutated, 0.99))
        es_after = np.asarray(expected_shortfall(pnl_mutated, 0.99))

        # Columns 0 and 1 are bit-identical.
        np.testing.assert_allclose(var_after[:2], var_before[:2], atol=1e-9)
        np.testing.assert_allclose(es_after[:2], es_before[:2], atol=1e-9)
        # Column 2 changed.
        assert abs(var_after[2] - var_before[2]) > 1.0
        assert var_after[2] > var_before[2]  # much larger losses injected

    def test_shuffling_one_trade_isolated_from_others_via_pnl(self):
        """Shifting one trade's NPV moves the portfolio P&L by exactly that shift, scenario
        by scenario."""
        rng = np.random.default_rng(9)
        n_scenarios, n_steps, n_trades = 400, 2, 3
        npv_cube_np = rng.normal(1000.0, 20.0, size=(n_scenarios, n_steps, n_trades))
        npv_cube = jnp.asarray(npv_cube_np, dtype=jnp.float64)
        base_npv = 3000.0

        pnl_before = portfolio_pnl(npv_cube, base_npv)
        var_before = np.asarray(value_at_risk(pnl_before, 0.95))

        # Perturb trade 1 only, by less than the VaR (~55-60 for this seed), so the zero
        # clamp is not crossed.
        shift = 10.0
        mutated_np = npv_cube_np.copy()
        mutated_np[:, :, 1] = mutated_np[:, :, 1] + shift  # shift trade 1 up
        npv_cube_mutated = jnp.asarray(mutated_np, dtype=jnp.float64)
        pnl_after = portfolio_pnl(npv_cube_mutated, base_npv)

        # The portfolio P&L shift equals the injected shift in every scenario.
        np.testing.assert_allclose(
            np.asarray(pnl_after) - np.asarray(pnl_before),
            np.full((n_scenarios, n_steps), shift),
            atol=1e-9,
        )
        var_after = np.asarray(value_at_risk(pnl_after, 0.95))
        # A uniform shift, away from the clamp, lowers VaR by exactly the shift.
        assert np.all(var_before > shift + 1.0), "test setup: VaR too close to the clamp boundary"
        np.testing.assert_allclose(var_before - var_after, np.full(n_steps, shift), atol=1e-9)


class TestBaselineEdgeCases:
    """Base NPV of zero, far above or below the scenarios, and equal to every scenario."""

    def test_zero_base_npv(self):
        rng = np.random.default_rng(11)
        npv_cube_np = rng.normal(0.0, 100.0, size=(200, 1, 1))
        npv_cube = jnp.asarray(npv_cube_np, dtype=jnp.float64)
        pnl = portfolio_pnl(npv_cube, base_npv=0.0)
        # With base_npv=0 the P&L is the raw NPV.
        np.testing.assert_allclose(
            np.asarray(pnl), npv_cube_np.sum(axis=-1), atol=1e-9
        )
        var = value_at_risk(pnl, 0.95)
        assert var.shape == (1,)
        assert np.isfinite(float(var[0]))

    def test_base_npv_much_larger_than_scenario_spread_reflects_drift(self):
        """A base NPV far above the scenarios makes every scenario a large loss; VaR
        reflects that drift."""
        rng = np.random.default_rng(12)
        npv_cube_np = rng.normal(1_000_000.0, 10.0, size=(500, 1, 1))  # tight spread
        npv_cube = jnp.asarray(npv_cube_np, dtype=jnp.float64)
        base_npv = 50_000_000.0  # far larger than the scenario cluster

        pnl = portfolio_pnl(npv_cube, base_npv)
        stats = _ore_risk_stats_bulk(np.asarray(pnl).ravel())
        var = float(value_at_risk(pnl, 0.99)[0])

        np.testing.assert_allclose(var, stats.valueAtRisk(0.99), atol=1e-6)
        # VaR is of order base_npv minus the scenario cluster.
        assert var > 40_000_000.0

    def test_base_npv_much_smaller_reflects_gain_drift_var_zero(self):
        """A base NPV far below every scenario: all gains, so VaR clamps to 0, as in ORE."""
        rng = np.random.default_rng(13)
        npv_cube_np = rng.normal(1_000_000.0, 10.0, size=(500, 1, 1))
        npv_cube = jnp.asarray(npv_cube_np, dtype=jnp.float64)
        base_npv = 0.0

        pnl = portfolio_pnl(npv_cube, base_npv)
        stats = _ore_risk_stats_bulk(np.asarray(pnl).ravel())
        var = float(value_at_risk(pnl, 0.99)[0])

        np.testing.assert_allclose(var, stats.valueAtRisk(0.99), atol=1e-9)
        assert var == 0.0

    def test_all_scenarios_equal_base_npv_zero_pnl_variance(self):
        """Every scenario equals the base NPV: VaR is 0.0 and the strict ES tail (pnl < 0) is
        empty, so ES is NaN (ORE raises)."""
        n = 40
        npv_cube = jnp.full((n, 2, 1), 500.0, dtype=jnp.float64)  # 2 timesteps
        pnl = portfolio_pnl(npv_cube, base_npv=500.0)
        np.testing.assert_allclose(np.asarray(pnl), np.zeros((n, 2)), atol=1e-12)

        pnl_np = np.zeros(n)
        stats = _ore_risk_stats_bulk(pnl_np)

        var = value_at_risk(pnl, 0.95)
        np.testing.assert_allclose(np.asarray(var), np.zeros(2), atol=1e-12)
        np.testing.assert_allclose(float(var[0]), stats.valueAtRisk(0.95), atol=1e-9)

        es = expected_shortfall(pnl, 0.95)
        assert bool(jnp.isnan(es[0]))
        assert bool(jnp.isnan(es[1]))
        with pytest.raises(RuntimeError, match="no data below the target"):
            stats.expectedShortfall(0.95)


class TestExpectedShortfallNaNIsolation:
    """In a cube where only some steps are degenerate (empty ES tail), NaN appears in
    exactly those cells, not across steps or percentiles."""

    def test_nan_confined_to_degenerate_timesteps_only(self):
        rng = np.random.default_rng(21)
        n_scenarios, n_steps, n_trades = 300, 5, 4

        # Random NPVs per trade and step.
        npv_cube_np = rng.normal(1000.0, 80.0, size=(n_scenarios, n_steps, n_trades))

        # Steps 1 and 3 degenerate: every trade constant there, so the P&L is constant.
        degenerate_steps = [1, 3]
        # base_npv at the centre of the non-degenerate steps (n_trades * 1000), so they have
        # a real loss tail; at the degenerate value they would all be gains (another NaN
        # case) and defeat the check.
        base_npv = float(n_trades * 1000.0)
        for t in degenerate_steps:
            npv_cube_np[:, t, :] = base_npv / n_trades  # constant, exactly == base_npv/trade

        npv_cube = jnp.asarray(npv_cube_np, dtype=jnp.float64)

        for p in [0.9, 0.95, 0.99]:
            metrics_es = expected_shortfall(portfolio_pnl(npv_cube, base_npv), p)
            nan_mask = np.asarray(jnp.isnan(metrics_es))
            expected_nan = np.zeros(n_steps, dtype=bool)
            for t in degenerate_steps:
                expected_nan[t] = True
            np.testing.assert_array_equal(nan_mask, expected_nan)
            # Every non-degenerate step is finite.
            for t in range(n_steps):
                if t not in degenerate_steps:
                    assert np.isfinite(float(metrics_es[t])), f"step {t} unexpectedly NaN at p={p}"

    def test_nan_confined_via_compute_risk_metrics_multi_percentile(self):
        """The same isolation through `compute_risk_metrics`, across percentiles."""
        rng = np.random.default_rng(22)
        n_scenarios, n_steps, n_trades = 250, 6, 2
        npv_cube_np = rng.normal(2000.0, 150.0, size=(n_scenarios, n_steps, n_trades))
        degenerate_steps = [0, 4]
        # base_npv centred on the non-degenerate steps (see the test above).
        base_npv = float(n_trades * 2000.0)
        for t in degenerate_steps:
            npv_cube_np[:, t, :] = base_npv / n_trades
        npv_cube = jnp.asarray(npv_cube_np, dtype=jnp.float64)

        metrics = compute_risk_metrics(npv_cube, base_npv, percentiles=(0.9, 0.95, 0.99))

        for label in ["90", "95", "99"]:
            var_nan = np.asarray(jnp.isnan(metrics[f"VaR_{label}"]))
            # VaR is never NaN: in the degenerate case it is 0.0.
            assert not var_nan.any(), f"VaR_{label} has unexpected NaN"

            es_nan = np.asarray(jnp.isnan(metrics[f"ES_{label}"]))
            expected_nan = np.zeros(n_steps, dtype=bool)
            for t in degenerate_steps:
                expected_nan[t] = True
            np.testing.assert_array_equal(
                es_nan, expected_nan, err_msg=f"ES_{label} NaN mask mismatch"
            )


class TestNumericalAndShapeEdgeCases:
    """Single trade and step, 100,000 scenarios, all-loss and all-gain portfolios."""

    def test_single_trade_single_timestep_shape(self):
        rng = np.random.default_rng(31)
        npv_cube = jnp.asarray(
            rng.normal(100.0, 5.0, size=(64, 1, 1)), dtype=jnp.float64
        )
        metrics = compute_risk_metrics(npv_cube, base_npv=100.0, percentiles=(0.95,))
        assert metrics["VaR_95"].shape == (1,)
        assert metrics["ES_95"].shape == (1,)

    def test_large_scenario_count_100000_matches_ore_no_overflow(self):
        """100,000 scenarios: the sort and index arithmetic stay exact, matching ORE."""
        rng = np.random.default_rng(32)
        n = 100_000
        pnl_np = rng.normal(loc=0.0, scale=1000.0, size=n)
        pnl = jnp.asarray(pnl_np[:, None], dtype=jnp.float64)

        stats = _ore_risk_stats_bulk(pnl_np)
        for p in [0.95, 0.99, 0.999]:
            var = float(value_at_risk(pnl, p)[0])
            es = float(expected_shortfall(pnl, p)[0])
            np.testing.assert_allclose(var, stats.valueAtRisk(p), atol=1e-6)
            np.testing.assert_allclose(es, stats.expectedShortfall(p), atol=1e-6)
            assert np.isfinite(var) and np.isfinite(es)

    def test_all_negative_npvs_all_loss_portfolio(self):
        """An all-loss portfolio: VaR and ES positive, matching ORE."""
        rng = np.random.default_rng(33)
        npv_cube_np = -np.abs(rng.normal(500.0, 50.0, size=(2000, 1, 1)))
        npv_cube = jnp.asarray(npv_cube_np, dtype=jnp.float64)
        pnl = portfolio_pnl(npv_cube, base_npv=0.0)

        stats = _ore_risk_stats_bulk(np.asarray(pnl).ravel())
        for p in [0.95, 0.99]:
            var = float(value_at_risk(pnl, p)[0])
            es = float(expected_shortfall(pnl, p)[0])
            np.testing.assert_allclose(var, stats.valueAtRisk(p), atol=1e-9)
            np.testing.assert_allclose(es, stats.expectedShortfall(p), atol=1e-9)
            assert var > 0.0
            assert es >= var

    def test_all_positive_npvs_all_gain_portfolio_empty_tail(self):
        """An all-gain portfolio: VaR clamps to 0 and the strict tail is empty, so ES is NaN
        (ORE raises)."""
        rng = np.random.default_rng(34)
        npv_cube_np = np.abs(rng.normal(500.0, 50.0, size=(2000, 1, 1)))
        npv_cube = jnp.asarray(npv_cube_np, dtype=jnp.float64)
        pnl = portfolio_pnl(npv_cube, base_npv=0.0)

        stats = _ore_risk_stats_bulk(np.asarray(pnl).ravel())
        for p in [0.95, 0.99]:
            var = float(value_at_risk(pnl, p)[0])
            np.testing.assert_allclose(var, stats.valueAtRisk(p), atol=1e-9)
            assert var == 0.0

            es = float(expected_shortfall(pnl, p)[0])
            assert jnp.isnan(es)
            with pytest.raises(RuntimeError, match="no data below the target"):
                stats.expectedShortfall(p)
