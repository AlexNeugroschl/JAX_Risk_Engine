import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.risk.statistics import (
    compute_risk_metrics,
    expected_shortfall,
    portfolio_pnl,
    value_at_risk,
)
from engine.instruments.swap import SwapConfig, price_swaps
from engine.scenarios import EVAL_DATE, SWAP_DEMO_MATURITIES

MATURITIES = np.array(SWAP_DEMO_MATURITIES)


def _ore_risk_stats(pnl: np.ndarray) -> "ORE.RiskStatistics":
    stats = ORE.RiskStatistics()
    for v in pnl:
        stats.add(float(v), 1.0)
    return stats


def _ore_risk_stats_bulk(pnl: np.ndarray) -> "ORE.RiskStatistics":
    """Same as _ore_risk_stats but uses the bulk DoubleVector overload --
    much faster for the large-N sweeps below, verified to agree with the
    scalar-loop overload above."""
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
        # real code path without needing a full simulation run.
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


class TestValueAtRiskEdgeCases:
    def test_all_losses_matches_ore(self):
        """Every prior test's sample has a mix of gains and losses -- an
        entirely loss-making sample (no gains at all) exercises a different
        branch of the ordering/index math and must still match ORE exactly."""
        pnl_np = np.arange(-200, -100, 1.0)  # 100 values, all losses
        stats = _ore_risk_stats(pnl_np)
        pnl = jnp.asarray(pnl_np[:, None], dtype=jnp.float64)

        for p in [0.90, 0.95, 0.99]:
            var = float(value_at_risk(pnl, p)[0])
            es = float(expected_shortfall(pnl, p)[0])
            np.testing.assert_allclose(var, stats.valueAtRisk(p), atol=1e-9)
            np.testing.assert_allclose(es, stats.expectedShortfall(p), atol=1e-9)

    def test_single_scenario_matches_ore(self):
        """N=1 is the smallest possible sample -- floor(N*(1-p)) collapses
        to index 0 regardless of p, which is a real edge case for the
        clamping logic in value_at_risk (idx = min(max(idx,0), N-1))."""
        stats = ORE.RiskStatistics()
        stats.add(-50.0, 1.0)
        pnl = jnp.asarray([[-50.0]], dtype=jnp.float64)  # [Scenarios=1, TimeSteps=1]

        for p in [0.90, 0.95, 0.99]:
            var = float(value_at_risk(pnl, p)[0])
            np.testing.assert_allclose(var, stats.valueAtRisk(p), atol=1e-9)

    def test_single_scenario_expected_shortfall_is_nan(self):
        """With N=1, the strict tail (pnl < -VaR) is always empty (the one
        observation IS the VaR boundary) -- matches ORE raising
        RuntimeError('no data below the target') for the same input."""
        stats = ORE.RiskStatistics()
        stats.add(-50.0, 1.0)
        with pytest.raises(RuntimeError, match="no data below the target"):
            stats.expectedShortfall(0.95)

        pnl = jnp.asarray([[-50.0]], dtype=jnp.float64)
        es = float(expected_shortfall(pnl, 0.95)[0])
        assert jnp.isnan(es)

    def test_boundary_percentile_0_9_matches_ore(self):
        """ORE's own RiskStatistics.valueAtRisk restricts percentile to
        [0.9, 1.0) (live-verified: values outside that range raise
        RuntimeError) -- 0.9 itself is the closed lower boundary and must
        match exactly, not just percentiles safely in the interior like
        0.95/0.99 every other test uses."""
        pnl_np = np.arange(-9, 1, 1.0)  # N=10
        stats = _ore_risk_stats(pnl_np)
        pnl = jnp.asarray(pnl_np[:, None], dtype=jnp.float64)

        var = float(value_at_risk(pnl, 0.9)[0])
        np.testing.assert_allclose(var, stats.valueAtRisk(0.9), atol=1e-9)

    def test_ore_rejects_percentile_outside_0_9_to_1(self):
        """Documents the range this module's formula is actually validated
        against: ORE.RiskStatistics itself raises outside [0.9, 1.0), so
        this module's own lack of validation at those percentiles is by
        design (matching ORE's supported range, not silently extending
        past what's been cross-checked) rather than an oversight."""
        stats = ORE.RiskStatistics()
        stats.add(-50.0, 1.0)
        for p in [0.0, 0.5, 0.89, 1.0]:
            with pytest.raises(RuntimeError, match="out of range"):
                stats.valueAtRisk(p)

    def test_negative_base_npv(self):
        """base_npv is a caller-supplied scalar with no sign constraint --
        a negative baseline (portfolio already underwater at t=0) must
        shift the P&L distribution correctly, not just the common
        positive-baseline case."""
        npv_cube = jnp.asarray([[[100.0]], [[50.0]], [[-20.0]]], dtype=jnp.float64)  # [S=3,T=1,Trades=1]
        pnl = portfolio_pnl(npv_cube, base_npv=-30.0)
        expected = jnp.asarray([[130.0], [80.0], [10.0]], dtype=jnp.float64)
        np.testing.assert_allclose(np.asarray(pnl), np.asarray(expected))

    def test_zero_variance_sample_all_identical_values(self):
        """A degenerate sample where every scenario has the exact same P&L
        (zero variance) -- VaR/ES should both collapse to that single
        value's magnitude, matching ORE, rather than dividing by zero
        variance anywhere (this module's formulas don't use variance
        directly, but this guards against any hidden assumption that the
        sample is non-degenerate)."""
        pnl_np = np.full(50, -75.0)
        stats = _ore_risk_stats(pnl_np)
        pnl = jnp.asarray(pnl_np[:, None], dtype=jnp.float64)

        var = float(value_at_risk(pnl, 0.95)[0])
        np.testing.assert_allclose(var, stats.valueAtRisk(0.95), atol=1e-9)
        np.testing.assert_allclose(var, 75.0)

        # ES's strict tail is empty here too (every value tied at VaR)
        es = float(expected_shortfall(pnl, 0.95)[0])
        assert jnp.isnan(es)
        with pytest.raises(RuntimeError, match="no data below the target"):
            stats.expectedShortfall(0.95)


class TestComputeRiskMetricsNaNPropagation:
    """compute_risk_metrics must propagate expected_shortfall's NaN through
    to its output dict rather than masking it (e.g. via a downstream
    comparison that silently treats NaN as 0) -- exercised at the
    dict-returning public entry point, not just the lower-level function."""

    def test_nan_es_reaches_public_entry_point(self):
        # every scenario tied at the same loss -> ES's strict tail is empty
        npv_cube = jnp.full((50, 1, 1), -75.0, dtype=jnp.float64)
        metrics = compute_risk_metrics(npv_cube, base_npv=0.0, percentiles=(0.95,))
        assert bool(jnp.isnan(metrics["ES_95"][0]))
        assert not bool(jnp.isnan(metrics["VaR_95"][0]))


class TestPercentileConventionSweepAgainstORE:
    """Systematic sweep of (num_scenarios, percentile) combinations chosen so
    that N*(1-p) is frequently non-integer -- exactly where the lower/
    nearest-rank-below convention this module implements diverges from
    numpy.percentile's default linear-interpolation convention. Each case is
    cross-checked directly against a real ORE.RiskStatistics instance fed the
    identical sample (not a hand-derived expected index), so this also
    catches any accidental drift in the module's own floor/clamp arithmetic,
    not just an numpy-vs-ORE mismatch."""

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

        # Confirm this is actually exercising the divergence: whenever
        # N*(1-p) is non-integer, numpy's default (linearly-interpolated)
        # percentile convention would generally disagree with ORE's
        # nearest-rank-below convention -- assert we understand *why* they
        # match by recomputing the ORE index directly, not just trusting
        # the module under test.
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
            # ORE raises when the strict tail is empty; module must return NaN.
            assert jnp.isnan(es_val)
            return
        np.testing.assert_allclose(es_val, ore_es, atol=1e-9)

    def test_var_diverges_from_numpy_default_percentile_at_chosen_point(self):
        """Explicitly demonstrates the documented divergence: pick N and p
        such that N*(1-p) is non-integer, then show ORE/this module's
        nearest-rank-below result differs from numpy.percentile's default
        (linear interpolation) result -- proving the test sweep above is
        actually discriminating between the two conventions rather than
        happening to agree everywhere."""
        n = 37
        p = 0.95  # N*(1-p) = 1.85, non-integer
        rng = np.random.default_rng(42)
        pnl_np = rng.normal(loc=0.0, scale=100.0, size=n)
        pnl = jnp.asarray(pnl_np[:, None], dtype=jnp.float64)

        var = float(value_at_risk(pnl, p)[0])
        stats = _ore_risk_stats_bulk(pnl_np)
        np.testing.assert_allclose(var, stats.valueAtRisk(p), atol=1e-9)

        # numpy's default ("linear") percentile of the *loss* distribution
        # at the (1-p) quantile, for comparison only.
        numpy_quantile = np.percentile(pnl_np, (1.0 - p) * 100.0, method="linear")
        numpy_var = max(-numpy_quantile, 0.0)
        assert abs(var - numpy_var) > 1e-6, (
            "expected the nearest-rank-below and linear-interpolation "
            "conventions to disagree for this N/p combination"
        )


class TestTimeStepIndependence:
    """VaR/ES are computed independently per time step -- perturbing one
    time step's scenario column must not leak into any other column's
    result. Exercised at [Scenarios, TimeSteps] shape (the direct input to
    value_at_risk/expected_shortfall) so a broadcasting bug in the
    axis=0 sort/mask logic would be caught precisely."""

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

        # Shuffling within a column doesn't change VaR/ES for THAT column
        # (order statistics are permutation-invariant within an axis), and
        # must not change any OTHER column at all.
        np.testing.assert_allclose(np.asarray(var_after), np.asarray(var_before), atol=1e-9)
        np.testing.assert_allclose(np.asarray(es_after), np.asarray(es_before), atol=1e-9)

    def test_changing_one_timestep_values_isolated_to_that_column(self):
        rng = np.random.default_rng(8)
        n_scenarios, n_steps = 300, 3
        pnl_np = rng.normal(0.0, 50.0, size=(n_scenarios, n_steps))
        pnl = jnp.asarray(pnl_np, dtype=jnp.float64)
        var_before = np.asarray(value_at_risk(pnl, 0.99))
        es_before = np.asarray(expected_shortfall(pnl, 0.99))

        # Replace time step 2's entire scenario column with a very different
        # distribution (much larger losses).
        mutated = pnl_np.copy()
        mutated[:, 2] = rng.normal(-5000.0, 2000.0, size=n_scenarios)
        pnl_mutated = jnp.asarray(mutated, dtype=jnp.float64)
        var_after = np.asarray(value_at_risk(pnl_mutated, 0.99))
        es_after = np.asarray(expected_shortfall(pnl_mutated, 0.99))

        # Columns 0 and 1 are untouched -> must be bit-identical.
        np.testing.assert_allclose(var_after[:2], var_before[:2], atol=1e-9)
        np.testing.assert_allclose(es_after[:2], es_before[:2], atol=1e-9)
        # Column 2 (the mutated one) must actually have changed.
        assert abs(var_after[2] - var_before[2]) > 1.0
        assert var_after[2] > var_before[2]  # much larger losses injected

    def test_shuffling_one_trade_isolated_from_others_via_pnl(self):
        """Trade-axis analogue: portfolio_pnl sums across trades, so
        perturbing a single trade's NPV column (holding the others fixed)
        must only move the portfolio P&L by exactly that trade's delta --
        i.e. VaR/ES computed on the resulting P&L reflects the
        per-trade contribution independently, not a cross-trade leak."""
        rng = np.random.default_rng(9)
        n_scenarios, n_steps, n_trades = 400, 2, 3
        npv_cube_np = rng.normal(1000.0, 20.0, size=(n_scenarios, n_steps, n_trades))
        npv_cube = jnp.asarray(npv_cube_np, dtype=jnp.float64)
        base_npv = 3000.0

        pnl_before = portfolio_pnl(npv_cube, base_npv)
        var_before = np.asarray(value_at_risk(pnl_before, 0.95))

        # Perturb only trade index 1 across all scenarios/timesteps, by an
        # amount small enough that VaR (~55-60 here, from a preliminary
        # check of this seed) does not cross the value_at_risk zero-clamp --
        # a large shift would push some/all scenarios into net gains and
        # break the simple "VaR shifts by exactly the same amount" identity
        # at the clamp boundary (a separate, already-covered edge case, not
        # what this test is targeting).
        shift = 10.0
        mutated_np = npv_cube_np.copy()
        mutated_np[:, :, 1] = mutated_np[:, :, 1] + shift  # shift trade 1 up
        npv_cube_mutated = jnp.asarray(mutated_np, dtype=jnp.float64)
        pnl_after = portfolio_pnl(npv_cube_mutated, base_npv)

        # Portfolio P&L shift must equal exactly the injected per-trade
        # shift, scenario-by-scenario -- proving trades combine additively
        # with no cross-trade leakage.
        np.testing.assert_allclose(
            np.asarray(pnl_after) - np.asarray(pnl_before),
            np.full((n_scenarios, n_steps), shift),
            atol=1e-9,
        )
        var_after = np.asarray(value_at_risk(pnl_after, 0.95))
        # A uniform shift to every scenario's P&L (small enough to stay
        # away from the zero-clamp boundary) reduces losses uniformly, so
        # VaR must decrease by exactly `shift` (order statistic commutes
        # with a constant shift, away from the clamp).
        assert np.all(var_before > shift + 1.0), "test setup: VaR too close to the clamp boundary"
        np.testing.assert_allclose(var_before - var_after, np.full(n_steps, shift), atol=1e-9)


class TestBaselineEdgeCases:
    """P&L baseline (base_npv) edge cases: zero, very large relative to the
    scenario spread, and the degenerate case where every scenario NPV
    exactly equals the baseline (zero P&L variance)."""

    def test_zero_base_npv(self):
        rng = np.random.default_rng(11)
        npv_cube_np = rng.normal(0.0, 100.0, size=(200, 1, 1))
        npv_cube = jnp.asarray(npv_cube_np, dtype=jnp.float64)
        pnl = portfolio_pnl(npv_cube, base_npv=0.0)
        # With base_npv=0, P&L is exactly the raw NPV (summed across the
        # single trade) -- no baseline subtraction should perturb values.
        np.testing.assert_allclose(
            np.asarray(pnl), npv_cube_np.sum(axis=-1), atol=1e-9
        )
        var = value_at_risk(pnl, 0.95)
        assert var.shape == (1,)
        assert np.isfinite(float(var[0]))

    def test_base_npv_much_larger_than_scenario_spread_reflects_drift(self):
        """base_npv far above the scenario NPV cluster means every scenario
        is a large loss relative to baseline -- VaR should be dominated by
        that drift (huge, and essentially equal to base_npv minus the best
        scenario NPV at the chosen percentile), not clamped to 0 or lost in
        floating-point noise."""
        rng = np.random.default_rng(12)
        npv_cube_np = rng.normal(1_000_000.0, 10.0, size=(500, 1, 1))  # tight spread
        npv_cube = jnp.asarray(npv_cube_np, dtype=jnp.float64)
        base_npv = 50_000_000.0  # far larger than the scenario cluster

        pnl = portfolio_pnl(npv_cube, base_npv)
        stats = _ore_risk_stats_bulk(np.asarray(pnl).ravel())
        var = float(value_at_risk(pnl, 0.99)[0])

        np.testing.assert_allclose(var, stats.valueAtRisk(0.99), atol=1e-6)
        # Every scenario is a huge loss relative to base_npv, so VaR must be
        # on the order of base_npv - npv_cluster, not near zero.
        assert var > 40_000_000.0

    def test_base_npv_much_smaller_reflects_gain_drift_var_zero(self):
        """The opposite drift direction: base_npv far below every scenario
        NPV means every scenario is a large GAIN -- VaR (a loss measure)
        must clamp to exactly 0, matching ORE, not go negative."""
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
        """Every scenario NPV exactly equals base_npv -> P&L is identically
        zero everywhere (zero variance). VaR must degrade gracefully to a
        real 0.0 (not NaN, not an exception) since 0 >= -0 satisfies the
        clamp; ES's strict tail (pnl < -VaR = pnl < 0) is empty since every
        value is exactly 0, so ES must be NaN, matching ORE raising for the
        identical input."""
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
    """A larger, multi-timestep multi-trade cube where only SOME time steps
    are degenerate (empty strict tail) -- verifies NaN appears in exactly
    those (percentile, timestep) cells and nowhere else, guarding against a
    broadcasting bug in expected_shortfall's `tail_mask = pnl < -var[None, :]`
    step leaking NaN across the TimeSteps axis (e.g. via a stray axis=None
    reduction) or across percentiles."""

    def test_nan_confined_to_degenerate_timesteps_only(self):
        rng = np.random.default_rng(21)
        n_scenarios, n_steps, n_trades = 300, 5, 4

        # Build NPVs per-trade, per-timestep, mostly random.
        npv_cube_np = rng.normal(1000.0, 80.0, size=(n_scenarios, n_steps, n_trades))

        # Force time steps 1 and 3 to be degenerate (every scenario has the
        # exact same portfolio P&L across trades) by making all trades
        # constant (and thus their sum constant) at those steps only.
        degenerate_steps = [1, 3]
        # base_npv sits at the center of the NON-degenerate steps' cluster
        # (n_trades * 1000.0) so those steps have a genuine two-sided P&L
        # spread with a real loss tail -- NOT at the degenerate value, which
        # would make every non-degenerate step an all-gain portfolio (its
        # own separate NaN-ES case, already covered by
        # TestNumericalAndShapeEdgeCases) and defeat the isolation check.
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
            # Every non-degenerate step must be a finite real number.
            for t in range(n_steps):
                if t not in degenerate_steps:
                    assert np.isfinite(float(metrics_es[t])), f"step {t} unexpectedly NaN at p={p}"

    def test_nan_confined_via_compute_risk_metrics_multi_percentile(self):
        """Same isolation check but through the public dict-returning entry
        point, across multiple percentiles simultaneously, to catch any bug
        where NaN from one percentile's ES computation bleeds into another
        percentile's key or into VaR."""
        rng = np.random.default_rng(22)
        n_scenarios, n_steps, n_trades = 250, 6, 2
        npv_cube_np = rng.normal(2000.0, 150.0, size=(n_scenarios, n_steps, n_trades))
        degenerate_steps = [0, 4]
        # base_npv centered on the non-degenerate steps' cluster (see
        # comment in test_nan_confined_to_degenerate_timesteps_only for why
        # this matters -- otherwise every step ends up degenerate/all-gain).
        base_npv = float(n_trades * 2000.0)
        for t in degenerate_steps:
            npv_cube_np[:, t, :] = base_npv / n_trades
        npv_cube = jnp.asarray(npv_cube_np, dtype=jnp.float64)

        metrics = compute_risk_metrics(npv_cube, base_npv, percentiles=(0.9, 0.95, 0.99))

        for label in ["90", "95", "99"]:
            var_nan = np.asarray(jnp.isnan(metrics[f"VaR_{label}"]))
            # VaR is never NaN anywhere -- it's a real order statistic even
            # in the degenerate case (it's 0.0, not undefined).
            assert not var_nan.any(), f"VaR_{label} has unexpected NaN"

            es_nan = np.asarray(jnp.isnan(metrics[f"ES_{label}"]))
            expected_nan = np.zeros(n_steps, dtype=bool)
            for t in degenerate_steps:
                expected_nan[t] = True
            np.testing.assert_array_equal(
                es_nan, expected_nan, err_msg=f"ES_{label} NaN mask mismatch"
            )


class TestNumericalAndShapeEdgeCases:
    """Shape/scale edge cases: single trade, single time step, very large
    scenario counts, all-loss and all-gain portfolios."""

    def test_single_trade_single_timestep_shape(self):
        rng = np.random.default_rng(31)
        npv_cube = jnp.asarray(
            rng.normal(100.0, 5.0, size=(64, 1, 1)), dtype=jnp.float64
        )
        metrics = compute_risk_metrics(npv_cube, base_npv=100.0, percentiles=(0.95,))
        assert metrics["VaR_95"].shape == (1,)
        assert metrics["ES_95"].shape == (1,)

    def test_large_scenario_count_100000_matches_ore_no_overflow(self):
        """100,000 scenarios is two orders of magnitude larger than any
        other test in this file -- confirms jnp.sort/floor-index arithmetic
        stays exact (no float32-style precision loss, no int overflow in
        the index math) at that scale, cross-checked against ORE directly
        rather than assumed."""
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
        """Every scenario NPV (and hence P&L, for base_npv=0) is negative --
        an all-loss portfolio. VaR/ES must both be strongly positive
        (real losses), matching ORE, not degrade because there are no
        gains to anchor against."""
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
        """Every scenario NPV (and hence P&L, for base_npv=0) is positive --
        an all-gain portfolio. VaR must clamp to exactly 0 (no losses to
        report), and ES's strict tail (pnl < -VaR = pnl < 0) is empty since
        every P&L value is positive, so ES must be NaN -- matching ORE
        raising for the identical input, not a silent 0 or negative value."""
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
