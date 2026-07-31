import jax.numpy as jnp
import numpy as np
import pytest

from engine.market_simulations import (
    ZeroCurveConfig,
    _build_bridge_matrix,
    apply_brownian_bridge,
    compute_hw_A_matrix,
    generate_paths,
    generate_sobol_normals,
)

from conftest import with_scenarios

TIME_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]


class TestBrownianBridge:
    def test_matrix_reproduces_bm_covariance(self):
        time_grid = np.array(TIME_GRID)
        B = _build_bridge_matrix(time_grid)
        times = time_grid[1:]
        expected_cov = np.minimum.outer(times, times)
        actual_cov = B @ B.T
        np.testing.assert_allclose(actual_cov, expected_cov, atol=1e-10)

    def test_bridge_is_not_identity(self):
        B = _build_bridge_matrix(np.array(TIME_GRID))
        assert not np.allclose(B, np.eye(B.shape[0]))

    def test_standardized_increments_have_unit_variance(self):
        time_grid = jnp.array(TIME_GRID)
        Z = generate_sobol_normals(8192, 4, 2, jnp.float64)
        Z_seq = apply_brownian_bridge(Z, time_grid)
        variances = jnp.var(Z_seq, axis=(1, 2))
        np.testing.assert_allclose(np.asarray(variances), 1.0, atol=0.02)
        means = jnp.mean(Z_seq, axis=(1, 2))
        np.testing.assert_allclose(np.asarray(means), 0.0, atol=0.02)


class TestGenerateSobolNormals:
    def test_honors_requested_dtype_regardless_of_global_x64_state(self):
        """Regression coverage: generate_sobol_normals used to silently
        return float64 when called directly (outside generate_paths) while
        the global x64 flag was on, even when float32 was explicitly
        requested -- jax.scipy.stats.norm.ppf ignores the input dtype
        internally. Fixed by an explicit cast at the end of the function."""
        Z = generate_sobol_normals(64, 4, 2, jnp.float32)
        assert Z.dtype == jnp.float32


class TestHullWhiteAMatrix:
    def test_reprices_flat_curve_at_t_zero(self):
        zero_curves = [ZeroCurveConfig(
            times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0],
            rates=[0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
        )]
        hw_a = np.array([0.1])
        hw_sigma = np.array([0.01])
        step_times = np.array([1e-8])
        maturities = np.array([1.0, 2.0, 5.0, 10.0])

        B = (1.0 - np.exp(-hw_a[None, None, :] *
             np.maximum(maturities[None, :, None] - step_times[:, None, None], 0.0))) / hw_a[None, None, :]
        A = compute_hw_A_matrix(zero_curves, hw_a, hw_sigma, step_times, maturities, B)

        r0 = 0.03
        discount_factors = A[0, :, 0] * np.exp(-B[0, :, 0] * r0)
        expected = np.exp(-0.03 * maturities)
        np.testing.assert_allclose(discount_factors, expected, atol=1e-6)

    def test_reprices_distinct_curves_per_rate_factor(self):
        """Regression coverage for the shared-curve bug: two rate factors
        with DIFFERENT flat curves (3% and 2%) must each reprice their OWN
        curve, not one shared curve -- matches ORE's Cross-Asset Model,
        where every currency's Hull-White process is calibrated against its
        own YieldTermStructureHandle (live-verified against the installed
        ORE package; see compute_hw_A_matrix's docstring)."""
        zero_curves = [
            ZeroCurveConfig(times=[0.0, 30.0], rates=[0.03, 0.03]),
            ZeroCurveConfig(times=[0.0, 30.0], rates=[0.02, 0.02]),
        ]
        hw_a = np.array([0.1, 0.12])
        hw_sigma = np.array([0.01, 0.008])
        step_times = np.array([1e-8])
        maturities = np.array([1.0, 5.0, 10.0])

        B = (1.0 - np.exp(-hw_a[None, None, :] *
             np.maximum(maturities[None, :, None] - step_times[:, None, None], 0.0))) / hw_a[None, None, :]
        A = compute_hw_A_matrix(zero_curves, hw_a, hw_sigma, step_times, maturities, B)

        factor0_df = A[0, :, 0] * np.exp(-B[0, :, 0] * 0.03)
        factor1_df = A[0, :, 1] * np.exp(-B[0, :, 1] * 0.02)
        np.testing.assert_allclose(factor0_df, np.exp(-0.03 * maturities), atol=1e-6)
        np.testing.assert_allclose(factor1_df, np.exp(-0.02 * maturities), atol=1e-6)
        # the two factors' discount curves must actually differ -- this is
        # exactly what a shared-curve bug would silently fail to produce
        assert not np.allclose(factor0_df, factor1_df)

    def test_rejects_mismatched_curve_count(self):
        """generate_paths must reject a rates config whose
        initial_zero_curves length doesn't match the number of rate
        factors, rather than silently reusing/misaligning curves."""
        from engine.market_simulations import EquityConfig, RatesConfig, SimulationConfig

        cfg = SimulationConfig(
            time_grid=[0.0, 1.0],
            scenarios=64,
            equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[1.0, 0.0]]),
            rates=RatesConfig(
                initial_rates=[0.03, 0.02],
                theta=[0.03, 0.02],
                mean_reversion=[0.1, 0.1],
                maturities=[1.0],
                initial_zero_curves=[ZeroCurveConfig(times=[0.0, 30.0], rates=[0.03, 0.03])],  # only 1, need 2
            ),
            joint_covariance=[[0.04, 0.0, 0.0], [0.0, 0.0001, 0.0], [0.0, 0.0, 0.0001]],
        )
        with pytest.raises(ValueError, match="one curve per"):
            generate_paths(cfg)


class TestGeneratePaths:
    @classmethod
    @pytest.fixture(scope="class")
    def result(cls, cross_asset_config):
        return generate_paths(with_scenarios(cross_asset_config, scenarios=2048))

    def test_output_shapes(self, result):
        assert result["equities"].shape == (2048, 4, 2)  # scenarios, steps, assets
        assert result["rates"].shape == (2048, 4, 2)
        assert result["numeraire"].shape == (2048, 4)
        assert result["yield_curves"].shape == (2048, 4, 4, 2)

    def test_discount_factors_in_unit_interval(self, result):
        yc = np.asarray(result["yield_curves"])
        assert np.all(yc > 0.0)
        assert np.all(yc <= 1.0001)  # small tolerance for HW convexity effects near t=0

    def test_discount_factors_decreasing_with_maturity(self, result):
        yc = np.asarray(result["yield_curves"])
        # scenario 0, first step, USD (index 0): should decrease Year1 -> Year10
        usd_curve = yc[0, 0, :, 0]
        assert np.all(np.diff(usd_curve) < 0)

    def test_deterministic_given_fixed_seed(self, cross_asset_config):
        cfg = with_scenarios(cross_asset_config, scenarios=2048)
        r1 = generate_paths(cfg)
        r2 = generate_paths(cfg)
        np.testing.assert_array_equal(np.asarray(r1["equities"]), np.asarray(r2["equities"]))
        np.testing.assert_array_equal(np.asarray(r1["rates"]), np.asarray(r2["rates"]))
