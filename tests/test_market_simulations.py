import jax.numpy as jnp
import numpy as np
import pytest

from engine.market_simulations import (
    _build_bridge_matrix,
    apply_brownian_bridge,
    compute_hw_A_matrix,
    generate_paths,
    generate_sobol_normals,
)


TIME_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]


def _demo_payload():
    return {
        "time_grid": TIME_GRID,
        "scenarios": 2048,
        "equities": {
            "initial_prices": [150.0, 1.10],
            "dividend_yields": [0.01, 0.00],
            "rate_mapping": [
                [1.0, 0.0],
                [1.0, -1.0],
            ],
        },
        "rates": {
            "initial_rates": [0.03, 0.02],
            "theta": [0.03, 0.02],
            "mean_reversion": [0.1, 0.15],
            "maturities": [1.0, 2.0, 5.0, 10.0],
            "initial_zero_curve": {
                "times": [0.0, 1.0, 2.0, 5.0, 10.0, 30.0],
                "rates": [0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
            },
        },
        "joint_covariance": [
            [0.0400, 0.0000, 0.0010, 0.0005],
            [0.0000, 0.0100, 0.0002, -0.0001],
            [0.0010, 0.0002, 0.0001, 0.00008],
            [0.0005, -0.0001, 0.00008, 0.0002],
        ],
    }


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


class TestHullWhiteAMatrix:
    def test_reprices_flat_curve_at_t_zero(self):
        zero_times = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 30.0])
        zero_rates = np.full(6, 0.03)
        hw_a = np.array([0.1])
        hw_sigma = np.array([0.01])
        step_times = np.array([1e-8])
        maturities = np.array([1.0, 2.0, 5.0, 10.0])

        B = (1.0 - np.exp(-hw_a[None, None, :] *
             np.maximum(maturities[None, :, None] - step_times[:, None, None], 0.0))) / hw_a[None, None, :]
        A = compute_hw_A_matrix(zero_times, zero_rates, hw_a, hw_sigma, step_times, maturities, B)

        r0 = 0.03
        discount_factors = A[0, :, 0] * np.exp(-B[0, :, 0] * r0)
        expected = np.exp(-0.03 * maturities)
        np.testing.assert_allclose(discount_factors, expected, atol=1e-6)


class TestGeneratePaths:
    @classmethod
    @pytest.fixture(scope="class")
    def result(cls):
        return generate_paths(_demo_payload())

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

    def test_deterministic_given_fixed_seed(self):
        r1 = generate_paths(_demo_payload())
        r2 = generate_paths(_demo_payload())
        np.testing.assert_array_equal(np.asarray(r1["equities"]), np.asarray(r2["equities"]))
        np.testing.assert_array_equal(np.asarray(r1["rates"]), np.asarray(r2["rates"]))
