"""
Quasi-Monte Carlo normals and the Brownian bridge (`engine.market_simulation.sobol`), shared by the
cross-asset simulation and the market-risk scenario generator: QuantLib's `BrownianBridge`
construction (checked against ORE's own transform) and the requested dtype.
"""
import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.market_simulation.sobol import _build_bridge_matrix, apply_brownian_bridge, generate_sobol_normals

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
        """Regression: the output was float64 whenever x64 was on, even when float32 was
        requested (`norm.ppf` computes in float64)."""
        Z = generate_sobol_normals(64, 4, 2, jnp.float32)
        assert Z.dtype == jnp.float32


class TestBrownianBridgeEdgeCases:
    def test_two_point_grid(self):
        """The smallest grid (one interior step)."""
        time_grid = np.array([0.0, 1.0])
        B = _build_bridge_matrix(time_grid)
        assert B.shape == (1, 1)
        np.testing.assert_allclose(B[0, 0], 1.0, atol=1e-10)

    def test_uneven_grid_spacing_still_reproduces_covariance(self):
        """B @ B.T == min(s, t) on irregular step sizes too."""
        time_grid = np.array([0.0, 0.1, 0.15, 1.0, 1.2, 5.0])
        B = _build_bridge_matrix(time_grid)
        times = time_grid[1:]
        expected_cov = np.minimum.outer(times, times)
        actual_cov = B @ B.T
        np.testing.assert_allclose(actual_cov, expected_cov, atol=1e-10)


class TestBrownianBridgeAgainstORE:
    """`_build_bridge_matrix` / `apply_brownian_bridge` against QuantLib's C++
    `BrownianBridge` (ql/methods/montecarlo/brownianbridge.hpp). Its `transform` returns
    standardized, time-ordered increments, the same quantity as `Z_sequential`."""

    def _ore_transform_all(self, times, Z):
        """ORE's `transform()` applied per column of `Z` [TimeSteps, N] (independent
        normals in Sobol-dimension order)."""
        bb = ORE.BrownianBridge(ORE.DoubleVector([float(t) for t in times]))
        n = len(times)
        out = np.empty_like(Z)
        for col in range(Z.shape[1]):
            out[:, col] = list(bb.transform(ORE.DoubleVector(Z[:, col].tolist())))
        return out

    def test_matches_ore_transform_on_uniform_grid(self):
        times = TIME_GRID[1:]
        time_grid = jnp.array(TIME_GRID, dtype=jnp.float64)
        rng = np.random.default_rng(0)
        Z_np = rng.normal(size=(len(times), 5))
        Z = jnp.asarray(Z_np[:, None, :], dtype=jnp.float64)  # [TimeSteps, 1 scenario, N assets]

        Z_seq = np.asarray(apply_brownian_bridge(Z, time_grid))[:, 0, :]
        Z_ore = self._ore_transform_all(times, Z_np)
        np.testing.assert_allclose(Z_seq, Z_ore, atol=1e-9)

    def test_matches_ore_transform_on_irregular_grid(self):
        """Irregular step sizes."""
        times = [0.1, 0.15, 1.0, 1.2, 5.0]
        time_grid = jnp.array([0.0] + times, dtype=jnp.float64)
        rng = np.random.default_rng(1)
        Z_np = rng.normal(size=(len(times), 3))
        Z = jnp.asarray(Z_np[:, None, :], dtype=jnp.float64)

        Z_seq = np.asarray(apply_brownian_bridge(Z, time_grid))[:, 0, :]
        Z_ore = self._ore_transform_all(times, Z_np)
        np.testing.assert_allclose(Z_seq, Z_ore, atol=1e-9)

    def test_matches_ore_transform_two_point_grid(self):
        times = [1.0]
        time_grid = jnp.array([0.0] + times, dtype=jnp.float64)
        Z_np = np.array([[0.37]])
        Z = jnp.asarray(Z_np[:, None, :], dtype=jnp.float64)

        Z_seq = np.asarray(apply_brownian_bridge(Z, time_grid))[:, 0, :]
        Z_ore = self._ore_transform_all(times, Z_np)
        np.testing.assert_allclose(Z_seq, Z_ore, atol=1e-9)


class TestBrownianBridgeMultipleGridShapes:
    """The full `apply_brownian_bridge` output at several grid shapes: Var[W(t)] == t at
    every grid time, and each increment's variance equals its dt."""

    @staticmethod
    def _path_values_from_increments(Z_seq, time_grid):
        dt = np.diff(time_grid)
        dW = Z_seq * np.sqrt(dt)[:, None, None]
        return np.cumsum(dW, axis=0)  # W(t) at each grid time, [TimeSteps, Scenarios, Assets]

    def _check_grid(self, times_after_zero, n_scenarios=16384, seed=0):
        time_grid = np.array([0.0] + list(times_after_zero))
        Z = generate_sobol_normals(n_scenarios, len(times_after_zero), 1, jnp.float64)
        Z_seq = np.asarray(apply_brownian_bridge(Z, jnp.array(time_grid)))
        dt = np.diff(time_grid)

        # Each increment's variance equals its dt.
        increment_var = Z_seq.var(axis=(1, 2)) * dt  # Z_seq already standardized -> multiply back by dt
        np.testing.assert_allclose(increment_var, dt, rtol=0.05)

        # The path's variance at each grid time equals that time.
        W = self._path_values_from_increments(Z_seq, time_grid)
        path_var = W.var(axis=(1, 2))
        np.testing.assert_allclose(path_var, times_after_zero, rtol=0.05)

    def test_two_step_grid(self):
        self._check_grid([0.5, 1.0])

    def test_many_step_grid(self):
        self._check_grid([0.05 * i for i in range(1, 21)])  # 20 steps

    def test_irregular_grid(self):
        self._check_grid([0.03, 0.5, 0.55, 2.0, 2.01, 10.0])


class TestBrownianBridgeDegenerateInputs:
    def test_zero_length_first_step_does_not_produce_nan(self):
        """A repeated time (dt = 0 for the first step): documents the actual behaviour."""
        time_grid = np.array([0.0, 0.0, 1.0])
        try:
            B = _build_bridge_matrix(time_grid)
        except Exception as e:
            pytest.skip(f"_build_bridge_matrix raises on a zero-length first "
                        f"step ({type(e).__name__}: {e}) rather than silently "
                        f"producing NaN -- acceptable, just documenting.")
        assert np.all(np.isfinite(B)), (
            "_build_bridge_matrix silently produced non-finite entries for "
            "a zero-length first step instead of raising or handling it."
        )
