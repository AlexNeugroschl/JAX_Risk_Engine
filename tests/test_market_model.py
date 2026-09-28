import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.simulation.market_model import (
    EquityConfig,
    RatesConfig,
    SimulationConfig,
    ZeroCurveConfig,
    _build_bridge_matrix,
    _initial_log_discount,
    apply_brownian_bridge,
    compute_hw_A_matrix,
    generate_paths,
    generate_sobol_normals,
    nearest_psd,
    validate_joint_covariance,
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
        """Regression: the output was float64 whenever x64 was on, even when float32 was
        requested (`norm.ppf` computes in float64)."""
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
        """Two rate factors on different curves (3%, 2%) each reprice their own curve, not a
        shared one."""
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
        # The two factors' discount curves must differ (a shared curve would not).
        assert not np.allclose(factor0_df, factor1_df)

    def test_rejects_mismatched_curve_count(self):
        """A curve count different from the number of rate factors is rejected."""
        from engine.simulation.market_model import EquityConfig, RatesConfig, SimulationConfig

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

    def test_discount_factors_are_positive_and_plausible(self, result):
        """Discount factors are strictly positive but may exceed 1 (negative simulated rates
        are possible under Hull-White). A loose upper bound catches a blow-up."""
        yc = np.asarray(result["yield_curves"])
        assert np.all(yc > 0.0)
        assert np.all(np.isfinite(yc))
        assert np.all(yc <= 2.0)  # generous bound -- catches a broken formula, not a real rate move

    def test_discount_factors_decreasing_with_maturity(self, result):
        yc = np.asarray(result["yield_curves"])
        # Scenario 0, first step, factor 0: decreasing from 1Y to 10Y.
        usd_curve = yc[0, 0, :, 0]
        assert np.all(np.diff(usd_curve) < 0)

    def test_deterministic_given_fixed_seed(self, cross_asset_config):
        cfg = with_scenarios(cross_asset_config, scenarios=2048)
        r1 = generate_paths(cfg)
        r2 = generate_paths(cfg)
        np.testing.assert_array_equal(np.asarray(r1["equities"]), np.asarray(r2["equities"]))
        np.testing.assert_array_equal(np.asarray(r1["rates"]), np.asarray(r2["rates"]))


class TestGeneratePathsEdgeCases:
    """Degenerate scenario/step counts, and that the two precisions really produce
    different dtypes."""

    def test_single_scenario(self, cross_asset_config):
        cfg = with_scenarios(cross_asset_config, scenarios=1)
        result = generate_paths(cfg)
        assert result["equities"].shape == (1, 4, 2)
        assert result["rates"].shape == (1, 4, 2)
        assert bool(jnp.all(jnp.isfinite(result["equities"])))
        assert bool(jnp.all(jnp.isfinite(result["rates"])))

    def test_single_time_step(self, cross_asset_config):
        import dataclasses
        cfg = dataclasses.replace(cross_asset_config, time_grid=[0.0, 1.0], scenarios=512)
        result = generate_paths(cfg)
        assert result["equities"].shape == (512, 1, 2)
        assert result["rates"].shape == (512, 1, 2)
        assert result["yield_curves"].shape == (512, 1, 4, 2)

    def test_float32_precision_end_to_end(self, cross_asset_config):
        cfg = with_scenarios(cross_asset_config, scenarios=256)
        result = generate_paths(cfg, precision=32)
        assert result["equities"].dtype == jnp.float32
        assert result["rates"].dtype == jnp.float32
        assert result["yield_curves"].dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(result["equities"])))

    def test_float64_precision_end_to_end(self, cross_asset_config):
        cfg = with_scenarios(cross_asset_config, scenarios=256)
        result = generate_paths(cfg, precision=64)
        assert result["equities"].dtype == jnp.float64
        assert result["rates"].dtype == jnp.float64

    def test_sequential_precision_switches_produce_correct_dtype_each_time(self, cross_asset_config):
        """Alternating precision=64/32/64 calls each produce the requested dtype."""
        cfg = with_scenarios(cross_asset_config, scenarios=128)
        r64a = generate_paths(cfg, precision=64)
        r32 = generate_paths(cfg, precision=32)
        r64b = generate_paths(cfg, precision=64)
        assert r64a["equities"].dtype == jnp.float64
        assert r32["equities"].dtype == jnp.float32
        assert r64b["equities"].dtype == jnp.float64

    def test_precision_32_restores_the_global_x64_flag(self, cross_asset_config):
        """I-14: `generate_paths(precision=32)` restores the global x64 flag.

        It once left float64 disabled, so later float64 work silently produced float32.
        The alternating-precision test above passes either way (each call re-sets the flag),
        so this asserts on the ambient flag and a plain float64 array.
        """
        cfg = with_scenarios(cross_asset_config, scenarios=128)
        before = jax.config.jax_enable_x64
        assert before, "this test needs x64 enabled going in (see conftest)"

        result = generate_paths(cfg, precision=32)

        assert result["equities"].dtype == jnp.float32, (
            "the float32 run must still produce float32 -- restoring the flag "
            "must not silently promote the simulation's own output"
        )
        assert jax.config.jax_enable_x64 == before, (
            "generate_paths(precision=32) leaked jax_enable_x64=False to the "
            "rest of the process"
        )
        assert jnp.zeros(3, dtype=jnp.float64).dtype == jnp.float64, (
            "a float64 request after a float32 simulation silently truncated "
            "to float32 -- the I-14 failure mode"
        )

    def test_no_maturities_omits_yield_curves_key(self, cross_asset_config):
        import dataclasses
        cfg = dataclasses.replace(
            cross_asset_config,
            rates=dataclasses.replace(cross_asset_config.rates, maturities=None, initial_zero_curves=None),
            scenarios=64,
        )
        result = generate_paths(cfg)
        assert "yield_curves" not in result
        assert "equities" in result and "rates" in result and "numeraire" in result

    def test_near_zero_mean_reversion_does_not_produce_nan(self, cross_asset_config):
        """A very small non-zero mean reversion (in several denominators) gives no NaN/inf."""
        import dataclasses
        tiny_a = [1e-6, 1e-6]
        cfg = dataclasses.replace(
            cross_asset_config,
            rates=dataclasses.replace(cross_asset_config.rates, mean_reversion=tiny_a),
            scenarios=128,
        )
        result = generate_paths(cfg)
        assert bool(jnp.all(jnp.isfinite(result["rates"])))
        assert bool(jnp.all(jnp.isfinite(result["yield_curves"])))


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


class TestComputeHwAMatrixEdgeCases:
    def test_sloped_zero_curve_reprices_exactly(self):
        """A(t,T) reprices a sloped curve at t -> 0 (the forward rate and interpolation
        matter only when adjacent zero rates differ)."""
        zero_curves = [ZeroCurveConfig(
            times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0],
            rates=[0.020, 0.025, 0.028, 0.032, 0.035, 0.038],
        )]
        hw_a = np.array([0.1])
        hw_sigma = np.array([0.01])
        step_times = np.array([1e-8])
        maturities = np.array([1.0, 2.0, 5.0, 10.0, 30.0])

        B = (1.0 - np.exp(-hw_a[None, None, :] *
             np.maximum(maturities[None, :, None] - step_times[:, None, None], 0.0))) / hw_a[None, None, :]
        A = compute_hw_A_matrix(zero_curves, hw_a, hw_sigma, step_times, maturities, B)

        discount_factors = A[0, :, 0] * np.exp(-B[0, :, 0] * zero_curves[0].rates[0])
        # Each pillar's continuously compounded zero rate is recovered.
        expected = np.exp(-np.array(zero_curves[0].rates[1:]) * maturities)
        np.testing.assert_allclose(discount_factors, expected, atol=1e-4)

    def test_three_or_more_rate_factors(self):
        """Three or more factors, without cross-contamination between them."""
        zero_curves = [
            ZeroCurveConfig(times=[0.0, 30.0], rates=[0.03, 0.03]),
            ZeroCurveConfig(times=[0.0, 30.0], rates=[0.02, 0.02]),
            ZeroCurveConfig(times=[0.0, 30.0], rates=[0.05, 0.05]),
        ]
        hw_a = np.array([0.1, 0.12, 0.08])
        hw_sigma = np.array([0.01, 0.008, 0.012])
        step_times = np.array([1e-8])
        maturities = np.array([1.0, 5.0, 10.0])

        B = (1.0 - np.exp(-hw_a[None, None, :] *
             np.maximum(maturities[None, :, None] - step_times[:, None, None], 0.0))) / hw_a[None, None, :]
        A = compute_hw_A_matrix(zero_curves, hw_a, hw_sigma, step_times, maturities, B)

        for k, rate in enumerate([0.03, 0.02, 0.05]):
            df = A[0, :, k] * np.exp(-B[0, :, k] * rate)
            np.testing.assert_allclose(df, np.exp(-rate * maturities), atol=1e-6)


class TestHullWhiteMeanReversionTransition:
    """Regression: the short-rate step once computed `r*decay + theta + shock` instead of
    the exact Ornstein-Uhlenbeck `r*decay + theta*(1-decay) + shock`, so theta acted as a
    per-step drift and the rate diverged (3% to ~14.6% by t=2y at dt=0.5, a=0.03). Every
    config then had theta == initial_rates, the one case where the two coincide, so it went
    unnoticed until the simulated mean was checked against the closed-form OU mean.
    """

    def test_mean_matches_analytic_ou_transition_multi_step(self):
        """theta == initial_rates stays at its fixed point at every step."""
        a = 0.03
        r0 = theta = 0.03
        config = SimulationConfig(
            time_grid=[0.0, 0.5, 1.0, 1.5, 2.0],
            scenarios=8192,
            equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[0.0]]),
            rates=RatesConfig(initial_rates=[r0], theta=[theta], mean_reversion=[a]),
            joint_covariance=[[0.0400, 0.0000], [0.0000, 0.0001]],
        )
        result = generate_paths(config)
        r_t = np.asarray(result["rates"][:, :, 0])  # [Scenarios, TimeSteps]

        for step in range(r_t.shape[1]):
            np.testing.assert_allclose(r_t[:, step].mean(), theta, atol=0.001)

    def test_mean_matches_analytic_ou_transition_theta_above_r0(self):
        """theta above r0: the simulated mean follows the closed-form OU mean step by step
        toward theta."""
        a = 0.1
        r0, theta = 0.02, 0.05
        dt = 0.5
        config = SimulationConfig(
            time_grid=[0.0, dt, 2 * dt, 3 * dt, 4 * dt],
            scenarios=8192,
            equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[0.0]]),
            rates=RatesConfig(initial_rates=[r0], theta=[theta], mean_reversion=[a]),
            joint_covariance=[[0.0400, 0.0000], [0.0000, 0.0001]],
        )
        result = generate_paths(config)
        r_t = np.asarray(result["rates"][:, :, 0])

        decay = np.exp(-a * dt)
        expected_mean = r0
        for step in range(4):
            expected_mean = expected_mean * decay + theta * (1.0 - decay)
            np.testing.assert_allclose(r_t[:, step].mean(), expected_mean, atol=0.002)

    def test_mean_matches_analytic_ou_transition_theta_below_r0(self):
        """theta below r0: the mean reverts downward."""
        a = 0.08
        r0, theta = 0.06, 0.02
        dt = 0.25
        config = SimulationConfig(
            time_grid=[0.0, dt, 2 * dt, 3 * dt],
            scenarios=8192,
            equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[0.0]]),
            rates=RatesConfig(initial_rates=[r0], theta=[theta], mean_reversion=[a]),
            joint_covariance=[[0.0400, 0.0000], [0.0000, 0.0001]],
        )
        result = generate_paths(config)
        r_t = np.asarray(result["rates"][:, :, 0])

        decay = np.exp(-a * dt)
        expected_mean = r0
        for step in range(3):
            expected_mean = expected_mean * decay + theta * (1.0 - decay)
            np.testing.assert_allclose(r_t[:, step].mean(), expected_mean, atol=0.002)

    def test_variance_matches_analytic_ou_transition(self):
        """The transition variance (not part of that bug; checked against
        `ORE.HullWhiteProcess.variance`) is pinned too."""
        a, sigma = 0.05, 0.015
        dt = 0.5
        config = SimulationConfig(
            time_grid=[0.0, dt],
            scenarios=16384,
            equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[0.0]]),
            rates=RatesConfig(initial_rates=[0.03], theta=[0.03], mean_reversion=[a]),
            joint_covariance=[[0.0400, 0.0000], [0.0000, sigma ** 2]],
        )
        result = generate_paths(config)
        r_t = np.asarray(result["rates"][:, 0, 0])

        expected_variance = sigma ** 2 / (2 * a) * (1 - np.exp(-2 * a * dt))
        np.testing.assert_allclose(r_t.var(), expected_variance, rtol=0.05)


class TestVolatilityIsNotDoubleApplied:
    """Regression: the correlation Cholesky factor was built from the raw covariance, whose
    diagonal already carries each volatility, and the step formulas then multiplied by the
    volatility again, squaring it (a 20% equity vol gave a ~4% log-return std). The factor
    is now built from the correlation matrix, so volatility is applied once."""

    def test_equity_log_return_variance_matches_configured_vol(self):
        """A 20% configured vol gives a ~20% log-return std, not ~4%."""
        sig_eq = 0.20
        config = SimulationConfig(
            time_grid=[0.0, 1.0],
            scenarios=16384,
            equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[0.0]]),
            rates=RatesConfig(initial_rates=[0.03], theta=[0.03], mean_reversion=[0.03]),
            joint_covariance=[[sig_eq ** 2, 0.0], [0.0, 0.0001]],
        )
        result = generate_paths(config)
        S = np.asarray(result["equities"][:, 0, 0])
        log_returns = np.log(S / 100.0)
        np.testing.assert_allclose(log_returns.std(), sig_eq, rtol=0.02)

    def test_rate_variance_matches_configured_vol_not_its_square(self):
        sigma = 0.015
        a = 0.05
        dt = 0.5
        config = SimulationConfig(
            time_grid=[0.0, dt],
            scenarios=16384,
            equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[0.0]]),
            rates=RatesConfig(initial_rates=[0.03], theta=[0.03], mean_reversion=[a]),
            joint_covariance=[[0.04, 0.0], [0.0, sigma ** 2]],
        )
        result = generate_paths(config)
        r_t = np.asarray(result["rates"][:, 0, 0])
        expected_std = sigma * np.sqrt((1 - np.exp(-2 * a * dt)) / (2 * a))
        # With the bug the std would be ~sigma times smaller than expected.
        assert r_t.std() > expected_std / 10.0
        np.testing.assert_allclose(r_t.std(), expected_std, rtol=0.03)

    def test_correlation_between_equity_and_rate_is_preserved(self):
        """The configured correlation between factors is preserved, not only each marginal
        variance."""
        rho = 0.5
        sig_eq, sig_r = 0.20, 0.01
        cov = [[sig_eq ** 2, rho * sig_eq * sig_r], [rho * sig_eq * sig_r, sig_r ** 2]]
        config = SimulationConfig(
            time_grid=[0.0, 1.0],
            scenarios=65536,
            equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[0.0]]),
            rates=RatesConfig(initial_rates=[0.03], theta=[0.03], mean_reversion=[0.03]),
            joint_covariance=cov,
        )
        result = generate_paths(config)
        S = np.asarray(result["equities"][:, 0, 0])
        log_returns = np.log(S / 100.0)
        r_t = np.asarray(result["rates"][:, 0, 0])

        empirical_corr = np.corrcoef(log_returns, r_t)[0, 1]
        np.testing.assert_allclose(empirical_corr, rho, atol=0.02)
        np.testing.assert_allclose(log_returns.std(), sig_eq, rtol=0.02)

    def test_two_correlated_rate_factors_each_match_own_configured_vol(self):
        """Two rate factors with different vols and non-zero correlation each match their
        configured vol."""
        sig_a, sig_b, rho = 0.012, 0.008, -0.3
        # generate_paths needs at least one equity; a zero-drift placeholder isolates the
        # two rate factors.
        config = SimulationConfig(
            time_grid=[0.0, 1.0],
            scenarios=32768,
            equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[0.0, 0.0]]),
            rates=RatesConfig(initial_rates=[0.03, 0.02], theta=[0.03, 0.02], mean_reversion=[0.05, 0.04]),
            joint_covariance=[
                [0.0400, 0.0000, 0.0000],
                [0.0000, sig_a ** 2, rho * sig_a * sig_b],
                [0.0000, rho * sig_a * sig_b, sig_b ** 2],
            ],
        )
        result = generate_paths(config)
        r_a = np.asarray(result["rates"][:, 0, 0])
        r_b = np.asarray(result["rates"][:, 0, 1])

        a1, a2 = 0.05, 0.04
        dt = 1.0
        expected_std_a = sig_a * np.sqrt((1 - np.exp(-2 * a1 * dt)) / (2 * a1))
        expected_std_b = sig_b * np.sqrt((1 - np.exp(-2 * a2 * dt)) / (2 * a2))
        np.testing.assert_allclose(r_a.std(), expected_std_a, rtol=0.03)
        np.testing.assert_allclose(r_b.std(), expected_std_b, rtol=0.03)

        empirical_corr = np.corrcoef(r_a, r_b)[0, 1]
        np.testing.assert_allclose(empirical_corr, rho, atol=0.03)


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


class TestHullWhiteAgainstORE:
    """Hull-White closed forms (OU transition variance, A(t,T)/B(t,T) bond price) against
    ORE's `HullWhiteProcess` and `HullWhite`."""

    def test_ou_transition_variance_matches_ore_hullwhiteprocess(self):
        a, sigma, r0, dt = 0.05, 0.015, 0.03, 0.5
        dc = ORE.Actual365Fixed()
        eval_date = ORE.Date(30, 7, 2026)
        flat = ORE.YieldTermStructureHandle(ORE.FlatForward(eval_date, r0, dc))
        hwp = ORE.HullWhiteProcess(flat, a, sigma)

        ore_variance = hwp.variance(0.0, r0, dt)
        our_variance = sigma ** 2 / (2 * a) * (1 - np.exp(-2 * a * dt))
        np.testing.assert_allclose(our_variance, ore_variance, rtol=1e-10)

    def test_discount_bond_matches_ore_hullwhite_discountbond(self):
        """`compute_hw_A_matrix` / `reconstruct_yield_curves` bond prices equal
        `ORE.HullWhite.discountBond(t, T, r)`."""
        a, sigma, r0 = 0.1, 0.01, 0.03
        dc = ORE.Actual365Fixed()
        eval_date = ORE.Date(30, 7, 2026)
        flat = ORE.YieldTermStructureHandle(ORE.FlatForward(eval_date, r0, dc))
        hw = ORE.HullWhite(flat, a, sigma)

        zero_curves = [ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0],
                                        rates=[r0] * 6)]
        hw_a = np.array([a])
        hw_sigma = np.array([sigma])
        step_times = np.array([1e-8])
        maturities = np.array([1.0, 2.0, 5.0, 10.0])
        B = (1.0 - np.exp(-hw_a[None, None, :] *
             np.maximum(maturities[None, :, None] - step_times[:, None, None], 0.0))) / hw_a[None, None, :]
        A = compute_hw_A_matrix(zero_curves, hw_a, hw_sigma, step_times, maturities, B)
        our_df = A[0, :, 0] * np.exp(-B[0, :, 0] * r0)

        ore_df = np.array([hw.discountBond(step_times[0], T, r0) for T in maturities])
        np.testing.assert_allclose(our_df, ore_df, atol=1e-6)

    def test_discount_bond_matches_ore_hullwhite_discountbond_sloped_curve(self):
        """The same on a sloped curve."""
        a, sigma = 0.1, 0.01
        times = [0.0, 1.0, 2.0, 5.0, 10.0, 30.0]
        rates = [0.020, 0.025, 0.028, 0.032, 0.035, 0.038]
        dc = ORE.Actual365Fixed()
        eval_date = ORE.Date(30, 7, 2026)
        dates = [eval_date + int(round(t * 365)) for t in times]
        # ORE's ZeroCurve needs a pillar at the evaluation date.
        curve = ORE.YieldTermStructureHandle(
            ORE.ZeroCurve(ORE.DateVector(dates), ORE.DoubleVector(rates), dc)
        )
        hw = ORE.HullWhite(curve, a, sigma)

        zero_curves = [ZeroCurveConfig(times=times, rates=rates)]
        hw_a = np.array([a])
        hw_sigma = np.array([sigma])
        step_times = np.array([1e-8])
        maturities = np.array([1.0, 2.0, 5.0, 10.0])
        B = (1.0 - np.exp(-hw_a[None, None, :] *
             np.maximum(maturities[None, :, None] - step_times[:, None, None], 0.0))) / hw_a[None, None, :]
        A = compute_hw_A_matrix(zero_curves, hw_a, hw_sigma, step_times, maturities, B)
        our_df = A[0, :, 0] * np.exp(-B[0, :, 0] * rates[0])

        ore_df = np.array([hw.discountBond(step_times[0], T, rates[0]) for T in maturities])
        # Both curves are linear in zero rate. The ~1e-6 gap comes from ORE's instantaneous
        # forward near t=0, which QuantLib takes as a finite difference over [0, 1e-4] on this
        # sloped first segment.
        np.testing.assert_allclose(our_df, ore_df, rtol=1e-3)


class TestComputeHwAMatrixCurveShapesAndExtrapolation:
    def _reprices(self, zero_curves, maturities=np.array([1.0, 2.0, 5.0, 10.0])):
        hw_a = np.array([0.1])
        hw_sigma = np.array([0.01])
        step_times = np.array([1e-8])
        B = (1.0 - np.exp(-hw_a[None, None, :] *
             np.maximum(maturities[None, :, None] - step_times[:, None, None], 0.0))) / hw_a[None, None, :]
        A = compute_hw_A_matrix(zero_curves, hw_a, hw_sigma, step_times, maturities, B)
        r0 = zero_curves[0].rates[0]
        df = A[0, :, 0] * np.exp(-B[0, :, 0] * r0)
        return df

    def test_flat_curve(self):
        zc = [ZeroCurveConfig(times=[0.0, 1.0, 5.0, 30.0], rates=[0.03, 0.03, 0.03, 0.03])]
        df = self._reprices(zc)
        np.testing.assert_allclose(df, np.exp(-0.03 * np.array([1.0, 2.0, 5.0, 10.0])), atol=1e-6)

    def test_upward_sloping_curve(self):
        zc = [ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0],
                               rates=[0.015, 0.02, 0.025, 0.03, 0.035, 0.04])]
        df = self._reprices(zc)
        expected = np.exp(-np.interp([1.0, 2.0, 5.0, 10.0], zc[0].times, zc[0].rates) *
                           np.array([1.0, 2.0, 5.0, 10.0]))
        np.testing.assert_allclose(df, expected, atol=1e-4)

    def test_downward_sloping_inverted_curve(self):
        zc = [ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0],
                               rates=[0.05, 0.045, 0.04, 0.035, 0.03, 0.025])]
        df = self._reprices(zc)
        expected = np.exp(-np.interp([1.0, 2.0, 5.0, 10.0], zc[0].times, zc[0].rates) *
                           np.array([1.0, 2.0, 5.0, 10.0]))
        np.testing.assert_allclose(df, expected, atol=1e-4)

    def test_single_pillar_curve_is_treated_as_flat(self):
        zc = [ZeroCurveConfig(times=[0.0], rates=[0.03])]
        df = self._reprices(zc)
        np.testing.assert_allclose(df, np.exp(-0.03 * np.array([1.0, 2.0, 5.0, 10.0])), atol=1e-6)

    def test_maturity_beyond_last_pillar_flat_extrapolates(self):
        """Beyond the last pillar the zero rate is held flat through the whole A(t,T)
        pipeline, with no NaN or negative discount factors."""
        zc = [ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0], rates=[0.02, 0.025, 0.03, 0.035])]
        df = self._reprices(zc, maturities=np.array([5.0, 10.0, 20.0, 50.0]))
        # Beyond t=5 the zero rate is flat at 0.035.
        expected = np.exp(-0.035 * np.array([10.0, 20.0, 50.0]))
        np.testing.assert_allclose(df[1:], expected, atol=1e-4)
        assert np.all(df > 0.0) and np.all(np.isfinite(df))
        # Discount factors still decrease under flat extrapolation.
        assert np.all(np.diff(df) < 0)


class TestZeroVolatilityAndSingularMeanReversion:
    """a = 0 is a removable 0/0 in B(t,T), the OU transition variance and A's variance term;
    its limit is arithmetic Brownian motion (B -> T-t, variance -> dt, A's term ->
    sigma^2*t/2). A zero-variance factor makes the correlation normalization 0/0, which
    Cholesky would spread to every factor. Both are guarded with `jnp.where` and give the
    finite limit."""

    def test_zero_mean_reversion_matches_arithmetic_brownian_motion_limit(self):
        """a = 0 gives finite output with the ABM limit: E[r(t)] = r0 and
        Var[r(t)] = sigma^2 * t."""
        r0 = 0.03
        sigma = 0.01
        cfg = SimulationConfig(
            time_grid=[0.0, 0.5, 1.0],
            scenarios=20000,
            equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[0.0]]),
            rates=RatesConfig(
                initial_rates=[r0], theta=[0.05], mean_reversion=[0.0],
                maturities=[1.0, 2.0],
                initial_zero_curves=[ZeroCurveConfig(times=[0.0, 30.0], rates=[0.03, 0.03])],
            ),
            joint_covariance=[[0.04, 0.0], [0.0, sigma ** 2]],
        )
        result = generate_paths(cfg)
        rates = result["rates"][:, :, 0]
        assert bool(jnp.all(jnp.isfinite(rates)))
        assert bool(jnp.all(jnp.isfinite(result["yield_curves"])))
        for step, t in enumerate([0.5, 1.0]):
            mean_r = float(jnp.mean(rates[:, step]))
            var_r = float(jnp.var(rates[:, step]))
            assert mean_r == pytest.approx(r0, abs=0.01)
            assert var_r == pytest.approx(sigma ** 2 * t, rel=0.15)

    def test_all_zero_volatility_produces_deterministic_path(self):
        """sigma = 0 for every factor gives the same deterministic path on every scenario."""
        a = 0.1
        r0 = theta = 0.03
        dt = 0.5
        cfg = SimulationConfig(
            time_grid=[0.0, dt, 2 * dt],
            scenarios=256,
            equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[1.0]]),
            rates=RatesConfig(initial_rates=[r0], theta=[theta], mean_reversion=[a]),
            joint_covariance=[[0.0, 0.0], [0.0, 0.0]],
        )
        result = generate_paths(cfg)
        assert bool(jnp.all(jnp.isfinite(result["rates"])))
        assert bool(jnp.all(jnp.isfinite(result["equities"])))
        # theta == r0, the OU fixed point: every scenario and step is exactly r0.
        np.testing.assert_allclose(np.asarray(result["rates"]), r0, atol=1e-9)
        assert float(jnp.var(result["rates"])) == pytest.approx(0.0, abs=1e-12)

    def test_one_zero_volatility_factor_does_not_poison_others(self):
        """One zero-vol factor (the equity) does not affect an uncorrelated rate factor: the
        equity stays at 100.0 and the rate factor keeps its own distribution."""
        cfg = SimulationConfig(
            time_grid=[0.0, 0.5],
            scenarios=2000,
            equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[0.0]]),
            rates=RatesConfig(initial_rates=[0.03], theta=[0.03], mean_reversion=[0.1]),
            joint_covariance=[[0.0, 0.0], [0.0, 0.0001]],  # equity vol=0, rate vol=1%, uncorrelated
        )
        result = generate_paths(cfg)
        assert bool(jnp.all(jnp.isfinite(result["rates"])))
        assert bool(jnp.all(jnp.isfinite(result["equities"])))
        np.testing.assert_allclose(np.asarray(result["equities"]), 100.0, atol=1e-9)
        mean_r = float(jnp.mean(result["rates"]))
        std_r = float(jnp.std(result["rates"]))
        assert mean_r == pytest.approx(0.03, abs=0.005)
        assert std_r == pytest.approx(0.01 * np.sqrt(0.5), rel=0.15)

    def test_negative_mean_reversion_runs_and_is_finite(self):
        """Negative a (anti-mean-reverting) runs and stays finite."""
        cfg = SimulationConfig(
            time_grid=[0.0, 0.5, 1.0],
            scenarios=128,
            equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[0.0]]),
            rates=RatesConfig(initial_rates=[0.03], theta=[0.03], mean_reversion=[-0.05]),
            joint_covariance=[[0.04, 0.0], [0.0, 0.0001]],
        )
        result = generate_paths(cfg)
        assert bool(jnp.all(jnp.isfinite(result["rates"])))


class TestCholeskyOnDegenerateCorrelation:
    """Degenerate correlations. `generate_paths` validates `joint_covariance` before any
    JAX work, so an invalid matrix raises instead of producing all-NaN paths. The rho = 1
    test documents Cholesky's floating-point behaviour at the boundary directly."""

    def test_non_positive_semidefinite_covariance_now_raises_instead_of_silently_producing_nan(self):
        """An implied rho > 1 is rejected by `validate_joint_covariance` (see
        TestCovarianceValidation)."""
        cfg = SimulationConfig(
            time_grid=[0.0, 1.0],
            scenarios=32,
            equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[1.0]]),
            rates=RatesConfig(initial_rates=[0.03], theta=[0.03], mean_reversion=[0.1]),
                # 0.05 against sqrt(0.04*0.0001) ~= 0.002: implies rho >> 1.
            joint_covariance=[[0.04, 0.05], [0.05, 0.0001]],
        )
        with pytest.raises(ValueError, match="not positive semi-definite"):
            generate_paths(cfg)

    def test_boundary_rho_equals_one_is_numerically_singular_not_a_jax_defect(self):
        """rho = 1 is PSD in exact arithmetic but numerically singular (smallest eigenvalue
        ~1e-20): numpy's Cholesky raises and JAX's returns NaN. A property of Cholesky at the
        boundary, not of JAX."""
        sig1, sig2 = 0.2, 0.01
        cov = np.array([[sig1 ** 2, 1.0 * sig1 * sig2], [1.0 * sig1 * sig2, sig2 ** 2]])
        eigvals = np.linalg.eigvalsh(cov)
        # The smallest eigenvalue is floating-point noise, not exactly 0.
        np.testing.assert_allclose(eigvals.min(), 0.0, atol=1e-15)

        with pytest.raises(np.linalg.LinAlgError):
            np.linalg.cholesky(cov)

        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        L_jax = jnp.linalg.cholesky(jnp.asarray(cov))
        assert bool(jnp.any(jnp.isnan(L_jax))), (
            "jax.numpy.linalg.cholesky no longer silently returns NaN for "
            "this boundary-singular matrix -- if JAX's implementation "
            "changed to raise instead, generate_paths would newly need "
            "exception handling around its own jnp.linalg.cholesky call "
            "for this input."
        )


class TestGeneratePathsMismatchedArrayLengths:
    """Every "one entry per factor" length rule is checked up front with a clear
    ValueError, rather than left to broadcasting."""

    def test_theta_shorter_than_initial_rates_raises(self):
        """One theta for two rate factors raises (rather than reusing theta[0])."""
        cfg = SimulationConfig(
            time_grid=[0.0, 1.0],
            scenarios=32,
            equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[1.0, 0.0]]),
            rates=RatesConfig(initial_rates=[0.03, 0.02], theta=[0.03], mean_reversion=[0.1, 0.1]),
            joint_covariance=[[0.04, 0.0, 0.0], [0.0, 0.0001, 0.0], [0.0, 0.0, 0.0001]],
        )
        with pytest.raises(ValueError, match="theta"):
            generate_paths(cfg)

    def test_rate_mapping_row_count_mismatched_with_num_equities_raises(self):
        """Two equities but one rate_mapping row raises (rather than broadcasting)."""
        cfg = SimulationConfig(
            time_grid=[0.0, 1.0],
            scenarios=32,
            equities=EquityConfig(initial_prices=[100.0, 50.0], dividend_yields=[0.0, 0.0],
                                   rate_mapping=[[1.0]]),
            rates=RatesConfig(initial_rates=[0.03], theta=[0.03], mean_reversion=[0.1]),
            joint_covariance=[[0.04, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.0001]],
        )
        with pytest.raises(ValueError, match="rate_mapping"):
            generate_paths(cfg)

    def test_oversized_joint_covariance_raises_a_shape_error(self):
        """A joint_covariance of the wrong overall size raises."""
        cfg = SimulationConfig(
            time_grid=[0.0, 1.0],
            scenarios=32,
            equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[1.0]]),
            rates=RatesConfig(initial_rates=[0.03], theta=[0.03], mean_reversion=[0.1]),
            joint_covariance=[[0.04, 0.0, 0.0], [0.0, 0.0001, 0.0], [0.0, 0.0, 0.0001]],  # 3x3 for 1eq+1hw
        )
        with pytest.raises(Exception):
            generate_paths(cfg)


class TestGeneratePathsSingleFactorConfigurations:
    def test_single_equity_single_rate_factor(self, ):
        cfg = SimulationConfig(
            time_grid=[0.0, 0.5, 1.0],
            scenarios=64,
            equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[1.0]]),
            rates=RatesConfig(initial_rates=[0.03], theta=[0.03], mean_reversion=[0.1]),
            joint_covariance=[[0.04, 0.0], [0.0, 0.0001]],
        )
        result = generate_paths(cfg)
        assert result["equities"].shape == (64, 2, 1)
        assert result["rates"].shape == (64, 2, 1)
        assert bool(jnp.all(jnp.isfinite(result["equities"])))
        assert bool(jnp.all(jnp.isfinite(result["rates"])))

    def test_many_rate_and_equity_factors(self):
        """8 equities and 6 rate factors with a random valid covariance."""
        rng = np.random.default_rng(7)
        num_eq, num_hw = 8, 6
        n = num_eq + num_hw
        # A random PSD covariance: A @ A.T plus a small ridge.
        M = rng.normal(size=(n, n)) * 0.05
        cov = M @ M.T + np.eye(n) * 1e-6

        cfg = SimulationConfig(
            time_grid=[0.0, 0.5, 1.0],
            scenarios=64,
            equities=EquityConfig(
                initial_prices=[100.0 + 10 * i for i in range(num_eq)],
                dividend_yields=[0.0] * num_eq,
                rate_mapping=[[1.0] + [0.0] * (num_hw - 1) for _ in range(num_eq)],
            ),
            rates=RatesConfig(
                initial_rates=[0.02 + 0.001 * i for i in range(num_hw)],
                theta=[0.02 + 0.001 * i for i in range(num_hw)],
                mean_reversion=[0.05 + 0.01 * i for i in range(num_hw)],
            ),
            joint_covariance=cov.tolist(),
        )
        result = generate_paths(cfg)
        assert result["equities"].shape == (64, 2, num_eq)
        assert result["rates"].shape == (64, 2, num_hw)
        assert bool(jnp.all(jnp.isfinite(result["equities"])))
        assert bool(jnp.all(jnp.isfinite(result["rates"])))


class TestGeneratePathsNaNInfInputs:
    """NaN/Inf inputs propagate visibly rather than being clipped or zeroed."""

    def test_nan_initial_rate_propagates_as_nan_not_silently_dropped(self):
        cfg = SimulationConfig(
            time_grid=[0.0, 0.5, 1.0],
            scenarios=16,
            equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[1.0]]),
            rates=RatesConfig(initial_rates=[float("nan")], theta=[0.03], mean_reversion=[0.1]),
            joint_covariance=[[0.04, 0.0], [0.0, 0.0001]],
        )
        result = generate_paths(cfg)
        assert bool(jnp.all(jnp.isnan(result["rates"])))
        assert bool(jnp.all(jnp.isnan(result["equities"])))  # UIP drift depends on r_t

    def test_inf_initial_price_propagates_as_inf_or_nan(self):
        cfg = SimulationConfig(
            time_grid=[0.0, 0.5, 1.0],
            scenarios=16,
            equities=EquityConfig(initial_prices=[float("inf")], dividend_yields=[0.0], rate_mapping=[[1.0]]),
            rates=RatesConfig(initial_rates=[0.03], theta=[0.03], mean_reversion=[0.1]),
            joint_covariance=[[0.04, 0.0], [0.0, 0.0001]],
        )
        result = generate_paths(cfg)
        eq = np.asarray(result["equities"])
        assert np.all(np.isinf(eq) | np.isnan(eq))
        # Rates do not depend on the equity price.
        assert bool(jnp.all(jnp.isfinite(result["rates"])))


class TestGeneratePathsScenariosEqualsOne:
    def test_scenarios_equals_one(self, cross_asset_config):
        cfg = with_scenarios(cross_asset_config, scenarios=1)
        result = generate_paths(cfg)
        assert result["equities"].shape[0] == 1
        assert result["rates"].shape[0] == 1
        assert result["numeraire"].shape[0] == 1
        assert result["yield_curves"].shape[0] == 1
        assert bool(jnp.all(jnp.isfinite(result["equities"])))
        assert bool(jnp.all(jnp.isfinite(result["yield_curves"])))


class TestInitialLogDiscountExtrapolation:
    """`_initial_log_discount` extrapolates flat beyond the pillars."""

    def test_extrapolates_flat_below_first_pillar(self):
        zero_times = np.array([1.0, 2.0, 5.0])
        zero_rates = np.array([0.02, 0.03, 0.04])
        # t=0.5 is before the first pillar (1.0): the rate is clamped to rate[0].
        log_p = _initial_log_discount(zero_times, zero_rates, np.array([0.5]))
        np.testing.assert_allclose(log_p, -0.02 * 0.5, atol=1e-12)

    def test_extrapolates_flat_above_last_pillar(self):
        zero_times = np.array([0.0, 1.0, 2.0, 5.0])
        zero_rates = np.array([0.02, 0.025, 0.03, 0.035])
        log_p = _initial_log_discount(zero_times, zero_rates, np.array([10.0, 100.0]))
        expected = -0.035 * np.array([10.0, 100.0])
        np.testing.assert_allclose(log_p, expected, atol=1e-12)


class TestCovarianceValidation:
    """`validate_joint_covariance` / `nearest_psd`: an invalid covariance is rejected rather
    than producing NaN paths."""

    def test_implied_correlation_above_one_is_rejected(self):
        """Implied rho = 0.05/sqrt(0.02*0.02) = 2.5 > 1."""
        matrix = [[0.02, 0.05], [0.05, 0.02]]
        with pytest.raises(ValueError, match="not positive semi-definite"):
            validate_joint_covariance(matrix)

    def test_deliberately_negative_eigenvalue_is_rejected(self):
        """A symmetric matrix with one explicitly negative eigenvalue."""
        eigenvectors, _ = np.linalg.qr(np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]]))
        eigenvalues = np.array([2.0, 1.0, -0.5])
        matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        with pytest.raises(ValueError, match="not positive semi-definite"):
            validate_joint_covariance(matrix)
        eigs = np.linalg.eigvalsh(matrix)
        assert np.any(eigs < -1e-8)

    def test_asymmetric_matrix_is_rejected(self):
        matrix = [[0.02, 0.01], [0.03, 0.02]]
        with pytest.raises(ValueError, match="symmetric"):
            validate_joint_covariance(matrix)

    def test_non_square_matrix_is_rejected(self):
        matrix = [[0.02, 0.01, 0.0], [0.01, 0.02, 0.0]]
        with pytest.raises(ValueError, match="square"):
            validate_joint_covariance(matrix)

    def test_valid_diagonal_covariance_passes(self):
        validate_joint_covariance([[0.04, 0.0], [0.0, 0.0001]])

    def test_valid_correlated_covariance_passes(self):
        validate_joint_covariance([[0.04, 0.001], [0.001, 0.0001]])

    def test_generate_paths_raises_on_invalid_covariance_instead_of_producing_nan(self):
        """`generate_paths` raises before any JAX work."""
        cfg = SimulationConfig(
            time_grid=[0.0, 0.5, 1.0],
            scenarios=16,
            equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[1.0]]),
            rates=RatesConfig(initial_rates=[0.03], theta=[0.03], mean_reversion=[0.1]),
            joint_covariance=[[0.02, 0.05], [0.05, 0.02]],
        )
        with pytest.raises(ValueError, match="not positive semi-definite"):
            generate_paths(cfg)

    def test_nearest_psd_repaired_matrix_passes_validation(self):
        eigenvectors, _ = np.linalg.qr(np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]]))
        eigenvalues = np.array([2.0, 1.0, -0.5])
        bad_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        repaired = nearest_psd(bad_matrix)
        validate_joint_covariance(repaired)  # must not raise

    def test_nearest_psd_repaired_matrix_produces_finite_generate_paths_output(self):
        """`nearest_psd` clips to a small positive epsilon, so the repaired matrix is not
        rank-deficient and gives finite paths."""
        eigenvectors, _ = np.linalg.qr(np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]]))
        eigenvalues = np.array([0.04, 0.0001, -0.00002])
        bad_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        repaired = nearest_psd(bad_matrix)

        cfg = SimulationConfig(
            time_grid=[0.0, 0.5, 1.0],
            scenarios=64,
            equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[1.0, 0.0]]),
            rates=RatesConfig(initial_rates=[0.03, 0.03], theta=[0.03, 0.03], mean_reversion=[0.1, 0.1]),
            joint_covariance=repaired.tolist(),
        )
        result = generate_paths(cfg)
        assert bool(jnp.all(jnp.isfinite(result["rates"])))
        assert bool(jnp.all(jnp.isfinite(result["equities"])))

    def test_nearest_psd_never_called_automatically_by_generate_paths(self):
        """`nearest_psd` is opt-in: `generate_paths` still rejects a bad matrix."""
        cfg = SimulationConfig(
            time_grid=[0.0, 0.5, 1.0],
            scenarios=16,
            equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[1.0]]),
            rates=RatesConfig(initial_rates=[0.03], theta=[0.03], mean_reversion=[0.1]),
            joint_covariance=[[0.02, 0.05], [0.05, 0.02]],
        )
        with pytest.raises(ValueError):
            generate_paths(cfg)


class TestSobolSeed:
    """`SimulationConfig.seed` selects the Sobol scrambling; the default is 42."""

    def test_default_seed_is_42(self, cross_asset_config):
        default = generate_paths(with_scenarios(cross_asset_config, 256))["rates"]
        explicit = generate_paths(dataclasses.replace(with_scenarios(cross_asset_config, 256), seed=42))["rates"]
        np.testing.assert_array_equal(np.asarray(default), np.asarray(explicit))

    def test_different_seed_gives_different_paths(self, cross_asset_config):
        a = generate_paths(dataclasses.replace(with_scenarios(cross_asset_config, 256), seed=1))["rates"]
        b = generate_paths(dataclasses.replace(with_scenarios(cross_asset_config, 256), seed=2))["rates"]
        assert not np.allclose(np.asarray(a), np.asarray(b))


class TestForwardRatePrecision:
    """`hull_white.forward_rate` is exact. It was once a 1e-6 finite difference on ln P,
    which in float32 put forward rates off by up to ~2 percentage points."""

    TIMES = [0.0, 1.0, 2.0, 5.0, 10.0, 30.0]
    RATES = [0.03, 0.031, 0.032, 0.035, 0.037, 0.04]

    def _exact(self, t):
        # f = z + t*z': the slope of the segment to the right, 0 outside the pillars.
        times, rates = np.asarray(self.TIMES), np.asarray(self.RATES)
        slopes = np.diff(rates) / np.diff(times)
        idx = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(slopes) - 1)
        slope = np.where((t >= times[0]) & (t < times[-1]), slopes[idx], 0.0)
        return np.interp(t, times, rates) + t * slope

    @pytest.mark.parametrize("dtype, atol", [(jnp.float64, 1e-12), (jnp.float32, 1e-6)])
    def test_matches_exact_derivative(self, dtype, atol):
        from engine.models.hull_white import ZeroCurve, forward_rate

        t = np.array([0.0, 0.5, 1.0, 1.5, 3.0, 4.5, 7.0, 9.5, 30.0, 40.0])
        curve = ZeroCurve(jnp.asarray(self.TIMES, dtype=dtype), jnp.asarray(self.RATES, dtype=dtype))
        got = np.asarray(forward_rate(curve, jnp.asarray(t, dtype=dtype)), dtype=np.float64)
        np.testing.assert_allclose(got, self._exact(t), atol=atol, rtol=0)
