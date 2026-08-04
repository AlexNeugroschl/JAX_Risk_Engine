import dataclasses

import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.simulation import (
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
        from engine.simulation import EquityConfig, RatesConfig, SimulationConfig

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
        """Discount factors must always be strictly positive (P(t,T) =
        A(t,T)*exp(-B(t,T)*r) is an exponential -- never zero or negative
        for any finite r), but are NOT bounded above by 1: a simulated
        short rate that has gone negative (a real, expected outcome under
        correctly-scaled HW1F volatility -- see
        TestHullWhiteMeanReversionTransition's docstring on the
        double-volatility bug this fixed) makes P(t,T) > 1, exactly as
        ORE's own negative-rate-capable HW1F model allows. A loose upper
        bound still guards against a genuinely broken (unboundedly large)
        discount factor from a NaN/inf-producing formula error."""
        yc = np.asarray(result["yield_curves"])
        assert np.all(yc > 0.0)
        assert np.all(np.isfinite(yc))
        assert np.all(yc <= 2.0)  # generous bound -- catches a broken formula, not a real rate move

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


class TestGeneratePathsEdgeCases:
    """Assumptions that TestGeneratePaths' happy-path fixture doesn't
    exercise: degenerate scenario/step counts, float32 end-to-end, and that
    the two supported precisions actually produce different-dtype output
    rather than both silently running in float64."""

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
        """The global jax_enable_x64 flag is toggled per-call inside
        generate_paths -- confirms alternating precision=64/32/64 calls in
        the same process each produce correctly-typed output, not whatever
        the previous call happened to leave the global flag set to."""
        cfg = with_scenarios(cross_asset_config, scenarios=128)
        r64a = generate_paths(cfg, precision=64)
        r32 = generate_paths(cfg, precision=32)
        r64b = generate_paths(cfg, precision=64)
        assert r64a["equities"].dtype == jnp.float64
        assert r32["equities"].dtype == jnp.float32
        assert r64b["equities"].dtype == jnp.float64

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
        """hw_a appears in several denominators (B(t,T), the HW1F transition
        variance, compute_hw_A_matrix's variance term) -- a very small but
        nonzero mean_reversion must not blow up into NaN/inf."""
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
        """The smallest meaningful grid (one interior step) -- the
        recursive bisection construction must not special-case away at
        this size."""
        time_grid = np.array([0.0, 1.0])
        B = _build_bridge_matrix(time_grid)
        assert B.shape == (1, 1)
        np.testing.assert_allclose(B[0, 0], 1.0, atol=1e-10)

    def test_uneven_grid_spacing_still_reproduces_covariance(self):
        """The covariance identity (B @ B.T == min(s,t)) must hold for
        irregular step sizes, not just the evenly-spaced demo grid --
        exercises the recursive bisection's midpoint selection more
        thoroughly."""
        time_grid = np.array([0.0, 0.1, 0.15, 1.0, 1.2, 5.0])
        B = _build_bridge_matrix(time_grid)
        times = time_grid[1:]
        expected_cov = np.minimum.outer(times, times)
        actual_cov = B @ B.T
        np.testing.assert_allclose(actual_cov, expected_cov, atol=1e-10)


class TestComputeHwAMatrixEdgeCases:
    def test_sloped_zero_curve_reprices_exactly(self):
        """TestHullWhiteAMatrix only exercises FLAT zero curves. A(t,T) must
        also exactly reprice a genuinely sloped (non-flat) curve at t->0,
        since compute_hw_A_matrix's forward-rate finite-difference and
        interpolation logic could silently be wrong specifically when
        adjacent zero rates differ."""
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
        # reprices each pillar's own zero rate under continuous compounding
        expected = np.exp(-np.array(zero_curves[0].rates[1:]) * maturities)
        np.testing.assert_allclose(discount_factors, expected, atol=1e-4)

    def test_three_or_more_rate_factors(self):
        """TestHullWhiteAMatrix only covers 1 and 2 factors -- confirm the
        per-factor loop generalizes to a larger NumHW axis without
        cross-contaminating adjacent factors."""
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
    """Regression coverage for a bug where the Hull-White short-rate step
    (_simulate_cross_asset_paths_jit's step_fn) computed
    `r_next = r_t*decay + theta_hw + shock_hw` instead of the correct
    exact Ornstein-Uhlenbeck transition
    `r_next = r_t*decay + theta_hw*(1-decay) + shock_hw`. The missing
    `(1-decay)` factor made theta act as a flat per-step drift increment
    rather than the long-run mean-reversion target, so the simulated short
    rate drifted upward (or downward, depending on sign) WITHOUT BOUND
    every step instead of reverting toward theta.

    This was invisible in every pre-existing demo/test because they all
    set theta == initial_rates -- a fixed point only under the CORRECT
    formula (theta*(1-decay) + r0*decay == r0 exactly when theta==r0), so
    the divergence never showed up in those single-value comparisons. It
    was only caught by directly checking the simulated distribution's mean
    against the closed-form OU transition mean over several steps, with a
    scenario matching the codebase's own existing swap-demo cadence
    (dt=0.5, a=0.03): under the buggy formula the mean rate drifted from
    3% at t=0 to ~14.6% by t=2y in that exact scenario.
    """

    def test_mean_matches_analytic_ou_transition_multi_step(self):
        """theta == initial_rates (the case every pre-existing config uses)
        must stay at its fixed point across every step -- the buggy formula
        failed this exact case."""
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
        """theta != initial_rates is the case that actually exposes the
        bug numerically (the buggy formula's fixed point isn't r0==theta
        here, so it visibly diverges) -- confirms the simulated mean
        converges toward theta from below, following the exact closed-form
        OU transition step by step, not just "ends up somewhere higher
        than r0"."""
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
        """Mirror of the above with theta < r0, confirming the fix
        reverts DOWNWARD correctly too, not just upward (the sign of
        theta - r0 flips which direction the old bug's extra drift term
        pushed the mean)."""
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
        """The variance formula was NOT part of the bug (confirmed
        separately against ORE.HullWhiteProcess.variance() directly), but
        is pinned down here too so a future change to the same step
        formula can't silently break it while fixing something else."""
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
    """Regression coverage for a second, independent bug found alongside
    the mean-reversion one: L_t (the per-step Cholesky factor used to
    correlate shocks) was built from the RAW covariance matrix, whose
    diagonal already encodes each factor's own volatility magnitude. The
    HW1F/GBM step formulas then multiplied the already-scaled shock by
    that SAME factor's volatility a second time
    (`sig_hw * sqrt(variance_hw) * Z_hw`, `sig_eq * sqrt(dt) * Z_eq`),
    squaring the effective volatility actually applied to every path.
    E.g. a configured 20% equity vol produced an actual simulated
    log-return std of ~4% (0.2^2); a configured 1.5% rate vol produced an
    actual short-rate std smaller by the same squared factor. This affected
    every equity, FX, and rate factor in every simulation the codebase has
    ever run. Fixed by building L_t from the CORRELATION matrix (unit
    diagonal) instead of the raw covariance matrix, so joint_sigma_t's
    explicit multiplication in step_fn is the only place volatility is
    applied.
    """

    def test_equity_log_return_variance_matches_configured_vol(self):
        """The clearest possible signal: a configured 20% vol MUST produce
        an actual ~20% log-return std, not ~4% (0.2^2, the bug's
        signature)."""
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
        # the bug's signature: actual std would be ~sigma times smaller
        # than expected (e.g. ~0.0105 * 0.015 = 0.000157 instead of 0.0105)
        assert r_t.std() > expected_std / 10.0
        np.testing.assert_allclose(r_t.std(), expected_std, rtol=0.03)

    def test_correlation_between_equity_and_rate_is_preserved(self):
        """The correlation-only-Cholesky fix must still reproduce the
        CONFIGURED correlation between factors, not just get each factor's
        own marginal variance right in isolation -- a fix that broke
        cross-correlation while fixing marginal variance would be an
        equally real regression."""
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
        """Two rate factors with DIFFERENT volatilities and nonzero
        cross-correlation -- confirms the fix generalizes beyond the
        single-equity/single-rate case to a multi-rate-factor covariance
        block, matching engine.scenarios.cross_asset_demo_config's actual
        shape."""
        sig_a, sig_b, rho = 0.012, 0.008, -0.3
        # generate_paths requires >=1 equity/FX factor (see
        # engine.scenarios.swaption_demo_config's own placeholder pattern)
        # -- use one zero-drift placeholder equity to isolate the two rate
        # factors' own covariance block.
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
    """Cross-checks _build_bridge_matrix / apply_brownian_bridge against
    QuantLib/ORE's own C++ BrownianBridge directly (not just re-deriving
    the same covariance formula the implementation uses). ORE's
    BrownianBridge::transform(begin,end,output) -- see
    reference/ORE/QuantLib/ql/methods/montecarlo/brownianbridge.hpp -- is
    documented to return the SAME thing apply_brownian_bridge's
    Z_sequential does: standardized (unit-variance), TIME-ordered
    increments, not raw path values (a plain covariance-of-columns probe
    of ORE.BrownianBridge.transform on unit vectors confirms this: it
    comes back orthonormal, i.e. NOT equal to min(s,t), because transform
    already normalizes by sqrt(dt) internally -- only the accumulated
    W-then-diff-then-normalize output matches our Z_sequential)."""

    def _ore_transform_all(self, times, Z):
        """Z: [TimeSteps, N] array of independent normals (Sobol-dimension
        order). Returns ORE's transform() output with the same shape,
        applied independently per column via ORE.BrownianBridge."""
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
        """Non-uniform step sizes exercise the recursive bisection's
        midpoint selection more thoroughly than the evenly-spaced demo
        grid does."""
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
    """TestBrownianBridge / TestBrownianBridgeEdgeCases only check the
    covariance identity B@B.T == min(s,t) at the raw-matrix level for two
    grid shapes. Here we check the full apply_brownian_bridge pipeline's
    statistical output (reconstructed path values, not just increments)
    at several more grid shapes, confirming the actual Brownian-motion
    property: Var[W(t)] == t at every grid time, and each increment's
    variance equals its own dt."""

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

        # increment variance must equal its own dt
        increment_var = Z_seq.var(axis=(1, 2)) * dt  # Z_seq already standardized -> multiply back by dt
        np.testing.assert_allclose(increment_var, dt, rtol=0.05)

        # path value variance at each grid time must equal that time (BM property)
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
        """A repeated time value (dt=0 for the first interior step) is an
        edge case a caller could plausibly construct by accident; document
        actual behavior rather than assume it's handled."""
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
    """Cross-checks the HW1F closed-form pieces (OU transition variance,
    and the A(t,T)/B(t,T) discount-bond formula) against ORE's own
    HullWhiteProcess / HullWhite classes directly, independent of this
    codebase's own re-derivation of the same formulas."""

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
        """ORE's HullWhite.discountBond(now, maturity, rate) implements the
        exact same closed-form A(t,T)*exp(-B(t,T)*r) affine formula that
        compute_hw_A_matrix/reconstruct_yield_curves implement -- a
        genuinely independent implementation to check against, not merely
        the same formula copy-pasted into the test."""
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
        """Same cross-check with a genuinely sloped curve, so the
        forward-rate finite-difference / interpolation logic is exercised
        against ORE's own curve interpolation too, not just a flat rate."""
        a, sigma = 0.1, 0.01
        times = [0.0, 1.0, 2.0, 5.0, 10.0, 30.0]
        rates = [0.020, 0.025, 0.028, 0.032, 0.035, 0.038]
        dc = ORE.Actual365Fixed()
        eval_date = ORE.Date(30, 7, 2026)
        dates = [eval_date + int(round(t * 365)) for t in times]
        # ORE's ZeroCurve needs dates[0] == eval_date (t=0 pillar)
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
        # slightly looser tolerance: ORE's ZeroCurve interpolation (log-linear
        # discount / default interpolator) is not bit-identical to this
        # codebase's linear-on-zero-rate interpolation used in
        # _initial_log_discount -- both are valid choices, so a few bp of
        # difference from interpolation-scheme choice alone is expected.
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
        """np.interp (used by _initial_log_discount) flat-extrapolates
        beyond the pillar range by construction -- confirm this holds
        through the full A(t,T) pipeline, not just at the np.interp call
        site, and that it does NOT silently produce nonsense (e.g.
        negative/NaN discount factors) for a maturity far past the last
        pillar."""
        zc = [ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0], rates=[0.02, 0.025, 0.03, 0.035])]
        df = self._reprices(zc, maturities=np.array([5.0, 10.0, 20.0, 50.0]))
        # beyond t=5.0 the effective zero rate is flat-clamped at 0.035
        expected = np.exp(-0.035 * np.array([10.0, 20.0, 50.0]))
        np.testing.assert_allclose(df[1:], expected, atol=1e-4)
        assert np.all(df > 0.0) and np.all(np.isfinite(df))
        # discount factors must still be monotonically decreasing even
        # under flat extrapolation
        assert np.all(np.diff(df) < 0)


class TestZeroVolatilityAndSingularMeanReversion:
    """hw_a (mean_reversion) appears in a literal denominator in three
    places: the B(t,T) formula (both generate_paths' inline version and
    every test's local reimplementation), the HW1F transition's
    variance_hw = (1-exp(-2*a*dt))/(2*a), and compute_hw_A_matrix's
    variance_term = sigma^2/(4a) * (...). a=0 is a removable 0/0
    singularity in those formulas as literally written; its analytic
    a->0 limit (arithmetic Brownian motion) is well-defined:
    B(t,T)->T-t, variance_hw->dt, compute_hw_A_matrix's
    variance_term->sigma^2*t/2. Similarly, a factor with exactly zero
    variance makes the correlation-normalization step a 0/0 whose naive
    NaN propagates through jnp.linalg.cholesky and poisons every other,
    unrelated factor. This class is regression coverage for the fix:
    both singularities are guarded (jnp.where against a safe placeholder
    denominator) so they produce the correct finite limit instead of
    NaN."""

    def test_zero_mean_reversion_matches_arithmetic_brownian_motion_limit(self):
        """mean_reversion=0.0 (arithmetic Brownian motion, the a->0 limit
        of OU mean reversion) now produces finite output whose simulated
        mean and variance match the closed-form ABM limit: E[r(t)] = r0
        (since decay=exp(0)=1, theta's own contribution vanishes) and
        Var[r(t)] = sigma^2 * t (the variance_hw->dt limit accumulated
        over each step, scaled by sigma^2)."""
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
        """sigma=0 for every factor is not a singularity in the HW1F/GBM
        step formulas themselves (shock_hw/shock_eq are simply multiplied
        by sigma=0) -- with the correlation-normalization 0/0 guarded,
        every scenario now collapses onto the same noiseless drift/OU-mean
        path exactly, with zero cross-scenario variance."""
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
        # theta == r0, a fixed point of the OU mean -- every scenario/step
        # should sit at exactly r0 with zero variance.
        np.testing.assert_allclose(np.asarray(result["rates"]), r0, atol=1e-9)
        assert float(jnp.var(result["rates"])) == pytest.approx(0.0, abs=1e-12)

    def test_one_zero_volatility_factor_does_not_poison_others(self):
        """Sharper variant: only ONE factor (the equity) has zero
        variance; the rate factor has an entirely normal, nonzero 0.01 vol
        and zero configured correlation to the equity. The two factors
        being uncorrelated means one factor's degeneracy must not affect
        the other -- the equity path is exactly deterministic (100.0
        throughout, zero drift/vol configured) while the rate factor is
        finite and matches its own well-defined HW1F distribution around
        theta=0.03."""
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
        """Negative a (divergent/anti-mean-reverting OU) is not a formula
        singularity (only a=0 is) -- confirm the implementation actually
        runs and produces finite output, without asserting a particular
        blow-up rate."""
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
    """rho=+-1 and non-PSD covariance matrices are not validated anywhere
    in generate_paths before jnp.linalg.cholesky is called on the
    correlation matrix. This class documents the actual (silent-NaN, not
    a raised error) behavior, and confirms it's an inherent floating-point
    fact about Cholesky at exact rank-deficient boundaries (also reproduced
    with plain numpy/scipy, not a JAX-specific defect) rather than
    something generate_paths could trivially avoid by swapping libraries."""

    def test_non_positive_semidefinite_covariance_silently_produces_nan(self):
        """rho implied by off-diagonal/sqrt(diag product) > 1 is not a
        valid correlation at all (mathematically invalid input) -- no
        validation catches this before jnp.linalg.cholesky, which does not
        raise; it silently returns NaN, which then propagates through
        every downstream path silently instead of failing loudly at the
        config-validation boundary."""
        cfg = SimulationConfig(
            time_grid=[0.0, 1.0],
            scenarios=32,
            equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[1.0]]),
            rates=RatesConfig(initial_rates=[0.03], theta=[0.03], mean_reversion=[0.1]),
            # off-diagonal 0.05 vs sqrt(0.04*0.0001) ~= 0.002 -- invalid, implies rho >> 1
            joint_covariance=[[0.04, 0.05], [0.05, 0.0001]],
        )
        result = generate_paths(cfg)
        assert bool(jnp.any(jnp.isnan(result["rates"]))), (
            "Expected generate_paths to silently propagate NaN for a "
            "non-PSD joint_covariance (jnp.linalg.cholesky does not raise "
            "on invalid input); if this now raises or produces finite "
            "output, generate_paths' validation behavior has changed."
        )

    def test_boundary_rho_equals_one_is_numerically_singular_not_a_jax_defect(self):
        """rho=1.0 is theoretically PSD (rank-deficient, smallest eigenvalue
        exactly 0), but the resulting 2x2 covariance is numerically
        singular/ill-conditioned under floating point (its smallest
        eigenvalue computes as ~1e-20, not exactly 0), so BOTH plain
        numpy's and jax.numpy's Cholesky reject it identically -- numpy
        raises LinAlgError, jax silently returns an all-NaN factor.
        Confirms this is an inherent floating-point property of Cholesky
        at the exact correlation boundary, not something specific to (or
        avoidable by) this codebase's particular choice of
        jnp.linalg.cholesky over an alternative library."""
        sig1, sig2 = 0.2, 0.01
        cov = np.array([[sig1 ** 2, 1.0 * sig1 * sig2], [1.0 * sig1 * sig2, sig2 ** 2]])
        eigvals = np.linalg.eigvalsh(cov)
        # the boundary case's smallest eigenvalue is only correct to
        # floating-point noise, not exactly the mathematical 0 -- confirms
        # the input is genuinely at the numerical edge of PSD-ness.
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
    """generate_paths validates every documented "one entry per factor"
    cross-field length invariant up front (initial_zero_curves vs num rate
    factors; RatesConfig.theta/mean_reversion vs initial_rates;
    EquityConfig.rate_mapping row/column counts vs num equities/rate
    factors; joint_covariance's overall shape) rather than relying on
    jnp broadcasting/jnp.dot shape rules to catch a mismatch incidentally
    (which used to sometimes raise a low-level JAX error and sometimes
    silently broadcast to a wrong-but-shaped result -- see git history for
    the pre-fix behavior). This class checks each contract is now enforced
    with a clear ValueError instead."""

    def test_theta_shorter_than_initial_rates_raises(self):
        """theta has 1 entry but there are 2 rate factors -- must raise
        rather than jnp.tile/broadcasting silently reusing theta[0] for
        BOTH rate factors, which used to silently violate the RatesConfig
        docstring's documented 'one entry per rate factor' contract."""
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
        """2 equities configured but rate_mapping has only 1 row -- must
        raise rather than letting dynamic_mu = jnp.dot(r_t, rate_mapping.T)
        silently broadcast a wrong-shaped drift."""
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
        """By contrast, a joint_covariance whose size doesn't match
        num_eq+num_hw at all (not just an ambiguous-but-broadcastable
        mismatch) DOES fail loudly, via jnp.dot's contracting-dimension
        check inside the jitted step function -- confirming that some,
        but not all, shape mismatches are caught."""
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
        """A larger factor count (8 equities, 6 rate factors) than any
        existing test exercises, with a random-but-valid PSD covariance
        matrix, confirms the pipeline generalizes rather than only working
        for the 1-4 factor configs every other test uses."""
        rng = np.random.default_rng(7)
        num_eq, num_hw = 8, 6
        n = num_eq + num_hw
        # build a random valid PSD covariance matrix via A @ A.T + small ridge
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
    """NaN/Inf in the config should not be silently swallowed into a
    plausible-looking but wrong result -- confirm they propagate visibly
    (as NaN/Inf in the output) rather than, say, being clipped or treated
    as zero."""

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
        # rates are unaffected by an infinite equity price (no feedback path)
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
    """Direct unit coverage of _initial_log_discount's documented
    flat-extrapolation behavior at the pillar boundaries, isolated from
    the full compute_hw_A_matrix pipeline."""

    def test_extrapolates_flat_below_first_pillar(self):
        zero_times = np.array([1.0, 2.0, 5.0])
        zero_rates = np.array([0.02, 0.03, 0.04])
        # t=0.5 is before the first pillar (1.0) -- np.interp clamps to rate[0]
        log_p = _initial_log_discount(zero_times, zero_rates, np.array([0.5]))
        np.testing.assert_allclose(log_p, -0.02 * 0.5, atol=1e-12)

    def test_extrapolates_flat_above_last_pillar(self):
        zero_times = np.array([0.0, 1.0, 2.0, 5.0])
        zero_rates = np.array([0.02, 0.025, 0.03, 0.035])
        log_p = _initial_log_discount(zero_times, zero_rates, np.array([10.0, 100.0]))
        expected = -0.035 * np.array([10.0, 100.0])
        np.testing.assert_allclose(log_p, expected, atol=1e-12)
