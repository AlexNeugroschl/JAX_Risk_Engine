import jax.numpy as jnp
import numpy as np
import pytest

from engine.market_simulations import (
    EquityConfig,
    RatesConfig,
    SimulationConfig,
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
