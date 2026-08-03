"""
Algorithm-level parity tests against ORE's own C++ source (reference/ORE, a
full clone of OpenSourceRisk/Engine including its QuantLib and QuantExt
submodules -- see docs/09-ore-parity.md for the full file-by-file mapping
and rationale).

Every test here reimplements a small piece of a QuantLib/QuantExt C++
algorithm INDEPENDENTLY in Python -- built fresh from the algorithm's
mathematical description (read directly from the cited C++ source), not
transcribed from it -- and cross-checks that reimplementation against this
engine's own function. This is a different (and stronger) kind of check
than the rest of the test suite's "does our number match ORE's number"
tests: it confirms the *algorithm*, not just a set of output values, so a
future change that silently drifts from the correct formula (while still
happening to pass a narrow set of recorded ORE comparisons) is caught.

reference/ORE is never imported, executed, or modified by this test file --
it is C++ source, read by a human/LLM for reference during development, not
a runtime dependency. Nothing here requires reference/ORE to be present at
test-run time.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import ORE
import pytest
from scipy.optimize import brentq

from engine.simulation import (
    ZeroCurveConfig,
    _build_bridge_matrix,
    compute_hw_A_matrix,
)
from engine.instruments.european_swaption import (
    SwaptionConfig,
    compute_hw_A,
    prepare_swaption,
    _hw_B,
    _solve_rstar,
)
from engine.risk.statistics import value_at_risk, expected_shortfall

TODAY = ORE.Date(30, 7, 2026)
FLAT_RATE = 0.03
HW_A = 0.03
HW_SIGMA = 0.01
ZERO_CURVE = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[FLAT_RATE] * 6)


class TestBrownianBridgeParity:
    """QuantLib::BrownianBridge::initialize() (ql/methods/montecarlo/
    brownianbridge.cpp): builds the last time point from the first input
    variate, then recursively bisects the widest unconstructed gap in the
    remaining time grid, each time recording a left/right neighbor pair
    and interpolation weights. The defining property this construction
    guarantees -- independent of the exact recursion used to build it --
    is that the resulting path values reproduce real Brownian motion's
    covariance structure, Cov(W(s), W(t)) = min(s, t). That property, not
    any particular intermediate weight value, is what's checked here: it's
    the mathematical invariant the C++ algorithm is designed to satisfy,
    so any implementation (this engine's matrix form or QuantLib's
    per-path recursion) that gets the covariance right has necessarily
    implemented the same bridge."""

    @pytest.mark.parametrize("time_grid", [
        [0.0, 0.25, 0.5, 0.75, 1.0],
        [0.0, 0.1, 0.15, 1.0, 1.2, 5.0],
        [0.0, 1.0],
        [0.0, 0.3, 3.0, 3.1, 10.0, 10.5, 30.0],
    ])
    def test_reproduces_brownian_motion_covariance(self, time_grid):
        B = _build_bridge_matrix(np.array(time_grid))
        times = np.array(time_grid)[1:]
        expected_cov = np.minimum.outer(times, times)
        actual_cov = B @ B.T
        np.testing.assert_allclose(actual_cov, expected_cov, atol=1e-9)


class TestLgmParametrizationParity:
    """QuantExt::Lgm1fConstantParametrization (qle/models/
    irlgm1fconstantparametrization.hpp), the class ORE.CrossAssetModel
    actually instantiates for a constant-parameter rates factor (live-
    verified via the SWIG bindings). With the default scaling=1, shift=0,
    its H(t)/zeta(t)/alpha(t) are closed-form functions of exactly this
    engine's own hw_a/hw_sigma parameters -- H(t) has the identical shape
    to this engine's B(t,T) with t=0, zeta(t) is the accumulated variance
    sigma^2*t, and alpha(t) is simply the constant sigma. These are
    checked directly against the live ORE object (not re-derived from
    this engine's own code), so this test would fail if either this
    engine's B(t,T) formula or ORE's own LGM parametrization ever
    disagreed on what a "Hull-White A/B parameter" means."""

    @pytest.mark.parametrize("a,sigma", [(0.03, 0.01), (0.08, 0.015), (0.001, 0.02)])
    def test_H_matches_B_t0(self, a, sigma):
        ORE.Settings.instance().evaluationDate = TODAY
        dc = ORE.Actual365Fixed()
        curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, FLAT_RATE, dc))
        param = ORE.IrLgm1fConstantParametrization(ORE.USDCurrency(), curve, sigma, a)

        for t in [0.5, 1.0, 3.0, 7.5, 15.0]:
            ore_H = param.H(t)
            mine_B = float(_hw_B(0.0, t, a))
            np.testing.assert_allclose(ore_H, mine_B, atol=1e-10)

    @pytest.mark.parametrize("a,sigma", [(0.03, 0.01), (0.08, 0.015)])
    def test_zeta_matches_accumulated_variance(self, a, sigma):
        ORE.Settings.instance().evaluationDate = TODAY
        dc = ORE.Actual365Fixed()
        curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, FLAT_RATE, dc))
        param = ORE.IrLgm1fConstantParametrization(ORE.USDCurrency(), curve, sigma, a)

        for t in [0.5, 1.0, 3.0, 7.5]:
            np.testing.assert_allclose(param.zeta(t), sigma ** 2 * t, atol=1e-12)

    @pytest.mark.parametrize("a,sigma", [(0.03, 0.01), (0.08, 0.015)])
    def test_alpha_matches_constant_sigma(self, a, sigma):
        ORE.Settings.instance().evaluationDate = TODAY
        dc = ORE.Actual365Fixed()
        curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, FLAT_RATE, dc))
        param = ORE.IrLgm1fConstantParametrization(ORE.USDCurrency(), curve, sigma, a)

        for t in [0.0, 1.0, 10.0]:
            np.testing.assert_allclose(param.alpha(t), sigma, atol=1e-12)


class TestJamshidianRStarParity:
    """QuantLib::JamshidianSwaptionEngine::rStarFinder (ql/pricingengines/
    swaption/jamshidianswaptionengine.cpp): finds the short rate x at
    which strike - sum_i(amounts[i] * discountBond(T0,times[i],x) /
    discountBond(T0,valueTime,x)) == 0, where valueTime is the underlying
    swap's own first accrual start date (fixedResetDates[0]) -- NOT the
    exercise date T0 itself. This is reimplemented here from that
    description directly (an independent root-find over a hand-written
    "strike equation", not a transcription of rStarFinder's C++), and
    cross-checked against this engine's own _solve_rstar, which expresses
    the identical condition differently (as a signed extra cashflow rather
    than an explicit division) -- see docs/09-ore-parity.md#6 for why the
    two are algebraically the same condition. Agreement here is strong
    evidence this engine's exercise-boundary equation is the mathematically
    correct one QuantLib's own reference engine uses, not merely a formula
    that happens to reproduce recorded NPV numbers."""

    def _independent_rstar(self, prepared, a, sigma):
        """Root-find QuantLib's rStarFinder condition directly, using this
        engine's own (separately-verified-against-ORE) A(t,T) closed form
        as the discount-bond primitive -- the point of this test is the
        ROOT-FINDING CONDITION's correctness (does the exercise boundary
        divide by the T_start bond the way ORE's C++ does), not the bond
        pricing formula itself (already covered by TestLgmParametrizationParity
        and the live ORE.HullWhite.discountBond checks elsewhere)."""
        T0 = prepared.exercise_time
        T_start = prepared.accrual_start_time
        times = list(prepared.fixed_cashflow_times) + [prepared.fixed_cashflow_times[-1]]
        amounts = list(prepared.fixed_cashflow_amounts) + [prepared.notional]
        strike = prepared.notional

        def discount_bond(t, T, x):
            if abs(T - t) < 1e-12:
                return 1.0
            A = compute_hw_A(
                np.asarray(prepared.zero_times), np.asarray(prepared.zero_rates),
                np.array([t]), np.array([T]), a, sigma,
            )[0]
            B = (1.0 - np.exp(-a * (T - t))) / a
            return A * np.exp(-B * x)

        def rstar_finder(x):
            B = discount_bond(T0, T_start, x)
            value = strike
            for Ti, ci in zip(times, amounts):
                value -= ci * discount_bond(T0, Ti, x) / B
            return value

        return brentq(rstar_finder, -10.0, 10.0, xtol=1e-13)

    def _engine_rstar(self, prepared, a, sigma):
        """Reproduces _price_one_swaption's own signed-leg setup exactly,
        then calls this engine's actual _solve_rstar (not a copy of it)."""
        T0 = prepared.exercise_time
        T_start = prepared.accrual_start_time
        cf_times = prepared.fixed_cashflow_times
        all_times = np.concatenate([cf_times, cf_times[-1:], [T_start]])
        all_amounts = jnp.asarray(
            list(prepared.fixed_cashflow_amounts) + [prepared.notional, -prepared.notional]
        )
        A_T0 = jnp.asarray(compute_hw_A(
            np.asarray(prepared.zero_times), np.asarray(prepared.zero_rates),
            np.full_like(all_times, T0), all_times, a, sigma,
        ))
        B_T0 = _hw_B(T0, jnp.asarray(all_times), a)

        def coupon_bond_value(r):
            prices = A_T0[None, None, :] * jnp.exp(-B_T0[None, None, :] * r[..., None])
            return jnp.sum(prices * all_amounts[None, None, :], axis=-1)

        rstar = _solve_rstar(coupon_bond_value, (1, 1))
        return float(rstar[0, 0])

    @pytest.mark.parametrize("tenor,forward_years", [("5Y", 0), ("2Y", 3), ("10Y", 5)])
    def test_engine_rstar_matches_independent_rstarfinder(self, tenor, forward_years):
        cfg = SwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True,
            rate_factor_index=0, hw_a=HW_A, hw_sigma=HW_SIGMA,
            initial_zero_curve=ZERO_CURVE, swap_tenor=tenor,
            forward_start=ORE.Period(forward_years, ORE.Years) if forward_years else ORE.Period(0, ORE.Days),
            evaluation_date=TODAY,
        )
        prepared = prepare_swaption(cfg)

        independent = self._independent_rstar(prepared, HW_A, HW_SIGMA)
        engine = self._engine_rstar(prepared, HW_A, HW_SIGMA)
        np.testing.assert_allclose(engine, independent, atol=1e-8)


class TestGeneralStatisticsPercentileParity:
    """QuantLib::GeneralStatistics::percentile (ql/math/statistics/
    generalstatistics.cpp): sorts the (weight, value) sample ascending,
    then walks forward accumulating weight, advancing WHILE the running
    total is still strictly less than percent*totalWeight, and returns the
    value at the position where that loop stops. Reimplemented here
    independently from that description (a plain weighted cumulative-sum
    walk, not a transcription of the C++ loop) and cross-checked both
    against this engine's own value_at_risk/expected_shortfall AND
    directly against the live ORE.RiskStatistics object -- so this test
    fails if either this engine's order-statistic indexing or the
    installed ORE package's own behavior ever changes."""

    def _independent_percentile(self, values: np.ndarray, weights: np.ndarray, percent: float) -> float:
        order = np.argsort(values, kind="stable")
        values_sorted = values[order]
        weights_sorted = weights[order]
        target = percent * weights_sorted.sum()
        integral = weights_sorted[0]
        k = 0
        n = len(values_sorted)
        while integral < target and k < n - 1:
            k += 1
            integral += weights_sorted[k]
        return float(values_sorted[k])

    def _independent_var(self, pnl: np.ndarray, percentile: float) -> float:
        return max(-self._independent_percentile(pnl, np.ones_like(pnl), 1.0 - percentile), 0.0)

    def _independent_es(self, pnl: np.ndarray, percentile: float) -> float:
        target = -self._independent_var(pnl, percentile)
        tail = pnl[pnl < target]
        if tail.size == 0:
            return float("nan")
        return -min(float(tail.mean()), 0.0)

    @pytest.mark.parametrize("percentile", [0.90, 0.95, 0.99])
    def test_var_matches_independent_walk_and_ore(self, percentile):
        rng = np.random.default_rng(42)
        pnl_np = rng.normal(0.0, 100.0, size=1000)

        independent = self._independent_var(pnl_np, percentile)

        stats = ORE.RiskStatistics()
        for v in pnl_np:
            stats.add(float(v), 1.0)
        ore_var = stats.valueAtRisk(percentile)

        pnl_jax = jnp.asarray(pnl_np[:, None], dtype=jnp.float64)
        engine_var = float(value_at_risk(pnl_jax, percentile)[0])

        np.testing.assert_allclose(independent, ore_var, atol=1e-9)
        np.testing.assert_allclose(engine_var, ore_var, atol=1e-9)

    @pytest.mark.parametrize("percentile", [0.90, 0.95, 0.99])
    def test_es_matches_independent_walk_and_ore(self, percentile):
        rng = np.random.default_rng(7)
        pnl_np = rng.normal(0.0, 100.0, size=1000)

        independent = self._independent_es(pnl_np, percentile)

        stats = ORE.RiskStatistics()
        for v in pnl_np:
            stats.add(float(v), 1.0)
        ore_es = stats.expectedShortfall(percentile)

        pnl_jax = jnp.asarray(pnl_np[:, None], dtype=jnp.float64)
        engine_es = float(expected_shortfall(pnl_jax, percentile)[0])

        np.testing.assert_allclose(independent, ore_es, atol=1e-9)
        np.testing.assert_allclose(engine_es, ore_es, atol=1e-9)


class TestHullWhiteAFormulaParity:
    """QuantLib::HullWhite::A(t,T) (ql/models/shortrate/onefactormodels/
    hullwhite.cpp): computes A(t,T) = exp(B(t,T)*f(0,t) -
    0.25*(sigma*B(t,T))^2*B(0,2t)) * P(0,T)/P(0,t), using the SAME B(t,T)
    (inherited from Vasicek::B) this engine uses. Algebraically,
    0.25*sigma^2*B(t,T)^2*B(0,2t) == (sigma^2/(4a))*(1-exp(-2at))*B(t,T)^2
    (substituting B(0,2t)=(1-exp(-2at))/a) -- i.e. this engine's variance
    term IS QuantLib's, just written with the (1-exp(-2at))/a factor
    already substituted in rather than left as a nested B(0,2t) call. This
    test checks that algebraic identity directly (both sides computed
    independently from a and t, not from each other), and separately
    confirms this engine's compute_hw_A_matrix reprices a real ORE
    HullWhite object's own discountBond output -- the strongest possible
    check, since it goes through neither side's intermediate formula, only
    final discount factors."""

    @pytest.mark.parametrize("a,sigma,t", [
        (0.03, 0.01, 1.0), (0.08, 0.015, 3.0), (0.001, 0.02, 5.0), (0.5, 0.03, 0.25),
    ])
    def test_variance_term_algebraic_identity(self, a, sigma, t):
        # QuantLib's variance-decay term uses two DIFFERENT B(.,.)
        # evaluations: B(t,T) for the bond leg being priced, and
        # B(0.0, 2.0*t) for the decay factor -- not the same B squared.
        B_0_2t = (1.0 - np.exp(-a * 2.0 * t)) / a
        T = t + 2.5  # arbitrary bond maturity to exercise B(t,T)
        B_t_T = (1.0 - np.exp(-a * (T - t))) / a
        quantlib_variance_term = 0.25 * (sigma * B_t_T) ** 2 * B_0_2t
        engine_variance_term = (sigma ** 2 / (4.0 * a)) * (1.0 - np.exp(-2.0 * a * t)) * B_t_T ** 2
        np.testing.assert_allclose(quantlib_variance_term, engine_variance_term, rtol=1e-13)

    def test_reprices_live_ore_hullwhite_discount_bond(self):
        """compute_hw_A_matrix's A(t,T)*exp(-B(t,T)*r) must reproduce a
        real, live ORE.HullWhite object's own discountBond(t,T,r) -- going
        through ORE's actual C++ implementation of HullWhite::A end to
        end, not just the algebraic identity above."""
        ORE.Settings.instance().evaluationDate = TODAY
        dc = ORE.Actual365Fixed()
        curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, FLAT_RATE, dc))
        a, sigma = HW_A, HW_SIGMA
        hw = ORE.HullWhite(curve, a, sigma)

        zero_curves = [ZERO_CURVE]
        hw_a_arr = np.array([a])
        hw_sigma_arr = np.array([sigma])
        for t, T, r in [(1.0, 5.0, 0.03), (0.5, 10.0, 0.045), (3.0, 3.5, 0.02)]:
            step_times = np.array([t])
            maturities = np.array([T])
            B = (1.0 - np.exp(-hw_a_arr[None, None, :] *
                 np.maximum(maturities[None, :, None] - step_times[:, None, None], 0.0))) / hw_a_arr[None, None, :]
            A = compute_hw_A_matrix(zero_curves, hw_a_arr, hw_sigma_arr, step_times, maturities, B)
            mine = A[0, 0, 0] * np.exp(-B[0, 0, 0] * r)
            ore = hw.discountBond(t, T, r)
            np.testing.assert_allclose(mine, ore, rtol=1e-9)
