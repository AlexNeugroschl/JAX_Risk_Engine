"""
Algorithm-level parity with QuantLib/QuantExt C++ (reference/ORE; mapping in
docs/reference/ore-parity.md). Each test reimplements a small algorithm independently from
its description in the C++ source and checks the engine's function against it (and, where
possible, against the live ORE object), so a formula drift is caught even if a few recorded
values still match. reference/ORE is read by people, not imported; it need not be present.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import ORE
import pytest
from scipy.optimize import brentq

from engine.simulation.market_model import (
    ZeroCurveConfig,
    _build_bridge_matrix,
    compute_hw_A_matrix,
)
from engine.instruments.european_swaption import (
    SwaptionConfig,
    prepare_swaption,
    _hw_B,
    _solve_rstar,
)
from engine.models.hull_white import A as hw_A, ZeroCurve
from engine.risk.var_es import value_at_risk, expected_shortfall

TODAY = ORE.Date(30, 7, 2026)
FLAT_RATE = 0.03
HW_A = 0.03
HW_SIGMA = 0.01
ZERO_CURVE = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[FLAT_RATE] * 6)


def _curve(prepared):
    """A prepared swaption's today's curve, as `hull_white.A` takes it."""
    return ZeroCurve(pillar_times=jnp.asarray(prepared.zero_times), pillar_rates=jnp.asarray(prepared.zero_rates))


class TestBrownianBridgeParity:
    """`QuantLib::BrownianBridge` (ql/methods/montecarlo/brownianbridge.cpp) builds the last
    point first and bisects the widest remaining gap. Its defining property,
    Cov(W(s), W(t)) = min(s, t), is what is checked: any construction with that covariance
    is the same bridge."""

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
    """`QuantExt::Lgm1fConstantParametrization` (irlgm1fconstantparametrization.hpp), what
    `ORE.CrossAssetModel` uses for a constant rates factor. With scaling 1 and shift 0,
    H(t) = B(0, t) = (1 - exp(-a t))/a, zeta(t) = sigma^2 t and alpha(t) = sigma; checked
    against the live ORE object."""

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
    """`QuantLib::JamshidianSwaptionEngine::rStarFinder` finds x where
    strike - sum_i amounts[i] * P(T0, t_i, x) / P(T0, valueTime, x) = 0, with valueTime the
    first fixed accrual start (not T0). Reimplemented as an independent root-find and checked
    against `_solve_rstar`, which states the same condition as a signed extra cashflow
    (docs/reference/ore-parity.md#6)."""

    def _independent_rstar(self, prepared, a, sigma):
        """rStarFinder's condition, root-found directly with the engine's A(t,T) as the bond
        price (the condition is under test, not the bond formula)."""
        T0 = prepared.exercise_time
        T_start = prepared.accrual_start_time
        times = list(prepared.fixed_cashflow_times) + [prepared.fixed_cashflow_times[-1]]
        amounts = list(prepared.fixed_cashflow_amounts) + [prepared.notional]
        strike = prepared.notional

        def discount_bond(t, T, x):
            if abs(T - t) < 1e-12:
                return 1.0
            A = float(hw_A(_curve(prepared), jnp.asarray(t), jnp.asarray(T), a, sigma))
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
        """The engine's own leg setup and `_solve_rstar`."""
        T0 = prepared.exercise_time
        T_start = prepared.accrual_start_time
        cf_times = prepared.fixed_cashflow_times
        all_times = np.concatenate([cf_times, cf_times[-1:], [T_start]])
        all_amounts = jnp.asarray(
            list(prepared.fixed_cashflow_amounts) + [prepared.notional, -prepared.notional]
        )
        A_T0 = hw_A(_curve(prepared), jnp.full_like(jnp.asarray(all_times), T0), jnp.asarray(all_times), a, sigma)
        B_T0 = _hw_B(T0, jnp.asarray(all_times), a)

        def coupon_bond_value(r, params):
            A_T0_p, all_amounts_p = params
            prices = A_T0_p[None, None, :] * jnp.exp(-B_T0[None, None, :] * r[..., None])
            return jnp.sum(prices * all_amounts_p[None, None, :], axis=-1)

        rstar = _solve_rstar(coupon_bond_value, (A_T0, all_amounts), (1, 1))
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
    """`QuantLib::GeneralStatistics::percentile` sorts (weight, value) pairs and walks
    forward while the running weight is below percent * total. Reimplemented as a weighted
    cumulative walk and checked against the engine's VaR/ES and `ORE.RiskStatistics`."""

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
    """`QuantLib::HullWhite::A` computes exp(B(t,T)*f(0,t) - 0.25*(sigma*B(t,T))^2*B(0,2t))
    * P(0,T)/P(0,t). Since B(0,2t) = (1-exp(-2at))/a, its variance term equals the engine's
    (sigma^2/4a)*(1-exp(-2at))*B(t,T)^2. Checks that identity, and that
    `compute_hw_A_matrix` reproduces `ORE.HullWhite.discountBond`."""

    @pytest.mark.parametrize("a,sigma,t", [
        (0.03, 0.01, 1.0), (0.08, 0.015, 3.0), (0.001, 0.02, 5.0), (0.5, 0.03, 0.25),
    ])
    def test_variance_term_algebraic_identity(self, a, sigma, t):
        # QuantLib's variance term uses B(t,T) for the bond and B(0, 2t) for the decay.
        B_0_2t = (1.0 - np.exp(-a * 2.0 * t)) / a
        T = t + 2.5  # arbitrary bond maturity to exercise B(t,T)
        B_t_T = (1.0 - np.exp(-a * (T - t))) / a
        quantlib_variance_term = 0.25 * (sigma * B_t_T) ** 2 * B_0_2t
        engine_variance_term = (sigma ** 2 / (4.0 * a)) * (1.0 - np.exp(-2.0 * a * t)) * B_t_T ** 2
        np.testing.assert_allclose(quantlib_variance_term, engine_variance_term, rtol=1e-13)

    def test_reprices_live_ore_hullwhite_discount_bond(self):
        """`compute_hw_A_matrix` reproduces `ORE.HullWhite.discountBond(t, T, r)`."""
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
