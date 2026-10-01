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

from engine.models.lgm import H as lgm_H
from engine.risk.var_es import value_at_risk, expected_shortfall
from engine.simulation.random import _build_bridge_matrix

TODAY = ORE.Date(30, 7, 2026)
FLAT_RATE = 0.03


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
            mine_B = float(lgm_H(a, jnp.asarray(t)))
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
