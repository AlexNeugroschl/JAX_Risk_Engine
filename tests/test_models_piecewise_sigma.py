"""
Piecewise-constant sigma(t) in `engine.models.lgm` (`Sigma`, `as_sigma`, `zeta`), checked
against ORE's `IrLgm1fPiecewiseConstantParametrization` and `LinearGaussMarkovModel`.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.models.hull_white import ZeroCurve
from engine.models.lgm import H, Sigma, as_sigma, bond_price, numeraire, zeta

TODAY = ORE.Date(30, 7, 2026)
FLAT_CURVE = ZeroCurve(pillar_times=jnp.array([0.0, 30.0]), pillar_rates=jnp.array([0.03, 0.03]))


def _ore_piecewise_param(alpha_times, alpha_values, a=0.03):
    ORE.Settings.instance().evaluationDate = TODAY
    dc = ORE.Actual365Fixed()
    curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, 0.03, dc))
    return ORE.IrLgm1fPiecewiseConstantParametrization(
        ORE.USDCurrency(), curve,
        ORE.Array(alpha_times), ORE.Array(alpha_values),
        ORE.Array([]), ORE.Array([a]),
    )


class TestSigmaFlatIsBackwardCompatible:
    def test_flat_sigma_zeta_matches_scalar_formula(self):
        s = Sigma.flat(0.02)
        for t in [0.0, 0.5, 1.0, 5.0, 30.0]:
            expected = 0.02 ** 2 * t
            assert float(zeta(s, jnp.array(t))) == pytest.approx(expected, rel=1e-12)

    def test_plain_float_auto_upgrades_via_as_sigma(self):
        """A plain float sigma is accepted by zeta/bond_price/numeraire."""
        for t in [0.0, 1.0, 3.0]:
            direct = zeta(0.02, jnp.array(t))
            via_sigma = zeta(Sigma.flat(0.02), jnp.array(t))
            assert float(direct) == pytest.approx(float(via_sigma), rel=1e-12)

    def test_as_sigma_idempotent_on_existing_sigma(self):
        s = Sigma(times=jnp.array([1.0]), values=jnp.array([0.01, 0.02]))
        assert as_sigma(s) is s


class TestZetaMatchesOREPiecewiseParametrization:
    """zeta against `ORE.IrLgm1fPiecewiseConstantParametrization.zeta`."""

    @pytest.mark.parametrize("t", [0.0, 0.5, 1.0, 1.5, 2.999, 3.0, 3.5, 5.0, 10.0])
    def test_matches_ore_three_bucket_curve(self, t):
        alpha_times = [1.0, 3.0]
        alpha_values = [0.008, 0.015, 0.02]
        ore_param = _ore_piecewise_param(alpha_times, alpha_values)
        sigma = Sigma(times=jnp.array(alpha_times), values=jnp.array(alpha_values))

        ore_zeta = ore_param.zeta(t)
        mine = float(zeta(sigma, jnp.array(t)))
        assert mine == pytest.approx(ore_zeta, rel=1e-10, abs=1e-14)

    @pytest.mark.parametrize("t", [0.0, 0.3, 0.5, 0.7, 1.0, 2.0])
    def test_matches_ore_single_breakpoint(self, t):
        alpha_times = [0.5]
        alpha_values = [0.01, 0.025]
        ore_param = _ore_piecewise_param(alpha_times, alpha_values)
        sigma = Sigma(times=jnp.array(alpha_times), values=jnp.array(alpha_values))
        ore_zeta = ore_param.zeta(t)
        mine = float(zeta(sigma, jnp.array(t)))
        assert mine == pytest.approx(ore_zeta, rel=1e-10, abs=1e-14)

    @pytest.mark.parametrize("t", [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0])
    def test_matches_ore_five_bucket_curve(self, t):
        alpha_times = [1.0, 2.0, 5.0, 7.0]
        alpha_values = [0.005, 0.01, 0.015, 0.012, 0.02]
        ore_param = _ore_piecewise_param(alpha_times, alpha_values)
        sigma = Sigma(times=jnp.array(alpha_times), values=jnp.array(alpha_values))
        ore_zeta = ore_param.zeta(t)
        mine = float(zeta(sigma, jnp.array(t)))
        assert mine == pytest.approx(ore_zeta, rel=1e-10, abs=1e-13)

    def test_vectorized_over_multiple_t_matches_ore_pointwise(self):
        """Array t (as in the Bermudan backward induction) matches ORE's scalar zeta
        element by element."""
        alpha_times = [1.0, 3.0]
        alpha_values = [0.008, 0.015, 0.02]
        ore_param = _ore_piecewise_param(alpha_times, alpha_values)
        sigma = Sigma(times=jnp.array(alpha_times), values=jnp.array(alpha_values))

        t_grid = jnp.array([0.1, 0.9, 1.0, 1.1, 2.5, 3.0, 3.1, 8.0])
        mine = np.asarray(zeta(sigma, t_grid))
        expected = np.array([ore_param.zeta(float(t)) for t in t_grid])
        np.testing.assert_allclose(mine, expected, rtol=1e-10, atol=1e-13)


class TestHIndependentOfSigmaPiecewise:
    """H(t) does not depend on sigma (as in ORE, where H uses only the reversion)."""

    @pytest.mark.parametrize("t", [0.5, 1.0, 3.0, 7.5])
    def test_H_same_regardless_of_sigma_shape(self, t):
        a = 0.03
        h_flat = float(H(a, jnp.array(t)))
        # H takes no sigma argument; this confirms it structurally.
        ore_param_flat = _ore_piecewise_param([], [0.02], a=a)
        ore_param_piecewise = _ore_piecewise_param([1.0, 3.0], [0.008, 0.015, 0.02], a=a)
        assert h_flat == pytest.approx(ore_param_flat.H(t), abs=1e-12)
        assert h_flat == pytest.approx(ore_param_piecewise.H(t), abs=1e-12)


class TestBondPriceAndNumeraireWithPiecewiseSigma:
    """Bond price and numeraire with a piecewise `Sigma` against a live
    `ORE.LinearGaussMarkovModel` on the same parametrization."""

    def _ore_lgm(self, alpha_times, alpha_values, a=0.03):
        param = _ore_piecewise_param(alpha_times, alpha_values, a=a)
        return ORE.LinearGaussMarkovModel(param)

    @pytest.mark.parametrize("t,T,x", [
        (0.5, 5.0, 0.0), (1.0, 3.0, 0.05), (3.0, 5.01643836, -0.1),
        (2.999, 5.0, 0.02), (0.0, 2.0, 0.0),
    ])
    def test_bond_price_matches_ore(self, t, T, x):
        alpha_times = [1.0, 3.0]
        alpha_values = [0.008, 0.015, 0.02]
        lgm = self._ore_lgm(alpha_times, alpha_values)
        sigma = Sigma(times=jnp.array(alpha_times), values=jnp.array(alpha_values))

        ore_val = lgm.discountBond(t, T, x)
        mine = float(bond_price(FLAT_CURVE, 0.03, sigma, jnp.array(t), jnp.array(T), jnp.array(x)))
        assert mine == pytest.approx(ore_val, rel=1e-9)

    @pytest.mark.parametrize("t,x", [(0.5, 0.0), (1.5, 0.02), (3.5, -0.05), (0.0, 0.0)])
    def test_numeraire_matches_ore(self, t, x):
        alpha_times = [1.0, 3.0]
        alpha_values = [0.008, 0.015, 0.02]
        lgm = self._ore_lgm(alpha_times, alpha_values)
        sigma = Sigma(times=jnp.array(alpha_times), values=jnp.array(alpha_values))

        ore_val = lgm.numeraire(t, x)
        mine = float(numeraire(FLAT_CURVE, 0.03, sigma, jnp.array(t), jnp.array(x)))
        assert mine == pytest.approx(ore_val, rel=1e-9)

    def test_bond_price_reduces_to_flat_sigma_case(self):
        """`Sigma.flat` gives the same bond price as the equivalent float."""
        t, T, x, a, sig = 2.0, 5.0, 0.03, 0.03, 0.015
        via_float = bond_price(FLAT_CURVE, a, sig, jnp.array(t), jnp.array(T), jnp.array(x))
        via_sigma = bond_price(FLAT_CURVE, a, Sigma.flat(sig), jnp.array(t), jnp.array(T), jnp.array(x))
        assert float(via_float) == pytest.approx(float(via_sigma), rel=1e-14)


class TestSigmaGradientCorrectness:
    """zeta/bond_price are differentiable in `Sigma.values` (what calibration fits and Vega
    differentiates), checked against finite differences."""

    def test_zeta_gradient_wrt_sigma_values_matches_finite_difference(self):
        t = jnp.array(2.5)
        times = jnp.array([1.0, 3.0])

        def f(values):
            return zeta(Sigma(times=times, values=values), t)

        values0 = jnp.array([0.008, 0.015, 0.02])
        grad = jax.grad(f)(values0)

        eps = 1e-6
        for i in range(3):
            bumped = values0.at[i].add(eps)
            fd = (float(f(bumped)) - float(f(values0))) / eps
            assert float(grad[i]) == pytest.approx(fd, rel=1e-3, abs=1e-8)

    def test_bond_price_gradient_wrt_sigma_values_is_finite(self):
        """The full bond price has a finite gradient in `Sigma.values`."""
        def f(values):
            sigma = Sigma(times=jnp.array([1.0, 3.0]), values=values)
            return bond_price(FLAT_CURVE, 0.03, sigma, jnp.array(2.0), jnp.array(5.0), jnp.array(0.01))

        grad = jax.grad(f)(jnp.array([0.008, 0.015, 0.02]))
        assert bool(jnp.all(jnp.isfinite(grad)))
