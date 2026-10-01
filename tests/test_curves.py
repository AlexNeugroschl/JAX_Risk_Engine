"""
Curve primitives (`engine.models.curves`) against QuantLib's `InterpolatedZeroCurve<Linear>`
(`ORE.ZeroCurve`): inside the pillars, and past the last one, where QuantLib holds the
instantaneous forward flat (`ContinuousForward`, I-48). Sloped curves only: a flat curve
cannot tell a flat zero from a flat forward.
"""
import jax
import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.models.curves import ZeroCurve, discount, forward_rate, log_discount, zero_rate

ASOF = ORE.Date(30, 7, 2026)
DC = ORE.Actual365Fixed()
# Whole ACT/365 days, so the ORE curve has exactly these times.
PILLARS = [0.0, 73 / 365, 1.0, 2.0, 5.0, 10.0]
CURVES = {
    "upward": [0.030, 0.031, 0.034, 0.038, 0.045, 0.050],
    "inverted": [0.050, 0.049, 0.045, 0.040, 0.034, 0.030],
    "humped": [0.020, 0.030, 0.045, 0.040, 0.030, 0.025],
}
TIMES = [0.05, 0.5, 1.0, 3.3, 9.99, 10.0, 10.01, 12.0, 30.0, 60.0]


def _ore_curve(rates):
    ORE.Settings.instance().evaluationDate = ASOF
    curve = ORE.ZeroCurve([ASOF + round(t * 365) for t in PILLARS], rates, DC)
    curve.enableExtrapolation()
    return curve


def _engine_curve(rates):
    return ZeroCurve(pillar_times=jnp.asarray(PILLARS), pillar_rates=jnp.asarray(rates))


@pytest.mark.parametrize("name", CURVES)
def test_discount_factors_equal_ore_inside_and_beyond_the_pillars(name):
    ore, engine = _ore_curve(CURVES[name]), _engine_curve(CURVES[name])
    for t in TIMES:
        assert float(discount(engine, jnp.asarray(t))) == pytest.approx(ore.discount(t), rel=1e-14, abs=1e-15)


@pytest.mark.parametrize("name", CURVES)
def test_zero_rates_equal_ore(name):
    ore, engine = _ore_curve(CURVES[name]), _engine_curve(CURVES[name])
    for t in TIMES:
        expected = ore.zeroRate(t, ORE.Continuous).rate()
        assert float(zero_rate(engine, jnp.asarray(t))) == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize("name", CURVES)
def test_the_forward_is_the_exact_derivative_and_flat_beyond_the_last_pillar(name):
    engine = _engine_curve(CURVES[name])
    for t in [0.3, 1.5, 4.0, 10.0, 11.0, 25.0]:
        exact = -jax.grad(lambda s: log_discount(engine, s))(jnp.asarray(t))
        assert float(forward_rate(engine, jnp.asarray(t))) == pytest.approx(float(exact), rel=1e-12)
    beyond = np.asarray(forward_rate(engine, jnp.asarray([10.0, 12.0, 40.0])))
    assert np.ptp(beyond) == 0.0


def test_the_zero_rate_is_flat_before_the_first_pillar():
    curve = ZeroCurve(pillar_times=jnp.asarray([1.0, 2.0, 5.0]), pillar_rates=jnp.asarray([0.02, 0.03, 0.04]))
    assert float(log_discount(curve, jnp.asarray(0.5))) == pytest.approx(-0.02 * 0.5, abs=1e-15)


def test_a_single_pillar_curve_is_flat_everywhere():
    curve = ZeroCurve(pillar_times=jnp.asarray([1.0]), pillar_rates=jnp.asarray([0.03]))
    np.testing.assert_allclose(np.asarray(zero_rate(curve, jnp.asarray([0.5, 1.0, 7.0]))), 0.03, rtol=1e-15)


def test_extrapolation_is_differentiable_in_the_pillar_rates():
    """Greeks differentiate through the extrapolation: a cashflow past the last pillar is
    sensitive to the last two pillars (the last segment's slope), and finitely so."""
    rates = jnp.asarray(CURVES["upward"])
    grad = jax.grad(lambda r: discount(ZeroCurve(jnp.asarray(PILLARS), r), jnp.asarray(15.0)))(rates)
    assert np.all(np.isfinite(np.asarray(grad)))
    assert np.count_nonzero(np.asarray(grad)) == 2


class TestForwardRatePrecision:
    """`engine.models.curves.forward_rate` is exact. It was once a 1e-6 finite difference on ln P,
    which in float32 put forward rates off by up to ~2 percentage points."""

    TIMES = [0.0, 1.0, 2.0, 5.0, 10.0, 30.0]
    RATES = [0.03, 0.031, 0.032, 0.035, 0.037, 0.04]

    def _exact(self, t):
        # f = z + t*z' (the slope of the segment to the right) inside the pillars; from the
        # last pillar on, the last segment's forward, held flat (I-48).
        times, rates = np.asarray(self.TIMES), np.asarray(self.RATES)
        slopes = np.diff(rates) / np.diff(times)
        idx = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(slopes) - 1)
        inside = np.interp(t, times, rates) + t * np.where(t >= times[0], slopes[idx], 0.0)
        return np.where(t >= times[-1], rates[-1] + times[-1] * slopes[-1], inside)

    @pytest.mark.parametrize("dtype, atol", [(jnp.float64, 1e-12), (jnp.float32, 1e-6)])
    def test_matches_exact_derivative(self, dtype, atol):
        t = np.array([0.0, 0.5, 1.0, 1.5, 3.0, 4.5, 7.0, 9.5, 30.0, 40.0])
        curve = ZeroCurve(jnp.asarray(self.TIMES, dtype=dtype), jnp.asarray(self.RATES, dtype=dtype))
        got = np.asarray(forward_rate(curve, jnp.asarray(t, dtype=dtype)), dtype=np.float64)
        np.testing.assert_allclose(got, self._exact(t), atol=atol, rtol=0)
