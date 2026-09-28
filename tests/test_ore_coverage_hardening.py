"""
Tests of the ORE cross-checks' own power to detect errors, plus curve shapes the other
Hull-White comparisons do not use.

1. Mutation testing (`TestVarianceTermIsActuallyChecked`, `TestForwardTermIsActuallyChecked`):
   corrupt one term of the Hull-White A(t,T) and assert the price moves by more than the
   tolerance the ORE comparisons use. A mutation that does not move the price is a term no
   test can see.

   At t=0 the variance term has a factor (1 - exp(-2at)) = 0, so deleting it changes an
   ATM swaption by ~7e-6, inside the rtol=1e-4 of the t=0 comparisons. Once only one test
   (conditional pricing at a later date) caught it (I-30); a conditional grid in
   tests/test_european_swaption.py now does, and
   `test_every_conditional_grid_point_catches_the_mutation` asserts each grid point does.

2. Curve shape (`TestNonFlatCurvesAgainstORE`): on a flat curve f(0,t) is constant and any
   interpolation gives the same number, so errors there are invisible. A(t,T) is checked on
   upward, inverted, humped and negative curves. Away from pillars the engine matches ORE's
   `HullWhite.discountBond` to ~1e-8; at a pillar they differ by up to ~1.5e-2, because
   under linear zero interpolation f(0,t) has a kink there and the two libraries evaluate
   the one-sided forward differently (`test_pillar_times_differ_by_the_interpolation_kink`).
"""
import numpy as np
import jax
import jax.numpy as jnp
import ORE
import pytest

import engine.models.hull_white as hull_white
import engine.instruments.european_swaption as european_swaption
from engine.simulation.market_model import ZeroCurveConfig, compute_hw_A_matrix
from engine.instruments.european_swaption import SwaptionConfig, prepare_swaption
from test_european_swaption import CONDITIONAL_GRID, _price_conditional

EVAL_DATE = ORE.Date(30, 7, 2026)
DC = ORE.Actual365Fixed()
FLAT_RATE = 0.03
HW_A = 0.03
HW_SIGMA = 0.01
FLAT_CURVE = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[FLAT_RATE] * 6)

# The tolerance the ORE swaption comparisons assert (rtol=1e-4 in
# tests/test_european_swaption.py); a smaller move is invisible to them.
SUITE_RTOL = 1e-4


# ---------------------------------------------------------------------------
# Mutation machinery
# ---------------------------------------------------------------------------
def _mutated_A(scale_variance=1.0, variance_sign=-1.0, drop_forward=False):
    """`engine.models.hull_white.A` with one term corrupted, otherwise identical.
    `scale_variance=0.0` deletes the variance term; `variance_sign=+1.0` flips it;
    `scale_variance=2.0` is the sigma^2/4a -> sigma^2/2a slip; `drop_forward=True` deletes
    B(t,T)*f(0,t)."""
    def A(curve, t, T, a, sigma, B_override=None):
        log_P0_t = hull_white.log_discount(curve, t)
        log_P0_T = hull_white.log_discount(curve, T)
        fwd_0_t = hull_white.forward_rate(curve, t)
        ratio = jnp.exp(log_P0_T - log_P0_t)
        B_t_T = hull_white.B(t, T, a) if B_override is None else B_override
        a_safe = jnp.where(a == 0.0, 1.0, a)
        variance_term = jnp.where(
            a == 0.0,
            0.5 * sigma ** 2 * t,
            (sigma ** 2 / (4.0 * a_safe)) * (1.0 - jnp.exp(-2.0 * a_safe * t)),
        ) * scale_variance
        forward_term = 0.0 if drop_forward else B_t_T * fwd_0_t
        return ratio * jnp.exp(forward_term + variance_sign * variance_term * B_t_T ** 2)
    return A


class _mutate_hull_white_A:
    """Swap in a mutated A(t,T) where `engine.instruments.european_swaption` calls it.

      * The module binds `A` at import (`from ... import A as _hw_A`), so the name must be
        replaced in that module, not in `engine.models.hull_white`.
      * `_price_one_swaption` is jitted, so its cache is cleared on entry and on exit, or
        the mutation is ignored or leaks into later tests.
    """

    def __init__(self, **kwargs):
        self._mutation = _mutated_A(**kwargs)
        self._original = None

    def __enter__(self):
        self._original = european_swaption.__dict__["_hw_A"]
        european_swaption.__dict__["_hw_A"] = self._mutation
        european_swaption._price_one_swaption.clear_cache()
        return self

    def __exit__(self, *exc):
        european_swaption.__dict__["_hw_A"] = self._original
        european_swaption._price_one_swaption.clear_cache()
        return False


def _swaption_price(evaluation_time, short_rate, forward_start_years=3,
                    fixed_rate=0.03, payer=True, tenor="5Y"):
    """One swaption priced conditional on `short_rate` at `evaluation_time`, as
    tests/test_european_swaption.py does."""
    cfg = SwaptionConfig(
        notional=1_000_000.0, fixed_rate=fixed_rate, payer=payer, rate_factor_index=0,
        hw_a=HW_A, hw_sigma=HW_SIGMA, initial_zero_curve=FLAT_CURVE, swap_tenor=tenor,
        forward_start=(ORE.Period(forward_start_years, ORE.Years) if forward_start_years
                       else ORE.Period(0, ORE.Days)),
        evaluation_date=EVAL_DATE,
    )
    prepared = prepare_swaption(cfg)
    npv = european_swaption._price_one_swaption(
        jnp.array([[[short_rate]]]), jnp.array([evaluation_time]), prepared)
    return float(npv[0, 0])


MUTATIONS = [
    ("delete-variance-term", dict(scale_variance=0.0)),
    ("flip-variance-sign", dict(variance_sign=+1.0)),
    ("sigma2-over-2a-not-4a", dict(scale_variance=2.0)),
]


class TestMutationHarnessItself:
    """Guards that the harness really applies mutations (else every test below passes
    vacuously)."""

    def test_patch_reaches_the_jitted_pricer(self):
        """An unmistakable mutation changes the price (the import binding and jit cache are
        handled)."""
        baseline = _swaption_price(1.0, 0.03)
        with _mutate_hull_white_A(drop_forward=True):
            mutated = _swaption_price(1.0, 0.03)
        assert abs(mutated - baseline) / baseline > 0.5

    def test_original_is_restored(self):
        """The mutation does not leak into later tests."""
        baseline = _swaption_price(1.0, 0.03)
        with _mutate_hull_white_A(drop_forward=True):
            pass
        assert _swaption_price(1.0, 0.03) == pytest.approx(baseline, rel=1e-12)


class TestVarianceTermIsActuallyChecked:
    """The variance term (sigma^2/4a)*(1-exp(-2at))*B(t,T)^2, QuantLib's
    `0.25*(sigma*B(t,T))^2*B(0,2t)`."""

    @pytest.mark.parametrize("name,kwargs", MUTATIONS, ids=[m[0] for m in MUTATIONS])
    @pytest.mark.parametrize("t_eval,short_rate", [(1.0, 0.03), (2.0, 0.04)])
    def test_mutation_is_visible_at_t_greater_than_zero(self, name, kwargs, t_eval, short_rate):
        """At t > 0 each corruption moves the price 3-8%, far outside the 1e-4 tolerance."""
        baseline = _swaption_price(t_eval, short_rate)
        with _mutate_hull_white_A(**kwargs):
            mutated = _swaption_price(t_eval, short_rate)
        relative_change = abs(mutated - baseline) / abs(baseline)
        assert relative_change > SUITE_RTOL * 100, (
            f"{name} at t={t_eval} changes the price by only {relative_change:.2e}; "
            f"a conditional-pricing test at this tolerance would not catch it"
        )

    @pytest.mark.parametrize("name,kwargs", MUTATIONS, ids=[m[0] for m in MUTATIONS])
    def test_variance_mutation_is_invisible_at_t0(self, name, kwargs):
        """At t=0 the term's factor (1 - exp(-2at)) is zero, so corrupting it is invisible to
        any t=0 comparison. Pinned as a fact: if the term ever matters at t=0 this fails and
        the coverage assumptions need revisiting."""
        baseline = _swaption_price(0.0, FLAT_RATE)
        with _mutate_hull_white_A(**kwargs):
            mutated = _swaption_price(0.0, FLAT_RATE)
        relative_change = abs(mutated - baseline) / abs(baseline)
        assert relative_change < SUITE_RTOL, (
            f"{name} now moves the t=0 price by {relative_change:.2e}, which is "
            f"outside the suite's own rtol -- the documented t=0 blind spot for "
            f"the variance term no longer holds and this test should be revisited"
        )

    def test_conditional_pricing_coverage_is_load_bearing(self):
        """At the later-date conditional comparison point (t=1), deleting the variance term
        is caught (the dependency recorded as I-30)."""
        baseline = _swaption_price(1.0, 0.03)
        with _mutate_hull_white_A(scale_variance=0.0):
            mutated = _swaption_price(1.0, 0.03)
        assert abs(mutated - baseline) / abs(baseline) > SUITE_RTOL, (
            "deleting the variance term is no longer detectable even at t>0; "
            "the suite would not catch a wrong A(t,T) at all"
        )

    @pytest.mark.parametrize("name,kwargs", MUTATIONS, ids=[m[0] for m in MUTATIONS])
    def test_every_conditional_grid_point_catches_the_mutation(self, name, kwargs):
        """Every point of
        `test_european_swaption.py::test_conditional_pricing_matches_ore_across_t_and_r`
        moves by at least 10x its tolerance (rtol=1e-4, atol=1e-2) under the mutation
        (measured minimum ~30x), so each point catches it on its own. Engine against itself;
        that test already ties the unmutated engine to ORE to ~2e-6."""
        points = [p.values for p in CONDITIONAL_GRID]
        baselines = [_price_conditional(*p) for p in points]
        with _mutate_hull_white_A(**kwargs):
            mutated = [_price_conditional(*p) for p in points]
        weak = [
            (p, abs(m - b) / (SUITE_RTOL * abs(b) + 1e-2))
            for p, b, m in zip(points, baselines, mutated)
            if abs(m - b) < 10 * (SUITE_RTOL * abs(b) + 1e-2)
        ]
        assert not weak, (
            f"{name} moves these conditional grid points by less than 10x the "
            f"grid's tolerance (point, multiple of tolerance): {weak}"
        )


class TestForwardTermIsActuallyChecked:
    """The B(t,T)*f(0,t) term, which carries today's curve into A(t,T)."""

    @pytest.mark.parametrize("t_eval,short_rate", [(0.0, FLAT_RATE), (1.0, 0.03), (2.0, 0.04)])
    def test_dropping_forward_term_is_always_visible(self, t_eval, short_rate):
        """Deleting B(t,T)*f(0,t) is visible everywhere, including t=0 (>100% change)."""
        baseline = _swaption_price(t_eval, short_rate)
        with _mutate_hull_white_A(drop_forward=True):
            mutated = _swaption_price(t_eval, short_rate)
        assert abs(mutated - baseline) / abs(baseline) > SUITE_RTOL * 100


class TestNonFlatCurvesAgainstORE:
    """A(t,T) against `ORE.HullWhite.discountBond` on non-flat curves, where the forward
    and interpolation terms matter."""

    SHAPES = {
        "upward": [0.010, 0.015, 0.020, 0.030, 0.040, 0.050],
        "inverted": [0.050, 0.045, 0.040, 0.030, 0.025, 0.020],
        "humped": [0.020, 0.030, 0.040, 0.045, 0.035, 0.025],
        "negative": [-0.005, -0.002, 0.000, 0.005, 0.010, 0.015],
    }
    PILLARS = [0.0, 1.0, 2.0, 5.0, 10.0, 30.0]

    def _ore_curve(self, rates):
        """An ORE `ZeroCurve` on the same pillars with the same linear zero-rate
        interpolation (its `discount(T)` equals exp(-r(T)*T) exactly at and between
        pillars)."""
        ORE.Settings.instance().evaluationDate = EVAL_DATE
        dates = [EVAL_DATE + int(round(t * 365)) for t in self.PILLARS]
        return ORE.YieldTermStructureHandle(ORE.ZeroCurve(dates, list(rates), DC))

    def _engine_bond(self, rates, t, T, r):
        step_times = np.array([t])
        maturities = np.array([T])
        B = (1.0 - np.exp(-HW_A * np.maximum(
            maturities[None, :, None] - step_times[:, None, None], 0.0))) / HW_A
        A = compute_hw_A_matrix(
            [ZeroCurveConfig(times=self.PILLARS, rates=list(rates))],
            np.array([HW_A]), np.array([HW_SIGMA]), step_times, maturities, B)
        return float(A[0, 0, 0] * np.exp(-B[0, 0, 0] * r))

    @pytest.mark.parametrize("shape", list(SHAPES))
    def test_curve_itself_matches_ore(self, shape):
        """Control: the two agree on the curve itself before any model is involved."""
        rates = self.SHAPES[shape]
        curve = self._ore_curve(rates)
        for T in [0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 30.0]:
            expected = float(np.exp(-np.interp(T, self.PILLARS, rates) * T))
            np.testing.assert_allclose(curve.discount(T), expected, rtol=1e-12)

    @pytest.mark.parametrize("shape", list(SHAPES))
    @pytest.mark.parametrize("t,T", [(1.5, 4.5), (2.5, 7.5), (3.5, 8.5), (0.5, 3.5)])
    def test_discount_bond_matches_ore_off_pillar(self, shape, t, T):
        """Away from pillars the engine matches ORE's `discountBond` to ~1e-8 on every
        shape, including a negative curve."""
        rates = self.SHAPES[shape]
        hw = ORE.HullWhite(self._ore_curve(rates), HW_A, HW_SIGMA)
        for r in (0.03, 0.0, -0.01):
            mine = self._engine_bond(rates, t, T, r)
            np.testing.assert_allclose(mine, hw.discountBond(t, T, r), rtol=1e-6)

    def test_pillar_times_differ_by_the_interpolation_kink(self):
        """At a pillar f(0,t) has a kink (different left and right derivatives), and the two
        libraries evaluate it differently: ~1e-2 apart at a pillar, ~1e-8 half a year away.
        Neither is wrong. A move to smooth interpolation would remove the kink, fail this,
        and should prompt extending the off-pillar test to pillar times."""
        rates = self.SHAPES["upward"]
        hw = ORE.HullWhite(self._ore_curve(rates), HW_A, HW_SIGMA)

        at_pillar = abs(self._engine_bond(rates, 5.0, 10.0, 0.04)
                        - hw.discountBond(5.0, 10.0, 0.04)) / hw.discountBond(5.0, 10.0, 0.04)
        off_pillar = abs(self._engine_bond(rates, 4.5, 10.0, 0.04)
                         - hw.discountBond(4.5, 10.0, 0.04)) / hw.discountBond(4.5, 10.0, 0.04)

        assert off_pillar < 1e-6, f"off-pillar agreement lost: {off_pillar:.2e}"
        assert at_pillar > 1e-4, (
            f"the pillar-kink divergence ({at_pillar:.2e}) has disappeared -- if the "
            f"curve interpolation is now smooth, pillar times should be folded into "
            f"test_discount_bond_matches_ore_off_pillar"
        )
