"""
Coverage-hardening tests for the ORE cross-checks: tests ABOUT the test
suite's own discriminating power, plus the curve shapes the existing ORE
comparisons never exercise.

This file answers a criticism the project already makes of itself
(docs/known-issues.md: "A green suite is evidence about the *tests*, not
proof about the *code*", and the observation that none of the defects
found during the TraderX integration came from running this suite). Every
other ORE test asserts that a number matches. Nothing asserts that the
tolerance those tests use is tight enough to NOTICE if the formula were
wrong. These do.

Two distinct concerns:

1. MUTATION TESTING (`TestVarianceTermIsActuallyChecked`,
   `TestForwardTermIsActuallyChecked`). Deliberately corrupt a term of the
   Hull-White A(t,T) formula and assert the resulting price moves by more
   than the tolerance the ORE comparisons use. A mutation that does NOT
   move the price is a term no test can see -- the formula could be wrong
   in that respect and the suite would stay green.

   This found something real and non-obvious, which
   `test_variance_mutation_is_invisible_at_t0` now pins: at t=0 the
   variance term of A(t,T) carries a factor (1 - exp(-2*a*t)) which is
   IDENTICALLY ZERO, so at t=0 the term contributes nothing and deleting
   it entirely changes an ATM swaption price by ~7e-6 relative -- an order
   of magnitude INSIDE the rtol=1e-4 those tests assert. The great
   majority of this suite's ORE swaption comparisons price at t=0. Run
   against the real suite, deleting the whole variance term fails exactly
   ONE test in tests/test_european_swaption.py (131 tests):
   `TestConditionalPricingAndExpiry::test_conditional_pricing_matches_ore_rebuilt_at_later_date`,
   the one that prices at a later evaluation date. That single test is
   carrying the entire suite's coverage of this term. These tests make
   that dependency explicit and load-bearing, so that deleting or
   weakening that one test fails here loudly rather than silently
   un-covering a term of the core bond-price formula.

2. CURVE SHAPE (`TestNonFlatCurvesAgainstORE`). Every ORE comparison in
   the suite runs on a FLAT curve at a single rate (3%). A flat curve
   cannot detect an error in either curve interpolation or the
   instantaneous-forward term f(0,t) of A(t,T), because on a flat curve
   f(0,t) == the flat rate for every t and any interpolation scheme
   returns the same number. These run the same A(t,T) check against
   upward, inverted, humped and negative-rate curves.

   That sweep also established a real and previously unrecorded boundary,
   pinned in `test_pillar_times_differ_by_the_interpolation_kink`: away
   from curve pillars this engine reproduces ORE's own
   `HullWhite.discountBond` to ~1e-8 on every shape tested, but AT a
   pillar the two disagree by up to ~1.5e-2. That is not a bug in either:
   under linear zero-rate interpolation f(0,t) has a genuine kink at each
   pillar (its left and right derivatives differ), so the instantaneous
   forward is undefined there and the two libraries resolve it
   differently. It is a documented-behavior boundary worth a standing
   test, because a future interpolation change would move it.
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

EVAL_DATE = ORE.Date(30, 7, 2026)
DC = ORE.Actual365Fixed()
FLAT_RATE = 0.03
HW_A = 0.03
HW_SIGMA = 0.01
FLAT_CURVE = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[FLAT_RATE] * 6)

# The tolerance the ORE swaption comparisons actually assert
# (tests/test_european_swaption.py uses rtol=1e-4 throughout). A mutation
# that moves a price by less than this is invisible to those tests.
SUITE_RTOL = 1e-4


# ---------------------------------------------------------------------------
# Mutation machinery
# ---------------------------------------------------------------------------
def _mutated_A(scale_variance=1.0, variance_sign=-1.0, drop_forward=False):
    """A stand-in for `engine.models.hull_white.A` with one term corrupted.

    Mirrors the real implementation exactly except for the requested
    corruption, so a price difference is attributable to that one change
    and nothing else. `scale_variance=0.0` deletes the variance term;
    `variance_sign=+1.0` flips it; `scale_variance=2.0` is the
    (sigma^2/4a) -> (sigma^2/2a) slip; `drop_forward=True` deletes the
    B(t,T)*f(0,t) term.
    """
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
    """Context manager swapping the A(t,T) that
    `engine.instruments.european_swaption` actually calls.

    Two details make this work, and both are easy to get wrong:

      * The swaption module does `from engine.models.hull_white import A as
        _hw_A`, which BINDS THE FUNCTION OBJECT at import time. Patching
        `engine.models.hull_white.A` therefore has no effect at all -- the
        bound name in the consuming module must be replaced instead.
      * `_price_one_swaption` is `jax.jit`-decorated, so a patch applied
        after it has been traced once is ignored: the cached executable
        still contains the original formula. Its cache must be cleared on
        the way in AND on the way out, or the mutation leaks into later
        tests in the same session.
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
    """Prices one swaption conditional on `short_rate` observed at
    `evaluation_time` -- the same conditional-pricing call shape
    tests/test_european_swaption.py uses."""
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
    """If the harness silently fails to apply a mutation, every test below
    it passes vacuously. These are the guards against that."""

    def test_patch_reaches_the_jitted_pricer(self):
        """A mutation with an unmistakable effect must actually change the
        price. This is what proves the import-binding + jit-cache handling
        in `_mutate_hull_white_A` works; without it a broken harness would
        make every mutation look 'invisible'."""
        baseline = _swaption_price(1.0, 0.03)
        with _mutate_hull_white_A(drop_forward=True):
            mutated = _swaption_price(1.0, 0.03)
        assert abs(mutated - baseline) / baseline > 0.5

    def test_original_is_restored(self):
        """The mutation must not leak into other tests in the session --
        the jit cache makes that a real risk, not a theoretical one."""
        baseline = _swaption_price(1.0, 0.03)
        with _mutate_hull_white_A(drop_forward=True):
            pass
        assert _swaption_price(1.0, 0.03) == pytest.approx(baseline, rel=1e-12)


class TestVarianceTermIsActuallyChecked:
    """The variance term of A(t,T) -- (sigma^2/4a)*(1-exp(-2at))*B(t,T)^2,
    QuantLib's `0.25*(sigma*B(t,T))^2*B(0,2t)`."""

    @pytest.mark.parametrize("name,kwargs", MUTATIONS, ids=[m[0] for m in MUTATIONS])
    @pytest.mark.parametrize("t_eval,short_rate", [(1.0, 0.03), (2.0, 0.04)])
    def test_mutation_is_visible_at_t_greater_than_zero(self, name, kwargs, t_eval, short_rate):
        """At t>0 each corruption moves the price by 3-8%, far outside the
        suite's 1e-4 tolerance -- so a conditional-pricing ORE comparison
        genuinely checks this term."""
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
        """The finding this file exists for, pinned as a fact rather than
        left as a hazard.

        At t=0 the variance term's (1 - exp(-2*a*t)) factor is exactly
        zero, so the term contributes nothing and corrupting it is
        undetectable at ANY tolerance a t=0 test could reasonably use.
        Most of this suite's ORE swaption comparisons price at t=0.

        Asserting the mutation is INVISIBLE (rather than visible) is
        deliberate: this records why t=0 comparisons cannot be the whole
        story. If a future change made this term matter at t=0, this test
        fails and the suite's coverage assumptions need revisiting.
        """
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
        """Makes explicit the dependency measured against the real suite:
        with the variance term deleted, `tests/test_european_swaption.py`
        fails exactly one test -- the conditional-pricing one at a later
        evaluation date -- and its other 130 pass.

        This reconstructs that test's own comparison point (t=1, priced
        conditional on a simulated short rate) and asserts the mutation is
        caught there. If conditional-pricing coverage were ever removed,
        deleting a whole term of the core bond-price formula would leave
        the suite green.
        """
        baseline = _swaption_price(1.0, 0.03)
        with _mutate_hull_white_A(scale_variance=0.0):
            mutated = _swaption_price(1.0, 0.03)
        assert abs(mutated - baseline) / abs(baseline) > SUITE_RTOL, (
            "deleting the variance term is no longer detectable even at t>0; "
            "the suite would not catch a wrong A(t,T) at all"
        )


class TestForwardTermIsActuallyChecked:
    """The B(t,T)*f(0,t) term, which carries today's curve into A(t,T)."""

    @pytest.mark.parametrize("t_eval,short_rate", [(0.0, FLAT_RATE), (1.0, 0.03), (2.0, 0.04)])
    def test_dropping_forward_term_is_always_visible(self, t_eval, short_rate):
        """Unlike the variance term, this one is checked everywhere --
        including at t=0, where deleting it changes the price by >100%."""
        baseline = _swaption_price(t_eval, short_rate)
        with _mutate_hull_white_A(drop_forward=True):
            mutated = _swaption_price(t_eval, short_rate)
        assert abs(mutated - baseline) / abs(baseline) > SUITE_RTOL * 100


class TestNonFlatCurvesAgainstORE:
    """A(t,T) against `ORE.HullWhite.discountBond` on curve shapes the rest
    of the suite never uses.

    On a flat curve f(0,t) equals the flat rate for every t and every
    interpolation scheme agrees, so a flat-curve comparison cannot detect
    an error in either. These shapes make both terms bite.
    """

    SHAPES = {
        "upward": [0.010, 0.015, 0.020, 0.030, 0.040, 0.050],
        "inverted": [0.050, 0.045, 0.040, 0.030, 0.025, 0.020],
        "humped": [0.020, 0.030, 0.040, 0.045, 0.035, 0.025],
        "negative": [-0.005, -0.002, 0.000, 0.005, 0.010, 0.015],
    }
    PILLARS = [0.0, 1.0, 2.0, 5.0, 10.0, 30.0]

    def _ore_curve(self, rates):
        """An ORE ZeroCurve on the same pillars with the same linear
        zero-rate interpolation this engine's ZeroCurveConfig uses, so the
        two describe the identical curve (verified: ORE's `discount(T)`
        matches exp(-r(T)*T) to 0.0 at and between pillars)."""
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
        """Control: the two libraries agree on the CURVE before any model
        is involved. Without this, a later disagreement on discountBond
        could be a curve-construction mismatch in this test rather than
        anything about A(t,T)."""
        rates = self.SHAPES[shape]
        curve = self._ore_curve(rates)
        for T in [0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 30.0]:
            expected = float(np.exp(-np.interp(T, self.PILLARS, rates) * T))
            np.testing.assert_allclose(curve.discount(T), expected, rtol=1e-12)

    @pytest.mark.parametrize("shape", list(SHAPES))
    @pytest.mark.parametrize("t,T", [(1.5, 4.5), (2.5, 7.5), (3.5, 8.5), (0.5, 3.5)])
    def test_discount_bond_matches_ore_off_pillar(self, shape, t, T):
        """The real check: away from pillar kinks, this engine reproduces
        ORE's own HullWhite.discountBond to ~1e-8 on every curve shape,
        including a negative-rate curve."""
        rates = self.SHAPES[shape]
        hw = ORE.HullWhite(self._ore_curve(rates), HW_A, HW_SIGMA)
        for r in (0.03, 0.0, -0.01):
            mine = self._engine_bond(rates, t, T, r)
            np.testing.assert_allclose(mine, hw.discountBond(t, T, r), rtol=1e-6)

    def test_pillar_times_differ_by_the_interpolation_kink(self):
        """Pins the boundary found while writing this file.

        Under linear zero-rate interpolation f(0,t) has a genuine kink at
        each pillar -- its left and right derivatives differ -- so the
        instantaneous forward is undefined exactly there and this engine
        and ORE resolve it differently, disagreeing by ~1e-2 AT a pillar
        while agreeing to ~1e-8 a half-year away on either side.

        Neither library is wrong; the quantity is genuinely ambiguous.
        This is pinned rather than fixed so the boundary is recorded, and
        so that a future move to a smooth interpolation (which would
        remove the kink and make both sides agree) fails here and prompts
        tightening `test_discount_bond_matches_ore_off_pillar` to cover
        pillar times too.
        """
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
