"""
Tests for engine.risk.greeks -- Delta, Gamma, and Theta for interest rate
swaps and European swaptions.

Methodology, matching the rest of this test suite:
1. Direct correctness checks against literal finite-difference bump-and-
   revalue (the SAME quantity ORE's own SensitivityAnalysis computes, just
   without ORE's own finite-difference truncation error) -- since this
   module's whole design is "autodiff gives ORE's numbers exactly," the
   sharpest test is comparing autodiff output against the literal bump ORE
   would perform, at a small-enough step size that residual disagreement
   is attributable only to finite-difference truncation, not a bug.
2. Direct ORE cross-checks where a real ORE object supports it (bumping a
   FlatForward curve and re-pricing a real ORE.VanillaSwap/ORE.Swaption).
3. Structural/edge-case tests (shape, zero-sensitivity-past-expiry,
   single-pillar curves, degenerate configs).

`_solve_rstar`'s gradient-correctness bug (see
engine/instruments/european_swaption.py's docstring) is exercised directly
here too, since Greeks were the first thing in this codebase to actually
differentiate through it.
"""
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.instruments.swap import SwapConfig
from engine.instruments.european_swaption import SwaptionConfig, _solve_rstar
from engine.simulation.market_model import ZeroCurveConfig
from engine.risk.greeks import (
    DEFAULT_RATE_BUMP,
    DEFAULT_THETA_DAYS,
    ZeroCurve,
    _swap_price_fn,
    _swaption_price_fn,
    swap_delta_gamma,
    swap_theta,
    swaption_delta_gamma,
    swaption_theta,
)
from engine.simulation.demo_scenarios import EVAL_DATE

TODAY = EVAL_DATE
PILLAR_TIMES = [1.0, 2.0, 5.0, 10.0, 30.0]


def _finite_difference_delta_gamma(price_fn, pillar_rates, bump=DEFAULT_RATE_BUMP):
    """Literal bump-and-revalue, matching ORE's own SensitivityCube
    formula exactly (delta = NPV_up - NPV_base, gamma = NPV_up - 2*NPV_base
    + NPV_down) -- the ground truth this module's autodiff-based
    swap_delta_gamma/swaption_delta_gamma are checked against."""
    base = float(price_fn(pillar_rates))
    n = len(pillar_rates)
    fd_delta = np.zeros(n)
    fd_gamma = np.zeros(n)
    for j in range(n):
        up = pillar_rates.at[j].add(bump)
        down = pillar_rates.at[j].add(-bump)
        p_up = float(price_fn(up))
        p_down = float(price_fn(down))
        fd_delta[j] = p_up - base
        fd_gamma[j] = p_up - 2 * base + p_down
    return fd_delta, fd_gamma


def _swap_price_fn_single_curve(cfg, disc_curve, fwd_curve):
    """Wraps _swap_price_fn's 2-argument closure into a 1-argument
    function of ONLY disc_curve.pillar_rates (fwd_curve held fixed) --
    what the finite-difference helper above needs when isolating one
    curve's own sensitivity."""
    inner = _swap_price_fn(cfg, disc_curve, fwd_curve)
    return lambda disc_rates: inner(disc_rates, fwd_curve.pillar_rates)


def _swap_price_fn_fwd_only(cfg, disc_curve, fwd_curve):
    inner = _swap_price_fn(cfg, disc_curve, fwd_curve)
    return lambda fwd_rates: inner(disc_curve.pillar_rates, fwd_rates)


# =============================================================================
# ZeroCurve / interpolation building blocks
# =============================================================================
class TestZeroCurve:
    def test_flat_curve_interpolates_to_constant(self):
        curve = ZeroCurve.flat(0.03, PILLAR_TIMES)
        from engine.risk.greeks import _zero_rate_at
        for t in [0.0, 0.5, 1.0, 3.7, 10.0, 15.0, 30.0, 50.0]:
            assert float(_zero_rate_at(curve, jnp.asarray(t))) == pytest.approx(0.03, abs=1e-12)

    def test_flat_extrapolation_beyond_pillars(self):
        """jnp.interp flat-extrapolates outside [min(pillar_times),
        max(pillar_times)] -- matching np.interp's own default and this
        codebase's other curve-interpolation helpers
        (european_swaption._initial_log_discount)."""
        curve = ZeroCurve(
            pillar_times=jnp.array([1.0, 5.0, 10.0]),
            pillar_rates=jnp.array([0.02, 0.03, 0.04]),
        )
        from engine.risk.greeks import _zero_rate_at
        assert float(_zero_rate_at(curve, jnp.asarray(0.1))) == pytest.approx(0.02)
        assert float(_zero_rate_at(curve, jnp.asarray(50.0))) == pytest.approx(0.04)

    def test_linear_interpolation_between_pillars(self):
        curve = ZeroCurve(
            pillar_times=jnp.array([1.0, 3.0]),
            pillar_rates=jnp.array([0.02, 0.04]),
        )
        from engine.risk.greeks import _zero_rate_at
        # Midpoint (t=2) should be exactly the average -- linear interp.
        assert float(_zero_rate_at(curve, jnp.asarray(2.0))) == pytest.approx(0.03)

    def test_discount_at_matches_continuous_compounding(self):
        curve = ZeroCurve.flat(0.03, PILLAR_TIMES)
        from engine.risk.greeks import _discount_at
        disc = float(_discount_at(curve, jnp.asarray(5.0)))
        assert disc == pytest.approx(np.exp(-0.03 * 5.0), rel=1e-12)


# =============================================================================
# _solve_rstar gradient correctness (the bug this module surfaced and fixed)
# =============================================================================
class TestSolveRstarGradientCorrectness:
    """Regression coverage for a real bug found while building this
    module: _solve_rstar's bisection produced the correct forward value
    but a silently WRONG gradient (naive autodiff through a comparison-
    based bisection loop gives zero gradient everywhere). Fixed via
    jax.custom_jvp implementing the implicit function theorem. These
    tests exercise _solve_rstar directly, independent of the swaption
    pricer, with a toy root-find whose analytic derivatives are known
    exactly."""

    def test_linear_root_gradient_matches_analytic(self):
        """f(r, c) = c - r has root r* = c, so d(r*)/dc = 1.0 exactly --
        the simplest possible case, and the one that exposed the bug in
        the first place (naive autodiff gave 0.0 here)."""
        def f(r, c):
            return c - r

        def solve(c):
            return _solve_rstar(f, c, ())

        c0 = jnp.array(0.5)
        assert float(solve(c0)) == pytest.approx(0.5)
        grad = jax.grad(solve)(c0)
        assert float(grad) == pytest.approx(1.0, abs=1e-9)

    def test_cubic_root_gradient_and_hessian_match_analytic(self):
        """f(r, c) = c - r^3 has root r* = c^(1/3), with known closed-form
        first AND second derivatives -- exercises jax.hessian (needed for
        Gamma), not just jax.grad (Delta), through the custom_jvp rule."""
        def f(r, c):
            return c - r ** 3

        def solve(c):
            return _solve_rstar(f, c, ())

        c0 = jnp.array(0.5)
        val = solve(c0)
        assert float(val) == pytest.approx(0.5 ** (1.0 / 3.0), rel=1e-9)

        analytic_grad = 1.0 / (3.0 * c0 ** (2.0 / 3.0))
        grad = jax.grad(solve)(c0)
        assert float(grad) == pytest.approx(float(analytic_grad), rel=1e-6)

        analytic_hess = (-2.0 / 9.0) * c0 ** (-5.0 / 3.0)
        hess = jax.hessian(solve)(c0)
        assert float(hess) == pytest.approx(float(analytic_hess), rel=1e-4)

    def test_pytree_params_gradient_matches_finite_difference(self):
        """The real swaption use case passes a pytree (A_T0_Ti array) as
        params, not a scalar -- confirms the custom_jvp rule handles
        vector-valued params and returns a correctly-shaped Jacobian."""
        B = jnp.array([0.5, 1.0, 1.5, 2.0, 0.1])
        amounts = jnp.array([0.02, 0.02, 0.02, 1.02, -1.0])

        def f(r, A):
            prices = A * jnp.exp(-B * r)
            return jnp.sum(prices * amounts)

        def solve(A):
            return _solve_rstar(f, A, ())

        A0 = jnp.array([0.99, 0.97, 0.95, 0.90, 0.999])
        base = float(solve(A0))
        analytic_jac = jax.jacobian(solve)(A0)

        eps = 1e-6
        for i in range(len(A0)):
            bumped = A0.at[i].add(eps)
            fd = (float(solve(bumped)) - base) / eps
            assert float(analytic_jac[i]) == pytest.approx(fd, rel=1e-3, abs=1e-6)

    def test_forward_value_unaffected_by_gradient_fix(self):
        """The custom_jvp wrapper must be a pure numerical no-op on the
        forward VALUE -- confirms the fix didn't silently change what
        price_swaptions itself computes (see test_european_swaption.py
        for the full pre-existing regression suite, unaffected by this
        change; this is a narrower, direct spot-check)."""
        def f(r, c):
            return c - r
        val = _solve_rstar(f, jnp.array(0.37), ())
        assert float(val) == pytest.approx(0.37, abs=1e-9)


# =============================================================================
# SWAP DELTA / GAMMA
# =============================================================================
class TestSwapDeltaGamma:
    def _cfg(self, **overrides):
        defaults = dict(
            notional=1_000_000.0, fixed_rate=0.032, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="5Y", evaluation_date=TODAY,
        )
        defaults.update(overrides)
        return SwapConfig(**defaults)

    def test_matches_finite_difference_discount_curve(self):
        cfg = self._cfg()
        disc_curve = ZeroCurve.flat(0.030, PILLAR_TIMES)
        fwd_curve = ZeroCurve.flat(0.035, PILLAR_TIMES)
        greeks = swap_delta_gamma(cfg, disc_curve, fwd_curve)

        price_fn = _swap_price_fn_single_curve(cfg, disc_curve, fwd_curve)
        fd_delta, fd_gamma = _finite_difference_delta_gamma(price_fn, disc_curve.pillar_rates)

        np.testing.assert_allclose(
            np.asarray(greeks["discount_delta"]), fd_delta, atol=1.0,
            err_msg="discount delta should match finite-difference to within FD truncation error",
        )
        np.testing.assert_allclose(
            np.asarray(greeks["discount_gamma"]), fd_gamma, atol=1e-3,
        )

    def test_matches_finite_difference_forward_curve(self):
        cfg = self._cfg()
        disc_curve = ZeroCurve.flat(0.030, PILLAR_TIMES)
        fwd_curve = ZeroCurve.flat(0.035, PILLAR_TIMES)
        greeks = swap_delta_gamma(cfg, disc_curve, fwd_curve)

        price_fn = _swap_price_fn_fwd_only(cfg, disc_curve, fwd_curve)
        fd_delta, fd_gamma = _finite_difference_delta_gamma(price_fn, fwd_curve.pillar_rates)

        np.testing.assert_allclose(
            np.asarray(greeks["forward_delta"]), fd_delta, atol=5.0,
        )
        np.testing.assert_allclose(
            np.asarray(greeks["forward_gamma"]), fd_gamma, atol=1e-2,
        )

    def test_output_shape(self):
        cfg = self._cfg()
        disc_curve = ZeroCurve.flat(0.03, PILLAR_TIMES)
        fwd_curve = ZeroCurve.flat(0.035, PILLAR_TIMES)
        greeks = swap_delta_gamma(cfg, disc_curve, fwd_curve)
        for key in ("discount_delta", "discount_gamma", "forward_delta", "forward_gamma"):
            assert greeks[key].shape == (len(PILLAR_TIMES),)

    def test_zero_beyond_swap_maturity(self):
        """A pillar far beyond the swap's own 5Y maturity should have
        essentially zero Delta -- no cashflow interpolates from that far
        out (jnp.interp's flat-extrapolation at the near end means the
        LAST pillar at 30Y, well past a 5Y swap, still has some support if
        it's the flat-extrapolation anchor -- but a pillar strictly
        between the swap's own maturity and the far end, isolated by a
        neighboring pillar on both sides, should have ~zero sensitivity)."""
        cfg = self._cfg(swap_tenor="2Y")
        disc_curve = ZeroCurve.flat(0.03, PILLAR_TIMES)  # last cashflow ~2Y, pillars go to 30Y
        fwd_curve = ZeroCurve.flat(0.035, PILLAR_TIMES)
        greeks = swap_delta_gamma(cfg, disc_curve, fwd_curve)
        # pillar index 3 (10Y) and 4 (30Y) are both well past a 2Y swap's
        # last cashflow and bounded by neighboring pillars on the near
        # side -- interpolation support for any real cashflow time doesn't
        # reach them.
        assert float(greeks["discount_delta"][3]) == pytest.approx(0.0, abs=1e-6)
        assert float(greeks["discount_delta"][4]) == pytest.approx(0.0, abs=1e-6)

    def test_payer_receiver_deltas_are_negations(self):
        """A receiver swap is the exact negation of the same payer swap
        (see swap.py's own payer/receiver sign convention) -- so its
        Greeks must be exact negations too, a model-independent identity
        check that doesn't rely on the finite-difference tolerance at
        all."""
        disc_curve = ZeroCurve.flat(0.030, PILLAR_TIMES)
        fwd_curve = ZeroCurve.flat(0.035, PILLAR_TIMES)
        payer_greeks = swap_delta_gamma(self._cfg(payer=True), disc_curve, fwd_curve)
        receiver_greeks = swap_delta_gamma(self._cfg(payer=False), disc_curve, fwd_curve)
        for key in ("discount_delta", "forward_delta", "discount_gamma", "forward_gamma"):
            np.testing.assert_allclose(
                np.asarray(payer_greeks[key]), -np.asarray(receiver_greeks[key]), atol=1e-8,
            )

    def test_single_curve_discounting_deltas_sum_to_total(self):
        """When the same ZeroCurve object is used for both discount_curve
        and forward_curve (single-curve discounting), the two independent
        per-curve deltas this function returns should sum, pillar by
        pillar, to the total sensitivity to that one shared curve --
        confirmed against a direct finite-difference bump of the SHARED
        curve (both discount and forward rates bumped together at once)."""
        cfg = self._cfg()
        shared_curve = ZeroCurve.flat(0.032, PILLAR_TIMES)
        greeks = swap_delta_gamma(cfg, shared_curve, shared_curve)
        combined_delta = np.asarray(greeks["discount_delta"]) + np.asarray(greeks["forward_delta"])

        def price_fn_shared(rates):
            inner = _swap_price_fn(cfg, shared_curve, shared_curve)
            return inner(rates, rates)

        fd_delta, _ = _finite_difference_delta_gamma(price_fn_shared, shared_curve.pillar_rates)
        np.testing.assert_allclose(combined_delta, fd_delta, atol=5.0)


class TestSwapDeltaGammaAgainstORE:
    """Direct cross-check: bump a real ORE.FlatForward curve by the same
    1bp, reprice a real ORE.VanillaSwap two ways, and confirm this
    module's autodiff Delta matches ORE's own bump-and-revalue number
    (not just this module's own finite-difference helper)."""

    def _reference_ore_swap_npv(self, disc_rate: float, fwd_rate: float, notional: float, fixed_rate: float) -> float:
        ORE.Settings.instance().evaluationDate = TODAY
        dc = ORE.Actual365Fixed()
        fwd_curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, fwd_rate, dc))
        disc_curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, disc_rate, dc))
        idx = ORE.IborIndex(
            "SimIndex", ORE.Period(6, ORE.Months), 2,
            ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False,
            dc, fwd_curve,
        )
        swap = ORE.MakeVanillaSwap(
            ORE.Period("5Y"), idx, fixed_rate,
            nominal=notional,
            swapType=ORE.VanillaSwap.Payer,
            discountingTermStructure=disc_curve,
            fixedLegDayCount=dc,
            floatingLegDayCount=dc,
        )
        swap.setPricingEngine(ORE.DiscountingSwapEngine(disc_curve))
        return swap.NPV()

    def test_parallel_delta_matches_ore_bump_and_revalue(self):
        """A PARALLEL 1bp bump (every pillar simultaneously) isolates the
        aggregate curve-level sensitivity, avoiding any dependence on this
        module's own triangular per-pillar interpolation shape agreeing
        exactly with a real, non-flat ORE curve's own interpolation --
        both sides use the identical FlatForward (flat = interpolation-
        shape-independent), so this is a clean, assumption-free
        cross-check."""
        notional, fixed_rate = 1_000_000.0, 0.032
        disc_rate, fwd_rate = 0.030, 0.035
        bump = DEFAULT_RATE_BUMP

        base = self._reference_ore_swap_npv(disc_rate, fwd_rate, notional, fixed_rate)
        disc_up = self._reference_ore_swap_npv(disc_rate + bump, fwd_rate, notional, fixed_rate)
        ore_parallel_disc_delta = disc_up - base

        cfg = SwapConfig(
            notional=notional, fixed_rate=fixed_rate, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="5Y", evaluation_date=TODAY,
        )
        disc_curve = ZeroCurve.flat(disc_rate, PILLAR_TIMES)
        fwd_curve = ZeroCurve.flat(fwd_rate, PILLAR_TIMES)
        greeks = swap_delta_gamma(cfg, disc_curve, fwd_curve)
        our_parallel_disc_delta = float(jnp.sum(greeks["discount_delta"]))

        assert our_parallel_disc_delta == pytest.approx(ore_parallel_disc_delta, rel=0.02)


# =============================================================================
# SWAP THETA
# =============================================================================
class TestSwapTheta:
    def _cfg(self, **overrides):
        defaults = dict(
            notional=1_000_000.0, fixed_rate=0.032, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="5Y", evaluation_date=TODAY,
        )
        defaults.update(overrides)
        return SwapConfig(**defaults)

    def test_matches_manual_reprice_difference(self):
        """Theta = NPV(t+1d) - NPV(t) + cashflow(t, t+1d) -- confirms
        swap_theta's output matches this definition computed by hand from
        the same building blocks (_swap_price_fn at two evaluation
        dates)."""
        cfg = self._cfg()
        disc_curve = ZeroCurve.flat(0.030, PILLAR_TIMES)
        fwd_curve = ZeroCurve.flat(0.035, PILLAR_TIMES)

        theta = swap_theta(cfg, disc_curve, fwd_curve)

        base_price_fn = _swap_price_fn(cfg, disc_curve, fwd_curve)
        base = float(base_price_fn(disc_curve.pillar_rates, fwd_curve.pillar_rates))
        theta_date = ORE.TARGET().advance(TODAY, DEFAULT_THETA_DAYS, ORE.Days)
        theta_cfg = self._cfg(evaluation_date=theta_date)
        theta_price_fn = _swap_price_fn(theta_cfg, disc_curve, fwd_curve)
        theta_npv = float(theta_price_fn(disc_curve.pillar_rates, fwd_curve.pillar_rates))

        assert theta == pytest.approx(theta_npv - base, abs=1e-6)

    def test_finite_and_reasonable_magnitude(self):
        """Theta for a 1-day roll on a $1MM notional 5Y swap should be a
        small number relative to the swap's own NPV -- not a NaN, not a
        wildly large blowup (a coarse sanity bound, not a tight
        cross-check)."""
        cfg = self._cfg()
        disc_curve = ZeroCurve.flat(0.030, PILLAR_TIMES)
        fwd_curve = ZeroCurve.flat(0.035, PILLAR_TIMES)
        theta = swap_theta(cfg, disc_curve, fwd_curve)
        assert np.isfinite(theta)
        assert abs(theta) < 10_000.0

    def test_zero_theta_days_is_a_noop(self):
        """Rolling forward by 0 days should reproduce the base NPV exactly
        (no cashflow window, no curve roll) -- Theta should come out
        exactly 0."""
        cfg = self._cfg()
        disc_curve = ZeroCurve.flat(0.030, PILLAR_TIMES)
        fwd_curve = ZeroCurve.flat(0.035, PILLAR_TIMES)
        theta = swap_theta(cfg, disc_curve, fwd_curve, theta_days=0)
        assert theta == pytest.approx(0.0, abs=1e-6)


# =============================================================================
# EUROPEAN SWAPTION: JAX-native A(t,T) matches the NumPy compute_hw_A
# =============================================================================
class TestComputeHwAJaxMatchesNumpy:
    def test_matches_compute_hw_A_across_grid(self):
        """`engine.risk.greeks` now imports its A(t,T) directly from
        `engine.models.hull_white` -- the SAME shared implementation
        `european_swaption.compute_hw_A` wraps for its own NumPy-facing
        callers (no more separately-maintained JAX twin of the formula).
        This test confirms both paths -- Greeks' JAX-native call and the
        main pricer's NumPy-facing wrapper -- agree exactly across a
        spread of (t, T, a, sigma) combinations, which they must, since
        they are now literally the same function underneath."""
        from engine.risk.greeks import ZeroCurve as GreeksZeroCurve, _hw_A
        from engine.instruments.european_swaption import compute_hw_A

        pillar_times = [0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        rates = [0.01, 0.02, 0.025, 0.03, 0.032, 0.035]
        curve = GreeksZeroCurve(
            pillar_times=jnp.asarray(pillar_times, dtype=jnp.float64),
            pillar_rates=jnp.asarray(rates, dtype=jnp.float64),
        )

        t_vals = jnp.array([0.0, 0.5, 1.5, 3.0, 7.0])
        T_vals = jnp.array([2.0, 3.0, 6.0, 8.0, 15.0])

        for a in [0.01, 0.03, 0.1]:
            for sigma in [0.005, 0.01, 0.02]:
                jax_A = _hw_A(curve, t_vals, T_vals, a, sigma)
                np_A = compute_hw_A(
                    np.array(pillar_times), np.array(rates),
                    np.array(t_vals), np.array(T_vals), a, sigma,
                )
                np.testing.assert_allclose(np.asarray(jax_A), np_A, rtol=1e-10)


# =============================================================================
# EUROPEAN SWAPTION: t=0 price matches the main pricer exactly
# =============================================================================
class TestSwaptionPriceFnMatchesMainPricer:
    def test_matches_price_swaptions_at_t0(self):
        """_swaption_price_fn is a from-scratch reimplementation of
        _price_one_swaption's t=0 path (needed to make it differentiable
        w.r.t. today's curve) -- it must reproduce price_swaptions'
        actual t=0 output exactly, conditioned on r(0) equal to the
        curve's own short end (the standard 'no shock' reference state)."""
        from engine.instruments.european_swaption import price_swaptions

        pillar_times = [1.0, 2.0, 5.0, 10.0, 30.0]
        r0 = 0.03
        curve = ZeroCurve.flat(r0, pillar_times)
        cfg = SwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
            hw_a=0.03, hw_sigma=0.01,
            initial_zero_curve=ZeroCurveConfig(times=pillar_times, rates=[r0] * len(pillar_times)),
            swap_tenor="5Y", forward_start=ORE.Period(3, ORE.Years), evaluation_date=TODAY,
        )

        hw_paths = jnp.array([[[r0]]])
        step_times = jnp.array([0.0])
        real_npv = float(price_swaptions(hw_paths, step_times, [cfg])[0, 0, 0])

        my_price_fn = _swaption_price_fn(cfg, curve)
        my_npv = float(my_price_fn(curve.pillar_rates))

        assert my_npv == pytest.approx(real_npv, rel=1e-8)

    def test_matches_for_receiver_swaption(self):
        from engine.instruments.european_swaption import price_swaptions

        pillar_times = [1.0, 2.0, 5.0, 10.0, 30.0]
        r0 = 0.03
        curve = ZeroCurve.flat(r0, pillar_times)
        cfg = SwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.028, payer=False, rate_factor_index=0,
            hw_a=0.03, hw_sigma=0.01,
            initial_zero_curve=ZeroCurveConfig(times=pillar_times, rates=[r0] * len(pillar_times)),
            swap_tenor="5Y", forward_start=ORE.Period(3, ORE.Years), evaluation_date=TODAY,
        )
        hw_paths = jnp.array([[[r0]]])
        step_times = jnp.array([0.0])
        real_npv = float(price_swaptions(hw_paths, step_times, [cfg])[0, 0, 0])
        my_npv = float(_swaption_price_fn(cfg, curve)(curve.pillar_rates))
        assert my_npv == pytest.approx(real_npv, rel=1e-8)


# =============================================================================
# EUROPEAN SWAPTION: DELTA / GAMMA
# =============================================================================
class TestSwaptionDeltaGamma:
    def _cfg(self, curve, **overrides):
        defaults = dict(
            notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
            hw_a=0.03, hw_sigma=0.01,
            initial_zero_curve=ZeroCurveConfig(
                times=[float(t) for t in curve.pillar_times],
                rates=[float(r) for r in curve.pillar_rates],
            ),
            swap_tenor="5Y", forward_start=ORE.Period(3, ORE.Years), evaluation_date=TODAY,
        )
        defaults.update(overrides)
        return SwaptionConfig(**defaults)

    def test_matches_finite_difference(self):
        curve = ZeroCurve.flat(0.03, PILLAR_TIMES)
        cfg = self._cfg(curve)
        greeks = swaption_delta_gamma(cfg, curve)

        price_fn = _swaption_price_fn(cfg, curve)
        fd_delta, fd_gamma = _finite_difference_delta_gamma(price_fn, curve.pillar_rates)

        np.testing.assert_allclose(np.asarray(greeks["delta"]), fd_delta, atol=1.0)
        np.testing.assert_allclose(np.asarray(greeks["gamma"]), fd_gamma, atol=1e-3)

    def test_matches_finite_difference_receiver(self):
        curve = ZeroCurve.flat(0.03, PILLAR_TIMES)
        cfg = self._cfg(curve, payer=False, fixed_rate=0.028)
        greeks = swaption_delta_gamma(cfg, curve)
        price_fn = _swaption_price_fn(cfg, curve)
        fd_delta, fd_gamma = _finite_difference_delta_gamma(price_fn, curve.pillar_rates)
        np.testing.assert_allclose(np.asarray(greeks["delta"]), fd_delta, atol=1.0)
        np.testing.assert_allclose(np.asarray(greeks["gamma"]), fd_gamma, atol=1e-3)

    def test_matches_finite_difference_deep_itm(self):
        """A deep-ITM swaption has a near-deterministic payoff -- Delta
        should be large and Gamma should be small (little optionality
        left), and both should still match finite-difference closely."""
        curve = ZeroCurve.flat(0.03, PILLAR_TIMES)
        cfg = self._cfg(curve, fixed_rate=0.01)  # deep ITM payer
        greeks = swaption_delta_gamma(cfg, curve)
        price_fn = _swaption_price_fn(cfg, curve)
        fd_delta, fd_gamma = _finite_difference_delta_gamma(price_fn, curve.pillar_rates)
        np.testing.assert_allclose(np.asarray(greeks["delta"]), fd_delta, atol=1.0)
        np.testing.assert_allclose(np.asarray(greeks["gamma"]), fd_gamma, atol=1e-3)

    def test_matches_finite_difference_deep_otm(self):
        curve = ZeroCurve.flat(0.03, PILLAR_TIMES)
        cfg = self._cfg(curve, fixed_rate=0.10)  # deep OTM payer
        greeks = swaption_delta_gamma(cfg, curve)
        price_fn = _swaption_price_fn(cfg, curve)
        fd_delta, fd_gamma = _finite_difference_delta_gamma(price_fn, curve.pillar_rates)
        np.testing.assert_allclose(np.asarray(greeks["delta"]), fd_delta, atol=0.5)
        np.testing.assert_allclose(np.asarray(greeks["gamma"]), fd_gamma, atol=1e-3)

    def test_output_shape(self):
        curve = ZeroCurve.flat(0.03, PILLAR_TIMES)
        cfg = self._cfg(curve)
        greeks = swaption_delta_gamma(cfg, curve)
        assert greeks["delta"].shape == (len(PILLAR_TIMES),)
        assert greeks["gamma"].shape == (len(PILLAR_TIMES),)

    def test_finite_for_various_hw_parameters(self):
        """Delta/Gamma should stay finite across a spread of hw_a/hw_sigma
        combinations -- guards against a hidden singularity (e.g. a
        divide-by-zero at small hw_a, matching the kind of edge case
        engine/simulation.py had to guard against for the same B(t,T)-
        style formula)."""
        curve = ZeroCurve.flat(0.03, PILLAR_TIMES)
        for hw_a in [0.01, 0.03, 0.1, 0.3]:
            for hw_sigma in [0.005, 0.01, 0.02]:
                cfg = self._cfg(curve, hw_a=hw_a, hw_sigma=hw_sigma)
                greeks = swaption_delta_gamma(cfg, curve)
                assert bool(jnp.all(jnp.isfinite(greeks["delta"])))
                assert bool(jnp.all(jnp.isfinite(greeks["gamma"])))


class TestSwaptionDeltaGammaAgainstORE:
    """Direct cross-check against a real ORE.Swaption priced with
    ORE.JamshidianSwaptionEngine under a bumped ORE.HullWhite model,
    isolating the aggregate parallel-curve sensitivity the same way
    TestSwapDeltaGammaAgainstORE does for swaps."""

    def _reference_ore_swaption_npv(self, flat_rate: float, notional: float, fixed_rate: float, payer: bool) -> float:
        ORE.Settings.instance().evaluationDate = TODAY
        dc = ORE.Actual365Fixed()
        curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, flat_rate, dc))
        idx = ORE.IborIndex(
            "SimIndex", ORE.Period(6, ORE.Months), 2,
            ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False,
            dc, curve,
        )
        swap_type = ORE.VanillaSwap.Payer if payer else ORE.VanillaSwap.Receiver
        underlying = ORE.MakeVanillaSwap(
            ORE.Period("5Y"), idx, fixed_rate,
            nominal=notional,
            swapType=swap_type,
            forwardStart=ORE.Period(3, ORE.Years),
            fixedLegDayCount=dc,
            floatingLegDayCount=dc,
        )
        exercise_date = ORE.TARGET().advance(
            ORE.TARGET().advance(TODAY, ORE.Period(3, ORE.Years)), 2, ORE.Days
        )
        exercise = ORE.EuropeanExercise(exercise_date)
        swaption = ORE.Swaption(underlying, exercise)
        model = ORE.HullWhite(curve, 0.03, 0.01)
        engine = ORE.JamshidianSwaptionEngine(model)
        swaption.setPricingEngine(engine)
        return swaption.NPV()

    def test_parallel_delta_matches_ore_bump_and_revalue(self):
        notional, fixed_rate = 1_000_000.0, 0.030
        flat_rate = 0.03
        bump = DEFAULT_RATE_BUMP

        base = self._reference_ore_swaption_npv(flat_rate, notional, fixed_rate, payer=True)
        up = self._reference_ore_swaption_npv(flat_rate + bump, notional, fixed_rate, payer=True)
        ore_parallel_delta = up - base

        curve = ZeroCurve.flat(flat_rate, PILLAR_TIMES)
        cfg = SwaptionConfig(
            notional=notional, fixed_rate=fixed_rate, payer=True, rate_factor_index=0,
            hw_a=0.03, hw_sigma=0.01,
            initial_zero_curve=ZeroCurveConfig(times=PILLAR_TIMES, rates=[flat_rate] * len(PILLAR_TIMES)),
            swap_tenor="5Y", forward_start=ORE.Period(3, ORE.Years), evaluation_date=TODAY,
        )
        greeks = swaption_delta_gamma(cfg, curve)
        our_parallel_delta = float(jnp.sum(greeks["delta"]))

        assert our_parallel_delta == pytest.approx(ore_parallel_delta, rel=0.05)


# =============================================================================
# EUROPEAN SWAPTION: THETA
# =============================================================================
class TestSwaptionTheta:
    def _cfg(self, curve, **overrides):
        defaults = dict(
            notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
            hw_a=0.03, hw_sigma=0.01,
            initial_zero_curve=ZeroCurveConfig(
                times=[float(t) for t in curve.pillar_times],
                rates=[float(r) for r in curve.pillar_rates],
            ),
            swap_tenor="5Y", forward_start=ORE.Period(3, ORE.Years), evaluation_date=TODAY,
        )
        defaults.update(overrides)
        return SwaptionConfig(**defaults)

    def test_matches_manual_reprice_difference(self):
        curve = ZeroCurve.flat(0.03, PILLAR_TIMES)
        cfg = self._cfg(curve)
        theta = swaption_theta(cfg, curve)

        base_price_fn = _swaption_price_fn(cfg, curve)
        base = float(base_price_fn(curve.pillar_rates))
        theta_date = ORE.TARGET().advance(TODAY, DEFAULT_THETA_DAYS, ORE.Days)
        theta_cfg = self._cfg(curve, evaluation_date=theta_date)
        theta_price_fn = _swaption_price_fn(theta_cfg, curve)
        theta_npv = float(theta_price_fn(curve.pillar_rates))

        assert theta == pytest.approx(theta_npv - base, abs=1e-6)

    def test_finite_and_reasonable_magnitude(self):
        curve = ZeroCurve.flat(0.03, PILLAR_TIMES)
        cfg = self._cfg(curve)
        theta = swaption_theta(cfg, curve)
        assert np.isfinite(theta)
        assert abs(theta) < 10_000.0

    def test_zero_theta_days_is_a_noop(self):
        curve = ZeroCurve.flat(0.03, PILLAR_TIMES)
        cfg = self._cfg(curve)
        theta = swaption_theta(cfg, curve, theta_days=0)
        assert theta == pytest.approx(0.0, abs=1e-6)


# =============================================================================
# PRECISION (PrecisionConfig.risk) -- direct dtype checks
# =============================================================================
class TestGreeksPrecisionDtype:
    """engine.portfolio.request._compute_all_greeks hands each Greeks
    function a `curve: ZeroCurve` built at PrecisionConfig.risk's dtype;
    this module's own closures must then derive their working dtype from
    that curve (see this module's docstring on why no new parameter is
    needed on the public entry points) rather than silently upcasting back
    to float64 -- these tests call the Greeks functions directly, at a
    curve built with dtype=jnp.float32, independent of the price_portfolio
    integration path (that path is covered by
    tests/test_portfolio_entrypoint.py::TestPricePortfolioPrecision)."""

    def test_swap_delta_gamma_float32_curve_stays_float32(self):
        disc_curve = ZeroCurve.flat(0.03, PILLAR_TIMES, dtype=jnp.float32)
        fwd_curve = ZeroCurve.flat(0.035, PILLAR_TIMES, dtype=jnp.float32)
        cfg = SwapConfig(
            notional=1_000_000.0, fixed_rate=0.032, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="5Y", evaluation_date=TODAY,
        )
        greeks = swap_delta_gamma(cfg, disc_curve, fwd_curve)
        for key, val in greeks.items():
            assert jnp.asarray(val).dtype == jnp.float32, f"{key} not float32"
        assert bool(jnp.all(jnp.isfinite(jnp.asarray(greeks["discount_delta"]))))

    def test_swaption_delta_gamma_float32_curve_stays_float32(self):
        curve = ZeroCurve.flat(0.03, PILLAR_TIMES, dtype=jnp.float32)
        cfg = SwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
            hw_a=0.03, hw_sigma=0.01,
            initial_zero_curve=ZeroCurveConfig(times=PILLAR_TIMES, rates=[0.03] * len(PILLAR_TIMES)),
            swap_tenor="5Y", forward_start=ORE.Period(3, ORE.Years), evaluation_date=TODAY,
        )
        greeks = swaption_delta_gamma(cfg, curve)
        assert greeks["delta"].dtype == jnp.float32
        assert greeks["gamma"].dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(greeks["delta"])))

    def test_swaption_float32_and_float64_deltas_numerically_close(self):
        curve32 = ZeroCurve.flat(0.03, PILLAR_TIMES, dtype=jnp.float32)
        curve64 = ZeroCurve.flat(0.03, PILLAR_TIMES, dtype=jnp.float64)
        cfg = SwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
            hw_a=0.03, hw_sigma=0.01,
            initial_zero_curve=ZeroCurveConfig(times=PILLAR_TIMES, rates=[0.03] * len(PILLAR_TIMES)),
            swap_tenor="5Y", forward_start=ORE.Period(3, ORE.Years), evaluation_date=TODAY,
        )
        greeks32 = swaption_delta_gamma(cfg, curve32)
        greeks64 = swaption_delta_gamma(cfg, curve64)
        np.testing.assert_allclose(
            np.asarray(greeks32["delta"]), np.asarray(greeks64["delta"]), rtol=1e-3, atol=1e-2,
        )

    def test_swaption_theta_float32_curve_produces_finite_float(self):
        curve = ZeroCurve.flat(0.03, PILLAR_TIMES, dtype=jnp.float32)
        cfg = SwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
            hw_a=0.03, hw_sigma=0.01,
            initial_zero_curve=ZeroCurveConfig(times=PILLAR_TIMES, rates=[0.03] * len(PILLAR_TIMES)),
            swap_tenor="5Y", forward_start=ORE.Period(3, ORE.Years), evaluation_date=TODAY,
        )
        theta = swaption_theta(cfg, curve)
        assert np.isfinite(theta)
