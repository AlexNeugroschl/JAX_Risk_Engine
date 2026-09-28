"""
`engine.risk.greeks` for swaps and European swaptions: Delta, Gamma, Theta.

1. Autodiff Delta/Gamma against literal bump-and-revalue with ORE's `SensitivityCube`
   formulas at 1bp. These are not identical: the forward-difference Delta also carries half
   the Gamma and higher terms, which the tolerances absorb.
2. ORE cross-checks by bumping a `FlatForward` curve and repricing a real
   `ORE.VanillaSwap` / `ORE.Swaption`.
3. Shapes, zero sensitivity past expiry, degenerate configurations.

Also the implicit-function-theorem gradient of `_solve_rstar` (bisection alone gives a
zero gradient).
"""
import dataclasses

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
    """Bump-and-revalue with ORE's `SensitivityCube` formulas: delta = NPV_up - NPV_base,
    gamma = NPV_up - 2*NPV_base + NPV_down."""
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
    """The two-curve price closure as a function of the discount rates only (forward curve
    fixed)."""
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
        """Flat extrapolation outside the pillar range (as `np.interp`)."""
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
        # The midpoint (t=2) is the average: linear interpolation.
        assert float(_zero_rate_at(curve, jnp.asarray(2.0))) == pytest.approx(0.03)

    def test_discount_at_matches_continuous_compounding(self):
        curve = ZeroCurve.flat(0.03, PILLAR_TIMES)
        from engine.risk.greeks import _discount_at
        disc = float(_discount_at(curve, jnp.asarray(5.0)))
        assert disc == pytest.approx(np.exp(-0.03 * 5.0), rel=1e-12)


# =============================================================================
# _solve_rstar gradient (implicit function theorem)
# =============================================================================
class TestSolveRstarGradientCorrectness:
    """Regression: bisection gave the right r* but a zero gradient (the comparison has no
    derivative). `jax.custom_jvp` applies the implicit function theorem. Tested on toy
    root-finds with known derivatives."""

    def test_linear_root_gradient_matches_analytic(self):
        """f(r, c) = c - r: r* = c and d(r*)/dc = 1 (naive autodiff gave 0)."""
        def f(r, c):
            return c - r

        def solve(c):
            return _solve_rstar(f, c, ())

        c0 = jnp.array(0.5)
        assert float(solve(c0)) == pytest.approx(0.5)
        grad = jax.grad(solve)(c0)
        assert float(grad) == pytest.approx(1.0, abs=1e-9)

    def test_cubic_root_gradient_and_hessian_match_analytic(self):
        """f(r, c) = c - r^3: r* = c^(1/3), with known first and second derivatives
        (`jax.hessian`, needed for Gamma)."""
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
        """Vector params (as the swaption passes `A_T0_Ti`) give a correctly shaped
        Jacobian."""
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
        """The custom JVP does not change the forward value."""
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
        """Pillars 10Y and 30Y, beyond a 2Y swap's last cashflow with another pillar in
        between, have zero Delta."""
        cfg = self._cfg(swap_tenor="2Y")
        disc_curve = ZeroCurve.flat(0.03, PILLAR_TIMES)  # last cashflow ~2Y, pillars go to 30Y
        fwd_curve = ZeroCurve.flat(0.035, PILLAR_TIMES)
        greeks = swap_delta_gamma(cfg, disc_curve, fwd_curve)
        # Pillars 3 (10Y) and 4 (30Y) are beyond the swap's interpolation support.
        assert float(greeks["discount_delta"][3]) == pytest.approx(0.0, abs=1e-6)
        assert float(greeks["discount_delta"][4]) == pytest.approx(0.0, abs=1e-6)

    def test_payer_receiver_deltas_are_negations(self):
        """A receiver's Greeks are the exact negation of the payer's."""
        disc_curve = ZeroCurve.flat(0.030, PILLAR_TIMES)
        fwd_curve = ZeroCurve.flat(0.035, PILLAR_TIMES)
        payer_greeks = swap_delta_gamma(self._cfg(payer=True), disc_curve, fwd_curve)
        receiver_greeks = swap_delta_gamma(self._cfg(payer=False), disc_curve, fwd_curve)
        for key in ("discount_delta", "forward_delta", "discount_gamma", "forward_gamma"):
            np.testing.assert_allclose(
                np.asarray(payer_greeks[key]), -np.asarray(receiver_greeks[key]), atol=1e-8,
            )

    def test_single_curve_discounting_deltas_sum_to_total(self):
        """With one curve for both roles, discount and forward deltas sum pillar by pillar
        to the sensitivity to that shared curve (checked by bumping it in both roles)."""
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
    """Autodiff Delta against ORE's own bump-and-revalue of a real `ORE.VanillaSwap`."""

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
        """A parallel 1bp bump on flat curves, so the comparison does not depend on how a
        non-flat ORE curve interpolates."""
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
        """Theta = NPV(t+1d) - NPV(t) + cashflow(t, t+1d], computed by hand from
        `_swap_price_fn` at both dates. The later valuation is the same swap one day older,
        with its first coupon fixed at t's forecast."""
        from engine.instruments.swap import _build_ore_swap
        from engine.models.ore_builders import TIME_AXIS_DAY_COUNTER

        cfg = self._cfg()
        disc_curve = ZeroCurve.flat(0.030, PILLAR_TIMES)
        fwd_curve = ZeroCurve.flat(0.035, PILLAR_TIMES)

        theta = swap_theta(cfg, disc_curve, fwd_curve)

        base_price_fn = _swap_price_fn(cfg, disc_curve, fwd_curve)
        base = float(base_price_fn(disc_curve.pillar_rates, fwd_curve.pillar_rates))
        theta_date = ORE.TARGET().advance(TODAY, DEFAULT_THETA_DAYS, ORE.Days)
        coupon = ORE.as_floating_rate_coupon(_build_ore_swap(cfg).floatingLeg()[0])
        assert coupon.fixingDate() == TODAY
        t_start, t_end = (TIME_AXIS_DAY_COUNTER.yearFraction(TODAY, d)
                          for d in (coupon.accrualStartDate(), coupon.accrualEndDate()))
        forecast = (np.exp(0.035 * (t_end - t_start)) - 1.0) / coupon.accrualPeriod()
        theta_cfg = dataclasses.replace(cfg, evaluation_date=theta_date, fixings={TODAY: forecast})
        assert theta_cfg.maturity_date == cfg.maturity_date
        theta_price_fn = _swap_price_fn(theta_cfg, disc_curve, fwd_curve)
        theta_npv = float(theta_price_fn(disc_curve.pillar_rates, fwd_curve.pillar_rates))

        assert theta == pytest.approx(theta_npv - base, abs=1e-6)

    def test_finite_and_reasonable_magnitude(self):
        """One day of Theta on a $1MM 5Y swap is finite and small against its NPV."""
        cfg = self._cfg()
        disc_curve = ZeroCurve.flat(0.030, PILLAR_TIMES)
        fwd_curve = ZeroCurve.flat(0.035, PILLAR_TIMES)
        theta = swap_theta(cfg, disc_curve, fwd_curve)
        assert np.isfinite(theta)
        assert abs(theta) < 10_000.0

    def test_zero_theta_days_is_a_noop(self):
        """theta_days=0 gives exactly 0."""
        cfg = self._cfg()
        disc_curve = ZeroCurve.flat(0.030, PILLAR_TIMES)
        fwd_curve = ZeroCurve.flat(0.035, PILLAR_TIMES)
        theta = swap_theta(cfg, disc_curve, fwd_curve, theta_days=0)
        assert theta == pytest.approx(0.0, abs=1e-6)


# =============================================================================
# European swaption: the JAX A(t,T) equals the NumPy compute_hw_A
# =============================================================================
class TestComputeHwAJaxMatchesNumpy:
    def test_matches_compute_hw_A_across_grid(self):
        """Greeks' `hull_white.A` and the pricer's NumPy wrapper `compute_hw_A` agree
        exactly (they are the same function)."""
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
# European swaption: the t=0 price function equals the main pricer
# =============================================================================
class TestSwaptionPriceFnMatchesMainPricer:
    def test_matches_price_swaptions_at_t0(self):
        """`_swaption_price_fn` (the differentiable t=0 path) equals `price_swaptions` at
        t=0 with r(0) at the curve's short end."""
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
        """Deep ITM: large Delta, small Gamma, both matching finite differences."""
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

    @pytest.mark.slow
    def test_finite_for_various_hw_parameters(self):
        """Delta/Gamma stay finite across hw_a/hw_sigma values (no hidden singularity at
        small hw_a)."""
        curve = ZeroCurve.flat(0.03, PILLAR_TIMES)
        for hw_a in [0.01, 0.03, 0.1, 0.3]:
            for hw_sigma in [0.005, 0.01, 0.02]:
                cfg = self._cfg(curve, hw_a=hw_a, hw_sigma=hw_sigma)
                greeks = swaption_delta_gamma(cfg, curve)
                assert bool(jnp.all(jnp.isfinite(greeks["delta"])))
                assert bool(jnp.all(jnp.isfinite(greeks["gamma"])))


class TestSwaptionDeltaGammaAgainstORE:
    """Autodiff Delta against `ORE.Swaption` with `ORE.JamshidianSwaptionEngine` on a bumped
    `ORE.HullWhite` model (parallel bump, as for swaps)."""

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
        # The same option one day older: its exercise date is unchanged.
        theta_cfg = dataclasses.replace(cfg, evaluation_date=theta_date)
        assert theta_cfg.exercise_date == cfg.exercise_date
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
# Precision (PrecisionConfig.risk)
# =============================================================================
class TestGreeksPrecisionDtype:
    """Greeks functions work in their curve's dtype: a float32 curve keeps them float32.
    (The `price_portfolio` path is covered by
    tests/test_portfolio_entrypoint.py::TestPricePortfolioPrecision.)"""

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


class TestSwapGreeksHonourTheTradesOwnConventions:
    """Regression: `_swap_price_fn` and `swap_theta` once copied SwapConfig field by field
    and dropped `accrual_day_count`, computing an ACT/ACT swap's Greeks on ACT/365."""

    def _cfg(self, **overrides):
        fields = dict(
            notional=1_000_000.0, fixed_rate=0.032, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="5Y", evaluation_date=TODAY,
        )
        fields.update(overrides)
        return SwapConfig(**fields)

    def test_parallel_discount_delta_matches_bumped_reprice(self):
        from engine.instruments.swap import price_swaps
        from engine.portfolio.request import _flat_curve_cube, derive_maturity_pillars

        cfg = self._cfg(accrual_day_count="ACT/ACT (ICMA)")
        pillars = np.asarray(derive_maturity_pillars([cfg], TODAY))

        def npv(disc_shift):
            disc = ZeroCurveConfig(times=PILLAR_TIMES, rates=[0.03 + disc_shift] * len(PILLAR_TIMES))
            fwd = ZeroCurveConfig(times=PILLAR_TIMES, rates=[0.035] * len(PILLAR_TIMES))
            return float(price_swaps(_flat_curve_cube(disc, fwd, pillars, TODAY), pillars, [cfg])[0, 0, 0])

        bumped = (npv(DEFAULT_RATE_BUMP) - npv(-DEFAULT_RATE_BUMP)) / 2.0
        greeks = swap_delta_gamma(cfg, ZeroCurve.flat(0.03, PILLAR_TIMES), ZeroCurve.flat(0.035, PILLAR_TIMES))
        assert float(jnp.sum(greeks["discount_delta"])) == pytest.approx(bumped, rel=1e-6)

    def test_theta_period_flow_includes_floating_coupons(self):
        """A floating coupon paid inside the Theta window is added back like a fixed one.
        With 1Y fixed / 6M floating, the window to the first floating payment holds only
        that coupon."""
        from engine.instruments.swap import _build_ore_swap
        from engine.models.ore_builders import TIME_AXIS_DAY_COUNTER
        from engine.risk.greeks import _swap_cashflows_in_period

        cfg = self._cfg()
        coupon = ORE.as_floating_rate_coupon(_build_ore_swap(cfg).floatingLeg()[0])
        fwd_curve = ZeroCurve.flat(0.035, PILLAR_TIMES)

        flow = _swap_cashflows_in_period(cfg, TODAY, coupon.date(), fwd_curve)

        t_start = TIME_AXIS_DAY_COUNTER.yearFraction(TODAY, coupon.accrualStartDate())
        t_end = TIME_AXIS_DAY_COUNTER.yearFraction(TODAY, coupon.accrualEndDate())
        expected = cfg.notional * (np.exp(-0.035 * t_start) / np.exp(-0.035 * t_end) - 1.0)
        assert flow == pytest.approx(expected, rel=1e-10)
