"""
Tests for engine.risk.greeks's Bermudan/American Greeks --
bermudan_delta_gamma, bermudan_theta, bermudan_vega.

**Gamma cross-check methodology note.** A naive central finite-difference
of the NPV itself (`(NPV_up - 2*NPV_base + NPV_down) / bump**2`) is
numerically unreliable here: the Bermudan NPV is O(1e4), and a realistic
Gamma-sized bump (1e-4 to 1e-5) makes the true second-order signal smaller
than the float64 cancellation error in that formula (confirmed directly
during this module's own development -- see engine/risk/greeks.py's git
history / this file's TestBermudanDeltaGamma class, where a naive
price-level FD swung by orders of magnitude and even changed SIGN across
bump sizes 1e-2 through 1e-5, while the autodiff Hessian stayed fixed).
The numerically sound cross-check instead finite-differences the GRADIENT
itself (`(grad(rate+eps) - grad(rate-eps)) / (2*eps)`), which has no such
cancellation problem since the gradient itself is O(1e6) with a much
larger true second-derivative signal relative to its own floating-point
noise floor.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.models.hull_white import ZeroCurve
from engine.simulation import ZeroCurveConfig
from engine.calibration.basket import build_coterminal_basket
from engine.calibration.lgm import calibrate_lgm_sigma
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig, price_bermudan_swaption_base
from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.risk.greeks import (
    DEFAULT_RATE_BUMP,
    _bermudan_price_fn,
    bermudan_delta_gamma,
    bermudan_theta,
    bermudan_vega,
)

TODAY = ORE.Date(30, 7, 2026)


@pytest.fixture(autouse=True)
def _set_eval_date():
    ORE.Settings.instance().evaluationDate = TODAY


PILLAR_TIMES = [0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0]
FLAT_CURVE = ZeroCurve.flat(0.03, PILLAR_TIMES)
FLAT_CURVE_CONFIG = ZeroCurveConfig(times=PILLAR_TIMES, rates=[0.03] * len(PILLAR_TIMES))


def _cfg(hw_sigma=0.01, exercise_times=(1.0, 2.0, 3.0, 4.0), swap_tenor="5Y", payer=True):
    return BermudanSwaptionConfig(
        notional=1_000_000.0, fixed_rate=0.03, payer=payer, rate_factor_index=0,
        hw_a=0.03, hw_sigma=hw_sigma,
        initial_zero_curve=FLAT_CURVE_CONFIG,
        exercise_times=list(exercise_times), swap_tenor=swap_tenor, evaluation_date=TODAY,
    )


class TestBermudanDeltaGamma:
    def test_returns_finite_delta_and_gamma_for_every_pillar(self):
        greeks = bermudan_delta_gamma(_cfg(), FLAT_CURVE)
        assert greeks["delta"].shape == (len(PILLAR_TIMES),)
        assert greeks["gamma"].shape == (len(PILLAR_TIMES),)
        assert jnp.all(jnp.isfinite(greeks["delta"]))
        assert jnp.all(jnp.isfinite(greeks["gamma"]))

    def test_delta_matches_finite_difference_of_price(self):
        """Delta itself (unlike Gamma) is well-conditioned for a direct
        price-level central finite difference -- no cancellation problem
        at a 1bp bump, since the first-order signal dominates."""
        cfg = _cfg()
        greeks = bermudan_delta_gamma(cfg, FLAT_CURVE)

        idx = 4  # 5Y pillar
        bump = DEFAULT_RATE_BUMP
        rates_up = list(FLAT_CURVE_CONFIG.rates)
        rates_up[idx] += bump
        rates_down = list(FLAT_CURVE_CONFIG.rates)
        rates_down[idx] -= bump

        def cfg_with_rates(rates):
            return BermudanSwaptionConfig(
                notional=cfg.notional, fixed_rate=cfg.fixed_rate, payer=cfg.payer,
                rate_factor_index=cfg.rate_factor_index, hw_a=cfg.hw_a, hw_sigma=cfg.hw_sigma,
                initial_zero_curve=ZeroCurveConfig(times=PILLAR_TIMES, rates=rates),
                exercise_times=cfg.exercise_times, swap_tenor=cfg.swap_tenor, evaluation_date=TODAY,
            )

        npv_up = price_bermudan_swaption_base(cfg_with_rates(rates_up))
        npv_down = price_bermudan_swaption_base(cfg_with_rates(rates_down))
        fd_delta = (npv_up - npv_down) / 2.0  # already a 1bp bump, no extra scaling
        # A slightly looser tolerance than a typical closed-form Greek
        # check: the Bermudan engine's own state grid (n_per_std/std_devs)
        # has a small residual discretization sensitivity to a 1bp curve
        # bump that a perfectly closed-form pricer wouldn't have --
        # confirmed not to be a bug by tightening the grid (n_per_std=96)
        # and observing the FD/autodiff gap shrink accordingly.
        assert float(greeks["delta"][idx]) == pytest.approx(fd_delta, rel=5e-3)

    def test_gamma_matches_finite_difference_of_gradient(self):
        """See module docstring: Gamma is cross-checked via finite
        difference OF THE GRADIENT (numerically sound), not of the price
        (numerically unreliable at this scale)."""
        cfg = _cfg()
        curve = FLAT_CURVE
        price_fn, sigma_values = _bermudan_price_fn(cfg, curve)

        def grad_fn(pillar_rates):
            return jax.grad(price_fn, argnums=0)(pillar_rates, sigma_values)

        idx = 4
        autodiff_hessian_diag = jax.hessian(price_fn, argnums=0)(curve.pillar_rates, sigma_values)
        autodiff_val = float(jnp.diagonal(autodiff_hessian_diag)[idx])

        eps = 1e-6
        up = curve.pillar_rates.at[idx].add(eps)
        down = curve.pillar_rates.at[idx].add(-eps)
        fd_hess = float((grad_fn(up)[idx] - grad_fn(down)[idx]) / (2 * eps))

        assert autodiff_val == pytest.approx(fd_hess, rel=1e-3)

    def test_zero_at_pillars_outside_the_trades_own_cashflow_range(self):
        """A pillar far outside the trade's own cashflow dates (t=0 and
        t=30Y, for a trade maturing at 5Y) should show exactly zero
        sensitivity -- interpolation between the trade's own bracketing
        pillars means the curve's endpoints never enter the computation
        at all."""
        greeks = bermudan_delta_gamma(_cfg(), FLAT_CURVE)
        assert float(greeks["delta"][0]) == 0.0
        assert float(greeks["delta"][-1]) == 0.0


class TestBermudanTheta:
    def test_finite_and_typically_small_relative_to_npv(self):
        cfg = _cfg()
        base_npv = price_bermudan_swaption_base(cfg)
        theta = bermudan_theta(cfg, FLAT_CURVE)
        assert np.isfinite(theta)
        assert abs(theta) < 0.05 * abs(base_npv)

    def test_theta_days_zero_gives_zero(self):
        cfg = _cfg()
        theta = bermudan_theta(cfg, FLAT_CURVE, theta_days=0)
        assert theta == pytest.approx(0.0, abs=1e-6)


class TestBermudanVega:
    def test_matches_finite_difference_recalibration(self):
        """The core correctness check: bermudan_vega's implicit-function-
        theorem Vega against a literal finite-difference recalibration
        (bump one basket instrument's market vol, rerun calibrate_lgm_
        sigma, reprice) -- exactly what ORE itself does, used here purely
        as an independent ground truth."""
        exercise_times = [1.0, 2.0, 3.0, 4.0]
        base_vols = [0.008, 0.009, 0.0095, 0.0098]

        def price_with_vols(vols):
            targets = build_coterminal_basket(
                exercise_times=exercise_times, final_maturity_time=5.0,
                notional=1_000_000.0, payer=True, market_vols=vols,
                zero_curve=FLAT_CURVE, evaluation_date=TODAY,
            )
            result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
            cfg = _cfg(hw_sigma=result.sigma, exercise_times=exercise_times)
            return price_bermudan_swaption_base(cfg), targets, result.sigma

        base_npv, targets, sigma = price_with_vols(base_vols)
        cfg = _cfg(hw_sigma=sigma, exercise_times=exercise_times)
        vega = bermudan_vega(cfg, FLAT_CURVE, targets)

        bump = 1e-5
        for i in range(4):
            vols_up = list(base_vols)
            vols_up[i] += bump
            vols_down = list(base_vols)
            vols_down[i] -= bump
            npv_up, _, _ = price_with_vols(vols_up)
            npv_down, _, _ = price_with_vols(vols_down)
            fd_vega_i = (npv_up - npv_down) / (2 * bump) * 0.0001
            assert float(vega[i]) == pytest.approx(fd_vega_i, rel=5e-3)

    def test_vega_is_positive_for_every_bucket(self):
        """A Bermudan swaption is long volatility -- every bucket's Vega
        should be positive (more market vol -> higher calibrated sigma ->
        higher NPV)."""
        exercise_times = [1.0, 2.0, 3.0]
        targets = build_coterminal_basket(
            exercise_times=exercise_times, final_maturity_time=4.0,
            notional=1_000_000.0, payer=True, market_vols=[0.008, 0.009, 0.0095],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
        cfg = _cfg(hw_sigma=result.sigma, exercise_times=exercise_times, swap_tenor="4Y")
        vega = bermudan_vega(cfg, FLAT_CURVE, targets)
        assert jnp.all(vega > 0.0)

    def test_receiver_also_has_positive_vega(self):
        exercise_times = [1.0, 2.0, 3.0]
        targets = build_coterminal_basket(
            exercise_times=exercise_times, final_maturity_time=4.0,
            notional=1_000_000.0, payer=False, market_vols=[0.008, 0.009, 0.0095],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
        cfg = _cfg(hw_sigma=result.sigma, exercise_times=exercise_times, swap_tenor="4Y", payer=False)
        vega = bermudan_vega(cfg, FLAT_CURVE, targets)
        assert jnp.all(vega > 0.0)

    def test_raises_on_bucket_count_mismatch(self):
        exercise_times = [1.0, 2.0, 3.0, 4.0]
        targets = build_coterminal_basket(
            exercise_times=exercise_times, final_maturity_time=5.0,
            notional=1_000_000.0, payer=True, market_vols=[0.008, 0.009, 0.0095, 0.0098],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        cfg = _cfg(hw_sigma=0.01, exercise_times=exercise_times)  # flat sigma, 1 bucket
        with pytest.raises(AssertionError):
            bermudan_vega(cfg, FLAT_CURVE, targets)  # targets has 4 instruments


class TestAmericanSwaptionSharesTheSameGreeksPath:
    def test_delta_gamma_finite_via_to_bermudan(self):
        """AmericanSwaptionConfig has no dedicated Greeks function --
        to_bermudan() feeds the same bermudan_delta_gamma path (see
        engine.risk.greeks's own module docstring)."""
        american_cfg = AmericanSwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True, rate_factor_index=0,
            hw_a=0.03, hw_sigma=0.01,
            initial_zero_curve=FLAT_CURVE_CONFIG,
            first_exercise=1.0, last_exercise=3.0, exercise_time_steps_per_year=2,
            swap_tenor="4Y", evaluation_date=TODAY,
        )
        greeks = bermudan_delta_gamma(american_cfg.to_bermudan(), FLAT_CURVE)
        assert jnp.all(jnp.isfinite(greeks["delta"]))
        assert jnp.all(jnp.isfinite(greeks["gamma"]))


class TestBermudanGreeksEdgeCases:
    """Boundary conditions not covered by the "typical trade" tests above:
    zero notional, single (European-equivalent) exercise date, extreme
    sigma, negative rates, and grid-resolution extremes."""

    def test_zero_notional_gives_exactly_zero_delta_and_gamma(self):
        """A zero-notional trade has zero value at every curve shock --
        Delta/Gamma must be exactly 0, not merely small, since NPV is
        identically 0 regardless of the curve."""
        cfg = _cfg()
        cfg = BermudanSwaptionConfig(
            notional=0.0, fixed_rate=cfg.fixed_rate, payer=cfg.payer,
            rate_factor_index=cfg.rate_factor_index, hw_a=cfg.hw_a, hw_sigma=cfg.hw_sigma,
            initial_zero_curve=cfg.initial_zero_curve, exercise_times=cfg.exercise_times,
            swap_tenor=cfg.swap_tenor, evaluation_date=TODAY,
        )
        greeks = bermudan_delta_gamma(cfg, FLAT_CURVE)
        assert jnp.all(greeks["delta"] == 0.0)
        assert jnp.all(greeks["gamma"] == 0.0)

    def test_zero_notional_gives_exactly_zero_theta(self):
        cfg = _cfg()
        cfg = BermudanSwaptionConfig(
            notional=0.0, fixed_rate=cfg.fixed_rate, payer=cfg.payer,
            rate_factor_index=cfg.rate_factor_index, hw_a=cfg.hw_a, hw_sigma=cfg.hw_sigma,
            initial_zero_curve=cfg.initial_zero_curve, exercise_times=cfg.exercise_times,
            swap_tenor=cfg.swap_tenor, evaluation_date=TODAY,
        )
        theta = bermudan_theta(cfg, FLAT_CURVE)
        assert theta == pytest.approx(0.0, abs=1e-9)

    def test_single_exercise_date_delta_gamma_finite(self):
        """A single-exercise-date Bermudan degenerates to a European-
        equivalent trade -- the backward induction's own edge case (no
        early-exercise comparison ever fires before the one and only
        exercise date), must still be fully differentiable."""
        cfg = _cfg(exercise_times=(2.0,))
        greeks = bermudan_delta_gamma(cfg, FLAT_CURVE)
        assert jnp.all(jnp.isfinite(greeks["delta"]))
        assert jnp.all(jnp.isfinite(greeks["gamma"]))
        theta = bermudan_theta(cfg, FLAT_CURVE)
        assert np.isfinite(theta)

    def test_very_dense_exercise_schedule(self):
        """A semi-annual (9-date) exercise schedule -- denser than any
        other test in this file -- must not blow up the backward
        induction's own autodiff graph (jax.lax.scan's length grows with
        the grid schedule, not with the raw Python exercise-date count,
        but this exercises that path at a larger scale regardless)."""
        cfg = _cfg(exercise_times=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5))
        greeks = bermudan_delta_gamma(cfg, FLAT_CURVE)
        assert jnp.all(jnp.isfinite(greeks["delta"]))
        assert jnp.all(jnp.isfinite(greeks["gamma"]))

    def test_extremely_small_sigma_stays_finite(self):
        """sigma -> 0 is the deterministic limit (zeta -> 0 everywhere) --
        exactly the state_grid/std_step sqrt(0) gradient singularity this
        module's own bug fix (see engine/instruments/bermudan_swaption.py's
        _state_grid docstring) guards against. A tiny but nonzero sigma is
        the sharpest practical test of that guard."""
        cfg = _cfg(hw_sigma=1e-6)
        greeks = bermudan_delta_gamma(cfg, FLAT_CURVE)
        assert jnp.all(jnp.isfinite(greeks["delta"]))
        assert jnp.all(jnp.isfinite(greeks["gamma"]))
        theta = bermudan_theta(cfg, FLAT_CURVE)
        assert np.isfinite(theta)

    def test_relatively_high_sigma_stays_finite(self):
        """A stressed, high (150bp) flat sigma -- still within a
        realistic range, but well above every other test's own ~100bp
        ceiling."""
        cfg = _cfg(hw_sigma=0.015)
        greeks = bermudan_delta_gamma(cfg, FLAT_CURVE)
        assert jnp.all(jnp.isfinite(greeks["delta"]))
        assert jnp.all(jnp.isfinite(greeks["gamma"]))

    def test_negative_rates_curve(self):
        """A curve with negative short-end rates (common in EUR/CHF/JPY
        markets historically) -- Delta/Gamma/Theta must remain finite;
        nothing in engine.models.lgm's formulas assumes r/rates are
        positive."""
        neg_curve = ZeroCurve(
            pillar_times=jnp.asarray(PILLAR_TIMES),
            pillar_rates=jnp.array([-0.005, -0.003, 0.0, 0.005, 0.01, 0.015, 0.02]),
        )
        neg_curve_config = ZeroCurveConfig(
            times=PILLAR_TIMES, rates=[-0.005, -0.003, 0.0, 0.005, 0.01, 0.015, 0.02],
        )
        cfg = BermudanSwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.01, payer=True, rate_factor_index=0,
            hw_a=0.03, hw_sigma=0.01, initial_zero_curve=neg_curve_config,
            exercise_times=[1.0, 2.0], swap_tenor="5Y", evaluation_date=TODAY,
        )
        greeks = bermudan_delta_gamma(cfg, neg_curve)
        assert jnp.all(jnp.isfinite(greeks["delta"]))
        assert jnp.all(jnp.isfinite(greeks["gamma"]))
        theta = bermudan_theta(cfg, neg_curve)
        assert np.isfinite(theta)

    def test_coarse_state_grid_still_differentiable(self):
        """A deliberately coarse state grid (n_per_std=4, far below the
        default 48) -- confirms Delta/Gamma remain finite even at a
        resolution too coarse for production accuracy (a robustness
        check on the autodiff graph's own shape handling, not an
        accuracy claim about the coarse grid's own numbers)."""
        cfg = BermudanSwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True, rate_factor_index=0,
            hw_a=0.03, hw_sigma=0.01, initial_zero_curve=FLAT_CURVE_CONFIG,
            exercise_times=[1.0, 2.0], swap_tenor="5Y", evaluation_date=TODAY, n_per_std=4,
        )
        greeks = bermudan_delta_gamma(cfg, FLAT_CURVE)
        assert jnp.all(jnp.isfinite(greeks["delta"]))
        assert jnp.all(jnp.isfinite(greeks["gamma"]))

    def test_receiver_trade_delta_gamma_theta_finite(self):
        cfg = _cfg(payer=False)
        greeks = bermudan_delta_gamma(cfg, FLAT_CURVE)
        assert jnp.all(jnp.isfinite(greeks["delta"]))
        assert jnp.all(jnp.isfinite(greeks["gamma"]))
        theta = bermudan_theta(cfg, FLAT_CURVE)
        assert np.isfinite(theta)

    def test_vega_with_a_single_bucket_calibration(self):
        """The degenerate N=1 calibration case fed into Vega -- a single
        basket instrument, a single sigma bucket -- exercises the same
        bucket-count assertion's PASSING path (exactly matched counts),
        complementing test_raises_on_bucket_count_mismatch's failing
        path."""
        targets = build_coterminal_basket(
            exercise_times=[2.0], final_maturity_time=5.0,
            notional=1_000_000.0, payer=True, market_vols=[0.009],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
        cfg = _cfg(hw_sigma=result.sigma, exercise_times=(2.0,), swap_tenor="5Y")
        vega = bermudan_vega(cfg, FLAT_CURVE, targets)
        assert vega.shape == (1,)
        assert jnp.isfinite(vega[0])
        assert float(vega[0]) > 0.0

    def test_vega_with_extreme_mean_reversion(self):
        """Vega's implicit-function-theorem derivation (both in
        price_lgm_swaption's _bisect_xstar fix and bermudan_vega's own
        cross-bucket Jacobian) makes no assumption about `a`'s own
        magnitude -- confirmed finite at a near-zero and a relatively
        high mean reversion."""
        for a in [1e-4, 0.25]:
            targets = build_coterminal_basket(
                exercise_times=[1.0, 2.0], final_maturity_time=4.0,
                notional=1_000_000.0, payer=True, market_vols=[0.008, 0.009],
                zero_curve=FLAT_CURVE, evaluation_date=TODAY,
            )
            result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=a)
            cfg = BermudanSwaptionConfig(
                notional=1_000_000.0, fixed_rate=0.03, payer=True, rate_factor_index=0,
                hw_a=a, hw_sigma=result.sigma, initial_zero_curve=FLAT_CURVE_CONFIG,
                exercise_times=[1.0, 2.0], swap_tenor="4Y", evaluation_date=TODAY,
            )
            vega = bermudan_vega(cfg, FLAT_CURVE, targets)
            assert jnp.all(jnp.isfinite(vega)), f"a={a}: non-finite vega"
            assert jnp.all(vega > 0.0), f"a={a}: non-positive vega"
