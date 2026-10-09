"""
`engine.calibration.basket`: co-terminal basket construction and the LGM closed-form
swaption price (`price_lgm_swaption`) that calibration uses.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.calibration.basket import (
    build_coterminal_basket,
    bachelier_swaption_price,
    price_lgm_swaption,
)
from engine.market_data.curves import ZeroCurve
from engine.models.lgm import Sigma, bond_price, numeraire

TODAY = ORE.Date(30, 7, 2026)


@pytest.fixture(autouse=True)
def _set_eval_date():
    ORE.Settings.instance().evaluationDate = TODAY


FLAT_CURVE = ZeroCurve.flat(0.03, [0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0])


class TestBuildCoterminalBasket:
    def test_produces_one_target_per_exercise_time(self):
        targets = build_coterminal_basket(
            exercise_times=[1.0, 2.0, 3.0, 4.0], final_maturity_time=5.0,
            notional=1_000_000.0, payer=True, market_vols=[0.008, 0.009, 0.0095, 0.0098],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        assert len(targets) == 4

    def test_expiry_times_are_increasing_and_before_final_maturity(self):
        targets = build_coterminal_basket(
            exercise_times=[1.0, 2.0, 3.0, 4.0], final_maturity_time=5.0,
            notional=1_000_000.0, payer=True, market_vols=[0.008] * 4,
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        expiries = [t.expiry_time for t in targets]
        assert expiries == sorted(expiries)
        for t in targets:
            assert t.expiry_time < 5.0
            assert t.fixed_cashflow_times[-1] == pytest.approx(5.0, abs=0.02)

    def test_coterminal_swaps_have_shrinking_cashflow_count(self):
        """Every instrument runs to the same final maturity, so a later exercise has fewer
        fixed cashflows (annual fixed leg here)."""
        targets = build_coterminal_basket(
            exercise_times=[1.0, 2.0, 3.0, 4.0], final_maturity_time=5.0,
            notional=1_000_000.0, payer=True, market_vols=[0.008] * 4,
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        counts = [len(t.fixed_cashflow_times) for t in targets]
        assert counts == sorted(counts, reverse=True)

    def test_atm_strike_matches_par_rate_identity(self):
        """`forward_rate` is the par rate (P_start - P_end) / annuity from the same curve."""
        targets = build_coterminal_basket(
            exercise_times=[2.0], final_maturity_time=7.0,
            notional=1_000_000.0, payer=True, market_vols=[0.009],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        t = targets[0]
        from engine.market_data.curves import discount
        P_start = float(discount(FLAT_CURVE, t.accrual_start_time))
        P_end = float(discount(FLAT_CURVE, t.fixed_cashflow_times[-1]))
        annuity = float(jnp.sum(jnp.asarray(t.fixed_accrual_fractions) * discount(FLAT_CURVE, jnp.asarray(t.fixed_cashflow_times))))
        expected_par = (P_start - P_end) / annuity
        assert t.forward_rate == pytest.approx(expected_par, rel=1e-8)


class TestPriceLgmSwaptionMatchesNumeraireDeflatedMonteCarlo:
    """The closed form against Monte Carlo on LGM's exact terminal distribution
    x(T0) ~ N(0, zeta(T0)) (QuantExt `IrLgm1fStateProcess::variance`), 2e6 paths, within 6
    standard errors. Payoffs are deflated by `engine.models.lgm.numeraire`; discounting by
    P(0,T0) is wrong under LGM's measure."""

    @pytest.mark.parametrize("payer", [True, False])
    def test_matches_mc_within_stderr(self, payer):
        a, sigma_flat = 0.03, 0.01
        sigma = Sigma.flat(sigma_flat)
        targets = build_coterminal_basket(
            exercise_times=[5.0], final_maturity_time=10.0,
            notional=1_000_000.0, payer=payer, market_vols=[0.009],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        target = targets[0]
        closed_form = float(price_lgm_swaption(FLAT_CURVE, a, sigma, target))

        T0 = target.expiry_time
        all_times = np.concatenate([
            np.asarray(target.fixed_cashflow_times), np.asarray(target.fixed_cashflow_times)[-1:],
            [target.accrual_start_time],
        ])
        all_amounts = np.concatenate([
            np.asarray(target.fixed_cashflow_amounts), [target.notional, -target.notional],
        ])

        zetaT0 = float(sigma_flat ** 2 * T0)
        rng = np.random.default_rng(123)
        N = 2_000_000
        x = rng.normal(0.0, np.sqrt(zetaT0), size=N)

        P_T0_Ti = np.asarray(bond_price(FLAT_CURVE, a, sigma, T0, jnp.asarray(all_times)[None, :], jnp.asarray(x)[:, None]))
        signed_value = np.sum(P_T0_Ti * all_amounts[None, :], axis=1)
        payoff = np.maximum(-signed_value, 0.0) if payer else np.maximum(signed_value, 0.0)
        N_T0 = np.asarray(numeraire(FLAT_CURVE, a, sigma, T0, jnp.asarray(x)))
        mc_price = np.mean(payoff / N_T0)
        mc_stderr = np.std(payoff / N_T0) / np.sqrt(N)

        assert closed_form == pytest.approx(mc_price, abs=6 * mc_stderr)

    def test_matches_mc_with_piecewise_sigma(self):
        """The same check with a piecewise `Sigma`."""
        a = 0.03
        sigma = Sigma(times=jnp.array([2.0, 6.0]), values=jnp.array([0.006, 0.012, 0.009]))
        targets = build_coterminal_basket(
            exercise_times=[5.0], final_maturity_time=10.0,
            notional=1_000_000.0, payer=True, market_vols=[0.009],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        target = targets[0]
        closed_form = float(price_lgm_swaption(FLAT_CURVE, a, sigma, target))

        T0 = target.expiry_time
        all_times = np.concatenate([
            np.asarray(target.fixed_cashflow_times), np.asarray(target.fixed_cashflow_times)[-1:],
            [target.accrual_start_time],
        ])
        all_amounts = np.concatenate([
            np.asarray(target.fixed_cashflow_amounts), [target.notional, -target.notional],
        ])

        from engine.models.lgm import zeta as zeta_fn
        zetaT0 = float(zeta_fn(sigma, jnp.array(T0)))
        rng = np.random.default_rng(99)
        N = 2_000_000
        x = rng.normal(0.0, np.sqrt(zetaT0), size=N)

        P_T0_Ti = np.asarray(bond_price(FLAT_CURVE, a, sigma, T0, jnp.asarray(all_times)[None, :], jnp.asarray(x)[:, None]))
        signed_value = np.sum(P_T0_Ti * all_amounts[None, :], axis=1)
        payoff = np.maximum(-signed_value, 0.0)
        N_T0 = np.asarray(numeraire(FLAT_CURVE, a, sigma, T0, jnp.asarray(x)))
        mc_price = np.mean(payoff / N_T0)
        mc_stderr = np.std(payoff / N_T0) / np.sqrt(N)

        assert closed_form == pytest.approx(mc_price, abs=6 * mc_stderr)


class TestPriceLgmSwaptionSanity:
    def test_higher_sigma_gives_higher_price(self):
        """Monotone increasing in sigma, payer and receiver."""
        targets = build_coterminal_basket(
            exercise_times=[3.0], final_maturity_time=8.0,
            notional=1_000_000.0, payer=True, market_vols=[0.009],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        target = targets[0]
        low = float(price_lgm_swaption(FLAT_CURVE, 0.03, 0.005, target))
        high = float(price_lgm_swaption(FLAT_CURVE, 0.03, 0.02, target))
        assert high > low

    def test_atm_payer_and_receiver_have_equal_price(self):
        """At an exactly ATM strike payer and receiver have equal value (parity)."""
        targets_payer = build_coterminal_basket(
            exercise_times=[3.0], final_maturity_time=8.0,
            notional=1_000_000.0, payer=True, market_vols=[0.009],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        targets_receiver = build_coterminal_basket(
            exercise_times=[3.0], final_maturity_time=8.0,
            notional=1_000_000.0, payer=False, market_vols=[0.009],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        payer_price = float(price_lgm_swaption(FLAT_CURVE, 0.03, 0.01, targets_payer[0]))
        receiver_price = float(price_lgm_swaption(FLAT_CURVE, 0.03, 0.01, targets_receiver[0]))
        assert payer_price == pytest.approx(receiver_price, rel=1e-6)

    def test_gradient_wrt_sigma_is_finite_and_positive(self):
        """d price / d sigma is positive and finite (calibration and Vega rely on it)."""
        targets = build_coterminal_basket(
            exercise_times=[3.0], final_maturity_time=8.0,
            notional=1_000_000.0, payer=True, market_vols=[0.009],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        target = targets[0]

        def f(sigma_val):
            return price_lgm_swaption(FLAT_CURVE, 0.03, sigma_val, target)

        grad = jax.grad(f)(0.01)
        assert jnp.isfinite(grad)
        assert float(grad) > 0.0

    def test_gradient_wrt_sigma_matches_finite_difference_value(self):
        """The autodiff sigma-gradient matches finite differences in value. It must include
        the path through the exercise boundary x* (`_bisect_xstar`, a custom_jvp); a plain
        bisection has zero gradient and drops the d(price)/d(x*) * d(x*)/d(sigma) term
        while keeping the right sign."""
        targets = build_coterminal_basket(
            exercise_times=[3.0], final_maturity_time=8.0,
            notional=1_000_000.0, payer=True, market_vols=[0.009],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        target = targets[0]

        def f(sigma_val):
            return price_lgm_swaption(FLAT_CURVE, 0.03, sigma_val, target)

        sigma0 = 0.01
        grad = float(jax.grad(f)(sigma0))
        eps = 1e-6
        fd = (float(f(sigma0 + eps)) - float(f(sigma0 - eps))) / (2 * eps)
        assert grad == pytest.approx(fd, rel=1e-4)

    def test_gradient_wrt_piecewise_sigma_bucket_matches_finite_difference(self):
        """The same value check with a multi-bucket `Sigma` nested in the params tuple (needs
        both the custom_jvp and `Sigma`'s pytree registration)."""
        targets = build_coterminal_basket(
            exercise_times=[3.0], final_maturity_time=8.0,
            notional=1_000_000.0, payer=True, market_vols=[0.009],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        target = targets[0]

        def f(s1):
            sigma = Sigma(times=jnp.array([1.5]), values=jnp.array([0.008, s1]))
            return price_lgm_swaption(FLAT_CURVE, 0.03, sigma, target)

        s1_0 = 0.012
        grad = float(jax.grad(f)(s1_0))
        eps = 1e-6
        fd = (float(f(s1_0 + eps)) - float(f(s1_0 - eps))) / (2 * eps)
        assert grad == pytest.approx(fd, rel=1e-4)


class TestBachelierSwaptionPrice:
    def test_zero_vol_gives_zero_price_at_atm(self):
        targets = build_coterminal_basket(
            exercise_times=[3.0], final_maturity_time=8.0,
            notional=1_000_000.0, payer=True, market_vols=[0.0],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        price = float(bachelier_swaption_price(targets[0], FLAT_CURVE))
        assert price == pytest.approx(0.0, abs=1e-6)

    def test_higher_vol_gives_higher_price(self):
        targets_low = build_coterminal_basket(
            exercise_times=[3.0], final_maturity_time=8.0,
            notional=1_000_000.0, payer=True, market_vols=[0.005],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        targets_high = build_coterminal_basket(
            exercise_times=[3.0], final_maturity_time=8.0,
            notional=1_000_000.0, payer=True, market_vols=[0.02],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        low = float(bachelier_swaption_price(targets_low[0], FLAT_CURVE))
        high = float(bachelier_swaption_price(targets_high[0], FLAT_CURVE))
        assert high > low

    def test_matches_atm_straddle_half_closed_form(self):
        """ATM, the Bachelier formula reduces to
        `notional * annuity * vol * sqrt(T0 / (2*pi))`."""
        targets = build_coterminal_basket(
            exercise_times=[3.0], final_maturity_time=8.0,
            notional=1_000_000.0, payer=True, market_vols=[0.01],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        target = targets[0]
        from engine.market_data.curves import discount
        annuity = float(jnp.sum(jnp.asarray(target.fixed_accrual_fractions) * discount(FLAT_CURVE, jnp.asarray(target.fixed_cashflow_times))))
        expected = target.notional * annuity * target.market_vol * np.sqrt(target.expiry_time / (2 * np.pi))
        price = float(bachelier_swaption_price(target, FLAT_CURVE))
        assert price == pytest.approx(expected, rel=1e-8)
