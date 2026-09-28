"""
`engine.calibration.lgm.calibrate_lgm_sigma`: bootstrap of a piecewise `Sigma` to a
co-terminal basket of market swaption vols.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.models.hull_white import ZeroCurve
from engine.models.lgm import Sigma
from engine.calibration.basket import build_coterminal_basket, price_lgm_swaption
from engine.calibration.lgm import calibrate_lgm_sigma

TODAY = ORE.Date(30, 7, 2026)


@pytest.fixture(autouse=True)
def _set_eval_date():
    ORE.Settings.instance().evaluationDate = TODAY


FLAT_CURVE = ZeroCurve.flat(0.03, [0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0])


def _basket(exercise_times, final_maturity, vols, payer=True, notional=1_000_000.0):
    return build_coterminal_basket(
        exercise_times=exercise_times, final_maturity_time=final_maturity,
        notional=notional, payer=payer, market_vols=vols,
        zero_curve=FLAT_CURVE, evaluation_date=TODAY,
    )


class TestCalibrateLgmSigmaExactBootstrapReprice:
    """A bootstrap reprices every instrument exactly (one new bucket per instrument), unlike
    a joint fit whose RMSE is generally nonzero."""

    def test_single_instrument_basket(self):
        targets = _basket([3.0], 8.0, [0.01])
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
        assert result.rmse < 1e-4
        np.testing.assert_allclose(np.asarray(result.model_prices), np.asarray(result.market_prices), atol=1e-3)

    def test_four_instrument_basket(self):
        targets = _basket([1.0, 2.0, 3.0, 4.0], 5.0, [0.008, 0.009, 0.0095, 0.0098])
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
        assert result.rmse < 1e-4
        np.testing.assert_allclose(np.asarray(result.model_prices), np.asarray(result.market_prices), atol=1e-3)

    def test_flat_market_vol_gives_calibrated_sigma_in_a_sane_range(self):
        """A flat market vol does not bootstrap to a flat model sigma (each bucket's
        instrument has a shorter tenor and expiry), so only boundedness relative to the
        market vol is checked (a sign or scale bug breaks it)."""
        targets = _basket([1.0, 2.0, 3.0, 4.0], 5.0, [0.01, 0.01, 0.01, 0.01])
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
        values = np.asarray(result.sigma.values)
        assert np.all(values > 0.005)
        assert np.all(values < 0.02)

    def test_sigma_bucket_structure_matches_ore_convention(self):
        """len(values) == len(times) + 1 == basket size; times are the basket expiries
        without the last, as in ORE's `LgmBuilder` (expiries[:-1])."""
        exercise_times = [1.0, 2.0, 3.0, 4.0]
        targets = _basket(exercise_times, 5.0, [0.008, 0.009, 0.0095, 0.0098])
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
        assert result.sigma.values.shape[0] == 4
        assert result.sigma.times.shape[0] == 3
        expected_times = [t.expiry_time for t in targets[:-1]]
        np.testing.assert_allclose(np.asarray(result.sigma.times), np.asarray(expected_times), rtol=1e-8)


class TestCalibrateLgmSigmaSanity:
    def test_higher_market_vols_give_higher_calibrated_sigma(self):
        low_targets = _basket([1.0, 2.0, 3.0], 5.0, [0.006, 0.006, 0.006])
        high_targets = _basket([1.0, 2.0, 3.0], 5.0, [0.015, 0.015, 0.015])
        low_result = calibrate_lgm_sigma(low_targets, FLAT_CURVE, a=0.03)
        high_result = calibrate_lgm_sigma(high_targets, FLAT_CURVE, a=0.03)
        assert np.all(np.asarray(high_result.sigma.values) > np.asarray(low_result.sigma.values))

    def test_calibrated_sigma_prices_a_held_out_swaption_reasonably(self):
        """Out of sample: a swaption not in the basket prices within a plausible range of its
        Bachelier market price (not exactly)."""
        calib_targets = _basket([1.0, 3.0, 5.0], 6.0, [0.008, 0.0095, 0.0105])
        result = calibrate_lgm_sigma(calib_targets, FLAT_CURVE, a=0.03)

        held_out = _basket([2.0], 6.0, [0.009])[0]
        model_price = float(price_lgm_swaption(FLAT_CURVE, 0.03, result.sigma, held_out))
        from engine.calibration.basket import bachelier_swaption_price
        market_price = float(bachelier_swaption_price(held_out, FLAT_CURVE))
        assert model_price == pytest.approx(market_price, rel=0.15)

    def test_requires_increasing_expiry_order(self):
        targets = _basket([1.0, 2.0, 3.0], 5.0, [0.008, 0.009, 0.0095])
        targets_shuffled = [targets[1], targets[0], targets[2]]
        with pytest.raises(ValueError):
            calibrate_lgm_sigma(targets_shuffled, FLAT_CURVE, a=0.03)

    def test_single_bucket_matches_flat_sigma_calibration(self):
        """A one-instrument basket gives a flat `Sigma` (no breakpoints)."""
        targets = _basket([3.0], 8.0, [0.011])
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
        assert result.sigma.times.shape[0] == 0
        assert result.sigma.values.shape[0] == 1


class TestCalibrationResultGradientCorrectness:
    """The calibrated `Sigma` is differentiable in the market vols (Vega uses this through
    the implicit function theorem)."""

    @pytest.mark.slow
    def test_calibrated_sigma_gradient_wrt_market_vol_is_finite_and_positive(self):
        def calibrate_last_bucket(vol_last):
            targets = _basket([1.0, 2.0, 3.0], 5.0, [0.008, 0.009, float(vol_last)])
            result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
            return result.sigma.values[-1]

        # Finite differences: calibrate_lgm_sigma builds the basket with ORE objects on the
        # host, so it cannot be traced end to end with jax.grad.
        eps = 1e-5
        base = float(calibrate_last_bucket(0.0098))
        bumped = float(calibrate_last_bucket(0.0098 + eps))
        fd_grad = (bumped - base) / eps
        assert np.isfinite(fd_grad)
        assert fd_grad > 0.0


class TestUnattainableMarketVolIsRefused:
    """The bisection bracket is [1e-6, 0.20] (0.01bp to 2000bp). A market price outside
    what it can reach raises instead of returning the bracket end."""

    def test_market_vol_above_bracket_raises(self):
        targets = _basket([1.0, 2.0], 5.0, [0.008, 5.0])
        with pytest.raises(ValueError, match="not attainable"):
            calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)

    def test_realistic_basket_still_calibrates(self):
        targets = _basket([1.0, 2.0], 5.0, [0.008, 0.009])
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
        assert np.all(np.isfinite(np.asarray(result.sigma.values)))
