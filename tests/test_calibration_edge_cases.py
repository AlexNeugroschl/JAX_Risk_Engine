"""
Calibration edge cases: degenerate baskets, extreme market inputs, sloped curves.

Regression: `build_coterminal_basket` once wrote a fractional tenor such as "0.750000Y", and
`ORE.Period("0.75Y")` silently parses as 0Y, building a zero- or wrong-length swap. Tenors
are now whole months (`ORE.Period("9M")`).
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.models.hull_white import ZeroCurve
from engine.calibration.basket import build_coterminal_basket, bachelier_swaption_price, price_lgm_swaption
from engine.calibration.lgm import calibrate_lgm_sigma

TODAY = ORE.Date(30, 7, 2026)


@pytest.fixture(autouse=True)
def _set_eval_date():
    ORE.Settings.instance().evaluationDate = TODAY


FLAT_CURVE = ZeroCurve.flat(0.03, [0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0])
SLOPED_CURVE = ZeroCurve(
    pillar_times=jnp.array([0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0]),
    pillar_rates=jnp.array([0.020, 0.025, 0.028, 0.030, 0.032, 0.035, 0.038]),
)
INVERTED_CURVE = ZeroCurve(
    pillar_times=jnp.array([0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0]),
    pillar_rates=jnp.array([0.045, 0.042, 0.038, 0.035, 0.032, 0.030, 0.028]),
)


class TestSubYearAndFractionalTenors:
    """The ORE.Period("0.75Y") -> 0Y regression."""

    def test_single_sub_year_gap_to_final_maturity(self):
        """A 3-month gap from exercise to final maturity."""
        targets = build_coterminal_basket(
            exercise_times=[0.75], final_maturity_time=1.0,
            notional=1_000_000.0, payer=True, market_vols=[0.01],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
        assert np.isfinite(float(result.sigma.values[0]))
        assert result.rmse < 1e-4

    @pytest.mark.slow
    def test_quarterly_exercise_schedule_all_sub_year_gaps(self):
        """Quarterly exercise against a 2Y maturity: every gap is sub-year."""
        exercise_times = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        vols = [0.008 + 0.0002 * i for i in range(len(exercise_times))]
        targets = build_coterminal_basket(
            exercise_times=exercise_times, final_maturity_time=2.0,
            notional=1_000_000.0, payer=True, market_vols=vols,
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
        assert np.all(np.isfinite(np.asarray(result.sigma.values)))
        assert result.rmse < 1e-4
        # Every bucket reprices its own instrument.
        np.testing.assert_allclose(
            np.asarray(result.model_prices), np.asarray(result.market_prices), atol=1e-3,
        )

    def test_odd_month_gap_9_months(self):
        """A 9-month gap is exact in whole months."""
        targets = build_coterminal_basket(
            exercise_times=[1.0], final_maturity_time=1.75,
            notional=1_000_000.0, payer=True, market_vols=[0.009],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
        assert np.isfinite(float(result.sigma.values[0]))
        # 9 months of semiannual fixed accrual: one fixed coupon.
        assert len(targets[0].fixed_cashflow_times) >= 1

    def test_gap_that_rounds_to_zero_months_raises(self):
        """An exercise within half a month of maturity rounds to a 0M swap and raises."""
        with pytest.raises(ValueError):
            build_coterminal_basket(
                exercise_times=[4.999], final_maturity_time=5.0,
                notional=1_000_000.0, payer=True, market_vols=[0.01],
                zero_curve=FLAT_CURVE, evaluation_date=TODAY,
            )


class TestDegenerateBasketShapes:
    def test_single_instrument_basket(self):
        targets = build_coterminal_basket(
            exercise_times=[3.0], final_maturity_time=8.0,
            notional=1_000_000.0, payer=True, market_vols=[0.01],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
        assert result.sigma.times.shape == (0,)
        assert result.sigma.values.shape == (1,)

    @pytest.mark.slow
    def test_very_large_basket_ten_instruments(self):
        """A ten-instrument basket (annual exercises on a 10Y trade)."""
        exercise_times = list(range(1, 10))
        vols = [0.007 + 0.0003 * i for i in range(len(exercise_times))]
        targets = build_coterminal_basket(
            exercise_times=[float(t) for t in exercise_times], final_maturity_time=10.0,
            notional=1_000_000.0, payer=True, market_vols=vols,
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
        assert result.sigma.values.shape == (9,)
        assert result.rmse < 1e-3
        np.testing.assert_allclose(
            np.asarray(result.model_prices), np.asarray(result.market_prices), atol=1e-2,
        )

    def test_two_instrument_basket(self):
        targets = build_coterminal_basket(
            exercise_times=[2.0, 4.0], final_maturity_time=6.0,
            notional=1_000_000.0, payer=True, market_vols=[0.009, 0.0095],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
        assert result.sigma.values.shape == (2,)
        assert result.sigma.times.shape == (1,)


class TestExtremeMarketVols:
    def test_near_zero_market_vol(self):
        """A tiny market vol calibrates to a tiny, positive, finite sigma."""
        targets = build_coterminal_basket(
            exercise_times=[2.0], final_maturity_time=5.0,
            notional=1_000_000.0, payer=True, market_vols=[1e-5],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
        s = float(result.sigma.values[0])
        assert np.isfinite(s)
        assert 0.0 < s < 1e-2

    def test_very_high_market_vol(self):
        """A 150bp normal vol, inside the [1e-6, 0.20] bracket, calibrates cleanly."""
        targets = build_coterminal_basket(
            exercise_times=[2.0], final_maturity_time=5.0,
            notional=1_000_000.0, payer=True, market_vols=[0.015],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
        s = float(result.sigma.values[0])
        assert np.isfinite(s)
        assert s > 0.0
        assert result.rmse < 1e-3

    def test_mildly_non_monotone_vols_still_reprice_exactly(self):
        """A mild dip in the vol curve still reprices every instrument exactly (only a large
        dip cannot)."""
        targets = build_coterminal_basket(
            exercise_times=[1.0, 2.0, 3.0, 4.0], final_maturity_time=5.0,
            notional=1_000_000.0, payer=True, market_vols=[0.008, 0.009, 0.0088, 0.0095],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
        assert np.all(np.isfinite(np.asarray(result.sigma.values)))
        assert np.all(np.asarray(result.sigma.values) > 0.0)
        np.testing.assert_allclose(
            np.asarray(result.model_prices), np.asarray(result.market_prices), atol=1e-2,
        )

    def test_large_dip_after_a_spike_cannot_reprice_exactly(self):
        """A structural limit of bootstrapping: zeta only grows, so once earlier buckets hold
        enough variance, no non-negative sigma for a later bucket can bring its price down
        to a much lower market target. That bucket's bisection stops at the floor (1e-6)
        with a residual error; the supplied vols are not representable by any piecewise
        constant variance. The failure is graceful: finite, floor-clamped sigma and a large,
        reported RMSE."""
        targets = build_coterminal_basket(
            exercise_times=[1.0, 2.0, 3.0, 4.0], final_maturity_time=5.0,
            notional=1_000_000.0, payer=True, market_vols=[0.006, 0.014, 0.005, 0.012],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
        values = np.asarray(result.sigma.values)
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)
        # Bucket 2 (after the spike-then-drop) sits at the bisection floor.
        assert values[2] == pytest.approx(1e-6, abs=1e-9)
        # The RMSE is large but finite and reported.
        assert np.isfinite(result.rmse)
        assert result.rmse > 100.0


class TestCurveShapes:
    @pytest.mark.parametrize("curve,label", [
        (SLOPED_CURVE, "upward-sloping"),
        (INVERTED_CURVE, "inverted"),
    ])
    def test_calibration_reprices_exactly_under_nonflat_curve(self, curve, label):
        targets = build_coterminal_basket(
            exercise_times=[1.0, 2.0, 3.0], final_maturity_time=5.0,
            notional=1_000_000.0, payer=True, market_vols=[0.008, 0.009, 0.0095],
            zero_curve=curve, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, curve, a=0.03)
        assert np.all(np.isfinite(np.asarray(result.sigma.values)))
        np.testing.assert_allclose(
            np.asarray(result.model_prices), np.asarray(result.market_prices), atol=1e-2,
        ), f"{label} curve failed to reprice exactly"

    def test_receiver_basket_under_sloped_curve(self):
        targets = build_coterminal_basket(
            exercise_times=[1.0, 2.0, 3.0], final_maturity_time=5.0,
            notional=1_000_000.0, payer=False, market_vols=[0.008, 0.009, 0.0095],
            zero_curve=SLOPED_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, SLOPED_CURVE, a=0.03)
        assert np.all(np.isfinite(np.asarray(result.sigma.values)))
        assert np.all(np.asarray(result.sigma.values) > 0.0)

    def test_payer_and_receiver_calibrate_to_the_same_sigma_atm(self):
        """ATM payer and receiver swaptions have equal value (parity), so calibrating
        either to the same vol gives the same sigma."""
        payer_targets = build_coterminal_basket(
            exercise_times=[2.0], final_maturity_time=5.0,
            notional=1_000_000.0, payer=True, market_vols=[0.01],
            zero_curve=SLOPED_CURVE, evaluation_date=TODAY,
        )
        receiver_targets = build_coterminal_basket(
            exercise_times=[2.0], final_maturity_time=5.0,
            notional=1_000_000.0, payer=False, market_vols=[0.01],
            zero_curve=SLOPED_CURVE, evaluation_date=TODAY,
        )
        payer_result = calibrate_lgm_sigma(payer_targets, SLOPED_CURVE, a=0.03)
        receiver_result = calibrate_lgm_sigma(receiver_targets, SLOPED_CURVE, a=0.03)
        assert float(payer_result.sigma.values[0]) == pytest.approx(
            float(receiver_result.sigma.values[0]), rel=1e-6,
        )


class TestMeanReversionSensitivity:
    @pytest.mark.parametrize("a", [0.001, 0.01, 0.03, 0.10, 0.30])
    def test_calibrates_cleanly_across_a_wide_mean_reversion_range(self, a):
        """Mean reversion is an input; the bootstrap is well posed across a range including
        near zero."""
        targets = build_coterminal_basket(
            exercise_times=[1.0, 2.0, 3.0], final_maturity_time=5.0,
            notional=1_000_000.0, payer=True, market_vols=[0.008, 0.009, 0.0095],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=a)
        assert np.all(np.isfinite(np.asarray(result.sigma.values)))
        assert np.all(np.asarray(result.sigma.values) > 0.0)
        assert result.rmse < 1e-3

    def test_zero_mean_reversion_calibrates(self):
        """a = 0 exactly (guarded in `engine.models.lgm.H`) gives no error or NaN."""
        targets = build_coterminal_basket(
            exercise_times=[2.0], final_maturity_time=5.0,
            notional=1_000_000.0, payer=True, market_vols=[0.01],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.0)
        assert np.isfinite(float(result.sigma.values[0]))
        assert float(result.sigma.values[0]) > 0.0


class TestNotionalAndScaleInvariance:
    def test_calibrated_sigma_is_notional_invariant(self):
        """Sigma does not depend on notional: model and market prices both scale with it."""
        targets_small = build_coterminal_basket(
            exercise_times=[1.0, 2.0], final_maturity_time=4.0,
            notional=1_000.0, payer=True, market_vols=[0.008, 0.009],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        targets_large = build_coterminal_basket(
            exercise_times=[1.0, 2.0], final_maturity_time=4.0,
            notional=100_000_000.0, payer=True, market_vols=[0.008, 0.009],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        result_small = calibrate_lgm_sigma(targets_small, FLAT_CURVE, a=0.03)
        result_large = calibrate_lgm_sigma(targets_large, FLAT_CURVE, a=0.03)
        np.testing.assert_allclose(
            np.asarray(result_small.sigma.values), np.asarray(result_large.sigma.values), rtol=1e-6,
        )
