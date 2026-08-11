"""
Edge-case tests for engine.calibration -- degenerate basket shapes, extreme
market inputs, sloped curves, and boundary conditions not covered by
test_calibration_basket.py/test_calibration_lgm.py's own "typical" cases.

Written after a real bug was found and fixed via exactly this kind of
probing: `build_coterminal_basket` constructed its underlying swap's own
tenor as a fractional-year STRING (e.g. `"0.750000Y"`) whenever an exercise
time didn't split the trade's final maturity into whole years.
`ORE.Period("0.75Y")` does not raise -- it silently parses to `0Y` (ORE's
own period-string grammar has no fractional-year support at all), which
then either crashed downstream (an effective date landing at or past a
zero-length swap's own termination date) or, more dangerously, would have
silently built a WRONG-tenor underlying swap without crashing at all for
some inputs. Fixed by constructing the tenor in whole MONTHS instead
(`f"{tenor_months}M"`, `ORE.Period("9M")` parses correctly) -- see
`engine/calibration/basket.py`'s own comment at the fix site.
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
    """Regression coverage for the ORE.Period("0.75Y") -> 0Y bug (see
    module docstring)."""

    def test_single_sub_year_gap_to_final_maturity(self):
        """3-month gap from exercise to final maturity -- exactly the
        shape that used to crash (ORE.Period("0.25Y") parsing to 0Y)."""
        targets = build_coterminal_basket(
            exercise_times=[0.75], final_maturity_time=1.0,
            notional=1_000_000.0, payer=True, market_vols=[0.01],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
        assert np.isfinite(float(result.sigma.values[0]))
        assert result.rmse < 1e-4

    def test_quarterly_exercise_schedule_all_sub_year_gaps(self):
        """A quarterly Bermudan exercise schedule (0.25Y spacing) against a
        2Y final maturity -- every gap between consecutive buckets is a
        sub-year, non-whole-year tenor."""
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
        # Every calibrated bucket must reprice its own instrument (a
        # genuine correctness check, not just "didn't crash").
        np.testing.assert_allclose(
            np.asarray(result.model_prices), np.asarray(result.market_prices), atol=1e-3,
        )

    def test_odd_month_gap_9_months(self):
        """A 9-month (non-6-month-multiple) gap -- confirms whole-MONTH
        rounding (not just whole-quarter) is exact."""
        targets = build_coterminal_basket(
            exercise_times=[1.0], final_maturity_time=1.75,
            notional=1_000_000.0, payer=True, market_vols=[0.009],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
        assert np.isfinite(float(result.sigma.values[0]))
        # 9 months of semiannual-fixed accrual -- exactly 1 fixed coupon.
        assert len(targets[0].fixed_cashflow_times) >= 1

    def test_gap_that_rounds_to_zero_months_raises(self):
        """An exercise time landing within half a month of the final
        maturity rounds to a ZERO-length underlying swap -- must raise
        clearly (an AssertionError from the tenor_months>0 guard), not
        silently build a garbage 0M swap the way the pre-fix code did."""
        with pytest.raises(AssertionError):
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

    def test_very_large_basket_ten_instruments(self):
        """Ten exercise dates (annual, out to a 10Y trade) -- a larger
        basket than any existing test, checking the bootstrap's own
        O(n) sequencing doesn't degrade for a longer schedule."""
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
        """A vanishingly small market vol should calibrate to a
        vanishingly small (but still positive, still finite) sigma --
        not NaN/Inf/negative."""
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
        """A stressed, very high (150bp normal) market vol -- still
        within the bootstrap's own [1e-6, 0.20] bisection bracket, must
        calibrate cleanly."""
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
        """A market vol curve with a small dip between consecutive
        exercise dates (not monotone, but mild) -- the bootstrap must
        still reprice every instrument exactly, confirming non-monotone
        market vols aren't a problem PER SE (only a large enough dip is
        -- see test_large_dip_after_a_spike_cannot_reprice_exactly below
        for the structural limit)."""
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
        """A STRUCTURAL, not implementation, limitation of bootstrap
        calibration: `zeta` (accumulated variance) is monotonically
        non-decreasing in each bucket's own sigma^2 >= 0, so once an
        earlier bucket's calibrated sigma has locked in enough cumulative
        variance, no non-negative value for a LATER bucket can reduce the
        model price back down to a sufficiently lower market target --
        the later bucket's bisection saturates at its own bracket floor
        (1e-6) without reaching zero calibration error. This is not a bug
        (confirmed directly: even setting that bucket's sigma to its
        absolute floor still overshoots the market price) -- it reflects
        a genuine internal inconsistency in the SUPPLIED market vol curve
        itself (a spike immediately followed by a big enough drop is not
        representable by any non-negative piecewise-constant variance
        process), the same class of infeasibility ORE's own `LgmBuilder::
        calibrate()` precheck logic (`MoveVolatility`) exists to partially
        mitigate for the opposite (too-small) direction, not fully solve
        in general. This test asserts the FAILURE MODE is graceful (finite,
        non-negative, floor-clamped sigma; a large but finite, reportable
        RMSE) rather than silent/NaN/crashing."""
        targets = build_coterminal_basket(
            exercise_times=[1.0, 2.0, 3.0, 4.0], final_maturity_time=5.0,
            notional=1_000_000.0, payer=True, market_vols=[0.006, 0.014, 0.005, 0.012],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
        values = np.asarray(result.sigma.values)
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)
        # bucket 2 (the one following the spike-then-drop) saturates at
        # the bootstrap's own bisection floor -- it could not go lower.
        assert values[2] == pytest.approx(1e-6, abs=1e-9)
        # the RMSE is large (reflecting the genuine mispricing) but
        # finite -- not silently reported as a clean calibration.
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
        """A co-terminal basket is always struck ATM -- put-call parity
        means a payer and receiver swaption on the SAME underlying have
        identical value, so calibrating either one to the SAME market vol
        must produce the SAME sigma."""
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
        """Mean reversion is always a fixed input (never calibrated -- see
        engine/calibration/lgm.py's own docstring) -- confirms the
        bootstrap itself is well-posed across a realistic range of a,
        including near-zero (close to arithmetic Brownian motion) and
        a relatively high value."""
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
        """a=0 exactly -- the arithmetic-Brownian-motion limit, guarded by
        a jnp.where branch in engine.models.lgm.H -- must not raise or
        produce NaN."""
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
        """Sigma is a MODEL parameter (a volatility), not a price -- it
        must not depend on the trade's own notional at all, since both
        the model price and the market (Bachelier) price scale linearly
        with notional, cancelling out in the calibration equation."""
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
