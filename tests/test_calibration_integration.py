"""
Calibration end to end: market swaption vols in, a Bermudan NPV out, with the `Sigma` from
`calibrate_lgm_sigma` (the standalone `POST /calibration/lgm` basket) passed straight to the grid
engine as its volatility (no conversion).
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.models.hull_white import ZeroCurve
from engine.market import ZeroCurveConfig
from engine.calibration.basket import build_coterminal_basket
from engine.calibration.lgm import calibrate_lgm_sigma
from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from date_helpers import in_years
from tests.support.lgm_engine import grid_npv

TODAY = ORE.Date(30, 7, 2026)


@pytest.fixture(autouse=True)
def _set_eval_date():
    ORE.Settings.instance().evaluationDate = TODAY


FLAT_CURVE = ZeroCurve.flat(0.03, [0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0])
FLAT_CURVE_CONFIG = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0], rates=[0.03] * 7)


def _bermudan(fixed_rate=0.03, payer=True, exercise_times=(1.0, 2.0, 3.0, 4.0), swap_tenor="5Y",
              notional=1_000_000.0) -> BermudanSwaptionConfig:
    return BermudanSwaptionConfig(notional=notional, fixed_rate=fixed_rate, payer=payer,
                                  exercise_dates=in_years(TODAY, list(exercise_times)), swap_tenor=swap_tenor,
                                  evaluation_date=TODAY, trade_id="bermudan")


def _npv(cfg, sigma, curve=FLAT_CURVE_CONFIG, **grid) -> float:
    """The grid engine on an LGM with reversion 0.03 and volatility `sigma`."""
    return grid_npv(cfg, a=0.03, sigma=sigma, curve=curve, **grid)


class TestCalibratedSigmaFeedsBermudanPricer:
    def test_calibrated_sigma_prices_a_finite_bermudan_npv(self):
        targets = build_coterminal_basket(
            exercise_times=[1.0, 2.0, 3.0, 4.0], final_maturity_time=5.0,
            notional=1_000_000.0, payer=True, market_vols=[0.008, 0.009, 0.0095, 0.0098],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
        npv = _npv(_bermudan(), result.sigma)
        assert np.isfinite(npv)
        assert npv > 0.0

    def test_calibrated_sigma_differs_meaningfully_from_flat_average_sigma(self):
        """The Bermudan value depends on the shape of the vol term structure, not just its
        level: an upward-sloping market vol curve gives a different price from a flat
        sigma."""
        targets = build_coterminal_basket(
            exercise_times=[1.0, 2.0, 3.0, 4.0], final_maturity_time=5.0,
            notional=1_000_000.0, payer=True, market_vols=[0.008, 0.009, 0.0095, 0.0098],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
        avg_sigma = float(jnp.mean(result.sigma.values))
        npv_calibrated = _npv(_bermudan(), result.sigma)
        npv_flat = _npv(_bermudan(), avg_sigma)
        assert abs(npv_calibrated - npv_flat) / npv_flat > 0.01

    def test_american_swaption_also_accepts_calibrated_sigma(self):
        """An American trade (same backward induction) accepts a calibrated `Sigma`."""
        targets = build_coterminal_basket(
            exercise_times=[1.0, 2.0, 3.0], final_maturity_time=4.0,
            notional=1_000_000.0, payer=True, market_vols=[0.008, 0.009, 0.0095],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
        cfg = AmericanSwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True,
            first_exercise_date=in_years(TODAY, 1.0), last_exercise_date=in_years(TODAY, 3.0),
            swap_tenor="4Y", evaluation_date=TODAY, trade_id="american",
        )
        npv = _npv(cfg, result.sigma, steps_per_year=2)
        assert np.isfinite(npv)
        assert npv > 0.0


SLOPED_CURVE = ZeroCurve(
    pillar_times=jnp.array([0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0]),
    pillar_rates=jnp.array([0.020, 0.025, 0.028, 0.030, 0.032, 0.035, 0.038]),
)
SLOPED_CURVE_CONFIG = ZeroCurveConfig(
    times=[0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0], rates=[0.020, 0.025, 0.028, 0.030, 0.032, 0.035, 0.038],
)


class TestCalibratedSigmaAcrossTradeVariations:
    """Receivers, a sloped curve, and several trades priced from one calibrated `Sigma` (as
    a desk would reuse one calibration per curve)."""

    def test_receiver_bermudan_with_calibrated_sigma(self):
        targets = build_coterminal_basket(
            exercise_times=[1.0, 2.0, 3.0, 4.0], final_maturity_time=5.0,
            notional=1_000_000.0, payer=False, market_vols=[0.008, 0.009, 0.0095, 0.0098],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
        npv = _npv(_bermudan(payer=False), result.sigma)
        assert np.isfinite(npv)
        assert npv > 0.0

    def test_calibrated_sigma_under_a_sloped_curve_prices_a_bermudan(self):
        """An upward-sloping curve used for both calibration and pricing."""
        targets = build_coterminal_basket(
            exercise_times=[1.0, 2.0, 3.0], final_maturity_time=5.0,
            notional=1_000_000.0, payer=True, market_vols=[0.008, 0.009, 0.0095],
            zero_curve=SLOPED_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, SLOPED_CURVE, a=0.03)
        npv = _npv(_bermudan(fixed_rate=0.032, exercise_times=(1.0, 2.0, 3.0)), result.sigma, curve=SLOPED_CURVE_CONFIG)
        assert np.isfinite(npv)
        assert npv > 0.0

    @pytest.mark.slow
    def test_one_calibrated_sigma_prices_a_diverse_multi_trade_portfolio(self):
        """One `Sigma` calibrated to a basket spanning the longest trade prices several
        Bermudan trades whose exercise dates need not match the basket's breakpoints; all
        values are finite."""
        exercise_times = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        targets = build_coterminal_basket(
            exercise_times=list(exercise_times), final_maturity_time=7.0,
            notional=1_000_000.0, payer=True,
            market_vols=[0.007, 0.0078, 0.0085, 0.009, 0.0093, 0.0095],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
        trades = [
            # Deep-ITM payer, full exercise schedule, matches calibration horizon.
            _bermudan(fixed_rate=0.01, exercise_times=exercise_times, swap_tenor="7Y"),
            # OTM receiver, sparse exercise (a subset of the basket dates), shorter underlying.
            _bermudan(fixed_rate=0.01, payer=False, exercise_times=(2.0, 4.0), notional=2_000_000.0),
            # ATM payer, single-exercise (European-equivalent) trade.
            _bermudan(exercise_times=(3.0,), swap_tenor="4Y", notional=500_000.0),
        ]
        npvs = [_npv(cfg, result.sigma) for cfg in trades]
        assert all(np.isfinite(v) for v in npvs)
        # The deep ITM payer is worth more than the small ATM single-exercise trade despite a
        # smaller notional.
        assert npvs[0] > npvs[2]
