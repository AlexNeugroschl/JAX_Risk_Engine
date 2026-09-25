"""
Integration test: engine.calibration end-to-end -- market swaption vols in,
a genuine Bermudan swaption NPV out, via calibrate_lgm_sigma feeding
BermudanSwaptionConfig.hw_sigma directly (no adapter/conversion code
needed -- this IS the point of `engine.models.lgm.Sigma` being the shared
representation both the calibration engine and the pricers speak, per
Phase 3's design).
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.models.hull_white import ZeroCurve
from engine.simulation.market_model import ZeroCurveConfig
from engine.calibration.basket import build_coterminal_basket
from engine.calibration.lgm import calibrate_lgm_sigma
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig, price_bermudan_swaption_base
from date_helpers import in_years

TODAY = ORE.Date(30, 7, 2026)


@pytest.fixture(autouse=True)
def _set_eval_date():
    ORE.Settings.instance().evaluationDate = TODAY


FLAT_CURVE = ZeroCurve.flat(0.03, [0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0])
FLAT_CURVE_CONFIG = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0], rates=[0.03] * 7)


class TestCalibratedSigmaFeedsBermudanPricer:
    def test_calibrated_sigma_prices_a_finite_bermudan_npv(self):
        exercise_times = [1.0, 2.0, 3.0, 4.0]
        targets = build_coterminal_basket(
            exercise_times=exercise_times, final_maturity_time=5.0,
            notional=1_000_000.0, payer=True, market_vols=[0.008, 0.009, 0.0095, 0.0098],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)

        cfg = BermudanSwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True, rate_factor_index=0,
            hw_a=0.03, hw_sigma=result.sigma,
            initial_zero_curve=FLAT_CURVE_CONFIG,
            exercise_dates=in_years(TODAY, exercise_times), swap_tenor="5Y", evaluation_date=TODAY,
        )
        npv = float(price_bermudan_swaption_base(cfg))
        assert np.isfinite(npv)
        assert npv > 0.0

    def test_calibrated_sigma_differs_meaningfully_from_flat_average_sigma(self):
        """The whole point of a piecewise-calibrated Sigma over a single
        flat scalar: a Bermudan's value depends on the SHAPE of the vol
        term structure, not just its average -- confirmed by checking the
        two prices genuinely differ (not just float-noise-close), for a
        market vol curve with real term structure (upward-sloping here)."""
        exercise_times = [1.0, 2.0, 3.0, 4.0]
        targets = build_coterminal_basket(
            exercise_times=exercise_times, final_maturity_time=5.0,
            notional=1_000_000.0, payer=True, market_vols=[0.008, 0.009, 0.0095, 0.0098],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
        avg_sigma = float(jnp.mean(result.sigma.values))

        cfg_calibrated = BermudanSwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True, rate_factor_index=0,
            hw_a=0.03, hw_sigma=result.sigma,
            initial_zero_curve=FLAT_CURVE_CONFIG,
            exercise_dates=in_years(TODAY, exercise_times), swap_tenor="5Y", evaluation_date=TODAY,
        )
        cfg_flat = BermudanSwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True, rate_factor_index=0,
            hw_a=0.03, hw_sigma=avg_sigma,
            initial_zero_curve=FLAT_CURVE_CONFIG,
            exercise_dates=in_years(TODAY, exercise_times), swap_tenor="5Y", evaluation_date=TODAY,
        )
        npv_calibrated = float(price_bermudan_swaption_base(cfg_calibrated))
        npv_flat = float(price_bermudan_swaption_base(cfg_flat))
        assert abs(npv_calibrated - npv_flat) / npv_flat > 0.01

    def test_american_swaption_also_accepts_calibrated_sigma(self):
        """An AmericanSwaptionConfig is priced by the same backward
        induction as a Bermudan -- confirms a calibrated Sigma is accepted
        on that path too, not just the Bermudan one."""
        from engine.instruments.american_swaption import AmericanSwaptionConfig
        from engine.instruments.bermudan_swaption import price_bermudan_swaption_base

        exercise_times = [1.0, 2.0, 3.0]
        targets = build_coterminal_basket(
            exercise_times=exercise_times, final_maturity_time=4.0,
            notional=1_000_000.0, payer=True, market_vols=[0.008, 0.009, 0.0095],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)

        cfg = AmericanSwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True, rate_factor_index=0,
            hw_a=0.03, hw_sigma=result.sigma,
            initial_zero_curve=FLAT_CURVE_CONFIG,
            first_exercise_date=in_years(TODAY, 1.0), last_exercise_date=in_years(TODAY, 3.0),
            exercise_time_steps_per_year=2,
            swap_tenor="4Y", evaluation_date=TODAY,
        )
        npv = float(price_bermudan_swaption_base(cfg))
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
    """Broader coverage than TestCalibratedSigmaFeedsBermudanPricer's three
    original cases: receiver trades, a sloped (non-flat) curve, and a
    genuinely diverse multi-trade portfolio all priced from ONE calibrated
    Sigma, since a real desk calibrates once per curve/currency and reuses
    the same Sigma across every trade sharing that curve."""

    def test_receiver_bermudan_with_calibrated_sigma(self):
        exercise_times = [1.0, 2.0, 3.0, 4.0]
        targets = build_coterminal_basket(
            exercise_times=exercise_times, final_maturity_time=5.0,
            notional=1_000_000.0, payer=False, market_vols=[0.008, 0.009, 0.0095, 0.0098],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)
        cfg = BermudanSwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=False, rate_factor_index=0,
            hw_a=0.03, hw_sigma=result.sigma,
            initial_zero_curve=FLAT_CURVE_CONFIG,
            exercise_dates=in_years(TODAY, exercise_times), swap_tenor="5Y", evaluation_date=TODAY,
        )
        npv = float(price_bermudan_swaption_base(cfg))
        assert np.isfinite(npv)
        assert npv > 0.0

    def test_calibrated_sigma_under_a_sloped_curve_prices_a_bermudan(self):
        """A non-flat (upward-sloping) curve, matching the SAME curve used
        both for calibration and for the Bermudan pricer's own
        initial_zero_curve -- confirms the pipeline works end-to-end when
        today's curve has real term structure, not just the flat curve
        every other integration test in this file uses."""
        exercise_times = [1.0, 2.0, 3.0]
        targets = build_coterminal_basket(
            exercise_times=exercise_times, final_maturity_time=5.0,
            notional=1_000_000.0, payer=True, market_vols=[0.008, 0.009, 0.0095],
            zero_curve=SLOPED_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, SLOPED_CURVE, a=0.03)
        cfg = BermudanSwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.032, payer=True, rate_factor_index=0,
            hw_a=0.03, hw_sigma=result.sigma,
            initial_zero_curve=SLOPED_CURVE_CONFIG,
            exercise_dates=in_years(TODAY, exercise_times), swap_tenor="5Y", evaluation_date=TODAY,
        )
        npv = float(price_bermudan_swaption_base(cfg))
        assert np.isfinite(npv)
        assert npv > 0.0

    @pytest.mark.slow
    def test_one_calibrated_sigma_prices_a_diverse_multi_trade_portfolio(self):
        """The realistic desk workflow: calibrate ONE Sigma from a market
        vol basket that spans the portfolio's own longest trade, then
        price SEVERAL Bermudan/American trades of varying tenor/exercise-
        schedule/moneyness/payer-receiver against that SAME calibrated
        Sigma -- confirms the calibrated term structure behaves sanely
        (finite, no NaN, no crash) when reused across trades whose own
        exercise schedules don't exactly match the calibration basket's
        own bucket breakpoints (a Bermudan need not exercise on every
        basket date)."""
        exercise_times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        targets = build_coterminal_basket(
            exercise_times=exercise_times, final_maturity_time=7.0,
            notional=1_000_000.0, payer=True,
            market_vols=[0.007, 0.0078, 0.0085, 0.009, 0.0093, 0.0095],
            zero_curve=FLAT_CURVE, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, FLAT_CURVE, a=0.03)

        trades = [
            # Deep-ITM payer, full exercise schedule, matches calibration horizon.
            BermudanSwaptionConfig(
                notional=1_000_000.0, fixed_rate=0.01, payer=True, rate_factor_index=0,
                hw_a=0.03, hw_sigma=result.sigma, initial_zero_curve=FLAT_CURVE_CONFIG,
                exercise_dates=in_years(TODAY, exercise_times), swap_tenor="7Y", evaluation_date=TODAY,
            ),
            # OTM receiver, sparse exercise schedule (a subset of the
            # calibration basket's own dates), shorter underlying.
            BermudanSwaptionConfig(
                notional=2_000_000.0, fixed_rate=0.01, payer=False, rate_factor_index=0,
                hw_a=0.03, hw_sigma=result.sigma, initial_zero_curve=FLAT_CURVE_CONFIG,
                exercise_dates=in_years(TODAY, [2.0, 4.0]), swap_tenor="5Y", evaluation_date=TODAY,
            ),
            # ATM payer, single-exercise (European-equivalent) trade.
            BermudanSwaptionConfig(
                notional=500_000.0, fixed_rate=0.03, payer=True, rate_factor_index=0,
                hw_a=0.03, hw_sigma=result.sigma, initial_zero_curve=FLAT_CURVE_CONFIG,
                exercise_dates=in_years(TODAY, [3.0]), swap_tenor="4Y", evaluation_date=TODAY,
            ),
        ]
        npvs = [float(price_bermudan_swaption_base(cfg)) for cfg in trades]
        assert all(np.isfinite(v) for v in npvs)
        # Deep ITM payer should be worth substantially more than the
        # small ATM single-exercise trade despite a smaller notional.
        assert npvs[0] > npvs[2]
