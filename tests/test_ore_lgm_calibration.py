"""
ORE's LGM calibration (`engine.calibration.ore_lgm`) against ORE.

  * Helper market values against QuantLib's own `SwaptionHelper.marketValue()`: tenor- and
    date-based helpers, ATM and deal strikes, on a sloped two-curve market.
  * The bootstrap reprices every helper: model value = market value.
  * End to end (I-47): a Bermudan calibrated to its co-terminal basket and priced on the grid
    equals ORE's `LGMGridSwaptionEngineBuilder` with `Calibration=Bootstrap` -- ORE builds
    the basket, calibrates its own LGM (`AnalyticLgmSwaptionEngine`, proRata spread mapping)
    and prices, through the in-process OREApp oracle. `CoterminalATM` and
    `CoterminalDealStrike`, including a strike far enough out of the money that ORE's
    fallback rule 1 moves it.
  * Batched curves (one per path) calibrate exactly as one curve at a time.

AnalyticLgmSwaptionEngine has no Python constructor, so the model value is checked through
the end-to-end NPV, where any error in it moves the calibrated volatility and the price.
"""
import dataclasses

import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.calibration.ore_lgm import (
    SwapIndexConventions, basket_vols, bootstrap_sigma, build_basket, market_price,
)
from engine.instruments.bermudan_swaption import (
    BermudanSwaptionConfig, _build_ore_swap, exercisable_dates,
)
from engine.market import SwaptionVolSurface, ZeroCurveConfig
from engine.models.curves import DiscountCurve, ZeroCurve, log_discount
from engine.models.lgm import Sigma
from engine.models.ore_builders import TIME_AXIS_DAY_COUNTER as DC, ibor_index
from tests.support.lgm_engine import grid_npv
from tests.support.ore_lgm_oracle import OreCalibration, ore_lgm_swaption_npv

ASOF = ORE.Date(30, 7, 2026)
PILLARS = [0.0, 1.0, 2.0, 5.0, 10.0, 30.0]
# Flat to the first non-zero pillar, so ORE's curve rebuild keeps it exactly (I-34).
DISC_RATES = [0.02, 0.02, 0.025, 0.03, 0.035, 0.04]
INDEX_RATES = [0.025, 0.025, 0.031, 0.036, 0.04, 0.044]
VOLS = SwaptionVolSurface(
    ("1Y", "2Y", "3Y", "5Y", "10Y"), ("1Y", "2Y", "5Y", "10Y"),
    ((0.0080, 0.0085, 0.0090, 0.0092), (0.0085, 0.0088, 0.0091, 0.0093), (0.0088, 0.0090, 0.0092, 0.0094),
     (0.0090, 0.0092, 0.0093, 0.0095), (0.0092, 0.0093, 0.0094, 0.0096)))
REVERSION = 0.03


def _curve(rates):
    return ZeroCurve.from_config(ZeroCurveConfig(PILLARS, rates))


def _ore_handle(rates):
    handle = ORE.YieldTermStructureHandle(ORE.ZeroCurve([ASOF + round(t * 365) for t in PILLARS], rates, DC))
    handle.enableExtrapolation()
    return handle


def _ore_helper(expiry, term, vol, strike=None):
    ORE.Settings.instance().evaluationDate = ASOF
    index = ibor_index(6, _ore_handle(INDEX_RATES))
    return ORE.SwaptionHelper(
        expiry, term, ORE.QuoteHandle(ORE.SimpleQuote(vol)), index, ORE.Period("1Y"), DC, DC,
        _ore_handle(DISC_RATES), ORE.BlackCalibrationHelper.RelativePriceError,
        ORE.nullDouble() if strike is None else strike, 1.0, ORE.Normal)


@pytest.mark.parametrize("expiry, term", [("1Y", "9Y"), ("2Y", "8Y"), ("5Y", "5Y"), ("6M", "18M")])
@pytest.mark.parametrize("strike", [None, 0.02, 0.05], ids=["atm", "receiver-otm", "payer-otm"])
def test_tenor_helper_market_value_equals_quantlib(expiry, term, strike):
    """QuantLib's bare helper takes a strike as given; ORE's fallback band (applied by the
    engine) is tested end to end below, so strikes beyond it are skipped here."""
    ours = build_basket(ASOF, [expiry], [term], deal_strikes=[strike])[0]
    vol = 0.009
    if strike is not None and abs(strike - _atm(ours)) > 3.0 * vol * np.sqrt(ours.expiry_time):
        pytest.skip("strike beyond ORE's fallback band")
    ore = _ore_helper(ORE.Period(expiry), ORE.Period(term), vol, strike)
    value = float(market_price(ours, _curve(DISC_RATES), _curve(INDEX_RATES), vol))
    assert value == pytest.approx(ore.marketValue(), rel=1e-13)


def _atm(instrument):
    from engine.calibration.ore_lgm import _legs
    return float(_legs(instrument, _curve(DISC_RATES), _curve(INDEX_RATES)).forward)


def test_date_helper_market_value_equals_quantlib():
    expiry, end = ORE.Date(3, 8, 2028), ORE.Date(5, 8, 2033)
    ours = build_basket(ASOF, [expiry], [end])[0]
    ore = _ore_helper(expiry, end, 0.0091)
    assert float(market_price(ours, _curve(DISC_RATES), _curve(INDEX_RATES), 0.0091)) == pytest.approx(
        ore.marketValue(), rel=1e-13)


def test_the_bootstrap_reprices_every_helper():
    basket = build_basket(ASOF, ["1Y", "2Y", "3Y", "5Y"], ["9Y", "8Y", "7Y", "5Y"])
    result = bootstrap_sigma(basket, _curve(DISC_RATES), _curve(INDEX_RATES), basket_vols(basket, VOLS, ASOF),
                             REVERSION)
    np.testing.assert_allclose(np.asarray(result.model), np.asarray(result.market), rtol=1e-12)
    assert not np.any(np.asarray(result.hit_ceiling))
    np.testing.assert_allclose(result.times, [b.expiry_time for b in basket][:-1])


def _bermudan(fixed_rate):
    booked = BermudanSwaptionConfig(
        notional=1e6, fixed_rate=fixed_rate, payer=True, exercise_dates=[ASOF + 400], swap_tenor="6Y",
        evaluation_date=ASOF, trade_id="bermudan")
    return dataclasses.replace(booked, exercise_dates=exercisable_dates(booked)[1:])


@pytest.mark.parametrize("strategy, fixed_rate", [
    ("CoterminalATM", 0.03), ("CoterminalDealStrike", 0.03), ("CoterminalDealStrike", 0.09),
], ids=["atm", "deal-strike", "deal-strike-beyond-3-std-devs"])
def test_calibrated_bermudan_equals_ores_bootstrap(strategy, fixed_rate):
    """I-47: the basket is built from the trade's own exercise dates and maturity, struck ATM
    or at the deal strike, and calibrated per trade, as ORE does."""
    cfg = _bermudan(fixed_rate)
    n = len(cfg.exercise_dates)
    strikes = [fixed_rate] * n if strategy == "CoterminalDealStrike" else None
    basket = build_basket(ASOF, list(cfg.exercise_dates), [cfg.maturity_date] * n, deal_strikes=strikes)
    result = bootstrap_sigma(basket, _curve(DISC_RATES), _curve(INDEX_RATES), basket_vols(basket, VOLS, ASOF),
                             REVERSION)
    ore = ore_lgm_swaption_npv(
        evaluation_date=ASOF, curve_times=PILLARS, curve_rates=DISC_RATES, swap=_build_ore_swap(cfg),
        notional=cfg.notional, fixed_rate=fixed_rate, payer=True, floating_spread=0.0, index_tenor_months=6,
        style="Bermudan", exercise_dates=cfg.exercise_dates, hw_a=REVERSION, hw_sigma=0.01, n_per_std=48,
        std_devs=6.0, index_curve_rates=INDEX_RATES, swaption_vols=VOLS,
        calibration=OreCalibration("Bootstrap", strategy, 1e-8)).npv
    # Measured worst case 2.1e-11 (ORE's LM optimiser and the engine's bisection find the same
    # root to ~1e-14 in price).
    engine = grid_npv(cfg, a=REVERSION, sigma=Sigma(jnp.asarray(result.times), result.values),
                      curve=ZeroCurveConfig(PILLARS, DISC_RATES), index_curve=ZeroCurveConfig(PILLARS, INDEX_RATES),
                      n_per_std=48, std_devs=6.0)
    assert engine == pytest.approx(ore, rel=1e-9)


def test_batched_curves_calibrate_as_one_at_a_time():
    """Per-path recalibration: a batch of path curves gives each path's own calibration."""
    basket = build_basket(ASOF, ["1Y", "2Y", "5Y"], ["9Y", "8Y", "5Y"])
    tenors = jnp.asarray([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    shifts = jnp.asarray([-0.004, 0.0, 0.006])
    base_disc, base_idx = _curve(DISC_RATES), _curve(INDEX_RATES)
    ln = lambda curve: log_discount(curve, tenors)  # noqa: E731
    disc = DiscountCurve(tenors, ln(base_disc)[None, :] - shifts[:, None] * tenors[None, :])
    idx = DiscountCurve(tenors, ln(base_idx)[None, :] - shifts[:, None] * tenors[None, :])
    vols = jnp.asarray([[0.008, 0.009, 0.0085], [0.009, 0.0095, 0.009], [0.0075, 0.008, 0.0082]])
    batched = bootstrap_sigma(basket, disc, idx, vols, REVERSION)
    for s in range(3):
        one = bootstrap_sigma(basket, DiscountCurve(tenors, disc.log_discounts[s]),
                              DiscountCurve(tenors, idx.log_discounts[s]), vols[s], REVERSION)
        np.testing.assert_allclose(np.asarray(batched.values[s]), np.asarray(one.values), rtol=1e-14)
    np.testing.assert_allclose(np.asarray(batched.model), np.asarray(batched.market), rtol=1e-12)


def test_swap_index_conventions_shape_the_helpers():
    annual = build_basket(ASOF, ["2Y"], ["5Y"])[0]
    semi = build_basket(ASOF, ["2Y"], ["5Y"], SwapIndexConventions(fixed_tenor="6M"))[0]
    assert annual.fixed_pay.size == 5 and semi.fixed_pay.size == 10
    assert annual.float_pay.size == semi.float_pay.size == 10


def test_the_basket_does_not_depend_on_ores_global_evaluation_date():
    """`build_basket` builds its helpers on their reference date whatever ORE's global
    evaluation date is (another caller's, or the wall clock's), and leaves it as it found it.
    A later global date once made the helper ask for fixings it could not have."""
    settings = ORE.Settings.instance()
    previous = settings.evaluationDate
    try:
        settings.evaluationDate = ASOF
        expected = build_basket(ASOF, ["1Y", "2Y", "5Y"], ["9Y", "8Y", "5Y"])
        later = ORE.Date(1, 1, 2035)
        settings.evaluationDate = later
        basket = build_basket(ASOF, ["1Y", "2Y", "5Y"], ["9Y", "8Y", "5Y"])
        assert settings.evaluationDate == later
    finally:
        settings.evaluationDate = previous
    for ours, theirs in zip(basket, expected):
        for f in dataclasses.fields(ours):
            np.testing.assert_array_equal(np.asarray(getattr(ours, f.name)), np.asarray(getattr(theirs, f.name)))
