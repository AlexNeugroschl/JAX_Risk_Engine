"""Numerical boundary regressions (I-104), including independently priced cashflows."""
import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.instruments.european_swaption import SwaptionConfig, underlying_swap
from engine.instruments.schedules import ibor_index
from engine.instruments.treasury import BondConfig, CouponPeriod
from engine.market_data.market import CurrencyMarket, Market, SwaptionVolSurface, ZeroCurveConfig, index_name
from engine.market_simulation.config import CamConfig, LgmConfig, simulate
from engine.models.lgm import H
from engine.pricing.config import PricingConfig
from engine.pricing.context import from_market
from engine.pricing.cube import trade_maturity, value_on, value_portfolio


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_h_preserves_small_reversion_increments(dtype):
    t = np.asarray([0., 1 / 365, 1., 30.], dtype=np.dtype(dtype))
    a = np.asarray(1e-8, dtype=np.dtype(dtype))
    expected = np.asarray([-math.expm1(-float(a) * float(v)) / float(a) for v in t])
    actual = np.asarray(jax.jit(H)(jnp.asarray(a), jnp.asarray(t)))
    np.testing.assert_allclose(actual, expected, rtol=4 * np.finfo(t.dtype).eps, atol=0.)


@pytest.mark.parametrize("t", [0., 1 / 365, 1., 30.])
def test_h_has_the_analytic_reversion_derivative_at_zero(t):
    assert float(jax.grad(lambda a: H(a, t))(0.)) == pytest.approx(-t * t / 2, abs=1e-14)
    assert float(jax.grad(jax.grad(lambda a: H(a, t)))(0.)) == pytest.approx(t ** 3 / 3, abs=1e-12)


@pytest.mark.parametrize("payer", [False, True])
def test_zero_variance_option_keeps_its_intrinsic_value_before_expiry(payer):
    asof = ORE.Date(30, 7, 2026)
    name = index_name("USD", 6)
    curve = ZeroCurveConfig([0., 1., 5., 20.], [0.03] * 4)
    market = Market(asof, {"USD": CurrencyMarket(curve, {name: curve},
                    SwaptionVolSurface(("1Y", "5Y"), ("1Y", "10Y"), ((0., 0.), (0., 0.))))})
    option = SwaptionConfig(notional=1e6, fixed_rate=0.05 if not payer else 0.01, payer=payer,
                            swap_tenor="5Y", forward_start=ORE.Period(2, ORE.Years),
                            evaluation_date=asof, trade_id="zero-vol")
    date = asof + ORE.Period(1, ORE.Years)
    dates = (date, option.exercise_date, option.exercise_date + 1)
    scenarios = simulate(market, CamConfig(dates=dates, base_currency="USD", ir={"USD": LgmConfig(.03, 0.)},
                                          samples=8, swaption_vol_decay="ConstantVariance"))
    result = value_portfolio([option], market, scenarios, "USD", decay="ConstantVariance")
    # Independent QuantLib price on the exact scenario curves (including tenor interpolation).
    previous = ORE.Settings.instance().evaluationDate
    try:
        ORE.Settings.instance().evaluationDate = date
        def handle(curves):
            times = np.asarray(curves.tenor_times[0])
            dates = [date + round(float(t) * 365) for t in times]
            curve = ORE.DiscountCurve(dates, np.exp(np.asarray(curves.log_discounts)[0, 0]).tolist(),
                                      ORE.Actual365Fixed())
            curve.enableExtrapolation()
            return ORE.YieldTermStructureHandle(curve)
        disc, index = handle(scenarios.discount["USD"]), handle(scenarios.index[name])
        booked = underlying_swap(option)
        swap = ORE.VanillaSwap(ORE.VanillaSwap.Payer if payer else ORE.VanillaSwap.Receiver,
                              option.notional, booked.fixedSchedule(), option.fixed_rate, ORE.Actual365Fixed(),
                              booked.floatingSchedule(), ibor_index(6, index), 0., ORE.Actual365Fixed())
        reference = ORE.Swaption(swap, ORE.EuropeanExercise(option.exercise_date))
        reference.setPricingEngine(ORE.BachelierSwaptionEngine(disc, ORE.QuoteHandle(ORE.SimpleQuote(0.))))
        assert reference.NPV() > 1000.
        np.testing.assert_allclose(np.asarray(result.cube)[:, 0, 0], reference.NPV(), rtol=1e-12)
    finally:
        ORE.Settings.instance().evaluationDate = previous


def test_delayed_final_coupon_and_redemption_match_ores_bond_through_payment():
    asof, start, maturity, payment = (ORE.Date(30, 7, 2026), ORE.Date(1, 8, 2026),
                                     ORE.Date(1, 8, 2027), ORE.Date(2, 8, 2027))
    cfg = BondConfig(100000., maturity, coupon_rate=.04,
                     coupon_schedule=(CouponPeriod(start, maturity, payment),),
                     evaluation_date=asof, trade_id="delayed")
    previous = ORE.Settings.instance().evaluationDate
    try:
        schedule = ORE.Schedule([start, maturity], ORE.NullCalendar(), ORE.Unadjusted)
        bond = ORE.FixedRateBond(0, cfg.face_amount, schedule, [cfg.coupon_rate],
                                 ORE.ActualActual(ORE.ActualActual.ISMA), ORE.Following,
                                 100., start, ORE.TARGET())
        for date in (asof, maturity, payment):
            ORE.Settings.instance().evaluationDate = date
            handle = ORE.YieldTermStructureHandle(ORE.FlatForward(date, .03, ORE.Actual365Fixed()))
            bond.setPricingEngine(ORE.DiscountingBondEngine(handle))
            context = from_market(Market(date, {"USD": CurrencyMarket(ZeroCurveConfig([0., 5.], [.03, .03]))}))
            assert value_on(cfg, context, PricingConfig()) == pytest.approx(bond.NPV(), rel=1e-12, abs=1e-9)
        assert trade_maturity(cfg) == payment
        # A trade can be booked after accrual maturity while its final cashflow is outstanding.
        rolled = dataclasses.replace(cfg, evaluation_date=maturity)
        assert rolled.evaluation_date < payment
    finally:
        ORE.Settings.instance().evaluationDate = previous
