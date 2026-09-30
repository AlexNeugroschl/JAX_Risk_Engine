"""
ORE's sensitivity analysis on the market path (`engine.risk.sensitivities`, plan T-18/T-19).

  * The bump machinery: a forward-difference Delta at a tenor equals AD's first derivative
    times the shift plus half the second-derivative term, and Gamma the second derivative
    term, to O(shift^3) (plan X-3: AD and ORE's Greeks agree to the bump's order).
  * Theta: the date moves one calendar day (Friday -> Saturday, I-38), a coupon paid on the
    Theta date is added back (I-39), and a fixing of the as-of date is backfilled from the
    Theta market rather than refused.
  * Vega: the matrix of quote bumps adds up to a parallel volatility bump.

Parity with ORE's own sensitivity analytic (an OREApp sensitivity run) is not yet in the
suite; see docs/planning/known-issues.md I-51.
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig, _build_ore_swap
from engine.market import CurrencyMarket, Market, SwaptionVolSurface, ZeroCurveConfig, index_name
from engine.models.curves import DiscountCurve
from engine.risk.sensitivities import (
    SensitivityConfig, portfolio_sensitivities, sensitivity_context, theta_context,
)
from engine.valuation.config import PricingConfig
from engine.valuation.legs import legs_of, today_npv
from engine.valuation.portfolio import value_on

PILLARS = [0.0, 1.0, 2.0, 5.0, 10.0, 30.0]
DISC = [0.020, 0.020, 0.025, 0.030, 0.035, 0.040]
INDEX = [0.025, 0.025, 0.031, 0.036, 0.040, 0.044]
VOLS = SwaptionVolSurface(("1Y", "5Y", "10Y"), ("1Y", "5Y", "10Y"),
                          ((0.0080, 0.0088, 0.0090), (0.0090, 0.0093, 0.0094), (0.0092, 0.0094, 0.0096)))
NAME = index_name("USD", 6)
CONFIG = SensitivityConfig()


def _market(asof):
    return Market(asof, {"USD": CurrencyMarket(ZeroCurveConfig(PILLARS, DISC), {NAME: ZeroCurveConfig(PILLARS, INDEX)},
                                               VOLS)})


def _history(cfg, asof):
    swap = _build_ore_swap(cfg)
    dates = [ORE.as_floating_rate_coupon(c).fixingDate() for c in swap.floatingLeg()]
    return {d: 0.021 + 0.0007 * i for i, d in enumerate(d for d in dates if d < asof)}


def _swap(asof, **overrides):
    fields = dict(notional=1e6, fixed_rate=0.031, payer=True, effective_date=ORE.Date(3, 2, 2026),
                  maturity_date=ORE.Date(3, 2, 2031), evaluation_date=asof)
    fields.update(overrides)
    cfg = SwapConfig(**fields)
    return dataclasses.replace(cfg, fixings=_history(cfg, asof))


ASOF = ORE.Date(30, 7, 2026)


def test_delta_and_gamma_are_the_forward_difference_of_the_npv():
    cfg = _swap(ASOF)
    market = _market(ASOF)
    greeks = portfolio_sensitivities([cfg], market, "USD", config=CONFIG)[0]
    base = sensitivity_context(market, CONFIG)
    legs = legs_of(_build_ore_swap(cfg), True, ASOF, cfg.fixings)
    index_curve = base.index[NAME]

    def npv(disc_logs):
        return today_npv(legs, ASOF, DiscountCurve(base.discount["USD"].times, disc_logs), index_curve)

    logs = base.discount["USD"].log_discounts
    first = jax.grad(npv)(logs)
    second = jnp.diagonal(jax.hessian(npv)(logs))
    shift = CONFIG.curve_shift * np.asarray(base.discount["USD"].times)   # the bump in ln P at each tenor
    expected_delta = -np.asarray(first) * shift + 0.5 * np.asarray(second) * shift ** 2
    np.testing.assert_allclose(greeks["delta:discount:USD"], expected_delta[1:], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(greeks["gamma:discount:USD"], (np.asarray(second) * shift ** 2)[1:],
                               rtol=1e-4, atol=1e-9)


def test_theta_rolls_one_calendar_day_from_a_friday():
    friday = ORE.Date(31, 7, 2026)
    assert theta_context(_market(friday), CONFIG).date == ORE.Date(1, 8, 2026)


def test_theta_adds_back_a_coupon_paid_on_the_theta_date():
    """The swap's fixed and floating coupons pay on 2026-08-03 (a Monday): valued on the
    Sunday before, the Theta date is the payment date. Without the add-back Theta would be
    about minus the net coupon; with it only the day's carry remains (I-39)."""
    asof = ORE.Date(2, 8, 2026)
    cfg = _swap(asof, effective_date=ORE.Date(3, 8, 2025), maturity_date=ORE.Date(3, 8, 2030))
    market = _market(asof)
    theta = float(portfolio_sensitivities([cfg], market, "USD", config=CONFIG)[0]["theta"])
    base = value_on(cfg, sensitivity_context(market, CONFIG), PricingConfig())
    rolled = value_on(cfg, theta_context(market, CONFIG), PricingConfig())
    coupon = rolled - base - theta   # minus the flow the theta adds back
    assert abs(coupon) > 1_000.0
    assert abs(theta) < 0.05 * abs(coupon)


def test_a_fixing_on_the_as_of_date_is_backfilled_on_the_theta_date():
    """The coupon fixing on the as-of date is history on the Theta date: FixingManager sets
    it to the index forecast off the Theta market (ORE refuses a missing past fixing)."""
    swap = _swap(ASOF)
    fixing_dates = [ORE.as_floating_rate_coupon(c).fixingDate() for c in _build_ore_swap(swap).floatingLeg()]
    asof = next(d for d in fixing_dates if d > ASOF)
    cfg = _swap(asof)
    assert asof not in cfg.fixings
    theta = theta_context(_market(asof), CONFIG)
    assert asof in theta.fixings[NAME]
    assert np.isfinite(float(portfolio_sensitivities([cfg], _market(asof), "USD", config=CONFIG)[0]["theta"]))


def test_the_vega_matrix_adds_up_to_a_parallel_bump():
    cfg = SwaptionConfig(notional=1e6, fixed_rate=0.034, payer=True, swap_tenor="4Y",
                         forward_start=ORE.Period(2, ORE.Years), evaluation_date=ASOF)
    market = _market(ASOF)
    vega = portfolio_sensitivities([cfg], market, "USD", config=CONFIG)[0]["vega:USD"]
    shifted = dataclasses.replace(VOLS, vols=tuple(tuple(v + CONFIG.vol_shift for v in row) for row in VOLS.vols))
    bumped_market = Market(ASOF, {"USD": dataclasses.replace(market.currency("USD"), swaption_vols=shifted)})
    parallel = (value_on(cfg, sensitivity_context(bumped_market, CONFIG), PricingConfig())
                - value_on(cfg, sensitivity_context(market, CONFIG), PricingConfig()))
    assert np.count_nonzero(vega) >= 2
    assert float(np.sum(vega)) == pytest.approx(parallel, rel=1e-3)


def test_theta_adds_back_a_bond_coupon_paid_on_the_theta_date():
    """A bond's coupon paid on the Theta date is added back too (I-39 on the market path)."""
    from engine.instruments.treasury import BondConfig, CouponPeriod
    asof = ORE.Date(14, 8, 2026)
    periods = (CouponPeriod(ORE.Date(15, 2, 2026), ORE.Date(15, 8, 2026)),
               CouponPeriod(ORE.Date(15, 8, 2026), ORE.Date(15, 2, 2027)))
    bond = BondConfig(face_amount=1e5, maturity_date=ORE.Date(15, 2, 2027), evaluation_date=asof, coupon_rate=0.04,
                      coupon_schedule=periods)
    market = _market(asof)
    theta = float(portfolio_sensitivities([bond], market, "USD", config=CONFIG)[0]["theta"])
    rolled = value_on(bond, theta_context(market, CONFIG), PricingConfig())
    base = value_on(bond, sensitivity_context(market, CONFIG), PricingConfig())
    assert theta == pytest.approx(rolled - base + 2_000.0, abs=1e-8)
    assert 0.0 < theta < 50.0
