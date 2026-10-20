"""
ORE's sensitivity analysis on today's market (`engine.risk.greeks.bump`, plan T-18/T-19).

  * The bump machinery: a forward-difference Delta at a tenor equals AD's first derivative
    times the shift plus half the second-derivative term, and Gamma the second derivative
    term, to O(shift^3) (plan X-3: AD and ORE's Greeks agree to the bump's order).
  * Theta: the date moves one calendar day (Friday -> Saturday, I-38), a coupon paid on the
    Theta date leaves Theta as it leaves the NPV, as in ORE's analytic (no flow is added back:
    `engine.risk.greeks.bump`'s module docstring; I-39's add-back reversed on 2026-10-18,
    I-116), and a fixing of the as-of date is backfilled from the Theta market rather than
    refused.
  * Vega: the matrix of quote bumps adds up to a parallel volatility bump.
  * One batched revaluation (I-115): a trade's shifts are scenarios of one revaluation of its
    price function, each Greek equal to the values on the shifted and the base market, valued
    one shift at a time, and only Theta values the trade on a market of its own.

Parity with ORE's own sensitivity analytic is tests/test_sensitivity_ore_parity.py (the shared
portfolio) and tests/test_ore_parity_sweep.py (seeded random portfolios, settings and rolls).
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig, underlying_swap
from engine.instruments.treasury import BondConfig
from engine.market_data.curves import DiscountCurve
from engine.market_data.market import CurrencyMarket, Market, SwaptionVolSurface, ZeroCurveConfig, index_name
from engine.pricing.config import LgmSwaptionEngineConfig, PricingConfig
from engine.pricing.cube import reads_swaption_vols, value_on
from engine.pricing.legs import legs_of, today_npv
from engine.risk.greeks.bump import (
    SensitivityConfig, portfolio_sensitivities, sensitivity_context, theta_context,
)
from tests.support import portfolio as shared

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
    swap = underlying_swap(cfg)
    dates = [ORE.as_floating_rate_coupon(c).fixingDate() for c in swap.floatingLeg()]
    return {d: 0.021 + 0.0007 * i for i, d in enumerate(d for d in dates if d < asof)}


def _swap(asof, **overrides):
    fields = dict(notional=1e6, fixed_rate=0.031, payer=True, effective_date=ORE.Date(3, 2, 2026),
                  maturity_date=ORE.Date(3, 2, 2031), evaluation_date=asof, trade_id="swap")
    fields.update(overrides)
    cfg = SwapConfig(**fields)
    return dataclasses.replace(cfg, fixings=_history(cfg, asof))


ASOF = ORE.Date(30, 7, 2026)


def test_delta_and_gamma_are_the_forward_difference_of_the_npv():
    cfg = _swap(ASOF)
    market = _market(ASOF)
    greeks = portfolio_sensitivities([cfg], market, "USD", config=CONFIG)[0]
    base = sensitivity_context(market, CONFIG)
    legs = legs_of(underlying_swap(cfg), True, ASOF, cfg.fixings)
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


def test_a_coupon_paid_on_the_theta_date_leaves_theta_as_it_leaves_the_npv():
    """The swap's fixed and floating coupons pay on 2026-08-03 (a Monday): valued on the
    Sunday before, the Theta date is the payment date, where the coupons have occurred. ORE's
    Theta adds back no flow (its period flows are read after the roll), so Theta is the rolled
    value less the base, about minus the net coupon (until 2026-10-18 the engine added the
    coupons back, I-39, leaving only the day's carry; ORE's analytic disproved it, I-116)."""
    asof = ORE.Date(2, 8, 2026)
    cfg = _swap(asof, effective_date=ORE.Date(3, 8, 2025), maturity_date=ORE.Date(3, 8, 2030))
    market = _market(asof)
    theta = float(portfolio_sensitivities([cfg], market, "USD", config=CONFIG)[0]["theta"])
    base = value_on(cfg, sensitivity_context(market, CONFIG), PricingConfig())
    rolled = value_on(cfg, theta_context(market, CONFIG), PricingConfig())
    assert theta == rolled - base
    assert abs(theta) > 1_000.0


def test_a_fixing_on_the_as_of_date_is_backfilled_on_the_theta_date():
    """The coupon fixing on the as-of date is history on the Theta date: FixingManager sets
    it to the index forecast off the Theta market (ORE refuses a missing past fixing)."""
    swap = _swap(ASOF)
    fixing_dates = [ORE.as_floating_rate_coupon(c).fixingDate() for c in underlying_swap(swap).floatingLeg()]
    asof = next(d for d in fixing_dates if d > ASOF)
    cfg = _swap(asof)
    assert asof not in cfg.fixings
    theta = theta_context(_market(asof), CONFIG)
    assert asof in theta.fixings[NAME]
    assert np.isfinite(float(portfolio_sensitivities([cfg], _market(asof), "USD", config=CONFIG)[0]["theta"]))


def test_the_vega_matrix_adds_up_to_a_parallel_bump():
    cfg = SwaptionConfig(trade_id="european-L120", notional=1e6, fixed_rate=0.034, payer=True, swap_tenor="4Y",
                         forward_start=ORE.Period(2, ORE.Years), evaluation_date=ASOF)
    market = _market(ASOF)
    vega = portfolio_sensitivities([cfg], market, "USD", config=CONFIG)[0]["vega:USD"]
    shifted = dataclasses.replace(VOLS, vols=tuple(tuple(v + CONFIG.vol_shift for v in row) for row in VOLS.vols))
    bumped_market = Market(ASOF, {"USD": dataclasses.replace(market.currency("USD"), swaption_vols=shifted)})
    parallel = (value_on(cfg, sensitivity_context(bumped_market, CONFIG), PricingConfig())
                - value_on(cfg, sensitivity_context(market, CONFIG), PricingConfig()))
    assert np.count_nonzero(vega) >= 2
    assert float(np.sum(vega)) == pytest.approx(parallel, rel=1e-3)


def test_a_bond_coupon_paid_on_the_theta_date_leaves_theta_too():
    """A bond's coupon paid on the Theta date leaves Theta as the swap's does: about minus the
    2,000 coupon, plus a day's carry."""
    from engine.instruments.treasury import BondConfig, CouponPeriod
    asof = ORE.Date(14, 8, 2026)
    periods = (CouponPeriod(ORE.Date(15, 2, 2026), ORE.Date(15, 8, 2026)),
               CouponPeriod(ORE.Date(15, 8, 2026), ORE.Date(15, 2, 2027)))
    bond = BondConfig(trade_id="bond-L138", face_amount=1e5, maturity_date=ORE.Date(15, 2, 2027), evaluation_date=asof, coupon_rate=0.04,
                      coupon_schedule=periods)
    market = _market(asof)
    theta = float(portfolio_sensitivities([bond], market, "USD", config=CONFIG)[0]["theta"])
    rolled = value_on(bond, theta_context(market, CONFIG), PricingConfig())
    base = value_on(bond, sensitivity_context(market, CONFIG), PricingConfig())
    assert theta == rolled - base
    assert -2_000.0 < theta < -1_950.0


# ---------------------------------------------------------------------------
# One batched revaluation (I-115)
# ---------------------------------------------------------------------------
#: The grid engine at a resolution a fast test affords, calibrated and on a fixed model.
SMALL_GRID = LgmSwaptionEngineConfig(n_per_std=8, std_devs=5.0)
BATCHED = PricingConfig(bermudan=SMALL_GRID, american=SMALL_GRID)
FIXED = PricingConfig(bermudan=dataclasses.replace(SMALL_GRID, calibration="None"))


def _batched_trades():
    """One trade of each price function the bump Greeks revalue: legs, Bachelier, a bond, and
    calibrated options (the shared portfolio's, on `SMALL_GRID`)."""
    shared_trades = shared.trades()
    return [shared_trades[name] for name in ("swap-receiver-seasoned-icma", "european-receiver-otm-cash", "bond",
                                             "bermudan-payer-physical", "american-payer")]


def _one_shift_at_a_time(cfg, market, pricing, config):
    """The bump Greeks as ORE computes them and the engine did until 2026-10-19: the trade valued
    on its sensitivity market with one tenor's zero rate, or one quote, shifted at a time
    (`value_on`), each Greek a difference of two values."""
    base_context = sensitivity_context(market, config)
    value = lambda context: value_on(cfg, context, pricing)  # noqa: E731
    base = value(base_context)
    greeks = {}
    keys = [("discount", cfg.currency)]
    if not isinstance(cfg, BondConfig):
        keys.append(("index", index_name(cfg.currency, cfg.index_tenor_months)))
    for kind, key in keys:
        curve = getattr(base_context, kind)[key]

        def shifted(k, shift, _kind=kind, _key=key, _curve=curve):
            logs = np.asarray(_curve.log_discounts).copy()
            logs[k] -= shift * float(_curve.times[k])
            table = {**getattr(base_context, _kind), _key: DiscountCurve(_curve.times, jnp.asarray(logs))}
            return value(dataclasses.replace(base_context, **{_kind: table}))

        ups = np.array([shifted(k, config.curve_shift) for k in range(1, len(curve.times))])
        downs = np.array([shifted(k, -config.curve_shift) for k in range(1, len(curve.times))])
        greeks[f"delta:{kind}:{key}"] = ups - base
        greeks[f"gamma:{kind}:{key}"] = ups - 2.0 * base + downs
    if reads_swaption_vols(cfg, pricing):
        surface = market.swaption_vols(cfg.currency)
        vega = np.zeros((len(surface.option_tenors), len(surface.swap_tenors)))
        for i, j in np.ndindex(vega.shape):
            vols = np.array(surface.vols)
            vols[i, j] += config.vol_shift
            bumped = dataclasses.replace(surface, vols=tuple(map(tuple, vols)))
            reading = lambda ccy, t, length, _b=bumped: float(_b.volatility(market.asof, t, length))  # noqa: E731
            vega[i, j] = value(dataclasses.replace(base_context, volatility=reading)) - base
        greeks[f"vega:{cfg.currency}"] = vega
    return base, greeks


class TestOneBatchedRevaluation:
    """A trade's shifts are the scenarios of one revaluation of its price function on the
    sensitivity market (`engine.risk.greeks.scenario_pnl`, market risk's), an option recalibrated
    under each, rather than one valuation per shift on the host."""

    def test_no_shift_is_valued_on_its_own(self, monkeypatch):
        """Only Theta values the trade on a market of its own: its base and its roll. One
        valuation per shift was about sixty host round trips per trade, which left an accelerator
        idle between small programs (I-115)."""
        from engine.risk.greeks import bump
        calls = []
        monkeypatch.setattr(bump, "value_on", lambda *a, **k: calls.append(a[0]) or value_on(*a, **k))
        trades = _batched_trades()
        greeks = portfolio_sensitivities(trades, shared.market(), "USD", BATCHED, CONFIG)
        assert len(greeks) == len(trades)
        assert len(calls) == 2 * len(trades), len(calls)

    @pytest.mark.parametrize("pricing", [BATCHED, FIXED], ids=["calibrated", "fixed"])
    def test_each_shift_is_the_value_on_its_shifted_market(self, pricing):
        """Every Delta, Gamma and Vega equals the difference of the trade's values on the shifted
        and the base sensitivity market, valued one shift at a time, to 1e-13 of its NPV (the
        difference form carries the shift exactly; the values differ in their last bits)."""
        market = shared.market()
        trades = _batched_trades()
        batched = portfolio_sensitivities(trades, market, "USD", pricing, CONFIG)
        for index, cfg in enumerate(trades):
            base, expected = _one_shift_at_a_time(cfg, market, pricing, CONFIG)
            mine = {key: value for key, value in batched[index].items() if key != "theta"}
            assert mine.keys() == expected.keys(), cfg.trade_id
            for key in expected:
                np.testing.assert_allclose(mine[key], expected[key], rtol=0.0, atol=1e-13 * abs(base),
                                           err_msg=f"{cfg.trade_id} {key}")
