"""
`price_portfolio` on a `Market` (`engine.run.pipeline`): the ORE-aligned path end to
end -- CAM calibration and simulation, every trade on every path with its t=0 engine, exposure
with the LGM numeraire, ORE's sensitivities.

The trade-by-trade ORE parity of each piece is in tests/test_cam.py, test_pricing.py and
test_ore_lgm_calibration.py; here the pieces are checked to be assembled correctly, and the
exposure numbers against identities that hold only if the numeraire and deflation are right.
"""
import dataclasses

import numpy as np
import ORE
import pytest

from engine.instruments.swap import SwapConfig
from engine.instruments.treasury import BondConfig, CouponPeriod
from engine.market_data.curves import ZeroCurve, discount
from engine.market_data.day_counts import TIME_AXIS_DAY_COUNTER as DC
from engine.market_data.market import CurrencyMarket, Market, SwaptionVolSurface, ZeroCurveConfig, index_name
from engine.market_simulation.config import CamConfig, LgmConfig, simulate
from engine.pricing.config import LgmSwaptionEngineConfig, PricingConfig
from engine.pricing.cube import value_portfolio, value_today
from engine.run import PortfolioRequest, RunConfig, price_portfolio

ASOF = ORE.Date(30, 7, 2026)
PILLARS = [0.0, 1.0, 2.0, 5.0, 10.0, 30.0]
DISC = [0.020, 0.020, 0.025, 0.030, 0.035, 0.040]
INDEX = [0.025, 0.025, 0.031, 0.036, 0.040, 0.044]
VOLS = SwaptionVolSurface(("1Y", "5Y", "10Y"), ("1Y", "5Y", "10Y"),
                          ((0.0080, 0.0088, 0.0090), (0.0090, 0.0093, 0.0094), (0.0092, 0.0094, 0.0096)))
FAST = LgmSwaptionEngineConfig(n_per_std=12, std_devs=4.0)
PRICING = PricingConfig(bermudan=FAST, american=FAST)
CONFIG = RunConfig(pricing=PRICING)
DATES = tuple(ASOF + ORE.Period(m, ORE.Months) for m in (3, 6, 12, 24, 36))


def _market():
    return Market(ASOF, {"USD": CurrencyMarket(
        ZeroCurveConfig(PILLARS, DISC), {index_name("USD", 6): ZeroCurveConfig(PILLARS, INDEX)}, VOLS)})


def _simulation(samples=512):
    return CamConfig(dates=DATES, base_currency="USD",
                     ir={"USD": LgmConfig(0.03, 0.01, ("1Y", "2Y", "5Y"), ("9Y", "8Y", "5Y"))},
                     samples=samples, seed=3)


def _bond():
    periods = tuple(CouponPeriod(ORE.Date(15, 2 if h == 0 else 8, y), ORE.Date(15, 8 if h == 0 else 2, y + h))
                    for y in (2026, 2027, 2028) for h in (0, 1))
    return BondConfig(trade_id="bond-L52", face_amount=1e6, maturity_date=ORE.Date(15, 2, 2029), evaluation_date=ASOF,
                      coupon_rate=0.03, coupon_schedule=periods)


def _bill():
    return BondConfig(trade_id="bond-L57", face_amount=1e6, maturity_date=DATES[2] + ORE.Period(1, ORE.Years), evaluation_date=ASOF)


def _trades():
    from engine.instruments.bermudan_swaption import BermudanSwaptionConfig, exercisable_dates
    from engine.instruments.european_swaption import SwaptionConfig
    booked = BermudanSwaptionConfig(trade_id="bermudan-L63", notional=1e6, fixed_rate=0.03, payer=False, exercise_dates=[ASOF + 400],
                                    swap_tenor="5Y", evaluation_date=ASOF)
    return [
        SwapConfig(trade_id="swap-L66", notional=1e6, fixed_rate=0.031, payer=True, swap_tenor="5Y", evaluation_date=ASOF),
        SwaptionConfig(trade_id="european-L67", notional=1e6, fixed_rate=0.032, payer=True, swap_tenor="4Y",
                       forward_start=ORE.Period(1, ORE.Years), evaluation_date=ASOF),
        dataclasses.replace(booked, exercise_dates=exercisable_dates(booked)[1:-1]),
        _bond(),
        _bill(),
    ]


@pytest.fixture(scope="module")
def result():
    request = PortfolioRequest(market=_market(), trades=_trades(), 
                               config=dataclasses.replace(CONFIG, simulation=_simulation()))
    return price_portfolio(request)


def test_the_result_is_the_valuation_layers(result):
    market = _market()
    scenarios = simulate(market, _simulation())
    direct = value_portfolio(_trades(), market, scenarios, "USD", PRICING)
    assert result.base_npv_per_trade == pytest.approx(direct.today, rel=1e-12)
    np.testing.assert_allclose(np.asarray(result.npv_cube), np.asarray(direct.cube), rtol=1e-12)
    assert result.base_npv == pytest.approx(sum(direct.today), rel=1e-12)
    assert result.measure is not None and result.scenario_risk_available
    assert len(result.trade_exposures) == len(_trades())


def test_a_bills_expected_exposure_is_its_forward_value(result):
    """A long bill is always worth more than 0, so EE_B(t) = E[V(t)/N(t)] / P(0,t); and V/N is
    a martingale, so EE_B(t) = face * P(0,T) / P(0,t). Exact only where the maturity is a
    simulation-market tenor point (between them the scenario curve interpolates, as ORE's
    does): here the bill matures one year after the 12M date. Holds only if the numeraire is
    the LGM numeraire and the cube is deflated by it."""
    profile = result.trade_exposures[4]
    curve = ZeroCurve.from_config(ZeroCurveConfig(PILLARS, DISC))
    t, T = (DC.yearFraction(ASOF, d) for d in (DATES[2], _bill().maturity_date))
    forward = _bill().face_amount * float(discount(curve, T)) / float(discount(curve, t))
    # Monte Carlo error, not bias. Measured 2026-09-29 (seeds 3 and 4): -1.4e-5 and +4.9e-6 at
    # 512 paths, falling to ~2e-7 at 32768 with the sign changing between seeds; plain-MC
    # standard errors are 8.8e-4 at 512 (Sobol does far better on this integrand).
    assert float(profile.ee_b[3]) == pytest.approx(forward, rel=1e-4)
    assert float(profile.ee_b[4]) == 0.0   # paid on the 24M date


def test_ores_time_weighted_and_basel_profiles_are_reported(result):
    profile = result.exposure
    assert profile.epe_b.shape == profile.ee_b.shape
    assert np.all(np.asarray(profile.eepe_b) >= np.asarray(profile.epe_b) - 1e-9)
    assert profile.basel_epe is not None and profile.basel_eepe >= profile.basel_epe - 1e-9
    # The one-year horizon ends on the 12M date (index 3 of the profile, after t=0, 3M, 6M).
    assert profile.basel_epe == pytest.approx(float(profile.epe_b[3]))


def test_without_scenario_risk_only_todays_npvs(result):
    request = PortfolioRequest(market=_market(), trades=_trades(), scenario_risk=False, config=CONFIG)
    today_only = price_portfolio(request)
    assert today_only.npv_cube.size == 0 and today_only.exposure is None
    assert today_only.base_npv_per_trade == pytest.approx(value_today(_trades(), _market(), "USD", PRICING),
                                                          rel=1e-12)


def test_scenario_risk_needs_a_simulation():
    with pytest.raises(ValueError, match="CamConfig"):
        price_portfolio(PortfolioRequest(market=_market(), trades=_trades()))


def test_greeks_are_oresstyle_sensitivities(result):
    request = PortfolioRequest(market=_market(), trades=_trades()[:2], scenario_risk=False, compute_greeks=True,
                               config=CONFIG)
    greeks = price_portfolio(request).greeks
    assert set(greeks[0]) == {"delta:discount:USD", "gamma:discount:USD", "delta:index:USD-SIMINDEX-6M",
                              "gamma:index:USD-SIMINDEX-6M", "theta"}
    assert "vega:USD" in greeks[1] and greeks[1]["vega:USD"].shape == (3, 3)


def test_every_result_carries_the_trade_ids():
    """I-10: each trade's id comes back on the result, in request order, whatever the model."""
    trades = [dataclasses.replace(t, trade_id=f"t{k}") for k, t in enumerate(_trades())]
    result = price_portfolio(PortfolioRequest(market=_market(), trades=trades, scenario_risk=False, config=CONFIG))
    assert result.trade_ids == [f"t{k}" for k in range(len(trades))]


def test_a_repeated_trade_id_is_refused():
    trades = [dataclasses.replace(t, trade_id="same") for t in _trades()[:2]]
    with pytest.raises(ValueError, match="unique"):
        PortfolioRequest(market=_market(), trades=trades)
