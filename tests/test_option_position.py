"""
An option's position (I-101): ORE's `OptionData/LongShort`, the option configs' `long_short`.

ORE books a short option as `<LongShort>Short</LongShort>` on a positive notional, and its
`OptionWrapper` multiplies the whole NPV by -1, the option's and, once exercised, its
underlying's, while the exercise stays the holder's (OREData/ored/portfolio/optionwrapper.cpp).
So a short option is exactly minus the long one: today under every engine, on every path and
date, in every Greek and in every market-risk scenario. A sign on the notional booked a short
European but a long Bermudan or American on the reversed swap (on the shared portfolio's
market the Bermudan payer at notional -1e6 was +15,017.85, where the short is -35,712.95), and a
zero notional priced a European to NaN, so an option's notional must be positive.

Against ORE itself: tests/test_shared_portfolio.py (each option booked short through ORE's own
trade builder, today) and tests/test_ore_xva_parity.py (on ORE's paths).
"""
import dataclasses

import numpy as np
import ORE
import pytest

from engine.api.requests import MarketPortfolioRequestSchema
from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig
from engine.market_simulation.config import CamConfig, LgmConfig, build_cross_asset_model, simulate
from engine.pricing.config import JamshidianEngineConfig, LgmSwaptionEngineConfig, PricingConfig
from engine.pricing.cube import OPTIONS, value_paths, value_today
from engine.precision import Precision, StagePrecision
from engine.risk.greeks.ad import portfolio_greeks, trade_greeks
from engine.risk.greeks.bump import portfolio_sensitivities
from engine.risk.market import MarketRiskRequest, RateRiskFactors, monte_carlo_scenarios, run_market_risk
from tests.support import portfolio as shared
from tests.support.compiles import count_compiles

FAST = LgmSwaptionEngineConfig(n_per_std=12, std_devs=4.0)
PRICING = PricingConfig(bermudan=FAST, american=FAST)
JAMSHIDIAN = PricingConfig(european="Jamshidian", jamshidian=JamshidianEngineConfig(0.03, 0.01),
                           bermudan=FAST, american=FAST)
#: The shared portfolio's options, by name: two Europeans (physical, and cash with a spread),
#: two Bermudans (physical payer, cash receiver) and an American.
OPTION_NAMES = [name for name, cfg in shared.trades().items() if isinstance(cfg, OPTIONS)]
TODAY = shared.ASOF


def short(cfg):
    """The same option, sold: `Short` on the same positive notional, its own id."""
    return dataclasses.replace(cfg, long_short="Short", trade_id=f"{cfg.trade_id}-short")


def pairs(names=OPTION_NAMES):
    """Each named option long, then short, side by side."""
    longs = [shared.trades()[n] for n in names]
    return [t for cfg in longs for t in (cfg, short(cfg))]


# --- the booking ----------------------------------------------------------------------------------

@pytest.mark.parametrize("cls", OPTIONS, ids=lambda c: c.__name__)
class TestTheBooking:
    BOOKINGS = {
        SwaptionConfig: dict(fixed_rate=0.03, payer=True, swap_tenor="5Y", forward_start=ORE.Period(1, ORE.Years)),
        BermudanSwaptionConfig: dict(fixed_rate=0.03, payer=True, swap_tenor="5Y",
                                     exercise_dates=[TODAY + 400, TODAY + 800]),
        AmericanSwaptionConfig: dict(fixed_rate=0.03, payer=True, swap_tenor="5Y", first_exercise_date=TODAY + 400,
                                     last_exercise_date=TODAY + 800),
    }

    def _book(self, cls, **fields):
        return cls(**{"notional": 1e6, **self.BOOKINGS[cls], **fields}, evaluation_date=TODAY, trade_id="opt")

    def test_long_is_the_default_and_short_is_booked_by_name(self, cls):
        assert self._book(cls).long_short == "Long"
        assert self._book(cls, long_short="Short").long_short == "Short"

    @pytest.mark.parametrize("notional", [0.0, -1e6], ids=["zero", "negative"])
    def test_a_zero_or_negative_notional_is_refused_naming_the_trade(self, cls, notional):
        """Before the fix a negative notional was a short European but a long Bermudan or
        American on the reversed swap, and a zero notional a NaN European."""
        with pytest.raises(ValueError, match=r"trade 'opt'.*notional must be positive.*long_short='Short'"):
            self._book(cls, notional=notional)

    @pytest.mark.parametrize("position", ["long", "Sell", "", None])
    def test_a_position_other_than_long_or_short_is_refused(self, cls, position):
        with pytest.raises(ValueError, match=r"trade 'opt'.*long_short must be one of \('Long', 'Short'\)"):
            self._book(cls, long_short=position)


def test_a_swap_keeps_its_signed_notional():
    """A swap's notional sign is the reversed swap, as ORE's is: unchanged."""
    cfg = SwapConfig(notional=-1e6, fixed_rate=0.03, payer=True, swap_tenor="5Y", evaluation_date=TODAY, trade_id="s")
    assert cfg.notional == -1e6 and not hasattr(cfg, "long_short")


# --- today ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("pricing", [PricingConfig(), JAMSHIDIAN], ids=["Bachelier", "Jamshidian"])
def test_a_short_option_today_is_minus_the_long_one_in_every_engine(pricing):
    """Bit for bit: the engines price the long option and the position multiplies by -1. The
    Jamshidian engine refuses a spread and cash settlement, as QuantLib's does."""
    names = OPTION_NAMES if pricing.european == "Bachelier" else \
        [n for n in OPTION_NAMES if n != "european-receiver-otm-cash"]
    values = value_today(pairs(names), shared.market(), "USD", pricing)
    longs, shorts = np.asarray(values[0::2]), np.asarray(values[1::2])
    assert np.all(longs > 0.0)
    np.testing.assert_array_equal(shorts, -longs)


# --- on the paths ---------------------------------------------------------------------------------

def _scenarios(samples=64):
    config = CamConfig(dates=tuple(TODAY + ORE.Period(m, ORE.Months) for m in (3, 7, 13, 25, 37)),
                       base_currency="USD", ir={"USD": LgmConfig(0.03, 0.01)}, samples=samples, seed=5)
    market = shared.market()
    return simulate(market, config, build_cross_asset_model(market, config))


@pytest.fixture(scope="module")
def scenarios():
    return _scenarios()


@pytest.mark.slow
@pytest.mark.parametrize("stage", ["float64", "float32"])
def test_a_short_option_on_every_path_is_minus_the_long_one(scenarios, stage):
    """The exercise is the holder's: a short option is minus the long one on every path and
    date, before and after exercise (a physical one then its swap, a cash one 0), in either
    plain arithmetic. A short option priced as the long option on the reversed swap would
    exercise where the long one does not."""
    precision = Precision(pricing=StagePrecision(stage, stage, stage))
    columns = value_paths(pairs(), shared.market(), scenarios, "USD", PRICING, precision=precision)
    for long, sold, name in zip(columns[0::2], columns[1::2], OPTION_NAMES):
        long, sold = np.asarray(long), np.asarray(sold)
        assert np.any(long > 0.0) and np.any(long[:, -1] != long[:, 0]), name
        np.testing.assert_array_equal(sold, -long, err_msg=name)


@pytest.mark.slow
def test_a_short_option_after_its_long_one_compiles_nothing(scenarios):
    """The position is data, not a static setting: a short option reuses the long one's
    programs on the paths and in the AD Greeks."""
    market, cfg = shared.market(), shared.trades()["bermudan-payer-physical"]
    value_paths([cfg], market, scenarios, "USD", PRICING)
    trade_greeks(cfg, market, PRICING)
    with count_compiles() as counter:
        value_paths([short(cfg)], market, scenarios, "USD", PRICING)
        trade_greeks(short(cfg), market, PRICING)
    assert sum(counter.values()) == 0, dict(counter)


# --- in the Greeks --------------------------------------------------------------------------------

def _assert_minus(greeks):
    for i in range(0, len(greeks), 2):
        long, sold = greeks[i], greeks[i + 1]
        assert long.keys() == sold.keys()
        for key in long:
            np.testing.assert_array_equal(sold[key], -np.asarray(long[key]), err_msg=f"trade {i // 2}: {key}")
        assert any(np.any(np.asarray(v) != 0.0) for v in long.values())


@pytest.mark.slow
def test_a_short_options_ad_greeks_are_minus_the_long_ones():
    """Delta, Gamma, Vega (a Bermudan's through its calibration) and Theta."""
    _assert_minus(portfolio_greeks(pairs(), shared.market(), "USD", PRICING))


@pytest.mark.slow
def test_a_short_options_bump_greeks_are_minus_the_long_ones():
    """ORE's bump-and-revalue sensitivities: each bumped value is the long one's negated."""
    names = ["european-payer", "bermudan-payer-physical"]
    _assert_minus(portfolio_sensitivities(pairs(names), shared.market(), "USD", PRICING))


# --- in market risk -------------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("stage", ["float64", "float32"])
def test_a_short_options_pnl_is_minus_the_long_ones(stage):
    """In difference form (the Europeans) and in full revaluation (the Bermudans and the
    American), in either plain arithmetic."""
    market = shared.market()
    factors = RateRiskFactors.from_market(market)
    shocks = monte_carlo_scenarios(factors, (1e-3 ** 2) * np.eye(factors.size), 10, 16, seed=3)
    trades = pairs()
    precision = Precision(pricing=StagePrecision(stage, stage, stage),
                          simulation=StagePrecision(stage, stage, stage))
    result = run_market_risk(MarketRiskRequest(trades, market, shocks, PRICING, precision=precision))
    base, pnl = np.asarray(result.base_npv_per_trade), np.asarray(result.pnl)
    np.testing.assert_array_equal(base[1::2], -base[0::2])
    np.testing.assert_array_equal(pnl[:, 1::2], -pnl[:, 0::2])
    assert np.all(np.any(pnl[:, 0::2] != 0.0, axis=0))
    assert result.base_npv == pytest.approx(0.0, abs=1e-6)


# --- over the API ---------------------------------------------------------------------------------

def _body(**european):
    trade = {"trade_type": "european_swaption", "notional": 2e6, "fixed_rate": 0.044, "payer": True,
             "swap_tenor": "5Y", "forward_start": "1Y", "trade_id": "e", **european}
    return {"market": shared.market_json(), "trades": [trade], "scenario_risk": False}


def test_the_api_books_a_short_option_by_long_short():
    request = MarketPortfolioRequestSchema(**_body(long_short="Short")).to_dataclass()
    assert request.trades[0].long_short == "Short"
    assert MarketPortfolioRequestSchema(**_body()).to_dataclass().trades[0].long_short == "Long"


def test_over_http_a_negative_option_notional_is_a_400_naming_the_trade(test_client):
    r = test_client.post("/portfolio/price", json=_body(notional=-2e6))
    assert r.status_code == 400, r.text
    assert "trade 'e'" in r.json()["detail"] and "long_short='Short'" in r.json()["detail"]


def test_over_http_an_unknown_position_is_a_422(test_client):
    r = test_client.post("/portfolio/price", json=_body(long_short="Sell"))
    assert r.status_code == 422 and "long_short" in r.text
