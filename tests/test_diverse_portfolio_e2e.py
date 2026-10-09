"""
A diverse portfolio (swaps of several tenors and directions, Europeans across expiries,
tenors and moneyness, Bermudans, an American, bills and notes) on the shared sloped two-curve
market, through one `price_portfolio` run under the Hull-White model calibrated to the
market's swaption volatilities.

  * Every trade's t=0 value equals ORE's (tests/support/portfolio.py `ore_npv`: ORE's
    discounting engines, QuantLib's Bachelier engine, the OREApp LGM engine).
  * Identities that hold path by path, whatever the model: a payer minus a receiver European
    on the same swap is the forward-starting swap (put-call parity) until expiry; offsetting
    positions net to zero; more exercise dates never lower a Bermudan's value.
  * The run is internally consistent: totals, ids, exposure and Greeks for every trade.

(Engine-vs-ORE on the simulated paths is tests/test_end_to_end.py; the Hull-White model's own
properties are tests/test_hull_white_model.py.)
"""
import dataclasses

import numpy as np
import ORE
import pytest

from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig
from engine.instruments.treasury import BondConfig
from engine.pricing.cube import value_today
from engine.run import (
    CamConfig, GreeksConfig, HullWhiteConfig, LgmSwaptionEngineConfig, PortfolioRequest, PricingConfig, RunConfig,
    price_portfolio,
)
from tests.support import portfolio as shared

ASOF = shared.ASOF
DATES = tuple(ASOF + ORE.Period(m, ORE.Months) for m in (6, 12, 18, 24, 36))
FAST = LgmSwaptionEngineConfig(n_per_std=16, std_devs=5.0)
PRICING = PricingConfig(bermudan=FAST, american=FAST)


def _swaps():
    return [SwapConfig(notional=n, fixed_rate=k, payer=p, swap_tenor=t, evaluation_date=ASOF, trade_id=f"swap-{t}-{i}")
            for i, (n, k, p, t) in enumerate([(5e6, 0.038, True, "2Y"), (3e6, 0.042, False, "5Y"),
                                              (1e7, 0.045, True, "7Y"), (2e6, 0.047, False, "10Y")])]


def _europeans():
    out = []
    for i, (fwd, tenor, strike, payer) in enumerate([(1, "2Y", 0.040, True), (1, "5Y", 0.044, False),
                                                    (2, "3Y", 0.030, True), (2, "5Y", 0.060, True),
                                                    (3, "5Y", 0.045, False)]):
        out.append(SwaptionConfig(notional=2e6, fixed_rate=strike, payer=payer, swap_tenor=tenor,
                                  forward_start=ORE.Period(fwd, ORE.Years), evaluation_date=ASOF,
                                  trade_id=f"european-{fwd}y{tenor}-{i}"))
    return out


def _bermudans():
    underlying = dict(notional=1e6, effective_date=ORE.Date(3, 2, 2027), maturity_date=ORE.Date(3, 2, 2032),
                      evaluation_date=ASOF)
    yearly = [ORE.Date(1, 2, 2027 + k) for k in range(4)]
    return [BermudanSwaptionConfig(fixed_rate=0.045, payer=True, exercise_dates=yearly, trade_id="bermudan-yearly",
                                   **underlying),
            BermudanSwaptionConfig(fixed_rate=0.045, payer=True, exercise_dates=yearly[-1:],
                                   trade_id="bermudan-last-only", **underlying),
            BermudanSwaptionConfig(fixed_rate=0.040, payer=False, exercise_dates=yearly, settlement="Cash",
                                   trade_id="bermudan-receiver-cash", **underlying)]


def _american():
    return AmericanSwaptionConfig(fixed_rate=0.045, payer=True, first_exercise_date=ORE.Date(1, 2, 2027),
                                  last_exercise_date=ORE.Date(1, 2, 2030), notional=1e6,
                                  effective_date=ORE.Date(3, 2, 2027), maturity_date=ORE.Date(3, 2, 2032),
                                  evaluation_date=ASOF, trade_id="american")


def _bonds():
    note = shared.trades()["bond"]
    return [note, dataclasses.replace(note, face_amount=-5e5, trade_id="short-note"),
            BondConfig(face_amount=1e6, maturity_date=ASOF + ORE.Period(9, ORE.Months), evaluation_date=ASOF,
                       trade_id="bill")]


def _trades():
    return _swaps() + _europeans() + _bermudans() + [_american()] + _bonds()


def _simulation(samples=256):
    return CamConfig(dates=DATES, base_currency="USD", samples=samples, seed=23,
                     ir={"USD": HullWhiteConfig(0.03, 0.01, ("1Y", "2Y", "5Y"), ("9Y", "8Y", "5Y"))})


@pytest.fixture(scope="module")
def result():
    request = PortfolioRequest(market=shared.market(), trades=_trades(),
                               config=RunConfig(simulation=_simulation(), pricing=PRICING))
    return price_portfolio(request)


def _column(result, trade_id):
    return np.asarray(result.npv_cube[:, :, result.trade_ids.index(trade_id)])


class TestEveryTradeIsOresAtT0:
    @pytest.mark.parametrize("cfg", _swaps() + _europeans() + _bonds(), ids=lambda c: c.trade_id)
    def test_swaps_europeans_and_bonds(self, cfg):
        assert value_today([cfg], shared.market(), "USD")[0] == pytest.approx(shared.ore_npv(cfg), rel=1e-10, abs=1e-6)

    @pytest.mark.slow
    @pytest.mark.parametrize("cfg", _bermudans() + [_american()], ids=lambda c: c.trade_id)
    def test_bermudans_and_americans_on_ores_calibrated_lgm(self, cfg):
        """With the default engine, against ORE's calibrated `NumericLgmMultiLegOptionEngine`
        (a cash-settled one as ORE prices it, as the physical; tests/support/portfolio.py)."""
        assert value_today([cfg], shared.market(), "USD")[0] == pytest.approx(shared.ore_npv(cfg), rel=1e-6)


@pytest.mark.slow
class TestThePortfolioRun:
    def test_every_trade_is_priced_today_and_on_every_path(self, result):
        cube = np.asarray(result.npv_cube)
        assert cube.shape == (256, len(DATES), len(_trades())) and np.all(np.isfinite(cube))
        assert result.trade_ids == [t.trade_id for t in _trades()]
        assert result.base_npv == pytest.approx(sum(result.base_npv_per_trade), rel=1e-14)

    def test_todays_values_are_the_engines_on_todays_market(self, result):
        np.testing.assert_allclose(result.base_npv_per_trade, value_today(_trades(), shared.market(), "USD", PRICING),
                                   rtol=1e-13)

    def test_every_trade_has_an_exposure(self, result):
        assert len(result.trade_exposures) == len(_trades())

    def test_one_trade_of_each_kind_has_greeks_and_only_options_a_vega(self):
        """Today's AD Greeks, one trade of each kind. (By bump and revalue the options
        recalibrate under every bump: measured 128 s for the Bermudan and 587 s for the
        American, against 31 s and 48 s by AD; I-53.)"""
        trades = [_swaps()[0], _europeans()[0], _bermudans()[0], _american(), _bonds()[2]]
        config = RunConfig(pricing=PRICING, greeks=GreeksConfig(method="AD"))
        greeks = price_portfolio(PortfolioRequest(market=shared.market(), trades=trades, compute_greeks=True,
                                                  scenario_risk=False, config=config)).greeks
        assert set(greeks) == set(range(len(trades)))
        assert all(np.all(np.isfinite(v)) for g in greeks.values() for v in g.values())
        for i, cfg in enumerate(trades):
            has_vega = any(k.startswith("vega") for k in greeks[i])
            assert has_vega == isinstance(cfg, (SwaptionConfig, BermudanSwaptionConfig, AmericanSwaptionConfig))

    def test_more_exercise_dates_never_lower_the_value(self, result):
        """Today and on every path before the first exercise."""
        yearly, last = _column(result, "bermudan-yearly"), _column(result, "bermudan-last-only")
        assert result.base_npv_per_trade[result.trade_ids.index("bermudan-yearly")] >= \
            result.base_npv_per_trade[result.trade_ids.index("bermudan-last-only")] - 1e-8
        assert np.all(yearly[:, :1] >= last[:, :1] - 1e-6 * np.abs(last[:, :1]).max())

    def test_offsetting_bonds_net_on_every_path(self, result):
        """A long and half its size short net to half the long, path by path."""
        np.testing.assert_allclose(_column(result, "bond") + 2 * _column(result, "short-note"), 0.0, atol=1e-6)


@pytest.mark.slow
class TestPutCallParityOnEveryPath:
    """Payer minus receiver European on the same swap, physically settled, is the forward
    swap: on every path and date before expiry, whatever the model, as long as both read the
    same curves and volatility (Bachelier: the same normal volatility)."""

    @pytest.fixture(scope="class")
    def parity(self):
        common = dict(notional=2e6, fixed_rate=0.043, swap_tenor="5Y", forward_start=ORE.Period(2, ORE.Years),
                      evaluation_date=ASOF)
        payer = SwaptionConfig(payer=True, trade_id="payer", **common)
        receiver = SwaptionConfig(payer=False, trade_id="receiver", **common)
        swap = SwapConfig(notional=2e6, fixed_rate=0.043, payer=True, effective_date=payer.effective_date,
                          maturity_date=payer.maturity_date, evaluation_date=ASOF, trade_id="swap")
        request = PortfolioRequest(market=shared.market(), trades=[payer, receiver, swap],
                                   config=RunConfig(simulation=_simulation(128)))
        return payer, price_portfolio(request)

    def test_today(self, parity):
        _, result = parity
        payer, receiver, swap = result.base_npv_per_trade
        assert payer - receiver == pytest.approx(swap, rel=1e-10)

    def test_on_every_path_before_expiry(self, parity):
        cfg, result = parity
        cube = np.asarray(result.npv_cube)
        alive = np.asarray([d < cfg.exercise_date for d in DATES])
        assert alive.any()
        np.testing.assert_allclose(cube[:, alive, 0] - cube[:, alive, 1], cube[:, alive, 2], rtol=1e-9, atol=1e-6)
