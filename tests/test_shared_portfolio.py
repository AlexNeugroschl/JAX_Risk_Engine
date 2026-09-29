"""
The shared test portfolio (tests/support/portfolio.py, plan §6.3) at t=0 against ORE, trade by
trade (L2), and its HTTP form against its configs (the schema round trip). The HTTP result on
it equals the direct call in tests/test_api_market_path.py (L6).

Measured 2026-09-29, relative to ORE: swaps, Europeans and the bond <= 2.3e-14; the
Bermudans and the American <= 3.8e-11 (their calibration runs to ORE's 1e-8 tolerance). This
is also the t=0 baseline of plan 1.5 for the market path: a change that moves one of these
numbers is a change in what the engine computes.
"""
import pytest

from engine.api.market_schemas import MarketPortfolioRequestSchema
from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.valuation.portfolio import value_today
from tests.support import portfolio

TRADES = portfolio.trades()


@pytest.mark.parametrize("name", TRADES)
def test_every_trade_today_equals_ore(name):
    cfg = TRADES[name]
    calibrated = isinstance(cfg, (BermudanSwaptionConfig, AmericanSwaptionConfig))
    ours = value_today([cfg], portfolio.market(), "USD")[0]
    assert ours == pytest.approx(portfolio.ore_npv(cfg), rel=1e-9 if calibrated else 1e-12)


def test_the_http_body_is_the_portfolio():
    body = {"market": portfolio.market_json(), "trades": list(portfolio.trades_json().values()),
            "scenario_risk": False}
    request = MarketPortfolioRequestSchema(**body).to_dataclass()
    assert request.market == portfolio.market()
    assert request.trades == list(TRADES.values())
