"""
The shared test portfolio (tests/support/portfolio.py, plan §6.3) at t=0 against ORE, trade by
trade (L2), and its HTTP form against its configs (the schema round trip). The HTTP result on
it equals the direct call in tests/test_api_portfolio.py (L6).

Measured 2026-09-29, relative to ORE: swaps, Europeans and the bond <= 2.3e-14; the
Bermudans and the American <= 3.8e-11 (their calibration runs to ORE's 1e-8 tolerance). This
is also the t=0 baseline of plan 1.5 for the pipeline: a change that moves one of these
numbers is a change in what the engine computes.

Each option booked `Short` is ORE's short option, booked through ORE's own trade builder with
`<LongShort>Short</LongShort>` (I-101; measured 2026-10-11 to the same tolerances).
"""
import dataclasses

import pytest

from engine.api.requests import MarketPortfolioRequestSchema
from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.pricing.cube import OPTIONS, value_today
from tests.support import portfolio
from tests.support.ore_parity_matrix import cells

TRADES = portfolio.trades()
SHORTS = [dataclasses.replace(cfg, long_short="Short") for cfg in TRADES.values() if isinstance(cfg, OPTIONS)]


@pytest.mark.ore_parity(cells("npv", TRADES.values()), tolerance="1e-12 relative (1e-9 calibrated Bermudans, Americans)")
@pytest.mark.parametrize("name", TRADES)
def test_every_trade_today_equals_ore(name):
    cfg = TRADES[name]
    calibrated = isinstance(cfg, (BermudanSwaptionConfig, AmericanSwaptionConfig))
    ours = value_today([cfg], portfolio.market(), "USD")[0]
    assert ours == pytest.approx(portfolio.ore_npv(cfg), rel=1e-9 if calibrated else 1e-12)


@pytest.mark.ore_parity(cells("npv", SHORTS), tolerance="1e-10 relative (1e-9 calibrated Bermudans, Americans)")
@pytest.mark.parametrize("name", [n for n, cfg in TRADES.items() if isinstance(cfg, OPTIONS)])
def test_a_short_option_today_is_ores_short_option(name):
    """I-101: a `Short` option on a positive notional, as ORE books one. Before the fix the only
    way to book a short was a negative notional, which priced the Bermudan payer as a long
    receiver: +15,017.85 where ORE's short is -35,712.95."""
    cfg = dataclasses.replace(TRADES[name], long_short="Short")
    ore = portfolio.ore_trade_npv(cfg)
    assert ore < 0.0
    calibrated = isinstance(cfg, (BermudanSwaptionConfig, AmericanSwaptionConfig))
    assert value_today([cfg], portfolio.market(), "USD")[0] == pytest.approx(ore, rel=1e-9 if calibrated else 1e-10)


def test_the_http_body_is_the_portfolio():
    body = {"market": portfolio.market_json(), "trades": list(portfolio.trades_json().values()),
            "scenario_risk": False}
    request = MarketPortfolioRequestSchema(**body).to_dataclass()
    assert request.market == portfolio.market()
    assert request.trades == list(TRADES.values())
