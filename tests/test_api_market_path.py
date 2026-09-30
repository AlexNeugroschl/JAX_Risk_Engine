"""
`POST /v2/portfolio/price` (`engine.api.market_schemas`; the `/v2` is a name, not a version): the market path
over HTTP (plan §6.2 L6). On the shared test portfolio (tests/support/portfolio.py) the polled
result equals a direct `price_portfolio` call, including ORE's time-weighted and Basel
exposure figures and the 2-D Vega matrix; a request the market path cannot price is a 400
before any job starts, and a trade carrying a model of its own is a 422.
"""
import time

import numpy as np
import ORE
import pytest

from engine.api.market_schemas import MarketPortfolioRequestSchema
from engine.portfolio import price_portfolio
from tests.support import portfolio

FAST_ENGINE = {"n_per_std": 12, "std_devs": 4.0}


def _simulation():
    dates = [(portfolio.ASOF + ORE.Period(m, ORE.Months)).ISO() for m in (3, 6, 12, 24)]
    return {"dates": dates, "base_currency": "USD", "samples": 128, "seed": 3,
            "ir": {"USD": {"reversion": 0.03, "volatility": 0.01, "calibration_expiries": ["1Y", "2Y"],
                           "calibration_terms": ["4Y", "3Y"]}}}


def _swap(**overrides):
    return {"trade_type": "swap", "notional": 1e6, "fixed_rate": 0.031, "payer": True, "swap_tenor": "5Y",
            **overrides}


def _body(trades, **overrides):
    body = {"schema_version": "2", "market": portfolio.market_json(), "trades": trades,
            "simulation": _simulation(), "pricing": {"bermudan": FAST_ENGINE, "american": FAST_ENGINE}}
    body.update(overrides)
    return body


def _submit_and_poll(client, body, timeout_s=600):
    r = client.post("/v2/portfolio/price", json=body)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        data = client.get(f"/portfolio/price/{job_id}").json()
        if data["status"] in ("done", "failed"):
            assert data["status"] == "done", data["error"]
            return data["result"]
        time.sleep(0.5)
    pytest.fail("job never reached a terminal state")


@pytest.mark.slow
def test_result_matches_direct_price_portfolio_call(test_client):
    body = _body([{**trade, "trade_id": name} for name, trade in portfolio.trades_json().items()])
    result = _submit_and_poll(test_client, body)
    assert result["trade_ids"] == list(portfolio.trades_json())
    direct = price_portfolio(MarketPortfolioRequestSchema(**body).to_dataclass())
    np.testing.assert_allclose(result["base_npv_per_trade"], direct.base_npv_per_trade, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(result["npv_cube"]), np.asarray(direct.npv_cube), rtol=1e-9, atol=1e-6)
    for name in ("times", "epe", "ene", "ee_b", "eee_b", "epe_b", "eepe_b"):
        np.testing.assert_allclose(result["exposure"][name], np.asarray(getattr(direct.exposure, name)),
                                   rtol=1e-9, atol=1e-6)
    assert result["exposure"]["basel_eepe"] == pytest.approx(direct.exposure.basel_eepe, rel=1e-9)
    assert result["measure"] == direct.measure and result["scenario_risk_available"]


@pytest.mark.slow
def test_greeks_cross_the_boundary_with_the_vega_matrix(test_client):
    shared = portfolio.trades_json()
    body = _body([shared["swap-payer"], shared["european-payer"]], scenario_risk=False, simulation=None,
                 compute_greeks=True)
    greeks = _submit_and_poll(test_client, body)["greeks"]
    direct = price_portfolio(MarketPortfolioRequestSchema(**body).to_dataclass()).greeks
    assert greeks["1"]["shapes"] == {"vega:USD": [4, 3]}
    np.testing.assert_allclose(np.reshape(greeks["1"]["values"]["vega:USD"], (4, 3)), direct[1]["vega:USD"],
                               rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(greeks["0"]["values"]["delta:discount:USD"], direct[0]["delta:discount:USD"],
                               rtol=1e-9, atol=1e-9)
    assert greeks["0"]["theta"] == pytest.approx(float(direct[0]["theta"]), rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("overrides, message", [
    ({"simulation": None}, "CamConfig"),
    ({"trades": [_swap(index_tenor_months=3)]}, "USD-SIMINDEX-3M"),
    ({"trades": [_swap(currency="EUR")]}, "EUR"),
])
def test_an_unpriceable_request_is_a_400_and_no_job(test_client, overrides, message):
    r = test_client.post("/v2/portfolio/price", json={**_body([_swap()]), **overrides})
    assert r.status_code == 400, r.text
    assert message in r.json()["detail"]


def test_a_trade_carrying_its_own_model_is_a_422(test_client):
    r = test_client.post("/v2/portfolio/price", json=_body([_swap(hw_sigma=0.01)]))
    assert r.status_code == 422
    assert "hw_sigma" in r.text


def test_trade_ids_are_given_on_every_trade_or_none(test_client):
    r = test_client.post("/v2/portfolio/price", json=_body([_swap(trade_id="a"), _swap()]))
    assert r.status_code == 400 and "every trade or on none" in r.json()["detail"]
    r = test_client.post("/v2/portfolio/price", json=_body([_swap(trade_id="a"), _swap(trade_id="a")]))
    assert r.status_code == 400 and "unique" in r.json()["detail"]
