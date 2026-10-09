"""
`POST /portfolio/price` (`engine.api.market_schemas`): the market path over HTTP (plan §6.2
L6). On the shared test portfolio (tests/support/portfolio.py) the polled result equals a direct
`price_portfolio` call, including ORE's time-weighted and Basel exposure figures, the 2-D Vega
matrix, and each trade's row keyed by its id (I-10); the cube returned by reference is the
inline cube, its hashes checked, and can be left out (A-17, I-09); a request the market path
cannot price is a 400 before any job starts, and a trade carrying a model of its own is a 422.
The result carries the precision report of the worker that ran it (I-12).
"""
import time

import jax
import numpy as np
import ORE
import pytest

from engine.api.artifacts import ArtifactError, read_array
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
    body = {"market": portfolio.market_json(), "trades": trades,
            "simulation": _simulation(), "pricing": {"bermudan": FAST_ENGINE, "american": FAST_ENGINE}}
    body.update(overrides)
    return body


def _submit_and_poll(client, body, timeout_s=600):
    r = client.post("/portfolio/price", json=body)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        data = client.get(f"/jobs/{job_id}").json()
        if data["status"] in ("done", "failed"):
            assert data["status"] == "done", data["error"]
            return data["result"]
        time.sleep(0.5)
    pytest.fail("job never reached a terminal state")


@pytest.mark.slow
def test_result_matches_direct_price_portfolio_call(test_client):
    body = _body([{**trade, "trade_id": name} for name, trade in portfolio.trades_json().items()])
    result = _submit_and_poll(test_client, body)
    assert [t["trade_id"] for t in result["trades"]] == list(portfolio.trades_json())
    direct = price_portfolio(MarketPortfolioRequestSchema(**body).to_dataclass())
    np.testing.assert_allclose([t["base_npv"] for t in result["trades"]], direct.base_npv_per_trade, rtol=1e-9)
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
    rows = {t["trade_id"]: t for t in _submit_and_poll(test_client, body)["trades"]}
    direct = price_portfolio(MarketPortfolioRequestSchema(**body).to_dataclass()).greeks
    swap, european = rows["swap-payer"]["greeks"], rows["european-payer"]["greeks"]
    assert european["shapes"] == {"vega:USD": [4, 3]} and rows["swap-payer"]["exposure"] is None
    np.testing.assert_allclose(np.reshape(european["values"]["vega:USD"], (4, 3)), direct[1]["vega:USD"],
                               rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(swap["values"]["delta:discount:USD"], direct[0]["delta:discount:USD"],
                               rtol=1e-9, atol=1e-9)
    assert swap["theta"] == pytest.approx(float(direct[0]["theta"]), rel=1e-9, abs=1e-9)


@pytest.mark.slow
def test_the_cube_by_reference_is_the_inline_cube_and_none_leaves_it_out(test_client):
    """Decision A-17 (I-09): `cube_output: "artifact"` returns a chunked, hashed reference
    whose bytes are the inline cube's, the trade order hashed beside it; `"none"` returns
    neither, and every other figure is unchanged."""
    trades = [portfolio.trades_json()[n] for n in ("swap-payer", "bond")]
    inline = _submit_and_poll(test_client, _body(trades))
    by_reference = _submit_and_poll(test_client, _body(trades, cube_output="artifact"))
    left_out = _submit_and_poll(test_client, _body(trades, cube_output="none"))

    reference = by_reference["npv_cube_artifact"]
    assert by_reference["npv_cube"] is None and inline["npv_cube_artifact"] is None
    assert reference["axes"] == ["scenario", "date", "trade"] and reference["items"]["ids"] == ["swap-payer", "bond"]
    fetch = lambda url: test_client.get(url).content  # noqa: E731
    np.testing.assert_array_equal(read_array(reference, fetch), np.asarray(inline["npv_cube"]))
    assert left_out["npv_cube"] is None and left_out["npv_cube_artifact"] is None
    for result in (by_reference, left_out):
        assert result["exposure"] == inline["exposure"] and result["trades"] == inline["trades"]

    tampered = {**reference, "items": {**reference["items"], "ids": ["bond", "swap-payer"]}}
    with pytest.raises(ArtifactError, match="trade order"):
        read_array(tampered, fetch)


@pytest.mark.parametrize("overrides, message", [
    ({"simulation": None}, "CamConfig"),
    ({"trades": [_swap(index_tenor_months=3)]}, "USD-SIMINDEX-3M"),
    ({"trades": [_swap(currency="EUR")]}, "EUR"),
    # The run configuration (I-68): a reporting currency contradicting the simulation's was
    # silently ignored before 2026-09-30; a precision format is refused until it is enabled
    # (compute below float32: F-07).
    ({"base_currency": "EUR"}, "contradicts"),
    ({"precision": {"pricing": {"storage": "float16", "compute": "float16", "accumulate": "float16"}}},
     "precision.pricing"),
])
def test_an_unpriceable_request_is_a_400_and_no_job(test_client, overrides, message):
    r = test_client.post("/portfolio/price", json={**_body([_swap()]), **overrides})
    assert r.status_code == 400, r.text
    assert message in r.json()["detail"]


def test_a_trade_carrying_its_own_model_is_a_422(test_client):
    r = test_client.post("/portfolio/price", json=_body([_swap(hw_sigma=0.01)]))
    assert r.status_code == 422
    assert "hw_sigma" in r.text


def test_trade_ids_are_given_on_every_trade_or_none(test_client):
    r = test_client.post("/portfolio/price", json=_body([_swap(trade_id="a"), _swap()]))
    assert r.status_code == 400 and "every trade or on none" in r.json()["detail"]
    r = test_client.post("/portfolio/price", json=_body([_swap(trade_id="a"), _swap(trade_id="a")]))
    assert r.status_code == 400 and "unique" in r.json()["detail"]


@pytest.mark.slow
def test_a_result_carries_the_workers_precision_report(test_client):
    """I-12: `/version` names the API process's backend; the devices, the realized formats and
    the paired-sample estimates of a job are read in the worker that ran it, on its result."""
    precision = {"pricing": {"storage": "float16", "compute": "float32", "accumulate": "float32"},
                 "paired_fraction": 0.5}
    report = _submit_and_poll(test_client, _body([_swap(trade_id="s1")], precision=precision))["precision"]
    assert report["realized"] == {"shocks": "float64", "states": "float64", "market": "float64", "values/s1": "float16"}
    assert report["devices"] and report["backend"] == jax.default_backend()
    assert report["policy"]["paired_fraction"] == 0.5 and (report["paths"], report["paired_paths"]) == (128, 64)
    assert report["figures"]["trades/s1/EPE"]["kind"] == "mean"
    assert report["figures"]["netting_set/PFE_95"]["kind"] == "quantile"


def test_a_paired_fraction_outside_the_unit_interval_is_a_422(test_client):
    response = test_client.post("/portfolio/price", json=_body([_swap()], precision={"paired_fraction": 2.0}))
    assert response.status_code == 422 and "paired_fraction" in response.text
