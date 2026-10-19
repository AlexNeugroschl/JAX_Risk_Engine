"""
The portfolio analytics over HTTP (`npv`, `exposure`, `sensitivities` of `POST /runs`; plan §6.2
L6). On the shared test portfolio (tests/support/portfolio.py) the polled result equals
`engine.run.run` on the same request, ORE's time-weighted and Basel exposure figures included,
each trade's row keyed by its id (I-10); the Greeks carry their factor labels and the 2-D Vega
matrix (I-87); the cube and the exercise decisions by reference are the inline arrays, their
hashes checked, and can be left out (A-17, I-09); a request the pipeline cannot price is a 400
before any run starts. The result carries the precision report of the worker that ran it (I-12).
"""
import jax
import numpy as np
import pytest

from engine.api.artifacts import ArtifactError, read_array
from tests.support import portfolio
from tests.support.runs import models, run_body, run_in_process, run_over_http, simulation

FAST_ENGINE = {"n_per_std": 12, "std_devs": 4.0}
SWAP = {"trade_type": "swap", "notional": 1e6, "fixed_rate": 0.031, "payer": True, "swap_tenor": "5Y"}


def _config(**overrides):
    return {"models": models(calibration_expiries=["1Y", "2Y"], calibration_terms=["4Y", "3Y"]),
            "simulation": simulation(samples=128, months=(3, 6, 12, 24)),
            "pricing": {"bermudan": FAST_ENGINE, "american": FAST_ENGINE}, **overrides}


def _body(trades, analytics=("npv", "exposure"), outputs=None, **overrides):
    return run_body(analytics, trades, outputs=outputs, **_config(**overrides))


@pytest.mark.slow
def test_the_result_equals_the_in_process_run(test_client):
    body = _body([{**trade, "trade_id": name} for name, trade in portfolio.trades_json().items()],
                 outputs={"npv_cube": "inline"})
    result = run_over_http(test_client, body)
    assert [t["trade_id"] for t in result["trades"]] == list(portfolio.trades_json())
    direct = run_in_process(body).portfolio
    assert [t["npv"] for t in result["trades"]] == direct.base_npv_per_trade
    np.testing.assert_array_equal(np.asarray(result["exposure"]["npv_cube"]["values"]), np.asarray(direct.npv_cube))
    netting_set = result["exposure"]["netting_set"]
    for name in ("times", "epe", "ene", "ee_b", "eee_b", "epe_b", "eepe_b"):
        np.testing.assert_array_equal(netting_set[name], np.asarray(getattr(direct.exposure, name)))
    assert netting_set["basel_eepe"] == direct.exposure.basel_eepe
    assert result["exposure"]["measure"] == direct.measure
    options = [n for n in portfolio.trades_json() if "bermudan" in n or "american" in n or "european" in n]
    assert result["exposure"]["exercised"] is None and set(direct.exercised) == set(options)
    settings = result["exposure"]["pricing_settings"]
    assert settings["bermudan_swaption"] == "engine=LgmGrid nx=12 sx=4 model=fixed"


@pytest.mark.slow
def test_the_greeks_carry_their_factors_and_the_vega_matrix(test_client):
    shared = portfolio.trades_json()
    body = run_body(["sensitivities"], [shared["swap-payer"], shared["european-payer"]])
    result = run_over_http(test_client, body)
    rows = {t["trade_id"]: t for t in result["trades"]}
    direct = run_in_process(body).portfolio.greeks
    swap, european = rows["swap-payer"]["greeks"], rows["european-payer"]["greeks"]
    assert european["shapes"] == {"vega:USD": [4, 3]} and rows["swap-payer"]["exposure"] is None
    np.testing.assert_array_equal(np.reshape(european["values"]["vega:USD"], (4, 3)), direct[1]["vega:USD"])
    np.testing.assert_array_equal(swap["values"]["delta:discount:USD"], direct[0]["delta:discount:USD"])
    assert swap["theta"] == float(direct[0]["theta"])
    section = result["sensitivities"]
    assert section["method"] == "Bump"
    assert section["factors"]["delta:discount:USD"] == result["config"]["sensitivity"]["curve_tenors"]
    assert len(section["factors"]["delta:discount:USD"]) == len(swap["values"]["delta:discount:USD"])
    vols = portfolio.VOLS
    assert section["factors"]["vega:USD"] == [f"{o}x{s}" for o in vols.option_tenors for s in vols.swap_tenors]
    assert result["exposure"] is None and result["npv"] == sum(t["npv"] for t in result["trades"])


@pytest.mark.slow
def test_arrays_by_reference_are_the_inline_arrays_and_none_leaves_them_out(test_client):
    """Decision A-17 (I-09), revised with the run request: `artifact` (the cube's default) is a
    chunked, hashed reference whose bytes are the inline array's, its trade (or option) order
    hashed beside it; `none` leaves it out (and the cube is never assembled, I-105); every other
    figure is unchanged."""
    trades = [{**portfolio.trades_json()[n], "trade_id": n} for n in ("swap-payer", "bond", "bermudan-payer-physical")]
    inline = run_over_http(test_client, _body(trades, outputs={"npv_cube": "inline", "exercised": "inline"}))
    by_reference = run_over_http(test_client, _body(trades, outputs={"exercised": "artifact"}))
    left_out = run_over_http(test_client, _body(trades, outputs={"npv_cube": "none"}))

    fetch = lambda url: test_client.get(url).content  # noqa: E731
    cube, exercised = by_reference["exposure"]["npv_cube"], by_reference["exposure"]["exercised"]
    assert cube["form"] == "artifact" and cube["axes"] == ["scenario", "date", "trade"]
    assert cube["items"]["ids"] == ["swap-payer", "bond", "bermudan-payer-physical"]
    np.testing.assert_array_equal(read_array(cube, fetch), np.asarray(inline["exposure"]["npv_cube"]["values"]))
    assert exercised["axes"] == ["scenario", "date", "option"] and exercised["items"]["ids"] == [
        "bermudan-payer-physical"]
    np.testing.assert_array_equal(read_array(exercised, fetch), np.asarray(inline["exposure"]["exercised"]["values"]))
    assert exercised["dtype"] == "bool" and read_array(exercised, fetch).shape == (128, 4, 1)
    assert left_out["exposure"]["npv_cube"] is None and left_out["exposure"]["exercised"] is None
    for result in (by_reference, left_out):
        assert result["exposure"]["netting_set"] == inline["exposure"]["netting_set"]
        assert result["trades"] == inline["trades"]

    tampered = {**cube, "items": {**cube["items"], "ids": ["bond", "swap-payer", "bermudan-payer-physical"]}}
    with pytest.raises(ArtifactError, match="trade order"):
        read_array(tampered, fetch)


@pytest.mark.parametrize("overrides, message", [
    ({"portfolio": [{**SWAP, "index_tenor_months": 3}]}, "USD-SIMINDEX-3M"),
    ({"portfolio": [{**SWAP, "currency": "EUR"}]}, "EUR"),
    # A stage the precision rules refuse (accumulating below float32).
    ({"precision": {"pricing": {"storage": "float16", "compute": "float16", "accumulate": "float16"}}},
     "config.precision.pricing"),
    # ORE's LgmBuilder requires a non-negative shift horizon (I-32).
    ({"pricing": {"bermudan": {"shift_horizon": -0.5}}}, "shift_horizon"),
    ({"exposure": {"pfe_quantiles": [0.95, 1.5]}}, "config.exposure: config.exposure.pfe_quantiles"),
])
def test_an_unpriceable_request_is_a_400_and_no_run(test_client, overrides, message):
    body = _body([SWAP])
    body["portfolio"] = overrides.pop("portfolio", body["portfolio"])
    body["config"].update(overrides)
    r = test_client.post("/runs", json=body)
    assert r.status_code == 400, r.text
    assert message in r.json()["detail"]


def test_trade_ids_are_given_on_every_trade_or_none_and_unique(test_client):
    r = test_client.post("/runs", json=_body([{**SWAP, "trade_id": "a"}, SWAP]))
    assert r.status_code == 400 and "every trade or on none" in r.json()["detail"]
    r = test_client.post("/runs", json=_body([{**SWAP, "trade_id": "a"}, {**SWAP, "trade_id": "a"}]))
    assert r.status_code == 400 and "unique" in r.json()["detail"]


@pytest.mark.slow
def test_a_result_carries_the_workers_precision_report(test_client):
    """I-12: `/version` names the API process's backend; the devices, the realized formats and
    the paired-sample estimates of a run are read in the worker that ran it, on its result."""
    precision = {"pricing": {"storage": "float16", "compute": "float32", "accumulate": "float32"},
                 "paired_fraction": 0.5}
    result = run_over_http(test_client, _body([{**SWAP, "trade_id": "s1"}], precision=precision))
    report = result["exposure"]["precision"]
    assert report["realized"] == {"shocks": "float64", "states": "float64", "market": "float64", "values/s1": "float16"}
    assert report["devices"] and report["backend"] == jax.default_backend()
    assert report["policy"]["paired_fraction"] == 0.5 and (report["paths"], report["paired_paths"]) == (128, 64)
    assert report["figures"]["trades/s1/EPE"]["kind"] == "mean"
    assert report["figures"]["netting_set/PFE_95"]["kind"] == "quantile"


def test_a_paired_fraction_outside_the_unit_interval_is_a_422(test_client):
    response = test_client.post("/runs", json=_body([SWAP], precision={"paired_fraction": 2.0}))
    assert response.status_code == 422 and "paired_fraction" in response.text
