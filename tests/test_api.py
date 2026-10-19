"""
`engine.api` through FastAPI's in-process TestClient: the run request's contract (I-87).
`/health`, `/version` (the engine's identity) and `/config/defaults`; `POST /runs` returns 202 and
a run id, and `GET /runs/{run_id}` reaches a result equal to `engine.run.run` on the same request,
here with the Hull-White model in `config.models` and the Jamshidian engine; the routes of
2026-10-07 are gone; a body the schema refuses is a 422 naming its field (every model strict,
I-103), a request the engine refuses a 400 naming its field and no run created; the calibration
analytic is the cross-asset model a run simulates with (I-94); `wait` long-polls; precision on the
wire (the HTTP side of tests/test_precision.py). The other analytics over HTTP are
tests/test_api_portfolio.py and tests/test_api_market_risk.py; idempotency, cancelling, retention
and the worker's start over HTTP are tests/test_engine_worker.py.
"""
import time

import numpy as np
import pytest

from engine.api.requests import RunRequestConfigSchema, RunRequestSchema
from engine.precision import Precision, StagePrecision
from engine.run import engine_identity
from tests.support import portfolio as shared
from tests.support.runs import models, request_of, run_body, run_in_process, run_over_http, simulation

SWAP = {"trade_type": "swap", "notional": 1_000_000.0, "fixed_rate": 0.032, "payer": True, "swap_tenor": "2Y"}
EUROPEAN = {"trade_type": "european_swaption", "notional": 500_000.0, "fixed_rate": 0.031, "payer": True,
            "swap_tenor": "2Y", "forward_start": "1Y"}
BASKET = {"calibration_expiries": ["1Y", "2Y", "5Y"], "calibration_terms": ["5Y", "4Y", "1Y"]}


def _exposure_body(trades=(SWAP, EUROPEAN), model="HullWhite", **config):
    return run_body(["npv", "exposure"], trades, outputs={"npv_cube": "inline"},
                    **{"models": models(model), "simulation": simulation(),
                       "exposure": {"pfe_quantiles": [0.95]}, **config})


def _add_eur_without_fx_volatility(body):
    body["config"]["models"]["ir"]["EUR"] = {"model": "HullWhite", "reversion": 0.02, "volatility": 0.008}


class TestHealthVersionAndDefaults:
    def test_health_returns_ok_and_the_worker(self, test_client):
        body = test_client.get("/health").json()
        assert body["status"] == "ok" and body["worker"]["mode"] == "spawn"

    def test_version_is_the_engines_identity(self, test_client):
        body = test_client.get("/version").json()
        identity = engine_identity()
        assert body["engine"] == {"version": identity.version, "revision": identity.revision,
                                  "modified": identity.modified, "jax": identity.jax, "jaxlib": identity.jaxlib,
                                  "ore": identity.ore}
        assert body["engine"]["jax"] and body["engine"]["ore"] and "jax_backend" in body

    def test_the_defaults_are_the_whole_config_to_copy(self, test_client):
        defaults = test_client.get("/config/defaults").json()
        assert defaults == RunRequestConfigSchema().model_dump(mode="json")
        assert defaults["models"] is None and defaults["pricing"]["bermudan"]["shift_horizon"] == 0.5
        assert defaults["sensitivity"]["method"] == "Bump" and defaults["exposure"]["pfe_quantiles"] == [0.95, 0.99]
        RunRequestSchema.model_validate({"market": shared.market_json(), "analytics": ["npv"], "config": defaults})


@pytest.mark.slow
class TestRunsOverHttp:
    def test_the_result_is_the_in_process_runs(self, test_client):
        """The polled result equals `engine.run.run` on the request the API built: every row's
        value, the cube and the netting set's profile, with the Hull-White model and the
        Jamshidian engine reaching the worker (the value differs from the LGM/Bachelier run's)."""
        jamshidian = {"european": "Jamshidian", "jamshidian": {"reversion": 0.03, "volatility": 0.01}}
        body = _exposure_body(pricing=jamshidian)
        result, direct = run_over_http(test_client, body), run_in_process(body).portfolio
        assert [t["trade_id"] for t in result["trades"]] == ["trade-0", "trade-1"]
        assert [t["npv"] for t in result["trades"]] == direct.base_npv_per_trade
        assert result["npv"] == direct.base_npv
        cube = result["exposure"]["npv_cube"]
        assert cube["form"] == "inline" and cube["axes"] == ["scenario", "date", "trade"]
        np.testing.assert_array_equal(np.asarray(cube["values"]), np.asarray(direct.npv_cube))
        for name in ("times", "epe", "ene", "ee_b", "eee_b"):
            np.testing.assert_array_equal(result["exposure"]["netting_set"][name], getattr(direct.exposure, name))
        assert result["exposure"]["dates"] == body["config"]["simulation"]["dates"]
        baseline = run_over_http(test_client, _exposure_body(model="LGM"))
        assert result["trades"][1]["npv"] != baseline["trades"][1]["npv"]

    def test_the_result_says_what_each_figure_is(self, test_client):
        """I-87 (the project audit, AUD-39): the configuration as run with every default, the
        as-of date, the reporting currency, the engine's identity, the products and pricing
        settings, without the request beside it."""
        body = _exposure_body()
        result = run_over_http(test_client, body)
        assert result["asof"] == shared.ASOF_ISO and result["base_currency"] == "USD"
        assert result["analytics"] == ["npv", "exposure"]
        assert result["config"] == RunRequestSchema.model_validate(body).config.model_dump(mode="json")
        assert result["config"]["pricing"]["bermudan"]["n_per_std"] == 30
        assert result["engine"]["revision"] == engine_identity().revision
        assert [t["product"] for t in result["trades"]] == ["swap", "european_swaption"]
        assert result["exposure"]["pricing_settings"] == {"european_swaption": "engine=Bachelier",
                                                          "swap": "engine=Discounting"}
        assert result["market_risk"] is None and result["sensitivities"] is None

    def test_runs_submitted_back_to_back_get_their_own_results(self, test_client):
        from tests.support.runs import poll, submit

        one = submit(test_client, _exposure_body(trades=({**SWAP, "notional": 1e6},)))
        nine = submit(test_client, _exposure_body(trades=({**SWAP, "notional": 9e6},)))
        assert one != nine
        cubes = [np.asarray(poll(test_client, r)["result"]["exposure"]["npv_cube"]["values"]) for r in (one, nine)]
        np.testing.assert_allclose(cubes[1], 9.0 * cubes[0], rtol=1e-12)

    def test_an_empty_portfolio_prices_to_a_zero_width_result(self, test_client):
        result = run_over_http(test_client, _exposure_body(trades=()))
        assert result["npv"] == 0.0 and result["exposure"]["npv_cube"]["shape"][-1] == 0

    def test_wait_answers_once_the_run_is_done(self, test_client):
        from tests.support.runs import submit

        run_id = submit(test_client, _exposure_body())
        started = time.monotonic()
        status = test_client.get(f"/runs/{run_id}", params={"wait": 30}).json()
        assert status["status"] == "done" and time.monotonic() - started < 30
        assert status["submitted"] <= status["started"] <= status["finished"]


class TestTheCalibrationAnalytic:
    """The cross-asset model's calibration per currency (`engine.calibration.cam`), the one a run's
    exposure analytic simulates with, which replaces `POST /calibration/cam` and the standalone
    `POST /calibration/lgm` (I-94)."""

    @pytest.mark.slow
    @pytest.mark.parametrize("model", ["LGM", "HullWhite"])
    def test_the_result_is_the_calibration_a_run_simulates_with(self, test_client, model):
        from engine.calibration.cam import calibrate_cam
        from engine.market_simulation.config import build_cross_asset_model

        body = run_body(["calibration"], models=models(model, **BASKET))
        usd = run_over_http(test_client, body)["calibration"]["currencies"]["USD"]
        assert usd["model"] == model and usd["reversion"] == 0.03
        request = request_of(body)
        direct = calibrate_cam(shared.market(), request.config.models.ir)["USD"]
        np.testing.assert_array_equal(usd["sigma_times"], np.asarray(direct.sigma.times))
        np.testing.assert_array_equal(usd["sigma_values"], np.asarray(direct.sigma.values))
        exposure = request_of(run_body(["exposure"], ["swap-payer"], models=models(model, **BASKET),
                                       simulation=simulation()))
        simulated = build_cross_asset_model(shared.market(), exposure.config.cam()).ir[0].sigma
        np.testing.assert_array_equal(usd["sigma_values"], np.asarray(simulated.values))
        assert [(h["expiry"], h["term"]) for h in usd["helpers"]] == [("1Y", "5Y"), ("2Y", "4Y"), ("5Y", "1Y")]
        np.testing.assert_allclose([h["model_value"] for h in usd["helpers"]],
                                   [h["market_value"] for h in usd["helpers"]], rtol=1e-8)
        assert usd["error"] == direct.error and 0.0 <= usd["error"] < 1e-4

    def test_a_currency_without_a_basket_is_refused_alone(self, test_client):
        r = test_client.post("/runs", json=run_body(["calibration"], models=models()))
        assert r.status_code == 400 and "no calibration basket" in r.json()["detail"]

    def test_a_currency_the_market_lacks_is_a_400(self, test_client):
        body = run_body(["calibration"], models={"ir": {"EUR": {"reversion": 0.03, **BASKET}}})
        r = test_client.post("/runs", json=body)
        assert r.status_code == 400 and "EUR" in r.json()["detail"]


class TestTheRequestIsRefusedBeforeAnyRun:
    def test_the_routes_of_2026_10_07_are_gone(self, test_client):
        for route in ("/portfolio/price", "/portfolio/market-risk", "/calibration/cam", "/calibration/lgm"):
            assert test_client.post(route, json={}).status_code == 404, route
        assert test_client.get("/jobs/any-id").status_code == 404

    @pytest.mark.parametrize("change, field", [
        (lambda b: b["portfolio"][0].update(hw_sigma=0.01), "hw_sigma"),
        (lambda b: b["market"]["currencies"]["USD"]["discount_curve"].update(day_count="ACT/360"), "day_count"),
        (lambda b: b["config"].update(scenario_risk=True), "scenario_risk"),
        (lambda b: b.update(trades=[]), "trades"),
        (lambda b: b["config"]["models"]["ir"]["USD"].update(model="CIR"), "CIR"),
        (lambda b: b.update(analytics=["xva"]), "xva"),
        (lambda b: b.update(market=12345), "market"),
        (lambda b: b["portfolio"].append({"trade_type": "not_a_real_type"}), "not_a_real_type"),
    ], ids=["trade-model", "curve-field", "retired-flag", "retired-trades", "model", "analytic", "market", "product"])
    def test_a_body_the_schema_refuses_is_a_422_naming_it(self, test_client, change, field):
        """Every model refuses an unknown field (I-103), so nothing is silently stripped."""
        body = _exposure_body()
        change(body)
        r = test_client.post("/runs", json=body)
        assert r.status_code == 422 and field in r.text

    @pytest.mark.parametrize("change, message", [
        (lambda b: b["config"].pop("simulation"), "config.simulation: the exposure analytic needs it"),
        (lambda b: b["config"].pop("models"), "config.models: the exposure analytic needs it"),
        (lambda b: b.update(analytics=["npv"]), "config.models: no requested analytic reads it"),
        (lambda b: b.update(analytics=["npv", "npv"]), "named more than once"),
        (lambda b: b.update(analytics=[]), "name at least one"),
        (lambda b: b["config"].update(sensitivity={"method": "AD"}), "config.sensitivity: no requested analytic"),
        (lambda b: b.update(outputs={"pnl": "inline"}), "outputs.pnl: the market_risk analytic"),
        (lambda b: b.update(portfolio=[{**SWAP, "currency": "GBP", "trade_id": "gbp"}]), "trade 'gbp'"),
        (lambda b: b["portfolio"][0].update(trade_id="a"), "every trade or on none"),
        (_add_eur_without_fx_volatility, "fx_volatilities"),
        (lambda b: b["config"].update(base_currency="EUR"), "config.base_currency"),
        (lambda b: b["market"]["currencies"]["USD"]["discount_curve"].update(times=[5.0, 1.0, 10.0, 15.0, 20.0, 30.0]),
         "market.currencies.USD.discount_curve: a zero curve's pillar times must be non-negative and strictly "
         "increasing"),
    ], ids=["no-simulation", "no-models", "unread-models", "repeated", "none", "unread-sensitivity",
            "unread-output", "currency", "ids", "fx-volatility", "base-currency", "unsorted-curve"])
    def test_a_request_the_engine_refuses_is_a_400_naming_it(self, test_client, change, message):
        body = _exposure_body()
        change(body)
        r = test_client.post("/runs", json=body)
        assert r.status_code == 400, r.text
        assert message in r.json()["detail"]


class TestPrecisionOnTheWire:
    """`engine.precision.Precision` as `config.precision` (the dataclass level is
    tests/test_precision.py)."""

    @pytest.mark.parametrize("old", [{"simulation": 32}, {"pricing": 64}, {"risk": 32}, {"calibration": 64}])
    def test_the_retired_32_64_shape_is_a_422_naming_the_replacement(self, test_client, old):
        """Refused, not translated (decision A-12)."""
        r = test_client.post("/runs", json=_exposure_body(precision=old))
        assert r.status_code == 422 and "retired on 2026-10-01" in r.text and "StagePrecision" in r.text

    def test_a_format_outside_the_table_is_a_422(self, test_client):
        r = test_client.post("/runs", json=_exposure_body(precision={"pricing": {"compute": "fp32"}}))
        assert r.status_code == 422 and "float8_e4m3fn" in r.text

    @pytest.mark.parametrize("stage, block, message", [
        ("pricing", {"storage": "float8_e4m3fn", "compute": "float16", "accumulate": "float16"},
         "accumulate='float16': narrower than float32"),
        ("market", {"compute": "bfloat16"}, "accumulate='float64': compute='bfloat16' runs on matrix units"),
        ("simulation", {"storage": "float64", "compute": "float32", "accumulate": "float32"}, "wider than accumulate"),
        ("pricing", {"storage": "tensorfloat32", "compute": "tensorfloat32", "accumulate": "float32"},
         "a matrix-product format"),
    ])
    def test_a_refused_stage_is_a_400_naming_the_stage_and_field(self, test_client, stage, block, message):
        r = test_client.post("/runs", json=_exposure_body(precision={stage: block}))
        detail = r.json()["detail"]
        assert r.status_code == 400 and f"config.precision.{stage}" in detail and message in detail

    def test_overrides_per_product_and_trade_reach_the_run(self):
        """Decision A-15: `by_product` keyed by `trade_type`, `by_trade` by `trade_id`."""
        f32 = {"storage": "float32", "compute": "float32", "accumulate": "float32"}
        body = _exposure_body(trades=({**SWAP, "trade_id": "s"}, {**EUROPEAN, "trade_id": "e"}),
                              precision={"by_product": {"european_swaption": f32}, "by_trade": {"s": {"storage": "float32"}}})
        request = request_of(body)
        precision = request.config.precision
        assert [precision.precision_for(t) for t in request.portfolio] == [
            StagePrecision("float32", "float64", "float64"), StagePrecision("float32", "float32", "float32")]

    def test_an_unknown_product_is_a_422_listing_the_products(self, test_client):
        r = test_client.post("/runs", json=_exposure_body(precision={"by_product": {"swaption": {}}}))
        assert r.status_code == 422 and "european_swaption" in r.text and "bermudan_swaption" in r.text

    def test_an_override_naming_no_trade_is_a_400(self, test_client):
        body = _exposure_body(trades=({**SWAP, "trade_id": "s"},), precision={"by_trade": {"s2": {"storage": "float32"}}})
        r = test_client.post("/runs", json=body)
        assert r.status_code == 400 and "Precision.by_trade: ['s2'] not the id of a trade" in r.json()["detail"]

    @pytest.mark.parametrize("name", ["by_product", "by_trade"])
    def test_an_override_not_enabled_is_a_400_naming_its_key(self, test_client, name):
        key = "swap" if name == "by_product" else "s"
        body = _exposure_body(trades=({**SWAP, "trade_id": "s"},), precision={name: {key: {"compute": "float16"}}})
        r = test_client.post("/runs", json=body)
        assert r.status_code == 400 and f"config.precision.{name}['{key}']" in r.json()["detail"]

    def test_scaled_storage_and_its_rounding_reach_the_run(self):
        """Float16, bfloat16 and FP8 storage, `rounding` and `rounding_seed`."""
        fp8 = {"storage": "float8_e4m3fn", "compute": "float32", "accumulate": "float32"}
        body = _exposure_body(precision={"simulation": fp8, "pricing": {"storage": "bfloat16"},
                                         "rounding": "stochastic", "rounding_seed": 7})
        assert request_of(body).config.precision == Precision(
            simulation=StagePrecision("float8_e4m3fn", "float32", "float32"), pricing=StagePrecision("bfloat16"),
            rounding="stochastic", rounding_seed=7)

    @pytest.mark.parametrize("precision, status, message", [
        ({"rounding": "up"}, 422, "stochastic"),
        ({"rounding_seed": -1}, 422, "rounding_seed"),
        ({"rounding": "stochastic"}, 400, "no stage or override stores in one"),
    ])
    def test_a_rounding_that_cannot_apply_is_refused(self, test_client, precision, status, message):
        r = test_client.post("/runs", json=_exposure_body(precision=precision))
        assert r.status_code == status and message in r.text

    def test_a_precision_no_analytic_reads_is_a_400(self, test_client):
        """Today's values are float64 at every precision (A-10): a precision with only `npv` is
        accepted and then ignored, which is refused (I-95)."""
        body = run_body(["npv"], [SWAP], precision={"pricing": {"storage": "float32"}})
        r = test_client.post("/runs", json=body)
        assert r.status_code == 400 and "config.precision: no requested analytic reads it" in r.json()["detail"]

    @pytest.mark.slow
    def test_omitted_precision_equals_explicit_float64(self, test_client):
        f64 = {"storage": "float64", "compute": "float64", "accumulate": "float64"}
        a = run_over_http(test_client, _exposure_body())
        b = run_over_http(test_client, _exposure_body(precision={s: f64 for s in ("simulation", "market", "pricing")}))
        assert a["exposure"]["npv_cube"]["values"] == b["exposure"]["npv_cube"]["values"]

    @pytest.mark.slow
    def test_two_precisions_submitted_back_to_back_each_run_at_their_own(self, test_client):
        from tests.support.runs import poll, submit

        f32 = {"storage": "float32", "compute": "float32", "accumulate": "float32"}
        body_64 = _exposure_body()
        body_32 = _exposure_body(precision={"simulation": f32, "market": f32, "pricing": f32})
        runs = [submit(test_client, b) for b in (body_64, body_32)]
        r64, r32 = (poll(test_client, r)["result"]["exposure"]["npv_cube"]["values"] for r in runs)
        np.testing.assert_array_equal(np.asarray(r64), np.asarray(run_in_process(body_64).portfolio.npv_cube))
        np.testing.assert_array_equal(np.asarray(r32), np.asarray(run_in_process(body_32).portfolio.npv_cube))
        assert r64 != r32, "the float32 run was not priced in float32"


def test_a_currency_market_and_its_equities_reach_the_market():
    """The demo's two-currency request, with an equity."""
    from demos.demo_scenarios import demo_market_json

    body = run_body(["npv"], [SWAP], market=demo_market_json(("USD", "EUR")))
    market = request_of(body).market
    assert set(market.equities) == set(body["market"]["equities"]) and set(market.currencies) == {"USD", "EUR"}
