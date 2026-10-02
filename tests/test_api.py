"""
`engine.api` through FastAPI's in-process TestClient: /health and /version; /portfolio/price
returns 202 and a job id, and polling reaches a result equal to a direct `price_portfolio`
call, here with the Hull-White model named in the request's `simulation.ir` (the market path
over HTTP on the shared portfolio is tests/test_api_market_path.py); the Hull-White request
shape retired by roadmap 1.3 is a 422 naming its replacement; invalid bodies give a 4xx with
the validator's message (malformed schemas a 422); /calibration/lgm.
"""
import copy
import time

import numpy as np
import ORE
import pytest

from demos.demo_scenarios import demo_market_json, demo_simulation_json
from engine.api.market_schemas import MarketPortfolioRequestSchema
from engine.portfolio import price_portfolio
from engine.portfolio.worker_pool import shutdown_pool
from engine.precision import Precision, StagePrecision
from tests.support import portfolio as shared

ZERO_CURVE_SCHEMA = {"times": [0.0, 1.0, 2.0, 5.0, 10.0, 30.0], "rates": [0.03, 0.03, 0.032, 0.035, 0.038, 0.04]}


@pytest.fixture(scope="module", autouse=True)
def _cleanup_pools():
    """Shut this module's worker pool down so later in-process compiles do not run with idle
    workers attached (I-27's first step)."""
    yield
    shutdown_pool(wait=True)


def _simulation(samples=64, model="HullWhite"):
    dates = [(shared.ASOF + ORE.Period(m, ORE.Months)).ISO() for m in (6, 12, 24)]
    return {"dates": dates, "base_currency": "USD", "samples": samples, "seed": 3,
            "ir": {"USD": {"model": model, "reversion": 0.03, "volatility": 0.01}}}


def _swap(**overrides):
    return {"trade_type": "swap", "notional": 1_000_000.0, "fixed_rate": 0.032, "payer": True, "swap_tenor": "2Y",
            **overrides}


def _european(**overrides):
    return {"trade_type": "european_swaption", "notional": 500_000.0, "fixed_rate": 0.031, "payer": True,
            "swap_tenor": "2Y", "forward_start": "1Y", **overrides}


def _body(trades, **overrides):
    body = {"market": shared.market_json(), "trades": trades, "simulation": _simulation(), "pfe_quantiles": [0.95]}
    body.update(overrides)
    return body


def _submit_and_poll(client, body, timeout_s=300):
    r = client.post("/portfolio/price", json=body)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    deadline = time.time() + timeout_s
    data = None
    while time.time() < deadline:
        r = client.get(f"/portfolio/price/{job_id}")
        assert r.status_code == 200
        data = r.json()
        if data["status"] in ("done", "failed"):
            break
        time.sleep(0.2)
    assert data is not None and data["status"] == "done", data.get("error") if data else "never terminal"
    return data["result"]


def _direct(body):
    return price_portfolio(MarketPortfolioRequestSchema.model_validate(body).to_dataclass())


class TestHealthAndVersion:
    def test_health_returns_ok(self, test_client):
        r = test_client.get("/health")
        assert r.status_code == 200 and r.json() == {"status": "ok"}

    def test_version_returns_engine_info(self, test_client):
        body = test_client.get("/version").json()
        assert "engine_version" in body and "jax_backend" in body


@pytest.mark.slow
class TestPortfolioPriceHappyPath:
    def test_result_matches_direct_price_portfolio_call(self, test_client):
        """The polled result equals `price_portfolio` on the equivalent dataclass request."""
        body = _body([_swap(), _european()], pfe_quantiles=[0.95, 0.99])
        result = _submit_and_poll(test_client, body)
        direct = _direct(body)
        assert result["base_npv"] == direct.base_npv
        np.testing.assert_array_equal(np.asarray(result["npv_cube"]), np.asarray(direct.npv_cube))
        for name in ("times", "epe", "ene", "ee_b", "eee_b"):
            np.testing.assert_array_equal(np.asarray(result["exposure"][name]),
                                          np.asarray(getattr(direct.exposure, name)))
        for key, values in direct.exposure.pfe.items():
            np.testing.assert_array_equal(np.asarray(result["exposure"]["pfe"][key]), np.asarray(values))
        assert len(result["trade_exposures"]) == 2
        assert result["trade_ids"] == ["trade-0", "trade-1"]

    def test_the_model_and_engine_are_the_requests(self, test_client):
        """The Hull-White model and the Jamshidian engine reach the worker: the result differs
        from the LGM/Bachelier run's, and equals the direct call's."""
        jamshidian = {"european": "Jamshidian", "jamshidian": {"reversion": 0.03, "volatility": 0.01}}
        body = _body([_european(trade_id="e")], pricing=jamshidian)
        other = _body([_european(trade_id="e")], simulation=_simulation(model="LGM"))
        result, baseline = _submit_and_poll(test_client, body), _submit_and_poll(test_client, other)
        assert result["base_npv"] == _direct(body).base_npv
        assert result["base_npv"] != baseline["base_npv"]

    def test_greeks_included_when_requested(self, test_client):
        result = _submit_and_poll(test_client, _body([_european()], compute_greeks=True))
        entry = result["greeks"]["0"]
        assert "vega:USD" in entry["values"] and entry["theta"] is not None


@pytest.mark.slow
class TestPortfolioPriceAtScale:
    def test_empty_trades_list_prices_to_a_zero_width_result(self, test_client):
        result = _submit_and_poll(test_client, _body([]))
        assert result["base_npv"] == 0.0 and np.asarray(result["npv_cube"]).shape[-1] == 0

    def test_many_identical_trades_price_identically(self, test_client):
        result = _submit_and_poll(test_client, _body([_swap(trade_id=f"s{i}") for i in range(8)]))
        cube = np.asarray(result["npv_cube"])
        for j in range(1, 8):
            np.testing.assert_array_equal(cube[:, :, j], cube[:, :, 0])

    def test_multiple_concurrent_jobs_do_not_cross_contaminate_results(self, test_client):
        """Two portfolios submitted back to back each get only their own result."""
        job_a = test_client.post("/portfolio/price", json=_body([_swap(notional=1e6)])).json()["job_id"]
        job_b = test_client.post("/portfolio/price", json=_body([_swap(notional=9e6)])).json()["job_id"]
        assert job_a != job_b
        deadline = time.time() + 120
        data = {}
        while time.time() < deadline and not all(data.get(j, {}).get("status") in ("done", "failed")
                                                  for j in (job_a, job_b)):
            data = {j: test_client.get(f"/portfolio/price/{j}").json() for j in (job_a, job_b)}
            time.sleep(0.2)
        npv = [np.asarray(data[j]["result"]["npv_cube"])[:, :, 0] for j in (job_a, job_b)]
        np.testing.assert_allclose(npv[1], 9.0 * npv[0], rtol=1e-12)


class TestPortfolioPriceInvalidPayload:
    def test_the_retired_hull_white_shape_is_a_422_naming_its_replacement(self, test_client):
        body = {"evaluation_date": "2026-07-30", "market": {"time_grid": [0.0, 1.0], "rates": {}},
                "trades": [_swap()]}
        r = test_client.post("/portfolio/price", json=body)
        assert r.status_code == 422
        assert "retired by roadmap 1.3" in r.text and "simulation.ir" in r.text

    def test_a_market_with_equities_is_not_mistaken_for_the_retired_shape(self):
        """The retired market had `equities` too; the current one must not be refused for it
        (the demo's two-currency request, with an equity, was)."""
        body = {"market": demo_market_json(("USD", "EUR")), "trades": [_swap()],
                "simulation": demo_simulation_json(samples=8, currencies=("USD", "EUR"))}
        assert body["market"]["equities"]
        request = MarketPortfolioRequestSchema.model_validate(body).to_dataclass()
        assert set(request.market.equities) == set(body["market"]["equities"])
        body["market"]["equities"] = {}
        MarketPortfolioRequestSchema.model_validate(body)

    def test_a_trade_carrying_model_parameters_is_a_422(self, test_client):
        r = test_client.post("/portfolio/price", json=_body([_european(hw_a=0.03, hw_sigma=0.01)]))
        assert r.status_code == 422 and "hw_a" in r.text

    def test_an_unknown_model_is_a_422(self, test_client):
        body = _body([_swap()])
        body["simulation"]["ir"]["USD"]["model"] = "CIR"
        assert test_client.post("/portfolio/price", json=body).status_code == 422

    def test_an_invalid_correlation_is_a_400_with_the_validators_message(self, test_client):
        body = _body([_swap()])
        body["simulation"] = copy.deepcopy(body["simulation"])
        body["simulation"]["ir"]["EUR"] = {"model": "HullWhite", "reversion": 0.02, "volatility": 0.008}
        r = test_client.post("/portfolio/price", json=body)
        assert 400 <= r.status_code < 500

    def test_a_currency_the_market_lacks_is_a_400_naming_the_trade(self, test_client):
        r = test_client.post("/portfolio/price", json=_body([_swap(currency="GBP", trade_id="gbp-swap")]))
        assert r.status_code == 400 and "gbp-swap" in r.json()["detail"]

    def test_malformed_schema_returns_422(self, test_client):
        assert test_client.post("/portfolio/price", json={"market": 12345}).status_code == 422

    def test_missing_required_field_returns_422(self, test_client):
        assert test_client.post("/portfolio/price", json={"market": shared.market_json()}).status_code == 422

    def test_invalid_trade_type_discriminator_returns_422(self, test_client):
        body = _body([{"trade_type": "not_a_real_type"}])
        assert test_client.post("/portfolio/price", json=body).status_code == 422


class TestPortfolioPriceUnknownJob:
    def test_unknown_job_id_returns_404(self, test_client):
        assert test_client.get("/portfolio/price/not-a-real-job-id").status_code == 404


class TestCalibrationEndpoint:
    """The standalone calibration route (its own Hagan bootstrap on the basket it is given;
    the portfolio's calibration is the CAM's, per currency; roadmap 4.1)."""

    def test_valid_calibration_request_returns_fitted_sigma(self, test_client):
        body = {"evaluation_date": shared.ASOF_ISO, "exercise_times": [1.0, 2.0, 3.0, 4.0], "final_maturity_time": 5.0,
                "notional": 1_000_000.0, "payer": True, "market_vols": [0.0080, 0.0088, 0.0095, 0.0100],
                "zero_curve": ZERO_CURVE_SCHEMA, "hw_a": 0.03}
        r = test_client.post("/calibration/lgm", json=body)
        assert r.status_code == 200, r.text
        assert len(r.json()["sigma_values"]) == 4 and r.json()["rmse"] < 1e-6

    def test_calibration_malformed_schema_returns_422(self, test_client):
        assert test_client.post("/calibration/lgm", json={"evaluation_date": shared.ASOF_ISO}).status_code == 422


class TestPortfolioPricePrecision:
    """The HTTP precision block, `engine.precision.Precision` on the wire (the dataclass level
    is tests/test_precision.py)."""

    @pytest.mark.parametrize("old", [{"simulation": 32}, {"pricing": 64}, {"risk": 32}, {"calibration": 64}])
    def test_the_retired_32_64_shape_is_a_422_naming_the_replacement(self, test_client, old):
        """Refused, not translated (decision A-12)."""
        r = test_client.post("/portfolio/price", json=_body([_swap()], precision=old))
        assert r.status_code == 422 and "retired by roadmap 1.4" in r.text and "StagePrecision" in r.text

    def test_a_format_outside_the_table_is_a_422(self, test_client):
        r = test_client.post("/portfolio/price", json=_body([_swap()], precision={"pricing": {"compute": "fp32"}}))
        assert r.status_code == 422 and "float8_e4m3fn" in r.text

    @pytest.mark.parametrize("stage, block, message", [
        ("pricing", {"storage": "float8_e4m3fn", "compute": "float16", "accumulate": "float16"}, "roadmap step 2.8"),
        ("market", {"compute": "bfloat16"}, "roadmap step 2.8"),
        ("simulation", {"storage": "float64", "compute": "float32", "accumulate": "float32"}, "wider than compute"),
    ])
    def test_a_format_not_enabled_is_a_400_naming_the_stage_and_field(self, test_client, stage, block, message):
        r = test_client.post("/portfolio/price", json=_body([_swap()], precision={stage: block}))
        assert r.status_code == 400 and f"precision.{stage}" in r.json()["detail"] and message in r.json()["detail"]

    def test_overrides_per_product_and_trade_reach_the_run_configuration(self):
        """Roadmap 1.5: `by_product` keyed by `trade_type`, `by_trade` by `trade_id`."""
        f32 = {"storage": "float32", "compute": "float32", "accumulate": "float32"}
        body = _body([_swap(trade_id="s"), _european(trade_id="e")],
                     precision={"by_product": {"european_swaption": f32}, "by_trade": {"s": {"storage": "float32"}}})
        request = MarketPortfolioRequestSchema.model_validate(body).to_dataclass()
        precision = request.config.precision
        assert [precision.precision_for(t) for t in request.trades] == [
            StagePrecision("float32", "float64", "float64"), StagePrecision("float32", "float32", "float32")]

    def test_an_unknown_product_is_a_422_listing_the_products(self, test_client):
        r = test_client.post("/portfolio/price", json=_body([_swap()], precision={"by_product": {"swaption": {}}}))
        assert r.status_code == 422 and "european_swaption" in r.text and "bermudan_swaption" in r.text

    def test_an_override_naming_no_trade_is_a_400(self, test_client):
        """Checked synchronously with the request (`validate_request`): no job is created."""
        r = test_client.post("/portfolio/price", json=_body([_swap(trade_id="s")],
                                                            precision={"by_trade": {"s2": {"storage": "float32"}}}))
        assert r.status_code == 400 and "Precision.by_trade: ['s2'] not the id of a trade" in r.json()["detail"]

    @pytest.mark.parametrize("name", ["by_product", "by_trade"])
    def test_an_override_not_enabled_is_a_400_naming_its_key(self, test_client, name):
        key = "swap" if name == "by_product" else "s"
        r = test_client.post("/portfolio/price", json=_body([_swap(trade_id="s")],
                                                            precision={name: {key: {"compute": "float16"}}}))
        assert r.status_code == 400 and f"precision.{name}['{key}']" in r.json()["detail"]

    def test_scaled_storage_and_its_rounding_reach_the_run_configuration(self):
        """Roadmap 1.6: float16, bfloat16 and FP8 storage, `rounding` and `rounding_seed`."""
        fp8 = {"storage": "float8_e4m3fn", "compute": "float32", "accumulate": "float32"}
        body = _body([_swap()], precision={"simulation": fp8, "pricing": {"storage": "bfloat16"},
                                           "rounding": "stochastic", "rounding_seed": 7})
        precision = MarketPortfolioRequestSchema.model_validate(body).to_dataclass().config.precision
        assert precision == Precision(simulation=StagePrecision("float8_e4m3fn", "float32", "float32"),
                                      pricing=StagePrecision("bfloat16"), rounding="stochastic", rounding_seed=7)

    @pytest.mark.parametrize("precision, status, message", [
        ({"rounding": "up"}, 422, "stochastic"),
        ({"rounding_seed": -1}, 422, "rounding_seed"),
        ({"rounding": "stochastic"}, 400, "no stage or override stores in one"),
    ])
    def test_a_rounding_that_cannot_apply_is_refused(self, test_client, precision, status, message):
        r = test_client.post("/portfolio/price", json=_body([_swap()], precision=precision))
        assert r.status_code == status and message in r.text

    @pytest.mark.slow
    def test_omitted_precision_equals_explicit_float64(self, test_client):
        f64 = {"storage": "float64", "compute": "float64", "accumulate": "float64"}
        a = _submit_and_poll(test_client, _body([_swap()]))
        b = _submit_and_poll(test_client, _body([_swap()], precision={s: f64 for s in ("simulation", "market",
                                                                                         "pricing")}))
        np.testing.assert_array_equal(np.asarray(a["npv_cube"]), np.asarray(b["npv_cube"]))


@pytest.mark.slow
class TestPortfolioPriceWorkerPoolDispatch:
    """`/portfolio/price` dispatches through `engine.portfolio.worker_pool`, one pool for every
    precision."""

    def test_two_precisions_submitted_back_to_back_both_complete_correctly(self, test_client):
        f32 = {"storage": "float32", "compute": "float32", "accumulate": "float32"}
        body_64 = _body([_swap(), _european()])
        body_32 = _body([_swap(), _european()], precision={"simulation": f32, "market": f32, "pricing": f32})
        job_64 = test_client.post("/portfolio/price", json=body_64).json()["job_id"]
        job_32 = test_client.post("/portfolio/price", json=body_32).json()["job_id"]
        deadline = time.time() + 180
        data = {}
        while time.time() < deadline and not all(data.get(j, {}).get("status") in ("done", "failed")
                                                  for j in (job_64, job_32)):
            data = {j: test_client.get(f"/portfolio/price/{j}").json() for j in (job_64, job_32)}
            time.sleep(0.1)
        r64, r32 = data[job_64]["result"], data[job_32]["result"]
        np.testing.assert_array_equal(np.asarray(r64["npv_cube"]), np.asarray(_direct(body_64).npv_cube))
        np.testing.assert_array_equal(np.asarray(r32["npv_cube"]), np.asarray(_direct(body_32).npv_cube))
        assert r64["npv_cube"] != r32["npv_cube"], "the float32 job was not priced in float32"

    def test_job_id_maps_to_a_future_not_an_eagerly_computed_result(self, test_client):
        """The route returns as soon as it has a `Future`; the 202 rules out blocking."""
        r = test_client.post("/portfolio/price", json=_body([_swap(), _european()], simulation=_simulation(2048)))
        assert r.status_code == 202, r.text
        assert test_client.get(f"/portfolio/price/{r.json()['job_id']}").json()["status"] in ("pending", "done")


@pytest.mark.slow
class TestGapFixesSurviveTheHttpBoundary:
    """The fixes of tests/test_portfolio_gap_fixes.py survive serialization and the
    worker-process boundary."""

    def test_swap_greeks_present_over_http(self, test_client):
        result = _submit_and_poll(test_client, _body([_swap()], compute_greeks=True))
        entry = result["greeks"]["0"]  # JSON object keys are strings
        assert "delta:discount:USD" in entry["values"] and f"delta:index:{shared.INDEX}" in entry["values"]
        assert entry["theta"] is not None

    def test_per_trade_base_npv_present_and_reconciles_over_http(self, test_client):
        result = _submit_and_poll(test_client, _body([_swap(), _swap(notional=250_000.0)]))
        assert len(result["base_npv_per_trade"]) == 2
        assert result["base_npv"] == pytest.approx(sum(result["base_npv_per_trade"]), rel=0.0, abs=1e-9)
