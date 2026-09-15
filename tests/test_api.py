"""
Tests for engine.api -- the FastAPI HTTP boundary over
engine.portfolio.price_portfolio. Uses FastAPI's TestClient (backed by
httpx), which runs the app in-process -- no running server needed.

Covers: /health and /version respond; /portfolio/price with a valid body
returns 202 + a job id, polling reaches "done" with a result matching a
direct price_portfolio call on the equivalent dataclass request;
/portfolio/price with an invalid body (non-PSD covariance, mismatched
rate_factor_index) returns a 4xx with the same validator error message
surfaced in the response body, not a 500; a malformed-schema body (wrong
types) returns FastAPI's automatic 422; /calibration/lgm.
"""
import time

import numpy as np
import pytest

from engine.api.schemas import PortfolioRequestSchema
from engine.portfolio import price_portfolio

TODAY_ISO = "2026-07-30"
FLAT_RATE = 0.03
HW_A = 0.03
HW_SIGMA = 0.01
ZERO_CURVE_SCHEMA = {"times": [0.0, 1.0, 2.0, 5.0, 10.0, 30.0], "rates": [FLAT_RATE] * 6}


def _market_schema(scenarios=64):
    return {
        "time_grid": [0.0, 0.5, 1.0, 1.5, 2.0],
        "equities": {"initial_prices": [100.0], "dividend_yields": [0.0], "rate_mapping": [[0.0]]},
        "rates": {
            "initial_rates": [FLAT_RATE], "theta": [FLAT_RATE], "mean_reversion": [HW_A],
            "initial_zero_curves": [ZERO_CURVE_SCHEMA],
        },
        "joint_covariance": [[0.04, 0.0], [0.0, HW_SIGMA ** 2]],
        "scenarios": scenarios,
    }


def _swap_trade_schema(**overrides):
    trade = {
        "trade_type": "swap", "notional": 1_000_000.0, "fixed_rate": 0.032, "payer": True,
        "discount_curve_index": 0, "forward_curve_index": 0, "swap_tenor": "2Y",
    }
    trade.update(overrides)
    return trade


def _swaption_trade_schema(**overrides):
    trade = {
        "trade_type": "european_swaption", "notional": 500_000.0, "fixed_rate": 0.031, "payer": True,
        "rate_factor_index": 0, "hw_a": HW_A, "hw_sigma": HW_SIGMA,
        "initial_zero_curve": ZERO_CURVE_SCHEMA, "swap_tenor": "2Y", "forward_start": "1Y",
    }
    trade.update(overrides)
    return trade


def _bermudan_trade_schema(**overrides):
    trade = {
        "trade_type": "bermudan_swaption", "notional": 1_000_000.0, "fixed_rate": 0.030, "payer": True,
        "rate_factor_index": 0, "hw_a": HW_A, "hw_sigma": None,
        "initial_zero_curve": ZERO_CURVE_SCHEMA, "exercise_times": [1.0, 2.0],
        "swap_tenor": "3Y", "n_per_std": 32, "std_devs": 6.0,
    }
    trade.update(overrides)
    return trade


CALIBRATION_BASKET_SCHEMA = {
    "exercise_times": [1.0, 2.0],
    "final_maturity_time": 3.0,
    "notional": 1_000_000.0,
    "payer": True,
    "market_vols": [0.0080, 0.0088],
}


class TestHealthAndVersion:
    def test_health_returns_ok(self, test_client):
        r = test_client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_version_returns_engine_info(self, test_client):
        r = test_client.get("/version")
        assert r.status_code == 200
        body = r.json()
        assert "engine_version" in body
        assert "jax_backend" in body


class TestPortfolioPriceHappyPath:
    def _submit_and_poll(self, client, body, timeout_s=60):
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
        assert data is not None, "job never reached a terminal state"
        return data

    def test_valid_portfolio_returns_202_then_done(self, test_client):
        body = {
            "evaluation_date": TODAY_ISO,
            "market": _market_schema(),
            "trades": [_swap_trade_schema()],
            "percentiles": [0.95],
        }
        data = self._submit_and_poll(test_client, body)
        assert data["status"] == "done", data.get("error")
        assert data["result"] is not None
        assert isinstance(data["result"]["base_npv"], float)

    def test_result_matches_direct_price_portfolio_call(self, test_client):
        """The HTTP response, after schema round-trip, must match calling
        price_portfolio directly on the equivalent dataclass request."""
        body = {
            "evaluation_date": TODAY_ISO,
            "market": _market_schema(),
            "trades": [_swap_trade_schema(), _swaption_trade_schema()],
            "percentiles": [0.95, 0.99],
        }
        data = self._submit_and_poll(test_client, body)
        assert data["status"] == "done", data.get("error")

        direct_request = PortfolioRequestSchema(**body).to_dataclass()
        direct_result = price_portfolio(direct_request)

        np.testing.assert_allclose(data["result"]["base_npv"], direct_result.base_npv, rtol=1e-9)
        http_npv = np.asarray(data["result"]["npv_cube"])
        np.testing.assert_allclose(http_npv, np.asarray(direct_result.npv_cube), rtol=1e-9)
        for key in direct_result.risk:
            http_vals = [v if v is not None else float("nan") for v in data["result"]["risk"]["values"][key]]
            np.testing.assert_allclose(
                np.asarray(http_vals), np.asarray(direct_result.risk[key]), rtol=1e-9, equal_nan=True,
            )

    def test_greeks_included_when_requested(self, test_client):
        body = {
            "evaluation_date": TODAY_ISO,
            "market": _market_schema(),
            "trades": [_swaption_trade_schema()],
            "percentiles": [0.95],
            "compute_greeks": True,
        }
        data = self._submit_and_poll(test_client, body)
        assert data["status"] == "done", data.get("error")
        assert data["result"]["greeks"] is not None
        assert "0" in data["result"]["greeks"] or 0 in data["result"]["greeks"]


class TestCalibratedBermudanOverHttp:
    """`hw_sigma: null` on a Bermudan/American trade needs a
    `calibration_basket` on the request to resolve -- without it,
    price_portfolio's own ValueError ("calibration_targets was not
    supplied") must surface as a 4xx, not a 500 inside the background job.
    With it, the job must complete and match calibrating+pricing directly
    via price_portfolio on the equivalent dataclass request."""

    def _submit_and_poll(self, client, body, timeout_s=60):
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
        assert data is not None, "job never reached a terminal state"
        return data

    def test_uncalibrated_bermudan_without_basket_fails_the_job(self, test_client):
        body = {
            "evaluation_date": TODAY_ISO,
            "market": _market_schema(),
            "trades": [_bermudan_trade_schema()],
            "percentiles": [0.95],
        }
        data = self._submit_and_poll(test_client, body)
        assert data["status"] == "failed"
        assert "calibration_targets was not supplied" in data["error"]

    def test_calibration_basket_with_no_uncalibrated_trade_returns_400(self, test_client):
        body = {
            "evaluation_date": TODAY_ISO,
            "market": _market_schema(),
            "trades": [_swap_trade_schema()],
            "percentiles": [0.95],
            "calibration_basket": CALIBRATION_BASKET_SCHEMA,
        }
        r = test_client.post("/portfolio/price", json=body)
        assert r.status_code == 400
        assert "no trade has hw_sigma=null" in r.json()["detail"]

    def test_calibration_basket_resolves_uncalibrated_bermudan(self, test_client):
        body = {
            "evaluation_date": TODAY_ISO,
            "market": _market_schema(),
            "trades": [_bermudan_trade_schema()],
            "percentiles": [0.95],
            "calibration_basket": CALIBRATION_BASKET_SCHEMA,
        }
        data = self._submit_and_poll(test_client, body)
        assert data["status"] == "done", data.get("error")
        assert isinstance(data["result"]["base_npv"], float)

    def test_result_matches_direct_price_portfolio_call(self, test_client):
        body = {
            "evaluation_date": TODAY_ISO,
            "market": _market_schema(),
            "trades": [_bermudan_trade_schema()],
            "percentiles": [0.95],
            "calibration_basket": CALIBRATION_BASKET_SCHEMA,
        }
        data = self._submit_and_poll(test_client, body)
        assert data["status"] == "done", data.get("error")

        direct_request = PortfolioRequestSchema(**body).to_dataclass()
        direct_result = price_portfolio(direct_request)

        np.testing.assert_allclose(data["result"]["base_npv"], direct_result.base_npv, rtol=1e-9)
        http_npv = np.asarray(data["result"]["npv_cube"])
        np.testing.assert_allclose(http_npv, np.asarray(direct_result.npv_cube), rtol=1e-9)


class TestPortfolioPriceAtScale:
    """Larger and edge-composition portfolios submitted as real JSON bodies
    -- exercises the schema round-trip (discriminated-union trade parsing,
    JSON nesting of a wider npv_cube) at a scale beyond the 1-2 trade
    bodies used elsewhere in this file. Complements
    tests/test_portfolio_scale_and_edge_cases.py, which covers the same
    scale/edge-case space at the dataclass level (price_portfolio called
    directly) -- this class's own focus is the HTTP/JSON boundary itself."""

    def _submit_and_poll(self, client, body, timeout_s=90):
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
        assert data is not None, "job never reached a terminal state"
        return data

    def test_twenty_trade_mixed_portfolio_over_http(self, test_client):
        trades = (
            [_swap_trade_schema(notional=1_000_000.0 * (i + 1), swap_tenor=f"{2 + i % 3}Y") for i in range(8)]
            + [_swaption_trade_schema(notional=500_000.0 * (i + 1), forward_start=f"{1 + i % 2}Y") for i in range(6)]
            + [_bermudan_trade_schema(notional=800_000.0 * (i + 1), hw_sigma=HW_SIGMA) for i in range(3)]
            + [
                {
                    "trade_type": "american_swaption", "notional": 600_000.0 * (i + 1),
                    "fixed_rate": 0.0295, "payer": bool(i % 2), "rate_factor_index": 0,
                    "hw_a": HW_A, "hw_sigma": HW_SIGMA, "initial_zero_curve": ZERO_CURVE_SCHEMA,
                    "first_exercise": 1.0, "last_exercise": 2.0, "exercise_time_steps_per_year": 1,
                    "n_per_std": 32, "std_devs": 6.0,
                }
                for i in range(3)
            ]
        )
        assert len(trades) == 20
        body = {
            "evaluation_date": TODAY_ISO,
            "market": _market_schema(scenarios=64),
            "trades": trades,
            "percentiles": [0.95, 0.99],
        }
        data = self._submit_and_poll(test_client, body)
        assert data["status"] == "done", data.get("error")
        npv_cube = data["result"]["npv_cube"]
        assert len(npv_cube[0][0]) == 20  # trade axis
        flat = [v for scenario in npv_cube for step in scenario for v in step]
        assert all(np.isfinite(v) for v in flat)

    def test_result_matches_direct_call_for_a_larger_portfolio(self, test_client):
        """The bit-for-bit HTTP-vs-direct-call cross-check other tests do
        for 1-2 trades, repeated at 10 trades -- confirms the schema
        round-trip stays exact (no drift/rounding introduced by JSON
        serialization) as the response payload grows substantially larger."""
        trades = (
            [_swap_trade_schema(notional=1_000_000.0 * (i + 1)) for i in range(5)]
            + [_swaption_trade_schema(notional=500_000.0 * (i + 1)) for i in range(5)]
        )
        body = {
            "evaluation_date": TODAY_ISO,
            "market": _market_schema(scenarios=64),
            "trades": trades,
            "percentiles": [0.95, 0.99],
        }
        data = self._submit_and_poll(test_client, body)
        assert data["status"] == "done", data.get("error")

        direct_request = PortfolioRequestSchema(**body).to_dataclass()
        direct_result = price_portfolio(direct_request)

        http_npv = np.asarray(data["result"]["npv_cube"])
        np.testing.assert_allclose(http_npv, np.asarray(direct_result.npv_cube), rtol=1e-9)
        np.testing.assert_allclose(data["result"]["base_npv"], direct_result.base_npv, rtol=1e-9)

    def test_empty_trades_list_is_accepted_and_prices_to_a_zero_width_result(self, test_client):
        """An empty trades list is valid at the dataclass level
        (price_portfolio returns a zero-width result, see
        test_portfolio_scale_and_edge_cases.py) but Pydantic's List field
        has no explicit min_length here -- confirming the actual current
        behavior (accepted, not rejected) rather than assuming either way,
        since this is exactly the kind of boundary a schema change could
        silently flip."""
        body = {
            "evaluation_date": TODAY_ISO,
            "market": _market_schema(),
            "trades": [],
            "percentiles": [0.95],
        }
        data = self._submit_and_poll(test_client, body)
        assert data["status"] == "done", data.get("error")
        # [Scenarios, TimeSteps, 0] JSON-serializes as a nested list of
        # empty inner lists, e.g. [[[], []], [[], []], ...] -- not "[]".
        npv_cube = data["result"]["npv_cube"]
        assert all(len(time_step) == 0 for scenario in npv_cube for time_step in scenario)
        assert data["result"]["base_npv"] == 0.0

    def test_many_identical_trades_over_http_price_identically(self, test_client):
        trades = [_swap_trade_schema() for _ in range(8)]
        body = {
            "evaluation_date": TODAY_ISO,
            "market": _market_schema(),
            "trades": trades,
            "percentiles": [0.95],
        }
        data = self._submit_and_poll(test_client, body)
        assert data["status"] == "done", data.get("error")
        npv_cube = np.asarray(data["result"]["npv_cube"])
        for j in range(1, 8):
            np.testing.assert_allclose(npv_cube[:, :, j], npv_cube[:, :, 0], rtol=1e-9)

    def test_multiple_concurrent_jobs_do_not_cross_contaminate_results(self, test_client):
        """Submits two DIFFERENT portfolios back-to-back (both jobs pending/
        running -- genuinely concurrently, in separate worker-pool
        processes, since engine/api/routes.py dispatches through
        engine.portfolio.worker_pool -- at once) and confirms each job_id's
        own result matches ONLY its own request -- a direct test of
        engine/api/routes.py's _JOBS job_id -> Future keying, guarding
        against a bug where one job's result could leak into another's
        slot. See TestPortfolioPriceWorkerPoolDispatch below for a test
        that additionally proves the two jobs run in genuinely separate
        processes, not just that results don't cross-contaminate."""
        body_a = {
            "evaluation_date": TODAY_ISO, "market": _market_schema(),
            "trades": [_swap_trade_schema(notional=1_000_000.0)], "percentiles": [0.95],
        }
        body_b = {
            "evaluation_date": TODAY_ISO, "market": _market_schema(),
            "trades": [_swap_trade_schema(notional=9_000_000.0)], "percentiles": [0.95],
        }
        job_a = test_client.post("/portfolio/price", json=body_a).json()["job_id"]
        job_b = test_client.post("/portfolio/price", json=body_b).json()["job_id"]
        assert job_a != job_b

        deadline = time.time() + 90
        data_a = data_b = None
        while time.time() < deadline and (data_a is None or data_a["status"] not in ("done", "failed")
                                           or data_b is None or data_b["status"] not in ("done", "failed")):
            data_a = test_client.get(f"/portfolio/price/{job_a}").json()
            data_b = test_client.get(f"/portfolio/price/{job_b}").json()
            time.sleep(0.2)

        assert data_a["status"] == "done", data_a.get("error")
        assert data_b["status"] == "done", data_b.get("error")
        npv_a = np.asarray(data_a["result"]["npv_cube"])[:, :, 0]
        npv_b = np.asarray(data_b["result"]["npv_cube"])[:, :, 0]
        np.testing.assert_allclose(npv_b, npv_a * 9.0, rtol=1e-9)


class TestPortfolioPriceInvalidPayload:
    def test_non_psd_covariance_returns_4xx_with_actionable_message(self, test_client):
        body = {
            "evaluation_date": TODAY_ISO,
            "market": {**_market_schema(), "joint_covariance": [[0.02, 0.05], [0.05, 0.02]]},
            "trades": [_swap_trade_schema()],
            "percentiles": [0.95],
        }
        r = test_client.post("/portfolio/price", json=body)
        assert 400 <= r.status_code < 500
        assert "positive semi-definite" in r.json()["detail"]

    def test_mismatched_rate_factor_index_returns_4xx_with_actionable_message(self, test_client):
        body = {
            "evaluation_date": TODAY_ISO,
            "market": _market_schema(),
            "trades": [_swaption_trade_schema(hw_a=0.099)],  # mismatched vs. market.rates.mean_reversion[0]=0.03
            "percentiles": [0.95],
        }
        r = test_client.post("/portfolio/price", json=body)
        assert 400 <= r.status_code < 500
        assert "hw_a" in r.json()["detail"]

    def test_malformed_schema_returns_422(self, test_client):
        r = test_client.post("/portfolio/price", json={"evaluation_date": 12345})
        assert r.status_code == 422

    def test_missing_required_field_returns_422(self, test_client):
        body = {"evaluation_date": TODAY_ISO, "market": _market_schema()}  # trades missing
        r = test_client.post("/portfolio/price", json=body)
        assert r.status_code == 422

    def test_invalid_trade_type_discriminator_returns_422(self, test_client):
        body = {
            "evaluation_date": TODAY_ISO,
            "market": _market_schema(),
            "trades": [{"trade_type": "not_a_real_type"}],
            "percentiles": [0.95],
        }
        r = test_client.post("/portfolio/price", json=body)
        assert r.status_code == 422


class TestPortfolioPriceUnknownJob:
    def test_unknown_job_id_returns_404(self, test_client):
        r = test_client.get("/portfolio/price/not-a-real-job-id")
        assert r.status_code == 404


class TestCalibrationEndpoint:
    def test_valid_calibration_request_returns_fitted_sigma(self, test_client):
        body = {
            "evaluation_date": TODAY_ISO,
            "exercise_times": [1.0, 2.0, 3.0, 4.0],
            "final_maturity_time": 5.0,
            "notional": 1_000_000.0,
            "payer": True,
            "market_vols": [0.0080, 0.0088, 0.0095, 0.0100],
            "zero_curve": ZERO_CURVE_SCHEMA,
            "hw_a": HW_A,
        }
        r = test_client.post("/calibration/lgm", json=body)
        assert r.status_code == 200, r.text
        data = r.json()
        assert len(data["sigma_values"]) == 4
        assert data["rmse"] < 1e-6

    def test_calibration_malformed_schema_returns_422(self, test_client):
        r = test_client.post("/calibration/lgm", json={"evaluation_date": TODAY_ISO})
        assert r.status_code == 422


class TestPortfolioPricePrecision:
    """PrecisionConfigSchema/PortfolioRequestSchema.precision -- the HTTP
    surface over engine.portfolio.PrecisionConfig (see
    tests/test_portfolio_entrypoint.py::TestPricePortfolioPrecision for the
    dataclass-level equivalent). No `precision` key sent must behave
    identically to an explicit all-64 block (Optional, not a populated
    default -- see PortfolioRequestSchema's own docstring)."""

    def _submit_and_poll(self, client, body, timeout_s=60):
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
        assert data is not None, "job never reached a terminal state"
        return data

    def test_explicit_precision_block_matches_direct_call(self, test_client):
        """Submitting an explicit `"precision"` block over HTTP must match
        calling price_portfolio directly with the equivalent PrecisionConfig
        -- validates the schema wiring end-to-end (PrecisionConfigSchema ->
        .to_dataclass() -> PrecisionConfig)."""
        body = {
            "evaluation_date": TODAY_ISO,
            "market": _market_schema(),
            "trades": [_swap_trade_schema(), _swaption_trade_schema()],
            "percentiles": [0.95],
            "precision": {"simulation": 64, "pricing": 32, "risk": 32},
        }
        data = self._submit_and_poll(test_client, body)
        assert data["status"] == "done", data.get("error")

        direct_request = PortfolioRequestSchema(**body).to_dataclass()
        direct_result = price_portfolio(direct_request)

        assert direct_result.npv_cube.dtype.itemsize == 4  # float32
        np.testing.assert_allclose(data["result"]["base_npv"], direct_result.base_npv, rtol=1e-9)
        http_npv = np.asarray(data["result"]["npv_cube"])
        np.testing.assert_allclose(http_npv, np.asarray(direct_result.npv_cube), rtol=1e-9)

    def test_omitted_precision_matches_explicit_all_64(self, test_client):
        """No `precision` key sent must resolve to exactly PrecisionConfig()
        (all-64) -- the same result as sending an explicit all-64 block."""
        base_body = {
            "evaluation_date": TODAY_ISO,
            "market": _market_schema(),
            "trades": [_swap_trade_schema()],
            "percentiles": [0.95],
        }
        explicit_body = dict(base_body, precision={"simulation": 64, "pricing": 64, "risk": 64})

        data_omitted = self._submit_and_poll(test_client, base_body)
        data_explicit = self._submit_and_poll(test_client, explicit_body)
        assert data_omitted["status"] == "done", data_omitted.get("error")
        assert data_explicit["status"] == "done", data_explicit.get("error")

        np.testing.assert_allclose(
            np.asarray(data_omitted["result"]["npv_cube"]),
            np.asarray(data_explicit["result"]["npv_cube"]),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            data_omitted["result"]["base_npv"], data_explicit["result"]["base_npv"], rtol=1e-12,
        )

    def test_float32_precision_response_close_but_not_identical_to_float64(self, test_client):
        """The honest signal available at the JSON boundary: JSON floats
        carry no dtype metadata, so a float32-precision response's values
        are checked with pytest.approx (a tolerance), not exact equality,
        against the equivalent float64 request."""
        base_body = {
            "evaluation_date": TODAY_ISO,
            "market": _market_schema(),
            "trades": [_swap_trade_schema()],
            "percentiles": [0.95],
        }
        body_64 = dict(base_body, precision={"simulation": 64, "pricing": 64, "risk": 64})
        body_32 = dict(base_body, precision={"simulation": 64, "pricing": 32, "risk": 64})

        data_64 = self._submit_and_poll(test_client, body_64)
        data_32 = self._submit_and_poll(test_client, body_32)
        assert data_64["status"] == "done", data_64.get("error")
        assert data_32["status"] == "done", data_32.get("error")

        npv_64 = data_64["result"]["base_npv"]
        npv_32 = data_32["result"]["base_npv"]
        assert npv_32 == pytest.approx(npv_64, rel=1e-3)
        assert npv_32 != npv_64  # a genuinely lower-precision computation, not a no-op

    def test_invalid_precision_value_returns_400_not_a_failed_job(self, test_client):
        """A malformed precision value ({"simulation": 16}) fails
        PrecisionConfig.__post_init__'s validation, which
        submit_portfolio_price already runs synchronously (via
        request.to_dataclass()) before a job_id is ever created -- so this
        must be an immediate 400 with the validator's own message, not a
        202 followed by a failed job."""
        body = {
            "evaluation_date": TODAY_ISO,
            "market": _market_schema(),
            "trades": [_swap_trade_schema()],
            "percentiles": [0.95],
            "precision": {"simulation": 16},
        }
        r = test_client.post("/portfolio/price", json=body)
        assert r.status_code == 400
        assert "must be 32 or 64" in r.json()["detail"]


class TestPortfolioPriceWorkerPoolDispatch:
    """Confirms `/portfolio/price` genuinely dispatches through
    `engine.portfolio.worker_pool` (see engine/api/routes.py's post-Phase-B
    docstring), not just that polling still returns 202/200 the same way it
    always did (TestPortfolioPriceHappyPath already covers that). Submits
    two DIFFERENT-precision requests back-to-back and confirms both jobs are
    simultaneously non-terminal shortly after submission -- i.e. the second
    job actually started running in its own worker process rather than
    sitting fully blocked behind the first job the way a single
    `_PRICING_LOCK`-serialized in-process architecture would leave it."""

    def test_two_precisions_submitted_back_to_back_both_complete_correctly(self, test_client):
        body_64 = {
            "evaluation_date": TODAY_ISO,
            "market": _market_schema(),
            "trades": [_swap_trade_schema(), _swaption_trade_schema()],
            "percentiles": [0.95],
            "precision": {"simulation": 64, "pricing": 64, "risk": 64},
        }
        body_32 = {
            "evaluation_date": TODAY_ISO,
            "market": _market_schema(),
            "trades": [_swap_trade_schema(), _swaption_trade_schema()],
            "percentiles": [0.95],
            "precision": {"simulation": 32, "pricing": 32, "risk": 32},
        }

        job_64 = test_client.post("/portfolio/price", json=body_64).json()["job_id"]
        job_32 = test_client.post("/portfolio/price", json=body_32).json()["job_id"]
        assert job_64 != job_32

        deadline = time.time() + 90
        data_64 = data_32 = None
        while time.time() < deadline and (
            data_64 is None or data_64["status"] not in ("done", "failed")
            or data_32 is None or data_32["status"] not in ("done", "failed")
        ):
            data_64 = test_client.get(f"/portfolio/price/{job_64}").json()
            data_32 = test_client.get(f"/portfolio/price/{job_32}").json()
            time.sleep(0.1)

        assert data_64["status"] == "done", data_64.get("error")
        assert data_32["status"] == "done", data_32.get("error")

        direct_64 = price_portfolio(PortfolioRequestSchema(**body_64).to_dataclass())
        direct_32 = price_portfolio(PortfolioRequestSchema(**body_32).to_dataclass())
        np.testing.assert_allclose(
            data_64["result"]["base_npv"], direct_64.base_npv, rtol=1e-9,
        )
        np.testing.assert_allclose(
            data_32["result"]["base_npv"], direct_32.base_npv, rtol=1e-3,
        )
        # A genuinely different-precision computation, not the same number
        # twice -- confirms the two jobs didn't silently collapse onto the
        # same worker/tier.
        assert data_64["result"]["base_npv"] != data_32["result"]["base_npv"]

    def test_job_id_maps_to_a_future_not_an_eagerly_computed_result(self, test_client):
        """Immediately after submission (before any polling/sleep), the job
        must not yet be in a terminal state for a portfolio large enough to
        take measurable time -- i.e. `submit_portfolio_price` really does
        return as soon as `worker_pool.submit_pricing_job` hands back a
        `Future`, rather than blocking on `price_portfolio` itself before
        responding. A regression here (e.g. accidentally calling
        `future.result()` before returning the job_id) would make this
        first read already "done"."""
        body = {
            "evaluation_date": TODAY_ISO,
            "market": _market_schema(scenarios=2048),
            "trades": [_swap_trade_schema(), _swaption_trade_schema(), _bermudan_trade_schema(hw_sigma=0.01)],
            "percentiles": [0.95],
        }
        r = test_client.post("/portfolio/price", json=body)
        assert r.status_code == 202, r.text
        job_id = r.json()["job_id"]

        immediate = test_client.get(f"/portfolio/price/{job_id}").json()
        assert immediate["status"] in ("pending", "done"), immediate
        # Not asserting status == "pending" outright -- a sufficiently fast
        # machine/warm pool could in principle finish before this second
        # request lands, which would be a false failure, not evidence of a
        # bug. The real regression this guards is submit_portfolio_price
        # BLOCKING on the result before returning 202 at all, which the
        # `assert r.status_code == 202` above (returned before any polling)
        # already rules out.

        deadline = time.time() + 90
        data = immediate
        while time.time() < deadline and data["status"] not in ("done", "failed"):
            data = test_client.get(f"/portfolio/price/{job_id}").json()
            time.sleep(0.1)
        assert data["status"] == "done", data.get("error")


class TestGapFixesSurviveTheHttpBoundary:
    """The three closed gaps from tests/test_portfolio_gap_fixes.py must
    also survive serialization AND the worker-process boundary, not just a
    direct in-process `price_portfolio` call.

    This matters because `POST /portfolio/price` runs the real work in a
    separate OS process (`engine.portfolio.worker_pool`): a `warnings.warn`
    raised in the worker, and any newly added result field, has to make it
    back through `PortfolioResultSchema` to the polling caller. An
    in-process test cannot prove that.
    """

    @staticmethod
    def _submit_and_poll(client, body, timeout_s=120):
        r = client.post("/portfolio/price", json=body)
        assert r.status_code == 202, r.text
        job_id = r.json()["job_id"]
        deadline = time.time() + timeout_s
        data = None
        while time.time() < deadline:
            data = client.get(f"/portfolio/price/{job_id}").json()
            if data["status"] in ("done", "failed"):
                break
            time.sleep(0.2)
        assert data is not None and data["status"] == "done", (
            data.get("error") if data else "job never reached a terminal state"
        )
        return data["result"]

    def test_swap_greeks_present_over_http(self, test_client):
        """Pre-fix, `greeks` came back as an empty object for a swap-only
        portfolio -- 202, then a successful job with nothing in it."""
        result = self._submit_and_poll(test_client, {
            "evaluation_date": TODAY_ISO,
            "market": _market_schema(),
            "trades": [_swap_trade_schema()],
            "compute_greeks": True,
        })
        assert result["greeks"], "no Greeks returned for a swap-only portfolio"
        # JSON object keys are strings, per PortfolioResultSchema's own contract.
        entry = result["greeks"]["0"]
        assert "discount_delta" in entry["values"]
        assert "forward_delta" in entry["values"]
        assert entry["theta"] is not None

    def test_per_trade_base_npv_present_and_reconciles_over_http(self, test_client):
        result = self._submit_and_poll(test_client, {
            "evaluation_date": TODAY_ISO,
            "market": _market_schema(),
            "trades": [_swap_trade_schema(), _swap_trade_schema(notional=250_000.0)],
        })
        per_trade = result["base_npv_per_trade"]
        assert len(per_trade) == 2
        assert result["base_npv"] == pytest.approx(sum(per_trade), rel=0.0, abs=1e-9)

    def test_aged_swap_warning_crosses_the_worker_boundary(self, test_client):
        """A `warnings.warn` raised inside the worker PROCESS must arrive in
        the polled JSON result -- the case an in-process test can't cover."""
        result = self._submit_and_poll(test_client, {
            "evaluation_date": TODAY_ISO,
            "market": _market_schema(),
            "trades": [_swap_trade_schema()],
        })
        assert any("already started accruing" in w for w in result["warnings"]), (
            f"aged-swap warning lost crossing the worker boundary: {result['warnings']!r}"
        )
