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
