"""
`POST /portfolio/market-risk` (I-56): market-risk VaR and ES over HTTP, queued for
the engine worker as a portfolio is. On the shared portfolio's market the polled result equals a
direct `run_market_risk` call, for Monte Carlo and historical scenarios, each trade's row keyed
by its id; the P&L by reference is the inline P&L; a request the run would refuse is a 400
before any job starts, naming the field.
"""
import time

import numpy as np
import pytest

from engine.api.artifacts import read_array
from engine.api.market_schemas import MarketRiskRequestSchema
from engine.market_risk import RateRiskFactors, run_market_risk
from tests.support import portfolio as shared

TRADES = ("swap-payer", "european-payer", "bond")


def _labels():
    return RateRiskFactors.from_market(shared.market()).labels()


def _covariance(vol=0.0025, decay=3.0):
    """Absolute 10-day pillar moves: equal vols, correlation decaying with pillar distance, the
    two curves' matching pillars 95% correlated."""
    n = len(shared.PILLARS)
    i = np.arange(2 * n)
    corr = np.exp(-np.abs(i[:, None] % n - i[None, :] % n) / decay) * np.where(i[:, None] // n == i[None, :] // n,
                                                                              1.0, 0.95)
    np.fill_diagonal(corr, 1.0)
    return (corr * vol ** 2).tolist()


def _history(days=40, seed=7):
    """Factor levels on consecutive dates: today's rates plus a seeded random walk."""
    base = np.concatenate([shared.DISCOUNT, shared.FORWARDING])
    steps = np.random.default_rng(seed).normal(0.0, 0.0006, size=(days, base.size))
    return (base + np.cumsum(steps, axis=0)).tolist()


MONTE_CARLO = {"source": "monte-carlo", "factors": _labels(), "covariance": _covariance(), "horizon_days": 10,
               "num_scenarios": 256, "seed": 3}
HISTORICAL = {"source": "historical", "factors": _labels(), "history": _history(), "horizon_days": 10,
              "dates": [f"d{i}" for i in range(40)]}


def _body(scenarios=MONTE_CARLO, trades=TRADES, **overrides):
    body = {"market": shared.market_json(), "trades": [shared.trades_json()[n] for n in trades],
            "scenarios": scenarios, "quantiles": [0.99, 0.975]}
    body.update(overrides)
    return body


def _submit_and_poll(client, body, timeout_s=600):
    r = client.post("/portfolio/market-risk", json=body)
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        data = client.get(f"/jobs/{job_id}").json()
        if data["status"] in ("done", "failed", "interrupted"):
            assert data["status"] == "done", data["error"]
            assert data["kind"] == "market-risk"
            return data["result"]
        time.sleep(0.2)
    pytest.fail("job never reached a terminal state")


@pytest.mark.slow
@pytest.mark.parametrize("scenarios", [MONTE_CARLO, HISTORICAL], ids=["monte-carlo", "historical"])
def test_the_result_equals_a_direct_run(test_client, scenarios):
    body = _body(scenarios)
    result = _submit_and_poll(test_client, body)
    direct = run_market_risk(MarketRiskRequestSchema.model_validate(body).to_dataclass())
    assert [t["trade_id"] for t in result["trades"]] == list(TRADES)
    assert [t["base_npv"] for t in result["trades"]] == direct.base_npv_per_trade
    np.testing.assert_array_equal(np.asarray(result["pnl"]), np.asarray(direct.pnl))
    assert result["risk"] == {k: (None if np.isnan(v) else v) for k, v in direct.risk.items()}
    assert result["risk"]["VaR_99"] > 0.0 and result["source"] == scenarios["source"]
    assert result["risk_factors"] == _labels() and result["num_scenarios"] == direct.num_scenarios
    assert result["warnings"] == direct.warnings and result["precision"]["paths"] == direct.num_scenarios


@pytest.mark.slow
def test_the_pnl_by_reference_is_the_inline_pnl(test_client):
    inline = _submit_and_poll(test_client, _body(trades=("swap-payer", "bond")))
    by_reference = _submit_and_poll(test_client, _body(trades=("swap-payer", "bond"), pnl_output="artifact"))
    reference = by_reference["pnl_artifact"]
    assert by_reference["pnl"] is None and reference["axes"] == ["scenario", "trade"]
    assert reference["items"]["ids"] == ["swap-payer", "bond"]
    pnl = read_array(reference, lambda url: test_client.get(url).content)
    np.testing.assert_array_equal(pnl, np.asarray(inline["pnl"]))
    assert by_reference["risk"] == inline["risk"]


def _swapped(labels, i, j):
    labels = list(labels)
    labels[i], labels[j] = labels[j], labels[i]
    return labels


@pytest.mark.parametrize("scenarios, overrides, message", [
    ({**MONTE_CARLO, "factors": _swapped(_labels(), 0, 1)}, {}, "scenarios.factors must give each curve's pillars"),
    ({**MONTE_CARLO, "factors": ["discount:EUR/1y"]}, {}, "not curves of the market"),
    ({**MONTE_CARLO, "covariance": (-np.asarray(_covariance())).tolist()}, {}, "positive semi-definite"),
    ({**MONTE_CARLO, "covariance": [[1e-6]]}, {}, "covariance must be [12, 12]"),
    ({**MONTE_CARLO, "num_scenarios": 1}, {}, "num_scenarios"),
    # only the index curve shocked: the swap reads the discount curve too
    ({**MONTE_CARLO, "factors": _labels()[6:], "covariance": np.asarray(_covariance())[6:, 6:].tolist()}, {},
     "'discount:USD' is not a risk factor"),
    ({**HISTORICAL, "history": _history()[:5]}, {}, "fewer than two"),
    (MONTE_CARLO, {"quantiles": [1.5]}, "quantile"),
    (MONTE_CARLO, {"trades": []}, "at least one trade"),
    (MONTE_CARLO, {"batch_size": 0}, "batch_size"),
], ids=["factor-order", "unknown-curve", "not-psd", "covariance-shape", "one-scenario", "missing-curve",
        "short-history", "quantile", "no-trades", "batch"])
def test_a_request_the_run_would_refuse_is_a_400_and_no_job(test_client, scenarios, overrides, message):
    r = test_client.post("/portfolio/market-risk", json={**_body(scenarios), **overrides})
    assert r.status_code == 400, r.text
    assert message in r.json()["detail"]


def test_an_unknown_scenario_source_is_a_422(test_client):
    r = test_client.post("/portfolio/market-risk", json=_body({**MONTE_CARLO, "source": "stress"}))
    assert r.status_code == 422 and "monte-carlo" in r.text


def test_a_subset_of_the_curves_in_any_order_is_accepted():
    """Factors name whole curves of the market; the scenarios' columns follow their order."""
    labels = _labels()
    body = _body({**MONTE_CARLO, "factors": labels[6:] + labels[:6],
                  "covariance": np.roll(np.asarray(_covariance()), 6, axis=(0, 1)).tolist()})
    MarketRiskRequestSchema.model_validate(body).check()
