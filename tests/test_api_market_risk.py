"""
The market-risk analytic over HTTP (`market_risk` of `POST /runs`, I-56, I-87): on the shared
portfolio's market the polled result equals `engine.run.run` on the same request, which is a
direct `run_market_risk` call (tests/test_run_request.py), for Monte Carlo and historical
scenarios, each trade's row keyed by its id, in the reporting currency it names; the P&L by
reference is the inline P&L; a request the run would refuse is a 400 before any run starts,
naming the field; the Monte Carlo scenarios are drawn by the worker, not the route.
"""
import numpy as np
import pytest

from engine.api.artifacts import read_array
from engine.risk.market import MonteCarloScenarios, RateRiskFactors
from tests.support import portfolio as shared
from tests.support.runs import request_of, run_body, run_in_process, run_over_http

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


def _body(scenarios=MONTE_CARLO, trades=TRADES, outputs=None, **section):
    named = [{**shared.trades_json()[n], "trade_id": n} for n in trades]
    return run_body(["market_risk"], named, outputs=outputs,
                    market_risk={"scenarios": scenarios, "quantiles": [0.99, 0.975], **section})


@pytest.mark.slow
@pytest.mark.parametrize("scenarios", [MONTE_CARLO, HISTORICAL], ids=["monte-carlo", "historical"])
def test_the_result_equals_the_in_process_run(test_client, scenarios):
    body = _body(scenarios, outputs={"pnl": "inline"})
    result = run_over_http(test_client, body)
    direct = run_in_process(body).market_risk
    risk = result["market_risk"]
    assert [t["trade_id"] for t in risk["trades"]] == list(TRADES)
    assert [t["base_npv"] for t in risk["trades"]] == direct.base_npv_per_trade
    assert risk["pnl"]["axes"] == ["scenario", "trade"] and risk["pnl"]["items"]["ids"] == list(TRADES)
    np.testing.assert_array_equal(np.asarray(risk["pnl"]["values"]), np.asarray(direct.pnl))
    assert risk["risk"] == {k: (None if np.isnan(v) else v) for k, v in direct.risk.items()}
    assert risk["risk"]["VaR_99"] > 0.0 and risk["source"] == scenarios["source"]
    assert risk["risk_factors"] == _labels() and risk["num_scenarios"] == direct.num_scenarios
    assert result["warnings"] == direct.warnings and risk["precision"]["paths"] == direct.num_scenarios
    assert result["base_currency"] == direct.base_currency == "USD"
    assert risk["pricing_settings"] == {"bond": "engine=Discounting", "european_swaption": "engine=Bachelier",
                                        "swap": "engine=Discounting"}
    assert [t["npv"] for t in result["trades"]] == [None] * len(TRADES)


@pytest.mark.slow
def test_the_pnl_by_reference_is_the_inline_pnl(test_client):
    inline = run_over_http(test_client, _body(trades=("swap-payer", "bond"), outputs={"pnl": "inline"}))
    by_reference = run_over_http(test_client, _body(trades=("swap-payer", "bond")))
    reference = by_reference["market_risk"]["pnl"]
    assert reference["form"] == "artifact" and reference["axes"] == ["scenario", "trade"]
    assert reference["items"]["ids"] == ["swap-payer", "bond"]
    pnl = read_array(reference, lambda url: test_client.get(url).content)
    np.testing.assert_array_equal(pnl, np.asarray(inline["market_risk"]["pnl"]["values"]))
    assert by_reference["market_risk"]["risk"] == inline["market_risk"]["risk"]


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
    (MONTE_CARLO, {"batch_size": 0}, "batch_size"),
    (MONTE_CARLO, {"quantiles": [0.99, 0.99]}, "0.99 is given more than once"),
], ids=["factor-order", "unknown-curve", "not-psd", "covariance-shape", "one-scenario", "missing-curve",
        "short-history", "quantile", "batch", "repeated-quantile"])
def test_a_request_the_run_would_refuse_is_a_400_and_no_run(test_client, scenarios, overrides, message):
    r = test_client.post("/runs", json=_body(scenarios, **overrides))
    assert r.status_code == 400, r.text
    assert message in r.json()["detail"]


@pytest.mark.parametrize("change, message", [
    (lambda b: b.update(portfolio=[]), "at least one trade"),
    (lambda b: b["config"].update(base_currency="EUR"), "base_currency='EUR': no market for currency 'EUR'"),
    (lambda b: b["config"].pop("market_risk"), "config.market_risk: the market_risk analytic needs it"),
    (lambda b: b["config"].update(precision=dict(market={"storage": "float32", "compute": "float32",
                                                         "accumulate": "float32"})),
     "config.precision.market is not read by market risk"),
    (lambda b: b["config"].update(precision=dict(simulation={"storage": "float32", "compute": "float32"})),
     "config.precision.simulation.compute='float32' is not read by market risk"),
], ids=["no-trades", "reporting-currency", "no-section", "market-stage", "simulation-compute"])
def test_a_request_the_analytic_does_not_read_or_cannot_run_is_a_400(test_client, change, message):
    """I-95: market risk refuses the precision settings it does not read."""
    body = _body()
    change(body)
    r = test_client.post("/runs", json=body)
    assert r.status_code == 400, r.text
    assert message in r.json()["detail"]


def test_an_unknown_scenario_source_is_a_422(test_client):
    r = test_client.post("/runs", json=_body({**MONTE_CARLO, "source": "stress"}))
    assert r.status_code == 422 and "monte-carlo" in r.text


def test_a_subset_of_the_curves_in_any_order_is_accepted_and_nothing_is_drawn():
    """Factors name whole curves of the market; the scenarios' columns follow their order. The
    request the route checks holds the Monte Carlo scenarios undrawn (no JAX work)."""
    from engine.api.requests import RunRequestSchema

    labels = _labels()
    body = _body({**MONTE_CARLO, "factors": labels[6:] + labels[:6],
                  "covariance": np.roll(np.asarray(_covariance()), 6, axis=(0, 1)).tolist()})
    request = RunRequestSchema.model_validate(body).check()
    assert isinstance(request.config.market_risk.scenarios, MonteCarloScenarios)
    assert request_of(body).config.market_risk.scenarios.factors.names[0] == "index:USD-SIMINDEX-6M"
