"""API timings must include result retrieval and never duplicate ORE scenario valuation."""
import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from benchmarks.api_comparison import compare, request_body, timed_request, write_report
from benchmarks.api_worker import ore_result
from engine.api.requests import RunRequestSchema
from tests.support.precision_campaigns import comparison


@pytest.mark.parametrize("single", [True, False])
def test_public_request_keeps_the_existing_workload_and_shocks(single):
    from engine.risk.market import historical_scenarios

    campaign = comparison(single)
    body = request_body(campaign, 128, 7, "float64", "artifact")
    request = RunRequestSchema.model_validate(body).check()
    assert request.portfolio == campaign.trades
    assert request.config.pricing == campaign.pricing
    source = body["config"]["market_risk"]["scenarios"]
    levels = np.asarray(source["history"])
    h = campaign.horizon_days
    np.testing.assert_allclose(levels[h:] - levels[:-h], np.asarray(campaign.scenarios(128, 7).shifts),
                               rtol=0, atol=1e-16)
    assert source["dates"] is None  # ORE assigns dates only within its bounded native batches.
    assert source["source"] == "historical"  # Synthetic input representation, stated in the report.
    assert historical_scenarios(request.config.market_risk.scenarios.factors, levels, h).num_scenarios == 128


def test_ore_executes_once_and_reuses_trade_pnl(monkeypatch):
    campaign = comparison(True)
    body = request_body(campaign, 128, 1, "float64", "inline")
    source = body["config"]["market_risk"]["scenarios"]
    import ORE
    from tests.support.ore_market_risk_oracle import business_days_before
    dates = business_days_before(campaign.market.asof, 128 + source["horizon_days"])
    h = source["horizon_days"]
    pnl = np.arange(-128., 0.)
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return {"historical_PnL": {"PLDate1": dates[:-h], "PLDate2": dates[h:],
                                  "Portfolio": ["swap"] * 128, "PLAmount": pnl,
                                  "RiskClass": ["All"] * 128, "RiskType": ["All"] * 128},
                "npv": {"TradeId": ["swap"], "NPV": [1000.]}}

    monkeypatch.setattr("tests.support.ore_market_risk_oracle._run", run)
    result = ore_result(RunRequestSchema.model_validate(body)).market_risk
    assert len(calls) == 1
    assert calls[0][0][-1] is True and calls[0][1] == {"include_npv": True}
    np.testing.assert_array_equal(result.pnl[:, 0], pnl)
    np.testing.assert_array_equal(result.portfolio_pnl, pnl)
    assert result.base_npv == 1000.
    stats = ORE.RiskStatistics()
    stats.add(ORE.DoubleVector(pnl.tolist()))
    assert result.risk["VaR_99"] == stats.valueAtRisk(.99)
    assert result.risk["ES_99"] == stats.expectedShortfall(.99)


def test_ore_does_not_silently_ignore_precision():
    body = request_body(comparison(True), 128, 1, "pricing:float32", "inline")
    with pytest.raises(ValueError, match="float64 only"):
        ore_result(RunRequestSchema.model_validate(body))


def test_native_batches_do_not_repeat_or_omit_scenarios(monkeypatch):
    from tests.support.ore_market_risk_oracle import _run

    schema = RunRequestSchema.model_validate(request_body(comparison(True), 128, 1, "float64", "inline"))
    baseline = ore_result(schema).market_risk
    calls = []

    def counted(*args, **kwargs):
        calls.append((np.array(args[2]), kwargs["include_npv"]))
        return _run(*args, **kwargs)

    monkeypatch.setattr("benchmarks.api_worker.ORE_BATCH_SCENARIOS", 64)
    monkeypatch.setattr("tests.support.ore_market_risk_oracle._run", counted)
    batched = ore_result(schema).market_risk
    assert len(calls) == 2 and [base for _, base in calls] == [True, False]
    h = schema.config.market_risk.scenarios.horizon_days
    observed = np.concatenate([levels[h:] - levels[:-h] for levels, _ in calls])
    levels = np.asarray(schema.config.market_risk.scenarios.history)
    np.testing.assert_array_equal(observed, levels[h:] - levels[:-h])
    np.testing.assert_allclose(batched.pnl, baseline.pnl, rtol=1e-12, atol=1e-8)
    assert batched.risk == baseline.risk


def test_clock_includes_submit_polls_and_artifact_fetch(monkeypatch):
    from engine.api.artifacts import array_artifact, items_record

    reference, chunks = array_artifact(np.ones((128, 1)), "pnl", "/runs/job/artifacts/pnl",
                                      ("scenario", "trade"), items_record("trade", ["swap"]))
    reference["form"] = "artifact"
    clock = [0.]
    calls, polls = [], []

    def handle(request):
        calls.append(request.url.path)
        clock[0] += 1.
        if request.method == "POST":
            return httpx.Response(202, json={"run_id": "job"})
        if request.url.path == "/runs/job":
            polls.append(1)
            if len(polls) == 1:
                return httpx.Response(200, json={"status": "running"})
            return httpx.Response(200, json={"status": "done", "result": {"market_risk": {"pnl": reference}}})
        return httpx.Response(200, content=chunks[int(request.url.path.rsplit("/", 1)[1])])

    monkeypatch.setattr("benchmarks.api_comparison.time.perf_counter", lambda: clock[0])
    with httpx.Client(base_url="http://test", transport=httpx.MockTransport(handle)) as client:
        elapsed, run_id, _, pnl = timed_request(client, {}, 30)
    assert elapsed == len(calls) == 3 + len(chunks)
    assert run_id == "job" and pnl.shape == (128, 1)
    assert calls[-1].startswith("/runs/job/artifacts/")


def test_failed_job_is_not_a_successful_timing():
    def handle(request):
        return httpx.Response(202, json={"run_id": "job"}) if request.method == "POST" else \
            httpx.Response(200, json={"status": "failed", "error": "calibration failed"})

    with httpx.Client(base_url="http://test", transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(RuntimeError, match="calibration failed"):
            timed_request(client, {}, 30)


def saved_point(root, implementation="ore", shift=0.):
    from demos.benchmark_reporting import atomic_json

    root.mkdir()
    body = request_body(comparison(True), 128, 1, "float64", "inline")
    atomic_json(root / "requests.json", {"1": body})
    np.savez(root / "sample.npz", **{"inputs/shifts": np.zeros((128, 20)),
                                     "portfolio/PnL": np.arange(128.) + shift})
    samples = [{"seed": 1, "phase": phase, "seconds": 2. if implementation == "ore" else 1.,
                "run_id": f"{implementation}-{phase}", "snapshot": "sample.npz", "compiles": 0}
               for phase in ("first", "repeat")]
    return {"campaign": "comparison_single", "implementation": implementation, "backend": "cpu", "paths": 128,
            "label": "ore:float64" if implementation == "ore" else "float64", "status": "ok", "root": str(root),
            "seeds": [1], "repeats": 1, "output": "inline", "samples": samples,
            "environment": {"environment": {"engine": "same", "tool_digest": "same", "versions": {"ore": "same"}}}}


def test_speedup_requires_matched_outputs_and_complete_fresh_jobs(tmp_path):
    ore = saved_point(tmp_path / "ore")
    jax = saved_point(tmp_path / "jax", "jax")
    report = write_report(tmp_path, [ore, jax], 1e-4, 1e-8)
    assert report["comparisons"][1]["ore_speedup"] == 2.
    jax["samples"][1]["run_id"] = jax["samples"][0]["run_id"]
    report = write_report(tmp_path, [ore, jax], 1e-4, 1e-8)
    assert report["comparisons"][1]["status"] == "failed"
    assert report["comparisons"][1]["ore_speedup"] is None


def test_bad_accuracy_retains_time_but_withholds_speedup(tmp_path):
    ore = saved_point(tmp_path / "ore")
    jax = saved_point(tmp_path / "jax", "jax", shift=1.)
    row = write_report(tmp_path, [ore, jax], 1e-4, 1e-8)["comparisons"][1]
    assert row["accuracy"] == "FAIL" and row["ore_speedup"] is None
    assert row["repeat_median_seconds"] == 1.


def test_financial_settings_and_output_modes_must_match(tmp_path):
    from demos.benchmark_reporting import atomic_json

    ore = saved_point(tmp_path / "ore")
    jax = saved_point(tmp_path / "jax", "jax")
    bodies = json.loads((tmp_path / "jax/requests.json").read_text())
    bodies["1"]["portfolio"][0]["notional"] *= 2
    atomic_json(tmp_path / "jax/requests.json", bodies)
    with pytest.raises(ValueError, match="workloads differ"):
        compare(ore, jax, 1e-4, 1e-8)
    jax["output"] = "artifact"
    with pytest.raises(ValueError, match="output requirements differ"):
        compare(ore, jax, 1e-4, 1e-8)


@pytest.mark.slow
@pytest.mark.parametrize("workload", ["comparison_single", "comparison_portfolio"])
def test_real_http_ore_and_jax_comparison(tmp_path, workload):
    from benchmarks.comparison import main
    from engine.api.job_queue import WorkerLock

    assert main(["--out", str(tmp_path), "--paths", "128", "--backends", "cpu", "--formats", "float64", "float32",
                 "--workloads", workload, "--repeats", "1", "--timeout", "180"]) == 0
    report = json.loads((tmp_path / "comparison.json").read_text())
    assert report["transport"] == "api" and len(report["points"]) == 3
    assert all(row["accuracy"] == "PASS" for row in report["comparisons"])
    for point in report["points"]:
        assert len(point["samples"]) == 4
        assert len({s["run_id"] for s in point["samples"]}) == 4
        assert all(s["seconds"] > 0 for s in point["samples"])
        if point["implementation"] == "jax":
            assert sum(s["compiles"] for s in point["samples"] if s["phase"] == "repeat") == 0
        # Services must release their isolated worker lock even on Windows.
        with_lock = WorkerLock(Path(point["root"]) / "worker.lock")
        assert with_lock.acquire(wait_seconds=1)
        with_lock.release()
