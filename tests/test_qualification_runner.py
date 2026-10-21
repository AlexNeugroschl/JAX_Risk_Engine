"""Experimental qualifications preserve their domain and cannot replace records silently."""
import json

import pytest

from benchmarks.qualification import build_campaign, main, source_digest
from benchmarks.test_suite import junit_counts


def test_fixture_and_model_variants_capture_their_actual_inputs():
    base = build_campaign("compute", cases=["pricing:compute:float16"])
    changed = build_campaign("compute", fixture_seed=2, model="HullWhite", grid="default",
                             cases=["pricing:compute:float16"])
    assert "fixture-2-HullWhite-default" in changed.name
    assert "fixture 2" in changed.market_name
    assert changed.simulation.ir["USD"].volatility_type == "HullWhite"
    assert changed.pricing.bermudan.n_per_std == 30
    assert len(changed.cases) == 2  # float16 and its single-pass control
    assert changed.market is not base.market
    assert len(source_digest()) == 64


def test_multi_currency_campaigns_match_factor_dimensions_and_include_fx():
    from engine.risk.market import RateRiskFactors
    exposure = build_campaign("compute", multi_currency=True)
    risk = build_campaign("compute_market_risk", multi_currency=True)
    assert set(exposure.simulation.ir) == {"USD", "EUR"}
    assert exposure.simulation.fx_volatilities["EUR"] == .15
    assert {t.currency for t in exposure.trades} == {"USD", "EUR"}
    size = RateRiskFactors.from_market(risk.market).size
    assert risk.covariance.shape == (size, size)


def test_fixture_calibration_reads_the_index_actually_present_in_the_market():
    from engine.market_data.market import index_name
    campaign = build_campaign("compute", fixture_seed=1)
    component = campaign.simulation.ir["USD"]
    assert component.swap_index.index_tenor_months == 3
    assert index_name("USD", component.swap_index.index_tenor_months) in campaign.market.currency("USD").index_curves


@pytest.mark.parametrize("args", [["--seed-ids", "1", "1", "2"], ["--paths", "100"],
    ["--campaigns", "unknown"], ["--models", "LGM", "LGM"], ["--timeout", "nan"]])
def test_bad_experiments_fail_before_writing(tmp_path, args):
    with pytest.raises(SystemExit) as exc:
        main(["--out", str(tmp_path), *args])
    assert exc.value.code == 2
    assert not list(tmp_path.iterdir())


def test_junit_counts_include_failures_errors_and_skips(tmp_path):
    path = tmp_path / "junit.xml"
    path.write_text('<testsuites><testsuite tests="4" failures="1" errors="1" skipped="1"/></testsuites>')
    assert junit_counts(path) == {"tests": 4, "failures": 1, "errors": 1, "skipped": 1}
    path.write_text("<testsuites/>")
    with pytest.raises(ValueError, match="no test suites"):
        junit_counts(path)


def test_failed_smaller_count_cannot_disappear_from_a_candidate_ceiling(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from demos.benchmark_reporting import atomic_json
    from benchmarks.qualification import report
    from tests.support.precision_harness import Measurements
    requests = [{"directory": "experiment/64"}, {"directory": "experiment/128"}]
    atomic_json(tmp_path / "experiment/64/result.json", {"status": "failed", "request": requests[0]})
    atomic_json(tmp_path / "experiment/128/result.json", {"status": "ok", "request": requests[1], "measurement": "m.gz"})
    monkeypatch.setattr(Measurements, "load", lambda p: SimpleNamespace(campaign="experiment"))
    (tmp_path / "candidate-evidence.json").write_text("stale evidence")
    results = report(tmp_path, requests)
    assert [r["status"] for r in results] == ["failed", "ok"]
    assert not (tmp_path / "candidate-evidence.json").exists()


@pytest.mark.slow
def test_real_isolated_accuracy_collection_is_reviewable_and_resumable(tmp_path):
    args = ["--out", str(tmp_path), "--campaigns", "compute", "--paths", "64",
            "--seed-ids", "101", "102", "103", "--cases", "pricing:compute:float16", "--fixture-seeds", "1"]
    assert main(args) == 0
    data = json.loads((tmp_path / "qualification.json").read_text())
    assert len(data["points"]) == 1 and data["points"][0]["status"] == "ok"
    assert data["points"][0]["request"]["seeds"] == [101, 102, 103]
    assert (tmp_path / "candidate-evidence.json").is_file()
    assert main([*args, "--resume"]) == 0
    with pytest.raises(SystemExit):
        main([*args, "--resume", "--paths", "128"])
