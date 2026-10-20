"""Comparison evidence must remain useful when runs fail or inputs are incompatible."""
import dataclasses
import json

import numpy as np
import pytest

from benchmarks.comparison import FORMATS, WORKLOADS, build_report, collect, compatible, load_points, main, markdown
from demos.benchmark_reporting import atomic_json, named_errors
from tests.support.precision_campaigns import CAMPAIGNS, COMPARISON_CAMPAIGNS
from tests.support.precision_timing import CaseTiming, json_settings, workload_settings


def snapshot(path, value=1., shifts=None):
    np.savez_compressed(path, **{"inputs/shifts": np.zeros((32, 2)) if shifts is None else shifts,
                                "trades/swap/scenario_PV": np.asarray([value, 0.]),
                                "portfolio/PnL": np.asarray([value, -value]),
                                "portfolio/VaR_99": np.asarray(value)})


def point(tmp_path, backend="cpu", label="float64", paths=32, shift=0.):
    root = tmp_path / f"{backend}-{label.replace(':', '-')}-{paths}"
    root.mkdir()
    seeds = [1, 2, 3]
    for seed in seeds:
        snapshot(root / f"seed-{seed}.npz", seed + shift)
    metadata = {"workload": {"market": "same"}, "input_generation": "cpu",
                "execution_devices": [backend + ":0 (test device)"],
                "environment": {"engine": "a", "tool_digest": "b", "jax": "pinned",
                                "versions": {"jaxlib": "pinned"}}}
    timing = CaseTiming(label, paths, None, 2., 1., 1, (0.5, 0.6), 0, None, 1, {},
                        {"portfolio/VaR_99": [[1. + shift], [2. + shift], [3. + shift]]},
                        seeds, {str(seed): f"seed-{seed}.npz" for seed in seeds}, metadata)
    return {"campaign": "comparison_single", "backend": backend, "label": label, "paths": paths,
            "seeds": seeds, "status": "ok", "root": str(root), "timing": dataclasses.asdict(timing)}


def test_named_errors_keep_absolute_error_when_reference_is_zero(tmp_path):
    snapshot(tmp_path / "a.npz", 0.)
    snapshot(tmp_path / "b.npz", 1e-7)
    row = named_errors(tmp_path / "a.npz", tmp_path / "b.npz", 1e-4, 1e-8)["portfolio/PnL"]
    assert row["status"] == "FAIL"
    assert row["relative_error"] is None
    assert row["max_absolute_error"] == pytest.approx(1e-7)


def test_figure_scale_does_not_punish_zero_entries(tmp_path):
    snapshot(tmp_path / "a.npz", 1.)
    snapshot(tmp_path / "b.npz", 1.00001)
    rows = named_errors(tmp_path / "a.npz", tmp_path / "b.npz", 1e-4, 1e-8)
    assert all(r["status"] == "PASS" for r in rows.values())
    assert rows["trades/swap/scenario_PV"]["elements"] == 2


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_outputs_are_explicit_failures(tmp_path, value):
    snapshot(tmp_path / "a.npz")
    snapshot(tmp_path / "b.npz", value)
    assert named_errors(tmp_path / "a.npz", tmp_path / "b.npz", 1e-4, 1e-8)["portfolio/PnL"] == {
        "status": "FAIL", "reason": "non-finite output"}


def test_matched_inputs_are_required(tmp_path):
    snapshot(tmp_path / "a.npz")
    snapshot(tmp_path / "b.npz", shifts=np.ones((32, 2)))
    with pytest.raises(ValueError, match="scenario inputs differ"):
        named_errors(tmp_path / "a.npz", tmp_path / "b.npz", 1e-4, 1e-8)


def test_error_overflow_is_a_reported_failure(tmp_path):
    snapshot(tmp_path / "a.npz", 1e308)
    snapshot(tmp_path / "b.npz", -1e308)
    rows = named_errors(tmp_path / "a.npz", tmp_path / "b.npz", 1e-4, 1e-8)
    assert rows["portfolio/PnL"] == {"status": "FAIL", "reason": "error overflow"}
    json.dumps(rows, allow_nan=False)


@pytest.mark.parametrize("arrays,reason", [({"different": [0.]}, "names differ"),
    ({"inputs/shifts": np.zeros((32, 2)), "trades/swap/scenario_PV": [1.],
      "portfolio/PnL": [1., -1.], "portfolio/VaR_99": np.asarray(1.)}, "shape differs")])
def test_output_contract_must_match(tmp_path, arrays, reason):
    snapshot(tmp_path / "a.npz")
    np.savez(tmp_path / "b.npz", **arrays)
    with pytest.raises(ValueError, match=reason):
        named_errors(tmp_path / "a.npz", tmp_path / "b.npz", 1e-4, 1e-8)


def test_report_joins_errors_timing_uncertainty_and_misses(tmp_path):
    points = [point(tmp_path), point(tmp_path, paths=64),
              point(tmp_path, "gpu", "pricing:float8_e4m3fn", shift=10.),
              point(tmp_path, "gpu", "pricing:float8_e4m3fn", paths=64, shift=10.)]
    data = build_report(points)
    assert data["comparisons"][2]["accuracy"] == "FAIL"
    assert all(r["paths"] is None for r in data["equal_accuracy"])
    assert data["reference_uncertainty"][0]["empirical_standard_error"] == pytest.approx([1 / np.sqrt(3)])
    assert "not reached" in markdown(data)
    json.dumps(data, allow_nan=False)


def test_missing_hardware_and_reference_remain_visible(tmp_path):
    gpu = point(tmp_path, "gpu")
    tpu = {k: v for k, v in gpu.items() if k not in {"timing", "root"}}
    tpu.update(backend="tpu", status="unmeasured", reason="no TPU runtime")
    data = build_report([gpu, tpu])
    assert data["comparisons"][0]["reason"] == "CPU FP64 reference unavailable"
    assert data["equal_accuracy"] == []
    assert "no TPU runtime" in markdown(data)


@pytest.mark.parametrize("field", ["engine", "tool_digest", "jax", "versions"])
def test_cross_checkout_or_version_join_is_refused(tmp_path, field):
    a, b = point(tmp_path), point(tmp_path, "gpu")
    b["timing"]["metadata"]["environment"][field] = "different"
    with pytest.raises(ValueError, match="differs|differ"):
        compatible(a, b)
    assert build_report([a, b])["comparisons"][1]["accuracy"] == "unmeasured"


@pytest.mark.parametrize("mutation,reason", [
    (lambda p: p["timing"].update(seeds=[1, 2, 4]), "seed identities"),
    (lambda p: p["timing"]["snapshots"].pop("3"), "snapshot seed"),
    (lambda p: p["timing"].update(warm_compiles=1), "compiled"),
    (lambda p: p["timing"]["metadata"].update(execution_devices=["cpu:0 (cpu)"]), "execution devices"),
    (lambda p: p["timing"]["metadata"].update(workload="other"), "workload differs")])
def test_incompatible_measurement_is_not_a_speed_claim(tmp_path, mutation, reason):
    a, b = point(tmp_path), point(tmp_path, "gpu")
    mutation(b)
    with pytest.raises(ValueError, match=reason):
        compatible(a, b)
    data = build_report([a, b])
    assert data["equal_accuracy"] == []


def test_missing_snapshot_does_not_abort_other_evidence(tmp_path):
    a, b = point(tmp_path), point(tmp_path, "gpu")
    (tmp_path / "gpu-float64-32" / "seed-2.npz").unlink()
    data = build_report([a, b])
    assert data["comparisons"][0]["accuracy"] == "PASS"
    assert data["comparisons"][1]["accuracy"] == "unmeasured"


def test_truncated_snapshot_does_not_abort_other_evidence(tmp_path):
    a, b = point(tmp_path), point(tmp_path, "gpu")
    archive = tmp_path / "gpu-float64-32" / "seed-2.npz"
    archive.write_bytes(archive.read_bytes()[:32])
    data = build_report([a, b])
    assert data["comparisons"][0]["accuracy"] == "PASS"
    assert data["comparisons"][1]["accuracy"] == "unmeasured"


def test_snapshot_path_cannot_escape_collection(tmp_path):
    a, b = point(tmp_path), point(tmp_path, "gpu")
    b["timing"]["snapshots"]["1"] = "../../elsewhere.npz"
    assert "escapes" in build_report([a, b])["comparisons"][1]["reason"]


def test_collection_retains_timeout_and_missing_rows(tmp_path, monkeypatch):
    monkeypatch.setattr("benchmarks.comparison.launch_worker", lambda *args: 124)
    collect(tmp_path, [32], [1, 2, 3], ["cpu"], ["float64"], ["comparison_single"], 1.)
    (row,) = load_points([tmp_path])
    assert row["status"] == "failed" and "124" in row["reason"]
    assert (tmp_path / "report.md").exists()
    (tmp_path / "comparison_single/cpu/float64/32/result.json").unlink()
    assert load_points([tmp_path])[0]["reason"] == "missing result"


def test_collection_refuses_stale_results(tmp_path):
    (tmp_path / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="empty"):
        collect(tmp_path, [32], [1, 2, 3], ["cpu"], ["float64"], ["comparison_single"], 1.)


def test_merge_refuses_duplicate_and_tampered_results(tmp_path):
    p = point(tmp_path)
    request = {k: p[k] for k in ("campaign", "backend", "label", "paths", "seeds")}
    request["directory"] = "cpu-float64-32"
    atomic_json(tmp_path / "manifest.json", {"schema_version": 1, "requests": [request]})
    atomic_json(tmp_path / request["directory"] / "result.json", p)
    with pytest.raises(ValueError, match="duplicate"):
        load_points([tmp_path, tmp_path])
    p["seeds"] = [4, 5, 6]
    atomic_json(tmp_path / request["directory"] / "result.json", p)
    with pytest.raises(ValueError, match="does not match"):
        load_points([tmp_path])


@pytest.mark.parametrize("arguments", [
    ["--seeds", "1", "1", "2"], ["--seeds", "1", "2"], ["--seeds", "-1", "2", "3"],
    ["--paths", "33"], ["--paths", "64"], ["--paths", "128", "128"], ["--rtol", "nan"], ["--atol", "-1"],
    ["--timeout", "0"], ["--formats", "float64", "float64"]])
def test_invalid_requests_fail_before_launch(tmp_path, arguments):
    with pytest.raises(SystemExit) as exc:
        main(["--out", str(tmp_path), *arguments])
    assert exc.value.code == 2
    assert not list(tmp_path.iterdir())


def test_workloads_reuse_campaign_definitions_and_capture_all_inputs():
    base = CAMPAIGNS["market_risk"]()
    portfolio = COMPARISON_CAMPAIGNS["comparison_portfolio"]()
    single = COMPARISON_CAMPAIGNS["comparison_single"]()
    assert set(WORKLOADS) == set(COMPARISON_CAMPAIGNS)
    assert set(COMPARISON_CAMPAIGNS).isdisjoint(CAMPAIGNS)
    assert set(FORMATS) == {"float64"} | {c.precision.pricing.compute for c in portfolio.cases}
    assert [t.trade_id for t in portfolio.trades] == [t.trade_id for t in base.trades]
    assert single.trades == portfolio.trades[:1]
    a, b = workload_settings(base), workload_settings(portfolio)
    assert a["market"] == b["market"] and a["pricing"] == b["pricing"] and a["covariance"] == b["covariance"]
    assert "date" in b["market"]["asof"]
    for case in portfolio.cases:
        policy = json_settings(case.precision)
        assert policy["simulation"]["compute"] == "float64"
        assert case.precision.pricing.compute in case.label
    json.dumps(b, allow_nan=False)


def test_metadata_does_not_scan_reference_submodule_contents(monkeypatch):
    """Reference working files are not runtime engine inputs; inspect gitlinks only."""
    import subprocess
    from tests.support.precision_harness import environment

    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", run)
    environment()
    status = next(c for c in commands if "status" in c)
    assert "--ignore-submodules=dirty" in status


def test_worker_selects_device_and_checks_observed_placement(tmp_path, monkeypatch):
    import jax
    import os
    from benchmarks.comparison import worker

    devices = jax.devices
    monkeypatch.setattr(jax, "devices", lambda backend=None: devices(backend or "cpu"))

    def child(*args, **kwargs):
        assert jax.config.jax_default_device == jax.devices("cpu")[0]
        return {"warm_compiles": 0, "metadata": {"execution_devices": ["gpu:0 (unexpected)"]}}
    monkeypatch.setattr("tests.support.precision_timing._child", child)
    arguments = {"backend": "cpu", "campaign": "comparison_single", "label": "float64", "paths": 128,
                 "seeds": [1, 2, 3]}
    with monkeypatch.context() as context:
        # The real worker is a fresh process. Keep its process flags isolated here too.
        context.setattr(os, "environ", dict(os.environ, JAX_DEFAULT_DEVICE="cpu"))
        worker(arguments, tmp_path)
    data = json.loads((tmp_path / "result.json").read_text())
    assert data["status"] == "invalid" and "unexpected devices" in data["reason"]


@pytest.mark.slow
@pytest.mark.parametrize("campaign", ["comparison_single", "comparison_portfolio"])
def test_real_comparison_worker_outputs_and_compilation_reuse(tmp_path, campaign):
    """Exercise the actual CLI, process isolation, snapshot comparison and host outputs."""
    assert main(["--out", str(tmp_path), "--paths", "128", "--backends", "cpu",
                 "--formats", "float64", "float32", "--workloads", campaign]) == 0
    data = json.loads((tmp_path / "comparison.json").read_text())
    assert len(data["points"]) == 2
    for p in data["points"]:
        assert p["status"] == "ok"
        t = p["timing"]
        assert t["compiles"] > 0 and t["warm_compiles"] == 0
        assert t["metadata"]["environment"]["compilation_cache"] == "disabled"
        assert t["metadata"]["environment"]["host"]["logical_cpus"] > 0
        assert len(t["snapshots"]) == 3
        with np.load(f"{p['root']}/seed-1.npz") as values:
            assert values["portfolio/scenario_PV"].shape == (128,)
            assert "trades/swap/scenario_PV" in values.files
    assert all(r["accuracy"] == "PASS" for r in data["comparisons"])
