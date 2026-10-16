"""The scaling study must compare full outputs and expose incomplete or unreliable runs."""
import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from demos.benchmark_reporting import atomic_json, compare_snapshots, numeric_arrays, read_json, trace_summary, union_seconds, write_report


def test_full_outputs_are_checked_instead_of_only_a_mean(tmp_path):
    cpu, gpu = tmp_path / "cpu.npz", tmp_path / "gpu.npz"
    np.savez(cpu, cube=np.array([1.0, 3.0]))
    np.savez(gpu, cube=np.array([2.0, 2.0]))  # same mean, wrong scenario values
    result = compare_snapshots(cpu, gpu, rtol=1e-8, atol=1e-6)
    assert result["status"] == "FAIL"
    assert result["worst_output"] == "cube"


@pytest.mark.parametrize("other", [np.array([True, False]), np.ones((1, 2)), np.array([1.0, np.nan])])
def test_discrete_shape_and_nonfinite_mismatches_are_rejected(tmp_path, other):
    cpu, gpu = tmp_path / "cpu.npz", tmp_path / "gpu.npz"
    np.savez(cpu, output=np.ones(2))
    np.savez(gpu, output=other)
    assert compare_snapshots(cpu, gpu, rtol=1e-8, atol=1e-6)["status"] == "FAIL"


def test_rounding_within_tolerance_passes_and_all_values_are_counted(tmp_path):
    cpu, gpu = tmp_path / "cpu.npz", tmp_path / "gpu.npz"
    np.savez(cpu, cube=np.array([1.0, 3.0]), exercised=np.array([True, False]))
    np.savez(gpu, cube=np.array([1.0 + 1e-10, 3.0]), exercised=np.array([True, False]))
    result = compare_snapshots(cpu, gpu, rtol=1e-8, atol=1e-6)
    assert result["status"] == "PASS" and result["elements"] == 4


def test_identical_values_with_different_precision_are_not_a_fair_comparison(tmp_path):
    cpu, gpu = tmp_path / "cpu.npz", tmp_path / "gpu.npz"
    np.savez(cpu, output=np.ones(2, dtype=np.float64))
    np.savez(gpu, output=np.ones(2, dtype=np.float32))
    result = compare_snapshots(cpu, gpu, rtol=1e-8, atol=1e-6)
    assert result["status"] == "FAIL" and "dtype differs" in result["reason"]


def test_observables_retain_dataclass_dictionary_and_sequence_names():
    @dataclasses.dataclass
    class Output:
        cube: np.ndarray
        labels: str
        risk: dict

    values = numeric_arrays(Output(np.zeros((2, 3)), "ignored", {"x": [1.0, np.ones(2)]}))
    assert set(values) == {"output/cube", "output/risk/x/0", "output/risk/x/1"}
    assert values["output/cube"].shape == (2, 3)


def test_gpu_activity_does_not_double_count_overlapping_streams_or_work_outside_window():
    assert union_seconds([(0, 4_000_000_000), (2_000_000_000, 8_000_000_000),
                          (10_000_000_000, 12_000_000_000)], 1_000_000_000, 11_000_000_000) == 8


def test_missing_worker_produces_failed_report_not_an_empty_success(tmp_path):
    atomic_json(tmp_path / "settings.json", {"devices": ["cpu", "gpu"], "instruments": ["swap"],
                                            "stages": ["pricing"], "rtol": 1e-8, "atol": 1e-6})
    assert write_report(tmp_path, plots=False)
    result = read_json(tmp_path / "results.json")
    assert result["incomplete_workers"] == ["cpu/swap", "gpu/swap"]
    assert "INCOMPLETE / FAILED CHECKS" in (tmp_path / "report.md").read_text()


def test_nonzero_worker_exit_is_not_hidden_by_a_completed_checkpoint(tmp_path):
    atomic_json(tmp_path / "settings.json", {"devices": ["cpu"], "instruments": ["swap"],
                                            "stages": ["pricing"], "rtol": 1e-8, "atol": 1e-6})
    atomic_json(tmp_path / "cpu" / "swap" / "results.json",
                {"complete": True, "metadata": {"backend": "cpu", "group": "swap"}, "rows": []})
    atomic_json(tmp_path / "worker_failures.json", [{"backend": "cpu", "group": "swap", "exit_code": 1}])
    assert write_report(tmp_path, plots=False)
    assert "Worker exited with code 1" in (tmp_path / "report.md").read_text()


@pytest.mark.parametrize("with_end", [True, False])
def test_trace_requires_end_marker_and_counts_device_overlap_once(tmp_path, with_end):
    from jax.profiler import ProfileData

    host = '''planes { name: "/host:CPU"
      event_metadata { key: 1 value { id: 1 name: "benchmark" } }
      event_metadata { key: 2 value { id: 2 name: "$engine.py:10 price" } }
      event_metadata { key: 3 value { id: 3 name: "capture_end" } }
      lines { id: 1 name: "main" timestamp_ns: 0
        events { metadata_id: 1 offset_ps: 0 duration_ps: 1000000000000 }
        events { metadata_id: 2 offset_ps: 0 duration_ps: 1000000000000 }
    '''
    if with_end:
        host += 'events { metadata_id: 3 offset_ps: 1000000000000 duration_ps: 1000 }'
    gpu = '''planes { name: "/device:GPU:0"
      event_metadata { key: 1 value { id: 1 name: "kernel" } }
      lines { id: 1 name: "Stream #1(Compute)" timestamp_ns: 0
        events { metadata_id: 1 offset_ps: 100000000000 duration_ps: 400000000000 } }
      lines { id: 2 name: "Stream #2(Compute)" timestamp_ns: 0
        events { metadata_id: 1 offset_ps: 300000000000 duration_ps: 400000000000 } }
    }'''
    (tmp_path / "test.xplane.pb").write_bytes(ProfileData.text_proto_to_serialized_xspace(host + '}}' + gpu))
    if with_end:
        summary = trace_summary(tmp_path, wall_seconds=1.0, baseline=0.5, compiles=0)
        assert summary["end_marker"] and summary["gpu_events"] == 2
        assert summary["gpu_kernel_seconds"] == pytest.approx(0.6)
        assert summary["profiler_slowdown"] == 2
    else:
        with pytest.raises(RuntimeError, match="boundary markers"):
            trace_summary(tmp_path, wall_seconds=1.0, baseline=0.5, compiles=0)
        assert not read_json(tmp_path / "summary.json")["end_marker"]


@pytest.mark.slow
def test_cpu_cli_writes_real_measurements_and_resume_keeps_completed_worker(tmp_path):
    script = Path(__file__).resolve().parents[1] / "demos" / "demo_gpu_scaling.py"
    command = [sys.executable, str(script), "--devices", "cpu", "--instruments", "swap", "--stages", "pricing",
               "--preset", "quick", "--paths", "4", "--trades", "1", "--layouts", "shared",
               "--warmups", "1", "--repeats", "2", "--traces", "none", "--no-plots", "--output", str(tmp_path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    run = next(p.parent for p in tmp_path.glob("*/settings.json"))
    data = read_json(run / "cpu" / "swap" / "results.json")
    assert data["complete"] and len(data["rows"]) == 2
    for row in data["rows"]:
        assert row["status"] == "ok" and len(row["seconds"]) == 2
        assert row["median_seconds"] > 0 and (run / row["snapshot"]).exists()
    log = run / "cpu" / "swap" / "worker.log"
    modified = log.stat().st_mtime_ns
    resumed = subprocess.run([sys.executable, str(script), "--resume", str(run)],
                             capture_output=True, text=True, timeout=30)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert log.stat().st_mtime_ns == modified
    assert "NOT COMPARED" in (run / "report.md").read_text()
