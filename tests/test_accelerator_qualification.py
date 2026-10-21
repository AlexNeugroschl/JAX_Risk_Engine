"""An execution label must not substitute for numerical and native lowering evidence."""
import jax
import json
import numpy as np
import pytest
import sys

from tests.support.precision_devices import normalized_error, qualify


def test_worker_gemm_selection_preserves_determinism_and_other_flags():
    from engine.api.worker import gemm_environment
    env = {"XLA_FLAGS": "--xla_gpu_enable_triton_gemm=true --xla_gpu_exclude_nondeterministic_ops=true"}
    assert gemm_environment(None, env) == {}
    cublas = gemm_environment("cublas", env)["XLA_FLAGS"]
    assert cublas.count("--xla_gpu_enable_triton_gemm=") == 1
    assert "--xla_gpu_enable_triton_gemm=false" in cublas
    assert "--xla_gpu_exclude_nondeterministic_ops=true" in cublas
    assert "triton_gemm" not in gemm_environment("default", {"XLA_FLAGS": cublas})["XLA_FLAGS"]


def test_zero_and_nonfinite_products_cannot_silently_pass():
    zero = np.zeros((2, 3))
    assert normalized_error(zero, zero, zero) == 0.
    assert np.isinf(normalized_error(np.ones((2, 3)), zero, zero))
    assert np.isinf(normalized_error(np.full((2, 3), np.nan), zero, np.ones((2, 3))))


def test_fp8_cpu_emulation_has_numerical_evidence_without_native_claims(tmp_path):
    result = qualify(jax.devices("cpu")[0], ("float8_e4m3fn",), tmp_path,
                     shapes=((3, 4, 2),), patterns=("wide", "zero", "cancellation"))
    assert result["passes"]
    assert {r["rounding"] for r in result["operations"]} == {"nearest", "stochastic"}
    assert all(r["dispatch"] == "emulated" and not r["observed_fp8_gemm"] for r in result["operations"])
    assert all((tmp_path / r["hlo"]).is_file() for r in result["operations"])
    assert all(r["device_emulation_error"] == 0. for r in result["operations"])


@pytest.mark.slow
def test_device_cli_runs_single_pass_and_product_format_engine_controls(tmp_path):
    from demos.benchmark_reporting import launch_worker
    out = tmp_path / "qualification"
    script = """
from tests.support import precision_devices as p
original = p.qualify
def small(device, formats, out):
    return original(device, formats, out, shapes=((3, 4, 2),), patterns=("wide",), roundings=("nearest",))
p.qualify = small
raise SystemExit(p.main(["--platform", "cpu", "--formats", "float8_e4m3fn_1x", "bfloat16_3x",
    "--engine-workloads", "comparison_single", "--out", %r]))
""" % str(out)
    assert launch_worker([sys.executable, "-c", script], tmp_path / "process", 120.) == 0
    result = json.loads((out / "qualification.json").read_text())
    assert result["passes"] and result["complete"]
    assert {r["timing"]["case"] for r in result["engine_workloads"]} == {
        "pricing:compute:float8_e4m3fn_1x", "pricing:compute:bfloat16_3x"}
    assert all(r["timing"]["warm_compiles"] == 0 for r in result["engine_workloads"])
