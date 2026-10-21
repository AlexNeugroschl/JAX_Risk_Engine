"""An execution label must not substitute for numerical and native lowering evidence."""
import jax
import json
import numpy as np
import os
import pytest
import sys

from tests.support.precision_devices import fp8_lowering, normalized_error, qualify


TRITON_FP8 = '''%gemm (lhs: f8e4m3fn[3,4], rhs: f8e5m2[4,2]) -> f32[3,2] {
  %lhs = f8e4m3fn[3,4]{1,0} parameter(0)
  %rhs = f8e5m2[4,2]{1,0} parameter(1)
  ROOT %dot = f32[3,2]{1,0} dot(%lhs, %rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
  ROOT %fusion = f32[3,2]{1,0} fusion(%a, %b), kind=kCustom, calls=%gemm, backend_config={"kind":"__triton_nested_gemm_fusion"}
'''


def test_native_lowering_accepts_triton_as_well_as_cublas():
    assert fp8_lowering(TRITON_FP8) == "triton"
    assert fp8_lowering('custom-call(), custom_call_target="__cublas$lt$matmul$f8"') == "cublas"


@pytest.mark.parametrize("hlo", [TRITON_FP8.replace("f8e5m2", "f16"),
    TRITON_FP8.replace("f32[3,2]{1,0} dot", "f16[3,2]{1,0} dot"),
    TRITON_FP8.replace("calls=%gemm,", "calls=%other,"), TRITON_FP8.replace("__triton", "__other")])
def test_unrelated_types_algorithms_and_fusions_are_not_native_fp8_evidence(hlo):
    assert fp8_lowering(hlo) is None


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
def test_device_cli_runs_single_pass_and_product_format_engine_controls(tmp_path, monkeypatch):
    from demos.benchmark_reporting import launch_worker
    out = tmp_path / "qualification"
    flags = [f for f in os.environ.get("XLA_FLAGS", "").split()
             if not f.startswith("--xla_gpu_enable_triton_gemm=")]
    monkeypatch.setenv("XLA_FLAGS", " ".join([*flags, "--xla_gpu_enable_triton_gemm=true"]))
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
    assert "--xla_gpu_enable_triton_gemm=true" in result["environment"]["flags"]["XLA_FLAGS"].split()
    assert {r["timing"]["case"] for r in result["engine_workloads"]} == {
        "pricing:compute:float8_e4m3fn_1x", "pricing:compute:bfloat16_3x"}
    assert all(r["timing"]["warm_compiles"] == 0 for r in result["engine_workloads"])
