"""Native dispatch is the default; compiler workarounds must be explicit process settings."""
import contextlib
import json
import os
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from engine.precision import Arithmetic, matmul, product_execution, product_precision

EMULATION = "JAX_RISK_EMULATE_PRODUCTS"
H100 = types.SimpleNamespace(platform="gpu", device_kind="NVIDIA H100", compute_capability="9.0")


@pytest.fixture(autouse=True)
def normal_execution(monkeypatch):
    monkeypatch.delenv(EMULATION, raising=False)


@pytest.mark.parametrize("flags", ["", "--xla_gpu_enable_triton_gemm=true", "--xla_gpu_enable_triton_gemm=false"])
@pytest.mark.parametrize("name", ["float8_e4m3fn", "float8_e5m2_1x"])
def test_capable_gpu_uses_native_fp8_under_either_emitter(monkeypatch, flags, name):
    monkeypatch.setenv("XLA_FLAGS", flags)
    assert product_execution(name, H100) == "native"


def test_native_bfloat16_algorithm_is_requested_for_vectors_too():
    assert product_precision("bfloat16_3x", H100, matrix_vector=True) == jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X3


@pytest.mark.parametrize("name", ["float8_e4m3fn", "float8_e5m2", "float8_e4m3fn_1x", "float8_e5m2_1x"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_default_gpu_program_contains_native_fp8_operands(monkeypatch, name, side):
    from engine.precision import products, FORMATS

    monkeypatch.setattr(products, "default_device", lambda: H100)
    monkeypatch.setenv("XLA_FLAGS", "")
    program = jax.make_jaxpr(lambda a, b: matmul(a, b, Arithmetic(name, "float32", execution="allow_emulation"), side))(
        np.ones((3, 4)), np.ones((4, 2)))
    dot, = (e for e in program.eqns if e.primitive.name == "dot_general")
    base = FORMATS[name].base
    held = FORMATS[name].shared_dtype or base
    expected = (held, base) if side == "left" else (base, held)
    assert tuple(v.aval.dtype for v in dot.invars) == tuple(jnp.dtype(n) for n in expected)
    assert dot.params["preferred_element_type"] == jnp.float32


def test_campaign_metadata_records_explicit_execution_settings(monkeypatch):
    from tests.support.precision_harness import environment

    monkeypatch.setenv(EMULATION, "bfloat16_3x")
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")
    metadata = environment()
    assert metadata["product_emulation"] == "bfloat16_3x"
    assert metadata["xla_flags"] == "--xla_gpu_enable_triton_gemm=false"


@pytest.mark.parametrize("name", ["float16", "float16_1x", "float8_e4m3fn", "float8_e4m3fn_1x"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_explicit_emulation_uses_float32_carriers_and_preserves_rounded_values(monkeypatch, name, side):
    from engine.precision import products

    a = np.arange(12, dtype=float).reshape(3, 4) / 7
    b = np.arange(8, dtype=float).reshape(4, 2) / 9
    expected = np.asarray(matmul(a, b, Arithmetic(name, "float32", execution="allow_emulation"), side))
    monkeypatch.setenv(EMULATION, "float16, float8_e4m3fn")
    monkeypatch.setattr(products, "default_device", lambda: H100)
    assert product_execution(name) == "emulated"
    assert product_execution("bfloat16", H100) == "native"
    program = jax.make_jaxpr(lambda x, y: matmul(x, y, Arithmetic(name, "float32", execution="allow_emulation"), side))(a, b)
    dot, = (e for e in program.eqns if e.primitive.name == "dot_general")
    assert all(v.aval.dtype == jnp.float32 for v in dot.invars)
    np.testing.assert_allclose(matmul(a, b, Arithmetic(name, "float32", execution="allow_emulation"), side), expected, rtol=1e-6, atol=1e-7)


def test_explicit_bfloat16_workaround_is_reported_as_emulated(monkeypatch):
    monkeypatch.setenv(EMULATION, "bfloat16_3x")
    assert product_execution("bfloat16_3x", H100) == "emulated"
    assert product_precision("bfloat16_3x", H100, matrix_vector=True) == jax.lax.Precision.HIGHEST
    assert product_execution("float8_e5m2", H100) == "native"


@pytest.mark.parametrize("value", ["fp88", "float64"])
def test_invalid_emulation_settings_fail_instead_of_silently_changing_dispatch(monkeypatch, value):
    monkeypatch.setenv(EMULATION, value)
    with pytest.raises(ValueError, match=EMULATION):
        product_execution("float8_e4m3fn", H100)


@pytest.mark.parametrize("label", ["float64", "pricing:float8_e4m3fn"])
@pytest.mark.parametrize("inherited", ["--xla_gpu_enable_triton_gemm=true", "--xla_gpu_enable_triton_gemm=false"])
def test_comparison_auto_keeps_the_same_inherited_emitter_for_all_formats(tmp_path, monkeypatch, label, inherited):
    from benchmarks.comparison import worker

    monkeypatch.setenv("XLA_FLAGS", inherited)
    monkeypatch.setattr(jax, "devices", lambda *args: [H100])
    monkeypatch.setattr(jax, "default_device", lambda device: contextlib.nullcontext())

    def child(*args, **kwargs):
        assert inherited in os.environ["XLA_FLAGS"].split()
        return {"warm_compiles": 0, "metadata": {"execution_devices": ["gpu:0 (NVIDIA H100)"]}}

    monkeypatch.setattr("tests.support.precision_timing._child", child)
    worker({"backend": "gpu", "gemm": "auto", "label": label, "campaign": "comparison_single",
            "paths": 128, "seeds": [1, 2, 3]}, tmp_path)
    assert json.loads((tmp_path / "result.json").read_text())["status"] == "ok"
