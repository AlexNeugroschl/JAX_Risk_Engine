"""Boundary diagnostics must detect faulty stages without changing the engine policy."""
import jax
import numpy as np
import pytest

from engine.precision import Arithmetic, products
from tests.support.product_boundaries import diagnose, product_check


@pytest.mark.parametrize("name", ["float16", "bfloat16", "float8_e4m3fn", "float8_e5m2",
                                "float16_1x", "bfloat16_1x", "float8_e4m3fn_1x", "float8_e5m2_1x"])
@pytest.mark.parametrize("shared", ["left", "right"])
@pytest.mark.parametrize("rounding", ["nearest", "stochastic"])
def test_boundary_captures_match_independent_grids_and_products(name, shared, rounding, tmp_path):
    cpu = jax.devices("cpu")[0]
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=(3, 4)), rng.normal(size=(4, 2))
    with jax.default_device(cpu):
        policy = Arithmetic(name, "float32", rounding=rounding,
            key=jax.random.PRNGKey(7) if rounding == "stochastic" else None, execution="allow_emulation")
        result = diagnose(cpu, a, b, policy, shared, tmp_path)
    assert result["complete"] and result["passes"], result
    assert result["native_instruction_qualification"] == "not_established"
    with np.load(tmp_path / "buffers.npz") as buffers:
        assert {"input/a", "input/b", "qa", "qb", "ka", "kb", "raw", "staged", "fused"} <= set(buffers.files)
    assert (tmp_path / "raw.hlo.txt").is_file() and (tmp_path / "fused.hlo.txt").is_file()


@pytest.mark.parametrize("fault", ["quantization", "dot", "compile"])
def test_wrong_stage_or_compilation_cannot_erase_saved_operands(fault, tmp_path, monkeypatch):
    cpu = jax.devices("cpu")[0]
    jax.clear_caches()
    if fault == "quantization":
        original = products._terms
        def wrong(*args, **kwargs):
            values, k = original(*args, **kwargs)
            return values * 0, k
        monkeypatch.setattr(products, "_terms", wrong)
    else:
        original = products._matmul32
        def wrong(a, b, precision):
            if fault == "compile":
                raise ValueError("injected compiler refusal")
            return original(a, b, precision) + 64
        monkeypatch.setattr(products, "_matmul32", wrong)
    with jax.default_device(cpu):
        result = diagnose(cpu, np.ones((3, 4)), np.ones((4, 2)),
            Arithmetic("float16", "float32", execution="allow_emulation"), "right", tmp_path)
    assert not result["passes"]
    if fault == "quantization":
        assert not result["checks"]["quantization/a"]["passes"]
    elif fault == "dot":
        assert not result["checks"]["raw_product"]["passes"]
    else:
        assert not result["complete"] and "injected compiler refusal" in result["error"]
    with np.load(tmp_path / "buffers.npz") as buffers:
        assert {"qa", "qb", "ka", "kb"} <= set(buffers.files)
    jax.clear_caches()


def test_cancellation_check_uses_absolute_products_and_rejects_wrong_exact_zero():
    assert product_check(np.array([0.]), np.array([0.]), np.array([0.]), 4)["passes"]
    assert not product_check(np.array([1.]), np.array([0.]), np.array([0.]), 4)["passes"]
    assert not product_check(np.array([np.nan]), np.array([0.]), np.array([1.]), 4)["passes"]
