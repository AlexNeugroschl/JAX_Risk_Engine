"""Numerical strategies are explicit, validated, compiled and reported independently."""
import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from engine.api.requests import PrecisionSchema
from engine.api.results import PrecisionReportSchema
from engine.precision import Arithmetic, ExecutionPolicy, Precision, PrecisionReport, ProductOptions, StagePrecision, matmul
from engine.precision import products, runtime
from engine.precision.evidence import combination
from engine.precision.execution import validate_product
from engine.precision.storage import Stored, load


@pytest.fixture
def cpu():
    with jax.default_device(jax.devices("cpu")[0]):
        yield jax.devices("cpu")[0]


def test_basic_default_is_one_unscaled_product_without_correction(cpu):
    policy = Arithmetic("float16", "float32", execution="allow_emulation")
    a = np.array([[1.0004, -2.003], [0.2, 0.3]], np.float32)
    b = np.array([[0.7, -0.8], [1.001, 0.03]], np.float32)
    fn = jax.jit(lambda a, b, p: matmul(a, b, p, "right"))
    expected = a.astype(np.float16).astype(np.float32) @ b.astype(np.float16).astype(np.float32)
    np.testing.assert_allclose(fn(a, b, policy), expected, rtol=2e-7)
    program = str(jax.make_jaxpr(lambda a, b: matmul(a, b, policy, "right"))(a, b))
    assert program.count("dot_general[") == 1
    assert "cond[" not in program
    assert ProductOptions() == policy.product == StagePrecision("float32", "float16", "float32").product


@pytest.mark.parametrize("compute", ["float16", "bfloat16", "tensorfloat32", "bfloat16_3x", "float8_e4m3fn"])
def test_forced_software_products_use_only_fp32_operands(compute, cpu):
    policy = Arithmetic(compute, "float32", execution="allow_emulation",
                        product=ProductOptions(implementation="emulated_fp32"))
    a, b = np.ones((3, 4), np.float32), np.ones((4, 2), np.float32)
    program = jax.make_jaxpr(lambda a, b: matmul(a, b, policy, "right"))(a, b)
    dots = [e for e in program.eqns if e.primitive.name == "dot_general"]
    assert len(dots) == (3 if compute == "bfloat16_3x" else 1)
    assert all(v.aval.dtype == jnp.float32 for e in dots for v in e.invars)
    np.testing.assert_array_equal(jax.jit(lambda a, b: matmul(a, b, policy, "right"))(a, b), a @ b)


def test_full_fp8_algorithm_is_an_explicit_request(cpu):
    gpu = SimpleNamespace(platform="gpu", device_kind="H100", compute_capability="9.0")
    requested = ProductOptions(algorithm="fp8_full")
    assert products.product_precision("float8_e4m3fn", gpu) == jax.lax.Precision.HIGHEST
    assert products.product_precision("float8_e4m3fn", gpu, options=requested) == jax.lax.DotAlgorithmPreset.ANY_F8_ANY_F8_F32
    validate_product("float8_e4m3fn", "float32", "allow_emulation", device=gpu, options=requested)
    validate_product("float8_e4m3fn", "float32", device=gpu, options=requested)


def test_basic_fp8_overflow_is_not_silently_repaired(cpu):
    basic = Arithmetic("float8_e4m3fn", "float32", execution="allow_emulation")
    scaled = dataclasses.replace(basic, product=ProductOptions(scaling="power_of_two"))
    fn = jax.jit(lambda a, b, p: matmul(a, b, p, "right"))
    a, b = np.ones((2, 2), np.float32), np.full((2, 2), 1000., np.float32)
    assert not np.isfinite(fn(a, b, basic)).any()
    assert np.isfinite(fn(a, b, scaled)).all()


@pytest.mark.parametrize("count", [2, 3, 4])
@pytest.mark.parametrize("scaling", ["none", "power_of_two"])
def test_residual_strategies_match_independent_staged_oracle(count, scaling, tmp_path, cpu):
    from tests.support.product_boundaries import diagnose
    rng = np.random.default_rng(19)
    a, b = rng.normal(size=(3, 4)), rng.normal(size=(4, 2))
    arithmetic = Arithmetic("float8_e5m2", "float32", execution="allow_emulation",
                            product=ProductOptions(shared_terms=count, scaling=scaling, shared_format="e4m3"))
    result = diagnose(cpu, a, b, arithmetic, "right", tmp_path)
    assert result["complete"] and result["passes"], result


@pytest.mark.parametrize("field,value,compute", [
    ("shared_terms", 2, "float16_1x"), ("shared_terms", 2, "bfloat16_3x"),
    ("scaling", "power_of_two", "bfloat16"), ("shared_format", "e4m3", "float16"),
    ("bf16_padding", "gemm8", "float16"), ("subnormals", "normalize_fp16", "bfloat16"),
    ("algorithm", "fp8_full", "float16"), ("shared_terms", 2, "float32"),
])
def test_incompatible_options_are_refused_during_configuration(field, value, compute):
    with pytest.raises(ValueError):
        StagePrecision("float32", compute, "float32", product=ProductOptions(**{field: value}))


@pytest.mark.parametrize("count", [True, 0, 5, 1.5])
def test_shared_terms_requires_an_actual_bounded_integer(count):
    with pytest.raises(ValueError):
        ProductOptions(shared_terms=count)


def test_api_round_trip_keeps_each_override_and_execution_requirement():
    stage = StagePrecision("float32", "float16", "float32", product=ProductOptions(
        shared_terms=3, scaling="power_of_two", subnormals="normalize_fp16"))
    policy = Precision(pricing=stage, by_product={"swap": StagePrecision.compensated("float32", "bfloat16", "float32")},
                       by_trade={"one": dataclasses.replace(stage, product=ProductOptions())},
                       execution=ExecutionPolicy("gpu", "allow_emulation", "cublas"))
    wire = PrecisionSchema.from_dataclass(policy).model_dump(mode="json")
    restored = PrecisionSchema.model_validate(wire).to_dataclass()
    assert restored == policy
    assert restored.precision_for(SimpleNamespace(trade_id="one", product="swap")).product == ProductOptions()
    assert restored.precision_for(SimpleNamespace(trade_id="two", product="swap")).product.shared_terms == 2
    assert restored.arithmetic_for(stage, "pricing").product == stage.product
    with pytest.raises(ValueError):
        PrecisionSchema.model_validate({"pricing": {"product": {"silent_fallback": True}}})


def test_strategy_changes_jit_identity_but_not_random_seeds(cpu):
    traces = []
    def calculate(a, b, p):
        traces.append(p.product)
        return matmul(a, b, p, "right")
    fn = jax.jit(calculate)
    # The shared operand is intentionally between grid points: its residual
    # should change the answer, while the basic product rounds it away.
    a, b = np.ones((2, 2), np.float32), np.full((2, 2), 1.0004, np.float32)
    basic = Arithmetic("float16", "float32", execution="allow_emulation")
    residual = dataclasses.replace(basic, product=ProductOptions(shared_terms=2))
    first = np.asarray(fn(a, b, basic))
    second = np.asarray(fn(a, b, residual))
    assert np.max(np.abs(first - second)) > 0.0005
    fn(a, b, residual).block_until_ready()
    assert traces == [basic.product, residual.product]
    stochastic = dataclasses.replace(residual, rounding="stochastic", key=jax.random.PRNGKey(1))
    fn(a, b, stochastic).block_until_ready()
    fn(a, b, dataclasses.replace(stochastic, key=jax.random.PRNGKey(2))).block_until_ready()
    assert len(traces) == 3


def test_storage_scaling_can_be_disabled_and_round_trips_over_api(cpu):
    policy = Precision(pricing=StagePrecision("float16"), storage_scaling="none", offset="none")
    x = np.array([1.0004, -2.003, 0.01], np.float64)
    stored = policy.store(x, "float16", "values/one")
    assert not isinstance(stored, Stored) and stored.dtype == jnp.float16
    np.testing.assert_array_equal(load(stored, jnp.float64), x.astype(np.float16).astype(np.float64))
    assert PrecisionSchema.from_dataclass(policy).to_dataclass() == policy
    with pytest.raises(ValueError, match="offset='none'"):
        Precision(pricing=StagePrecision("float16"), storage_scaling="none")


def test_startup_validation_rejects_native_only_strategies_on_cpu(cpu):
    for compute, options in [("float16", ProductOptions(subnormals="normalize_fp16")),
                             ("float8_e4m3fn", ProductOptions(algorithm="fp8_full"))]:
        policy = Precision.research(pricing=StagePrecision("float32", compute, "float32", product=options))
        with pytest.raises(ValueError, match="require a capable GPU"):
            policy.validate_execution(device=cpu)
    with pytest.raises(ValueError, match="requires native execution"):
        validate_product("float16", "float32", device=cpu, options=ProductOptions(implementation="emulated_fp32"))


def test_job_rejects_unsupported_strategy_before_calibration(cpu, portfolio_request, monkeypatch):
    from engine.run import pipeline
    policy = Precision.research(pricing=StagePrecision("float32", "float16", "float32",
        product=ProductOptions(subnormals="normalize_fp16")))
    portfolio_request.config = dataclasses.replace(portfolio_request.config, precision=policy)
    def no_work(*args, **kwargs):
        pytest.fail("calibration started before strategy validation")
    monkeypatch.setattr(pipeline, "_build_cross_asset_model", no_work)
    with pytest.raises(ValueError, match="require a capable GPU"):
        pipeline.price_on_market(portfolio_request)


def test_plain_storage_and_product_options_survive_a_real_job(cpu, portfolio_request):
    from engine.run import price_portfolio
    options = ProductOptions(shared_terms=2, scaling="power_of_two", implementation="emulated_fp32")
    stage = StagePrecision("float16", "float16", "float32", product=options)
    policy = Precision.research(pricing=stage, storage_scaling="none", offset="none")
    portfolio_request.config = dataclasses.replace(portfolio_request.config, precision=policy)
    result = price_portfolio(portfolio_request)
    assert np.isfinite(result.npv_cube).all()
    assert result.precision.realized["values/swap-2y"] == "float16"
    assert result.precision.resolved_policy["storage_encoding"] == "plain"
    assert result.precision.resolved_policy["pricing"]["product"] == dataclasses.asdict(options)
    assert any(item["options"]["implementation"] == "emulated_fp32"
               for item in result.precision.resolved_policy["executed_products"])
    PrecisionReportSchema.from_dataclass(result.precision).model_dump(mode="json")


def test_worker_route_is_frozen_and_mismatches_fail_before_work(monkeypatch):
    monkeypatch.setattr(runtime, "_RUNTIME", None)
    frozen = runtime.capture_compiler_runtime({"XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false --some-other-flag"})
    assert frozen.gpu_gemm == "cublas"
    assert runtime.capture_compiler_runtime({"XLA_FLAGS": ""}) is frozen
    gpu = SimpleNamespace(platform="gpu", compute_capability="9.0")
    policy = Precision(execution=ExecutionPolicy("gpu", "native", "cublas"))
    assert policy.validate_execution(device=gpu) is gpu
    with pytest.raises(ValueError, match="worker started with --gpu-gemm default"):
        dataclasses.replace(policy, execution=ExecutionPolicy("gpu", "native", "default")).validate_execution(device=gpu)
    with pytest.raises(ValueError, match="requires an active GPU"):
        ExecutionPolicy(gpu_gemm="cublas").validate(device=SimpleNamespace(platform="cpu"))


def test_report_keeps_mixed_dispatch_and_selected_strategy(monkeypatch):
    gpu = SimpleNamespace(platform="gpu", id=0, device_kind="test GPU", compute_capability="9.0")
    from engine.precision import report
    monkeypatch.setattr(report, "_devices", lambda arrays: [gpu])
    policy = Precision.research()
    native = Arithmetic("float16", "float32", product=ProductOptions(subnormals="normalize_fp16"))
    emulated = dataclasses.replace(native, product=ProductOptions(implementation="emulated_fp32"))
    result = PrecisionReport.of(policy, (), {}, (), paths=0, computed=(native, emulated))
    assert result.products == {"float16": "mixed"}
    choices = result.resolved_policy["executed_products"]
    assert choices[0]["options"]["subnormals"] == "normalize_fp16"
    assert choices[1]["dispatch"] == "emulated"
    assert "compiler_runtime" in result.resolved_policy
    PrecisionReportSchema.from_dataclass(result).model_dump(mode="json")


def test_old_acceptance_evidence_does_not_apply_to_new_algorithms():
    basic = StagePrecision("float32", "float16", "float32")
    old = StagePrecision.compensated("float32", "float16", "float32")
    old_key = combination(Precision(pricing=old), ("pricing",), old)
    assert old_key == "pricing=float32/float16/float32"
    assert combination(Precision(pricing=basic), ("pricing",), basic) != old_key
    raw = Precision(pricing=StagePrecision("float16"), storage_scaling="none", offset="none")
    assert "storage_scaling=none" in combination(raw, ("pricing",), raw.pricing)


def test_archived_measurements_keep_the_historical_profile_without_changing_requests():
    from tests.support.precision_evidence import recorded_precision
    old = {"pricing": {"storage": "float32", "compute": "float16", "accumulate": "float32"}}
    decoded = recorded_precision(old)
    assert decoded.pricing == StagePrecision.compensated("float32", "float16", "float32")
    assert "product" not in old["pricing"]  # archived source stays intact
    request = PrecisionSchema.model_validate(old).to_dataclass()
    assert request.pricing.product == ProductOptions()
    explicit = PrecisionSchema.from_dataclass(request).model_dump(mode="json")
    assert recorded_precision(explicit) == request


def test_normalization_preserves_the_known_subnormal_counterexample_on_gpu(tmp_path):
    gpu = next((d for d in jax.devices() if d.platform == "gpu"), None)
    if gpu is None:
        pytest.skip("requires a native FP16 GPU")
    a, b = np.zeros((16, 16), np.float32), np.zeros((16, 16), np.float32)
    a[0, :2] = [-2.**-24, -18 * 2.**-24]
    b[:2, 0] = [-6380., -10.0390625]
    expected = a.astype(np.float64) @ b.astype(np.float64)
    policy = Arithmetic("float16", "float32", product=ProductOptions(subnormals="normalize_fp16"))
    with jax.default_device(gpu):
        program = jax.jit(lambda a, b: matmul(a, b, policy, "right"))
        actual = np.asarray(program(a, b), np.float64)
        np.testing.assert_allclose(actual, expected, rtol=4e-7, atol=0.)
        # The selected fast branch still gives normal native products.
        np.testing.assert_array_equal(program(np.ones_like(a), np.ones_like(b)), np.ones_like(a) * 16)
        assert "cond[" in str(jax.make_jaxpr(lambda a, b: matmul(a, b, policy, "right"))(a, b))
        from tests.support.product_boundaries import diagnose
        captured = diagnose(gpu, a, b, policy, "right", tmp_path)
        assert captured["complete"] and captured["passes"], captured
