"""Execution requirements fail before numerical work; reports never hide fixed FP64."""
import dataclasses
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from engine.precision import Arithmetic, ExecutionPolicy, InternalPrecision, Precision, PrecisionReport, StagePrecision, matmul
from engine.precision import products
from engine.precision.execution import validate_product
from engine.api.requests import PrecisionSchema, RunRequestSchema
from engine.api.results import PrecisionReportSchema


CPU = SimpleNamespace(platform="cpu", device_kind="cpu")
TPU = SimpleNamespace(platform="tpu", device_kind="TPU v6e")
H100 = SimpleNamespace(platform="gpu", device_kind="NVIDIA H100", compute_capability="9.0")


@pytest.fixture(autouse=True)
def policy_tests_on_cpu():
    """Test policy behavior deterministically even when the suite defaults to a GPU."""
    with jax.default_device(jax.devices("cpu")[0]):
        yield


@pytest.mark.parametrize("device", [CPU, H100])
def test_fp64_reference_requires_and_accepts_native_fp64(device):
    assert Precision.reference(backend=device.platform).validate_execution(device=device) is device


@pytest.mark.parametrize("policy", [Precision(), Precision.throughout("float32"), Precision.research(),
                                    Precision(paired_fraction=0.5)])
def test_full_tpu_run_is_refused_even_with_adjustable_fp32_or_emulation(policy):
    with pytest.raises(ValueError, match="internal.calibration=float64 is unsupported on TPU"):
        policy.validate_execution(device=TPU)


@pytest.mark.parametrize("role", ["compute", "accumulate"])
@pytest.mark.parametrize("mode", ["native", "allow_emulation"])
def test_tpu_never_accepts_fp64_products_or_accumulators(role, mode):
    args = {"compute": "float32", "accumulate": "float32", role: "float64"}
    with pytest.raises(ValueError, match=f"{role}=float64 is unsupported on TPU"):
        validate_product(**args, mode=mode, device=TPU)


def test_tpu_primitive_fp32_and_bf16_pass_capability_checks():
    for name in ("float32", "bfloat16", "bfloat16_3x"):
        validate_product(name, "float32", device=TPU)


@pytest.mark.parametrize("field", [f.name for f in dataclasses.fields(InternalPrecision)])
def test_fixed_kernel_precision_cannot_be_relabelled_float32(field):
    with pytest.raises(ValueError, match=f"InternalPrecision.{field}: only float64"):
        InternalPrecision(**{field: "float32"})
    with pytest.raises(ValueError):
        PrecisionSchema.model_validate({"internal": {field: "float32"}})


def test_named_backend_is_a_requirement_and_never_a_fallback():
    with pytest.raises(ValueError, match="execution.backend='gpu'.*active device is cpu"):
        Precision(execution=ExecutionPolicy(backend="gpu")).validate_execution(device=CPU)


def test_x64_disabled_is_refused_without_changing_the_global_setting():
    with jax.enable_x64(False):
        with pytest.raises(ValueError, match="implicit FP32 demotion is refused"):
            Precision().validate_execution(device=CPU)
        assert not jax.config.x64_enabled


@pytest.mark.parametrize("name", ["float16", "bfloat16", "tensorfloat32", "float8_e4m3fn"])
def test_native_only_rejects_cpu_emulation(name):
    with pytest.raises(ValueError, match="requires native execution on cpu"):
        validate_product(name, "float32", device=CPU)
    validate_product(name, "float32", mode="allow_emulation", device=CPU)


def test_explicit_process_emulation_cannot_override_native_requirement(monkeypatch):
    monkeypatch.setenv(products.EMULATE_PRODUCTS, "bfloat16")
    with pytest.raises(ValueError, match="this path is emulated"):
        validate_product("bfloat16", "float32", device=H100)


@pytest.mark.parametrize("name", ["float8_e4m3fn", "float8_e5m2_1x"])
def test_capability_is_not_fp8_instruction_qualification(name):
    assert products.product_execution(name, H100) == "native"
    with pytest.raises(ValueError, match="native FP8 execution is not qualified"):
        validate_product(name, "float32", device=H100)
    validate_product(name, "float32", mode="allow_emulation", device=H100)


def test_primitive_research_execution_is_explicit_and_survives_jit(monkeypatch):
    monkeypatch.setattr(products, "default_device", lambda: CPU)
    a, b = np.arange(12, dtype=np.float32).reshape(3, 4), np.ones((4, 2), dtype=np.float32)
    native = Arithmetic("float16", "float32")
    with pytest.raises(ValueError, match="requires native execution on cpu"):
        matmul(a, b, native, "right")
    research = dataclasses.replace(native, execution="allow_emulation")
    leaves, tree = jax.tree_util.tree_flatten(research)
    assert jax.tree_util.tree_unflatten(tree, leaves) == research
    compute = jax.jit(lambda x, y, arithmetic: matmul(x, y, arithmetic, "right"))
    np.testing.assert_array_equal(np.asarray(compute(a, b, research)), a @ b)
    with pytest.raises(ValueError, match="requires native execution on cpu"):
        compute(a, b, native)  # must not reuse the permissive executable


def test_reports_resolve_default_and_preserve_every_contract_field():
    policy = Precision.research(pricing=StagePrecision("float16", "float16", "float32"))
    report = PrecisionReport.of(policy, (), {}, (), paths=0)
    wire = PrecisionReportSchema.from_dataclass(report).model_dump(mode="json")
    resolved = wire["resolved_policy"]
    assert resolved["execution"] == {"backend": report.backend, "mode": "allow_emulation", "gpu_gemm": "inherit"}
    assert resolved["internal"] == dataclasses.asdict(InternalPrecision())
    assert resolved["pricing"]["compute"] == "float16"
    assert resolved["host_preprocessing"] == {"dtype": "float64", "backend": "cpu"}
    assert "algorithms" in resolved
    json.dumps(wire)
    assert PrecisionSchema.model_validate(wire["policy"]).to_dataclass() == policy


def test_http_validation_does_not_compare_worker_backend_to_api_cpu():
    from tests.test_api_completeness import MARKET
    request = RunRequestSchema.model_validate({"market": MARKET, "analytics": ["npv"],
        "config": {"precision": {"execution": {"backend": "gpu"}}}}).check()
    assert request.config.precision.execution.backend == "gpu"


def test_portfolio_refuses_tpu_before_calibration(portfolio_request, monkeypatch):
    from engine.run import pipeline
    monkeypatch.setattr(products, "default_device", lambda: TPU)
    def no_work(*args, **kwargs):
        pytest.fail("calibration ran before execution requirements were checked")
    monkeypatch.setattr(pipeline, "_build_cross_asset_model", no_work)
    with pytest.raises(ValueError, match="unsupported on TPU"):
        pipeline.price_on_market(portfolio_request)


def test_grid_rejects_known_invalid_tpu_fp64_representation(monkeypatch):
    from engine.precision.grid import _frexp_exponent, pow2
    one = jnp.asarray(1.0, dtype=jnp.float64)
    monkeypatch.setattr(products, "default_device", lambda: TPU)
    with pytest.raises(ValueError, match="precision grid=float64 is unsupported on TPU"):
        _frexp_exponent(one)
    with pytest.raises(ValueError, match="precision grid=float64 is unsupported on TPU"):
        pow2(jnp.asarray(0), jnp.float64)


def test_storage_and_loading_do_not_silently_create_tpu_fp64(monkeypatch):
    from engine.precision import load, store
    x = jnp.ones(4, dtype=jnp.float32)
    monkeypatch.setattr(products, "default_device", lambda: TPU)
    with pytest.raises(ValueError, match="storage=float64 is unsupported on TPU"):
        store(x, "float64")
    with pytest.raises(ValueError, match="load=float64 is unsupported on TPU"):
        load(x, jnp.float64)


def test_preexisting_arrays_on_another_platform_require_explicit_placement():
    x = jax.device_put(np.ones(4), jax.devices("cpu")[0])
    with pytest.raises(ValueError, match="input arrays are on.*cpu.*execution is on gpu"):
        Precision.reference(backend="gpu").validate_execution(device=H100, arrays=x)


def test_scenario_mesh_uses_selected_platform(monkeypatch):
    from engine.market_simulation import sharding
    calls = []
    monkeypatch.setattr(sharding, "default_device", lambda: CPU)
    monkeypatch.setattr(sharding.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(sharding.jax, "local_devices", lambda **kw: calls.append(kw) or [CPU])
    assert sharding._local_devices() == [CPU]
    assert calls == [{"backend": "cpu"}]


def test_native_emulation_refusal_happens_before_portfolio_calibration(portfolio_request, monkeypatch):
    from engine.run import pipeline
    policy = Precision(pricing=StagePrecision("float16", "float16", "float32"))
    portfolio_request.config = dataclasses.replace(portfolio_request.config, precision=policy)
    monkeypatch.setattr(products, "default_device", lambda: CPU)
    def no_work(*args, **kwargs):
        pytest.fail("calibration ran before native execution was checked")
    monkeypatch.setattr(pipeline, "_build_cross_asset_model", no_work)
    with pytest.raises(ValueError, match="requires native execution on cpu"):
        pipeline.price_on_market(portfolio_request)


def test_execution_checks_the_resolved_trade_override():
    trade = SimpleNamespace(trade_id="bond", product="bond")
    low = StagePrecision("float32", "float8_e4m3fn", "float32")
    policy = Precision(pricing=low, by_trade={"bond": StagePrecision()})
    policy.validate_execution(stages=("pricing",), trades=[trade], device=CPU)
    with pytest.raises(ValueError, match="requires native execution on cpu"):
        dataclasses.replace(policy, by_trade={}).validate_execution(
            stages=("pricing",), trades=[trade], device=CPU)


def test_direct_casts_and_levels_reject_tpu_fp64(monkeypatch):
    from engine.precision import level_terms
    level = jnp.asarray(1.0, dtype=jnp.float64)
    monkeypatch.setattr(products, "default_device", lambda: TPU)
    with pytest.raises(ValueError, match="accumulate=float64 is unsupported on TPU"):
        Arithmetic().cast(1.0)
    with pytest.raises(ValueError, match="level=float64 is unsupported on TPU"):
        level_terms(level, jnp.float32)


def _count_execution_checks(monkeypatch):
    checks = []
    original = Precision.validate_execution
    def checking(self, **kwargs):
        checks.append((self, kwargs))
        return original(self, **kwargs)
    monkeypatch.setattr(Precision, "validate_execution", checking)
    return checks


@pytest.mark.parametrize("analytics", [("npv",), ("exposure",), ("sensitivities",), ("market_risk",),
    ("calibration",), ("npv", "market_risk"), ("exposure", "market_risk", "calibration")])
def test_job_checks_execution_once_for_all_requested_analytics(analytics, monkeypatch):
    from engine.run import run
    from tests.test_run_request import _request
    checks = _count_execution_checks(monkeypatch)
    result = run(_request(analytics))
    assert len(checks) == 1
    if result.portfolio is not None:
        assert result.portfolio.trade_ids == list(result.trade_ids)


def test_paired_portfolio_checks_execution_once_including_reference_run(portfolio_request, monkeypatch):
    from engine.run import price_portfolio
    policy = Precision.throughout("float32")
    policy = dataclasses.replace(policy, paired_fraction=1.0)
    portfolio_request.config = dataclasses.replace(portfolio_request.config, precision=policy)
    checks = _count_execution_checks(monkeypatch)
    result = price_portfolio(portfolio_request)
    assert len(checks) == 1
    assert result.precision.paired_paths == result.precision.paths


def test_direct_market_risk_checks_once_including_paired_trades(monkeypatch):
    from engine.risk.market import MarketRiskRequest, run_market_risk
    from tests import market_risk_support as mr
    from engine.risk.market import monte_carlo_scenarios
    request = MarketRiskRequest([mr.swap(), mr.bond()], mr.market(),
        monte_carlo_scenarios(mr.factors(), mr.covariance(), 10, 64, seed=7),
        precision=Precision(paired_fraction=1.0))
    checks = _count_execution_checks(monkeypatch)
    result = run_market_risk(request)
    assert len(checks) == 1
    assert result.precision.paired_paths == 64


def test_all_analytics_are_validated_before_any_analytic_starts(monkeypatch):
    from engine.run import run
    from engine.run import run_request
    from tests.test_run_request import _request
    policy = Precision(pricing=StagePrecision("float32", "float8_e4m3fn", "float32"))
    request = _request(("npv", "market_risk"))
    request = dataclasses.replace(request, config=dataclasses.replace(request.config, precision=policy))
    def no_work(*args, **kwargs):
        pytest.fail("an analytic started before the complete job's execution validation")
    monkeypatch.setattr(run_request, "_price_portfolio", no_work)
    monkeypatch.setattr(run_request, "shock_scenarios", no_work)
    with pytest.raises(ValueError, match="requires native execution on cpu"):
        run(request)


def test_runtime_check_is_part_of_existing_validator_and_opt_in_on_worker():
    from engine.run import validate_run
    from tests.test_run_request import _request
    request = _request(("npv",))
    request = dataclasses.replace(request, config=dataclasses.replace(request.config,
        precision=Precision.reference(backend="gpu")))
    assert validate_run(request) is None  # HTTP/API CPU checks structure only.
    with pytest.raises(ValueError, match="execution.backend='gpu'"):
        validate_run(request, execution=True)


def test_cached_report_metadata_never_caches_execution_authorization(monkeypatch):
    from engine.precision.execution import prepare_execution
    policy = Precision()
    first = prepare_execution(policy)
    first.report_policy()  # Populate static metadata, then change runtime state.
    with jax.enable_x64(False):
        with pytest.raises(ValueError, match="implicit FP32 demotion is refused"):
            prepare_execution(policy)
    monkeypatch.setattr(products, "default_device", lambda: TPU)
    with pytest.raises(ValueError, match="unsupported on TPU"):
        prepare_execution(policy)


def test_emulation_flag_is_rechecked_for_each_job_despite_metadata_cache(monkeypatch):
    from engine.precision.execution import prepare_execution
    monkeypatch.setattr(products, "default_device", lambda: H100)
    stage = StagePrecision("float32", "bfloat16", "float32")
    policy = Precision(pricing=stage)
    prepare_execution(policy, stages=("pricing",)).report_policy()
    monkeypatch.setenv(products.EMULATE_PRODUCTS, "bfloat16")
    with pytest.raises(ValueError, match="this path is emulated"):
        prepare_execution(policy, stages=("pricing",))


def test_context_is_frozen_and_report_metadata_cannot_leak_between_jobs():
    from engine.precision.execution import prepare_execution
    first = prepare_execution(Precision())
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.device = None
    report = first.report_policy()
    report["internal"]["levels"] = "float32"
    report["execution"]["backend"] = "tpu"
    second = prepare_execution(Precision())
    assert second.report_policy()["internal"]["levels"] == "float64"
    assert second.report_policy()["execution"]["backend"] == "cpu"


def test_standalone_helpers_still_reject_unsupported_execution(portfolio_request, monkeypatch):
    from engine.market_simulation.config import simulate, build_cross_asset_model
    from engine.pricing.cube import value_today
    from engine.calibration.cam import calibrate_cam
    from engine.risk.market.revaluation import revalue_trade
    from tests import market_risk_support as mr
    monkeypatch.setattr(products, "default_device", lambda: TPU)
    operations = [
        lambda: simulate(portfolio_request.market, portfolio_request.config.simulation),
        lambda: build_cross_asset_model(portfolio_request.market, portfolio_request.config.simulation),
        lambda: value_today(portfolio_request.trades, portfolio_request.market, "USD"),
        lambda: calibrate_cam(portfolio_request.market, portfolio_request.config.simulation.ir),
        lambda: revalue_trade(mr.swap(), mr.market(), mr.factors(), np.zeros((4, mr.factors().size))),
    ]
    for operation in operations:
        with pytest.raises(ValueError, match="unsupported on TPU"):
            operation()


def test_static_metadata_is_resolved_once_without_caching_runtime_checks(monkeypatch):
    from engine.precision.execution import prepare_execution
    policy = Precision(rounding_seed=918273)
    resolutions = []
    original = Precision.resolved
    def resolving(self, device=None):
        resolutions.append(self)
        return original(self, device)
    monkeypatch.setattr(Precision, "resolved", resolving)
    checks = _count_execution_checks(monkeypatch)
    for _ in range(2):
        context = prepare_execution(policy)
        context.report_policy()
        context.report_policy()
    assert len(resolutions) == 1
    assert len(checks) == 2


def test_concurrent_jobs_keep_their_policies_and_report_copies_separate():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from engine.precision.execution import prepare_execution
    barrier = Barrier(2)
    def job(name):
        context = prepare_execution(Precision.throughout(name))
        barrier.wait(timeout=10)
        report = context.report_policy()
        report["pricing"]["compute"] = "mutated by this caller"
        return context.report_policy()["pricing"]["compute"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(job, ("float64", "float32"))) == ["float64", "float32"]
