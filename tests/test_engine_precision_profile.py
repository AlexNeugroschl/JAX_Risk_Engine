"""End-to-end profiles must change arithmetic, not just cast final outputs."""
import dataclasses
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from engine.api.requests import PrecisionSchema, RunRequestSchema
from engine.precision import ExecutionPolicy, Precision, StagePrecision
from engine.run import run
from tests.test_run_request import _request


@pytest.mark.parametrize("profile", ["float32", "float64"])
def test_profile_expands_and_round_trips(profile):
    policy = Precision(profile=profile)
    assert PrecisionSchema.model_validate({"profile": profile}).to_dataclass() == policy
    assert PrecisionSchema.from_dataclass(policy).to_dataclass() == policy
    for role in dataclasses.fields(policy.internal):
        assert getattr(policy.internal, role.name) == profile
    assert policy.simulation == StagePrecision(profile, profile, profile)


def test_profile_overrides_are_explicit_and_do_not_change_placement():
    policy = PrecisionSchema.model_validate({"profile": "float32", "internal": {"reductions": "float64"},
        "pricing": {"compute": "bfloat16"}}).to_dataclass()
    assert policy.pricing == StagePrecision("float32", "bfloat16", "float32")
    assert policy.internal.reductions == "float64"
    assert policy.internal.calibration == "float32"
    assert policy.execution == ExecutionPolicy()
    override = PrecisionSchema.model_validate({"profile": "float32", "by_trade": {
        "swap": {"compute": "bfloat16"}}}).to_dataclass().by_trade["swap"]
    assert override == StagePrecision("float32", "bfloat16", "float32")


@pytest.mark.parametrize("analytic", ["npv", "exposure", "market_risk", "calibration", "sensitivities"])
def test_fp32_analytics_work_without_x64_or_implicit_demotion(analytic):
    request = _request((analytic,))
    request = dataclasses.replace(request, config=dataclasses.replace(request.config, precision=Precision(profile="float32")))
    # Disabling x64 is a test instrument only. Production never toggles it per job.
    with jax.default_device(jax.devices("cpu")[0]), jax.enable_x64(False), warnings.catch_warnings():
        warnings.filterwarnings("error", message="Explicitly requested dtype.*")
        result = run(request)
    if result.portfolio:
        assert result.portfolio.npv_cube.dtype == jnp.float32
        assert np.isfinite(result.portfolio.base_npv)
        if result.portfolio.exposure:
            assert result.portfolio.exposure.epe.dtype == jnp.float32
        if result.portfolio.greeks:
            assert all(v.dtype == np.float32 for g in result.portfolio.greeks.values() for v in g.values())
    if result.market_risk:
        assert result.market_risk.pnl.dtype == jnp.float32
        assert np.isfinite(result.market_risk.portfolio_pnl).all()
    if result.calibration:
        assert all(c.sigma.values.dtype == jnp.float32 for c in result.calibration.values())


def test_fp32_profile_allows_tpu_but_never_downgrades_fp64():
    from types import SimpleNamespace
    tpu = SimpleNamespace(platform="tpu", device_kind="TPU v6e")
    assert Precision(profile="float32").validate_execution(device=tpu) is tpu
    with pytest.raises(ValueError, match="float64 is unsupported on TPU"):
        Precision().validate_execution(device=tpu)
    with pytest.raises(ValueError, match="paired_reference=float64"):
        Precision(profile="float32", paired_fraction=0.5).validate_execution(device=tpu)
    # Unused FP64 declarations must not block an explicitly FP32 calibration-only job.
    from engine.precision import InternalPrecision
    policy = Precision(internal=InternalPrecision(calibration="float32", output="float32"))
    assert policy.validate_execution(stages=(), internal_roles=("calibration", "output"), device=tpu) is tpu


@pytest.mark.slow
def test_fp32_reference_prices_calibration_and_ad_greeks():
    from engine.run import GreeksConfig

    request = _request(("npv", "calibration", "sensitivities"))
    request = dataclasses.replace(request, config=dataclasses.replace(
        request.config, sensitivity=GreeksConfig(method="AD")))
    with jax.default_device(jax.devices("cpu")[0]):
        reference = run(request)
        low = run(dataclasses.replace(request, config=dataclasses.replace(
            request.config, precision=Precision(profile="float32"))))
    # A deterministic, well-conditioned swap/European/bond fixture. This is a
    # regression tolerance for FP32 rounding, not broad financial qualification.
    np.testing.assert_allclose(low.portfolio.base_npv_per_trade,
                               reference.portfolio.base_npv_per_trade, rtol=1e-4)
    for currency, calibration in reference.calibration.items():
        np.testing.assert_allclose(low.calibration[currency].sigma.values,
                                   calibration.sigma.values, rtol=1e-4)
    for index, greeks in reference.portfolio.greeks.items():
        for name, expected in greeks.items():
            # Theta subtracts today's and tomorrow's full valuations, so FP32
            # cancellation has a larger tolerance than the AD derivatives.
            tolerance = 5e-3 if name == "theta" else 1e-4
            bound = max(1e-3, tolerance * float(np.max(np.abs(expected))))
            np.testing.assert_allclose(low.portfolio.greeks[index][name], expected,
                                       rtol=0.0, atol=bound, err_msg=name)


def test_api_execution_is_separate_and_conflicts_are_refused():
    from tests.test_api_completeness import MARKET
    request = RunRequestSchema.model_validate({"market": MARKET, "analytics": ["npv"], "config": {
        "precision": {"profile": "float32"}, "execution": {"backend": "cpu"}}}).check()
    assert request.config.execution.backend == "cpu"
    assert request.config.precision.execution.backend == "cpu"
    from engine.run.run_request import portfolio_request
    assert portfolio_request(request).config.execution.backend == "cpu"
    with pytest.raises(ValueError, match="contradicts"):
        RunRequestSchema.model_validate({"market": MARKET, "analytics": ["npv"], "config": {
            "precision": {"execution": {"backend": "gpu"}}, "execution": {"backend": "cpu"}}}).check()


@pytest.mark.parametrize("extra", [{"simulation": None}, {"internal": 32}, {"by_trade": None}, {"by_trade": {"s": 32}}])
def test_invalid_profile_overrides_are_validation_errors(extra):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        PrecisionSchema.model_validate({"profile": "float32", **extra})


def test_fp32_kernel_compilation_has_no_fp64_arithmetic(monkeypatch):
    import re
    from jax._src import compiler
    request = _request(("exposure", "market_risk", "calibration", "sensitivities"))
    request = dataclasses.replace(request, config=dataclasses.replace(request.config,
                                   precision=Precision(profile="float32")))
    # Inspect loaded executables too: a persistent-cache hit contains the same
    # optimized HLO, and must not make this check depend on prior test order.
    original = compiler.compile_or_get_cached
    compiled, violations = [], []

    def inspect(*args, **kwargs):
        executable = original(*args, **kwargs)
        for module in executable.hlo_modules():
            compiled.append(module.name)
            # Weak Python scalar parameters/constants and data movement are not
            # arithmetic. Inspect fusion bodies as well as the entry computation.
            for line in module.to_string().splitlines():
                if (re.search(r"= f64(?:\[|\s)", line)
                        and not re.search(r"\b(parameter|constant|convert|copy|fusion|get-tuple-element|dynamic-slice|select)\(", line)):
                    violations.append((module.name, line))
        return executable

    jax.clear_caches()
    monkeypatch.setattr(compiler, "compile_or_get_cached", inspect)
    with jax.default_device(jax.devices("cpu")[0]), jax.enable_x64(True):
        result = run(request)
    assert compiled and any("bootstrap" in name for name in compiled)
    assert not violations, violations
    assert set(result.numerics["internal"].values()) == {"float32"}
    assert result.numerics["qualification"]["affects_execution"] is False


def test_api_preserves_fp32_pnl_artifact_and_resolved_numerics():
    from engine.api.results import run_result_document
    from tests.test_api_completeness import MARKET, MONTE_CARLO, SWAP
    schema = RunRequestSchema.model_validate({"market": MARKET, "analytics": ["market_risk"],
        "portfolio": [{"trade_type": "swap", "trade_id": "swap", **SWAP}],
        "config": {"precision": {"profile": "float32"}, "execution": {"backend": "cpu"},
        "market_risk": {"scenarios": {**MONTE_CARLO, "num_scenarios": 128}}}})
    request = schema.check()
    with jax.default_device(jax.devices("cpu")[0]):
        result = run(request)
        document, chunks = run_result_document(result, schema, "fp32", "/artifacts")
    assert document.market_risk.pnl.dtype == "float32"
    np.testing.assert_array_equal(np.frombuffer(b"".join(chunks["pnl"]), dtype="<f4").reshape(result.market_risk.pnl.shape),
                                  np.asarray(result.market_risk.pnl))
    assert document.numerics["profile"] == "float32"
