"""
Range diagnostics of products below float32 (`ProductOptions.diagnostics="counts"`,
`engine.precision.diagnostics`): each operand value an FP8 product could not hold is counted, by
stage and operand, in the analytic's precision report; off, nothing is counted or changed.
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from engine.precision import Arithmetic, ExecutionPolicy, Precision, ProductOptions, StagePrecision
from engine.precision.diagnostics import collecting
from engine.precision.products import matmul
from engine.risk.market import RateRiskFactors
from engine.risk.market.scenarios import MonteCarloScenarios
from engine.run import MarketRiskConfig, ResultChecks, RunRequest, RunRequestConfig, run
from tests.support import portfolio as shared

COUNTED = ProductOptions(diagnostics="counts")
FP8 = Arithmetic("float8_e4m3fn", "float32", execution="allow_emulation", product=COUNTED, stage="test")
#: Per-path operand values with a known fate in E4M3 (largest 448, smallest normal 2^-6, smallest
#: subnormal 2^-9): overflow, overflow, underflow to 0, subnormal, normal, already NaN, zero.
VALUES = np.asarray([500.0, -1000.0, 1e-4, 0.005, 1.0, np.nan, 0.0], dtype=np.float32)
EXPECTED = {"values": 7, "nonfinite": 1, "overflow": 2, "underflow": 1, "subnormal": 1}


def _ranges(f, *args):
    with collecting(True) as ranges:
        jax.block_until_ready(f(*args))
    return ranges.counts


def test_each_operand_value_is_counted_by_its_fate():
    counts = _ranges(lambda a, b: matmul(a, b, FP8, shared="right"), VALUES[None, :], np.ones((7, 1), np.float32))
    assert counts["test"]["per_path"] == EXPECTED
    assert counts["test"]["shared"] == {"values": 7, "nonfinite": 0, "overflow": 0, "underflow": 0, "subnormal": 0}


def test_a_batch_is_summed_once_under_jit_and_vmap():
    one = lambda a: matmul(a, jnp.ones((7, 1), jnp.float32), FP8, shared="right")  # noqa: E731
    batch = jnp.stack([jnp.asarray(VALUES)[None, :]] * 3)
    counts = _ranges(jax.jit(jax.vmap(one)), batch)
    assert counts["test"]["per_path"] == {k: 3 * v for k, v in EXPECTED.items()}


def test_scaled_products_overflow_nothing_finite():
    scaled = dataclasses.replace(FP8, product=ProductOptions(scaling="power_of_two", diagnostics="counts"))
    counts = _ranges(lambda a: matmul(a, np.ones((7, 1), np.float32), scaled, shared="right"),
                     np.where(np.isnan(VALUES), 1.0, VALUES)[None, :])
    assert counts["test"]["per_path"]["overflow"] == 0


def test_off_nothing_is_counted_and_the_stage_is_unnamed():
    assert Precision(pricing=StagePrecision(compute="float8_e4m3fn", accumulate="float32")).arithmetic_for(
        StagePrecision(compute="float8_e4m3fn", accumulate="float32"), "values/x", "pricing").stage == ""
    off = dataclasses.replace(FP8, product=ProductOptions())
    assert _ranges(lambda a: matmul(a, np.ones((7, 1), np.float32), off, shared="right"), VALUES[None, :]) == {}


def test_two_collections_at_once_are_refused():
    with collecting(True):
        with pytest.raises(RuntimeError, match="another run in this process is collecting"):
            with collecting(True):
                pass


def test_bfloat16_3x_has_no_ranges_to_count():
    with pytest.raises(ValueError, match="bfloat16_3x"):
        COUNTED.validate("bfloat16_3x")


def _market_risk(product):
    factors = RateRiskFactors.from_market(shared.market())
    scenarios = MonteCarloScenarios(factors, np.eye(factors.size) * 0.002 ** 2, 10, 64, seed=3)
    precision = Precision(pricing=StagePrecision(compute="float8_e4m3fn", accumulate="float32", storage="float32",
                                                 product=product),
                          execution=ExecutionPolicy(mode="allow_emulation"))
    config = RunRequestConfig(market_risk=MarketRiskConfig(scenarios), precision=precision,
                              checks=ResultChecks("allow"))
    return run(RunRequest(shared.market(), ("market_risk",), (shared.trades()["swap-payer"],), config))


def test_a_run_reports_why_unscaled_fp8_is_nan_and_scaled_fp8_is_not():
    unscaled = _market_risk(COUNTED).market_risk.precision.product_ranges["pricing"]
    assert sum(r["overflow"] for r in unscaled.values()) > 0
    assert all(r["values"] > 0 for r in unscaled.values())
    scaled = _market_risk(dataclasses.replace(ProductOptions.compensated("float8_e4m3fn"), diagnostics="counts"))
    ranges = scaled.market_risk.precision.product_ranges["pricing"]
    assert sum(r["overflow"] for r in ranges.values()) == 0 and np.isfinite(np.asarray(scaled.market_risk.pnl)).all()
