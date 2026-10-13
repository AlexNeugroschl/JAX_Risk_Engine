"""
Compute below float32 (F-07; docs/planning/details/precision.md §3.1, §8): the kernels in
difference form with explicit accumulators, one implementation for every precision (A-16), and
the policy's product formats (A-22).

  * The products (`engine.precision.products`): every compute format multiplies what a unit of
    that format multiplies (operands rounded to its grid, per-row and per-column power-of-two
    scales into its range, float32 accumulation), the operand every path shares in two terms of
    the format and in one by its single pass (A-23), checked against an independent reference
    built with ml_dtypes; a row's value does not depend on the other rows (a path's value is the
    same in any batch); the shared operand's rounding biases a single pass's paths, not two
    terms'; native or emulated by device.
  * The arithmetic (`engine.precision.Arithmetic`) and the policy's rules for it.
  * Per family, at float64 the difference form is the direct formula to rounding, at any level;
    below float32 its error is relative to the paths' deviation from the level, not to the level
    (the near cancellation of a swap's legs, a long log discount factor, a log equity spot stays
    in the level): the simulation scan and the bridge, the scenario market, the legs and their
    path fixings, the Bachelier and Jamshidian Europeans, market risk's P&L, the Bermudan's
    included (the grid engine's own tests are tests/test_rollback_difference_form.py).
  * The pipeline: every compute format runs end to end under strict dtype promotion, the report
    names each format's execution, a second run compiles nothing, Bermudans and Americans compute
    in any format (refused below float32 until 2026-10-13), and a scenario that moves nothing has
    exactly zero P&L at every compute format.
"""
import dataclasses
import types

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import ORE
import pytest

from engine.instruments.swap import underlying_swap
from engine.market_data.curves import DiscountCurve, ZeroCurve, log_discount, log_discount_deviation, node_values
from engine.market_simulation.config import CamConfig, LgmConfig, build_cross_asset_model, simulate
from engine.market_simulation.paths import evolve_states, standardized_steps
from engine.market_simulation.scenario_market import ScenarioCurves, build_scenario_market
from engine.market_simulation.sobol import _build_bridge_matrix, apply_brownian_bridge, bridge_increments_matrix
from engine.models.cam import CrossAssetModel, EqComponent, FxComponent, IrComponent, path_moments, step_moments
from engine.models.lgm import as_sigma
from engine.precision import (
    FLOAT64, FORMAT_NAMES, FORMATS, Arithmetic, Precision, StagePrecision, add_level, deviation_from, level_terms,
    matmul, product_execution, product_precision,
)
from engine.pricing.legs import legs_cube, legs_npv, legs_of, path_fixings, path_schedule
from engine.risk.market import MarketRiskRequest, monte_carlo_scenarios, run_market_risk
from engine.risk.market.revaluation import revalue, revalue_trade
from engine.run import LgmSwaptionEngineConfig, PortfolioRequest, PricingConfig, RunConfig, price_portfolio
from tests import market_risk_support as mr
from tests.support import portfolio as shared
from tests.support.compiles import count_compiles

#: Every compute format below float32, with the float32 accumulator they all take.
BELOW = tuple(name for name, f in FORMATS.items() if f.below_float32)
#: The arrays formats below float32 (their products run on operands of their own dtype).
LOW_ARRAYS = ("float16", "bfloat16", "float8_e4m3fn", "float8_e5m2")
#: Every format whose product is one pass of its rounded operands, and each one's single pass:
#: their names and their bases (A-23).
ROUNDED = tuple(name for name, f in FORMATS.items() if f.below_float32 and f.product != "bfloat16_3x")
#: Each compute format's unit roundoff: half a spacing at 1 of its operands.
EPS = {name: 2.0 ** -(FORMATS[name].mantissa_bits + 1) for name in FORMAT_NAMES}
DATES = tuple(shared.ASOF + ORE.Period(m, ORE.Months) for m in (3, 12, 36))
LINEAR = ("swap-payer", "swap-receiver-seasoned-icma", "european-payer", "european-receiver-otm-cash", "bond")


def _arithmetic(name: str) -> Arithmetic:
    return Arithmetic(name, "float32" if FORMATS[name].below_float32 else name)


def _stage(name: str, storage: str = "float32") -> StagePrecision:
    return StagePrecision(storage if FORMATS[name].below_float32 else name, name, _arithmetic(name).accumulate)


def _simulation(samples=64, seed=3) -> CamConfig:
    return CamConfig(dates=DATES, base_currency="USD", ir={"USD": LgmConfig(0.03, 0.01, ("1Y", "2Y"), ("9Y", "8Y"))},
                     samples=samples, seed=seed)


# ---------------------------------------------------------------------------
# The products
# ---------------------------------------------------------------------------
def _reference_operand(x: np.ndarray, name: str, axis: int) -> np.ndarray:
    """`x` as a unit of format `name` (an array format, or TensorFloat-32) sees it, independently
    of the engine: scaled along `axis` by the power of two that brings each slice's largest
    magnitude to the format's top (none for bfloat16 and TensorFloat-32, whose range is
    float32's), rounded by ml_dtypes' correct conversion (TensorFloat-32: to a 10-bit mantissa),
    scaled back."""
    fmt = FORMATS[name]
    if name == "tensorfloat32":
        return _round_mantissa(x, fmt.mantissa_bits)
    if name == "bfloat16":
        return x.astype(fmt.dtype).astype(np.float64)
    amax = np.max(np.abs(x), axis=axis, keepdims=True)
    mantissa, exponent = np.frexp(amax)
    top_mantissa, top_exponent = np.frexp(fmt.max)
    k = np.where(amax > 0, top_exponent - exponent - (mantissa > top_mantissa), 0)
    return (x * 2.0 ** k).astype(fmt.dtype).astype(np.float64) / 2.0 ** k


def _reference_terms(x: np.ndarray, name: str, axis: int, count: int) -> list:
    """`x` as `count` terms of format `name` (`_reference_operand`), each the rounding of what the
    ones before leave."""
    terms = []
    for _ in range(count):
        terms.append(_reference_operand(x - sum(terms), name, axis))
    return terms


def _round_mantissa(x: np.ndarray, bits: int) -> np.ndarray:
    """`x` rounded to nearest (ties to even) with `bits` mantissa bits: TensorFloat-32's grid."""
    mantissa, exponent = np.frexp(x)
    return np.ldexp(np.round(np.ldexp(mantissa, bits + 1)), exponent - bits - 1)


def _bias_significance(error: np.ndarray) -> float:
    """The worst over the columns of `error` `[paths, ...]` of its mean over the paths in units of
    that mean's standard error."""
    return float(np.max(np.abs(error.mean(axis=0)) / (error.std(axis=0) / np.sqrt(error.shape[0]) + 1e-300)))


class TestProducts:
    RNG = np.random.default_rng(7)
    A = RNG.standard_normal((6, 40)) * np.logspace(-3, 3, 6)[:, None]     # rows of very different scales
    B = RNG.standard_normal((40, 5)) * np.logspace(4, -2, 5)[None, :]     # columns too

    def _bound(self, a, b):
        """A float32 accumulation's error bound on `a @ b`: n eps32 sum |a||b|."""
        return a.shape[-1] * 2.0 ** -24 * (np.abs(a) @ np.abs(b))

    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("name", ROUNDED)
    def test_a_format_multiplies_its_rounded_operands_the_shared_one_in_its_terms(self, name, side):
        """Each operand rounded to the format's grid with its scales, the shared one in two terms
        of it (one for a single pass), the products of the terms accumulated in float32 and
        summed there."""
        fmt = FORMATS[name]
        a_terms = _reference_terms(self.A, fmt.base, -1, fmt.shared_terms if side == "left" else 1)
        b_terms = _reference_terms(self.B, fmt.base, -2, fmt.shared_terms if side == "right" else 1)
        a, b = sum(a_terms), sum(b_terms)
        result = np.asarray(matmul(self.A, self.B, Arithmetic(name, "float32"), side), dtype=np.float64)
        bound = sum(self._bound(np.abs(x), np.abs(y)) for x in a_terms for y in b_terms) \
            + 2.0 ** -23 * (np.abs(a) @ np.abs(b))
        assert np.all(np.abs(result - a @ b) <= bound)

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_the_shared_operands_rounding_biases_a_single_pass_not_two_terms(self, side):
        """4,096 paths of a per-path operand whose mean is far from 0, the worst case: the shared
        operand's rounding then moves every path the same way. Its mean error over the paths, in
        units of its standard error, is that bias's significance (measured 2026-10-11: one pass 40
        to 150, two terms 1.5 for a 10-bit grid and 2.3 for bfloat16's). FP8's two terms leave 5
        (e4m3) and 8 (e5m2), a twelfth of one pass's or less; that rest is each path's own
        rounding on FP8's grid, not the shared operand's: a third term, or the shared operand
        unrounded, leaves the same (docs/planning/details/precision.md §15.5)."""
        rng = np.random.default_rng(11)
        paths = 1.0 + 0.3 * rng.standard_normal((4096, 12))
        weights = rng.standard_normal((12, 6)) * np.logspace(0, -2, 6)
        exact = paths @ weights
        for base in ("float16", "tensorfloat32", "bfloat16", "float8_e4m3fn", "float8_e5m2"):
            significance = {}
            for name in (base, f"{base}_1x"):
                arithmetic = Arithmetic(name, "float32")
                if side == "right":
                    result = matmul(jnp.asarray(paths, jnp.float32), weights, arithmetic, side)
                else:
                    result = matmul(weights.T, jnp.asarray(paths.T, jnp.float32), arithmetic, side).T
                significance[name] = _bias_significance(np.asarray(result, np.float64) - exact)
            assert significance[f"{base}_1x"] > 20, (base, significance)
            if FORMATS[base].mantissa_bits >= 7:
                assert significance[base] < 4, (base, significance)
            else:
                assert significance[base] < significance[f"{base}_1x"] / 8, (base, significance)

    def test_a_product_in_a_stages_arithmetic_names_its_shared_operand(self):
        for side in (None, "both"):
            with pytest.raises(ValueError, match=r"names the operand every path shares, shared='left' or 'right' "
                                                 r"\(decision A-23\)"):
                matmul(self.A, self.B, Arithmetic("float64", "float64"), side)

    def test_bfloat16_3x_is_three_products_of_the_bfloat16_pieces(self):
        def pieces(x):
            x = x.astype(np.float32)
            high = x.astype(ml_dtypes.bfloat16)
            low = (x - high.astype(np.float32)).astype(ml_dtypes.bfloat16)
            return high.astype(np.float64), low.astype(np.float64)

        (a_hi, a_lo), (b_hi, b_lo) = pieces(self.A), pieces(self.B)
        exact = a_hi @ b_hi + a_hi @ b_lo + a_lo @ b_hi
        result = np.asarray(matmul(self.A, self.B, Arithmetic("bfloat16_3x", "float32"), "right"), dtype=np.float64)
        assert np.all(np.abs(result - exact) <= 3 * self._bound(np.abs(a_hi), np.abs(b_hi)))
        # ~16 bits: far closer than one bfloat16 pass, as far as TensorFloat-32 or better
        relative = np.max(np.abs(result - self.A @ self.B) / (np.abs(self.A) @ np.abs(self.B)))
        assert relative < 2 ** -14

    @pytest.mark.parametrize("name", FORMAT_NAMES)
    def test_every_format_is_within_its_unit_roundoff_and_accumulates_where_it_is_told(self, name):
        arithmetic = _arithmetic(name)
        result = matmul(self.A, self.B, arithmetic, "right")
        assert result.dtype == arithmetic.dtype
        error = np.abs(np.asarray(result, np.float64) - self.A @ self.B)
        scale = np.abs(self.A) @ np.abs(self.B)
        assert np.all(error <= (2 * EPS[name] + 40 * 2.0 ** -24) * scale)

    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("name", ROUNDED)
    def test_a_paths_rounding_does_not_depend_on_the_other_paths(self, name, side):
        """Scales are per path (a row of the left operand, or a column of the right one when the
        left is shared) and per row or column of the shared operand's terms, never per tensor:
        rescaling every other path a millionfold leaves a path's product as it was, bit for bit (in
        a batch of the same shape; XLA may sum a product of another shape in another order, an ulp
        apart, as it does at float32)."""
        arithmetic = Arithmetic(name, "float32")

        def product(paths):
            """`paths @ B` `[paths, N]`, the paths on the side that is not shared."""
            if side == "right":
                return np.asarray(matmul(paths, self.B, arithmetic, side))
            return np.asarray(matmul(self.B.T, paths.T, arithmetic, side)).T

        full = product(self.A)
        for i in range(self.A.shape[0]):
            others = np.where(np.arange(self.A.shape[0])[:, None] == i, 1.0, 1e6) * self.A
            np.testing.assert_array_equal(product(others)[i], full[i])

    def test_vectors_batches_zeros_and_an_empty_contraction(self):
        arithmetic = Arithmetic("float8_e4m3fn", "float32")
        for side in ("left", "right"):
            assert matmul(self.A[0], self.B, arithmetic, side).shape == (5,)
            assert matmul(self.A, self.B[:, 0], arithmetic, side).shape == (6,)
            batched = matmul(np.stack([self.A, -self.A]), self.B, arithmetic, side)
            np.testing.assert_array_equal(np.asarray(batched[1]), -np.asarray(batched[0]))
            np.testing.assert_array_equal(np.asarray(matmul(np.zeros((3, 4)), np.ones((4, 2)), arithmetic, side)),
                                          0.0)
            np.testing.assert_array_equal(np.asarray(matmul(np.ones((3, 0)), np.ones((0, 2)), arithmetic, side)),
                                          0.0)

    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("name", LOW_ARRAYS + tuple(f"{name}_1x" for name in LOW_ARRAYS))
    def test_the_program_is_one_product_of_operands_of_the_format_into_float32(self, name, side):
        """Native, the format's own dtype; emulated FP8 (on every device until one's FP8 product
        is shown right), float16, which holds every FP8 value. The shared operand's terms are one
        product with it, as wide as its terms (twice the work for two terms, A-23)."""
        program = jax.make_jaxpr(lambda a, b: matmul(a, b, Arithmetic(name, "float32"), side))
        (dot,) = [e for e in program(self.A, self.B).eqns if e.primitive.name == "dot_general"]
        base = FORMATS[name].base
        carrier = jnp.float16 if base.startswith("float8") else jnp.dtype(base)
        assert [v.aval.dtype for v in dot.invars] == [jnp.dtype(carrier)] * 2
        terms = FORMATS[name].shared_terms
        assert [v.aval.shape for v in dot.invars] == ([(6 * terms, 40), (40, 5)] if side == "left"
                                                      else [(6, 40), (40, 5 * terms)])
        assert dot.params["preferred_element_type"] == jnp.float32
        assert dot.params["precision"] == (jax.lax.Precision.HIGHEST,) * 2

    def test_float32_operands_may_accumulate_in_float64(self):
        """float32's operands, float64's sums, on every device: up to the sums' order, far below
        float32's roundoff. Red first on an RTX 5060 (2026-10-11), with the product asked of
        the device as float32 operands with a float64 result: 7.6e-7 off, float32's sums."""
        a32, b32 = self.A.astype(np.float32).astype(np.float64), self.B.astype(np.float32).astype(np.float64)
        result = matmul(self.A, self.B, Arithmetic("float32", "float64"), "right")
        assert result.dtype == jnp.float64
        expected, scale = a32 @ b32, np.abs(a32) @ np.abs(b32)
        assert np.max(np.abs(np.asarray(result) - expected) / scale) <= 64 * 2.0 ** -53

    @pytest.mark.parametrize("platform, kind, capability, expected", [
        ("cpu", "cpu", None, {"float64": "native", "float32": "native"}),
        ("gpu", "NVIDIA GeForce RTX 5060", "12.0", {"float64": "native", "float32": "native", "float16": "native",
                                                     "bfloat16": "native", "tensorfloat32": "native"}),
        ("gpu", "Tesla V100", "7.0", {"float64": "native", "float32": "native", "float16": "native"}),
        ("gpu", "NVIDIA A100", "8.0", {"float64": "native", "float32": "native", "float16": "native",
                                        "bfloat16": "native", "tensorfloat32": "native"}),
        ("tpu", "TPU v5 lite", None, {"float32": "native", "bfloat16": "native", "bfloat16_3x": "native"}),
        ("tpu", "TPU7x", None, {"float32": "native", "bfloat16": "native", "bfloat16_3x": "native"}),
    ])
    def test_native_where_the_device_has_units_for_the_format_else_emulated(self, platform, kind, capability,
                                                                             expected):
        """FP8 nowhere, and bfloat16 passes not on a GPU, until a device's are shown right
        (`tests/support/precision_devices.py`: an RTX 5060's were not, 2026-10-10). A single pass
        runs on its format's units."""
        device = types.SimpleNamespace(platform=platform, device_kind=kind, compute_capability=capability)
        assert {name: product_execution(name, device) for name in FORMAT_NAMES} == {
            name: expected.get(name.removesuffix("_1x"), "emulated") for name in FORMAT_NAMES}

    def test_a_native_product_format_states_its_algorithm(self):
        ampere = types.SimpleNamespace(platform="gpu", device_kind="NVIDIA A100", compute_capability="8.0")
        tpu = types.SimpleNamespace(platform="tpu", device_kind="TPU v5 lite", compute_capability=None)
        assert product_precision("tensorfloat32", ampere) == product_precision("tensorfloat32_1x", ampere) \
            == jax.lax.DotAlgorithmPreset.TF32_TF32_F32
        assert product_precision("bfloat16_3x", tpu) == jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X3
        assert product_precision("bfloat16_3x", ampere) == jax.lax.Precision.HIGHEST
        cpu = jax.devices("cpu")[0]
        assert product_precision("tensorfloat32", cpu) == product_precision("bfloat16_3x", cpu) \
            == jax.lax.Precision.HIGHEST


class TestArithmetic:
    def test_the_default_is_float64_and_hashable(self):
        assert Arithmetic() == FLOAT64 == Arithmetic.of(jnp.float64) == Arithmetic("float64", "float64")
        assert hash(Arithmetic("bfloat16", "float32")) == hash(Arithmetic("bfloat16", "float32"))
        assert Arithmetic.of(jnp.float32) == Arithmetic("float32", "float32")

    @pytest.mark.parametrize("compute, accumulate, message", [
        ("bfloat16", "bfloat16", r"Arithmetic\.accumulate='bfloat16': narrower than float32"),
        ("float8_e5m2", "float64", r"Arithmetic\.accumulate='float64': compute='float8_e5m2' runs on matrix units"),
        ("float64", "float32", r"Arithmetic\.accumulate='float32': narrower than compute='float64'"),
        ("float32", "bfloat16_3x", r"Arithmetic\.accumulate='bfloat16_3x': bfloat16_3x is a matrix-product format"),
        ("fp8", "float32", r"Arithmetic\.compute: unknown number format 'fp8'"),
    ])
    def test_refusals_name_the_field(self, compute, accumulate, message):
        with pytest.raises(ValueError, match=message):
            Arithmetic(compute, accumulate)


class TestLevels:
    """A level meets a path in two terms of the path's format (`engine.precision.add_level`)."""
    LEVEL = np.array([1.0e6 + 1.0 / 3.0, -1.2345678901234567, 0.0])

    def test_the_two_terms_hold_the_level_and_vanish_at_its_own_format(self):
        high, low = level_terms(self.LEVEL, jnp.float32)
        assert high.dtype == low.dtype == jnp.float32
        np.testing.assert_array_equal(np.asarray(high), self.LEVEL.astype(np.float32))
        assert np.all(np.abs(np.asarray(high, np.float64) + np.asarray(low, np.float64) - self.LEVEL)
                      <= 2.0 ** -48 * np.abs(self.LEVEL))
        high, low = level_terms(self.LEVEL, jnp.float64)
        assert low is None and np.array_equal(np.asarray(high), self.LEVEL)

    def test_add_and_remove_are_inverse_and_float64_is_a_plain_sum(self):
        deviation = jnp.asarray(np.random.default_rng(0).standard_normal((64, 3)), jnp.float32)
        value = add_level(self.LEVEL, deviation)
        assert value.dtype == jnp.float32
        np.testing.assert_allclose(np.asarray(deviation_from(self.LEVEL, value)), np.asarray(deviation),
                                   atol=float(2.0 ** -23 * 1.0e6))
        deviation64 = deviation.astype(jnp.float64)
        np.testing.assert_array_equal(np.asarray(add_level(self.LEVEL, deviation64)),
                                      self.LEVEL + np.asarray(deviation64))

    def test_no_rounding_of_the_level_is_shared_by_every_path(self):
        """Red first: a level rounded once to float32 and added to every path moves their mean by
        the level's rounding, a bias; in two terms each path rounds by itself."""
        level = 1.0e6 + 0.0311
        deviation = np.random.default_rng(1).standard_normal(100_000) * 50.0
        exact = level + deviation
        two_terms = np.asarray(add_level(level, jnp.asarray(deviation, jnp.float32)), np.float64)
        deviation32 = np.asarray(jnp.asarray(deviation, jnp.float32), np.float64)
        rounded_once = (np.float32(level).astype(np.float64) + deviation32).astype(np.float32).astype(np.float64)
        noise = np.std(two_terms - exact) / np.sqrt(deviation.size)
        assert abs(np.mean(two_terms - exact)) < 4 * noise
        assert abs(np.mean(rounded_once - exact)) > 50 * noise


# ---------------------------------------------------------------------------
# The simulation: the bridge and the scan
# ---------------------------------------------------------------------------
def _two_currency_model() -> CrossAssetModel:
    """USD and EUR LGMs, EURUSD and an equity: components whose states have very different
    levels and scales (a log equity spot near 4.6 beside rate states near 0)."""
    usd, eur = (ZeroCurve(jnp.asarray([0.0, 1.0, 10.0, 30.0]), jnp.asarray(r)) for r in
                ([0.03, 0.032, 0.04, 0.045], [0.02, 0.021, 0.028, 0.03]))
    ir = (IrComponent("USD", usd, 0.03, as_sigma(0.01)), IrComponent("EUR", eur, 0.02, as_sigma(0.008)))
    correlation = np.array([[1.0, 0.5, 0.2, 0.3], [0.5, 1.0, -0.1, 0.1], [0.2, -0.1, 1.0, 0.25],
                            [0.3, 0.1, 0.25, 1.0]])
    return CrossAssetModel(ir=ir, fx=(FxComponent("EUR", 1.1, as_sigma(0.1)),),
                           eq=(EqComponent("SP5", "USD", 100.0, as_sigma(0.2), usd, usd),), correlation=correlation)


class TestSimulation:
    TIMES = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0])

    def test_the_path_moments_are_the_states_law(self):
        """The mean and variance carried through each step equal a large sample's; each scale is a
        power of two within a factor two of the standard deviation (1 where it is 0: at t = 0)."""
        model = _two_currency_model()
        moments = step_moments(model, self.TIMES)
        path = path_moments(model, moments)
        np.testing.assert_array_equal(path.mean[0], model.initial_state())
        assert np.all(path.scale[0] == 1.0)
        mantissa, _ = np.frexp(path.scale)
        assert np.all(mantissa == 0.5)
        normals = np.random.default_rng(1).standard_normal((len(self.TIMES) - 1, 200_000, model.dimension))
        states = np.asarray(evolve_states((path.mean, path.scale), standardized_steps(moments, path), normals, FLOAT64))
        np.testing.assert_allclose(states.mean(axis=0), path.mean[1:], atol=4e-3 * np.max(path.scale))
        deviation = states.std(axis=0)
        assert np.all((deviation / path.scale[1:] > 0.45) & (deviation / path.scale[1:] < 1.05))

    def test_float64_is_the_direct_recursion_to_rounding(self):
        """`x_{i+1} = M x_i + b + L z_i`, as the code before the rewrite ran it."""
        model = _two_currency_model()
        moments = step_moments(model, self.TIMES)
        path = path_moments(model, moments)
        normals = np.random.default_rng(2).standard_normal((len(self.TIMES) - 1, 64, model.dimension))
        states = np.asarray(evolve_states((path.mean, path.scale), standardized_steps(moments, path), normals, FLOAT64))
        x = np.broadcast_to(model.initial_state(), (64, model.dimension))
        for i in range(len(self.TIMES) - 1):
            x = x @ moments.transition[i].T + moments.drift[i] + normals[i] @ moments.cholesky[i].T
            np.testing.assert_allclose(states[:, i], x, rtol=0, atol=1e-14 * np.max(np.abs(x)))

    def test_the_bridge_is_one_product_of_the_differenced_matrix(self):
        """The bridged increments standardized by sqrt(dt), as the code before the rewrite formed
        them (the path values, differenced on the device, then scaled)."""
        times = self.TIMES
        z = np.random.default_rng(3).standard_normal((len(times) - 1, 32, 2))
        path = np.einsum("ij,jsd->isd", _build_bridge_matrix(times), z)
        direct = np.diff(np.concatenate([np.zeros((1, 32, 2)), path]), axis=0) / np.sqrt(np.diff(times))[:, None, None]
        np.testing.assert_allclose(np.asarray(apply_brownian_bridge(jnp.asarray(z), times)), direct, rtol=0, atol=1e-14)
        np.testing.assert_allclose(bridge_increments_matrix(times) @ z.reshape(len(times) - 1, -1),
                                   direct.reshape(len(times) - 1, -1), atol=1e-14)

    @pytest.mark.parametrize("name", BELOW)
    def test_below_float32_the_error_is_relative_to_the_deviation_not_the_level(self, name):
        """Each step rounds its operands once, so the error grows with the steps, in units of the
        state's scale. The equity's log spot sits near 4.6 with a deviation near its scale, 0.06
        to 0.5: the same scan on the states themselves (`x <- [x, z] @ [M, L]' + b`, operands
        rounded alike) errs on the level, several times more."""
        model = _two_currency_model()
        moments = step_moments(model, self.TIMES)
        path = path_moments(model, moments)
        normals = np.random.default_rng(4).standard_normal((len(self.TIMES) - 1, 256, model.dimension))
        args = ((path.mean, path.scale), standardized_steps(moments, path))
        arithmetic = _arithmetic(name)
        exact = np.asarray(evolve_states(*args, normals, FLOAT64))
        low = np.asarray(evolve_states(*args, jnp.asarray(normals, jnp.float32), arithmetic), np.float64)
        assert np.all(np.isfinite(low))
        error = np.max(np.abs(low - exact), axis=0)                                   # [T, d]
        assert np.all(error <= 4 * np.arange(1, len(self.TIMES))[:, None] * EPS[name] * path.scale[1:])
        x = jnp.broadcast_to(jnp.asarray(model.initial_state(), jnp.float32), (256, model.dimension))
        direct = []
        for i in range(len(self.TIMES) - 1):
            moments_i = np.concatenate([moments.transition[i].T, moments.cholesky[i].T])
            x = (arithmetic.matmul(jnp.concatenate([x, jnp.asarray(normals[i], jnp.float32)], axis=-1), moments_i,
                                   shared="right")
                 + jnp.asarray(moments.drift[i], jnp.float32))
            direct.append(np.asarray(x, np.float64))
        direct_error = np.max(np.abs(np.stack(direct, axis=1) - exact), axis=0)
        equity = model.eq_state(0)
        assert np.max(error[:, equity]) < 0.3 * np.max(direct_error[:, equity])


# ---------------------------------------------------------------------------
# The scenario market
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def float64_market():
    """The shared market's LGM simulation, float64, and its model."""
    model = build_cross_asset_model(shared.market(), _simulation())
    return model, simulate(shared.market(), _simulation(), model)


class TestScenarioMarket:
    def test_float64_is_the_models_formulas_and_the_level_its_mean_state(self, float64_market):
        """Every curve, the numeraire: the LGM's formulas on the states (`implied_log_discounts`,
        `lgm_numeraire`), as the code before the rewrite computed them; the level is the curve at
        the mean state (0 for the domestic LGM)."""
        from engine.market_simulation.scenario_market import implied_log_discounts, lgm_numeraire, tenor_times
        model, sm = float64_market
        z = sm.states[:, :, 0]
        taus = tenor_times(sm.dates, _simulation().curve_tenors)
        direct = implied_log_discounts(model.ir[0].curve, model.ir[0], sm.times, taus, z)
        np.testing.assert_allclose(np.asarray(sm.discount["USD"].log_discounts), np.asarray(direct), rtol=0,
                                   atol=1e-15)
        np.testing.assert_allclose(np.asarray(sm.numeraire), np.asarray(lgm_numeraire(model.ir[0], sm.times, z)),
                                   rtol=1e-15)
        level = implied_log_discounts(model.ir[0].curve, model.ir[0], sm.times, taus, jnp.zeros((1, len(DATES))))[0]
        np.testing.assert_allclose(np.asarray(sm.discount["USD"].level), np.asarray(level), rtol=0, atol=1e-15)

    @pytest.mark.parametrize("name", BELOW)
    def test_below_float32_the_curves_error_is_relative_to_their_deviation(self, name, float64_market):
        """A 30-year log discount factor near -1.2 deviates by a few hundredths: its error is the
        format's share of the deviation, not of the level, which a format below float32 cannot
        hold (FP8 e4m3 rounds -1.2 by up to 0.06)."""
        model, sm = float64_market
        low = build_scenario_market(model, shared.ASOF, DATES, sm.states, _simulation().curve_tenors, _index_curves(),
                                    _arithmetic(name))
        for curves, exact_curves in ((low.discount["USD"], sm.discount["USD"]),
                                     (low.index[shared.INDEX], sm.index[shared.INDEX])):
            exact = np.asarray(exact_curves.log_discounts)
            approx = np.asarray(curves.log_discounts, np.float64)
            deviation = np.max(np.abs(exact - np.asarray(exact_curves.level)), axis=0)        # [D, K+1]
            assert np.all(np.abs(approx - exact) <= 4 * EPS[name] * deviation + 1e-6)
        np.testing.assert_allclose(np.asarray(low.numeraire, np.float64), np.asarray(sm.numeraire),
                                   rtol=8 * EPS[name] + 1e-6)


def _index_curves():
    """The shared market's index curves as `build_scenario_market` takes them."""
    market = shared.market()
    return {name: ("USD", ZeroCurve.from_config(curve)) for name, curve in market.currency("USD").index_curves.items()}


# ---------------------------------------------------------------------------
# The legs, the Europeans
# ---------------------------------------------------------------------------
def _with_level(curves: ScenarioCurves, level) -> ScenarioCurves:
    return dataclasses.replace(curves, level=jnp.broadcast_to(jnp.asarray(level, curves.level.dtype),
                                                              curves.level.shape))


class TestCurveDeviation:
    @pytest.mark.parametrize("curve", [
        ZeroCurve(jnp.asarray([0.25, 1.0, 5.0, 10.0]), jnp.asarray([0.02, 0.025, 0.035, 0.04])),
        ZeroCurve(jnp.asarray([2.0]), jnp.asarray([0.03])),
        DiscountCurve(jnp.asarray([0.0, 0.5, 1.0, 5.0, 30.0]), jnp.asarray([0.0, -0.01, -0.025, -0.15, -1.2])),
    ])
    def test_a_log_discount_factor_is_linear_in_the_curves_values(self, curve):
        """Before the first node, between nodes, on them and past the last: the deviation of
        the values themselves from zero is the log discount factor, exactly, and the deviation
        of a sum is the sum of the deviations, path by path."""
        t = jnp.asarray([0.0, 0.1, 0.25, 0.7, 1.0, 3.3, 5.0, 10.0, 12.5, 40.0])
        values = node_values(curve)
        np.testing.assert_array_equal(np.asarray(log_discount_deviation(curve, values, t)),
                                      np.asarray(log_discount(curve, t)))
        paths = jnp.stack([values, 2.0 * values, -values])
        np.testing.assert_allclose(np.asarray(log_discount_deviation(curve, paths, t)),
                                   np.asarray(log_discount(curve, t))[None] * np.array([[1.0], [2.0], [-1.0]]),
                                   rtol=1e-15, atol=1e-16)


class TestLegs:
    @pytest.mark.parametrize("name", ["swap-payer", "swap-receiver-seasoned-icma"])
    def test_float64_at_any_level_is_the_direct_valuation_on_the_path_curves(self, name, float64_market):
        """The level is any curve: the model's, or none at all (0), the values equal ORE's
        arithmetic on each path's own curve (`legs_npv` on it) to rounding."""
        _, sm = float64_market
        cfg = shared.trades()[name]
        legs = legs_of(underlying_swap(cfg), cfg.payer, shared.ASOF, cfg.fixings)
        schedule = path_schedule(legs, shared.ASOF, sm.dates)
        disc, index = sm.discount["USD"], sm.index[shared.INDEX]
        fixings = path_fixings(6, shared.ASOF, sm.dates, sm.times, index)
        direct = []
        for j in range(len(sm.dates)):
            known = np.where(schedule.path_fixed[j], np.asarray(fixings.rates)[:, schedule.fixing_step],
                             np.nan_to_num(legs.history))
            direct.append(legs_npv(legs, disc.on_date(j), index.on_date(j), sm.times[j], schedule.alive_fixed[j],
                                   schedule.alive_float[j], schedule.projected[j], known))
        direct = np.stack([np.asarray(v) for v in direct], axis=1)
        scale = np.max(np.abs(direct))
        for level in (None, 0.0):
            d, i = (disc, index) if level is None else (_with_level(disc, level), _with_level(index, level))
            values = np.asarray(legs_cube(legs, schedule, sm.times, d, i, fixings))
            np.testing.assert_allclose(values, direct, rtol=0, atol=1e-13 * cfg.notional + 1e-12 * scale)

    def test_a_path_on_the_level_is_the_level_exactly(self, float64_market):
        _, sm = float64_market
        cfg = shared.trades()["swap-payer"]
        legs = legs_of(underlying_swap(cfg), cfg.payer, shared.ASOF, cfg.fixings)
        schedule = path_schedule(legs, shared.ASOF, sm.dates)
        flat = {k: dataclasses.replace(c, log_discounts=jnp.broadcast_to(c.level, c.log_discounts.shape))
                for k, c in {"disc": sm.discount["USD"], "index": sm.index[shared.INDEX]}.items()}
        fixings = path_fixings(6, shared.ASOF, sm.dates, sm.times, flat["index"])
        np.testing.assert_array_equal(np.asarray(fixings.deviation), 0.0)
        values = np.asarray(legs_cube(legs, schedule, sm.times, flat["disc"], flat["index"], fixings))
        np.testing.assert_array_equal(values, np.broadcast_to(values[:1], values.shape))

    @pytest.mark.parametrize("name", BELOW)
    def test_below_float32_a_swaps_error_is_relative_to_its_deviation_not_its_legs(self, name, float64_market):
        """A 10-year payer swap's legs are each near its notional, its value on a path near a
        tenth of it: the error is the format's share of the value's move from the level."""
        _, sm = float64_market
        cfg = shared.trades()["swap-payer"]
        legs = legs_of(underlying_swap(cfg), cfg.payer, shared.ASOF, cfg.fixings)
        schedule = path_schedule(legs, shared.ASOF, sm.dates)
        loaded = sm.map_arrays(lambda a: a.astype(jnp.float32))
        disc, index = loaded.discount["USD"], loaded.index[shared.INDEX]
        exact = np.asarray(legs_cube(legs, schedule, sm.times, sm.discount["USD"], sm.index[shared.INDEX],
                                     path_fixings(6, shared.ASOF, sm.dates, sm.times, sm.index[shared.INDEX])))
        arithmetic = _arithmetic(name)
        low = np.asarray(legs_cube(legs, schedule, sm.times, disc, index,
                                   path_fixings(6, shared.ASOF, sm.dates, sm.times, index), arithmetic), np.float64)
        moves = np.max(np.abs(exact - exact.mean(axis=0)), axis=0)                    # [D]
        assert np.all(np.abs(low - exact) <= 30 * EPS[name] * moves + 1e-6 * cfg.notional)


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
def _request(names=LINEAR, precision=Precision(), samples=64, **pricing) -> PortfolioRequest:
    return PortfolioRequest(market=shared.market(), trades=[shared.trades()[n] for n in names],
                            config=RunConfig(simulation=_simulation(samples), pricing=PricingConfig(**pricing),
                                             precision=precision))


@pytest.fixture(scope="module")
def float64_run():
    return price_portfolio(_request())


class TestPipeline:
    @pytest.mark.parametrize("name", BELOW)
    def test_every_stage_computes_in_the_format_under_strict_promotion(self, name, float64_run):
        """Simulation, market and pricing in the format, accumulating in float32: no dtype mix, the
        report names the format's execution, and every trade's mean value on every date is within
        a few of the format's unit roundoffs of its spread over the paths of float64's (measured
        2026-10-10: at most 1.5, 3.1 for bfloat16_3x), plus float32's rounding of the level, where
        the accumulator holds it."""
        stage = _stage(name)
        with jax.numpy_dtype_promotion("strict"):
            result = price_portfolio(_request(precision=Precision(simulation=stage, market=stage, pricing=stage)))
        assert result.precision.products == {name: product_execution(name)}
        cube, exact = np.asarray(result.npv_cube), np.asarray(float64_run.npv_cube)
        assert np.all(np.isfinite(cube))
        spread = np.std(exact, axis=0) + 1e-9
        mean_error = np.abs(cube.mean(axis=0) - exact.mean(axis=0))
        assert np.all(mean_error <= 8 * EPS[name] * spread + 4 * 2.0 ** -24 * np.abs(exact.mean(axis=0)))

    @pytest.mark.parametrize("name", ["float16", "bfloat16", "tensorfloat32"])
    def test_no_rounding_of_a_shared_operand_biases_the_paths(self, name):
        """I-96, red first: every stage in the format, accumulating and stored in float32, and
        each trade's value on each date over 2,048 paths against float64's on the same paths: the
        mean error in units of its standard error is the significance of a bias that every path
        shares, which more paths do not remove. Measured 2026-10-11: at most 1.4 with the shared
        operands in two terms (A-23), 14 to 24 with the single pass (`<name>_1x`, what the code
        before ran under the format's name). FP8 e4m3's two terms leave up to 9 on the bond, the
        paths' own rounding on FP8's grid, which a third term does not change
        (docs/planning/details/precision.md §15.5)."""
        exact = np.asarray(price_portfolio(_request(samples=2048)).npv_cube)

        def significance(compute):
            stage = StagePrecision("float32", compute, "float32")
            policy = Precision(simulation=stage, market=stage, pricing=stage)
            return _bias_significance(np.asarray(price_portfolio(_request(samples=2048, precision=policy)).npv_cube,
                                                 np.float64) - exact)

        assert significance(name) < 4
        assert significance(f"{name}_1x") > 6

    def test_float32_products_accumulating_in_float64(self, float64_run):
        stage = StagePrecision("float32", "float32", "float64")
        result = price_portfolio(_request(precision=Precision(simulation=stage, market=stage, pricing=stage)))
        exact = np.asarray(float64_run.npv_cube)
        np.testing.assert_allclose(np.asarray(result.npv_cube), exact, rtol=0, atol=1e-5 * np.max(np.abs(exact)))
        assert result.precision.products == {"float32": "native"}

    def test_the_report_names_every_compute_format_of_the_run(self):
        policy = Precision(simulation=_stage("bfloat16"), by_trade={"bond": _stage("float8_e4m3fn")},
                           by_product={"european_swaption": _stage("tensorfloat32")})
        result = price_portfolio(_request(precision=policy))
        assert set(result.precision.products) == {"bfloat16", "float64", "float8_e4m3fn", "tensorfloat32"}

    def test_a_second_run_compiles_nothing(self):
        stage = _stage("float8_e4m3fn")
        request = _request(precision=Precision(simulation=stage, market=stage, pricing=stage))
        price_portfolio(request)
        with count_compiles() as compiled:
            price_portfolio(request)
        assert sum(compiled.values()) == 0, dict(compiled)

    @pytest.mark.parametrize("name", ["bermudan-payer-physical", "american-payer"])
    def test_a_bermudan_or_american_computes_in_any_format(self, name):
        """Refused below float32, and with float32 products accumulating in float64, until the
        rollback was rewritten in difference form (2026-10-13): now any compute format, under
        strict dtype promotion, the report naming it; float32 products accumulated in float64
        within float32's rounding of float64's values."""
        grid = LgmSwaptionEngineConfig(n_per_std=8, std_devs=3.0)
        exact = np.asarray(price_portfolio(_request(names=(name,), bermudan=grid, american=grid)).npv_cube)
        with jax.numpy_dtype_promotion("strict"):
            low = price_portfolio(_request(names=(name,), precision=Precision(pricing=_stage("bfloat16")),
                                           bermudan=grid, american=grid))
        assert np.all(np.isfinite(np.asarray(low.npv_cube)))
        assert set(low.precision.products) == {"bfloat16", "float64"}
        wide = price_portfolio(_request(names=(name,), precision=Precision(pricing=StagePrecision("float32", "float32",
                                                                                                "float64")),
                                        bermudan=grid, american=grid))
        np.testing.assert_allclose(np.asarray(wide.npv_cube), exact, rtol=0, atol=1e-6 * np.max(np.abs(exact)))

    def test_a_bermudan_kept_wide_beside_low_precision_trades(self):
        policy = Precision(pricing=_stage("float8_e4m3fn"),
                           by_product={"bermudan_swaption": StagePrecision("float32", "float32", "float32")})
        names = ("swap-payer", "bermudan-payer-physical")
        result = price_portfolio(_request(names=names, precision=policy, samples=32,
                                          bermudan=LgmSwaptionEngineConfig(n_per_std=8, std_devs=3.0)))
        assert np.all(np.isfinite(np.asarray(result.npv_cube)))
        assert result.precision.products == {"float32": "native", "float64": "native",
                                             "float8_e4m3fn": product_execution("float8_e4m3fn")}

    def test_the_paired_sample_corrects_a_low_precision_run(self, float64_run):
        """Every path paired: the two-level EPE is the float64 run's own, the uncorrected one not."""
        stage = _stage("float8_e4m3fn")
        policy = Precision(simulation=stage, market=stage, pricing=stage, paired_fraction=1.0)
        result = price_portfolio(_request(precision=policy))
        estimate = result.precision.figures["netting_set/EPE"]
        exact = np.asarray(float64_run.exposure.epe)[1:]
        np.testing.assert_allclose(np.asarray(estimate.value), exact, rtol=1e-12, atol=1e-9)
        assert not np.allclose(np.asarray(estimate.uncorrected), exact, rtol=1e-6)


# ---------------------------------------------------------------------------
# Market risk
# ---------------------------------------------------------------------------
MR_TRADES = (mr.swap, mr.european, mr.bermudan, mr.bond)


class TestMarketRisk:
    @pytest.mark.parametrize("name", FORMAT_NAMES)
    @pytest.mark.parametrize("pricing", ["Bachelier", "Jamshidian"])
    def test_a_scenario_that_moves_nothing_has_exactly_zero_pnl(self, name, pricing):
        """By construction (the P&L is the deviation, an option's exactly 0 where nothing moved),
        not by two evaluations of one value agreeing."""
        engine = mr.PRICING if pricing == "Bachelier" else mr.JAMSHIDIAN
        moves = np.zeros((3, mr.factors().size))
        moves[1, :] = 1e-4                                                            # a scenario that does move
        for make in MR_TRADES:
            _, pnl = revalue_trade(make(), mr.market(), mr.factors(), moves, engine, arithmetic=_arithmetic(name))
            assert pnl.dtype == _arithmetic(name).dtype
            assert float(pnl[0]) == 0.0 and float(pnl[2]) == 0.0 and float(pnl[1]) != 0.0

    @pytest.mark.parametrize("pricing", ["Bachelier", "Jamshidian"])
    def test_float64_pnl_is_the_full_revaluation_to_rounding(self, pricing):
        """The deviation from the base equals `f(base + shift) - f(base)` (the code before the
        rewrite), without its cancellation."""
        from engine.risk.greeks.price_functions import trade_price_function
        from engine.risk.market.revaluation import factor_indices
        engine = mr.PRICING if pricing == "Bachelier" else mr.JAMSHIDIAN
        scenarios = monte_carlo_scenarios(mr.factors(), mr.covariance(), 10, 32, seed=5)
        trades = [make() for make in MR_TRADES]
        base, shocked = revalue(trades, mr.market(), mr.factors(), scenarios.shifts, engine)
        rates = np.asarray(mr.factors().base_rates())
        for k, cfg in enumerate(trades):
            fn = trade_price_function(cfg, mr.market(), engine)
            own = [mr.factors().slice_of(i) for i in factor_indices(cfg, mr.factors())]
            full = [float(fn.price(*[jnp.asarray((rates + move)[s]) for s in own])) for move in scenarios.shifts]
            np.testing.assert_allclose(np.asarray(shocked[:, k]), full, rtol=1e-12, atol=1e-8)

    @pytest.mark.parametrize("name", BELOW)
    def test_below_float32_the_pnl_is_near_float64s(self, name):
        """Within a few of the format's unit roundoffs of each trade's largest P&L (measured
        2026-10-10: at most 1.5; the Bermudan's, rounded at every step of its induction, 3.4 on
        2026-10-13): the P&L is the deviation itself, no level in sight."""
        trades = [make() for make in MR_TRADES]
        scenarios = monte_carlo_scenarios(mr.factors(), mr.covariance(), 10, 256, seed=5)
        exact = run_market_risk(MarketRiskRequest(trades, mr.market(), scenarios, mr.PRICING))
        low = run_market_risk(MarketRiskRequest(trades, mr.market(), scenarios, mr.PRICING,
                                                precision=Precision(pricing=_stage(name))))
        pnl, reference = np.asarray(low.pnl), np.asarray(exact.pnl)
        assert np.all(np.abs(pnl - reference) <= 4 * EPS[name] * np.max(np.abs(reference), axis=0))
        assert low.precision.products == {name: product_execution(name)}

