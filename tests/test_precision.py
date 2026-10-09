"""
`engine.precision` (I-55, F-07; docs/planning/details/precision.md): the
format table, the `Precision` policy and its refusals, `store`/`load`, the dtype discipline of
the pipeline built on them, the pricing stage per product and per trade, and storage below
32 bits.

  * The table, the policy and storage need no market, trade or ORE (§13.2, §13.4).
  * Every adjustable stage's arrays come out in the formats the policy names (§13.3), read
    from the arrays, not the configuration.
  * Float32 runs end to end under strict dtype promotion, so no kernel mixes float32 and
    float64 behind the policy's back (§6.4). Before 1.4 the "float32" path priced in float64:
    NumPy float64 coupons and volatilities promoted the float32 curves.
  * A float32 recalibration at its bracket's top is flagged (§6.5): `hi * (1 - 1e-9)` rounds
    to `hi` in float32.
  * Float32 throughout on the sloped shared market: no NaN, the cube near float64 (§13.5).
  * Per product and per trade (1.5, decision A-15): `precision_for` resolves trade over product
    over stage; a mixed run equals each trade run alone at its precision, column for column,
    in the portfolio cube and the market-risk P&L; an override naming nothing is refused.
  * Storage below 32 bits (1.6, §6.2, §13.4): block scales along the scenario axis, nearest
    and stochastic rounding, checked against ml_dtypes' correctly rounded conversion; every
    stage stored in every scaled format, on the sloped shared market, under strict promotion;
    each kernel compiled once per shape.
"""
import dataclasses
import pathlib
import pickle
import types
import typing

import jax
import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.api.requests import MarketTradeSchema
from engine.api.requests import PrecisionSchema, StagePrecisionSchema
from engine.calibration.ore_lgm import SIGMA_BRACKET, bootstrap_sigma, ceiling_tolerance
from engine.market_data.curves import ZeroCurve
from engine.market_simulation.config import build_cross_asset_model, simulate
from engine.precision import (
    BLOCK, FORMAT_NAMES, FORMATS, OVERRIDES, RETIRED_SHAPE, ROUNDINGS, STAGES, Overrides, Precision, StagePrecision,
    Stored, dtype_of, format_of, load, name_of, rounding_key, store,
)
from engine.precision import storage as storage_module
from engine.pricing.bermudan import calibration_basket
from engine.pricing.cube import PRODUCTS, Trade, value_portfolio
from engine.risk.market import MarketRiskRequest, monte_carlo_scenarios, run_market_risk
from engine.run import (
    CamConfig, HullWhiteConfig, JamshidianEngineConfig, LgmConfig, LgmSwaptionEngineConfig, PortfolioRequest,
    PricingConfig, RunConfig, price_portfolio,
)
from tests import market_risk_support as mr
from tests.support import portfolio as shared

ENABLED = ("float64", "float32")
F32 = StagePrecision("float32", "float32", "float32")
#: float64 arithmetic, float32 storage.
STORED32 = StagePrecision("float32", "float64", "float64")
SCALED = ("float16", "bfloat16", "float8_e4m3fn", "float8_e5m2")
FAST = LgmSwaptionEngineConfig(n_per_std=8, std_devs=3.0)
DATES = tuple(shared.ASOF + ORE.Period(m, ORE.Months) for m in (3, 12, 24))


def _simulation(model=LgmConfig, samples=16) -> CamConfig:
    return CamConfig(dates=DATES, base_currency="USD", ir={"USD": model(0.03, 0.01, ("1Y", "2Y"), ("9Y", "8Y"))},
                     samples=samples, seed=3)


# ---------------------------------------------------------------------------
# The format table
# ---------------------------------------------------------------------------
class TestFormats:
    def test_the_table_is_the_design_table(self):
        """precision.md §6.1: name, bits, mantissa bits, max, scaled, the items that enable them."""
        rows = {f.name: (f.bits, f.mantissa_bits, f.min_exponent, f.scaled, f.storage_pending, f.compute_pending)
                for f in FORMATS.values()}
        assert rows == {
            "float64": (64, 52, -1022, False, None, None), "float32": (32, 23, -126, False, None, None),
            "float16": (16, 10, -14, True, None, "F-07"), "bfloat16": (16, 7, -126, True, None, "F-07"),
            "float8_e4m3fn": (8, 3, -6, True, None, "F-07"), "float8_e5m2": (8, 2, -14, True, None, "F-07"),
        }
        assert FORMATS["float8_e4m3fn"].max == 448.0 and FORMATS["float16"].max == 65504.0

    @pytest.mark.parametrize("name", FORMAT_NAMES)
    def test_bits_are_the_dtypes_and_names_round_trip(self, name):
        assert dtype_of(name).itemsize * 8 == FORMATS[name].bits
        assert name_of(dtype_of(name)) == name

    def test_a_name_outside_the_table_is_refused_listing_the_table(self):
        with pytest.raises(ValueError, match=r"unknown number format 'fp8'.*float8_e4m3fn"):
            format_of("fp8")


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------
class TestPolicy:
    def test_the_default_is_float64_everywhere(self):
        assert Precision() == Precision.throughout("float64")
        for stage in STAGES:
            assert getattr(Precision(), stage) == StagePrecision("float64", "float64", "float64")

    def test_it_is_a_frozen_hashable_value(self):
        assert hash(Precision.throughout("float32")) == hash(Precision.throughout("float32"))
        with pytest.raises(dataclasses.FrozenInstanceError):
            Precision().pricing = StagePrecision("float32")

    @pytest.mark.parametrize("compute", ENABLED)
    @pytest.mark.parametrize("storage", FORMAT_NAMES)
    def test_every_storage_no_wider_than_compute_is_accepted(self, storage, compute):
        """Since 1.6 every format of the table stores, below float64 or float32 compute."""
        if FORMATS[storage].bits > FORMATS[compute].bits:
            pytest.skip("storage wider than compute is refused below")
        stage = StagePrecision(storage, compute, compute)
        assert stage.storage_dtype == jnp.dtype(storage) and stage.compute_dtype == jnp.dtype(compute)
        assert stage.scaled_storage == (storage in SCALED)

    def test_a_format_not_enabled_for_storage_is_refused_naming_the_item(self, monkeypatch):
        """The refusal FP4 will meet until F-07 enables it (no format of the table has one today)."""
        monkeypatch.setitem(FORMATS, "float16", dataclasses.replace(FORMATS["float16"], storage_pending="F-07"))
        with pytest.raises(ValueError, match=r"StagePrecision\.storage='float16': storage in float16 is not enabled "
                                             r"yet \(F-07\)"):
            StagePrecision("float16")

    @pytest.mark.parametrize("fields, message", [
        ({"compute": "bfloat16", "accumulate": "bfloat16"}, r"StagePrecision\.compute='bfloat16'.*not enabled yet \(F-07"),
        ({"storage": "float8_e4m3fn", "compute": "float16", "accumulate": "float16"},
         r"StagePrecision\.compute='float16'.*not enabled yet \(F-07"),
        ({"storage": "float64", "compute": "float32", "accumulate": "float32"},
         r"StagePrecision\.storage='float64': wider than compute"),
        ({"compute": "float64", "accumulate": "float32"}, r"StagePrecision\.accumulate='float32': narrower"),
        ({"compute": "float32", "accumulate": "float64", "storage": "float32"},
         r"StagePrecision\.accumulate='float64'.*not enabled yet \(F-07"),
        ({"compute": "float128"}, r"StagePrecision\.compute: unknown number format"),
    ])
    def test_refusals_name_the_field_and_the_step(self, fields, message):
        with pytest.raises(ValueError, match=message):
            StagePrecision(**fields)

    @pytest.mark.parametrize("build, owner", [
        (lambda: StagePrecision(32), r"StagePrecision\.storage must be a format name"),
        (lambda: Precision(simulation="float32"), r"Precision\.simulation must be a StagePrecision"),
        (lambda: RunConfig(precision={"simulation": 32}), r"RunConfig\.precision must be an engine\.precision"),
    ])
    def test_the_retired_32_64_shape_is_refused_naming_the_replacement(self, build, owner):
        """Decision A-12: refused, not translated; the message says what replaces it."""
        with pytest.raises(TypeError, match=owner) as raised:
            build()
        assert RETIRED_SHAPE in str(raised.value)

    def test_rounding_defaults_to_nearest(self):
        assert Precision().rounding == "nearest" and Precision().rounding_seed == 0
        assert ROUNDINGS == ("nearest", "stochastic")

    @pytest.mark.parametrize("fields, error, message", [
        ({"rounding": "up"}, ValueError, r"Precision\.rounding='up': the roundings are \['nearest', 'stochastic'\]"),
        ({"rounding": "stochastic"}, ValueError, r"rounding='stochastic'.*no stage or override stores in one"),
        ({"rounding": "stochastic", "pricing": STORED32}, ValueError, r"no stage or override stores in one"),
        ({"rounding_seed": -1}, TypeError, r"Precision\.rounding_seed must be a non-negative integer"),
        ({"rounding_seed": 1.5}, TypeError, r"Precision\.rounding_seed must be a non-negative integer"),
        ({"rounding_seed": True}, TypeError, r"Precision\.rounding_seed must be a non-negative integer"),
    ])
    def test_a_rounding_that_cannot_apply_is_refused(self, fields, error, message):
        """Stochastic rounding with nothing stored in a scaled format would be accepted and
        then ignored."""
        with pytest.raises(error, match=message):
            Precision(**fields)

    @pytest.mark.parametrize("fields", [
        {"market": StagePrecision("bfloat16")},
        {"by_product": {"swap": StagePrecision("float8_e5m2")}},
        {"by_trade": {"t": StagePrecision("float16", "float32", "float32")}},
    ])
    def test_stochastic_rounding_is_accepted_with_any_scaled_storage(self, fields):
        assert Precision(rounding="stochastic", **fields).rounding == "stochastic"

    def test_the_retired_names_are_gone(self):
        import engine.run
        for name in ("PrecisionConfig", "PricingPrecisionOverride", "RiskPrecisionOverride"):
            assert not hasattr(engine.run, name)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
class TestStorage:
    VALUES = np.array([0.0, -0.0, 1.0, -2.5, 1e-30, 3.0e38, np.pi, np.nan, np.inf], dtype=np.float64)

    @pytest.mark.parametrize("name", ENABLED)
    def test_storing_and_loading_at_the_arrays_own_dtype_is_the_identity(self, name):
        x = jnp.asarray(self.VALUES, dtype=dtype_of(name))
        for out in (store(x, name), load(x, dtype_of(name))):
            assert out.dtype == x.dtype
            np.testing.assert_array_equal(np.asarray(out), np.asarray(x))

    def test_float32_storage_rounds_to_nearest_and_loads_back_exactly(self):
        x = jnp.asarray(self.VALUES)
        stored = store(x, "float32")
        assert stored.dtype == jnp.float32
        np.testing.assert_array_equal(np.asarray(stored), self.VALUES.astype(np.float32))
        loaded = load(stored, jnp.float64)
        assert loaded.dtype == jnp.float64
        np.testing.assert_array_equal(np.asarray(loaded), self.VALUES.astype(np.float32).astype(np.float64))

    @pytest.mark.parametrize("name", ENABLED)
    def test_an_unscaled_format_ignores_the_rounding_and_carries_no_scales(self, name):
        """float64/float32 storage is a plain cast, rounded to nearest."""
        x = jnp.asarray(self.VALUES)
        stochastic = store(x, name, "stochastic", rounding_key(0, "x"))
        assert not isinstance(stochastic, Stored)
        np.testing.assert_array_equal(np.asarray(stochastic), np.asarray(store(x, name)))


# ---------------------------------------------------------------------------
# Storage below 32 bits (§6.2, §13.4)
# ---------------------------------------------------------------------------
def _format_values(name, n, rng, normal_only=False):
    """Finite values of the format from random bit patterns, as float64 (`normal_only`: no
    subnormals, which XLA's CPU flushes in float32)."""
    fmt = FORMATS[name]
    raw = rng.integers(0, 2 ** fmt.bits, n).astype({8: np.uint8, 16: np.uint16}[fmt.bits])
    with np.errstate(invalid="ignore"):  # NaN bit patterns
        values = raw.view(fmt.dtype).astype(np.float64)
    keep = np.isfinite(values) & (~normal_only | (values == 0) | (np.abs(values) >= 2.0 ** fmt.min_exponent))
    return values[keep]


def _wide_range(rng, shape):
    """Normals times a magnitude per column from 1e-6 to 1e6: blocks of very different size."""
    return rng.standard_normal(shape) * np.logspace(-6, 6, shape[-1])


def _per_entry_scales(stored):
    scales = np.asarray(stored.scales, np.float64)
    return np.repeat(scales, BLOCK, axis=stored.axis).take(range(stored.shape[stored.axis]), axis=stored.axis)


def _half_spacing(v, name):
    """Half the format's spacing at each value `v` of the scaled domain."""
    fmt = FORMATS[name]
    binade = np.frexp(v)[1] - 1
    return 0.5 * 2.0 ** (np.maximum(binade, fmt.min_exponent) - fmt.mantissa_bits)


class TestScaledStorage:
    """`store`/`load` in float16, bfloat16 and both FP8 formats: block scales along the
    scenario axis, nearest and stochastic rounding. No market, trade or ORE."""

    @pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
    @pytest.mark.parametrize("rounding", ROUNDINGS)
    @pytest.mark.parametrize("name", SCALED)
    def test_values_of_the_format_round_trip_exactly(self, name, rounding, dtype):
        """Any value the format holds, in any block, comes back bit for bit, with either
        rounding (in float32 without subnormals, which XLA's CPU flushes)."""
        rng = np.random.default_rng(1)
        values = _format_values(name, 4096, rng, normal_only=dtype == jnp.float32)[:BLOCK * 40]
        x = jnp.asarray(values.reshape(-1, 4), dtype)
        stored = store(x, name, rounding, rounding_key(0, "x"))
        assert isinstance(stored, Stored) and stored.dtype == FORMATS[name].dtype and stored.shape == x.shape
        for at in (jnp.float64, dtype):
            np.testing.assert_array_equal(np.asarray(load(stored, at)), np.asarray(x).astype(at))

    @pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
    @pytest.mark.parametrize("name", SCALED)
    def test_nearest_is_the_correctly_rounded_value_within_half_a_step(self, name, dtype):
        """Equal to ml_dtypes' conversion of the scaled values, so at most half the format's
        spacing times the block scale from the input. JAX's own float64 -> FP8 conversion
        rounds twice (through float32) and is not used."""
        rng = np.random.default_rng(2)
        x = jnp.asarray(_wide_range(rng, (300, 9)), dtype)
        stored = store(x, name)
        exact = np.asarray(x, np.float64)
        scales = _per_entry_scales(stored)
        reference = (exact / scales).astype(FORMATS[name].dtype).astype(np.float64) * scales
        loaded = np.asarray(load(stored, jnp.float64))
        np.testing.assert_array_equal(loaded, reference)
        assert np.all(np.abs(loaded - exact) <= _half_spacing(exact / scales, name) * scales)

    @pytest.mark.parametrize("name", SCALED)
    def test_each_scale_brings_its_blocks_largest_magnitude_to_the_formats_maximum(self, name):
        """A float32 power of two per block of `BLOCK` paths, the largest possible: the block's
        largest magnitude lands in (max / 2, max], unless the scale is at its bound (a normal
        float32: bfloat16's range is float32's). No block overflows, however wide the range of
        the whole array."""
        fmt, rng = FORMATS[name], np.random.default_rng(3)
        x = _wide_range(rng, (BLOCK * 5 + 7, 6))
        stored = store(jnp.asarray(x), name)
        scales = np.asarray(stored.scales)
        assert stored.scales.dtype == jnp.float32 and scales.shape == (6, 6)  # ceil(167 / 32) blocks
        assert np.all(np.frexp(scales)[0] == 0.5)
        padded = np.concatenate([np.abs(x), np.zeros((-x.shape[0] % BLOCK, 6))]).reshape(-1, BLOCK, 6)
        top = padded.max(axis=1) / scales.astype(np.float64)
        assert np.all(top <= fmt.max) and np.all((top > fmt.max / 2) | (scales == 2.0 ** -126))
        assert np.all(np.isfinite(np.asarray(load(stored, jnp.float64))))

    @pytest.mark.parametrize("name", SCALED)
    def test_stochastic_rounding_is_unbiased_where_nearest_is_not(self, name):
        """The same value on every path: nearest moves all of them the same way (a bias);
        stochastic rounding moves each to one of its two neighbours, the mean within three
        standard errors of the value."""
        x = jnp.full((BLOCK * 2048,), 1.1)
        nearest = np.asarray(load(store(x, name), jnp.float64))
        draws = np.asarray(load(store(x, name, "stochastic", rounding_key(4, "x")), jnp.float64))
        assert np.ptp(nearest) == 0 and nearest[0] != 1.1
        neighbours = np.unique(draws)
        assert len(neighbours) == 2 and neighbours[0] < 1.1 < neighbours[1]
        error = draws - 1.1
        assert abs(error.mean()) < 3 * error.std() / np.sqrt(error.size)
        assert abs(nearest[0] - 1.1) > 10 * error.std() / np.sqrt(error.size)

    @pytest.mark.parametrize("name", SCALED)
    def test_stochastic_rounding_errors_average_out_on_varied_values(self, name):
        rng = np.random.default_rng(5)
        x = jnp.asarray(rng.standard_normal((BLOCK * 512, 4)))
        stored = store(x, name, "stochastic", rounding_key(5, "x"))
        error = np.asarray(load(stored, jnp.float64)) - np.asarray(x)
        scales = _per_entry_scales(stored)
        assert np.all(np.abs(error) < 2 * _half_spacing(np.asarray(x) / scales, name) * scales)  # a neighbour
        assert np.all(np.abs(error.mean(axis=0)) < 3 * error.std(axis=0) / np.sqrt(error.shape[0]))

    def test_stochastic_rounding_repeats_for_its_key_and_differs_for_another(self):
        x = jnp.asarray(np.random.default_rng(6).standard_normal((100, 3)))
        a, b = (store(x, "float8_e4m3fn", "stochastic", rounding_key(1, "values/t")) for _ in range(2))
        np.testing.assert_array_equal(np.asarray(a.values, np.float32), np.asarray(b.values, np.float32))
        for key in (rounding_key(2, "values/t"), rounding_key(1, "values/u")):
            other = store(x, "float8_e4m3fn", "stochastic", key)
            assert not np.array_equal(np.asarray(other.values, np.float32), np.asarray(a.values, np.float32))

    @pytest.mark.parametrize("name", SCALED)
    def test_zeros_and_nan_pass_through_and_infinities_where_the_format_has_them(self, name):
        x = jnp.asarray([0.0, -0.0, np.nan, np.inf, -np.inf, 1.0, -3.0])
        for rounding in ROUNDINGS:
            loaded = np.asarray(load(store(x, name, rounding, rounding_key(0, "x")), jnp.float64))
            assert loaded[0] == 0.0 and np.signbit(loaded[1]) and np.isnan(loaded[2])
            if name == "float8_e4m3fn":  # no infinities in the format
                assert np.all(np.isnan(loaded[3:5]))
            else:
                assert loaded[3] == np.inf and loaded[4] == -np.inf
            np.testing.assert_array_equal(loaded[5:], [1.0, -3.0])

    def test_an_all_zero_block_stays_zero(self):
        stored = store(jnp.zeros((BLOCK + 3, 2)), "float8_e5m2")
        np.testing.assert_array_equal(np.asarray(load(stored, jnp.float64)), np.zeros((BLOCK + 3, 2)))

    @pytest.mark.parametrize("shape, axis", [((5,), 0), ((BLOCK + 1, 3), 0), ((4, 70, 2), 1), ((3, 2, 33), -1)])
    def test_any_scenario_count_and_axis_short_last_block(self, shape, axis):
        """Blocks run along `axis` (the shocks' scenario axis is 1); a count not divisible by
        `BLOCK` ends in a short block."""
        x = jnp.asarray(np.random.default_rng(7).standard_normal(shape))
        stored = store(x, "float16", axis=axis)
        expected = list(shape)
        expected[axis] = -(-shape[axis] // BLOCK)
        assert stored.shape == shape and stored.scales.shape == tuple(expected) and stored.axis == axis % len(shape)
        loaded = np.asarray(load(stored, jnp.float64))
        np.testing.assert_allclose(loaded, np.asarray(x), rtol=2.0 ** -11, atol=0)

    def test_blocks_are_independent_along_the_scenario_axis_only(self):
        """A block's scale depends on its own 32 paths: changing another block's paths, or the
        same paths on another date, changes nothing else."""
        rng = np.random.default_rng(8)
        x = rng.standard_normal((BLOCK * 3, 2))
        y = x.copy()
        y[BLOCK:2 * BLOCK, 1] *= 1000.0
        a, b = (np.asarray(load(store(jnp.asarray(v), "float8_e4m3fn"), jnp.float64)) for v in (x, y))
        mask = np.ones_like(x, bool)
        mask[BLOCK:2 * BLOCK, 1] = False
        np.testing.assert_array_equal(a[mask], b[mask])

    def test_fp8_holds_an_eighth_of_float64_plus_the_scales(self):
        stored = store(jnp.ones((BLOCK * 32, 10)), "float8_e4m3fn")
        assert stored.values.nbytes == BLOCK * 32 * 10 and stored.nbytes == BLOCK * 32 * 10 * (1 + 4 / BLOCK)

    def test_it_is_a_pytree_and_loads_inside_jit(self):
        stored = store(jnp.linspace(-1.0, 1.0, 64), "bfloat16")
        leaves, tree = jax.tree_util.tree_flatten(stored)
        assert len(leaves) == 2
        _assert_same_bits(jax.tree_util.tree_unflatten(tree, leaves), stored)
        np.testing.assert_array_equal(np.asarray(jax.jit(lambda s: load(s, jnp.float64))(stored)),
                                      np.asarray(load(stored, jnp.float64)))

    @pytest.mark.parametrize("call, message", [
        (lambda: store(jnp.ones(3), "float16", "up"), r"rounding 'up': the roundings are"),
        (lambda: store(jnp.ones(3), "float16", "stochastic"), r"stochastic rounding needs a key"),
        (lambda: store(jnp.float64(1.0), "float16"), r"needs a scenario axis"),
    ])
    def test_misuse_is_refused(self, call, message):
        with pytest.raises(ValueError, match=message):
            call()

    def test_each_kernel_compiles_once_per_shape(self):
        """New values, keys and formats of known shapes reuse the compiled programs."""
        rng = np.random.default_rng(9)
        arrays = [jnp.asarray(rng.standard_normal((50, 3))) for _ in range(3)]
        for i, x in enumerate(arrays[:1] * 2):
            load(store(x, "float16", "stochastic", rounding_key(i, "x")), jnp.float64)
        quantize, dequantize = storage_module._quantize._cache_size(), storage_module._dequantize._cache_size()
        for i, x in enumerate(arrays):
            load(store(x, "float16", "stochastic", rounding_key(i, "y")), jnp.float64)
        assert storage_module._quantize._cache_size() == quantize
        assert storage_module._dequantize._cache_size() == dequantize


# ---------------------------------------------------------------------------
# The HTTP schema
# ---------------------------------------------------------------------------
class TestSchemaCompleteness:
    def test_every_policy_field_is_in_the_schema(self):
        assert set(PrecisionSchema.model_fields) == {f.name for f in dataclasses.fields(Precision)}
        assert set(StagePrecisionSchema.model_fields) == {f.name for f in dataclasses.fields(StagePrecision)}

    @pytest.mark.parametrize("name", FORMAT_NAMES)
    def test_every_format_in_the_table_is_accepted_by_the_schema(self, name):
        """The schema takes every name; the policy then enables or refuses it by step."""
        StagePrecisionSchema(storage=name, compute=name, accumulate=name)

    def test_the_schema_round_trips_a_policy(self):
        policy = Precision(simulation=StagePrecision("float8_e4m3fn", "float32", "float32"),
                           pricing=StagePrecision("float32"), by_product={"bermudan_swaption": F32},
                           by_trade={"swap-1": StagePrecision("bfloat16")}, rounding="stochastic", rounding_seed=11)
        wire = {s: dataclasses.asdict(getattr(policy, s)) for s in STAGES}
        wire.update(rounding=policy.rounding, rounding_seed=policy.rounding_seed)
        wire.update({name: {k: dataclasses.asdict(v) for k, v in getattr(policy, name).items()} for name in OVERRIDES})
        assert PrecisionSchema.model_validate(wire).to_dataclass() == policy

    def test_the_products_are_the_trade_types_on_the_wire(self):
        """`by_product` is keyed by the name each trade carries, which is its HTTP `trade_type`."""
        schemas = typing.get_args(typing.get_args(MarketTradeSchema)[0])
        on_the_wire = {typing.get_type_hints(s.to_dataclass)["return"]: s.model_fields["trade_type"].default
                       for s in schemas}
        assert on_the_wire == {cls: cls.product for cls in typing.get_args(Trade)}
        assert set(PRODUCTS) == set(on_the_wire.values()) and len(PRODUCTS) == len(set(PRODUCTS))

    def test_an_override_on_the_wire_names_its_key_when_refused(self):
        with pytest.raises(ValueError, match=r"precision\.by_trade\['s'\]: StagePrecision\.compute='float16'"):
            PrecisionSchema.model_validate({"by_trade": {"s": {"compute": "float16"}}}).to_dataclass()

    @pytest.mark.parametrize("old", [{"simulation": 32}, {"pricing": 64}, {"risk": 32}])
    def test_the_retired_shape_is_refused_but_an_integer_seed_is_not(self, old):
        with pytest.raises(ValueError, match="retired on 2026-10-01"):
            PrecisionSchema.model_validate(old)
        assert PrecisionSchema.model_validate({"rounding_seed": 32}).to_dataclass() == Precision(rounding_seed=32)


# ---------------------------------------------------------------------------
# Dtype discipline of the pipeline
# ---------------------------------------------------------------------------
def _path_arrays(scenarios):
    """Every path array of a scenario market (the tenor grids are not path arrays)."""
    curves = (*scenarios.discount.values(), *scenarios.index.values())
    return [scenarios.numeraire, scenarios.states, *scenarios.fx.values(), *(c.log_discounts for c in curves)]


def _scenario_dtypes(scenarios):
    return {a.dtype for a in _path_arrays(scenarios)}


def _grid_dtypes(scenarios):
    return {c.tenor_times.dtype for c in (*scenarios.discount.values(), *scenarios.index.values())}


def _bits(stored):
    """A stored array's content, comparable bit for bit: its values (and scales, if scaled)."""
    if isinstance(stored, Stored):
        return stored.format, stored.axis, np.asarray(stored.values), np.asarray(stored.scales)
    return np.asarray(stored)


def _assert_same_bits(a, b, err_msg=""):
    a, b = _bits(a), _bits(b)
    assert type(a) is type(b), err_msg
    for x, y in (zip(a, b) if isinstance(a, tuple) else [(a, b)]):
        if isinstance(x, np.ndarray):
            assert x.dtype == y.dtype, err_msg
            np.testing.assert_array_equal(x, y, err_msg=err_msg)
        else:
            assert x == y, err_msg


class TestRealizedDtypes:
    """Each stage's arrays are in the policy's formats, read from the arrays."""

    @pytest.mark.parametrize("market", [StagePrecision(), StagePrecision("float32"),
                                        StagePrecision("float32", "float32", "float32")])
    @pytest.mark.parametrize("simulation", [StagePrecision(), StagePrecision("float32", "float32", "float32")])
    def test_the_scenario_market_is_stored_at_the_market_storage(self, simulation, market):
        """Its tenor grid, which has no scenario axis, stays at the market compute (since 2026-10-02:
        before, a float32 storage under float64 compute rounded it to float32)."""
        policy = Precision(simulation=simulation, market=market)
        scenarios = simulate(shared.market(), _simulation(), precision=policy)
        assert _scenario_dtypes(scenarios) == {market.storage_dtype}
        assert _grid_dtypes(scenarios) == {market.compute_dtype}

    @pytest.mark.parametrize("pricing", [StagePrecision(), StagePrecision("float32"),
                                         StagePrecision("float32", "float32", "float32")])
    def test_the_cube_is_stored_at_the_pricing_storage_and_today_is_float64(self, pricing):
        trades = [shared.trades()[n] for n in ("swap-payer", "european-payer", "bond")]
        scenarios = simulate(shared.market(), _simulation())
        valuation = value_portfolio(trades, shared.market(), scenarios, "USD", precision=Precision(pricing=pricing))
        assert valuation.cube.dtype == pricing.storage_dtype
        assert all(isinstance(v, float) for v in valuation.today)
        exact = value_portfolio(trades, shared.market(), scenarios, "USD")
        assert valuation.today == exact.today

    def test_the_result_cube_and_exposure_are_float64(self):
        request = PortfolioRequest(market=shared.market(), trades=[shared.trades()["swap-payer"]],
                                   config=RunConfig(simulation=_simulation(), precision=Precision.throughout("float32")))
        result = price_portfolio(request)
        assert result.npv_cube.dtype == jnp.float64 and result.exposure.epe.dtype == jnp.float64
        cube = np.asarray(result.npv_cube)
        np.testing.assert_array_equal(cube, cube.astype(np.float32).astype(np.float64))  # stored in float32


#: Every product, both models, both European engines (the American and the cash Bermudan
#: are the Bermudan's kernels; the cash European is refused by Jamshidian).
_STRICT_CASES = {
    "LGM-Bachelier": (LgmConfig, PricingConfig(bermudan=FAST, american=FAST),
                      ("swap-payer", "swap-receiver-seasoned-icma", "european-receiver-otm-cash",
                       "bermudan-payer-physical", "american-payer", "bond")),
    "HullWhite-Jamshidian": (HullWhiteConfig, PricingConfig(european="Jamshidian", bermudan=FAST, american=FAST,
                                                            jamshidian=JamshidianEngineConfig(0.03, 0.01)),
                             ("european-payer", "bermudan-payer-physical")),
}


class TestStrictPromotion:
    """`jax_numpy_dtype_promotion="strict"` turns any float32/float64 mix into an error: the
    float32 pipeline computes in float32 throughout (§6.4). Each case failed on the code
    before 2026-10-01 at the first leg kernel (float64 coupon arrays against float32 curves)."""

    @pytest.mark.parametrize("case", _STRICT_CASES)
    def test_float32_throughout_never_mixes_dtypes(self, case):
        model, pricing, names = _STRICT_CASES[case]
        request = PortfolioRequest(market=shared.market(), trades=[shared.trades()[n] for n in names],
                                   config=RunConfig(simulation=_simulation(model), pricing=pricing,
                                                    precision=Precision.throughout("float32")))
        with jax.numpy_dtype_promotion("strict"):
            result = price_portfolio(request)
        assert np.all(np.isfinite(np.asarray(result.npv_cube)))

    @pytest.mark.parametrize("stage", STAGES)
    def test_one_stage_in_float32_never_mixes_dtypes(self, stage):
        """Each boundary between a float32 and a float64 stage goes through `load`."""
        model, pricing, names = _STRICT_CASES["LGM-Bachelier"]
        policy = Precision(**{stage: StagePrecision("float32", "float32", "float32")})
        request = PortfolioRequest(market=shared.market(), trades=[shared.trades()[n] for n in names[:3]],
                                   config=RunConfig(simulation=_simulation(model), pricing=pricing, precision=policy))
        with jax.numpy_dtype_promotion("strict"):
            price_portfolio(request)


class TestRecalibrationInFloat32:
    """The per-path recalibration runs at the pricing compute dtype; its ceiling is
    resolvable there (§6.5)."""

    def _bootstrap(self, dtype, vol_scale=1.0):
        cfg = shared.trades()["bermudan-payer-physical"]
        engine = LgmSwaptionEngineConfig()
        basket = calibration_basket(cfg, engine, shared.ASOF, shared.ASOF)
        curve = lambda rates: ZeroCurve.from_config(shared.market().currency("USD").discount_curve, dtype=dtype) \
            if rates is None else ZeroCurve(jnp.asarray(shared.PILLARS, dtype), jnp.asarray(rates, dtype))  # noqa: E731
        disc, index = curve(None), curve(shared.FORWARDING)
        vols = np.full(len(basket), 0.009) * vol_scale
        return bootstrap_sigma(basket, disc, index, vols, engine.reversion)

    def test_float32_follows_the_curves_and_agrees_with_float64(self):
        exact, rounded = self._bootstrap(jnp.float64), self._bootstrap(jnp.float32)
        assert rounded.values.dtype == jnp.float32 and exact.values.dtype == jnp.float64
        np.testing.assert_allclose(np.asarray(rounded.values), np.asarray(exact.values), rtol=1e-4)
        assert not np.any(np.asarray(rounded.hit_ceiling))

    @pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
    def test_an_unattainable_volatility_is_flagged_at_either_precision(self, dtype):
        """Red first: with the fixed 1e-9, a float32 bucket stuck one step below the bracket's
        top was not flagged, since `hi * (1 - 1e-9)` is `hi` in float32."""
        result = self._bootstrap(dtype, vol_scale=100.0)
        assert np.all(np.asarray(result.hit_ceiling))
        assert np.all(np.asarray(result.values) >= SIGMA_BRACKET[1] * (1.0 - ceiling_tolerance(dtype)))

    def test_the_tolerance_is_unchanged_in_float64(self):
        assert ceiling_tolerance(jnp.float64) == 1e-9
        assert np.float32(SIGMA_BRACKET[1]) * np.float32(1.0 - ceiling_tolerance(jnp.float32)) < np.float32(
            SIGMA_BRACKET[1])


@pytest.mark.slow
def test_float32_throughout_on_the_sloped_shared_market_is_sane(shared_float64):
    """§13.5: every trade of the shared portfolio (3% -> 5% curves) at float32 throughout:
    no NaN, today's values float64 and unchanged, the cube within float32 noise of float64."""
    exact, notional = shared_float64
    rounded = _shared_run(Precision.throughout("float32"))
    assert rounded.base_npv_per_trade == exact.base_npv_per_trade
    cube, reference = np.asarray(rounded.npv_cube), np.asarray(exact.npv_cube)
    assert np.all(np.isfinite(cube))
    assert np.max(np.abs(cube - reference) / notional) < 1e-5


def test_the_precision_package_imports_nothing_from_the_pipeline():
    """precision.md §2.4: `engine.precision` depends on JAX and NumPy only."""
    import ast
    offenders = []
    for path in sorted((pathlib.Path(__file__).resolve().parents[1] / "engine" / "precision").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules = ([a.name for a in node.names] if isinstance(node, ast.Import)
                       else [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 else [])
            offenders += [f"{path.name} imports {m}" for m in modules
                          if m.startswith("engine") and not m.startswith("engine.precision")]
    assert not offenders, offenders


# ---------------------------------------------------------------------------
# Per product and per trade (decision A-15)
# ---------------------------------------------------------------------------
def _trade(trade_id, product):
    return types.SimpleNamespace(trade_id=trade_id, product=product)


class TestOverrides:
    """`Precision.by_product`, `by_trade` and `precision_for`, the one lookup (§4.2)."""
    POLICY = Precision(pricing=STORED32, by_product={"bermudan_swaption": F32}, by_trade={"b2": StagePrecision()})

    def test_the_default_overrides_nothing(self):
        assert Precision().by_product == {} and Precision().by_trade == {}
        assert Precision().precision_for(_trade("x", "swap")) == StagePrecision()

    @pytest.mark.parametrize("trade, expected", [
        (_trade("s1", "swap"), STORED32),                       # no override: the pricing stage
        (_trade("b1", "bermudan_swaption"), F32),               # its product's
        (_trade("b2", "bermudan_swaption"), StagePrecision()),  # its own, over its product's
    ])
    def test_trade_over_product_over_stage(self, trade, expected):
        assert self.POLICY.precision_for(trade) == expected

    def test_it_stays_a_frozen_hashable_picklable_value(self):
        """A dict given is kept as an immutable `Overrides`, equal to that dict."""
        same = Precision(pricing=STORED32, by_product={"bermudan_swaption": F32}, by_trade={"b2": StagePrecision()})
        assert same == self.POLICY and hash(same) == hash(self.POLICY)
        assert isinstance(self.POLICY.by_product, Overrides) and self.POLICY.by_product == {"bermudan_swaption": F32}
        assert pickle.loads(pickle.dumps(self.POLICY)) == self.POLICY
        with pytest.raises(TypeError):
            self.POLICY.by_trade["b3"] = F32
        assert self.POLICY != dataclasses.replace(self.POLICY, by_trade={})

    def test_the_mapping_given_is_copied(self):
        given = {"s1": F32}
        policy = Precision(by_trade=given)
        given["s2"] = F32
        assert set(policy.by_trade) == {"s1"}

    @pytest.mark.parametrize("fields, message", [
        ({"by_product": {"swap": "float32"}}, r"Precision\.by_product\['swap'\] must be a StagePrecision"),
        ({"by_trade": {"": F32}}, r"Precision\.by_trade: every key is a trade id"),
        ({"by_trade": {7: F32}}, r"Precision\.by_trade: every key is a trade id"),
        ({"by_product": ["swap"]}, r"Precision\.by_product must be a mapping of product"),
    ])
    def test_malformed_overrides_are_refused_naming_the_field(self, fields, message):
        with pytest.raises(TypeError, match=message):
            Precision(**fields)

    def test_an_override_naming_no_product_or_trade_is_refused(self):
        trades = [_trade("s1", "swap")]
        Precision(by_product={p: F32 for p in PRODUCTS}, by_trade={"s1": F32}).check_overrides(trades, PRODUCTS)
        with pytest.raises(ValueError, match=r"Precision\.by_product: \['bermudan'\] not a product; the products"):
            Precision(by_product={"bermudan": F32}).check_overrides(trades, PRODUCTS)
        with pytest.raises(ValueError, match=r"Precision\.by_trade: \['s2'\] not the id of a trade in this run"):
            Precision(by_trade={"s2": F32}).check_overrides(trades, PRODUCTS)


#: The shared portfolio's trades the per-trade tests price: one of each kind of kernel.
_MIXED_TRADES = ("swap-payer", "european-payer", "bermudan-payer-physical", "bond")

#: Mixed policies: a product and a trade overridden, narrower and wider than `pricing`, on a
#: float64 and a float32 scenario market.
_MIXED = {
    "bermudans-f32": Precision(by_product={"bermudan_swaption": F32}, by_trade={"swap-payer": STORED32}),
    "f32-but-two": Precision(simulation=F32, market=F32, pricing=F32, by_product={"european_swaption": STORED32},
                             by_trade={"bond": StagePrecision()}),
    # 1.6: scaled formats with stochastic rounding, drawn per trade id, so each column rounds
    # as it does alone.
    "fp8-stochastic": Precision(market=StagePrecision("bfloat16"), pricing=StagePrecision("float8_e4m3fn"),
                                by_product={"bermudan_swaption": StagePrecision("float16", "float32", "float32")},
                                by_trade={"bond": STORED32}, rounding="stochastic", rounding_seed=5),
}


def _alone(policy, stage):
    """The policy that prices one trade alone at `stage`: no overrides, the rest kept (the
    shared stages, the rounding and its seed)."""
    return dataclasses.replace(policy, pricing=stage, by_product={}, by_trade={})


@pytest.fixture(scope="module")
def mixed_scenarios():
    return {name: simulate(shared.market(), _simulation(), precision=policy) for name, policy in _MIXED.items()}


class TestPerTradePortfolio:
    """Cast point 4 per trade: each cube column is priced at its trade's compute dtype and
    stored at its storage format (precision.md §12, an exit criterion of per-trade precision, decision A-15)."""

    @pytest.mark.parametrize("case", _MIXED)
    def test_a_mixed_run_equals_each_trade_alone_at_its_precision(self, case, mixed_scenarios):
        policy, scenarios, pricing = _MIXED[case], mixed_scenarios[case], PricingConfig(bermudan=FAST)
        trades = [shared.trades()[n] for n in _MIXED_TRADES]
        mixed = value_portfolio(trades, shared.market(), scenarios, "USD", pricing, precision=policy)
        for i, cfg in enumerate(trades):
            stage = policy.precision_for(cfg)
            alone = value_portfolio([cfg], shared.market(), scenarios, "USD", pricing, precision=_alone(policy, stage))
            assert mixed.columns[i].dtype == stage.storage_dtype, cfg.trade_id
            _assert_same_bits(mixed.columns[i], alone.columns[0], err_msg=cfg.trade_id)
            assert mixed.today[i] == alone.today[0]
        assert mixed.cube.dtype == jnp.float64  # formats differ: loaded at float64, every value as stored
        for i, column in enumerate(mixed.columns):
            np.testing.assert_array_equal(np.asarray(mixed.cube[..., i]), np.asarray(load(column, jnp.float64)))

    def test_overrides_equal_to_the_stage_change_no_bit(self, mixed_scenarios):
        """Overriding every product and trade with the pricing stage itself is the run without
        overrides."""
        trades = [shared.trades()[n] for n in _MIXED_TRADES]
        run = lambda policy: value_portfolio(trades, shared.market(), mixed_scenarios["bermudans-f32"],  # noqa: E731
                                             "USD", PricingConfig(bermudan=FAST), precision=policy)
        plain = run(Precision())
        overridden = run(Precision(by_product={p: StagePrecision() for p in PRODUCTS},
                                   by_trade={t.trade_id: StagePrecision() for t in trades}))
        assert plain.cube.dtype == overridden.cube.dtype == jnp.float64
        np.testing.assert_array_equal(np.asarray(plain.cube), np.asarray(overridden.cube))

    def test_a_mixed_run_never_mixes_dtypes(self):
        """Under strict promotion, end to end: per-trade loads and stores, the float64 result."""
        model, pricing, names = _STRICT_CASES["LGM-Bachelier"]
        policy = Precision(simulation=F32, market=F32, pricing=F32, by_product={"bermudan_swaption": StagePrecision()},
                           by_trade={"swap-payer": STORED32})
        request = PortfolioRequest(market=shared.market(), trades=[shared.trades()[n] for n in names],
                                   config=RunConfig(simulation=_simulation(model), pricing=pricing, precision=policy))
        with jax.numpy_dtype_promotion("strict"):
            result = price_portfolio(request)
        assert result.npv_cube.dtype == jnp.float64 and np.all(np.isfinite(np.asarray(result.npv_cube)))

    @pytest.mark.parametrize("policy, message", [
        (Precision(by_trade={"swap-payr": F32}), r"by_trade: \['swap-payr'\] not the id of a trade"),
        (Precision(by_product={"swaption": F32}), r"by_product: \['swaption'\] not a product"),
    ])
    def test_a_misspelt_override_is_refused_before_any_work(self, policy, message, monkeypatch):
        import engine.run.pipeline as pipeline
        monkeypatch.setattr(pipeline, "build_cross_asset_model", lambda *a: pytest.fail("work was done"))
        request = PortfolioRequest(market=shared.market(), trades=[shared.trades()["swap-payer"]],
                                   config=RunConfig(simulation=_simulation(), precision=policy))
        with pytest.raises(ValueError, match=message):
            price_portfolio(request)


class TestPerTradeMarketRisk:
    """Market risk on the same resolver: each trade revalued at its compute dtype, its P&L
    stored at its storage format."""

    @staticmethod
    def _scenarios(num):
        return monte_carlo_scenarios(mr.factors(), mr.covariance(), 10, num, seed=3)

    @pytest.mark.parametrize("policy", [
        Precision(by_product={"european_swaption": F32}, by_trade={"bond": STORED32}),
        Precision(simulation=F32, pricing=F32, by_trade={"swap": StagePrecision()}),
        Precision(simulation=StagePrecision("float8_e5m2"), pricing=StagePrecision("bfloat16", "float32", "float32"),
                  by_trade={"swap": StagePrecision("float8_e4m3fn")}, rounding="stochastic", rounding_seed=2),
    ], ids=["europeans-f32", "f32-but-one", "scaled-stochastic"])
    def test_a_mixed_run_equals_each_trade_alone_at_its_precision(self, policy):
        scenarios, trades = self._scenarios(64), [mr.swap(), mr.european(), mr.bermudan(), mr.bond()]
        mixed = run_market_risk(MarketRiskRequest(trades, mr.market(), scenarios, mr.PRICING, precision=policy))
        assert mixed.pnl.dtype == jnp.float64
        for i, cfg in enumerate(trades):
            stage = policy.precision_for(cfg)
            alone = run_market_risk(MarketRiskRequest([cfg], mr.market(), scenarios, mr.PRICING,
                                                      precision=_alone(policy, stage)))
            pnl = np.asarray(mixed.pnl[:, i])
            np.testing.assert_array_equal(pnl, np.asarray(alone.pnl[:, 0]), err_msg=cfg.trade_id)
            if not stage.scaled_storage:
                np.testing.assert_array_equal(pnl, pnl.astype(stage.storage_dtype).astype(np.float64))  # stored there
            assert mixed.base_npv_per_trade[i] == alone.base_npv_per_trade[0]

    def test_an_override_naming_no_trade_is_refused(self):
        with pytest.raises(ValueError, match=r"by_trade: \['bermudan-1'\] not the id of a trade"):
            run_market_risk(MarketRiskRequest([mr.swap()], mr.market(), self._scenarios(16),
                                              precision=Precision(by_trade={"bermudan-1": F32})))

    def test_repeated_trade_ids_are_refused(self):
        """Overrides and per-trade results are keyed by id, so ids are unique, as in a portfolio."""
        with pytest.raises(ValueError, match=r"trade ids must be unique.*\['swap'\]"):
            run_market_risk(MarketRiskRequest([mr.swap(), mr.swap()], mr.market(), self._scenarios(16)))


# ---------------------------------------------------------------------------
# Storage below 32 bits in the pipelines
# ---------------------------------------------------------------------------
#: Trades with a cheap kernel each: legs, a European, a bond.
_CHEAP_TRADES = ("swap-payer", "european-payer", "bond")


def _block_max(column):
    """Each entry's block's largest magnitude (blocks of `BLOCK` paths, axis 0)."""
    column = np.abs(np.asarray(column, np.float64))
    padded = np.concatenate([column, np.zeros((-column.shape[0] % BLOCK,) + column.shape[1:])])
    top = padded.reshape((-1, BLOCK) + column.shape[1:]).max(axis=1)
    return np.repeat(top, BLOCK, axis=0)[:column.shape[0]]


def _step_bound(stored, exact, rounding):
    """How far a stored column may be from the values it stored: half the format's spacing at
    its block's largest magnitude to nearest, a whole spacing stochastically (precision.md §6.2)."""
    return (0.5 if rounding == "nearest" else 1.0) * 2.0 ** -FORMATS[stored.format].mantissa_bits * _block_max(exact)


class TestScaledStoragePipeline:
    """Every adjustable stage stored in float16, bfloat16 and FP8 (cast points 1 to 4)."""

    @pytest.mark.parametrize("name", SCALED)
    def test_every_stage_stores_in_the_format(self, name):
        """Read from the arrays: the market's path arrays and the cube columns are `Stored` in
        the format; the tenor grid stays at the market compute; the cube loads at float64."""
        stage = StagePrecision(name, "float32", "float32")
        policy = Precision(simulation=stage, market=stage, pricing=stage)
        trades = [shared.trades()[n] for n in _CHEAP_TRADES]
        scenarios = simulate(shared.market(), _simulation(), precision=policy)
        assert all(isinstance(a, Stored) and a.dtype == stage.storage_dtype for a in _path_arrays(scenarios))
        assert _grid_dtypes(scenarios) == {jnp.dtype(jnp.float32)}
        valuation = value_portfolio(trades, shared.market(), scenarios, "USD", precision=policy)
        assert all(isinstance(c, Stored) and c.dtype == stage.storage_dtype for c in valuation.columns)
        assert valuation.cube.dtype == jnp.float64 and np.all(np.isfinite(np.asarray(valuation.cube)))
        assert valuation.today == value_portfolio(trades, shared.market(), scenarios, "USD").today

    @pytest.mark.parametrize("name", SCALED)
    def test_every_stage_scaled_never_mixes_dtypes(self, name):
        """End to end under strict promotion, every product's kernel, the exposure loading the
        stored numeraire (cast point 5)."""
        model, pricing, names = _STRICT_CASES["LGM-Bachelier"]
        stage = StagePrecision(name, "float32", "float32")
        policy = Precision(simulation=stage, market=stage, pricing=stage, rounding="stochastic")
        request = PortfolioRequest(market=shared.market(), trades=[shared.trades()[n] for n in names],
                                   config=RunConfig(simulation=_simulation(model), pricing=pricing, precision=policy))
        with jax.numpy_dtype_promotion("strict"):
            result = price_portfolio(request)
        assert result.npv_cube.dtype == jnp.float64 and np.all(np.isfinite(np.asarray(result.npv_cube)))
        assert np.all(np.isfinite(np.asarray(result.exposure.epe)))

    @pytest.mark.filterwarnings("ignore:The balance properties of Sobol")  # 48 paths: a short last block
    @pytest.mark.parametrize("rounding", ROUNDINGS)
    @pytest.mark.parametrize("name", SCALED)
    def test_a_stored_cube_column_is_within_a_step_of_its_float64_values(self, name, rounding):
        """Pricing storage alone (float64 compute): each column differs from the float64 run's
        only by its storage, within the format's step at its block's largest value."""
        trades = [shared.trades()[n] for n in _CHEAP_TRADES]
        scenarios = simulate(shared.market(), _simulation(samples=48))
        exact = value_portfolio(trades, shared.market(), scenarios, "USD")
        stored = value_portfolio(trades, shared.market(), scenarios, "USD",
                                 precision=Precision(pricing=StagePrecision(name), rounding=rounding))
        for column, values in zip(stored.columns, exact.columns):
            error = np.abs(np.asarray(load(column, jnp.float64)) - np.asarray(values))
            assert np.all(error <= _step_bound(column, values, rounding))

    def test_stochastic_rounding_reproduces_and_moves_with_its_seed(self):
        """A run repeats bit for bit (market and cube), compiles no storage kernel again, and
        another `rounding_seed` rounds differently."""
        fp8 = StagePrecision("float8_e4m3fn", "float32", "float32")
        trades = [shared.trades()[n] for n in _CHEAP_TRADES]

        def run(seed):
            policy = Precision(market=fp8, pricing=fp8, rounding="stochastic", rounding_seed=seed)
            scenarios = simulate(shared.market(), _simulation(), precision=policy)
            return scenarios, value_portfolio(trades, shared.market(), scenarios, "USD", precision=policy)

        first = run(1)
        compiled = storage_module._quantize._cache_size(), storage_module._dequantize._cache_size()
        again, other = run(1), run(2)
        assert (storage_module._quantize._cache_size(), storage_module._dequantize._cache_size()) == compiled
        for a, b in zip(_path_arrays(first[0]) + first[1].columns, _path_arrays(again[0]) + again[1].columns):
            _assert_same_bits(a, b)
        assert not np.array_equal(np.asarray(first[1].cube), np.asarray(other[1].cube))

    @pytest.mark.parametrize("name", SCALED)
    def test_market_risk_pnl_is_stored_in_the_format_and_a_zero_shift_is_zero(self, name):
        """The shifts and each P&L stored scaled; the P&L within a step of the float64 P&L
        (pricing storage alone); a scenario that moves nothing still has exactly zero P&L."""
        scenarios = TestPerTradeMarketRisk._scenarios(64)
        shifts = np.array(scenarios.shifts)
        shifts[5] = 0.0
        scenarios = dataclasses.replace(scenarios, shifts=shifts)
        trades = [mr.swap(), mr.european(), mr.bond()]
        exact = run_market_risk(MarketRiskRequest(trades, mr.market(), scenarios, mr.PRICING))
        stored = run_market_risk(MarketRiskRequest(trades, mr.market(), scenarios, mr.PRICING,
                                                   precision=Precision(pricing=StagePrecision(name))))
        pnl, reference = np.asarray(stored.pnl), np.asarray(exact.pnl)
        bound = 0.5 * 2.0 ** -FORMATS[name].mantissa_bits * _block_max(reference)
        assert np.all(np.abs(pnl - reference) <= bound) and not np.array_equal(pnl, reference)
        np.testing.assert_array_equal(pnl[5], 0.0)
        assert stored.base_npv_per_trade == exact.base_npv_per_trade
        everything = Precision.throughout("float32")
        scaled = dataclasses.replace(everything, simulation=StagePrecision(name, "float32", "float32"),
                                     pricing=StagePrecision(name, "float32", "float32"), rounding="stochastic")
        with jax.numpy_dtype_promotion("strict"):
            result = run_market_risk(MarketRiskRequest(trades, mr.market(), scenarios, mr.PRICING, precision=scaled))
        assert np.all(np.isfinite(np.asarray(result.pnl))) and np.all(np.asarray(result.pnl[5]) == 0.0)


#: The slow sanity bounds of scaled storage (nearest) on the shared portfolio, per format and
#: stage: the largest cube error per unit notional, and the largest bias (the mean over 256
#: paths) per unit notional; about four times the values measured on 2026-10-02 (precision.md
#: §15.3). Regression guards, not acceptance (that is I-55's evidence table). The largest errors are exercise
#: decisions that flip on a path (a cash Bermudan worth 0 or 1.7% of its notional).
_SCALED_BOUNDS = {
    ("float16", "simulation"): (3e-4, 3e-6), ("float16", "market"): (4e-4, 1.2e-5),
    ("float16", "pricing"): (2e-3, 4e-5),
    ("bfloat16", "simulation"): (2.5e-3, 1.1e-5), ("bfloat16", "market"): (7e-2, 2.7e-4),
    ("bfloat16", "pricing"): (1.4e-2, 4.3e-4),
    ("float8_e4m3fn", "simulation"): (7e-2, 3.3e-4), ("float8_e4m3fn", "market"): (7e-2, 1.7e-3),
    ("float8_e4m3fn", "pricing"): (1.3e-1, 2.7e-2),
    ("float8_e5m2", "simulation"): (6.2e-2, 1.25e-3), ("float8_e5m2", "market"): (1e-1, 4.6e-3),
    ("float8_e5m2", "pricing"): (2.6e-1, 1.5e-1),
}


def _shared_run(precision):
    """The whole shared portfolio on 256 paths at `precision`."""
    return price_portfolio(PortfolioRequest(
        market=shared.market(), trades=list(shared.trades().values()),
        config=RunConfig(simulation=_simulation(samples=256), pricing=PricingConfig(bermudan=FAST, american=FAST),
                         precision=precision)))


@pytest.fixture(scope="module")
def shared_float64():
    """`_shared_run` at float64, and each trade's notional."""
    notional = np.array([getattr(t, "notional", getattr(t, "face_amount", None)) for t in shared.trades().values()])
    return _shared_run(Precision()), notional


@pytest.mark.slow
@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("name", SCALED)
def test_scaled_storage_on_the_sloped_shared_market_is_sane(name, stage, shared_float64):
    """§13.5 for storage below 32 bits: every trade of the shared portfolio (3% -> 5% curves),
    one stage stored in the format (the others float64): no NaN, today's values unchanged,
    the cube's error and its bias over the paths within the measured bounds."""
    exact, notional = shared_float64
    compute = "float64" if stage == "pricing" else "float32"
    stored = _shared_run(Precision(**{stage: StagePrecision(name, compute, compute)}))
    assert stored.base_npv_per_trade == exact.base_npv_per_trade
    cube, reference = np.asarray(stored.npv_cube), np.asarray(exact.npv_cube)
    assert np.all(np.isfinite(cube))
    error, bias = _SCALED_BOUNDS[name, stage]
    assert np.max(np.abs(cube - reference) / notional) < error
    assert np.max(np.abs(np.mean(cube - reference, axis=0)) / notional) < bias


@pytest.mark.slow
@pytest.mark.parametrize("name", ["float8_e4m3fn", "float8_e5m2"])
def test_stochastic_rounding_removes_the_bias_of_a_concentrated_fp8_cube(name, shared_float64):
    """§15.3: a column whose paths sit close together against their level (the bond's, about
    1e6 with a spread of 1e4) rounds the same way on every path of a block to nearest, a bias
    that more paths do not remove; rounded stochastically, the error of its path mean is noise
    (I-75). Measured on 2026-10-02: e4m3 6.8e-3 of notional to nearest, 2.7e-3 stochastically;
    e5m2 3.7e-2 and 8.5e-3."""
    exact, notional = shared_float64
    reference = np.asarray(exact.npv_cube)
    bias = {rounding: np.max(np.abs(np.mean(np.asarray(_shared_run(Precision(
        pricing=StagePrecision(name), rounding=rounding)).npv_cube) - reference, axis=0)) / notional)
        for rounding in ROUNDINGS}
    assert bias["stochastic"] < bias["nearest"] / 2
