"""
`engine.precision` (roadmap 1.4, I-55; docs/planning/details/precision.md): the format table,
the `Precision` policy and its refusals, `store`/`load`, and the dtype discipline of the
pipeline built on them.

  * The table, the policy and storage need no market, trade or ORE (§13.2, §13.4).
  * Every adjustable stage's arrays come out in the formats the policy names (§13.3), read
    from the arrays, not the configuration.
  * Float32 runs end to end under strict dtype promotion, so no kernel mixes float32 and
    float64 behind the policy's back (§6.4). Before 1.4 the "float32" path priced in float64:
    NumPy float64 coupons and volatilities promoted the float32 curves.
  * A float32 recalibration at its bracket's top is flagged (§6.5): `hi * (1 - 1e-9)` rounds
    to `hi` in float32.
  * Float32 throughout on the sloped shared market: no NaN, the cube near float64 (§13.5).
"""
import dataclasses
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.api.schemas import PrecisionSchema, StagePrecisionSchema
from engine.calibration.ore_lgm import SIGMA_BRACKET, bootstrap_sigma, ceiling_tolerance
from engine.models.curves import ZeroCurve
from engine.portfolio import (
    CamConfig, HullWhiteConfig, JamshidianEngineConfig, LgmConfig, LgmSwaptionEngineConfig, PortfolioRequest,
    PricingConfig, RunConfig, price_portfolio,
)
from engine.precision import (
    FORMAT_NAMES, FORMATS, RETIRED_SHAPE, STAGES, Precision, StagePrecision, dtype_of, format_of, load, name_of,
    store,
)
from engine.simulation.config import build_cross_asset_model, simulate
from engine.valuation.bermudan import calibration_basket
from engine.valuation.portfolio import value_portfolio
from tests.support import portfolio as shared

ENABLED = ("float64", "float32")
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
        """precision.md §6.1: name, bits, mantissa bits, max, scaled, enabling steps."""
        rows = {f.name: (f.bits, f.mantissa_bits, f.scaled, f.storage_step, f.compute_step) for f in FORMATS.values()}
        assert rows == {
            "float64": (64, 52, False, None, None), "float32": (32, 23, False, None, None),
            "float16": (16, 10, True, "1.6", "2.8"), "bfloat16": (16, 7, True, "1.6", "2.8"),
            "float8_e4m3fn": (8, 3, True, "1.6", "2.8"), "float8_e5m2": (8, 2, True, "1.6", "2.8"),
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

    @pytest.mark.parametrize("storage, compute", [("float64", "float64"), ("float32", "float64"), ("float32", "float32")])
    def test_every_enabled_combination_is_accepted(self, storage, compute):
        stage = StagePrecision(storage, compute, compute)
        assert stage.storage_dtype == jnp.dtype(storage) and stage.compute_dtype == jnp.dtype(compute)

    @pytest.mark.parametrize("fields, message", [
        ({"storage": "float16"}, r"StagePrecision\.storage='float16'.*roadmap step 1\.6"),
        ({"storage": "float8_e4m3fn"}, r"StagePrecision\.storage='float8_e4m3fn'.*roadmap step 1\.6"),
        ({"compute": "bfloat16", "accumulate": "bfloat16"}, r"StagePrecision\.compute='bfloat16'.*roadmap step 2\.8"),
        ({"storage": "float64", "compute": "float32", "accumulate": "float32"},
         r"StagePrecision\.storage='float64': wider than compute"),
        ({"compute": "float64", "accumulate": "float32"}, r"StagePrecision\.accumulate='float32': narrower"),
        ({"compute": "float32", "accumulate": "float64", "storage": "float32"},
         r"StagePrecision\.accumulate='float64'.*roadmap step 2\.8"),
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

    def test_the_retired_names_are_gone(self):
        import engine.portfolio
        for name in ("PrecisionConfig", "PricingPrecisionOverride", "RiskPrecisionOverride"):
            assert not hasattr(engine.portfolio, name)


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

    @pytest.mark.parametrize("name", SCALED)
    def test_a_scaled_format_is_refused_until_its_step(self, name):
        with pytest.raises(ValueError, match="block scales.*roadmap step 1.6"):
            store(jnp.ones(3), name)


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
        policy = Precision(simulation=StagePrecision("float32", "float32", "float32"),
                           pricing=StagePrecision("float32"))
        wire = {s: dataclasses.asdict(getattr(policy, s)) for s in STAGES}
        assert PrecisionSchema.model_validate(wire).to_dataclass() == policy


# ---------------------------------------------------------------------------
# Dtype discipline of the pipeline
# ---------------------------------------------------------------------------
def _scenario_dtypes(scenarios):
    arrays = [scenarios.numeraire, scenarios.states, *scenarios.fx.values()]
    for curves in (*scenarios.discount.values(), *scenarios.index.values()):
        arrays += [curves.tenor_times, curves.log_discounts]
    return {a.dtype for a in arrays}


class TestRealizedDtypes:
    """Each stage's arrays are in the policy's formats, read from the arrays."""

    @pytest.mark.parametrize("market", [StagePrecision(), StagePrecision("float32"),
                                        StagePrecision("float32", "float32", "float32")])
    @pytest.mark.parametrize("simulation", [StagePrecision(), StagePrecision("float32", "float32", "float32")])
    def test_the_scenario_market_is_stored_at_the_market_storage(self, simulation, market):
        policy = Precision(simulation=simulation, market=market)
        scenarios = simulate(shared.market(), _simulation(), precision=policy)
        assert _scenario_dtypes(scenarios) == {market.storage_dtype}

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
    before 1.4 at the first leg kernel (float64 coupon arrays against float32 curves)."""

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
def test_float32_throughout_on_the_sloped_shared_market_is_sane():
    """§13.5: every trade of the shared portfolio (3% -> 5% curves) at float32 throughout:
    no NaN, today's values float64 and unchanged, the cube within float32 noise of float64."""
    trades = list(shared.trades().values())
    pricing = PricingConfig(bermudan=FAST, american=FAST)
    run = lambda precision: price_portfolio(PortfolioRequest(  # noqa: E731
        market=shared.market(), trades=trades,
        config=RunConfig(simulation=_simulation(samples=256), pricing=pricing, precision=precision)))
    exact, rounded = run(Precision()), run(Precision.throughout("float32"))
    assert rounded.base_npv_per_trade == exact.base_npv_per_trade
    cube, reference = np.asarray(rounded.npv_cube), np.asarray(exact.npv_cube)
    assert np.all(np.isfinite(cube))
    notional = np.array([getattr(t, "notional", getattr(t, "face_amount", None)) for t in trades])
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
