"""
The precision of a run (docs/planning/details/precision.md §3, §4; decisions A-10, A-12, D-9).

    Precision
      simulation   StagePrecision   the Sobol shocks and the model states
      market       StagePrecision   the scenario market: curves, numeraire, FX and equity
      pricing      StagePrecision   pricing on paths (and market-risk revaluation) and the
                                    values it stores (the cube, the scenario NPVs)

      by_product   {product: StagePrecision}    overrides `pricing` for every trade of a product
      by_trade     {trade_id: StagePrecision}   overrides `by_product` and `pricing` for one trade

      rounding       "nearest" | "stochastic"   how values are rounded into a scaled storage format
      rounding_seed  int                        the seed of the stochastic rounding
      offset         "block" | "none"           a scaled format's block layout: an offset beside
                                                each block scale, or none (I-75)
      paired_fraction  float in [0, 1]          the share of paths also run at float64 (A-13)

Each stage also has `compute_rounding`: nearest (the default research baseline) or
stochastic for single-grid compute below float32. Stochastic products round each per-path
operand with an independent keyed draw; the shared operand retains its configured
representation (`product.shared_terms`, one by default).
`rounding_seed` seeds both storage and compute rounding in separate named streams.

Each adjustable stage has three precisions: `storage`, the format its output is kept in until
the next stage reads it; `compute`, the format of its matrix products; `accumulate`, the format
those products accumulate in and every other operation of the stage is evaluated in (its
`arithmetic`, `engine.precision.Arithmetic`: the kernels are in difference form, so a product
multiplies a path's deviation from a level, not the level). `Precision()` is float64
everywhere, the engine's default and the one every ORE parity suite runs on.

The pricing stage is per trade (decision A-15): `precision_for(trade)` is the one lookup, and
returns `by_trade[trade.trade_id]`, else `by_product[trade.product]`, else `pricing`. The
simulation and the scenario market are shared by every trade, so they have no overrides.
`check_overrides` refuses a key that names no trade of the run or no product, so a misspelt
override is never silently ignored.

Storage below 32 bits (float16, bfloat16, FP8) is kept with block scales along
the scenario axis, laid out by `offset` and rounded by `rounding` (`engine.precision.storage`):
with `offset="block"`, the default, a block whose entries sit close together against their
level (all of one sign, the farthest from zero at most three times the nearest) stores their
deviations from its midpoint, an offset kept beside its scale, so it keeps its spread to the
format's precision (I-75); `"none"` stores the entries themselves, the layout of the OCP MX
formats. float64 and float32 storage is a plain cast whatever `rounding` and `offset` say.
`Precision.store` is the one way the pipeline stores: it names each array (`"shocks"`, `"values/<trade id>"`), and a stochastic
rounding draws from `rounding_seed` and that name, so arrays round independently, a trade's
column rounds the same in a mixed run as alone, and a run reproduces.

With `paired_fraction > 0`, that share of the paths (or market-risk scenarios), rounded up to
whole blocks within each independent scramble (`engine.precision.estimate.paired_count`), is also run at float64 throughout, and
the result's precision report gives each figure's distance from float64 on it: means are
corrected by the two-level estimator, quantiles measured (decision A-13). At float64 everywhere
it measures zero, and is allowed as a check of the pairing.

Calibration, t=0 values, levels, Greeks, reductions and output are declared in
`internal` (FP32 or FP64 for each role). `execution` requires a backend and
native products by default; software product paths require explicit research permission.

Validation refuses, naming the field, before any work: a name outside the format table, a
format not enabled yet (the format table names the item that enables it), a matrix-product
format (`tensorfloat32`, `bfloat16_3x`, a single pass `<name>_1x`) as `storage`, an `accumulate` the stage's arithmetic
refuses (`engine.precision.arithmetic`: narrower than float32 or than `compute`, or other than
float32 under a compute format below float32), a rounding outside
`ROUNDINGS`, a block layout outside `OFFSETS`, `stochastic` rounding or `offset="none"` when no
stage stores in a scaled format (it would change nothing), and a `paired_fraction` outside
[0, 1]. Every other combination may be run (D-9).
The 32/64 shape retired on 2026-10-01 (`PrecisionConfig`) is refused, not translated (A-12): `RETIRED_SHAPE` says what replaces it.
"""
import math
from dataclasses import asdict, dataclass, field, fields, replace
from numbers import Real
from typing import Iterable, Iterator, Mapping, Sequence

from engine.precision.arithmetic import Arithmetic, arithmetic_refusal
from engine.precision.formats import format_of
from engine.precision.execution import ExecutionPolicy, InternalPrecision, validate_arrays, validate_product
from engine.precision.storage import OFFSETS, ROUNDINGS, rounding_key, store
from engine.precision.estimate import validate_replicates
from engine.precision.storage import BLOCK
from engine.precision.options import ProductOptions

#: The adjustable stages, in pipeline order.
STAGES = ("simulation", "market", "pricing")

#: The overrides of the pricing stage and what they are keyed by, most general first
#: (`Precision.precision_for` reads them in reverse).
OVERRIDES = {"by_product": "product", "by_trade": "trade id"}

#: Why the 32/64 shape is refused, and what replaces it (Python and HTTP).
RETIRED_SHAPE = (
    "the 32/64 precision shape (PrecisionConfig with simulation/pricing/risk/calibration bits, "
    "PricingPrecisionOverride, RiskPrecisionOverride, MarketRiskRequest.precision=64|32) was retired on 2026-10-01 "
    "(decision A-12). Give an engine.precision.Precision: a StagePrecision(storage, compute, accumulate) of format "
    "names per stage (simulation, market, pricing), e.g. Precision.throughout('float32'), or over HTTP "
    "{\"simulation\": {\"storage\": \"float32\", \"compute\": \"float32\", \"accumulate\": \"float32\"}, ...}. "
    "Use profile=float32 for FP32 internal kernels; the default remains FP64")


@dataclass(frozen=True)
class StagePrecision:
    """The storage, compute and accumulate formats of one stage, by name (see the module
    docstring): `storage` an array format; `compute` any format of the
    table; `accumulate` float32 or float64, no narrower than `compute`, and float32 under a
    compute format below float32."""
    storage: str = "float64"
    compute: str = "float64"
    accumulate: str = "float64"
    compute_rounding: str = "nearest"
    product: ProductOptions = field(default_factory=ProductOptions)

    def __post_init__(self):
        rows = {}
        for f in fields(self):
            if f.name in ("compute_rounding", "product"):
                continue
            value = getattr(self, f.name)
            if not isinstance(value, str):
                raise TypeError(f"StagePrecision.{f.name} must be a format name such as 'float32', got {value!r}; "
                                f"{RETIRED_SHAPE}")
            try:
                rows[f.name] = format_of(value)
            except ValueError as exc:
                raise ValueError(f"StagePrecision.{f.name}: {exc}") from None
        storage, compute, accumulate = rows["storage"], rows["compute"], rows["accumulate"]
        if not storage.array:
            _refuse("storage", storage.name, "a matrix-product format, not an array format to store in")
        if storage.storage_pending:
            _refuse("storage", storage.name, f"storage in {storage.name} is not enabled yet "
                                             f"({storage.storage_pending})")
        refusal = arithmetic_refusal(compute, accumulate)
        if refusal is not None:
            _refuse(refusal[0], getattr(self, refusal[0]), refusal[1])
        # The arithmetic owns the validation shared by direct and configured kernels.
        try:
            Arithmetic(self.compute, self.accumulate, self.compute_rounding, product=self.product)
        except ValueError as exc:
            raise ValueError(f"StagePrecision arithmetic: {exc}") from None

    @property
    def storage_dtype(self):
        return format_of(self.storage).dtype

    @property
    def arithmetic(self) -> Arithmetic:
        """The stage's compute and accumulate formats, as its kernels take them."""
        return Arithmetic(self.compute, self.accumulate, self.compute_rounding, product=self.product)

    @classmethod
    def compensated(cls, storage="float64", compute="float64", accumulate="float64", compute_rounding="nearest"):
        return cls(storage, compute, accumulate, compute_rounding, ProductOptions.compensated(compute))

    @property
    def scaled_storage(self) -> bool:
        """Whether `storage` is a scaled format (stored with block scales and `rounding`)."""
        return format_of(self.storage).scaled


def _refuse(name: str, value: str, reason: str) -> None:
    raise ValueError(f"StagePrecision.{name}={value!r}: {reason}")


class Overrides(Mapping):
    """An immutable, hashable mapping of override keys (a product or a trade id) to the
    `StagePrecision` they select; `Precision` keeps its overrides in one so that it stays a
    frozen, hashable value."""

    def __init__(self, items: Mapping = ()):
        self._items = dict(items)

    def __getitem__(self, key):
        return self._items[key]

    def __iter__(self) -> Iterator:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return hash(frozenset(self._items.items()))

    def __repr__(self) -> str:
        return repr(self._items)


@dataclass(frozen=True)
class Precision:
    """Profile defaults, explicit stage overrides and internal numerical roles.

    Precision() preserves FP64; profile="float32" supplies FP32 defaults for every role.
    """
    simulation: StagePrecision = None
    market: StagePrecision = None
    pricing: StagePrecision = None
    by_product: Mapping[str, StagePrecision] = field(default_factory=Overrides)
    by_trade: Mapping[str, StagePrecision] = field(default_factory=Overrides)
    rounding: str = "nearest"
    rounding_seed: int = 0
    offset: str = "block"
    paired_fraction: float = 0.0
    storage_scaling: str = "power_of_two"
    internal: InternalPrecision = None
    execution: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    profile: str = "float64"

    def __post_init__(self):
        if self.profile not in ("float32", "float64"):
            raise ValueError("Precision.profile must be float32 or float64")
        for name in STAGES:
            if getattr(self, name) is None:
                object.__setattr__(self, name, StagePrecision(self.profile, self.profile, self.profile))
        if self.internal is None:
            object.__setattr__(self, "internal", InternalPrecision.throughout(self.profile))
        if self.storage_scaling not in ("power_of_two", "none"):
            raise ValueError("Precision.storage_scaling must be 'power_of_two' or 'none'")
        for name, expected in (("internal", InternalPrecision), ("execution", ExecutionPolicy)):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"Precision.{name} must be an {expected.__name__}")
        for stage in STAGES:
            value = getattr(self, stage)
            if not isinstance(value, StagePrecision):
                raise TypeError(f"Precision.{stage} must be a StagePrecision, got {value!r}; {RETIRED_SHAPE}")
        for name in OVERRIDES:
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"Precision.{name} must be a mapping of {OVERRIDES[name]} to StagePrecision, "
                                f"got {value!r}")
            for key, stage in value.items():
                if not isinstance(key, str) or not key:
                    raise TypeError(f"Precision.{name}: every key is a {OVERRIDES[name]} (a non-empty string), "
                                    f"got {key!r}")
                if not isinstance(stage, StagePrecision):
                    raise TypeError(f"Precision.{name}[{key!r}] must be a StagePrecision, got {stage!r}")
            object.__setattr__(self, name, Overrides(value))
        if self.rounding not in ROUNDINGS:
            raise ValueError(f"Precision.rounding={self.rounding!r}: the roundings are {list(ROUNDINGS)}")
        if self.offset not in OFFSETS:
            raise ValueError(f"Precision.offset={self.offset!r}: the block layouts are {list(OFFSETS)}")
        scaled = any(s.scaled_storage for s in self._stages())
        if scaled and self.storage_scaling == "none" and self.offset != "none":
            raise ValueError("plain low-precision storage requires offset='none'")
        if self.rounding == "stochastic" and not scaled:
            raise ValueError("Precision.rounding='stochastic' rounds values into a scaled storage format (float16, "
                             "bfloat16, FP8), and no stage or override stores in one")
        if self.offset == "none" and not scaled:
            raise ValueError("Precision.offset='none' lays out a scaled storage format's blocks (float16, bfloat16, "
                             "FP8), and no stage or override stores in one")
        if not isinstance(self.rounding_seed, int) or isinstance(self.rounding_seed, bool) or self.rounding_seed < 0:
            raise TypeError(f"Precision.rounding_seed must be a non-negative integer, got {self.rounding_seed!r}")
        fraction = self.paired_fraction
        if not isinstance(fraction, Real) or isinstance(fraction, bool) or not math.isfinite(fraction):
            raise TypeError(f"Precision.paired_fraction must be a number in [0, 1], got {fraction!r}")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"Precision.paired_fraction={fraction!r}: the share of paths also run at float64 is in "
                             f"[0, 1]")
        object.__setattr__(self, "paired_fraction", float(fraction))

    def _stages(self) -> Iterator[StagePrecision]:
        """Every `StagePrecision` of the policy, overrides included."""
        yield from (getattr(self, s) for s in STAGES)
        for name in OVERRIDES:
            yield from getattr(self, name).values()

    @classmethod
    def throughout(cls, name: str, *, execution: ExecutionPolicy = ExecutionPolicy(),
                   product: ProductOptions = ProductOptions()) -> "Precision":
        """Every adjustable stage computed in format `name`, stored and accumulated where it can be.
        Fixed internal kernels remain explicitly FP64; this is NOT an all-FP32 engine profile.
        A format below float32 accumulates in float32, a single pass (`<name>_1x`) is stored in
        the format it is of, and `tensorfloat32` and `bfloat16_3x` in float32."""
        fmt = format_of(name)
        storage = fmt.base if format_of(fmt.base).array else "float32"
        stage = StagePrecision(storage, name, "float32" if fmt.below_float32 else name, product=product)
        return cls(**{s: stage for s in STAGES}, execution=execution)

    @classmethod
    def reference(cls, *, backend: str = "default") -> "Precision":
        """FP64 reference profile, with native execution required."""
        return cls(execution=ExecutionPolicy(backend=backend))

    @classmethod
    def research(cls, **kwargs) -> "Precision":
        """Explicitly allow software/emitter-dependent products in a numerical research run.

        This permits emulation; it does not force it or certify hardware correctness.
        The profile and explicit internal roles determine the FP64 requirements.
        """
        kwargs.setdefault("execution", ExecutionPolicy(mode="allow_emulation"))
        return cls(**kwargs)

    def validate_execution(self, *, stages=STAGES, trades=None, arrays=(), device=None, internal_roles=None):
        """Refuse unsupported work before calibration, allocation or tracing.

        Only stages and internal roles selected by the entry point are checked.
        Missing numerical qualification is never an execution refusal.
        """
        selected = [(s, getattr(self, s)) for s in stages if s != "pricing"]
        if "pricing" in stages:
            selected += ([(f"pricing/{t.trade_id}", self.precision_for(t)) for t in trades]
                         if trades is not None else [("pricing", self.pricing),
                             *[(f"{name}/{key}", value) for name in OVERRIDES
                               for key, value in getattr(self, name).items()]])
        roles = tuple(item.name for item in fields(self.internal)) if internal_roles is None else internal_roles
        formats = [(f"internal.{name}", getattr(self.internal, name)) for name in roles]
        if self.paired_fraction:
            formats.append(("paired_reference", "float64"))
        formats += [(f"{name}.{role}", getattr(stage, role)) for name, stage in selected
                    for role in ("storage", "compute", "accumulate")]
        device = self.execution.validate(formats, device=device)
        validate_arrays(arrays, device)
        for stage in set(stage for _, stage in selected):
            validate_product(stage.compute, stage.accumulate, self.execution.mode, device=device, options=stage.product)
        return device

    def resolved(self, device=None):
        """JSON-ready complete contract, including internal roles and declared placement.

        This describes requirements and algorithms, not proof of emitted instructions.
        """
        from engine.precision.products import default_device
        device = device or default_device()
        result = {s: asdict(getattr(self, s)) for s in STAGES}
        result.update({name: {key: asdict(value) for key, value in getattr(self, name).items()}
                       for name in OVERRIDES})
        result.update(profile=self.profile, internal=asdict(self.internal), execution={**asdict(self.execution),
                      "backend": device.platform}, rounding=self.rounding, rounding_seed=self.rounding_seed,
                      offset=self.offset, paired_fraction=self.paired_fraction,
                      storage_scaling=self.storage_scaling,
                      host_preprocessing={"dtype": "float64", "backend": "cpu"},
                      paired_reference={"dtype": "float64", "backend": device.platform,
                                        "enabled": self.paired_fraction > 0})
        result["algorithms"] = {
            "float32_float64_products": "HIGHEST",
            "float32_compute_float64_accumulate": "FP32 grid represented in FP64; FP64 dot at HIGHEST",
            "tensorfloat32": "TF32_TF32_F32",
            "bfloat16_3x": "BF16_BF16_F32_X3",
            "other_low_products": "explicit stage product options; FP32 accumulation",
            "scalar_operations": "stage accumulate format (other kernels use internal precision)",
        }
        result["product_formats"] = {
            name: {"product": format_of(name).product,
                   "instruction_qualification": "not_established_by_dispatch"}
            for name in sorted({stage.compute for stage in self._stages()})}
        result['product_strategies'] = {name: stage.product.description(stage.compute)
                                       for name, stage in ((s, getattr(self, s)) for s in STAGES)}
        result['product_overrides'] = {name: {key: stage.product.description(stage.compute)
                                            for key, stage in getattr(self, name).items()} for name in OVERRIDES}
        result['storage_encoding'] = 'plain' if self.storage_scaling == 'none' else 'block_scaled_where_applicable'
        result['placement'] = {name: device.platform for name in (*STAGES,
                               *(f"internal.{item.name}" for item in fields(self.internal)))}
        result['qualification'] = {
            'numerical_evidence': 'see figure warnings; configuration does not certify accuracy',
            'instructions': 'unverified; dispatch is not an instruction trace',
            'affects_execution': False,
        }
        result['known_limitations'] = [
            note for enabled, note in (
                (any(s.compute.startswith('float8') for s in self._stages()),
                 'FP8 native instruction use is unverified and shape/compiler dependent; some tested routes widen.'),
                (any(s.compute.startswith('float16') and s.product.subnormals == 'native' for s in self._stages()),
                 'Native FP16 subnormal-sensitive products have known numerical failures in recorded device tests.'),
                (any(s.compute == 'bfloat16_3x' and s.product.bf16_padding == 'none' for s in self._stages()),
                 'Unpadded GPU BF16 three-pass vectors have known numerical failures; padding is opt-in.'),
            ) if enabled]
        return result

    def precision_for(self, trade) -> StagePrecision:
        """The pricing stage of `trade` (any object with `trade_id` and `product`):
        `by_trade`, else `by_product`, else `pricing` (decision A-15)."""
        if trade.trade_id in self.by_trade:
            return self.by_trade[trade.trade_id]
        return self.by_product.get(trade.product, self.pricing)

    def store(self, x, storage: str, stream: str, axis: int = 0):
        """`x` stored at format `storage` with this policy's rounding and block layout
        (`engine.precision.store`), its block scales along `axis`, the scenario axis. `stream`
        names the array in the run (see the module docstring): it seeds a stochastic rounding."""
        if self.rounding == "stochastic" and format_of(storage).scaled:
            return store(x, storage, "stochastic", rounding_key(self.rounding_seed, stream), axis, self.offset,
                         scaling=self.storage_scaling)
        return store(x, storage, axis=axis, offset=self.offset, scaling=self.storage_scaling)

    def arithmetic_for(self, stage: StagePrecision, stream: str) -> Arithmetic:
        """The stage's arithmetic with its own traced rounding stream (stage or trade)."""
        arithmetic = replace(stage.arithmetic, execution=self.execution.mode, levels=self.internal.levels,
                             calibration=self.internal.calibration)
        return (replace(arithmetic, key=rounding_key(self.rounding_seed, "compute/" + stream))
                if arithmetic.rounding == "stochastic" else arithmetic)

    def validate_sampling(self, paths: int, replicates: int, owner: str) -> None:
        """Replicate boundaries coincide with stored/paired block boundaries."""
        validate_replicates(paths, replicates, f"{owner}.replicates")
        if replicates > 1 and (self.paired_fraction or (self.storage_scaling != "none" and
                                                       any(s.scaled_storage for s in self._stages()))):
            if (paths // replicates) % BLOCK:
                raise ValueError(f"{owner}: sample count / replicates must be a multiple of storage block {BLOCK} "
                                 "for paired or scaled-storage runs")

    def check_overrides(self, trades: Iterable, products: Sequence[str]) -> None:
        """Refuse an override that selects nothing, before any work: a `by_product` key outside
        `products` (every product the pipeline prices), a `by_trade` key that is no trade's id."""
        unknown = sorted(set(self.by_product) - set(products))
        if unknown:
            raise ValueError(f"Precision.by_product: {unknown} not a product; the products are {list(products)}")
        unknown = sorted(set(self.by_trade) - {t.trade_id for t in trades})
        if unknown:
            raise ValueError(f"Precision.by_trade: {unknown} not the id of a trade in this run")


def require_precision(owner: str, value) -> None:
    """Refuse anything but a `Precision` as `owner`'s precision, naming the replacement of the
    retired 32/64 shape (A-12)."""
    if not isinstance(value, Precision):
        raise TypeError(f"{owner} must be an engine.precision.Precision, got {type(value).__name__} {value!r}; "
                        f"{RETIRED_SHAPE}")
