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
      paired_fraction  float in [0, 1]          the share of paths also run at float64 (1.7)

Each adjustable stage has three precisions: `storage`, the format its output is kept in until
the next stage reads it; `compute`, the format its arithmetic is done in; `accumulate`, the
format its sums accumulate in. `Precision()` is float64 everywhere, the engine's default and
the one every ORE parity suite runs on.

The pricing stage is per trade (decision A-15): `precision_for(trade)` is the one lookup, and
returns `by_trade[trade.trade_id]`, else `by_product[trade.product]`, else `pricing`. The
simulation and the scenario market are shared by every trade, so they have no overrides.
`check_overrides` refuses a key that names no trade of the run or no product, so a misspelt
override is never silently ignored.

Storage below 32 bits (float16, bfloat16, FP8, roadmap 1.6) is kept with block scales along
the scenario axis and rounded by `rounding` (`engine.precision.storage`); float64 and float32
storage rounds to nearest whatever `rounding` says. `Precision.store` is the one way the
pipeline stores: it names each array (`"shocks"`, `"values/<trade id>"`), and a stochastic
rounding draws from `rounding_seed` and that name, so arrays round independently, a trade's
column rounds the same in a mixed run as alone, and a run reproduces.

With `paired_fraction > 0`, that share of the paths (or market-risk scenarios), rounded up to
whole blocks (`engine.precision.estimate.paired_paths`), is also run at float64 throughout, and
the result's precision report gives each figure's distance from float64 on it: means are
corrected by the two-level estimator, quantiles measured (decision A-13). At float64 everywhere
it measures zero, and is allowed as a check of the pairing.

Calibration, t=0 values, Greeks and every reduction over paths or scenarios (exposure, VaR/ES)
are not stages here: they are float64 by decision (A-10).

Validation refuses, naming the field, before any work: a name outside the format table, a
format used before the roadmap step that enables it, `storage` wider than `compute` (storing
wider gains nothing), `accumulate` narrower than `compute`, a rounding outside
`ROUNDINGS`, `stochastic` rounding when no stage stores in a scaled format (it would round
nothing), and a `paired_fraction` outside [0, 1]. Every other combination may be run (D-9).
The 32/64 shape before roadmap 1.4 (`PrecisionConfig`) is refused, not translated (A-12): `RETIRED_SHAPE` says what replaces it.
"""
import math
from dataclasses import dataclass, field, fields
from numbers import Real
from typing import Iterable, Iterator, Mapping, Sequence

from engine.precision.formats import format_of
from engine.precision.storage import ROUNDINGS, rounding_key, store

#: The adjustable stages, in pipeline order.
STAGES = ("simulation", "market", "pricing")

#: The overrides of the pricing stage and what they are keyed by, most general first
#: (`Precision.precision_for` reads them in reverse).
OVERRIDES = {"by_product": "product", "by_trade": "trade id"}

#: Why the 32/64 shape is refused, and what replaces it (Python and HTTP).
RETIRED_SHAPE = (
    "the 32/64 precision shape (PrecisionConfig with simulation/pricing/risk/calibration bits, "
    "PricingPrecisionOverride, RiskPrecisionOverride, MarketRiskRequest.precision=64|32) was retired by roadmap 1.4 "
    "(decision A-12). Give an engine.precision.Precision: a StagePrecision(storage, compute, accumulate) of format "
    "names per stage (simulation, market, pricing), e.g. Precision.throughout('float32'), or over HTTP "
    "{\"simulation\": {\"storage\": \"float32\", \"compute\": \"float32\", \"accumulate\": \"float32\"}, ...}. "
    "Calibration, t=0 values, Greeks and reductions over paths are float64 by decision A-10")


@dataclass(frozen=True)
class StagePrecision:
    """The storage, compute and accumulate formats of one stage, by name (see the module
    docstring). Until roadmap 2.8, `compute` is float64 or float32 and `accumulate` equals
    it; `storage` is any format no wider than `compute`."""
    storage: str = "float64"
    compute: str = "float64"
    accumulate: str = "float64"

    def __post_init__(self):
        rows = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if not isinstance(value, str):
                raise TypeError(f"StagePrecision.{f.name} must be a format name such as 'float32', got {value!r}; "
                                f"{RETIRED_SHAPE}")
            try:
                rows[f.name] = format_of(value)
            except ValueError as exc:
                raise ValueError(f"StagePrecision.{f.name}: {exc}") from None
        storage, compute, accumulate = rows["storage"], rows["compute"], rows["accumulate"]
        if storage.storage_step:
            _refuse("storage", storage.name, f"storage in {storage.name} is enabled by roadmap step "
                                             f"{storage.storage_step}")
        if compute.compute_step:
            _refuse("compute", compute.name, f"compute in {compute.name} is enabled by roadmap step "
                                             f"{compute.compute_step} (difference-form kernels); until then float64 "
                                             f"or float32")
        if storage.bits > compute.bits:
            _refuse("storage", storage.name, f"wider than compute={compute.name!r}; storing wider than computed adds "
                                             f"no precision")
        if accumulate.bits < compute.bits:
            _refuse("accumulate", accumulate.name, f"narrower than compute={compute.name!r}")
        if accumulate.name != compute.name:
            _refuse("accumulate", accumulate.name, f"an accumulate format other than compute={compute.name!r} is "
                                                   f"enabled by roadmap step 2.8 (kernels with explicit "
                                                   f"accumulators); until then they are equal")

    @property
    def storage_dtype(self):
        return format_of(self.storage).dtype

    @property
    def compute_dtype(self):
        return format_of(self.compute).dtype

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
    """The precision of each adjustable stage, and the pricing stage's overrides per product
    and per trade (see the module docstring). `Precision()` is float64 everywhere."""
    simulation: StagePrecision = StagePrecision()
    market: StagePrecision = StagePrecision()
    pricing: StagePrecision = StagePrecision()
    by_product: Mapping[str, StagePrecision] = field(default_factory=Overrides)
    by_trade: Mapping[str, StagePrecision] = field(default_factory=Overrides)
    rounding: str = "nearest"
    rounding_seed: int = 0
    paired_fraction: float = 0.0

    def __post_init__(self):
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
        if self.rounding == "stochastic" and not any(s.scaled_storage for s in self._stages()):
            raise ValueError("Precision.rounding='stochastic' rounds values into a scaled storage format (float16, "
                             "bfloat16, FP8), and no stage or override stores in one")
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
    def throughout(cls, name: str) -> "Precision":
        """Every stage stored, computed and accumulated in format `name`."""
        stage = StagePrecision(name, name, name)
        return cls(**{s: stage for s in STAGES})

    def precision_for(self, trade) -> StagePrecision:
        """The pricing stage of `trade` (any object with `trade_id` and `product`):
        `by_trade`, else `by_product`, else `pricing` (decision A-15)."""
        if trade.trade_id in self.by_trade:
            return self.by_trade[trade.trade_id]
        return self.by_product.get(trade.product, self.pricing)

    def store(self, x, storage: str, stream: str, axis: int = 0):
        """`x` stored at format `storage` with this policy's rounding (`engine.precision.store`),
        its block scales along `axis`, the scenario axis. `stream` names the array in the run
        (see the module docstring): it seeds a stochastic rounding."""
        if self.rounding == "stochastic" and format_of(storage).scaled:
            return store(x, storage, "stochastic", rounding_key(self.rounding_seed, stream), axis)
        return store(x, storage, axis=axis)

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
