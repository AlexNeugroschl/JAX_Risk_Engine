"""
The precision of a run (docs/planning/details/precision.md §3, §4; decisions A-10, A-12, D-9).

    Precision
      simulation   StagePrecision   the Sobol shocks and the model states
      market       StagePrecision   the scenario market: curves, numeraire, FX and equity
      pricing      StagePrecision   pricing on paths (and market-risk revaluation) and the
                                    values it stores (the cube, the scenario NPVs)

Each adjustable stage has three precisions: `storage`, the format its output is kept in until
the next stage reads it; `compute`, the format its arithmetic is done in; `accumulate`, the
format its sums accumulate in. `Precision()` is float64 everywhere, the engine's default and
the one every ORE parity suite runs on.

Calibration, t=0 values, Greeks and every reduction over paths or scenarios (exposure, VaR/ES)
are not stages here: they are float64 by decision (A-10).

Validation refuses, naming the field, before any work: a name outside the format table, a
format used before the roadmap step that enables it, `storage` wider than `compute` (storing
wider gains nothing), `accumulate` narrower than `compute`. Every other combination may be run
(D-9). The 32/64 shape before roadmap 1.4 (`PrecisionConfig`) is refused, not translated
(A-12): `RETIRED_SHAPE` says what replaces it.
"""
from dataclasses import dataclass, fields

from engine.precision.formats import format_of

#: The adjustable stages, in pipeline order.
STAGES = ("simulation", "market", "pricing")

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
    it; until 1.6, `storage` is float64 or float32."""
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
            _refuse("storage", storage.name, f"storage in {storage.name} (with block scales) is enabled by roadmap "
                                             f"step {storage.storage_step}; until then float64 or float32")
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


def _refuse(name: str, value: str, reason: str) -> None:
    raise ValueError(f"StagePrecision.{name}={value!r}: {reason}")


@dataclass(frozen=True)
class Precision:
    """The precision of each adjustable stage (see the module docstring). `Precision()` is
    float64 everywhere."""
    simulation: StagePrecision = StagePrecision()
    market: StagePrecision = StagePrecision()
    pricing: StagePrecision = StagePrecision()

    def __post_init__(self):
        for stage in STAGES:
            value = getattr(self, stage)
            if not isinstance(value, StagePrecision):
                raise TypeError(f"Precision.{stage} must be a StagePrecision, got {value!r}; {RETIRED_SHAPE}")

    @classmethod
    def throughout(cls, name: str) -> "Precision":
        """Every stage stored, computed and accumulated in format `name`."""
        stage = StagePrecision(name, name, name)
        return cls(**{s: stage for s in STAGES})


def require_precision(owner: str, value) -> None:
    """Refuse anything but a `Precision` as `owner`'s precision, naming the replacement of the
    retired 32/64 shape (A-12)."""
    if not isinstance(value, Precision):
        raise TypeError(f"{owner} must be an engine.precision.Precision, got {type(value).__name__} {value!r}; "
                        f"{RETIRED_SHAPE}")
