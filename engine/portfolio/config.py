"""
The run configuration (roadmap 1.2 and 1.3, I-68): every choice of model, engine, method and
precision a portfolio run makes, in one value, with ORE's defaults (decisions A-1, A-5, A-9
in compliance/decisions.md). ORE configures a run with files; each component is one of them:

    RunConfig
      simulation     CamConfig        simulation.xml: the date grid, the model per currency
                                      (`ir[ccy]`: `LgmConfig`, or `HullWhiteConfig`, ORE's
                                      `<LGM>` with `VolatilityType HullWhite`), the
                                      simulation-market tenors, samples and seed
      pricing        PricingConfig    pricingengine.xml: the engine per product
      greeks         GreeksConfig     the Greeks method; sensitivity.xml's settings
      precision      PrecisionConfig  the dtype of each stage (not in ORE: the research axis)
      base_currency                   ore.xml's `baseCurrency`, when there is no simulation

| Component | Options | Default |
|---|---|---|
| Model per currency | LGM (`LgmConfig`); Hull-White (`HullWhiteConfig`) | none: named per currency |
| Simulation | classic revaluation (ORE's AMC is F-03) | classic |
| Engine per product | swap: discounting; European: `Bachelier`, `Jamshidian`; Bermudan/American: LGM grid | ORE's builders |
| Greeks method | `Bump`, `AD` | `Bump` |
| Precision per stage | 32 or 64 for simulation, pricing, risk, calibration | 64 |

Every option runs with every other: the models differ only in the simulation, and the engines
and Greeks methods price whatever the simulation produced. What the pipeline does not
implement yet is refused before any work, naming the field (`check_run`), never done some
other way: today, a stage other than the simulation below float64 (I-55, roadmap 1.4).
"""
from dataclasses import dataclass, field
from typing import Optional, Union

import jax.numpy as jnp

from engine.risk.sensitivities import SensitivityConfig
from engine.simulation.config import CamConfig
from engine.valuation.config import PricingConfig

#: The Greeks methods: ORE's bump-and-revalue sensitivities, or automatic differentiation.
GREEKS_METHODS = ("Bump", "AD")


def _refuse(name: str, value, reason: str) -> None:
    raise ValueError(f"{name}={value!r}: {reason}")


def _require_type(name: str, value, expected: type, hint: str = "") -> None:
    if not isinstance(value, expected):
        raise TypeError(f"{name} must be a {expected.__name__}, got {type(value).__name__} {hint}".rstrip())


@dataclass(frozen=True)
class PricingPrecisionOverride:
    """Per-instrument-type overrides for `PrecisionConfig.pricing`; `None` fields fall back
    to `default`. Refused by `check_run` until roadmap 1.4 makes the pricing stage
    adjustable."""
    default: int = 64
    swap: Optional[int] = None
    european_swaption: Optional[int] = None
    bermudan_swaption: Optional[int] = None
    american_swaption: Optional[int] = None

    def __post_init__(self):
        for name in ("default", "swap", "european_swaption", "bermudan_swaption", "american_swaption"):
            value = getattr(self, name)
            if value is not None and value not in (32, 64):
                raise ValueError(f"PricingPrecisionOverride.{name} must be 32 or 64, got {value!r}")


@dataclass(frozen=True)
class RiskPrecisionOverride:
    """Per-metric overrides for `PrecisionConfig.risk`; `None` fields fall back to
    `default`. Delta and Gamma share `delta_gamma` (one gradient/Hessian computation).
    Refused by `check_run` until roadmap 1.4 makes the risk stage adjustable."""
    default: int = 64
    delta_gamma: Optional[int] = None
    theta: Optional[int] = None
    vega: Optional[int] = None
    exposure: Optional[int] = None

    def __post_init__(self):
        for name in ("default", "delta_gamma", "theta", "vega", "exposure"):
            value = getattr(self, name)
            if value is not None and value not in (32, 64):
                raise ValueError(f"RiskPrecisionOverride.{name} must be 32 or 64, got {value!r}")


@dataclass(frozen=True)
class PrecisionConfig:
    """
    Dtype knobs, each 32 or 64 (default 64): `simulation` (the scenario paths), `pricing`
    (NPVs and `npv_cube`), `risk` (exposure and Greeks), `calibration` (the sigma
    bootstrap). `pricing` and `risk` also accept a per-type/per-metric override object; a
    plain int means every sub-field. A run honours `simulation` only and refuses the others
    below 64 until roadmap 1.4 (I-55).

    Sub-float32 dtypes are not supported: `jnp.linalg.cholesky` and
    `jax.scipy.stats.norm.ppf` raise on them on the installed CPU backend (see
    docs/concepts/architecture.md, "Adjustable precision").
    """
    simulation: int = 64
    pricing: Union[int, PricingPrecisionOverride] = 64
    risk: Union[int, RiskPrecisionOverride] = 64
    calibration: int = 64

    def __post_init__(self):
        for name in ("simulation", "calibration"):
            value = getattr(self, name)
            if value not in (32, 64):
                raise ValueError(f"PrecisionConfig.{name} must be 32 or 64, got {value!r}")
        if isinstance(self.pricing, int) and self.pricing not in (32, 64):
            raise ValueError(f"PrecisionConfig.pricing must be 32, 64, or a PricingPrecisionOverride, got {self.pricing!r}")
        if isinstance(self.risk, int) and self.risk not in (32, 64):
            raise ValueError(f"PrecisionConfig.risk must be 32, 64, or a RiskPrecisionOverride, got {self.risk!r}")


def _dtype_of(precision_bits: int):
    return jnp.float64 if precision_bits == 64 else jnp.float32


@dataclass(frozen=True)
class GreeksConfig:
    """How `compute_greeks` computes them (decision A-5).

    method: `Bump` (ORE's sensitivity analysis, `engine.risk.sensitivities`; the default) or
        `AD` (automatic differentiation, `engine.risk.greeks`), for any model and engine.
    sensitivity: ORE's `sensitivity.xml`: curve tenors, shift sizes, Theta horizon, volatility
        decay on the Theta date. The AD method reads the shift sizes (Greeks per unit shift),
        the Theta horizon and decay; its Deltas are per market-curve pillar, not per tenor.
    """
    method: str = "Bump"
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)

    def __post_init__(self):
        if self.method not in GREEKS_METHODS:
            raise ValueError(f"GreeksConfig.method must be one of {GREEKS_METHODS}; got {self.method!r}")
        _require_type("GreeksConfig.sensitivity", self.sensitivity, SensitivityConfig)


@dataclass(frozen=True)
class RunConfig:
    """The run configuration (see the module docstring). `RunConfig()` is ORE's defaults.

    simulation: required for scenario risk (`PortfolioRequest.scenario_risk`).
    base_currency: the reporting currency. `None` means the simulation's base currency, or
        USD without a simulation; a value that contradicts the simulation's is refused.
    """
    simulation: Optional[CamConfig] = None
    pricing: PricingConfig = field(default_factory=PricingConfig)
    greeks: GreeksConfig = field(default_factory=GreeksConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    base_currency: Optional[str] = None

    def __post_init__(self):
        if self.simulation is not None:
            _require_type("RunConfig.simulation", self.simulation, CamConfig,
                          "(the Hull-White model is a HullWhiteConfig in CamConfig.ir)")
        _require_type("RunConfig.pricing", self.pricing, PricingConfig)
        _require_type("RunConfig.greeks", self.greeks, GreeksConfig)
        _require_type("RunConfig.precision", self.precision, PrecisionConfig)
        if (self.simulation is not None and self.base_currency is not None
                and self.base_currency != self.simulation.base_currency):
            raise ValueError(f"RunConfig.base_currency={self.base_currency!r} contradicts the simulation's "
                             f"base currency {self.simulation.base_currency!r}; give one or make them equal")

    @property
    def reporting_currency(self) -> str:
        if self.simulation is not None:
            return self.simulation.base_currency
        return self.base_currency or "USD"


def check_run(config: RunConfig) -> None:
    """Refuse, naming the field, what the pipeline does not implement yet: any stage but the
    simulation below float64 (roadmap 1.4, I-55). Engines and Greeks methods are refused where
    a trade meets them (`engine.valuation.portfolio.validate_trades`)."""
    for stage in ("pricing", "risk", "calibration"):
        value = getattr(config.precision, stage)
        if value != 64:
            _refuse(f"config.precision.{stage}", value,
                    "this stage is computed in float64; only precision.simulation is adjustable until roadmap 1.4 "
                    "(I-55)")
