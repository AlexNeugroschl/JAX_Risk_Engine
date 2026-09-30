"""
The run configuration (roadmap 1.2, I-68): every choice of model, engine, method and
precision a portfolio run makes, in one value, with ORE's defaults (decisions A-1, A-5, A-9
in compliance/decisions.md). ORE configures a run with files; each component is one of them:

    RunConfig
      simulation     CamConfig        simulation.xml: the date grid, the model per currency
                                      (`ir[ccy]`, as CrossAssetModelData's `<LGM ccy=...>`),
                                      the simulation-market tenors, samples and seed
      pricing        PricingConfig    pricingengine.xml: the engine per product
      greeks         GreeksConfig     the Greeks method; for bump-and-revalue, sensitivity.xml
      precision      PrecisionConfig  the dtype of each stage (not in ORE: the research axis)
      base_currency                   ore.xml's `baseCurrency`, when there is no simulation

| Component | Options | Default |
|---|---|---|
| Model per currency | LGM (`LgmConfig` in `simulation.ir`); Hull-White (below) | LGM |
| Simulation | classic revaluation (ORE's AMC is F-03) | classic |
| Engine per product | swap: discounting; European: `Bachelier`, `Jamshidian`; Bermudan/American: LGM grid | ORE's builders |
| Greeks method | `Bump`, `AD` | `Bump` |
| Precision per stage | 32 or 64 for simulation, pricing, risk, calibration | 64 |

Each model implements some of the options. A run asking a model for an option it does not
implement is refused before any work, naming the field (`check_market_path`,
`check_hull_white`), never priced some other way.

The Hull-White model is, until roadmap 1.3, selected by passing its `SimulationConfig` as the
request's market; its Bermudan/American engine is configured on the trades (`hw_a`,
`hw_sigma`) and by `calibration_targets` (I-63, I-47). `HULL_WHITE_CONFIG` names the engines
and Greeks method it implements.
"""
from dataclasses import dataclass, field
from typing import Optional, Union

import jax.numpy as jnp

from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig
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
    to `default`. Resolved only in `_resolve_pricing_dtype`."""
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
    `exposure` has no curve, so `price_portfolio` casts `npv_cube` to it before the
    statistics."""
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
    plain int means every sub-field. The market path honours `simulation` only and refuses
    the others below 64 until roadmap 1.4 (I-55).

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


_PRICING_TYPE_FIELD = {
    SwapConfig: "swap",
    SwaptionConfig: "european_swaption",
    BermudanSwaptionConfig: "bermudan_swaption",
    AmericanSwaptionConfig: "american_swaption",
}


def _resolve_pricing_dtype(pricing: Union[int, PricingPrecisionOverride], trade_type: type):
    """`PrecisionConfig.pricing` (int or override) -> dtype for one trade type."""
    if isinstance(pricing, int):
        return _dtype_of(pricing)
    bits = getattr(pricing, _PRICING_TYPE_FIELD[trade_type]) or pricing.default
    return _dtype_of(bits)


def _resolve_risk_dtype(risk: Union[int, RiskPrecisionOverride], metric: str):
    """`PrecisionConfig.risk` (int or override) -> dtype for one metric ('delta_gamma',
    'theta', 'vega', 'exposure')."""
    if isinstance(risk, int):
        return _dtype_of(risk)
    bits = getattr(risk, metric) or risk.default
    return _dtype_of(bits)


@dataclass(frozen=True)
class GreeksConfig:
    """How `compute_greeks` computes them (decision A-5).

    method: `Bump` (ORE's sensitivity analysis, `engine.risk.sensitivities`; the default) or
        `AD` (automatic differentiation, `engine.risk.greeks`).
    sensitivity: the bump-and-revalue settings, ORE's `sensitivity.xml` (curve tenors, shift
        sizes, Theta horizon, volatility decay on the Theta date).
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

    simulation: required for scenario risk on the market path (`PortfolioRequest.scenario_risk`).
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
                          "(a Hull-White SimulationConfig is the request's market, not its simulation)")
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


#: The engines and Greeks method the Hull-White model implements (until roadmap 1.3):
#: Jamshidian Europeans on the model's volatility (I-46) and AD Greeks.
HULL_WHITE_CONFIG = RunConfig(pricing=PricingConfig(european="Jamshidian"), greeks=GreeksConfig(method="AD"))


def check_market_path(config: RunConfig, trades, compute_greeks: bool, calibration_targets=None) -> None:
    """Refuse, naming the field, what the market path (ORE's LGM cross-asset model) does not
    implement: the European engine for Europeans, the Greeks method when Greeks are asked
    for, any stage but the simulation below float64, and the Hull-White model's shared
    calibration basket."""
    if any(isinstance(t, SwaptionConfig) for t in trades) and config.pricing.european != "Bachelier":
        _refuse("config.pricing.european", config.pricing.european,
                "the market path prices Europeans with ORE's Bachelier engine on the market volatility; "
                "Jamshidian on the LGM is feature F-01")
    if compute_greeks and config.greeks.method != "Bump":
        _refuse("config.greeks.method", config.greeks.method,
                "the market path computes ORE's bump-and-revalue sensitivities; AD Greeks on it are feature F-01")
    for stage in ("pricing", "risk", "calibration"):
        value = getattr(config.precision, stage)
        if value != 64:
            _refuse(f"config.precision.{stage}", value,
                    "the market path computes this stage in float64; only precision.simulation is adjustable on "
                    "it until roadmap 1.4 (I-55)")
    if calibration_targets is not None:
        _refuse("calibration_targets", "<given>",
                "the Hull-White model's shared basket; the market path builds each trade's basket as ORE does")


def check_hull_white(config: RunConfig, trades, compute_greeks: bool) -> None:
    """Refuse, naming the field, what the Hull-White model does not implement or read: the
    European engine and Greeks method other than its own when used, and every setting of the
    market path's LGM model and engines changed from its default (the Hull-White model's
    simulation is its market, and its Bermudan engine is set on the trades)."""
    if any(isinstance(t, SwaptionConfig) for t in trades) and config.pricing.european != "Jamshidian":
        _refuse("config.pricing.european", config.pricing.european,
                "the Hull-White model prices Europeans with Jamshidian on the model volatility (I-46, roadmap 1.3); "
                "use HULL_WHITE_CONFIG or pricing.european='Jamshidian'")
    if compute_greeks and config.greeks.method != "AD":
        _refuse("config.greeks.method", config.greeks.method,
                "the Hull-White model computes Greeks by AD only (feature F-01); use HULL_WHITE_CONFIG or "
                "greeks.method='AD'")
    lgm_only = {
        "config.simulation": (config.simulation, None),
        "config.base_currency": (config.base_currency, None),
        "config.pricing.bermudan": (config.pricing.bermudan, PricingConfig().bermudan),
        "config.pricing.american": (config.pricing.american, PricingConfig().american),
        "config.pricing.recalibrate": (config.pricing.recalibrate, PricingConfig().recalibrate),
        "config.greeks.sensitivity": (config.greeks.sensitivity, SensitivityConfig()),
    }
    for name, (value, default) in lgm_only.items():
        if value != default:
            _refuse(name, value, "a setting of the market path's LGM model and engines, which the Hull-White model "
                                 "does not read (its simulation is the request's market, its Bermudan engine is set "
                                 "on the trades: I-63, roadmap 1.3); leave it at its default")

