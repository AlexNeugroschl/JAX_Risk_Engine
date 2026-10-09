"""
The run configuration (I-68): every choice of model, engine, method and
precision a portfolio run makes, in one value, with ORE's defaults (decisions A-1, A-5, A-9
in compliance/decisions.md). ORE configures a run with files; each component is one of them:

    RunConfig
      simulation     CamConfig        simulation.xml: the date grid, the model per currency
                                      (`ir[ccy]`: `LgmConfig`, or `HullWhiteConfig`, ORE's
                                      `<LGM>` with `VolatilityType HullWhite`), the
                                      simulation-market tenors, samples and seed
      pricing        PricingConfig    pricingengine.xml: the engine per product
      greeks         GreeksConfig     the Greeks method; sensitivity.xml's settings
      precision      Precision        storage, compute and accumulate formats per adjustable stage,
                                      pricing per product and per trade (`engine.precision`; not
                                      in ORE: the research axis)
      base_currency                   ore.xml's `baseCurrency`, when there is no simulation

| Component | Options | Default |
|---|---|---|
| Model per currency | LGM (`LgmConfig`); Hull-White (`HullWhiteConfig`) | none: named per currency |
| Simulation | classic revaluation (ORE's AMC is F-03) | classic |
| Engine per product | swap: discounting; European: `Bachelier`, `Jamshidian`; Bermudan/American: LGM grid | ORE's builders |
| Greeks method | `Bump`, `AD` | `Bump` |
| Precision per stage | simulation, market, pricing: float64 or float32 compute; storage also float16, bfloat16 or FP8 (block scales, nearest or stochastic rounding); pricing overridable per product and per trade | float64 |

Every option runs with every other: the models differ only in the simulation, and the engines
and Greeks methods price whatever the simulation produced. What the pipeline does not
implement yet is refused before any work, naming the field, never done some other way: a
precision format not enabled yet (`engine.precision.policy`), an
engine option where a trade meets it (`engine.pricing.cube.validate_trades`).
Calibration, t=0 values, Greeks and reductions over paths are float64 by decision (A-10).
"""
from dataclasses import dataclass, field
from typing import Optional

from engine.market_simulation.config import CamConfig
from engine.precision import Precision, require_precision
from engine.pricing.config import PricingConfig
from engine.risk.greeks.bump import SensitivityConfig

#: The Greeks methods: ORE's bump-and-revalue sensitivities, or automatic differentiation.
GREEKS_METHODS = ("Bump", "AD")


def _require_type(name: str, value, expected: type, hint: str = "") -> None:
    if not isinstance(value, expected):
        raise TypeError(f"{name} must be a {expected.__name__}, got {type(value).__name__} {hint}".rstrip())


@dataclass(frozen=True)
class GreeksConfig:
    """How `compute_greeks` computes them (decision A-5).

    method: `Bump` (ORE's sensitivity analysis, `engine.risk.greeks.bump`; the default) or
        `AD` (automatic differentiation, `engine.risk.greeks.ad`), for any model and engine.
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
    precision: Precision = field(default_factory=Precision)
    base_currency: Optional[str] = None

    def __post_init__(self):
        if self.simulation is not None:
            _require_type("RunConfig.simulation", self.simulation, CamConfig,
                          "(the Hull-White model is a HullWhiteConfig in CamConfig.ir)")
        _require_type("RunConfig.pricing", self.pricing, PricingConfig)
        _require_type("RunConfig.greeks", self.greeks, GreeksConfig)
        require_precision("RunConfig.precision", self.precision)
        if (self.simulation is not None and self.base_currency is not None
                and self.base_currency != self.simulation.base_currency):
            raise ValueError(f"RunConfig.base_currency={self.base_currency!r} contradicts the simulation's "
                             f"base currency {self.simulation.base_currency!r}; give one or make them equal")

    @property
    def reporting_currency(self) -> str:
        if self.simulation is not None:
            return self.simulation.base_currency
        return self.base_currency or "USD"

