"""
Pricing-engine configuration: ORE's `pricingengine.xml` as dataclasses, the engine per product
of the run configuration (`engine.portfolio.config.RunConfig.pricing`).

Swaps have one engine (ORE's `DiscountingSwapEngine`). Europeans: `Bachelier`, ORE's default
(`EuropeanSwaptionEngineBuilder` -> `BlackMultiLegOptionEngine` on the market's normal
volatility, `engine.valuation.european`), or `Jamshidian`, QuantLib's
`JamshidianSwaptionEngine` on a Hull-White model whose reversion and volatility are
`JamshidianEngineConfig` (`engine.valuation.jamshidian`). ORE has no builder for it, so its
model has no default: choosing it requires `PricingConfig.jamshidian`. Bermudans/Americans:
ORE's LGM grid engine, `LgmSwaptionEngineConfig`.

Every engine is the same for either simulation model (`CamConfig.ir`): an engine has its own
model, as in ORE's `pricingengine.xml`, and on a path it prices on the path's curves.

Bermudan/American defaults are ORE's example configuration (Examples/Products/Input/pricingengine.xml,
`BermudanSwaption`), with one recorded exception: `shift_horizon` is 0 (Basel decision D-10,
the configuration ORE parity is proven for), where ORE's example uses 0.5; see I-32. Other
shift horizons are refused rather than silently priced at 0.
"""
import math
from dataclasses import dataclass, field
from typing import Optional

import ORE

from engine.calibration.ore_lgm import SwapIndexConventions

#: European swaption engines; the first is ORE's default.
EUROPEAN_ENGINES = ("Bachelier", "Jamshidian")
CALIBRATION_METHODS = ("Bootstrap", "None")
CALIBRATION_STRATEGIES = ("CoterminalDealStrike", "CoterminalATM")


@dataclass(frozen=True)
class LgmSwaptionEngineConfig:
    """ORE's `LGMGridSwaptionEngineBuilder` model and engine parameters.

    reversion / volatility: the LGM's constant reversion, and its volatility (fixed with
        `calibration="None"`, the bootstrap's start otherwise).
    calibration / strategy: `Bootstrap` to the trade's co-terminal basket struck at the deal
        strike (`CoterminalDealStrike`) or ATM (`CoterminalATM`), or `None`.
    reference_calibration_grid: ORE's `ReferenceCalibrationGrid`: at most one helper per
        interval of this grid, and an American's basket expiries.
    n_per_std / std_devs: the Grid engine's `nx`/`sx` (ORE's example: 30 and 5).
    exercise_time_steps_per_year: an American's exercise grid.
    swap_index: the conventions the calibration helpers are built on.
    """
    reversion: float = 0.0
    volatility: float = 0.01
    calibration: str = "Bootstrap"
    strategy: str = "CoterminalDealStrike"
    reference_calibration_grid: str = "400,3M"
    shift_horizon: float = 0.0
    n_per_std: int = 30
    std_devs: float = 5.0
    exercise_time_steps_per_year: int = 24
    swap_index: SwapIndexConventions = SwapIndexConventions()

    def __post_init__(self):
        if not math.isfinite(self.reversion):
            raise ValueError(f"reversion must be finite; got {self.reversion!r}")
        if not math.isfinite(self.volatility) or self.volatility < 0.0:
            raise ValueError(f"volatility must be finite and non-negative; got {self.volatility!r}")
        if self.n_per_std < 1 or not math.isfinite(self.std_devs) or self.std_devs <= 0.0:
            raise ValueError(f"the grid needs n_per_std >= 1 and std_devs > 0; got {self.n_per_std}, {self.std_devs}")
        if self.exercise_time_steps_per_year < 1:
            raise ValueError(f"exercise_time_steps_per_year must be >= 1; got {self.exercise_time_steps_per_year}")
        if self.calibration not in CALIBRATION_METHODS:
            raise ValueError(f"calibration must be one of {CALIBRATION_METHODS}; got {self.calibration!r}")
        if self.strategy not in CALIBRATION_STRATEGIES:
            raise ValueError(f"strategy must be one of {CALIBRATION_STRATEGIES}; got {self.strategy!r}")
        if self.shift_horizon != 0.0:
            raise ValueError(
                "shift_horizon must be 0 (Basel decision D-10): ORE's shifted grid is not reproduced (I-32)")
        reference_grid_dates(ORE.Date(1, 1, 2020), self.reference_calibration_grid)


@dataclass(frozen=True)
class JamshidianEngineConfig:
    """The Hull-White model of the `Jamshidian` European engine: `dr = (theta(t) - a r) dt +
    sigma dW` fitted to the curve it prices on, with constant `reversion` a and `volatility`
    sigma (QuantLib's `HullWhite(termStructure, a, sigma)`, whose parameters are held under a
    positive constraint, so both must be positive: at a = 0 QuantLib raises, I-41)."""
    reversion: float
    volatility: float

    def __post_init__(self):
        for name in ("reversion", "volatility"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"JamshidianEngineConfig.{name} must be finite and > 0 (QuantLib's HullWhite "
                                 f"holds it positive); got {value!r}")


@dataclass(frozen=True)
class PricingConfig:
    """The engine per product (see the module docstring). `jamshidian` is the model of the
    `Jamshidian` European engine, required with it and refused without it. `recalibrate` is
    ORE's `ValuationEngine` flag (default true): model-based trades are recalibrated on every
    path and date."""
    european: str = "Bachelier"
    bermudan: LgmSwaptionEngineConfig = field(default_factory=LgmSwaptionEngineConfig)
    american: LgmSwaptionEngineConfig = field(default_factory=LgmSwaptionEngineConfig)
    recalibrate: bool = True
    jamshidian: Optional[JamshidianEngineConfig] = None

    def __post_init__(self):
        if self.european not in EUROPEAN_ENGINES:
            raise ValueError(f"european must be one of {EUROPEAN_ENGINES}; got {self.european!r}")
        if self.jamshidian is not None and not isinstance(self.jamshidian, JamshidianEngineConfig):
            raise TypeError(f"jamshidian must be a JamshidianEngineConfig; got {type(self.jamshidian).__name__}")
        if self.european == "Jamshidian" and self.jamshidian is None:
            raise ValueError("european='Jamshidian' needs its Hull-White model: set PricingConfig.jamshidian "
                             "(JamshidianEngineConfig(reversion, volatility)); ORE has no default for it")
        if self.european != "Jamshidian" and self.jamshidian is not None:
            raise ValueError(f"PricingConfig.jamshidian is the Jamshidian engine's model, which european="
                             f"{self.european!r} does not read; leave it unset")


def reference_grid_dates(reference: ORE.Date, grid: str):
    """ORE's `DateGrid("N,P")` from `reference`: `TARGET.advance(reference, i * P, Following)`
    for i = 1..N (OREData/ored/utilities/dategrid.cpp, `buildDates`)."""
    count, _, period = grid.partition(",")
    tenor = ORE.Period(period or "1Y")
    if tenor.units() == ORE.Days:
        raise ValueError(f"a daily reference calibration grid is not supported; got {grid!r}")
    dates = []
    for i in range(1, int(count) + 1):
        d = ORE.TARGET().advance(reference, ORE.Period(i * tenor.length(), tenor.units()), ORE.Following, False)
        if not dates or d != dates[-1]:
            dates.append(d)
    return dates
