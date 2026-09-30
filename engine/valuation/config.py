"""
Pricing-engine configuration: ORE's `pricingengine.xml` as dataclasses, the engine per product
of the run configuration (`engine.portfolio.config.RunConfig.pricing`).

Swaps have one engine (ORE's `DiscountingSwapEngine`). Europeans: `Bachelier`, ORE's default
(`EuropeanSwaptionEngineBuilder` -> `BlackMultiLegOptionEngine` on the market's normal
volatility, `engine.valuation.european`), or `Jamshidian` on the model's volatility (the
Hull-White model's only engine, I-46). Bermudans/Americans: ORE's LGM grid engine,
`LgmSwaptionEngineConfig`.

Bermudan/American defaults are ORE's example configuration (Examples/Products/Input/pricingengine.xml,
`BermudanSwaption`), with one recorded exception: `shift_horizon` is 0 (Basel decision D-10,
the configuration ORE parity is proven for), where ORE's example uses 0.5; see I-32. Other
shift horizons are refused rather than silently priced at 0.
"""
from dataclasses import dataclass, field

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
        if self.calibration not in CALIBRATION_METHODS:
            raise ValueError(f"calibration must be one of {CALIBRATION_METHODS}; got {self.calibration!r}")
        if self.strategy not in CALIBRATION_STRATEGIES:
            raise ValueError(f"strategy must be one of {CALIBRATION_STRATEGIES}; got {self.strategy!r}")
        if self.shift_horizon != 0.0:
            raise ValueError(
                "shift_horizon must be 0 (Basel decision D-10): ORE's shifted grid is not reproduced (I-32)")
        reference_grid_dates(ORE.Date(1, 1, 2020), self.reference_calibration_grid)


@dataclass(frozen=True)
class PricingConfig:
    """The engine per product (see the module docstring). `recalibrate` is ORE's
    `ValuationEngine` flag (default true): model-based trades are recalibrated on every path
    and date. Which engines a model implements is checked by the run
    (`engine.portfolio.config`)."""
    european: str = "Bachelier"
    bermudan: LgmSwaptionEngineConfig = field(default_factory=LgmSwaptionEngineConfig)
    american: LgmSwaptionEngineConfig = field(default_factory=LgmSwaptionEngineConfig)
    recalibrate: bool = True

    def __post_init__(self):
        if self.european not in EUROPEAN_ENGINES:
            raise ValueError(f"european must be one of {EUROPEAN_ENGINES}; got {self.european!r}")


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
