"""
Pricing-engine configuration: ORE's `pricingengine.xml` as dataclasses, the engine per product
of the run configuration (`engine.run.config.RunConfig.pricing`).

Swaps have one engine (ORE's `DiscountingSwapEngine`). Europeans: `Bachelier`, ORE's default
(`EuropeanSwaptionEngineBuilder` -> `BlackMultiLegOptionEngine` on the market's normal
volatility, `engine.pricing.european`), or `Jamshidian`, QuantLib's
`JamshidianSwaptionEngine` on a Hull-White model whose reversion and volatility are
`JamshidianEngineConfig` (`engine.pricing.jamshidian`). ORE has no builder for it, so its
model has no default: choosing it requires `PricingConfig.jamshidian`. Bermudans/Americans:
ORE's LGM grid engine, `LgmSwaptionEngineConfig`.

Every engine is the same for either simulation model (`CamConfig.ir`): an engine has its own
model, as in ORE's `pricingengine.xml`, and on a path it prices on the path's curves.

Bermudan/American defaults are ORE's example configuration (Examples/Products/Input/pricingengine.xml,
`BermudanSwaption`), `shift_horizon` its builder's default 0.5 (decisions A-3, D-10; I-32), and
`bootstrap_tolerance` (ORE's `Tolerance`) its shared example configuration's 1e-4 (I-73).

**Kernel settings** (`PricingConfig.kernel_settings`, I-114). The precision evidence
(`engine.precision.evidence`) validates a product's figures only under the settings that change
the arithmetic of its kernel, where its rounding happens and accumulates: the European engine
(Bachelier's formula, or Jamshidian's root x* found by its solver in the stage's format), the
grid engine's state grid (`n_per_std`, `std_devs`: the operator's size), an American's exercise
steps (the rollback's length), and whether the analytic runs a calibrated option on one model
or recalibrates it under each scenario (market risk, I-112: each scenario its own operator).
The model's parameters (reversion, volatility, calibration strategy, tolerance, reference grid,
shift horizon, the calibration's solver, all float64 work, A-10) change the numbers a kernel
rounds, as the market does, not the kernel: the evidence names them in each campaign's
definition, not in its key.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import ORE

from engine.calibration.ore_lgm import BOOTSTRAP_TOLERANCE, SwapIndexConventions, check_tolerance
from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig
from engine.instruments.treasury import BondConfig
from engine.solvers.roots import DEFAULT_SOLVER, check_solver

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
    bootstrap_tolerance: ORE's `Tolerance`: a calibration whose error (the root mean square of
        its helpers' relative price errors) is not below it misses its basket and is refused,
        as ORE's `LgmBuilder` fails (`engine.calibration.ore_lgm.BootstrapResult.misses`, I-73).
        ORE has no default; 1e-4 is its shared example configuration's
        (`engine.calibration.ore_lgm.BOOTSTRAP_TOLERANCE`).
    reference_calibration_grid: ORE's `ReferenceCalibrationGrid`: at most one helper per
        interval of this grid, and an American's basket expiries.
    shift_horizon: ORE's `ShiftHorizon`, a fraction of the time from the trade's build to its
        underlying's maturity; the LGM's H is shifted to vanish there, which moves the grid's
        state, not the model (`engine.pricing.bermudan.shift_horizon_time`). 0 is the unshifted
        model.
    n_per_std / std_devs: the Grid engine's `nx`/`sx` (ORE's example: 30 and 5).
    exercise_time_steps_per_year: an American's exercise grid.
    swap_index: the conventions the calibration helpers are built on.
    solver: the root solver of the bootstrap's buckets and of each helper's y*
        (`engine.solvers.roots`, decision A-21): `"Newton"` (the default) or `"Bisection"`
        (the reference, the engine's numbers before Newton bit for bit). Both reach
        ORE's root, more exactly than ORE's own solvers.
    """
    reversion: float = 0.0
    volatility: float = 0.01
    calibration: str = "Bootstrap"
    strategy: str = "CoterminalDealStrike"
    bootstrap_tolerance: float = BOOTSTRAP_TOLERANCE
    reference_calibration_grid: str = "400,3M"
    shift_horizon: float = 0.5
    n_per_std: int = 30
    std_devs: float = 5.0
    exercise_time_steps_per_year: int = 24
    swap_index: SwapIndexConventions = SwapIndexConventions()
    solver: str = DEFAULT_SOLVER

    def __post_init__(self):
        check_solver(self.solver, "LgmSwaptionEngineConfig.solver")
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
        check_tolerance(self.bootstrap_tolerance, "LgmSwaptionEngineConfig.bootstrap_tolerance")
        if not math.isfinite(self.shift_horizon) or self.shift_horizon < 0.0:
            raise ValueError(f"shift_horizon must be finite and >= 0 (ORE's LgmBuilder requires a non-negative "
                             f"horizon); got {self.shift_horizon!r}")
        reference_grid_dates(ORE.Date(1, 1, 2020), self.reference_calibration_grid)

    @property
    def calibrates(self) -> bool:
        """Whether the trade's LGM is calibrated to its basket (`calibration="Bootstrap"`), or
        fixed at `reversion` and `volatility`."""
        return self.calibration != "None"


@dataclass(frozen=True)
class JamshidianEngineConfig:
    """The Hull-White model of the `Jamshidian` European engine: `dr = (theta(t) - a r) dt +
    sigma dW` fitted to the curve it prices on, with constant `reversion` a and `volatility`
    sigma (QuantLib's `HullWhite(termStructure, a, sigma)`, whose parameters are held under a
    positive constraint, so both must be positive: at a = 0 QuantLib raises, I-41). `solver`
    finds the exercise boundary x* (`engine.solvers.roots`, decision A-21): `"Newton"` (the
    default) or `"Bisection"` (the reference)."""
    reversion: float
    volatility: float
    solver: str = DEFAULT_SOLVER

    def __post_init__(self):
        check_solver(self.solver, "JamshidianEngineConfig.solver")
        for name in ("reversion", "volatility"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"JamshidianEngineConfig.{name} must be finite and > 0 (QuantLib's HullWhite "
                                 f"holds it positive); got {value!r}")


@dataclass(frozen=True)
class PricingConfig:
    """The engine per product (see the module docstring). `jamshidian` is the model of the
    `Jamshidian` European engine, required with it and refused without it. A Bermudan's or
    American's LGM is calibrated where ORE builds the trade and kept on every path, as ORE's
    exposure simulation keeps it, which never recalibrates (I-49, decision A-24,
    `engine.pricing.bermudan`)."""
    european: str = "Bachelier"
    bermudan: LgmSwaptionEngineConfig = field(default_factory=LgmSwaptionEngineConfig)
    american: LgmSwaptionEngineConfig = field(default_factory=LgmSwaptionEngineConfig)
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

    def kernel_settings(self, products: Iterable[str], *, recalibrated: bool) -> Dict[str, str]:
        """Per product of `products`, the settings of this configuration that change its kernel's
        arithmetic, as the precision evidence keys them (see the module docstring), e.g.
        `"engine=LgmGrid nx=30 sx=5 model=fixed"`. `recalibrated`: whether the analytic
        recalibrates a calibrated option under each of its scenarios (market risk, I-112); the
        exposure simulation keeps on every path the model calibrated when the trade was built
        (A-24), a fixed model as `calibration="None"` is. In product order."""
        return {product: self._kernel_settings(product, recalibrated) for product in sorted(set(products))}

    def _kernel_settings(self, product: str, recalibrated: bool) -> str:
        if product in (SwapConfig.product, BondConfig.product):
            return "engine=Discounting"
        if product == SwaptionConfig.product:
            if self.european == "Jamshidian":
                return f"engine=Jamshidian solver={self.jamshidian.solver}"
            return f"engine={self.european}"
        if product in (BermudanSwaptionConfig.product, AmericanSwaptionConfig.product):
            american = product == AmericanSwaptionConfig.product
            engine = self.american if american else self.bermudan
            steps = f" steps_per_year={engine.exercise_time_steps_per_year}" if american else ""
            model = "recalibrated" if recalibrated and engine.calibrates else "fixed"
            # The state grid by ORE's names, `nx`/`sx`.
            return f"engine=LgmGrid nx={engine.n_per_std} sx={engine.std_devs:g}{steps} model={model}"
        raise ValueError(f"no pricing engine for product {product!r}")


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
