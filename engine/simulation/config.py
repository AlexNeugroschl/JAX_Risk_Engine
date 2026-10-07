"""
Simulation configuration: ORE's `simulation.xml` (Parameters, CrossAssetModel, Market) as one
dataclass, and the function that turns it plus today's `Market` into a scenario market.

    CamConfig          the date grid, the models per currency/pair/equity, correlations,
                       the simulation-market tenors, samples and seed.
    LgmConfig          one currency's `<LGM>`: constant reversion and a volatility, either
                       fixed or bootstrapped to a tenor-based ATM swaption basket
                       (`CalibrationSwaptions`, plan V-6). The default model.
    HullWhiteConfig    the same with the Hull-White model: the short rate's volatility
                       (ORE's `<LGM>` with `VolatilityType HullWhite`).
    simulate           Market + CamConfig -> ScenarioMarket.

The model of a currency is the type of its entry in `CamConfig.ir` (decision A-1: models are
options of the run configuration). Both are one-factor Gaussian models fitted to the
currency's discount curve and simulated exactly under the domestic LGM measure; they differ
in how the volatility is parametrized, so the same volatility number means a different model.

Defaults are ORE's: LGM measure, exact discretization, `ShiftHorizon = 0` (the CAM's default,
unlike the per-trade builder), log-linear scenario curves with flat-forward extrapolation,
ForwardVariance time decay of non-simulated swaption volatilities (the value ORE's example
configurations use; ORE has no code default).
"""
import itertools
from dataclasses import dataclass, field
from typing import ClassVar, Mapping, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import ORE

from engine.calibration.ore_lgm import SwapIndexConventions
from engine.market import Market
from engine.models.curves import ZeroCurve
from engine.models.lgm import Sigma, as_sigma
from engine.models.ore_builders import TIME_AXIS_DAY_COUNTER
from engine.numerics.roots import DEFAULT_SOLVER, check_solver
from engine.precision import Precision, format_name, load, require_precision
from engine.simulation.cam import (
    CrossAssetModel, EqComponent, FxComponent, IrComponent, evolve_states, step_moments,
)
from engine.simulation.random import apply_brownian_bridge, generate_sobol_normals
from engine.simulation.scenario_market import ScenarioMarket, build_scenario_market
from engine.simulation.sharding import shard_scenarios

#: Simulation-market curve tenors when none are configured: the yield-curve tenors of ORE's
#: example simulation configuration (Examples/Exposure/Input/simulation.xml).
DEFAULT_CURVE_TENORS = ("3M", "6M", "1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y", "12Y", "15Y", "20Y", "30Y")

#: ORE's `ReactionToTimeDecay` values for non-simulated swaption volatilities.
SWAPTION_VOL_DECAY_MODES = ("ForwardVariance", "ConstantVariance")

Volatility = Union[float, Sigma]


@dataclass(frozen=True)
class LgmConfig:
    """One currency's LGM (`CrossAssetModelData`'s `<LGM ccy=...>`): Hagan volatility,
    HullWhite (constant) reversion.

    reversion: the constant reversion `a` (`Reversion/InitialValue`, `Calibrate=false`).
    volatility: the volatility if not calibrated; the bootstrap's starting value otherwise.
    calibration_expiries / calibration_terms: the `CalibrationSwaptions` basket, as tenors
        (e.g. `("1Y", "2Y")` and `("9Y", "8Y")`). Empty means no calibration: `volatility`
        is used as given. ATM strikes (`<Strikes/>`). Bootstrapped by
        `engine.calibration.cam` when the model is built.
    swap_index: the conventions the basket's helpers are built on.
    solver: the bootstrap's root solver (`engine.numerics.roots`, decision A-21): `"Newton"`
        (the default) or `"Bisection"` (the reference).
    """
    reversion: float
    volatility: Volatility = 0.01
    calibration_expiries: Tuple[str, ...] = ()
    calibration_terms: Tuple[str, ...] = ()
    swap_index: SwapIndexConventions = SwapIndexConventions()
    solver: str = DEFAULT_SOLVER

    #: ORE's `LgmData::VolatilityType` of this model's volatility (`engine.models.lgm`).
    volatility_type: ClassVar[str] = "Hagan"

    def __post_init__(self):
        check_solver(self.solver, f"{type(self).__name__}.solver")
        if not np.isfinite(self.reversion):
            raise ValueError(f"reversion must be finite; got {self.reversion}")
        if len(self.calibration_expiries) != len(self.calibration_terms):
            raise ValueError("calibration_expiries and calibration_terms must have the same length")
        object.__setattr__(self, "calibration_expiries", tuple(self.calibration_expiries))
        object.__setattr__(self, "calibration_terms", tuple(self.calibration_terms))

    @property
    def calibrated(self) -> bool:
        return bool(self.calibration_expiries)


@dataclass(frozen=True)
class HullWhiteConfig(LgmConfig):
    """One currency's Hull-White model: `dr = (theta(t) - a r) dt + sigma(t) dW`, with theta(t)
    fitted to the currency's discount curve (Brigo-Mercurio 3.36), simulated exactly with its
    own numeraire. ORE's `<LGM>` with `ReversionType HullWhite` and `VolatilityType HullWhite`
    (`LgmBuilder` -> `IrLgm1fPiecewiseConstantHullWhiteAdaptor`), which is how ORE's cross-asset
    model carries a Hull-White currency with the exact discretization.

    `volatility` is the short rate's volatility sigma (constant, or a piecewise `Sigma`), not
    the LGM's alpha: the model is the LGM with alpha(t) = sigma(t) exp(a t). The calibration
    basket and its conventions are `LgmConfig`'s; the bootstrap solves the same helpers, and
    each bucket's Hull-White volatility is the one with the Hagan bootstrap's zeta at the
    helper's expiry (`engine.models.lgm.hull_white_matching_zeta`), which prices every helper
    identically."""

    volatility_type: ClassVar[str] = "HullWhite"


#: The model configurations a currency of `CamConfig.ir` can take.
IrModelConfig = Union[LgmConfig, HullWhiteConfig]


@dataclass(frozen=True)
class CamConfig:
    """ORE's simulation configuration.

    dates: the simulation grid (plan T-7), ascending, all after the market's as-of date.
    ir: currency -> its model, `LgmConfig` (the default) or `HullWhiteConfig`.
        `base_currency` is the domestic currency (ORE's `DomesticCcy`); the other currencies
        follow in the order given.
    fx_volatilities: foreign currency -> Black-Scholes volatility of its FX rate against
        the base currency, for every non-base currency in `ir`.
    equity_volatilities: equity name -> Black-Scholes volatility (equities simulated).
    correlations: `{(factor, factor): rho}` keyed by `CrossAssetModel.factor_names`
        (`IR:USD`, `FX:EURUSD`, `EQ:SP5`), ORE's `InstantaneousCorrelations`. Unlisted pairs
        are 0.
    curve_tenors: simulation-market tenors of every discount and index curve.
    samples / seed: paths and Sobol scrambling seed.
    swaption_vol_decay: `ReactionToTimeDecay` of the (not simulated) swaption vols.
    """
    dates: Tuple[ORE.Date, ...]
    base_currency: str
    ir: Mapping[str, IrModelConfig]
    fx_volatilities: Mapping[str, Volatility] = field(default_factory=dict)
    equity_volatilities: Mapping[str, Volatility] = field(default_factory=dict)
    correlations: Mapping[Tuple[str, str], float] = field(default_factory=dict)
    curve_tenors: Tuple[str, ...] = DEFAULT_CURVE_TENORS
    samples: int = 1000
    seed: int = 42
    swaption_vol_decay: str = "ForwardVariance"

    def __post_init__(self):
        dates = tuple(self.dates)
        if not dates or any(not isinstance(d, ORE.Date) for d in dates):
            raise TypeError("dates must be a non-empty sequence of ORE.Date")
        if any(b <= a for a, b in zip(dates, dates[1:])):
            raise ValueError("dates must increase strictly")
        for currency, model in self.ir.items():
            if not isinstance(model, LgmConfig):
                raise TypeError(f"ir[{currency!r}] must be an LgmConfig or a HullWhiteConfig; "
                                f"got {type(model).__name__}")
        if self.base_currency not in self.ir:
            raise ValueError(f"base_currency {self.base_currency!r} needs a model (LgmConfig, HullWhiteConfig) in ir")
        foreign = [c for c in self.ir if c != self.base_currency]
        if sorted(self.fx_volatilities) != sorted(foreign):
            raise ValueError(
                f"fx_volatilities must give one volatility per non-base currency {foreign}; "
                f"got {sorted(self.fx_volatilities)}")
        if self.samples < 1:
            raise ValueError(f"samples must be positive; got {self.samples}")
        if self.swaption_vol_decay not in SWAPTION_VOL_DECAY_MODES:
            raise ValueError(f"swaption_vol_decay must be one of {SWAPTION_VOL_DECAY_MODES}")
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "curve_tenors", tuple(self.curve_tenors))

    @property
    def currencies(self) -> Tuple[str, ...]:
        """IR components in ORE's order: the base currency first."""
        return (self.base_currency,) + tuple(c for c in self.ir if c != self.base_currency)


def build_cross_asset_model(market: Market, config: CamConfig,
                            sigmas: Optional[Mapping[str, Sigma]] = None) -> CrossAssetModel:
    """The CAM for `config` on `market`, as ORE's `CrossAssetModelBuilder` builds it: every
    currency with a calibration basket is bootstrapped to it first
    (`engine.calibration.cam`). `sigmas` overrides a currency's volatility instead, in that
    currency's parametrization (the short rate's for a `HullWhiteConfig`)."""
    from engine.calibration.cam import calibrate_cam

    sigmas = dict(sigmas or {})
    to_calibrate = {c: lgm for c, lgm in config.ir.items() if c not in sigmas}
    sigmas.update({c: result.sigma for c, result in calibrate_cam(market, to_calibrate).items()})
    base = config.base_currency
    ir = tuple(
        IrComponent(currency=c, curve=ZeroCurve.from_config(market.currency(c).discount_curve),
                    reversion=float(config.ir[c].reversion),
                    sigma=as_sigma(sigmas.get(c, config.ir[c].volatility)),
                    volatility_type=config.ir[c].volatility_type)
        for c in config.currencies)
    fx = tuple(
        FxComponent(currency=c, spot=market.fx_spot(c, base), sigma=as_sigma(config.fx_volatilities[c]))
        for c in config.currencies[1:])
    eq = []
    for name, vol in config.equity_volatilities.items():
        if name not in market.equities:
            raise KeyError(f"equity {name!r} has a simulation volatility but no market data")
        data = market.equities[name]
        dividend = data.dividend_curve
        eq.append(EqComponent(
            name=name, currency=data.currency, spot=float(data.spot), sigma=as_sigma(vol),
            forecast_curve=ZeroCurve.from_config(market.currency(data.currency).discount_curve),
            dividend_curve=(ZeroCurve.flat(0.0, [0.0, 1.0]) if dividend is None
                            else ZeroCurve.from_config(dividend))))
    names = ([f"IR:{c}" for c in config.currencies] + [f"FX:{c}{base}" for c in config.currencies[1:]]
             + [f"EQ:{e.name}" for e in eq])
    correlation = np.eye(len(names))
    for (a, b), rho in config.correlations.items():
        if a not in names or b not in names:
            raise KeyError(f"correlation ({a}, {b}) names a factor not simulated; factors are {names}")
        i, j = names.index(a), names.index(b)
        correlation[i, j] = correlation[j, i] = float(rho)
    return CrossAssetModel(ir=ir, fx=fx, eq=tuple(eq), correlation=correlation)


def grid_times(market: Market, config: CamConfig) -> np.ndarray:
    """`[0, t_1, ..., t_D]`: ACT/365 times of the grid dates from the as-of date."""
    if not config.dates[0] > market.asof:
        raise ValueError(f"simulation dates must be after the as-of date {market.asof}; first is {config.dates[0]}")
    return np.concatenate([[0.0], [TIME_AXIS_DAY_COUNTER.yearFraction(market.asof, d) for d in config.dates]])


def simulate(market: Market, config: CamConfig, model: Optional[CrossAssetModel] = None,
             precision: Precision = Precision()) -> ScenarioMarket:
    """Scenario market for `config`: Sobol normals with a Brownian bridge over the grid,
    the CAM's exact step moments, and the model-implied curves on every path and date.

    `model` defaults to `build_cross_asset_model(market, config)` (uncalibrated volatilities);
    pass a calibrated one to use it.

    `precision` sets the simulation and market stages (`engine.precision`; cast points 1 to
    3 of docs/planning/details/precision.md §6.3): the shocks are generated and bridged at
    `simulation.compute` and stored; the states are evolved at `simulation.compute` from the
    loaded shocks and stored; the market is built at `market.compute` from the loaded states
    and returned stored at `market.storage`, its tenor grid kept at `market.compute` (it has
    no scenario axis: it is the curves' coordinates, not scenario data). The step moments and
    the path-independent parts of the curves are computed in float64 and cast. A scaled
    storage format keeps block scales along each array's scenario axis (axis 1 of the shocks
    `[T, S, d]`, axis 0 of everything else). The shocks and states are not kept: the market
    records the formats they were stored in (`ScenarioMarket.simulation_formats`)."""
    require_precision("simulate's precision", precision)
    sim, mkt = precision.simulation, precision.market
    model = model or build_cross_asset_model(market, config)
    times = grid_times(market, config)
    moments = step_moments(model, times)
    compute = sim.compute_dtype
    # Split across the host's devices along the paths; the rest of the run follows (I-61).
    normals = shard_scenarios(generate_sobol_normals(config.samples, len(config.dates), model.dimension, compute,
                                                     seed=config.seed), axis=1)
    shocks = precision.store(apply_brownian_bridge(normals, jnp.asarray(times, dtype=compute)), sim.storage,
                             "shocks", axis=1)
    cast = lambda a: jnp.asarray(a, dtype=compute)  # noqa: E731
    states = precision.store(evolve_states(cast(model.initial_state()),
                                           (cast(moments.transition), cast(moments.drift), cast(moments.cholesky)),
                                           load(shocks, compute)), sim.storage, "states")
    index_curves = {
        name: (ccy, ZeroCurve.from_config(curve))
        for ccy in config.currencies
        for name, curve in market.currency(ccy).index_curves.items()
    }
    scenarios = build_scenario_market(model, market.asof, config.dates, load(states, mkt.compute_dtype),
                                      config.curve_tenors, index_curves)
    scenarios.simulation_formats = {"shocks": format_name(shocks), "states": format_name(states)}
    arrays = itertools.count()  # names each market array, in map_arrays' fixed order
    return scenarios.map_arrays(lambda a: precision.store(a, mkt.storage, f"market/{next(arrays)}"),
                                grid=lambda grid: grid)
