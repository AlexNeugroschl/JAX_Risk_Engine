"""
Bermudan and American swaptions in the pipeline: the trade's own LGM, calibrated as ORE's
`LGMGridSwaptionEngineBuilder` calibrates it, priced with the grid engine
(`engine.pricing.lgm_grid`, parity with ORE ~1e-11) -- today and on every
simulated path and date (plan T-13, 5.3, 5.4).

Calibration basket (`calibration_basket`, plan V-5): one `SwaptionHelper` per alive exercise
date (an American: per `ReferenceCalibrationGrid` date inside its window), each into the
underlying's maturity, struck at the deal strike (`CoterminalDealStrike`: the fixed rate
less the float spread) or ATM, at most one per interval of the reference grid
(`IrModelBuilder::buildSwaptionBasket`). The LGM's volatility is bootstrapped to it
(`engine.calibration.ore_lgm`), with the engine's constant reversion.

Every calibration on a market of the as-of date (today's, the simulation market's, a bumped
one) is checked as ORE's `LgmBuilder` checks it, against the engine's `bootstrap_tolerance`
(ORE's `Tolerance`), and a miss is refused naming the trade (`CalibrationError`), as ORE fails
without `ContinueOnCalibrationError` (I-73).

The model's H is shifted to vanish at the shift horizon (`shift_horizon_time`): ORE's
`ShiftHorizon` (default 0.5) of the time from the trade's build to the underlying's maturity.

On a path (`built_sigma`, `bermudan_cube`), as ORE's exposure simulation values the trade. ORE
builds every trade once, on the simulation market of the as-of date (its curves sampled at the
simulation tenors), the trade's LGM calibrated there, and its `XvaAnalytic` never recalibrates
it: its single-threaded `ValuationEngine` is given no model builders and its multi-threaded one
`recalibrateModels = false` (OREAnalytics/orea/app/analytics/xvaanalytic.cpp). So on every
path and date the grid engine prices with that calibration, its bucket times as built (times
from the evaluation date, which moves), on the path's curves with the fixings FixingManager
stored, the trade as ORE built it on the as-of date: the shift horizon measured from it
(`initParametrization`) and an American's window starting no earlier than the day after it, so
exercisable on a path date inside the window. This is ORE's cube on ORE's own paths to 2e-11
(`tests/test_ore_xva_parity.py`). Until 2026-10-12 the engine recalibrated the LGM on every path
and date, 0.5-4% above ORE's values (I-49); recalibrating on a path is no ORE analytic's (ORE
recalibrates per scenario only on the as-of date, in its sensitivity and historical P&L
analytics), so it is not offered (decision A-24).
"""
import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import Mapping, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import ORE

from engine.calibration.ore_lgm import BasketInstrument, bootstrap_sigma, build_basket, require_accepted
from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig, underlying_swap
from engine.market_data.curves import DiscountCurve
from engine.market_data.market import Market, index_name
from engine.market_simulation.scenario_market import ScenarioCurves, ScenarioMarket
from engine.market_simulation.sharding import scenario_device_count
from engine.models.lgm import Sigma
from engine.precision import Arithmetic, add_level, deviation_from
from engine.pricing.config import LgmSwaptionEngineConfig, reference_grid_dates
from engine.pricing.context import PricingContext, simulation_market_today
from engine.pricing.legs import Deviations, PathFixings
from engine.pricing.lgm_grid import (
    blocked, grid_value, one_shape, option_values, path_blocks, prepare_bermudan, working_bytes_per_path,
)

OptionConfig = Union[BermudanSwaptionConfig, AmericanSwaptionConfig]


def contract_exercise_dates(cfg: OptionConfig) -> Tuple[ORE.Date, ...]:
    """The dates ORE's option wrapper and trade builder work with: a Bermudan's exercise
    dates; an American's window, first and last day (`ExerciseBuilder`)."""
    if isinstance(cfg, AmericanSwaptionConfig):
        return cfg.first_exercise_date, cfg.last_exercise_date
    return tuple(cfg.exercise_dates)


def basket_expiries(cfg: OptionConfig, engine: LgmSwaptionEngineConfig, built_on: ORE.Date) -> Tuple[ORE.Date, ...]:
    """Expiries of the calibration helpers of the trade built on `built_on`: the exercise dates
    after it; for an American, the reference grid dates from it inside the window
    [max(built_on + 1, first), last). At most one per reference-grid interval, the first
    (`buildSwaptionBasket`)."""
    grid_dates = reference_grid_dates(built_on, engine.reference_calibration_grid)
    if isinstance(cfg, AmericanSwaptionConfig):
        first = max(built_on + 1, cfg.first_exercise_date)
        candidates = [d for d in grid_dates if first <= d < cfg.last_exercise_date]
    else:
        candidates = list(cfg.exercise_dates)
    grid = [d.serialNumber() for d in grid_dates]
    kept, last = [], None
    for expiry in (d for d in candidates if d > built_on):
        slot = int(np.searchsorted(grid, expiry.serialNumber(), side="left"))
        if slot == len(grid) or last is None or slot > last:
            kept.append(expiry)
            if slot < len(grid):
                last = slot
    return tuple(kept)


def calibration_basket(cfg: OptionConfig, engine: LgmSwaptionEngineConfig,
                       built_on: ORE.Date) -> Tuple[BasketInstrument, ...]:
    """The helpers of the trade built on `built_on` (see the module docstring)."""
    expiries = basket_expiries(cfg, engine, built_on)
    strike = cfg.fixed_rate - cfg.floating_spread if engine.strategy == "CoterminalDealStrike" else None
    return tuple(build_basket(built_on, list(expiries), [cfg.maturity_date] * len(expiries), engine.swap_index,
                              [strike] * len(expiries)))


def shift_horizon_time(cfg: OptionConfig, shift_horizon: float, built_on: ORE.Date) -> float:
    """The LGM's shift horizon in years for a trade built on `built_on`: `shift_horizon` (ORE's
    `ShiftHorizon`, a fraction) of the ActualActual(ISDA) time from `built_on` to the underlying's
    maturity, its last payment date (`LGMBermudanAmericanSwaptionEngineBuilder::model`, which
    measures it from the evaluation date to the trade's `maturity()`)."""
    maturity = underlying_swap(cfg).maturityDate()
    return ORE.ActualActual(ORE.ActualActual.ISDA).yearFraction(built_on, maturity) * shift_horizon


def prepared_option(cfg: OptionConfig, engine: LgmSwaptionEngineConfig, sigma, curve=None, index_curve=None,
                    built_on: Optional[ORE.Date] = None):
    """The grid engine's prepared trade, as ORE built it on `built_on` (the trade's own date
    by default) and values it on the trade's evaluation date: the engine's model (its
    reversion, `sigma` and the shift horizon from `built_on`), its grid settings, an American's
    window from `built_on`, on `curve`/`index_curve` (set later per path when omitted)."""
    built_on = cfg.evaluation_date if built_on is None else built_on
    return prepare_bermudan(cfg, reversion=engine.reversion, sigma=sigma,
                            shift_horizon=shift_horizon_time(cfg, engine.shift_horizon, built_on),
                            n_per_std=engine.n_per_std, std_devs=engine.std_devs,
                            exercise_time_steps_per_year=engine.exercise_time_steps_per_year, built_on=built_on,
                            curve=curve, index_curve=index_curve)


def _on_date(cfg: OptionConfig, context: PricingContext) -> OptionConfig:
    """The trade valued on the context's date, with the context's extra fixings."""
    extra = context.fixings.get(index_name(cfg.currency, cfg.index_tenor_months), {})
    return dataclasses.replace(cfg, evaluation_date=context.date, fixings={**cfg.fixings, **extra})


@dataclass
class Calibration:
    """A calibration on one date: bucket times and values, and each helper's market and model
    value (`LgmCalibrationInfo`)."""
    sigma: Sigma
    market: np.ndarray
    model: np.ndarray


def calibrate_on(cfg: OptionConfig, engine: LgmSwaptionEngineConfig, context: PricingContext) -> Optional[Calibration]:
    """The trade's LGM on the context's date, as a trade built on that date (None with
    `calibration="None"`, or when no exercise is left); a calibration that misses ORE's
    acceptance is refused, naming the trade (`engine.calibration.ore_lgm.CalibrationError`)."""
    if engine.calibration == "None":
        return None
    basket = calibration_basket(cfg, engine, context.date)
    if not basket:
        return None
    disc, index = context.curves(cfg.currency, index_name(cfg.currency, cfg.index_tenor_months))
    vols = np.array([context.volatility(cfg.currency, b.vol_option_time, b.vol_swap_length) for b in basket])
    result = bootstrap_sigma(basket, disc, index, vols, engine.reversion, engine.solver)
    require_accepted(result, engine.bootstrap_tolerance, f"{cfg.product} {cfg.trade_id!r} on {context.date}")
    return Calibration(Sigma(jnp.asarray(result.times), result.values), np.asarray(result.market),
                       np.asarray(result.model))


def bermudan_value(cfg: OptionConfig, engine: LgmSwaptionEngineConfig, context: PricingContext) -> float:
    """ORE's NPV on the context's date (0 once expired): calibrate, then the grid engine on
    the context's curves."""
    dated = _on_date(cfg, context)
    if dated.is_expired():
        return 0.0
    calibration = calibrate_on(dated, engine, context)
    sigma = Sigma.flat(engine.volatility) if calibration is None else calibration.sigma
    disc, index = context.curves(cfg.currency, index_name(cfg.currency, cfg.index_tenor_months))
    return float(grid_value(prepared_option(dated, engine, sigma, disc, index)))


def built_sigma(cfg: OptionConfig, engine: LgmSwaptionEngineConfig, market: Market,
                scenarios: ScenarioMarket) -> Sigma:
    """The LGM volatility ORE builds the trade with and keeps on every path (see the module
    docstring): calibrated on the simulation market of the as-of date (the engine's own with
    `calibration="None"`, or with no exercise left), in float64, as every calibration (A-10) and
    the model the induction's level and grid are on. A calibration that misses its basket is
    refused, naming the trade."""
    built = calibrate_on(cfg, engine, simulation_market_today(market, scenarios.tenors))
    return Sigma.flat(engine.volatility) if built is None else built.sigma


def bermudan_cube(cfg: OptionConfig, engine: LgmSwaptionEngineConfig, market: Market, scenarios: ScenarioMarket,
                  fixings: PathFixings, arithmetic: Optional[Arithmetic] = None) -> jax.Array:
    """`[S, D]` NPVs of the option (its continuation value; exercise is `engine.pricing.
    options.wrap`'s job) on every path and date, with the LGM ORE built it with
    (`built_sigma`); 0 once no exercise date is after the date.

    In difference form (F-07, `engine.pricing.lgm_grid.option_values`): on each date the
    induction runs on the curves' level (`ScenarioCurves.level`, float64) and on each path's
    deviation from it, the discount and index curves' and the coupons' path fixings' (`fixings`,
    FixingManager's for the trade's index), in `arithmetic` (the curves' own format, float64 or
    float32, by default), whose accumulate format the values are in. Every date is the trade
    padded to one shape (`one_shape`), so the dates share one compiled program."""
    asof = market.asof
    disc_curves = scenarios.discount[cfg.currency]
    index_curves = scenarios.index[index_name(cfg.currency, cfg.index_tenor_months)]
    arithmetic = arithmetic or Arithmetic.of(disc_curves.log_discounts.dtype)
    exercises = contract_exercise_dates(cfg)
    alive = [j for j, date in enumerate(scenarios.dates) if any(d > date for d in exercises)]
    columns = [jnp.zeros(scenarios.num_paths, dtype=arithmetic.dtype)] * len(scenarios.dates)
    if not alive:
        return jnp.stack(columns, axis=1)
    sigma = built_sigma(cfg, engine, market, scenarios)
    history = dict(cfg.fixings)
    fixing_serials = _fixing_serials(cfg)
    grid_serials = np.array([d.serialNumber() for d in scenarios.dates], dtype=np.int64)
    fixing_levels = np.asarray(fixings.level)
    prepared, known = [], []
    for j in alive:
        date = scenarios.dates[j]
        fixed_on_path = {ORE.Date(int(s)): 0.0 for s in fixing_serials
                         if asof.serialNumber() <= s < date.serialNumber()}
        dated = dataclasses.replace(cfg, evaluation_date=date, fixings={**history, **fixed_on_path})
        option = prepared_option(dated, engine, sigma, built_on=asof)
        level, step, path_known = _known_rates(option, history, asof, grid_serials, fixing_levels)
        prepared.append(dataclasses.replace(option, float_known_rates=level))
        known.append((step, path_known))
    shaped = one_shape(prepared)
    blocks = path_blocks(scenarios.num_paths, working_bytes_per_path(shaped[0][0], jnp.dtype(arithmetic.dtype).itemsize),
                         scenario_device_count(scenarios.num_paths))
    for j, (option, rows), (step, path_known) in zip(alive, shaped, known):
        width = (0, len(option.float_pay_times) - len(step))
        columns[j] = _date_column(option, rows, disc_curves, index_curves, j, fixings.deviation, np.pad(step, width),
                                  np.pad(path_known, width), arithmetic, blocks)
    return jnp.stack(columns, axis=1)


@partial(jax.jit, static_argnames=("arithmetic", "blocks"))
def _date_column(option, rows, disc: ScenarioCurves, index: ScenarioCurves, j, fixing_deviations: jax.Array,
                 step: np.ndarray, path_known: np.ndarray, arithmetic: Arithmetic, blocks: int) -> jax.Array:
    """`[S]`: the option on date `j` of the scenario curves on every path, as one program for
    every date (the date is traced): the induction on the curves' level and each path's
    deviation from it (`option_values`), the known coupons' deviations FixingManager's path
    fixings' at their grid `step` where `path_known`; the paths in `blocks` blocks
    (`engine.pricing.lgm_grid.blocked`), so the induction's memory stays within its budget at any
    path count."""
    def split(curves):
        level = curves.level[j]
        return DiscountCurve(curves.tenor_times[j], level), deviation_from(level[None], curves.log_discounts[:, j])

    (disc_level, disc_moved), (index_level, index_moved) = split(disc), split(index)
    known = jnp.where(path_known[None, :], fixing_deviations[:, step], 0.0)
    option = dataclasses.replace(option, curve=disc_level, index_curve=index_level)
    level, moved = blocked(lambda deviations: option_values(option, rows, deviations, arithmetic),
                           Deviations(disc_moved, index_moved, known), blocks)
    return add_level(level, moved)


def _fixing_serials(cfg: OptionConfig):
    return [ORE.as_floating_rate_coupon(c).fixingDate().serialNumber() for c in underlying_swap(cfg).floatingLeg()]


def _known_rates(prepared, history: Mapping[ORE.Date, float], asof: ORE.Date, grid_serials: np.ndarray,
                 fixing_levels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each kept coupon's known rate as a level `[Ncf]`, and where its paths deviate from it:
    its history before the as-of date (no deviation), FixingManager's path fixing after it (the
    level of the fixing at its grid `step`, `PathFixings.level`, each path's deviation where
    `path_known`); 0 for an unknown coupon, which is projected. On the host."""
    serials = prepared.float_fixing_serials
    step = np.minimum(np.searchsorted(grid_serials, serials, side="right"), grid_serials.size - 1)
    past = serials < asof.serialNumber()
    history_rates = np.array([history.get(ORE.Date(int(s)), 0.0) if p else 0.0 for s, p in zip(serials, past)])
    path_known = prepared.float_is_known & ~past
    return np.where(path_known, fixing_levels[step], history_rates), step, path_known
