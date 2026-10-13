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
from engine.market_simulation.scenario_market import ScenarioMarket
from engine.models.lgm import Sigma
from engine.pricing.config import LgmSwaptionEngineConfig, reference_grid_dates
from engine.pricing.context import PricingContext, simulation_market_today
from engine.pricing.legs import PathFixings
from engine.pricing.lgm_grid import (
    _backward_induction_arrays,
    _build_grid_schedule,
    grid_value,
    prepare_bermudan,
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
    `calibration="None"`, or with no exercise left), in the path curves' dtype. A calibration
    that misses its basket is refused, naming the trade."""
    dtype = scenarios.discount[cfg.currency].log_discounts.dtype
    built = calibrate_on(cfg, engine, simulation_market_today(market, scenarios.tenors))
    return Sigma.flat(engine.volatility, dtype=dtype) if built is None else built.sigma.astype(dtype)


def bermudan_cube(cfg: OptionConfig, engine: LgmSwaptionEngineConfig, market: Market, scenarios: ScenarioMarket,
                  fixings: PathFixings) -> jax.Array:
    """`[S, D]` NPVs of the option (its continuation value; exercise is `engine.pricing.
    options.wrap`'s job) on every path and date, with the LGM ORE built it with
    (`built_sigma`); 0 once no exercise date is after the date.

    `fixings`: FixingManager's path fixings for the trade's index, read as rates `[S, D]` (the
    rollback is not in difference form yet: it computes in the curves' one format)."""
    fixings = fixings.rates
    asof = market.asof
    history = dict(cfg.fixings)
    grid_serials = np.array([d.serialNumber() for d in scenarios.dates], dtype=np.int64)
    disc_curves = scenarios.discount[cfg.currency]
    index_curves = scenarios.index[index_name(cfg.currency, cfg.index_tenor_months)]
    fixing_serials = _fixing_serials(cfg)
    exercises = contract_exercise_dates(cfg)
    sigma = None
    columns = []
    for j, date in enumerate(scenarios.dates):
        if not any(d > date for d in exercises):
            columns.append(jnp.zeros(scenarios.num_paths, dtype=fixings.dtype))
            continue
        sigma = built_sigma(cfg, engine, market, scenarios) if sigma is None else sigma
        fixed_on_path = {ORE.Date(int(s)): 0.0 for s in fixing_serials if asof.serialNumber() <= s < date.serialNumber()}
        dated = dataclasses.replace(cfg, evaluation_date=date, fixings={**history, **fixed_on_path})
        prepared = prepared_option(dated, engine, engine.volatility, built_on=asof)
        schedule = _build_grid_schedule(prepared)
        known = _known_rates(prepared, history, asof, grid_serials, fixings)
        columns.append(_rollback_every_path(prepared, schedule, disc_curves.on_date(j), index_curves.on_date(j),
                                            sigma, known))
    return jnp.stack(columns, axis=1)


def _fixing_serials(cfg: OptionConfig):
    return [ORE.as_floating_rate_coupon(c).fixingDate().serialNumber() for c in underlying_swap(cfg).floatingLeg()]


def _known_rates(prepared, history: Mapping[ORE.Date, float], asof: ORE.Date, grid_serials: np.ndarray,
                 fixings: jax.Array) -> jax.Array:
    """`[S, Ncf]`: each kept coupon's known rate on every path: its history before the as-of
    date, FixingManager's path fixing after it (unknown coupons get 0; they are projected)."""
    serials = prepared.float_fixing_serials
    step = np.minimum(np.searchsorted(grid_serials, serials, side="right"), grid_serials.size - 1)
    past = serials < asof.serialNumber()
    values = np.array([history.get(ORE.Date(int(s)), 0.0) if p else 0.0 for s, p in zip(serials, past)])
    path_known = prepared.float_is_known & ~past
    return jnp.where(path_known[None, :], fixings[:, step], jnp.asarray(values, dtype=fixings.dtype))


@jax.jit
def _rollback_every_path(prepared, schedule, disc: DiscountCurve, index: DiscountCurve, sigma: Sigma,
                         known: jax.Array) -> jax.Array:
    """The grid engine's t=0 value on every path: the rollback vmapped over the path curves and
    known fixings, the volatility the one ORE built the trade with."""

    def one_path(disc_logs, index_logs, known_rates):
        swap = dataclasses.replace(
            prepared,
            curve=DiscountCurve(disc.times, disc_logs), index_curve=DiscountCurve(index.times, index_logs),
            sigma=sigma, float_known_rates=known_rates,
            notional=jnp.asarray(prepared.notional, dtype=disc_logs.dtype),
            fixed_amounts=jnp.asarray(prepared.fixed_amounts, dtype=disc_logs.dtype))
        _, values = _backward_induction_arrays(swap, schedule)
        return values[-1, values.shape[1] // 2]

    return jax.vmap(one_path)(disc.log_discounts, index.log_discounts, known)
