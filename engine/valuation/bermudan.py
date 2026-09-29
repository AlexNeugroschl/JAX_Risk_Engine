"""
Bermudan and American swaptions on the market path: the trade's own LGM, calibrated as ORE's
`LGMGridSwaptionEngineBuilder` calibrates it, priced with the grid engine
(`engine.instruments.bermudan_swaption`, parity with ORE ~1e-11) -- today and on every
simulated path and date (plan T-13, 5.3, 5.4).

Calibration basket (`calibration_basket`, plan V-5): one `SwaptionHelper` per alive exercise
date (an American: per `ReferenceCalibrationGrid` date inside its window), each into the
underlying's maturity, struck at the deal strike (`CoterminalDealStrike`: the fixed rate
less the float spread) or ATM, at most one per interval of the reference grid
(`IrModelBuilder::buildSwaptionBasket`). The LGM's volatility is bootstrapped to it
(`engine.calibration.ore_lgm`), with the engine's constant reversion.

On a path (ORE's `ValuationEngine` with `recalibrate = true`, its default): on each date the
basket is rebuilt from that date, its volatilities read off the t=0 surface seen from that
date (`DynamicSwaptionVolatilityMatrix`), and the LGM bootstrapped to the path's own curves,
every path at once; the grid engine then prices on the path curves with the fixings
FixingManager stored. With `recalibrate = false` the t=0 volatility is kept.

Not yet confirmed against an ORE simulation (gate V-1, docs/known-issues.md I-49): ORE keeps
the parametrization's time grid from its first build and still passes helpers whose expiry has
passed on a later date; here each date's basket holds only the exercise dates after it, with
bucket times measured from it.
"""
import dataclasses
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import ORE

from engine.calibration.ore_lgm import BasketInstrument, bootstrap_sigma, build_basket
from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import (
    BermudanSwaptionConfig, _backward_induction_arrays, _build_grid_schedule, _run_backward_induction,
    prepare_bermudan,
)
from engine.market import Market, SwaptionVolSurface, ZeroCurveConfig, index_name
from engine.models.curves import DiscountCurve
from engine.models.lgm import Sigma
from engine.simulation.scenario_market import ScenarioMarket
from engine.valuation.config import LgmSwaptionEngineConfig, reference_grid_dates
from engine.valuation.context import PricingContext, from_market
from engine.valuation.european import volatility_on_path

OptionConfig = Union[BermudanSwaptionConfig, AmericanSwaptionConfig]


def contract_exercise_dates(cfg: OptionConfig) -> Tuple[ORE.Date, ...]:
    """The dates ORE's option wrapper and trade builder work with: a Bermudan's exercise
    dates; an American's window, first and last day (`ExerciseBuilder`)."""
    if isinstance(cfg, AmericanSwaptionConfig):
        return cfg.first_exercise_date, cfg.last_exercise_date
    return tuple(cfg.exercise_dates)


def basket_expiries(cfg: OptionConfig, engine: LgmSwaptionEngineConfig, reference: ORE.Date,
                    asof: ORE.Date) -> Tuple[ORE.Date, ...]:
    """Expiries of the calibration helpers on `reference`: the exercise dates after it; for an
    American, the reference grid dates (from the as-of date, as the builder sets them up)
    inside the window [max(asof + 1, first), last) that are after `reference`. At most one per
    reference-grid interval, the first (`buildSwaptionBasket`)."""
    if isinstance(cfg, AmericanSwaptionConfig):
        first = max(asof + 1, cfg.first_exercise_date)
        candidates = [d for d in reference_grid_dates(asof, engine.reference_calibration_grid)
                      if first <= d < cfg.last_exercise_date]
    else:
        candidates = list(cfg.exercise_dates)
    grid = [d.serialNumber() for d in reference_grid_dates(reference, engine.reference_calibration_grid)]
    kept, last = [], None
    for expiry in (d for d in candidates if d > reference):
        slot = int(np.searchsorted(grid, expiry.serialNumber(), side="left"))
        if slot == len(grid) or last is None or slot > last:
            kept.append(expiry)
            if slot < len(grid):
                last = slot
    return tuple(kept)


def calibration_basket(cfg: OptionConfig, engine: LgmSwaptionEngineConfig, reference: ORE.Date,
                       asof: ORE.Date) -> Tuple[BasketInstrument, ...]:
    """The trade's helpers on `reference` (see the module docstring)."""
    expiries = basket_expiries(cfg, engine, reference, asof)
    strike = cfg.fixed_rate - cfg.floating_spread if engine.strategy == "CoterminalDealStrike" else None
    return tuple(build_basket(reference, list(expiries), [cfg.maturity_date] * len(expiries), engine.swap_index,
                              [strike] * len(expiries)))


#: A curve for `prepare_bermudan`, which needs one; every pricer here replaces it with the
#: context's or the path's curves before pricing.
_PLACEHOLDER_CURVE = ZeroCurveConfig([0.0, 1.0], [0.0, 0.0])


def _model_bound(cfg: OptionConfig, engine: LgmSwaptionEngineConfig, sigma) -> OptionConfig:
    """The trade with the engine's model and grid settings (curves are placeholders)."""
    fields = dict(hw_a=engine.reversion, hw_sigma=sigma, initial_zero_curve=_PLACEHOLDER_CURVE,
                  index_zero_curve=_PLACEHOLDER_CURVE, n_per_std=engine.n_per_std, std_devs=engine.std_devs)
    if isinstance(cfg, AmericanSwaptionConfig):
        fields["exercise_time_steps_per_year"] = engine.exercise_time_steps_per_year
    return dataclasses.replace(cfg, **fields)


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
    `calibration="None"`, or when no exercise is left)."""
    if engine.calibration == "None":
        return None
    basket = calibration_basket(cfg, engine, context.date, context.date)
    if not basket:
        return None
    disc, index = context.curves(cfg.currency, index_name(cfg.currency, cfg.index_tenor_months))
    vols = np.array([context.volatility(cfg.currency, b.vol_option_time, b.vol_swap_length) for b in basket])
    result = bootstrap_sigma(basket, disc, index, vols, engine.reversion)
    if np.any(np.asarray(result.hit_ceiling)):
        raise ValueError(f"calibration of the {type(cfg).__name__} on {context.date} failed: a helper's market "
                         f"volatility is not attainable with a volatility in the bisection bracket")
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
    prepared = prepare_bermudan(_model_bound(dated, engine, sigma))
    swap = dataclasses.replace(prepared, curve=disc, index_curve=index, hw_sigma=sigma)
    return float(_run_backward_induction(swap, condition_times=[]).value_at_t0)


def bermudan_cube(cfg: OptionConfig, engine: LgmSwaptionEngineConfig, market: Market, scenarios: ScenarioMarket,
                  fixings: jax.Array, decay: str, recalibrate: bool = True) -> jax.Array:
    """`[S, D]` NPVs of the option (its continuation value; exercise is `engine.valuation.
    options.wrap`'s job) on every path and date; 0 once no exercise date is after the date.

    `fixings` `[S, D]`: FixingManager's path fixings for the trade's index."""
    asof = market.asof
    surface = market.swaption_vols(cfg.currency)
    today = calibrate_on(cfg, engine, from_market(market)) if not recalibrate else None
    history = dict(cfg.fixings)
    grid_serials = np.array([d.serialNumber() for d in scenarios.dates], dtype=np.int64)
    disc_curves = scenarios.discount[cfg.currency]
    index_curves = scenarios.index[index_name(cfg.currency, cfg.index_tenor_months)]
    columns = []
    for j, date in enumerate(scenarios.dates):
        if not any(d > date for d in contract_exercise_dates(cfg)):
            columns.append(jnp.zeros(scenarios.num_paths, dtype=fixings.dtype))
            continue
        disc = disc_curves.on_date(j)
        index = index_curves.on_date(j)
        fixed_on_path = {ORE.Date(int(s)): 0.0 for s in _fixing_serials(cfg) if asof.serialNumber() <= s < date.serialNumber()}
        placeholder = _model_bound(cfg, engine, engine.volatility)
        dated = dataclasses.replace(placeholder, evaluation_date=date, fixings={**history, **fixed_on_path})
        sigma = _path_sigma(cfg, engine, surface, asof, date, disc, index, decay, today)
        prepared = prepare_bermudan(dated)
        schedule = _build_grid_schedule(prepared, [])
        known = _known_rates(prepared, history, asof, grid_serials, fixings)
        columns.append(_rollback_every_path(prepared, schedule, disc, index, sigma, known))
    return jnp.stack(columns, axis=1)


def _fixing_serials(cfg: OptionConfig):
    from engine.instruments.bermudan_swaption import _build_ore_swap
    return [ORE.as_floating_rate_coupon(c).fixingDate().serialNumber() for c in _build_ore_swap(cfg).floatingLeg()]


def _path_sigma(cfg, engine, surface: SwaptionVolSurface, asof, date, disc: DiscountCurve, index: DiscountCurve,
                decay: str, today: Optional[Calibration]) -> Sigma:
    """The LGM volatility on `date`: recalibrated to the basket on every path, or today's
    (`recalibrate=false`, or `calibration="None"`: the engine's fixed volatility)."""
    if engine.calibration == "None":
        return Sigma.flat(engine.volatility)
    if today is not None:
        return today.sigma
    basket = calibration_basket(cfg, engine, date, asof)
    vols = np.array([volatility_on_path(surface, asof, date, b.vol_option_time, b.vol_swap_length, decay)
                     for b in basket])
    result = bootstrap_sigma(basket, disc, index, vols, engine.reversion)
    return Sigma(times=jnp.asarray(result.times), values=result.values)   # values [S, n]


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


def _rollback_every_path(prepared, schedule, disc: DiscountCurve, index: DiscountCurve, sigma: Sigma,
                         known: jax.Array) -> jax.Array:
    """The grid engine's t=0 value on every path: the rollback vmapped over the path curves,
    volatilities and known fixings."""
    batched_sigma = sigma.values.ndim == 2

    def one_path(disc_logs, index_logs, sigma_values, known_rates):
        swap = dataclasses.replace(
            prepared,
            curve=DiscountCurve(disc.times, disc_logs), index_curve=DiscountCurve(index.times, index_logs),
            hw_sigma=Sigma(times=sigma.times, values=sigma_values), float_known_rates=known_rates,
            notional=jnp.asarray(prepared.notional, dtype=disc_logs.dtype),
            fixed_amounts=jnp.asarray(prepared.fixed_amounts, dtype=disc_logs.dtype))
        _, values = _backward_induction_arrays(swap, schedule)
        return values[-1, values.shape[1] // 2]

    return jax.vmap(one_path, in_axes=(0, 0, 0 if batched_sigma else None, 0))(
        disc.log_discounts, index.log_discounts, sigma.values, known)
