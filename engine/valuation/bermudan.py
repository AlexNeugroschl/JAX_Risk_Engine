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
every path at once, and the dates whose baskets have one shape at once too (`_path_sigmas`);
the grid engine then prices on the path curves with the fixings FixingManager stored. With
`recalibrate = false` the t=0 volatility is kept. A date with an exercise left but no helper
(an American's last days, after the last reference-grid date in its window) keeps the
engine's volatility, as a calibration today with no helper does.

Not yet confirmed against an ORE simulation (gate V-1, docs/planning/known-issues.md I-49): ORE keeps
the parametrization's time grid from its first build and still passes helpers whose expiry has
passed on a later date; here each date's basket holds only the exercise dates after it, with
bucket times measured from it.
"""
import dataclasses
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import ORE

from engine.calibration.ore_lgm import (
    SIGMA_BRACKET, BasketInstrument, bootstrap_sigma, build_basket, stack_instruments,
)
from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import (
    BermudanSwaptionConfig, _backward_induction_arrays, _build_grid_schedule, grid_value, prepare_bermudan,
)
from engine.market import Market, index_name
from engine.models.curves import DiscountCurve
from engine.models.lgm import Sigma
from engine.simulation.scenario_market import ScenarioCurves, ScenarioMarket
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


def prepared_option(cfg: OptionConfig, engine: LgmSwaptionEngineConfig, sigma, curve=None, index_curve=None):
    """The grid engine's prepared trade with the engine's model (its reversion and `sigma`)
    and grid settings, on `curve`/`index_curve` (set later per path when omitted)."""
    return prepare_bermudan(cfg, reversion=engine.reversion, sigma=sigma, n_per_std=engine.n_per_std,
                            std_devs=engine.std_devs, exercise_time_steps_per_year=engine.exercise_time_steps_per_year,
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
    `calibration="None"`, or when no exercise is left)."""
    if engine.calibration == "None":
        return None
    basket = calibration_basket(cfg, engine, context.date, context.date)
    if not basket:
        return None
    disc, index = context.curves(cfg.currency, index_name(cfg.currency, cfg.index_tenor_months))
    vols = np.array([context.volatility(cfg.currency, b.vol_option_time, b.vol_swap_length) for b in basket])
    result = bootstrap_sigma(basket, disc, index, vols, engine.reversion, engine.solver)
    if np.any(np.asarray(result.hit_ceiling)):
        raise ValueError(f"calibration of the {type(cfg).__name__} on {context.date} failed: a helper's market "
                         f"volatility is not attainable with a volatility in the bracket {list(SIGMA_BRACKET)}")
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


def bermudan_cube(cfg: OptionConfig, engine: LgmSwaptionEngineConfig, market: Market, scenarios: ScenarioMarket,
                  fixings: jax.Array, decay: str, recalibrate: bool = True) -> jax.Array:
    """`[S, D]` NPVs of the option (its continuation value; exercise is `engine.valuation.
    options.wrap`'s job) on every path and date; 0 once no exercise date is after the date.

    `fixings` `[S, D]`: FixingManager's path fixings for the trade's index."""
    asof = market.asof
    history = dict(cfg.fixings)
    grid_serials = np.array([d.serialNumber() for d in scenarios.dates], dtype=np.int64)
    disc_curves = scenarios.discount[cfg.currency]
    index_curves = scenarios.index[index_name(cfg.currency, cfg.index_tenor_months)]
    fixing_serials = _fixing_serials(cfg)
    sigmas = path_sigmas(cfg, engine, market, scenarios, decay, recalibrate)
    columns = []
    for j, date in enumerate(scenarios.dates):
        if j not in sigmas:
            columns.append(jnp.zeros(scenarios.num_paths, dtype=fixings.dtype))
            continue
        fixed_on_path = {ORE.Date(int(s)): 0.0 for s in fixing_serials if asof.serialNumber() <= s < date.serialNumber()}
        dated = dataclasses.replace(cfg, evaluation_date=date, fixings={**history, **fixed_on_path})
        prepared = prepared_option(dated, engine, engine.volatility)
        schedule = _build_grid_schedule(prepared)
        known = _known_rates(prepared, history, asof, grid_serials, fixings)
        columns.append(_rollback_every_path(prepared, schedule, disc_curves.on_date(j), index_curves.on_date(j),
                                            sigmas[j], known))
    return jnp.stack(columns, axis=1)


def _fixing_serials(cfg: OptionConfig):
    from engine.instruments.bermudan_swaption import _build_ore_swap
    return [ORE.as_floating_rate_coupon(c).fixingDate().serialNumber() for c in _build_ore_swap(cfg).floatingLeg()]


def path_sigmas(cfg: OptionConfig, engine: LgmSwaptionEngineConfig, market: Market, scenarios: ScenarioMarket,
                decay: str, recalibrate: bool = True) -> Dict[int, Sigma]:
    """The option's LGM volatility on each simulation date with an exercise after it (date
    index -> `Sigma`, its values `[S, n]` when recalibrated), in the path curves' dtype:
    recalibrated to the date's basket on every path (`decay` reads the basket's volatilities
    on the date), or today's (`recalibrate=false`), or the engine's own (`calibration="None"`,
    and a date with an exercise left but no helper).

    The dates whose baskets have one shape are calibrated in one call over dates and paths,
    their count padded to a power of two (the last date repeated), so a trade compiles a
    bootstrap per basket shape and power of two, whatever its dates (I-53). The reference
    solver (`"Bisection"`) keeps one date per call: it reproduces the engine's numbers before
    Newton (A-21) bit for bit, which a batch over dates cannot (XLA vectorizes another shape, and
    a bisection's last comparisons move with the residual's last bit)."""
    asof, dates = market.asof, scenarios.dates
    disc = scenarios.discount[cfg.currency]
    index = scenarios.index[index_name(cfg.currency, cfg.index_tenor_months)]
    alive = [j for j, date in enumerate(dates) if any(d > date for d in contract_exercise_dates(cfg))]
    dtype = disc.log_discounts.dtype
    flat = Sigma.flat(engine.volatility, dtype=dtype)
    if engine.calibration == "None":
        return {j: flat for j in alive}
    if not recalibrate:
        today = calibrate_on(cfg, engine, from_market(market))
        return {j: today.sigma.astype(dtype) if today is not None else flat for j in alive}
    surface = market.swaption_vols(cfg.currency)
    sigmas, groups = {}, defaultdict(list)
    baskets = {j: calibration_basket(cfg, engine, dates[j], asof) for j in alive}
    for j, basket in baskets.items():
        if basket:
            groups[_basket_shape(basket)].append(j)
        else:
            sigmas[j] = flat
    def vols(j):
        return [volatility_on_path(surface, asof, dates[j], b.vol_option_time, b.vol_swap_length, decay)
                for b in baskets[j]]

    for js in groups.values():
        if engine.solver == "Bisection":
            for j in js:
                result = bootstrap_sigma(baskets[j], disc.on_date(j), index.on_date(j), np.array(vols(j)),
                                         engine.reversion, engine.solver)
                sigmas[j] = Sigma(times=jnp.asarray(result.times, dtype=dtype), values=result.values)
            continue
        padded = js + [js[-1]] * (_power_of_two(len(js)) - len(js))
        stacked = [stack_instruments(helpers) for helpers in zip(*(baskets[j] for j in padded))]
        dates_of = np.asarray(padded)
        result = bootstrap_sigma(stacked, _on_dates(disc, dates_of), _on_dates(index, dates_of),
                                 np.array([vols(j) for j in padded]), engine.reversion, engine.solver, dated=True)
        values = _unstack(result.values)
        for g, j in enumerate(js):
            sigmas[j] = Sigma(times=jnp.asarray(result.times[g], dtype=dtype), values=values[g])  # [S, n]
    return sigmas


def _basket_shape(basket: Sequence[BasketInstrument]):
    """What fixes a basket's bootstrap programs: each helper's pytree structure (its static
    fields included) and array shapes."""
    return tuple((jax.tree_util.tree_structure(b), tuple(np.shape(x) for x in jax.tree_util.tree_leaves(b)))
                 for b in basket)


def _power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


@jax.jit
def _on_dates(curves: ScenarioCurves, dates: jax.Array) -> DiscountCurve:
    """The curves on `dates`, batched over dates then paths: times `[D, K]`, logs `[D, S, K]`
    (jitted, as `_unstack`: one program per shape, not one per eager operation, I-81)."""
    return DiscountCurve(times=curves.tenor_times[dates],
                         log_discounts=jnp.swapaxes(curves.log_discounts[:, dates], 0, 1))


@jax.jit
def _unstack(values: jax.Array):
    """`values` `[D, ...]` as D arrays."""
    return tuple(values)


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
    """The grid engine's t=0 value on every path: the rollback vmapped over the path curves,
    volatilities and known fixings."""
    batched_sigma = sigma.values.ndim == 2

    def one_path(disc_logs, index_logs, sigma_values, known_rates):
        swap = dataclasses.replace(
            prepared,
            curve=DiscountCurve(disc.times, disc_logs), index_curve=DiscountCurve(index.times, index_logs),
            sigma=Sigma(times=sigma.times, values=sigma_values), float_known_rates=known_rates,
            notional=jnp.asarray(prepared.notional, dtype=disc_logs.dtype),
            fixed_amounts=jnp.asarray(prepared.fixed_amounts, dtype=disc_logs.dtype))
        _, values = _backward_induction_arrays(swap, schedule)
        return values[-1, values.shape[1] // 2]

    return jax.vmap(one_path, in_axes=(0, 0, 0 if batched_sigma else None, 0))(
        disc.log_discounts, index.log_discounts, sigma.values, known)
