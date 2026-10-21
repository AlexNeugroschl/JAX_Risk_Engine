"""
A portfolio on today's market and on the simulated scenario market: ORE's
`ValuationEngine::buildCube` (OREAnalytics/orea/engine/valuationengine.cpp) with its default
engines (plan T-8 to T-16).

Every trade is priced with its t=0 method on each path and date: swaps with
`DiscountingSwapEngine`, Europeans with the configured engine (`BlackMultiLegOptionEngine`, the
default, or Jamshidian on its Hull-White model), Bermudans and Americans with their own LGM on
the grid, calibrated where ORE builds them (the simulation market of the as-of date) and kept,
as ORE's exposure simulation keeps it (`engine.pricing.bermudan`), bonds by discounting. The valuation reads only the scenario
market, so it is the same whichever model (`CamConfig.ir`) simulated it. Swaptions are wrapped as
ORE wraps them (`engine.pricing.options`), so an exercised physical option becomes its swap,
a cash-settled one leaves the portfolio, and a short option is minus the long one, the holder
deciding the exercise (ORE's `LongShort`, I-101). Paid cashflows drop out and fixings follow
`FixingManager` (`engine.pricing.legs`). NPVs are in the base currency, converted with the
path's FX rate, and not deflated (ORE's cube stores NPVs; the numeraire travels beside them).

Each trade is priced on paths at its own precision, `Precision.precision_for(trade)` (an
override for its id, else for its product, else the pricing stage; decision A-15), in its
arithmetic (`engine.precision.Arithmetic`): the scenario market is loaded at the accumulate
format, the kernels' products run in the compute format, and the trade's cube column is stored
at the storage format. Every product is priced in difference form at every precision (F-07;
the Bermudan/American rollback since 2026-10-13). t=0 values are float64 (A-10). An option's
exercise decisions on each path and date are the wrapper's (`PortfolioValuation.exercised`).
`value_paths` prices the paths alone, as the paired float64 sample of a reduced-precision run
does (decision A-13).

Trades name their currency and index; the market supplies the curves and volatilities and the
pricing configuration the models (audit A-3, I-63). A trade valued on another date than the
market's is refused.
"""
import dataclasses
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union, get_args

import jax
import jax.numpy as jnp
import numpy as np
import ORE

from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig, underlying_swap as _option_underlying
from engine.instruments.european_swaption import SwaptionConfig, underlying_swap as _european_underlying
from engine.instruments.swap import SwapConfig, underlying_swap as _swap_underlying
from engine.instruments.treasury import BondConfig, _remaining_cashflows
from engine.market_data.day_counts import TIME_AXIS_DAY_COUNTER
from engine.market_data.market import Market, index_name
from engine.market_simulation.scenario_market import ScenarioMarket
from engine.precision import Arithmetic, Precision, Stored, load, require_precision
from engine.pricing.bermudan import bermudan_cube, bermudan_value, contract_exercise_dates
from engine.pricing.config import PricingConfig
from engine.pricing.context import PricingContext, from_market
from engine.pricing.european import (
    check_exercise_into_whole_periods, european_cube, european_terms, european_value, variance_on_path,
)
from engine.pricing.jamshidian import jamshidian_cube, jamshidian_value, validate_jamshidian
from engine.pricing.legs import Legs, PathFixings, legs_cube, legs_of, path_fixings, path_schedule, today_npv
from engine.pricing.options import effective_steps, position_sign, underlying_start, wrap

Trade = Union[SwapConfig, SwaptionConfig, BermudanSwaptionConfig, AmericanSwaptionConfig, BondConfig]

#: The option trades: each has a position (`long_short`, ORE's `LongShort`), and its engines
#: value the long option (`engine.pricing.options`).
OPTIONS = (SwaptionConfig, BermudanSwaptionConfig, AmericanSwaptionConfig)

#: Every product the pipeline prices, by the name its trades carry (`product`): the keys of
#: `Precision.by_product`.
PRODUCTS: Tuple[str, ...] = tuple(t.product for t in get_args(Trade))

@dataclass
class PortfolioValuation:
    """t=0 NPVs `[T]` (float64) and the cube, both in the base currency, in trade order.

    columns: each trade's `[S, D]` cube column, stored at its own pricing storage format
        (`Precision.precision_for`): a plain array, or a `Stored` (block scales along the
        paths) in a scaled format.
    exercised: each option's exercise state, by trade id, `[S, D]` bool: whether the holder has
        exercised on the path on or before the date (`engine.pricing.options.wrap`).
    """
    today: List[float]
    columns: List[Union[jax.Array, Stored]]
    num_paths: int
    num_dates: int
    warnings: List[str] = field(default_factory=list)
    exercised: Dict[str, jax.Array] = field(default_factory=dict)

    @property
    def cube(self) -> jax.Array:
        """`[S, D, T]`: the columns side by side, in their shared format if it is float64 or
        float32, else loaded at float64 (exact: float64 holds every format's values, block
        scales applied)."""
        if not self.columns:
            return jnp.zeros((self.num_paths, self.num_dates, 0))
        dtypes = {c.dtype for c in self.columns}
        shared = len(dtypes) == 1 and not any(isinstance(c, Stored) for c in self.columns)
        dtype = dtypes.pop() if shared else jnp.float64
        return jnp.stack([load(c, dtype) for c in self.columns], axis=-1)

    def float64_cube(self) -> jax.Array:
        """The requested output cube, decoded and assembled in one compiled program.

        Unlike `load(self.cube, float64)`, this allocates neither a stack at the stored
        dtype nor a list of separately decoded columns beside the final array (I-105).
        Exposure reductions read `columns` directly and never call this method.
        """
        if not self.columns:
            return jnp.zeros((self.num_paths, self.num_dates, 0), dtype=jnp.float64)
        return _float64_cube(tuple(self.columns))


@jax.jit
def _float64_cube(columns):
    return jnp.stack([load(c, jnp.float64) for c in columns], axis=-1)


def require_unique_ids(trades: Sequence[Trade]) -> None:
    """Trade ids are unique in a portfolio, as ORE requires: every per-trade result and
    precision override is keyed by them. The requests check it (`PortfolioRequest`,
    `MarketRiskRequest`); below them a `by_trade` override applies to every trade of its id."""
    duplicates = sorted(i for i, n in Counter(cfg.trade_id for cfg in trades).items() if n > 1)
    if duplicates:
        raise ValueError(f"trade ids must be unique in a portfolio; repeated: {duplicates}")


#: The reporting currency of a run that names none and has no simulation to take one from
#: (`RunConfig.reporting_currency`, `MarketRiskRequest.base_currency`). ORE's `ore.xml`
#: requires its `baseCurrency`; the engine defaults it.
DEFAULT_REPORTING_CURRENCY = "USD"


def reporting_fx_spots(trades: Sequence[Trade], market: Market, currency: str, field: str) -> List[float]:
    """Each trade's conversion into the reporting `currency` at the market's as-of FX spot:
    units of `currency` per unit of the trade's currency, 1.0 for a trade in it.

    Refuses, before any work, a reporting currency the market has no market for and a trade
    whose currency has no spot against it (quoted either way, never a cross), naming the trade,
    the pair and `field`, the setting that chose the currency."""
    try:
        market.currency(currency)
    except KeyError as exc:
        raise KeyError(f"{field}={currency!r}: {exc.args[0]}") from None
    spots = []
    for cfg in trades:
        try:
            spots.append(market.fx_spot(cfg.currency, currency))
        except KeyError:
            raise KeyError(f"trade {cfg.trade_id!r} is in {cfg.currency} and the reporting currency is {currency} "
                           f"({field}), but the market has no {cfg.currency}{currency} spot: give fx_spots "
                           f"{cfg.currency + currency!r} (or {currency + cfg.currency!r}), or report in "
                           f"{cfg.currency}") from None
    return spots


def validate_trades(trades: Sequence[Trade], market: Market, pricing: PricingConfig = PricingConfig(),
                    precision: Precision = Precision()) -> None:
    """Refuse what the pipeline cannot price as specified: a precision override that names
    no trade or product (`Precision.check_overrides`), a trade valued on another date than the
    market's, a trade whose curves or volatilities the market lacks, or one its configured
    engine refuses."""
    precision.check_overrides(trades, PRODUCTS)
    for cfg in trades:
        label = f"trade {cfg.trade_id!r} ({type(cfg).__name__})"
        if cfg.evaluation_date != market.asof:
            raise ValueError(f"{label}: evaluation_date {cfg.evaluation_date} is not the market's as-of date "
                             f"{market.asof}; book trades on the valuation date (audit A-4)")
        currency = cfg.currency
        try:
            market.currency(currency)
            if not isinstance(cfg, BondConfig):
                market.index_curve(currency, index_name(currency, cfg.index_tenor_months))
            if reads_swaption_vols(cfg, pricing):
                market.swaption_vols(currency)
        except KeyError as exc:
            raise KeyError(f"{label}: {exc.args[0]}") from None
        if isinstance(cfg, SwaptionConfig) and cfg.exercise_date > market.asof:
            check_exercise_into_whole_periods(cfg)
            if pricing.european == "Jamshidian":
                validate_jamshidian(cfg)


def trade_maturity(cfg: Trade) -> ORE.Date:
    """ORE's `Trade::maturity()`, after which a trade's exposure takes no time weight
    (`ExposureCalculator`): a swap's or a bond's last date and a physically settled option's
    underlying maturity, but a cash-settled option's last exercise date, on which it settles
    (`Swaption::build`, step 8). Until 2026-10-07 every option took its underlying's, so a
    cash-settled option's time-weighted EPE kept accruing after it had settled."""
    if isinstance(cfg, SwaptionConfig) and cfg.settlement == "Cash":
        return cfg.exercise_date
    if isinstance(cfg, (BermudanSwaptionConfig, AmericanSwaptionConfig)) and cfg.settlement == "Cash":
        return max(contract_exercise_dates(cfg))
    if isinstance(cfg, BondConfig):
        return cfg.last_payment_date
    return cfg.maturity_date


def reads_swaption_vols(cfg: Trade, pricing: PricingConfig) -> bool:
    """Whether the trade's engine reads the market's swaption volatilities: a European on the
    Bachelier engine, a Bermudan/American calibrated to them."""
    if isinstance(cfg, SwaptionConfig):
        return pricing.european == "Bachelier"
    if isinstance(cfg, (BermudanSwaptionConfig, AmericanSwaptionConfig)):
        return _engine(cfg, pricing).calibrates
    return False


def value_portfolio(trades: Sequence[Trade], market: Market, scenarios: ScenarioMarket, base_currency: str,
                    pricing: PricingConfig = PricingConfig(), decay: str = "ForwardVariance",
                    precision: Precision = Precision()) -> PortfolioValuation:
    """Every trade today and on every path and date (see the module docstring): `value_today`
    and `value_paths` in one."""
    require_precision("value_portfolio's precision", precision)
    validate_trades(trades, market, pricing, precision)
    columns, exercised = _columns(trades, market, scenarios, base_currency, pricing, decay, precision)
    return PortfolioValuation(today=_today(trades, market, base_currency, pricing), columns=columns,
                              num_paths=scenarios.num_paths, num_dates=len(scenarios.dates), exercised=exercised)


def value_today(trades: Sequence[Trade], market: Market, base_currency: str,
                pricing: PricingConfig = PricingConfig(), context: Optional[PricingContext] = None) -> List[float]:
    """t=0 NPVs only (no simulation), in the base currency, float64 (decision A-10): on today's
    market, or on another market of the as-of date, `context` (the simulation market's,
    `engine.pricing.context.simulation_market_today`, where an exposure profile starts)."""
    validate_trades(trades, market, pricing)
    return _today(trades, market, base_currency, pricing, context)


def value_paths(trades: Sequence[Trade], market: Market, scenarios: ScenarioMarket, base_currency: str,
                pricing: PricingConfig = PricingConfig(), decay: str = "ForwardVariance",
                precision: Precision = Precision()) -> List[Union[jax.Array, Stored]]:
    """Every trade on every path and date only (no t=0 value): `PortfolioValuation.columns`.
    The paired float64 sample of a reduced-precision run reprices its paths with it
    (`engine.run.pipeline`)."""
    require_precision("value_paths' precision", precision)
    validate_trades(trades, market, pricing, precision)
    return _columns(trades, market, scenarios, base_currency, pricing, decay, precision)[0]


def _today(trades, market: Market, base_currency: str, pricing: PricingConfig,
           context: Optional[PricingContext] = None) -> List[float]:
    context = context or from_market(market)
    spots = reporting_fx_spots(trades, market, base_currency, "base_currency")
    return [float(value_on(cfg, context, pricing)) * spot for cfg, spot in zip(trades, spots)]


def _columns(trades, market: Market, scenarios: ScenarioMarket, base_currency: str, pricing: PricingConfig,
             decay: str, precision: Precision) -> Tuple[List[Union[jax.Array, Stored]], Dict[str, jax.Array]]:
    """Cast point 4 of docs/planning/details/precision.md §6.3, per trade: the scenario market
    is loaded at the accumulate format of `precision.precision_for(trade)` (once per format, with
    its path fixings), the trade priced in its arithmetic, and its cube column stored at its
    storage format, named by the trade's id (`Precision.store`); and each option's exercise state
    by trade id (`PortfolioValuation.exercised`). Each trade is a phase of its own
    (`trade_pricing_phase`), labelled in a profiler trace and a boundary a run can stop at."""
    # At call time: importing `engine.run` imports the pipeline, which imports this module.
    from engine.run.trace import trade_pricing_phase

    loaded = {}  # accumulate dtype -> (the scenario market at it, its path fixings)
    columns, exercised = [], {}
    for i, cfg in enumerate(trades):
        with trade_pricing_phase(i, cfg):
            stage = precision.precision_for(cfg)
            arithmetic = precision.arithmetic_for(stage, f"values/{cfg.trade_id}")
            if arithmetic.dtype not in loaded:
                at_dtype = scenarios.map_arrays(lambda a: load(a, arithmetic.dtype), grid=lambda grid: grid)
                loaded[arithmetic.dtype] = at_dtype, _index_fixings(trades, market, at_dtype)
            sm, fixings = loaded[arithmetic.dtype]
            fx_path = 1.0 if cfg.currency == base_currency else sm.fx[cfg.currency]
            cube, state = _path_values(cfg, market, sm, fixings, pricing, decay, arithmetic)
            columns.append(precision.store(cube * fx_path, stage.storage, f"values/{cfg.trade_id}"))
            if state is not None:
                exercised[cfg.trade_id] = state
    return columns, exercised


def _index_fixings(trades, market: Market, scenarios: ScenarioMarket) -> Dict[str, PathFixings]:
    """FixingManager's path fixings for every index a trade references."""
    tenors = {index_name(c.currency, c.index_tenor_months): c.index_tenor_months
              for c in trades if not isinstance(c, BondConfig)}
    return {name: path_fixings(tenor, market.asof, scenarios.dates, scenarios.times, scenarios.index[name])
            for name, tenor in tenors.items()}


def value_on(cfg: Trade, context: PricingContext, pricing: PricingConfig) -> float:
    """One trade's NPV in its currency on the context's date, with its configured engine; an
    option's in its position, a short minus the long (`engine.pricing.options.position_sign`)."""
    if isinstance(cfg, OPTIONS):
        return position_sign(cfg) * _long_option_on(cfg, context, pricing)
    currency = cfg.currency
    if isinstance(cfg, BondConfig):
        if context.date >= cfg.last_payment_date:
            # Rolled to or past maturity (the Theta date): nothing left to value; a flow paid on
            # that date has occurred, so it leaves Theta as it leaves ORE's (I-116).
            return 0.0
        disc = context.discount[currency]
        return float(today_npv(bond_legs(dataclasses.replace(cfg, evaluation_date=context.date)), context.date,
                               disc, disc))
    name = index_name(currency, cfg.index_tenor_months)
    if isinstance(cfg, SwapConfig):
        disc, index = context.curves(currency, name)
        fixings = {**cfg.fixings, **context.fixings.get(name, {})}
        return float(today_npv(legs_of(_swap_underlying(cfg), cfg.payer, context.date, fixings), context.date,
                               disc, index))
    raise TypeError(f"no pricer for {type(cfg).__name__}")


def _long_option_on(cfg, context: PricingContext, pricing: PricingConfig) -> float:
    """The long option's NPV on the context's date, by its configured engine."""
    if isinstance(cfg, SwaptionConfig):
        if pricing.european == "Jamshidian":
            return float(jamshidian_value(cfg, context, pricing.jamshidian))
        return float(european_value(cfg, context))
    return bermudan_value(cfg, _engine(cfg, pricing), context)


def _engine(cfg, pricing: PricingConfig):
    return pricing.american if isinstance(cfg, AmericanSwaptionConfig) else pricing.bermudan


def _path_values(cfg: Trade, market: Market, sm: ScenarioMarket, fixings, pricing: PricingConfig, decay: str,
                 arithmetic: Arithmetic):
    """The `[S, D]` cube column of one trade in its own currency, in `arithmetic`, and an
    option's exercise state `[S, D]` (`wrap`; None for a swap or a bond)."""
    currency = cfg.currency
    disc = sm.discount[currency]
    if isinstance(cfg, BondConfig):
        legs = bond_legs(cfg)
        return legs_cube(legs, path_schedule(legs, market.asof, sm.dates), sm.times, disc, disc, None,
                         arithmetic), None
    name = index_name(currency, cfg.index_tenor_months)
    index, index_fixings = sm.index[name], fixings[name]

    def swap_cube(legs: Legs):
        return legs_cube(legs, path_schedule(legs, market.asof, sm.dates), sm.times, disc, index, index_fixings,
                         arithmetic)

    if isinstance(cfg, SwapConfig):
        return swap_cube(legs_of(_swap_underlying(cfg), cfg.payer, market.asof, cfg.fixings)), None
    if isinstance(cfg, SwaptionConfig) and not cfg.exercise_date > market.asof:
        # Expired: no exercise is left, so the option is worth nothing on any date, as ORE builds
        # no option for it (`Swaption::build`, step 6), and its exercise-into coupons are not read.
        option = jnp.zeros((sm.num_paths, len(sm.dates)), dtype=arithmetic.dtype)
        swap = _european_underlying(cfg)
        exercises, fixings_history = (cfg.exercise_date,), {}
    elif isinstance(cfg, SwaptionConfig):
        terms = european_terms(cfg, market.asof)
        schedule = path_schedule(terms.legs, market.asof, sm.dates)
        if pricing.european == "Jamshidian":
            alive = np.asarray([terms.expiry_serial > d.serialNumber() for d in sm.dates])
            option = jamshidian_cube(terms, pricing.jamshidian, schedule, sm.times, disc, index, index_fixings, alive,
                                     arithmetic)
        else:
            surface = market.swaption_vols(currency)
            variances = [variance_on_path(terms, surface, market.asof, d, decay) for d in sm.dates]
            option = european_cube(terms, schedule, sm.times, disc, index, index_fixings, variances, arithmetic)
        swap = _european_underlying(cfg)
        exercises, fixings_history = (cfg.exercise_date,), {}
    else:
        option = bermudan_cube(cfg, _engine(cfg, pricing), market, sm, index_fixings, arithmetic)
        swap = _option_underlying(cfg)
        exercises, fixings_history = contract_exercise_dates(cfg), cfg.fixings
    underlyings = [swap_cube(_underlying_legs(swap, cfg.payer, market.asof, fixings_history, e)) for e in exercises]
    steps = effective_steps(exercises, market.asof, sm.dates)
    return wrap(option, underlyings, steps, physical=cfg.settlement == "Physical", sign=position_sign(cfg))


def _underlying_legs(swap: ORE.VanillaSwap, payer: bool, asof: ORE.Date, fixings, exercise: ORE.Date) -> Legs:
    """The swap an exercise on `exercise` enters (`Swaption::buildUnderlyingSwaps`)."""
    starts = lambda leg, as_coupon: [as_coupon(c).accrualStartDate() for c in leg]  # noqa: E731
    return legs_of(swap, payer, asof, fixings,
                   underlying_start(starts(swap.fixedLeg(), ORE.as_fixed_rate_coupon), exercise),
                   underlying_start(starts(swap.floatingLeg(), ORE.as_floating_rate_coupon), exercise))


def bond_legs(cfg: BondConfig) -> Legs:
    """A bond's remaining cashflows as a received fixed leg (no floating leg): discounting them
    is ORE's `DiscountingRiskyBondEngine` with no credit curve and no security spread (plan
    V-9), and on a path paid flows drop out like any leg's."""
    flows = _remaining_cashflows(cfg)
    t_of = lambda d: TIME_AXIS_DAY_COUNTER.yearFraction(cfg.evaluation_date, d)  # noqa: E731
    empty, empty_int = np.zeros(0), np.zeros(0, dtype=np.int64)
    return Legs(
        payer=False,
        fixed_pay_serial=np.array([d.serialNumber() for d, _ in flows], dtype=np.int64),
        fixed_pay=np.array([t_of(d) for d, _ in flows]),
        fixed_amount=np.array([a * cfg.face_amount for _, a in flows]),
        fixed_bps=np.zeros(len(flows)), fixed_rate=0.0,
        float_pay_serial=empty_int, float_pay=empty, float_nominal=empty, float_accrual=empty, float_spread=0.0,
        forecast_start=empty, forecast_end=empty, spanning=empty, fixing_serial=empty_int, history=empty,
    )
