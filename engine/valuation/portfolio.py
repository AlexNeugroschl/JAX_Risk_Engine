"""
A portfolio on today's market and on the simulated scenario market: ORE's
`ValuationEngine::buildCube` (OREAnalytics/orea/engine/valuationengine.cpp) with its default
engines (plan T-8 to T-16).

Every trade is priced with its t=0 method on each path and date: swaps with
`DiscountingSwapEngine`, Europeans with the configured engine (`BlackMultiLegOptionEngine`, the
default, or Jamshidian on its Hull-White model), Bermudans and Americans with their own
calibrated LGM on the grid, bonds by discounting. The valuation reads only the scenario
market, so it is the same whichever model (`CamConfig.ir`) simulated it. Swaptions are wrapped as
ORE wraps them (`engine.valuation.options`), so an exercised physical option becomes its swap
and a cash-settled one leaves the portfolio. Paid cashflows drop out and fixings follow
`FixingManager` (`engine.valuation.legs`). NPVs are in the base currency, converted with the
path's FX rate, and not deflated (ORE's cube stores NPVs; the numeraire travels beside them).

Trades name their currency and index; the market supplies the curves and volatilities and the
pricing configuration the models (audit A-3, I-63). A trade valued on another date than the
market's is refused.
"""
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
import ORE

from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig, _build_ore_swap as _option_underlying
from engine.instruments.european_swaption import SwaptionConfig, _build_ore_swap as _european_underlying
from engine.instruments.swap import SwapConfig, _build_ore_swap as _swap_underlying
from engine.instruments.treasury import BondConfig, _remaining_cashflows
from engine.market import Market, index_name
from engine.models.ore_builders import TIME_AXIS_DAY_COUNTER
from engine.simulation.scenario_market import ScenarioMarket
from engine.valuation.bermudan import bermudan_cube, bermudan_value, contract_exercise_dates
from engine.valuation.config import PricingConfig
from engine.valuation.context import PricingContext, from_market
from engine.valuation.european import european_cube, european_terms, european_value, variance_on_path
from engine.valuation.jamshidian import jamshidian_cube, jamshidian_value, validate_jamshidian
from engine.valuation.legs import Legs, legs_cube, legs_of, path_fixings, path_schedule, today_npv
from engine.valuation.options import effective_steps, underlying_start, wrap

Trade = Union[SwapConfig, SwaptionConfig, BermudanSwaptionConfig, AmericanSwaptionConfig, BondConfig]


@dataclass
class PortfolioValuation:
    """t=0 NPVs `[T]` and the cube `[S, D, T]`, both in the base currency, in trade order."""
    today: List[float]
    cube: jax.Array
    warnings: List[str] = field(default_factory=list)


def validate_trades(trades: Sequence[Trade], market: Market, pricing: PricingConfig = PricingConfig()) -> None:
    """Refuse what the market path cannot price as specified: a trade valued on another date
    than the market's, a trade whose curves or volatilities the market lacks, or one its
    configured engine refuses."""
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
        if isinstance(cfg, SwaptionConfig) and pricing.european == "Jamshidian":
            validate_jamshidian(cfg)


def reads_swaption_vols(cfg: Trade, pricing: PricingConfig) -> bool:
    """Whether the trade's engine reads the market's swaption volatilities: a European on the
    Bachelier engine, a Bermudan/American calibrated to them."""
    if isinstance(cfg, SwaptionConfig):
        return pricing.european == "Bachelier"
    if isinstance(cfg, (BermudanSwaptionConfig, AmericanSwaptionConfig)):
        return _engine(cfg, pricing).calibration != "None"
    return False


def value_portfolio(trades: Sequence[Trade], market: Market, scenarios: ScenarioMarket, base_currency: str,
                    pricing: PricingConfig = PricingConfig(), decay: str = "ForwardVariance") -> PortfolioValuation:
    """Every trade today and on every path and date (see the module docstring)."""
    validate_trades(trades, market, pricing)
    fixings = _index_fixings(trades, market, scenarios)
    today, columns = [], []
    for cfg in trades:
        currency = cfg.currency
        spot = market.fx_spot(currency, base_currency)
        fx_path = 1.0 if currency == base_currency else scenarios.fx[currency]
        value, cube = _value_trade(cfg, market, scenarios, fixings, pricing, decay)
        today.append(float(value) * spot)
        columns.append(cube * fx_path)
    cube = jnp.stack(columns, axis=-1) if columns else jnp.zeros((scenarios.num_paths, len(scenarios.dates), 0))
    return PortfolioValuation(today=today, cube=cube)


def value_today(trades: Sequence[Trade], market: Market, base_currency: str,
                pricing: PricingConfig = PricingConfig()) -> List[float]:
    """t=0 NPVs only (no simulation), in the base currency."""
    validate_trades(trades, market, pricing)
    context = from_market(market)
    return [value_on(cfg, context, pricing) * market.fx_spot(cfg.currency, base_currency) for cfg in trades]


def _index_fixings(trades, market: Market, scenarios: ScenarioMarket) -> Dict[str, jax.Array]:
    """FixingManager's path fixings for every index a trade references."""
    tenors = {index_name(c.currency, c.index_tenor_months): c.index_tenor_months
              for c in trades if not isinstance(c, BondConfig)}
    return {name: path_fixings(tenor, market.asof, scenarios.dates, scenarios.times, scenarios.index[name])
            for name, tenor in tenors.items()}


def value_on(cfg: Trade, context: PricingContext, pricing: PricingConfig) -> float:
    """One trade's NPV in its currency on the context's date, with its configured engine."""
    currency = cfg.currency
    if isinstance(cfg, BondConfig):
        if context.date >= cfg.maturity_date:
            # Rolled to or past maturity (the Theta date): nothing left to value; a flow paid on
            # that date is ORE's excluded settlement-date flow, counted by Theta's paid flows.
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
    if isinstance(cfg, SwaptionConfig):
        if pricing.european == "Jamshidian":
            return float(jamshidian_value(cfg, context, pricing.jamshidian))
        return float(european_value(cfg, context))
    if isinstance(cfg, (BermudanSwaptionConfig, AmericanSwaptionConfig)):
        return bermudan_value(cfg, _engine(cfg, pricing), context)
    raise TypeError(f"no market-path pricer for {type(cfg).__name__}")


def _engine(cfg, pricing: PricingConfig):
    return pricing.american if isinstance(cfg, AmericanSwaptionConfig) else pricing.bermudan


def _value_trade(cfg: Trade, market: Market, sm: ScenarioMarket, fixings, pricing: PricingConfig, decay: str):
    """(t=0 NPV, `[S, D]` cube) of one trade in its own currency."""
    currency = cfg.currency
    disc = sm.discount[currency]
    today = value_on(cfg, from_market(market), pricing)
    if isinstance(cfg, BondConfig):
        legs = bond_legs(cfg)
        return today, legs_cube(legs, path_schedule(legs, market.asof, sm.dates), sm.times, disc, disc,
                                jnp.zeros((sm.num_paths, len(sm.dates)), dtype=disc.log_discounts.dtype))
    name = index_name(currency, cfg.index_tenor_months)
    index, index_fixings = sm.index[name], fixings[name]

    def swap_cube(legs: Legs):
        return legs_cube(legs, path_schedule(legs, market.asof, sm.dates), sm.times, disc, index, index_fixings)

    if isinstance(cfg, SwapConfig):
        return today, swap_cube(legs_of(_swap_underlying(cfg), cfg.payer, market.asof, cfg.fixings))
    if isinstance(cfg, SwaptionConfig):
        terms = european_terms(cfg, market.asof)
        schedule = path_schedule(terms.legs, market.asof, sm.dates)
        if pricing.european == "Jamshidian":
            alive = np.asarray([terms.expiry > d for d in sm.dates])
            option = jamshidian_cube(terms, pricing.jamshidian, schedule, sm.times, disc, index, index_fixings, alive)
        else:
            surface = market.swaption_vols(currency)
            variances = [variance_on_path(terms, surface, market.asof, d, decay) for d in sm.dates]
            option = european_cube(terms, schedule, sm.times, disc, index, index_fixings, variances)
        swap = _european_underlying(cfg)
        exercises, fixings_history = (cfg.exercise_date,), {}
    else:
        option = bermudan_cube(cfg, _engine(cfg, pricing), market, sm, index_fixings, decay, pricing.recalibrate)
        swap = _option_underlying(cfg)
        exercises, fixings_history = contract_exercise_dates(cfg), cfg.fixings
    underlyings = [swap_cube(_underlying_legs(swap, cfg.payer, market.asof, fixings_history, e)) for e in exercises]
    steps = effective_steps(exercises, market.asof, sm.dates)
    return today, wrap(option, underlyings, steps, physical=cfg.settlement == "Physical")


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
