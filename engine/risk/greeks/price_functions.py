"""
Each trade's t=0 price as a pure JAX function of the market it reads (the node values of its
curves and, where its engine reads them, its currency's swaption volatility quotes), with the
trade's configured engine: what the AD Greeks differentiate (`engine.risk.greeks.ad`) and
market risk revalues under shocked curves (`engine.risk.market`).

    trade_price_function(cfg, market, pricing, dtype, tenors=None) -> TradePriceFunction
        .curves  the curves it reads, as `(kind, name)`: `("discount", ccy)` and, for every
                 trade but a bond, `("index", index name)`
        .nodes   each curve as a function of its node values (`PillarNodes` or `TenorNodes`,
                 below), and `.rates` their values on the market
        .quotes  the currency's ATM normal volatilities `[option tenors, swap tenors]` where the
                 engine reads them (`reads_swaption_vols`), else None; float64 whatever the
                 working dtype: what is read from them (a European's variance, a calibration)
                 is computed in float64, as a model input (A-10), and rounded once to it
        .price   `price(*rates, quotes=None)`, one node-value array per curve in that order
                 (the market's quotes by default)
        .pricer, .terms
                 the same function as data: `price` is `pricer(terms, curves, quotes)` on the
                 curves `nodes` make of `rates`, with `pricer` a module-level function (or an
                 object equal for equal settings) and `terms` the trade's data as a pytree, so a
                 caller jits it with `pricer` static and compiles once per product and shape,
                 not per trade or call (`engine.risk.greeks.ad`)
        .pnl     `pnl(terms, curves, quotes, moves, arithmetic)`, each scenario's change of the
                 price when the node values of `curves` move by `moves.shifts` (`[S, K]` per
                 curve), every product in difference form (F-07): computed as the deviation from
                 the level in the matrix products of `arithmetic` (the legs'
                 `engine.pricing.legs.leg_values`, the grid engine's
                 `engine.pricing.lgm_grid.option_values`), so a move of nothing is exactly 0
                 and the legs' cancellation stays in the level
        .calibration
                 for an option whose engine calibrates its LGM, `calibration(terms, curves,
                 quotes, shifts)`: the LGM recalibrated under each scenario, `(values [S, n],
                 misses [S])` in float64 (A-10), which `pnl` then reads as `moves.models`; None
                 for every other trade

**The curves' nodes.** A curve is a function of its node values, and the representation is the
analytic's: market risk shocks the market curves' pillar zero rates (`PillarNodes`: ORE's
`ZeroCurve`, linear in the zero rate, `engine.pricing.context.from_market`); the Greeks move the
zero rates of ORE's sensitivity market at its tenors (`TenorNodes`, `tenors` given: log-linear
discount factors at the tenors' times from the as-of date, flat forward beyond,
`engine.pricing.context.simulation_market_today`), which is what ORE's sensitivity analysis
shifts, so a derivative there is the limit of ORE's bump Greek as its shift vanishes (I-78).

The engines are the valuation pipeline's own (`engine.pricing`), so the price at the market's
values is the trade's t=0 NPV on that market: swaps and bonds by `legs_npv`, Europeans by
`black_multileg_npv` (the normal volatility read from the quotes: the surface is linear in
them, `SwaptionVolSurface.weights`) or `jamshidian_npv`, Bermudans and Americans by the grid
engine.

**A calibrated Bermudan/American calibrates inside its price function**, as ORE's sensitivity
analysis and historical P&L recalibrate its model builder under every shift and scenario
(I-78, I-112): the bootstrap (`engine.calibration.lgm.bootstrap`) on the curves it is given
and the quotes, in float64 whatever the curves' dtype (decision A-10), then the grid engine with
the calibrated volatility. Its derivatives are the implicit function theorem's through every
bucket, so a Delta moves the calibration with the curve and a Vega moves it with the quotes.
Under market-risk scenarios the calibration is a first pass (`calibration`, refused by the
revaluation where a scenario's misses ORE's acceptance), and the P&L of each scenario its own
induction (`recalibrated_pnl`): the scenario's model on the base curves is its level, the
curves' move its deviation in the compute format. An option on a fixed model (`calibration =
"None"`) prices with it everywhere; its P&L's level is shared by every scenario.

An option's price is in its position (`long_short`, ORE's `LongShort`): its engine prices the
long option and `OnPosition` multiplies by the position's sign, carried in its `Positioned`
terms as data, as ORE's `OptionWrapper` multiplies its engine's NPV. A short option's price, its
derivatives and its P&L are minus the long one's, and a long and a short option of one shape are
one compiled program (I-101).

The curves' dtype is the working dtype (the calibration's is float64). An option expired on the
market's date prices to a constant 0 (ORE's `isExpired`), so its sensitivities are 0.
"""
import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from engine.calibration.lgm import (
    BasketInstrument, basket_expiries, bootstrap, calibrated_values, calibration_misses,
)
from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig, underlying_swap as _swap_underlying
from engine.instruments.treasury import BondConfig
from engine.market_data.curves import DiscountCurve, ZeroCurve, curve_dtype, with_node_deviation
from engine.market_data.market import VOL_DAY_COUNTER, Market, index_name
from engine.models.lgm import Sigma
from engine.precision import Arithmetic, add_level
from engine.pricing.bermudan import calibrate_on, calibration_basket, prepared_option
from engine.pricing.config import JamshidianEngineConfig, LgmSwaptionEngineConfig, PricingConfig
from engine.pricing.context import PricingContext, from_market, simulation_market_today
from engine.pricing.cube import bond_legs, reads_swaption_vols
from engine.pricing.european import (
    EuropeanTerms, bachelier_on_legs, black_multileg_npv, european_legs, european_terms, on_path,
)
from engine.pricing.jamshidian import jamshidian_discounts, jamshidian_npv, jamshidian_on_discounts
from engine.pricing.legs import Deviations, Legs, leg_values, legs_npv, legs_of, today_schedule
from engine.pricing.lgm_grid import _Induction, _PreparedBermudan, grid_value, induction_rows, option_values
from engine.pricing.options import position_sign

#: A market curve: `("discount", currency)` or `("index", index name)`.
CurveKey = Tuple[str, str]


# ---------------------------------------------------------------------------
# The curves' nodes
# ---------------------------------------------------------------------------
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PillarNodes:
    """A market curve on its pillars (ORE's `ZeroCurve`, linear in the zero rate): its node
    values are the pillar zero rates, market risk's factors."""
    times: Any

    def curve(self, rates) -> ZeroCurve:
        return ZeroCurve(pillar_times=self.times, pillar_rates=rates)

    def tree_flatten(self):
        return (self.times,), None

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TenorNodes:
    """A curve of ORE's sensitivity market: log-linear discount factors at the tenors' times
    (`times[0] = 0`), flat forward beyond (`DiscountCurve`); its node values are the zero rates at
    the tenors after t=0, `z_k = -ln P(t_k) / t_k`, what ORE's sensitivity analysis shifts (an
    absolute shift of `z_k` moves `ln P(t_k)` by `-shift t_k` alone)."""
    times: Any

    def curve(self, rates) -> DiscountCurve:
        return DiscountCurve(self.times, jnp.concatenate([jnp.zeros_like(rates[..., :1]), -rates * self.times[1:]],
                                                         axis=-1))

    def tree_flatten(self):
        return (self.times,), None

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)


def nodes_of(curve, dtype) -> Tuple[Any, np.ndarray]:
    """`(nodes, rates)` of a context's curve (see the module docstring), in `dtype`."""
    if isinstance(curve, DiscountCurve):
        times = np.asarray(curve.times, dtype=np.float64)
        rates = -np.asarray(curve.log_discounts, dtype=np.float64)[1:] / times[1:]
        return TenorNodes(jnp.asarray(times, dtype=dtype)), np.asarray(rates, dtype=dtype)
    return PillarNodes(jnp.asarray(curve.pillar_times, dtype=dtype)), np.asarray(curve.pillar_rates, dtype=dtype)


def curves_on(nodes: Sequence, rates: Sequence) -> Tuple:
    """The curve each of `nodes` makes of its node values."""
    return tuple(n.curve(r) for n, r in zip(nodes, rates))


# ---------------------------------------------------------------------------
# The price function
# ---------------------------------------------------------------------------
class Moves(NamedTuple):
    """Each scenario's market-risk move, its leading axis the scenarios: the node values' shift
    of each curve `[S, K]`, and the LGM volatilities an option recalibrated under the scenario
    `[S, n]` (float64; None for every other trade)."""
    shifts: Tuple
    models: Optional[jax.Array] = None


@dataclass(frozen=True)
class TradePriceFunction:
    """A trade's price as a function of its market (see the module docstring)."""
    curves: Tuple[CurveKey, ...]
    nodes: Tuple[Any, ...]
    rates: Tuple[np.ndarray, ...]
    quotes: Optional[np.ndarray]
    pricer: Callable
    terms: Any
    pnl: Callable
    calibration: Optional[Callable] = None

    def price(self, *rates, quotes=None):
        return self.pricer(self.terms, curves_on(self.nodes, rates), self.quotes if quotes is None else quotes)


class Positioned(NamedTuple):
    """An option's terms for its long pricer, and its position's sign
    (`engine.pricing.options.position_sign`) in the working dtype: data, not a static setting."""
    sign: np.ndarray
    terms: Any


@dataclass(frozen=True)
class OnPosition:
    """An option's pricer, or P&L, in its position: `sign * long(terms, ...)` on `Positioned`
    terms, from the long option's `long` (see the module docstring). Equal for equal `long`, so
    one compiled program serves a long and a short option."""
    long: Callable

    def __call__(self, terms: Positioned, *args):
        return terms.sign * self.long(terms.terms, *args)


def in_position(cfg, long: Callable, terms, dtype) -> Tuple[OnPosition, Positioned]:
    """`(pricer, terms)` of the option `cfg` in its position, from its long pricer and terms."""
    return OnPosition(long), Positioned(np.asarray(position_sign(cfg), dtype=dtype), terms)


def curve_keys(cfg) -> Tuple[CurveKey, ...]:
    """The market curves a trade's price reads: its currency's discount curve and, but for a
    bond, its index's forwarding curve (read even by an engine that ignores it, so every
    method reports the same curves)."""
    if isinstance(cfg, BondConfig):
        return (("discount", cfg.currency),)
    return (("discount", cfg.currency), ("index", index_name(cfg.currency, cfg.index_tenor_months)))


def base_context(market: Market, tenors: Optional[Sequence[str]] = None) -> PricingContext:
    """The market a price function's base values are read from: today's, or ORE's sensitivity
    market at `tenors` (see the module docstring)."""
    return from_market(market) if tenors is None else simulation_market_today(market, tenors)


def trade_price_function(cfg, market: Market, pricing: PricingConfig = PricingConfig(), dtype=jnp.float64,
                         tenors: Optional[Sequence[str]] = None) -> TradePriceFunction:
    """The trade's t=0 price on `market` as a function of its curves' node values and its
    quotes: on the market curves' pillars, or on ORE's sensitivity market at `tenors` (see the
    module docstring). The trade must be valued on the market's as-of date."""
    keys = curve_keys(cfg)
    context = base_context(market, tenors)
    tables = {"discount": context.discount, "index": context.index}
    nodes, rates = zip(*(nodes_of(tables[kind][name], dtype) for kind, name in keys))
    quotes = (np.asarray(market.swaption_vols(cfg.currency).vols, dtype=np.float64)
              if reads_swaption_vols(cfg, pricing) else None)
    asof = market.asof

    def priced(pricer, terms, pnl, calibration=None):
        return TradePriceFunction(keys, nodes, rates, quotes, pricer, terms, pnl, calibration)

    def positioned(long, terms, long_pnl, calibration=None):
        pricer, terms = in_position(cfg, long, terms, dtype)
        return priced(pricer, terms, OnPosition(long_pnl), calibration)

    if isinstance(cfg, BondConfig):
        return priced(_legs_price, _legs_terms(bond_legs(cfg), asof, dtype), _legs_pnl)
    if isinstance(cfg, SwapConfig):
        legs = legs_of(_swap_underlying(cfg), cfg.payer, asof, cfg.fixings)
        return priced(_legs_price, _legs_terms(legs, asof, dtype), _legs_pnl)
    if isinstance(cfg, SwaptionConfig):
        if not cfg.exercise_date > asof:
            return priced(_zero_price, None, _zero_pnl)
        if pricing.european == "Jamshidian":
            return positioned(JamshidianPrice(pricing.jamshidian), european_terms(cfg, asof).astype(dtype),
                              JamshidianPnl(pricing.jamshidian))
        return positioned(bachelier_price, bachelier_terms(cfg, market, dtype), bachelier_pnl)
    if isinstance(cfg, (BermudanSwaptionConfig, AmericanSwaptionConfig)):
        if cfg.is_expired():
            return priced(_zero_price, None, _zero_pnl)
        engine = pricing.american if isinstance(cfg, AmericanSwaptionConfig) else pricing.bermudan
        terms = option_terms(cfg, engine, context, market, dtype)
        if terms.calibration is None:
            return positioned(option_price, terms, option_pnl)
        price = CalibratedOptionPrice(engine.reversion, engine.solver)
        return positioned(price, terms, recalibrated_pnl, ScenarioCalibration(price))
    raise TypeError(f"no price function for {type(cfg).__name__}")


def _zero_price(_terms, curves, quotes):
    """An expired option's price: 0, still a function of the curves and the quotes."""
    zero = jnp.zeros((), dtype=curve_dtype(curves[0]))
    return zero * sum(jnp.sum(leaf) for leaf in jax.tree_util.tree_leaves((curves, quotes)))


def _zero_pnl(_terms, curves, quotes, moves: Moves, arithmetic: Arithmetic):
    """An expired option's P&L: 0 in every scenario."""
    return jnp.zeros(moves.shifts[0].shape[:-1], dtype=arithmetic.dtype)


def _deviations(terms_legs: Legs, shifts) -> Deviations:
    """The `Deviations` of a leg whose curves' node values move by `shifts` (one curve: a bond,
    whose one curve is both) and whose known fixings do not."""
    discount, index = shifts[0], shifts[-1]
    return Deviations(discount, index, jnp.zeros(discount.shape[:-1] + terms_legs.float_pay.shape, discount.dtype))


def _changed(level_value, shocked_value, shifts):
    """`shocked - level` per scenario, exactly 0 where nothing moved (rather than the difference
    of one value evaluated twice, which XLA may round apart)."""
    return jnp.where(_moved(shifts), shocked_value - level_value, 0.0)


def _moved(shifts):
    """`[S]`: whether any node of any curve moves in the scenario."""
    moved = jnp.zeros(shifts[0].shape[:-1], dtype=bool)
    for shift in shifts:
        moved = moved | jnp.any(shift != 0, axis=-1)
    return moved


# ---------------------------------------------------------------------------
# Swaps and bonds
# ---------------------------------------------------------------------------
class _LegsTerms(NamedTuple):
    """Fixed and floating coupons with their date rules on the as-of date (`today_npv`)."""
    legs: Legs
    alive_fixed: np.ndarray
    alive_float: np.ndarray
    projected: np.ndarray
    known_rates: np.ndarray


def _legs_terms(legs: Legs, asof, dtype) -> _LegsTerms:
    today = today_schedule(legs, asof)
    return _LegsTerms(legs.astype(dtype), today.alive_fixed, today.alive_float, today.projected,
                      np.asarray(today.known_rates, dtype=dtype))


def _legs_price(terms: _LegsTerms, curves, _quotes):
    """ORE's t=0 NPV of the coupons; a bond's coupons are forecast off its one curve."""
    disc, index = curves[0], curves[-1]
    return legs_npv(terms.legs, disc, index, 0.0, terms.alive_fixed, terms.alive_float, terms.projected,
                    terms.known_rates)


def _legs_pnl(terms: _LegsTerms, curves, _quotes, moves: Moves, arithmetic: Arithmetic):
    """The coupons' t=0 NPV's change under the moves of the node values: the deviation of
    `leg_values`, nothing else (see the module docstring)."""
    disc, index = curves[0], curves[-1]
    _, moved = leg_values(terms.legs, disc, index, 0.0, terms.alive_fixed, terms.alive_float, terms.projected,
                          terms.known_rates, _deviations(terms.legs, moves.shifts), arithmetic=arithmetic)
    return moved.value if terms.legs.payer else -moved.value


# ---------------------------------------------------------------------------
# Europeans
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class JamshidianPrice:
    """`(terms, curves, quotes) -> NPV` on the Jamshidian engine of `model`: a pricer that
    carries its engine's settings, equal (and so one compiled program) for equal settings."""
    model: JamshidianEngineConfig

    def __call__(self, terms: EuropeanTerms, curves, _quotes):
        return jamshidian_npv(terms, self.model, curves[0], 0.0)


@dataclass(frozen=True)
class JamshidianPnl:
    """`(terms, curves, quotes, moves, arithmetic) -> P&L` on the Jamshidian engine of `model`:
    the discount factors in difference form (`engine.pricing.jamshidian.jamshidian_discounts`),
    x* and the bond options in the accumulate format, their sum in the compute format, less the
    price at the level."""
    model: JamshidianEngineConfig

    def __call__(self, terms: EuropeanTerms, curves, _quotes, moves: Moves, arithmetic: Arithmetic):
        disc = curves[0]
        deviations = _deviations(terms.legs, moves.shifts)
        p_bond, p_start = jamshidian_discounts(terms, disc, 0.0, deviations.discount)
        return _changed(jamshidian_npv(terms, self.model, disc, 0.0),
                        jamshidian_on_discounts(terms, self.model, p_bond, p_start, 0.0, arithmetic), moves.shifts)


class BachelierTerms(NamedTuple):
    """A European on ORE's `BlackMultiLegOptionEngine`: its terms, where its normal volatility
    is read on the quotes (`weights`, `[option tenors, swap tenors]`: the surface is linear in
    them) and the expiry time, both float64, and the coupons' date rules."""
    terms: EuropeanTerms
    weights: np.ndarray
    expiry: np.ndarray
    projected: np.ndarray
    known_rates: np.ndarray

    def variance(self, quotes, dtype):
        """The normal variance to expiry on the quotes, computed in float64 and rounded once to
        `dtype` (the curves')."""
        volatility = jnp.sum(jnp.asarray(self.weights, jnp.float64) * jnp.asarray(quotes, jnp.float64))
        return (volatility ** 2 * jnp.asarray(self.expiry, jnp.float64)).astype(dtype)


def bachelier_terms(cfg: SwaptionConfig, market: Market, dtype) -> BachelierTerms:
    """The `BachelierTerms` of a European on `market`, its volatility read where the engine
    reads it (`vol_point`)."""
    asof = market.asof
    terms = european_terms(cfg, asof)
    t, swap_len = vol_point(cfg, market)
    weights = market.swaption_vols(cfg.currency).weights(asof, t, swap_len)
    known = today_schedule(terms.legs, asof)
    return BachelierTerms(terms.astype(dtype), np.asarray(weights, dtype=np.float64), np.asarray(t, dtype=np.float64),
                          known.projected, np.asarray(known.known_rates, dtype=dtype))


def bachelier_price(terms: BachelierTerms, curves, quotes):
    """`(terms, curves, quotes) -> NPV`."""
    disc, index = curves
    return black_multileg_npv(terms.terms, disc, index, 0.0, terms.variance(quotes, curve_dtype(disc)),
                              terms.projected, terms.known_rates)


def bachelier_pnl(terms: BachelierTerms, curves, quotes, moves: Moves, arithmetic: Arithmetic):
    """The European's t=0 NPV's change under the moves of the node values, the quotes held: the
    exercise-into swap's values in difference form (`european_legs`), the Bachelier formula at
    the level and on each scenario in the accumulate format."""
    disc, index = curves
    deviations = _deviations(terms.terms.legs, moves.shifts)
    level, moved = european_legs(terms.terms, disc, index, 0.0, terms.projected, terms.known_rates, deviations,
                                 arithmetic)
    variance = terms.variance(quotes, curve_dtype(disc))
    return _changed(bachelier_on_legs(terms.terms, level, variance),
                    bachelier_on_legs(terms.terms, on_path(level, moved), variance), moves.shifts)


def vol_point(cfg: SwaptionConfig, market: Market) -> Tuple[float, float]:
    """(option time, swap length) where the European's engine reads its volatility."""
    return VOL_DAY_COUNTER.yearFraction(market.asof, cfg.exercise_date), european_terms(cfg, market.asof).swap_length


# ---------------------------------------------------------------------------
# Bermudans and Americans
# ---------------------------------------------------------------------------
class CalibrationTerms(NamedTuple):
    """An option's LGM calibration as data: its basket (float64), each bucket's length `dt`
    (the first from 0), where each helper's volatility is read on the quotes (`weights`,
    `[n, option tenors, swap tenors]`) and ORE's acceptance tolerance."""
    basket: Tuple[BasketInstrument, ...]
    dt: np.ndarray
    weights: np.ndarray
    tolerance: np.ndarray


class OptionTerms(NamedTuple):
    """A Bermudan/American on its grid engine: the prepared trade with its LGM (no curves:
    the pricer sets them; a calibrated LGM's buckets as calibrated on the base market), the
    induction's rows, built on the host, and its calibration (None on a fixed model)."""
    prepared: _PreparedBermudan
    rows: _Induction
    calibration: Optional[CalibrationTerms] = None


def option_terms(cfg, engine: LgmSwaptionEngineConfig, context: PricingContext, market: Market,
                 dtype) -> OptionTerms:
    """The `OptionTerms` of a live option on the base `context`: the LGM calibrated there as
    `engine.pricing.bermudan.bermudan_value` calibrates it (refused where it misses ORE's
    acceptance, I-73), and the calibration's terms (None when the engine does not calibrate or
    no helper is left)."""
    calibrated = calibrate_on(cfg, engine, context)
    sigma = Sigma.flat(engine.volatility) if calibrated is None else calibrated.sigma
    sigma = Sigma(times=jnp.asarray(sigma.times, dtype=dtype), values=jnp.asarray(sigma.values, dtype=dtype))
    prepared = prepared_option(cfg, engine, sigma)
    prepared = dataclasses.replace(prepared, notional=jnp.asarray(prepared.notional, dtype=dtype),
                                   fixed_amounts=jnp.asarray(prepared.fixed_amounts, dtype=dtype))
    calibration = None
    if calibrated is not None:
        basket = tuple(b.astype(np.float64) for b in calibration_basket(cfg, engine, context.date))
        surface = market.swaption_vols(cfg.currency)
        weights = np.stack([surface.weights(market.asof, b.vol_option_time, b.vol_swap_length) for b in basket])
        calibration = CalibrationTerms(basket, np.diff(basket_expiries(basket), prepend=0.0), weights,
                                       np.asarray(engine.bootstrap_tolerance))
    return OptionTerms(prepared, induction_rows(prepared), calibration)


def _on(terms: OptionTerms, curves, values=None) -> _PreparedBermudan:
    """The prepared option on `curves`, its LGM's buckets `values` (the terms' own by default),
    in the curves' dtype."""
    disc, index = curves
    sigma = terms.prepared.sigma
    if values is not None:
        sigma = Sigma(times=sigma.times, values=jnp.asarray(values, dtype=curve_dtype(disc)))
    return dataclasses.replace(terms.prepared, curve=disc, index_curve=index, sigma=sigma)


def option_price(terms: OptionTerms, curves, _quotes):
    """`(terms, curves, quotes) -> NPV` on the grid engine, its LGM fixed."""
    return grid_value(_on(terms, curves), terms.rows)


def option_pnl(terms: OptionTerms, curves, _quotes, moves: Moves, arithmetic: Arithmetic):
    """The option's t=0 NPV's change under the moves of the node values, its LGM fixed: the grid
    engine's deviation from its value on `curves` (`engine.pricing.lgm_grid.option_values`), the
    discount and index curves moved and the known fixings not."""
    prepared = _on(terms, curves)
    shifts = moves.shifts
    fixings = jnp.zeros(shifts[0].shape[:-1] + prepared.float_pay_times.shape, dtype=arithmetic.dtype)
    _, moved = option_values(prepared, terms.rows, Deviations(shifts[0], shifts[-1], fixings), arithmetic)
    return moved


@dataclass(frozen=True)
class CalibratedOptionPrice:
    """`(terms, curves, quotes) -> NPV` of an option whose LGM is calibrated on the curves and
    quotes it is priced on (see the module docstring), with the engine's constant `reversion`
    and root `solver`: a pricer equal for equal settings. The calibration is
    `engine.calibration.lgm.calibrated_values`, whose derivatives are the implicit function
    theorem's on the whole basket."""
    reversion: float
    solver: str

    def __call__(self, terms: OptionTerms, curves, quotes):
        calibration = terms.calibration
        values = calibrated_values(calibration.basket, *self.market(calibration, curves, quotes), calibration.dt,
                                   self.reversion, self.solver)
        return grid_value(_on(terms, curves, values), terms.rows)

    @staticmethod
    def market(calibration: CalibrationTerms, curves, quotes):
        """`(disc, index, vols)`: the calibration's market in float64 (A-10), the helpers'
        volatilities read on the quotes."""
        disc, index = (jax.tree_util.tree_map(lambda a: jnp.asarray(a, dtype=jnp.float64), c) for c in curves)
        # The surface is bilinear in the quotes: each helper's volatility is a weighted sum of them.
        vols = jnp.sum(calibration.weights * jnp.asarray(quotes, dtype=jnp.float64), axis=(-2, -1))
        return disc, index, vols


@dataclass(frozen=True)
class ScenarioCalibration:
    """`(terms, curves, quotes, shifts) -> (values [S, n], misses [S])`: an option's LGM
    recalibrated under each scenario's moves of the node values, the quotes held, in float64
    (A-10), and whether each calibration misses ORE's acceptance, as ORE's historical P&L
    recalibrates its model builders (I-112). The values reach `recalibrated_pnl` as
    `Moves.models`."""
    price: CalibratedOptionPrice

    def __call__(self, terms: Positioned, curves, quotes, shifts):
        calibration = terms.terms.calibration

        def one(*shift):
            moved = tuple(with_node_deviation(c, s) for c, s in zip(curves, shift))
            values, market, model, hit = bootstrap(calibration.basket, *self.price.market(calibration, moved, quotes),
                                                   calibration.dt, self.price.reversion, self.price.solver)
            return values, calibration_misses(market, model, hit, calibration.tolerance)

        return jax.vmap(one)(*shifts)


def recalibrated_pnl(terms: OptionTerms, curves, _quotes, moves: Moves, arithmetic: Arithmetic):
    """The P&L of an option recalibrated under each scenario: per scenario its own induction, on
    its LGM's buckets `moves.models` (`ScenarioCalibration`), in difference form: the level is
    the scenario's model on the base `curves` (in their dtype), the deviation the curves' move
    in `arithmetic`, so `P&L = (level - base) + deviation`, the base the option on the base
    calibration. The level and the base meet the deviation in two terms of the accumulate
    format (`add_level`), and a scenario that moves nothing has exactly 0 (a recalibration on
    unmoved curves can round its buckets an ulp apart)."""
    disc, index = curves
    shifts = moves.shifts
    base = grid_value(_on(terms, curves), terms.rows)
    ids = (jnp.zeros(shifts[0].shape[0], dtype=jnp.uint32) if arithmetic.sample_ids is None
           else arithmetic.sample_ids)

    def one(values, discount_shift, index_shift, sample):
        prepared = _on(terms, curves, values)
        scenario = (arithmetic if arithmetic.sample_ids is None
                    else dataclasses.replace(arithmetic, sample_ids=sample[None]))
        fixings = jnp.zeros((1,) + prepared.float_pay_times.shape, dtype=arithmetic.dtype)
        level, moved = option_values(prepared, terms.rows,
                                     Deviations(discount_shift[None], index_shift[None], fixings), scenario)
        return level, moved[0]

    level, moved = jax.vmap(one)(moves.models, shifts[0], shifts[-1], ids)
    return jnp.where(_moved(shifts), add_level(level - base, moved), 0.0)
