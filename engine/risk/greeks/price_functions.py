"""
Each trade's t=0 price as a pure JAX function of the pillar zero rates of the market curves it
reads, with the trade's configured engine: what AD Greeks differentiate (`engine.risk.greeks.ad`)
and market risk revalues under shocked curves (`engine.risk.market`).

    trade_price_function(cfg, market, pricing, dtype) -> TradePriceFunction
        .curves  the curves it reads, as `(kind, name)`: `("discount", ccy)` and, for every
                 trade but a bond, `("index", index name)`
        .price   f(*pillar_rates), one array per curve in that order, on the market curve's
                 pillar times
        .pricer, .terms, .times
                 the same function as data: `price(*rates)` is
                 `pricer(terms, *on_pillars(times, rates))`, with `pricer` a module-level
                 function (or an object equal for equal settings) and `terms` the trade's
                 data as a pytree, so a caller jits it with `pricer` static and compiles once
                 per product and shape, not per trade or call (`engine.risk.greeks.ad`)
        .pnl     for a trade priced in difference form (swaps, bonds, Europeans; F-07):
                 `pnl(terms, curves, shifts, arithmetic)`, each scenario's change of the price
                 when the pillar rates of `curves` (`ZeroCurve`s) move by `shifts` (`[S, K]`
                 per curve), computed as the deviation from the level in the matrix products
                 of `arithmetic` (`engine.pricing.legs.leg_values`), so a move of nothing is
                 exactly 0 and the legs' cancellation stays in the level. None for a
                 Bermudan/American, which market risk revalues in full (`engine.risk.market`)

The engines are the valuation pipeline's own (`engine.pricing`), so the price at the market's
rates is the portfolio's t=0 NPV: swaps and bonds by `legs_npv`, Europeans by
`black_multileg_npv` (the normal volatility read once from the market and held fixed) or
`jamshidian_npv`, Bermudans and Americans by the grid engine with the LGM calibrated on today's
market and held fixed as the rates move (`bermudan_price_function`, which also exposes the
volatility buckets for Vega). ORE's bump sensitivities recalibrate under every bump instead
(`engine.risk.greeks.bump`); the two differ by the calibration's response to the curve.

An option's price is in its position (`long_short`, ORE's `LongShort`): its engine prices the
long option and `OnPosition` multiplies by the position's sign, carried in its `Positioned`
terms as data, as ORE's `OptionWrapper` multiplies its engine's NPV. A short option's price, its
derivatives and its P&L are minus the long one's, and a long and a short option of one shape are
one compiled program (I-101).

The curve's dtype is the working dtype. An option expired on the market's date prices to a
constant 0 (ORE's `isExpired`), so its sensitivities are 0.
"""
import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig, underlying_swap as _swap_underlying
from engine.instruments.treasury import BondConfig
from engine.market_data.curves import ZeroCurve
from engine.market_data.market import VOL_DAY_COUNTER, Market, index_name
from engine.models.lgm import Sigma
from engine.precision import Arithmetic
from engine.pricing.bermudan import Calibration, calibrate_on, prepared_option
from engine.pricing.config import JamshidianEngineConfig, PricingConfig
from engine.pricing.context import from_market
from engine.pricing.cube import bond_legs
from engine.pricing.european import (
    EuropeanTerms, bachelier_on_legs, black_multileg_npv, european_legs, european_terms, on_path,
)
from engine.pricing.jamshidian import jamshidian_discounts, jamshidian_npv, jamshidian_on_discounts
from engine.pricing.legs import Deviations, Legs, leg_values, legs_npv, legs_of, today_schedule
from engine.pricing.lgm_grid import _build_grid_schedule, _GridSchedule, _PreparedBermudan, grid_value
from engine.pricing.options import position_sign

#: A market curve: `("discount", currency)` or `("index", index name)`.
CurveKey = Tuple[str, str]


@dataclass(frozen=True)
class TradePriceFunction:
    """`price(*pillar_rates)`, one rates array per curve of `curves`, in that order:
    `pricer(terms, *curves)` on `ZeroCurve`s at the pillar `times`; and `pnl`, the change of
    the price under shifts of the rates in difference form, or None (see the module
    docstring)."""
    curves: Tuple[CurveKey, ...]
    pricer: Callable
    terms: Any
    times: Tuple[jnp.ndarray, ...]
    pnl: Optional[Callable] = None

    def price(self, *rates):
        return self.pricer(self.terms, *on_pillars(self.times, rates))


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


def on_pillars(times: Sequence, rates: Sequence) -> Tuple[ZeroCurve, ...]:
    """A `ZeroCurve` per curve, from its pillar times and rates."""
    return tuple(ZeroCurve(pillar_times=t, pillar_rates=r) for t, r in zip(times, rates))


def curve_keys(cfg) -> Tuple[CurveKey, ...]:
    """The market curves a trade's price reads: its currency's discount curve and, but for a
    bond, its index's forwarding curve (read even by an engine that ignores it, so every
    method reports the same curves)."""
    if isinstance(cfg, BondConfig):
        return (("discount", cfg.currency),)
    return (("discount", cfg.currency), ("index", index_name(cfg.currency, cfg.index_tenor_months)))


def market_curve(market: Market, key: CurveKey):
    """The `ZeroCurveConfig` of a curve key."""
    kind, name = key
    if kind == "discount":
        return market.currency(name).discount_curve
    return market.index_curve(name.split("-", 1)[0], name)


def trade_price_function(cfg, market: Market, pricing: PricingConfig = PricingConfig(),
                         dtype=jnp.float64) -> TradePriceFunction:
    """The trade's t=0 price on `market` as a function of its curves' pillar rates (see the
    module docstring). The trade must be valued on the market's as-of date."""
    keys = curve_keys(cfg)
    times = tuple(jnp.asarray(market_curve(market, k).times, dtype=dtype) for k in keys)
    asof = market.asof

    def priced(pricer, terms=None, pnl=None):
        return TradePriceFunction(keys, pricer, terms, times, pnl)

    def positioned(long, terms, long_pnl):
        pricer, terms = in_position(cfg, long, terms, dtype)
        return priced(pricer, terms, OnPosition(long_pnl))

    if isinstance(cfg, BondConfig):
        return priced(_legs_price, _legs_terms(bond_legs(cfg), asof, dtype), _legs_pnl)
    if isinstance(cfg, SwapConfig):
        legs = legs_of(_swap_underlying(cfg), cfg.payer, asof, cfg.fixings)
        return priced(_legs_price, _legs_terms(legs, asof, dtype), _legs_pnl)
    if isinstance(cfg, SwaptionConfig):
        if not cfg.exercise_date > asof:
            return priced(_zero_price, pnl=_zero_pnl)
        if pricing.european == "Jamshidian":
            return positioned(JamshidianPrice(pricing.jamshidian), european_terms(cfg, asof).astype(dtype),
                              JamshidianPnl(pricing.jamshidian))
        return positioned(bachelier_price, bachelier_terms(cfg, market, dtype), bachelier_pnl)
    if isinstance(cfg, (BermudanSwaptionConfig, AmericanSwaptionConfig)):
        option = bermudan_price_function(cfg, market, pricing, dtype)
        return priced(option.pricer, option.terms)
    raise TypeError(f"no price function for {type(cfg).__name__}")


def _zero_price(_terms, *curves):
    """An expired option's price: 0, still a function of the curves."""
    return jnp.zeros((), dtype=curves[0].pillar_rates.dtype) * sum(jnp.sum(c.pillar_rates) for c in curves)


def _zero_pnl(_terms, curves, shifts, arithmetic: Arithmetic):
    """An expired option's P&L: 0 in every scenario."""
    return jnp.zeros(shifts[0].shape[:-1], dtype=arithmetic.dtype)


def _deviations(terms_legs: Legs, shifts) -> Deviations:
    """The `Deviations` of a leg whose curves' rates move by `shifts` (one curve: a bond, whose
    one curve is both) and whose known fixings do not."""
    discount, index = shifts[0], shifts[-1]
    return Deviations(discount, index, jnp.zeros(discount.shape[:-1] + terms_legs.float_pay.shape, discount.dtype))


def _changed(level_value, shocked_value, deviations: Deviations):
    """`shocked - level` per scenario, exactly 0 where nothing moved (rather than the difference
    of one value evaluated twice, which XLA may round apart)."""
    moved = jnp.zeros(deviations.discount.shape[:-1], dtype=bool)
    for deviation in (deviations.discount, deviations.index, deviations.fixings):
        moved = moved | jnp.any(deviation != 0, axis=-1)
    return jnp.where(moved, shocked_value - level_value, 0.0)


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


def _legs_price(terms: _LegsTerms, disc, index=None):
    """ORE's t=0 NPV of the coupons; a bond's coupons are forecast off its one curve."""
    return legs_npv(terms.legs, disc, disc if index is None else index, 0.0, terms.alive_fixed, terms.alive_float,
                    terms.projected, terms.known_rates)


def _legs_pnl(terms: _LegsTerms, curves, shifts, arithmetic: Arithmetic):
    """The coupons' t=0 NPV's change under `shifts` of the pillar rates: the deviation of
    `leg_values`, nothing else (see the module docstring)."""
    disc, index = curves[0], curves[-1]
    _, moved = leg_values(terms.legs, disc, index, 0.0, terms.alive_fixed, terms.alive_float, terms.projected,
                          terms.known_rates, _deviations(terms.legs, shifts), arithmetic=arithmetic)
    return moved.value if terms.legs.payer else -moved.value


@dataclass(frozen=True)
class JamshidianPrice:
    """`(terms, disc, index) -> NPV` on the Jamshidian engine of `model`: a pricer that
    carries its engine's settings, equal (and so one compiled program) for equal settings."""
    model: JamshidianEngineConfig

    def __call__(self, terms: EuropeanTerms, disc, index):
        return jamshidian_npv(terms, self.model, disc, 0.0)


@dataclass(frozen=True)
class JamshidianPnl:
    """`(terms, curves, shifts, arithmetic) -> P&L` on the Jamshidian engine of `model`: the
    discount factors in difference form (`engine.pricing.jamshidian.jamshidian_discounts`), x*
    and the bond options in the accumulate format, their sum in the compute format, less the
    price at the level."""
    model: JamshidianEngineConfig

    def __call__(self, terms: EuropeanTerms, curves, shifts, arithmetic: Arithmetic):
        disc = curves[0]
        deviations = _deviations(terms.legs, shifts)
        p_bond, p_start = jamshidian_discounts(terms, disc, 0.0, deviations.discount)
        return _changed(jamshidian_npv(terms, self.model, disc, 0.0),
                        jamshidian_on_discounts(terms, self.model, p_bond, p_start, 0.0, arithmetic), deviations)


class BachelierTerms(NamedTuple):
    """A European on ORE's `BlackMultiLegOptionEngine`: its terms, the normal variance to
    expiry read from the market (held fixed), the expiry time (for the variance of another
    volatility: Vega) and the coupons' date rules."""
    terms: EuropeanTerms
    variance: float
    expiry: float
    projected: np.ndarray
    known_rates: np.ndarray


def bachelier_terms(cfg: SwaptionConfig, market: Market, dtype) -> BachelierTerms:
    """The `BachelierTerms` of a European on `market`, its volatility read where the engine
    reads it (`vol_point`)."""
    asof = market.asof
    terms = european_terms(cfg, asof)
    t, swap_len = vol_point(cfg, market)
    volatility = float(market.swaption_vols(cfg.currency).volatility(asof, t, swap_len))
    known = today_schedule(terms.legs, asof)
    return BachelierTerms(terms.astype(dtype), volatility ** 2 * t, t, known.projected,
                          np.asarray(known.known_rates, dtype=dtype))


def bachelier_price(terms: BachelierTerms, disc, index, volatility=None):
    """`(terms, disc, index) -> NPV`; a `volatility` replaces the market's (Vega)."""
    variance = terms.variance if volatility is None else volatility ** 2 * terms.expiry
    return black_multileg_npv(terms.terms, disc, index, 0.0, variance, terms.projected, terms.known_rates)


def bachelier_pnl(terms: BachelierTerms, curves, shifts, arithmetic: Arithmetic):
    """The European's t=0 NPV's change under `shifts` of the pillar rates: the exercise-into
    swap's values in difference form (`european_legs`), the Bachelier formula at the level and
    on each scenario in the accumulate format."""
    disc, index = curves
    deviations = _deviations(terms.terms.legs, shifts)
    level, moved = european_legs(terms.terms, disc, index, 0.0, terms.projected, terms.known_rates, deviations,
                                 arithmetic)
    return _changed(bachelier_on_legs(terms.terms, level, terms.variance),
                    bachelier_on_legs(terms.terms, on_path(level, moved), terms.variance), deviations)


def vol_point(cfg: SwaptionConfig, market: Market) -> Tuple[float, float]:
    """(option time, swap length) where the European's engine reads its volatility."""
    return VOL_DAY_COUNTER.yearFraction(market.asof, cfg.exercise_date), european_terms(cfg, market.asof).swap_length


class OptionTerms(NamedTuple):
    """A Bermudan/American on its grid engine: the prepared trade with its LGM (no curves:
    the pricer sets them) and the induction's schedule, built on the host."""
    prepared: _PreparedBermudan
    schedule: _GridSchedule


def option_price(terms: OptionTerms, disc, index, values=None):
    """`(terms, disc, index) -> NPV` on the grid engine; `values` replace the LGM's
    volatility buckets (Vega)."""
    sigma = terms.prepared.sigma
    if values is not None:
        sigma = Sigma(times=sigma.times, values=values)
    return grid_value(dataclasses.replace(terms.prepared, curve=disc, index_curve=index, sigma=sigma), terms.schedule)


@dataclass(frozen=True)
class BermudanPriceFunction:
    """`price(disc, index, sigma_values)`: the grid engine on the curves with the calibrated
    LGM's buckets `sigma_values` (None: `sigma`'s own), as `pricer(terms, disc, index,
    sigma_values)`, in the option's position (`OnPosition` of `option_price`). `calibration` is
    today's (None when the engine does not calibrate, or the option has expired, when `terms`
    is None too and the price is 0)."""
    pricer: Callable
    terms: Optional[Positioned]
    sigma: Sigma
    calibration: Optional[Calibration]

    def price(self, disc, index, values=None):
        if self.terms is None:
            return self.pricer(None, disc, index)
        return self.pricer(self.terms, disc, index, values)


def bermudan_price_function(cfg, market: Market, pricing: PricingConfig, dtype) -> BermudanPriceFunction:
    """The Bermudan/American grid engine with the LGM calibrated on today's market as
    `engine.pricing.bermudan.bermudan_value` calibrates it, as a function of the curves and
    the volatility buckets, in the option's position."""
    engine = pricing.american if isinstance(cfg, AmericanSwaptionConfig) else pricing.bermudan
    if cfg.is_expired():
        return BermudanPriceFunction(_zero_price, None, Sigma.flat(engine.volatility, dtype=dtype), None)
    calibration = calibrate_on(cfg, engine, from_market(market))
    sigma = Sigma.flat(engine.volatility) if calibration is None else calibration.sigma
    sigma = Sigma(times=jnp.asarray(sigma.times, dtype=dtype), values=jnp.asarray(sigma.values, dtype=dtype))
    prepared = prepared_option(cfg, engine, sigma)
    prepared = dataclasses.replace(prepared, notional=jnp.asarray(prepared.notional, dtype=dtype),
                                   fixed_amounts=jnp.asarray(prepared.fixed_amounts, dtype=dtype))
    pricer, terms = in_position(cfg, option_price, OptionTerms(prepared, _build_grid_schedule(prepared)), dtype)
    return BermudanPriceFunction(pricer, terms, sigma, calibration)


def curves_of(fn: TradePriceFunction, market: Market, dtype) -> Sequence[jnp.ndarray]:
    """The base pillar rates of each of `fn`'s curves on `market`, in `dtype`."""
    return [jnp.asarray(market_curve(market, k).rates, dtype=dtype) for k in fn.curves]
