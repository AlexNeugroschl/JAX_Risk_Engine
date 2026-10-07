"""
Each trade's t=0 price as a pure JAX function of the pillar zero rates of the market curves it
reads, with the trade's configured engine: what AD Greeks differentiate (`engine.risk.greeks`)
and market risk revalues under shocked curves (`engine.market_risk`).

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
                 per product and shape, not per trade or call (`engine.risk.greeks`)

The engines are the valuation pipeline's own (`engine.valuation`), so the price at the market's
rates is the portfolio's t=0 NPV: swaps and bonds by `legs_npv`, Europeans by
`black_multileg_npv` (the normal volatility read once from the market and held fixed) or
`jamshidian_npv`, Bermudans and Americans by the grid engine with the LGM calibrated on today's
market and held fixed as the rates move (`bermudan_price_function`, which also exposes the
volatility buckets for Vega). ORE's bump sensitivities recalibrate under every bump instead
(`engine.risk.sensitivities`); the two differ by the calibration's response to the curve.

The curve's dtype is the working dtype. An option expired on the market's date prices to a
constant 0 (ORE's `isExpired`), so its sensitivities are 0.
"""
import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import (
    BermudanSwaptionConfig, _build_grid_schedule, _GridSchedule, _PreparedBermudan, grid_value,
)
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig, _build_ore_swap as _swap_underlying
from engine.instruments.treasury import BondConfig
from engine.market import VOL_DAY_COUNTER, Market, index_name
from engine.models.curves import ZeroCurve
from engine.models.lgm import Sigma
from engine.valuation.bermudan import Calibration, calibrate_on, prepared_option
from engine.valuation.config import JamshidianEngineConfig, PricingConfig
from engine.valuation.context import from_market
from engine.valuation.european import EuropeanTerms, black_multileg_npv, european_terms
from engine.valuation.jamshidian import jamshidian_npv
from engine.valuation.legs import Legs, legs_npv, legs_of, today_schedule
from engine.valuation.portfolio import bond_legs

#: A market curve: `("discount", currency)` or `("index", index name)`.
CurveKey = Tuple[str, str]


@dataclass(frozen=True)
class TradePriceFunction:
    """`price(*pillar_rates)`, one rates array per curve of `curves`, in that order:
    `pricer(terms, *curves)` on `ZeroCurve`s at the pillar `times` (see the module
    docstring)."""
    curves: Tuple[CurveKey, ...]
    pricer: Callable
    terms: Any
    times: Tuple[jnp.ndarray, ...]

    def price(self, *rates):
        return self.pricer(self.terms, *on_pillars(self.times, rates))


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

    def priced(pricer, terms=None):
        return TradePriceFunction(keys, pricer, terms, times)

    if isinstance(cfg, BondConfig):
        return priced(_legs_price, _legs_terms(bond_legs(cfg), asof, dtype))
    if isinstance(cfg, SwapConfig):
        legs = legs_of(_swap_underlying(cfg), cfg.payer, asof, cfg.fixings)
        return priced(_legs_price, _legs_terms(legs, asof, dtype))
    if isinstance(cfg, SwaptionConfig):
        if not cfg.exercise_date > asof:
            return priced(_zero_price)
        if pricing.european == "Jamshidian":
            return priced(JamshidianPrice(pricing.jamshidian), european_terms(cfg, asof).astype(dtype))
        return priced(bachelier_price, bachelier_terms(cfg, market, dtype))
    if isinstance(cfg, (BermudanSwaptionConfig, AmericanSwaptionConfig)):
        option = bermudan_price_function(cfg, market, pricing, dtype)
        return priced(option.pricer, option.terms)
    raise TypeError(f"no price function for {type(cfg).__name__}")


def _zero_price(_terms, *curves):
    """An expired option's price: 0, still a function of the curves."""
    return jnp.zeros((), dtype=curves[0].pillar_rates.dtype) * sum(jnp.sum(c.pillar_rates) for c in curves)


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


@dataclass(frozen=True)
class JamshidianPrice:
    """`(terms, disc, index) -> NPV` on the Jamshidian engine of `model`: a pricer that
    carries its engine's settings, equal (and so one compiled program) for equal settings."""
    model: JamshidianEngineConfig

    def __call__(self, terms: EuropeanTerms, disc, index):
        return jamshidian_npv(terms, self.model, disc, 0.0)


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
    sigma_values)`. `calibration` is today's (None when the engine does not calibrate, or
    the option has expired, when `terms` is None too and the price is 0)."""
    pricer: Callable
    terms: Optional[OptionTerms]
    sigma: Sigma
    calibration: Optional[Calibration]

    def price(self, disc, index, values=None):
        if self.terms is None:
            return self.pricer(None, disc, index)
        return self.pricer(self.terms, disc, index, values)


def bermudan_price_function(cfg, market: Market, pricing: PricingConfig, dtype) -> BermudanPriceFunction:
    """The Bermudan/American grid engine with the LGM calibrated on today's market as
    `engine.valuation.bermudan.bermudan_value` calibrates it, as a function of the curves and
    the volatility buckets."""
    engine = pricing.american if isinstance(cfg, AmericanSwaptionConfig) else pricing.bermudan
    if cfg.is_expired():
        return BermudanPriceFunction(_zero_price, None, Sigma.flat(engine.volatility, dtype=dtype), None)
    calibration = calibrate_on(cfg, engine, from_market(market))
    sigma = Sigma.flat(engine.volatility) if calibration is None else calibration.sigma
    sigma = Sigma(times=jnp.asarray(sigma.times, dtype=dtype), values=jnp.asarray(sigma.values, dtype=dtype))
    prepared = prepared_option(cfg, engine, sigma)
    prepared = dataclasses.replace(prepared, notional=jnp.asarray(prepared.notional, dtype=dtype),
                                   fixed_amounts=jnp.asarray(prepared.fixed_amounts, dtype=dtype))
    return BermudanPriceFunction(option_price, OptionTerms(prepared, _build_grid_schedule(prepared)), sigma,
                                 calibration)


def curves_of(fn: TradePriceFunction, market: Market, dtype) -> Sequence[jnp.ndarray]:
    """The base pillar rates of each of `fn`'s curves on `market`, in `dtype`."""
    return [jnp.asarray(market_curve(market, k).rates, dtype=dtype) for k in fn.curves]
