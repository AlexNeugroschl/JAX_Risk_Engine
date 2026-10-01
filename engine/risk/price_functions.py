"""
Each trade's t=0 price as a pure JAX function of the pillar zero rates of the market curves it
reads, with the trade's configured engine: what AD Greeks differentiate (`engine.risk.greeks`)
and market risk revalues under shocked curves (`engine.market_risk`).

    trade_price_function(cfg, market, pricing, dtype) -> TradePriceFunction
        .curves  the curves it reads, as `(kind, name)`: `("discount", ccy)` and, for every
                 trade but a bond, `("index", index name)`
        .price   f(*pillar_rates), one array per curve in that order, on the market curve's
                 pillar times

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
from typing import Callable, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig, grid_value
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig, _build_ore_swap as _swap_underlying
from engine.instruments.treasury import BondConfig
from engine.market import VOL_DAY_COUNTER, Market, index_name
from engine.models.curves import ZeroCurve
from engine.models.lgm import Sigma
from engine.valuation.bermudan import Calibration, calibrate_on, prepared_option
from engine.valuation.config import PricingConfig
from engine.valuation.context import from_market
from engine.valuation.european import black_multileg_npv, european_terms
from engine.valuation.jamshidian import jamshidian_npv
from engine.valuation.legs import Legs, legs_npv, legs_of, today_schedule
from engine.valuation.portfolio import bond_legs

#: A market curve: `("discount", currency)` or `("index", index name)`.
CurveKey = Tuple[str, str]


@dataclass(frozen=True)
class TradePriceFunction:
    """`price(*pillar_rates)`, one rates array per curve of `curves`, in that order."""
    curves: Tuple[CurveKey, ...]
    price: Callable


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
    times = [jnp.asarray(market_curve(market, k).times, dtype=dtype) for k in keys]
    on = lambda i, rates: ZeroCurve(pillar_times=times[i], pillar_rates=rates)  # noqa: E731
    asof = market.asof

    if isinstance(cfg, BondConfig):
        price_legs = _price_legs(bond_legs(cfg), asof, dtype)
        return TradePriceFunction(keys, lambda disc: price_legs(on(0, disc), on(0, disc)))
    if isinstance(cfg, SwapConfig):
        price_legs = _price_legs(legs_of(_swap_underlying(cfg), cfg.payer, asof, cfg.fixings), asof, dtype)
        return TradePriceFunction(keys, lambda disc, index: price_legs(on(0, disc), on(1, index)))
    if isinstance(cfg, SwaptionConfig):
        if not cfg.exercise_date > asof:
            return TradePriceFunction(keys, _zero)
        if pricing.european == "Jamshidian":
            terms = european_terms(cfg, asof).astype(dtype)
            return TradePriceFunction(
                keys, lambda disc, index: jamshidian_npv(terms, pricing.jamshidian, on(0, disc), 0.0))
        bachelier = european_price_function(cfg, market, dtype)
        return TradePriceFunction(keys, lambda disc, index: bachelier(on(0, disc), on(1, index), None))
    if isinstance(cfg, (BermudanSwaptionConfig, AmericanSwaptionConfig)):
        option = bermudan_price_function(cfg, market, pricing, dtype)
        return TradePriceFunction(keys, lambda disc, index: option.price(on(0, disc), on(1, index), None))
    raise TypeError(f"no price function for {type(cfg).__name__}")


def _zero(*rates):
    return jnp.zeros((), dtype=rates[0].dtype) * sum(jnp.sum(r) for r in rates)


def _price_legs(legs: Legs, asof, dtype):
    """`(disc, index) -> ORE's t=0 NPV` of fixed and floating coupons (`today_npv`)."""
    today = today_schedule(legs, asof)
    legs = legs.astype(dtype)
    known = np.asarray(today.known_rates, dtype=dtype)
    return lambda disc, index: legs_npv(legs, disc, index, 0.0, today.alive_fixed, today.alive_float,
                                        today.projected, known)


def european_price_function(cfg: SwaptionConfig, market: Market, dtype):
    """`(disc, index, volatility) -> NPV` with ORE's `BlackMultiLegOptionEngine`; `volatility`
    None reads the market's, held fixed. Also the Vega point: `vol_point(cfg, market)` gives
    where the volatility is read."""
    asof = market.asof
    terms = european_terms(cfg, asof)
    t, swap_len = vol_point(cfg, market)
    base = float(market.swaption_vols(cfg.currency).volatility(asof, t, swap_len))
    known = today_schedule(terms.legs, asof)
    terms = terms.astype(dtype)
    known_rates = np.asarray(known.known_rates, dtype=dtype)

    def price(disc, index, volatility: Optional[float] = None):
        vol = base if volatility is None else volatility
        return black_multileg_npv(terms, disc, index, 0.0, vol ** 2 * t, known.projected, known_rates)

    return price


def vol_point(cfg: SwaptionConfig, market: Market) -> Tuple[float, float]:
    """(option time, swap length) where the European's engine reads its volatility."""
    return VOL_DAY_COUNTER.yearFraction(market.asof, cfg.exercise_date), european_terms(cfg, market.asof).swap_length


@dataclass(frozen=True)
class BermudanPriceFunction:
    """`price(disc, index, sigma_values)`: the grid engine on the curves with the calibrated
    LGM's buckets `sigma_values` (None: `sigma`'s own). `calibration` is today's (None when
    the engine does not calibrate, or the option has expired)."""
    price: Callable
    sigma: Sigma
    calibration: Optional[Calibration]


def bermudan_price_function(cfg, market: Market, pricing: PricingConfig, dtype) -> BermudanPriceFunction:
    """The Bermudan/American grid engine with the LGM calibrated on today's market as
    `engine.valuation.bermudan.bermudan_value` calibrates it, as a function of the curves and
    the volatility buckets."""
    engine = pricing.american if isinstance(cfg, AmericanSwaptionConfig) else pricing.bermudan
    if cfg.is_expired():
        sigma = Sigma.flat(engine.volatility, dtype=dtype)
        return BermudanPriceFunction(lambda disc, index, values=None: _zero(disc.pillar_rates, index.pillar_rates),
                                     sigma, None)
    calibration = calibrate_on(cfg, engine, from_market(market))
    sigma = Sigma.flat(engine.volatility) if calibration is None else calibration.sigma
    sigma = Sigma(times=jnp.asarray(sigma.times, dtype=dtype), values=jnp.asarray(sigma.values, dtype=dtype))
    prepared = prepared_option(cfg, engine, sigma)
    prepared = dataclasses.replace(prepared, notional=jnp.asarray(prepared.notional, dtype=dtype),
                                   fixed_amounts=jnp.asarray(prepared.fixed_amounts, dtype=dtype))

    def price(disc, index, values=None):
        buckets = sigma if values is None else Sigma(times=sigma.times, values=values)
        return grid_value(dataclasses.replace(prepared, curve=disc, index_curve=index, sigma=buckets))

    return BermudanPriceFunction(price, sigma, calibration)


def curves_of(fn: TradePriceFunction, market: Market, dtype) -> Sequence[jnp.ndarray]:
    """The base pillar rates of each of `fn`'s curves on `market`, in `dtype`."""
    return [jnp.asarray(market_curve(market, k).rates, dtype=dtype) for k in fn.curves]
