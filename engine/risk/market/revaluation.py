"""
Full revaluation of a portfolio at t=0 under shocked curves.

Every trade becomes a pure JAX function of the pillar rates of the market curves it reads, with
its configured engine (`engine.risk.greeks.price_functions.trade_price_function`, the functions AD
Greeks differentiate). The same function prices the base market and every scenario, so there is
no second pricer whose base could disagree. Every product is revalued in difference form
(F-07): a scenario's P&L is the deviation of the trade's value from its value at the base, the
shift carried by matrix products in the trade's compute format (`TradePriceFunction.pnl`: the
legs' flows, the grid engine's induction for Bermudans and Americans since 2026-10-13), so a
scenario that moves nothing has exactly zero P&L and the legs' cancellation stays in the base.

Scenarios are evaluated in batches (`_pnl`, one program per product, shape and arithmetic). The
batch is bounded by memory per trade (`scenario_batch_size`): a Bermudan's induction holds its
columns on the state grid for every scenario of the batch. The P&L is jitted on the trade's terms
as an argument, so a repeated run, or another trade of the same shape, reuses its programs.

**Engines.** Each product's engine is the pricing configuration's (`PricingConfig`): a European
on ORE's Bachelier engine (the normal volatility read from the market, held fixed) or on
Jamshidian; a Bermudan/American on its LGM grid engine, calibrated on today's market and held
fixed (decision A-8: the engine is chosen by configuration, not by the trade). ORE's historical
P&L recalibrates a calibrated LGM to each scenario's curves instead (its `HistoricalPnlGenerator`
runs the valuation engine with the trades' model builders): I-112.

**Which risk factors move.** Only curve pillar rates. Volatilities are held at their base
values (the surface, a calibrated LGM, the Jamshidian model), so volatility risk is not
captured. `run_market_risk` says so in its warnings for every option trade.
"""
from functools import partial
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.market_data.market import Market
from engine.market_simulation.sharding import scenario_device_count, shard_scenarios
from engine.precision import Arithmetic
from engine.pricing.bermudan import prepared_option
from engine.pricing.config import PricingConfig
from engine.pricing.lgm_grid import INDUCTION_MEMORY_BUDGET, working_bytes_per_path
from engine.risk.greeks.price_functions import curve_keys, on_pillars, trade_price_function
from engine.risk.market.factors import RateRiskFactors, curve_name

def factor_indices(cfg, factors: RateRiskFactors) -> Tuple[int, ...]:
    """The factor curves a trade reads, by position in `factors`; a curve that is not a factor
    is refused (`RateRiskFactors.index_of`)."""
    return tuple(factors.index_of(curve_name(*key)) for key in curve_keys(cfg))


def scenario_batch_size(cfg, pricing: PricingConfig, requested: int, itemsize: int) -> int:
    """Scenarios to revalue at once for `cfg`: `requested`, reduced so one batch stays within
    the induction's budget (`engine.pricing.lgm_grid.INDUCTION_MEMORY_BUDGET`). Only the grid
    engine needs it: its induction holds its columns on the state grid per scenario
    (`working_bytes_per_path`)."""
    if not isinstance(cfg, (BermudanSwaptionConfig, AmericanSwaptionConfig)) or cfg.is_expired():
        return requested
    engine = pricing.american if isinstance(cfg, AmericanSwaptionConfig) else pricing.bermudan
    per_scenario = working_bytes_per_path(prepared_option(cfg, engine, engine.volatility), itemsize)
    return max(1, min(requested, INDUCTION_MEMORY_BUDGET // per_scenario))


def revalue(
    trades: Sequence,
    market: Market,
    factors: RateRiskFactors,
    moves,
    pricing: PricingConfig = PricingConfig(),
    batch_size: int = 256,
    arithmetic: Optional[Arithmetic] = None,
) -> Tuple[np.ndarray, jnp.ndarray]:
    """Base values `[N]` and shocked values `[S, N]` of every trade (each its base value plus its
    P&L, `revalue_trade`), in `arithmetic` (the moves' own format, float64 or float32, by
    default).

    moves: `[S, F]` absolute factor moves (`ShockScenarios.shifts`).
    batch_size: the most scenarios to revalue at once; the grid engine may use fewer to stay
        within `INDUCTION_MEMORY_BUDGET`.
    """
    values = [revalue_trade(cfg, market, factors, moves, pricing, batch_size, arithmetic) for cfg in trades]
    return (np.asarray([base for base, _ in values]),
            jnp.stack([jnp.asarray(base, dtype=pnl.dtype) + pnl for base, pnl in values], axis=-1))


def revalue_trade(cfg, market: Market, factors: RateRiskFactors, moves, pricing: PricingConfig = PricingConfig(),
                  batch_size: int = 256, arithmetic: Optional[Arithmetic] = None) -> Tuple[float, jnp.ndarray]:
    """One trade's base value and its P&L `[S]`, in `arithmetic` (the moves' own format,
    float64 or float32, by default; `run_market_risk` gives each trade its pricing stage's), its
    accumulate format the moves' and the base's, in difference form (`TradePriceFunction.pnl`)."""
    moves = jnp.asarray(moves)
    arithmetic = arithmetic or Arithmetic.of(moves.dtype)
    moves = arithmetic.cast(moves)
    base = arithmetic.cast(factors.base_rates())
    fn = trade_price_function(cfg, market, pricing, arithmetic.dtype)
    own = [factors.slice_of(i) for i in factor_indices(cfg, factors)]
    value = float(fn.price(*[base[s] for s in own]))
    batch = scenario_batch_size(cfg, pricing, batch_size, jnp.dtype(arithmetic.dtype).itemsize)
    curves = on_pillars(fn.times, [base[s] for s in own])
    return value, _map_scenarios(
        lambda shifts: _pnl(fn.pnl, fn.terms, curves, tuple(shifts[..., s] for s in own), arithmetic), moves, batch)


@partial(jax.jit, static_argnames=("pnl", "arithmetic"))
def _pnl(pnl, terms, curves, shifts, arithmetic: Arithmetic):
    """`pnl(terms, curves, shifts, arithmetic)`: one program per product, shape and arithmetic."""
    return pnl(terms, curves, shifts, arithmetic)


def _map_scenarios(batched, moves, batch_size):
    """`batched(moves[i:j])` `[j - i]` for every batch of `batch_size` scenarios. With the
    scenarios split across devices (`engine.market_simulation.sharding`), each batch takes
    `batch_size` scenarios from every device's share, so every device works on every batch
    and holds one batch's memory."""
    devices = scenario_device_count(moves.shape[0])
    if devices == 1:
        return jnp.concatenate([batched(moves[i:i + batch_size]) for i in range(0, moves.shape[0], batch_size)])
    shares = shard_scenarios(moves.reshape(devices, -1, moves.shape[-1]), axis=0)   # [devices, S / devices, F]
    per_device = jax.vmap(batched)
    values = [per_device(shares[:, i:i + batch_size]) for i in range(0, shares.shape[1], batch_size)]
    return jnp.concatenate(values, axis=1).reshape(-1)
