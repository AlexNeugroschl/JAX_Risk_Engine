"""
Full revaluation of a portfolio at t=0 under shocked curves.

Every trade becomes a pure JAX function of the pillar rates of the market curves it reads, with
its configured engine (`engine.risk.greeks.price_functions.trade_price_function`, the functions AD
Greeks differentiate). The same function prices the base market and every scenario, so there is
no second pricer whose base could disagree. Swaps, bonds and Europeans are revalued in
difference form (F-07): a scenario's P&L is the deviation of the trade's value from its value
at the base, the shift carried to every flow by matrix products in the trade's compute format
(`TradePriceFunction.pnl`), so a scenario that moves nothing has exactly zero P&L and the legs'
cancellation stays in the base. Bermudans and Americans, whose kernel is not in difference form
yet, are revalued in full, `f(base + shift) - f(base)`, in one format, float32 or float64.

Scenarios are evaluated in batches of vmapped rows. The batch is bounded by memory per trade
(`scenario_batch_size`): a Bermudan's rollback interpolates every grid node at every quadrature
node for every cashflow column, about 80 MB per scenario at `n_per_std=64`, so a fixed batch of
a few hundred would need tens of gigabytes, while a loop over single scenarios would leave the
accelerator idle. The price function is not jitted as a closure, which would compile again on
every run: the pricers it calls are jitted with the trade as an argument, so a repeated run, or
another trade of the same shape, reuses their programs. (JAX keeps those batched programs in
internal caches of 2,048 entries, which a large enough job can overflow; jitting the batch on
`TradePriceFunction.pricer` and `.terms`, as the AD Greeks do, is I-81.)

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
from engine.pricing.lgm_grid import grid_half_width
from engine.risk.greeks.price_functions import curve_keys, on_pillars, trade_price_function
from engine.risk.market.factors import RateRiskFactors, curve_name

#: Upper bound on the working memory of one vmapped batch of scenarios.
BATCH_MEMORY_BUDGET = 512 * 2 ** 20


def factor_indices(cfg, factors: RateRiskFactors) -> Tuple[int, ...]:
    """The factor curves a trade reads, by position in `factors`; a curve that is not a factor
    is refused (`RateRiskFactors.index_of`)."""
    return tuple(factors.index_of(curve_name(*key)) for key in curve_keys(cfg))


def scenario_batch_size(cfg, pricing: PricingConfig, requested: int, itemsize: int) -> int:
    """Scenarios to vmap at once for `cfg`: `requested`, reduced so one batch
    stays within `BATCH_MEMORY_BUDGET`.

    Only the grid pricers need it. Their rollback materializes a
    `[nodes, quadrature nodes]` interpolation for the option, the underlying
    and each cached cashflow column, per scenario."""
    if not isinstance(cfg, (BermudanSwaptionConfig, AmericanSwaptionConfig)) or cfg.is_expired():
        return requested
    engine = pricing.american if isinstance(cfg, AmericanSwaptionConfig) else pricing.bermudan
    nodes = 2 * grid_half_width(engine.std_devs, engine.n_per_std) + 1
    prepared = prepared_option(cfg, engine, engine.volatility)
    columns = len(prepared.fixed_times) + len(prepared.float_pay_times) + 2
    per_scenario = nodes * nodes * columns * itemsize
    return max(1, min(requested, BATCH_MEMORY_BUDGET // per_scenario))


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
    batch_size: the most scenarios to revalue at once; a grid pricer may use
        fewer to stay within `BATCH_MEMORY_BUDGET`.
    """
    values = [revalue_trade(cfg, market, factors, moves, pricing, batch_size, arithmetic) for cfg in trades]
    return (np.asarray([base for base, _ in values]),
            jnp.stack([jnp.asarray(base, dtype=pnl.dtype) + pnl for base, pnl in values], axis=-1))


def revalue_trade(cfg, market: Market, factors: RateRiskFactors, moves, pricing: PricingConfig = PricingConfig(),
                  batch_size: int = 256, arithmetic: Optional[Arithmetic] = None) -> Tuple[float, jnp.ndarray]:
    """One trade's base value and its P&L `[S]`, in `arithmetic` (the moves' own format,
    float64 or float32, by default; `run_market_risk` gives each trade its pricing stage's), its
    accumulate format the moves' and the base's: in difference form where the trade's price
    function has one (`TradePriceFunction.pnl`), else the shocked values less the base value."""
    moves = jnp.asarray(moves)
    arithmetic = arithmetic or Arithmetic.of(moves.dtype)
    moves = arithmetic.cast(moves)
    base = arithmetic.cast(factors.base_rates())
    fn = trade_price_function(cfg, market, pricing, arithmetic.dtype)
    own = [factors.slice_of(i) for i in factor_indices(cfg, factors)]
    value = float(fn.price(*[base[s] for s in own]))
    batch = scenario_batch_size(cfg, pricing, batch_size, jnp.dtype(arithmetic.dtype).itemsize)
    if fn.pnl is not None:
        curves = on_pillars(fn.times, [base[s] for s in own])
        return value, _map_scenarios(
            lambda shifts: _pnl(fn.pnl, fn.terms, curves, tuple(shifts[..., s] for s in own), arithmetic), moves, batch)

    def shocked(vector):
        return fn.price(*[vector[s] for s in own])

    values = _map_scenarios(jax.vmap(lambda move: shocked(base + move)), moves, batch)
    return value, values - jnp.asarray(value, dtype=values.dtype)


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
