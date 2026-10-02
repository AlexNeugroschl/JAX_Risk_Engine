"""
Full revaluation of a portfolio at t=0 under shocked curves.

Every trade becomes a pure JAX function of the pillar rates of the market curves it reads, with
its configured engine (`engine.risk.price_functions.trade_price_function`, the functions AD
Greeks differentiate). The same function prices the base market and every scenario, so a
trade's P&L is exactly `f(base + shift) - f(base)`: there is no second pricer whose base could
disagree.

Scenarios are evaluated in batches of vmapped rows. The batch is bounded by memory per trade
(`scenario_batch_size`): a Bermudan's rollback interpolates every grid node at every quadrature
node for every cashflow column, about 80 MB per scenario at `n_per_std=64`, so a fixed batch of
a few hundred would need tens of gigabytes, while a loop over single scenarios would leave the
accelerator idle. The price function is not jitted as a closure, which would compile again on
every run: the pricers it calls are jitted with the trade as an argument, so a repeated run, or
another trade of the same shape, reuses their programs.

**Engines.** Each product's engine is the pricing configuration's (`PricingConfig`): a European
on ORE's Bachelier engine (the normal volatility read from the market, held fixed) or on
Jamshidian; a Bermudan/American on its LGM grid engine, calibrated on today's market and held
fixed (decision A-8: the engine is chosen by configuration, not by the trade).

**Which risk factors move.** Only curve pillar rates. Volatilities are held at their base
values (the surface, a calibrated LGM, the Jamshidian model), so volatility risk is not
captured. `run_market_risk` says so in its warnings for every option trade.
"""
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig, _grid_half_width
from engine.market import Market
from engine.market_risk.factors import RateRiskFactors, curve_name
from engine.risk.price_functions import curve_keys, trade_price_function
from engine.valuation.bermudan import prepared_option
from engine.valuation.config import PricingConfig

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
    nodes = 2 * _grid_half_width(engine.std_devs, engine.n_per_std) + 1
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
) -> Tuple[np.ndarray, jnp.ndarray]:
    """Base values `[N]` and shocked values `[S, N]` of every trade, computed in `moves`' dtype
    (`revalue_trade` per trade).

    moves: `[S, F]` absolute factor moves (`ShockScenarios.shifts`).
    batch_size: the most scenarios to vmap at once; a grid pricer may use
        fewer to stay within `BATCH_MEMORY_BUDGET`.
    """
    values = [revalue_trade(cfg, market, factors, moves, pricing, batch_size) for cfg in trades]
    return np.asarray([base for base, _ in values]), jnp.stack([shocked for _, shocked in values], axis=-1)


def revalue_trade(cfg, market: Market, factors: RateRiskFactors, moves, pricing: PricingConfig = PricingConfig(),
                  batch_size: int = 256) -> Tuple[float, jnp.ndarray]:
    """One trade's base value and its shocked values `[S]`, computed in `moves`' dtype (the
    trade's pricing compute dtype; `run_market_risk` casts them per trade)."""
    moves = jnp.asarray(moves)
    dtype = moves.dtype
    base = jnp.asarray(factors.base_rates(), dtype=dtype)
    fn = trade_price_function(cfg, market, pricing, dtype)
    own = [factors.slice_of(i) for i in factor_indices(cfg, factors)]

    def on_factors(vector):
        return fn.price(*[vector[s] for s in own])

    batch = scenario_batch_size(cfg, pricing, batch_size, jnp.dtype(dtype).itemsize)
    return float(on_factors(base)), _map_scenarios(on_factors, base, moves, batch)


def _map_scenarios(on_factors, base, moves, batch_size):
    """`on_factors(base + moves[s])` for every scenario `s`, vmapped in batches."""
    batched = jax.vmap(lambda move: on_factors(base + move))
    return jnp.concatenate([batched(moves[i:i + batch_size]) for i in range(0, moves.shape[0], batch_size)])
