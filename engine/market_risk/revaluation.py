"""
Full revaluation of a portfolio at t=0 under shocked curves.

Every trade becomes a pure JAX function of the pillar rates of the curves it
depends on (`engine.risk.price_functions`). The same function prices the base
market and every scenario, so a trade's P&L is exactly `f(base + shift) -
f(base)`: there is no second pricer whose base could disagree.

Scenarios are evaluated with `jax.lax.map` in batches of vmapped rows. The
batch is bounded by memory per trade (`scenario_batch_size`): a Bermudan's
rollback interpolates every grid node at every quadrature node for every
cashflow column, about 80 MB per scenario at `n_per_std=64`, so a fixed
batch of a few hundred would need tens of gigabytes, while a Python loop
would leave the accelerator idle.

**Which risk factors move.** Only curve pillar rates. Model parameters are
held at their base values: a swaption's `hw_a`/`hw_sigma` do not move, so
volatility risk is not captured. `run_market_risk` says so in its warnings
for every option trade.
"""
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig, _grid_half_width, prepare_bermudan
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig
from engine.instruments.treasury import BondConfig
from engine.market_risk.factors import RateRiskFactors
from engine.models.hull_white import ZeroCurve
from engine.risk.price_functions import (
    bermudan_price_function,
    bond_price_function,
    swap_price_function,
    swaption_price_function,
)

OPTION_TYPES = (SwaptionConfig, BermudanSwaptionConfig, AmericanSwaptionConfig)

#: Upper bound on the working memory of one vmapped batch of scenarios.
BATCH_MEMORY_BUDGET = 512 * 2 ** 20


@dataclass(frozen=True)
class TradeRevaluer:
    """One trade's t=0 price as a function of its curves' pillar rates:
    `price(*[rates of curve i for i in curve_indices])`."""
    curve_indices: Tuple[int, ...]
    price: Callable


def curve_indices(cfg) -> Tuple[int, ...]:
    """The risk-factor curves a trade depends on, by index."""
    if isinstance(cfg, SwapConfig):
        return (cfg.discount_curve_index, cfg.forward_curve_index)
    if isinstance(cfg, OPTION_TYPES):
        return (cfg.rate_factor_index,)
    if isinstance(cfg, BondConfig):
        if cfg.curve_index is None:
            raise ValueError(
                "a BondConfig in a market-risk run must set curve_index: the market curve it "
                "discounts on"
            )
        return (cfg.curve_index,)
    raise TypeError(f"no revaluation for trade type {type(cfg).__name__}")


def scenario_batch_size(cfg, requested: int, itemsize: int) -> int:
    """Scenarios to vmap at once for `cfg`: `requested`, reduced so one batch
    stays within `BATCH_MEMORY_BUDGET`.

    Only the grid pricers need it. Their rollback materializes a
    `[nodes, quadrature nodes]` interpolation for the option, the underlying
    and each cached cashflow column, per scenario."""
    if not isinstance(cfg, (BermudanSwaptionConfig, AmericanSwaptionConfig)):
        return requested
    nodes = 2 * _grid_half_width(cfg.std_devs, cfg.n_per_std) + 1
    prepared = prepare_bermudan(cfg)
    columns = len(prepared.fixed_times) + len(prepared.float_pay_times) + 2
    per_scenario = nodes * nodes * columns * itemsize
    return max(1, min(requested, BATCH_MEMORY_BUDGET // per_scenario))


def build_revaluer(cfg, factors: RateRiskFactors, dtype) -> TradeRevaluer:
    """The pure price function for one trade, at `dtype`. Curve pillar
    times come from `factors`; the trade must already be validated to
    reference curves that exist (see `engine.market_risk.run`)."""
    indices = curve_indices(cfg)
    curves = [ZeroCurve.from_config(factors.curves[i], dtype=dtype) for i in indices]
    if isinstance(cfg, SwapConfig):
        price = swap_price_function(cfg, curves[0], curves[1])
    elif isinstance(cfg, SwaptionConfig):
        price = swaption_price_function(cfg, curves[0])
    elif isinstance(cfg, (BermudanSwaptionConfig, AmericanSwaptionConfig)):
        with_sigma, sigma_values = bermudan_price_function(cfg, curves[0])

        def price(rates, _f=with_sigma, _s=sigma_values):
            return _f(rates, jnp.asarray(_s, dtype=dtype))
    else:
        price = bond_price_function(cfg)
    return TradeRevaluer(curve_indices=indices, price=price)


def revalue(
    trades: Sequence,
    factors: RateRiskFactors,
    shifts: np.ndarray,
    dtype=jnp.float64,
    batch_size: int = 256,
) -> Tuple[np.ndarray, jnp.ndarray]:
    """Base values `[N]` and shocked values `[S, N]` of every trade.

    shifts: `[S, F]` absolute factor moves (`ShockScenarios.shifts`).
    batch_size: the most scenarios to vmap at once; a grid pricer may use
        fewer to stay within `BATCH_MEMORY_BUDGET`.
    """
    base = jnp.asarray(factors.base_rates(), dtype=dtype)
    moves = jnp.asarray(shifts, dtype=dtype)
    slices = [factors.slice_of(i) for i in range(len(factors.curves))]

    base_values: List[float] = []
    shocked_columns: List[jnp.ndarray] = []
    for cfg in trades:
        revaluer = build_revaluer(cfg, factors, dtype)
        own = [slices[i] for i in revaluer.curve_indices]

        def on_factors(vector, _price=revaluer.price, _own=own):
            return _price(*[vector[s] for s in _own])

        base_values.append(float(jax.jit(on_factors)(base)))
        batch = scenario_batch_size(cfg, batch_size, jnp.dtype(dtype).itemsize)
        shocked_columns.append(_map_scenarios(on_factors, base, moves, batch))
    return np.asarray(base_values), jnp.stack(shocked_columns, axis=-1)


def _map_scenarios(on_factors, base, moves, batch_size):
    """`on_factors(base + moves[s])` for every scenario `s`, in batches."""
    @jax.jit
    def run(all_moves):
        return jax.lax.map(lambda move: on_factors(base + move), all_moves, batch_size=batch_size)
    return run(moves)
