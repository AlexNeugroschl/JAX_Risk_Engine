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

The revaluation itself is the bump Greeks' (`engine.risk.greeks.scenario_pnl.price_function_pnl`):
scenarios in batches bounded by memory per trade (`scenario_batch_size`: a Bermudan's induction
holds its columns on the state grid for every scenario of the batch), one program per product,
shape and arithmetic, with the trade's terms as an argument, so a repeated run, or another trade
of the same shape, reuses its programs.

**Engines.** Each product's engine is the pricing configuration's (`PricingConfig`): a European
on ORE's Bachelier engine (the normal volatility read from the market, held fixed) or on
Jamshidian; a Bermudan/American on its LGM grid engine (decision A-8: the engine is chosen by
configuration, not by the trade). A calibrated LGM is recalibrated under each scenario, as ORE's
historical P&L recalibrates the trades' model builders (`HistoricalPnlGenerator`'s
`ValuationEngine` with `factory->modelBuilders()`, I-112): a first pass calibrates it on every
scenario's curves, in float64 (A-10), and refuses the run, naming the trade and the scenarios,
where a calibration misses ORE's acceptance (ORE drops that trade's sample, logging an error; the
engine refuses a misfit everywhere, A-24); each scenario is then priced on its own model
(`engine.risk.greeks.price_functions.recalibrated_pnl`). An LGM on a fixed model
(`calibration="None"`) is the same in every scenario.

**Which risk factors move.** Only curve pillar rates. Volatilities are held at their base
values (the surface, the Jamshidian model), so volatility risk is not captured.
`run_market_risk` says so in its warnings for every option trade.
"""
from typing import Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

from engine.market_data.market import Market
from engine.precision import Arithmetic
from engine.pricing.config import PricingConfig
from engine.risk.greeks.price_functions import Moves, curve_keys, trade_price_function
from engine.risk.greeks.scenario_pnl import price_function_pnl
from engine.risk.market.factors import RateRiskFactors, curve_name


def factor_indices(cfg, factors: RateRiskFactors) -> Tuple[int, ...]:
    """The factor curves a trade reads, by position in `factors`; a curve that is not a factor
    is refused (`RateRiskFactors.index_of`)."""
    return tuple(factors.index_of(curve_name(*key)) for key in curve_keys(cfg))


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
        within `induction_memory_budget()` per device.
    """
    execution = _prepare_revaluation(moves, arithmetic)
    values = [_revalue_trade(cfg, market, factors, moves, pricing, batch_size, arithmetic, execution) for cfg in trades]
    return (np.asarray([base for base, _ in values]),
            jnp.stack([jnp.asarray(base, dtype=pnl.dtype) + pnl for base, pnl in values], axis=-1))


def revalue_trade(cfg, market: Market, factors: RateRiskFactors, moves, pricing: PricingConfig = PricingConfig(),
                  batch_size: int = 256, arithmetic: Optional[Arithmetic] = None) -> Tuple[float, jnp.ndarray]:
    """One trade's base value and its P&L `[S]`, in `arithmetic` (the moves' own format,
    float64 or float32, by default; `run_market_risk` gives each trade its pricing stage's), its
    accumulate format the moves' and the base's, in difference form (`TradePriceFunction.pnl`).
    An option recalibrated under each scenario (`TradePriceFunction.calibration`) is calibrated
    first, in float64; a scenario whose calibration misses ORE's acceptance is refused
    (`CalibrationError`, naming the trade and the scenarios)."""
    execution = _prepare_revaluation(moves, arithmetic)
    return _revalue_trade(cfg, market, factors, moves, pricing, batch_size, arithmetic, execution)


def _prepare_revaluation(moves, arithmetic):
    from engine.precision import Precision
    from engine.precision.execution import prepare_execution
    execution = prepare_execution(Precision.reference(), arrays=moves)
    if arithmetic is not None:
        from engine.precision.execution import validate_product
        validate_product(arithmetic.compute, arithmetic.accumulate, arithmetic.execution)
    return execution


def _revalue_trade(cfg, market, factors, moves, pricing, batch_size, arithmetic, execution):
    """Internal single-trade revaluation; the job has validated all pricing overrides."""
    moves = jnp.asarray(moves)
    arithmetic = arithmetic or Arithmetic.of(moves.dtype)
    moves = arithmetic.cast(moves)
    base = arithmetic.cast(factors.base_rates())
    fn = trade_price_function(cfg, market, pricing, arithmetic.dtype)
    own = [factors.slice_of(i) for i in factor_indices(cfg, factors)]
    return price_function_pnl(cfg, fn, [base[s] for s in own], Moves(tuple(moves[..., s] for s in own)), pricing,
                              batch_size, arithmetic)
