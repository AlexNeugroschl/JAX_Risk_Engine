"""
Value at Risk and Expected Shortfall over a Monte Carlo NPV cube `[Scenarios, TimeSteps,
Trades]`. Instrument-agnostic: nothing here imports `engine.instruments`.

Statistics match `ORE.RiskStatistics` (QuantLib `GenericRiskStatistics`), checked against the
installed ORE package:

    sorted_pnl = ascending sort of the per-step P&L sample
    idx        = floor(N * (1 - percentile))
    VaR(p)     = max(-sorted_pnl[idx], 0.0)     # positive loss, clamped at 0
    tail       = pnl[pnl < -VaR(p)]             # strict, value-based
    ES(p)      = -mean(tail)                    # NaN if the tail is empty

The quantile is a nearest-rank-below order statistic, not interpolated (as
`numpy.percentile`'s default is). The ES tail is value-based, which differs from slicing
`sorted[0:idx]` when there are ties at the VaR boundary. Where the tail is empty ORE raises
"no data below the target"; this returns NaN, since traced JAX code cannot raise.

P&L(scenario, t) = portfolio NPV(scenario, t) - base_npv, where `base_npv` is the t=0 NPV
supplied by the caller.

Differs from ORE: ORE's VaR is over a single horizon. Here the same P&L definition is
applied at every step of a risk-neutral exposure simulation, so each step's figure mixes
market risk with the portfolio's expected drift and roll-down. The cube's known limitations
(I-04, audit M-1, M-3) carry into these numbers. Market-risk VaR from t=0 revaluation is in
`engine.market_risk`.
"""
import math
from functools import partial
from typing import Dict, Sequence

import jax
import jax.numpy as jnp

# Risk-measure labels. A VaR/ES figure is only meaningful with the measure that generated it:
#
#   - risk-neutral-pricing   exposure simulation under the pricing measure (exposure, CVA,
#                            limits); not a forecast of real-world P&L.
#   - historical-forecast    a calibrated real-world loss forecast (capital, backtesting).
#   - deterministic-stress   revaluation under a prescribed scenario; no probability.
#
# `generate_paths` simulates under the pricing measure, so this module's output is
# risk-neutral-pricing. The label travels with the number (I-11).
RISK_MEASURE_RISK_NEUTRAL = "risk-neutral-pricing"
RISK_MEASURE_HISTORICAL = "historical-forecast"
RISK_MEASURE_STRESS = "deterministic-stress"
RISK_MEASURES = (
    RISK_MEASURE_RISK_NEUTRAL,
    RISK_MEASURE_HISTORICAL,
    RISK_MEASURE_STRESS,
)

#: The measure `generate_paths` + this module produce. A fact, not a default.
ENGINE_RISK_MEASURE = RISK_MEASURE_RISK_NEUTRAL


def quantile_label(q: float) -> str:
    """Quantile as a result-key suffix: 0.95 -> "95", 0.975 -> "97.5" (never rounded, so
    two quantiles cannot share a key)."""
    percent = round(q * 100, 6)
    return f"{int(percent)}" if percent == int(percent) else f"{percent:g}"


def portfolio_pnl(npv_cube: jax.Array, base_npv: float) -> jax.Array:
    """[Scenarios, TimeSteps, Trades] -> [Scenarios, TimeSteps]: portfolio NPV minus
    `base_npv`."""
    portfolio_npv = jnp.sum(npv_cube, axis=-1)
    return portfolio_npv - base_npv


def value_at_risk(pnl: jax.Array, percentile: float) -> jax.Array:
    """
    [Scenarios, TimeSteps] P&L -> [TimeSteps] VaR at `percentile` (0.99 for 99%), as
    `ORE.RiskStatistics.valueAtRisk`: the nearest-rank-below order statistic, negated, clamped
    at 0.

    `percentile` is converted to a Python float because it is a static jit argument and
    must be hashable.
    """
    return _value_at_risk_jit(pnl, float(percentile))


@partial(jax.jit, static_argnums=1)
def _value_at_risk_jit(pnl: jax.Array, percentile: float) -> jax.Array:
    num_scenarios = pnl.shape[0]
    # The rank index is a compile-time constant (static percentile, static shape), so it
    # is computed in Python; a traced index could not be converted with int().
    idx = int(math.floor(num_scenarios * (1.0 - percentile)))
    idx = min(max(idx, 0), num_scenarios - 1)
    sorted_pnl = jnp.sort(pnl, axis=0)
    return jnp.maximum(-sorted_pnl[idx], 0.0)


def expected_shortfall(pnl: jax.Array, percentile: float) -> jax.Array:
    """
    [Scenarios, TimeSteps] P&L -> [TimeSteps] Expected Shortfall at `percentile`, as
    `ORE.RiskStatistics.expectedShortfall`: minus the mean of observations strictly below
    -VaR.

    NaN where that tail is empty (ORE raises there); callers must check for NaN.
    """
    return _expected_shortfall_jit(pnl, float(percentile))


@partial(jax.jit, static_argnums=1)
def _expected_shortfall_jit(pnl: jax.Array, percentile: float) -> jax.Array:
    var = _value_at_risk_jit(pnl, percentile)
    tail_mask = pnl < -var[None, :]
    masked = jnp.where(tail_mask, pnl, jnp.nan)
    return -jnp.nanmean(masked, axis=0)


def tail_sample_size(pnl: jax.Array, percentile: float) -> jax.Array:
    """
    [Scenarios, TimeSteps] P&L -> [TimeSteps] number of observations in the ES mean (the
    strict tail `pnl < -VaR`). The effective sample size of the tail estimate: at 99% over
    10,000 scenarios, about 100. 0 where `expected_shortfall` is NaN.
    """
    return _tail_sample_size_jit(pnl, float(percentile))


@partial(jax.jit, static_argnums=1)
def _tail_sample_size_jit(pnl: jax.Array, percentile: float) -> jax.Array:
    var = _value_at_risk_jit(pnl, percentile)
    return jnp.sum(pnl < -var[None, :], axis=0)


def expected_shortfall_standard_error(pnl: jax.Array, percentile: float) -> jax.Array:
    """
    [Scenarios, TimeSteps] P&L -> [TimeSteps] Monte Carlo standard error of the ES estimate:
    `s / sqrt(n)`, with `s` the sample standard deviation (ddof=1) of the tail and `n` the
    tail count. Makes a sparse tail visible (I-11).

    NaN where `n < 2` (no spread can be estimated), never 0.
    """
    return _es_standard_error_jit(pnl, float(percentile))


@partial(jax.jit, static_argnums=1)
def _es_standard_error_jit(pnl: jax.Array, percentile: float) -> jax.Array:
    var = _value_at_risk_jit(pnl, percentile)
    tail_mask = pnl < -var[None, :]
    masked = jnp.where(tail_mask, pnl, jnp.nan)

    count = jnp.sum(tail_mask, axis=0)
    # ddof=1: `jnp.nanstd` has no ddof, so rescale s_pop by sqrt(n/(n-1)).
    population_std = jnp.nanstd(masked, axis=0)

    # Keep every intermediate in the P&L's dtype: integer `count` would promote the result
    # to float64 under jax_enable_x64 and defeat a float32 precision override.
    dtype = population_std.dtype
    safe_count = jnp.maximum(count, 2).astype(dtype)  # guards n<2; masked out below
    one = jnp.asarray(1, dtype=dtype)
    sample_std = population_std * jnp.sqrt(safe_count / (safe_count - one))

    standard_error = sample_std / jnp.sqrt(safe_count)
    # n < 2: no spread is estimable; NaN, not 0.
    return jnp.where(count >= 2, standard_error, jnp.asarray(jnp.nan, dtype=dtype))


def compute_risk_metrics(
    npv_cube: jax.Array,
    base_npv: float,
    percentiles: Sequence[float] = (0.95, 0.99),
    include_diagnostics: bool = True,
) -> Dict[str, jax.Array]:
    """
    [Scenarios, TimeSteps, Trades] NPV cube and t=0 portfolio NPV -> `{"VaR_95": [T],
    "ES_95": [T], ...}`, VaR and ES together at each percentile, as ORE reports them.

    With `include_diagnostics` (the default), also `ES_<q>_tailCount` and
    `ES_<q>_standardError` per percentile.

    The risk measure is not labelled here, because a cube does not say which measure
    produced it; `RISK_MEASURE_*` and `engine.integration.result` carry it (I-11).
    """
    pnl = portfolio_pnl(npv_cube, base_npv)
    metrics: Dict[str, jax.Array] = {}
    for p in percentiles:
        label = quantile_label(p)
        metrics[f"VaR_{label}"] = value_at_risk(pnl, p)
        metrics[f"ES_{label}"] = expected_shortfall(pnl, p)
        if include_diagnostics:
            metrics[f"ES_{label}_tailCount"] = tail_sample_size(pnl, p)
            metrics[f"ES_{label}_standardError"] = expected_shortfall_standard_error(pnl, p)
    return metrics


# Demo
if __name__ == "__main__":
    from engine.simulation.market_model import generate_paths
    from engine.instruments.swap import SwapConfig, price_swaps
    from engine.simulation.demo_scenarios import EVAL_DATE, SWAP_DEMO_MATURITIES, flat_yield_curves, single_currency_swap_demo_config

    market_cubes = generate_paths(single_currency_swap_demo_config())

    # fixed_rate near the 3.5% forward rate, so the swap starts near fair value and the
    # demo shows two-sided P&L.
    swap_cfg = SwapConfig(
        notional=1_000_000.0,
        fixed_rate=0.035,
        payer=True,
        discount_curve_index=0,
        forward_curve_index=1,
        swap_tenor="2Y",
        # Explicit: SWAP_DEMO_MATURITIES is pinned to EVAL_DATE (I-28).
        evaluation_date=EVAL_DATE,
    )
    npv_cube = price_swaps(market_cubes["yield_curves"], SWAP_DEMO_MATURITIES, [swap_cfg])

    # t=0 baseline: the swap on today's (unshocked) curves.
    base_cube = flat_yield_curves(disc_rate=0.030, fwd_rate=0.035)
    base_npv = float(price_swaps(base_cube, SWAP_DEMO_MATURITIES, [swap_cfg])[0, 0, 0])

    metrics = compute_risk_metrics(npv_cube, base_npv, percentiles=(0.95, 0.99))
    print("Base (t=0) NPV:", base_npv)
    for key, values in metrics.items():
        print(f"{key}: {[round(float(v), 2) for v in values]}")
