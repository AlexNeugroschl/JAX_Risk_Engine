"""
Value at Risk and Expected Shortfall over a Monte Carlo NPV cube `[Scenarios, TimeSteps,
Trades]`. Instrument-agnostic: nothing here imports `engine.instruments`.

Statistics match `ORE.RiskStatistics` (QuantLib `GenericRiskStatistics`), checked against the
installed ORE package:

    sorted_pnl = ascending sort of the per-step P&L sample
    idx        = ceil(N * (1 - percentile)) - 1, clamped to [0, N - 1]   (`var_rank`)
    VaR(p)     = max(-sorted_pnl[idx], 0.0)     # positive loss, clamped at 0
    tail       = pnl[pnl < -VaR(p)]             # strict, value-based
    ES(p)      = -mean(tail)                    # NaN if the tail is empty

The quantile is an order statistic, not interpolated (as `numpy.percentile`'s default is):
QuantLib's `GeneralStatistics::percentile` advances through the sorted sample while the
cumulative weight is strictly below the target N(1 - p), so with unit weights it stops at
index ceil(N(1 - p)) - 1. Where N(1 - p) is not a whole number that is floor(N(1 - p)); where it
is (16 scenarios at 0.9375), it is the observation below. The target is computed as QuantLib
computes it, `(1 - p) * N` in float64, so a level whose 1 - p is inexact in binary (0.95, 0.99)
meets the boundary exactly when ORE's does. The ES tail is value-based, which differs from
slicing `sorted[0:idx]` when there are ties at the VaR boundary. Where the tail is empty ORE
raises "no data below the target"; this returns NaN, since traced JAX code cannot raise.

ORE accepts confidence levels in [0.9, 1); these functions take any level in (0, 1), a broader
domain on the same rule (`validate_quantiles` checks a run's levels).

P&L(scenario, t) = portfolio NPV(scenario, t) - base_npv, where `base_npv` is the t=0 NPV
supplied by the caller.

Differs from ORE: ORE's VaR is over a single horizon. Here the same P&L definition is
applied at every step of a risk-neutral exposure simulation, so each step's figure mixes
market risk with the portfolio's expected drift and roll-down. The cube's known limitations
(I-04, audit M-1, M-3) carry into these numbers. Market-risk VaR from t=0 revaluation is in
`engine.risk.market`.
"""
import math
import numbers
import warnings
from decimal import Decimal
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
# The cross-asset simulation (`engine.market_simulation`) runs under the pricing measure, so this
# module's output is risk-neutral-pricing. The label travels with the number (I-11).
RISK_MEASURE_RISK_NEUTRAL = "risk-neutral-pricing"
RISK_MEASURE_HISTORICAL = "historical-forecast"
RISK_MEASURE_STRESS = "deterministic-stress"
RISK_MEASURES = (
    RISK_MEASURE_RISK_NEUTRAL,
    RISK_MEASURE_HISTORICAL,
    RISK_MEASURE_STRESS,
)

#: The measure the simulation + this module produce. A fact, not a default.
ENGINE_RISK_MEASURE = RISK_MEASURE_RISK_NEUTRAL


def quantile_label(q: float) -> str:
    """Quantile as a result-key suffix, its percentage exactly: 0.95 -> "95", 0.975 -> "97.5",
    0.9500000001 -> "95.00000001". The shortest decimal that reads back as `q` (`repr`), its
    point moved two places in decimal arithmetic, so no rounding: two levels share a label only
    if they are the same float."""
    percent = (Decimal(repr(float(q))) * 100).normalize()
    return format(percent, "f")


def validate_quantiles(name: str, quantiles: Sequence[float]) -> None:
    """Refuse, naming the field `name`, a confidence level that is not a number in (0, 1), or
    a level given twice (its figures would share one key). ORE's statistics take [0.9, 1);
    this domain is broader, on the same rank rule."""
    labels = set()
    for q in quantiles:
        if isinstance(q, bool) or not isinstance(q, numbers.Real) or not 0.0 < float(q) < 1.0:
            raise ValueError(f"{name}: a quantile must lie in (0, 1); got {q!r}")
        if quantile_label(q) in labels:
            raise ValueError(f"{name}: {float(q)!r} is given more than once")
        labels.add(quantile_label(q))


def var_rank(num_scenarios: int, percentile: float) -> int:
    """The index into the ascending P&L sample that VaR at `percentile` reads, as QuantLib's
    `GeneralStatistics::percentile` (see the module docstring): ceil(N(1 - p)) - 1, clamped to
    [0, N - 1]. It is also the number of observations below that order statistic, the ES tail
    when there are no ties at it."""
    target = (1.0 - float(percentile)) * num_scenarios
    return min(max(math.ceil(target) - 1, 0), num_scenarios - 1)


def portfolio_pnl(npv_cube: jax.Array, base_npv: float) -> jax.Array:
    """[Scenarios, TimeSteps, Trades] -> [Scenarios, TimeSteps]: portfolio NPV minus
    `base_npv`."""
    portfolio_npv = jnp.sum(npv_cube, axis=-1)
    return portfolio_npv - base_npv


def value_at_risk(pnl: jax.Array, percentile: float) -> jax.Array:
    """
    [Scenarios, TimeSteps] P&L -> [TimeSteps] VaR at `percentile` (0.99 for 99%), as
    `ORE.RiskStatistics.valueAtRisk`: the order statistic at `var_rank`, negated, clamped
    at 0.

    `percentile` is converted to a Python float because it is a static jit argument and
    must be hashable.
    """
    return _value_at_risk_jit(pnl, float(percentile))


@partial(jax.jit, static_argnums=1)
def _value_at_risk_jit(pnl: jax.Array, percentile: float) -> jax.Array:
    # The rank is a compile-time constant (static percentile, static shape), computed in Python.
    sorted_pnl = jnp.sort(pnl, axis=0)
    return jnp.maximum(-sorted_pnl[var_rank(pnl.shape[0], percentile)], 0.0)


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
    strict tail `pnl < -VaR`). This is an observation count, not an effective independent
    sample size: at 99% over
    10,000 scenarios, about 100. 0 where `expected_shortfall` is NaN.
    """
    return _tail_sample_size_jit(pnl, float(percentile))


@partial(jax.jit, static_argnums=1)
def _tail_sample_size_jit(pnl: jax.Array, percentile: float) -> jax.Array:
    var = _value_at_risk_jit(pnl, percentile)
    return jnp.sum(pnl < -var[None, :], axis=0)


def conditional_tail_standard_error(pnl: jax.Array, percentile: float) -> jax.Array:
    """
    [Scenarios, TimeSteps] P&L -> [TimeSteps] legacy conditional tail-spread diagnostic:
    `s / sqrt(n)`, with `s` the sample standard deviation (ddof=1) of the tail and `n` the
    tail count. It holds the fitted threshold fixed and assumes IID tail observations;
    it is not the sampling error of the complete ES estimator (I-111). Market-risk runs
    use replicate deletion or historical moving blocks instead.

    NaN where `n < 2` (no spread can be estimated), never 0.
    """
    return _conditional_tail_standard_error_jit(pnl, float(percentile))


def expected_shortfall_standard_error(pnl: jax.Array, percentile: float) -> jax.Array:
    """Deprecated compatibility name for `conditional_tail_standard_error` (I-111).

    This is not the complete ES estimator's sampling standard error.
    """
    warnings.warn("expected_shortfall_standard_error is a conditional tail-spread diagnostic, not ES sampling "
                  "uncertainty; use conditional_tail_standard_error", DeprecationWarning, stacklevel=2)
    return conditional_tail_standard_error(pnl, percentile)


@partial(jax.jit, static_argnames=("percentile", "replicates"))
def replicated_es_error(pnl, percentile: float, replicates: int):
    """SE of the pooled ES by deleting independent randomized-QMC replicates.

    The tail and its threshold are recomputed after each deletion. A single scramble
    or a non-estimable delete-one tail yields NaN, rather than an IID tail error.
    """
    from engine.precision.uncertainty import delete_group, jackknife_error
    if replicates < 2:
        return jnp.asarray(jnp.nan, dtype=pnl.dtype)
    estimates = jax.lax.map(lambda i: expected_shortfall(delete_group(pnl, i, replicates)[:, None], percentile)[0],
                            jnp.arange(replicates))
    return jackknife_error(estimates)


@partial(jax.jit, static_argnames=("percentile", "block", "replicates"))
def historical_es_error(pnl, key, percentile: float, block: int, replicates: int):
    """Conditional moving-block bootstrap SE, preserving within-block dependence."""
    from engine.precision.uncertainty import moving_block_indices
    if block * 2 > pnl.shape[0]:
        return jnp.asarray(jnp.nan, dtype=pnl.dtype)
    def resample(index):
        indices = moving_block_indices(jax.random.fold_in(key, index), pnl.shape[0], block, 1)[0]
        return expected_shortfall(pnl[indices, None], percentile)[0]
    # Only one resampled history is live at once, including at large bootstrap counts.
    estimates = jax.lax.map(resample, jnp.arange(replicates))
    return jnp.std(estimates, ddof=1)


@partial(jax.jit, static_argnums=1)
def _conditional_tail_standard_error_jit(pnl: jax.Array, percentile: float) -> jax.Array:
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
    `ES_<q>_standardError` per percentile. The latter is the legacy conditional tail-spread
    diagnostic above, not dependence-aware ES uncertainty; market-risk runs replace it.

    The risk measure is not labelled here, because a cube does not say which measure
    produced it; `RISK_MEASURE_*` and `engine.traderx.result` carry it (I-11).
    """
    pnl = portfolio_pnl(npv_cube, base_npv)
    metrics: Dict[str, jax.Array] = {}
    for p in percentiles:
        label = quantile_label(p)
        metrics[f"VaR_{label}"] = value_at_risk(pnl, p)
        metrics[f"ES_{label}"] = expected_shortfall(pnl, p)
        if include_diagnostics:
            metrics[f"ES_{label}_tailCount"] = tail_sample_size(pnl, p)
            # Keep the published key for compatibility; its conditional meaning is
            # explicit in the canonical helper's name and the API documentation.
            metrics[f"ES_{label}_standardError"] = conditional_tail_standard_error(pnl, p)
    return metrics
