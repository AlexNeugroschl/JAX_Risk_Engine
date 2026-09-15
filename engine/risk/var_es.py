"""
Value at Risk (VaR) and Expected Shortfall (ES) over a Monte Carlo NPV cube.

Instrument-agnostic: this module only ever consumes a
[Scenarios, TimeSteps, Trades] NPV cube (the shape every pricer under
engine/instruments/ returns) and a scalar base NPV. It never imports
engine.instruments.* -- any pricer producing that shape works here.

Mathematically matched to ORE's RiskStatistics (venv/Lib/site-packages/
ORE.py, QuantLib::GeneralStatistics-derived), live-verified against the
installed ORE package rather than assumed from documentation:

    sorted_pnl = ascending sort of the per-step P&L sample
    idx        = floor(N * (1 - percentile))
    VaR(p)     = max(-sorted_pnl[idx], 0.0)     # positive loss, clamped at 0
    tail       = pnl[pnl < -VaR(p)]             # STRICT value-based filter
    ES(p)      = -mean(tail)                    # NaN if tail is empty

Two details that are easy to get wrong and were confirmed by direct,
adversarial live testing against ORE.RiskStatistics (not read from any
formula reference):

1. The quantile is a lower/nearest-rank-below order statistic
   (`floor(N*(1-p))` indexing into the ascending-sorted sample), NOT
   linearly interpolated between order statistics the way
   `numpy.percentile`'s default method is. Verified with non-round
   `N*(1-p)` values where the two conventions diverge.
2. Expected Shortfall averages the STRICT value-based tail (`pnl < -VaR`),
   not a positional slice of the sorted array (`sorted[0:idx]`). The two
   formulas agree only when there are no ties at the VaR boundary; with
   ties, ORE's result matches only the value-based filter -- verified with
   a sample containing repeated values exactly at the VaR cutoff. When
   that filter is empty (the worst observations are all tied exactly at
   VaR), ORE's own `expectedShortfall` raises `RuntimeError: no data below
   the target`; this module returns NaN for that (percentile, time step)
   instead, since JAX cannot raise from traced code -- callers must check
   for NaN explicitly (see expected_shortfall's docstring).

P&L definition: P&L(scenario, t) = portfolio_NPV(scenario, t) - base_npv,
where base_npv is the portfolio's NPV at t=0 (before any simulated shocks),
supplied by the caller -- not inferred from the cube, and not a per-step
mean. This matches ORE's historical-VaR P&L definition literally
(NPV(scenario) - NPV(base case)), applied at every simulated time step, so
the resulting VaR/ES profile reflects both market risk and the portfolio's
expected drift/rolldown over time -- a deliberate choice, not the
alternative of measuring deviation from each step's own cross-scenario mean
(which would isolate pure risk from drift; that was considered and rejected
in favor of ORE's literal semantics).
"""
import math
from functools import partial
from typing import Dict, Sequence

import jax
import jax.numpy as jnp

# =============================================================================
# RISK MEASURE VOCABULARY (plan §W0.6; part of I-11)
#
# **A risk number without its measure is unactionable.** `VaR_95 = 2.1mm`
# means materially different things depending on what generated it, and
# nothing in a bare float distinguishes them:
#
#   - risk-neutral-pricing   an exposure simulation under the pricing
#                            measure. Correct for CVA/exposure/limits. NOT
#                            a forecast of tomorrow's P&L -- the drift is
#                            the risk-neutral one, not the real-world one.
#   - historical-forecast    a calibrated real-world forecast of realised
#                            loss. What a capital or backtesting process
#                            wants, and what this engine does NOT produce.
#   - deterministic-stress   a prescribed scenario's revaluation. No
#                            probability attaches to it at all.
#
# Everything this engine computes today is `risk-neutral-pricing`: it
# simulates under the pricing measure. Reporting a risk-neutral exposure
# where a consumer expects a historical forecast is a category error that
# no amount of numerical accuracy fixes, which is why the label travels
# with the number rather than living in documentation.
# =============================================================================
RISK_MEASURE_RISK_NEUTRAL = "risk-neutral-pricing"
RISK_MEASURE_HISTORICAL = "historical-forecast"
RISK_MEASURE_STRESS = "deterministic-stress"
RISK_MEASURES = (
    RISK_MEASURE_RISK_NEUTRAL,
    RISK_MEASURE_HISTORICAL,
    RISK_MEASURE_STRESS,
)

#: What `generate_paths` + this module actually produce. Not a default a
#: caller may override -- it is a statement of fact about the simulation.
ENGINE_RISK_MEASURE = RISK_MEASURE_RISK_NEUTRAL


def portfolio_pnl(npv_cube: jax.Array, base_npv: float) -> jax.Array:
    """
    [Scenarios, TimeSteps, Trades] -> [Scenarios, TimeSteps] portfolio P&L,
    summing across trades and subtracting the fixed t=0 baseline. Pure
    tensor op -- no instrument-specific knowledge.
    """
    portfolio_npv = jnp.sum(npv_cube, axis=-1)
    return portfolio_npv - base_npv


def value_at_risk(pnl: jax.Array, percentile: float) -> jax.Array:
    """
    [Scenarios, TimeSteps] P&L -> [TimeSteps] VaR at `percentile`
    (e.g. 0.99 for 99% VaR), matching ORE.RiskStatistics.valueAtRisk exactly:
    the lower/nearest-rank-below order statistic of the ascending-sorted
    per-step P&L sample, sign-flipped to a positive loss and clamped at 0.

    `percentile` is coerced to a plain Python float before reaching the
    jitted implementation below. It is a jit STATIC argument there (the rank
    index it produces is a compile-time constant, not a traced value), and
    static arguments must be hashable -- a 0-d `np.ndarray` or JAX scalar
    is not, and would otherwise raise "Non-hashable static arguments are not
    supported". Coercing here keeps every scalar-like `percentile` this
    function has always accepted working unchanged.
    """
    return _value_at_risk_jit(pnl, float(percentile))


@partial(jax.jit, static_argnums=1)
def _value_at_risk_jit(pnl: jax.Array, percentile: float) -> jax.Array:
    num_scenarios = pnl.shape[0]
    # Plain Python math, deliberately not jnp: `percentile` is a static
    # argument and `num_scenarios` comes from the shape, so this rank index
    # is a compile-time constant. Computing it via `jnp.floor` would build a
    # traced array and then need `int()` to index with it, which raises
    # ConcretizationTypeError under jit.
    idx = int(math.floor(num_scenarios * (1.0 - percentile)))
    idx = min(max(idx, 0), num_scenarios - 1)
    sorted_pnl = jnp.sort(pnl, axis=0)
    return jnp.maximum(-sorted_pnl[idx], 0.0)


def expected_shortfall(pnl: jax.Array, percentile: float) -> jax.Array:
    """
    [Scenarios, TimeSteps] P&L -> [TimeSteps] Expected Shortfall at
    `percentile`, matching ORE.RiskStatistics.expectedShortfall exactly:
    the negated mean of every P&L observation STRICTLY worse than
    -VaR(percentile) (a value-based filter, not a positional slice of the
    sorted array -- the two differ whenever the tail has ties at the VaR
    boundary, verified directly against ORE).

    Returns NaN for any time step whose strict tail is empty (every
    observation tied exactly at the VaR cutoff) -- the case where ORE's own
    expectedShortfall raises RuntimeError("no data below the target").
    Callers must check `jnp.isnan(...)` explicitly; this is a real,
    data-dependent edge case, not an oversight.

    `percentile` is coerced to a plain Python float for the same
    static-argument-hashability reason as `value_at_risk` above.
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
    [Scenarios, TimeSteps] P&L -> [TimeSteps] count of observations that
    actually entered the Expected Shortfall mean at `percentile`.

    **This is the effective sample size for a tail statistic** (plan §W0.6),
    and it is usually far smaller than the scenario count: at 99% over
    10,000 scenarios roughly 100 observations carry the estimate, and every
    one of them is in the part of the distribution the Monte Carlo sampled
    least. The tail count is what makes a sparse estimate distinguishable
    from a well-converged one -- without it, `ES_99` computed from 3
    observations and from 300 are the same number on the wire.

    Counts the STRICT value-based tail (`pnl < -VaR`), i.e. exactly the
    observations `expected_shortfall` averages -- not a positional
    `floor(N*(1-p))` slice, which would disagree whenever there are ties at
    the VaR boundary (see this module's docstring, point 2). A count of 0
    is the case where `expected_shortfall` returns NaN.
    """
    return _tail_sample_size_jit(pnl, float(percentile))


@partial(jax.jit, static_argnums=1)
def _tail_sample_size_jit(pnl: jax.Array, percentile: float) -> jax.Array:
    var = _value_at_risk_jit(pnl, percentile)
    return jnp.sum(pnl < -var[None, :], axis=0)


def expected_shortfall_standard_error(pnl: jax.Array, percentile: float) -> jax.Array:
    """
    [Scenarios, TimeSteps] P&L -> [TimeSteps] Monte Carlo standard error of
    the Expected Shortfall estimate at `percentile`.

    The plain standard error of the tail mean: `s / sqrt(n)`, where `s` is
    the sample standard deviation of the tail observations and `n` is
    `tail_sample_size`. ES is an average over the tail, so the standard
    error of that average is what quantifies its Monte Carlo noise.

    **Why report it.** `ES_99 = 1,240,000` reads as a precise figure. With
    a standard error of 380,000 it is not one, and nothing else in the
    result would say so -- a sparse tail and a converged one are otherwise
    indistinguishable (I-11). This does not make the estimate better; it
    makes its uncertainty visible, which is the difference between a number
    a reader can weigh and one they must simply trust.

    Uses the sample standard deviation (`ddof=1`, dividing by `n-1`): the
    tail IS a sample, and the population form would understate the spread.

    Returns NaN where `n < 2` -- with one observation there is no spread to
    estimate, and with none there is no estimate at all. That is the honest
    answer, and it is deliberately NOT 0.0, which would read as "perfectly
    converged" for the single worst case where the estimate is least
    trustworthy. Callers must check `jnp.isnan(...)`, exactly as they
    already must for `expected_shortfall` itself.
    """
    return _es_standard_error_jit(pnl, float(percentile))


@partial(jax.jit, static_argnums=1)
def _es_standard_error_jit(pnl: jax.Array, percentile: float) -> jax.Array:
    var = _value_at_risk_jit(pnl, percentile)
    tail_mask = pnl < -var[None, :]
    masked = jnp.where(tail_mask, pnl, jnp.nan)

    count = jnp.sum(tail_mask, axis=0)
    # ddof=1 on the masked tail. `jnp.nanstd` has no ddof, so the Bessel
    # correction is applied by rescaling: s_sample = s_pop * sqrt(n/(n-1)).
    population_std = jnp.nanstd(masked, axis=0)
    safe_count = jnp.maximum(count, 2)  # guards the n<2 slots; masked out below
    sample_std = population_std * jnp.sqrt(safe_count / (safe_count - 1))

    standard_error = sample_std / jnp.sqrt(safe_count)
    # n < 2: no spread is estimable. NaN, not 0.0 -- see the docstring.
    return jnp.where(count >= 2, standard_error, jnp.nan)


def compute_risk_metrics(
    npv_cube: jax.Array,
    base_npv: float,
    percentiles: Sequence[float] = (0.95, 0.99),
    include_diagnostics: bool = True,
) -> Dict[str, jax.Array]:
    """
    Public entry point: [Scenarios, TimeSteps, Trades] NPV cube + t=0
    portfolio NPV -> dict of [TimeSteps] VaR/ES arrays, one pair per
    requested percentile, keyed "VaR_95"/"ES_95"/"VaR_99"/"ES_99"/... --
    matching ORE's convention of always reporting VaR and ES together at
    each configured quantile (see module docstring / plan for the recovered
    ore_histsimvar.xml evidence).

    **Convergence diagnostics** (plan §W0.6, part of I-11) are added
    alongside, two per percentile:

        "ES_99_tailCount"      effective sample size -- how many
                               observations the ES mean actually averaged
        "ES_99_standardError"  Monte Carlo standard error of that mean

    `include_diagnostics=False` returns exactly the pre-W0.6 key set, for a
    caller that wants the bare statistics.

    **The existing VaR_*/ES_* keys and values are unchanged**, deliberately.
    This is purely additive: the diagnostics sit beside the statistics
    rather than wrapping them, so every existing consumer keeps working
    untouched and no number moves. What was missing was never the tail
    statistics themselves -- it was any way to tell a sparse estimate from
    a converged one.

    **What this does NOT do.** It does not label the measure. A risk-neutral
    exposure simulation is not a calibrated forecast of tomorrow's loss, and
    that distinction belongs to the run, not to a single cube -- this
    function cannot know which it was handed. `RISK_MEASURE_*` below and
    `engine.integration.result` carry it at the level that does know. See
    I-11 in docs/known-issues.md.
    """
    pnl = portfolio_pnl(npv_cube, base_npv)
    metrics: Dict[str, jax.Array] = {}
    for p in percentiles:
        label = f"{int(round(p * 100))}"
        metrics[f"VaR_{label}"] = value_at_risk(pnl, p)
        metrics[f"ES_{label}"] = expected_shortfall(pnl, p)
        if include_diagnostics:
            metrics[f"ES_{label}_tailCount"] = tail_sample_size(pnl, p)
            metrics[f"ES_{label}_standardError"] = expected_shortfall_standard_error(pnl, p)
    return metrics


# =============================================================================
# EXECUTION DEMONSTRATION
# =============================================================================
if __name__ == "__main__":
    from engine.simulation.market_model import generate_paths
    from engine.instruments.swap import SwapConfig, price_swaps
    from engine.simulation.demo_scenarios import SWAP_DEMO_MATURITIES, flat_yield_curves, single_currency_swap_demo_config

    market_cubes = generate_paths(single_currency_swap_demo_config())

    # fixed_rate close to the forwarding curve's own rate (3.5%) so the demo
    # swap starts close to fair value -- shows genuine two-sided VaR/ES
    # instead of a always-in-the-money trade that never registers a loss.
    swap_cfg = SwapConfig(
        notional=1_000_000.0,
        fixed_rate=0.035,
        payer=True,
        discount_curve_index=0,
        forward_curve_index=1,
        swap_tenor="2Y",
    )
    npv_cube = price_swaps(market_cubes["yield_curves"], SWAP_DEMO_MATURITIES, [swap_cfg])

    # t=0 baseline: the same swap priced against today's actual (zero-shock)
    # curves -- a real deterministic revaluation, not a cross-scenario proxy.
    base_cube = flat_yield_curves(disc_rate=0.030, fwd_rate=0.035)
    base_npv = float(price_swaps(base_cube, SWAP_DEMO_MATURITIES, [swap_cfg])[0, 0, 0])

    metrics = compute_risk_metrics(npv_cube, base_npv, percentiles=(0.95, 0.99))
    print("Base (t=0) NPV:", base_npv)
    for key, values in metrics.items():
        print(f"{key}: {[round(float(v), 2) for v in values]}")
