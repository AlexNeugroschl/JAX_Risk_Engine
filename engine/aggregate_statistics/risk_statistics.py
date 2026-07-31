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
from typing import Dict, Sequence

import jax
import jax.numpy as jnp


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
    """
    num_scenarios = pnl.shape[0]
    idx = int(jnp.floor(num_scenarios * (1.0 - percentile)))
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
    """
    var = value_at_risk(pnl, percentile)
    tail_mask = pnl < -var[None, :]
    masked = jnp.where(tail_mask, pnl, jnp.nan)
    return -jnp.nanmean(masked, axis=0)


def compute_risk_metrics(
    npv_cube: jax.Array,
    base_npv: float,
    percentiles: Sequence[float] = (0.95, 0.99),
) -> Dict[str, jax.Array]:
    """
    Public entry point: [Scenarios, TimeSteps, Trades] NPV cube + t=0
    portfolio NPV -> dict of [TimeSteps] VaR/ES arrays, one pair per
    requested percentile, keyed "VaR_95"/"ES_95"/"VaR_99"/"ES_99"/... --
    matching ORE's convention of always reporting VaR and ES together at
    each configured quantile (see module docstring / plan for the recovered
    ore_histsimvar.xml evidence).
    """
    pnl = portfolio_pnl(npv_cube, base_npv)
    metrics: Dict[str, jax.Array] = {}
    for p in percentiles:
        label = f"{int(round(p * 100))}"
        metrics[f"VaR_{label}"] = value_at_risk(pnl, p)
        metrics[f"ES_{label}"] = expected_shortfall(pnl, p)
    return metrics


# =============================================================================
# EXECUTION DEMONSTRATION
# =============================================================================
if __name__ == "__main__":
    from engine.market_simulations import generate_paths
    from engine.instruments.interest_rate_swap import SwapConfig, price_swaps
    from engine.scenarios import SWAP_DEMO_MATURITIES, flat_yield_curves, single_currency_swap_demo_config

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
