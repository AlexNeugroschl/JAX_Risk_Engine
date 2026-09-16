# Risk Statistics: Value at Risk & Expected Shortfall

**Module:** [`engine/risk/var_es.py`](../../engine/risk/var_es.py)
**Public entry point:** `compute_risk_metrics(npv_cube, base_npv, percentiles=(0.95, 0.99))`

## Plain-language summary

This is the final step: turning "what a trade (or portfolio of trades) is worth in
thousands of simulated alternate futures" into the actual risk numbers a bank reports and
manages against.

Imagine lining up all 4,096 (or however many) simulated outcomes for a portfolio's value
at some future date, from worst to best. Two standard questions get asked about that
lineup:

- **Value at Risk (VaR):** "What's the cutoff for the worst 5% (or worst 1%) of
  outcomes?" If 95% VaR is \$1 million, that means: in 95% of simulated futures, the
  portfolio loses less than \$1 million; in the worst 5%, it loses at least that much.
- **Expected Shortfall (ES)**, also called Conditional VaR: "Given that we're in that
  worst 5% (or 1%), what's the *average* loss?" This answers a question VaR alone can't:
  VaR tells you *how often* things get bad, but not *how bad* they get when they do. ES
  fills that gap — it's always at least as large as the corresponding VaR.

Both numbers are always reported as **positive numbers representing the size of a
potential loss** — even though the underlying scenario values might be gains or losses
in either direction.

This module is completely **instrument-agnostic**: it has no idea what a "swap" is and
never imports anything from `engine/instruments/`. It only requires
*some* NPV cube shaped `[Scenarios, TimeSteps, Trades]` — which means the exact same code
will work unmodified for any future instrument type this engine adds (options, bonds,
whatever comes next), with zero changes needed here. See
[Architecture: stages agree on shapes, not code](../concepts/architecture.md#design-principle-stages-agree-on-shapes-not-code).

## Why it's built this way: matching ORE's exact formula

"What's the cutoff for the worst 5%" sounds like it should have one obvious answer, but
it doesn't — there are several mathematically reasonable ways to define "the cutoff" when
your data points don't land on a perfectly round percentage boundary (this is called the
**interpolation method**, and different tools genuinely disagree on it — for example,
`numpy`'s default percentile method gives a different answer than the one used here).

Because this project's stated goal is numerical parity with ORE, the formulas below were
not written from a textbook — they were **reverse-engineered by directly testing ORE's
own installed software**, using carefully constructed test data designed to expose
exactly where different conventions disagree, then comparing this module's output against
`ORE.RiskStatistics` line-for-line until they matched exactly.

## The formulas

Given a sample of `N` profit-and-loss (P&L) values (positive = gain, negative = loss),
sorted from worst (most negative) to best:

```
index = floor(N × (1 − percentile))

VaR(percentile)  =  max( −sorted_pnl[index],  0 )
```

This is a **"lower" / nearest-rank-below order statistic** — i.e. it picks an actual
observed value from the data (the `index`-th worst outcome), rather than mathematically
interpolating between two neighboring observed values the way `numpy.percentile`'s
default method does. The `max(..., 0)` clamp means: if there are no losses at all in the
sample, VaR is reported as exactly `0`, not a negative number — you can't have "negative
risk."

```
tail  =  { pnl values that are STRICTLY worse than −VaR(percentile) }

ES(percentile)  =  −mean(tail)
```

**The strict-inequality detail matters.** This formula uses a *value-based* filter (every
entry strictly worse than the VaR cutoff), not a *positional* slice of the sorted array
(the worst `index` entries). The two give the same answer *unless* there are tied values
sitting exactly at the VaR boundary — in which case they diverge, and only the
value-based filter matches ORE. This is verified with a test P&L sample containing
deliberate ties at the VaR cutoff, cross-checked against
`ORE.RiskStatistics.expectedShortfall()` directly. See
`tests/test_var_es.py::TestExpectedShortfallAgainstORE::test_matches_ore_with_ties_at_var_boundary`.

**What happens when the tail is empty?** If every one of the worst observations is
exactly tied at the VaR cutoff, the strict `<` filter can end up with nothing in it.
ORE's own `RiskStatistics.expectedShortfall()` raises an error (`RuntimeError: no data
below the target`) in this situation. This module cannot raise a Python exception from
inside JAX's compiled/traced code the way ORE can, so instead it returns `NaN` (Not a
Number) for that specific percentile/time-step combination — the numerically closest
equivalent of "this value is undefined here." Callers must check for `NaN` explicitly.
This is documented behavior with a direct regression test
(`test_empty_tail_matches_ore_raising`), not an oversight.

## The P&L baseline: what are gains/losses measured against?

VaR/ES need a P&L number, not just a raw portfolio value — "lost money" only means
something relative to *some* starting point. This module computes:

```
P&L(scenario, t)  =  portfolio_NPV(scenario, t)  −  base_npv
```

where `base_npv` is a single number: **the portfolio's actual value today, before any
simulated shocks** — supplied explicitly by the caller (typically by pricing the same
trade(s) against today's real, un-simulated market curve; see
[`engine/simulation/demo_scenarios.py`](../../engine/simulation/demo_scenarios.py)'s
`flat_yield_curves()` helper for how the
demos build this). This is applied identically at *every* simulated future time step,
matching ORE's own historical-VaR P&L definition literally.

This was a deliberate choice between two reasonable options: measuring against a fixed
"today's value" baseline (chosen — matches ORE's actual definition, and means the
resulting risk profile reflects both market risk *and* a trade's ordinary
value-changes-over-time as it approaches maturity) versus measuring each future time
step against *that step's own* average simulated value (which would isolate pure market
risk from ordinary time-decay, but doesn't match ORE's definition). See the root
[README.md](../../README.md)'s Phase 3 notes for where this decision is recorded.

## The functions

```python
def portfolio_pnl(npv_cube: jax.Array, base_npv: float) -> jax.Array:
```
`[Scenarios, TimeSteps, Trades] → [Scenarios, TimeSteps]`. Sums every trade's value
together to get one portfolio-level number per scenario/step (a real portfolio's risk is
measured on the *combined* position, not trade-by-trade), then subtracts `base_npv`.

```python
def value_at_risk(pnl: jax.Array, percentile: float) -> jax.Array:
def expected_shortfall(pnl: jax.Array, percentile: float) -> jax.Array:
```
`[Scenarios, TimeSteps] → [TimeSteps]`. Implement the formulas above, vectorized across
every time step at once via `jnp.sort` — sorting the whole scenario axis is the
"expensive" part of this computation, and it's exactly the kind of operation GPUs/TPUs are
efficient at.

```python
def compute_risk_metrics(npv_cube, base_npv, percentiles=(0.95, 0.99),
                         include_diagnostics=True) -> Dict[str, jax.Array]:
```
The public entry point. Computes both VaR and ES at every requested confidence level (by
default, both 95% and 99% — matching ORE's convention of always reporting VaR and ES
together, at multiple confidence levels, rather than a single number in isolation), and
returns them in a dictionary keyed like `"VaR_95"`, `"ES_95"`, `"VaR_99"`, `"ES_99"`.

## Convergence diagnostics — how much to trust the tail

`ES_99 = 1,240,000` reads as a precise figure. Computed from 3 tail observations it is not
one, and a bare float cannot tell you which you have. So each percentile also gets:

```python
def tail_sample_size(pnl, percentile)                 -> jax.Array  # [TimeSteps]
def expected_shortfall_standard_error(pnl, percentile) -> jax.Array  # [TimeSteps]
```

surfaced through `compute_risk_metrics` as `"ES_99_tailCount"` and
`"ES_99_standardError"`.

**`tailCount` is the effective sample size** — the number of observations that actually
entered the ES mean. It is usually far smaller than the scenario count: **at 99% over 10,000
scenarios roughly 100 observations carry the estimate**, and every one of them sits in the
part of the distribution the Monte Carlo sampled least. It counts the same *strict
value-based* tail (`pnl < -VaR`) that `expected_shortfall` averages — not a positional
`floor(N·(1−p))` slice, which disagrees whenever there are ties at the VaR boundary (the
same distinction described above for ES itself).

**`standardError`** is the standard error of that tail mean, `s/√n`, with the *sample*
standard deviation (`ddof=1`) — the tail is a sample, and the population form would
understate the spread. It is **NaN, never `0.0`, when `n < 2`**: with one observation the ES
is defined but its spread is not, and `0.0` would read as "perfectly converged" for exactly
the case where the estimate is least trustworthy.

This does not make any estimate better. It makes the uncertainty visible — the difference
between a number a reader can weigh and one they must simply trust. Part of
[I-11](../known-issues.md#i-11); added by
[W0.6](../reference/eod-integration.md#w06--market-input-selection--closes-part-of-i-11),
purely additively, so every pre-existing key and value is unchanged.

**Precision.** `standardError` is a statistic and follows a
`RiskPrecisionOverride(var_es=...)` like `VaR`/`ES` do — the override is applied by casting
the P&L cube, so output dtype is how a caller observes it. `tailCount` is a *count* and stays
integral at every precision: float32 cannot represent integers exactly above 2²⁴, so following
the override would let a large-scenario count silently round.

## The measure a risk number is unactionable without

`RISK_MEASURES` names the three things a VaR figure can be, and `ENGINE_RISK_MEASURE`
records which one this engine produces:

| Measure | What it is |
|---|---|
| `risk-neutral-pricing` | Exposure under the pricing measure. **What this engine computes.** Correct for CVA/exposure/limits. |
| `historical-forecast` | A calibrated real-world forecast of realised loss. **Not produced here.** |
| `deterministic-stress` | A prescribed scenario's revaluation. No probability attaches to it. |

A risk-neutral exposure simulation is *not* a forecast of tomorrow's P&L — its drift is the
risk-neutral one. Reporting it where a capital or backtesting process expects a historical
forecast is a category error that no amount of numerical accuracy fixes, which is why the
label travels with the number rather than living only in documentation.

## Tested by

- `tests/test_var_es.py::TestValueAtRiskAgainstORE` /
  `TestExpectedShortfallAgainstORE` — direct numeric comparison against
  `ORE.RiskStatistics`, including the tie-at-boundary and empty-tail edge cases described
  above.
- `TestRiskMetricsProperties` — sanity properties that should always hold regardless of
  the exact formula details (ES is never smaller than VaR; VaR at 99% is never smaller
  than VaR at 95%; single-scenario/single-trade edge case doesn't crash).
- `TestPortfolioPnlSumsTrades` — confirms the trade-summing behavior directly.
- `TestRobustAcrossInstrumentSources` — the instrument-agnostic property described above:
  runs the same code against both a synthetic, non-swap-derived cube and a real
  swap-pricer cube.

`tests/test_var_es_diagnostics.py` covers the convergence diagnostics and the measure
vocabulary:

- `TestTailSampleSize` — counts the strict value-based tail, including the tie-at-boundary
  case where a positional implementation would disagree.
- `TestExpectedShortfallStandardError` — cross-checked against numpy's `ddof=1` computation
  on the same tail; `test_single_observation_is_nan_not_zero` pins the `n < 2` behavior.
- **`TestAdditiveOnly`** — every pre-existing key and value is unchanged, which is what lets
  `engine/api/schemas.py`, `engine/portfolio/request.py` and the other consumers keep working
  untouched.
- `TestDiagnosticsReachTheHttpBoundary` — the new keys serialize through `RiskMetricsSchema`,
  with a NaN standard error arriving as `null` rather than a readable float.
- `TestRiskMeasureVocabulary` — the three measures, and that this engine reports
  `risk-neutral-pricing`.
