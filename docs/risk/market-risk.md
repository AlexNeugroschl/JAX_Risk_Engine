# Market Risk: Short-Horizon VaR and Expected Shortfall

**Package:** [`engine/market_risk/`](../../engine/market_risk/)
**Entry point:** `run_market_risk(MarketRiskRequest(trades, scenarios))`

## What it measures

The loss the portfolio could suffer over a short horizon (typically 1 or 10 business
days) if today's curves moved. Every scenario is a move of every curve pillar; the whole
portfolio is **revalued at t=0** under each moved market, and VaR and Expected Shortfall
are read off the resulting P&L distribution.

This is the engine's market-risk number: the one an end-of-day risk report, a limit, a
backtest or Basel's internal-models approach means by VaR/ES. It is **not** what
`price_portfolio` produces. That path simulates the portfolio forward for months or years
under the risk-neutral measure, which gives an exposure profile
([Exposure](exposure.md)), not a VaR. The two were once reported under the same name;
see [audit finding R-1](../planning/engine-audit.md#r-1).

```
scenarios (Monte Carlo or historical)      engine/market_risk/scenarios.py
    -> shocked curves at t=0
    -> every trade repriced in every scenario   engine/market_risk/revaluation.py
    -> P&L per scenario  [S]
    -> VaR / ES + tail count + standard error   engine/risk/var_es.py
```

## Using it

```python
from engine.market_risk import (
    MarketRiskRequest, RateRiskFactors, covariance_from_history,
    historical_scenarios, monte_carlo_scenarios, run_market_risk,
)

factors = RateRiskFactors.from_curves([ois_curve, ibor_curve], names=["OIS", "IBOR"])

# Monte Carlo: many draws from a Gaussian with a given 10-day covariance...
scenarios = monte_carlo_scenarios(factors, covariance, horizon_days=10,
                                  num_scenarios=2**14, seed=1)
# ...or historical: every overlapping 10-day move in a history of pillar levels.
scenarios = historical_scenarios(factors, history, horizon_days=10, dates=dates)

result = run_market_risk(MarketRiskRequest(trades, scenarios, quantiles=(0.99, 0.975)))
result.risk["VaR_99"], result.risk["ES_97.5"]          # positive losses
result.risk["ES_97.5_standardError"]                    # Monte Carlo noise of the ES
result.pnl                                               # [S, N] per-trade P&L
```

`demos/demo.py` ends with a worked run; `demos/demo_precision.py` runs it at FP64 and
FP32.

## Risk factors

The pillar zero rates of a list of named curves (`RateRiskFactors`). A factor vector is
every curve's pillars end to end, curve 0 first, and every input — covariance, history,
shock — is expressed on that vector. `factors.labels()` names each position
(`"OIS/5y"`).

Shocks are **absolute** zero-rate moves, because a relative move is undefined for a zero
or negative rate (Basel plan decision D-6).

Trades refer to curves by index into that list:

| Trade | Curves |
|---|---|
| `SwapConfig` | `discount_curve_index`, `forward_curve_index` |
| `SwaptionConfig`, `BermudanSwaptionConfig`, `AmericanSwaptionConfig` | `rate_factor_index` |
| `BondConfig` | `curve_index` (must be set for a market-risk run) |

A trade that carries its own `initial_zero_curve` must carry exactly the risk-factor curve
it points at; otherwise it would be shocked from a base it is not priced on, and the run
refuses it.

## Scenarios

| Function | Source | Notes |
|---|---|---|
| `monte_carlo_scenarios(factors, covariance, horizon_days, num_scenarios, seed)` | Zero-mean Gaussian with the given horizon covariance | Scrambled Sobol normals; different seeds are independent randomized-QMC replicates. The covariance may be singular: it is factorized by eigendecomposition, not Cholesky. |
| `historical_scenarios(factors, history, horizon_days, dates=None)` | Overlapping `horizon_days` moves of a `[dates, factors]` history of levels | Each scenario records its window when `dates` are given. |
| `covariance_from_history(history, horizon_days)` | Sample covariance of those moves | Draws Monte Carlo scenarios from the same distribution a historical run samples. |

Both sources are real-world forecasts of the horizon move and carry the measure label
`historical-forecast` — not `risk-neutral-pricing`, the exposure simulation's measure.

## Revaluation

Every trade is a pure JAX function of its curves' pillar rates
([`engine/risk/price_functions.py`](../../engine/risk/price_functions.py), the same
functions the Greeks differentiate). The same function prices the base market and every
scenario, so a trade's P&L is exactly `f(base + shift) − f(base)`.

Scenarios run through `jax.lax.map` in vmapped batches, which keeps the work on the
accelerator. The batch is `batch_size` (default 256) for closed-form trades, and smaller
for a Bermudan or American: its rollback interpolates every grid node at every quadrature
node for the option, the underlying and each cashflow column, about 80 MB per scenario at
`n_per_std=64`. `scenario_batch_size` caps each batch at `BATCH_MEMORY_BUDGET` (512 MB).

Without the cap, a fixed batch of 256 on a 64-per-std American needed ~19 GB and swapped.

**Cost.** Swaps, European swaptions and bonds cost microseconds to a millisecond per
scenario. A Bermudan or American reruns its whole backward induction per scenario: on a
CPU, a 64-per-std American costs about 0.1–0.2s per scenario, and a 16-per-std one about
a hundredth of that. For market risk, size the exercise grid for the P&L precision the
VaR needs rather than the finest grid a single price would use; `demos/demo.py` shows the
base-value difference this makes. (One measurement is unexplained: the first grid
revaluation in a process sometimes runs up to 50× faster than later ones, with identical
results. It is logged as an open performance item in the
[engine audit](../planning/engine-audit.md#p-2).)

## Statistics

`engine.risk.var_es.compute_risk_metrics` on the one-date portfolio P&L sample — ORE's
`RiskStatistics` conventions, including the nearest-rank quantile and the strict
value-based ES tail ([VaR & ES](var_es.md)). For each quantile `q`:

| Key | Meaning |
|---|---|
| `VaR_<q>` | Loss at the `q` quantile, positive |
| `ES_<q>` | Mean loss beyond it; NaN if that tail is empty |
| `ES_<q>_tailCount` | Observations the ES averaged |
| `ES_<q>_standardError` | Monte Carlo standard error of the ES |

Fractional percentages keep their decimals: Basel's 97.5% is `ES_97.5`.

## Precision

`MarketRiskRequest.precision` (64 or 32) sets the dtype of the revaluation and the
statistics. Every price function derives its working dtype from the curve it is given, so
a float32 run is float32 end to end — a property pinned by
`tests/test_market_risk.py::TestRevaluation::test_swaption_price_function_keeps_float32`,
which caught two constants that were silently promoting the European swaption to float64.

## What is not in the number

`run_market_risk` states these in `result.warnings` where they apply:

- **Volatility risk.** Model parameters (`hw_a`, `hw_sigma`) are held at their base
  values; only curve pillars move. An option's vega is therefore not in the VaR/ES.
- **Thin tails.** Fewer than 10 observations beyond a quantile is flagged.

Not yet covered: equity, FX, credit and volatility risk factors; stressed calibration and
the liquidity-horizon cascade Basel's ES needs (plan phase P3); an HTTP endpoint.

## Validation against ORE

[`tests/test_market_risk_ore_parity.py`](../../tests/test_market_risk_ore_parity.py)
reprices every shocked scenario independently in ORE and runs ORE's own `RiskStatistics`
over ORE's P&L vector:

| Instrument | ORE side | Measured agreement per scenario |
|---|---|---|
| Swap | `MakeVanillaSwap` + `DiscountingSwapEngine`, separate forwarding curve | ~1e-14 relative |
| Bond | `FixedRateBond` (ACT/ACT ISMA) + `DiscountingBondEngine` | ~1e-16 |
| European swaption | `Swaption` + `JamshidianSwaptionEngine` on `HullWhite` | ~3e-7, inside the Jamshidian parity envelope of [European Swaptions](../instruments/european-swaptions.md) |
| Bermudan swaption | `NumericLgmMultiLegOptionEngine` via in-process `OREApp` | ~2e-13 |

VaR and ES agree with ORE's to 1e-6 relative on 512 Monte Carlo scenarios. On a
synthetic 300-day history, run through `historical_scenarios`, they agree to 1e-10.

ORE's own `ParametricVarCalculator` (which has a Monte Carlo mode) and `ExposureCalculator`
have no constructor in the Python bindings, so they cannot be called as oracles directly.
Parity is instead established by full revaluation in ORE plus ORE's statistics.

## Tested by

- [`tests/test_market_risk.py`](../../tests/test_market_risk.py) — factors, scenario
  generators (covariance recovered, seeds, singular covariance, historical windows),
  revaluation (zero shift, batching invariance, first-order agreement with AD Delta, bond
  float/JAX identity), the run's statistics, labels, warnings, precision and validation.
- [`tests/test_market_risk_ore_parity.py`](../../tests/test_market_risk_ore_parity.py) —
  the ORE parity above.
