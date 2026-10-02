# Market Risk: Short-Horizon VaR and Expected Shortfall

**Package:** [`engine/market_risk/`](../../engine/market_risk/)
**Entry point:** `run_market_risk(MarketRiskRequest(trades, market, scenarios, pricing))`

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
see [audit finding R-1](../planning/known-issues.md#r-1).

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

factors = RateRiskFactors.from_market(market)     # discount:USD, index:USD-SIMINDEX-6M, ...

# Monte Carlo: many draws from a Gaussian with a given 10-day covariance...
scenarios = monte_carlo_scenarios(factors, covariance, horizon_days=10,
                                  num_scenarios=2**14, seed=1)
# ...or historical: every overlapping 10-day move in a history of pillar levels.
scenarios = historical_scenarios(factors, history, horizon_days=10, dates=dates)

result = run_market_risk(MarketRiskRequest(trades, market, scenarios, quantiles=(0.99, 0.975)))
result.risk["VaR_99"], result.risk["ES_97.5"]          # positive losses
result.risk["ES_97.5_standardError"]                    # Monte Carlo noise of the ES
result.pnl                                               # [S, N] per-trade P&L
```

`demos/demo.py` ends with a worked run; `demos/demo_precision.py` runs it at FP64, FP32,
and with the P&L only stored in FP32.

## Risk factors

The pillar zero rates of named market curves (`RateRiskFactors`). `from_market(market)`
takes every curve of the `Market`: per currency its discount curve (`discount:<ccy>`) and its
index curves (`index:<index name>`), the names the trades read. A factor vector is every
curve's pillars end to end, and every input — covariance, history, shock — is expressed on
that vector. `factors.labels()` names each position (`"discount:USD/5y"`).

Shocks are **absolute** zero-rate moves, because a relative move is undefined for a zero
or negative rate (Basel plan decision D-6).

A trade reads its currency's discount curve and, unless it is a bond, its index's
forwarding curve, by name; every curve it reads must be a factor, and every factor must be
the market's curve of its name (otherwise a trade would be shocked from a base it is not
priced on). The run refuses either, naming the trade or the factor.

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
functions the AD Greeks differentiate), with the engine its `PricingConfig` names (decision
A-8): a European on Bachelier or Jamshidian, a Bermudan/American on the LGM grid engine
calibrated on today's market and held fixed under every scenario. The same function prices
the base market and every scenario, so a trade's P&L is exactly `f(base + shift) − f(base)`.

Scenarios run in vmapped batches, which keeps the work on the accelerator; the pricers are
jitted with the trade as an argument, so a repeated run compiles nothing. The batch is `batch_size` (default 256) for closed-form trades, and smaller
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
results. It is part of the open performance item
[I-53](../planning/known-issues.md#i-53).)

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

`MarketRiskRequest.precision` is the portfolio run's `engine.precision.Precision`
([Architecture: Adjustable precision](../concepts/architecture.md#adjustable-precision)), its
stages read for this pipeline: the shifts are rounded to `simulation.compute` and stored at
`simulation.storage`; each trade is revalued and its P&L computed at the compute format of
its own pricing stage, `precision.precision_for(trade)` (its `by_trade` entry, else its
product's `by_product` entry, else `pricing`), and its P&L stored at that stage's storage
format; `result.pnl` is the P&L read back at float64, and VaR/ES are float64 reductions of it
(decision A-10). The base values (`base_npv_per_trade`) are each trade's revaluation of the
unshocked curves at its compute format: they are the anchor its P&L is measured from, so a
zero shift is exactly zero P&L at every precision. Trade ids must be unique (the overrides
are keyed by them). The integer
`precision=64|32` of before roadmap 1.4 is refused, naming the replacement.

`result.precision` is the run's `PrecisionReport`: the policy, each trade's stage, the
formats the shifts and each P&L were stored in (read from the arrays) and the device. With
`precision.paired_fraction > 0` the first scenarios (whole blocks of 32) are revalued again at
float64, and each VaR and ES is measured on them: `figures["portfolio/VaR_99"]` holds the run's
VaR, the VaR of the paired scenarios at the run's precision and at float64, and their
difference. VaR and ES are quantiles, so the reported figures are the run's own, not corrected
(decision A-13); an ES whose tail is empty on the paired scenarios is NaN.

Every price function derives its working dtype from the curve it is given, so a float32
revaluation is float32 end to end (strict dtype promotion in CI) — a property pinned by
`tests/test_market_risk.py::TestRevaluation::test_the_european_price_function_keeps_float32`
(for both European engines), which once caught two constants silently promoting a European to
float64.

## What is not in the number

`run_market_risk` states these in `result.warnings` where they apply:

- **Volatility risk.** Volatilities are held at their base values: the market's swaption
  volatilities, a Bermudan's calibrated LGM, the Jamshidian engine's model. Only curve pillars
  move, so an option's vega is not in the VaR/ES.
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
| European swaption (Bachelier, the default) | `Swaption` + `BachelierSwaptionEngine` on ORE's `SwaptionVolatilityMatrix` (the formula of ORE's default `BlackMultiLegOptionEngine`) | ~2e-14 (measured 1.6e-14) |
| European swaption (`european="Jamshidian"`) | `Swaption` + `JamshidianSwaptionEngine` on `HullWhite` (single-curve, as QuantLib's) | within QuantLib's Brent tolerance on its root, [European Swaptions](../instruments/european-swaptions.md) |
| Bermudan swaption (fixed LGM, `calibration="None"`) | `NumericLgmMultiLegOptionEngine` via in-process `OREApp`, the index on its own curve | ~2e-13 |

VaR and ES agree with ORE's to 1e-6 relative on 512 Monte Carlo scenarios (a swap, a
Bachelier European and a bond). On a
synthetic 300-day history, run through `historical_scenarios`, they agree to 1e-10.

ORE's own `ParametricVarCalculator` (which has a Monte Carlo mode) and `ExposureCalculator`
have no constructor in the Python bindings, so they cannot be called as oracles directly.
Parity is instead established by full revaluation in ORE plus ORE's statistics.

## Tested by

- [`tests/test_market_risk.py`](../../tests/test_market_risk.py) — factors, scenario
  generators (covariance recovered, seeds, singular covariance, historical windows),
  revaluation (zero shift, the base equals the portfolio's value, batching invariance,
  first-order agreement with AD Delta, the configured European engine), the run's statistics,
  labels, warnings, precision and validation.
- [`tests/test_market_risk_ore_parity.py`](../../tests/test_market_risk_ore_parity.py) —
  the ORE parity above.
