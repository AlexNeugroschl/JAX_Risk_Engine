# Market Risk: Short-Horizon VaR and Expected Shortfall

**Package:** [`engine/risk/market/`](../../engine/risk/market/)
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
scenarios (Monte Carlo or historical)      engine/risk/market/scenarios.py
    -> shocked curves at t=0
    -> every trade repriced in every scenario   engine/risk/market/revaluation.py
    -> P&L per scenario  [S]
    -> VaR / ES + tail count + standard error   engine/risk/market/var_es.py
```

## Using it

```python
from engine.risk.market import (
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

Over HTTP the same run is a job: `POST /portfolio/market-risk` with the market, the trades and
the scenarios' inputs (a covariance or a history of the factors, the factors named in the
order of its columns), polled at `GET /jobs/{job_id}`; the P&L comes inline, by hashed
reference or not at all (`pnl_output`). `demos/demo_api.py` runs one
([HTTP API](../reference/http-api.md#post-portfoliomarket-risk)).

## Reporting currency

Every amount of a run is in one currency, `MarketRiskRequest.base_currency` (ORE's
`baseCurrency`; USD by default, as a portfolio run without a simulation), which the result
names as `result.base_currency`. Each trade is revalued in its own currency, and its base
value and P&L are converted at the market's as-of FX spot (`Market.fx_spots`, quoted either way
round, never crossed through a third currency), in float64 after its P&L is read back from its
storage format; a trade in the reporting currency is multiplied by 1, which changes no bit. The
portfolio P&L, and so every VaR and ES, adds amounts in one unit.

```python
market = Market(asof, {"USD": usd, "EUR": eur}, fx_spots={"EURUSD": 1.10})
result = run_market_risk(MarketRiskRequest(trades, market, scenarios))                       # in USD
result = run_market_risk(MarketRiskRequest(trades, market, scenarios, base_currency="EUR"))  # in EUR
```

A trade whose currency has no spot against the reporting currency is refused before any work,
naming the trade and the pair, as is a reporting currency the market has no market for. Only
rate pillars are shocked, so a converted trade's spot is the same in every scenario: FX risk is
not in the VaR/ES, and the result's warnings say so, naming the converted trades. Until
2026-10-11 the run had no reporting currency and added a EUR trade's P&L to a USD trade's as if
both were dollars ([I-100](../planning/known-issues.md#i-100), closed); FX as a shocked factor
waits on FX trades in the pipeline ([F-04](../planning/features.md#f-04)).

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
([`engine/risk/greeks/price_functions.py`](../../engine/risk/greeks/price_functions.py), the same
functions the AD Greeks differentiate), with the engine its `PricingConfig` names (decision
A-8): a European on Bachelier or Jamshidian, a Bermudan/American on the LGM grid engine
calibrated on today's market and held fixed under every scenario. The same function prices
the base market and every scenario. A trade's P&L is its deviation from its base value in
difference form (F-07): the legs' flows carried by matrix products in the trade's compute
format, and a Bermudan's or American's induction run on the base curves and on each scenario's
deviation from them (since 2026-10-13), so a scenario that moves nothing has exactly zero P&L,
and at float64 the P&L is `f(base + shift) − f(base)` to rounding.

Scenarios run in batches, each one program per product, trade shape and arithmetic (the P&L is
jitted with the trade's terms as an argument), so a repeated run compiles nothing. The batch is
`batch_size` (default 256), smaller only where a Bermudan's or American's induction would exceed
its memory budget (`engine.pricing.lgm_grid.INDUCTION_MEMORY_BUDGET`, 1 GiB): it holds a few
copies of its columns on the state grid per scenario, about 0.1 MB at `n_per_std=64`
(`scenario_batch_size`). Until 2026-10-13 the rollback interpolated every grid node at every
quadrature node per scenario, about 80 MB at `n_per_std=64`, and the batch was cut to fit 512 MB.

**Cost.** Swaps, European swaptions and bonds cost microseconds to a millisecond per
scenario. A Bermudan or American runs one backward induction for a whole batch, its rollback and
its cashflows matrix products shared by every scenario: measured on a CPU (2026-10-13, 256
scenarios, warm), 0.035 ms per scenario at 16 points per standard deviation and 0.18 ms at 64,
against 5.6 ms and 75 ms when the induction ran per scenario. For market risk, size the exercise
grid for the P&L precision the VaR needs rather than the finest grid a single price would use;
`demos/demo.py` shows the base-value difference this makes.

## Statistics

`engine.risk.market.var_es.compute_risk_metrics` on the one-date portfolio P&L sample — ORE's
`RiskStatistics` conventions, including QuantLib's rank, `ceil(N(1−q)) − 1`, and the strict
value-based ES tail ([VaR & ES](var_es.md)). Each quantile `q` lies in (0, 1) and is given once;
ORE's own statistics take [0.9, 1), so the engine's domain is broader, on the same rule. For
each quantile `q`:

| Key | Meaning |
|---|---|
| `VaR_<q>` | Loss at the `q` quantile, positive |
| `ES_<q>` | Mean loss beyond it; NaN if that tail is empty |
| `ES_<q>_tailCount` | Observations the ES averaged |
| `ES_<q>_standardError` | Monte Carlo standard error of the ES |

The key is the level's percentage exactly: Basel's 97.5% is `ES_97.5`, and two levels share a
key only if they are equal (0.9500000001 is `95.00000001`).

## Precision

`MarketRiskRequest.precision` is the portfolio run's `engine.precision.Precision`
([Architecture: Adjustable precision](../concepts/architecture.md#adjustable-precision)), its
stages read for this pipeline: the shifts are rounded to `simulation.compute` and stored at
`simulation.storage`; each trade is revalued and its P&L computed at the compute format of
its own pricing stage, `precision.precision_for(trade)` (its `by_trade` entry, else its
product's `by_product` entry, else `pricing`), and its P&L stored at that stage's storage
format, in its own currency; `result.pnl` is the P&L read back at float64 and converted into the
reporting currency, and VaR/ES are float64 reductions of it (decision A-10). The base values
(`base_npv_per_trade`) are each trade's revaluation of the unshocked curves at its compute
format, converted alike: they are the anchor its P&L is measured from, so a
zero shift is exactly zero P&L at every precision. Trade ids must be unique (the overrides
are keyed by them). The integer
`precision=64|32` of before 2026-10-01 is refused, naming the replacement.

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
- **FX risk.** A trade converted into the reporting currency is converted at today's spot in
  every scenario ([Reporting currency](#reporting-currency)).
- **Thin tails.** Fewer than 10 observations beyond a quantile's rank is flagged.
- **The ES standard error's assumption.** `ES_<q>_standardError` is the tail's standard
  deviation over the square root of its count, as if the scenarios were independent draws;
  Sobol scenarios and overlapping historical horizons are not, so it is indicative
  ([I-111](../planning/known-issues.md#i-111)). Every result says so.
- **Unvalidated precision.** At a precision below float64, each figure (each trade's mean P&L,
  the portfolio's with its VaR and ES) without a passing row in the precision evidence table
  ([compliance/precision-evidence.md](../../compliance/precision-evidence.md); decision A-11:
  the P&L attribution test's green zone and a precision error within 1% of the figure's
  statistical error) is named, with what is validated for it and up to how many scenarios.

Not yet covered: equity, FX, credit and volatility risk factors; stressed calibration and
the liquidity-horizon cascade Basel's ES needs (plan phase P3).

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
  labels, warnings, precision and validation, and the reporting currency (a USD and a EUR swap
  whose figures move with the EURUSD spot as the converted P&Ls say, the inverse quote, the
  paired sample converted alike, a missing spot refused before any work).
- [`tests/test_market_risk_ore_parity.py`](../../tests/test_market_risk_ore_parity.py) —
  the ORE parity above.
