# Exposure Profiles: EPE, ENE, EE_B, EEE_B, EPE_B, EEPE_B and PFE

**Module:** [`engine/risk/exposure.py`](../../engine/risk/exposure.py)
**Produced by:** `price_portfolio` → `PortfolioResult.exposure` (the portfolio as one
netting set) and `PortfolioResult.trade_exposures` (each trade standalone)

## What it measures

How much the counterparty could owe you, or you them, at each future date. This is what
the multi-step risk-neutral simulation in `price_portfolio` is for: it prices every trade
on every path at every simulated date, and the exposure profile summarises that cube
through time.

It is **not** market-risk VaR. A loss quantile of a risk-neutral, multi-year simulation
is not a forecast of tomorrow's loss. Short-horizon VaR/ES is
[Market Risk](market-risk.md). The cube's statistics were reported as `VaR_95`/`ES_95`
until the 2026-09-24 audit ([R-1](../planning/known-issues.md#r-1)).

## Definitions

ORE's `ExposureCalculator`
(`OREAnalytics/orea/aggregation/exposurecalculator.cpp`, lines 150–230), one trade or one
netting set, no collateral. `V_k(t)` is the NPV on path `k` at date `t` divided by the
numeraire on that path, which is how ORE's cube stores it (`valuationcalculator.cpp:74`).

| Statistic | Definition | Meaning |
|---|---|---|
| `epe` | `mean_k max(V_k(t), 0)` | Expected positive exposure, discounted to today |
| `ene` | `mean_k max(−V_k(t), 0)` | Expected negative exposure, discounted |
| `ee_b` | `epe(t) / P(0,t)` | Expected exposure, undiscounted (Basel) |
| `eee_b` | running maximum of `ee_b` | Effective expected exposure, non-decreasing |
| `pfe["PFE_q"]` | `max(sorted_k V_k(t)[⌊q(S−1)+0.5⌋], 0)` | Potential future exposure at quantile `q`, discounted |
| `epe_b` | `sum_{j<=i} ee_b(t_j) dt_j / T_i` | ORE's time-weighted EPE_B (`epe_bTimeWeighted_`), weights ActualActual(ISDA) from the as-of date, only dates up to the trade's maturity |
| `eepe_b` | the same over `eee_b` | ORE's time-weighted EEPE_B |
| `basel_epe`, `basel_eepe` | `epe_b`, `eepe_b` at the last date on or before `WeekendsOnly().adjust(asof + 1Y + 4D)` | ORE's Basel EPE_B / EEPE_B; `None` on the Hull-White path (no dates) |

Every profile starts at t=0, where ORE sets EPE = EE_B = EEE_B = PFE = max(NPV₀, 0) and
ENE = max(−NPV₀, 0). `times[0]` is 0; the rest are the simulation's `time_grid[1:]`.

On the market path the numeraire is the base currency's LGM numeraire `N(t, x)`, exact at
each date (ORE's), and `P(0,t)` comes from the base currency's discount curve; the times are
the simulation dates'. On the Hull-White path the numeraire is the simulation's money-market
account on rate factor 0 ([I-45](../planning/known-issues.md#i-45)), `P(0,t)` comes from that factor's
initial curve, and the time weights use `time_grid` (there are no dates).

**Netting.** A netting set's exposure is computed on the *sum* of its trades' paths, so
offsetting trades net before any statistic is taken. It is never the sum of the trades'
standalone exposures, and it is never larger.

## Requesting it

```python
result = price_portfolio(PortfolioRequest(market, trades, pfe_quantiles=(0.95, 0.99)))
result.exposure.epe, result.exposure.pfe["PFE_99"]      # [T+1] each
result.trade_exposures[0].ene
```

`RiskPrecisionOverride.exposure` sets the dtype of the statistics (the cube is cast
first). `scenario_risk=False` (required for a portfolio holding a bond) returns
`exposure=None` and no trade exposures — absent, not zero.

Over HTTP, `pfe_quantiles` is a request field and the response carries `exposure` and
`trade_exposures` objects with the same fields ([HTTP API](../reference/http-api.md)).

## Known limitations of the simulated cube

The statistics above are exact on the cube they are given. On the **Hull-White path** the
cube itself has three known weaknesses ([I-42](../planning/known-issues.md#i-42), [I-04](../planning/known-issues.md#i-04), [I-43](../planning/known-issues.md#i-43)), and
`price_portfolio` warns about each one whenever it applies to a run. The market path has none
of them: its simulation is ORE's cross-asset model, paid flows drop out, and options are
wrapped as ORE wraps them ([I-42](../planning/known-issues.md#i-42), [I-43](../planning/known-issues.md#i-43)).
What it lacks is a comparison of the assembled profiles with an ORE run
([I-50](../planning/known-issues.md#i-50)).


| Finding | Effect on exposure | Warning fires when |
|---|---|---|
| [M-1](../planning/known-issues.md#m-1) | Simulated discount factors are not arbitrage-free against a non-flat curve: 4–9% off at 2y on a 3%→5% curve | a rate factor's curve is not flat, or `initial_rates`/`theta` differ from its level |
| [M-2](../planning/known-issues.md#m-2) | Swap cashflows already paid stay in the NPV; a matured swap keeps a value | a swap's floating leg is aged at any simulated step |
| [M-3](../planning/known-issues.md#m-3) | Options are worth 0 after their last exercise date on every path; exercise into the swap is not tracked | a swaption's last exercise falls inside the simulated horizon |

## Tested by

[`tests/test_exposure.py`](../../tests/test_exposure.py) checks every statistic against a
line-by-line transcription of ORE's C++ loop at several quantiles (ORE's
`ExposureCalculator` cannot be constructed from Python). It also checks the defining
identities: EPE−ENE is the mean deflated NPV, EEE_B never decreases, the PFE rank rule
and floor, and netting. `tests/test_portfolio_entrypoint.py` checks that `price_portfolio`
wires the numeraire and discount curve in correctly, and `tests/test_portfolio.py`
checks the three warnings.
