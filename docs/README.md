# Documentation

Documentation for the JAX Risk Engine — a JAX-based market simulation and trade pricing
engine built to run across multiple TPUs (deployed on a Google Cloud TPU VM) and study
precision tradeoffs on that hardware, designed to mathematically mirror
[ORE (Open Source Risk Engine)](https://www.opensourcerisk.org/) for correctness. It also
runs correctly on CPU/GPU — the architecture is backend-agnostic by construction — but is
optimized for TPU. See the root [README.md](../README.md) for a quick overview and setup.

## Where to start

| If you want to... | Read... |
|---|---|
| Understand what this project does, no finance/math background needed | [Overview](getting-started/overview.md) |
| Actually run the code | [User Guide](getting-started/user-guide.md) |
| Understand how the code is organized as software | [Architecture](concepts/architecture.md) |
| Look up exact function signatures and data shapes | [API Reference](reference/api-reference.md) |
| Price a whole portfolio in one call, from Python | [The Portfolio Entry Point](reference/portfolio-entrypoint.md) |
| Price a whole portfolio over HTTP | [HTTP API](reference/http-api.md) |
| Consume a TraderX end-of-day bundle (and know what gets refused) | [EOD Integration Boundary](reference/eod-integration.md) |
| Actually submit an EOD bundle over HTTP, end to end | [User Guide: Pricing a TraderX EOD bundle](getting-started/user-guide.md#pricing-a-traderx-eod-bundle) |
| Profile a pricing job's JAX vs. Python time | [User Guide](getting-started/user-guide.md#profiling-a-pricing-job) |
| Understand the tracer, and why Greeks used to dominate a trace | [Profiling & the Tracer](concepts/profiling.md) |
| Understand a term you don't recognize | [Glossary](concepts/glossary.md) |
| **Know what's broken, approximated, or missing before trusting a number** | **[Known Issues](planning/known-issues.md)** |

## Concepts

- **[Architecture](concepts/architecture.md)** — how the codebase is organized: the
  repository layout, how the pieces connect, typed configuration, and testing philosophy.
- **[Market Simulation](concepts/market-simulation.md)** — the math behind simulating
  interest rates, equities, and FX rates: ORE's cross-asset model (Sobol QMC, Brownian
  bridge, the LGM or the Hull-White model per currency, Black-Scholes FX and equity) and the
  scenario market of model-implied curves.
- **[Coding Style & Technical Constraints](concepts/coding-style.md)** — the rules that
  apply throughout the codebase (JAX purity/vectorization constraints, how ORE's C++ gets
  translated into JAX).
- **[Profiling & the Tracer](concepts/profiling.md)** — how the XProf hook works, what a
  trace contains, which programs still recompile on every warm repeat (the calibrations and
  the AD Greeks' closures, [I-21](planning/known-issues.md#i-21),
  [I-22](planning/known-issues.md#i-22)), and how to read a trace's phase annotations.
- **[Glossary](concepts/glossary.md)** — plain-language definitions for every finance and
  engineering term used in these docs.

## Instruments

Every trade is valued today and on every simulated path by the engine ORE uses for it
(`engine/valuation/`), producing a common `[Scenarios, Dates, Trades]` NPV cube:

- **[Interest Rate Swaps](instruments/swaps.md)** — discounted cashflows, as ORE's
  `DiscountingSwapEngine`, with path fixings.
- **[European Swaptions](instruments/european-swaptions.md)** — Bachelier on the market's
  swaption volatility (ORE's default), or Jamshidian's decomposition on a configured
  Hull-White model.
- **[American & Bermudan Swaptions](instruments/american-bermudan-swaptions.md)** —
  multi/continuous-exercise-date options via a numeric LGM backward-induction engine
  (Hagan's quadrature convolution), calibrated to the trade's own basket, matching ORE's
  production engine.

**Treasury bills and notes** (`engine/instruments/treasury.py`) are discounted cashflows on
their currency's curve, today and on every path, as ORE's `DiscountingRiskyBondEngine`
without credit. Their TraderX-bundle counterparts live at the integration boundary
([EOD Integration](reference/eod-integration.md)).

## Risk

- **[Market Risk](risk/market-risk.md)** — short-horizon VaR and Expected Shortfall by
  revaluing the portfolio at t=0 under Monte Carlo or historical shocks of every curve
  pillar; validated scenario by scenario against ORE.
- **[Exposure](risk/exposure.md)** — EPE, ENE, EE_B and PFE through time from the
  multi-step risk-neutral simulation, using ORE's `ExposureCalculator` definitions.
- **[VaR & Expected Shortfall statistics](risk/var_es.md)** — the order-statistic and
  tail-mean conventions behind every VaR/ES number, matching `ORE.RiskStatistics` exactly.
- **[Delta, Gamma, Vega, and Theta](risk/greeks.md)** — ORE's sensitivity analysis
  (bump and revalue per curve tenor and swaption quote, Theta on the rolled market), the
  default, or the same Greeks by JAX automatic differentiation (`GreeksConfig.method="AD"`),
  for every trade type and either model.

## Reference

- **[API Reference](reference/api-reference.md)** — exact inputs/outputs for every public
  function and config dataclass.
- **[The Portfolio Entry Point](reference/portfolio-entrypoint.md)** — `engine/portfolio/`'s
  `PortfolioRequest`/`PortfolioResult`/`price_portfolio` and the run configuration
  (`RunConfig`), the single call that ties every module together.
- **[HTTP API](reference/http-api.md)** — the FastAPI wrapper (`engine/api/`) over
  `price_portfolio`: endpoint-by-endpoint reference, the async job pattern and why, request/
  response schemas. The same app also serves the EOD contract under `/eod` — a second,
  deliberately different contract governed by a published JSON Schema rather than Pydantic.
- **[EOD Integration Boundary](reference/eod-integration.md)** — `engine/integration/`'s
  hash-verified TraderX bundle ingestion, terms join, unit normalization, convention
  allowlist and per-calculation coverage model, plus both Treasury pricers, (W1.6) the
  versioned contract interface served over HTTP under `/eod`, and (W0.8) the crash-safe
  durable result store behind it. Read it for the zero-coupon accrued rule, the CRLF hash
  trap, why a USD-SOFR booking is refused rather than routed through the generic swap
  builder, why an equity is refused rather than valued at its own exported mark, and why
  a published manifest — not a pointer file — is the commit point for a result.
- **[Models & Trades](reference/models-and-trades.md)** — the shared foundation layer
  (`engine/models/`): the LGM's analytics in both parametrizations (Hagan's, and the
  Hull-White model's), curves, and ORE trade building.
- **[Calibration](reference/calibration.md)** — `engine/calibration/`'s bootstrap fit of a
  piecewise LGM (or Hull-White) volatility term structure to market swaption quotes,
  matching `ore::data::LgmBuilder::calibrate()`'s own bootstrap convention.
- **[ORE Parity](reference/ore-parity.md)** — maps every algorithm in this codebase to its
  exact counterpart in ORE's own C++ source (`reference/ORE`), file and function name.

## Known issues and planning

- **[Planning](planning/README.md)** — how the plan is organized and the rules for adding
  to it. Three documents carry it:
  - **[Known Issues](planning/known-issues.md)** — every open defect and important
    shortcoming, what it does to a number a user would see, and what closing it takes. Read
    it before trusting an exposure profile or an EOD result. Fixed issues keep one line in
    its closed ledger.
  - **[Features](planning/features.md)** — additive work beyond today's scope (Basel III
    figures, sub-FP32 precision, FX/equity trades, CVA, engine options).
  - **[Roadmap](planning/roadmap.md)** — the order of work: structure, then correctness,
    performance, API, tests, features.
  - **[details/](planning/details/)** — design for the large items: the
    [configurable engine](planning/details/configurable-engine.md),
    [ORE parity validation](planning/details/ore-parity-validation.md),
    [Basel III](planning/details/basel-iii.md),
    [precision](planning/details/precision.md) (down to FP8, and the engine worker) and the
    [TraderX integration](planning/details/traderx-integration.md) (the agreed EOD contract
    and what is open with TraderX).
- **[Decisions](../compliance/decisions.md)** — the dated modelling decisions behind the
  numbers: the ORE alignment's targets, the differences from ORE that remain, and the owner's
  decisions on how the engine is configured.

## Document conventions

- Every deep-dive doc opens with a **"Plain-language summary"** section that assumes no
  finance or math background, before moving into formulas and code.
- Every deep-dive doc ends with a **"Tested by"** section pointing to the exact test file
  and test classes that verify what's described.
- Code is referenced by path and, where helpful, by function/class name — e.g.
  `engine/simulation/config.py::simulate`.
- Where a claim about ORE's own behavior is made (a formula, a convention, a design
  decision), it's backed by either a citation of what was read in ORE's own source, or a
  description of how it was live-tested against the installed ORE package — not assumed
  from general finance knowledge. See
  [Architecture: ORE as a dependency](concepts/architecture.md#ore-as-a-dependency).
