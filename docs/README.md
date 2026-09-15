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
| Profile a pricing job's JAX vs. Python time | [User Guide](getting-started/user-guide.md#profiling-a-pricing-job) |
| Understand a term you don't recognize | [Glossary](concepts/glossary.md) |
| **Know what's broken, approximated, or missing before trusting a number** | **[Known Issues](known-issues.md)** |

## Concepts

- **[Architecture](concepts/architecture.md)** — how the codebase is organized: the
  repository layout, how the pieces connect, typed configuration, and testing philosophy.
- **[Market Simulation](concepts/market-simulation.md)** — the math behind simulating
  interest rates, equities, and FX rates (Sobol QMC, Brownian bridge, Hull-White 1-Factor,
  Geometric Brownian Motion, yield curve reconstruction).
- **[Coding Style & Technical Constraints](concepts/coding-style.md)** — the rules that
  apply throughout the codebase (JAX purity/vectorization constraints, how ORE's C++ gets
  translated into JAX).
- **[Glossary](concepts/glossary.md)** — plain-language definitions for every finance and
  engineering term used in these docs.

## Instruments

Each of these prices a specific trade type against the simulated market data, producing a
common `[Scenarios, TimeSteps, Trades]` NPV cube:

- **[Interest Rate Swaps](instruments/swaps.md)** — linear (no optionality) pricing via
  discounted cashflows.
- **[European Swaptions](instruments/european-swaptions.md)** — single-exercise-date
  options via Jamshidian's closed-form decomposition.
- **[American & Bermudan Swaptions](instruments/american-bermudan-swaptions.md)** —
  multi/continuous-exercise-date options via a numeric LGM backward-induction engine
  (Hagan's quadrature convolution), matching ORE's actual production engine.

## Risk

- **[VaR & Expected Shortfall](risk/var_es.md)** — turns any instrument's NPV cube
  into standard risk numbers, matching `ORE.RiskStatistics` exactly.
- **[Delta, Gamma, Vega, and Theta](risk/greeks.md)** — per-curve-pillar sensitivities for
  every instrument in this codebase (including Bermudan/American Vega, via
  `engine/calibration/`), via JAX automatic differentiation scaled to ORE's own
  bump-and-revalue convention.

## Reference

- **[API Reference](reference/api-reference.md)** — exact inputs/outputs for every public
  function and config dataclass.
- **[The Portfolio Entry Point](reference/portfolio-entrypoint.md)** — `engine/portfolio/`'s
  `PortfolioRequest`/`PortfolioResult`/`price_portfolio`, the single call that ties every
  module together, plus the validation/assembly layer it's built on
  (`docs/planning/traderx-integration.md`).
- **[HTTP API](reference/http-api.md)** — the FastAPI wrapper (`engine/api/`) over
  `price_portfolio`: endpoint-by-endpoint reference, the async job pattern and why, request/
  response schemas.
- **[Models & Trades](reference/models-and-trades.md)** — the shared foundation layer
  (`engine/models/`) every instrument pricer is built on: Hull-White and
  LGM closed-form math, and shared ORE trade-building/cashflow extraction.
- **[Calibration](reference/calibration.md)** — `engine/calibration/`'s bootstrap fit of a
  piecewise LGM volatility term structure to market swaption quotes, matching
  `ore::data::LgmBuilder::calibrate()`'s own bootstrap convention.
- **[ORE Parity](reference/ore-parity.md)** — maps every algorithm in this codebase to its
  exact counterpart in ORE's own C++ source (`reference/ORE`), file and function name.

## Known issues

- **[Known Issues and Limitations Register](known-issues.md)** — every known defect and
  scope gap, what each does to a number a user would see, and what closing it actually
  requires. Read this before trusting an exposure, VaR/ES, or multi-day number: the two
  highest-severity entries (aged-swap pricing and USD-SOFR conventions) are **not fixed**,
  and the register is explicit about the difference between *fixed* and *warned about*.

## Planning

- **[Roadmap](planning/roadmap-and-history.md)** — the phased build-out plan and what's
  done vs. planned.
- **[TraderX Integration Plan](planning/traderx-integration.md)** — the original gap
  analysis for safely accepting arbitrary portfolios from an external trading system, and
  what actually landed (now implemented — see
  [The Portfolio Entry Point](reference/portfolio-entrypoint.md)).
- **[EOD Contract Proposal](planning/eod-contract-proposal.md)** — the proposed request/
  result contracts for the TraderX end-of-day batch integration, and the capability matrix
  of what this engine can and cannot price today.

## Document conventions

- Every deep-dive doc opens with a **"Plain-language summary"** section that assumes no
  finance or math background, before moving into formulas and code.
- Every deep-dive doc ends with a **"Tested by"** section pointing to the exact test file
  and test classes that verify what's described.
- Code is referenced by path and, where helpful, by function/class name — e.g.
  `engine/simulation/market_model.py::generate_paths`.
- Where a claim about ORE's own behavior is made (a formula, a convention, a design
  decision), it's backed by either a citation of what was read in ORE's own source, or a
  description of how it was live-tested against the installed ORE package — not assumed
  from general finance knowledge. See
  [Architecture: ORE as a dependency](concepts/architecture.md#ore-as-a-dependency).
