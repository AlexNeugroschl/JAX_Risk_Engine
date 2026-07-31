# Documentation

Documentation for the JAX Risk Engine — a GPU-accelerated market simulation and trade
pricing engine built in JAX, designed to mathematically mirror
[ORE (Open Source Risk Engine)](https://www.opensourcerisk.org/). See the root
[README.md](../README.md) for the project's overall goals and development roadmap.

## Where to start

| If you want to... | Read... |
|---|---|
| Understand what this project does, no finance/math background needed | [Overview](01-overview.md) |
| Understand how the code is organized as software | [Architecture](02-architecture.md) |
| Actually run the code | [User Guide](06-user-guide.md) |
| Look up exact function signatures and data shapes | [API Reference](07-api-reference.md) |
| Understand a term you don't recognize | [Glossary](glossary.md) |

## Deep dives, one per pipeline stage

The engine is a three-stage pipeline; each stage has its own doc covering both the math
and the code, starting with a plain-language summary before getting technical:

1. **[Market Simulation](03-market-simulation.md)** — simulates thousands of alternate
   futures for interest rates, stock prices, and FX rates.
2. **[Instruments: Interest Rate Swaps](04-instruments.md)** — prices a specific type of
   trade against every simulated future.
3. **[Risk Statistics: VaR & ES](05-risk-statistics.md)** — turns simulated trade values
   into standard risk numbers.

## Document conventions

- Every deep-dive doc opens with a **"Plain-language summary"** section that assumes no
  finance or math background, before moving into formulas and code.
- Every deep-dive doc ends with a **"Tested by"** section pointing to the exact test file
  and test classes that verify what's described.
- Code is referenced by path and, where helpful, by function/class name — e.g.
  `engine/market_simulations.py::generate_paths`.
- Where a claim about ORE's own behavior is made (a formula, a convention, a design
  decision), it's backed by either a citation of what was read in ORE's own source, or a
  description of how it was live-tested against the installed ORE package — not assumed
  from general finance knowledge. See [Architecture: ORE as a dependency](02-architecture.md#ore-as-a-dependency).
