# Overview

*No finance or math background required for this page.*

## What is this project?

Large investment banks run overnight batch jobs that ask one question, over and over, for
every trade in their books: **"If markets move tomorrow, how much money could we gain or
lose?"**

To answer that question they don't guess — they simulate. They generate thousands of
plausible "alternate future" versions of the market (interest rates a little higher here,
a stock price a little lower there), re-price every trade in every one of those alternate
futures, and then look at the *spread* of outcomes. If a trade loses money in the worst
5% of those simulated futures, that tells the bank how much risk it's carrying.

This project is a from-scratch reimplementation of that pipeline, built in
[JAX](https://github.com/google/jax) (a Python library for fast, GPU-accelerated numerical
computing) so it can run on GPUs and be dramatically faster than the traditional CPU-based
tools banks use today — specifically [ORE (Open Source Risk Engine)](https://www.opensourcerisk.org/),
a widely-used open-source risk engine that this project both learns from and validates
itself against.

## The three-step pipeline, in plain language

1. **Simulate the market.** Generate thousands of "alternate future" scenarios for
   interest rates, stock prices, and currency exchange rates, spread out over time (e.g.
   6 months from now, 1 year from now, 2 years from now). See
   [Market Simulation](../concepts/market-simulation.md).

2. **Price every trade in every scenario.** For each simulated future, calculate what
   every trade in the portfolio would be worth. This project currently supports interest
   rate swaps (a common trade where two parties exchange fixed and floating interest
   payments) as well as European, Bermudan, and American swaptions (options on swaps) —
   see [Instruments: Interest Rate Swaps](../instruments/swaps.md),
   [European Swaptions](../instruments/european-swaptions.md), and
   [American & Bermudan Swaptions](../instruments/american-bermudan-swaptions.md).

3. **Measure the risk.** Look at the full spread of "what-if" trade values across every
   scenario and compute standard risk numbers: **Value at Risk (VaR)** ("in the worst 5%
   of outcomes, how much do we lose?") and **Expected Shortfall (ES)** ("on average, how
   bad is that worst 5%?"). See [Risk Statistics: VaR & ES](../risk/statistics.md).

Each step produces a bigger, richer version of the same idea: a big table of numbers,
organized by *simulated scenario* and *point in time*. The engine literally calls this a
"cube" internally — picture a 3D spreadsheet where one axis is "which alternate future,"
one axis is "how far in the future," and one axis is "which trade."

## Why does this exist? (the project's actual goals)

This isn't just "redo ORE in Python" — it's built around three specific, longer-term
ambitions (see the root [README.md](../../README.md) for the full roadmap):

- **Speed via GPUs.** Traditional risk engines like ORE run on CPUs. This engine is
  written in JAX specifically so the heavy numerical work (generating scenarios,
  pricing trades across all of them at once) can run on a GPU, which is dramatically
  faster for this kind of "do the same math millions of times in parallel" workload.
- **A research question about precision.** Computers can do math with different levels
  of numeric precision — think of it like the difference between calculating in exact
  decimals versus rounding to fewer digits at each step. Higher precision is more
  accurate but slower; lower precision is faster but noisier. The project's long-term
  research goal is to test whether running *many more* lower-precision simulations
  reaches the same risk answer, in the same amount of compute time, as running *fewer*
  high-precision ones. That's why every piece of this engine is built to support
  switching precision on and off (see [Adjustable Precision](../concepts/architecture.md#adjustable-precision)).
- **Correctness against a known-good reference.** Rather than inventing new math, this
  project continuously checks its own output against ORE's — a mature, real-world risk
  engine used by actual financial institutions. Every pricing formula and risk formula
  in this codebase has been checked line-by-line against ORE's own installed software,
  and the test suite includes tests that run ORE itself and compare answers directly. This
  is described more in [Architecture](../concepts/architecture.md).
- **Eventually, a live API.** The long-term plan is to expose this engine as a web API
  that another system ("TraderX") can call in real time, not just run as an overnight
  batch script.

## What's actually built right now

| Piece | What it does | Status |
|---|---|---|
| Market simulation | Simulates future interest rates, stock prices, and FX rates | ✅ Working |
| Interest rate swap pricing | Prices interest rate swaps across every simulated scenario | ✅ Working |
| European swaption pricing | Prices single-exercise-date swaptions (Jamshidian's closed-form decomposition) | ✅ Working |
| Bermudan & American swaption pricing | Prices multi/continuous-exercise-date swaptions (numeric LGM backward induction) | ✅ Working |
| Value at Risk / Expected Shortfall | Turns trade values into risk numbers | ✅ Working |
| XVA (valuation adjustments) | Not yet built | 🔜 Planned |
| Live API | Not yet built | 🔜 Planned |

## Where to go next

- **New to finance/math and want the plain-language version of how the math works?**
  Each deep-dive doc ([Market Simulation](../concepts/market-simulation.md),
  [Instruments: Interest Rate Swaps](../instruments/swaps.md),
  [European Swaptions](../instruments/european-swaptions.md),
  [American & Bermudan Swaptions](../instruments/american-bermudan-swaptions.md),
  [Risk Statistics](../risk/statistics.md)) starts
  with a "Plain-language summary" section before getting into formulas.
- **Want to actually run the code?** See the [User Guide](user-guide.md).
- **Want to understand how the pieces fit together as software?** See
  [Architecture](../concepts/architecture.md).
- **Want the exact inputs/outputs of every function?** See the
  [API Reference](../reference/api-reference.md).
- **Don't know what a term means?** See the [Glossary](../concepts/glossary.md).
