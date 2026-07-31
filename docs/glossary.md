# Glossary

Plain-language definitions for terms used throughout these docs. Finance terms and
math/engineering terms are mixed together alphabetically — if you only know one side,
skim for the terms you don't recognize.

**At-par (coupon pricing)** — A convention for pricing a floating-rate payment using a
single forward interest rate covering its whole accrual period, rather than
compounding several shorter fixings within that period. The standard/default approach,
and the one this project uses (matching ORE's own default).

**Brownian bridge** — A technique for reordering random numbers used in a simulation so
that the numbers with the best statistical properties are used for the parts of the
simulated path that matter most (typically, the final and midpoint values). Named after
Brownian motion, the mathematical model of a continuously random path. See
[Market Simulation](03-market-simulation.md#phase-1--quasi-monte-carlo-shock-generation).

**Confidence level / percentile** — In the context of VaR, "how far into the bad
outcomes are we willing to look." 95% VaR looks at the worst 5% of outcomes; 99% VaR
looks at the worst 1% (a stricter, rarer, typically larger number).

**Correlation / covariance** — How much two things tend to move together. If stock
prices and interest rates are correlated, knowing one tells you something about the
likely direction of the other. A **covariance matrix** is the mathematical object that
encodes correlation (and individual variance/volatility) between every pair of things
being simulated at once.

**Discount factor** — A number between 0 and 1 answering "how much is $1, promised at
some future date, worth today?" A discount factor of 0.95 means $1 in the future is
worth 95 cents today (because money today can be invested and grow). Directly related to
interest rates: higher rates mean lower discount factors.

**Discounting curve vs. forwarding curve** — See **multi-curve discounting**.

**dtype** — Short for "data type." In this codebase, almost always refers to numeric
precision: `float64` (64-bit, more precise, slower) or `float32` (32-bit, less precise,
faster). See **precision**.

**Expected Shortfall (ES)** — Also called **Conditional VaR (CVaR)**. The average loss,
given that a loss at least as bad as the Value at Risk cutoff has occurred. Answers "how
bad does it get in the worst case," where VaR alone only answers "how often does it get
bad." See [Risk Statistics](05-risk-statistics.md).

**Fixed leg / floating leg** — The two sides of an interest rate swap. The fixed leg
pays a rate agreed today and locked in; the floating leg pays a rate that resets
periodically based on actual market conditions. See [Instruments](04-instruments.md).

**FX (foreign exchange)** — The market for exchanging one currency for another; an "FX
rate" is the price of one currency in terms of another (e.g. how many US dollars one
euro buys).

**GBM (Geometric Brownian Motion)** — The standard mathematical model for how stock
prices (and similar assets) are assumed to move randomly over time, where it's the
*percentage* change that's random and roughly bell-curve-shaped, not the absolute dollar
change. See [Market Simulation](03-market-simulation.md#phase-2--the-cross-asset-model-engine).

**Hull-White model (HW1F)** — The standard mathematical model this project uses for how
interest rates move randomly over time. "1F" means "one factor" — one source of
randomness per curve, as opposed to more complex multi-factor rate models. Named after
its inventors, John Hull and Alan White. See
[Market Simulation](03-market-simulation.md#phase-2--the-cross-asset-model-engine).

**JAX** — A Python library, made by Google, for fast numerical computing that can run on
CPUs, GPUs, or TPUs, and that automatically compiles Python math code into efficient,
hardware-accelerated instructions. This project is built on top of it.

**jax.jit / JIT compilation** — "Just-In-Time compilation." A JAX feature that takes a
Python function and compiles it into fast, hardware-specific machine code the first time
it's called (rather than interpreting the Python line-by-line every time, which JAX would
otherwise do slowly).

**jax.lax.scan** — A JAX construct for efficiently repeating an operation many times in a
row (e.g. "take one simulation step" repeated for every time step), without writing an
ordinary, slow Python loop. Used throughout the simulation engine.

**Mean reversion** — A property of some random processes (interest rates, in this
project) where the value tends to drift back toward some long-run average over time,
rather than wandering off arbitrarily far. Governed by a "mean reversion speed"
parameter (`a`, or `mean_reversion` in this codebase's config).

**Monte Carlo simulation** — A general technique for answering "what's likely to happen"
by simulating many random possible outcomes and looking at the pattern across all of
them, rather than solving an equation directly. Named after the Monte Carlo casino,
referencing the role of randomness. See [Market Simulation](03-market-simulation.md).

**Multi-curve discounting** — The modern (post-2008) practice of using two *different*
interest rate curves for a single trade: one to figure out what a floating payment will
actually be (the "forwarding curve," tied to a specific lending benchmark), and a
separate one to discount all cashflows back to today (the "discounting curve," usually
tied to an overnight/OIS rate). See [Instruments](04-instruments.md#2-building-the-real-trade-_build_ore_swap-prepare_swap).

**Notional** — The reference amount of money a trade's payments are calculated from,
without that amount itself ever actually changing hands (in an interest rate swap,
neither side hands over the notional — only the resulting interest payments).

**NPV (Net Present Value)** — What a trade or portfolio is worth today, expressed as a
single number, accounting for all its future cashflows discounted back to the present.

**Numéraire** — A reference asset (in this project, a money-market account that accrues
at a simulated interest rate) used as a bookkeeping device in certain pricing approaches.
See [Market Simulation: the numéraire](03-market-simulation.md#phase-2--the-cross-asset-model-engine).

**ORE (Open Source Risk Engine)** — A real, widely-used, open-source risk engine
software package that this project both learns its math from and validates its own
output against. See [Architecture: ORE as a dependency](02-architecture.md#ore-as-a-dependency).

**P&L (Profit and Loss)** — How much money was gained or lost, relative to some starting
point. Central to VaR/ES, which are computed *from* a P&L distribution. See
[Risk Statistics: the P&L baseline](05-risk-statistics.md#the-pl-baseline-what-are-gainslosses-measured-against).

**Precision (numeric)** — How many digits of accuracy a computer keeps when doing math.
64-bit ("double precision," `float64`) keeps more digits and is more accurate but
slower; 32-bit ("single precision," `float32`) keeps fewer digits and is faster but
noisier. This project's long-term research question is about whether lower precision,
run many more times, can match higher precision's risk answers. See
[Architecture: Adjustable precision](02-architecture.md#adjustable-precision).

**QMC (Quasi-Monte Carlo)** — A refinement of ordinary Monte Carlo simulation that uses
specially constructed, evenly-spread sequences of numbers (like a **Sobol sequence**)
instead of ordinary randomness, so that fewer simulated scenarios are needed to get a
stable answer.

**Scenario** — One simulated "alternate future" — one complete, self-consistent
simulated path for every rate/price being modeled, from today out to the simulation's
final time step. This project typically simulates thousands of scenarios at once.

**Sobol sequence** — A specific, well-known type of quasi-random sequence used for QMC.
See **QMC**.

**Swap (interest rate swap)** — A common trade where two parties exchange interest
payments on a shared notional amount — one side pays a fixed rate, the other pays a
floating rate. See [Instruments](04-instruments.md).

**TraderX** — The name (per this project's roadmap in the root [README.md](../README.md))
of an external system this engine is eventually meant to serve as a live API for, rather
than only running as an offline script.

**Value at Risk (VaR)** — The most standard risk number in finance: "what's the cutoff
loss such that we expect to lose *more* than that only X% of the time?" E.g. 95% VaR of
$1M means: in 95% of simulated outcomes, the loss is under $1M; in the worst 5%, it's at
least that much. See [Risk Statistics](05-risk-statistics.md).

**Vectorization** — Doing a mathematical operation on an entire array of numbers at once
(e.g. "add these two lists of a million numbers together") instead of looping over each
number one at a time in Python. Essential for anything to run fast on a GPU, and a hard
requirement throughout this codebase's JIT-compiled code.

**Volatility** — How much a price or rate tends to fluctuate randomly; a higher
volatility means bigger, more frequent swings. Usually written as `σ` (sigma) in
formulas.

**Yield curve** — A full set of interest rates (or, equivalently, discount factors)
across every future maturity date, as observed (or, in this project, simulated) at one
point in time. See [Market Simulation: Phase 3](03-market-simulation.md#phase-3--yield-curve-reconstruction).
