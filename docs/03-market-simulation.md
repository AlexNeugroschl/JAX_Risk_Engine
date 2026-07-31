# Market Simulation

**Module:** [`engine/market_simulations.py`](../engine/market_simulations.py)
**Public entry point:** `generate_paths(config: SimulationConfig, precision: int = 64)`

## Plain-language summary

This module answers the question: *"Generate thousands of plausible alternate futures for
interest rates, stock prices, and currency exchange rates, at several points in time."*

Think of it like a weather simulator, but for markets. You tell it: here's where interest
rates and prices stand today, here's roughly how volatile each of them tends to be, and
here's how they tend to move together (e.g. stock prices and interest rates aren't
independent — they're correlated). The simulator then generates thousands of independent
"alternate timelines," each one a full path from today out to some future date, step by
step.

It does this for two kinds of things:
- **Interest rates**, for one or more currencies/curves (e.g. "USD rates" and "EUR
  rates," or "a discounting curve" and a separate "lending-rate curve" for the same
  currency).
- **Equities and FX rates** (stock prices and currency exchange rates), whose simulated
  drift is tied to the simulated interest rates — this is what makes it a *cross-asset*
  simulation rather than several unrelated simulations bolted together.

The output isn't just "the interest rate at each future date" — it's expanded into a
full **yield curve** at every simulated date, in every simulated scenario. A yield curve
answers "what is $1 promised at some future date T worth today, if I'm standing at future
date t?" for every combination of t and T the caller asked for. That expanded object is
what lets [Stage 2 (Instrument Pricing)](04-instruments.md) actually price a real trade's
cashflows, which land on many different future dates.

## Why it's built this way: matching ORE's Cross-Asset Model

ORE's own simulation engine is called the **Cross-Asset Model (CAM)**. This module is a
line-by-line reimplementation of CAM's math in JAX, verified against the actual, installed
ORE software rather than against a textbook description — every formula below has either
been checked by reading ORE's own source, or by writing a small script that builds the
equivalent object in ORE and compares numbers directly. Where this was done, it's called
out explicitly.

## The pipeline, step by step

`generate_paths()` runs four phases in order. Each is implemented as (mostly) one function,
and the file is organized into matching `PHASE 1`–`PHASE 4` sections.

### Phase 1 — Quasi-Monte Carlo shock generation

**Functions:** `generate_sobol_normals()`, `_build_bridge_matrix()` /
`_apply_bridge_matrix()` / `apply_brownian_bridge()`

Simulating "thousands of alternate futures" requires thousands of sets of random numbers
— one set per scenario, one number per (time step × thing-being-simulated). This module
does **not** use ordinary random numbers. It uses a **Sobol sequence**: a specially
constructed sequence of points that fills the space of possibilities much more evenly
than ordinary randomness does, so fewer scenarios are needed to get a stable answer. This
is a standard technique in quantitative finance called Quasi-Monte Carlo (QMC).

```python
def generate_sobol_normals(num_scenarios: int, num_steps: int, num_assets: int, dtype) -> jax.Array:
```
Generates the raw Sobol sequence (via `scipy.stats.qmc.Sobol`) and converts it from
"evenly spread points between 0 and 1" into "evenly spread points that also follow a
bell-curve (Normal) distribution" — the shape random market shocks are assumed to follow.
Returns an array shaped `[TimeSteps, Scenarios, Assets]`.

**dtype caveat, and the fix applied:** `jax.scipy.stats.norm.ppf` (the function that does
the bell-curve conversion) always computes internally in 64-bit precision whenever JAX's
64-bit mode is globally turned on, *regardless* of what precision was requested for this
specific call. This function now explicitly converts its result back to the requested
`dtype` before returning, so calling it directly with `dtype=float32` reliably returns
32-bit numbers. (See [Adjustable Precision](02-architecture.md#adjustable-precision) for
why this global-setting behavior exists in the first place, and
`tests/test_market_simulations.py::TestGenerateSobolNormals` for the regression test.)

**The Brownian Bridge.** Sobol sequences are most accurate in their *first* few
dimensions and progressively noisier in later ones. A naive mapping (dimension 1 → time
step 1, dimension 2 → time step 2, ...) would waste that accuracy on early time steps and
under-serve later ones. The **Brownian bridge** construction reorders things instead: the
*final* time step gets the most accurate dimension, then the midpoint, then the
quarter-points, recursively bisecting — because that ordering captures the overall shape
of a random path with the fewest samples.

```python
def _build_bridge_matrix(time_grid: np.ndarray) -> np.ndarray:
```
Builds this reordering as an explicit matrix `B`, following the same recursive
bisection algorithm QuantLib/ORE's own `BrownianBridge` class uses (see the function's
docstring for the exact right/left/midpoint bookkeeping). It runs on the CPU with plain
NumPy, since it depends only on the time grid, not on any simulated data — it's the same
matrix for every scenario, so it's cheap to compute once.

*Verified:* `B @ B.T` (the matrix multiplied by its own transpose) is checked to exactly
equal the true covariance structure of Brownian motion, `Cov(W(s), W(t)) = min(s, t)`
— this is a strong, closed-form correctness check on the whole construction, and it's
enforced by `tests/test_market_simulations.py::TestBrownianBridge::test_matrix_reproduces_bm_covariance`.

```python
def apply_brownian_bridge(Z: jax.Array, time_grid: jax.Array) -> jax.Array:
```
Applies that matrix to the raw Sobol-derived shocks (via a small `@jax.jit`-compiled
helper, `_apply_bridge_matrix`, since this multiplication *is* data-dependent and worth
running on the GPU), then converts the result from "the bridged path's absolute value at
each time" back into "the standardized shock *between* each consecutive pair of time
steps" — which is the form the actual simulation step function (Phase 2) needs.

### Phase 2 — The Cross-Asset Model engine

**Function:** `_simulate_cross_asset_paths_jit()`

This is where the shocks generated in Phase 1 actually turn into simulated interest
rates, stock prices, and FX rates, marching forward one time step at a time.

It models two different processes, jointly correlated:

**Interest rates — the Hull-White 1-Factor (HW1F) model.** Each interest rate curve
(e.g. "USD," "EUR," or "USD discounting" vs. "USD lending") is modeled as a value that
randomly wanders but is pulled back toward a long-run average — a "mean-reverting" random
walk, the standard assumption for interest rates (they don't drift off to infinity or
negative infinity the way a stock price model might allow). The per-step update is:

```
r(t+dt) = r(t) · e^(−a·dt) + θ + σ·√variance · Z
```

where `a` (mean reversion speed), `θ` (long-run drift level), and `σ` (volatility) are
per-curve parameters supplied in the config, and `variance = (1 − e^(−2·a·dt)) / (2·a)`
is the closed-form Ornstein-Uhlenbeck transition variance for this exact time step (not
an approximation — this is the exact formula for how much a mean-reverting process
should have moved after time `dt`).

**Equities and FX — Geometric Brownian Motion (GBM) with a rate-linked drift.** Stock
prices and FX rates are modeled with the standard assumption that their *percentage*
returns (not absolute dollar changes) follow a random walk. What makes this a genuinely
*cross-asset* model rather than a bolted-on equity simulator is that each asset's drift
is tied to the simulated interest rates via **Uncovered Interest Rate Parity (UIP)** — a
standard finance principle stating that, in a risk-neutral world, an asset's expected
growth rate should equal the (risk-free) interest rate applicable to it, minus any
dividend yield it pays out. The `rate_mapping` config field encodes exactly which
interest rate curve(s) each equity/FX pair's drift depends on (and with what sign — e.g.
an FX rate depends on the *difference* between two currencies' rates).

**The numéraire.** Alongside the simulated paths, the model also tracks a money-market
account value ("numéraire") that accrues at the simulated short rate of *one* designated
base curve (curve index 0). This is a standard risk-neutral-pricing bookkeeping device;
this project's current instrument pricer ([Instruments](04-instruments.md)) doesn't
actually use it (it discounts using the yield curve cube directly instead — see that
doc for why), but it's part of a faithful CAM reimplementation and is exposed in the
output for future use.

All of this is wrapped in a single `@jax.jit`-compiled function using `jax.lax.scan` to
step through time — the JAX idiom for "run this per-step update function T times in a
row, efficiently, without a Python-level loop." This is a hard requirement from the
project's own coding constraints (see the root [README.md](../README.md)'s Technical
Constraints section): no ordinary Python `for` loops inside JIT-compiled code.

### Phase 3 — Yield curve reconstruction

**Functions:** `compute_hw_A_matrix()`, `reconstruct_yield_curves()`

Phase 2 produces a single number per curve per time step per scenario — "the short-term
interest rate right now." That alone isn't enough to price a real trade, because a
trade's cashflows land on many different future dates, each of which needs its own
discount factor. Phase 3 expands each simulated short rate into a **full curve** of
discount factors, using the Hull-White model's closed-form **affine bond-price formula**:

```
P(t, T) = A(t, T) · e^(−B(t, T) · r(t))
```

This says: "the price today (from the model's perspective, standing at future time `t`)
of $1 payable at future time `T`" is a simple function of the currently-simulated short
rate `r(t)`, plus two deterministic (non-random) terms `A` and `B` that only depend on
the model's parameters and on today's actual market curve — not on any specific
simulated scenario. Because `A` and `B` don't depend on the scenario, they're computed
**once**, on the CPU, in plain NumPy — not once per scenario, not inside the
GPU-accelerated simulation loop.

```
B(t, T) = (1 − e^(−a·(T−t))) / a
```
A closed-form function of the time gap `T − t` and the mean-reversion speed `a`.

```
A(t, T) = [P(0,T) / P(0,t)] · exp( B(t,T)·f(0,t) − (σ²/4a)·(1 − e^(−2at))·B(t,T)² )
```
This is the term that **calibrates** the model to today's actual market curve — it
guarantees that if you plug in `t = 0` (today, no simulated randomness yet), the formula
reproduces today's actual observed curve exactly. `P(0, t)` is derived from the caller's
`initial_zero_curve` input via linear interpolation on zero rates (`_initial_log_discount`
handles this), and `f(0, t)` (the initial *instantaneous forward rate*) is estimated via
a small finite-difference step.

**Important: one curve per rate factor, not one shared curve.** `compute_hw_A_matrix`
calibrates *each* rate factor independently, against *that factor's own* entry in
`config.rates.initial_zero_curves` (a list, one `ZeroCurveConfig` per factor — see
[API Reference](07-api-reference.md#ratesconfig)). This matches ORE's actual Cross-Asset
Model design: every one of ORE's `IrLgm1fParametrization` objects (its equivalent of one
Hull-White factor) is constructed with its own specific `(Currency, YieldTermStructureHandle)`
pair, and `ORE.CrossAssetModel` only ever combines a list of these already-curve-bound
objects — there is no code path anywhere in ORE that shares a single curve across
multiple currencies or factors. This was confirmed by directly constructing a live,
2-currency `ORE.CrossAssetModel` (USD at 3%, EUR at 2%, distinct flat curves) and
verifying each currency's discount factors stayed independent throughout. An earlier
version of this module *did* share one curve across every factor; that was a real bug,
now fixed and covered by
`tests/test_market_simulations.py::TestHullWhiteAMatrix::test_reprices_distinct_curves_per_rate_factor`.

*Verified:* given a flat (constant-rate) input curve, the reconstructed discount factors
exactly match the simple closed-form `e^(−rate × time)` formula, to `1e-6`
(`test_reprices_flat_curve_at_t_zero`).

```python
def reconstruct_yield_curves(hw_paths: jax.Array, A: jax.Array, B: jax.Array) -> jax.Array:
```
The `@jax.jit`-compiled function that combines the (per-scenario, per-step, simulated)
short rate paths with the (deterministic) `A`/`B` matrices into the full 4D discount
factor cube — this part **does** scale with the number of scenarios, so it runs on the
GPU.

### Phase 4 — Public API

**Function:** `generate_paths(config: SimulationConfig, precision: int = 64) -> Dict[str, jax.Array]`

Wires all three phases together: reads the typed `SimulationConfig`, builds the Sobol
shocks, bridges them, runs the cross-asset simulation, and (if the config's
`rates.maturities` field is set) reconstructs the full yield curve cube. See the
[API Reference](07-api-reference.md#generate_paths) for the exact input/output schema,
and the [User Guide](06-user-guide.md) for a runnable example.

## Output shapes at a glance

| Key | Shape | Always present? |
|---|---|---|
| `"equities"` | `[Scenarios, TimeSteps, NumEquities]` | Yes |
| `"rates"` | `[Scenarios, TimeSteps, NumRateFactors]` | Yes |
| `"numeraire"` | `[Scenarios, TimeSteps]` | Yes |
| `"yield_curves"` | `[Scenarios, TimeSteps, Maturities, NumRateFactors]` | Only if `config.rates.maturities` is set |

## Tested by

- `tests/test_market_simulations.py` — every class in this file maps to one phase above:
  `TestBrownianBridge` (Phase 1), `TestGenerateSobolNormals` (Phase 1 dtype regression),
  `TestHullWhiteAMatrix` (Phase 3, including the per-factor-curve regression test),
  `TestGeneratePaths` (end-to-end Phase 4 shape/sanity/determinism checks).
