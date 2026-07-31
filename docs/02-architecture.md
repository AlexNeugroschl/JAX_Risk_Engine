# Architecture

## Plain-language summary

The codebase is organized as three independent "stations" on an assembly line, plus a
shared library of example configurations and a test suite that checks every station's
output against ORE. Each station reads the output of the one before it, but none of them
know about each other's internal details — they agree only on the *shape* of the data
passed between them. That decoupling is deliberate: it means a fourth station (say, an
options pricer) could be added later without touching the first two at all.

## The repository layout

```
JAX_Risk_Engine/
├── README.md                            High-level goals + phased roadmap
├── requirements.txt                      Python dependencies
├── docs/                                 You are here
├── engine/
│   ├── market_simulations.py             Station 1: simulate the market
│   ├── scenarios.py                      Shared demo/reference configurations
│   ├── instruments/
│   │   └── interest_rate_swap.py         Station 2: price interest rate swaps
│   └── aggregate_statistics/
│       └── risk_statistics.py            Station 3: compute VaR / Expected Shortfall
└── tests/
    ├── conftest.py                       Shared pytest fixtures
    ├── test_market_simulations.py
    ├── test_interest_rate_swap.py
    └── test_risk_statistics.py
```

Every `engine/` subpackage has an `__init__.py`, so the whole thing is importable as
`engine.market_simulations`, `engine.instruments.interest_rate_swap`, and
`engine.aggregate_statistics.risk_statistics` from the repository root — no path hacks
required in application code or tests.

## The three-stage pipeline

```
                    ┌─────────────────────────┐
   SimulationConfig │   engine/                │  {"equities": [...],
   ───────────────► │   market_simulations.py  │  "rates": [...],
                     │   generate_paths()       │  "yield_curves": [...]}
                    └─────────────────────────┘
                                  │
                                  │ yield_curves cube
                                  │ [Scenarios, TimeSteps, Maturities, NumRates]
                                  ▼
                    ┌─────────────────────────┐
   SwapConfig(s)     │   engine/instruments/    │  NPV cube
   ───────────────► │   interest_rate_swap.py  │  [Scenarios, TimeSteps, Trades]
                     │   price_swaps()          │
                    └─────────────────────────┘
                                  │
                                  │ NPV cube
                                  ▼
                    ┌─────────────────────────┐
   base_npv          │   engine/aggregate_      │  {"VaR_95": [...], "ES_95": [...],
   ───────────────► │   statistics/            │   "VaR_99": [...], "ES_99": [...]}
                     │   risk_statistics.py     │
                     │   compute_risk_metrics() │
                    └─────────────────────────┘
```

Full field-level detail on every input/output is in the [API Reference](07-api-reference.md);
this page is about *why* the pieces are shaped the way they are.

### Stage 1 — Market Simulation (`engine/market_simulations.py`)

**Input:** a `SimulationConfig` (time grid, starting prices/rates, correlations).
**Output:** simulated paths for equities/FX, interest rates, and (optionally) a full
4D "yield curve cube" of discount factors.

This is the only stage with no dependency on ORE at runtime — it's pure JAX/NumPy/SciPy,
so it can, in principle, run on a GPU with no external process involved. See
[Market Simulation](03-market-simulation.md) for the math.

### Stage 2 — Instrument Pricing (`engine/instruments/interest_rate_swap.py`)

**Input:** the yield curve cube from Stage 1, plus one or more `SwapConfig` objects
describing specific trades.
**Output:** an NPV ("Net Present Value" — what a trade is worth today) cube.

This stage **does** depend on ORE at runtime (see [ORE as a dependency](#ore-as-a-dependency)
below) — it uses ORE's own trade-schedule and day-count-convention machinery so that
"when does this swap pay cash, and how much" is computed exactly the way a real trading
desk's software would compute it, rather than being reimplemented from scratch. See
[Instruments](04-instruments.md) for the math.

### Stage 3 — Risk Aggregation (`engine/aggregate_statistics/risk_statistics.py`)

**Input:** any NPV cube shaped `[Scenarios, TimeSteps, Trades]` (not necessarily from
Stage 2 — see below) plus a baseline value.
**Output:** Value at Risk and Expected Shortfall numbers, one per requested confidence
level, one per time step.

See [Risk Statistics](05-risk-statistics.md) for the math.

## Design principle: stages agree on shapes, not code

At the Python-module level, Stage 2 and Stage 3 do **not** import Stage 1 (or each
other) at the top of the file — `from engine.market_simulations import generate_paths`
only appears inside each module's `if __name__ == "__main__":` demo block, not in the
library code itself. `price_swaps()` only needs *some* array shaped
`[Scenarios, TimeSteps, Maturities, NumRates]`; it doesn't care whether that array came
from `generate_paths()`, a hand-built NumPy array, or a completely different simulation
engine. The same is true of `compute_risk_metrics()`: it only needs *some* array shaped
`[Scenarios, TimeSteps, Trades]`.

This is what makes the pipeline modular in practice, not just in diagrams — it's directly
exercised by the test suite (`tests/test_risk_statistics.py`'s
`TestRobustAcrossInstrumentSources` tests feed `risk_statistics.py` both a fabricated,
non-swap-derived cube and a real swap-pricer cube, and assert both work identically) and
it's what will let future instrument types (options, bonds, etc.) plug into the exact
same risk-aggregation code without any changes to `risk_statistics.py`.

## `engine/scenarios.py`: shared example configurations

Every module's `__main__` demo block, and every test file, needs *some* realistic
`SimulationConfig` to run against. Originally each file built its own copy of this
by hand; `engine/scenarios.py` now centralizes two canonical example scenarios:

- `cross_asset_demo_config()` — two equities/FX pairs and two interest rate
  currencies (USD, EUR), used to show off the full breadth of what Stage 1 can
  simulate.
- `single_currency_swap_demo_config()` — one currency with two correlated interest
  rate factors (a discounting curve and a separate forwarding curve), sized to exactly
  match a demo 2-year interest rate swap. Used by the Stage 2 and Stage 3 demos and by
  the ORE cross-check tests.

It also provides `flat_yield_curves()`, a helper that builds a deterministic (no random
simulation noise) yield curve cube directly from ORE's own curve objects — used
whenever code needs a "today's actual market, no what-if" baseline, most importantly for
Stage 3's `base_npv` input and for the tests that compare this engine's output directly
against ORE's.

`engine/scenarios.py` depends on `engine/market_simulations.py` (it constructs
`SimulationConfig` objects) but nothing depends on `engine/scenarios.py` except demo
code and tests — it is never required for the pipeline itself to function.

## ORE as a dependency

[ORE (Open Source Risk Engine)](https://www.opensourcerisk.org/) shows up in this codebase
in two different roles, and it's important to keep them distinct:

1. **As a design reference.** Every formula in this engine (the Brownian bridge
   construction, the Hull-White affine bond-price formula, the swap pricing formulas,
   the VaR/ES formulas) was checked against ORE's actual behavior — either by reading
   ORE's Python bindings' source directly, or by running small scripts against the
   installed `ORE` package and comparing numbers. This is *validation*, not a runtime
   dependency.
2. **As a runtime dependency, in `engine/instruments/` only.** `interest_rate_swap.py`
   imports the `ORE` Python package (`open-source-risk-engine` on PyPI) and calls it
   directly — `ORE.MakeVanillaSwap`, `ORE.Actual365Fixed`, and related classes build the
   swap's payment schedule and compute each payment's day-count fraction. This is a
   deliberate choice: schedule/day-count logic is fiddly, well-tested in ORE already, and
   not performance-critical (it runs once per trade, not once per simulated scenario), so
   there is no benefit to reimplementing it in JAX. `engine/market_simulations.py` and
   `engine/aggregate_statistics/risk_statistics.py` have **no** runtime ORE dependency —
   only pure JAX/NumPy.

This means `pip install`-ing this project's core simulation and risk-statistics
functionality does not strictly require ORE, but pricing any real trade currently does
(since `engine/instruments/` is presently the only pricer). If the eventual TraderX API
(see the roadmap in the root [README.md](../README.md)) is deployed as a microservice,
whatever machine runs the pricing endpoint needs ORE installed.

## Adjustable precision

One of the project's core long-term research goals (see [Overview](01-overview.md)) is
comparing risk results computed with different numeric precision — 64-bit ("double",
very precise, slower) versus 32-bit ("single", less precise, faster). This shows up in
the code as the `precision` argument to `generate_paths(config, precision=64)`.

The tricky part: JAX (the numerical library this project is built on) can only create
64-bit numbers at all if a single global setting, `jax_enable_x64`, is turned on — and
that setting applies to the *entire process*, not to individual function calls. This
isn't a limitation of this codebase; it's how JAX itself works, because 64-bit support
changes how JAX talks to the GPU. `generate_paths()` toggles this global setting itself,
right before doing any math, based on the `precision` argument it was given — so calling
`generate_paths(config, precision=64)` and then `generate_paths(config, precision=32)`
later in the same program each produce correctly-sized numbers, in sequence.

The one function that does **not** automatically manage this is
`generate_sobol_normals()`, if called directly instead of through `generate_paths()` —
its `dtype` argument is honored (an earlier bug where it wasn't has been fixed and is
covered by a regression test), but the global `jax_enable_x64` setting still needs to
already be in the state the caller wants before other, unrelated JAX code runs
elsewhere in the same process.

## Typed configuration

Every stage takes a Python `@dataclass` as its primary input — `SimulationConfig` (and
its nested `EquityConfig`, `RatesConfig`, `ZeroCurveConfig`) for Stage 1, `SwapConfig` for
Stage 2. This was a deliberate choice over passing plain dictionaries: a typo in a
dictionary key silently produces a confusing error deep inside the pipeline, while a
typo in a dataclass field name fails immediately, at the point the config object is
constructed, with a clear Python error. It's also the natural shape for the eventual
TraderX API layer to build a request/response schema around directly (see the roadmap's
Phase 8 in the root [README.md](../README.md)).

## Testing philosophy

Every non-trivial formula in this codebase is tested two ways:

1. **Direct correctness checks** — e.g. "does the Brownian bridge matrix produce the
   exact covariance structure real Brownian motion should have?"
2. **Cross-checks against ORE itself** — the installed `ORE` Python package is used
   inside the test suite to build the same trade/curve/statistic using ORE's own code,
   and the two answers are compared numerically (typically to within `1e-6` relative
   tolerance, or exactly for things like VaR/ES where the formula involves no
   floating-point-sensitive steps like matrix decompositions).

See the [User Guide](06-user-guide.md#running-the-tests) for how to run these, and each
deep-dive doc's "Tested by" section for what's covered where.
