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
│   ├── simulation.py                     Station 1: simulate the market
│   ├── scenarios.py                      Shared demo/reference configurations
│   ├── instruments/
│   │   ├── swap.py                       Station 2a: price interest rate swaps
│   │   ├── european_swaption.py          Station 2b: price European swaptions
│   │   ├── bermudan_swaption.py          Station 2c: price Bermudan swaptions
│   │   │                                 (the numeric LGM backward-induction engine)
│   │   └── american_swaption.py          Station 2d: price American swaptions
│   │                                     (a thin wrapper around bermudan_swaption.py)
│   └── risk/
│       └── statistics.py                 Station 3: compute VaR / Expected Shortfall
└── tests/
    ├── conftest.py                       Shared pytest fixtures
    ├── test_simulation.py
    ├── test_swap.py
    ├── test_european_swaption.py
    ├── test_bermudan_swaption.py
    ├── test_american_swaption.py
    ├── test_statistics.py
    ├── test_end_to_end.py
    └── test_ore_parity.py
```

Every `engine/` subpackage has an `__init__.py`, so the whole thing is importable as
`engine.simulation`, `engine.instruments.swap`, and `engine.risk.statistics` from the
repository root — no path hacks required in application code or tests.

`bermudan_swaption.py` and `american_swaption.py` are two separate files rather than one,
even though `american_swaption.py`'s content is small: `AmericanSwaptionConfig` is a
config wrapper that expands a continuous exercise window into a discrete list of dates
(`AmericanSwaptionConfig.to_bermudan()`) and then delegates entirely to
`bermudan_swaption.py`'s pricing engine — this mirrors ORE's own design, where American
swaptions are priced by discretizing the exercise window and running the exact same
numeric engine Bermudan swaptions use (`QuantExt::NumericLgmMultiLegOptionEngine`, see
[10-american-swaptions.md](10-american-swaptions.md)). Keeping the actual backward-
induction engine (state grid, Hagan's quadrature, numeraire-deflated rollback) in its own
`bermudan_swaption.py` file, separate from the thin American-specific wrapper, makes clear
that Bermudan swaptions are a fully independent, directly-usable capability — not a
byproduct of American support.

## The three-stage pipeline

```
                    ┌─────────────────────────┐
   SimulationConfig │   engine/                │  {"equities": [...],
   ───────────────► │   simulation.py          │  "rates": [...],
                     │   generate_paths()       │  "yield_curves": [...]}
                    └─────────────────────────┘
                                  │
                                  │ yield_curves cube
                                  │ [Scenarios, TimeSteps, Maturities, NumRates]
                                  ▼
                    ┌─────────────────────────┐
   SwapConfig(s)     │   engine/instruments/    │  NPV cube
   ───────────────► │   swap.py                │  [Scenarios, TimeSteps, Trades]
                     │   price_swaps()          │
                    └─────────────────────────┘
                                  │
   rates path        ┌─────────────────────────┐
   (hw_paths)         │   engine/instruments/    │  NPV cube
   ───────────────► │   european_swaption.py   │  [Scenarios, TimeSteps, Trades]
   SwaptionConfig(s)  │   price_swaptions()      │  (same shape, stacks alongside
                     └─────────────────────────┘   every other pricer's cube)
                                  │
   rates path        ┌─────────────────────────┐
   (hw_paths)         │   engine/instruments/    │  NPV cube
   ───────────────► │   bermudan_swaption.py   │  [Scenarios, TimeSteps, Trades]
   BermudanSwaption   │   price_bermudan_        │
   Config(s)          │   swaptions()            │
                     └─────────────────────────┘
                                  │
   rates path        ┌─────────────────────────┐
   (hw_paths)         │   engine/instruments/    │  NPV cube
   ───────────────► │   american_swaption.py   │  (delegates to
   AmericanSwaption   │   price_american_        │   bermudan_swaption.py)
   Config(s)          │   swaptions()            │
                     └─────────────────────────┘
                                  │
                                  │ NPV cube(s)
                                  ▼
                    ┌─────────────────────────┐
   base_npv          │   engine/risk/            │  {"VaR_95": [...], "ES_95": [...],
   ───────────────► │   statistics.py          │   "VaR_99": [...], "ES_99": [...]}
                     │   compute_risk_metrics() │
                    └─────────────────────────┘
```

Full field-level detail on every input/output is in the [API Reference](07-api-reference.md);
this page is about *why* the pieces are shaped the way they are.

### Stage 1 — Market Simulation (`engine/simulation.py`)

**Input:** a `SimulationConfig` (time grid, starting prices/rates, correlations).
**Output:** simulated paths for equities/FX, interest rates, and (optionally) a full
4D "yield curve cube" of discount factors.

This is the only stage with no dependency on ORE at runtime — it's pure JAX/NumPy/SciPy,
so it can, in principle, run on a GPU with no external process involved. See
[Market Simulation](03-market-simulation.md) for the math.

### Stage 2 — Instrument Pricing (`engine/instruments/`)

**Input:** market data from Stage 1 (the yield curve cube for the swap pricer; the raw
simulated rate paths for every swaption pricer), plus one or more trade configs.
**Output:** an NPV ("Net Present Value" — what a trade is worth today) cube.

Four pricers currently live here:

- `swap.py` — linear (no optionality) swap pricing. See
  [Instruments: Interest Rate Swaps](04-instruments.md).
- `european_swaption.py` — non-linear (single exercise date) swaption pricing via
  Jamshidian's trick, priced directly off Stage 1's simulated Hull-White rate paths
  rather than the yield-curve cube (it needs the model's own parameters, not just
  discount factors — see [Instruments: European Swaptions](08-swaptions.md)).
- `bermudan_swaption.py` — non-linear (multiple discrete exercise dates) swaption
  pricing via a numeric LGM backward-induction engine (Hagan's Gaussian-quadrature
  convolution), matching ORE's own `NumericLgmMultiLegOptionEngine` — early exercise
  has no closed form, so this is the pricing engine every other Bermudan/American
  capability builds on (see [Instruments: American & Bermudan Swaptions](10-american-swaptions.md)).
- `american_swaption.py` — a thin wrapper: discretizes a continuous exercise window
  into a dense list of dates (`AmericanSwaptionConfig.to_bermudan()`) and prices
  through `bermudan_swaption.py`'s engine, exactly the way ORE itself treats American
  exercise as a finely-discretized Bermudan.

`swap.py` and `european_swaption.py` are peer modules (neither depends on the other);
`american_swaption.py` depends on `bermudan_swaption.py` (its engine), which does not
depend on either of the other two pricers.

Every pricer **depends** on ORE at runtime (see [ORE as a dependency](#ore-as-a-dependency)
below) — they use ORE's own trade-schedule and day-count-convention machinery so that
"when does this swap pay cash, and how much" is computed exactly the way a real trading
desk's software would compute it, rather than being reimplemented from scratch.

### Stage 3 — Risk Aggregation (`engine/risk/statistics.py`)

**Input:** any NPV cube shaped `[Scenarios, TimeSteps, Trades]` (not necessarily from
Stage 2 — see below) plus a baseline value.
**Output:** Value at Risk and Expected Shortfall numbers, one per requested confidence
level, one per time step.

See [Risk Statistics](05-risk-statistics.md) for the math.

## Design principle: stages agree on shapes, not code

At the Python-module level, Stage 2 and Stage 3 do **not** import Stage 1 (or each
other) at the top of the file — `from engine.simulation import generate_paths`
only appears inside each module's `if __name__ == "__main__":` demo block, not in the
library code itself. `price_swaps()` only needs *some* array shaped
`[Scenarios, TimeSteps, Maturities, NumRates]`; every swaption pricer only needs *some*
array shaped `[Scenarios, TimeSteps, NumHW]`; none of them care whether that array came
from `generate_paths()`, a hand-built NumPy array, or a completely different simulation
engine. The same is true of `compute_risk_metrics()`: it only needs *some* array shaped
`[Scenarios, TimeSteps, Trades]` — which is exactly what every pricer produces, despite
consuming different-shaped inputs and using entirely different pricing math (linear
cashflow summation, Jamshidian's closed-form option decomposition, or numeric LGM
backward induction).

This is what makes the pipeline modular in practice, not just in diagrams — it's directly
exercised by the test suite (`tests/test_statistics.py`'s
`TestRobustAcrossInstrumentSources` tests feed `risk/statistics.py` both a fabricated,
non-swap-derived cube and a real swap-pricer cube, and assert both work identically) and
it's what let `european_swaption.py`, `bermudan_swaption.py`, and `american_swaption.py`
— three more, genuinely different instrument types after the original swap pricer —
each plug into `risk/statistics.py` with zero changes to that module.

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

`engine/scenarios.py` depends on `engine/simulation.py` (it constructs
`SimulationConfig` objects) but nothing depends on `engine/scenarios.py` except demo
code and tests — it is never required for the pipeline itself to function.

## ORE as a dependency

[ORE (Open Source Risk Engine)](https://www.opensourcerisk.org/) shows up in this codebase
in two different roles, and it's important to keep them distinct:

1. **As a design reference.** Every formula in this engine (the Brownian bridge
   construction, the Hull-White affine bond-price formula, the swap pricing formulas,
   Jamshidian's swaption formula, the LGM backward-induction engine, the VaR/ES formulas)
   was checked against ORE's actual behavior — either by reading ORE's own C++ source
   (`reference/ORE`) or Python bindings directly, or by running small scripts against the
   installed `ORE` package and comparing numbers. This is *validation*, not a runtime
   dependency.
2. **As a runtime dependency, in `engine/instruments/` only.** Every pricer in
   `engine/instruments/` (`swap.py`, `european_swaption.py`, `bermudan_swaption.py`,
   `american_swaption.py`) imports the `ORE` Python package
   (`open-source-risk-engine` on PyPI) and calls it directly — `ORE.MakeVanillaSwap`,
   `ORE.Actual365Fixed`, and related classes build the underlying swap's payment schedule
   and compute each payment's day-count fraction. This is a deliberate choice:
   schedule/day-count logic is fiddly, well-tested in ORE already, and not
   performance-critical (it runs once per trade, not once per simulated scenario), so
   there is no benefit to reimplementing it in JAX. `engine/simulation.py` and
   `engine/risk/statistics.py` have **no** runtime ORE dependency — only pure JAX/NumPy.

This means `pip install`-ing this project's core simulation and risk-statistics
functionality does not strictly require ORE, but pricing any real trade currently does
(every pricer lives under `engine/instruments/`, the only directory with a runtime ORE
dependency). If the eventual TraderX API (see the roadmap in the root
[README.md](../README.md)) is deployed as a microservice, whatever machine runs the
pricing endpoint needs ORE installed.

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
its nested `EquityConfig`, `RatesConfig`, `ZeroCurveConfig`) for Stage 1, `SwapConfig` and
`SwaptionConfig` for Stage 2. This was a deliberate choice over passing plain dictionaries: a typo in a
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
