# Architecture

## Plain-language summary

The codebase is organized as three independent modules — the simulation module, the
instrument pricers, and the risk aggregation module — plus a shared foundation layer
(model math and ORE trade-building, factored out of the instrument pricers so no formula
or schedule-building loop is implemented twice), a calibration engine, a shared library of
example configurations, and a test suite that checks every module's output against ORE.
Each module reads the output of the one before it, but none of them know about each
other's internal details — they agree only on the *shape* of the data passed between
them. That decoupling is deliberate: it means another module (say, a new instrument
pricer) could be added later without touching the others at all.

## The repository layout

```
JAX_Risk_Engine/
├── README.md                            Project pitch, status, quick start
├── pyproject.toml                        Package metadata, core deps, api/dev extras
├── requirements.txt                      Thin `-e .[api,dev]` wrapper around pyproject.toml
├── demos/                                Runnable end-to-end walkthroughs (see
│   ├── demo.py                            User Guide: Running the demos)
│   ├── demo_api.py                        - direct price_portfolio call, over the HTTP
│   ├── demo_structured.py                  API, and the HTTP API split into explicit
│   │                                       given-inputs/server-setup/server-inputs/
│   │                                       submit-and-print stages
│   ├── demo_profile_small.py             Same end-to-end path, sized so its profiler
│   │                                     trace is small enough to actually open
│   └── demo_precision.py                 FP64 vs FP32 market-risk VaR/ES against
│                                         Monte Carlo noise
├── docs/                                 Organized by topic (you are here)
│   ├── getting-started/                  Overview, user guide
│   ├── concepts/                         Architecture, market simulation, glossary,
│   │                                     coding style, profiling & the tracer
│   ├── instruments/                      Swaps, European/Bermudan/American swaptions
│   ├── risk/                             Market risk (VaR/ES), exposure, the VaR/ES
│   │                                     statistics, Delta/Gamma/Vega/Theta
│   ├── reference/                        API reference, ORE parity mapping, models &
│   │                                     trades, calibration, portfolio entry point,
│   │                                     HTTP API, EOD integration boundary
│   ├── known-issues.md                   The defect/scope-gap register -- read before
│   │                                     trusting any number
│   └── planning/                         Roadmap/history, TraderX integration plan and
│                                         the EOD contract exchange
├── engine/
│   ├── day_count.py                      Accrual day-count vocabulary, and nothing else.
│   │                                     A leaf because both models/ore_builders.py and
│   │                                     integration/note.py need the table, and
│   │                                     integration/ may not import models/ (see I-05)
│   ├── portfolio/
│   │   ├── __init__.py                   Re-exports request.py's/validation.py's public
│   │   │                                 surface, so engine.portfolio's callers see the
│   │   │                                 same names whether it's a module or a package
│   │   ├── request.py                    Top-level entry point: PortfolioRequest/
│   │   │                                 PortfolioResult/price_portfolio, plus the
│   │   │                                 validation/assembly layer (cross-field checks,
│   │   │                                 automatic maturity-pillar assembly)
│   │   ├── validation.py                 Leaf-level field validators shared by every
│   │   │                                 trade config's __post_init__ (kept separate from
│   │   │                                 request.py to avoid a circular import -- see
│   │   │                                 its own module docstring)
│   │   ├── worker_pool.py                One process pool per precision tier, plus the
│   │   │                                 opt-in XProf profiler hook and its
│   │   │                                 silent-truncation guard (see profiling.md)
│   │   └── profiling.py                  phase() -- the TraceAnnotation/named_scope pair
│   │                                     that labels each pricing stage on a trace
│   ├── api/                              FastAPI HTTP boundary -- TWO separate contracts
│   │   ├── app.py                        FastAPI app factory, mounting both routers
│   │   ├── routes.py                     /health, /version, /portfolio/price (async job
│   │   │                                 pattern), /calibration/lgm
│   │   ├── eod_routes.py                 W1.6.4 /eod/* -- the TraderX EOD contract. Plain
│   │   │                                 dicts under a published JSON Schema, NOT Pydantic:
│   │   │                                 one contract definition, not two that can drift
│   │   └── schemas.py                    Pydantic v2 request/response schemas, each with
│   │                                     .to_dataclass()/.from_dataclass()
│   ├── integration/                      TraderX EOD boundary -- hash-verified bundle in,
│   │   │                                 identified result out: both Treasury shapes price,
│   │   │                                 everything else is REFUSED. Imports no simulation
│   │   │                                 pricer, no FastAPI, no Pydantic, no JAX
│   │   │                                 (see eod-integration.md)
│   │   ├── bundle.py                     W0.1 read + hash-verify a v1/v2 bundle, in binary
│   │   ├── terms.py                      W0.2 join instrument-terms.json onto rows; W1.6.1
│   │   │                                 terms v2 + validated accrualBasis
│   │   ├── normalize.py                  W0.3 source units -> engine units, incl. the
│   │   │                                 zero-coupon accrued rule (key on terms, not blanks)
│   │   ├── conventions.py                W0.4 positive allowlist + refusal, BEFORE any
│   │   │                                 pricing object is constructed (part of I-05)
│   │   ├── result.py                     W0.5 RiskResult + per-calculation coverage model
│   │   ├── market_inputs.py              W0.6 explicit market-input mode (no silent
│   │   │                                 fallback), curve provenance, measure label
│   │   ├── identity.py                   W0.7 opaque itemId + source identity (I-10)
│   │   ├── capabilities.py               W0.9 supported product x convention x calculation
│   │   ├── bill.py                       W1.2 zero-coupon Treasury NPV -- the first pricer
│   │   ├── note.py                       W1.3 coupon-bearing Treasury NPV + rateSensitivity
│   │   ├── equity.py                     W1.4 cash equity -- a REFUSAL naming the missing
│   │   │                                 spot/FX source (I-18)
│   │   ├── schema_version.py             W1.6.2 the two document versions -- a leaf that
│   │   │                                 imports nothing, breaking result <-> schema
│   │   ├── schema.py                     W1.6.2 JSON Schema, DERIVED from the frozen
│   │   │                                 calculation/status vocabulary, never hand-written
│   │   ├── workload.py                   W1.6.4 canonical workload key + immutable attempt
│   │   │                                 store, four lookup states (part of I-08)
│   │   ├── publication.py                W0.8 crash-safe publication + the durable result
│   │   │                                 store: stage -> verify what was written ->
│   │   │                                 atomically publish -> advance pointer, with
│   │   │                                 lookup falling back to a manifest scan so a
│   │   │                                 stale pointer never loses a result (I-08)
│   │   └── pipeline.py                   Composition of the above: price_bundle()
│   ├── simulation/
│   │   ├── market_model.py               Simulates the market (Sobol/Brownian bridge,
│   │   │                                 cross-asset Hull-White paths, yield-curve
│   │   │                                 reconstruction); also validate_joint_covariance/
│   │   │                                 nearest_psd
│   │   └── demo_scenarios.py             Shared demo/reference SimulationConfig builders
│   ├── models/
│   │   ├── hull_white.py                 HW1F closed-form math (constant sigma only) --
│   │   │                                 single source of truth, used by swap.py,
│   │   │                                 european_swaption.py, simulation.py, greeks.py
│   │   ├── lgm.py                        Linear Gauss-Markov closed-form math
│   │   │                                 (piecewise-constant Sigma) -- used by
│   │   │                                 bermudan_swaption.py and engine/calibration/
│   │   ├── ore_builders.py               Shared ORE VanillaSwap construction and
│   │   │                                 cashflow extraction -- used by every pricer
│   │   └── static_key.py                 By-value hashing for the _Prepared* trade
│   │                                     structures, so they can be jax.jit STATIC
│   │                                     arguments instead of recompiling every call
│   ├── calibration/
│   │   ├── basket.py                     Co-terminal swaption basket construction and
│   │   │                                 LGM's own closed-form swaption pricer
│   │   └── lgm.py                        Bootstrap calibration of a piecewise LGM Sigma
│   │                                     to market swaption volatilities
│   ├── instruments/
│   │   ├── swap.py                       Prices interest rate swaps
│   │   ├── european_swaption.py          Prices European swaptions
│   │   ├── bermudan_swaption.py          Prices Bermudan swaptions
│   │   │                                 (the numeric LGM backward-induction engine)
│   │   ├── american_swaption.py          Prices American swaptions (a thin config
│   │   │                                 wrapper that discretizes the exercise window
│   │   │                                 and delegates to bermudan_swaption.py)
│   │   └── treasury.py                   W1.5 prices Treasury bills and notes
│   │                                     (BondConfig) by closed-form discounted
│   │                                     cashflows against one deterministic curve --
│   │                                     t=0 only, so no scenario cube and no VaR/ES
│   │                                     (I-24). The only non-JAX pricer here; imports
│   │                                     no other pricer
│   ├── market_risk/                      Short-horizon VaR/ES by full revaluation at t=0:
│   │   ├── factors.py                    RateRiskFactors -- curve pillars as risk factors
│   │   ├── scenarios.py                  Monte Carlo and historical shock scenarios
│   │   ├── revaluation.py                Every trade repriced under every scenario
│   │   └── run.py                        MarketRiskRequest/Result, run_market_risk
│   └── risk/
│       ├── var_es.py                     VaR / Expected Shortfall statistics (ORE's
│       │                                 RiskStatistics conventions)
│       ├── exposure.py                   EPE/ENE/EE_B/EEE_B/PFE over the simulated cube
│       │                                 (ORE's ExposureCalculator definitions)
│       ├── price_functions.py            Each trade's t=0 price as a JAX function of its
│       │                                 curves -- shared by Greeks and market risk
│       └── greeks.py                     Computes Delta / Gamma / Theta / Vega
└── tests/
    ├── conftest.py                       Shared pytest fixtures
    ├── test_market_model.py
    ├── test_demo_scenarios.py
    ├── test_swap.py
    ├── test_european_swaption.py
    ├── test_bermudan_swaption.py
    ├── test_american_swaption.py
    ├── test_models_piecewise_sigma.py
    ├── test_calibration_basket.py
    ├── test_calibration_lgm.py
    ├── test_calibration_integration.py
    ├── test_calibration_edge_cases.py
    ├── test_var_es.py
    ├── test_var_es_diagnostics.py        Monte Carlo standard error / tail-count
    ├── test_greeks.py
    ├── test_greeks_bermudan.py
    ├── test_treasury_instrument.py       W1.5 BondConfig, incl. the refused scenario path
    ├── test_day_count_roles.py           engine/day_count.py's convention table
    ├── test_end_to_end.py
    ├── test_diverse_portfolio_e2e.py
    ├── test_ore_parity.py
    ├── test_portfolio.py                 Cross-field validation, maturity-pillar assembly
    ├── test_portfolio_entrypoint.py       price_portfolio vs. hand-orchestrated pricing
    ├── test_portfolio_gap_fixes.py       Regressions for I-01/I-03 and friends
    ├── test_portfolio_scale_and_edge_cases.py
    ├── test_portfolio_bond_wire_through.py  Bonds reaching price_portfolio, pinned
    │                                     bit-exact against the integration pricers
    ├── test_worker_pool.py               Per-precision process pools
    ├── test_profiling_and_jit.py         phase() annotations + XLA compile counts
    ├── test_api.py                       FastAPI TestClient tests for engine/api/
    ├── test_api_bond_schemas.py          Pydantic round-trip for BondConfig
    ├── test_integration_*.py             engine/integration/, one file per task, run
    │                                     against the delivered TraderX fixtures
    │                                     (incl. test_integration_publication.py, W0.8)
    └── fixtures/traderx-eod/             Real TraderX YU18 bundles (bill/note/sofr/equity,
                                          each v1+v2), hash-pinned. LF bytes committed and
                                          held that way by .gitattributes -- CRLF translation
                                          breaks every hash (see eod-integration.md)
```

Every `engine/` subpackage has an `__init__.py`, so the whole thing is importable as
`engine.simulation.market_model`, `engine.instruments.swap`, and `engine.risk.var_es`
from the repository root — no path hacks required in application code or tests.

`bermudan_swaption.py` and `american_swaption.py` are two separate files rather than one,
even though `american_swaption.py`'s content is small: `AmericanSwaptionConfig` supplies
ORE's American option times and exercise style, and is priced by `bermudan_swaption.py`'s
engine directly — this mirrors ORE's own design, where both exercise types run through
the same numeric engine (`QuantExt::NumericLgmMultiLegOptionEngine`) and differ only in
their option times and in which coupons an exercise enters (see
[American & Bermudan Swaptions](../instruments/american-bermudan-swaptions.md)). Keeping the actual backward-
induction engine (state grid, Hagan's quadrature, numeraire-deflated rollback) in its own
`bermudan_swaption.py` file, separate from the thin American-specific wrapper, makes clear
that Bermudan swaptions are a fully independent, directly-usable capability — not a
byproduct of American support.

## The shared foundation layer: `engine/models/`

Every instrument pricer needs two kinds of thing that have nothing to do with what makes
that instrument distinctive: closed-form interest-rate model math (bond prices, bond
options, discount factors) and a real ORE trade object with a real payment schedule to
price against.

`engine/models/hull_white.py` and `engine/models/lgm.py` are the single source of
truth for that math — one JAX-native implementation of each formula, used by every pricer
that needs it.
`engine/models/ore_builders.py` is the equivalent consolidation for ORE trade-building and
cashflow extraction. See [Models & Trades](../reference/models-and-trades.md) for the full
breakdown of this shared layer, including a genuine finding this consolidation surfaced:
Hull-White and LGM (`hull_white.py` and `lgm.py` respectively) are **not** the same model
for `t>0`, despite sharing `(a, sigma)` and today's curve — a real ORE parametrization
difference between `QuantLib::HullWhite` and `QuantExt::LinearGaussMarkovModel`, not a bug
in this codebase.

## `engine/calibration/`: fitting LGM's volatility to market swaption quotes

A real
trading desk doesn't treat `hw_sigma` as an arbitrary config input — it calibrates a model's volatility parameter to reproduce
the market-quoted prices of simpler, liquid options first, and only then prices a more
complex, illiquid trade off that fitted parameter. `engine/calibration/basket.py` and
`engine/calibration/lgm.py` implement exactly that step for Bermudan/American swaptions: a
co-terminal basket of market European swaption volatilities goes in, a piecewise-constant
`engine.models.lgm.Sigma` term structure that exactly reprices every one of them comes out,
via the same bootstrap algorithm ORE itself uses by default
(`ore::data::LgmBuilder::calibrate()`'s `Bootstrap` path). See
[Calibration](../reference/calibration.md) for the full algorithm, including the two
independent verification routes used since ORE's own `AnalyticLgmSwaptionEngine` isn't
constructible through this codebase's installed Python bindings.

This module exists specifically because [Bermudan/American
Vega](../risk/greeks.md#vega-bermudanamerican-only) has no meaning without it — "how much
does this Bermudan's value change if a market quote moves" is only a well-posed question
once there is an actual market-quote-to-model relationship to differentiate through.

## The simulation-to-risk data flow

`engine/portfolio/request.py::price_portfolio` sits above every stage below and drives all of them
in sequence — see [The Public API](#the-public-api) and
[The Portfolio Entry Point](../reference/portfolio-entrypoint.md) for the full picture.
The diagram below is the same stage-by-stage flow `price_portfolio` orchestrates
internally; a caller using the pricers directly (as `demo.py` used to, before Phase 2)
still wires these together by hand.

```
                    ┌─────────────────────────┐
   PortfolioRequest │   engine/                │  PortfolioResult
   ───────────────► │   portfolio.py           │  {base_npv, npv_cube, risk,
                     │   price_portfolio()      │   greeks, warnings}
                    └─────────────────────────┘
                                  │  validates, derives maturity pillars,
                                  │  calibrates, then drives every stage below
                                  ▼
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
   base_npv,         │   engine/risk/            │  ExposureProfile: EPE, ENE, EE_B,
   numeraire ─────► │   exposure.py            │   EEE_B, PFE_95, PFE_99 per date
                     │   netting_set_profile()  │
                    └─────────────────────────┘
```

Market risk is a separate, shorter pipeline that does not simulate paths at all:

```
   ShockScenarios     ┌─────────────────────────┐   [S, N] P&L   ┌──────────────────┐
   (Monte Carlo or ─► │ engine/market_risk/      │ ─────────────► │ engine/risk/      │ VaR_99,
   historical)        │ revaluation.py: every    │                │ var_es.py         │ ES_97.5, ...
   trades ──────────► │ trade repriced at t=0    │                │ compute_risk_     │
                      │ under every shock        │                │ metrics()         │
                     └─────────────────────────┘                └──────────────────┘
```

Full field-level detail on every input/output is in the [API Reference](../reference/api-reference.md);
this page is about *why* the pieces are shaped the way they are.

### Market Simulation (`engine/simulation/market_model.py`)

**Input:** a `SimulationConfig` (time grid, starting prices/rates, correlations).
**Output:** simulated paths for equities/FX, interest rates, and (optionally) a full
4D "yield curve cube" of discount factors.

This is the only module with no dependency on ORE at runtime — it's pure JAX/NumPy/SciPy,
so it can, in principle, run on a GPU with no external process involved. See
[Market Simulation](market-simulation.md) for the math.

### Instrument Pricing (`engine/instruments/`)

**Input:** market data from the simulation module (the yield curve cube for the swap
pricer; the raw simulated rate paths for every swaption pricer), plus one or more trade
configs.
**Output:** an NPV ("Net Present Value" — what a trade is worth today) cube.

Five pricers currently live here:

- `swap.py` — linear (no optionality) swap pricing. See
  [Instruments: Interest Rate Swaps](../instruments/swaps.md).
- `european_swaption.py` — non-linear (single exercise date) swaption pricing via
  Jamshidian's trick, priced directly off the simulation module's simulated Hull-White
  rate paths rather than the yield-curve cube (it needs the model's own parameters, not
  just discount factors — see [Instruments: European Swaptions](../instruments/european-swaptions.md)).
- `bermudan_swaption.py` — non-linear (multiple discrete exercise dates) swaption
  pricing via a numeric LGM backward-induction engine (Hagan's Gaussian-quadrature
  convolution), matching ORE's own `NumericLgmMultiLegOptionEngine` — early exercise
  has no closed form, so this is the pricing engine every other Bermudan/American
  capability builds on (see [Instruments: American & Bermudan Swaptions](../instruments/american-bermudan-swaptions.md)).
- `american_swaption.py` — American exercise: ORE's uniform option-time grid over the
  window and ORE's broken-period exercise (a coupon belongs until its accrual end,
  credited `couponRatio`), priced through `bermudan_swaption.py`'s engine.
- `treasury.py` (W1.5) — Treasury bills and notes (`BondConfig`), closed-form
  discounted cashflows against the bond's **own** zero curve. **The odd one out in two
  ways.** It is the only pricer here that is not JAX — plain `math.exp` over an ORE day
  count, since there is no cube to vectorize over and nothing to differentiate. And it
  is the only one that produces **no NPV cube**: a bond has no stochastic driver, so it
  has no scenario dimension and therefore no VaR/ES. That is refused explicitly rather
  than filled with a broadcast constant, which would report VaR 0.00 / ES NaN for a
  position whose risk was never modelled — see [I-24](../known-issues.md#i-24).

`swap.py` and `european_swaption.py` are peer modules (neither depends on the other);
`american_swaption.py` depends on `bermudan_swaption.py` (its engine), which does not
depend on either of the other two pricers.

Every pricer **depends** on ORE at runtime (see [ORE as a dependency](#ore-as-a-dependency)
below) — they use ORE's own trade-schedule and day-count-convention machinery, via
`engine/models/ore_builders.py`, so that "when does this swap pay cash, and how much" is
computed exactly the way a real trading desk's software would compute it, rather than
being reimplemented from scratch. Their own model math (bond prices, bond options) comes
from `engine/models/hull_white.py` (`swap.py`, `european_swaption.py`) or
`engine/models/lgm.py` (`bermudan_swaption.py`/`american_swaption.py`) — see
[Models & Trades](../reference/models-and-trades.md).

### Exposure (`engine/risk/exposure.py`)

**Input:** the simulated NPV cube, the numeraire paths and today's discount factors.
**Output:** ORE's exposure profile per date — EPE, ENE, EE_B, EEE_B, PFE — for the
netting set and for each trade. See [Exposure](../risk/exposure.md).

### Market Risk (`engine/market_risk/`)

**Input:** trades and `ShockScenarios` (absolute moves of every curve pillar over a short
horizon). **Output:** the P&L of every trade under every scenario, and its VaR/ES. It
does not use the simulated cube. See [Market Risk](../risk/market-risk.md).

### Risk Statistics (`engine/risk/var_es.py`)

**Input:** any cube shaped `[Scenarios, TimeSteps, Trades]` (not necessarily from the
instrument pricers — see below) plus a baseline value. The market-risk path passes its
P&L sample as a one-date cube.
**Output:** Value at Risk and Expected Shortfall numbers, one per requested confidence
level, one per time step.

See [Risk Statistics](../risk/var_es.md) for the math.

### Sensitivities (`engine/risk/greeks.py`)

**Input:** one `SwapConfig`/`SwaptionConfig`/`BermudanSwaptionConfig` plus its own
`ZeroCurve` (a JAX-array counterpart of `ZeroCurveConfig` — see below), and, for Vega, a
list of `CalibrationTarget`s.
**Output:** per-curve-pillar Delta/Gamma, a single Theta number, and (Bermudan/American
only) per-basket-instrument Vega, for that one trade.

Unlike `var_es.py`, this module is **not** instrument-agnostic — it imports directly from
`engine.instruments.swap`/`engine.instruments.european_swaption`/
`engine.instruments.bermudan_swaption` and reuses their own JAX-native pricing building
blocks (via automatic differentiation, `jax.grad`/`jax.hessian`), rather than only
consuming a generic NPV cube. It covers every *rate-derivative* instrument in this
codebase — Bermudan/American Greeks became possible once `bermudan_swaption.py`'s backward
induction was ported to `jax.lax.scan`, and Vega became well-defined once
`engine/calibration/` existed to supply a genuine market-vol-to-model relationship. See
[Delta, Gamma, and Theta](../risk/greeks.md) for the full story, including two real
autodiff-through-bisection gradient bugs found and fixed while building this.

**Bonds are the exception, and their Greeks do not live here.** `BondConfig` is priced by
`engine/instruments/treasury.py`, which is plain `math.exp` arithmetic rather than JAX, so
there is nothing for `jax.grad` to differentiate. Its Delta/Gamma/Theta are computed by
`engine/portfolio/request.py::_bond_greeks` as **bumped revaluations** instead — a central
difference at ±1bp for Delta and Gamma, and a one-calendar-day reprice for Theta. Vega is
*omitted* rather than reported as zero (a fixed-coupon bond off a deterministic curve has
no volatility input), and Theta is likewise omitted for a bond maturing tomorrow, where
there is no next day on which the instrument still exists. See
[The Portfolio Entry Point](../reference/portfolio-entrypoint.md#greeks).

## The Public API

**Input:** a `PortfolioRequest` (`engine/portfolio/request.py`) — market data, a heterogeneous
list of trades, and risk parameters, everything the pipeline above needs, in one object.
**Output:** a `PortfolioResult` — prices, risk, and (optionally) Greeks, for the whole
portfolio, in one object.

`engine/portfolio/request.py::price_portfolio` is the single entry point that ties every stage
above together — the "one function to call" answer to "what should a caller of the whole
system hand over, and what do they get back." It also carries the validation/assembly
layer that makes an arbitrary (not hand-built) portfolio safe to run through this pipeline:
cross-checking every trade's own duplicated `hw_a`/`hw_sigma`/`initial_zero_curve` against
the simulation's `RatesConfig` (`validate_portfolio_against_simulation`), automatically
deriving `RatesConfig.maturities` from every swap's real ORE schedule
(`derive_maturity_pillars`) instead of requiring a caller to hand-compute pillars the way
early demos did, and surfacing (not silently absorbing) known scope boundaries like a swap
aged past its first accrual as warnings. See
[The Portfolio Entry Point](../reference/portfolio-entrypoint.md) for the full field-level
reference and [`docs/planning/traderx-integration.md`](../planning/traderX_integration/traderx-integration.md)
for the gap analysis this validation layer closes.

`engine/api/` (`app.py`/`routes.py`/`schemas.py`) wraps `price_portfolio` behind a FastAPI
HTTP API — a thin transport layer, not a second place orchestration logic lives. Pydantic
models in `engine/api/schemas.py` mirror `PortfolioRequest`/`PortfolioResult`/every
instrument config field-for-field, converting to/from the real dataclasses at the HTTP
boundary; `engine.portfolio` and everything below it has zero Pydantic/FastAPI dependency,
so the core simulation/pricing/risk engine still doesn't require the heavier `api` extra to
use as a plain Python library. See [HTTP API](../reference/http-api.md) for the endpoint
reference, including why `POST /portfolio/price` returns a job id and polls rather than
blocking (a measured ~52-second wall-clock time for a modest portfolio, dominated by JAX
JIT compilation and Monte Carlo simulation).

## Design principle: modules agree on shapes, not code

At the Python-module level, the instrument pricers and the risk aggregation module do
**not** import the simulation module (or each other) at the top of the file — `from engine.simulation.market_model import generate_paths`
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
exercised by the test suite (`tests/test_var_es.py`'s
`TestRobustAcrossInstrumentSources` tests feed `risk/var_es.py` both a fabricated,
non-swap-derived cube and a real swap-pricer cube, and assert both work identically) and
it's what let `european_swaption.py`, `bermudan_swaption.py`, and `american_swaption.py`
— three more, genuinely different instrument types after the original swap pricer —
each plug into `risk/var_es.py` with zero changes to that module.

## `engine/simulation/demo_scenarios.py`: shared example configurations

Every module's `__main__` demo block, and every test file, needs *some* realistic
`SimulationConfig` to run against. Originally each file built its own copy of this
by hand; `engine/simulation/demo_scenarios.py` now centralizes two canonical example
scenarios:

- `cross_asset_demo_config()` — two equities/FX pairs and two interest rate
  currencies (USD, EUR), used to show off the full breadth of what the simulation module
  can simulate.
- `single_currency_swap_demo_config()` — one currency with two correlated interest
  rate factors (a discounting curve and a separate forwarding curve), sized to exactly
  match a demo 2-year interest rate swap. Used by the instrument-pricing and
  risk-aggregation demos and by the ORE cross-check tests.

It also provides `flat_yield_curves()`, a helper that builds a deterministic (no random
simulation noise) yield curve cube directly from ORE's own curve objects — used
whenever code needs a "today's actual market, no what-if" baseline, most importantly for
the risk aggregation module's `base_npv` input and for the tests that compare this
engine's output directly against ORE's.

`engine/simulation/demo_scenarios.py` depends on `engine/simulation/market_model.py`
(it constructs `SimulationConfig` objects) but nothing depends on
`engine/simulation/demo_scenarios.py` except demo code and tests — it is never required
for the pipeline itself to function.

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
   there is no benefit to reimplementing it in JAX. `engine/simulation/market_model.py`
   and `engine/risk/var_es.py` have **no** runtime ORE dependency — only pure JAX/NumPy.

This means `pip install`-ing this project's core simulation and risk-statistics
functionality does not strictly require ORE, but pricing any real trade currently does
(every pricer lives under `engine/instruments/`, the only directory with a runtime ORE
dependency). If the eventual TraderX API (see the roadmap in the root
[README.md](../../README.md)) is deployed as a microservice, whatever machine runs the
pricing endpoint needs ORE installed.

## Adjustable precision

One of the project's core long-term research goals (see [Overview](../getting-started/overview.md)) is
comparing risk results computed with different numeric precision — 64-bit ("double",
very precise, slower) versus 32-bit ("single", less precise, faster), and eventually
pushing well below that to 8-bit and 4-bit formats.

As of `PrecisionConfig` (`engine/portfolio/request.py`), this is exposed as **four
independent knobs**, two of which optionally drill down further:

```python
@dataclass(frozen=True)
class PrecisionConfig:
    simulation: int = 64                                     # Monte Carlo path generation (generate_paths)
    pricing: Union[int, PricingPrecisionOverride] = 64        # instrument NPV / npv_cube dtype
    risk: Union[int, RiskPrecisionOverride] = 64              # VaR/ES + Greeks
    calibration: int = 64                                     # LGM sigma bootstrap dtype
```

Passed as `PortfolioRequest(..., precision=PrecisionConfig(simulation=64, pricing=32, risk=32))`
(or, over HTTP, a `"precision": {"simulation": 64, "pricing": 32, "risk": 32}` block on
`POST /portfolio/price` — see [HTTP API](../reference/http-api.md)). All four default to
64, byte-identical to this project's behavior before `PrecisionConfig` existed.

**`pricing` and `risk` each optionally accept a structured override** instead of a flat
`int`, for finer-than-module-level control:

```python
@dataclass(frozen=True)
class PricingPrecisionOverride:
    default: int = 64
    swap: Optional[int] = None
    european_swaption: Optional[int] = None
    bermudan_swaption: Optional[int] = None
    american_swaption: Optional[int] = None

@dataclass(frozen=True)
class RiskPrecisionOverride:
    default: int = 64
    delta_gamma: Optional[int] = None   # ONE knob for both -- see below
    theta: Optional[int] = None
    vega: Optional[int] = None
    exposure: Optional[int] = None
```

A flat `int` is sugar for "every sub-field at this precision" — it resolves through the
exact same `_resolve_pricing_dtype`/`_resolve_risk_dtype` helpers a structured override
uses, never a separate code path, so `PrecisionConfig(pricing=32)` (today's exact call
shape) keeps working byte-identically. Any override field left `None` falls back to that
override's own `default`. `simulation` and `calibration` stay flat-`int`-only: each has
exactly one call site in the pipeline (`generate_paths`, the LGM bootstrap), so no
drill-down axis applies to either.

`delta_gamma` is one shared field, not split further into Delta/Gamma: `swaption_delta_gamma`/
`bermudan_delta_gamma` each derive both from a single `_grad_and_hessian_diagonal` call
against one curve (one gradient plus one batched Hessian-vector-product pass — see
[Profiling & the Tracer §3.4](profiling.md)) — splitting them would mean either duplicating
the curve-build and the autodiff trace, or restructuring `engine/risk/greeks.py`'s public
functions themselves, out of scope for a wrap-don't-invade config redesign.

`exposure` is structurally different from the other three `risk` sub-fields: the exposure
statistics derive their dtype from `npv_cube` (i.e. from `pricing`'s own output), not from
any curve. `price_portfolio` honors an `exposure` override that differs from `pricing` via
an explicit re-cast of `npv_cube` immediately before computing the profiles — not a curve
substitution like `delta_gamma`/`theta`/`vega`. This means `risk.exposure` can only ever
*narrow* precision relative to whatever `pricing` already produced; it can't recover
precision `pricing` already lost.

(The market-risk path, `engine.market_risk`, has a single `precision` of its own: the
revaluation and its VaR/ES statistics run at one dtype.)

A real, honest consequence of per-instrument-type `pricing`: when different buckets resolve
to different dtypes, `_price_by_type`'s final `jnp.stack` promotes the assembled `npv_cube`
to the WIDEST dtype present (confirmed directly: `jnp.stack([float64_arr, float32_arr],
axis=-1).dtype == float64`). A mixed-precision pricing request still controls the *cost* of
computing each bucket's own cube — an expensive Bermudan tree running cheaper while a
trivial swap stays exact — but `npv_cube.dtype` itself reflects the widest bucket present,
not necessarily the one a caller drilled down on.

**`simulation`** works exactly as before: it's passed straight through to
`generate_paths(config, precision=...)`.

**`pricing`** is *not* a parameter any pricer function accepts. `price_swaps`/
`price_swaptions`/`price_bermudan_swaptions`/`price_american_swaptions` already derive
their own working dtype from whatever JAX array they're handed (`yield_curves.dtype`,
`hw_paths.dtype`, etc.) — the actual gap was `engine/portfolio/request.py` constructing
some of *its own* arrays (`step_times`, `_base_npv`'s swaption zero-shock path,
`_flat_curve_cube`'s output) as hardcoded `float64` before ever reaching a pricer.
`pricing` fixes that at the source: `price_portfolio` builds every one of these arrays at
`precision.pricing`'s dtype, and the pricers propagate it onward exactly as they already
did. There is deliberately no signature change to any of the four pricer modules.

Two subtler gaps surfaced during implementation, beyond what the initial code-reading
pass found, and both needed a real fix rather than just dtype plumbing in `request.py`:

- `price_portfolio`'s main (non-base) pricing path was originally handing `price_swaps`/
  `price_swaptions`/etc. `generate_paths`' own output (`market["rates"]`/
  `market["yield_curves"]`) directly — which is governed by `precision.simulation`, not
  `precision.pricing`. Since the pricers derive dtype from *whichever* JAX array they're
  handed, `pricing=32` with `simulation=64` (the plan's own explicit end-to-end
  verification scenario) had no effect on `npv_cube` at all until `price_portfolio` was
  changed to re-cast `market["rates"]`/`market["yield_curves"]` to `precision.pricing`'s
  dtype before routing to the pricers (a no-op cast, and free, whenever the two knobs
  already agree — the common case).
- `european_swaption.py`'s Jamshidian root-find (`_solve_rstar`/`_bisect_rstar`)
  initialized its bisection bracket via bare `jnp.ones(t_shape) * 2.0` — `jnp.ones` with
  no explicit `dtype` silently picks up JAX's *ambient* default float dtype (float64
  whenever `jax_enable_x64` is on, regardless of what dtype the rest of the pricing
  computation actually wants), which upcast the solved root `r*` back to float64 and, in
  turn, every downstream Black-formula quantity built from it — even though `A_T0_Ti`/
  `B_T0_Ti`/every other array in `_price_one_swaption` was already correctly float32.
  This was invisible in the pre-`PrecisionConfig` codebase because every caller used the
  same precision throughout; it surfaced immediately once `pricing=32` was exercised with
  `jax_enable_x64` left on by a `simulation=64` sibling knob. Fixed by deriving `dtype`
  from `params`' own leaves (`jax.tree_util.tree_leaves` + `jnp.result_type`) in
  `_solve_rstar` and threading it through to `_bisect_rstar`'s bracket construction.

Both fixes were found by literally exercising `pricing=32` end-to-end (per-pricer-type,
not just a per-array code read) rather than trusting the initial "these four pricers need
no changes" research alone — the plan that scoped this feature flagged exactly this kind
of gap as something to re-verify during implementation, not assume away.

**`risk`** governs the exposure statistics (`engine.risk.exposure`, fully dtype-agnostic — they
reflect whatever dtype the precision-controlled `npv_cube` already has) and Greeks
(`engine/risk/greeks.py`). The mechanism here is curve-driven, not a new Greeks
parameter: `_compute_all_greeks` builds each trade's `ZeroCurve` at `precision.risk`'s
dtype (via `ZeroCurve.from_config(config, dtype=...)`), and every Greeks closure that
used to hardcode `jnp.float64` for its own intermediate arrays (cashflow times/amounts,
the Bermudan/American state grid's quadrature weights and trade-schedule arrays, the
Vega Jacobian's accumulator) now derives that dtype from the `curve`/`x_nodes` parameter
it's already handed. This mattered more than it looks: because `jax_enable_x64` is a
single process-global flag (see below), any *one* hardcoded-`float64` array left in this
chain silently upcasts a `risk=32` computation back to float64 the moment it's combined
with the correctly-sized array — confirmed directly (`jnp.interp`/elementwise ops promote
a float32/float64 mix to float64 whenever `jax_enable_x64` is on, regardless of which
operand is which dtype). Getting this right for `bermudan_swaption.py`'s `_state_grid`/
`_run_backward_induction`/`_cashflow_values_at_nodes` in particular required tracing the
*entire* chain of arrays feeding the backward induction, not just the one function whose
docstring already mentioned a dtype, since that function is shared between plain
(non-Greeks) Bermudan/American pricing — which must stay governed by `pricing`, not
`risk` — and Greeks. The scoping trick: `_zero_curve_of` (both the one in `request.py`
and `bermudan_swaption.py`'s own copy) preserves whatever dtype it's handed rather than
hardcoding one, so plain pricing's curve stays float64 (built from a plain `np.ndarray`)
while Greeks' curve carries whatever `risk`-precision JAX array
`engine.risk.greeks._bermudan_price_fn` substituted in — one signal, two correct
behaviors, no separate parameter needed.

When `risk` is a `RiskPrecisionOverride` with `delta_gamma` and `theta` resolving to
different dtypes, `_compute_all_greeks` builds two separate `ZeroCurve`s per trade (one per
metric) rather than one shared curve — the same curve-driven mechanism, just invoked twice.
When both resolve to the same dtype (the common flat-`risk=N` case), this pays one small,
redundant extra curve-construction call: a deliberate simplicity-over-micro-optimization
choice, since building a `ZeroCurve` is a cheap pillar-count array build, not a
JIT-compiled trace.

**`calibration`** governs the LGM sigma bootstrap (`engine/calibration/lgm.py::calibrate_lgm_sigma`,
invoked by `_fill_calibrated_sigma` once per distinct `rate_factor_index` needing
calibration). Unlike `risk`, this was *not* free: `calibrate_lgm_sigma` hardcoded
`jnp.float64` at four internal sites (the bisection's per-bucket `times`/`values` arrays and
the final `Sigma`'s own arrays) regardless of what curve it was handed. Fixed by deriving
`dtype = curve.pillar_rates.dtype` once at the top of the function and using it at all four
sites — the same derive-from-curve pattern `engine/risk/greeks.py` already established, so
`_fill_calibrated_sigma` controls it purely by handing `calibrate_lgm_sigma` a curve built
at `precision.calibration`'s dtype, with no new parameter on the calibration function
itself. One caveat worth stating plainly rather than assuming away: the bisection's
`market_price`/`new_value` round-trip through plain Python `float` (always float64-precision
in CPython) between bucket iterations, even at `calibration=32` — verified end-to-end (see
`tests/test_portfolio_entrypoint.py::TestPricePortfolioPrecision::
test_calibration_precision_flows_through_lgm_bootstrap`) to be immaterial: `jnp.asarray`
downcasts the Python float back to float32 correctly on the very next line, and the
resulting `Sigma`'s own dtype is confirmed float32 throughout.

**The `jax_enable_x64` process-global-flag mechanism itself is unchanged**: JAX (the
numerical library this project is built on) can only create 64-bit numbers at all if a
single global setting, `jax_enable_x64`, is turned on — and that setting applies to the
*entire process*, not to individual function calls or threads. This isn't a limitation of
this codebase; it's how JAX itself works, because 64-bit support changes how JAX talks to
the accelerator (CPU, GPU, or TPU). `generate_paths()` toggles this global setting itself,
right before doing any math, based on the `precision` argument it was given.

The one function that does **not** automatically manage this is
`generate_sobol_normals()`, if called directly instead of through `generate_paths()` —
its `dtype` argument is honored (covered by a regression test), but the global
`jax_enable_x64` setting still needs to already be in the state the caller wants before
other, unrelated JAX code runs elsewhere in the same process.

### Concurrency: a multi-process worker pool, with `_PRICING_LOCK` as defense-in-depth

**This section describes a genuine architecture change**, not a terminology fix over the
previous "single-consumer, low-volume, queue behind a lock" design. That previous design
directly contradicted this project's actual purpose (see [Overview](../getting-started/overview.md)):
running pricing/simulation across *multiple* TPU devices concurrently is the point, and a
single process-wide lock caps the whole process at one device's worth of work in flight at
a time, no matter how many devices are available.

**The underlying JAX fact hasn't changed and can't be worked around**: `jax_enable_x64` is
process-global state, not thread-local, and JAX provides no per-thread, per-device, or
per-mesh scoped alternative (confirmed at the JAX source level: `jax/_src/config.py`'s own
maintainers explicitly excluded this one flag from the context-manager-scoping mechanism
every other JAX config flag gets). Two *threads* in one process wanting different
precisions at the same time cannot both be correct without serializing. Two *processes*,
each fixing `jax_enable_x64` once at boot and never touching it again, have entirely
independent JAX/XLA runtimes and never race each other — that's the mechanism this
architecture uses to get real concurrency instead of accepting the lock's serialization as
a permanent ceiling.

**The architecture**: `engine/portfolio/worker_pool.py` maintains one
`ProcessPoolExecutor`-backed pool per precision tier (float32 and float64 — the two values
`PrecisionConfig.simulation` allows), not one pool per raw device. Each worker process, via
the executor's `initializer=` parameter (`_worker_init`, a plain top-level function — see
that module's own docstring for why it can't be a lambda/closure), does exactly two things
once at worker boot and never again for its lifetime: sets `jax.config.update("jax_enable_x64", ...)`
matching its own fixed tier, and pins itself to one device via process-launch-time
environment configuration (a deliberate no-op on this CPU-only dev machine, which has
exactly one `CpuDevice` — see that module's docstring for the real-TPU-deployment shape
this leaves ready without implementing against hardware this repo cannot test). Each
worker then processes jobs strictly sequentially, one at a time, by construction — which is
what makes `_PRICING_LOCK` unnecessary *at the worker level*: no second thread in that
process ever calls into JAX-executing code while a job is in flight. `N` workers in a
tier's pool means `N` jobs of that tier can run genuinely concurrently; a float32-tier job
and a float64-tier job run in separate pools/processes and are therefore always genuinely
concurrent with each other, not time-sliced behind one flag.

`submit_pricing_job(request)` routes purely by `request.precision.simulation` — the one
knob that actually drives `generate_paths`'s own `jax.config.update` call.
`pricing`/`risk` stay independently-settable dtypes *within* a job, honored by
`price_portfolio`'s own explicit casts once inside whichever tier's worker the job landed
on (including the existing re-enable-`jax_enable_x64`-after-`generate_paths` logic
described above, unchanged). `PrecisionConfig`'s/`PortfolioRequest`'s public shape did not
change at all for this — `simulation` already was the natural routing selector.

**`_PRICING_LOCK` is kept, not removed — its role narrowed to defense-in-depth.** No code
change to the lock itself. The underlying JAX fact it protects against doesn't go away
just because the worker pool makes it unreachable through the normal HTTP path: anything
that ever puts two threads of the *same* process inside `price_portfolio` concurrently — a
worker-pool sizing bug, or a future direct Python caller spinning up their own threads
against `engine.portfolio` directly (that module is deliberately zero-Pydantic/FastAPI-
dependency, designed to be called directly, not only through the HTTP/worker-pool layer) —
would hit the exact same corruption bug the lock was built to prevent. The lock is cheap
(uncontended-lock overhead is negligible next to a JIT-compile-dominated multi-second job)
and remains correctness-critical whenever "one job per worker process" doesn't hold; it is
no longer, however, the *primary* mechanism limiting concurrency — that role now belongs to
the worker pool's process boundaries.

**What this buys, and what it doesn't (yet).** Concurrent `/portfolio/price` jobs across
*different* precision tiers now genuinely run in parallel, on separate OS processes, rather
than queuing behind one lock. Same-tier jobs beyond that tier's own pool size still queue —
expected pool exhaustion, not a bug, and no different in kind from any fixed-size worker
pool. A real cost worth stating honestly: JIT compilation cache was shared process-wide
across threads under the old design; under N worker processes, each process pays its own
compilation cost independently, since compiled XLA programs aren't shared across separate
OS processes. Real device-count-aware pool sizing (matching `len(jax.devices())` on an
actual Cloud TPU VM host) and TPU-specific environment-variable device pinning
(`JAX_PLATFORMS=tpu`/`TPU_VISIBLE_CHIPS`) are deferred to actual TPU deployment, not
designed here — see [Roadmap & History](../planning/roadmap-and-history.md). See
[HTTP API](../reference/http-api.md) for the dispatcher-level job-store details and
[`engine/portfolio/worker_pool.py`](../../engine/portfolio/worker_pool.py) for the full
implementation and its own extensive docstring.

### Option B: why uniform sub-float32 precision is not achievable today

This project's long-term research goal includes pushing precision down to 8-bit and 4-bit
formats *throughout* the pipeline, not just at isolated points. Two materially different
paths exist, and it matters which one this codebase has:

- **Option A (designed, not yet implemented in this codebase — `MatmulPrecisionConfig`).**
  FP8/FP4 applied only at two matmul-shaped sub-steps inside `generate_paths`, paired with
  float32 accumulation — deliberately narrow, targeting exactly the operations where a
  low-precision matmul kernel exists and is numerically sound. Everything else stays at
  `PrecisionConfig`'s float32-or-float64 knobs described above. This is a separate,
  previously-scoped piece of work, tracked on its own; it does not exist as code yet.
- **Option B (NOT implemented, this section).** Uniform sub-float32 precision *everywhere*,
  including the two operations Option A targets. A fundamentally different, larger
  undertaking — not a natural extension of Option A or of the `PrecisionConfig` hierarchy
  this document otherwise describes.

**What's confirmed broken, on which exact backend.** Live-tested against this project's
installed `jax==0.10.2`/`jaxlib==0.10.2` CPU backend: `jnp.linalg.cholesky` raises
`NotImplementedError` for `bfloat16`, `float16`, `float8_e4m3fn`, `float8_e5m2`, and
`float4_e2m1fn` alike — the CPU backend's LAPACK-backed linalg path has no reduced-precision
kernel at all, for any format below float32, not specifically for bfloat16 or the FP8/FP4
family. `jax.scipy.stats.norm.ppf` raises `TypeError` on the same set of dtypes — an
independently confirmed, not inferred, second instance of the same class of gap.
`PrecisionConfig` therefore accepts only `{32, 64}` for every one of its four knobs.

**A previously-considered belief, corrected.** An earlier planning pass considered bfloat16
specifically "a first-class XLA dtype with full CPU kernel coverage," reasoning from its
broad ML-training use. Tested live, this doesn't hold: `cholesky`/`norm.ppf` fail on
bfloat16 with the *exact same* signature as FP8/FP4 — there is no meaningfully
easier-to-reach reduced-precision tier on this backend.

**Why these two operations are architecturally central, not a peripheral gap.**
`jnp.linalg.cholesky` factorizes the joint covariance matrix into the correlation structure
coupling every rate/equity factor's simulated shocks — this *is* how cross-asset correlation
enters the simulation. `jax.scipy.stats.norm.ppf` converts each Sobol uniform draw into a
normal shock — the fundamental step every path/factor/time-step consumes. Both run once per
(scenario, time step, factor), not once per simulation — there's no way to "work around"
them the way Option A already routes around the two matmul sub-steps; Option B means running
these operations *themselves* below float32, which is exactly the part with no kernel to
fall back to.

**What would actually need to be built.** Two independent, from-scratch numerical-kernel
efforts: (1) a custom low-precision Cholesky avoiding LAPACK entirely, built from primitives
that *do* have low-precision coverage — matmul, proven by Option A's own two insertion
points — via Newton-Schulz iteration (matmul-only, quadratically convergent) or
blocked/recursive elimination; (2) a custom inverse-normal-CDF avoiding `norm.ppf`'s
special-function kernel, built from pure elementwise arithmetic — a rational/polynomial
minimax approximation (e.g. Wichura's AS 241) or an Acklam/Beasley-Springer-Moro-style
closed-form approximation. Both would then need this codebase's own established validation
bar, not a lighter one: cross-checked against the float64 originals for numerical agreement,
*and* a separate pass confirming every downstream computation (pricing, VaR/ES, Greeks)
stays within an acceptable error band at the target precision.

**Honest sizing.** This is materially larger and higher-risk than the `PrecisionConfig`
hierarchy this document otherwise describes — not a quick follow-on. The hierarchy is
dtype-plumbing through code paths that already work correctly at every precision they
support; Option B means designing, implementing, and independently validating two new
numerical algorithms from scratch, each replacing a library primitive this codebase has
relied on since its first ORE cross-check. No timeline is given deliberately — an honest
estimate needs a working prototype of at least one candidate first, which is research work,
not implementation work with a knowable estimate.

**TPU behavior: an explicit, labeled, unverified hypothesis.** Everything above was tested
on this CPU-only dev machine. It is *plausible* — genuinely unverified, not merely "probably
fine" — that a Cloud TPU's native XLA backend has broader low-precision kernel coverage,
since bfloat16 is TPU's own native compute format. This cannot be tested on this repo's
current CPU-only environment and is not claimed as fact. Confirming or refuting it is real
TPU deployment work, listed here once, not duplicated speculatively elsewhere.

### Out of scope for v1

- **Sub-float32 precision outside Option A's two targeted matmul sub-steps.** See "Option B"
  above for the full accounting of what's blocked, why, and what building it would require.
- **FP8/INT8/INT4/NF4 anywhere in this codebase**, including Option A's own
  `MatmulPrecisionConfig` mechanism, remain a further-out research goal not yet implemented
  (see [Overview](../getting-started/overview.md)) — this pass built `PrecisionConfig`'s
  {32, 64} hierarchy (Option C) and documented Option A/B; it did not build Option A itself.

## Typed configuration

Every module takes a Python `@dataclass` as its primary input — `SimulationConfig` (and
its nested `EquityConfig`, `RatesConfig`, `ZeroCurveConfig`) for the simulation module,
`SwapConfig` and `SwaptionConfig` for the instrument pricers, `PortfolioRequest` for the
top-level entry point (see [The Public API](#the-public-api)). This was a deliberate choice over passing plain dictionaries: a typo in a
dictionary key silently produces a confusing error deep inside the pipeline, while a
typo in a dataclass field name fails immediately, at the point the config object is
constructed, with a clear Python error. It's also the shape the HTTP API's own Pydantic
schemas (`engine/api/schemas.py`) mirror field-for-field and convert to/from at the HTTP
boundary — see [HTTP API](../reference/http-api.md).

## Testing philosophy

Every non-trivial formula in this codebase is tested two ways:

1. **Direct correctness checks** — e.g. "does the Brownian bridge matrix produce the
   exact covariance structure real Brownian motion should have?"
2. **Cross-checks against ORE itself** — the installed `ORE` Python package is used
   inside the test suite to build the same trade/curve/statistic using ORE's own code,
   and the two answers are compared numerically (typically to within `1e-6` relative
   tolerance, or exactly for things like VaR/ES where the formula involves no
   floating-point-sensitive steps like matrix decompositions).

See the [User Guide](../getting-started/user-guide.md#running-the-tests) for how to run these, and each
deep-dive doc's "Tested by" section for what's covered where.
