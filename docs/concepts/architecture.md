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

## One pricing pipeline

`price_portfolio` takes today's `Market`, the trades, and the run configuration, `RunConfig`
(`engine/portfolio/config.py`): the simulation and its model per currency, the engine per
product, the Greeks method and settings, the precision per stage, the reporting currency,
with ORE's defaults ([The Portfolio Entry Point](../reference/portfolio-entrypoint.md#runconfig)).
It reproduces ORE's classic pipeline: trades name their currency and index, the market
supplies curves and volatilities, the configuration supplies the models and engines.

```
   Market + CamConfig ──calibrate (engine.calibration.cam)──► CrossAssetModel
        (per currency: LgmConfig, or HullWhiteConfig, ORE's <LGM> in either parametrization)
        ──simulate (engine.simulation.config)──► ScenarioMarket
             [paths, dates] discount and index curves, LGM numeraire, FX/EQ spots
        ──value (engine.valuation.portfolio)──► t=0 NPVs [T], NPV cube [S, D, T]
             every trade with its t=0 engine on each path; FixingManager; OptionWrapper
        ──exposure (engine.risk.exposure)──► EPE, ENE, EE_B, EEE_B, EPE_B, EEPE_B, PFE, Basel
   Market ──Greeks (engine.risk.sensitivities: bump; engine.risk.greeks: AD)──► per trade
```

Every option runs with every other: the models differ only in the simulation, and the
engines and Greeks methods price whatever the simulation produced. What is not implemented
yet is refused before any work, naming the field (`engine.precision.policy`, `validate_trades`).

**History.** Until roadmap 1.3 the Hull-White model was a second pipeline, chosen by passing
a `SimulationConfig` instead of a `Market`, with its own simulation, pricers and Greeks and
the defects listed in the closed ledger ([I-42](../planning/known-issues.md#i-42) to
[I-47](../planning/known-issues.md#i-47)); it is now one model option of the configuration
([configurable engine](../planning/details/configurable-engine.md)). Owner decisions
(2026-09-30, [compliance/decisions.md](../../compliance/decisions.md)): new models,
instruments and methods are added as options, and none that works is removed.

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
│   ├── demo_precision.py                 Market-risk VaR/ES at three precisions against
│   │                                     Monte Carlo noise
│   ├── demo_components.py                One engine module at a time, one section each
│   └── demo_scenarios.py                 The shared demo/test market and simulation
│                                         (dataclasses and their HTTP JSON)
├── docs/                                 Organized by topic (you are here)
├── compliance/decisions.md             The owner's dated decisions on how the engine is
│                                         configured, and the differences from ORE
├── engine/
│   ├── __init__.py                       Enables jax_enable_x64 once, at import
│   ├── market.py                         Today's market: curves per currency and index,
│   │                                     the ATM normal swaption matrix, FX and equity spots
│   ├── day_count.py                      Accrual day-count vocabulary, and nothing else.
│   │                                     A leaf because both models/ore_builders.py and
│   │                                     integration/note.py need the table, and
│   │                                     integration/ may not import models/ (see I-05)
│   ├── portfolio/
│   │   ├── __init__.py                   The public surface: PortfolioRequest/Result,
│   │   │                                 price_portfolio, RunConfig and its parts
│   │   ├── config.py                     RunConfig: simulation, engines, Greeks, precision
│   │   ├── request.py                    PortfolioRequest/PortfolioResult/price_portfolio
│   │   │                                 (unique trade ids)
│   │   ├── market_path.py                The pipeline: calibrate the CAM, simulate, value,
│   │   │                                 exposure, Greeks; validate_request
│   │   ├── validation.py                 Re-exports the trade validators
│   │   ├── worker_pool.py                The process pool behind HTTP jobs, plus
│   │   │                                 the opt-in XProf profiler hook and its
│   │   │                                 silent-truncation guard (see profiling.md)
│   │   └── profiling.py                  phase() -- the TraceAnnotation/named_scope pair
│   │                                     that labels each pricing stage on a trace
│   ├── api/                              FastAPI HTTP boundary -- TWO separate contracts
│   │   ├── app.py                        FastAPI app factory, mounting both routers
│   │   ├── routes.py                     /health, /version, /portfolio/price (also served
│   │   │                                 as /v2/portfolio/price; async job pattern),
│   │   │                                 /calibration/lgm
│   │   ├── market_schemas.py             The portfolio request; refuses unknown fields
│   │   │                                 and the retired Hull-White shape
│   │   ├── schemas.py                    Shared Pydantic schemas (curves, precision,
│   │   │                                 results, jobs, the calibration route)
│   │   └── eod_routes.py                 W1.6.4 /eod/* -- the TraderX EOD contract. Plain
│   │                                     dicts under a published JSON Schema, NOT Pydantic
│   ├── integration/                      TraderX EOD boundary -- hash-verified bundle in,
│   │                                     identified result out: both Treasury shapes price,
│   │                                     everything else is REFUSED. Imports no simulation
│   │                                     pricer, no FastAPI, no Pydantic, no JAX
│   │                                     (one module per W-task; see eod-integration.md)
│   ├── precision/                        The precision of a run (details/precision.md):
│   │   ├── formats.py                    the format table, the only name -> dtype map
│   │   ├── policy.py                     Precision / StagePrecision and their validation
│   │   └── storage.py                    store / load, the only casts between stages
│   ├── simulation/
│   │   ├── cam.py                        ORE's CrossAssetModel: per currency the LGM in
│   │   │                                 Hagan's or the Hull-White parametrization,
│   │   │                                 FX/EQ Black-Scholes, exact step moments,
│   │   │                                 Cholesky, the jitted state evolution
│   │   ├── scenario_market.py            Model-implied scenario curves per path and date,
│   │   │                                 the LGM numeraire, FX/EQ spots
│   │   ├── config.py                     CamConfig/LgmConfig/HullWhiteConfig (ORE's
│   │   │                                 simulation.xml) and simulate(market, config)
│   │   └── random.py                     Sobol normals and the Brownian bridge
│   ├── models/
│   │   ├── curves.py                     ZeroCurve (linear zero, QuantLib's flat-forward
│   │   │                                 extrapolation) and DiscountCurve (log-linear,
│   │   │                                 batched): the curve primitives
│   │   ├── lgm.py                        LGM closed forms (piecewise-constant Sigma) in
│   │   │                                 both parametrizations: H, zeta, Hull-White zeta
│   │   ├── hull_white.py                 Bond options on the Hull-White model (the
│   │   │                                 Jamshidian engine's building block)
│   │   └── ore_builders.py               ORE VanillaSwap construction and the time axis
│   ├── calibration/
│   │   ├── ore_lgm.py                    ORE's LgmBuilder: SwaptionHelper baskets and the
│   │   │                                 bootstrap, batched over path curves
│   │   ├── cam.py                        The CAM's IR calibration to CalibrationSwaptions
│   │   │                                 (either parametrization)
│   │   ├── basket.py, lgm.py             The standalone /calibration/lgm route's basket
│   │   │                                 and Hagan bootstrap
│   ├── instruments/                      The trade configs (what ORE's trade XML holds):
│   │   ├── _validation.py                id, date and field validators every config shares
│   │   ├── swap.py, european_swaption.py,
│   │   │   american_swaption.py, treasury.py
│   │   └── bermudan_swaption.py          The config, and the numeric LGM backward-
│   │                                     induction engine (ORE's grid engine) every
│   │                                     Bermudan/American is priced by
│   ├── valuation/                        ORE's ValuationEngine: every trade with its t=0
│   │   │                                 engine, today and on every path
│   │   ├── legs.py                       Swap legs: DiscountingSwapEngine, FixingManager,
│   │   │                                 paid flows, on any date and every path
│   │   ├── european.py                   BlackMultiLegOptionEngine (Bachelier), cash by
│   │   │                                 ParYieldCurve, the vol surface seen from a date
│   │   ├── jamshidian.py                 QuantLib's JamshidianSwaptionEngine on a
│   │   │                                 configured Hull-White model
│   │   ├── bermudan.py                   Each Bermudan/American on its own calibrated LGM,
│   │   │                                 recalibrated per path
│   │   ├── options.py                    OptionWrapper's exercise, physical and cash
│   │   ├── portfolio.py                  value_portfolio / value_today / value_on,
│   │   │                                 validate_trades, bond legs
│   │   ├── context.py                    PricingContext: one date's curves, vols, fixings
│   │   └── config.py                     PricingConfig (engine per product), the LGM
│   │                                     engine settings, the Jamshidian model
│   ├── market_risk/                      Short-horizon VaR/ES by full revaluation at t=0:
│   │   ├── factors.py                    RateRiskFactors -- the market's curve pillars
│   │   ├── scenarios.py                  Monte Carlo and historical shock scenarios
│   │   ├── revaluation.py                Every trade repriced under every scenario
│   │   └── run.py                        MarketRiskRequest/Result, run_market_risk
│   └── risk/
│       ├── var_es.py                     VaR / Expected Shortfall statistics (ORE's
│       │                                 RiskStatistics conventions)
│       ├── exposure.py                   EPE/ENE/EE_B/EEE_B/PFE over the simulated cube
│       │                                 (ORE's ExposureCalculator definitions)
│       ├── price_functions.py            Each trade's t=0 price as a JAX function of its
│       │                                 market curves -- shared by AD Greeks and market risk
│       ├── sensitivities.py              ORE's bump-and-revalue sensitivities, and Theta
│       └── greeks.py                     AD Delta / Gamma / Vega (Theta shared)
└── tests/
    ├── conftest.py                       Shared pytest fixtures
    ├── support/                          Shared helpers: the shared portfolio and its ORE
    │                                     references (portfolio.py), ORE's LGM engine via
    │                                     OREApp (ore_lgm_oracle.py), the grid engine with
    │                                     an explicit model (lgm_engine.py), Greeks helpers
    ├── test_cam.py, test_curves.py,       Each piece of the pipeline against ORE/QuantLib,
    │   test_random.py, test_valuation.py, the path tests under both models
    │   test_ore_lgm_calibration.py,
    │   test_sensitivities.py, test_jamshidian.py
    ├── test_hull_white_model.py          The Hull-White model: calibration, and the
    │                                     regressions of the closed Hull-White defects
    ├── test_end_to_end.py                The Hull-White simulation's paths priced by
    │                                     QuantLib/ORE, NPV and VaR/ES compared
    ├── test_swap.py, test_*_swaption.py,  The trade configs and the grid engine
    │   test_treasury_instrument.py,
    │   test_trade_configs.py, test_trade_dates.py
    ├── test_ore_*.py                     The grid engine and calibration against ORE
    ├── test_portfolio_*.py,              price_portfolio end to end: wiring, gap-fix
    │   test_run_config.py,               regressions, the run configuration, scale,
    │   test_diverse_portfolio_e2e.py,    breadth, the shared portfolio vs ORE
    │   test_shared_portfolio.py
    ├── test_greeks*.py                   AD Greeks against finite differences and the
    │                                     bump method
    ├── test_market_risk*.py, test_var_es*.py, test_exposure.py
    ├── test_api*.py, test_worker_pool.py, test_profiling_and_jit.py
    ├── test_import_layering.py           No package imports a layer above it; no demo
    │                                     or test code in engine/ (I-65)
    ├── test_demos.py, test_demo_scenarios.py
    ├── test_integration_*.py             engine/integration/, one file per task, run
    │                                     against the delivered TraderX fixtures
    └── fixtures/traderx-eod/             Real TraderX YU18 bundles (bill/note/sofr/equity,
                                          each v1+v2), hash-pinned. LF bytes committed and
                                          held that way by .gitattributes -- CRLF translation
                                          breaks every hash (see eod-integration.md)
```

Every `engine/` subpackage has an `__init__.py`, so the whole thing is importable as
`engine.portfolio`, `engine.valuation.portfolio`, `engine.risk.var_es` and so on from the
repository root — no path hacks required in application code or tests.

`bermudan_swaption.py` holds the backward-induction engine (state grid, Hagan's quadrature,
numeraire-deflated rollback) every Bermudan and American is priced by;
`AmericanSwaptionConfig` supplies ORE's American option times and exercise style. This
mirrors ORE's own design, where both exercise types run through the same numeric engine
(`QuantExt::NumericLgmMultiLegOptionEngine`) and differ only in their option times and in
which coupons an exercise enters (see
[American & Bermudan Swaptions](../instruments/american-bermudan-swaptions.md)).

## The shared foundation layer: `engine/models/`

`engine/models/lgm.py` is the single source of truth for the interest-rate model's
closed forms, in both of ORE's parametrizations: Hagan's (the LGM's own volatility α) and the
Hull-White adaptor's (α(t) = σ(t)e^{at}, with ζ and H to match), so the simulation, the
scenario curves, the numeraire and the calibration serve both models with one formula each.
`engine/models/curves.py` holds the curve primitives every stage reads, and
`engine/models/ore_builders.py` ORE trade building and the time axis. See
[Models & Trades](../reference/models-and-trades.md).

## `engine/calibration/`: fitting the volatility to market swaption quotes

A real trading desk calibrates a model's volatility to reproduce the market-quoted prices of
simpler, liquid options before pricing a more complex trade off it. `engine/calibration/`
does that step as ORE's `LgmBuilder` does: per currency for the cross-asset model
(`cam.py`, the CAM's `CalibrationSwaptions`, for the LGM or the Hull-White model) and per
Bermudan/American for its own engine (`ore_lgm.py`, a co-terminal basket built from the
trade, recalibrated on every path). See [Calibration](../reference/calibration.md).

Bermudan/American Vega exists because of it: "how much does this Bermudan's value change if
a market quote moves" is only a well-posed question once there is a market-quote-to-model
relationship to differentiate through (the AD method differentiates through the bootstrap;
the bump method recalibrates).

## The stages

Full field-level detail on every input/output is in the
[API Reference](../reference/api-reference.md); this section is about *why* the pieces are
shaped the way they are.

### Simulation (`engine/simulation/`)

**Input:** the `Market` and a `CamConfig` (dates, the model per currency, FX/equity
volatilities, correlations, samples, the simulation-market tenors). **Output:** a
`ScenarioMarket`: per path and date, each currency's discount curve and each index's
forwarding curve at the simulation tenors (ORE's `ScenarioSimMarket`), the LGM numeraire,
FX and equity spots. Exact discretization of ORE's cross-asset model, Sobol normals with a
Brownian bridge. See [Market Simulation](market-simulation.md).

### Valuation (`engine/valuation/`)

**Input:** the trades, the market and a `ScenarioMarket`, and the `PricingConfig`.
**Output:** every trade's value today and its `[Scenarios, Dates]` column of the cube, in the
reporting currency. Each trade is priced on each path by its t=0 engine, as ORE's
`ValuationEngine` does: swaps by discounting their legs with path fixings and paid flows
dropped (`legs.py`); Europeans by Bachelier on the market volatility seen from the date
(`european.py`) or Jamshidian on a configured Hull-White model (`jamshidian.py`);
Bermudans/Americans by the grid engine on their own basket, recalibrated per path
(`bermudan.py`); bonds by discounting their remaining flows. After an exercise, ORE's
`OptionWrapper` carries the swap entered (physical) or nothing (cash) (`options.py`).
ORE's own trade schedules come from `engine/models/ore_builders.py`, so "when does this swap
pay cash, and how much" is computed as a trading desk's software computes it.

### Exposure (`engine/risk/exposure.py`)

**Input:** the cube, the numeraire paths and today's discount factors.
**Output:** ORE's exposure profile per date — EPE, ENE, EE_B, EEE_B, PFE — for the
netting set and for each trade. See [Exposure](../risk/exposure.md).

### Greeks (`engine/risk/sensitivities.py`, `engine/risk/greeks.py`)

**Bump** (the default, ORE's sensitivity analysis): every trade repriced on today's
sensitivity market with one curve tenor or volatility quote shifted, and Theta on the
market rolled one day. **AD** (`GreeksConfig.method="AD"`): the same keys by automatic
differentiation of each trade's price as a function of the market curves' pillar rates
(`engine/risk/price_functions.py`), a Bermudan's Vega through its calibration by the
implicit function theorem; Theta shared with the bump method. See
[Greeks](../risk/greeks.md).

### Market Risk (`engine/market_risk/`)

**Input:** trades, the `Market`, `ShockScenarios` (absolute moves of every curve pillar over
a short horizon), the `PricingConfig`. **Output:** the P&L of every trade under every
scenario, and its VaR/ES. It does not use the simulated cube. See
[Market Risk](../risk/market-risk.md).

```
   ShockScenarios     ┌─────────────────────────┐   [S, N] P&L   ┌──────────────────┐
   (Monte Carlo or ─► │ engine/market_risk/      │ ─────────────► │ engine/risk/      │ VaR_99,
   historical)        │ revaluation.py: every    │                │ var_es.py         │ ES_97.5, ...
   trades, Market ──► │ trade repriced at t=0    │                │ compute_risk_     │
                      │ under every shock        │                │ metrics()         │
                     └─────────────────────────┘                └──────────────────┘
```

### Risk Statistics (`engine/risk/var_es.py`)

**Input:** any cube shaped `[Scenarios, TimeSteps, Trades]` plus a baseline value. The
market-risk path passes its P&L sample as a one-date cube. **Output:** Value at Risk and
Expected Shortfall, one per requested confidence level, one per time step. It does not care
where the cube came from (`tests/test_var_es.py::TestRobustAcrossInstrumentSources` feeds it
a synthetic cube and a portfolio run's). See [Risk Statistics](../risk/var_es.md).

## The Public API

**Input:** a `PortfolioRequest` (`engine/portfolio/request.py`): today's `Market`, a
heterogeneous list of trades (each with a unique `trade_id`), the `RunConfig`, the PFE
quantiles, and whether to compute Greeks and scenario risk.
**Output:** a `PortfolioResult`: today's values per trade and in total, the cube, the exposure
profiles, the Greeks, the trade ids, and the risk measure label.

`engine/portfolio/request.py::price_portfolio` is the single entry point that ties every
stage together. Before any JAX work it refuses what it cannot price as specified (a trade
valued on another date than the market's, a curve the market lacks, an engine's refusal, a
precision the pipeline does not implement yet), naming the trade and the field. See
[The Portfolio Entry Point](../reference/portfolio-entrypoint.md).

`engine/api/` wraps `price_portfolio` behind a FastAPI HTTP API — a thin transport layer,
not a second place orchestration logic lives. The request schema
(`engine/api/market_schemas.py`) mirrors the market, trades and run configuration
field for field, converting to the dataclasses at the HTTP boundary; `engine.portfolio` and
everything below it has no Pydantic/FastAPI dependency, so the engine does not require the
`api` extra as a plain Python library. See [HTTP API](../reference/http-api.md), including
why `POST /portfolio/price` returns a job id and polls rather than blocking.

## Design principle: modules agree on shapes, not code

The stages agree on data, not on each other's internals: the scenario market is curves per
path and date, and every pricer reads only those curves (`DiscountCurve`, log discount
factors at tenor times), never the model that produced them. That is why the Hull-White
model needed no pricer of its own: the same pricers, unchanged, price its paths, and the
per-path ORE comparisons of `tests/test_valuation.py` run under both models.
`compute_risk_metrics()` likewise needs only *some* array shaped `[Scenarios, TimeSteps,
Trades]`.

## `demos/demo_scenarios.py`: shared example configurations

Every demo and many test files need *some* realistic market and simulation to run against.
`demos/demo_scenarios.py` centralizes them: `demo_market(currencies)` (USD and EUR on curves
rising from 3% to 5% and from 2% to 3.2%, with 6M forwarding curves and swaption
volatilities, the EURUSD spot and an equity) and `demo_simulation(model, ...)` (the
cross-asset simulation on it, with `model` `"HullWhite"` or `"LGM"` for every currency,
optionally calibrated), plus their HTTP JSON (`demo_market_json`, `demo_simulation_json`).
The market is sloped on purpose: a flat curve hides drift and convexity errors, which is how
the Hull-White model's old simulation hid a 4-9% bias (I-42).

The engine never imports it. The demos, run as scripts from `demos/`, import it as
`demo_scenarios`; the tests as `demos.demo_scenarios`. `tests/test_import_layering.py` keeps
`__main__` blocks and imports of `demos`/`tests` out of `engine/`
([I-65](../planning/known-issues.md#i-65)).

## ORE as a dependency

[ORE (Open Source Risk Engine)](https://www.opensourcerisk.org/) shows up in this codebase
in two different roles, and it's important to keep them distinct:

1. **As a design reference.** Every formula in this engine was checked against ORE's actual
   behavior — either by reading ORE's own C++ source (`reference/ORE`) or Python bindings
   directly, or by running the installed `ORE` package in the test suite and comparing
   numbers. This is *validation*, not a runtime dependency.
2. **As a runtime dependency, for schedules and day counts.** Trade configs, the valuation
   layer and the calibration build ORE's own objects (`ORE.VanillaSwap` schedules, day
   counters, calendars, `SwaptionHelper` baskets) through `engine/models/ore_builders.py` and
   `engine/calibration/ore_lgm.py`. This runs once per trade or basket, not once per path:
   schedule logic is fiddly, well tested in ORE, and not performance-critical. The
   simulation (`engine/simulation/`) and the risk statistics (`engine/risk/var_es.py`) are
   pure JAX/NumPy.

Pricing any real trade therefore needs ORE installed, as does whatever machine serves the
pricing endpoint.

## Adjustable precision

**Decisions (2026-10-01,** [compliance/decisions.md](../../compliance/decisions.md) **A-9 to
A-16, D-9).** Which precision each calculation needs is what the project studies, so precision
is part of the run configuration: any combination may be run, the default is float64
everywhere and carries ORE parity, and a combination not yet shown adequate for a figure is
to carry a warning ([I-55](../planning/known-issues.md#i-55), roadmap 2.7). The full design,
down to FP8 storage, is [details/precision.md](../planning/details/precision.md).

**The policy.** `engine.precision.Precision` (on `RunConfig.precision` and
`MarketRiskRequest.precision`) gives each adjustable stage a `StagePrecision` of three format
names:

```python
Precision(
    simulation=StagePrecision(storage, compute, accumulate),  # Sobol shocks, model states
    market=StagePrecision(...),                               # scenario curves, numeraire, FX
    pricing=StagePrecision(...),                              # path pricing and the cube it stores
    by_product={"bermudan_swaption": StagePrecision(...)},    # overrides pricing for a product
    by_trade={"swap-7": StagePrecision(...)},                 # overrides both for one trade
    rounding="nearest",                                       # or "stochastic", into a scaled format
    rounding_seed=0,                                          # the stochastic rounding's seed
    paired_fraction=0.0,                                      # share of paths also run at float64
)
Precision()                       # float64 everywhere (the default)
Precision.throughout("float32")   # every stage stored, computed and accumulated in float32
```

The pricing stage is per trade (decision A-15, roadmap 1.5): `Precision.precision_for(trade)`
is the one lookup, `by_trade[trade_id]`, else `by_product[product]`, else `pricing`, and both
the portfolio and the market-risk pipeline call it. A product is the name each trade config
carries (`SwapConfig.product == "swap"`, and so on: the HTTP `trade_type`s). An override that
names no trade of the run or no product is refused before any work. The simulation and the
scenario market are shared by every trade, so they have no overrides.

`storage` is the format a stage's output is kept in until the next stage reads it; `compute`
the format its arithmetic runs in; `accumulate` the format its sums accumulate in. The format
names come from one table (`engine/precision/formats.py`), which validation, the HTTP schema
and storage all read. `storage` is any format of the table no wider than `compute` (`float64`,
`float32`, and since roadmap 1.6 `float16`, `bfloat16`, `float8_e4m3fn`, `float8_e5m2`);
`compute` is `float64` or `float32` and `accumulate` equals it until roadmap 2.8, and using
one earlier is refused naming the step. Calibration, t=0 values, Greeks and every reduction
over paths (exposure, VaR/ES) are float64 by decision (A-10).

**The cast points.** Only these read the policy; everything between them follows the dtype
of its inputs (kernels cast their own constant inputs, coupon tables, volatilities and
calibration baskets, to the dtype of the curves they are given):

| Stage boundary | Where | What happens |
|---|---|---|
| shocks, states | `engine.simulation.config.simulate` | normals generated and bridged at `simulation.compute`, stored; states evolved at `simulation.compute` from the loaded shocks, stored |
| market | `simulate` | the scenario market built at `market.compute` from the loaded states, returned stored at `market.storage`; its tenor grid (no scenario axis) stays at `market.compute` |
| values | `engine.valuation.portfolio.value_portfolio` | per trade, at `precision_for(trade)`: the market loaded at its `compute` (once per dtype), the trade priced, its cube column stored at its `storage` |
| reductions | `engine.portfolio.market_path` | the cube and the numeraire loaded at float64, then exposure; `PortfolioResult.npv_cube` is the float64-loaded cube, so columns stored in different formats never meet in arithmetic |
| market risk | `engine.market_risk.run_market_risk` | shifts rounded to `simulation.compute` and stored; per trade, revaluation and P&L at its `precision_for(trade).compute`, the P&L stored at its `storage`; VaR/ES in float64 |

`store`/`load` (`engine/precision/storage.py`) are the only casts between stages, and the
pipeline stores through `Precision.store`, which adds the policy's rounding. For float64 and
float32 storing is a plain cast and storing at an array's own dtype returns it unchanged, so
the float64 default runs exactly the arithmetic it ran before the policy
existed.

**Storage below 32 bits** (roadmap 1.6). float16, bfloat16 and the two FP8 formats are stored
as a `Stored`: the values in the format and, per block of 32 consecutive scenarios (paths),
a float32 power-of-two scale that brings the block's largest magnitude to the format's
maximum, so FP8's 448 or float16's 65,504 never limits the range. A power of two scales
exactly; loading multiplies it back. Blocks run along the scenario axis, the axis multi-device
runs will shard. Values are rounded to the format's grid either to nearest or stochastically
(up or down in proportion to the distance, so the rounding has mean zero), with draws from
`Precision.rounding_seed` and the array's name in the run (`"shocks"`, `"values/<trade id>"`),
so a run reproduces and a trade's column rounds the same in a mixed run as alone. FP8 holds a
column in an eighth of float64's memory plus an eighth for the scales. Compute stays float32
or float64: `load` reads the stored values back at the next stage's compute dtype.
Continuous integration runs the fast tier with JAX's strict dtype promotion, under
which any accidental float32/float64 mix is an error.

**The paired sample and the report** (roadmap 1.7, decision A-13). With `paired_fraction > 0`
the first paths (whole blocks of 32) are simulated and priced again at float64 throughout,
`engine.portfolio.market_path._paired_sample` (market risk: the first scenarios revalued). A
scrambled Sobol sequence's first points do not depend on the sample size and every kernel is
per path, so these are the paths a float64 run gives, bit for bit. Means (EPE, ENE) become
two-level estimates, the run's mean corrected by the paired paths' mean float64 difference
(`engine/precision/estimate.py`), so a precision bias becomes variance; quantiles (PFE, VaR,
ES) are measured on the pair, not corrected. Every result carries a `PrecisionReport`
(`engine/precision/report.py`): the policy, each trade's stage, the format of every stored
array and the devices, read from the arrays in the process that ran the job, and each
figure's estimate.

**The x64 flag.** JAX can create 64-bit arrays only while one process-global setting,
`jax_enable_x64`, is on; it cannot be scoped per thread or per call. `engine` turns it on once,
when imported, and nothing turns it off: a float32 computation is float32 because its arrays
are float32, not because the flag is off.

### Concurrency

`price_portfolio` may run on several threads at once: every precision is a dtype of the
run's own arrays, and the pipeline keeps no module-level state and never reads ORE's global
evaluation date (each trade carries its own, I-64). Until roadmap 1.4 a lock serialized runs.

HTTP jobs run in `engine/portfolio/worker_pool.py`'s one `ProcessPoolExecutor`, whatever
their precision (until 1.4, one pool per simulation precision). Each worker runs one job at a
time with x64 on, as the parent does, so a job prices bit for bit as a direct `price_portfolio`
call (until 1.3 a float32 worker turned the flag off, [I-71](../planning/known-issues.md#i-71)).
Compiled programs are not shared across processes, so each worker pays its own compilation.
Workers are always spawned, never forked (forking a process that has initialized JAX hangs,
I-33). Roadmap 1.8 replaces the pool with one engine worker process per host
([I-72](../planning/known-issues.md#i-72)).

### Below float32

Measured on this project's CPU backend (`jax==0.10.2`): `jnp.linalg.cholesky` and
`jax.scipy.stats.norm.ppf` have no kernel below float32, and computing the inverse normal CDF
in a low format overflows. Storage below float32 needs neither, since values are loaded to a
float32-or-wider compute dtype; compute below float32 needs the heavy kernels rewritten in
difference form with explicit accumulators. The measurements and the plan are in
[details/precision.md §8 and §15](../planning/details/precision.md#8-compute-below-float32).

## Typed configuration

Every module takes a Python `@dataclass` as its primary input — `Market` (and its
`CurrencyMarket`, `ZeroCurveConfig`, `SwaptionVolSurface`) for today's market, `CamConfig`
(with `LgmConfig`/`HullWhiteConfig`) for the simulation, `SwapConfig`, `SwaptionConfig`,
`BermudanSwaptionConfig`, `AmericanSwaptionConfig` and `BondConfig` for the trades,
`RunConfig` for the run, `PortfolioRequest` for the top-level entry point (see
[The Public API](#the-public-api)). This was a deliberate choice over passing plain
dictionaries: a typo in a
dictionary key silently produces a confusing error deep inside the pipeline, while a
typo in a dataclass field name fails immediately, at the point the config object is
constructed, with a clear Python error. It's also the shape the HTTP API's own Pydantic
schemas (`engine/api/market_schemas.py`) mirror field-for-field and convert to/from at the HTTP
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
