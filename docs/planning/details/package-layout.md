# Package layout

An evaluation of `engine/`'s packages and names (2026-10-08, at the owner's request), and the
layout proposed in their place. It closes [I-92](../known-issues.md#i-92); the renames and
moves are one roadmap step, and the two merges of §3 are steps of their own, because they
change behaviour and the renames do not.

**Rule.** `engine/` holds `__init__.py` and subpackages only. A package is named for what it
holds, in a word or two, and no module is named after a standard-library module.

## 1. What is confusing today

**"Market" means five things.** `engine.market` is today's market data; `ScenarioMarket` the
simulated market on every path; `engine.portfolio.market_path` the pipeline (named when there
were two pipelines, and "the market path" was one of them); `engine.api.market_schemas` the
HTTP requests (named after that pipeline too); `engine.market_risk` VaR and ES. Adding an
`engine/market/` package would have made a sixth.

**Packages named for a layer, not for what they hold.**

| Package | Holds | Why the name misleads |
|---|---|---|
| `api/` | The FastAPI app and routes, the request and result schemas, the TraderX routes, the job queue, the engine worker and its supervisor | "API" does not say HTTP, and half the package is the execution layer (queue and worker), which the HTTP layer only talks to |
| `integration/` | TraderX's bundle in, a result document out, with its own bill and note pricers, attempt store and publication | "Integration" with what? The docs call it "the EOD boundary", which reads as if the rest of the engine were not end-of-day; all of it is. It is the TraderX path |
| `portfolio/` | The run: `price_portfolio`, `PortfolioRequest`, `RunConfig`, the pipeline, the profiler's phase labels | It holds no portfolio; it runs one. Roadmap's run request (I-87) belongs in the same place |
| `risk/` | Exposure profiles, AD Greeks, ORE's bump sensitivities, the price functions both use, VaR/ES estimators | Every package computes risk. It holds three unrelated things, and VaR/ES sits apart from `market_risk/` |
| `instruments/` | The trade dataclasses, and ORE's Bermudan grid engine (`NumericLgmMultiLegOptionEngine`, in `bermudan_swaption.py` beside the trade, 726 lines) | The other pricing engines are in `valuation/`, where nobody looks for this one |
| `models/` | The LGM, the Hull-White bond-option formula, yield-curve primitives (`curves.py`), and ORE trade builders, schedules and fixings (`ore_builders.py`) | Two of its four modules are not models |

**Smaller things.** `engine/market.py` and `engine/day_count.py` at the root (I-92).
`simulation/random.py` shares a name with the standard library's `random`.
`portfolio/validation.py` is a re-export shim "for existing callers". `valuation/portfolio.py`
is ORE's valuation engine (`buildCube`), in a package beside another called `portfolio/`.

**Two implementations of one thing** (§3).

- *Treasury bills and notes* are priced twice: by the engine (`instruments/treasury.py`,
  through `valuation.portfolio.bond_legs`, on any curve, today and on every path) and by the
  TraderX path (`integration/bill.py`, `integration/note.py`, closed form on a flat assumed
  rate). Two pricers of one product can disagree, and every fix is made twice.
- *The LGM calibration* exists twice: ORE's bootstrap in `calibration/ore_lgm.py`, which every
  run uses, and the older standalone `calibration/lgm.py` with `basket.py` (Bachelier helper
  prices, its own solver steps), reached only by `POST /calibration/lgm`, which the run
  request removes.

Kept as they are: `precision/`, `numerics/`, `calibration/` (once §3's merge is made),
`simulation/`, `valuation/` and `market_risk/` are each one subject under a plain name. Three
modules called `config.py` (`RunConfig`, `CamConfig`, `PricingConfig`) are fine: each sits in
the package whose configuration it is, as ORE's `simulation.xml` and `pricingengine.xml` do.

## 2. Proposed layout

```
engine/
  __init__.py      x64 only (A-22); no other code
  marketdata/      today's market
  trades/          what a trade is
  models/          the models
  calibration/     fitting them to the market
  simulation/      the paths
  valuation/       pricing engines and the valuation engine
  exposure/        exposure profiles (later CVA/DVA, Basel's IMM figures)
  sensitivities/   Greeks: ORE's bump-and-revalue and AD
  market_risk/     VaR and ES by revaluation
  precision/       the precision policy, storage, report
  numerics/        root solver, VaR/ES estimators
  run/             the run request, its configuration, the pipeline
  jobs/            the job queue, the engine worker, its supervisor
  server/          HTTP: the app, routes, schemas, artifacts
  traderx/         TraderX's bundle in, its result document out
  regulatory/      Basel III, when it lands (details/basel-iii.md §5.1)
```

| New | Takes | From |
|---|---|---|
| `marketdata/` | `Market`, `CurrencyMarket`, `ZeroCurveConfig`, the swaption volatility matrix (`market.py`); the curve primitives (`curves.py`); accrual day counts and the time axis's ACT/365 (`daycounts.py`) | `engine/market.py`, `models/curves.py`, `engine/day_count.py`, `TIME_AXIS_DAY_COUNTER` from `models/ore_builders.py` |
| `trades/` | The trade dataclasses (`swap.py`, `european.py`, `bermudan.py`, `treasury.py`, `validation.py`); ORE's schedules, fixings and trade builders (`schedules.py`) | `instruments/` without the grid engine, `models/ore_builders.py` |
| `models/` | `lgm.py`, `hull_white.py` | `models/` without `curves.py` and `ore_builders.py` |
| `calibration/` | `lgm.py` (ORE's bootstrap), `cam.py` | `calibration/ore_lgm.py` renamed, once §3's merge retires the standalone `lgm.py` and `basket.py` |
| `simulation/` | `cam.py`, `scenario_market.py`, `sobol.py`, `sharding.py`, `config.py` | `simulation/random.py` renamed |
| `valuation/` | The engines per product, with ORE's LGM grid engine as `lgm_grid.py`; the valuation engine as `cube.py`; `config.py`, `context.py`, `options.py`, `legs.py` | `instruments/bermudan_swaption.py`'s engine, `valuation/portfolio.py` renamed |
| `exposure/` | `profiles.py` | `risk/exposure.py` |
| `sensitivities/` | `bump.py` (ORE's), `ad.py`, `price_functions.py` | `risk/sensitivities.py`, `risk/greeks.py`, `risk/price_functions.py` |
| `numerics/` | `roots.py`, `var_es.py` | `risk/var_es.py` (JAX only; exposure and market risk both use it) |
| `run/` | `request.py`, `config.py`, `pipeline.py`, `trace.py`; the run request and its Python entry `run` (I-87) | `portfolio/`, `market_path.py` and `profiling.py` renamed, `validation.py` deleted |
| `jobs/` | `store.py`, `worker.py`, `supervisor.py` | `api/job_queue.py` (not `queue.py`: the standard library has one), `api/worker.py`, `api/supervisor.py` |
| `server/` | `app.py`, `routes.py`, `requests.py`, `results.py`, `artifacts.py`, `traderx_routes.py` | `api/` without the job layer; `market_schemas.py` and `schemas.py` renamed by what they hold, `eod_routes.py` by whose routes they are |
| `traderx/` | Everything in `integration/` | `integration/` |

Every package and every module name is one word or two. "Market" is left in two names,
`marketdata` and `market_risk`, both standard terms for one thing each. The words "EOD
boundary" and "the market path" leave the code and docs with their packages: the first
becomes "the TraderX path", the second "the pipeline". The HTTP route prefix `/eod` stays,
being part of TraderX's contract.

**Open at the step.** Whether the cross-asset model's formulas in `simulation/cam.py` (507
lines: the model and its exact discretization together) move to `models/cam.py`, leaving the
path generation in `simulation/`; it depends on how cleanly they separate.

**How.** One step, all renames and moves, with `git mv` so history follows each file; no
code changes but names and imports, so float64 stays bit for bit and the suite's count is
unchanged. The layering tests (`tests/test_import_layering.py`,
`tests/test_integration_pipeline.py`'s import rules) are restated on the new names, and a test
fails on any module at the root of `engine/` but `__init__.py`. `docs/reference/`, the demos
and the planning documents follow in the same change. The `marketdata/` package imports no
JAX at its root (curves are imported as `engine.marketdata.curves`), so the TraderX path,
which may not import JAX or the models, can still import its day counts.

## 3. Merges

Each changes what runs, so each is a step of its own, after the renames.

- **One Treasury pricer.** The TraderX path builds a run request from its bundle (its
  convention checks and refusals first, unchanged, so a refused booking still never reaches
  a pricer, I-05) and the engine prices it; `traderx/bill.py` and `traderx/note.py` go. What
  only the TraderX path does stays there: the accrued-interest reconciliation against the
  exporter's figure, the result document, the attempt store. The engine's bond and the
  TraderX note must first agree on every bundle row of the tests, on the assumed profile's
  flat curve, to rounding. Part of the API goal: one way into the engine.
- **One LGM calibration.** `calibration/lgm.py` and `basket.py` retire once the run request's
  `calibration` analytic returns what `POST /calibration/lgm` returns, or move to
  `tests/support/` as an independent check of the bootstrap if their tests are worth keeping.
  Decision A-1 (nothing that works is removed without a replacement) is met by the analytic.
