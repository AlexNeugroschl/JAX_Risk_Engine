# Package layout

The layout of `engine/`: an evaluation of its packages and names (2026-10-08, at the owner's
request), the layout chosen in their place (the owner's review of the same day), and its
implementation, also on 2026-10-08, which closed [I-92](../known-issues.md#i-92). The renames
and moves were one roadmap step; the two merges of §3 are steps of their own, because they
change behaviour and the renames did not.

**Rule.** `engine/` holds `__init__.py` and subpackages only. A package is named for what it
holds, in a word or two joined by an underscore; packages of one kind nest under a common
parent rather than repeat a word at the top level; no module is named after a
standard-library module. `tests/test_import_layering.py` enforces the structural half: no
module at the root of `engine/` or of `engine/risk/` but `__init__.py`, every package placed
in a layer, and no package importing its own layer or one above.

## 1. Why

Before the change, two modules sat at the root of `engine/` (`market.py`, `day_count.py`), and
packages were named for a layer or too generally: `integration/` was the TraderX path,
`portfolio/` the run (it held no portfolio), `valuation/` the pricing engines, `numerics/` one
root solver, `simulation/` the cross-asset model's paths, and `risk/` (exposure, two Greeks
methods, the VaR/ES estimators) sat beside `market_risk/` with nothing saying how they differed.
Things sat outside the package of their kind: ORE's Bermudan grid engine in
`instruments/bermudan_swaption.py` beside the trade, ORE's schedules and the curve primitives
in `models/`. "Market" named five things (today's market data, the simulated market, the
pipeline, the HTTP requests named after it, VaR and ES), `simulation/random.py` shared a name
with the standard library, and `api/`'s `schemas.py` held request schemas beside the results.
The full evaluation is in git (this file at commit `7d52e17`).

## 2. The layout

```
engine/
  __init__.py          x64 only (A-22); no other code
  precision/           the precision policy, storage, report
  solvers/             the root solver
  market_data/         today's market, curves, day counts
  models/              LGM, Hull-White, the cross-asset model
  instruments/         trade definitions, ORE schedules and fixings
  traderx/             TraderX's bundle in, its result document out
  calibration/         fitting the models to the market
  market_simulation/   the paths and the scenario market on them
  pricing/             pricing engines, the valuation cube
  risk/
    greeks/            ORE's bump-and-revalue and AD
    market/            VaR and ES by revaluation, and the VaR/ES estimators
    counterparty/      exposure profiles (later CVA/DVA, Basel's IMM figures)
  run/                 the run request, its configuration, the pipeline
  api/                 HTTP, the job queue, the engine worker
  regulatory/          Basel III, when it lands (details/basel-iii.md §5.1)
```

The packages are listed in their import layers, lowest first (`LAYERS` in
`tests/test_import_layering.py`): a package imports only those above it in this list, never
one of its own layer (`models`, `instruments` and `traderx` share one) or below. `risk/`'s
kinds are layered the same way: `greeks`, then `market` (it revalues with the Greeks' price
functions), then `counterparty` (it borrows the quantile labels of `risk/market/var_es.py`).
`regulatory/` will sit above `risk/`: its figures span market and counterparty risk and carry
their own profile, catalogue and manifest.

| Package | Holds | From |
|---|---|---|
| `market_data/` | `Market`, `CurrencyMarket`, `ZeroCurveConfig`, the swaption volatility matrix (`market.py`); the curve primitives (`curves.py`); the time axis's ACT/365 and `time_from_reference`, and the accrual day counts (`day_counts.py`) | `engine/market.py`, `models/curves.py`, `engine/day_count.py`, the time axis from `models/ore_builders.py` |
| `models/` | `lgm.py`, `hull_white.py`; ORE's cross-asset model and its exact discretization's step moments (`cam.py`) | `models/` without `curves.py` and `ore_builders.py`; `simulation/cam.py` without its path recursion |
| `instruments/` | The trade dataclasses; ORE's schedules, fixings and trade builders (`schedules.py`) | `instruments/` without the grid engine, `models/ore_builders.py` without the time axis |
| `traderx/` | Everything in `integration/` | `integration/` |
| `calibration/` | `ore_lgm.py` (ORE's bootstrap), `cam.py`, and until §3's merge the standalone `lgm.py` and `basket.py` | unchanged |
| `market_simulation/` | `config.py` (`simulate`), `sobol.py`, `paths.py` (the state recursion on every path), `scenario_market.py`, `sharding.py` | `simulation/`, with `random.py` renamed and `evolve_states` from `cam.py` |
| `pricing/` | The engines per product, with ORE's LGM grid engine as `lgm_grid.py`; the valuation engine as `cube.py`; `config.py`, `context.py`, `options.py`, `legs.py`, `bermudan.py`, `european.py`, `jamshidian.py` | `valuation/`, `instruments/bermudan_swaption.py`'s engine, `valuation/portfolio.py` renamed |
| `risk/greeks/` | `bump.py` (ORE's), `ad.py`, `price_functions.py` | `risk/sensitivities.py`, `risk/greeks.py`, `risk/price_functions.py` |
| `risk/market/` | `factors.py`, `scenarios.py`, `revaluation.py`, `run.py`, `var_es.py` | `market_risk/`, `risk/var_es.py` |
| `risk/counterparty/` | `exposure.py` | `risk/exposure.py` |
| `solvers/` | `roots.py` | `numerics/` |
| `run/` | `request.py`, `config.py`, `pipeline.py`, `trace.py`; the run request and its Python entry `run` (I-87) when they land | `portfolio/`, `market_path.py` and `profiling.py` renamed, `validation.py` deleted |
| `api/` | `app.py`, `routes.py`, `requests.py` (the requests and the wire forms they share: curves, coupon periods, the precision policy), `results.py` (the results, the precision report, job status), `artifacts.py`, `traderx_routes.py`, `job_queue.py`, `worker.py`, `supervisor.py` | `api/`: `market_schemas.py` as `requests.py`, `schemas.py` as `results.py`, `eod_routes.py` as `traderx_routes.py` |

"Market" is left in three names, `market_data`, `market_simulation` and `risk/market`, each a
standard term for one thing. The words "EOD boundary" and "the market path" left the code and
docs with their packages: the first is "the TraderX path", the second "the pipeline". The HTTP
route prefix `/eod` stays, being part of TraderX's contract, as do TraderX's own terms (its
end-of-day bundle).

### 2.1 Decided at the step

- **The cross-asset model is a model.** `simulation/cam.py` separated cleanly: everything but
  `evolve_states` is ORE's `CrossAssetModel` and `CrossAssetAnalytics` (components,
  integrals, the exact discretization's moments, the Cholesky factor), computed on the host;
  `evolve_states` is the 15-line device recursion over those moments. The model went to
  `models/cam.py`, the recursion to `market_simulation/paths.py`.
- **The time axis moved with the accrual day counts.** `TIME_AXIS_DAY_COUNTER` and
  `time_from_reference` sit in `market_data/day_counts.py`, named apart from the accrual table
  as before, so `instruments/` imports nothing but `market_data/` and the Bermudan trade no
  longer reaches the models at all.
- **`api/requests.py` and `api/results.py` hold what their names say.** The old `schemas.py`
  held the curve, coupon-period and precision wire forms, the date parsers and the standalone
  calibration route's request beside the results; they moved to `requests.py`, and
  `results.py` imports the precision schemas from it (a result echoes the policy it ran with),
  so the dependency runs from results to requests. Moving `PrecisionSchema` beside the
  requests' own `RETIRED_SHAPE` (the retired Hull-White request) needed precision's
  `RETIRED_SHAPE` imported under its own name, `RETIRED_PRECISION_SHAPE`.
- **Names used across packages are public.** The Bermudan's `_build_ore_swap` is
  `underlying_swap` (the pricing engine, the cube and the tests build the option's underlying
  with it) and the grid's `_grid_half_width` is `grid_half_width` (market risk sizes its
  batches by it).
- **Old paths are gone, not aliased.** The re-export shim `portfolio/validation.py`, the
  curve re-exports of `models/hull_white.py`, the accrual re-exports of `ore_builders.py` and
  the time axis's deprecated alias `DAY_COUNTER` were deleted with their callers updated; a
  test now fails if any module binds `DAY_COUNTER`
  (`tests/test_day_count_roles.py::TestOnlyTheAccrualRoleIsConfigurable::test_the_time_axis_has_one_name`).
- **One import runs against the layers.** The Greeks label each trade's region of a profiler
  trace with the run's phase names, so `risk/greeks/` imports `run/trace.py` at call time.
  `trace.py` imports no engine module (a test checks it), and the import is the only entry of
  the layering test's `EXCEPTIONS`. Moving the labels below `risk/` would split the pipeline's
  phase names from the pipeline; the exception is the smaller cost.
- **Tests and docs follow.** Test files named after a retired package or phrase were renamed
  (`test_integration_*` to `test_traderx_*`, `test_integration_eod_routes` to
  `test_traderx_routes`, `test_portfolio_market_path` to `test_pipeline`,
  `test_api_market_path` to `test_api_portfolio`, `test_random` to `test_sobol`,
  `test_valuation` to `test_pricing`), and `docs/reference/eod-integration.md` is
  `docs/reference/traderx-path.md`. The TraderX path's import rule is an allowlist now
  (`engine.market_data.day_counts` and its own package), so a renamed package cannot slip
  past it; it had been a list of banned names, which a rename would have emptied.

**How it was done.** `git mv` for every file, so history follows each; references rewritten
in one pass per separator (`engine.x.y`, `engine/x/y`), longest name first; the four splits
(`ore_builders.py`, `bermudan_swaption.py`, `cam.py`, the API schemas) by hand. No code changed
but names, imports and docstrings: the golden snapshot is bit for bit (359 of 359 arrays,
dtype and shape included), and every test of the suite maps to one after it, but for the six
replaced by stronger ones (I-92's ledger row).

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
  Decision A-1 (nothing that works is removed without a replacement) is met by the analytic;
  `calibration/ore_lgm.py` then becomes `calibration/lgm.py`.
