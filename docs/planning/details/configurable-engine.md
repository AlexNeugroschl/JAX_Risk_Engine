# Configurable engine

Design for roadmap [stage 1](../roadmap.md#stage-1--structure), the API reaching every setting (done
2026-10-07) and the run request (I-87): one run configuration whose options are
models, engines, methods and precision, as ORE configures a run. Implements owner decisions
A-1 to A-9 ([compliance/decisions.md](../../../compliance/decisions.md) §1–2).

## Principles

1. **ORE is the specification.** New code cites the ORE/QuantLib source it reproduces (file
   and function) in its docstring; "ORE" means the vendored `reference/ORE/`.
2. **Defaults are ORE's defaults.** Where ORE has no code default and its configuration must
   name a value, the default is what ORE's `Examples/` use, and the code says so.
3. **Options are added, not swapped.** Nothing that works is removed without a replacement;
   an option leaves the default when something better arrives, but stays available.
4. **One request shape for every configuration.** No API versions per model.
5. **Red first, on sloped curves.** Every fix is shown failing against the pre-fix code, on a
   curve rising 3% → 5% as well as flat; flat curves make drift and convexity terms cancel.

## Today

Every analytic is one run request (`engine.run.run`, since 2026-10-18), whose portfolio
analytics are one `RunConfig` (`engine/run/config.py`) on `PortfolioRequest.config` and one
pipeline (`engine.run.pipeline`):

| | |
|---|---|
| Entry | `run(RunRequest(market, analytics, portfolio, RunRequestConfig(...)))`; under it `price_portfolio(PortfolioRequest(market=Market(...), trades=[...], config=RunConfig(...)))`, `run_market_risk`, `calibrate_cam` |
| HTTP | `POST /runs`, polled at `GET /runs/{run_id}`; every setting has an API field and every field's value arrives in its setting (`tests/test_api_completeness.py`) |
| Simulation | `config.simulation` (`CamConfig`), `engine.models.cam`: per currency `LgmConfig` or `HullWhiteConfig`, exact step moments, LGM numeraire |
| Valuation | `engine.pricing`: every trade by its t=0 engine on every path (scenario market, legs, `OptionWrapper`, bond legs, per-trade basket) |
| European engine | `Bachelier` (ORE's default) or `Jamshidian` with `PricingConfig.jamshidian` |
| Greeks | `Bump` (`engine.risk.greeks.bump`, settings `config.greeks.sensitivity`) or `AD` (`engine.risk.greeks.ad`) |
| Market risk | `engine.risk.market.run_market_risk` on a `Market` with the same `PricingConfig` (A-8) |
| Precision | `config.precision` (`engine.precision.Precision`, the precision mechanism (2026-10-01)): storage, compute and accumulate per adjustable stage (simulation, market, pricing), float64 or float32 compute, storage down to FP8 with block scales and nearest or stochastic rounding since sub-32-bit storage (2026-10-02); per product and per trade since per-trade precision (2026-10-02); the paired float64 sample and the precision report on every result since the precision report (2026-10-02) ([precision.md](precision.md)) |

## The run configuration (I-68) — done 2026-09-30

| Component | Field | Options | Default |
|---|---|---|---|
| Model per currency | `simulation.ir[ccy]` (`CamConfig`, ORE's `CrossAssetModelData`) | `LgmConfig`; `HullWhiteConfig` (the shared pipeline (2026-10-01)) | none: named per currency |
| Simulation | `simulation` | Classic revaluation; AMC is [F-03](../features.md#f-03) | Classic |
| Engine per product | `pricing` (`PricingConfig`) | Swap: discounting. European: `Bachelier`, `Jamshidian`. Bermudan/American: `LgmSwaptionEngineConfig` (FD solver in F-01) | ORE's builder defaults |
| Greeks method | `greeks.method` | `Bump`; `AD` | `Bump` |
| Sensitivity settings | `greeks.sensitivity` (`SensitivityConfig`) | Tenors, shifts, Theta horizon, vol decay | ORE's |
| Precision per stage | `precision` (`Precision`, the precision mechanism (2026-10-01)) | Storage, compute and accumulate format per adjustable stage | FP64 |
| Swaption vol decay | `simulation.swaption_vol_decay` | `ForwardVariance`; `ConstantVariance` (A-4) | `ForwardVariance` |
| Reporting currency | `base_currency` | Any market currency; `None` is the simulation's, else USD | `None` |

Rules the implementation follows, which every change since keeps:

- **An option the pipeline does not implement is refused, never substituted.** The
  configuration's own validation (a precision format not enabled yet, `engine.precision`)
  and `validate_trades` run before any work and name the field (since 2026-10-01 one check for
  both models). An engine is checked where the run uses it (the Jamshidian engine's
  refusals only for a European on it).
- **One fact, one field.** The reporting currency is the simulation's; a `base_currency`
  contradicting it is refused (before 2026-09-30 it was silently ignored).
- **The request travels whole.** The engine worker parses the HTTP body exactly as the
  route did, so every configuration component reaches it and is validated again there
  (until 2026-10-04 a worker pool froze the request into a picklable form).

Evidence that the defaults reproduce the pipeline bit for bit: the shared portfolio
(8 trades, scenario risk, exposure, bump Greeks), an FP32-simulation run and the Hull-White
model, compared array for array against the code before the change (114 arrays, all
identical: [verification status](../known-issues.md#verification-status)), and the parity
suites in the full run.

## The Hull-White model on the shared pipeline — done 2026-10-01

The Hull-White model is a model per currency (`HullWhiteConfig` in `CamConfig.ir`, `"model":
"HullWhite"` over HTTP) on the same pipeline as the LGM; the separate Hull-White pipeline
(`SimulationConfig` market, `engine.simulation.market_model`, the `engine.instruments`
scenario pricers, `HULL_WHITE_CONFIG`) is removed. Trades carry no model, name themselves and
their date. What closed which issue:

| Change | Closed |
|---|---|
| The model is ORE's `<LGM>` with `ReversionType`/`VolatilityType` `HullWhite` (`IrLgm1fPiecewiseConstantHullWhiteAdaptor`): α(t) = σ(t)e^{at}, ζ(t) = ∫σ²e^{2as}ds, H(t) = (1 − e^{−at})/a, simulated exactly under the LGM measure by the CAM; its curves are its own bond prices, fitted to today's curve by construction | [I-42](../known-issues.md#i-42), [I-44](../known-issues.md#i-44) |
| The model's exact LGM numeraire | [I-45](../known-issues.md#i-45) |
| Every trade valued by `engine.pricing` on the paths: legs (paid flows drop out, path fixings), `OptionWrapper`, bond legs, per-trade LGM as ORE built it on the as-of date (I-49), vectorized on device | [I-04](../known-issues.md#i-04) (model half), [I-43](../known-issues.md#i-43), [I-24](../known-issues.md#i-24), [I-47](../known-issues.md#i-47), [I-62](../known-issues.md#i-62) |
| Bachelier on the market volatility is the default European engine for every model; Jamshidian is an option with its own Hull-White model | [I-46](../known-issues.md#i-46) |
| Trades name currency and index only; a model or curve field on a trade is refused | [I-63](../known-issues.md#i-63) |
| `evaluation_date` and `trade_id` required keyword fields on every trade config | [I-64](../known-issues.md#i-64), [I-10](../known-issues.md#i-10) (configs) |
| The model is a field of the configuration, not the market's type | [I-68](../known-issues.md#i-68) |

Design decisions taken in the step, with their reasons:

- **LGM-measure form, not the bank-account drift.** The plan named Brigo–Mercurio's
  r(t) = x(t) + α(t) with the curve-fitted drift. ORE does not simulate that: its Hull-White
  model is the LGM adaptor above, under the LGM measure with the LGM numeraire. The two are the
  same model (the short rate is r = f(0,t) + H′(t)z + ζ(t)H(t)H′(t); the path curve is
  QuantLib's `HullWhite::discountBond(t, T, r)` to 1e-12, `tests/test_cam.py`), so the CAM's
  exact step moments, numeraire and scenario market serve both models unchanged, and ORE
  parity holds by construction.
- **Calibration converts the Hagan bootstrap bucket by bucket.** A helper's price depends on
  the model only through ζ at its expiry and H, so the Hull-White σ is the one with the
  LGM's ζ at every bucket end (`engine.models.lgm.hull_white_matching_zeta`); exact, and the
  same calibration ORE's bootstrap of the adaptor reaches (`tests/test_hull_white_model.py::
  TestCalibration`).
- **Jamshidian's model in `PricingConfig.jamshidian`.** ORE has no Jamshidian builder, so the
  engine's Hull-White (a, σ) is configured with it (`JamshidianEngineConfig`, both positive,
  as QuantLib's `HullWhite` requires); required with `european="Jamshidian"`, refused
  without. The engine is QuantLib's decomposition in discount factors only, so it prices on
  any curve: today's, a bump, a path's.
- **`trade_id`, not "instrument id".** Named as ORE's `<Trade id>`; unique in a portfolio,
  echoed as `PortfolioResult.trade_ids`. Over HTTP, give it on every trade or on none (none
  numbers them `trade-0`, ...).
- **ORE globals.** The engine sets no ORE global (every date is passed explicitly; the
  calibration helpers take their dates from their own curve's reference date), so there is
  nothing to scope.
- **Precision narrowed until 2026-10-01.** The Hull-White pipeline took `pricing`/`risk` below 64;
  the shared pipeline computed those stages in float64 and refused less (I-55). Kept rather
  than ported: the precision mechanism made them adjustable for both models at once.

Evidence: the shared portfolio's pipeline numbers (t=0, an FP64 and an FP32-simulation
scenario run, bump Greeks) bit for bit before and after, 103 of 103 arrays; each closed
Hull-White defect measured on the code before the step and asserted after, on a 3% → 5%
curve; every per-path ORE comparison of `tests/test_pricing.py` run under both models;
`tests/test_end_to_end.py` prices the Hull-White simulation's paths in QuantLib
([verification status](../known-issues.md#verification-status)).

## Precision and the engine worker (I-55, I-12, I-72) — done 2026-10-01 to 10-04

Designed in [precision.md](precision.md): a `Precision` with storage, compute and accumulate
per adjustable stage, overridable per product and trade (A-10, A-15); the old configuration
refused (A-12); five cast points with inputs following dtype; storage down to FP8; the paired
float64 sample, the two-level estimator and the precision report (A-13); then one engine
worker process per host behind a durable job queue (A-14). Each change kept the default bit
for bit. Adjustable precision stays available throughout.

## Precision evidence and low-precision kernels (I-55, F-07)

In [precision.md](precision.md) §8 and §10: the evidence table per figure and precision
against the acceptance standard (A-11), shared with Basel P6, read off the storage harness
(§15.4, 2026-10-09, which closed I-75; built 2026-10-11, §10.4, which closed I-55); then the
kernels in difference form, one implementation for every precision (A-16).

<a id="one-run-request-i-87-i-88-i-89"></a>
## One run request (I-87, I-88, I-89, I-95) — done 2026-10-18

The API of 2026-10-07 made every setting reachable but kept one route per engine function, so
the API was shaped like the engine's Python entry points rather than like a user's task. The run
request does what A-2 says: a single route and a single request, as ORE runs a portfolio, market
data and its configuration files with a list of analytics.

**As built.** `engine/run/run_request.py`: `RunRequest(market, analytics, portfolio, config,
include_npv_cube)`, `RunRequestConfig` with its sections (`base_currency`, `models`
(`ModelsConfig`, simulation.xml's `CrossAssetModel`), `simulation` (`SimulationParameters`, its
`Parameters` and `Market`), `pricing`, `sensitivity` (`GreeksConfig`), `exposure`
(`ExposureConfig`), `market_risk` (`MarketRiskConfig`), `precision`), `run()`, `validate_run()`
and `RunResult`; `engine/run/identity.py` (the engine's identity); `engine/api/requests.py`'s
`RunRequestSchema` mirroring it; `engine/api/results.py`'s `RunResultSchema`; the routes
`POST /runs`, `GET /runs/{run_id}` (`?wait=`, at most 30 s), `POST /runs/{run_id}/cancel`,
`GET /runs/{run_id}/artifacts/...` and `GET /config/defaults` ([HTTP API](../../reference/http-api.md)).

- **What to compute is the `analytics` list; how is `config`.** The list replaced
  `scenario_risk` and `compute_greeks`. Each analytic names the sections it needs (`exposure`:
  `models` and `simulation`; `market_risk` its section; `calibration` `models` with a basket),
  and a section, a portfolio or a setting differing from its default that no requested analytic
  reads is refused, naming it (I-59's rule). Calibration alone reads only the calibrated
  currencies' models and the tolerance.
- **Each analytic is its entry point.** `npv`, `exposure` and `sensitivities` are one
  `price_portfolio` (today's values are in all three), `market_risk` is `run_market_risk`,
  `calibration` is `calibrate_cam`, each unchanged: a run of one analytic is its direct call bit
  for bit, a run of several each alone, compiling nothing they would not
  (`tests/test_run_request.py`). Sharing work between analytics in one run is not done:
  [I-118](../known-issues.md#i-118).
- **Each setting in one place.** The model per currency is `config.models`, which calibration
  and exposure both read; the reporting currency is `config.base_currency` (default `"USD"`),
  also the simulation's domestic currency (`RunRequestConfig.cam()` builds the `CamConfig`).
  The Greeks method and sensitivity.xml are one section, `sensitivity`, flattened on the wire.
- **Checked before any work, with no JAX work** (`validate_run`, which the route runs): the Monte
  Carlo scenarios are a spec drawn when the run starts (`engine.risk.market.MonteCarloScenarios`,
  which `monte_carlo_scenarios` now draws through, bit for bit), so the request is checked whole
  without drawing one; `engine.risk.market.run.validate_scenarios` reads only their factors,
  count and replicates.
- **Precision per analytic** (I-95): only `exposure` and `market_risk` read it (today's values,
  Greeks and calibration are float64, A-10). Market risk refuses a `market` stage and a
  `simulation.compute` other than its accumulate format, also beside `exposure`, so its
  precision report never states a policy it did not run; a run of both at another market-stage
  precision is two runs (A-27, confirmed by the owner).
- **One result** that says what each figure is without the request beside it: the
  configuration as run (the request's `config` as parsed, every default filled in), the as-of
  date and the simulation dates, the reporting currency, one row per trade (its `product`, `npv`,
  `exposure`, `greeks`), each Greek's factor labels, each analytic's pricing settings (its
  calibration policy: `model=fixed` in the exposure simulation, `recalibrated` under the shifts
  and scenarios), every array described alike inline or by reference (name, dtype, shape, axes,
  the hashed trade or option order), the exercise decisions on request, and the engine's identity
  (version, git revision and whether the tree was modified, jax, jaxlib and ORE versions) in place
  of the fixed `0.1.0`. `GET /runs/{run_id}` returns the run's status with the result in it, not
  the `{status, result, config}` the design sketched: the configuration as run is the result's.
- **Outputs** (decided in the step, as the design asked, and confirmed by the owner; A-17's revision,
  A-27): the cube and
  the P&L by reference by default, the exercise decisions on request.
- **Idempotent submission** (I-89): the same `idempotency_key` and body (its canonical JSON,
  defaults filled in) return the first `run_id`; another body is a `409`; the key is unique in
  the queue, so two API processes get one run. `keep` exempts a run from retention (I-76).
- **Wiring, not only names** (I-88): for every field of every schema, the completeness test sets
  it to a value other than its default, converts the schema, and asserts the value changed its
  own setting and nothing else; the run request's mapping onto `price_portfolio`,
  `run_market_risk` and the cross-asset model is checked field by field.

**Removed, there being no clients** (A-2, revised 2026-10-07): `POST /portfolio/price`,
`POST /portfolio/market-risk`, `POST /calibration/cam`, `POST /calibration/lgm` (and with it the
standalone bootstrap, I-94: the `calibration` analytic is ORE's), `GET /jobs/...`, and the fields
`scenario_risk`, `compute_greeks`, `cube_output` and `pnl_output`. `/health`, `/version` and the
EOD routes stay; the `/eod` contract is TraderX's. A version-2 queue file is migrated in place
(schema 3, one transaction, I-106), its unrun jobs marked `interrupted`.

Saving a market or a configuration on the server to use by reference is not planned.

## One API reaching every setting (I-56, I-10, I-09) — done 2026-10-07

- **One route per analytic.** `POST /portfolio/price` (the portfolio request since the shared pipeline),
  `POST /portfolio/market-risk` (`MarketRiskRequestSchema`: the market and trades, Monte Carlo
  or historical scenarios on named factors, engines, quantiles, precision), both jobs for the
  engine worker, polled at `GET /jobs/{job_id}`; `POST /calibration/cam`, synchronous. The job
  queue keeps each job's kind (schema version 2, migrating version 1 in place) and the worker
  runs the kind's entry point (`engine.api.worker.JOB_KINDS`).
- **Validated before any job starts**, without JAX work: each queued request's `.check()` runs
  the engine's own validation (`validate_request`; for market risk `validate_portfolio`,
  `validate_factors` and the scenario generators' input checks, never drawing a scenario).
- **Names say what they are.** `/v2/portfolio/price`, `schema_version: "2"` and
  `GET /portfolio/price/{job_id}` are removed, not aliased: there was no client (A-2, revised
  2026-10-07).
- **Every setting reachable.** `tests/test_api_completeness.py` walks every configuration type a
  request can hold and fails on a field without an API field. It found three: the LGM engine's
  `shift_horizon` (now a field, any non-negative value with 0.5 the default since I-32), piecewise volatilities (now
  `{"times", "values"}` wherever a volatility is taken), and market risk and the CAM calibration
  (the new routes).
- **Every per-trade figure keyed by its trade** (I-10): one row per trade (`trade_id`, t=0
  value, exposure, Greeks); the arrays' trade axes follow the rows.
- **Arrays by reference on request** (A-17, I-09): `cube_output` / `pnl_output` `"inline"`
  (default), `"artifact"` (chunks of at most 8 MiB, each hashed, the whole array hashed, the
  trade order hashed beside it, written to the queue in the result's transaction) or `"none"`.

The Python result dataclasses keep their array-oriented fields (`trade_ids` beside positional
lists and the cube's trade axis), which NumPy code indexes; the rows are the wire's.
