# HTTP API

**Modules:** [`engine/api/app.py`](../../engine/api/app.py),
[`engine/api/routes.py`](../../engine/api/routes.py),
[`engine/api/market_schemas.py`](../../engine/api/market_schemas.py) (the requests),
[`engine/api/schemas.py`](../../engine/api/schemas.py) (shared schemas and the results),
[`engine/api/artifacts.py`](../../engine/api/artifacts.py) (arrays returned by reference)

## Plain-language summary

Everything described in [The Portfolio Entry Point](portfolio-entrypoint.md) and
[Market Risk](../risk/market-risk.md) is reachable from plain Python already —
`price_portfolio(request)`, `run_market_risk(request)`, `calibrate_cam(market, models)`. This
module wraps those same functions behind an HTTP API, for a caller (like TraderX — see
[TraderX integration](../planning/details/traderx-integration.md)) that isn't a Python process
sharing this codebase's own memory space. Every setting the engine has is a field of a request
(decision A-2): `tests/test_api_completeness.py` walks the configuration types and fails on any
setting without one.

**Wrap, not replace.** `engine.portfolio.PortfolioRequest`/`PortfolioResult` and every
instrument config dataclass stay the single source of truth for the engine's own internal
shape (see [Architecture: Typed configuration](../concepts/architecture.md#typed-configuration)).
Pydantic models in `engine/api/market_schemas.py` and `engine/api/schemas.py` mirror them field-for-field, each with a
`.to_dataclass()` method converting into the real dataclass and — for results — a
`.from_dataclass()` classmethod for the reverse direction. `engine/portfolio/` and
everything below it has **zero** Pydantic/FastAPI dependency; the heavy `api` extra
(FastAPI, Pydantic, uvicorn) is only needed to run this HTTP layer, not the core
simulation/pricing/risk engine.

## Stack: FastAPI + Pydantic v2 + uvicorn

Chosen over the original roadmap's "FastAPI/gRPC" placeholder — gRPC was evaluated and explicitly not
used:

- Zero existing web framework lock-in to displace.
- [Coding Style: API-first design](../concepts/coding-style.md) already lists this as a
  core constraint — FastAPI/Pydantic's request-validation-first model matches directly.
- This workload (numerically heavy, low request volume, seconds-to-a-minute per request —
  see "Why async, not sync" below) doesn't need gRPC's binary-protocol performance or
  streaming. FastAPI's automatic OpenAPI/JSON-schema generation (`/docs`, `/openapi.json`)
  is a meaningful documentation win for near-zero extra cost, and `TestClient` (via
  `httpx`) gives the test suite (`tests/test_api.py`) a synchronous, in-process harness
  requiring no running server.

## Running it

```bash
venv/Scripts/pip install -e .[api]      # if not already installed via requirements.txt
venv/Scripts/python.exe -m uvicorn engine.api.app:app --reload
```

The server starts its engine worker, a second Python process, on the first portfolio job and
stops it on shutdown. Jobs are kept in `JAX_RISK_JOB_QUEUE` (default: `jax-risk-jobs/` under
the system temp directory). To run the worker under your own supervisor instead, start the
server with `JAX_RISK_WORKER=external` and run `jax-risk-worker --queue <the same file>`
([Jobs](#jobs-the-queue-and-the-engine-worker)).

Then visit `http://127.0.0.1:8000/docs` for FastAPI's interactive Swagger UI (a live
supplement to this doc, not a replacement for it — this page stays the authoritative
narrative reference, matching every other doc in this repository).

## Two contracts on one app

This app serves **two independent contracts**, mounted as separate routers:

| Prefix | Contract | Shape |
|---|---|---|
| *(none)* | The portfolio API — simulate, price, aggregate | Pydantic-wrapped engine dataclasses |
| `/eod` | The [TraderX EOD boundary](eod-integration.md) (W1.6.4) | Plain dicts governed by a **published JSON Schema** |

They share no state and speak deliberately different shapes. The EOD result is *not* wrapped
in a Pydantic model, because its contract is the published schema — re-describing it here
would create a second definition that can drift from the one consumers pin against. Keeping
the routers separate keeps either free to change.

**Every route on the app, so one page answers "what is served here?":**

| Route | Contract | Documented in |
|---|---|---|
| `GET /health` | portfolio | below |
| `GET /version` | portfolio | below |
| `POST /portfolio/price` | portfolio | below |
| `POST /portfolio/market-risk` | portfolio | below |
| `GET /jobs/{job_id}` | portfolio | below |
| `GET /jobs/{job_id}/artifacts/{name}/{chunk}` | portfolio | below |
| `POST /calibration/cam` | portfolio | below |
| `POST /calibration/lgm` | portfolio | below |
| `GET /eod/capabilities` | EOD | [EOD Integration](eod-integration.md#w164--the-eod-http-routes) |
| `GET /eod/schemas/result` | EOD | [EOD Integration](eod-integration.md#w164--the-eod-http-routes) |
| `GET /eod/schemas/capabilities` | EOD | [EOD Integration](eod-integration.md#w164--the-eod-http-routes) |
| `POST /eod/price` | EOD | [EOD Integration](eod-integration.md#w164--the-eod-http-routes) |
| `GET /eod/results/by-workload/{key}` | EOD | [EOD Integration](eod-integration.md#w164--the-eod-http-routes) |
| `GET /eod/attempts/{attemptId}` | EOD | [EOD Integration](eod-integration.md#w164--the-eod-http-routes) |

The EOD routes are documented in full in
[The EOD Integration Boundary](eod-integration.md#w164--the-eod-http-routes); this page covers
the portfolio contract.

On 2026-10-07 the API removed `POST /v2/portfolio/price`, the request's
`schema_version: "2"` (names that looked like versions and were not) and
`GET /portfolio/price/{job_id}` (now `GET /jobs/{job_id}`, for every kind of job), and replaced
the result's position-keyed fields with one row per trade. There were no clients to keep them
for (decision A-2, revised 2026-10-07).

**One behavioural difference worth knowing up front.** `POST /portfolio/price` and
`POST /portfolio/market-risk` are **asynchronous** (`202` + a `job_id` to poll, because a
4096-scenario Monte Carlo measured ~52s — see below), while `POST /eod/price` and the two
calibration routes are **synchronous**: the EOD path is closed-form discounted cashflows over a
handful of rows, and a calibration a bootstrap of a few helpers. The contracts differ here on
purpose, not by accident of implementation order.

---

## Endpoints

### `GET /health`

Liveness only — confirms the process is up. Does no engine work.

**Response:** `{"status": "ok"}`

### `GET /version`

Engine package version, JAX backend (CPU/GPU/TPU), and git commit (best-effort — `null` if
this isn't a git checkout).

**Response:**
```json
{"engine_version": "0.1.0", "jax_backend": "cpu (Windows)", "git_commit": "abc1234..."}
```

`jax_backend` is the backend *of the API process*, which does no pricing. A served API keeps
its own JAX on the CPU (`engine.api.app.keep_jax_on_the_cpu`), so this says `cpu`
on a GPU host too: the accelerators belong to the engine worker, a separate process with its
own JAX runtime, and each result names the devices and backend it actually ran on in its
`precision` report (I-12).

### `POST /portfolio/price`

The main endpoint. Body: the portfolio request, `MarketPortfolioRequestSchema` (mirrors
`PortfolioRequest` and its `RunConfig` — see "Request schema" below). Validates
synchronously (`engine.portfolio.market_path.validate_request`, no JAX work), then writes the
body, byte for byte as received, to the job queue as a `portfolio` job and returns immediately
(see [Jobs: the queue and the engine worker](#jobs-the-queue-and-the-engine-worker)).

**Success:** `202 Accepted`
```json
{"job_id": "b3f1c2a0-..."}
```

**Validation failure:** `400 Bad Request`, with the validator's own message. Examples:
scenario risk without a `simulation`; a trade whose currency or index curve is not in the
market (naming the trade); a trade not valued on the market's date; `trade_id` on some trades
but not others, or repeated; an engine's refusal (the Jamshidian engine and a floating
spread); a precision the pipeline does not implement yet.
```json
{"detail": "trade 'gbp-swap' (SwapConfig): no market for currency 'GBP'; have ['USD']"}
```

**Malformed schema, or an unknown field:** `422 Unprocessable Entity`. The request refuses
unknown fields, so a trade carrying `hw_sigma` or a curve is not silently stripped of its
model: curves come from the market and models from the run configuration (audit A-3).

**The Hull-White request shape retired on 2026-10-01** (a `SimulationConfig` market with
`time_grid`/`rates`/`joint_covariance`, model parameters on the trades, a
`calibration_basket`, a top-level `evaluation_date`) is a `422` whose message names its
replacement: the Hull-White model is `"model": "HullWhite"` per currency in `simulation.ir`.

### `POST /portfolio/market-risk`

Short-horizon VaR and Expected Shortfall of a portfolio by full revaluation under shock
scenarios ([Market Risk](../risk/market-risk.md), `engine.market_risk.run_market_risk`). Body:
`MarketRiskRequestSchema` (see [Market-risk request](#market-risk-request-marketriskrequestschema)).
Validated synchronously without drawing a scenario (no JAX work; a refusal is a `400` naming
the field), then queued as a `market-risk` job: `202` with a `job_id`, polled at
`GET /jobs/{job_id}`. The result is `MarketRiskResultSchema`
([below](#market-risk-result-marketriskresultschema)).

### `GET /jobs/{job_id}`

Poll for a job's status/result, of either kind: a read of the job's row in the queue.

**Response:**
```json
{"kind": "portfolio", "status": "done", "result": { "...": "PortfolioResultSchema" }, "error": null, "failure_class": null}
```

`kind` is `portfolio` (the result a `PortfolioResultSchema`) or `market-risk` (a
`MarketRiskResultSchema`).

| `status` | Meaning | `result` | `error`, `failure_class` |
|---|---|---|---|
| `pending` | Queued; the worker has not started it | `null` | `null` |
| `running` | The engine worker is pricing it | `null` | `null` |
| `done` | Priced | the result | `null` |
| `failed` | The job raised; the worker went on to the next job | `null` | the exception and traceback; the class below |
| `interrupted` | The worker stopped during the job (killed, crashed, restarted). Nothing was priced to completion; submit the request again | `null` | the reason; `null` |

`failure_class` (I-08), from the exception's type (`engine.api.worker.failure_class`):
`missing-market-data` (a curve, currency or fixing not supplied: `KeyError`,
`MissingFixingError`), `unsupported-product` (`NotImplementedError`, an unsupported day
count), `numerical-failure` (`ArithmeticError`: overflow, division by zero, floating-point
error), `bad-terms` (any other `ValueError` or `TypeError`), `infrastructure` (anything else:
memory, XLA runtime, a result too large to store). The route validates before queueing, so
terms the engine refuses are normally a `400`, not a failed job.

Until 2026-10-04 `running` was never reported (`pending` covered both) and there was no
`interrupted` or `failure_class`.

**Unknown `job_id`:** `404 Not Found`.

### `GET /jobs/{job_id}/artifacts/{name}/{chunk}`

One chunk of an array a done job returns by reference (`cube_output` or `pnl_output`
`"artifact"`; decision A-17, [I-09](../planning/known-issues.md#closed) closed): the raw bytes,
`application/octet-stream`, exactly as the result's reference describes them. A chunk that does
not exist is a `404`. The chunks are written to the job queue in the same transaction as the
result, so a `done` job always has all of them.

**The reference** (`npv_cube_artifact`, `pnl_artifact`; `engine.api.artifacts`):

```json
{"name": "npv_cube", "dtype": "float64", "byte_order": "little",
 "shape": [4096, 24, 8], "axes": ["scenario", "date", "trade"],
 "chunks": [{"url": "/jobs/<id>/artifacts/npv_cube/0", "rows": [0, 5461], "bytes": 8387328, "sha256": "..."}, ...],
 "sha256": "<of every chunk's bytes, in order>",
 "items": {"axis": "trade", "ids": ["swap-1", "..."], "sha256": "<of the ids' canonical JSON>"}}
```

The array is C-ordered (the last axis fastest), little-endian, split along its first axis into
chunks of whole rows of at most 8 MiB. To read it: fetch each chunk, check its `sha256`,
concatenate in order, check the whole array's `sha256`, and reshape. The trade order travels as
its own hashed record (`items`: the ids' canonical JSON, `json.dumps(ids, separators=(",",
":"))`), so a consumer can verify the order it read is the order published, as the EOD result's
`itemOrder`. `engine.api.artifacts.read_array(reference, fetch)` does all of it in Python;
`demos/demo_api.py` does it with `hashlib` and `struct` alone.

### `POST /calibration/cam`

The cross-asset model's calibration (`engine.calibration.cam.calibrate_cam`): each currency's
model bootstrapped to its calibration basket on today's market, the calibration a portfolio
run with this market and these models simulates with ([Calibration](calibration.md)).
Synchronous. Body: today's `market` (the portfolio request's) and `ir`, each currency's model
as the simulation's `ir` gives it (`"model": "LGM"` or `"HullWhite"`, `reversion`,
`volatility` as the bootstrap's start, `calibration_expiries` × `calibration_terms`,
`swap_index`, `solver`). A currency without a basket is a `422` ("nothing to calibrate"); one
the market lacks, or a helper whose volatility cannot be reached, a `400`.

**Request:**
```json
{"market": {"asof": "2026-07-30", "currencies": {"USD": {"...": "..."}}},
 "ir": {"USD": {"model": "LGM", "reversion": 0.03, "volatility": 0.01,
                "calibration_expiries": ["1Y", "2Y", "5Y"], "calibration_terms": ["5Y", "4Y", "1Y"]}}}
```

**Response:** per currency, the model, the reversion it was calibrated with, the volatility in
its model's parametrization (the LGM's alpha, or the Hull-White short rate's sigma):
`sigma_values[i]` on bucket `i` of `[0, sigma_times[0]), ..., [sigma_times[-1], inf)`; and each
helper's expiry, term, market value (Bachelier on the ATM volatility) and model value.

```json
{"currencies": {"USD": {"model": "LGM", "reversion": 0.03, "sigma_times": [1.0, 2.005...],
  "sigma_values": [0.0081, 0.0089, 0.0094],
  "helpers": [{"expiry": "1Y", "term": "5Y", "market_value": 0.0164..., "model_value": 0.0164...}, "..."]}}}
```

### `POST /calibration/lgm`

Standalone calibration — wraps `engine.calibration.basket.build_coterminal_basket` +
`engine.calibration.lgm.calibrate_lgm_sigma`: a Hagan bootstrap of an LGM `Sigma` to a
caller-given co-terminal basket (its times and volatilities given directly, on a zero curve),
each bucket by the body's `solver` (`"Newton"`, the default, or `"Bisection"`;
[the root solver](calibration.md#the-root-solver)). Synchronous. It is not the portfolio's
calibration, which is the cross-asset model's per currency (`POST /calibration/cam`) and each
option's own basket ([Calibration](calibration.md)).

**Request:**
```json
{
  "evaluation_date": "2026-07-30",
  "exercise_times": [1.0, 2.0, 3.0, 4.0],
  "final_maturity_time": 5.0,
  "notional": 1000000.0,
  "payer": true,
  "market_vols": [0.008, 0.0088, 0.0095, 0.01],
  "zero_curve": {"times": [0.0, 1.0, 2.0, 5.0, 10.0, 30.0], "rates": [0.03, 0.03, 0.03, 0.03, 0.03, 0.03]},
  "hw_a": 0.03,
  "solver": "Newton"
}
```

**Response:**
```json
{
  "sigma_times": [1.0, 2.008..., 3.008...],
  "sigma_values": [0.00848, 0.01038, 0.01202, 0.01304],
  "market_prices": [11510.94, 13247.34, 11498.11, 6870.55],
  "model_prices": [11510.94, 13247.34, 11498.11, 6870.55],
  "rmse": 1.8e-10
}
```

## Why async, not sync: the measured latency

`POST /portfolio/price` (and `POST /portfolio/market-risk`) returns `202 Accepted` + a job id
and prices in the background, rather than blocking the HTTP response until pricing finishes. This is a direct consequence
of measured timing, not a default framework choice:

**A 4-trade, 4096-scenario portfolio took ~52 seconds wall time end-to-end**, dominated by
JAX JIT compilation (a one-time-per-shape cost that partially amortizes across repeated
calls with the same array shapes, but the first call after process start — or any call with
a new time-grid/scenario-count shape — pays it in full) plus the Monte Carlo simulation
itself. A larger, TraderX-realistic portfolio (dozens of trades, more scenarios) will only
be slower.

A synchronous HTTP response held open for tens of seconds to minutes is fragile: client and
reverse-proxy timeouts, no progress visibility, no ability to retry a request without
recomputing everything from scratch. The async job pattern avoids all three at essentially
no extra infrastructure cost for this system's actual scope.

**If you're reading this because someone wants to "simplify" `/portfolio/price` back into a
single synchronous call: re-derive this reasoning first.** The measured latency above is
the reason this exists, not a design preference — a synchronous version would need to
re-solve the timeout/retry/progress problems this pattern already avoids.

## Jobs: the queue and the engine worker

Since 2026-10-04 (decision A-14; [details/precision.md §11](../planning/details/precision.md#11-execution-architecture)):

```
 API process(es)                   job queue (SQLite)                 engine worker (one per host)
 POST: validate, insert body ───▶  pending ─▶ running ─▶ done    ◀──  claim oldest, parse body by
       and its kind                           └─▶ failed / interrupted   kind, run it, store result
 GET:  read the row, chunks  ◀───                                         (and artifact chunks)
```

- **The queue** (`engine/api/job_queue.py`) is one SQLite file, `JAX_RISK_JOB_QUEUE`
  (default `jax-risk-jobs/jobs.sqlite3` under the system temp directory). Each row holds the
  job's kind (`portfolio` or `market-risk`), the request body as received, the status, the
  failure class and error, the result document, the worker that claimed it, the number of XLA
  programs the job built, and timestamps; a second table holds the chunks of the arrays a
  result returns by reference, written with the result. It survives restarts of the API and the
  worker, and every API process that opens it sees the same jobs, so `uvicorn --workers N`
  works: a `job_id` issued by one API process is served by any other. A queue file of the
  first schema (version 1, 2026-10-04) is migrated in place when first opened: its jobs become portfolio
  jobs.
- **The engine worker** (`engine/api/worker.py`, command `jax-risk-worker`) is one
  single-threaded process per host. It takes jobs in submission order, parses each body
  exactly as the route did (so nothing is pickled), runs the kind's entry point
  (`price_portfolio` or `run_market_risk`), and stores the result document, which the route
  then sends verbatim. It owns every device JAX sees on the
  host (a served API keeps its own JAX on the CPU) and keeps its compiled
  programs for its lifetime, so a repeated job shape compiles nothing. It also keeps them on disk, in JAX's persistent compilation cache, so a restarted
  worker reads them back instead of compiling again: in `JAX_COMPILATION_CACHE_DIR` if set
  (set it empty to turn the cache off), else in `xla-cache/` beside the queue file. A file lock (`<queue>.worker.lock`) makes it the queue's only worker; the
  operating system releases the lock however the worker dies.
- **The worker's device settings** (decision A-22). It runs deterministic GPU
  kernels, so a job gives the same bits on every run of its compiled program: at start-up it
  adds `--xla_gpu_exclude_nondeterministic_ops=true` to its `XLA_FLAGS`, unless they already
  name `--xla_gpu_exclude_nondeterministic_ops` or `--xla_gpu_deterministic_ops` (either
  value; the second also pins autotuning, at twice the compile time). GPU memory is the
  deployment's choice: the worker keeps JAX's default of preallocating 75% of a GPU, right for
  one worker per GPU; start the server (or `jax-risk-worker`) with
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` where other processes share the card, as the demos
  do. Both apply on a GPU only; neither changes a CPU number.
- **Failures stay in their row.** A job that raises is `failed` with its class and traceback,
  and the worker takes the next job. A worker that dies mid-job leaves the row `running`; the
  next worker to start marks it `interrupted` before claiming anything, so a job that kills
  the worker is not retried into a crash loop.
- **Supervision**, `JAX_RISK_WORKER` (`engine/api/supervisor.py`):
  - `spawn` (default): the API starts the worker as a child process on the first job and
    checks it on every submission and poll, restarting it if it died. No thread watches it.
    With several API processes, a supervisor starts no worker while another process's worker
    holds the lock. The worker exits once the API process that started it is gone, so a
    killed server leaves no engine process behind.
  - `external`: the API only reads and writes the queue; run `jax-risk-worker --queue PATH`
    under systemd (`Restart=always`), a container restart policy or a pod's process manager.
    `pip install -e .` registers the command; `python -c "from engine.api.worker import
    main; main()"` is the same thing.
- **Jobs run one at a time, each on every device of the host.** A job's scenarios (the
  simulation's paths, market risk's shocks) are split across the host's devices
  (`engine/simulation/sharding.py`): as many devices as divide the scenario count evenly,
  at most `JAX_RISK_SCENARIO_DEVICES` if set (`1` keeps a job on one device). The result's
  precision report lists the devices. Several hosts (a TPU pod slice) are not yet supported
  ([I-61](../planning/known-issues.md#i-61)).

Not yet built: rows are never deleted (results accumulate in the file,
[I-76](../planning/known-issues.md#i-76)); a worker that cannot start at all leaves jobs
`pending` with no signal ([I-77](../planning/known-issues.md#i-77)); there is no cancel
route.

**The EOD path keeps its own store.** `POST /eod/price` publishes every *terminal* attempt
through a crash-safe filesystem store (`engine/integration/publication.py`), so an EOD result
survives a restart and stays addressable by `attemptId`; a *running* EOD attempt is still
memory-only ([I-08](../planning/known-issues.md#i-08),
[EOD Integration](eod-integration.md#w164--the-eod-http-routes)).

## Portfolio request: `MarketPortfolioRequestSchema`

Mirrors `engine.portfolio.PortfolioRequest` and its run configuration
(`engine/api/market_schemas.py`):

| Field | Meaning |
|---|---|
| `market` | `asof` (ISO date; every trade is valued on it); `currencies`: per currency a `discount_curve`, `index_curves` keyed by index name (`"USD-SIMINDEX-6M"`), and `swaption_vols` (ATM normal matrix: `option_tenors`, `swap_tenors`, `vols`); `fx_spots` keyed `"EURUSD"`; `equities` |
| `trades` | Discriminated by `trade_type`: `swap`, `european_swaption`, `bermudan_swaption`, `american_swaption`, `bond`. Each names its `currency` and `index_tenor_months` and carries no model or curve. Swaptions take `settlement` (`Physical` or `Cash`). `trade_id` on every trade or on none (none numbers them `trade-0`, `trade-1`, ...) |
| `simulation` | `RunConfig.simulation`: ORE's `simulation.xml` as `CamConfigSchema`: `dates`, `base_currency`, `ir` per currency (`model`: `"LGM"`, the default, or `"HullWhite"`; `reversion`, `volatility`, optional calibration basket `calibration_expiries` × `calibration_terms`, `swap_index`, and the bootstrap's root `solver`: `"Newton"`, the default, or `"Bisection"`), `fx_volatilities`, `equity_volatilities`, `correlations` between factors `IR:USD`, `FX:EURUSD`, `EQ:SP5`, `curve_tenors`, `samples`, `seed`, `swaption_vol_decay`. Every volatility is a number, or piecewise constant: `{"times": [1.0, 3.0], "values": [0.008, 0.011, 0.009]}`, `values[i]` on bucket `i` of `[0, times[0]), ..., [times[-1], inf)` (ORE's `VolatilityTimes`/`Volatility`). Required with `scenario_risk`. Every `solver` is [the root solver](calibration.md#the-root-solver) (decision A-21) |
| `pricing` | `RunConfig.pricing`: `european` (`"Bachelier"`, the default, or `"Jamshidian"` with `jamshidian: {"reversion", "volatility", "solver"}`), the `bermudan` and `american` engines (`LgmEngineSchema`: `reversion`, `volatility`, `calibration`, `strategy`, `reference_calibration_grid`, `shift_horizon` (ORE's `ShiftHorizon`; only `0` is implemented, another value is a `400`, [I-32](../planning/known-issues.md#i-32)), `n_per_std`, `std_devs`, `exercise_time_steps_per_year`, `swap_index`, and `solver`, that of the calibration and of each helper's exercise boundary), and `recalibrate` (default `true`, as ORE's `ValuationEngine`) |
| `greeks` | `RunConfig.greeks`: `method` (`"Bump"`, the default, or `"AD"`) and `sensitivity` (ORE's `sensitivity.xml`: `curve_tenors`, `curve_shift`, `vol_shift`, `theta_days`, `swaption_vol_decay`) |
| `base_currency` | The reporting currency. Omitted: the simulation's base currency, or USD without a simulation. One contradicting the simulation's is a 400 |
| `precision` | `RunConfig.precision`, see below |
| `pfe_quantiles` | Quantiles of the PFE profiles. Default `[0.95, 0.99]` |
| `compute_greeks` | Default `false` |
| `scenario_risk` | Default `true`; `false` prices today's values (and Greeks) only |
| `cube_output` | How the result carries the NPV cube (decision A-17): `"inline"` (the default, `npv_cube`), `"artifact"` (a chunked, hashed reference, `npv_cube_artifact`, read at [`GET /jobs/{job_id}/artifacts/...`](#get-jobsjob_idartifactsnamechunk)) or `"none"` (neither; ORE too writes its cube only when asked, `cubeFile`) |

Representational differences from the dataclasses (SWIG-bound `ORE` types are not natively
Pydantic-serializable): dates are ISO strings (`"2026-07-30"`), periods are ORE strings
(`"5Y"`, `"18M"`), historical `fixings` are `{"YYYY-MM-DD": rate}`. A trade's schedule is
given **either** as `effective_date`/`maturity_date` (plus `exercise_date` for a
`european_swaption`) **or** as `swap_tenor` (plus `forward_start`/`exercise_lag_days`),
resolved on the market's date. A `"bond"` is a Treasury bill (no `coupon_schedule`) or note;
its `face_amount` is signed.

A minimal body (one swap under the Hull-White model, with exposure):

```json
{"market": {"asof": "2026-07-30", "currencies": {"USD": {
     "discount_curve": {"times": [0, 1, 5, 30], "rates": [0.03, 0.03, 0.04, 0.05]},
     "index_curves": {"USD-SIMINDEX-6M": {"times": [0, 1, 5, 30], "rates": [0.034, 0.034, 0.044, 0.052]}}}}},
 "trades": [{"trade_type": "swap", "trade_id": "swap-1", "notional": 1e7, "fixed_rate": 0.042,
             "payer": true, "swap_tenor": "5Y"}],
 "simulation": {"dates": ["2027-07-30", "2028-07-30"], "base_currency": "USD", "samples": 1024,
                "ir": {"USD": {"model": "HullWhite", "reversion": 0.03, "volatility": 0.01}}},
 "compute_greeks": true}
```

### Precision control: `PrecisionSchema`

Mirrors `engine.precision.Precision` (see [The Portfolio Entry Point:
Precision](portfolio-entrypoint.md#precision) and
[Architecture](../concepts/architecture.md#adjustable-precision)): `simulation`, `market` and
`pricing`, each an object of format names `storage`, `compute` and `accumulate`, every field
`"float64"` when omitted; and the pricing stage's overrides, `by_product` (keyed by a
`trade_type`) and `by_trade` (keyed by a `trade_id`), each mapping to such an object. A trade
is priced at its `by_trade` entry, else its product's, else `pricing` (2026-10-02).
`rounding` (`"nearest"`, the default, or `"stochastic"`) is how values are rounded into a
storage format below 32 bits, and `rounding_seed` (a non-negative integer, 0 by default) seeds
the stochastic rounding (2026-10-02). `paired_fraction` (in [0, 1], 0 by default; outside it a
`422`) is the share of paths also run at float64, whose estimates the result's `precision`
carries (2026-10-02). Unknown fields are refused.

```json
"precision": {"simulation": {"storage": "float32", "compute": "float32", "accumulate": "float32"},
              "pricing": {"storage": "float32"},
              "by_product": {"bermudan_swaption": {}},
              "by_trade": {"swap-7": {"storage": "float32", "compute": "float32", "accumulate": "float32"}}}
```

Storage below 32 bits, the cube in FP8 with stochastic rounding:

```json
"precision": {"pricing": {"storage": "float8_e4m3fn"}, "rounding": "stochastic", "rounding_seed": 7}
```

A `by_product` key that is not a `trade_type` is a `422`; a `by_trade` key that names no
trade of the request is a `400` (checked with the request, so no job is created).

The format names are the format table's (`float64`, `float32`, `float16`, `bfloat16`,
`float8_e4m3fn`, `float8_e5m2`); another name, or another rounding, is a `422`. Every format
stores; a format not yet enabled for compute (below float32, F-07), or an inconsistent
stage, is a `400` naming the stage (or override, e.g. `precision.by_trade['swap-7']`), the
field and the roadmap step that enables it. So is `"rounding": "stochastic"` when no stage
stores below 32 bits:

```
POST /portfolio/price
{"precision": {"pricing": {"storage": "float8_e4m3fn", "compute": "float16", "accumulate": "float16"}}, ...}
-> 400 {"detail": "precision.pricing: StagePrecision.compute='float16': compute in float16 is not enabled
        yet (F-07: difference-form kernels); until then float64 or float32"}
```

The 32/64 shape before 2026-10-01 (`{"simulation": 32, "pricing": 64, "risk": ..., "calibration": ...}`)
is a `422` whose message names the replacement (decision A-12); it is not translated.

**Concurrency note:** jobs of every precision go to the one engine worker, in submission
order. Each job's result is independent of what was queued with it: jobs queued together
give the bits they give run one after another (`tests/test_engine_worker.py`).

## Market-risk request: `MarketRiskRequestSchema`

Mirrors `engine.market_risk.MarketRiskRequest` ([Market Risk](../risk/market-risk.md)):

| Field | Meaning |
|---|---|
| `market`, `trades` | As the portfolio request's; every trade valued on `market.asof`, `trade_id` on every trade or none |
| `scenarios` | The shock scenarios, by `source` (below) |
| `pricing` | The engine per product, as the portfolio request's `pricing`; options are priced with fixed volatility (the result's warnings say so) |
| `quantiles` | VaR/ES confidence levels; default `[0.99, 0.975]` |
| `precision` | As the portfolio request's: the simulation stage stores the shifts, each trade's pricing stage its revaluation and P&L |
| `batch_size` | Scenarios revalued at once per trade (lowered automatically for a large Bermudan); default 256 |
| `pnl_output` | How the result carries the P&L matrix, as `cube_output`: `"inline"` (default), `"artifact"`, `"none"` |

`scenarios.factors` names the risk factors, the pillar zero rates of market curves, in the order
of the covariance's or the history's columns: each curve's labels `"<curve>/<pillar time>y"`
for every pillar in the market's order, curve after curve, e.g. `"discount:USD/1y"`,
`"index:USD-SIMINDEX-6M/1y"` (`engine.market_risk.RateRiskFactors.labels`). Any curves of the
market, in any order; every curve a trade reads must be among them. A label out of its curve's
pillar order, a curve the market lacks, or a trade's curve left out is a `400` naming the
expected labels.

- `{"source": "monte-carlo", "factors", "covariance", "horizon_days", "num_scenarios", "seed"}`:
  zero-mean Gaussian moves over the horizon with the given `[F][F]` covariance (symmetric
  positive semi-definite), scrambled Sobol normals (`monte_carlo_scenarios`). Drawn by the engine
  worker, not at validation.
- `{"source": "historical", "factors", "history", "horizon_days", "dates"}`: one scenario per
  overlapping `horizon_days` window of a `[D][F]` history of the factors' levels, oldest first;
  `dates`, optional, label each window (`historical_scenarios`).

```json
{"market": {"...": "..."}, "trades": [{"trade_type": "swap", "trade_id": "swap-1", "...": "..."}],
 "scenarios": {"source": "monte-carlo", "factors": ["discount:USD/0y", "discount:USD/1y", "..."],
               "covariance": [[6.4e-6, "..."], "..."], "horizon_days": 10, "num_scenarios": 16384, "seed": 1},
 "quantiles": [0.99, 0.975]}
```

## Design: one configurable API

Decided 2026-09-30 ([compliance/decisions.md](../../compliance/decisions.md) A-2; ORE
alignment plan 9.2), done on 2026-10-07:

- **One route per analytic, one request each.** A portfolio's pricing and exposure, its market
  risk, and the cross-asset model's calibration; every job polled at one route. The request's
  configuration selects the model (LGM, the default, or Hull-White), the simulation, the pricing
  engine per product, the Greeks method, the settlement method and the precision per stage, as
  ORE's configuration files do. Defaults are ORE's.
- **Every setting reachable.** Anything the engine can be configured to do, the API can ask
  for. `tests/test_api_completeness.py` walks every configuration type a request can hold,
  from `PortfolioRequest`, `MarketRiskRequest`, the scenario generators and `calibrate_cam`,
  and fails on a field without an API field (by name, renamed, or exempt with its reason: a
  trade's `evaluation_date` is the market's, a curve's `provenance` is the EOD boundary's
  metadata), on a configuration type without a schema, and on an exemption naming no field.
- **Robust.** Validated before any job starts: types, unknown fields refused, cross-field
  checks, each refusal a `400` or `422` naming its field.
- **Names say what they are.** No route or field is named like a version unless it marks a
  revision of the contract itself. `/v2` and `schema_version: "2"` went on 2026-10-07.
- **Every per-trade figure is keyed by its trade.** Each result has one row per trade with its
  `trade_id` ([I-10](../planning/known-issues.md#closed)); the arrays' trade axes follow the
  rows, and an array by reference carries the trade order hashed.

## Portfolio result: `PortfolioResultSchema`

Mirrors `engine.portfolio.PortfolioResult`, with one row per trade:

| Field | Type | Meaning |
|---|---|---|
| `base_npv` | `float` | Portfolio total on today's market: the sum of the rows' `base_npv`, the same numbers. |
| `trades` | `List[{"trade_id", "base_npv", "exposure", "greeks"}]` | One row per trade, in the request's `trades` order, which is also the cube's trade axis: its id ([I-10](../planning/known-issues.md#closed)), its t=0 NPV on today's market, its standalone exposure (the object below; `null` without scenario risk) and its Greeks (`null` unless `compute_greeks`). Each Greek is flattened row-major into `values`; one of more than one dimension (`vega:<ccy>`, option tenors × swap tenors) also has its shape in `shapes`. The keys are `delta:discount:<ccy>`, `gamma:discount:<ccy>`, `delta:index:<name>`, `gamma:index:<name>` (per curve tenor for `Bump`, per market pillar for `AD`), `vega:<ccy>` for a trade whose engine reads the swaption volatilities, and `theta`. See [Greeks](../risk/greeks.md). |
| `exposure` | `{"times": [...], "epe": [...], "ene": [...], "ee_b": [...], "eee_b": [...], "pfe": {"PFE_95": [...], ...}} \| null` | The whole portfolio as one netting set; every list has one entry per date in `times`, starting at t=0, where each trade is valued on the simulation market of the as-of date, as ORE's cube starts (2026-10-07). See [Exposure](../risk/exposure.md). |
| `exposure.epe_b`, `exposure.eepe_b` | `List[float]` | ORE's time-weighted EPE_B / EEPE_B profiles. |
| `exposure.basel_epe`, `exposure.basel_eepe` | `float \| null` | ORE's Basel EPE_B / EEPE_B at the one-year horizon. |
| `npv_cube` | `List[List[List[float]]] \| null` | `[Scenarios, Dates, Trades]`, JSON-nested, with `cube_output: "inline"` (the default). |
| `npv_cube_artifact` | `object \| null` | The cube by reference, with `cube_output: "artifact"` ([format](#get-jobsjob_idartifactsnamechunk)). |
| `measure` | `str \| null` | `risk-neutral-pricing`, or `null` without scenario risk (I-11). |
| `precision` | `object` | The precision report, built in the worker that ran the job, so its `devices` and `backend` are the worker's, not those `/version` names ([I-12](../planning/known-issues.md#i-12), closed): `policy` (the request's `precision`, defaults filled in), `trades`, `realized`, `devices`, `backend`, `jax_version`, `paths`, `paired_paths`, `figures` (each with `"kind": "mean"` or `"quantile"`; NaN as `null`). See [The Portfolio Entry Point](portfolio-entrypoint.md#precision). |
| `warnings` | `List[str]` | Run warnings (none are emitted today; see [The Portfolio Entry Point: Known-limitation flagging](portfolio-entrypoint.md#known-limitation-flagging)). |

## Market-risk result: `MarketRiskResultSchema`

Mirrors `engine.market_risk.MarketRiskResult`:

| Field | Meaning |
|---|---|
| `base_npv`, `trades` | The total and one row per trade (`trade_id`, `base_npv`), in request order: the P&L's trade axis |
| `risk` | `VaR_<q>`, `ES_<q>` as positive losses, `ES_<q>_tailCount`, `ES_<q>_standardError`, per quantile label (`99`, `97.5`); NaN as `null` |
| `measure`, `source`, `horizon_days`, `num_scenarios` | The scenarios' description (`historical-forecast`, `monte-carlo` or `historical`) |
| `risk_factors` | The factors shocked, `"<curve>/<pillar time>y"`, in the scenarios' order |
| `pnl` / `pnl_artifact` | The P&L `[Scenarios, Trades]`, inline or by reference (`pnl_output`); the portfolio P&L the statistics use is each scenario's sum |
| `warnings` | Volatility held fixed for options; a tail too thin for a stable estimate |
| `precision` | The precision report, as the portfolio result's; with a paired float64 sample each VaR/ES measured on it |

## Example: a Python `requests` session

```python
import time
import requests

BASE = "http://127.0.0.1:8000"

body = {
    "market": {"asof": "2026-07-30", "currencies": {"USD": {
        "discount_curve": {"times": [0.0, 1.0, 2.0, 5.0, 10.0, 30.0], "rates": [0.030, 0.030, 0.034, 0.040, 0.046, 0.050]},
        "index_curves": {"USD-SIMINDEX-6M": {"times": [0.0, 1.0, 2.0, 5.0, 10.0, 30.0],
                                             "rates": [0.034, 0.034, 0.038, 0.044, 0.049, 0.052]}}}}},
    "trades": [
        {"trade_type": "swap", "trade_id": "swap-1", "notional": 1_000_000.0, "fixed_rate": 0.036,
         "payer": True, "swap_tenor": "2Y"},
    ],
    "simulation": {"dates": ["2027-01-30", "2027-07-30", "2028-07-30"], "base_currency": "USD", "samples": 4096,
                   "ir": {"USD": {"model": "HullWhite", "reversion": 0.03, "volatility": 0.01}}},
    "pfe_quantiles": [0.95, 0.99],
}

r = requests.post(f"{BASE}/portfolio/price", json=body)
r.raise_for_status()
job_id = r.json()["job_id"]
print("submitted job", job_id)

while True:
    r = requests.get(f"{BASE}/jobs/{job_id}")
    data = r.json()
    if data["status"] in ("done", "failed", "interrupted"):
        break
    time.sleep(1.0)

if data["status"] != "done":
    raise RuntimeError(data["error"])

print("base NPV:", data["result"]["base_npv"])
print("per trade:", {row["trade_id"]: row["base_npv"] for row in data["result"]["trades"]})
print("PFE 95% at each date:", data["result"]["exposure"]["pfe"]["PFE_95"])
```

See a curl-only version in [User Guide: Running the API](../getting-started/user-guide.md#running-the-api).

## Tested by

`tests/test_api_market_path.py`: on the shared test portfolio (`tests/support/portfolio.py`),
the polled result equals a direct `price_portfolio` call: NPVs, cube, the exposure profiles
including EPE_B/EEPE_B and Basel, each trade's row, and the 2-D Vega; the cube by reference is
the inline cube, its hashes checked, and `"none"` leaves it out. It also checks the `400`s and
the `422` for a trade carrying `hw_sigma`. `tests/test_shared_portfolio.py` checks that the
HTTP body of that portfolio is the dataclass portfolio.

`tests/test_api_market_risk.py`: the market-risk result equals a direct `run_market_risk` call
for Monte Carlo and historical scenarios, the P&L by reference is the inline P&L, and each
request the run would refuse is a `400` before any job (factor order, an unknown curve, a
trade's curve left out, a covariance that is not PSD or of the wrong shape, a short history, a
quantile, no trades, the batch). `tests/test_api_artifacts.py`: the artifact format rebuilds an
array bit for bit and refuses a tampered chunk, order or array. `tests/test_api_completeness.py`:
every configuration setting has an API field. `tests/test_engine_worker.py`: the queue's kinds,
artifacts written with their result, the version-1 migration, the poll and chunk routes.

`tests/test_api.py`, using FastAPI's `TestClient` (backed by `httpx`) — no running server
process needed:

- `TestHealthAndVersion` — `/health`/`/version` respond.
- `TestPortfolioPriceHappyPath` — the Hull-White model named in the request: the polled
  result equals a direct `price_portfolio` call bit for bit; the model and the Jamshidian
  engine reach the worker; Greeks included when requested.
- `TestPortfolioPriceAtScale` — an empty `trades` list; many identical trades pricing
  identically; two concurrent jobs not cross-contaminating each other's results.
- `TestPortfolioPriceInvalidPayload` — the retired Hull-White shape is a `422` naming its
  replacement; a trade carrying model parameters and an unknown model are `422`s; a currency
  the market lacks is a `400` naming the trade; malformed/missing/invalid-discriminator
  bodies are `422`s.
- `TestPortfolioPricePrecision` — the overrides per product and trade reach the run configuration, an unknown product is a `422` and an unknown trade id a `400`; the retired 32/64 shape and an unknown format are `422`s
  naming the replacement or the table; a format not yet enabled is a `400` naming the stage,
  field and step, not a failed job; omitted equals explicit float64.
- `TestPortfolioPriceWorkerPoolDispatch` — a float64 and a float32 job complete, each equal to
  the direct call, and differ from each other.
- `TestGapFixesSurviveTheHttpBoundary` — swap Greeks and per-trade NPVs cross the worker
  boundary.
- `TestPortfolioPriceUnknownJob` — an unknown `job_id` returns `404`.
- `TestCalibrationEndpoint` — `/calibration/lgm` happy path and malformed-schema `422`.
- `TestCamCalibrationEndpoint` — `/calibration/cam` under both models equals
  `calibrate_cam` and the model a portfolio run simulates with; a currency without a basket is
  a `422`, one the market lacks a `400`.
- The names retired on 2026-10-07 are refused: `schema_version` a `422`, `/v2/portfolio/price` a
  `404`.
