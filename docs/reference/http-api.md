# HTTP API

**Modules:** [`engine/api/app.py`](../../engine/api/app.py),
[`engine/api/routes.py`](../../engine/api/routes.py),
[`engine/api/requests.py`](../../engine/api/requests.py) (the run request and the wire forms it shares),
[`engine/api/results.py`](../../engine/api/results.py) (the run's result, status and health),
[`engine/api/artifacts.py`](../../engine/api/artifacts.py) (arrays returned by reference),
[`engine/api/job_queue.py`](../../engine/api/job_queue.py),
[`engine/api/worker.py`](../../engine/api/worker.py),
[`engine/api/supervisor.py`](../../engine/api/supervisor.py) (the queue and the engine worker)

## Plain-language summary

Everything the engine computes is one request, as ORE runs a portfolio, its market data and its
configuration files with a list of analytics (decision A-2, [I-87](../planning/known-issues.md#closed)):

```
POST /runs   {"market": ..., "portfolio": [...], "analytics": ["npv", "exposure", ...], "config": {...}}
```

The same request is `engine.run.run(RunRequest)` in Python
([The Portfolio Entry Point](portfolio-entrypoint.md)), and the HTTP layer only translates:
the engine worker parses the body into a `RunRequest`, runs it, and serializes the
`RunResult`. Every setting the engine has is a field of the request:
`tests/test_api_completeness.py` walks the configuration types from `RunRequest` and fails on a
setting without a field, and sets every field to a value other than its default to check that the
value arrives in its own setting of the engine and nowhere else (I-88).

**Wrap, not replace.** `engine.run.RunRequest`, `RunResult` and every configuration and trade
dataclass stay the single source of truth for the engine's shape (see
[Architecture: Typed configuration](../concepts/architecture.md#typed-configuration)). The
Pydantic models mirror them field for field, each with a `.to_dataclass()`; `engine/run/` and
everything below it has **zero** Pydantic/FastAPI dependency, so the `api` extra (FastAPI,
Pydantic, uvicorn) is only needed to run this layer.

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

The server starts its engine worker, a second Python process, on the first run and stops it on
shutdown. Runs are kept in `JAX_RISK_JOB_QUEUE` (default: `jax-risk-jobs/jobs.sqlite3` under the
system temp directory) for `JAX_RISK_RETENTION_DAYS` after they finish (30 by default). To run
the worker under your own supervisor instead, start the server with `JAX_RISK_WORKER=external`
and run `jax-risk-worker --queue <the same file>` ([Runs: the queue and the engine
worker](#runs-the-queue-and-the-engine-worker)).

| Variable | Meaning | Default |
|---|---|---|
| `JAX_RISK_JOB_QUEUE` | The job queue's SQLite file | `jax-risk-jobs/jobs.sqlite3` in the temp directory |
| `JAX_RISK_WORKER` | `spawn` (the API starts its worker) or `external` | `spawn` |
| `JAX_RISK_WORKER_LOCK` | The host's worker lock: one engine worker per host (or per set of devices given a lock of its own) | `jax-risk-engine/worker.lock` in the temp directory |
| `JAX_RISK_RETENTION_DAYS` | How long a finished run is kept; `off` keeps every run | `30` |
| `JAX_RISK_SCENARIO_DEVICES` | At most this many devices per run | every device that divides the scenario count |

Then visit `http://127.0.0.1:8000/docs` for FastAPI's interactive Swagger UI (a live
supplement to this doc, not a replacement for it — this page stays the authoritative
narrative reference, matching every other doc in this repository).

## Two contracts on one app

This app serves **two independent contracts**, mounted as separate routers:

| Prefix | Contract | Shape |
|---|---|---|
| *(none)* | The run request — every analytic of the engine | Pydantic-wrapped engine dataclasses |
| `/eod` | The [TraderX path](traderx-path.md) (W1.6.4) | Plain dicts governed by a **published JSON Schema** |

They share no state and speak deliberately different shapes. The EOD result is *not* wrapped
in a Pydantic model, because its contract is the published schema — re-describing it here
would create a second definition that can drift from the one consumers pin against. Keeping
the routers separate keeps either free to change.

**Every route on the app, so one page answers "what is served here?":**

| Route | Contract | Documented in |
|---|---|---|
| `GET /health` | run request | below |
| `GET /version` | run request | below |
| `GET /config/defaults` | run request | below |
| `POST /runs` | run request | below |
| `GET /runs/{run_id}` | run request | below |
| `POST /runs/{run_id}/cancel` | run request | below |
| `GET /runs/{run_id}/artifacts/{name}/{chunk}` | run request | below |
| `GET /eod/capabilities` | EOD | [The TraderX Path](traderx-path.md#w164--the-eod-http-routes) |
| `GET /eod/schemas/result` | EOD | [The TraderX Path](traderx-path.md#w164--the-eod-http-routes) |
| `GET /eod/schemas/capabilities` | EOD | [The TraderX Path](traderx-path.md#w164--the-eod-http-routes) |
| `POST /eod/price` | EOD | [The TraderX Path](traderx-path.md#w164--the-eod-http-routes) |
| `GET /eod/results/by-workload/{key}` | EOD | [The TraderX Path](traderx-path.md#w164--the-eod-http-routes) |
| `GET /eod/attempts/{attemptId}` | EOD | [The TraderX Path](traderx-path.md#w164--the-eod-http-routes) |

**Removed on 2026-10-18**, there being no client (A-2's revision of 2026-10-07): `POST
/portfolio/price`, `POST /portfolio/market-risk`, `POST /calibration/cam`, `POST
/calibration/lgm` and `GET /jobs/...`, and their fields `scenario_risk`, `compute_greeks`,
`cube_output`, `pnl_output`, top-level `simulation`, `greeks`, `pfe_quantiles` and `trades`. Each
route is an analytic of the run request, each flag its `analytics` list, each output its
`outputs`. The standalone calibration's own bootstrap is retired (I-94): the `calibration`
analytic is ORE's. Before that, on 2026-10-07, the API removed `POST /v2/portfolio/price`,
`schema_version: "2"` and `GET /portfolio/price/{job_id}`.

**One behavioural difference worth knowing up front.** `POST /runs` is **asynchronous** (`202`
+ a `run_id` to poll, because a 4096-scenario Monte Carlo measured ~52s — see below), while
`POST /eod/price` is **synchronous**: the TraderX path is closed-form discounted cashflows over
a handful of rows. A quick run (a calibration alone) needs one long poll (`?wait=`), not a sleep
loop.

---

## Endpoints

### `GET /health`

Liveness, and the engine worker as this API process sees it (I-77, I-106). Does no engine
work.

**Response:**
```json
{"status": "ok", "worker": {"mode": "spawn", "serving": true, "failed_starts": 0, "error": null}}
```

`serving` is whether a worker holds the host's lock for this API's queue. `error` says why no
worker can serve it, when that is known: the worker cannot start (its failed starts running, the
last exit status and the tail of its output), or the host's worker serves another queue. In
`external` mode only the second is known.

### `GET /version`

The engine's identity (`engine.run.engine_identity`) and the JAX backend of the API process.

**Response:**
```json
{"engine": {"version": "0.1.0", "revision": "9276ea0a...", "modified": false, "jax": "0.10.2",
            "jaxlib": "0.10.2", "ore": "1.8.16.0"},
 "jax_backend": "cpu (Windows)"}
```

`revision` is the git commit of the engine's source tree (`null` outside a checkout), and
`modified` whether its tracked files differed from it, in which case the revision alone does
not reproduce a result. Every run's result carries the identity of the worker that ran it.
`jax_backend` is the backend *of the API process*, which does no pricing: a served API keeps
its own JAX on the CPU (`engine.api.app.keep_jax_on_the_cpu`), so this says `cpu` on a GPU host
too; each result names the devices and backend it ran on in its precision report (I-12).

### `GET /config/defaults`

The run request's whole `config` with every default (ORE's), to copy and edit:
`RunRequestConfigSchema()` as JSON. `models`, `simulation` and `market_risk` are `null`, having
no defaults (they are the inputs of the analytics that need them); their schemas are at `/docs`.

### `POST /runs`

Body: the run request, `RunRequestSchema` ([below](#the-run-request-post-runs)). Validated
synchronously without JAX work (`RunRequestSchema.check`, which runs `engine.run.validate_run`;
the Monte Carlo scenarios are not drawn), then the body, byte for byte as received, is written to
the job queue and the route returns at once.

**Success:** `202 Accepted`
```json
{"run_id": "b3f1c2a0-..."}
```

**Refused:**

| Status | When |
|---|---|
| `422` | The body does not parse: a malformed field, an unknown field (every model refuses one, I-103: a trade carrying `hw_sigma` or a curve is not silently stripped of its model), an unknown analytic, product, format or rounding |
| `400` | The engine refuses it, the message naming the section, field or trade: a section an analytic needs missing, a section or setting no requested analytic reads, a precision setting an analytic does not read, a curve the interpolation cannot read, a trade the market cannot value, an expiry entering no whole period, a precision the rules refuse, ... ([what is checked](#what-is-checked-before-any-run)) |
| `409` | Its `idempotency_key` was submitted with another body ([idempotency](#idempotency)) |
| `503` | This host's engine worker serves another queue: this API's queue would have no worker, so nothing is queued (I-106) |

```json
{"detail": "trade 'gbp-swap' (SwapConfig): no market for currency 'GBP'; have ['USD']"}
```

A worker that cannot start never fails a submission: the run is queued and its id returned, and
its status says why it waits (I-77).

### `GET /runs/{run_id}`

The run's status, and its result once done: a read of its row in the queue. With
`?wait=N` (N at most 30 seconds), the answer comes as soon as the run is finished or N seconds
have passed, so a client long-polls instead of sleeping.

**Response:**
```json
{"run_id": "b3f1...", "kind": "run", "status": "done", "result": {"...": "RunResultSchema"},
 "error": null, "failure_class": null, "cancel_requested": false, "keep": false,
 "submitted": "2026-10-18T09:12:03.512+00:00", "started": "...", "finished": "..."}
```

| `status` | Meaning | `result` | `error`, `failure_class` |
|---|---|---|---|
| `pending` | Queued; no worker has started it | `null` | why no worker is running it, if one cannot (I-77, I-106); `null` |
| `running` | The engine worker is running it | `null` | `null` |
| `done` | Finished | the result | `null` |
| `failed` | It raised; the worker went on to the next run | `null` | the exception and traceback; the class below |
| `interrupted` | The worker stopped during it (killed, crashed, restarted). Nothing finished; submit it again | `null` | the reason; `null` |
| `cancelled` | Cancelled: before it started, or at the phase boundary named | `null` | where it stopped; `null` |

`failure_class` (I-08), from the exception's type (`engine.api.worker.failure_class`):
`missing-market-data` (a curve, currency or fixing not supplied: `KeyError`,
`MissingFixingError`), `unsupported-product` (`NotImplementedError`, an unsupported day
count), `numerical-failure` (`ArithmeticError`: overflow, division by zero, floating-point
error), `bad-terms` (any other `ValueError` or `TypeError`), `infrastructure` (anything else:
memory, XLA runtime, a result too large to store). The route validates before queueing, so
terms the engine refuses are normally a `400`, not a failed run.

`kind` is `run`; a job a version-2 queue file held (2026-10-07 to 10-18) keeps the kind of the
route it was submitted to (`portfolio`, `market-risk`), and its result that route's document.

**Unknown `run_id`:** `404 Not Found` (also once retention has purged it).

### `POST /runs/{run_id}/cancel`

Cancel the run (I-91): a `pending` run is `cancelled` at once; a `running` one is flagged
(`cancel_requested: true`) and stops at its next phase boundary — the calibration, the
simulation, each trade's pricing on the paths or the scenarios, the paired sample, the exposure,
the Greeks (each trade's), the scenarios' draw, the revaluation — and then reads `cancelled`.
Between boundaries a run is never interrupted, and the worker goes on to the next run with its
compiled programs. Answers with the run's status; a run already `cancelled` is answered as it
is, one that finished otherwise is a `409`, an unknown one a `404`. A run that reaches its end
before a boundary sees the cancel finishes `done`.

### `GET /runs/{run_id}/artifacts/{name}/{chunk}`

One chunk of an array a done run returns by reference (decision A-17,
[I-09](../planning/known-issues.md#closed)): the raw bytes, `application/octet-stream`, exactly as
the result's reference describes them. A chunk that does not exist is a `404`. The chunks are
written to the queue in the same transaction as the result, so a `done` run always has all of
them.

**An array, inline or by reference** (`engine.api.artifacts`, `engine.api.results.array_output`),
says everything needed to read it without the request:

```json
{"form": "artifact", "name": "npv_cube", "dtype": "float64", "byte_order": "little",
 "shape": [4096, 24, 8], "axes": ["scenario", "date", "trade"],
 "chunks": [{"url": "/runs/<id>/artifacts/npv_cube/0", "rows": [0, 5461], "bytes": 8387328, "sha256": "..."}, ...],
 "sha256": "<of every chunk's bytes, in order>",
 "items": {"axis": "trade", "ids": ["swap-1", "..."], "sha256": "<of the ids' canonical JSON>"}}
```

Inline, the same description with `"form": "inline"` and `values` (nested lists of `shape`) in
place of the chunks. The array is C-ordered (the last axis fastest), little-endian, split along
its first axis into chunks of whole rows of at most 8 MiB. To read it: fetch each chunk, check
its `sha256`, concatenate in order, check the whole array's `sha256`, and reshape. The trade (or
option) order travels as its own hashed record (`items`: the ids' canonical JSON,
`json.dumps(ids, separators=(",", ":"))`), so a consumer can verify the order it read is the
order published, as the EOD result's `itemOrder`. `engine.api.artifacts.read_array(reference,
fetch)` does all of it in Python; `demos/demo_api.py` does it with `hashlib` and `struct` alone.

## Why async, not sync: the measured latency

`POST /runs` returns `202 Accepted` + a run id and runs in the background, rather than blocking
the HTTP response until the run finishes. This is a direct consequence of measured timing, not a
default framework choice:

**A 4-trade, 4096-scenario portfolio took ~52 seconds wall time end-to-end**, dominated by
JAX JIT compilation (a one-time-per-shape cost that partially amortizes across repeated
calls with the same array shapes, but the first call after process start — or any call with
a new time-grid/scenario-count shape — pays it in full) plus the Monte Carlo simulation
itself. A larger, TraderX-realistic portfolio (dozens of trades, more scenarios) will only
be slower.

A synchronous HTTP response held open for tens of seconds to minutes is fragile: client and
reverse-proxy timeouts, no progress visibility, no ability to retry a request without
recomputing everything from scratch. The async pattern avoids all three at essentially no
extra infrastructure cost for this system's actual scope; the long poll (`?wait=`) gives a
quick run's client its answer without a sleep loop, and the idempotency key a retry without
recomputing.

**If you're reading this because someone wants to "simplify" `/runs` back into a single
synchronous call: re-derive this reasoning first.** The measured latency above is the reason
this exists, not a design preference — a synchronous version would need to re-solve the
timeout/retry/progress problems this pattern already avoids.

## Runs: the queue and the engine worker

Since 2026-10-04 (decision A-14; [details/precision.md §11](../planning/details/precision.md#11-execution-architecture)):

```
 API process(es)                    job queue (SQLite)                    engine worker (one per host)
 POST: validate, insert body ───▶   pending ─▶ running ─▶ done       ◀──  claim oldest, parse body, run it,
       (or the keyed run's id)         │          └─▶ failed / interrupted  store result (and artifact chunks);
 GET:  read the row, chunks  ◀───      └─▶ cancelled ◀── a phase boundary   purge expired runs between runs
 POST cancel: flag or cancel ───▶
```

- **The queue** (`engine/api/job_queue.py`) is one SQLite file, `JAX_RISK_JOB_QUEUE`. Each row
  holds the request body as received, the status, the failure class and error, the result
  document, the worker that claimed it, the number of XLA programs the run built, timestamps,
  and its idempotency key, cancel flag and `keep`; a second table holds the chunks of the arrays
  a result returns by reference, written with the result. It survives restarts of the API and
  the worker, and every API process that opens it sees the same runs, so `uvicorn --workers N`
  works: a `run_id` issued by one API process is served by any other. Its schema is version 3
  (2026-10-18); a file of version 1 or 2 is migrated in place in one `BEGIN IMMEDIATE`
  transaction, so a crash or a second process mid-migration leaves it as it was (I-106), its
  rows kept, a job of theirs that never ran marked `interrupted` (its route is gone).
- <a id="idempotency"></a>**A retried submission is the same run** (I-89). With an
  `idempotency_key` the queue records the key and the SHA-256 of the body's canonical JSON
  (every default filled in, so a retry with its fields reordered or a default spelt out is the
  same body). The same key and body return the first `run_id` (`202`, nothing queued); the same
  key with another body is a `409`, checked before anything is returned (the binding I-57 found
  missing on the TraderX path). The key is unique in the file, so two API processes racing on
  one key get one run.
- **Retention** (I-76). Between runs, at most once a minute, the worker deletes the finished
  runs (done, failed, interrupted, cancelled) that finished more than `JAX_RISK_RETENTION_DAYS`
  ago (a positive number of days, 30 by default; `off` keeps every run), with their artifacts,
  in one transaction; a run submitted with `"keep": true` is never deleted (Basel P0.9 keeps
  regulatory runs this way). A purged run's id is a `404`. SQLite reuses the freed pages; the
  file does not shrink.
- **The engine worker** (`engine/api/worker.py`, command `jax-risk-worker`) is one
  single-threaded process per host. It takes runs in submission order, parses each body
  exactly as the route did (so nothing is pickled), runs it (`engine.run.run`), and stores the
  result document, which the route then sends verbatim. It owns every device JAX sees on the
  host (a served API keeps its own JAX on the CPU) and keeps its compiled programs for its
  lifetime, so a repeated run shape compiles nothing. It also keeps them on disk, in JAX's
  persistent compilation cache, so a restarted worker reads them back instead of compiling
  again: in `JAX_COMPILATION_CACHE_DIR` if set (set it empty to turn the cache off), else in
  `xla-cache/` beside the queue file.
- **One worker per host** (decision A-14, I-106). The worker holds the host's lock
  (`JAX_RISK_WORKER_LOCK`, by default one fixed path in the temp directory) and records in it the
  queue it serves; the operating system releases the lock however the worker dies. A second
  worker on the host exits at once, whatever its queue. An API whose queue is not the one the
  host's worker serves refuses runs with a `503` naming both queues: point it at that queue, or
  give the worker of another set of devices a lock of its own.
- **The worker's device settings** (decision A-22). It runs deterministic GPU
  kernels, so a run gives the same bits on every run of its compiled program: at start-up it
  adds `--xla_gpu_exclude_nondeterministic_ops=true` to its `XLA_FLAGS`, unless they already
  name `--xla_gpu_exclude_nondeterministic_ops` or `--xla_gpu_deterministic_ops` (either
  value; the second also pins autotuning, at twice the compile time). GPU memory is the
  deployment's choice: the worker keeps JAX's default of preallocating 75% of a GPU, right for
  one worker per GPU; start the server (or `jax-risk-worker`) with
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` where other processes share the card, as the demos
  do. Both apply on a GPU only; neither changes a CPU number.
- **Failures stay in their row.** A run that raises is `failed` with its class and traceback,
  and the worker takes the next one. A worker that dies mid-run leaves the row `running`; the
  next worker to start marks it `interrupted` before claiming anything, so a run that kills
  the worker is not retried into a crash loop. A cancelled run stops with `JobCancelled`, a
  `BaseException` (as asyncio's `CancelledError` is), so no `except Exception` in the engine
  turns it into a failure.
- **Supervision**, `JAX_RISK_WORKER` (`engine/api/supervisor.py`):
  - `spawn` (default): the API starts the worker as a child process on the first run and
    checks it on every submission and poll, restarting it if it died. No thread watches it.
    With several API processes, a supervisor starts no worker while another process's worker
    holds the lock. The worker exits once the API process that started it is gone, so a
    killed server leaves no engine process behind. The child's stderr goes to
    `<queue>.worker.log`. A child that exits with an error having served no run is a failed
    start; after two running, or one that could not even be launched, a pending run's status
    and `/health` say the worker cannot start, with the tail of that log (I-77).
  - `external`: the API only reads and writes the queue; run `jax-risk-worker --queue PATH`
    under systemd (`Restart=always`), a container restart policy or a pod's process manager.
    `pip install -e .` registers the command; `python -c "from engine.api.worker import
    main; main()"` is the same thing. Its failed starts are its supervisor's to report.
- **Runs go one at a time, each on every device of the host.** A run's scenarios (the
  simulation's paths, market risk's shocks) are split across the host's devices
  (`engine/market_simulation/sharding.py`): as many devices as divide the scenario count evenly,
  at most `JAX_RISK_SCENARIO_DEVICES` if set (`1` keeps a run on one device). The result's
  precision reports list the devices, and a run on fewer devices than it has (a scenario count
  without a fitting divisor: a power of two uses them all) says so in its warnings. Priorities
  between callers are not planned until there are several callers.
- **On a TPU pod slice, one worker per host** (I-61). Start `jax-risk-worker --distributed --queue
  PATH` on every host (`JAX_RISK_WORKER=external`; the API beside process 0, whose disk holds the
  queue). The workers join one `jax.distributed` cluster, detected by JAX on a Cloud TPU VM or
  given by `JAX_RISK_COORDINATOR` (`host:port` of process 0), `JAX_RISK_NUM_PROCESSES` and
  `JAX_RISK_PROCESS_ID`. Process 0 claims each run and every host runs it, the scenarios split
  across every host's devices; a cancel stops every host at the same phase boundary. A failure
  the request determines fails the run as on one host; any other stops every worker of the set
  (their supervisor restarts them), and the restarted process 0 runs the run again once, whole:
  claimed a second time and stopped again, it is `interrupted`. Process 0 alone purges finished
  runs.

**The TraderX path keeps its own store.** `POST /eod/price` publishes every *terminal* attempt
through a crash-safe filesystem store (`engine/traderx/publication.py`), so an EOD result
survives a restart and stays addressable by `attemptId`; a *running* EOD attempt is still
memory-only ([I-08](../planning/known-issues.md#i-08),
[The TraderX Path](traderx-path.md#w164--the-eod-http-routes)).

<a id="the-run-request-post-runs"></a>
## The run request: `POST /runs`

`RunRequestSchema` mirrors `engine.run.RunRequest` (`engine/api/requests.py`):

| Field | Meaning |
|---|---|
| `market` | `asof` (ISO date; every trade is valued on it); `currencies`: per currency a `discount_curve`, `index_curves` keyed by index name (`"USD-SIMINDEX-6M"`), and `swaption_vols` (ATM normal matrix: `option_tenors`, `swap_tenors`, `vols`); `fx_spots` keyed `"EURUSD"`; `equities`. Each curve is `{"times", "rates"}`: at least one pillar, as many rates as times, every value finite, times non-negative and strictly increasing (negative rates allowed), checked when built (I-103) |
| `portfolio` | The trades, discriminated by `trade_type`: `swap`, `european_swaption`, `bermudan_swaption`, `american_swaption`, `bond`. Each names its `currency` and `index_tenor_months` and carries no model or curve. Swaptions take `settlement` (`Physical` or `Cash`) and `long_short` (`Long`, the default, or `Short`: ORE's `LongShort`) on a positive `notional`; a zero or negative option notional is a 400 naming the trade. `trade_id` on every trade or on none (none numbers them `trade-0`, `trade-1`, ...). Empty only when no analytic prices trades (calibration alone) |
| `analytics` | What to compute, each once: `npv` (today's values), `exposure` (the simulation, the cube and the exposure profiles), `sensitivities` (Delta, Gamma, Vega, Theta), `market_risk` (VaR and ES), `calibration` (the cross-asset model's calibration per currency) |
| `config` | How: one section per ORE file ([below](#config-one-section-per-ore-file)) |
| `outputs` | How the result carries each large array ([below](#outputs)) |
| `idempotency_key` | Optional: a retried submission with the same key and body is the same run ([idempotency](#idempotency)) |
| `keep` | Keep the run, its result and its artifacts whatever the retention period (default `false`) |

Representational differences from the dataclasses (SWIG-bound `ORE` types are not natively
Pydantic-serializable): dates are ISO strings (`"2026-07-30"`), periods are ORE strings
(`"5Y"`, `"18M"`), historical `fixings` are `{"YYYY-MM-DD": rate}`. A trade's schedule is
given **either** as `effective_date`/`maturity_date` (plus `exercise_date` for a
`european_swaption`) **or** as `swap_tenor` (plus `forward_start`/`exercise_lag_days`),
resolved on the market's date. A live European's expiry must leave a whole period of each leg to
enter, else it is a 400 naming the trade and the dates (I-103). A `"bond"` is a Treasury bill
(no `coupon_schedule`) or note; its `face_amount` is signed.

### `config`: one section per ORE file

Every section is optional; `GET /config/defaults` returns them all with ORE's defaults.

| Section | ORE | Meaning | Read by |
|---|---|---|---|
| `base_currency` | `ore.xml`'s `baseCurrency` | The reporting currency every amount is in, and the simulation's domestic currency; default `"USD"`. A trade with no spot into it in `market.fx_spots` is a 400 | every analytic but `calibration` |
| `models` | `simulation.xml`'s `CrossAssetModel` | `ir` per currency (`model`: `"LGM"`, the default, or `"HullWhite"`; `reversion`, `volatility`, an optional calibration basket `calibration_expiries` × `calibration_terms`, `swap_index`, and the bootstrap's root `solver`: `"Newton"`, the default, or `"Bisection"`), `fx_volatilities` (one per currency but `base_currency`), `equity_volatilities`, `correlations` between factors `IR:USD`, `FX:EURUSD`, `EQ:SP5`, `bootstrap_tolerance` (ORE's `BootstrapTolerance`, one for the model, default `1e-4`). Every volatility is a number, or piecewise constant: `{"times": [1.0, 3.0], "values": [0.008, 0.011, 0.009]}`, `values[i]` on bucket `i` of `[0, times[0]), ..., [times[-1], inf)` (ORE's `VolatilityTimes`/`Volatility`) | `exposure` (all of it), `calibration` (each calibrated currency's model and the tolerance) |
| `simulation` | `simulation.xml`'s `Parameters` and `Market` | `dates` (the grid, after the as-of date), `samples`, `seed`, `replicates` (independent scrambles, `samples` divided equally), `curve_tenors` (the scenario market's), `swaption_vol_decay` (`ForwardVariance` or `ConstantVariance`) | `exposure` |
| `pricing` | `pricingengine.xml` | `european` (`"Bachelier"`, the default, or `"Jamshidian"` with `jamshidian: {"reversion", "volatility", "solver"}`), the `bermudan` and `american` engines (`reversion`, `volatility`, `calibration`, `strategy`, `reference_calibration_grid`, `bootstrap_tolerance` (ORE's `Tolerance`, default `1e-4`), `shift_horizon` (ORE's `ShiftHorizon`, default `0.5`; any value `>= 0`), `n_per_std`, `std_devs`, `exercise_time_steps_per_year`, `swap_index`, and `solver`, that of the calibration and of each helper's exercise boundary). In the exposure simulation each option keeps the LGM calibrated where ORE builds the trade (A-24); the sensitivities and market risk recalibrate a calibrated option under every shift and scenario | `npv`, `exposure`, `sensitivities`, `market_risk` |
| `sensitivity` | `sensitivity.xml`, and the Greeks method | `method` (`"Bump"`, ORE's sensitivity analysis, the default, or `"AD"`), `curve_tenors`, `curve_shift`, `vol_shift`, `theta_days` | `sensitivities` |
| `exposure` | the exposure analytic | `pfe_quantiles`, each in (0, 1) and given once; default `[0.95, 0.99]` | `exposure` |
| `market_risk` | the market-risk analytic | `scenarios` ([below](#market-risk-scenarios)), `quantiles` (VaR/ES confidence levels in (0, 1), each once; default `[0.99, 0.975]`), `batch_size` (scenarios revalued at once per trade, lowered automatically for a large Bermudan; default 256) | `market_risk` |
| `precision` | not ORE's: the research axis | storage, compute and accumulate formats per stage ([below](#precision-control-precisionschema)) | `exposure`, `market_risk` (each only the settings it reads) |

Every `solver` is [the root solver](calibration.md#the-root-solver) (decision A-21).

<a id="what-is-checked-before-any-run"></a>
### What is checked before any run

`engine.run.validate_run`, synchronously, with no JAX work:

- **The analytics**: at least one, each from the list, each once.
- **What each analytic needs**: `exposure` the `models` and `simulation` sections, `market_risk`
  its section, `calibration` `models` with at least one currency with a basket (and the
  market's curves and swaption volatilities for each calibrated one).
- **Nothing accepted and then ignored** (I-59's rule): a section, a portfolio, or a setting
  that differs from its default where no requested analytic reads it (the table's last column)
  is a 400 naming it, and so is an `outputs` setting other than its default for an analytic not
  requested. Calibration alone reads only the calibrated currencies' models and the tolerance:
  a currency without a basket, FX and equity volatilities and correlations are the simulation's.
- **Precision, per analytic** (I-95). Today's values, Greeks and calibration are float64 by
  decision (A-10), so a precision other than float64 is a 400 unless `exposure` or `market_risk`
  is requested. `exposure` reads every setting. `market_risk` has no scenario market and its
  shifts no products: it refuses a `market` stage other than float64 and a `simulation.compute`
  other than its `accumulate` format, even beside `exposure`, so its precision report never
  states a policy it did not run. To run both at another market-stage precision, submit two
  runs.
- **Then each analytic's own checks**: the portfolio pipeline's (`validate_request`: the
  cross-asset model built from `models`, `simulation` and `base_currency`, every trade the
  market and its engine can value, the PFE quantiles, the precision overrides naming trades),
  market risk's (`validate_portfolio`, `validate_scenarios`: the factors are market curves and
  cover every trade's, the quantiles, the batch size), on the scenarios not yet drawn.

### `outputs`

How the result carries each large array (decision A-17, revised with this contract: the
reference became the default of the cube and the P&L):

| Field | Array | Default |
|---|---|---|
| `npv_cube` | the exposure analytic's NPV cube `[scenario, date, trade]` | `"artifact"` |
| `exercised` | each option's exercise decisions `[scenario, date, option]`, bool: whether the holder has exercised on the path on or before the date (decision A-24) | `"none"` |
| `pnl` | the market-risk P&L `[scenario, trade]` | `"artifact"` |

Each is `"artifact"` (a chunked, hashed reference read at
[`GET /runs/{run_id}/artifacts/...`](#get-runsrun_idartifactsnamechunk)), `"inline"`, or
`"none"` (left out; ORE too writes its cube only when asked, `cubeFile`). `npv_cube: "none"`
also avoids allocating the float64 output cube: exposure reduces stored columns individually,
retaining exact sorted PFE and paired precision estimates (I-105).

A minimal body (one swap under the Hull-White model, today's value and exposure, the cube
inline):

```json
{"market": {"asof": "2026-07-30", "currencies": {"USD": {
     "discount_curve": {"times": [0, 1, 5, 30], "rates": [0.03, 0.03, 0.04, 0.05]},
     "index_curves": {"USD-SIMINDEX-6M": {"times": [0, 1, 5, 30], "rates": [0.034, 0.034, 0.044, 0.052]}}}}},
 "portfolio": [{"trade_type": "swap", "trade_id": "swap-1", "notional": 1e7, "fixed_rate": 0.042,
                "payer": true, "swap_tenor": "5Y"}],
 "analytics": ["npv", "exposure", "sensitivities"],
 "config": {"models": {"ir": {"USD": {"model": "HullWhite", "reversion": 0.03, "volatility": 0.01}}},
            "simulation": {"dates": ["2027-07-30", "2028-07-30"], "samples": 1024}},
 "outputs": {"npv_cube": "inline"}}
```

The calibration alone:

```json
{"market": {"...": "..."}, "analytics": ["calibration"],
 "config": {"models": {"ir": {"USD": {"model": "LGM", "reversion": 0.03,
                                      "calibration_expiries": ["1Y", "2Y", "5Y"],
                                      "calibration_terms": ["5Y", "4Y", "1Y"]}}}}}
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
the stochastic rounding (2026-10-02). `offset` (`"block"`, the default, or `"none"`) is the
layout of such a format: with `"block"` a block of 32 paths whose values sit close together
against their level (one sign, the farthest at most three times the nearest) is kept as its
deviations from its midpoint, beside its scale, so it keeps its spread (I-75, 2026-10-09); `"none"` stores the
values themselves. `paired_fraction` (in [0, 1], 0 by default; outside it a
`422`) is the share of paths also run at float64, whose estimates the result's precision
reports carry (2026-10-02). Unknown fields are refused.

In market risk the `simulation` stage stores the shifts (its `storage` and `accumulate`
formats) and each trade's pricing stage its revaluation and P&L; there is no `market` stage
([what is checked](#what-is-checked-before-any-run)).

Each stage also accepts `compute_rounding: "nearest" | "stochastic"` (default `"nearest"`).
Stochastic compute rounding is supported for single-grid formats below float32, and uses
`rounding_seed` in streams separate from storage rounding. Keys are traced data: changing
the seed reuses compiled programs. Float32, float64 and bfloat16 passes refuse this setting;
market risk also refuses it on `simulation` and `market`, which have no products there.

`config.simulation.replicates` (default 1) divides `samples` into equal independent scrambled
Sobol draws. Set at least 8 for practical uncertainty estimates, and use a power of two per
scramble for Sobol balance. Paired/scaled-storage runs need
each replicate's path count to be a multiple of 32. The paired sample takes a whole-block
prefix of every scramble. A precision report's `replicates` and `uncertainty_method` describe
the result. Paired mean errors use the independent replicate estimates, including covariance;
one scramble produces `null` standard errors and a warning. A report's `interval` applies
to all figures and entries: `status: "unavailable"`, `reason: "no_validated_tail_bound"`,
with `confidence_level`, `lower` and `upper` all `null`. Market risk repeats this object
in `uncertainty.interval`. Raw errors are variance diagnostics and cannot certify
nominal coverage, even when zero. More scrambles do not enable nominal intervals.
See [Sampling uncertainty](sampling-uncertainty.md).

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

Compute below 32 bits (2026-10-10), the pricing stage's matrix products in FP8, accumulated in
float32, the Bermudans kept in float32:

```json
"precision": {"pricing": {"storage": "float8_e4m3fn", "compute": "float8_e4m3fn", "accumulate": "float32"},
              "by_product": {"bermudan_swaption": {"storage": "float32", "compute": "float32", "accumulate": "float32"}}}
```

A `by_product` key that is not a `trade_type` is a `422`; a `by_trade` key that names no
trade of the request is a `400` (checked with the request, so no run is created).

The format names are the format table's (`float64`, `float32`, `float16`, `bfloat16`,
`float8_e4m3fn`, `float8_e5m2`, and the product formats `tensorfloat32`, `bfloat16_3x` and
each format's single pass `float16_1x`, `bfloat16_1x`, `float8_e4m3fn_1x`, `float8_e5m2_1x`,
`tensorfloat32_1x`, compute only); another name, rounding or layout is a `422`. Below float32
a product carries the operand every path shares in two terms of the format; the single pass
rounds it once, a bias more paths do not remove, and is there to be measured (decision A-23). `compute` is the format of a
stage's matrix products, `accumulate` the format they accumulate in and the rest of the stage
runs in: float32 or float64, at least as wide as `compute`, and float32 under a compute format
below float32; `storage` no wider than `accumulate` and not a product format. Every product
takes every compute format (the Bermudan/American rollback since 2026-10-13). An inconsistent
stage is a `400` naming the stage (or override, e.g. `config.precision.by_trade['swap-7']`), the
field and the rule. So is `"rounding": "stochastic"` or `"offset": "none"` when no stage
stores below 32 bits:

```
POST /runs
{"config": {"precision": {"pricing": {"storage": "float8_e4m3fn", "compute": "float16", "accumulate": "float16"}}}, ...}
-> 400 {"detail": "config.precision.pricing: StagePrecision.accumulate='float16': narrower than float32: no matrix
        unit accumulates narrower; float32 or float64"}
```

A precision report carries `products`: each compute format the run used, and whether the
worker's device ran it natively or the engine emulated it (`"native"`, `"emulated"`; a CPU
emulates every format below float32).

The 32/64 shape before 2026-10-01 (`{"simulation": 32, "pricing": 64, "risk": ..., "calibration": ...}`)
is a `422` whose message names the replacement (decision A-12); it is not translated.

**Concurrency note:** runs of every precision go to the one engine worker, in submission
order. Each run's result is independent of what was queued with it: runs queued together
give the bits they give run one after another (`tests/test_engine_worker.py`).

<a id="market-risk-scenarios"></a>
### Market-risk scenarios

`config.market_risk.scenarios.factors` names the risk factors, the pillar zero rates of market
curves, in the order of the covariance's or the history's columns: each curve's labels
`"<curve>/<pillar time>y"` for every pillar in the market's order, curve after curve, e.g.
`"discount:USD/1y"`, `"index:USD-SIMINDEX-6M/1y"` (`engine.risk.market.RateRiskFactors.labels`).
Any curves of the market, in any order; every curve a trade reads must be among them. A label
out of its curve's pillar order, a curve the market lacks, or a trade's curve left out is a
`400` naming the expected labels.

- `{"source": "monte-carlo", "factors", "covariance", "horizon_days", "num_scenarios", "seed",
  "replicates"}`: zero-mean Gaussian moves over the horizon with the given `[F][F]` covariance
  (symmetric positive semi-definite), scrambled Sobol normals (`MonteCarloScenarios`), checked at
  validation and drawn by the engine worker when the run starts.
- `{"source": "historical", "factors", "history", "horizon_days", "dates"}`: one scenario per
  overlapping `horizon_days` window of a `[D][F]` history of the factors' levels, oldest first;
  `dates`, optional, label each window (`historical_scenarios`).

Monte Carlo scenarios' `replicates` (default 1) has the same equal-size and whole-block pairing
requirements as the exposure simulation's. ES standard errors recompute the pooled tail after
deleting each independent scramble; with one scramble they are null. Historical scenarios accept
`bootstrap_replicates` (default 128, at least 2), `bootstrap_seed` (default 42), and
`block_length` (horizon windows per moving block; default max(horizon_days,
ceil(num_scenarios ** (1/3)))). An explicit block length must be between the horizon and
scenario count. Fewer than two blocks gives unavailable error. These are conditional bootstrap
estimates, dependent on stationarity and block length; they do not measure forecast/model
uncertainty. `uncertainty` echoes the method and settings.

```json
"market_risk": {"scenarios": {"source": "monte-carlo", "factors": ["discount:USD/0y", "discount:USD/1y", "..."],
                              "covariance": [[6.4e-6, "..."], "..."], "horizon_days": 10, "num_scenarios": 16384,
                              "seed": 1},
                "quantiles": [0.99, 0.975]}
```

## The result: `RunResultSchema`

One document for every analytic of the run, which says what each figure is without the request
beside it (I-87; the project audit's AUD-39). Mirrors `engine.run.RunResult`:

| Field | Meaning |
|---|---|
| `run_id`, `analytics` | The run, and what it computed |
| `asof`, `base_currency` | The market's date and the reporting currency every amount is in |
| `engine` | What computed it: the engine's version, git `revision` and `modified`, and the `jax`, `jaxlib` and `ore` versions (`GET /version`'s object, read in the worker that ran it) |
| `config` | The configuration as run, every default filled in: the request's `config` as the API parsed it, so a result reproduces itself |
| `trades` | One row per trade, in the request's order, which is also every array's trade axis: `trade_id`, `product` (its `trade_type`), `npv` (its value today on today's market, with `npv`, `exposure` or `sensitivities`; `null` for market risk alone, whose values on today's curves are its section's), `exposure` (its standalone profile, with `exposure`) and `greeks` (with `sensitivities`). Each Greek is flattened row-major into `values`; one of more than one dimension (`vega:<ccy>`, option tenors × swap tenors) also has its shape in `shapes`; `theta` alone |
| `npv` | The portfolio's value today, the sum of the rows' (with a portfolio analytic) |
| `exposure` | With `exposure`: `dates` (the simulation dates, the cube's date axis), `netting_set` (the whole portfolio, no collateral: `times`, `epe`, `ene`, `ee_b`, `eee_b`, `pfe`, `epe_b`, `eepe_b`, `basel_epe`, `basel_eepe`; every list indexed like `times`, whose first entry is the as-of date, where each trade is valued on the simulation market as ORE's cube starts, the rest `dates`), `npv_cube` and `exercised` (as `outputs` asked), `measure` (`risk-neutral-pricing`, I-11), `precision` (the report below), `pricing_settings` (each product's engine, grid and calibration policy: `model=fixed`, the model calibrated when the trade was built, A-24). See [Exposure](../risk/exposure.md) |
| `sensitivities` | With `sensitivities`: `method` (`Bump` or `AD`), `factors` (each Greek's factor labels in its row's flattened order: the curve tenors for `delta:<curve>` and `gamma:<curve>`, `"<option tenor>x<swap tenor>"` for `vega:<ccy>`), `pricing_settings` (`model=recalibrated`: a calibrated option recalibrated under every shift). See [Greeks](../risk/greeks.md) |
| `market_risk` | With `market_risk`: `base_npv` and `trades` (each trade's value on today's curves, as revalued), `risk` (`VaR_<q>`, `ES_<q>` as positive losses, `ES_<q>_tailCount`, `ES_<q>_standardError`, per quantile label, the level's percentage exactly: `99`, `97.5`, `95.00000001`; NaN as `null`), `measure` (`historical-forecast`), `source`, `horizon_days`, `num_scenarios`, `risk_factors` (the factors shocked, in the scenarios' order), `pnl` (as `outputs` asked; the portfolio P&L the statistics use is each scenario's sum), `precision`, `uncertainty` (Monte Carlo: `method` `"replicate_jackknife"` or `"unavailable"` and `replicates`; historical: `"moving_block_bootstrap"`, `replicates`, `block_length` and `seed`), `pricing_settings` (`model=recalibrated`: a calibrated option recalibrated under every scenario, I-112). See [Market Risk](../risk/market-risk.md) |
| `calibration` | With `calibration`: per calibrated currency, `model`, the `reversion` it was calibrated with, the volatility in its model's parametrization (the LGM's alpha, or the Hull-White short rate's sigma: `sigma_values[i]` on bucket `i` of `[0, sigma_times[0]), ..., [sigma_times[-1], inf)`), each helper's `expiry`, `term`, `market_value` (Bachelier on the ATM volatility) and `model_value`, and the calibration's `error` as ORE measures it against `config.models.bootstrap_tolerance` (the root mean square of the helpers' relative price errors). The calibration a run's exposure analytic simulates with. See [Calibration](calibration.md) |
| `warnings` | Every analytic's: at a precision below float64, each figure without a passing row in the precision evidence table (with what is validated for it and up to how many paths); the AD Greeks of a grid-priced option; market risk's held volatilities and FX, a thin tail, unavailable or conditional uncertainty |

**The precision report** (`precision` of `exposure` and `market_risk`) is built in the worker
that ran the run, so its `devices` and `backend` are the worker's, not those `/version` names
([I-12](../planning/known-issues.md#i-12), closed): `policy` (the request's `precision`, defaults
filled in), `trades`, `realized` (the format of every stored array), `products`, `devices`,
`backend`, `jax_version`, `paths`, `paired_paths`, `figures` (each with `"kind": "mean"` or
`"quantile"`; NaN as `null`), `replicates`, `uncertainty_method`, `interval`. See [The Portfolio
Entry Point](portfolio-entrypoint.md#precision).

## Example: a Python `requests` session

```python
import requests

BASE = "http://127.0.0.1:8000"

body = {
    "market": {"asof": "2026-07-30", "currencies": {"USD": {
        "discount_curve": {"times": [0.0, 1.0, 2.0, 5.0, 10.0, 30.0], "rates": [0.030, 0.030, 0.034, 0.040, 0.046, 0.050]},
        "index_curves": {"USD-SIMINDEX-6M": {"times": [0.0, 1.0, 2.0, 5.0, 10.0, 30.0],
                                             "rates": [0.034, 0.034, 0.038, 0.044, 0.049, 0.052]}}}}},
    "portfolio": [
        {"trade_type": "swap", "trade_id": "swap-1", "notional": 1_000_000.0, "fixed_rate": 0.036,
         "payer": True, "swap_tenor": "2Y"},
    ],
    "analytics": ["npv", "exposure"],
    "config": {"models": {"ir": {"USD": {"model": "HullWhite", "reversion": 0.03, "volatility": 0.01}}},
               "simulation": {"dates": ["2027-01-30", "2027-07-30", "2028-07-30"], "samples": 4096},
               "exposure": {"pfe_quantiles": [0.95, 0.99]}},
    "outputs": {"npv_cube": "none"},
    "idempotency_key": "eod-2026-07-30-book-7",
}

r = requests.post(f"{BASE}/runs", json=body)
r.raise_for_status()
run_id = r.json()["run_id"]
print("submitted run", run_id)

while True:
    data = requests.get(f"{BASE}/runs/{run_id}", params={"wait": 30}).json()  # a long poll
    if data["status"] not in ("pending", "running"):
        break

if data["status"] != "done":
    raise RuntimeError(data["error"])

result = data["result"]
print("NPV today:", result["npv"], result["base_currency"])
print("per trade:", {row["trade_id"]: row["npv"] for row in result["trades"]})
print("PFE 95% at each date:", dict(zip(["asof"] + result["exposure"]["dates"],
                                        result["exposure"]["netting_set"]["pfe"]["PFE_95"])))
```

See a curl-only version in [User Guide: Running the API](../getting-started/user-guide.md#running-the-api),
and `demos/demo_api.py` for every analytic, the cube by reference and the idempotency key.

## Design: one request for every analytic

Decided 2026-09-30 ([compliance/decisions.md](../../compliance/decisions.md) A-2; ORE
alignment plan 9.2); every setting reachable on 2026-10-07; one request on 2026-10-18 (I-87,
[details/configurable-engine.md](../planning/details/configurable-engine.md#one-run-request-i-87-i-88-i-89)):

- **One route, one request.** What to compute is the `analytics` list; how is the `config`,
  one section per ORE file, each setting in one place (the model per currency is
  `config.models`, which calibration and exposure both read; the reporting currency is
  `config.base_currency`). Defaults are ORE's. Each analytic runs the engine's own entry point,
  so a run of one analytic is that entry point's direct call, and a run of several each alone;
  sharing work between analytics in one run (the cross-asset model's calibration, today's
  values) is an optimization not yet made ([I-118](../planning/known-issues.md#i-118)).
- **Every setting reachable, and arriving.** `tests/test_api_completeness.py` walks every
  configuration type from `RunRequest` and fails on a field without an API field, a type
  without a schema, or an exemption naming no field; and sets every field of every schema to a
  value other than its default, failing unless the value changes its own setting and nothing
  else (I-88).
- **Robust.** Validated before any run starts: types, unknown fields refused, cross-field
  checks, nothing accepted and then ignored, each refusal a `400` or `422` naming its field. A
  retried submission is the same run. A run can be cancelled; a worker that cannot start is
  reported.
- **Names say what they are.** No route or field is named like a version unless it marks a
  revision of the contract itself. `/v2` and `schema_version: "2"` went on 2026-10-07.
- **Every figure says what it is.** One row per trade with its `trade_id`
  ([I-10](../planning/known-issues.md#closed)); every array names its axes and carries its trade
  order hashed; every section its pricing settings; the result its configuration as run and the
  engine that ran it.

## Tested by

- `tests/test_api.py` (FastAPI's `TestClient`, no running server): `/health`, `/version` (the
  engine's identity) and `/config/defaults`; over the worker, the result equals `engine.run.run`
  on the same request (the Hull-White model and the Jamshidian engine reaching it), says what each
  figure is, and runs back to back keep their own results; `wait`; the calibration analytic is
  the cross-asset model a run simulates with; the routes of 2026-10-07 are gone; every refusal
  (`422`s for bodies the schema refuses, `400`s naming the section, field or trade); precision on
  the wire.
- `tests/test_api_portfolio.py`: on the shared test portfolio (`tests/support/portfolio.py`),
  the result equals the in-process run: values, cube, the exposure profiles including
  EPE_B/EEPE_B and Basel, each trade's row; the Greeks with their factor labels and the 2-D Vega;
  the cube and the exercise decisions by reference are the inline arrays, their hashes checked,
  and `"none"` leaves them out; the `400`s; the worker's precision report.
- `tests/test_api_market_risk.py`: the market-risk result equals the in-process run for Monte
  Carlo and historical scenarios, the P&L by reference is the inline P&L, each request the run
  would refuse is a `400` before any run (factor order, an unknown curve, a trade's curve left
  out, a covariance that is not PSD or of the wrong shape, a short history, a quantile, no trades,
  the batch, a precision setting market risk does not read), and the route draws no scenario.
- `tests/test_run_request.py`: in Python, each analytic is its entry point's direct call, bit
  for bit, several are each alone and compile nothing more, every check before any work, the
  phases as boundaries.
- `tests/test_api_completeness.py`: every setting has an API field and every field's value
  arrives in its own setting (I-88); the run request's mapping onto the entry points, field by
  field.
- `tests/test_engine_worker.py`: the queue (its migration in one transaction, idempotency,
  cancelling, retention), the host's lock and the queue it records, the worker loop (failures,
  cancels at phase boundaries of a stub and of a real run, the purge), the supervisor's reports of
  a worker that cannot start, the routes over the queue (status, artifacts, `409`, `503`, cancel,
  `wait`), real runs in the worker process.
- `tests/test_api_artifacts.py`: the artifact format rebuilds an array bit for bit and refuses a
  tampered chunk, order or array. `tests/test_input_checks.py`: the inputs refused before any
  work (I-103).
