# HTTP API

**Modules:** [`engine/api/app.py`](../../engine/api/app.py),
[`engine/api/routes.py`](../../engine/api/routes.py),
[`engine/api/market_schemas.py`](../../engine/api/market_schemas.py) (the portfolio request),
[`engine/api/schemas.py`](../../engine/api/schemas.py) (shared schemas and the result)

## Plain-language summary

Everything described in [The Portfolio Entry Point](portfolio-entrypoint.md) is reachable
from plain Python already — `price_portfolio(request)`. This module wraps that same
function behind an HTTP API, for a caller (like TraderX — see
[TraderX integration](../planning/details/traderx-integration.md)) that isn't a Python process
sharing this codebase's own memory space.

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
| `POST /v2/portfolio/price` | portfolio (the same request; an older name) | below |
| `GET /portfolio/price/{job_id}` | portfolio | below |
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

**One behavioural difference worth knowing up front.** `POST /portfolio/price` is
**asynchronous** (`202` + a `job_id` to poll, because a 4096-scenario Monte Carlo measured
~52s — see below), while `POST /eod/price` is **synchronous**: the EOD path is closed-form
discounted cashflows over a handful of rows and returns the priced result in the response.
The two contracts differ here on purpose, not by accident of implementation order.

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

**Known nuance, not solved in this phase:** `jax_backend` reports `jax.default_backend()`
*of the dispatcher process itself*, which does no JAX work (see `engine/api/routes.py`'s
module docstring) — it does not necessarily reflect the actual device a given
`/portfolio/price` job ran on, since that job's real work happens inside a separate
`engine.portfolio.worker_pool` worker process with its own independent JAX runtime. On this
single-`CpuDevice` dev machine dispatcher and worker report the same backend, so the
distinction is invisible; on a real multi-TPU-chip host it would not be.

### `POST /portfolio/price`

The main endpoint. Body: the portfolio request, `MarketPortfolioRequestSchema` (mirrors
`PortfolioRequest` and its `RunConfig` — see "Request schema" below). Validates
synchronously (`engine.portfolio.market_path.validate_request`, no JAX work), then submits
the pricing to the worker pool and returns immediately.

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

**The Hull-White request shape retired by roadmap 1.3** (a `SimulationConfig` market with
`time_grid`/`rates`/`joint_covariance`, model parameters on the trades, a
`calibration_basket`, a top-level `evaluation_date`) is a `422` whose message names its
replacement: the Hull-White model is `"model": "HullWhite"` per currency in `simulation.ir`.

### `POST /v2/portfolio/price`

The same request and behaviour as `POST /portfolio/price`, polled at the same
`GET /portfolio/price/{job_id}`. Until roadmap 1.3 `/portfolio/price` took the Hull-White
model's request and `/v2` the market path's; the `/v2` and the body's optional
`schema_version: "2"` are historical names, not versions. Roadmap 4.1 keeps one route (see
[Target: one configurable API](#target-one-configurable-api)).

### `GET /portfolio/price/{job_id}`

Poll for a job's status/result.

**Response:**
```json
{"status": "done", "result": { "...": "PortfolioResultSchema" }, "error": null}
```

`status` is one of `pending` / `running` / `done` / `failed`. `result` is `null` until
`status == "done"`. `error` is `null` unless `status == "failed"`, in which case it carries
the exception message and traceback. `pending` currently covers both "genuinely queued
behind the worker pool" and "actively running in a worker process" — the worker
can't cheaply report its own sub-states back to the dispatcher without a mechanism this
phase doesn't build (see `engine/api/routes.py`'s `get_portfolio_price` docstring); a
`done`/`failed` job's error message/traceback come from
`concurrent.futures.Future.result()` re-raising the worker-side exception, which
`concurrent.futures.process` automatically annotates with the full remote (worker-process)
traceback.

**Unknown `job_id`:** `404 Not Found`.

### `POST /calibration/lgm`

Standalone calibration — wraps `engine.calibration.basket.build_coterminal_basket` +
`engine.calibration.lgm.calibrate_lgm_sigma`: a Hagan bootstrap of an LGM `Sigma` to a
caller-given co-terminal basket. Synchronous (a bootstrap bisection, not a Monte Carlo
simulation). It is not the portfolio's calibration, which is the cross-asset model's per
currency and each option's own basket ([Calibration](calibration.md)); a route for those is
roadmap 4.1 ([I-56](../planning/known-issues.md#i-56)).

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
  "hw_a": 0.03
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

`POST /portfolio/price` returns `202 Accepted` + a job id and prices in the background,
rather than blocking the HTTP response until pricing finishes. This is a direct consequence
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

## Job store: in-process `job_id -> Future` table, over a multi-process worker pool

**This section describes a genuine architecture change**, not a terminology fix. The job
store backing `GET /portfolio/price/{job_id}` is still a plain in-process Python `dict` in
the dispatcher (`engine/api/routes.py`'s `_JOBS`) — that part hasn't changed. What changed
is what it maps to and where the actual pricing work runs: `_JOBS[job_id]` now holds a
`concurrent.futures.Future`, returned by `engine.portfolio.worker_pool.submit_pricing_job`,
whose underlying `price_portfolio` call executes in a separate OS process — one of a fixed
pool of worker processes, each with its own independent JAX/XLA runtime (until roadmap 1.4
there was one pool per simulation precision). Polling
`GET /portfolio/price/{job_id}` now checks `future.done()`/`future.result()` instead of
reading fields a background thread mutated directly, but the response shape/status values
are unchanged. See [Architecture: Concurrency](../concepts/architecture.md) and
`engine/portfolio/worker_pool.py`'s own module docstring for the full mechanism. Roadmap 1.8
replaces the pool with one engine worker process per host behind a durable job queue
([I-72](../planning/known-issues.md#i-72)).

There remain **two separate, distinct motivations for a future shared store (Redis, a
database table)**, worth keeping apart:

1. **HTTP-scaling to multiple uvicorn worker processes.** Running more than one uvicorn
   worker still means each worker process has its own, mutually invisible `_JOBS` dict (and
   its own separate `engine.portfolio.worker_pool` pools underneath it) — a `job_id`
   returned by one uvicorn worker would still 404 against another. This motivation is
   **still deferred, still out of scope** — nothing in this phase changes it; it's a
   question about the *HTTP/dispatcher* layer's own process count, one level above the
   pricing worker pool.
2. **Concurrent pricing.** This was the *other* reason a shared store might once have
   seemed necessary. It is solved one layer down, by `engine.portfolio.worker_pool`'s
   process pool under a single dispatcher, not by Redis or a database.

**The EOD path already has the durable store this section defers.** Since W0.8,
`POST /eod/price` publishes every *terminal* attempt through a crash-safe filesystem store
(`engine/integration/publication.py`), so an EOD result survives a restart and stays
addressable by `attemptId`. That work is deliberately **EOD-only**: `_JOBS` above is
untouched, and a `job_id` from `/portfolio/price` is still lost on restart. The two paths
have different durability guarantees today, which is a real difference a caller needs to
know rather than an inconsistency to gloss over — see
[I-08](../planning/known-issues.md#i-08) and
[EOD Integration](eod-integration.md#w164--the-eod-http-routes).

## Request schema: `MarketPortfolioRequestSchema`

Mirrors `engine.portfolio.PortfolioRequest` and its run configuration
(`engine/api/market_schemas.py`):

| Field | Meaning |
|---|---|
| `market` | `asof` (ISO date; every trade is valued on it); `currencies`: per currency a `discount_curve`, `index_curves` keyed by index name (`"USD-SIMINDEX-6M"`), and `swaption_vols` (ATM normal matrix: `option_tenors`, `swap_tenors`, `vols`); `fx_spots` keyed `"EURUSD"`; `equities` |
| `trades` | Discriminated by `trade_type`: `swap`, `european_swaption`, `bermudan_swaption`, `american_swaption`, `bond`. Each names its `currency` and `index_tenor_months` and carries no model or curve. Swaptions take `settlement` (`Physical` or `Cash`). `trade_id` on every trade or on none (none numbers them `trade-0`, `trade-1`, ...) |
| `simulation` | `RunConfig.simulation`: ORE's `simulation.xml` as `CamConfigSchema`: `dates`, `base_currency`, `ir` per currency (`model`: `"LGM"`, the default, or `"HullWhite"`; `reversion`, `volatility`, optional calibration basket `calibration_expiries` × `calibration_terms`, `swap_index`), `fx_volatilities`, `equity_volatilities`, `correlations` between factors `IR:USD`, `FX:EURUSD`, `EQ:SP5`, `curve_tenors`, `samples`, `seed`, `swaption_vol_decay`. Required with `scenario_risk` |
| `pricing` | `RunConfig.pricing`: `european` (`"Bachelier"`, the default, or `"Jamshidian"` with `jamshidian: {"reversion", "volatility"}`), the `bermudan` and `american` engines (`LgmEngineSchema`), and `recalibrate` (default `true`, as ORE's `ValuationEngine`) |
| `greeks` | `RunConfig.greeks`: `method` (`"Bump"`, the default, or `"AD"`) and `sensitivity` (ORE's `sensitivity.xml`: `curve_tenors`, `curve_shift`, `vol_shift`, `theta_days`, `swaption_vol_decay`) |
| `base_currency` | The reporting currency. Omitted: the simulation's base currency, or USD without a simulation. One contradicting the simulation's is a 400 |
| `precision` | `RunConfig.precision`, see below |
| `pfe_quantiles` | Quantiles of the PFE profiles. Default `[0.95, 0.99]` |
| `compute_greeks` | Default `false` |
| `scenario_risk` | Default `true`; `false` prices today's values (and Greeks) only |
| `schema_version` | Optional, `"2"` only: a historical name, not a version |

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
is priced at its `by_trade` entry, else its product's, else `pricing` (roadmap 1.5).
`rounding` (`"nearest"`, the default, or `"stochastic"`) is how values are rounded into a
storage format below 32 bits, and `rounding_seed` (a non-negative integer, 0 by default) seeds
the stochastic rounding (roadmap 1.6). Unknown fields are refused.

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
stores; a format not yet enabled for compute (below float32, roadmap 2.8), or an inconsistent
stage, is a `400` naming the stage (or override, e.g. `precision.by_trade['swap-7']`), the
field and the roadmap step that enables it. So is `"rounding": "stochastic"` when no stage
stores below 32 bits:

```
POST /portfolio/price
{"precision": {"pricing": {"storage": "float8_e4m3fn", "compute": "float16", "accumulate": "float16"}}, ...}
-> 400 {"detail": "precision.pricing: StagePrecision.compute='float16': compute in float16 is enabled by
        roadmap step 2.8 (difference-form kernels); until then float64 or float32"}
```

The 32/64 shape before roadmap 1.4 (`{"simulation": 32, "pricing": 64, "risk": ..., "calibration": ...}`)
is a `422` whose message names the replacement (decision A-12); it is not translated.

**Concurrency note:** jobs of every precision share one pool of worker processes; jobs
beyond the pool's size queue for a free worker. Each job's result is independent of what
else is running.

## Target: one configurable API

Decided 2026-09-30 ([compliance/decisions.md](../../compliance/decisions.md) A-2; ORE
alignment plan 9.2; [I-56](../planning/known-issues.md#i-56)):

- **One route, one request.** The request's configuration selects the model (LGM, the default,
  or Hull-White), the simulation, the pricing engine per product, the Greeks method, the
  settlement method and the precision per stage, as ORE's configuration files do. Defaults are
  ORE's.
- **Every setting reachable.** Anything the engine can be configured to do, the API can ask
  for. Since roadmap 1.3 one request reaches the model per currency, the engines, the Greeks
  method and settings and precision; it cannot yet reach market-risk VaR/ES, the
  cross-asset calibration as a standalone run, or `shift_horizon`
  ([I-56](../planning/known-issues.md#i-56)). A completeness test will compare the configuration types
  with the request schema, so a new setting cannot ship without its API field.
- **Robust.** Validated before any job starts: types, unknown fields refused, cross-field
  checks, each refusal a `400` or `422` naming its field.
- **Names say what they are.** No route or field is named like a version unless it marks a
  revision of the contract itself. `/v2` and `schema_version: "2"` go.
- **Nothing silently breaks.** Both route names keep answering the one request; the
  retired Hull-White shape is a `422` naming its replacement.

## Response schema: `PortfolioResultSchema`

Mirrors `engine.portfolio.PortfolioResult`:

| Field | Type | Meaning |
|---|---|---|
| `base_npv` | `float` | Portfolio total. By construction `sum(base_npv_per_trade)` — the total and the breakdown are the same numbers, not two independent computations. |
| `base_npv_per_trade` | `List[float]` | Per-trade t=0 NPV, in the request's own `trades` order. Lets a caller reconcile the portfolio total against identified positions/contracts instead of receiving only an unattributable aggregate. |
| `npv_cube` | `List[List[List[float]]]` | `[Scenarios, TimeSteps, Trades]`, JSON-nested. |
| `exposure` | `{"times": [...], "epe": [...], "ene": [...], "ee_b": [...], "eee_b": [...], "pfe": {"PFE_95": [...], ...}} \| null` | The whole portfolio as one netting set; every list has one entry per date in `times`, starting at t=0. See [Exposure](../risk/exposure.md). |
| `trade_exposures` | `List[...]` | The same object per trade, in the request's `trades` order. |
| `exposure.epe_b`, `exposure.eepe_b` | `List[float]` | ORE's time-weighted EPE_B / EEPE_B profiles. |
| `exposure.basel_epe`, `exposure.basel_eepe` | `float \| null` | ORE's Basel EPE_B / EEPE_B at the one-year horizon. |
| `greeks` | `{"<trade_index>": {"values": {"<key>": [...]}, "shapes": {...}, "theta": ...}} \| null` | `null` unless the request set `compute_greeks: true`. Keys are trade indices (as strings, JSON's own object-key requirement) matching the request's `trades` order. Every Greek is flattened row-major into `values`; one of more than one dimension (`vega:<ccy>`, option tenors × swap tenors) also has its shape in `shapes`. The keys are `delta:discount:<ccy>`, `gamma:discount:<ccy>`, `delta:index:<name>`, `gamma:index:<name>` (per curve tenor for `Bump`, per market pillar for `AD`), `vega:<ccy>` for a trade whose engine reads the swaption volatilities, and `theta`. See [Greeks](../risk/greeks.md). |
| `trade_ids` | `List[str]` | The trades' ids in request order (`trade-0`, ... when the request gave none) ([I-10](../planning/known-issues.md#i-10)). |
| `measure` | `str \| null` | `risk-neutral-pricing`, or `null` without scenario risk (I-11). |
| `warnings` | `List[str]` | Run warnings (none are emitted today; see [The Portfolio Entry Point: Known-limitation flagging](portfolio-entrypoint.md#known-limitation-flagging)). |

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
    r = requests.get(f"{BASE}/portfolio/price/{job_id}")
    data = r.json()
    if data["status"] in ("done", "failed"):
        break
    time.sleep(1.0)

if data["status"] == "failed":
    raise RuntimeError(data["error"])

print("base NPV:", data["result"]["base_npv"])
print("PFE 95% at each date:", data["result"]["exposure"]["pfe"]["PFE_95"])
```

See a curl-only version in [User Guide: Running the API](../getting-started/user-guide.md#running-the-api).

## Tested by

`tests/test_api_market_path.py`: on the shared test portfolio (`tests/support/portfolio.py`),
the polled result equals a direct `price_portfolio` call: NPVs, cube, the exposure profiles
including EPE_B/EEPE_B and Basel, trade ids, and the 2-D Vega. It also checks the `400`s and
the `422` for a trade carrying `hw_sigma`. `tests/test_shared_portfolio.py` checks that the
HTTP body of that portfolio is the dataclass portfolio.

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
