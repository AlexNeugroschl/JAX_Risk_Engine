# HTTP API

**Modules:** [`engine/api/app.py`](../../engine/api/app.py),
[`engine/api/routes.py`](../../engine/api/routes.py),
[`engine/api/schemas.py`](../../engine/api/schemas.py)

## Plain-language summary

Everything described in [The Portfolio Entry Point](portfolio-entrypoint.md) is reachable
from plain Python already — `price_portfolio(request)`. This module wraps that same
function behind an HTTP API, for a caller (like TraderX — see
[TraderX Integration Plan](../planning/traderX_integration/traderx-integration.md)) that isn't a Python process
sharing this codebase's own memory space.

**Wrap, not replace.** `engine.portfolio.PortfolioRequest`/`PortfolioResult` and every
instrument config dataclass stay the single source of truth for the engine's own internal
shape (see [Architecture: Typed configuration](../concepts/architecture.md#typed-configuration)).
Pydantic models in `engine/api/schemas.py` mirror them field-for-field, each with a
`.to_dataclass()` method converting into the real dataclass and — for results — a
`.from_dataclass()` classmethod for the reverse direction. `engine/portfolio/` and
everything below it has **zero** Pydantic/FastAPI dependency; the heavy `api` extra
(FastAPI, Pydantic, uvicorn) is only needed to run this HTTP layer, not the core
simulation/pricing/risk engine.

## Stack: FastAPI + Pydantic v2 + uvicorn

Chosen over the roadmap's original "FastAPI/gRPC" placeholder (see
[Roadmap](../planning/roadmap-and-history.md)) — gRPC was evaluated and explicitly not
used:

- Zero existing web framework lock-in to displace.
- `SimulationConfig`'s own docstring already forward-references a future Pydantic schema;
  adopting Pydantic validates rather than contradicts existing intent.
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

The main endpoint. Body: a `PortfolioRequestSchema` (mirrors `PortfolioRequest` — see
"Request schema" below). Validates synchronously (cheap — Phase 1's validators do no JAX
work), then schedules the actual pricing as a background task and returns immediately.

**Success:** `202 Accepted`
```json
{"job_id": "b3f1c2a0-..."}
```

**Validation failure (bad covariance, mismatched `hw_a`, etc.):** `400 Bad Request`, with
the exact same validator error message `engine.portfolio`'s own `ValueError` carries:
```json
{"detail": "trade[1] (SwaptionConfig, notional=500000.0): hw_a=0.099 does not match sim_config.rates.mean_reversion[0]=0.03"}
```

**Malformed schema (wrong types, missing required fields):** `422 Unprocessable Entity`
(FastAPI's automatic Pydantic validation response).

### `GET /portfolio/price/{job_id}`

Poll for a job's status/result.

**Response:**
```json
{"status": "done", "result": { "...": "PortfolioResultSchema" }, "error": null}
```

`status` is one of `pending` / `running` / `done` / `failed`. `result` is `null` until
`status == "done"`. `error` is `null` unless `status == "failed"`, in which case it carries
the exception message and traceback. `pending` currently covers both "genuinely queued
behind this tier's worker pool" and "actively running in a worker process" — the worker
can't cheaply report its own sub-states back to the dispatcher without a mechanism this
phase doesn't build (see `engine/api/routes.py`'s `get_portfolio_price` docstring); a
`done`/`failed` job's error message/traceback come from
`concurrent.futures.Future.result()` re-raising the worker-side exception, which
`concurrent.futures.process` automatically annotates with the full remote (worker-process)
traceback.

**Unknown `job_id`:** `404 Not Found`.

### `POST /calibration/lgm`

Standalone calibration — wraps `engine.calibration.basket.build_coterminal_basket` +
`engine.calibration.lgm.calibrate_lgm_sigma`, for a caller who wants a fitted `Sigma`
term structure back before submitting a full portfolio request. Synchronous (a bootstrap
bisection, not a Monte Carlo simulation — no async job pattern needed).

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
pool of worker processes, sized per precision tier (float32/float64), each with its own
independent JAX/XLA runtime pinned to its own `jax_enable_x64` setting at boot. Polling
`GET /portfolio/price/{job_id}` now checks `future.done()`/`future.result()` instead of
reading fields a background thread mutated directly, but the response shape/status values
are unchanged. See [Architecture: Concurrency](../concepts/architecture.md) and
`engine/portfolio/worker_pool.py`'s own module docstring for the full mechanism and why
multi-process (not multi-thread, and not single-process device sharding) is the only model
compatible with "different precision tiers running genuinely concurrently" under JAX's
real constraints.

There remain **two separate, distinct motivations for a future shared store (Redis, a
database table)**, worth keeping apart:

1. **HTTP-scaling to multiple uvicorn worker processes.** Running more than one uvicorn
   worker still means each worker process has its own, mutually invisible `_JOBS` dict (and
   its own separate `engine.portfolio.worker_pool` pools underneath it) — a `job_id`
   returned by one uvicorn worker would still 404 against another. This motivation is
   **still deferred, still out of scope** — nothing in this phase changes it; it's a
   question about the *HTTP/dispatcher* layer's own process count, one level above the
   pricing worker pool.
2. **Process-isolated precision/device concurrency.** This was the *other* reason a shared
   store might once have seemed necessary — if the fix for `_PRICING_LOCK`'s serialization
   had been "spread jobs across multiple dispatcher processes" instead of "give
   `price_portfolio` itself a multi-process worker pool underneath one dispatcher." **This
   motivation is now solved**, by `engine.portfolio.worker_pool`'s `ProcessPoolExecutor`-based
   per-tier pools, not by Redis/a database — the dispatcher itself can stay a single
   process while still achieving genuine cross-precision, cross-device concurrency one
   layer down.

**The EOD path already has the durable store this section defers.** Since W0.8,
`POST /eod/price` publishes every *terminal* attempt through a crash-safe filesystem store
(`engine/integration/publication.py`), so an EOD result survives a restart and stays
addressable by `attemptId`. That work is deliberately **EOD-only**: `_JOBS` above is
untouched, and a `job_id` from `/portfolio/price` is still lost on restart. The two paths
have different durability guarantees today, which is a real difference a caller needs to
know rather than an inconsistency to gloss over — see
[I-08](../known-issues.md#i-08) and
[EOD Integration](eod-integration.md#w164--the-eod-http-routes).

## Request schema: `PortfolioRequestSchema`

Mirrors `engine.portfolio.PortfolioRequest`:

| Field | Type | Meaning |
|---|---|---|
| `evaluation_date` | `str` (ISO `YYYY-MM-DD`) | Default evaluation date applied to any trade that doesn't specify its own. |
| `market` | `SimulationConfigSchema` | Mirrors `SimulationConfig` field-for-field. |
| `trades` | `List[TradeSchema]` | A discriminated union on each trade object's own `trade_type` field: `"swap"`, `"european_swaption"`, `"bermudan_swaption"`, `"american_swaption"`, or `"bond"`. |
| `percentiles` | `List[float]` | Default `[0.95, 0.99]`. |
| `calibration_basket` | `CalibrationBasketRequestSchema \| null` | Optional. Required if any Bermudan/American trade has `hw_sigma: null` — see "Automatic calibration" below. |
| `compute_greeks` | `bool` | Default `false`. |
| `precision` | `PrecisionConfigSchema \| null` | Optional (default `null`). `null`/omitted behaves identically to an explicit all-64 block — see "Precision control" below. |
| `scenario_risk` | `bool` | Default `true`. **Must be `false` for any portfolio containing a `"bond"`** — see "Bonds and scenario risk" below. |

### Bonds and scenario risk

`trade_type: "bond"` (W1.5) is a Treasury bill or note. A **bill** is simply a bond with
`coupon_schedule` omitted and `coupon_rate` left at `0.0`; a **note** supplies an explicit
schedule. `face_amount` is **signed** — a short position is a negative face, and there is no
separate sign field.

A bond is priced by closed-form discounting against its **own** `initial_zero_curve`, so it
has **no scenario NPV**: no stochastic driver, no time evolution, and therefore no VaR or ES.

| `scenario_risk` | Portfolio contains a bond | Outcome |
|---|---|---|
| `true` (default) | no | Normal: full `npv_cube`, VaR/ES. |
| `true` | **yes** | **Refused**, naming the offending trade. |
| `false` | either | `base_npv`, `base_npv_per_trade` and `greeks` are real; `npv_cube` is empty and `risk` is `{}`. |

The response carries **`scenario_risk_available`** saying which happened. When it is `false`,
the VaR/ES numbers are **absent, not zero** — an empty `risk` asserts nothing, whereas a
`VaR_95` of `0.00` would assert a *measured* absence of risk. See
[I-24](../known-issues.md#i-24) for why a constant column is refused rather than broadcast.

A bond's `delta`/`gamma` are **scalars** (one parallel 1bp bump against its single curve),
unlike a swap's per-pillar `discount_delta`/`forward_delta` vectors. They are still delivered
as one-element lists so `greeks.values` stays uniformly a list per Greek. No `vega` is
reported: a fixed-coupon bond off a deterministic curve has no volatility input, and it is
omitted rather than reported as `0.0`.

Every trade schema mirrors its dataclass field-for-field, with two representational
differences (SWIG-bound `ORE` types aren't natively Pydantic-serializable):

- `evaluation_date` fields are ISO date strings (`"2026-07-30"`), parsed via
  `ORE.DateParser.parseISO`.
- `forward_start` (on `european_swaption` trades) is an ORE period string (`"5Y"`, `"18M"`,
  `"0D"`), parsed via `ORE.Period(str)` — the same parse
  `engine.portfolio.validation._validate_tenor` already validates for `swap_tenor`.

### Automatic calibration: `hw_sigma: null` + `calibration_basket`

`BermudanSwaptionConfigSchema`/`AmericanSwaptionConfigSchema`'s `hw_sigma` accepts `null`
(Python `None`) to request automatic calibration — mirroring
`engine.portfolio.PortfolioRequest.calibration_targets` (see [The Portfolio Entry Point:
Automatic calibration](portfolio-entrypoint.md#automatic-calibration)) — but the dataclass
field takes an already-built `List[CalibrationTarget]`, which isn't directly expressible in
a JSON request body (each target carries full ORE-derived cashflow arrays, not raw market
data). `PortfolioRequestSchema.calibration_basket` bridges this: it mirrors
`engine.calibration.basket.build_coterminal_basket`'s own raw inputs instead —

| Field | Type | Meaning |
|---|---|---|
| `exercise_times` | `List[float]` | One per basket instrument, ascending. |
| `final_maturity_time` | `float` | Every basket instrument's underlying swap matures here (the co-terminal/diagonal convention). |
| `notional` | `float` | Shared by every basket instrument. |
| `payer` | `bool` | Shared by every basket instrument. |
| `market_vols` | `List[float]` | Market normal (Bachelier) volatility per `exercise_times` entry, same length/order. |

and the server builds the actual `CalibrationTarget` list from it: the curve, mean
reversion (`hw_a`), and `evaluation_date`/`index_tenor_months` used to build the basket come
from the *first* trade in `trades` with `hw_sigma: null` (matching
`_fill_calibrated_sigma`'s own "one shared basket, applied per `rate_factor_index`
that needs it" design — there is currently no way to submit more than one basket per
request). Submitting `calibration_basket` when no trade actually needs it returns `400`;
leaving it unset while a trade has `hw_sigma: null` fails once pricing actually runs (the
job reaches `status: "failed"` with the same `"calibration_targets was not supplied"`
message `price_portfolio` itself raises).

Piecewise (post-calibration) `Sigma` term structures still aren't expressible directly —
a request always starts from either a flat `hw_sigma`, `null` (paired with
`calibration_basket`), or is a validation error.

### Precision control: `PrecisionConfigSchema`

Mirrors `engine.portfolio.PrecisionConfig` field-for-field (see [The Portfolio Entry
Point: PrecisionConfig](portfolio-entrypoint.md#precisionconfig) and
[Architecture](../concepts/architecture.md#adjustable-precision) for the full mechanism):

| Field | Type | Default | Meaning |
|---|---|---|---|
| `simulation` | `int` (`32`\|`64`) | `64` | Monte Carlo path generation dtype. |
| `pricing` | `int` (`32`\|`64`) | `64` | Instrument NPV / `npv_cube` dtype. |
| `risk` | `int` (`32`\|`64`) | `64` | VaR/ES + Greeks dtype (one shared setting). |

`precision` on `PortfolioRequestSchema` is `Optional`, not a populated default, so "no
`precision` key sent" and "explicit all-64 sent" resolve identically (both become
`PrecisionConfig()`). Any value outside `{32, 64}` fails `PrecisionConfig.__post_init__`'s
own validation, which `POST /portfolio/price` already runs synchronously (as part of the
same up-front `request.to_dataclass()` call that validates the rest of the body) — so an
invalid precision returns an immediate `400` with that validator's own message, not a
`202` followed by a failed job:

```json
{
  "evaluation_date": "2026-07-30",
  "market": { "...": "..." },
  "trades": [ "..." ],
  "precision": { "simulation": 64, "pricing": 32, "risk": 32 }
}
```

```
POST /portfolio/price
{"precision": {"simulation": 16}}
-> 400 {"detail": "PrecisionConfig.simulation must be 32 or 64, got 16"}
```

**Concurrency note:** concurrent `/portfolio/price` jobs now genuinely parallelize across
precision tiers — a `simulation: 32` job and a `simulation: 64` job submitted back-to-back
run in separate worker processes at the same time, not serialized behind one process-wide
lock (see [Architecture: Concurrency](../concepts/architecture.md) for the full mechanism,
`engine.portfolio.worker_pool`). Same-tier jobs beyond that tier's own worker-pool size
still queue for a free worker — expected pool exhaustion, not a bug, and no different in
kind from any fixed-size worker pool. Either way, correctness is unaffected: each job's own
result is always independent of what else is running concurrently, whether it runs
immediately or waits for a worker to free up.

## Response schema: `PortfolioResultSchema`

Mirrors `engine.portfolio.PortfolioResult`:

| Field | Type | Meaning |
|---|---|---|
| `base_npv` | `float` | Portfolio total. By construction `sum(base_npv_per_trade)` — the total and the breakdown are the same numbers, not two independent computations. |
| `base_npv_per_trade` | `List[float]` | Per-trade t=0 NPV, in the request's own `trades` order. Lets a caller reconcile the portfolio total against identified positions/contracts instead of receiving only an unattributable aggregate. |
| `npv_cube` | `List[List[List[float]]]` | `[Scenarios, TimeSteps, Trades]`, JSON-nested. |
| `risk` | `{"values": {"VaR_95": [...], "ES_95": [...], ...}}` | `NaN` values (an empty-tail Expected Shortfall — see [Risk Statistics](../risk/var_es.md)) serialize as JSON `null`, not the non-standard literal `NaN`. |
| `greeks` | `{"<trade_index>": {"values": {"delta": [...], "gamma": [...]}, "theta": ...}} \| null` | `null` unless the request set `compute_greeks: true`. Keys are trade indices (as strings, JSON's own object-key requirement) matching the request's own `trades` order. **Swaps** report `discount_delta`/`discount_gamma`/`forward_delta`/`forward_gamma` (differentiated against the curves their own `discount_curve_index`/`forward_curve_index` name) plus `theta`; swaptions report `delta`/`gamma`/`theta`. A **calibrated** Bermudan/American trade additionally reports `vega`, one entry per `calibration_basket` instrument — omitted for a flat (hand-set) `hw_sigma`, which has no market quote to be sensitive to. |
| `warnings` | `List[str]` | Known-limitation warnings (mid-coupon exercise misalignment, etc.) — see [The Portfolio Entry Point: Known-limitation flagging](portfolio-entrypoint.md#known-limitation-flagging). |

## Example: a Python `requests` session

```python
import time
import requests

BASE = "http://127.0.0.1:8000"

body = {
    "evaluation_date": "2026-07-30",
    "market": {
        "time_grid": [0.0, 0.5, 1.0, 1.5, 2.0],
        "equities": {"initial_prices": [100.0], "dividend_yields": [0.0], "rate_mapping": [[0.0]]},
        "rates": {
            "initial_rates": [0.03], "theta": [0.03], "mean_reversion": [0.03],
            "initial_zero_curves": [{"times": [0.0, 1.0, 2.0, 5.0, 10.0, 30.0], "rates": [0.03] * 6}],
        },
        "joint_covariance": [[0.04, 0.0], [0.0, 0.0001]],
        "scenarios": 4096,
    },
    "trades": [
        {"trade_type": "swap", "notional": 1_000_000.0, "fixed_rate": 0.032, "payer": True,
         "discount_curve_index": 0, "forward_curve_index": 0, "swap_tenor": "2Y"},
    ],
    "percentiles": [0.95, 0.99],
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
print("95% VaR at each step:", data["result"]["risk"]["values"]["VaR_95"])
```

See a curl-only version in [User Guide: Running the API](../getting-started/user-guide.md#running-the-api).

## Tested by

`tests/test_api.py`, using FastAPI's `TestClient` (backed by `httpx`) — no running server
process needed:

- `TestHealthAndVersion` — `/health`/`/version` respond.
- `TestPortfolioPriceHappyPath` — a valid body returns `202` + a job id; polling reaches
  `"done"` with a result that matches a direct `price_portfolio` call on the equivalent
  dataclass request, bit-for-bit after the schema round-trip; Greeks are included when
  requested.
- `TestCalibratedBermudanOverHttp` — `calibration_basket` resolving an uncalibrated
  Bermudan/American trade end to end; the `400`/failed-job error paths when it's missing
  or unnecessary.
- `TestPortfolioPriceAtScale` — a 20-trade mixed-instrument portfolio and a 10-trade
  bit-for-bit HTTP-vs-direct-call cross-check submitted as real JSON bodies (the schema
  round-trip at a payload size well beyond the 1-2 trade bodies used elsewhere); an empty
  `trades` list; many identical trades pricing identically; two concurrent jobs in the
  shared in-process job store not cross-contaminating each other's results.
- `TestPortfolioPriceInvalidPayload` — non-PSD covariance and a mismatched
  `rate_factor_index` both return a `4xx` with the underlying validator's own message, not
  a `500`; malformed/missing/invalid-discriminator bodies return `422`.
- `TestPortfolioPriceUnknownJob` — an unknown `job_id` returns `404`.
- `TestCalibrationEndpoint` — `/calibration/lgm` happy path and malformed-schema `422`.
