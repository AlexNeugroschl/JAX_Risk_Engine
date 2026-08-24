# HTTP API

**Modules:** [`engine/api/app.py`](../../engine/api/app.py),
[`engine/api/routes.py`](../../engine/api/routes.py),
[`engine/api/schemas.py`](../../engine/api/schemas.py)

## Plain-language summary

Everything described in [The Portfolio Entry Point](portfolio-entrypoint.md) is reachable
from plain Python already — `price_portfolio(request)`. This module wraps that same
function behind an HTTP API, for a caller (like TraderX — see
[TraderX Integration Plan](../planning/traderx-integration.md)) that isn't a Python process
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

## Endpoints

### `GET /health`

Liveness only — confirms the process is up. Does no engine work.

**Response:** `{"status": "ok"}`

### `GET /version`

Engine package version, JAX backend (CPU/GPU), and git commit (best-effort — `null` if
this isn't a git checkout).

**Response:**
```json
{"engine_version": "0.1.0", "jax_backend": "cpu (Windows)", "git_commit": "abc1234..."}
```

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
the exception message and traceback.

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

## Job store: in-process, single-process only

The job store backing `GET /portfolio/price/{job_id}` is a plain in-process Python `dict`
(see `engine/api/routes.py`'s own module docstring) — appropriate for this system's
current scope (low request volume, a single downstream consumer per the roadmap, not
public internet traffic). **This does not survive a multi-worker deployment**: running
more than one uvicorn worker process means each worker has its own, mutually invisible job
dict, so a `job_id` returned by one worker will 404 against another. A production
multi-process deployment would need a shared store (Redis, a database table) instead —
explicitly out of scope for this phase.

Similarly, `BackgroundTasks` runs the pricing job in a thread from FastAPI's own worker
thread pool, not a separate process — fine for a first cut, but JAX/XLA compilation is not
safely shared across threads the way async I/O is. If concurrent-job throughput becomes a
real requirement, upgrading to a process-pool executor is the natural next step (flagged
here as a future scaling note, not something this phase implements).

## Request schema: `PortfolioRequestSchema`

Mirrors `engine.portfolio.PortfolioRequest`:

| Field | Type | Meaning |
|---|---|---|
| `evaluation_date` | `str` (ISO `YYYY-MM-DD`) | Default evaluation date applied to any trade that doesn't specify its own. |
| `market` | `SimulationConfigSchema` | Mirrors `SimulationConfig` field-for-field. |
| `trades` | `List[TradeSchema]` | A discriminated union on each trade object's own `trade_type` field: `"swap"`, `"european_swaption"`, `"bermudan_swaption"`, or `"american_swaption"`. |
| `percentiles` | `List[float]` | Default `[0.95, 0.99]`. |
| `compute_greeks` | `bool` | Default `false`. |

Every trade schema mirrors its dataclass field-for-field, with two representational
differences (SWIG-bound `ORE` types aren't natively Pydantic-serializable):

- `evaluation_date` fields are ISO date strings (`"2026-07-30"`), parsed via
  `ORE.DateParser.parseISO`.
- `forward_start` (on `european_swaption` trades) is an ORE period string (`"5Y"`, `"18M"`,
  `"0D"`), parsed via `ORE.Period(str)` — the same parse
  `engine.portfolio.validation._validate_tenor` already validates for `swap_tenor`.

`BermudanSwaptionConfigSchema`/`AmericanSwaptionConfigSchema`'s `hw_sigma` accepts `null`
(Python `None`) to request automatic calibration — see [The Portfolio Entry Point:
Automatic calibration](portfolio-entrypoint.md#automatic-calibration). Piecewise
(post-calibration) `Sigma` term structures are not currently expressible directly in a
request body — a request always starts from either a flat `hw_sigma` or `null`.

## Response schema: `PortfolioResultSchema`

Mirrors `engine.portfolio.PortfolioResult`:

| Field | Type | Meaning |
|---|---|---|
| `base_npv` | `float` | |
| `npv_cube` | `List[List[List[float]]]` | `[Scenarios, TimeSteps, Trades]`, JSON-nested. |
| `risk` | `{"values": {"VaR_95": [...], "ES_95": [...], ...}}` | `NaN` values (an empty-tail Expected Shortfall — see [Risk Statistics](../risk/var_es.md)) serialize as JSON `null`, not the non-standard literal `NaN`. |
| `greeks` | `{"<trade_index>": {"values": {"delta": [...], "gamma": [...]}, "theta": ...}} \| null` | `null` unless the request set `compute_greeks: true`. Keys are trade indices (as strings, JSON's own object-key requirement) matching the request's own `trades` order. |
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
- `TestPortfolioPriceInvalidPayload` — non-PSD covariance and a mismatched
  `rate_factor_index` both return a `4xx` with the underlying validator's own message, not
  a `500`; malformed/missing/invalid-discriminator bodies return `422`.
- `TestPortfolioPriceUnknownJob` — an unknown `job_id` returns `404`.
- `TestCalibrationEndpoint` — `/calibration/lgm` happy path and malformed-schema `422`.
