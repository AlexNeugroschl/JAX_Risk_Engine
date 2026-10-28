# API Guide

This guide shows how to use the engine through its HTTP API: starting the server, describing
a market and a portfolio, running each analytic, choosing models and precision, and reading
the results. Every example on this page was run against a live server.

For a field-by-field reference of every request and response schema, see
[HTTP API](../reference/http-api.md). The running server also publishes its own interactive
schema at `/docs`.

## Contents

- [Starting the server](#starting-the-server)
- [Your first run](#your-first-run)
- [How a run works](#how-a-run-works)
- [A small Python client](#a-small-python-client)
- [The request](#the-request)
- [Market data](#market-data)
- [Trades](#trades)
- [Analytics](#analytics): [npv](#npv), [exposure](#exposure), [sensitivities](#sensitivities),
  [market_risk](#market_risk), [calibration](#calibration)
- [Models and pricing engines](#models-and-pricing-engines)
- [Precision](#precision)
- [Large arrays](#large-arrays)
- [Retries, cancellation and retention](#retries-cancellation-and-retention)
- [Errors](#errors)
- [Running on accelerators](#running-on-accelerators)

## Starting the server

Install the package with the `api` extra (`requirements.txt` includes it), then start the
server with Uvicorn:

```bash
.venv/bin/python -m pip install -r requirements.txt      # Windows: .venv\Scripts\python.exe
.venv/bin/python -m uvicorn engine.api.app:app --host 127.0.0.1 --port 8000
```

The server is a thin front end. On the first submitted run it starts an **engine worker**, a
second process that does all the numerical work and owns the machine's CPU, GPU or TPU devices.
Runs are stored in a small SQLite job queue, so they survive restarts of the server.

Check that it is up:

```bash
curl http://127.0.0.1:8000/health
# {"status": "ok", "worker": {"mode": "spawn", "serving": true, "failed_starts": 0, "error": null}}

curl http://127.0.0.1:8000/version
# {"engine": {"version": "0.1.0", "revision": "...", "jax": "0.10.2", "ore": "1.8.16.0", ...}, ...}
```

Optional settings, read from the environment when the server starts:

| Variable | Meaning | Default |
|---|---|---|
| `JAX_RISK_JOB_QUEUE` | Path of the job queue's SQLite file | `jax-risk-jobs/jobs.sqlite3` in the temp directory |
| `JAX_RISK_RETENTION_DAYS` | Days a finished run is kept; `off` keeps every run | `30` |
| `JAX_RISK_SCENARIO_DEVICES` | The most devices one run is split across | every device that divides the scenario count |
| `JAX_RISK_WORKER` | `spawn` (the server starts the worker) or `external` (you run `jax-risk-worker --queue <file>` yourself) | `spawn` |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | Set to `false` when other processes share the GPU | JAX's default (75% of GPU memory) |

**The first run of a new shape is slower.** The engine compiles each program the first time it
sees a given portfolio shape, number of paths and date grid. Repeating a run of the same shape
reuses the compiled programs, and the worker also caches them on disk across restarts.

## Your first run

A run is submitted with `POST /runs`. This one values a two-year USD swap today and simulates
its exposure on 4,096 paths:

```bash
curl -X POST http://127.0.0.1:8000/runs \
  -H "Content-Type: application/json" \
  -d '{
    "market": {"asof": "2026-07-30", "currencies": {"USD": {
      "discount_curve": {"times": [0.0,1.0,2.0,5.0,10.0,30.0], "rates": [0.030,0.030,0.034,0.040,0.046,0.050]},
      "index_curves": {"USD-SIMINDEX-6M": {"times": [0.0,1.0,2.0,5.0,10.0,30.0], "rates": [0.034,0.034,0.038,0.044,0.049,0.052]}}}}},
    "portfolio": [{"trade_type": "swap", "trade_id": "swap-1", "notional": 1000000.0, "fixed_rate": 0.036,
                   "payer": true, "swap_tenor": "2Y"}],
    "analytics": ["npv", "exposure"],
    "config": {"models": {"ir": {"USD": {"model": "HullWhite", "reversion": 0.03, "volatility": 0.01}}},
               "simulation": {"dates": ["2027-01-30", "2027-07-30", "2028-07-30"], "samples": 4096}}
  }'
# {"run_id": "3f6c1a52-..."}
```

The server checks the request, queues it and answers `202 Accepted` straight away. Fetch the
result with the returned id. `?wait=30` holds the request open until the run finishes, for at
most 30 seconds:

```bash
curl "http://127.0.0.1:8000/runs/3f6c1a52-...?wait=30"
```

## How a run works

```
POST /runs ──▶ 202 {"run_id"} ──▶ GET /runs/{run_id}?wait=30 ──▶ ... ──▶ status "done", result
```

`GET /runs/{run_id}` always answers with the run's current status:

| `status` | Meaning |
|---|---|
| `pending` | Queued, not started. If the worker cannot start, `error` says why |
| `running` | The engine worker is running it |
| `done` | Finished; `result` holds the result |
| `failed` | The run raised an error; `error` has the message and `failure_class` its category ([Errors](#errors)) |
| `cancelled` | Cancelled before it started or at a checkpoint ([Cancellation](#retries-cancellation-and-retention)) |
| `interrupted` | The worker stopped during the run (killed or crashed); submit it again |

Runs execute one at a time in submission order. Each run uses every device of the machine.

## A small Python client

The rest of this guide uses Python with [`requests`](https://requests.readthedocs.io/). The
helpers below submit a run, wait for it and return its result. Each later example builds on
the definitions before it, so the blocks can be pasted into one session in order.

```python
import requests

BASE = "http://127.0.0.1:8000"


def submit(body: dict) -> str:
    """Submit a run and return its id."""
    response = requests.post(f"{BASE}/runs", json=body, timeout=60)
    if response.status_code != 202:
        raise RuntimeError(f"{response.status_code}: {response.json()['detail']}")
    return response.json()["run_id"]


def wait(run_id: str) -> dict:
    """Long-poll a run until it finishes and return its result."""
    while True:
        status = requests.get(f"{BASE}/runs/{run_id}", params={"wait": 30}, timeout=60).json()
        if status["status"] not in ("pending", "running"):
            break
    if status["status"] != "done":
        raise RuntimeError(f"run {status['status']}: {status['error']}")
    return status["result"]


def run(body: dict) -> dict:
    return wait(submit(body))
```

## The request

Every analytic is requested through the same body:

| Field | Required | Meaning |
|---|---|---|
| `market` | yes | Today's market data ([Market data](#market-data)) |
| `portfolio` | for every analytic but `calibration` | The trades ([Trades](#trades)) |
| `analytics` | yes | What to compute: any of `npv`, `exposure`, `sensitivities`, `market_risk`, `calibration` |
| `config` | depends on the analytics | How to compute it: models, simulation grid, pricing engines, Greeks settings, scenarios, precision |
| `outputs` | no | Whether large arrays come inline, by reference or not at all ([Large arrays](#large-arrays)) |
| `idempotency_key` | no | Makes a retried submission return the same run ([Retries](#retries-cancellation-and-retention)) |
| `keep` | no | `true` keeps the run forever instead of for the retention period |

Every `config` section is optional, and anything left out takes the same default that ORE
uses. `GET /config/defaults` returns every section with its defaults filled in, which is a
convenient starting point to copy and edit:

```python
defaults = requests.get(f"{BASE}/config/defaults", timeout=60).json()
print(sorted(defaults))
# ['base_currency', 'execution', 'exposure', 'market_risk', 'models', 'precision', 'pricing',
#  'sensitivity', 'simulation']
```

`models`, `simulation` and `market_risk` have no defaults (they are `null` there): they describe
your model and scenarios, and the analytics that need them refuse to run without them.

**Strict validation.** The server checks the whole request before queuing it. An unknown
field, a section that no requested analytic reads, a trade the market cannot value or an
inconsistent setting is refused immediately with a message naming the field, so nothing is
silently ignored.

Conventions used throughout:

- Dates are ISO strings: `"2026-07-30"`.
- Tenors are period strings: `"6M"`, `"2Y"`, `"18M"`.
- Rates and volatilities are decimals: `0.035` is 3.5%, a normal volatility of `0.0080` is 80
  basis points.
- Every amount in a result is in the reporting currency, `config.base_currency` (`"USD"` by
  default).

## Market data

The `market` section is a snapshot of today's market:

```python
PILLARS = [0.0, 1.0, 2.0, 5.0, 10.0, 30.0]

MARKET = {
    "asof": "2026-07-30",
    "currencies": {
        "USD": {
            "discount_curve": {"times": PILLARS, "rates": [0.030, 0.030, 0.034, 0.040, 0.046, 0.050]},
            "index_curves": {
                "USD-SIMINDEX-6M": {"times": PILLARS, "rates": [0.034, 0.034, 0.038, 0.044, 0.049, 0.052]},
            },
            "swaption_vols": {
                "option_tenors": ["1Y", "2Y", "5Y", "10Y"],
                "swap_tenors": ["1Y", "5Y", "10Y"],
                "vols": [[0.0080, 0.0088, 0.0090],
                         [0.0085, 0.0091, 0.0093],
                         [0.0090, 0.0093, 0.0094],
                         [0.0092, 0.0094, 0.0096]],
            },
        },
    },
}
```

| Field | Meaning |
|---|---|
| `asof` | The valuation date. Every trade is valued on it |
| `currencies.<CCY>.discount_curve` | The discount curve: pillar `times` in years from `asof` and continuously compounded zero `rates` |
| `currencies.<CCY>.index_curves` | Forwarding curves, keyed by index name `<CCY>-SIMINDEX-<n>M`. A trade with `index_tenor_months: 6` reads `USD-SIMINDEX-6M` |
| `currencies.<CCY>.swaption_vols` | ATM normal (Bachelier) swaption volatilities: one row per option tenor, one column per swap tenor. Needed for swaptions and for model calibration |
| `fx_spots` | FX spots keyed by pair, e.g. `{"EURUSD": 1.10}`: one for every currency other than the reporting currency |
| `equities` | Equities for the simulation, keyed by name: `{"AAPL": {"currency": "USD", "spot": 150.0}}` |

Curve times must start at or after 0 and strictly increase. Rates may be zero or negative.

A second currency adds its own curves and an FX spot into the reporting currency:

```python
EUR = {
    "discount_curve": {"times": PILLARS, "rates": [0.020, 0.020, 0.022, 0.026, 0.030, 0.032]},
    "index_curves": {"EUR-SIMINDEX-6M": {"times": PILLARS, "rates": [0.023, 0.023, 0.025, 0.029, 0.033, 0.034]}},
}
MARKET_USD_EUR = {**MARKET, "currencies": {**MARKET["currencies"], "EUR": EUR}, "fx_spots": {"EURUSD": 1.10}}
```

## Trades

`portfolio` is a list of trades, each identified by `trade_type`. Every trade takes an optional
`trade_id`, echoed on every per-trade figure of the result. Give one on every trade or on
none; without them trades are numbered `trade-0`, `trade-1`, and so on. A trade carries only
its own terms: curves, volatilities and models come from the market and the configuration.

### Swaps

```python
swap = {
    "trade_type": "swap", "trade_id": "swap-5y",
    "notional": 10_000_000.0, "fixed_rate": 0.040, "payer": True,   # pay fixed, receive floating
    "swap_tenor": "5Y",                                             # starts at spot on asof
}
```

| Field | Meaning | Default |
|---|---|---|
| `notional`, `fixed_rate` | The swap's notional and fixed rate | required |
| `payer` | `true` pays fixed and receives floating; `false` the reverse | required |
| `swap_tenor` | Length of a swap starting at spot on `asof` | |
| `effective_date`, `maturity_date` | Explicit start and end, instead of `swap_tenor` | |
| `currency` | Currency of the swap and the curves it reads | `"USD"` |
| `index_tenor_months` | Floating leg's index tenor | `6` |
| `floating_spread` | Spread over the floating index | `0.0` |
| `fixings` | Past index fixings, `{"YYYY-MM-DD": rate}`, needed for a seasoned swap whose current coupon has already fixed | `{}` |

A seasoned swap, booked in the past, gives its dates and the fixing of its current coupon:

```python
seasoned_swap = {
    "trade_type": "swap", "trade_id": "swap-seasoned",
    "notional": 5_000_000.0, "fixed_rate": 0.031, "payer": False,
    "effective_date": "2025-03-17", "maturity_date": "2030-03-17",
    "fixings": {"2026-03-13": 0.0352},
}
```

### European swaptions

An option to enter a swap on one exercise date. The underlying swap is described as for a swap:

```python
european = {
    "trade_type": "european_swaption", "trade_id": "european-3y2y",
    "notional": 1_000_000.0, "fixed_rate": 0.042, "payer": True,
    "forward_start": "3Y", "swap_tenor": "2Y",    # exercise in 3 years into a 2-year swap
}
```

The exercise can be given as `forward_start` (a tenor from `asof`, with an optional
`exercise_lag_days`) or as explicit `exercise_date`, `effective_date` and `maturity_date`.
Every swaption also takes `settlement` (`"Physical"`, the default, or `"Cash"`) and `long_short`
(`"Long"`, the default, or `"Short"`).

### Bermudan swaptions

Exercisable on a list of dates into the remaining swap:

```python
bermudan = {
    "trade_type": "bermudan_swaption", "trade_id": "bermudan-5y",
    "notional": 1_000_000.0, "fixed_rate": 0.042, "payer": True, "swap_tenor": "5Y",
    "exercise_dates": ["2027-07-30", "2028-07-30", "2029-07-30", "2030-07-30"],
}
```

### American swaptions

Exercisable on any date in a window:

```python
american = {
    "trade_type": "american_swaption", "trade_id": "american-5y",
    "notional": 800_000.0, "fixed_rate": 0.040, "payer": False, "swap_tenor": "5Y",
    "first_exercise_date": "2027-07-30", "last_exercise_date": "2030-07-30",
}
```

Bermudan and American swaptions are each calibrated to their own basket of market swaptions
and priced on a numerical grid, so they need `swaption_vols` in their currency's market and are
the most expensive trades to price.

### Treasury bills and notes

A `bond` is a bill when it has no coupons, or a note with an explicit coupon schedule.
`face_amount` is signed: negative for a short position.

```python
bill = {"trade_type": "bond", "trade_id": "bill-6m", "face_amount": 1_000_000.0, "maturity_date": "2027-01-28"}

note = {
    "trade_type": "bond", "trade_id": "note-2029",
    "face_amount": 1_000_000.0, "maturity_date": "2029-02-15", "coupon_rate": 0.0375,
    "coupon_schedule": [{"start_date": s, "end_date": e} for s, e in [
        ("2026-02-15", "2026-08-15"), ("2026-08-15", "2027-02-15"), ("2027-02-15", "2027-08-15"),
        ("2027-08-15", "2028-02-15"), ("2028-02-15", "2028-08-15"), ("2028-08-15", "2029-02-15")]],
}
```

Each coupon period may also give a `payment_date` (default: its `end_date`). Other fields:
`currency` (default `"USD"`), `redemption_fraction` (default `1.0`) and `accrual_day_count`
(default `"ACT/ACT (ICMA)"`).

## Analytics

### `npv`

Today's value of every trade. It needs only the market and the portfolio:

```python
PORTFOLIO = [swap, seasoned_swap, european, bermudan, american, bill, note]

result = run({"market": MARKET, "portfolio": PORTFOLIO, "analytics": ["npv"]})

for row in result["trades"]:
    print(f"{row['trade_id']:>15} {row['product']:>18} {row['npv']:>14,.2f}")
print("portfolio:", result["npv"], result["base_currency"])
```

`result["trades"]` has one row per trade, in the order of the request. The same order is the
trade axis of every array in the result.

### `exposure`

Simulates the market forward on many paths, prices every trade on every path and date, and
computes the exposure profiles. It needs two config sections:

- `models`: the interest rate model of each currency ([Models](#models-and-pricing-engines)).
- `simulation`: the dates to simulate and the number of paths.

```python
SIMULATION = {
    "dates": ["2026-10-30", "2027-01-30", "2027-07-30", "2028-07-30", "2029-07-30", "2030-07-30", "2031-07-30"],
    "samples": 1024,
}
MODELS = {"ir": {"USD": {"model": "HullWhite", "reversion": 0.03, "volatility": 0.01}}}

result = run({
    "market": MARKET,
    "portfolio": [swap, european, note],
    "analytics": ["npv", "exposure"],
    "config": {"models": MODELS, "simulation": SIMULATION, "exposure": {"pfe_quantiles": [0.95, 0.99]}},
    "outputs": {"npv_cube": "inline"},
})

exposure = result["exposure"]
profile = exposure["netting_set"]           # the whole portfolio, without collateral
for i, t in enumerate(profile["times"]):
    print(f"t={t:5.2f}  EPE {profile['epe'][i]:>12,.0f}  ENE {profile['ene'][i]:>12,.0f}  "
          f"PFE 99% {profile['pfe']['PFE_99'][i]:>12,.0f}")

cube = exposure["npv_cube"]                 # [path, date, trade]
print(cube["shape"], cube["axes"])          # [1024, 7, 3] ['scenario', 'date', 'trade']
```

The `netting_set` profile lists every figure at `times`: today (`0.0`) followed by each
simulation date, in years. It contains:

| Field | Meaning |
|---|---|
| `epe`, `ene` | Expected positive and negative exposure on each date |
| `ee_b`, `eee_b` | Basel expected exposure, and its non-decreasing (effective) version |
| `pfe` | Potential future exposure at each requested quantile, keyed `PFE_95`, `PFE_99` |
| `epe_b`, `eepe_b` | Time-weighted averages of `ee_b` and `eee_b` |
| `basel_epe`, `basel_eepe` | The same averages over the first year, as Basel defines them |

Each row of `result["trades"]` also has an `exposure` entry with that trade's standalone
profile. `exposure["dates"]` lists the simulation dates (the cube's date axis), and
`exposure["pricing_settings"]` records how each product was priced.

Simulation settings:

| Field | Meaning | Default |
|---|---|---|
| `dates` | The simulation dates, after `asof`, ascending | required |
| `samples` | Number of paths. Powers of two suit the Sobol sequence and split evenly across devices | `1000` |
| `seed` | Seed of the scrambled Sobol sequence | `42` |
| `replicates` | Split `samples` into this many independently scrambled sets, which gives standard errors | `1` |
| `curve_tenors` | Tenors at which each simulated curve is represented | 3M to 30Y |
| `swaption_vol_decay` | How swaption volatilities roll forward: `"ForwardVariance"` or `"ConstantVariance"` | `"ForwardVariance"` |

**Several currencies.** Each currency gets its own model. Every currency other than the
reporting currency also needs an FX volatility, and `correlations` link any pair of factors
(`IR:<CCY>`, `FX:<pair>`, `EQ:<name>`); pairs left out are uncorrelated:

```python
eur_swap = {"trade_type": "swap", "trade_id": "eur-swap", "currency": "EUR",
            "notional": 5_000_000.0, "fixed_rate": 0.027, "payer": True, "swap_tenor": "5Y"}

result = run({
    "market": MARKET_USD_EUR,
    "portfolio": [swap, eur_swap],
    "analytics": ["npv", "exposure"],
    "config": {
        "models": {
            "ir": {"USD": {"model": "LGM", "reversion": 0.03, "volatility": 0.01},
                   "EUR": {"model": "LGM", "reversion": 0.02, "volatility": 0.008}},
            "fx_volatilities": {"EUR": 0.10},
            "correlations": [{"factor1": "IR:USD", "factor2": "IR:EUR", "value": 0.6},
                             {"factor1": "IR:USD", "factor2": "FX:EURUSD", "value": 0.2},
                             {"factor1": "IR:EUR", "factor2": "FX:EURUSD", "value": -0.1}],
        },
        "simulation": SIMULATION,
    },
    "outputs": {"npv_cube": "none"},
})
print({row["trade_id"]: row["npv"] for row in result["trades"]})   # both in USD
```

### `sensitivities`

Delta and Gamma to each curve tenor, Vega to each swaption volatility quote, and one-day
Theta, for every trade:

```python
result = run({
    "market": MARKET,
    "portfolio": [swap, european, note],
    "analytics": ["sensitivities"],
})

factors = result["sensitivities"]["factors"]     # the labels of each Greek's entries
greeks = result["trades"][1]["greeks"]           # the European swaption
for key, values in greeks["values"].items():
    print(key, dict(zip(factors[key], values)))
print("theta", greeks["theta"])
```

Greeks are keyed by type and curve: `delta:discount:USD`, `delta:index:USD-SIMINDEX-6M`,
`gamma:...`, and `vega:USD`. Vega is a matrix of option tenor × swap tenor: its `values` are
flattened row by row, its labels read `"<option tenor>x<swap tenor>"`, and `greeks["shapes"]`
gives its shape.

Settings, in `config.sensitivity`:

| Field | Meaning | Default |
|---|---|---|
| `method` | `"Bump"`: bump and revalue, as ORE's sensitivity analysis. `"AD"`: the same Greeks by automatic differentiation | `"Bump"` |
| `curve_tenors` | Tenors at which curves are bumped | 3M to 30Y |
| `curve_shift` | Size of a curve bump | `0.0001` (1bp) |
| `vol_shift` | Size of a volatility bump | `0.0001` |
| `theta_days` | Days the market is rolled for Theta | `1` |

With `"Bump"`, a calibrated Bermudan or American is recalibrated under every bump, which is
accurate but slow on a large book. `"AD"` gives the same keys in a fraction of the time:

```python
result = run({
    "market": MARKET,
    "portfolio": [bermudan],
    "analytics": ["npv", "sensitivities"],
    "config": {"sensitivity": {"method": "AD"}},
})
print(result["trades"][0]["greeks"]["values"]["delta:discount:USD"])
```

### `market_risk`

Short-horizon Value at Risk and Expected Shortfall by full revaluation: every trade is
repriced under each shocked market, and the statistics are taken over the resulting P&L.

The risk factors are the pillar zero rates of the market's curves. Each is named
`"<curve>/<pillar>y"`, for example `"discount:USD/5y"` or `"index:USD-SIMINDEX-6M/5y"`. Every
curve a trade reads must be included. This helper lists them for a market:

```python
def factor_labels(market: dict) -> list:
    labels = []
    for ccy, data in market["currencies"].items():
        curves = [(f"discount:{ccy}", data["discount_curve"])]
        curves += [(f"index:{name}", curve) for name, curve in sorted(data["index_curves"].items())]
        labels += [f"{name}/{t:g}y" for name, curve in curves for t in curve["times"]]
    return labels

FACTORS = factor_labels(MARKET)
print(FACTORS[:3])   # ['discount:USD/0y', 'discount:USD/1y', 'discount:USD/2y']
```

**Monte Carlo scenarios** draw correlated normal moves of the factors over the horizon from a
covariance matrix you supply (rows and columns in the order of `factors`):

```python
horizon_days, daily_vol = 10, 0.0008      # 8bp daily moves, correlated across pillars
covariance = [[daily_vol ** 2 * horizon_days * 0.9 ** abs(i - j) for j in range(len(FACTORS))]
              for i in range(len(FACTORS))]

result = run({
    "market": MARKET,
    "portfolio": [swap, european, note],
    "analytics": ["market_risk"],
    "config": {"market_risk": {
        "scenarios": {"source": "monte-carlo", "factors": FACTORS, "covariance": covariance,
                      "horizon_days": horizon_days, "num_scenarios": 4096, "seed": 1, "replicates": 8},
        "quantiles": [0.99, 0.975],
    }},
})

risk = result["market_risk"]
print("10-day VaR 99%:", risk["risk"]["VaR_99"])
print("10-day ES 97.5%:", risk["risk"]["ES_97.5"], "+/-", risk["risk"]["ES_97.5_standardError"])
```

VaR and ES are reported as positive losses, keyed by quantile: `VaR_99`, `ES_97.5`. With
`replicates` above 1 the scenarios are split into independently scrambled sets, and each
figure gets a `_standardError`; with one set the standard errors are `null`.

**Historical scenarios** replay observed moves instead. `history` is a table of past levels of
the factors, one row per day, oldest first. Each overlapping window of `horizon_days` becomes
one scenario:

```python
import random

random.seed(0)
today_levels = [r for ccy in MARKET["currencies"].values()
                for curve in [ccy["discount_curve"], *ccy["index_curves"].values()] for r in curve["rates"]]
history = [today_levels]
for _ in range(500):                              # 500 days of made-up daily moves
    history.append([level + random.gauss(0.0, 0.0006) for level in history[-1]])

result = run({
    "market": MARKET,
    "portfolio": [swap, european, note],
    "analytics": ["market_risk"],
    "config": {"market_risk": {
        "scenarios": {"source": "historical", "factors": FACTORS, "history": history, "horizon_days": 10},
    }},
})
print(result["market_risk"]["num_scenarios"], "scenarios; VaR 99%:", result["market_risk"]["risk"]["VaR_99"])
```

Historical standard errors come from a moving-block bootstrap. It is configured with
`bootstrap_replicates` (default 128), `block_length` and `bootstrap_seed`, and `dates` can
label each window.

Other `market_risk` settings: `quantiles` (default `[0.99, 0.975]`) and `batch_size`, the
number of scenarios revalued at once (default 256). The result also holds each trade's value
on today's curves (`trades`, `base_npv`), the factors shocked, and the P&L per scenario and
trade (`pnl`, see [Large arrays](#large-arrays)).

### `calibration`

Fits each currency's model volatility to a basket of market swaptions, and reports the fit.
Give the basket as matching lists of option expiries and swap terms:

```python
CALIBRATED_MODELS = {"ir": {"USD": {
    "model": "LGM", "reversion": 0.03,
    "calibration_expiries": ["1Y", "2Y", "5Y"],
    "calibration_terms": ["9Y", "8Y", "5Y"],          # co-terminal: every swap ends in 10 years
}}}

result = run({"market": MARKET, "analytics": ["calibration"], "config": {"models": CALIBRATED_MODELS}})

usd = result["calibration"]["currencies"]["USD"]
print(usd["model"], "volatility", usd["sigma_values"], "changing at", usd["sigma_times"], "error", usd["error"])
for helper in usd["helpers"]:
    print(f"{helper['expiry']} x {helper['term']}: market {helper['market_value']:.6f}  model {helper['model_value']:.6f}")
```

The calibration analytic needs no portfolio. The fitted volatility is piecewise constant:
`sigma_values[i]` applies between consecutive `sigma_times`. A model with a calibration basket
is calibrated the same way whenever it is used for `exposure`, so the two always agree.

### Several analytics at once

List several analytics to compute them in one run. Each reads the config sections it needs:

```python
result = run({
    "market": MARKET,
    "portfolio": [swap, european, bermudan],
    "analytics": ["npv", "exposure", "sensitivities", "calibration"],
    "config": {"models": CALIBRATED_MODELS, "simulation": {**SIMULATION, "samples": 256},
               "sensitivity": {"method": "AD"},
               "pricing": {"bermudan": {"n_per_std": 16}}},
    "outputs": {"npv_cube": "none"},
})
print(result["analytics"], result["npv"])
```

## Models and pricing engines

### Interest rate models

Each currency in `config.models.ir` has one model:

| Field | Meaning | Default |
|---|---|---|
| `model` | `"LGM"` (ORE's default) or `"HullWhite"` | `"LGM"` |
| `reversion` | Mean reversion | required |
| `volatility` | The volatility, used as given when there is no calibration basket, or the starting point of the calibration | `0.01` |
| `calibration_expiries`, `calibration_terms` | A basket of market swaptions to calibrate the volatility to | none |
| `solver` | Root solver of the calibration: `"Newton"` or `"Bisection"` | `"Newton"` |

A volatility can be a single number or piecewise constant in time. Here `values[i]` applies on
each interval before, between and after `times`, so there is one more value than times:

```json
{"model": "HullWhite", "reversion": 0.03,
 "volatility": {"times": [1.0, 3.0], "values": [0.008, 0.011, 0.009]}}
```

`models.equity_volatilities` sets the Black-Scholes volatility of each equity in the market,
and `models.bootstrap_tolerance` (default `1e-4`) the calibration's acceptance tolerance.

### Pricing engines

`config.pricing` chooses how each product is priced, today and on every path:

| Field | Meaning | Default |
|---|---|---|
| `european` | `"Bachelier"` on the market's swaption volatilities, or `"Jamshidian"` on a Hull-White model given in `jamshidian` | `"Bachelier"` |
| `jamshidian` | `{"reversion", "volatility"}` of the Jamshidian engine's model | none |
| `bermudan`, `american` | The grid engine of each: `reversion`, `volatility`, `calibration` (`"Bootstrap"` or `"None"`), `strategy` (`"CoterminalDealStrike"` or `"CoterminalATM"`), and grid density `n_per_std`, `std_devs`, `exercise_time_steps_per_year` | ORE's |

A coarser Bermudan grid (`n_per_std` 16 instead of 30) is much faster and is often good enough
for exposure studies:

```json
"pricing": {"european": "Jamshidian", "jamshidian": {"reversion": 0.03, "volatility": 0.01},
            "bermudan": {"n_per_std": 16}, "american": {"n_per_std": 16}}
```

## Precision

Precision is configured in `config.precision` and applies to the heavy, per-path parts of a
run: the simulation, the simulated market and the pricing on paths. Each of these stages has
three formats:

- `storage`: the format results are kept in between stages.
- `compute`: the format of the stage's matrix products.
- `accumulate`: the format the products accumulate in, and the rest of the stage runs in
  (`float32` or `float64`).

Everything defaults to `float64`. The formats are `float64`, `float32`, `float16`, `bfloat16`,
`float8_e4m3fn` and `float8_e5m2`. Compute also accepts `tensorfloat32`, `bfloat16_3x` (three
BF16 passes), and single-pass variants such as `bfloat16_1x`.

The stage settings apply to the per-path work of the `exposure` and `market_risk` analytics,
and a request that sets them without either is refused.

**Run everything in FP32.** The `profile` sets every stage to float32, and also the internal
arithmetic (calibration, today's values, Greeks and the reductions over paths), so it applies
to every analytic:

```python
FP32 = {"profile": "float32"}

result = run({
    "market": MARKET, "portfolio": [swap, european, note], "analytics": ["exposure"],
    "config": {"models": MODELS, "simulation": SIMULATION, "precision": FP32},
    "outputs": {"npv_cube": "none"},
})
print(result["exposure"]["netting_set"]["epe"])
```

**Set stages, products and trades individually.** A trade's own entry overrides its product's,
which overrides `pricing`:

```python
MIXED = {
    "simulation": {"storage": "float32", "compute": "float32", "accumulate": "float32"},
    "pricing": {"storage": "float32", "compute": "float32", "accumulate": "float32"},
    "by_product": {"european_swaption": {}},                  # Europeans back in float64
    "by_trade": {"note-2029": {"storage": "float32"}},       # the note stored in FP32, computed in FP64
}
```

**Go below 32 bits.** Storage below 32 bits keeps one power-of-two scale per block of 32 paths.
Rounding can be to nearest or stochastic (reproducible from `rounding_seed`). This example keeps
the simulated market and the cube in FP8, with pricing products in FP8 accumulated in FP32:

```python
FP8 = {
    "market": {"storage": "float8_e4m3fn"},
    "pricing": {"storage": "float8_e4m3fn", "compute": "float8_e4m3fn", "accumulate": "float32"},
    "rounding": "stochastic", "rounding_seed": 7,
}
```

**Native or emulated.** By default the engine runs a compute format only where the device
supports it natively. A CPU has no native arithmetic below float32, and not every GPU or TPU
supports every format. Such a run fails when it starts, saying which format needs emulation.
To allow software emulation, set `config.execution.mode` to `"allow_emulation"`. Storage
formats need no setting: any device can store any format.

```python
EMULATION = {"mode": "allow_emulation"}
```

**Measure the error against float64.** `paired_fraction` also runs that share of the paths in
float64, and each figure's precision report then compares the two:

```python
result = run({
    "market": MARKET, "portfolio": [swap, european, note], "analytics": ["exposure"],
    "config": {"models": MODELS, "simulation": SIMULATION,
               "precision": {**FP8, "paired_fraction": 0.25}, "execution": EMULATION},
    "outputs": {"npv_cube": "none"},
})

report = result["exposure"]["precision"]
print(report["realized"])          # the format each stored array was actually kept in
print(report["products"])          # each compute format, and whether it ran native or emulated
print(report["devices"], report["paths"], report["paired_paths"])   # ... 1024 256
for warning in result["warnings"]:
    print("warning:", warning)
```

Every exposure and market-risk result carries this report, under `exposure.precision` and
`market_risk.precision`. It holds the policy as run, the stored formats, the devices, the paired
sample's estimates, and whether each compute format ran natively or was emulated. Every result
also has `numerics`, the resolved precision of every stage and internal role. At a precision
below float64, the `warnings` list names every figure whose precision has not been validated
for that use. Warnings never stop a run.

The same `precision` section applies to market risk, where the `simulation` stage holds the
shocks and `pricing` the revaluation and P&L.

**Require a backend.** `config.execution.backend` states which kind of device the run must run
on: `"default"` (whatever the worker runs on), `"cpu"`, `"gpu"` or `"tpu"`. A run that asks for a
backend the worker does not have fails before any numerical work, rather than falling back to
another device:

```json
"execution": {"backend": "gpu", "mode": "native"}
```

See [Precision profiles and execution](../reference/explicit-precision-policy.md) and
[Selecting numerical experiments](../reference/numerical-experiments.md) for the advanced
options.

## Large arrays

The NPV cube (`[path, date, trade]`), the market-risk P&L (`[scenario, trade]`) and options'
exercise decisions (`[path, date, option]`) can be large. `outputs` chooses how each comes
back:

| Value | Meaning |
|---|---|
| `"artifact"` | A reference to binary chunks, downloaded separately. Default for `npv_cube` and `pnl` |
| `"inline"` | Nested JSON lists in the result |
| `"none"` | Left out. Default for `exercised`. `npv_cube: "none"` also saves the memory of the cube |

A reference describes everything needed to read the array: its `dtype`, `shape`, `axes`, the
trade order (`items`), and a list of `chunks`, each with a download `url` and a SHA-256 hash.
Chunks are rows of a little-endian, row-major array. To read one with NumPy:

```python
import hashlib

import numpy as np


def read_array(reference: dict) -> np.ndarray:
    """Download an array returned by reference, checking every hash."""
    whole = hashlib.sha256()
    parts = []
    for chunk in reference["chunks"]:
        data = requests.get(f"{BASE}{chunk['url']}", timeout=60).content
        if hashlib.sha256(data).hexdigest() != chunk["sha256"]:
            raise ValueError(f"chunk {chunk['url']} is corrupt")
        whole.update(data)
        parts.append(data)
    if whole.hexdigest() != reference["sha256"]:
        raise ValueError("array does not match its hash")
    dtype = np.dtype(reference["dtype"]).newbyteorder("<")
    return np.frombuffer(b"".join(parts), dtype=dtype).reshape(reference["shape"])


result = run({
    "market": MARKET, "portfolio": [swap, european], "analytics": ["exposure"],
    "config": {"models": MODELS, "simulation": SIMULATION},
})
reference = result["exposure"]["npv_cube"]
cube = read_array(reference)
print(cube.shape, reference["items"]["ids"])         # (1024, 7, 2) ['swap-5y', 'european-3y2y']
print("mean NPV per date:", cube.mean(axis=0))
```

## Retries, cancellation and retention

**Retrying safely.** If a client loses the response to a submission, it can submit again with
the same `idempotency_key` and get the original `run_id` back instead of a second run. Reusing a
key with a different body is refused with `409`:

```python
body = {"market": MARKET, "portfolio": [swap], "analytics": ["npv"], "idempotency_key": "eod-2026-07-30-book-7"}
first, retry = submit(body), submit(body)
print(first == retry)    # True
```

**Cancelling.** `POST /runs/{run_id}/cancel` cancels a queued run at once. A running run stops
at its next checkpoint (after calibration, the simulation, each trade's pricing, and so on) and
then reads `cancelled`:

```python
run_id = submit({"market": MARKET, "portfolio": [bermudan], "analytics": ["exposure"],
                 "config": {"models": CALIBRATED_MODELS, "simulation": SIMULATION}})
status = requests.post(f"{BASE}/runs/{run_id}/cancel", timeout=60).json()
print(status["status"], status["cancel_requested"])    # running True

status = requests.get(f"{BASE}/runs/{run_id}", params={"wait": 30}, timeout=60).json()
print(status["status"])                                # cancelled
```

**Retention.** Finished runs and their arrays are deleted after `JAX_RISK_RETENTION_DAYS` (30 by
default). A run submitted with `"keep": true` is never deleted. A deleted run answers `404`.

**Reproducing a result.** Every result carries `config`, the configuration as run with every
default filled in, and `engine`, the version and git revision of the engine that produced it.

## Errors

| Status | When |
|---|---|
| `422` | The body does not match the schema: a malformed or unknown field, an unknown analytic, product or format |
| `400` | The request is well formed but cannot be run: a missing section, a setting no requested analytic reads, a trade the market cannot value, an inconsistent precision |
| `404` | Unknown `run_id`, or a run past its retention period |
| `409` | An `idempotency_key` reused with a different body, or cancelling a run that already finished |
| `503` | This machine's engine worker is serving a different job queue |

The `detail` field names the problem:

```python
response = requests.post(f"{BASE}/runs", timeout=60, json={
    "market": MARKET, "analytics": ["npv"],
    "portfolio": [{**swap, "currency": "GBP"}],
})
print(response.status_code, response.json()["detail"])
# 400 trade 'swap-5y' (SwapConfig): no market for currency 'GBP'; have ['USD']
```

Because requests are checked before they are queued, most problems are reported this way. A
run that fails while running has status `failed`, with the error in `error` and its category in
`failure_class`: `missing-market-data`, `unsupported-product`, `numerical-failure`, `bad-terms`
or `infrastructure`.

## Running on accelerators

Nothing in the request changes between CPU, GPU and TPU: the engine worker runs on whatever
devices JAX finds on the machine. Install the `gpu` or `tpu` extra and start the server as
usual. See the [User Guide](user-guide.md) for setting up each platform.

- **Several devices.** Each run's paths, or market-risk scenarios, are split across every
  device of the machine. The split uses as many devices as divide the number of paths evenly,
  so a power-of-two `samples` uses them all. `JAX_RISK_SCENARIO_DEVICES` caps the number used,
  and each result's precision report lists the devices it ran on.
- **Several hosts.** On a multi-host TPU slice, start one worker per host with
  `jax-risk-worker --distributed --queue <file>` and the server with `JAX_RISK_WORKER=external`
  beside the first host. Every run is then split across the devices of all hosts.
- **Sharing a GPU.** The worker reserves most of the GPU's memory by default. Start the server
  with `XLA_PYTHON_CLIENT_PREALLOCATE=false` when other processes need the same GPU.
