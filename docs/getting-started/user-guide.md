# User Guide

This page is about *running* the code. For how it works internally, see
[Architecture](../concepts/architecture.md) and the per-stage deep dives
([Market Simulation](../concepts/market-simulation.md), [Instruments](../instruments/swaps.md),
[Risk Statistics](../risk/var_es.md)).

## Prerequisites

- Python 3.11 (`requires-python = ">=3.11"`; the project's `.venv/` was built against this
  version).
- The core dependencies declared in [`pyproject.toml`](../../pyproject.toml):
  `open-source-risk-engine` (the ORE Python bindings — see
  [Architecture: ORE as a dependency](../concepts/architecture.md#ore-as-a-dependency)), `pandas`,
  `jax`, `jaxlib`, `numpy`, `scipy`.
- Three optional extras: `api` (`fastapi`, `pydantic>=2`, `uvicorn[standard]` — needed only
  to run [the HTTP API](../reference/http-api.md)), `dev` (`pytest`, `httpx`, `jsonschema` —
  needed to run the test suite; `httpx` is required by FastAPI's own `TestClient`, and
  `jsonschema` is deliberately test-only, since the engine must emit correct EOD documents
  without depending on a validator to produce them — see
  [the EOD boundary doc](../reference/eod-integration.md)), and `profiling`
  (`xprof` — needed only to collect/view a profiler trace of a pricing job, see
  [Profiling a pricing job](#profiling-a-pricing-job)).

> **⚠ Run everything through the venv's own interpreter**, e.g.
> `.venv/Scripts/python.exe` on Windows (`.venv/bin/python` on Linux/macOS). A bare `python`
> may resolve to a system interpreter where `pydantic` and `jsonschema` are absent, which
> makes whole test files **silently uncollectable** rather than failing — see
> [I-25](../known-issues.md#i-25). The commands below write `python` for brevity; substitute
> the venv interpreter.

## Setting up

From the repository root, with your Python environment activated:

```bash
pip install -r requirements.txt
```

This installs the package itself (editable) plus the `api` and `dev` extras — equivalent to
`pip install -e .[api,dev]`, which `requirements.txt` wraps (`profiling` is not included;
install it separately if you need it). See
[`pyproject.toml`](../../pyproject.toml) for the actual dependency declarations; that file,
not `requirements.txt`, is the source of truth for which packages are needed). If you only need the core
engine as a library (no HTTP API, no test suite), `pip install -e .` alone is enough — see
[Architecture: ORE as a dependency](../concepts/architecture.md#ore-as-a-dependency) for
what stays a hard runtime dependency either way.

### Pinned versions

`requirements.txt` also applies [`constraints.txt`](../../constraints.txt), which pins every
package to the exact version the test suite was last verified against. The two files split
the job:

| File | Says | Example |
|---|---|---|
| `pyproject.toml` | which packages, and the range they must fall in | `jax>=0.10.2,<0.11` |
| `constraints.txt` | the exact versions verified | `jax==0.10.2` |

Pinning matters here more than in most projects. The ORE-parity tests assert agreement to
1e-12, and that holds only for the jax, jaxlib and `open-source-risk-engine` builds it was
measured on. `tests/test_environment.py` fails if the installed jax, jaxlib, ORE, numpy or
scipy differ from `constraints.txt`, so a drifted environment shows up as one named
package, not as a puzzling parity failure. It also fails if `pyproject.toml`'s ranges
exclude a pinned version, or if a declared dependency is missing from the lock.

`pip install -e .` without the constraints file installs the newest versions inside the
ranges. That is fine for using the engine, but parity results are only verified for the
pinned set.

**Upgrading a pin** (for example to a new jax):

1. In a fresh venv, `pip install -e ".[api,dev,profiling]" jax==<new> jaxlib==<new>`.
   Widen the range in `pyproject.toml` first if the new version falls outside it.
2. Run the **full** suite, not only the fast tier: several parity and Greeks tests are in
   the slow tier.
3. `pip freeze --exclude-editable > constraints.txt`, then restore the file's header comment.
4. Commit both files together.

The lock was frozen on Windows. On Linux, pip skips pins for packages it does not need
(`colorama`), and the Linux-only `uvloop` (pulled in by `uvicorn[standard]`) installs
unpinned. Neither affects numerical results.

The examples on this page assume you're running from the repository root. `engine` itself
is importable from anywhere once installed — `pip install -e .` puts it on the path, so
`python -m engine.simulation.market_model` and
`from engine.simulation.market_model import ...` work without any extra path setup and
without `cd`-ing anywhere in particular. What the repository root buys you is that the
**relative paths in these examples resolve**: `tests/fixtures/traderx-eod/...`,
`demos/demo.py`, `tests/`.

If you're using the project's own `.venv/` on Windows, replace `python` in the commands
below with `.venv\Scripts\python.exe` (or activate the venv first with
`.venv\Scripts\activate`). On Linux/macOS the interpreter is `.venv/bin/python`.

## Running the demos

All five demos live in [`demos/`](../../demos/). Run them from the repository root, as
written below — they import `engine`, which an editable install makes importable from any
directory, but the paths in these commands are relative to the root.

**The whole pipeline in one call:**
```bash
python demos/demo.py
```
Simulates a market, calibrates a volatility term structure, prices one of each instrument
type (swap, European/Bermudan/American swaption) via
[`engine.portfolio.price_portfolio`](../reference/portfolio-entrypoint.md), and prints
NPVs, the exposure profile and Bermudan Greeks, then ends with a 10-day market-risk
VaR/ES of the same portfolio — the same walkthrough the individual module demos below
show piece-by-piece, but as a single, realistic entry-point call rather than hand-wired
pipeline plumbing. Start here if you want to see the whole system working end to end before
digging into any one stage.

**The same portfolio, over the real HTTP API:**
```bash
python demos/demo_api.py
```
Requires the `api` extra (see [Running the API](#running-the-api) below). Launches its own
`uvicorn` server (or reuses one already running at `http://127.0.0.1:8000` if
`JAX_RISK_ENGINE_DEMO_SKIP_SERVER=1` is set), builds the same market/portfolio as `demo.py`
as a `PortfolioRequestSchema` JSON body, submits it to `POST /portfolio/price`, polls
`GET /portfolio/price/{job_id}` until it completes, and prints the same NPV/risk/Greeks
summary read back out of the JSON response — see [HTTP API](../reference/http-api.md) for
what's actually happening on the wire. Both Bermudan/American trades are left uncalibrated
(`hw_sigma: null`) and resolved server-side via a `calibration_basket` on the request — see
[HTTP API: Automatic calibration](../reference/http-api.md#automatic-calibration-hw_sigma-null--calibration_basket).

**The same portfolio again, restructured to show the shape of a real integration:**
```bash
python demos/demo_structured.py
```
Functionally identical to `demo_api.py` (same market, same portfolio, same HTTP calls), but
organized into four explicit, clearly labeled stages: **given inputs** (plain market/
portfolio facts — curve, model parameters, trades, market vol quotes — with zero server/
schema concepts in sight), **server setup** (pure infrastructure: get a running server,
independent of what portfolio you're about to price), **server inputs** (the mechanical
translation from stage 1's facts into `PortfolioRequestSchema`'s exact JSON shape), and
**submit and print** (send, poll, display). Useful as a template to copy from when wiring up
a real integration, since it makes explicit which parts of the script would change for a
different portfolio (stage 1) versus which parts wouldn't (stage 2) versus which parts are
pure boilerplate reshaping (stage 3).

**The same end-to-end path, sized so its profiler trace is readable:**
```bash
python demos/demo_profile_small.py
```
Exercises exactly what `demo_structured.py` does — calibration, simulation, all four
instrument pricers, risk and Greeks, over the real HTTP API, in a real pool worker, under
`jax.profiler.trace` — on a deliberately small portfolio, producing roughly a 41 MB trace in
about 25 seconds under `.profile-out-small/`. It trades portfolio realism for trace
ergonomics and nothing else. Note that it leaves Greeks **on**: Greeks is the one knob that
genuinely moves trace size (~5x), which is precisely why a trace without it would not
represent where this engine spends its time. See
[Profiling a pricing job](#profiling-a-pricing-job) below and
[Profiling & the Tracer](../concepts/profiling.md).

**How much precision market risk needs:**
```bash
python demos/demo_precision.py
```
Runs `engine.market_risk.run_market_risk` on a sloped two-curve market and a mixed
portfolio at FP64 and FP32 over five Sobol seeds, and compares the FP32 error in VaR 99%
and ES 97.5% with the Monte Carlo noise those numbers already carry (the ES standard error
and the spread across seeds).

Each pipeline module also has its own runnable demo in its own
`if __name__ == "__main__":` block, showing that module's public API used end-to-end
against a shared example scenario (see
[`engine/simulation/demo_scenarios.py`](../../engine/simulation/demo_scenarios.py)) — useful
when you want to see one stage in isolation.

**Market simulation:**
```bash
python -m engine.simulation.market_model
```
Prints the shapes of the simulated equity/rate paths and a sample of reconstructed
discount factors.

**Swap pricing** (runs the simulation internally first, to get a yield curve cube to
price against):
```bash
python -m engine.instruments.swap
```
Prints the resulting NPV cube's shape and its mean value at the first simulated time
step.

**European swaption pricing** (runs the simulation internally first, against a scenario
sized so several simulated steps land before the demo swaption's own exercise date):
```bash
python -m engine.instruments.european_swaption
```
Prints the resulting NPV cube's shape and its mean value at every simulated time step —
increasing as the exercise date approaches, then exactly `0.00` after it.

**Bermudan swaption pricing** (runs the simulation internally first; prices a Bermudan
swaption with several exercise dates against the simulated paths):
```bash
python -m engine.instruments.bermudan_swaption
```
Prints the swaption's baseline (t=0) NPV, then the resulting NPV cube's shape and its
mean value at every simulated time step.

**American swaption pricing** (runs the simulation internally first; prices an American
swaption, discretized into a grid of exercise dates over its exercise window, against the
simulated paths):
```bash
python -m engine.instruments.american_swaption
```
Prints how many discretized exercise dates the exercise window was converted into, the
resulting baseline (t=0) NPV, then the NPV cube's shape and its mean value at every
simulated time step.

**Risk statistics on a simulated cube** (runs simulation and pricing internally first):
```bash
python -m engine.risk.var_es
```
Prints the portfolio's baseline (t=0) value and loss quantiles of the simulated cube at
each requested confidence level, for every simulated time step. This exercises the
statistics functions; the engine's market-risk VaR is `demos/demo.py`'s last section.

This demo crashed until 2026-09-24, because its `SwapConfig` omitted `evaluation_date` and
scheduled off *today* against pillars pinned to 2026-07-30 ([I-28](../known-issues.md#i-28)).
`tests/test_risk_measure_label.py` now runs it.

## Running the tests

```bash
python -m pytest tests/ -m "not slow" -q    # fast tier: what CI runs on every push
python -m pytest tests/ -q                   # full suite
```

See each deep-dive doc's "Tested by" section for what's covered where, and
[Architecture: Testing philosophy](../concepts/architecture.md#testing-philosophy)
for the general approach (every formula is checked both for internal mathematical
correctness and against ORE's own installed software directly).

**Two tiers.** The full suite takes 22–27 minutes (see [Known Issues](../known-issues.md)
for the current verified figure), because the Monte Carlo and ORE-parity tests genuinely
simulate and reprice. Tests marked `@pytest.mark.slow` make up more than half of that time (101 of 2,087 tests);
the rest is the **fast tier**, `-m "not slow"`. The fast tier takes about 9½ minutes on the reference Windows machine and
9m32s on a 4-core Linux container (1,986 tests; the slow tier adds 12m58s there). A test is marked `slow` when either:

- it starts `engine.portfolio.worker_pool` processes (the job-submitting classes in
  `tests/test_api.py` and `tests/test_worker_pool.py`), or
- it takes 5 seconds or more.

**ORE-parity tests are never marked slow**, however long they take
(`tests/test_ore_*.py`, `tests/test_market_risk_ore_parity.py`). They are what the
pinned versions protect, so they run on every push. The slow tier is mostly portfolio
end-to-end runs, Bermudan Greeks, and compile-count checks. Give a new test the marker
if it meets either rule. `--strict-markers` is on, so a misspelt marker fails collection
and can't silently put a test in the wrong tier.

**CI.** [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) installs through
`requirements.txt` (so with the pinned versions) on Linux, Python 3.11, and runs the fast
tier on every push to `main` and every pull request. The full suite is the `full` job:
start it by hand from the repository's Actions tab ("Run workflow"). Run it before
merging anything that touches pricing, calibration or Greeks, and after upgrading a pin.
CI does not check out the `reference/` submodules; the one test that reads
`reference/traderX` skips without it.

For a fast inner loop while working on
the TraderX EOD boundary, the integration tests are a self-contained subset — 751 tests in
a few seconds:

```bash
python -m pytest tests/test_integration_*.py -q
```

They are fast because [`engine.integration`](../reference/eod-integration.md#module-map)
imports **no simulation pricer, no ORE builder and no curve construction** — it does import
`ORE` itself, which W1.2 permitted for date and day-count arithmetic, but nothing that
builds a pricing object. (Running them via `tests/` rather than by path still pays for
`tests/conftest.py`, which imports JAX at collection time to enable x64 for the rest of the
suite.)

`tests/conftest.py` provides shared `pytest` fixtures (the example scenario
configurations from `engine/simulation/demo_scenarios.py`, wrapped as fixtures, plus a
`portfolio_request` fixture and a `test_client` fixture for `engine.portfolio`/`engine.api`
tests) so individual test files don't each need to build their own copy of the same setup.

## Running the API

Requires the `api` extra (`pip install -e .[api]` — already included if you installed via
`requirements.txt`). Start the server:

```bash
.venv/Scripts/python.exe -m uvicorn engine.api.app:app --reload
```

Then visit `http://127.0.0.1:8000/docs` for FastAPI's interactive Swagger UI, or submit a
request directly:

```bash
curl -X POST http://127.0.0.1:8000/portfolio/price \
  -H "Content-Type: application/json" \
  -d '{
    "evaluation_date": "2026-07-30",
    "market": {
      "time_grid": [0.0, 0.5, 1.0, 1.5, 2.0],
      "equities": {"initial_prices": [100.0], "dividend_yields": [0.0], "rate_mapping": [[0.0]]},
      "rates": {"initial_rates": [0.03], "theta": [0.03], "mean_reversion": [0.03],
                "initial_zero_curves": [{"times": [0.0,1.0,2.0,5.0,10.0,30.0], "rates": [0.03,0.03,0.03,0.03,0.03,0.03]}]},
      "joint_covariance": [[0.04, 0.0], [0.0, 0.0001]],
      "scenarios": 4096
    },
    "trades": [{"trade_type": "swap", "notional": 1000000.0, "fixed_rate": 0.032, "payer": true,
                "discount_curve_index": 0, "forward_curve_index": 0, "swap_tenor": "2Y"}],
    "pfe_quantiles": [0.95, 0.99]
  }'
```

This returns `202 Accepted` with a `job_id` — pricing runs in the background (see
[HTTP API: Why async, not sync](../reference/http-api.md#why-async-not-sync-the-measured-latency)
for why). Poll for the result:

```bash
curl http://127.0.0.1:8000/portfolio/price/<job_id>
```

See [HTTP API](../reference/http-api.md) for the full endpoint reference, request/response
schemas, and the async job pattern's reasoning.

The same app also mounts a second router at `/eod`, the TraderX overnight batch contract —
see [Pricing a TraderX EOD bundle](#pricing-a-traderx-eod-bundle) below.

## Pricing a TraderX EOD bundle

The same server also exposes a **second, separate contract** under `/eod` — the overnight
batch boundary for TraderX. It is worth knowing which one you are talking to, because they
behave differently on purpose:

| | `/portfolio/price` | `/eod/price` |
|---|---|---|
| Shape | **Asynchronous** — `202` + `job_id`, then poll | **Synchronous** — one call returns the result |
| Body | `PortfolioRequestSchema` (Pydantic) | `EodSubmissionSchema`, pointing at a bundle on disk |
| Durability | In-memory `_JOBS`, lost on restart ([I-08](../known-issues.md#i-08)) | Published to a crash-safe store; survives restart |
| Refusals | An unsupported trade is an error | An unsupported instrument is a **`200`** whose coverage names the refusal |

That last row is the design: returning an HTTP error for a refusal would make "we correctly
declined to guess" indistinguishable from "we broke." See
[the EOD boundary doc](../reference/eod-integration.md) for the reasoning behind all four.

**Ask what the engine can price, before submitting anything:**

```bash
curl http://127.0.0.1:8000/eod/capabilities
```

Returns the capability document — supported products, conventions, calculations, market-input
modes and known limitations — derived from the allowlist on every call, so it cannot go stale
relative to the engine serving it. This is what makes "no silent exclusions" checkable in
advance rather than discovered afterwards.

**Price a bundle:**

```bash
curl -X POST http://127.0.0.1:8000/eod/price \
  -H "Content-Type: application/json" \
  -d '{
    "bundlePath": "tests/fixtures/traderx-eod/bill/v2",
    "marketInputs": {"mode": "assumed-profile", "assumedProfileId": "flat-3pct-v1"},
    "submissionId": "batch-2025-06-02-001"
  }'
```

Only `bundlePath` is required, and it must point at a **versioned** bundle directory
(`.../bill/v2`, not `.../bill`). The vendored fixtures under
[`tests/fixtures/traderx-eod/`](../../tests/fixtures/traderx-eod/) — `bill`, `note`, `sofr`,
`equity`, each with a `v1` and `v2` — are real delivered TraderX bundles and are the easiest
thing to try this against.

- **`marketInputs` is optional, and omitting it prices nothing** rather than falling back to
  an assumed curve. There is deliberately no default profile at this layer: the fallback this
  contract refuses to have would have to be introduced right here, so its absence is stated
  rather than implied.
- **`submissionId` is a caller-generated idempotency key.** Retrying a lost response with the
  same value recovers the *same* attempt instead of launching a second overnight batch — and
  since W0.8 that holds across an engine restart too. Reusing it with *different* inputs is a
  `409`, not a silently different answer.
- **`reuseExistingResult`** (default `true`) controls whether a previously completed result
  may be served. `false` means "don't serve me a cache" — it does not disable `submissionId`
  idempotency.

The response is a completed attempt. Against the `bill/v2` fixture above:

```json
{
  "workloadKey": "sha256:0564833c09a8e927...",
  "attemptId": "...",
  "state": "completed",
  "reused": false,
  "result": {
    "itemOrder": {"scheme": "traderx-item-v1", "itemCount": 2, "itemIds": ["389c658c...", "f453f240..."], "sha256": "5475633f..."},
    "items": [
      {
        "itemId": "389c658c1cc130bc4acea7496b59bd82",
        "sourceIdentity": {"kind": "position", "accountId": "22214", "security": "UST-BILL-20251202"},
        "calculations": {
          "npv": {
            "status": "ok",
            "value": 98507.14563826029,
            "method": "discounted-cashflow",
            "discountFactor": 0.9850714563826029,
            "dayCount": "ACT/365 (Fixed)",
            "curveProvenance": {"curveId": "flat-3pct-v1", "inputOrigin": "assumed", "construction": "flat-constant"}
          }
        }
      }
    ],
    "coverage": {"byCalculation": {"npv": {"ok": 2, "unsupported": 0, "unavailable": 0, "failed": 0, "notApplicable": 0}}}
  }
}
```

Two things in there carry most of the contract. **Every calculation has a `status`**, so a
number and a refusal are distinguishable per instrument per calculation rather than lumped
into one job-level verdict — and `coverage` counts those statuses so a coordinator can assert
a batch was complete without walking every item. **`curveProvenance` travels with the
number**, so `"inputOrigin": "assumed"` is visible on the price itself rather than buried in
a submission you would have to go back and find.

Try the same call against `tests/fixtures/traderx-eod/bill/v1` to see the other half: it
returns `200` with `npv` counted as `unsupported: 2` — a refusal, reported rather than raised.

**Look a result up again**, either by its workload key (the content-addressed identity of the
computation) or by attempt id:

```bash
curl http://127.0.0.1:8000/eod/results/by-workload/<workload_key>
curl http://127.0.0.1:8000/eod/attempts/<attempt_id>
```

Both survive a restart, because terminal attempts are published to a durable store rather
than held in memory — see
[EOD: W0.8](../reference/eod-integration.md#w08--crash-safe-publication-and-the-durable-result-store)
for the four-step protocol and where the store lives (`JAX_EOD_STORE_ROOT`, else a per-user
directory under the system temp root).

The published JSON Schemas for the result and capability documents are served alongside:

```bash
curl http://127.0.0.1:8000/eod/schemas/result
curl http://127.0.0.1:8000/eod/schemas/capabilities
```

## Writing your own market simulation config

`generate_paths()` takes a `SimulationConfig` — see
[API Reference: SimulationConfig](../reference/api-reference.md#simulationconfig) for every field.
Here's a minimal, verified-working example with one equity and one interest rate curve:

```python
from engine.simulation.market_model import (
    SimulationConfig, EquityConfig, RatesConfig, ZeroCurveConfig, generate_paths,
)

config = SimulationConfig(
    time_grid=[0.0, 0.5, 1.0],       # simulate out to 1 year, in two steps
    scenarios=1024,                   # number of simulated alternate futures
    equities=EquityConfig(
        initial_prices=[100.0],       # one stock, starting at $100
        dividend_yields=[0.0],
        rate_mapping=[[1.0]],         # this stock's drift depends on the one rate factor below
    ),
    rates=RatesConfig(
        initial_rates=[0.03],         # 3% starting interest rate
        theta=[0.03],                 # long-run mean-reversion target
        mean_reversion=[0.1],
        maturities=[1.0, 5.0],        # request discount factors for 1Y and 5Y
        initial_zero_curves=[
            ZeroCurveConfig(times=[0.0, 1.0, 5.0, 10.0], rates=[0.03, 0.03, 0.03, 0.03]),
        ],
    ),
    joint_covariance=[                # [equity, rate] x [equity, rate] covariance matrix
        [0.04, 0.0],
        [0.0, 0.0001],
    ],
)

result = generate_paths(config)
print(result["equities"].shape)      # (1024, 2, 1)  ->  [Scenarios, TimeSteps, NumEquities]
print(result["yield_curves"].shape)  # (1024, 2, 2, 1)  ->  [Scenarios, TimeSteps, Maturities, NumRates]
```

A few things worth knowing before writing your own config:

- **`joint_covariance`'s row/column order is equities first, then rates**, in the same
  order they appear in `equities.initial_prices` and `rates.initial_rates`.
- **`rates.initial_zero_curves` must have exactly one entry per rate factor** — see
  [Market Simulation: one curve per rate factor](../concepts/market-simulation.md#phase-3--yield-curve-reconstruction).
  A mismatched count raises a clear `ValueError`.
- **`rates.maturities` is optional** — omit it (and `initial_zero_curves`) if you only
  need the raw simulated rate/equity paths and not a full discount-factor cube. It's
  required if you intend to price any trade against this simulation (see next section).

## Pricing a swap

Pricing a swap requires **two** things to line up with each other: the simulation needs
at least two rate factors (one to discount cashflows, one to set floating payments — see
[Instruments: multi-curve discounting](../instruments/swaps.md#1-describing-a-swap-swapconfig)),
and `rates.maturities` must be set to the *exact* payment/accrual dates the swap will
generate (see
[Instruments: maturity-pillar alignment](../instruments/swaps.md#a-known-limitation-maturity-pillar-alignment))
— it is **not** simply "any list of future dates you want discount factors for," as the
minimal example above used. This is a self-contained, verified-working example for a 1
year swap:

```python
import ORE
from engine.simulation.market_model import (
    SimulationConfig, EquityConfig, RatesConfig, ZeroCurveConfig, generate_paths,
)
from engine.instruments.swap import SwapConfig, price_swaps

# These specific times are the swap's own accrual/payment dates -- for a 1Y swap
# with a 6-month floating index, starting at the standard 2-day spot lag. Computing
# these by hand is exactly the fiddly work ORE's schedule-building code does for
# you (see docs/instruments/swaps.md) -- in practice, build the swap first, inspect
# its schedule, and pass those dates into the simulation config's maturities.
maturities = [0.010958904109589041, 0.5150684931506849, 1.010958904109589]

config = SimulationConfig(
    time_grid=[0.0, 0.5, 1.0],
    scenarios=1024,
    equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[1.0, 0.0]]),
    rates=RatesConfig(
        initial_rates=[0.030, 0.032],   # two factors: discounting, forwarding
        theta=[0.030, 0.032],
        mean_reversion=[0.1, 0.03],
        maturities=maturities,
        initial_zero_curves=[
            ZeroCurveConfig(times=[0.0, 1.0, 5.0], rates=[0.030, 0.030, 0.030]),
            ZeroCurveConfig(times=[0.0, 1.0, 5.0], rates=[0.032, 0.032, 0.032]),
        ],
    ),
    joint_covariance=[
        [0.0400, 0.0000, 0.0000],
        [0.0000, 0.0001, 0.00003],
        [0.0000, 0.00003, 0.0001],
    ],
)
market = generate_paths(config)

swap = SwapConfig(
    notional=1_000_000.0,
    fixed_rate=0.031,
    payer=True,                    # this side pays fixed, receives floating
    discount_curve_index=0,        # which rate factor discounts cashflows
    forward_curve_index=1,         # which rate factor sets floating payments
    swap_tenor="1Y",
    evaluation_date=ORE.Date(30, 7, 2026),
)

npv_cube = price_swaps(market["yield_curves"], config.rates.maturities, [swap])
print(npv_cube.shape)              # (1024, 2, 1)  ->  [Scenarios, TimeSteps, Trades]
```

`price_swaps` accepts a *list* of `SwapConfig` objects — pass several to price a whole
portfolio at once; the output's last axis (`Trades`) will have one entry per swap, in
the order given.

## Pricing a swaption

Unlike `price_swaps`, `price_swaptions` works directly off the simulated Hull-White rate
paths (`generate_paths(...)["rates"]`), not the yield-curve cube — so `rates.maturities`
doesn't need to be set at all, and there's no maturity-pillar-alignment requirement to
satisfy. It does need the simulation's own `hw_a`/`hw_sigma`/zero-curve for the rate
factor being priced off (see
[Instruments: European Swaptions](../instruments/european-swaptions.md#1-describing-a-swaption-swaptionconfig)
for why). This is a self-contained, verified-working example for a swaption exercisable
in 3 years, on a 2-year underlying swap:

```python
import jax.numpy as jnp
import ORE
from engine.simulation.market_model import (
    SimulationConfig, EquityConfig, RatesConfig, ZeroCurveConfig, generate_paths,
)
from engine.instruments.european_swaption import SwaptionConfig, price_swaptions

config = SimulationConfig(
    time_grid=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],  # simulate out to 5 years
    scenarios=1024,
    equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[0.0]]),
    rates=RatesConfig(
        initial_rates=[0.03],
        theta=[0.03],
        mean_reversion=[0.03],
        # no `maturities` needed -- the swaption pricer works off the raw
        # rate paths, not a yield_curves cube.
    ),
    joint_covariance=[
        [0.0400, 0.0000],
        [0.0000, 0.0001],
    ],
)
market = generate_paths(config)

swaption = SwaptionConfig(
    notional=1_000_000.0,
    fixed_rate=0.03,
    payer=True,
    rate_factor_index=0,
    hw_a=0.03,              # must match config.rates.mean_reversion[0]
    hw_sigma=0.01,          # must match the volatility implied by joint_covariance for this factor
    initial_zero_curve=ZeroCurveConfig(
        times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[0.03] * 6,
    ),
    swap_tenor="2Y",
    forward_start=ORE.Period(3, ORE.Years),  # exercisable in 3 years
    evaluation_date=ORE.Date(30, 7, 2026),
)

step_times = jnp.array(config.time_grid[1:])
npv_cube = price_swaptions(market["rates"], step_times, [swaption])
print(npv_cube.shape)                         # (1024, 5, 1)  ->  [Scenarios, TimeSteps, Trades]
print(float(npv_cube[:, 2, 0].mean()))         # mean NPV at t=3.0 (just before exercise): > 0
print(float(npv_cube[:, 4, 0].mean()))         # mean NPV at t=5.0 (after exercise): exactly 0.0
```

`price_swaptions` accepts a *list* of `SwaptionConfig` objects, exactly like `price_swaps`
— several swaptions price into one `[Scenarios, TimeSteps, Trades]` cube, one entry per
trade in `Trades`, in the order given.

For swaptions with multiple exercise dates (Bermudan) or a continuous exercise window
(American), see
[American & Bermudan Swaptions](../instruments/american-bermudan-swaptions.md) and
[API Reference](../reference/api-reference.md#engineinstrumentsbermudan_swaption) —
`BermudanSwaptionConfig`/`price_bermudan_swaptions` and
`AmericanSwaptionConfig`/`price_american_swaptions` follow the same
list-of-configs-in, NPV-cube-out pattern as `price_swaptions` above.

## Computing market risk (VaR / ES)

Short-horizon VaR and ES come from revaluing the portfolio at t=0 under shocked curves
([Market Risk](../risk/market-risk.md)):

```python
from engine.market_risk import MarketRiskRequest, RateRiskFactors, monte_carlo_scenarios, run_market_risk

factors = RateRiskFactors.from_curves([zero_curve_config], names=["USD"])
scenarios = monte_carlo_scenarios(factors, covariance, horizon_days=10, num_scenarios=4096, seed=1)
result = run_market_risk(MarketRiskRequest(trades, scenarios, quantiles=(0.99, 0.975)))
print(result.risk["VaR_99"], result.risk["ES_97.5"])
```

`covariance` is the `[F, F]` covariance of 10-day absolute moves of the curve pillars, in
`factors.labels()` order; `historical_scenarios(factors, history, horizon_days=10)` uses
observed moves instead. The end of `demos/demo.py` is a complete example.

## Exposure profiles

`price_portfolio`'s multi-step simulation yields exposure through time, not VaR:
`result.exposure.epe`, `.ene`, `.ee_b`, `.eee_b` and `.pfe["PFE_95"]`, one entry per date
starting at t=0 ([Exposure](../risk/exposure.md)).

## Computing VaR/ES statistics on your own cube

```python
from engine.risk.var_es import compute_risk_metrics

# base_npv: the portfolio's actual value today, from a separate zero-shock
# revaluation -- see engine/simulation/demo_scenarios.py's flat_yield_curves()
# for a worked example of building one directly from ORE's own curve objects.
metrics = compute_risk_metrics(npv_cube, base_npv, percentiles=(0.95, 0.99))

print(metrics["VaR_95"])   # [TimeSteps] array
print(metrics["ES_99"])    # [TimeSteps] array
```

See [Risk Statistics: the P&L baseline](../risk/var_es.md#the-pl-baseline-what-are-gainslosses-measured-against)
for exactly what `base_npv` should be and why it can't be inferred automatically from
the NPV cube itself. **`ES_*` values can be `NaN`** for a given time step if there were
no simulated losses severe enough to have anything "worse than the VaR cutoff" — check
for this explicitly rather than assuming a numeric result (see
[Risk Statistics: the formulas](../risk/var_es.md#the-formulas)).

## Precision (float32 vs float64)

`generate_paths(config, precision=64)` (the default) runs in 64-bit precision. Pass
`precision=32` to run in 32-bit instead — see
[Architecture: Adjustable precision](../concepts/architecture.md#adjustable-precision) for what
this changes and why it's a single, per-call argument rather than something set once
globally by the caller.

## Profiling a pricing job

To see where a pricing job's wall clock goes — XLA compilation, XLA execution, or Python
(ORE calls, dispatch, the calibration bisection) — collect an
[XProf](https://github.com/openxla/xprof) trace and open it in the TensorBoard-style
profiler UI.

**1. Install the `profiling` extra** (`xprof`, which bundles the profiler plugin and a
standalone `xprof` viewer):

```bash
pip install -e .[api,profiling]
```

**2. Run a job with the profiler enabled.** The hook lives in
`engine.portfolio.worker_pool._run_pricing_job` and is **opt-in**: it does nothing unless
the environment variable `JAX_RISK_PROFILE_DIR` is set, in which case it wraps the
`price_portfolio` call in `jax.profiler.trace(...)` and writes a trace into
`$JAX_RISK_PROFILE_DIR/pid-<pid>/` (one subdir per process — safe whether the job runs
directly or fans out across pool workers).

This traces the job **as it actually runs in a fresh worker — XLA lowering and compilation
included**, not just steady-state execution. That is on purpose: for this engine the
compilation cost is a first-class thing to measure (the Bermudan/American tree pricers and
the LGM calibration bisection lower a number of `jit` programs — on a small portfolio that
compilation *is* most of the wall time, and the trace's "mostly Python" flame graph is
largely XLA lowering, which is real work).

Set `JAX_RISK_PROFILE_WARMUP=1` to run the job once and discard it before the trace opens,
so the traced run measures **warm steady-state execution** against populated compilation
caches instead. Which default you want depends on the question: leave it off for "what does
this job cost from cold," turn it on for "where does the *execution* time go."

> **Background:** why compilation dominates, and what was done to reduce it (the Bermudan
> Greeks path went from ~600 XLA compilations per job to 13), is written up in
> [Profiling & the Tracer](../concepts/profiling.md).

The turnkey way is [`demos/demo_structured.py`](../../demos/demo_structured.py), which
launches its own API server **with the profiler already on** (it sets `JAX_RISK_PROFILE_DIR`
for that server, defaulting to `./.profile-out`):

```bash
python demos/demo_structured.py
# -> "pricing-job profiler ON -> traces in '.profile-out' ..."
# -> .profile-out/pid-<worker-pid>/plugins/profile/<timestamp>/*.xplane.pb
```

To profile a single job directly, with no server or pool, set the variable yourself and
call `_run_pricing_job` on a frozen request:

```python
import os
os.environ["JAX_RISK_PROFILE_DIR"] = ".profile-out"   # set before the call

from dataclasses import replace
from engine.portfolio.worker_pool import _run_pricing_job, _freeze_trade
# build `request` as a PortfolioRequest (see "Running the demos" / demos/demo.py)
frozen = replace(request, trades=[_freeze_trade(t) for t in request.trades])
_run_pricing_job(frozen)
```

**3. Open the timeline:**

```bash
xprof --port 8791 .profile-out
```

Then open `http://localhost:8791`, pick the `pid-<pid>` entry under **Runs**, then pick a
tool from the **Tools** dropdown:

- **`overview_page`** — start here. It splits the step time into compilation vs. execution
  vs. input/other, which is the top-level number for this engine.
- **`framework_op_stats`** / **`hlo_stats`** / **`op_profile`** — aggregate every op with an
  on-device vs. on-host column; use these for which ops dominate.
- **`trace_viewer`** — the timeline. On the **CPU backend there is no separate device row**:
  XLA kernels run on the same host threads as the Python driver (`tf_XLAPjRtCpuClient`,
  `Thread_*`), tagged as `xla_op`, interleaved with dispatch. `tf_PjRtCompilerThreadPool`
  and `tf_xla-cpu-codegen` are the background compilation threads. A tall Python stack
  (`price_portfolio` → `scan` → `_run_python_pjit` → `_uncached_lowering` →
  `compile_or_get_cached`) is XLA *lowering/compilation*, not the math.

**Finding your way around the timeline.** The pricing path is annotated with named regions
— `calibration`, `simulation`, `pricing`, `base_npv`, `risk`, `greeks`, and one
`greeks/trade<i>/<Type>` per trade — so you can attribute time per phase and per trade
without turning the (very expensive) Python tracer on. See
[Profiling & the Tracer §4](../concepts/profiling.md).

If a small-portfolio trace looks entirely compile-bound, that is the real result — see
step 2. Profile a bigger portfolio (more trades, `scenarios` 16k+) to see execution take
over; drop `compute_greeks` if you only care about the forward pricing path, which is
roughly a 5x difference in trace size.

**A truncated trace looks exactly like a complete one.** The profiler's event buffer is a
fixed ~1M-event cap with no backpressure — once full, the rest is dropped silently. Every
traced job now self-checks for this and emits a `UserWarning` if the captured events span
far less than the job's wall time; heed it rather than trusting a partial timeline.

Unset `JAX_RISK_PROFILE_DIR` (or run any other demo/test — none of them set it) to go back
to zero-overhead normal runs; the hook is completely inert when the variable is absent.
