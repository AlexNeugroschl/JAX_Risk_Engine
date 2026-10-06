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
- Four optional extras: `api` (`fastapi`, `pydantic>=2`, `uvicorn[standard]` — needed only
  to run [the HTTP API](../reference/http-api.md)), `dev` (`pytest`, `pytest-xdist`, `httpx`,
  `jsonschema` — needed to run the test suite; `pytest-xdist` runs it in parallel
  processes, `httpx` is required by FastAPI's own `TestClient`, and
  `jsonschema` is deliberately test-only, since the engine must emit correct EOD documents
  without depending on a validator to produce them — see
  [the EOD boundary doc](../reference/eod-integration.md)), and `profiling`
  (`xprof` — needed only to collect/view a profiler trace of a pricing job, see
  [Profiling a pricing job](#profiling-a-pricing-job)), and `gpu` (JAX's CUDA 13 plugin, Linux
  or WSL2 only — see [On a GPU](#on-a-gpu-linux-or-wsl2-on-windows)).

> **⚠ Run everything through the venv's own interpreter**, e.g.
> `.venv/Scripts/python.exe` on Windows (`.venv/bin/python` on Linux/macOS). A bare `python`
> may resolve to a system interpreter where `pydantic` and `jsonschema` are absent, which
> makes whole test files **silently uncollectable** rather than failing — see
> [I-25](../planning/known-issues.md#i-25). The commands below write `python` for brevity; substitute
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

The lock was frozen on Windows, with a Linux block at its end: the `gpu` extra's packages and
the Linux-only `uvloop` (pulled in by `uvicorn[standard]`), frozen from the GPU environment
below. pip skips pins for packages a platform does not need (`colorama` on Linux, the Linux
block on Windows).

### On a GPU (Linux, or WSL2 on Windows)

The engine runs on an NVIDIA GPU through JAX's CUDA 13 plugin, the `gpu` extra. JAX publishes
no CUDA build for native Windows, so on Windows it runs under WSL2. Verified on the
reference machine's RTX 5060 Laptop GPU (Blackwell, 8 GB) under WSL2 Ubuntu 24.04
(roadmap 2.2). It needs an NVIDIA driver of 580 or later, which on WSL2 is the Windows driver
(nothing NVIDIA is installed inside WSL; the CUDA libraries come as pip wheels with the
extra). `nvidia-smi` inside WSL should list the card.

```bash
# In WSL2: clone onto the Linux filesystem, not under /mnt/c
git clone https://github.com/AlexNeugroschl/JAX_Risk_Engine.git ~/JAX_Risk_Engine
cd ~/JAX_Risk_Engine
python3.11 -m venv .venv          # any Python 3.11, e.g. conda-forge's
.venv/bin/python -m pip install -c constraints.txt -e ".[api,dev,gpu]"   # requirements.txt plus the extra
.venv/bin/python -c "import jax; print(jax.devices())"   # [CudaDevice(id=0)]
```

The clone has to be on WSL's own filesystem: the job queue and the engine worker's lock are
SQLite and OS file locks, which are unreliable on the Windows drive as WSL mounts it
(`/mnt/c`), and file access there is several times slower. The extra brings about 3 GB of
CUDA wheels.

**What the engine sets, and what you set.** Importing `engine` changes no device setting:
only x64 (decision A-22). The engine's matrix products state their own precision, so a float32
run is float32 on any device (see [Precision](#precision-float32-vs-float64)), and the engine
worker runs deterministic kernels (below). How a process holds GPU memory is the deployment's
choice, made in the environment before the process first opens the GPU:

| Who | GPU preallocation (`XLA_PYTHON_CLIENT_PREALLOCATE`) | Deterministic kernels (`XLA_FLAGS`) |
|---|---|---|
| A served API's engine worker | JAX's default: 75% of the card, which suits one worker per GPU. Set `false` where other processes share the card | Added by the worker unless your flags name one |
| The demos that start a server (`demo_api.py`, `demo_structured.py`, `demo_profile_small.py`) | `false` (`demos/demo_http.py`), unless your environment sets it | As any engine worker |
| The test suite | `false` (`tests/conftest.py`), unless set | Added (`tests/conftest.py`), unless set |
| Your own script or notebook | JAX's default; set `false` to share the card | JAX's default; set the flag for reproducible bits |

**Device memory.** XLA's default is to take 75% of a GPU's memory in every process that opens
it, which leaves nothing for a second process: on WSL2 the default's 6 GB request failed on the
8 GB card while another process held it, and XLA logged out-of-memory errors until a 4 GB one
succeeded. With `XLA_PYTHON_CLIENT_PREALLOCATE=false` a process holds what it has used and grows
on demand. To run your own server beside a notebook or the tests on one card:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python -m uvicorn engine.api.app:app
```

The API server keeps its own JAX on the CPU (it prices nothing; `/version` and
`/calibration/lgm` are its only JAX work), so the engine worker is the one process with a GPU
client. `/version` therefore says `cpu`; a job's result names the GPU in its `precision`
report. The demo's whole job peaked at 2.6 GB of the card.

**Tests on the GPU.** Each test process opens its own GPU client (the suite turns
preallocation off and deterministic kernels on for itself, in `tests/conftest.py`), so choose
`-n` by device memory as well as host memory: `-n 4` on the 8 GB card, where the full suite
peaked at 5.5 GB of the card and took 22 minutes. Without a GPU plugin installed the same commands run on the CPU,
as before.

**What to expect.** Results equal the CPU's to the last bits, not bit for bit (a GPU sums in
a different order): the parity suites hold at their tolerances. A job is the same bits on
every run of its compiled program: the engine worker adds `--xla_gpu_exclude_nondeterministic_ops=true`
to its own `XLA_FLAGS` at start-up (without it the AD Greeks moved by an ulp between
identical runs, the GPU's atomics adding in whatever order they land), and the test suite
does the same. A program compiled again can differ by an ulp, most likely because the compiler tunes its
choice of kernels on the card: the demo's job, compiled from scratch on 2026-10-05 and
2026-10-06, matched to the last bit within each day but differed by about an ulp between them
(the cube by 4e-11 at a scale of 1e6, the Greeks by 3e-13 at 571), with the same code.
`--xla_gpu_deterministic_ops=true` instead also pins that choice, at twice the compile time;
either one set by you, true or false, wins. Pricing
in your own process (`price_portfolio` in a script or notebook) gets JAX's default unless
you set the flag before JAX opens the GPU:

```python
import os
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_exclude_nondeterministic_ops=true").strip()
from engine.portfolio import price_portfolio   # before the first JAX computation
```

This card's float64 runs at 1/64 of its float32 rate, and a job of a few hundred paths is too
small to fill it, so the demo is slower than on the CPU (a repeat: about 6 s against 2.4 s). A consumer GPU is where the
GPU path is checked, not where speed is measured (roadmap 5.2). Profiling works as on the
CPU, and the trace gains the GPU's own lanes, but tracing a GPU job costs several times its
untraced time ([profiling §2.0](../concepts/profiling.md#20-the-demo-measured-2026-10-05-roadmap-21)).

The examples on this page assume you're running from the repository root. `engine` itself
is importable from anywhere once installed — `pip install -e .` puts it on the path, so
`from engine.portfolio import ...` works without any extra path setup and
without `cd`-ing anywhere in particular. What the repository root buys you is that the
**relative paths in these examples resolve**: `tests/fixtures/traderx-eod/...`,
`demos/demo.py`, `tests/`.

If you're using the project's own `.venv/` on Windows, replace `python` in the commands
below with `.venv\Scripts\python.exe` (or activate the venv first with
`.venv\Scripts\activate`). On Linux/macOS the interpreter is `.venv/bin/python`.

## Running the demos

All six demos live in [`demos/`](../../demos/). Run them from the repository root, as
written below — they import `engine`, which an editable install makes importable from any
directory, but the paths in these commands are relative to the root.

**The whole pipeline in one call:**
```bash
python demos/demo.py
```
Builds today's USD market (a curve rising from 3% to 5%), a run configuration with the
Hull-White model calibrated to the market's swaption volatilities, and one of each trade
type (swap, European/Bermudan/American swaption, Treasury note); prices them in one
[`engine.portfolio.price_portfolio`](../reference/portfolio-entrypoint.md) call; prints the
NPVs on every date, the exposure profile and the Bermudan's Greeks; shows the same run under
the LGM (the model is one field of the configuration); and ends with a 10-day market-risk
VaR/ES of the same portfolio. Start here to see the whole system working end to end.

**The same portfolio, over the real HTTP API:**
```bash
python demos/demo_api.py
```
Requires the `api` extra (see [Running the API](#running-the-api) below). Launches its own
`uvicorn` server (or reuses one already running at `http://127.0.0.1:8000` if
`JAX_RISK_ENGINE_DEMO_SKIP_SERVER=1` is set), builds the same market, portfolio and
configuration as `demo.py` as the portfolio request's JSON, submits it to
`POST /portfolio/price`, polls `GET /portfolio/price/{job_id}` until it completes, and prints
the same summary read back out of the JSON response — see [HTTP API](../reference/http-api.md)
for what's on the wire.

**The same portfolio again, restructured to show the shape of a real integration:**
```bash
python demos/demo_structured.py
```
Functionally identical to `demo_api.py`, but organized into four labeled stages: **given
inputs** (plain market/portfolio facts — curves, volatility quotes, the model, trades — with
no server or schema concepts), **server setup** (infrastructure, independent of the
portfolio), **server inputs** (the mechanical translation of stage 1 into the request's
JSON), and **submit and print**. A template for a real integration: it makes explicit which
parts change for a different portfolio (stage 1), which don't (stage 2), and which are
boilerplate reshaping (stage 3).

**The same end-to-end path, sized for a profiler trace:**
```bash
python demos/demo_profile_small.py                          # traces a repeat of the job
python demos/demo_profile_small.py --cold                   # traces the first run
python demos/demo_profile_small.py --cold --no-disk-cache   # ... compiling everything
```
Exercises what `demo_structured.py` does — calibration, simulation, every trade type,
exposure and Greeks, over the real HTTP API, in the real engine worker, under
`jax.profiler.trace` — on a deliberately small portfolio, writing its trace under
`.profile-out-small/` and printing the worker's summary of it: wall time, compiles, events,
and time per phase and per trade's Greeks. It leaves Greeks **on**: they are a large part of
where the engine spends its time. On CPU the job takes 43 s from scratch, 19 s with the
worker's disk cache and 2.8 s repeated
([measured](../concepts/profiling.md#20-the-demo-measured-2026-10-05-roadmap-21)). See [Profiling a pricing job](#profiling-a-pricing-job) below and
[Profiling & the Tracer](../concepts/profiling.md).

**How much precision market risk needs:**
```bash
python demos/demo_precision.py
```
Runs `engine.market_risk.run_market_risk` on a sloped two-curve market and a mixed
portfolio over five Sobol seeds at three `Precision` settings (FP64; FP32 throughout; the
P&L computed in FP64 and stored in FP32), and compares each one's error in VaR 99% and ES
97.5% with the Monte Carlo noise those numbers already carry (the ES standard error and the
spread across seeds).

**One engine module at a time:**
```bash
python demos/demo_components.py                 # every section
python demos/demo_components.py swap var_es     # only these
```
Each section runs one module's public API end to end on the shared example scenarios of
[`demos/demo_scenarios.py`](../../demos/demo_scenarios.py) (the Hull-White model), and
prints what it returns:

| Section | Runs | Prints |
|---|---|---|
| `simulation` | `simulate` on the two-currency demo market | the scenario market's shapes, curves, FX and equities, a path's discount factors, and E[1/N(t)] (today's discount factors) |
| `swap` | `value_portfolio` for a 2Y swap | today's value and the mean NPV per date: coupons drop out as they pay, exactly 0 after maturity |
| `european` | a 3Y-into-2Y European, Bachelier then Jamshidian | the mean NPV per date; after expiry an exercised path carries its swap |
| `bermudan` | a Bermudan, recalibrated on every path and date | today's value and the mean NPV per date |
| `american` | the same for an American | the number of exercise opportunities, then the same |
| `greeks` | a swap's and a European's Greeks, bump and AD | every Greek, per tenor (bump) and per pillar (AD) |
| `var_es` | `compute_risk_metrics` on a swap's NPV cube | the t=0 value and loss quantiles per date. These are statistics of the cube; the engine's market-risk VaR is `demos/demo.py`'s last section |

These sections were once `__main__` blocks inside the engine modules; they moved to
`demos/` so the shipped package holds no demo code
([I-65](../planning/known-issues.md#i-65)). `tests/test_demos.py` runs every section, which
also guards [I-28](../planning/known-issues.md#i-28).

## Running the tests

```bash
.venv/Scripts/python.exe -m pytest tests/ -m "not slow" -q -n 8   # fast tier, in 8 processes
.venv/Scripts/python.exe -m pytest tests/ -q -n 8                  # full suite
.venv/Scripts/python.exe -m pytest tests/test_valuation.py -q      # one area while working on it
```

(`.venv/bin/python` on Linux.) Use the virtualenv's interpreter: the system one lacks the API
and schema dependencies.

See each deep-dive doc's "Tested by" section for what's covered where, and
[Architecture: Testing philosophy](../concepts/architecture.md#testing-philosophy)
for the general approach (every formula is checked both for internal mathematical
correctness and against ORE's own installed software directly).

**What makes it fast.** Most of a test's time is XLA compiling programs, not arithmetic, so
three things matter:

- **The pricers compile once per shape.** They are jitted with the trade as an argument
  ([profiling §3.7](../concepts/profiling.md#37-trade-data-as-traced-arguments-2026-10-02)),
  so a test reuses the programs every earlier test of the same shape compiled.
- **Compiled programs are kept on disk** in `.jax_cache/` (JAX's persistent compilation
  cache, set up by `tests/conftest.py`), so a rerun reads most of them back. The first run
  after a fresh clone, or after a jax upgrade, pays the full compile. Point
  `JAX_COMPILATION_CACHE_DIR` elsewhere to keep it outside the tree; delete the directory
  to reset it, for example to measure cold compiles.
- **`-n N` runs the tests in `N` processes** (`pytest-xdist`). Each takes 1.5–3 GB, so
  choose `N` by memory, not cores: `-n auto` starts one per logical core, which on the
  reference machine (24 threads, 32 GB) ran out of memory and failed tests with
  `MemoryError`. `-n 8` is as fast there. `--dist loadscope` keeps each module on one
  process, so its module-scoped fixtures (a portfolio run, a market-risk run) are computed
  once, about 10% faster again.

Measured on the reference Windows machine on 2026-10-02 (2,254 fast-tier tests):

| Fast tier | Time |
|---|---:|
| Before (one process, every program compiled per trade, date and call) | 28m33s |
| One process, empty `.jax_cache/` | 11m42s |
| `-n 8`, warm `.jax_cache/` | 2m17s |
| `-n 8 --dist loadscope`, warm `.jax_cache/` | 2m05s |

The full suite, `-n 8`, with the fast tier's programs already on disk: 7m45s (2,340 tests).
Before this change it took 46 minutes in one process (2026-09-29), and longer since.

**While changing one area**, run its test files directly (they are named by area:
`test_valuation.py`, `test_ore_lgm_*.py`, `test_greeks*.py`, `test_market_risk*.py`,
`test_integration_*.py`, ...), add `-x` to stop at the first failure, and run the fast tier
before committing. `--lf` reruns only what failed last time.

**Two tiers.** Tests marked `@pytest.mark.slow` are excluded by the **fast tier**,
`-m "not slow"`. A test is marked `slow` when either:

- it starts engine worker processes (the job-submitting classes in `tests/test_api.py`
  and the process tests of `tests/test_engine_worker.py`), or
- it takes 5 seconds or more.

**ORE-parity tests are never marked slow**, however long they take
(`tests/test_ore_*.py`, `tests/test_market_risk_ore_parity.py`). They are what the
pinned versions protect, so they run on every push. The slow tier is mostly portfolio
end-to-end runs, Bermudan Greeks, and compile-count checks. Give a new test the marker
if it meets either rule. `--strict-markers` is on, so a misspelt marker fails collection
and can't silently put a test in the wrong tier.

**CI.** [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) installs through
`requirements.txt` (so with the pinned versions) on Linux, Python 3.11, and runs the fast
tier on every push to `main` and every pull request, with `-n auto` (one process per core of
the 4-core runner). The full suite is the `full` job:
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

`tests/conftest.py` provides shared `pytest` fixtures (the demo scenarios' evaluation date,
a `portfolio_request` fixture — a USD swap on the demo market, simulated by the Hull-White
model — and a `test_client` fixture for `engine.api` tests); `tests/support/` holds the shared
test portfolio and its ORE references, and other helpers shared across test modules.

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
    "market": {"asof": "2026-07-30", "currencies": {"USD": {
      "discount_curve": {"times": [0.0,1.0,2.0,5.0,10.0,30.0], "rates": [0.030,0.030,0.034,0.040,0.046,0.050]},
      "index_curves": {"USD-SIMINDEX-6M": {"times": [0.0,1.0,2.0,5.0,10.0,30.0], "rates": [0.034,0.034,0.038,0.044,0.049,0.052]}}}}},
    "trades": [{"trade_type": "swap", "trade_id": "swap-1", "notional": 1000000.0, "fixed_rate": 0.036,
                "payer": true, "swap_tenor": "2Y"}],
    "simulation": {"dates": ["2027-01-30", "2027-07-30", "2028-07-30"], "base_currency": "USD", "samples": 4096,
                   "ir": {"USD": {"model": "HullWhite", "reversion": 0.03, "volatility": 0.01}}},
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
| Body | the portfolio request (`MarketPortfolioRequestSchema`, Pydantic) | `EodSubmissionSchema`, pointing at a bundle on disk |
| Durability | A durable SQLite job queue; a job survives restarts, and one killed mid-run reads `interrupted` (roadmap 1.8) | Published to a crash-safe store; survives restart ([I-08](../planning/known-issues.md#i-08): a running attempt does not) |
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

## Pricing a portfolio from Python

A run takes today's market, the trades, and the run configuration. Each trade names its id,
its valuation date (the market's), its currency and index; the curves, volatilities and model
come from the market and the configuration, never from the trade. This is a verified-working
example:

```python
import ORE
from engine.market import CurrencyMarket, Market, SwaptionVolSurface, ZeroCurveConfig, index_name
from engine.instruments.swap import SwapConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.portfolio import CamConfig, HullWhiteConfig, PortfolioRequest, RunConfig, price_portfolio

today = ORE.Date(30, 7, 2026)
pillars = [0.0, 1.0, 2.0, 5.0, 10.0, 30.0]
market = Market(today, {"USD": CurrencyMarket(
    discount_curve=ZeroCurveConfig(pillars, [0.030, 0.030, 0.034, 0.040, 0.046, 0.050]),
    index_curves={index_name("USD", 6): ZeroCurveConfig(pillars, [0.034, 0.034, 0.038, 0.044, 0.049, 0.052])},
    swaption_vols=SwaptionVolSurface(("1Y", "5Y"), ("1Y", "5Y"), ((0.0080, 0.0088), (0.0090, 0.0093))),
)})

trades = [
    SwapConfig(trade_id="swap-1", notional=1_000_000.0, fixed_rate=0.036, payer=True, swap_tenor="2Y",
               evaluation_date=today),
    SwaptionConfig(trade_id="european-1", notional=1_000_000.0, fixed_rate=0.042, payer=True, swap_tenor="2Y",
                   forward_start=ORE.Period(3, ORE.Years), evaluation_date=today),
]

simulation = CamConfig(
    dates=tuple(today + ORE.Period(m, ORE.Months) for m in (6, 12, 24, 36, 48)),
    base_currency="USD",
    ir={"USD": HullWhiteConfig(reversion=0.03, volatility=0.01)},   # or LgmConfig(...)
    samples=1024,
)
result = price_portfolio(PortfolioRequest(market=market, trades=trades, config=RunConfig(simulation=simulation),
                                          compute_greeks=True))
print(result.trade_ids)              # ['swap-1', 'european-1']
print(result.base_npv_per_trade)     # today's values, in USD
print(result.npv_cube.shape)         # (1024, 5, 2)  ->  [Scenarios, Dates, Trades]
print(result.exposure.epe)           # the netting set's EPE, today and on each date
print(sorted(result.greeks[1]))      # ['delta:discount:USD', 'delta:index:USD-SIMINDEX-6M', ...]
```

A few things worth knowing:

- **The model is a field.** `CamConfig.ir` names a model per currency: `LgmConfig` (ORE's
  default) or `HullWhiteConfig`, each with a reversion and a volatility, or calibrated to a
  co-terminal basket of the market's swaption volatilities (`calibration_expiries`,
  `calibration_terms`). Several currencies add the FX rates (`fx_volatilities`) and
  correlations; see [Market Simulation](../concepts/market-simulation.md).
- **Dates are the trade.** `swap_tenor="2Y"` books a two-year swap starting at spot on
  `evaluation_date`, resolved once to its `effective_date`/`maturity_date`. A trade booked in
  the past can be given its dates directly, and a coupon that fixed before the evaluation
  date needs its rate in `fixings={fixing_date: rate}`, which ORE requires too.
- **Without a simulation**, `scenario_risk=False` prices today's values and Greeks only.
- **Every `trade_id` must be unique**; the results' per-trade rows follow the request order,
  and `result.trade_ids` names them.

## Choosing engines and the Greeks method

The configuration's other fields choose the engine per product and how Greeks are computed,
for any model:

```python
from engine.portfolio import GreeksConfig, JamshidianEngineConfig, PricingConfig

config = RunConfig(
    simulation=simulation,
    pricing=PricingConfig(european="Jamshidian", jamshidian=JamshidianEngineConfig(0.03, 0.01)),
    greeks=GreeksConfig(method="AD"),
)
```

Europeans default to Bachelier on the market's volatility (ORE's default); Jamshidian prices
them on its own Hull-White model, given with it. Bermudans and Americans use ORE's LGM grid
engine (`PricingConfig.bermudan`/`american`, `LgmSwaptionEngineConfig`), calibrated to the
trade's own basket. Greeks default to ORE's bump and revalue; `"AD"` gives the same keys by
automatic differentiation ([Greeks](../risk/greeks.md)).

## Pricing without the whole pipeline

The pieces `price_portfolio` assembles can be called directly:

```python
from engine.simulation.config import simulate
from engine.valuation.portfolio import value_portfolio, value_today

value_today(trades, market, "USD")                     # today's values, no simulation
scenarios = simulate(market, simulation)               # the scenario market
valuation = value_portfolio(trades, market, scenarios, "USD")
valuation.cube.shape                                   # (1024, 5, 2)
```

## Computing market risk (VaR / ES)

Short-horizon VaR and ES come from revaluing the portfolio at t=0 under shocked curves
([Market Risk](../risk/market-risk.md)):

```python
import numpy as np
from engine.market_risk import MarketRiskRequest, RateRiskFactors, monte_carlo_scenarios, run_market_risk

factors = RateRiskFactors.from_market(market)          # discount:USD, index:USD-SIMINDEX-6M
covariance = np.eye(factors.size) * 0.0008 ** 2 * 10    # 8bp daily, independent pillars
shocks = monte_carlo_scenarios(factors, covariance, horizon_days=10, num_scenarios=4096, seed=1)
risk = run_market_risk(MarketRiskRequest(trades, market, shocks, quantiles=(0.99, 0.975)))
print(risk.risk["VaR_99"], risk.risk["ES_97.5"])
```

The factors are the pillar zero rates of the market's curves, named as the trades read them;
a `pricing=` `PricingConfig` chooses the engines as in a portfolio run.
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

# base_npv: the portfolio's value today (result.base_npv from price_portfolio)
metrics = compute_risk_metrics(result.npv_cube, result.base_npv, percentiles=(0.95, 0.99))

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

The run's `precision` sets the storage and compute format of each adjustable stage: the
simulation, the scenario market and the pricing on paths. The default is float64 everywhere.

```python
from engine.portfolio import Precision, StagePrecision

RunConfig(simulation=..., precision=Precision.throughout("float32"))        # everything float32
RunConfig(simulation=..., precision=Precision(
    pricing=StagePrecision(storage="float32", compute="float64", accumulate="float64")))  # priced in
                                                                                 # float64, cube kept in float32
RunConfig(simulation=..., precision=Precision(
    pricing=StagePrecision("float32", "float32", "float32"),
    by_product={"bermudan_swaption": StagePrecision()},          # but Bermudans in float64
    by_trade={"swap-7": StagePrecision(storage="float32")}))     # and this swap computed in float64
```

The pricing stage can be set per product (`by_product`, keyed by `"swap"`,
`"european_swaption"`, `"bermudan_swaption"`, `"american_swaption"`, `"bond"`) and per trade
(`by_trade`, keyed by `trade_id`); a trade's own entry wins over its product's, which wins over
`pricing`. Each trade's cube column comes out exactly as if it were priced alone at its
precision, and a key that names no product or no trade of the request is refused. Market risk
(`MarketRiskRequest.precision`) takes the same overrides.

Calibration, today's values, Greeks and the exposure statistics are always float64; the
result's `npv_cube` is the stored cube read back at float64. A compute format means the same on
every device: on a GPU or TPU, XLA would by default run a float32 matrix product in
TensorFloat-32 or bfloat16 passes, so each of the engine's matrix products states the
precision of its compute format itself (`engine.precision.matmul`, full precision for float32
and float64). No process setting changes it, JAX's `jax_default_matmul_precision` included,
which the engine neither sets nor reads (roadmap 2.3); your own JAX code in the same process
keeps whatever you set. Over HTTP the same is
`"precision": {"pricing": {"storage": "float32", "compute": "float32", "accumulate": "float32"}}`.
Storage can go below 32 bits (roadmap 1.6): `float16`, `bfloat16`, `float8_e4m3fn` and
`float8_e5m2`, kept with a power-of-two scale per block of 32 paths, and rounded to nearest or,
with `Precision(..., rounding="stochastic")`, stochastically (reproducibly, from
`rounding_seed`). For example `pricing=StagePrecision("float8_e4m3fn")` prices in float64 and
keeps the cube in FP8. Compute below float32 is enabled by roadmap 3.7, and naming it earlier
is refused. How much a low-precision market or cube costs in accuracy is measured, not
assumed: curves stored below 32 bits lose forward rates to cancellation
([I-75](../planning/known-issues.md#i-75)). The 32/64 shape of before roadmap 1.4 (`PrecisionConfig(simulation=32)`) is refused
with a message naming its replacement. See
[Architecture: Adjustable precision](../concepts/architecture.md#adjustable-precision).

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
`engine.api.worker._profiled`, around each job the engine worker runs, and is **opt-in**: it
does nothing unless the environment variable `JAX_RISK_PROFILE_DIR` is set, in which case it
wraps the `price_portfolio` call in `jax.profiler.trace(...)` and writes a trace into
`$JAX_RISK_PROFILE_DIR/pid-<pid>/` (one subdir per process).

The first job of a fresh worker is traced **as it actually runs — XLA lowering and
compilation included**, not just steady-state execution; later jobs of the same shape reuse
the worker's compiled programs. That is on purpose: for this engine the
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

To profile a single job directly, with no server or worker process, set the variable
yourself and call `price_job` on the request body (the JSON the HTTP route takes):

```python
import json, os
os.environ["JAX_RISK_PROFILE_DIR"] = ".profile-out"   # set before the call

from engine.api.worker import price_job
# `body` is the HTTP request as a dict (see demos/demo_api.py)
result_json = price_job(json.dumps(body).encode())
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
— `calibration`, `simulation`, `pricing`, `exposure` (or `base_npv` without scenario risk)
and `greeks`, with one region per trade inside it (`greeks/trade<i>/<type>`) — so you can
attribute time per phase without turning the (very expensive) Python tracer on. See
[Profiling & the Tracer §4](../concepts/profiling.md).

If a small-portfolio trace looks entirely compile-bound, that is the real result — see
step 2. Profile a bigger portfolio (more trades, `samples` 16k+) to see execution take
over; drop `compute_greeks` if you only care about the forward pricing path, which is
roughly a 5x difference in trace size.

**Open the trace in xprof, not from its `.trace.json.gz`.** The profiler records every
event in the run's `.xplane.pb`, which xprof reads. The `.trace.json.gz` beside it keeps only
the ~1M events that start first, so in a viewer that reads that file (Perfetto,
`chrome://tracing`) a large trace — the demo's cold run has 1.5M events — silently loses its
second half. After every traced job the worker writes `<run>.summary.json` beside the run
(wall time, compiles, events, seconds per phase) and emits a `UserWarning` when the JSON file
is partial or the trace spans less than half the job
([Profiling & the Tracer §5](../concepts/profiling.md#5-the-trace-summary-and-its-checks)).

Unset `JAX_RISK_PROFILE_DIR` (or run any other demo/test — none of them set it) to go back
to zero-overhead normal runs; the hook is completely inert when the variable is absent.
