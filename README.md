# JAX Risk Engine

An end-of-day risk engine written in [JAX](https://github.com/google/jax). It simulates
future market scenarios or replays historical ones, prices a portfolio in each of them, and
turns the results into the risk figures a bank reports: exposure profiles, sensitivities
(Greeks), Value at Risk and Expected Shortfall.

The precision of every stage is configurable, from FP64 down to FP8, so the same job can be
run at different precisions and the results compared. The whole pipeline compiles through
XLA and runs on CPUs, GPUs and TPUs, splitting a job across several devices or hosts.

The models are ported from [ORE (Open Source Risk Engine)](https://www.opensourcerisk.org/),
the widely used open-source C++ risk library, and every result is tested against ORE itself.

## Why

The project exists to answer two questions:

- **How well does risk calculation run on TPUs?** TPUs are built for large, parallel,
  low-precision arithmetic, which is what a risk run is: the same pricing math repeated over
  thousands of scenarios. ORE has no TPU support, so this engine re-implements its models as
  vectorized JAX programs that do.
- **How much precision does risk actually need?** Lower precision is much faster on
  accelerators. The engine measures whether many cheaper low-precision scenarios can match the
  accuracy of fewer double-precision ones in the same time, and which figures hold up at
  which formats.

## Features

**Simulation**
- ORE's cross-asset model: the LGM or Hull-White model per currency, Black-Scholes FX and
  equity, with correlations between them
- Calibration to swaption volatility baskets
- Sobol quasi-random paths with a Brownian bridge

**Instruments**
- Interest rate swaps
- European swaptions (Bachelier, or Jamshidian on Hull-White), physical or cash settled
- Bermudan and American swaptions, each calibrated to its own basket and priced by backward
  induction on an LGM grid
- US Treasury bills and notes

**Analytics**
- **NPV**: today's value of every trade
- **Exposure**: a `[paths, dates, trades]` NPV cube and the standard profiles (EPE, ENE, EE,
  EEE, PFE, EEPE)
- **Sensitivities**: Delta and Gamma per curve tenor, Vega per volatility quote, and Theta, by
  bump-and-revalue or by automatic differentiation
- **Market risk**: short-horizon VaR and Expected Shortfall by full revaluation under Monte
  Carlo or historical shocks
- **Calibration**: the fitted model parameters and how well they fit

**Precision**
- Storage and compute formats chosen per stage (simulation, scenario market, pricing), per
  product or per trade: FP64, FP32, TF32, BF16, FP16 and FP8
- Low-precision storage with per-block scaling and optional stochastic rounding
- Every result reports the precision it actually ran at, and can carry a float64 reference
  sample to measure the error

**Running it**
- One request for every analytic over an HTTP API, with a background job queue, cancellation
  and chunked result download
- Integration with [TraderX](https://github.com/finos/traderX), the FINOS reference trading
  platform, which submits end-of-day portfolio bundles

## Validation against ORE

The test suite runs ORE in the same process and compares results directly:

- Today's prices for every trade type match ORE's own pricing engines to between `1e-14` and
  `4e-11`.
- On simulated paths, model analytics and every pricer agree with ORE to `1e-10` or better.
- Exposure cubes match ORE's exposure simulation on the same paths.
- Bump-and-revalue sensitivities match ORE's sensitivity analysis; AD Greeks match in the limit
  of a small shift.
- VaR and ES match ORE repricing every shocked scenario.
- Randomized markets, portfolios and settings are checked end to end against ORE.

## Installation

Requires Python 3.11 or later.

```bash
git clone https://github.com/AlexNeugroschl/JAX_Risk_Engine.git
cd JAX_Risk_Engine
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt     # Windows: .venv\Scripts\python.exe
```

`requirements.txt` installs the package with the HTTP API, at the exact versions the ORE
comparisons were verified against.

Optional extras:

| Extra | For |
|---|---|
| `api` | the HTTP server (FastAPI, Uvicorn) |
| `gpu` | GPUs, via JAX's CUDA 13 plugin (Linux, or WSL2 on Windows) |
| `tpu` | TPUs |
| `profiling` | collecting and viewing profiler traces (XProf) |

For example, on a Linux machine with a GPU:

```bash
.venv/bin/python -m pip install -c constraints.txt -e ".[api,gpu]"
.venv/bin/python -c "import jax; print(jax.devices())"   # [CudaDevice(id=0)]
```

## Usage

Start the server:

```bash
.venv/bin/python -m uvicorn engine.api.app:app
```

Interactive API docs are at `http://127.0.0.1:8000/docs`.

A run takes today's market, a portfolio, the analytics to compute and a configuration. This
one prices a two-year swap and simulates its exposure under Hull-White on 4,096 paths:

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
```

The server answers `202 Accepted` with a `run_id` and runs the job in the background. Fetch the
result, waiting up to 30 seconds for it to finish:

```bash
curl "http://127.0.0.1:8000/runs/<run_id>?wait=30"
```

The result has one row per trade (its value today, exposure and Greeks), the portfolio's
exposure profile, and the NPV cube.

The available analytics are `npv`, `exposure`, `sensitivities`, `market_risk` and
`calibration`. Anything left out of the configuration takes ORE's default, and
`GET /config/defaults` returns every section with its defaults. To change the precision, add
a `precision` section to `config`, for example to price on the paths in FP32:

```json
"precision": {"pricing": {"storage": "float32", "compute": "float32", "accumulate": "float32"}}
```

## Project layout

```
engine/
  market_data/        today's market: curves, volatility surfaces, FX spots
  models/             the interest rate, FX and equity models
  calibration/        fitting models to market volatilities
  solvers/            root solvers used by calibration and exercise boundaries
  market_simulation/  path generation and the simulated scenario market
  instruments/        trade definitions (swaps, swaptions, Treasuries)
  pricing/            pricing engines, today and on every path
  risk/               exposure, Greeks, VaR and Expected Shortfall
  precision/          precision policies and low-precision storage
  run/                the run request: the single entry point for every analytic
  api/                the HTTP server and engine worker
  traderx/            TraderX end-of-day bundle support
```

## Documentation

See the [User Guide](docs/getting-started/user-guide.md) for setup on GPUs and TPUs, the full
set of options, and more examples.
