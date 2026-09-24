# JAX Risk Engine

An end-of-day market risk engine written in [JAX](https://github.com/google/jax). It
simulates market scenarios, prices a portfolio across them, and computes sensitivities,
Value at Risk and Expected Shortfall. The models are ported from
[ORE (Open Source Risk Engine)](https://www.opensourcerisk.org/) and validated against it.

The engine is designed to run across multiple TPUs, with a Google Cloud TPU VM as the
target deployment, and also runs on CPU and GPU. A main research goal is to measure how
much numeric precision Monte Carlo risk needs: whether many lower-precision simulations,
run concurrently across TPU devices, can match the VaR and Expected Shortfall of fewer
double-precision simulations in the same wall-clock time.

## Features

- Cross-asset Monte Carlo simulation: Sobol sequences with a Brownian bridge, Hull-White
  one-factor rates, lognormal equities and FX
- Pricing for interest rate swaps, European swaptions (Jamshidian decomposition), Bermudan
  and American swaptions (numeric LGM, as in ORE's production engine), and US Treasury
  bills and notes
- LGM volatility calibration to market swaption quotes, following ORE's `LgmBuilder`
- Delta, Gamma and Theta via automatic differentiation, scaled to ORE's bump-and-revalue
  convention, and Vega for Bermudan and American swaptions
- Short-horizon VaR and Expected Shortfall by full revaluation of the portfolio under
  Monte Carlo or historical shocks of every curve pillar, with ORE's `RiskStatistics`
  conventions and Monte Carlo error estimates
- Exposure profiles (EPE, ENE, PFE through time) from the multi-step simulation, using ORE's
  `ExposureCalculator` definitions
- Independent FP64/FP32 precision settings for simulation, pricing, risk and calibration,
  with each precision tier running in its own worker processes
- HTTP API for portfolio pricing and calibration, plus a versioned end-of-day contract for
  hash-verified portfolio bundles

## ORE and hardware acceleration

ORE is written in C++ and runs on CPU, with multi-threaded valuation and adjoint
algorithmic differentiation. It also has a compute-framework interface that offloads its
scripted-trade and AMC workloads to GPUs through OpenCL or CUDA, in single precision by
default. It has no TPU support.

This project re-implements the relevant ORE models as vectorized JAX programs, so the full
pipeline (simulation, pricing, calibration, sensitivities and risk) compiles through XLA
and can run on TPUs.

## Validation

Pricing and risk formulas are mapped to their counterparts in ORE's C++ source and tested
against ORE running in the same process:

- A mixed portfolio priced end to end agrees with ORE, on the same simulated rates, to
  within `1e-3` relative error per scenario.
- Market-risk VaR and ES agree with ORE repricing every shocked scenario and running its
  own `RiskStatistics`: swaps and bonds per scenario to about `1e-14`, Bermudans to `2e-13`,
  European swaptions to `3e-7`.
- Bermudan and American swaption prices agree with ORE's LGM grid engine to about `1e-12`.

See [ORE Parity](docs/reference/ore-parity.md) for the full mapping.

The engine is also integrated with [TraderX](https://github.com/finos/traderX), the FINOS
reference trading platform, which sends it end-of-day portfolio bundles. This checks the
engine against another system's data and conventions, and the TraderX team verifies the
results independently. Treasury bills and notes are currently priced on this path; other
instruments are added as their market inputs and conventions are agreed. See
[EOD Integration](docs/reference/eod-integration.md).

## Getting started

Requires Python 3.11 or later.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Linux/macOS: .venv/bin/python
.venv/Scripts/python.exe -m pytest tests/ -q
```

Use the virtualenv's interpreter to run the tests; the API and schema tests need
`pydantic` and `jsonschema`.

The [User Guide](docs/getting-started/user-guide.md) walks through setup and pricing a
portfolio from Python or over HTTP.

## Documentation

- [Overview](docs/getting-started/overview.md)
- [Architecture](docs/concepts/architecture.md)
- [HTTP API](docs/reference/http-api.md)
- [EOD Integration](docs/reference/eod-integration.md)
- [Precision Research](docs/planning/precision%20research/README.md)
- [Full documentation index](docs/README.md)
