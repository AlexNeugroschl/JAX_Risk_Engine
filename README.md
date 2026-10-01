# JAX Risk Engine

An end-of-day market risk engine written in [JAX](https://github.com/google/jax). It
simulates market scenarios, prices a portfolio across them, and computes sensitivities,
Value at Risk and Expected Shortfall. The models are ported from
[ORE (Open Source Risk Engine)](https://www.opensourcerisk.org/) and validated against it.

The engine is designed to run across multiple TPUs, with a Google Cloud TPU VM as the
target deployment, and also runs on CPU and GPU. Today each job runs on one device;
sharding the scenario axis across devices is planned ([I-61](docs/planning/known-issues.md#i-61)). A main research goal is to measure how
much numeric precision Monte Carlo risk needs: whether many lower-precision simulations,
run concurrently across TPU devices, can match the VaR and Expected Shortfall of fewer
double-precision simulations in the same wall-clock time.

## Features

- ORE's exposure pipeline, reproduced component by component: the cross-asset model (per
  currency ORE's LGM, or the Hull-White model in ORE's LGM form; Black-Scholes FX and
  equity; exact discretization; calibrated to swaption baskets), a scenario market of
  model-implied curves, and every trade repriced on every path with its own t=0 engine,
  including ORE's fixing, cash-flow and exercise rules. Sobol sequences with a Brownian
  bridge.
- Pricing, as ORE's default engines price: interest rate swaps (`DiscountingSwapEngine`),
  European swaptions (Bachelier on market swaption volatilities, physical or cash settled),
  Bermudan and American swaptions (each on its own LGM, calibrated to ORE's basket for the
  trade and priced by ORE's numeric LGM engine), and US Treasury bills and notes
- Exposure profiles with ORE's `ExposureCalculator` definitions: EPE, ENE, EE_B, EEE_B, PFE,
  time-weighted EPE_B/EEPE_B and the Basel one-year figures
- Sensitivities as ORE's sensitivity analysis defines them: bump-and-revalue Delta and Gamma
  per curve tenor, Vega per swaption quote, and Theta on the rolled market; or the same
  Greeks by automatic differentiation, a Bermudan's Vega through its calibration
- Short-horizon VaR and Expected Shortfall by full revaluation of the portfolio under
  Monte Carlo or historical shocks of every curve pillar, with ORE's `RiskStatistics`
  conventions and Monte Carlo error estimates
- One run configuration (`RunConfig`), as ORE configures a run with its files and with
  ORE's defaults: the simulation and its model per currency (LGM or Hull-White), the engine
  per product (Bachelier or Jamshidian Europeans, ORE's LGM grid for Bermudans/Americans),
  the Greeks method (bump or AD) and ORE's sensitivity settings, and the precision per stage.
  Every option runs with every other; an option not implemented yet is refused by name
  rather than substituted ([decisions](compliance/decisions.md))
- Trades as ORE books them: each names its id, its valuation date, its currency and index,
  and carries no model or curve of its own
- Adjustable precision, since which precision each calculation needs is what the project
  studies: FP64 or FP32 for the simulation today; the scenario market and path pricing,
  per product and per trade, and storage down to FP8 follow in roadmap 1.4 to 1.7
  ([I-55](docs/planning/known-issues.md#i-55), [design](docs/planning/details/precision.md))
- HTTP API: one portfolio request reaching the run configuration (market risk and the
  cross-asset calibration routes are still to come,
  [I-56](docs/planning/known-issues.md#i-56)); plus a versioned end-of-day contract for
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

- Every trade type at t=0 agrees with ORE's own engine: swaps, Europeans and bonds to about
  `1e-14`, calibrated Bermudans and Americans to `4e-11` (a shared test portfolio on a
  sloped two-curve market).
- On simulated paths each component agrees with ORE: the cross-asset model's analytics to
  `1e-12` (the Hull-White model's bonds against QuantLib's `HullWhite` too), and every
  pricer on a path's curves against the matching ORE engine to between `1e-12` and `1e-8`
  (a Bermudan recalibrated on the path), under either interest-rate model.
- Market-risk VaR and ES agree with ORE repricing every shocked scenario and running its
  own `RiskStatistics`: swaps, bonds and European swaptions per scenario to about `1e-14`,
  Bermudans to `2e-13`.
- Not yet shown: that the assembled simulation, exposure and sensitivities equal an ORE run
  end to end ([I-50](docs/planning/known-issues.md#i-50), [I-51](docs/planning/known-issues.md#i-51)).

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
.venv/Scripts/python.exe -m pytest tests/ -m "not slow" -q   # fast tier, as CI runs it
.venv/Scripts/python.exe -m pytest tests/ -q                  # full suite, about 45–50 minutes (I-53)
```

Use the virtualenv's interpreter to run the tests; the API and schema tests need
`pydantic` and `jsonschema`. `requirements.txt` installs the exact versions in
`constraints.txt`, which the ORE-parity tolerances were verified against. CI runs the fast
tier on every push and pull request; see
[Running the tests](docs/getting-started/user-guide.md#running-the-tests) for the tiers,
the pins, and how to upgrade one.

The [User Guide](docs/getting-started/user-guide.md) walks through setup and pricing a
portfolio from Python or over HTTP.

## Documentation

- [Overview](docs/getting-started/overview.md)
- [Architecture](docs/concepts/architecture.md)
- [HTTP API](docs/reference/http-api.md)
- [EOD Integration](docs/reference/eod-integration.md)
- [Planning](docs/planning/README.md): [known issues](docs/planning/known-issues.md), [features](docs/planning/features.md), [roadmap](docs/planning/roadmap.md), and [decisions](compliance/decisions.md)
- [Precision design](docs/planning/details/precision.md)
- [Full documentation index](docs/README.md)
