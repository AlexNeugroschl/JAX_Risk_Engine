# User Guide

This page is about *running* the code. For how it works internally, see
[Architecture](../concepts/architecture.md) and the per-stage deep dives
([Market Simulation](../concepts/market-simulation.md), [Instruments](../instruments/swaps.md),
[Risk Statistics](../risk/statistics.md)).

## Prerequisites

- Python 3.11 (the project's `venv/` was built against this version).
- The dependencies listed in [`requirements.txt`](../../requirements.txt):
  `open-source-risk-engine` (the ORE Python bindings — see
  [Architecture: ORE as a dependency](../concepts/architecture.md#ore-as-a-dependency)), `pandas`,
  `jax`, `jaxlib`, `numpy`, `scipy`, `pytest`.

## Setting up

From the repository root, with your Python environment activated:

```bash
pip install -r requirements.txt
```

The examples on this page assume you're running from the repository root, so that
`engine` is importable as a top-level package (it has an `__init__.py`, so
`python -m engine.simulation` and `from engine.simulation import ...`
both work without any extra path setup).

If you're using the project's own `venv/` on Windows, replace `python` in the commands
below with `venv\Scripts\python.exe` (or activate the venv first with
`venv\Scripts\activate`).

## Running the demos

Each pipeline module has a runnable demo in its own `if __name__ == "__main__":` block,
showing that module's public API used end-to-end against a shared example scenario (see
[`engine/scenarios.py`](../../engine/scenarios.py)).

**Market simulation:**
```bash
python -m engine.simulation
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

**Risk statistics** (runs simulation and pricing internally first):
```bash
python -m engine.risk.statistics
```
Prints the portfolio's baseline (t=0) value and the VaR/ES numbers at each requested
confidence level, for every simulated time step.

## Running the tests

```bash
python -m pytest tests/ -v
```

This runs the full suite — see each deep-dive doc's "Tested by" section for what's
covered where, and [Architecture: Testing philosophy](../concepts/architecture.md#testing-philosophy)
for the general approach (every formula is checked both for internal mathematical
correctness and against ORE's own installed software directly). As of this writing, the
suite has 502 tests across `tests/`, all passing.

`tests/conftest.py` provides shared `pytest` fixtures (the example scenario
configurations from `engine/scenarios.py`, wrapped as fixtures) so individual test files
don't each need to build their own copy of the same setup.

## Writing your own market simulation config

`generate_paths()` takes a `SimulationConfig` — see
[API Reference: SimulationConfig](../reference/api-reference.md#simulationconfig) for every field.
Here's a minimal, verified-working example with one equity and one interest rate curve:

```python
from engine.simulation import (
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
from engine.simulation import (
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
from engine.simulation import (
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

## Computing risk metrics

```python
from engine.risk.statistics import compute_risk_metrics

# base_npv: the portfolio's actual value today, from a separate zero-shock
# revaluation -- see engine/scenarios.py's flat_yield_curves() for a worked
# example of building one directly from ORE's own curve objects.
metrics = compute_risk_metrics(npv_cube, base_npv, percentiles=(0.95, 0.99))

print(metrics["VaR_95"])   # [TimeSteps] array
print(metrics["ES_99"])    # [TimeSteps] array
```

See [Risk Statistics: the P&L baseline](../risk/statistics.md#the-pl-baseline-what-are-gainslosses-measured-against)
for exactly what `base_npv` should be and why it can't be inferred automatically from
the NPV cube itself. **`ES_*` values can be `NaN`** for a given time step if there were
no simulated losses severe enough to have anything "worse than the VaR cutoff" — check
for this explicitly rather than assuming a numeric result (see
[Risk Statistics: the formulas](../risk/statistics.md#the-formulas)).

## Precision (float32 vs float64)

`generate_paths(config, precision=64)` (the default) runs in 64-bit precision. Pass
`precision=32` to run in 32-bit instead — see
[Architecture: Adjustable precision](../concepts/architecture.md#adjustable-precision) for what
this changes and why it's a single, per-call argument rather than something set once
globally by the caller.
