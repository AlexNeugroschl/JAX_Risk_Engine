# Market Simulation

**Modules:** [`engine/simulation/`](../../engine/simulation/) — `config.py` (the configuration and
`simulate`), `cam.py` (ORE's cross-asset model), `scenario_market.py` (the simulated market),
`random.py` (Sobol normals and the Brownian bridge).
**Public entry point:** `simulate(market: Market, config: CamConfig, model=None, precision=Precision()) -> ScenarioMarket`

## Plain-language summary

This module answers the question: *"Generate thousands of plausible alternate futures for
interest rates, stock prices, and currency exchange rates, at several points in time."*

Think of it like a weather simulator, but for markets. You tell it: here's where interest
rates and prices stand today (the `Market`), here's roughly how volatile each of them tends
to be, and here's how they tend to move together (the `CamConfig`). The simulator then
generates thousands of independent "alternate timelines," each one a full path from today out
to some future date, step by step.

It does this for:
- **Interest rates**, one model per currency, fitted exactly to that currency's curve today:
  ORE's Linear Gauss-Markov (LGM) model by default, or the Hull-White model.
- **FX rates and equities**, whose drift is tied to the simulated interest rates — this is
  what makes it a *cross-asset* simulation rather than several unrelated simulations bolted
  together.

The output isn't just "the interest rate at each future date" — on every simulated date of
every path it is a full set of **curves** (each currency's discounting curve and each index's
forwarding curve), what ORE calls the scenario market. A curve answers "what is $1 promised at
some future date T worth, if I'm standing at future date t?" That is what lets every trade be
priced on every path (see [Interest Rate Swaps](../instruments/swaps.md)).

## Why it's built this way: matching ORE's Cross-Asset Model

ORE's own simulation engine is called the **Cross-Asset Model (CAM)**. This module
reimplements CAM's math in JAX (`QuantExt::CrossAssetModel`, its exact discretization
`CrossAssetStateProcess::ExactDiscretization`, and `CrossAssetModelScenarioGenerator`),
verified against the installed ORE software: the model's analytics, a step of the state
process, the scenario curves and the numeraire are compared with ORE's objects in the test
suite (`tests/test_cam.py`).

## The configuration

`CamConfig` (`engine/simulation/config.py`, ORE's `simulation.xml`) holds the date grid, the
model of each currency (`ir`), the FX and equity volatilities, the correlations, the
simulation-market tenors, the number of paths (`samples`) and the seed:

```python
from engine.simulation.config import CamConfig, HullWhiteConfig, LgmConfig

CamConfig(
    dates=(...),                          # simulation dates, after the market's as-of date
    base_currency="USD",
    ir={"USD": HullWhiteConfig(0.03, 0.01),                       # fixed volatility
        "EUR": LgmConfig(0.02, 0.008, ("1Y", "2Y", "5Y"), ("9Y", "8Y", "5Y"))},  # calibrated
    fx_volatilities={"EUR": 0.10},
    correlations={("IR:USD", "IR:EUR"): 0.6, ("IR:USD", "FX:EURUSD"): 0.2},
    samples=4096, seed=42,
)
```

**The model per currency.** Both models are ORE's `<LGM>`: a one-factor Gaussian model with
constant mean reversion `a`, fitted exactly to the currency's discount curve, simulated under
the domestic LGM measure. They differ in how the volatility is parametrized
(`LgmData::VolatilityType`), so the same volatility number means a different model:

- `LgmConfig` (`Hagan`, the default): `volatility` is the LGM's own α; ζ(t) = ∫α².
- `HullWhiteConfig` (`HullWhite`, ORE's `IrLgm1fPiecewiseConstantHullWhiteAdaptor`):
  `volatility` is the short rate's σ, and α(t) = σ(t)e^{at}, ζ(t) = ∫σ²e^{2as}ds. This *is* the
  Hull-White model with its curve-fitted drift, written in the LGM's state: the short rate is
  r(t) = f(0,t) + H′(t)z + ζ(t)H(t)H′(t) with H(t) = (1 − e^{−at})/a, and every scenario curve
  equals QuantLib's `HullWhite::discountBond(t, T, r)` (`tests/test_cam.py::
  test_hull_white_path_curves_equal_quantlibs_hull_white`).

Either is given a fixed volatility, or bootstrapped to a co-terminal basket of the market's
swaption volatilities (`calibration_expiries` × `calibration_terms`, ORE's
`CalibrationSwaptions`; see [Calibration](../reference/calibration.md)).

## The pipeline, step by step

`simulate()` runs four phases.

### Phase 1 — Quasi-Monte Carlo shock generation

**Module:** `engine/simulation/random.py` — `generate_sobol_normals()`,
`_build_bridge_matrix()` / `_apply_bridge_matrix()` / `apply_brownian_bridge()`
Simulating "thousands of alternate futures" requires thousands of sets of random numbers
— one set per scenario, one number per (time step × thing-being-simulated). This module
does **not** use ordinary random numbers. It uses a **Sobol sequence**: a specially
constructed sequence of points that fills the space of possibilities much more evenly
than ordinary randomness does, so fewer scenarios are needed to get a stable answer. This
is a standard technique in quantitative finance called Quasi-Monte Carlo (QMC).

```python
def generate_sobol_normals(num_scenarios: int, num_steps: int, num_assets: int, dtype) -> jax.Array:
```
Generates the raw Sobol sequence (via `scipy.stats.qmc.Sobol`) and converts it from
"evenly spread points between 0 and 1" into "evenly spread points that also follow a
bell-curve (Normal) distribution" — the shape random market shocks are assumed to follow.
Returns an array shaped `[TimeSteps, Scenarios, Assets]`.

**dtype caveat, and the fix applied:** `jax.scipy.stats.norm.ppf` (the function that does
the bell-curve conversion) always computes internally in 64-bit precision whenever JAX's
64-bit mode is globally turned on, *regardless* of what precision was requested for this
specific call. This function now explicitly converts its result back to the requested
`dtype` before returning, so calling it directly with `dtype=float32` reliably returns
32-bit numbers. (See [Adjustable Precision](architecture.md#adjustable-precision) for
why this global-setting behavior exists in the first place, and
`tests/test_random.py::TestGenerateSobolNormals` for the regression test.)

**The Brownian Bridge.** Sobol sequences are most accurate in their *first* few
dimensions and progressively noisier in later ones. A naive mapping (dimension 1 → time
step 1, dimension 2 → time step 2, ...) would waste that accuracy on early time steps and
under-serve later ones. The **Brownian bridge** construction reorders things instead: the
*final* time step gets the most accurate dimension, then the midpoint, then the
quarter-points, recursively bisecting — because that ordering captures the overall shape
of a random path with the fewest samples.

```python
def _build_bridge_matrix(time_grid: np.ndarray) -> np.ndarray:
```
Builds this reordering as an explicit matrix `B`, following the same recursive
bisection algorithm QuantLib/ORE's own `BrownianBridge` class uses (see the function's
docstring for the exact right/left/midpoint bookkeeping). It runs on the CPU with plain
NumPy, since it depends only on the time grid, not on any simulated data — it's the same
matrix for every scenario, so it's cheap to compute once.

*Verified:* `B @ B.T` (the matrix multiplied by its own transpose) is checked to exactly
equal the true covariance structure of Brownian motion, `Cov(W(s), W(t)) = min(s, t)`
— this is a strong, closed-form correctness check on the whole construction, and it's
enforced by `tests/test_random.py::TestBrownianBridge::test_matrix_reproduces_bm_covariance`.

```python
def apply_brownian_bridge(Z: jax.Array, time_grid: jax.Array) -> jax.Array:
```
Applies that matrix to the raw Sobol-derived shocks (via a small `@jax.jit`-compiled
helper, `_apply_bridge_matrix`, since this multiplication *is* data-dependent and worth
running on the accelerator), then converts the result from "the bridged path's absolute value at
each time" back into "the standardized shock *between* each consecutive pair of time
steps" — which is the form the state recursion (Phase 2) needs.

### Phase 2 — The cross-asset model's states

**Module:** `engine/simulation/cam.py` — `step_moments()`, `evolve_states()`

The model's state, in ORE's order: one LGM state z per currency (the domestic first), the log
FX rate of each foreign currency, then the log equity spots. Over each step the state is
Gaussian given its start, with a mean and covariance that depend only on the time grid and
the parameters (ORE's exact discretization: `ir/fx/eq_expectation_1/2` and `covarianceImpl`).
So they are computed once on the host in float64, and the path recursion
`x_{i+1} = M_i x_i + b_i + L_i Z_i` runs on the device, vectorized over paths. `L_i` is the
Cholesky factor of the step covariance, as ORE's `pseudoSqrt` computes it
(`CholeskyDecomposition(cov, flexible = true)`, reproduced by `flexible_cholesky`). Every
formula reads the IR components through their α and ζ, so the two parametrizations share it.

### Phase 3 — The scenario market

**Module:** `engine/simulation/scenario_market.py` — `build_scenario_market()`

On each date of each path, ORE's `CrossAssetModelScenarioGenerator`:

- each currency's discount curve, model-implied
  (`P(t, t+τ | z) = P(0, t+τ)/P(0, t) · exp(−(H(t+τ) − H(t)) z − ½ (H(t+τ)² − H(t)²) ζ(t))`),
  sampled at the simulation-market tenors (`CamConfig.curve_tenors`), every discount factor
  floored at 1e-5;
- each index's forwarding curve, the same with the index's own t=0 curve, so the basis
  between the index and the discount curve is deterministic;
- the domestic LGM numeraire, `N(t, z) = exp(H z + ½ H² ζ) / P(0, t)`, exact (exposures are
  NPV/N);
- FX and equity spots.

A scenario curve is held at its tenors only and read log-linearly in between (ORE's
`ScenarioSimMarket`): a trade's flow between two tenors is discounted on the interpolated
curve, not on the model's exact bond price. Configure denser tenors where that matters.

### Phase 4 — Public API

**Function:** `simulate(market, config, model=None, precision=Precision()) -> ScenarioMarket`

Builds (and calibrates) the cross-asset model from the market and the configuration
(`build_cross_asset_model`), draws the Sobol normals with the Brownian bridge, evolves the
states, and builds the scenario market. Pass a calibrated `CrossAssetModel` as `model` to
reuse it. `precision` is the run's `engine.precision.Precision`: the shocks and states are
computed at `precision.simulation.compute` and stored at its `storage`; the market is built
at `precision.market.compute` and every array of the returned `ScenarioMarket` is in
`precision.market.storage` (float64 by default). The step moments and the path-independent
parts of the curves are computed in float64 and cast.

## Output shapes at a glance

| Field | Shape | |
|---|---|---|
| `numeraire` | `[Scenarios, Dates]` | the domestic LGM numeraire |
| `discount[ccy].log_discounts` | `[Scenarios, Dates, Tenors + 1]` | at `tenor_times [Dates, Tenors + 1]` from each date |
| `index[name].log_discounts` | `[Scenarios, Dates, Tenors + 1]` | the forwarding curves |
| `fx[ccy]`, `equity[name]` | `[Scenarios, Dates]` | spots |
| `states` | `[Scenarios, Dates, Factors]` | the CAM states, for pricers that condition on them |

## Tested by

- `tests/test_cam.py` — the step moments make every deflated asset a martingale exactly
  (both parametrizations), and on simulated paths on sloped curves in float64 and float32;
  the path curves and numeraire against `ORE.LinearGaussMarkovModel`; a Hull-White currency's
  curves against QuantLib's `HullWhite.discountBond`; one step against
  `ORE.IrLgm1fStateProcess`; the square root against QuantLib's `CholeskyDecomposition`;
  configuration refusals.
- `tests/test_random.py` — the Sobol normals and the Brownian bridge (its matrix reproduces
  Brownian motion's covariance; agreement with QuantLib's `BrownianBridge`; grids and edge
  cases).
- `tests/test_hull_white_model.py` — the Hull-White model's calibration, and deflated zero
  bonds and swaps as martingales on a curve rising from 3% to 5% (the defect of the model's
  pre-1.3 simulation, I-42), with the exact numeraire.
- `tests/test_valuation.py` — every pricer on path curves against ORE, under both models.
