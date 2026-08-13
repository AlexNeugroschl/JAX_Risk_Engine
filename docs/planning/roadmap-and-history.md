# Roadmap & Development History

This page tracks the project's phased build-out and records notable bugs found and fixed
along the way — useful for understanding *why* the code looks the way it does in a few
places, not required reading to use or extend the engine. For current architecture, see
[Architecture](../concepts/architecture.md); for what's implemented right now, see the
[Overview](../getting-started/overview.md).

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| 1. Market simulation | Port ORE's Cross-Asset Model (Sobol QMC, Brownian bridge, Hull-White 1F, GBM) to JAX | ✅ Done |
| 2. Interest rate swaps | Vectorized linear swap pricing, multi-curve discounting | ✅ Done |
| 3. VaR / Expected Shortfall | Risk aggregation over the NPV cube, matching `ORE.RiskStatistics` | ✅ Done |
| 4. End-to-end validation | Full-pipeline parity check against ORE on a mixed portfolio | ✅ Done |
| 5. European swaptions | Jamshidian's decomposition under Hull-White 1F | ✅ Done |
| 6. Bermudan & American swaptions | Numeric LGM backward induction (Hagan convolution), matching ORE's actual production engine | ✅ Done |
| 7. Greeks (Delta, Gamma, Theta) | Curve-pillar sensitivities for swaps and European swaptions via JAX autodiff, matching ORE's bump-and-revalue convention | ✅ Done (swap, European swaption only at the time) |
| 7b. Models/trades consolidation | Extract duplicated Hull-White/LGM math and ORE trade-building into `engine/models/`, `engine/trades/` | ✅ Done |
| 7c. LGM calibration | Bootstrap-fit a piecewise LGM `Sigma` to market swaption vols, matching `ore::data::LgmBuilder::calibrate()` | ✅ Done — see [Calibration](../reference/calibration.md) |
| 7d. Bermudan/American Greeks & Vega | Port the Bermudan backward induction to `jax.lax.scan`; Delta/Gamma/Theta/Vega for Bermudan/American swaptions | ✅ Done — closes the Phase 7 gap, see [American & Bermudan Swaptions](../instruments/american-bermudan-swaptions.md#delta-gamma-theta-and-vega) |
| 8. XVA (CVA/DVA) | Convert NPV cube to exposure, aggregate expected exposure | 🔜 Planned |
| 9. TraderX API integration | FastAPI/gRPC microservice wrapping the pricing pipeline | 🔜 Planned — see [TraderX Integration Plan](traderx-integration.md) |
| 10. Compute-precision research | FP32/BF16 vs FP64 statistical parity study at scale | 🔜 Planned |

## Phase notes and bugs found along the way

### Phase 1 — Market simulation

- Extracted ORE's exact recursive Brownian Bridge matrix
  (`engine/simulation.py::_build_bridge_matrix`) — verified to exactly reproduce Brownian
  motion's covariance structure.
- Calibrated the Hull-White `A(t,T)` term **independently per rate factor** against that
  factor's own curve (`rates.initial_zero_curves`, one per factor). This was initially
  built sharing one curve across every rate factor — live-verified against ORE's installed
  Cross-Asset Model that this was wrong (every `IrLgm1fParametrization` is constructed with
  its own `(Currency, YieldTermStructureHandle)` pair; there's no shared-curve code path
  anywhere in ORE). Fixed; `RatesConfig.initial_zero_curves` is now a list, one curve per
  factor, validated to match the factor count.

### Phase 2 — Interest rate swaps

- Full multi-curve discounting (separate `discount_curve_index`/`forward_curve_index` per
  swap), mirroring ORE's `DiscountingSwapEngine` + `IborIndex.forwardingTermStructure()`
  split.
- Trade schedules and coupon accrual are built with ORE's own `MakeVanillaSwap`/day-count
  classes, not reimplemented — cross-checked directly against `ORE.VanillaSwap.NPV()`.
- **Known limitation:** cashflow dates must land exactly on a configured `maturities`
  pillar (no curve interpolation yet).

### Phase 3 — VaR / Expected Shortfall

- Formula reverse-engineered by live-testing `ORE.RiskStatistics`: lower/nearest-rank-below
  order statistic (NOT numpy's linearly-interpolated percentile), and a strict value-based
  tail filter for ES (NOT a positional slice) — the two diverge whenever the tail has ties
  at the VaR boundary.
- VaR/ES are computed against the portfolio's actual t=0 NPV (a separate zero-shock
  revaluation supplied by the caller), applied at every simulated time step — matching
  ORE's literal historical-VaR P&L definition.
- **Known limitation:** `expected_shortfall` returns `NaN` for any (percentile, time step)
  whose strict loss tail is empty, mirroring `ORE.RiskStatistics.expectedShortfall`'s own
  `RuntimeError`. JAX can't raise from traced code, so callers must check for `NaN`
  explicitly.

### Post-Phase-3 architecture review

`generate_paths` was changed to take a typed `SimulationConfig` dataclass instead of a
loosely-typed dict, and the demo scenario configs / ORE flat-curve-building helper
(previously hand-copied across every module) were centralized into `engine/scenarios.py`
and `tests/conftest.py`.

### Post-Phase-5 thorough-testing pass — two significant bugs found

Building an end-to-end test that priced a mixed swap + swaption portfolio and cross-checked
it against real ORE objects at scale surfaced two bugs in `engine/simulation.py`'s core
Hull-White/GBM step, invisible to every prior test because those all validated pricing
formulas at a *given* simulated rate, never the *distribution* of simulated rates itself:

1. **Mean-reversion drift bug.** The HW1F step computed
   `r_next = r_t*decay + theta_hw + shock_hw` instead of the correct closed-form
   Ornstein-Uhlenbeck transition `r_next = r_t*decay + theta_hw*(1-decay) + shock_hw`. The
   missing `(1-decay)` factor made `theta` act as a flat per-step drift increment instead of
   a mean-reversion target, so every simulated rate factor drifted upward (or downward)
   *without bound* every step — a 3%-mean scenario's simulated mean rate reached ~14.6% by
   t=2y. Invisible in every pre-existing test because they all set `theta == initial_rates`
   (a fixed point only under the *correct* formula). Fixed; see
   `tests/test_simulation.py::TestHullWhiteMeanReversionTransition`.
2. **Double-applied volatility bug.** The per-step correlation matrix was built as the
   Cholesky factor of the *raw* covariance matrix (whose diagonal already encodes each
   factor's own volatility), and the step formulas then multiplied the already-scaled shock
   by that same factor's volatility a *second* time — squaring the effective volatility
   actually simulated (a configured 20% equity vol produced an actual ~4% simulated
   log-return std). Fixed by building the Cholesky factor from the *correlation* matrix
   (unit diagonal) instead. See `tests/test_simulation.py::TestVolatilityIsNotDoubleApplied`.

A related, smaller-scope limitation was found and documented (not fixed, by deliberate
scoping decision) during the same pass: `engine/instruments/swap.py::price_swaps` has no
representation of an already-fixed/elapsed floating coupon, producing a small NPV error
when a swap is priced at any simulated time past its own first accrual date. See
`swap.py`'s "Known limitation" docstring and
`tests/test_swap.py::TestAgedSwapKnownLimitation`.

### End-to-end validation

`tests/test_end_to_end.py` prices a mixed portfolio through this engine's complete
pipeline and, independently, through real ORE objects conditioned on the exact same
simulated short-rate values, isolating pricing/risk correctness from RNG differences.
Per-scenario portfolio NPV matches ORE to better than `1e-3` relative error across the
full simulated distribution, and VaR/ES match to `1e-3` relative tolerance.

### Phase 5 — European swaptions

- Jamshidian's decomposition, matching `ORE.JamshidianSwaptionEngine.NPV()` to 1e-6
  relative precision or better across payer/receiver, ITM/ATM/OTM, and multiple
  tenors/forward-starts.
- The exercise boundary (`r*`) is solved via a vectorized bisection across every
  `[Scenarios, TimeSteps]` entry at once under `jax.jit`.
- **Bug caught during development (forward-starting swaptions):** an early version assumed
  the underlying swap's floating leg always redeems its notional exactly at the option's
  own exercise date `T0` — true only when the spot lag and "exercise lag" coincide, and off
  by ~1% vs. ORE for any genuinely forward-starting swaption. Fixed by adding the swap's
  own `P(T0,T_start)` discount factor as a genuine signed leg of the decomposition. See
  `tests/test_european_swaption.py::TestAgainstOREJamshidianEngine::test_matches_ore_forward_starting`.

### Phase 6 — Bermudan & American swaptions

- **Deviation from the original plan, authorized by direct source verification, not a
  shortcut:** ORE itself does not use Longstaff-Schwartz for Bermudan/American swaptions.
  Reading `OREData/ored/portfolio/builders/swaption.hpp`/`.cpp` and
  `QuantExt/qle/pricingengines/numericlgmmultilegoptionengine.cpp` directly (confirmed via
  a repository-wide search that `TreeSwaptionEngine`, `JamshidianSwaptionEngine`, and
  Longstaff-Schwartz-style regression are never referenced anywhere in `OREData`/`QuantExt`)
  shows ORE's actual production engine is a numeric LGM backward-induction grid
  (`QuantExt::NumericLgmMultiLegOptionEngine`, backed by `LgmConvolutionSolver2`'s
  Hagan-quadrature convolution scheme), with American exercise priced as a
  finely-discretized Bermudan, not continuously. This phase implements that algorithm
  instead. See
  [American & Bermudan Swaptions](../instruments/american-bermudan-swaptions.md) for the
  full algorithm writeup.
- **Finding made during implementation:** `ORE.HullWhite` (used by this codebase's other
  pricers) and `ORE.LinearGaussMarkovModel` (ORE's own Bermudan/American engine's actual
  model), despite sharing `(a, sigma)` and today's curve, are live-verified to NOT be the
  same numerical model realization for `t>0` (~0.6% bond-price divergence at `t=3y`) — a
  genuine parametrization/calibration difference between the two ORE classes, not a bug in
  this codebase's formulas. `bermudan_swaption.py` therefore uses a separate,
  independently ORE-verified closed form (`_lgm_bond`) exclusively.
- **Known limitation:** exercise dates must coincide with the underlying swap's own reset
  dates for exact pricing; a genuinely mid-coupon exercise date forfeits that period's
  already-accrued value rather than prorating it. See
  `tests/test_bermudan_swaption.py::TestMidCouponKnownLimitation`.

### Robustness-testing pass (post-Phase-6)

A dedicated pass adding edge-case and diverse-portfolio tests across every module (172 →
502 tests) surfaced and fixed five further bugs:

1. **`swap.py::_maturity_indices`** — asymmetric tolerance: a cashflow time landing just
   *above* a pillar (e.g. by 9e-7, well inside the documented 1e-6 tolerance) was silently
   rejected because `np.searchsorted`'s `side='left'` compared it against the wrong
   neighbor, while the same magnitude *below* a pillar matched fine. Fixed by comparing
   against both neighboring pillars and picking the closer one before the tolerance check.
2. **`european_swaption.py::_solve_rstar`** — the bisection root-find used a fixed `[-2, 2]`
   short-rate bracket; a sufficiently deep-ITM payer swaption (very negative `fixed_rate`)
   has its true root outside that range, causing silent convergence to the bracket's own
   edge and an NPV wildly wrong in both magnitude and sign vs. ORE. Fixed by expanding the
   bracket outward (geometric doubling) whenever it doesn't already contain a sign change,
   before bisecting.
3. **`simulation.py`** — `mean_reversion=0.0` (a valid HW1F parametrization, the
   arithmetic-Brownian-motion limit) produced all-NaN output from literal `1/a` divisions
   in three places. Fixed by guarding each with the correct analytic `a→0` limit.
4. **`simulation.py`** — any single zero-variance factor poisoned the *entire* correlation
   matrix with NaN via a 0/0 in the covariance-to-correlation normalization, contaminating
   unrelated, well-behaved factors through the shared Cholesky factor. Fixed by
   substituting the identity row/column for any zero-variance factor before Cholesky.
5. **`simulation.py`** — `theta`/`mean_reversion`/`rate_mapping`/`joint_covariance` length
   mismatches against the number of factors were silently absorbed by JAX broadcasting
   instead of raising. Added explicit validation for all of them.

### Reorganization (post robustness-testing pass)

The codebase was reorganized from a flatter, less consistently named layout into the
current structure — `engine/instruments/american_swaption.py` used to contain both the
Bermudan engine and the American wrapper in one file, which made it look like Bermudan
swaptions had never been implemented. Split into `bermudan_swaption.py` (the engine) and a
slimmed `american_swaption.py` (a thin wrapper), alongside a broader consistent-naming
pass (`market_simulations.py` → `simulation.py`, `interest_rate_swap.py` → `swap.py`,
`aggregate_statistics/risk_statistics.py` → `risk/statistics.py`). Pure reorganization —
all 502 tests passed unchanged before and after.

### Greeks (Delta, Gamma, Theta)

`engine/risk/statistics.py` was renamed to `engine/risk/var_es.py` (unchanged content) to
make room for a sibling `engine/risk/greeks.py` module — the "statistics" name no longer
described everything under `engine/risk/`.

Scope (at the time): Delta/Gamma/Theta for `engine.instruments.swap` and
`engine.instruments.european_swaption` only. `bermudan_swaption.py`/`american_swaption.py`
run their backward-induction engine in plain NumPy on the CPU (a grid method, not a
GPU-vectorized computation), with no JAX computational graph to differentiate through at
all — a documented gap, not an oversight, left for a follow-up (either a bump-and-revalue
fallback or porting the backward induction to JAX). Vega was scoped out entirely: ORE's
swaption Vega bumps the market-quoted implied-volatility surface used to CALIBRATE the
Hull-White model, and this codebase's swaption pricers take `hw_sigma` directly as a
config input with no such calibration step — there is no ORE-equivalent quantity to
differentiate, and reporting d(NPV)/d(hw_sigma) under the name "Vega" would misrepresent
what it means.

**Both gaps are now closed** — see
["Bermudan/American Greeks and Vega" below](#bermudanamerican-greeks-and-vega-the-gaps-this-project-had-explicitly-scoped-out-are-now-closed)
for the resolution: the backward induction was ported to `jax.lax.scan` (removing the
first blocker), and `engine/calibration/` was built to supply the missing
market-vol-to-model relationship (removing the second).

Method: JAX automatic differentiation (`jax.grad`/`jax.hessian`), scaled to ORE's own
bump-and-revalue convention (`OREAnalytics::SensitivityAnalysis`'s 1bp absolute zero-rate
bump, per-pillar with a triangular interpolation shape) rather than literal finite-
difference bump-and-revalue — mathematically the same quantity ORE reports, computed
without finite-difference truncation error.

**Bug found and fixed during implementation:** `european_swaption.py`'s `_solve_rstar`
(Jamshidian's bisection root-find for the exercise boundary `r*`) produced the *correct
forward value* but a *silently wrong gradient* — naively differentiating through a
comparison-based bisection loop (`jnp.where(val_mid > 0.0, ...)`) gives zero gradient
everywhere, since the comparison itself has no gradient, regardless of how the true root
actually moves with the function's own parameters (confirmed on a toy case: `d(root)/dc`
came out `0.0` instead of the correct `1.0`). This was invisible until Greeks needed to
differentiate through it — `price_swaptions` itself never needed a gradient of its own
root-find. Fixed via `jax.custom_jvp` implementing the implicit function theorem directly
(`dr*/dparams = -(df/dparams) / (df/dr)`, evaluated at the converged root), which also
required changing `_solve_rstar`'s signature so the curve-dependent quantities it should
be differentiable with respect to are passed as an explicit `params` argument rather than
captured in a closure (`jax.custom_jvp` requires an explicit primal argument). Verified
against literal finite-difference bump-and-revalue to full expected precision for both
Delta and Gamma — see `tests/test_greeks.py`.

### Models & trades consolidation

The same Hull-White closed-form math had drifted into four separate implementations
(`simulation.py`, `european_swaption.py`, `greeks.py`, each written independently as new
capabilities needed it), and the same ORE swap-building/cashflow-extraction code into
three-to-four more (`swap.py`, `european_swaption.py`, `bermudan_swaption.py`). Extracted
both into `engine/models/hull_white.py` and `engine/trades/ore_builders.py` respectively —
one implementation of each, used everywhere. This was a prerequisite, not just tidying: it
made the next two phases (piecewise LGM sigma, and Bermudan Greeks/Vega) tractable to build
in one place rather than four. `engine/models/lgm.py` was added alongside as the LGM
counterpart to `hull_white.py`, generalizing `bermudan_swaption.py`'s own previously
inline LGM formulas into a shared module and adding genuine piecewise-constant sigma
support (`Sigma`, matching `QuantExt::Lgm1fPiecewiseConstantParametrization`) — the
prerequisite calibration itself needed. See
[Models & Trades](../reference/models-and-trades.md) for the full breakdown, including the
live-verified finding (already known from Phase 6, restated here in its permanent home)
that Hull-White and LGM are not the same model for `t>0`.

### LGM calibration: closing the "Vega was scoped out" gap

Built `engine/calibration/` — a bootstrap fit of a piecewise LGM `Sigma` to a co-terminal
basket of market swaption volatilities, matching `ore::data::LgmBuilder::calibrate()`'s own
`Bootstrap` path (`calibrateVolatilitiesIterative`, called per-instrument in increasing
expiry order — the `aTimes = swaptionExpiries[:-1]` convention makes this exact, not an
approximation of a joint fit). See [Calibration](../reference/calibration.md) for the full
algorithm.

Two real bugs surfaced during this phase, both eventually caught by the *next* phase's own
finite-difference cross-check (Bermudan Vega), not by this phase's own tests in isolation
— which is itself notable: `price_lgm_swaption`'s forward value was correct the whole time,
so nothing calling this module for a price alone had any way to notice its gradient was
wrong.

1. **`_bisect_xstar`'s gradient** — the LGM analogue of `_solve_rstar`'s already-known bug
   class (see above): its exercise-boundary bisection understated `price_lgm_swaption`'s
   own gradient with respect to sigma by ~6%, for the identical reason (`jnp.where(val >
   0.0, ...)` has zero gradient, silently dropping the indirect `d(price)/d(x*) *
   d(x*)/d(sigma)` term). Fixed with the same `jax.custom_jvp`/implicit-function-theorem
   pattern.
2. **`engine.models.lgm.Sigma` needed JAX pytree registration** — an unregistered
   `@dataclass` is an opaque leaf to `jax.tree_util`, silently blocking any tangent from
   reaching `Sigma.values` whenever `Sigma` was nested inside a larger differentiated
   tuple (as it is inside `_bisect_xstar`'s own `params = (a, sigma)`). Fixed via
   `@jax.tree_util.register_pytree_node_class`. Both fixes were required together — either
   alone still produced a wrong gradient.

`price_lgm_swaption` itself — needed because `QuantExt::AnalyticLgmSwaptionEngine`'s
constructor turned out not to be exposed through this codebase's installed ORE Python
bindings (confirmed by reading the SWIG interface file directly) — was verified via two
independent routes: every formula piece against live `ORE.LinearGaussMarkovModel` methods,
and the full price against a numeraire-deflated Monte Carlo simulation. That Monte Carlo
check itself needed a fix along the way: an initial version discounted sampled payoffs
naively by `P(0,T0)` and showed a spurious ~11% "error" that was actually a bug in the MC
test's own discounting convention (LGM's own measure is not the T0-forward measure) — once
corrected to properly deflate by the model's own numeraire, it matched to ~0.05%, well
within Monte Carlo standard error. See [Calibration](../reference/calibration.md) for the
full account of both routes.

### Bermudan/American Greeks and Vega: the gaps this project had explicitly scoped out are now closed

The "Greeks (Delta, Gamma, Theta)" phase above explicitly scoped out two things:
Bermudan/American Greeks (no JAX computational graph existed for the NumPy backward
induction to differentiate through) and Vega entirely (no market-vol-to-model calibration
step existed for any pricer). Both prerequisites now exist — `bermudan_swaption.py`'s
backward induction was ported to `jax.lax.scan` as part of the models/trades consolidation
above, and `engine/calibration/` supplies the calibration step — so both gaps were closed
in this phase, not left open indefinitely as originally implied by "documented, not an
oversight, left for a follow-up."

`greeks.bermudan_delta_gamma`/`bermudan_theta` follow the identical pattern and units as
their swap/European-swaption counterparts, with no separate JAX reimplementation of the
pricing formula needed (`_bermudan_price_fn` substitutes differentiable JAX values into the
prepared trade via `dataclasses.replace` and calls the existing backward induction
directly).

Two more bugs were found while building this, on top of the two `calibrate_lgm_sigma`
already surfaced:

1. **`_state_grid`/`std_step`'s `sqrt` gradient** — both compute `sqrt(zeta(...))`, and
   `sqrt`'s derivative is a `0/0` indeterminate form exactly at `zeta(0)=0` (which
   genuinely occurs at `t=0`). The forward value was always correct; `jax.grad`/
   `jax.hessian` produced `NaN`. Fixed via the same branch-free `jnp.where` guard pattern
   already used in `engine/models/hull_white.py`'s own `a==0` singularity guards, applied
   to a different singularity.
2. **`bermudan_vega`'s own missing cross-bucket Jacobian term** — the first version assumed
   bumping basket instrument `i`'s market vol only moves calibrated bucket `i`'s own sigma
   value (`d(s_j)/d(v_i) = 0` for `j != i`). Correct for `j < i` (the bootstrap is
   triangular forward in time), **wrong** for `j > i` (later buckets' own calibration
   equations depend on every earlier bucket via accumulated `zeta`, so a change to an
   earlier bucket cascades forward). Finite-difference cross-checking (bump one market vol,
   literally rerun `calibrate_lgm_sigma`, reprice) caught a 45-78% error in every bucket
   except the last (which, having no later bucket to affect, was the one case where the
   wrong diagonal-only assumption happened to be correct) — that pattern is what pinpointed
   the missing term. Fixed by building the full lower-triangular Jacobian `d(s_j)/d(v_i)`
   via forward substitution over `j`. After the fix, all buckets match finite-difference
   recalibration to within ~0.005%.

See [Delta, Gamma, and Theta](../risk/greeks.md) for the full technical account of both
fixes and every function's exact behavior, [Calibration](../reference/calibration.md) for
the two bugs found in the prerequisite calibration phase, and
[American & Bermudan Swaptions](../instruments/american-bermudan-swaptions.md#delta-gamma-theta-and-vega)
for how the new Greeks fit into that module. Full regression suite: 615/615 tests passing,
including the 44+16+9+3+11 new tests these four phases added
(`tests/test_models_piecewise_sigma.py`, `tests/test_calibration_basket.py`,
`tests/test_calibration_lgm.py`, `tests/test_calibration_integration.py`,
`tests/test_greeks_bermudan.py`).

### Market simulation moved into its own package

`engine/simulation.py` and `engine/scenarios.py` were the last two loose files sitting
directly under `engine/` — every other pricing/risk concern (`instruments/`, `risk/`,
`models/`, `trades/`, `calibration/`) already lived in its own subpackage, which made the
market-simulation code look like an afterthought rather than the module every other stage
in the pipeline reads its input from. Moved into `engine/simulation/`, renamed to describe
what each file actually does rather than just repeating the package name:
`engine/simulation.py` → `engine/simulation/market_model.py` (the Sobol/Brownian-bridge/
cross-asset Hull-White path generator and yield-curve reconstruction — the module doing the
actual modeling work), `engine/scenarios.py` → `engine/simulation/demo_scenarios.py` (the
canonical demo/reference `SimulationConfig` builders, which are reference data *about* the
model, not the model itself — a distinction the shared `scenarios.py` name obscured).
`tests/test_simulation.py`/`tests/test_scenarios.py` renamed to match
(`tests/test_market_model.py`/`tests/test_demo_scenarios.py`), continuing the same
test-file-mirrors-source-file-name convention every other module in this codebase already
follows. Every import across `engine/instruments/`, `engine/risk/`, and every test file
was updated to the new `engine.simulation.market_model`/`engine.simulation.demo_scenarios`
paths; `engine/simulation/__init__.py` is empty, matching every other subpackage's own
`__init__.py`. Pure reorganization, following the exact precedent set by the "Reorganization
(post robustness-testing pass)" entry above — no pricing logic, formula, or test assertion
changed. Full regression suite: 659/659 tests passing unchanged before and after.
