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
| 7. XVA (CVA/DVA) | Convert NPV cube to exposure, aggregate expected exposure | 🔜 Planned |
| 8. TraderX API integration | FastAPI/gRPC microservice wrapping the pricing pipeline | 🔜 Planned — see [TraderX Integration Plan](traderx-integration.md) |
| 9. Compute-precision research | FP32/BF16 vs FP64 statistical parity study at scale | 🔜 Planned |

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
