# ORE Parity: Algorithm-by-Algorithm Correspondence

This page maps every mathematical algorithm implemented in this codebase to its exact
counterpart in ORE's own C++ source, now available locally under
[`reference/ORE`](../../reference/ORE) (a full clone of the
[OpenSourceRisk/Engine](https://github.com/OpenSourceRisk/Engine) repository, including
its `QuantLib` and `QuantExt` submodules). Everywhere a formula or algorithm is described
below, it was verified by directly reading the cited C++ file — not assumed from
documentation, textbooks, or memory of earlier live-testing sessions (which is how every
prior formula claim in this project was verified, before this C++ source was available
locally; see [Architecture: ORE as a dependency](../concepts/architecture.md#ore-as-a-dependency)).
Where earlier live-testing (calling the installed `ORE` Python package directly and
comparing numbers) had already established a result, this page cross-checks that result
against the actual C++ that produces it, closing the gap between "the numbers match" and
"the numbers match *because* the formulas are the same."

**Convention used below:** ORE/QuantLib class and method names are given as
`ClassName::methodName`, with the file path relative to `reference/ORE/`. `reference/ORE`
itself is never modified by this project — it exists purely as a read-only reference.

## Where ORE's algorithms actually live

ORE (`reference/ORE/`) is built in three layers, and this matters for where to look for a
given algorithm:

1. **QuantLib** (`reference/ORE/QuantLib/`) — the base open-source quant library ORE is
   built on. Plain single-currency models (`HullWhite`), plain pricing engines
   (`DiscountingSwapEngine`, `JamshidianSwaptionEngine`), the Sobol/Brownian-bridge Monte
   Carlo machinery, and the empirical risk-statistics tools (`GeneralStatistics`,
   `RiskStatistics`) all live here. This is a **git submodule** of the ORE repository —
   it does not come checked out by default (`git submodule update --init QuantLib` is
   required, which is how it was populated for this comparison).
2. **QuantExt** (`reference/ORE/QuantExt/`) — ORE's own extension layer on top of
   QuantLib, adding the multi-currency **Cross-Asset Model** (`CrossAssetModel`,
   `IrLgm1fParametrization`, `LinearGaussMarkovModel`) our simulation engine's rates leg
   is actually modeled on (as opposed to QuantLib's simpler, single-currency `HullWhite`
   class, which was the class most of this project's earlier live-testing sessions used
   as a validation stand-in — see [below](#a-parametrization-note-lgm-vs-plain-hull-white)
   for why the two are equivalent).
3. **OREData / OREAnalytics** (`reference/ORE/OREData/`, `reference/ORE/OREAnalytics/`) —
   ORE's own trade-configuration and analytics layer (XML parsing, scenario generation
   orchestration). Not a source of core math this project's own pricing formulas need to
   match; not covered on this page.

---

## 1. Sobol sequence generation

**This engine:** `engine/simulation.py::generate_sobol_normals`, via
`scipy.stats.qmc.Sobol`.

**ORE:** `QuantLib::SobolRsg` —
[`QuantLib/ql/math/randomnumbers/sobolrsg.hpp`](../../reference/ORE/QuantLib/ql/math/randomnumbers/sobolrsg.hpp),
[`sobolrsg.cpp`](../../reference/ORE/QuantLib/ql/math/randomnumbers/sobolrsg.cpp).

**Correspondence:** Not a numerical-parity claim, and deliberately so. Both this engine
and QuantLib generate a **Sobol low-discrepancy sequence** — the same well-known class of
quasi-random sequence, described in `sobolrsg.hpp`'s own header comment as based on the
Bratley–Fox / Jäckel primitive-polynomial direction-number construction (a specific,
published, standard method for generating Sobol sequences). `scipy.stats.qmc.Sobol`
implements an independent, well-established implementation of the same Sobol
construction, not QuantLib's own Gray-code C++ generator — so the two produce
*different* (but equally valid) point sets, not bit-identical ones. This is a deliberate
engineering choice, not a gap: the mathematical property this project's parity claim
actually depends on is that *whatever* well-formed low-discrepancy/uniform sequence feeds
into the Brownian bridge (below) produces the statistically correct bridged distribution
— which is independently and exactly verified — not that the two engines draw the same
literal random numbers.

## 2. Brownian bridge construction

**This engine:** `engine/simulation.py::_build_bridge_matrix`,
`apply_brownian_bridge`.

**ORE:** `QuantLib::BrownianBridge` —
[`QuantLib/ql/methods/montecarlo/brownianbridge.hpp`](../../reference/ORE/QuantLib/ql/methods/montecarlo/brownianbridge.hpp),
[`brownianbridge.cpp`](../../reference/ORE/QuantLib/ql/methods/montecarlo/brownianbridge.cpp)
— specifically `BrownianBridge::initialize()` (construction) and
`BrownianBridge::transform()` (application to one path).

**Correspondence: algorithmically identical, restructured for vectorization.**
`BrownianBridge::initialize()` builds a set of index/weight arrays via a recursive
bisection: the last time point is always constructed first (from the first input
variate), then each subsequent variate bisects the widest still-unconstructed gap in the
time grid, recording a left/right neighbor pair, an interpolation weight for each
neighbor, and a conditional standard deviation for that gap — the standard
"Path Generation by Brownian Bridge" construction (originally due to Peter Jäckel's
*Monte Carlo Methods in Finance*, credited in the QuantLib source's own header comment).
`_build_bridge_matrix` in this codebase implements the *exact same* index/weight
recursion — the same `j`/`k`/`l` bisection search, the same `j==0` vs. `j!=0` branch for
the weight and standard-deviation formulas — confirmed line-for-line against
`initialize()`.

The one structural difference is deliberate and value-preserving: QuantLib's
`transform()` applies the recursion directly to **one path's** input variates via a
sequential loop (`output[l] = leftWeight*output[j-1] + rightWeight*output[k] +
stdDev*begin[i]`, an in-place recursive substitution), because QuantLib generates and
prices one Monte Carlo path at a time. This engine instead *unrolls that same recursion
once into an explicit linear operator* — a matrix `B` such that `W = B @ Z` reproduces
exactly what `transform()` would compute for any input `Z` — so that every simulated
scenario can be bridged in a single batched matrix multiply (`_apply_bridge_matrix`)
rather than a per-path loop, which is what makes it JAX/GPU-vectorizable. The final step
in both — converting the bridged absolute path values back into standardized sequential
increments (`output[i] -= output[i-1]; output[i] /= sqrtdt_[i]` in QuantLib, `dW =
diff(W); dW / sqrt(dt)` in `apply_brownian_bridge`) — is identical.

**Verified:** `tests/test_simulation.py::TestBrownianBridge` (the resulting
matrix reproduces the exact covariance structure `min(s,t)` real Brownian motion has —
the property this construction exists to guarantee) and
`tests/test_ore_parity.py::TestBrownianBridgeParity` (below).

## 3. Interest rate model: Hull-White 1-Factor

**This engine:** `engine/simulation.py::_simulate_cross_asset_paths_jit`
(short-rate step), `compute_hw_A_matrix` (today's-curve calibration).

**ORE, two equivalent formulations:**
- The plain, single-currency `QuantLib::HullWhite` model —
  [`QuantLib/ql/models/shortrate/onefactormodels/hullwhite.hpp`](../../reference/ORE/QuantLib/ql/models/shortrate/onefactormodels/hullwhite.hpp),
  [`hullwhite.cpp`](../../reference/ORE/QuantLib/ql/models/shortrate/onefactormodels/hullwhite.cpp)
  (specifically `HullWhite::A`, inherited `Vasicek::B`, `HullWhite::discountBondOption`) —
  this is the class every direct ORE cross-check test in this project's test suite
  actually uses.
- The multi-currency `QuantExt::CrossAssetModel`'s rates leg,
  `QuantExt::Lgm1fConstantParametrization` —
  [`QuantExt/qle/models/irlgm1fparametrization.hpp`](../../reference/ORE/QuantExt/qle/models/irlgm1fparametrization.hpp),
  [`irlgm1fconstantparametrization.hpp`](../../reference/ORE/QuantExt/qle/models/irlgm1fconstantparametrization.hpp),
  and the bond-pricing formulas in `QuantExt::LinearGaussMarkovModel` —
  [`QuantExt/qle/models/lgm.hpp`](../../reference/ORE/QuantExt/qle/models/lgm.hpp) — this is
  the class `ORE.CrossAssetModel` actually instantiates (live-verified via the SWIG
  bindings in an earlier session: `ORE.IrLgm1fConstantParametrization`, not
  `ORE.HullWhite`, is what a `CrossAssetModel` is built from). See
  [below](#a-parametrization-note-lgm-vs-plain-hull-white) for why both formulations are
  the same model.

### 3a. Short-rate transition (Monte Carlo step)

**This engine's formula** (`_simulate_cross_asset_paths_jit`'s `step_fn`):
```
decay      = exp(-a * dt)
variance   = (1 - exp(-2*a*dt)) / (2*a)
r(t+dt)    = r(t)*decay + theta*(1 - decay) + sigma*sqrt(variance)*Z
```
This is the exact closed-form transition of the Ornstein-Uhlenbeck / Hull-White SDE
`dr = a(theta - r)dt + sigma*dW` — a standard, textbook result, not itself something
QuantLib's `HullWhite` class computes directly (that class is calibrated to reproduce
today's curve exactly and doesn't expose a "simulate the short rate forward" method on
its own; simulation is normally done via a `StochasticProcess`, e.g.
`QuantLib::HullWhiteProcess`, or, in ORE's multi-currency case, `IrLgm1fStateProcess`).
The `variance` term matches `QuantExt::IrLgm1fStateProcess::variance()`
(`QuantExt/qle/processes/irlgm1fstateprocess.hpp`) under the LGM-to-short-rate identity
below.

**Regression note:** this exact transition formula is the one whose `theta*(1-decay)`
term was found missing (and fixed) during this project's most recent thorough-testing
pass — see the root [README.md](../../README.md)'s "Correctness fixes (post-Phase-5)"
section. This C++ source read is additional, independent confirmation that
`theta*(1-decay)`, not a bare `theta`, is the mathematically correct term: it is exactly
the standard OU/Vasicek/Hull-White transition mean any textbook derivation (or a
from-scratch derivation of the SDE's solution) produces, and is consistent with
`IrLgm1fStateProcess::expectation()` returning the *unchanged* state value (LGM's own
state variable is driftless — see below), which is only consistent with a Hull-White
short rate `r(t)` derived from that state reverting correctly to `theta`, not drifting.

### 3b. Today's-curve calibration: A(t,T) and B(t,T)

**This engine's formula** (`compute_hw_A_matrix`, and the `B_matrix` computation inline
in `generate_paths`):
```
B(t,T) = (1 - exp(-a*(T-t))) / a

A(t,T) = [P(0,T)/P(0,t)] * exp( B(t,T)*f(0,t) - (sigma^2/(4a))*(1-exp(-2at))*B(t,T)^2 )
```
where `f(0,t)` is today's instantaneous forward rate at `t` (computed here by finite
difference on the interpolated zero curve; see `_initial_log_discount`).

**ORE's formula**, `HullWhite::A(Time t, Time T)`
(`QuantLib/ql/models/shortrate/onefactormodels/hullwhite.cpp`, lines 75-83):
computes `B(t,T)` via the inherited `Vasicek::B` (the same `(1-exp(-a*(T-t)))/a`), reads
`forward = termStructure()->forwardRate(t,t,...)` (today's instantaneous forward, the
same `f(0,t)` this engine computes independently), and combines them as
`exp(B(t,T)*forward - 0.25*(sigma*B(t,T))^2 * B(0,2t)) * P(0,T)/P(0,t)` — algebraically
identical to this engine's formula, since `0.25*sigma^2*B(t,T)^2*B(0,2t) =
(sigma^2/(4a))*(1-exp(-2at))*B(t,T)^2` (substituting `B(0,2t) = (1-exp(-2at))/a`).

**Verified exactly** (not just algebraically): this project's earlier live-testing
sessions confirmed this engine's `A(t,T)*exp(-B(t,T)*r)` reproduces
`ORE.HullWhite.discountBond(t,T,r)` to machine precision (`~1e-12` relative) across many
`(t,T,r)` combinations, both flat and sloped input curves. Reading `HullWhite::A`'s
actual C++ here confirms *why*: it's the identical closed-form expression, not a
coincidental numerical match.

### A parametrization note: LGM vs. plain Hull-White

`QuantExt::CrossAssetModel`'s interest rate factors are, by default, parametrized as
**Linear Gaussian Markov (LGM)** models (`Lgm1fConstantParametrization`), not as plain
`QuantLib::HullWhite` objects — a different (but provably equivalent) way of writing
the same short-rate model down. Reading `irlgm1fconstantparametrization.hpp` directly
gives the exact relationship, with `scaling=1, shift=0` (the default, and the case this
engine's own parameters `hw_a`/`hw_sigma` correspond to):

```
H(t)    = (1 - exp(-kappa*t)) / kappa        <- identical shape to this engine's B(t,T),
                                                 with kappa == this engine's hw_a
zeta(t) = alpha^2 * t                         <- accumulated variance
alpha(t) = alpha                              <- constant; this engine's hw_sigma
```

LGM represents the model state as a driftless variable `x(t)` (confirmed directly in
`QuantExt::IrLgm1fStateProcess::expectation()`, which returns `x0` unchanged — no drift
term at all) and expresses bond prices and the numéraire as closed-form functions of `x`,
`H(t)`, and `zeta(t)` (`QuantExt::LinearGaussMarkovModel::discountBond`/`numeraire`,
`QuantExt/qle/models/lgm.hpp` lines 227-280) — algebraically the same
`A(t,T)*exp(-B(t,T)*r)` affine bond-price family this engine and plain `HullWhite` both
use, under the standard affine change of variables relating LGM's `x` to a short rate
`r`. This engine simulates `r(t)` directly (the plain Hull-White parametrization); ORE's
`CrossAssetModel` simulates `x(t)` (the LGM parametrization) for its own internal
numerical/calibration convenience. Both are the same physical model; this project's
existing formula-level cross-checks (against `ORE.HullWhite`, the plain-parametrized
class that is directly comparable to this engine's own direct-`r(t)` formulas) remain the
correct and sufficient verification route — re-deriving this engine's simulation in
terms of LGM's `x` state purely to match `CrossAssetModel`'s internal variable choice
would not change any output number, only which intermediate variable is carried through
the computation.

**Live-verified parameter identities** (`tests/test_ore_parity.py`): `H(t)` computed by
`ORE.IrLgm1fConstantParametrization` matches this engine's `B(t,T)` (with `t=0`) exactly;
`zeta(t)` matches `hw_sigma^2 * t` exactly; `alpha(t)` equals `hw_sigma` exactly.

## 4. Multi-asset correlation and the Cholesky factor

**This engine:** `generate_paths`'s "4. Joint Matrix" section — builds a **correlation**
(not covariance) Cholesky factor `L_t`, then multiplies each factor's own volatility in
explicitly inside `step_fn`.

**ORE:** `QuantExt::CrossAssetModel`'s own correlation handling
(`QuantExt/qle/models/crossassetmodel.cpp`) keeps a `correlation()` matrix (unit
diagonal, off-diagonal entries in `[-1,1]`) as a first-class, separate object from each
factor's own `alpha`/`sigma` volatility parameter — i.e. ORE's own class design already
enforces the same separation this engine's `L_t`-from-correlation-not-covariance fix
established. This is a useful independent design confirmation for the double-applied-
volatility bug described in the root [README.md](../../README.md)'s "Correctness fixes
(post-Phase-5)" section: ORE's own model never conflates "correlation structure" and
"marginal volatility" into one matrix in the first place, which is exactly the
distinction whose absence caused that bug.

## 5. Vanilla interest rate swap pricing

**This engine:** `engine/instruments/swap.py::_price_one_swap`.

**ORE:** `QuantLib::DiscountingSwapEngine::calculate()` —
[`QuantLib/ql/pricingengines/swap/discountingswapengine.cpp`](../../reference/ORE/QuantLib/ql/pricingengines/swap/discountingswapengine.cpp)
— plus the underlying coupon-amount formula, `QuantLib::IborCoupon::indexFixing()` —
[`QuantLib/ql/cashflows/iborcoupon.cpp`](../../reference/ORE/QuantLib/ql/cashflows/iborcoupon.cpp),
lines 119-137.

**Correspondence:** `DiscountingSwapEngine::calculate()` is thin orchestration: for each
leg, it calls `CashFlows::npvbps` (sum each cashflow's discounted amount off one shared
discount curve) and multiplies by a `+1`/`-1` payer/receiver sign, then sums the legs.
`_price_one_swap` implements the identical structure directly: `fixed_leg_pv` and
`float_leg_pv` are each `notional * rate_or_forward * accrual` summed and discounted
against `swap.discount_curve_index`'s curve, combined as `float_leg_pv - fixed_leg_pv`
and sign-flipped for `payer=False` — matching `DiscountingSwapEngine`'s own
`legNPV[i] *= arguments_.payer[i]` sign convention exactly (payer receives the floating
leg and pays the fixed leg, matching this engine's `npv = float - fixed`).

The floating leg's forward-rate formula
(`(P_fwd(t,T_start)/P_fwd(t,T_end) - 1)/accrual` in `_price_one_swap`) is the standard
simple-forward-rate-from-two-discount-factors identity, and corresponds to
`IborCoupon::indexFixing()`'s at-par branch, which forwards to
`IborIndex::forecastFixing(valueDate, endDate, spanningTime)` — the same "single forward
rate spanning the whole accrual period" convention (as opposed to a compounded
sub-period average), matching this engine's own single-period forward-rate formula and
this project's own `IborCoupon.usingAtParCoupons()` default, both live-verified in an
earlier session and confirmed here to be the actual C++ code path.

**Verified:** `tests/test_swap.py::TestPriceSwapsAgainstORE` (direct NPV
comparison against a real `ORE.VanillaSwap` + `ORE.DiscountingSwapEngine`, `<1e-6`
relative tolerance, across payer/receiver/par/spread/single-curve cases).

## 6. European swaption pricing: Jamshidian's decomposition

**This engine:** `engine/instruments/european_swaption.py::_price_one_swaption`,
`_solve_rstar`, `_bond_call`/`_bond_put`.

**ORE:** `QuantLib::JamshidianSwaptionEngine::calculate()` (and its private
`rStarFinder` functor) —
[`QuantLib/ql/pricingengines/swaption/jamshidianswaptionengine.cpp`](../../reference/ORE/QuantLib/ql/pricingengines/swaption/jamshidianswaptionengine.cpp)
— plus `HullWhite::discountBondOption`
(`QuantLib/ql/models/shortrate/onefactormodels/hullwhite.cpp`, lines 89-131) for the
zero-coupon bond option closed form.

**Correspondence: algorithmically identical, confirmed line-for-line.**
`JamshidianSwaptionEngine::calculate()`:
1. Builds `amounts` = every fixed coupon amount, with the notional added to the last
   entry — exactly this engine's `all_amounts = concatenate([cf_amounts, [notional,
   -notional]])` (see below for the sign difference on the second `notional` term).
2. Computes `maturity` = the exercise date's year-fraction (this engine's `T0`) and
   `valueTime` = `fixedResetDates[0]`'s year-fraction — **the fixed leg's own first
   accrual start date, not the exercise date** — exactly this engine's
   `accrual_start_time`/`T_start`, kept deliberately distinct from `T0`.
3. `rStarFinder::operator()` solves for the rate `x` at which
   `strike - sum_i(amounts[i] * discountBond(maturity, times[i], x) / discountBond(maturity, valueTime, x)) == 0`
   — i.e. discounting every leg back to `valueTime` (not to `maturity`/`T0`) via a
   division by `B = discountBond(maturity, valueTime, x)`, using Brent's method
   (`QuantLib::Brent`) over `x` in `[-10, 10]`.
4. Once `rStar` is found, each leg's strike is `discountBond(maturity, fixedPayTime,
   rStar) / B` (again normalized by the same `B`), and prices via
   `discountBondOption(w, strike, maturity, valueTime, fixedPayTime)` — a bond option
   whose *underlying* bond itself spans `[valueTime, fixedPayTime]`, not `[maturity,
   fixedPayTime]`.
5. `w = Payer ? Put : Call` — confirmed exactly this engine's own
   `bond_fn = _bond_put if swaption.payer else _bond_call` sign convention.

**This confirms, from the actual C++ source, exactly the fix this project made for
forward-starting swaptions** (see the root [README.md](../../README.md)'s Phase 5 section):
an earlier version of this engine incorrectly assumed the exercise date `T0` and the
underlying swap's accrual start `T_start` were the same point (true only for a
spot-starting swaption, where the two coincide up to the standard settlement lag).
Reading `JamshidianSwaptionEngine::calculate()` here shows QuantLib's own reference
implementation was *never* making that assumption — `valueTime` is explicitly read from
`fixedResetDates[0]`, independent of `maturity` (`T0`), for every swaption regardless of
whether it is spot- or forward-starting. This engine's fix (adding the `P(T0,T_start)`
leg with a negative amount, so the sum is normalized relative to `T_start` the same way
QuantLib's `B`-division does) is confirmed here to be the mathematically correct
approach, not merely a fix that happened to make test numbers match.

**One presentational difference, not a formula difference:** this engine represents
QuantLib's `.../B` normalization (every price divided by `discountBond(maturity,
valueTime, x)`) as an explicit extra cashflow (`-notional` at `T_start`) summed
alongside the others and priced with the *same* bond-option formula as every other leg,
rather than as a division applied after the fact. Both are the same algebra (dividing by
`B` and multiplying every strike by `1/B` is equivalent to adding a `-notional` leg
whose own bond option cancels the `T_start` numéraire term in the sum) — confirmed
numerically to agree to `~1e-6` relative precision or better in every existing
`tests/test_european_swaption.py` cross-check, both spot- and forward-starting.

**The bond-option closed form** (`_bond_call`/`_bond_put` vs. `HullWhite::
discountBondOption`): both compute the standard Black-formula-on-a-bond-price value,
with volatility `sigma_p = sigma*B(T_opt,S)*sqrt((1-exp(-2*a*(T_opt-t)))/(2a))` (this
engine's `_bond_option_sigma`) matching QuantLib's `v = sigma()*B(maturity,
bondMaturity)*sqrt(0.5*(1-exp(-2a*maturity))/a)` term-for-term (QuantLib's `maturity`
here is this engine's `T_opt - t`, i.e. QuantLib always conditions from `t=0`, while
this engine's conditional-pricing generalization allows an arbitrary `t` — see
[Instruments: European Swaptions](../instruments/european-swaptions.md#6-conditional-future-time-pricing)).

**Verified:** `tests/test_european_swaption.py::TestAgainstOREJamshidianEngine` (direct
NPV comparison against real `ORE.Swaption` + `ORE.JamshidianSwaptionEngine`, spot- and
forward-starting, payer/receiver, `<1e-4` relative tolerance) and
`tests/test_ore_parity.py::TestJamshidianRStarParity` (below — an independent
reimplementation of `rStarFinder`'s exact root-finding condition, cross-checked against
this engine's own `_solve_rstar` output).

## 7. American & Bermudan swaptions: numeric LGM backward induction

**This engine:** `engine/instruments/bermudan_swaption.py` — `_lgm_bond`,
`_hagan_quadrature_weights`, `_rollback_one_step`, `_run_backward_induction` — plus
`engine/instruments/american_swaption.py`, a thin wrapper that discretizes a continuous
exercise window into a `bermudan_swaption.BermudanSwaptionConfig` and delegates entirely
to this same engine.

**ORE:** `QuantExt::NumericLgmMultiLegOptionEngineBase::calculate()` —
[`QuantExt/qle/pricingengines/numericlgmmultilegoptionengine.cpp`](../../reference/ORE/QuantExt/qle/pricingengines/numericlgmmultilegoptionengine.cpp)
— backed by `QuantExt::LgmConvolutionSolver2` —
[`QuantExt/qle/models/lgmconvolutionsolver2.hpp/.cpp`](../../reference/ORE/QuantExt/qle/models/lgmconvolutionsolver2.cpp)
— built on `QuantExt::LinearGaussMarkovModel` —
[`QuantExt/qle/models/lgm.hpp`](../../reference/ORE/QuantExt/qle/models/lgm.hpp). Trade-level
routing confirmed in
[`OREData/ored/portfolio/builders/swaption.hpp`](../../reference/ORE/OREData/ored/portfolio/builders/swaption.hpp)/`.cpp`.

**Correspondence: full algorithm-by-algorithm writeup in
[american-bermudan-swaptions.md](../instruments/american-bermudan-swaptions.md)**, which is more extensive than a
single-section summary can cover — includes the exercise-window discretization formula
(American-as-fine-Bermudan), the state-grid/quadrature construction, and the
numeraire-deflation requirement for the backward induction to be mathematically valid at
all. One deviation from this codebase's usual pattern is important enough to call out
here directly: **this module does not reuse `compute_hw_A`/`_hw_B`** (section 3 above) —
building it surfaced a live, verified finding that `ORE.HullWhite` and
`ORE.LinearGaussMarkovModel`, despite sharing `(a, sigma)` and today's curve, are not the
same numerical model realization for `t>0` (a genuine ~0.6% bond-price difference at their
own respective "no shock" reference states, `t=3y`). Since ORE's actual Bermudan/American
engine is built on `LinearGaussMarkovModel`, this module uses a separate, independently
live-verified closed form (`_lgm_bond`, matching `ORE.LinearGaussMarkovModel.discountBond`
to ~1e-16 relative) exclusively, rather than the `HullWhite`-parametrized formula used
everywhere else in this codebase. This nuances, but does not contradict, section 3's
"parametrization note" above (verified equivalent at `t=0`; the two diverge only for
`t>0`, which the swap/Jamshidian pricers never need to evaluate since they always condition
either at `t=0` or via the model's own Markov-conditional formula rather than a second,
independently-parametrized model object).

**Verified:** `tests/test_bermudan_swaption.py` (the engine) and
`tests/test_american_swaption.py` (the discretization wrapper) — see
[american-bermudan-swaptions.md](../instruments/american-bermudan-swaptions.md)'s "Tested by" section for the full
breakdown (closed-form primitives vs. live ORE LGM objects, single-exercise-date
convergence to an independent Jamshidian-style closed form, monotonicity bounds, and the
American-discretization/mid-coupon edge cases).

## 8. Value at Risk & Expected Shortfall

**This engine:** `engine/risk/statistics.py::value_at_risk`,
`expected_shortfall`.

**ORE:** `QuantLib::GenericRiskStatistics<GaussianStatistics>::valueAtRisk`,
`::expectedShortfall` (the `QuantLib::RiskStatistics` typedef) —
[`QuantLib/ql/math/statistics/riskstatistics.hpp`](../../reference/ORE/QuantLib/ql/math/statistics/riskstatistics.hpp),
lines 178-205 — built on `QuantLib::GeneralStatistics::percentile` —
[`QuantLib/ql/math/statistics/generalstatistics.cpp`](../../reference/ORE/QuantLib/ql/math/statistics/generalstatistics.cpp),
lines 88-110.

**Correspondence: confirmed exactly, formula and edge cases both.**
`GeneralStatistics::percentile(percent)` sorts the (weight, value) sample ascending, then
walks forward accumulating weight until the cumulative weight reaches `percent *
totalWeight`, returning that sample's value — the "lower/nearest-rank" order statistic
this engine's `value_at_risk` reproduces via `sorted_pnl[floor(N*(1-percentile))]` for
unit weights (confirmed identical by direct construction: with every weight equal to 1,
walking forward until cumulative count reaches `percent*N` is exactly indexing
`sorted[floor(percent*N)]` for the standard 0-indexed convention `percentile.cpp` uses,
confirmed to `1e-9` absolute tolerance against `ORE.RiskStatistics.valueAtRisk` directly
in `tests/test_statistics.py`).

`RiskStatistics::valueAtRisk(centile)` calls `percentile(1-centile)`, floors at `0.0`,
and negates — exactly this engine's `max(-sorted_pnl[idx], 0.0)`.
`RiskStatistics::expectedShortfall(centile)` sets `target = -valueAtRisk(centile)`, then
averages every sample **strictly less than** `target` (`xi < target`, a value-based
filter, not a positional slice of the sorted array) — exactly this engine's
`tail_mask = pnl < -var`, confirmed as the deliberately-chosen-over-a-positional-slice
formula in this project's own regression test
(`tests/test_statistics.py::TestExpectedShortfallAgainstORE::
test_matches_ore_with_ties_at_var_boundary`, written before this C++ source was
available, purely from adversarial live-testing — this read confirms that test's
positional-vs-value-based conclusion was correct by reading the actual source, not just
inferring it from output numbers).

Both `valueAtRisk` and `expectedShortfall` require `centile` in `[0.9, 1.0)`
(`QL_REQUIRE(centile>=0.9 && centile<1.0, ...)`) — the exact range this project's own
`tests/test_statistics.py::TestValueAtRiskEdgeCases::
test_ore_rejects_percentile_outside_0_9_to_1` locks in from live-testing; this read
confirms it's an explicit, deliberate `QL_REQUIRE` in the source, not an implementation
accident. The empty-tail case (`expectedShortfall` with no samples below `target`) is
guarded by `QL_ENSURE(N != 0, "no data below the target")` — the exact error message
string this project's own tests assert against, confirmed here as the literal C++
source text (not independently re-derived).

**Verified:** `tests/test_statistics.py` (all classes; direct `ORE.RiskStatistics`
comparison, including the tie-at-VaR-boundary and empty-tail edge cases).

## Summary table

| Algorithm | This engine | ORE C++ source |
|---|---|---|
| Sobol sequence | `generate_sobol_normals` | `QuantLib::SobolRsg` (`ql/math/randomnumbers/sobolrsg.cpp`) — same sequence *class*, independent implementation |
| Brownian bridge | `_build_bridge_matrix`, `apply_brownian_bridge` | `QuantLib::BrownianBridge::initialize`/`transform` (`ql/methods/montecarlo/brownianbridge.cpp`) |
| HW1F short-rate transition | `_simulate_cross_asset_paths_jit` | Standard OU/Vasicek transition; consistent with `QuantExt::IrLgm1fStateProcess::variance` (`qle/processes/irlgm1fstateprocess.hpp`) |
| HW1F A(t,T)/B(t,T) | `compute_hw_A_matrix` | `QuantLib::HullWhite::A`, `Vasicek::B` (`ql/models/shortrate/onefactormodels/hullwhite.cpp`) |
| Correlation vs. volatility separation | `joint_covariance` → correlation-only `L_t` | `QuantExt::CrossAssetModel` correlation matrix (`qle/models/crossassetmodel.cpp`) |
| Swap pricing | `_price_one_swap` | `QuantLib::DiscountingSwapEngine::calculate`, `IborCoupon::indexFixing` |
| Jamshidian swaption decomposition | `_price_one_swaption`, `_solve_rstar` | `QuantLib::JamshidianSwaptionEngine::calculate`, `rStarFinder` |
| Bond option (Black-on-bond) | `_bond_call`/`_bond_put` | `QuantLib::HullWhite::discountBondOption` |
| American/Bermudan LGM bond price | `bermudan_swaption._lgm_bond` | `QuantExt::LinearGaussMarkovModel::discountBond` (`qle/models/lgm.hpp`) |
| American/Bermudan backward induction | `bermudan_swaption._run_backward_induction` | `QuantExt::NumericLgmMultiLegOptionEngineBase::calculate`, `LgmConvolutionSolver2` |
| American exercise-window discretization | `american_swaption.AmericanSwaptionConfig.to_bermudan` | `NumericLgmMultiLegOptionEngineBase::calculate`'s American branch |
| VaR | `value_at_risk` | `QuantLib::RiskStatistics::valueAtRisk` → `GeneralStatistics::percentile` |
| Expected Shortfall | `expected_shortfall` | `QuantLib::RiskStatistics::expectedShortfall` |

## Tested by

- `tests/test_ore_parity.py` — the tests specific to this page: independent
  reimplementations of small pieces of the cited C++ algorithms (the LGM parameter
  identities, the Jamshidian `rStarFinder` condition, the `GeneralStatistics::percentile`
  walk), each cross-checked against this engine's own output, so a future change to this
  engine's formulas that silently drifts from the *algorithm* (not just from a
  previously-recorded ORE output number) fails loudly.
- Every other test file listed in each section above, which cross-check against the
  *installed* `ORE` package's actual runtime behavior — this page's own contribution is
  connecting those already-passing behavioral tests to the specific C++ source lines that
  produce that behavior.
