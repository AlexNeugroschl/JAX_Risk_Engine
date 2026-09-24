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

**This engine:** `engine/simulation/market_model.py::generate_sobol_normals`, via
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

**This engine:** `engine/simulation/market_model.py::_build_bridge_matrix`,
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
rather than a per-path loop, which is what makes it JAX-vectorizable across accelerators (GPU/TPU). The final step
in both — converting the bridged absolute path values back into standardized sequential
increments (`output[i] -= output[i-1]; output[i] /= sqrtdt_[i]` in QuantLib, `dW =
diff(W); dW / sqrt(dt)` in `apply_brownian_bridge`) — is identical.

**Verified:** `tests/test_market_model.py::TestBrownianBridge` (the resulting
matrix reproduces the exact covariance structure `min(s,t)` real Brownian motion has —
the property this construction exists to guarantee) and
`tests/test_ore_parity.py::TestBrownianBridgeParity` (below).

## 3. Interest rate model: Hull-White 1-Factor

**This engine:** `engine/simulation/market_model.py::_simulate_cross_asset_paths_jit`
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
  bindings: `ORE.IrLgm1fConstantParametrization`, not
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

**Why `theta*(1-decay)`, not a bare `theta`.** This C++ source confirms
`theta*(1-decay)` is the mathematically correct term: it is exactly
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
established. This is a useful independent design confirmation for this engine's own
`L_t`-from-correlation-not-covariance approach: ORE's own model never conflates
"correlation structure" and "marginal volatility" into one matrix.

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
this project's own `IborCoupon.usingAtParCoupons()` default, both live-verified to be
the actual C++ code path.

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

**This confirms, from the actual C++ source, that this engine handles forward-starting
swaptions correctly.** The exercise date `T0` and the underlying swap's accrual start
`T_start` are not generally the same point — they coincide only for a spot-starting
swaption, up to the standard settlement lag. Reading `JamshidianSwaptionEngine::calculate()`
shows QuantLib's own reference implementation never assumes otherwise: `valueTime` is
explicitly read from `fixedResetDates[0]`, independent of `maturity` (`T0`), for every
swaption regardless of whether it is spot- or forward-starting. This engine mirrors that:
the `P(T0,T_start)` leg is included with a negative amount, so the sum is normalized
relative to `T_start` the same way QuantLib's `B`-division does.

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

### 7a. An external multi-exercise oracle does exist (correction, 2026-09-18)

**This project recorded, in `tests/test_bermudan_swaption.py`'s own module docstring and
in §10 below, that the full backward induction could not be cross-checked against a live
ORE engine end to end** — on the grounds that `ORE.NumericLgmMultiLegOptionEngine` is not
constructible through the installed SWIG bindings. **The premise is correct and still
holds** — re-verified 2026-09-18: both `ORE.NumericLgmMultiLegOptionEngine` and
`ORE.AnalyticLgmSwaptionEngine` expose no usable constructor. **The conclusion drawn from
it was too broad.** Two *other* multi-exercise engines in the same bindings are fully
constructible and price a genuine `ORE.BermudanExercise`:

- `ORE.TreeSwaptionEngine` — QuantLib's Hull-White trinomial tree.
- `ORE.FdHullWhiteSwaptionEngine` — QuantLib's Hull-White finite-difference solver.

So multi-exercise Bermudan values *can* be cross-checked end to end against a real,
independent ORE engine, and now are: `tests/test_ore_bermudan_oracle.py` (51 tests).
(And, since 2026-09-23, against ORE's own LGM engine itself — see
[7b](#7b-ores-own-lgm-engine-reached-in-process-2026-09-23).)

**What that comparison shows, and what it does not.** ORE's two engines are
`HullWhite`-parametrized while this module is `LinearGaussMarkovModel`-parametrized — the
very non-equivalence this section documents above. The agreement is therefore a
**model-level** agreement of a few percent (measured: ~3% at the money, up to ~15% deep
out of the money where the option is worth ~830 on a 1e6 notional), not the ~1e-4
numerical parity the European-swaption tests get against `ORE.JamshidianSwaptionEngine`,
where both sides share a parametrization. Three controls in that file attribute the
residual gap to the parametrization rather than to the backward induction:

1. **ORE's own two engines agree with each other to ≤1.3e-3** despite being entirely
   different numerical schemes (tree vs. PDE) — so the target value is not in doubt.
2. **This engine is fully grid-converged**: refining `n_per_std` from 48 to 384 moves the
   price by ~1e-5 relative, so the gap is not discretization error.
3. **The same-sized gap appears with a *single* exercise date**, where
   `tests/test_bermudan_swaption.py::TestSingleExerciseMatchesDirectIntegration` already
   proves this engine matches an independent direct integration to 2e-5. A discrepancy present with
   one exercise date and no larger with four is not coming from the early-exercise logic.

A fourth check is parametrization-free and therefore holds tightly: at `sigma -> 0` the
Bermudan must collapse to the intrinsic value of the forward-starting underlying, which
ORE values with a plain `DiscountingSwapEngine` and no model at all. It must be built with
QuantLib's **indexed** Ibor coupons, which project over each index fixing period as ORE's
LGM engine does; QuantLib's default at-par coupons project over the accrual period and
differ by 2.3e-3 on this trade (1211.47 vs 1214.23, [I-31](../known-issues.md#i-31)).

**No pricing defect was found *against these engines*.** (Against ORE's own LGM engine,
two were — see [7b](#7b-ores-own-lgm-engine-reached-in-process-2026-09-23). A few-percent
model gap is too coarse to see a 2e-4 projection error, and these Bermudan-only engines
cannot see an American one.) The three apparent discrepancies hit while building this
oracle all resolved to the test, not the engine: a ~40x error from building the ORE side's
underlying with `build_vanilla_swap` (whose 0% dummy forward curve is correct for this
engine and fatal for an ORE pricing engine, which actually reads it); a ~1e-2 curve-shape
error from building ORE's comparison curve with log-linear discount interpolation instead
of linear zero-rate; and a 12x overstatement at low vol from passing a *rounded* exercise
time. That last one was recorded as a **sensitivity, not a bug**: an exercise time of
`2.0137` instead of the true `2.0136986301369864` is 1.4e-6 late, far outside the
coupon-liveness tolerance then in use, so that date's fixed coupon was dropped from the
exercise value entirely ([I-29](../known-issues.md#i-29)). It is gone by construction now:
exercise is given in **dates**, as ORE takes it, and an exercise date equal to an accrual
date maps to the identical time. `TestExerciseDatesAreExact` pins that.

### 7b. ORE's own LGM engine, reached in-process (2026-09-23)

The missing SWIG constructor blocks building `NumericLgmMultiLegOptionEngine` directly. It
does not block reaching it: ORE users never build it directly either. They run an analytic
over a trade, and ORE's engine factory builds it. `engine/validation/ore_lgm_oracle.py` does exactly
that, in-process and entirely in memory: an `OREApp` run of the `NPV` analytic over a
`Swaption` trade XML, priced by `LGMGridSwaptionEngineBuilder` →
`NumericLgmMultiLegOptionEngine` with `Calibration=None`, on

- the engine's own underlying, passed as explicit schedule dates;
- a convention-defined `USD-SIMINDEX-6M` with `SimIndex`'s terms;
- the engine's zero curve, date-quoted and linear in zero rate;
- the engine's LGM parameters (`ReversionType=HullWhite`, `VolatilityType=Hagan`,
  `ShiftHorizon=0`; a piecewise `Sigma` as `VolatilityTimes`/`Volatility`).

Two inputs ORE requires but never reads for pricing are supplied so the model builder does
not fall back to dummies. One of them, the swap index, is **not** inert:
`IrModelBuilder` takes the LGM's own term structure from its discounting curve, so a
fallback would silently replace the model curve with a flat 1%.

**What it found, the first time it ran.** Three things the Hull-White comparison in 7a
could not see:

1. **American exercise was mispriced** — up to 6.0x for a payer, 0.09x for a receiver.
   ORE keeps a coupon until its accrual *end* for an American and credits
   `couponRatio(t)`; the engine priced Americans with the Bermudan rule
   ([I-06](../known-issues.md#i-06)). The mid-period *Bermudan* behaviour the register had
   called a 7.4x error turned out to be ORE's own rule.
2. **Floating coupons were projected over the wrong period** — ORE's LGM engine uses the
   index's fixing period, the engine used the accrual period; ~2e-4 on every trade
   ([I-31](../known-issues.md#i-31)).
3. **A closed-form exercise value is not ORE's number.** Mathematically it has the same
   limit as ORE's rolled-back `underlyingNpv`; numerically it differs by up to 1e-4 at a
   48-point grid (shrinking ~4x per doubling). The engine now replays ORE's cashflow
   bookkeeping itself.

After the changes, `tests/test_ore_lgm_parity.py` asserts equality to **1e-10** across 23
cases (aligned and mid-period Bermudans, Americans including high strike at low vol and a
truncating step count, piecewise volatility, zero vol); the measured worst case is
**8.7e-12**. 22 of the 23 fail against the previous engine.

**Verified:** `tests/test_bermudan_swaption.py` (the engine) and
`tests/test_american_swaption.py` (American option times and broken periods) — see
[american-bermudan-swaptions.md](../instruments/american-bermudan-swaptions.md)'s "Tested by" section for the full
breakdown (closed-form primitives vs. live ORE LGM objects, single-exercise-date
agreement with an independent direct integration, monotonicity bounds, and the
exercise-membership rules) — plus `tests/test_ore_bermudan_oracle.py` (51 tests, the
Hull-White oracle in 7a) and `tests/test_ore_lgm_parity.py` (23 tests, ORE's own LGM
engine, 7b).

## 8. Value at Risk & Expected Shortfall

**This engine:** `engine/risk/var_es.py::value_at_risk`,
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
in `tests/test_var_es.py`).

`RiskStatistics::valueAtRisk(centile)` calls `percentile(1-centile)`, floors at `0.0`,
and negates — exactly this engine's `max(-sorted_pnl[idx], 0.0)`.
`RiskStatistics::expectedShortfall(centile)` sets `target = -valueAtRisk(centile)`, then
averages every sample **strictly less than** `target` (`xi < target`, a value-based
filter, not a positional slice of the sorted array) — exactly this engine's
`tail_mask = pnl < -var`, confirmed as the deliberately-chosen-over-a-positional-slice
formula in this project's own regression test
(`tests/test_var_es.py::TestExpectedShortfallAgainstORE::
test_matches_ore_with_ties_at_var_boundary`, written before this C++ source was
available, purely from adversarial live-testing — this read confirms that test's
positional-vs-value-based conclusion was correct by reading the actual source, not just
inferring it from output numbers).

Both `valueAtRisk` and `expectedShortfall` require `centile` in `[0.9, 1.0)`
(`QL_REQUIRE(centile>=0.9 && centile<1.0, ...)`) — the exact range this project's own
`tests/test_var_es.py::TestValueAtRiskEdgeCases::
test_ore_rejects_percentile_outside_0_9_to_1` locks in from live-testing; this read
confirms it's an explicit, deliberate `QL_REQUIRE` in the source, not an implementation
accident. The empty-tail case (`expectedShortfall` with no samples below `target`) is
guarded by `QL_ENSURE(N != 0, "no data below the target")` — the exact error message
string this project's own tests assert against, confirmed here as the literal C++
source text (not independently re-derived).

**Verified:** `tests/test_var_es.py` (all classes; direct `ORE.RiskStatistics`
comparison, including the tie-at-VaR-boundary and empty-tail edge cases).

## 9. Delta, Gamma, Vega, and Theta

**This engine:** `engine/risk/greeks.py::swap_delta_gamma`, `swap_theta`,
`swaption_delta_gamma`, `swaption_theta`, `bermudan_delta_gamma`, `bermudan_theta`,
`bermudan_vega`.

**ORE:** `OREAnalytics::SensitivityAnalysis::generateSensitivities` —
[`OREAnalytics/orea/engine/sensitivityanalysis.cpp`](../../reference/ORE/OREAnalytics/orea/engine/sensitivityanalysis.cpp)
— backed by `OREAnalytics::SensitivityScenarioGenerator`/`ShiftScenarioGenerator` —
[`OREAnalytics/orea/scenario/sensitivityscenariogenerator.cpp`](../../reference/ORE/OREAnalytics/orea/scenario/sensitivityscenariogenerator.cpp),
[`shiftscenariogenerator.cpp`](../../reference/ORE/OREAnalytics/orea/scenario/shiftscenariogenerator.cpp)
— and `OREAnalytics::SensitivityCube` —
[`OREAnalytics/orea/cube/sensitivitycube.cpp`](../../reference/ORE/OREAnalytics/orea/cube/sensitivitycube.cpp).

**Correspondence: the same numerical quantity ORE reports, computed via automatic
differentiation instead of literal bump-and-revalue.** ORE's production path is finite
differences: `SensitivityScenarioGenerator` builds one perturbed market scenario per
curve pillar (a triangular-weighted bump, `ShiftScenarioGenerator::applyShift`, ORE's own
example config using `ShiftType=Absolute`, `ShiftSize=0.0001` — 1bp), reprices the whole
portfolio under each scenario, and `SensitivityCube` differences the resulting NPVs
(`delta = NPV_up - NPV_base`, `gamma = NPV_up - 2*NPV_base + NPV_down`). This engine
computes the exact analytic derivative of the SAME NPV with respect to the SAME curve
pillars (`jax.grad`/`jax.hessian`, via a JAX-differentiable curve interpolation,
`greeks.ZeroCurve`/`_zero_rate_at`, using the identical piecewise-linear-on-zero-rates
shape `compute_hw_A`/`_initial_log_discount` already use elsewhere in this codebase), then
scales by the same 1bp `bump_size` — giving ORE's own "dollar Delta/Gamma for a 1bp move,"
without finite-difference truncation error. This mirrors ORE's own use of closed-form
`DiscountingSwapEngineDeltaGamma`/`BlackSwaptionEngineDeltaGamma` engines
([`QuantExt/qle/pricingengines/discountingswapenginedeltagamma.hpp`](../../reference/ORE/QuantExt/qle/pricingengines/discountingswapenginedeltagamma.hpp),
[`blackswaptionenginedeltagamma.hpp`](../../reference/ORE/QuantExt/qle/pricingengines/blackswaptionenginedeltagamma.hpp))
as an independent, closed-form cross-check on its own bump-and-revalue numbers
(exercised by
[`OREAnalytics/test/sensitivityvsanalytic.cpp`](../../reference/ORE/OREAnalytics/test/sensitivityvsanalytic.cpp))
— this engine uses that closed-form route as its primary implementation, not just a
validation side-channel.

**Rho:** confirmed absent from ORE entirely — `QuantExt::RiskFactorKey::KeyType`
([`QuantExt/qle/termstructures/scenario.hpp`](../../reference/ORE/QuantExt/qle/termstructures/scenario.hpp))
has no rho-specific entry, and `ReportWriter::writeSensitivityReport`
([`OREAnalytics/orea/app/reportwriter.cpp`](../../reference/ORE/OREAnalytics/orea/app/reportwriter.cpp))
emits only "Delta"/"Gamma" columns for whatever risk factor was bumped, curves included —
so this engine's own interest-rate-curve Delta is ORE's exact equivalent of a textbook
Rho, and no separate function exists for it.

**Vega: implemented for Bermudan/American, via `engine/calibration/`.**
`SensitivityScenarioGenerator::generateSwaptionVolScenarios` bumps the market-quoted
implied-volatility surface used to CALIBRATE ORE's model — there is no
`RiskFactorKey::KeyType` anywhere for a raw model parameter, so `d(NPV)/d(hw_sigma)` was
never a well-defined ORE-equivalent Vega on its own. This was a genuine blocker, not
merely a missing function: until [`engine/calibration/`](calibration.md) existed to
calibrate a piecewise LGM `Sigma` to a basket of market swaption vols
(`ore::data::LgmBuilder::calibrate()`'s own `Bootstrap` path), there was no
market-vol-to-model relationship for this engine to differentiate through either.
`greeks.bermudan_vega` now computes `d(NPV)/d(market_vol_i)` for each basket instrument by
differentiating through `calibrate_lgm_sigma`'s bootstrap via the implicit function
theorem — see [Delta, Gamma, and Theta: Vega](../risk/greeks.md#vega-bermudanamerican-only)
for the full derivation, including two real bugs found while building it. Vega for
`swap.py`/`european_swaption.py` remains out of scope: a swap has no volatility exposure
at all, and `SwaptionConfig` (the European swaption pricer's config) was never migrated to
accept a calibrated `Sigma` the way `BermudanSwaptionConfig` was, so there is still no
market-vol-to-model relationship to differentiate through for a European swaption.

**Theta:** `SensitivityAnalysis::generateSensitivities`'s theta branch (lines ~253-307 of
the same file cited above) advances the evaluation date by a configured period (default
1 day), holds every market quote fixed, rebuilds term structures at the new reference
date, reprices, and adds back any interim cashflow —
`Theta = NPV(t+dt, same curve) - NPV(t) + CF(t, t+dt)`. This engine's `swap_theta`/
`swaption_theta`/`bermudan_theta` reproduce this literally; unlike Delta/Gamma, this is a
genuine forward difference along the time axis in both engines, not an autodiff
computation in either.

**Scope: every instrument in this codebase, not a formula gap.** This engine's Greeks used
to cover `swap.py` and `european_swaption.py` only, since `bermudan_swaption.py`/
`american_swaption.py`'s backward induction ran entirely in plain NumPy (a CPU grid
method), with no JAX computational graph for `jax.grad` to differentiate at all. Porting
that backward induction to `jax.lax.scan` (see
[American & Bermudan Swaptions](../instruments/american-bermudan-swaptions.md)) removed
that blocker; `bermudan_delta_gamma`/`bermudan_theta` now cover Bermudan/American exactly
as `swap_delta_gamma`/`swap_theta` cover swaps. See
[Delta, Gamma, and Theta: Scope](../risk/greeks.md#scope-every-rate-derivative-instrument-in-this-codebase).

**Two bugs found while building this correspondence, unrelated to the correspondence
itself, both the same root cause.** Naively differentiating through a bisection-based
root-find gives a silently *wrong*, not merely imprecise, gradient — a bisection's own
comparison (`jnp.where(val > 0.0, ...)`) has zero gradient everywhere, so `jax.grad`
straight through the unrolled loop ignores how the converged root actually moves with the
function's own inputs:

1. **`european_swaption.py::_solve_rstar`** (Jamshidian's bisection for the exercise
   boundary `r*`) — its naive gradient was silently `0.0` everywhere, regardless of the
   true root's actual sensitivity. Fixed via `jax.custom_jvp` implementing the implicit
   function theorem directly.
2. **`engine.calibration.basket._bisect_xstar`** — the LGM analogue of `_solve_rstar`,
   found while building `bermudan_vega`: understated `price_lgm_swaption`'s own gradient
   with respect to sigma by ~6%, rather than dropping it to exactly zero (a different
   magnitude of the same class of error, since this bisection's root feeds into a further
   set of formulas rather than being returned directly). Fixed with the identical
   `custom_jvp`/implicit-function-theorem pattern, plus registering
   `engine.models.lgm.Sigma` as a proper JAX pytree so a tangent can propagate into its
   `values` field when nested inside a larger argument tuple. See
   [Calibration](calibration.md#the-_bisect_xstar-gradient-bug) for the full incident.

See [Delta, Gamma, and Theta: Two real bugs this module found and
fixed](../risk/greeks.md#differentiating-through-bisection-root-finds) for the full account of
both, and each function's own docstring in `european_swaption.py`/`engine/calibration/basket.py`.

**Verified:** `tests/test_greeks.py` — finite-difference bump-and-revalue cross-checks
(the literal ORE-style computation) for both swap and swaption Delta/Gamma, plus a direct
cross-check against a real `ORE.VanillaSwap`/`ORE.Swaption` repriced under a bumped
`ORE.FlatForward` curve. `tests/test_greeks_bermudan.py` — the same style of check for
Bermudan/American Delta/Gamma/Theta, plus Vega against a literal finite-difference
recalibration (bump one basket instrument's market vol, rerun
`calibrate_lgm_sigma`, reprice).

## 10. LGM calibration: bootstrap fit of a piecewise sigma to market swaption vols

**This engine:** `engine/calibration/basket.py::build_coterminal_basket`,
`price_lgm_swaption`; `engine/calibration/lgm.py::calibrate_lgm_sigma`.

**ORE:** `ore::data::IrModelBuilder::buildSwaptionBasket()`
(`OREData/ored/model/irmodelbuilder.cpp`) for the basket; `ore::data::LgmBuilder::calibrate()`
(`OREData/ored/model/lgmbuilder.cpp`, lines 209-212, `calibrateVolatilitiesIterative`) for
the bootstrap itself, which calls `QuantLib::CalibratedModel::calibrateIterative`; the LGM
analogue of `QuantExt::AnalyticLgmSwaptionEngine` (`QuantExt/qle/pricingengines/`) for the
per-instrument pricer.

**Full writeup in [Calibration](calibration.md)**, which is more extensive than a
single-section summary can cover here — includes the `aTimes = swaptionExpiries[:-1]`
triangular-bootstrap construction, why mean reversion is never calibrated, and the
`_bisect_xstar` gradient bug (cross-referenced above in section 9). Two findings worth
calling out directly on this page:

**`price_lgm_swaption` could not be checked against ORE's own engine directly.**
`QuantExt::AnalyticLgmSwaptionEngine`'s constructor is not exposed through this codebase's
installed ORE Python bindings — confirmed by reading
`ORE-SWIG/QuantExt-SWIG/SWIG/qle_pricingengines.i` directly, which declares only
`enableCache`/`clearCache`/`setZetaShift`/`resetZetaShift` for
`ORE.AnalyticLgmSwaptionEngine`, no usable constructor. Verification therefore runs two
independent routes instead of a single direct NPV comparison: every individual formula
piece (`bond_price`, `bond_option_sigma`, `numeraire`) checked to machine precision against
`ORE.LinearGaussMarkovModel`'s own exposed methods, and the full swaption price
cross-checked against a numeraire-deflated Monte Carlo simulation of `x(T0) ~ N(0,
zeta(T0))` — LGM's own exact terminal distribution, per
`QuantExt::IrLgm1fStateProcess::variance`. See [Calibration: two-route
verification](calibration.md#two-route-verification) for the full account, including the
Monte Carlo test's own bug (naive `P(0,T0)` discounting instead of numeraire deflation,
initially showing a spurious ~11% "error" that was fixed once the test correctly deflated
by `engine.models.lgm.numeraire` — a lesson about LGM's own measure, not a pricer defect).

**Scope note:** the missing constructor blocks a direct NPV comparison for
`price_lgm_swaption` specifically, against the *analytic* LGM engine. It does **not** mean
Bermudan/American values have no external oracle — `ORE.TreeSwaptionEngine` and
`ORE.FdHullWhiteSwaptionEngine` are constructible and supply one; see
[7a](#7a-an-external-multi-exercise-oracle-does-exist-correction-2026-09-18).

**The `HullWhite` vs. `LinearGaussMarkovModel` non-equivalence, confirmed relevant here
too.** Section 3's ["parametrization note"](#a-parametrization-note-lgm-vs-plain-hull-white)
already documents that `ORE.HullWhite` and `ORE.LinearGaussMarkovModel` are not the same
numerical model realization for `t>0`, despite sharing `(a, sigma)` and today's curve (a
~0.6% bond-price divergence at `t=3y`, found while building `bermudan_swaption.py`;
`tests/test_ore_coverage_hardening.py::TestHullWhiteVersusLgmBondPrices` now measures this
directly across a grid of `(t,T)` rather than leaving it as a single remembered figure,
and bounds it from *below* as well as above so that a future ORE version making the two
agree fails loudly rather than silently leaving percent-level tolerances unjustified).
`engine/calibration/` inherits this directly: `price_lgm_swaption` is built exclusively on
`engine.models.lgm`, never `engine.models.hull_white`, for the same reason
`bermudan_swaption.py` is — the model being calibrated is the one ORE's own Bermudan engine
actually uses.

**Verified:** `tests/test_calibration_basket.py` (15 tests), `tests/test_calibration_lgm.py`
(9 tests), `tests/test_calibration_integration.py` (6 tests) — see
[Calibration: Tested by](calibration.md#tested-by) for the full breakdown.

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
| American/Bermudan backward induction, including ORE's cashflow bookkeeping | `bermudan_swaption._backward_induction_arrays`, `_GridSchedule` | `QuantExt::NumericLgmMultiLegOptionEngineBase::calculate`, `LgmConvolutionSolver2` — equal to ORE's own engine to 1e-10, reached in-process via `OREApp`, see [7b](#7b-ores-own-lgm-engine-reached-in-process-2026-09-23) |
| Coupon membership and proration by exercise style | `bermudan_swaption.ExerciseStyle`, `prepare_bermudan` | `NumericLgmMultiLegOptionEngineBase::buildCashflowInfo` (`belongsToUnderlyingMaxTime_`, `couponRatio`) |
| Ibor projection in the LGM | `bermudan_swaption._cashflow_values_at_nodes` | `QuantExt::LgmVectorised::fixing` (index fixing period) |
| American exercise-window discretization | `american_swaption.AmericanSwaptionConfig.option_times` | `NumericLgmMultiLegOptionEngineBase::calculate`'s American branch (truncating step count) |
| VaR | `value_at_risk` | `QuantLib::RiskStatistics::valueAtRisk` → `GeneralStatistics::percentile` |
| Expected Shortfall | `expected_shortfall` | `QuantLib::RiskStatistics::expectedShortfall` |
| Delta / Gamma (curve pillar bump-and-revalue, via autodiff) | `greeks.swap_delta_gamma`, `greeks.swaption_delta_gamma`, `greeks.bermudan_delta_gamma` | `SensitivityScenarioGenerator`/`ShiftScenarioGenerator::applyShift`, `SensitivityCube::delta`/`gamma` |
| Theta (evaluation-date roll) | `greeks.swap_theta`, `greeks.swaption_theta`, `greeks.bermudan_theta` | `SensitivityAnalysis::generateSensitivities`'s theta branch |
| Vega (Bermudan/American, via calibration) | `greeks.bermudan_vega` | `SensitivityScenarioGenerator::generateSwaptionVolScenarios`, bumping the vol surface calibration itself consumes |
| Co-terminal swaption calibration basket | `calibration.basket.build_coterminal_basket` | `ore::data::IrModelBuilder::buildSwaptionBasket` (`OREData/ored/model/irmodelbuilder.cpp`) |
| LGM closed-form swaption pricer | `calibration.basket.price_lgm_swaption` | LGM analogue of `QuantExt::AnalyticLgmSwaptionEngine` (constructor not exposed via SWIG bindings — see section 10) |
| LGM sigma bootstrap calibration | `calibration.lgm.calibrate_lgm_sigma` | `ore::data::LgmBuilder::calibrate` → `calibrateVolatilitiesIterative` (`OREData/ored/model/lgmbuilder.cpp`), `QuantLib::CalibratedModel::calibrateIterative` |

## Tested by

- `tests/test_ore_parity.py` — the tests specific to this page: independent
  reimplementations of small pieces of the cited C++ algorithms (the LGM parameter
  identities, the Jamshidian `rStarFinder` condition, the `GeneralStatistics::percentile`
  walk), each cross-checked against this engine's own output, so a future change to this
  engine's formulas that silently drifts from the *algorithm* (not just from a
  previously-recorded ORE output number) fails loudly.
- `tests/test_ore_bermudan_oracle.py` (50 tests) — the external multi-exercise Bermudan
  oracle described in [7a](#7a-an-external-multi-exercise-oracle-does-exist-correction-2026-09-18):
  this engine's backward induction against real `ORE.TreeSwaptionEngine` /
  `ORE.FdHullWhiteSwaptionEngine` objects, plus the controls that attribute the residual
  ~3% to the HW/LGM parametrization rather than to the induction, plus a
  parametrization-free `sigma -> 0` collapse to intrinsic that holds to 1e-3.
- `tests/test_ore_coverage_hardening.py` (36 tests) — tests *about* the discriminating
  power of the ORE comparisons themselves, plus the curve shapes the rest of the suite
  never exercises. Two findings of record, both pinned as assertions:
  - **A t=0 blind spot for the variance term of `A(t,T)`.** That term carries a factor
    `(1 - exp(-2at))` which is identically zero at `t=0`, so at `t=0` deleting it entirely
    moves an ATM swaption price by ~7e-6 relative — an order of magnitude *inside* the
    `rtol=1e-4` those comparisons assert. Most of this suite's ORE swaption comparisons
    price at `t=0`. Measured against the real suite: deleting the whole term fails exactly
    **one** test in `tests/test_european_swaption.py` (131 tests),
    `TestConditionalPricingAndExpiry::test_conditional_pricing_matches_ore_rebuilt_at_later_date`
    — that single test carries the suite's entire coverage of the term. This is a gap in
    the *tests*, not a defect in the formula (which is independently verified against
    QuantLib's C++ in section 3b and against live `ORE.HullWhite.discountBond`).
  - **Non-flat curves agree off-pillar to ~1e-8** (upward, inverted, humped, and
    negative-rate), but **at a pillar** the two disagree by up to ~1.5e-2. Neither library
    is wrong: under linear zero-rate interpolation `f(0,t)` has a genuine kink at each
    pillar (left and right derivatives differ), so the instantaneous forward is undefined
    exactly there. Pinned rather than fixed, with both an upper and a lower bound, so that
    a future move to a smooth interpolation fails the test and prompts folding pillar
    times into the off-pillar check.
- Every other test file listed in each section above, which cross-check against the
  *installed* `ORE` package's actual runtime behavior — this page's own contribution is
  connecting those already-passing behavioral tests to the specific C++ source lines that
  produce that behavior.
- `tests/test_models_piecewise_sigma.py`, `tests/test_calibration_basket.py`,
  `tests/test_calibration_lgm.py`, `tests/test_calibration_integration.py`,
  `tests/test_greeks_bermudan.py` — the tests specific to sections 9-10's newer additions
  (Bermudan/American Greeks and Vega, LGM calibration); see
  [Delta, Gamma, and Theta](../risk/greeks.md#tested-by),
  [Models & Trades](models-and-trades.md#tested-by), and
  [Calibration](calibration.md#tested-by) for the full per-file breakdown.
