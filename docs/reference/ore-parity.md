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

## One pipeline

`price_portfolio` on an `engine.market.Market` (HTTP `POST /portfolio/price`) reproduces ORE's
classic exposure and sensitivity pipeline: `CrossAssetModel` simulation, `ScenarioSimMarket`,
`ValuationEngine` with each trade's own t=0 engine, `ExposureCalculator` and
`SensitivityAnalysis`. The next section maps it; sections 1 to 10 map the individual
algorithms. The interest-rate model of a currency is ORE's LGM in either of its
parametrizations (Hagan's, the default, or Hull-White's); until roadmap 1.3 the Hull-White
model was a separate pipeline with known differences from ORE, closed with it
([I-42](../planning/known-issues.md#i-42) to [I-47](../planning/known-issues.md#i-47)).

## The market path

(The pipeline; the section keeps its earlier name for links.)

| ORE (file under `reference/ORE/`) | Engine | Checked against ORE by | Measured |
|---|---|---|---|
| `CrossAssetModel` (QuantExt/qle/models/crossassetmodel.cpp): LGM per currency, FX/EQ Black-Scholes, LGM measure | `engine.simulation.cam` | tests/test_cam.py: ζ, H, bond price and numeraire vs `LinearGaussMarkovModel`; one step vs `IrLgm1fStateProcess` | 1e-12 |
| `IrLgm1fPiecewiseConstantHullWhiteAdaptor` (`<LGM>` with `VolatilityType HullWhite`), not bound in Python | `engine.simulation.cam` (`volatility_type="HullWhite"`), `engine.models.lgm.hull_white_zeta` | tests/test_cam.py: path curves vs QuantLib `HullWhite::discountBond` at the state's short rate; martingales; ζ against its integral. tests/test_end_to_end.py: the simulation's paths priced by QuantLib | 1e-12 |
| Exact discretization, `CrossAssetAnalytics` (qle/models/crossassetanalytics.hpp) | `cam.step_moments` (Gauss-Legendre per volatility piece), `flexible_cholesky` | tests/test_cam.py: every covariance block = the integral of the Brownian loadings; Cholesky vs `CholeskyDecomposition`; martingales exact and by Monte Carlo (FP64, FP32) | 1e-12 (analytic) |
| `CrossAssetModelScenarioGenerator::nextPath` (OREAnalytics/orea/scenario/crossassetmodelscenariogenerator.cpp): model-implied curves, `ModelImpliedYtsFwdFwdCorrected` index curves, floor 1e-5, LGM numeraire | `engine.simulation.scenario_market` | tests/test_cam.py (curves reprice the input at t=0, basis deterministic, floor) | exact |
| `ScenarioSimMarket` curves (`addYieldCurve`): LogLinear, FlatFwd, the tenor points at the times from the as-of date on every simulation date | `engine.models.curves.DiscountCurve`, `scenario_market.as_of_tenor_times` (since roadmap 3.2, I-84) | tests/test_valuation.py hands each path curve to ORE as `ORE.DiscountCurve`; tests/test_ore_xva_parity.py: ORE's own scenario dump rebuilt from each path's state | 1e-12; 1.2e-11 (dump) |
| `CrossAssetModelBuilder` IR calibration (`CalibrationSwaptions`, Bootstrap) | `engine.calibration.cam` | same helpers and bootstrap as below; tests/test_ore_xva_parity.py: ORE's calibrated simulation rebuilt by the engine's model | 1.2e-11 (dump) |
| The cube's `T0`: every trade on the `ScenarioSimMarket` of the as-of date | `engine.valuation.context.simulation_market_today` (since roadmap 3.2, I-85) | tests/test_ore_xva_parity.py | 3.9e-11 |
| `Trade::maturity()` (OREData/ored/portfolio/swaption.cpp step 8: a cash-settled option's last exercise date) | `engine.valuation.portfolio.trade_maturity` (since roadmap 3.2, I-86) | tests/test_ore_xva_parity.py (the profiles of ORE's cube) | exact |
| `ValuationEngine::buildCube` (orea/engine/valuationengine.cpp), `recalibrate = true` | `engine.valuation.portfolio` | per trade type, below | — |
| `FixingManager::applyFixings` (orea/simulation/fixingmanager.cpp) | `engine.valuation.legs.path_fixings` | tests/test_valuation.py | 1e-12 |
| `DiscountingSwapEngine` with at-par coupons (`IborCouponPricer::initializeCachedData`), `hasOccurred` | `engine.valuation.legs` | tests/test_valuation.py (t=0 and every path/date), tests/test_shared_portfolio.py | 1e-10 |
| `BlackMultiLegOptionEngine` (qle/pricingengines/blackmultilegoptionengine.cpp), incl. `ParYieldCurve` cash settlement; `DynamicSwaptionVolatilityMatrix` on paths | `engine.valuation.european` | OREApp oracle at t=0; QuantLib `BachelierSwaptionEngine` on every path | 1e-10 (t=0), 2e-14 (cash) |
| QuantLib `JamshidianSwaptionEngine` (configured, not ORE's default) | `engine.valuation.jamshidian` | tests/test_jamshidian.py (QuantLib's engine, and its formula at the exact root); tests/test_end_to_end.py on Hull-White paths | Brent's 1e-8 on r* (≤ 5e-6); 1e-10 at the exact root |
| `LgmBuilder` + `IrModelBuilder::buildSwaptionBasket` (OREData/ored/model/), `AnalyticLgmSwaptionEngine`, `NumericLgmMultiLegOptionEngine` | `engine.valuation.bermudan`, `engine.calibration.ore_lgm` | tests/test_ore_lgm_calibration.py (OREApp, `Calibration=Bootstrap`), tests/test_valuation.py (paths, recalibrated on ORE's path curves) | 2e-11 (t=0), 1e-8 (paths) |
| `OptionWrapper` / `BermudanOptionWrapper` (OREData/ored/portfolio/optionwrapper.cpp) | `engine.valuation.options` | tests/test_valuation.py (exercise, physical vs cash) | rule-level |
| `DiscountingRiskyBondEngine` without credit | `engine.valuation.portfolio.bond_legs` | tests/test_shared_portfolio.py vs `DiscountingBondEngine` | 2e-16 |
| `ExposureCalculator` (orea/aggregation/exposurecalculator.cpp): EPE, ENE, EE_B, EEE_B, PFE, time-weighted EPE_B/EEPE_B, Basel horizon | `engine.risk.exposure` | tests/test_ore_xva_parity.py: the engine's profiles of ORE's own cube are ORE's reports, per trade and netting set; tests/test_portfolio_market_path.py (a bill's EE_B identity) | 1.8e-13 |
| The assembled pipeline: `XvaAnalytic` (`EXPOSURE`, `PFE`) | `price_portfolio` | tests/test_ore_xva_parity.py: on ORE's paths each trade's cube against ORE's (L3); independent simulations within four standard errors (L4) | swaps, Europeans, bonds 3.0e-11 of scale; Bermudans/Americans 0.5–4% ([I-49](../planning/known-issues.md#i-49)) |
| `SensitivityAnalysis`, `SensitivityCube` (orea/engine/sensitivityanalysis.cpp, orea/cube/sensitivitycube.cpp) | `engine.risk.sensitivities` (Bump), `engine.risk.greeks` (AD) | tests/test_sensitivities.py, tests/test_greeks.py (the two methods agree; AD against finite differences) — **not against an OREApp sensitivity run** ([I-51](../planning/known-issues.md#i-51)) | — |

The whole portfolio at t=0 against ORE, trade by trade, on one sloped market:
tests/test_shared_portfolio.py ([the shared portfolio](../planning/details/ore-parity-validation.md#the-shared-portfolio)), worst case 3.8e-11 (a calibrated Bermudan).

**The assembled pipeline against an ORE simulation** (roadmap 3.2,
`tests/support/ore_xva_oracle.py`, [details](../planning/details/ore-parity-validation.md#the-ore-simulation-oracle-roadmap-32)):
swaps, Europeans and bonds equal ORE's cube on ORE's own paths, and the exposure equals ORE's
in distribution; Bermudans and Americans are 0.5–4% apart on the paths
([I-49](../planning/known-issues.md#i-49), roadmap 3.5). **Not yet compared:** the Greeks
against ORE's sensitivity analytic (L5, [I-51](../planning/known-issues.md#i-51)).

### Verification gates

The plan's gates (§4) ask for the ORE source that decides each question and an OREApp run
that shows it. Most were closed from the source plus tests against ORE's Python bindings, not
by an OREApp run; the column says which.

| Gate | Answer | Source | Evidence |
|---|---|---|---|
| V-1 | `ValuationEngine` recalibrates every model on each scenario (`recalibrate = true` by default, `recalibrateModels` → `LgmBuilder::recalibrate`); non-simulated swaption vols are the t=0 surface seen from the scenario date (`DynamicSwaptionVolatilityMatrix`) | valuationengine.cpp, lgmbuilder.cpp, qle/termstructures/dynamicswaptionvolmatrix.cpp | **OREApp run** (tests/test_ore_xva_parity.py): Europeans on the decayed vols equal ORE on its paths. **Half closed:** two recalibration details differ, 0.5–4% on the paths ([I-49](../planning/known-issues.md#i-49)) |
| V-2 | Absolute shift, `ShiftScheme::Forward`; delta `up − base`, gamma `up − 2·base + down`; scaling by target over actual shift | sensitivitycube.cpp, sensitivityscenariogenerator.cpp | Source; tests/test_sensitivities.py |
| V-3 | `thetaDate = asof + thetaPeriod` (calendar); sim market rebuilt at `thetaDate` from the original curves, fixed in dates (not renormalised); fixings backfilled; period flows added | sensitivityanalysis.cpp | Source; tests/test_sensitivities.py, tests/test_trade_dates.py |
| V-4 | Can ORE's scenario dump reprice the cube? Yes: its numeraire gives each path's LGM state, which rebuilds every dumped curve | crossassetmodelscenariogenerator.cpp, scenariowriter.cpp | **OREApp run**: tests/test_ore_xva_parity.py, 1.2e-11 (roadmap 3.2) |
| V-5 | Co-terminal basket from the trade's exercise dates; `CoterminalDealStrike` (first fixed rate less spread) with the ±3 std-dev fallback, or ATM; `ReferenceCalibrationGrid` keeps one helper per interval; an American's expiries are the grid dates in its window | lgmbuilder.cpp, irmodelbuilder.cpp | **OREApp run**: tests/test_ore_lgm_calibration.py, calibrated price = ORE's to 2e-11 |
| V-6 | `CalibrationSwaptions`: tenor-based expiries and terms, ATM | crossassetmodelbuilder.cpp | **OREApp run**: tests/test_ore_xva_parity.py (ORE's calibrated simulation rebuilt by the engine's calibration, both models) |
| V-7 | Drift and covariance of every IR/FX/EQ block under the LGM measure | crossassetanalytics.hpp | Bindings: tests/test_cam.py (analytic martingales exact; covariance = loading integral) |
| V-8 | `includeReferenceDateEvents = false`: a flow paid on the valuation date has occurred | cashflow.cpp, valuationengine.cpp | Bindings: tests/test_valuation.py |
| V-9 | A Treasury without credit is `DiscountingRiskyBondEngine` with no credit curve and no security spread, i.e. discounting | discountingriskybondengine.cpp | Bindings: tests/test_shared_portfolio.py |
| V-10 | Index curves use the index day counter's time `t_dc`; on the engine's ACT/365 grid `t_dc = t` | crossassetmodelscenariogenerator.cpp | Source |

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
   for how the two relate; they are not the same model under the same parameters).
3. **OREData / OREAnalytics** (`reference/ORE/OREData/`, `reference/ORE/OREAnalytics/`) —
   ORE's own trade-configuration and analytics layer (XML parsing, scenario generation
   orchestration). Not a source of core math this project's own pricing formulas need to
   match; not covered on this page.

---

## 1. Sobol sequence generation

**This engine:** `engine/simulation/random.py::generate_sobol_normals`, via
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

**This engine:** `engine/simulation/random.py::_build_bridge_matrix`,
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

**Verified:** `tests/test_random.py::TestBrownianBridge` (the resulting
matrix reproduces the exact covariance structure `min(s,t)` real Brownian motion has —
the property this construction exists to guarantee) and
`tests/test_ore_parity.py::TestBrownianBridgeParity` (below).

## 3. Interest rate model: the LGM, in two parametrizations

**This engine:** `engine/simulation/cam.py` (the components and the exact step moments),
`engine/simulation/scenario_market.py` (the model-implied curves and numeraire),
`engine/models/lgm.py` (the closed forms).

**ORE:** `QuantExt::CrossAssetModel`'s rates leg, an `IrLgm1fParametrization` per currency —
[`QuantExt/qle/models/irlgm1fparametrization.hpp`](../../reference/ORE/QuantExt/qle/models/irlgm1fparametrization.hpp) —
with the bond prices and numeraire of `QuantExt::LinearGaussMarkovModel`
([`QuantExt/qle/models/lgm.hpp`](../../reference/ORE/QuantExt/qle/models/lgm.hpp)). ORE's
`LgmData::VolatilityType` chooses the parametrization: `Hagan`
(`IrLgm1fPiecewiseConstantParametrization`, α piecewise constant) or `HullWhite`
(`IrLgm1fPiecewiseConstantHullWhiteAdaptor`, the short rate's σ piecewise constant).

### 3a. Short-rate transition (Monte Carlo step)

The LGM state is driftless under the LGM measure, with variance ζ: the step from t to
t + dt is Gaussian with variance ζ(t+dt) − ζ(t), coupled to the FX and equity states by
ORE's exact discretization (`CrossAssetStateProcess::ExactDiscretization`,
`CrossAssetAnalytics`). This engine computes the step moments once on the host and evolves
every path on the device; one step equals ORE's `IrLgm1fStateProcess` (tests/test_cam.py).
There is no short-rate recursion: the short rate is a function of the state,
r(t) = f(0,t) + H′(t)x + ζ(t)H(t)H′(t). (Until roadmap 1.3 the Hull-White model simulated an
Ornstein-Uhlenbeck short rate toward a constant θ, which was not fitted to a sloped curve,
[I-42](../planning/known-issues.md#i-42).)

### 3b. Today's-curve calibration: A(t,T) and B(t,T)

The model is fitted to today's curve by construction:
P(t,T|x) = P(0,T)/P(0,t) · exp(−(H(T)−H(t))x − ½(H(T)² − H(t)²)ζ(t)). In the Hull-White
parametrization (H(t) = (1 − e^{−at})/a, ζ(t) = ∫σ²e^{2as}ds) this is QuantLib's
`HullWhite::discountBond(t, T, r) = A(t,T)e^{−B(t,T)r}` at the state's short rate:
`B(t,T) = (H(T) − H(t))/H′(t)` and `HullWhite::A` (`ql/models/shortrate/onefactormodels/hullwhite.cpp`)
are the same expression. Verified to 1e-12 on sloped curves
(`tests/test_cam.py::test_hull_white_path_curves_equal_quantlibs_hull_white`).

### A parametrization note: LGM vs. plain Hull-White

The two parametrizations are one model family. α(t) = σ(t)e^{at} turns a Hull-White
volatility into the LGM's; with the same *number* they are different models (LGM with
constant α is Hull-White with the decaying short-rate volatility αe^{−at}: 14% lower at 5y at
a = 3%). An earlier version of this page called them "provably equivalent" under the same
(a, σ), and the Bermudan engine's development measured a 0.6% bond-price difference
between `ORE.HullWhite` and the constant-α LGM at 3y; both are this parametrization
difference (that comparison also put r = f(0,t) against x = 0, which are different states).
The engine therefore implements both parametrizations of the one model, as ORE does, and its
Bermudan engine is always ORE's LGM.

**Live-verified parameter identities** (`tests/test_ore_parity.py::TestLgmParametrizationParity`):
`H(t)` matches `ORE.IrLgm1fConstantParametrization` exactly, ζ(t) = α²t, α(t) = α.

## 4. Multi-asset correlation and the Cholesky factor

**This engine:** `engine/simulation/cam.py::flexible_cholesky` of each step's covariance
(`step_moments`).

**ORE:** `QuantExt::CrossAssetModel` keeps a `correlation()` matrix (unit diagonal) separate
from each factor's volatility parameter (`qle/models/crossassetmodel.cpp`), and its state
process takes the square root of each step's covariance with `pseudoSqrt`, which with the
CAM's default `SalvagingAlgorithm::None` is `CholeskyDecomposition(cov, flexible = true)`.
`flexible_cholesky` reproduces it, including a zero-volatility factor's zero column
(tests/test_cam.py).

## 5. Vanilla interest rate swap pricing

**This engine:** `engine/valuation/legs.py` (`legs_npv`, `today_npv`, `legs_cube`).

**ORE:** `QuantLib::DiscountingSwapEngine::calculate()` —
[`QuantLib/ql/pricingengines/swap/discountingswapengine.cpp`](../../reference/ORE/QuantLib/ql/pricingengines/swap/discountingswapengine.cpp)
— plus the underlying coupon-amount formula, `QuantLib::IborCoupon::indexFixing()` —
[`QuantLib/ql/cashflows/iborcoupon.cpp`](../../reference/ORE/QuantLib/ql/cashflows/iborcoupon.cpp),
lines 119-137.

**Correspondence:** `DiscountingSwapEngine::calculate()` is thin orchestration: for each
leg, it calls `CashFlows::npvbps` (sum each cashflow's discounted amount off one shared
discount curve) and multiplies by a `+1`/`-1` payer/receiver sign, then sums the legs.
`legs_npv` implements the identical structure directly: the fixed and floating legs are each
`notional * rate_or_forward * accrual` summed and discounted on the currency's discount
curve, combined as floating minus fixed and sign-flipped for `payer=False` — matching `DiscountingSwapEngine`'s own
`legNPV[i] *= arguments_.payer[i]` sign convention exactly (payer receives the floating
leg and pays the fixed leg, matching this engine's `npv = float - fixed`).

The floating leg's forward is `(P_fwd(t,start)/P_fwd(t,end) - 1)/spanning` over each
coupon's at-par forecast period, `IborCouponPricer::initializeCachedData`'s at-par branch
(`QuantLib/ql/cashflows/couponpricer.cpp`): the period is the fixing's value date to the
index maturity (or the accrual period, adjusted, for an at-par coupon) and `spanning` is the
**index** day counter's year fraction over it, which is then multiplied by the leg's accrual.
`engine.models.ore_builders.par_coupon_forecast_period` returns the three from QuantLib's own
coupon. Until 2026-09-29 the engine divided by the leg's accrual instead, which differs for
any leg day count other than the index's ([I-36](../planning/known-issues.md#i-36)).

**Verified:** `tests/test_swap.py::TestAgainstORE` (direct NPV comparison against a real
`ORE.VanillaSwap` + `ORE.DiscountingSwapEngine`, 1e-12, across payer/receiver/par/spread
cases) and `tests/test_valuation.py` (today and on every path and date).

## 6. European swaption pricing: Jamshidian's decomposition

**The configured alternative.** ORE's default European engine is `BlackMultiLegOptionEngine`
(Bachelier on the market volatility, `engine.valuation.european`, [above](#the-market-path)).
This section documents the Jamshidian engine (`PricingConfig.european="Jamshidian"` with its
Hull-White model), which matches QuantLib's `JamshidianSwaptionEngine` and refuses what it
refuses: a floating spread, `a <= 0`, cash settlement ([I-37](../planning/known-issues.md#i-37),
[I-41](../planning/known-issues.md#i-41), [I-52](../planning/known-issues.md#i-52)).

**This engine:** `engine/valuation/jamshidian.py` (`jamshidian_npv`, `_solve_decreasing_root`),
`engine/models/hull_white.py::bond_call`/`bond_put`. It is written in discount factors only
(the LGM form of the Hull-White model), so it prices on any curve; the description below of
QuantLib's algorithm applies term by term.

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
   `option = bond_put if legs.payer else bond_call` sign convention.

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
numerically to agree within QuantLib's root tolerance in every `tests/test_jamshidian.py`
cross-check, both spot- and forward-starting.

**The bond-option closed form** (`bond_call`/`bond_put` vs. `HullWhite::
discountBondOption`): both compute the standard Black-formula-on-a-bond-price value,
with volatility `sigma_p = sigma*B(T_opt,S)*sqrt((1-exp(-2*a*(T_opt-t)))/(2a))` (this
engine's `lgm.bond_option_sigma`) matching QuantLib's `v = sigma()*B(maturity,
bondMaturity)*sqrt(0.5*(1-exp(-2a*maturity))/a)` term-for-term (QuantLib's `maturity`
here is this engine's `T_opt - t`, i.e. QuantLib always conditions from `t=0`, while
this engine's conditional-pricing generalization allows an arbitrary `t` — see
[Instruments: European Swaptions](../instruments/european-swaptions.md#6-conditional-future-time-pricing)).

**Verified:** `tests/test_jamshidian.py::TestAgainstQuantLibJamshidianEngine` (direct NPV
comparison against real `ORE.Swaption` + `ORE.JamshidianSwaptionEngine`, spot- and
forward-starting, payer/receiver, within QuantLib's Brent tolerance on r*, 5e-6) and
`::TestAgainstQuantLibWithAnExactRoot` (QuantLib's formula evaluated at the exact root,
1e-10). The engine solves x* to float64 rounding (by `JamshidianEngineConfig.solver`, Newton
by default) where QuantLib's Brent stops at 1e-8.

## 7. American & Bermudan swaptions: numeric LGM backward induction

**This engine:** `engine/instruments/bermudan_swaption.py` — `_lgm_bond`,
`_hagan_quadrature_weights`, `_rollback_one_step`, `_run_backward_induction` — plus
`engine/instruments/american_swaption.py`, which supplies ORE's American option times and
exercise style to this same engine; `engine/valuation/bermudan.py` calibrates each trade's
basket and runs the engine today and on every path.

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
here directly: building it surfaced that `ORE.HullWhite` and `ORE.LinearGaussMarkovModel`,
given the same `(a, sigma)` and today's curve, are not the same model for `t>0` (a ~0.6%
bond-price difference at `t=3y`). Since ORE's actual Bermudan/American
engine is built on `LinearGaussMarkovModel`, this module uses its closed form (`_lgm_bond`,
matching `ORE.LinearGaussMarkovModel.discountBond` to ~1e-16 relative) exclusively. The
"difference" is the parametrization (section 3's note): with the same number the two are
different models, and the Hull-White parametrization of the LGM is the Hull-White model.

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
differ by 2.3e-3 on this trade (1211.47 vs 1214.23, [I-31](../planning/known-issues.md#i-31)).

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
exercise value entirely ([I-29](../planning/known-issues.md#i-29)). It is gone by construction now:
exercise is given in **dates**, as ORE takes it, and an exercise date equal to an accrual
date maps to the identical time. `TestExerciseDatesAreExact` pins that.

### 7b. ORE's own LGM engine, reached in-process (2026-09-23)

The missing SWIG constructor blocks building `NumericLgmMultiLegOptionEngine` directly. It
does not block reaching it: ORE users never build it directly either. They run an analytic
over a trade, and ORE's engine factory builds it. `tests/support/ore_lgm_oracle.py` does exactly
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
   ([I-06](../planning/known-issues.md#i-06)). The mid-period *Bermudan* behaviour the register had
   called a 7.4x error turned out to be ORE's own rule.
2. **Floating coupons were projected over the wrong period** — ORE's LGM engine uses the
   index's fixing period, the engine used the accrual period; ~2e-4 on every trade
   ([I-31](../planning/known-issues.md#i-31)).
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

**This engine:** `engine/risk/sensitivities.py` (`Bump`, the default: ORE's sensitivity
analysis, and Theta for both methods), `engine/risk/greeks.py` (`AD`). See
[Greeks](../risk/greeks.md).

**ORE:** `OREAnalytics::SensitivityAnalysis::generateSensitivities` —
[`OREAnalytics/orea/engine/sensitivityanalysis.cpp`](../../reference/ORE/OREAnalytics/orea/engine/sensitivityanalysis.cpp)
— backed by `SensitivityScenarioGenerator`/`ShiftScenarioGenerator`
([`OREAnalytics/orea/scenario/sensitivityscenariogenerator.cpp`](../../reference/ORE/OREAnalytics/orea/scenario/sensitivityscenariogenerator.cpp),
[`shiftscenariogenerator.cpp`](../../reference/ORE/OREAnalytics/orea/scenario/shiftscenariogenerator.cpp))
and `SensitivityCube`
([`OREAnalytics/orea/cube/sensitivitycube.cpp`](../../reference/ORE/OREAnalytics/orea/cube/sensitivitycube.cpp)).

**Bump: the same computation.** ORE's sensitivity simulation market samples every curve at
the configured tenors; one tenor's zero rate (or one swaption quote) is shifted at a time
(`ShiftType=Absolute`, 1bp), the portfolio repriced, and `SensitivityCube` differences the
NPVs (`delta = NPV_up − NPV_base`, `gamma = NPV_up − 2·NPV_base + NPV_down`). Bermudans are
recalibrated under every shift. `portfolio_sensitivities` reproduces it (gates V-2, V-3),
not yet against an OREApp sensitivity run ([I-51](../planning/known-issues.md#i-51)).

**AD: the same quantity's limit.** The exact derivative of each trade's price in each
market-curve pillar (`jax.grad` and a Hessian-diagonal), scaled by the shift — ORE's Delta
without the forward difference's truncation, as ORE's own closed-form
`DiscountingSwapEngineDeltaGamma`/`BlackSwaptionEngineDeltaGamma` engines compute it
([`QuantExt/qle/pricingengines/discountingswapenginedeltagamma.hpp`](../../reference/ORE/QuantExt/qle/pricingengines/discountingswapenginedeltagamma.hpp),
cross-checked by ORE in
[`OREAnalytics/test/sensitivityvsanalytic.cpp`](../../reference/ORE/OREAnalytics/test/sensitivityvsanalytic.cpp)).
Vega per swaption quote: through the bilinear surface's weights, and for a
Bermudan/American through its calibration by the implicit function theorem, since
`RiskFactorKey::KeyType` has no entry for a raw model parameter — ORE's Vega is always on the
quotes the model is calibrated to.

**Rho:** confirmed absent from ORE entirely — `QuantExt::RiskFactorKey::KeyType`
([`QuantExt/qle/termstructures/scenario.hpp`](../../reference/ORE/QuantExt/qle/termstructures/scenario.hpp))
has no rho-specific entry, so the interest-rate-curve Delta is ORE's equivalent of a
textbook Rho.

**Theta:** `generateSensitivities`'s theta branch advances the evaluation date by a
configured period (calendar days, [I-38](../planning/known-issues.md#i-38)), builds the market
there from today's curves fixed in dates, reprices, and adds back the interim cashflows:
`Theta = NPV(t+dt) − NPV(t) + CF(t, t+dt]`. Both methods report it (`trade_theta`).

**Root-finds.** Naively differentiating through a root solver gives a silently wrong
gradient (a bisection's comparisons have none, Newton's iterations only the steps'). The
Jamshidian root and each calibration bucket get the implicit function theorem's derivative
instead (`engine.numerics.roots.implicit_root`'s `custom_jvp`, `_bootstrap_jacobian`); see [Greeks](../risk/greeks.md#differentiating-through-bisection-root-finds).

**Verified:** `tests/test_sensitivities.py`, `tests/test_trade_dates.py` (Theta against ORE),
`tests/test_greeks.py` and `tests/test_greeks_bermudan.py` (AD against finite differences,
Vega through the recalibration, agreement with the bump method).

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
x* gradient bug (cross-referenced above in section 9), and the root solver both
calibrations use ([decision A-21](../../compliance/decisions.md): Newton by default, the
bisection kept as the reference; each reaches ORE's root more exactly than ORE's own
solvers). Two findings worth
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

**The parametrizations.** Section 3's ["parametrization note"](#a-parametrization-note-lgm-vs-plain-hull-white)
applies here: the calibration is of the LGM ORE's Bermudan engine uses. A Hull-White currency
of the cross-asset model is bootstrapped to the same helpers and converted bucket by bucket
(`hull_white_matching_zeta`): helper prices depend on the model only through ζ at their
expiry and H, so matching ζ at every bucket end is exact (`tests/test_hull_white_model.py::TestCalibration`).
The cross-asset model's and each trade's calibration is `engine/calibration/ore_lgm.py`
(QuantLib's `SwaptionHelper`s, ORE's bootstrap, 2e-11 against `LgmBuilder`,
`tests/test_ore_lgm_calibration.py`); `basket.py`/`lgm.py` serve the standalone route.

**Verified:** `tests/test_calibration_basket.py` (15 tests), `tests/test_calibration_lgm.py`
(9 tests), `tests/test_calibration_integration.py` (6 tests) — see
[Calibration: Tested by](calibration.md#tested-by) for the full breakdown.

## Summary table (the algorithms)

| Algorithm | This engine | ORE C++ source |
|---|---|---|
| Sobol sequence | `random.generate_sobol_normals` | `QuantLib::SobolRsg` (`ql/math/randomnumbers/sobolrsg.cpp`) — same sequence *class*, independent implementation |
| Brownian bridge | `random._build_bridge_matrix`, `apply_brownian_bridge` | `QuantLib::BrownianBridge::initialize`/`transform` (`ql/methods/montecarlo/brownianbridge.cpp`) |
| LGM state step (exact) | `cam.step_moments`, `evolve_states` | `CrossAssetStateProcess::ExactDiscretization`, `CrossAssetAnalytics`, `IrLgm1fStateProcess` |
| Hull-White parametrization | `cam.IrComponent(volatility_type="HullWhite")`, `lgm.hull_white_zeta` | `IrLgm1fPiecewiseConstantHullWhiteAdaptor`; bonds equal `QuantLib::HullWhite::discountBond` |
| Model-implied curves, numeraire | `scenario_market.implied_log_discounts`, `lgm_numeraire` | `CrossAssetModelScenarioGenerator::nextPath`, `LinearGaussMarkovModel::discountBond`/`numeraire` |
| Correlation and square root | `cam.flexible_cholesky` | `CrossAssetModel` correlation; `CholeskyDecomposition(cov, flexible = true)` |
| Swap pricing | `valuation.legs.legs_npv` | `QuantLib::DiscountingSwapEngine::calculate`, `IborCouponPricer` at par |
| European (default) | `valuation.european` | `BlackMultiLegOptionEngine` (Bachelier), `ParYieldCurve` cash settlement |
| Jamshidian swaption decomposition | `valuation.jamshidian.jamshidian_npv`, `_solve_decreasing_root` | `QuantLib::JamshidianSwaptionEngine::calculate`, `rStarFinder` |
| Bond option (Black-on-bond) | `hull_white.bond_call`/`bond_put` | `QuantLib::HullWhite::discountBondOption` |
| American/Bermudan LGM bond price | `bermudan_swaption._lgm_bond` | `QuantExt::LinearGaussMarkovModel::discountBond` (`qle/models/lgm.hpp`) |
| American/Bermudan backward induction, including ORE's cashflow bookkeeping | `bermudan_swaption._backward_induction_arrays`, `_GridSchedule` | `QuantExt::NumericLgmMultiLegOptionEngineBase::calculate`, `LgmConvolutionSolver2` — equal to ORE's own engine to 1e-10, reached in-process via `OREApp`, see [7b](#7b-ores-own-lgm-engine-reached-in-process-2026-09-23) |
| Coupon membership and proration by exercise style | `bermudan_swaption.ExerciseStyle`, `prepare_bermudan` | `NumericLgmMultiLegOptionEngineBase::buildCashflowInfo` (`belongsToUnderlyingMaxTime_`, `couponRatio`) |
| Ibor projection in the LGM | `bermudan_swaption._cashflow_values_at_nodes` | `QuantExt::LgmVectorised::fixing` (index fixing period) |
| American exercise-window discretization | `AmericanSwaptionConfig.option_times` | `NumericLgmMultiLegOptionEngineBase::calculate`'s American branch (truncating step count) |
| Per-trade basket and bootstrap | `valuation.bermudan.calibration_basket`, `calibration.ore_lgm` | `IrModelBuilder::buildSwaptionBasket`, `LgmBuilder::calibrate` |
| Exercise wrapper | `valuation.options.wrap` | `OptionWrapper`, `BermudanOptionWrapper` |
| VaR | `var_es.value_at_risk` | `QuantLib::RiskStatistics::valueAtRisk` → `GeneralStatistics::percentile` |
| Expected Shortfall | `var_es.expected_shortfall` | `QuantLib::RiskStatistics::expectedShortfall` |
| Delta / Gamma / Vega (bump) | `sensitivities.portfolio_sensitivities` | `SensitivityScenarioGenerator`, `ShiftScenarioGenerator::applyShift`, `SensitivityCube` |
| Delta / Gamma / Vega (AD) | `greeks.portfolio_greeks` | the limit of the above; `DiscountingSwapEngineDeltaGamma`-style closed forms |
| Theta (evaluation-date roll) | `sensitivities.trade_theta` | `SensitivityAnalysis::generateSensitivities`'s theta branch |
| Co-terminal swaption basket (standalone route) | `calibration.basket.build_coterminal_basket` | `IrModelBuilder::buildSwaptionBasket` |
| LGM sigma bootstrap (standalone route) | `calibration.lgm.calibrate_lgm_sigma` | `LgmBuilder::calibrate` → `calibrateVolatilitiesIterative`, `CalibratedModel::calibrateIterative` |

## Tested by

- The pipeline: `tests/test_cam.py`, `tests/test_valuation.py` (every path test under both
  models), `tests/test_ore_lgm_calibration.py`, `tests/test_shared_portfolio.py`,
  `tests/test_portfolio_market_path.py`, `tests/test_sensitivities.py`, `tests/test_curves.py`,
  `tests/test_end_to_end.py` (the Hull-White simulation's paths priced in QuantLib) and
  `tests/test_market_risk_ore_parity.py`.
- `tests/test_ore_parity.py` — the tests specific to this page: independent
  reimplementations of small pieces of the cited C++ algorithms (the Brownian bridge, the LGM
  parameter identities, the `GeneralStatistics::percentile` walk), each cross-checked against
  this engine's own output, so a change that drifts from the *algorithm* fails loudly.
- `tests/test_jamshidian.py` — the Jamshidian engine against QuantLib's.
- `tests/test_ore_bermudan_oracle.py` (51 tests) — the external multi-exercise Bermudan
  oracle described in [7a](#7a-an-external-multi-exercise-oracle-does-exist-correction-2026-09-18).
- Every other test file listed in each section above, which cross-check against the
  *installed* `ORE` package's actual runtime behavior — this page's own contribution is
  connecting those already-passing behavioral tests to the specific C++ source lines that
  produce that behavior.
- `tests/test_models_piecewise_sigma.py`, `tests/test_calibration_basket.py`,
  `tests/test_calibration_lgm.py`, `tests/test_calibration_integration.py`,
  `tests/test_greeks.py`, `tests/test_greeks_bermudan.py` — the tests specific to sections
  9-10 (Greeks, calibration); see
  [Delta, Gamma, and Theta](../risk/greeks.md#tested-by),
  [Models & Trades](models-and-trades.md#tested-by), and
  [Calibration](calibration.md#tested-by) for the full per-file breakdown.
