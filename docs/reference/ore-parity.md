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

`price_portfolio` on an `engine.market_data.market.Market` (the run request's `npv`, `exposure` and `sensitivities`, HTTP `POST /runs`) reproduces ORE's
classic exposure and sensitivity pipeline: `CrossAssetModel` simulation, `ScenarioSimMarket`,
`ValuationEngine` with each trade's own t=0 engine, `ExposureCalculator` and
`SensitivityAnalysis`. The next section maps it; sections 1 to 10 map the individual
algorithms. The interest-rate model of a currency is ORE's LGM in either of its
parametrizations (Hagan's, the default, or Hull-White's); until 2026-10-01 the Hull-White
model was a separate pipeline with known differences from ORE, closed with it
([I-42](../planning/known-issues.md#i-42) to [I-47](../planning/known-issues.md#i-47)).

## The pipeline

(The pipeline; the section keeps its earlier name for links.)

| ORE (file under `reference/ORE/`) | Engine | Checked against ORE by | Measured |
|---|---|---|---|
| `CrossAssetModel` (QuantExt/qle/models/crossassetmodel.cpp): LGM per currency, FX/EQ Black-Scholes, LGM measure | `engine.models.cam` | tests/test_cam.py: ζ, H, bond price and numeraire vs `LinearGaussMarkovModel`; one step vs `IrLgm1fStateProcess` | 1e-12 |
| `IrLgm1fPiecewiseConstantHullWhiteAdaptor` (`<LGM>` with `VolatilityType HullWhite`), not bound in Python | `engine.models.cam` (`volatility_type="HullWhite"`), `engine.models.lgm.hull_white_zeta` | tests/test_cam.py: path curves vs QuantLib `HullWhite::discountBond` at the state's short rate; martingales; ζ against its integral. tests/test_end_to_end.py: the simulation's paths priced by QuantLib | 1e-12 |
| Exact discretization, `CrossAssetAnalytics` (qle/models/crossassetanalytics.hpp) | `cam.step_moments` (Gauss-Legendre per volatility piece), `flexible_cholesky` | tests/test_cam.py: every covariance block = the integral of the Brownian loadings; Cholesky vs `CholeskyDecomposition`; martingales exact and by Monte Carlo (FP64, FP32) | 1e-12 (analytic) |
| `CrossAssetModelScenarioGenerator::nextPath` (OREAnalytics/orea/scenario/crossassetmodelscenariogenerator.cpp): model-implied curves, `ModelImpliedYtsFwdFwdCorrected` index curves, floor 1e-5, LGM numeraire | `engine.market_simulation.scenario_market` | tests/test_cam.py (curves reprice the input at t=0, basis deterministic, floor) | exact |
| `ScenarioSimMarket` curves (`addYieldCurve`): LogLinear, FlatFwd, the tenor points at the times from the as-of date on every simulation date | `engine.market_data.curves.DiscountCurve`, `scenario_market.as_of_tenor_times` (since 2026-10-07, I-84) | tests/test_pricing.py hands each path curve to ORE as `ORE.DiscountCurve`; tests/test_ore_xva_parity.py: ORE's own scenario dump rebuilt from each path's state | 1e-12; 1.2e-11 (dump) |
| `CrossAssetModelBuilder` IR calibration (`CalibrationSwaptions`, Bootstrap) | `engine.calibration.cam` | same helpers and bootstrap as below; tests/test_ore_xva_parity.py: ORE's calibrated simulation rebuilt by the engine's model | 1.2e-11 (dump) |
| The cube's `T0`: every trade on the `ScenarioSimMarket` of the as-of date | `engine.pricing.context.simulation_market_today` (since 2026-10-07, I-85) | tests/test_ore_xva_parity.py | 3.9e-11 |
| `Trade::maturity()` (OREData/ored/portfolio/swaption.cpp step 8: a cash-settled option's last exercise date) | `engine.pricing.cube.trade_maturity` (since 2026-10-07, I-86) | tests/test_ore_xva_parity.py (the profiles of ORE's cube) | exact |
| `ValuationEngine::buildCube` (orea/engine/valuationengine.cpp) as `XvaAnalytic` runs it: without model builders, so each trade keeps the model it was built with on the simulation market of the as-of date | `engine.pricing.cube` (since 2026-10-12, I-49; no ORE analytic recalibrates on a path, so the engine does not, A-24) | per trade type, below; tests/test_ore_xva_parity.py | 1.9e-11 (options) |
| `FixingManager::applyFixings` (orea/simulation/fixingmanager.cpp) | `engine.pricing.legs.path_fixings` | tests/test_pricing.py | 1e-12 |
| `DiscountingSwapEngine` with at-par coupons (`IborCouponPricer::initializeCachedData`), `hasOccurred` | `engine.pricing.legs` | tests/test_pricing.py (t=0 and every path/date), tests/test_shared_portfolio.py | 1e-10 |
| `BlackMultiLegOptionEngine` (qle/pricingengines/blackmultilegoptionengine.cpp), incl. `ParYieldCurve` cash settlement; `DynamicSwaptionVolatilityMatrix` on paths | `engine.pricing.european` | OREApp oracle at t=0; QuantLib `BachelierSwaptionEngine` on every path | 1e-10 (t=0), 2e-14 (cash) |
| QuantLib `JamshidianSwaptionEngine` (configured, not ORE's default) | `engine.pricing.jamshidian` | tests/test_jamshidian.py (QuantLib's engine, and its formula at the exact root); tests/test_end_to_end.py on Hull-White paths | Brent's 1e-8 on r* (≤ 5e-6); 1e-10 at the exact root |
| `LgmBuilder` + `IrModelBuilder::buildSwaptionBasket` (OREData/ored/model/), `AnalyticLgmSwaptionEngine`, `NumericLgmMultiLegOptionEngine`; `LgmBuilder::calibrate`'s acceptance (`getCalibrationError` against `Tolerance`, since 2026-10-12, [I-73](../planning/known-issues.md#i-73)) | `engine.pricing.bermudan`, `engine.calibration.lgm` | tests/test_ore_lgm_calibration.py (OREApp, `Calibration=Bootstrap`), tests/test_pricing.py (on ORE's path curves, the model as built), tests/test_calibration_acceptance.py | 2e-11 (t=0), 1e-10 (paths) |
| `OptionWrapper` / `BermudanOptionWrapper` (OREData/ored/portfolio/optionwrapper.cpp), its position (`OptionData/LongShort`: the NPV times -1 for a short, the exercise the holder's) | `engine.pricing.options` (`wrap`, `position_sign`) | tests/test_pricing.py (exercise, physical vs cash); a short option against ORE's trade builder with `LongShort` Short, today (tests/test_shared_portfolio.py) and on ORE's paths (tests/test_ore_xva_parity.py) | rule-level; a short 1e-10 (Europeans), 1e-9 (calibrated) of ORE's |
| `DiscountingRiskyBondEngine` without credit | `engine.pricing.cube.bond_legs` | tests/test_shared_portfolio.py vs `DiscountingBondEngine` | 2e-16 |
| `ExposureCalculator` (orea/aggregation/exposurecalculator.cpp): EPE, ENE, EE_B, EEE_B, PFE, time-weighted EPE_B/EEPE_B, Basel horizon | `engine.risk.counterparty.exposure` | tests/test_ore_xva_parity.py: the engine's profiles of ORE's own cube are ORE's reports, per trade and netting set; tests/test_pipeline.py (a bill's EE_B identity) | 1.8e-13 |
| The assembled pipeline: `XvaAnalytic` (`EXPOSURE`, `PFE`) | `price_portfolio` | tests/test_ore_xva_parity.py: on ORE's paths each trade's cube against ORE's (L3); independent simulations within four standard errors (L4) | every trade 3.0e-11 of its scale; Bermudans/Americans since 2026-10-12 ([I-49](../planning/known-issues.md#i-49)) |
| `SensitivityAnalysis`, `SensitivityCube` (orea/engine/sensitivityanalysis.cpp, orea/cube/sensitivitycube.cpp), its Theta and `recalibrateModels` | `engine.risk.greeks.bump` (Bump), `engine.risk.greeks.ad` (AD, ORE's `UseAD` semantics) | **OREApp run** (tests/test_sensitivity_ore_parity.py, `tests/support/ore_sensitivity_oracle.py`): every Delta, Gamma, Vega and Theta of the shared portfolio, calibrated options recalibrated ([I-51](../planning/known-issues.md#i-51), 2026-10-17); the AD Greeks of its swaps, Europeans and bond as the shift vanishes; tests/test_greeks.py, tests/test_greeks_bermudan.py | 2e-14 of the trade's NPV (Bump); 2e-5 of the curve's largest Delta (AD, O(h³)) |
| `HistoricalPnlGenerator` / `MarketRiskReport` (orea/engine/historicalpnlgenerator.cpp, marketriskreport.cpp): the `ValuationEngine` with the trades' model builders, recalibrating under every scenario | `engine.risk.market` (`revaluation`: a calibrated option's LGM recalibrated per scenario, `price_functions.recalibrated_pnl`) | tests/test_market_risk_ore_parity.py: each scenario of a calibrated Bermudan and American against ORE's LGM engine calibrated on that scenario's curves ([I-112](../planning/known-issues.md#i-112), 2026-10-17); **ORE's `HISTSIM_VAR` analytic end to end** (tests/test_ore_parity_sweep.py, `tests/support/ore_market_risk_oracle.py`, 2026-10-18): every trade's P&L in every Monte Carlo or historical scenario of the seeded sweeps, and ORE's VaR | 7e-11; 5e-13 of the notional (sweeps) |

The whole portfolio at t=0 against ORE, trade by trade, on one sloped market:
tests/test_shared_portfolio.py ([the shared portfolio](../planning/details/ore-parity-validation.md#the-shared-portfolio)), worst case 3.8e-11 (a calibrated Bermudan).

**The assembled pipeline against an ORE simulation** (2026-10-07,
`tests/support/ore_xva_oracle.py`, [details](../planning/details/ore-parity-validation.md#the-ore-simulation-oracle-2026-10-07)):
every trade equals ORE's cube on ORE's own paths, Bermudans and Americans since 2026-10-12
([I-49](../planning/known-issues.md#i-49): ORE's exposure simulation keeps the model it calibrated when it built the
trade, which the engine had recalibrated on every path, 0.5–4% apart), and the exposure equals
ORE's in distribution. Since 2026-10-17 the Greeks equal ORE's sensitivity analytic, run in-process
(L5, [I-51](../planning/known-issues.md#i-51)), and a calibrated Bermudan's market risk ORE recalibrating per
scenario ([I-112](../planning/known-issues.md#i-112)).

### Verification gates

The plan's gates (§4) ask for the ORE source that decides each question and an OREApp run
that shows it. Most were closed from the source plus tests against ORE's Python bindings, not
by an OREApp run; the column says which.

| Gate | Answer | Source | Evidence |
|---|---|---|---|
| V-1 | `ValuationEngine` recalibrates the model builders it is given on each scenario (`recalibrate = true` by default, `recalibrateModels` → `LgmBuilder::recalibrate`), but `XvaAnalytic` gives it none (single-threaded) or `recalibrateModels = false` (multi-threaded), so ORE's exposure simulation never recalibrates; its sensitivity analysis and historical P&L, on the as-of date, do. Non-simulated swaption vols are the t=0 surface seen from the scenario date (`DynamicSwaptionVolatilityMatrix`) | valuationengine.cpp, xvaanalytic.cpp, historicalpnlgenerator.cpp, lgmbuilder.cpp, qle/termstructures/dynamicswaptionvolmatrix.cpp | **OREApp run** (tests/test_ore_xva_parity.py): Europeans on the decayed vols, and calibrated Bermudans and Americans on their as-of models, equal ORE on its paths (closed 2026-10-12, [I-49](../planning/known-issues.md#i-49); ORE's debug log of the run shows no recalibration after the build) |
| V-2 | Absolute shift, `ShiftScheme::Forward`; delta `up − base`, gamma `up − 2·base + down`; scaling by target over actual shift | sensitivitycube.cpp, sensitivityscenariogenerator.cpp | Source; tests/test_sensitivities.py |
| V-3 | `thetaDate = asof + thetaPeriod` (calendar); a new sim market from today's market with the evaluation date moved: its curves hold their tenor points at the tenors' times from the as-of date and take their reference date from the evaluation date (`InterpolatedDiscountCurve2`, `setAdjustReferenceDate(false)`), the simulated ATM vol matrix floats with it, from its reference date, the first business day on or after the Theta date (the reading of 'fixed in dates', the engine's until 2026-10-17, was wrong by 60-290% of Theta; the Theta date's own reference, the engine's until 2026-10-18, by a weekend's variance); fixings backfilled; the period flows ORE adds are read after the roll, where they have occurred, so none (the engine added them until 2026-10-18, I-116) | sensitivityanalysis.cpp, scenariosimmarket.cpp | **OREApp run** (tests/test_sensitivity_ore_parity.py: every trade's Theta to 1e-13 of its NPV, [I-51](../planning/known-issues.md#i-51)); tests/test_sensitivities.py, tests/test_trade_dates.py |
| V-4 | Can ORE's scenario dump reprice the cube? Yes: its numeraire gives each path's LGM state, which rebuilds every dumped curve | crossassetmodelscenariogenerator.cpp, scenariowriter.cpp | **OREApp run**: tests/test_ore_xva_parity.py, 1.2e-11 (2026-10-07) |
| V-5 | Co-terminal basket from the trade's exercise dates; `CoterminalDealStrike` (first fixed rate less spread) with the ±3 std-dev fallback, or ATM; `ReferenceCalibrationGrid` keeps one helper per interval; an American's expiries are the grid dates in its window | lgmbuilder.cpp, irmodelbuilder.cpp | **OREApp run**: tests/test_ore_lgm_calibration.py, calibrated price = ORE's to 2e-11 |
| V-6 | `CalibrationSwaptions`: tenor-based expiries and terms, ATM | crossassetmodelbuilder.cpp | **OREApp run**: tests/test_ore_xva_parity.py (ORE's calibrated simulation rebuilt by the engine's calibration, both models) |
| V-7 | Drift and covariance of every IR/FX/EQ block under the LGM measure | crossassetanalytics.hpp | Bindings: tests/test_cam.py (analytic martingales exact; covariance = loading integral) |
| V-8 | `includeReferenceDateEvents = false`: a flow paid on the valuation date has occurred | cashflow.cpp, valuationengine.cpp | Bindings: tests/test_pricing.py |
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

**This engine:** `engine/market_simulation/sobol.py::generate_sobol_normals`, via
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

**This engine:** `engine/market_simulation/sobol.py::_build_bridge_matrix`,
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

**Verified:** `tests/test_sobol.py::TestBrownianBridge` (the resulting
matrix reproduces the exact covariance structure `min(s,t)` real Brownian motion has —
the property this construction exists to guarantee) and
`tests/test_ore_parity.py::TestBrownianBridgeParity` (below).

## 3. Interest rate model: the LGM, in two parametrizations

**This engine:** `engine/models/cam.py` (the components and the exact step moments),
`engine/market_simulation/scenario_market.py` (the model-implied curves and numeraire),
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
r(t) = f(0,t) + H′(t)x + ζ(t)H(t)H′(t). (Until 2026-10-01 the Hull-White model simulated an
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

**This engine:** `engine/models/cam.py::flexible_cholesky` of each step's covariance
(`step_moments`).

**ORE:** `QuantExt::CrossAssetModel` keeps a `correlation()` matrix (unit diagonal) separate
from each factor's volatility parameter (`qle/models/crossassetmodel.cpp`), and its state
process takes the square root of each step's covariance with `pseudoSqrt`, which with the
CAM's default `SalvagingAlgorithm::None` is `CholeskyDecomposition(cov, flexible = true)`.
`flexible_cholesky` reproduces it, including a zero-volatility factor's zero column
(tests/test_cam.py).

## 5. Vanilla interest rate swap pricing

**This engine:** `engine/pricing/legs.py` (`legs_npv`, `today_npv`, `legs_cube`).

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
`engine.instruments.schedules.par_coupon_forecast_period` returns the three from QuantLib's own
coupon. Until 2026-09-29 the engine divided by the leg's accrual instead, which differs for
any leg day count other than the index's ([I-36](../planning/known-issues.md#i-36)).

**Verified:** `tests/test_swap.py::TestAgainstORE` (direct NPV comparison against a real
`ORE.VanillaSwap` + `ORE.DiscountingSwapEngine`, 1e-12, across payer/receiver/par/spread
cases) and `tests/test_pricing.py` (today and on every path and date).

## 6. European swaption pricing: Jamshidian's decomposition

**The configured alternative.** ORE's default European engine is `BlackMultiLegOptionEngine`
(Bachelier on the market volatility, `engine.pricing.european`, [above](#the-pipeline)).
This section documents the Jamshidian engine (`PricingConfig.european="Jamshidian"` with its
Hull-White model), which matches QuantLib's `JamshidianSwaptionEngine` and refuses what it
refuses: a floating spread, `a <= 0`, cash settlement ([I-37](../planning/known-issues.md#i-37),
[I-41](../planning/known-issues.md#i-41), [I-52](../planning/known-issues.md#i-52)).

**This engine:** `engine/pricing/jamshidian.py` (`jamshidian_npv`, `_solve_decreasing_root`),
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

**This engine:** `engine/pricing/lgm_grid.py` — `_hagan_quadrature_weights`,
`_rollback_operator` (the convolution as one matrix per step), `_basis` and `_Coefficients`
(each cashflow's deflated value as a coefficient times a function of the state), `_induction`
(the loop, on a level and on each path's deviation, 2026-10-13) — with the LGM's bond price in
`engine/models/lgm.py`, plus
`engine/instruments/american_swaption.py`, which supplies ORE's American option times and
exercise style to this same engine; `engine/pricing/bermudan.py` calibrates each trade's
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
  `ShiftHorizon` 0 or ORE's default 0.5, or 1.5, each case at both, I-32; a piecewise `Sigma` as
  `VolatilityTimes`/`Volatility`).

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

**This engine:** `engine/risk/market/var_es.py::value_at_risk`,
`expected_shortfall`.

**ORE:** `QuantLib::GenericRiskStatistics<GaussianStatistics>::valueAtRisk`,
`::expectedShortfall` (the `QuantLib::RiskStatistics` typedef) —
[`QuantLib/ql/math/statistics/riskstatistics.hpp`](../../reference/ORE/QuantLib/ql/math/statistics/riskstatistics.hpp),
lines 178-205 — built on `QuantLib::GeneralStatistics::percentile` —
[`QuantLib/ql/math/statistics/generalstatistics.cpp`](../../reference/ORE/QuantLib/ql/math/statistics/generalstatistics.cpp),
lines 88-110.

**Correspondence: confirmed exactly, formula and edge cases both.**
`GeneralStatistics::percentile(percent)` sorts the (weight, value) sample ascending, then
walks forward while the cumulative weight is **strictly below** `target = percent *
totalWeight`, returning the sample it stops on. With unit weights the cumulative weight at
index `k` is `k + 1`, so it stops at `k = ceil(target) - 1`, clamped to the sample; this
engine's `value_at_risk` reads `sorted_pnl[var_rank(N, p)]`, `var_rank` being that index with
`target = (1 - p) * N` computed in double precision as QuantLib computes it. Where the target
is not a whole number the index is `floor(target)`, which the engine took until 2026-10-11;
where it is (16 scenarios at 0.9375), QuantLib's is the observation below, and so the engine's
now is ([I-102](../planning/known-issues.md#i-102), closed). Confirmed against
`ORE.RiskStatistics.valueAtRisk` directly in `tests/test_var_es.py`: the sweep of sizes and
levels, and `TestExactRankBoundaryAgainstORE` at exact boundaries (dyadic levels, ties, a
target weight of two, and `var_rank` against ORE's index at every count from 1 to 299). ORE
requires a confidence level in [0.9, 1); the engine accepts any in (0, 1) on the same rule.

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

**ORE's historical simulation VaR analytic** (`HISTSIM_VAR`,
OREAnalytics/orea/engine/historicalsimulationvar.cpp) has an estimator of its own: its quantile
is boost's right tail quantile of the P&L (`HistoricalSimulationVarCalculator::var`), the same
scenario as `RiskStatistics`' wherever `N p` is not a whole number (the `floor(N p) + 1`-th
smallest P&L), and undefined (NaN) with fewer than `1 / p` scenarios, where `RiskStatistics` and
the engine take the worst P&L. At a whole `N p` the report reads the `(N p + 1)`-th smallest
always, `RiskStatistics` the `ceil(N (1 - c))`-th, which is the one below where `1 - c` rounds
low in binary (c = 0.8, 0.9: 90% of ten moves, ORE's second worst, `RiskStatistics`' the worst)
and the same where it rounds high (0.95, 0.975), measured 2026-10-18. The report's quantile is
the P&L itself, so a tail of gains reports a negative VaR where `RiskStatistics` floors it at
zero. Its expected shortfall averages the P&L **at or below** that quantile, the VaR scenario
included, where `RiskStatistics` averages strictly below. The engine's figures are `RiskStatistics`' (decision X-12); against ORE's own analytic run
in-process the VaR is the report's and every trade's P&L ORE's
(`tests/test_ore_parity_sweep.py::test_market_risk_is_ores`, and at ORE's default option
engine `tests/test_ore_parity_settings.py::test_market_risk_of_options_at_ores_default_engine_is_ores`),
the whole-`N p` difference held by `test_at_a_whole_n_p_the_var_is_risk_statistics_not_ores_report`.

**Verified:** `tests/test_var_es.py` (all classes; direct `ORE.RiskStatistics`
comparison, including the tie-at-VaR-boundary and empty-tail edge cases);
`tests/test_ore_parity_sweep.py` (ORE's `HISTSIM_VAR` analytic end to end).

## 9. Delta, Gamma, Vega, and Theta

**This engine:** `engine/risk/greeks/bump.py` (`Bump`, the default: ORE's sensitivity
analysis, and Theta for both methods), `engine/risk/greeks/ad.py` (`AD`). See
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
recalibrated under every shift. `portfolio_sensitivities` reproduces it (gates V-2, V-3), and
equals an OREApp sensitivity run to 2e-14 of each trade's NPV
([I-51](../planning/known-issues.md#i-51), tests/test_sensitivity_ore_parity.py). One engine curve is two of
ORE's factors there: ORE's LGM builder discounts its model and its helpers on the swap index's
discounting index, and a bond on its reference curve, each an `IndexCurve` factor beside the
`DiscountCurve`, both from the engine's one discount curve; the oracle shifts both and reads
the joint move from ORE's cross gammas (its docstring has the algebra).

**AD: the same quantity's limit.** The exact derivative of each trade's price on ORE's
sensitivity market in each tenor's zero rate and each quote, scaled by the shift — what ORE's own
AD sensitivities are (`UseAD` in a `SensitivityDelta` run: the engine's gradient on the sensitivity
market, each shift scenario's NPV the base plus the gradient times its move,
[`OREData/ored/scripting/engines/amccgbaseengine.cpp`](../../reference/ORE/OREData/ored/scripting/engines/amccgbaseengine.cpp)),
and ORE's Delta without the forward difference's truncation, as ORE's closed-form
`DiscountingSwapEngineDeltaGamma`/`BlackSwaptionEngineDeltaGamma` engines compute it on their
bucket grid
([`QuantExt/qle/pricingengines/discountingswapenginedeltagamma.hpp`](../../reference/ORE/QuantExt/qle/pricingengines/discountingswapenginedeltagamma.hpp),
cross-checked by ORE in
[`OREAnalytics/test/sensitivityvsanalytic.cpp`](../../reference/ORE/OREAnalytics/test/sensitivityvsanalytic.cpp)).
Vega per swaption quote: through the bilinear surface's weights, and for a
Bermudan/American through its calibration by the implicit function theorem, since
`RiskFactorKey::KeyType` has no entry for a raw model parameter — ORE's Vega is always on the
quotes the model is calibrated to. A calibrated option calibrates inside its price function, so
its Delta and Gamma recalibrate as ORE's bumps do. For swaps, Europeans and bonds the AD Greeks
are ORE's as the shift vanishes (2e-5, against ORE's report); a Bermudan's/American's grid price
is piecewise smooth in the rates, so its AD Greeks are the price's derivatives, not ORE's 1bp
secants across the grid's kinks (Delta within 0.4%, Gamma not comparable), and an AD run says so
([I-78](../planning/known-issues.md#i-78)).

**Rho:** confirmed absent from ORE entirely — `QuantExt::RiskFactorKey::KeyType`
([`QuantExt/qle/termstructures/scenario.hpp`](../../reference/ORE/QuantExt/qle/termstructures/scenario.hpp))
has no rho-specific entry, so the interest-rate-curve Delta is ORE's equivalent of a
textbook Rho.

**Theta:** `generateSensitivities`'s theta branch advances the evaluation date by a
configured period (calendar days, [I-38](../planning/known-issues.md#i-38)), builds a new
simulation market from today's, reprices, and means to add back the interim cashflows,
`Theta = NPV(t+dt) − NPV(t) + CF(t, t+dt]`; but it reads them with `aggregateTradeFlow` from the
trade's cash flows on the Theta date without its past flows (`Trade::cashflows(...,
includePastCashflows = false)`), where every flow of the roll has occurred
(`CashFlow::hasOccurred`, `includeTodaysCashFlows` unset), so the sum is empty and ORE's Theta
is `NPV(t+dt) − NPV(t)`: a coupon paid in the roll leaves Theta as it leaves the NPV. The
simulation market's curves hold their tenor points at the tenors' times from the as-of date and
take their reference date from the evaluation date (`InterpolatedDiscountCurve2`,
`setAdjustReferenceDate(false)`), so on the Theta date every curve is the base sensitivity
market's as a function of the time from its reference date; the simulated ATM volatility matrix
(`SwaptionVolatilityMatrix` on a calendar, no settlement days) floats to the first business day
on or after the Theta date, where each option's variance starts
(`engine.pricing.context.floating_volatility_date`). Both methods report it (`trade_theta`),
equal to ORE's sensitivity analytic; the engine's curves fixed in dates were 60–290% off ORE's
Theta until 2026-10-17 ([I-51](../planning/known-issues.md#i-51)), and its add-back of the
roll's flows ([I-39](../planning/known-issues.md#i-39)) and the Theta date's own volatility
reference until 2026-10-18, both found by the seeded sweeps
([I-116](../planning/known-issues.md#i-116)).

**Root-finds.** Naively differentiating through a root solver gives a silently wrong
gradient (a bisection's comparisons have none, Newton's iterations only the steps'). The
Jamshidian root, each helper's y* and a whole calibration get the implicit function theorem's
derivative instead (`engine.solvers.roots.implicit_root`'s `custom_jvp`, and each helper's equation
in its own zeta, `engine.calibration.lgm.calibrated_values`); see [Greeks](../risk/greeks.md#differentiating-through-bisection-root-finds).

**Verified:** `tests/test_sensitivity_ore_parity.py` (both methods against an OREApp sensitivity
run), `tests/test_ore_parity_sweep.py` and `tests/test_ore_parity_settings.py` (random portfolios,
markets, settings and Theta rolls against it), `tests/test_sensitivities.py`,
`tests/test_trade_dates.py` (Theta's rules),
`tests/test_greeks.py` and `tests/test_greeks_bermudan.py` (AD against finite differences
through the recalibration, agreement with the bump method).

## 10. LGM calibration: bootstrap fit of a piecewise sigma to market swaption vols

**This engine:** `engine/calibration/lgm.py` (`build_basket`, `market_price`, `price_pair`,
`bootstrap_sigma`), every LGM calibration the engine makes: each currency's of the cross-asset
model (`engine/calibration/cam.py`) and each Bermudan's/American's
(`engine/pricing/bermudan.py::calibration_basket`).

**ORE:** `ore::data::IrModelBuilder::buildSwaptionBasket()`
(`OREData/ored/model/irmodelbuilder.cpp`) for the basket; `ore::data::LgmBuilder::calibrate()`
(`OREData/ored/model/lgmbuilder.cpp`, lines 209-212, `calibrateVolatilitiesIterative`) for
the bootstrap itself, which calls `QuantLib::CalibratedModel::calibrateIterative`;
`QuantExt::AnalyticLgmSwaptionEngine` (`QuantExt/qle/pricingengines/`) with
`FloatSpreadMapping::proRata` for each helper's model value.

**Full writeup in [Calibration](calibration.md)** — the `aTimes = swaptionExpiries[:-1]`
triangular-bootstrap construction, why mean reversion is never calibrated, the helpers'
values, ORE's acceptance, the x* gradient bug (cross-referenced above in section 9), and the
root solver every calibration uses ([decision A-21](../../compliance/decisions.md): Newton by
default, the bisection kept as the reference; each reaches ORE's root more exactly than ORE's
own solvers).

**`AnalyticLgmSwaptionEngine` has no Python constructor** (`ORE-SWIG/QuantExt-SWIG/SWIG/
qle_pricingengines.i` declares only `enableCache`/`clearCache`/`setZetaShift`/`resetZetaShift`),
so a helper's model value is checked through the end-to-end NPV: a Bermudan calibrated to its
co-terminal basket and priced on the grid equals ORE's `LGMGridSwaptionEngineBuilder` with
`Calibration=Bootstrap`, where any error in it would move the calibrated volatility and the
price; the helpers' market values are checked against QuantLib's own
`SwaptionHelper.marketValue()` directly.

**The parametrizations.** Section 3's ["parametrization note"](#a-parametrization-note-lgm-vs-plain-hull-white)
applies here: the calibration is of the LGM ORE's Bermudan engine uses. A Hull-White currency
of the cross-asset model is bootstrapped to the same helpers and converted bucket by bucket
(`hull_white_matching_zeta`): helper prices depend on the model only through ζ at their
expiry and H, so matching ζ at every bucket end is exact (`tests/test_hull_white_model.py::TestCalibration`).

Until 2026-10-18 a second, standalone bootstrap (`calibrate_lgm_sigma`, `POST /calibration/lgm`)
was checked only by re-derivation (its formula pieces against `ORE.LinearGaussMarkovModel` and
its price against a numeraire-deflated Monte Carlo); it is retired (I-94), and the run request's
`calibration` analytic is ORE's bootstrap.

**Verified:** `tests/test_ore_lgm_calibration.py` (2e-11 against `LgmBuilder`, through the
OREApp oracle), `tests/test_calibration_acceptance.py` (ORE's acceptance, I-73),
`tests/test_ore_xva_parity.py` (the cross-asset model's calibration through ORE's own
simulation) — see [Calibration: Tested by](calibration.md#tested-by) for the full breakdown.

## The parity matrix

What is compared with ORE, product by analytic by setting, and what is not, with the reason
([I-116](../planning/known-issues.md#i-116), closed 2026-10-18). Each parity test declares the
cells it compares (an `ore_parity` marker, derived from the very trades and configurations it
runs: tests/support/ore_parity_matrix.py); tests/test_ore_parity_matrix.py fails on a public
setting, a value of one or an analytic with neither a test nor a reason, on a field of a request
that is neither a setting nor an input, and on this table when it is not what the tests declare.
The seeded sweeps (tests/test_ore_parity_sweep.py, cases from tests/support/parity_sweep.py:
random markets, as-of dates, trades and settings, each case rebuilt from its seed) declare what
each case draws; the fast tier runs their fixed sample, the full tier the whole sweep. Trade
terms (notionals, strikes, dates) and market data are inputs the sweeps draw, not rows.

<!-- ore-parity-matrix:begin -->

*Generated by `python -m tests.support.ore_parity_matrix --write` from the parity tests' `ore_parity` markers; do not edit by hand. tests/test_ore_parity_matrix.py fails if it is stale, or if a public setting or analytic has neither a test nor a reason.*

Each cell names the tests that compare it with ORE (**T**n, listed below with their tolerances; *full* where only the full tier's sweep reaches it), or the reason it is not compared (**R**n); · where the setting does not apply to the product. The first row of each table is the product compared at all.

#### Today's value (`npv`)

| Setting | Value | Swap | European | Bermudan | American | Bond |
|---|---|---|---|---|---|---|
| *compared* |  | T21, T36, T40, T41, T43 | T2, T3, T4, T21, T32, T39, T40 | T9, T10, T11, T12, T21, T34, T39, T40, T42 | T10, T11, T13, T21, T34, T39, T40, T42 | T21, T40 |
| `SwapConfig.payer` | payer | T21, T36, T40, T41, T43 | · | · | · | · |
| `SwapConfig.payer` | receiver | T21, T36, T40, T43 | · | · | · | · |
| `SwapConfig.index_tenor_months` | = 6 | T21, T36, T40, T41, T43 | · | · | · | · |
| `SwapConfig.index_tenor_months` | ≠ 6 | T21 | · | · | · | · |
| `SwapConfig.floating_spread` | = 0.0 | T21, T36, T40, T41, T43 | · | · | · | · |
| `SwapConfig.floating_spread` | ≠ 0.0 | T21, T36 | · | · | · | · |
| `SwapConfig.currency` | = USD | T21, T36, T40, T41, T43 | · | · | · | · |
| `SwapConfig.currency` | ≠ USD | R1 | · | · | · | · |
| `SwapConfig.fixings` | none | T21, T36, T40 | · | · | · | · |
| `SwapConfig.fixings` | past fixings | T21, T36, T40, T41, T43 | · | · | · | · |
| `SwapConfig.accrual_day_count` | ACT/365 | T21, T36, T40, T41, T43 | · | · | · | · |
| `SwapConfig.accrual_day_count` | ACT/ACT (ICMA) | T21, T36, T40, T41 | · | · | · | · |
| `SwaptionConfig.payer` | payer | · | T2, T3, T4, T21, T32, T39, T40 | · | · | · |
| `SwaptionConfig.payer` | receiver | · | T2, T3, T4, T21, T32, T39, T40 | · | · | · |
| `SwaptionConfig.index_tenor_months` | = 6 | · | T2, T3, T4, T21, T32, T39, T40 | · | · | · |
| `SwaptionConfig.index_tenor_months` | ≠ 6 | · | T21 | · | · | · |
| `SwaptionConfig.floating_spread` | = 0.0 | · | T2, T3, T4, T21, T32, T39, T40 | · | · | · |
| `SwaptionConfig.floating_spread` | ≠ 0.0 | · | T21, T32, T39, T40 | · | · | · |
| `SwaptionConfig.currency` | = USD | · | T2, T3, T4, T21, T32, T39, T40 | · | · | · |
| `SwaptionConfig.currency` | ≠ USD | · | R1 | · | · | · |
| `SwaptionConfig.settlement` | Physical | · | T2, T3, T4, T21, T32, T39, T40 | · | · | · |
| `SwaptionConfig.settlement` | Cash | · | T21, T39, T40 | · | · | · |
| `SwaptionConfig.long_short` | Long | · | T2, T3, T4, T21, T32, T40 | · | · | · |
| `SwaptionConfig.long_short` | Short | · | T21, T39 | · | · | · |
| `BermudanSwaptionConfig.payer` | payer | · | · | T9, T11, T12, T21, T34, T39, T40, T42 | · | · |
| `BermudanSwaptionConfig.payer` | receiver | · | · | T10, T11, T21, T34, T39, T40, T42 | · | · |
| `BermudanSwaptionConfig.index_tenor_months` | = 6 | · | · | T9, T10, T11, T12, T21, T34, T39, T40, T42 | · | · |
| `BermudanSwaptionConfig.index_tenor_months` | ≠ 6 | · | · | T21 | · | · |
| `BermudanSwaptionConfig.floating_spread` | = 0.0 | · | · | T9, T10, T11, T12, T21, T34, T39, T40, T42 | · | · |
| `BermudanSwaptionConfig.floating_spread` | ≠ 0.0 | · | · | T21, T34 | · | · |
| `BermudanSwaptionConfig.currency` | = USD | · | · | T9, T10, T11, T12, T21, T34, T39, T40, T42 | · | · |
| `BermudanSwaptionConfig.currency` | ≠ USD | · | · | R1 | · | · |
| `BermudanSwaptionConfig.fixings` | none | · | · | T9, T10, T11, T12, T21, T34, T39, T40 | · | · |
| `BermudanSwaptionConfig.fixings` | past fixings | · | · | T21, T42 | · | · |
| `BermudanSwaptionConfig.settlement` | Physical | · | · | T9, T10, T11, T12, T21, T34, T39, T40, T42 | · | · |
| `BermudanSwaptionConfig.settlement` | Cash | · | · | T21, T39, T40 | · | · |
| `BermudanSwaptionConfig.long_short` | Long | · | · | T9, T10, T11, T12, T21, T34, T40, T42 | · | · |
| `BermudanSwaptionConfig.long_short` | Short | · | · | T21, T39 | · | · |
| `AmericanSwaptionConfig.payer` | payer | · | · | · | T10, T11, T13, T21, T34, T39, T40, T42 | · |
| `AmericanSwaptionConfig.payer` | receiver | · | · | · | T11, T21, T42 | · |
| `AmericanSwaptionConfig.index_tenor_months` | = 6 | · | · | · | T10, T11, T13, T21, T34, T39, T40, T42 | · |
| `AmericanSwaptionConfig.index_tenor_months` | ≠ 6 | · | · | · | T21 | · |
| `AmericanSwaptionConfig.floating_spread` | = 0.0 | · | · | · | T10, T11, T13, T21, T34, T39, T40, T42 | · |
| `AmericanSwaptionConfig.floating_spread` | ≠ 0.0 | · | · | · | T21 | · |
| `AmericanSwaptionConfig.currency` | = USD | · | · | · | T10, T11, T13, T21, T34, T39, T40, T42 | · |
| `AmericanSwaptionConfig.currency` | ≠ USD | · | · | · | R1 | · |
| `AmericanSwaptionConfig.fixings` | none | · | · | · | T10, T11, T21, T34, T39, T40, T42 | · |
| `AmericanSwaptionConfig.fixings` | past fixings | · | · | · | T13, T21, T42 | · |
| `AmericanSwaptionConfig.settlement` | Physical | · | · | · | T10, T11, T13, T21, T34, T39, T40, T42 | · |
| `AmericanSwaptionConfig.settlement` | Cash | · | · | · | T21 | · |
| `AmericanSwaptionConfig.long_short` | Long | · | · | · | T10, T11, T13, T21, T34, T40, T42 | · |
| `AmericanSwaptionConfig.long_short` | Short | · | · | · | T21, T39 | · |
| `BondConfig.currency` | = USD | · | · | · | · | T21, T40 |
| `BondConfig.currency` | ≠ USD | · | · | · | · | R1 |
| `BondConfig.accrual_day_count` | ACT/365 | · | · | · | · | T21 |
| `BondConfig.accrual_day_count` | ACT/ACT (ICMA) | · | · | · | · | T21, T40 |
| `BondConfig.coupon_schedule` | coupons | · | · | · | · | T21, T40 |
| `BondConfig.coupon_schedule` | none (a bill) | · | · | · | · | T21 |
| `BondConfig.redemption_fraction` | = 1.0 | · | · | · | · | T21, T40 |
| `BondConfig.redemption_fraction` | ≠ 1.0 | · | · | · | · | R2 |
| `BondConfig.coupon_schedule.payment_date` | at each period's end | · | · | · | · | T21, T40 |
| `BondConfig.coupon_schedule.payment_date` | delayed | · | · | · | · | R3 |
| `LgmSwaptionEngineConfig.calibration` | Bootstrap | · | · | T9, T12, T21, T34, T39, T40 | T13, T21, T34, T39, T40 | · |
| `LgmSwaptionEngineConfig.calibration` | None | · | · | T10, T11, T21, T42 | T10, T11, T21, T42 | · |
| `LgmSwaptionEngineConfig.strategy` | CoterminalDealStrike | · | · | T9, T12, T21, T34, T39, T40 | T13, T21, T34, T39, T40 | · |
| `LgmSwaptionEngineConfig.strategy` | CoterminalATM | · | · | T9, T21 | T21 | · |
| `LgmSwaptionEngineConfig.bootstrap_tolerance` | = 0.0001 | · | · | T12, T21, T34, T39, T40 | T13, T21, T34, T39, T40 | · |
| `LgmSwaptionEngineConfig.bootstrap_tolerance` | ≠ 0.0001 | · | · | T9, T21 | T21 | · |
| `LgmSwaptionEngineConfig.reversion` | = 0.0 | · | · | T12, T21, T34, T39, T40 | T13, T21, T34, T39, T40 | · |
| `LgmSwaptionEngineConfig.reversion` | ≠ 0.0 | · | · | T9, T10, T11, T21, T42 | T10, T11, T21, T42 | · |
| `LgmSwaptionEngineConfig.volatility` | = 0.01 | · | · | T9, T10, T11, T12, T34, T39, T40, T42 | T10, T11, T13, T34, T39, T40, T42 | · |
| `LgmSwaptionEngineConfig.volatility` | ≠ 0.01 | · | · | T21 | T21 | · |
| `LgmSwaptionEngineConfig.reference_calibration_grid` | = 400,3M | · | · | T9, T10, T11, T12, T21, T34, T39, T40, T42 | T10, T11, T21, T34, T39, T40, T42 | · |
| `LgmSwaptionEngineConfig.reference_calibration_grid` | ≠ 400,3M | · | · | T21 | T13, T21 | · |
| `LgmSwaptionEngineConfig.shift_horizon` | = 0.5 | · | · | T9, T11, T12, T21, T34, T39, T40 | T11, T13, T21, T34, T39, T40 | · |
| `LgmSwaptionEngineConfig.shift_horizon` | ≠ 0.5 | · | · | T9, T10, T11, T21, T42 | T10, T11, T21, T42 | · |
| `LgmSwaptionEngineConfig.n_per_std` | = 30 | · | · | T34, T39, T40 | T34, T39, T40 | · |
| `LgmSwaptionEngineConfig.n_per_std` | ≠ 30 | · | · | T9, T10, T11, T12, T21, T42 | T10, T11, T13, T21, T42 | · |
| `LgmSwaptionEngineConfig.std_devs` | = 5.0 | · | · | T21, T34, T39, T40 | T21, T34, T39, T40 | · |
| `LgmSwaptionEngineConfig.std_devs` | ≠ 5.0 | · | · | T9, T10, T11, T12, T21, T42 | T10, T11, T13, T21, T42 | · |
| `LgmSwaptionEngineConfig.exercise_time_steps_per_year` | = 24 | · | · | · | T10, T11, T13, T21, T34, T39, T40, T42 | · |
| `LgmSwaptionEngineConfig.exercise_time_steps_per_year` | ≠ 24 | · | · | · | T21 | · |
| `LgmSwaptionEngineConfig.swap_index.fixed_tenor` | = 1Y | · | · | T9, T12, T21, T34, T39, T40 | T13, T21, T34, T39, T40 | · |
| `LgmSwaptionEngineConfig.swap_index.fixed_tenor` | ≠ 1Y | · | · | T21 | T21 | · |
| `LgmSwaptionEngineConfig.swap_index.fixed_day_counter` | = ACT/365 | · | · | T9, T12, T21, T34, T39, T40 | T13, T21, T34, T39, T40 | · |
| `LgmSwaptionEngineConfig.swap_index.fixed_day_counter` | ≠ ACT/365 | · | · | T12, T21 | T21 | · |
| `LgmSwaptionEngineConfig.swap_index.index_tenor_months` | = 6 | · | · | T9, T12, T21, T34, T39, T40 | T13, T21, T34, T39, T40 | · |
| `LgmSwaptionEngineConfig.swap_index.index_tenor_months` | ≠ 6 | · | · | T21 | T21 | · |
| `LgmSwaptionEngineConfig.solver` | Newton | · | · | T9, T10, T11, T12, T21, T34, T39, T40, T42 | T10, T11, T13, T21, T34, T39, T40, T42 | · |
| `LgmSwaptionEngineConfig.solver` | Bisection | · | · | T21 | T21 | · |
| `PricingConfig.european` | Bachelier | · | T21, T32, T39, T40 | · | · | · |
| `PricingConfig.european` | Jamshidian | · | T2, T3, T4 | · | · | · |
| `JamshidianEngineConfig.reversion` | any | · | T2, T3, T4 | · | · | · |
| `JamshidianEngineConfig.volatility` | any | · | T2, T3, T4 | · | · | · |
| `JamshidianEngineConfig.solver` | Newton | · | T2, T3, T4 | · | · | · |
| `JamshidianEngineConfig.solver` | Bisection | · | R4 | · | · | · |
| `RunConfig.base_currency` | USD | T21, T36, T40, T41, T43 | T2, T3, T4, T21, T32, T39, T40 | T9, T10, T11, T12, T21, T34, T39, T40, T42 | T10, T11, T13, T21, T34, T39, T40, T42 | T21, T40 |
| `RunConfig.base_currency` | another currency | R1 | R1 | R1 | R1 | R1 |

#### Exposure: the cube on ORE's paths, the profiles, the simulation in distribution (`exposure`)

| Setting | Value | Swap | European | Bermudan | American | Bond |
|---|---|---|---|---|---|---|
| *compared* |  | T1, T24, T25, T26, T27, T28, T29, T33 | T1, T22, T24, T25, T26, T27, T28, T29, T31 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T24, T25, T26, T27, T28, T29 |
| `SwapConfig.payer` | payer | T1, T24, T26, T27, T28, T29, T33 | · | · | · | · |
| `SwapConfig.payer` | receiver | T24, T25, T26, T27, T28, T29, T33 | · | · | · | · |
| `SwapConfig.index_tenor_months` | = 6 | T1, T24, T25, T26, T27, T28, T29, T33 | · | · | · | · |
| `SwapConfig.index_tenor_months` | ≠ 6 | T25 | · | · | · | · |
| `SwapConfig.floating_spread` | = 0.0 | T1, T24, T26, T27, T28, T29, T33 | · | · | · | · |
| `SwapConfig.floating_spread` | ≠ 0.0 | T25, T33 | · | · | · | · |
| `SwapConfig.currency` | = USD | T1, T24, T25, T26, T27, T28, T29, T33 | · | · | · | · |
| `SwapConfig.currency` | ≠ USD | R1 | · | · | · | · |
| `SwapConfig.fixings` | none | T1, T24, T25, T26, T27, T28, T29, T33 | · | · | · | · |
| `SwapConfig.fixings` | past fixings | T24, T26, T27, T28, T29, T33 | · | · | · | · |
| `SwapConfig.accrual_day_count` | ACT/365 | T1, T24, T25, T26, T27, T28, T29, T33 | · | · | · | · |
| `SwapConfig.accrual_day_count` | ACT/ACT (ICMA) | T24, T26, T27, T28, T29, T33 | · | · | · | · |
| `SwaptionConfig.payer` | payer | · | T1, T22, T24, T26, T27, T28, T29, T31 | · | · | · |
| `SwaptionConfig.payer` | receiver | · | T1, T22, T24, T25, T26, T27, T28, T29, T31 | · | · | · |
| `SwaptionConfig.index_tenor_months` | = 6 | · | T1, T22, T24, T25, T26, T27, T28, T29, T31 | · | · | · |
| `SwaptionConfig.index_tenor_months` | ≠ 6 | · | T25 | · | · | · |
| `SwaptionConfig.floating_spread` | = 0.0 | · | T1, T22, T24, T26, T27, T28, T29, T31 | · | · | · |
| `SwaptionConfig.floating_spread` | ≠ 0.0 | · | T22, T24, T25, T26, T27, T28, T29, T31 | · | · | · |
| `SwaptionConfig.currency` | = USD | · | T1, T22, T24, T25, T26, T27, T28, T29, T31 | · | · | · |
| `SwaptionConfig.currency` | ≠ USD | · | R1 | · | · | · |
| `SwaptionConfig.settlement` | Physical | · | T1, T22, T24, T25, T26, T27, T28, T29, T31 | · | · | · |
| `SwaptionConfig.settlement` | Cash | · | T22, T24, T26, T27, T28, T29, T31 | · | · | · |
| `SwaptionConfig.long_short` | Long | · | T1, T24, T25, T26, T27, T28, T29, T31 | · | · | · |
| `SwaptionConfig.long_short` | Short | · | T22 | · | · | · |
| `BermudanSwaptionConfig.payer` | payer | · | · | T22, T23, T24, T26, T27, T28, T29, T35 | · | · |
| `BermudanSwaptionConfig.payer` | receiver | · | · | T22, T23, T24, T25, T26, T27, T28, T29 | · | · |
| `BermudanSwaptionConfig.index_tenor_months` | = 6 | · | · | T22, T23, T24, T25, T26, T27, T28, T29, T35 | · | · |
| `BermudanSwaptionConfig.index_tenor_months` | ≠ 6 | · | · | T25 | · | · |
| `BermudanSwaptionConfig.floating_spread` | = 0.0 | · | · | T22, T23, T24, T26, T27, T28, T29, T35 | · | · |
| `BermudanSwaptionConfig.floating_spread` | ≠ 0.0 | · | · | T25 | · | · |
| `BermudanSwaptionConfig.currency` | = USD | · | · | T22, T23, T24, T25, T26, T27, T28, T29, T35 | · | · |
| `BermudanSwaptionConfig.currency` | ≠ USD | · | · | R1 | · | · |
| `BermudanSwaptionConfig.fixings` | none | · | · | T22, T23, T24, T26, T27, T28, T29, T35 | · | · |
| `BermudanSwaptionConfig.fixings` | past fixings | · | · | T25 | · | · |
| `BermudanSwaptionConfig.settlement` | Physical | · | · | T22, T23, T24, T25, T26, T27, T28, T29, T35 | · | · |
| `BermudanSwaptionConfig.settlement` | Cash | · | · | T22, T23, T24, T26, T27, T28, T29 | · | · |
| `BermudanSwaptionConfig.long_short` | Long | · | · | T23, T24, T25, T26, T27, T28, T29, T35 | · | · |
| `BermudanSwaptionConfig.long_short` | Short | · | · | T22 | · | · |
| `AmericanSwaptionConfig.payer` | payer | · | · | · | T22, T23, T24, T26, T27, T28, T29, T35 | · |
| `AmericanSwaptionConfig.payer` | receiver | · | · | · | T25 | · |
| `AmericanSwaptionConfig.index_tenor_months` | = 6 | · | · | · | T22, T23, T24, T25, T26, T27, T28, T29, T35 | · |
| `AmericanSwaptionConfig.index_tenor_months` | ≠ 6 | · | · | · | T25 | · |
| `AmericanSwaptionConfig.floating_spread` | = 0.0 | · | · | · | T22, T23, T24, T26, T27, T28, T29, T35 | · |
| `AmericanSwaptionConfig.floating_spread` | ≠ 0.0 | · | · | · | T25 | · |
| `AmericanSwaptionConfig.currency` | = USD | · | · | · | T22, T23, T24, T25, T26, T27, T28, T29, T35 | · |
| `AmericanSwaptionConfig.currency` | ≠ USD | · | · | · | R1 | · |
| `AmericanSwaptionConfig.fixings` | none | · | · | · | T22, T23, T24, T26, T27, T28, T29, T35 | · |
| `AmericanSwaptionConfig.fixings` | past fixings | · | · | · | T25 | · |
| `AmericanSwaptionConfig.settlement` | Physical | · | · | · | T22, T23, T24, T26, T27, T28, T29, T35 | · |
| `AmericanSwaptionConfig.settlement` | Cash | · | · | · | T25 | · |
| `AmericanSwaptionConfig.long_short` | Long | · | · | · | T23, T24, T25, T26, T27, T28, T29, T35 | · |
| `AmericanSwaptionConfig.long_short` | Short | · | · | · | T22 | · |
| `BondConfig.currency` | = USD | · | · | · | · | T24, T25, T26, T27, T28, T29 |
| `BondConfig.currency` | ≠ USD | · | · | · | · | R1 |
| `BondConfig.accrual_day_count` | ACT/365 | · | · | · | · | T25 |
| `BondConfig.accrual_day_count` | ACT/ACT (ICMA) | · | · | · | · | T24, T25, T26, T27, T28, T29 |
| `BondConfig.coupon_schedule` | coupons | · | · | · | · | T24, T25, T26, T27, T28, T29 |
| `BondConfig.coupon_schedule` | none (a bill) | · | · | · | · | T25 |
| `BondConfig.redemption_fraction` | = 1.0 | · | · | · | · | T24, T25, T26, T27, T28, T29 |
| `BondConfig.redemption_fraction` | ≠ 1.0 | · | · | · | · | R2 |
| `BondConfig.coupon_schedule.payment_date` | at each period's end | · | · | · | · | T24, T25, T26, T27, T28, T29 |
| `BondConfig.coupon_schedule.payment_date` | delayed | · | · | · | · | R3 |
| `LgmSwaptionEngineConfig.calibration` | Bootstrap | · | · | T22, T24, T25, T26, T27, T28, T29, T35 | T22, T24, T25, T26, T27, T28, T29, T35 | · |
| `LgmSwaptionEngineConfig.calibration` | None | · | · | T23 | T23 | · |
| `LgmSwaptionEngineConfig.strategy` | CoterminalDealStrike | · | · | T22, T24, T25, T26, T27, T28, T29, T35 | T22, T24, T25, T26, T27, T28, T29, T35 | · |
| `LgmSwaptionEngineConfig.strategy` | CoterminalATM | · | · | T25 | T25 | · |
| `LgmSwaptionEngineConfig.bootstrap_tolerance` | = 0.0001 | · | · | T22, T24, T25, T26, T27, T28, T29, T35 | T22, T24, T25, T26, T27, T28, T29, T35 | · |
| `LgmSwaptionEngineConfig.bootstrap_tolerance` | ≠ 0.0001 | · | · | T25 | T25 | · |
| `LgmSwaptionEngineConfig.reversion` | = 0.0 | · | · | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | · |
| `LgmSwaptionEngineConfig.reversion` | ≠ 0.0 | · | · | T25 | T25 | · |
| `LgmSwaptionEngineConfig.volatility` | = 0.01 | · | · | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | · |
| `LgmSwaptionEngineConfig.volatility` | ≠ 0.01 | · | · | T25 | T25 | · |
| `LgmSwaptionEngineConfig.reference_calibration_grid` | = 400,3M | · | · | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | · |
| `LgmSwaptionEngineConfig.reference_calibration_grid` | ≠ 400,3M | · | · | T25 | T25 | · |
| `LgmSwaptionEngineConfig.shift_horizon` | = 0.5 | · | · | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | · |
| `LgmSwaptionEngineConfig.shift_horizon` | ≠ 0.5 | · | · | T25 | T25 | · |
| `LgmSwaptionEngineConfig.n_per_std` | = 30 | · | · | T35 | T35 | · |
| `LgmSwaptionEngineConfig.n_per_std` | ≠ 30 | · | · | T22, T23, T24, T25, T26, T27, T28, T29 | T22, T23, T24, T25, T26, T27, T28, T29 | · |
| `LgmSwaptionEngineConfig.std_devs` | = 5.0 | · | · | T35 | T35 | · |
| `LgmSwaptionEngineConfig.std_devs` | ≠ 5.0 | · | · | T22, T23, T24, T25, T26, T27, T28, T29 | T22, T23, T24, T25, T26, T27, T28, T29 | · |
| `LgmSwaptionEngineConfig.exercise_time_steps_per_year` | = 24 | · | · | · | T22, T23, T24, T25, T26, T27, T28, T29, T35 | · |
| `LgmSwaptionEngineConfig.exercise_time_steps_per_year` | ≠ 24 | · | · | · | T25 | · |
| `LgmSwaptionEngineConfig.swap_index.fixed_tenor` | = 1Y | · | · | T22, T24, T25, T26, T27, T28, T29, T35 | T22, T24, T25, T26, T27, T28, T29, T35 | · |
| `LgmSwaptionEngineConfig.swap_index.fixed_tenor` | ≠ 1Y | · | · | T25 | T25 | · |
| `LgmSwaptionEngineConfig.swap_index.fixed_day_counter` | = ACT/365 | · | · | T22, T24, T25, T26, T27, T28, T29, T35 | T22, T24, T25, T26, T27, T28, T29, T35 | · |
| `LgmSwaptionEngineConfig.swap_index.fixed_day_counter` | ≠ ACT/365 | · | · | T25 | T25 | · |
| `LgmSwaptionEngineConfig.swap_index.index_tenor_months` | = 6 | · | · | T22, T24, T25, T26, T27, T28, T29, T35 | T22, T24, T25, T26, T27, T28, T29, T35 | · |
| `LgmSwaptionEngineConfig.swap_index.index_tenor_months` | ≠ 6 | · | · | T25 | T25 | · |
| `LgmSwaptionEngineConfig.solver` | Newton | · | · | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | · |
| `LgmSwaptionEngineConfig.solver` | Bisection | · | · | T25 | T25 | · |
| `PricingConfig.european` | Bachelier | · | T22, T24, T25, T26, T27, T28, T29, T31 | · | · | · |
| `PricingConfig.european` | Jamshidian | · | T1 | · | · | · |
| `JamshidianEngineConfig.reversion` | any | · | T1 | · | · | · |
| `JamshidianEngineConfig.volatility` | any | · | T1 | · | · | · |
| `JamshidianEngineConfig.solver` | Newton | · | T1 | · | · | · |
| `JamshidianEngineConfig.solver` | Bisection | · | R4 | · | · | · |
| `CamConfig.ir` | LgmConfig | T24, T25, T26, T27, T28, T29, T33 | T22, T24, T25, T26, T27, T28, T29, T31 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T24, T25, T26, T27, T28, T29 |
| `CamConfig.ir` | HullWhiteConfig | T1, T24, T25, T26, T27, T28, T29, T33 | T1, T24, T25, T26, T27, T28, T29, T31 | T24, T25, T26, T27, T28, T29, T35 | T24, T25, T26, T27, T28, T29, T35 | T24, T25, T26, T27, T28, T29 |
| `CamConfig.curve_tenors` | = the 13 defaults | T1, T24, T25, T26, T27, T28, T29 | T1, T22, T24, T25, T26, T27, T28, T29 | T22, T23, T24, T25, T26, T27, T28, T29 | T22, T23, T24, T25, T26, T27, T28, T29 | T24, T25, T26, T27, T28, T29 |
| `CamConfig.curve_tenors` | ≠ the 13 defaults | T25, T33 | T25, T31 | T25, T35 | T25, T35 | T25 |
| `CamConfig.swaption_vol_decay` | ForwardVariance | T1, T24, T25, T26, T27, T28, T29, T33 | T1, T22, T24, T25, T26, T27, T28, T29, T31 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T24, T25, T26, T27, T28, T29 |
| `CamConfig.swaption_vol_decay` | ConstantVariance | T25 | T25 | T25 | T25 | T25 |
| `CamConfig.bootstrap_tolerance` | = 0.0001 | T1, T24, T25, T26, T27, T28, T29, T33 | T1, T22, T24, T25, T26, T27, T28, T29, T31 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T24, T25, T26, T27, T28, T29 |
| `CamConfig.bootstrap_tolerance` | ≠ 0.0001 | T25 | T25 | T25 | T25 | T25 |
| `CamConfig.replicates` | = 1 | T1, T24, T25, T26, T27, T28, T29, T33 | T1, T22, T24, T25, T26, T27, T28, T29, T31 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T24, T25, T26, T27, T28, T29 |
| `CamConfig.replicates` | ≠ 1 | R5 | R5 | R5 | R5 | R5 |
| `CamConfig.base_currency` | USD | T1, T24, T25, T26, T27, T28, T29, T33 | T1, T22, T24, T25, T26, T27, T28, T29, T31 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T24, T25, T26, T27, T28, T29 |
| `CamConfig.base_currency` | another currency | R1 | R1 | R1 | R1 | R1 |
| `CamConfig.fx_volatilities` | none | T1, T24, T25, T26, T27, T28, T29, T33 | T1, T22, T24, T25, T26, T27, T28, T29, T31 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T24, T25, T26, T27, T28, T29 |
| `CamConfig.fx_volatilities` | FX simulated | R1 | R1 | R1 | R1 | R1 |
| `CamConfig.equity_volatilities` | none | T1, T24, T25, T26, T27, T28, T29, T33 | T1, T22, T24, T25, T26, T27, T28, T29, T31 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T24, T25, T26, T27, T28, T29 |
| `CamConfig.equity_volatilities` | equities simulated | R6 | R6 | R6 | R6 | R6 |
| `CamConfig.correlations` | none | T1, T24, T25, T26, T27, T28, T29, T33 | T1, T22, T24, T25, T26, T27, T28, T29, T31 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T24, T25, T26, T27, T28, T29 |
| `CamConfig.correlations` | correlated | R1 | R1 | R1 | R1 | R1 |
| `LgmConfig.reversion` | zero | T25 | T25 | T25 | T25 | T25 |
| `LgmConfig.reversion` | positive | T1, T24, T26, T27, T28, T29, T33 | T1, T22, T24, T26, T27, T28, T29, T31 | T22, T23, T24, T26, T27, T28, T29, T35 | T22, T23, T24, T26, T27, T28, T29, T35 | T24, T26, T27, T28, T29 |
| `LgmConfig.reversion` | negative | T25 | T25 | T25 | T25 | T25 |
| `LgmConfig.volatility` | constant | T1, T24, T25, T26, T27, T28, T29, T33 | T1, T22, T24, T25, T26, T27, T28, T29, T31 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T24, T25, T26, T27, T28, T29 |
| `LgmConfig.volatility` | piecewise | T25 | T25 | T25 | T25 | T25 |
| `LgmConfig.calibration_expiries` | calibrated | T24, T25, T26, T27, T28, T29 | T22, T24, T25, T26, T27, T28, T29 | T22, T23, T24, T25, T26, T27, T28, T29 | T22, T23, T24, T25, T26, T27, T28, T29 | T24, T25, T26, T27, T28, T29 |
| `LgmConfig.calibration_expiries` | not calibrated | T1, T25, T33 | T1, T25, T31 | T25, T35 | T25, T35 | T25 |
| `LgmConfig.swap_index.fixed_tenor` | = 1Y | T1, T24, T25, T26, T27, T28, T29, T33 | T1, T22, T24, T25, T26, T27, T28, T29, T31 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T24, T25, T26, T27, T28, T29 |
| `LgmConfig.swap_index.fixed_tenor` | ≠ 1Y | T25 | T25 | T25 | T25 | T25 |
| `LgmConfig.swap_index.fixed_day_counter` | = ACT/365 | T1, T24, T25, T26, T27, T28, T29, T33 | T1, T22, T24, T25, T26, T27, T28, T29, T31 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T24, T25, T26, T27, T28, T29 |
| `LgmConfig.swap_index.fixed_day_counter` | ≠ ACT/365 | T25 | T25 | T25 | T25 | T25 |
| `LgmConfig.swap_index.index_tenor_months` | = 6 | T1, T24, T25, T26, T27, T28, T29, T33 | T1, T22, T24, T25, T26, T27, T28, T29, T31 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T24, T25, T26, T27, T28, T29 |
| `LgmConfig.swap_index.index_tenor_months` | ≠ 6 | T25 | T25 | T25 | T25 | T25 |
| `LgmConfig.solver` | Newton | T1, T24, T25, T26, T27, T28, T29, T33 | T1, T22, T24, T25, T26, T27, T28, T29, T31 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T24, T25, T26, T27, T28, T29 |
| `LgmConfig.solver` | Bisection | T25 | T25 | T25 | T25 | T25 |
| `PortfolioRequest.pfe_quantiles` | = (0.95, 0.99) | R7 | R7 | R7 | R7 | R7 |
| `PortfolioRequest.pfe_quantiles` | ≠ (0.95, 0.99) | T27, T28, T29 | T27, T28, T29 | T27, T28, T29 | T27, T28, T29 | T27, T28, T29 |
| `RunConfig.base_currency` | USD | T1, T24, T25, T26, T27, T28, T29, T33 | T1, T22, T24, T25, T26, T27, T28, T29, T31 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T22, T23, T24, T25, T26, T27, T28, T29, T35 | T24, T25, T26, T27, T28, T29 |
| `RunConfig.base_currency` | another currency | R1 | R1 | R1 | R1 | R1 |

#### Sensitivities: Delta, Gamma, Vega, Theta (`sensitivities`)

| Setting | Value | Swap | European | Bermudan | American | Bond |
|---|---|---|---|---|---|---|
| *compared* |  | T16, T18, T20, T37, T38 | T16, T17, T20, T37, T38 | T14, T20, T37 | T14, T20, T37 | T16, T18, T20, T37, T38 |
| `SwapConfig.payer` | payer | T16, T18, T20, T37, T38 | · | · | · | · |
| `SwapConfig.payer` | receiver | T16, T18, T20, T37, T38 | · | · | · | · |
| `SwapConfig.index_tenor_months` | = 6 | T16, T18, T20, T37, T38 | · | · | · | · |
| `SwapConfig.index_tenor_months` | ≠ 6 | T20 | · | · | · | · |
| `SwapConfig.floating_spread` | = 0.0 | T16, T18, T20, T37, T38 | · | · | · | · |
| `SwapConfig.floating_spread` | ≠ 0.0 | T20 | · | · | · | · |
| `SwapConfig.currency` | = USD | T16, T18, T20, T37, T38 | · | · | · | · |
| `SwapConfig.currency` | ≠ USD | R1 | · | · | · | · |
| `SwapConfig.fixings` | none | T16, T18, T20, T37, T38 | · | · | · | · |
| `SwapConfig.fixings` | past fixings | T16, T18, T20, T37, T38 | · | · | · | · |
| `SwapConfig.accrual_day_count` | ACT/365 | T16, T18, T20, T37, T38 | · | · | · | · |
| `SwapConfig.accrual_day_count` | ACT/ACT (ICMA) | T16, T18, T20, T37, T38 | · | · | · | · |
| `SwaptionConfig.payer` | payer | · | T16, T17, T20, T37, T38 | · | · | · |
| `SwaptionConfig.payer` | receiver | · | T16, T17, T20, T37, T38 | · | · | · |
| `SwaptionConfig.index_tenor_months` | = 6 | · | T16, T17, T20, T37, T38 | · | · | · |
| `SwaptionConfig.index_tenor_months` | ≠ 6 | · | T20 | · | · | · |
| `SwaptionConfig.floating_spread` | = 0.0 | · | T16, T17, T20, T37, T38 | · | · | · |
| `SwaptionConfig.floating_spread` | ≠ 0.0 | · | T16, T17, T20, T37, T38 | · | · | · |
| `SwaptionConfig.currency` | = USD | · | T16, T17, T20, T37, T38 | · | · | · |
| `SwaptionConfig.currency` | ≠ USD | · | R1 | · | · | · |
| `SwaptionConfig.settlement` | Physical | · | T16, T17, T20, T37, T38 | · | · | · |
| `SwaptionConfig.settlement` | Cash | · | T16, T17, T20, T37, T38 | · | · | · |
| `SwaptionConfig.long_short` | Long | · | T16, T17, T20, T37, T38 | · | · | · |
| `SwaptionConfig.long_short` | Short | · | T20 | · | · | · |
| `BermudanSwaptionConfig.payer` | payer | · | · | T14, T20, T37 | · | · |
| `BermudanSwaptionConfig.payer` | receiver | · | · | T14, T20, T37 | · | · |
| `BermudanSwaptionConfig.index_tenor_months` | = 6 | · | · | T14, T20, T37 | · | · |
| `BermudanSwaptionConfig.index_tenor_months` | ≠ 6 | · | · | T20 | · | · |
| `BermudanSwaptionConfig.floating_spread` | = 0.0 | · | · | T14, T20, T37 | · | · |
| `BermudanSwaptionConfig.floating_spread` | ≠ 0.0 | · | · | T20 | · | · |
| `BermudanSwaptionConfig.currency` | = USD | · | · | T14, T20, T37 | · | · |
| `BermudanSwaptionConfig.currency` | ≠ USD | · | · | R1 | · | · |
| `BermudanSwaptionConfig.fixings` | none | · | · | T14, T20, T37 | · | · |
| `BermudanSwaptionConfig.fixings` | past fixings | · | · | T20 | · | · |
| `BermudanSwaptionConfig.settlement` | Physical | · | · | T14, T20, T37 | · | · |
| `BermudanSwaptionConfig.settlement` | Cash | · | · | T14, T20, T37 | · | · |
| `BermudanSwaptionConfig.long_short` | Long | · | · | T20, T37 | · | · |
| `BermudanSwaptionConfig.long_short` | Short | · | · | T14, T20 | · | · |
| `AmericanSwaptionConfig.payer` | payer | · | · | · | T14, T20, T37 | · |
| `AmericanSwaptionConfig.payer` | receiver | · | · | · | T20 | · |
| `AmericanSwaptionConfig.index_tenor_months` | = 6 | · | · | · | T14, T20, T37 | · |
| `AmericanSwaptionConfig.index_tenor_months` | ≠ 6 | · | · | · | T20 | · |
| `AmericanSwaptionConfig.floating_spread` | = 0.0 | · | · | · | T14, T20, T37 | · |
| `AmericanSwaptionConfig.floating_spread` | ≠ 0.0 | · | · | · | T20 | · |
| `AmericanSwaptionConfig.currency` | = USD | · | · | · | T14, T20, T37 | · |
| `AmericanSwaptionConfig.currency` | ≠ USD | · | · | · | R1 | · |
| `AmericanSwaptionConfig.fixings` | none | · | · | · | T14, T20, T37 | · |
| `AmericanSwaptionConfig.fixings` | past fixings | · | · | · | T20 | · |
| `AmericanSwaptionConfig.settlement` | Physical | · | · | · | T14, T20, T37 | · |
| `AmericanSwaptionConfig.settlement` | Cash | · | · | · | T20 | · |
| `AmericanSwaptionConfig.long_short` | Long | · | · | · | T20, T37 | · |
| `AmericanSwaptionConfig.long_short` | Short | · | · | · | T14, T20 | · |
| `BondConfig.currency` | = USD | · | · | · | · | T16, T18, T20, T37, T38 |
| `BondConfig.currency` | ≠ USD | · | · | · | · | R1 |
| `BondConfig.accrual_day_count` | ACT/365 | · | · | · | · | T20 |
| `BondConfig.accrual_day_count` | ACT/ACT (ICMA) | · | · | · | · | T16, T18, T20, T37, T38 |
| `BondConfig.coupon_schedule` | coupons | · | · | · | · | T16, T18, T20, T37, T38 |
| `BondConfig.coupon_schedule` | none (a bill) | · | · | · | · | T20 |
| `BondConfig.redemption_fraction` | = 1.0 | · | · | · | · | T16, T18, T20, T37, T38 |
| `BondConfig.redemption_fraction` | ≠ 1.0 | · | · | · | · | R2 |
| `BondConfig.coupon_schedule.payment_date` | at each period's end | · | · | · | · | T16, T18, T20, T37, T38 |
| `BondConfig.coupon_schedule.payment_date` | delayed | · | · | · | · | R3 |
| `LgmSwaptionEngineConfig.calibration` | Bootstrap | · | · | T14, T20, T37 | T14, T20, T37 | · |
| `LgmSwaptionEngineConfig.calibration` | None | · | · | T14, T20 | T14, T20 | · |
| `LgmSwaptionEngineConfig.strategy` | CoterminalDealStrike | · | · | T14, T20, T37 | T14, T20, T37 | · |
| `LgmSwaptionEngineConfig.strategy` | CoterminalATM | · | · | T14, T20 | T14, T20 | · |
| `LgmSwaptionEngineConfig.bootstrap_tolerance` | = 0.0001 | · | · | T14, T20, T37 | T14, T20, T37 | · |
| `LgmSwaptionEngineConfig.bootstrap_tolerance` | ≠ 0.0001 | · | · | T20 | T20 | · |
| `LgmSwaptionEngineConfig.reversion` | = 0.0 | · | · | T14, T20, T37 | T14, T20, T37 | · |
| `LgmSwaptionEngineConfig.reversion` | ≠ 0.0 | · | · | T14, T20 | T14, T20 | · |
| `LgmSwaptionEngineConfig.volatility` | = 0.01 | · | · | T14, T37 | T14, T37 | · |
| `LgmSwaptionEngineConfig.volatility` | ≠ 0.01 | · | · | T14, T20 | T14, T20 | · |
| `LgmSwaptionEngineConfig.reference_calibration_grid` | = 400,3M | · | · | T14, T20, T37 | T14, T20, T37 | · |
| `LgmSwaptionEngineConfig.reference_calibration_grid` | ≠ 400,3M | · | · | T20 | T20 | · |
| `LgmSwaptionEngineConfig.shift_horizon` | = 0.5 | · | · | T14, T20, T37 | T14, T20, T37 | · |
| `LgmSwaptionEngineConfig.shift_horizon` | ≠ 0.5 | · | · | T14, T20 | T14, T20 | · |
| `LgmSwaptionEngineConfig.n_per_std` | = 30 | · | · | T37 | T37 | · |
| `LgmSwaptionEngineConfig.n_per_std` | ≠ 30 | · | · | T14, T20 | T14, T20 | · |
| `LgmSwaptionEngineConfig.std_devs` | = 5.0 | · | · | T20, T37 | T20, T37 | · |
| `LgmSwaptionEngineConfig.std_devs` | ≠ 5.0 | · | · | T14, T20 | T14, T20 | · |
| `LgmSwaptionEngineConfig.exercise_time_steps_per_year` | = 24 | · | · | · | T14, T20, T37 | · |
| `LgmSwaptionEngineConfig.exercise_time_steps_per_year` | ≠ 24 | · | · | · | T20 | · |
| `LgmSwaptionEngineConfig.swap_index.fixed_tenor` | = 1Y | · | · | T14, T20, T37 | T14, T20, T37 | · |
| `LgmSwaptionEngineConfig.swap_index.fixed_tenor` | ≠ 1Y | · | · | T20 | T20 | · |
| `LgmSwaptionEngineConfig.swap_index.fixed_day_counter` | = ACT/365 | · | · | T14, T20, T37 | T14, T20, T37 | · |
| `LgmSwaptionEngineConfig.swap_index.fixed_day_counter` | ≠ ACT/365 | · | · | T20 | T20 | · |
| `LgmSwaptionEngineConfig.swap_index.index_tenor_months` | = 6 | · | · | T14, T20, T37 | T14, T20, T37 | · |
| `LgmSwaptionEngineConfig.swap_index.index_tenor_months` | ≠ 6 | · | · | T20 | T20 | · |
| `LgmSwaptionEngineConfig.solver` | Newton | · | · | T14, T20, T37 | T14, T20, T37 | · |
| `LgmSwaptionEngineConfig.solver` | Bisection | · | · | T20 | T20 | · |
| `PricingConfig.european` | Bachelier | · | T16, T17, T20, T37, T38 | · | · | · |
| `PricingConfig.european` | Jamshidian | · | R4 | · | · | · |
| `JamshidianEngineConfig.reversion` | any | · | R4 | · | · | · |
| `JamshidianEngineConfig.volatility` | any | · | R4 | · | · | · |
| `JamshidianEngineConfig.solver` | Newton | · | R4 | · | · | · |
| `JamshidianEngineConfig.solver` | Bisection | · | R4 | · | · | · |
| `GreeksConfig.method` | Bump | T16, T18, T20, T37 | T16, T17, T20, T37 | T14, T20, T37 | T14, T20, T37 | T16, T18, T20, T37 |
| `GreeksConfig.method` | AD | T20, T38 | T20, T38 | R8 | R8 | T20, T38 |
| `SensitivityConfig.curve_tenors` | = the 13 defaults | T20, T37, T38 | T20, T37, T38 | T20, T37 | T20, T37 | T20, T37, T38 |
| `SensitivityConfig.curve_tenors` | ≠ the 13 defaults | T16, T18, T20 | T16, T17, T20 | T14, T20 | T14, T20 | T16, T18, T20 |
| `SensitivityConfig.curve_shift` | = 0.0001 | T18, T20, T37, T38 | T17, T20, T37, T38 | T14, T20, T37 | T14, T20, T37 | T18, T20, T37, T38 |
| `SensitivityConfig.curve_shift` | ≠ 0.0001 | T16, T20 | T16, T20 | T20 | T20 | T16, T20 |
| `SensitivityConfig.vol_shift` | = 0.0001 | · | T17, T20, T37, T38 | T14, T20, T37 | T14, T20, T37 | · |
| `SensitivityConfig.vol_shift` | ≠ 0.0001 | · | T16, T20 | T20 | T20 | · |
| `SensitivityConfig.theta_days` | = 1 | T20, T37, T38 | T20, T37, T38 | T14, T20, T37 | T14, T20, T37 | T20, T37, T38 |
| `SensitivityConfig.theta_days` | ≠ 1 | T16, T18, T20 | T16, T17, T20 | T20 | T20 | T16, T18, T20 |
| `RunConfig.base_currency` | USD | T16, T18, T20, T37, T38 | T16, T17, T20, T37, T38 | T14, T20, T37 | T14, T20, T37 | T16, T18, T20, T37, T38 |
| `RunConfig.base_currency` | another currency | R1 | R1 | R1 | R1 | R1 |

#### Market risk: the P&L of every scenario, VaR and ES (`market_risk`)

| Setting | Value | Swap | European | Bermudan | American | Bond |
|---|---|---|---|---|---|---|
| *compared* |  | T5, T6, T7, T8, T19 | T6, T7, T8, T19 | T8, T15, T19 | T8, T15, T19 | T5, T6, T7, T8, T19 |
| `SwapConfig.payer` | payer | T5, T6, T7, T8, T19 | · | · | · | · |
| `SwapConfig.payer` | receiver | T19 | · | · | · | · |
| `SwapConfig.index_tenor_months` | = 6 | T5, T6, T7, T8, T19 | · | · | · | · |
| `SwapConfig.index_tenor_months` | ≠ 6 | T19 | · | · | · | · |
| `SwapConfig.floating_spread` | = 0.0 | T5, T6, T7, T8, T19 | · | · | · | · |
| `SwapConfig.floating_spread` | ≠ 0.0 | T19 | · | · | · | · |
| `SwapConfig.currency` | = USD | T5, T6, T7, T8, T19 | · | · | · | · |
| `SwapConfig.currency` | ≠ USD | R1 | · | · | · | · |
| `SwapConfig.fixings` | none | T5, T6, T7, T8, T19 | · | · | · | · |
| `SwapConfig.fixings` | past fixings | T19 | · | · | · | · |
| `SwapConfig.accrual_day_count` | ACT/365 | T5, T6, T7, T8, T19 | · | · | · | · |
| `SwapConfig.accrual_day_count` | ACT/ACT (ICMA) | T19 | · | · | · | · |
| `SwaptionConfig.payer` | payer | · | T19 | · | · | · |
| `SwaptionConfig.payer` | receiver | · | T6, T7, T8, T19 | · | · | · |
| `SwaptionConfig.index_tenor_months` | = 6 | · | T6, T7, T8, T19 | · | · | · |
| `SwaptionConfig.index_tenor_months` | ≠ 6 | · | T19 | · | · | · |
| `SwaptionConfig.floating_spread` | = 0.0 | · | T6, T7, T8, T19 | · | · | · |
| `SwaptionConfig.floating_spread` | ≠ 0.0 | · | T19 | · | · | · |
| `SwaptionConfig.currency` | = USD | · | T6, T7, T8, T19 | · | · | · |
| `SwaptionConfig.currency` | ≠ USD | · | R1 | · | · | · |
| `SwaptionConfig.settlement` | Physical | · | T6, T7, T8, T19 | · | · | · |
| `SwaptionConfig.settlement` | Cash | · | T19 | · | · | · |
| `SwaptionConfig.long_short` | Long | · | T6, T7, T8, T19 | · | · | · |
| `SwaptionConfig.long_short` | Short | · | T19 | · | · | · |
| `BermudanSwaptionConfig.payer` | payer | · | · | T8, T15, T19 | · | · |
| `BermudanSwaptionConfig.payer` | receiver | · | · | T15, T19 | · | · |
| `BermudanSwaptionConfig.index_tenor_months` | = 6 | · | · | T8, T15, T19 | · | · |
| `BermudanSwaptionConfig.index_tenor_months` | ≠ 6 | · | · | T19 | · | · |
| `BermudanSwaptionConfig.floating_spread` | = 0.0 | · | · | T8, T15, T19 | · | · |
| `BermudanSwaptionConfig.floating_spread` | ≠ 0.0 | · | · | T19 | · | · |
| `BermudanSwaptionConfig.currency` | = USD | · | · | T8, T15, T19 | · | · |
| `BermudanSwaptionConfig.currency` | ≠ USD | · | · | R1 | · | · |
| `BermudanSwaptionConfig.fixings` | none | · | · | T8, T15, T19 | · | · |
| `BermudanSwaptionConfig.fixings` | past fixings | · | · | T19 | · | · |
| `BermudanSwaptionConfig.settlement` | Physical | · | · | T8, T15, T19 | · | · |
| `BermudanSwaptionConfig.settlement` | Cash | · | · | T15, T19 | · | · |
| `BermudanSwaptionConfig.long_short` | Long | · | · | T8, T15, T19 | · | · |
| `BermudanSwaptionConfig.long_short` | Short | · | · | T19 | · | · |
| `AmericanSwaptionConfig.payer` | payer | · | · | · | T15, T19 | · |
| `AmericanSwaptionConfig.payer` | receiver | · | · | · | T8, T19 | · |
| `AmericanSwaptionConfig.index_tenor_months` | = 6 | · | · | · | T8, T15, T19 | · |
| `AmericanSwaptionConfig.index_tenor_months` | ≠ 6 | · | · | · | T19 | · |
| `AmericanSwaptionConfig.floating_spread` | = 0.0 | · | · | · | T8, T15, T19 | · |
| `AmericanSwaptionConfig.floating_spread` | ≠ 0.0 | · | · | · | T19 | · |
| `AmericanSwaptionConfig.currency` | = USD | · | · | · | T8, T15, T19 | · |
| `AmericanSwaptionConfig.currency` | ≠ USD | · | · | · | R1 | · |
| `AmericanSwaptionConfig.fixings` | none | · | · | · | T8, T15, T19 | · |
| `AmericanSwaptionConfig.fixings` | past fixings | · | · | · | T19 | · |
| `AmericanSwaptionConfig.settlement` | Physical | · | · | · | T8, T15, T19 | · |
| `AmericanSwaptionConfig.settlement` | Cash | · | · | · | T19 | · |
| `AmericanSwaptionConfig.long_short` | Long | · | · | · | T8, T15, T19 | · |
| `AmericanSwaptionConfig.long_short` | Short | · | · | · | T19 | · |
| `BondConfig.currency` | = USD | · | · | · | · | T5, T6, T7, T8, T19 |
| `BondConfig.currency` | ≠ USD | · | · | · | · | R1 |
| `BondConfig.accrual_day_count` | ACT/365 | · | · | · | · | T19 |
| `BondConfig.accrual_day_count` | ACT/ACT (ICMA) | · | · | · | · | T5, T6, T7, T8, T19 |
| `BondConfig.coupon_schedule` | coupons | · | · | · | · | T5, T6, T7, T8, T19 |
| `BondConfig.coupon_schedule` | none (a bill) | · | · | · | · | T19 |
| `BondConfig.redemption_fraction` | = 1.0 | · | · | · | · | T5, T6, T7, T8, T19 |
| `BondConfig.redemption_fraction` | ≠ 1.0 | · | · | · | · | R2 |
| `BondConfig.coupon_schedule.payment_date` | at each period's end | · | · | · | · | T5, T6, T7, T8, T19 |
| `BondConfig.coupon_schedule.payment_date` | delayed | · | · | · | · | R3 |
| `LgmSwaptionEngineConfig.calibration` | Bootstrap | · | · | T8, T15, T19 | T8, T15, T19 | · |
| `LgmSwaptionEngineConfig.calibration` | None | · | · | T8, T19 | T19 | · |
| `LgmSwaptionEngineConfig.strategy` | CoterminalDealStrike | · | · | T8, T15, T19 | T8, T15, T19 | · |
| `LgmSwaptionEngineConfig.strategy` | CoterminalATM | · | · | T19 | T19 | · |
| `LgmSwaptionEngineConfig.bootstrap_tolerance` | = 0.0001 | · | · | T8, T15, T19 | T8, T15, T19 | · |
| `LgmSwaptionEngineConfig.bootstrap_tolerance` | ≠ 0.0001 | · | · | T19 | T19 | · |
| `LgmSwaptionEngineConfig.reversion` | = 0.0 | · | · | T15, T19 | T15, T19 | · |
| `LgmSwaptionEngineConfig.reversion` | ≠ 0.0 | · | · | T8, T19 | T8, T19 | · |
| `LgmSwaptionEngineConfig.volatility` | = 0.01 | · | · | T8, T15 | T8, T15 | · |
| `LgmSwaptionEngineConfig.volatility` | ≠ 0.01 | · | · | T19 | T19 | · |
| `LgmSwaptionEngineConfig.reference_calibration_grid` | = 400,3M | · | · | T8, T15, T19 | T8, T15, T19 | · |
| `LgmSwaptionEngineConfig.reference_calibration_grid` | ≠ 400,3M | · | · | T19 | T19 | · |
| `LgmSwaptionEngineConfig.shift_horizon` | = 0.5 | · | · | T8, T15, T19 | T8, T15, T19 | · |
| `LgmSwaptionEngineConfig.shift_horizon` | ≠ 0.5 | · | · | T19 | T19 | · |
| `LgmSwaptionEngineConfig.n_per_std` | = 30 | · | · | T15 | T15 | · |
| `LgmSwaptionEngineConfig.n_per_std` | ≠ 30 | · | · | T8, T19 | T8, T19 | · |
| `LgmSwaptionEngineConfig.std_devs` | = 5.0 | · | · | T8, T15, T19 | T8, T15, T19 | · |
| `LgmSwaptionEngineConfig.std_devs` | ≠ 5.0 | · | · | T19 | T19 | · |
| `LgmSwaptionEngineConfig.exercise_time_steps_per_year` | = 24 | · | · | · | T8, T15, T19 | · |
| `LgmSwaptionEngineConfig.exercise_time_steps_per_year` | ≠ 24 | · | · | · | T19 | · |
| `LgmSwaptionEngineConfig.swap_index.fixed_tenor` | = 1Y | · | · | T8, T15, T19 | T8, T15, T19 | · |
| `LgmSwaptionEngineConfig.swap_index.fixed_tenor` | ≠ 1Y | · | · | T19 | T19 | · |
| `LgmSwaptionEngineConfig.swap_index.fixed_day_counter` | = ACT/365 | · | · | T8, T15, T19 | T8, T15, T19 | · |
| `LgmSwaptionEngineConfig.swap_index.fixed_day_counter` | ≠ ACT/365 | · | · | T19 | T19 | · |
| `LgmSwaptionEngineConfig.swap_index.index_tenor_months` | = 6 | · | · | T8, T15, T19 | T8, T15, T19 | · |
| `LgmSwaptionEngineConfig.swap_index.index_tenor_months` | ≠ 6 | · | · | T19 | T19 | · |
| `LgmSwaptionEngineConfig.solver` | Newton | · | · | T8, T15, T19 | T8, T15, T19 | · |
| `LgmSwaptionEngineConfig.solver` | Bisection | · | · | T19 | T19 | · |
| `PricingConfig.european` | Bachelier | · | T6, T7, T8, T19 | · | · | · |
| `PricingConfig.european` | Jamshidian | · | T8 | · | · | · |
| `JamshidianEngineConfig.reversion` | any | · | T8 | · | · | · |
| `JamshidianEngineConfig.volatility` | any | · | T8 | · | · | · |
| `JamshidianEngineConfig.solver` | Newton | · | T8 | · | · | · |
| `JamshidianEngineConfig.solver` | Bisection | · | R4 | · | · | · |
| `ShockScenarios.source` | monte-carlo | T6, T7, T8, T19 | T6, T7, T8, T19 | T8, T19 | T8, T19 | T6, T7, T8, T19 |
| `ShockScenarios.source` | historical | T5, T19 | T19 | T15, T19 | T15, T19 | T5, T19 |
| `ShockScenarios.replicates` | one | T5, T6, T7, T8, T19 | T6, T7, T8, T19 | T8, T15, T19 | T8, T15, T19 | T5, T6, T7, T8, T19 |
| `ShockScenarios.replicates` | several | R9 | R9 | R9 | R9 | R9 |
| `ShockScenarios.bootstrap_replicates` | = 128 | T5, T6, T7, T8, T19 | T6, T7, T8, T19 | T8, T15, T19 | T8, T15, T19 | T5, T6, T7, T8, T19 |
| `ShockScenarios.bootstrap_replicates` | ≠ 128 | R9 | R9 | R9 | R9 | R9 |
| `ShockScenarios.block_length` | = None | T5, T6, T7, T8, T19 | T6, T7, T8, T19 | T8, T15, T19 | T8, T15, T19 | T5, T6, T7, T8, T19 |
| `ShockScenarios.block_length` | ≠ None | R9 | R9 | R9 | R9 | R9 |
| `ShockScenarios.bootstrap_seed` | = 42 | T5, T6, T7, T8, T19 | T6, T7, T8, T19 | T8, T15, T19 | T8, T15, T19 | T5, T6, T7, T8, T19 |
| `ShockScenarios.bootstrap_seed` | ≠ 42 | R9 | R9 | R9 | R9 | R9 |
| `MarketRiskRequest.quantiles` | = (0.99, 0.975) | T7, T8, T19 | T7, T8, T19 | T8, T19 | T8, T19 | T7, T8, T19 |
| `MarketRiskRequest.quantiles` | ≠ (0.99, 0.975) | T5, T6, T19 | T6, T19 | T15, T19 | T15, T19 | T5, T6, T19 |
| `MarketRiskRequest.batch_size` | = 256 | T5, T6, T7, T8, T19 | T6, T7, T8, T19 | T8, T15, T19 | T8, T15, T19 | T5, T6, T7, T8, T19 |
| `MarketRiskRequest.batch_size` | ≠ 256 | R10 | R10 | R10 | R10 | R10 |
| `MarketRiskRequest.base_currency` | = USD | T5, T6, T7, T8, T19 | T6, T7, T8, T19 | T8, T15, T19 | T8, T15, T19 | T5, T6, T7, T8, T19 |
| `MarketRiskRequest.base_currency` | ≠ USD | R1 | R1 | R1 | R1 | R1 |

#### Calibration of the simulation's model (`calibration`)

| Setting | Value | Cross-asset model |
|---|---|---|
| *compared* |  | T25, T30 |
| `CamConfig.ir` | LgmConfig | T25, T30 |
| `CamConfig.ir` | HullWhiteConfig | T25, T30 |
| `CamConfig.bootstrap_tolerance` | = 0.0001 | T25, T30 |
| `CamConfig.bootstrap_tolerance` | ≠ 0.0001 | T25 |
| `LgmConfig.reversion` | zero | T25 |
| `LgmConfig.reversion` | positive | T30 |
| `LgmConfig.reversion` | negative | T25 |
| `LgmConfig.volatility` | constant | T25, T30 |
| `LgmConfig.volatility` | piecewise | T25 |
| `LgmConfig.calibration_expiries` | calibrated | T25, T30 |
| `LgmConfig.calibration_expiries` | not calibrated | T25 |
| `LgmConfig.swap_index.fixed_tenor` | = 1Y | T25, T30 |
| `LgmConfig.swap_index.fixed_tenor` | ≠ 1Y | T25 |
| `LgmConfig.swap_index.fixed_day_counter` | = ACT/365 | T25, T30 |
| `LgmConfig.swap_index.fixed_day_counter` | ≠ ACT/365 | T25 |
| `LgmConfig.swap_index.index_tenor_months` | = 6 | T25, T30 |
| `LgmConfig.swap_index.index_tenor_months` | ≠ 6 | T25 |
| `LgmConfig.solver` | Newton | T25, T30 |
| `LgmConfig.solver` | Bisection | T25 |

**Tests.**

- **T1** `tests/test_end_to_end.py::TestEndToEndEngineVsORE::test_every_paths_value_matches_ore`: 1e-10 relative (the swap), 5e-6 (the Jamshidian Europeans), each path
- **T2** `tests/test_jamshidian.py::TestAgainstQuantLibJamshidianEngine::test_grid`: 5e-6 relative (QuantLib's Brent root on r*)
- **T3** `tests/test_jamshidian.py::TestAgainstQuantLibJamshidianEngine::test_matches_quantlib_forward_starting`: 5e-6 relative (QuantLib's Brent root on r*)
- **T4** `tests/test_jamshidian.py::TestAgainstQuantLibJamshidianEngine::test_matches_quantlib_spot_starting`: 5e-6 relative (QuantLib's Brent root on r*)
- **T5** `tests/test_market_risk_ore_parity.py::TestHistoricalVarEsMatchesOre::test_var_and_es_match`: VaR, ES 1e-10 relative to ORE's RiskStatistics on ORE's P&L
- **T6** `tests/test_market_risk_ore_parity.py::TestMonteCarloVarEsMatchesOre::test_pnl_vector_matches`: P&L 1e-6 relative; VaR, ES 1e-6 relative to ORE's RiskStatistics
- **T7** `tests/test_market_risk_ore_parity.py::test_base_value_is_ores`: 1e-10 relative
- **T8** `tests/test_market_risk_ore_parity.py::test_every_scenario_revalues_as_ore_does`: 5e-10 relative, each scenario's value
- **T9** `tests/test_ore_lgm_calibration.py::test_calibrated_bermudan_equals_ores_bootstrap`: 1e-9 relative
- **T10** `tests/test_ore_lgm_parity.py::test_a_shift_horizon_beyond_the_maturity_is_ores`: 1e-10 relative
- **T11** `tests/test_ore_lgm_parity.py::test_engine_equals_ore_lgm_engine`: 1e-10 relative
- **T12** `tests/test_ore_parity_settings.py::test_a_calibrated_bermudans_helpers_are_ores`: 1e-12 of the notional
- **T13** `tests/test_ore_parity_settings.py::test_an_american_without_a_calibrating_helper_is_refused_as_ore_refuses_it`: refused, as ORE refuses it
- **T14** `tests/test_ore_parity_settings.py::test_each_calibration_strategy_and_shift_horizon_is_ores`: 1e-10 of the notional
- **T15** `tests/test_ore_parity_settings.py::test_market_risk_of_options_at_ores_default_engine_is_ores`: P&L 1e-10 of the notional; VaR 1e-9 relative
- **T16** `tests/test_ore_parity_settings.py::test_the_sensitivity_settings_are_ores`: 1e-12 of the notional
- **T17** `tests/test_ore_parity_settings.py::test_theta_across_a_weekend_reads_the_volatilities_from_the_monday_as_ore`: 1e-12 of the notional
- **T18** `tests/test_ore_parity_settings.py::test_theta_onto_a_coupon_adds_nothing_back_as_ores`: 1e-12 of the notional
- **T19** `tests/test_ore_parity_sweep.py::test_market_risk_is_ores`: each trade's P&L in every scenario 1e-11 of the notional; VaR 1e-9 relative to ORE's report; ES to ORE's RiskStatistics on ORE's P&L
- **T20** `tests/test_ore_parity_sweep.py::test_the_greeks_are_ores`: bump Greeks 1e-10 of the notional (Bermudans, Americans), 1e-12 (the rest); AD as the shift vanishes (Richardson), 1e-5 of each curve's largest
- **T21** `tests/test_ore_parity_sweep.py::test_todays_values_are_ores`: 2e-11 of the notional (Bermudans, Americans), 1e-12 (the rest); a trade ORE refuses to build refused
- **T22** `tests/test_ore_xva_parity.py::test_a_short_option_on_ores_paths_is_ores_short_option`: each cell 1e-10 of the trade's largest value
- **T23** `tests/test_ore_xva_parity.py::test_an_uncalibrated_option_equals_ores_cube_on_ores_paths`: each cell 1e-10 of the trade's largest value
- **T24** `tests/test_ore_xva_parity.py::test_every_trades_cube_equals_ores_on_ores_paths`: each cell 1e-10 of the trade's largest value
- **T25** `tests/test_ore_xva_parity.py::test_other_settings_equal_ores_simulation_on_ores_paths`: every simulated curve 1e-10 relative, rebuilt from the model's states
- **T26** `tests/test_ore_xva_parity.py::test_t0_is_the_value_on_the_simulation_market_of_the_as_of_date`: T0 1e-10 relative
- **T27** `tests/test_ore_xva_parity.py::test_the_engines_simulation_and_exposure_are_ores_in_distribution` (full tier only): EPE, ENE, PFE within four combined standard errors (independent paths)
- **T28** `tests/test_ore_xva_parity.py::test_the_pipelines_profiles_start_from_ores_t0`: EPE, ENE at t=0 1e-10 relative
- **T29** `tests/test_ore_xva_parity.py::test_the_profiles_of_ores_cube_are_ores_reports`: every profile on ORE's cube 1e-10 relative
- **T30** `tests/test_ore_xva_parity.py::test_the_scenario_dump_is_the_engines_simulation_market_on_ores_paths`: every simulated curve 1e-10 relative, rebuilt from the calibrated model's states
- **T31** `tests/test_pricing.py::test_european_on_every_path_equals_quantlibs_bachelier_engine`: QuantLib's Bachelier engine on each path, each cell
- **T32** `tests/test_pricing.py::test_european_today_equals_ores_default_engine`: 1e-10 relative
- **T33** `tests/test_pricing.py::test_every_path_and_date_equals_ores_discounting_swap_engine`: 1e-12 relative, each cell
- **T34** `tests/test_pricing.py::test_option_today_equals_ores_calibrated_grid_engine`: 1e-9 relative
- **T35** `tests/test_pricing.py::test_the_path_model_is_the_calibration_ore_built_the_trade_with`: 1e-10 of the value, each cell (ORE's grid engine on the model as built)
- **T36** `tests/test_pricing.py::test_today_equals_ores_discounting_swap_engine`: 1e-12 relative
- **T37** `tests/test_sensitivity_ore_parity.py::test_every_greek_of_every_trade_is_ores`: 1e-11 of the NPV
- **T38** `tests/test_sensitivity_ore_parity.py::test_the_ad_greeks_are_ores_as_the_shift_vanishes`: Delta, Gamma 1e-4 of each curve's largest, as the shift vanishes; Theta 1e-11 of the NPV
- **T39** `tests/test_shared_portfolio.py::test_a_short_option_today_is_ores_short_option`: 1e-10 relative (1e-9 calibrated Bermudans, Americans)
- **T40** `tests/test_shared_portfolio.py::test_every_trade_today_equals_ore`: 1e-12 relative (1e-9 calibrated Bermudans, Americans)
- **T41** `tests/test_trade_dates.py::test_any_leg_day_count_equals_ore`: 1e-10 relative
- **T42** `tests/test_trade_dates.py::test_seasoned_bermudan_and_american_equal_ore`: 1e-10 relative
- **T43** `tests/test_trade_dates.py::test_seasoned_swap_equals_ore`: 1e-10 relative

**Reasons.**

- **R1** (not compared yet) a second currency: the oracles are one-currency until F-04 extends them.
- **R2** (ORE cannot express it) the oracle books a coupon bond redeemed at par; a bill redeemed below par is a TraderX instrument priced by its own pricer (I-93).
- **R3** (ORE cannot express it) the oracle's bond pays each coupon at its period's end (ORE's leg data pays at a lag in days, not on a date per period), and a delayed last payment is valued wrong today (I-104).
- **R4** (ORE cannot express it) ORE has no builder for QuantLib's Jamshidian engine, so no ORE analytic prices with it; its value and every market-risk scenario are QuantLib's engine's, in tests/test_jamshidian.py and tests/test_market_risk_ore_parity.py.
- **R5** (not ORE's) the engine's own scrambles of its paths (X-1); on ORE's paths (L3) a cell is compared whatever drew them.
- **R6** (not compared yet) an equity in the simulation: no equity trade yet (I-18, F-04).
- **R7** (ORE cannot express it) ORE's exposure analytic reports one PFE quantile (`PfeQuantile`); each quantile is its own figure of one definition, compared at 0.95.
- **R8** (not ORE's) a grid-priced option's AD Greeks are its price's derivatives, not ORE's 1bp secants across the grid's kinks (I-78); tests/test_greeks_bermudan.py holds them to central differences of the engine's own recalibrating price, and the result says so.
- **R9** (not ORE's) a sampling-uncertainty setting (I-111, F-10): ORE has no error estimate of a VaR; every scenario's P&L is compared, whatever the setting.
- **R10** (not ORE's) how many scenarios are revalued at once, a memory setting; the same P&L within the margin (tests/test_market_risk.py).

<!-- ore-parity-matrix:end -->

## Summary table (the algorithms)

| Algorithm | This engine | ORE C++ source |
|---|---|---|
| Sobol sequence | `random.generate_sobol_normals` | `QuantLib::SobolRsg` (`ql/math/randomnumbers/sobolrsg.cpp`) — same sequence *class*, independent implementation |
| Brownian bridge | `random._build_bridge_matrix`, `apply_brownian_bridge` | `QuantLib::BrownianBridge::initialize`/`transform` (`ql/methods/montecarlo/brownianbridge.cpp`) |
| LGM state step (exact) | `models.cam.step_moments`, `market_simulation.paths.evolve_states` | `CrossAssetStateProcess::ExactDiscretization`, `CrossAssetAnalytics`, `IrLgm1fStateProcess` |
| Hull-White parametrization | `cam.IrComponent(volatility_type="HullWhite")`, `lgm.hull_white_zeta` | `IrLgm1fPiecewiseConstantHullWhiteAdaptor`; bonds equal `QuantLib::HullWhite::discountBond` |
| Model-implied curves, numeraire | `scenario_market.implied_log_discounts`, `lgm_numeraire` | `CrossAssetModelScenarioGenerator::nextPath`, `LinearGaussMarkovModel::discountBond`/`numeraire` |
| Correlation and square root | `cam.flexible_cholesky` | `CrossAssetModel` correlation; `CholeskyDecomposition(cov, flexible = true)` |
| Swap pricing | `valuation.legs.legs_npv` | `QuantLib::DiscountingSwapEngine::calculate`, `IborCouponPricer` at par |
| European (default) | `valuation.european` | `BlackMultiLegOptionEngine` (Bachelier), `ParYieldCurve` cash settlement |
| Jamshidian swaption decomposition | `valuation.jamshidian.jamshidian_npv`, `_solve_decreasing_root` | `QuantLib::JamshidianSwaptionEngine::calculate`, `rStarFinder` |
| Bond option (Black-on-bond) | `hull_white.bond_call`/`bond_put` | `QuantLib::HullWhite::discountBondOption` |
| American/Bermudan LGM bond price | `models.lgm.bond_price`; deflated, `pricing.lgm_grid._basis` | `QuantExt::LinearGaussMarkovModel::discountBond` (`qle/models/lgm.hpp`) |
| American/Bermudan backward induction, including ORE's cashflow bookkeeping | `pricing.lgm_grid._induction`, `_rollback_operator`, `_GridSchedule` (replayed as `_Induction`) | `QuantExt::NumericLgmMultiLegOptionEngineBase::calculate`, `LgmConvolutionSolver2` — equal to ORE's own engine to 1e-10, reached in-process via `OREApp`, see [7b](#7b-ores-own-lgm-engine-reached-in-process-2026-09-23) |
| Coupon membership and proration by exercise style | `bermudan_swaption.ExerciseStyle`, `prepare_bermudan` | `NumericLgmMultiLegOptionEngineBase::buildCashflowInfo` (`belongsToUnderlyingMaxTime_`, `couponRatio`) |
| Ibor projection in the LGM | `pricing.lgm_grid._Coefficients`, `_basis` | `QuantExt::LgmVectorised::fixing` (index fixing period) |
| American exercise-window discretization | `AmericanSwaptionConfig.option_times` | `NumericLgmMultiLegOptionEngineBase::calculate`'s American branch (truncating step count) |
| Per-trade basket and bootstrap | `valuation.bermudan.calibration_basket`, `calibration.lgm` | `IrModelBuilder::buildSwaptionBasket`, `LgmBuilder::calibrate` |
| Exercise wrapper, and an option's position | `valuation.options.wrap`, `position_sign` | `OptionWrapper`, `BermudanOptionWrapper` (`isLong_`, `multiplier2_`) |
| VaR | `var_es.value_at_risk` | `QuantLib::RiskStatistics::valueAtRisk` → `GeneralStatistics::percentile` |
| Expected Shortfall | `var_es.expected_shortfall` | `QuantLib::RiskStatistics::expectedShortfall` |
| Delta / Gamma / Vega (bump) | `sensitivities.portfolio_sensitivities` | `SensitivityScenarioGenerator`, `ShiftScenarioGenerator::applyShift`, `SensitivityCube` |
| Delta / Gamma / Vega (AD) | `greeks.portfolio_greeks` | the limit of the above; `DiscountingSwapEngineDeltaGamma`-style closed forms |
| Theta (evaluation-date roll) | `sensitivities.trade_theta` | `SensitivityAnalysis::generateSensitivities`'s theta branch |
| The cross-asset model's basket and bootstrap | `calibration.cam.calibrate_currency`, `calibration.lgm.build_basket`, `bootstrap_sigma` | `CrossAssetModelBuilder` → `IrModelBuilder::buildSwaptionBasket`, `LgmBuilder::calibrate` → `calibrateVolatilitiesIterative`, `CalibratedModel::calibrateIterative` |

## Tested by

- The pipeline: `tests/test_cam.py`, `tests/test_pricing.py` (every path test under both
  models), `tests/test_ore_lgm_calibration.py`, `tests/test_shared_portfolio.py`,
  `tests/test_pipeline.py`, `tests/test_sensitivities.py`, `tests/test_curves.py`,
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
- `tests/test_models_piecewise_sigma.py`, `tests/test_ore_lgm_calibration.py`,
  `tests/test_calibration_acceptance.py`, `tests/test_greeks.py`, `tests/test_greeks_bermudan.py`
  — the tests specific to sections 9-10 (Greeks, calibration); see
  [Delta, Gamma, and Theta](../risk/greeks.md#tested-by),
  [Models & Trades](models-and-trades.md#tested-by), and
  [Calibration](calibration.md#tested-by) for the full per-file breakdown.
