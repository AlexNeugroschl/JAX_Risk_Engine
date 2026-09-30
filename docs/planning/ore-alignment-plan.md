# ORE Alignment Plan — ORE's models and valuation semantics, configurable

**Date:** 2026-09-28 · **Status:** Phases 0–6 and 8 implemented in part, 2026-09-29; owner
decisions recorded 2026-09-30 and Phases 9–10 added for them (see
[Implementation status](#implementation-status)) · **Supersedes:** the "recommended order" in
[engine-audit.md](engine-audit.md) for M-1, M-2, M-3, A-1 to A-4, P-2.

**Goal.** Make every number the engine produces, at t=0 and on every simulated path, the
number ORE would produce for the same trade, market and configuration, computed the same way
mathematically. Where that is impossible, the difference is named in §3 and stays named in
the code.

**Why this order.** The open model-correctness issues (I-42, I-44, I-45, I-43, audit M-2)
all sit in the simulation, the Hull-White pricers, the cube swap kernel and the trade
configs. Fixing them one at a time rewrites the same files several times, and performance
work (P-2, I-21, P-1) on those files would be discarded. This plan makes one structural change
(Phases 3–5), preceded only by work that survives it (Phases 0–2), and followed by work that
depends on it (Phases 6–8). The owner's decisions of 2026-09-30 make the engine configurable
rather than single-model (Phase 9) and close the remaining differences they chose to close
(Phase 10).

---

## Implementation status

Recorded 2026-09-29, updated 2026-09-30 with the owner's decisions
([compliance/decisions.md](../../compliance/decisions.md)). The market path (`price_portfolio`
on a `Market`, the LGM cross-asset model) is the result, and the default. The Hull-White model
stays as a supported, non-default option (decision A-1); today it is a separate code path, and
Phase 9 makes both models options of one configuration. Issue numbers are in
[known-issues.md](../known-issues.md).

| Phase | Status | What is not done |
|---|---|---|
| 0 Decisions and gates | Targets, differences and decisions recorded in `compliance/decisions.md`; the owner decided A-1 to A-9, D-9, D-10 and X-5, X-9 to X-11 on 2026-09-30 (the other X entries stay engineering defaults). Gates V-2, V-3, V-5 to V-10 closed (V-5 by an OREApp run, the rest from source plus binding-level tests); V-1 half closed | V-1's two recalibration details (I-49, decided: close, Phase 10); V-4 |
| 1 Safety net and oracles | The OREApp oracle takes index curves, swaption vols, calibration and discount-segment curves; baseline worktree at `fc7cd3e` used for red-first checks; t=0 baseline = tests/test_shared_portfolio.py | I-27 not diagnosed (no abort in the recorded runs); oracle for the sensitivity and XVA analytics (I-50, I-51); scenario import (V-4); strict `xfail` acceptance tests (the acceptance tests were written directly against the finished code instead) |
| 2 Fixes that survive the rewrite | All seven: I-36, I-37, I-38, I-40, I-41, I-48, A-5, red first | — |
| 3 Market and configuration model | `Market`, `CamConfig`, trades without model fields on the market path (refused if set), a request schema for it | 3.3 trade id: optional `trade_ids` only (I-10). 3.4 superseded by decision A-2: one configurable request, no versions (Phase 9.2); today there are still two schemas and routes. 3.5 (move the oracle and demo scenarios out of `engine/`) not done |
| 4 Cross-asset simulation | 4.1–4.6, 4.8: LGM per currency, exact discretization, scenario curves, CAM IR calibration, LGM numeraire, FX and equity | 4.7 superseded by decision A-9: replace the precision mechanism, keep adjustable precision (Phase 9.4). FX/EQ calibration (X-10, Phase 10.3). L3 and L4 tests (I-50) |
| 5 Revaluation with ORE semantics | 5.1–5.6 on the market path; 5.7 moot there | 5.2's "retire Jamshidian" and 5.7 superseded by decision A-1: the Hull-White model stays an option, with its defects closed within it (Phase 9.3). L3 cube and L4 profile parity (I-50); recalibration details (I-49, Phase 10.1) |
| 6 Risk outputs | 6.1 EPE_B/EEPE_B and Basel; 6.2 and 6.3 ORE's sensitivities and Theta; 6.4 market-risk Europeans on Bachelier with the parity test switched | L5 against ORE's sensitivity analytic (I-51). AD as a configurable Greeks method (A-5, Phase 9.7); market-risk engine by configuration (A-8, Phase 9.9) |
| 7 Performance | Not started: its rule is bit-identical parity after correctness is frozen, and I-49 to I-51 may still move numbers | I-53 (the market path doubled the suite's run time), I-21, I-22, P-1, FP32 study |
| 8 Documentation and register | ORE parity page (the "provably equivalent" claim corrected, gate evidence), register, README, architecture and topic pages; updated 2026-09-30 for the owner's decisions | — |
| 9 Configurable engine | Not started. Precondition met: both models exist and each is tested | All of 9.1–9.9 |
| 10 Remaining differences | Not started | 10.1 needs I-50's oracle first |

Found and fixed while implementing: I-39 (bond Theta add-back) and I-52 (cash settlement).

---

## 1. Principles

1. **ORE is the specification.** Every behaviour in the new code cites the ORE/QuantLib
   source it reproduces (file and function) in its docstring. "ORE" below means the vendored
   tree in `reference/ORE/`, which is the version the parity tests run against.
2. **Defaults are ORE's defaults.** Where ORE is configurable, the engine uses ORE's default,
   and exposes the same option under ORE's name when a non-default is needed.
3. **Verify before building.** Where this plan states ORE behaviour not yet confirmed from the
   source, it is listed as a verification gate (§4, V-n). The gate is closed by reading the
   source *and* by an OREApp run that shows the behaviour, recorded in
   `docs/reference/ore-parity.md`. Code for a gated item does not start before its gate closes.
4. **Parity tests use ORE, not a re-derivation.** The reference number comes from an ORE
   engine or an OREApp analytic, never from a formula written again in the test.
5. **No flat-curve-only tests.** Every model or pricer test runs on a sloped curve (3% → 5%)
   as well; flat curves make the drift and convexity terms cancel.
6. **Red first.** A test for a fix is shown failing against the pre-fix code before the fix
   lands, and the register entry records that.
7. **Configurable, not hard-wired** (owner decision, 2026-09-30). Models, simulations, pricing
   engines, sensitivity methods, settlement methods and precision are choices in the run's
   configuration, as in ORE. The engine grows by adding options. Nothing that works is removed
   without a replacement, and there is one request shape for every configuration
   ([compliance/decisions.md](../../compliance/decisions.md) §1).

---

## 2. Target design and ORE correspondence

| # | Component | ORE reference | Target in this engine | Replaces / closes |
|---|---|---|---|---|
| T-1 | Simulation model | `CrossAssetModel` (QuantExt/qle/models/crossassetmodel.hpp), built by `CrossAssetModelBuilder` (OREData/ored/model/crossassetmodelbuilder.cpp) | One LGM state per **currency** (`IrLgm1fPiecewiseConstantParametrization`), Black-Scholes equity and FX components, constant instantaneous correlations | Hull-White short rate with constant `theta` (I-42, I-44) |
| T-2 | Measure | `IrModel::Measure::LGM`, the builder's default when `Measure` is empty | Domestic LGM measure; `BA` offered later as an option only if needed | — |
| T-3 | Discretization | `Discretization` default `Exact` (crossassetmodeldata.cpp, "Fall back to Exact") with `CrossAssetStateProcess` (QuantExt/qle/processes/crossassetstateprocess.cpp) and `CrossAssetAnalytics` (QuantExt/qle/models/crossassetanalytics.hpp) | Exact Gaussian increments: drift and covariance of each step from the model's integrals | Euler-style OU step |
| T-4 | Numeraire | `model->numeraire(0, t, x)`, `CrossAssetModelScenarioGenerator::nextPath` | LGM numeraire of the base currency, `N(t,x) = exp(H x + ½H²ζ) / P(0,t)`; cube stores NPV / N | Discretely accrued bank account on factor 0 (I-45) |
| T-5 | Rate curves on a path | `CrossAssetModelScenarioGenerator::nextPath`: each curve is the model-implied curve `curves_[j]->move(t, x)`, sampled at the simulation tenors, floored at 1e-5 | Discount factors at the simulation-market tenors from the currency's LGM state; every curve of a currency (discount and index forwarding) moves with that currency's state, so basis is deterministic | Independent "rate factors" per curve; cube pillars must equal cashflow times |
| T-6 | Scenario market | `ScenarioSimMarket` with `ScenarioSimMarketParameters` defaults `Interpolation = LogLinear`, `Extrapolation = FlatFwd` | Scenario curves interpolated log-linearly in discount factor between tenors, flat-forward extrapolation | Pillar-exact lookup (`_maturity_indices`) |
| T-7 | Date grid | `DateGrid` (dates; model time on the CAM term structure's day counter) | Simulation grid given as dates; times derived once, ACT/365 as ORE's CAM curve | Float `time_grid` |
| T-8 | Scenario revaluation | `ValuationEngine::buildCube` (OREAnalytics/orea/engine/valuationengine.cpp): every trade repriced with its own t=0 pricing engine on the scenario market; `recalibrateModels()` per scenario, `recalibrate = true` by default | Same: per path and date, each trade priced with its t=0 method on that path's scenario market, model-based trades recalibrated to it (subject to V-1) | Conditioning the Bermudan rollback on a Hull-White rate (I-44) |
| T-9 | Fixings on a path | `FixingManager::applyFixings` (OREAnalytics/orea/simulation/fixingmanager.cpp): a fixing dated in `[previous grid date, current grid date)` is set to the index's forecast fixing at the current grid date on that path | Same rule | Coupons fixing during the simulation priced off a clamped cube (audit M-2, sim half of I-04) |
| T-10 | Cashflows on a path | QuantLib `CashFlow::hasOccurred` against the scenario date | Paid cashflows drop out on each path and date | Paid coupons kept and grown (audit M-2) |
| T-11 | Options after exercise | `OptionWrapper::NPV` / `BermudanOptionWrapper::exercise` (OREData/ored/portfolio/optionwrapper.cpp): exercise checked at the first grid date on or after each contract exercise date; Bermudan exercises when underlying NPV > option NPV on that path; physical settlement then carries the underlying, cash settlement carries the cached value until settlement | Same rule, per path | Options worth 0 after expiry (I-43) |
| T-12 | European swaptions | `EuropeanSwaptionEngineBuilder` → `BlackBachelierSwaptionEngine` on the market swaption volatility (OREData/ored/portfolio/builders/swaption.hpp) | Bachelier/Black on a swaption vol surface, at t=0 and on every path (scenario curve, simulation-market vol surface per V-1) | Hull-White Jamshidian (I-46, I-37, I-41) |
| T-13 | Bermudan/American | `LGMGridSwaptionEngineBuilder` + `LgmBuilder` (OREData/ored/model/lgmbuilder.cpp): per-trade LGM calibrated to a basket built from the trade's own exercise dates and underlying | Per-trade LGM with ORE's basket construction and bootstrap; existing grid rollback (parity ~1e-11) reused unchanged | Caller-supplied shared basket (I-47) |
| T-14 | CAM calibration | `CrossAssetModelBuilder`: IR components bootstrapped to the swaption basket in the simulation config; FX/EQ to options | Per-currency bootstrap to a configured basket, same semantics as ORE's `CalibrationSwaptions` | Uncalibrated simulation vol |
| T-15 | Swaps | `DiscountingSwapEngine`, at-par Ibor coupons, forward over the index spanning time (`IborCouponPricer::initializeCachedData`) | Same, at t=0 and on the scenario market | I-36 |
| T-16 | Bonds | `DiscountingRiskyBondEngine` (QuantExt/qle/pricingengines/discountingriskybondengine.hpp) with zero credit and zero security spread for Treasuries | JAX pricer on the shared curve primitives, priceable on the scenario market | Plain-Python pricer (A-7), I-24, I-39 |
| T-17 | Exposure statistics | `ExposureCalculator` (OREAnalytics/orea/aggregation/exposurecalculator.cpp) | Existing statistics plus ORE's time-weighted EPE_B / EEPE_B | Missing EEPE |
| T-18 | Sensitivities | `SensitivityAnalysis` + `SensitivityCube::delta/gamma` (OREAnalytics/orea/cube/sensitivitycube.cpp): bump-and-revalue on the configured shift tenors (`ShiftScenarioGenerator`), delta by the configured shift scheme, gamma `up − 2·base + down` | ORE's definitions as the reported Greeks, computed by vectorized revaluation; AD kept as a cross-check | AD derivatives as the reported value |
| T-19 | Theta | `SensitivityAnalysis` (sensitivityanalysis.cpp:258): `thetaDate = asof + thetaPeriod` (calendar), new sim market at thetaDate, fixings backfilled, period cashflows added | Same | I-38, I-39 |
| T-20 | Zero-curve extrapolation (t=0) | `InterpolatedZeroCurve::zeroYieldImpl`, default `ContinuousForward` | Flat instantaneous forward past the last pillar | I-48 |

---

## 3. Differences from ORE that remain

| # | Difference | Why it cannot or will not be removed | How it is handled |
|---|---|---|---|
| X-1 | Random numbers are not bit-identical to ORE's | ORE's Sobol direction numbers, Brownian-bridge ordering and state layout would all have to be reproduced; not worth it | Path-level parity uses ORE's own scenarios fed into the engine (§6.2 L3); distribution-level parity is statistical (§6.2 L4) |
| X-2 | FP32 and multi-device execution | Not in ORE; the research goal of this engine | FP64 is the default. Any precision may be chosen for any calculation, regulatory figures included; a combination not yet shown adequate for a figure carries a warning with the evidence (decision D-9, Phase 9.5) |
| X-3 | AD Greeks | Not in ORE | A configurable Greeks method beside ORE's bump-and-revalue, which stays the default (decision A-5, Phase 9.7); a test checks they agree to O(bump²) |
| X-4 | VaR/ES per exposure step (`engine.risk.var_es` on the cube) | ORE has no such figure | Kept, labelled risk-neutral (I-11); market-risk VaR stays in `engine.market_risk`, which is ORE-parity |
| X-5 | Swaption smile | ORE supports vol cubes and SABR; first version takes an ATM normal matrix | The surface has no strike axis, so an option away from the money reads the ATM volatility. Decided: close it (I-54, Phase 10.2) |
| X-6 | ORE's AMC engine | ORE's alternative to classic revaluation; this plan reproduces the classic `ValuationEngine` | Out of scope |
| X-7 | Calibration root-finder | ORE's bootstrap uses its optimizer per instrument; the engine bisects | Same root when it exists; parity test on calibrated parameters to ORE's tolerance |
| X-8 | I-04 pre-t=0 fixings, I-05 SOFR | Blocked on TraderX data and convention agreement | Unchanged: refused, not guessed |

Any further difference found while implementing is added here and to
[known-issues.md](../known-issues.md) with the tag *Difference from ORE*.

---

## 4. Verification gates

Each gate is closed by (a) quoting the ORE source that decides it and (b) an OREApp run that
shows the behaviour, both recorded in `docs/reference/ore-parity.md`.

| Gate | Question | Blocks |
|---|---|---|
| V-1 | In classic simulation, does `LgmBuilder` recalibrate on every scenario (`recalibrateModels` → `LgmBuilder::recalibrate`, its `requiresRecalibration` test), and what swaption vol surface does `ScenarioSimMarket` provide when swaption vols are not simulated (moved reference date, sticky strike or sticky moneyness)? | Phase 5.3, 5.4 |
| V-2 | Sensitivity defaults: shift type, size, `ShiftScheme` default, and the `scaling` applied in `SensitivityCube::delta` | Phase 6.2 |
| V-3 | Theta market roll: the theta sim market is rebuilt at `thetaDate` from the original market; are curves fixed in dates (roll-down) or in tenors? | Phase 6.3 |
| V-4 | Can ORE export per-path scenarios (`simulation/scenariodump`, `xvaanalytic.cpp`) or model states (AMC path data) completely enough to reprice the engine's cube from them? | Phase 1.3, L3 tests |
| V-5 | `LgmBuilder` basket: calibration strategy default, how expiries/terms are derived from the trade (`buildSwaptionBasket`), strikes, and the swap index conventions used | Phase 5.4 |
| V-6 | CAM simulation basket semantics (`CalibrationSwaptions` expiries/terms: fixed terms or co-terminal) and the reversion setting in ORE's example simulation configs | Phase 4.4 |
| V-7 | Exact-discretization drift and covariance between IR, FX and EQ components under the LGM measure (`CrossAssetAnalytics`), including the terms that appear for foreign currencies | Phase 4.2, 4.6 |
| V-8 | Scenario-date cashflow conventions (`includeReferenceDateEvents`, `includeTodaysCashFlows`) during simulation | Phase 5.1 |
| V-9 | Which ORE bond configuration matches a Treasury without credit (reference curve, security spread, settlement days) | Phase 5.5 |
| V-10 | Index forwarding curves on a path use the index's day counter (`t_dc`) while discount curves use model time `t`; confirm and reproduce | Phase 4.3 |

---

## 5. Phases

Sizes: **S** ≤ 3 days, **M** ≤ 2 weeks, **L** > 2 weeks. Each phase ends with the full suite
green (§6.5) and the register updated.

### Phase 0 — Decisions and gates (S)

| Task | Output | Exit |
|---|---|---|
| 0.1 | Record the T-1 … T-20 targets and X-1 … X-8 in `compliance/decisions.md` (shared with Basel P0.1) | Owner sign-off |
| 0.2 | Close V-2, V-3, V-5, V-8, V-9, V-10 (source reading + small OREApp runs) | Each gate recorded |
| 0.3 | Start V-1, V-4, V-6, V-7 (need the Phase 1 oracle) | Plan for each |

### Phase 1 — Safety net and oracles (M)

Nothing in later phases can be judged without these.

| Task | Detail | Test / exit |
|---|---|---|
| 1.1 | Diagnose I-27 (suite hard-aborts) | Five consecutive full runs complete with a summary line |
| 1.2 | Generalize `engine/validation/ore_lgm_oracle.py` to an OREApp oracle for: t=0 NPV, calibration report, sensitivity analytic, XVA/exposure analytic (Basel P0.8) | Each analytic reproduces an `Examples/` expected output |
| 1.3 | Scenario import: read ORE's scenario dump (V-4) into arrays `[path, date, curve tenor]` | Round trip: engine rebuilds ORE's discount factors from the dump exactly |
| 1.4 | Write the acceptance tests for Phases 4–5 now, marked `xfail(strict=True)` with the issue ID: sloped-curve martingale test, exposure-profile parity against ORE, post-exercise parity, fixing-on-path parity | They fail today for the stated reason |
| 1.5 | Freeze the t=0 baseline: record every t=0 ORE-parity test's measured worst case | Baseline file committed; Phases 3–5 must not move any t=0 number except where this plan says so (T-12, T-13) |

### Phase 2 — Fixes that survive the rewrite (S–M)

Each is at a boundary or in code the new design keeps.

| Task | Issue | Change | Test |
|---|---|---|---|
| 2.1 | I-37 | Refuse a non-zero `floating_spread` on `SwaptionConfig` (QuantLib's Jamshidian refuses it). Supported again in 5.2 | Red-first: 100bp spread raised, not priced |
| 2.2 | I-41 | Refuse `hw_a <= 0` for Hull-White pricers (ORE raises) | Red-first on a=0 |
| 2.3 | I-38 | Theta date `evaluation_date + theta_days` (calendar) for swap, European, Bermudan/American; tests' helper to calendar days; add a Friday case | Friday Theta equals ORE's sensitivity analytic |
| 2.4 | I-36 | Divide the projected forward by the index spanning time | ACT/ACT (ICMA) swap equals `DiscountingSwapEngine` to 1e-10 |
| 2.5 | I-40 | Pass `fraction_decimals` through `rate_sensitivity`; separate `try` per outcome | Red-first on the fixture with 4 decimals |
| 2.6 | I-48 | Flat-forward extrapolation in `zero_rate`/`log_discount` | Cashflow past the last pillar equals `ORE.ZeroCurve` |
| 2.7 | A-5 | Move validators to `engine/instruments/_validation.py` | Import-layering test |

Do **not** in this phase: change `market_model.py`, the Hull-White European pricer or the
swap cube kernel beyond the refusals above, or add tests that construct
`RatesConfig(theta=..., initial_rates=...)`.

### Phase 3 — Market and configuration model (M)

One breaking change to the Python configs and the HTTP schema, versioned once.

| Task | Detail |
|---|---|
| 3.1 | `Market`: per currency, named curves (discount, index forwarding) as today's zero curves; swaption vol surface (ATM normal matrix, expiry × tenor); equity spots/vols; FX spots/vols; correlations keyed by factor name as in ORE's `InstantaneousCorrelations` |
| 3.2 | `SimulationConfig`: date grid (T-7), simulation-market tenors per curve (T-6), CAM calibration basket per currency (T-14), reversion per currency (input, as ORE's `Reversion` with `Calibrate=false`), measure (T-2), samples, seed. Removes `theta`, `initial_rates`, `joint_covariance`, per-curve rate factors |
| 3.3 | Trades name their currency and curves/index; no model parameters on trades (A-3). `evaluation_date` required (A-4). Trade id on every trade and result (I-10, Basel P0.6) |
| 3.4 | HTTP: new request/result schema version; old version refused with a message naming the new one. **Superseded (decision A-2, 2026-09-30):** one configurable request, no versions, nothing refused (9.2) |
| 3.5 | Move `demo_scenarios.py` and the oracle out of `engine/` (A-6) and rebuild the demo configs in the new shape |

Tests: schema round trip; every refusal names its field; the t=0 baseline (1.5) is unchanged
for swaps, Bermudans and bonds.

### Phase 4 — Cross-asset simulation (L)

| Task | Detail | Closes |
|---|---|---|
| 4.1 | LGM per currency on the existing `engine.models.lgm` (already ORE-parity for H, ζ, bond price, numeraire) | — |
| 4.2 | Exact discretization for IR-only: drift 0 under own measure, step variance ζ(t_{i+1}) − ζ(t_i), cross-currency covariance from `CrossAssetAnalytics` (V-7); correlated normals by the same Cholesky of the step covariance | I-42, I-44 |
| 4.3 | Scenario curves: model-implied discount factors at the simulation-market tenors, floor 1e-5, index curves with the index day counter (V-10); scenario market interpolation LogLinear, FlatFwd extrapolation | I-42 |
| 4.4 | CAM calibration of each currency's σ(t) to its basket (V-6), reusing the bootstrap in `engine.calibration.lgm` with ORE's basket construction | — |
| 4.5 | Numeraire N(t, x) of the base currency; cube stores NPV / N | I-45 |
| 4.6 | Equity and FX components (`EqBs`, `FxBs` parametrizations), exact discretization under the LGM measure (V-7). Single-currency IR + EQ first; FX second | — |
| 4.7 | Precision: all new code takes `dtype` explicitly; remove the process-global x64 toggle, the pricing lock and the per-precision pool tiers (A-1). **Superseded (decision A-9):** replaced by 9.4, which removes the mechanism only once its replacement works | A-1 |
| 4.8 | Scenario axis leading, whole path on device, no host loops (prepares P-1) | — |

Tests (§6): L1 martingale and curve-repricing on sloped curves; L2 ζ, H, step covariance and
drift against `ORE.CrossAssetModel` (constructible from Python) to 1e-12; L3 the engine's
scenario curves rebuilt from ORE's dumped states equal ORE's scenario discount factors to
1e-12; L4 moments of simulated discount factors and FX/EQ against ORE's simulation within
4 standard errors.

### Phase 5 — Scenario revaluation with ORE semantics (L)

Every trade is priced on each path with its t=0 method on that path's scenario market.

| Task | Detail | Closes |
|---|---|---|
| 5.1 | Swaps on the scenario market: paid cashflows drop out (T-10, V-8), fixings on the path by `FixingManager`'s rule (T-9), historical fixings before t=0 as today | audit M-2, sim half of I-04 |
| 5.2 | Europeans: Bachelier/Black on the vol surface (T-12) at t=0 and per path, spread supported as in QuantLib's Black engines; retire Hull-White Jamshidian from pricing (keep only if a test needs it as an ORE reference). **Retirement superseded (decision A-1):** Jamshidian stays as the Hull-White model's European engine, a non-default option | I-46, I-37 (re-enable spread), I-41 moot |
| 5.3 | Bermudan/American per path: build the trade's LGM on the path's scenario curve, recalibrate to the simulation-market vol surface if V-1 says ORE does, roll back with the existing grid. Vectorized over paths per date. If V-1 shows ORE does not recalibrate, use the t=0 calibration | I-44 (Bermudan half) |
| 5.4 | Per-trade calibration basket exactly as `LgmBuilder` (V-5), at t=0 and per path | I-47 |
| 5.5 | Bonds on the shared curve primitives (A-7), priced per path; Theta with period cashflows | I-24, I-39 |
| 5.6 | Exercise per path by `OptionWrapper`'s rule (T-11), physical and cash settlement | I-43 |
| 5.7 | Remove `x_from_r`/`r_from_x` conditioning and the host-side loop. Moot on the market path; on the Hull-White option it goes with 9.3 | audit P-2 (moot) |

Tests: L2 each pricer on a given scenario curve against the matching ORE engine to 1e-10;
L3 the whole cube from ORE's dumped scenarios against ORE's NPV cube, per trade, path and
date (1e-10 for swaps, Europeans, bonds; Bermudans to ORE's calibration tolerance when
recalibrating); exercise decisions identical path by path; L4 exposure profiles against
ORE's XVA analytic.

### Phase 6 — Risk outputs with ORE definitions (M)

| Task | Detail | Closes |
|---|---|---|
| 6.1 | Time-weighted EPE_B and EEPE_B as `ExposureCalculator` | — |
| 6.2 | Reported Greeks by ORE's definitions (T-18, V-2): shift tenors, shift type and size, delta scheme, gamma `up − 2·base + down`, vega by bumping the swaption vol surface (recalibration follows for Bermudans); AD as a configurable alternative (X-3, 9.7) | — |
| 6.3 | Theta as ORE (T-19, V-3), all instruments | — |
| 6.4 | `engine.market_risk`: Europeans revalued with the Bachelier engine; its ORE-parity test switches its reference from `JamshidianSwaptionEngine` to ORE's Black/Bachelier engine | — |

Tests: every Greek against ORE's sensitivity analytic on a sloped curve, trade by trade, to
1e-8 relative; AD vs ORE agree to O(bump²) (checked by halving the bump).

### Phase 7 — Performance (M, after correctness is frozen)

| Task | Closes |
|---|---|
| 7.1 | Memoize compiled Greek programs keyed on `static_key(prepared)` | I-21 |
| 7.2 | Calibration constants as traced arguments | I-22 |
| 7.3 | Shard the scenario axis across devices | audit P-1 |
| 7.4 | FP32 study on the new path (X-2), per figure; its results are the evidence behind 9.5's warnings (any precision may be run, D-9) | Basel P6 |

Rule: every performance change leaves the Phase 5–6 parity tests bit-identical in FP64.

### Phase 8 — Documentation and register (S)

Rewrite `docs/reference/ore-parity.md` (remove the "provably equivalent" claim, add every
gate's evidence); close I-36 … I-48 and the audit items in the register with red-first
evidence; update README claims (one device until 7.3).

### Phase 9 — Configurable engine (L)

The owner's direction of 2026-09-30 ([compliance/decisions.md](../../compliance/decisions.md)
§1): the engine is configured as ORE is (models in the simulation configuration, an engine per
product in the pricing-engine configuration). Defaults are ORE's defaults, and options are added
without removing what works.

| Task | Decision | Detail | Closes |
|---|---|---|---|
| 9.1 | A-1 | One run configuration naming, per component: the model per currency (LGM default, Hull-White option), the simulation, the pricing engine per product type, the Greeks method, the precision per stage. Defaults reproduce today's market path exactly | — |
| 9.2 | A-2 | One request and one route for every configuration. Today's two request shapes both keep working, translated into configuration; neither is deprecated or refused. Result shapes stay as they are | — |
| 9.3 | A-1 | The Hull-White model as an option of that configuration, with its own defects closed within it: curve-fitted (arbitrage-free) drift, the exact numeraire, `OptionWrapper` exercise, paid flows and fixings on paths, bonds on every path, market-vol Europeans unless configured otherwise, the per-trade basket. Each is tested as on the market path (sloped curves, red first) | I-42 to I-47 and I-24 on the Hull-White model, audit P-2 |
| 9.4 | A-9 | Replace the precision mechanism: x64 enabled once per process, every stage takes an explicit dtype from the precision configuration, every array created with one. Then remove the flag toggling, `_PRICING_LOCK` and the per-precision pool tiers. Adjustable precision is never unavailable during the change | audit A-1, I-55 (mechanism) |
| 9.5 | D-9 | Precision evidence: a table per figure (NPV, exposure profile, VaR/ES, each Greek, calibration) and precision, recording what was validated, how, and at how many paths. A run whose combination is not validated for a figure it reports carries a warning naming the evidence and what is missing. Any combination may be run | I-55 (warnings) |
| 9.6 | A-3, D-10 | Bermudan/American engine: `ShiftHorizon` configurable, parity at 0.5 proven against the oracle (it already accepts a shift horizon), then 0.5 as the default (ORE's). ORE's FD solver as an option | I-32 |
| 9.7 | A-5 | Greeks method per run: bump-and-revalue (default) or AD, on either model | — |
| 9.8 | A-6 | Settlement method as a trade field with ORE's values and defaults (`PhysicalOTC`, `CollateralizedCashPrice`, `ParYieldCurve`, ...). Low priority | — |
| 9.9 | A-8 | Market-risk engine per product from the pricing configuration, not inferred from a trade's fields | — |

Tests: each option's defaults reproduce today's numbers bit for bit (the shared portfolio and
the parity suites); each new option is tested against its ORE reference where ORE has one.

### Phase 10 — Remaining differences from ORE

| Task | Decision | Detail | Closes |
|---|---|---|---|
| 10.1 | X-9 | Reproduce ORE's per-path recalibration details (the parametrization's time grid from the first build; helpers expired on the path date), measured against ORE's simulation through the oracle of I-50 | I-49 |
| 10.2 | X-5 | Swaption smile: a volatility cube (strike axis) in the market, read at each helper's and option's strike as ORE does, with SABR as a later option | I-54 |
| 10.3 | X-10 | FX and equity volatilities calibrated to FX and equity options, as `CrossAssetModelBuilder` does. Not urgent | — |
| 10.4 | X-11 | FX and equity trades on the market path. Not urgent | part of I-07, I-18 |

### Work that can run in parallel

Basel P0, P2 (history acquisition: start now; calendar time) and P3 use t=0 revaluation and
`engine.market_risk`, not the simulation. Basel P1 (FRTB-SA) waits for 5.2, because European
sensitivities change with T-12. Basel P5 (SA-CCR, CVA, IMM) waits for Phase 6. I-08 (job store)
and the TraderX integration are independent.

---

## 6. Testing

### 6.1 Rules

- Run with `.venv/Scripts/python.exe -m pytest`. Take counts from
  `pytest --collect-only -q tests/`; a run counts as green only if the summary line was printed
  and there are no `FAILED`/`ERROR` lines (exit code alone is not evidence).
- Every model and pricer test runs on a flat **and** a sloped curve (3% → 5%), and in FP64;
  FP32 variants where the code path exists.
- Parity references come from ORE engines or OREApp analytics (§1.4). The measured worst case
  of each parity test is recorded in its docstring; the tolerance is set from it with margin,
  not chosen first.
- Every fix is shown red against the pre-fix code before it lands.
- Changes to processes, paths or platform defaults also run in Linux (Docker `python:3.11`,
  as CI).

### 6.2 Layers

| Layer | What | Reference | Tolerance |
|---|---|---|---|
| **L1 Correctness (model properties)** | Martingale `E[P(t,T)/N(t)] = P(0,T)` and `E[FX(t)·P_f(t,T)/N(t)] = FX(0)·P_f(0,T)` at several t, T on sloped curves; scenario curve at t=0 equals the input curve; exact discretization reproduces ζ(t) as the sample variance of x(t); numeraire N(0)=1; put-call parity; monotonicity in vol; zero vol gives intrinsic; exercise decision consistency (never exercise an option worth more alive) | Analytic | Martingale within 4 MC standard errors at 2^16 paths, and the error must shrink as 1/√paths; exact identities to 1e-12 |
| **L2 Component parity** | ζ, H, bond price, numeraire, step covariance/drift (`ORE.CrossAssetModel`); each pricer on a fixed curve: swap (`DiscountingSwapEngine`), European (`BachelierSwaptionEngine`/`BlackSwaptionEngine`), Bermudan/American (OREApp LGM grid, existing oracle), bond (OREApp bond); calibration: basket and calibrated σ against ORE's calibration report | ORE Python bindings, OREApp | 1e-12 (model analytics), 1e-10 relative (pricers), ORE's end-criteria tolerance (calibrated σ) |
| **L3 Path parity** | Feed ORE's dumped scenarios (V-4) to the engine: scenario curves, per-path NPV cube for every trade and date, exercise flags, fixings applied | OREApp XVA analytic with `scenariodump` and cube output | 1e-10 relative per cell (swaps, Europeans, bonds); calibration tolerance for recalibrated Bermudans; exercise flags identical |
| **L4 Distribution parity** | Engine's own simulation vs ORE's with independent RNG: EPE/ENE/EE_B/EEE_B/PFE/EEPE_B profiles; moments of discount factors, FX, EQ | OREApp XVA analytic | Within 4 combined standard errors at every grid date, 2^14 paths each; PFE by a bootstrap interval |
| **L5 Risk parity** | Delta/gamma/vega/theta per trade; market-risk VaR/ES (existing) | OREApp sensitivity analytic; ORE `RiskStatistics` | 1e-8 relative |
| **L6 End to end** | HTTP: submit portfolio (every instrument type, sloped curves, 2 currencies once 4.6 lands) → poll → result schema validates → NPVs equal the direct Python call → exposure profiles equal L4's; TraderX EOD bundles (fixtures) unchanged; demos run; full suite on Windows and Linux | Engine's own paths, published schemas, recorded fixture results | Exact equality between HTTP and direct call; fixture results unchanged byte for byte unless the phase says otherwise |

### 6.3 Portfolio used across layers

One fixed test portfolio in `tests/support/`, on a sloped two-curve market: payer and receiver
swaps (ACT/365 and ACT/ACT (ICMA) legs, one seasoned with a past fixing), an ATM and an OTM
European (one with a spread, after 5.2), a physically and a cash-settled Bermudan, an
American, a Treasury bond with a coupon inside the first simulation step, and an equity
position. Every layer runs on it, so a discrepancy can be followed from L6 down to L2.

### 6.4 Per-phase acceptance

| Phase | Must pass |
|---|---|
| 1 | Oracles reproduce ORE examples; `xfail` acceptance tests fail for the stated reason |
| 2 | Red-first tests for each issue; t=0 baseline otherwise unchanged |
| 3 | Schema tests; t=0 baseline unchanged for swaps, Bermudans, bonds |
| 4 | L1 martingale (the `xfail` from 1.4 now passes, strict), L2 model analytics, L3 curves, L4 moments |
| 5 | L2 pricers, L3 cube and exercise flags, L4 exposure profiles, L6 HTTP |
| 6 | L5 sensitivities and theta; market-risk parity with the new European reference |
| 7 | All of the above bit-identical in FP64 |

### 6.5 Suite hygiene

Parity tests are never marked `slow` (CI's fast tier runs them). L4 statistical tests are
seeded and marked `slow`; their seeds are fixed so a failure is reproducible. A statistical
test that fails is rerun with a second seed before being investigated, and both results are
recorded.

---

## 7. Issue map

| Issue / audit item | Phase |
|---|---|
| I-36, I-38, I-40, I-48, A-5 | 2 (fixed) |
| I-37, I-41 | 2 (refused), 5.2 (supported / moot) |
| I-10, A-3, A-4, A-6 | 3 |
| I-42 (M-1), I-44 (A-2), I-45, A-1 | 4 |
| audit M-2, I-04 (simulation half), I-43 (M-3), I-46, I-47, I-24, I-39, A-7, P-2 | 5 |
| I-21, I-22, P-1 | 7 |
| I-27 | 1 |
| I-32 | 9.6 (decided 2026-09-30: configurable, ORE's default `ShiftHorizon = 0.5` once parity there is proven) |
| I-42 to I-47, I-24 on the Hull-White model; audit P-2, A-2, A-3 | 9.3, 9.1 |
| I-49 | 10.1 (after I-50) |
| I-50, I-51 | 1.2 (oracle for the XVA and sensitivity analytics) |
| I-53 | 7 |
| I-54 | 10.2 |
| I-55, A-1 | 9.4, 9.5 |
| I-04 (pre-t=0 fixings), I-05, I-16, I-18, I-23 | Blocked externally; unchanged |
| I-07, I-08, I-09, I-12 | Independent; any time |
