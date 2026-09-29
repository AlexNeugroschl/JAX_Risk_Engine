# Decisions

Dated record of the modelling and scope decisions the engine's numbers rest on: the ORE
alignment's targets and remaining differences
([ORE alignment plan](../docs/planning/ore-alignment-plan.md) Phase 0.1), and the decisions taken
while implementing it. The Basel decisions D-1 to D-10
([Basel III plan](../docs/planning/basel-iii-compliance-plan.md) §3) will be recorded here too
(Basel P0.1); only D-9 and D-10 are recorded so far, because the alignment depends on them.

**Sign-off.** Every entry below is the engineering default, taken so work could proceed.
**None has owner sign-off yet**; each row's *Sign-off* column says so. Changing an entry later is
a tracked change with a re-run of the evidence it cites.

---

## 1. ORE alignment targets (T-1 to T-20)

Recorded 2026-09-29. *Where* names the implementing module; *Evidence* the test that checks it
against ORE.

| # | Target | Status | Where | Evidence |
|---|---|---|---|---|
| T-1 | One LGM per currency, Black-Scholes FX and equity, constant correlations | Done | `engine.simulation.cam` | tests/test_cam.py (ζ, H, bond, numeraire, state process vs ORE) |
| T-2 | LGM measure | Done (only measure offered) | `engine.simulation.cam` | tests/test_cam.py |
| T-3 | Exact discretization | Done | `cam.step_moments`, `evolve_states` | tests/test_cam.py (covariance = loading integral; one step = `IrLgm1fStateProcess`) |
| T-4 | LGM numeraire; cube stores NPV, deflated by it | Done | `scenario_market.lgm_numeraire`, `market_path._exposures` | tests/test_portfolio_market_path.py (bill EE_B identity) |
| T-5 | Model-implied curves at the sim tenors, floored at 1e-5, basis deterministic | Done | `scenario_market.implied_log_discounts` | tests/test_cam.py (martingales, floor, basis) |
| T-6 | LogLinear interpolation, FlatFwd extrapolation | Done | `models.curves.DiscountCurve` | tests/test_valuation.py (path curves handed to ORE as `DiscountCurve`) |
| T-7 | Simulation grid as dates | Done | `simulation.config.CamConfig.dates` | — |
| T-8 | Every trade repriced with its t=0 engine on each path; recalibration on | Done (V-1 open, see X-9) | `engine.valuation` | tests/test_valuation.py |
| T-9 | `FixingManager` rule on paths | Done | `valuation.legs.path_fixings` | tests/test_valuation.py (1e-12) |
| T-10 | Paid flows drop out (`hasOccurred`) | Done | `valuation.legs` | tests/test_valuation.py |
| T-11 | `OptionWrapper` exercise, physical and cash | Done | `valuation.options` | tests/test_valuation.py |
| T-12 | Europeans on `BlackMultiLegOptionEngine` (Bachelier) | Done, t=0, paths and market risk | `valuation.european`, `risk.price_functions` | tests/test_valuation.py, test_market_risk_ore_parity.py |
| T-13 | Per-trade LGM, ORE's basket and bootstrap | Done | `valuation.bermudan`, `calibration.ore_lgm` | tests/test_ore_lgm_calibration.py (1e-9), test_valuation.py |
| T-14 | CAM IR calibration to the configured basket | Done for IR; FX/EQ not calibrated (X-10) | `calibration.cam` | tests/test_valuation.py (`test_the_cam_is_calibrated_to_its_basket`) |
| T-15 | Swaps with at-par coupons over the index spanning time | Done | `valuation.legs`, `instruments.swap` (I-36) | tests/test_valuation.py, test_trade_dates.py |
| T-16 | Bonds by discounting, no credit | Done | `valuation.portfolio.bond_legs` | tests/test_shared_portfolio.py |
| T-17 | Time-weighted EPE_B/EEPE_B, Basel horizon | Done | `risk.exposure` | tests/test_portfolio_market_path.py |
| T-18 | ORE's bump-and-revalue sensitivities | Done; not yet against an OREApp sensitivity run (I-51) | `risk.sensitivities` | tests/test_sensitivities.py |
| T-19 | Theta: calendar roll, backfilled fixings, period flows | Done | `risk.sensitivities` | tests/test_sensitivities.py |
| T-20 | ContinuousForward zero-curve extrapolation | Done | `models.curves.zero_rate` (I-48) | tests/test_curves.py |

## 2. Differences from ORE that remain (X-n)

X-1 to X-8 are the plan's; X-9 onward were found while implementing it.

| # | Difference | Why | How it is handled | Sign-off |
|---|---|---|---|---|
| X-1 | Random numbers not bit-identical to ORE's | Reproducing ORE's Sobol direction numbers and bridge ordering is not worth it | Distribution-level checks; path-level parity needs ORE's scenario dump (V-4, not done) | Pending |
| X-2 | FP32 and multi-device execution | The engine's research goal | Opt-in; FP64 is the default and the only precision for parity (D-9) | Pending |
| X-3 | AD Greeks | Not in ORE | Kept beside ORE's bump Greeks (`engine.risk.greeks`, legacy path); `test_sensitivities` checks bump = AD to the bump's order | Pending |
| X-4 | VaR/ES per exposure step | Not in ORE | Labelled risk-neutral (I-11) | Pending |
| X-5 | ATM normal swaption matrix only (no smile) | First version | `SwaptionVolSurface` has no strike axis, so a smile cannot be given | Pending |
| X-6 | ORE's AMC engine | Out of scope | Classic revaluation only | Pending |
| X-7 | Calibration root-finder (bisection vs ORE's optimizer per helper) | Same root when it exists | Calibrated Bermudans equal ORE to ≤ 4e-11 | Pending |
| X-8 | Pre-t=0 fixings (I-04), SOFR (I-05) | Blocked externally | Refused, not guessed | Pending |
| X-9 | Per-path recalibration of Bermudans/Americans: the mechanics (the parametrization's time grid, helpers already expired on the path date) and the `DynamicSwaptionVolatilityMatrix` transcription are read from source, not confirmed by an ORE simulation | ORE's dynamic vol matrix and CAM state process have no Python constructor; confirming needs an OREApp XVA run (V-1) | Implemented as read (`valuation.bermudan`); registered as I-49 | Pending |
| X-10 | FX and equity volatilities are constant inputs, not calibrated to FX/equity options | No FX or equity option market in `Market` yet | `CamConfig.fx_volatilities` / `equity_volatilities` | Pending |
| X-11 | No equity or FX trade on the market path | No such instrument yet (I-07, I-18) | The CAM simulates FX and equity; trades convert with the path FX | Pending |

## 3. Decisions taken while implementing

| # | Decision | Rationale | Date | Sign-off |
|---|---|---|---|---|
| A-1 | The Hull-White path stays, beside the market path, not deleted | It works, has its own tests and callers (HTTP v1, TraderX EOD, `engine.market_risk`); the market path is the default for new work | 2026-09-29 | Pending |
| A-2 | HTTP: version 1 (`POST /portfolio/price`) kept for the Hull-White path; version 2 at `POST /v2/portfolio/price`. **Deviates from plan 3.4**, which refuses the old version | Refusing v1 would delete a working path (A-1) | 2026-09-29 | Pending |
| A-3 | The Bermudan/American engine uses `ShiftHorizon = 0` (Basel D-10), not ORE's builder default 0.5 | Parity is proven at 0 (I-32); `LgmSwaptionEngineConfig` refuses another value | 2026-09-29 | Pending |
| A-4 | Swaption volatilities on a path decay by `ForwardVariance` by default (`ConstantVariance` available) | The value ORE's example configurations use for non-simulated swaption volatilities; ORE has no code default | 2026-09-29 | Pending |
| A-5 | Reported Greeks are ORE's bump-and-revalue on the sensitivity simulation market, not AD | ORE's definitions (T-18); AD stays on the legacy path | 2026-09-29 | Pending |
| A-6 | A cash-settled European uses `ParYieldCurve`, ORE's default method for it; a cash-settled Bermudan/American is priced as the physical one at t=0 | ORE's `defaultSettlementMethod`; ORE's LGM engine approximates `ParYieldCurve` by `CollateralizedCashPrice` for Bermudans (swaption.cpp) | 2026-09-29 | Pending |
| A-7 | An American's calibration basket on a path keeps the as-of reference grid | ORE builds the model once at the as-of date and recalibrates the same basket | 2026-09-29 | Pending |
| A-8 | In `engine.market_risk`, a European without Hull-White parameters is priced with the Bachelier engine; one with them keeps Jamshidian (plan 6.4) | Moves the market-risk path to ORE's engine without breaking existing requests | 2026-09-29 | Pending |
| A-9 | The process-global x64 flag, the pricing lock and the per-precision pools stay (plan 4.7 / A-1 not done) | The legacy path needs them; the market-path code takes `dtype` explicitly | 2026-09-29 | Pending |

## 4. Basel decisions needed by the alignment

| # | Decision | Default | Date | Sign-off |
|---|---|---|---|---|
| D-9 | Precision of regulatory runs | FP64 only until the FP32 gate passes per figure | 2026-09-29 | Pending |
| D-10 | ORE reference configuration for Bermudans/Americans (I-32) | Grid solver, `ShiftHorizon = 0` (A-3) | 2026-09-29 | Pending |
