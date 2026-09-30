# Decisions

Dated record of the modelling and scope decisions the engine's numbers rest on: the ORE
alignment's targets and remaining differences, and the owner's decisions on how the engine
is configured. The work that implements them is planned in
[docs/planning](../docs/planning/README.md); "roadmap n.m" below is a step of
[roadmap.md](../docs/planning/roadmap.md). The Basel decisions D-1 to D-10
([Basel III plan](../docs/planning/details/basel-iii.md) §3) belong here too (Basel
P0.1); only D-9 and D-10 are recorded so far, because the alignment depends on them.

**Status of an entry.** *Decided* entries were decided by the owner on 2026-09-30. *Engineering
default* entries are unreviewed choices taken so work could proceed. Changing a decided entry
is a tracked change, with a re-run of the evidence it cites. *Implemented* says whether the
code does it today. The design for the configurable engine is
[details/configurable-engine.md](../docs/planning/details/configurable-engine.md).

---

## 1. Direction: a configurable engine (decided 2026-09-30)

The engine is **modular and configurable**, as ORE is. Models, simulations, pricing engines,
sensitivity methods, settlement methods and precision are choices in the run's configuration,
not code paths hard-wired to one another. The engine grows by adding options (models,
instruments, methods) without replacing the ones it has.

- **Defaults are ORE's defaults.** Where ORE has a code default, the engine uses it. Where ORE
  has none and its configuration must name a value, the default is the value ORE's example
  configurations use, and the entry says so.
- **Nothing that works is removed without a replacement.** An option leaves the default when
  something better arrives, but it stays available.
- **One request, many configurations.** The same request shape runs any configuration. There
  are no API versions for different models (A-2).

## 2. Owner decisions (2026-09-30)

| # | Decision | Default | Options | Implemented | Work |
|---|---|---|---|---|---|
| A-1 | Models are options. The Hull-White model stays as a supported option, **not the default and not retired**. The LGM cross-asset model (the market path) is the default | LGM (ORE's `CrossAssetModel`) | LGM; Hull-White | Both exist, as two separate code paths chosen by the type of `PortfolioRequest.market` | Make them options of one configuration (roadmap 1.2, 1.3). The Hull-White option's own defects ([I-42](../docs/planning/known-issues.md#i-42) to [I-47](../docs/planning/known-issues.md#i-47)) are closed within that option, not by removing it |
| A-2 | **One API that reaches every setting.** A single route and a single request whose configuration selects the model, engines and methods, and covers every setting the engine has, robustly: validated before any job starts, unknown fields refused, every refusal naming its field. **No names that look like versions but mean something else:** a version marks a revision of the contract itself, never a model. Nothing that works is deprecated | — | — | No. Today there are two request shapes, one per model, on two routes, and the second is named like a version: `POST /portfolio/price` (Hull-White) and `POST /v2/portfolio/price` with `schema_version: "2"` (market path). Several settings are unreachable over HTTP ([I-56](../docs/planning/known-issues.md#i-56)) | One route, one configurable request; today's routes keep answering, translated into it; a completeness test fails when a configuration setting has no API field (roadmap 4.1; [I-56](../docs/planning/known-issues.md#i-56)) |
| A-9 | **Adjustable precision is required.** The current mechanism (the process-global x64 flag, `_PRICING_LOCK`, one worker pool per precision tier; audit [A-1](../docs/planning/known-issues.md#a-1)) is to be replaced by a better one, and removed only once the replacement works | FP64 everywhere | Any precision per stage (D-9) | Precision is adjustable per stage today, through the current mechanism | Explicit per-stage dtypes with x64 enabled once per process; then remove the flag toggling, the lock and the tiers (roadmap 1.4; [I-55](../docs/planning/known-issues.md#i-55)) |
| A-3 | Bermudan/American engine settings are **configurable, with ORE's defaults**: `ShiftHorizon` (ORE's builder default 0.5, `OREData/ored/portfolio/builders/swaption.cpp`) and the solver (Grid, or ORE's FD solver) | `ShiftHorizon = 0.5`, Grid | `ShiftHorizon` any value; Grid; FD | No. The engine accepts `ShiftHorizon = 0` and the Grid solver only, the configuration where parity with ORE is proven | Implement the shift, prove parity at 0.5 against the oracle, then make 0.5 the default; add the FD solver as an option (roadmap 2.5; [I-32](../docs/planning/known-issues.md#i-32)). Not urgent |
| A-4 | Swaption volatility time decay on a path: **both** `ForwardVariance` and `ConstantVariance` supported | `ForwardVariance` | `ForwardVariance`; `ConstantVariance` | Yes (`CamConfig.swaption_vol_decay`) | None. ORE has no code default: its simulation configuration must name `ReactionToTimeDecay`, and an empty value fails ("Decay mode not recognized", scenariosimmarket.cpp). 329 of the 365 values in ORE's `Examples/` are `ForwardVariance`, so that is the default |
| A-5 | The Greeks method is **configurable**: ORE's bump-and-revalue, or automatic differentiation | Bump-and-revalue (ORE's) | Bump; AD (to explore later) | Bump on the market path; AD on the Hull-White path only | A per-run choice of method, for any model ([F-01](../docs/planning/features.md#f-01)) |
| A-6 | Cash settlement follows ORE: a European by `ParYieldCurve`, a Bermudan/American valued as physical at t=0 (ORE's own approximation). The settlement **method** becomes a configurable trade field later | ORE's `defaultSettlementMethod` | Later: ORE's methods (`PhysicalOTC`, `CollateralizedCashPrice`, `ParYieldCurve`, ...) | Defaults only | [F-01](../docs/planning/features.md#f-01), low priority |
| A-7 | An American's calibration basket on a path keeps the as-of reference grid: ORE builds the model once and recalibrates the same basket | — | — | Yes | Confirm against ORE's simulation once [I-50](../docs/planning/known-issues.md#i-50) exists |
| A-8 | In `engine.market_risk` the European engine is chosen today by the trade's fields (no Hull-White parameters: Bachelier; with them: Jamshidian). Fine for now; the target is the engine chosen **by configuration**, per product | Bachelier (ORE's default) | Bachelier; Jamshidian | By trade fields | Engine per product from the pricing configuration ([F-01](../docs/planning/features.md#f-01)) |
| D-9 | **Precision is a free choice.** Any combination of precisions is allowed for any calculation, regulatory figures included. A run using a combination not yet shown adequate for a figure carries a **warning**: what has been validated for that figure, at how many paths, and what is missing | FP64 | Any | Choice yes; warnings no | Record the evidence per figure and precision, and emit the warnings (roadmap 2.7; [I-55](../docs/planning/known-issues.md#i-55)). Replaces "FP64 only until the FP32 gate passes" |
| D-10 | ORE reference configuration for Bermudans/Americans: as A-3 (configurable; default ORE's `ShiftHorizon = 0.5`) | As A-3 | As A-3 | As A-3 | As A-3 |

## 3. Differences from ORE (X-n)

X-1 to X-8 were set by the ORE alignment plan; X-9 onward were found while implementing it.

| # | Difference | Status | How it is handled |
|---|---|---|---|
| X-1 | Random numbers not bit-identical to ORE's | Engineering default | Reproducing ORE's Sobol direction numbers and bridge ordering is not worth it. Distribution-level checks now; path-level parity from ORE's scenario dump once V-4 closes |
| X-2 | FP32 and multi-device execution | Decided (D-9, A-9) | Not in ORE; the engine's research goal. Any precision is allowed, with warnings where unproven |
| X-3 | AD Greeks | Decided (A-5) | A configurable method beside ORE's bump Greeks, which are the default |
| X-4 | VaR/ES per exposure step | Engineering default | Not in ORE; labelled risk-neutral (I-11) |
| X-5 | ATM normal swaption matrix only, no smile | **Decided: close it** | [I-54](../docs/planning/known-issues.md#i-54); roadmap 2.6 |
| X-6 | ORE's AMC engine | Engineering default | Out of scope for now: classic revaluation only. A candidate simulation option under the direction in §1 |
| X-7 | Calibration root-finder (bisection vs ORE's optimizer per helper) | Engineering default | Same root when it exists; calibrated Bermudans equal ORE to ≤ 4e-11 |
| X-8 | Pre-t=0 fixings (I-04), SOFR (I-05) | Engineering default | Blocked externally; refused, not guessed |
| X-9 | Per-path recalibration differs from ORE's in two details, unconfirmed by an ORE simulation | **Decided: close it** | [I-49](../docs/planning/known-issues.md#i-49); roadmap 2.4, after the ORE simulation oracle ([I-50](../docs/planning/known-issues.md#i-50)) |
| X-10 | FX and equity volatilities are constant inputs, not calibrated to FX/equity options | **Decided: close eventually, not urgent** | [F-04](../docs/planning/features.md#f-04) |
| X-11 | No equity or FX trade on the market path | **Decided: close eventually, not urgent** | [F-04](../docs/planning/features.md#f-04); related to [I-07](../docs/planning/known-issues.md#i-07) and [I-18](../docs/planning/known-issues.md#i-18) |

## 4. ORE alignment targets (T-1 to T-20)

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
| T-8 | Every trade repriced with its t=0 engine on each path; recalibration on | Done (X-9 open) | `engine.valuation` | tests/test_valuation.py |
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
