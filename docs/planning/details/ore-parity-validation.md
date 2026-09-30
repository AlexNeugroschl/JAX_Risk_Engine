# ORE parity validation

How the engine is shown to equal ORE, what is proven, and the work that remains:
[I-50](../known-issues.md#i-50), [I-51](../known-issues.md#i-51),
[I-49](../known-issues.md#i-49) and [I-34](../known-issues.md#i-34) (roadmap
[2.2 – 2.4](../roadmap.md#stage-2--correctness-and-precision)). The formula-by-formula
mapping is in [ORE Parity](../../reference/ore-parity.md); the differences from ORE that remain
by decision (X-1 to X-11) are in [compliance/decisions.md](../../../compliance/decisions.md) §3.

## Rules

- **The reference comes from ORE**, an ORE engine or an OREApp analytic run in-process,
  never from a formula written again in the test.
- **Verify ORE's behaviour before building on it.** A behaviour not yet confirmed from the
  source is a gate (below); it closes by quoting the source *and* a run that shows it,
  recorded in [ORE Parity](../../reference/ore-parity.md).
- **Sloped curves.** Every model and pricer test runs on a curve rising 3% → 5% as well as
  flat, in FP64, and in FP32 where the code path exists.
- **Tolerances from measurements.** Record each parity test's measured worst case in its
  docstring and set the tolerance from it with margin, never first.
- **Parity tests are never marked `slow`** (CI's fast tier runs them). Statistical tests
  are seeded and marked `slow`; a failing one is rerun with a second seed before
  investigating, and both results recorded.
- **The t=0 baseline** is `tests/test_shared_portfolio.py`: a change that moves one of its
  numbers must say why.

## Test layers

| Layer | What | Reference | Tolerance | Status |
|---|---|---|---|---|
| L1 Model properties | Martingale `E[P(t,T)/N(t)] = P(0,T)` and its FX analogue on sloped curves; scenario curve at t=0 equals the input; ζ(t) equals the sample variance of x(t); N(0)=1; put-call parity; zero vol gives intrinsic; exercise consistency | Analytic | Martingale within 4 MC standard errors at 2^16 paths, shrinking as 1/√paths; identities to 1e-12 | Done (`tests/test_cam.py`) |
| L2 Components | ζ, H, bond price, numeraire, step covariance (`ORE.CrossAssetModel`); each pricer on a fixed or path curve against its ORE engine; calibrated σ against ORE's calibration | ORE bindings, OREApp | 1e-12 analytics; 1e-10 pricers (1e-8 for a Bermudan recalibrated on the path); ORE's end criteria for σ | Done (`tests/test_cam.py`, `test_valuation.py`, `test_ore_lgm_calibration.py`, `test_ore_lgm_parity.py`) |
| L3 Path parity | ORE's dumped scenarios fed to the engine: scenario curves, NPV cube per trade/path/date, exercise flags, fixings | OREApp XVA with `scenariodump` and cube output | 1e-10 relative per cell; calibration tolerance for recalibrated Bermudans; exercise flags identical | Open: gate V-4 ([I-50](../known-issues.md#i-50)) |
| L4 Distribution parity | The engine's simulation against ORE's, independent random numbers: EPE, ENE, EE_B, EEE_B, EPE_B, EEPE_B, PFE; moments of discount factors, FX, EQ | OREApp XVA | 4 combined standard errors at each grid date, 2^14 paths each; PFE by bootstrap interval | Open ([I-50](../known-issues.md#i-50)) |
| L5 Risk parity | Delta, Gamma, Vega, Theta per trade; market-risk VaR/ES | OREApp sensitivity analytic; ORE `RiskStatistics` | 1e-8 relative | VaR/ES done (`tests/test_market_risk_ore_parity.py`); sensitivities open ([I-51](../known-issues.md#i-51)) |
| L6 End to end | HTTP submit → poll → schema → NPVs equal the direct call; TraderX fixtures unchanged; demos run; full suite on Windows and Linux | The engine's own paths, published schemas | Exact | Done for one currency (`tests/test_api_market_path.py`); two currencies wait for [F-04](../features.md#f-04) |

### The shared portfolio

`tests/support/portfolio.py`, on a sloped two-curve market: payer and receiver swaps
(ACT/365 and ACT/ACT (ICMA) legs, one seasoned with a past fixing), an ATM and an OTM
European (one with a spread), a physical and a cash-settled Bermudan, an American, and a
Treasury with a coupon inside the first step. Every layer runs on it, so a discrepancy can
be followed from L6 down to L2. An equity position joins it with [F-04](../features.md#f-04).

## Verification gates

| Gate | Question | Status |
|---|---|---|
| V-1 | Does `LgmBuilder` recalibrate on every scenario, and what swaption vol surface does `ScenarioSimMarket` provide on a path? | Half closed: recalibration confirmed and implemented; the two differing details are [I-49](../known-issues.md#i-49) |
| V-2 | Sensitivity defaults: shift type, size, scheme, scaling | Closed |
| V-3 | Theta market roll | Closed |
| V-4 | Can ORE export per-path scenarios (`scenariodump`) or model states completely enough to reprice the engine's cube from them? | **Open**; blocks L3 |
| V-5 | `LgmBuilder` basket construction | Closed (OREApp run) |
| V-6 | CAM simulation basket semantics | Closed |
| V-7 | Exact-discretization drift and covariance across IR, FX, EQ | Closed |
| V-8 | Scenario-date cashflow conventions | Closed |
| V-9 | ORE bond configuration for a Treasury without credit | Closed |
| V-10 | Index curves on a path use the index day counter | Closed |

## Remaining work

**Step 2.2 — an OREApp XVA oracle ([I-50](../known-issues.md#i-50)).** Generalize the in-process
oracle (moved to `tests/support/` by roadmap 1.1) from pricing analytics to an XVA run:
write `simulation.xml` from a `CamConfig`, the portfolio from the shared portfolio's trades,
and read ORE's cube and exposure reports. Check it first by reproducing an ORE
`Examples/Exposure` expected output. Then:

1. L4 on the shared portfolio with independent random numbers (X-1: ORE's Sobol direction
   numbers and bridge ordering are not reproduced).
2. Close V-4: import ORE's scenario dump into `[path, date, curve tenor]` arrays and check
   the round trip rebuilds ORE's discount factors exactly; then L3.
3. While in the oracle, fix [I-34](../known-issues.md#i-34): hand ORE a curve its zero-curve
   build does not re-read (a segment type that keeps quotes, if ORE has one), or solve for the
   as-of quote that its rebuild maps onto `z0`. Until then keep oracle checks off sloped
   first segments.

**Step 2.3 — sensitivities ([I-51](../known-issues.md#i-51)).** Run ORE's sensitivity analytic
through the same oracle on the shared portfolio; compare per trade, factor and tenor to 1e-8
relative. Also confirms A-7 (an American's path basket keeps the as-of reference grid).

**Step 2.4 — recalibration details ([I-49](../known-issues.md#i-49)).** Compare a Bermudan's cube
with ORE's (L3, or L4 if V-4 stays closed), then reproduce ORE's two details: the
parametrization's time grid kept from the as-of build, and helpers whose expiry has passed
still passed on later dates.
