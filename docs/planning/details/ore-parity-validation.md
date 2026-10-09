# ORE parity validation

How the engine is shown to equal ORE, what is proven, and the work that remains:
[I-51](../known-issues.md#i-51) and [I-49](../known-issues.md#i-49), both in the
[roadmap's core](../roadmap.md#stage-3--the-core). The work of
2026-10-07 built the ORE simulation oracle and closed I-50 and I-34. The formula-by-formula
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
| L3 Path parity | ORE's paths priced by the engine: each path's state read from ORE's scenario dump rebuilds every simulated curve; each trade's cube against ORE's per path and date; t=0 (ORE's `T0`); the engine's exposure definitions on ORE's own cube | OREApp XVA (`EXPOSURE`, `PFE`) with the scenario dump and a double-precision cube | Curves 1e-10 relative; a trade's cube 1e-10 of its largest value; `T0` and the profiles 1e-10 relative | Done for swaps, Europeans and bonds under both models (`tests/test_ore_xva_parity.py`: measured 1.2e-11, 3.0e-11, 3.9e-11, 1.8e-13); Bermudans and Americans 0.5–4% apart, strict expected failures until [I-49](../known-issues.md#i-49) is closed |
| L4 Distribution parity | The engine's simulation against ORE's, independent random numbers: EPE, ENE, PFE per trade and netting set; the mean of every simulated discount factor | OREApp XVA | 4 combined standard errors at each grid date; linear trades at 2^14 paths, options at 2^10 ([I-83](../known-issues.md#i-83)); PFE by an order-statistic interval | Done (`tests/test_ore_xva_parity.py`, `slow`). EE_B, EEE_B, EPE_B and EEPE_B follow from EPE (L3 checks the definitions); FX and EQ wait for a second currency (F-04) |
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
| V-4 | Can ORE export per-path scenarios (`scenariodump`) or model states completely enough to reprice the engine's cube from them? | Closed (2026-10-07): the dump's numeraire gives each path's LGM state, which rebuilds every dumped discount and index curve to 1.2e-11 (`tests/support/ore_xva_oracle.py::implied_states`) |
| V-5 | `LgmBuilder` basket construction | Closed (OREApp run) |
| V-6 | CAM simulation basket semantics | Closed |
| V-7 | Exact-discretization drift and covariance across IR, FX, EQ | Closed |
| V-8 | Scenario-date cashflow conventions | Closed |
| V-9 | ORE bond configuration for a Treasury without credit | Closed |
| V-10 | Index curves on a path use the index day counter | Closed |

## The ORE simulation oracle (2026-10-07)

`tests/support/ore_xva_oracle.py` runs ORE's `EXPOSURE` and `PFE` analytics in-process over the
engine's market and trades, with `simulation.xml` written from the engine's `CamConfig`, and
reads back ORE's double-precision NPV cube and its `T0`, the scenario dump (numeraire and every
simulated curve) and the exposure reports. Its inputs are built by `tests/support/ore_inputs.py`,
shared with the single-swaption NPV oracle (`ore_lgm_oracle.py`). Checked first by running ORE's
own `Examples/Exposure` swap both ways: the in-memory run equals ORE's file-driven run of the
example to the report's rounding (the release's `ExpectedOutput` for it was written by an
earlier build and differs by up to 30% in EPE, so ORE's file-driven run is the reference).

What the oracle found, each fixed on 2026-10-07 with its test shown red on the code before:

- The simulation market holds each curve's tenor points at the tenors' times from the **as-of
  date** (`ScenarioSimMarket::addYieldCurve` builds the curve once, on those times, and moves
  its reference date), while the scenario generator computes them at the times from each
  simulation date. The engine held each value where it was computed: on a 6M tenor two days
  apart, and up to 0.45% of a swap's path values off ORE's cube ([I-84](../known-issues.md#i-84)).
- ORE's cube starts from each trade's value on the simulation market of the as-of date (the
  curves sampled at the tenors), not on today's market: up to 4.2% at t=0 on the shared
  portfolio ([I-85](../known-issues.md#i-85)).
- A cash-settled option's maturity, where its time-weighted exposure stops, is its last
  exercise date (`Swaption::build`), not its underlying's ([I-86](../known-issues.md#i-86)).
- I-34: ORE's zero-curve rebuild reads the t=0 rate at t = 1e-4 (`flattenPiecewiseCurve` ->
  `zeroRate(asof)`, QuantLib's `dt`); the oracle hands it the as-of quote that maps onto the
  engine's, so any curve, sloped before its first pillar or not, reaches ORE as the engine's.

Not compared yet: the Basel EPE/EEPE scalars (ORE writes them in the XVA report, which needs
credit curves; the time-weighted profiles they are read from are compared), and a second
currency (FX, EQ; [F-04](../features.md#f-04) extends the oracle).

## Remaining work

**Recalibration details ([I-49](../known-issues.md#i-49)).** Reproduce ORE's two
details against `tests/test_ore_xva_parity.py`, whose Bermudan/American L3 cases are strict
expected failures until then: the parametrization's time grid kept from the as-of build
(the pattern of I-84), and helpers whose expiry has passed still passed on later dates. The
same cases confirm A-7 (an American's path basket keeps the as-of reference grid).

**Sensitivities ([I-51](../known-issues.md#i-51)).** Run ORE's sensitivity analytic
through the same oracle on the shared portfolio; compare per trade, factor and tenor to 1e-8
relative. Also confirms A-7 (an American's path basket keeps the as-of reference grid).


