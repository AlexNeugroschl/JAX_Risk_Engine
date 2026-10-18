# ORE parity validation

How the engine is shown to equal ORE, what is proven, and the work that remains (a second
currency, [F-04](../features.md#f-04)). The work of 2026-10-07 built the ORE simulation oracle
and closed I-50 and I-34; that of 2026-10-12 closed I-32, I-49 and I-73, so every trade of the
shared portfolio equals ORE's simulation on its paths; that of 2026-10-17 built the ORE
sensitivity oracle and closed I-51, I-78 and I-112, so the Greeks and a calibrated option's
market risk equal ORE's too; that of 2026-10-18 built the parity matrix, the NPV and market-risk
oracles and the seeded sweeps and closed I-116, so what is compared beyond the shared portfolio,
and what is not, is stated cell by cell. The formula-by-formula
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
- **Every parity test runs in the fast tier** (CI's). Only its statistical extension (L4) or
  its swept one is `slow`: the seeded sweeps' fixed sample runs in the fast tier, the rest of the
  sweep in the full tier. A failing statistical test is rerun with a second seed before
  investigating, and both results recorded; a failing sweep case is reproduced from its seed
  (`parity_sweep.case(seed).describe()`).
- **Every parity test declares its cells** (`@pytest.mark.ore_parity(cells(...), tolerance=...)`,
  from the trades and configurations it runs), so the
  [parity matrix](../../reference/ore-parity.md#the-parity-matrix) is generated from the tests;
  a public setting, value or analytic without a cell needs a stated reason
  (tests/test_ore_parity_matrix.py).
- **Tolerances of the notional** where a sweep draws values near zero (a deep out-of-the-money
  option), of the value elsewhere.
- **The t=0 baseline** is `tests/test_shared_portfolio.py`: a change that moves one of its
  numbers must say why.

## Test layers

| Layer | What | Reference | Tolerance | Status |
|---|---|---|---|---|
| L1 Model properties | Martingale `E[P(t,T)/N(t)] = P(0,T)` and its FX analogue on sloped curves; scenario curve at t=0 equals the input; ζ(t) equals the sample variance of x(t); N(0)=1; put-call parity; zero vol gives intrinsic; exercise consistency | Analytic | Martingale within 4 MC standard errors at 2^16 paths, shrinking as 1/√paths; identities to 1e-12 | Done (`tests/test_cam.py`) |
| L2 Components | ζ, H, bond price, numeraire, step covariance (`ORE.CrossAssetModel`); each pricer on a fixed or path curve against its ORE engine; calibrated σ against ORE's calibration | ORE bindings, OREApp | 1e-12 analytics; 1e-10 pricers; ORE's end criteria for σ | Done (`tests/test_cam.py`, `test_pricing.py`, `test_ore_lgm_calibration.py`, `test_ore_lgm_parity.py`) |
| L3 Path parity | ORE's paths priced by the engine: each path's state read from ORE's scenario dump rebuilds every simulated curve; each trade's cube against ORE's per path and date; t=0 (ORE's `T0`); the engine's exposure definitions on ORE's own cube | OREApp XVA (`EXPOSURE`, `PFE`) with the scenario dump and a double-precision cube | Curves 1e-10 relative; a trade's cube 1e-10 of its largest value; `T0` and the profiles 1e-10 relative | Done for every trade under both models (`tests/test_ore_xva_parity.py`: measured 1.2e-11, 3.0e-11, 3.9e-11, 1.8e-13); Bermudans and Americans since 2026-10-12, their cubes to 1.9e-11, each LGM calibrated when ORE built the trade on the simulation market of the as-of date and kept on every path, as ORE's exposure simulation keeps it ([I-49](../known-issues.md#i-49)), at ORE's default `ShiftHorizon` ([I-32](../known-issues.md#i-32)); uncalibrated too, to 2e-11 |
| L4 Distribution parity | The engine's simulation against ORE's, independent random numbers: EPE, ENE, PFE per trade and netting set; the mean of every simulated discount factor | OREApp XVA | 4 combined standard errors at each grid date; linear trades at 2^14 paths, options at 2^12 for ORE's time (2^10 until the engine's rollback became one shared operator per step, 2026-10-13, [I-83](../known-issues.md#i-83)); PFE by an order-statistic interval | Done (`tests/test_ore_xva_parity.py`, `slow`). EE_B, EEE_B, EPE_B and EEPE_B follow from EPE (L3 checks the definitions); FX and EQ wait for a second currency (F-04) |
| L5 Risk parity | Delta, Gamma, Vega, Theta per trade; market-risk VaR/ES | OREApp sensitivity analytic; ORE's historical simulation VaR analytic (`HISTSIM_VAR`) end to end; ORE `RiskStatistics` | 1e-11 of the trade's NPV (Greeks, shared portfolio), 1e-10 and 1e-12 of the notional (sweeps); P&L 1e-11 of the notional; VaR 1e-9 relative | Done: VaR/ES (`tests/test_market_risk_ore_parity.py`), a calibrated Bermudan and American recalibrated per scenario since 2026-10-17 ([I-112](../known-issues.md#i-112), 7e-11); every Greek of the shared portfolio against ORE's sensitivity analytic, measured within 2e-14 of the NPV (`tests/test_sensitivity_ore_parity.py`, [I-51](../known-issues.md#i-51)); since 2026-10-18 the seeded sweeps and the settings (`tests/test_ore_parity_sweep.py`, `tests/test_ore_parity_settings.py`, [I-116](../known-issues.md#i-116)): every Greek of random portfolios on random markets and settings, Theta across weekends, coupons, fixings, exercises and months, and every trade's P&L in every scenario against ORE's own market-risk analytic |
| L6 End to end | HTTP submit → poll → schema → NPVs equal the direct call; TraderX fixtures unchanged; demos run; full suite on Windows and Linux | The engine's own paths, published schemas | Exact | Done for one currency (`tests/test_api_portfolio.py`); two currencies wait for [F-04](../features.md#f-04) |

### The shared portfolio

`tests/support/portfolio.py`, on a sloped two-curve market: payer and receiver swaps
(ACT/365 and ACT/ACT (ICMA) legs, one seasoned with a past fixing), an ATM and an OTM
European (one with a spread), a physical and a cash-settled Bermudan, an American, and a
Treasury with a coupon inside the first step. Every layer runs on it, so a discrepancy can
be followed from L6 down to L2. An equity position joins it with [F-04](../features.md#f-04).

## Verification gates

| Gate | Question | Status |
|---|---|---|
| V-1 | Does `LgmBuilder` recalibrate on every scenario, and what swaption vol surface does `ScenarioSimMarket` provide on a path? | Closed (2026-10-12): ORE recalibrates per scenario only where an analytic hands the valuation engine the trades' model builders, its sensitivity analysis and historical P&L, both on the as-of date; its exposure simulation (`XvaAnalytic`) hands it none, or `recalibrateModels = false`, so no option is recalibrated on a path ([I-49](../known-issues.md#i-49), measured against ORE's cube; the engine's market risk holds the calibration, [I-112](../known-issues.md#i-112)). On a path the swaption volatilities are `DynamicSwaptionVolatilityMatrix`'s (`ReactionToTimeDecay`) |
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

## The ORE sensitivity oracle (2026-10-17)

`tests/support/ore_sensitivity_oracle.py` runs ORE's `SENSITIVITY` analytic in-process over the
same in-memory inputs as the other oracles (`tests/support/ore_inputs.py`): ORE's
`sensitivity.xml` and sensitivity simulation market written from the engine's
`SensitivityConfig` (every curve at its tenors, absolute zero-rate shifts at the same tenors,
forward scheme, Gamma; the ATM volatilities simulated and shifted per quote; Theta; models
recalibrated), its report read into the engine's keys.

- **One engine curve is two ORE factors.** ORE's LGM builder discounts its model and its
  calibration helpers on the swap index's discounting index (`IrModelBuilder`:
  `swapIndex_->discountingTermStructure()`), and a bond on its reference curve; in ORE's
  sensitivity market each is an `IndexCurve` factor beside the `DiscountCurve`, though both are
  the engine's one discount curve. Compared against ORE's `DiscountCurve` alone, a calibrated
  Bermudan's 7Y discount Delta was 30 times ORE's and the bond had none. The oracle shifts both
  factors and reads the engine's curve as both moved: ORE's own cross scenario (both up at once)
  gives the joint Delta exactly, and a second run at the negated shift the joint Gamma (the
  up-up cross term standing for the missing down-down one was a third off where the grid's price
  has kinks).
- **ORE's cross-gamma filter names an index curve by its currency** (`IndexCurve/USD` for every
  USD index), so its pair is the discount curve with every index curve.
- **What it found.** Every Delta, Gamma and Vega already equalled ORE's to 2e-14 of the trade's
  NPV once read this way; Theta was 60–290% off on every trade, the engine's Theta market fixed
  in dates where ORE's holds its tenor points at their times as the reference date moves (the
  rule of the simulated paths, I-84). Fixed, every Theta equals ORE's to 1e-13.
- **Cost.** Two OREApp runs, about a minute on the shared portfolio (the cross scenarios
  reprice every option with its recalibration); `tests/test_sensitivity_ore_parity.py` holds two
  tests, each over every trade, so pytest-xdist runs the oracle on two workers, not one per
  trade.

Market risk's calibrated options (I-112) are checked by the LGM oracle (`ore_lgm_swaption_npv`)
calibrating on each shocked scenario's curves, as ORE's historical P&L recalibrates the trades'
model builders under every scenario.

## The parity sweeps and the new oracles (2026-10-18)

**The parity matrix** (tests/support/ore_parity_matrix.py, published in
[ORE Parity](../../reference/ore-parity.md#the-parity-matrix)): every public setting is a path
from the class holding it (`LgmSwaptionEngineConfig.strategy`, `LgmConfig.swap_index.fixed_tenor`)
with the analytics and products it applies to and the values that need a cell (each choice, or
the default and another value); every field of every dataclass reachable from the public
requests and trades is a setting, an input (trade terms and market data, which the sweeps draw,
and identities) or in a class with one reason (the precision policy, decision X-2); every HTTP
route names its analytics. A cell is compared by a test or has a reason of one of three kinds:
ORE cannot express it (the oracles' refusals), not ORE's, or not compared yet (with its item).

**The oracles.** `ore_npv_oracle.py` (ORE's NPV analytic over a portfolio, and the trades ORE
refuses to build, with ORE's messages); `ore_market_risk_oracle.py` (ORE's `HISTSIM_VAR`
analytic end to end over a history of the engine's factors: a simulation market linear in the
zero rate at a one-day tenor and the engine's pillars, which is the engine's curve between them;
log returns of discount factors, which are the engine's absolute zero-rate moves; ORE's windows
on TARGET business days; each trade's P&L, the portfolio's and ORE's VaR report). Monte Carlo
shifts reach it as a history of one-day moves. All the portfolio oracles build their inputs with
`ore_inputs.portfolio_inputs`, which refuses by name what they cannot express (a second currency,
two indices, the Jamshidian engine, two swap-index conventions).

**The sweeps** (tests/support/parity_sweep.py): from one seed, a market (sloped, inverted,
negative, flat, steep first segment, humped; random pillars; a 3M or 6M index on a basis; a
volatility surface whose co-terminal baskets fit), a TARGET business day as of, a portfolio of
every product (strikes 300bp either side, near-expiry Europeans, seasoned trades with one fixing
history, both positions and settlements, every accrual day count, Bermudan exercises inside
periods and past, Americans opened before the as-of date, bonds and bills), the option engines'
every setting, a sensitivity configuration with a Theta roll chosen to cross a weekend or the
portfolio's next coupon, fixing or exercise, and Monte Carlo or historical scenarios. The fixed
sample (6 seeds) takes every enumerated choice in turn.

**What they found**, each fixed on 2026-10-18 with its test red on the code before:

- ORE's builder writes a deal strike with `std::to_string`, six decimals, and calibrates to it
  (`engine.pricing.bermudan.calibration_strike`; 7.9e-9 of a deep in-the-money Bermudan's
  notional).
- ORE ends each calibration helper on the underlying trade's last payment date (`maturity()`),
  the maturity adjusted (`underlying_maturity`; a Saturday maturity under an ACT/ACT (ICMA) swap
  index moved an option 1.7e-4).
- An American whose window holds no reference calibration grid date has no calibrating helper:
  ORE refuses it, the engine priced it on its starting volatility (now refused).
- ORE's Theta adds back no flow: its period flows are read after the roll, where every flow of
  the roll has occurred, so a coupon paid in the roll leaves Theta as it leaves the NPV. The
  engine added it back (I-39, read from the source): 17,000 off on a swap whose coupon paid on the
  Theta date.
- On a Theta date that is not a business day ORE's simulation-market volatility matrix runs from
  the next business day, where an option's variance starts
  (`engine.pricing.context.floating_volatility_date`; a European's Theta 1.06 where ORE's is
  -4.94).
- On ORE's paths (the settings runs of tests/test_ore_xva_parity.py) an American whose window
  opened before the as-of date is exercisable from the day after (`ExerciseBuilder`'s
  `max(today + 1, first)`); the engine's wrapper never exercised it (`contract_exercise_dates`).

The sweep's generator found two of its own faults the same way (fixings drawn per trade where ORE
holds one per index and date; as-of dates on holidays), and four configurations where ORE's
sensitivity analytic disagrees with itself or fails a calibration the engine fits
([I-117](../known-issues.md#i-117)), which it does not draw.

Covering the matrix's last cells (market risk of the calibrated options at ORE's default
30-point grid) found that ORE's historical simulation report and `RiskStatistics` can read
different VaR scenarios at a whole `N p` (90% of ten moves: the report's second worst, the
engine's and `RiskStatistics`' the worst), and that the report does not floor a tail of gains at
zero. The engine keeps `RiskStatistics`' figures, decided by the owner in X-12
(2026-10-18), and `tests/test_ore_parity_settings.py` holds both behaviours; the sweep draws no whole
`N p`.

**Cost.** The fixed sample: about two minutes in one process (each case's ORE sensitivity run
5 to 40 s, its historical simulation 1 to 10 s). The full sweep is most of the full suite's 27
minutes at `-n 6` (2026-10-18); its slowest case, a 32-helper American recalibrated under 128
scenarios, takes 314 s in market risk and 172 s in its Greeks, ORE's run most of each. The
generator bounds its draws for that (at most 128 scenarios, 24 grid points a standard deviation,
36 exercise steps a year); ORE's default 30-point grid is compared in
`tests/test_ore_parity_settings.py` and on the shared portfolio.

## Remaining work

A second currency (FX and equity factors in the simulation, a cross-currency portfolio):
[F-04](../features.md#f-04) extends the oracles.
