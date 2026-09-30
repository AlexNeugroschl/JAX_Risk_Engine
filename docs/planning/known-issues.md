# Known Issues

Open defects and important shortcomings: what each does to a number or a caller, and what
closing it takes. The order of work is in [roadmap.md](roadmap.md); the rules for this file
(statuses, severities, how to add and close an entry) are in [README.md](README.md). Fixed
issues keep one line in the [closed ledger](#closed).

Read this before trusting an exposure profile, a number off the Hull-White model, or a
result served by the EOD routes: [I-57](#i-57) can return another submission's result, and
the Hull-White model still has the defects the default market path fixed
([I-42](#i-42) to [I-47](#i-47)).

## Verification status

Last full run, 2026-09-30, on the code after roadmap 1.1 (I-65): **2,319 passed, 0 failed**
on Windows (40m43s) and **2,318 passed, 1 skipped, 0 failed** in a Linux `python:3.11`
container on 4 cores (43m57s; the skip is `reference/traderX`, absent in the container).
Both ran the complete suite (`.venv/Scripts/python.exe -m pytest tests/`), 2,319 collected
(2,320 before: the I-65 layering test, `tests/test_demos.py` and a curve test added; the
old `var_es` demo test, a test comparing `hull_white.A` with a wrapper of itself and two
tests of the removed `_initial_log_discount` wrapper removed),
summary line printed. The fast tier (`-m "not slow"`) is not a full verification and is
never recorded here. Rules: [README.md](README.md#verification-rules).

## Summary

| ID | Issue | Sev. | Status | Category | Stage |
|---|---|---|---|---|---|
| [I-04](#i-04) | Aged swaps: the Hull-White model keeps paid flows; TraderX exports no past fixings | High | PARTIAL | Correctness | 1.3; external |
| [I-05](#i-05) | No faithful USD-SOFR / ACT-360 swap construction | High | OPEN | Scope | External |
| [I-07](#i-07) | No corporate bond, equity or listed-option pricer | Medium | OPEN | Scope | By demand |
| [I-08](#i-08) | Portfolio job store in memory; a running EOD attempt is lost on restart | Medium | PARTIAL | API | 4.2 |
| [I-09](#i-09) | Whole scenario cube serialized into the JSON response | Medium | OPEN | API | 4.1 |
| [I-10](#i-10) | No trade identity on the configs; results keyed by position | Medium | OPEN | API | 1.3, 4.1 |
| [I-12](#i-12) | `/version` reports the dispatcher's backend, not the worker's device | Low | OPEN | Correctness | 2.8 |
| [I-16](#i-16) | `rateSensitivity` is parallel-only | Medium | OPEN | Scope | External |
| [I-18](#i-18) | No equity spot or FX source; equity positions refused | Medium | OPEN | Scope | External |
| [I-21](#i-21) | Hull-White Greeks recompile 23 XLA programs per call | Medium | OPEN | Performance | 3.3 |
| [I-22](#i-22) | Hull-White calibration recompiles 8 XLA programs per call | Low | OPEN | Performance | 3.3 |
| [I-23](#i-23) | `accrualBasis` strictness rests on an unconfirmed reading | Medium | ASSUMPTION | API | 4.3 |
| [I-24](#i-24) | Bonds have no scenario NPV on the Hull-White model | Medium | PARTIAL | Correctness | 1.3 |
| [I-27](#i-27) | Long full-suite runs can hard-abort inside XLA | Medium | OPEN | Tooling | 5.1 |
| [I-32](#i-32) | Bermudan/American engine only at `ShiftHorizon = 0`, not ORE's default 0.5 | Medium | OPEN | Correctness | 2.5 |
| [I-34](#i-34) | The ORE oracle's curve differs before the first pillar | Low | OPEN | Validation | 2.2 |
| [I-42](#i-42) | Hull-White: simulated curves not arbitrage-free against the input curve | High | PARTIAL | Correctness | 1.3 |
| [I-43](#i-43) | Hull-White: options worth zero after expiry instead of becoming the swap | High | PARTIAL | Correctness | 1.3 |
| [I-44](#i-44) | Hull-White: scenario pricing mixes Hull-White and LGM | Medium | PARTIAL | Correctness | 1.3 |
| [I-45](#i-45) | Hull-White: numeraire is a discretely accrued bank account | Medium | PARTIAL | Correctness | 1.3 |
| [I-46](#i-46) | Hull-White: Europeans priced off the model vol, not the market vol | Medium | PARTIAL | Correctness | 1.3 |
| [I-47](#i-47) | Hull-White: calibration basket is not the one ORE builds for the trade | Medium | PARTIAL | Correctness | 1.3 |
| [I-49](#i-49) | Per-path recalibration differs from ORE's in two details | Medium | OPEN | Correctness | 2.4 |
| [I-50](#i-50) | No path- or distribution-level parity test against an ORE simulation | Medium | OPEN | Validation | 2.2 |
| [I-51](#i-51) | Sensitivities not checked against ORE's sensitivity analytic | Medium | OPEN | Validation | 2.3 |
| [I-53](#i-53) | The market path is slow (full suite 23 → 46 min) | Medium | OPEN | Performance | 3.1 |
| [I-54](#i-54) | No swaption smile: options away from the money read the ATM vol | Medium | OPEN | Correctness | 2.6 |
| [I-55](#i-55) | Precision switched by a process-global flag; unproven combinations not flagged | Medium | OPEN | Architecture | 1.4, 2.7 |
| [I-56](#i-56) | The API cannot reach every setting; two routes named like versions | Medium | OPEN | API | 4.1 |
| [I-57](#i-57) | EOD: a cached result is served before the submission id is checked | High | OPEN | API | 2.1 |
| [I-58](#i-58) | EOD: two concurrent submissions of one workload both execute | Medium | OPEN | API | 2.1 |
| [I-59](#i-59) | EOD: `calculations` and `reportingCurrency` accepted, keyed, then ignored | Medium | OPEN | API | 2.1 |
| [I-60](#i-60) | EOD result schema has no stated policy on added fields | Low | ASSUMPTION | API | 4.3 |
| [I-61](#i-61) | Nothing runs on more than one device | Medium | OPEN | Performance | 3.2 |
| [I-62](#i-62) | Hull-White Bermudan/American scenario pricing runs on the host | Low | OPEN | Performance | 1.3 |
| [I-63](#i-63) | Hull-White trade configs carry copies of model parameters | Medium | OPEN | Architecture | 1.3 |
| [I-64](#i-64) | A trade's evaluation date defaults to ORE's thread-local global | Medium | OPEN | Correctness | 1.3 |
| [I-66](#i-66) | No linter or type checker | Low | OPEN | Tooling | 5.2 |
| [I-67](#i-67) | Test modules import each other and repeat fixtures | Low | OPEN | Tooling | 5.3 |
| [I-68](#i-68) | Models and engines are separate code paths, not options of one configuration | Medium | OPEN | Architecture | 1.2 |

**Paths.** The *market path* is `price_portfolio` on a `Market` (`engine.portfolio.market_path`,
HTTP `POST /v2/portfolio/price`): ORE's LGM cross-asset model and valuation, the default.
The *Hull-White model* is `price_portfolio` on a `SimulationConfig`
(`engine.portfolio.request`, HTTP `POST /portfolio/price`): the original simulation, kept as a
supported non-default option (decision A-1 in
[compliance/decisions.md](../../compliance/decisions.md)). PARTIAL entries are closed on the
first and open on the second.

---

## Correctness

<a id="i-04"></a><a id="m-2"></a>
### I-04 — Aged swaps: the Hull-White model keeps paid flows; TraderX exports no past fixings

**Severity:** High · **Status:** PARTIAL · **Found:** TraderX EOD review; widened by the
2026-09-24 audit (M-2)

**What is wrong.** Two halves.

1. *Hull-White model.* `price_swaps` sums every cashflow of a swap at every simulated step.
   A paid flow (`T < t`) gets a clamped `B(t,T)` and `A(t,T) = P(0,T)/P(0,t) > 1`, so it is
   kept and grown: the demo's 3Y payer swap has a mean NPV of −9,852 at t=4, after maturity.
   A coupon fixing during the simulation has no path fixing.
2. *Data.* A coupon fixed before the as-of date needs its historical fixing. Every trade
   config takes `fixings` and refuses a missing one (`MissingFixingError`), as ORE does, but
   TraderX exports no `pastFixings`, so a seasoned TraderX swap cannot be priced.

**Reach.** Hull-White: every cube value past a swap's first accrual start, and the exposure
built on it; t=0 is exact. Market path: correct (paid flows drop out, path fixings by
`FixingManager`'s rule, `tests/test_valuation.py`).

**Current handling.** `price_portfolio` warns per affected swap on the Hull-White model; the
EOD capability document lists I-04 as a known limitation.

**To close.** Half 1 with roadmap 1.3 (the Hull-White model on the shared valuation
pipeline). Half 2 needs `pastFixings` from TraderX ([details](details/traderx-integration.md)).

<a id="i-12"></a>
### I-12 — `/version` reports the dispatcher's backend, not the worker's device

**Severity:** Low · **Status:** OPEN

**What is wrong.** `GET /version` reports `jax.default_backend()` of the HTTP process, which
runs no JAX work; pricing runs in `worker_pool` processes. On a multi-device host a precision
or hardware study reading this field would attribute results to the wrong device.

**To close.** Report the device and the realised per-stage dtypes from the worker, on each
result.

<a id="i-24"></a>
### I-24 — Bonds have no scenario NPV on the Hull-White model

**Severity:** Medium · **Status:** PARTIAL

**What is wrong.** On the Hull-White model a `BondConfig` with `scenario_risk=True` is
refused (`ScenarioPricingNotSupported`); with `scenario_risk=False` the result has the bond's
t=0 NPV and Greeks, an empty `risk`, and `scenario_risk_available=False`. The market path
prices a bond on every path (ORE's `DiscountingRiskyBondEngine` without credit,
`engine.valuation.portfolio.bond_legs`).

**Do not close it** by broadcasting the t=0 NPV across the cube (measured: VaR 0.00 and ES
NaN for a $100k bill, a false "no risk"), by zero-filling, or by defaulting `scenario_risk`
to false. `tests/test_treasury_instrument.py::TestScenarioPricingIsRefused` pins the
refusal and fails against the broadcast.

**To close.** Roadmap 1.3: the Hull-White model reuses the market path's bond legs on its
own scenario curves.

<a id="i-32"></a>
### I-32 — Bermudan/American engine only at `ShiftHorizon = 0`, not ORE's default 0.5

**Severity:** Medium · **Status:** OPEN · **Found:** 2026-09-23

**What is wrong.** Parity with ORE's LGM engine (about 1e-11) holds for ORE's Grid solver at
`ShiftHorizon = 0`, the only configuration the engine accepts. ORE's builder defaults
`ShiftHorizon` to 0.5 of the trade's maturity, and its shipped American example uses the FD
solver. Measured on 5Y trades (engine at its 48-point grid):

| ORE setting | Bermudan | Mid-period Bermudan | American |
|---|---:|---:|---:|
| Grid, `ShiftHorizon = 0` | ~1e-13 | ~1e-13 | ~1e-12 |
| Grid, `ShiftHorizon = 0.5` (ORE's default) | 1.6e-6 | 2.7e-5 | 2.7e-5 – 1.5e-4 |
| FD, shipped settings | 3.8e-4 | 1.0e-3 | 5.7e-4 – 1.6e-3 |

The shift is an exact LGM invariance that moves the state grid, so the gap is ORE's
discretization error.

**To close.** Decided (A-3, D-10): implement the shift (`H → H + shift`, state grid in the
shifted variable, in `engine.models.lgm` and `_state_grid`), prove parity at 0.5 against
`tests/support/ore_lgm_oracle.py` (which already takes `shift_horizon=`), then make 0.5
the default. The FD solver is an option, [F-01](features.md#f-01).

<a id="i-42"></a><a id="m-1"></a>
### I-42 — Hull-White: simulated curves not arbitrage-free against the input curve

**Severity:** High · **Status:** PARTIAL · **Found:** 2026-09-24, audit M-1

**What is wrong.** `engine.simulation.market_model` evolves the short rate toward a constant
`theta` from `initial_rates`, while its discount factors use the Hull-White `A(t,T)` fitted to
the input curve, which assumes the curve-fitted drift θ(t). Martingale error
`E[P(t,T)/N(t)]/P(0,T) − 1` at t=2y: 0.06% (5y) and 0.14% (10y) on a flat 3% curve; 4.2% and
8.8% on a curve rising 3% → 5%.

**Reach.** Every Hull-White cube value past t=0 on a non-flat curve, the exposure built on
it, and the Bermudan/American scenario pricers conditioned on the simulated rate. Closed on
the market path (`tests/test_cam.py` checks the martingale exactly and by Monte Carlo on
sloped curves).

**Current handling.** `price_portfolio` warns on the Hull-White model.

**To close.** Roadmap 1.3: simulate the zero-mean state and add the curve-fitted drift
(`r = x + α(t)`, Brigo–Mercurio 3.36), with the martingale test on a sloped curve.

<a id="i-43"></a><a id="m-3"></a>
### I-43 — Hull-White: options worth zero after expiry instead of becoming the swap

**Severity:** High · **Status:** PARTIAL · **Found:** 2026-09-24, audit M-3

**What is wrong.** A European's scenario NPV is 0 after expiry, a Bermudan's or American's
after its last exercise date, on every path. An exercised physical option is the swap. In
`demos/demo.py` the exposure "risk" at t=2..5 is the options vanishing everywhere at once.

**Reach.** Hull-White exposure after the first expiry. Closed on the market path
(`engine.valuation.options`, ORE's `OptionWrapper`).

**Current handling.** `price_portfolio` warns when an option expires inside the horizon.

**To close.** Roadmap 1.3: reuse `engine.valuation.options` on the Hull-White model's paths.

<a id="i-44"></a><a id="a-2"></a>
### I-44 — Hull-White: scenario pricing mixes Hull-White and LGM

**Severity:** Medium · **Status:** PARTIAL · **Found:** 2026-09-24, audit A-2

**What is wrong.** The Hull-White model simulates a Hull-White short rate and prices
Bermudans/Americans with LGM, conditioning the rollback on the rate through `x_from_r`. Both
read the same `(a, σ)`, but LGM treats σ as the volatility of x, whose short-rate equivalent
is `σ·e^{−at}`: at a = 3% the two differ by 14% at 5y.

**Reach.** Hull-White Bermudan/American values past t=0. Closed on the market path (one
model, LGM, recalibrated per path).

**To close.** Roadmap 1.3: make the Hull-White model consistent within itself (Hull-White
simulation, Hull-White or Hull-White-equivalent pricers, no state conversion). Decision A-2
keeps the model rather than removing it.

<a id="i-45"></a>
### I-45 — Hull-White: numeraire is a discretely accrued bank account

**Severity:** Medium · **Status:** PARTIAL · **Found:** 2026-09-28

**What is wrong.** `N(t_{i+1}) = N(t_i)·exp(r(t_i)·dt)` on factor 0 (left-point rule), biased
by the rate's change over each step; ORE uses the model's exact numeraire. The error grows
with the grid spacing.

**Reach.** Every Hull-White exposure profile (all are NPV/N). Closed on the market path
(`scenario_market.lgm_numeraire`).

**To close.** Roadmap 1.3: the model's exact numeraire in the reporting currency.

<a id="i-46"></a>
### I-46 — Hull-White: Europeans priced off the model vol, not the market vol

**Severity:** Medium · **Status:** PARTIAL · **Found:** 2026-09-28

**What is wrong.** ORE's default European engine is Black/Bachelier on the market swaption
volatility. The Hull-White model prices Europeans with Jamshidian on the trade's `hw_sigma`
(`ORE.JamshidianSwaptionEngine`, not ORE's default), so NPV differs from the market price and
there is no Vega.

**Reach.** Hull-White European NPVs and Greeks. Closed on the market path and in
`engine.market_risk` (Bachelier, `engine.valuation.european`).

**To close.** Roadmap 1.3: market-vol Bachelier as the default European engine on the
Hull-White model; Jamshidian stays as a configurable engine.

<a id="i-47"></a>
### I-47 — Hull-White: calibration basket is not the one ORE builds for the trade

**Severity:** Medium · **Status:** PARTIAL · **Found:** 2026-09-28

**What is wrong.** The Hull-White model calibrates every Bermudan/American, on every rate
factor, to one caller-supplied basket (`engine.calibration.basket`, rounded to whole months,
built on the first uncalibrated trade: `engine.portfolio.request._fill_calibrated_sigma`,
mirrored in `engine/api/schemas.py`). ORE builds a co-terminal basket per trade from its own
exercise dates and underlying.

**Reach.** Calibrated Hull-White Bermudan/American σ, hence NPV, Greeks and Vega. Closed on
the market path (`engine.valuation.bermudan.calibration_basket`, `engine.calibration.ore_lgm`;
2e-11 against ORE's `Calibration=Bootstrap`).

**To close.** Roadmap 1.3: per-trade basket and calibration, shared with the market path.

<a id="i-49"></a>
### I-49 — Per-path recalibration differs from ORE's in two details

**Severity:** Medium · **Status:** OPEN · *Difference from ORE* · **Found:** 2026-09-29

**What is wrong.** On the market path every Bermudan/American is recalibrated on each path
and date, as ORE's `ValuationEngine` does with `recalibrate = true`
(`engine.valuation.bermudan`). Two details differ from ORE's source:

- ORE keeps the parametrization's time grid from the as-of build; the engine measures each
  date's bucket times from that date.
- ORE still passes helpers whose expiry has passed; the engine's basket on a date keeps only
  later exercise dates.

The path-date volatility transcribes `DynamicSwaptionVolatilityMatrix` (`ForwardVariance`),
checked against the formula but not against ORE running it (no Python constructor).

**Reach.** Market-path Bermudan/American values past t=0 and their exposure. Size unmeasured.

**To close.** Decided (X-9): after [I-50](#i-50)'s oracle, compare a Bermudan's cube with
ORE's, then reproduce both details.

<a id="i-54"></a>
### I-54 — No swaption smile: options away from the money read the ATM vol

**Severity:** Medium · **Status:** OPEN · *Difference from ORE* · **Found:** 2026-09-29

**What is wrong.** `engine.market.SwaptionVolSurface` is an ATM normal matrix (expiry × tenor).
ORE reads a vol cube or SABR smile at each option's strike. Every European, and every
Bermudan/American calibration helper (struck at the deal rate, `CoterminalDealStrike`), reads
the ATM vol.

**Reach.** None while markets are ATM-only, as every market given to the engine so far is.
With a smile: European NPV and Vega away from the money, and calibrated Bermudan/American σ.

**To close.** Decided (X-5): a strike axis in the market's volatilities, read at each
option's and helper's strike as ORE reads its cube. SABR is [F-02](features.md#f-02).

<a id="i-64"></a><a id="a-4"></a>
### I-64 — A trade's evaluation date defaults to ORE's thread-local global

**Severity:** Medium · **Status:** OPEN · **Found:** 2026-09-24, audit A-4

**What is wrong.** Every Hull-White trade config defaults `evaluation_date` to
`ORE.Settings.instance().evaluationDate`, which is thread-local and defaults to the
wall-clock date ([I-28](#i-28) was one symptom). A result can depend on which thread ran
and what ran before. `price_portfolio` checks that all trades share one date, and the market
path refuses a trade not valued on the market's as-of date, but a single trade built without
a date silently takes "today".

**To close.** Roadmap 1.3: `evaluation_date` required on every trade config; scope any ORE
global that must be set with a context manager that restores it.

---

## Validation

<a id="i-34"></a>
### I-34 — The ORE oracle's curve differs before the first pillar

**Severity:** Low · **Status:** OPEN · **Found:** 2026-09-25

**What is wrong.** Validation tooling only. `tests/support/ore_lgm_oracle.py` hands ORE
the engine's zero curve as date-quoted zero rates; ORE's zero-curve build re-reads the t=0
rate as `zeroRate(1e-4)`, tilting the first segment by `slope·1e-4`. A Bermudan whose flows
fall inside a first segment rising 3% → 3.2% differs from ORE by up to 2.4e-6 relative.

**Current handling.** Parity tests needing 1e-10 use a curve flat to its first non-zero
pillar (`tests/test_trade_dates.py`); the oracle's docstring states the limit.

**To close.** With roadmap 2.2 (same file): hand ORE a curve it does not rebuild, or solve
for the as-of quote that ORE's rebuild maps onto `z0`.

<a id="i-50"></a>
### I-50 — No path- or distribution-level parity test against an ORE simulation

**Severity:** Medium · **Status:** OPEN · **Found:** 2026-09-29

**What is missing.** Every market-path component equals ORE (model analytics to 1e-12,
pricers on scenario curves to 1e-8 – 1e-12, fixing, cash-flow and exercise rules), but
nothing compares the assembled cube or the exposure profiles with an ORE simulation.
Assembly errors across components would pass every current test.

**To close.** Generalize the oracle to an OREApp XVA run, then the L4 distribution test,
then L3 path parity once gate V-4 closes
([details/ore-parity-validation.md](details/ore-parity-validation.md)).

<a id="i-51"></a>
### I-51 — Sensitivities not checked against ORE's sensitivity analytic

**Severity:** Medium · **Status:** OPEN · **Found:** 2026-09-29

**What is missing.** `engine.risk.sensitivities` implements ORE's definitions (zero-rate
shifts at the curve tenors, forward-difference Delta, `up − 2·base + down` Gamma, Vega per
quote, Theta on the rolled market), checked for internal consistency
(`tests/test_sensitivities.py`), but never against an OREApp sensitivity run. A different
shift convention in ORE's simulation market would pass every current test.

**To close.** Run ORE's sensitivity analytic through the oracle on the shared portfolio
(`tests/support/portfolio.py`), compare per trade, factor and tenor to 1e-8 relative
([details](details/ore-parity-validation.md)).

---

## Performance

<a id="i-21"></a>
### I-21 — Hull-White Greeks recompile 23 XLA programs per call

**Severity:** Medium · **Status:** OPEN · every number is correct

**What is wrong.** A repeated identical `price_portfolio` call on the Hull-White model
recompiles 31 programs (208 cold), 23 of them in `engine/risk/greeks.py`: each Greek jits a
closure built fresh per call (`_swap_price_fn`, `_swaption_price_fn`, `_bermudan_price_fn`,
`bermudan_vega`'s per-bucket closures), and `jax.jit` caches on function identity.

**To close.** Re-measure after roadmap 1.3, which may replace this code. If it survives:
memoize the jitted wrapper in a bounded LRU keyed on `static_key(prepared)` plus the curve's
shape and dtype. Never key on the config (misses what date generation derives) or on `id()`.
Prototyped on Europeans: steady-state compiles 1 → 0, bit-identical. The test must also show
that trades differing only in `notional`, `fixed_rate` or tenor still get their own answer;
`tests/test_profiling_and_jit.py::TestCompileCounts::test_repeated_greeks_call_costs_one_compile_not_zero`
pins today's count and changes with the fix.

<a id="i-22"></a>
### I-22 — Hull-White calibration recompiles 8 XLA programs per call

**Severity:** Low · **Status:** OPEN · every number is correct

**What is wrong.** `engine/calibration/lgm.py` bakes Python floats into traced programs
(`_bisect_bucket_sigma` closes over `market_price`; `calibrate_lgm_sigma`'s closures over
`target`), so each bucket and call compiles again. A different mechanism from I-21: a
content-keyed memo would miss every time.

**To close.** Pass `market_price` as a traced argument to a stable, module-level jitted
function. `price_fn` differs per bucket (it depends on the calibrated prefix) and stays
static, so expect about 8 → 2. Do not trade the 60 bisection iterations for compile count
(`rmse < 1e-8` is asserted).

<a id="i-53"></a>
### I-53 — The market path is slow (full suite 23 → 46 min)

**Severity:** Medium · **Status:** OPEN · **Found:** 2026-09-29

**What is wrong.** The full suite went from 22:54 before the ORE alignment to 46:00. The
slowest test, `tests/test_api_market_path.py::test_result_matches_direct_price_portfolio_call`,
takes about 275 s for 8 trades on 128 paths and 4 dates. Unprofiled suspects: each
Bermudan/American rebuilds its basket through `ORE.SwaptionHelper` and bootstraps on every
path date (ORE's `recalibrate = true`), sensitivities revalue one bump at a time in Python
loops, nothing is jitted end to end, and each worker process recompiles.

**To close.** Profile a market-path job (`JAX_RISK_PROFILE_DIR`,
[profiling](../concepts/profiling.md)), then optimize, keeping every parity test
bit-identical in FP64. `PricingConfig(recalibrate=False)` exists where ORE's semantics are
not needed.

<a id="i-61"></a><a id="p-1"></a>
### I-61 — Nothing runs on more than one device

**Severity:** Medium · **Status:** OPEN · **Found:** 2026-09-24, audit P-1

**What is wrong.** The project's goal is to run across multiple TPUs and compare many
low-precision paths against fewer FP64 paths in equal wall time. No code uses `shard_map`,
`jax.sharding` or `pmap`; the worker pool runs whole jobs side by side, one per process.

**To close.** Shard the scenario axis: Sobol draws (per-device skip-ahead or scrambles; the
seed exists), path evolution, pricing and exposure are scenario-parallel; VaR/ES order
statistics need one cross-device step. Then device-count-aware pool sizing and
`JAX_PLATFORMS`/`TPU_VISIBLE_CHIPS` pinning on a real Cloud TPU VM.

<a id="i-62"></a><a id="p-2"></a>
### I-62 — Hull-White Bermudan/American scenario pricing runs on the host

**Severity:** Low · **Status:** OPEN · **Found:** 2026-09-24, audit P-2

**What is wrong.** `price_bermudan_swaptions` copies the paths to host memory and loops over
steps with `np.interp` (`engine/instruments/bermudan_swaption.py`), so the scenario half
never runs on the accelerator. The market path is vectorized over paths.

**To close.** Moot once roadmap 1.3 moves the Hull-White model onto the shared pipeline.

---

## API

<a id="i-08"></a>
### I-08 — Portfolio job store in memory; a running EOD attempt is lost on restart

**Severity:** Medium · **Status:** PARTIAL

**What is wrong.** `_JOBS` in `engine/api/routes.py` is a dict in the HTTP process: a
restart loses every portfolio job, and a second uvicorn worker 404s on ids issued by the
first. States are `pending/running/done/failed` only, with no failure classes. The EOD path
is durable for finished work (`engine/integration/publication.py`: manifest as commit point,
scan recovery, idempotent `submissionId` across restarts), but a *running* attempt is memory
only and reads as unknown after a restart. The store is single-machine.

**To close.** Port the publication design to the portfolio path, with failure classes
(`bad-terms`, `missing-market-data`, `unsupported-product`, `numerical-failure`,
`infrastructure`). On the EOD path, add the durable accepted-attempt record, a boot sweep and
an `interrupted` lookup state (TraderX acceptance case A-09).

<a id="i-09"></a>
### I-09 — Whole scenario cube serialized into the JSON response

**Severity:** Medium · **Status:** OPEN

**What is wrong.** `PortfolioResultSchema.npv_cube` is nested JSON: 4096 × 24 × 211 is about
20M floats in one HTTP body.

**To close.** With roadmap 4.1: write the cube to a chunked artifact (shape, dtype, axis
order, hash, item-order file) and return a reference plus summaries, as the EOD contract
already specifies.

<a id="i-10"></a>
### I-10 — No trade identity on the configs; results keyed by position

**Severity:** Medium · **Status:** OPEN

**What is wrong.** `PortfolioResult.greeks` and `base_npv_per_trade` are keyed by array
position. Optional `PortfolioRequest.trade_ids` (echoed as `PortfolioResult.trade_ids`, and a
`trade_id` per trade over HTTP on the market path) are caller labels; the configs carry no
identity, and the Hull-White request shape has none. Reordering or filtering would silently
misattribute. The EOD boundary is closed (`engine/integration/identity.py`).

**To close.** An instrument id on every trade config (roadmap 1.3, while every config
changes), echoed on every result row (4.1).

<a id="i-23"></a>
### I-23 — `accrualBasis` strictness rests on an unconfirmed reading

**Severity:** Medium · **Status:** ASSUMPTION · **Raised:** 2026-09-16

**The premise.** `engine/integration/terms.py` accepts exactly `dateBasis = SESSION_DATE`,
`settlementAdjustment = NONE`, `rounding = HALF_EVEN`, schema
`traderx.accrual-basis.v1`, and refuses anything else. TraderX has not said whether new
values land in `v1` or force a `v2` (asked in responses v4, v6 and v7).

**Risk.** If they add values in place, bundles they consider valid are refused on the day a
real calendar is exported. That fails safe (a refusal, not a wrong number), but reads as a
defect on call. A `TermsJoinError` naming these fields may be this allowlist, not a bad
bundle.

**To close.** Their answer. "New version": close with no change. "In place": widen
`SUPPORTED_DATE_BASES` / `SUPPORTED_SETTLEMENT_ADJUSTMENTS` / `SUPPORTED_ACCRUAL_ROUNDING`
with a test per value. `tests/test_integration_terms_v2.py::TestUnrecognizedValuesAreRefused`
pins today's rule.

<a id="i-56"></a>
### I-56 — The API cannot reach every setting; two routes named like versions

**Severity:** Medium · **Status:** OPEN · **Found:** 2026-09-30, decision A-2

**What is wrong.**

1. The market path is served at `POST /v2/portfolio/price` with `schema_version: "2"`; the
   Hull-White model at `POST /portfolio/price`. These are two models, not versions.
2. The model is chosen by the request's shape, so no request can mix options across them.
3. Unreachable: the sensitivity settings (`SensitivityConfig`: tenors, shifts, Theta horizon,
   vol decay; unreachable from `price_portfolio` too), market-risk VaR/ES
   (`engine.market_risk.run_market_risk`, no route), the market path's calibrations as
   standalone runs (`POST /calibration/lgm` serves only the Hull-White basket), and trade ids
   on the Hull-White shape.

Nothing is priced wrongly; unreachable settings run at documented defaults.

**To close.** Decided (A-2): one route and one request whose configuration
([I-68](#i-68)) reaches every setting, validated before any job starts (unknown fields
refused, each refusal naming its field), no version-like names except for real contract
revisions. Today's routes keep answering, translated. A completeness test fails when a
configuration setting has no API field
([details/configurable-engine.md](details/configurable-engine.md)).

<a id="i-57"></a>
### I-57 — EOD: a cached result is served before the submission id is checked

**Severity:** High · **Status:** OPEN · **Found:** 2026-09-17, TraderX acceptance case A-02
(FR-07), reproduced in response v7; still present

**What is wrong.** In `engine/api/eod_routes.py` the `reuseExistingResult` branch returns a
completed attempt from `STORE.lookup(key)` before `STORE.start(key, submission_id=...)`, which
is where `SubmissionIdConflict` is raised. A coordinator that reuses a submission id across
bundles gets another workload's priced result under its own id, with `"reused": true`.

**To close.** Extract the binding check from `AttemptStore.start` and run it before any
cache read. Red first with A-02's case, added to this suite.

<a id="i-58"></a>
### I-58 — EOD: two concurrent submissions of one workload both execute

**Severity:** Medium · **Status:** OPEN · **Found:** 2026-09-17, TraderX case A-03 (FR-08);
still present

**What is wrong.** `AttemptStore.start` deduplicates on `submissionId` only. Two submissions
of the same workload with no id (or `reuseExistingResult=false`) each get a running attempt,
and both run `price_bundle`. The lock protects the dict, not the computation.

**To close.** A per-workload running-attempt index under the existing lock: a second
submission joins the running attempt (200 with its `attemptId` if it finishes in time, else
202 `running`). Regression test with a deterministic barrier, not timing. Cross-process
exclusion is not claimed.

<a id="i-59"></a>
### I-59 — EOD: `calculations` and `reportingCurrency` accepted, keyed, then ignored

**Severity:** Medium · **Status:** OPEN · **Found:** 2026-09-17, TraderX cases A-04, A-05
(FR-06); still present

**What is wrong.** `EodSubmissionSchema` accepts both, `_key_for` hashes both into the
workload key, and `price_bundle(bundle, request.marketInputs)` receives neither. An `EUR`
request returns USD results with 200 OK, cached separately from the identical USD run; an
unknown calculation name is accepted.

**To close.** Reject an unknown calculation (400 `UNKNOWN_CALCULATION`, allowlist
`engine.integration.result.CALCULATIONS`); add `reportingCurrencies: ["USD"]` to the
capability document and reject others (400 `UNSUPPORTED_REPORTING_CURRENCY`). Tests assert
the consequence, not that the field parses.

<a id="i-60"></a>
### I-60 — EOD result schema has no stated policy on added fields

**Severity:** Low · **Status:** ASSUMPTION · **Raised:** 2026-09-17, response v7 §5.2

**The premise.** `jaxrisk.eod-result.v1` does not say whether consumers may pin its exact
shape. TraderX's validator does, so the next field added breaks them. Proposed to TraderX:
`additionalProperties: false` at the root and any new field is a schema version bump.

**To close.** Their answer, recorded as a decision; then state it in the published schema
with a test.

---

## Architecture

<a id="i-55"></a><a id="a-1"></a>
### I-55 — Precision switched by a process-global flag; unproven combinations not flagged

**Severity:** Medium · **Status:** OPEN · **Found:** 2026-09-24, audit A-1; decisions A-9, D-9

**What is wrong.**

1. *Mechanism.* `jax_enable_x64` is process-global. `market_model` enables it at import,
   `generate_paths` toggles it, `price_portfolio` forces it on under `_PRICING_LOCK`, and the
   worker pool keeps one process pool per precision. Because `price_portfolio` turns x64
   back on in every job, a 32-bit worker runs with x64 on after its first job, so the tiers do
   not isolate what they were built to. The market-path code already takes explicit dtypes.
2. *No warning.* Any per-stage combination may be run (decision D-9), but nothing records
   which combinations are shown adequate for which figure, so an FP32 exposure profile looks
   exactly like a validated one.

**To close.** Decided (A-9): (1) roadmap 1.4: x64 on once per process, every stage and array
with an explicit dtype from the configuration, then remove the toggling, the lock and the
tiers, never leaving precision unadjustable in between. (2) roadmap 2.7: an evidence table per
figure and precision (what was validated, how, at how many paths) and a warning on any
result whose combination is unproven.

<a id="i-63"></a><a id="a-3"></a>
### I-63 — Hull-White trade configs carry copies of model parameters

**Severity:** Medium · **Status:** OPEN · **Found:** 2026-09-24, audit A-3

**What is wrong.** Hull-White swaption configs carry `hw_a`, `hw_sigma` and
`initial_zero_curve`, which must equal the simulation's entries for their
`rate_factor_index`; `validate_portfolio_against_simulation` exists to catch the copies
drifting, and swaps use curve indices instead. Two conventions for one fact caused I-13.
Closed on the market path (trades carry no model; refused if set).

**To close.** Roadmap 1.3: trades name their curves and index; model parameters and
calibrated σ come from the market and the configuration.

<a id="i-68"></a>
### I-68 — Models and engines are separate code paths, not options of one configuration

**Severity:** Medium · **Status:** OPEN · **Found:** 2026-09-30, decisions A-1, A-5, A-8

**What is wrong.** The model is chosen by the type of `PortfolioRequest.market`
(`SimulationConfig` for Hull-White, `Market` for LGM), and each path hard-wires its engines
and Greeks method (bump on the market path, AD on Hull-White). `engine.market_risk` picks a
European's engine from the trade's fields. Decision A-1 makes models, engines, Greeks method
and precision options of one run configuration, with ORE's defaults.

**To close.** Roadmap 1.2: one run configuration naming, per component, the model per
currency, the simulation, the engine per product, the Greeks method and the precision per
stage; its defaults reproduce today's market path bit for bit
([details/configurable-engine.md](details/configurable-engine.md)).

---

## Scope gaps

Refused with an identified reason, never approximated. Each is listed because a current
consumer (TraderX) sends the input.

<a id="i-05"></a>
### I-05 — No faithful USD-SOFR / ACT-360 swap construction

**Severity:** High · **Status:** OPEN · blocked on the D03/D04 convention agreement

**What is wrong.** `build_vanilla_swap` builds a generic term-IBOR swap (`SimIndex6M`,
ACT/365 legs, TARGET, no compounding). A USD-SOFR booking is overnight, ACT/360, US
calendars, compounded in arrears, with lookback/lockout/payment lag. ACT/360 vs ACT/365 alone
moves every accrual by 1.389% (about 46 × a 1bp DV01 on a 5Y fixed leg).

**Current handling.** The EOD boundary refuses any booking outside an explicit allowlist
(`engine/integration/conventions.py`, `CONVENTION_NOT_SUPPORTED` naming the fields; the
TraderX SOFR fixture's 13 missing terms). A direct Python caller cannot express SOFR in a
`SwapConfig`.

**To close.** The convention set from TraderX, then a new builder beside the existing one,
accepted against a same-terms ORE reference
([details/traderx-integration.md](details/traderx-integration.md)). Do not start on guessed
conventions.

<a id="i-07"></a>
### I-07 — No corporate bond, equity or listed-option pricer

**Severity:** Medium · **Status:** OPEN

**What is wrong.** Treasuries price on both paths; TraderX's position export also carries
corporate bonds (refused: a Treasury-discounted corporate is not credit pricing, it needs a
credit model), cash equities ([I-18](#i-18): a market-data gap, not a pricer gap), listed
options, TIPS and FRNs. `SimulationConfig.equities` drives risk-factor paths; it is not a
position pricer. FX and equity trades on the market path are [F-04](features.md#f-04).

**To close.** Per instrument, on demand: a config, a pricer, ORE parity tests.

<a id="i-16"></a>
### I-16 — `rateSensitivity` is parallel-only

**Severity:** Medium · **Status:** OPEN · blocked on an observed curve (TraderX W2)

**What is wrong.** The note's `rateSensitivity` is a 1bp parallel shift, labelled
`shockedFactor: "zero-curve-parallel"`. The only market input the EOD boundary accepts is an
assumed flat profile, so there is no pillar structure to shift. Parallel DV01 cannot tell a
2y position from a 10y one of equal duration.

**Do not close it** by bumping the flat profile per pillar: that yields an all-equal or
single-non-zero vector that passes a shape check.

**To close.** `marketInputs.mode: "package"` with bootstrapped pillars (W2); the bump loop
then shifts one pillar at a time and names it.

<a id="i-18"></a>
### I-18 — No equity spot or FX source; equity positions refused

**Severity:** Medium · **Status:** OPEN · blocked on a market-data decision

**What is wrong.** An equity position is `signedQuantity × multiplier × spot × fx`, and the
EOD boundary has no spot or FX input (`engine/integration/market_inputs.py` registers flat
rate profiles only). Positions are read, validated and refused (`SPOT_SOURCE_NOT_SUPPLIED`,
`FX_SOURCE_NOT_SUPPLIED`), echoing the quantities read; `capabilities()` reports equity NPV
as `blockedOnMarketInput`.

**Do not close it** with `closingMark`: `quantity × closingMark × multiplier` reproduces
TraderX's own `marketValue` exactly, an echo presented as a valuation.
`tests/test_integration_equity.py::TestDoesNotEchoTheExportedMark` fails against it.

**To close.** A registered spot/FX surface in `marketInputs`, or observed spots in the
bundle. The pricer is four multiplications.

---

## Tooling

<a id="i-27"></a>
### I-27 — Long full-suite runs can hard-abort inside XLA

**Severity:** Medium · **Status:** OPEN · located, not root-caused · **Found:** 2026-09-17

**What is wrong.** A long `pytest tests/` run has died with `Fatal Python error: Aborted` and
no summary line, the crashing thread inside `jax/_src/compiler.py`
(`backend_compile_and_load`) while `ProcessPoolExecutor` threads were alive. Intermittent: the
same command later passed in full, and no abort occurred in the recorded runs since
2026-09-24. `test_cross_tier_jobs_correct_and_concurrent` (a wall-clock overlap assertion)
has failed intermittently on the same premise.

**Hypothesis, unproven.** `worker_pool._POOLS` lives for the interpreter; `tests/test_api.py`
creates pools and never calls `shutdown_pools`, so later in-process compiles run with worker
children attached. Pairing modules does not reproduce it. Also check whether XLA's on-disk
compilation cache is shared unsafely with spawned workers.

**To close.** Add a `shutdown_pools()` autouse fixture to `tests/test_api.py`, then repeated
clean full runs against a known-bad baseline; one green run proves nothing. Replace the
wall-clock overlap assertion with a deterministic one.

<a id="i-66"></a><a id="q-2"></a>
### I-66 — No linter or type checker

**Severity:** Low · **Status:** OPEN · **Found:** 2026-09-24, audit Q-2 (pins and CI done)

**What is wrong.** No ruff/flake8, mypy/pyright or pre-commit. `pyflakes engine` today
reports unused imports in `integration/{bundle,market_inputs,pipeline,result,terms}.py`,
`models/hull_white.py`, `risk/greeks.py` and `risk/price_functions.py`, an f-string without
placeholders in `integration/equity.py`, and re-exports that need `# noqa` or `__all__`
(`portfolio/validation.py`, `portfolio/request.py`, `models/ore_builders.py`,
`simulation/market_model.py`). `requirements.txt` does not mention the `profiling` extra.

**To close.** Add ruff to `pyproject.toml` and CI, fix or mark each finding (check that a
"re-export" is actually imported elsewhere first), then a type checker on `engine/`.

<a id="i-67"></a><a id="q-3"></a>
### I-67 — Test modules import each other and repeat fixtures

**Severity:** Low · **Status:** OPEN · **Found:** 2026-09-24, audit Q-3

**What is wrong.** `tests/test_worker_pool.py` and `tests/test_portfolio_market_path.py`
import helpers from other test modules; `tests/test_market_model.py` imports from `conftest`.
`FIXTURES = ...traderx-eod` is defined in 15 files and `MARKET = {...flat-3pct-v1}` in 7.
Many tests reach into private functions, which freezes internal structure and makes
refactors (roadmap stage 1) break tests without behaviour changing.

**To close.** Shared helpers and constants in `tests/support/`; replace private-symbol tests
with public-entry tests where the refactors of stage 1 touch them. Collection must stay
identical except where a test is deliberately rewritten.

---

<a id="closed"></a>
## Closed

Fixed issues. The full account is in the commit that closed each (`git log --grep "I-NN"`,
or the register's text at commit `8306073`). The test named guards the fix.

| ID | Fixed | Regression test |
|---|---|---|
| <a id="i-01"></a>I-01 | Swap Delta/Gamma/Theta were silently absent from portfolio results | `tests/test_portfolio_gap_fixes.py::TestSwapGreeksReachThePortfolioPath` |
| <a id="i-02"></a>I-02 | Bermudan Vega was never computed | `tests/test_portfolio_gap_fixes.py::TestBermudanVegaReachesThePortfolioPath` |
| <a id="i-03"></a>I-03 | No per-instrument NPV | `tests/test_portfolio_gap_fixes.py::TestPerTradeBaseNpv` |
| <a id="i-06"></a>I-06 | American exercise ignored ORE's broken-period `couponRatio` (up to 6× off) | `tests/test_ore_lgm_parity.py`, `tests/test_bermudan_swaption.py::TestMidPeriodBermudanExercise` |
| <a id="i-11"></a>I-11 | Risk measure unlabelled; no Monte Carlo error reported | `tests/test_risk_measure_label.py::TestPortfolioResultStatesItsMeasure` |
| <a id="i-13"></a>I-13 | A negative curve index priced against the wrong curve | `tests/test_portfolio_gap_fixes.py::TestCurveIndexValidatedBeforeAllPricing` |
| <a id="i-14"></a>I-14 | `generate_paths(precision=32)` leaked `jax_enable_x64=False` | `tests/test_market_model.py::TestGeneratePathsEdgeCases` |
| <a id="i-15"></a>I-15 | The worker-pool concurrency test could not observe concurrency | `tests/test_worker_pool.py::TestWorkerPoolConcurrency` |
| <a id="i-17"></a>I-17 | A malformed note date failed the whole bundle | `tests/test_integration_note.py::TestRefusalsAreNotePricingErrors` |
| <a id="i-19"></a>I-19 | The accrual tolerance rounded its own bound | `tests/test_integration_note.py::TestToleranceIsDerivedNotConstant` |
| <a id="i-20"></a>I-20 | Impossible calendar dates aborted the whole bundle | `tests/test_integration_note.py::TestImpossibleCalendarDates` |
| <a id="i-25"></a>I-25 | A scalar Greek crashed the HTTP result serializer | `tests/test_api_bond_schemas.py::TestBondGreeksSerializeOverHttp` |
| <a id="i-26"></a>I-26 | Greeks for a bond maturing tomorrow crashed on the Theta reprice | `tests/test_portfolio_bond_wire_through.py::TestBondGreeksReachThePortfolioPath` |
| <a id="i-28"></a>I-28 | The `var_es` module demo crashed on a moved date | `tests/test_demos.py::TestComponentDemosRun` |
| <a id="i-29"></a>I-29 | A rounded exercise time silently dropped a coupon (exercise now given as dates) | `tests/test_ore_bermudan_oracle.py::TestExerciseDatesAreExact` |
| <a id="i-30"></a>I-30 | The `A(t,T)` variance term was nearly uncovered at t=0 | `tests/test_ore_coverage_hardening.py::TestVarianceTermIsActuallyChecked` |
| <a id="i-31"></a>I-31 | Bermudan/American floating coupons projected over the wrong period | `tests/test_ore_lgm_parity.py` |
| <a id="i-33"></a>I-33 | On Linux, worker-pool jobs hung once the parent had run JAX (fork) | `tests/test_worker_pool.py::TestPoolsSpawnOnEveryPlatform` |
| <a id="i-35"></a>I-35 | An American already in its window was exercisable on the evaluation date | `tests/test_trade_dates.py::test_seasoned_bermudan_and_american_equal_ore` |
| <a id="i-36"></a>I-36 | A non-ACT/365 floating leg projected the wrong forward | `tests/test_trade_dates.py::test_any_leg_day_count_equals_ore` |
| <a id="i-37"></a>I-37 | A European silently ignored `floating_spread` (refused on Hull-White, priced on the market path) | `tests/test_european_swaption.py::TestJamshidianRefusals`, `tests/test_valuation.py::test_european_today_equals_ores_default_engine` |
| <a id="i-38"></a>I-38 | Theta rolled a business day; ORE rolls a calendar day | `tests/test_trade_dates.py::test_swap_theta_equals_ore`, `tests/test_sensitivities.py::test_theta_rolls_one_calendar_day_from_a_friday` |
| <a id="i-39"></a>I-39 | Bond Theta had no add-back for a coupon paid in the period | `tests/test_sensitivities.py::test_theta_adds_back_a_bond_coupon_paid_on_the_theta_date` |
| <a id="i-40"></a>I-40 | The note's `rateSensitivity` ignored `fractionDecimals` | `tests/test_integration_note.py::TestSensitivityUsesTheDeclaredFractionDecimals` |
| <a id="i-41"></a>I-41 | A European at zero mean reversion priced at intrinsic value (now refused) | `tests/test_european_swaption.py::TestJamshidianRefusals` |
| <a id="i-48"></a>I-48 | Zero curves extrapolated a flat zero rate; ORE a flat forward | `tests/test_treasury_instrument.py::TestCurveInterpolation`, `tests/test_curves.py` |
| <a id="i-52"></a>I-52 | Cash settlement was priced as physical | `tests/test_valuation.py::test_a_cash_settled_european_uses_the_par_yield_annuity` |
| <a id="i-65"></a><a id="a-6"></a>I-65 | Demo data, the modules' `__main__` demos and the ORE test oracle shipped inside `engine/` (now `demos/`, `tests/support/`) | `tests/test_import_layering.py::test_engine_ships_no_demo_or_test_code` |
| <a id="m-4"></a>Audit M-4 | Trades were defined relative to the evaluation date (now absolute dates) | `tests/test_trade_dates.py` |
| <a id="m-5"></a>Audit M-5 | Theta re-rolled the trade instead of ageing it | `tests/test_trade_dates.py` |
| <a id="r-1"></a>Audit R-1 | Cube quantiles were reported as VaR/ES (now exposure profiles; market-risk VaR/ES by t=0 revaluation) | `tests/test_exposure.py`, `tests/test_market_risk.py`, `tests/test_market_risk_ore_parity.py` |
| <a id="p-3"></a>Audit P-3 | The precision study bypassed the engine's own FP32 path | `demos/demo_precision.py` |
| <a id="a-5"></a>Audit A-5 | Instruments imported validators from the layer above | `tests/test_import_layering.py` |
| <a id="a-7"></a>Audit A-7 | The bond pricer was a separate plain-Python implementation | `tests/test_market_risk.py::TestRevaluation` |
