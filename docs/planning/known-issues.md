# Known Issues

Open defects and important shortcomings: what each does to a number or a caller, and what
closing it takes. The order of work is in [roadmap.md](roadmap.md); the rules for this file
(statuses, severities, how to add and close an entry) are in [README.md](README.md). Fixed
issues keep one line in the [closed ledger](#closed).

Read this before trusting an exposure profile: a Bermudan's or American's values on the
paths, and so its exposure, are 0.5–4% above ORE's simulation ([I-49](#i-49)); swaps,
Europeans and bonds equal ORE's simulation path by path since 2026-10-07.

## Verification status

Last full run, 2026-10-12, on the precision evidence table ([I-55](#i-55), closed): the
acceptance standard as data (`compliance/precision-acceptance.json`), the generator
(`tests/support/precision_evidence.py`) and its files (`compliance/precision-evidence.md`,
`engine/precision/evidence.json`), the run-time lookup and warnings (`engine/precision/evidence.py`,
`engine.run.pipeline.precision_warnings`, `engine.risk.market.run`), the harness's market name and
engine digest, all five precision campaigns measured again. 3,308 collected on Windows: the 3,272
of the run before plus 35 in `tests/test_precision_evidence.py` and 1 in
`tests/test_precision_harness.py`. Each run printed its summary line:

- **Windows**, `-n 8`: **3,299 passed, 3 skipped, 6 xfailed, 0 failed**, 13m15s (the skips and
  expected failures as before: the two accelerator-only defaults and the GPU plugin check; the
  Bermudan/American L3 cases of `tests/test_ore_xva_parity.py`, strict, waiting on [I-49](#i-49)).
- **Linux** (Docker `python:3.11`, 4 CPUs, `-n 4`, the fast tier of the seven modules the change
  touches): **233 passed, 0 failed**, and `--check` passes: the change reads a data file through
  `importlib.resources`, which a Linux checkout and install must find
  ([rule 5](README.md#verification-rules)). Its first attempt failed, finding a defect: the
  repository's `.gitignore` ignores every `*.json`, so `compliance/precision-acceptance.json`
  and `engine/precision/evidence.json` would never have been committed; both are now negated
  beside the TraderX fixtures.
- **GPU**: not rerun; the change touches no device code and computes nothing on a device (the
  warnings are plain Python).
- **Golden snapshot**: not rerun; no array of any result changes (the change adds warnings and
  docstrings, and no kernel, pricer or estimator line). The precision campaigns show it on
  23,736 measured rows: every one equal bit for bit to its record before the change.
- **Precision harness**: all five campaigns rerun on one revision (engine digest
  `805c1a7f8cad963b`), 57 minutes; no case compiles after its first seed.

Red first: on the code before, an FP8 portfolio run's `warnings` were `[]` (the pipeline built
them empty) and `engine.precision.evidence` did not exist; the ceiling test fails when a ceiling
skips a failing path count (checked by changing the generator's rule and restoring it).

The fast tier (`-m "not slow"`) alone is not a full verification and is never recorded here. Rules:
[README.md](README.md#verification-rules).

## Summary

| ID | Issue | Sev. | Status | Category | Roadmap |
|---|---|---|---|---|---|
| [I-04](#i-04) | Seasoned TraderX swaps: the export has no past fixings | High | OPEN | Scope | External |
| [I-05](#i-05) | No faithful USD-SOFR / ACT-360 swap construction | High | OPEN | Scope | External |
| [I-07](#i-07) | No corporate bond, equity or listed-option pricer | Medium | OPEN | Scope | By demand |
| [I-08](#i-08) | A running TraderX attempt is lost on restart | Medium | PARTIAL | API | TraderX |
| [I-16](#i-16) | `rateSensitivity` is parallel-only | Medium | OPEN | Scope | External |
| [I-18](#i-18) | No equity spot or FX source; equity positions refused | Medium | OPEN | Scope | External |
| [I-23](#i-23) | `accrualBasis` strictness rests on an unconfirmed reading | Medium | ASSUMPTION | API | TraderX |
| [I-27](#i-27) | Long full-suite runs can hard-abort inside XLA or lose a worker process | Medium | OPEN | Tooling | Fix what exists |
| [I-32](#i-32) | Bermudan/American engine only at `ShiftHorizon = 0`, not ORE's default 0.5 | Medium | OPEN | Correctness | Focus |
| [I-49](#i-49) | Per-path recalibration differs from ORE's in two details | Medium | OPEN | Correctness | Focus |
| [I-73](#i-73) | A calibration that misses its basket is accepted unless it hits the bracket's ceiling, and on a path not even then | Medium | OPEN | Correctness | Focus |
| [I-103](#i-103) | Malformed market and trade inputs are accepted: an unsorted curve prices silently, a late expiry crashes | Medium | OPEN | Validation | Focus |
| [I-104](#i-104) | Edge cases valued wrong: a zero-variance European, a bond awaiting a delayed payment, a long Theta, H at small a·t | Low | OPEN | Correctness | Fix what exists |
| [I-105](#i-105) | Storage below float64 does not lower a run's peak memory: the reductions load the whole cube at float64 | Medium | OPEN | Performance | Focus |
| [I-106](#i-106) | The job queue's file: an interrupted migration stops it opening; a second file starts a second worker | Low | OPEN | API | Focus |
| [I-107](#i-107) | TraderX: `bundleId` is trusted, not recomputed, so a changed bundle can get the old result | Medium | OPEN | API | TraderX |
| [I-108](#i-108) | TraderX: NaN amounts priced as `ok`, terms contradicting the row priced, duplicate rows counted twice | Medium | OPEN | API | TraderX |
| [I-109](#i-109) | TraderX: after a failed publication a same-id retry never prices or publishes | Medium | OPEN | API | TraderX |
| [I-110](#i-110) | The documentation contradicts the code in places; no single parity table | Low | OPEN | Tooling | Fix what exists |
| [I-111](#i-111) | Reported standard errors treat Sobol paths and historical windows as independent | Medium | FLAGGED | Correctness | Focus |
| [I-98](#i-98) | FP8 compute keeps a bias: each path's operand is rounded to nearest on FP8's grid | Medium | OPEN | Correctness | Focus |
| [I-97](#i-97) | XLA computes some native GPU products wrong, so FP8 and bfloat16 passes run emulated on a GPU | Medium | OPEN | Performance | Focus |
| [I-87](#i-87) | The HTTP API is one route per engine function, not one run request | Medium | OPEN | API | Focus |
| [I-88](#i-88) | The API completeness test checks field names, not that a value reaches the engine | Low | OPEN | Tooling | Focus |
| [I-89](#i-89) | A retried job submission runs the job twice | Low | OPEN | API | Focus |
| [I-95](#i-95) | A precision setting the run does not read is accepted and ignored | Low | OPEN | API | Focus |
| [I-51](#i-51) | Sensitivities not checked against ORE's sensitivity analytic | Medium | OPEN | Validation | Focus |
| [I-78](#i-78) | AD and bump Greeks differ by up to 2% on a sloped market | Medium | OPEN | Validation | Focus |
| [I-53](#i-53) | On Windows, pricing runs slower after AD Greeks; Newton's bootstrap compiles 3–4 s longer from scratch | Low | PARTIAL | Performance | Fix what exists |
| [I-54](#i-54) | No swaption smile: options away from the money read the ATM vol | Medium | OPEN | Correctness | New |
| [I-58](#i-58) | TraderX: two concurrent submissions of one workload under different ids both execute | Medium | PARTIAL | API | TraderX |
| [I-59](#i-59) | TraderX: `calculations` and `reportingCurrency` accepted, keyed, then ignored | Medium | OPEN | API | TraderX |
| [I-60](#i-60) | TraderX result schema has no stated policy on added fields | Low | ASSUMPTION | API | TraderX |
| [I-76](#i-76) | The job queue keeps every job and result forever | Low | OPEN | API | Focus |
| [I-77](#i-77) | A worker that cannot start leaves jobs `pending` with no signal | Low | OPEN | API | Focus |
| [I-91](#i-91) | A long job delays every job behind it, and no job can be cancelled | Low | OPEN | API | Focus |
| [I-90](#i-90) | The HTTP API has no authentication, rate limit or body-size limit | Low | OPEN | API | TraderX |
| [I-61](#i-61) | Nothing runs on more than one host; multi-device speed unmeasured | Medium | PARTIAL | Performance | Focus |
| [I-66](#i-66) | No linter or type checker | Low | OPEN | Tooling | Fix what exists |
| [I-67](#i-67) | Test modules import each other and repeat fixtures | Low | OPEN | Tooling | Fix what exists |
| [I-81](#i-81) | A cold job compiles about 180 one-operation programs; market risk vmaps a closure | Low | OPEN | Performance | Fix what exists |
| [I-83](#i-83) | A Bermudan/American on the paths needs 0.3–0.7 MB per path and step: 64k paths run out of memory | Medium | OPEN | Performance | Focus |
| [I-93](#i-93) | TraderX's submissions are priced by their own pricers, not the engine's | Medium | OPEN | API | TraderX |
| [I-94](#i-94) | The LGM is calibrated by two implementations | Low | OPEN | Architecture | Focus |

**One pipeline.** Since 2026-10-01 every run is `price_portfolio` on a `Market`
(`engine.run.pipeline`; HTTP `POST /portfolio/price`): ORE's cross-asset model with a
model per currency
(`CamConfig.ir`: the LGM by default, or Hull-White, decision A-1 in
[compliance/decisions.md](../../compliance/decisions.md)), and ORE's valuation of every trade on
every path. "Market path" in older entries and commits means this pipeline; "the Hull-White
model" before 2026-10-01 meant the separate pipeline removed that day.

---

## Correctness

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

<a id="i-49"></a>
### I-49 — Per-path recalibration differs from ORE's in two details

**Severity:** Medium · **Status:** OPEN · *Difference from ORE* · **Found:** 2026-09-29

**What is wrong.** Every Bermudan/American is recalibrated on each path
and date, as ORE's `ValuationEngine` does with `recalibrate = true`
(`engine.pricing.bermudan`). Two details differ from ORE's source:

- ORE keeps the parametrization's time grid from the as-of build; the engine measures each
  date's bucket times from that date.
- ORE still passes helpers whose expiry has passed; the engine's basket on a date keeps only
  later exercise dates. So on a date with an exercise left but no helper ahead (an American's
  last days, after the last reference-grid date in its window), the engine prices on its
  configured volatility, as a calibration today with no helper does ([I-82](#i-82)), where ORE
  would keep the parametrization it last calibrated.

The path-date volatility transcribes `DynamicSwaptionVolatilityMatrix` (`ForwardVariance`),
checked against the formula but not against ORE running it (no Python constructor).

**Reach.** Pipeline Bermudan/American values past t=0 and their exposure. Measured by
the ORE simulation oracle on ORE's own paths (`tests/test_ore_xva_parity.py`, the shared
portfolio, both models, 32 paths, dates 3M to 3Y): before exercise the engine's values are
0.5–4% above ORE's on every path (largest cell gap 0.5% of the trade's largest value for the
physical Bermudan, 0.6% for the American; mean 0.6–0.7% of their values), growing with the
date; after exercise into the swap both are the swap. Near the exercise boundary the gap moves
a decision: one cell in 160, a cash-settled Bermudan ORE exercised (0 afterwards) and the
engine did not (54% of its largest value). L4 at 1,024 paths does not resolve the bias.

**To close.** Decided (X-9): reproduce both details against `tests/test_ore_xva_parity.py`,
whose Bermudan/American L3 cases are strict expected failures until the cube equals ORE's to
the recalibration's tolerance. The same comparison confirms decision A-7 (an American's basket
on a path keeps the as-of reference grid), which is implemented but not yet checked against
ORE's simulation. Detail 1 is the pattern of [I-84](#i-84): ORE keeps a structure built on the
as-of date (there the parametrization's time grid) and moves its reference date.

<a id="i-73"></a>
### I-73 — A calibration that misses its basket is accepted unless it hits the bracket's ceiling, and on a path not even then

**Severity:** Medium · **Status:** OPEN · **Category:** Correctness · **Found:** 2026-10-01, the
dtype review; widened 2026-10-11 from the project audit (AUD-06, AUD-07, AUD-13)

**What is wrong.** The LGM bootstrap (`engine.calibration.ore_lgm.bootstrap_sigma`) solves each
bucket's σ on [1e-6, 0.2] with the engine's root solver, which returns the bracket's end for a
root beyond it, and flags only the top (`hit_ceiling`). Three gaps follow:

- **The floor and the fit.** A bucket whose helper is worth less than the model gives at the
  floor (a volatility dip after a spike, which no non-negative variance can fit) ends at 1e-6,
  unflagged, and nothing compares a helper's model value with its market value. Today's
  calibrations (`engine.pricing.bermudan.calibrate_on`, `engine.calibration.cam`) refuse only
  the ceiling. `tests/test_calibration_edge_cases.py::test_large_dip_after_a_spike_cannot_reprice_exactly`
  shows the mechanism on the standalone bootstrap (σ at the floor, a large RMSE).
- **A zero volatility.** The helper's market value (`ore_lgm._market`) divides by its standard
  deviation without the zero guard of the European's Bachelier (`european.bachelier`); a zero
  volatility at an ATM helper, which the surface accepts, gives a NaN market value, and the
  bucket a meaningless σ at an end of the bracket.
- **The paths.** On every path and date a Bermudan/American is recalibrated the same way, and
  `engine.pricing.bermudan.path_sigmas` drops even the ceiling flag, so a path is priced on a
  model that does not reprice its basket and nothing says so.

ORE's `LgmBuilder` computes the calibration's RMSE, logs a structured warning when it exceeds
`BootstrapTolerance`, and fails unless `continueOnCalibrationError`
(`OREData/ored/model/lgmbuilder.cpp`).

**Reach.** Bermudan/American values, today and on the paths, wherever a basket cannot be fitted:
volatility term structures with a sharp dip, zero volatilities, and paths where rates move far
enough (extreme paths, long horizons); exposure figures through them. Not seen on the test
markets. Since 1.4 the ceiling flag is also correct in float32
(`tests/test_precision.py::TestRecalibrationInFloat32`).

**Current handling.** Today's calibrations refuse a bucket at the ceiling; nothing else.

**To close.** With [I-49](#i-49): ORE's acceptance on every calibration, the RMSE of model
against market over the basket compared with ORE's `BootstrapTolerance`, a non-finite market
value or σ a miss; today's calibration refuses a miss naming the currency or trade (as ORE
without `continueOnCalibrationError`), and the paths count their misses per trade and carry
them as a result warning, as ORE's structured warning. Tests: a dip basket refused at t=0, a
zero ATM volatility refused, an unattainable path volatility counted.

<a id="i-104"></a>
### I-104 — Edge cases valued wrong: a zero-variance European, a bond awaiting a delayed payment, a long Theta, H at small a·t

**Severity:** Low · **Status:** OPEN · **Category:** Correctness · **Found:** 2026-10-10, the
project audit (AUD-08, AUD-11, AUD-12, AUD-14); confirmed in the code 2026-10-11

**What is wrong.** Four narrow cases, each a number off on an input the engine accepts:

- **A zero-variance European before expiry is 0 on the paths.** `european_cube`
  (`engine/pricing/european.py`) marks a date alive by `variance > 0`, but expiry is a date.
  Under `ConstantVariance` with a zero volatility the option's discounted intrinsic value is
  dropped. (`ForwardVariance`, the default, floors the variance, so it never meets this.)
- **A bond between its maturity and a delayed last payment is 0.** A coupon period may pay
  after its accrual end (`CouponPeriod.payment_date`), so the last coupon can pay after
  `maturity_date`, which must equal the last accrual end; `value_on`
  (`engine/pricing/cube.py`) values the bond at 0 from `maturity_date` on, and the redemption
  is paid on `maturity_date` though the coupon beside it is paid later. A weekend maturity
  with a Monday payment loses that coupon for two days: a Theta date or exposure date there.
- **Theta beyond an index tenor adds back an unfixed coupon at 0%.** The rolled market
  backfills fixings (`engine/risk/greeks/bump.py`, `theta_context`), but the add-back of flows
  paid in the roll (`_period_flows`) rebuilds the legs from the trade's own fixings and reads a
  missing rate as 0 (`nan_to_num`). With the default `theta_days = 1` every paying coupon fixed
  before the as-of date; a `theta_days` reaching a coupon that fixes and pays inside the roll
  does not, and nothing bounds it.
- **The LGM's H(t) cancels at small a·t.** `engine.models.lgm.H` computes (1 − exp(−a t))/a;
  the host's twin (`engine.models.cam`) uses −expm1(−a t)/a. The relative error is about
  ε/(a t): 2e-3 in float32 at a = 0.01 and a one-day t, its effect on a path's log discount
  of float32's own rounding size. float64 is unaffected at the reversions used.

**Reach.** Configurations no test, demo or default uses: zero volatility under
`ConstantVariance`, explicit bond payment dates after maturity, multi-month Theta; and float32
runs, by about one rounding.

**Current handling.** None.

**To close.** Liveness from the expiry date, the Bachelier formula taking zero variance (it
does); a bond valued until its last payment, the redemption's payment date stated and checked
against ORE's bond; `theta_days` at least 1 and the add-back on
the rolled market's fixings, or a horizon past the shortest index tenor refused; H by
`expm1`, host and JAX alike, with the a → 0 limit for its derivative. A test per case; the
golden snapshot within the margin.

<a id="i-54"></a>
### I-54 — No swaption smile: options away from the money read the ATM vol

**Severity:** Medium · **Status:** OPEN · *Difference from ORE* · **Found:** 2026-09-29

**What is wrong.** `engine.market_data.market.SwaptionVolSurface` is an ATM normal matrix (expiry × tenor).
ORE reads a vol cube or SABR smile at each option's strike. Every European, and every
Bermudan/American calibration helper (struck at the deal rate, `CoterminalDealStrike`), reads
the ATM vol.

**Reach.** None while markets are ATM-only, as every market given to the engine so far is.
With a smile: European NPV and Vega away from the money, and calibrated Bermudan/American σ.

**To close.** Decided (X-5): a strike axis in the market's volatilities, read at each
option's and helper's strike as ORE reads its cube. SABR is [F-02](features.md#f-02).

<a id="i-111"></a>
### I-111 — Reported standard errors treat Sobol paths and historical windows as independent

**Severity:** Medium · **Status:** FLAGGED · **Category:** Correctness · **Found:** 2026-10-09,
the storage harness's coverage (I-75); 2026-10-10, the project audit (AUD-19); split from
[I-55](#i-55) when the evidence table closed it, 2026-10-11

**What is wrong.** Every standard error a result reports is a sample standard deviation over
the square root of a count, as if the paths or scenarios were independent draws. They are not:
a scrambled Sobol sequence's points are dependent within a draw, the 32 paths of a storage block
share a scale and offset, and overlapping historical horizons share days. Measured across
independent seeds by the precision harness (16 seeds; §15.4 and §15.6 of
[details/precision.md](details/precision.md)):

- the paired estimate's total standard error (`MeanEstimate.standard_error`) is about three
  times too wide on Sobol paths (standardized errors' root mean square 0.30 to 0.40);
- the paired correction's (`MeanEstimate.correction_standard_error`) is right for storage of
  the cube (root mean square 0.95 to 1.03 in every format) and for FP8 shocks and curves (0.94
  to 1.09), but too small with the shocks, states or curves stored in float16 or bfloat16: root
  mean square 1.5 to 3.3 and single errors up to 74 standard errors, at 1,024 and 4,096 paths,
  in figures that differ from float64 on few paths (a rounded curve that flips an option's
  payoff on one path), though pooled coverage stays 0.91 to 0.97. A standard error by blocks of
  paths (batch means over the storage blocks, the fix I-55 proposed) was measured on 2026-10-11
  against a threshold fixed before (root mean square within 0.75 to 1.33 and coverage at least
  0.90 for every case): it fails it, being no better (root mean square 1.0 to 3.2, coverage
  0.83 to 0.94), the misses being rare large differences that no block structure explains;
- market risk's `ES_<q>_standardError` is the tail's standard deviation over the square root of
  its count; its calibration on Sobol or historical scenarios is unmeasured.

**Reach.** The standard errors in the precision report and market risk's ES standard error: a
reader who takes one as a 95% interval is wrong by the factors above. No figure is wrong: the
figures themselves and the evidence table, which judges every row by its errors across
independent seeds, never by these standard errors, are unaffected.

**Current handling.** Flagged on the result: a paired run with the shocks, states or curves
stored below 32 bits carries a warning that its correction's standard errors understate the
error (`engine.precision.evidence.paired_error_warning`), and every market-risk result one that
its ES standard errors treat the scenarios as independent; `MeanEstimate`'s docstring states
the measurements.

**To close.** Standard errors from independent replicates within a run: the run's paths as K
independently scrambled Sobol sequences (randomized quasi-Monte Carlo), each figure's standard
error the spread of its K replicate values over sqrt(K), the paired correction's likewise per
replicate; for historical scenarios, block resampling over non-overlapping windows. The
replicates move the run's numbers (their union is not one Sobol sequence), so the precision
harness measures both the coverage and the change; the project audit (AUD-19) recommends the
same. Red first: on the paired cases of the `linear` campaign, the replicate standard error's
standardized errors have a root mean square within 0.75 to 1.33 where today's reaches 3.3.

<a id="i-98"></a>
### I-98 — FP8 compute keeps a bias: each path's operand is rounded to nearest on FP8's grid

**Severity:** Medium · **Status:** OPEN · **Category:** Correctness · **Found:** 2026-10-11, the
`compute` campaign rerun with the shared operands in two terms (I-96, decision A-23)

**What is wrong.** Since each product carries the operand every path shares in two terms of
its format (A-23), what is left of its error below float32 is the per-path operand's own
rounding. On the 10-bit grids (float16, TensorFloat-32) and bfloat16's that averages out: one
stage computed in the format moves no mean figure by more than 1.2 of its statistical errors at
16,384 paths (compliance/precision-measurements/compute.md). On FP8's it does not. Rounded to
nearest on a grid whose spacing doubles at every power of two, a smooth spread of values has a
mean error that is not zero: on 786,432 operands near 1 ± 0.3, each row scaled as the products
scale it, −2.7% of the rounding's own spread for e4m3, −5% for e5m2 (float16's 0.02%). So a stage
computed in FP8 still moves its figures systematically: pricing in e4m3 9.5 statistical errors on
a mean at 4,096 paths and 30 at 16,384 (significance 35 and 86), e5m2 a quantile 73 and 213 (a
payer swap's 5-year PFE), one stage's means 8 to 34 times closer than the single pass's (I-96)
but growing with the paths as a bias does. Neither a third term of the shared operand nor the shared operand
unrounded (eight terms) changes any FP8 case beyond its noise ([precision.md
§15.5](details/precision.md#155-compute-below-float32-2026-10-10)).

**Reach.** Runs that compute a stage in FP8 (`float8_e4m3fn`, `float8_e5m2`; opt-in). Every
other compute format is unaffected. Means are corrected by a paired float64 sample
(`Precision.paired_fraction`, A-13); quantiles are not: market risk's VaR and ES at 8,192
scenarios are 0.26 (e4m3) and 0.42 (e5m2) statistical errors off.

**Current handling.** The precision report names the compute format; the precision harness
measures the bias, and the evidence table (2026-10-11,
[compliance/precision-evidence.md](../../compliance/precision-evidence.md)) passes FP8 compute
for no figure it moves at any path count measured (only figures no format moves, such as a
bond's ENE, pass), so every run that computes a stage in FP8 carries the table's warning,
naming its figures.

**To close.** Round the per-path operand without bias: stochastically, as storage can
(`Precision.rounding`), each path drawing its own uniforms so that its rounding depends on no
other path (I-75's rule). On the product alone it removes the per-path part (e4m3 at 65,536
paths: significance 12.3 to 4.7, the rest the second term's own rounding) at 40% more root mean
square error, the trade storage measured too (stochastic rounding worse for VaR and ES in 12 of
16 storage comparisons, §15.4). So: a choice of the policy, both measured in the `compute`
campaigns, the FP8 default decided by the evidence table's path ceilings; FP4 compute (F-07)
meets the same rounding, coarser. Red first: an FP8 campaign case whose significance at 4,096
paths falls below 3 with it and not without.


<a id="i-97"></a>
### I-97 — XLA computes some native GPU products wrong, so FP8 and bfloat16 passes run emulated on a GPU

**Severity:** Medium (a stated goal unmet: the speed of native units) · **Status:** OPEN ·
**Category:** Performance · **Found:** 2026-10-10, the device check (`tests/support/precision_devices.py`);
located 2026-10-11

**What is wrong.** Three matrix products XLA compiles for the GPU return wrong numbers without
an error (RTX 5060, compute capability 12.0, jax and jaxlib 0.10.2, driver 592.01; each against
the exact product of the same operands, relative to `|a|·|b|`, five shapes):

- **FP8** (e4m3 and e5m2 operands, float32 result): 0.39 to 0.72 off on every shape but the
  matrix-vector one, under any precision flag. XLA compiles these products with its own Triton
  GEMM emitter; with `--xla_gpu_enable_triton_gemm=false` it calls cuBLASLt instead
  (`__cublas$lt$matmul$f8` for e4m3) and every shape is exact to 3e-8. The card's FP8 units
  and NVIDIA's library are right; XLA's Triton code for FP8 on this architecture is not.
  e5m2 by e5m2 goes to cuBLASLt's non-FP8 path, consistent with FP8 units that do not take two
  e5m2 operands: a native e5m2 product would pair it with e4m3.
- **bfloat16 passes** (`BF16_BF16_F32_X3`): 2e-6 to 1.4e-5, as three passes should be, on every
  matrix shape, Triton or cuBLASLt; 9.9e-4, a single pass's error, on the matrix-vector shape
  (512 by 64 by 1), which XLA compiles as a fusion, not a GEMM, ignoring the requested
  algorithm. `--xla_allow_excess_precision=false` changes nothing.
- **float32 operands with a float64 result** (`preferred_element_type`): float32's error (2e-7)
  through Triton; cuBLASLt refuses it ("GEMM is not supported"), so only Triton's answer is
  silent. Fixed in the engine on 2026-10-11 (`engine.precision.products`: float32's values
  multiplied as float64).

**Reach.** No number the engine reports: `product_execution` marks FP8 and bfloat16 passes
emulated on every GPU, and the emulation is exact. What it costs is speed: on a GPU the FP8
compute formats run on float16 units, and bfloat16 passes as three float32 products, so their
timing on a GPU (F-07's research result) is not yet measurable. Unknown elsewhere: H100
(compute capability 9.0) is XLA's most exercised FP8 target and may compile FP8 another way;
the matrix-vector and float64-result faults look independent of the architecture.

**Current handling.** Emulated on a GPU; the precision report says so per format. The device
check covers every allowed arithmetic and a matrix-vector shape.

**To close.** Native FP8 and bfloat16 passes on any GPU that has the units, each shown right by
the device check: (1) route the engine's FP8 products to cuBLASLt (the engine worker's
`--xla_gpu_enable_triton_gemm=false`, timed against Triton on every other product, or a
narrower mechanism if XLA offers one per product); (2) a product format's matrix-vector product
run in a shape XLA honours the algorithm for, or emulated; (3) e5m2 paired with e4m3 where the
units need it; (4) the device check on H100 and on a newer jax, and minimal reproductions
reported to XLA. Red first: the device check with the formats marked native fails on the RTX 5060
today (FP8 0.7 off) and passes after.

---

## Validation

<a id="i-51"></a>
### I-51 — Sensitivities not checked against ORE's sensitivity analytic

**Severity:** Medium · **Status:** OPEN · **Found:** 2026-09-29

**What is missing.** `engine.risk.greeks.bump` implements ORE's definitions (zero-rate
shifts at the curve tenors, forward-difference Delta, `up − 2·base + down` Gamma, Vega per
quote, Theta on the rolled market), checked for internal consistency
(`tests/test_sensitivities.py`), but never against an OREApp sensitivity run. A different
shift convention in ORE's simulation market would pass every current test.

**To close.** Run ORE's sensitivity analytic through the oracle on the shared portfolio
(`tests/support/portfolio.py`), compare per trade, factor and tenor to 1e-8 relative
([details](details/ore-parity-validation.md)). Check Theta's rolled market in particular:
the ORE simulation oracle found that ORE's simulation market holds its tenor points at the times from the
as-of date as its reference date moves ([I-84](#i-84)), where the engine's Theta market
measures them from the Theta date (`engine.risk.greeks.bump.theta_context`).

<a id="i-78"></a>
### I-78 — AD and bump Greeks agree on flat curves only; on a sloped market they differ by up to 2%

**Severity:** Medium · **Status:** OPEN · **Category:** Validation · **Found:** 2026-10-04,
roadmap review (moved from F-01)

**What is wrong.** The AD Greeks method (decision A-5, `GreeksConfig.method="AD"`) is a
shipped option for every model and engine. It agrees with ORE's bump-and-revalue Greeks on
flat curves (`tests/test_greeks.py::TestAgainstTheBumpMethod`: parallel Deltas to 1e-4 once
the forward difference's curvature is removed). On the sloped shared market the two differ:
a near-par swap's discount Delta by 2% and a European's Vega by 0.9%. The AD derivatives are
correct derivatives of the engine's own curves (finite differences hold to 1e-6 there); the
gap is that the bump method differentiates ORE's sensitivity market and AD the market's curve
representation, so an AD Greek is not ORE's Greek on a sloped curve. A Bermudan's AD Delta
also holds its calibration fixed while its bump Delta recalibrates.

**Reach.** Every sloped-market result run with `method="AD"`, by the percentages above;
the default (bump) is unaffected.

**Current handling.** The difference is stated in the test's docstring, not on the result.

**To close.** With [I-51](#i-51): differentiate the sensitivity market's representation
(so AD equals the bump halves on the sloped shared portfolio, `tests/support/portfolio.py`,
per trade, factor and tenor), or label AD Greeks as the derivative of a different quantity
and state the gap; measure and document the Bermudan's fixed-calibration difference.

<a id="i-103"></a>
### I-103 — Malformed market and trade inputs are accepted: an unsorted curve prices silently, a late expiry crashes

**Severity:** Medium · **Status:** OPEN · **Category:** Validation · **Found:** 2026-10-10, the
project audit (AUD-05, AUD-10, AUD-35, AUD-40); measured 2026-10-11

**What is wrong.** The checks before work stop short of what the pricers assume:

- **Curves.** `ZeroCurveConfig` (`engine/market_data/market.py`) and its API schema check
  nothing. On the shared test portfolio's market, a discount curve with pillars [5, 1, 10]
  prices the European payer at 50,697.19 instead of 51,133.41, with no error; a NaN rate gives
  a NaN price; three times and two rates fail inside the interpolation (`xp and fp must be
  one-dimensional arrays of equal size`), after the job has started.
- **An expiry inside the last accrual period.** `SwaptionConfig` checks that expiry precedes
  maturity, but `european_terms` (`engine/pricing/european.py`) looks for each leg's first
  whole period after expiry with `next` and no default: an expiry of 2028-12-01 on a
  2027-01-15 → 2029-01-15 swap passes the config and raises a bare `StopIteration` when
  priced.
- **The standalone calibration's notional.** `POST /calibration/lgm` (`CalibrationRequestSchema`,
  `build_coterminal_basket`) takes any `notional`, though its helpers' prices scale with it: on
  a 1Y/2Y/3Y basket into 6Y at 0.9% normal volatility, 1e6 fits σ 0.0096–0.0110, while 0 returns
  σ ≈ 7e-5 and −1e6 every bucket at the bracket's floor, 1e-6, both with no error (measured
  2026-10-11; trade options refuse such a notional since I-101).
- **Preflight.** `validate_request` (`engine/run/pipeline.py`) and market risk's
  `validate_portfolio` check trades against the market's names, not the market's values, so
  these become failed jobs, not `400`s.
- **Unknown fields.** Four API models accept and drop fields they do not know, though the API
  documents refusing them: `ZeroCurveConfigSchema`, `CouponPeriodSchema`,
  `CalibrationRequestSchema` (`engine/api/requests.py`, plain `BaseModel`, not `_Strict`) and
  TraderX's `EodSubmissionSchema`.

**Reach.** A caller's mistake: a mis-ordered or NaN curve gives a wrong or NaN number with no
error, the others an error late. Well-formed inputs are unaffected. The swaption volatility
surface is validated (shape, finite, non-negative, increasing tenors).

**Current handling.** None.

**To close.** Host-side checks on construction, before any JAX work: a curve's pillars
non-empty, as many rates as times, finite, times non-negative and strictly increasing
(negative rates allowed); an exercise that leaves no whole period of a leg refused, naming the
trade and the dates; the standalone calibration's notional positive (until [I-94](#i-94)
retires it); every API model strict (`extra="forbid"`). The run request's preflight
([I-87](#i-87)) then runs them all. A test per refusal, red first.

---

## Performance

<a id="i-53"></a>
### I-53 — On Windows, pricing runs slower after AD Greeks; Newton's bootstrap compiles 3–4 s longer from scratch

**Severity:** Low · **Status:** PARTIAL · **Category:** Performance · **Found:** 2026-09-29 ·
re-measured 2026-10-02, 2026-10-05, 2026-10-06, 2026-10-07

**Closed parts.** The pipeline was slow, first by compiling the same work again per trade, path
date, bump and call (pricers jitted once per shape, 2026-10-02; [I-21](#i-21), [I-22](#i-22));
then a repeated job compiled 9 programs, the AD Greeks' derivatives living only in JAX's internal
caches (`tests/test_profiling_and_jit.py::TestCompileCounts::test_a_repeated_job_compiles_nothing`,
`::test_an_option_s_greeks_are_a_program_per_derivative`, both red on the code before); and the
recalibration of every Bermudan/American, on every path date and in its Greeks, ran bisections
of 60 steps per bucket around 160 per y\*, the dates one after another (decision
A-21: one root solver, Newton by default, the dates of one basket shape calibrated together;
`tests/test_root_solvers.py`). Measured untraced, old and new code alternately: the path
recalibration 35–55 times faster on the CPU (262,144 paths: 144–185 s to 2.8–3.4 s) and 15–38
times on the RTX 5060 (7.3 s to 0.19 s); the demo's repeated job on the GPU 5.4–5.6 s to
1.35–1.43 s, and its trace 32 s to 1.6 s
([profiling §2.0](../concepts/profiling.md#20-the-demo-measured-2026-10-05)).

**What is wrong.**

- **On Windows only, the AD Greeks slow later pricing.** In a process that has run any AD
  Greeks, every later job's pricing phase runs 10–25% slower: pricing-only jobs 1.57–1.83 s
  before one Greeks call and 1.98–2.25 s after, the code before Newton and after it alike; on Linux
  1.32 → 1.32–1.38 s (2026-10-06). No compile, trace or garbage collection is involved and one large
  allocation does not do it; the Windows heap is suspected, not shown. The deployments that
  matter run Linux.
- **A cold job is 3–4 s (8–11%) slower since the Newton solver.** Each bootstrap bucket's program under Newton
  traces the helper's price, with its nested y\* solve, at the bracket's ends, in the step (under
  `jax.jvp`) and for the model value: the demo's ten bucket programs compile in 5.4 s against
  3.3 s (7.1 s before the bracket's two ends were evaluated in one call). The programs stay in
  the worker and on disk, so a repeated or restarted job does not pay it.
- **The leg kernel in difference form is a third slower on its own on a CPU** (since the kernel
  rewrite, 2026-10-10, F-07). A swap's cube on 4,096 paths and 20 dates takes 13.4 ms against
  10.2 ms in float64, though the operations are the same work (11.6 ms against 11.7 ms summed
  over a trace): the old code's one reduction overlapped its two operand fusions, the new
  kernel's products (one per block of columns) run after theirs. End to end it is within the
  run-to-run spread: a five-trade linear portfolio 4% slower (medians 0.370 s and 0.352 s, three
  interleaved passes), market risk on its trades three times faster (0.075 s against 0.23 s).
  Concatenating the blocks into one product's operand cost a quarter more; two columns per coupon
  doubled it (both measured and discarded). Accepted by the owner (2026-10-11): the one kernel at
  every precision stays (A-16), rather than a float64 kernel of its own beside it; the fix is the
  batching below, done with the timing on accelerators if it shows there, else when fixing what
  exists.

**Reach.** Wall time only; no number. Windows CPU runs; a worker's first job; float64 path
pricing of linear trades on a CPU.

**Current handling.** None needed for correctness.

**To close.** The Windows slowdown's cause, and the helper's price traced once per
bucket program, the snapshot within the margin; the leg kernel's products scheduled with their
operand fusions (or the trades of one shape priced as one product, their columns side by side,
which a matrix unit wants anyway), timed against the code before the kernel rewrite. Related: the grid rollback's memory and time on the
paths ([I-83](#i-83), F-07) and the cold job's one-operation programs ([I-81](#i-81)).

<a id="i-83"></a>
### I-83 — A Bermudan/American on the paths needs 0.3–0.7 MB per path and step: 64k paths run out of memory

**Severity:** Medium · **Status:** OPEN · **Category:** Performance · **Found:** 2026-10-07,
the Newton solver's baseline

**What is wrong.** The grid engine on the paths (`engine.pricing.bermudan._rollback_every_path`,
the rollback vmapped over paths) builds each step's `[nodes, nodes]` interpolation operator per
path: arrays `[paths, steps, nodes, nodes]`, 193 nodes at the demo's grid (`n_per_std=16`,
`std_devs=6`), 301 at ORE's default (30 and 5), so 0.3 MB (0.7 MB) per path and exercise step.
The demo's job at 65,536 paths asked for 277 GB and failed (`Out of memory allocating
277260800480 bytes`) on a 31 GB machine, before 2026-10-07 and after it; 4,096 paths fit, and
their time is this rollback (the options' path cubes take 31–36 s of the CPU's 36 s pricing,
their recalibration 0.09 s, against 2.3 s for the whole job at 256 paths). The 8 GB GPU fails
at 1,024 paths (a 6 GB allocation).

**Reach.** Every pipeline run with a Bermudan or American beyond a few thousand paths: the
path counts the precision research needs (F-07), any exposure run of size, and the
ORE distribution test's options (`tests/test_ore_xva_parity.py`, L4 at 1,024 paths where the
linear trades run 16,384). Not market risk (t=0 revaluation) or the t=0 Greeks.

**Current handling.** None: the job fails with XLA's out-of-memory error.

**To close.** The rollback as one `[nodes, nodes]` operator per step applied to
all columns and paths at once, as a matrix product ([details/precision.md
§8.3](details/precision.md#83-emulation-and-native-speed)), which holds one such buffer per
step; then the demo's job at 65,536 and 262,144 paths on the CPU and the GPU, the measurement
the Newton solver's work could not make.

<a id="i-105"></a>
### I-105 — Storage below float64 does not lower a run's peak memory: the reductions load the whole cube at float64

**Severity:** Medium · **Status:** OPEN · **Category:** Performance · **Found:** 2026-10-10, the
project audit (AUD-42); confirmed in the code 2026-10-11

**What is wrong.** `engine.run.pipeline` loads the stored cube whole at float64
(`load(valuation.cube, jnp.float64)`; `PortfolioValuation.cube` stacks every column at once),
and the exposure reductions read it whole, beside the paired float64 cube when there is one. A
cube stored in FP8 holds one byte per path, date and trade, but the run's peak still holds
eight, whether or not the result exports the cube (`cube_output`).

**Reach.** The path count a run reaches on one device, which the precision research's
equal-time comparison rests on ([F-07](features.md#f-07)): many low-precision paths need the
memory of as many float64 paths at the reductions. No number is wrong. The rollback's working
memory ([I-83](#i-83)) is larger today and comes first.

**Current handling.** None.

**To close.** The reductions over the stored columns trade by trade, each loaded at float64
alone: a trade's profile from its column, the netting set from a running float64 sum
`[paths, dates]`, PFE's quantiles from that sum as today (exact, not approximated); the float64
cube built only when the result asks for it. The peak measured against path count before and
after, at float64 and at FP8 storage; the golden snapshot within the margin.

<a id="i-61"></a><a id="p-1"></a>
### I-61 — Nothing runs on more than one host; multi-device speed unmeasured

**Severity:** Medium · **Status:** PARTIAL · **Found:** 2026-09-24, audit P-1

**Closed part (2026-10-04, the one-host half).** A job's scenarios are split across
the devices of its host (`engine/market_simulation/sharding.py`): the simulation's Sobol normals and
market risk's shifts are placed on a one-axis mesh along the scenario axis, and XLA's sharding
propagation carries the split through path evolution, the scenario market, pricing and the
stored cube, gathering at the reductions (exposure, VaR/ES). Market risk's memory-bounded
batches take their scenarios from every device's share. The device count is the largest that
divides the scenario count, capped by `JAX_RISK_SCENARIO_DEVICES`; on one device nothing is
placed, so one-device runs are unchanged bit for bit. On four XLA host devices a portfolio run
(LGM, Bermudan, paired sample), an FP8 stochastic-rounding run and a market-risk run hold
their results on all four devices and equal the one-device run to about 6e-16 relative (the
batch shape's rounding; FP8 bit for bit). Tests: `tests/test_sharding.py`; red first: before
the change every result sat on one device.

**What is wrong.** The project's goal is to run across multiple TPUs and compare many
low-precision paths against fewer FP64 paths in equal wall time. Two parts are missing:

- **Several hosts.** On a pod slice every host must run the same job (SPMD); the worker
  claims jobs for its own host only and uses `jax.local_devices()`.
- **Speed.** The split is verified for its numbers on CPU host devices, which share the
  host's cores, so it says nothing about speed. Scaling on real devices is unmeasured.
- **Awkward path counts use fewer devices, silently.** The device count is the largest divisor
  of the path count (`scenario_device_count`): 1,001 paths on eight devices run on seven, a
  prime count on one, with no warning (found by the project audit, AUD-45).

**To close.** One worker per host (`jax.distributed.initialize`), process 0
claiming each job and handing it to the others
([details](details/precision.md#113-multi-device-and-multi-host)); then wall time against
device count on TPU (and H100), feeding the research result ([F-07](features.md#f-07)). A run on fewer devices than
it was given says so in its precision report (or pads its paths to a multiple of the devices
and drops the padding before any reduction); the storage blocks' alignment with each device's
share is checked on the real devices.

---

## API

<a id="i-08"></a>
### I-08 — A running TraderX attempt is lost on restart

**Severity:** Medium · **Status:** PARTIAL

**Closed part (2026-10-04).** Portfolio jobs: the durable SQLite job queue
(`engine/api/job_queue.py`, decision A-14) replaced the in-memory `_JOBS` dict. A job
survives restarts, every API process on the queue serves every `job_id`, a job running when
its worker died reads `interrupted`, and a failure names its class (`bad-terms`,
`missing-market-data`, `unsupported-product`, `numerical-failure`, `infrastructure`).
Tests: `tests/test_engine_worker.py` (`test_jobs_survive_reopening_the_file`,
`test_a_worker_killed_mid_job_leaves_it_interrupted`,
`test_supervisors_of_one_queue_share_one_worker`, the failure classes); red first: on the code
before 2026-10-04 a job id from one app process was a `404` in the next.

**What is wrong.** The TraderX path is durable for finished work
(`engine/traderx/publication.py`: manifest as commit point, scan recovery, idempotent
`submissionId` across restarts), but a *running* attempt is memory only and reads as unknown
after a restart. The store is single-machine.

Three more gaps of the same store (found by the project audit, AUD-24 to AUD-26, confirmed
2026-10-11):

- **A failed workload is unknown after a restart.** `ResultStore.lookup` serves successes only,
  by design, and `AttemptStore.lookup` falls back to attempts in memory; after a restart a
  workload whose only attempt failed, published, reads as `UNKNOWN_WORKLOAD` (its attempt is
  still found by id).
- **Bindings and publication are one process's.** `AttemptStore` holds an in-process lock and
  `ResultStore` has no cross-process coordination, so two API processes on one directory can
  bind one id to two workloads and both publish.
- **Power loss.** Publication fsyncs each file and renames it atomically, but does not fsync
  the directory after the rename, so on Linux a published manifest may not survive a power cut
  that the code's comments call durable; a failed stage can leave its temporary file behind.

**To close.** On the TraderX path, the durable accepted-attempt record, a boot sweep
and an `interrupted` lookup state (TraderX acceptance case A-09); the portfolio queue's
`interrupt_running` is the pattern. With it, attempts, bindings and terminal states in the
queue's SQLite transactions (one owner across processes), a workload's latest terminal state
served when it has no success, and stale stages swept at boot. Until then one TraderX process
per store, as [docs/reference/traderx-path.md](../reference/traderx-path.md) ("What this is
still not") says of publication, and now of bindings too. Tests: a
fresh store finds a published failure by workload; two processes binding one id to two
workloads, one refused.

<a id="i-23"></a>
### I-23 — `accrualBasis` strictness rests on an unconfirmed reading

**Severity:** Medium · **Status:** ASSUMPTION · **Raised:** 2026-09-16

**The premise.** `engine/traderx/terms.py` accepts exactly `dateBasis = SESSION_DATE`,
`settlementAdjustment = NONE`, `rounding = HALF_EVEN`, schema
`traderx.accrual-basis.v1`, and refuses anything else. TraderX has not said whether new
values land in `v1` or force a `v2` (asked in responses v4, v6 and v7).

**Risk.** If they add values in place, bundles they consider valid are refused on the day a
real calendar is exported. That fails safe (a refusal, not a wrong number), but reads as a
defect on call. A `TermsJoinError` naming these fields may be this allowlist, not a bad
bundle.

**To close.** Their answer. "New version": close with no change. "In place": widen
`SUPPORTED_DATE_BASES` / `SUPPORTED_SETTLEMENT_ADJUSTMENTS` / `SUPPORTED_ACCRUAL_ROUNDING`
with a test per value. `tests/test_traderx_terms_v2.py::TestUnrecognizedValuesAreRefused`
pins today's rule.

<a id="i-58"></a>
### I-58 — TraderX: two concurrent submissions of one workload under different ids both execute

**Severity:** Medium · **Status:** PARTIAL · **Found:** 2026-09-17, TraderX case A-03 (FR-08)

**Closed part (2026-10-10, with [I-57](#i-57)).** A retry under the *same* `submissionId`
while its attempt is pricing is answered `200 {"state": "running", "reused": true}` and
not executed: `AttemptStore.submit` returns `reused` for an attempt it did not create, and
the route prices only a new one. Before, the retry priced the same attempt again and the
second to finish raised on the immutable attempt, a `500`. Test:
`tests/test_traderx_routes.py::TestARetryOfARunningSubmissionIsNotExecutedAgain`; TraderX's
A-03 starter retries under one id, so it now passes and does not exercise the open part.

**What is wrong.** Two submissions of one workload under *different* ids, or with none, while
nothing is completed (or with `reuseExistingResult=false`) each get a running attempt, and
both run `price_bundle`. The lock protects the binding, not the computation.

**To close.** In `AttemptStore.submit`, under its lock, a per-workload running-attempt index:
a submission of a workload with a running attempt joins it (`reused`, so the route does not
price it), and its id is bound to that attempt. Decide there whether a joined or retried
running attempt answers `200 running` (today's retry) or `202`, the same for both.
Regression test with a deterministic barrier, not timing. Cross-process exclusion is not
claimed.

<a id="i-59"></a>
### I-59 — TraderX: `calculations` and `reportingCurrency` accepted, keyed, then ignored

**Severity:** Medium · **Status:** OPEN · **Found:** 2026-09-17, TraderX cases A-04, A-05
(FR-06); still present

**What is wrong.** `EodSubmissionSchema` accepts both, `_key_for` hashes both into the
workload key, and `price_bundle(bundle, request.marketInputs)` receives neither. An `EUR`
request returns USD results with 200 OK, cached separately from the identical USD run; an
unknown calculation name is accepted.

**To close.** As decided (A-18, as ORE fails on an unknown analytic): reject
an unknown calculation (400 `UNKNOWN_CALCULATION`, allowlist
`engine.traderx.result.CALCULATIONS`); add `reportingCurrencies: ["USD"]` to the
capability document and reject others (400 `UNSUPPORTED_REPORTING_CURRENCY`). Tests assert
the consequence, not that the field parses. Reporting in other currencies is
[F-08](features.md#f-08), once an FX source exists.

<a id="i-107"></a>
### I-107 — TraderX: `bundleId` is trusted, not recomputed, so a changed bundle can get the old result

**Severity:** Medium · **Status:** OPEN · **Category:** API · **Found:** 2026-10-10, the project
audit (AUD-22); confirmed in the code 2026-10-11

**What is wrong.** The exchange contract
(`reference/traderX/specs/YU18-eod-risk-bundles/contracts/bundle-v2-and-terms.md`) defines
`bundleId` as the SHA-256 of the canonical manifest without it (sorted keys, two-space
indentation, ASCII escaping, finite numbers, UTF-8, one final LF). `engine.traderx.bundle`
checks each artifact against the hash the manifest gives, but takes `bundleId` as given
(`Bundle.bundle_id`), and the workload key is built from it (`_key_for`,
`engine/api/traderx_routes.py`), not from the manifest's content. A bundle re-exported with a
corrected artifact and that artifact's new hash, under the old `bundleId`, loads, gets the old
workload key, and with `reuseExistingResult` is answered with the old bundle's result.

**Reach.** Only a bundle that breaks the contract's identity rule (an exporter defect or a hand
edit); then a stale result for changed positions or terms, with `200`. A conforming exporter
changes the id with any artifact.

**Current handling.** Artifact hashes are checked against the manifest; the id is not.

**To close.** On load, the canonical manifest hash recomputed and a mismatch refused (`422`,
naming both ids); test fixtures that edit an artifact rebuild the id. Tests: a changed artifact
under the old id refused; the rebuilt bundle gets a new workload key.

<a id="i-108"></a>
### I-108 — TraderX: NaN amounts priced as `ok`, terms contradicting the row priced, duplicate rows counted twice

**Severity:** Medium · **Status:** OPEN · **Category:** API · **Found:** 2026-10-10, the project
audit (AUD-27 to AUD-31); confirmed in the code 2026-10-11

**What is wrong.** A bundle's rows are checked for shape and identity, not for every property
the pricers assume:

- **Non-finite numbers.** `normalize._parse_float` accepts `nan` and `inf` as Python's `float`
  does (`quantity`, `coupon`, `closingMark`, `accruedInterestFraction`), and so do the terms'
  numbers (`note._terms_float`, `bill.py`). A NaN quantity is priced to a NaN value with status
  `ok`; publication writes it as `json.dumps`' `NaN`, which is not JSON, and an HTTP answer
  that serializes it fails.
- **Terms that contradict the row.** `terms.join_terms` matches rows to entries by identity and
  refuses duplicate or unused entries, but compares no economics: the row's `coupon` is parsed
  and the terms' `couponRatePercent` priced, never checked against each other, though the
  contract forbids a supplement changing a booked rate, index, date, currency or direction.
- **Duplicate rows.** Duplicate terms entries are refused; a position row repeated (one account
  and security twice) or a contract row repeated is priced and aggregated twice.
- **Structure.** A manifest that is not a JSON object, or a field of the wrong type, fails
  outside `BundleIntegrityError` (a `500`, not a `422`); `CalculationOutcome.to_dict`
  (`engine/traderx/result.py`) lets a `payload` overwrite `status` and `value`.

**Reach.** Bundles with such rows, an exporter defect: then a NaN or double-counted figure
reported `ok`, or a `500`. Conforming fixtures are unaffected.

**Current handling.** Artifacts are hash-checked, terms entries de-duplicated, and accrued
interest reconciled against the exporter's figure, which can catch a contradicting coupon when
the accrual is not zero.

**To close.** Every parsed number finite (the row refused with a reason, not priced); each
joined row's booked fields reconciled with its terms (rate, currency, maturity, direction), a
contradiction refused with its own reason; repeated position and contract identities refused;
the manifest's structure validated before `Bundle` is built (`422` naming the field);
`CalculationOutcome` keeping its own keys; publication with `allow_nan=False`. A test per case,
from the CSV to the HTTP answer.

<a id="i-109"></a>
### I-109 — TraderX: after a failed publication a same-id retry never prices or publishes

**Severity:** Medium · **Status:** OPEN · **Category:** API · **Found:** 2026-10-10, the project
audit (AUD-23); confirmed in the code 2026-10-11

**What is wrong.** `Attempt.complete` (`engine/traderx/workload.py`) publishes before it changes
state, so a publication error leaves the attempt `running` and its priced result dropped. The
route answers `500 RESULT_NOT_PUBLISHED` and asks the coordinator to retry; the retry under the
same `submissionId` gets the bound attempt back as `reused` ([I-58](#i-58)'s closed part), and
the route returns `running` without pricing or publishing. Until the process restarts, that
submission cannot complete, even after the storage recovers. The staging file's open, write
and fsync also raise outside the `PublicationError` the route catches.

**Reach.** A TraderX submission whose publication fails transiently (a full or unavailable
disk). A new `submissionId` recovers it; the retry the route asks for does not.

**Current handling.** None.

**To close.** A priced-but-unpublished state that keeps the result, a same-id retry publishing
it (or the attempt failed retryably and the retry pricing again); every staging error a
`PublicationError`. Test: the first publication fails, storage is restored, a retry under the
same id yields one discoverable result; red first.

<a id="i-60"></a>
### I-60 — TraderX result schema has no stated policy on added fields

**Severity:** Low · **Status:** ASSUMPTION · **Raised:** 2026-09-17, response v7 §5.2

**The premise.** `jaxrisk.eod-result.v1` does not say whether consumers may pin its exact
shape. TraderX's validator does, so the next field added breaks them. Proposed to TraderX:
`additionalProperties: false` at the root and any new field is a schema version bump.

**To close.** Their answer, recorded as a decision; then state it in the published schema
with a test.

<a id="i-76"></a>
### I-76 — The job queue keeps every job and result forever

**Severity:** Low · **Status:** OPEN · **Category:** API · **Found:** 2026-10-04, building
the job queue

**What is wrong.** `engine/api/job_queue.py` never deletes a row, so the SQLite file grows by
each job's request and result document (a 4096-path cube is tens of MB of JSON inline, or
as many bytes of artifact chunks since 2026-10-07).

**Reach.** Disk on a long-running API host; no number. A result stays retrievable, which the
Basel plan's audit trail wants (P0).

**Current handling.** None; delete the file, or old rows, by hand.

**To close.** A retention policy chosen by the owner (an age or a count, kept results for
regulatory runs), applied by the worker between jobs; a test that a purged id is a `404` and a
kept one is served.

<a id="i-77"></a>
### I-77 — A worker that cannot start leaves jobs `pending` with no signal

**Severity:** Low · **Status:** OPEN · **Category:** API · **Found:** 2026-10-04, building
the job queue

**What is wrong.** If the engine worker dies at startup (a broken install, an unwritable
queue directory), the API restarts it on every poll and the job stays `pending`; the client
sees no error. The worker's traceback reaches the server's stderr, not the job.

**Reach.** HTTP portfolio jobs, only when the worker cannot run at all. A job that kills a
running worker is `interrupted`, not stuck.

**Current handling.** The traceback in the server log.

The route also queues the job before it starts the worker (`_validate_and_submit`,
`engine/api/routes.py`): if starting it raises, the client gets an error for a job that is
queued, and a retry queues a second (project audit, AUD-37).

**To close.** The supervisor records its child's exit status, and a poll of a `pending` job
while the worker has exited at startup N times running reports it (in `error`, or a
`worker` field on `/health`, which today shows only that the HTTP process is up); a failed
worker start answered with the queued job's id; a test with a worker command that exits at
once.

<a id="i-106"></a>
### I-106 — The job queue's file: an interrupted migration stops it opening; a second file starts a second worker

**Severity:** Low · **Status:** OPEN · **Category:** API · **Found:** 2026-10-10, the project
audit (AUD-36, AUD-38); confirmed in the code 2026-10-11

**What is wrong.** `JobQueue.__init__` (`engine/api/job_queue.py`) migrates a version-1 file in
autocommit statements: an `ALTER TABLE` adds the `kind` column, the schema script runs (and
commits), then `user_version` is set. A crash between the first and the last leaves the
column at version 1, and the next start repeats the `ALTER` and fails on the duplicate
column; two processes opening an old file at once can race the same way. Separately, the
worker lock is per queue file (`<queue>.worker.lock`, `engine/api/supervisor.py`), so two API
instances on one host with different `JAX_RISK_JOB_QUEUE` paths start two workers on the same
devices, against decision A-14's one worker per host.

**Reach.** Only a version-1 queue file (none outside development) and a host serving two queue
files. No number.

**Current handling.** None.

**To close.** The migration in one `BEGIN IMMEDIATE` transaction, or idempotent (the column
added only if missing), and a test that interrupts it; the worker's lock taken per host (a
fixed path per device set), or a second queue file on one host refused; a test each.

<a id="i-87"></a>
### I-87 — The HTTP API is one route per engine function, not one run request

**Severity:** Medium · **Status:** OPEN · **Category:** API · **Found:** 2026-10-08, owner
review of the API of 2026-10-07

**What is wrong.** Decision A-2 asks for a single route and a single request. The API of
2026-10-07 reached every setting but kept one route per engine function: `POST /portfolio/price`,
`POST /portfolio/market-risk`, `POST /calibration/cam` and `POST /calibration/lgm`.

- **The same input is sent twice.** Exposure and VaR on one book are two requests with the
  same market, trades, pricing and precision, and two jobs.
- **One setting has two places.** The model per currency is `simulation.ir` in one request and
  `ir` in another. The reporting currency is both `base_currency` and
  `simulation.base_currency`.
- **What to compute is mixed in with how.** `scenario_risk` and `compute_greeks` are flags
  beside the configuration they switch on.
- **The routes behave differently.** Two are queued jobs and two answer at once, and
  `/calibration/lgm` takes year fractions and `hw_a` instead of a market.

**Reach.** Every HTTP caller. No number is wrong. There are no clients yet, so this is the
cheapest time to change the contract.

**Current handling.** None; [docs/reference/http-api.md](../reference/http-api.md) documents
each route.

**To close.** One `POST /runs` with `market`, `portfolio`, an `analytics` list
and a `config` whose sections follow ORE's files, as
[details/configurable-engine.md](details/configurable-engine.md#one-run-request-i-87-i-88-i-89)
describes. A run of one analytic must equal today's direct call within the margin. The one
result says what each figure is without the request beside it (the project audit, AUD-39):
the reporting currency, the as-of and simulation dates, the trade ids on every per-trade axis
(today's Python `MarketRiskResult` keys its P&L columns by request order alone), each Greek
array's factor labels (curve and tenor, or option and swap tenor), the Greeks method and
calibration policy as run, and the engine's identity (its git revision and the versions of
jax, jaxlib and ORE) in place of the fixed `0.1.0`.

<a id="i-95"></a>
### I-95 — A precision setting the run does not read is accepted and ignored

**Severity:** Low · **Status:** OPEN · **Category:** API · **Found:** 2026-10-09, building the
storage harness (I-75)

**What is wrong.** A market-risk run has no scenario market: it reads `Precision.simulation`
for its shifts and `pricing` (with its overrides) for its revaluation, and accepts a `market`
stage, which `Precision.throughout` also sets, without reading it. Since the kernel rewrite
(2026-10-10) a stage's `compute` is the format of its matrix products, and market risk's
simulation stage has none (the shifts are drawn before the run, then rounded to the stage's
accumulate format and stored): its `simulation.compute` is accepted and not read either. A portfolio run with
`scenario_risk=False` simulates nothing and accepts every stage, the rounding, the block layout
and `paired_fraction`, and reads none of them. Both are inputs accepted and then ignored, which
the API otherwise refuses ([I-59](#i-59)'s rule; a stochastic rounding with nothing stored
below 32 bits is refused for that reason).

**Reach.** No number: the setting does nothing. But the result's precision report echoes the
policy as given, so it states a precision the run did not use. `demos/demo_precision.py` runs
market risk with `Precision.throughout("float32")`, whose `market` stage is one of these.

**Current handling.** None.

**To close.** In the run request ([I-87](#i-87)): each analytic refuses a precision setting it
does not read, naming the field, as it refuses a configuration section no requested analytic
reads (market risk: the `market` stage, and `simulation.compute` other than its accumulate
format); a test per analytic; the demo's market-risk policies set the two stages they use.

<a id="i-89"></a>
### I-89 — A retried job submission runs the job twice

**Severity:** Low · **Status:** OPEN · **Category:** API · **Found:** 2026-10-08, review of
the API of 2026-10-07

**What is wrong.** `POST /portfolio/price` and `POST /portfolio/market-risk` give every
submission a new `job_id` (`engine/api/job_queue.py`, `submit`). A client that loses the
`202` and retries queues the job a second time, and the worker prices both. TraderX's routes
have `submissionId` for this; the job routes have nothing.

**Reach.** Engine time, and a duplicate result a client may not know about. No number is
wrong.

**Current handling.** None.

**To close.** An optional `idempotency_key`. The same key and body return the
first `run_id`; the same key with a different body is a `409`, checked before anything is
returned (the binding I-57 found missing on the TraderX path). Tests for a retry, a conflicting
body, and two API processes on one queue.

<a id="i-90"></a>
### I-90 — The HTTP API has no authentication, rate limit or body-size limit

**Severity:** Low · **Status:** OPEN · **Category:** API · **Found:** 2026-10-08, review of
the API of 2026-10-07

**What is wrong.** Every route, TraderX's included, serves anyone who can reach the
port. Nothing limits how many jobs a caller queues or how large a body it sends; one large
body is held in memory and stored in the queue whole.

**Reach.** Only a server reachable from outside a trusted network. uvicorn binds
`127.0.0.1` unless started with `--host`, and every use so far has been local.

**Current handling.** None; deploy behind a trusted network or a reverse proxy that
authenticates.

TraderX's `POST /eod/price` also reads a server-side `bundlePath` and joins each manifest
path to it without containment (`engine/traderx/bundle.py`): an absolute path or `..` reads
any file the server can, its hash supplied by the same manifest; files are read whole, unbounded
(project audit, AUD-34).

**To close.** Before the API is reachable from outside a trusted network:
authentication chosen with TraderX (a bearer token or mutual TLS), a body-size limit, and a
per-caller limit on queued jobs; bundle paths confined to configured roots, every artifact
path resolved inside its bundle (symlinks included), and a byte limit per artifact; a test per
refusal (`401`, `413`, `429`, a path outside the root).

<a id="i-91"></a>
### I-91 — A long job delays every job behind it, and no job can be cancelled

**Severity:** Low · **Status:** OPEN · **Category:** API · **Found:** 2026-10-08, review of
the API of 2026-10-07

**What is wrong.** The engine worker runs jobs one at a time in submission order (decision
A-14), so a calibration queued behind a ten-minute simulation waits ten minutes. There is no
route to cancel a job, so a job submitted by mistake runs to the end.

**Reach.** Wall time of HTTP jobs when several are queued. No number is wrong. Running one
job at a time on every device is deliberate.

**Current handling.** None; stop the worker, which marks its job `interrupted`.

**To close.** With the worker loop's other changes. A cancel route: a pending
job becomes `cancelled` at once, and a running job stops at the worker's next phase boundary
(calibration, simulation, each trade's pricing, Greeks) and reads `cancelled`, with the
worker kept alive and its compiled programs kept. A test for each case. Priorities between
callers are not planned until there are several callers.

<a id="i-93"></a>
### I-93 — TraderX's submissions are priced by their own pricers, not the engine's

**Severity:** Medium · **Status:** OPEN · **Category:** API · **Found:** 2026-10-08, owner
review of the roadmap

**What is wrong.** The engine has two ways in. Portfolio, market-risk and calibration
requests reach the engine's pipeline; TraderX's end-of-day bundles (`POST /eod/price`) are
priced by the TraderX path's own closed-form pricers (`engine/traderx/bill.py`,
`note.py`) on a flat assumed rate, while the engine prices the same bills and notes in
`engine/instruments/treasury.py` on any curve, today and on every path. Two pricers of one
product can disagree, every fix is made twice, and nothing the engine gains (curves, paths,
Greeks, precision, Basel figures) reaches TraderX.

**Reach.** Every TraderX result. No disagreement between the two is known: both discount the
same cashflows, and on the assumed profile's flat curve they should agree to rounding, which
no test checks today.

**Current handling.** None.

**To close.** The TraderX path builds a run request from its bundle, after its convention
checks and refusals (so a refused booking never reaches a pricer, I-05), and the engine
prices it; its own pricers go. Its routes, result document, attempt store and the
accrued-interest reconciliation against the exporter's figure stay. First, a test that the
engine's bond and the TraderX note agree on every bundle row of the tests, to rounding
([details/package-layout.md §3](details/package-layout.md#3-merges)). The adapter then calls
`run/`, which sits above it in the layers `tests/test_import_layering.py` enforces (`traderx`
shares a layer with `models` and `instruments`): its orchestration moves above `run/` (or into
`api/`), its terms and provenance types stay low, rather than an exception in the layer test
(project audit).

---

## Architecture

<a id="i-94"></a>
### I-94 — The LGM is calibrated by two implementations

**Severity:** Low · **Status:** OPEN · **Category:** Architecture · **Found:** 2026-10-08,
evaluating the package layout

**What is wrong.** Every run calibrates with ORE's bootstrap (`engine.calibration.ore_lgm`).
An older standalone bootstrap (`engine.calibration.lgm.calibrate_lgm_sigma`, with
`engine.calibration.basket`: Bachelier helper prices, its own solver steps) serves only
`POST /calibration/lgm`, which the run request of [I-87](#i-87) removes.

**Reach.** `POST /calibration/lgm` and its Python function. The two are tested separately and
can drift; a fix to one is not made to the other.

**Current handling.** None.

**To close.** Once the run request's `calibration` analytic returns what
`POST /calibration/lgm` returns, retire the standalone bootstrap, or move it to
`tests/support/` as an independent check of ORE's if its tests are worth keeping
([details/package-layout.md §3](details/package-layout.md#3-merges)).

---

## Scope gaps

Refused with an identified reason, never approximated. Each is listed because a current
consumer (TraderX) sends the input.

<a id="i-04"></a><a id="m-2"></a>
### I-04 — Seasoned TraderX swaps: the export has no past fixings

**Severity:** High · **Status:** OPEN · blocked on TraderX · **Found:** TraderX EOD review;
widened by the 2026-09-24 audit (M-2)

**What is wrong.** A coupon fixed before the as-of date needs its historical fixing. Every
trade config takes `fixings` and refuses a missing one (`MissingFixingError`), as ORE does,
but TraderX exports no `pastFixings`, so a seasoned TraderX swap cannot be priced.

**Reach.** Seasoned swaps from the TraderX path only; the engine itself prices seasoned
swaps given their fixings, and on every path paid flows drop out and coupons fix by
`FixingManager`'s rule, under either model (the first half of this entry, the Hull-White
model keeping paid flows, closed on 2026-10-01: `tests/test_hull_white_model.py::
TestPaidFlowsAndMaturity`, `tests/test_pricing.py`).

**To close.** `pastFixings` from TraderX ([details](details/traderx-integration.md)).

<a id="i-05"></a>
### I-05 — No faithful USD-SOFR / ACT-360 swap construction

**Severity:** High · **Status:** OPEN · blocked on the D03/D04 convention agreement

**What is wrong.** `build_vanilla_swap` builds a generic term-IBOR swap (`SimIndex6M`,
ACT/365 legs, TARGET, no compounding). A USD-SOFR booking is overnight, ACT/360, US
calendars, compounded in arrears, with lookback/lockout/payment lag. ACT/360 vs ACT/365 alone
moves every accrual by 1.389% (about 46 × a 1bp DV01 on a 5Y fixed leg).

**Current handling.** The TraderX path refuses any booking outside an explicit allowlist
(`engine/traderx/conventions.py`, `CONVENTION_NOT_SUPPORTED` naming the fields; the
TraderX SOFR fixture's 13 missing terms). A direct Python caller cannot express SOFR in a
`SwapConfig`.

**To close.** The convention set from TraderX, then a new builder beside the existing one,
accepted against a same-terms ORE reference
([details/traderx-integration.md](details/traderx-integration.md)). Do not start on guessed
conventions.

<a id="i-07"></a>
### I-07 — No corporate bond, equity or listed-option pricer

**Severity:** Medium · **Status:** OPEN

**What is wrong.** Treasuries price today and on every path; TraderX's position export also carries
corporate bonds (refused: a Treasury-discounted corporate is not credit pricing, it needs a
credit model), cash equities ([I-18](#i-18): a market-data gap, not a pricer gap), listed
options, TIPS and FRNs. The simulation's equities (`CamConfig.equity_volatilities`) drive risk-factor paths; they
are not a position pricer. FX and equity trades are [F-04](features.md#f-04).

**To close.** Per instrument, on demand: a config, a pricer, ORE parity tests.

<a id="i-16"></a>
### I-16 — `rateSensitivity` is parallel-only

**Severity:** Medium · **Status:** OPEN · blocked on an observed curve (TraderX W2)

**What is wrong.** The note's `rateSensitivity` is a 1bp parallel shift, labelled
`shockedFactor: "zero-curve-parallel"`. The only market input the TraderX path accepts is an
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
TraderX path has no spot or FX input (`engine/traderx/market_inputs.py` registers flat
rate profiles only). Positions are read, validated and refused (`SPOT_SOURCE_NOT_SUPPLIED`,
`FX_SOURCE_NOT_SUPPLIED`), echoing the quantities read; `capabilities()` reports equity NPV
as `blockedOnMarketInput`.

**Do not close it** with `closingMark`: `quantity × closingMark × multiplier` reproduces
TraderX's own `marketValue` exactly, an echo presented as a valuation.
`tests/test_traderx_equity.py::TestDoesNotEchoTheExportedMark` fails against it.

**To close.** A registered spot/FX surface in `marketInputs`, or observed spots in the
bundle. The pricer is four multiplications.

---

## Tooling

<a id="i-27"></a>
### I-27 — Long full-suite runs can hard-abort inside XLA or lose a worker process

**Severity:** Medium · **Status:** OPEN · located, not root-caused · **Found:** 2026-09-17

**What is wrong.** A long `pytest tests/` run has died with `Fatal Python error: Aborted` and
no summary line, the crashing thread inside `jax/_src/compiler.py`
(`backend_compile_and_load`) while `ProcessPoolExecutor` threads were alive (the worker pool
of the time; it was removed on 2026-10-04, so no test starts one now). Intermittent: the
same command later passed in full, and no abort occurred in the recorded runs since
2026-09-24.

**Hypothesis, unproven.** The worker pool lives for the interpreter; before 2026-10-01
`tests/test_api.py` created pools and never shut them down, so later in-process compiles ran
with worker children attached. Pairing modules does not reproduce it. Since 2026-10-02 the
suite does share JAX's on-disk compilation cache between its processes (xdist workers and
the engine workers they start, `tests/conftest.py`); JAX writes an entry without a lock and recompiles
when one cannot be read, so a torn entry costs a compile, not a crash, but an abort after
that date should rule the cache in or out first (rerun with `JAX_COMPILATION_CACHE_DIR`
pointing at an empty directory).

**Second hypothesis: memory.** On 2026-10-04 a Linux full run (Docker, 4 CPUs, `-n 4`, 25.2 GB)
lost one xdist process with no traceback ("node down: Not properly terminated") inside an FP8
storage test that passes alone at a 2.9 GB peak; the green rerun peaked at 24.2 GB (cgroup
peak, page cache included). GitHub's 4-vCPU runner, where CI's `full` job runs with `-n auto`,
has 16 GB.

**2026-10-04, three Linux full runs, none green** (see the verification status): one
beside another container in the same Docker VM, 6 failed; one alone on a loaded host, the
four-device sharding test (a fifth JAX process with four XLA host devices) timed out after
30 minutes at a cgroup peak of 24.4 GB of 25.2; one on a quiet host where the Docker VM went
down at 98%. Memory is the leading suspect: the suite now runs at the VM's limit.

**2026-10-05, five Linux full runs under WSL2, all with summary lines** (a 23 GB
VM, `-n 4`): four on the GPU (the first two with failures in the code since fixed, none a lost
process or an abort; the last two green, 20–22 min) and one on CPU, green in 9m24s. WSL2 is the steadier Linux host here;
the Docker attempts above remain unexplained.

**CI, 2026-10-05: memory, confirmed and fixed for the fast tier.** Every CI run since `-n auto`
(`f51227c`, run #14) died after 7–9 minutes with "The operation was canceled": the 4-vCPU,
16 GB runner ran out of memory and lost its runner agent. Reproduced in Docker with the
runner's limits (4 CPUs, 16 GB, no swap, no compile cache): OOM-killed, three of four xdist
processes lost. Two causes, two test-only fixes, the engine unchanged:
`tests/test_ore_bermudan_oracle.py::test_engine_is_grid_converged` alone peaked at 11.5 GB on
its 384-per-std, 10-std control grid, now 192 and 8 (within 1.6e-6 of it, 3.3 GB; the
rollback's memory is fixed properly by its matrix form, [I-83](#i-83), [details](details/precision.md#83-emulation-and-native-speed));
and each test process kept every XLA program it compiled, now dropped after each module
(`jax.clear_caches()` in `tests/conftest.py`). The same container then passed the fast tier,
2,461 passed, 2 skipped, at a 13.1 GB peak (page cache included), in 5m22s. The full suite
on that runner is not yet shown.

**2026-10-08, the Linux run of the 2026-10-07 code under WSL2 (24 GB VM, CPU, `-n 4`)** reached 78% with no
failure when the kernel's global OOM killer took a test process (5.9 GB resident); the suite
gained the ORE simulation tests (an OREApp run and the engine on the same portfolio per model)
since the green runs of 2026-10-05. Afterwards the distro instance stopped about 90 s after each
start, idle or not, until WSL was restarted. Memory remains the leading suspect.

**2026-10-11, memory confirmed for WSL2 at `-n 8` and `-n 4`.** The Linux runs of the kernel
rewrite (24 GB VM, CPU) took the whole VM down twice, at about 11% and with no summary line. A
log of each xdist process's resident memory and running test every 3 s showed the VM's use
rising past 17 GB at a rate of 1.4 GB per 5 s: four processes at 2.9 to 5.1 GB each plus the
demos' subprocesses. A process holds more than its test needs alone (the American's finer
grids at 5.1 GB, against a 1.3 GB peak alone): it keeps memory from the tests it ran before,
despite `jax.clear_caches()` after each module. The heaviest tests at that point need the same peak alone before and
after the rewrite (`/usr/bin/time`, HEAD against the working copy: Hull-White's pipeline 2,342
and 2,323 MB, the American's finer grids 1,273 and 1,272, the calibration portfolio 1,590 and
1,591, the component demos 2,803 and 2,808), so the rewrite added nothing; four processes no
longer fit in 24 GB. At `-n 2` the run finished in 18m11s, the VM at a 19.4 GB peak, two
processes at 7.8 GB (`tests/test_ore_xva_parity.py`'s simulation tests) and 8.8 GB
(`tests/test_profiling_and_jit.py`'s Greeks compile counts). A Linux full run here now uses
`-n 2`, and `-n auto` on a 16 GB CI runner would not fit.

**Current handling.** Since 2026-10-04 no pool, and no executor thread, exists: HTTP tests
start one engine worker, a separate process with no thread in the test process, and each
module stops it when it ends (`tests/conftest.py`); a worker also exits once its parent is
gone. If an abort recurs, the pool hypothesis is ruled out.

**To close.** Measure each xdist process's peak resident memory over a full run (Linux,
`-n 4`) and the heaviest tests; if memory explains the deaths, lower `-n` for the `full` CI job
or slim the heaviest fixtures. Then repeated clean full runs against a known-bad baseline,
with `JAX_COMPILATION_CACHE_DIR` on an empty directory to rule the cache in or out; one green
run proves nothing. Once the full tier fits CI's runner, run it on a schedule as well as by
hand: it holds the slow ORE simulation tests (L4) and the worker tests, which no automatic run
covers today (project audit, AUD-47).

<a id="i-66"></a><a id="q-2"></a>
### I-66 — No linter or type checker

**Severity:** Low · **Status:** OPEN · **Found:** 2026-09-24, audit Q-2 (pins and CI done)

**What is wrong.** No ruff/flake8, mypy/pyright or pre-commit, so nothing keeps the code
clean. `pyflakes engine` reports nothing since the layout change of 2026-10-08 (I-92), which
removed its findings with the modules they were in (re-export shims, unused imports, a string
forward reference, an f-string without placeholders); `pyflakes tests demos` still reports 19
unused imports and variables. `requirements.txt` does not mention the `profiling` extra.
`pyproject.toml` declares `pandas`, which nothing in `engine/`, `tests/`, `demos/` or
`compliance/` imports (project audit, AUD-49).

**To close.** As decided (A-20), in two steps. First, now: ruff in
`pyproject.toml` and CI with pyflakes' rules only, each finding fixed or marked (a re-export
listed in `__all__`), so `engine/` stays clean; `pandas` dropped from the dependencies. Then, among the fixes of what exists: a type
checker on `engine/`.

<a id="i-67"></a><a id="q-3"></a>
### I-67 — Test modules import each other and repeat fixtures

**Severity:** Low · **Status:** OPEN · **Found:** 2026-09-24, audit Q-3

**What is wrong.** Most cross-module imports moved to `tests/support/` on 2026-10-01
(`portfolio`, `lgm_engine`, `greeks`); test modules still import `bermudan_references` and
`date_helpers` from the tests directory itself.
`FIXTURES = ...traderx-eod` is defined in 15 files and `MARKET = {...flat-3pct-v1}` in 7.
Many tests reach into private functions, which freezes internal structure and makes
refactors (the configurable engine of 2026-10-04) break tests without behaviour changing.

**To close.** Shared helpers and constants in `tests/support/`; replace private-symbol tests
with public-entry tests where those refactors touched them. Collection must stay
identical except where a test is deliberately rewritten.

<a id="i-81"></a>
### I-81 — A cold job compiles about 180 one-operation programs; market risk vmaps a closure

**Severity:** Low · **Status:** OPEN · **Category:** Performance · **Found:** 2026-10-06,
making the AD Greeks one program

**What is wrong.** Code outside any jit runs JAX operations one by one, and each operation
compiles once per shape: on a cold demo job (`demos/demo_profile_small.py`, CPU) 180 of the
217 programs are a single `multiply`, `where`, `stack` and the like, about 2.5 s of
compilation plus their tracing. Most come from the scenario market's construction
(`lgm_numeraire`, `implied_log_discounts` in `engine/market_simulation/scenario_market.py`, about 60),
the CAM calibration (`cam.py`'s `zeta`, `calibrate_currency`), the options' known path
fixings (`engine/pricing/bermudan.py::_known_rates`) and the sensitivity market. Separately,
market risk vmaps a fresh closure over the jitted pricers per run (`revalue_trade`), the
pattern whose derivative programs were found (2026-10-06) to be kept by JAX only in internal caches of 2,048
entries (profiling §3.8): a market-risk job large enough to overflow them would compile its
batched programs again on the next run.

**Reach.** First-call time only: each such program is compiled once per process and shape and
then reused (a repeated demo job compiles nothing). No number is wrong.

**Current handling.** None.

**To close.** Jit the scenario market's construction and the other eager sites as
module-level functions with their data as arguments, and give market risk's batch a
module-level jitted function of `TradePriceFunction.pricer` and `.terms`, as the AD Greeks
(`engine.risk.greeks.ad._curve_derivatives`). Jitting can move float64 at rounding level (XLA
fuses what ran op by op), so it shows the golden snapshot's change per array. Measured by the
compile probe of profiling §3.8 on the demo's job.

<a id="i-88"></a>
### I-88 — The API completeness test checks field names, not that a value reaches the engine

**Severity:** Low · **Status:** OPEN · **Category:** Tooling · **Found:** 2026-10-08, review
of the API of 2026-10-07

**What is wrong.** `tests/test_api_completeness.py` fails on a configuration setting without
an API field of the same name, but it does not check what a schema's `.to_dataclass()` does
with the field. A field that is accepted, then dropped or sent to the wrong setting, passes.

**Reach.** Any setting not exercised by a route test that compares the HTTP result with a
direct call (`tests/test_api_portfolio.py`, `tests/test_api_market_risk.py`,
`tests/test_api.py`). Those cover the defaults and the commonly used settings, not every
field. No such field is known to be dropped today.

**Current handling.** The route tests above.

**To close.** For every field the walk finds, build a request with that field set
to a non-default value, convert it, and assert the value arrives in the engine's
configuration. Red first: a schema that drops a field must fail the test.

<a id="i-110"></a>
### I-110 — The documentation contradicts the code in places; no single parity table

**Severity:** Low · **Status:** OPEN · **Category:** Tooling · **Found:** 2026-10-10, the
project audit (AUD-15, AUD-46, AUD-47); checked 2026-10-11

**What is wrong.**

- [docs/getting-started/overview.md](../getting-started/overview.md) says engines like ORE run
  single-threaded on CPUs, where the README describes ORE's threaded and GPU paths.
- [docs/concepts/architecture.md](../concepts/architecture.md) calls the codebase three
  independent modules, and `demo_precision.py` a demo at three precisions; it runs ten.
- The README says the full pipeline compiles through XLA; ORE schedules and calendars, Sobol
  draws and other host work do not.
- `pyproject.toml` says ORE-parity tests are never marked slow; two in
  `tests/test_ore_xva_parity.py` are.
- ORE parity is stated per component ([docs/reference/ore-parity.md](../reference/ore-parity.md)),
  but no one table says, per product and analytic (t=0 value, path values, exposure, Greeks,
  VaR/ES), what is shown equal to ORE, to what tolerance and by which test, so "matches ORE"
  reads broader than it is (Bermudan paths, [I-49](#i-49); Greeks, [I-51](#i-51)).

**Reach.** Readers; no number.

**To close.** Each statement corrected; one table of products by analytics in
docs/reference/ore-parity.md, linked from the README and the overview, which a change that
moves a cell updates.

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
| <a id="i-09"></a>I-09 | The whole scenario cube was serialized into the JSON response (a request now asks for it inline, as a chunked, hashed artifact reference, or not at all: `cube_output`, decision A-17) | `tests/test_api_portfolio.py::test_the_cube_by_reference_is_the_inline_cube_and_none_leaves_it_out`, `tests/test_api_artifacts.py` |
| <a id="i-10"></a>I-10 (results) | Per-trade results were keyed by position beside the echoed ids (now one row per trade with its `trade_id`, the arrays' trade axes in that order) | `tests/test_api_portfolio.py::test_result_matches_direct_price_portfolio_call`, `tests/test_api_market_risk.py::test_the_result_equals_a_direct_run` |
| <a id="i-10-configs"></a>I-10 (configs) | Trade configs had no identity (now a required `trade_id`, echoed as `PortfolioResult.trade_ids`) | `tests/test_trade_configs.py::TestEveryTradeNamesItselfAndItsDate`, `tests/test_pipeline.py::test_a_repeated_trade_id_is_refused` |
| <a id="i-12"></a>I-12 | `/version` named the API process's backend, the only device a result could be attributed to (every result now carries a `PrecisionReport` built where the job ran: its devices, backend and the formats read from its arrays) | `tests/test_api_portfolio.py::test_a_result_carries_the_workers_precision_report`, `tests/test_precision_report.py::TestReport` |
| <a id="i-13"></a>I-13 | A negative curve index priced against the wrong curve (trades now name a currency and index; a missing curve is refused before pricing, naming the trade) | `tests/test_portfolio_gap_fixes.py::TestCurveIndexValidatedBeforeAllPricing` |
| <a id="i-14"></a>I-14 | `generate_paths(precision=32)` leaked `jax_enable_x64=False` (`generate_paths` removed on 2026-10-01; nothing toggles the flag) | `tests/test_portfolio_entrypoint.py::TestPricePortfolioConcurrency` |
| <a id="i-15"></a>I-15 | The worker-pool concurrency test could not observe concurrency (and, until 2026-10-01, failed after the pricing tests by reusing one worker) | The pool and its test went on 2026-10-04; jobs now run one at a time (`tests/test_engine_worker.py::TestEngineWorkerPricing`) |
| <a id="i-17"></a>I-17 | A malformed note date failed the whole bundle | `tests/test_traderx_note.py::TestRefusalsAreNotePricingErrors` |
| <a id="i-19"></a>I-19 | The accrual tolerance rounded its own bound | `tests/test_traderx_note.py::TestToleranceIsDerivedNotConstant` |
| <a id="i-20"></a>I-20 | Impossible calendar dates aborted the whole bundle | `tests/test_traderx_note.py::TestImpossibleCalendarDates` |
| <a id="i-21"></a>I-21 | AD Greeks recompiled about 30 XLA programs per repeated call (the pricers are jitted with the trade as an argument; the Greeks no longer jit a fresh closure) | `tests/test_profiling_and_jit.py::TestCompileCounts::test_repeated_greeks_call_compiles_nothing`, `::test_a_different_trade_gets_its_own_greeks_from_warm_programs` |
| <a id="i-22"></a>I-22 | Each LGM calibration recompiled its bisection (now one program per helper shape, the basket an argument) | `tests/test_profiling_and_jit.py::TestCompileCounts::test_calibrations_of_one_basket_shape_share_one_program` |
| <a id="i-25"></a>I-25 | A scalar Greek crashed the HTTP result serializer | `tests/test_api_bond_schemas.py::TestBondGreeksSerializeOverHttp` |
| <a id="i-24"></a>I-24 | Bonds had no scenario NPV on the Hull-White model (refused; a broadcast t=0 value gave VaR 0, ES NaN) | `tests/test_hull_white_model.py::TestBondsOnEveryPath`, `tests/test_portfolio_bond_wire_through.py::TestBondsArePricedOnEveryPath` |
| <a id="i-26"></a>I-26 | Greeks for a bond maturing tomorrow crashed on the Theta reprice | `tests/test_portfolio_bond_wire_through.py::TestBondGreeksReachThePortfolioPath` |
| <a id="i-28"></a>I-28 | The `var_es` module demo crashed on a moved date | `tests/test_demos.py::TestComponentDemosRun` |
| <a id="i-29"></a>I-29 | A rounded exercise time silently dropped a coupon (exercise now given as dates) | `tests/test_ore_bermudan_oracle.py::TestExerciseDatesAreExact` |
| <a id="i-30"></a>I-30 | The `A(t,T)` variance term was nearly uncovered at t=0 (that formula went with the Hull-White pipeline; the model's bonds are QuantLib's on every state) | `tests/test_cam.py::test_hull_white_path_curves_equal_quantlibs_hull_white` |
| <a id="i-31"></a>I-31 | Bermudan/American floating coupons projected over the wrong period | `tests/test_ore_lgm_parity.py` |
| <a id="i-33"></a>I-33 | On Linux, worker-pool jobs hung once the parent had run JAX (fork) | The engine worker is a fresh interpreter (`subprocess`, never fork) since 2026-10-04: `tests/test_engine_worker.py::TestWorkerProcesses`, the full suite on Linux |
| <a id="i-34"></a>I-34 | The ORE oracle tilted a curve's first segment (ORE's rebuild reads the t=0 rate at t = 1e-4; the oracle now hands it the as-of quote that maps onto the engine's, 2.3e-6 → 1.5e-12 on a sloped first segment) | `tests/test_ore_lgm_parity.py::test_a_curve_sloped_in_its_first_segment_is_the_engines` |
| <a id="i-35"></a>I-35 | An American already in its window was exercisable on the evaluation date | `tests/test_trade_dates.py::test_seasoned_bermudan_and_american_equal_ore` |
| <a id="i-36"></a>I-36 | A non-ACT/365 floating leg projected the wrong forward | `tests/test_trade_dates.py::test_any_leg_day_count_equals_ore` |
| <a id="i-37"></a>I-37 | A European silently ignored `floating_spread` (refused by the Jamshidian engine, priced by Bachelier) | `tests/test_jamshidian.py::TestConfiguration::test_a_spread_is_refused_naming_the_trade`, `tests/test_pricing.py::test_european_today_equals_ores_default_engine` |
| <a id="i-38"></a>I-38 | Theta rolled a business day; ORE rolls a calendar day | `tests/test_trade_dates.py::test_swap_theta_equals_ore`, `tests/test_sensitivities.py::test_theta_rolls_one_calendar_day_from_a_friday` |
| <a id="i-39"></a>I-39 | Bond Theta had no add-back for a coupon paid in the period | `tests/test_sensitivities.py::test_theta_adds_back_a_bond_coupon_paid_on_the_theta_date` |
| <a id="i-40"></a>I-40 | The note's `rateSensitivity` ignored `fractionDecimals` | `tests/test_traderx_note.py::TestSensitivityUsesTheDeclaredFractionDecimals` |
| <a id="i-41"></a>I-41 | A European at zero mean reversion priced at intrinsic value (now refused) | `tests/test_jamshidian.py::TestConfiguration::test_non_positive_reversion_is_refused` |
| <a id="i-42"></a><a id="m-1"></a>I-42 | Hull-White simulated curves were not arbitrage-free against the input curve (deflated 10y bond +6.8% at 2y on a 3→5% curve) | `tests/test_hull_white_model.py::TestCurveFittedDrift`, `tests/test_cam.py` |
| <a id="i-43"></a><a id="m-3"></a>I-43 | Hull-White options were worth 0 after expiry instead of carrying the swap | `tests/test_pricing.py::test_an_exercised_physical_option_becomes_its_swap_and_a_cash_one_leaves` (both models) |
| <a id="i-44"></a><a id="a-2"></a>I-44 | Hull-White scenario pricing mixed Hull-White and LGM | `tests/test_pricing.py::test_bermudan_on_every_path_equals_ore_recalibrated_on_the_path_curves` (both models), `tests/test_hull_white_model.py::TestOneModelOnePipeline` |
| <a id="i-45"></a>I-45 | The Hull-White numeraire was a left-point bank account (E[1/N] +1.8% at 3y) | `tests/test_hull_white_model.py::TestExactNumeraire` |
| <a id="i-46"></a>I-46 | Hull-White Europeans were priced off the model volatility, without Vega | `tests/test_hull_white_model.py::TestEuropeansOnTheMarketVolatility` |
| <a id="i-47"></a>I-47 | Hull-White options calibrated to one caller basket, not their own | `tests/test_pricing.py::test_option_today_equals_ores_calibrated_grid_engine`, `tests/test_hull_white_model.py::TestOneModelOnePipeline` |
| <a id="i-48"></a>I-48 | Zero curves extrapolated a flat zero rate; ORE a flat forward | `tests/test_treasury_instrument.py::TestCurveInterpolation`, `tests/test_curves.py` |
| <a id="i-50"></a>I-50 | Nothing compared the assembled simulation, cube or exposure with an ORE simulation (now an OREApp XVA oracle: its scenario dump rebuilds the engine's simulation market, swaps, Europeans and bonds equal ORE's cube path by path, the exposure definitions on ORE's cube are ORE's, and L4 holds; Bermudans/Americans wait on I-49) | `tests/test_ore_xva_parity.py` |
| <a id="i-52"></a>I-52 | Cash settlement was priced as physical | `tests/test_pricing.py::test_a_cash_settled_european_uses_the_par_yield_annuity` |
| <a id="i-56"></a>I-56 | Market risk and the CAM calibration had no route, and two routes were named like versions (now `POST /portfolio/market-risk` and `POST /calibration/cam`, one poll route for every job, `/v2/portfolio/price` and `schema_version` removed, and a test that fails on any setting without an API field; 2026-10-07, decision A-2) | `tests/test_api_market_risk.py`, `tests/test_api.py::TestCamCalibrationEndpoint`, `tests/test_api_completeness.py` |
| <a id="i-57"></a>I-57 | TraderX: a cached result was served before the submission id's binding was checked, so an id reused for a cached workload came back with that workload's result (TraderX case A-02; now one atomic `AttemptStore.submit` checks the binding, then the cache, binds the id to whatever it returns, and records the binding durably first; 2026-10-10) | `tests/test_traderx_routes.py::TestTheSubmissionIdBindsBeforeAnyCacheReturn`, `::TestSubmissionIdCannotCrossWorkloads::test_store_level_conflict_raises_even_when_the_target_is_cached`, `tests/test_traderx_publication.py::TestSubmissionBindingsAreDurable` |
| <a id="i-62"></a><a id="p-2"></a>I-62 | Hull-White Bermudan/American scenario pricing ran on the host (that pricer is removed; options are priced by the vectorized per-path engine) | `tests/test_pricing.py::test_bermudan_on_every_path_equals_ore_recalibrated_on_the_path_curves` |
| <a id="i-63"></a><a id="a-3"></a>I-63 | Hull-White trade configs carried copies of model parameters | `tests/test_trade_configs.py::TestEveryTradeNamesItselfAndItsDate::test_a_model_or_curve_on_the_trade_is_refused` |
| <a id="i-64"></a><a id="a-4"></a>I-64 | A trade's evaluation date defaulted to ORE's thread-local global | `tests/test_trade_configs.py::TestEveryTradeNamesItselfAndItsDate::test_without_an_evaluation_date_it_is_refused` |
| <a id="i-65"></a><a id="a-6"></a>I-65 | Demo data, the modules' `__main__` demos and the ORE test oracle shipped inside `engine/` (now `demos/`, `tests/support/`) | `tests/test_import_layering.py::test_engine_ships_no_demo_or_test_code` |
| <a id="i-68"></a>I-68 | The Hull-White model was chosen by the market's type, not by the configuration (now `HullWhiteConfig` in `CamConfig.ir`) | `tests/test_run_config.py::TestEveryModelRunsWithEveryEngineAndMethod` |
| <a id="i-69"></a>I-69 | The market path silently ignored `precision.pricing`/`risk`/`calibration`, `calibration_targets`, and a `base_currency` contradicting the simulation (now refused by name) | `tests/test_run_config.py::TestWhatThePipelineDoesNotImplementIsRefused`, `::TestConfigurationValues`, `tests/test_api_portfolio.py::test_an_unpriceable_request_is_a_400_and_no_job` |
| <a id="i-70"></a>I-70 | Bump Theta of a bond maturing the next day raised `BondPricingError` (now redemption − NPV, as ORE) | `tests/test_portfolio_bond_wire_through.py::TestBondGreeksReachThePortfolioPath::test_a_bond_maturing_tomorrow_does_not_crash_the_greeks` |
| <a id="i-71"></a>I-71 | A float32-tier worker turned x64 off, so its pricing and exposure ran in float32 where an in-process run used float64 | `tests/test_engine_worker.py::TestEngineWorkerPricing::test_jobs_queued_together_give_the_bits_of_jobs_run_one_after_another` (float64 and float32 jobs equal the direct call) |
| <a id="i-72"></a>I-72 | HTTP jobs ran in a process pool that pickled each request with its ORE dates frozen as text, compiled every job shape once per worker, broke for every later job when one worker died (`BrokenProcessPool`), and would have needed chip pinning on TPU (now one engine worker per host behind a durable queue) | `tests/test_engine_worker.py` (`test_a_second_identical_job_compiles_nothing`, `test_a_failing_job_fails_only_its_own_row`, `test_a_worker_killed_mid_job_leaves_it_interrupted`, `test_the_worker_prices_the_request_the_route_validated`) |
| <a id="i-75"></a>I-75 | Storage below 32 bits kept few bits of a concentrated column's spread: FP8 put its bits on the level of a bond's cube (1e6 ± 1e4) or a long log discount factor, and every path of a block rounded alike, a bias more paths did not remove (now a concentrated block, one sign and the farthest entry at most three times the nearest, is stored as its deviations from its midpoint, `Precision.offset="block"`, the default; the storage harness measures every format, 2026-10-09) | `tests/test_precision_harness.py::test_a_bonds_fp8_cube_column_is_unbiased_with_block_offsets`, `tests/test_precision.py::TestBlockOffsets` |
| <a id="i-74"></a>I-74 | A calibration basket read ORE's global evaluation date, so a later date (another caller's, or the wall clock past a helper's fixing) failed it with a missing fixing | `tests/test_ore_lgm_calibration.py::test_the_basket_does_not_depend_on_ores_global_evaluation_date` |
| <a id="i-79"></a>I-79 | Never run on a GPU (now the `gpu` extra under Linux or WSL2, verified on an RTX 5060: no preallocation, the API's JAX on the CPU, float32 products at float32 rather than TensorFloat-32, deterministic kernels; the demo and the full suite run on the GPU) | `tests/test_accelerator_defaults.py`, `tests/test_environment.py::test_a_gpu_plugin_is_jaxlibs_version`, the full suite on the GPU |
| <a id="i-80"></a>I-80 | Importing `engine` changed JAX and XLA settings for the whole process (now x64 only: each matrix product states its precision through `engine.precision.matmul`; deterministic GPU kernels set by the engine worker and the tests; GPU preallocation off in the demos and the tests; 2026-10-06, decision A-22) | `tests/test_accelerator_defaults.py` (`TestImportingTheEngine`; `TestEveryMatrixProductStatesItsPrecision::test_in_every_pipeline`, red on the bare products) |
| <a id="i-82"></a>I-82 | An American priced on a path date after the last reference-grid date in its window and before its last exercise (no calibration helper left) raised `ValueError: Need at least one array to stack` (now the engine's volatility, as a calibration today with no helper; found on 2026-10-07) | `tests/test_root_solvers.py::test_a_date_with_an_exercise_left_and_no_helper_keeps_the_engine_volatility` |
| <a id="i-84"></a>I-84 | The simulation market held each curve's tenor points at the tenors' times from the simulation date; ORE's `ScenarioSimMarket` holds them at the times from the as-of date (swaps and Europeans up to 0.45% of their largest path value off ORE's cube; found by the ORE simulation oracle's L3, 2026-10-07) | `tests/test_ore_xva_parity.py::test_every_trades_cube_equals_ores_on_ores_paths`, `tests/test_pricing.py` (path curves at ORE's times) |
| <a id="i-85"></a>I-85 | Exposure profiles started from each trade's value on today's market; ORE's start from its value on the simulation market of the as-of date, the cube's `T0` (up to 4.2% on the shared portfolio) | `tests/test_ore_xva_parity.py::test_the_pipelines_profiles_start_from_ores_t0` |
| <a id="i-86"></a>I-86 | A cash-settled option's exposure stopped at its underlying's maturity, so its time-weighted EPE kept accruing after it settled; ORE's `Trade::maturity()` is its last exercise date (2026-10-07) | `tests/test_ore_xva_parity.py::test_the_profiles_of_ores_cube_are_ores_reports` |
| <a id="i-92"></a>I-92 | `engine/`'s packages were named for layers and two modules sat at its root (now subpackages only, each named for what it holds and layered: `market_data`, `models`, `instruments`, `traderx`, `calibration`, `market_simulation`, `pricing`, `risk/{greeks,market,counterparty}`, `run`, `api`; [details/package-layout.md](details/package-layout.md)) | `tests/test_import_layering.py::test_no_module_but_init_at_the_root`, `::test_packages_import_only_the_layers_below`, `::test_risk_kinds_import_only_the_kinds_below` |
| <a id="i-96"></a>I-96 | Compute below float32 biased every figure: each product's path-independent operand was rounded once for every path (now carried in two terms of the format, the single pass a format of its own, `<name>_1x`; 2026-10-11, decision A-23) | `tests/test_compute_precision.py::TestPipeline::test_no_rounding_of_a_shared_operand_biases_the_paths` (red on the code before: 14 to 24 standard errors), `TestProducts::test_the_shared_operands_rounding_biases_a_single_pass_not_two_terms` |
| <a id="i-99"></a>I-99 | On a GPU a block-offset storage took a block's deviations from its unrounded midpoint while keeping the float32 one, so a stored entry missed its value by float32's rounding of it: XLA removed the float64 -> float32 -> float64 round trip of the offset's cast (found and fixed 2026-10-11, the offset now rounded on float32's grid) | `tests/test_precision.py::TestScaledStorage::test_any_scenario_count_and_axis_short_last_block` (red on an RTX 5060 at the code before, its one-entry blocks 5e-8 off) |
| <a id="i-100"></a>I-100 | Market risk added the P&Ls of trades in different currencies unconverted (now every amount is in the request's `base_currency`, USD by default as a portfolio run's without a simulation, each trade converted at the as-of FX spot in float64, a missing spot refused before any work naming the trade and the pair, the result naming its currency and warning that FX is not shocked; the portfolio run's preflight refuses a missing spot, and a trade in a currency its simulation does not model, before any work; 2026-10-11) | `tests/test_market_risk.py::TestReportingCurrency` (red on the code before: the figures did not move with the EURUSD spot; VaR_99 601,048 where the converted P&Ls give 633,967 at 1.10), `tests/test_run_config.py::TestWhatThePipelineDoesNotImplementIsRefused::test_a_trade_with_no_spot_into_the_reporting_currency_is_refused_naming_it`, `::test_a_trade_in_a_currency_the_simulation_does_not_model_is_refused` |
| <a id="i-101"></a>I-101 | A negative option notional booked a short European but a long Bermudan or American on the reversed swap (the Bermudan payer at -1e6 was +15,017.85 where ORE's short is -35,712.95), and a zero one priced a European to NaN (now every option has ORE's `LongShort`, `long_short` on the configs and the API, `Long` by default; a short is minus the long one in every engine, today, on the paths with the holder's exercise as ORE's `OptionWrapper`, in the AD and bump Greeks and in market risk; a zero or negative option notional refused naming the trade; 2026-10-11) | `tests/test_option_position.py`, `tests/test_shared_portfolio.py::test_a_short_option_today_is_ores_short_option` (against ORE's trade builder with `LongShort` Short), `tests/test_ore_xva_parity.py::test_a_short_option_on_ores_paths_is_ores_short_option` (each red on the code before) |
| <a id="i-102"></a>I-102 | VaR took index floor(N(1−p)), one above QuantLib's ceil(N(1−p)) − 1 where N(1−p) is a whole number (VaR 15 where ORE's is 16 on 16 scenarios at 0.9375), and quantile keys were rounded to six decimals, so close levels shared one (now QuantLib's rank, `var_rank`; each key the level's percentage exactly; a run's levels in (0, 1), each once, documented as broader than ORE's [0.9, 1); 2026-10-11) | `tests/test_var_es.py::TestExactRankBoundaryAgainstORE`, `::TestQuantileLabels`, `tests/test_market_risk.py::TestRunValidation::test_close_quantiles_keep_their_own_figures`, `tests/test_exposure.py::TestDefinitions::test_close_quantiles_keep_their_own_keys` (each red on the code before) |
| <a id="i-55"></a><a id="a-1"></a>I-55 | No result said whether its precision combination had been shown adequate for its figures, so a float32 exposure profile looked exactly like a validated one (now the evidence table, `compliance/precision-evidence.md`, generated from the precision harness's measurements against decision A-11's standard, `compliance/precision-acceptance.json`: the statistical rule, an absolute rule for figures float64 holds fixed, MAR32.42's P&L attribution test for market risk, each row's path ceiling, code revision, market, seeds and device; every portfolio and market-risk result warns of each figure without a passing row; the paired standard errors' limits are [I-111](#i-111); 2026-10-11) | `tests/test_precision_evidence.py` (the rules, the lookup, the files of record against the measurements, a reduced run's warnings end to end; red on the code before, where an FP8 run's `warnings` were `[]`, and the ceiling test red under a ceiling that skips a failing path count), `tests/test_precision_harness.py::TestMeasure::test_the_figures_measured_are_those_the_engine_warns_of` |
| <a id="m-4"></a>Audit M-4 | Trades were defined relative to the evaluation date (now absolute dates) | `tests/test_trade_dates.py` |
| <a id="m-5"></a>Audit M-5 | Theta re-rolled the trade instead of ageing it | `tests/test_trade_dates.py` |
| <a id="r-1"></a>Audit R-1 | Cube quantiles were reported as VaR/ES (now exposure profiles; market-risk VaR/ES by t=0 revaluation) | `tests/test_exposure.py`, `tests/test_market_risk.py`, `tests/test_market_risk_ore_parity.py` |
| <a id="p-3"></a>Audit P-3 | The precision study bypassed the engine's own FP32 path | `demos/demo_precision.py` |
| <a id="a-5"></a>Audit A-5 | Instruments imported validators from the layer above | `tests/test_import_layering.py` |
| <a id="a-7"></a>Audit A-7 | The bond pricer was a separate plain-Python implementation | `tests/test_market_risk.py::TestRevaluation` |