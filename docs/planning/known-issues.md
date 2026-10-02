# Known Issues

Open defects and important shortcomings: what each does to a number or a caller, and what
closing it takes. The order of work is in [roadmap.md](roadmap.md); the rules for this file
(statuses, severities, how to add and close an entry) are in [README.md](README.md). Fixed
issues keep one line in the [closed ledger](#closed).

Read this before trusting an exposure profile or a result served by the EOD routes:
[I-57](#i-57) can return another submission's result, and the assembled simulation and
exposure are not yet compared with an ORE run ([I-50](#i-50)).

## Verification status

Last full run, 2026-10-02, on the code of roadmap 1.6 (storage below 32 bits), 2,454
collected (2,340 before it; +110 in `tests/test_precision.py`, the scaled formats' storage,
policy and pipeline tests; +4 in `tests/test_api.py`, the rounding fields on the wire),
summary line printed:

- **Windows**, `-n 8`: **2,453 passed, 1 skipped, 0 failed**, 7m44s (the skip is the
  parametrized case of a storage wider than its compute, refused by design).
- **Fast tier under strict dtype promotion** (`JAX_NUMPY_DTYPE_PROMOTION=strict`, the CI
  job, `-n 8`): 2,353 passed, 1 skipped, 2m13s.
- **Linux** was not rerun: 1.6 changes no process, path or platform default. The last Linux
  run (the jit change, same day): 2,339 passed, 1 skipped (`reference/traderX` not in the
  container), 18m06s.
- `-n auto` on the 24-thread, 32 GB Windows machine started 24 processes and failed 20 tests
  with `MemoryError` and its after-effects: memory, not the code; see the user guide.

Bit for bit (the exit criterion of 1.6): the 232 snapshot arrays from a worktree of `f51227c`
(the jit change) against the 1.6 tree, all identical in value, dtype and shape, float32 runs
included ([details/precision.md §13.1](details/precision.md#131-bit-for-bit-and-ore-parity)).
float64 market risk on four trades, old and new trees interleaved, 18 warm runs each: median
1.85 s before and 1.92 s after, fastest 1.43 and 1.41 s, within the spread. Red first: on the
code before 1.6 every new storage test fails at `store`, which refused the scaled formats.
The fast tier (`-m "not slow"`) alone is not a full verification and is never recorded here. Rules:
[README.md](README.md#verification-rules).

## Summary

| ID | Issue | Sev. | Status | Category | Stage |
|---|---|---|---|---|---|
| [I-04](#i-04) | Seasoned TraderX swaps: the export has no past fixings | High | OPEN | Scope | External |
| [I-05](#i-05) | No faithful USD-SOFR / ACT-360 swap construction | High | OPEN | Scope | External |
| [I-07](#i-07) | No corporate bond, equity or listed-option pricer | Medium | OPEN | Scope | By demand |
| [I-08](#i-08) | Portfolio job store in memory; a running EOD attempt is lost on restart | Medium | PARTIAL | API | 1.8, 4.2 |
| [I-09](#i-09) | Whole scenario cube serialized into the JSON response | Medium | OPEN | API | 4.1 |
| [I-10](#i-10) | Per-trade results keyed by position beside the echoed ids | Low | PARTIAL | API | 4.1 |
| [I-12](#i-12) | `/version` reports the dispatcher's backend, not the worker's device | Low | OPEN | Correctness | 1.7 |
| [I-16](#i-16) | `rateSensitivity` is parallel-only | Medium | OPEN | Scope | External |
| [I-18](#i-18) | No equity spot or FX source; equity positions refused | Medium | OPEN | Scope | External |
| [I-23](#i-23) | `accrualBasis` strictness rests on an unconfirmed reading | Medium | ASSUMPTION | API | 4.3 |
| [I-27](#i-27) | Long full-suite runs can hard-abort inside XLA | Medium | OPEN | Tooling | 5.1 |
| [I-32](#i-32) | Bermudan/American engine only at `ShiftHorizon = 0`, not ORE's default 0.5 | Medium | OPEN | Correctness | 2.5 |
| [I-34](#i-34) | The ORE oracle's curve differs before the first pillar | Low | OPEN | Validation | 2.2 |
| [I-49](#i-49) | Per-path recalibration differs from ORE's in two details | Medium | OPEN | Correctness | 2.4 |
| [I-73](#i-73) | A per-path recalibration that misses its basket is not flagged | Low | OPEN | Correctness | 2.4 |
| [I-75](#i-75) | Storage below 32 bits keeps few bits of a concentrated array's spread | Low | OPEN | Correctness | 2.7 |
| [I-50](#i-50) | No path- or distribution-level parity test against an ORE simulation | Medium | OPEN | Validation | 2.2 |
| [I-51](#i-51) | Sensitivities not checked against ORE's sensitivity analytic | Medium | OPEN | Validation | 2.3 |
| [I-53](#i-53) | The pipeline is slow: per-path recalibration and bump Greeks of options | Medium | PARTIAL | Performance | 3.1 |
| [I-54](#i-54) | No swaption smile: options away from the money read the ATM vol | Medium | OPEN | Correctness | 2.6 |
| [I-55](#i-55) | Unproven precision combinations are not flagged | Medium | PARTIAL | Architecture | 2.7 |
| [I-72](#i-72) | Worker pools pickle ORE objects, compile per worker and would contend for TPU chips | Medium | OPEN | Architecture | 1.8 |
| [I-56](#i-56) | Market risk and the CAM calibration have no route; two routes named like versions | Medium | PARTIAL | API | 4.1 |
| [I-57](#i-57) | EOD: a cached result is served before the submission id is checked | High | OPEN | API | 2.1 |
| [I-58](#i-58) | EOD: two concurrent submissions of one workload both execute | Medium | OPEN | API | 2.1 |
| [I-59](#i-59) | EOD: `calculations` and `reportingCurrency` accepted, keyed, then ignored | Medium | OPEN | API | 2.1 |
| [I-60](#i-60) | EOD result schema has no stated policy on added fields | Low | ASSUMPTION | API | 4.3 |
| [I-61](#i-61) | Nothing runs on more than one device | Medium | OPEN | Performance | 3.2 |
| [I-66](#i-66) | No linter or type checker | Low | OPEN | Tooling | 5.2 |
| [I-67](#i-67) | Test modules import each other and repeat fixtures | Low | OPEN | Tooling | 5.3 |

**One pipeline.** Since roadmap 1.3 every run is `price_portfolio` on a `Market`
(`engine.portfolio.market_path`; HTTP `POST /portfolio/price`, also served as
`/v2/portfolio/price`): ORE's cross-asset model with a model per currency
(`CamConfig.ir`: the LGM by default, or Hull-White, decision A-1 in
[compliance/decisions.md](../../compliance/decisions.md)), and ORE's valuation of every trade on
every path. "Market path" in older entries and commits means this pipeline; "the Hull-White
model" before 1.3 meant the separate pipeline 1.3 removed.

---

## Correctness

<a id="i-12"></a>
### I-12 — `/version` reports the dispatcher's backend, not the worker's device

**Severity:** Low · **Status:** OPEN

**What is wrong.** `GET /version` reports `jax.default_backend()` of the HTTP process, which
runs no JAX work; pricing runs in `worker_pool` processes. On a multi-device host a precision
or hardware study reading this field would attribute results to the wrong device.

**To close.** Report the device and the realised per-stage dtypes from the worker, on each
result: the precision report of roadmap 1.7 ([details](details/precision.md#95-the-report)).

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

<a id="i-73"></a>
### I-73 — A per-path recalibration that misses its basket is not flagged

**Severity:** Low · **Status:** OPEN · **Category:** Correctness · **Found:** 2026-10-01, roadmap
1.4's dtype review

**What is wrong.** On every path and date a Bermudan/American is recalibrated by bisection on
σ ∈ [1e-6, 0.2] (`engine.calibration.ore_lgm.bootstrap_sigma`). A helper whose volatility is
not attainable in the bracket ends at its edge, and `bootstrap_sigma` flags it
(`hit_ceiling`), but `engine.valuation.bermudan._path_sigma` drops the flag, so that path is
priced on a model that does not reprice its basket and nothing says so. Today's calibration
refuses the same case (`calibrate_on`). ORE's `LgmBuilder` logs a structured warning when the
calibration error exceeds its tolerance and fails unless `continueOnCalibrationError`
(`OREData/ored/model/lgmbuilder.cpp`).

**Reach.** Bermudan/American cube values on paths where rates move far enough that a
helper's market volatility is out of reach (extreme paths, long horizons); exposure figures
through them. Not seen on the test markets. Since 1.4 the flag is also correct in float32 (its
`1 - 1e-9` tolerance rounded away there; `tests/test_precision.py::TestRecalibrationInFloat32`).

**Current handling.** None.

**To close.** With roadmap 2.4 (the per-path recalibration against ORE's): count the paths and
dates whose recalibration hit the bracket and carry a warning naming the trade and the counts
on the result, as ORE's structured warning; a test with an unattainable path volatility.

<a id="i-75"></a>
### I-75 — Storage below 32 bits keeps few bits of a concentrated array's spread

**Severity:** Low · **Status:** OPEN · **Category:** Correctness · **Found:** 2026-10-02,
roadmap 1.6's measurements ([details/precision.md §15.3](details/precision.md#153-storage-through-the-pipeline))

**What is wrong.** A scaled storage format (roadmap 1.6) keeps each value to the format's
precision relative to its block's largest value: FP8 keeps 3 (e4m3) or 2 (e5m2) mantissa bits.
Where the 32 paths of a block sit close together against their level, as a bond's cube column
(near 1e6, spread near 1e4) or a long log discount factor of the scenario curves do, those
bits go to the level and the spread between paths, the part that carries risk, is kept to
about one bit. Rounded to nearest, every path of such a block moves the same way: a bias that
more paths do not remove. Measured on the shared portfolio at 256 paths: an FP8 e5m2 cube is
biased by 3.7% of a trade's notional and an FP8 e4m3 market moves the EPE by 2.6%; float16 is
within 1e-5 of notional in every stage.

**Reach.** Runs that choose a storage below 32 bits, and only the stages they choose (no
default and no float32 run). Stochastic rounding (`Precision(rounding="stochastic")`) already
turns the bias into noise that shrinks with the path count (the FP8 cube's bias falls 3 to 4
times at 256 paths), but its variance stays that of a format with one bit of the spread, and
added variance biases a quantile: in market risk, where offsetting trades' stored P&Ls
cancel in the portfolio's, FP8 storage moves VaR and ES by 1.8 to 4.5 times the spread of
float64 across seeds, stochastic rounding more than nearest (float16: at most 0.044;
[details/precision.md §15.3](details/precision.md#153-storage-through-the-pipeline)).

**Current handling.** None: the runs are allowed (D-9) and carry no warning until
[I-55](#i-55)'s evidence table and warnings (roadmap 2.7). Documented in the user guide and the
portfolio entry point.

**To close.** Roadmap 2.7: store such classes relative to a level, so the format's bits go to
the spread: the cube relative to each trade's t=0 value and the curves relative to their
path-independent part (the difference form of [§8.2](details/precision.md#82-the-difference-form),
already computed in float64 by `build_scenario_market`), or a per-block offset beside the scale
in `Stored` (a decision on A-10's single rule). Measure each class with and without it, keep
what the evidence table needs, and test that an FP8 bond column's bias falls below its Monte
Carlo standard error.

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

<a id="i-53"></a>
### I-53 — The pipeline is slow: per-path recalibration and bump Greeks of options

**Severity:** Medium · **Status:** PARTIAL · **Found:** 2026-09-29 · re-measured 2026-10-02

**What is wrong.** Most of the cost was XLA compiling the same work again: per trade, per
path date, per bump and per call ([I-21](#i-21), [I-22](#i-22), profiling §3.7). With the
pricers jitted once per shape (2026-10-02), measured on CPU, shared test market, grid at
`n_per_std=16`, each job in a fresh process with no disk cache, old and new code
interleaved; "repeat" is the same call again in the process:

| Work | Before: first / repeat | Now: first / repeat |
|---|---|---|
| One Bermudan on 64 paths × 3 dates, recalibrated per path date | 41 s / 13 s | 15 s / 1.3 s |
| One American (3-year window), the same | 74 s / 50 s | 39 s / 20 s |
| Bump Greeks of the Bermudan / the American | 164 s / 583 s first, 168 s / 754 s repeat | 7.4 s / 22 s first, 1.3 s / 12 s repeat |
| AD Greeks of the Bermudan / the American | 38 s / 73 s first, 21 s / 57 s repeat | 21 s / 27 s first, 0.4 s / 5.9 s repeat |
| Market risk, 4 trades × 512 scenarios | 8.1 s / 6.9 s | 6.1 s / 2.8 s |

What remains: an American's repeat on the path is arithmetic, not compiling (13 s alone,
no compiles): every path date recalibrates a basket of one helper per reference-grid month,
each bucket a 60-step bisection around the 160-step root solve of `_analytic_lgm`. A first
call still compiles a few large programs (each bootstrap bucket per helper shape, the grid
induction and its derivatives), and each worker process compiles them again unless a
persistent compilation cache is configured (`JAX_COMPILATION_CACHE_DIR`; [I-72](#i-72)).
`demos/demo_profile_small.py`'s trace was truncated at the profiler's event cap before this
change and has not been re-measured.

**To close.** Profile a portfolio job (`JAX_RISK_PROFILE_DIR`,
[profiling](../concepts/profiling.md)); cut the American recalibration's arithmetic without
changing its root (an early exit of a bisection once its bracket stops moving keeps every
value); measure first-call compile time per product. `PricingConfig(recalibrate=False)` and
the AD Greeks method exist where ORE's semantics are not needed.

<a id="i-61"></a><a id="p-1"></a>
### I-61 — Nothing runs on more than one device

**Severity:** Medium · **Status:** OPEN · **Found:** 2026-09-24, audit P-1

**What is wrong.** The project's goal is to run across multiple TPUs and compare many
low-precision paths against fewer FP64 paths in equal wall time. No code uses `shard_map`,
`jax.sharding` or `pmap`; the worker pool runs whole jobs side by side, one per process.

**To close.** Shard the scenario axis: Sobol draws (per-device skip-ahead or scrambles; the
seed exists), path evolution, pricing and exposure are scenario-parallel; VaR/ES order
statistics need one cross-device step. The engine worker of roadmap 1.8 owns every device on its
host, so there is no pool sizing or chip pinning; on a pod slice, one worker per host
(`jax.distributed.initialize`).

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

**To close.** Portfolio jobs: the durable SQLite job queue of roadmap 1.8 (decision A-14,
[details](details/precision.md#114-the-queue-a-14-not-a-main-priority)), with failure classes
(`bad-terms`, `missing-market-data`, `unsupported-product`, `numerical-failure`,
`infrastructure`) and an `interrupted` state. On the EOD path, add the durable accepted-attempt record, a boot sweep and
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
### I-10 — Per-trade results keyed by position beside the echoed ids

**Severity:** Low · **Status:** PARTIAL · configs closed by roadmap 1.3

**What is wrong.** Every trade config carries a required `trade_id` (ORE's `<Trade id>`,
unique in a portfolio) and `PortfolioResult.trade_ids` echoes them in request order, but
`greeks`, `base_npv_per_trade`, `trade_exposures` and the cube's trade axis are still keyed by
position, so a consumer must zip them with `trade_ids`. The EOD boundary is closed
(`engine/integration/identity.py`).

**To close.** Roadmap 4.1: every per-trade result row carries its id.

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
### I-56 — Market risk and the CAM calibration have no route; two routes named like versions

**Severity:** Medium · **Status:** PARTIAL · **Found:** 2026-09-30, decision A-2 · narrowed by
roadmap 1.3

**What is wrong.** Since 1.3 one request shape (`MarketPortfolioRequestSchema`) serves
`POST /portfolio/price` and `POST /v2/portfolio/price` and reaches the run configuration: the
model per currency (`simulation.ir`, `"model": "HullWhite"`), the engine per product (with the
Jamshidian engine's model), the Greeks method and ORE's sensitivity settings, precision and
the reporting currency; trades carry ids. Left:

1. Two routes for one request, one named like a version (`/v2`, `schema_version: "2"`).
2. Unreachable over HTTP: market-risk VaR/ES (`engine.market_risk.run_market_risk`, no
   route); the CAM's calibration as a standalone run (`POST /calibration/lgm` is the older
   Hagan bootstrap on a caller-given basket, not the per-currency CAM calibration or a trade's
   basket); `LgmSwaptionEngineConfig.shift_horizon` (only 0 is implemented, [I-32](#i-32)).
3. No completeness test fails when a configuration setting has no API field.

Nothing is priced wrongly; unreachable settings run at documented defaults.

**To close.** Decided (A-2): one route, the old names kept as aliases; a market-risk route; the
CAM calibration route; the completeness test
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
### I-55 — Unproven precision combinations are not flagged

**Severity:** Medium · **Status:** PARTIAL · **Category:** Architecture · **Found:** 2026-09-24,
audit A-1; decisions A-9, D-9

**What is wrong.** Any per-stage combination may be run (decision D-9), but nothing records
which combinations are shown adequate for which figure, so a float32 exposure profile looks
exactly like a validated one.

**Closed part (roadmap 1.4, 2026-10-01).** The stages and the mechanism: `engine/precision/`
gives the simulation, the scenario market and path pricing (market-risk revaluation included)
a storage, compute and accumulate format each (`Precision`, float64 by default, float64 or
float32 today), at explicit cast points; kernels follow their inputs' dtype under strict
promotion (CI); reductions over paths are float64. The 32/64 `PrecisionConfig` is refused,
naming the replacement. `check_run`'s refusal, `run_market_risk`'s flag set, `_PRICING_LOCK`
and the per-precision pool tiers are gone. Tests: `tests/test_precision.py`,
`tests/test_portfolio_entrypoint.py::TestPricePortfolioPrecision`,
`tests/test_market_risk.py::TestRun`. Before 1.4 a "float32" (`simulation=32`) run priced its
paths partly in float64: NumPy float64 coupons and volatilities promoted the float32 curves
(the cube was float64), while path fixings ran in float32. Roadmap 1.5 (2026-10-02) made the
pricing stage per product and per trade (`Precision.by_product`, `by_trade`, one resolver
`precision_for`, decision A-15) in both pipelines: each cube column and each market-risk P&L
column is priced and stored at its trade's precision, exactly as the trade alone at that
precision (`tests/test_precision.py::TestPerTradePortfolio`, `TestPerTradeMarketRisk`).
Roadmap 1.6 (2026-10-02) enabled storage in float16, bfloat16 and both FP8 formats, with a
power-of-two block scale per 32 paths and nearest or stochastic rounding (`Stored`,
`Precision.rounding`, `rounding_seed`; `tests/test_precision.py::TestScaledStorage`,
`TestScaledStoragePipeline`); its first measurements are
[details/precision.md §15.3](details/precision.md#153-storage-through-the-pipeline) and
[I-75](#i-75).

**Reach.** Every reduced-precision result: its figures carry no statement of whether the
combination has been validated for them. Default (float64) runs are unaffected.

**To close.** Roadmap 2.7 (A-11): the evidence table per figure and precision combination
against the acceptance standard (Basel III's P&L attribution test and the Basel plan's P6.2
rule), and a warning on any result whose combination has no passing row. Compute below
float32 is [F-07](features.md#f-07) (roadmap 2.8).

<a id="i-72"></a>
### I-72 — Worker pools pickle ORE objects, compile per worker and would contend for TPU chips

**Severity:** Medium · **Status:** OPEN · **Category:** Architecture · **Found:** 2026-10-01,
precision design review; decision A-14

**What is wrong.** `engine/portfolio/worker_pool.py` runs HTTP jobs in a
`ProcessPoolExecutor` (one pool for every precision since roadmap 1.4; one per precision tier
before). Each job's request is frozen and thawed because ORE's
SWIG objects do not pickle; each worker compiles every job shape again; a crashed worker
likely breaks its pool for every later job (nothing handles `BrokenProcessPool`); and on a
TPU host, where one process owns a chip, several workers would need chip pinning and would
stand in the way of sharding one job across all devices ([I-61](#i-61)).

**Reach.** HTTP portfolio jobs only. `price_portfolio` called from Python runs in the
caller's process and is unaffected; the EOD path has its own execution and store.

**Current handling.** Works on one CPU device; the cost is complexity and a compile per
worker.

**To close.** Roadmap 1.8: the API writes the request JSON to a durable SQLite job queue; one
single-threaded engine worker process per host takes jobs from it, parses the JSON itself
(no freeze/thaw), owns every device on the host and writes results back; a supervisor
restarts a crashed worker, whose running job is marked `interrupted`
([details](details/precision.md#11-execution-architecture)). Tests: jobs queued together give
the same bits as run one after another; a failing job fails only its own row; a killed worker
leaves `interrupted`; a second identical job compiles nothing; the full suite on Linux.

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

**Reach.** Seasoned swaps from the EOD boundary only; the engine itself prices seasoned
swaps given their fixings, and on every path paid flows drop out and coupons fix by
`FixingManager`'s rule, under either model (the first half of this entry, the Hull-White
model keeping paid flows, closed with roadmap 1.3: `tests/test_hull_white_model.py::
TestPaidFlowsAndMaturity`, `tests/test_valuation.py`).

**To close.** `pastFixings` from TraderX ([details](details/traderx-integration.md)).

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
2026-09-24.

**Hypothesis, unproven.** The worker pool lives for the interpreter; before roadmap 1.3
`tests/test_api.py` created pools and never shut them down, so later in-process compiles ran
with worker children attached. Pairing modules does not reproduce it. Since 2026-10-02 the
suite does share JAX's on-disk compilation cache between its processes (xdist workers and
the pools they start, `tests/conftest.py`); JAX writes an entry without a lock and recompiles
when one cannot be read, so a torn entry costs a compile, not a crash, but an abort after
that date should rule the cache in or out first (rerun with `JAX_COMPILATION_CACHE_DIR`
pointing at an empty directory).

**Current handling.** `tests/test_api.py` and `tests/test_worker_pool.py` shut the pool down
after the module, and the concurrency test builds a fresh pool (it failed deterministically
when run after the pricing tests, because the executor reused one started worker instead of
spawning a second). The intermittent wall-clock overlap assertion between pricing jobs of two
precision tiers went with the tiers (roadmap 1.4); concurrency is shown by two sleeping jobs
on two worker PIDs, which is deterministic.

**To close.** Repeated clean full runs against a known-bad baseline; one green run proves
nothing.

<a id="i-66"></a><a id="q-2"></a>
### I-66 — No linter or type checker

**Severity:** Low · **Status:** OPEN · **Found:** 2026-09-24, audit Q-2 (pins and CI done)

**What is wrong.** No ruff/flake8, mypy/pyright or pre-commit. `pyflakes engine` today
reports unused imports in `integration/{bundle,market_inputs,pipeline,result,terms}.py`, an
f-string without placeholders in `integration/equity.py`, a string forward reference in
`instruments/bermudan_swaption.py`, and re-exports that need `__all__` (pyflakes ignores
`# noqa`: `portfolio/{__init__,validation,request}.py`, `models/{ore_builders,hull_white}.py`,
`instruments/bermudan_swaption.py`). `requirements.txt` does not mention the `profiling` extra.

**To close.** Add ruff to `pyproject.toml` and CI, fix or mark each finding (check that a
"re-export" is actually imported elsewhere first), then a type checker on `engine/`.

<a id="i-67"></a><a id="q-3"></a>
### I-67 — Test modules import each other and repeat fixtures

**Severity:** Low · **Status:** OPEN · **Found:** 2026-09-24, audit Q-3

**What is wrong.** Most cross-module imports moved to `tests/support/` with roadmap 1.3
(`portfolio`, `lgm_engine`, `greeks`); test modules still import `bermudan_references` and
`date_helpers` from the tests directory itself.
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
| <a id="i-10-configs"></a>I-10 (configs) | Trade configs had no identity (now a required `trade_id`, echoed as `PortfolioResult.trade_ids`) | `tests/test_trade_configs.py::TestEveryTradeNamesItselfAndItsDate`, `tests/test_portfolio_market_path.py::test_a_repeated_trade_id_is_refused` |
| <a id="i-13"></a>I-13 | A negative curve index priced against the wrong curve (trades now name a currency and index; a missing curve is refused before pricing, naming the trade) | `tests/test_portfolio_gap_fixes.py::TestCurveIndexValidatedBeforeAllPricing` |
| <a id="i-14"></a>I-14 | `generate_paths(precision=32)` leaked `jax_enable_x64=False` (`generate_paths` removed by roadmap 1.3; nothing toggles the flag) | `tests/test_portfolio_entrypoint.py::TestPricePortfolioConcurrency` |
| <a id="i-15"></a>I-15 | The worker-pool concurrency test could not observe concurrency (and, until 1.3, failed after the pricing tests by reusing one worker) | `tests/test_worker_pool.py::TestWorkerPoolConcurrency` |
| <a id="i-17"></a>I-17 | A malformed note date failed the whole bundle | `tests/test_integration_note.py::TestRefusalsAreNotePricingErrors` |
| <a id="i-19"></a>I-19 | The accrual tolerance rounded its own bound | `tests/test_integration_note.py::TestToleranceIsDerivedNotConstant` |
| <a id="i-20"></a>I-20 | Impossible calendar dates aborted the whole bundle | `tests/test_integration_note.py::TestImpossibleCalendarDates` |
| <a id="i-21"></a>I-21 | AD Greeks recompiled about 30 XLA programs per repeated call (the pricers are jitted with the trade as an argument; the Greeks no longer jit a fresh closure) | `tests/test_profiling_and_jit.py::TestCompileCounts::test_repeated_greeks_call_compiles_nothing`, `::test_a_different_trade_gets_its_own_greeks_from_warm_programs` |
| <a id="i-22"></a>I-22 | Each LGM calibration recompiled its bisection (now one program per helper shape, the basket an argument) | `tests/test_profiling_and_jit.py::TestCompileCounts::test_calibrations_of_one_basket_shape_share_one_program` |
| <a id="i-25"></a>I-25 | A scalar Greek crashed the HTTP result serializer | `tests/test_api_bond_schemas.py::TestBondGreeksSerializeOverHttp` |
| <a id="i-24"></a>I-24 | Bonds had no scenario NPV on the Hull-White model (refused; a broadcast t=0 value gave VaR 0, ES NaN) | `tests/test_hull_white_model.py::TestBondsOnEveryPath`, `tests/test_portfolio_bond_wire_through.py::TestBondsArePricedOnEveryPath` |
| <a id="i-26"></a>I-26 | Greeks for a bond maturing tomorrow crashed on the Theta reprice | `tests/test_portfolio_bond_wire_through.py::TestBondGreeksReachThePortfolioPath` |
| <a id="i-28"></a>I-28 | The `var_es` module demo crashed on a moved date | `tests/test_demos.py::TestComponentDemosRun` |
| <a id="i-29"></a>I-29 | A rounded exercise time silently dropped a coupon (exercise now given as dates) | `tests/test_ore_bermudan_oracle.py::TestExerciseDatesAreExact` |
| <a id="i-30"></a>I-30 | The `A(t,T)` variance term was nearly uncovered at t=0 (that formula went with the Hull-White pipeline; the model's bonds are QuantLib's on every state) | `tests/test_cam.py::test_hull_white_path_curves_equal_quantlibs_hull_white` |
| <a id="i-31"></a>I-31 | Bermudan/American floating coupons projected over the wrong period | `tests/test_ore_lgm_parity.py` |
| <a id="i-33"></a>I-33 | On Linux, worker-pool jobs hung once the parent had run JAX (fork) | `tests/test_worker_pool.py::TestPoolsSpawnOnEveryPlatform` |
| <a id="i-35"></a>I-35 | An American already in its window was exercisable on the evaluation date | `tests/test_trade_dates.py::test_seasoned_bermudan_and_american_equal_ore` |
| <a id="i-36"></a>I-36 | A non-ACT/365 floating leg projected the wrong forward | `tests/test_trade_dates.py::test_any_leg_day_count_equals_ore` |
| <a id="i-37"></a>I-37 | A European silently ignored `floating_spread` (refused by the Jamshidian engine, priced by Bachelier) | `tests/test_jamshidian.py::TestConfiguration::test_a_spread_is_refused_naming_the_trade`, `tests/test_valuation.py::test_european_today_equals_ores_default_engine` |
| <a id="i-38"></a>I-38 | Theta rolled a business day; ORE rolls a calendar day | `tests/test_trade_dates.py::test_swap_theta_equals_ore`, `tests/test_sensitivities.py::test_theta_rolls_one_calendar_day_from_a_friday` |
| <a id="i-39"></a>I-39 | Bond Theta had no add-back for a coupon paid in the period | `tests/test_sensitivities.py::test_theta_adds_back_a_bond_coupon_paid_on_the_theta_date` |
| <a id="i-40"></a>I-40 | The note's `rateSensitivity` ignored `fractionDecimals` | `tests/test_integration_note.py::TestSensitivityUsesTheDeclaredFractionDecimals` |
| <a id="i-41"></a>I-41 | A European at zero mean reversion priced at intrinsic value (now refused) | `tests/test_jamshidian.py::TestConfiguration::test_non_positive_reversion_is_refused` |
| <a id="i-42"></a><a id="m-1"></a>I-42 | Hull-White simulated curves were not arbitrage-free against the input curve (deflated 10y bond +6.8% at 2y on a 3→5% curve) | `tests/test_hull_white_model.py::TestCurveFittedDrift`, `tests/test_cam.py` |
| <a id="i-43"></a><a id="m-3"></a>I-43 | Hull-White options were worth 0 after expiry instead of carrying the swap | `tests/test_valuation.py::test_an_exercised_physical_option_becomes_its_swap_and_a_cash_one_leaves` (both models) |
| <a id="i-44"></a><a id="a-2"></a>I-44 | Hull-White scenario pricing mixed Hull-White and LGM | `tests/test_valuation.py::test_bermudan_on_every_path_equals_ore_recalibrated_on_the_path_curves` (both models), `tests/test_hull_white_model.py::TestOneModelOnePipeline` |
| <a id="i-45"></a>I-45 | The Hull-White numeraire was a left-point bank account (E[1/N] +1.8% at 3y) | `tests/test_hull_white_model.py::TestExactNumeraire` |
| <a id="i-46"></a>I-46 | Hull-White Europeans were priced off the model volatility, without Vega | `tests/test_hull_white_model.py::TestEuropeansOnTheMarketVolatility` |
| <a id="i-47"></a>I-47 | Hull-White options calibrated to one caller basket, not their own | `tests/test_valuation.py::test_option_today_equals_ores_calibrated_grid_engine`, `tests/test_hull_white_model.py::TestOneModelOnePipeline` |
| <a id="i-48"></a>I-48 | Zero curves extrapolated a flat zero rate; ORE a flat forward | `tests/test_treasury_instrument.py::TestCurveInterpolation`, `tests/test_curves.py` |
| <a id="i-52"></a>I-52 | Cash settlement was priced as physical | `tests/test_valuation.py::test_a_cash_settled_european_uses_the_par_yield_annuity` |
| <a id="i-62"></a><a id="p-2"></a>I-62 | Hull-White Bermudan/American scenario pricing ran on the host (that pricer is removed; options are priced by the vectorized per-path engine) | `tests/test_valuation.py::test_bermudan_on_every_path_equals_ore_recalibrated_on_the_path_curves` |
| <a id="i-63"></a><a id="a-3"></a>I-63 | Hull-White trade configs carried copies of model parameters | `tests/test_trade_configs.py::TestEveryTradeNamesItselfAndItsDate::test_a_model_or_curve_on_the_trade_is_refused` |
| <a id="i-64"></a><a id="a-4"></a>I-64 | A trade's evaluation date defaulted to ORE's thread-local global | `tests/test_trade_configs.py::TestEveryTradeNamesItselfAndItsDate::test_without_an_evaluation_date_it_is_refused` |
| <a id="i-65"></a><a id="a-6"></a>I-65 | Demo data, the modules' `__main__` demos and the ORE test oracle shipped inside `engine/` (now `demos/`, `tests/support/`) | `tests/test_import_layering.py::test_engine_ships_no_demo_or_test_code` |
| <a id="i-68"></a>I-68 | The Hull-White model was chosen by the market's type, not by the configuration (now `HullWhiteConfig` in `CamConfig.ir`) | `tests/test_run_config.py::TestEveryModelRunsWithEveryEngineAndMethod` |
| <a id="i-69"></a>I-69 | The market path silently ignored `precision.pricing`/`risk`/`calibration`, `calibration_targets`, and a `base_currency` contradicting the simulation (now refused by name) | `tests/test_run_config.py::TestWhatThePipelineDoesNotImplementIsRefused`, `::TestConfigurationValues`, `tests/test_api_market_path.py::test_an_unpriceable_request_is_a_400_and_no_job` |
| <a id="i-70"></a>I-70 | Bump Theta of a bond maturing the next day raised `BondPricingError` (now redemption − NPV, as ORE) | `tests/test_portfolio_bond_wire_through.py::TestBondGreeksReachThePortfolioPath::test_a_bond_maturing_tomorrow_does_not_crash_the_greeks` |
| <a id="i-71"></a>I-71 | A float32-tier worker turned x64 off, so its pricing and exposure ran in float32 where an in-process run used float64 | `tests/test_worker_pool.py::TestSubmitPricingJobRouting::test_float32_job_returns_correct_result` |
| <a id="i-74"></a>I-74 | A calibration basket read ORE's global evaluation date, so a later date (another caller's, or the wall clock past a helper's fixing) failed it with a missing fixing | `tests/test_ore_lgm_calibration.py::test_the_basket_does_not_depend_on_ores_global_evaluation_date` |
| <a id="m-4"></a>Audit M-4 | Trades were defined relative to the evaluation date (now absolute dates) | `tests/test_trade_dates.py` |
| <a id="m-5"></a>Audit M-5 | Theta re-rolled the trade instead of ageing it | `tests/test_trade_dates.py` |
| <a id="r-1"></a>Audit R-1 | Cube quantiles were reported as VaR/ES (now exposure profiles; market-risk VaR/ES by t=0 revaluation) | `tests/test_exposure.py`, `tests/test_market_risk.py`, `tests/test_market_risk_ore_parity.py` |
| <a id="p-3"></a>Audit P-3 | The precision study bypassed the engine's own FP32 path | `demos/demo_precision.py` |
| <a id="a-5"></a>Audit A-5 | Instruments imported validators from the layer above | `tests/test_import_layering.py` |
| <a id="a-7"></a>Audit A-7 | The bond pricer was a separate plain-Python implementation | `tests/test_market_risk.py::TestRevaluation` |