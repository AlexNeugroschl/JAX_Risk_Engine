# Known Issues

Open defects and important shortcomings: what each does to a number or a caller, and what
closing it takes. The order of work is in [roadmap.md](roadmap.md); the rules for this file
(statuses, severities, how to add and close an entry) are in [README.md](README.md). Fixed
issues keep one line in the [closed ledger](#closed).

Read this before trusting an exposure profile or a result served by the TraderX routes:
[I-57](#i-57) can return another submission's result, and a Bermudan's or American's values on
the paths, and so its exposure, are 0.5–4% above ORE's simulation ([I-49](#i-49)); swaps,
Europeans and bonds equal ORE's simulation path by path since 2026-10-07.

## Verification status

Last full runs, 2026-10-08/09, on the code of the package layout (I-92: renames and moves
only). 2,772 collected on Windows and Linux: the 2,767 of before, each mapped to its test in
the renamed files, less six replaced (the two `DAY_COUNTER` alias tests of
`tests/test_day_count_roles.py`, the three cases of the old layering test and its eager-import
test) and plus eleven (their replacements and the layout's rules in
`tests/test_import_layering.py`). Every run printed its summary line, and no run has a
`FAILED` or `ERROR` line:

- **Windows**, `-n 8`: **2,762 passed, 4 skipped, 6 xfailed, 0 failed**, 15m00s (the skips as
  before; the expected failures are the Bermudan/American L3 cases of
  `tests/test_ore_xva_parity.py`, strict, waiting on [I-49](#i-49)). After it, three edits
  that change no behaviour (an f-string without placeholders in `engine/traderx/equity.py`,
  two docstrings); their modules' tests passed (148), and the runs below include the first.
- **Linux, CPU** (Docker `python:3.11`, the pinned requirements, `-n 4`): **2,760 passed,
  6 skipped, 6 xfailed, 0 failed**, 24m35s; the two further skips need the `reference/` ORE
  and TraderX checkouts, which were not copied in. The first complete Linux run since the
  WSL run of 2026-10-07 stopped at 78% (out of memory), so rule 5 is met for the job queue's
  changes of 2026-10-07 too.
- **Fast tier under strict dtype promotion** (`JAX_NUMPY_DTYPE_PROMOTION=strict`, the CI job,
  `-n 8`): 2,645 passed, 4 skipped, 6 xfailed, 3m18s.
- The GPU suite was not run: no kernel changed.

Red first, on the code before (a worktree of `7d52e17`, the new `tests/test_import_layering.py`
copied in): 6 of its 11 tests fail, naming `market.py` and `day_count.py` at the root of
`engine/`, `var_es.py` and the other modules at the root of `engine/risk/`, the packages with
no layer, and 62 imports against the layers (among them every instrument importing
`engine.models`, where the schedules and the Bermudan's grid engine's model then lived).

Golden snapshot (2026-10-09), on CPU, from a worktree of `7d52e17` against the new tree, 359
arrays: all 359 identical in value, dtype and shape
([details/precision.md §13.1](details/precision.md#131-bit-for-bit-and-ore-parity)).

The fast tier (`-m "not slow"`) alone is not a full verification and is never recorded here. Rules:
[README.md](README.md#verification-rules).

## Summary

| ID | Issue | Sev. | Status | Category | Roadmap |
|---|---|---|---|---|---|
| [I-04](#i-04) | Seasoned TraderX swaps: the export has no past fixings | High | OPEN | Scope | External |
| [I-05](#i-05) | No faithful USD-SOFR / ACT-360 swap construction | High | OPEN | Scope | External |
| [I-07](#i-07) | No corporate bond, equity or listed-option pricer | Medium | OPEN | Scope | By demand |
| [I-08](#i-08) | A running TraderX attempt is lost on restart | Medium | PARTIAL | API | Core |
| [I-16](#i-16) | `rateSensitivity` is parallel-only | Medium | OPEN | Scope | External |
| [I-18](#i-18) | No equity spot or FX source; equity positions refused | Medium | OPEN | Scope | External |
| [I-23](#i-23) | `accrualBasis` strictness rests on an unconfirmed reading | Medium | ASSUMPTION | API | Fix what exists |
| [I-27](#i-27) | Long full-suite runs can hard-abort inside XLA or lose a worker process | Medium | OPEN | Tooling | Fix what exists |
| [I-32](#i-32) | Bermudan/American engine only at `ShiftHorizon = 0`, not ORE's default 0.5 | Medium | OPEN | Correctness | Core |
| [I-49](#i-49) | Per-path recalibration differs from ORE's in two details | Medium | OPEN | Correctness | Core |
| [I-73](#i-73) | A per-path recalibration that misses its basket is not flagged | Low | OPEN | Correctness | Core |
| [I-75](#i-75) | Storage below 32 bits keeps few bits of a concentrated array's spread | Low | OPEN | Correctness | Core |
| [I-87](#i-87) | The HTTP API is one route per engine function, not one run request | Medium | OPEN | API | Core |
| [I-88](#i-88) | The API completeness test checks field names, not that a value reaches the engine | Low | OPEN | Tooling | Core |
| [I-89](#i-89) | A retried job submission runs the job twice | Low | OPEN | API | Core |
| [I-51](#i-51) | Sensitivities not checked against ORE's sensitivity analytic | Medium | OPEN | Validation | Core |
| [I-78](#i-78) | AD and bump Greeks differ by up to 2% on a sloped market | Medium | OPEN | Validation | Core |
| [I-53](#i-53) | On Windows, pricing runs slower after AD Greeks; Newton's bootstrap compiles 3–4 s longer from scratch | Low | PARTIAL | Performance | Fix what exists |
| [I-54](#i-54) | No swaption smile: options away from the money read the ATM vol | Medium | OPEN | Correctness | New |
| [I-55](#i-55) | Unproven precision combinations are not flagged | Medium | PARTIAL | Architecture | Core |
| [I-57](#i-57) | TraderX: a cached result is served before the submission id is checked | High | OPEN | API | Fix what exists |
| [I-58](#i-58) | TraderX: two concurrent submissions of one workload both execute | Medium | OPEN | API | Fix what exists |
| [I-59](#i-59) | TraderX: `calculations` and `reportingCurrency` accepted, keyed, then ignored | Medium | OPEN | API | Fix what exists |
| [I-60](#i-60) | TraderX result schema has no stated policy on added fields | Low | ASSUMPTION | API | Fix what exists |
| [I-76](#i-76) | The job queue keeps every job and result forever | Low | OPEN | API | Core |
| [I-77](#i-77) | A worker that cannot start leaves jobs `pending` with no signal | Low | OPEN | API | Core |
| [I-91](#i-91) | A long job delays every job behind it, and no job can be cancelled | Low | OPEN | API | Core |
| [I-90](#i-90) | The HTTP API has no authentication, rate limit or body-size limit | Low | OPEN | API | Fix what exists |
| [I-61](#i-61) | Nothing runs on more than one host; multi-device speed unmeasured | Medium | PARTIAL | Performance | Core |
| [I-66](#i-66) | No linter or type checker | Low | OPEN | Tooling | Fix what exists |
| [I-67](#i-67) | Test modules import each other and repeat fixtures | Low | OPEN | Tooling | Fix what exists |
| [I-81](#i-81) | A cold job compiles about 180 one-operation programs; market risk vmaps a closure | Low | OPEN | Performance | Fix what exists |
| [I-83](#i-83) | A Bermudan/American on the paths needs 0.3–0.7 MB per path and step: 64k paths run out of memory | Medium | OPEN | Performance | Core |
| [I-93](#i-93) | TraderX's submissions are priced by their own pricers, not the engine's | Medium | OPEN | API | Core |
| [I-94](#i-94) | The LGM is calibrated by two implementations | Low | OPEN | Architecture | Fix what exists |

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
### I-73 — A per-path recalibration that misses its basket is not flagged

**Severity:** Low · **Status:** OPEN · **Category:** Correctness · **Found:** 2026-10-01, the
dtype review

**What is wrong.** On every path and date a Bermudan/American is recalibrated on
σ ∈ [1e-6, 0.2] (`engine.calibration.ore_lgm.bootstrap_sigma`, by the engine's root solver). A
helper whose volatility is not attainable in the bracket ends at its edge, and
`bootstrap_sigma` flags it (`hit_ceiling`), but `engine.pricing.bermudan.path_sigmas` drops
the flag, so that path is
priced on a model that does not reprice its basket and nothing says so. Today's calibration
refuses the same case (`calibrate_on`). ORE's `LgmBuilder` logs a structured warning when the
calibration error exceeds its tolerance and fails unless `continueOnCalibrationError`
(`OREData/ored/model/lgmbuilder.cpp`).

**Reach.** Bermudan/American cube values on paths where rates move far enough that a
helper's market volatility is out of reach (extreme paths, long horizons); exposure figures
through them. Not seen on the test markets. Since 1.4 the flag is also correct in float32 (its
`1 - 1e-9` tolerance rounded away there; `tests/test_precision.py::TestRecalibrationInFloat32`).

**Current handling.** None.

**To close.** With [I-49](#i-49) (the per-path recalibration against ORE's): count the paths and
dates whose recalibration hit the bracket and carry a warning naming the trade and the counts
on the result, as ORE's structured warning; a test with an unattainable path volatility.

<a id="i-75"></a>
### I-75 — Storage below 32 bits keeps few bits of a concentrated array's spread

**Severity:** Low · **Status:** OPEN · **Category:** Correctness · **Found:** 2026-10-02,
the storage measurements ([details/precision.md §15.3](details/precision.md#153-storage-through-the-pipeline))

**What is wrong.** A scaled storage format (2026-10-02) keeps each value to the format's
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
[I-55](#i-55)'s evidence table and warnings (I-55). Documented in the user guide and the
portfolio entry point.

**To close.** Store such classes relative to a level, so the format's bits go to
the spread: the cube relative to each trade's t=0 value and the curves relative to their
path-independent part (the difference form of [§8.2](details/precision.md#82-the-difference-form),
already computed in float64 by `build_scenario_market`), or a per-block offset beside the scale
in `Stored` (a decision on A-10's single rule). Measure each class with and without it, keep
what the evidence table needs, and test that an FP8 bond column's bias falls below its Monte
Carlo standard error.

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

**Reach.** Wall time only; no number. Windows CPU runs; a worker's first job.

**Current handling.** None needed for correctness.

**To close.** The Windows slowdown's cause, and the helper's price traced once per
bucket program, the snapshot bit for bit. Related: the grid rollback's memory and time on the
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

**To close.** One worker per host (`jax.distributed.initialize`), process 0
claiming each job and handing it to the others
([details](details/precision.md#113-multi-device-and-multi-host)); then wall time against
device count on TPU (and H100), feeding 5.2.

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

**To close.** On the TraderX path, the durable accepted-attempt record, a boot sweep
and an `interrupted` lookup state (TraderX acceptance case A-09); the portfolio queue's
`interrupt_running` is the pattern.

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

<a id="i-57"></a>
### I-57 — TraderX: a cached result is served before the submission id is checked

**Severity:** High · **Status:** OPEN · **Found:** 2026-09-17, TraderX acceptance case A-02
(FR-07), reproduced in response v7; still present

**What is wrong.** In `engine/api/traderx_routes.py` the `reuseExistingResult` branch returns a
completed attempt from `STORE.lookup(key)` before `STORE.start(key, submission_id=...)`, which
is where `SubmissionIdConflict` is raised. A coordinator that reuses a submission id across
bundles gets another workload's priced result under its own id, with `"reused": true`.

**To close.** Extract the binding check from `AttemptStore.start` and run it before any
cache read. Red first with A-02's case, added to this suite.

<a id="i-58"></a>
### I-58 — TraderX: two concurrent submissions of one workload both execute

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

**To close.** The supervisor records its child's exit status, and a poll of a `pending` job
while the worker has exited at startup N times running reports it (in `error`, or a
`worker` field on `/health`); a test with a worker command that exits at once.

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
describes. A run of one analytic must equal today's direct call bit for bit.

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

**To close.** Before the API is reachable from outside a trusted network:
authentication chosen with TraderX (a bearer token or mutual TLS), a body-size limit, and a
per-caller limit on queued jobs; a test per refusal (`401`, `413`, `429`).

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
([details/package-layout.md §3](details/package-layout.md#3-merges)).

---

## Architecture

<a id="i-55"></a><a id="a-1"></a>
### I-55 — Unproven precision combinations are not flagged

**Severity:** Medium · **Status:** PARTIAL · **Category:** Architecture · **Found:** 2026-09-24,
audit A-1; decisions A-9, D-9

**What is wrong.** Any per-stage combination may be run (decision D-9), but nothing records
which combinations are shown adequate for which figure, so a float32 exposure profile looks
exactly like a validated one.

**Closed part (2026-10-01).** The stages and the mechanism: `engine/precision/`
gives the simulation, the scenario market and path pricing (market-risk revaluation included)
a storage, compute and accumulate format each (`Precision`, float64 by default, float64 or
float32 today), at explicit cast points; kernels follow their inputs' dtype under strict
promotion (CI); reductions over paths are float64. The 32/64 `PrecisionConfig` is refused,
naming the replacement. `check_run`'s refusal, `run_market_risk`'s flag set, `_PRICING_LOCK`
and the per-precision pool tiers are gone. Tests: `tests/test_precision.py`,
`tests/test_portfolio_entrypoint.py::TestPricePortfolioPrecision`,
`tests/test_market_risk.py::TestRun`. Before then a "float32" (`simulation=32`) run priced its
paths partly in float64: NumPy float64 coupons and volatilities promoted the float32 curves
(the cube was float64), while path fixings ran in float32. On 2026-10-02 the
pricing stage became per product and per trade (`Precision.by_product`, `by_trade`, one resolver
`precision_for`, decision A-15) in both pipelines: each cube column and each market-risk P&L
column is priced and stored at its trade's precision, exactly as the trade alone at that
precision (`tests/test_precision.py::TestPerTradePortfolio`, `TestPerTradeMarketRisk`).
Then storage was enabled in float16, bfloat16 and both FP8 formats, with a
power-of-two block scale per 32 paths and nearest or stochastic rounding (`Stored`,
`Precision.rounding`, `rounding_seed`; `tests/test_precision.py::TestScaledStorage`,
`TestScaledStoragePipeline`); its first measurements are
[details/precision.md §15.3](details/precision.md#153-storage-through-the-pipeline) and
[I-75](#i-75). The same day every result gained a `PrecisionReport` (the policy as
run, the formats read from the stored arrays, the devices) and the paired float64 sample
(`Precision.paired_fraction`, decision A-13): EPE and ENE are two-level estimates, PFE, VaR and
ES measured against float64 on the same paths (`tests/test_precision_report.py`). A result
now says how far it is from float64; it does not yet say whether that is good enough.

**Reach.** Every reduced-precision result: its figures carry no statement of whether the
combination has been validated for them. Default (float64) runs are unaffected.

**To close.** The evidence table (A-11) per figure and precision combination
against the acceptance standard (Basel III's P&L attribution test and the Basel plan's P6.2
rule), and a warning on any result whose combination has no passing row. It is measured by
[I-75](#i-75)'s harness on each kernel family as F-07 rewrites it (the Bermudan/American rows
after [I-32](#i-32) and [I-49](#i-49)), so it describes the kernels that ship. The harness also measures the paired estimator's coverage through the pipeline: its standard errors treat paths as
independent, while Sobol paths are not and the rounding errors of the 32 paths of a block
share a scale (the synthetic coverage tests and three pipeline seeds pass; that is not yet
evidence at scale). Compute below
float32 is [F-07](features.md#f-07).

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

**Current handling.** Since 2026-10-04 no pool, and no executor thread, exists: HTTP tests
start one engine worker, a separate process with no thread in the test process, and each
module stops it when it ends (`tests/conftest.py`); a worker also exits once its parent is
gone. If an abort recurs, the pool hypothesis is ruled out.

**To close.** Measure each xdist process's peak resident memory over a full run (Linux,
`-n 4`) and the heaviest tests; if memory explains the deaths, lower `-n` for the `full` CI job
or slim the heaviest fixtures. Then repeated clean full runs against a known-bad baseline,
with `JAX_COMPILATION_CACHE_DIR` on an empty directory to rule the cache in or out; one green
run proves nothing.

<a id="i-66"></a><a id="q-2"></a>
### I-66 — No linter or type checker

**Severity:** Low · **Status:** OPEN · **Found:** 2026-09-24, audit Q-2 (pins and CI done)

**What is wrong.** No ruff/flake8, mypy/pyright or pre-commit, so nothing keeps the code
clean. `pyflakes engine` reports nothing since the layout change of 2026-10-08 (I-92), which
removed its findings with the modules they were in (re-export shims, unused imports, a string
forward reference, an f-string without placeholders); `pyflakes tests demos` still reports 19
unused imports and variables. `requirements.txt` does not mention the `profiling` extra.

**To close.** As decided (A-20), in two steps. First, now: ruff in
`pyproject.toml` and CI with pyflakes' rules only, each finding fixed or marked (a re-export
listed in `__all__`), so `engine/` stays clean. Then, among the fixes of what exists: a type
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
| <a id="i-62"></a><a id="p-2"></a>I-62 | Hull-White Bermudan/American scenario pricing ran on the host (that pricer is removed; options are priced by the vectorized per-path engine) | `tests/test_pricing.py::test_bermudan_on_every_path_equals_ore_recalibrated_on_the_path_curves` |
| <a id="i-63"></a><a id="a-3"></a>I-63 | Hull-White trade configs carried copies of model parameters | `tests/test_trade_configs.py::TestEveryTradeNamesItselfAndItsDate::test_a_model_or_curve_on_the_trade_is_refused` |
| <a id="i-64"></a><a id="a-4"></a>I-64 | A trade's evaluation date defaulted to ORE's thread-local global | `tests/test_trade_configs.py::TestEveryTradeNamesItselfAndItsDate::test_without_an_evaluation_date_it_is_refused` |
| <a id="i-65"></a><a id="a-6"></a>I-65 | Demo data, the modules' `__main__` demos and the ORE test oracle shipped inside `engine/` (now `demos/`, `tests/support/`) | `tests/test_import_layering.py::test_engine_ships_no_demo_or_test_code` |
| <a id="i-68"></a>I-68 | The Hull-White model was chosen by the market's type, not by the configuration (now `HullWhiteConfig` in `CamConfig.ir`) | `tests/test_run_config.py::TestEveryModelRunsWithEveryEngineAndMethod` |
| <a id="i-69"></a>I-69 | The market path silently ignored `precision.pricing`/`risk`/`calibration`, `calibration_targets`, and a `base_currency` contradicting the simulation (now refused by name) | `tests/test_run_config.py::TestWhatThePipelineDoesNotImplementIsRefused`, `::TestConfigurationValues`, `tests/test_api_portfolio.py::test_an_unpriceable_request_is_a_400_and_no_job` |
| <a id="i-70"></a>I-70 | Bump Theta of a bond maturing the next day raised `BondPricingError` (now redemption − NPV, as ORE) | `tests/test_portfolio_bond_wire_through.py::TestBondGreeksReachThePortfolioPath::test_a_bond_maturing_tomorrow_does_not_crash_the_greeks` |
| <a id="i-71"></a>I-71 | A float32-tier worker turned x64 off, so its pricing and exposure ran in float32 where an in-process run used float64 | `tests/test_engine_worker.py::TestEngineWorkerPricing::test_jobs_queued_together_give_the_bits_of_jobs_run_one_after_another` (float64 and float32 jobs equal the direct call) |
| <a id="i-72"></a>I-72 | HTTP jobs ran in a process pool that pickled each request with its ORE dates frozen as text, compiled every job shape once per worker, broke for every later job when one worker died (`BrokenProcessPool`), and would have needed chip pinning on TPU (now one engine worker per host behind a durable queue) | `tests/test_engine_worker.py` (`test_a_second_identical_job_compiles_nothing`, `test_a_failing_job_fails_only_its_own_row`, `test_a_worker_killed_mid_job_leaves_it_interrupted`, `test_the_worker_prices_the_request_the_route_validated`) |
| <a id="i-74"></a>I-74 | A calibration basket read ORE's global evaluation date, so a later date (another caller's, or the wall clock past a helper's fixing) failed it with a missing fixing | `tests/test_ore_lgm_calibration.py::test_the_basket_does_not_depend_on_ores_global_evaluation_date` |
| <a id="i-79"></a>I-79 | Never run on a GPU (now the `gpu` extra under Linux or WSL2, verified on an RTX 5060: no preallocation, the API's JAX on the CPU, float32 products at float32 rather than TensorFloat-32, deterministic kernels; the demo and the full suite run on the GPU) | `tests/test_accelerator_defaults.py`, `tests/test_environment.py::test_a_gpu_plugin_is_jaxlibs_version`, the full suite on the GPU |
| <a id="i-80"></a>I-80 | Importing `engine` changed JAX and XLA settings for the whole process (now x64 only: each matrix product states its precision through `engine.precision.matmul`; deterministic GPU kernels set by the engine worker and the tests; GPU preallocation off in the demos and the tests; 2026-10-06, decision A-22) | `tests/test_accelerator_defaults.py` (`TestImportingTheEngine`; `TestEveryMatrixProductStatesItsPrecision::test_in_every_pipeline`, red on the bare products) |
| <a id="i-82"></a>I-82 | An American priced on a path date after the last reference-grid date in its window and before its last exercise (no calibration helper left) raised `ValueError: Need at least one array to stack` (now the engine's volatility, as a calibration today with no helper; found on 2026-10-07) | `tests/test_root_solvers.py::test_a_date_with_an_exercise_left_and_no_helper_keeps_the_engine_volatility` |
| <a id="i-84"></a>I-84 | The simulation market held each curve's tenor points at the tenors' times from the simulation date; ORE's `ScenarioSimMarket` holds them at the times from the as-of date (swaps and Europeans up to 0.45% of their largest path value off ORE's cube; found by the ORE simulation oracle's L3, 2026-10-07) | `tests/test_ore_xva_parity.py::test_every_trades_cube_equals_ores_on_ores_paths`, `tests/test_pricing.py` (path curves at ORE's times) |
| <a id="i-85"></a>I-85 | Exposure profiles started from each trade's value on today's market; ORE's start from its value on the simulation market of the as-of date, the cube's `T0` (up to 4.2% on the shared portfolio) | `tests/test_ore_xva_parity.py::test_the_pipelines_profiles_start_from_ores_t0` |
| <a id="i-86"></a>I-86 | A cash-settled option's exposure stopped at its underlying's maturity, so its time-weighted EPE kept accruing after it settled; ORE's `Trade::maturity()` is its last exercise date (2026-10-07) | `tests/test_ore_xva_parity.py::test_the_profiles_of_ores_cube_are_ores_reports` |
| <a id="i-92"></a>I-92 | `engine/`'s packages were named for layers and two modules sat at its root (now subpackages only, each named for what it holds and layered: `market_data`, `models`, `instruments`, `traderx`, `calibration`, `market_simulation`, `pricing`, `risk/{greeks,market,counterparty}`, `run`, `api`; [details/package-layout.md](details/package-layout.md)) | `tests/test_import_layering.py::test_no_module_but_init_at_the_root`, `::test_packages_import_only_the_layers_below`, `::test_risk_kinds_import_only_the_kinds_below` |
| <a id="m-4"></a>Audit M-4 | Trades were defined relative to the evaluation date (now absolute dates) | `tests/test_trade_dates.py` |
| <a id="m-5"></a>Audit M-5 | Theta re-rolled the trade instead of ageing it | `tests/test_trade_dates.py` |
| <a id="r-1"></a>Audit R-1 | Cube quantiles were reported as VaR/ES (now exposure profiles; market-risk VaR/ES by t=0 revaluation) | `tests/test_exposure.py`, `tests/test_market_risk.py`, `tests/test_market_risk_ore_parity.py` |
| <a id="p-3"></a>Audit P-3 | The precision study bypassed the engine's own FP32 path | `demos/demo_precision.py` |
| <a id="a-5"></a>Audit A-5 | Instruments imported validators from the layer above | `tests/test_import_layering.py` |
| <a id="a-7"></a>Audit A-7 | The bond pricer was a separate plain-Python implementation | `tests/test_market_risk.py::TestRevaluation` |