# Roadmap

The order in which to work through [known-issues.md](known-issues.md) and
[features.md](features.md). Every open item appears here exactly once; an item that is not
here is not planned ([rules](README.md#lifecycle)).

**How the order is decided** (decision A-19, revised 2026-10-10; precision priorities
updated by the owner, 2026-10-15).

1. **The near-term milestone**: `demos/demo_profile_small.py` runs on the owner's local GPU
   and its profiler trace covers the whole job (stage 2, done 2026-10-07).
2. **The focus** (stage 3), three goals, in this order of priority where they compete:
   1. **Accuracy and measurement of existing precision formats**: fix rounding bias and
      error estimates in the supported formats through FP8, each format's error measured
      and stated on every result (the evidence table), then measure time versus accuracy
      on TPU and H100. FP4 and multilevel quantile correction are later, independent research
      extensions; neither blocks this first research result. The precision research is the
      project's purpose (root [README](../../README.md)).
   2. **Critical bugs**: a High defect (a wrong number on an ordinary input, or another
      caller's result served), and the numbers that are wrong or unproven where the precision
      work and every later stage build on them (Bermudans and Americans against ORE, the
      Greeks).
   3. **A robust API on every feature**: one run request for every analytic, over HTTP and in
      Python; every input checked before any work, every setting reaching the engine, a setting
      the run does not read refused by name, a retried submission run once, a job that can be
      cancelled and a worker failure that is reported.
3. **TraderX integration** (stage 4): its submissions through the run request into the
   engine's pricers, and the defects of its path.
4. **Basel III** (stage 5): regulatory figures as the Basel Framework specifies them, each
   traceable and reproduced by an oracle. It waits for the focus and for TraderX; the
   precision research's acceptance standard (A-11) already uses its tests, transcribed.
5. **Fix what exists** (stage 6): the remaining narrow defects, then speed, test and tooling
   defects.
6. **New capabilities** (stage 7): what the engine does not do yet.

A clear `engine/` (packages named for what they hold, nothing at the root but `__init__.py`,
[details/package-layout.md](details/package-layout.md)) was done first (2026-10-08), so that
every later step is written in the final layout.

Within a stage, work in the order of its table. Items marked *parallel* touch code that no
earlier step changes, and can start at any time. Items waiting on someone outside the project
are listed [separately](#waiting-on-others), with who to chase. A defect found along the way
goes to stage 3 if it is High or blocks a step there, ahead of the work it affects; otherwise
to stage 6, or before the step it blocks.

The project audit of 2026-10-10 ([project-audit.md](project-audit.md)) is merged into this
plan (2026-10-11): each of its findings is an item here, part of one, or set aside with its
reason in the audit's [disposition](project-audit.md#disposition-2026-10-11).

**Step numbers are this page's alone.** Steps are numbered in the order of work and
renumbered whenever it changes. Nothing else cites a step: code, error messages, tests and the
other documents cite the item (I-NN, F-NN), the decision (A-n), or a date, and
`tests/test_import_layering.py` fails on a step number in code.

**Every step keeps the project's goals.** Nothing above trades one away:

- **ORE parity.** Every ORE parity suite passes at its existing tolerance after every step. A
  new capability comes with its own ORE parity test (through the ORE simulation oracle,
  `tests/support/ore_xva_oracle.py`, where it needs a simulation), or it is refused by name.
- **Accuracy: correct, not bit for bit** (owner, 2026-10-10). A step may move numbers; it may
  not make them wrong. After a step, every ORE parity suite and every test passes at its
  tolerance, and the golden snapshot agrees with the one before within a small margin,
  relative to each array's largest magnitude (a Greek's to its trade's NPV): float64 to 1e-10,
  float32 to 1e-4. Formats below float32 are judged by the storage harness's statistical
  thresholds, not by the snapshot. A change beyond the margin is either a planned change of the
  model or sampling design (as I-49 moved default Bermudan/American values on the paths onto ORE's, 2026-10-12) or a
  bug; a planned one states its largest change per array and re-baselines. Nothing is held to bit-for-bit
  equality for its own sake, on CPU or GPU; the bisection solver stays as A-21's reference.
- **Speed and JAX.** A new or changed pricer is a module-level jitted function that takes the
  trade's data as a pytree argument
  ([profiling §3.7](../concepts/profiling.md#37-trade-data-as-traced-arguments-2026-10-02)),
  so there is no compile per trade, date, bump or call. It runs under the scenario-axis
  sharding, and the compile-count tests cover it.
- **Precision research.** Every path kernel is written in difference form, its matrix products
  in the stage's compute format and everything else in its accumulate format, one
  implementation for every precision (A-16; [details/precision.md
  §8](details/precision.md#8-compute-below-float32), 2026-10-10), and gets its rows in the
  precision harness (`tests/support/precision_harness.py`). Calibration, Greeks,
  reductions and the other fixed stages (A-10) stay float64. A formula, interpolation,
  arithmetic or storage change to a kernel reruns its rows; old rows are not reused because
  the format's name is unchanged.
- **Basel III.** Nothing a regulatory figure relies on is approximated silently: an input
  outside scope is refused by name, as today. Results stay auditable: the job queue keeps
  them, and 3.14's retention policy can keep a run indefinitely, which regulatory runs will be.
- **API compatibility.** Once the HTTP API has a client, changes are additive: every route,
  field, default and result shape keeps working, and a change of the contract is a revision of
  it (A-2; the names A-2 retired were removed outright on 2026-10-07, there being no client
  yet, by the owner's revision of that day). 3.12 replaces the routes with one run request on
  the same terms, before any client exists. A new setting arrives with its API field, which
  `tests/test_api_completeness.py` checks. The planned narrowings are each deliberate (as was
  the refusal of a zero or negative option notional, a short option taking ORE's `LongShort`,
  I-101, 2026-10-11): 3.11 refuses malformed inputs that are now accepted (an unsorted curve,
  an expiry with no whole period after it), 4.1 refuses bundles that break their contract,
  4.3 refuses inputs that are now accepted and then ignored (the old answer is wrong); a default
  changed to ORE's moved a number (`ShiftHorizon` 0.5, decision A-3, I-32, 2026-10-12). TraderX's routes (`/eod/...`) are
  TraderX's contract and keep their shape when 4.2 changes what runs behind them.
- **Tests.** A step that breaks a test of a private symbol rewrites it against the public
  entry ([I-67](known-issues.md#i-67)'s rule), so restructuring does not wait for 6.7.

## Where things stand

One pipeline, ORE's: a cross-asset model with a model per currency (the LGM by default, or
Hull-White, both ORE's `<LGM>` parametrizations), every trade valued on every path as ORE's
valuation engine does. It matches ORE component by component: t=0 prices to 1e-14 (swaps,
Europeans, bonds) and 4e-11 (calibrated Bermudans/Americans, at ORE's default `ShiftHorizon`
since 2026-10-12, I-32), model analytics to 1e-12
(the Hull-White model's bonds against QuantLib's `HullWhite` too), pricers on path curves to
1e-10 – 1e-12 under either model, market-risk VaR/ES per scenario to 2e-13, and since
2026-10-07 an ORE simulation run in-process is the reference for the assembled pipeline: every
trade of the shared portfolio equals it path by path, Bermudans and Americans since 2026-10-12
(to 1.9e-11), each the trade as ORE built it on the as-of date with the LGM it calibrated there,
kept on every path, since ORE's exposure simulation never recalibrates (I-49; decision A-24
removed the engine's per-path recalibration, which no ORE analytic does). Every LGM calibration
is accepted or refused as ORE's builder accepts it, a miss named (I-73, 2026-10-12; the
tolerance 1e-4 and one for the cross-asset model, A-24). Sensitivities are not yet shown to equal ORE's, and
market risk holds a calibrated Bermudan's LGM where ORE's historical P&L recalibrates it (I-112,
flagged on the result). Since 2026-10-11 an option
has ORE's `LongShort` (I-101): a short is minus the long one in every engine, today, on the paths
and in every risk figure, and equals ORE's short option booked through ORE's own trade builder,
today and on ORE's paths.
Since 2026-10-11 market risk reports in one currency, each trade converted at the as-of FX spot
(FX is not a shocked factor, and the result says so), and takes QuantLib's VaR rank at exact
boundaries (I-100, I-102).

Every model, engine, Greeks and precision choice is one run configuration with ORE's
defaults. The simulation, the scenario market and path pricing each take a storage, a compute
and an accumulate precision (since 2026-10-10 compute down to FP8, TensorFloat-32 and bfloat16
passes in kernels in difference form, each product's operand that every path shares in two terms
of the format since 2026-10-11, native or emulated by device; the Bermudan/American rollback since
2026-10-13, as matrix products, one operator per step shared by every path, so the demo's job runs
at 262,144 paths; storage down to float16, bfloat16 and FP8
with block scales, block offsets that keep a concentrated column's spread, and nearest or
stochastic rounding), set per product and per trade for pricing, in portfolio and market-risk
runs alike; every result carries a precision report, and a paired float64 sample corrects the
mean figures by a two-level estimator and measures the quantiles. Since 2026-10-09 a storage
harness measures every format's error against float64 per stage, figure, product and path
count, over independent seeds, rerunnable on any kernel change (I-75); since 2026-10-11 the
evidence table reads A-11's verdicts and path ceilings off it, and every result warns of each
figure whose combination has no passing row (I-55), and since 2026-10-13 it measures the share
of each option's exercise decisions a format flips (decision A-24). HTTP jobs go through a durable SQLite job queue to one single-threaded engine
worker process per host, which owns the host's devices (decision A-14) and splits each job's
scenarios across them. The engine runs on an NVIDIA GPU (Linux or WSL2, the `gpu` extra) with
deterministic kernels, and each matrix product states its own precision (A-22). A repeated job
compiles nothing; every calibration and exercise boundary is solved by one configurable root
solver (A-21). The HTTP API reaches every setting the engine has, through one route per
analytic (portfolio pricing, market risk, the cross-asset model's calibration). Nothing runs
on more than one host. TraderX submits end-of-day bundles through its own routes (`/eod/...`),
which price its Treasuries with their own pricers and refuse everything else by name; a
submission id binds to one workload before any cached result is returned (2026-10-10). Since
2026-10-08 `engine/` holds subpackages only, each named for what it holds and importing only
the layers below it ([details/package-layout.md](details/package-layout.md)).

---

<a id="stage-1--structure"></a>
Stage 1, structure (the configurable engine, decision A-1), is done (2026-10-04): one run
configuration, the precision mechanism through storage below 32 bits and the precision
report, and the engine worker behind the job queue, each change reproducing the numbers
before it bit for bit; the evidence is in
[details/precision.md §13.1](details/precision.md#131-bit-for-bit-and-ore-parity) and the
designs in [details/configurable-engine.md](details/configurable-engine.md) and
[details/precision.md](details/precision.md).

<a id="stage-2--the-demo-on-a-local-gpu"></a>
Stage 2, the demo on a local GPU with a whole trace, is done (2026-10-07):
`demos/demo_profile_small.py` runs through the HTTP API and the engine worker on the owner's
RTX 5060 (Linux or WSL2, the `gpu` extra) as on the CPU, and its trace covers the whole job,
each phase labelled, or one phase on request. On the way: importing `engine` sets only x64 and
every matrix product states its precision (A-22); each trade's AD Greeks are one program per
product, and a repeated job compiles nothing; every calibration and exercise boundary is
solved by one configurable root solver, a safeguarded Newton method by default with the
bisection kept as the reference (A-21), and an option's path dates of one basket shape are
calibrated together. The recalibration is 35–55 times faster on the CPU and 15–38 times on the
GPU, where a repeated demo job went from 5.5 s to 1.4 s
([profiling §2.0](../concepts/profiling.md#20-the-demo-measured-2026-10-05)). It left: the
cold job's one-operation programs (6.5), a slower pricing on Windows after AD Greeks and a cold
job 3 s slower for Newton's compile (6.6), and the grid rollback's memory, which stopped the
demo's job beyond a few thousand paths (fixed 2026-10-13, I-83: 262,144 paths run in 3.0 s and
2.0 GB on the CPU, 2.4 s on the GPU). The card's float64 runs at 1/64 of its float32
rate, so it is where correctness and the trace are checked, not where speed is measured (3.17).

<a id="stage-3--the-focus"></a>
## Stage 3 — The focus: low precision, critical bugs, a robust API

Existing-format precision fixes first where the three compete. The critical bugs are here because the precision
work and the later stages build on the numbers they fix (3.10; the Bermudans and Americans on
the paths, I-49 and I-73, closed 2026-10-12). Both High defects
the project audit of 2026-10-10 found, each small and touching no path kernel, are fixed:
market risk added trades' currencies unconverted (I-100, 2026-10-11, with VaR's rank at exact
boundaries and colliding quantile keys, I-102), and a negative option notional meant a short
European but a long Bermudan (I-101, 2026-10-11: ORE's `LongShort` on every option). The High
defect of 2026-10-10 (I-57, one caller served another's result) is fixed.

The evidence table is built (I-55, 2026-10-11): `compliance/precision-evidence.md`, generated
from the precision harness's campaigns of record against A-11's standard as cited data
(`compliance/precision-acceptance.json`: the P6.2 statistical rule, an absolute rule for
figures float64 holds fixed, MAR32.42's P&L attribution test transcribed once from the BIS
text, its second entry Basel P0.3's), each row with its path ceiling, code revision, market,
seeds and device; every portfolio and market-risk result warns of each figure without a
passing row. All five campaigns were measured again on one revision for it, every row equal to
its record. Under the 1% rule float32 storage and compute and the three bfloat16 passes pass
every figure to 4,096 or 16,384 paths (8,192 scenarios; a European under the bfloat16 passes
in the market stage to 1,024); below them, market risk's ES (and in
float16 and TensorFloat-32 compute its VaR) passes to 8,192 scenarios (ES_99 with the P&L stored in
float16 by stochastic rounding to 2,048 since its re-measurement after I-32, 2026-10-12), while no format at 10
bits or fewer passes a portfolio figure it moves beyond a few float16 ones, the 10-bit formats
failing by one to ten times the rule and FP8 by tenths to a hundred
([precision.md §15.6](details/precision.md#156-the-evidence-table-2026-10-11)). So the rule is an
acceptance bar, not the research's equal-accuracy measure, which 3.17 reads off the same
measurements. A standard error by blocks of paths was measured for the paired correction and
does not fix it. Independent scrambles now replace IID-path errors (2026-10-16), but
rare-event interval coverage remains I-111 (3.9), explicitly flagged on results;
[implementation and measurements](details/precision-accuracy.md).

**Precision scope (owner, 2026-10-15).** Fix the existing formats' rounding bias, option
bounds and error estimates first (3.9), then measure their time versus accuracy (3.17).
PFE and VaR in that first comparison use the existing, uncorrected quantile estimators;
ES uses the existing tail estimator. The paired mean correction remains available, with
corrected and uncorrected means distinguished and its float64 work included in timing.
A format that fails to reach the required accuracy is a valid research result. Multilevel
quantile correction is a separate experiment only if the measurements justify its cost
(7.9); FP4 is lower-priority capability work (7.10). These take precedence over the earlier
grouping of quantile correction with FP4 in F-07's precision design. Neither extension
blocks the API, TraderX or Basel work.

| Step | Goal | Work | Closes | After | Size |
|---|---|---|---|---|---|
| 3.9 | Precision | **Remaining: calibrated uncertainty for rare events.** Keyed stochastic compute rounding, nearest/stochastic campaigns, long-option bounds, independent Sobol replicates, per-replicate float64 pairing and historical moving-block errors are implemented (2026-10-16; I-98 and I-113 closed). The replicated-storage experiment still fails interval-coverage/RMS criteria on rare ENE and correction events; increasing the scramble count alone did not fix selected cases. Establish calibrated intervals or a principled availability rule, preserving the predeclared criteria and explicit warnings. [Implementation and remaining gate](details/precision-accuracy.md). FP4 and multilevel quantile correction remain deferred to 7.10 and 7.9 | [I-111](known-issues.md#i-111) (partial) | — | L |
| 3.10 | Critical bug | *Parallel, can start now* (it moves no path kernel). Sensitivities against ORE's sensitivity analytic on the shared portfolio; the AD Greeks against the bump Greeks on the same sloped portfolio, until AD is ORE's Greek or is labelled as the derivative of a different quantity, with the gap stated; a calibrated Bermudan's/American's LGM recalibrated inside the revaluation's price function under each market-risk scenario, as ORE's historical P&L recalibrates it, against ORE so recalibrating (the one calibration inside the price function serves the AD Greeks' fixed-calibration gap too) | [I-51](known-issues.md#i-51), [I-78](known-issues.md#i-78), [I-112](known-issues.md#i-112) | — | M |
| 3.11 | API | *Parallel, can start now.* Every input checked before any work, on the host: a curve's pillars (non-empty, as many rates as times, finite, strictly increasing; negative rates allowed), an option's exercise leaving a whole period of each leg, the standalone calibration's notional positive, every API model strict (`extra="forbid"`); each refusal names the field or trade. 3.12's preflight runs them all | [I-103](known-issues.md#i-103) | — | S |
| 3.12 | API | One run request in place of a route per engine function, as ORE runs a portfolio with a list of analytics: `POST /runs` with `market`, `portfolio`, `analytics` (`npv`, `exposure`, `sensitivities`, `market_risk`, `calibration`) and a `config` whose sections follow ORE's files, each setting in one place; one result with the configuration as run, which says what each figure is without the request beside it (reporting currency, dates, trade ids on every axis, each Greek array's factors, the Greeks method and calibration policy, the engine's revision and its jax, jaxlib and ORE versions); `GET /config/defaults`; an idempotency key; a Python entry, `run(RunRequest)`, in `run/`. Each analytic refuses a precision setting or configuration section it does not read, naming the field. The four routes and `/jobs` go, there being no client. The completeness test also checks each field's wiring: every field set to a non-default value arrives in the engine's configuration. A run of one analytic equals today's direct call within the margin ([design](details/configurable-engine.md#one-run-request-i-87-i-88-i-89)) | [I-87](known-issues.md#i-87), [I-88](known-issues.md#i-88), [I-89](known-issues.md#i-89), [I-95](known-issues.md#i-95) | 3.11 | M |
| 3.13 | API | One LGM calibration: the standalone bootstrap (`calibrate_lgm_sigma`, its basket and Bachelier helper prices) retires once 3.12's `calibration` analytic returns what `POST /calibration/lgm` returned, or moves to `tests/support/` as an independent check ([design](details/package-layout.md#3-merges)) | [I-94](known-issues.md#i-94) | 3.12 | S |
| 3.14 | API | The job queue: a job status that says when the engine worker cannot start, and a failed start answered with the queued job's id; a cancel route (a pending job at once, a running one at the worker's next phase boundary); a retention policy, with a way to keep a run indefinitely (Basel P0.9 keeps regulatory runs with it); the schema migration in one transaction; the worker's lock per host, not per queue file | [I-76](known-issues.md#i-76), [I-77](known-issues.md#i-77), [I-91](known-issues.md#i-91), [I-106](known-issues.md#i-106) | 3.12 | M |
| 3.15 | Precision | One worker per host on a Cloud TPU pod slice, with process 0 claiming each job and handing it to the other hosts, since under SPMD every host runs the same job ([details/precision.md §11.3](details/precision.md#113-multi-device-and-multi-host)); a failed host fails the whole job, which is retried whole. Measure wall time against device count on TPU and H100; a run on fewer devices than it was given (a path count with no fitting divisor) says so. The one-host split of the scenario axis is done (2026-10-04) | [I-61](known-issues.md#i-61) | 3.14 (the same worker loop) | M |
| 3.16 | Precision | *Parallel, can start now* (it moves no number). Native FP8 and bfloat16 passes on a GPU: XLA's Triton GEMM computes FP8 products 40–70% wrong on the RTX 5060 while cuBLASLt computes them exactly, and XLA ignores the bfloat16-passes algorithm on a matrix-vector product (2026-10-11). Route FP8 products to cuBLASLt in the engine worker, timed against Triton on every other product; run product formats' matrix-vector products in a shape XLA honours or emulate them; pair e5m2 with e4m3 where the units need it; the device check on H100 and a newer jax; minimal reproductions reported to XLA. Each format turns native per device in `product_execution`'s table once the device check passes | [I-97](known-issues.md#i-97) | — | M |
| 3.17 | Precision | The first research result on existing precision formats: on several devices of Ironwood and H100, wall time per figure at equal accuracy, many low-precision paths against fewer float64 paths, read from the precision harness's relative errors (equal accuracy must use observed RQMC error scaling and covariance; the evidence table's 1% path ceilings are an acceptance bar that little below float32 meets, 2026-10-11); the storage formats, then the matrix-product kernels (2026-10-10's and the rollback's of 2026-10-13, compute-bound, where a shared operand's second term costs most) on native FP8, the native product formats' code paths confirmed on each device (`engine.precision.product_execution`'s table, TensorFloat-32 and bfloat16 passes by their algorithm, FP8 by its dtype), each format's two terms against its single pass (A-23; no cost measurable on an RTX 5060's float16, bfloat16 and TensorFloat-32 units for the narrow products, 2026-10-11, and 1.37 to 1.62 times the single pass for the compute-bound rollback, 2026-10-13). Each time is the whole job's (host preparation, the fixed float64 stages, the paired sample, transfers and reductions, the device synchronized before the clock stops), cold and warm compiles stated apart, with equal outputs on both sides; measure the induction budget/time/peak tradeoff per device (precision design §17); a format that does not pay is a result too | [F-07](features.md#f-07) (speed), [I-114](known-issues.md#i-114) (engine/grid evidence matching before timing) | 3.9, 3.15, 3.16 | M |

Order within the stage:

- **The High defects first** (both done 2026-10-11). Market risk in one unit (I-100) moved no
  precision campaign, all in one currency; every later market-risk figure has a currency. A
  short option (I-101) is the long one's kernel and a sign, so every evidence row of a long
  option holds for the short.
- **Existing formats before precision extensions.** The storage harness is built
  (2026-10-09, I-75) and the kernels it measures are rewritten (2026-10-10; the rollback
  2026-10-13). Their first measurements found a bias floor of compute below float32 (I-96), fixed
  on 2026-10-11 by carrying each product's shared operand in two terms (A-23), so swaps,
  Europeans and bonds have evidence rows, and since 2026-10-12 the Bermudan/American kernel is
  ORE's on the paths (its shift horizon, I-32, and its model kept as ORE built it, I-49, the
  options' rows measured again after each). Stochastic operand rounding is now available
  (I-98, closed); nonlinear precision bias and variance still limit FP8's measured path
  ceilings. The evidence table is built on them (I-55, 2026-10-11),
  every result warning of figures without a passing row. The rollback family was then rewritten
  once, after them (2026-10-13), its rows measured and each option's flipped exercise decisions
  with them; the exposure reductions now read stored columns without a float64 cube
  (I-105, closed 2026-10-15; [measurements](details/precision.md#17-exposure-storage-and-peak-memory-i-105-2026-10-15)).
  Keyed stochastic rounding and option bounds are implemented (2026-10-16). Independent
  replicate errors are implemented too, but rare-event interval calibration remains
  in 3.9; its failed acceptance gate stays open. The first research
  result then measures the existing formats, after the API steps and multi-host (3.15), on
  hardware that has to be booked (3.17). Quantile correction (7.9) and FP4 (7.10) follow
  separately, without delaying that measurement.
- **The Bermudan/American kernel is ORE's on the paths** (2026-10-12). Its values were
  0.5–4% from ORE's, all of it a per-path recalibration that ORE's exposure simulation never
  makes (I-49): ORE keeps the LGM it calibrated when it built the trade. So the rollback family
  (rewritten 2026-10-13), its evidence rows and Basel's IMM exposure build on ORE's numbers; the
  per-path recalibration is gone, since no ORE analytic
  recalibrates on a path (decision A-24). A calibration that cannot fit its basket fails
  explicitly, as ORE's does (I-73, A-24), so no evidence row rests on a misfit model.
- **3.10 alongside.** The Greeks touch no path kernel, so they can be fixed at any time; Basel's
  FRTB-SA sensitivities rest on them.
- **The API before TraderX and Basel.** 3.12 changes the HTTP contract and the way into the
  engine, so it comes before the analytics Basel (stage 5) and stage 7 add, which then arrive
  as entries of the `analytics` list rather than as routes, and before TraderX's submissions
  are moved onto it (4.2). 3.11's input checks go first, so the run request's preflight is
  written once on them. A setting added before 3.12 gets its API field in the request shape of
  the day, and 3.12 carries it over. 3.14 comes before 3.15, which is written on the worker
  loop 3.14 changes.
- **Numbers.** 3.11 to 3.15 move no number beyond the margin (3.11 refuses inputs that gave
  wrong ones). 3.10 may move the AD Greeks and a calibrated Bermudan's market-risk P&L to ORE's,
  planned changes of the model. On fixed paths, 3.9's arithmetic changes keep float64 within
  the margin. I-111's independent replicates intentionally change the sampling design:
  compare precisions on matched replicate draws, validate distributions and interval coverage,
  and document and re-baseline the changed sample rather than treating different draws as
  arithmetic drift. ORE parity tolerances stay unchanged.

Parity methodology: [details/ore-parity-validation.md](details/ore-parity-validation.md);
precision: [details/precision.md](details/precision.md); layout:
[details/package-layout.md](details/package-layout.md).

<a id="stage-4--traderx"></a>
## Stage 4 — TraderX integration

One way into the engine for TraderX's submissions, and the defects of its path. After the
focus, before Basel.

| Step | Work | Closes | After | Size |
|---|---|---|---|---|
| 4.1 | *Parallel, can start now* (the adapter's loader and attempt store, which stay). TraderX's contract checked at the door: `bundleId` recomputed from the canonical manifest and a mismatch refused; every parsed number finite; each joined row's booked rate, currency, maturity and direction reconciled with its terms; repeated position and contract identities refused; the manifest's structure checked before use (`422` naming the field); outcomes that cannot be overwritten by their payload, publication with strict JSON. A publication that fails leaves a state a same-id retry completes, every staging error a `PublicationError` | [I-107](known-issues.md#i-107), [I-108](known-issues.md#i-108), [I-109](known-issues.md#i-109) | — | M |
| 4.2 | One way into the engine for TraderX: its bundle becomes a run request, after its convention checks and refusals (so a refused booking still never reaches a pricer, I-05), and the engine prices its bills and notes; the TraderX path's own pricers go. Its routes, result document, attempt store and accrued-interest reconciliation stay. First, the engine's bond and the TraderX note agree on every bundle row of the tests to rounding ([design](details/package-layout.md#3-merges)); the adapter's orchestration moves above `run/` in the import layers, its terms and provenance types staying low | [I-93](known-issues.md#i-93) | 3.12 | M |
| 4.3 | TraderX's submissions: one execution owner per workload (a retry under the same id has had one since 2026-10-10); refuse unknown calculations and a reporting currency other than USD (A-18, as ORE rejects an unknown analytic) | [I-58](known-issues.md#i-58), [I-59](known-issues.md#i-59) | 4.2 | S |
| 4.4 | TraderX's attempts in the job queue: the accepted-attempt record, boot sweep and `interrupted` state that portfolio jobs have had since 2026-10-04; bindings and terminal states in the queue's transactions, so one owner holds across processes; a workload's latest failure served when it has no success; stale stages swept | [I-08](known-issues.md#i-08) | 3.14, 4.2 | S |
| 4.5 | Before the API is reachable from outside a trusted network: authentication chosen with TraderX, a body-size limit, a per-caller limit on queued jobs, bundle paths confined to configured roots and each artifact to its bundle | [I-90](known-issues.md#i-90) | — | S |
| 4.6 | Chase TraderX's answers; apply them (a widened allowlist, a schema statement). When they arrive | [I-23](known-issues.md#i-23), [I-60](known-issues.md#i-60) | Their answers | S |

4.1 goes first because 4.2's row-by-row agreement is only as good as the rows it is given, and
because I-107 and I-109 are the path's remaining ways to serve a stale result or hang a retry.
4.4 makes the TraderX path's ownership the job queue's, so the two do not keep separate
recovery rules.

Integration notes and the open asks: [details/traderx-integration.md](details/traderx-integration.md).

<a id="stage-5--basel-iii"></a>
## Stage 5 — Basel III

Regulatory figures as the Basel Framework specifies them, each traceable and reproduced by an
oracle, after the focus and TraderX. Each figure is an entry of 3.12's `analytics`.

| Step | Work | Closes | After | Size |
|---|---|---|---|---|
| 5.1 | Basel P0 (decisions D-1 to D-10, pinned text, the BCBS profile, the requirement catalogue and traceability check, measure guards, desk and book on every row, the run manifest, the remaining ORE oracles; P0.9 keeps regulatory runs by 3.14's retention policy) and P2's data acquisition, which is calendar time and so is started first | [F-05](features.md#f-05) (P0, P2) | Stage 4 | L |
| 5.2 | Basel P1 (FRTB-SA) onward, phase by phase of the [Basel plan](details/basel-iii.md#6-phases-and-tasks): historical scenarios and IMA expected shortfall, backtesting and PLA, SA-CCR and BA-CVA, the precision gate, the evidence pack | [F-05](features.md#f-05) | 5.1's P0, 3.10 (USD swaps also I-05; IMA's ES with calibrated Bermudans also I-112); P7's evidence pack 6.4 | L |

5.2 follows the Basel plan's phases, each waiting only for what its row lists: FRTB-SA's
sensitivities on 3.10 (shown to be ORE's); P5's IMM needed a Bermudan's exposure equal to ORE's,
which it is since 2026-10-12 (I-49). P6's precision gate is the evidence table (built 2026-10-11, I-55), one table shared
with the precision research: P6 adds the regulatory figures' rows, and moves MAR32's
thresholds into the regulatory profile once P0.3 has entered them a second time. The
precision research uses Basel's criteria (A-11) as an engineering acceptance standard; a row
that passes them is not a regulatory approval, and the evidence table says so. Basel III:
[details/basel-iii.md](details/basel-iii.md).

<a id="stage-6--fix-what-exists"></a>
## Stage 6 — Fix what exists

The remaining defects of what the engine already does: numbers wrong only on inputs nothing
uses yet, documentation, speed, tests and tooling. In order: unchecked behaviour, then speed,
then tooling; the two parallel steps at any time.

| Step | Work | Closes | After | Size |
|---|---|---|---|---|
| 6.1 | *Parallel, start now* (A-20). Ruff in `pyproject.toml` and CI with pyflakes' rules only (ruff's `F` set: unused imports and variables, undefined names, empty f-strings; no style or formatting rules). Fix or mark each finding; an "unused" import that other modules import from there is a re-export and goes in `__all__`, checked one by one. Earlier is cheaper: every later step's code is then written lint-clean once. `pandas`, which nothing imports, leaves the dependencies | [I-66](known-issues.md#i-66) (linter) | — | S |
| 6.2 | *Parallel, start now.* The documentation's contradictions corrected (ORE described as single-threaded, three independent modules, the precision demo's count, the whole job as one XLA computation, parity tests never slow); one table of products by analytics stating what equals ORE, to what tolerance and by which test, linked from the README | [I-110](known-issues.md#i-110) | — | S |
| 6.3 | The narrow cases: a European's liveness by its expiry date, not its variance; a bond valued until its last payment; Theta's add-back on the rolled market's fixings, `theta_days` bounded; the LGM's H by `expm1`, host and JAX alike | [I-104](known-issues.md#i-104) | — | S |
| 6.4 | Characterize the full-suite aborts and lost worker processes. The fast tier's CI deaths were memory and are fixed (2026-10-05: the grid-convergence test's grid, compiled programs dropped per module). Still to do: show the full suite on CI's 16 GB runner (`-n 4` peaked at 24.2 GB with page cache before those fixes), then repeated full runs against a known-bad baseline, then the full tier on a schedule. Before Basel P7 (5.2), whose evidence pack rests on a complete suite run | [I-27](known-issues.md#i-27) | — | M |
| 6.5 | The cold job's one-operation programs: jit the scenario market's construction and the other eager sites, and market risk's batch on the price function as data, as was done for the AD Greeks; the snapshot within the margin | [I-81](known-issues.md#i-81) | — | S |
| 6.6 | The pipeline's remaining speed: find why, on Windows' CPU only, pricing runs 10–25% slower in a process that has run AD Greeks (the Windows heap is suspected, not shown; Linux is unaffected); and win back the 3 s (8%) a cold job lost to the bootstrap's compile under Newton, by tracing the helper's price once per bucket program where it is traced at the bracket's ends, the step and the model value today | [I-53](known-issues.md#i-53) | — | S |
| 6.7 | Shared test helpers in `tests/support/`; public-entry tests where stage 1 made private-symbol tests obsolete | [I-67](known-issues.md#i-67) | — | S |
| 6.8 | A type checker on `engine/` (A-20) | [I-66](known-issues.md#i-66) (type checker) | 6.1 | S |

6.4's runs can be made whenever a full run is being made anyway. Five full runs under WSL2
(2026-10-06, `-n 4`, a 23 GB VM; four on the GPU, one on CPU) all printed their summary lines,
the last of each green; see the verification status. Only the conclusion waits for repeated
runs.

<a id="stage-7--new-capabilities"></a>
## Stage 7 — New capabilities

What the engine does not do yet, beyond the focus, TraderX and Basel. A step that adds a
path-pricing kernel (7.2, 7.3, 7.4, 7.6) writes it in difference form, one implementation for
every precision (A-16, [details/precision.md §8](details/precision.md#8-compute-below-float32)),
and gets its rows in the evidence table (a campaign of the precision harness, I-55's table).

| Step | Work | Closes | After | Size |
|---|---|---|---|---|
| 7.1 | Swaption smile: a strike axis in the market's volatilities, read at each option's and helper's strike; an additive market field, with ATM-only markets unchanged within the margin. The lookup it widens is written in difference form | [I-54](known-issues.md#i-54) | — | M |
| 7.2 | Engine options: ORE's `AnalyticLgm` European engine, settlement methods, FD solver (the AD Greeks method and the market-risk engine by configuration are done) | [F-01](features.md#f-01) | — | M |
| 7.3 | FX and equity trades on the pipeline; FX/EQ calibration; the two-currency end-to-end test (L6) against the ORE simulation oracle, extended to a second currency, and market risk's FX factors | [F-04](features.md#f-04) | — | L |
| 7.4 | SABR volatility | [F-02](features.md#f-02) | 7.1 | M |
| 7.5 | CVA/DVA | [F-06](features.md#f-06) | 3.12 | M |
| 7.6 | AMC engine | [F-03](features.md#f-03) | — | L |
| 7.7 | Reporting currencies other than USD for TraderX: convert at the as-of FX spot, as ORE reports in its `baseCurrency`; parity against an ORE run with that base currency | [F-08](features.md#f-08) | 4.3; an FX source ([I-18](known-issues.md#i-18), waiting on others) | S |
| 7.8 | ORE's XML input files (`ore.xml`'s analytics, `simulation.xml`, `pricingengine.xml`, `sensitivity.xml`, `portfolio.xml`, zero-rate market data) translated into a run request, refusing by name what the engine does not support. Nice to have, not critical (owner, 2026-10-08) | [F-09](features.md#f-09) | 3.12 | M |
| 7.9 | **Conditional research experiment: multilevel quantile correction.** Only if the existing-format timing/accuracy results show a useful opportunity, combine many low-precision paths with paired float64 paths to estimate PFE/VaR through a corrected distribution or equivalent estimator, with a related tail-estimation method for ES. Compare against the uncorrected estimators and float64 at equal total wall time, including paired simulation, correction and inversion costs; validate tail accuracy and uncertainty. Retain raw results alongside corrected ones. Independent of FP4; do not implement merely to make a failing format pass | [F-07](features.md#f-07) (quantile/tail estimation) | 3.9, 3.17; evidence of likely benefit | L |
| 7.10 | **Deferred: FP4 storage and compute.** Storage with a variance correction through block scales, emulated compute using the existing precision architecture and rounding lessons from I-98, and its own precision campaigns and evidence rows; native compute on TPU 8t/8i measured when available. Lower priority than the supported formats' first time/accuracy comparison; quantile correction is not a prerequisite | [F-07](features.md#f-07) (FP4) | 3.9, 3.17 | L |

Order within the stage: the smile first (no market given to the engine so far has one, so it
changes no number yet, but SABR builds on it); then the smallest gaps in ORE's own choices
(7.2), the largest scope gap (7.3: the model already simulates FX and equity, but nothing
prices them), SABR (7.4), and the analytics built on proven exposure (7.5, 7.6). 7.7 as soon
as its FX source arrives; 7.8 has no dependants. The deferred precision extensions (7.9, 7.10)
come after the first existing-format research result: quantile correction only when its
measured opportunity warrants an experiment, and FP4 separately at lower priority. Neither
is required to complete stage 3.

<a id="waiting-on-others"></a>
## Waiting on others

Not engineering work until the input arrives. Chase the dependency, not the code.

| Item | Waiting for | From |
|---|---|---|
| [I-05](known-issues.md#i-05) USD-SOFR swaps | The D03/D04 convention set | TraderX |
| [I-04](known-issues.md#i-04) seasoned TraderX swaps | `pastFixings` in the export | TraderX |
| [I-16](known-issues.md#i-16) per-pillar `rateSensitivity` | An observed market-data package (W2) | TraderX |
| [I-18](known-issues.md#i-18) equity positions; [F-08](features.md#f-08) reporting currencies (step 7.7) | A spot/FX source | TraderX or a market-data decision |
| [I-23](known-issues.md#i-23), [I-60](known-issues.md#i-60) | Answers on versioning and added fields | TraderX (then step 4.6) |
| [I-07](known-issues.md#i-07) corporate bonds, listed options | Demand, and for corporates a credit model | Owner |
| The project's licence (no `LICENSE` file or licence metadata; project audit, AUD-49) | A decision on first-party licensing | Owner |

Open TraderX asks are collected in
[details/traderx-integration.md](details/traderx-integration.md#open-with-traderx).
