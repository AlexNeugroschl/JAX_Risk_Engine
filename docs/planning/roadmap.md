# Roadmap

The order in which to work through [known-issues.md](known-issues.md) and
[features.md](features.md). Every open item appears here exactly once; an item that is not
here is not planned ([rules](README.md#lifecycle)).

**How the order is decided** (decision A-19, revised 2026-10-10).

1. **The near-term milestone**: `demos/demo_profile_small.py` runs on the owner's local GPU
   and its profiler trace covers the whole job (stage 2, done 2026-10-07).
2. **The focus** (stage 3), three goals, in this order of priority where they compete:
   1. **Low precision, done**: every path kernel in its final form at every precision down to
      FP8 and FP4, each format's error measured and stated on every result (the evidence
      table), then timed on TPU and H100. The precision research is the project's purpose
      (root [README](../../README.md)).
   2. **Critical bugs**: a High defect (a wrong number on an ordinary input, or another
      caller's result served), and the numbers that are wrong or unproven where the precision
      work and every later stage build on them (Bermudans and Americans against ORE, the
      Greeks).
   3. **A robust API on every feature**: one run request for every analytic, over HTTP and in
      Python; every setting reaching the engine, a setting the run does not read refused by
      name, a retried submission run once, a job that can be cancelled and a worker failure
      that is reported.
3. **TraderX integration** (stage 4): its submissions through the run request into the
   engine's pricers, and the defects of its path.
4. **Basel III** (stage 5): regulatory figures as the Basel Framework specifies them, each
   traceable and reproduced by an oracle. It waits for the focus and for TraderX; the
   precision research's acceptance standard (A-11) already uses its tests, transcribed.
5. **Fix what exists** (stage 6): the remaining speed, test and tooling defects.
6. **New capabilities** (stage 7): what the engine does not do yet.

A clear `engine/` (packages named for what they hold, nothing at the root but `__init__.py`,
[details/package-layout.md](details/package-layout.md)) was done first (2026-10-08), so that
every later step is written in the final layout.

Within a stage, work in the order of its table. Items marked *parallel* touch code that no
earlier step changes, and can start at any time. Items waiting on someone outside the project
are listed [separately](#waiting-on-others), with who to chase. A defect found along the way
goes to stage 3 if it is High or blocks a step there, ahead of the work it affects; otherwise
to stage 6, or before the step it blocks.

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
  model (3.4 and 3.5 move default Bermudan/American values towards ORE's) or a bug; a planned
  one states its largest change per array and re-baselines. Nothing is held to bit-for-bit
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
  reductions and the other fixed stages (A-10) stay float64.
- **Basel III.** Nothing a regulatory figure relies on is approximated silently: an input
  outside scope is refused by name, as today. Results stay auditable: the job queue keeps
  them, and 3.11's retention policy can keep a run indefinitely, which regulatory runs will be.
- **API compatibility.** Once the HTTP API has a client, changes are additive: every route,
  field, default and result shape keeps working, and a change of the contract is a revision of
  it (A-2; the names A-2 retired were removed outright on 2026-10-07, there being no client
  yet, by the owner's revision of that day). 3.9 replaces the routes with one run request on
  the same terms, before any client exists. A new setting arrives with its API field, which
  `tests/test_api_completeness.py` checks. Two narrowings are planned, both deliberate: 4.2
  refuses inputs that are now accepted and then ignored (the old answer is wrong), and 3.4
  changes a default number to ORE's (decision A-3). TraderX's routes (`/eod/...`) are
  TraderX's contract and keep their shape when 4.1 changes what runs behind them.
- **Tests.** A step that breaks a test of a private symbol rewrites it against the public
  entry ([I-67](known-issues.md#i-67)'s rule), so restructuring does not wait for 6.5.

## Where things stand

One pipeline, ORE's: a cross-asset model with a model per currency (the LGM by default, or
Hull-White, both ORE's `<LGM>` parametrizations), every trade valued on every path as ORE's
valuation engine does. It matches ORE component by component: t=0 prices to 1e-14 (swaps,
Europeans, bonds) and 4e-11 (calibrated Bermudans/Americans), model analytics to 1e-12
(the Hull-White model's bonds against QuantLib's `HullWhite` too), pricers on path curves to
1e-8 – 1e-12 under either model, market-risk VaR/ES per scenario to 2e-13, and since
2026-10-07 an ORE simulation run in-process is the reference for the assembled pipeline:
swaps, Europeans and bonds equal it path by path (Bermudans and Americans 0.5–4% apart, I-49,
which 3.5 closes). Sensitivities are not yet shown to equal ORE's.

Every model, engine, Greeks and precision choice is one run configuration with ORE's
defaults. The simulation, the scenario market and path pricing each take a storage, a compute
and an accumulate precision (since 2026-10-10 compute down to FP8, TensorFloat-32 and bfloat16
passes in kernels in difference form, native or emulated by device, but for the
Bermudan/American rollback, still float32 or float64; storage down to float16, bfloat16 and FP8
with block scales, block offsets that keep a concentrated column's spread, and nearest or
stochastic rounding), set per product and per trade for pricing, in portfolio and market-risk
runs alike; every result carries a precision report, and a paired float64 sample corrects the
mean figures by a two-level estimator and measures the quantiles. Since 2026-10-09 a storage
harness measures every format's error against float64 per stage, figure, product and path
count, over independent seeds, rerunnable on any kernel change (I-75). HTTP jobs go through a durable SQLite job queue to one single-threaded engine
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
cold job's one-operation programs (6.3), a slower pricing on Windows after AD Greeks and a cold
job 3 s slower for Newton's compile (6.4), and the grid rollback's memory, which stops the
demo's job beyond a few thousand paths (3.6). The card's float64 runs at 1/64 of its float32
rate, so it is where correctness and the trace are checked, not where speed is measured (3.14).

<a id="stage-3--the-focus"></a>
## Stage 3 — The focus: low precision, critical bugs, a robust API

Low precision first where the three compete. The critical bugs are here because the precision
work and the later stages build on the numbers they fix (3.4, 3.5, 3.8). The stage's one High
defect of engineering work is fixed (2026-10-10, I-57): a TraderX submission id binds to one
workload before any cached result is returned, so one caller is never served another's result.

| Step | Goal | Work | Closes | After | Size |
|---|---|---|---|---|---|
| 3.2 | Precision | Compute below float32's bias floor: each product's path-independent operand is rounded once for every path, so every figure computed below float32 is biased (the `compute` campaign, 2026-10-10). Carry the operand in two terms of the format by default, the single pass kept as a product format of its own (decided, A-23); rerun the `compute` campaigns with both, and where two terms leave a significant bias (FP8 e4m3), measure a third term or a wider shared operand. Before the evidence table, whose rows it changes | [I-96](known-issues.md#i-96) | — | S |
| 3.3 | Precision | The evidence table against the acceptance standard (A-11: Basel III's P&L attribution test and the Basel plan's P6.2 rule, as transcribed; P6.2 labelled as engineering for figures Basel does not cover), from the precision harness on the kernels rewritten in difference form (the simulation, scenario market, legs and Europeans, 2026-10-10: the `compute` and `compute_market_risk` campaigns beside the storage ones) and on 3.6's as it lands, then on each kernel stage 7 adds. With what the harness's first campaigns found (I-55): a robust standard error for the paired correction where the shocks or curves are stored below 32 bits (by blocks of paths or across seeds), an absolute rule for figures float64 holds fixed, and each row's path ceiling; a warning on any result whose combination has no passing row | [I-55](known-issues.md#i-55) (warnings) | 3.2; 3.6 for the Bermudan/American rows | M |
| 3.4 | Critical bug, precision | `ShiftHorizon`: the setting and its API field exist and refuse anything but 0; implement the shift, parity at 0.5 against the LGM oracle, then 0.5 as the default | [I-32](known-issues.md#i-32) | — | M |
| 3.5 | Critical bug, precision | Reproduce ORE's two per-path recalibration details against `tests/test_ore_xva_parity.py`, until its Bermudan/American L3 cases (strict expected failures today, 0.5–4% apart) pass; confirm an American's basket on a path against ORE's (decision A-7); warn, as ORE's `LgmBuilder` does, when a path's recalibration misses its basket | [I-49](known-issues.md#i-49), [I-73](known-issues.md#i-73) | — | M |
| 3.6 | Precision | The Bermudan/American rollback and recalibration in difference form, as the other path kernels were rewritten (2026-10-10, [details/precision.md §8](details/precision.md#8-compute-below-float32)), the rollback as a matrix product; then its pricing stage takes any compute format, as the others' do (`RESTRICTED_PRODUCTS` empties). The matrix form holds one operator per step instead of one per path and step, so the demo's job runs at 65,536 and 262,144 paths | [F-07](features.md#f-07) (compute), [I-83](known-issues.md#i-83) | 3.4, 3.5 | L |
| 3.7 | Precision | FP4: storage with a variance correction through the block scales, compute emulated as FP8's is (2026-10-10); multilevel estimation of quantiles (PFE, VaR, ES); its rows in the evidence table. Native FP4 compute on TPU 8t/8i is timed in 3.14 | [F-07](features.md#f-07) (FP4) | 3.3, 3.6 | L |
| 3.8 | Critical bug | *Parallel, can start now* (it moves no path kernel). Sensitivities against ORE's sensitivity analytic on the shared portfolio; the AD Greeks against the bump Greeks on the same sloped portfolio, until AD is ORE's Greek or is labelled as the derivative of a different quantity, with the gap stated | [I-51](known-issues.md#i-51), [I-78](known-issues.md#i-78) | — | M |
| 3.9 | API | One run request in place of a route per engine function, as ORE runs a portfolio with a list of analytics: `POST /runs` with `market`, `portfolio`, `analytics` (`npv`, `exposure`, `sensitivities`, `market_risk`, `calibration`) and a `config` whose sections follow ORE's files, each setting in one place; one result with the configuration as run; `GET /config/defaults`; an idempotency key; a Python entry, `run(RunRequest)`, in `run/`. Each analytic refuses a precision setting or configuration section it does not read, naming the field. The four routes and `/jobs` go, there being no client. The completeness test also checks each field's wiring: every field set to a non-default value arrives in the engine's configuration. A run of one analytic equals today's direct call within the margin ([design](details/configurable-engine.md#one-run-request-i-87-i-88-i-89)) | [I-87](known-issues.md#i-87), [I-88](known-issues.md#i-88), [I-89](known-issues.md#i-89), [I-95](known-issues.md#i-95) | — | M |
| 3.10 | API | One LGM calibration: the standalone bootstrap (`calibrate_lgm_sigma`, its basket and Bachelier helper prices) retires once 3.9's `calibration` analytic returns what `POST /calibration/lgm` returned, or moves to `tests/support/` as an independent check ([design](details/package-layout.md#3-merges)) | [I-94](known-issues.md#i-94) | 3.9 | S |
| 3.11 | API | The job queue: a job status that says when the engine worker cannot start; a cancel route (a pending job at once, a running one at the worker's next phase boundary); a retention policy, with a way to keep a run indefinitely (Basel P0.9 keeps regulatory runs with it) | [I-76](known-issues.md#i-76), [I-77](known-issues.md#i-77), [I-91](known-issues.md#i-91) | 3.9 | M |
| 3.12 | Precision | One worker per host on a Cloud TPU pod slice, with process 0 claiming each job and handing it to the other hosts, since under SPMD every host runs the same job ([details/precision.md §11.3](details/precision.md#113-multi-device-and-multi-host)). Measure wall time against device count on TPU and H100. The one-host split of the scenario axis is done (2026-10-04) | [I-61](known-issues.md#i-61) | 3.11 (the same worker loop) | M |
| 3.13 | Precision | *Parallel, can start now* (it moves no number). Native FP8 and bfloat16 passes on a GPU: XLA's Triton GEMM computes FP8 products 40–70% wrong on the RTX 5060 while cuBLASLt computes them exactly, and XLA ignores the bfloat16-passes algorithm on a matrix-vector product (2026-10-11). Route FP8 products to cuBLASLt in the engine worker, timed against Triton on every other product; run product formats' matrix-vector products in a shape XLA honours or emulate them; pair e5m2 with e4m3 where the units need it; the device check on H100 and a newer jax; minimal reproductions reported to XLA. Each format turns native per device in `product_execution`'s table once the device check passes | [I-97](known-issues.md#i-97) | — | M |
| 3.14 | Precision | The research result: on several devices of Ironwood and H100, wall time per figure at equal accuracy, many low-precision paths against fewer float64 paths, read from the evidence table's path ceilings; the storage formats, then the matrix-product kernels (2026-10-10's and 3.6's) on native FP8, the native product formats' code paths confirmed on each device (`engine.precision.product_execution`'s table, TensorFloat-32 and bfloat16 passes by their algorithm, FP8 by its dtype), and FP4 compute on TPU 8t/8i | [F-07](features.md#f-07) (speed) | 3.3, 3.7, 3.12, 3.13 | M |

Order within the stage:

- **Low precision in full before anything else competes.** The storage harness is built
  (2026-10-09, I-75) and the kernels it measures are rewritten (2026-10-10) but for the
  rollback. Their first measurements found a bias floor of compute below float32 (I-96), which
  3.2 settles before 3.3 reads the table off them, so swaps, Europeans and bonds have evidence
  rows while 3.4 and 3.5 make the Bermudan/American kernel ORE's. The harness's first
  campaigns already bear on the table: a robust standard error for the paired correction where
  the shocks or curves are stored low, an absolute rule for figures float64 holds fixed, and a
  path ceiling for every format below float32 (I-55). The rollback family is then rewritten once, after them (3.6), its rows
  follow, and FP4 (3.7) closes the arithmetic. What remains of the precision research after
  the API steps, multi-host (3.12) and the timing (3.14), is measurement on hardware that has
  to be booked, not arithmetic.
- **3.4 and 3.5 are bugs and blockers both.** Default Bermudan/American values are 0.5–4% from
  ORE's on the paths; the rollback family, its evidence rows and Basel's IMM exposure all need
  them fixed first.
- **3.8 alongside.** The Greeks touch no path kernel, so they can be fixed at any time; Basel's
  FRTB-SA sensitivities rest on them.
- **The API before TraderX and Basel.** 3.9 changes the HTTP contract and the way into the
  engine, so it comes before the analytics Basel (stage 5) and stage 7 add, which then arrive
  as entries of the `analytics` list rather than as routes, and before TraderX's submissions
  are moved onto it (4.1). A setting added before it gets its API field in the request shape of
  the day, and 3.9 carries it over. 3.11 comes before 3.12, which is written on the worker loop
  3.11 changes.
- **Numbers.** 3.9 to 3.12 move no number beyond the margin. 3.4 and 3.5 move default
  Bermudan/American numbers towards ORE's, and 3.8 may move the AD Greeks to ORE's, planned
  changes of the model. 3.6 and 3.7 may move float64 at rounding level, within the margin, as
  the kernel rewrite of 2026-10-10 did (2.2e-14 of each array's scale).

Parity methodology: [details/ore-parity-validation.md](details/ore-parity-validation.md);
precision: [details/precision.md](details/precision.md); layout:
[details/package-layout.md](details/package-layout.md).

<a id="stage-4--traderx"></a>
## Stage 4 — TraderX integration

One way into the engine for TraderX's submissions, and the defects of its path. After the
focus, before Basel.

| Step | Work | Closes | After | Size |
|---|---|---|---|---|
| 4.1 | One way into the engine for TraderX: its bundle becomes a run request, after its convention checks and refusals (so a refused booking still never reaches a pricer, I-05), and the engine prices its bills and notes; the TraderX path's own pricers go. Its routes, result document, attempt store and accrued-interest reconciliation stay. First, the engine's bond and the TraderX note agree on every bundle row of the tests to rounding ([design](details/package-layout.md#3-merges)) | [I-93](known-issues.md#i-93) | 3.9 | M |
| 4.2 | TraderX's submissions: one execution owner per workload (a retry under the same id has had one since 2026-10-10); refuse unknown calculations and a reporting currency other than USD (A-18, as ORE rejects an unknown analytic) | [I-58](known-issues.md#i-58), [I-59](known-issues.md#i-59) | 4.1 | S |
| 4.3 | TraderX's attempts in the job queue: the accepted-attempt record, boot sweep and `interrupted` state that portfolio jobs have had since 2026-10-04 | [I-08](known-issues.md#i-08) | 3.11, 4.1 | S |
| 4.4 | Before the API is reachable from outside a trusted network: authentication chosen with TraderX, a body-size limit, a per-caller limit on queued jobs | [I-90](known-issues.md#i-90) | — | S |
| 4.5 | Chase TraderX's answers; apply them (a widened allowlist, a schema statement). When they arrive | [I-23](known-issues.md#i-23), [I-60](known-issues.md#i-60) | Their answers | S |

Integration notes and the open asks: [details/traderx-integration.md](details/traderx-integration.md).

<a id="stage-5--basel-iii"></a>
## Stage 5 — Basel III

Regulatory figures as the Basel Framework specifies them, each traceable and reproduced by an
oracle, after the focus and TraderX. Each figure is an entry of 3.9's `analytics`.

| Step | Work | Closes | After | Size |
|---|---|---|---|---|
| 5.1 | Basel P0 (decisions D-1 to D-10, pinned text, the BCBS profile, the requirement catalogue and traceability check, measure guards, desk and book on every row, the run manifest, the remaining ORE oracles; P0.9 keeps regulatory runs by 3.11's retention policy) and P2's data acquisition, which is calendar time and so is started first | [F-05](features.md#f-05) (P0, P2) | Stage 4 | L |
| 5.2 | Basel P1 (FRTB-SA) onward, phase by phase of the [Basel plan](details/basel-iii.md#6-phases-and-tasks): historical scenarios and IMA expected shortfall, backtesting and PLA, SA-CCR and BA-CVA, the precision gate, the evidence pack | [F-05](features.md#f-05) | 5.1's P0, 3.8 (USD swaps also I-05); P5's IMM 3.5; P6's precision gate 3.3; P7's evidence pack 6.2 | L |

5.2 follows the Basel plan's phases, each waiting only for what its row lists: FRTB-SA's
sensitivities on 3.8 (shown to be ORE's), P5's IMM on 3.5 (a Bermudan's exposure equal to
ORE's), P6's precision gate on 3.3, which is one table shared with the precision research.
Basel III: [details/basel-iii.md](details/basel-iii.md).

<a id="stage-6--fix-what-exists"></a>
## Stage 6 — Fix what exists

The remaining defects of what the engine already does, none of which gives a wrong number:
speed, tests and tooling. In order: unchecked behaviour, then speed, then tooling.

| Step | Work | Closes | After | Size |
|---|---|---|---|---|
| 6.1 | *Parallel, start now* (A-20). Ruff in `pyproject.toml` and CI with pyflakes' rules only (ruff's `F` set: unused imports and variables, undefined names, empty f-strings; no style or formatting rules). Fix or mark each finding; an "unused" import that other modules import from there is a re-export and goes in `__all__`, checked one by one. Earlier is cheaper: every later step's code is then written lint-clean once | [I-66](known-issues.md#i-66) (linter) | — | S |
| 6.2 | Characterize the full-suite aborts and lost worker processes. The fast tier's CI deaths were memory and are fixed (2026-10-05: the grid-convergence test's grid, compiled programs dropped per module). Still to do: show the full suite on CI's 16 GB runner (`-n 4` peaked at 24.2 GB with page cache before those fixes), then repeated full runs against a known-bad baseline. Before Basel P7 (5.2), whose evidence pack rests on a complete suite run | [I-27](known-issues.md#i-27) | — | M |
| 6.3 | The cold job's one-operation programs: jit the scenario market's construction and the other eager sites, and market risk's batch on the price function as data, as was done for the AD Greeks; the snapshot within the margin | [I-81](known-issues.md#i-81) | — | S |
| 6.4 | The pipeline's remaining speed: find why, on Windows' CPU only, pricing runs 10–25% slower in a process that has run AD Greeks (the Windows heap is suspected, not shown; Linux is unaffected); and win back the 3 s (8%) a cold job lost to the bootstrap's compile under Newton, by tracing the helper's price once per bucket program where it is traced at the bracket's ends, the step and the model value today | [I-53](known-issues.md#i-53) | — | S |
| 6.5 | Shared test helpers in `tests/support/`; public-entry tests where stage 1 made private-symbol tests obsolete | [I-67](known-issues.md#i-67) | — | S |
| 6.6 | A type checker on `engine/` (A-20) | [I-66](known-issues.md#i-66) (type checker) | 6.1 | S |

6.2's runs can be made whenever a full run is being made anyway. Five full runs under WSL2
(2026-10-06, `-n 4`, a 23 GB VM; four on the GPU, one on CPU) all printed their summary lines,
the last of each green; see the verification status. Only the conclusion waits for repeated
runs.

<a id="stage-7--new-capabilities"></a>
## Stage 7 — New capabilities

What the engine does not do yet, beyond the focus, TraderX and Basel. A step that adds a
path-pricing kernel (7.2, 7.3, 7.4, 7.6) writes it in difference form, one implementation for
every precision (A-16, [details/precision.md §8](details/precision.md#8-compute-below-float32)),
and gets its rows in 3.3's table.

| Step | Work | Closes | After | Size |
|---|---|---|---|---|
| 7.1 | Swaption smile: a strike axis in the market's volatilities, read at each option's and helper's strike; an additive market field, with ATM-only markets unchanged within the margin. The lookup it widens is written in difference form | [I-54](known-issues.md#i-54) | 3.6 | M |
| 7.2 | Engine options: ORE's `AnalyticLgm` European engine, settlement methods, FD solver (the AD Greeks method and the market-risk engine by configuration are done) | [F-01](features.md#f-01) | 3.6; the FD solver also 3.4 | M |
| 7.3 | FX and equity trades on the pipeline; FX/EQ calibration; the two-currency end-to-end test (L6) against the ORE simulation oracle, extended to a second currency | [F-04](features.md#f-04) | 3.6 | L |
| 7.4 | SABR volatility | [F-02](features.md#f-02) | 7.1 | M |
| 7.5 | CVA/DVA | [F-06](features.md#f-06) | 3.9 | M |
| 7.6 | AMC engine | [F-03](features.md#f-03) | 3.6 | L |
| 7.7 | Reporting currencies other than USD for TraderX: convert at the as-of FX spot, as ORE reports in its `baseCurrency`; parity against an ORE run with that base currency | [F-08](features.md#f-08) | 4.2; an FX source ([I-18](known-issues.md#i-18), waiting on others) | S |
| 7.8 | ORE's XML input files (`ore.xml`'s analytics, `simulation.xml`, `pricingengine.xml`, `sensitivity.xml`, `portfolio.xml`, zero-rate market data) translated into a run request, refusing by name what the engine does not support. Nice to have, not critical (owner, 2026-10-08) | [F-09](features.md#f-09) | 3.9 | M |

Order within the stage: the smile first (no market given to the engine so far has one, so it
changes no number yet, but SABR builds on it); then the smallest gaps in ORE's own choices
(7.2), the largest scope gap (7.3: the model already simulates FX and equity, but nothing
prices them), SABR (7.4), and the analytics built on proven exposure (7.5, 7.6). 7.7 as soon
as its FX source arrives; 7.8 last, since nothing waits on it.

<a id="waiting-on-others"></a>
## Waiting on others

Not engineering work until the input arrives. Chase the dependency, not the code.

| Item | Waiting for | From |
|---|---|---|
| [I-05](known-issues.md#i-05) USD-SOFR swaps | The D03/D04 convention set | TraderX |
| [I-04](known-issues.md#i-04) seasoned TraderX swaps | `pastFixings` in the export | TraderX |
| [I-16](known-issues.md#i-16) per-pillar `rateSensitivity` | An observed market-data package (W2) | TraderX |
| [I-18](known-issues.md#i-18) equity positions; [F-08](features.md#f-08) reporting currencies (step 7.7) | A spot/FX source | TraderX or a market-data decision |
| [I-23](known-issues.md#i-23), [I-60](known-issues.md#i-60) | Answers on versioning and added fields | TraderX (then step 4.5) |
| [I-07](known-issues.md#i-07) corporate bonds, listed options | Demand, and for corporates a credit model | Owner |

Open TraderX asks are collected in
[details/traderx-integration.md](details/traderx-integration.md#open-with-traderx).
