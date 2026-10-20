# Roadmap

The order in which to work through [known-issues.md](known-issues.md) and
[features.md](features.md). Every open item appears here exactly once; an item that is not
here is not planned ([rules](README.md#lifecycle)).

**How the order is decided** (decision A-19, revised by the owner on 2026-10-19).

1. **Accurate and fast on TPU and GPU clusters** (stage 1). This is the absolute priority: the
   engine's numbers are shown to be right on the target accelerators (TPU, H100), its products
   run natively wherever they are right, and its speed is measured across one host's devices and
   across a pod slice's hosts. Stage 1 holds only work that is critical to this goal.
2. **The precision research result** (stage 2): wall time per figure at equal accuracy on those
   clusters, which is the project's purpose (root [README](../../README.md)).
3. **Basel III** (stage 3): regulatory figures as the Basel Framework specifies them, each
   traceable and reproduced by an oracle.
4. **Fix what exists** (stage 4): the remaining defects in what the engine already does, then
   its speed, tests and tooling.
5. **New capabilities** (stage 5): what the engine does not do yet.

A defect found along the way goes into stage 1, ahead of the step it blocks, if it makes a run
on a TPU or GPU cluster wrong, or slow enough to matter for stage 1's goal. A High defect (a
wrong number on an ordinary input, or another caller's result served) also goes into stage 1.
Any other defect goes into stage 4, or before the step it blocks. Within a stage, work in the
order of its table. Items marked *parallel* touch code that no earlier step changes, and can
start at any time. Items waiting on someone outside the project are listed
[separately](#waiting-on-others), with who to chase.

**Step numbers belong to this page alone.** Steps are numbered in the order of work and are
renumbered whenever that order changes. Nothing else cites a step: code, error messages, tests
and the other documents cite the item (I-NN, F-NN), the decision (A-n) or a date.
`tests/test_import_layering.py` fails on a step number in code.

**Every step keeps the project's goals.** None of the rules below trades one goal for another:

- **ORE parity.** Every ORE parity suite passes at its existing tolerance after every step. A
  new capability comes with its own ORE parity test, or it is refused by name. Where the test
  needs a simulation, it runs through the ORE simulation oracle,
  `tests/support/ore_xva_oracle.py`. The [parity matrix](../reference/ore-parity.md#the-parity-matrix)
  gains the capability's cells.
- **Accuracy: correct, not bit for bit** (owner, 2026-10-10). A step may move numbers, but it
  may not make them wrong. After a step, every ORE parity suite and every test passes at its
  tolerance. The golden snapshot agrees with the one before within a small margin, relative to
  each array's largest magnitude (for a Greek, relative to its trade's NPV): float64 to 1e-10,
  float32 to 1e-4. Formats below float32 are judged by the storage harness's statistical
  thresholds, not by the snapshot. A change beyond the margin is either a bug or a planned
  change of the model or sampling design; a planned change states its largest change per array
  and re-baselines. On a TPU or a GPU, the margin is measured against the CPU's snapshot.
- **Speed and JAX.** A new or changed pricer is a module-level jitted function that takes the
  trade's data as a pytree argument
  ([profiling §3.7](../concepts/profiling.md#37-trade-data-as-traced-arguments-2026-10-02)),
  so nothing is compiled again per trade, date, bump or call. It runs under the scenario-axis
  sharding, and the compile-count tests cover it.
- **Precision research.** Every path kernel is written in difference form: its matrix products
  run in the stage's compute format and everything else in its accumulate format, with one
  implementation for every precision (A-16;
  [details/precision.md §8](details/precision.md#8-compute-below-float32)). Each kernel has rows
  in the precision harness (`tests/support/precision_harness.py`). Calibration, Greeks,
  reductions and the other fixed stages (A-10) stay float64. A change to a kernel's formula,
  interpolation, arithmetic or storage reruns its rows; old rows are not reused just because the
  format's name is unchanged. The evidence holds per pricing setting (I-114).
- **Basel III.** Nothing a regulatory figure relies on is approximated silently: an input
  outside scope is refused by name. Results stay auditable: the job queue keeps them for its
  retention period (30 days by default, A-27), and keeps a run submitted with `keep`
  indefinitely, as regulatory runs will be.
- **API compatibility.** Once the HTTP API has a client, changes are additive. Every route,
  field, default and result shape keeps working, and a change of the contract is a revision of
  it (A-2). A new setting arrives with its API field in a section of the run request's
  `config`, and `tests/test_api_completeness.py` checks that the field's value reaches the
  engine. A narrowing (an input that is now refused) is deliberate and stated, because the
  answer it gave before was wrong.
- **Tests.** A step that breaks a test of a private symbol rewrites that test against the
  public entry point ([I-67](known-issues.md#i-67)'s rule), so restructuring does not wait for
  I-67.

## Where things stand

The engine has one pipeline, ORE's. It uses a cross-asset model with one model per currency
(the LGM by default, or Hull-White) and values every trade on every path as ORE's valuation
engine does. It matches ORE component by component and on the assembled pipeline: today's
prices, the exposure simulation path by path, the bump Greeks against ORE's sensitivity
analytic, and market-risk VaR and ES. The [parity matrix](../reference/ore-parity.md#the-parity-matrix)
says what is compared with ORE, per product, analytic and setting. Every analytic is one run
request (`engine.run.run`, HTTP `POST /runs`, I-87). The request is checked whole before any
work, and its result says what each figure is. TraderX submits through that request like any
other caller, so its own `/eod` path is to be retired ([I-120](known-issues.md#i-120)).

Precision is set per stage, product and trade, down to FP8 for storage and compute. Every
result carries a precision report and warns of each figure whose format has no passing row in
the evidence table (`compliance/precision-evidence.md`). The timing harness for the research
(`tests/support/precision_timing.py`, precision design §19) runs end to end.

On hardware: one engine worker per host splits each job's scenarios across the host's devices.
With `--distributed`, one worker per host of a pod slice runs each job together
([I-61](known-issues.md#i-61)); this is built and tested on CPU processes only. The engine is
verified on CPU and on one consumer GPU (the owner's RTX 5060, Linux or WSL2), where FP8 is native
through cuBLAS ([I-97](known-issues.md#i-97)). It has never run on a TPU or an H100
([I-121](known-issues.md#i-121)), and no number or speed has been measured on either.

---

<a id="stage-1"></a>
## Stage 1 — Accurate and fast on TPU and GPU clusters

The engine is to show its numbers right on a TPU and an H100, run its products natively where
they are right, and have its speed against device and host count measured. Nothing else is in
this stage.

| Step | Work | Closes | After | Size |
|---|---|---|---|---|
| 1.1 | *Parallel, can start now.* Install on a TPU VM: a `tpu` extra in `pyproject.toml` at the pinned jax; `tests/test_environment.py` passing on a TPU VM against `constraints.txt`; a TPU section in the user guide beside the GPU one | [I-121](known-issues.md#i-121) (install) | — | S |
| 1.2 | The engine's numbers on a TPU and an H100. On each: the device check (`tests/support/precision_devices.py`; on a TPU, its bfloat16 and bfloat16 passes, and Ironwood's FP8); the full suite (split as [I-27](known-issues.md#i-27) requires); every ORE parity suite at its tolerance; the golden snapshot within the margin of the CPU's. This qualifies the TPU's emulated float64, which the fixed stages and every kernel's float64 level run on. Each difference found is a defect of stage 1, fixed before 1.4. A format shown right natively is marked native for that device | [I-121](known-issues.md#i-121) | 1.1; booked hardware | M |
| 1.3 | The H100's products: the device check under both GEMM emitters (XLA's Triton, and cuBLAS with `--xla_gpu_enable_triton_gemm=false`), repeated on each new jax; the two XLA reproductions filed ([details/xla-gpu-reproductions.md](details/xla-gpu-reproductions.md), the owner's to file). FP8 runs natively under XLA's default emitter once XLA's Triton code is right | [I-97](known-issues.md#i-97) | Booked hardware | S |
| 1.4 | Speed across devices and hosts. Measure wall time against device count on TPU and H100, on one host and on a pod slice (`jax-risk-worker --distributed`, with the cluster's variables set), using the timing harness (precision design §19). Check the storage blocks' alignment with each device's share of the scenarios. Measure the float64 fixed stages' share of a whole job there, decide where they run (decision A-25: on the accelerator as today, on the host's CPU, or per device), and build the choice if it is not the accelerator | [I-61](known-issues.md#i-61) | 1.2, 1.3 | M |

1.1 comes first because nothing runs on a TPU without it. 1.2 comes before 1.4 because speed
measured on unverified numbers shows nothing. 1.3 runs alongside 1.2 on the same H100. A speed
defect that 1.4 finds dominating a cluster job, such as the cold job's compiles
([I-81](known-issues.md#i-81)) or Newton's bootstrap compile ([I-53](known-issues.md#i-53)),
moves into this stage before the research timing.

<a id="stage-2"></a>
## Stage 2 — The precision research result

| Step | Work | Closes | After | Size |
|---|---|---|---|---|
| 2.1 | The first research result on the existing precision formats, on several devices of Ironwood and H100: wall time per figure at equal accuracy, many low-precision paths against fewer float64 paths, measured by the timing harness (`tests/support/precision_timing.py`, precision design §19). The harness times each case's whole job cold and warm, keeps compiles apart, and records peak memory and how its products ran. It measures accuracy over seeds against float64's truth and reads equal accuracy off the observed errors, assuming no rate; a format that does not pay is a result too. The order: the storage formats; then the matrix-product kernels on native FP8 (H100 with `--xla_gpu_enable_triton_gemm=false`, I-97); then each format's two terms against its single pass (A-23); then the induction's budget, time and peak tradeoff (`--budget-divisors`). The research's path counts start at 65,536, since on a CPU and the RTX 5060 a job of up to 16,384 paths was all fixed work. A figure timed on another engine or grid than its campaign's has its rows measured first, or is timed on the measured one (I-114) | [F-07](features.md#f-07) (speed) | Stage 1 | M |

**Precision scope** (owner, 2026-10-15). In this first comparison, PFE and VaR use the
existing, uncorrected quantile estimators, and ES uses the existing tail estimator. The paired
mean correction remains available: corrected and uncorrected means are told apart, and its
float64 work is included in the timing. A format that fails to reach the required accuracy is a
valid research result. Three extensions follow separately, in stage 5, and none of them delays
this measurement: multilevel quantile correction (only if the measurements justify its cost),
FP4, and supported confidence intervals (F-10).

Precision design: [details/precision.md](details/precision.md); its rounding and uncertainty
work: [details/precision-accuracy.md](details/precision-accuracy.md).

<a id="stage-3"></a>
## Stage 3 — Basel III

Regulatory figures as the Basel Framework specifies them, each traceable and reproduced by an
oracle. Each figure is an entry in the run request's `analytics` list.

| Step | Work | Closes | After | Size |
|---|---|---|---|---|
| 3.1 | Basel P0 (decisions D-1 to D-10, pinned text, the BCBS profile, the requirement catalogue and traceability check, measure guards, desk and book on every row, the run manifest, the remaining ORE oracles; P0.9 keeps regulatory runs with the job queue's `keep`), and P2's data acquisition, which takes calendar time and so starts first | [F-05](features.md#f-05) (P0, P2) | Stage 2 | L |
| 3.2 | Basel P1 (FRTB-SA) onward, phase by phase of the [Basel plan](details/basel-iii.md#6-phases-and-tasks): historical scenarios and IMA expected shortfall, backtesting and PLA, SA-CCR and BA-CVA, the precision gate, the evidence pack | [F-05](features.md#f-05) | 3.1's P0 (USD swaps also need [I-05](known-issues.md#i-05)); P7's evidence pack needs 4.5 | L |

3.2 follows the Basel plan's phases, and each phase waits only for what its row lists. Several
prerequisites are already met: FRTB-SA's sensitivities are ORE's (I-51), as is a calibrated
option's historical P&L (I-112), and P5's IMM exposure equals ORE's (I-49). P6's precision gate
is the evidence table, one table shared with the precision research. P6 adds the regulatory
figures' rows, and moves MAR32's thresholds into the regulatory profile once P0.3 has entered
them a second time. The research uses Basel's criteria (A-11) as an engineering acceptance
standard, and a row that passes them is not a regulatory approval. Basel III:
[details/basel-iii.md](details/basel-iii.md).

<a id="stage-4"></a>
## Stage 4 — Fix what exists

The remaining defects in what the engine already does: numbers that are wrong only on inputs
nothing uses yet, documentation, speed, tests and tooling. Work in this order: the parallel
steps at any time, then unchecked behaviour, then speed, then tooling.

| Step | Work | Closes | After | Size |
|---|---|---|---|---|
| 4.1 | *Parallel, can start now* (A-20). Ruff in `pyproject.toml` and CI with pyflakes' rules only (ruff's `F` set: unused imports and variables, undefined names, empty f-strings; no style or formatting rules). Fix or mark each finding; an "unused" import that other modules import from there is a re-export and goes in `__all__`, checked one by one. `pandas`, which nothing imports, leaves the dependencies | [I-66](known-issues.md#i-66) (linter) | — | S |
| 4.2 | *Parallel, can start now.* Retire the `/eod` path: delete `engine/traderx/`, its routes, tests, fixtures and documents, once the owner has confirmed that TraderX needs nothing that only that path gives it | [I-120](known-issues.md#i-120) | — | S |
| 4.3 | *Parallel, can start now.* Correct the documentation's contradictions | [I-110](known-issues.md#i-110) | — | S |
| 4.4 | The narrow cases: a European's liveness by its expiry date, not its variance; a bond valued until its last payment; the LGM's H by `expm1`, host and JAX alike | [I-104](known-issues.md#i-104) | — | S |
| 4.5 | The full-suite aborts and lost worker processes. Show the full suite on CI's 16 GB runner; then run it repeatedly against a known-bad baseline; then put the full tier on a schedule. This comes before Basel P7 (3.2), whose evidence pack rests on a complete suite run | [I-27](known-issues.md#i-27) | — | M |
| 4.6 | The cold job's one-operation programs: jit the scenario market's construction and the other eager sites, and market risk's batch on the price function as data, as was done for the AD Greeks; the snapshot within the margin | [I-81](known-issues.md#i-81) | — | S |
| 4.7 | The pipeline's remaining speed. Find why, on a Windows CPU only, pricing runs 10–25% slower in a process that has run AD Greeks. Win back the 3 s that a cold job lost to the bootstrap's compile under Newton. Schedule the leg kernel's products with their operand fusions | [I-53](known-issues.md#i-53) | — | S |
| 4.8 | Shared test helpers in `tests/support/`; public-entry tests where private-symbol tests are obsolete | [I-67](known-issues.md#i-67) | 4.2 | S |
| 4.9 | A type checker on `engine/` (A-20) | [I-66](known-issues.md#i-66) (type checker) | 4.1 | S |
| 4.10 | Where ORE's sensitivity analytic disagrees with itself or fails. Report the two inconsistencies to ORE, each with a one-trade reproduction: a holiday as-of date's volatility reference, and a deal strike that its fallback moves on an ACT/ACT (ICMA) swap index. Report its optimiser's miss on a large basket under a 5bp volatility shift. Each then becomes a parity test or a difference by decision. Decide in the run request's preflight (`engine.run.validate_run`) whether an as-of date that is not a business day is refused | [I-117](known-issues.md#i-117) | — | S |
| 4.11 | *Parallel, can start now* (the run request's dispatch alone). Do once the work that several analytics in one run share: hand the cross-asset model's calibration from `calibration` to `exposure`, validate the request once, build the trades' ORE schedules once per run. The result still equals each analytic alone bit for bit; each shared stage is counted and timed against the code before | [I-118](known-issues.md#i-118) | — | S |
| 4.12 | XLA's reverse-mode compile failure on a GPU for a calibrated option's price: a minimal reproduction, reported with I-97's (1.3), rechecked on each new jax | [I-119](known-issues.md#i-119) | — | S |
| 4.13 | Before the API is reachable from outside a trusted network: authentication, a body-size limit, a per-caller limit on queued jobs | [I-90](known-issues.md#i-90) | — | S |

4.2 comes before 4.8 because it removes the fixtures that 15 test files repeat. 4.5's runs can
be made whenever a full run is being made anyway; only the conclusion waits for repeated runs.

<a id="stage-5"></a>
## Stage 5 — New capabilities

What the engine does not do yet. A step that adds a path-pricing kernel (5.2, 5.3, 5.4, 5.6)
writes it in difference form, with one implementation for every precision (A-16,
[details/precision.md §8](details/precision.md#8-compute-below-float32)), and gets its rows in
the evidence table.

| Step | Work | Closes | After | Size |
|---|---|---|---|---|
| 5.1 | Swaption smile: a strike axis in the market's volatilities, read at each option's and helper's strike. It is an additive market field, and ATM-only markets are unchanged within the margin. The lookup it widens is written in difference form | [I-54](known-issues.md#i-54) | — | M |
| 5.2 | Engine options: ORE's `AnalyticLgm` European engine, settlement methods, the FD solver | [F-01](features.md#f-01) | — | M |
| 5.3 | FX and equity trades on the pipeline; FX/EQ calibration; the two-currency end-to-end test (L6) against the ORE simulation oracle, extended to a second currency; market risk's FX factors | [F-04](features.md#f-04) | — | L |
| 5.4 | SABR volatility | [F-02](features.md#f-02) | 5.1 | M |
| 5.5 | CVA/DVA | [F-06](features.md#f-06) | — | M |
| 5.6 | AMC engine | [F-03](features.md#f-03) | — | L |
| 5.7 | ORE's XML input files (`ore.xml`'s analytics, `simulation.xml`, `pricingengine.xml`, `sensitivity.xml`, `portfolio.xml`, zero-rate market data) translated into a run request, refusing by name whatever the engine does not support. Nice to have, not critical (owner, 2026-10-08) | [F-09](features.md#f-09) | — | M |
| 5.8 | **Conditional research experiment: multilevel quantile correction.** Do it only if 2.1's results show a useful opportunity. Combine many low-precision paths with paired float64 paths to estimate PFE and VaR through a corrected distribution or an equivalent estimator, and ES by a related tail-estimation method. Compare against the uncorrected estimators and float64 at equal total wall time, including the costs of the paired simulation, the correction and the inversion. Validate the tail accuracy and its uncertainty, and keep the raw results beside the corrected ones. Do not implement it merely to make a failing format pass | [F-07](features.md#f-07) (quantile and tail estimation) | 2.1; evidence of likely benefit | L |
| 5.9 | **Deferred: FP4 storage and compute.** Storage with a variance correction through block scales; emulated compute using the existing precision architecture and the rounding lessons of I-98; its own precision campaigns and evidence rows; native compute measured on TPU 8t/8i when available | [F-07](features.md#f-07) (FP4) | 2.1 | L |
| 5.10 | Supported sampling confidence intervals in a stated estimator and model domain, with validated tail assumptions or bounds; rare-event validation on independent outer seeds, the diagnostic criteria unchanged, and intervals explicitly unavailable everywhere else | [F-10](features.md#f-10) | 2.1 | L |
| 5.11 | The cash flows paid in Theta's roll, reported per trade beside Theta from both Greek methods and never folded into Theta, which stays ORE's (A-26); tested against the trades' schedules and ORE's cash-flow report on the Theta date | [F-11](features.md#f-11) | — | S |

Order within the stage: the smile comes first. No market given to the engine so far has a
smile, so it changes no number yet, but SABR builds on it. Then come the smallest gaps in ORE's
own choices (5.2); then the largest scope gap (5.3: the model already simulates FX and equity,
but nothing prices them); then SABR (5.4); then the analytics built on proven exposure (5.5,
5.6). 5.7 has no dependants, and 5.11 is small and can start at any time. The precision
extensions (5.8 to 5.10) come after the first research result: quantile correction only when
its measured opportunity warrants an experiment, and FP4 separately, at lower priority.

<a id="waiting-on-others"></a>
## Waiting on others

None of these is engineering work until its input arrives. Chase the dependency, not the code.

| Item | Waiting for | From |
|---|---|---|
| Steps 1.2 to 1.4 and 2.1 | Booked time on TPUs (Ironwood, and a pod slice for 1.4) and H100s ([access request](../google-cloud-research-access.md)) | Owner, the cloud administrator |
| [I-05](known-issues.md#i-05) USD-SOFR swaps | The convention set to build (D03/D04) | Owner, with the caller booking SOFR swaps |
| [I-07](known-issues.md#i-07) corporate bonds, listed options | Demand, and for corporate bonds a credit model | Owner |
| The project's licence (no `LICENSE` file or licence metadata; project audit, AUD-49) | A decision on first-party licensing | Owner |
