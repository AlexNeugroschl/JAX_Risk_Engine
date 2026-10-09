# Roadmap

The order in which to work through [known-issues.md](known-issues.md) and
[features.md](features.md). Every open item appears here exactly once; an item that is not
here is not planned ([rules](README.md#lifecycle)).

**How the order is decided** (decision A-19, revised 2026-10-08).

1. **The near-term milestone**: `demos/demo_profile_small.py` runs on the owner's local GPU
   and its profiler trace covers the whole job (stage 2, done 2026-10-07).
2. **The core** (stage 3), four goals, in this order of priority where they compete:
   1. **Precision research**, the project's purpose (root [README](../../README.md)): the
      measurement harness, the kernels in their final form, the evidence table, and the
      research result on TPU and H100.
   2. **One clear, unified API**: one run request for every analytic, over HTTP and in
      Python, and one way into the engine for TraderX's submissions too.
   3. **Basel III**: regulatory figures as the Basel Framework specifies them, each traceable
      and reproduced by an oracle. It is also the precision research's acceptance standard
      (A-11), so the two meet in the evidence table.
   4. **A clear `engine/`**: packages named for what they hold, nothing at the root but
      `__init__.py` ([details/package-layout.md](details/package-layout.md)). Done first
      (2026-10-08), so that every later step is written in the final layout.
3. **Fix what exists** (stage 4): the defects of what the engine already does, small and
   large, and its tests and tooling. What the engine has must work before it gains more.
4. **New capabilities** (stage 5): what the engine does not do yet, beyond the core.

Within a stage, work in the order of its table. Items marked *parallel* touch code that no
earlier step changes, and can start at any time. Items waiting on someone outside the project
are listed [separately](#waiting-on-others), with who to chase. A defect found along the way
goes to stage 4 (at its head if it is High: a wrong number on an ordinary input, or another
caller's result served), or before the step it blocks.

**Step numbers are this page's alone.** Steps are numbered in the order of work and
renumbered whenever it changes. Nothing else cites a step: code, error messages, tests and the
other documents cite the item (I-NN, F-NN), the decision (A-n), or a date, and
`tests/test_import_layering.py` fails on a step number in code.

**Every step keeps the project's goals.** Nothing above trades one away:

- **ORE parity.** Every ORE parity suite passes at its existing tolerance after every step. A
  new capability comes with its own ORE parity test (through the ORE simulation oracle,
  `tests/support/ore_xva_oracle.py`, where it needs a simulation), or it is refused by name.
- **Accuracy.** float64 numbers stay bit for bit unless the step says it moves them. The AD
  Greeks moved at rounding level twice, as planned: on 2026-10-06, when each became one
  compiled program, and on 2026-10-07 under the Newton root solver (A-21; its reference, the
  bisection, keeps the numbers before it bit for bit). The steps planned to move them are 3.8
  and 3.9 (default Bermudan/American values, towards ORE's) and 3.6 (rounding level, each
  kernel family once). Such a step shows parity first, then re-baselines the golden snapshot
  and records the largest change per array
  ([details/precision.md §13.1](details/precision.md#131-bit-for-bit-and-ore-parity)).
  Bit-for-bit checks run on CPU. A GPU's float64 differs in the last bits and is held to the
  parity tolerances.
- **Speed and JAX.** A new or changed pricer is a module-level jitted function that takes the
  trade's data as a pytree argument
  ([profiling §3.7](../concepts/profiling.md#37-trade-data-as-traced-arguments-2026-10-02)),
  so there is no compile per trade, date, bump or call. It runs under the scenario-axis
  sharding, and the compile-count tests cover it.
- **Precision research.** Every path kernel is written in 3.6's form, one implementation for
  every precision (A-16), and gets its rows in 3.2's harness. Calibration, Greeks,
  reductions and the other fixed stages (A-10) stay float64.
- **Basel III.** Nothing a regulatory figure relies on is approximated silently: an input
  outside scope is refused by name, as today. Results stay auditable: the job queue keeps
  them, and 3.13's retention policy keeps regulatory runs.
- **API compatibility.** Once the HTTP API has a client, changes are additive: every route,
  field, default and result shape keeps working, and a change of the contract is a revision of
  it (A-2; the names A-2 retired were removed outright on 2026-10-07, there being no client
  yet, by the owner's revision of that day). 3.3 replaces the routes with one run request on
  the same terms, before any client exists. A new setting arrives with its API field, which
  `tests/test_api_completeness.py` checks. Two narrowings are planned, both deliberate: 4.1
  refuses inputs that are now accepted and then ignored (the old answer is wrong), and 3.8
  changes a default number to ORE's (decision A-3). TraderX's routes (`/eod/...`) are
  TraderX's contract and keep their shape when 3.4 changes what runs behind them.
- **Tests.** A step that breaks a test of a private symbol rewrites it against the public
  entry ([I-67](known-issues.md#i-67)'s rule), so restructuring does not wait for 4.8.

## Where things stand

One pipeline, ORE's: a cross-asset model with a model per currency (the LGM by default, or
Hull-White, both ORE's `<LGM>` parametrizations), every trade valued on every path as ORE's
valuation engine does. It matches ORE component by component: t=0 prices to 1e-14 (swaps,
Europeans, bonds) and 4e-11 (calibrated Bermudans/Americans), model analytics to 1e-12
(the Hull-White model's bonds against QuantLib's `HullWhite` too), pricers on path curves to
1e-8 – 1e-12 under either model, market-risk VaR/ES per scenario to 2e-13, and since
2026-10-07 an ORE simulation run in-process is the reference for the assembled pipeline:
swaps, Europeans and bonds equal it path by path (Bermudans and Americans 0.5–4% apart, I-49,
which 3.9 closes). Sensitivities are not yet shown to equal ORE's.

Every model, engine, Greeks and precision choice is one run configuration with ORE's
defaults. The simulation, the scenario market and path pricing each take a storage and a
compute precision (float64 or float32 compute; storage down to float16, bfloat16 and FP8 with
block scales and nearest or stochastic rounding), set per product and per trade for pricing,
in portfolio and market-risk runs alike; every result carries a precision report, and a
paired float64 sample corrects the mean figures by a two-level estimator and measures the
quantiles. HTTP jobs go through a durable SQLite job queue to one single-threaded engine
worker process per host, which owns the host's devices (decision A-14) and splits each job's
scenarios across them. The engine runs on an NVIDIA GPU (Linux or WSL2, the `gpu` extra) with
deterministic kernels, and each matrix product states its own precision (A-22). A repeated job
compiles nothing; every calibration and exercise boundary is solved by one configurable root
solver (A-21). The HTTP API reaches every setting the engine has, through one route per
analytic (portfolio pricing, market risk, the cross-asset model's calibration). Nothing runs
on more than one host. TraderX submits end-of-day bundles through its own routes (`/eod/...`),
which price its Treasuries with their own pricers and refuse everything else by name. Since
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
cold job's one-operation programs (4.6), a slower pricing on Windows after AD Greeks and a cold
job 3 s slower for Newton's compile (4.7), and the grid rollback's memory, which stops the
demo's job beyond a few thousand paths (3.6). The card's float64 runs at 1/64 of its float32
rate, so it is where correctness and the trace are checked, not where speed is measured (3.11).

<a id="stage-3--the-core"></a>
## Stage 3 — The core

The precision research, one API, Basel III and a clear `engine/`. Where they compete,
precision comes first. A defect is here only where a core step needs it fixed first (3.8 and
3.9, for the Bermudan/American kernel and Basel's exposure).

| Step | Goal | Work | Closes | After | Size |
|---|---|---|---|---|---|
| 3.2 | Precision | *Parallel, can start now* (it changes no float64 number). Store classes whose level swamps their spread relative to a level (the cube to its t=0 value, the curves to their path-independent part, or a block offset). Build a measurement harness, rerunnable on any kernel change, for the storage formats per class, product and path count, with the paired sample and its estimator's coverage on the pipeline | [I-75](known-issues.md#i-75) | — | M |
| 3.3 | API | One run request in place of a route per engine function, as ORE runs a portfolio with a list of analytics: `POST /runs` with `market`, `portfolio`, `analytics` (`npv`, `exposure`, `sensitivities`, `market_risk`, `calibration`) and a `config` whose sections follow ORE's files, each setting in one place; one result with the configuration as run; `GET /config/defaults`; an idempotency key; a Python entry, `run(RunRequest)`, in `run/`. The four routes and `/jobs` go, there being no client. The completeness test also checks each field's wiring. A run of one analytic equals today's direct call bit for bit ([design](details/configurable-engine.md#one-run-request-i-87-i-88-i-89)) | [I-87](known-issues.md#i-87), [I-88](known-issues.md#i-88), [I-89](known-issues.md#i-89) | — | M |
| 3.4 | API | One way into the engine for TraderX: its bundle becomes a run request, after its convention checks and refusals (so a refused booking still never reaches a pricer, I-05), and the engine prices its bills and notes; the TraderX path's own pricers go. Its routes, result document, attempt store and accrued-interest reconciliation stay. First, the engine's bond and the TraderX note agree on every bundle row of the tests to rounding ([design](details/package-layout.md#3-merges)) | [I-93](known-issues.md#i-93) | 3.3 | M |
| 3.5 | Basel | *Parallel, start now.* Basel P0 (decisions D-1 to D-10, pinned text, the BCBS profile, the requirement catalogue and traceability check, measure guards, desk and book on every row, the run manifest, the remaining ORE oracles) and P2's data acquisition, which is calendar time. P0.9 (regulatory runs kept by the job queue) waits for 3.13; the rest of P0 does not | [F-05](features.md#f-05) (P0, P2) | P0.9: 3.13 | L |
| 3.6 | Precision | Kernels in difference form with explicit accumulators, one family at a time, one implementation for every precision (A-16); compute below float32 enabled. The families 3.8 and 3.9 do not change come first (simulation scan, scenario curves, legs, Europeans, exposure), so 3.7 starts on them; the Bermudan/American rollback and recalibration come last, after 3.8 and 3.9, so no kernel is rewritten twice. Where a family can be a matrix product (leg pricing, the Bermudan rollback), it takes that form in the same rewrite, so native FP8 (3.11) needs no second one. A product's own precision (TensorFloat-32, bfloat16 passes, FP8) becomes a compute format of the policy, named in the report: the product helper (`product_precision`) maps the policy's format to it, so every product follows the policy and never a device's default. Where a device has no such unit (TensorFloat-32 on a CPU, which computes float32 instead) the format is emulated by rounding the operands to its mantissa, as FP8 storage is, and the report says which ran ([details/precision.md §8.3](details/precision.md#83-emulation-and-native-speed)). Per family, ORE parity at existing tolerances first, then the float64 snapshot re-baselined. The rollback's matrix form also holds one operator per step instead of one per path and step, so the demo's job runs at 65,536 and 262,144 paths | [F-07](features.md#f-07) (compute), [I-83](known-issues.md#i-83) | 3.2; the Bermudan/American family 3.8, 3.9 | L |
| 3.7 | Precision, Basel | The evidence table against the acceptance standard (A-11: Basel III's P&L attribution test and the Basel plan's P6.2 rule; P6.2 labelled as engineering for figures Basel does not cover), from 3.2's harness on each family of 3.6 as it lands (the Bermudan/American rows after 3.8 and 3.9), then on each kernel stage 5 adds; a warning on any result whose combination has no passing row | [I-55](known-issues.md#i-55) (warnings) | Each family of 3.6 | S |
| 3.8 | Precision, Basel | `ShiftHorizon`: the setting and its API field exist and refuse anything but 0; implement the shift, parity at 0.5 against the LGM oracle, then 0.5 as the default | [I-32](known-issues.md#i-32) | — | M |
| 3.9 | Precision, Basel | Reproduce ORE's two per-path recalibration details against `tests/test_ore_xva_parity.py`, until its Bermudan/American L3 cases (strict expected failures today, 0.5–4% apart) pass; confirm an American's basket on a path against ORE's (decision A-7); warn, as ORE's `LgmBuilder` does, when a path's recalibration misses its basket | [I-49](known-issues.md#i-49), [I-73](known-issues.md#i-73) | — | M |
| 3.10 | Precision | *Parallel, can start now* (it moves no kernel's arithmetic). One worker per host on a Cloud TPU pod slice, with process 0 claiming each job and handing it to the other hosts, since under SPMD every host runs the same job ([details/precision.md §11.3](details/precision.md#113-multi-device-and-multi-host)). Measure wall time against device count on TPU and H100. The one-host split of the scenario axis is done (2026-10-04) | [I-61](known-issues.md#i-61) | — | M |
| 3.11 | Precision | The research result: on several devices of Ironwood and H100, wall time per figure at equal accuracy, many low-precision paths against fewer float64 paths, read from the evidence table's path ceilings; the storage formats, then 3.6's matrix-product kernels on native FP8 | [F-07](features.md#f-07) (speed) | 3.7, 3.10 | M |
| 3.12 | Basel | *Parallel, can start now.* Sensitivities against ORE's sensitivity analytic on the shared portfolio; the AD Greeks against the bump Greeks on the same sloped portfolio. FRTB-SA's sensitivities rest on it | [I-51](known-issues.md#i-51), [I-78](known-issues.md#i-78) | — | M |
| 3.13 | API, Basel | The job queue: TraderX's accepted-attempt record, boot sweep and `interrupted` state (portfolio jobs have had all three since 2026-10-04); a retention policy that keeps regulatory runs; a job status that says when the engine worker cannot start; a cancel route (a pending job at once, a running one at the worker's next phase boundary) | [I-08](known-issues.md#i-08), [I-76](known-issues.md#i-76), [I-77](known-issues.md#i-77), [I-91](known-issues.md#i-91) | 3.3, 3.10 | M |
| 3.14 | Basel | Basel P1 (FRTB-SA) onward, phase by phase of the [Basel plan](details/basel-iii.md#6-phases-and-tasks): historical scenarios and IMA expected shortfall, backtesting and PLA, SA-CCR and BA-CVA, the precision gate, the evidence pack. Each figure an entry of 3.3's `analytics` | [F-05](features.md#f-05) | 3.5's P0, 3.12 (USD swaps also I-05); P5's IMM 3.9; P6's precision gate 3.7; P7's evidence pack 4.5 | L |
| 3.15 | Precision | FP4: storage with a variance correction through the block scales, compute on TPU 8t/8i; multilevel estimation of quantiles (PFE, VaR, ES) | [F-07](features.md#f-07) (FP4) | 3.6, 3.11 | L |

Order within the stage:

- **The layout is done** (2026-10-08, [I-92](known-issues.md#i-92)): `engine/` is the
  [layout](details/package-layout.md#2-the-layout) every later step writes in (3.3's run
  request in `engine/run/`, 3.14's `engine/regulatory/` above `engine/risk/`), and
  `tests/test_import_layering.py` keeps it: a module at the root of `engine/` or
  `engine/risk/`, a package without a layer, or an import of a layer above fails.
- **Precision ahead where steps compete.** 3.2 builds the harness, 3.6 rewrites the kernels it
  measures, and 3.7 reads the table off each family as it lands, so swaps, Europeans and bonds
  have evidence rows while 3.8 and 3.9 make the Bermudan/American kernel ORE's. The rollback
  family is then rewritten once, after them, and its rows follow. 3.10 can run at any time
  before 3.11, which needs it and the table. FP4 (3.15) comes last, being F-07's stretch goal.
- **3.3 and 3.4 before any analytic is added.** They change the HTTP contract and the way into
  the engine, so they come before the analytics 3.12 and 3.14 add here and 5.5 and 5.6 add in
  stage 5, which then arrive as entries of the `analytics` list rather than as routes. A
  setting added before them gets its API field in the request shape of the day, and 3.3
  carries it over.
- **Basel alongside, from now.** 3.5 starts at once, because P2's data is calendar time and P0
  fixes the text and decisions every figure cites. 3.12 comes before 3.14, because FRTB-SA's
  sensitivities must first be shown to be ORE's. 3.14 then follows the Basel plan's phases,
  each waiting only for what its row lists: P5's IMM for 3.9 (a Bermudan's exposure equal to
  ORE's), P6's precision gate for 3.7, which is one table shared with the precision research.
- **3.8 and 3.9 as blockers.** They are ORE parity defects, which would otherwise wait for
  stage 4; they are here because the Bermudan/American kernel family, its evidence rows and
  Basel's IMM exposure need them first.
- **3.10 before 3.13.** 3.10 needs a pod slice, and it has to finish before 3.13, which
  changes the same worker loop; 3.13 in turn comes before Basel P0.9, which keeps regulatory
  runs in the job queue.
- **Numbers.** 3.2, 3.3, 3.10 and 3.13 move no float64 number; 3.4 moves none of the
  engine's, and TraderX's only within the agreement it starts by showing. 3.8 and 3.9 move
  default Bermudan/American numbers towards ORE's. 3.6 moves float64 at rounding level, each
  family once, with the snapshot re-baselined after each.

Parity methodology: [details/ore-parity-validation.md](details/ore-parity-validation.md);
precision: [details/precision.md](details/precision.md); Basel III:
[details/basel-iii.md](details/basel-iii.md); layout:
[details/package-layout.md](details/package-layout.md).

<a id="stage-4--fix-what-exists"></a>
## Stage 4 — Fix what exists

The defects of what the engine already does, and its tests and tooling, before it gains
anything beyond the core. In order: a High defect first, then wrong or unchecked behaviour,
then speed, then tooling.

| Step | Work | Closes | After | Size |
|---|---|---|---|---|
| 4.1 | TraderX's submissions: check submission binding before any cache return (I-57 is High: one caller can be served another's result); one execution owner per workload; refuse unknown calculations and a reporting currency other than USD (A-18, as ORE rejects an unknown analytic) | [I-57](known-issues.md#i-57), [I-58](known-issues.md#i-58), [I-59](known-issues.md#i-59) | 3.4 | S |
| 4.2 | Chase TraderX's answers; apply them (a widened allowlist, a schema statement). When they arrive | [I-23](known-issues.md#i-23), [I-60](known-issues.md#i-60) | Their answers | S |
| 4.3 | One LGM calibration: the standalone bootstrap (`calibrate_lgm_sigma`, its basket and Bachelier helper prices) retires once 3.3's `calibration` analytic returns what `POST /calibration/lgm` returned, or moves to `tests/support/` as an independent check ([design](details/package-layout.md#3-merges)) | [I-94](known-issues.md#i-94) | 3.3 | S |
| 4.4 | *Parallel, start now* (A-20). Ruff in `pyproject.toml` and CI with pyflakes' rules only (ruff's `F` set: unused imports and variables, undefined names, empty f-strings; no style or formatting rules). Fix or mark each finding; an "unused" import that other modules import from there is a re-export and goes in `__all__`, checked one by one. Earlier is cheaper: every later step's code is then written lint-clean once | [I-66](known-issues.md#i-66) (linter) | — | S |
| 4.5 | Characterize the full-suite aborts and lost worker processes. The fast tier's CI deaths were memory and are fixed (2026-10-05: the grid-convergence test's grid, compiled programs dropped per module). Still to do: show the full suite on CI's 16 GB runner (`-n 4` peaked at 24.2 GB with page cache before those fixes), then repeated full runs against a known-bad baseline. Before Basel P7 (3.14), whose evidence pack rests on a complete suite run | [I-27](known-issues.md#i-27) | — | M |
| 4.6 | The cold job's one-operation programs: jit the scenario market's construction and the other eager sites, and market risk's batch on the price function as data, as was done for the AD Greeks; the golden snapshot's change per array shown | [I-81](known-issues.md#i-81) | — | S |
| 4.7 | The pipeline's remaining speed: find why, on Windows' CPU only, pricing runs 10–25% slower in a process that has run AD Greeks (the Windows heap is suspected, not shown; Linux is unaffected); and win back the 3 s (8%) a cold job lost to the bootstrap's compile under Newton, by tracing the helper's price once per bucket program where it is traced at the bracket's ends, the step and the model value today, with the snapshot bit for bit | [I-53](known-issues.md#i-53) | — | S |
| 4.8 | Shared test helpers in `tests/support/`; public-entry tests where stage 1 made private-symbol tests obsolete | [I-67](known-issues.md#i-67) | — | S |
| 4.9 | A type checker on `engine/` (A-20) | [I-66](known-issues.md#i-66) (type checker) | 4.4 | S |
| 4.10 | Before the API is reachable from outside a trusted network: authentication chosen with TraderX, a body-size limit, a per-caller limit on queued jobs | [I-90](known-issues.md#i-90) | — | S |

4.5's runs can be made whenever a full run is being made anyway. Five full runs under WSL2
(2026-10-06, `-n 4`, a 23 GB VM; four on the GPU, one on CPU) all printed their summary lines,
the last of each green; see the verification status. Only the conclusion waits for repeated
runs.

<a id="stage-5--new-capabilities"></a>
## Stage 5 — New capabilities

What the engine does not do yet, beyond the core. A step that adds a path-pricing kernel
(5.2, 5.3, 5.4, 5.6) writes it in 3.6's form, one implementation for every precision (A-16),
and gets its rows in 3.7's table.

| Step | Work | Closes | After | Size |
|---|---|---|---|---|
| 5.1 | Swaption smile: a strike axis in the market's volatilities, read at each option's and helper's strike; an additive market field, with ATM-only markets bit for bit. The lookup it widens is written in 3.6's form | [I-54](known-issues.md#i-54) | 3.6 | M |
| 5.2 | Engine options: ORE's `AnalyticLgm` European engine, settlement methods, FD solver (the AD Greeks method and the market-risk engine by configuration are done) | [F-01](features.md#f-01) | 3.6; the FD solver also 3.8 | M |
| 5.3 | FX and equity trades on the pipeline; FX/EQ calibration; the two-currency end-to-end test (L6) against the ORE simulation oracle, extended to a second currency | [F-04](features.md#f-04) | 3.6 | L |
| 5.4 | SABR volatility | [F-02](features.md#f-02) | 5.1 | M |
| 5.5 | CVA/DVA | [F-06](features.md#f-06) | 3.3 | M |
| 5.6 | AMC engine | [F-03](features.md#f-03) | 3.6 | L |
| 5.7 | Reporting currencies other than USD for TraderX: convert at the as-of FX spot, as ORE reports in its `baseCurrency`; parity against an ORE run with that base currency | [F-08](features.md#f-08) | 4.1; an FX source ([I-18](known-issues.md#i-18), waiting on others) | S |
| 5.8 | ORE's XML input files (`ore.xml`'s analytics, `simulation.xml`, `pricingengine.xml`, `sensitivity.xml`, `portfolio.xml`, zero-rate market data) translated into a run request, refusing by name what the engine does not support. Nice to have, not critical (owner, 2026-10-08) | [F-09](features.md#f-09) | 3.3 | M |

Order within the stage: the smile first (no market given to the engine so far has one, so it
changes no number yet, but SABR builds on it); then the smallest gaps in ORE's own choices
(5.2), the largest scope gap (5.3: the model already simulates FX and equity, but nothing
prices them), SABR (5.4), and the analytics built on proven exposure (5.5, 5.6). 5.7 as soon
as its FX source arrives; 5.8 last, since nothing waits on it.

<a id="waiting-on-others"></a>
## Waiting on others

Not engineering work until the input arrives. Chase the dependency, not the code.

| Item | Waiting for | From |
|---|---|---|
| [I-05](known-issues.md#i-05) USD-SOFR swaps | The D03/D04 convention set | TraderX |
| [I-04](known-issues.md#i-04) seasoned TraderX swaps | `pastFixings` in the export | TraderX |
| [I-16](known-issues.md#i-16) per-pillar `rateSensitivity` | An observed market-data package (W2) | TraderX |
| [I-18](known-issues.md#i-18) equity positions; [F-08](features.md#f-08) reporting currencies (step 5.7) | A spot/FX source | TraderX or a market-data decision |
| [I-23](known-issues.md#i-23), [I-60](known-issues.md#i-60) | Answers on versioning and added fields | TraderX (then step 4.2) |
| [I-07](known-issues.md#i-07) corporate bonds, listed options | Demand, and for corporates a credit model | Owner |

Open TraderX asks are collected in
[details/traderx-integration.md](details/traderx-integration.md#open-with-traderx).
