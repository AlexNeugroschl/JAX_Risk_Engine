# Roadmap

The order in which to work through [known-issues.md](known-issues.md) and
[features.md](features.md). Every open item appears here exactly once; an item that is not
here is not planned ([rules](README.md#lifecycle)).

**How the order is decided** (decision A-19, revised 2026-10-08). The core of the system comes
first; the rest of the scope next; defects with small, local fixes, however important, once the
core works.

1. **The near-term milestone**: `demos/demo_profile_small.py` runs on the owner's local GPU
   and its profiler trace covers the whole job (stage 2, done 2026-10-07).
2. **The core** (stage 3), four goals, in this order of priority where they compete:
   1. **Precision research**, the project's purpose (root [README](../../README.md)): the
      measurement harness, the kernels in their final form, the evidence table, and the
      research result on TPU and H100.
   2. **One clear, unified API**: one run request for every analytic, over HTTP and in Python.
   3. **Basel III**: regulatory figures as the Basel Framework specifies them, each traceable
      and reproduced by an oracle. It is also the precision research's acceptance standard
      (A-11), so the two meet in the evidence table.
   4. **A clean `engine/`**: `__init__.py` and subpackages only, so later steps land in the
      final layout. It is the smallest of the four and goes first for that reason.
3. **Completeness** (stage 4): every other capability in scope, in the order its dependencies
   allow.
4. **Local defects and hardening** (stage 5): defects whose fix is small and local, tests and
   tooling.

Within a stage, work in the order of its table. Items marked *parallel* touch code that no
earlier step changes, and can start at any time. Items waiting on someone outside the project
are listed [separately](#waiting-on-others), with who to chase. A defect found along the way
goes to stage 5, unless it blocks a step, in which case it goes before that step, or it is
High (a wrong number on an ordinary input, or another caller's result served) on a path that
has a real caller, in which case it goes first.

**Step numbers are permanent.** Code, error messages and other documents cite them, so a step
keeps its number when it moves; the stage it is in and its place in the order are set by this
page, not by its number. Stage 3 holds steps numbered 3, 4 and 5; stage 5 holds 4.1, 4.3 and
the 6s.

**Every step keeps the project's goals.** Nothing above trades one away:

- **ORE parity.** Every ORE parity suite passes at its existing tolerance after every step. A
  new capability comes with its own ORE parity test (through the ORE simulation oracle,
  `tests/support/ore_xva_oracle.py`, where it needs a simulation), or it is refused by name.
- **Accuracy.** float64 numbers stay bit for bit unless the step says it moves them. 2.4,
  planned bit for bit, moved the AD Greeks at rounding level (each became one compiled
  program, which removed the repeated job's recompiles; everything else stayed bit for bit).
  2.5 moved them at rounding level once more (the default root solver, Newton; its reference,
  the bisection, keeps the numbers before it bit for bit). The steps planned to move them are
  3.4 and 3.5 (default Bermudan/American values, towards ORE's) and 3.7 (rounding level,
  each kernel family once). Such a step shows parity first, then re-baselines the golden
  snapshot and records the largest change per array
  ([details/precision.md §13.1](details/precision.md#131-bit-for-bit-and-ore-parity)).
  Bit-for-bit checks run on CPU. A GPU's float64 differs in the last bits and is held to the
  parity tolerances.
- **Speed and JAX.** A new or changed pricer is a module-level jitted function that takes the
  trade's data as a pytree argument
  ([profiling §3.7](../concepts/profiling.md#37-trade-data-as-traced-arguments-2026-10-02)),
  so there is no compile per trade, date, bump or call. It runs under the scenario-axis
  sharding, and the compile-count tests cover it.
- **Precision research.** Every path kernel is written in 3.7's form, one implementation for
  every precision (A-16), and gets its rows in 3.6's harness. Calibration, Greeks,
  reductions and the other fixed stages (A-10) stay float64.
- **Basel III.** Nothing a regulatory figure relies on is approximated silently: an input
  outside scope is refused by name, as today. Results stay auditable: the job queue keeps
  them, and 4.2's retention policy keeps regulatory runs.
- **API compatibility.** Once the HTTP API has a client, changes are additive: every route,
  field, default and result shape keeps working, and a change of the contract is a revision of
  it (A-2; 3.1 removed the names A-2 retired outright, there being no client yet, by the
  owner's revision of 2026-10-07). 3.9 replaces the routes with one run request on the same
  terms, before any client exists. A new setting arrives with its API field, which
  `tests/test_api_completeness.py` checks. Two narrowings are planned, both deliberate: 4.1
  refuses inputs that are now accepted and then ignored (the old answer is wrong), and 3.4
  changes a default number to ORE's (decision A-3).
- **Tests.** A step that breaks a test of a private symbol rewrites it against the public
  entry ([I-67](known-issues.md#i-67)'s rule), so restructuring does not wait for 6.3.

## Where things stand

One pipeline, ORE's: a cross-asset model with a model per currency (the LGM by default, or
Hull-White, both ORE's `<LGM>` parametrizations), every trade valued on every path as ORE's
valuation engine does. It matches ORE component by component: t=0 prices to 1e-14 (swaps,
Europeans, bonds) and 4e-11 (calibrated Bermudans/Americans), model analytics to 1e-12
(the Hull-White model's bonds against QuantLib's `HullWhite` too), pricers on path curves to
1e-8 – 1e-12 under either model, market-risk VaR/ES per scenario to 2e-13. What is not yet
shown is that the assembled simulation, exposure and sensitivities equal an ORE run. Every
model, engine, Greeks and precision choice is one run configuration with ORE's defaults; since
1.4 the simulation, the scenario market and path pricing each take a storage and a compute
precision (float64 or float32), in portfolio and market-risk runs alike, with the float64
default bit for bit as before; since 1.5 path pricing is set per product and per trade too;
since 1.6 storage goes down to float16, bfloat16 and FP8, with block scales and nearest or
stochastic rounding; since 1.7 every result carries a precision report (the policy as run,
the formats read from the arrays, the device), and a paired float64 sample corrects the mean
figures by a two-level estimator and measures the quantiles; since 1.8 HTTP jobs go through a
durable SQLite job queue to one single-threaded engine worker process per host, which owns
the host's devices (decision A-14); since 2026-10-04 each job's scenarios are split across
them; since 2.2 the engine runs on an NVIDIA GPU (Linux or WSL2, the `gpu` extra), shown on
the owner's RTX 5060 by the demo and the full suite, with float32 products at float32 and
deterministic kernels on any accelerator; since 2.3 each matrix product states its own
precision, and importing `engine` changes no process setting but x64; since 2.4 a repeated job
compiles nothing, each trade's AD Greeks are one compiled program per product, and a trace can
be narrowed to one phase; since 2.5 every calibration and exercise boundary is solved by one
configurable root solver (a safeguarded Newton method by default, the bisection as the
reference), and a Bermudan's or American's path dates of one basket shape are calibrated
together; since 3.1 the HTTP API reaches every setting the engine has (one route per analytic:
portfolio pricing, market risk, the cross-asset model's calibration; every per-trade result a
row keyed by its trade; a cube or P&L by hashed reference on request); since 3.2 an ORE
simulation run in-process is the reference for the assembled pipeline, and swaps, Europeans and
bonds equal it path by path (Bermudans and Americans 0.5–4% apart, I-49, which 3.5 closes).
Nothing runs on more than one host.
The TraderX EOD boundary prices Treasuries end to end and refuses everything else by name.

---

<a id="stage-1--structure"></a>
Stage 1, structure (the configurable engine, decision A-1, steps 1.2 to 1.8), is done
(2026-10-04): one run configuration, the precision mechanism through storage below 32 bits
and the precision report, and the engine worker behind the job queue. Each step reproduced
the previous step's numbers bit for bit; the evidence is in
[details/precision.md §13.1](details/precision.md#131-bit-for-bit-and-ore-parity) and the
designs in [details/configurable-engine.md](details/configurable-engine.md) and
[details/precision.md](details/precision.md).

<a id="stage-2--the-demo-on-a-local-gpu"></a>
Stage 2, the demo on a local GPU with a whole trace (steps 2.1 to 2.5), is done (2026-10-07):
`demos/demo_profile_small.py` runs through the HTTP API and the engine worker on the owner's
RTX 5060 (Linux or WSL2, the `gpu` extra) as on the CPU, and its trace covers the whole job, each
phase labelled, or one phase on request. On the way: importing `engine` sets only x64 and every
matrix product states its precision (2.3, A-22); each trade's AD Greeks are one program per
product, and a repeated job compiles nothing (2.4); every calibration and exercise boundary is
solved by one configurable root solver, a safeguarded Newton method by default with the
bisection kept as the reference (2.5, A-21), and an option's path dates of one basket shape are
calibrated together. The recalibration is 35–55 times faster on the CPU and 15–38 times on the
GPU, where a repeated demo job went from 5.5 s to 1.4 s
([profiling §2.0](../concepts/profiling.md#20-the-demo-measured-2026-10-05-roadmap-21)). Each step
kept float64 bit for bit but 2.4 (the AD Greeks) and 2.5 (the Newton default, at rounding level;
the reference keeps the numbers before it bit for bit); the evidence is in
[details/precision.md §13.1](details/precision.md#131-bit-for-bit-and-ore-parity). What it left:
the cold job's one-operation programs (6.5), a slower pricing on Windows after AD Greeks and a
cold job 3 s slower for Newton's compile (6.6), and the grid rollback's memory, which stops the
demo's job beyond a few thousand paths (3.7). The card's float64 runs at 1/64 of its float32
rate, so it is where correctness and the trace are checked, not where speed is measured (5.2).

<a id="stage-3--the-core"></a><a id="stage-3--foundations"></a><a id="stage-5--precision-research"></a>
## Stage 3 — The core

The precision research, one API, Basel III and a clean `engine/`, in the order below. Where
they compete, precision comes first. ORE parity defects come here only where a core step
needs them fixed first (3.4 and 3.5, for the Bermudan/American kernel and Basel's exposure).

| Step | Goal | Work | Closes | After | Size |
|---|---|---|---|---|---|
| 3.10 | Layout | *Start now* (code moves; no number does). `engine/` holds `__init__.py` and subpackages only. A new package `engine/market/` takes `engine/market.py`, re-exported by its `__init__.py` so every `engine.market` import stands, and `engine/day_count.py` as `engine.market.day_count`, with the simulation time axis's ACT/365 (`TIME_AXIS_DAY_COUNTER`, today in `engine.models.ore_builders`) moved beside it so the package imports neither JAX nor `engine.models` and the EOD boundary may still import it. A test fails on any module at the root but `__init__.py`. Later packages (3.9's run request, the Basel plan's `engine/regulatory/`) follow the same rule | [I-92](known-issues.md#i-92) | — | S |
| 3.6 | Precision | *Parallel, can start now* (it changes no float64 number). Store classes whose level swamps their spread relative to a level (the cube to its t=0 value, the curves to their path-independent part, or a block offset). Build a measurement harness, rerunnable on any kernel change, for the storage formats per class, product and path count, with 1.7's paired sample and its estimator's coverage on the pipeline | [I-75](known-issues.md#i-75) | — | M |
| 3.9 | API | *Parallel, can start now* (the HTTP layer and one dispatching entry point; no engine number moves). One run request in place of a route per engine function, as ORE runs a portfolio with a list of analytics: `POST /runs` with `market`, `portfolio`, `analytics` (`npv`, `exposure`, `sensitivities`, `market_risk`, `calibration`) and a `config` whose sections follow ORE's files, each setting in one place; one result with the configuration as run; `GET /config/defaults`; an idempotency key; `engine.run(RunRequest)` for Python callers, its code in a subpackage (3.10). The four routes and `/jobs` go, there being no client. The completeness test also checks each field's wiring. A run of one analytic equals today's direct call bit for bit ([design](details/configurable-engine.md#step-39--one-run-request-i-87-i-88-i-89)) | [I-87](known-issues.md#i-87), [I-88](known-issues.md#i-88), [I-89](known-issues.md#i-89) | 3.10 | M |
| 4.4 | Basel | *Parallel, start now.* Basel P0 (decisions D-1 to D-10, pinned text, the BCBS profile, the requirement catalogue and traceability check, measure guards, desk and book on every row, the run manifest, the remaining ORE oracles) and P2's data acquisition, which is calendar time. P0.9 (regulatory runs kept by the job queue) waits for 4.2; the rest of P0 does not | [F-05](features.md#f-05) (P0, P2) | P0.9: 4.2 | L |
| 3.7 | Precision | Kernels in difference form with explicit accumulators, one family at a time, one implementation for every precision (A-16); compute below float32 enabled. The families 3.4 and 3.5 do not change come first (simulation scan, scenario curves, legs, Europeans, exposure), so 5.1 starts on them; the Bermudan/American rollback and recalibration come last, after 3.4 and 3.5, so no kernel is rewritten twice. Where a family can be a matrix product (leg pricing, the Bermudan rollback), it takes that form in the same rewrite, so native FP8 (5.2) needs no second one. A product's own precision (TensorFloat-32, bfloat16 passes, FP8) becomes a compute format of the policy, named in the report: 2.3's product helper (`engine.precision.product_precision`) maps the policy's format to it, so every product follows the policy and never a device's default. Where a device has no such unit (TensorFloat-32 on a CPU, which computes float32 instead) the format is emulated by rounding the operands to its mantissa, as FP8 storage is, and the report says which ran ([details/precision.md §8.3](details/precision.md#83-emulation-and-native-speed)). Per family, ORE parity at existing tolerances first, then the float64 snapshot re-baselined. The rollback's matrix form also holds one operator per step instead of one per path and step, so the demo's job runs at 65,536 and 262,144 paths | [F-07](features.md#f-07) (compute), [I-83](known-issues.md#i-83) | 3.6; the Bermudan/American family 3.4, 3.5 | L |
| 5.1 | Precision, Basel | The evidence table against the acceptance standard (A-11: Basel III's P&L attribution test and the Basel plan's P6.2 rule; P6.2 labelled as engineering for figures Basel does not cover), from 3.6's harness on each family of 3.7 as it lands (the Bermudan/American rows after 3.4 and 3.5), then on each kernel stage 4 adds; a warning on any result whose combination has no passing row | [I-55](known-issues.md#i-55) (warnings) | Each family of 3.7 | S |
| 3.4 | Precision, Basel | `ShiftHorizon`: the setting and its API field exist since 3.1 and refuse anything but 0; implement the shift, parity at 0.5 against the LGM oracle, then 0.5 as the default | [I-32](known-issues.md#i-32) | — | M |
| 3.5 | Precision, Basel | Reproduce ORE's two per-path recalibration details against `tests/test_ore_xva_parity.py`, until its Bermudan/American L3 cases (strict expected failures today, 0.5–4% apart) pass; confirm an American's basket on a path against ORE's (decision A-7); warn, as ORE's `LgmBuilder` does, when a path's recalibration misses its basket | [I-49](known-issues.md#i-49), [I-73](known-issues.md#i-73) | — | M |
| 3.8 | Precision | *Parallel, can start now* (it moves no kernel's arithmetic). One worker per host on a Cloud TPU pod slice, with process 0 claiming each job and handing it to the other hosts, since under SPMD every host runs the same job ([details/precision.md §11.3](details/precision.md#113-multi-device-and-multi-host)). Measure wall time against device count on TPU and H100. The one-host split of the scenario axis is done (2026-10-04) | [I-61](known-issues.md#i-61) | — | M |
| 5.2 | Precision | The research result: on several devices of Ironwood and H100, wall time per figure at equal accuracy, many low-precision paths against fewer float64 paths, read from the evidence table's path ceilings; the storage formats, then 3.7's matrix-product kernels on native FP8 | [F-07](features.md#f-07) (speed) | 3.8, 5.1 | M |
| 4.5 | Basel | *Parallel, can start now.* Sensitivities against ORE's sensitivity analytic on the shared portfolio; the AD Greeks against the bump Greeks on the same sloped portfolio. FRTB-SA's sensitivities rest on it | [I-51](known-issues.md#i-51), [I-78](known-issues.md#i-78) | — | M |
| 4.2 | API, Basel | The job queue: EOD accepted-attempt record, boot sweep, `interrupted` state (portfolio jobs have had all three since 1.8); a retention policy that keeps regulatory runs; a job status that says when the engine worker cannot start; a cancel route (a pending job at once, a running one at the worker's next phase boundary) | [I-08](known-issues.md#i-08), [I-76](known-issues.md#i-76), [I-77](known-issues.md#i-77), [I-91](known-issues.md#i-91) | 3.8, 3.9 | M |
| 4.11 | Basel | Basel P1 (FRTB-SA) onward, phase by phase of the [Basel plan](details/basel-iii.md#6-phases-and-tasks): historical scenarios and IMA expected shortfall, backtesting and PLA, SA-CCR and BA-CVA, the precision gate, the evidence pack. Each figure an entry of 3.9's `analytics` | [F-05](features.md#f-05) | 4.4's P0, 4.5 (USD swaps also I-05); P5's IMM 3.5; P6's precision gate 5.1; P7's evidence pack 6.1 | L |
| 5.3 | Precision | FP4: storage with a variance correction through the block scales, compute on TPU 8t/8i; multilevel estimation of quantiles (PFE, VaR, ES) | [F-07](features.md#f-07) (FP4) | 3.7, 5.2 | L |

Steps 3.1 (one API reaching every setting) and 3.2 (the ORE simulation oracle) are done
(2026-10-07). From here every step that adds a setting (3.7's compute formats, stage 4's
options, trades and analytics) adds its API field once, and `tests/test_api_completeness.py`
holds it to that; and 3.5, 4.5, 4.7's two-currency test, 4.9, 4.10 and 4.11 prove themselves
against the ORE simulation oracle. 3.2 moved float64 exposure numbers towards ORE's (I-84,
I-85, I-86), with the golden snapshot re-baselined
([details/precision.md §13.1](details/precision.md#131-bit-for-bit-and-ore-parity)).

Order within the stage:

- **3.10 first.** It is small and moves only code, and every step after it adds code: 3.9 a
  run request and its entry point, 4.11 `engine/regulatory/`. Done first, nothing is moved
  twice.
- **Precision ahead where steps compete.** 3.6 builds the harness, 3.7 rewrites the kernels it
  measures, and 5.1 reads the table off each family as it lands, so swaps, Europeans and bonds
  have evidence rows while 3.4 and 3.5 make the Bermudan/American kernel ORE's. The rollback
  family is then rewritten once, after them, and its rows follow. 3.8 can run at any time
  before 5.2, which needs it and the table. FP4 (5.3) comes last, being F-07's stretch goal.
- **3.9 before any analytic is added.** It changes the HTTP contract, so it comes before the
  analytics 4.5 and 4.11 add here and 4.9 and 4.10 add in stage 4, which then arrive as
  entries of the `analytics` list rather than as routes. A setting added before it gets its
  API field in the request shape of the day, and 3.9 carries it over.
- **Basel alongside, from now.** 4.4 starts at once, because P2's data is calendar time and P0
  fixes the text and decisions every figure cites. 4.5 comes before 4.11, because FRTB-SA's
  sensitivities must first be shown to be ORE's. 4.11 then follows the Basel plan's phases,
  each waiting only for what its row lists: P5's IMM for 3.5 (a Bermudan's exposure equal to
  ORE's), P6's precision gate for 5.1, which is one table shared with the precision research.
- **3.4 and 3.5 as blockers, not as defects.** They are ORE parity fixes, and would otherwise
  wait for stage 5; they are here because the Bermudan/American kernel family, its evidence
  rows and Basel's IMM exposure need them first.
- **3.8 before 4.2.** 3.8 needs a pod slice, and it has to finish before 4.2, which changes
  the same worker loop; 4.2 in turn comes before Basel P0.9, which keeps regulatory runs in
  the job queue.
- **Numbers.** 3.10, 3.6, 3.8, 3.9 and 4.2 move no float64 number. 3.4 and 3.5 move default
  Bermudan/American numbers towards ORE's. 3.7 moves float64 at rounding level, each family
  once, with the snapshot re-baselined after each.

Parity methodology: [details/ore-parity-validation.md](details/ore-parity-validation.md);
precision: [details/precision.md](details/precision.md); Basel III:
[details/basel-iii.md](details/basel-iii.md).

<a id="stage-4--completeness"></a>
## Stage 4 — Completeness

Everything else in scope that the engine does not do yet. A step that adds a path-pricing
kernel (4.6, 4.7, 4.8, 4.10) writes it in 3.7's form, one implementation for every precision
(A-16), and gets its rows in 5.1's table.

| Step | Work | Closes | After | Size |
|---|---|---|---|---|
| 3.3 | Swaption vol strike axis, read at each option's and helper's strike: an additive market field, with ATM-only markets bit for bit. The lookup it widens is written in 3.7's form | [I-54](known-issues.md#i-54) | 3.7 | M |
| 4.6 | Engine options: ORE's `AnalyticLgm` European engine, settlement methods, FD solver (the AD Greeks method and the market-risk engine by configuration are done) | [F-01](features.md#f-01) | 3.7; the FD solver also 3.4 | M |
| 4.7 | FX and equity trades on the market path; FX/EQ calibration; the two-currency end-to-end test (L6) against the ORE simulation oracle, extended to a second currency | [F-04](features.md#f-04) | 3.7 | L |
| 4.8 | SABR volatility | [F-02](features.md#f-02) | 3.3, 3.7 | M |
| 4.9 | CVA/DVA | [F-06](features.md#f-06) | 3.9 | M |
| 4.10 | AMC engine | [F-03](features.md#f-03) | 3.7 | L |
| 4.12 | Reporting currencies other than USD at the EOD boundary: convert at the as-of FX spot, as ORE reports in its `baseCurrency`; parity against an ORE run with that base currency | [F-08](features.md#f-08) | 4.1; an FX source ([I-18](known-issues.md#i-18), waiting on others) | S |
| 4.13 | ORE's XML input files (`ore.xml`'s analytics, `simulation.xml`, `pricingengine.xml`, `sensitivity.xml`, `portfolio.xml`, zero-rate market data) translated into a run request, refusing by name what the engine does not support. Nice to have, not critical (owner, 2026-10-08) | [F-09](features.md#f-09) | 3.9 | M |

Order within the stage: 3.3 first (no market given to the engine so far has a smile, so it
changes no number yet, but 4.8 builds on it); then the smallest gaps in ORE's own choices
(4.6), the largest scope gap (4.7: the model already simulates FX and equity, but nothing
prices them), SABR (4.8), and the analytics built on proven exposure (4.9, 4.10). 4.12 as soon
as its FX source arrives; 4.13 last, since nothing waits on it.

<a id="stage-5--local-defects-and-hardening"></a><a id="stage-6--hardening"></a>
## Stage 5 — Local defects and hardening

Defects whose fix is small and local, and the tests and tooling. They matter, and they wait
until stage 3's core works, except where the step says otherwise.

| Step | Work | Closes | Size |
|---|---|---|---|
| 4.1 | EOD boundary: check submission binding before any cache return; one execution owner per workload; reject unknown calculations and non-USD reporting currency (A-18, as ORE rejects an unknown analytic). I-57 is High (one caller can be served another's result), so this goes first, ahead of the core, as soon as TraderX calls the EOD routes for real | [I-57](known-issues.md#i-57), [I-58](known-issues.md#i-58), [I-59](known-issues.md#i-59) | S |
| 4.3 | Chase TraderX's answers; apply them (a widened allowlist, a schema statement). When they arrive | [I-23](known-issues.md#i-23), [I-60](known-issues.md#i-60) | S |
| 6.1 | Characterize the full-suite aborts and lost worker processes. The fast tier's CI deaths were memory and are fixed (2026-10-05: the grid-convergence test's grid, compiled programs dropped per module). Still to do: show the full suite on CI's 16 GB runner (`-n 4` peaked at 24.2 GB with page cache before those fixes), then repeated full runs against a known-bad baseline. Before Basel P7 (4.11), whose evidence pack rests on a complete suite run | [I-27](known-issues.md#i-27) | M |
| 6.2 | *Parallel, start now* (A-20). Ruff in `pyproject.toml` and CI with pyflakes' rules only (ruff's `F` set: unused imports and variables, undefined names, empty f-strings; no style or formatting rules). Fix or mark each finding; an "unused" import that other modules import from there is a re-export and goes in `__all__`, checked one by one. Earlier is cheaper: every later step's code is then written lint-clean once | [I-66](known-issues.md#i-66) (linter) | S |
| 6.3 | Shared test helpers in `tests/support/`; public-entry tests where stage 1 made private-symbol tests obsolete | [I-67](known-issues.md#i-67) | S |
| 6.4 | A type checker on `engine/` (A-20) | [I-66](known-issues.md#i-66) (type checker) | S |
| 6.5 | The cold job's one-operation programs: jit the scenario market's construction and the other eager sites, and market risk's batch on the price function as data, as 2.4 did for the AD Greeks; the golden snapshot's change per array shown | [I-81](known-issues.md#i-81) | S |
| 6.6 | What roadmap 2.5 left of the pipeline's speed: find why, on Windows' CPU only, pricing runs 10–25% slower in a process that has run AD Greeks (the Windows heap is suspected, not shown; Linux is unaffected); and win back the 3 s (8%) a cold job lost to the bootstrap's compile under Newton, by tracing the helper's price once per bucket program where it is traced at the bracket's ends, the step and the model value today, with the snapshot bit for bit | [I-53](known-issues.md#i-53) | S |
| 6.7 | Before the API is reachable from outside a trusted network: authentication chosen with TraderX, a body-size limit, a per-caller limit on queued jobs | [I-90](known-issues.md#i-90) | S |

6.1's runs can be made whenever a full run is being made anyway. 2.2's five full runs under
WSL2 (`-n 4`, a 23 GB VM; four on the GPU, one on CPU) all printed their summary lines, the
last of each green; see the verification status. Only the conclusion waits for repeated runs.

<a id="waiting-on-others"></a>
## Waiting on others

Not engineering work until the input arrives. Chase the dependency, not the code.

| Item | Waiting for | From |
|---|---|---|
| [I-05](known-issues.md#i-05) USD-SOFR swaps | The D03/D04 convention set | TraderX |
| [I-04](known-issues.md#i-04) seasoned TraderX swaps | `pastFixings` in the export | TraderX |
| [I-16](known-issues.md#i-16) per-pillar `rateSensitivity` | An observed market-data package (W2) | TraderX |
| [I-18](known-issues.md#i-18) equity positions; [F-08](features.md#f-08) reporting currencies (step 4.12) | A spot/FX source | TraderX or a market-data decision |
| [I-23](known-issues.md#i-23), [I-60](known-issues.md#i-60) | Answers on versioning and added fields | TraderX (then step 4.3) |
| [I-07](known-issues.md#i-07) corporate bonds, listed options | Demand, and for corporates a credit model | Owner |

Open TraderX asks are collected in
[details/traderx-integration.md](details/traderx-integration.md#open-with-traderx).
