# Roadmap

The order in which to work through [known-issues.md](known-issues.md) and
[features.md](features.md). Every open item appears here exactly once; an item that is not
here is not planned ([rules](README.md#lifecycle)).

**How the order is decided** (decision A-19, 2026-10-05). A complete, working system comes first;
defects and inefficiencies that do not stop it working come after.

1. **The near-term milestone**: `demos/demo_profile_small.py` runs on the owner's local GPU
   and its profiler trace covers the whole job.
2. **Foundations**: changes that reshape what later work builds on (the request and result
   shapes, the ORE reference, the market's volatility shape, the Bermudan/American engine's
   semantics, how values are stored, the kernels' form), so that no later step is written
   twice.
3. **Completeness**: every capability in scope that the engine does not have yet, in the order
   its dependencies allow.
4. **Precision research results**, on the finished kernels.
5. **Hardening**: the remaining defects, tests and tooling.

Within a stage, work in numbered order. Items marked *parallel* touch code that no earlier
step changes, and can start at any time. Items waiting on someone outside the project are
listed [separately](#waiting-on-others), with who to chase. A defect found along the way goes
to stage 6, unless it is High (a wrong number on an ordinary input, or another caller's
result served) or blocks a step, in which case it goes before the step it blocks.

**Every step keeps the project's goals.** Nothing above trades one away:

- **ORE parity.** Every ORE parity suite passes at its existing tolerance after every step. A
  new capability comes with its own ORE parity test (through 3.2's oracle where it needs a
  simulation), or it is refused by name.
- **Accuracy.** float64 numbers stay bit for bit unless the step says it moves them. The steps
  planned to move them are 2.5 (rounding level: the default root solver), 3.4 and 3.5
  (default Bermudan/American values, towards ORE's) and 3.7 (rounding level, once). Such a step shows parity first, then re-baselines the golden
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
- **API compatibility.** Changes are additive. Every existing route, field, default and result
  shape keeps working, and old names become aliases, never removals. A new setting arrives
  with its API field, which 3.1's completeness test checks. Two narrowings are planned, both
  deliberate: 4.1 refuses inputs that are now accepted and then ignored (the old answer is
  wrong), and 3.4 changes a default number to ORE's (decision A-3).
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
deterministic kernels on any accelerator. Nothing runs on more than one host.
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
## Stage 2 — The demo on a local GPU, with a whole trace

`demos/demo_profile_small.py` runs one of every trade type through calibration, simulation,
pricing, exposure and AD Greeks, over the HTTP API, in the engine worker, under
`jax.profiler.trace`. The milestone: it runs on the owner's GPU (an RTX 5060 Laptop GPU,
Blackwell, 8 GB, on Windows), and its trace covers the whole job and labels every phase, on
CPU and GPU alike.

| Step | Work | Closes | Size |
|---|---|---|---|
| 2.3 | Process settings where they belong (decision A-22): importing `engine` changes no process-wide JAX or XLA setting but x64. Roadmap 2.2 put three device settings in `engine/__init__.py`, set on import unless already set ([I-80](known-issues.md#i-80) gives the measurements behind each); this step moves each to the process that owns it, and `engine/__init__.py` goes back to x64 only. **GPU preallocation** (`XLA_PYTHON_CLIENT_PREALLOCATE=false`; XLA's default takes 75% of a GPU in every process that opens it): set, unless the environment sets it, only where several processes share a GPU: the server environment of the demos that start one (`demo_profile_small.py`, `demo_structured.py`, `demo_api.py`, whose three copies of `start_server` become one helper in `demos/`) and `tests/conftest.py`. A deployment with one worker per GPU keeps JAX's default. **Matrix-product precision**: `jax_default_matmul_precision` is left alone. One helper in `engine/precision/` (for example `dot(a, b)`) multiplies at the precision of its operands' compute format, full precision (`jax.lax.Precision.HIGHEST`) for float32 and float64 (step 3.7 adds the policy's own product formats). Every matrix product of the engine's JAX code goes through it, nine today: `engine/simulation/cam.py`'s simulation step (2), `engine/instruments/bermudan_swaption.py`'s rollback bookkeeping (4), `engine/calibration/ore_lgm.py`'s spread correction (2) and `engine/simulation/random.py`'s Brownian bridge (1); NumPy products on the host are untouched. A test traces the pipelines (simulation, the scenario market, every product's path pricing, calibration, AD Greeks, market risk) and fails on any `dot_general` without an explicit precision, the transposes AD generates included. **Deterministic GPU kernels**: the engine worker appends `--xla_gpu_exclude_nondeterministic_ops=true` to its own `XLA_FLAGS` at start-up, before JAX starts and beside its compilation-cache environment (`engine.api.worker.compilation_cache_environment`), unless the flags name `deterministic_ops` in either form; `tests/conftest.py` does the same, since the tests price in-process. (`--xla_gpu_deterministic_ops=true` also pins autotuning, at twice the compile time measured in 2.2; an operator may choose it.) A library user who wants reproducible GPU bits sets the flag. **The convention, documented**: in `docs/concepts/coding-style.md`'s core constraints, no bare `@`, `jnp.matmul`, `jnp.dot`, `jnp.einsum` or `jnp.tensordot` on JAX arrays in engine code, the helper instead, why (a device's default product precision is below float32 and would make a float32 policy's report false), and the test that enforces it; its "Vectorization over loops" advice to use `jnp.einsum` amended to match. Also updated: [details/precision.md](details/precision.md) §6 (the mechanism) and §8.3, the user guide's "On a GPU" and "Precision" sections (what the engine sets and what a deployment, test run or notebook sets itself), `docs/concepts/architecture.md`, `docs/reference/http-api.md`, the worker's and `engine/__init__.py`'s docstrings. **Tests**: `tests/test_accelerator_defaults.py` rewritten: importing `engine` sets no device variable; the worker's environment carries the determinism flag, keeps other flags and yields to an explicit one; a demo's server environment turns preallocation off and yields to an explicit setting; the product-precision test; the accelerator-only tests stay (a float32 product through the engine is float32; the AD Greeks repeat bit for bit). **Proof**: the golden snapshot bit for bit on CPU (a product's precision does nothing to float64, nor to float32 on a CPU); the full suite on Windows, Linux CPU and the GPU; the demo on the GPU. No number, ORE parity tolerance or speed changes against today: the same product precision, applied per product instead of process-wide | [I-80](known-issues.md#i-80) | S |
| 2.4 | Make the demo's trace useful on both backends, by fixing what 2.1 and 2.2 found. 2.1 (CPU, 2026-10-05; [profiling §2.0](../concepts/profiling.md#20-the-demo-measured-2026-10-05-roadmap-21)): the trace is already whole in xprof in every mode, and the job is 43 s from scratch, 19 s with the disk cache, 2.8 s repeated. What dominates: the options' AD Greeks compile (26 s of the 43 s), and the recalibration's bisection on every path date (`_bootstrap_bucket`, 251k of the trace's kernel events; 2.5); a repeated job still compiles 9 programs once (I-53). 2.2 (GPU, RTX 5060 under WSL2, same section): the trace is whole too and already has the GPU's own lanes (CUPTI works under WSL2; 1.37M kernel events on the compute stream), but tracing is the cost: a repeated job is about 6 s untraced and 35.7 s traced, and a cold trace holds 6M events. Untraced the job is 89 s from scratch and 6–9 s repeated, slower than the CPU's 44 s and 2.4 s; the Bermudan's and American's Greeks are 29 s of a traced repeat (0.36 s on the CPU), likely their loops' kernels launched one by one with little float64 work each (1/64 of the card's float32 rate, 256 paths), which the device lane will show. Cut the profiler's cost on the GPU (which CUPTI activities are recorded, or a traced window of the job), the first-call compile time of the options' AD Greeks, and the repeated job's 9 compiles. Read from the device lane what share of the job the recalibration is, for 2.5's baseline. A cold trace's events also exceed the `.trace.json.gz` export (~1M), which only xprof-less viewers need. Then re-measure on CPU and GPU with the demo's summary, and update the demo's docstring, [profiling.md](../concepts/profiling.md)'s tables and the user guide | [I-53](known-issues.md#i-53) (compiles, trace) | M |
| 2.5 | A configurable root solver for every calibration and exercise boundary (decision A-21). **Baseline first**: the job untraced against path count (256, 4k, 64k, 256k) on CPU and GPU, and 2.4's device-lane share of the recalibration. **One module**, `engine/numerics/roots.py`, replacing the five bisections (`_bootstrap_bucket` and `_solve_monotone_root` in `engine/calibration/ore_lgm.py`, `engine/valuation/jamshidian.py`, `engine/calibration/lgm.py`, `engine/calibration/basket.py`), with two solvers: `"Newton"`, the default, a safeguarded Newton method (the derivative by `jax.jvp`, a bisection step wherever Newton would leave the bracket, each y* started from the last one found), and `"Bisection"`, today's, the reference, bit for bit. Both run a fixed number of steps on every backend (no data-dependent stop, which on a GPU reports to the host each step); Newton's count is set from its measured convergence, with a test that every root's residual is at float64 rounding on the parity suites' markets. **The setting**, `solver`, on every configuration that solves a root (the Bermudan/American engine, `LgmSwaptionEngineConfig`; the CAM's interest-rate models' calibration; the Jamshidian European engine), with its API field. A market value out of reach in the bracket is refused under either solver, as today. **The dates batched**: `bermudan_cube`'s loop over path dates becomes one program per basket shape over dates × paths. Jitted with the configuration static, so no compile per date, trade or call (compile-count tests), under the scenario sharding and strict promotion, and in 3.7's form where it already applies (one implementation for every precision). **Proof**: every ORE parity suite at its existing tolerance under both solvers (ORE's own solvers stop at 1e-6 for y* and 1e-8 per bucket, so the gap to ORE is ORE's); the two solvers agree to about 1e-14 on every parity market; `"Bisection"` reproduces the golden snapshot bit for bit; then the snapshot is re-baselined for the `"Newton"` default, with the largest change per array recorded. **Then** re-measure the baseline, old and new interleaved, and update profiling.md, the user guide and the solver's reference docs | [I-53](known-issues.md#i-53) (recalibration) | M |

Order within the stage: 2.1 (measuring the demo on CPU) and 2.2 (running it on the GPU) are
done (2026-10-05). 2.3 comes first, so that 2.4 and 2.5 measure under the settings the engine
keeps; 2.4 then fixes the trace and the compiles and measures what 2.5 starts from. 2.2 changed
no kernel: it set how processes hold device memory and three process defaults, and the golden
snapshot is bit for bit on CPU
([details/precision.md §13.1](details/precision.md#131-bit-for-bit-and-ore-parity)). 2.3 moves
those settings without changing a number, and 2.4 changes when work is dispatched, never the
arithmetic, so float64 numbers stay bit for bit (golden snapshot, on CPU). 2.5 moves the
default float64 numbers at rounding level, once (the reference solver keeps them bit for
bit); it comes before stage 3 so that 3.5's and 3.7's recalibration are written on the shared
solver, and stage 3 rewrites nothing that 2.3 to 2.5 did. The
card's float64 runs at 1/64 of its float32 rate, so it is where correctness and the trace
are checked, not where speed is measured (5.2).

<a id="stage-3--foundations"></a>
## Stage 3 — Foundations

Changes that reshape what later steps build on. They come before stage 4 so that every
feature is written once, against its final request shape, market, engine semantics and kernel
form.

| Step | Work | Closes | Size |
|---|---|---|---|
| 3.1 | One API (A-2). One route, with the old names as aliases (the request is already one shape since 1.3). A market-risk route and the CAM calibration route. A completeness test that fails on any configuration setting without an API field. Every per-trade result row carries its trade id, beside the positional fields, which stay. The cube can be returned as a chunked artifact reference, or left out, on request; inline stays the default (A-17, as ORE writes its cube only when asked) | [I-56](known-issues.md#i-56), [I-10](known-issues.md#i-10) (results), [I-09](known-issues.md#i-09) | L |
| 3.2 | *Parallel with 3.1* (test side only). Generalize the oracle to an OREApp XVA run; L4 distribution parity of exposure profiles; L3 path parity once gate V-4 closes. Fix the oracle's first-segment curve while in that file | [I-50](known-issues.md#i-50), [I-34](known-issues.md#i-34) | L |
| 3.3 | Swaption vol strike axis, read at each option's and helper's strike: an additive market field, with ATM-only markets bit for bit | [I-54](known-issues.md#i-54) | M |
| 3.4 | `ShiftHorizon` as a setting, with its API field; parity at 0.5 against the LGM oracle; then 0.5 as the default | [I-32](known-issues.md#i-32) | M |
| 3.5 | Reproduce ORE's two per-path recalibration details, measured against 3.2's cube; confirm an American's basket on a path against ORE's (decision A-7); warn, as ORE's `LgmBuilder` does, when a path's recalibration misses its basket | [I-49](known-issues.md#i-49), [I-73](known-issues.md#i-73) | M |
| 3.6 | *Parallel, can start now* (it changes no float64 number). Store classes whose level swamps their spread relative to a level (the cube to its t=0 value, the curves to their path-independent part, or a block offset). Build a measurement harness, rerunnable on any kernel change, for the storage formats per class, product and path count, with 1.7's paired sample and its estimator's coverage on the pipeline | [I-75](known-issues.md#i-75) | M |
| 3.7 | Kernels in difference form with explicit accumulators, one family at a time (simulation scan, scenario curves, legs, Europeans, Bermudan rollback and recalibration, exposure), one implementation for every precision (A-16); compute below float32 enabled. Where a family can be a matrix product (leg pricing, the Bermudan rollback), it takes that form in the same rewrite, so native FP8 (5.2) needs no second one. A product's own precision (TensorFloat-32, bfloat16 passes, FP8) becomes a compute format of the policy, named in the report: 2.3's product helper maps the policy's format to it, so every product follows the policy and never a device's default. Where a device has no such unit (TensorFloat-32 on a CPU, which computes float32 instead) the format is emulated by rounding the operands to its mantissa, as FP8 storage is, and the report says which ran ([details/precision.md §8.3](details/precision.md#83-emulation-and-native-speed)). ORE parity at existing tolerances first, then the float64 snapshot re-baselined once | [F-07](features.md#f-07) (compute) | L |
| 3.8 | *Parallel, can start now* (it moves no kernel's arithmetic). One worker per host on a Cloud TPU pod slice, with process 0 claiming each job and handing it to the other hosts, since under SPMD every host runs the same job ([details/precision.md §11.3](details/precision.md#113-multi-device-and-multi-host)). Measure wall time against device count on TPU and H100. The one-host split of the scenario axis is done (2026-10-04) | [I-61](known-issues.md#i-61) | M |

Order within the stage:

- **3.1 first.** Every later step that adds a setting (3.3's strikes, 3.4's shift, stage 4's
  options, trades and analytics) then adds its API field once, and the completeness test holds
  it to that.
- **3.2 alongside 3.1.** It is the reference that 3.5, 4.5, 4.7's two-currency test, 4.9 and
  4.10 prove themselves against.
- **3.3 to 3.5 before 3.7.** They change what the European and Bermudan/American kernels
  compute: the volatility a helper reads, the state grid, and the recalibration's basket and
  time grid. 3.6 changes what is stored and builds the harness that measures 3.7. 3.7 comes
  after all of them so that no kernel is rewritten twice, and before any stage 4 step adds a
  kernel (A-16).
- **Numbers.** 3.3 changes no number on any market given to the engine so far (all are
  ATM-only). 3.4 and 3.5 move default Bermudan/American numbers towards ORE's. 3.7 moves
  float64 at rounding level, with one re-baseline after it.
- **3.8 before 4.2.** 3.8 needs a pod slice, and it has to finish before 4.2, which changes
  the same worker loop.

Parity methodology: [details/ore-parity-validation.md](details/ore-parity-validation.md);
precision: [details/precision.md](details/precision.md).

<a id="stage-4--completeness"></a>
## Stage 4 — Completeness

Everything in scope that the engine does not do yet. A step that adds a path-pricing kernel
(4.6, 4.7, 4.8, 4.10) writes it in 3.7's form, one implementation for every precision (A-16).

| Step | Work | Closes | After | Size |
|---|---|---|---|---|
| 4.1 | *Parallel, can start now* (the EOD routes only). EOD boundary: check submission binding before any cache return; one execution owner per workload; reject unknown calculations and non-USD reporting currency (A-18, as ORE rejects an unknown analytic). It is here rather than in stage 6 because the boundary can serve one caller another's result (I-57 is High), so it does not yet work | [I-57](known-issues.md#i-57), [I-58](known-issues.md#i-58), [I-59](known-issues.md#i-59) | — | S |
| 4.2 | EOD accepted-attempt record, boot sweep, `interrupted` state (portfolio jobs have had all three since 1.8); a retention policy for the job queue; a job status that says when the engine worker cannot start | [I-08](known-issues.md#i-08), [I-76](known-issues.md#i-76), [I-77](known-issues.md#i-77) | 3.8 | M |
| 4.3 | Chase TraderX's answers; apply them (a widened allowlist, a schema statement) | [I-23](known-issues.md#i-23), [I-60](known-issues.md#i-60) | Their answers | S |
| 4.4 | *Parallel, start now.* Basel P0 (decisions, pinned text, profile, traceability) and P2 data acquisition, which is calendar time | [F-05](features.md#f-05) (P0, P2) | — | L |
| 4.5 | Sensitivities against ORE's sensitivity analytic on the shared portfolio; the AD Greeks against the bump Greeks on the same sloped portfolio | [I-51](known-issues.md#i-51), [I-78](known-issues.md#i-78) | 3.2 | M |
| 4.6 | Engine options: ORE's `AnalyticLgm` European engine, settlement methods, FD solver (the AD Greeks method and the market-risk engine by configuration are done) | [F-01](features.md#f-01) | 3.7; the FD solver also 3.4 | M |
| 4.7 | FX and equity trades on the market path; FX/EQ calibration; the two-currency end-to-end test (L6) against 3.2's oracle | [F-04](features.md#f-04) | 3.2, 3.7 | L |
| 4.8 | SABR volatility | [F-02](features.md#f-02) | 3.3, 3.7 | M |
| 4.9 | CVA/DVA | [F-06](features.md#f-06) | 3.2 | M |
| 4.10 | AMC engine | [F-03](features.md#f-03) | 3.2, 3.7 | L |
| 4.11 | Basel P1 (FRTB-SA) onward | [F-05](features.md#f-05) | 4.5 (USD swaps also I-05); P5's IMM 3.2; P6's precision gate 5.1 | L |
| 4.12 | Reporting currencies other than USD at the EOD boundary: convert at the as-of FX spot, as ORE reports in its `baseCurrency`; parity against an ORE run with that base currency | [F-08](features.md#f-08) | 4.1; an FX source ([I-18](known-issues.md#i-18), waiting on others) | S |

Order within the stage: 4.1, 4.3 and 4.4 can be done at any time; 4.12 as soon as its FX source arrives. 4.5 comes before 4.11,
because FRTB-SA's sensitivities must first be shown to be ORE's. 4.6 to 4.10 follow their
dependencies, starting with the smallest gaps in ORE's own choices (4.6), then the
largest scope gap (4.7: the model already simulates FX and equity, but nothing prices them),
then the analytics built on proven exposure (4.9, 4.10).

<a id="stage-5--precision-research"></a>
## Stage 5 — Precision research

| Step | Work | Closes | After | Size |
|---|---|---|---|---|
| 5.1 | The evidence table against the acceptance standard (A-11: Basel III's P&L attribution test and the Basel plan's P6.2 rule; P6.2 labelled as engineering for figures Basel does not cover), from 3.6's harness on 3.7's kernels, then on each kernel stage 4 adds as it lands; a warning on any result whose combination has no passing row | [I-55](known-issues.md#i-55) (warnings) | 3.4, 3.5, 3.7 | S |
| 5.2 | The research result: on several devices of Ironwood and H100, wall time per figure at equal accuracy, many low-precision paths against fewer float64 paths, read from the evidence table's path ceilings; the storage formats, then 3.7's matrix-product kernels on native FP8 | [F-07](features.md#f-07) (speed) | 3.8, 5.1 | M |
| 5.3 | FP4: storage with a variance correction through the block scales, compute on TPU 8t/8i; multilevel estimation of quantiles (PFE, VaR, ES) | [F-07](features.md#f-07) (FP4) | 3.7, 5.2 | L |

5.1 does not wait for stage 4: it starts once 3.7 is done and runs alongside it, because an
evidence table measured before 3.4, 3.5 and 3.7 would describe kernels that are about to change.
The local Blackwell card has FP8 and FP4 matrix units, so 5.2 and 5.3 can be rehearsed on it.
Its float64 rate makes its speed ratios unlike an H100's, so no result is drawn from it.

<a id="stage-6--hardening"></a>
## Stage 6 — Hardening

| Step | Work | Closes | Size |
|---|---|---|---|
| 6.1 | Characterize the full-suite aborts and lost worker processes. The fast tier's CI deaths were memory and are fixed (2026-10-05: the grid-convergence test's grid, compiled programs dropped per module). Still to do: show the full suite on CI's 16 GB runner (`-n 4` peaked at 24.2 GB with page cache before those fixes), then repeated full runs against a known-bad baseline | [I-27](known-issues.md#i-27) | M |
| 6.2 | *Parallel, start now* (A-20). Ruff in `pyproject.toml` and CI with pyflakes' rules only (ruff's `F` set: unused imports and variables, undefined names, empty f-strings; no style or formatting rules). Fix or mark each finding; an "unused" import that other modules import from there is a re-export and goes in `__all__`, checked one by one. Earlier is cheaper: every later step's code is then written lint-clean once | [I-66](known-issues.md#i-66) (linter) | S |
| 6.3 | Shared test helpers in `tests/support/`; public-entry tests where stage 1 made private-symbol tests obsolete | [I-67](known-issues.md#i-67) | S |
| 6.4 | A type checker on `engine/` (A-20) | [I-66](known-issues.md#i-66) (type checker) | S |

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
