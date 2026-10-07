# Profiling & the Tracer

How this engine is profiled, what the trace actually contains, why Greeks used to dominate
it, and what was done about that.

For the step-by-step "how do I collect and open a trace" recipe, see the User Guide's
[Profiling a pricing job](../getting-started/user-guide.md#profiling-a-pricing-job). This
document is the *why*.

## Plain-language summary

JAX does not run your Python code on the hardware. It **traces** it — runs it once
symbolically to record the operations — then compiles that recording into a single
optimized program that XLA executes. Compiling is expensive; executing the compiled
program is cheap. So the cost of a JAX job is mostly decided by **how many separate
programs get compiled**, not by how big the numbers in them are.

That distinction is the whole story here. This engine used to compile ~1,200 separate
programs for one small portfolio, most of them a single multiply or comparison, because
one function in the Bermudan pricer could not be compiled at all. Fixing that meant the
same job compiles a handful of programs instead. The trace — the recording the profiler
writes — shrank along with it, because the trace is mostly a record of compilation.

---

## 1. The tracer: what it is and how it works

There is no custom tracer in this codebase. The profiling hook is a thin, **opt-in**
wrapper around JAX's own [XProf](https://github.com/openxla/xprof) (TensorBoard-profiler)
integration, living in
[`engine/api/worker.py`](../../engine/api/worker.py)'s `_profiled`, around each job's
`price_portfolio` call:

```python
profile_dir = os.environ.get("JAX_RISK_PROFILE_DIR")
if not profile_dir:
    return run()                     # inert: does not even import jax

import jax
warmup = None
if os.environ.get("JAX_RISK_PROFILE_WARMUP") == "1":
    _, warmup = _measured(jax, run)            # untraced warm-up run: wall time, compiles

out_dir = os.path.join(profile_dir, f"pid-{os.getpid()}")
with jax.profiler.trace(out_dir, profiler_options=_profile_options(jax)):
    result, traced = _measured(jax, run)       # waits for device work inside the trace
_record_trace(out_dir, traced, warmup)         # <run>.summary.json, and a warning if partial
```

Five decisions define its behavior:

### 1.1 It traces inside the engine worker

The trace is taken where `price_portfolio` actually executes — the engine worker process
(roadmap 1.8), whose first job runs on cold JAX caches — not in the API handler. Output goes
to `$JAX_RISK_PROFILE_DIR/pid-<pid>/`, **one directory per process**, so a restarted worker
or a direct call in another process writes its own trace instead of a corrupted shared file.
(Until 1.8 jobs ran in a pool of workers, one trace directory each.)

### 1.2 Completely inert when unset

With `JAX_RISK_PROFILE_DIR` absent the hook does not run, does not allocate, and does not
even `import jax`. Every test and the entire CI HTTP path are byte-identical to a build
without the hook. `tests/test_profiling_and_jit.py::TestProfilerHook` pins this.

### 1.3 `python_tracer_level=0`, always

This is the most consequential setting in the file, and it overrides JAX's own default
of `1`. With the Python tracer on, the profiler instruments the **CPython interpreter**,
not just this engine's frames. Measured on the 4-trade demo portfolio:

| | Python tracer ON (JAX default) | OFF (this engine's default) |
|---|---|---|
| Events | 1,000,115 | — |
| …of which CPython frames | **971,080 (97.1%)** | 0 |
| Trace size | 467 MB | 50 MB |
| Wall time | 82.7 s | 45.4 s |
| **Job wall time covered** | **2%** | **93%** |

The top "hot" entries with it on were `isinstance` × 90,235, `append` × 33,156, `len` ×
21,073 — all from inside JAX's own dispatch machinery, none of them this engine's code.

Worse than the noise: the trace's `.trace.json.gz` keeps only the **~1M events that start
first**, with no warning. Those interpreter frames filled it during startup, so that file
covered only **the first 1.6 s of a ~90 s job**. (This was read as a cap on the profiler's
own buffer until roadmap 2.1 found the cap in the export: the `.xplane.pb` beside it, which
xprof reads, held all 1.5M events of a cold demo run. §5 is the check that now reports it.)

Turning it off loses exactly one thing — attribution of a dispatch back to the engine
function that issued it, since no event then carries a Python source file or line
(verified: 0 of 926,463 events had `source_file`/`source_line`/`long_name`). §3 is how
that attribution was bought back for ~0 cost.

`JAX_RISK_PROFILE_PYTHON_TRACER=1` opts back in, accepting the ~9x size, ~2x slowdown and
a `.trace.json.gz` that covers only the start of the job (§5).

### 1.4 Cold-start compilation is deliberately included

`JAX_RISK_PROFILE_WARMUP` defaults to **off**, so the traced run includes XLA lowering and
compilation. For this engine that is the point: compilation is a first-class cost. Set it
to `1` to run the job once and discard it first, so the traced run hits populated caches.

| Question | Setting |
|---|---|
| "What does this job cost from cold?" | warmup **off** (default) |
| "What does a *repeat* of this job cost?" | `JAX_RISK_PROFILE_WARMUP=1` |

**Warmup reduces compilation; it does not eliminate it.** Measured on the 4-trade demo
portfolio, three consecutive `price_portfolio` calls in one process:

| Run | Compilations | Wall |
|---|---:|---:|
| 1 (cold) | 208 | 26.5 s |
| 2 | **31** | 17.8 s |
| 3 | **31** | 16.0 s |

The 31 never go away, and compilation still visibly dominates a warm timeline (~27k MLIR
pass events against ~630 `ThunkExecutor::Execute`). Two distinct reasons:

1. **Those 31 are genuine recompiles.** `engine.risk.greeks` builds a fresh `price_fn`
   closure per call and `jax.jit` keys on function identity — the §3.5 residue, at
   portfolio scale rather than single-trade scale.
2. **Event count is not proportional to time.** 31 compilations of *large fused* programs
   emit far more trace events than 630 kernel executions over 256 scenarios.

So warmup answers "what does a steady-state repeat cost," not "show me execution only." A
genuinely execution-dominated timeline needs the closure-identity recompile fixed first
(done, §3.7: a repeated Greeks call now compiles nothing; these measurements predate it).
§2.2's warm half is the full lane/category breakdown, which quantifies exactly that: 79%
compilation against 1.6% arithmetic even after the cache is warm.
All 31 are enumerated with exact callsites, root cause and a vetted fix plan in
[Known Issues I-21 and I-22](../planning/known-issues.md#i-21) — two different mechanisms needing
two different fixes, which is why they are filed separately.

### 1.5 `block_until_ready` before the context exits

JAX dispatch is asynchronous. If the trace context closes while device work is still
queued, the timeline is truncated. `npv_cube` is the dominant device-side tail.

### 1.6 A phase window (roadmap 2.4)

`JAX_RISK_PROFILE_PHASE=<phase>` traces one phase of the job instead of all of it: a label of
`PHASES` (`calibration`, `simulation`, `pricing`, `exposure`, `greeks`, ...) or one trade's
Greeks (`greeks/trade3/AmericanSwaptionConfig`). The job runs whole; the worker arms
`engine.portfolio.profiling.traced_phase`, and `phase()` starts the profiler when that phase
is entered for the first time and stops it when the phase is left. Before starting and before
stopping it waits for every device to finish its queued work (`jax.live_arrays()`), since
dispatch is asynchronous: the window holds exactly that phase's device work, which an untraced
run would leave to the next phase to wait for. A name the job does not have traces nothing and
warns. The window's trace is written inside the job, so the job's own wall time includes it.

Why it exists: on a GPU the profiler's cost is per kernel launch, whatever it records (§2.0,
after 2.4), and grows with the trace's length, so a whole job traced runs several times slower
than untraced. A phase is traced at its own cost alone: the American's Greeks took 7.3 s
traced on their own and 22.5 s inside a whole-job trace. On the CPU, where tracing is cheap,
it narrows the timeline to what is being studied.

---

## 2. What the trace contains

### 2.0 The demo, measured (2026-10-05, roadmap 2.1)

`demos/demo_profile_small.py` as it stands: five trades (swap, European, Bermudan, American,
bill), 256 paths on 3 dates, the Hull-White model calibrated to a two-helper basket, AD
Greeks, on CPU (Windows, 24 threads), through the HTTP API and the engine worker. Each mode
was run in a fresh API and worker on a fresh queue, one run at a time, and read back from the
worker's summary (§5). Wall time is the traced `price_portfolio` call; compiles are XLA
programs the worker had to build or read back (`/jax/core/compile/backend_compile_duration`).

| Mode (demo switches) | Wall | Compiles | Events | JSON export | Size |
|---|---:|---:|---:|---|---:|
| Cold, no disk cache (`--cold --no-disk-cache`) | 43.1 s | 269 | 1,499,875 | partial | 122 MB |
| Cold, empty disk cache, which it fills (`--cold`, first run) | 46.7 s | 269 | 1,499,270 | partial | 122 MB |
| Cold, disk cache read back (`--cold`, a restarted worker) | 18.8 s | 269 | 916,387 | whole | 108 MB |
| Warm, the job repeated in the worker (default) | 2.8 s | 9 | 830,171 | whole | 105 MB |

A first run in the same series (before the summary read the `.xplane.pb`) gave 47.9 s, 20.4 s
and 2.8 s for the same modes, and 45.1 s without the disk cache.

**Every trace is whole.** In every mode the trace spans the whole run and holds every phase,
including each trade's Greeks. The profiler records everything in the `.xplane.pb`, and xprof
serves all of it (its `trace_viewer` returned 1,530,243 events for the cold trace). Only the
`.trace.json.gz` beside it is capped: it keeps the ~1,000,000 events that start first, so for
a cold trace a viewer reading that file (Perfetto, `chrome://tracing`) loses everything that
started after about 23 s, except the long regions that began earlier. The 2026-10-01 reading
"truncated at the event cap" was this file.

Time per phase (host time, so device work lands in the phase that waits for it):

| Phase | Cold, no disk cache | Disk cache | Warm |
|---|---:|---:|---:|
| calibration | 1.06 s | 0.41 s | 0.00 s |
| simulation | 1.78 s | 0.42 s | 0.01 s |
| pricing | 9.04 s | 3.58 s | 0.80 s |
| exposure | 0.71 s | 0.81 s | 1.09 s |
| greeks | 30.44 s | 13.59 s | 0.90 s |
| &nbsp;&nbsp;swap | 1.59 s | 0.65 s | 0.19 s |
| &nbsp;&nbsp;European | 2.24 s | 0.87 s | 0.27 s |
| &nbsp;&nbsp;Bermudan | 13.89 s | 6.82 s | 0.17 s |
| &nbsp;&nbsp;American | 11.85 s | 4.84 s | 0.19 s |
| &nbsp;&nbsp;bill | 0.62 s | 0.25 s | 0.01 s |

What the numbers say:

- **From scratch, compilation is the job.** 269 programs; the disk cache, reading them back,
  takes the job from 43 s to 19 s, and a repeat in the same worker to 2.8 s. The Bermudan's
  and American's AD Greeks are 26 s of the 43 s: two large differentiated programs each,
  mostly MLIR passes (`CSEPass`, `CanonicalizerPass`) on the compiler threads.
- **Events follow executed loop iterations, not programs.** About 624k events on
  `tf_XLAEigen` in every mode, warm included: XLA's CPU runtime records each op of a loop
  body on every iteration. By HLO module, 251,554 are `_bootstrap_bucket`, the 60-step
  bisection that recalibrates the Bermudan's and American's model on every path date
  (I-53), and 34,608 `_backward_induction_arrays`; 517k of the cold trace's events fall
  inside `pricing`. Compilation adds ~470k on `tf_xla-cpu-codegen`, which is what pushes a
  cold trace past the JSON export's cap.
- **A repeat is not yet compile-free.** The warm repeat compiles 9 programs: the swap's and
  European's AD Greeks (`legs_npv`, `black_multileg_npv` under the gradient and the
  Hessian-vector product) and one European Vega. A third run of the job compiles none, and a
  trade's Greeks repeated on their own compile none from the second call, so the cause is
  in how the first full job's tracing seeds JAX's caches, not a closure per call (I-53).
  Roadmap 2.4 found it (JAX's internal caches of 2,048 entries evicted those programs) and
  fixed it: a repeat compiles nothing (§3.8, and the re-measurement below).
- **On CPU, `exposure` is mostly waiting.** Dispatch is asynchronous; `exposure` is the first
  phase to read the cube, so it absorbs pricing's device time (1.09 s of a 2.8 s repeat).

#### On the GPU (2026-10-05, roadmap 2.2)

The same demo and modes on the owner's RTX 5060 Laptop GPU (Blackwell, 8 GB) under WSL2,
with JAX's CUDA 13 plugin and the engine's accelerator defaults (no preallocation, matrix
products at full precision, no non-deterministic kernels; set by `engine/__init__.py` then,
and since roadmap 2.3 by the demo's server environment, each product and the engine worker,
with the same effect). Each
mode was run once, in a fresh API and worker, one after another, from
`.venv/bin/python demos/demo_profile_small.py`:

| Mode (demo switches) | Wall | Compiles | Events | Size |
|---|---:|---:|---:|---:|
| Cold, no disk cache (`--cold --no-disk-cache`) | 118.8 s | 269 | 5,971,696 | 359 MB |
| Cold, empty disk cache, which it fills (`--cold`, first run) | 126.4 s | 269 | 5,969,631 | 359 MB |
| Cold, disk cache read back (`--cold`, a restarted worker) | 55.3 s | 269 | 3,961,501 | 304 MB |
| Warm, the job repeated in the worker (default) | 35.7 s | 9 | 3,860,704 | 302 MB |

| Phase | Cold, no disk cache | Disk cache | Warm |
|---|---:|---:|---:|
| calibration | 2.93 s | 0.87 s | 0.48 s |
| simulation | 2.94 s | 0.62 s | 0.06 s |
| pricing | 17.30 s | 7.31 s | 4.56 s |
| exposure | 0.69 s | 0.18 s | 0.09 s |
| greeks | 94.46 s | 46.07 s | 30.45 s |
| &nbsp;&nbsp;swap | 2.68 s | 0.64 s | 0.22 s |
| &nbsp;&nbsp;European | 3.31 s | 0.93 s | 0.26 s |
| &nbsp;&nbsp;Bermudan | 35.11 s | 14.71 s | 6.59 s |
| &nbsp;&nbsp;American | 51.45 s | 28.92 s | 22.82 s |
| &nbsp;&nbsp;bill | 1.08 s | 0.30 s | 0.03 s |

The same job untraced, `price_portfolio` called three times in one process with no disk cache
(two interleaved rounds): 89 s, then 8.4–9.1 s, then 6.1 s; on the CPU 44.0 s, 4.1 s and
2.4 s. The results equal the CPU's to the last bits (the cube to 9e-16 of its scale, the AD
Greeks to 2.3e-11 relative).

What the numbers say:

- **The trace has the GPU's own lanes.** CUPTI works under WSL2: 1.37M events on the compute
  stream (`Stream #14(Compute,MemcpyD2D,MemcpyH2D,Memset)`), the same in every mode, so these
  are the job's kernel executions; compiles add none. The job's host thread is `python`.
- **Tracing is the cost on the GPU.** A warm repeat is about 6 s untraced and 35.7 s traced;
  from scratch 89 s untraced and 119 s traced. On the CPU tracing cost little. A GPU trace
  measures where time goes, not how long a job takes.
- **The options' Greeks dominate even when warm.** The Bermudan's and American's AD Greeks are
  29 s of the 35.7 s traced repeat (0.36 s on the CPU). The likely reason, as read then
  before the device lane: their loops, the per-path-date bisection and the rollback, are many
  small kernels in sequence, which a CPU runs in-thread and a GPU launches one by one, and at
  256 paths in float64 (1/64 of this card's float32 rate) there is little arithmetic to hide
  a launch behind. Roadmap 2.4 measured it (below): most of the launches are the LGM
  bootstrap's, in the options' Greeks more than on the path dates, and 2.5 replaces the
  bisections' fixed 60- and 160-step loops with a configurable solver (decision A-21); the
  re-measurement after it is below.
- **The defaults' cost.** Excluding non-deterministic kernels costs nothing measurable warm;
  `--xla_gpu_deterministic_ops=true`, which also pins autotuning, doubled the compile from
  scratch (184 s against 89 s untraced) for the same bits, so it is an opt-in, not the
  default.

#### After roadmap 2.4 (2026-10-06)

2.4 made the AD Greeks one compiled program per product and derivative (§3.8) and added a
phase window to the profiler hook (§1.6). The same demo and modes, each in a fresh API and
worker, one run each, one after another; CPU on Windows (24 threads), GPU the RTX 5060 under
WSL2 (§2.0 above for the settings). Every mode compiles 217 programs where 2.1's compiled 269,
and the warm repeat compiles none where it compiled 9.

| Mode (demo switches) | CPU 2.1 | CPU 2.4 | GPU 2.2 | GPU 2.4 |
|---|---:|---:|---:|---:|
| Cold, no disk cache (`--cold --no-disk-cache`) | 43.1 s | **32.6 s** | 118.8 s | **102.6 s** |
| Cold, empty disk cache, which it fills (`--cold`, first run) | 46.7 s | 35.2 s | 126.4 s | 104.7 s |
| Cold, disk cache read back (`--cold`, a restarted worker) | 18.8 s | 16.1 s | 55.3 s | 47.2 s |
| Warm, the job repeated in the worker (default) | 2.8 s, 9 compiles | **1.8 s, 0** | 35.7 s, 9 compiles | **32.4 s, 0** |
| Warm, `--phase pricing` (the phase alone traced) | — | 1.5 s | — | 3.3 s |
| Warm, `--phase greeks/trade3/AmericanSwaptionConfig` | — | 0.1 s | — | 7.3 s |

Events: on the CPU 1.23M cold, 0.77M warm; on the GPU 5.3M cold, 3.8M warm, 1.0M for the
pricing phase alone. The phase window's run time includes writing its trace (the warm GPU job
with the pricing phase traced took 14.3 s against 32.4 s for the whole job traced).

Untraced, `price_portfolio` called three times in one process with no disk cache, the old tree
(a worktree of `2df78cb`) and the new run alternately, three pairs on Windows and Linux CPU,
two on the GPU:

| | Run 1 (cold) | Run 2 | Run 3 |
|---|---:|---:|---:|
| Windows CPU, old | 36.8–37.0 s | 2.95–2.98 s | 1.88–1.90 s |
| Windows CPU, new | **31.7–32.5 s** | **1.96–2.09 s** | 2.12–2.32 s |
| Linux CPU (WSL2), old | 74.2–80.7 s | 4.48–4.81 s | 2.38–2.45 s |
| Linux CPU (WSL2), new | **62.0–70.4 s** | **2.28–2.38 s** | 2.19–2.53 s |
| GPU, old | 75.6–76.1 s | 7.85–7.94 s | 5.57–5.61 s |
| GPU, new | **58.5–62.9 s** | **5.34–5.67 s** | 5.38–5.51 s |

From scratch the job is 14–22% faster, and the second run costs what the third does. The
steady state is unchanged on Linux and the GPU, and 0.2–0.4 s slower on Windows' CPU, where
any AD Greeks call (the old code's too) slows the pricing of every later job in the process,
the new single programs more than the old pieces: measured, pricing-only jobs before and after
one Greeks call, 1.60 → 1.89 s new and 1.49 → 1.71 s old; on Linux 1.32 → 1.32–1.38 s new. No
compile, trace or garbage collection is involved, and one large allocation does not do it
([I-53](../planning/known-issues.md#i-53)).

**What tracing costs on the GPU.** Measured on the warm repeat in process, the same job under
each profiler setting (untraced 5.8 s):

| Setting | Traced run | Writing the trace | Events | Size |
|---|---:|---:|---:|---:|
| Default (host tracer level 2, the GPU's CUPTI tracer on) | 33.3 s | 20.9 s | 3.85M | 303 MB |
| Host tracer level 1 | 32.8 s | 22.6 s | 3.81M | 300 MB |
| Host tracer off | 36.0 s | 16.2 s | 2.32M | 272 MB |
| No CUPTI callback events (`gpu_max_callback_api_events=0`) | 34.0 s | 18.8 s | 3.85M | 303 MB |
| No CUPTI activity events (`gpu_max_activity_api_events=0`) | 32.8 s | 8.3 s | 2.40M | 144 MB |
| Aggregated kernel records (`gpu_aggregated_tracing`) | 32.6 s | 5.4 s | 1.54M | 97 MB |
| Default, with XLA's command buffers kept during profiling (`--xla_enable_command_buffers_during_profiling=true`) | 50.3 s | 28.9 s | 3.85M | 304 MB |

No setting cuts the run: what a GPU trace costs is the tracer being attached at all, about
20 µs on every kernel launch (the repeat launches 1.36M kernels, about 4 µs each untraced),
and more the longer the trace runs. Options change only what is written: aggregated records
take the export from 21 s to 5 s, but replace each kernel's record by a summary, which loses
the device timeline. The engine keeps the defaults and narrows the window instead
(`JAX_RISK_PROFILE_PHASE`, §1.6). The JSON export's cap (~1M events) still cuts a whole-job or
a pricing trace short for viewers that read the `.trace.json.gz`; xprof reads all of it.

**The device lane** (the warm repeat's trace, kernel time on the compute stream, attributed to
the host phase whose region the kernel started in):

| | Kernel time | Kernels |
|---|---:|---:|
| Whole job (a 34.1 s traced span) | 2.37 s | 1,360,342 |
| … `_bootstrap_bucket` (the LGM bootstrap's bisection) | **1.99 s (84%)** | **1,322,350 (97%)** |
| … `_curve_derivatives`, `_option_vega` (the AD derivatives) | 0.22 s | 17,562 |
| … `_rollback_every_path` (the grid engine on the paths) | 0.14 s | 1,432 |
| `pricing` (the options recalibrated on every path date) | 0.72 s, of which 0.68 s the bootstrap | 339,829 |
| `greeks` (each option calibrated on today's, the sensitivity and the Theta markets, and its Vega Jacobian) | 1.54 s, of which 1.21 s the bootstrap | 939,683 |

The card is busy for 7% of the trace and the job is launch-bound: the recalibration is the
job on the GPU, by launches and by kernel time, and most of it is not on the path dates
but in the options' Greeks (61% of the bootstrap's kernel time, 73% of its launches), where each
Greek calibrates the trade's LGM again. That is 2.5's
baseline: a Newton solver cuts every one of these loops; batching the path dates
(`bermudan_cube`) cuts only the third in `pricing`.

#### After roadmap 2.5 (2026-10-07)

2.5 solves every calibration and exercise boundary with one solver (`engine.numerics.roots`,
decision A-21): a safeguarded Newton method by default, 7 steps per bootstrap bucket and 5 per
y\*, where the bisection took 60 and 100 (plus 60 widening steps for each y\*); and it
calibrates a Bermudan's or American's path dates of one basket shape in one call. Measured
untraced in fresh processes with no disk cache, the old tree (a worktree of `eadbcc6`, 2.4) and
the new one alternately, two pairs each; Windows CPU (24 threads) and the RTX 5060 under WSL2.

**The recalibration alone** (the demo's Bermudan and American on every path date, the demo's
simulation at each path count; the second and third call of a process):

| Paths | CPU old | CPU new | GPU old | GPU new |
|---:|---:|---:|---:|---:|
| 256 | 0.08–0.10 s | **0.010 s** | 0.59–0.67 s | **0.018–0.022 s** |
| 4,096 | 3.35–3.52 s | **0.068–0.071 s** | 0.41–0.42 s | **0.020–0.024 s** |
| 65,536 | 27–30 s | **0.53–0.90 s** | 2.13 s | **0.06–0.14 s** |
| 262,144 | 144–185 s | **2.8–3.4 s** | 7.3 s | **0.19 s** |

35 to 55 times faster on the CPU, where the cost was arithmetic (9,600 dependent loop
iterations per bucket and date), and 15 to 38 times on the GPU, where it was kernel launches.
The first call compiles: 3.2–5.9 s on the CPU against 1.7–4.2 s before (and 187 s at 256k
paths, which the old code spent computing), 4.9–6.4 s on the GPU against 3.2–10.2 s.

**The demo's job** (`demos/demo_profile_small.py`'s request, five trades with AD Greeks; run 1
compiles, run 2 repeats; "pricing" is the same request without Greeks, after them):

| | Paths | Run 1 (cold) | Run 2 | Pricing |
|---|---:|---:|---:|---:|
| CPU, old | 256 | 37.1–38.0 s | 2.32–2.33 s | 1.97–2.37 s |
| CPU, new | 256 | 40.9–41.3 s | 2.15–2.29 s | 1.92–2.17 s |
| CPU, old | 4,096 | 67.8–77.1 s | 39.6–43.3 s | 38.5–43.2 s |
| CPU, new | 4,096 | 74.3–76.6 s | 37.5–38.7 s | 35.8–37.9 s |
| GPU, old | 256 | 60.8–62.9 s | 5.42–5.55 s | 1.89–1.99 s |
| GPU, new | 256 | 64.0–64.4 s | **1.35–1.43 s** | **0.37–0.40 s** |

On the GPU a repeated job is four times faster and its pricing five: the recalibration was
2.4's device lane's 84% of kernel time and 97% of launches, in the options' Greeks (their
calibration today, on the sensitivity markets and the Theta market) and on the path dates. On
the CPU the repeat is unchanged at 256 paths and 3–4 s faster at 4,096: there the job is now the
grid rollback on the paths (at 4,096 paths the options' path cubes take 31–36 s, their
recalibration 0.07–0.09 s of it), whose memory also stops the job at 65,536
paths on the CPU (a 277 GB allocation, the old code's too) and at 1,024 on the 8 GB GPU (6 GB),
so the roadmap's 64k and 256k job baseline cannot be taken until step 3.7 gives the rollback its
matrix form ([I-83](../planning/known-issues.md#i-83)). A cold job is 3 s (8%) slower on the CPU
and 1–3 s on the GPU: each bootstrap bucket's program now holds Newton's derivative of the
helper's price and the nested y\* solve in several places (the bracket's ends, the step, the
model value), 5.4 s of compile for the job's ten bucket programs against 3.3 s. Evaluating a
widened bracket's two sides in one call took it from 7.1 s; the compiled programs stay for the
worker's lifetime and on disk, so a repeated or restarted job does not pay it.

**The demo traced** (`demos/demo_profile_small.py`, a fresh API and worker per mode). The
warm repeat, traced whole: on the RTX 5060 1.6 s and 0 compiles, 276,344 events of which
46,213 on the compute stream (2.4: 32.4 s, 3.86M events, 1.36M kernels; the profiler's cost is
per launch, so the trace now costs what the job does, 1.4 s untraced); on the CPU 1.9 s,
181,523 events (2.4: 1.8 s, 772,284). From scratch on the CPU (`--cold --no-disk-cache`) 42.2 s
and 680,210 events against 37.9 s and 1,226,160 for the old tree the same hour; with the disk
cache read back (`--cold`) 20.2 s. Every trace spans 100% of its run. The milestone stands on
both: the job runs through the API and the engine worker on the GPU and its trace is whole.

Unchanged by 2.5, on Windows' CPU only: pricing after an AD Greeks call is slower, 1.57–1.83 s
before one Greeks call and 1.98–2.25 s after, with the old and the new code alike
([I-53](../planning/known-issues.md#i-53)). The options' bump Greeks (each bump recalibrates)
take 0.58–0.67 s warm on the CPU, as before (0.61–0.75 s): their cost there is the grid, not the
calibration.

### 2.1 and 2.2: the 4-trade pipeline before roadmap 1.3 (history)

The two sections below measured an older pipeline (four trades, the legacy Hull-White
path) and are kept for the reasoning that followed from them.

#### Cold (warmup off — the default)

With the Python tracer off, the host tracer still records everything JAX and XLA do.
Measured lane breakdown on the 4-trade demo portfolio (durations exceed wall time because
lanes run concurrently and nest):

```
/host:CPU (main)             301,105 events   107.2s
tf_PjRtCompilerThreadPool    237,300 events    81.2s
tf_xla-cpu-codegen           241,552 events    47.1s
tf_XLAEigen                  132,769 events     2.9s
tf_XLAPjRtCpuClient           13,737 events     1.4s
```

Classified by what the event names mean — **this is the finding that motivated everything
below**:

| Category | Time |
|---|---|
| XLA compilation | ~118.6 s |
| pjit dispatch (host) | ~76.3 s |
| JAX tracing | ~3.1 s |
| **Device execute (the actual math)** | **~3.0 s** |

**~98% of the job was compile and dispatch overhead around 3 seconds of arithmetic.**

#### Warm (`JAX_RISK_PROFILE_WARMUP=1` — what `demo_profile_small.py` ships as)

The cold table above answers "what does this job cost from scratch." It is **not** the
steady-state picture, and the difference is large enough that quoting the cold numbers for
a repeat job is simply wrong. Measured on the same 4-trade demo portfolio, warm, via
`demos/demo_profile_small.py` unmodified (`WARM_CACHE = True`):

Job wall time **15.19 s**, 278,534 events. Lane breakdown, top-level spans only so nested
events are not double-counted:

```
main Python thread (tid)      34,143 events   15.19s   <- the job itself
tf_xla-cpu-codegen           153,021 events   29.12s   (24 threads, concurrent)
tf_PjRtCompilerThreadPool     38,011 events   10.78s   (1 thread, concurrent)
tf_XLAEigen                   39,668 events    0.14s   <- actual kernels
tf_XLAPjRtCpuClient           13,135 events    0.11s   <- actual kernels
```

The codegen and compiler-pool lanes exceed wall time because they are background threads
running concurrently with the main thread and with each other; only the main-thread column
is a wall-clock budget.

Where the 15.19 s actually goes:

| Category | Time | Share |
|---|---:|---:|
| **XLA compilation** (31× `backend_compile_and_load`) | **11.98 s** | **79%** |
| Everything else on the main thread (tracing, dispatch, Python) | ~3.2 s | 21% |
| **Device execute (the actual math)** | **~0.25 s** | **1.6%** |

Device execute is `ThunkExecutor::Execute`, 1,131 spans totalling 0.246 s, corroborated
independently by the two kernel lanes (`tf_XLAEigen` 0.14 s + `tf_XLAPjRtCpuClient` 0.11 s).

By phase annotation (§4), the concentration is extreme:

| Phase | Wall |
|---|---:|
| greeks | **14.04 s** |
| calibration | 1.12 s |
| pricing | 0.01 s |
| base_npv | 0.01 s |
| simulation | 0.01 s |
| risk | 0.00 s |

And inside `greeks`, per trade — every entry is dominated by two jitted programs:

| Trade | Wall | `combined` | `price_fn` | other |
|---|---:|---:|---:|---:|
| trade2 `BermudanSwaptionConfig` | 5.40 s | 2.85 s | 1.81 s | 0.72 s |
| trade3 `AmericanSwaptionConfig` | 5.03 s | 2.58 s | 1.69 s | 0.73 s |
| trade1 `SwaptionConfig` | 2.96 s | 2.53 s | 0.41 s | 0.02 s |
| trade0 `SwapConfig` | 0.65 s | 0.49 s | 0.15 s | 0.01 s |

**What this means.** Warming the cache removes 177 of the 208 cold compilations (208 → 31,
verified by counting `backend_compile_and_load` inside the trace window) and cuts wall time
~26 s → ~15 s. It does **not** change the shape of the answer: compilation still accounts
for ~79% of a warm job against ~1.6% real arithmetic, with the remaining ~19% being
main-thread tracing and dispatch. Warmup buys back absolute seconds, not a different
verdict — the job is overwhelmingly overhead either way.

The reason is entirely §3.5's residual recompile. All 31 warm compilations sit under
`PjitFunction(combined)` and `PjitFunction(price_fn)` in the Greeks phase — `price_fn` is a
fresh closure per call and `jax.jit` keys on function identity, so the grad+Hessian-diagonal
program recompiles on every Greeks call even for an identical trade. That is the whole warm
cost. Fixing it (Known Issues [I-21](../planning/known-issues.md#i-21)/[I-22](../planning/known-issues.md#i-22),
done in §3.7) is what would turn this into an execution-dominated timeline; nothing else on the list would
move the number meaningfully, because there is only 0.25 s of arithmetic to expose.

**On "Python vs JAX".** This trace cannot answer that question, and neither can any trace
this engine collects by default: `python_tracer_level=0` (§1.3) means **no event carries a
Python source file or line**. What it separates is *compilation* (11.98 s, XLA's own C++/MLIR
machinery, not your Python), *device execute* (0.25 s), and a ~3.2 s residue of main-thread
tracing, dispatch and interpreter time that is **not further attributable from this trace**.
Engine-Python attribution needs a separate `cProfile` run — not
`JAX_RISK_PROFILE_PYTHON_TRACER=1`, which at this event count would blow the ~1M cap (§5).

### Reading the timeline on CPU

On the CPU backend there is **no separate device row**. XLA kernels run on the same host
threads as the Python driver (`tf_XLAPjRtCpuClient`, `Thread_*`), tagged `xla_op`,
interleaved with dispatch. `tf_PjRtCompilerThreadPool` and `tf_xla-cpu-codegen` are the
background compilation threads. A tall Python stack (`price_portfolio` → `scan` →
`_run_python_pjit` → `_uncached_lowering` → `compile_or_get_cached`) is XLA
*lowering/compilation*, not the math.

### Reading the timeline on the GPU

On a GPU (roadmap 2.2) the device has its own rows: one per CUDA stream, named for the work on
it (`Stream #14(Compute,MemcpyD2D,MemcpyH2D,Memset)` on the RTX 5060), one event per kernel or
copy. The host rows are as on the CPU, with the job's thread named `python` and compilation on
`tf_pjrt_compile_thread_pool`. A gap on the stream row while the host row is busy is the device
waiting for the host: dispatch, a launch, or a compile.

---

## 3. Why Greeks dominated the trace

> **History.** This investigation and its numbers predate roadmap 1.3: they were measured on
> the Hull-White pipeline, where each trade carried its model (`hw_sigma`) and the Greeks
> were the bump-and-AD hybrid of that path. The lessons (§3.2–3.4, §3.6) carry over
> unchanged; the current pipeline's costs are in [I-53](../planning/known-issues.md#i-53)
> and [I-21](../planning/known-issues.md#i-21).

### 3.1 The measurements

Per-knob, on the small demo portfolio (fresh process each, raw trace bytes):

```
baseline (demo_structured's portfolio)      ~50 MB   ~36 s
scenarios 4096→256, grid 9→4 points         ~42 MB   ~31 s
n_per_std 64→16, exercise dates 4→2         ~42 MB   ~31 s
curve pillars 6→2                           ~42 MB   ~31 s
compute_greeks: on→off                      ~11 MB    ~9 s   ← the only knob that moved
```

Every "make the numbers smaller" knob was nearly free; Greeks was a 4x. Per-trade:

```
swap + European swaption only     ~7.8 MB    ~8 s
+ Bermudan swaption              ~29.8 MB   ~27 s     ← +22 MB
+ American instead               ~29.9 MB   ~27 s
all four                         ~41.6 MB   ~31 s
```

**Each tree-priced trade added a fixed ~22 MB regardless of how small its grid was.** A
fixed cost, invariant to problem size, is the signature of compile overhead rather than
compute. The HLO protos confirmed it: **1,241 separate XLA programs**, of which 142 were
`jit_multiply`, 132 `jit__where`, 73 `jit_broadcast_in_dim` — single scalar primitives,
each compiled and dispatched on its own.

### 3.2 The root cause

`engine.risk.greeks.bermudan_delta_gamma` runs `jax.grad`/`jax.hessian` through
`bermudan_swaption._run_backward_induction`, which **could not be `jax.jit`-wrapped**:

> `engine.risk.greeks` differentiates straight through this function, calling it with a
> `_PreparedBermudan` whose `zero_rates` (and, for Vega, its volatility) are live `jax.grad`
> **tracers** rather than concrete arrays. A jit static argument must be **hashable and
> concrete**, so a tracer-carrying `_PreparedBermudan` can never be one.

Every other pricer passed its `_Prepared*` struct as a jit static argument via
`StaticKeyMixin`'s by-value hashing (removed since, §3.7). The Bermudan
pricer structurally could not, because under `jax.grad` the very fields being hashed are
tracers. So every elementwise op around the `lax.scan` dispatched as its own tiny program
— and `jax.hessian`, being `jacfwd(jacrev(f))`, traced all of it twice more.

### 3.3 The fix: split the struct along the differentiability line

The constraint was real; the conclusion "therefore no jit" was stronger than necessary.
`_PreparedBermudan` mixes two kinds of field:

| Kind | Fields | Belongs as |
|---|---|---|
| Differentiation targets | the curves, the volatility | pytree **child** (traced) |
| Pure numeric scale | `notional`, `fixed_amounts` | pytree **child** (traced) |
| Trade structure | schedule, `exercise_times`, `n_per_std`, … | **aux data** (static) |

[`_PreparedBermudan`](../../engine/instruments/bermudan_swaption.py) is now a registered
pytree with exactly that split, and `_backward_induction_arrays` is
`@partial(jax.jit, static_argnums=1)` — `swap` crosses as a pytree, `schedule` stays
static. Tracers flow through as **leaves**, which is what pytrees are for, while jit keys
its cache on the static structure.

`notional`/`fixed_amounts` are children for a different reason: they are pure scale, not
structure, so two trades differing only in size now share one compiled kernel instead of
forcing a recompile per notional. (Since §3.7 every field but `payer` and the grid settings
is a child, and the schedule is traced too.)

### 3.4 The other four changes

**Hessian diagonal via HVP.** Every Delta/Gamma function reports only the same-pillar
second partial (ORE's `SensitivityCube::gamma` is a cross-scenario second difference, so
there is no cross-pillar term to match). Building the full `[n, n]` Hessian to keep `n`
entries meant tracing the pricer `n` times over and discarding `n² − n` results.
Each diagonal entry is one Hessian-vector product against a basis vector instead —
`hvp(f, x, eᵢ)[i] == ∂²f/∂xᵢ²` — batched under `vmap` (since §3.8 from one linearization of
the gradient over all of a trade's curves, `_gradients_and_hessian_diagonals`).

**Jitted Theta valuations.** `swaption_theta` measured **56 compilations** — worse than
the entire Delta/Gamma pair — because both Jamshidian valuations ran eagerly. Now 2.

**Vega Jacobian accumulation.** The bootstrap's forward substitution is genuinely
sequential (row *j* reads rows *< j*), but it was writing rows with `J.at[j, :].set(...)`,
producing ~1,000 eager `dynamic_update_index_in_dim` dispatches. Rows now accumulate in a
Python list and `jnp.stack` once.

**Jitted calibration pricing.** `calibrate_lgm_sigma` measured **137 compilations**, which
the §4 phase annotations pinned to calibration rather than to Vega as first assumed. Its
bisection was never the problem — `_bisect_bucket_sigma`'s `lax.scan` already traces
`price_fn` once and compiles all 60 iterations together — but the
`bachelier_swaption_price`/`price_lgm_swaption` calls *around* it ran eagerly, once per
basket instrument plus once more for the diagnostics.

Those were jitted through a closure rather than with `static_argnums` (superseded by §3.7,
which passes the target as a pytree argument):
`CalibrationTarget` holds NumPy arrays and so is not hashable, and it **must not become**
hashable/frozen, because `bermudan_vega` substitutes a live `jax.grad` tracer into its
`market_vol` via `dataclasses.replace`. Closing over the target and jitting a nullary
function needs no hashing at all.

### 3.5 Results

Reference single-trade Greeks job (3 pillars, 2 exercise dates, `n_per_std=16`), counting
XLA compilations directly at `backend_compile_and_load`:

| | Before | After |
|---|---:|---:|
| `price_bermudan_swaption_base` | 128 | **4** |
| `bermudan_delta_gamma` | 470 | **5** |
| `bermudan_theta` | 2 | **2** |
| `swaption_theta` | 56 | **2** |
| **Total** | **602** | **13** |
| Wall time | 16.6 s | **4.7 s** |

Separately, on the 2-instrument demo basket:

| | Before | After |
|---|---:|---:|
| `calibrate_lgm_sigma` | 137 | **18** |
| `bermudan_vega` | — | 19 |

`calibrate_lgm_sigma` separately went from **137 compilations / 3.4 s** to **18 / 1.3 s**
(and its bootstrap RMSE improved, 3.1e-11 → 2.3e-12). The bisection was never the culprit
there — `_bisect_bucket_sigma`'s `lax.scan` already compiles all 60 iterations as one
program; the eager `bachelier_swaption_price`/`price_lgm_swaption` calls around it were.

Full 4-trade portfolio through the real HTTP/worker path (the pool of the time). Two configurations, because
they answer different questions — the first is `demos/demo_profile_small.py` exactly as
shipped (uncalibrated tree trades, so calibration and Vega are both exercised), the second
a hand-built request with a flat trade volatility (no calibration, no Vega):

| | Before | After |
|---|---:|---:|
| `demo_profile_small.py`, Greeks **on** (what it ships as) | ~42 MB, ~31 s | **41.1 MB, 25.6 s** |
| same portfolio, Greeks off | ~11 MB, ~9 s | **8.5 MB, 6.5 s** |
| flat-sigma portfolio, Greeks **on** | — | **29.9 MB, 19.2 s** |
| flat-sigma portfolio, Greeks off | — | **5.6 MB, 5.4 s** |

The Greeks-off rows are recorded for the ~5x comparison only; the demo no longer has a
switch for them (it always computes Greeks — a trace without them is not representative of
where this engine spends its time).

Note the demo's Greeks-on trace barely shrank in BYTES (42 → 41 MB) while its wall time
fell ~20%. That is the §3.6 effect: the remaining trace is per-compilation MLIR pass
events for a few large fused programs, not op-by-op dispatch, and byte count is no longer
a good proxy for dispatch count. Compile counts and wall time are.

> Measuring this yourself: the profiler writes a new `pid-<pid>/` per run and never cleans
> up, so summing the whole directory reports every run ever done into it. Delete
> `.profile-out-small` between runs — the demo now reports per-run size and warns when
> older runs are present, after exactly this mistake hid a real reduction behind three
> stale runs.

Every reference value is bit-identical across the change (`npv = 8,521.0223`,
`delta = [0.0, −56.8552, 150.8868]`, `theta = 26.931068`).

Cache behavior, verified directly:

| Repeat of… | Compilations |
|---|---:|
| identical forward pricing call | **0** |
| forward pricing, same structure, different notional | **0** |
| forward pricing, different `n_per_std` (a real shape change) | 1 (correct — must recompile) |
| identical `bermudan_delta_gamma` call | **1** (see below) |

**One residual compile per Greeks call** (fixed by §3.7: a repeated call now compiles
nothing). `price_fn` is a fresh closure every time (each
`_*_price_fn` rebuilds it around that trade's prepared structure), and `jax.jit` keys on
function identity — so the combined grad+Hessian-diagonal program recompiles once per call
even for an identical trade. Fixing it properly means a closure cache keyed on the full
`_Prepared*` structure, which risks returning a stale program for a mutated config; against
a 470 → 5 improvement that is not currently worth the correctness risk. Pinned by
`test_repeated_greeks_call_costs_one_compile_not_zero` so it cannot silently regress in
either direction. Worth revisiting if repeated same-trade Greeks (an intraday re-risk loop)
ever becomes the dominant access pattern.

### 3.6 What did *not* shrink, and why

Greeks is still a ~5x multiplier on trace size, and that residue is **genuine, irreducible
compilation of a few large programs** rather than thousands of small ones. The remaining
trace is dominated by MLIR compiler-internal passes (`CSEPass` × 23,148,
`OpToOpPassAdaptor` × 15,944, `CanonicalizerPass` × 12,642) — the cost of compiling the
fused programs properly, which is exactly what you *want* to be paying for.

Two honest limits remain:

- **A Bermudan tree is intrinsically expensive to compile.** One fused backward-induction
  program over a 33-node state grid and a multi-time-step `lax.scan`, differentiated twice,
  is a large HLO module. It compiles once and is cached, but the first one is not cheap.
- **Trace size no longer tracks program count linearly**, because per-compilation MLIR
  pass events now dominate. Shrinking further means compiling *fewer distinct programs*
  (already near the floor: 13 on the single-trade job), not warming the cache — warmup
  leaves 31 genuine recompiles on this portfolio, per §1.4.

**On returning to `python_tracer_level=1`:** it is still the wrong trade. The Greeks-on
trace is ~262k events; the Python tracer multiplied event count by ~35x in the original
measurement, which would put this well past the JSON export's ~1M cap (§5).
The attribution it would buy is already available for free via §4's phase annotations. The
flag remains available for a narrowly-scoped single-phase investigation, which is the only
context where it fits under the cap.

### 3.7 Trade data as traced arguments (2026-10-02)

Measured across the test suite, most wall time was still XLA compilation: 75-90% of the
market-path, calibration and Greeks tests, from two sources. Programs were keyed on a
trade's *values* (a static `_Prepared*`, a calibration basket closed over, a fresh closure
per Greeks or market-risk call), so each trade, each path date and each call compiled anew;
and pricers that ran eagerly compiled one tiny program per primitive and shape.

The rule now: **a pricer is a module-level `jax.jit` function, and the trade's data reaches
it as a pytree argument.** Only what fixes a program's shape or Python control flow is
static (`payer`, the grid's `n_per_std`/`std_devs`, the proRata mapping of a calibration
helper, a Jamshidian model). Programs are therefore keyed on shapes (coupon, exercise and
helper counts, paths, dates), so one compile serves every trade, date, bump and call of
that shape.

| Pytree | Jitted with it |
|---|---|
| `Legs`, `PathSchedule` | `legs_npv`, `_legs_cube`, `_path_fixings` |
| `EuropeanTerms` (expiry as a serial date) | `black_multileg_npv`, `_european_cube`, `_jamshidian_cube` |
| `_PreparedBermudan`, `_GridSchedule` | `_backward_induction_arrays`, `_rollback_every_path` |
| `BasketInstrument` | `_bootstrap` (I-22), the Vega Jacobian's `_residual_gradient` |
| `CalibrationTarget` | the legacy `calibrate_lgm_sigma`'s prices and bisection |
| `TradePriceFunction.terms` (`_LegsTerms`, `BachelierTerms`, `EuropeanTerms`, `OptionTerms`), the pricer static (since 2.4, §3.8) | the AD Greeks' `_curve_derivatives`, `_option_vega`, `_bachelier_vega` |

Callers that build a fresh closure per call were deliberately *not* jitted as a whole: the
AD Greeks and market risk (`revalue_trade`) differentiated or vmapped closures over the
jitted pricers, relying on JAX to cache the derivative and batched programs per pricer. That
cache turned out to be bounded (§3.8), and since roadmap 2.4 the AD Greeks are jitted whole
on the price function as data; market risk still vmaps its closure ([I-53](../planning/known-issues.md#i-53)).
`jamshidian_npv` itself stays eager when pricing because its x* derivative rule closes over
intermediates that `jax.grad` cannot carry through a jit boundary; its cube is jitted, and its
Greeks trace it inside their own jit (§3.8).

Measured, cold process, no disk cache (repeat = the same call again in the process):

| | Before | After |
|---|---:|---:|
| AD Greeks, one Bermudan, repeat call | 12.8 s, 30 compiles | **0.3 s, 0 compiles** |
| AD Greeks, one Bermudan, first call | 24.4 s, 276 compiles | 18.9 s, 60 compiles |
| Market risk, 4 trades × 512 scenarios, repeat | 7.7 s, 8 compiles | **1.5 s, 0 compiles** |
| `tests/test_portfolio_market_path.py` | 65 s, 1,247 compiles | 36 s, 700 compiles |

Floating-point results move at rounding level: XLA fuses a whole program differently from
op-by-op dispatch, and the old eager results also depended on which operands XLA folded as
constants. This was accepted for this change on 2026-10-02 (planning `details/precision.md`
§13.1 records the measured size); every ORE parity suite passes at its tolerance.

**Across processes**, the test suite also keeps JAX's persistent compilation cache in
`.jax_cache/` (`tests/conftest.py`), so a rerun reads back most programs instead of
compiling them. The engine worker enables it too (`engine.api.worker.compilation_cache_environment`):
in `JAX_COMPILATION_CACHE_DIR` if set, else `xla-cache/` beside its job queue, every program
cached, so a restarted worker reads its programs back. An in-process `price_portfolio`
caller enables it as JAX documents, if wanted.

### 3.8 The AD Greeks as one program per product (2026-10-06, roadmap 2.4)

**What 2.1 found.** The demo's job repeated in the worker compiled 9 programs (the swap's and
European's pricers under the gradient and the Hessian-vector product, and the European's Vega);
a third run compiled none.

**Why.** The AD Greeks differentiated the jitted pricers eagerly (`jax.grad` outside any
jit). JAX then builds each derivative's forward and backward programs from jaxprs it keeps in
internal caches of 2,048 entries, least recently used first out (`_dce_jaxpr_pjit`,
`_cached_abstract_eval`, ... in `jax._src`), and lowers them keyed on the jaxpr object. One
cold demo job filled several of those caches (8,036 misses on `_dce_jaxpr`, 2,914 on
`_dce_jaxpr_pjit`), so by the end of the job the first trades' derivative jaxprs were evicted;
the next job built new ones and compiled them again, and churned little enough that the third
found them. Found with `jax_explain_cache_misses` and the caches' `cache_info()`.

**The fix.** A trade's price function is data: a module-level pricer, the trade's terms as a
pytree, its curves' pillar times (`engine.risk.price_functions.TradePriceFunction`: `pricer`,
`terms`, `times`; `price(*rates)` as before, which market risk and the tests use). The
derivatives are module-level jits with the pricer static: `_curve_derivatives` (every curve's
Delta and Gamma), `_option_vega` and `_bachelier_vega`. Each is one program per product and
shape, held by JAX's jit cache for the process like any pricer, and shared by every trade of
that shape. A pricer that carries settings is an object equal for equal settings
(`JamshidianPrice`), so it keys the cache by value.

**Compile time.** An option's Greeks were about ten programs: per curve, a forward and a
backward program for the gradient and for the Hessian-vector product, then the Vega's two.
Measured on the demo's Bermudan, in a fresh process (trace, lower, compile):

| Structure | Trace | Lower | Compile | HLO | Temporary memory |
|---|---:|---:|---:|---:|---:|
| Per curve, a gradient and a Hessian-vector product, each jitted (4 programs) | 2.2 s | 1.3 s | 6.9 s | 3.0 MB | 34–94 MB |
| One jit, the same per-curve gradients and products | 2.4 s | 0.7 s | 6.7 s | 2.6 MB | — |
| **One jit, one linearization of the gradient over every curve** (`_gradients_and_hessian_diagonals`) | 1.8 s | 0.5 s | **3.5 s** | 1.6 MB | 138 MB |
| Vega (`_option_vega`) | 0.5 s | 0.2 s | 1.3 s | 0.7 MB | 58 MB |

One merged program compiles no faster than the same work split, but the linearization runs
the forward and backward pass once for all curves instead of once per curve and per Hessian
product, which cuts an option's Delta and Gamma compile by 40%. Finding the interpolation's
interval without a loop (`searchsorted(method="compare_all")` inside `jnp.interp`) cut the
compile by 7% and slowed the run, so the grid engine keeps JAX's default. On the demo's job, a
cold run compiles 217 programs instead of 269 and takes 32 s instead of 37 on the CPU, and a
repeat compiles none (§2.0, after 2.4).

**Numbers.** The derivatives are the same mathematics, but one XLA program fuses what ran as
separate programs and the linearization accumulates the gradient over every curve in one
backward pass, so the AD Greeks move at rounding level: on the golden snapshot (332 arrays,
CPU) 46 AD Greek arrays moved, by at most 3.4e-14 of each array's largest magnitude (the
demo Bermudan's discount-curve Gamma), and all but three Greeks of the demo's options (each
in both of its runs) by less than 1e-14; everything else, cubes, exposures, today's values,
bump Greeks, market risk and the float32 runs, is bit for bit. On the GPU the cube and exposure are bit for bit with the old code, the AD Greeks
within 6.6e-16 of their scale (planning `details/precision.md` §13.1).

---

## 4. Reading a trace: phase annotations

With the Python tracer off, no event carries a Python source line, so "which phase is this
dispatch from?" is unanswerable from the raw trace. That is bought back by
[`engine/portfolio/profiling.py`](../../engine/portfolio/profiling.py):

```python
with phase("calibration"):
    model = build_cross_asset_model(market, simulation)
with phase("simulation"):
    scenarios = simulate(market, simulation, model, run.precision)
```

`phase()` enters **two** mechanisms, deliberately — neither alone is sufficient:

- **`jax.profiler.TraceAnnotation`** — a *host-side* region. Wraps the wall-clock interval
  on the Python thread, so everything dispatched inside it (**including XLA compilation**,
  which is host work) lands under that label. This is the one that matters for this engine.
- **`jax.named_scope`** — labels ops as they are *traced into a jaxpr*, baking the name
  into the compiled HLO so it appears in `op_profile`/`hlo_stats`. It only affects work
  traced inside the scope.

Using only `named_scope` was tried first and measured: it produced **zero** labelled host
events for every phase whose work was eager dispatch rather than fresh tracing. Hence both.

Cost is a string push/pop with no synchronization, so these stay permanently in the pricing
path rather than being conditional on a profiling flag.

What this buys, from a real traced run:

```
calibration                            0.00s
simulation                             1.47s
pricing                                1.85s
base_npv                               1.34s
risk                                   0.16s
greeks                                12.49s
  greeks/trade0/SwapConfig              0.70s
  greeks/trade1/SwaptionConfig          3.19s
  greeks/trade2/BermudanSwaptionConfig  4.68s
  greeks/trade3/AmericanSwaptionConfig  3.92s
```

Per-phase and **per-trade** attribution, with the Python tracer off and no size penalty.

These annotations immediately earned their keep: on the calibrated demo portfolio they
showed `calibration` at 4.16 s — a phase that had been assumed cheap, and whose 137
compilations (§3.4) would otherwise have been attributed to Vega, which was the prime
suspect and turned out to cost only 19.

---

## 5. The trace summary and its checks

After every traced job the worker reads the trace back
(`engine.portfolio.profiling.summarize_trace`, on the `.xplane.pb` through
`jax.profiler.ProfileData`, so it sees every event) and writes
`$JAX_RISK_PROFILE_DIR/pid-<pid>/<run>.summary.json` beside the run: the traced run's wall
time and compiles (and the untraced warm-up's), the trace's events, size and span, the share
of the run it spans, events per thread, and seconds per phase label. With a phase window
(§1.6) the traced run is the phase: `traced` holds its wall time, compiles and name, beside
the whole job's (`job_wall_seconds`, `job_compiles`). `demos/demo_profile_small.py`
prints it. Reading 1.5M events back takes about 3 s, after the job's result is computed.

It then warns (`UserWarning`, and the `warning` field of the summary) when either:

- the trace spans less than 50% of the run's wall time: the record itself stopped early; or
- the trace has more events than the `.trace.json.gz` export keeps (~1M, the earliest-starting):
  that file is partial, though xprof, which reads the `.xplane.pb`, shows everything.

A span check alone cannot catch the second: the export keeps the long regions that start
early (`greeks` ran to the end of the cold demo's partial file), so its span still looks
complete. It **warns rather than raises**, and swallows every error reading the trace back: a
degraded diagnostic must never fail a pricing job whose result is already correct.

---

## 6. How to collect and read a trace

```bash
pip install -e ".[api,profiling]"     # brings in xprof; quote it in PowerShell
rm -rf .profile-out-small             # the profiler never cleans up after itself
python demos/demo_profile_small.py    # warm; --cold [--no-disk-cache] for a first run
python demos/demo_profile_small.py --phase pricing   # one phase only (the GPU's way, §1.6)
xprof --port 8791 .profile-out-small
```

`xprof` is an optional extra, not a core dependency — collecting a trace needs only JAX,
so a missing `xprof` means you can still capture but not view. If the command is not found
after installing, `.venv/Scripts` is probably not on `PATH`; `python -m xprof --port 8791
.profile-out-small` sidesteps that.

Open `http://localhost:8791`, pick the `pid-<pid>` run, then a tool:

| Tool | Answers |
|---|---|
| `overview_page` | compile vs. execute vs. other — start here |
| `trace_viewer` | the timeline; find the `phase()` regions from §4 |
| `op_profile` / `hlo_stats` / `framework_op_stats` | which ops dominate, on-device vs. host |

Environment variables:

| Variable | Default | Effect |
|---|---|---|
| `JAX_RISK_PROFILE_DIR` | unset | Enables profiling; traces to `$DIR/pid-<pid>/` |
| `JAX_RISK_PROFILE_WARMUP` | `0` | `1` = discard one run first, so the trace shows a repeat (§1.4) |
| `JAX_RISK_PROFILE_PHASE` | unset | A phase's name: trace only that phase of the job (§1.6) |
| `JAX_RISK_PROFILE_PYTHON_TRACER` | `0` | `1` = CPython frames (~9x size; see §1.3) |

---

## 7. Tests

[`tests/test_profiling_and_jit.py`](../../tests/test_profiling_and_jit.py) pins the
structure, since the numerical suites cannot see it — the answers were correct the whole
time, they were merely arrived at via ~600 compilations.

- **Pytree split** — children are exactly the traced fields; flatten/unflatten round-trips
  exactly; aux data is hashable.
- **Compile counts** — upper bounds per operation (deliberately loose: they catch a return
  to eager dispatch, not a JAX version bump). Includes both directions of cache behavior:
  a repeat call must compile **zero**, a changed `n_per_std` (a genuine shape change) must
  compile **at least one**. Since §3.8: an option's Greeks are one Delta/Gamma program and
  one Vega program, and the demo's whole job, run twice in a fresh process, compiles nothing
  the second time (it compiled 9 programs before).
- **HVP diagonal ≡ full Hessian diagonal** — `curve_greeks`' Delta and Gamma against
  `jax.grad` and `jnp.diagonal(jax.hessian(...))` in every curve, for all three instrument
  types (since §3.8 they come from one linearization over all the trade's curves).
- **Gradients survive the jit boundary** — the failure mode the split most easily
  introduces is a differentiable field landing in static aux data, for which JAX does not
  raise: it silently returns a **zero gradient**. Guarded by a nonzero-delta check plus a
  finite-difference cross-check that shares no autodiff machinery.
- **Trace summary** — a real traced job (warm-up on) writes `<run>.summary.json` with both
  runs' compiles (the repeat compiles none) and its phase; with `JAX_RISK_PROFILE_PHASE`
  only that phase is in the trace, its compiles and wall time are the phase's and the job's
  are kept beside them; a phase the job lacks traces nothing and warns; the window opens on
  the phase's first run only, and closes even when the phase raises; on a synthetic `.xplane.pb`, events,
  span, phase seconds and threads are counted; a short trace and one past the JSON export's
  cap are reported, a healthy one is not; a corrupt trace never raises.
- **Phase annotations** — both mechanisms entered; a real `price_portfolio` run hits every
  expected phase, all of them in `PHASES`; both Greeks methods label each trade
  (`greeks/trade<i>/<type>`).
