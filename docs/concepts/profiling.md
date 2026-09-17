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
[`engine/portfolio/worker_pool.py`](../../engine/portfolio/worker_pool.py)'s
`_run_pricing_job`:

```python
profile_dir = os.environ.get("JAX_RISK_PROFILE_DIR")
if not profile_dir:
    return _run()                    # inert: does not even import jax

import jax
if os.environ.get("JAX_RISK_PROFILE_WARMUP") == "1":
    jax.block_until_ready(_run().npv_cube)     # discarded warm-up run

out_dir = os.path.join(profile_dir, f"pid-{os.getpid()}")
with jax.profiler.trace(out_dir, profiler_options=_profile_options(jax)):
    result = _run()
    jax.block_until_ready(result.npv_cube)     # trace must outlive device work
_warn_if_trace_truncated(out_dir, elapsed)
```

Five decisions define its behavior:

### 1.1 It traces inside the pool worker

The trace is taken where `price_portfolio` actually executes — a freshly spawned worker
process with cold JAX caches — not in the API handler. Output goes to
`$JAX_RISK_PROFILE_DIR/pid-<pid>/`, **one directory per process**, so a job fanned out
across pool workers produces one independent trace per worker instead of a corrupted
shared file.

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

Worse than the noise: the profiler's event buffer is a **fixed ~1M-event cap with no
backpressure and no warning**. Those interpreter frames saturated it during startup, so
the trace silently covered only **the first 1.6 s of a ~90 s job** while reporting
success. See §4 for the guard that now catches this.

Turning it off loses exactly one thing — attribution of a dispatch back to the engine
function that issued it, since no event then carries a Python source file or line
(verified: 0 of 926,463 events had `source_file`/`source_line`/`long_name`). §3 is how
that attribution was bought back for ~0 cost.

`JAX_RISK_PROFILE_PYTHON_TRACER=1` opts back in, accepting the ~9x size, ~2x slowdown and
near-certain silent truncation.

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
genuinely execution-dominated timeline needs the closure-identity recompile fixed first.
All 31 are enumerated with exact callsites, root cause and a vetted fix plan in
[Known Issues I-21 and I-22](../known-issues.md#i-21) — two different mechanisms needing
two different fixes, which is why they are filed separately.

### 1.5 `block_until_ready` before the context exits

JAX dispatch is asynchronous. If the trace context closes while device work is still
queued, the timeline is truncated. `npv_cube` is the dominant device-side tail.

---

## 2. What the trace contains

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

### Reading the timeline on CPU

On the CPU backend there is **no separate device row**. XLA kernels run on the same host
threads as the Python driver (`tf_XLAPjRtCpuClient`, `Thread_*`), tagged `xla_op`,
interleaved with dispatch. `tf_PjRtCompilerThreadPool` and `tf_xla-cpu-codegen` are the
background compilation threads. A tall Python stack (`price_portfolio` → `scan` →
`_run_python_pjit` → `_uncached_lowering` → `compile_or_get_cached`) is XLA
*lowering/compilation*, not the math.

---

## 3. Why Greeks dominated the trace

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
> `_PreparedBermudan` whose `zero_rates` (and, for Vega, `hw_sigma`) are live `jax.grad`
> **tracers** rather than concrete arrays. A jit static argument must be **hashable and
> concrete**, so a tracer-carrying `_PreparedBermudan` can never be one.

Every other pricer passes its `_Prepared*` struct as a jit static argument via
[`StaticKeyMixin`](../../engine/models/static_key.py)'s by-value hashing. The Bermudan
pricer structurally could not, because under `jax.grad` the very fields being hashed are
tracers. So every elementwise op around the `lax.scan` dispatched as its own tiny program
— and `jax.hessian`, being `jacfwd(jacrev(f))`, traced all of it twice more.

### 3.3 The fix: split the struct along the differentiability line

The constraint was real; the conclusion "therefore no jit" was stronger than necessary.
`_PreparedBermudan` mixes two kinds of field:

| Kind | Fields | Belongs as |
|---|---|---|
| Differentiation targets | `zero_rates`, `hw_sigma` | pytree **child** (traced) |
| Pure numeric scale | `notional`, `fixed_amounts` | pytree **child** (traced) |
| Trade structure | schedule, `exercise_times`, `n_per_std`, … | **aux data** (static) |

[`_PreparedBermudan`](../../engine/instruments/bermudan_swaption.py) is now a registered
pytree with exactly that split, and `_backward_induction_arrays` is
`@partial(jax.jit, static_argnums=1)` — `swap` crosses as a pytree, `schedule` stays
static. Tracers flow through as **leaves**, which is what pytrees are for, while jit keys
its cache on the static structure.

`notional`/`fixed_amounts` are children for a different reason: they are pure scale, not
structure, so two trades differing only in size now share one compiled kernel instead of
forcing a recompile per notional.

### 3.4 The other four changes

**Hessian diagonal via HVP.** Every Delta/Gamma function reports only the same-pillar
second partial (ORE's `SensitivityCube::gamma` is a cross-scenario second difference, so
there is no cross-pillar term to match). Building the full `[n, n]` Hessian to keep `n`
entries meant tracing the pricer `n` times over and discarding `n² − n` results.
`_grad_and_hessian_diagonal` computes each diagonal entry as one Hessian-vector product
against a basis vector instead — `hvp(f, x, eᵢ)[i] == ∂²f/∂xᵢ²` — batched under `vmap`.

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

Those are jitted through a closure rather than with `static_argnums`, deliberately:
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

Full 4-trade portfolio through the real HTTP/worker-pool path. Two configurations, because
they answer different questions — the first is `demos/demo_profile_small.py` exactly as
shipped (uncalibrated tree trades, so calibration and Vega are both exercised), the second
a hand-built request with flat `hw_sigma` (no calibration, no Vega):

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

**One residual compile per Greeks call.** `price_fn` is a fresh closure every time (each
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
measurement, which would put this well past the ~1M cap and back into silent truncation.
The attribution it would buy is already available for free via §4's phase annotations. The
flag remains available for a narrowly-scoped single-phase investigation, which is the only
context where it fits under the cap.

---

## 4. Reading a trace: phase annotations

With the Python tracer off, no event carries a Python source line, so "which phase is this
dispatch from?" is unanswerable from the raw trace. That is bought back by
[`engine/portfolio/profiling.py`](../../engine/portfolio/profiling.py):

```python
with _phase("calibration"):
    trades = _fill_calibrated_sigma(...)
with _phase("simulation"):
    market = generate_paths(...)
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

## 5. The silent-truncation guard

The profiler's ~1M-event buffer has no backpressure: once full, remaining events are
dropped silently and the resulting trace is **byte-indistinguishable from a complete one**.
That failure mode already hid 98% of a job once (§1.3).

`_warn_if_trace_truncated` now runs after every traced job and emits a `UserWarning` when
either:

- the trace is at or near the ~1M-event cap, or
- the captured events' own timestamp span covers less than 50% of the job's wall time.

It **warns rather than raises**, and swallows every error reading the trace back: a
degraded diagnostic must never fail a pricing job whose result is already correct. This
was previously only in `demos/demo_profile_small.py`; it now guards the production hook.

---

## 6. How to collect and read a trace

```bash
pip install -e ".[api,profiling]"     # brings in xprof; quote it in PowerShell
rm -rf .profile-out-small             # the profiler never cleans up after itself
python demos/demo_profile_small.py    # small, openable trace (~41 MB, ~25 s)
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
  compile **at least one**.
- **HVP diagonal ≡ full Hessian diagonal** — against an analytic case with a known answer,
  and against `jnp.diagonal(jax.hessian(...))` for all three instrument types.
- **Gradients survive the jit boundary** — the failure mode the split most easily
  introduces is a differentiable field landing in static aux data, for which JAX does not
  raise: it silently returns a **zero gradient**. Guarded by a nonzero-delta check plus a
  finite-difference cross-check that shares no autodiff machinery.
- **Truncation guard** — warns on a short trace, quiet on a healthy one, never raises on a
  corrupt one.
- **Phase annotations** — both mechanisms entered; a real `price_portfolio` run hits every
  expected phase.
