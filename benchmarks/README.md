# Precision comparison

The comparison includes FP16. `--policies combined compute storage` separates product and
storage errors; the default remains `combined` for bounded comparisons. Supported GPU formats
request native arithmetic by default. `--gemm auto` preserves the inherited emitter for every
format. Use `--gemm default` and `--gemm cublas` for explicit emitter comparisons; process
flags remain in the evidence. No format silently selects a different emitter.
`--expected-gpu H100` refuses a different GPU model.

Independent reference seeds can be collected alongside matched-path references:

```bash
python -m benchmarks.comparison --out .profile-out-h100-comparison --backends cpu gpu \
  --expected-gpu H100 --formats float64 float32 float16 float8_e4m3fn float8_e5m2 \
  --policies combined compute storage --paths 16384 65536 262144 \
  --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 \
  --reference-seeds 1001 1002 1003 1004 1005 1006 1007 1008 \
  --reference-paths 262144 1048576
```

Reference means, empirical SEs and count-to-count changes are retained. An independent
reference's empirical SE must be at most 10% of the target error before a speedup is
reported. Increase reference counts/seeds until it resolves the comparison; a finite
reference and an empirical SE do not supply a nominal confidence bound.

## H100 native qualification

On the target Linux host, in the pinned GPU environment:

The worker requests native arithmetic on capable GPUs without an emitter override.
`jax-risk-worker --gpu-gemm cublas` explicitly selects cuBLAS before backend initialization
when qualification calls for it. It cannot switch a running worker per job.

```bash
python -m tests.support.precision_devices --platform gpu --expected-device H100 \
  --formats float16 float16_1x float8_e4m3fn float8_e4m3fn_1x \
  float8_e5m2 float8_e5m2_1x --require-native-fp8 \
  --out .profile-out-h100-native --engine-workloads comparison_single comparison_portfolio
```

This retains numerical comparisons with CPU emulation, per-operation optimized HLO,
actual workload HLO and a device trace. FP8 cuBLAS calls and Triton FP8/FP32 dot fusions
are recognized separately; actual instruction use still requires trace inspection.
The matrix includes batched, odd-sized and
matrix-vector products, zeros, cancellation and stochastic rounding. An FP8 custom call
is recorded independently of the dispatch label. Inspect the actual workload trace for
hardware use; these synthetic checks do not certify financial accuracy. Incomplete runs
remain marked incomplete. Repeat after JAX, driver or accelerator-library changes.

## Broader accuracy campaigns

```bash
python -m benchmarks.qualification --out .profile-out-accuracy-large --tier large \
  --campaigns compute compute_market_risk --cases float32 float16 float8
python -m benchmarks.qualification --out .profile-out-accuracy-holdout --paths 1024 16384 \
  --campaigns compute --fixture-seeds 40 41 --models LGM HullWhite --grids recorded default \
  --seed-ids 1001 1002 1003 1004 --cases pricing:compute:float16 --multi-currency
```

The tiers are `reproduce` (1K/4K/16K, 16 seeds), `large` (16K/64K/256K, 16 seeds), and
`extended` (256K/1M/4M, 32 seeds). Explicit counts/seeds override a tier. Each point runs
in a fresh process; the harness retains one seed's complete reference and candidate at a
time. Fixture seeds reuse independently priced ORE sweep markets/trades and are separate
from Sobol seed identities. Holdout fixtures and seeds must be disjoint from policy tuning.
The multicurrency option adds correlated USD/EUR rates and simulated FX for exposures.
`--grids default refined` permits FP64 discretization checks separately from precision.

The output retains full measurements, exact workloads/environments, failures and
experimental A-11 evidence. Missing/failed counts cannot establish a passing ceiling.
Results never replace the shipped acceptance records automatically. `--resume` requires
the identical source digest and manifest. No-speedup, failed accuracy and OOM/timeout
results remain results; precision warnings never refuse research runs (D-9).

For recording more independent seeds in an existing campaign, `--seeds 64` now creates
all 64 identities; `--seed-ids 101 203 409` selects explicit identities. Previously the
requested count was silently truncated to the campaign's existing seed list.

## Complete suite evidence

```bash
python -m benchmarks.test_suite --out .profile-out-full-suite --batch-size 8
```

This collects every node under `tests/` once, runs bounded batches in fresh processes,
and requires successful exits, complete JUnit counts and printed summaries. Manifests,
per-batch logs/JUnit and `suite.json` survive a failed or interrupted run. Only
`complete=true` establishes a complete passing partitioned run. This avoids accumulating
the whole suite's compiled state in long-lived workers (I-27). CI's full job uses it
weekly and on manual dispatch, retaining the artifacts.

## Original bounded comparison and output contract

Run from the checkout with the pinned environment in `constraints.txt`:

```powershell
.venv/Scripts/python.exe -m benchmarks.comparison --out .profile-out-comparison
```

The defaults are a smoke study: 128 and 512 scenarios, seeds 1, 2 and 3, the single swap
and existing five-trade market-risk portfolio, CPU/GPU/TPU, FP64/FP32/FP16/BF16 and both FP8
formats. Every requested point appears in the report, including unavailable devices,
timeouts, compilation failures and formats that miss the numerical tolerance. Use a new,
empty output directory for each collection. Never run competing jobs when measuring speed.

For an NVIDIA GPU use Linux/WSL2 and the pinned `gpu` environment. GPU FP8 is native by
default on capable hardware. `--gemm auto` keeps inherited settings for all formats;
`--gemm default` explicitly restores the compiler's default emitter.
`--gemm cublas` explicitly selects cuBLAS for every GPU format. The report records flags
and dispatch. BF16/FP8 on CPU are emulated. A missing backend remains unmeasured.
Native dispatch alone is not hardware proof; retain lowering and trace qualification.

The RTX 5060 / JAX 0.10.2 compiler faults are explicit **local test configuration**:

```bash
XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \
JAX_RISK_EMULATE_PRODUCTS=bfloat16_3x \
python -m tests.support.precision_devices --platform gpu --out .profile-out-local-device
```

This keeps FP8 native through cuBLAS and emulates BF16_3x for its local vector-fusion fault.
Neither override is a project default. `JAX_RISK_EMULATE_PRODUCTS` accepts comma-separated
compute formats; a base name includes its `_1x` variant. Set it before tracing in a fresh
process. The report records emulated dispatch and the environment; incompatible execution
settings cannot be merged into one campaign. Default compiler failures remain failures.
See the [routing change and validation](../compliance/precision-research/2026-10-21/README.md).

The command does not initialize a multi-host JAX worker set; cluster qualification remains I-61.
Every worker explicitly selects its requested device, verifies the devices holding its
outputs, and refuses disabled JIT. An inherited `JAX_DEFAULT_DEVICE` cannot substitute CPU
execution for an accelerator point.

The workloads reuse `tests.support.precision_campaigns.market_risk`: a payer swap alone,
or payer/receiver swaps, a European, a fixed-LGM Bermudan and a bond on the sloped USD
two-curve market. The Bermudan uses the existing measured coarse grid (8 points per
standard deviation, 3 deviations). These are synthetic research inputs. This bounded
workload does not establish accuracy for calibrated-scenario options or other settings.
`--formats` selects the formats; `--policies combined` changes pricing compute and storage together, while `compute` and `storage` isolate them; shock generation and storage
stay FP64. The market-risk simulation has no adjustable matrix product. Today’s PV,
calibration, Greeks and final reductions retain the engine’s policy; the market-risk
P&L anchor retains its existing arithmetic. Scenario PV means that anchor plus reported
P&L, not an independent FP64 today’s PV. Greeks are not requested by this workload.

## Evidence and timing

- `report.md` joins named trade/portfolio errors, cold/compile/warm whole-job times,
  device allocator peaks, dispatch and equal-accuracy results.
- `comparison.json` retains exact workload and precision settings, seed identities,
  every warm duration, compile counts, source digests, library/platform versions, flags,
  CPU model/count/affinity, visible-device and thread-control settings,
  realized storage, warnings, per-device memory counters, per-seed errors and empirical
  reference sampling diagnostics. `manifest.json` is the complete requested matrix.
- Each point has `result.json`, a persistent `worker.log`, and per-seed `.npz` snapshots
  containing scenario shifts and named observable outputs. Keep the collection together
  when moving it; file references inside each point are relative.

Each point is a fresh process with persistent compilation caching off. Cold includes the
first whole job and its compilation; backend compile seconds are a subset of cold.
Startup, imports and campaign construction are outside cold. Other seeds reuse those
shapes; any warm compilation invalidates that timing. Different counts and formats are
intentional specializations. CPU Sobol scenario generation, host preparation, calibration,
conversions, transfers, reductions and all required host output copies are timed.
Compression and report IO are outside the clock. Device peak is the maximum of per-device
allocator high-water marks over all seeds, including compilation; it is neither host RSS
nor a sum of simultaneous device peaks. Missing counters are explicitly unavailable.

Inputs must match exactly before numerical comparison. Output comparisons preserve names
and shapes and report max absolute error, RMS, signed bias and relative error using the
reference array's largest absolute value. At zero scale relative error is undefined;
absolute error remains. The displayed engineering check is `atol + rtol * reference_scale`
(`--atol 1e-8 --rtol 1e-4` by default), **not** A-11 statistical acceptance, an ORE parity
certificate or a native-device qualification. FP64/FP32 cross-device checks can be tightened
with these flags. Non-finite outputs fail. At least 128 scenarios are required; tied P&Ls can still leave an empty strict 99% ES tail and make ES unavailable. Small counts remain smoke tests, not tail-accuracy evidence.

The equal-accuracy calculation reuses `tests.support.precision_timing`: multi-seed RMS
against CPU FP64's mean at the largest measured count, log interpolation only within
measured counts, and explicit `not reached` results. The finite FP64 mean is not truth.
Without explicit independent references, shared seeds make reference/candidate errors
dependent. Explicit `--reference-seeds` must be disjoint; reference SE and count-to-count
changes are retained, and unresolved equal-accuracy speedups are withheld. Neither mode
claims validated confidence intervals. Fixed entries excluded
by that estimator still appear in matched comparisons. A figure’s zero mean/scale can make
relative equal-accuracy estimation unavailable; its absolute matched errors remain.

For meaningful timing, use more seeds and counts from 65,536 upward as memory permits:

```bash
python -m benchmarks.comparison --out .profile-out-research --backends cpu gpu \
  --paths 65536 131072 262144 --seeds 1 2 3 4 5 6 7 8
```

Collections made on different machines can be joined without rerunning the engine:

```bash
python -m benchmarks.comparison --merge cpu-collection gpu-collection --out joined
```

Run the CPU reference in one collection (`--backends cpu`) and accelerators in another
(`--backends gpu tpu`); each can emit a partial report until joined. A missing CPU reference
is explicitly unmeasured and the command exits 1. Duplicate points and different workload,
source digests, versions or seed identities are refused. Identical CPU-generated shifts are
verified in snapshots, including across operating systems; a difference is unmeasured,
never folded into precision error. Source digests cover the engine and supporting tooling,
so save the matching source checkout, including local changes, with published evidence.

Exit 0 means collection/reporting completed, including numerical misses and explicitly
unavailable hardware. Exit 1 means failures, warm compilation or unavailable comparisons
on measured points. Partial JSON/Markdown reports are refreshed after each completed point.
This command is a checkout research tool, not part of the installed engine API.

## Recorded implementation research

The [2026-10-20 research collection](../compliance/precision-research/2026-10-20/README.md)
contains refreshed original campaigns, 4M/32-seed single-swap accuracy, larger exposure,
stress/grid holdouts, independent ORE checks, local native execution and complete-suite
evidence. All collection timings are diagnostic because work overlapped and the host was
suspended; no speed claim follows. H100 and TPU remain unmeasured.

For a separate warm engine capture on H100 after numerical/device qualification:

```bash
python compliance/precision-research/2026-10-20/reproduction/warm_native_trace.py \
  --out .profile-out-h100-warm --expected-device H100 --paths 512
```

The script warms the exact workload objects before each trace and fails if compilation
occurs inside capture. It retains twelve base/single-pass workload traces by default.
Execution evidence does not establish financial accuracy or controlled performance.
