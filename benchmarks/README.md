# Precision comparison

Run from the checkout with the pinned environment in `constraints.txt`:

```powershell
.venv/Scripts/python.exe -m benchmarks.comparison --out .profile-out-comparison
```

The defaults are a smoke study: 128 and 512 scenarios, seeds 1, 2 and 3, the single swap
and existing five-trade market-risk portfolio, CPU/GPU/TPU, FP64/FP32/BF16 and both FP8
formats. Every requested point appears in the report, including unavailable devices,
timeouts, compilation failures and formats that miss the numerical tolerance. Use a new,
empty output directory for each collection. Never run competing jobs when measuring speed.

For an NVIDIA GPU use Linux/WSL2 and the existing `gpu` environment. Select the qualified
cuBLAS route for native FP8 **before** starting the command (I-97):

```bash
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_enable_triton_gemm=false"
python -m benchmarks.comparison --out .profile-out-comparison-gpu --backends cpu gpu tpu
```

Without that flag, GPU FP8 is currently emulated; the report records the actual dispatch.
BF16/FP8 on CPU are emulated. Native dispatch alone is not hardware proof: target-device
lowering/trace checks remain I-97/I-121. A missing backend is unmeasured, never a CPU fallback.
The command does not initialize a multi-host JAX worker set; cluster qualification remains I-61.
Every worker explicitly selects its requested device, verifies the devices holding its
outputs, and refuses disabled JIT. An inherited `JAX_DEFAULT_DEVICE` cannot substitute CPU
execution for an accelerator point.

The workloads reuse `tests.support.precision_campaigns.market_risk`: a payer swap alone,
or payer/receiver swaps, a European, a fixed-LGM Bermudan and a bond on the sloped USD
two-curve market. The Bermudan uses the existing measured coarse grid (8 points per
standard deviation, 3 deviations). These are synthetic research inputs. This bounded
workload does not establish accuracy for calibrated-scenario options or other settings.
`--formats` changes pricing compute and storage together; shock generation and storage
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
with these flags. Non-finite outputs fail. At least 128 scenarios are required so ES at
99% has a tail observation; small counts remain smoke tests, not tail-accuracy evidence.

The equal-accuracy calculation reuses `tests.support.precision_timing`: multi-seed RMS
against CPU FP64's mean at the largest measured count, log interpolation only within
measured counts, and explicit `not reached` results. The finite FP64 mean is not truth.
The report retains its empirical SE across seeds; no validated confidence interval is
claimed, and shared seeds make reference/candidate errors dependent. Fixed entries excluded
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
