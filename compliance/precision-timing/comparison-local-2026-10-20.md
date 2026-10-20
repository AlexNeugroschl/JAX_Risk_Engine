# Joined comparison smoke record — 2026-10-20

This is local tooling evidence on WSL2 Ubuntu, an Intel Core Ultra 9 275HX (24 logical CPUs
available to the worker) and one NVIDIA GeForce RTX 5060 Laptop GPU (driver 592.01), not the
F-07 target-device research result or native-unit qualification.
The [compressed report](comparison-local-2026-10-20.json.gz) retains every point's settings,
versions, source digests, flags, timings, memory counters, per-seed named errors and empirical
reference diagnostics. It contains report numbers, not the individual snapshot files; those
remain in the local, ignored `.profile-out-comparison-release/` collection. Recreate that
collection using the matching source checkout and the command below.

```bash
XLA_FLAGS=--xla_gpu_enable_triton_gemm=false python -m benchmarks.comparison \
  --out .profile-out-comparison-release --paths 128 512
```

The defaults select both workloads, seeds 1/2/3, CPU/GPU/TPU, FP64/FP32/BF16 and both FP8
formats. Core versions: JAX/jaxlib 0.10.2, ORE 1.8.16.0, NumPy 2.4.6, SciPy 1.17.1.
Full package/accelerator-library versions and process flags are in the compressed report.
All measured points share the current engine and tooling source digests. Windows also
rebuilt the report from this Linux collection with `--merge`, without executing the engine.

| Observation | Result |
|---|---|
| Requested points | 60 |
| Completed CPU/GPU points | 40 |
| Unavailable TPU points | 20, explicitly unmeasured |
| Failed workers / invalid timings | 0 / 0 |
| Warm compilations | 0 across all 40 measured points |
| Exact source-shock checks | Passed on every compared seed |
| FP64/FP32 engineering tolerance | All 16 points passed |
| BF16/FP8 engineering tolerance | All 24 points missed; retained as valid numerical results |
| GPU FP64, tighter check | 192 named checks within `1e-8 + 1e-10 * reference_scale`; largest relative error 3.53e-15 |
| Execution | CPU BF16/FP8 emulated; GPU BF16/FP8 native dispatch through the configured route; output placement verified |
| Memory | Per-device allocator high-water marks recorded on GPU; CPU counters and host RSS unavailable |

The default check is `1e-8 + 1e-4 * reference_scale`, not A-11 acceptance. Native dispatch
labels do not prove every product uses native units; representative-shape lowering/trace
qualification remains I-97/I-121. No speedup conclusion is drawn from these small counts.
The finite, shared-seed FP64 reference and descriptive equal-accuracy interpolation are
recorded without supported confidence bounds. Larger counts/more seeds and target hardware
remain the next research work. See [method and usage](../../benchmarks/README.md) and
[execution plan](../../docs/planning/details/comparison-report.md).
