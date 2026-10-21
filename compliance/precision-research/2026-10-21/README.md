# Native execution defaults

2026-10-21. Engine digest `0d67abfbb8ee40fc`; change based on commit `dafbce5` with local
changes. Normal GPU usage requests native arithmetic whenever the backend/device supports
the format. A local compiler fault requires an explicit process/test setting.

## Behavior

- FP8 native dispatch no longer requires the cuBLAS flag. FP16 remains native on capable
  GPUs. Unsupported backend/device combinations retain emulation.
- BF16_3x requests its native algorithm for vectors as well as matrices. Its local vector
  fusion workaround is no longer a blanket fallback on every GPU.
- Comparison `--gemm auto` preserves inherited emitter settings for every format. Device
  qualification also inherits settings when `--gemm` is omitted. Explicit `default` and
  `cublas` selections remain available before backend initialization.
- `JAX_RISK_EMULATE_PRODUCTS` explicitly emulates comma-separated compute formats; a base
  name includes its single-pass variant. Rounding/scaling is preserved, and products use
  FP32 carriers/accumulation. Invalid formats fail. Reports identify emulated dispatch;
  campaign metadata retains the setting and XLA flags, and incompatible records cannot merge.
- Qualification recognizes FP8 cuBLAS calls and Triton FP8/FP32 dot fusions. It verifies the
  actual called computation's operand/result types; an unrelated conversion or fusion does
  not count. Lowering still requires device traces to establish actual instruction use.

Settings must precede tracing in a fresh process. Changing the environment does not change
cached JAX executables. Importing the engine sets no workaround, and no GPU-name blacklist
or automatic emulation is introduced. The default precision policy remains FP64.

## Verification

The new default/override regressions were run against the previous implementation:
**17 failed, five passed**. Final scoped CPU validation: **480 passed, two accelerator-only
skips**. Four additional real process checks pass: both comparison workloads, device CLI
controls preserving the inherited emitter, and resumable isolated accuracy collection.
These are scoped checks for this revision; the earlier full 4,011-case audit belongs to
the [2026-10-20 source snapshot](../2026-10-20/README.md).

Real GPU checks use the RTX 5060 Laptop GPU, JAX 0.10.2, deterministic kernels. FP8 covers
both formats, two-term/single-pass controls, both shared directions and rounding modes,
odd matrix and matrix-vector shapes, wide dynamic range and cancellation. BF16_3x covers
matrix/vector products and both shared directions. Each row compares with CPU emulation.

| Explicit process configuration | Dispatch | Numerical passes | Observed FP8 lowering |
|---|---|---:|---:|
| Normal FP8 default emitter | Native | 26/64; failed qualification | 40 Triton fusions |
| FP8, cuBLAS override | Native | 64/64 | 40 cuBLAS calls |
| Normal BF16_3x default | Native | 6/8; failed qualification | Not applicable |
| BF16_3x explicit emulation | Emulated | 8/8 | Not applicable |

The failed defaults are retained as failures. The cuBLAS setting is an explicit workaround
for this measured runtime/device, not an H100 default. H100 remains unmeasured. These tests
check product arithmetic; they do not establish financial accuracy, universal native
instruction use, performance, or a complete-suite pass on this revision.

## Accuracy implications

This dispatch change does not improve FP8's numerical resolution or change A-11. Existing
FP16/FP8 arithmetic, storage scaling, shared-operand terms and acceptance thresholds remain
the same. The [larger studies](../2026-10-20/README.md#larger-counts-expose-precision-floors)
still apply to their archived configurations: FP16 ES can pass where its mean/VaR fails;
E4M3 passes none of the selected large single-swap figures. Precision error, sampling
convergence, grid error and hardware/compiler correctness remain separate checks.

## Reproduction and artifacts

Normal usage needs no emulation setting. On the affected local test device only:

```bash
XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \
JAX_RISK_EMULATE_PRODUCTS=bfloat16_3x \
python -m tests.support.precision_devices --platform gpu --out .profile-out-local-device
```

For H100, begin with the [normal qualification command](../../../benchmarks/README.md#h100-native-qualification),
then compare explicit emitter settings in a separate collection. Numerical failures remain
failures; native labels alone never establish qualification.

| File | Evidence |
|---|---|
| [summary.json.gz](summary.json.gz) | Counts, engine digest, actual device and explicit process flags |
| [default.json.gz](default.json.gz), [cublas.json.gz](cublas.json.gz) | Every FP8 operation, error, tolerance, dispatch and observed lowering |
| [bfloat-native.json.gz](bfloat-native.json.gz), [bfloat-workaround.json.gz](bfloat-workaround.json.gz) | Failed native and passing explicit-emulation BF16 controls |
| [optimized-lowering.zip](optimized-lowering.zip) | All optimized operation HLO, including failed default routes |
| [red.log.gz](red.log.gz) | New regressions failing on the old implementation |
| [final-targeted.log.gz](final-targeted.log.gz), [JUnit](final-targeted.xml.gz) | 480 passed, two skipped |
| [process.log.gz](process.log.gz), [JUnit](process.xml.gz) | Four passing real process checks |
| [source.zip](source.zip), [gpu_check.py](gpu_check.py) | Source snapshot, pins and exact local GPU-check script |

Run the local script from a direct child directory of a fresh checkout, for example
`.profile-out-native-defaults/gpu_check.py`, with `default`, `cublas`, `bfloat-native` or
`bfloat-workaround`. It writes the corresponding collection under `.profile-out-native-defaults`.
The default checks intentionally exit nonzero when reproducing this local compiler fault.
`SHA256SUMS` identifies all published files. No historical acceptance records were changed
or experimental limits promoted by this follow-up.
