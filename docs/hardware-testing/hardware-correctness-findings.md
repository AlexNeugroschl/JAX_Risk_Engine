# H100 and TPU failures: causes, fixes and remaining evidence

Analysis of the October 26, 2026 trial at engine revision
`f9cc4c16fa843859729a9fb0956dd8cdfd2fa09b`, JAX/jaxlib 0.10.2 and libtpu 0.0.42.
The [original results](../../artifacts/accelerator-native-2026-10-26/RESULTS.md)
remain the record of what ran. Changes described here were made afterward.
No new cloud resources or dependency upgrades were used for this initial analysis.

**Subsequent actual-hardware rerun:** the H100 BF16 three-pass fix passed, but the
proposed TPU fixes failed validation. TPU FP64 bitcasts do not expose the IEEE-64
words assumed below, and the compiler narrowed widened FP16 operands back to an
FP16 convolution. The [rerun results](hardware-correctness-rerun.md) supersede the
pending/portability claims in this initial analysis. Full native-precision
qualification still fails.

Most arithmetic cases passed. Full qualification failed because specific paths were
incorrect or unverified, and some requested operations were unsupported. A compiler
error, an unsupported instruction, a timeout and a numerical discrepancy require
different fixes. Bitwise repeatability alone does not establish correctness.

## What is established

| Failure | Cause established by evidence | Proper treatment |
|---|---|---|
| H100 BF16 three-pass matrix-vector arithmetic | Optimized HLO multiplies split pieces in BF16 and widens their already-rounded products to FP32. It also truncates the high pieces. | Keep these shapes on the native GEMM path with zero padding and slice the result. The subsequent H100 rerun passed 60 targeted arithmetic cases, three regressions and five engine workloads with native BF16 instruction evidence. |
| Default GPU FP8 native claim | Typed FP8 HLO is not physical instruction evidence. The H100 run emitted FP16 matrix instructions. The previous gate needed only one observed HLO shape per format. | Require numerical correctness and matching FP8 instruction evidence for every operation. Implemented. Missing or opaque instructions remain unverified. |
| Local GPU native FP8 numerical failures | Identical CPU/GPU quantized operands produce wrong values in the isolated FP8 dot; FP16 controls and explicit full-accumulation FP8 calls agree with the oracle. | Isolate the compiler emitter/library route and request an explicit arithmetic algorithm. The local successful route is not yet a qualified H100 fix. |
| H100 cuBLAS FP8 discrepancies | Same-policy arithmetic failed, but the saved data do not identify the first faulty boundary. Fast accumulation is not an established explanation. | Capture operands, scales, raw GEMM output and rescaling separately on H100; qualify a corrected compiler/library route with actual device evidence. Remains unresolved. |
| TPU precision scaling compiler errors | FP64 `jax.numpy.frexp` reproduces a compiler type error. The replacement's FP64 bitcasts expose different words on TPU than on CPU/GPU. | The proposed replacement failed its TPU rerun: it compiles but returns wrong FP64 exponents/scales/rounding. A backend-aware implementation must avoid the IEEE-64 bitcast assumption and retain the original rounding contract. |
| TPU raw FP16 product discrepancies | Saved matrix errors match a truncated BF16 decomposition. The rerun's optimized program narrows the new FP32 carriers back to FP16 convolution operands. | Widening alone failed validation. Keep emulated quantization outputs in FP32 without a low-dtype round trip and qualify an explicit full-precision arithmetic route before promotion. No native FP16/FP8 TPU pass is claimed. |
| TPU TF32 and FP64 LU solve | The backend explicitly rejects these requested operations. TF32 is an NVIDIA-specific algorithm; v6e has no native FP64 hardware. | Report unsupported capability. Emulation or an explicit CPU stage can be a separate policy, never a silent native pass. A generic LU refusal does not prove that every engine workload needs LU. |
| Two TPU FP64 arithmetic discrepancies | Software FP64 missed the unchanged bound in two batched cases. Exact internal cause is unisolated. | Retain the failures; test software FP64 separately or use CPU/GPU for a native FP64 requirement. Do not loosen the bound simply to obtain a pass. |
| CPU/accelerator pipeline differences | End-to-end comparisons confound scenario generation, intermediate rounding, storage and pricing. Several low-storage differences are adjacent-bin-sized, but that does not establish their cause. | Freeze stage inputs and compare each boundary before diagnosing an individual operation. Retain independent end-to-end comparisons as separate tests. |
| Calibrated TPU timeout / Spot preemption | No completed numerical result before the deadline / provider reclaimed capacity. | Record incomplete coverage / infrastructure interruption. Use checkpoints and separate setup, compile and execution budgets. Neither event proves an arithmetic bug. |

## Native BF16 three-pass fix

The H100 artifact
`h100/evidence/arithmetic-bfloat16_3x/operation-0006.hlo.txt` contains BF16
`multiply` operations followed by FP32 conversions and reduction. FP32 accumulation
cannot recover bits already discarded by BF16 multiplication. This explains errors
near `1e-3` instead of the three-pass algorithm's approximately `1e-5` allowance.

The same failure reproduced locally on the RTX 5060 with JAX 0.10.2: normalized
error `0.00118182` for `[512,64,1]`. Merely expressing three explicit BF16 dots did
not fix the GPU GEMV expansion. Padding free matrix dimensions to at least eight,
retaining `BF16_BF16_F32_X3`, and slicing the output did. The inserted values are
zero; the mathematical product and output shape are unchanged. This costs extra
work for small shapes, but does not switch to CPU or FP32 emulation.

`engine/precision/products.py::_matmul32` implements this only for GPU BF16
three-pass matrix-vector shapes. CPU emulation and TPU's working native BF16
three-pass path retain their existing routes. Local validation passed all 42
original arithmetic cases plus 18 cases covering vector/vector and very small
dimensions. Emitted PTX includes:

```text
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
```

The three BF16-three-pass pipeline regression identities inferred from the H100
log also pass on the local GPU after this fix. These checks support the fix, but
did not substitute for a rerun on the actual H100 compiler/driver environment.
That subsequent rerun has now confirmed this fix for the recorded coverage.

## FP8: what the new isolation establishes

The local experiment captures quantized operands before the dot and compares them
with the CPU's quantization. For both `[16,32,16]` and `[33,13,7]`, they are identical.
An isolated default FP8 dot then has normalized errors of approximately `0.801` and
`0.933`; a fused engine product reproduces those errors. This excludes the
engine's quantization and final rescaling as the cause of those two **local**
failures. Native FP8 MMA instructions are present in these wrong kernels.

FP16 controls on those same operands have errors `0` and `5.33e-9`. Explicit
`ANY_F8_ANY_F8_F32` requests produce the same good numbers locally, and the saved
HLO routes them to an FP8 cuBLASLt custom call. Its library instructions were not
extracted, so it remains a numerically successful candidate rather than a complete
native-instruction qualification. The H100 trial rejected that explicit request
with `Unsupported dot precision algorithm: 1`, even with cuBLAS selected. Installing
the local successful request as a universal fix would therefore be incorrect.

The pinned XLA source is especially useful here:

- Its GEMM rewriter explicitly contains an FP8-to-FP16 fallback when an FP8 rewrite
  cannot be selected. The dispatch label cannot certify hardware execution.
  [Pinned GEMM rewriter](https://github.com/openxla/xla/blob/5a9e73cbd92530cac2ac36f4736a774b2412afe2/xla/backends/gpu/transforms/gemm_rewriter.cc).
- `GemmConfig` takes the maximum operand precision; the recorded H100 custom calls
  contain `HIGHEST,HIGHEST`. Its cuBLASLt `IsFastAccumEnabled` enables legacy FP8
  fast accumulation only for default precision. The source therefore does **not**
  support blaming an enabled fast-accumulation flag for those recorded H100 failures.
  [Pinned configuration](https://github.com/openxla/xla/blob/5a9e73cbd92530cac2ac36f4736a774b2412afe2/xla/service/gpu/matmul_utils.cc),
  [pinned cuBLASLt code](https://github.com/openxla/xla/blob/5a9e73cbd92530cac2ac36f4736a774b2412afe2/xla/stream_executor/cuda/cuda_blas_lt.cc).

The proper remaining H100 experiment is concrete: capture CPU and GPU quantized
values/scales; feed those exact buffers to an isolated native FP8 GEMM; compare
its unscaled FP32 result with a high-precision reference; then compare rescaling
and shared-term summation. Repeat for the failing shapes, both FP8 variants,
nearest/stochastic rounding and both shared operands. This separates compiler,
library/kernel and engine boundary errors. A compiler/library update or a dedicated
cuBLASLt/Pallas kernel must pass those checks before promotion. An FFI integration
should explicitly control its operand layouts, output type and accumulation mode.
No such unverified kernel or dependency change was shipped in this analysis.

The corrected `--require-native-fp8` collects compiler PTX and checks every operation,
alongside native dispatch, typed lowering and numerical results. Opaque vendor code
and cached/missing dumps fail the strict evidence gate without being called emulation.
On the local GPU, a complete single-pass E4M3 run had 50/84 numerical passes and
matching FP8 instruction evidence in 60/84 operations. The combined gate correctly
failed. Actual device traces remain necessary when vendor kernels are opaque.

## TPU scaling and emulation fixes

The minimal saved reproduction makes the scope clear: FP32 `frexp`, `scale_exponent`
and `round_to_grid` run; their FP64 counterparts fail during compilation. The failing
instructions are signed/unsigned bitwise operations created while expanding FP64
`frexp`. This occurs without a portfolio or calibration, and is a pinned-backend
compatibility failure rather than evidence of defective TPU hardware.

`engine/precision/grid.py` now derives the exponent from IEEE fields using uint32
words, including subnormals, signed zero and non-finite handling. FP64 powers of two
are assembled from two 32-bit words. Scaling compares an exactly power-of-two-scaled
value with the target maximum; grid rounding still occurs in the input dtype.
It preserves float64-to-FP16/FP8 halfway decisions that would be lost by casting
through FP32. CPU and local GPU tests cover random bit patterns, format boundaries,
every normal power of two and double-rounding examples. The subsequent TPU rerun
failed: its FP64 bitcast of `1.0` exposes high word `0x3f800000`, rather than IEEE-64
`0x3ff00000`, so this helper incorrectly returns exponent `-6` instead of `1`.

For FP16, flushing subnormals does **not** explain the saved failures: some failing
inputs contain no FP16 subnormals. An independent NumPy model splitting the FP16
values into truncated BF16 high/low pieces and omitting the low-by-low product
reproduces the seven failing raw cases to within `8.72e-8` normalized error, with
two cancellation cases matching exactly. The remaining small differences fit FP32
reduction rounding. This strongly identifies a reduced-accuracy BF16 decomposition
of the FP16 matrix path; it is not evidence of correct native FP16 multiplication.

The engine already labels FP16/FP8 emulated on v6e, but previously used FP16 carriers
for those products. That exposed it to the backend's reduced-accuracy FP16 dot.
The corrected emulated route uses FP32 carriers for the **already quantized** values
and requests `HIGHEST`; it retains the emulated label. It changes the carrier, not
the requested operand grid or accumulation contract. Native BF16 remains native.
The subsequent TPU rerun found that the compiler removes the intended widening
for FP16. Its optimized convolution still takes FP16 operands, and the isolated
FP32-input arithmetic checks still fail. This proposed change is therefore not a
qualified TPU FP16 fix. See the rerun report for separate FP8 sample coverage.

True native FP64 on TPU cannot be created by fixing a Python function. If native
execution is mandatory, TPU needs a separately designed policy that explicitly
uses its supported operations; the engine's current fixed FP64 stages cannot
silently become FP32. CPU calibration/reference stages would make a hybrid policy,
not an all-native TPU pass. [JAX accuracy and TPU FP64 limitation](https://docs.jax.dev/en/latest/accuracy.html),
[JAX dot-algorithm semantics](https://docs.jax.dev/en/latest/201/precision.html).

## Pipeline flags need stage-level diagnosis

Examples from the saved H100 outputs include an FP16 option NPV-cube difference of
64 against a comparison bound of 16.93, and market-risk P&L differences of 16 or 32.
Those are compatible with adjacent low-storage bins, but saved final arrays alone
cannot establish whether input differences, reduction order, quantization or a
pricing operation first caused them. Some FP32-storage diagnostics also failed,
so storage alone is not a complete explanation.

The TPU's smaller FP64 workload has an almost common NPV offset around 0.056 across
paths and seeds. Its exact cause remains unresolved. The generic function probe's
simple sum of ones/negative ones is too weak to certify fractional FP64 reductions;
dtype labels and a few easy scalar cases cannot certify an entire pipeline.

The proper test sequence is:

1. Store common host uniforms/normals and fixed model/calibration inputs. Transfer
   the same values to both devices. Test native scenario generation separately.
2. Capture each stage before storage, quantized values/scales/offsets, and loaded
   values. Find the first disagreement, rather than starting from the final NPV.
3. At that boundary compare the **same actual operands** with an independent
   mathematical oracle and an operation-specific rounding bound. Include reduction
   cancellation, mixed magnitudes, halfway values and inverse-normal CDF cases.
4. Re-run the independent end-to-end experiment and within-device repeats. Keep
   arithmetic correctness, scenario-generation portability and financial error
   versus FP64 as separate conclusions. Do not widen the old tolerance to erase a flag.

The calibrated TPU runs timed out before completed results; the smaller uncalibrated
swap workload is additional evidence with narrower coverage. Increasing time alone
does not fix a compiler type error. Future runs should record setup, cold compilation
and execution phases independently and checkpoint before the cloud deletion deadline.

## Validation and remaining qualification

Completed without cloud spending:

- 182 CPU grid/product/scaled-storage regressions passed.
- Eight additional CPU strict-promotion/level regressions passed.
- 39 CPU qualification/default tests passed; seven hardware-dependent tests skipped.
- 31 local GPU grid/qualification regressions passed.
- 60 local GPU BF16 arithmetic cases passed, with native BF16 MMA PTX captured.
- All three previously failing BF16 pipeline regression identities passed locally.
- The corrected strict FP8 gate rejected the failing 84-operation local run.
- FP8 boundary experiments retain exact inputs, reference values, outputs and HLO.

Local evidence is in `artifacts/hardware-failure-analysis-2026-10-26/`, including
JUnit files, original/padded HLO, PTX, FP8 boundary NPZs and pinned upstream sources.
Installed dependency versions and original acceptance tolerances were not changed.
The H100 and TPU were not reprovisioned for the initial analysis. The subsequent
authorized rerun confirmed H100 BF16 correctness and rejected the proposed TPU
grid/FP16 fixes. The unresolved FP8/pipeline investigations and TPU fixes remain
requirements before claiming full cross-hardware qualification.
