# Cloud failures: boundary analysis and correction plan

October 27, 2026. Analysis of the saved October 26 cloud trials, with new local
CPU/RTX 5060 experiments on JAX/jaxlib 0.10.2. No cloud resources were provisioned.
The [cloud rerun](hardware-correctness-rerun.md) remains the record of actual H100
and TPU execution. Local experiments below do not qualify a new H100/TPU fix.

The failures have several causes. Some pipeline discrepancies reproduce with
correct individual products; a new real-workload FP16 counterexample also fails
an isolated product check. Treating every discrepancy as either a compiler bug
or harmless rounding would be wrong.

| Area | Evidence and cause | Correction and status |
| --- | --- | --- |
| H100-like BF16 pipeline differences | With frozen curves, the local CPU and GPU reproduce both saved cloud European output arrays bit for bit. Small upstream floating-point differences cross operand quantization boundaries. Every captured quantization and dot in this reproduction passes its own reference. | Use identical actual operands for arithmetic checks. Keep the end-to-end flag as a portability result. Changing low-precision rounding to force equal final arrays would change the research policy. |
| FP16/BF16 final storage differences | Independent replay of 40 captured CPU/GPU storage operations agrees exactly. For example, a raw European P&L difference of 0.162258 becomes 32 after FP16 storage. | Check storage against each device's actual pre-storage values and record distance to rounding boundaries. Do not infer a broken storage conversion from unequal decoded outputs alone. |
| Native GPU FP16 on subnormal operands | The American workload contains isolated FP16 dots outside the existing arithmetic allowance. Native instructions are present; changing to an explicit FP16 algorithm does not repair the example. | A native four-product normalization prototype corrects all eight captured dots outside the standard FP32 sum bound locally. It is not enabled in engine dispatch: H100 confirmation and a performance decision are still needed. |
| Native GPU FP8 | H100 default emits FP16; the separate H100 cuBLAS numerical discrepancy still lacks intermediate buffers. Locally, the new diagnostic finds correct quantization followed by an incorrect isolated default FP8 dot. | Qualify a controlled FP8 kernel/library route, including accumulation and instructions. A locally successful explicit FP8 algorithm was rejected on H100, so it cannot be installed as a universal fix. |
| TPU FP64 | IEEE-64 bit helpers read a different software representation; this corrupts scaling and rounding. The original `frexp` also failed compilation. | Continue rejecting FP64. No FP32 downgrade or CPU fallback. Full TPU jobs remain refused because implemented fixed kernels require FP64. |
| TPU FP16 research emulation | Casting the rounded values to FP16 and then FP32 let the compiler recover an FP16 convolution, defeating the intended full-precision product. | The engine now chooses the FP32 carrier before quantization's final cast. The low-dtype round trip is removed, with local regression coverage. Actual TPU validation remains pending. |
| Missing TPU portfolio coverage | Setup, compilation and execution did not all finish within the trial budgets. | Test supported primitives first; a full FP32 internal engine profile requires implementation and separate qualification. Increasing time alone cannot repair incompatible arithmetic. |

## Pipeline analysis

The reviewed paths include precision dispatch, grid rounding, storage, execution
validation, simulation and scenario curves, interpolation, leg and European
pricing, LGM rollback, option wrapping, market-risk revaluation and reductions.
The exact failing cloud workloads were reused. This is a focused investigation
of the recorded failures, not a claim that every branch of the engine is proven.

For market risk, 256 identical host-generated shifts were passed to both devices
for seeds 7 and 8, for all five trades at FP16, BF16 and FP32. Values were captured
before storage and after decoding. Generating scenarios separately was therefore
eliminated as an explanation for these local reproductions.

The BF16 swap at seed 8 has a maximum raw difference of 63.84375. Its coupon
product inputs differ by at most about 1.86e-8 before rounding. One rounded operand
entry then differs by 1.52588e-5. The raw dot on each device agrees with the
float64 NumPy product of that device's actual quantized buffers: worst normalized
error below 7.82e-8. All eight operand quantization checks pass independently.
The dot is calculating the correct product of different rounded inputs.

Market-risk curve nodes, trade terms and the revaluation's base anchor are
constructed in the pricing accumulator dtype (`price_functions.py` and
`scenario_pnl.py`); with reduced compute that is FP32. These intermediates are
not all FP64 merely because calibration and exposure-path level kernels retain
fixed FP64 requirements. This is another reason the same input quotes alone
do not establish identical intermediate floating-point values across devices.

The cash European at seed 7 was repriced on one common scenario market. Both
local output arrays exactly match the saved CPU/H100 FP32-storage diagnostic.
The mean CPU/GPU difference at the fourth date is 1.1517754065571353. Its 120
quantization checks pass, and its 30 captured dots have worst normalized error
8.58e-8. The small pre-quantization differences occur in the interpolation,
discount/forecast and coupon calculations in `curves.py` and `legs.py`, before
the low-precision product. The captures do not single out one scalar instruction
as the source of every difference.

The FP16 Bermudan market-risk reproduction has a raw difference of 6.728515625,
which becomes 32 after storage. All 36 captured quantizations pass; its nine dots
have worst normalized error 3.83e-7. These results explain this reproduction
without establishing every other option workload's correctness.

For the current research question, the proper separation is:

1. Compare quantization with an independent grid/rounding reference on the same
   input buffers, including residual terms and scales.
2. Compare each raw dot with a mathematical reference on its actual rounded
   operands, using an accumulation bound rather than error relative to a nearly
   cancelled output.
3. Check rescaling, shared-term addition and storage separately.
4. Retain independent end-to-end CPU/device results as portability checks.
   Financial error relative to FP64 remains a different research measurement.

The old end-to-end 1e-4 comparison bound is unchanged. No failed cloud record was
rewritten or converted to a pass. A discontinuous low-precision quantizer cannot
promise that arbitrarily close inputs always produce equally close outputs.

## New native FP16 counterexample

Tracing the FP16 American at seed 8 captures 888 dots. Its GPU output matches the
saved H100 FP32-storage diagnostic exactly. The local CPU output differs from the
saved cloud CPU by up to 3.109375, so full cross-host CPU identity is not claimed.
All 3,552 CPU/GPU operand quantization checks pass. However, eight GPU dots exceed
the standard FP32 round-to-nearest accumulation bound; six also exceed the
original arithmetic campaign's 3.814697265625e-6 allowance.

The worst dot has operands shaped `[64,4,49]` and `[49,98]`. At its worst entry:

| Quantity | Value |
| --- | --- |
| NumPy product of the exact FP16 buffers | 0.0003910483792424202 |
| Native GPU result | 0.0003910437226295471 |
| Absolute-product-normalized error | 1.1908022434714266e-5 |
| Same buffers multiplied through FP32 controls, worst error | 1.7627586234793566e-7 |

Replaying the raw FP16 buffers outside the engine reproduces the captured result
exactly. An explicit `F16_F16_F32` request gives the same error. PTX contains
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`.

The counterexample also reduces to two terms in otherwise zero `[16,16]`
FP16 matrices: `A[0,:2] = [-2^-24, -18*2^-24]` and
`B[:2,0] = [-6380, -10.0390625]`. A direct FP16 matmul with FP32 output reproduces
the same error locally. This small input is suitable for a targeted H100 replay;
it does not require rebuilding the full option workload first.

The worst output multiplies FP16 subnormal inputs, including `-2^-24` and
`-18*2^-24`, by normal coefficients. Removing subnormal entries and comparing
with the correspondingly changed oracle restores the bound. This is not simple
flush-to-zero: a model flushing all subnormal inputs differs enormously from the
actual result. NVIDIA explicitly leaves MMA subnormal handling and rounding
unspecified. Native FP16 instructions plus an FP32 result dtype do not establish
the engine's desired arithmetic contract. See the
[NVIDIA MMA precision specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma).

A tested local prototype splits each operand as
`A = A_normal + 2^-24 * A_small`, with subnormal entries scaled exactly into
normal FP16 integers, and likewise for B. Four native FP16 products reconstruct
AB, with FP32 scaling/addition. Worst error on the isolated example falls to
2.0103330370903012e-7; all eight offending captured dots pass their unchanged
mathematical bounds. PTX retains FP16 MMA instructions.

This is a correction candidate, not a shipped general solution. Four products
can materially increase matrix work; detecting subnormals dynamically also costs
reductions and dispatch. A qualified implementation should keep ordinary inputs
on a fast path and handle subnormal-containing tiles explicitly. Flushing the
values or substituting FP32 products silently would change the requested policy.
The H100 trial did not capture these internal buffers, so its precise internal
subnormal behavior still needs a targeted replay on that device.

## FP8 diagnosis and reproducible boundary capture

`tests/support/product_boundaries.py` now preserves original inputs, quantized
operands, scales, rounding uniforms, raw GEMM output, reconstructed results and
an unchanged fused engine control. It records source hashes, versions, device,
dispatch controls and optimized HLO. Buffers are saved before the suspect GEMM
compiles, so a compiler refusal no longer discards its inputs.

For the local `[33,13,7]` E4M3 two-term product, both quantizations match the
independent host oracle. The default raw FP8 dot has normalized error 0.9767;
the reconstructed/fused error is 0.9055. The failure is downstream of
quantization. The separate explicit `ANY_F8_ANY_F8_F32` request gives raw error
3.28e-8 and reconstructed error 2.03e-8 locally. The unchanged default fused
control still fails. No compiler override was installed in production.

The H100 default's FP16 fallback is established by emitted instructions. Its
older cuBLAS discrepancy remains unisolated because that run saved HLO and final
errors, not the raw input/output boundaries. One mechanism worth checking is
the Hopper WGMMA FP8 accumulation behavior: NVIDIA documents internal accuracy
below full single precision even with an FP32 accumulator type. This is a
candidate explanation, not proof that the opaque cuBLAS kernel used that route
or that a fast-accumulation flag was enabled. See the
[NVIDIA WGMMA specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma).

Run boundary diagnostics on an already available device in a fresh process:

```bash
python -m tests.support.product_boundaries --platform gpu \
  --format float8_e4m3fn --shape 33 13 7 --shared right \
  --out artifacts/fp8-default

python -m tests.support.product_boundaries --platform gpu \
  --format float8_e4m3fn --shape 33 13 7 --shared right --gemm cublas \
  --out artifacts/fp8-cublas

python -m tests.support.product_boundaries --platform gpu \
  --format float8_e4m3fn --shape 33 13 7 --shared right --algorithm full-fp8 \
  --out artifacts/fp8-explicit
```

Repeat both shared sides, both FP8 formats, rounding modes and the actual failing
shapes. The diagnostic also accepts FP16/BF16 and single-pass variants. Its exit
status reports numerical checks only; it explicitly does not certify native
instructions. Compiler/library changes or an FFI/Pallas integration must pass
the native instruction gate separately, including per-operation evidence and
controlled layouts/accumulation. The successful local explicit request was
rejected on H100 in the old trial, so it is not a portable fix yet.

## TPU correction and limits

Most old TPU format failures are downstream of the same broken FP64 input-grid
path. They do not independently establish that every native BF16 operation is
wrong. FP32 compute with FP32 accumulation passed 42/42; the additional failed
FP32 cases requested FP64 accumulation. BF16 three-pass passed 42/42 because
that path casts the operands to FP32 rather than using the FP64 scaled grid.
FP32-input FP8 samples passed through an explicitly emulated route.

The FP16 carrier change in `engine/precision/products.py` addresses a separate
observed compiler pattern. Previously:

```text
grid-rounded values -> FP16 -> FP32 -> dot
```

The TPU compiler could remove the widening and run an inaccurate FP16
convolution. Now:

```text
grid-rounded values -> selected FP32 carrier -> dot
```

The requested grid, scales, residual terms and stochastic probabilities remain
the same. Explicit overflow handling retains the target encoding's infinities
or NaNs when using a wider carrier, including E4M3FN's lack of infinities.
Explicit process emulation also uses this path; native GPU operands
and default native dispatch are unchanged. Tests now check the entire traced
emulated program for low-dtype conversions, not just the dot's input dtype,
and compare results with independent NumPy/ml_dtypes operand references.
These checks prevent the known source pattern from returning; only a TPU
optimized-program and numerical replay can validate the target compiler.

Full TPU jobs, FP64 rejection, native-only FP8 rejection, execution-context
validation and numerical acceptance tolerances remain unchanged. The engine
has no qualified native TPU FP16/FP8 route. Future capability work should use
actual target/compiler evidence rather than treating that conservative engine
dispatch table as a complete specification of every TPU generation.

## Evidence and validation

Local evidence is in `artifacts/cloud-root-cause-2026-10-27/`: unchanged cloud
failure summaries, common shifts, per-trade arrays, callback traces, independent
quantization/storage checks, isolated FP16/FP8 programs, emitted PTX, source
hashes and test XML. Instrumented outputs were checked against separate
uninstrumented runs and were bitwise identical for all four traced workloads.
The traces remain local evidence, not substitutes for device qualification.

- 237 CPU product/qualification tests passed; five GPU-only cases skipped.
- 580 CPU policy, grid, storage, reporting, run and market-risk tests passed
  across the main run and the targeted recheck. Two old spies were updated to
  the private model-builder entry introduced by the earlier validation refactor.
- 36 CPU boundary-diagnostic tests passed, including injected wrong
  quantization, wrong products and compiler refusals.
- 80 local GPU regression checks passed, including native BF16 vector controls,
  explicit FP32-carrier numerical checks and diagnostic regression tests.

The confirmed negative numerical experiments above are retained separately
from passing regression tests. No dependency versions were changed, no
financial tolerance was widened, and no cloud resource or shared project
configuration was changed.
