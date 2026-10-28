# Native TPU probes: FP8 matrix candidates pass; FP16 remains unsuccessful

October 27, 2026, New York time. One v6e chip; correctness testing, not a benchmark.
This investigation follows the [earlier emulation survey](tpu-precision-survey.md).

**Passing FP8-typed TPU matrix paths now exist for E5M2, E4M3FN and the distinct
E4M3B11FNUZ encoding.** Ordinary typed JAX products and Pallas kernels pass the
tested arithmetic and range checks. This is stronger evidence than the earlier
FP32-carrier emulation result. It does not establish physical FP8 instructions or
a speed gain: the retained evidence is optimized HLO and Mosaic compiler IR.

FP16's explicit native algorithm is unsupported. Its unconstrained typed products
miss the existing arithmetic tolerance, including with `HIGHEST`. TF32's native
algorithm is unsupported. FP64 requests remain explicitly refused. Production
dispatch and API behavior were not changed during this investigation.

## Results

The [probe runner](../../tests/support/tpu_native_precision.py) calls JAX and Pallas
directly. It bypasses the engine's conservative capability table rather than
relaxing its native gate. No diagnostic catches a failed native request and
replaces it with an emulated product. Compiler widening is inspected separately.

| Candidate | Ordinary typed JAX product | Pallas matrix kernel | Explicit algorithm request |
| --- | --- | --- | --- |
| FP8 E5M2 x E5M2 | 20/20 numerical cases pass | 3/3 pass | `ANY_F8_ANY_F8_F32` rejected on all 5 shapes |
| FP8 E4M3B11FNUZ x itself | 20/20 pass | 3/3 pass | Same rejection |
| FP8 E4M3FN x itself | 20/20 pass | 3/3 pass | Same rejection |
| Four mixed FP8 pairs, listed below | 80/80 pass | 12/12 pass | Same rejection |
| BF16 | 15/15 pass | 3/3 pass | `BF16_BF16_F32`: 15/15 pass |
| FP16 | `DEFAULT`: 10/20 pass; `HIGHEST`: 16/20 pass | Kernel rejected | `F16_F16_F32` rejected on all 5 shapes |
| FP32 | `HIGHEST`: 15/15 pass | Not repeated | `F32_F32_F32` rejected for 4 matrix/batch shapes; vector shape passes 3/3 |
| BF16 three-pass | Not repeated | Not repeated | `BF16_BF16_F32_X3`: 15/15 pass within its separate allowance |
| TF32 | No substitute tested | No substitute tested | `TF32_TF32_F32` rejected on all 5 shapes |
| FP64 | No software FP64 executed | Not attempted | Engine compute and accumulation requests both refused |

The mixed pairs are E5M2 x E4M3B11FNUZ, the reverse ordering, E5M2 x E4M3FN,
and the reverse ordering. E4M3B11FNUZ is an explicit, distinct experiment; it was
never substituted for E4M3FN. Its maximum is 30, versus E4M3FN's 448.

Raw coverage comprises 115 compilation attempts: 66 compiled, with 229/243
completed numerical cases passing after the diagnostic replay described below.
All 14 numerical failures are FP16. Compiler rejections are separate outcomes,
not numerical passes or inferred hardware failures.

### What stayed FP8 and what widened

For each of the seven FP8 pairs, four shapes retain the requested FP8 operand
types in optimized matrix operations: square `(256,256,256)`, small `(32,32,8)`,
odd `(33,13,7)`, and batched `(2,9,13,7)`. Their 112 raw numerical cases pass.
Pallas also compiles explicit `tpu.matmul` operations with the selected FP8 inputs
and FP32 accumulation; all 21 FP8 Pallas cases pass.

The `(32,32,1)` matrix-vector shape produces **FP32 vector multiplication and
reduction** in optimized HLO. Its 28 FP8 numerical cases pass, but they are not
native FP8 matrix-compute passes. This is the compiler's chosen lowering, even
though the diagnostic itself requests no fallback. A future integrated native
path needs shape-aware validation or an explicit padded/custom-kernel experiment.
The ordinary BF16 vector route similarly widens; the explicit BF16 preset retains
typed matrix operations in this coverage.

The installed JAX capability helper declares no-cast support for E5M2 and
E4M3B11FNUZ on v6e, but not E4M3FN. Actual E4M3FN kernels nevertheless compile and
pass these checks, including across their range. Static capability metadata alone
therefore neither proves correctness nor completely describes accepted compiler
paths. No physical instruction trace was collected; E4M3FN's discrepancy with the
static helper remains a reason for caution before declaring a hardware route.

### Fused and range coverage

All **40 fused diagnostic cases pass**, including quantization against an
independent NumPy reference, the raw product on those quantized operands, final
rescaling/reconstruction, and repeatability. These use odd `(33,13,7)` matrices,
nearest/stochastic rounding, both shared sides, and basic/compensated profiles.
Of these, 32 are FP8 and 8 are BF16. Their optimized matrix operations retain the
selected low operand types. The compensated FP8 cases include scaling and two
shared residual terms. These are diagnostic candidate kernels using engine
quantization helpers, not full jobs through the engine's current dispatch gate.

All **14 exact range-boundary cases pass**: seven FP8 operand pairs through both
ordinary JAX and Pallas. Sparse diagonal matrices exercise each encoding's smallest
subnormal, smallest normal, largest finite value, signed values and powers of two
across its exponent range. There is one nonzero contraction per output, so each
FP8 product is exactly representable in FP32; the reference requires exact array
value equality, without a numerical tolerance.

Ordinary raw cases use identical quantized operand bytes and host NumPy FP64
product references, normal/zero/cancelling inputs, plus FP16/FP8 small-grid cases.
The ordinary normalized arithmetic allowance remains `3.814697265625e-6`; the X3
algorithm keeps its separate `3.0517578125e-5` allowance. All repeated numerical
executions return equal array values. `_1x` names share the same raw dtype kernel;
one/two shared-term behavior is exercised explicitly in fused coverage.

FP16's largest failing normalized errors are about `7.76e-6` with `HIGHEST` and
`2.14e-3` with `DEFAULT`. The prior passing FP32-emulation fix did not repair these
unconstrained low-dtype products. Correct emulation and correct native arithmetic
remain separate results.

## Diagnostic corrections retained in the evidence

The first raw pass stopped 60 cases at a host byte-view check because some TPU
arrays copy back in a different memory layout. Making both host views contiguous
repairs the comparison. Only those 60 cases were replayed; 48 pass and 12 have
genuine FP16 numerical failures. Two additional FP16 failures were already measured
in the initial pass. The final totals replace those diagnostic errors rather than
counting the replay as additional independent coverage.

The first Pallas attempt rejected its BF16 control and FP8 kernels because Python
zero constants became i64 index-map returns under the engine's x64 setting. Explicit
i32 index-map constants repaired that shared harness issue. The replay compiled
all seven FP8 pairs and BF16; FP16 still failed with an unsupported vector-load
type. Both initial errors and corrected IR/results remain saved. Production
arithmetic, rounding policy and acceptance tolerances were unchanged.

## What this means for the engine

The blanket TPU FP8-emulation dispatch is overly restrictive for the matrix
candidates just tested. The explicit generic FP8 preset is also not a portable
way to request them on this compiler. Integration should select a measured typed
JAX/Pallas route explicitly and report shape-dependent widening, unsupported
algorithms and unverified instruction use accurately.

The engine still routes TPU FP8 through its earlier emulation path; this testing
task did not change runtime dispatch, add E4M3B11FNUZ to the public format/API table,
or promote native qualification. Full TPU jobs also remain unsupported because
their fixed internal kernels require FP64. No H100 rerun or performance study was
performed. Native low-format compute, physical instruction qualification, financial
accuracy and useful end-to-end speed are distinct questions.

For v6e, JAX's hardware reference lists equal BF16 and FP8 peak rates; smaller FP8
storage can still reduce memory traffic, but a speed gain needs measurement on the
engine's workloads. [JAX TPU hardware reference](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html).

## Reproduction, evidence and cleanup

One `ct6e-standard-1t`, `TPU v6 lite`, in `europe-west4-a`; JAX/jaxlib `0.10.2`,
libtpu `0.0.42`, NumPy `2.4.6`, repository constraints. The initial Tokyo request
was rejected for capacity; absence of matching resources was verified before
retrying in Europe. No larger machine, additional simultaneous VM or reservation
was used. Existing CT6E quota was 48 in the selected region.

Main command on the prepared temporary VM:

```bash
JAX_PLATFORMS=tpu,cpu timeout --signal=TERM --kill-after=30s 1800s \
  .venv/bin/python -u -m tests.support.tpu_native_precision \
  --platform tpu --out evidence/survey
```

The runner isolates raw, Pallas and fused phases with independent process limits.
Exit zero means the survey completed; individual compiler and arithmetic outcomes
are in its JSON summaries. The VM had a 45-minute automatic DELETE limit and an
auto-deleting 100 GB boot disk. The service used `LimitMEMLOCK=infinity` and a
35-minute limit; targeted supplements had 120–180-second limits.

Evidence is under `artifacts/tpu-native-2026-10-27/`: original source archive and
manifest, `summary.json`, extracted per-phase JSON, exact operand/result NPZ files,
optimized HLO, Mosaic IR, package versions, original/replayed diagnostic source
hashes, cloud receipts and cleanup status. Source archive SHA256:
`9e95e21e4ff44dc552955d9bc21e97c1c2b24750524c3ee3ef326bf98b9ee0f9`.
Both downloaded archive hashes match the VM:

- Main: `a5844f3fb20322e0e255a9fff1330d120a1edcb5f589124ecf5ebbcace0d8314`.
- Range supplement: `004cd092d31a2c399e5ee3c90c5002138f82b0909a3424b0ee93abeb0648878f`.

The VM and boot disk were manually deleted after evidence verification. Successful
API receipts verify no matching VM, disk or reserved address remains. The unique
temporary OS Login registration and local key files were removed. No shared IAM,
network, API, quota, reservation or billing configuration was changed.

Creation through verified absence took about 15.5 minutes. At the listed regional
v6e rate of $2.97 per chip-hour, the conservative successful-machine estimate is
**$0.77**, excluding ancillary charges/taxes; this is not a verified invoice.
The 45-minute machine ceiling was about $2.23, within the prior $20 allowance.
[Google TPU pricing](https://cloud.google.com/tpu/pricing).
