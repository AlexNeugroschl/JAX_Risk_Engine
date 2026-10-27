# Hardware correctness rerun: H100 fix passes; TPU fixes fail

October 26, 2026. Correctness testing; no performance conclusion.

Subsequent implementation: the [explicit execution policy](explicit-precision-policy.md)
now refuses full TPU jobs with their fixed FP64 requirements, implicit product
emulation, and native-only FP8 on the unqualified compiler path. It does not change
the numerical findings below or establish new hardware correctness.

**The H100 BF16 three-pass fix worked. The proposed TPU fixes did not establish correct TPU execution. Do not use this source revision for TPU precision research requiring the FP64 grid or FP16 emulation.**

The actual TPU rerun found incorrect FP64 exponent/rounding values and a compiler rewrite that defeats FP32 carriers for FP16. Passing local CPU/GPU tests had not exposed either issue. No numerical tolerance or production arithmetic was changed during this rerun. Post-test documentation and code comments now flag the TPU incompatibility.

## Reproduction and environment

Both devices used the same frozen working-tree source snapshot, based on commit `f9cc4c16fa843859729a9fb0956dd8cdfd2fa09b`. Archive SHA256: `356f0c3b3f1efa103a5c47ccca45ea2e55082052f2baa66cb5c42778bc99c620`. The source manifest records individual file hashes, including the uncommitted fixes. JAX/jaxlib remained 0.10.2; TPU libtpu remained 0.0.42. Packages, device reports, flags, HLO and process receipts are retained.

H100: NVIDIA H100 80GB HBM3, driver 615.71.10, `a3-highgpu-1g` Spot in `us-central1-c`. The first zone rejected the request for capacity; absence was verified before retry. TPU: `TPU v6 lite`, `ct6e-standard-1t` STANDARD in `asia-northeast1-b`. One accelerator VM ran at a time.

## Fix verdicts

| Change | Actual hardware result | Verdict |
| --- | --- | --- |
| H100 BF16 three-pass vector padding | 60/60 targeted arithmetic cases; native BF16/FP32-accumulation matrix instructions; 3/3 formerly failing regressions; 5/5 primary engine workloads | Confirmed for this coverage |
| Stricter native FP8 gate | 336/336 FP8 numerical cases pass, but 0/336 have matching native FP8 instruction evidence; emitted FP16 instructions are retained | Gate correctly rejects; native FP8 remains unqualified |
| TPU precision-grid replacement | FP32 targeted checks pass; FP64 exponent, scaling and rounding checks fail | Proposed TPU fix failed |
| TPU FP32 carriers for emulation | FP8 FP32-input samples pass; FP16 still fails, and optimized HLO uses FP16 convolution operands | Partial evidence only; FP16 fix failed |

## Arithmetic coverage

Passed / completed numerical cases below use the original tolerance. The native FP8 gate is an additional requirement and fails despite the numerical passes. Samples are separate from the full seven-shape campaign.

| Format | H100 full arithmetic | TPU full arithmetic | TPU separate sample |
| --- | --- | --- | --- |
| `float64` | 42/42 | 40/42 | — |
| `float32` | 84/84 | 50/84 | — |
| `float16` | 84/84 | 4/84 | — |
| `bfloat16` | 84/84 | 10/84 | — |
| `float8_e4m3fn` | 84/84 | 4/84 | — |
| `float8_e5m2` | 84/84 | 4/84 | — |
| `tensorfloat32` | 84/84 | incomplete (no saved cases) | 0/4 |
| `bfloat16_3x` | 42/42 | 42/42 | — |
| `float16_1x` | 84/84 | 4/84 | — |
| `bfloat16_1x` | 84/84 | 20/84 | — |
| `float8_e4m3fn_1x` | 84/84 | 4/84 | — |
| `float8_e5m2_1x` | 84/84 | 4/84 | — |
| `tensorfloat32_1x` | 84/84 | not completed | 0/4 |

H100: all 1,008 original arithmetic identities pass. The before/after match identifies six BF16-three-pass failures becoming passes, no previous numerical pass becoming a failure, and unchanged tolerances. The 60-case targeted set overlaps the 42-case original BF16 set; these are not independent counts.

Eleven full TPU format campaigns completed: 186/840 numerical cases passed. Both TF32 variants also have separately labeled 0/4 samples. FP32 with FP32 accumulation passed 42/42; the additional FP64-accumulation path contributes only 8/42 passes. TPU BF16 three-pass remains a numerical pass; TPU FP64 remains software arithmetic. Missing, interrupted or sampled points do not complete a full qualification.

## Why TPU failed

### FP64 bitcasts do not expose the assumed IEEE-64 words

The direct diagnostic round-trips `1.0` correctly as a value on both devices. Bitcasting that FP64 value into uint32 words yields `[0x0, 0x3ff00000]` on CPU and `[0x0, 0x3f800000]` on TPU. The latter high word is the FP32 encoding of `1.0`. Thus the helper reads a different representation as if it were IEEE-64. The test observed exponent `-6` for `1.0`, where `frexp` requires `1`; scaled/rounded values then become wrong. This is stronger evidence than the prior compile error: the replacement compiles and returns incorrect numbers.

The proper next fix must avoid this FP64 bitcast assumption on TPU and validate real values, scales, boundaries and halfway cases on that backend. Converting the original FP64 input through FP32 would discard rounding information and would not preserve the current contract. True native FP64 is unavailable on TPU; a hybrid or different precision policy is a separate research choice. [JAX numerical accuracy](https://docs.jax.dev/en/latest/accuracy.html).

### FP32 carrier conversions do not force FP32 multiplication

The separately captured failing program still contains FP16 operands feeding an optimized TPU convolution with `operand_precision={highest,highest}`. The compiler removed the intended widening. A Python-level FP32 carrier is therefore insufficient evidence that the product was computed at that precision.

A proper follow-up should keep emulated quantization outputs in FP32 without a low-dtype round trip and qualify an explicit full-precision arithmetic route. It must compare identical quantized buffers before the dot, the raw product, rescaling and term summation, including the fused engine path. These are proposed follow-ups, not shipped or qualified fixes.

The isolated diagnostic deliberately uses FP32 inputs to separate this issue from the broken FP64 grid. Its unchanged tolerance is `3.814697265625e-6`. It reached its 240-second cap, so the last format is incomplete. No native FP16/FP8 TPU result is claimed.

| Isolated format | Passed / completed |
| --- | --- |
| `float16` | 47/84 |
| `float8_e4m3fn` | 84/84 |
| `float8_e5m2` | 84/84 |
| `float16_1x` | 51/84 |
| `float8_e4m3fn_1x` | 84/84 |
| `float8_e5m2_1x` | 48/48 (36 requested cases not run) |

### Initial TPU service setup was also corrected

The first service attempt failed before arithmetic with `Couldn't mmap: Resource temporarily unavailable`. Interactive SSH had unlimited locked memory, while the service inherited a finite limit. Restarting only the temporary test service with `LimitMEMLOCK=infinity` allowed TPU initialization. The initial failure logs are retained separately; they are not counted as arithmetic results. This is a temporary VM service setting, not an IAM, quota, network or billing change. [Google TPU initialization troubleshooting](https://docs.cloud.google.com/kubernetes-engine/docs/troubleshooting/tpus#tpu-initialization-failed).

## Engine and regression coverage

H100 primary workloads cover simulation, market construction, pricing, option exposure and market risk with seeds 7/8. These CPU/device comparisons generate scenarios independently; a differential flag alone does not locate a faulty arithmetic operation. Financial error relative to FP64 is a separate research question.

| H100 format | Primary engine passed / completed | Additional FP32-storage diagnostics |
| --- | --- | --- |
| `bfloat16` | 3/5 | 0/2 |
| `bfloat16_3x` | 5/5 | — |
| `float16` | 3/5 | 0/2 |
| `float32` | 5/5 | — |
| `float64` | 5/5 | — |
| `float8_e4m3fn` | 5/5 | — |
| `float8_e5m2` | 5/5 | — |

FP16/BF16 pipeline differences remain unresolved. H100 FP64, FP32, BF16-three-pass and the two FP8-base numerical pipeline controls pass; the latter do not override failed FP8 native qualification.

TPU primary portfolio coverage remains incomplete. The full runner prioritized arithmetic within its shortened 20-minute workload budget after setup diagnosis; any subsequent small single-swap attempts are separate, with 75-second limits and retained partial outputs.

| TPU engine attempt | Passed / recorded cases | Complete |
| --- | --- | --- |
| `post-budget-engine-bfloat16_3x` | 3/3 | True |
| `post-budget-engine-float16` | 0/6 (includes three FP32-storage diagnostics) | True |

H100 `bf16-pipeline-regressions.xml`: 3 passed, 0 failures, 0 errors, 0 skipped.

H100 `environment.xml`: 28 passed, 0 failures, 0 errors, 0 skipped.

H100 `grid-regressions.xml`: 31 passed, 0 failures, 0 errors, 0 skipped.

TPU `grid-regressions.xml`: 19 passed, 7 failures, 0 errors, 5 skipped.

The unchanged raw H100 BF16 control still fails 3/21 cases because it bypasses the engine workaround. The unchanged raw TPU FP16 control still fails 7/21 cases. Their failures are retained and do not invalidate the distinction between an engine workaround and an unfixed direct compiler operation.

## Evidence, spending and cleanup

Local evidence: `artifacts/accelerator-rerun-2026-10-26/`. Reviewable extracted evidence is under `h100/evidence/` and `tpu/evidence/`. Per-device summaries, `before-after.json`, JUnit XML, source manifests, optimized programs, partial results and all cloud command receipts are saved. Full compiler/trace data remain in the verified archives; redundant LLVM/pre-optimization/buffer reports were omitted from transfer. Initial TPU setup failures are included in the TPU archive.

- `h100-evidence.tar.gz`: SHA256 `50ca64f4f0af43531470c7f40263e4ce5ca8b7ee2bbd4b1f9c11b8ef46c8bd0a`; local/remote hashes match. VM `alex-h100-rerun-20261026`, immutable ID `4591135225713657889`, had automatic DELETE after 90 minutes and auto-deleting disks. Successful API receipts verify VM, matching disk and reserved-address absence.
- `tpu-evidence.tar.gz`: SHA256 `90e29e7198acb57c9b756b4ccf7a9cd6a63e550e120915306a9daecca2df0396`; local/remote hashes match. VM `alex-tpu-rerun-20261026`, immutable ID `5787881674665786607`, had automatic DELETE after 45 minutes and auto-deleting disks. Successful API receipts verify VM, matching disk and reserved-address absence.

The exact temporary OS Login key registration was removed and absence verified. Estimated combined machine charges are **$7.08**, using elapsed creation through verified deletion at the recorded rates. This excludes ancillary charges/taxes and is not a verified invoice. The rerun used a $20 allowance; configured maximum machine estimate was $11.89, leaving a reserve for extras. [Google H100 Spot rates](https://cloud.google.com/spot-vms/pricing), [Google TPU rates](https://cloud.google.com/tpu/pricing).

No four-hour extension, reservations, shared IAM/network/API/quota changes, or billing-configuration changes were made. Other project workloads were not modified. No passing native FP8 or all-native TPU evidence was promoted.
