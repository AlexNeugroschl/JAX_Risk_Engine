# TPU precision survey: all tested non-FP64 arithmetic passes

October 27, 2026, New York time. Correctness and repeatability testing, not a
benchmark. This is newer TPU evidence than the October 26
[hardware rerun](hardware-correctness-rerun.md).

**Subsequent native investigation:** [native TPU probes](tpu-native-precision.md)
find passing FP8-typed matrix candidates and failing/unavailable FP16 routes.
Those direct JAX/Pallas probes bypass engine dispatch. This page preserves the
earlier engine-selected paths, including FP32-based emulation.

**All 1,200 product cases, 32 staged product diagnostics, and 24 storage cases
passed on one TPU v6e.** FP64 compute and accumulation were explicitly refused.
The revised FP16 emulation passed this coverage. FP16, FP8 and TF32 products
remain FP32-based emulation on this engine's TPU path. These results do not
establish native FP16/FP8/TF32 instructions, full portfolio correctness, financial
accuracy against FP64, or H100 correctness.

## Results by compute format

Every product uses identical FP32 inputs and FP32 accumulation. The host NumPy
reference uses FP64 sums on independently quantized operands; no FP64 calculation
is sent to the TPU. Basic and historical compensated strategies are explicit.
FP32 and X3 controls are run once rather than repeating equivalent TPU behavior.

| Compute format | Basic passed / completed | Compensated passed / completed | Engine TPU route |
| --- | --- | --- | --- |
| `float32` | 24/24 | Same control | Native dispatch, highest precision requested |
| `float16` | 64/64 | 64/64 | Emulated with FP32 carriers/products |
| `bfloat16` | 48/48 | 48/48 | Native BF16 dispatch |
| `float8_e4m3fn` | 64/64 | 64/64 | Emulated with FP32 carriers/products |
| `float8_e5m2` | 64/64 | 64/64 | Emulated with FP32 carriers/products |
| `tensorfloat32` | 48/48 | 48/48 | Emulated TF32 operand grid, FP32 products |
| `bfloat16_3x` | 24/24 | Same control | Native BF16 three-pass algorithm request |
| `float16_1x` | 64/64 | 64/64 | Emulated with FP32 carriers/products |
| `bfloat16_1x` | 48/48 | 48/48 | Native BF16 dispatch |
| `float8_e4m3fn_1x` | 64/64 | 64/64 | Emulated with FP32 carriers/products |
| `float8_e5m2_1x` | 64/64 | 64/64 | Emulated with FP32 carriers/products |
| `tensorfloat32_1x` | 48/48 | 48/48 | Emulated TF32 operand grid, FP32 products |
| `float64` | Refused as intended | Refused as intended | Unsupported; no silent downgrade |

FP64 accumulation with FP32 compute was also refused. All twelve strict native
request checks agreed with the engine's declared capability: FP32, BF16 and
BF16 three-pass accepted; FP16, FP8 and TF32 requests refused unless the caller
explicitly permits research emulation. Acceptance is a dispatch-policy check,
not a new physical instruction trace.

## What the pass establishes

The [runner](../../tests/support/tpu_precision_survey.py) covers shapes
`(32,32,8)`, `(32,32,1)`, `(33,13,7)` and batched `(2,9,13,7)`; both shared-operand
sides; nearest and stochastic rounding where applicable; ordinary, zero and
cancelling inputs; and additional FP16/FP8 small-grid values including halfway
points and subnormals. Ordinary inputs stay in the basic formats' representable
range. Every product was executed twice and repeated exactly.

The comparison retains the existing arithmetic allowances: normalized error at
most `3.814697265625e-6` for ordinary products and `3.0517578125e-5` for the X3
algorithm. Normalization uses the sum of absolute reference products, so
cancellation cannot hide a large arithmetic discrepancy. Exact-zero cases must
produce exact zero. The largest observed normalized error outside X3 was about
`8.77e-8`. X3's largest was `1.25e-5`, within its separate algorithm allowance;
this is not a promise of full FP32 accuracy for X3.

The 32 staged diagnostics cover all eight low array-format names, both profiles
and both rounding modes on an odd matrix shape. They check quantized values and
scales against an independent NumPy implementation, raw products on those exact
operands, rescaling/reconstruction, and a separately compiled fused engine product.
Compensated probes additionally stress dynamic range. All checks passed without
changing the production implementation or tolerances during the trial.

All 256 captured optimized emulation programs contain no typed FP16 or FP8
carriers. This removes the specific low-dtype round trip seen in the failing
October 26 FP16 rerun. Optimized HLO is retained; this inspection is not an
instruction-level hardware qualification. The TF32 result is an emulation pass,
not a pass for the previously unsupported native TF32 algorithm.

Storage checks cover FP16, BF16, E4M3 and E5M2, with six cases each: nearest and
stochastic rounding in raw storage, scaled storage without offsets, and scaled
storage with offsets. Arrays contain 65 scenarios to cross block boundaries and
exercise a short final block, concentrated values, signed zeros and small values.
Stored values, block scales/offsets, separate decoding and fused round trips
match an independent NumPy oracle exactly. This is FP32-input/FP32-load coverage,
not FP64 input or load coverage.

## Remaining limits

Full TPU portfolio jobs remain refused because calibration, level reconstruction
and other fixed internals still require FP64. A genuine FP32 internal profile
must be implemented and tested before those jobs can run. Increasing the trial
duration does not remove that requirement.

This bounded survey does not exhaust all shapes, values, subnormal behavior,
strategy combinations, seeds or compiler versions. In particular, BF16/TF32
subnormals near FP32 underflow need a separate contract and were not exercised
by the FP16/FP8 small-grid cases. GPU-only normalization and explicit native FP8
algorithm options were not tested on TPU. Product passes measure arithmetic
relative to the chosen quantized operands; low-precision financial estimates can
still need scaling, residual terms or other explicit numerical experiments.
No acceptance-evidence table was promoted and no performance conclusion is drawn.

## Reproduction and evidence

Device: `TPU v6 lite`, one `ct6e-standard-1t` VM in `asia-northeast1-b`.
JAX/jaxlib `0.10.2`, libtpu `0.0.42`, NumPy `2.4.6`, pinned repository constraints.
Base commit: `5f344cc` with the new survey runner included in the frozen snapshot.
Source archive SHA256:
`0d0fb35be58f6b8a323810a13ca6d430e30e7a91f2f192c5836289b83198cd1b`.
Individual source hashes and the separately hashed storage probe are saved.

Main command, after preparing the pinned environment on the existing temporary VM:

```bash
JAX_PLATFORMS=tpu,cpu timeout --signal=TERM --kill-after=30s 1800s \
  .venv/bin/python -u -m tests.support.tpu_precision_survey \
  --platform tpu --out evidence/survey --budget-seconds 1500
```

The main survey completed in about 205 seconds. Its service had
`LimitMEMLOCK=infinity`, avoiding the previous initialization failure, and an
independent 35-minute service limit. The small storage supplement had a two-minute
service limit. These are correctness-run durations, not performance measurements.

Evidence is under `artifacts/tpu-survey-2026-10-27/`: `summary.json`,
`extracted/evidence/survey/summary.json`, `extracted/evidence/storage/summary.json`,
per-case NPZ buffers, optimized HLO, packages, source manifest, setup logs,
cloud command receipts, `integrity.json`, `lowering-review.json`, and cleanup
receipts. The downloaded archive hash matches the VM's hash:
`637dc46378489ce7e995d87182605c89a8dfd03fdd62128b9b72cf63e94b294f`.

## Resource use and cleanup

Only one single-chip TPU VM was created. It had a cloud-enforced 45-minute DELETE
limit, no automatic restart and one auto-deleting 100 GB boot disk. The VM was
manually deleted immediately after collecting and verifying the evidence.
Successful API receipts verify absence of matching VMs, disks and reserved
addresses. The unique temporary OS Login key was removed and its absence verified;
its local private/public/PPK files were also deleted. Other registrations and
project workloads were preserved. No shared IAM, network, API, quota, reservation
or billing configuration was changed.

Creation through verified absence took about 10.5 minutes. At the listed Tokyo
v6e rate of $3.24 per chip-hour, the conservative machine estimate is **$0.57**,
excluding ancillary charges and taxes; this is not a verified invoice. The
45-minute maximum machine estimate was $2.43, within the prior $20 allowance.
[Google TPU pricing](https://cloud.google.com/tpu/pricing).
