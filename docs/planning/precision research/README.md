# Precision Research

Planning for extending this engine's precision control from today's `{32, 64}` down to
**64 / 32 / 16 / 8 / 4 bits** across pricing and all downstream calculations.

**Date:** 2026-09-17 · **Owner:** Alex · **Status:** Plan only. No engine code changed.

| Document | What it covers |
|---|---|
| [00-findings.md](00-findings.md) | Measured results from live probes on this checkout — what actually works, what returns NaN, and the numbers the plan rests on |
| [01-architecture.md](01-architecture.md) | How `PrecisionConfig` and the worker pool change to carry 5 tiers; the compute/storage dtype split |
| [02-kernels.md](02-kernels.md) | The two blocking primitives (`cholesky`, `norm.ppf`) and the replacement kernels, with prototype results |
| [03-performance.md](03-performance.md) | Expected speed/memory effects, the precision ceiling on path count, and why the core research question now has a quantitative form |
| [04-roadmap.md](04-roadmap.md) | Phased delivery, validation bar, risks, and what stays out of scope |

---

## The short version

**The existing blocker analysis is correct, and I re-verified it rather than trusting it.**
`docs/concepts/architecture.md` §"Option B" says `jnp.linalg.cholesky` and
`jax.scipy.stats.norm.ppf` fail on every dtype below float32. Re-run live against this
checkout's `jax==0.10.2`: confirmed exactly, for bfloat16, float16, float8_e4m3fn,
float8_e5m2 and float4_e2m1fn alike.

**But `matmul` works at every one of those dtypes, including FP4.** That is the opening.
Both blocked primitives can be rebuilt from operations that do have low-precision coverage,
and I prototyped both:

- A matmul-only matrix square root (Newton–Schulz) **runs at bfloat16 and float16**, where
  `cholesky` hard-fails, reconstructing the covariance to 2.5e-2 / 3.4e-3 max error.
- An elementwise inverse-normal-CDF (Acklam) replaces `norm.ppf` with no special-function
  kernel — to 3.0e-4 max error at float32.

**The decisive finding is not that low precision is reachable — it is that uniform low
precision is the wrong target.** Evaluating the Acklam polynomial *in* bfloat16/float16/FP8
returns **NaN**: its coefficients (~±276) overflow the format's dynamic range in the
intermediate terms. Computing in float32 and *storing* the result low works at every tier
down to FP8. So the architecture this project should build is a **compute-dtype / storage-dtype
split**, not a single dtype knob pushed lower — which is also exactly what the existing code
already does at `engine/simulation/market_model.py:67` (`norm.ppf(...).astype(dtype)`).

**And the project's core research question now has a measurable answer.** The roadmap frames
phase 12 as "whether running *many more* lower-precision simulations reaches the same risk
answer as fewer high-precision ones." Each dtype turns out to impose a **ceiling on useful
path count** — the point where its quantization noise overtakes Monte Carlo error and extra
paths stop buying accuracy:

| Storage dtype | Shock quantization rms | Useful path ceiling |
|---|---|---|
| float16 | 2.1e-4 | ~22,800,000 |
| bfloat16 | 1.7e-3 | ~359,000 |
| float8_e4m3fn | 2.6e-2 | ~1,400 |
| float8_e5m2 | 5.3e-2 | ~360 |

That table is the plan's central result. **float16 is the surprise winner** over bfloat16 —
10 bits of mantissa beats 8 for this workload, because MC shocks need precision, not the wide
exponent range ML training needs. FP8 and FP4 cap out far below any useful portfolio path
count, which reframes them from "a lower tier of the same thing" into a genuinely different
use (see [03-performance.md](03-performance.md)).

**Honest scope.** Tiers 16 and 8 are buildable against the evidence here. Tier 4 is not
yet — `float4_e2m1fn` has ~3 representable values in the shock range, and no prototype
survives it as a storage format for path data. It is kept in the plan as a research
target with an explicit falsification test, not a delivery commitment.

All measurements were produced on this CPU-only dev machine. TPU behavior remains the
labeled, unverified hypothesis it already is in the architecture doc — see
[03-performance.md](03-performance.md#tpu-the-hypothesis-that-changes-the-answer).
