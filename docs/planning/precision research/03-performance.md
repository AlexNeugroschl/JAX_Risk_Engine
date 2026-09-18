# 03 — Performance and the research question

**No timing was measured** ([F-6](00-findings.md#f-6-what-remains-unmeasured)). Every
throughput number here is a projection labeled as such. The *accuracy* numbers are measured,
and they turn out to constrain the performance story more tightly than hardware does.

---

## P-1. The precision ceiling — the plan's central result

The roadmap states this project's core research question:

> whether running *many more* lower-precision simulations reaches the same risk answer, in
> the same wall-clock time, as running *fewer* high-precision ones

[F-5](00-findings.md#f-5-quantization-noise-vs-monte-carlo-error--the-precision-ceiling)
turns that into arithmetic. Monte Carlo error falls as `1/√N`. Quantization noise does not
fall at all — it is a fixed floor set by the storage format. So each tier has a path count
beyond which **more paths buy nothing**:

| Tier | shock rms error | useful path ceiling |
|---|---|---|
| float16 | 2.1e-04 | ~22,800,000 |
| bfloat16 | 1.7e-03 | ~359,000 |
| float8_e4m3fn | 2.6e-02 | ~1,400 |
| float8_e5m2 | 5.3e-02 | ~360 |

**The trade only pays inside the ceiling.** Low precision buys throughput, throughput buys
paths, paths buy accuracy as `1/√N` — but only until quantization takes over, at which point
additional paths are pure cost. The strategy is not "use the lowest precision that runs,"
it is **"use the lowest precision whose ceiling is above your required path count."**

This reframes the research question from open-ended to a selection rule, and it is the single
most useful thing in this plan for anyone deciding what to run.

### What it implies per tier

- **float16 — the workhorse.** A ceiling above 22M paths exceeds any realistic portfolio
  run. Effectively unconstrained for this engine's workloads, at half the memory traffic of
  float32. This is the tier most likely to pay off, and should be built first.
- **bfloat16 — viable, narrower.** ~359k paths is enough for many runs but not comfortably
  above a 1M-path run. Retained mainly because TPU may favor it on hardware grounds.
- **FP8 — not a simulation storage format.** A ceiling of ~1,400 paths is below any
  defensible risk run. **This is a genuinely useful negative result**: it means FP8 should
  not be pursued as "tier 16, but lower." If FP8 has a role it is somewhere other than path
  storage — see [P-4](#p-4-where-fp8fp4-might-actually-belong).
- **FP4 — research only.** No working prototype
  ([F-6](00-findings.md#f-6-what-remains-unmeasured)).

---

## P-2. Projected throughput — what the tiers could buy

Unmeasured. Stated as expectations with their mechanisms, so they can be checked later.

**Memory traffic is the reliable win.** The engine's simulation step is bandwidth-bound: it
generates and stores a large `(scenarios × steps × factors)` array and streams it into
pricing. Storage dtype sets that volume exactly:

| Tier | bytes/element | vs float64 | vs float32 |
|---|---|---|---|
| 64 | 8 | 1.0× | 0.5× |
| 32 | 4 | 2.0× | 1.0× |
| 16 | 2 | 4.0× | 2.0× |
| 8 | 1 | 8.0× | 4.0× |

For a bandwidth-bound stage this transfers close to proportionally, and it also raises the
path count that fits in memory at all — which for large portfolios can be the binding
constraint rather than speed.

**Arithmetic throughput is the unreliable win.** Compute stays at float32
([A-1](01-architecture.md#a-1-the-central-change-one-knob-becomes-two)), so no
low-precision *arithmetic* speedup is claimed. On TPU, low-precision matmul units could make
K-1's Newton–Schulz faster, but K-1 runs once per simulation, not per path — so it is not
where time goes. **The tiers' value is memory, not FLOPs.**

**On this CPU dev machine, expect the tiers to be slower, not faster.** Sub-float32 formats
are typically emulated on CPU: stored narrow, unpacked to float32 to compute, repacked.
That is pure overhead when bandwidth is not the bottleneck. **A CPU benchmark showing tier 16
slower than tier 32 would confirm the design, not refute it** — it should be recorded as
expected, so that a disappointing first measurement is not misread as failure. The
throughput case belongs on TPU.

**Costs that partly offset the win:**

- **Compilation.** Each tier is a distinct XLA program. More tiers in flight means more
  compilation, and separate worker processes do not share compiled kernels — a cost the
  worker-pool docstring already notes for the existing two tiers.
- **Conversion.** Compute-high/store-low inserts casts. Cheap, not free.
- **`_dtype_of` dispatch.** Negligible.

---

## P-3. The multi-device story is unchanged, and that is good news

[A-4](01-architecture.md#a-4-worker-pool-tiers-not-a-boolean) establishes that tiers 16/8/4
all run with `jax_enable_x64` disabled, exactly like tier 32 — so they need no new worker
pool, and the existing process-isolation design carries them unmodified. The concurrency
mechanism, which is the trickiest part of the current system, does not have to be reopened.

The interesting deployment shape stays what it already is: fp64-tier and fp32-compute-tier
pools running genuinely concurrently on separate devices. Tiers 16/8 add *density* within
the fp32-compute pool — more paths per device-hour — rather than a new axis of parallelism.

---

## P-4. Where FP8/FP4 might actually belong

[P-1](#p-1-the-precision-ceiling--the-plans-central-result) rules FP8 out for path storage.
That is a sharper statement than "FP8 is future work," and it redirects rather than closes
the question. Plausible remaining roles, none validated:

- **Option A's original scope.** The architecture doc's `MatmulPrecisionConfig` targets FP8/FP4
  at two matmul sub-steps *with float32 accumulation* — which is a compute-side use where the
  accumulator, not the format, carries the precision. The ceiling analysis above does not
  apply to it, because nothing is stored at FP8. Option A remains independently viable and
  is not superseded by this plan.
- **Non-path data.** Intermediate quantities with far looser tolerances than the shocks.
- **Screening runs.** A ~1,400-path ceiling is useless for a reported VaR but may be fine for
  a fast pre-screen that flags which portfolios deserve a precise run. This is a real
  workflow pattern and the most promising FP8 use identified here.

**Recommendation: do not build FP8/FP4 as storage tiers.** Add them to the `Tier` enum for
completeness and rejection, and treat Option A as the live FP8 track.

---

## TPU: the hypothesis that changes the answer

Unchanged from [architecture.md §Option B](../../concepts/architecture.md#option-b-why-uniform-sub-float32-precision-is-not-achievable-today),
and restated because it is the biggest open variable:

> It is *plausible* — genuinely unverified — that a Cloud TPU's native XLA backend has
> broader low-precision kernel coverage, since bfloat16 is TPU's own native compute format.

If true, `cholesky`/`norm.ppf` may work natively at bfloat16 on TPU, making **K-1 and K-2
unnecessary at that one tier** (not at float16 or below, and not on CPU). This would not
waste the kernel work — the tiers below bfloat16 still need it, and having a CPU-runnable
path is what makes the whole thing testable on this machine — but it would change which tier
to reach for first on real hardware.

**First TPU task should be re-running the [F-1](00-findings.md#f-1-kernel-coverage-by-dtype--confirmed-not-assumed)
sweep**, before any tier work is ported. It is a few minutes of work and determines whether
bfloat16 or float16 leads on that backend.

---

## P-5. The honest summary

- **Measured:** accuracy per tier, the precision ceilings, which primitives work where.
- **Projected:** memory-traffic wins (high confidence — arithmetic), tier-16 practicality
  (good confidence), TPU throughput (low confidence — unverified backend).
- **Unmeasured:** all wall-clock timing, all downstream (pricing/VaR/Greeks) error
  propagation, FP4 entirely.

The plan's value does not depend on the projections. Even if every throughput hope fails,
[P-1](#p-1-the-precision-ceiling--the-plans-central-result) stands: the precision ceiling is
a measured property of the numerics, and it answers the project's core research question in
a form that holds on any hardware.
