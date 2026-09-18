# 04 — Roadmap, validation, and risks

Phased delivery. Each phase is independently valuable and independently abandonable — the
sequencing is deliberately arranged so that the riskiest work (new numerical algorithms)
is validated at *known-good* precision before any new dtype is introduced.

---

## Sequencing principle

**Never debug a new algorithm and a new dtype at the same time.** K-1 and K-2 replace
library primitives the engine has trusted since its first ORE cross-check. If they land
simultaneously with float16 storage, any discrepancy has two candidate causes and the
investigation is twice as hard. So: kernels first at 64/32 (where they can be checked against
LAPACK and scipy directly), dtypes second.

---

## Phase 1 — Make the tier plumbing safe *(small, no new dtypes)*

Introduce the compute/storage split with the existing two tiers only. No behavior change.

1. **Fix `_dtype_of`** ([`request.py:227`](../../../engine/portfolio/request.py#L227)) —
   replace the `if 64 else float32` fallthrough with an explicit table that **raises** on
   unknown input. See [A-2](01-architecture.md#a-2-precisionconfig-keeps-its-shape); this is
   the highest-risk line in the whole migration, because it converts an invalid tier into a
   silently wrong answer rather than an error.
2. **Introduce `Tier(storage, compute)`** with `compute >= max(storage, 32)` validation;
   plain ints keep meaning exactly what they mean today.
3. **Widen schema validation** to reject-not-coerce at the HTTP boundary
   ([A-5](01-architecture.md#a-5-http-and-integration-boundary)).

**Exit criterion:** the full existing test suite passes unchanged, and `tests/test_ore_parity.py`
is byte-identical. Phase 1 is pure safety scaffolding — if it changes any number, it is wrong.

---

## Phase 2 — Land the kernels at existing precision *(the real work)*

Implement K-1 and K-2, dispatched only when explicitly requested, running at float32/float64.

1. **K-2 (Acklam) first** — it is elementwise, has no structural caveats, and is the
   most-executed step. Validate against `scipy.stats.norm.ppf` in bulk *and* in the tails
   separately ([K-2's tail note](02-kernels.md#k-2-elementwise-inverse-normal-cdf-replaces-jaxscipystatsnormppf)).
2. **Audit the `cholesky` call sites** before K-1: confirm neither
   [`market_model.py:347`](../../../engine/simulation/market_model.py#L347) nor `:401`
   depends on triangularity, and keep the `:401` validation path at float32+
   — Newton–Schulz returns garbage rather than failing on a non-SPD matrix
   ([K-1's caveat](02-kernels.md#the-triangularity-caveat--read-before-implementing)).
3. **K-1 (Newton–Schulz)**, with a tuned iteration count.

**Exit criterion:** with both kernels enabled at float64, the engine reproduces ORE parity
within the existing tolerances. This is a strong test — it exercises the new numerics against
the project's most authoritative reference, with dtype held constant.

---

## Phase 3 — Tier 16 *(the payoff tier)*

Enable `storage=16` end to end, float16 default, bfloat16 selectable
([A-1](01-architecture.md#a-1-the-central-change-one-knob-becomes-two)).

Route through the existing fp32-compute worker pool — **no new pool**
([A-4](01-architecture.md#a-4-worker-pool-tiers-not-a-boolean)).

**This is where the unmeasured half of the analysis gets measured**
([F-6](00-findings.md#f-6-what-remains-unmeasured)): error propagation from shocks through
pricing, VaR/ES and Greeks. Everything measured so far concerns the pipeline's *input*.

**Exit criterion:** the validation bar below, at tier 16, for every instrument type.

---

## Phase 4 — Tier 8, rejection, and FP4 research

[P-1](03-performance.md#p-1-the-precision-ceiling--the-plans-central-result) already
indicates FP8 is not a viable path-storage format (~1,400-path ceiling). Phase 4 is therefore
mostly about **recording that conclusion rigorously** rather than building a tier:

1. Confirm the FP8 ceiling with the downstream measurements from phase 3's methodology.
2. Add tiers 8 and 4 to the enum as **explicitly rejected for path storage**, with the
   measured reason — so a future reader does not re-litigate it from ML intuition, which is
   exactly the failure mode the architecture doc already records once.
3. **FP4 falsification test:** `float4_e2m1fn` has ~3 representable magnitudes in the shock
   range. The test is to demonstrate it cannot represent a standard normal usefully; if that
   demonstration fails, FP4 deserves a second look. Framed as falsification because the
   expected outcome is negative.
4. Revisit FP8 in its genuinely promising roles
   ([P-4](03-performance.md#p-4-where-fp8fp4-might-actually-belong)) — Option A's
   float32-accumulation matmuls, or fast screening runs.

---

## Phase 5 — TPU validation

Deferred to real hardware, as all TPU work in this project is.

**First task, before porting anything:** re-run the
[F-1](00-findings.md#f-1-kernel-coverage-by-dtype--confirmed-not-assumed) kernel sweep on
TPU. If native bfloat16 `cholesky`/`norm.ppf` exist there, K-1/K-2 become unnecessary at that
one tier and bfloat16 may lead over float16 on hardware grounds
([TPU hypothesis](03-performance.md#tpu-the-hypothesis-that-changes-the-answer)). Minutes of
work; determines the rest of the phase.

Then: the first real throughput measurements, device-count-aware pool sizing, and
`JAX_PLATFORMS`/`TPU_VISIBLE_CHIPS` pinning that
[`worker_pool.py`](../../../engine/portfolio/worker_pool.py) already leaves ready.

---

## Validation bar

Unchanged from the architecture doc's own standard, which is stricter than
agreement-with-reference:

> cross-checked against the float64 originals for numerical agreement, *and* a separate pass
> confirming every downstream computation (pricing, VaR/ES, Greeks) stays within an
> acceptable error band at the target precision.

Concretely, per tier:

| Check | Against | Notes |
|---|---|---|
| Shock distribution | float64 shocks | Bulk **and** tails separately — tails drive VaR/ES |
| Covariance reconstruction | `S·Sᵀ` vs input | K-1's own error, per [F-2](00-findings.md#f-2-a-matmul-only-matrix-square-root-runs-where-cholesky-cannot) |
| Per-instrument NPV | tier-64 NPV | Every type in `tests/test_swap.py`, `test_european_swaption.py`, `test_bermudan_swaption.py`, `test_american_swaption.py` |
| VaR / ES | tier-64 | `tests/test_var_es.py` — tail-sensitive |
| Greeks | tier-64 | `tests/test_greeks.py`, `test_greeks_bermudan.py` — **the known risk** |
| ORE parity | ORE | `tests/test_ore_parity.py` — must stay byte-identical at tier 64 |

**Greeks are the expected failure point.** They difference nearby values, which amplifies
relative error — a tier that is fine for NPV can be unusable for Delta/Gamma. The existing
`RiskPrecisionOverride` already allows per-Greek precision
([`request.py:160`](../../../engine/portfolio/request.py#L160)), so the natural remedy is
available: run Greeks a tier higher than pricing. **Expect to need it**, and treat
"Greeks require tier+1" as a likely finding rather than a defect.

**Tolerances must be set before measuring, not after.** Otherwise the bar becomes whatever
the implementation happens to achieve.

---

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| `_dtype_of` fallthrough silently downgrades a new tier to float32 | **High** — wrong numbers, no error | Phase 1, item 1; raise on unknown |
| Newton–Schulz silently succeeds on a non-SPD matrix where `cholesky` correctly failed | **High** — loses an existing validation | Keep the `:401` validation path at float32+ regardless of tier |
| Greeks unusable at tier 16 | Medium | Per-Greek override already exists; run Greeks a tier higher |
| Acklam tail error distorts VaR/ES | Medium | Test tails separately; optional Halley refinement ([K-2](02-kernels.md#k-2-elementwise-inverse-normal-cdf-replaces-jaxscipystatsnormppf)) |
| CPU benchmarks show tier 16 slower | Low — expected | Documented as expected ([P-2](03-performance.md#p-2-projected-throughput--what-the-tiers-could-buy)); throughput case is TPU-side |
| Tier proliferation raises XLA compile cost | Low | No new worker pools in phase 1–3 |
| A consumer cannot tell which tier produced a number | Medium | Echo tier in results ([A-5](01-architecture.md#a-5-http-and-integration-boundary)) |

---

## Out of scope

- **FP8/FP4 as path-storage tiers** — measured as not viable
  ([P-1](03-performance.md#p-1-the-precision-ceiling--the-plans-central-result)). Rejected
  with a reason, not deferred.
- **Option A / `MatmulPrecisionConfig`** — a separate, still-valid track
  ([P-4](03-performance.md#p-4-where-fp8fp4-might-actually-belong)); not superseded by this
  plan and not folded into it.
- **New worker pools per tier** — unnecessary for correctness
  ([A-4](01-architecture.md#a-4-worker-pool-tiers-not-a-boolean)); revisit only on TPU
  profiling evidence.
- **Mixed precision within a single pricing call** beyond the existing per-instrument and
  per-Greek overrides.
- **Any timing commitment** — no wall-clock measurement exists yet
  ([F-6](00-findings.md#f-6-what-remains-unmeasured)).

---

## What would make this plan wrong

Stated so the plan is falsifiable rather than merely persuasive:

- If downstream error propagation (phase 3) shows tier 16 fails the bar for NPV — not just
  Greeks — the payoff tier disappears and only the kernel work retains value.
- If TPU has native bfloat16 linalg coverage, K-1/K-2 are unnecessary at bfloat16 and the
  sequencing above is suboptimal (though not wrong) for that hardware.
- If the simulation stage turns out not to be bandwidth-bound in production shapes, the
  memory-traffic argument in [P-2](03-performance.md#p-2-projected-throughput--what-the-tiers-could-buy)
  — the main projected benefit — weakens considerably.
