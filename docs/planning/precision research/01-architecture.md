# 01 — Architecture

How the engine changes to carry five precision tiers. Grounded in the measured results in
[00-findings.md](00-findings.md); every current-state claim names the file and line it
comes from.

---

## A-1. The central change: one knob becomes two

Today a precision knob is a single integer that means both "what dtype do we compute in"
and "what dtype do we store" — they are the same thing, because at `{32, 64}` they can be.

[F-3](00-findings.md#f-3-the-nan-result--uniform-low-precision-is-the-wrong-target) kills
that equivalence. Below float32, computing in the storage dtype returns NaN, while computing
in float32 and storing low works at every tier. So the knob must split:

```python
@dataclass(frozen=True)
class Tier:
    """A precision tier: what we compute in, and what we keep."""
    storage: int   # 64 | 32 | 16 | 8 | 4  -- what arrays are held as
    compute: int   # 64 | 32              -- what arithmetic runs in
```

with the rule that **`compute >= max(storage, 32)`**. The tiers become:

| Tier | storage | compute | Status |
|---|---|---|---|
| 64 | float64 | float64 | Built today |
| 32 | float32 | float32 | Built today |
| 16 | float16 *(default)* / bfloat16 | float32 | Buildable — [F-2](00-findings.md#f-2-a-matmul-only-matrix-square-root-runs-where-cholesky-cannot), [F-4](00-findings.md#f-4-compute-high--store-low-works-at-every-tier) |
| 8 | float8_e4m3fn | float32 | Buildable, narrow use — [F-5](00-findings.md#f-5-quantization-noise-vs-monte-carlo-error--the-precision-ceiling) |
| 4 | float4_e2m1fn | float32 | Research only — no working prototype |

**Why float16 is the default 16-bit format, not bfloat16.** [F-2](00-findings.md#f-2-a-matmul-only-matrix-square-root-runs-where-cholesky-cannot)
and [F-4](00-findings.md#f-4-compute-high--store-low-works-at-every-tier) both put float16
ahead by roughly 7–8×, because this workload needs mantissa bits and not bfloat16's wide
exponent range — the shocks are O(1) standard normals, not the wide-dynamic-range gradients
bfloat16 was designed for. This is worth stating loudly because it inverts the ML-world
default, and [architecture.md §Option B](../../concepts/architecture.md#option-b-why-uniform-sub-float32-precision-is-not-achievable-today)
already records one earlier planning pass that reasoned wrongly from bfloat16's ML
reputation. bfloat16 stays selectable — on TPU it may win on hardware grounds regardless.

**Backward compatibility.** `Tier(storage=64, compute=64)` and `Tier(32, 32)` reproduce
today's two settings exactly, so plain `64`/`32` ints stay valid input and keep their current
meaning. Nothing existing has to change at a call site.

---

## A-2. `PrecisionConfig` keeps its shape

The current design in [`engine/portfolio/request.py:184`](../../../engine/portfolio/request.py#L184)
is four knobs — `simulation`, `pricing`, `risk`, `calibration` — with `pricing` and `risk`
additionally accepting structured per-instrument / per-Greek overrides resolved through the
single helpers `_resolve_pricing_dtype` and `_resolve_risk_dtype`.

**That hierarchy needs no structural change.** The extension is entirely in the *value* each
knob accepts (`int` → `int | Tier`) and in the validation sets. This matters: it means the
whole per-instrument and per-Greek drill-down machinery, the HTTP schema shape, and the
worker-pool routing keep working, and the change stays a widening rather than a redesign.

The concrete edits:

1. **`_dtype_of` — the migration hazard.** Today
   ([`request.py:227`](../../../engine/portfolio/request.py#L227)) it reads
   `return jnp.float64 if precision_bits == 64 else jnp.float32`. That `else` silently maps
   *any* unrecognized value to float32. Adding tiers without changing it means
   `simulation=16` quietly runs at float32 and returns plausible, wrong-precision results
   with no error. **This one-line fallthrough is the single most dangerous spot in the
   migration** and must become an explicit table with a raise on unknown input.
2. **Validation sets.** The `(32, 64)` membership tests in `PrecisionConfig.__post_init__`,
   `PricingPrecisionOverride.__post_init__` and `RiskPrecisionOverride.__post_init__` widen
   to the five tiers, and gain the `compute >= max(storage, 32)` check.
3. **Docstrings.** `PrecisionConfig`'s docstring currently states the sub-float32 blocker as
   settled fact. It stays accurate for *uniform* low precision and should be amended rather
   than deleted — the blocker is real; what changed is that a compute/storage split routes
   around it.

---

## A-3. Where the replacement kernels plug in

Two call sites, both in the simulation layer, both already isolated behind a dtype argument:

- **Correlation.** `jnp.linalg.cholesky` on the joint covariance
  ([`market_model.py:347`](../../../engine/simulation/market_model.py#L347), and the
  validation path at `:401`) → matmul-only Newton–Schulz square root
  ([K-1](02-kernels.md#k-1)).
- **Shocks.** `norm.ppf` at
  [`market_model.py:67`](../../../engine/simulation/market_model.py#L67) → elementwise
  Acklam approximation ([K-2](02-kernels.md#k-2)).

**The second one is nearly free, architecturally.** That line already reads
`norm.ppf(uniform_clipped).astype(dtype)` — compute high, store low, with a docstring
explaining exactly why the explicit cast is required. The new tiers change *what function*
computes the high-precision value, not the surrounding pattern. The engine has been doing
compute/storage splitting at this exact line since the float32 tier existed.

**Dispatch.** Both sites select kernel by tier: `storage >= 32` keeps the LAPACK/special-function
path (no behavior change, no risk to existing tiers), `storage < 32` takes the replacement.
Existing 64/32 runs never touch the new code.

---

## A-4. Worker pool: tiers, not a boolean

[`engine/portfolio/worker_pool.py`](../../../engine/portfolio/worker_pool.py) keys one
`ProcessPoolExecutor` per precision tier, because `jax_enable_x64` is a process-global flag
and two threads wanting different precisions cannot both be correct. `_POOLS` is
`{32, 64} -> pool`, routed by `precision.simulation`.

**This generalizes almost for free, and the reason is worth stating.** `jax_enable_x64` is
binary — it gates whether float64 can be *created*. Tiers 16/8/4 all have `compute=32`, so
they all want x64 **disabled**, exactly like tier 32. The pool key stays a boolean in
substance:

| Tier | x64 | Pool |
|---|---|---|
| 64 | enabled | fp64 pool |
| 32, 16, 8, 4 | disabled | fp32-compute pool |

So no new pool is strictly required — tiers 16/8/4 are safe to run in the existing fp32 pool,
because dtype below float32 is expressed in array dtypes, not in a process-global flag.
Whether to add *separate* pools per tier is then a throughput question (different tiers have
different XLA compilation profiles and, on TPU, may want different chips), not a correctness
one. **Recommendation: do not add pools in phase 1.** Route 16/8/4 into the fp32-compute
pool, and revisit only if TPU profiling shows tier-mixing hurts. This keeps a genuinely
tricky concurrency mechanism untouched.

---

## A-5. HTTP and integration boundary

[`engine/api/schemas.py`](../../../engine/api/schemas.py) validates precision values at the
request boundary and must widen the same way. Two boundary-specific points:

- **Reject, do not coerce.** Given the `_dtype_of` fallthrough above, an unknown precision
  value must be a 4xx at the schema boundary, never silently downgraded. A wrong-precision
  risk number that looks plausible is worse than a rejected request.
- **Echo the tier in the result.** A result computed at tier 8 must say so in its payload.
  The engine already carries strong identity/coverage discipline at the EOD boundary
  (see the [TraderX integration docs](../traderX_integration/eod-contract-response-v7.md));
  precision belongs in that same contract, because a consumer cannot otherwise tell a
  float16 VaR from a float64 one, and the difference can exceed the tolerance they check
  against.

---

## A-6. What does not change

Worth stating, because the blast radius is the main risk in a change like this:

- **Pricers.** Instruments consume whatever dtype they are handed; none hardcode float64
  except at the deliberate points already documented in `price_portfolio`.
- **The x64 re-enable after `generate_paths`.** The subtle correctness fix at
  [`request.py:742`](../../../engine/portfolio/request.py#L742) — re-enabling x64 so that a
  `simulation=32, pricing=64` combination does not silently truncate — keeps working
  unchanged, and its reasoning extends to the new tiers verbatim.
- **`_PRICING_LOCK`.** Untouched, still defense-in-depth.
- **ORE parity tests.** These run at tier 64 and must stay byte-identical. They are the
  regression canary for the entire change.
