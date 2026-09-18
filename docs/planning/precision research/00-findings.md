# 00 — Measured findings

Everything in this plan rests on probes run live against this checkout, not on inference
from the ML literature. This document records what was run, what came back, and which
claims are therefore *measured* versus *reasoned*. Every number below is reproducible
from the snippets included here.

**Environment:** `jax==0.10.2`, `jaxlib==0.10.2`, CPU backend, Windows, commit `e4ca50b`.

---

## F-1. Kernel coverage by dtype — confirmed, not assumed

The claim in [architecture.md §Option B](../../concepts/architecture.md#option-b-why-uniform-sub-float32-precision-is-not-achievable-today)
is that `cholesky` and `norm.ppf` have no kernel below float32. I re-ran it rather than
inherit it, and added `matmul` to the sweep — which is the column that changes the plan.

| dtype | `jnp.linalg.cholesky` | `jax.scipy.stats.norm.ppf` | `matmul` |
|---|---|---|---|
| float64 | OK | OK | OK |
| float32 | OK | OK | OK |
| bfloat16 | `NotImplementedError` | `TypeError` | **OK** |
| float16 | `NotImplementedError` | `TypeError` | **OK** |
| float8_e4m3fn | `NotImplementedError` | `TypeError` | **OK** |
| float8_e5m2 | `NotImplementedError` | `TypeError` | **OK** |
| float4_e2m1fn | `NotImplementedError` | `TypeError` | **OK** |

**The existing analysis is confirmed in full.** No reduced-precision tier is easier to
reach than any other — bfloat16 fails identically to FP4, exactly as the architecture doc
says (and its "previously-considered belief, corrected" note remains correct).

**What is new: `matmul` is unbroken all the way down to FP4.** The architecture doc already
anticipated this for Option A's two insertion points ("matmul, proven by Option A's own two
insertion points"). The sweep confirms it holds for *every* sub-float32 dtype, which is what
makes the replacement kernels in [02-kernels.md](02-kernels.md) viable: both blocked
primitives can be rebuilt from matmul plus elementwise arithmetic.

```python
# reproduce
import jax, jax.numpy as jnp, numpy as np
jax.config.update('jax_enable_x64', True)
A = np.eye(4)*2 + 0.1
for name in ['float64','float32','bfloat16','float16','float8_e4m3fn','float8_e5m2','float4_e2m1fn']:
    dt = getattr(jnp, name)
    for label, fn in [('cholesky', lambda: jnp.linalg.cholesky(jnp.asarray(A, dtype=dt))),
                      ('ppf', lambda: jax.scipy.stats.norm.ppf(jnp.asarray([0.3], dtype=dt))),
                      ('matmul', lambda: (lambda x: x @ x)(jnp.asarray(A, dtype=dt)))]:
        try: fn(); print(name, label, 'OK')
        except Exception as e: print(name, label, type(e).__name__)
```

---

## F-2. A matmul-only matrix square root runs where `cholesky` cannot

Newton–Schulz iteration computes a symmetric matrix square root using *only* matmul —
no LAPACK, no factorization kernel. On an 8×8 SPD covariance-like matrix, reconstructing
`S·Sᵀ ≈ A`:

| compute dtype | max abs reconstruction error | vs `\|A\|max = 2.83` |
|---|---|---|
| float32 | 1.1e-06 | ~4e-7 relative |
| bfloat16 | 2.5e-02 | ~9e-3 relative |
| float16 | 3.4e-03 | ~1e-3 relative |

**This is the load-bearing result for the simulation step.** It demonstrates the
architecture doc's own proposed remedy — "a custom low-precision Cholesky avoiding LAPACK
entirely... via Newton–Schulz iteration (matmul-only, quadratically convergent)" — actually
running at bfloat16 and float16 on this backend.

Two caveats that the plan must carry, not bury:

- **A square root is not a Cholesky.** Newton–Schulz returns a symmetric `S`, not a lower
  triangular `L`. For generating correlated shocks this is fine — any `S` with `S·Sᵀ = A`
  induces the same covariance, and the engine only ever uses the factor as a mixing matrix.
  It is *not* a drop-in for any code path that depends on triangularity. See
  [02-kernels.md](02-kernels.md#k-1) for where that distinction bites.
- **float16 beat bfloat16 by ~7×.** First appearance of the theme that runs through this
  whole plan: mantissa bits matter more than exponent range for this workload.

---

## F-3. The NaN result — uniform low precision is the wrong target

This is the finding that most changes the architecture, and it was not visible from the
existing analysis.

An Acklam inverse-normal-CDF (rational minimax, elementwise arithmetic only — the second
remedy the architecture doc proposes) evaluated **entirely in** each dtype, over 20,000
uniforms, against `scipy.stats.norm.ppf`:

| compute dtype | max abs error |
|---|---|
| float64 | 3.9e-09 |
| float32 | 3.0e-04 |
| bfloat16 | **NaN** |
| float16 | **NaN** |
| float8_e4m3fn | **NaN** |

**Diagnosis.** Not a kernel gap — an overflow. Acklam's coefficients reach ±276, and the
polynomial's intermediate terms exceed the dynamic range of the low formats
(float16 max ≈ 65,504, and far tighter once intermediates compound). The approximation is
numerically fine; the *format* cannot hold its intermediates.

**This generalizes beyond Acklam.** Any minimax polynomial for this function has
large-magnitude coefficients — that is how they achieve accuracy across the tail. So
"implement `norm.ppf` in low precision" is not achievable by swapping the dtype, for *any*
of the candidate approximations the architecture doc lists (Wichura AS 241 has the same
character). The conclusion is structural, not specific to one formula.

---

## F-4. Compute-high / store-low works at every tier

Same Acklam approximation, evaluated in **float32**, result cast to the target dtype:

| storage dtype | max abs error | rms error |
|---|---|---|
| bfloat16 | 7.8e-03 | 1.7e-03 |
| float16 | 9.8e-04 | 2.1e-04 |
| float8_e4m3fn | 1.2e-01 | 2.6e-02 |
| float8_e5m2 | 2.5e-01 | 5.3e-02 |

No NaN at any tier. Errors are now pure *quantization* error — the cost of representing the
answer, not of computing it — which is well-understood, bounded, and predictable, unlike
overflow.

**This is already the codebase's established pattern**, which is a strong argument for it:
[`engine/simulation/market_model.py:67`](../../../engine/simulation/market_model.py#L67)
computes `norm.ppf(uniform_clipped).astype(dtype)` and its docstring already notes that
"`norm.ppf` computes internally in [higher precision]... the explicit `.astype(dtype)` is
required regardless of the ambient x64 state." The 16/8/4 tiers extend a pattern the engine
has relied on since the float32 tier, rather than introducing a new one.

---

## F-5. Quantization noise vs. Monte Carlo error — the precision ceiling

The engine's answers are already noisy: Monte Carlo error over `N` paths falls as `1/√N`.
Quantization noise only matters when it rises above that floor. Setting the F-4 rms errors
equal to `1/√N` gives, for each dtype, the path count beyond which **more paths buy nothing**:

| storage dtype | shock rms error | useful path ceiling |
|---|---|---|
| float16 | 2.1e-04 | ~22,800,000 |
| bfloat16 | 1.7e-03 | ~359,000 |
| float8_e4m3fn | 2.6e-02 | ~1,400 |
| float8_e5m2 | 5.3e-02 | ~360 |

Read against realistic path counts (using a 3× margin — quantization noise should sit well
under MC noise, not merely beneath it):

| paths | MC error | float16 | bfloat16 | FP8 |
|---|---|---|---|---|
| 10,000 | 1.0e-02 | OK | OK | dominates |
| 100,000 | 3.2e-03 | OK | dominates | dominates |
| 1,000,000 | 1.0e-03 | OK | dominates | dominates |

**This is the quantitative form of the project's core research question.** The roadmap asks
whether many more low-precision simulations reach the same risk answer as fewer
high-precision ones. The answer is that it depends entirely on which side of the dtype's
ceiling you are on, and the trade only pays inside it. Consequences in
[03-performance.md](03-performance.md).

**Scope honesty.** These are errors in the *shocks*, the pipeline's input. They are a
necessary condition, not a sufficient one — error propagation through pricing, VaR/ES and
Greeks is not measured here and must be established per instrument before any tier is
trusted (see [04-roadmap.md](04-roadmap.md#validation-bar)). Greeks are the known risk:
they difference nearby values, which amplifies relative error.

---

## F-6. What remains unmeasured

Stated explicitly so the plan is not read as more settled than it is.

- **Everything downstream of shock generation.** Pricing, VaR/ES and Greeks error at each
  tier — the necessary/sufficient gap in F-5.
- **FP4 as a storage format.** `float4_e2m1fn` supports matmul, but with ~3 representable
  magnitudes in the shock range no useful prototype was obtained. Treated as a research
  target with a falsification test, not a deliverable.
- **Any speedup.** No timing was measured. On this CPU backend low-precision formats are
  generally *emulated* and can be slower than float32; the throughput case for 16/8 rests
  on TPU hardware and is unverified here. No performance claim in this plan is measured —
  all are labeled as projections.
- **TPU kernel coverage.** Unchanged from the architecture doc's labeled hypothesis: a TPU
  backend may have native bfloat16 coverage for `cholesky`/`norm.ppf`, which would make the
  replacement kernels unnecessary *for that tier only*. Cannot be tested here.
