# 02 — The two replacement kernels

Everything below float32 is blocked by exactly two library primitives
([F-1](00-findings.md#f-1-kernel-coverage-by-dtype--confirmed-not-assumed)). This document
specifies their replacements. Both were prototyped; the prototype code and its measured
error are included so implementation starts from something known to run, not from a
description.

The architecture doc already named both remedies — "a custom low-precision Cholesky avoiding
LAPACK entirely... via Newton–Schulz iteration" and "a custom inverse-normal-CDF... a
rational/polynomial minimax approximation (e.g. Wichura's AS 241) or an Acklam... style
closed-form approximation." This document is the step past naming them: they run, here is
what they cost.

---

## K-1. Matmul-only matrix square root (replaces `jnp.linalg.cholesky`)

**Why it is needed.** The Cholesky factor of the joint covariance is how cross-asset
correlation enters the simulation — it is not peripheral. `cholesky` has no kernel below
float32, but `matmul` works down to FP4
([F-1](00-findings.md#f-1-kernel-coverage-by-dtype--confirmed-not-assumed)), so a
factorization built only from matmuls is reachable.

**Method.** Newton–Schulz iteration, which converges quadratically to `A^{1/2}` using only
matrix multiplication and scalar arithmetic:

```python
def ns_sqrt(A, dt, iters=20):
    """Matmul-only symmetric matrix square root. No LAPACK, no factorization kernel."""
    A = jnp.asarray(A, dtype=dt)
    nrm = jnp.sqrt(jnp.sum(jnp.asarray(A, dtype=jnp.float32)**2)).astype(dt)
    Y = A / nrm
    Z = jnp.eye(A.shape[0], dtype=dt)
    for _ in range(iters):
        T = (3.0 * jnp.eye(A.shape[0], dtype=dt) - Z @ Y) / 2.0
        Y = Y @ T
        Z = T @ Z
    return Y * jnp.sqrt(nrm)
```

**Measured** (8×8 SPD matrix, `|A|max = 2.83`, reconstruction `max|S·Sᵀ − A|`):

| dtype | error | note |
|---|---|---|
| float32 | 1.1e-06 | reference-quality |
| float16 | 3.4e-03 | usable |
| bfloat16 | 2.5e-02 | ~7× worse than float16 |

### The triangularity caveat — read before implementing

**Newton–Schulz returns a symmetric `S`, not a lower-triangular `L`.** For shock generation
this is harmless: any `S` satisfying `S·Sᵀ = A` induces the correct covariance, and the
engine uses the factor purely as a mixing matrix applied to independent normals. The
simulated distribution is identical.

It is **not** a drop-in anywhere triangularity itself is load-bearing — forward/back
substitution, or any code reading the factor's structure. Before implementing, audit both
`cholesky` call sites
([`market_model.py:347`](../../../engine/simulation/market_model.py#L347) and `:401`) to
confirm neither depends on the triangular form. The `:401` site is a *validation* path
(detecting invalid correlation matrices via `cholesky` failure) and needs separate handling:
**Newton–Schulz does not fail on a non-SPD matrix the way `cholesky` does — it silently
returns garbage.** Keeping that validation at float32/64 regardless of tier is the safe
choice, and costs nothing: it runs once per config, not per path.

**Cost.** 20 iterations × 2 matmuls = 40 matmuls of size `n×n`, where `n` is the factor
count (small — single digits to low tens). Negligible against path generation, and it runs
once per simulation, not per path. Iteration count should be tuned — 20 is conservative and
quadratic convergence likely reaches tolerance far sooner.

**Alternative if it disappoints.** The architecture doc also lists blocked/recursive
elimination. Newton–Schulz is preferred for being matmul-only (the one op with proven
coverage at every tier) and branch-free, which suits XLA.

---

## K-2. Elementwise inverse normal CDF (replaces `jax.scipy.stats.norm.ppf`)

**Why it is needed.** Every Sobol uniform draw becomes a normal shock through `norm.ppf` —
once per (scenario, time step, factor), the most-executed numerical step in the engine.
It is blocked below float32 by a missing special-function kernel.

**Method.** Acklam's rational minimax approximation — pure elementwise arithmetic
(`+ − × ÷ sqrt log`), no special-function kernel:

```python
def acklam_ppf(u, compute_dt=jnp.float32):
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    u = jnp.asarray(u, dtype=compute_dt)          # NOTE: compute dtype, never storage
    pl = jnp.asarray(0.02425, dtype=compute_dt)

    def lower(u):
        q = jnp.sqrt(-2 * jnp.log(u))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)

    def central(u):
        q = u - 0.5; r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

    return jnp.where(u < pl, lower(u), jnp.where(u > 1 - pl, -lower(1 - u), central(u)))
```

**Measured**, 20,000 uniforms vs `scipy.stats.norm.ppf`:

| mode | dtype | max abs error |
|---|---|---|
| compute *in* dtype | float32 | 3.0e-04 |
| compute *in* dtype | bfloat16 / float16 / FP8 | **NaN** |
| compute fp32, store | float16 | 9.8e-04 |
| compute fp32, store | bfloat16 | 7.8e-03 |
| compute fp32, store | float8_e4m3fn | 1.2e-01 |

### The NaN is the whole design constraint

**The `compute_dt` argument above must never be set to the storage dtype below float32.**
Acklam's coefficients reach ±276 and the polynomial's intermediates overflow the low formats'
dynamic range — this is overflow, not a kernel gap
([F-3](00-findings.md#f-3-the-nan-result--uniform-low-precision-is-the-wrong-target)).

This is structural, not an artifact of choosing Acklam. Every minimax approximation to this
function carries large coefficients — that is the mechanism by which they stay accurate into
the tails — so Wichura's AS 241 (the architecture doc's other candidate) fails the same way.
**No inverse-normal-CDF approximation of this family can be evaluated natively below
float32.** That is why [01-architecture.md](01-architecture.md#a-1-the-central-change-one-knob-becomes-two)
makes compute-vs-storage a first-class split rather than a single lower knob.

**Accuracy is adequate but not free.** At float32 compute, the approximation's own 3.0e-04
error is *larger* than the float16 quantization it feeds
([F-4](00-findings.md#f-4-compute-high--store-low-works-at-every-tier)) — so at tier 16 the
error is dominated by the approximation, not the storage format. If tier 16 needs to be
sharper, the fix is a refinement step (one Halley/Newton iteration against the forward CDF,
cheap and elementwise) rather than a different storage dtype. Recommended: implement plain
first, measure, add refinement only if the validation bar demands it.

**Tail behavior needs its own test.** Acklam's accuracy degrades in the extreme tails, which
is exactly where VaR and ES read. The existing clipping at
[`market_model.py:65`](../../../engine/simulation/market_model.py#L65) already bounds inputs
to `[eps, 1-eps]`, and `eps` is dtype-dependent — so tier choice shifts the clip point and
interacts with tail accuracy. Test the tails explicitly and separately from the bulk; a
uniform-random error metric will not catch a tail problem, and tail accuracy is what risk
numbers depend on.

---

## K-3. Both kernels, one shared obligation

Neither kernel may be trusted on agreement-with-reference alone. The bar from the
architecture doc applies unchanged: cross-checked against the float64 originals **and** a
separate pass confirming every downstream computation (pricing, VaR/ES, Greeks) stays within
an acceptable error band. [F-6](00-findings.md#f-6-what-remains-unmeasured) is explicit that
the downstream half is unmeasured — see
[04-roadmap.md](04-roadmap.md#validation-bar).

**Both kernels should land at tiers 64/32 first, behind a flag, before any low tier exists.**
They can be validated against the LAPACK/scipy originals at full precision, where any
discrepancy is unambiguously the kernel's fault and not the dtype's. Debugging a new
algorithm and a new dtype simultaneously is the failure mode to avoid.
