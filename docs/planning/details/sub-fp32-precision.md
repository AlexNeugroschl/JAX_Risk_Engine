# Sub-FP32 precision

Research findings and the plan for [F-07](../features.md#f-07): extending per-stage
precision from `{64, 32}` to 16-bit storage, and recording why 8- and 4-bit are rejected for
path storage. Measured on CPU with `jax==0.10.2` (2026-09-17, commit `e4ca50b`); all
throughput statements are projections until run on TPU.

## Findings

**F-1. Kernel coverage.** Re-run live, every sub-float32 dtype behaves the same:

| dtype | `jnp.linalg.cholesky` | `jax.scipy.stats.norm.ppf` | `matmul` |
|---|---|---|---|
| float64, float32 | OK | OK | OK |
| bfloat16, float16, float8_e4m3fn, float8_e5m2, float4_e2m1fn | `NotImplementedError` | `TypeError` | **OK** |

```python
import jax, jax.numpy as jnp, numpy as np
jax.config.update('jax_enable_x64', True)
A = np.eye(4) * 2 + 0.1
for name in ['float64', 'float32', 'bfloat16', 'float16', 'float8_e4m3fn', 'float8_e5m2', 'float4_e2m1fn']:
    dt = getattr(jnp, name)
    for label, fn in [('cholesky', lambda: jnp.linalg.cholesky(jnp.asarray(A, dtype=dt))),
                      ('ppf', lambda: jax.scipy.stats.norm.ppf(jnp.asarray([0.3], dtype=dt))),
                      ('matmul', lambda: (lambda x: x @ x)(jnp.asarray(A, dtype=dt)))]:
        try: fn(); print(name, label, 'OK')
        except Exception as e: print(name, label, type(e).__name__)
```

**F-2. A matmul-only matrix square root runs below float32.** Newton–Schulz on an 8×8 SPD
matrix (`max|A| = 2.83`), error `max|S·Sᵀ − A|`: float32 1.1e-6, float16 3.4e-3, bfloat16
2.5e-2. It returns a symmetric `S`, not a triangular `L`: fine as a mixing matrix, wrong
anywhere triangularity matters, and it returns garbage instead of failing on a non-SPD
matrix.

**F-3. Computing *in* a low format overflows.** Acklam's inverse normal CDF (elementwise, no
special function) evaluated in the target dtype over 20,000 uniforms: float64 3.9e-9,
float32 3.0e-4, bfloat16/float16/FP8 **NaN**. Its coefficients reach ±276 and the
intermediates overflow the format. Every minimax approximation of this function has large
coefficients (Wichura AS 241 too), so this is structural.

**F-4. Computing in float32 and storing low works at every tier.** Same approximation, float32
compute, result cast:

| Storage | max abs error | rms error |
|---|---|---|
| float16 | 9.8e-4 | 2.1e-4 |
| bfloat16 | 7.8e-3 | 1.7e-3 |
| float8_e4m3fn | 1.2e-1 | 2.6e-2 |
| float8_e5m2 | 2.5e-1 | 5.3e-2 |

This is already the engine's pattern: `engine/simulation/random.py` computes
`norm.ppf(uniform_clipped).astype(dtype)`.

**F-5. Each storage format caps the useful path count.** Quantization noise is a fixed floor;
Monte Carlo error falls as `1/√N`. Setting the F-4 rms error equal to `1/√N`:

| Storage | Useful path ceiling |
|---|---|
| float16 | ~22,800,000 |
| bfloat16 | ~359,000 |
| float8_e4m3fn | ~1,400 |
| float8_e5m2 | ~360 |

So the rule is "the lowest precision whose ceiling is above the path count you need", not
"the lowest precision that runs". float16 beats bfloat16 (10 mantissa bits beat 8 for O(1)
shocks, inverting the ML default); FP8 is not a path-storage format; FP4 has about 3
representable magnitudes in the shock range and no working prototype.

**F-6. Unmeasured.** Error propagation into pricing, VaR/ES and Greeks (the findings concern
only the shocks, a necessary condition); any timing (on CPU, sub-float32 formats are emulated
and likely slower; the case rests on memory traffic on TPU); TPU kernel coverage (bfloat16 may
have native `cholesky`/`norm.ppf` there).

## Where the kernels sit today

- **Inverse CDF**: `engine/simulation/random.py`, `norm.ppf(...).astype(dtype)`. Blocking
  below float32.
- **Cholesky:** `engine.simulation.cam.flexible_cholesky` factors each step's covariance in
  NumPy on the host (FP64) once per configuration, then casts to the normals' dtype. Only a
  matmul runs in the storage dtype, so no replacement kernel is needed. Since roadmap 1.3 this
  is the only simulation, for either interest-rate model (the Hull-White model's separate
  `jnp.linalg.cholesky` per step went with its pipeline).

## Design

**Storage and compute split.** A precision setting becomes `(storage, compute)` with
`compute ≥ max(storage, 32)`. `64` and `32` keep meaning `(64, 64)` and `(32, 32)`.

| Tier | Storage | Compute | Status |
|---|---|---|---|
| 64 | float64 | float64 | Built |
| 32 | float32 | float32 | Built |
| 16 | float16 (default), bfloat16 selectable | float32 | Build |
| 8, 4 | FP8, FP4 | float32 | Rejected for path storage, with F-5's reason |

**The per-stage configuration keeps its shape** (simulation, pricing, risk, calibration, with
per-instrument and per-Greek overrides); only the values widen. With the precision
mechanism replaced by explicit dtypes ([I-55](../known-issues.md#i-55), roadmap 1.4) there are
no per-precision process pools to extend.

**Reject, never coerce.** `engine.portfolio.request._dtype_of` maps anything not 64 to
float32; today membership checks guard it, but a new tier added without replacing it with an
explicit table that raises would silently run at float32. Same at the HTTP boundary: an
unknown precision is a 4xx. Every result states the tiers it ran at
([I-12](../known-issues.md#i-12)).

## Plan

Never debug a new algorithm and a new dtype at once.

1. **Plumbing, no new dtypes.** The `(storage, compute)` type, the raising dtype table,
   schema validation. Exit: the suite and every parity test unchanged.
2. **Inverse-CDF kernel at FP64/FP32.** Acklam (plus one Halley step if the validation bar
   needs it), dispatched only when requested. Validate against `scipy.stats.norm.ppf` in the
   bulk and, separately, in the tails that VaR/ES read; the clip `eps` is dtype-dependent and
   moves with the tier. Exit: with the kernel on at FP64, ORE parity within existing
   tolerances.
3. **Tier 16.** FP16 storage end to end. Measure the unmeasured half (F-6): per-instrument NPV,
   exposure profiles, VaR/ES and Greeks against tier 64. Expect Greeks to need a tier higher
   than pricing (they difference nearby values); per-Greek overrides already allow it.
   Tolerances are set before measuring, and the results enter the evidence table of roadmap
   2.7.
4. **Record the FP8/FP4 rejection** with downstream measurements, and an FP4 falsification
   test (show it cannot represent a standard normal usefully). FP8 may still suit
   float32-accumulated matmuls or fast screening runs; that is a separate track.
5. **On TPU, first:** re-run F-1's sweep. If bfloat16 `cholesky`/`norm.ppf` are native there,
   bfloat16 may lead on hardware grounds. Then throughput measurements and device-count-aware
   pool sizing ([I-61](../known-issues.md#i-61)).

**What would make this plan wrong:** tier 16 failing the bar for NPV (not just Greeks), native
bfloat16 coverage on TPU, or the simulation stage not being bandwidth-bound at production
shapes (the main projected benefit is memory traffic: 2 bytes per element against 4 or 8).
