# XLA GPU products: minimal reproductions (I-97)

Two faults of XLA's GPU matrix products, found by the engine's device check
(`tests/support/precision_devices.py`) and reduced to plain JAX, for reporting upstream
([I-97](../known-issues.md#i-97)). Measured on an NVIDIA GeForce RTX 5060 Laptop GPU (compute
capability 12.0), driver 592.01, jax and jaxlib 0.10.2 with the CUDA 13 plugin, WSL2 Ubuntu
24.04, 2026-10-19. Not yet filed: filing is the owner's (an outward-facing action).

## 1. FP8 products through the Triton GEMM emitter are wrong

```python
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
rng = np.random.default_rng(3)
for m, k, n in [(512, 64, 8), (64, 64, 64), (16, 32, 16), (4096, 128, 32)]:
    a = jnp.asarray(rng.standard_normal((m, k)), jnp.float32).astype(jnp.float8_e4m3fn)
    b = jnp.asarray(rng.standard_normal((k, n)), jnp.float32).astype(jnp.float8_e4m3fn)
    out = jax.jit(lambda x, y: jnp.matmul(x, y, preferred_element_type=jnp.float32))(a, b)
    exact = np.asarray(a, np.float64) @ np.asarray(b, np.float64)
    scale = np.abs(np.asarray(a, np.float64)) @ np.abs(np.asarray(b, np.float64))
    print(m, k, n, float(np.max(np.abs(np.asarray(out, np.float64) - exact) / scale)))
```

With `XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true` (the engine worker's, decision A-22)
every product is 0.60 to 0.74 of `|a|·|b|` off (0.54 to 0.79 over every FP8 pair, e4m3 or e5m2 on
either side, in the device check); without it, XLA's autotuner picks cuBLASLt for most shapes and
only the smallest (16 × 32 × 16) is wrong, 0.74 off. Adding `--xla_gpu_enable_triton_gemm=false`,
XLA calls `__cublas$lt$matmul$f8` (two e5m2 operands its non-FP8 path) and every product is exact
to 3e-8 (float32's sums). A matrix-vector shape (n = 1) compiles to a fusion and is exact under
both. Expected: the Triton emitter's FP8 product equal to cuBLASLt's, or the Triton path not
chosen for FP8 on this architecture.

## 2. A matrix-vector product ignores `BF16_BF16_F32_X3`

```python
import jax, jax.numpy as jnp, numpy as np
from jax import lax
rng = np.random.default_rng(3)
for m, k, n in [(512, 64, 8), (512, 64, 1), (1, 64, 512)]:
    a, b = rng.standard_normal((m, k)).astype(np.float32), rng.standard_normal((k, n)).astype(np.float32)
    f = jax.jit(lambda x, y: jnp.matmul(x, y, precision=lax.DotAlgorithmPreset.BF16_BF16_F32_X3))
    exact = a.astype(np.float64) @ b.astype(np.float64)
    scale = np.abs(a.astype(np.float64)) @ np.abs(b.astype(np.float64))
    print(m, k, n, float(np.max(np.abs(np.asarray(f(a, b), np.float64) - exact) / scale)))
```

The matrix shape is 4e-6 off (1.1e-5 through cuBLAS), three bfloat16 passes' error; both
matrix-vector shapes 9.4e-4 to 9.8e-4, a single pass's: XLA compiles them as fusions, not GEMMs, without the algorithm, under
either GEMM emitter. `TF32_TF32_F32` is honoured on the same shapes. Expected: the algorithm
honoured, or the dot refused, rather than computed at a lower precision than asked.

## What the engine does meanwhile

FP8 runs natively on a GPU only in a process whose products go to cuBLAS (`engine.precision.
products.cublas_gemms`), emulated exactly elsewhere; bfloat16 passes run natively on matrix
shapes, emulated on matrix-vector ones (`EMULATED_MATRIX_VECTOR`); the device check passes every
format both ways.
