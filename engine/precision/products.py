"""
Matrix products at the precision of their operands' compute format (decision A-22;
docs/planning/details/precision.md §6.7).

XLA leaves a product's precision to the device unless the program states it: by default a
float32 product runs in TensorFloat-32 (a 10-bit mantissa) on an NVIDIA GPU and in bfloat16
passes on a TPU, so a policy that computes in float32 would compute below it while its
precision report says float32 (measured on an RTX 5060 on 2026-10-06: a float32 cube 1.5%
off). Every matrix product of the engine's JAX code therefore goes through `matmul`, which
states the precision in the program itself; JAX's process-wide `jax_default_matmul_precision`
is neither set nor read. NumPy products on the host are exact in their dtype and do not need it.

Today every compute format is float64 or float32, multiplied at full precision
(`lax.Precision.HIGHEST`, a no-op on a CPU). F-07's compute formats add the policy's own product formats
(TensorFloat-32, bfloat16 passes, FP8) here, so the policy, never the device, chooses them.
`tests/test_accelerator_defaults.py::TestEveryMatrixProductStatesItsPrecision` traces the
pipelines and fails on any product of the engine's that does not.
"""
import jax.numpy as jnp
from jax import lax

from engine.precision.formats import format_of, name_of


def product_precision(dtype) -> lax.Precision:
    """The precision of a matrix product whose operands' compute format is `dtype`. A format
    not enabled for compute is refused, naming the planning item that enables it."""
    fmt = format_of(name_of(dtype))
    if fmt.compute_pending is not None:
        raise ValueError(f"a matrix product in {fmt.name} is not enabled yet ({fmt.compute_pending})")
    return lax.Precision.HIGHEST


def matmul(a, b):
    """`a @ b` (`jnp.matmul`: broadcast over leading axes, a 1-D operand as a vector) at
    `product_precision` of the operands' dtype."""
    return jnp.matmul(a, b, precision=product_precision(jnp.result_type(a, b)))
