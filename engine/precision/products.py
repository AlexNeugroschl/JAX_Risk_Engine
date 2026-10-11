"""
Matrix products at the policy's compute format, accumulated in its accumulate format
(decisions A-16, A-22; docs/planning/details/precision.md §6.7, §8.3).

XLA leaves a product's precision to the device unless the program states it: by default a
float32 product runs in TensorFloat-32 (a 10-bit mantissa) on an NVIDIA GPU and in bfloat16
passes on a TPU, so a policy that computes in float32 would compute below it while its
precision report says float32 (measured on an RTX 5060 on 2026-10-06: a float32 cube 1.5%
off). Every matrix product of the engine's JAX code therefore goes through `matmul`, which
states the product in the program itself; JAX's process-wide `jax_default_matmul_precision`
is neither set nor read. NumPy products on the host are exact in their dtype and do not need it.

A product in compute format `c`, accumulated in `a` (`matmul(x, y, arithmetic)`):

  * float64, float32: the operands cast to `c`, multiplied at full precision
    (`lax.Precision.HIGHEST`), accumulated in `a`. float32 operands accumulated in float64 are
    rounded onto float32's grid in float64 (exact) and multiplied in float64, where every
    product of two is exact: a GPU asked for float32 operands with a float64 result
    (`preferred_element_type`) accumulates in float32 (an RTX 5060, 2026-10-11);
  * float16, bfloat16, FP8 (an array format below float32): each row of the left operand and
    each column of the right one is scaled by a power of two that brings its largest magnitude
    to the format's top (but bfloat16's, whose range is float32's), rounded to the format's grid
    exactly (`engine.precision.grid`: XLA would round float64 to FP8 twice) and cast to it; the
    product accumulates in float32 and the scales are taken back out. The scales are exact and
    per row (a path), so no path's rounding depends on another path's operands, whatever its
    batch, sample or device split (XLA may still sum a product of another shape in another
    order, an ulp apart, as at float32). This is how an FP8 matrix unit is fed (per-tensor or
    finer scales, float32 accumulation);
  * tensorfloat32: float32 operands rounded to a 10-bit mantissa, float32 accumulation;
  * bfloat16_3x: each float32 operand split into its bfloat16 and the bfloat16 of the
    remainder, three products of the pieces (the remainders' product dropped), float32
    accumulation.

Every format below float32 accumulates in float32, as the matrix units that run it do
(`engine.precision.arithmetic`). A product format is **native** on a device with units for it,
where the program hands XLA the format itself (the format's dtype, or its algorithm,
`lax.DotAlgorithmPreset`), and **emulated** elsewhere: the engine rounds the operands itself and
multiplies them in float32, which gives the format's numbers (a product of two values of a
format below float32 is exact in float32) up to the order of the float32 sums; an emulated FP8
product runs its rounded operands as float16 values, which hold them exactly. On a CPU every
format below float32 is emulated, and FP8 on every device, and bfloat16 passes on a GPU, until
a device's are shown right (an RTX 5060's were not, 2026-10-10); `product_execution` says which
ran, and the precision report names it per compute format (`PrecisionReport.products`).

`tests/test_accelerator_defaults.py::TestEveryMatrixProductStatesItsPrecision` traces the
pipelines and fails on any product of the engine's that does not state its precision.
"""
import math
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax

from engine.precision.formats import Format, format_of, name_of
from engine.precision.grid import pow2, round_to_grid, scale_exponent

#: Compute capability (major.minor) from which an NVIDIA GPU has units for each format below
#: float32 that the engine runs natively: float16 from Volta (7.0), TensorFloat-32 and bfloat16
#: from Ampere (8.0), each shown equal to the emulation on an RTX 5060 (2026-10-10,
#: `tests/support/precision_devices.py`). Not FP8, nor bfloat16 passes: there (compute capability
#: 12.0, jax 0.10.2) XLA's FP8 product returned products 70% off for some shapes and precisions,
#: and its `BF16_BF16_F32_X3` one a single pass's error (8.8e-4 of the scale) for some shapes, so
#: both are emulated on a GPU until a device's are shown right. The faults are XLA's (I-97): its
#: Triton GEMM emitter gets FP8 wrong where cuBLASLt is exact, and a matrix-vector product
#: compiles to a fusion that ignores the passes' algorithm.
_GPU_UNITS = {"float16": 7.0, "bfloat16": 8.0, "tensorfloat32": 8.0}
#: The formats below float32 a TPU's matrix units run: bfloat16 and its passes on every
#: generation, XLA's own way of running a float32 product there, to be shown right on the device
#: as on a GPU (FP8, from Ironwood, emulated until then).
_TPU_UNITS = {"bfloat16": "", "bfloat16_3x": ""}
#: The dtype an emulated format's operands are multiplied in: one that holds every value of the
#: format exactly and whose products are exact in float32 (every FP8 value is a float16 value).
_CARRIERS = {"float8_e4m3fn": jnp.float16, "float8_e5m2": jnp.float16}
#: The algorithm XLA is given for a native product format.
_ALGORITHMS = {"tensorfloat32": lax.DotAlgorithmPreset.TF32_TF32_F32,
               "bfloat16_3x": lax.DotAlgorithmPreset.BF16_BF16_F32_X3}
#: The exponent of float32's largest binade: a format that reaches it has float32's range.
_FLOAT32_TOP_EXPONENT = math.frexp(float(jnp.finfo(jnp.float32).max))[1]


def product_execution(name: str, device: Optional[jax.Device] = None) -> str:
    """`"native"` if `device` (`default_device()` by default: the one a product traced now runs
    on) has matrix units for compute format `name`, else `"emulated"` (see the module docstring). float64 and
    float32 are native on a CPU and a GPU; a TPU emulates float64 and runs float32 in bfloat16
    passes, which XLA arranges itself (native)."""
    fmt = format_of(name)
    device = device or default_device()
    platform = device.platform
    if not fmt.below_float32:
        return "emulated" if platform == "tpu" and fmt.bits == 64 else "native"
    if platform == "gpu":
        capability = getattr(device, "compute_capability", None)
        units = _GPU_UNITS.get(fmt.name)
        return "native" if None not in (capability, units) and float(capability) >= units else "emulated"
    if platform == "tpu" and fmt.name in _TPU_UNITS:
        return "native" if _TPU_UNITS[fmt.name] in device.device_kind else "emulated"
    return "emulated"


def default_device() -> jax.Device:
    """The device a product traced now runs on: JAX's configured default device
    (`jax.default_device`, a device or a platform), else its first device."""
    configured = jax.config.jax_default_device
    if isinstance(configured, str):
        return jax.devices(configured)[0]
    return configured or jax.devices()[0]


def product_precision(compute, device: Optional[jax.Device] = None):
    """The precision a product in compute format `compute` (a name, or the dtype of an array
    format) states in the program: its algorithm where a product format runs natively, else full
    precision on the (rounded) operands."""
    fmt = format_of(compute if isinstance(compute, str) else name_of(compute))
    if fmt.compute_pending is not None:
        raise ValueError(f"a matrix product in {fmt.name} is not enabled yet ({fmt.compute_pending})")
    if fmt.product in _ALGORITHMS and product_execution(fmt.name, device) == "native":
        return _ALGORITHMS[fmt.product]
    return lax.Precision.HIGHEST


def matmul(a, b, arithmetic=None):
    """`a @ b` (`jnp.matmul`: broadcast over leading axes, a 1-D operand as a vector) in
    `arithmetic`'s compute format, accumulated and returned in its accumulate format (an
    `engine.precision.Arithmetic`). Without one, the operands' own format, float64 or float32,
    accumulated in it: the products of the stages fixed at float64 or float32 (A-10)."""
    a, b = jnp.asarray(a), jnp.asarray(b)
    if arithmetic is None:
        name = name_of(jnp.result_type(a, b))
        if format_of(name).below_float32:
            raise ValueError(f"a product of {name} operands needs the stage's arithmetic "
                             f"(engine.precision.Arithmetic): the format alone does not say what it accumulates in")
        compute, accumulate = name, name
    else:
        compute, accumulate = arithmetic.compute, arithmetic.accumulate
    fmt, out = format_of(compute), format_of(accumulate).dtype
    precision = product_precision(compute)
    if not fmt.below_float32:
        if fmt.dtype != out:   # float32 operands, float64 accumulation: float32's values in float64
            a, b = (round_to_grid(x.astype(out), fmt.mantissa_bits, fmt.min_exponent) for x in (a, b))
        return jnp.matmul(a.astype(out), b.astype(out), precision=precision)
    vector_a, vector_b = a.ndim == 1, b.ndim == 1
    a, b = (a[None, :] if vector_a else a), (b[:, None] if vector_b else b)
    if fmt.product == "dtype":
        carrier = fmt.dtype if product_execution(compute) == "native" else _CARRIERS.get(compute, fmt.dtype)
        product = _scaled_product(a, b, fmt, carrier, precision)
    elif fmt.product == "tensorfloat32" and precision is lax.Precision.HIGHEST:
        product = _matmul32(*(round_to_grid(x.astype(jnp.float32), fmt.mantissa_bits, fmt.min_exponent)
                              for x in (a, b)), precision)
    elif fmt.product == "bfloat16_3x" and precision is lax.Precision.HIGHEST:
        (a_hi, a_lo), (b_hi, b_lo) = _split_bfloat16(a), _split_bfloat16(b)
        product = (_matmul32(a_lo, b_hi, precision) + _matmul32(a_hi, b_lo, precision)) + _matmul32(a_hi, b_hi,
                                                                                                     precision)
    else:
        product = _matmul32(a.astype(jnp.float32), b.astype(jnp.float32), precision)
    product = product[..., 0, :] if vector_a else product
    return (product[..., 0] if vector_b else product).astype(out)


def _matmul32(a, b, precision):
    """`a @ b` accumulated in float32: an algorithm states its own accumulation."""
    if isinstance(precision, lax.DotAlgorithmPreset):
        return jnp.matmul(a, b, precision=precision)
    return jnp.matmul(a, b, precision=precision, preferred_element_type=jnp.float32)


def _scaled_product(a, b, fmt: Format, carrier, precision):
    """`a @ b` on operands rounded to the array format `fmt` (below float32), each row of `a`
    and column of `b` scaled into its range by a power of two (see the module docstring), held
    in `carrier` (the format's dtype, or one that holds it exactly)."""
    qa, ka = _quantize(a, fmt, axis=-1)          # ka [..., M, 1]: per row
    qb, kb = _quantize(b, fmt, axis=-2)          # kb [..., 1, N]: per column
    product = _matmul32(qa.astype(carrier), qb.astype(carrier), precision)
    return product * pow2(-ka, jnp.float32) * pow2(-kb, jnp.float32)


def _quantize(x, fmt: Format, axis: int):
    """`x` scaled along `axis` by the powers of two `2^k` that bring each slice's largest finite
    magnitude into (max/2, max] of `fmt`, rounded to its grid and cast to it; and `k`. A format
    with float32's range (bfloat16) needs no scale: `k` is 0, and scaling it to its top would
    overflow the float32 accumulator."""
    if math.frexp(fmt.max)[1] >= _FLOAT32_TOP_EXPONENT:
        k = jnp.zeros(x.shape[:axis % x.ndim] + (1,) + x.shape[axis % x.ndim + 1:], dtype=jnp.int32)
    else:
        amax = jnp.max(jnp.where(jnp.isfinite(x), jnp.abs(x), 0), axis=axis, keepdims=True, initial=0)
        k = scale_exponent(amax, fmt.max)
    scaled = x * pow2(k, x.dtype)
    return round_to_grid(scaled, fmt.mantissa_bits, fmt.min_exponent).astype(fmt.dtype), k


def _split_bfloat16(x):
    """`x` (as float32) as a bfloat16 and the bfloat16 of what it leaves, each rounded on
    bfloat16's grid in float32 arithmetic (`engine.precision.grid`) and cast, exactly, at the
    end: XLA may remove a float32 -> bfloat16 -> float32 round trip (`xla_allow_excess_precision`,
    on by default), which made the remainder 0 on a GPU (2026-10-10), so the split never
    round-trips through the dtype."""
    bfloat16 = format_of("bfloat16")
    x = x.astype(jnp.float32)
    high = round_to_grid(x, bfloat16.mantissa_bits, bfloat16.min_exponent)
    low = round_to_grid(x - high, bfloat16.mantissa_bits, bfloat16.min_exponent)
    return high.astype(jnp.bfloat16), low.astype(jnp.bfloat16)
