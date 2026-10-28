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

A product in compute format `c`, accumulated in `a` (`matmul(x, y, arithmetic, shared=...)`):

  * float64, float32: the operands cast to `c`, multiplied at full precision
    (`lax.Precision.HIGHEST`), accumulated in `a`. float32 operands accumulated in float64 are
    rounded onto float32's grid in float64 (exact) and multiplied in float64, where every
    product of two is exact: a GPU asked for float32 operands with a float64 result
    (`preferred_element_type`) accumulates in float32 (an RTX 5060, 2026-10-11);
  * float16, bfloat16, FP8, tensorfloat32 and their single passes: operands rounded to
    the selected grid and multiplied with float32 accumulation. `Arithmetic.product`
    explicitly selects optional row/column power-of-two scaling and reconstruction;
  * bfloat16_3x: each float32 operand split into its bfloat16 and the bfloat16 of the
    remainder, three products of the pieces (the remainders' product dropped), float32
    accumulation.

**The shared operand** (decision A-23, I-96). Every product of a stage's kernels multiplies a
per-path operand by one that every path shares, and says which (`shared="left"`: the Brownian
bridge's matrix; `"right"`: the step moments, the market's loadings, each flow's value at the
level). Rounded once, the shared operand's error is the same on every path, so it moves every
figure the same way: a bias, which grows against the statistical error with the paths. A format
below float32 carries it in `ProductOptions.shared_terms` terms: one by default, or
explicit residual terms stacked along its free axis and summed smallest first.
`ProductOptions.compensated(compute)` selects the previous scaled/residual profile.
These strategies trade extra quantization, product width and reconstruction for
less shared-operand rounding bias. See docs/reference/numerical-experiments.md.

`Arithmetic.rounding="stochastic"` rounds only the per-path operand between its two
neighbours with the exact distance probability. This removes its conditional mean
rounding error (I-98); it does not make nonlinear payoffs or exercise decisions unbiased.
Nearest remains the baseline. Named keys and global sample ids are traced pytree leaves,
so changing a seed, sample batch, date or trade does not require a new program.

Every format below float32 accumulates in float32, as the matrix units that run it do
(`engine.precision.arithmetic`). A product format is **native** on a device with units for it,
where the program hands XLA the format itself (the format's dtype, or its algorithm,
`lax.DotAlgorithmPreset`), and **emulated** elsewhere: the engine rounds the operands itself and
multiplies them in float32, which gives the format's numbers (a product of two values of a
format below float32 is exact in float32) up to the order of the float32 sums; an emulated FP8
product uses float16 carriers where their products support FP32 precision, and float32
carriers on TPU. Both hold FP8 operands exactly; TPU's FP16 product loses precision.
On a CPU every
format below float32 is emulated; `product_execution` says which ran, and the precision report
names it per compute format (`PrecisionReport.products`).

**Native dispatch** is selected wherever the backend/device supports the format,
independently of the GEMM emitter. E5M2 uses its own shared grid unless
`product.shared_format="e4m3"` is explicitly selected.
A dispatch label states the requested arithmetic; lowering/traces establish which shapes
actually use matrix units. A compiler may fuse a matrix-vector product instead.
Arithmetic requires native execution by default and refuses implicit emulation.
Numerical qualification is metadata. Software research explicitly opts into
`Arithmetic.execution='allow_emulation'` (or `Precision.research`).

**Compiler controls** (I-97): the RTX 5060 / JAX 0.10.2 checks found wrong FP8
under Triton and a bfloat16-passes GEMV fusion rounding products in BF16. The latter
can explicitly use `product.bf16_padding="gemm8"` to retain native GEMM and FP32
accumulation; it is not enabled by default. cuBLAS (`TRITON_GEMM_OFF`) remains a separate diagnostic route, not a
native FP8 qualification: the H100 trial found arithmetic discrepancies there too.
`JAX_RISK_EMULATE_PRODUCTS` accepts comma-separated
compute formats; a base name includes its single-pass variant. It preserves operand rounding
but uses float32 carriers/products, and the report calls the format emulated. Set it before
tracing, in a fresh process, as with XLA flags; changing it cannot alter cached executables.

`tests/test_accelerator_defaults.py::TestEveryMatrixProductStatesItsPrecision` traces the
pipelines and fails on any product of the engine's that does not state its precision.
"""
import math
import os
from dataclasses import replace
from functools import lru_cache
from typing import Mapping, Optional

import jax
import jax.numpy as jnp
from jax import lax

from engine.precision.formats import Format, format_of, name_of, shared_format
from engine.precision.grid import pow2, round_to_grid, scale_exponent

#: Compute capability (major.minor) from which an NVIDIA GPU has units for each format below
#: float32 that the engine runs natively, by base (a single pass runs on its base's units):
#: float16 from Volta (7.0), TensorFloat-32, bfloat16 and its passes from Ampere (8.0), FP8 from Ada
#: (8.9), each shown equal to the emulation on an RTX 5060 (compute capability 12.0, jax 0.10.2,
#: `tests/support/precision_devices.py`: 2026-10-10; FP8 and the passes 2026-10-19, see the module
#: docstring). Qualification remains specific to the actual compiler/device.
_GPU_UNITS = {"float16": 7.0, "bfloat16": 8.0, "tensorfloat32": 8.0, "bfloat16_3x": 8.0, "float8_e4m3fn": 8.9,
              "float8_e5m2": 8.9}
#: XLA's flag that sends a GPU's matrix products to cuBLAS/cuBLASLt instead of its Triton GEMM
#: emitter (I-97). XLA reads it when it opens the GPU.
TRITON_GEMM_OFF = "--xla_gpu_enable_triton_gemm=false"
#: Optional process-level numerical control or compiler workaround, never set on import.
EMULATE_PRODUCTS = "JAX_RISK_EMULATE_PRODUCTS"
#: TPU compiler candidates. Capability/accuracy qualification does not gate requests.
# These are compiler candidates, not a numerical or instruction qualification.
# Unsupported algorithms/dtypes are refused by the target compiler, without fallback.
_TPU_CANDIDATES = frozenset(("bfloat16", "bfloat16_3x", "float16", "tensorfloat32",
                             "float8_e4m3fn", "float8_e5m2"))
#: The dtype an emulated format's operands are multiplied in, by base: one that holds every value
#: of the format exactly and whose products are exact in float32 (every FP8 value is a float16
#: value).
_CARRIERS = {"float8_e4m3fn": jnp.float16, "float8_e5m2": jnp.float16}
#: The algorithm XLA is given for a native product format.
_ALGORITHMS = {"tensorfloat32": lax.DotAlgorithmPreset.TF32_TF32_F32,
               "bfloat16_3x": lax.DotAlgorithmPreset.BF16_BF16_F32_X3}
#: The exponent of float32's largest binade: a format that reaches it has float32's range.
_FLOAT32_TOP_EXPONENT = math.frexp(float(jnp.finfo(jnp.float32).max))[1]
#: Which operand of a product every path shares (see the module docstring).
SHARED = ("left", "right")


def product_execution(name: str, device: Optional[jax.Device] = None, *, options=None) -> str:
    """`"native"` if `device` (`default_device()` by default: the one a product traced now runs
    on) has a compiler route for compute format `name`, else `"emulated"` (see the module docstring). float64 and
    float32 are native on a CPU and a GPU; a TPU emulates float64 and runs float32 in bfloat16
    passes, which XLA arranges itself (native). Explicit process emulation overrides capability;
    the GEMM emitter never changes this default. Actual instruction use needs lowering evidence."""
    fmt = format_of(name)
    if _explicit_emulation(fmt) or (options is not None and options.implementation == "emulated_fp32"):
        return "emulated"
    device = device or default_device()
    platform = device.platform
    if not fmt.below_float32:
        return "emulated" if platform == "tpu" and fmt.bits == 64 else "native"
    if platform == "gpu":
        capability = getattr(device, "compute_capability", None)
        units = _GPU_UNITS.get(fmt.base)
        return "native" if None not in (capability, units) and float(capability) >= units else "emulated"
    if platform == "tpu" and fmt.base in _TPU_CANDIDATES:
        return "native"
    return "emulated"


@lru_cache(maxsize=32)
def _emulation_formats(value: str):
    names = frozenset(name.strip() for name in value.split(",") if name.strip())
    try:
        if any(not format_of(name).below_float32 for name in names):
            raise ValueError("only compute formats below float32 can be emulated")
    except ValueError as error:
        raise ValueError(f"invalid {EMULATE_PRODUCTS}: {error}") from None
    return names


def _explicit_emulation(fmt: Format) -> bool:
    names = _emulation_formats(os.environ.get(EMULATE_PRODUCTS, ""))
    return fmt.name in names or fmt.base in names


def cublas_gemms(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Whether this process's GPU matrix products go to cuBLAS rather than XLA's Triton GEMM
    emitter: its `XLA_FLAGS` (which XLA reads when it opens the GPU) hold `TRITON_GEMM_OFF`."""
    return TRITON_GEMM_OFF in (os.environ if environ is None else environ).get("XLA_FLAGS", "").split()


def default_device() -> jax.Device:
    """The device a product traced now runs on: JAX's configured default device
    (`jax.default_device`, a device or a platform), else its first device."""
    configured = jax.config.jax_default_device
    if isinstance(configured, str):
        return jax.devices(configured)[0]
    return configured or jax.devices()[0]


def product_precision(compute, device: Optional[jax.Device] = None, matrix_vector: bool = False, *, options=None):
    """The precision a product in compute format `compute` (a name, or the dtype of an array
    format) states in the program: its algorithm where a product format runs natively, else full
    precision on the (rounded) operands. The same algorithm is requested for matrix and
    matrix-vector products; compiler-specific exceptions require explicit emulation."""
    fmt = format_of(compute if isinstance(compute, str) else name_of(compute))
    if fmt.compute_pending is not None:
        raise ValueError(f"a matrix product in {fmt.name} is not enabled yet ({fmt.compute_pending})")
    device = device or default_device()
    if options is not None and options.algorithm == "fp8_full":
        return lax.DotAlgorithmPreset.ANY_F8_ANY_F8_F32
    if device.platform == "tpu" and product_execution(fmt.name, device, options=options) == "native":
        if fmt.base.startswith("float8"):
            return lax.Precision.DEFAULT
        if fmt.base == "float16":
            return lax.DotAlgorithmPreset.F16_F16_F32
        if fmt.base == "bfloat16":
            return lax.DotAlgorithmPreset.BF16_BF16_F32
    if fmt.product in _ALGORITHMS and product_execution(fmt.name, device, options=options) == "native":
        return _ALGORITHMS[fmt.product]
    return lax.Precision.HIGHEST


def matmul(a, b, arithmetic=None, shared: Optional[str] = None, *, sample_axis: int = 0):
    """`a @ b` (`jnp.matmul`: broadcast over leading axes, a 1-D operand as a vector) in
    `arithmetic`'s compute format, accumulated and returned in its accumulate format (an
    `engine.precision.Arithmetic`), `shared` the operand every path shares, `"left"` or `"right"`,
    which a product in a stage's arithmetic must name (see the module docstring). Without an
    arithmetic, the operands' own format, float64 or float32, accumulated in it: the products of
    the stages fixed at float64 or float32 (A-10), whose operands are never rounded below
    float32."""
    a, b = jnp.asarray(a), jnp.asarray(b)
    if arithmetic is None:
        name = name_of(jnp.result_type(a, b))
        if format_of(name).below_float32:
            raise ValueError(f"a product of {name} operands needs the stage's arithmetic "
                             f"(engine.precision.Arithmetic): the format alone does not say what it accumulates in")
        compute, accumulate = name, name
    else:
        if shared not in SHARED:
            raise ValueError(f"a product in a stage's arithmetic names the operand every path shares, "
                             f"shared='left' or 'right' (decision A-23), got {shared!r}")
        compute, accumulate = arithmetic.compute, arithmetic.accumulate
    fmt, out = format_of(compute), format_of(accumulate).dtype
    from engine.precision.execution import validate_product
    from engine.precision.options import ProductOptions
    options = arithmetic.product if arithmetic is not None else ProductOptions()
    validate_product(compute, accumulate, arithmetic.execution if arithmetic is not None else "native", options=options)
    matrix_vector = min(a.shape[-2] if a.ndim > 1 else 1, b.shape[-1] if b.ndim > 1 else 1) == 1
    precision = product_precision(compute, matrix_vector=matrix_vector, options=options)
    if not fmt.below_float32:
        if fmt.dtype != out:   # float32 operands, float64 accumulation: float32's values in float64
            a, b = (round_to_grid(x.astype(out), fmt.mantissa_bits, fmt.min_exponent) for x in (a, b))
        return jnp.matmul(a.astype(out), b.astype(out), precision=precision)
    vector_a, vector_b = a.ndim == 1, b.ndim == 1
    a, b = (a[None, :] if vector_a else a), (b[:, None] if vector_b else b)
    if fmt.product == "bfloat16_3x" and precision is lax.Precision.HIGHEST:
        (a_hi, a_lo), (b_hi, b_lo) = _split_bfloat16(a), _split_bfloat16(b)
        if options.implementation == "emulated_fp32" or _explicit_emulation(fmt):
            a_hi, a_lo, b_hi, b_lo = (x.astype(jnp.float32) for x in (a_hi, a_lo, b_hi, b_lo))
        product = (_matmul32(a_lo, b_hi, precision) + _matmul32(a_hi, b_lo, precision)) + _matmul32(a_hi, b_hi,
                                                                                                     precision)
    elif fmt.product == "bfloat16_3x":
        product = _matmul32(a.astype(jnp.float32), b.astype(jnp.float32), precision, padding=options.bf16_padding)
    else:
        native = product_execution(compute, options=options) == "native"
        product = _rounded_product(a, b, fmt, native, precision, shared, arithmetic, sample_axis)
    product = product[..., 0, :] if vector_a else product
    return (product[..., 0] if vector_b else product).astype(out)


def _matmul32(a, b, precision, *, padding="none"):
    """`a @ b` accumulated in float32: an algorithm states its own accumulation."""
    if (padding == "gemm8" and precision is lax.DotAlgorithmPreset.BF16_BF16_F32_X3
            and default_device().platform == "gpu"
            and (a.shape[-2] == 1 or b.shape[-1] == 1)):
        # JAX 0.10.2's GPU GEMV expansion multiplies the split pieces in BF16,
        # rounding each product before FP32 reduction. Keep this a native GEMM,
        # then discard zero-padded output lanes; never replace it with emulation.
        m, n = a.shape[-2], b.shape[-1]
        a = jnp.pad(a, [(0, 0)] * (a.ndim - 2) + [(0, max(0, 8 - m)), (0, 0)])
        b = jnp.pad(b, [(0, 0)] * (b.ndim - 2) + [(0, 0), (0, max(0, 8 - n))])
        return jnp.matmul(a, b, precision=precision)[..., :m, :n]
    return jnp.matmul(a, b, precision=precision, preferred_element_type=jnp.float32)


def _rounded_product(a, b, fmt: Format, native: bool, precision, shared: str, arithmetic, sample_axis: int):
    """`a @ b` on operands rounded to `fmt`'s grid (below float32), each row of `a` and column of
    `b` optionally scaled into range by powers of two, the `shared` operand in configured residual terms
    on its own grid (`engine.precision.formats.shared_format`), as one product: see the module
    docstring. Natively each operand is held in its grid's dtype; emulated, both in one that holds
    them exactly (`_CARRIERS`)."""
    options = arithmetic.product
    fmt = replace(fmt, shared_terms=options.shared_terms,
                  shared_dtype=fmt.shared_dtype if options.shared_format == "e4m3" else None)
    terms = options.shared_terms
    uniforms = None
    if arithmetic.rounding == "stochastic":
        if arithmetic.key is None:
            raise ValueError("stochastic product rounding needs an explicit random key")
        operand = b if shared == "left" else a
        uniforms = _operand_uniforms(operand, arithmetic, sample_axis)
    held = shared_format(fmt)
    carrier = None
    if not native:
        # Select the carrier *before* quantization's final cast. On TPU, a
        # float32(float16(rounded)) round trip is narrowed back to an FP16 dot
        # by libtpu 0.0.42, losing the requested FP32 product accuracy. Keep
        # explicitly rounded grid values in FP32 throughout emulation instead.
        # This removes the observed narrowing pattern; actual TPU qualification
        # is still required. Native operand types and dispatch are unchanged.
        carrier = (jnp.float32 if _explicit_emulation(fmt) or options.implementation == "emulated_fp32"
                   or default_device().platform == "tpu"
                   else _CARRIERS.get(fmt.base, fmt.dtype))
    qa, ka = _terms(a, held if shared == "left" else fmt, axis=-1, count=terms if shared == "left" else 1,
                    uniforms=uniforms if shared == "right" else None, carrier=carrier,
                    scaling=options.scaling == "power_of_two")
    qb, kb = _terms(b, held if shared == "right" else fmt, axis=-2, count=terms if shared == "right" else 1,
                    uniforms=uniforms if shared == "left" else None, carrier=carrier,
                    scaling=options.scaling == "power_of_two")
    product = (_normalized_fp16_product(qa, qb, precision) if options.subnormals == "normalize_fp16"
               else _matmul32(qa, qb, precision))
    product = product * pow2(-ka, jnp.float32) * pow2(-kb, jnp.float32)
    parts = jnp.split(product, terms, axis=-2 if shared == "left" else -1)
    total = parts[-1]
    for part in reversed(parts[:-1]):
        total = total + part
    return total


def _terms(x, fmt: Format, axis: int, count: int, uniforms=None, *, carrier=None, scaling=True):
    """`x` as `count` terms of `fmt`, each rounded and scaled along `axis` by `_quantize`, the
    first `x`'s rounding and each next one the rounding of what the terms before leave, stacked
    along the other matrix axis; and their scale exponents, stacked alike. What a term leaves is
    exact in `x`'s dtype (float32 or float64), being a rounding error onto a coarser grid.
    `carrier` overrides only the final dtype, retaining `fmt`'s grid and scale rules."""
    terms, exponents, rest = [], [], x
    for i in range(count):
        term, k = _quantize(rest, fmt, axis, uniforms, carrier=carrier, scaling=scaling)
        terms.append(term)
        exponents.append(k)
        if i + 1 < count:
            taken = term.astype(rest.dtype) * pow2(-k, rest.dtype)
            rest = jnp.where(jnp.isfinite(taken), rest - taken, 0)
    stack = -1 if axis == -2 else -2
    return jnp.concatenate(terms, axis=stack), jnp.concatenate(exponents, axis=stack)


def _quantize(x, fmt: Format, axis: int, uniforms=None, *, carrier=None, scaling=True):
    """`x` scaled along `axis` by the powers of two `2^k` that bring each slice's largest finite
    magnitude into (max/2, max] of `fmt`, rounded to its grid and cast to it; and `k`. A format
    with float32's range (bfloat16, TensorFloat-32) needs no scale: `k` is 0, and scaling it to
    its top would overflow the float32 accumulator."""
    if not scaling or math.frexp(fmt.max)[1] >= _FLOAT32_TOP_EXPONENT:
        k = jnp.zeros(x.shape[:axis % x.ndim] + (1,) + x.shape[axis % x.ndim + 1:], dtype=jnp.int32)
    else:
        amax = jnp.max(jnp.where(jnp.isfinite(x), jnp.abs(x), 0), axis=axis, keepdims=True, initial=0)
        k = scale_exponent(amax, fmt.max)
    scaled = x * pow2(k, x.dtype)
    # round_to_grid implements the requested operand format in the input dtype.
    # In-range grid values fit exactly in the selected carrier. Emulation must
    # not visit a low dtype merely to widen back to FP32 afterwards.
    rounded = round_to_grid(scaled, fmt.mantissa_bits, fmt.min_exponent, uniforms)
    if carrier is not None:
        # A wider carrier must still obey the target encoding's overflow rules.
        # E4M3FN has no infinities (also when it carries E5M2's shared terms).
        overflow = (jnp.full_like(rounded, jnp.nan) if fmt.dtype == jnp.dtype(jnp.float8_e4m3fn)
                    else jnp.copysign(jnp.full_like(rounded, jnp.inf), rounded))
        rounded = jnp.where(jnp.abs(rounded) > fmt.max, overflow, rounded)
    return rounded.astype(fmt.dtype if carrier is None else carrier), k


def _normalized_fp16_product(a, b, precision):
    """Opt-in: one ordinary dot when possible; four native dots for subnormals.

    The two tests/reductions and conditional exist only in this selected strategy.
    Normalization preserves the quantized operands; it never flushes them to zero.
    Qualification remains device/compiler/shape specific.
    """
    def small(x):
        x = x.astype(jnp.float32)
        return (jnp.abs(x) < 2.**-14) & (x != 0)

    def corrected(operands):
        a, b = operands
        def split(x):
            x = x.astype(jnp.float32)
            mask = small(x)
            return (jnp.where(mask, 0, x).astype(jnp.float16),
                    jnp.where(mask, x * 2.**24, 0).astype(jnp.float16))
        ah, al = split(a)
        bh, bl = split(b)
        return (_matmul32(ah, bh, precision)
                + (_matmul32(al, bh, precision) + _matmul32(ah, bl, precision)) * 2.**-24
                + _matmul32(al, bl, precision) * 2.**-48)

    return lax.cond(jnp.any(small(a)) | jnp.any(small(b)), corrected,
                    lambda operands: _matmul32(*operands, precision), (a, b))


def _operand_uniforms(x, arithmetic, sample_axis):
    """One independent stream per global sample, stable under prefixes and batches.

    Non-sample coordinates have the operand's fixed shape. Randomness is independent of
    operand values; using their bits to seed rounding would introduce a new bias.
    """
    axis = sample_axis % x.ndim
    ids = arithmetic.sample_ids
    if ids is None:
        ids = jnp.arange(x.shape[axis], dtype=jnp.uint32)
    if ids.shape != (x.shape[axis],):
        raise ValueError(f"rounding sample_ids {ids.shape} must match operand sample axis {x.shape[axis]}")
    shape = x.shape[:axis] + x.shape[axis + 1:]
    keys = jax.vmap(lambda i: jax.random.fold_in(arithmetic.key, i))(ids)
    draws = jax.vmap(lambda k: jax.random.uniform(k, shape, dtype=x.dtype))(keys)
    return jnp.moveaxis(draws, 0, axis)


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
