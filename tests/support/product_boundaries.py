"""Preserve the boundaries missing from the original cloud FP8 investigation.

    python -m tests.support.product_boundaries --platform gpu --format float8_e4m3fn \
        --shape 33 13 7 --shared right --out artifacts/fp8-boundaries

Run default/cuBLAS and explicit-FP8 algorithms in separate processes. Staged
captures can affect fusion, so the unchanged fused engine product is a separate
control. Numerical passes do not qualify native instructions. This tool creates
no cloud resources and changes no engine policy or acceptance record.
"""
import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np


def operand_reference(x, fmt, axis, count, uniforms=None, *, scaling=True):
    """Independent host implementation of the operand grid, scales and residuals.

    NumPy arithmetic runs in the original input dtype. No engine grid helper or
    JAX calculation participates in this reference.
    """
    original_dtype = x.dtype
    rest = x.copy()
    terms, exponents = [], []
    for _ in range(count):
        amax = np.max(np.where(np.isfinite(rest), np.abs(rest), 0), axis=axis, keepdims=True, initial=0)
        if not scaling or np.frexp(fmt.max)[1] >= 128:
            k = np.zeros(amax.shape, dtype=np.int32)
        else:
            k = np.frexp(fmt.max)[1] - np.frexp(amax)[1]
            comparison = rest.dtype.type(2.) ** np.clip(k, -126, 126)
            k = np.clip(k - (amax * comparison > fmt.max), -126, 126)
        scaled = rest * np.exp2(k).astype(original_dtype)
        exponent = np.frexp(scaled)[1]
        spacing = np.maximum(np.maximum(exponent - 1, fmt.min_exponent) - fmt.mantissa_bits,
                             np.finfo(original_dtype).minexp)
        step = np.exp2(spacing).astype(original_dtype)
        units = scaled / step
        if uniforms is None:
            whole = np.rint(units)
        else:
            lower = np.floor(units)
            whole = np.where(uniforms < units - lower, lower + 1, lower)
        rounded = whole * step
        terms.append(rounded.astype(np.float64))
        exponents.append(k)
        taken = rounded * np.exp2(-k).astype(original_dtype)
        rest = np.where(np.isfinite(taken), rest - taken, 0)
    stack = -1 if axis == -2 else -2
    return np.concatenate(terms, axis=stack), np.concatenate(exponents, axis=stack)


def product_check(actual, expected, scale, contractions):
    """FP32 accumulation bound on identical operands, including cancellation.

    This is an arithmetic check, not a financial error bound. Exact zeros cannot
    hide an arbitrary nonzero output. Inputs in this diagnostic are normal-range.
    """
    from tests.support.precision_devices import normalized_error
    error = normalized_error(actual, expected, scale)
    nu = max(1, contractions) * 2. ** -24
    tolerance = nu / (1. - nu) if nu < 1 else float("inf")
    return {"passes": bool(np.isfinite(error) and error <= tolerance),
            "normalized_error": error if np.isfinite(error) else None, "tolerance": tolerance}


def diagnose(device, a, b, arithmetic, shared, out, *, algorithm="default"):
    """Capture quantization, raw GEMM, rescaling and an unchanged fused control."""
    import jax
    import jax.numpy as jnp
    from engine.precision import products
    from engine.precision.formats import format_of, shared_format
    from tests.support.precision_devices import fp8_lowering

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fmt = format_of(arithmetic.compute)
    options = arithmetic.product
    fmt = replace(fmt, shared_terms=options.shared_terms,
                  shared_dtype=fmt.shared_dtype if options.shared_format == "e4m3" else None)
    if fmt.product != "dtype" or not fmt.below_float32 or a.ndim != 2 or b.ndim != 2:
        raise ValueError("boundary diagnostics require two matrices and an FP16/BF16/FP8 array format")
    engine = Path(__file__).resolve().parents[2] / "engine"
    digest = hashlib.sha256()
    for path in sorted(engine.rglob("*.py")):
        digest.update(path.relative_to(engine).as_posix().encode() + b"\0" + path.read_bytes().replace(b"\r\n", b"\n"))
    metadata = {"complete": False, "passes": False, "format": fmt.name, "shared": shared,
                "rounding": arithmetic.rounding, "product_options": options.description(arithmetic.compute), "algorithm": algorithm, "device": device.device_kind,
                "platform": device.platform, "jax": jax.__version__, "xla_flags": os.environ.get("XLA_FLAGS", ""),
                "jaxlib": importlib.metadata.version("jaxlib"), "numpy": np.__version__,
                "compute_capability": getattr(device, "compute_capability", None),
                "product_emulation": os.environ.get(products.EMULATE_PRODUCTS, ""),
                "engine_source_sha256": digest.hexdigest(),
                "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                "native_instruction_qualification": "not_established",
                "scope": "staged boundary captures plus separately compiled unchanged fused engine product",
                "checks": {}}
    buffers = {"input/a": a, "input/b": b}
    if arithmetic.key is not None:
        buffers["rounding_key"] = np.asarray(jax.random.key_data(arithmetic.key))

    def save():
        np.savez_compressed(out / "buffers.npz", **{k: np.asarray(v, np.float64) for k, v in buffers.items()})
        (out / "result.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")

    def quantize(x, y, policy):
        uniforms = (products._operand_uniforms(y if shared == "left" else x, policy, 0)
                    if policy.rounding == "stochastic" else None)
        native = products.product_execution(fmt.name, options=options) == "native"
        carrier = None if native else (jnp.float32 if device.platform == "tpu" or products._explicit_emulation(fmt) or options.implementation == "emulated_fp32"
                                      else products._CARRIERS.get(fmt.base, fmt.dtype))
        held = shared_format(fmt)
        qa, ka = products._terms(x, held if shared == "left" else fmt, -1,
                                fmt.shared_terms if shared == "left" else 1,
                                uniforms if shared == "right" else None, carrier=carrier, scaling=options.scaling == "power_of_two")
        qb, kb = products._terms(y, held if shared == "right" else fmt, -2,
                                fmt.shared_terms if shared == "right" else 1,
                                uniforms if shared == "left" else None, carrier=carrier, scaling=options.scaling == "power_of_two")
        return qa, qb, ka, kb, uniforms

    save()
    try:
        with jax.default_device(device):
            args = (jax.device_put(a, device), jax.device_put(b, device), arithmetic)
            quantized = jax.jit(quantize)(*args)
            qa, qb, ka, kb, uniforms = quantized
            for name, value in zip(("qa", "qb", "ka", "kb", "uniforms"), quantized):
                if value is not None:
                    buffers[name] = np.asarray(value, np.float64)
            metadata["operand_dtypes"] = [str(qa.dtype), str(qb.dtype)]
            for name, x, axis in (("a", a, -1), ("b", b, -2)):
                is_shared = shared == {"a": "left", "b": "right"}[name]
                expected_q, expected_k = operand_reference(x, shared_format(fmt) if is_shared else fmt,
                    axis, fmt.shared_terms if is_shared else 1, None if is_shared else buffers.get("uniforms"),
                    scaling=options.scaling == "power_of_two")
                buffers["reference/q" + name] = expected_q
                metadata["checks"]["quantization/" + name] = {"passes": bool(
                    np.array_equal(buffers["q" + name], expected_q) and
                    np.array_equal(buffers["k" + name], expected_k))}
            # Preserve operands before any suspect GEMM compiles or fails.
            save()
            precision = (products.product_precision(fmt.name, options=options) if algorithm == "default"
                         else jax.lax.DotAlgorithmPreset.ANY_F8_ANY_F8_F32)
            dot = (products._normalized_fp16_product if options.subnormals == "normalize_fp16"
                   else products._matmul32)
            raw_fn = jax.jit(lambda x, y: dot(x, y, precision) if algorithm == "default"
                            else jnp.matmul(x, y, precision=precision, preferred_element_type=jnp.float32))
            raw_program = raw_fn.lower(qa, qb).compile()
            raw = raw_program(qa, qb)
            raw_hlo = raw_program.as_text()
            (out / "raw.hlo.txt").write_text(raw_hlo, encoding="utf-8")
            metadata["raw_fp8_lowering"] = fp8_lowering(raw_hlo)
            buffers["raw"] = np.asarray(raw, np.float64)
            expected_raw = buffers["qa"] @ buffers["qb"]
            scale_raw = np.abs(buffers["qa"]) @ np.abs(buffers["qb"])
            buffers["reference/raw"] = expected_raw
            metadata["checks"]["raw_product"] = product_check(buffers["raw"], expected_raw, scale_raw, a.shape[-1])

            def combine(product, ka, kb):
                product = product * products.pow2(-ka, jnp.float32) * products.pow2(-kb, jnp.float32)
                parts = jnp.split(product, fmt.shared_terms, axis=-2 if shared == "left" else -1)
                total = parts[-1]
                for part in reversed(parts[:-1]):
                    total = total + part
                return total

            buffers["staged"] = np.asarray(jax.jit(combine)(raw, ka, kb), np.float64)
            unscaled = expected_raw * np.exp2(-buffers["ka"]) * np.exp2(-buffers["kb"])
            scales = scale_raw * np.exp2(-buffers["ka"]) * np.exp2(-buffers["kb"])
            axis = -2 if shared == "left" else -1
            expected = sum(np.split(unscaled, fmt.shared_terms, axis=axis))
            scale = sum(np.split(scales, fmt.shared_terms, axis=axis))
            buffers["reference/final"] = expected
            bound_steps = a.shape[-1] + fmt.shared_terms + 2
            metadata["checks"]["staged_result"] = product_check(buffers["staged"], expected, scale, bound_steps)
            save()
            # The fused control retains the requested policy; --algorithm changes only the staged diagnostic.
            fused = jax.jit(lambda x, y, p: products.matmul(x, y, p, shared)).lower(*args).compile()
            buffers["fused"] = np.asarray(fused(*args), np.float64)
            (out / "fused.hlo.txt").write_text(fused.as_text(), encoding="utf-8")
            metadata["checks"]["unchanged_fused_result"] = product_check(
                buffers["fused"], expected, scale, bound_steps)
            metadata["complete"] = True
            metadata["passes"] = all(row["passes"] for row in metadata["checks"].values())
    except Exception as error:
        metadata["error"] = f"{type(error).__name__}: {error}"
    save()
    return metadata


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--platform", choices=("cpu", "gpu", "tpu"), required=True)
    formats = ("float16", "bfloat16", "float8_e4m3fn", "float8_e5m2")
    p.add_argument("--format", choices=formats + tuple(name + "_1x" for name in formats), required=True)
    p.add_argument("--shape", type=int, nargs=3, default=(33, 13, 7), metavar=("M", "K", "N"))
    p.add_argument("--shared", choices=("left", "right"), default="right")
    p.add_argument("--profile", choices=("basic", "compensated"), default="basic")
    p.add_argument("--rounding", choices=("nearest", "stochastic"), default="nearest")
    p.add_argument("--gemm", choices=("default", "cublas"), default="default")
    p.add_argument("--algorithm", choices=("default", "full-fp8"), default="default")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if min(args.shape) <= 0:
        p.error("matrix dimensions must be positive")
    if args.algorithm == "full-fp8" and (args.platform != "gpu" or not args.format.startswith("float8")):
        p.error("full-fp8 is a separate experimental GPU FP8 request")
    os.environ["JAX_PLATFORMS"] = {"cpu": "cpu", "gpu": "cuda,cpu", "tpu": "tpu,cpu"}[args.platform]
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from engine.api.worker import deterministic_kernels_environment, gemm_environment
    os.environ.update(deterministic_kernels_environment(os.environ))
    if args.platform == "gpu":
        os.environ.update(gemm_environment(args.gemm, os.environ))
    import jax
    from engine.precision import Arithmetic, ProductOptions
    device = jax.devices(args.platform)[0]
    m, k, n = args.shape
    rng = np.random.default_rng(7)
    dtype = np.float32 if args.platform == "tpu" else np.float64
    a = (rng.normal(size=(m, k)) * np.logspace(-3, 3, m)[:, None]).astype(dtype)
    b = (rng.normal(size=(k, n)) * np.logspace(3, -2, n)[None, :]).astype(dtype)
    with jax.default_device(device):
        policy = Arithmetic(args.format, "float32", rounding=args.rounding,
            key=jax.random.PRNGKey(7) if args.rounding == "stochastic" else None, execution="allow_emulation",
            product=ProductOptions.compensated(args.format) if args.profile == "compensated" else ProductOptions())
    result = diagnose(device, a, b, policy, args.shared, args.out, algorithm=args.algorithm)
    print(json.dumps(result, indent=2))
    return 0 if result["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
