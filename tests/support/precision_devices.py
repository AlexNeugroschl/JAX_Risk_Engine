"""
Whether a device's matrix products give each compute format's numbers (F-07;
docs/planning/details/precision.md §8.3): every arithmetic a stage accepts (each compute format
with each accumulate format it allows) multiplied on the device, against the same product
emulated on the CPU, and against the exact product. Native dispatch follows device capability;
run this check to qualify the actual compiler/device, and after a JAX or driver upgrade:

    python -m tests.support.precision_devices [--platform gpu]

A format the device runs natively (`product_execution`) must equal the CPU's emulation up to the
order of the sums in the accumulate format, or, for a product format (TensorFloat-32, bfloat16 passes), up to the
device's own rounding of the operands (a few of the format's unit roundoffs); an emulated one
must equal it up to the sums' order. On an RTX 5060 (2026-10-10, jax 0.10.2) this check found
XLA's FP8 product 70% off for some shapes and its bfloat16 passes (`BF16_BF16_F32_X3`) a single
pass's error for some shapes (XLA's faults, not the card's: its Triton GEMM emitter for FP8, and a
matrix-vector product compiled without the algorithm; I-97). These are local compiler
failures, not reasons to emulate on every GPU. Run the native default and an explicit emitter
override in separate processes:

    python -m tests.support.precision_devices --platform gpu --gemm default
    python -m tests.support.precision_devices --platform gpu --gemm cublas

The local FP8 workaround is `--gemm cublas`; the local bfloat16-passes fusion workaround is
`JAX_RISK_EMULATE_PRODUCTS=bfloat16_3x`. Both are explicit test/process settings. A default
compiler failure remains a failed qualification, never an automatic passing emulation.
TensorFloat-32 is within 4.5e-4 of emulation on this card (the device rounds operands within
its unit roundoff). The same card summed float32
operands asked for a float64 result in float32 (7.6e-7 off, 2026-10-11): the engine now
multiplies them as float64 values on float32's grid.

Exit status 1 when a format fails.
"""
import argparse
import itertools
import os
import re
import sys

from engine.api.worker import deterministic_kernels_environment  # noqa: E402  (no JAX)

# A process of its own on a GPU shared with others, run as the engine worker runs: no
# preallocation, deterministic kernels; its GEMM emitter the environment's (see the docstring).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.update(deterministic_kernels_environment(os.environ))

from pathlib import Path

# Include GEMV, odd dimensions, a shard boundary and batched products. Engine HLO capture
# below supplements these adversarial synthetic cases with the actual compiled workloads.
SHAPES = ((512, 64, 8), (512, 64, 1), (64, 64, 64), (16, 32, 16),
          (33, 13, 7), (1, 13, 7), (2, 33, 13, 7))


def fp8_lowering(hlo):
    """Recognize optimized FP8 products, independently of their numerical qualification.

    Triton evidence requires the *called computation* to contain FP8 operands and an FP32
    dot. An unrelated FP8 conversion or another Triton fusion is insufficient. This is
    lowering evidence; it does not prove the generated kernel's matrix instructions.
    """
    if 'custom_call_target="__cublas$lt$matmul$f8"' in hlo:
        return "cublas"
    for name, body in re.findall(r"^%([^\s(]+)[^\n]*\{\n(.*?)^\}", hlo, re.M | re.S):
        operands = set(re.findall(r"^\s*%(\S+)\s*=\s*f8(?:e4m3fn|e5m2)\[", body, re.M))
        dots = re.findall(r"=\s*f32\[[^\n]*?\s+dot\(%([^,\s]+),\s*%([^)\s]+)\)", body)
        if any(a in operands and b in operands for a, b in dots) and re.search(
                rf"fusion\([^\n]*calls=%{re.escape(name)}(?:,|\s)[^\n]*__triton", hlo):
            return "triton"
    return None


def normalized_error(actual, expected, scale):
    """Absolute-product normalization, including exact zero products and non-finite failures."""
    import numpy as np
    if not (np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))):
        return float("inf")
    difference = np.abs(actual - expected)
    ratio = np.full(difference.shape, np.inf)
    np.divide(difference, scale, out=ratio, where=scale > 0)
    ratio[(scale == 0) & (difference == 0)] = 0.
    return float(np.max(ratio))


def qualify(device, formats=None, out=None, *, roundings=("nearest", "stochastic"),
            patterns=("wide", "zero", "cancellation"), shapes=SHAPES):
    """Per-operation numerical and optimized-lowering evidence; never infer a hardware pass
    from device capability alone. Artifacts describe this runtime and device only."""
    import dataclasses
    import jax
    import numpy as np
    from engine.precision import FORMAT_NAMES, FORMATS, SHARED, Arithmetic, matmul, product_execution
    from engine.precision.arithmetic import arithmetic_refusal
    from tests.support.precision_timing import execution_environment

    formats = FORMAT_NAMES if formats is None else formats
    arithmetics = [Arithmetic(name, accumulate) for name in formats for accumulate in ("float32", "float64")
                   if arithmetic_refusal(FORMATS[name], FORMATS[accumulate]) is None]
    cpu = jax.devices("cpu")[0]
    rows = []
    metadata = execution_environment()
    if out is not None:
        from demos.benchmark_reporting import atomic_json
        atomic_json(Path(out) / "qualification.json", {"schema_version": 1, "complete": False,
                    "passes": False, "device": device.device_kind, "environment": metadata, "operations": rows})
    for arithmetic in arithmetics:
        fmt, execution = FORMATS[arithmetic.compute], product_execution(arithmetic.compute, device)
        policies = roundings if fmt.below_float32 and fmt.product != "bfloat16_3x" else ("nearest",)
        for shape, shared, rounding, pattern in itertools.product(shapes, SHARED, policies, patterns):
            rng = np.random.default_rng(7)
            *batch, m, k, n = shape
            a = rng.standard_normal((*batch, m, k)) * np.logspace(-3, 3, m)[:, None]
            b = rng.standard_normal((*batch, k, n)) * np.logspace(3, -2, n)[None, :]
            if pattern == "zero":
                a.fill(0.)
            elif pattern == "cancellation":
                a[..., 1::2] = a[..., :2 * (k // 2):2]
                b[..., 1::2, :] = -b[..., :2 * (k // 2):2, :]
            key = jax.random.PRNGKey(7) if rounding == "stochastic" else None
            policy = dataclasses.replace(arithmetic, rounding=rounding, key=key)
            program = jax.jit(lambda x, y, p: matmul(x, y, p, shared))
            with jax.default_device(device):
                args = (jax.device_put(a, device), jax.device_put(b, device), policy)
                compiled = program.lower(*args).compile()
                actual = np.asarray(compiled(*args), np.float64)
                hlo = compiled.as_text()
            with jax.default_device(cpu):
                expected = np.asarray(program(a, b, policy), np.float64)
            scale = np.abs(a) @ np.abs(b)
            error = normalized_error(actual, expected, scale)
            tolerance = 64 * np.finfo(arithmetic.accumulate).eps / 2
            if fmt.product != "dtype" and execution == "native":
                tolerance = 4 * 2. ** -(fmt.mantissa_bits + 1)
            lowering = fp8_lowering(hlo)
            artifact = f"operation-{len(rows):04d}.hlo.txt"
            if out is not None:
                Path(out, artifact).write_text(hlo, encoding="utf-8")
            rows.append({"format": fmt.name, "accumulate": arithmetic.accumulate, "rounding": rounding,
                         "shape": list(shape), "shared": shared, "pattern": pattern, "dispatch": execution,
                         "observed_fp8_gemm": lowering is not None, "fp8_lowering": lowering,
                         "hlo": artifact if out is not None else None,
                         "device_emulation_error": error if np.isfinite(error) else None,
                         "device_exact_error": (lambda e: e if np.isfinite(e) else None)(
                             normalized_error(actual, a @ b, scale)),
                         "tolerance": float(tolerance), "passes": bool(np.isfinite(error) and error <= tolerance)})
        print(f"checked {arithmetic.compute}/{arithmetic.accumulate}", flush=True)
        if out is not None:
            atomic_json(Path(out) / "qualification.json", {"schema_version": 1, "complete": False,
                        "passes": False, "device": device.device_kind, "environment": metadata, "operations": rows})
    return {"schema_version": 1, "device": device.device_kind, "platform": device.platform,
            "compute_capability": getattr(device, "compute_capability", None),
            "environment": metadata, "operations": rows, "complete": True,
            "passes": all(r["passes"] for r in rows),
            "native_scope": "optimized FP8 cuBLAS calls or Triton FP8 dots observed per operation; matrix instructions require device traces"}


def check(device) -> int:
    return sum(not row["passes"] for row in qualify(device)["operations"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.support.precision_devices",
                                     description="Check every compute format's product on a device.")
    parser.add_argument("--platform", default=None, help="gpu, tpu or cpu (default: JAX's default backend)")
    parser.add_argument("--formats", nargs="+", help="compute formats to check (default: every accepted format)")
    parser.add_argument("--out", type=Path, help="new output directory for JSON, optimized HLO and engine traces")
    parser.add_argument("--gemm", choices=("default", "cublas"),
                        help="explicit emitter override; omitted keeps the runtime/inherited settings")
    parser.add_argument("--expected-device", help="required device-kind substring, e.g. H100")
    parser.add_argument("--require-native-fp8", action="store_true", help="fail unless each selected FP8 format has an observed FP8 GEMM")
    parser.add_argument("--engine-workloads", nargs="*", choices=("comparison_single", "comparison_portfolio"), default=[])
    args = parser.parse_args(argv)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        if any(args.out.iterdir()):
            parser.error("output directory must be empty")
        # Read before JAX opens the device; optimized engine HLO is retained beside traces.
        if args.engine_workloads:
            os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                + f" --xla_dump_to={args.out.resolve() / 'engine-hlo'} --xla_dump_hlo_as_text=true").strip()
    elif args.engine_workloads:
        parser.error("--engine-workloads requires --out")
    from engine.api.worker import gemm_environment
    os.environ.update(gemm_environment(args.gemm, os.environ))
    import jax
    from demos.benchmark_reporting import atomic_json
    from engine.precision import FORMATS
    if args.formats and any(f not in FORMATS for f in args.formats):
        parser.error("unknown compute format")
    device = jax.devices(args.platform)[0] if args.platform else jax.devices()[0]
    if jax.config.jax_disable_jit:
        parser.error("qualification requires JIT execution")
    if args.expected_device and args.expected_device.lower() not in device.device_kind.lower():
        parser.error(f"expected {args.expected_device}, got {device.device_kind}")
    result = qualify(device, args.formats, args.out)
    if args.require_native_fp8:
        requested = {r["format"] for r in result["operations"] if r["format"].startswith("float8")}
        observed = {r["format"] for r in result["operations"] if r["observed_fp8_gemm"]}
        result["native_requirement_passes"] = bool(requested) and requested <= observed
        result["passes"] &= result["native_requirement_passes"]
    if args.engine_workloads:
        from tests.support.precision_timing import _child
        result["engine_workloads"] = []
        with jax.default_device(device), jax.profiler.trace(str(args.out / "trace")):
            for campaign in args.engine_workloads:
                from tests.support.precision_campaigns import COMPARISON_CAMPAIGNS
                labels = {c.label for c in COMPARISON_CAMPAIGNS[campaign]().cases}
                for name in args.formats or ("float16", "float8_e4m3fn", "float8_e5m2"):
                    label = "float64" if name == "float64" else f"pricing:{name}"
                    if name != "float64" and label not in labels:
                        label = f"pricing:compute:{name}"
                    timing = _child(campaign, label,
                                    128, (1, 2, 3), None,
                                    snapshot_directory=str(args.out / campaign / name), cpu_inputs=True)
                    result["engine_workloads"].append({"campaign": campaign, "format": name, "timing": timing})
                    observed = timing["metadata"]["execution_devices"]
                    result["passes"] &= not timing["warm_compiles"] and bool(observed) and all(
                        d.startswith(device.platform + ":") for d in observed)
    if args.out is not None:
        atomic_json(args.out / "qualification.json", result)
    print(f"{len(result['operations'])} operations; {'PASS' if result['passes'] else 'FAILED'}", flush=True)
    return 0 if result["passes"] else 1


if __name__ == "__main__":
    sys.exit(main())
