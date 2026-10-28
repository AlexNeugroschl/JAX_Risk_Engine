"""Bounded arithmetic survey; creates no cloud resources.

Run on an already provisioned single-chip TPU::

    timeout --signal=TERM --kill-after=30s 1800s python -m \
        tests.support.tpu_precision_survey --platform tpu --out evidence

FP32 inputs and accumulators deliberately avoid the rejected TPU FP64 path.
CPU execution validates this diagnostic, not TPU correctness. JSON checkpoints,
inputs/results, and optimized HLO survive a failed case or an interrupted survey.
"""
import argparse
from dataclasses import asdict, replace
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np


SHAPES = ((32, 32, 8), (32, 32, 1), (33, 13, 7), (2, 9, 13, 7))


def reference(a, b, arithmetic, shared, uniforms):
    """Independent NumPy grid, residual reconstruction, and FP64 host sums.

    FP64 is used only by NumPy on the host as an oracle, never on the TPU.
    The X3 algorithm drops the low-low product by definition.
    """
    from engine.precision import FORMATS
    from engine.precision.formats import shared_format
    from tests.support.product_boundaries import operand_reference

    fmt = FORMATS[arithmetic.compute]
    options = arithmetic.product
    if arithmetic.compute == "float32":
        return a.astype(np.float64) @ b.astype(np.float64), np.abs(a.astype(np.float64)) @ np.abs(b.astype(np.float64))
    if fmt.product == "bfloat16_3x":
        base = FORMATS["bfloat16"]
        def split(x):
            high, _ = operand_reference(x, base, -1, 1, scaling=False)
            low, _ = operand_reference(x - high.astype(np.float32), base, -1, 1, scaling=False)
            return high, low
        ah, al = split(a)
        bh, bl = split(b)
        expected = (al @ bh + ah @ bl) + ah @ bh
        scale = np.abs(al) @ np.abs(bh) + np.abs(ah) @ np.abs(bl) + np.abs(ah) @ np.abs(bh)
        return expected, scale
    fmt = replace(fmt, shared_terms=options.shared_terms,
                  shared_dtype=fmt.shared_dtype if options.shared_format == "e4m3" else None)
    qa, ka = operand_reference(a, shared_format(fmt) if shared == "left" else fmt, -1,
        options.shared_terms if shared == "left" else 1, uniforms if shared == "right" else None,
        scaling=options.scaling == "power_of_two")
    qb, kb = operand_reference(b, shared_format(fmt) if shared == "right" else fmt, -2,
        options.shared_terms if shared == "right" else 1, uniforms if shared == "left" else None,
        scaling=options.scaling == "power_of_two")
    factor = np.exp2(-ka.astype(np.float64)) * np.exp2(-kb.astype(np.float64))
    raw = (qa @ qb) * factor
    scale = (np.abs(qa) @ np.abs(qb)) * factor
    axis = -2 if shared == "left" else -1
    return sum(np.split(raw, options.shared_terms, axis=axis)), sum(np.split(scale, options.shared_terms, axis=axis))


def inputs(shape, pattern, fmt):
    *batch, m, k, n = shape
    rng = np.random.default_rng(7)
    a = rng.uniform(-2, 2, (*batch, m, k)).astype(np.float32)
    b = rng.uniform(-2, 2, (*batch, k, n)).astype(np.float32)
    if pattern == "zero":
        a.fill(0)
    elif pattern == "cancellation":
        a[..., 1::2] = a[..., :2 * (k // 2):2]
        b[..., 1::2, :] = -b[..., :2 * (k // 2):2, :]
    elif pattern == "small_grid":
        # FP16/FP8 subnormal grids, including halfway points and signed zeros.
        # BF16/TF32's very small subnormals require a separate underflow contract.
        exponent = max(fmt.min_exponent - fmt.mantissa_bits, -24)
        a = (np.rint(a * 4) + np.float32(.5)) * np.float32(2. ** exponent)
    return a, b


def survey(device, out, *, formats, profiles, shapes=SHAPES, budget_seconds=1500):
    import jax
    from engine.precision import Arithmetic, FORMATS, ProductOptions, products
    from engine.precision.execution import validate_product
    from tests.support.precision_devices import normalized_error
    from tests.support.product_boundaries import diagnose

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("output directory must be empty")
    started = time.monotonic()
    result = dict(complete=False, platform=device.platform, device=device.device_kind,
        jax=jax.__version__, jaxlib=importlib.metadata.version("jaxlib"), numpy=np.__version__,
        input_dtype="float32", accumulate="float32", profiles=list(profiles),
        scope="synthetic arithmetic only; no portfolio or instruction-level qualification",
        budget_seconds=budget_seconds, xla_flags=os.environ.get("XLA_FLAGS", ""),
        guards=[], native_requests=[], operations=[], boundaries=[],
        native_instruction_qualification="not_established")

    def save():
        tmp = out / "summary.tmp"
        tmp.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        # Windows readers can briefly hold the destination without delete sharing.
        for attempt in range(10):
            try:
                tmp.replace(out / "summary.json")
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(.05)

    def remaining():
        if time.monotonic() - started >= budget_seconds:
            result["stop_reason"] = "workload_time_limit"
            save()
            return False
        return True

    with jax.default_device(device):
        # Refusals are expected outcomes, not pretend FP64 arithmetic passes.
        if device.platform == "tpu":
            for compute, accumulate in (("float64", "float64"), ("float32", "float64")):
                try:
                    validate_product(compute, accumulate, "allow_emulation", device=device)
                    row = dict(compute=compute, accumulate=accumulate, refused=False)
                except ValueError as error:
                    row = dict(compute=compute, accumulate=accumulate, refused=True, reason=str(error))
                result["guards"].append(row)
            for name in formats:
                if name == "float64":
                    continue
                expected_native = products.product_execution(name, device) == "native"
                try:
                    validate_product(name, "float32", "native", device=device)
                    row = dict(format=name, accepted=True, passes=expected_native)
                except ValueError as error:
                    row = dict(format=name, accepted=False, passes=not expected_native, reason=str(error))
                result["native_requests"].append(row)
        save()
        for name, profile in itertools.product(formats, profiles):
            if name == "float64":
                continue
            fmt = FORMATS[name]
            # These two profiles are identical for the wide and X3 controls.
            if profile == "compensated" and name in ("float32", "bfloat16_3x"):
                continue
            options = ProductOptions.compensated(name) if profile == "compensated" else ProductOptions()
            dispatch = products.product_execution(name, device, options=options)
            roundings = ("nearest", "stochastic") if fmt.below_float32 and fmt.product != "bfloat16_3x" else ("nearest",)
            for shape, shared, rounding in itertools.product(shapes, ("left", "right"), roundings):
                if not remaining():
                    return result
                policy = Arithmetic(name, "float32", execution="allow_emulation", product=options,
                    rounding=rounding, key=jax.random.PRNGKey(7) if rounding == "stochastic" else None)
                program = jax.jit(lambda x, y, p: products.matmul(x, y, p, shared))
                compiled = None
                # Reuse each executable across value patterns and exact repeats.
                patterns = ["normal", "zero", "cancellation"]
                if name.startswith(("float16", "float8")):
                    patterns.append("small_grid")
                for pattern in patterns:
                    if not remaining():
                        return result
                    index = len(result["operations"])
                    row = dict(format=name, profile=profile, shape=list(shape), shared=shared,
                        rounding=rounding, pattern=pattern, dispatch=dispatch,
                        product_options=asdict(options), passes=False)
                    a, b = inputs(shape, pattern, fmt)
                    buffers = dict(a=a, b=b)
                    try:
                        args = (jax.device_put(a, device), jax.device_put(b, device), policy)
                        if compiled is None:
                            compiled = program.lower(*args).compile()
                            hlo_path = f"operation-{index:04d}.hlo.txt"
                            hlo = compiled.as_text()
                            (out / hlo_path).write_text(hlo, encoding="utf-8")
                        actual = np.asarray(compiled(*args), np.float64)
                        again = np.asarray(compiled(*args), np.float64)
                        uniforms = None
                        if rounding == "stochastic":
                            operand = args[1] if shared == "left" else args[0]
                            uniforms = np.asarray(jax.jit(lambda x, p: products._operand_uniforms(x, p, 0))(operand, policy))
                            buffers["uniforms"] = uniforms
                        expected, scale = reference(a, b, policy, shared, uniforms)
                        error = normalized_error(actual, expected, scale)
                        # Keep the pre-existing, FP32 accumulator arithmetic tolerance.
                        # This compares to quantized operands, not unrounded inputs.
                        tolerance = 64 * np.finfo(np.float32).eps / 2
                        if fmt.product == "bfloat16_3x":
                            tolerance = 4 * 2. ** -(fmt.mantissa_bits + 1)
                        repeatable = np.array_equal(actual, again)
                        row.update(normalized_error=error if np.isfinite(error) else None,
                            tolerance=float(tolerance), repeatable=repeatable, hlo=hlo_path,
                            passes=bool(np.isfinite(error) and error <= tolerance and repeatable),
                            arithmetic_bound="existing device campaign tolerance")
                        buffers.update(actual=actual, repeated=again, expected=expected, scale=scale)
                    except Exception as error:
                        row["error"] = f"{type(error).__name__}: {error}"
                    row["buffers"] = f"operation-{index:04d}.npz"
                    np.savez_compressed(out / row["buffers"], **buffers)
                    result["operations"].append(row)
                    save()
            print(f"completed {name} {profile}", flush=True)
            # One staged matrix probe per rounding/profile for array formats.
            # Compensated probes stress dynamic range; basic probes stay in range.
            if fmt.product == "dtype" and fmt.below_float32:
                for rounding in roundings:
                    if not remaining():
                        return result
                    shape = (33, 13, 7)
                    a, b = inputs(shape, "normal", fmt)
                    if profile == "compensated":
                        a *= np.logspace(-3, 3, shape[0], dtype=np.float32)[:, None]
                        b *= np.logspace(3, -2, shape[2], dtype=np.float32)[None, :]
                    policy = Arithmetic(name, "float32", execution="allow_emulation", product=options,
                        rounding=rounding, key=jax.random.PRNGKey(7) if rounding == "stochastic" else None)
                    folder = f"boundary-{name}-{profile}-{rounding}"
                    evidence = diagnose(device, a, b, policy, "right", out / folder)
                    result["boundaries"].append(dict(format=name, profile=profile, rounding=rounding,
                        directory=folder, complete=evidence["complete"], passes=evidence["passes"],
                        checks=evidence["checks"], error=evidence.get("error")))
                    save()
    result["complete"] = True
    result["passes"] = (all(row["passes"] for row in result["operations"])
        and all(row["passes"] for row in result["boundaries"])
        and all(row["refused"] for row in result["guards"])
        and all(row["passes"] for row in result["native_requests"]))
    result["elapsed_seconds"] = time.monotonic() - started
    save()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("cpu", "tpu"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--formats", nargs="+")
    parser.add_argument("--profiles", nargs="+", choices=("basic", "compensated"), default=("basic", "compensated"))
    parser.add_argument("--budget-seconds", type=int, default=1500)
    parser.add_argument("--smoke", action="store_true", help="one small shape; diagnostic validation only")
    args = parser.parse_args(argv)
    if args.budget_seconds < 1:
        parser.error("budget must be positive")
    os.environ["JAX_PLATFORMS"] = "cpu" if args.platform == "cpu" else "tpu,cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    from engine.precision import FORMAT_NAMES
    formats = args.formats or FORMAT_NAMES
    if any(name not in FORMAT_NAMES for name in formats):
        parser.error("unknown format")
    device = jax.devices(args.platform)[0]
    if jax.config.jax_disable_jit:
        parser.error("survey requires JIT execution")
    if os.environ.get("JAX_RISK_EMULATE_PRODUCTS", "").strip():
        parser.error("unset JAX_RISK_EMULATE_PRODUCTS; this survey checks default TPU dispatch")
    if args.platform == "tpu" and "v6" not in device.device_kind.lower():
        parser.error(f"expected the single-chip v6e test resource, got {device.device_kind}")
    if args.platform == "tpu" and jax.device_count() != 1:
        parser.error("this bounded survey requires exactly one TPU device")
    result = survey(device, args.out, formats=formats, profiles=args.profiles,
        shapes=((9, 13, 7),) if args.smoke else SHAPES, budget_seconds=args.budget_seconds)
    print(json.dumps(dict(complete=result["complete"], operations=len(result["operations"]),
        passed=sum(row["passes"] for row in result["operations"]),
        boundaries=len(result["boundaries"]), passes=result.get("passes", False))))
    return 0 if result.get("passes", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
