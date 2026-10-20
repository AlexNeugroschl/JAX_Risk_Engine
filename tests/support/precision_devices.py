"""
Whether a device's matrix products give each compute format's numbers (F-07;
docs/planning/details/precision.md §8.3): every arithmetic a stage accepts (each compute format
with each accumulate format it allows) multiplied on the device, against the same product
emulated on the CPU, and against the exact product. Run it on each device
before `engine.precision.products` counts a format as native there, and after a JAX or driver
upgrade:

    python -m tests.support.precision_devices [--platform gpu]

A format the device runs natively (`product_execution`) must equal the CPU's emulation up to the
order of the sums in the accumulate format, or, for a product format (TensorFloat-32, bfloat16 passes), up to the
device's own rounding of the operands (a few of the format's unit roundoffs); an emulated one
must equal it up to the sums' order. On an RTX 5060 (2026-10-10, jax 0.10.2) this check found
XLA's FP8 product 70% off for some shapes and its bfloat16 passes (`BF16_BF16_F32_X3`) a single
pass's error for some shapes (XLA's faults, not the card's: its Triton GEMM emitter for FP8, and a
matrix-vector product compiled without the algorithm; I-97). The engine emulates a GPU's
bfloat16-passes matrix-vector products (`engine.precision.products.EMULATED_MATRIX_VECTOR`), and
its FP8 under Triton, XLA's default; run the check both ways, as the engine runs:

    python -m tests.support.precision_devices --platform gpu
    XLA_FLAGS=--xla_gpu_enable_triton_gemm=false python -m tests.support.precision_devices --platform gpu

the second with the GPU's products through cuBLAS, where FP8 is native (exact on every shape,
2026-10-19). Every format passes both ways, TensorFloat-32 within 4.5e-4 of the scale of the
emulation (the device rounds its operands its own way, within the format's unit roundoff). The same card summed float32
operands asked for a float64 result in float32 (7.6e-7 off, 2026-10-11): the engine now
multiplies them as float64 values on float32's grid.

Exit status 1 when a format fails.
"""
import argparse
import itertools
import os
import sys

from engine.api.worker import deterministic_kernels_environment  # noqa: E402  (no JAX)

# A process of its own on a GPU shared with others, run as the engine worker runs: no
# preallocation, deterministic kernels; its GEMM emitter the environment's (see the docstring).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.update(deterministic_kernels_environment(os.environ))

import jax  # noqa: E402
import numpy as np  # noqa: E402

from engine.precision import (  # noqa: E402  (importing engine turns float64 on)
    FORMAT_NAMES, FORMATS, SHARED, Arithmetic, matmul, product_execution, product_precision,
)
from engine.precision.arithmetic import arithmetic_refusal  # noqa: E402

#: Every arithmetic a stage accepts: each compute format with float32 and float64 accumulation.
ARITHMETICS = tuple(Arithmetic(compute, accumulate) for compute in FORMAT_NAMES for accumulate in ("float32", "float64")
                    if arithmetic_refusal(FORMATS[compute], FORMATS[accumulate]) is None)

#: Operand shapes checked, each with either operand shared by the paths (its terms in one product,
#: A-23): a tall product (paths by flows), a matrix-vector one, a square one and a small one, rows
#: and columns of very different scales.
SHAPES = ((512, 64, 8), (512, 64, 1), (64, 64, 64), (16, 32, 16))


def check(device) -> int:
    """Print every arithmetic's row on `device` and return how many failed."""
    cpu = jax.devices("cpu")[0]
    print(f"device {device.device_kind} ({device.platform}), compute capability "
          f"{getattr(device, 'compute_capability', None)}, jax {jax.__version__}")
    rng = np.random.default_rng(7)
    failures = 0
    for arithmetic in ARITHMETICS:
        name, fmt = arithmetic.compute, FORMATS[arithmetic.compute]
        execution = product_execution(name, device)
        worst_cpu = worst_exact = 0.0
        for (m, k, n), shared in itertools.product(SHAPES, SHARED):
            a = rng.standard_normal((m, k)) * np.logspace(-3, 3, m)[:, None]
            b = rng.standard_normal((k, n)) * np.logspace(3, -2, n)[None, :]
            scale = np.abs(a) @ np.abs(b)
            program = jax.jit(lambda x, y: matmul(x, y, arithmetic, shared))
            with jax.default_device(device):
                on_device = np.asarray(program(jax.device_put(a, device), jax.device_put(b, device)), np.float64)
            with jax.default_device(cpu):
                emulated = np.asarray(program(a, b), np.float64)
            worst_cpu = max(worst_cpu, float(np.max(np.abs(on_device - emulated) / scale)))
            worst_exact = max(worst_exact, float(np.max(np.abs(on_device - a @ b) / scale)))
        tolerance = 64 * float(np.finfo(arithmetic.accumulate).eps) / 2
        if fmt.product != "dtype" and execution == "native":
            tolerance = 4 * 2.0 ** -(fmt.mantissa_bits + 1)
        ok = worst_cpu <= tolerance
        failures += not ok
        print(f"  {name:17s} {arithmetic.accumulate:8s} {execution:9s} {str(product_precision(name, device)):30s} "
              f"device - emulation {worst_cpu:.2e}  device - exact {worst_exact:.2e}  {'ok' if ok else 'FAILED'}")
    return failures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.support.precision_devices",
                                     description="Check every compute format's product on a device.")
    parser.add_argument("--platform", default=None, help="gpu, tpu or cpu (default: JAX's default backend)")
    args = parser.parse_args(argv)
    device = jax.devices(args.platform)[0] if args.platform else jax.devices()[0]
    return 1 if check(device) else 0


if __name__ == "__main__":
    sys.exit(main())
