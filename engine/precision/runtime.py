"""Compiler settings captured once per worker, never mutated by a job."""
from dataclasses import dataclass
import os
import threading


@dataclass(frozen=True)
class CompilerRuntime:
    gpu_gemm: str
    xla_flags: str


_RUNTIME = None
_LOCK = threading.Lock()


def capture_compiler_runtime(environ=None):
    """Call before backend initialization; subsequent captures retain the first settings."""
    global _RUNTIME
    with _LOCK:
        if _RUNTIME is None:
            flags = (os.environ if environ is None else environ).get("XLA_FLAGS", "")
            route = "cublas" if "--xla_gpu_enable_triton_gemm=false" in flags.split() else "default"
            _RUNTIME = CompilerRuntime(route, flags)
        return _RUNTIME


def validate_compiler_route(requested, platform):
    runtime = capture_compiler_runtime()
    if requested != "inherit":
        if platform != "gpu":
            raise ValueError("execution.gpu_gemm requires an active GPU backend")
        if requested != runtime.gpu_gemm:
            raise ValueError(f"execution.gpu_gemm={requested!r} requires a worker started with --gpu-gemm {requested}; "
                             f"this worker uses {runtime.gpu_gemm!r}. Compiler settings cannot change per job")
    return runtime
