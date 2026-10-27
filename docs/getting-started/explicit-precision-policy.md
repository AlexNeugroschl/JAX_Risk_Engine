# Explicit precision and execution

Every full engine request carries numerical formats and an execution policy.
Unsupported work raises an error before the pipeline starts. The engine does not
change a requested format, select another backend, or move the job to CPU.

```python
from engine.precision import ExecutionPolicy, Precision, StagePrecision

# Default: native FP64 reference arithmetic.
reference = Precision.reference(backend="gpu")

# FP16 products/storage and FP32 accumulation in adjustable stages.
# Fixed internal kernels remain FP64.
stage = StagePrecision("float16", "float16", "float32")
gpu_policy = Precision(
    simulation=stage, market=stage, pricing=stage,
    execution=ExecutionPolicy(backend="gpu", mode="native"),
)

# Explicit permission for software or unqualified product paths in research.
research = Precision.research(
    pricing=StagePrecision("float32", "float8_e4m3fn", "float32"),
)
```

Pass the policy as `RunConfig(precision=...)`, `RunRequestConfig(precision=...)`,
or `MarketRiskRequest(precision=...)`. `simulate` and path valuation also accept
it directly. Select the backend in the caller's JAX configuration/context first;
`backend="gpu"` verifies that selection rather than changing it. The default
`backend="default"` uses the active JAX default device and resolves to its concrete
platform in the report. Scenario sharding stays on that platform.

Direct matrix-product experiments also require explicit permission for emulation:

```python
from engine.precision import Arithmetic, matmul

arithmetic = Arithmetic("float8_e4m3fn", "float32", execution="allow_emulation")
result = matmul(a, b, arithmetic, shared="right")
```

`allow_emulation` permits fallback; it does not force it, change precision, or
establish correctness. Numerical harnesses explicitly opt into this mode.
`JAX_RISK_EMULATE_PRODUCTS` can force selected software product paths, but conflicts
with a `native` requirement cause an error. Set compiler/emulation environment
variables before tracing in a fresh process, as before.

The HTTP configuration uses the same fields:

```json
{
  "config": {
    "precision": {
      "execution": {"backend": "gpu", "mode": "native"},
      "pricing": {"storage": "float16", "compute": "float16", "accumulate": "float32"}
    }
  }
}
```

The API validates request structure on its CPU. The worker checks hardware
requirements on its own device before numerical work. A mismatch fails the job;
the API server cannot infer worker capabilities from its own CPU.

Runtime checks are integrated into the existing validation functions:
`validate_run(request, execution=True)` for a complete job, and the corresponding
`validate_request(..., execution=True)` for a direct portfolio or market-risk job.
The returned immutable execution context is passed through private implementations
of simulation, calibration, pricing, Greeks and paired reference calculations.
Nested analytics do not repeat policy validation. In a multi-analytic job, all
requested stages and pricing overrides are checked before the first analytic runs.

Public standalone helpers still validate their own execution requirements. Low-level
format/product guards remain in place, including checks during JIT tracing, and
execution mode remains part of the JIT cache identity. There is no process-wide
"already validated" flag or cached permission to execute. Each new job rechecks
the active backend, x64 support and emulation controls.

Only static policy-report metadata is cached, by immutable policy and concrete
backend. Callers receive fresh copies; actual output formats, devices, path counts,
figures and qualification observations are read anew for each result.

See [Selecting numerical experiments](numerical-experiments.md) for per-stage
product strategies, storage scaling and worker compiler requirements. Low-format
products now default to one unscaled shared term; the historical compensated
profile is available explicitly.

## Fixed precision is part of the contract

`Precision.internal` declares the existing FP64 kernels: `calibration`, `levels`,
`initial_values`, `reductions`, `greeks`, and `output`. These fields currently
accept only `"float64"`. Asking for FP32 raises an error because those kernels
have not been converted and validated. Host input processing and model moments
also use FP64; host preprocessing is recorded separately from device work.

`Precision.throughout("float32")` retains its historical meaning: all three
**adjustable** stages use FP32. Fixed kernels remain FP64. There is currently no
all-FP32 full-engine profile.

Consequently, **full TPU engine jobs are currently refused**, including jobs with
FP32 adjustable stages or `allow_emulation`. TPU software FP64 is incompatible
with the engine's IEEE-word precision helpers. Requested FP64 with JAX x64 disabled
is also refused instead of accepting implicit FP32 demotion. Individual FP32/BF16
primitives can still be tested on TPUs. Supporting full TPU jobs requires real
FP32 versions of the fixed kernels and new device validation. A warning or changed
policy label would not implement that support.

Paired FP64 correction runs stay on the selected backend. There is no automatic
CPU reference pass or hybrid-device execution.

## Read the result

Every precision report includes `resolved_policy`: stage formats and overrides,
rounding and seed, fixed kernel formats, concrete backend, execution mode, product
algorithms/representations, host preprocessing and paired-reference requirements.
`policy` preserves the original request. `realized`, `devices`, and `backend`
describe the stored arrays and actual device platform.

`products` remains a **dispatch** description. A native dispatch label does not
prove that every compiled shape used the requested hardware instructions. Native-only
FP8 currently raises an error even on a capable GPU: the H100 trial emitted FP16
instructions for FP8 operand programs. Explicit research mode can run that path
without claiming native FP8 qualification. See the
[hardware rerun findings](hardware-correctness-rerun.md).

The default FP64/FP32 kernels keep their pricing formulas and tolerances.
Low-format algorithms follow the selected product options; checks run in Python
at entry/tracing boundaries.
Execution mode participates in JIT cache identity, so a permissive compiled
product cannot satisfy a later strict request by cache reuse. The policy does not
resolve the measured H100 FP16/BF16 pipeline differences or qualify broad TPU correctness.
