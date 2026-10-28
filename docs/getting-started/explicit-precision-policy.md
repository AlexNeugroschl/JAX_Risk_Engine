# Precision profiles and execution

The default remains FP64 numerical execution on the active JAX backend. Selecting a
precision never selects another device, silently downgrades a format, or enables
emulation. Missing numerical qualification does not prevent a research run.

## A small starting configuration

The complete API configuration has independent precision and execution sections:

```json
{
  "config": {
    "precision": {"profile": "float32"},
    "execution": {"backend": "tpu", "mode": "native"}
  }
}
```

This is a configuration fragment; a POST /runs request still needs its market,
portfolio, analytics and the sections those analytics consume. The worker must
already have the requested backend selected. A mismatch fails before numerical
work; the CPU API server's own backend does not determine the worker's capability.

`profile` accepts `float64` (default) and `float32`. It supplies defaults for all
three adjustable stages and all six internal numerical roles. Explicit stage
fields and internal fields override the profile. For example:

```json
{
  "config": {
    "precision": {
      "profile": "float32",
      "pricing": {"compute": "bfloat16"}
    },
    "execution": {"backend": "gpu"}
  }
}
```

This selects BF16 pricing products, FP32 accumulation and storage, and FP32
internal arithmetic. It does not activate residual terms, product scaling,
padding, subnormal normalization or stochastic rounding.

The corresponding Python configuration is:

```python
from engine.precision import ExecutionPolicy, Precision, StagePrecision
from engine.run import RunRequestConfig

config = RunRequestConfig(
    precision=Precision(profile="float32",
                        pricing=StagePrecision("float32", "bfloat16", "float32")),
    execution=ExecutionPolicy(backend="gpu"),
)
```

A Python StagePrecision is already fully specified: StagePrecision("float16")
continues to mean FP16 storage and FP64 compute/accumulation. API partial stage
objects inherit missing fields from their selected profile. API per-product/trade
overrides inherit the selected profile too; Python StagePrecision objects remain
fully specified.

## What each setting controls

| Setting | Actual numerical work |
| --- | --- |
| simulation | Shock storage, state evolution products and stage arithmetic |
| market | Scenario market storage, products and stage arithmetic |
| pricing | Scenario revaluation products, accumulation and stored values |
| internal.calibration | Model bootstrap arithmetic and calibration-dependent model inputs |
| internal.levels | Path-independent levels and prepared kernel data |
| internal.initial_values | Today's trade valuations |
| internal.reductions | Portfolio aggregation, exposure reductions and risk statistics |
| internal.greeks | Bump or AD sensitivity arithmetic |
| internal.output | Exported financial values and arrays; counts and identifiers retain their meaning |

Internal roles accept FP32 or FP64; scenario product formats also include the
supported low-format algorithms. Low products accumulate in FP32. Overrides per
trade take precedence over overrides per product, then the pricing stage.

Host input parsing, schedules, scenario input preparation and numerical quadrature
setup still contain NumPy/SciPy FP64 work. These are reported as host preprocessing;
`profile=float32` specifies the engine's configurable numerical kernels, not an
absence of double precision anywhere in Python or external libraries. It is not
an all-CPU-free workload: scalar materialization and result assembly also run on
the host. Declared placement describes the backend required for JAX kernels.
JSON numerical scalars do not have a floating dtype;
array artifacts retain their requested dtype. Wider storage is permitted and does
not recover information lost in narrower arithmetic. Market-risk P&L anchors use the
revaluation stage arithmetic; internal.initial_values controls portfolio NPV and
exposure start values.

Precision.throughout("float32") keeps its historical meaning: only the adjustable
stages become FP32. Use Precision(profile="float32") for the expanded profile.
Existing config.precision.execution is a compatibility location. New requests
should use config.execution. Contradictory nondefault declarations are refused.

## Execution and research evidence

`backend` is default/cpu/gpu/tpu. `default` resolves to the active backend, without
hardware-dependent precision changes. `mode=native` forbids engine-selected
emulation. `allow_emulation` retains the explicit research permission for software
routes; it does not force emulation. ProductOptions(implementation="emulated_fp32")
explicitly selects FP32-carrier products and requires that permission.

Native means an engine/compiler route request, not proof of physical instructions.
FP8 requests are no longer refused because previous instruction or financial
qualification is missing. TPU low-format candidates are submitted to the compiler:
BF16 uses its explicit algorithm; FP8 uses typed operands and DEFAULT; FP16 and
TF32 use explicit algorithms that the tested TPU compiler rejected. Compiler errors
propagate; there is no catch-and-emulate fallback. Shape-dependent widening remains
possible, and requires separate lowering/profiler inspection.

Known FP16 subnormal and unpadded GPU BF16-three-pass issues are reported in
resolved policy metadata, without forbidding the experiment. Product corrections
remain individually opt-in. Native FP8 qualification and financial accuracy have
not been established by this configuration change.

FP64 on TPU remains an error. A paired FP64 sample also requires native FP64 on the
selected backend; it does not cause a CPU pass. There is no automatic hybrid
placement. An explicit hybrid execution plan is future work.

`gpu_gemm` is a requirement on worker startup (inherit/default/cublas), not a
per-job compiler switch. Process flags remain frozen after initialization.

## Validation and results

Structural request validation runs at submission; execution validation runs once
on the worker before numerical work. The immutable context records the stages and
internal roles actually used and is passed through the analytics. Unused FP64
declarations do not block an FP32-only analytic. Standalone numerical helpers retain
their own entry/tracing guards.
Policies and prepared array dtypes participate in JIT specialization; no per-job
x64 flag toggling or global precision context is used.

Every complete run, including calibration-only runs, carries `numerics`: the
resolved profile, role dtypes, backend requirements, declared placement, compiler
settings, known limitations and qualification status. Portfolio/market-risk
precision reports additionally record stored formats and observed array devices.
Qualification never authorizes or blocks execution. Internal precision choices
are included in accuracy evidence keys, so older FP64-internal results cannot
certify a new FP32 profile.

Local CPU regression tests cover the FP32 profile. GPU/TPU end-to-end execution,
physical instruction use, financial accuracy and performance of the new profile
still require hardware testing. Historical cloud reports describe their recorded
source revisions; they do not validate this implementation.
