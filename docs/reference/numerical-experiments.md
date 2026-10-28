# Selecting numerical experiments

Numerical strategies are part of the Python and HTTP precision contracts. Configure
them per stage through `StagePrecision.product`; pricing overrides in `by_product`
and `by_trade` carry the same options. The job validates the selected combinations
and hardware requirements at startup. Changing an option creates a separate JIT
specialization; changing a stochastic rounding seed reuses that specialization.

## Product options

`ProductOptions()` selects the basic profile below. It does not turn on a numerical
correction when an input triggers a known compiler problem.

| Field | Default | Explicit experiment and cost |
| --- | --- | --- |
| `shared_terms` | `1` | `2`, `3`, or `4` residual terms for the shared operand. Expands a matrix dimension and adds reconstruction work. `_1x` formats require `1`; `bfloat16_3x` has its own decomposition. |
| `scaling` | `"none"` | `"power_of_two"` scales each row/column into the FP16/FP8 range and rescales its product. Adds maxima, exponent calculations and elementwise operations. BF16/TF32 already have FP32's exponent range. |
| `shared_format` | `"same"` | `"e4m3"` holds an E5M2 product's shared operand in E4M3. This changes its rounding grid, not just its implementation. |
| `bf16_padding` | `"none"` | `"gemm8"` pads GPU `bfloat16_3x` vector products to at least eight output lanes, then slices the result. Avoids the observed GEMV lowering error at the cost of additional output lanes. |
| `subnormals` | `"native"` | `"normalize_fp16"` preserves quantized FP16 subnormals by splitting and normalizing them before native dots. Adds input checks and a conditional; uses one dot for normal inputs and four when subnormals occur. Requires a capable GPU. |
| `algorithm` | `"format"` | `"fp8_full"` explicitly requests `ANY_F8_ANY_F8_F32`. Requires a capable GPU. A compiler may reject this algorithm. |
| `implementation` | `"auto"` | `"emulated_fp32"` preserves the requested low-format operand grids but performs software products using FP32 operands. Requires research mode. It cannot be combined with native-only algorithm/normalization requests. |

`"auto"` dispatch follows the selected backend's capabilities, subject to the
execution policy. `mode="native"` refuses software fallback. `"allow_emulation"`
permits software paths; it does not force emulation or certify accuracy. Missing
FP8 instruction or numerical qualification is metadata, rather than a refusal.
See [precision profiles and execution](explicit-precision-policy.md) for the
end-to-end FP32 profile and the execution contract.

An unpadded GPU `bfloat16_3x` vector is known inaccurate on the tested compiler.
The report records that limitation; select padding explicitly to try the workaround.
Shape-dependent compiler errors can still arise after startup validation.

```python
from engine.precision import ExecutionPolicy, Precision, ProductOptions, StagePrecision

# Change only the pricing stage. Keep stored values in FP32 to isolate compute.
precision = Precision(
    pricing=StagePrecision(
        storage="float32", compute="float16", accumulate="float32",
        product=ProductOptions(subnormals="normalize_fp16"),
    ),
    execution=ExecutionPolicy(backend="gpu", mode="native"),
)
```

The equivalent HTTP fragment, inside the request's `config`, is:

```json
{
  "execution": {"backend": "gpu", "mode": "native", "gpu_gemm": "inherit"},
  "precision": {
    "pricing": {
      "storage": "float32", "compute": "float16", "accumulate": "float32",
      "product": {
        "algorithm": "format", "shared_terms": 1, "scaling": "none",
        "shared_format": "same", "bf16_padding": "none",
        "subnormals": "normalize_fp16", "implementation": "auto"
      }
    }
  }
}
```

The same options are available in standalone `Arithmetic(..., product=...)`.

## Reproducing the previous product profile

**Compatibility change:** low-format products without explicit options now use
one shared term, no power-of-two scaling, no mixed shared format, and no BF16
padding. Their numbers and speed can change. FP64/FP32 default products retain
their previous behavior. The experiment above does not retain the old scaling
or residual terms unless you also request them.

To select the historical product profile, use
`StagePrecision.compensated("float32", "float16", "float32")`,
`Arithmetic.compensated("float16", "float32", execution="allow_emulation")`, or
`Precision.throughout(name, product=ProductOptions.compensated(name))`.
This profile includes the previous scaling, residual terms, mixed E5M2/E4M3
representation, and BF16 padding where applicable. It does **not** enable the
new FP16 subnormal correction. Use `dataclasses.replace` on its `ProductOptions`
to add that correction explicitly. HTTP callers supply the corresponding fields.

Historical numerical campaigns select this profile explicitly. To reproduce their
synthetic product checks use `python -m tests.support.precision_devices --platform gpu --profile compensated`; omit that profile to study basic products. The
boundary diagnostic also accepts `--profile basic|compensated` and records its
options. Its `--algorithm full-fp8` remains a separate staged diagnostic against
an unchanged fused control, rather than modifying the control's policy.

## Storage, rounding and reference runs

Storage is independent of product scaling. `Precision.storage_scaling` defaults
to `"power_of_two"`, preserving the existing block storage encoding. Choose
`storage_scaling="none", offset="none"` for raw low-format arrays. These arrays
can underflow or overflow where block scaling would preserve their range.
Choose FP32 storage when isolating low-format compute behavior.

Existing API controls remain available: `offset="block"|"none"`,
storage `rounding="nearest"|"stochastic"`, per-stage
`compute_rounding="nearest"|"stochastic"`, `rounding_seed`, and
`paired_fraction`. Storage scaling and rounding apply across the policy; product
options and compute rounding can differ per stage/product/trade. Paired runs
are off by default and require FP64 on the same backend.

## Compiler route belongs to a worker

`ExecutionPolicy.gpu_gemm` / `execution.gpu_gemm` accepts `"inherit"` (default),
`"default"`, or `"cublas"`. An explicit value is a requirement that the worker
already has matching startup settings. It does not change a running compiler.

Start an external GPU worker with `jax-risk-worker --queue <queue-path> --gpu-gemm cublas`, and submit `gpu_gemm="cublas"` to it. The API and external
worker must use the same queue; see the [worker guide](../reference/http-api.md#runs-the-queue-and-the-engine-worker).
A mismatched route fails before numerical work. This implementation does not
create pools or route jobs to another worker automatically. Different route
experiments require separate worker processes. A cuBLAS route request disables
Triton GEMM selection; it does not prove a particular library kernel or native
instruction executed.

For direct Python experiments, set compiler flags in a fresh process before JAX
opens the backend, then call
`engine.precision.runtime.capture_compiler_runtime()` before backend initialization.
The API worker captures these settings at startup. Subsequent environment edits
do not change the recorded worker contract. The legacy
`JAX_RISK_EMULATE_PRODUCTS` setting remains available for process-level diagnostics;
use `implementation="emulated_fp32"` for experiments configured through the API.

## Reading results and limits

`resolved_policy` records the selected strategies, overrides, storage encoding,
and captured compiler flags. `executed_products` records the requested arithmetic
and dispatch for products reported by the pipeline. The summary `products` can
say `"mixed"` when a format uses both native and emulated dispatch in one job.
These records describe policy and dispatch, not instruction-level qualification.
New product profiles receive separate acceptance-evidence keys, so they cannot
inherit passing evidence measured under the old profile.

The difference-form pricing kernels, level reconstruction and mandatory FP64
internals remain core implementations, with no alternative implementation exposed.
Changing them requires implementing and validating another kernel, not adding a
label. Full TPU jobs therefore remain unsupported. The new opt-ins do not resolve
native FP8 qualification or establish H100 correctness. The subsequent
[TPU survey](../hardware-testing/tpu-precision-survey.md) passes the basic/compensated FP32-input product
and storage coverage, including revised FP16 emulation. Run device-specific numerical
and lowering checks for additional strategies you intend to study.

The subsequent [native TPU probes](../hardware-testing/tpu-native-precision.md) find passing FP8-typed
matrix candidates using ordinary JAX and Pallas. They also find FP32 widening for
the tested vector shape and compiler rejection of the generic explicit FP8 preset.
These candidates are not integrated into the current engine/API dispatch.
