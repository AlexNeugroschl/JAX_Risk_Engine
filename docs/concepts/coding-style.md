# Coding Style & Technical Constraints

Rules that apply throughout this codebase, independent of any one module. See
[Architecture](architecture.md) for how the pieces fit together, and
[ORE Parity](../reference/ore-parity.md) for how ORE's own C++ source gets translated into
this codebase's JAX code.

## Core constraints

- **Precision follows the inputs.** Only the cast points read the run's `Precision`
  (`simulate`, `value_portfolio`, the exposure and market-risk reductions; see
  [Architecture: Adjustable precision](architecture.md#adjustable-precision)). Between them a
  kernel computes in the dtype of the arrays it is given and takes no dtype argument: it
  casts its own constant inputs (coupon tables, volatilities, calibration baskets, grid
  nodes) to the dtype of its curves (`engine.market_data.curves.curve_dtype`), and never builds an
  array with JAX's default dtype (`jnp.ones(shape)` is float64 under x64 whatever the curves
  are). Scalars that enter arithmetic with curve arrays are Python floats, which follow the
  array's dtype, not NumPy scalars, which do not. Strict dtype promotion in CI catches a
  violation. Constants that depend on the dtype (tolerances, clip epsilons) are derived from
  it (`jnp.finfo(dtype).eps`), never written for float64 only. Calibration today, t=0 values,
  Greeks and reductions over paths are float64 by decision (A-10); the factories that build a
  stage's functions (`trade_price_function`) take the stage's dtype once.
- **Matrix products through `engine.precision.matmul`.** No bare `@`, `jnp.matmul`,
  `jnp.dot`, `jnp.einsum` or `jnp.tensordot` on JAX arrays in engine code: `matmul(a, b,
  arithmetic)` (the semantics of `a @ b`) runs the product in the stage's compute format and
  accumulates it in its accumulate format (`engine.precision.Arithmetic`, a static argument of
  the kernel), stating its precision in the program; without an arithmetic, float32 or float64
  operands in their own format, full precision (decision A-22). A kernel with products is in
  difference form: what enters a product is a path's deviation from a level no path depends on,
  the level computed and added at the accumulate format (F-07; [details/precision.md
  §8](../planning/details/precision.md#8-compute-below-float32)). Never round a value by a
  dtype round trip inside a kernel (`x.astype(low).astype(float32)`): XLA may remove it; round
  on the format's grid (`engine.precision.grid`). Why: a product that states none
  runs at the device's default, which is below float32 on accelerators (TensorFloat-32 on an
  NVIDIA GPU, bfloat16 passes on a TPU), so a float32 policy would compute below float32 while
  its precision report says float32; on an RTX 5060 a float32 cube came out 1.5% off. Nor is
  the process-wide `jax_default_matmul_precision` set, which any process could override. A
  contraction that is not a matrix product takes that form first (reshape, transpose; the
  Brownian bridge's `[T, T] x [T, S, d]` is one product over `[T, S*d]`), or an elementwise
  product and a sum where it is small. NumPy products on the host are exact in their dtype and
  stay as they are. Enforced by `tests/test_accelerator_defaults.py::
  TestEveryMatrixProductStatesItsPrecision`, which traces the pipelines (AD's transposes
  included) and fails on any product without its precision, naming the line (slow tier).
- **No state mutation.** JAX requires pure functions — never use in-place array updates
  (`x[0] = 1`).
- **Vectorization over loops.** Never use an ordinary Python `for` loop inside
  JIT-compiled code. Use `jax.lax.scan` for chronological time-stepping, and broadcasting,
  `jnp.where` and matrix products (`engine.precision.matmul`, above) for cross-sectional
  trade logic.
- **API-first design.** Data-ingestion logic is written to expect dictionaries/JSON
  natively, treating the XML parser (where used) as a test/validation adapter rather than
  a hard dependency.

## Porting from ORE's C++ source

Where a formula or algorithm is translated from ORE's own C++ (`reference/ORE`, a
read-only clone — see [ORE Parity](../reference/ore-parity.md)), the rule is: extract the
**mathematical logic**, discard the **software architecture**.

- **Do not port objects.** ORE uses heavy OOP (classes, inheritance, mutable state). JAX
  requires pure, stateless functions — a `QuantExt::LinearGaussMarkovModel` becomes a
  handful of plain functions operating on arrays, not a class.
- **Do not port loops.** ORE iterates over scenarios and time steps with ordinary `for`
  loops. Translate these into `jax.lax.scan` for time, and vectorized tensor operations
  (broadcasting, `jnp.where`, `engine.precision.matmul`) for scenarios and assets.
- **Preserve variable names where mathematically logical.** Keep parameter names aligned
  with ORE's own math (`A(t,T)`, `B(t,T)`, `mean_reversion`, `theta`) so the correspondence
  to ORE's source stays legible — this is what makes the
  [ORE Parity](../reference/ore-parity.md) doc's line-by-line mapping possible.

## Testing philosophy

Every non-trivial formula is tested two ways — see
[Architecture: Testing philosophy](architecture.md#testing-philosophy) for the full
explanation:

1. Direct correctness checks (does a construction have the mathematical property it's
   supposed to have?).
2. Cross-checks against the real, installed `ORE` Python package.
