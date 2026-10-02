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
  nodes) to the dtype of its curves (`engine.models.curves.curve_dtype`), and never builds an
  array with JAX's default dtype (`jnp.ones(shape)` is float64 under x64 whatever the curves
  are). Scalars that enter arithmetic with curve arrays are Python floats, which follow the
  array's dtype, not NumPy scalars, which do not. Strict dtype promotion in CI catches a
  violation. Constants that depend on the dtype (tolerances, clip epsilons) are derived from
  it (`jnp.finfo(dtype).eps`), never written for float64 only. Calibration today, t=0 values,
  Greeks and reductions over paths are float64 by decision (A-10); the factories that build a
  stage's functions (`trade_price_function`) take the stage's dtype once.
- **No state mutation.** JAX requires pure functions — never use in-place array updates
  (`x[0] = 1`).
- **Vectorization over loops.** Never use an ordinary Python `for` loop inside
  JIT-compiled code. Use `jax.lax.scan` for chronological time-stepping and
  `jnp.einsum`/`jnp.where` for cross-sectional trade logic.
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
  (`jnp.where`, `jnp.einsum`) for scenarios and assets.
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
