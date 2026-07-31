# API Reference

Exact inputs and outputs for every public function and configuration dataclass. For the
*why* behind these shapes, see the per-stage deep dives
([Market Simulation](03-market-simulation.md), [Instruments](04-instruments.md),
[Risk Statistics](05-risk-statistics.md)). For runnable examples, see the
[User Guide](06-user-guide.md).

**Notation:** `[Scenarios, TimeSteps, ...]` describes an array's shape. `Scenarios` is
however many simulated alternate futures were requested; `TimeSteps` is
`len(time_grid) - 1` (the simulation's output steps are the points *after* time zero, not
including time zero itself).

---

## `engine.market_simulations`

### `SimulationConfig`

The top-level input to `generate_paths()`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `time_grid` | `List[float]` | *required* | Absolute simulation times, ascending, starting at `0.0`. E.g. `[0.0, 0.5, 1.0]` simulates two steps, at 6 months and 1 year. |
| `equities` | `EquityConfig` | *required* | The equity/FX leg of the model. |
| `rates` | `RatesConfig` | *required* | The interest rate leg of the model. |
| `joint_covariance` | `List[List[float]]` | *required* | `[NumEq+NumHW, NumEq+NumHW]` covariance matrix. Row/column order: equities first (in `equities.initial_prices`' order), then rates (in `rates.initial_rates`' order). |
| `scenarios` | `int` | `10000` | Number of simulated alternate futures. |

### `EquityConfig`

| Field | Type | Meaning |
|---|---|---|
| `initial_prices` | `List[float]` | Starting price of each equity/FX pair. Length = `NumEq`. |
| `dividend_yields` | `List[float]` | Dividend yield (or, for FX, the foreign risk-free rate) per equity/FX pair, same length/order as `initial_prices`. |
| `rate_mapping` | `List[List[float]]` | `[NumEq, NumHW]`. Row `i` gives the Uncovered-Interest-Rate-Parity drift coefficients for equity/FX `i` against every interest rate factor — see [Market Simulation](03-market-simulation.md#phase-2--the-cross-asset-model-engine). |

### `RatesConfig`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `initial_rates` | `List[float]` | *required* | Starting short rate per rate factor. Length = `NumHW` (number of Hull-White factors). |
| `theta` | `List[float]` | *required* | Long-run mean-reversion target per factor, same length/order as `initial_rates`. |
| `mean_reversion` | `List[float]` | *required* | Mean-reversion speed (`a`) per factor. |
| `maturities` | `Optional[List[float]]` | `None` | Absolute future times to reconstruct discount factors for. **If set, triggers `"yield_curves"` in `generate_paths`'s output; if omitted, no yield curve cube is built.** |
| `initial_zero_curves` | `Optional[List[ZeroCurveConfig]]` | `None` | **Required if `maturities` is set.** One `ZeroCurveConfig` per rate factor, same order as `initial_rates`. Length must exactly match `len(initial_rates)`, or `generate_paths` raises `ValueError`. |

### `ZeroCurveConfig`

Today's market zero curve for **one** rate factor, used to calibrate that factor's
Hull-White `A(t,T)` term (see
[Market Simulation: Phase 3](03-market-simulation.md#phase-3--yield-curve-reconstruction)).

| Field | Type | Meaning |
|---|---|---|
| `times` | `List[float]` | Zero-curve pillar times, e.g. `[0.0, 1.0, 2.0, 5.0, 10.0, 30.0]`. |
| `rates` | `List[float]` | Zero rate at each pillar, same length/order as `times`. |

### `generate_paths(config: SimulationConfig, precision: int = 64) -> Dict[str, jax.Array]`

Runs the full Sobol → Brownian bridge → cross-asset Monte Carlo → yield curve
reconstruction pipeline (see [Market Simulation](03-market-simulation.md) for what each
stage does).

**Parameters**
- `config` — a `SimulationConfig`.
- `precision` — `64` (default, float64) or `32` (float32). See
  [Architecture: Adjustable precision](02-architecture.md#adjustable-precision).

**Returns** a `dict`:

| Key | Shape | Always present? |
|---|---|---|
| `"equities"` | `[Scenarios, TimeSteps, NumEq]` | Yes |
| `"rates"` | `[Scenarios, TimeSteps, NumHW]` | Yes |
| `"numeraire"` | `[Scenarios, TimeSteps]` | Yes |
| `"yield_curves"` | `[Scenarios, TimeSteps, Maturities, NumHW]` | Only if `config.rates.maturities` is set |

**Raises** `ValueError` if `len(config.rates.initial_zero_curves) != len(config.rates.initial_rates)` (when `maturities` is set).

### Lower-level functions

These are used internally by `generate_paths` but are independently documented and
tested — see [Market Simulation](03-market-simulation.md) for what each one does
mathematically.

| Function | Signature | Notes |
|---|---|---|
| `generate_sobol_normals` | `(num_scenarios: int, num_steps: int, num_assets: int, dtype) -> jax.Array` | Returns `[TimeSteps, Scenarios, Assets]`. Honors `dtype` unconditionally on output. |
| `apply_brownian_bridge` | `(Z: jax.Array, time_grid: jax.Array) -> jax.Array` | Returns standardized sequential shocks, same shape as `Z`. |
| `compute_hw_A_matrix` | `(zero_curves: List[ZeroCurveConfig], hw_a, hw_sigma, step_times, maturities, B_matrix) -> np.ndarray` | Plain NumPy (CPU-only). Returns `[TimeSteps, Maturities, NumRates]`. |
| `reconstruct_yield_curves` | `(hw_paths: jax.Array, A: jax.Array, B: jax.Array) -> jax.Array` | `@jax.jit`-compiled. Returns `[Scenarios, TimeSteps, Maturities, NumRates]`. |

---

## `engine.instruments.interest_rate_swap`

### `SwapConfig`

Describes one vanilla fixed-vs-floating interest rate swap.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `notional` | `float` | *required* | Notional amount the interest payments are calculated on. |
| `fixed_rate` | `float` | *required* | The agreed fixed interest rate. |
| `payer` | `bool` | *required* | `True` = this side pays fixed, receives floating. `False` = the reverse. |
| `discount_curve_index` | `int` | *required* | Which rate factor (index into the simulation's `NumRates` axis) discounts this swap's cashflows. |
| `forward_curve_index` | `int` | *required* | Which rate factor sets the floating leg's forward rates. Equal to `discount_curve_index` for single-curve discounting. |
| `swap_tenor` | `str` | `"5Y"` | ORE `Period` string, e.g. `"5Y"`, `"18M"`. |
| `index_tenor_months` | `int` | `6` | Floating leg reset frequency in months (`6` = semi-annual). |
| `floating_spread` | `float` | `0.0` | Fixed spread added to every floating payment. |
| `evaluation_date` | `ORE.Date` | today's global ORE evaluation date | The swap's "as-of" date. |

### `price_swaps(yield_curves: jax.Array, maturities: np.ndarray, swap_configs: List[SwapConfig]) -> jax.Array`

**Parameters**
- `yield_curves` — `[Scenarios, TimeSteps, Maturities, NumRates]`, typically
  `generate_paths(...)["yield_curves"]`.
- `maturities` — the same absolute-time pillar array passed as
  `config.rates.maturities` to `generate_paths`. **Every swap's payment/accrual dates
  must land exactly on one of these pillars** — see
  [Instruments: maturity-pillar alignment](04-instruments.md#a-known-limitation-maturity-pillar-alignment).
- `swap_configs` — a list of one or more `SwapConfig` objects.

**Returns** `[Scenarios, TimeSteps, Trades]` — one NPV value per scenario, per time step,
per swap in `swap_configs` (in the order given).

**Raises** `ValueError` if any swap's payment/accrual dates don't land exactly on a
`maturities` pillar.

### Lower-level functions

| Function | Signature | Notes |
|---|---|---|
| `prepare_swap` | `(cfg: SwapConfig, maturities: np.ndarray) -> _PreparedSwap` | CPU-only, per-trade one-time setup. Builds the real ORE trade and resolves cashflow dates onto maturity-pillar indices. |

---

## `engine.aggregate_statistics.risk_statistics`

Every function here is instrument-agnostic — see
[Risk Statistics](05-risk-statistics.md) and
[Architecture: stages agree on shapes, not code](02-architecture.md#design-principle-stages-agree-on-shapes-not-code).

### `portfolio_pnl(npv_cube: jax.Array, base_npv: float) -> jax.Array`

**Parameters**
- `npv_cube` — `[Scenarios, TimeSteps, Trades]`, from any pricer.
- `base_npv` — the portfolio's value today (t=0), from a separate zero-shock
  revaluation. See [Risk Statistics: the P&L baseline](05-risk-statistics.md#the-pl-baseline-what-are-gainslosses-measured-against).

**Returns** `[Scenarios, TimeSteps]` — `sum(npv_cube, axis=Trades) - base_npv`.

### `value_at_risk(pnl: jax.Array, percentile: float) -> jax.Array`

**Parameters**
- `pnl` — `[Scenarios, TimeSteps]`, typically from `portfolio_pnl`.
- `percentile` — e.g. `0.99` for 99% VaR.

**Returns** `[TimeSteps]` — always `>= 0`.

### `expected_shortfall(pnl: jax.Array, percentile: float) -> jax.Array`

Same signature as `value_at_risk`. **Returns `NaN`** for any time step whose loss tail
(strictly worse than that step's VaR) is empty — see
[Risk Statistics: the formulas](05-risk-statistics.md#the-formulas). Callers must check
for this explicitly.

### `compute_risk_metrics(npv_cube: jax.Array, base_npv: float, percentiles: Sequence[float] = (0.95, 0.99)) -> Dict[str, jax.Array]`

The main entry point — combines the three functions above.

**Parameters**
- `npv_cube` — `[Scenarios, TimeSteps, Trades]`.
- `base_npv` — see `portfolio_pnl` above.
- `percentiles` — which confidence levels to compute VaR/ES at. Default `(0.95, 0.99)`.

**Returns** a `dict` with one `"VaR_<pct>"` and one `"ES_<pct>"` key per entry in
`percentiles` (e.g. `percentiles=(0.95, 0.99)` produces `"VaR_95"`, `"ES_95"`, `"VaR_99"`,
`"ES_99"`), each shaped `[TimeSteps]`.

---

## `engine.scenarios`

Reference/demo configurations and shared test helpers — not part of the pricing
pipeline itself, but used throughout the codebase's demos and tests. See
[Architecture: engine/scenarios.py](02-architecture.md#enginescenariospy-shared-example-configurations).

| Name | Type | Meaning |
|---|---|---|
| `EVAL_DATE` | `ORE.Date` | Shared evaluation date for every demo/test scenario in this module. |
| `SWAP_DEMO_MATURITIES` | `List[float]` | The maturity pillars required by `single_currency_swap_demo_config()`'s swap. |
| `cross_asset_demo_config()` | `() -> SimulationConfig` | Two-equity, two-currency (USD/EUR) example scenario. |
| `single_currency_swap_demo_config()` | `() -> SimulationConfig` | One-currency, two-rate-factor (discounting + forwarding) example scenario, sized for a 2Y demo swap. |
| `flat_yield_curves(disc_rate, fwd_rate, maturities=SWAP_DEMO_MATURITIES, eval_date=EVAL_DATE)` | `(...) -> jax.Array` | Builds a deterministic `[1, 1, len(maturities), 2]` yield curve cube directly from ORE's own flat curve objects — no simulation randomness. Used for VaR's `base_npv` baseline and for ORE cross-check tests. |
