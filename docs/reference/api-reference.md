# API Reference

Exact inputs and outputs for every public function and configuration dataclass. For the
*why* behind these shapes, see the per-stage deep dives
([Market Simulation](../concepts/market-simulation.md), [Instruments](../instruments/swaps.md),
[Risk Statistics](../risk/statistics.md)). For runnable examples, see the
[User Guide](../getting-started/user-guide.md).

**Notation:** `[Scenarios, TimeSteps, ...]` describes an array's shape. `Scenarios` is
however many simulated alternate futures were requested; `TimeSteps` is
`len(time_grid) - 1` (the simulation's output steps are the points *after* time zero, not
including time zero itself).

---

## `engine.simulation`

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
| `rate_mapping` | `List[List[float]]` | `[NumEq, NumHW]`. Row `i` gives the Uncovered-Interest-Rate-Parity drift coefficients for equity/FX `i` against every interest rate factor — see [Market Simulation](../concepts/market-simulation.md#phase-2--the-cross-asset-model-engine). |

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
[Market Simulation: Phase 3](../concepts/market-simulation.md#phase-3--yield-curve-reconstruction)).

| Field | Type | Meaning |
|---|---|---|
| `times` | `List[float]` | Zero-curve pillar times, e.g. `[0.0, 1.0, 2.0, 5.0, 10.0, 30.0]`. |
| `rates` | `List[float]` | Zero rate at each pillar, same length/order as `times`. |

### `generate_paths(config: SimulationConfig, precision: int = 64) -> Dict[str, jax.Array]`

Runs the full Sobol → Brownian bridge → cross-asset Monte Carlo → yield curve
reconstruction pipeline (see [Market Simulation](../concepts/market-simulation.md) for what each
stage does).

**Parameters**
- `config` — a `SimulationConfig`.
- `precision` — `64` (default, float64) or `32` (float32). See
  [Architecture: Adjustable precision](../concepts/architecture.md#adjustable-precision).

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
tested — see [Market Simulation](../concepts/market-simulation.md) for what each one does
mathematically.

| Function | Signature | Notes |
|---|---|---|
| `generate_sobol_normals` | `(num_scenarios: int, num_steps: int, num_assets: int, dtype) -> jax.Array` | Returns `[TimeSteps, Scenarios, Assets]`. Honors `dtype` unconditionally on output. |
| `apply_brownian_bridge` | `(Z: jax.Array, time_grid: jax.Array) -> jax.Array` | Returns standardized sequential shocks, same shape as `Z`. |
| `compute_hw_A_matrix` | `(zero_curves: List[ZeroCurveConfig], hw_a, hw_sigma, step_times, maturities, B_matrix) -> np.ndarray` | Plain NumPy (CPU-only). Returns `[TimeSteps, Maturities, NumRates]`. |
| `reconstruct_yield_curves` | `(hw_paths: jax.Array, A: jax.Array, B: jax.Array) -> jax.Array` | `@jax.jit`-compiled. Returns `[Scenarios, TimeSteps, Maturities, NumRates]`. |

---

## `engine.instruments.swap`

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
  [Instruments: maturity-pillar alignment](../instruments/swaps.md#a-known-limitation-maturity-pillar-alignment).
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

## `engine.instruments.european_swaption`

### `SwaptionConfig`

Describes one European swaption (the option to enter a vanilla fixed-vs-floating swap at
a future exercise date). See [Instruments: European Swaptions](../instruments/european-swaptions.md) for the
Jamshidian's-trick pricing model.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `notional` | `float` | *required* | Notional amount of the underlying swap. |
| `fixed_rate` | `float` | *required* | The underlying swap's agreed fixed rate. |
| `payer` | `bool` | *required* | `True` = the option to enter a swap paying fixed, receiving floating. `False` = the reverse. |
| `rate_factor_index` | `int` | *required* | Which simulation Hull-White factor prices BOTH the underlying swap and the option (Jamshidian's trick is single-model — see [Instruments: European Swaptions](../instruments/european-swaptions.md#1-describing-a-swaption-swaptionconfig)). |
| `hw_a` | `float` | *required* | That rate factor's own Hull-White mean-reversion speed — must match `RatesConfig.mean_reversion[rate_factor_index]` in the simulation this swaption is priced against. |
| `hw_sigma` | `float` | *required* | That rate factor's own Hull-White volatility — must match the per-step volatility implied by the simulation's `joint_covariance` for this factor. |
| `initial_zero_curve` | `ZeroCurveConfig` | *required* | That rate factor's own today's-market zero curve — must match `RatesConfig.initial_zero_curves[rate_factor_index]`. |
| `swap_tenor` | `str` | `"5Y"` | ORE `Period` string for the underlying swap's length, e.g. `"5Y"`, `"18M"`. |
| `index_tenor_months` | `int` | `6` | Floating leg reset frequency in months (`6` = semi-annual). |
| `floating_spread` | `float` | `0.0` | Fixed spread added to every floating payment. |
| `forward_start` | `ORE.Period` | `ORE.Period(0, ORE.Days)` | How far in the future the underlying swap's accrual is delayed beyond the standard 2-day spot lag — e.g. `ORE.Period(5, ORE.Years)` for a swaption exercisable in ~5Y. |
| `exercise_lag_days` | `int` | `2` | Business days from `evaluation_date + forward_start` to the exercise date (standard spot-lag convention). |
| `evaluation_date` | `ORE.Date` | today's global ORE evaluation date | The swaption's "as-of" date. |

### `price_swaptions(hw_paths: jax.Array, step_times: jax.Array, swaption_configs: List[SwaptionConfig]) -> jax.Array`

**Parameters**
- `hw_paths` — `[Scenarios, TimeSteps, NumHW]`, typically
  `generate_paths(...)["rates"]`. Unlike `price_swaps`, this pricer needs the raw
  simulated short-rate paths directly (not the yield-curve cube), since Jamshidian's
  trick needs the model's own conditional bond-price formula, not just pre-tabulated
  discount factors.
- `step_times` — `[TimeSteps]` absolute simulation times (year-fractions from
  `evaluation_date`) — the same values as `config.time_grid[1:]`.
- `swaption_configs` — a list of one or more `SwaptionConfig` objects.

**Returns** `[Scenarios, TimeSteps, Trades]` — one NPV value per scenario, per time step,
per swaption in `swaption_configs` (in the order given). NPV is exactly `0` for any
`(scenario, step)` at or after that swaption's own exercise date (see
[Instruments: European Swaptions](../instruments/european-swaptions.md#6-conditional-future-time-pricing)).

### Lower-level functions

| Function | Signature | Notes |
|---|---|---|
| `prepare_swaption` | `(cfg: SwaptionConfig) -> _PreparedSwaption` | CPU-only, per-trade one-time setup. Builds the real ORE underlying swap and extracts its cashflow times/amounts, exercise time, and accrual start time. |
| `compute_hw_A` | `(zero_times, zero_rates, t, T, a, sigma) -> np.ndarray` | Plain NumPy (CPU-only). Closed-form Hull-White `A(t,T)` at an arbitrary `(t,T)` pair (not a fixed pillar grid) — see [Instruments: European Swaptions](../instruments/european-swaptions.md#3-the-closed-form-building-blocks-compute_hw_a-_hw_b-_bond_option_sigma-_bond_call_bond_put). |

---

## `engine.instruments.bermudan_swaption`

Prices a swaption with a discrete list of exercise dates via a numeric LGM
backward-induction engine (Hagan's Gaussian-quadrature convolution) — see
[Instruments: American & Bermudan Swaptions](../instruments/american-bermudan-swaptions.md) for the full
algorithm.

### `BermudanSwaptionConfig`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `notional` | `float` | *required* | Notional amount of the underlying swap. |
| `fixed_rate` | `float` | *required* | The underlying swap's agreed fixed rate. |
| `payer` | `bool` | *required* | `True` = the option to enter a swap paying fixed, receiving floating. `False` = the reverse. |
| `rate_factor_index` | `int` | *required* | Which simulation Hull-White factor prices the underlying swap and the option. |
| `hw_a` | `float` | *required* | That rate factor's mean-reversion speed — must match the simulation this swaption is priced against. |
| `hw_sigma` | `float` | *required* | That rate factor's volatility — must match the simulation's `joint_covariance` for this factor. |
| `initial_zero_curve` | `ZeroCurveConfig` | *required* | That rate factor's today's-market zero curve. |
| `exercise_times` | `Sequence[float]` | *required* | Year-fractions from `evaluation_date`, each a date the holder may exercise into the (then-remaining) swap — must coincide with the underlying's own reset dates (see [american-bermudan-swaptions.md](../instruments/american-bermudan-swaptions.md#known-limitation-no-mid-coupon-proration)). |
| `swap_tenor` | `str` | `"5Y"` | ORE `Period` string for the underlying swap's length. |
| `index_tenor_months` | `int` | `6` | Floating leg reset frequency in months. |
| `floating_spread` | `float` | `0.0` | Fixed spread added to every floating payment. |
| `n_per_std` | `int` | `48` | State-grid resolution: points per standard deviation of the model's conditional distribution. |
| `std_devs` | `float` | `6.0` | How many standard deviations the state grid spans. |
| `evaluation_date` | `ORE.Date` | today's global ORE evaluation date | The swaption's "as-of" date. |

### `price_bermudan_swaption_base(cfg: BermudanSwaptionConfig) -> float`

The t=0 NPV of a single Bermudan swaption (no simulated conditioning) — read off the
backward induction's own `x=0` node.

### `price_bermudan_swaptions(bermudan_configs: List[BermudanSwaptionConfig], hw_paths: jax.Array, step_times: jax.Array) -> jax.Array`

**Parameters**
- `bermudan_configs` — a list of one or more `BermudanSwaptionConfig` objects.
- `hw_paths` — `[Scenarios, TimeSteps, NumHW]`, typically `generate_paths(...)["rates"]`.
- `step_times` — `[TimeSteps]` absolute simulation times, same values as
  `config.time_grid[1:]`.

**Returns** `[Scenarios, TimeSteps, Trades]` — one NPV value per scenario, per time step,
per trade in `bermudan_configs` (in the order given). NPV is exactly `0` at or after each
trade's own last exercise date.

### Lower-level functions

| Function | Signature | Notes |
|---|---|---|
| `prepare_bermudan` | `(cfg: BermudanSwaptionConfig) -> _PreparedBermudan` | CPU-only, per-trade one-time setup. Extracts both legs' full cashflow schedules (unlike Jamshidian, early exercise needs the actual remaining swap value at every node). |
| `_lgm_bond` | `(zero_times, zero_rates, a, sigma, t, T, x) -> np.ndarray` | Plain NumPy (CPU-only). LGM's own closed-form `P(t,T,x)`, live-verified against `ORE.LinearGaussMarkovModel.discountBond` — deliberately NOT `compute_hw_A`/`_hw_B` (a different model realization for `t>0`, see [american-bermudan-swaptions.md](../instruments/american-bermudan-swaptions.md#3-the-model-lgm-not-plain-hull-white--and-why-that-distinction-matters-here)). |

---

## `engine.instruments.american_swaption`

A thin wrapper around `engine.instruments.bermudan_swaption` — American exercise is
priced by discretizing the exercise window into a dense grid of dates and running the
same Bermudan engine, exactly matching ORE's own design. See
[Instruments: American & Bermudan Swaptions](../instruments/american-bermudan-swaptions.md).

### `AmericanSwaptionConfig`

Same fields as `BermudanSwaptionConfig` above, except `exercise_times` is replaced by:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `first_exercise` | `float` | *required* | Start of the continuous exercise window, year-fractions from `evaluation_date`. |
| `last_exercise` | `float` | *required* | End of the exercise window. |
| `exercise_time_steps_per_year` | `int` | `24` | ORE's own `ExerciseTimeStepsPerYear` model parameter — how finely the window is discretized. |

### `AmericanSwaptionConfig.to_bermudan() -> BermudanSwaptionConfig`

Expands the continuous window into ORE's own discretized exercise-date grid
(`steps = round((last_exercise - first_exercise) * exercise_time_steps_per_year)` equally
spaced dates, including both endpoints).

### `price_american_swaptions(american_configs: List[AmericanSwaptionConfig], hw_paths: jax.Array, step_times: jax.Array) -> jax.Array`

Same parameter/return shape as `price_bermudan_swaptions` above — expands each config via
`.to_bermudan()` and delegates entirely to `bermudan_swaption.price_bermudan_swaptions`.

---

## `engine.risk.statistics`

Every function here is instrument-agnostic — see
[Risk Statistics](../risk/statistics.md) and
[Architecture: stages agree on shapes, not code](../concepts/architecture.md#design-principle-stages-agree-on-shapes-not-code).

### `portfolio_pnl(npv_cube: jax.Array, base_npv: float) -> jax.Array`

**Parameters**
- `npv_cube` — `[Scenarios, TimeSteps, Trades]`, from any pricer.
- `base_npv` — the portfolio's value today (t=0), from a separate zero-shock
  revaluation. See [Risk Statistics: the P&L baseline](../risk/statistics.md#the-pl-baseline-what-are-gainslosses-measured-against).

**Returns** `[Scenarios, TimeSteps]` — `sum(npv_cube, axis=Trades) - base_npv`.

### `value_at_risk(pnl: jax.Array, percentile: float) -> jax.Array`

**Parameters**
- `pnl` — `[Scenarios, TimeSteps]`, typically from `portfolio_pnl`.
- `percentile` — e.g. `0.99` for 99% VaR.

**Returns** `[TimeSteps]` — always `>= 0`.

### `expected_shortfall(pnl: jax.Array, percentile: float) -> jax.Array`

Same signature as `value_at_risk`. **Returns `NaN`** for any time step whose loss tail
(strictly worse than that step's VaR) is empty — see
[Risk Statistics: the formulas](../risk/statistics.md#the-formulas). Callers must check
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
[Architecture: engine/scenarios.py](../concepts/architecture.md#enginescenariospy-shared-example-configurations).

| Name | Type | Meaning |
|---|---|---|
| `EVAL_DATE` | `ORE.Date` | Shared evaluation date for every demo/test scenario in this module. |
| `SWAP_DEMO_MATURITIES` | `List[float]` | The maturity pillars required by `single_currency_swap_demo_config()`'s swap. |
| `cross_asset_demo_config()` | `() -> SimulationConfig` | Two-equity, two-currency (USD/EUR) example scenario. |
| `single_currency_swap_demo_config()` | `() -> SimulationConfig` | One-currency, two-rate-factor (discounting + forwarding) example scenario, sized for a 2Y demo swap. |
| `swaption_demo_config()` | `() -> SimulationConfig` | One rate factor (USD, 3%), simulated out to 5Y in six-month steps -- used by `engine.instruments.european_swaption`'s demo (`rates.maturities` left unset, since the swaption pricer works directly off simulated rate paths rather than a yield-curve cube). |
| `flat_yield_curves(disc_rate, fwd_rate, maturities=SWAP_DEMO_MATURITIES, eval_date=EVAL_DATE)` | `(...) -> jax.Array` | Builds a deterministic `[1, 1, len(maturities), 2]` yield curve cube directly from ORE's own flat curve objects — no simulation randomness. Used for VaR's `base_npv` baseline and for ORE cross-check tests. |
