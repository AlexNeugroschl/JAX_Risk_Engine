# API Reference

Exact inputs and outputs for every public function and configuration dataclass. For the
*why* behind these shapes, see the per-stage deep dives
([Market Simulation](../concepts/market-simulation.md), [Instruments](../instruments/swaps.md),
[Risk Statistics](../risk/var_es.md)). For runnable examples, see the
[User Guide](../getting-started/user-guide.md).

**Notation:** `[Scenarios, TimeSteps, ...]` describes an array's shape. `Scenarios` is
however many simulated alternate futures were requested; `TimeSteps` is
`len(time_grid) - 1` (the simulation's output steps are the points *after* time zero, not
including time zero itself).

---

## `engine.simulation.market_model`

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
| `provenance` | `Optional[CurveProvenance]` = `None` | **Metadata only — never read by the simulation math.** Where these numbers came from (`curveId`, `inputOrigin` ∈ `observed\|assumed\|mixed\|synthetic`, `construction`, `inputHashes`). Added by W0.6 so assumed and observed curves are distinguishable *inside* the engine, not only at its edges; see [EOD Integration: W0.6](eod-integration.md#w06--market-input-selection--closes-part-of-i-11). `None` means **unstated**, which is deliberately not the same as `observed`. |

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

### `validate_joint_covariance(matrix: List[List[float]] | np.ndarray) -> None`

Called at the very top of `generate_paths`, before any JAX computation. Confirms `matrix`
is square, symmetric (within float tolerance), and positive semi-definite (via
`np.linalg.eigvalsh`). **Raises** `ValueError` naming the offending eigenvalue(s)/index, or
the symmetry mismatch, rather than letting an invalid matrix silently NaN every simulated
path through `jnp.linalg.cholesky` — see
[docs/planning/traderx-integration.md](../planning/traderX_integration/traderx-integration.md#1-joint_covariance-psd-validationrepair-highest-priority).

### `nearest_psd(matrix, epsilon: float = 1e-10) -> np.ndarray`

Opt-in "best effort" repair for a `joint_covariance` that fails `validate_joint_covariance`
— standard eigenvalue-clipping projection onto the nearest PSD matrix, clipping every
eigenvalue below `epsilon` up to `epsilon` (not literally `0.0` — see the function's own
docstring for why an exact-zero clip produces a numerically rank-deficient matrix that
would still NaN downstream). **Never called automatically** by `generate_paths` or
`engine.portfolio` — a genuinely invalid correlation input is always rejected, never
silently altered and priced anyway; a caller who wants "close enough" behavior calls this
explicitly.

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

**Validated at construction (`__post_init__`)**: `notional`/`fixed_rate` must be finite
(zero and negative values are explicitly supported — only `NaN`/`Inf` are rejected);
`swap_tenor` must parse as a valid `ORE.Period`. Raises `ValueError` naming the bad field.
This is deliberately scoped to reject malformed input, not impose business-rule limits
(e.g. no "no rate above 20%" check) — see
[docs/planning/traderx-integration.md](../planning/traderX_integration/traderx-integration.md#4-trade-level-input-validation-notional-rate-ranges-tenor-sanity).

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

**Validated at construction (`__post_init__`)**: same `notional`/`fixed_rate`/`swap_tenor`
checks as `SwapConfig`, plus `hw_sigma` must be finite.

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
| `exercise_dates` | `Sequence[ORE.Date]` | *required* | Ascending dates on which the holder may exercise into the (then-remaining) swap. Dates on or before `evaluation_date` are not exercise opportunities. A date inside an accrual period exercises into the next whole period, as in ORE (see [american-bermudan-swaptions.md](../instruments/american-bermudan-swaptions.md#which-coupons-an-exercise-enters)); `exercisable_dates(cfg)` lists the underlying's accrual starts. |
| `swap_tenor` | `str` | `"5Y"` | ORE `Period` string for the underlying swap's length. |
| `index_tenor_months` | `int` | `6` | Floating leg reset frequency in months. |
| `floating_spread` | `float` | `0.0` | Fixed spread added to every floating payment. |
| `n_per_std` | `int` | `48` | State-grid resolution: points per standard deviation of the model's conditional distribution. |
| `std_devs` | `float` | `6.0` | How many standard deviations the state grid spans. |
| `evaluation_date` | `ORE.Date` | today's global ORE evaluation date | The swaption's "as-of" date. |

**Validated at construction (`__post_init__`)**: same `notional`/`fixed_rate`/`swap_tenor`
checks as `SwapConfig`; `hw_sigma` (a plain `float` or a piecewise `Sigma` — every bucket
value is checked) must be finite, **or exactly `None`** (a valid sentinel meaning
"uncalibrated" — see [The Portfolio Entry Point: Automatic calibration](portfolio-entrypoint.md#automatic-calibration));
`exercise_dates` must be non-empty, sorted ascending, and `ORE.Date` objects (a year
fraction is refused with `TypeError`).

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
| `prepare_bermudan` | `(cfg: BermudanSwaptionConfig \| AmericanSwaptionConfig) -> _PreparedBermudan` | CPU-only, per-trade one-time setup. Resolves ORE's option times and, per coupon, ORE's `CashflowInfo` (pay/accrual times, belongs-until time by exercise style, the floating coupon's index fixing period). |
| `exercisable_dates` | `(cfg) -> List[ORE.Date]` | The underlying's own fixed accrual start dates -- the exercise dates of a standard coterminal Bermudan. |
| `_lgm_bond` | `(zero_times, zero_rates, a, sigma, t, T, x) -> np.ndarray` | Plain NumPy (CPU-only). LGM's own closed-form `P(t,T,x)`, live-verified against `ORE.LinearGaussMarkovModel.discountBond` — deliberately NOT `compute_hw_A`/`_hw_B` (a different model realization for `t>0`, see [american-bermudan-swaptions.md](../instruments/american-bermudan-swaptions.md#3-the-model-lgm-not-plain-hull-white--and-why-that-distinction-matters-here)). |

---

## `engine.instruments.american_swaption`

American exercise, priced by the same backward induction as a Bermudan. It differs in
exactly the two places ORE's engine does: ORE's uniform option-time grid over the window
(truncating step count), and broken-period exercise, where each coupon belongs until its
accrual end and is credited `couponRatio(t)`. See
[Instruments: American & Bermudan Swaptions](../instruments/american-bermudan-swaptions.md#american-exercise-ores-grid-and-ores-broken-periods).

### `AmericanSwaptionConfig`

Same fields as `BermudanSwaptionConfig` above, except `exercise_dates` is replaced by:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `first_exercise_date` | `ORE.Date` | *required* | First day of the exercise window. A window already open starts at `t = 0`. |
| `last_exercise_date` | `ORE.Date` | *required* | Last day of the exercise window. |
| `exercise_time_steps_per_year` | `int` | `24` | ORE's own `ExerciseTimeStepsPerYear` model parameter — how finely the window is discretized. |

**Validated at construction (`__post_init__`)**: same `notional`/`fixed_rate`/`swap_tenor`/
`hw_sigma`(-or-`None`) checks as `BermudanSwaptionConfig`, plus `first_exercise_date <=
last_exercise_date` (equal dates — a zero-width window — are valid) and
`exercise_time_steps_per_year >= 1`.

### `AmericanSwaptionConfig.option_times() -> List[float]`

ORE's own American option times: `t1 = max(0, t(first))`, `t2 = max(t1, t(last))`,
`steps = max(1, floor((t2 - t1) * exercise_time_steps_per_year))` (ORE truncates), and the
times `t1 + i * (t2 - t1) / steps` for `i = 0..steps`.

### `price_american_swaptions(american_configs: List[AmericanSwaptionConfig], hw_paths: jax.Array, step_times: jax.Array) -> jax.Array`

Same parameter/return shape as `price_bermudan_swaptions` above, which it calls directly:
that function prices either config type.

---

## `engine.instruments.treasury`

W1.5. Treasury bills and notes, priced by **closed-form discounted cashflows against a single
deterministic curve**. The odd one out in this package in two ways: it is the only pricer that
is **not JAX** (plain `math.exp` over ORE day-count arithmetic), and the only one that produces
**no NPV cube** — so no VaR/ES. See
[The Portfolio Entry Point: Bonds](portfolio-entrypoint.md#bonds) and
[I-24](../known-issues.md#i-24).

### `CouponPeriod`

One explicit coupon period. **Supplied, never generated** — a schedule derived by stepping
back from maturity that disagreed with the booked one would silently reprice every coupon.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `start_date` | `ORE.Date` | *required* | Accrual period start. |
| `end_date` | `ORE.Date` | *required* | Accrual period end. |
| `payment_date` | `Optional[ORE.Date]` | `None` | Payment date; falls back to `end_date`. Read via the `.payment()` method. |

### `BondConfig`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `face_amount` | `float` | *required* | **Signed** — a short position is a negative face and yields a negative NPV directly. There is no separate sign factor, and applying one on top would flip a short position positive. |
| `maturity_date` | `ORE.Date` | *required* | Must be strictly after `evaluation_date`, else `BondPricingError` — a matured bond is a settlement question, not a pricing one. |
| `evaluation_date` | `ORE.Date` | *required* | Valuation date. |
| `initial_zero_curve` | `ZeroCurveConfig` | *required* | **This bond's own curve**, not an index into `SimulationConfig.rates.initial_zero_curves` — the same shape the swaption family uses, and the reason a bond cannot reproduce [I-01](../known-issues.md#i-01). |
| `coupon_rate` | `float` | `0.0` | Annual coupon as a **decimal** (`0.04` == 4%), not a percent. |
| `coupon_schedule` | `Tuple[CouponPeriod, ...]` | `()` | Empty ⇒ this is a **bill** (the degenerate zero-coupon case, not a separate type). |
| `redemption_fraction` | `float` | `1.0` | Redemption as a fraction of face. Must be non-negative. |
| `accrual_day_count` | `str` | `"ACT/ACT (ICMA)"` | The instrument's own **accrual** day count (W1.1), resolved at construction and **refused if unsupported** — never defaulted. Ignored when there are no coupons. |

**Properties:** `is_bill` (no coupon schedule), `notional` (alias for `face_amount`, so the
shared `trade[i] (Type, notional=…)` labelling helpers work without a special case).

**Construction refuses four contradictions**, each naming the reason: a maturity at or before
the evaluation date; a negative `redemption_fraction`; a non-empty schedule with
`coupon_rate=0.0`; and a non-zero `coupon_rate` with no schedule. The day count is resolved
**eagerly**, so an unsupported convention fails at construction naming the trade rather than
mid-pricing.

### Discounting convention

**Continuously compounded over an ACT/365 Fixed year fraction**, matching
`engine.integration.bill`/`note` exactly. The curve's zero rate is **linearly interpolated
between pillars and held flat beyond both ends** — flat extrapolation is stated rather than
assumed, because extrapolating a slope past the last pillar produces a confident number from
no data, and for a long bond that error compounds through every discount factor.

### `price_bond_base(cfg: BondConfig, rate_shift: float = 0.0) -> float`

t=0 **dirty** (full) NPV — every remaining cashflow discounted. Dirty rather than clean
deliberately, matching `engine.integration.note.NotePrice.npv`: a "bond NPV" that silently
meant the clean value would be off by the accrued interest (~$1,857 on a $100k note), which
is large enough to matter and small enough to look like a curve difference. A coupon already
paid on or before the evaluation date is excluded, not discounted from the past.
`rate_shift` parallel-shifts the curve.

### `accrued_interest(cfg: BondConfig) -> float`

Accrued interest in currency, recomputed from the coupon schedule and position-signed via
`face_amount`. Returns `0.0` for a bill and `0.0` before the first period starts — both
structural facts, not missing values. This is the **`recomputed-schedule` path only**; the
integration boundary additionally reconciles against the exporter's published fraction and
reports that one (see [EOD Integration](eod-integration.md)).

### `clean_npv_of(cfg: BondConfig) -> float`

`price_bond_base(cfg) - accrued_interest(cfg)`. Carried because a quoted bond price is
conventionally clean, so a consumer reconciling against a market quote compares like with
like.

### `rate_sensitivity(cfg: BondConfig, bump: float = RATE_BUMP) -> float`

Change in dirty NPV for a `bump` (default `RATE_BUMP = 1e-4`, i.e. 1bp) parallel curve shift.
A **bumped revaluation through the same code path**, not a differentiated formula — a
sensitivity derived from an expression that has drifted from the pricer measures the
expression, not the price. Parallel-only ([I-16](../known-issues.md#i-16)).

### `price_bond_scenarios(*args, **kwargs)`

**Always raises `ScenarioPricingNotSupported`.** It exists so the refusal has a name and a
docstring where a contributor would look for the missing capability, rather than being an
absence someone fills in with a broadcast. Filling it in naively produces a zero-variance
column measuring out to **VaR `0.00` and ES `NaN`** — a position reported as risk-measured
when its risk was never modelled ([I-24](../known-issues.md#i-24)).

### Exceptions

| Exception | Raised when |
|---|---|
| `BondPricingError` | A bond could not be priced, or a `BondConfig` is self-contradictory. Distinct from `ValueError` so a caller can tell a *pricing* refusal from a malformed request. |
| `ScenarioPricingNotSupported` | A `BondConfig` reached the scenario/`npv_cube` path. Subclass of `BondPricingError`. |

---

## `engine.risk.var_es`

Every function here is instrument-agnostic — see
[Risk Statistics](../risk/var_es.md) and
[Architecture: modules agree on shapes, not code](../concepts/architecture.md#design-principle-modules-agree-on-shapes-not-code).

### `portfolio_pnl(npv_cube: jax.Array, base_npv: float) -> jax.Array`

**Parameters**
- `npv_cube` — `[Scenarios, TimeSteps, Trades]`, from any pricer.
- `base_npv` — the portfolio's value today (t=0), from a separate zero-shock
  revaluation. See [Risk Statistics: the P&L baseline](../risk/var_es.md#the-pl-baseline-what-are-gainslosses-measured-against).

**Returns** `[Scenarios, TimeSteps]` — `sum(npv_cube, axis=Trades) - base_npv`.

### `value_at_risk(pnl: jax.Array, percentile: float) -> jax.Array`

**Parameters**
- `pnl` — `[Scenarios, TimeSteps]`, typically from `portfolio_pnl`.
- `percentile` — e.g. `0.99` for 99% VaR.

**Returns** `[TimeSteps]` — always `>= 0`.

### `expected_shortfall(pnl: jax.Array, percentile: float) -> jax.Array`

Same signature as `value_at_risk`. **Returns `NaN`** for any time step whose loss tail
(strictly worse than that step's VaR) is empty — see
[Risk Statistics: the formulas](../risk/var_es.md#the-formulas). Callers must check
for this explicitly.

### `compute_risk_metrics(npv_cube, base_npv, percentiles=(0.95, 0.99), include_diagnostics=True) -> Dict[str, jax.Array]`

The main entry point — combines the three functions above.

**Parameters**
- `npv_cube` — `[Scenarios, TimeSteps, Trades]`.
- `base_npv` — see `portfolio_pnl` above.
- `percentiles` — which confidence levels to compute VaR/ES at. Default `(0.95, 0.99)`.
- `include_diagnostics` — attach per-percentile convergence diagnostics. Default `True`;
  `False` returns exactly the pre-W0.6 key set.

**Returns** a `dict`, every value shaped `[TimeSteps]`, with these keys per entry in
`percentiles` (shown for `0.99`):

| Key | Meaning |
|---|---|
| `VaR_99` | Value at Risk. |
| `ES_99` | Expected Shortfall. NaN where the strict tail is empty. |
| `ES_99_tailCount` | **Effective sample size** — how many observations the ES mean actually averaged. At 99% over 10,000 scenarios this is ~100, so the estimate rests on 1% of the sample. |
| `ES_99_standardError` | Monte Carlo standard error of the ES mean (`s/√n`, `ddof=1`). **NaN, never `0.0`, when `n < 2`** — `0.0` would read as "perfectly converged" for the least trustworthy case. |

The two diagnostic keys are additive (W0.6, part of [I-11](../known-issues.md#i-11)); the
`VaR_*`/`ES_*` keys and values are unchanged. See
[EOD Integration: tail diagnostics](eod-integration.md#tail-statistics-carry-their-own-convergence-diagnostics).

### Risk measure constants

`RISK_MEASURES` = `("risk-neutral-pricing", "historical-forecast", "deterministic-stress")`,
and `ENGINE_RISK_MEASURE` = `"risk-neutral-pricing"` — what this engine actually produces. A
risk-neutral exposure is **not** a calibrated forecast of tomorrow's loss; reporting one where
the other is expected is a category error no numerical accuracy fixes.

---

## `engine.risk.greeks`

Delta, Gamma, and Theta for `engine.instruments.swap` and
`engine.instruments.european_swaption` only — see [Delta, Gamma, and Theta](../risk/greeks.md)
for the full explanation, including why Bermudan/American swaptions and Vega are out of
scope.

### `ZeroCurve`

A zero curve as JAX arrays (unlike `engine.simulation.market_model.ZeroCurveConfig`, whose `rates` is
a plain Python list) — the differentiable input every Greek in this module is computed
with respect to.

| Field | Type | Meaning |
|---|---|---|
| `pillar_times` | `jax.Array` | Zero-curve pillar times. Not differentiated (ORE never bumps a pillar's own time, only its rate). |
| `pillar_rates` | `jax.Array` | Zero rate at each pillar — what Delta/Gamma differentiate with respect to. |

`ZeroCurve.flat(rate: float, pillar_times: List[float]) -> ZeroCurve` — a convenience
constructor for a flat curve (the same rate at every pillar).

### `swap_delta_gamma(cfg: SwapConfig, disc_curve: ZeroCurve, fwd_curve: ZeroCurve, bump_size: float = DEFAULT_RATE_BUMP) -> Dict[str, jax.Array]`

**Parameters**
- `cfg` — a `SwapConfig` (see `engine.instruments.swap` above). Its own
  `discount_curve_index`/`forward_curve_index` are ignored — internally, `disc_curve` and
  `fwd_curve` always play those two roles respectively.
- `disc_curve`/`fwd_curve` — the swap's discount and forward `ZeroCurve`s. Pass the same
  object for both to compute single-curve-discounting Greeks (see Returns below).
- `bump_size` — the zero-rate move each unit of Delta/Gamma represents, matching ORE's
  own 1bp (`0.0001`) default.

**Returns** a `dict`: `"discount_delta"`, `"discount_gamma"` (w.r.t. `disc_curve.pillar_rates`)
and `"forward_delta"`, `"forward_gamma"` (w.r.t. `fwd_curve.pillar_rates`), each shaped
`[len(pillar_rates)]`. If `disc_curve is fwd_curve`, summing `discount_delta +
forward_delta` pillar-by-pillar recovers the total sensitivity to that one shared curve.

### `swap_theta(cfg: SwapConfig, disc_curve: ZeroCurve, fwd_curve: ZeroCurve, theta_days: int = DEFAULT_THETA_DAYS) -> float`

**Returns** a single number: `NPV(today + theta_days, same curves) − NPV(today) +`
cashflow paid in between (ORE's own Theta definition — see
[Delta, Gamma, and Theta: Theta](../risk/greeks.md#theta-advance-the-evaluation-date-hold-the-market-fixed)).

### `swaption_delta_gamma(cfg: SwaptionConfig, curve: ZeroCurve, bump_size: float = DEFAULT_RATE_BUMP) -> Dict[str, jax.Array]`

**Parameters**
- `cfg` — a `SwaptionConfig` (see `engine.instruments.european_swaption` above).
- `curve` — the swaption's Hull-White calibration curve, as a `ZeroCurve`. Should carry
  the same pillar times/rates as `cfg.initial_zero_curve` (this function does not read
  `cfg.initial_zero_curve` itself, since it's a plain-Python `ZeroCurveConfig`, not JAX
  array).
- `bump_size` — same meaning as `swap_delta_gamma` above.

**Returns** a `dict`: `"delta"`, `"gamma"`, each shaped `[len(curve.pillar_rates)]`.

### `swaption_theta(cfg: SwaptionConfig, curve: ZeroCurve, theta_days: int = DEFAULT_THETA_DAYS) -> float`

**Returns** a single number: `NPV(today + theta_days, same curve) − NPV(today)` — no
interim-cashflow term (a European swaption pays no cashflow before its own exercise date).

### `bermudan_delta_gamma(cfg: BermudanSwaptionConfig, curve: ZeroCurve, bump_size: float = DEFAULT_RATE_BUMP) -> Dict[str, jax.Array]`

Per-pillar Delta and Gamma of one Bermudan/American swaption's t=0 NPV with respect to its
own LGM calibration curve — same signature/return shape and same `curve`-shares-
`cfg.initial_zero_curve`'s-pillar-times convention as `swaption_delta_gamma` above. Works
for an `AmericanSwaptionConfig` passed directly (see
[American & Bermudan Swaptions](../instruments/american-bermudan-swaptions.md)) — there is
no separate `american_delta_gamma` function.

**Returns** a `dict`: `"delta"`, `"gamma"`, each shaped `[len(curve.pillar_rates)]`.

### `bermudan_theta(cfg: BermudanSwaptionConfig, curve: ZeroCurve, theta_days: int = DEFAULT_THETA_DAYS) -> float`

**Returns** a single number: `NPV(today + theta_days, same curve) − NPV(today)` — no
interim-cashflow term, same reasoning as `swaption_theta` (see
[Delta, Gamma, and Theta: Theta](../risk/greeks.md#theta-advance-the-evaluation-date-hold-the-market-fixed)).

### `bermudan_vega(cfg: BermudanSwaptionConfig, curve: ZeroCurve, calibration_targets: List[CalibrationTarget], market_vol_bump: float = 0.0001) -> jax.Array`

Per-basket-instrument Vega: dollar NPV change for a `market_vol_bump` (1bp of normal vol by
default) move in **one** market swaption's own quoted volatility, holding every other
market quote fixed — computed via the implicit function theorem through
`engine.calibration.lgm.calibrate_lgm_sigma`'s own bootstrap root-find, not by literally
re-running calibration once per bumped market vol (see
[Delta, Gamma, and Theta: Vega](../risk/greeks.md#vega-bermudanamerican-only) for the full
derivation, including the cross-bucket Jacobian term a naive first attempt missed).

**Parameters**
- `cfg.hw_sigma` **must** be the exact `Sigma` `calibrate_lgm_sigma` produced from
  `calibration_targets` (same order) — this function differentiates through that
  relationship, it does not re-run calibration itself.

**Returns** `[len(calibration_targets)]` — Vega to each basket instrument's own market vol,
in dollars per `market_vol_bump`.

### Constants

| Name | Value | Meaning |
|---|---|---|
| `DEFAULT_RATE_BUMP` | `0.0001` | ORE's own example-config default: 1 basis point, absolute. |
| `DEFAULT_THETA_DAYS` | `1` | ORE's own default Theta horizon: 1 calendar day. |

---

## `engine.calibration.basket`

Co-terminal calibration basket construction and LGM's own closed-form swaption pricer used
during calibration — see [Calibration](calibration.md) for the full algorithm and its
two-route verification against ORE.

### `CalibrationTarget`

One co-terminal European swaption calibration target.

| Field | Type | Meaning |
|---|---|---|
| `expiry_time` | `float` | T0, year-fraction from `evaluation_date`. |
| `accrual_start_time` | `float` | The underlying's own first accrual date. |
| `fixed_cashflow_times` | `np.ndarray` | `[N]` year-fractions from `evaluation_date`. |
| `fixed_cashflow_amounts` | `np.ndarray` | `[N]`, at the ATM fixed rate. |
| `fixed_accrual_fractions` | `np.ndarray` | `[N]`, ORE's own `accrualPeriod()` per coupon. |
| `notional` | `float` | |
| `payer` | `bool` | |
| `forward_rate` | `float` | The underlying's own par/ATM rate (== strike, by construction). |
| `market_vol` | `float` | Normal (basis-point) volatility. |

### `build_coterminal_basket(exercise_times, final_maturity_time, notional, payer, market_vols, zero_curve, evaluation_date, index_tenor_months=6) -> List[CalibrationTarget]`

Builds one co-terminal `CalibrationTarget` per exercise date — see
[Calibration: The co-terminal calibration basket](calibration.md#the-co-terminal-calibration-basket).

### `price_lgm_swaption(curve: ZeroCurve, a: float, sigma: Union[float, Sigma], target: CalibrationTarget) -> jax.Array`

t=0 NPV of one co-terminal European swaption under LGM, for a trial `(a, sigma)` — the
model-price half of calibration's error function. Differentiable end-to-end in `sigma`
(and `a`) via an implicit-function-theorem-corrected bisection — see
[Calibration: the `_bisect_xstar` gradient bug](calibration.md#the-_bisect_xstar-gradient-bug).

### `bachelier_swaption_price(target: CalibrationTarget, curve: ZeroCurve) -> jax.Array`

Market price implied by `target.market_vol` via the Bachelier (normal) formula — the
target `price_lgm_swaption` is calibrated to match.

---

## `engine.calibration.lgm`

### `CalibrationResult`

Output of `calibrate_lgm_sigma`.

| Field | Type | Meaning |
|---|---|---|
| `sigma` | `Sigma` | The fitted piecewise volatility term structure. |
| `market_prices` | `jax.Array` | `[N]`, Bachelier price implied by each target's `market_vol`. |
| `model_prices` | `jax.Array` | `[N]`, `price_lgm_swaption` at the final calibrated `Sigma`. |
| `rmse` | `float` | `sqrt(mean((model-market)^2))` — ORE's own `error_` metric. |

### `calibrate_lgm_sigma(targets: List[CalibrationTarget], curve: ZeroCurve, a: float) -> CalibrationResult`

Bootstrap-calibrates a piecewise `Sigma` to `targets` (in increasing expiry order,
asserted explicitly) — see [Calibration: Why bootstrap, not joint least-squares](calibration.md#why-bootstrap-not-joint-least-squares)
for the full algorithm. `a` (mean reversion) is always an input, never calibrated (see
[Calibration: Why mean reversion is never calibrated](calibration.md#why-mean-reversion-is-never-calibrated)).

**Raises** an `AssertionError` if `targets` is empty or not in increasing expiry order.

---

## `engine.models.hull_white`

The single source of truth for this codebase's Hull-White 1-Factor closed-form math — see
[Models & Trades](models-and-trades.md#enginemodelshull_whitepy).

| Name | Signature | Notes |
|---|---|---|
| `ZeroCurve` | dataclass: `pillar_times: jax.Array`, `pillar_rates: jax.Array` | JAX-array counterpart of `engine.simulation.market_model.ZeroCurveConfig`, differentiable via `jnp.interp`. `ZeroCurve.flat(rate, pillar_times)` is a convenience constructor. |
| `B` | `(t, T, a) -> jax.Array` | `B(t,T) = (1-exp(-a*(T-t)))/a`, with the `a==0` limit handled explicitly. |
| `A` | `(curve, t, T, a, sigma, B_override=None) -> jax.Array` | `A(t,T)` calibrated to `curve`'s own market. |
| `discount` | `(curve, t) -> jax.Array` | `P(0,t) = exp(-zero_rate(t)*t)`. |
| `zero_rate` | `(curve, t) -> jax.Array` | Linear interpolation on `curve`'s own zero rates, flat-extrapolated at the ends. |
| `log_discount` | `(curve, t) -> jax.Array` | `ln P(0,t)`. |
| `bond_option_sigma` | `(T0, T, t, a, sigma) -> jax.Array` | The bond-option volatility term Jamshidian's formula needs. |
| `bond_call` / `bond_put` | `(P_t_T0, P_t_Ti, K, sigma_p) -> jax.Array` | Black-on-bond formulas. |

---

## `engine.models.lgm`

Linear Gauss-Markov closed-form math (piecewise-constant `Sigma`) — the model
`bermudan_swaption.py`/`american_swaption.py` and `engine/calibration/` are built on. See
[Models & Trades](models-and-trades.md#enginemodelslgmpy) for why this is **not**
interchangeable with `engine.models.hull_white` for `t>0`, despite sharing `(a, sigma)` and
today's curve.

| Name | Signature | Notes |
|---|---|---|
| `Sigma` | dataclass (JAX pytree): `times: jax.Array`, `values: jax.Array` | Piecewise-constant vol; `len(values) == len(times) + 1`. `Sigma.flat(sigma)` builds a one-bucket flat `Sigma`. |
| `as_sigma` | `(sigma: float \| jax.Array \| Sigma) -> Sigma` | Upgrades a plain scalar to a one-bucket `Sigma`; the normalization point every function below calls first. |
| `H` / `H_prime` | `(a, t) -> jax.Array` | LGM's own state-space mapping function and its derivative. |
| `zeta` | `(sigma, t) -> jax.Array` | Cumulative variance; accepts `sigma` as `Sigma` or scalar. |
| `bond_price` | `(curve, a, sigma, t, T, x) -> jax.Array` | `P(t,T,x)`, live-verified against `ORE.LinearGaussMarkovModel.discountBond`. |
| `numeraire` | `(curve, a, sigma, t, x) -> jax.Array` | LGM's own numeraire — required for correct martingale-measure discounting (see [Calibration](calibration.md#two-route-verification)). |
| `bond_option_sigma` | `(a, sigma, T0, T, t) -> jax.Array` | LGM analogue of the Hull-White version above. |
| `r_from_x` / `x_from_r` | `(curve, a, sigma, t, x_or_r) -> jax.Array` | Converts between LGM's own state variable `x` and the direct short rate `r` (affine, exact). |

---

## `engine.models.ore_builders`

Shared ORE trade-building and cashflow-extraction helpers — the single source of truth for
turning a trade config into a real ORE object and its cashflow schedule, used by every
pricer in `engine/instruments/`. See
[Models & Trades](models-and-trades.md#enginemodelsore_builderspy).

| Name | Signature | Notes |
|---|---|---|
| `DAY_COUNTER` | `ORE.Actual365Fixed()` | The single day-count convention used throughout this codebase, on both legs of every trade. |
| `build_vanilla_swap` | `(notional, fixed_rate, payer, swap_tenor, index_tenor_months, floating_spread, evaluation_date, forward_start=None) -> ORE.VanillaSwap` | Builds a real ORE swap via `ORE.MakeVanillaSwap`. |
| `LegCashflows` | dataclass: `payment_times`, `accrual_start_times`, `accrual_end_times`, `accrual_fractions` (each `np.ndarray`), `notional: float` | One leg's cashflow schedule, as year-fractions from `today`. |
| `fixed_leg_cashflows` / `floating_leg_cashflows` | `(swap, today) -> LegCashflows` | Extracts each coupon's dates/accrual fraction from a real ORE-generated schedule. |

---

## `engine.portfolio`

The top-level entry point tying every module above together — see
[The Portfolio Entry Point](portfolio-entrypoint.md) for the full write-up (this table is
the field-level quick reference; that doc explains the *why*).

### `PortfolioRequest`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `market` | `SimulationConfig` | *required* | Curves, vols, correlations. |
| `trades` | `List[SwapConfig \| SwaptionConfig \| BermudanSwaptionConfig \| AmericanSwaptionConfig]` | *required* | Heterogeneous, any order/mix. |
| `percentiles` | `Sequence[float]` | `(0.95, 0.99)` | |
| `calibration_targets` | `Optional[List[CalibrationTarget]]` | `None` | Used when any Bermudan/American trade's `hw_sigma is None`. |
| `compute_greeks` | `bool` | `False` | |

### `PortfolioResult`

| Field | Type | Meaning |
|---|---|---|
| `base_npv` | `float` | |
| `npv_cube` | `jax.Array` | `[Scenarios, TimeSteps, Trades]`, caller's own `trades` order. |
| `risk` | `Dict[str, jax.Array]` | `compute_risk_metrics`'s own output. |
| `greeks` | `Optional[Dict[int, Dict[str, jax.Array]]]` | Keyed by trade index in `request.trades`. |
| `warnings` | `List[str]` | Known-limitation warnings surfaced during validation. |

### `price_portfolio(request: PortfolioRequest) -> PortfolioResult`

The main entry point — see [The Portfolio Entry Point](portfolio-entrypoint.md#price_portfoliorequest-portfoliorequest---portfolioresult)
for the full 9-step orchestration.

### `validate_portfolio_against_simulation(sim_config: SimulationConfig, trade_configs: Sequence[...]) -> None`

Cross-checks every trade's duplicated `hw_a`/`hw_sigma`/`initial_zero_curve` against
`sim_config`. **Raises** `ValueError` naming the trade and the mismatched field. See
[The Portfolio Entry Point](portfolio-entrypoint.md#validate_portfolio_against_simulationsim_config-trade_configs---none).

### `derive_maturity_pillars(trade_configs: Sequence[...], evaluation_date: ORE.Date) -> List[float]`

Automatic maturity-pillar assembly from every `SwapConfig`'s real ORE schedule. See
[The Portfolio Entry Point](portfolio-entrypoint.md#derive_maturity_pillarstrade_configs-evaluation_date---listfloat).

---

## `engine.simulation.demo_scenarios`

Reference/demo configurations and shared test helpers — not part of the pricing
pipeline itself, but used throughout the codebase's demos and tests. See
[Architecture: engine/simulation/demo_scenarios.py](../concepts/architecture.md#enginesimulationdemo_scenariospy-shared-example-configurations).

| Name | Type | Meaning |
|---|---|---|
| `EVAL_DATE` | `ORE.Date` | Shared evaluation date for every demo/test scenario in this module. |
| `SWAP_DEMO_MATURITIES` | `List[float]` | The maturity pillars required by `single_currency_swap_demo_config()`'s swap. |
| `cross_asset_demo_config()` | `() -> SimulationConfig` | Two-equity, two-currency (USD/EUR) example scenario. |
| `single_currency_swap_demo_config()` | `() -> SimulationConfig` | One-currency, two-rate-factor (discounting + forwarding) example scenario, sized for a 2Y demo swap. |
| `swaption_demo_config()` | `() -> SimulationConfig` | One rate factor (USD, 3%), simulated out to 5Y in six-month steps -- used by `engine.instruments.european_swaption`'s demo (`rates.maturities` left unset, since the swaption pricer works directly off simulated rate paths rather than a yield-curve cube). |
| `flat_yield_curves(disc_rate, fwd_rate, maturities=SWAP_DEMO_MATURITIES, eval_date=EVAL_DATE)` | `(...) -> jax.Array` | Builds a deterministic `[1, 1, len(maturities), 2]` yield curve cube directly from ORE's own flat curve objects — no simulation randomness. Used for VaR's `base_npv` baseline and for ORE cross-check tests. |

---

## `engine.api`

The HTTP boundary over `engine.portfolio.price_portfolio` — see [HTTP API](http-api.md)
for the full endpoint reference, the sync-vs-async job pattern's reasoning, and request/
response schema tables. Requires the `api` optional-dependency extra
(`pip install -e .[api]`) — `engine.portfolio` and everything below it has zero dependency
on this package or its own dependencies (FastAPI, Pydantic, uvicorn).

| Module | Contents |
|---|---|
| `engine.api.app` | `create_app() -> FastAPI` / `app` — the FastAPI application factory. Run with `uvicorn engine.api.app:app`. |
| `engine.api.routes` | `router: APIRouter` — `GET /health`, `GET /version`, `POST /portfolio/price`, `GET /portfolio/price/{job_id}`, `POST /calibration/lgm`. |
| `engine.api.eod_routes` | `router: APIRouter` (prefix `/eod`) — the TraderX EOD contract: `GET /eod/capabilities`, `GET /eod/schemas/result`, `GET /eod/schemas/capabilities`, `POST /eod/price`, `GET /eod/results/by-workload/{key}`, `GET /eod/attempts/{attemptId}`. Returns **plain dicts**, not Pydantic models — the contract is the published JSON Schema, and a second definition could drift from it. See [EOD Integration](eod-integration.md#w164--the-eod-http-routes). |
| `engine.api.schemas` | Pydantic v2 models mirroring `engine.portfolio`/`engine/instruments/*.py`'s dataclasses field-for-field, each with `.to_dataclass()`/`.from_dataclass()` — see [HTTP API: Request/response schemas](http-api.md#request-schema-portfoliorequestschema). |

---

## `engine.integration`

The TraderX EOD boundary: a hash-verified bundle in, an **identified** risk result out. Its
governing rule is that **nothing is ever silently approximated** — an explicit `unsupported`
is recoverable, a plausible wrong number is not.

Documented in full in [The EOD Integration Boundary](eod-integration.md); this table is the
module index. **This package imports no FastAPI, no Pydantic, no JAX, and no simulation
pricer** — only `ORE`, for date and day-count arithmetic. The dependency runs
`engine.api` → `engine.integration`, never the reverse, and it is enforced by
`tests/test_integration_pipeline.py::TestPackageImportsNoSimulationPricer`.

| Module | Task | Contents |
|---|---|---|
| `bundle` | W0.1 | `load_bundle(root) -> Bundle`, `Bundle`, `BundleIntegrityError`. Reads and hash-verifies every artifact **in binary, exactly as read** — see the CRLF trap. |
| `terms` | W0.2 / W1.6.1 | `join_terms`, `TermsEntry`, `AccrualBasis`, `TermsJoinError`, `SUPPORTED_TERMS_SCHEMAS`. |
| `normalize` | W0.3 | `normalize_position`, `NormalizedPosition`, `Quantity`, `MAPPING_VERSION`. |
| `conventions` | W0.4 | `check_conventions`, `ConventionRefusal` — a positive **allowlist**, applied before any pricing object is constructed ([I-05](../known-issues.md#i-05)). |
| `result` | W0.5 | `RiskResult`, `ItemResult`, `Coverage`, `CALCULATIONS`, `STATUSES`. |
| `market_inputs` | W0.6 | `resolve_market_inputs`, `MarketInputs`, `MarketInputsNotSupplied`, `ASSUMED_PROFILES`, `CurveProvenance`, `ENGINE_RISK_MEASURE`. No silent fallback ([I-11](../known-issues.md#i-11)). |
| `identity` | W0.7 | `item_id`, `ItemIdentity` — an opaque, stable id plus the source identity block, on **every** row including refusals ([I-10](../known-issues.md#i-10)). |
| `publication` | W0.8 | `ResultStore`, `PublicationError`, `default_store_root` — the crash-safe durable result store ([I-08](../known-issues.md#i-08)). |
| `capabilities` | W0.9 | `capabilities()` — the supported (product × convention × calculation) matrix. |
| `bill` | W1.2 | `price_bill`, `BillPrice`, `is_bill`, `BillPricingError`. |
| `note` | W1.3 | `price_note`, `NotePrice`, `rate_sensitivity`, `AccruedReconciliation`, `accrual_mismatch_tolerance`, `is_note`. |
| `equity` | W1.4 | `price_equity`, `EquityPosition`, `read_position`, `is_equity` — a **refusal** naming the missing spot/FX source ([I-18](../known-issues.md#i-18)). |
| `schema_version` | W1.6.2 | `RESULT_SCHEMA_VERSION`, `CAPABILITY_SCHEMA_VERSION` — a dependency-free leaf that breaks the `result` ↔ `schema` cycle. |
| `schema` | W1.6.2 | `result_schema()`, `capability_schema()` — JSON Schema **derived** from the frozen vocabulary, never hand-written. |
| `workload` | W1.6.4 / W0.8 | `workload_key`, `AttemptStore`, `Attempt`, `UNKNOWN_WORKLOAD` — the attempt state machine and the four lookup states. |
| `pipeline` | — | `price_bundle(bundle_or_path, market_inputs=None) -> RiskResult` — the composition of all of the above. Accepts a loaded `Bundle` or a path to load one from. |
