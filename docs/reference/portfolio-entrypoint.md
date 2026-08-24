# The Portfolio Entry Point

**Module:** [`engine/portfolio/request.py`](../../engine/portfolio/request.py)
**Public entry point:** `price_portfolio(request: PortfolioRequest) -> PortfolioResult`

## Plain-language summary

Every module described elsewhere in these docs — market simulation, the four instrument
pricers, VaR/ES, Greeks, calibration — is real, tested, and correct, but until this module
existed there was no single place that tied them together for a *caller*. Pricing even one
mixed portfolio required knowing exactly which pricer wants `yield_curves` versus
`hw_paths`, hand-building maturity pillars from a swap's own real cashflow dates, summing
each type's own base NPV by hand, and never being warned if a swaption's `hw_a` silently
drifted out of sync with the simulation's own calibration.

`engine.portfolio` answers the question "what should a caller of the whole system hand
over, and what do they get back?" with two dataclasses — `PortfolioRequest` in,
`PortfolioResult` out — and one function, `price_portfolio`, that does everything in
between: validate, simulate, calibrate (if needed), price every trade, and aggregate risk.
`demo.py` is now a thin example calling this one function instead of hand-orchestrating
~200 lines of pipeline plumbing.

This module is also where `docs/planning/traderx-integration.md`'s validation/assembly
layer lives — the layer that stands between "hand-built, internally consistent demo
configs" and "an arbitrary portfolio a real caller assembled, with irregular dates and
possibly-inconsistent market data." See that plan's own gap items for the full rationale;
this doc covers what actually landed.

## `PortfolioRequest`

What a caller of the whole system hands over: a portfolio of trades + market data + risk
parameters.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `market` | `SimulationConfig` | *required* | Curves, vols (via `joint_covariance`), correlations — the same config `generate_paths` consumes (see [API Reference](api-reference.md#enginesimulationmarket_model)). If `market.rates.maturities` is left unset, `price_portfolio` derives it automatically (see `derive_maturity_pillars` below). |
| `trades` | `List[SwapConfig \| SwaptionConfig \| BermudanSwaptionConfig \| AmericanSwaptionConfig]` | *required* | Heterogeneous, any order or mix. `PortfolioResult`'s NPV cube and `greeks` dict are always reported back in this same order, regardless of how `price_portfolio` internally groups trades by type for pricing. |
| `percentiles` | `Sequence[float]` | `(0.95, 0.99)` | Confidence levels `compute_risk_metrics` computes VaR/ES at. |
| `calibration_targets` | `Optional[List[CalibrationTarget]]` | `None` | Used when any Bermudan/American trade's `hw_sigma` is left as `None` (uncalibrated) — see "Automatic calibration" below. |
| `compute_greeks` | `bool` | `False` | If `True`, also computes Delta/Gamma/Theta (and, implicitly, Vega where the trade's own calibration makes it well-defined) per trade — see "Greeks" below. |

A future `BondConfig` instrument type (see
[TraderX Bond Integration Roadmap](../planning/traderx-bond-integration-roadmap.md)) would
join `trades`' `Union` here once it exists — out of scope for this module today.

## `PortfolioResult`

| Field | Type | Meaning |
|---|---|---|
| `base_npv` | `float` | The whole portfolio's t=0 NPV, against today's actual (zero-shock) curves — not read off the NPV cube. |
| `npv_cube` | `jax.Array` | `[Scenarios, TimeSteps, Trades]`, one column per trade in `request.trades`' own order. |
| `risk` | `Dict[str, jax.Array]` | `compute_risk_metrics`'s own output — `"VaR_95"`, `"ES_95"`, etc., each `[TimeSteps]`. |
| `greeks` | `Optional[Dict[int, Dict[str, jax.Array]]]` | `None` unless `request.compute_greeks=True`. Keyed by each trade's own index in `request.trades` (not by pricing-group order — see "Greeks" below). |
| `warnings` | `List[str]` | Known-limitation warnings surfaced during validation (see "Known-limitation flagging" below) — e.g. a Bermudan exercise date that isn't reset-aligned with its own underlying. |

## `price_portfolio(request: PortfolioRequest) -> PortfolioResult`

Orchestrates, in order:

1. **Validate `market.joint_covariance`** via `validate_joint_covariance` (see
   [Market Simulation](../concepts/market-simulation.md)) — an explicit, request-scoped
   check before any trade-level work happens (the same check also runs inside
   `generate_paths`, but failing here first gives a clearer, earlier error).
2. **Cross-check every trade against `market`** via `validate_portfolio_against_simulation`
   (below).
3. **Auto-derive maturity pillars** via `derive_maturity_pillars` if `market.rates.maturities`
   is unset (below).
4. **Simulate the market** (`generate_paths`).
5. **Calibrate any uncalibrated Bermudan/American trade's `hw_sigma`** (below).
6. **Route every trade to its pricer by type** (`price_swaps`/`price_swaptions`/
   `price_bermudan_swaptions`/`price_american_swaptions`), concatenating into one NPV cube
   reassembled in the caller's original `trades` order — this is exactly what unifies the
   four pricers' different input shapes (`yield_curves`+pillars for swaps, `hw_paths`+
   `step_times` for every swaption type) behind one call.
7. **Compute the base (t=0) NPV** by repricing every trade against zero-shock curves —
   generalizes what `demo.py` used to do by hand, per instrument type.
8. **Aggregate VaR/ES** (`compute_risk_metrics`).
9. **Optionally compute Greeks** per trade (below).

## Validation and assembly helpers

These implement
[`docs/planning/traderx-integration.md`](../planning/traderx-integration.md)'s
validation/assembly layer — see that plan for the full gap analysis; the sections below
cover what actually shipped.

### `validate_portfolio_against_simulation(sim_config, trade_configs) -> None`

For every trade carrying a `rate_factor_index` (every type except `SwapConfig`), cross-checks
that trade's own duplicated `hw_a` / `hw_sigma` / `initial_zero_curve` against
`sim_config`'s corresponding entry for that rate factor — the fields every
`SwaptionConfig`-family class's own docstring says "MUST match that factor's own
calibration in the simulation's `RatesConfig`," now actually enforced. Raises `ValueError`
naming the trade (by index/type/notional) and the specific mismatched field.

A trade whose `hw_sigma` is a genuinely piecewise (calibrated) `Sigma` is **not**
cross-checked against `joint_covariance`'s single flat per-step vol — the simulation only
ever propagates one constant volatility per rate factor for path generation, while a
calibrated `Sigma` is the model's own richer view of volatility used for *pricing*; a real
desk workflow (calibrate once, reuse the fitted term structure across many trades,
simulate paths off one representative vol level) has these legitimately diverge. Only a
flat `float` `hw_sigma` is checked against the implied per-step vol.

This function also emits (via Python's `warnings` module — not a hard error) a warning for
any Bermudan/American trade whose `exercise_times` aren't reset-aligned with its own
underlying's accrual dates — see "Known-limitation flagging" below.

### `derive_maturity_pillars(trade_configs, evaluation_date) -> List[float]`

Builds every `SwapConfig`'s real ORE schedule (via
`engine.models.ore_builders.build_vanilla_swap`, the same shared construction every pricer
already uses) and returns the sorted union of every leg's accrual/payment year-fractions —
the exact maturity-pillar set `engine.instruments.swap`'s `_maturity_indices` requires.
Automates what `demo.py` used to compute by hand for a single swap.

Only `SwapConfig` trades contribute pillars — every swaption-family pricer prices directly
off simulated `hw_paths`, not the `yield_curves` cube, so their cashflow dates impose no
pillar-alignment requirement. A portfolio with no `SwapConfig` trades derives just the
anchor pillar `[0.0]`.

### Automatic calibration

If any Bermudan/American trade's `hw_sigma is None`, `price_portfolio` calibrates it via
`engine.calibration.lgm.calibrate_lgm_sigma`, using `request.calibration_targets` — **once
per distinct `rate_factor_index`** needing it, not once per trade (every trade sharing a
rate factor shares the calibrated `Sigma`). Raises `ValueError` if `calibration_targets`
wasn't supplied but a trade needs it.

`hw_sigma=None` is a valid sentinel value on `BermudanSwaptionConfig`/
`AmericanSwaptionConfig` specifically to support this — it means "uncalibrated," not
"malformed"; `__post_init__` on both configs treats it as valid and skips the finite-value
check for it.

### Known-limitation flagging

Two documented, deliberate scope boundaries already exist elsewhere in this codebase; this
module surfaces them rather than silently producing a slightly-wrong number:

- **Mid-coupon Bermudan/American exercise** (see
  [American & Bermudan Swaptions](../instruments/american-bermudan-swaptions.md)): if any
  exercise date isn't reset-aligned with the underlying's own accrual schedule,
  `validate_portfolio_against_simulation` emits a `UserWarning` naming the trade and
  pointing at that doc's mid-coupon-approximation section — collected into
  `PortfolioResult.warnings` by `price_portfolio`, not just printed to stderr.
- **Aged-swap discounting gap** (see [Interest Rate Swaps](../instruments/swaps.md)): this
  module's own docstring documents (but does not fix) that any *exposure profile* (t>0
  valuation, as opposed to a t=0 NPV/VaR run) of a swap inherits `engine.instruments.swap`'s
  known aged-swap limitation. This is not separately flagged per-request — it's a blanket
  documented scope boundary, since every `SwapConfig` trade is affected identically.

## Greeks

When `request.compute_greeks=True`, `price_portfolio` computes Delta/Gamma/Theta for every
`SwaptionConfig`/`BermudanSwaptionConfig`/`AmericanSwaptionConfig` trade (routing to
`engine.risk.greeks.swaption_delta_gamma`/`swaption_theta` or
`bermudan_delta_gamma`/`bermudan_theta` respectively), keyed by **the trade's own index in
`request.trades`** — not by internal pricing-group order, so `result.greeks[3]` always
means "Greeks for `request.trades[3]`" regardless of how many other trades of other types
sit between them in the request.

`SwapConfig` trades are **skipped**: `engine.risk.greeks.swap_delta_gamma` needs an
explicit `ZeroCurve` the caller must supply separately (a `SwapConfig` indexes into the
simulation's own curves rather than carrying its own `ZeroCurveConfig` the way every
swaption-family config does), so `price_portfolio` does not guess one on the caller's
behalf. Compute swap Greeks directly via `engine.risk.greeks.swap_delta_gamma` if needed.

## Tested by

- `tests/test_portfolio.py::TestCrossFieldValidation` — `validate_portfolio_against_simulation`,
  including the reset-alignment warning.
- `tests/test_portfolio.py::TestPillarAssembly` — `derive_maturity_pillars`.
- `tests/test_portfolio_entrypoint.py::TestPricePortfolioMatchesHandOrchestration` — proves
  `price_portfolio`'s output is bit-for-bit identical to the equivalent hand-orchestrated
  `demo.py`-style sequence, trade-by-trade, for a portfolio mixing all four instrument
  types.
- `TestPricePortfolioReorderingIndependence` — confirms the NPV cube always reassembles in
  the caller's original trade order, regardless of internal type-grouping.
- `TestPricePortfolioAutoDerivesMaturityPillars` — the automatic-pillar-assembly path.
- `TestPricePortfolioCalibration` — the `hw_sigma=None` auto-calibration path, and the
  clear error when `calibration_targets` is missing.
- `TestPricePortfolioGreeks` — per-trade Greeks keyed by original request order.
