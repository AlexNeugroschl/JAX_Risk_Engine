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
| `trades` | `List[SwapConfig \| SwaptionConfig \| BermudanSwaptionConfig \| AmericanSwaptionConfig \| BondConfig]` | *required* | Heterogeneous, any order or mix. `PortfolioResult`'s NPV cube and `greeks` dict are always reported back in this same order, regardless of how `price_portfolio` internally groups trades by type for pricing. |
| `percentiles` | `Sequence[float]` | `(0.95, 0.99)` | Confidence levels `compute_risk_metrics` computes VaR/ES at. |
| `calibration_targets` | `Optional[List[CalibrationTarget]]` | `None` | Used when any Bermudan/American trade's `hw_sigma` is left as `None` (uncalibrated) — see "Automatic calibration" below. |
| `compute_greeks` | `bool` | `False` | If `True`, also computes Delta/Gamma/Theta (and, implicitly, Vega where the trade's own calibration makes it well-defined) per trade — see "Greeks" below. |
| `precision` | `PrecisionConfig` | `PrecisionConfig()` (all-64) | Independent simulation/pricing/risk dtype control — see "`PrecisionConfig`" below. |
| `scenario_risk` | `bool` | `True` | Whether to build `npv_cube` and derive VaR/ES from it. **Must be `False` for any portfolio containing a `BondConfig`** — see "Bonds" below. |

### Bonds

`BondConfig` (W1.5, `engine.instruments.treasury`) covers Treasury bills and notes. A **bill**
is the degenerate case: `coupon_schedule=()` with `coupon_rate=0.0`. `face_amount` is
**signed**, so a short position is a negative face and yields a negative NPV directly.

Unlike `SwapConfig`, a bond carries **its own `initial_zero_curve`** rather than an index
into `market.rates.initial_zero_curves` — the same shape the swaption family uses, and the
reason a bond cannot reproduce [I-01](../known-issues.md#i-01)'s silent-skip failure.

**A bond has no scenario NPV**, so it never enters `npv_cube` and has no VaR/ES. Submitting
one with the default `scenario_risk=True` raises `ScenarioPricingNotSupported`, naming the
trade. With `scenario_risk=False` you get real `base_npv`, `base_npv_per_trade` and `greeks`,
with `risk` **empty** and `npv_cube` zero-width. The refusal is deliberate: a broadcast
constant column measures out to VaR `0.00` and ES `NaN` — see
[I-24](../known-issues.md#i-24).

Its Greeks are bumped revaluations (`delta` central-difference at 1bp, `gamma` a second
difference, `theta` a one-day reprice), all **scalars** rather than per-pillar vectors, and
**no `vega`** — a fixed-coupon bond off a deterministic curve has no volatility input, so it
is omitted rather than reported as `0.0`. `theta` is likewise **omitted for a bond maturing
tomorrow**: the one-day reprice would land exactly on maturity, a state `BondConfig` refuses
to construct, and the decay is genuinely undefined across that boundary rather than zero
([I-26](../known-issues.md#i-26)). `delta`/`gamma` are unaffected and still reported.

Note that this `delta` is **not bit-identical** to the EOD boundary's `rateSensitivity` for
the same bond, deliberately: this is a central difference, while the published contract at
[EOD Integration](eod-integration.md) is the one-sided `P(+1bp) − P(0)` TraderX agreed to
reconcile against. On a 6-month bill at 100k face they differ by ~1.4e-4 — the curvature
term, not an error in either.

## `PrecisionConfig`

Three independent dtype knobs, each `32` (float32) or `64` (float64, default) — see
[Architecture](../concepts/architecture.md#adjustable-precision) for the full mechanism
and why `pricing`/`risk` need no new parameters on any pricer or Greeks function.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `simulation` | `int` (`32`\|`64`) | `64` | Passed straight through to `generate_paths(config, precision=...)`. |
| `pricing` | `int` (`32`\|`64`) | `64` | Governs every array `price_portfolio` constructs itself (`step_times`, the swaption zero-shock `r0_path`, `_flat_curve_cube`'s output) before handing it to a pricer — `npv_cube`/`base_npv`'s dtype follows from this. |
| `risk` | `int` (`32`\|`64`) | `64` | Governs the `ZeroCurve` built for VaR/ES and Greeks (`_compute_all_greeks`) — every Greeks closure derives its own working dtype from that curve. |

`__post_init__` raises `ValueError` if any field is outside `{32, 64}` — bfloat16/float16
are not supported (see Architecture's "Out of scope for v1"). Constructing
`PortfolioRequest()` without a `precision` argument defaults to `PrecisionConfig()`
(all-64), byte-identical to this project's behavior before `PrecisionConfig` existed.

```python
from engine.portfolio import PortfolioRequest, PrecisionConfig

request = PortfolioRequest(
    market=market_config, trades=trades,
    precision=PrecisionConfig(simulation=64, pricing=32, risk=32),
)
```

**Concurrency note:** calling `price_portfolio` directly, yourself, from more than one
thread in your own process is still unsafe without external serialization — `_PRICING_LOCK`
inside this module protects against exactly that (`jax_enable_x64`, which `generate_paths`
toggles per `precision.simulation`, is process-global state, not thread-local). Via
`engine/api/routes.py`'s HTTP layer, this is no longer the primary concurrency mechanism:
concurrent jobs are now dispatched to `engine.portfolio.worker_pool`'s per-precision-tier
`ProcessPoolExecutor` pools, which achieve genuine cross-process concurrency instead of
queuing behind one lock — see [Architecture: Concurrency](../concepts/architecture.md) for
the full mechanism and why the lock is kept as narrower defense-in-depth rather than
removed.

## `PortfolioResult`

| Field | Type | Meaning |
|---|---|---|
| `base_npv` | `float` | The whole portfolio's t=0 NPV, against today's actual (zero-shock) curves — not read off the NPV cube. |
| `npv_cube` | `jax.Array` | `[Scenarios, TimeSteps, Trades]`, one column per trade in `request.trades`' own order. |
| `risk` | `Dict[str, jax.Array]` | `compute_risk_metrics`'s own output — `"VaR_95"`, `"ES_95"`, etc., each `[TimeSteps]`. |
| `greeks` | `Optional[Dict[int, Dict[str, jax.Array]]]` | `None` unless `request.compute_greeks=True`. Keyed by each trade's own index in `request.trades` (not by pricing-group order — see "Greeks" below). |
| `warnings` | `List[str]` | Known-limitation warnings surfaced during validation (see "Known-limitation flagging" below) — e.g. a Bermudan exercise date that isn't reset-aligned with its own underlying. |
| `base_npv_per_trade` | `List[float]` | Each trade's own t=0 NPV, in `request.trades` order. `base_npv` is by construction their sum, so the total and the breakdown cannot disagree. |
| `scenario_risk_available` | `bool` | `False` when the run was `scenario_risk=False`, meaning `risk` is **empty** and `npv_cube` zero-width. Carried on the *result* because a consumer holding one has no access to the request — without it, an empty `risk` is ambiguous between "not requested" and "computed as nothing". |
| `measure` | `Optional[str]` | Which measure `risk` is under: `"risk-neutral-pricing"` (`engine.risk.var_es.ENGINE_RISK_MEASURE`) whenever `risk` was computed, `None` when `scenario_risk_available` is `False`. An exposure under the pricing measure, **not** a forecast of tomorrow's loss ([I-11](../known-issues.md#i-11)). |

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
   `step_times` for every swaption type) behind one call. **Skipped entirely when
   `scenario_risk=False`**, which instead yields a deliberately *zero-width* cube rather
   than a zero-filled one — see "Bonds" above.
7. **Compute the base (t=0) NPV** by repricing every trade against zero-shock curves —
   generalizes what `demo.py` used to do by hand, per instrument type. This step runs
   either way, which is what lets a bond portfolio still return real numbers.
8. **Aggregate VaR/ES** (`compute_risk_metrics`). Also skipped when `scenario_risk=False`,
   leaving `risk` an **empty dict** — a missing key asserts nothing, where a `0.00` would
   assert a measured absence of risk.
9. **Optionally compute Greeks** per trade (below). Runs either way.

## Validation and assembly helpers

These implement
[`docs/planning/traderx-integration.md`](../planning/traderX_integration/traderx-integration.md)'s
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
any swap that will be aged past its first accrual date at a simulated step — see
"Known-limitation flagging" below. A Bermudan/American exercise date inside an accrual period
is not warned about: it is priced exactly as ORE prices it, not approximated.

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

A documented, deliberate scope boundary exists elsewhere in this codebase; this module
surfaces it rather than silently producing a slightly-wrong number:

- **Aged-swap discounting gap** (see [Interest Rate Swaps](../instruments/swaps.md)): this
  module's own docstring documents (but does not fix) that any *exposure profile* (t>0
  valuation, as opposed to a t=0 NPV/VaR run) of a swap inherits `engine.instruments.swap`'s
  known aged-swap limitation. `_warn_if_aged_swap_exposure` flags every `SwapConfig` that
  will be aged at one or more simulated steps with a `UserWarning` naming the trade,
  collected into `PortfolioResult.warnings` by `price_portfolio`.

## Greeks

When `request.compute_greeks=True`, `price_portfolio` computes Delta/Gamma/Theta for **every**
trade type, keyed by **the trade's own index in `request.trades`** — not by internal
pricing-group order, so `result.greeks[3]` always means "Greeks for `request.trades[3]`"
regardless of how many other trades of other types sit between them in the request.

| Trade type | Routed to | Shape |
|---|---|---|
| `SwapConfig` | `swap_delta_gamma` / `swap_theta` | per-pillar vectors, named `discount_delta`/`forward_delta` per curve |
| `SwaptionConfig` | `swaption_delta_gamma` / `swaption_theta` | per-pillar vectors |
| `BermudanSwaptionConfig`/`AmericanSwaptionConfig` | `bermudan_delta_gamma` / `bermudan_theta` (+ `bermudan_vega` where calibrated) | per-pillar vectors |
| `BondConfig` | `_bond_greeks` (bumped revaluation) | **scalars**; no `vega` |

> **Historical note — this section previously said `SwapConfig` trades are "skipped".** That
> was [I-01](../known-issues.md#i-01): swaps silently returned no Greeks because
> `_compute_all_greeks` had no access to the `SimulationConfig` their curve *indexes* resolve
> against. It was fixed by passing `market_config` through, and the documentation above is
> corrected to match. Swap Greeks have been computed by `price_portfolio` since that fix; the
> old text survived it.

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
- `tests/test_portfolio_scale_and_edge_cases.py` — `price_portfolio` at varying portfolio
  sizes (1, 12, and 50 trades), across multiple rate factors (untested at the entry-point
  level elsewhere), and composition edge cases: empty portfolios, single-instrument-type
  portfolios at scale, duplicate trades, zero/negative/very-large notionals, and large
  offsetting positions netting to near-zero risk end to end.
- `tests/test_portfolio.py::TestCrossFieldValidation`'s two-rate-factor cases — confirm
  `validate_portfolio_against_simulation` indexes into the *correct* factor's own curve/
  mean-reversion/vol at factor counts above one, not factor 0 by coincidence.
- `tests/test_treasury_instrument.py` (40) — `BondConfig` in isolation: bill and note
  pricing, ACT/ACT (ICMA) accrual, refusals, curve interpolation, and the bit-exact
  cross-check against `engine.integration.bill`/`note`.
- `tests/test_portfolio_bond_wire_through.py` (36) — the wire-through through
  `price_portfolio` itself. `TestBondGreeksReachThePortfolioPath` is the
  [I-01](../known-issues.md#i-01) regression class (**19 of 19 verified to fail** with the
  Greeks branch deleted); `TestScenarioRiskIsRefusedForBonds` pins
  [I-24](../known-issues.md#i-24).
- `tests/test_api_bond_schemas.py` (21) — the HTTP surface, including
  [I-25](../known-issues.md#i-25)'s scalar-Greek serialization.
