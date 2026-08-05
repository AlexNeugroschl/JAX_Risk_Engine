# TraderX Integration Readiness Plan

**Status:** Not started — planning document only, nothing in this file has been implemented.

## Context

Phase 8 of the roadmap ([`README.md`](../../README.md)) calls for wrapping this engine as a stateless API
consumed by TraderX. `SimulationConfig` (`engine/simulation.py`) and each instrument's own
config dataclass (`SwapConfig`, `SwaptionConfig`, `BermudanSwaptionConfig`,
`AmericanSwaptionConfig`) already form a typed, IDE-friendly input surface — the
`SimulationConfig` docstring explicitly calls this out as "the natural shape for a future
Pydantic schema to mirror or subclass." That surface is sufficient for the demo scenarios in
`engine/scenarios.py`, which are hand-built, internally consistent, and never exercise the
gaps below.

A real TraderX portfolio will not arrive hand-built or internally consistent. It will have
irregular cashflow dates across many trades, correlation/vol data assembled from disparate
sources (not guaranteed positive semi-definite), and configs that can disagree with each
other (e.g. a swaption's `hw_a` not matching its stated `rate_factor_index`'s calibration in
`RatesConfig`). This plan identifies what needs to be built — as a validation/assembly layer
in front of the existing engine, not changes to the engine's pricing math — before arbitrary
TraderX input can flow through safely. The intended outcome is a documented, testable
boundary: a `PortfolioRequest`-style entry point that either produces a valid `SimulationConfig`
+ trade configs, or raises a clear, actionable error before anything reaches `generate_paths`.

## Inventory of required inputs (already present, for reference)

| Input | Dataclass / field | Source module |
|---|---|---|
| Per-factor short-rate calibration | `RatesConfig.initial_rates` / `theta` / `mean_reversion` | `engine/simulation.py` |
| Per-factor today's zero curve | `RatesConfig.initial_zero_curves` (one `ZeroCurveConfig` per factor) | `engine/simulation.py` |
| Cross-asset covariance | `SimulationConfig.joint_covariance` (equities/FX first, then rates) | `engine/simulation.py` |
| Equity/FX legs + UIP drift mapping | `EquityConfig.initial_prices` / `dividend_yields` / `rate_mapping` | `engine/simulation.py` |
| Output discount-curve pillars | `RatesConfig.maturities` | `engine/simulation.py` |
| Swap trades | `SwapConfig` (notional, fixed_rate, payer, discount/forward curve index, tenor, index tenor, spread) | `engine/instruments/swap.py` |
| European swaption trades | `SwaptionConfig` (adds `hw_a`/`hw_sigma`/`initial_zero_curve`, duplicated per-trade from the matching rate factor) | `engine/instruments/european_swaption.py` |
| Bermudan swaption trades | `BermudanSwaptionConfig` (adds `exercise_times`, `n_per_std`/`std_devs` grid resolution) | `engine/instruments/bermudan_swaption.py` |
| American swaption trades | `AmericanSwaptionConfig` (adds `first_exercise`/`last_exercise`/`exercise_time_steps_per_year`) | `engine/instruments/american_swaption.py` |
| Portfolio base NPV / percentiles | `compute_risk_metrics(npv_cube, base_npv, percentiles)` | `engine/risk/statistics.py` |

## Gaps to close, in priority order

### 1. `joint_covariance` PSD validation/repair (highest priority)

**Problem:** `generate_paths` (`engine/simulation.py`) calls `jnp.linalg.cholesky` on the
correlation matrix derived from `joint_covariance` with no upstream validation. An invalid
(non-PSD) matrix — the expected case when correlations are assembled from independently
estimated pairwise correlations across many currencies/assets, a classic real-world
occurrence, not an edge case — silently produces an all-NaN Cholesky factor, which then
silently NaNs every simulated path with no error raised anywhere (confirmed in this session's
`TestCholeskyOnDegenerateCorrelation`).

**Plan:**
- Add a `validate_joint_covariance(matrix) -> None` check in `engine/simulation.py`, run at
  the top of `generate_paths` before any JAX computation: confirm symmetry (within float
  tolerance) and confirm positive semi-definiteness via eigenvalue check (`np.linalg.eigvalsh`,
  cheap on CPU since this runs once per call, not per scenario). Raise `ValueError` with the
  offending eigenvalue(s) reported, not a bare failure.
- Add an optional `nearest_psd(matrix) -> matrix` repair utility (standard eigenvalue-clipping
  projection: clip negative eigenvalues to ~0, reconstruct, re-symmetrize) for the TraderX
  assembly layer to call explicitly when it wants "best effort" rather than "reject" — this
  must be an opt-in call, not silently applied inside `generate_paths` itself, so a genuinely
  bad correlation input is never priced without the caller knowing it was altered.
- Tests: a `TestCovarianceValidation` class feeding known-invalid matrices (implied rho > 1,
  a matrix with one negative eigenvalue by construction) and confirming both the raise path and
  the repair path (repaired matrix passes validation, and produces finite `generate_paths`
  output).

### 2. Cross-field consistency between `RatesConfig` and each instrument's duplicated fields

**Problem:** `SwaptionConfig`/`BermudanSwaptionConfig`/`AmericanSwaptionConfig` each carry
their own `hw_a`/`hw_sigma`/`initial_zero_curve`, which — per their own docstrings — "MUST
match that factor's own calibration in the simulation's `RatesConfig`." Nothing enforces
this. A caller (or an assembly layer with a bug) can point a swaption at
`rate_factor_index=1` while its `hw_a` was copied from factor 0's calibration, and the trade
prices against a silently self-inconsistent model with no error.

**Plan:**
- Add a `validate_portfolio_against_simulation(sim_config, trade_configs) -> None` helper
  (new module, `engine/portfolio.py`, or a function in `engine/scenarios.py` if that's judged
  the more natural home) that, for every trade with a `rate_factor_index`, cross-checks
  `hw_a`/`hw_sigma`/`initial_zero_curve` against `sim_config.rates.mean_reversion[idx]` /
  the implied per-step vol from `sim_config.joint_covariance` / `sim_config.rates.initial_zero_curves[idx]`,
  raising `ValueError` naming the trade and the specific mismatched field on any divergence
  beyond a small float tolerance.
- This closes the door on the single most likely TraderX-integration bug class: an assembly
  layer that builds `RatesConfig` and per-trade swaption configs from the same upstream
  market-data source but has a transcription bug between the two.
- Tests: construct a `SimulationConfig` + a swaption config with a deliberately mismatched
  `hw_a`, confirm the validator raises; confirm a correctly-matched config passes.

### 3. Automatic maturity-pillar assembly for arbitrary portfolios

**Problem:** `RatesConfig.maturities` and every swap/swaption's cashflow dates must align to
within the tolerance `swap.py::_maturity_indices` enforces (now symmetric, ±1e-6, after this
session's fix). The demo scenarios hand-pick `maturities` to match one hand-built portfolio's
cashflow dates. A real TraderX portfolio has many trades with irregular dates; hand-picking
pillars does not scale and is exactly the kind of manual step that will drift and break
silently in production.

**Plan:**
- Add a `derive_maturity_pillars(trade_configs, evaluation_date) -> List[float]` helper that,
  given a list of trade configs, builds each trade's ORE schedule (reusing the existing
  `_build_ore_swap`-style construction already in `swap.py`/`bermudan_swaption.py`, not
  reimplementing schedule logic) and returns the sorted union of every leg's accrual/payment
  year-fractions — i.e., automates what `SWAP_DEMO_MATURITIES` currently does by hand in
  `engine/scenarios.py`.
- This becomes the standard way `RatesConfig.maturities` gets populated for a TraderX
  portfolio, rather than a manually maintained list.
- Tests: feed a multi-trade, multi-tenor portfolio through the helper, confirm every trade's
  own cashflow dates are a subset of the returned pillar list (i.e. `_maturity_indices` would
  accept all of them), and confirm a portfolio requiring >1 trade's dates produces a strictly
  larger pillar set than either trade alone.

### 4. Trade-level input validation (notional, rate ranges, tenor sanity)

**Problem:** No `*Config` dataclass validates its own field ranges (notional sign,
`fixed_rate` magnitude, tenor parseability) — currently anything reaching a pricer either
prices correctly (as of this session's `_solve_rstar` bracket-expansion fix, even extreme
inputs), silently mis-prices (pre-fix state), or raises an opaque low-level `ORE`/JAX error.
A production integration point receiving arbitrary TraderX trade payloads needs a clear
`ValueError` naming the bad field, not a stack trace from inside `ORE.MakeVanillaSwap`.

**Plan:**
- Add `__post_init__` validation to each `*Config` dataclass (or a shared
  `_validate_common_fields` helper called from each): notional is finite and (for now)
  documented as to whether zero/negative is intentionally supported (it already is, per this
  session's tests — just needs an explicit docstring statement, not silent support), fixed
  rates are finite, tenors parse as valid `ORE.Period` strings before use, exercise
  schedules are internally ordered (`first_exercise <= last_exercise`, `exercise_times`
  sorted and within the underlying's maturity).
- This is deliberately scoped to *reject clearly malformed input early*, not to impose
  business-rule limits (e.g. "no rate above 20%") — those belong in a TraderX-side policy
  layer, not the pricing engine.
- Tests: one test per validated field per config class, confirming a bad value raises
  `ValueError` at construction time rather than later inside `generate_paths`/a pricer.

### 5. Known limitations to surface explicitly in the integration layer (not fixed, by design)

These are documented, tested, deliberate scope boundaries already in the codebase — a
TraderX integration needs to either respect them or explicitly flag trades that fall outside
scope, not silently produce a slightly-wrong number:

- **Aged-swap limitation** (`swap.py`, `TestAgedSwapKnownLimitation`): conditional pricing at
  any simulated time past a swap's first accrual date doesn't represent an already-fixed
  floating coupon. Fine for t=0 valuation; a real gap for any time-stepped exposure/XVA
  profile — which is exactly what a risk system built on top of this would want. **Action for
  this plan:** flag in the integration-layer docstring/API response that exposure profiles
  (as opposed to t=0 NPV/VaR) inherit this approximation, and consider whether closing this
  gap in `swap.py` itself becomes a separate, later plan.
- **Mid-coupon Bermudan/American exercise** (`bermudan_swaption.py`/`american_swaption.py`,
  `TestMidCouponKnownLimitation`): exact only when exercise dates are reset-aligned;
  otherwise a conservative (understating) approximation. TraderX-submitted trades won't
  naturally respect this. **Action:** the trade-level validation in item 4 above should emit
  a warning (not a hard reject — this is a documented approximation, not an error) when an
  American/Bermudan trade's exercise schedule isn't reset-aligned with its underlying.

## Suggested build order

1. Item 1 (PSD validation) — standalone, no dependency on the others, highest blast-radius if
   skipped (silent NaN across an entire simulation).
2. Item 4 (trade-level validation) — standalone, small, immediately useful.
3. Item 2 (cross-field consistency) — depends on having both `SimulationConfig` and trade
   configs assembled, natural to build alongside item 3.
4. Item 3 (pillar assembly) — depends on trade configs existing; naturally follows item 2.
5. Item 5 — documentation/flagging work threaded through items 2-4 rather than a standalone
   step.

## Verification

- Each item above gets its own test class, run via the existing
  `venv/Scripts/python.exe -m pytest tests/ -q` workflow.
- End-to-end check once items 1-4 exist: build a synthetic "TraderX-shaped" portfolio request
  (irregular trade dates, a correlation matrix assembled from independent pairwise estimates
  that is *not* exactly PSD, a couple of trades with deliberately mismatched
  `rate_factor_index` calibration) and confirm the new validation layer rejects it with clear,
  specific errors — then fix the synthetic input and confirm the same portfolio flows through
  `generate_paths` → pricers → `compute_risk_metrics` end to end.
- No changes to existing pricer math are in scope for this plan — the existing 502-test suite
  must continue to pass unchanged throughout.
