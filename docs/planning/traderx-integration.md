# TraderX Integration Readiness Plan

**Status:** Implemented — see [`engine/portfolio/request.py`](../../engine/portfolio/request.py) and
[The Portfolio Entry Point](../reference/portfolio-entrypoint.md) for the shipped design
(the validation/assembly layer described below), plus
[HTTP API](../reference/http-api.md) for the FastAPI wrapper built on top of it. This
document is kept as the original gap analysis/design rationale; each gap item below now
points at the actual landed function and test class rather than a "Plan:".

## Context

The [Roadmap](roadmap-and-history.md) calls for wrapping this engine as a stateless API
consumed by TraderX. `SimulationConfig` (`engine/simulation/market_model.py`) and each instrument's own
config dataclass (`SwapConfig`, `SwaptionConfig`, `BermudanSwaptionConfig`,
`AmericanSwaptionConfig`) already form a typed, IDE-friendly input surface — the
`SimulationConfig` docstring explicitly calls this out as "the natural shape for a future
Pydantic schema to mirror or subclass." That surface is sufficient for the demo scenarios in
`engine/simulation/demo_scenarios.py`, which are hand-built, internally consistent, and never
exercise the gaps below.

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
| Per-factor short-rate calibration | `RatesConfig.initial_rates` / `theta` / `mean_reversion` | `engine/simulation/market_model.py` |
| Per-factor today's zero curve | `RatesConfig.initial_zero_curves` (one `ZeroCurveConfig` per factor) | `engine/simulation/market_model.py` |
| Cross-asset covariance | `SimulationConfig.joint_covariance` (equities/FX first, then rates) | `engine/simulation/market_model.py` |
| Equity/FX legs + UIP drift mapping | `EquityConfig.initial_prices` / `dividend_yields` / `rate_mapping` | `engine/simulation/market_model.py` |
| Output discount-curve pillars | `RatesConfig.maturities` | `engine/simulation/market_model.py` |
| Swap trades | `SwapConfig` (notional, fixed_rate, payer, discount/forward curve index, tenor, index tenor, spread) | `engine/instruments/swap.py` |
| European swaption trades | `SwaptionConfig` (adds `hw_a`/`hw_sigma`/`initial_zero_curve`, duplicated per-trade from the matching rate factor) | `engine/instruments/european_swaption.py` |
| Bermudan swaption trades | `BermudanSwaptionConfig` (adds `exercise_times`, `n_per_std`/`std_devs` grid resolution) | `engine/instruments/bermudan_swaption.py` |
| American swaption trades | `AmericanSwaptionConfig` (adds `first_exercise`/`last_exercise`/`exercise_time_steps_per_year`) | `engine/instruments/american_swaption.py` |
| Portfolio base NPV / percentiles | `compute_risk_metrics(npv_cube, base_npv, percentiles)` | `engine/risk/var_es.py` |

## Gaps to close, in priority order

### 1. `joint_covariance` PSD validation/repair (highest priority)

**Problem:** `generate_paths` (`engine/simulation/market_model.py`) calls `jnp.linalg.cholesky` on the
correlation matrix derived from `joint_covariance` with no upstream validation. An invalid
(non-PSD) matrix — the expected case when correlations are assembled from independently
estimated pairwise correlations across many currencies/assets, a classic real-world
occurrence, not an edge case — silently produces an all-NaN Cholesky factor, which then
silently NaNs every simulated path with no error raised anywhere (confirmed in this session's
`TestCholeskyOnDegenerateCorrelation`).

**Implemented:**
- `validate_joint_covariance(matrix) -> None` in `engine/simulation/market_model.py`, called
  at the top of `generate_paths` before any JAX computation: confirms symmetry (within float
  tolerance) and positive semi-definiteness via eigenvalue check (`np.linalg.eigvalsh`).
  Raises `ValueError` naming the offending eigenvalue(s)/index.
- `nearest_psd(matrix, epsilon=1e-10) -> np.ndarray` repair utility (eigenvalue-clipping
  projection, clipping to a small positive `epsilon` rather than literally `0.0` — an
  exact-zero clip produces a numerically rank-deficient matrix that still fails Cholesky, see
  the function's own docstring) — opt-in only, never called automatically by `generate_paths`
  or `engine.portfolio`.
- Tests: `tests/test_market_model.py::TestCovarianceValidation` (implied rho > 1, a matrix
  with a deliberately negative eigenvalue, the repair path producing finite `generate_paths`
  output) and `TestCholeskyOnDegenerateCorrelation` (the original silent-NaN behavior this
  closes, updated to assert the new raise).

### 2. Cross-field consistency between `RatesConfig` and each instrument's duplicated fields

**Problem:** `SwaptionConfig`/`BermudanSwaptionConfig`/`AmericanSwaptionConfig` each carry
their own `hw_a`/`hw_sigma`/`initial_zero_curve`, which — per their own docstrings — "MUST
match that factor's own calibration in the simulation's `RatesConfig`." Nothing enforces
this. A caller (or an assembly layer with a bug) can point a swaption at
`rate_factor_index=1` while its `hw_a` was copied from factor 0's calibration, and the trade
prices against a silently self-inconsistent model with no error.

**Implemented:**
- `validate_portfolio_against_simulation(sim_config, trade_configs) -> None` in
  `engine/portfolio/request.py`, cross-checking every trade with a `rate_factor_index` against
  `sim_config.rates.mean_reversion[idx]` / the implied per-step vol from
  `sim_config.joint_covariance` / `sim_config.rates.initial_zero_curves[idx]`. Raises
  `ValueError` naming the trade (index/type/notional) and the specific mismatched field.
  A trade whose `hw_sigma` is a genuinely piecewise (calibrated) `Sigma` is deliberately
  **not** cross-checked against `joint_covariance`'s flat per-step vol — see
  [The Portfolio Entry Point](../reference/portfolio-entrypoint.md#validate_portfolio_against_simulationsim_config-trade_configs---none)
  for why that divergence is legitimate, not the transcription-bug class this item targets.
- Called automatically as step 2 of `price_portfolio`'s own orchestration — a caller doesn't
  need to remember to call this separately.
- Tests: `tests/test_portfolio.py::TestCrossFieldValidation`.

### 3. Automatic maturity-pillar assembly for arbitrary portfolios

**Problem:** `RatesConfig.maturities` and every swap/swaption's cashflow dates must align to
within the tolerance `swap.py::_maturity_indices` enforces (now symmetric, ±1e-6, after this
session's fix). The demo scenarios hand-pick `maturities` to match one hand-built portfolio's
cashflow dates. A real TraderX portfolio has many trades with irregular dates; hand-picking
pillars does not scale and is exactly the kind of manual step that will drift and break
silently in production.

**Implemented:**
- `derive_maturity_pillars(trade_configs, evaluation_date) -> List[float]` in
  `engine/portfolio/request.py`, building each `SwapConfig`'s real ORE schedule (via
  `engine.models.ore_builders.build_vanilla_swap`, the same shared construction every
  pricer uses) and returning the sorted union of every leg's accrual/payment
  year-fractions. Swaption-family trades contribute no pillars (they price directly off
  simulated `hw_paths`, not the maturity-pillar cube).
- `price_portfolio` calls this automatically (step 3 of its own orchestration) whenever the
  caller leaves `SimulationConfig.rates.maturities` unset — the standard way pillars get
  populated for an arbitrary portfolio now, rather than a manually maintained list.
- Tests: `tests/test_portfolio.py::TestPillarAssembly` and
  `tests/test_portfolio_entrypoint.py::TestPricePortfolioAutoDerivesMaturityPillars`.

### 4. Trade-level input validation (notional, rate ranges, tenor sanity)

**Problem:** No `*Config` dataclass validates its own field ranges (notional sign,
`fixed_rate` magnitude, tenor parseability) — currently anything reaching a pricer either
prices correctly (as of this session's `_solve_rstar` bracket-expansion fix, even extreme
inputs), silently mis-prices (pre-fix state), or raises an opaque low-level `ORE`/JAX error.
A production integration point receiving arbitrary TraderX trade payloads needs a clear
`ValueError` naming the bad field, not a stack trace from inside `ORE.MakeVanillaSwap`.

**Implemented:**
- `__post_init__` on `SwapConfig`, `SwaptionConfig`, `BermudanSwaptionConfig`, and
  `AmericanSwaptionConfig` (each in its own module), calling a shared
  `_validate_common_fields(notional, fixed_rate, evaluation_date)` helper from
  `engine/portfolio/validation.py` (a leaf module separate from `engine/portfolio/request.py` to
  avoid a circular import — see that module's own docstring) plus `_validate_tenor` for
  every `swap_tenor`. Notional/`fixed_rate` finiteness is checked; zero and negative values
  are explicitly documented and tested as supported (see e.g.
  `tests/test_swap.py::TestZeroNotional`), not silently allowed. Bermudan/American
  `hw_sigma` accepts `None` as a valid "uncalibrated" sentinel (see
  [The Portfolio Entry Point](../reference/portfolio-entrypoint.md#automatic-calibration)).
  `BermudanSwaptionConfig.exercise_times` must be non-empty and sorted ascending;
  `AmericanSwaptionConfig` requires `first_exercise <= last_exercise`.
- Deliberately scoped to reject malformed input, not impose business-rule limits (no "no
  rate above 20%" check) — matches this plan's original scope boundary.
- Tests: `TestSwapConfigValidation`/`TestSwaptionConfigValidation`/
  `TestBermudanSwaptionConfigValidation`/`TestAmericanSwaptionConfigValidation` in each
  instrument's own `tests/test_*.py` file.

### 5. Known limitations to surface explicitly in the integration layer (not fixed, by design)

These are documented, tested, deliberate scope boundaries already in the codebase — a
TraderX integration needs to either respect them or explicitly flag trades that fall outside
scope, not silently produce a slightly-wrong number:

- **Aged-swap limitation** (`swap.py`, `TestAgedSwapKnownLimitation`): conditional pricing at
  any simulated time past a swap's first accrual date doesn't represent an already-fixed
  floating coupon. Fine for t=0 valuation; a real gap for any time-stepped exposure/XVA
  profile — which is exactly what a risk system built on top of this would want.
  **Implemented:** documented (not fixed) in `engine/portfolio/request.py`'s own module docstring —
  any exposure profile (t>0 valuation) for a swap inherits this gap; closing it in `swap.py`
  itself remains a separate, not-yet-started future plan.
- **Mid-coupon Bermudan/American exercise** (`bermudan_swaption.py`/`american_swaption.py`,
  `TestMidCouponKnownLimitation`): exact only when exercise dates are reset-aligned;
  otherwise a conservative (understating) approximation. **Implemented:**
  `validate_portfolio_against_simulation` (item 2) emits a `UserWarning` (not a hard reject)
  naming the trade when any American/Bermudan exercise date isn't reset-aligned with its own
  underlying's accrual schedule, pointing at
  [american-bermudan-swaptions.md](../instruments/american-bermudan-swaptions.md)'s
  mid-coupon-approximation section. `price_portfolio` collects these into
  `PortfolioResult.warnings` rather than only printing to stderr. Tests:
  `tests/test_portfolio.py::TestCrossFieldValidation::test_bermudan_mid_coupon_exercise_time_warns`/
  `test_american_mid_coupon_exercise_window_warns`.

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

- Each item above got its own test class, run via the existing
  `venv/Scripts/python.exe -m pytest tests/ -q` workflow.
- This plan's own validation layer is now Phase 1 of a larger effort — see
  [The Portfolio Entry Point](../reference/portfolio-entrypoint.md) (Phase 2: the
  `price_portfolio` entry point built on top of it) and [HTTP API](../reference/http-api.md)
  (Phase 3: the FastAPI wrapper). `tests/test_portfolio_entrypoint.py` is the end-to-end
  check this section originally called for: a multi-instrument-type portfolio request,
  validated, priced through `generate_paths` → every pricer → `compute_risk_metrics` via
  `price_portfolio`, cross-checked bit-for-bit against the equivalent hand-orchestrated
  sequence.
- No changes to existing pricer math were made — the full test suite (659 tests before this
  work, growing with each new test file this plan and its follow-on phases added) continued
  to pass unchanged throughout, per the original scope boundary.
