# Project Overview: JAX GPU-Accelerated Risk Engine

## 1. The Core Objective & The Precision Hypothesis
The primary goal of this project is to build a hardware-accelerated market simulation and pricing engine in pure JAX that mathematically mirrors the Open Source Risk Engine (ORE), but executes in milliseconds.

The ultimate research objective (to be conducted once the engine is fully built) is to **test the compute-precision tradeoff in Monte Carlo risk systems**. Specifically, we will test whether running a significantly higher volume of **lower-precision simulations (e.g., FP32 or BF16)** can reach statistical and risk parity with a smaller number of **high-precision (FP64) calculations**, within the same GPU compute budget. 

**Target Integration:** The engine is being designed for eventual integration into **TraderX** as a stateless API. However, the system will maintain full support for ingesting standard ORE XML configuration files to allow for continuous automated testing and exact mathematical validation against ORE's C++ outputs.

## 2. System Architecture
The engine is structured as a 3-stage pipeline:

*   **Stage 1: ETL & Parameter Ingestion (CPU):** 
    *   *Dual Ingestion:* Supports parsing ORE XML/CSV files (`xml.etree.ElementTree`) for validation testing, alongside a FastAPI/gRPC endpoint for real-time TraderX payloads.
    *   *Transform:* Pre-computes deterministic affine matrices ($A$ and $B$) on the CPU and passes the strictly typed payload to JAX.
    *   **Update (Phase 2):** `engine/instruments/` extends this beyond XML-only validation — it imports the `ORE` Python package (`open-source-risk-engine`) as a **hard runtime dependency**, using `ORE.MakeVanillaSwap`/`ORE.Schedule`/day-count classes directly to build real trade schedules and coupon accrual (not reimplemented in numpy), matching the "use ORE directly for CPU-side, non-performance-heavy work" approach. GPU/JAX code stays ORE-free. This means Phase 8's TraderX microservice will need `ORE` installed wherever `engine/instruments/` runs, not just in test/validation environments.
*   **Stage 2: The Cross-Asset Time Machine (GPU):** A `jax.lax.scan` loop executing Quasi-Monte Carlo (Sobol + Brownian Bridge) paths for Equities, FX (Uncovered Interest Parity), and Interest Rates (Hull-White 1-Factor).
*   **Stage 3: Vectorized Pricing & Risk (GPU):** Consumes the generated 4D Yield Curve cubes and asset paths to output Net Present Value (NPV) cubes and aggregate risk metrics.

**Architecture review (post-Phase 3):** `generate_paths` now takes a typed `SimulationConfig` dataclass (`engine/market_simulations.py`) instead of a loosely-typed dict — every config field is validated by Python's own dataclass machinery at construction time rather than by a `KeyError` deep inside the pipeline, and this is the natural shape for the Phase 8 Pydantic/FastAPI schema to mirror or subclass directly. The demo scenario configs and ORE flat-curve-building helper, previously hand-copied (and drifting) across every module's `__main__` block and the test suite, are now centralized in `engine/scenarios.py` and `tests/conftest.py`. `generate_sobol_normals` now honors its `dtype` argument unconditionally (an explicit cast fixes a case where `jax.scipy.stats.norm.ppf` silently ignored it), closing a gap in Phase 9's precision-adjustability requirement for any direct caller.

**Correctness fix (post-architecture-review):** `RatesConfig.initial_zero_curve` (singular, one curve shared across every rate factor) is now `initial_zero_curves` (a list, one curve per factor, validated to match the factor count). The shared-curve behavior was a bug, confirmed by directly introspecting and live-testing the installed ORE package's Cross-Asset Model — `ORE.IrLgm1fConstantParametrization` and `ORE.IrLgm1fPiecewiseConstantParametrization` both require a `(Currency, YieldTermStructureHandle)` pair at construction, and `ORE.CrossAssetModel` only ever aggregates a list of already-curve-bound per-currency parametrizations; there is no ORE constructor path, XML config path, or design intent anywhere that shares one curve across factors. A live 2-currency `CrossAssetModel` (USD 3% / EUR 2%, distinct flat curves) was built and confirmed to retain each currency's own discount factors throughout. `engine/scenarios.py`'s demo configs now give each rate factor its own curve consistent with that factor's own `initial_rates`/`theta`.

**Correctness fixes (post-Phase-5, thorough-testing pass):** Building an end-to-end test that prices a mixed swap + swaption portfolio through this engine and cross-checks it against real ORE objects at scale surfaced two previously-undetected bugs in `engine/market_simulations.py`'s core Hull-White/GBM step (`_simulate_cross_asset_paths_jit`), invisible to every prior test because those all validated pricing formulas at a *given* simulated rate, never the *distribution* of simulated rates itself:
1.  **Mean-reversion drift bug:** the HW1F step computed `r_next = r_t*decay + theta_hw + shock_hw` instead of the correct closed-form Ornstein-Uhlenbeck transition `r_next = r_t*decay + theta_hw*(1-decay) + shock_hw`. The missing `(1-decay)` factor made `theta` act as a flat per-step drift increment instead of a mean-reversion target, so every simulated rate factor drifted upward (or downward) *without bound* every step — e.g. a 3%-mean scenario's simulated mean rate reached ~14.6% by t=2y under the existing swap demo's own cadence. Invisible in every pre-existing test because they all set `theta == initial_rates`, a fixed point only under the *correct* formula. Fixed; regression tests in `tests/test_market_simulations.py::TestHullWhiteMeanReversionTransition` check simulated means against the closed-form OU transition directly, both where `theta == r0` and where they genuinely differ.
2.  **Double-applied volatility bug:** the per-step correlation matrix `L_t` was built as the Cholesky factor of the *raw* covariance matrix (whose diagonal already encodes each factor's own volatility), and the step formulas then multiplied the already-scaled shock by that same factor's volatility a *second* time — squaring the effective volatility actually simulated (e.g. a configured 20% equity vol produced an actual ~4% simulated log-return std). Fixed by building `L_t` from the *correlation* matrix (unit diagonal) instead, so `joint_sigma_t`'s explicit multiplication in `step_fn` is the only place volatility is applied. Regression tests in `TestVolatilityIsNotDoubleApplied` check simulated variance and cross-factor correlation directly against their configured values, for both the equity/rate and multi-rate-factor cases.

Both bugs affected every multi-step simulation this project had ever run (every existing demo and every risk-statistics test), yet every *point-in-time* ORE cross-check still passed, because those compare a pricing formula against ORE at a *given* rate value, not the statistical behavior of the rate-generation process itself — the gap this testing pass was specifically aimed at closing. A related, smaller-scope limitation was also found and documented (not fixed, by deliberate scoping decision) while building the same end-to-end test: `engine/instruments/interest_rate_swap.py::price_swaps` has no representation of an already-fixed/elapsed floating coupon, producing a small but real NPV error when a swap is priced at any simulated time past its own first accrual date (i.e. every step after t=0 for a spot-starting swap) — see that module's "Known limitation" docstring and `tests/test_interest_rate_swap.py::TestAgedSwapKnownLimitation` for the regression coverage pinning this down as a documented, tested gap rather than a silent one.

**End-to-end validation:** `tests/test_end_to_end.py` prices a mixed portfolio (one interest rate swap, two European swaptions — one spot-starting-equivalent, one forward-starting) through this engine's complete pipeline and, independently, through real `ORE.DiscountingSwapEngine`/`ORE.JamshidianSwaptionEngine`/`ORE.RiskStatistics` objects conditioned on the *exact same* simulated short-rate values (so the comparison isolates pricing/risk correctness from random-number-generator differences). Per-scenario portfolio NPV matches ORE to better than `1e-3` relative error across the full simulated distribution (not just the mean), and VaR/ES computed from each side's own NPV cube match to `1e-3` relative tolerance. Timing is reported honestly at multiple scales rather than cherry-picked: at small scenario counts (~500) ORE's plain Python loop is faster in absolute terms (this engine pays a fixed per-call JIT-compilation/dispatch cost), while at large scenario counts (~32,000+) this engine's vectorized tensor pricing wins decisively (~1.6x and growing) — see `TestEndToEndScaling`.

## 3. Development Roadmap
We will build this system out in the following sequential phases:

### Phase 1: Port the Market Simulations to JAX (Done)
*   **Goal:** Finalize the exact replica of ORE's Cross-Asset Model (CAM).
*   **Tasks:** 
    *   Extract ORE’s exact recursive Brownian Bridge matrix. ✅ `engine/market_simulations.py::_build_bridge_matrix` — QuantLib/ORE's recursive bisection construction, verified to exactly reproduce Brownian motion's covariance structure.
    *   Extract calibrated today's yield curves for the Hull-White $A(t,T)$ parameter. ✅ `compute_hw_A_matrix`, calibrated **independently per rate factor** against that factor's own curve (`rates.initial_zero_curves`, one per factor), verified to exactly reprice each factor's own input curve at $t \to 0$. (Corrected post-Phase-3: this previously calibrated every rate factor against one shared curve — live-verified against ORE's installed Cross-Asset Model that this is wrong, since every `IrLgm1fParametrization` there is constructed with its own `(Currency, YieldTermStructureHandle)` pair with no shared-curve code path at all.)
    *   Implement exact piecewise variance integration. ✅ closed-form HW1F transition variance per step (`_simulate_cross_asset_paths_jit`).

### Phase 2: Port Interest Rate Swaps (Done)
*   **Goal:** Build the foundational linear pricing engine.
*   **Tasks:** 
    *   Write a vectorized cash-flow mapping function. ✅ `engine/instruments/interest_rate_swap.py::price_swaps`.
    *   Consume the 4D Yield Curve cube to calculate Swap NPVs via tensor dot-products. ✅, with **full multi-curve discounting** (a `discount_curve_index` and a separate `forward_curve_index` per swap, into the yield curve cube's `NumRates` axis) — mirrors ORE's `DiscountingSwapEngine` + `IborIndex.forwardingTermStructure()` split rather than assuming a single curve.
    *   Output a structured `[Scenarios, TimeSteps, Trades]` NPV cube. ✅.
    *   Trade schedules and coupon accrual are built with ORE's own `MakeVanillaSwap`/day-count classes (see Stage 1 update above), not reimplemented — cross-checked directly against `ORE.VanillaSwap.NPV()` in `tests/test_interest_rate_swap.py` (`< 1e-6` relative tolerance).
    *   **Known limitation:** cashflow dates must land exactly on a configured `maturities` pillar (no curve interpolation yet) — the caller must configure the simulation's `rates.maturities` to be the union of both legs' payment/accrual dates.

### Phase 3: Port VaR (Value at Risk) Calculations (Done)
*   **Goal:** Implement the primary market risk metric.
*   **Tasks:** 
    *   Calculate Expected Shortfall (ES) and 99th percentile VaR over the NPV cube. ✅ `engine/aggregate_statistics/risk_statistics.py::value_at_risk`/`expected_shortfall`, formula reverse-engineered by live-testing `ORE.RiskStatistics` (lower/nearest-rank-below order statistic, NOT numpy's linearly-interpolated percentile; ES uses a strict value-based tail filter, NOT a positional slice — the two diverge whenever the tail has ties at the VaR boundary, cross-checked directly in `tests/test_risk_statistics.py`).
    *   Build sorting/percentile logic that remains highly efficient on GPUs. ✅ vectorized via `jnp.sort` + static (Python-level) percentile-derived indexing across the `TimeSteps` axis; instrument-agnostic (never imports `engine.instruments.*` — only assumes the `[Scenarios, TimeSteps, Trades]` NPV cube shape every pricer returns).
    *   **P&L baseline decision:** VaR/ES are computed against the portfolio's actual t=0 NPV (supplied explicitly by the caller from a separate zero-shock revaluation), applied at every simulated time step — matching ORE's literal historical-VaR P&L definition (`NPV(scenario) - NPV(base case)`) rather than each step's own cross-scenario mean, so the resulting profile reflects both market risk and the portfolio's expected drift over time.
    *   **Known limitation:** `expected_shortfall` returns `NaN` for any (percentile, time step) whose strict loss tail is empty — the same condition under which `ORE.RiskStatistics.expectedShortfall` itself raises `RuntimeError("no data below the target")`. JAX can't raise from traced code, so callers must check for `NaN` explicitly; this is documented behavior, not a silent gap.

### Phase 4: End-to-End System Validation
*   **Goal:** Prove the JAX engine works perfectly end-to-end and is mathematically precise.
*   **Tasks:** 
    *   Use the XML parser to ingest a standard test portfolio from ORE.
    *   Run the JAX FP64 pipeline and compare the resulting simulation paths, NPV cubes, and VaR figures against ORE's standard C++ output.
    *   Ensure exact pathwise and statistical parity.

### Phase 5: Port European Swaptions (Done)
*   **Goal:** Introduce non-linear derivative pricing.
*   **Tasks:** 
    *   Implement Jamshidian's trick or Black's model mapped to the simulated HW paths. ✅ `engine/instruments/european_swaption.py::price_swaptions` -- Jamshidian's decomposition (a coupon-bond option decomposed into a portfolio of zero-coupon bond options, each closed-form under HW1F), the same formula `ORE.JamshidianSwaptionEngine` uses. Formula reverse-engineered by live-testing the installed ORE package directly (not derived from a textbook), matching `ORE.JamshidianSwaptionEngine.NPV()` to a relative precision of 1e-6 or better across payer/receiver, ITM/ATM/OTM, and multiple tenors/forward-start dates -- cross-checked directly in `tests/test_european_swaption.py`.
    *   Vectorize the calculation of the exercise boundary at maturity. ✅ the exercise boundary (Jamshidian's critical short rate `r*`) is solved via a vectorized bisection (`_solve_rstar`) across every `[Scenarios, TimeSteps]` entry at once under `jax.jit`, rather than a per-scenario scalar root-find.
    *   **Single-model pricing:** unlike the linear swap pricer's independent discount/forward curves, Jamshidian's trick prices the underlying swap and the option under one Hull-White factor (`rate_factor_index`) -- mirrors `ORE.JamshidianSwaptionEngine` itself, which takes a single `ShortRateModel` (there is no multi-curve Jamshidian formula in ORE either).
    *   **Conditional (future-time) pricing:** produces the same `[Scenarios, TimeSteps, Trades]` NPV cube contract as the swap pricer by evaluating Jamshidian's formula at every simulated (scenario, time step), conditional on that scenario's simulated short rate -- a direct consequence of HW1F's Markov property (the same `A(t,T)`/`B(t,T)` conditional bond-price formula the yield-curve cube already uses). This generalization was itself live-verified against ORE by rebuilding ORE's own evaluation date and implied curve at a later time and re-pricing with a fresh `JamshidianSwaptionEngine`.
    *   **Bug caught during development (forward-starting swaptions):** an early version assumed the underlying swap's floating leg always redeems its notional exactly at the option's own exercise date `T0` -- true only when the spot lag and "exercise lag" coincide (a non-forward-starting swaption), and off by ~1% vs. ORE for any genuinely forward-starting swaption, where the true first accrual date `T_start` is `exercise_lag_days` after `T0`. Fixed by adding the swap's own `P(T0,T_start)` discount factor as a genuine (signed) leg of the decomposition rather than assuming it equals 1; re-verified to match ORE across several forward-start tenors after the fix. See `tests/test_european_swaption.py::TestAgainstOREJamshidianEngine::test_matches_ore_forward_starting`.

### Phase 6: Port Bermudan Swaptions
*   **Goal:** Tackle early-exercise mechanics (The hardest Monte Carlo challenge).
*   **Tasks:** 
    *   Implement the Longstaff-Schwartz (American Monte Carlo) algorithm in JAX.
    *   Write a backward-induction loop utilizing GPU-accelerated polynomial regressions.

### Phase 7: Port XVA (Credit & Funding Valuation Adjustments)
*   **Goal:** Calculate counterparty risk exposures.
*   **Tasks:** 
    *   Convert the NPV cube into an Exposure cube (`max(NPV, 0)`).
    *   Aggregate Expected Exposure (EE) across the time grid to compute CVA and DVA.

### Phase 8: TraderX API Integration
*   **Goal:** Transition to a production microservice.
*   **Tasks:** 
    *   Wrap the ETL CPU layer in a FastAPI (REST) or gRPC service.
    *   Define Pydantic/Protobuf schemas for incoming TraderX portfolios and market data.
    *   Retain the XML parser as a dedicated test harness.

### Phase 9: The Compute-Precision Tradeoff Research
*   **Goal:** Execute the core research thesis.
*   **Tasks:** 
    *   Establish the JAX FP64 engine as the "Ground Truth."
    *   Run JAX FP32 and BF16 at 20,000+ scenarios and compare the statistical convergence of VaR/ES against the FP64 baseline at lower scenario counts.
    *   Benchmark memory usage, GPU utilization, and execution time across precision profiles to find the optimal enterprise configuration.

## 4. Technical Constraints & Code Style
*   **Dynamic Precision:** Never hardcode `jnp.float64`. All functions must accept a `dtype` parameter to facilitate Phase 9 testing.
*   **No State Mutation:** JAX requires pure functions. Do not use in-place array updates (`x[0] = 1`). 
*   **Vectorization over Loops:** Never use standard `for` loops inside JIT-compiled functions. Use `jax.lax.scan` for chronological time-stepping and `jnp.einsum` or `jnp.where` for cross-sectional trade logic.
*   **API-First Design:** Write data-ingestion logic to expect dictionaries/JSON natively, utilizing the XML parser as an adapter for testing rather than a hard dependency.