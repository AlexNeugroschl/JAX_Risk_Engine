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

## 3. Development Roadmap
We will build this system out in the following sequential phases:

### Phase 1: Port the Market Simulations to JAX (Done)
*   **Goal:** Finalize the exact replica of ORE's Cross-Asset Model (CAM).
*   **Tasks:** 
    *   Extract ORE’s exact recursive Brownian Bridge matrix. ✅ `engine/market_simulations.py::_build_bridge_matrix` — QuantLib/ORE's recursive bisection construction, verified to exactly reproduce Brownian motion's covariance structure.
    *   Extract calibrated today's yield curves for the Hull-White $A(t,T)$ parameter. ✅ `compute_hw_A_matrix`, calibrated **independently per rate factor** against that factor's own curve (`rates.initial_zero_curves`, one per factor), verified to exactly reprice each factor's own input curve at $t \to 0$. (Corrected post-Phase-3: this previously calibrated every rate factor against one shared curve — live-verified against ORE's installed Cross-Asset Model that this is wrong, since every `IrLgm1fParametrization` there is constructed with its own `(Currency, YieldTermStructureHandle)` pair with no shared-curve code path at all.)
    *   Implement exact piecewise variance integration. ✅ closed-form HW1F transition variance per step (`_simulate_cross_asset_paths_jit`).

### Phase 2: Port Interest Rate Swaps (Current)
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

### Phase 5: Port European Swaptions
*   **Goal:** Introduce non-linear derivative pricing.
*   **Tasks:** 
    *   Implement Jamshidian's trick or Black's model mapped to the simulated HW paths.
    *   Vectorize the calculation of the exercise boundary at maturity.

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