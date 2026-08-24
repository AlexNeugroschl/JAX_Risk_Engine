# Roadmap

This page tracks the project's phased build-out. For current architecture, see
[Architecture](../concepts/architecture.md); for what's implemented right now, see the
[Overview](../getting-started/overview.md).

| Phase | Goal | Status |
|---|---|---|
| 1. Market simulation | Port ORE's Cross-Asset Model (Sobol QMC, Brownian bridge, Hull-White 1F, GBM) to JAX | ✅ Done |
| 2. Interest rate swaps | Vectorized linear swap pricing, multi-curve discounting | ✅ Done |
| 3. VaR / Expected Shortfall | Risk aggregation over the NPV cube, matching `ORE.RiskStatistics` | ✅ Done |
| 4. End-to-end validation | Full-pipeline parity check against ORE on a mixed portfolio | ✅ Done |
| 5. European swaptions | Jamshidian's decomposition under Hull-White 1F | ✅ Done |
| 6. Bermudan & American swaptions | Numeric LGM backward induction (Hagan convolution), matching ORE's actual production engine | ✅ Done |
| 7. Greeks (Delta, Gamma, Theta, Vega) | Curve-pillar and volatility sensitivities for swaps, European swaptions, and Bermudan/American swaptions via JAX autodiff, matching ORE's bump-and-revalue convention | ✅ Done |
| 8. Models/trades consolidation | Shared Hull-White/LGM math and ORE trade-building in `engine/models/` | ✅ Done |
| 9. LGM calibration | Bootstrap-fit a piecewise LGM `Sigma` to market swaption vols, matching `ore::data::LgmBuilder::calibrate()` | ✅ Done — see [Calibration](../reference/calibration.md) |
| 10. XVA (CVA/DVA) | Convert NPV cube to exposure, aggregate expected exposure | 🔜 Planned |
| 11. TraderX API integration | FastAPI microservice wrapping the pricing pipeline (`engine/portfolio/request.py::price_portfolio` + `engine/api/`) — gRPC was evaluated and explicitly not chosen | ✅ Done — see [TraderX Integration Plan](traderx-integration.md), [The Portfolio Entry Point](../reference/portfolio-entrypoint.md), [HTTP API](../reference/http-api.md) |
| 12. Compute-precision research | 3-knob (simulation/pricing/risk) FP64/FP32 control via `PrecisionConfig`, reachable from `PortfolioRequest` and the HTTP API — see [Architecture](../concepts/architecture.md#adjustable-precision), [The Portfolio Entry Point](../reference/portfolio-entrypoint.md#precisionconfig). This is the project's core research question — whether running *many more* lower-precision simulations reaches the same risk answer, in the same wall-clock time, as running *fewer* high-precision ones, across multiple TPU devices running concurrently (see [Overview](../getting-started/overview.md)) — and this phase's work uncovered and fixed a live `jax_enable_x64` cross-thread race in `price_portfolio` as a prerequisite (`_PRICING_LOCK`, added as a correctness fix — see [Architecture](../concepts/architecture.md)), which phase 13 below then replaced as the *primary* concurrency mechanism with a genuinely concurrent multi-process design, since a single process-wide lock directly capped the whole process to one device's worth of work in flight, contradicting the multi-TPU goal this phase exists to serve. bfloat16/float16 confirmed broken on this stack (`jnp.linalg.cholesky`/`jax.scipy.stats.norm.ppf` both raise) and documented as a blocker; FP8/INT8/INT4/NF4 still planned | ✅ 3-knob FP64/FP32 done — lower-precision formats 🔜 Planned |
| 13. Multi-process worker pool for cross-device concurrency | Replaced the single-process, `_PRICING_LOCK`-serialized dispatch model with `engine/portfolio/worker_pool.py`: one `ProcessPoolExecutor`-backed pool per precision tier (float32/float64), each worker pinning its own fixed `jax_enable_x64` state (and, on real hardware, its own device) once at process boot — the only model where two different precision tiers can run genuinely concurrently rather than time-sliced behind one process-global JAX/XLA flag. `engine/api/routes.py` now dispatches through `submit_pricing_job`, keeping a `job_id -> Future` table instead of a plain in-process result dict. `_PRICING_LOCK` itself is unchanged in code — kept as narrower defense-in-depth (see [Architecture](../concepts/architecture.md)) rather than removed. Verified CPU-provable on this dev machine (single `CpuDevice`) via a genuine wall-clock overlap test (`tests/test_worker_pool.py::TestWorkerPoolConcurrency`); real TPU-hardware validation (device-count-aware pool sizing, `JAX_PLATFORMS=tpu`/`TPU_VISIBLE_CHIPS` wiring on an actual Cloud TPU VM) is deferred to actual deployment, not designed here | ✅ Done (CPU-provable) — real TPU deployment/sizing 🔜 Planned |

## Validation

`tests/test_end_to_end.py` prices a mixed portfolio through this engine's complete
pipeline and, independently, through real ORE objects conditioned on the exact same
simulated short-rate values, isolating pricing/risk correctness from RNG differences.
Per-scenario portfolio NPV matches ORE to better than `1e-3` relative error across the
full simulated distribution, and VaR/ES match to `1e-3` relative tolerance.

See [ORE Parity](../reference/ore-parity.md) for the full formula-by-formula cross-check
against ORE's own source, and [Calibration](../reference/calibration.md) and
[Delta, Gamma, and Theta](../risk/greeks.md) for the calibration and Greeks methodology.
