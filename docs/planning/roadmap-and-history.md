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
| 8. Models/trades consolidation | Shared Hull-White/LGM math and ORE trade-building in `engine/models/`, `engine/trades/` | ✅ Done |
| 9. LGM calibration | Bootstrap-fit a piecewise LGM `Sigma` to market swaption vols, matching `ore::data::LgmBuilder::calibrate()` | ✅ Done — see [Calibration](../reference/calibration.md) |
| 10. XVA (CVA/DVA) | Convert NPV cube to exposure, aggregate expected exposure | 🔜 Planned |
| 11. TraderX API integration | FastAPI/gRPC microservice wrapping the pricing pipeline | 🔜 Planned — see [TraderX Integration Plan](traderx-integration.md) |
| 12. Compute-precision research | Statistical parity study of low-precision compute (down to 8-bit and 4-bit formats, e.g. FP8/INT8 and INT4/NF4) against FP64/FP32 baselines, at scale | 🔜 Planned |

## Validation

`tests/test_end_to_end.py` prices a mixed portfolio through this engine's complete
pipeline and, independently, through real ORE objects conditioned on the exact same
simulated short-rate values, isolating pricing/risk correctness from RNG differences.
Per-scenario portfolio NPV matches ORE to better than `1e-3` relative error across the
full simulated distribution, and VaR/ES match to `1e-3` relative tolerance.

See [ORE Parity](../reference/ore-parity.md) for the full formula-by-formula cross-check
against ORE's own source, and [Calibration](../reference/calibration.md) and
[Delta, Gamma, and Theta](../risk/greeks.md) for the calibration and Greeks methodology.
