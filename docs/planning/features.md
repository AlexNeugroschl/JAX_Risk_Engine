# Features

Additive work: capability beyond today's scope. The engine is correct and honest without
any of it. Defects and shortcomings are in [known-issues.md](known-issues.md); the order of
work is in [roadmap.md](roadmap.md#stage-6--features); the rules are in
[README.md](README.md).

Size: **S** ≤ 3 days, **M** ≤ 2 weeks, **L** longer.

| ID | Feature | Size | Depends on | Stage |
|---|---|---|---|---|
| [F-01](#f-01) | Engine options beyond ORE's defaults | M | Roadmap 1.2 | 6.2 |
| [F-02](#f-02) | SABR swaption volatility | M | I-54 | 6.5 |
| [F-03](#f-03) | ORE's AMC engine as a valuation option | L | Roadmap 1.2 | 6.7 |
| [F-04](#f-04) | FX and equity trades on the market path; FX/EQ vol calibration | L | Roadmap 1.2 | 6.4 |
| [F-05](#f-05) | Basel III regulatory figures | L | See entry | 6.1, 6.6 |
| [F-06](#f-06) | CVA/DVA from the exposure profiles | M | I-50 | 6.6 |
| [F-07](#f-07) | Sub-FP32 precision tiers (FP16, bfloat16) | L | I-55, I-61 | 6.3 |

---

<a id="f-01"></a>
### F-01 — Engine options beyond ORE's defaults

**Value.** Decision A-1 makes the engine configurable, with ORE's defaults; these are the
non-default options the owner asked for, each a choice in the run configuration of
[I-68](known-issues.md#i-68).

- **Greeks method** (A-5): AD as an option beside ORE's bump-and-revalue, on either model. A
  test checks they agree to O(bump²) by halving the bump.
- **Settlement method** (A-6): a trade field with ORE's values and defaults (`PhysicalOTC`,
  `CollateralizedCashPrice`, `ParYieldCurve`, ...). Low priority.
- **Bermudan/American solver** (A-3): ORE's `LgmFdSolver` beside the Grid solver. ORE's two
  solvers disagree by up to 1e-3 on broken-period exercise, so understand that gap first.
- **Market-risk engine per product** (A-8): `engine.market_risk` chooses a European's engine
  from the configuration, not from whether the trade carries Hull-White fields.

**Depends on.** Roadmap 1.2. **Size.** M. **Details.**
[details/configurable-engine.md](details/configurable-engine.md).

<a id="f-02"></a>
### F-02 — SABR swaption volatility

**Value.** ORE offers SABR smiles beside volatility cubes. After [I-54](known-issues.md#i-54)
adds a strike axis, SABR is a second way to supply it.

**Depends on.** I-54. **Size.** M.

<a id="f-03"></a>
### F-03 — ORE's AMC engine as a valuation option

**Value.** ORE's American Monte Carlo exposure engine is its alternative to classic
revaluation, and much cheaper for Bermudans than per-path recalibration
([I-53](known-issues.md#i-53)). Out of scope so far (X-6); a candidate simulation option under
decision A-1. Parity against ORE's AMC analytic.

**Depends on.** Roadmap 1.2; I-50's oracle for parity. **Size.** L.

<a id="f-04"></a><a id="x-10"></a><a id="x-11"></a>
### F-04 — FX and equity trades on the market path; FX/EQ vol calibration

**Value.** The cross-asset model already simulates FX and equity (`engine.simulation.cam`),
but no FX or equity trade prices on the market path (X-11), and FX/EQ volatilities are
constant inputs rather than calibrated to options as `CrossAssetModelBuilder` does (X-10).
Both decided: close eventually, not urgent. Brings the two-currency end-to-end test (layer
L6) within reach.

**Depends on.** Roadmap 1.2. Equity positions from TraderX additionally need
[I-18](known-issues.md#i-18)'s market data. **Size.** L.

<a id="f-05"></a>
### F-05 — Basel III regulatory figures

**Value.** Regulatory figures computed as the Basel Framework specifies, each traceable to
its paragraph, reproduced by an independent oracle, and refused outside scope: FRTB-SA first
(SBM, DRC, RRAO), then historical data and IMA expected shortfall, backtesting and PLA,
SA-CCR and BA-CVA, a precision gate per figure, and an evidence pack.

**Depends on.** P0 and P2's data acquisition can start any time (P2 is calendar time).
P1 (FRTB-SA) needs [I-51](known-issues.md#i-51) (sensitivities proven against ORE).
P5's IMM needs the exposure proven ([I-50](known-issues.md#i-50)). USD swaps need
[I-05](known-issues.md#i-05). P6's precision gate reuses
[I-55](known-issues.md#i-55)'s evidence table. PLA needs an independent front-office P&L from
TraderX (decision D-7).

**Size.** L (about 26 engineer-weeks). **Details.** [details/basel-iii.md](details/basel-iii.md).

<a id="f-06"></a>
### F-06 — CVA/DVA from the exposure profiles

**Value.** The market path produces ORE's exposure profiles (EPE, ENE, EE_B, EEPE_B, PFE);
CVA/DVA is the next step of ORE's XVA analytic: default curves per counterparty and the
bank, netting sets, and the integral of discounted exposure against default probability.
Parity against ORE's XVA analytic through the same oracle as I-50. The regulatory CVA
(BA-CVA, SA-CVA) is part of F-05.

**Depends on.** I-50 (exposure proven first). **Size.** M.

<a id="f-07"></a>
### F-07 — Sub-FP32 precision tiers (FP16, bfloat16)

**Value.** The research goal: whether many low-precision paths match fewer FP64 paths in
equal wall time. Measured so far: `norm.ppf` and `cholesky` have no kernel below float32,
`matmul` works down to FP4, and computing in float32 while storing low works at every tier.
Each storage format caps the useful path count (float16 about 23M, bfloat16 about 359k,
FP8 about 1,400), so FP16 storage is the tier worth building; FP8/FP4 are rejected for path
storage with the measured reason.

**Depends on.** [I-55](known-issues.md#i-55) (explicit dtypes and the evidence table);
the throughput case needs TPUs ([I-61](known-issues.md#i-61)). **Size.** L.
**Details.** [details/sub-fp32-precision.md](details/sub-fp32-precision.md).
