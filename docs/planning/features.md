# Features

Additive work: capability beyond today's scope. The engine is correct and honest without
any of it. Defects and shortcomings are in [known-issues.md](known-issues.md); the order of
work is in [roadmap.md](roadmap.md); the rules are in
[README.md](README.md).

Size: **S** ≤ 3 days, **M** ≤ 2 weeks, **L** longer.

| ID | Feature | Size | Depends on | Stage |
|---|---|---|---|---|
| [F-01](#f-01) | Engine options beyond ORE's defaults | M | Roadmap 3.7; the FD solver also I-32 | 4.6 |
| [F-02](#f-02) | SABR swaption volatility | M | I-54, roadmap 3.7 | 4.8 |
| [F-03](#f-03) | ORE's AMC engine as a valuation option | L | Roadmap 3.7 | 4.10 |
| [F-04](#f-04) | FX and equity trades on the market path; FX/EQ vol calibration | L | Roadmap 3.7 | 4.7 |
| [F-05](#f-05) | Basel III regulatory figures | L | See entry | 4.4, 4.11 |
| [F-06](#f-06) | CVA/DVA from the exposure profiles | M | — | 4.9 |
| [F-07](#f-07) | Precision below float32, down to FP8 (FP4 later), per stage, product and trade | L | I-55 | 3.7, 5.1 to 5.3 |
| [F-08](#f-08) | Reporting currencies other than USD at the EOD boundary | S | I-18 (an FX source), I-59 | 4.12 |
| [F-09](#f-09) | ORE's XML input files as a run request | M | Roadmap 3.9 | 4.13 |

---

<a id="f-01"></a>
### F-01 — Engine options beyond ORE's defaults

**Value.** Decision A-1 makes the engine configurable, with ORE's defaults; these are the
non-default options the owner asked for, each a choice in the run configuration
(`RunConfig`, `engine/portfolio/config.py`). Done so far: the **Greeks method** (A-5,
`GreeksConfig.method="AD"` for any model and engine, beside ORE's bump-and-revalue), the
**Jamshidian European engine** on a configured Hull-White model (`PricingConfig.european=
"Jamshidian"` with `PricingConfig.jamshidian`, whichever model simulates), the **Hull-White
model** per currency (`HullWhiteConfig`, roadmap 1.3), and the **market-risk engine per
product** (A-8: `engine.market_risk` takes the `PricingConfig`). Left:

- **ORE's `AnalyticLgmSwaptionEngine`** for Europeans on the simulated LGM, beside Bachelier
  and Jamshidian.
- **Settlement method** (A-6): a trade field with ORE's values and defaults (`PhysicalOTC`,
  `CollateralizedCashPrice`, `ParYieldCurve`, ...). Low priority.
- **Bermudan/American solver** (A-3): ORE's `LgmFdSolver` beside the Grid solver. ORE's two
  solvers disagree by up to 1e-3 on broken-period exercise, so understand that gap first.

The AD Greeks method's agreement with the bump method on sloped curves is an open issue,
[I-78](known-issues.md#i-78).

**Depends on.** Roadmap 3.7: each option adds a path-pricing kernel, written in 3.7's form
(A-16). The FD solver also needs [I-32](known-issues.md#i-32)'s shift. **Size.** M.
**Details.** [details/configurable-engine.md](details/configurable-engine.md).

<a id="f-02"></a>
### F-02 — SABR swaption volatility

**Value.** ORE offers SABR smiles beside volatility cubes. After [I-54](known-issues.md#i-54)
adds a strike axis, SABR is a second way to supply it.

**Depends on.** I-54; roadmap 3.7 (A-16). **Size.** M.

<a id="f-03"></a>
### F-03 — ORE's AMC engine as a valuation option

**Value.** ORE's American Monte Carlo exposure engine is its alternative to classic
revaluation, and much cheaper for Bermudans than per-path recalibration
([I-53](known-issues.md#i-53)). Out of scope so far (X-6); a candidate simulation option under
decision A-1. Parity against ORE's AMC analytic.

**Depends on.** The ORE simulation oracle (`tests/support/ore_xva_oracle.py`, roadmap 3.2) for parity; roadmap 3.7 (A-16). It adds a simulation option to `RunConfig`
(classic revaluation is the only one today). **Size.** L.

<a id="f-04"></a><a id="x-10"></a><a id="x-11"></a>
### F-04 — FX and equity trades on the market path; FX/EQ vol calibration

**Value.** The cross-asset model already simulates FX and equity (`engine.simulation.cam`),
but no FX or equity trade prices on the market path (X-11), and FX/EQ volatilities are
constant inputs rather than calibrated to options as `CrossAssetModelBuilder` does (X-10).
Both decided: close eventually, not urgent. Brings the two-currency end-to-end test (layer
L6) within reach.

**Depends on.** Roadmap 3.7: the new trades' path kernels are written in its form (A-16).
Equity positions from TraderX additionally need
[I-18](known-issues.md#i-18)'s market data. **Size.** L.

<a id="f-05"></a>
### F-05 — Basel III regulatory figures

**Value.** Regulatory figures computed as the Basel Framework specifies, each traceable to
its paragraph, reproduced by an independent oracle, and refused outside scope: FRTB-SA first
(SBM, DRC, RRAO), then historical data and IMA expected shortfall, backtesting and PLA,
SA-CCR and BA-CVA, a precision gate per figure, and an evidence pack.

**Depends on.** P0 and P2's data acquisition can start any time (P2 is calendar time).
P1 (FRTB-SA) needs [I-51](known-issues.md#i-51) (sensitivities proven against ORE).
P5's IMM needs the exposure proven: done for linear trades by roadmap 3.2, Bermudans and Americans by 3.5 ([I-49](known-issues.md#i-49)). USD swaps need
[I-05](known-issues.md#i-05). P6's precision gate reuses
[I-55](known-issues.md#i-55)'s evidence table. PLA needs an independent front-office P&L from
TraderX (decision D-7).

**Size.** L (about 26 engineer-weeks). **Details.** [details/basel-iii.md](details/basel-iii.md).

<a id="f-06"></a>
### F-06 — CVA/DVA from the exposure profiles

**Value.** The market path produces ORE's exposure profiles (EPE, ENE, EE_B, EEPE_B, PFE);
CVA/DVA is the next step of ORE's XVA analytic: default curves per counterparty and the
bank, netting sets, and the integral of discounted exposure against default probability.
Parity against ORE's XVA analytic through the ORE simulation oracle (roadmap 3.2), with credit curves added. The regulatory CVA
(BA-CVA, SA-CVA) is part of F-05.

**Depends on.** The exposure proven against ORE (roadmap 3.2; Bermudans and Americans 3.5). **Size.** M.

<a id="f-07"></a>
### F-07 — Precision below float32, down to FP8 (FP4 later), per stage, product and trade

**Value.** The research goal: whether many low-precision paths match fewer FP64 paths in
equal wall time. Storage, compute and accumulate precision per adjustable stage (simulation,
scenario market, path pricing), overridable per product and per trade (A-15; done in roadmap
1.5, 2026-10-02: `Precision.by_product`, `by_trade`, `precision_for`); sub-32-bit storage with
block scales and nearest or stochastic rounding (done in roadmap 1.6, 2026-10-02: float16,
bfloat16, `float8_e4m3fn`, `float8_e5m2`; `Stored`, `Precision.rounding`); compute below
float32 through kernels in difference form, matrix products where a kernel can be one, one
implementation for every precision (A-16, 3.7), with a product's own precision (TensorFloat-32,
bfloat16 passes, FP8) chosen by the policy (roadmap 2.3 makes every product state its precision
through one helper, never the device's default); the evidence table and warnings (5.1);
timing across devices on Ironwood and H100 (5.2, after the sharding of 3.8); FP4 on TPU 8t/8i (5.3). Measured so far: FP8
storage of the shocks biases a call payoff by about one Monte Carlo standard error at 4M
paths (stochastic rounding: 0.15), so the earlier rejection of FP8, which compared error per
value with Monte Carlo error, is withdrawn; FP4 stored naively is biased with either
rounding. Through the pipeline (1.6, [details/precision.md §15.3](details/precision.md#153-storage-through-the-pipeline)):
float16 storage is within 1e-5 of notional in every stage; FP8 is limited where an array's
level swamps its spread ([I-75](known-issues.md#i-75)).

**Depends on.** The mechanism and cast points (roadmap 1.4, done 2026-10-01; [I-55](known-issues.md#i-55));
the speed case needs the target hardware, which the owner has. **Size.** L.
**Details.** [details/precision.md](details/precision.md).

<a id="f-08"></a>
### F-08 — Reporting currencies other than USD at the EOD boundary

**Value.** The EOD submission's `reportingCurrency` lets a consumer ask for results in its
own currency, as ORE reports in the `baseCurrency` of its run, converting with its market's
FX. Decision A-18 refuses anything but USD until then
([I-59](known-issues.md#i-59), roadmap 4.1), so a request for EUR gets a 400 rather than
USD numbers labelled EUR. This feature turns that refusal into a conversion: t=0 figures
at the as-of FX spot, ORE's convention; the capability document's `reportingCurrencies`
lists each currency the FX source covers. Parity against an ORE run with that
`baseCurrency`.

**Depends on.** An FX source at the boundary ([I-18](known-issues.md#i-18), blocked on
TraderX or a market-data decision); I-59's refusal first, so the field is never silently
ignored meanwhile. **Size.** S once the FX source exists.

<a id="f-09"></a>
### F-09 — ORE's XML input files as a run request

**Value.** An ORE user can run an existing setup without rewriting it as JSON: `ore.xml`'s
analytics, `simulation.xml`, `pricingengine.xml`, `sensitivity.xml` and `portfolio.xml`,
translated into roadmap 3.9's run request. The engine and an ORE run could then be compared
on the same files. Whatever the engine does not support (a product, an engine, a model
parameter) is refused by name, never dropped. Nice to have, not critical (owner,
2026-10-08); nothing waits on it.

**Scope.** The configuration and portfolio files. ORE's market data only as zero-rate quotes
on `Zero` curve configurations, the form the ORE oracles already write
(`tests/support/ore_inputs.py`, which does the reverse translation). The engine takes zero
curves, so bootstrapping curves from instrument quotes is out of scope.

**Depends on.** Roadmap 3.9, whose configuration sections follow ORE's files, so the adapter
is a translation and adds no setting of its own. **Size.** M.
