# Features

Additive work: capability beyond today's scope. The engine is correct and honest without
any of it. Defects and shortcomings are in [known-issues.md](known-issues.md); the order of
work is in [roadmap.md](roadmap.md); the rules are in
[README.md](README.md).

Size: **S** ≤ 3 days, **M** ≤ 2 weeks, **L** longer.

| ID | Feature | Size | Depends on | Roadmap |
|---|---|---|---|---|
| [F-01](#f-01) | Engine options beyond ORE's defaults | M | F-07's kernel rewrite | New |
| [F-02](#f-02) | SABR swaption volatility | M | I-54, F-07's kernel rewrite | New |
| [F-03](#f-03) | ORE's AMC engine as a valuation option | L | F-07's kernel rewrite | New |
| [F-04](#f-04) | FX and equity trades on the pipeline; FX/EQ vol calibration | L | F-07's kernel rewrite | New |
| [F-05](#f-05) | Basel III regulatory figures | L | See entry | Basel |
| [F-06](#f-06) | CVA/DVA from the exposure profiles | M | I-87 | New |
| [F-07](#f-07) | Precision below float32, down to FP8 (FP4 later), per stage, product and trade | L | Comparison tooling done; target-device validation (I-121, I-97); I-61 for cluster claims | Comparison and research; FP4 and quantile correction New |
| [F-09](#f-09) | ORE's XML input files as a run request | M | I-87 | New |
| [F-10](#f-10) | Supported sampling confidence intervals with validated tail assumptions | L | I-111 availability rule; first precision timing result | New |
| [F-11](#f-11) | The cash flows paid in Theta's roll, reported beside Theta | S | I-87 | New |

---

<a id="f-01"></a>
### F-01 — Engine options beyond ORE's defaults

**Value.** Decision A-1 makes the engine configurable, with ORE's defaults; these are the
non-default options the owner asked for, each a choice in the run configuration
(`RunConfig`, `engine/run/config.py`). Done so far: the **Greeks method** (A-5,
`GreeksConfig.method="AD"` for any model and engine, beside ORE's bump-and-revalue), the
**Jamshidian European engine** on a configured Hull-White model (`PricingConfig.european=
"Jamshidian"` with `PricingConfig.jamshidian`, whichever model simulates), the **Hull-White
model** per currency (`HullWhiteConfig`), and the **market-risk engine per
product** (A-8: `engine.risk.market` takes the `PricingConfig`). Left:

- **ORE's `AnalyticLgmSwaptionEngine`** for Europeans on the simulated LGM, beside Bachelier
  and Jamshidian.
- **Settlement method** (A-6): a trade field with ORE's values and defaults (`PhysicalOTC`,
  `CollateralizedCashPrice`, `ParYieldCurve`, ...). Low priority.
- **Bermudan/American solver** (A-3): ORE's `LgmFdSolver` beside the Grid solver. ORE's two
  solvers disagree by up to 1e-3 on broken-period exercise, so understand that gap first.

The AD Greeks are ORE's in the limit of a vanishing shift on ORE's sensitivity market since
2026-10-17 ([I-78](known-issues.md#i-78)); a grid-priced option's are the derivatives of its
piecewise-smooth price, labelled on the result.

**Depends on.** [F-07](#f-07)'s kernel rewrite: each option adds a path-pricing kernel, written in
its form (A-16); the FD solver takes ORE's shift horizon, as the Grid solver does since I-32
(2026-10-12). **Size.** M.
**Details.** [details/configurable-engine.md](details/configurable-engine.md).

<a id="f-02"></a>
### F-02 — SABR swaption volatility

**Value.** ORE offers SABR smiles beside volatility cubes. After [I-54](known-issues.md#i-54)
adds a strike axis, SABR is a second way to supply it.

**Depends on.** I-54; [F-07](#f-07)'s kernel rewrite (A-16). **Size.** M.

<a id="f-03"></a>
### F-03 — ORE's AMC engine as a valuation option

**Value.** ORE's American Monte Carlo exposure engine is its alternative to classic
revaluation, and cheaper for Bermudans than a grid rollback on every path (the classic
simulation's way, which ORE's own exposure simulation and the engine take; since 2026-10-13 one
shared operator per step, [I-83](known-issues.md#i-83)). Out of scope so far (X-6); a candidate simulation option under
decision A-1. Parity against ORE's AMC analytic.

**Depends on.** The ORE simulation oracle (`tests/support/ore_xva_oracle.py`) for parity; [F-07](#f-07)'s kernel rewrite (A-16). It adds a simulation option to `RunConfig`
(classic revaluation is the only one today). **Size.** L.

<a id="f-04"></a><a id="x-10"></a><a id="x-11"></a>
### F-04 — FX and equity trades in the pipeline; FX/EQ vol calibration

**Value.** The cross-asset model already simulates FX and equity (`engine.models.cam`),
but no FX or equity trade prices in the pipeline (X-11), and FX/EQ volatilities are
constant inputs rather than calibrated to options as `CrossAssetModelBuilder` does (X-10).
Both decided: close eventually, not urgent. Brings the two-currency end-to-end test (layer
L6) within reach. With FX trades priced, market risk shocks FX spots as factors too: until
then it converts at the fixed as-of spot and says FX is not shocked (since 2026-10-11,
[I-100](known-issues.md#i-100)).

**Depends on.** [F-07](#f-07)'s kernel rewrite: the new trades' path kernels are written in
its form (A-16). Equity trades also need an equity spot in the market. **Size.** L.

<a id="f-05"></a>
### F-05 — Basel III regulatory figures

**Value.** Regulatory figures computed as the Basel Framework specifies, each traceable to
its paragraph, reproduced by an independent oracle, and refused outside scope: FRTB-SA first
(SBM, DRC, RRAO), then historical data and IMA expected shortfall, backtesting and PLA,
SA-CCR and BA-CVA, a precision gate per figure, and an evidence pack.

**Depends on.** P0 and P2's data acquisition can start any time (P2 is calendar time).
P1 (FRTB-SA)'s sensitivities are proven against ORE's sensitivity analytic (2026-10-17,
[I-51](known-issues.md#i-51)).
P5's IMM needs the exposure proven: done for linear trades on 2026-10-07, for Bermudans and Americans on 2026-10-12 ([I-49](known-issues.md#i-49)). USD swaps need
[I-05](known-issues.md#i-05). P6's precision gate reuses
[I-55](known-issues.md#i-55)'s evidence table (built 2026-10-11), adding the regulatory figures'
rows. PLA needs an independent front-office P&L from
TraderX (decision D-7).

**Size.** L (about 26 engineer-weeks). **Details.** [details/basel-iii.md](details/basel-iii.md).

<a id="f-06"></a>
### F-06 — CVA/DVA from the exposure profiles

**Value.** The pipeline produces ORE's exposure profiles (EPE, ENE, EE_B, EEPE_B, PFE);
CVA/DVA is the next step of ORE's XVA analytic: default curves per counterparty and the
bank, netting sets, and the integral of discounted exposure against default probability.
Parity against ORE's XVA analytic through the ORE simulation oracle (2026-10-07), with credit curves added. The regulatory CVA
(BA-CVA, SA-CVA) is part of F-05.

**Depends on.** The exposure proven against ORE (every trade of the shared portfolio since 2026-10-12, [I-49](known-issues.md#i-49)). **Size.** M.

<a id="f-07"></a>
### F-07 — Precision below float32, down to FP8 (FP4 later), per stage, product and trade

**Value.** The research goal: whether many low-precision paths match fewer FP64 paths in
equal wall time. Storage, compute and accumulate precision per adjustable stage (simulation,
scenario market, path pricing), overridable per product and per trade (A-15; done
2026-10-02: `Precision.by_product`, `by_trade`, `precision_for`); sub-32-bit storage with
block scales and nearest or stochastic rounding (done 2026-10-02: float16,
bfloat16, `float8_e4m3fn`, `float8_e5m2`; `Stored`, `Precision.rounding`); compute below
float32 through kernels in difference form, matrix products where a kernel can be one, one
implementation for every precision (A-16), with a product's own precision (TensorFloat-32,
bfloat16 passes, FP8) chosen by the policy (done 2026-10-10 for the simulation, the scenario
market, the legs, the Europeans and market risk: compute in float16, bfloat16, both FP8
formats, `tensorfloat32` and `bfloat16_3x`, accumulated in float32, each native or emulated by
device and named in the report; since 2026-10-11 the operand every path shares in two terms of
the format, the single pass `<format>_1x` a format of its own, A-23;
[details/precision.md §8](details/precision.md#8-compute-below-float32);
and on 2026-10-13 the Bermudan/American rollback, as matrix products in difference form, its
memory a few columns per path, [I-83](known-issues.md#i-83), and each option's flipped exercise
decisions measured by the harness, decision A-24); the evidence table and warnings (I-55,
done 2026-10-11: [compliance/precision-evidence.md](../../compliance/precision-evidence.md)),
each row keyed by its products' pricing settings since 2026-10-18 (I-114);
timing across devices on Ironwood and H100 (its harness built 2026-10-19,
`tests/support/precision_timing.py`, [details/precision.md §19](details/precision.md#19-the-research-timing-wall-time-per-figure-at-equal-accuracy-f-07-2026-10-19):
whole jobs cold and warm against float64 at equal accuracy measured over seeds, run end to end
on a CPU and an RTX 5060; the sharding across a pod slice's hosts built the same day, I-61, and
a GPU's FP8 native through cuBLAS, I-97); FP4 on TPU 8t/8i. Measured so far: FP8
storage of the shocks biases a call payoff by about one Monte Carlo standard error at 4M
paths (stochastic rounding: 0.15), so the earlier rejection of FP8, which compared error per
value with Monte Carlo error, is withdrawn; FP4 stored naively is biased with either
rounding. Through the pipeline (2026-10-02, [details/precision.md §15.3](details/precision.md#153-storage-through-the-pipeline)):
float16 storage is within 1e-5 of notional in every stage; FP8 was limited where an array's
level swamps its spread ([I-75](known-issues.md#i-75)), which block offsets remove
(2026-10-09). Since then the storage harness measures every format per stage, figure, product
and path count against float64 over 16 seeds ([details/precision.md
§15.4](details/precision.md#154-the-storage-harness-2026-10-09)): float16 cube storage stays
below a fifth of the statistical error to 16,384 paths, FP8 is many statistical errors off
but unbiased in a concentrated mean, and every format below float32 has a path ceiling. The
evidence table (2026-10-11, [details/precision.md
§15.6](details/precision.md#156-the-evidence-table-2026-10-11)) judges them by decision A-11's
1% rule.

**Deadline deliverable (owner, 2026-10-19).** First extend the existing accuracy, timing and
snapshot-reporting tools into one reproducible comparison of CPU FP64/FP32 references and
accelerator FP64/FP32, native FP8 and supported BF16. Report per-trade scenario PV and
portfolio P&L/VaR/ES errors beside whole-job cold/compile/warm time, memory and native/fallback
evidence. Use matched scenarios for precision error and several seeds/counts for
equal-accuracy timing, with the FP64 reference's sampling uncertainty stated. Begin with
a single trade and a small representative portfolio at existing measured pricing settings;
retain the fixed FP64 stages (A-10), counting their cost. Hardware qualification starts
alongside the tool. The first verified target device produces a bounded result while other
hardware and cluster checks continue; a correct cuBLAS route suffices for GPU FP8 without
waiting for an upstream compiler fix. Failures and no speedup remain reportable results.

**Comparison tooling complete, 2026-10-20.** [`benchmarks.comparison`](../../benchmarks/README.md)
composes the existing campaigns, timing and reporting tools. It retains named matched-scenario
errors, cold/compile/warm times, memory availability, observed output devices, dispatch labels,
exact reproduction settings and finite-reference sampling diagnostics together. Both bounded
workloads smoke-checked on CPU and the RTX 5060, with unavailable TPU cases and failed precision
targets retained; no warm compilation. The [execution plan and outcome](details/comparison-report.md)
describe the checks and limitations. This completes tooling, not target-device qualification,
native-unit proof, larger-count research or cluster scaling.

INT8, additional multi-pass methods/Qwix integration and reduced-precision today's PV or
Greeks are outside this deadline, not prerequisites or promised new capabilities. FP4,
quantile correction and supported confidence intervals stay separate later experiments.
Exhaustive sweeps and tuning follow only when the first measurements justify them.

**Depends on.** The mechanism and cast points (done 2026-10-01; [I-55](known-issues.md#i-55)).
Comparison tooling is available. Each speed claim needs the engine's numbers
shown right on that target device ([I-121](known-issues.md#i-121)), and native FP8 needs its
execution path verified ([I-97](known-issues.md#i-97), I-121). Cluster claims additionally need
scaling measured there ([I-61](known-issues.md#i-61)); a single-device report does not wait for it.
**Size.** L.
**Details.** [details/precision.md](details/precision.md).

<a id="f-09"></a>
### F-09 — ORE's XML input files as a run request

**Value.** An ORE user can run an existing setup without rewriting it as JSON: `ore.xml`'s
analytics, `simulation.xml`, `pricingengine.xml`, `sensitivity.xml` and `portfolio.xml`,
translated into the run request of [I-87](known-issues.md#i-87). The engine and an ORE run could then be compared
on the same files. Whatever the engine does not support (a product, an engine, a model
parameter) is refused by name, never dropped. Nice to have, not critical (owner,
2026-10-08); nothing waits on it.

**Scope.** The configuration and portfolio files. ORE's market data only as zero-rate quotes
on `Zero` curve configurations, the form the ORE oracles already write
(`tests/support/ore_inputs.py`, which does the reverse translation). The engine takes zero
curves, so bootstrapping curves from instrument quotes is out of scope.

**Depends on.** The run request ([I-87](known-issues.md#i-87)), whose configuration sections follow ORE's files, so the adapter
is a translation and adds no setting of its own. **Size.** M.

<a id="f-10"></a>
### F-10 — Supported sampling confidence intervals with validated tail assumptions

**Value.** Offer nominal sampling intervals for a stated estimator/model domain where
mathematical tail bounds or a validated model-specific method support them. I-111
currently declines nominal intervals for all figures and entries while retaining raw
variance diagnostics. This is explicit supported behavior, not a claim that rare-event
Student t intervals have become calibrated.

**Scope.** Establish the assumptions for the complete estimator, including paired
correction covariance and ES threshold estimation. Validate on independent outer seeds,
rare and unseen-event examples, numerical constants and zero-width misses. Preserve
the predeclared coverage >=0.90 and standardized-error RMS 0.75–1.33 criteria where
applicable; unavailable entries never pass by exclusion. A model-independent event-count
cutoff or fitted variance inflation is insufficient. Extend the shared interval contract
only for the validated domain, keeping other inputs explicitly unavailable.

**Depends on.** I-111's conservative availability rule (implemented). Scheduled after
the first existing-format timing/accuracy result so it does not block that research;
its uncertainty can be measured across independent outer runs. **Size.** L.
**Details.** [Public uncertainty contract](../reference/sampling-uncertainty.md),
[precision experiments](details/precision-accuracy.md).

<a id="f-11"></a>
### F-11 — The cash flows paid in Theta's roll, reported beside Theta

**Value.** Theta is ORE's, the change in NPV over the roll with nothing added back (decision
A-26, 2026-10-18), so a coupon paid in the roll shows as a fall in Theta by about the coupon. A
P&L explain or a carry with the cash received needs those flows: report them per trade, in the
reporting currency, as a figure of their own beside Theta (`theta + roll_cash_flows` is the
carry with the cash), from both Greek methods, so that Theta itself stays equal to ORE's
sensitivity analytic.

**Scope.** The flows of each trade paid in `(asof, thetaDate]` by ORE's `hasOccurred` rule on
the Theta date (coupons, a bond's redemption, a cash-settled option's settlement), each in the
currency it is paid and converted as the NPV is; no discounting, as ORE's add-back would have
summed them. Tested against the trades' own cash-flow schedules and, where ORE reports the
flows (its cash-flow report on the Theta date), against ORE. The matrix's cells for Theta gain
the figure's row.

**Depends on.** The run request's one result ([I-87](known-issues.md#i-87)), which names each
figure and its units. **Size.** S.
