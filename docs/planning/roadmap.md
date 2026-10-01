# Roadmap

The order in which to work through [known-issues.md](known-issues.md) and
[features.md](features.md). Every open item appears here exactly once; an item that is not
here is not planned ([rules](README.md#lifecycle)).

**How the order is decided.**

1. **Structure first**, where doing it later would redo work: changes that rewrite the files
   other fixes would touch.
2. **Correctness and precision**: wrong, unverified or silently imprecise numbers.
3. **Performance**, only once stage 2 has frozen the numbers: every performance change keeps
   the parity tests bit-identical in FP64, so optimizing numbers that are about to move is
   wasted.
4. **API robustness.**
5. **Tests and tooling.**
6. **Features**, in the order their dependencies allow.

Within a stage, order by the numbering. Items marked *parallel* touch code no earlier step
changes and can start at any time. Items waiting on someone outside the project are listed
[separately](#waiting-on-others), with who to chase.

## Where things stand

One pipeline, ORE's: a cross-asset model with a model per currency (the LGM by default, or
Hull-White, both ORE's `<LGM>` parametrizations), every trade valued on every path as ORE's
valuation engine does. It matches ORE component by component: t=0 prices to 1e-14 (swaps,
Europeans, bonds) and 4e-11 (calibrated Bermudans/Americans), model analytics to 1e-12
(the Hull-White model's bonds against QuantLib's `HullWhite` too), pricers on path curves to
1e-8 – 1e-12 under either model, market-risk VaR/ES per scenario to 2e-13. What is not yet
shown is that the assembled simulation, exposure and sensitivities equal an ORE run. Every
model, engine, Greeks and precision choice is one run configuration with ORE's defaults; only
the simulation's precision is adjustable until 1.4. Nothing runs on more than one device.
The TraderX EOD boundary prices Treasuries end to end and refuses everything else by name.

---

<a id="stage-1--structure"></a>
## Stage 1 — Structure

The configurable engine (decision A-1). The run configuration (`RunConfig`, step 1.2) and the
Hull-White model as one of its options on the shared pipeline, with trades that carry no model
and name themselves (step 1.3, 2026-10-01), are done; 1.4 moves the precision mechanism onto
it. Fixing the precision mechanism, the Greeks recompiles or the API before 1.4 would be
redone. Design: [details/configurable-engine.md](details/configurable-engine.md).

| Step | Work | Closes | Size |
|---|---|---|---|
| 1.4 | Explicit dtypes from the configuration in every stage (scenario market, legs, per-path Bermudan engine, Greeks, calibration), so pricing, risk and calibration become adjustable for either model and `check_run`'s refusal goes; then remove `_PRICING_LOCK`, `run_market_risk`'s flag set and the per-precision pool tiers (x64 is already on once per process since 1.3) | [I-55](known-issues.md#i-55) (stages, mechanism) | M |

Exit: every step's defaults reproduce the previous step's numbers bit for bit (the shared
portfolio and the parity suites). 1.3 met it: 103 of 103 saved arrays identical, and each
Hull-White fix measured red on the code before it, on a sloped curve
([known-issues.md](known-issues.md#verification-status)).

<a id="stage-2--correctness-and-precision"></a>
## Stage 2 — Correctness and precision

| Step | Work | Closes | Size |
|---|---|---|---|
| 2.1 | *Parallel, do first.* EOD boundary: check submission binding before any cache return; one execution owner per workload; reject unknown calculations and non-USD reporting currency | [I-57](known-issues.md#i-57), [I-58](known-issues.md#i-58), [I-59](known-issues.md#i-59) | S |
| 2.2 | Generalize the oracle to an OREApp XVA run; L4 distribution parity of exposure profiles; L3 path parity once gate V-4 closes. Fix the oracle's first-segment curve while in that file | [I-50](known-issues.md#i-50), [I-34](known-issues.md#i-34) | L |
| 2.3 | Sensitivities against ORE's sensitivity analytic on the shared portfolio | [I-51](known-issues.md#i-51) | M |
| 2.4 | Reproduce ORE's two per-path recalibration details, measured against 2.2's cube | [I-49](known-issues.md#i-49) | M |
| 2.5 | `ShiftHorizon` as a setting; parity at 0.5; then 0.5 as the default | [I-32](known-issues.md#i-32) | M |
| 2.6 | Swaption vol strike axis, read at each option's and helper's strike | [I-54](known-issues.md#i-54) | M |
| 2.7 | Precision evidence table per figure and precision; warning on unproven combinations | [I-55](known-issues.md#i-55) (warnings) | M |
| 2.8 | Report the worker's device and realised dtypes on each result | [I-12](known-issues.md#i-12) | S |

Order within the stage: 2.2 before 2.4 (2.4 needs 2.2's cube); 2.5 and 2.6 change default or
calibrated numbers, so they finish before stage 3. 2.7 needs 1.4. Parity methodology:
[details/ore-parity-validation.md](details/ore-parity-validation.md).

<a id="stage-3--performance"></a>
## Stage 3 — Performance

Rule: every change leaves the FP64 parity tests bit-identical.

| Step | Work | Closes | Size |
|---|---|---|---|
| 3.1 | Profile a portfolio job, then remove the dominant costs (per-path recalibration, Python bump loops, recompiles per process) | [I-53](known-issues.md#i-53) | M |
| 3.2 | Shard the scenario axis across devices; device-aware pool sizing on a Cloud TPU VM | [I-61](known-issues.md#i-61) | L |
| 3.3 | Pass the calibration's market prices as traced arguments (6 → 1 compile per basket), then memoize the AD Greeks' jitted closures (re-measured after 1.3: 30 programs per repeated call, 28 of them calibrations) | [I-22](known-issues.md#i-22), [I-21](known-issues.md#i-21) | S |

<a id="stage-4--api-robustness"></a>
## Stage 4 — API robustness

| Step | Work | Closes | Size |
|---|---|---|---|
| 4.1 | One route (the old names as aliases; the request is already one shape since 1.3), a market-risk route, the CAM calibration route, and a completeness test failing on any configuration setting without an API field. Every per-trade result row carries its trade id; the cube returns as a chunked artifact reference | [I-56](known-issues.md#i-56), [I-10](known-issues.md#i-10) (results), [I-09](known-issues.md#i-09) | L |
| 4.2 | Durable portfolio jobs with failure classes; EOD accepted-attempt record, boot sweep, `interrupted` state | [I-08](known-issues.md#i-08) | M |
| 4.3 | Chase TraderX's answers; apply them (a widened allowlist, a schema statement) | [I-23](known-issues.md#i-23), [I-60](known-issues.md#i-60) | S |

<a id="stage-5--tests-and-tooling"></a>
## Stage 5 — Tests and tooling

| Step | Work | Closes | Size |
|---|---|---|---|
| 5.1 | Characterize the full-suite XLA abort (repeated full runs against a known-bad baseline; `tests/test_api.py`'s pool shutdown is in since 1.3); make the cross-tier concurrency test deterministic | [I-27](known-issues.md#i-27) | M |
| 5.2 | Ruff in `pyproject.toml` and CI, then a type checker on `engine/` | [I-66](known-issues.md#i-66) | S |
| 5.3 | Shared test helpers in `tests/support/`; public-entry tests where stage 1 made private-symbol tests obsolete | [I-67](known-issues.md#i-67) | S |

5.1's runs can be made any time a full run is being made anyway; only the conclusion waits
for repeated runs.

<a id="stage-6--features"></a>
## Stage 6 — Features

| Step | Work | Feature | Can start after |
|---|---|---|---|
| 6.1 | *Parallel, start now.* Basel P0 (decisions, pinned text, profile, traceability) and P2 data acquisition, which is calendar time | [F-05](features.md#f-05) | — |
| 6.2 | Engine options: ORE's `AnalyticLgm` European engine, settlement methods, FD solver (the AD Greeks method and the market-risk engine by configuration are done) | [F-01](features.md#f-01) | — |
| 6.3 | Sub-FP32 tiers: inverse-CDF kernel at FP64/FP32 first, then FP16 storage | [F-07](features.md#f-07) | 1.4, 2.7 |
| 6.4 | FX and equity trades on the market path; FX/EQ calibration | [F-04](features.md#f-04) | — |
| 6.5 | SABR volatility | [F-02](features.md#f-02) | 2.6 |
| 6.6 | Basel P1 (FRTB-SA) onward; CVA/DVA | [F-05](features.md#f-05), [F-06](features.md#f-06) | 2.3; 2.2 |
| 6.7 | AMC engine | [F-03](features.md#f-03) | 2.2 |

<a id="waiting-on-others"></a>
## Waiting on others

Not engineering work until the input arrives. Chase the dependency, not the code.

| Item | Waiting for | From |
|---|---|---|
| [I-05](known-issues.md#i-05) USD-SOFR swaps | The D03/D04 convention set | TraderX |
| [I-04](known-issues.md#i-04) seasoned TraderX swaps | `pastFixings` in the export | TraderX |
| [I-16](known-issues.md#i-16) per-pillar `rateSensitivity` | An observed market-data package (W2) | TraderX |
| [I-18](known-issues.md#i-18) equity positions | A spot/FX source | TraderX or a market-data decision |
| [I-23](known-issues.md#i-23), [I-60](known-issues.md#i-60) | Answers on versioning and added fields | TraderX (then step 4.3) |
| [I-07](known-issues.md#i-07) corporate bonds, listed options | Demand, and for corporates a credit model | Owner |

Open TraderX asks are collected in
[details/traderx-integration.md](details/traderx-integration.md#open-with-traderx).
