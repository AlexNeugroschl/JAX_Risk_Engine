# Roadmap

The order in which to work through [known-issues.md](known-issues.md) and
[features.md](features.md). Every open item appears here exactly once; an item that is not
here is not planned ([rules](README.md#lifecycle)).

**How the order is decided.**

1. **Structure first**, where doing it later would redo work: changes that rewrite the files
   other fixes would touch.
2. **Correctness and precision**: wrong, unverified or silently imprecise numbers.
3. **Performance**, only once stage 2 has frozen the numbers: every performance change keeps
   the parity suites passing and records any float64 change it makes, so optimizing numbers
   that are about to move is wasted. (Removing recompiles could not wait: it made the test
   suite usable again, and was done ahead of 1.6.)
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
model, engine, Greeks and precision choice is one run configuration with ORE's defaults; since
1.4 the simulation, the scenario market and path pricing each take a storage and a compute
precision (float64 or float32), in portfolio and market-risk runs alike, with the float64
default bit for bit as before; since 1.5 path pricing is set per product and per trade too;
since 1.6 storage goes down to float16, bfloat16 and FP8, with block scales and nearest or
stochastic rounding; since 1.7 every result carries a precision report (the policy as run,
the formats read from the arrays, the device), and a paired float64 sample corrects the mean
figures by a two-level estimator and measures the quantiles.
Nothing runs on more than one device.
The TraderX EOD boundary prices Treasuries end to end and refuses everything else by name.

---

<a id="stage-1--structure"></a>
## Stage 1 — Structure

The configurable engine (decision A-1). The run configuration (`RunConfig`, step 1.2), the
Hull-White model as one of its options on the shared pipeline, with trades that carry no model
and name themselves (step 1.3), and the precision mechanism, `engine/precision/` with a
storage, compute and accumulate format per adjustable stage at float64/float32 (step 1.4,
2026-10-01), the pricing stage per product and per trade (step 1.5, 2026-10-02), and storage
below 32 bits, float16, bfloat16 and FP8 with block scales along the scenario axis and nearest
or stochastic rounding (step 1.6, 2026-10-02), and the paired float64 sample, the two-level
estimator and the precision report on every result (step 1.7, 2026-10-02), are done. Step 1.8
replaces the worker pool with one engine worker process.
Designs: [details/configurable-engine.md](details/configurable-engine.md),
[details/precision.md](details/precision.md) (decisions A-10 to A-16, D-9 revised
2026-10-01).

| Step | Work | Closes | Size |
|---|---|---|---|
| 1.8 | One engine worker process per host behind a durable SQLite job queue (A-14); delete the pools and the freeze/thaw of ORE objects. Not a main priority: nothing in stage 2 waits for it; 3.2 does | [I-72](known-issues.md#i-72), [I-08](known-issues.md#i-08) (portfolio jobs) | M |

Exit: every step's defaults reproduce the previous step's numbers bit for bit (the shared
portfolio and the parity suites; [details/precision.md §13.1](details/precision.md#131-bit-for-bit-and-ore-parity)).
1.3 met it: 103 of 103 saved arrays identical, and each Hull-White fix measured red on the
code before it, on a sloped curve ([known-issues.md](known-issues.md#verification-status)).
1.4 met it: 164 of 164 default-precision arrays identical (the LGM and Hull-White runs,
exposure, bump and AD Greeks, market risk), and the float32 scenario market and float32
market-risk revaluation identical to before. A float32 portfolio cube is not, by design: the
old `simulation=32` cube was priced in mixed precision, mostly float64
([details/precision.md §13.1](details/precision.md#131-bit-for-bit-and-ore-parity)). 1.5 met
it: 232 of 232 arrays identical, the float32 runs included, and a mixed run equals each trade
priced alone at its precision, column for column. 1.6 met it: 232 of 232 arrays identical
against the snapshot of the jit change, and the mixed run's equality holds with storage below
32 bits and stochastic rounding too. 1.7 met it: 232 of 232 arrays identical against a
worktree of `f0a438c` (1.6), and at float64 a paired sample measures exactly zero, its figures
the run's own bit for bit. 1.8 runs the full suite on Linux.

<a id="stage-2--correctness-and-precision"></a>
## Stage 2 — Correctness and precision

| Step | Work | Closes | Size |
|---|---|---|---|
| 2.1 | *Parallel, do first.* EOD boundary: check submission binding before any cache return; one execution owner per workload; reject unknown calculations and non-USD reporting currency | [I-57](known-issues.md#i-57), [I-58](known-issues.md#i-58), [I-59](known-issues.md#i-59) | S |
| 2.2 | Generalize the oracle to an OREApp XVA run; L4 distribution parity of exposure profiles; L3 path parity once gate V-4 closes. Fix the oracle's first-segment curve while in that file | [I-50](known-issues.md#i-50), [I-34](known-issues.md#i-34) | L |
| 2.3 | Sensitivities against ORE's sensitivity analytic on the shared portfolio | [I-51](known-issues.md#i-51) | M |
| 2.4 | Reproduce ORE's two per-path recalibration details, measured against 2.2's cube; warn, as ORE's `LgmBuilder` does, when a path's recalibration misses its basket | [I-49](known-issues.md#i-49), [I-73](known-issues.md#i-73) | M |
| 2.5 | `ShiftHorizon` as a setting; parity at 0.5; then 0.5 as the default | [I-32](known-issues.md#i-32) | M |
| 2.6 | Swaption vol strike axis, read at each option's and helper's strike | [I-54](known-issues.md#i-54) | M |
| 2.7 | *Parallel, can start now.* Measurement of the storage formats per class, product and path count, with 1.7's paired sample and its estimator's coverage on the pipeline; storage relative to a level where a class's level swamps its spread (the cube to its t=0 value, the curves to their path-independent part, or a block offset); the evidence table against the acceptance standard (A-11: Basel III's P&L attribution test and the Basel plan's P6.2 rule; P6.2 labelled as engineering for figures Basel does not cover); a warning on any result whose combination has no passing row | [I-55](known-issues.md#i-55) (warnings), [I-75](known-issues.md#i-75) | M |
| 2.8 | Kernels in difference form with explicit accumulators, one family at a time (simulation scan, scenario curves, legs, Europeans, Bermudan rollback and recalibration, exposure), one implementation for every precision (A-16); compute below float32 enabled. ORE parity at existing tolerances first, then the float64 snapshot re-baselined once | [F-07](features.md#f-07) (compute) | L |

Order within the stage: 2.2 before 2.4 (2.4 needs 2.2's cube); 2.5 and 2.6 change default or
calibrated numbers, so they finish before stage 3. 2.7 runs beside the rest of the stage: it
changes no float64 number. 2.8 comes after 2.4, 2.5 and 2.6, which change the same kernels, so
none is rewritten twice, and before stage 3, because it moves float64 numbers at rounding
level. Parity methodology: [details/ore-parity-validation.md](details/ore-parity-validation.md);
precision: [details/precision.md](details/precision.md).

<a id="stage-3--performance"></a>
## Stage 3 — Performance

Rule: every ORE parity suite passes at its existing tolerance after every change. A change
that moves float64 numbers re-baselines the golden snapshot and records the largest change
per array ([details/precision.md §13.1](details/precision.md#131-bit-for-bit-and-ore-parity)).
Jitting the pricers with the trade as a traced argument (2026-10-02, ahead of 1.6, which
closed I-21 and I-22) moved them at rounding level, the one such change so far.

| Step | Work | Closes | Size |
|---|---|---|---|
| 3.1 | Profile a portfolio job, then remove the remaining dominant costs (an American's per-path recalibration arithmetic, first-call compiles, compiles per worker process); recompiles per trade, date, bump and call went on 2026-10-02 | [I-53](known-issues.md#i-53) | M |
| 3.2 | Shard the scenario axis across the devices the engine worker (1.8) owns; one worker per host on a Cloud TPU pod slice | [I-61](known-issues.md#i-61) | L |
| 3.4 | Low-precision timing on Ironwood and H100: the storage formats, then matrix-product forms of the heavy kernels (leg pricing, Bermudan rollback) on native FP8; wall time per figure at equal accuracy, read from the evidence table's path ceilings | [F-07](features.md#f-07) (speed) | M |

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
| 5.1 | Characterize the full-suite XLA abort (repeated full runs against a known-bad baseline; `tests/test_api.py`'s pool shutdown is in since 1.3; the flaky cross-tier overlap test went with the tiers in 1.4) | [I-27](known-issues.md#i-27) | M |
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
| 6.3 | FP4: storage with a variance correction through the block scales, compute on TPU 8t/8i; multilevel estimation of quantiles (PFE, VaR, ES) | [F-07](features.md#f-07) (FP4) | 2.8, 3.4 |
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
