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
   suite usable again, and was done ahead of 1.6. Splitting a job's scenarios across a host's
devices went ahead too, on 2026-10-04: it changes no kernel's arithmetic, leaves one-device
runs bit for bit, and the research goal cannot be tested without it.)
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
figures by a two-level estimator and measures the quantiles; since 1.8 HTTP jobs go through a
durable SQLite job queue to one single-threaded engine worker process per host, which owns
the host's devices (decision A-14); since 2026-10-04 each job's scenarios are split across
them. Nothing runs on more than one host.
The TraderX EOD boundary prices Treasuries end to end and refuses everything else by name.

---

<a id="stage-1--structure"></a>
Stage 1, structure (the configurable engine, decision A-1, steps 1.2 to 1.8), is done
(2026-10-04): one run configuration, the precision mechanism through storage below 32 bits
and the precision report, and the engine worker behind the job queue. Each step reproduced
the previous step's numbers bit for bit; the evidence is in
[details/precision.md §13.1](details/precision.md#131-bit-for-bit-and-ore-parity) and the
designs in [details/configurable-engine.md](details/configurable-engine.md) and
[details/precision.md](details/precision.md).

<a id="stage-2--correctness-and-precision"></a>
## Stage 2 — Correctness and precision

| Step | Work | Closes | Size |
|---|---|---|---|
| 2.1 | *Parallel, do first.* API work, here for its severity (I-57 is High). EOD boundary: check submission binding before any cache return; one execution owner per workload; reject unknown calculations and non-USD reporting currency | [I-57](known-issues.md#i-57), [I-58](known-issues.md#i-58), [I-59](known-issues.md#i-59) | S |
| 2.2 | Generalize the oracle to an OREApp XVA run; L4 distribution parity of exposure profiles; L3 path parity once gate V-4 closes. Fix the oracle's first-segment curve while in that file | [I-50](known-issues.md#i-50), [I-34](known-issues.md#i-34) | L |
| 2.3 | Sensitivities against ORE's sensitivity analytic on the shared portfolio; the AD Greeks against the bump Greeks on the same sloped portfolio | [I-51](known-issues.md#i-51), [I-78](known-issues.md#i-78) | M |
| 2.4 | Reproduce ORE's two per-path recalibration details, measured against 2.2's cube; confirm an American's basket on a path against ORE's (decision A-7); warn, as ORE's `LgmBuilder` does, when a path's recalibration misses its basket | [I-49](known-issues.md#i-49), [I-73](known-issues.md#i-73) | M |
| 2.5 | `ShiftHorizon` as a setting; parity at 0.5; then 0.5 as the default | [I-32](known-issues.md#i-32) | M |
| 2.6 | *Parallel.* Swaption vol strike axis, read at each option's and helper's strike | [I-54](known-issues.md#i-54) | M |
| 2.7 | *Parallel, can start now.* Store classes whose level swamps their spread relative to a level (the cube to its t=0 value, the curves to their path-independent part, or a block offset); a measurement harness, rerunnable on any kernel change, for the storage formats per class, product and path count, with 1.7's paired sample and its estimator's coverage on the pipeline | [I-75](known-issues.md#i-75) | M |
| 2.8 | Kernels in difference form with explicit accumulators, one family at a time (simulation scan, scenario curves, legs, Europeans, Bermudan rollback and recalibration, exposure), one implementation for every precision (A-16); compute below float32 enabled. Where a family can be a matrix product (leg pricing, the Bermudan rollback), it takes that form in the same rewrite, so native FP8 (3.4) needs no second one. ORE parity at existing tolerances first, then the float64 snapshot re-baselined once | [F-07](features.md#f-07) (compute) | L |
| 2.9 | The evidence table against the acceptance standard (A-11: Basel III's P&L attribution test and the Basel plan's P6.2 rule; P6.2 labelled as engineering for figures Basel does not cover), from 2.7's harness on the kernels of 2.4, 2.5 and 2.8; a warning on any result whose combination has no passing row | [I-55](known-issues.md#i-55) (warnings) | S |

Order within the stage: 2.2 before 2.4 (2.4 needs 2.2's cube); 2.5 changes the default
Bermudan/American numbers, so it finishes before stage 3. 2.6 changes no number on any market
given to the engine so far (all are ATM-only) and no consumer supplies a smile, so it is
parallel and gates nothing. 2.7 runs beside the rest of the stage: it changes no float64
number. 2.8 comes after 2.4 and 2.5, which change the same kernels, so none is rewritten
twice, and before stage 3, because it moves float64 numbers at rounding level. 2.9 comes
last: an evidence table measured before 2.4, 2.5 and 2.8 would describe kernels about to
change. Parity methodology: [details/ore-parity-validation.md](details/ore-parity-validation.md);
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
| 3.1 | Profile a portfolio job, then remove the remaining dominant costs (an American's per-path recalibration arithmetic, first-call compiles); recompiles per trade, date, bump and call went on 2026-10-02, per worker process with 1.8, and per worker restart on 2026-10-04 (the worker keeps JAX's persistent compilation cache) | [I-53](known-issues.md#i-53) | M |
| 3.2 | *Parallel, can start now* (it moves no kernel's arithmetic). One worker per host on a Cloud TPU pod slice, process 0 claiming each job and handing it to the other hosts, since under SPMD every host runs the same job ([details/precision.md §11.3](details/precision.md#113-multi-device-and-multi-host)); wall time against device count on TPU and H100. The one-host split of the scenario axis is done (2026-10-04) | [I-61](known-issues.md#i-61) | M |
| 3.4 | After 3.2. The research result: on several devices of Ironwood and H100, wall time per figure at equal accuracy, many low-precision paths against fewer float64 paths, read from the evidence table's path ceilings; the storage formats, then 2.8's matrix-product kernels on native FP8 | [F-07](features.md#f-07) (speed) | M |

<a id="stage-4--api-robustness"></a>
## Stage 4 — API robustness

| Step | Work | Closes | Size |
|---|---|---|---|
| 4.1 | One route (the old names as aliases; the request is already one shape since 1.3), a market-risk route, the CAM calibration route, and a completeness test failing on any configuration setting without an API field. Every per-trade result row carries its trade id; the cube returns as a chunked artifact reference | [I-56](known-issues.md#i-56), [I-10](known-issues.md#i-10) (results), [I-09](known-issues.md#i-09) | L |
| 4.2 | EOD accepted-attempt record, boot sweep, `interrupted` state (portfolio jobs have had all three since 1.8); a retention policy for the job queue; a job status that says when the engine worker cannot start | [I-08](known-issues.md#i-08), [I-76](known-issues.md#i-76), [I-77](known-issues.md#i-77) | M |
| 4.3 | Chase TraderX's answers; apply them (a widened allowlist, a schema statement) | [I-23](known-issues.md#i-23), [I-60](known-issues.md#i-60) | S |

<a id="stage-5--tests-and-tooling"></a>
## Stage 5 — Tests and tooling

| Step | Work | Closes | Size |
|---|---|---|---|
| 5.1 | Characterize the full-suite aborts and lost worker processes: the fast tier's CI deaths were memory and are fixed (2026-10-05: the grid-convergence test's grid, compiled programs dropped per module); show the full suite on CI's 16 GB runner (`-n 4` peaked at 24.2 GB with page cache before those fixes), then repeated full runs against a known-bad baseline | [I-27](known-issues.md#i-27) | M |
| 5.2 | Ruff in `pyproject.toml` and CI, then a type checker on `engine/` | [I-66](known-issues.md#i-66) | S |
| 5.3 | Shared test helpers in `tests/support/`; public-entry tests where stage 1 made private-symbol tests obsolete | [I-67](known-issues.md#i-67) | S |

5.1's runs can be made any time a full run is being made anyway; only the conclusion waits
for repeated runs.

<a id="stage-6--features"></a>
## Stage 6 — Features

| Step | Work | Feature | Can start after |
|---|---|---|---|
| 6.1 | *Parallel, start now.* Basel P0 (decisions, pinned text, profile, traceability) and P2 data acquisition, which is calendar time | [F-05](features.md#f-05) | — |
| 6.2 | Engine options: ORE's `AnalyticLgm` European engine, settlement methods, FD solver (the AD Greeks method and the market-risk engine by configuration are done) | [F-01](features.md#f-01) | 2.8; the FD solver also 2.5 |
| 6.3 | FP4: storage with a variance correction through the block scales, compute on TPU 8t/8i; multilevel estimation of quantiles (PFE, VaR, ES) | [F-07](features.md#f-07) (FP4) | 2.8, 3.4 |
| 6.4 | FX and equity trades on the market path; FX/EQ calibration | [F-04](features.md#f-04) | 2.8 |
| 6.5 | SABR volatility | [F-02](features.md#f-02) | 2.6, 2.8 |
| 6.6 | Basel P1 (FRTB-SA) onward; CVA/DVA | [F-05](features.md#f-05), [F-06](features.md#f-06) | 2.3 (USD swaps also I-05); 2.2 |
| 6.7 | AMC engine | [F-03](features.md#f-03) | 2.2, 2.8 |

A step that adds a path-pricing kernel (6.2, 6.4, 6.5, 6.7) starts after 2.8 and writes it in
2.8's form, one implementation for every precision (A-16); written earlier, it would be
rewritten.

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
