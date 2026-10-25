# Reproducible precision comparison (F-07)

Since 2026-10-22, `benchmarks.comparison` defaults to an end-to-end API comparison with
an ORE CPU baseline. Each timed request executes once, from submission through numerical
result retrieval; the first job is reported separately from repeated fresh submissions.
Both use the production HTTP routes and queue, with a benchmark-only ORE worker. See
[the benchmark guide](../../../benchmarks/README.md#end-to-end-api-comparison-default).
The direct precision study and historical records described below remain available with
`--transport direct`; their methodology and schema have not changed.

## Plain-language summary

One report now shows how precision changes the results, runtime and memory of the same
job on CPU and accelerators. Missing hardware and inaccurate formats remain visible.
The implementation and local smoke matrix below are complete as of 2026-10-20;
target-device research and cluster claims remain open in F-07, I-121, I-97 and I-61.

Implementation plan, 2026-10-19. Reuse the precision campaigns, accuracy estimators and
isolated timing workers; add a small `benchmarks/` command to collect and join evidence.

1. Extend the existing market-risk campaign with named snapshots of scenario PV, trade
   and portfolio P&L, base values and VaR/ES. Select a single swap and the existing small
   portfolio without copying their market, trades or pricing definitions.
2. Extend the timing worker with optional snapshots, exact workload/policy/environment
   metadata, seed identities, compile counts and explicit memory availability. Time host
   preparation through required host outputs; keep compression/report writing outside it.
3. Collect each device/format/count in a fresh process with disk compilation caching off.
   Retain failures, timeouts, missing hardware and warm compilations. Join CPU FP64/FP32,
   accelerator FP64/FP32, BF16 and FP8; distinguish dispatch labels from hardware proof.
4. Compare matched inputs and named outputs, retaining absolute errors near zero. Reuse
   the timing harness's multi-seed/count equal-accuracy calculations, describe the finite
   FP64 reference and report empirical sampling diagnostics without confidence claims.
5. Test comparisons, validation, partial/failing collections and reproducibility joins;
   smoke-run both workloads on available hardware, check warm compilation reuse, run the
   existing tests and ORE parity, and document usage and remaining hardware work.

No pricing kernel or precision policy needs to change. Intentional specializations for
different shapes and formats compile once per worker; subsequent seeds must reuse them.
Hardware qualification and cluster measurements remain I-121, I-97 and I-61. A dispatch
label alone will never qualify native accelerator speed.

## Outcome and remaining work

Implemented as [`benchmarks.comparison`](../../../benchmarks/README.md), composing the
existing campaigns, timing worker and snapshot/reporting mechanics. No engine pricing
kernel or precision policy changed. Each worker selects its requested backend, checks the
observed output placement and records its realized formats. Identical CPU-generated shocks
are checked before comparing outputs; different source digests, settings, versions, seeds,
shapes or names cannot silently become precision error.
Comparison workloads have a separate registry from campaigns of record, so smoke evidence
does not alter the published precision acceptance table. Host CPU model/count/affinity,
accelerator libraries, visible-device and thread-control flags accompany the reproduction
settings; unavailable host information is explicit.

The local smoke matrix uses 128/512 scenarios and seeds 1/2/3, both workloads, FP64/FP32,
BF16 and both FP8 formats on CPU and the RTX 5060 (cuBLAS). **40 measured points, 20 TPU
points explicitly unmeasured, zero failed workers and zero warm compilations.** All 16
FP64/FP32 points pass the engineering smoke tolerance; all 24 BF16/FP8 points miss it.
The GPU FP64 snapshots also pass `1e-8 + 1e-10 * reference_scale` on 192 named checks,
with maximum relative error 3.53e-15. These are local tooling checks, not A-11 acceptance,
native-unit qualification or a speedup result. Raw report numbers are retained in
[the smoke record](../../../compliance/precision-timing/comparison-local-2026-10-20.md).

Findings handled in this implementation:

- Counts below 128 leave ES at 99% without a tail observation under ORE's convention.
  The comparison refuses them before launching a worker; the estimators are unchanged.
- A previous result held alive across iterations inflated the timing worker's memory
  footprint. It is now released before starting the next job.
- Metadata collection scanned reference-submodule working files for every point on WSL.
  It now checks changed gitlinks without traversing their contents; source digests and
  the installed ORE library version still identify the runtime implementation. The new
  regression was demonstrated failing before that fix.
- An inherited default device could misplace a requested accelerator run. Worker device
  selection is explicit, observed output placement is checked and disabled JIT is refused.
- Partial collections, missing/corrupt snapshots, timeouts, overflow and incompatible joins
  retain explicit failures or unmeasured comparisons instead of producing speed claims.
- A full-suite attempt encountered the existing long-run problem (I-27): the expanded ORE
  sweep reached its last case's teardown but cleanup stalled with a large resident footprint.
  The final case passed, including cleanup, in a fresh process. The full attempt is not
  claimed green; successful scoped reruns and the fast tier are recorded in the
  [verification status](../known-issues.md#verification-status).

Remaining work is already scheduled: TPU/H100 install and correctness (I-121), native
execution proof for their actual engine shapes (I-97/I-121), scaling (I-61), and F-07's
larger-count, more-seed equal-accuracy research. CPU memory counters and host RSS are
explicitly unavailable; accelerator peaks are allocator high-water marks across cold/warm
jobs, not whole-machine memory. The fixed-LGM portfolio does not qualify calibrated-scenario
options (I-114). Finite-reference and shared-seed diagnostics remain descriptive, with no
supported confidence interval (F-10). None of those limitations is hidden by a passing smoke
check, and none calls for a new pricing or precision architecture.

## Tested by

- `tests/test_comparison_report.py`: comparison math and edge cases, failure retention,
  reproducibility checks, process/manifest validation and both real CLI workloads.
- `tests/test_market_risk_ore_parity.py::test_comparison_snapshots_preserve_ore_scenario_prices`:
  independently repriced ORE scenario PVs on both comparison workloads, including the
  actual coarse-grid Bermudan settings.
- `tests/test_precision_timing.py`: compile reuse and the existing equal-accuracy estimator.
