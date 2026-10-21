# Native FP8 and numerical reliability: measured results

Research started 2026-10-20; overnight verification completed 2026-10-21 UTC.
Engine source digest: `351a34c6d655168e`. These are empirical results on the exact
archived inputs. FP64 remains the default. A native matrix product and a financially
acceptable reduced-precision result are separate checks.

The implementation adds FP16 comparisons, separate compute/storage policies, explicit
independent seed identities, independent FP64 reference diagnostics, bounded-memory
measurement, resumable isolated campaigns, operation/workload lowering and traces, and
bounded fresh-process full-suite validation. It fixes zero-volatility European liveness,
delayed bond redemption/payment and small-reversion LGM H(t), plus Windows worker ownership,
Unicode logging and ORE failure cleanup. It preserves A-10/A-11 and existing ORE tolerances.
The [execution plan](../../../docs/planning/details/native-fp8-reliability.md) tracks the
remaining target hardware, performance and statistical work.

## What was measured

Counts below are per independent seed, never totals summed over seeds. All accuracy
studies use matched FP64 scenarios to isolate precision error. That does not establish
sampling convergence, model suitability, nominal interval coverage or economic validity.

| Study | Counts / independent seeds | Scope | Result |
|---|---|---|---|
| Eleven campaigns of record | Original 1K/4K/16K exposure and 2K/8K market risk; original 16 seeds, 32 for replicated storage | All existing storage, compute, rounding and recalibrated-option cases | All refreshed; four ceilings changed; zero warm recompiles |
| Large single-swap market risk | 65,536 / 262,144 / 1,048,576 / 4,194,304; 32 seeds, 1–32 | Pricing compute FP32, FP16 and two-term FP8 E4M3; FP32 storage | Per-figure limits below; FP8 passes none |
| Larger exposure | 16,384 / 65,536 / 262,144; 16 seeds, 1–16 | Both LGM and Hull-White; FP16/E4M3 two-term and single-pass controls | Passing figure groups decrease with count; no universal policy |
| Stress holdout | 1,024; 16 disjoint seeds, 1001–1016 | Fixtures 44/45/51 (steep/humped/negative rates), both models, correlated USD/EUR and FX | Many precision failures retained |
| Flat-market holdout | 1,024; four disjoint seeds, 1001–1004 | Fixture 40, Hull-White, default grid, correlated USD/EUR and FX; FP16 controls | 53/80 figure groups pass |
| Option-grid holdout | 128; 16 disjoint seeds, 1001–1016 | Negative-rate fixture 51, LGM, default (30,5) and refined (48,6) grids; FP16/E4M3 controls | 32/120 groups pass on each grid; small count is diagnostic |
| Independent ORE holdout | 32 ORE paths per check | Fixtures 40/44/45/51 × two models × USD/EUR component pricing | 68 today NPVs and 68 trade cubes pass existing parity tolerances |
| Independent ORE option grids | 32 ORE paths, three option trades per grid | Fixture 51, default/refined grids | Today NPVs and cubes pass; material FP64 grid differences remain |
| Independent-reference FP16 comparison | Candidate 128/512, seeds 1/2/3; CPU FP64 reference 4K/16K, seeds 1001–1008 | Single swap and five-trade portfolio; compute-only/storage-only | Eight FP16 engineering checks fail; four FP64 matched controls pass; reference cannot resolve two equal-accuracy rows |

The USD/EUR ORE checks price each currency's component separately; the existing USD-only
oracle adapter relabels EUR inputs to USD. They independently check native-currency pricing,
**not the joint multicurrency simulation**. Stress holdouts are precision comparisons,
not independent full-system multicurrency ORE qualification.

## Larger counts expose precision floors

The unchanged A-11 random-entry gate is precision RMSE at most 1% of the matched FP64
across-seed spread. A ceiling requires every smaller measured count to pass. These limits
apply only to this single payer swap, fixed market, model, seeds, pricing and sampling design.

| Pricing compute (FP32 storage) | Mean P&L | VaR 97.5% | VaR 99% | ES 97.5% | ES 99% |
|---|---:|---:|---:|---:|---:|
| FP32 | 1,048,576 | 4,194,304 | 4,194,304 | 4,194,304 | 4,194,304 |
| FP16, two-term | No passing count | No passing count | 65,536 | 4,194,304 | 4,194,304 |
| FP8 E4M3, two-term | No passing count | No passing count | No passing count | No passing count | No passing count |

At 4,194,304 scenarios (USD; ratio = precision RMSE / FP64 seed spread):

| Compute | Mean P&L RMSE / ratio | VaR 99% RMSE / ratio | ES 99% RMSE / ratio |
|---|---:|---:|---:|
| FP32 | 0.000328 / 0.0275 | 0.0594 / 0.000307 | 0.0526 / 0.000482 |
| FP16 | 0.00873 / 0.731 | 5.33 / 0.0276 | 0.168 / 0.00154 |
| FP8 E4M3 | 2.13 / 178 | 312 / 1.62 | 338 / 3.10 |

Even FP32's mean P&L misses this relative precision budget at 4M, despite a sub-cent absolute
error. FP16 can pass ES and fail VaR and the mean in the same run. More scenarios reduce
FP64 sampling spread and reveal persistent rounding floors. Keep FP64 for production unless
each required figure has an applicable passing policy; these experimental rows are not
automatically installed into runtime acceptance evidence.

The larger-exposure passing-group counts, out of 160 at each count, are LGM **21/20/18**
and Hull-White **19/17/17**. Stress holdout counts out of 160 are fixture 44 **84/85**,
fixture 45 **61/57**, and fixture 51 **60/55** (LGM/Hull-White). Raw failures and smaller
counts remain in the archives; these totals are diagnostics, not policy acceptance rates.

## Native execution evidence

On the available RTX 5060, cuBLASLt passes **504 unique operation checks** across the
two-term formats and single-pass controls: seven shapes, both shared directions,
nearest/stochastic rounding, and wide-range/zero/cancellation inputs. **264 operation rows**
contain observed FP8 cuBLAS calls; GEMV/fusion fallbacks are retained separately. Optimized
HLO records operand types and FP32 accumulation. Dispatch labels alone are not hardware proof.

Twelve separate warm workload captures cover FP16 and both FP8 formats, two-term and
single-pass, on both comparison workloads at 512 scenarios. Each has finite reported
figures, GPU kernel events and **zero cold or warm compilations inside the capture** after
three warm-up seeds. Actual portfolio traces contain `nvjet_sm120_qqsss_mma` and
`nvjet_sm120_qrsss_mma` kernels correlated with FP8 cuBLAS HLO. These establish execution
on this device/runtime, not accuracy or speed on H100.

The worker's `--gpu-gemm cublas` selects the correct process-level route before JAX starts.
Comparison workers choose it automatically for GPU FP8; inherited FP64 defaults remain
available. Emulation stays available for unsupported/unqualified routes and numerical
cross-checking. **H100 and Ironwood are unmeasured**; no H100 host was available. The pinned
TPU extra is an installation candidate, not a hardware qualification.

One earlier 128-scenario execution probe produced undefined ES 99% for combined FP8 E5M2
(single swap, seed 6): all raw P&Ls were finite, but the two worst tied at −524,288 and
the existing strict tail beyond VaR was empty. The engine correctly returns unavailable
ES, matching its existing ORE contract. The raw snapshot is retained. The 512-scenario warm
captures have finite figures; neither count establishes tail convergence.

## Independent references, discretization and timing

The independent-reference comparison retains empirical SE and count-to-count changes.
Of 64 equal-accuracy rows, 62 meet the declared reference-resolution rule and two do not
(single-swap VaR 97.5%, target 512, both policies). Those two speedups are withheld.
An empirical SE below 10% of the target does not establish nominal confidence coverage.
Supported confidence intervals remain unavailable without validated tail assumptions.

The option-grid ORE checks agree on each grid, with bit-identical numeraires. Changing the
FP64 grid changes the American's cube by up to **54.43899427 USD**; the other two options
change by up to 0.31161094 and 5.16771188 USD. This isolates a discretization effect but
does not establish a converged grid. Precision accuracy cannot remove that effect.

**All timings in this collection are diagnostic.** Runs overlapped and the host was suspended
for several hours. No throughput, equal-accuracy speedup, memory-capacity or H100 performance
claim is made from these times. Idle target-device timing remains required.

## Refreshed acceptance evidence and validation

All eleven original campaigns retain their original seeds, counts, products, settings and
policies. Nine use CPU; `options` and `rounding_options` use GPU/cuBLAS. All have zero warm
recompiles. Only four ceilings move:

| Campaign / figure | Cases | Previous ceiling | Refreshed ceiling |
|---|---|---:|---:|
| `compute_market_risk` / Bermudan P&L | FP32, BF16_3x | 8,192 | 2,048 |
| `market_risk_calibrated` / portfolio P&L | FP32, BF16_3x | 2,048 | 8,192 |

The fixed Bermudan's former ratios 0.009046/0.009384 become 0.011706/0.012849 at 8K
(about 0.021/0.023 USD absolute RMSE). The 0.01 threshold is unchanged. The
[generated acceptance evidence](../../precision-evidence.md) uses all reviewed original
records; the larger and held-out research remains separate.

The FP64/FP32 golden comparison checks 460 arrays across all eight product families and
both IR models, including cubes, profiles, today's PV and risk. Maximum relative changes
are 2.0143e-14 (FP64) and 4.9250e-7 (FP32), inside the existing 1e-10 / 1e-4 margins.
The numerical-boundary tests were first run against the old engine: eight failed, one
passed. After the fixes, all nine pass. Two known-bad suite repetitions retain the eight
expected failures and `complete=false`. Worker and ORE lifecycle regressions were also
demonstrated failing before their fixes.

**Complete Windows CPU suite: 4,002 passed, 9 skipped, zero failures/errors.** All 4,011 collected IDs occur exactly once across 503 successful fresh batches, with the same frozen Python digest `9e9866f4ca8867887d031a670a92f559aa90b2db08cf0f945095b5579e61eecc`. The audit requires exact JUnit counts and printed summaries; it excludes interrupted earlier attempts.

After promotion of the eleven reviewed records and regeneration of acceptance evidence,
**127 precision-evidence/report tests pass**; [JUnit](post-refresh.xml.gz) and
[printed summary](post-refresh.log.gz) are retained.

This establishes a complete Windows CPU suite on the archived source. It does not qualify
H100, TPU, repeated 16 GB CI execution, unvalidated statistical intervals or cluster scaling.

## Reproduction and artifacts

Use the pinned checkout environment and an empty output directory for each experiment;
the [benchmark guide](../../../benchmarks/README.md) documents supported CLI entry points.
The archived reproduction scripts are as executed. Scripts using `__file__` place output
beside themselves: copy the script into a new ignored scratch directory in the checkout,
then run from the repository root with `python -c "import runpy; runpy.run_path('SCRATCH/large_single.py', run_name='__main__')"`.
Never overwrite the published collection. The warm-trace script accepts an explicit output:

```bash
python compliance/precision-research/2026-10-20/reproduction/warm_native_trace.py \
  --out .profile-out-h100-warm --platform gpu --expected-device H100 --paths 512
python -m tests.support.precision_devices --platform gpu --expected-device H100 \
  --gemm cublas --require-native-fp8 \
  --formats float16 float16_1x float8_e4m3fn float8_e4m3fn_1x float8_e5m2 float8_e5m2_1x \
  --engine-workloads comparison_single comparison_portfolio --out .profile-out-h100-native
```

Repeat the operation check with `--gemm default` in a separate directory; retain any
emulation/failure result. Hardware-use claims require numerical checks, optimized lowering
and device traces together. Verify engine ORE/golden parity on H100 before timing it.

| Artifact | Contents |
|---|---|
| [summary.json.gz](summary.json.gz) | Exact largest-count errors, ceilings, canonical review, final suite metadata and scope |
| [large-single](large-single/candidate-evidence.md) | 4M/32-seed measurements, workload/environment and all experimental per-figure limits |
| [large-exposure](large-exposure/qualification.json.gz), [stress-holdout](stress-holdout/qualification.json.gz), [option-grid-holdout](option-grid-holdout/qualification.json.gz), [holdout](holdout/qualification.json.gz) | Full completed point registries, manifests, compressed measurements and candidate evidence |
| [heldout-oracle](heldout-oracle/results.json.gz), [option-grid-oracle](option-grid-oracle/results.json.gz) | Independent parity inputs/results, maxima and option cube snapshots |
| [FP16 comparison](fp16-reference-comparison.json.gz) | Matched errors, independent reference diagnostics, timings explicitly unsuitable for performance claims |
| [native two-term](local-native/qualification.json.gz), [native single-pass](local-native-singlepass-final/qualification.json.gz) | All 504 operation rows, environments, actual workload results and lowering paths |
| [two-term lowering](local-native-lowering.zip), [single-pass lowering](local-native-singlepass-final-lowering.zip) | Optimized operation/workload HLO |
| [warm captures](warm-traces.json.gz), [traces and HLO](warm-traces-and-lowering.zip) | Twelve GPU-timeline extracts with source hashes, one complete portfolio E4M3 trace, and corresponding optimized lowering |
| [undefined-tail snapshot](undefined-tail-128-seed-6.npz) | Finite P&Ls with empty strict ES tail |
| [golden check](golden-comparison.json.gz), [canonical review](record-refresh-review.json.gz) | Before/after numerical margins and every acceptance-ceiling change |
| [suite audit](full-suite-audit.json.gz), [complete batch evidence](full-suite-evidence.zip) | Exact collected node union, source digest, successful exits, printed summaries and JUnit counts |
| [known-bad repetition 1](known-bad-suite-7.zip), [repetition 2](known-bad-suite-8.zip), `negative-controls/` | Expected failures and red-first lifecycle evidence |
| [source.zip](source.zip), [source-final.zip](source-final.zip), [source-verification.zip](source-verification.zip) | Numerical-study source, subsequent worker fixes, final verified source (the final snapshot supersedes earlier helpers) |
| [reproduction scripts](reproduction-scripts.zip) | Exact experiment-specific research scripts; standard runners are in the source snapshots |

The trace archive omits repetitive host timelines from eleven captures; its index records
original trace hashes and retained event counts, and one complete portfolio E4M3 trace is
included. Full originals remain in the ignored local collection and can be reproduced
with the archived script. GPU timestamps/kernel names and optimized HLO are preserved.

Archives use gzip JSON because this repository ignores plain research JSON/CSV globally.
`SHA256SUMS` identifies each published file. Original failed/interrupted test attempts are
excluded from passing-suite evidence. Experimental accuracy records were not silently
promoted, and no tolerance or failing test was weakened.
