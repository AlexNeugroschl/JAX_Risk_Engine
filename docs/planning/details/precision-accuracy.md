# Existing-format accuracy and uncertainty (I-98, I-111, I-113)

## Plain-language summary

Low precision can introduce systematic pricing errors. This work adds randomized operand
rounding, enforces valid option values, and estimates sampling uncertainty from independent
runs within the same path budget. Rare events can make ordinary confidence intervals
too optimistic. Nominal intervals now
report unavailable under a conservative rule; raw variance estimates remain available for
research. The numerical acceptance rules and their failures are unchanged.

## Plan

Completion plan, 2026-10-16 (I-111):

1. Add failing regressions for a zero-standard-error miss, rare-event interval
   availability, Python/HTTP metadata and the misleading low-level ES name.
2. Adopt a conservative availability rule: no nominal confidence interval without a
   validated tail bound or interval method for the estimator. The current pipeline
   has neither, so all figures and entries report intervals unavailable. Preserve
   every point estimate and raw standard error; no empirical event-count threshold,
   variance inflation or relaxed acceptance criterion.
3. Make that rule machine-readable in precision reports and market-risk uncertainty
   metadata, using one shared immutable description. Keep it outside JAX kernels.
   Rename the conditional ES diagnostic with a deprecated compatibility entry.
4. Audit diagnostic Student t coverage per date. Include nonzero zero-width misses,
   distinguish numerical equality from proof of a constant, and record unavailable
   intervals separately from diagnostic successes. Keep the 0.75–1.33 RMS and >=0.90
   coverage criteria unchanged, including their failures.
5. Validate rare-event examples on independent outer seeds, rerun the replicated
   campaign, verify warm compile reuse, numerical compatibility and ORE parity, run
   the full suite, and update planning and public documentation. Numerical precision
   evidence remains based on matched float64 errors, independently of interval status.

Implementation plan, 2026-10-15. The scope is the current formats through FP8;
FP4 and multilevel quantile correction remain independent later experiments.

1. Reproduce the operand rounding bias, negative long-option values, and uncertainty
   defects with regression tests before changing the kernels. Capture fixed-draw
   float64/float32 results for comparison.
2. Add explicit stochastic rounding of per-path product operands, keeping the shared
   operand's two-term representation and nearest rounding as a research baseline.
   Random keys are traced data, with independent streams for stages, trades, dates,
   products and induction steps; no seed or trade identity enters a compile key.
3. Project a long option's reconstructed value onto zero before the wrapper applies
   LongShort or delivers an exercised physical underlying. Keep the evidence rule.
4. Generate equal-sized independent scrambled Sobol replicates within a run. Pair a
   prefix within **each** replicate at float64. Estimate mean uncertainty across
   replicate estimates, including the correction and its covariance. Retain pooled
   quantiles and use replicate deletion for ES uncertainty. Historical uncertainty
   uses moving blocks of consecutive horizon windows, labelled as conditional bootstrap
   uncertainty. A single replicate cannot establish an RQMC standard error.
5. Wire settings and metadata through Python and HTTP. Validate before work, reuse
   module-level JIT kernels, retain column-wise exposure reductions, and distinguish
   statistical error from measured precision bias.
6. Run regression, coverage, compilation, ORE parity and full-suite checks. Rerun
   affected precision campaigns and regenerate the evidence table; never reuse rows
   for changed rounding or sampling. Record measurements, limitations and follow-up
   work here and in the issue register/roadmap.

Acceptance: existing ORE tolerances; fixed-draw float64 relative change <= 1e-10
and float32 <= 1e-4 of each array's scale (Greeks scaled by trade NPV); no warm
compilation on new seeds or equal-shaped trades; A-11's unchanged evidence rules;
coverage assessed across independent outer seeds with Student t intervals for
replicate means. New replicate sampling is compared on matched draws, not against
the old single-scramble sample as if the change were arithmetic drift.

References: [SciPy's Sobol documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html)
describes independent scrambled replicates for error estimation;
[JAX's PRNG documentation](https://docs.jax.dev/en/latest/jax.random.html)
describes explicit keys and independent derived streams.

## Implementation and measured limits, 2026-10-16

The rounding, domain bounds, independent scrambles, per-scramble pairing, historical
moving-block bootstrap and Python/HTTP controls are implemented. Nearest remains the
default baseline. Arithmetic is a pytree with static formats/rounding and traced keys
and sample ids; streams are independent per stage, trade, date, product and rollback
step. Global sample ids preserve rounding under prefixes, batches, blocking and sharding.
Calendar-day streams round the Actual365Fixed day count before converting to an integer:
truncation aliased days 2 and 3, reproduced by a failing regression before the fix.
The leg and Jamshidian products share this date-stream helper; Jamshidian's missing
date derivation also had a failing regression before the fix.
Delete-group and bootstrap indices use explicit int32 operations; strict-promotion tests
exposed and now guard the implicit Boolean/integer conversion in replicate deletion.
The option bound precedes LongShort and physical delivery; market-risk P&L is bounded
before storage and after decoding. A regression found a stressed bfloat16 Bermudan
worth -40.1640625 before this bound and passes after it. The initial eight rounding,
replicate and wrapper regressions failed on the original implementation.

**The initial interval experiment failed calibration.** Independent-replicate variance
estimates remove the IID-path assumption. They do not establish nominal Student t coverage
for rare events. The completion below uses the availability-rule alternative.
The eight-scramble storage experiment at 1,024, 4,096 and 16,384 total paths, with 32
outer seeds, found substantial undercoverage on rare netting-set ENE and correction
events. Increasing to 16 and 32 scrambles on selected 4,096-path cases did not solve
it. For example, at 4,096 paths the netting set's ENE on one date averages only 8.32
while its spread across outer seeds is 16.98; many inner replicates see no event.
Do not inflate the standard error by a fitted factor or redefine the acceptance bar.
Reported errors are labelled variance estimates, with an explicit warning against
interpreting them as validated 95% bounds. One scramble reports NaN/HTTP null.

The coverage harness now compares a correction with an expectation estimated on
**independent outer seeds**, including that reference's uncertainty. Comparing it with
the full-draw correction and applying an IID finite-population adjustment would be
invalid for RQMC. The corrected mean is compared with independent float64 expectations.
The raw precision-error evidence still uses matched full draws, independently of any SE.

The final eight-scramble record includes 2,160 figure rows and 432 coverage rows, with
zero warm recompilations. At 4,096 paths, simulation bfloat16 with stochastic storage has
netting-set ENE total coverage 39/64 (60.9%) and standardized-error RMS 1.89. At 16,384
paths, market float16 with nearest storage has European ENE correction coverage 15/64
(23.4%) and RMS 4.12; the largest correction RMS is 4.90 (market bfloat16, nearest).
Pooled coverage can conceal these failures: acceptance must inspect individual figures
and dates, including zero-error estimates that miss their independent reference.

**I-111 completion: conservative interval availability.** The shared immutable
`IntervalAvailability` metadata reports `status="unavailable"` and
`reason="no_validated_tail_bound"`, with null confidence level and endpoints, for all
figures and entries. Python and HTTP expose it in `precision.interval`, market risk
also in `uncertainty.interval`. Nonzero empirical variance, many observed events,
additional scrambles and a passing precision row cannot establish the missing tail
assumption. This deliberately declines intervals even on apparently benign entries;
it does not claim calibrated finite intervals for rare events. The raw standard errors,
point estimates and paired corrections are unchanged. See the constructive rare-event
argument and public contract in [Sampling uncertainty](../../reference/sampling-uncertainty.md).

The coverage audit now retains zero-width misses beyond numerical roundoff as infinite
standardized errors, records zero-width numerical equalities separately without
certifying constants, and reports each date independently. It counts nominal interval
unavailability separately; unavailable/uninformative diagnostics never pass the
unchanged >=0.90 coverage and 0.75–1.33 RMS criteria. A diagnostic pass does not make
a nominal interval available. Fewer than three outer seeds
reports unavailable coverage diagnostics without invalid sample-variance warnings.
The renamed `conditional_tail_standard_error` makes the low-level ES diagnostic's
meaning explicit; its old name remains with a deprecation warning and identical numbers.
Future supported confidence intervals are an independent capability (F-10), not an
implicit promise that more scrambles fix the failed experiment.

The review also found an evidence-scope gap (I-114, closed 2026-10-18): lookup keys
distinguished formats, rounding, pairing and replicate counts, but not the pricing engine,
rollback grid or market risk's per-scenario recalibration. Each row now holds under its
campaign's pricing settings ([precision.md §10.4](precision.md#104-as-built-2026-10-11)), so
Bachelier or coarse-grid rows establish nothing for unmeasured settings. Equal-accuracy timing
must also measure the actual RQMC error scaling and covariance, rather than assume an IID
variance/path-count formula.

## Numerical compatibility

The I-111 availability-rule completion changes **none of the 678 numeric arrays** in
the before/after snapshot: float64, float32 and mixed paired storage, all five product
families, exposures, Greeks and exercise flags are identical, including NaN positions.
The new interval fields are host metadata and introduce no JAX program or compile key.
The following paragraph describes the earlier independent-scramble implementation.

The before/after shared-portfolio snapshots cover all five product families and Greeks at
float64, float32 and mixed paired storage precision. Matched valuations, exposures,
Greeks and exercise flags are identical. The new snapshot has 678 numeric outputs versus 675:
three additional replicate-count fields. The only changed existing arrays are 72 error
fields deliberately becoming NaN for a single scramble. Greek comparisons use trade NPV
as their scale; ORE parity tolerances are unchanged. Changing the replicate count changes
the draws by design, so compare precision policies on matched replicate draws.

## Campaign refresh

The I-111 completion reran `replicated_storage` on CPU: 32 independent outer seeds,
eight scrambles per run, 1,024/4,096/16,384 total paths, every existing case. All **2,160
figure measurements are identical** to the previous record, with zero warm recompilations.
The **2,592 per-date coverage rows** record 82,944 nominal interval unavailability
decisions, separately from diagnostic Student t coverage. Among entries with diagnostic
checks, correction/total criteria fail on 119/77 entries at 1,024 paths, 134/193 at
4,096, and 134/98 at 16,384. These failures remain failures. For example, stochastic
bfloat16 simulation storage's netting-set ENE at 4,096 paths on 2029-07-30 covers only
12/32 independent outer references, RMS 2.29. No observed zero-width miss beyond
roundoff occurred in this campaign; explicit regressions exercise that failure mode.
Numerical equalities never count as interval-availability evidence.

The refreshed record and generated evidence table retain the same numerical precision
verdicts. The other campaigns need no numerical rerun for metadata or audit changes;
their historical interval diagnostics remain historical. The following paragraphs
describe the preceding rounding/bounds refresh. Concurrent validation runs again make
these elapsed times unsuitable for device throughput claims.

Eight campaigns were added or refreshed: `options`, `compute_options`, `market_risk`,
`compute_market_risk`, `rounding`, `rounding_options`, `rounding_market_risk` and
`replicated_storage`: 11,680 figure rows, zero warm recompilations. The ten records of
record contain 34,000 figure rows in total. The nearest/R=1 `linear` and `compute`
numerical paths are unchanged. Their older uncertainty diagnostics are historical;
the precision verdicts use independent outer-seed errors, not those diagnostics.

The final date-key refinement leaves every recorded campaign stream unchanged: the
six measured dates already recover exact day ids [92, 184, 365, 731, 1096, 1826];
market risk uses day zero. No recorded campaign uses Jamshidian. Explicit int32
bootstrap/deletion indices select the same observations. These refinements fix
adjacent-day and strict-promotion edge cases without relabelling old measurements.
Concurrent runs and suspension affected elapsed times; the refresh establishes
accuracy and compile reuse, not a device speed ranking.

## Tested by

- `tests/test_uncertainty_availability.py`: rare-event runs with 1/8/32 independent
  replicates, empirical zero versus a known nonzero expectation, infinite zero-width
  misses, unchanged diagnostic criteria and the deprecated ES entry.
- `tests/test_precision_harness.py`: per-date diagnostics, interval-unavailability counts,
  numerical equalities, two-seed reference unavailability and record round-trips.
- `tests/test_precision_accuracy.py`: operand bias, explicit keys, prefix/batch/block
  invariance, option bounds before delivery and after storage, replicate covariance,
  analytic mean/ES coverage, historical dependence, wire controls and warm compile reuse.
- `tests/test_precision_report.py`, `tests/test_market_risk.py` and
  `tests/test_api_completeness.py`: estimator integration, API metadata and compatibility.
- `tests/test_sharding.py`: four virtual CPU devices, including stochastic compute.
- The registered precision campaigns and generated evidence table: matched draws,
  independent outer seeds, unchanged A-11 acceptance and zero warm recompilations.

The full-suite and device verification record is in
[known-issues.md](../known-issues.md#verification-status).
