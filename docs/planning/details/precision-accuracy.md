# Existing-format accuracy and uncertainty (I-98, I-111, I-113)

## Plain-language summary

Low precision can introduce systematic pricing errors. This work adds randomized operand
rounding, enforces valid option values, and estimates sampling uncertainty from independent
runs within the same path budget. The remaining problem is rare events: many runs observe
none, making ordinary confidence intervals too optimistic. Those intervals remain explicitly
unvalidated; the numerical acceptance rules are unchanged.

## Plan

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

**I-111 is partial, not closed.** Independent-replicate variance estimates remove the
IID-path assumption. They do not establish nominal Student t coverage for rare events.
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

Remaining acceptance work for I-111: an uncertainty/interval method with calibrated
coverage on rare events, or a principled rule that reports an interval as unavailable
when its data cannot support one. Preserve raw variance estimates for research, validate
on independent outer seeds, and retain the predeclared RMS/coverage criteria. This
remains focus work before hardware timing. The implementation must not be described as
completing that acceptance gate merely because the dependence assumption is corrected.

The review also found an existing evidence-scope gap (I-114): lookup keys distinguish
formats, rounding, pairing and replicate counts, but omit pricing-engine and rollback-grid
settings. Measured Bachelier/grid rows must not establish acceptance for unmeasured settings.
That is tracked before hardware timing. Equal-accuracy timing must also measure the actual
RQMC error scaling and covariance, rather than assume an IID variance/path-count formula.

## Numerical compatibility

The before/after shared-portfolio snapshots cover all five product families and Greeks at
float64, float32 and mixed paired storage precision. Matched valuations, exposures,
Greeks and exercise flags are identical. The new snapshot has 678 numeric outputs versus 675:
three additional replicate-count fields. The only changed existing arrays are 72 error
fields deliberately becoming NaN for a single scramble. Greek comparisons use trade NPV
as their scale; ORE parity tolerances are unchanged. Changing the replicate count changes
the draws by design, so compare precision policies on matched replicate draws.

## Campaign refresh

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
