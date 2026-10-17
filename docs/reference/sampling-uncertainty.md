# Sampling uncertainty

The engine reports point estimates and empirical standard errors. It reports nominal
sampling confidence intervals as **unavailable**, for every figure and entry, including
figures whose empirical error is zero. This is the conservative availability rule for
I-111; it does not claim that the underlying variance is zero or that a confidence
interval has passed calibration.

In Python, inspect `result.precision.interval`. Over HTTP, the same object is in the
result's `precision.interval`. Market risk also includes it in `result.uncertainty["interval"]`:

```json
{"status": "unavailable", "reason": "no_validated_tail_bound",
 "confidence_level": null, "lower": null, "upper": null}
```

This applies to all figures, dates, corrected and uncorrected means, corrections,
quantiles, and historical and Monte Carlo results. It also applies to unpaired runs,
float64 and runs with many independent scrambles. There is no user setting that
certifies interval availability. A precision evidence row certifies neither an
interval nor the absence of sampling or model error.

## Why the rule is conservative

Independent scrambled Sobol replicates allow a variance estimate without treating
dependent paths as IID. They do not alone guarantee a Student t confidence interval.
The replicated-storage experiment finds rare ENE and correction events for which
those diagnostic intervals fail the unchanged coverage and standardized-error tests.

An empirical event-count or variance cutoff cannot certify that a tail is absent.
For example, let a replicate contribution be zero with probability `1-p`, and `1/p`
with probability `p`. Its expectation is one and its variance is finite. With
`p = 0.001`, even 32 independent replicates all observe zero about 96.8% of the time:
both the estimate and its empirical standard error are zero. Adding this rare
contribution to an ordinary distribution also defeats rules that merely require
nonzero variance or many observed events. This argument concerns inference from
samples alone; model-specific mathematical tail bounds could support a different rule.

The current pipeline supplies no validated bound on the unseen tail or model-specific
interval method for these estimators. It therefore declines nominal intervals for
all entries, rather than selecting apparently benign entries using the same sample.
This preserves research diagnostics without presenting them as calibrated bounds.
Student t mean intervals require distributional assumptions; see
[SciPy's one-sample t-test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html).

Enabling a future interval method requires stated assumptions or a proven bound for
the complete estimator, and validation on independent outer seeds, including rare
events and zero-error observations. A normal-example test or a passing precision
evidence row is insufficient. This implementation does not fit an inflation factor,
change acceptance thresholds, clip payoffs for statistical convenience, or add a
quantile correction.

## What remains available

- Paired EPE/ENE means retain their two-level correction, raw mean, correction,
  maximum paired difference and all three standard-error diagnostics. With `R > 1`
  independent scrambles, errors use the sample spread of the complete replicate
  estimates divided by `sqrt(R)`, including the correction covariance. With one
  scramble they are NaN in Python and null over HTTP.
- PFE, VaR and ES retain the existing pooled point estimators and their paired
  float64 precision measurements. Market-risk ES variance diagnostics delete whole
  scrambles and recompute the threshold and tail, or conditionally resample historical
  moving blocks. Historical diagnostics depend on stationarity and block length;
  they do not estimate forecast or model error.
- The low-level `conditional_tail_standard_error(pnl, percentile)` measures only
  tail spread divided by the square root of its observation count, with the fitted
  threshold held fixed. `expected_shortfall_standard_error` is a deprecated alias
  that warns about this distinction. The existing `compute_risk_metrics` key
  `ES_<q>_standardError` retains this conditional diagnostic for compatibility;
  `run_market_risk` replaces it with the complete-estimator variance diagnostic.
  Tail observation count is not effective independent sample size.

The precision harness audits diagnostic Student t intervals separately, per date.
Nonzero zero-width misses beyond numerical roundoff count as infinite standardized
errors. Numerical equality is recorded separately and never certifies a constant.
Unavailable nominal intervals are counted explicitly, never as successful coverage.
A passing diagnostic does not change their availability. The
predeclared diagnostic criteria remain coverage at least 0.90 and standardized-error
RMS in 0.75–1.33; their failures remain research results. At least three outer seeds
are needed to estimate the independent reference's uncertainty. Historical records
without entry labels pooled dates and omitted zero-width misses.

See the [precision measurements](../../compliance/precision-measurements/README.md)
and [implementation plan and measurements](../planning/details/precision-accuracy.md).
