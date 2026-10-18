# Precision measurements

The precision harness's measurements of record (I-75;
[docs/planning/details/precision.md §15.4](../../docs/planning/details/precision.md#154-the-storage-harness-2026-10-09)):
how far each storage and compute format, per stage, moves every figure of a run from float64,
on the run's own paths, per product and path count, over independent Sobol seeds. The precision
evidence table ([../precision-evidence.md](../precision-evidence.md), I-55) is generated from
them.

| File | Holds |
|---|---|
| `<campaign>.json.gz` | Every number: per case, figure and path count the precision error's mean (bias), standard error and root mean square, and the figure's statistical error, per date; the paired estimator's coverage; the P&L attribution statistics; recompiles and timings; the market; each product's pricing settings; the commit, whether the tree had local changes, and the digest of the engine's Python sources. `tests.support.precision_harness.Measurements.from_json` reads it |
| `<campaign>.md` | The summary generated from it |

The campaigns are `linear` (the shared portfolio's swaps, Europeans and bond), `options` (its
Bermudans and American) and `market_risk`, which store in each format, and `compute` and
`compute_market_risk` (2026-10-10), which compute in each format below float32
([precision.md §15.5](../../docs/planning/details/precision.md#155-compute-below-float32-2026-10-10)):
each such format by its name, the operand every path shares in two terms, and as its single
pass, `<format>_1x` (decision A-23); `tests/support/precision_campaigns.py` defines them. All
five were measured on one revision on 2026-10-11 for the evidence table. Rerun one after any
change to a kernel it prices, from the repository's root, then regenerate the evidence table
(the suite fails until it is):

```
.venv/Scripts/python.exe -m tests.support.precision_campaigns linear
.venv/Scripts/python.exe -m tests.support.precision_evidence
```

Timings belong to the device and machine named in each file. They establish compile reuse
(none after a case's first seed), not a controlled comparison of speed. The 2026-10-16
refresh includes concurrent CPU/GPU runs and a suspended long-running CPU campaign;
its elapsed-time outliers must not be used for throughput claims. Dedicated hardware
timing remains separate work.

The 2026-10-16 accuracy campaigns compare per-path product rounding through FP8:
`rounding` (linear products), `rounding_options` (rollback) and `rounding_market_risk`.
Each compares nearest with stochastic rounding on matched draws; the seed varies as
traced data. Stochastic operands are conditionally unbiased; nonlinear figures need
not be. Long continuation values are bounded at zero before LongShort and delivery.
The options storage/compute campaigns are rerun after that projection.

`replicated_storage` measures paired means with eight independent scrambles within
each run and 32 independent outer seeds. Pair a whole-block prefix within each scramble.
Coverage audits diagnostic Student t intervals (7 degrees of freedom) per date,
including zero-width misses beyond roundoff. Nominal intervals are unavailable without
a validated tail bound (I-111), and are counted separately from diagnostic checks; the earlier single-scramble
coverage rows describe the old IID estimates and are historical measurements. The
precision acceptance table uses errors against matched float64 draws across outer
seeds, independently of any reported standard error. Its keys distinguish replicate
count and compute rounding. Formats that fail the unchanged standard remain failures.

Every record names its products' pricing settings (`settings`, I-114): per product, the
settings that change its kernel's arithmetic, as the campaign's analytic ran them
(`PricingConfig.kernel_settings`): the European engine, the rollback grid (ORE's `nx` and
`sx`, an American's exercise steps) and whether a calibrated option keeps one model
(`model=fixed`) or is recalibrated under each market-risk scenario (`model=recalibrated`).
The evidence table carries them per campaign and the engine's lookup matches them, so a row
validates a run only under the settings it was measured with. The campaigns measured
before 2026-10-18 had their settings recorded on that date from their definitions, which
are unchanged from each record's commit (no number of theirs moved); a campaign measured
since records its own.

`market_risk_calibrated` (2026-10-18) measures the kernel the three fixed-LGM market-risk
campaigns do not: their trades, market, shifts and coarse grid, with the Bermudan calibrated
and recalibrated under every scenario (I-112), in every combination they measure: the
storage cases, each compute format to nearest, and the `rounding_market_risk` formats
stochastically.
