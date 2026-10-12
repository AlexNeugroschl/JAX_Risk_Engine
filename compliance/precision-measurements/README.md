# Precision measurements

The precision harness's measurements of record (I-75;
[docs/planning/details/precision.md §15.4](../../docs/planning/details/precision.md#154-the-storage-harness-2026-10-09)):
how far each storage and compute format, per stage, moves every figure of a run from float64,
on the run's own paths, per product and path count, over independent Sobol seeds. The precision
evidence table ([../precision-evidence.md](../precision-evidence.md), I-55) is generated from
them.

| File | Holds |
|---|---|
| `<campaign>.json.gz` | Every number: per case, figure and path count the precision error's mean (bias), standard error and root mean square, and the figure's statistical error, per date; the paired estimator's coverage; the P&L attribution statistics; recompiles and timings; the market; the commit, whether the tree had local changes, and the digest of the engine's Python sources. `tests.support.precision_harness.Measurements.from_json` reads it |
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

Timings are CPU timings of the machine named in each file: evidence of compiles (none after a
case's first seed), not of speed.
