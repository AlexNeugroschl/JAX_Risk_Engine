# Precision measurements

The storage harness's measurements of record (I-75;
[docs/planning/details/precision.md §15.4](../../docs/planning/details/precision.md#154-the-storage-harness-2026-10-09)):
how far each storage format, per stage, moves every figure of a run from float64, on the run's
own paths, per product and path count, over 16 independent Sobol seeds. The precision evidence
table (I-55, `compliance/precision-evidence.md` when it lands) is read off them.

| File | Holds |
|---|---|
| `<campaign>.json.gz` | Every number: per case, figure and path count the precision error's mean (bias), standard error and root mean square, and the figure's statistical error, per date; the paired estimator's coverage; the P&L attribution statistics; recompiles and timings; the commit and environment. `tests.support.precision_harness.Measurements.from_json` reads it |
| `<campaign>.md` | The summary generated from it |

The campaigns are `linear` (the shared portfolio's swaps, Europeans and bond), `options` (its
Bermudans and American) and `market_risk`; `tests/support/precision_campaigns.py` defines
them. Rerun one after any change to a kernel it prices, from the repository's root:

```
.venv/Scripts/python.exe -m tests.support.precision_campaigns linear
```

Timings are CPU timings of the machine named in each file: evidence of compiles (none after a
case's first seed), not of speed.
