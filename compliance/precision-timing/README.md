# Precision timing records

For the joined CPU/accelerator report with matched scenario outputs, use
[`python -m benchmarks.comparison`](../../benchmarks/README.md). It composes the same
campaigns and timing workers; its collection retains snapshots, failures, unavailable
hardware, exact settings and empirical reference uncertainty in one report.

What the timing harness (`tests/support/precision_timing.py`; method:
[docs/planning/details/precision.md §19](../../docs/planning/details/precision.md#19-the-research-timing-wall-time-per-figure-at-equal-accuracy-f-07-2026-10-19))
measured: per campaign of record and device, `<campaign>-<device>.json.gz` holds every number (each
case's cold and warm whole-job times, compile seconds, peak device memory, how its products ran,
every seed's figures, the speedups at equal accuracy) and `<campaign>-<device>.md` its summary.

    python -m tests.support.precision_timing compute --seeds 8 --paths 4096 16384 65536

The research result is measured on Ironwood and H100 (F-07). A record from a CPU or a consumer GPU
(an RTX 5060 runs float64 at 1/64 of its float32 rate, and FP8 natively only with
`XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`, I-97) is local evidence that the harness runs end
to end, not that result.
