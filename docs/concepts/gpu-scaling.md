# Measure GPU efficiency by instrument

[`demos/demo_gpu_scaling.py`](../../demos/demo_gpu_scaling.py) runs a CPU/GPU scaling
study for every implemented instrument family, then records selected Python-inclusive
GPU traces. Its output is a results table, full numerical comparisons, scaling charts
and diagnostic traces. Performance comes from **untraced warm measurements**; traces
explain those measurements.

The workload definitions, contract variation, engine calls, timing loops and profiler
boundaries are visible in `main()`. Backend launching, file handling, comparisons and
plotting live in [`benchmark_reporting.py`](../../demos/benchmark_reporting.py).

## Run it

Use a Linux/WSL Python environment for the GPU study, from the repository root:

```bash
python -m pip install -c constraints.txt -e ".[gpu,profiling,benchmark]"

# Validate the full study on a small workload first.
python demos/demo_gpu_scaling.py --preset quick

# Standard pricing grids and larger scaling sweeps.
python demos/demo_gpu_scaling.py --preset standard
```

Using the existing GPU environment on the development workstation:

```bash
cd /mnt/c/PROJECTS/JAX_Risk_Engine
PYTHONPATH=. ~/venvs/jre311/bin/python demos/demo_gpu_scaling.py --preset quick
```

`benchmark` installs Matplotlib for standalone PDF/PNG plots. `--no-plots` omits only
the charts and does not need Matplotlib. `profiling` installs the XProf viewer; Python
tracing uses JAX itself. Native Windows can run `--devices cpu` to check execution and
reporting, but CPU/GPU comparisons must run through a Linux/WSL interpreter.

Every invocation prints a unique directory under `.profile-out-benchmark/`. It keeps
progress after each point and logs each worker. The default repeats each warmed point
**five times**, following two unmeasured warmups. Both presets cover the same cases;
`quick` deliberately uses coarse option grids and small counts. It is a functional check,
not a production-resolution performance result. Standard runs can take substantially
longer, particularly with many American/Bermudan trades or bump-and-revalue Greeks.

| Setting | Quick | Standard (default) |
|---|---|---|
| Exposure path sweep | 64, 256 | 1,024, 4,096, 16,384 |
| Trade-count sweep | 1, 2 | 1, 8, 32 |
| Risk scenario sweep | 256, 512 | 1,024, 4,096, 16,384 |
| Requested risk batch sweep | 128, 256 | 256, 1,024, 4,096 |
| Option `n_per_std` / standard deviations | 8 / 5 | 30 / 5 |
| American exercise steps/year | 2 | 24 |
| Arithmetic | float64 | float64 |
| Warm measurements | 5 | 5 |

Override any sweep with explicit lists:

```bash
python demos/demo_gpu_scaling.py --instruments american --stages pricing market_risk \
  --paths 1024 4096 16384 --trades 1 8 --risk-scenarios 4096 16384 \
  --risk-batches 256 1024 4096
```

Path counts must be powers of two. Risk counts must be at least 256 so the reported
99% ES standard error has more than one tail observation. This minimum does not make
256 scenarios sufficient for stable production risk estimates.

## What the experiment covers

There are seven instrument/engine cases:

- Vanilla interest-rate swap.
- European swaption using Bachelier.
- European swaption using Jamshidian.
- Calibrated Bermudan swaption using the LGM grid.
- Calibrated American swaption using the LGM grid.
- Treasury bill.
- Treasury coupon note.

Each instrument has these independently measured stages:

| Stage | Inside the timer | Prepared beforehand |
|---|---|---|
| `pricing` | Public `value_paths` API: per-trade preparation and path pricing | Market, calibrated simulation model and scenario market |
| `workflow` | `price_portfolio`: model calibration, simulation, pricing and exposure | Market, trade configs and run request |
| `greeks` | All requested Greeks for the trades; AD by default | Market, trade configs, pricing settings |
| `market_risk` | Full scenario revaluation, P&L, VaR/ES and reporting | Market, trade configs and seeded rate shocks |

The simulation workers separately measure calibration and simulation for **Hull-White
and LGM**, on the shared two-currency IR/FX/equity market. Simulation timing uses a
prebuilt calibrated model; calibration has its own row.

A final `mixed` worker prices a six-trade portfolio (one per instrument family), with
simulation, exposure and AD Greeks together. It uses Bachelier for the European; a run's
pricing config selects that engine for all Europeans, so the Jamshidian case is measured
separately. Market risk is separately measured by product rather than included in this
mixed exposure job. Shocks move rates only; the engine's fixed-volatility/fixed-FX market
risk limitations still apply.

All contracts are alive through the simulation grid: quarterly in year one, then 18 and
24 months. This avoids making throughput look artificially good by filling late dates
with expired contracts. The `bill` case uses the engine's generic zero-coupon bond config
with a synthetic three-year maturity for that purpose; it is not a literal market Treasury
bill booking. For real short-dated bills, shorten the simulation grid as well. The note has
semiannual coupons. The Bermudan has multiple
exercise dates and the American a nontrivial exercise window. This samples instrument
families and engines; it does not exhaust all settlement, position, curve, calendar,
precision or integration configurations.

### Vary one dimension at a time

For pricing and workflows, the path sweep holds one trade fixed. The trade sweep holds
the smallest requested path count fixed. Greeks vary trade count, because today's
Greeks do not gain work from additional exposure paths.

There are two trade layouts:

- `shared`: values vary, while contract shapes remain the same. This measures per-trade
  overhead and reuse of compiled programs for common shapes.
- `varied`: coupon counts/maturities change, and Bermudan/American exercise schedules
  also change. This tests a small set of distinct shapes as a separate experiment.

Varied layouts cycle through a few shapes rather than creating an entirely unique
program for every trade. Compare scaling **within** a layout: its contract complexities
need not equal the other layout's. The current engine prices trades in a Python loop;
increasing trade count does not imply batching that entire axis into one GPU program.

Market risk has separate scenario-count, requested-batch-size and trade-count sweeps.
The batch sweep fixes the largest requested scenario count. Increasing total scenarios
at a fixed batch size often means **more launches**, not larger GPU work per launch.
`effective_batches` records each trade's actual maximum batch size after the grid
engine's memory limit is applied; the final remainder may be smaller still.

## Timing and reproducibility

The coordinator executes workers **sequentially**, with a fresh process per backend and
instrument/model. CPU and GPU do not compete with each other during a measurement.
Run with other heavy workloads stopped. Use the same machine, power mode and numerical
settings when comparing runs. A laptop GPU's float64 results are not performance forecasts
for a datacenter GPU.

Each worker requires its requested backend; it never silently substitutes CPU. Scenario
sharding is limited to one device. Inputs use fixed seeds. The process disables persistent
compilation caching and records JAX/NumPy/Python versions, platform, flags, CPU logical-core
count, demo/engine source hashes and resolved settings. Deterministic GPU operations are
requested unless your XLA flags specify a choice.

The worker waits for outstanding device work before starting **and** before stopping
every timer. Otherwise a JAX timer can measure only enqueue time. See
[JAX benchmarking](https://docs.jax.dev/en/latest/201/profiling.html).

`first_call_seconds` includes any compilation for that local first use. It is **not a
whole-engine cold-start measurement**: imports and input construction are outside it,
and previous points or input preparation may already have compiled shared programs.
Both first-use and measured warm samples record backend compilation counts. Any warm
compilation is flagged so it cannot masquerade as ordinary kernel execution time.

Snapshot copying, compression, tracing, report writing and plotting occur outside the
timed samples. GPU allocator statistics in the raw JSON contain cumulative worker peaks;
they are not per-point peaks and do not include all driver/context memory. A worker
process isolates cache/memory lifetime between products.

## Read the results in this order

### 1. Check execution and numerical agreement

Open `report.md`. Incomplete workers, failed points and trace errors appear explicitly,
and the command exits nonzero if execution or numerical checks fail. A timeout defaults
to 3,600 seconds **per worker**; override with `--timeout SECONDS` for longer experiments.
An out-of-memory point is reported as failed, never silently rerun at a smaller size.

When both devices are requested, corresponding full numeric snapshots are compared:
scenario arrays, pricing cubes, exposure outputs and exercise decisions, Greeks, or P&L
and risk outputs, as appropriate. This checks every saved value, not a mean or checksum.
Shapes and dtypes must agree; discrete arrays must agree exactly. For floating outputs the default gate is:

```text
abs(cpu - gpu) <= 1e-6 + 1e-8 * max(abs(cpu), abs(gpu))
```

The report's **max error / tolerance** must be at most 1. `comparisons.json` identifies
the worst output and maximum absolute difference. These are configurable demonstration
tolerances, not a replacement for the repository's tighter ORE parity tests. Investigate
mismatches, especially differing exercise decisions, before judging speed. CPU-only or
GPU-only studies say `NOT COMPARED`, never imply numerical agreement.

### 2. Check whether timing is trustworthy

Check **warm compiles**, and the observed minimum/maximum across repeats. A range over
20% of the median is flagged for confirmation; this is a heuristic, not a universal failure
threshold. Increase `--repeats`, rerun on an otherwise idle system and investigate compiling
programs. Five samples do not support reliable production p95/p99 latency claims.
Equivalent configurations can occur under different sweep axes (for example, one trade
at the base path count). Compare those medians as an additional consistency check. A
substantial change can reveal clock/thermal drift or other changing conditions that a
single point's sample range misses; confirm it before attributing a curve to the code.

### 3. Inspect latency, throughput and speedup

Open `scaling.pdf` or the PNGs under `charts/`. Each product has latency and throughput
pages with separate panels for stage, varying dimension, layout and Greeks method.
CPU and GPU curves share the panel. Error bars are observed min/max, not confidence
intervals. Axes use logarithmic scales: equal spacing means equal ratios, not equal
absolute differences; x ticks display the actual counts. `results.csv` contains raw per-point summaries and `results.json` retains the
samples, device metadata, memory statistics and diagnostics.

- **Latency down** at identical work/precision means an improvement.
- **Throughput up** means more useful work completed per second within that same case.
- **GPU speedup above 1** means the GPU beat the CPU for that point. Below 1 means the CPU
  was faster; that can be reasonable for small, cheap workloads.
- If four times more work takes much less than four times longer, fixed overhead is being
  amortized. If time becomes approximately proportional to work at large sizes, throughput
  may have reached a plateau. A plateau alone does not establish hardware efficiency.
- Falling throughput, abrupt memory-related cliffs, persistent warm recompilation or
  increasing host gaps at large batches identify places to investigate.

Throughput units depend on the stage: trade-path-dates/s, trade-scenarios/s, trade Greeks/s,
path-dates/s or currency calibrations/s. They are bookkeeping units, not equal computational
costs across products. Do not compare a swap's number directly with an American's, or a
workflow's with pricing alone. Some simulation dates carry exercised options' underlyings;
their work differs from pre-exercise rollback even though both count as a valuation.

### 4. Use the selected traces to explain the curves

By default (`--traces selected`) each GPU worker replays its **slowest measured point
per stage**. These are additional calls after all its untraced measurements. Rewarming
occurs before capture. Python tracing stays enabled. Trace summaries record the traced
duration, profiler slowdown against that point's median, compilation count, Python/GPU
event counts, file size, and `benchmark`/`capture_end` markers.

```bash
xprof --port 8791 --logdir .profile-out-benchmark/<printed-run-directory>
```

Select the product/point run, then `trace_viewer`. Search for `benchmark`, select that
region and press **F** to frame it. Compare the main Python thread and GPU compute/copy
streams over the same interval. Horizontal width measures time; nested Python rows do
not represent extra CPU cores or independent elapsed time.

| Trace pattern | Next investigation |
|---|---|
| GPU empty while Python prepares inputs | Vectorization/reuse of preparation; host reads and synchronization |
| Many tiny kernels with gaps | Larger useful batches, dispatch overhead and opportunities for fusion |
| Unexpected compilation in warm work | Changing shapes/static arguments or unstable callable/cache identities |
| Frequent small CPU↔GPU copies | Scalar conversions, array placement and repeated transfers |
| Long GPU kernels with poor throughput | Profile the dominant kernels' compute/memory behaviour |

The recorded union of GPU kernel/activity intervals measures busy time **under tracing**.
It is not SM occupancy, memory-bandwidth efficiency, or unprofiled GPU utilization. High
busy time is not proof of efficient kernels. Use Nsight Systems for host/device scheduling
and Nsight Compute for selected dominant kernels if hardware-counter analysis is needed;
see [NVIDIA's triage guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/).
The demo does not collect those hardware counters.

XProf streams data as you zoom; **Switch to V2** is useful for large traces. It reads
the `.xplane.pb`, avoiding the separate JSON export's approximate million-event cap.
XPlane itself is not unlimited: collector buffers can drop events before writing, and
host RAM/disk may run out. End markers establish that recording reached the end, not
that every individual event survived. Inspect worker logs for CUPTI/dropped-event errors.
See [XProf Trace Viewer](https://openxla.org/xprof/trace_viewer).

Use `--traces none` for a timing-only study; `--traces all` records every successful
GPU point and can consume substantial time and disk. Trace settings do not change the
timing workload. For narrower investigations, select `--instruments` and `--stages`.

## Extend, resume and review

```bash
# Include the separately implemented bump Greeks; potentially much slower.
python demos/demo_gpu_scaling.py --instruments swap european_bachelier \
  --stages greeks --greeks-methods AD Bump

# Resume an interrupted study using its saved settings; completed workers are retained.
python demos/demo_gpu_scaling.py --resume .profile-out-benchmark/<run>

# Recompute comparisons and charts without running the engine.
python demos/demo_gpu_scaling.py --report-only .profile-out-benchmark/<run>
```

Resume retries an incomplete worker from its beginning, not from its last point; this
keeps that worker's cache-warming history consistent. Trace runs have unique timestamps.
Keep source code and environment fixed when resuming, or start a new study. Worker logs
and per-point snapshots are retained for review. A report-only invocation uses its
`--no-plots` option independently of the original run.

The current scope is a single-process, single-device computational study. HTTP ingestion,
queues, production concurrency, multi-device scaling and operational tail latency need
separate system/load tests. A production decision should combine numerical parity,
per-instrument scaling, representative mixed-book throughput, memory headroom and those
operational tests. There is no universal GPU-percentage threshold that replaces them.

## Implementation validation

The initial development check on the RTX 5060 Laptop GPU under WSL2 ran all quick cases
with one warmup and two measured repeats: **350 device/point rows, 175 full CPU/GPU output
comparisons passing, and 33 Python/GPU traces with valid boundary markers**. Both latency
and throughput charts were generated for all ten instrument/model/mixed groups. Those
reduced-repeat, coarse-grid timings serve functional validation, not production estimates.

A separate American run used standard grids, two warmups, five measured repeats,
1,024/4,096/16,384 paths, 4,096/16,384 risk scenarios and 256/1,024/4,096 batches.
All ten CPU/GPU point comparisons and both selected traces passed. Some CPU sample ranges
exceeded the variability warning threshold, so its speedup measurements need confirmation
before using them for capacity planning. CPU-only swap/European runs also exercised both
AD and bump-and-revalue Greeks.

Regression checks cover full-output comparisons (including equal means with wrong paths),
dtype/shape/discrete/nonfinite failures, overlapping GPU intervals, required end markers,
missing/failed workers, and an actual CPU CLI run with resume:

```bash
python -m pytest tests/test_gpu_scaling_demo.py -q
```
