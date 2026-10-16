# Trace Python and GPU execution together

For untraced per-instrument CPU/GPU timing, full numerical comparisons and scaling charts,
use the separate [GPU scaling study](gpu-scaling.md). This page covers the mixed-workload
trace demo.

[`demos/demo_profile_python.py`](../../demos/demo_profile_python.py) records Python
function calls, JAX dispatch and compilation, and GPU kernels on one timeline. Python
still runs on the CPU; the compiled numerical work runs on the GPU. These settings use
JAX's [profiling API](https://docs.jax.dev/en/latest/profiling.html).

The workload, input configuration, warm-up, profiler options, synchronization, and both
engine calls are in `main()`. The only helper reads the trace and writes its summary.
No API server is needed: the demo calls the engine in the process being profiled.

Verified on 2026-10-14 with JAX 0.10.2 on an RTX 5060 Laptop GPU under WSL2: the
default whole-engine capture contained 2,123,625 events, including 1,319,388 Python call
events and 66,173 GPU events. All six split captures also recorded Python and GPU work;
the largest contained 784,528 events, below the approximate JSON export limit. XProf
successfully discovered the whole-engine recording. These measurements are for the warm
default workload, not a cold compilation run.

## Run on an NVIDIA GPU

Use **Linux or WSL2**. The native Windows `.venv/Scripts/python.exe` runs JAX on CPU;
it cannot collect a CUDA execution trace. See the
[GPU setup instructions](../getting-started/user-guide.md#on-a-gpu-linux-or-wsl2-on-windows)
for the driver requirements and environment setup.

From the repository root in a Linux/WSL terminal, with a Linux virtual environment:

```bash
.venv/bin/python -m pip install -c constraints.txt -e ".[gpu,profiling]"
.venv/bin/python demos/demo_profile_python.py
.venv/bin/xprof --port 8791 --logdir .profile-out-python
```

If your environment is elsewhere, replace `.venv/bin/python` with its interpreter path.
The demo requires CUDA by default and exits if it cannot initialize it. It prints the GPU
name, checks the result arrays' backend, and checks that the saved trace actually contains
GPU events and Python calls. CUDA execution can work while CUPTI collection fails; a
host-only capture is reported as an error, with the diagnostic files retained.

Open **http://localhost:8791**, select the run and `trace_viewer`. Expand the host thread
for Python calls and the GPU streams for kernel execution. Search for `portfolio`,
`calibration`, `simulation`, `pricing`, `exposure`, `greeks`, and `market_risk`.
Python call labels include the filename, line number and function. The nested
`greeks/trade<i>/<ConfigType>` regions identify each trade's Greeks.

The script prints an XProf command scoped to the new run. Every invocation creates a
unique timestamp/PID directory, so existing traces are kept. Each capture has a
`summary.json` with its settings, device, event counts, recorded span, host time per phase,
and path to the `.xplane.pb`. The summary is read from that full record.

**Use XProf to read the `.xplane.pb`.** JAX also writes a `.trace.json.gz` export that can
keep only approximately the first million events. Python tracing can exceed that even
on a small warm run. A partial JSON export does not mean that the XPlane recording is
partial. The summary reports this threshold and unusually short recorded spans; those
checks do not prove that CUPTI dropped no individual events. Keep profiler stderr visible.

## What runs

The default is a small representative run through the engine's two analytics entry points:

| Region | Work included |
|---|---|
| `portfolio` | `price_portfolio`: validation and orchestration, calibrated cross-asset model, simulation, all pricers, exposure statistics, and Greeks |
| `calibration` | USD and EUR Hull-White models fitted to two swaption helpers each |
| `simulation` | 64 paths on three future dates, with correlated interest rates, EURUSD FX and AAPL equity |
| `pricing` | USD and EUR swaps, European/Bermudan/American swaptions, Treasury bill and coupon note, today and on paths |
| `exposure` | Netting-set and individual-trade exposure profiles, including EPE/ENE/PFE and Basel figures |
| `greeks` | AD sensitivities for all seven trades, with per-trade labels |
| `market_risk` | Shock generation, 256 rate scenarios, all seven trades revalued, portfolio P&L, VaR and ES |

This covers the computational engine on one configuration. HTTP transport, job queues,
TraderX ingestion, startup/imports, initial input construction, alternative configurations,
and the optional reduced-precision paired sample are outside this capture. Pricing's
schedule construction and Python orchestration inside the entry points are included.
The market-risk calculation shocks rate pillars; its fixed-volatility and fixed-FX
limitations are printed with the result.

The option grid is deliberately coarse (`n_per_std=8`, five standard deviations,
one American exercise step per year). The small scenario counts and grids serve profiling;
the reported figures are demonstration outputs. The larger preset below uses the standard
option grid. Edit the visible configuration in `main()` to match your intended portfolio.

## Larger workloads and representative measurements

```bash
.venv/bin/python demos/demo_profile_python.py --preset representative --baseline
.venv/bin/xprof --port 8791 --logdir .profile-out-python
```

The presets resolve to these settings; explicit command-line overrides take precedence:

| Setting | `small` (default) | `representative` |
|---|---:|---:|
| Trades | 7 | 28 |
| Exposure paths | 64 | 4,096 |
| Market-risk scenarios | 256 | 8,192 |
| Simulation dates | 3 annual | 8: quarterly in year one, then annual to year five |
| Option state grid (`n_per_std`, `std_devs`) | 8, 5 | 30, 5 |
| American exercise steps/year | 1 | 24 |
| Greeks | AD | AD |
| Arithmetic | float64 | float64 |

The larger preset uses four groups of the seven base contracts. Notionals, fixed rates,
coupon rates and option/swap sides vary, while schedules and contract shapes are shared.
This exercises per-trade preparation and reuse of compiled programs for common shapes.
It is a synthetic capacity test: matching a production book also needs its actual product
mix, maturity/exercise distribution, simulation grid, curves, calibration settings,
precision policy and Greeks method. Increasing all counts does not by itself establish
production representativeness. Some of the short base contracts mature before the last
simulation dates; adjust those contracts when modelling a longer-dated book.

`--baseline` runs one additional **untraced warm repeat** between warm-up and capture.
`timings.json` and each trace summary retain its time. Compare that baseline with the
full trace's `wall_seconds` to measure profiler overhead; the baseline covers the whole
job, so it is not comparable directly to one selected phase's time. `--baseline` and
`--cold` cannot be combined. First-run compilation is measured separately by `--cold`.

You can vary one dimension at a time:

```bash
# Same seven trades and coarse grid; isolate scenario-count scaling.
.venv/bin/python demos/demo_profile_python.py --paths 4096 --risk-scenarios 8192 --baseline

# Standard grids with fewer trades; explicit options override the larger preset.
.venv/bin/python demos/demo_profile_python.py --preset representative --portfolio-copies 1 --baseline

# Keep Python tracing and limit capture to the larger book's pricing phase.
.venv/bin/python demos/demo_profile_python.py --preset representative --phase pricing
```

Other controls are `--portfolio-copies N` (seven trades per group), `--date-grid annual|demo`,
`--n-per-std N`, and `--exercise-steps N`. Larger GPU batches often increase compute time
without a proportional increase in trace events. Additional trades, simulation dates,
exercise rollbacks and market-risk batches can increase the number of events substantially.
Portfolio count and scenario count therefore test different costs.

Measured on 2026-10-14, JAX 0.10.2, RTX 5060 Laptop GPU (WSL2), with Python tracing on:

| Workload | Warm traced workload | Events | XPlane size |
|---|---:|---:|---:|
| Small: 7 trades, 64 paths, 256 risk scenarios, coarse option grid | 4.06 s | 2,123,561 | 65.1 MiB |
| Same contracts/grid: 4,096 paths and 8,192 risk scenarios | 5.15 s | 2,408,199 | 75.9 MiB |
| Larger preset: 28 trades, 4,096 paths, 8,192 risk scenarios, eight dates, standard grids | 31.54 s | 7,661,803 | 268.2 MiB |

The larger preset's first untraced pass took 90.50 s including compilation; its untraced
warm baseline took 19.91 s. Capture plus profiler startup/export took 55.20 s, before
reading the record for its summary. The record held 3,951,231 Python-call events and
744,623 GPU events. Within the two pipeline intervals, the union of GPU kernel intervals
was 9.82 s; including device copies, the union of GPU activity was 10.06 s. These are
recorded activity durations under instrumentation, not hardware utilization counters or
an unprofiled GPU/CPU time breakdown.

All expected phases and the final `capture_end` marker were present. The capture log
contained no dropped-event/CUPTI error messages; this is not a guarantee against every
possible missing event. XProf successfully served the overview timeline (71,568 viewer
events, about 15.6 s for initial preparation) and a late 29–33 s window (about 0.39 s).
This checks the streaming backend used by the viewer; browser rendering speed depends
on the client. The JSON export remains subject to its separate event cap.

## Collection and rendering limits

XPlane avoids the JSON export's approximately one-million-event limit, but **it is not
unlimited or guaranteed lossless**. The collector first buffers events in host memory,
then writes the record to disk. Finite collector buffers can drop events before they
reach the file; insufficient RAM, disk capacity or GPU memory can fail collection or the
workload. JAX documents separate GPU callback and activity event limits, each defaulting
to 2,097,152 in the current profiler. These limits are not the total event count printed
by this demo. See [JAX profiler options](https://docs.jax.dev/en/latest/profiling.html).

Keep stderr visible and inspect it for dropped-event, CUPTI, allocation or write errors.
Summaries record the XPlane file size, device allocator memory statistics (when supported),
and expected phase markers. A full capture also has a `capture_end` marker after both
pipelines and their pending device work have finished. Its presence demonstrates that
recording reached the end; it cannot prove that every intervening event survived.

XProf streams timeline detail as you pan and zoom, so it need not render every event
at once. Its **Switch to V2** viewer is intended for large recordings. Loading/converting
the trace still consumes host RAM and disk space; a large file may take time to open.
See [XProf Trace Viewer](https://openxla.org/xprof/trace_viewer). If a capture becomes
unwieldy, retain the same larger workload and use `--split` or `--phase`, keeping Python
tracing enabled. Do not reduce the workload merely to make the viewer faster.

## Warm and cold traces

The default runs both analytics once **without tracing**, waits for all pending device
work, then records a repeat. This keeps first-use compilation out of most of the trace
and makes Python-inclusive capture more manageable. It can still contain compilation
if some programs were not reused.

To include first-use Python tracing, lowering and compilation, start a new process with:

```bash
.venv/bin/python demos/demo_profile_python.py --cold
```

`--cold` skips warm-up and disables JAX's persistent compilation cache, including a cache
configured through the environment. Backend initialization and initial input construction
remain outside the trace. Python-inclusive cold traces can be much larger than warm ones.

The demo waits for device work before recording and before ending the capture. Whole-run
phase labels show host intervals: asynchronous GPU execution can extend into the next
phase. Separate-phase captures synchronize at the selected phase's boundaries to include
its kernels. Printed completion time includes profiler startup/export; the summary's
`wall_seconds` measures the selected capture window, excluding export.

## If the full capture is too large

Keep Python tracing enabled and collect each engine phase separately:

```bash
.venv/bin/python demos/demo_profile_python.py --split
```

This writes **six traces**: calibration, simulation, pricing, exposure, Greeks, and
market risk. Each capture reruns the real workload, starts tracing only upon entry to
that phase, waits for its device work and then stops tracing. The other phases execute
untraced so the selected phase has the same inputs. Warm-up runs once before the six
passes. The seven pricers are all included in `pricing`; all seven trades' sensitivities
are in `greeks`. Running phases separately takes longer overall but bounds the amount
recorded in each file. Python calls and GPU events remain enabled in every capture.

To narrow a particularly large capture further:

```bash
.venv/bin/python demos/demo_profile_python.py --phase pricing
.venv/bin/python demos/demo_profile_python.py --phase greeks/trade4/AmericanSwaptionConfig
.venv/bin/python demos/demo_profile_python.py --phase market_risk
```

`--phase portfolio` captures the entire exposure/Greeks pipeline while leaving market
risk untraced. `--split` and `--phase` are mutually exclusive.

For a **cold phase**, use a fresh invocation per phase, for example
`--cold --phase pricing`. In `--cold --split`, only the first pass starts with empty
in-process caches; later passes reuse programs from earlier passes.

Other useful controls:

```bash
.venv/bin/python demos/demo_profile_python.py --paths 1024 --risk-scenarios 4096
.venv/bin/python demos/demo_profile_python.py --model LGM
.venv/bin/python demos/demo_profile_python.py --greeks Bump --phase greeks
```

Bump-and-revalue can take considerably longer than AD. `--paths` must be a positive power
of two for the Sobol simulation. `--output DIRECTORY` changes the trace root.

## CPU checks and collection problems

For a native Windows smoke check, from PowerShell:

```powershell
.venv/Scripts/python.exe demos/demo_profile_python.py --device cpu
.venv/Scripts/xprof.exe --port 8791 --logdir .profile-out-python
```

On Linux use the same `--device cpu` option with your Linux interpreter. CPU mode keeps
Python tracing on and reports zero GPU events.

If the CUDA workload runs but the trace has no GPU events, inspect stderr for CUPTI
loading or permission errors. Follow JAX's
[GPU profiling troubleshooting](https://docs.jax.dev/en/latest/profiling.html#xprof-cuptierror)
and your host's NVIDIA profiling permissions. Switching to CPU verifies the demo's host
tracing but does not validate GPU collection. The older
[`demo_profile_small.py`](../../demos/demo_profile_small.py) remains useful for HTTP/worker
profiling; its Python tracer is disabled by default.
