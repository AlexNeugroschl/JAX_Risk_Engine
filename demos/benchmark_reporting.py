"""Process, file, comparison and plotting mechanics for demo_gpu_scaling.py.

No engine execution lives here; the benchmark workload and its timing are in main().
"""
import csv
import dataclasses
import json
import re
import subprocess
import threading
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def launch_worker(command, destination, timeout):
    """Stream the child's messages to both the terminal and a persistent log."""
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)

        def copy_output():
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)

        thread = threading.Thread(target=copy_output, daemon=True)
        thread.start()
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            code = 124
            print(f"Worker timed out after {timeout}s; partial results are retained.", flush=True)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            thread.join()
            raise
        thread.join()
    return code


def numeric_arrays(value, prefix="output"):
    """Flatten observable numeric outputs, preserving names/shapes, not just checksums."""
    import numpy as np

    result = {}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            result.update(numeric_arrays(getattr(value, field.name), f"{prefix}/{field.name}"))
    elif isinstance(value, dict):
        for key, child in value.items():
            result.update(numeric_arrays(child, f"{prefix}/{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            result.update(numeric_arrays(child, f"{prefix}/{index}"))
    elif isinstance(value, (bool, int, float, np.number)) or (hasattr(value, "shape") and hasattr(value, "dtype")):
        array = np.asarray(value)
        if array.dtype.kind in "biufc":
            result[prefix] = array
    return result


def compare_snapshots(cpu_path, gpu_path, rtol, atol):
    import numpy as np

    with np.load(cpu_path, allow_pickle=False) as cpu, np.load(gpu_path, allow_pickle=False) as gpu:
        if set(cpu.files) != set(gpu.files):
            return {"status": "FAIL", "reason": "output names differ"}
        maximum, absolute, worst, count = 0.0, 0.0, "", 0
        for key in cpu.files:
            a, b = cpu[key], gpu[key]
            if a.shape != b.shape:
                return {"status": "FAIL", "reason": f"shape differs: {key}"}
            if a.dtype != b.dtype:
                return {"status": "FAIL", "reason": f"dtype differs: {key} ({a.dtype}, {b.dtype})"}
            count += a.size
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                return {"status": "FAIL", "reason": f"non-finite output: {key}"}
            if a.dtype.kind in "biu" or b.dtype.kind in "biu":
                if not np.array_equal(a, b):
                    return {"status": "FAIL", "reason": f"discrete output differs: {key}"}
                continue
            difference = np.abs(a - b)
            error = float(np.max(difference / (atol + rtol * np.maximum(np.abs(a), np.abs(b))), initial=0))
            absolute = max(absolute, float(np.max(difference, initial=0)))
            if error > maximum:
                maximum, worst = error, key
        return {"status": "PASS" if maximum <= 1 else "FAIL", "max_tolerance_ratio": maximum,
                "max_absolute_difference": absolute, "worst_output": worst, "elements": count}


def union_seconds(intervals, start, end):
    total, previous = 0, start
    for left, right in sorted((max(left, start), min(right, end)) for left, right in intervals
                              if right > start and left < end):
        total += max(0, right - max(left, previous))
        previous = max(previous, right)
    return total / 1e9


def trace_summary(directory, wall_seconds, baseline, compiles):
    from jax.profiler import ProfileData

    path = max(directory.rglob("*.xplane.pb"), key=lambda p: p.stat().st_mtime)
    events = python = gpu = 0
    region, end_time = None, None
    activity, kernels = [], []
    for plane in ProfileData.from_file(str(path)).planes:
        for line in plane.lines:
            for event in line.events:
                events += 1
                python += int(event.name.startswith("$"))
                if event.name == "benchmark":
                    region = (event.start_ns, event.start_ns + event.duration_ns)
                if event.name == "capture_end":
                    end_time = event.start_ns
                if plane.name.startswith("/device:GPU:") and line.name.startswith("Stream"):
                    gpu += 1
                    interval = (event.start_ns, event.start_ns + event.duration_ns)
                    activity.append(interval)
                    if not event.name.startswith("Memcpy"):
                        kernels.append(interval)
    end_marker = end_time is not None
    complete_markers = region is not None and end_marker and end_time >= region[1]
    document = {"xplane": str(path), "bytes": path.stat().st_size, "events": events,
                "python_events": python, "gpu_events": gpu, "end_marker": end_marker,
                "benchmark_marker": region is not None, "ordered_markers": complete_markers, "wall_seconds": wall_seconds,
                "untraced_median_seconds": baseline, "profiler_slowdown": wall_seconds / baseline,
                "compiles": compiles, "json_may_be_truncated": events > 1_000_000,
                "gpu_activity_seconds": union_seconds(activity, *region) if region else None,
                "gpu_kernel_seconds": union_seconds(kernels, *region) if region else None}
    atomic_json(directory / "summary.json", document)
    if not (complete_markers and python and gpu):
        raise RuntimeError("Trace lacks Python/GPU activity or expected boundary markers; see its summary.json")
    print(f"  trace: {events:,} events; {document['bytes'] / 2**20:.1f} MiB; "
          f"{document['profiler_slowdown']:.2f}x profiler slowdown; end marker present", flush=True)
    return document


def write_report(directory, plots=True):
    """Join device results, compare full snapshots, write CSV/Markdown and plot artifacts."""
    directory = Path(directory)
    settings = read_json(directory / "settings.json")
    documents = [read_json(p) for p in sorted(directory.glob("*/*/results.json"))]
    rows = [row for document in documents for row in document["rows"]]
    failure_file = directory / "worker_failures.json"
    worker_failures = read_json(failure_file) if failure_file.exists() else []
    groups = list(settings["instruments"]) if any(s in settings["stages"] for s in ("pricing", "workflow", "greeks", "market_risk")) else []
    if "simulation" in settings["stages"]:
        groups += ["simulation_HullWhite", "simulation_LGM"]
    if "mixed" in settings["stages"]:
        groups += ["mixed"]
    complete = {(d["metadata"]["backend"], d["metadata"]["group"]) for d in documents if d.get("complete")}
    missing = [f"{device}/{group}" for device in dict.fromkeys(settings["devices"]) for group in groups
               if (device, group) not in complete]
    pairs = {}
    for row in rows:
        pairs.setdefault((row["group"], row["id"]), {})[row["backend"]] = row
    comparisons = []
    failed = bool(missing or worker_failures) or any(row["status"] != "ok" or row.get("trace_error") for row in rows)
    for (group, point_id), pair in pairs.items():
        cpu, gpu = pair.get("cpu"), pair.get("gpu")
        comparison = {"group": group, "id": point_id, "status": "NOT COMPARED"}
        if cpu and gpu and cpu["status"] == gpu["status"] == "ok":
            comparison.update(compare_snapshots(directory / cpu["snapshot"], directory / gpu["snapshot"],
                                                settings["rtol"], settings["atol"]))
            comparison["speedup"] = cpu["median_seconds"] / gpu["median_seconds"]
            failed |= comparison["status"] == "FAIL"
        elif {"cpu", "gpu"}.issubset(settings["devices"]):
            comparison["status"] = "MISSING/FAILED"
            failed = True
        comparisons.append(comparison)
        for row in pair.values():
            row["comparison"] = comparison["status"]
            row["cpu_over_gpu"] = comparison.get("speedup")
    atomic_json(directory / "comparisons.json", comparisons)
    atomic_json(directory / "results.json", {"settings": settings, "documents": documents,
                                             "comparisons": comparisons, "incomplete_workers": missing,
                                             "worker_failures": worker_failures,
                                             "has_failures": failed})
    fields = ("backend", "group", "stage", "axis", "x", "layout", "method", "paths", "trades", "scenarios",
              "batch", "effective_batches", "status", "comparison", "cpu_over_gpu", "first_call_seconds",
              "first_call_compiles", "median_seconds", "min_seconds", "max_seconds", "spread_fraction",
              "warm_compiles", "throughput", "units", "error", "trace_error")
    with (directory / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    warnings = [f"CPU/GPU mismatch {c['group']}/{c['id']}: "
                f"{c.get('reason', c.get('worst_output', 'see comparisons.json'))}"
                for c in comparisons if c["status"] == "FAIL"]
    warnings += [f"Worker exited with code {f['exit_code']}: {f['backend']}/{f['group']}; inspect worker.log."
                 for f in worker_failures]
    for row in rows:
        label = f"{row['backend']}/{row['group']}/{row['id']}"
        if row["status"] != "ok":
            warnings.append(f"FAILED {label}: {row.get('error', '')}")
        if sum(row.get("warm_compiles", [])):
            warnings.append(f"Warm compilation occurred in {label}; investigate before interpreting throughput.")
        if row.get("spread_fraction", 0) > 0.2:
            warnings.append(f"Sample range exceeds 20% of median in {label}; timing needs confirmation.")
        if row.get("trace_error"):
            warnings.append(f"TRACE FAILED {label}: {row['trace_error']}")
    pattern = re.compile(r"CUPTI_ERROR|RESOURCE_EXHAUSTED|out of memory|dropped\s+[1-9]|[1-9]\d*\s+(?:events|records).*dropped", re.I)
    for log in directory.glob("*/*/worker.log"):
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if pattern.search(line):
                warnings.append(f"Inspect {log.relative_to(directory)}: {line[:300]}")
    overview = ["## Overview", "",
                "Latency below is the largest path-pricing point, or the simulation/mixed point for those groups.",
                "Warm-compile and timing-warning counts cover all device/point rows. Inspect them before drawing conclusions.", "",
                "| Case | Numerical comparisons | Warm-compile rows | Variable-timing rows | CPU s | GPU s | Speedup |",
                "|---|---|---:|---:|---:|---:|---:|"]
    for group in groups:
        checks = [c for c in comparisons if c["group"] == group]
        passed = sum(c["status"] == "PASS" for c in checks)
        check_text = f"{passed}/{len(checks)} PASS" if any(c["status"] != "NOT COMPARED" for c in checks) else "NOT COMPARED"
        members = [r for r in rows if r["group"] == group and r["status"] == "ok"]
        stage = "simulation" if group.startswith("simulation_") else "mixed" if group == "mixed" else "pricing"
        selected = [r for r in members if r["stage"] == stage and r["axis"] == "paths"]
        largest = max((r["x"] for r in selected), default=None)
        latencies = {r["backend"]: r["median_seconds"] for r in selected if r["x"] == largest}
        cpu = f"{latencies['cpu']:.4f}" if "cpu" in latencies else "—"
        gpu = f"{latencies['gpu']:.4f}" if "gpu" in latencies else "—"
        speed = f"{latencies['cpu'] / latencies['gpu']:.2f}x" if len(latencies) == 2 else "—"
        compiling = sum(bool(sum(r.get("warm_compiles", []))) for r in members)
        varying = sum(r.get("spread_fraction", 0) > 0.2 for r in members)
        overview.append(f"| {group} | {check_text} | {compiling} | {varying} | {cpu} | {gpu} | {speed} |")
    lines = ["# GPU scaling results", "", f"Run: `{directory}`", "",
             f"Status: {'INCOMPLETE / FAILED CHECKS' if failed else 'REQUESTED WORKERS AND OUTPUT CHECKS COMPLETED'}.", "",
             "This is a synthetic scaling study, not a production certification or an ORE parity suite.", "",
             f"Preset: {settings.get('preset', 'custom')}; measured repeats: {settings.get('repeats', '?')}; "
             f"warmups: {settings.get('warmups', '?')}; option n_per_std: {settings.get('n_per_std', '?')}; "
             f"American steps/year: {settings.get('exercise_steps', '?')}.", "",
             f"CPU/GPU comparison: rtol={settings['rtol']}, atol={settings['atol']}; discrete outputs must match exactly.",
             "First-call times are local first-use timings; preceding points/setup may already have compiled shared programs.",
             "Warm timings exclude profiling and snapshot/report IO, and include waiting for device completion.",
             "Workflow = calibration + simulation + pricing + exposure; Greeks are separate. Mixed also includes AD Greeks.",
             "Pricing = public path-pricing API on prepared scenarios; schedule preparation remains included.",
             "Market risk uses prepared shocks; batch is requested size, effective_batches records memory-based limits.",
             "Device allocator peaks in raw JSON are cumulative within each worker, not per-point peaks or total GPU memory.", "",
             *overview, "",
             "## CPU/GPU comparison", "",
             "Speedup = CPU median / GPU median. Above 1 means GPU faster for that same point.", "",
             "| Case / point | Check | GPU speedup | Max error / tolerance |",
             "|---|---|---:|---:|"]
    for comparison in comparisons:
        speed = f"{comparison['speedup']:.2f}x" if "speedup" in comparison else "—"
        error = f"{comparison['max_tolerance_ratio']:.3g}" if "max_tolerance_ratio" in comparison else "—"
        lines.append(f"| {comparison['group']} / {comparison['id']} | {comparison['status']} | {speed} | {error} |")
    lines += ["", "## Timings", "", "| Device / case / point | First call s | Warm median s | Min–max s | Warm compiles | Throughput |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        label = f"{row['backend']} / {row['group']} / {row['id']}"
        if row["status"] == "ok":
            lines.append(f"| {label} | {row['first_call_seconds']:.4f} | {row['median_seconds']:.4f} | "
                         f"{row['min_seconds']:.4f}–{row['max_seconds']:.4f} | {sum(row['warm_compiles'])} | "
                         f"{row['throughput']:,.0f} {row['units']} |")
        else:
            lines.append(f"| {label} | FAILED | — | — | — | — |")
    lines += ["", "## Traces", "", "Only replay captures are profiled; benchmarks above are untraced.",
              "XPlane markers check the boundaries, not the absence of every possible dropped event.",
              "GPU activity durations are timeline intervals under profiling, not hardware utilization counters.", ""]
    for row in rows:
        trace = row.get("trace")
        if trace:
            trace_path = Path(trace["xplane"])
            if trace_path.is_absolute():
                trace_path = trace_path.relative_to(directory)
            lines += [f"- {row['group']} / {row['id']}: {trace['events']:,} events, {trace['bytes'] / 2**20:.1f} MiB; "
                      f"{trace['profiler_slowdown']:.2f}x slowdown; end marker {trace['end_marker']}; "
                      f"[XPlane]({trace_path.as_posix()})."]
    lines += ["", "View all captures with XProf (`xprof --port 8791 --logdir <this-run-directory>`).",
              "Read the XPlane-backed view; the separately exported JSON can be truncated at about one million events.",
              "Use benchmark to locate the selected workload and capture_end to find its end.", "", "## Checks requiring attention", ""]
    lines += [f"- Incomplete worker: {name}" for name in missing]
    lines += [f"- {warning}" for warning in warnings] or ["- No additional point-level warnings."]
    if plots and any(row["status"] == "ok" for row in rows):
        write_plots(directory, rows)
        lines += ["", "## Charts", "", "[Scaling charts (PDF)](scaling.pdf)", "",
                  "Each panel holds other point settings fixed. Error bars are observed min/max, not confidence intervals.",
                  "Compare throughput only within the same product, stage, layout and numerical settings."]
    lines += ["", "See docs/concepts/gpu-scaling.md for methodology and interpretation."]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return failed


def write_plots(directory, rows):
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    charts = directory / "charts"
    charts.mkdir(exist_ok=True)
    successful = [r for r in rows if r["status"] == "ok"]
    with PdfPages(directory / "scaling.pdf") as pdf:
        for group in dict.fromkeys(r["group"] for r in successful):
            members = [r for r in successful if r["group"] == group]
            panels = sorted({(r["stage"], r["axis"], r["layout"], r["method"]) for r in members})
            for metric in ("median_seconds", "throughput"):
                fig, axes = plt.subplots(math.ceil(len(panels) / 3), 3, figsize=(14, 3.6 * math.ceil(len(panels) / 3)), squeeze=False)
                for axis, key in zip(axes.flat, panels):
                    panel = [r for r in members if (r["stage"], r["axis"], r["layout"], r["method"]) == key]
                    for backend, color in (("cpu", "#c05b28"), ("gpu", "#276ac5")):
                        series = sorted((r for r in panel if r["backend"] == backend), key=lambda r: r["x"])
                        if not series:
                            continue
                        y = [r[metric] for r in series]
                        if metric == "median_seconds":
                            low = [r["median_seconds"] - r["min_seconds"] for r in series]
                            high = [r["max_seconds"] - r["median_seconds"] for r in series]
                        else:
                            low = [r["throughput"] - r["work_items"] / r["max_seconds"] for r in series]
                            high = [r["work_items"] / r["min_seconds"] - r["throughput"] for r in series]
                        axis.errorbar([r["x"] for r in series], y, yerr=[low, high], fmt="o-", capsize=3,
                                      label=backend.upper(), color=color)
                    axis.set_title(" / ".join(key if key[0] == "greeks" else key[:3]), fontsize=9)
                    axis.set_xlabel(key[1])
                    axis.set_ylabel("seconds" if metric == "median_seconds" else panel[0]["units"])
                    axis.set_xscale("log", base=2)
                    ticks = sorted({r["x"] for r in panel})
                    axis.set_xticks(ticks, labels=[f"{tick:,}" for tick in ticks])
                    axis.set_yscale("log")
                    axis.grid(True, alpha=0.2)
                    axis.legend(fontsize=8)
                for axis in list(axes.flat)[len(panels):]:
                    axis.set_visible(False)
                fig.suptitle(f"{group}: {'warm latency' if metric == 'median_seconds' else 'warm throughput'}\n"
                             "Untraced measurements; min/max error bars; check correctness in report.md", fontsize=12)
                fig.tight_layout(rect=(0, 0, 1, 0.96))
                pdf.savefig(fig)
                fig.savefig(charts / f"{group}-{metric}.png", dpi=130)
                plt.close(fig)
