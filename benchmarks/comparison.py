"""Collect and join the bounded precision comparison (F-07).

Run from the checkout: python -m benchmarks.comparison --help.
By default, time jobs through the production HTTP API, including an ORE CPU baseline.
Use --transport direct for the original in-process precision/equal-accuracy study.
"""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import statistics
import sys
import zipfile

from demos.benchmark_reporting import atomic_json, launch_worker, named_errors, read_json

FORMATS = ("float64", "float32", "float16", "bfloat16", "float8_e4m3fn", "float8_e5m2")
POLICIES = ("combined", "compute", "storage")
BACKENDS = {"cpu": "cpu", "gpu": "cuda,cpu", "tpu": "tpu,cpu"}
WORKLOADS = ("comparison_single", "comparison_portfolio")


def worker(arguments, destination):
    """One fresh process per point; select a backend before JAX is imported."""
    os.environ["JAX_PLATFORMS"] = BACKENDS[arguments["backend"]]
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    route = arguments.get("gemm", "auto")
    if arguments["backend"] == "gpu" and route != "auto":
        from engine.api.worker import gemm_environment
        os.environ.update(gemm_environment(route, os.environ))
    from engine.api.worker import deterministic_kernels_environment
    os.environ.update(deterministic_kernels_environment(os.environ))
    import jax

    try:
        devices = jax.devices()
    except RuntimeError as exc:
        atomic_json(destination / "result.json", {"status": "unmeasured", "reason": str(exc), **arguments})
        return 0
    expected = arguments["backend"]
    if devices[0].platform != expected:
        raise RuntimeError(f"requested {expected}, got {devices[0].platform}")
    required = arguments.get("expected_gpu")
    if expected == "gpu" and required and any(required.lower() not in d.device_kind.lower() for d in devices):
        raise ValueError(f"expected GPU {required}, got {[d.device_kind for d in devices]}")
    if jax.config.jax_disable_jit:
        raise ValueError("comparison requires JIT execution; JAX_DISABLE_JIT is set")
    from tests.support.precision_timing import _child
    # An inherited JAX_DEFAULT_DEVICE must not turn a requested GPU point into CPU work.
    with jax.default_device(devices[0]):
        timing = _child(arguments["campaign"], arguments["label"], arguments["paths"], arguments["seeds"], None,
                        snapshot_directory=str(destination), cpu_inputs=True)
    reason = "warm compilation" if timing["warm_compiles"] else None
    observed = timing["metadata"]["execution_devices"]
    if not observed or any(not d.startswith(expected + ":") for d in observed):
        reason = f"outputs on unexpected devices: {observed}"
    status = "invalid" if reason else "ok"
    atomic_json(destination / "result.json", {**arguments, "status": status, "timing": timing,
                                            "reason": reason})
    return 0


def collect(directory, paths, seeds, backends, formats, workloads, timeout, rtol=1e-4, atol=1e-8,
            policies=("combined",), reference_seeds=None, reference_paths=None, gemm="auto", expected_gpu=None):
    """Persist the full expected matrix before starting; never reuse stale worker results."""
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError(f"output directory must be empty: {directory}")
    requests = []
    for campaign in workloads:
        for backend in backends:
            for name in formats:
                for policy in (("combined",) if name == "float64" else policies):
                    for count in paths:
                        label = "float64" if name == "float64" else f"pricing:{name}" if policy == "combined" else f"pricing:{policy}:{name}"
                        point = {"campaign": campaign, "backend": backend, "label": label,
                                 "paths": count, "seeds": seeds}
                        folder = name if policy == "combined" else f"{policy}-{name}"
                        point["directory"] = f"{campaign}/{backend}/{folder}/{count}"
                        requests.append(point)
        if reference_seeds:
            for count in reference_paths or [max(paths)]:
                requests.append({"campaign": campaign, "backend": "cpu", "label": "float64:reference",
                                 "paths": count, "seeds": reference_seeds,
                                 "directory": f"{campaign}/cpu/independent-reference/{count}"})
    for point in requests:
        point.update(gemm=gemm, expected_gpu=expected_gpu)
    atomic_json(directory / "manifest.json", {"schema_version": 1, "requests": requests})
    for point in requests:
        destination = directory / point["directory"]
        print(f"Measuring {point['directory']}", flush=True)
        command = [sys.executable, "-m", "benchmarks.comparison", "--worker", json.dumps(point),
                   "--out", str(destination)]
        code = launch_worker(command, destination, timeout)
        if code or not (destination / "result.json").exists():
            atomic_json(destination / "result.json", {**point, "status": "failed",
                        "reason": f"worker exit {code}; see worker.log (124 means timeout)"})
        # Keep a reviewable partial report if a later process is interrupted or runs out of memory.
        write_report(directory, load_points([directory]), rtol, atol)
    return requests


def load_points(directories):
    """Read only manifest-listed results, retain missing rows and reject ambiguous joins."""
    points, keys = [], set()
    for directory in directories:
        directory = Path(directory).resolve()
        manifest = read_json(directory / "manifest.json")
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported comparison manifest")
        for request in manifest["requests"]:
            key = tuple(request[k] for k in ("campaign", "backend", "label", "paths"))
            if key in keys:
                raise ValueError(f"duplicate comparison point: {key}")
            keys.add(key)
            path = (directory / request["directory"]).resolve()
            if not path.is_relative_to(directory):
                raise ValueError("point directory escapes collection")
            point = read_json(path / "result.json") if (path / "result.json").exists() else {
                **request, "status": "failed", "reason": "missing result"}
            if any(point.get(k) != request[k] for k in ("campaign", "backend", "label", "paths", "seeds")):
                raise ValueError(f"result does not match manifest: {path}")
            point["root"] = str(path)
            points.append(point)
    return points


def compatible(reference, candidate, *, independent=False):
    """Precision/device may differ; workload, implementation, inputs and outputs may not."""
    a, b = reference["timing"], candidate["timing"]
    if independent and set(a["seeds"]) & set(b["seeds"]):
        raise ValueError("independent reference shares seed identities")
    if not independent and a["seeds"] != b["seeds"]:
        raise ValueError("seed identities differ")
    for name in ("workload", "input_generation"):
        if a["metadata"][name] != b["metadata"][name]:
            raise ValueError(f"{name} differs")
    for name in ("engine", "tool_digest", "jax"):
        if a["metadata"]["environment"][name] != b["metadata"]["environment"][name]:
            raise ValueError(f"implementation differs: {name}")
    if a["metadata"]["environment"]["versions"] != b["metadata"]["environment"]["versions"]:
        raise ValueError("library versions differ")
    for point in (reference, candidate):
        timing = point["timing"]
        if timing["seeds"] != point["seeds"] or set(timing["snapshots"]) != {str(s) for s in point["seeds"]}:
            raise ValueError("snapshot seed identities differ from request")
        if timing["warm_compiles"] or not timing["warm_seconds"]:
            raise ValueError("warm timing unavailable or compiled")
        observed = timing["metadata"]["execution_devices"]
        if not observed or any(not d.startswith(point["backend"] + ":") for d in observed):
            raise ValueError("execution devices differ from requested backend")


def snapshot_path(point, filename):
    root = Path(point["root"])
    path = (root / filename).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("snapshot escapes point directory")
    return path


def build_report(points, rtol=1e-4, atol=1e-8):
    """One evidence object: timings, named errors, uncertainty and equal-accuracy costs."""
    import numpy as np
    from tests.support.precision_harness import FIXED
    from tests.support.precision_timing import CaseTiming, equal_accuracy, figure_errors

    references = {(p["campaign"], p["paths"]): p for p in points
                  if p["backend"] == "cpu" and p["label"] == "float64" and p["status"] == "ok"}
    comparisons = []
    for point in points:
        row = {k: point[k] for k in ("campaign", "backend", "label", "paths", "status")}
        row["reason"] = point.get("reason")
        ref = references.get((point["campaign"], point["paths"]))
        if point["status"] == "ok" and point["label"] == "float64:reference":
            row["accuracy"] = "independent reference"
        elif point["status"] == "ok" and ref is not None:
            try:
                compatible(ref, point)
                row["seeds"] = {seed: named_errors(snapshot_path(ref, ref["timing"]["snapshots"][seed]),
                                                    snapshot_path(point, filename), rtol, atol)
                                for seed, filename in point["timing"]["snapshots"].items()}
                if not row["seeds"]:
                    raise ValueError("no snapshots")
                row["accuracy"] = "PASS" if all(r["status"] == "PASS" for outputs in row["seeds"].values()
                                                               for r in outputs.values()) else "FAIL"
            except (ValueError, KeyError, OSError, zipfile.BadZipFile, EOFError) as exc:
                row.update(accuracy="unmeasured", reason=str(exc))
        elif point["status"] == "ok":
            row.update(accuracy="unmeasured", reason="CPU FP64 reference unavailable")
        comparisons.append(row)

    accuracy, uncertainty, errors, reference_checks = [], [], [], []
    for campaign in sorted({p["campaign"] for p in points}):
        valid = [p for p, r in zip(points, comparisons) if p["campaign"] == campaign and "seeds" in r]
        if not valid:
            continue
        timings = []
        for p in valid:
            t = {**p["timing"], "case": "float64" if p["backend"] == "cpu" and p["label"] == "float64"
                 else f"{p['backend']}/{p['label']}"}
            timings.append(CaseTiming(**t))
        reference = [t for t in timings if t.case == "float64"]
        independent = [p for p in points if p["campaign"] == campaign and p["label"] == "float64:reference"]
        independent_timings = []
        if independent:
            try:
                for p in independent:
                    if p["status"] != "ok":
                        raise ValueError("independent reference unavailable")
                    for candidate in valid:
                        compatible(p, candidate, independent=True)
                    independent_timings.append(CaseTiming(**p["timing"]))
            except (ValueError, KeyError) as exc:
                reference_checks.append({"campaign": campaign, "status": "invalid", "reason": str(exc)})
                continue
        reference_runs = independent_timings or reference
        largest = max(reference_runs, key=lambda t: t.paths)
        for name, values in largest.figures.items():
            values = np.asarray(values)
            uncertainty.append({"campaign": campaign, "figure": name, "paths": largest.paths,
                                "mean": values.mean(axis=0).tolist(),
                                "empirical_standard_error": (values.std(axis=0, ddof=1) / np.sqrt(len(values))).tolist(),
                                "seeds": list(largest.seeds), "confidence_interval": "unavailable",
                                "independent": bool(independent_timings)})
        for t in sorted(reference_runs, key=lambda t: t.paths):
            reference_checks.append({"campaign": campaign, "status": "measured", "paths": t.paths,
                                     "seeds": list(t.seeds), "independent": bool(independent_timings),
                                     "mean_changes_to_largest": {
                                         name: (np.mean(t.figures[name], axis=0) - np.mean(values, axis=0)).tolist()
                                         for name, values in largest.figures.items()}})
        accuracy.extend({"campaign": campaign, **dataclasses.asdict(r), "speedup": r.speedup}
                        for r in equal_accuracy(timings, FIXED, references=independent_timings or None))
        errors.extend({"campaign": campaign, "case": case, "paths": paths, "figure": figure, "relative_rms": error}
                      for (case, paths, figure), error in figure_errors(timings, FIXED, references=independent_timings or None).items())
    return {"schema_version": 1, "tolerances": {"rtol": rtol, "atol": atol}, "points": points,
            "comparisons": comparisons, "equal_accuracy": accuracy, "reference_uncertainty": uncertainty,
            "sampling_errors": errors, "reference_convergence": reference_checks}


def write_report(directory, points, rtol=1e-4, atol=1e-8):
    data = build_report(points, rtol, atol)
    atomic_json(directory / "comparison.json", data)
    temporary = directory / "report.md.tmp"
    temporary.write_text(markdown(data), encoding="utf-8")
    temporary.replace(directory / "report.md")
    return data


def markdown(data):
    lines = ["# Precision comparison", "",
             "Local measurements only. Native/emulated labels describe dispatch, not verified hardware use.",
             "Scenario PV = arithmetic-consistent base anchor + stored P&L. Fixed stages retain A-10.",
             "Times include CPU scenario generation, host preparation, calibration, transfers, reductions and required host outputs.",
             "Cold excludes process startup/imports; compile is backend compilation within cold, not an additive cost.",
             "Snapshot compression and report IO are outside the clock. Peak is allocator high-water per device across all seeds; host RSS unavailable.",
             "Warm compilation invalidates timing. Missing hardware, failures and accuracy misses are retained.", "",
             "| Workload / backend / format / paths | Status / accuracy | Cold s | Compile s | Warm median s | Warm compiles | Peak bytes | Dispatch |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    notes = []
    for p, r in zip(data["points"], data["comparisons"]):
        label = f"{p['campaign']} / {p['backend']} / {p['label']} / {p['paths']}"
        t = p.get("timing")
        if t:
            lines.append(f"| {label} | {p['status']} / {r.get('accuracy', 'unmeasured')} | {t['cold_seconds']:.4f} | "
                         f"{t['compile_seconds']:.4f} | {statistics.median(t['warm_seconds']):.4f} | "
                         f"{t['warm_compiles']} | {t['device_peak_bytes'] if t['device_peak_bytes'] is not None else 'unavailable'} | {t['products']} |")
        else:
            lines.append(f"| {label} | {p['status']} | — | — | — | — | — | — |")
        if r.get("reason") and t:
            notes.append(f"- {label}: {r['reason']}")
        elif p.get("reason"):
            notes.append(f"- {label}: {' '.join(p['reason'].splitlines())}")
    if notes:
        lines += ["", "## Unavailable or invalid measurements", "", *notes]
    lines += ["", "## Matched-scenario errors", "", "Absolute errors are retained for zero/near-zero figures; tolerance = atol + rtol × reference maximum.",
              "These are engineering tolerances, not A-11 acceptance or ORE parity certification.", "",
              "| Workload / backend / format / paths / output | Check | Max absolute error over seeds | Max relative error |",
              "|---|---|---:|---:|"]
    for row in data["comparisons"]:
        seeds = row.get("seeds", {})
        for name in sorted({k for outputs in seeds.values() for k in outputs}):
            values = [outputs[name] for outputs in seeds.values()]
            finite = [v for v in values if "max_absolute_error" in v]
            absolute = max((v["max_absolute_error"] for v in finite), default=None)
            relative = max((v["relative_error"] for v in finite if v["relative_error"] is not None), default=None)
            check = "PASS" if all(v["status"] == "PASS" for v in values) else "FAIL"
            lines.append(f"| {row['campaign']} / {row['backend']} / {row['label']} / {row['paths']} / {name} | "
                         f"{check} | {absolute} | {relative if relative is not None else 'undefined (zero scale)'} |")
    lines += ["", "## Equal accuracy", "",
              "Empirical RMS error against CPU FP64's multi-seed mean at the largest count; this finite reference is not truth.",
              "Its mean and empirical SE per figure are in comparison.json; nominal confidence bounds are unavailable.",
              "Reference seed independence and convergence are recorded per campaign in comparison.json. Without separate reference seeds errors are dependent.",
              "Finite-reference uncertainty remains empirical; convergence diagnostics are not confidence bounds. Interpolation never extrapolates.",
              "With independent references, speedups are withheld unless reference empirical SE is <=10% of the target error.",
              "Fixed entries are excluded by the existing timing estimator; matched errors above still check them.", "",
              "| Workload / case / figure | CPU FP64 paths | Candidate paths | Whole-job speedup |", "|---|---:|---:|---:|"]
    for r in data["equal_accuracy"]:
        lines.append(f"| {r['campaign']} / {r['case']} / {r['figure']} | {r['float64_paths']} | "
                     f"{r['paths'] if r['paths'] is not None else 'not reached'} | {r['speedup']} |")
    lines += ["", "Exact settings, versions, process flags, source digests, raw per-seed timings, errors and warnings: [comparison.json](comparison.json).",
              "Reproduce with the source checkout identified by the digests and each collection's manifest.json.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--transport", choices=("api", "direct"), default="api",
                        help="API end-to-end timing (default), or the original direct precision study")
    parser.add_argument("--repeats", type=int, default=3,
                        help="fresh repeated API jobs per seed, after the first job")
    parser.add_argument("--output", choices=("artifact", "inline"), default="artifact",
                        help="API P&L output; artifact downloads are included in elapsed time")
    parser.add_argument("--merge", type=Path, nargs="+", help="join existing collections; no engine execution")
    parser.add_argument("--paths", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--formats", nargs="+", choices=FORMATS, default=list(FORMATS))
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=["combined"],
                        help="vary pricing storage/compute together or independently")
    parser.add_argument("--reference-seeds", type=int, nargs="+", help="disjoint CPU FP64 reference seeds")
    parser.add_argument("--reference-paths", type=int, nargs="+", help="reference convergence counts (default: largest count)")
    parser.add_argument("--gemm", choices=("auto", "default", "cublas"), default="auto",
                        help="auto preserves the inherited emitter for every format; explicit routes override it")
    parser.add_argument("--expected-gpu", help="required GPU model substring, e.g. H100")
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=list(WORKLOADS))
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(json.loads(args.worker), args.out)
    import math
    if (len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 3 or min(args.seeds) < 0
            or len(set(args.paths)) != len(args.paths) or any(n < 128 or n & (n - 1) for n in args.paths)):
        parser.error("use at least three distinct non-negative seeds and distinct power-of-two counts >= 128 "
                     "(ES at 99% needs at least one tail observation)")
    if any(not math.isfinite(v) or v < 0 for v in (args.rtol, args.atol)) or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("tolerances must be finite/non-negative and timeout finite/positive")
    for items in (args.backends, args.formats, args.workloads, args.policies):
        if len(items) != len(set(items)):
            parser.error("duplicate matrix selections")
    if args.reference_paths and not args.reference_seeds:
        parser.error("--reference-paths requires --reference-seeds")
    if args.reference_seeds is not None and (len(args.reference_seeds) < 3
            or len(set(args.reference_seeds)) != len(args.reference_seeds)
            or min(args.reference_seeds) < 0 or max(args.reference_seeds) >= 2 ** 32
            or set(args.reference_seeds) & set(args.seeds)):
        parser.error("use at least three distinct uint32 reference seeds disjoint from candidate seeds")
    if max(args.seeds) >= 2 ** 32:
        parser.error("seed identities must fit uint32")
    if args.reference_paths and (len(set(args.reference_paths)) != len(args.reference_paths)
            or any(n < 128 or n & (n - 1) for n in args.reference_paths)):
        parser.error("reference counts must be distinct powers of two >= 128")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.transport == "api":
        if args.reference_seeds or args.reference_paths:
            parser.error("independent precision references require --transport direct; API mode compares matched ORE outputs")
        from benchmarks.api_comparison import run
        return run(args)
    if not args.merge:
        collect(args.out, sorted(args.paths), args.seeds, args.backends, args.formats, args.workloads, args.timeout,
                args.rtol, args.atol, args.policies, args.reference_seeds, args.reference_paths,
                args.gemm, args.expected_gpu)
    points = load_points(args.merge or [args.out])
    data = write_report(args.out, points, args.rtol, args.atol)
    print(f"Comparison report: {args.out / 'report.md'}", flush=True)
    return int(any(p["status"] in ("failed", "invalid") or r.get("accuracy") == "unmeasured"
                   for p, r in zip(points, data["comparisons"]))
               or any(r["status"] == "invalid" for r in data["reference_convergence"]))


if __name__ == "__main__":
    sys.exit(main())
