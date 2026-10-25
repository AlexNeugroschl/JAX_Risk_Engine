"""End-to-end /runs timings: submit, wait, retrieve numerical outputs, stop.

The production HTTP API and SQLite queue are used for JAX and a benchmark-only
ORE CPU worker. Service startup is outside the timer; the first job, including
its initialization/compilation, is separate from repeated fresh submissions.
"""
import contextlib
from collections import Counter
import dataclasses
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time
import zipfile

from demos.benchmark_reporting import atomic_json, launch_worker, named_errors, read_json


def wire(value):
    """Dataclass inputs as API JSON, without deepcopying ORE's SWIG objects."""
    import numpy as np
    import ORE

    if isinstance(value, ORE.Date):
        return value.ISO()
    if isinstance(value, ORE.Period):
        return str(value)
    if dataclasses.is_dataclass(value):
        return {f.name: wire(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {wire(k): wire(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [wire(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def request_body(campaign, paths, seed, label, output):
    """Identical synthetic histories for both engines, built outside the API timer.

    history[i+h] - history[i] represents each existing campaign's Sobol shock.
    ORE's native HISTSIM_VAR and the public JAX API can both consume this history.
    These are synthetic research inputs, not observed historical market data.
    """
    import numpy as np
    from engine.api.requests import PrecisionSchema, RunRequestSchema
    from engine.precision import Precision

    factors = campaign.scenarios(paths, seed)
    shifts = np.asarray(factors.shifts)
    horizon = campaign.horizon_days
    levels = np.empty((paths + horizon, shifts.shape[1]))
    levels[:horizon] = np.concatenate([np.asarray(c.rates) for c in factors.factors.curves])
    for i, shift in enumerate(shifts):
        levels[i + horizon] = levels[i] + shift
    precision = Precision() if label == "float64" else next(c.precision for c in campaign.cases if c.label == label)
    trades = []
    for trade in campaign.trades:
        body = wire(trade)
        body.pop("evaluation_date")
        body["trade_type"] = trade.product
        trades.append(body)
    market = wire(campaign.market)
    for currency in market["currencies"].values():
        # TraderX provenance is not a field of the public curve-input schema.
        for curve in [currency["discount_curve"], *currency["index_curves"].values()]:
            curve.pop("provenance", None)
    body = {"market": market, "portfolio": trades, "analytics": ["market_risk"],
            "config": {"pricing": wire(campaign.pricing),
                       "precision": PrecisionSchema.from_dataclass(precision).model_dump(mode="json"),
                       "market_risk": {"quantiles": list(campaign.quantiles), "scenarios": {
                           "source": "historical", "history": levels.tolist(),
                           "factors": factors.factors.labels(),
                           "horizon_days": horizon}}}, "outputs": {"pnl": output}}
    schema = RunRequestSchema.model_validate(body)
    schema.check()
    return schema.model_dump(mode="json")


def snapshot(document, pnl):
    import numpy as np

    risk = document["market_risk"]
    ids = [t["trade_id"] for t in risk["trades"]]
    if risk["pnl"]["items"]["ids"] != ids or pnl.shape != (risk["num_scenarios"], len(ids)):
        raise ValueError("P&L shape or trade order differs from the result")
    base = np.asarray([t["base_npv"] for t in risk["trades"]])
    out = {"portfolio/PnL": pnl.sum(axis=1), "portfolio/base_PV": np.asarray(risk["base_npv"]),
           "portfolio/scenario_PV": pnl.sum(axis=1) + risk["base_npv"]}
    for i, trade in enumerate(ids):
        out.update({f"trades/{trade}/PnL": pnl[:, i], f"trades/{trade}/base_PV": base[i],
                    f"trades/{trade}/scenario_PV": base[i] + pnl[:, i]})
    out.update({f"portfolio/{k}": np.asarray(v) for k, v in risk["risk"].items()
                if k.startswith(("VaR_", "ES_")) and k.count("_") == 1})
    if any(not np.all(np.isfinite(v)) for v in out.values()):
        raise ValueError("non-finite numerical output")
    return out


def timed_request(client, body, timeout):
    """One clock around POST, long polling, JSON decoding and required artifact downloads."""
    import numpy as np
    from engine.api.artifacts import read_array

    start = time.perf_counter()
    deadline = time.monotonic() + timeout

    def remaining():
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("API job/result retrieval timed out")
        return left

    response = client.post("/runs", json=body, timeout=remaining())
    response.raise_for_status()
    run_id = response.json()["run_id"]
    while True:
        response = client.get(f"/runs/{run_id}", params={"wait": min(30, remaining())}, timeout=remaining())
        response.raise_for_status()
        status = response.json()
        if status["status"] not in ("pending", "running"):
            break
    if status["status"] != "done":
        raise RuntimeError(f"API job {run_id} {status['status']}: {status.get('error')}")
    document = status["result"]
    array = document["market_risk"]["pnl"]

    def fetch(url):
        response = client.get(url, timeout=remaining())
        response.raise_for_status()
        return response.content

    pnl = read_array(array, fetch) if array["form"] == "artifact" else np.asarray(array["values"], dtype=np.float64)
    elapsed = time.perf_counter() - start
    # Snapshot persistence and accuracy checks are outside the API stopwatch.
    return elapsed, run_id, document, pnl


@contextlib.contextmanager
def service(command, env, log_path):
    """Own the process tree, including Windows venv launchers, on success and failure."""
    from demos.worker_process import WindowsJob

    job = WindowsJob() if os.name == "nt" else None
    with log_path.open("w", encoding="utf-8") as log:
        process = None
        try:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=0x00000004 | subprocess.CREATE_NO_WINDOW
                                       if job else 0, start_new_session=job is None)
            if job:
                job.attach(process)
                job.resume(process)
            yield process
        finally:
            if job:
                job.close()
            elif process is not None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            if process is not None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name != "nt":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()


def server(directory):
    """Bind a private loopback port without a find-free-port race."""
    import uvicorn

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    atomic_json(directory / "address.json", {"url": f"http://127.0.0.1:{sock.getsockname()[1]}"})
    try:
        uvicorn.Server(uvicorn.Config("engine.api.app:app", log_level="warning")).run(sockets=[sock])
    finally:
        sock.close()


def wait_ready(client, process, timeout):
    import httpx

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"API server exited {process.returncode}; see server.log")
        try:
            response = client.get("/health", timeout=1)
            if response.status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(.05)
    raise TimeoutError("API server startup timed out; see server.log")


def worker(point, directory):
    import httpx
    import numpy as np

    # The measuring client generates common inputs on CPU; accelerators belong to
    # its separate engine worker. Settings are set before importing engine/JAX.
    os.environ["JAX_PLATFORMS"] = "cpu"
    from benchmarks.comparison import BACKENDS
    from engine.api.worker import deterministic_kernels_environment, gemm_environment
    from engine.api.job_queue import JobQueue
    from tests.support.precision_campaigns import COMPARISON_CAMPAIGNS

    campaign = COMPARISON_CAMPAIGNS[point["campaign"]]()
    label = "float64" if point["label"] in ("ore:float64", "float64:reference") else point["label"]
    bodies = {str(seed): request_body(campaign, point["paths"], seed, label, point["output"])
              for seed in point["seeds"]}
    atomic_json(directory / "requests.json", bodies)
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_RISK_WORKER="external",
               JAX_RISK_JOB_QUEUE=str(directory / "jobs.sqlite"),
               JAX_RISK_WORKER_LOCK=str(directory / "worker.lock"),
               JAX_ENABLE_COMPILATION_CACHE="false", JAX_COMPILATION_CACHE_DIR="",
               XLA_PYTHON_CLIENT_PREALLOCATE="false")
    # Profiling inherited from a shell would change user-visible elapsed time.
    for name in ("JAX_RISK_PROFILE_DIR", "JAX_RISK_PROFILE_PHASE", "JAX_DEFAULT_DEVICE"):
        env.pop(name, None)
    engine_env = dict(env, JAX_PLATFORMS=BACKENDS[point["backend"]])
    engine_env.update(deterministic_kernels_environment(engine_env))
    if point["backend"] == "gpu" and point["gemm"] != "auto":
        engine_env.update(gemm_environment(point["gemm"], engine_env))
    command = [sys.executable, "-m", "benchmarks.api_worker", "--implementation", point["implementation"],
               "--backend", point["backend"], "--queue", env["JAX_RISK_JOB_QUEUE"],
               "--lock", env["JAX_RISK_WORKER_LOCK"], "--metadata", str(directory / "environment.json"),
               "--parent-pid", str(os.getpid())]
    if point.get("expected_gpu"):
        command += ["--expected-gpu", point["expected_gpu"]]
    with service([sys.executable, "-m", "benchmarks.api_comparison", "--serve", str(directory)],
                 env, directory / "server.log") as api, service(command, engine_env, directory / "engine.log") as engine:
        deadline = time.monotonic() + point["timeout"]
        address, metadata = directory / "address.json", directory / "environment.json"
        while not address.exists() or not metadata.exists():
            if metadata.exists() and read_json(metadata)["status"] == "unmeasured":
                atomic_json(directory / "result.json", {**point, **read_json(metadata)})
                return 0
            if api.poll() is not None or engine.poll() is not None:
                raise RuntimeError("API or engine startup failed; see server.log and engine.log")
            if time.monotonic() >= deadline:
                raise TimeoutError("service startup timed out")
            time.sleep(.05)
        environment = read_json(metadata)
        if environment["status"] != "ok":
            atomic_json(directory / "result.json", {**point, **environment})
            return 0
        with httpx.Client(base_url=read_json(address)["url"], trust_env=False) as client:
            wait_ready(client, api, point["timeout"])
            samples = []
            queue = JobQueue(env["JAX_RISK_JOB_QUEUE"])
            try:
                # First request uses the same workload as repeats, with no warm-up
                # pricing job and no idempotency key that could return cached results.
                plan = [(point["seeds"][0], "first")]
                plan += [(seed, "repeat") for seed in point["seeds"] for _ in range(point["repeats"])]
                for index, (seed, phase) in enumerate(plan):
                    elapsed, run_id, document, pnl = timed_request(client, bodies[str(seed)], point["timeout"])
                    risk = document["market_risk"]
                    precision = risk.get("precision")
                    if point["implementation"] == "jax" and (precision is None or precision["backend"] != point["backend"]):
                        raise ValueError("result did not execute on the requested backend")
                    if point["implementation"] == "ore" and document.get("benchmark_implementation") != "ore":
                        raise ValueError("result did not execute with ORE")
                    values = snapshot(document, pnl)
                    history = np.asarray(bodies[str(seed)]["config"]["market_risk"]["scenarios"]["history"])
                    horizon = campaign.horizon_days
                    values["inputs/shifts"] = history[horizon:] - history[:-horizon]
                    filename = f"sample-{index}.npz"
                    np.savez_compressed(directory / filename, **values)
                    job = queue.get(run_id)
                    samples.append({"seed": seed, "phase": phase, "seconds": elapsed, "run_id": run_id,
                                    "snapshot": filename, "compiles": job.compiles,
                                    "submitted": job.submitted, "started": job.started, "finished": job.finished,
                                    "products": precision["products"] if precision else {},
                                    "execution_devices": precision["devices"] if precision else environment["devices"],
                                    "realized": precision["realized"] if precision else {},
                                    "ore_app_runs": document.get("benchmark_ore_app_runs"),
                                    "warnings": document.get("warnings", [])})
                    atomic_json(directory / "result.json", {**point, "status": "ok", "environment": environment,
                                                             "samples": samples})
                    print(f"{point['implementation']}/{point['backend']} {phase} seed={seed}: {elapsed:.3f}s", flush=True)
            finally:
                queue.close()
    return 0


def collect(args):
    directory = args.out.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError(f"output directory must be empty: {directory}")
    requests = []
    for campaign in args.workloads:
        for count in sorted(args.paths):
            requests.append({"campaign": campaign, "backend": "cpu", "implementation": "ore", "label": "ore:float64",
                             "paths": count, "directory": f"{campaign}/ore/{count}"})
            for backend in args.backends:
                for name in args.formats:
                    for policy in (["combined"] if name == "float64" else args.policies):
                        label = "float64" if name == "float64" else f"pricing:{name}" if policy == "combined" else f"pricing:{policy}:{name}"
                        requests.append({"campaign": campaign, "backend": backend, "implementation": "jax", "label": label,
                                         "paths": count, "directory": f"{campaign}/{backend}/{policy}-{name}/{count}"})
    for point in requests:
        point.update(seeds=args.seeds, repeats=args.repeats, output=args.output, timeout=args.timeout,
                     gemm=args.gemm, expected_gpu=args.expected_gpu)
    atomic_json(directory / "manifest.json", {"schema_version": 2, "transport": "api", "requests": requests})
    for point in requests:
        destination = directory / point["directory"]
        print(f"Measuring API {point['directory']}", flush=True)
        code = launch_worker([sys.executable, "-m", "benchmarks.api_comparison", "--worker", json.dumps(point),
                              "--out", str(destination)], destination, args.timeout)
        if code or not (destination / "result.json").exists():
            partial = read_json(destination / "result.json") if (destination / "result.json").exists() else point
            atomic_json(destination / "result.json", {**partial, "status": "failed", "reason": f"worker exit {code}; see worker.log"})
        write_report(directory, load_points([directory]), args.rtol, args.atol)


def load_points(directories):
    points, keys = [], set()
    for directory in map(Path, directories):
        manifest = read_json(directory / "manifest.json")
        if manifest.get("schema_version") != 2 or manifest.get("transport") != "api":
            raise ValueError("API reports require API collections; use --transport direct for old collections")
        for request in manifest["requests"]:
            key = tuple(request[k] for k in ("campaign", "implementation", "backend", "label", "paths"))
            if key in keys:
                raise ValueError(f"duplicate comparison point: {key}")
            keys.add(key)
            root = (directory / request["directory"]).resolve()
            if not root.is_relative_to(directory.resolve()):
                raise ValueError("point directory escapes collection")
            point = read_json(root / "result.json") if (root / "result.json").exists() else {**request, "status": "failed", "reason": "missing result"}
            if any(point.get(k) != v for k, v in request.items()):
                raise ValueError("result does not match manifest")
            point["root"] = str(root)
            points.append(point)
    return points


def compare(reference, candidate, rtol, atol):
    """Match every repeat to the ORE result with the same seed, never just summary means."""
    if reference["seeds"] != candidate["seeds"] or reference["output"] != candidate["output"]:
        raise ValueError("seeds or output requirements differ")
    for key in ("engine", "tool_digest", "versions"):
        if reference["environment"]["environment"][key] != candidate["environment"]["environment"][key]:
            raise ValueError(f"implementation environment differs: {key}")
    a = read_json(Path(reference["root"]) / "requests.json")
    b = read_json(Path(candidate["root"]) / "requests.json")
    for seed in a:
        # Precision is the experimental variable. All financial inputs and required
        # outputs must match; no setting is silently dropped by the ORE adapter.
        left, right = a[seed], b[seed]
        left["config"].pop("precision")
        right["config"].pop("precision")
        if left != right:
            raise ValueError("API workloads differ")
    refs = {s["seed"]: s for s in reference["samples"] if s["phase"] == "repeat"}
    checks = []
    for sample in candidate["samples"]:
        def path(point, filename):
            result = (Path(point["root"]) / filename).resolve()
            if not result.is_relative_to(Path(point["root"]).resolve()):
                raise ValueError("snapshot escapes point directory")
            return result
        checks.append(named_errors(path(reference, refs[sample["seed"]]["snapshot"]),
                                   path(candidate, sample["snapshot"]), rtol, atol))
    return checks


def write_report(directory, points, rtol, atol):
    references = {(p["campaign"], p["paths"]): p for p in points if p["implementation"] == "ore" and p["status"] == "ok"}
    rows = []
    for point in points:
        row = {k: point[k] for k in ("campaign", "implementation", "backend", "label", "paths", "status")}
        row.update(accuracy="unmeasured", ore_speedup=None, reason=point.get("reason"))
        if point["status"] == "ok":
            samples = point["samples"]
            first = [s["seconds"] for s in samples if s["phase"] == "first"]
            warm = [s["seconds"] for s in samples if s["phase"] == "repeat"]
            expected = len(point["seeds"]) * point["repeats"]
            seed_counts = Counter(s["seed"] for s in samples if s["phase"] == "repeat")
            if (len(first) != 1 or len(warm) != expected
                    or seed_counts != {seed: point["repeats"] for seed in point["seeds"]}
                    or len({s["run_id"] for s in samples}) != len(samples)
                    or any(not math.isfinite(s["seconds"]) or s["seconds"] <= 0 for s in samples)):
                row.update(status="failed", reason="incomplete measurements or reused run ids")
            else:
                row.update(first_seconds=first[0], repeat_median_seconds=statistics.median(warm),
                           repeat_min_seconds=min(warm), repeat_max_seconds=max(warm),
                           warm_compiles=sum(s["compiles"] or 0 for s in samples if s["phase"] == "repeat"))
                reference = references.get((point["campaign"], point["paths"]))
                if reference is not None:
                    try:
                        row["checks"] = compare(reference, point, rtol, atol)
                        row["accuracy"] = "PASS" if all(r["status"] == "PASS" for checks in row["checks"] for r in checks.values()) else "FAIL"
                        if row["accuracy"] == "PASS":
                            ore_times = [s["seconds"] for s in reference["samples"] if s["phase"] == "repeat"]
                            row["ore_speedup"] = statistics.median(ore_times) / row["repeat_median_seconds"]
                    except (ValueError, KeyError, OSError, zipfile.BadZipFile, EOFError) as exc:
                        row["reason"] = str(exc)
        rows.append(row)
    data = {"schema_version": 2, "transport": "api", "tolerances": {"rtol": rtol, "atol": atol},
            "points": points, "comparisons": rows}
    atomic_json(directory / "comparison.json", data)
    lines = ["# API comparison: ORE and JAX", "",
             "Client wall time from POST /runs through completed numerical results, including P&L artifact downloads.",
             "Both use the production HTTP API and durable queue. ORE uses a benchmark-only CPU worker and RiskStatistics on its trade P&L. Each scenario is valued once; ORE uses disjoint batches of at most 4096 scenarios to fit its historical calendar, with all batch setup inside the timer.",
             "Service/hardware discovery and common synthetic-history preparation are outside the timer. First-job time includes lazy initialization and compilation; repeats are new jobs on the same worker.",
             "Synthetic histories express the existing campaign's Sobol shocks; these are not observed market histories. JAX also computes its usual uncertainty diagnostics; the ORE adapter returns VaR/ES without those diagnostics.",
             "Speedup = median ORE API time / median candidate API time at the same scenario count; withheld when numerical outputs fail or cannot be compared. This is not an equal-accuracy or ORE GPU comparison.", "",
             "| Workload | Implementation / device / format | Scenarios | Status | Accuracy vs ORE | First job s | Repeat median s | ORE speedup |",
             "|---|---|---:|---|---|---:|---:|---:|"]
    for row in rows:
        def number(key):
            value = row.get(key)
            return f"{value:.4f}" if value is not None else "—"
        lines.append(f"| {row['campaign']} | {row['implementation']} / {row['backend']} / {row['label']} | {row['paths']} | {row['status']} | {row['accuracy']} | {number('first_seconds')} | {number('repeat_median_seconds')} | {number('ore_speedup')} |")
        if row.get("reason"):
            lines += ["", f"{row['campaign']} / {row['implementation']} / {row['label']}: {row['reason']}", ""]
    lines += ["", "Raw request timings, job ids, compilation counts, hardware/library metadata, numerical checks and snapshot paths: [comparison.json](comparison.json).", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return data


def run(args):
    if not args.merge:
        collect(args)
    points = load_points(args.merge or [args.out])
    args.out.mkdir(parents=True, exist_ok=True)
    data = write_report(args.out, points, args.rtol, args.atol)
    print(f"API comparison report: {args.out / 'report.md'}", flush=True)
    # A measured low-precision accuracy miss is a research result, not a broken
    # collection. Its speedup stays withheld; execution/unmeasured checks fail.
    return int(any(r["status"] == "failed" or (r["status"] == "ok" and r["accuracy"] == "unmeasured")
                   for r in data["comparisons"]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", type=Path)
    parser.add_argument("--worker")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.serve:
        server(args.serve)
    elif args.worker:
        raise SystemExit(worker(json.loads(args.worker), args.out.resolve()))
    else:
        parser.error("use python -m benchmarks.comparison")
