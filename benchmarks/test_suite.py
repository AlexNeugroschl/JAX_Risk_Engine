"""Run every collected test in bounded fresh processes, with retained JUnit evidence (I-27).

This is the full suite, partitioned by collected node IDs; no tests are silently dropped.
"""
import argparse
import hashlib
import math
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

from demos.benchmark_reporting import atomic_json, launch_worker


def junit_counts(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    if not suites:
        raise ValueError("JUnit report contains no test suites")
    return {key: sum(int(s.get(key, "0")) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=1200.)
    parser.add_argument("--markers", help="optional pytest marker expression; omit for the full suite")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("batch size and finite timeout must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        parser.error("output directory must be empty")
    select = ["-m", args.markers] if args.markers else []
    collection = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests/", *select],
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    (args.out / "collection.log").write_text(collection.stdout + collection.stderr, encoding="utf-8")
    nodes = [line.strip() for line in collection.stdout.splitlines() if re.match(r"^tests[/\\].+::", line)]
    if collection.returncode or not nodes or len(nodes) != len(set(nodes)):
        raise ValueError("test collection failed, was empty or contained duplicate node IDs; see collection.log")
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for folder in ("engine", "tests", "benchmarks", "demos"):
        for path in sorted((root / folder).rglob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    batches = [nodes[i:i + args.batch_size] for i in range(0, len(nodes), args.batch_size)]
    data = {"schema_version": 1, "source_digest": digest.hexdigest(), "collected": len(nodes),
            "markers": args.markers, "batch_size": args.batch_size, "batches": [], "complete": False}
    atomic_json(args.out / "manifest.json", {**data, "nodes": nodes})
    for i, batch in enumerate(batches):
        destination = args.out / f"batch-{i:04d}"
        destination.mkdir()
        junit = destination.resolve() / "junit.xml"
        command = [sys.executable, "-m", "pytest", "-q", *batch, f"--junitxml={junit}"]
        print(f"Batch {i + 1}/{len(batches)}: {len(batch)} tests", flush=True)
        code = launch_worker(command, destination, args.timeout)
        row = {"batch": i, "nodes": batch, "exit_code": code, "passes": False}
        try:
            counts = junit_counts(junit)
            row["counts"] = counts
            log = (destination / "worker.log").read_text(encoding="utf-8")
            summary = bool(re.search(r"\b\d+ (?:passed|skipped|xfailed)\b.*in [\d.]+s", log))
            row["passes"] = (code == 0 and counts["tests"] == len(batch)
                             and not counts["errors"] and not counts["failures"] and summary)
        except (OSError, ValueError, ET.ParseError) as exc:
            row["reason"] = str(exc)
        data["batches"].append(row)
        data["complete"] = len(data["batches"]) == len(batches) and all(r["passes"] for r in data["batches"])
        atomic_json(args.out / "suite.json", data)
    print(f"Collected {len(nodes)}; complete={data['complete']}; "
          f"failed batches={sum(not r['passes'] for r in data['batches'])}", flush=True)
    return 0 if data["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
