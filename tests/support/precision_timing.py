"""
The timing harness: wall time per figure at equal accuracy (F-07's research result; I-61's
scaling; docs/planning/details/precision.md §19), on whatever devices the process has:

    python -m tests.support.precision_timing CAMPAIGN [--paths N ...] [--seeds K] [--cases TEXT ...]
        [--budget-divisors D ...] [--out DIR]

`CAMPAIGN` is a campaign of record (`tests/support/precision_campaigns.py`); its cases are kept
whose label contains any `--cases` text (by default `throughout:`, every stage in one format),
and float64 is always measured, the reference. Each case at each path count runs in a process of
its own (`--child`), so its first run compiles everything it uses (the persistent cache off) and
its peak device memory is its own:

  * every seed's run is the whole job (its scenarios drawn, then `campaign.run`: host
    preparation, the calibrations and every fixed float64 stage, the simulation, pricing, a paired
    sample, the reductions), the clock stopped once its figures are on the host
    (`campaign.figures`, each array transferred);
  * the first seed's run is the case's **cold** run, its XLA compile seconds stated apart
    (`backend_compile_duration`); the other seeds' runs are **warm** (the same shapes, nothing
    compiled), their median the case's time at that path count;
  * the device's peak allocation, the grid engine's induction budget (`--budget-divisors D`: the
    budget an eighth or `1/D` of the accelerator's allocator limit, the tradeoff of §17), and how
    each compute format's products ran on the device (`product_execution`: native or emulated);
  * each seed's figures, from which the parent measures accuracy.

**Accuracy** is measured, never assumed: per figure entry, the truth is the float64 figure's
mean over the seeds at the largest path count, and a case's error at `N` paths is the root mean
square over the seeds of its figure less the truth, so it holds the statistical error, the
precision error, their covariance and any bias as observed on the Sobol paths (RQMC: no rate is
assumed). A figure's error is the root mean square of its entries' (those float64 does not hold
fixed, `precision_harness.FIXED`) relative to its scale. **Equal accuracy**: for each float64
path count but the largest, the target is float64's error there; a case reaches it at the path
count where its error falls to the target, interpolated log-linearly between the two measured
counts that bracket it, and costs its warm time there, interpolated alike; a case whose error
stays above the target over the measured counts does not reach it (a valid result, a format
that does not pay). The speedup is float64's time over the case's. A job is as fast as its
slowest figure: the summary gives each case's speedup per figure and its worst.

Writes `DIR/<campaign>-<device>.json.gz` (every number) and `.md` (the summary), `DIR`
`compliance/precision-timing` by default. One host: every device the process has splits the
scenarios (`engine.market_simulation.sharding`); a pod slice's hosts run jobs as a worker set
(`jax-risk-worker --distributed`), whose queue records each job's start and end. Timings on CPU devices or a consumer GPU (whose float64
runs at 1/64 of its float32 rate) are local evidence, not the research result, which is measured
on Ironwood and H100 (F-07, I-61).
"""
import argparse
import dataclasses
import gzip
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

#: Where the timings of record are kept.
RECORD = Path(__file__).resolve().parents[2] / "compliance" / "precision-timing"
#: The cases timed by default: every stage in one format.
DEFAULT_CASES = ("throughout:",)
#: The reference case's label.
FLOAT64 = "float64"


@dataclasses.dataclass(frozen=True)
class CaseTiming:
    """One case at one path count, as its process measured it (see the module docstring)."""
    case: str
    paths: int
    budget_divisor: Optional[int]
    cold_seconds: float
    compile_seconds: float
    compiles: int
    warm_seconds: Tuple[float, ...]
    warm_compiles: int
    device_peak_bytes: Optional[int]
    induction_budget: int
    products: Dict[str, str]
    figures: Dict[str, List[List[float]]]   # figure -> per seed, its entries

    @property
    def warm(self) -> float:
        """The median warm wall time."""
        return statistics.median(self.warm_seconds)


@dataclasses.dataclass(frozen=True)
class EqualAccuracy:
    """One case's cost at float64's accuracy at `float64_paths`, for one figure (see the module
    docstring): None for `paths`, `seconds` and `speedup` where the case does not reach it."""
    case: str
    figure: str
    float64_paths: int
    target: float
    float64_seconds: float
    paths: Optional[float]
    seconds: Optional[float]

    @property
    def speedup(self) -> Optional[float]:
        return None if self.seconds is None else self.float64_seconds / self.seconds


# ---------------------------------------------------------------------------
# One case's process
# ---------------------------------------------------------------------------
def _child(campaign: str, label: str, paths: int, seeds: Sequence[int], budget_divisor: Optional[int]) -> dict:
    """Run one case at one path count over `seeds` in this process and return its `CaseTiming`
    as JSON-ready data."""
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    import jax
    import numpy as np

    jax.config.update("jax_enable_compilation_cache", False)
    from engine.precision import Precision, product_execution
    from engine.pricing import lgm_grid
    from tests.support.precision_campaigns import CAMPAIGNS

    campaign = CAMPAIGNS[campaign]()
    precision = Precision() if label == FLOAT64 else next(c.precision for c in campaign.cases if c.label == label)
    budget = lgm_grid.induction_memory_budget()
    if budget_divisor:
        limit = _device_limit()
        budget = max(1, limit // budget_divisor) if limit else budget
        lgm_grid.induction_memory_budget = lambda: budget
    compiled = []

    def listen(event, duration, **_):
        if event == "/jax/core/compile/backend_compile_duration":
            compiled.append(duration)

    jax.monitoring.register_event_duration_secs_listener(listen)
    seconds, compiles, figures = [], [], {}
    for seed in seeds:
        run_precision = precision
        if precision.rounding == "stochastic" or any(s.compute_rounding == "stochastic" for s in precision._stages()):
            run_precision = dataclasses.replace(precision, rounding_seed=seed)
        before = len(compiled)
        start = time.perf_counter()
        result = campaign.run(campaign.scenarios(paths, seed), run_precision)
        figured = {name: np.asarray(value, dtype=np.float64) for name, value in campaign.figures(result).items()}
        seconds.append(time.perf_counter() - start)
        compiles.append(compiled[before:])
        for name, value in figured.items():
            figures.setdefault(name, []).append(np.atleast_1d(value).tolist())
    stats = jax.devices()[0].memory_stats() if jax.devices()[0].platform != "cpu" else None
    names = sorted({s.compute for s in precision._stages()})
    return dataclasses.asdict(CaseTiming(
        case=label, paths=paths, budget_divisor=budget_divisor, cold_seconds=seconds[0],
        compile_seconds=float(sum(compiles[0])), compiles=len(compiles[0]), warm_seconds=tuple(seconds[1:]),
        warm_compiles=sum(len(c) for c in compiles[1:]),
        device_peak_bytes=None if not stats else int(stats.get("peak_bytes_in_use", 0)), induction_budget=budget,
        products={name: product_execution(name) for name in names}, figures=figures))


def _device_limit() -> Optional[int]:
    """The smallest accelerator allocator limit (`bytes_limit`), None on CPU devices."""
    import jax

    limits = [(d.memory_stats() or {}).get("bytes_limit") for d in jax.local_devices() if d.platform != "cpu"]
    limits = [int(v) for v in limits if v]
    return min(limits) if limits else None


def run_case(campaign: str, label: str, paths: int, seeds: Sequence[int], budget_divisor: Optional[int] = None,
             timeout: float = 6 * 3600) -> CaseTiming:
    """One case at one path count in a process of its own (see the module docstring)."""
    arguments = {"campaign": campaign, "label": label, "paths": paths, "seeds": list(seeds),
                 "budget_divisor": budget_divisor}
    completed = subprocess.run([sys.executable, "-m", "tests.support.precision_timing", "--child", json.dumps(arguments)],
                               cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True,
                               timeout=timeout, stdin=subprocess.DEVNULL)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} at {paths} paths failed:\n{completed.stderr[-4000:]}")
    data = json.loads(completed.stdout.strip().splitlines()[-1])
    data["warm_seconds"] = tuple(data["warm_seconds"])
    return CaseTiming(**data)


# ---------------------------------------------------------------------------
# Accuracy and equal accuracy
# ---------------------------------------------------------------------------
def figure_errors(timings: Sequence[CaseTiming], fixed: float) -> Dict[Tuple[str, int, str], float]:
    """`(case, paths, figure) -> error`: each figure's root mean square error over the seeds
    against the truth, root mean square over its varying entries, relative to its scale (see the
    module docstring). `fixed` is the share of the scale below which float64's spread holds an
    entry fixed."""
    import numpy as np

    timings = [t for t in timings if t.budget_divisor is None]
    reference = [t for t in timings if t.case == FLOAT64]
    largest = max(reference, key=lambda t: t.paths)
    errors = {}
    for figure, seeds in largest.figures.items():
        truth = np.mean(np.asarray(seeds), axis=0)
        scale = float(np.max(np.abs(truth)))
        spreads = np.stack([np.std(np.asarray(t.figures[figure]), axis=0, ddof=1) for t in reference])
        varying = np.max(spreads, axis=0) > fixed * scale
        if scale == 0.0 or not np.any(varying):
            continue
        for t in timings:
            values = np.asarray(t.figures[figure])
            rms = np.sqrt(np.mean((values - truth) ** 2, axis=0))
            errors[(t.case, t.paths, figure)] = float(np.sqrt(np.mean(rms[varying] ** 2)) / scale)
    return errors


def _log_interpolate(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    """`y` at `x` on the piecewise log-linear curve through the points `(xs, ys)`, `xs` increasing
    and `x` within them (a measured `x` its own `y`)."""
    if x in xs:
        return ys[list(xs).index(x)]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            w = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return math.exp((1 - w) * math.log(y0) + w * math.log(y1))
    raise ValueError(f"{x} outside [{xs[0]}, {xs[-1]}]")


def reached_at(errors: Sequence[Tuple[int, float]], target: float) -> Optional[float]:
    """The path count where an error measured at `(paths, error)` points, path counts increasing,
    first falls to `target`: log-linearly between the two points that bracket it, the first
    measured count if it is there already; None if it never does over the measured counts."""
    for i, (paths, error) in enumerate(errors):
        if error <= target:
            if i == 0:
                return float(paths)
            (p0, e0), (p1, e1) = errors[i - 1], (paths, error)
            if e0 <= 0 or e1 <= 0 or e0 == e1:
                return float(p1)
            w = (math.log(target) - math.log(e0)) / (math.log(e1) - math.log(e0))
            return math.exp((1 - w) * math.log(p0) + w * math.log(p1))
    return None


def equal_accuracy(timings: Sequence[CaseTiming], fixed: float) -> List[EqualAccuracy]:
    """Every case's cost at float64's accuracy, per figure and float64 path count but the
    largest (see the module docstring)."""
    errors = figure_errors(timings, fixed)
    plain = [t for t in timings if t.budget_divisor is None]
    by_case: Dict[str, List[CaseTiming]] = {}
    for t in sorted(plain, key=lambda t: t.paths):
        by_case.setdefault(t.case, []).append(t)
    reference = by_case[FLOAT64]
    rows = []
    for figure in sorted({f for _, _, f in errors}):
        for target_timing in reference[:-1]:
            target = errors[(FLOAT64, target_timing.paths, figure)]
            for case, runs in by_case.items():
                if case == FLOAT64:
                    continue
                measured = [(t.paths, errors[(case, t.paths, figure)]) for t in runs]
                paths = reached_at(measured, target)
                seconds = None
                if paths is not None:
                    seconds = _log_interpolate(paths, [t.paths for t in runs], [t.warm for t in runs])
                rows.append(EqualAccuracy(case, figure, target_timing.paths, target, target_timing.warm, paths, seconds))
    return rows


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------
def device_name() -> str:
    """The devices' kind and count, as a file name's part (`cpu-x1`, `nvidia-geforce-rtx-5060-x1`)."""
    import jax

    kind = jax.devices()[0].device_kind
    return f"{re.sub(r'[^a-z0-9]+', '-', kind.lower()).strip('-')}-x{jax.device_count()}"


def record(campaign: str, timings: Sequence[CaseTiming], fixed: float) -> dict:
    """Every number of a timing campaign, with the environment it ran in."""
    from tests.support.precision_harness import environment

    return {"campaign": campaign, "environment": environment(), "fixed": fixed,
            "timings": [dataclasses.asdict(t) for t in timings],
            "equal_accuracy": [dataclasses.asdict(r) for r in equal_accuracy(timings, fixed)]}


def summary(data: dict) -> str:
    """A markdown summary of a record: each case's times, then its speedup at equal accuracy
    per figure kind and its worst over the figures."""
    timings = [CaseTiming(**{**t, "warm_seconds": tuple(t["warm_seconds"])}) for t in data["timings"]]
    rows = [EqualAccuracy(**r) for r in data["equal_accuracy"]]
    env = data["environment"]
    lines = [f"# Timing: {data['campaign']}", "",
             f"Devices {env['devices']} ({env['backend']}), jax {env['jax']}, commit {env['commit'][:10]}"
             f"{' with changes' if env['changes'] == 'yes' else ''}, engine {env['engine']}, {env['measured']}.", "",
             "| Case | Paths | Budget | Cold s | Compile s | Warm s | Peak MB | Products |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for t in sorted(timings, key=lambda t: (t.case != FLOAT64, t.case, t.paths, t.budget_divisor or 0)):
        peak = "" if t.device_peak_bytes is None else f"{t.device_peak_bytes / 1e6:.0f}"
        budget = f"1/{t.budget_divisor}" if t.budget_divisor else "default"
        products = ", ".join(f"{k} {v}" for k, v in t.products.items())
        lines.append(f"| {t.case} | {t.paths} | {budget} | {t.cold_seconds:.2f} | {t.compile_seconds:.2f} | {t.warm:.3f} | "
                     f"{peak} | {products} |")
    lines += ["", "Speedup at float64's accuracy (float64's warm time over the case's, at the path count where the "
                  "case's error reaches float64's): the worst figure, and the figures not reached.", "",
              "| Case | float64 paths | Worst speedup | Worst figure | Not reached |", "|---|---:|---:|---|---:|"]
    keys = sorted({(r.case, r.float64_paths) for r in rows})
    for case, paths in keys:
        mine = [r for r in rows if (r.case, r.float64_paths) == (case, paths)]
        reached = [r for r in mine if r.speedup is not None]
        missed = len(mine) - len(reached)
        worst = min(reached, key=lambda r: r.speedup) if reached else None
        lines.append(f"| {case} | {paths} | {'' if worst is None else f'{worst.speedup:.2f}'} | "
                     f"{'' if worst is None else worst.figure} | {missed} of {len(mine)} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.support.precision_timing",
                                     description="Time a campaign's cases at equal accuracy (F-07).")
    parser.add_argument("campaign", nargs="?")
    parser.add_argument("--paths", type=int, nargs="+", help="path counts (default: the campaign's)")
    parser.add_argument("--seeds", type=int, help="the first K of the campaign's seeds (at least 3)")
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES),
                        help="keep the cases whose label contains any of these (default: throughout:)")
    parser.add_argument("--budget-divisors", type=int, nargs="+", default=[],
                        help="also time each case at the largest path count with the induction budget 1/D of "
                             "the accelerator's memory (the default is 1/8)")
    parser.add_argument("--out", type=Path, default=RECORD)
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        print(json.dumps(_child(**json.loads(args.child))), flush=True)
        return 0

    from tests.support.precision_campaigns import CAMPAIGNS
    from tests.support.precision_harness import FIXED

    campaign = CAMPAIGNS[args.campaign]()
    seeds = campaign.seeds[:args.seeds] if args.seeds else campaign.seeds
    if len(seeds) < 3:
        raise SystemExit("an error over the seeds needs at least three")
    labels = [FLOAT64] + [c.label for c in campaign.cases if any(text in c.label for text in args.cases)]
    paths = sorted(args.paths or campaign.paths)
    args.out.mkdir(parents=True, exist_ok=True)
    timings: List[CaseTiming] = []
    stem = f"{args.campaign}-{device_name()}"
    for count in paths:
        for label in labels:
            timings.append(run_case(args.campaign, label, count, seeds))
            print(f"{label} {count} paths: cold {timings[-1].cold_seconds:.1f}s, warm {timings[-1].warm:.3f}s",
                  flush=True)
            _write(args.out, stem, record(args.campaign, timings, FIXED))
    for divisor in args.budget_divisors:
        for label in labels:
            timings.append(run_case(args.campaign, label, paths[-1], seeds, divisor))
            _write(args.out, stem, record(args.campaign, timings, FIXED))
    return 0


def _write(directory: Path, stem: str, data: dict) -> None:
    (directory / f"{stem}.json.gz").write_bytes(
        gzip.compress(json.dumps(data, separators=(",", ":")).encode("utf-8"), compresslevel=9, mtime=0))
    (directory / f"{stem}.md").write_text(summary(data) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
