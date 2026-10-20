"""Per-instrument CPU/GPU scaling study, numerical comparisons, plots and Python traces.

Run from the repository root in a Linux/WSL GPU environment:
    python -m pip install -e ".[gpu,profiling,benchmark]"
    python demos/demo_gpu_scaling.py --preset quick
    python demos/demo_gpu_scaling.py --preset standard

Default: standard grids, five warm measurements, CPU then GPU in isolated processes.
Open the generated report.md and scaling.pdf; view traces with XProf.
See docs/concepts/gpu-scaling.md for methodology, controls and interpretation.
Inputs, point construction, engine calls, timing and trace boundaries are in main().
"""
import argparse
import dataclasses
import gc
import hashlib
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from demos.benchmark_reporting import (
        atomic_json, launch_worker, numeric_arrays, read_json, trace_summary, write_report,
    )
else:
    from benchmark_reporting import atomic_json, launch_worker, numeric_arrays, read_json, trace_summary, write_report

INSTRUMENTS = ("swap", "european_bachelier", "european_jamshidian", "bermudan", "american", "bill", "note")
STAGES = ("pricing", "workflow", "greeks", "market_risk", "simulation", "mixed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=("quick", "standard"), default="standard")
    parser.add_argument("--devices", nargs="+", choices=("cpu", "gpu"), default=["cpu", "gpu"])
    parser.add_argument("--instruments", nargs="+", choices=INSTRUMENTS, default=list(INSTRUMENTS))
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--paths", nargs="+", type=int)
    parser.add_argument("--trades", nargs="+", type=int)
    parser.add_argument("--risk-scenarios", nargs="+", type=int)
    parser.add_argument("--risk-batches", nargs="+", type=int)
    parser.add_argument("--layouts", nargs="+", choices=("shared", "varied"), default=["shared", "varied"])
    parser.add_argument("--greeks-methods", nargs="+", choices=("AD", "Bump"), default=["AD"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--n-per-std", type=int)
    parser.add_argument("--exercise-steps", type=int)
    parser.add_argument("--traces", choices=("none", "selected", "all"), default="selected",
                        help="selected: slowest measured point per stage in each GPU worker")
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--timeout", type=int, default=3600, help="seconds per isolated worker")
    parser.add_argument("--no-plots", action="store_true", help="omit PDF/PNG plots; matplotlib not needed")
    parser.add_argument("--output", default=".profile-out-benchmark")
    parser.add_argument("--resume", type=Path, help="reuse a run's settings and skip completed workers")
    parser.add_argument("--report-only", type=Path, help="regenerate a run's report without engine work")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=("cpu", "gpu"), help=argparse.SUPPRESS)
    parser.add_argument("--group", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    # The coordinator imports no JAX. Backend isolation prevents silent CPU fallback,
    # shares no compiled executables between devices, and bounds cache growth per product.
    if not args.worker:
        if args.report_only:
            failed = write_report(args.report_only.resolve(), plots=not args.no_plots)
            return int(failed)
        if args.resume:
            run_dir = args.resume.resolve()
            settings = read_json(run_dir / "settings.json")
        else:
            quick = args.preset == "quick"
            defaults = {"paths": [64, 256] if quick else [1024, 4096, 16384],
                        "trades": [1, 2] if quick else [1, 8, 32],
                        "risk_scenarios": [256, 512] if quick else [1024, 4096, 16384],
                        "risk_batches": [128, 256] if quick else [256, 1024, 4096],
                        "n_per_std": 8 if quick else 30, "exercise_steps": 2 if quick else 24}
            for name, default in defaults.items():
                if getattr(args, name) is None:
                    setattr(args, name, default)
            for name in ("paths", "trades", "risk_scenarios", "risk_batches"):
                values = sorted(set(getattr(args, name)))
                if not values or min(values) < 1:
                    parser.error(f"--{name.replace('_', '-')} needs positive integers")
                setattr(args, name, values)
            if any(n & (n - 1) for n in args.paths):
                parser.error("--paths must be powers of two")
            if min(args.risk_scenarios) < 256:
                parser.error("--risk-scenarios must be at least 256 (keep at least two observations in the 99% ES tail)")
            if min(args.repeats, args.warmups, args.n_per_std, args.exercise_steps, args.timeout) < 1:
                parser.error("repeats, warmups, grid settings and timeout must be positive")
            if not (0 <= args.rtol < float("inf") and 0 < args.atol < float("inf")):
                parser.error("--rtol must be finite and nonnegative; --atol finite and positive")
            settings = {k: v for k, v in vars(args).items()
                        if k not in ("worker", "backend", "group", "resume", "report_only")}
            run_dir = Path(args.output).resolve() / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%fZ}-pid-{os.getpid()}"
            run_dir.mkdir(parents=True)
            atomic_json(run_dir / "settings.json", settings)
        if "gpu" in settings["devices"] and sys.platform == "win32":
            parser.exit(1, "Run the CPU/GPU study under Linux/WSL2. Native Windows checks can use --devices cpu.\n")
        if not settings["no_plots"]:
            try:
                import matplotlib  # noqa: F401 -- fail before a long study if the plot extra is missing
            except ImportError:
                parser.exit(1, 'Install the benchmark extra: python -m pip install -e ".[benchmark]"; '
                               'or use --no-plots.\n')
        groups = list(settings["instruments"]) if any(s in settings["stages"] for s in STAGES[:4]) else []
        if "simulation" in settings["stages"]:
            groups += ["simulation_HullWhite", "simulation_LGM"]
        if "mixed" in settings["stages"]:
            groups += ["mixed"]
        print(f"Results: {run_dir}\nRun workers sequentially; keep other GPU/CPU workloads quiet.", flush=True)
        failure_file = run_dir / "worker_failures.json"
        previous_failures = read_json(failure_file) if failure_file.exists() else []
        retry = {(item["backend"], item["group"]) for item in previous_failures}
        failures = []
        atomic_json(failure_file, failures)
        for backend in dict.fromkeys(settings["devices"]):
            for group in groups:
                destination = run_dir / backend / group
                completed = destination / "results.json"
                if (args.resume and completed.exists() and read_json(completed).get("complete")
                        and (backend, group) not in retry):
                    print(f"Keeping completed {backend}/{group}", flush=True)
                    continue
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(run_dir),
                           "--backend", backend, "--group", group]
                print(f"\n===== {backend}/{group} =====", flush=True)
                code = launch_worker(command, destination, settings["timeout"])
                if code:
                    failures.append({"backend": backend, "group": group, "exit_code": code})
                atomic_json(run_dir / "worker_failures.json", failures)
        failed = write_report(run_dir, plots=not settings["no_plots"])
        print(f"\nRead {run_dir / 'report.md'}", flush=True)
        print(f"Resume interrupted workers: python demos/demo_gpu_scaling.py --resume \"{run_dir}\"")
        return int(failed or bool(failures))

    # Everything below executes in a fresh worker for ONE backend and product.
    settings = read_json(args.worker / "settings.json")
    backend, group = args.backend, args.group
    destination = args.worker / backend / group
    destination.mkdir(parents=True, exist_ok=True)
    os.environ["JAX_PLATFORMS"] = "cuda" if backend == "gpu" else "cpu"
    os.environ["JAX_RISK_SCENARIO_DEVICES"] = "1"  # compare one CPU client with one GPU
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    flags = os.environ.get("XLA_FLAGS", "")
    if "xla_gpu_exclude_nondeterministic_ops" not in flags and "xla_gpu_deterministic_ops" not in flags:
        os.environ["XLA_FLAGS"] = (flags + " --xla_gpu_exclude_nondeterministic_ops=true").strip()
    import jax
    import numpy as np
    import ORE

    if __package__:
        from demos.demo_scenarios import EVAL_DATE, demo_market, demo_simulation
    else:
        from demo_scenarios import EVAL_DATE, demo_market, demo_simulation
    from engine.instruments.american_swaption import AmericanSwaptionConfig
    from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
    from engine.instruments.european_swaption import SwaptionConfig
    from engine.instruments.swap import SwapConfig
    from engine.instruments.treasury import BondConfig, CouponPeriod
    from engine.market_simulation.config import build_cross_asset_model, simulate
    from engine.pricing.config import JamshidianEngineConfig, LgmSwaptionEngineConfig, PricingConfig
    from engine.pricing.cube import value_paths
    from engine.risk.greeks.ad import portfolio_greeks
    from engine.risk.greeks.bump import portfolio_sensitivities
    from engine.risk.market import MarketRiskRequest, RateRiskFactors, monte_carlo_scenarios, run_market_risk
    from engine.risk.greeks.scenario_pnl import scenario_batch_size
    from engine.run import GreeksConfig, PortfolioRequest, RunConfig, price_portfolio

    devices = jax.local_devices()
    if devices[0].platform != backend:
        raise RuntimeError(f"Requested {backend}, got {devices}")
    # First-use measurements start from empty in-memory worker caches; later points
    # intentionally share programs. No persistent disk compilation cache masks compilation.
    jax.config.update("jax_enable_compilation_cache", False)
    compile_events = []

    def on_compile(event, duration, **_):
        if event == "/jax/core/compile/backend_compile_duration":
            compile_events.append(duration)

    jax.monitoring.register_event_duration_secs_listener(on_compile)
    engine_source = hashlib.sha256()
    repository = Path(__file__).resolve().parents[1]
    for source in sorted((repository / "engine").rglob("*.py")):
        engine_source.update(source.relative_to(repository).as_posix().encode())
        engine_source.update(source.read_bytes())
    metadata = {"backend": backend, "group": group, "device": devices[0].device_kind,
                "jax": jax.__version__, "numpy": np.__version__, "python": platform.python_version(),
                "platform": platform.platform(), "settings": settings,
                "xla_flags": os.environ.get("XLA_FLAGS", ""), "precision": "float64",
                "cpu_logical_cores": os.cpu_count(), "engine_source_sha256": engine_source.hexdigest(),
                "demo_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    document = {"metadata": metadata, "complete": False, "rows": []}
    atomic_json(destination / "results.json", document)
    print(f"Device: {devices[0]} ({devices[0].device_kind}); float64", flush=True)

    # 1. Visible market, schedules and all seven instrument/engine cases. Every base
    # contract remains alive through the final simulation date (two years).
    market = demo_market(("USD", "EUR"))
    dates = tuple(EVAL_DATE + ORE.Period(m, ORE.Months) for m in (3, 6, 9, 12, 18, 24))
    grid = LgmSwaptionEngineConfig(n_per_std=settings["n_per_std"], std_devs=5.0,
                                  exercise_time_steps_per_year=settings["exercise_steps"])
    pricing = PricingConfig(bermudan=grid, american=grid)
    jamshidian = dataclasses.replace(pricing, european="Jamshidian", jamshidian=JamshidianEngineConfig(0.03, 0.01))
    swap = SwapConfig(notional=1_000_000.0, fixed_rate=0.04, payer=True, swap_tenor="5Y",
                      evaluation_date=EVAL_DATE, trade_id="swap")
    european = SwaptionConfig(notional=1_000_000.0, fixed_rate=0.04, payer=True, swap_tenor="3Y",
                              forward_start=ORE.Period(3, ORE.Years), evaluation_date=EVAL_DATE, trade_id="european")
    bermudan = BermudanSwaptionConfig(notional=1_000_000.0, fixed_rate=0.04, payer=True, swap_tenor="5Y",
                                     exercise_dates=[EVAL_DATE + ORE.Period(y, ORE.Years) for y in (1, 2, 3)],
                                     evaluation_date=EVAL_DATE, trade_id="bermudan")
    american = AmericanSwaptionConfig(notional=1_000_000.0, fixed_rate=0.04, payer=True, swap_tenor="5Y",
                                     first_exercise_date=EVAL_DATE + ORE.Period(1, ORE.Years),
                                     last_exercise_date=EVAL_DATE + ORE.Period(2, ORE.Years),
                                     evaluation_date=EVAL_DATE, trade_id="american")
    bill = BondConfig(face_amount=1_000_000.0, maturity_date=EVAL_DATE + ORE.Period(3, ORE.Years),
                      evaluation_date=EVAL_DATE, trade_id="bill")
    note_dates = tuple(EVAL_DATE + ORE.Period(m, ORE.Months) for m in range(0, 61, 6))
    note = BondConfig(face_amount=1_000_000.0, maturity_date=note_dates[-1], coupon_rate=0.04,
                      coupon_schedule=tuple(CouponPeriod(a, b) for a, b in zip(note_dates, note_dates[1:])),
                      evaluation_date=EVAL_DATE, trade_id="note")
    templates = dict(zip(INSTRUMENTS, (swap, european, european, bermudan, american, bill, note)))

    # 2. A one-dimension-at-a-time study, not an enormous Cartesian product.
    points = []
    base_paths, base_risk, base_batch = settings["paths"][0], settings["risk_scenarios"][0], settings["risk_batches"][0]
    if group.startswith("simulation_"):
        points.append(dict(stage="calibration", axis="paths", x=base_paths, paths=base_paths, trades=0,
                           layout="shared", model=group.removeprefix("simulation_"), batch=0, scenarios=0, method="AD"))
        for paths in settings["paths"]:
            points.append(dict(stage="simulation", axis="paths", x=paths, paths=paths, trades=0,
                               layout="shared", model=group.removeprefix("simulation_"), batch=0, scenarios=0, method="AD"))
    elif group == "mixed":
        points.append(dict(stage="mixed", axis="paths", x=settings["paths"][-1], paths=settings["paths"][-1],
                           trades=6, layout="shared", model="HullWhite", batch=0, scenarios=0, method="AD"))
    else:
        for stage in (s for s in ("pricing", "workflow") if s in settings["stages"]):
            for paths in settings["paths"]:
                points.append(dict(stage=stage, axis="paths", x=paths, paths=paths, trades=1, layout="shared"))
            for layout in dict.fromkeys(settings["layouts"]):
                for count in settings["trades"]:
                    points.append(dict(stage=stage, axis="trades", x=count, paths=base_paths, trades=count, layout=layout))
        if "greeks" in settings["stages"]:
            for method in dict.fromkeys(settings["greeks_methods"]):
                for layout in dict.fromkeys(settings["layouts"]):
                    for count in settings["trades"]:
                        points.append(dict(stage="greeks", axis="trades", x=count, paths=base_paths,
                                           trades=count, layout=layout, method=method))
        if "market_risk" in settings["stages"]:
            for scenarios in settings["risk_scenarios"]:
                points.append(dict(stage="market_risk", axis="scenarios", x=scenarios, paths=base_paths,
                                   trades=1, layout="shared", scenarios=scenarios, batch=base_batch))
            for batch in settings["risk_batches"]:
                points.append(dict(stage="market_risk", axis="batch", x=batch, paths=base_paths,
                                   trades=1, layout="shared", scenarios=settings["risk_scenarios"][-1], batch=batch))
            for layout in dict.fromkeys(settings["layouts"]):
                for count in settings["trades"]:
                    points.append(dict(stage="market_risk", axis="trades", x=count, paths=base_paths,
                                       trades=count, layout=layout, scenarios=base_risk, batch=base_batch))
    for point in points:
        for key, default in dict(model="HullWhite", method="AD", scenarios=0, batch=0).items():
            point.setdefault(key, default)
        point["id"] = (f"{point['stage']}-{point['axis']}{point['x']}-{point['layout']}-{point['method']}"
                       f"-p{point['paths']}-t{point['trades']}-s{point['scenarios']}-b{point['batch']}")

    # Prepared scenarios/models/shocks are outside pricing-only and risk timers.
    scenario_cache, model_cache, shocks_cache = {}, {}, {}
    operations = {}
    any_failure = False
    try:
        for point in points:
            print(f"\n{backend}/{group}: {point['id']}", flush=True)
            row = {**point, "status": "failed", "backend": backend, "group": group}
            try:
                # Construct actual contracts here. Shared layouts change values; varied
                # layouts also change coupon counts, maturities and exercise counts.
                trades = []
                selected = ([templates[k] for k in ("swap", "european_bachelier", "bermudan", "american", "bill", "note")]
                            if group == "mixed" else [templates[group]] if not group.startswith("simulation_") else [])
                for i in range(point["trades"]):
                    template = selected[i % len(selected)]
                    changes = {"trade_id": f"{template.trade_id}-{i}"}
                    if isinstance(template, BondConfig):
                        changes["face_amount"] = template.face_amount * (1 + 0.05 * (i % 5))
                        if point["layout"] == "varied":
                            years = 3 + i % 3
                            changes["maturity_date"] = EVAL_DATE + ORE.Period(years, ORE.Years)
                            if template.coupon_schedule:
                                schedule = [EVAL_DATE + ORE.Period(m, ORE.Months) for m in range(0, years * 12 + 1, 6)]
                                changes["coupon_schedule"] = tuple(CouponPeriod(a, b) for a, b in zip(schedule, schedule[1:]))
                    else:
                        changes.update(notional=template.notional * (1 + 0.05 * (i % 5)),
                                       fixed_rate=template.fixed_rate + 0.0001 * (i % 3))
                        if point["layout"] == "varied":
                            years = (3 if isinstance(template, SwaptionConfig) else 5) + i % 3
                            changes["maturity_date"] = template.effective_date + ORE.Period(years, ORE.Years)
                            if isinstance(template, BermudanSwaptionConfig):
                                changes["exercise_dates"] = [EVAL_DATE + ORE.Period(y, ORE.Years) for y in range(1, 3 + i % 2)]
                            elif isinstance(template, AmericanSwaptionConfig):
                                changes["last_exercise_date"] = EVAL_DATE + ORE.Period(24 + 6 * (i % 2), ORE.Months)
                    trades.append(dataclasses.replace(template, **changes))
                engine = jamshidian if group == "european_jamshidian" else pricing
                simulation = demo_simulation(point["model"], samples=point["paths"], dates=dates)
                simulation = dataclasses.replace(simulation, ir={
                    c: dataclasses.replace(ir, calibration_expiries=("1Y", "2Y"), calibration_terms=("4Y", "3Y"))
                    for c, ir in simulation.ir.items()})
                observable = lambda value: value
                if point["stage"] in ("simulation", "pricing"):
                    if point["model"] not in model_cache:
                        model_cache[point["model"]] = build_cross_asset_model(market, simulation)
                    model = model_cache[point["model"]]
                if point["stage"] == "calibration":
                    operation = lambda simulation=simulation: build_cross_asset_model(market, simulation)
                    observable = lambda value: [component.sigma.values for component in value.ir]
                    units, work = "currency calibrations/s", len(simulation.ir)
                elif point["stage"] == "simulation":
                    operation = lambda simulation=simulation, model=model: simulate(market, simulation, model)
                    observable = lambda value: value.path_arrays()
                    units, work = "path-dates/s", point["paths"] * len(dates)
                elif point["stage"] == "pricing":
                    cache_key = (point["model"], point["paths"])
                    if cache_key not in scenario_cache:
                        scenario_cache[cache_key] = simulate(market, simulation, model)
                    scenarios = scenario_cache[cache_key]
                    operation = lambda trades=trades, scenarios=scenarios, engine=engine: value_paths(trades, market, scenarios, "USD", engine)
                    units, work = "trade-path-dates/s", len(trades) * point["paths"] * len(dates)
                elif point["stage"] in ("workflow", "mixed"):
                    request = PortfolioRequest(market=market, trades=trades, compute_greeks=point["stage"] == "mixed",
                                               config=RunConfig(simulation=simulation, pricing=engine,
                                                                greeks=GreeksConfig(method="AD")))
                    operation = lambda request=request: price_portfolio(request)
                    observable = lambda value: {"today": value.base_npv_per_trade, "cube": value.npv_cube,
                                                "exposure": value.exposure, "trades": value.trade_exposures,
                                                "greeks": value.greeks, "exercise": value.exercised}
                    units, work = "trade-path-dates/s", len(trades) * point["paths"] * len(dates)
                elif point["stage"] == "greeks":
                    compute = portfolio_greeks if point["method"] == "AD" else portfolio_sensitivities
                    operation = lambda trades=trades, engine=engine, compute=compute: compute(trades, market, "USD", engine)
                    units, work = "trade Greeks/s", len(trades)
                else:
                    count = point["scenarios"]
                    if count not in shocks_cache:
                        factors = RateRiskFactors.from_market(market)
                        pillars = np.arange(factors.size)
                        covariance = np.exp(-np.abs(pillars[:, None] - pillars[None, :]) / 3.0) * 0.0008**2 * 10
                        shocks_cache[count] = monte_carlo_scenarios(factors, covariance, horizon_days=10,
                                                                  num_scenarios=count, seed=11)
                    request = MarketRiskRequest(trades=trades, market=market, scenarios=shocks_cache[count],
                                                pricing=engine, batch_size=point["batch"], quantiles=(0.95, 0.99))
                    row["effective_batches"] = [min(point["scenarios"], scenario_batch_size(t, engine, point["batch"], 8))
                                                for t in trades]
                    operation = lambda request=request: run_market_risk(request)
                    observable = lambda value: {"base": value.base_npv_per_trade, "pnl": value.pnl,
                                                "portfolio_pnl": value.portfolio_pnl, "risk": value.risk}
                    units, work = "trade-scenarios/s", len(trades) * point["scenarios"]

                # 3. Synchronize BEFORE and AFTER every call. Warmups, first-use timing,
                # warm samples and compile counts are explicit; no profiler is active.
                durations, misses = [], []
                for sample in range(settings["warmups"] + settings["repeats"]):
                    jax.block_until_ready(jax.live_arrays())
                    before = len(compile_events)
                    started = time.perf_counter()
                    result = operation()
                    jax.block_until_ready(jax.live_arrays())
                    duration = time.perf_counter() - started
                    if sample == 0:
                        row["first_call_seconds"] = duration
                        row["first_call_compiles"] = len(compile_events) - before
                    if sample >= settings["warmups"]:
                        durations.append(duration)
                        misses.append(len(compile_events) - before)
                # Full numerical snapshots are copied/saved outside the timing window.
                values = numeric_arrays(observable(result))
                if not values or any(not np.isfinite(a).all() for a in values.values()):
                    raise ValueError("Missing or non-finite benchmark outputs")
                snapshot = destination / "snapshots" / f"{point['id']}.npz"
                snapshot.parent.mkdir(exist_ok=True)
                np.savez_compressed(snapshot, **values)
                median = statistics.median(durations)
                row.update(status="ok", seconds=durations, median_seconds=median, min_seconds=min(durations),
                           max_seconds=max(durations), warm_compiles=misses,
                           spread_fraction=(max(durations) - min(durations)) / median,
                           throughput=work / median, work_items=work, units=units,
                           snapshot=str(snapshot.relative_to(args.worker)),
                           cumulative_device_memory=devices[0].memory_stats())
                # Defaults in each lambda bind this point's inputs for later replay.
                operations[point["id"]] = operation
                print(f"  first {row['first_call_seconds']:.4f}s; warm median {median:.4f}s; "
                      f"{row['throughput']:,.0f} {units}; warm compiles {sum(misses)}", flush=True)
            except Exception as error:
                any_failure = True
                row["error"] = f"{type(error).__name__}: {error}"
                print(f"  FAILED: {row['error']}", flush=True)
            document["rows"].append(row)
            atomic_json(destination / "results.json", document)

        # 4. Only AFTER untraced measurements: replay selected points with Python ON.
        selected_rows = []
        if backend == "gpu" and settings["traces"] != "none":
            successful = [r for r in document["rows"] if r["status"] == "ok"]
            if settings["traces"] == "all":
                selected_rows = successful
            else:
                for stage in dict.fromkeys(r["stage"] for r in successful):
                    selected_rows.append(max((r for r in successful if r["stage"] == stage), key=lambda r: r["median_seconds"]))
        options = jax.profiler.ProfileOptions()
        options.python_tracer_level = 1
        options.host_tracer_level = 2
        options.raise_error_on_start_failure = True
        for row in selected_rows:
            trace_dir = destination / "traces" / row["id"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            trace_dir.mkdir(parents=True)
            print(f"Tracing {row['id']} (Python enabled)", flush=True)
            try:
                operation = operations[row["id"]]
                result = operation()  # rewarm after other shapes; excluded from trace/timing
                jax.block_until_ready(jax.live_arrays())
                jax.profiler.start_trace(str(trace_dir), profiler_options=options)
                try:
                    before = len(compile_events)
                    started = time.perf_counter()
                    with jax.profiler.TraceAnnotation("benchmark", point=row["id"]):
                        result = operation()
                        jax.block_until_ready(jax.live_arrays())
                    with jax.profiler.TraceAnnotation("capture_end"):
                        pass
                    traced_seconds = time.perf_counter() - started
                finally:
                    jax.profiler.stop_trace()
                row["trace"] = trace_summary(trace_dir, traced_seconds, row["median_seconds"], len(compile_events) - before)
                row["trace"]["xplane"] = Path(row["trace"]["xplane"]).relative_to(args.worker).as_posix()
            except Exception as error:
                any_failure = True
                row["trace_error"] = f"{type(error).__name__}: {error}"
                print(f"  TRACE FAILED: {error}", flush=True)
            atomic_json(destination / "results.json", document)
        document["complete"] = not any_failure
        atomic_json(destination / "results.json", document)
    finally:
        jax.monitoring.unregister_event_duration_listener(on_compile)
        operations.clear()
        scenario_cache.clear()
        jax.clear_caches()
        gc.collect()
    return int(any_failure)


if __name__ == "__main__":
    sys.exit(main())
