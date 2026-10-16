"""Trace the engine's Python calls, JAX dispatch/compilation and GPU kernels together.

From the repository root, in a Linux/WSL GPU environment:
    python demos/demo_profile_python.py
    xprof --port 8791 --logdir .profile-out-python

Larger synthetic workload, with an untraced warm repeat for comparison:
    python demos/demo_profile_python.py --preset representative --baseline

If the whole trace is too large:
    python demos/demo_profile_python.py --split
    python demos/demo_profile_python.py --phase greeks/trade4/AmericanSwaptionConfig

Use --cold to include compilation from scratch, or --device cpu on native Windows.
See docs/concepts/profiling-python.md for setup, scope and how to read the result.
The workload, warm-up, profiler options and engine calls are deliberately in main().
"""
import argparse
import dataclasses
import json
import os
import sys
import time
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path


def write_summary(directory, target, wall_seconds, device, args, timings):
    """Read the full XPlane record and write diagnostics; no engine work here."""
    from jax.profiler import ProfileData

    from engine.run.trace import JSON_EXPORT_EVENT_CAP, latest_trace

    path = latest_trace(str(directory))
    if path is None:
        raise RuntimeError(f"The profiler wrote no XPlane trace in {directory}")
    events, gpu_events, python_events = 0, 0, 0
    phases = Counter()
    planes = Counter()
    first, last = float("inf"), float("-inf")
    for plane in ProfileData.from_file(path).planes:
        for line in plane.lines:
            for event in line.events:
                events += 1
                planes[plane.name] += 1
                gpu_events += int(plane.name.startswith("/device:GPU:"))
                # XProf names Python calls "$file.py:line function" (also C calls).
                python_events += int(event.name.startswith("$"))
                first = min(first, event.start_ns)
                last = max(last, event.start_ns + event.duration_ns)
                if event.name in ("portfolio", "market_risk", "calibration", "simulation", "pricing", "exposure",
                                  "greeks", "base_npv", "capture_end") or event.name.startswith("greeks/"):
                    phases[event.name] += event.duration_ns / 1e9
    warnings = []
    if events > JSON_EXPORT_EVENT_CAP:
        warnings.append("The JSON export may be truncated. Open the .xplane.pb through XProf; "
                        "use --split or --phase for smaller captures.")
    span = (last - first) / 1e9 if events else 0.0
    if span < 0.5 * wall_seconds:
        warnings.append("The recorded span is less than half the capture window; inspect for missing events.")
    if not python_events:
        warnings.append("No Python call events were recorded; Python tracing may be unavailable in this runtime.")
    if target == "full" and not {"portfolio", "calibration", "simulation", "pricing", "exposure", "greeks",
                                 "market_risk", "capture_end"}.issubset(phases):
        warnings.append("Expected phase/end markers are missing from the full XPlane record; inspect for data loss.")
    summary = {"target": target, "device": str(device), "device_kind": device.device_kind,
               "settings": vars(args), "wall_seconds": wall_seconds, "span_seconds": span,
               "events": events, "python_events": python_events, "gpu_events": gpu_events,
               "planes": dict(planes), "phase_host_seconds": dict(phases), "warnings": warnings,
               "xplane": path, "xplane_bytes": os.path.getsize(path),
               "untraced_seconds": dict(timings), "device_memory": device.memory_stats()}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  {events:,} events ({python_events:,} Python, {gpu_events:,} GPU); "
          f"XPlane {summary['xplane_bytes'] / 2**20:.1f} MiB; summary: {directory / 'summary.json'}", flush=True)
    for warning in warnings:
        print(f"  NOTE: {warning}")
    if device.platform == "gpu" and not gpu_events:
        raise RuntimeError("The workload ran on a GPU but the trace contains no GPU events. "
                           "Check the profiler's CUPTI messages; see docs/concepts/profiling-python.md.")
    if not python_events:
        raise RuntimeError("The requested Python trace is missing; see the saved summary and profiler messages.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu", help="GPU required by default")
    capture = parser.add_mutually_exclusive_group()
    capture.add_argument("--split", action="store_true", help="one trace per engine phase, including market risk")
    capture.add_argument("--phase", help="trace only this phase (or one trade's Greeks)")
    parser.add_argument("--cold", action="store_true", help="skip warm-up and disable the persistent compile cache")
    parser.add_argument("--preset", choices=("small", "representative"), default="small",
                        help="representative: 28 trades, 4096 paths, 8192 risk scenarios, 8 dates, standard option grid")
    parser.add_argument("--paths", type=int, help="exposure paths, a positive power of two (preset: 64 / 4096)")
    parser.add_argument("--risk-scenarios", type=int, help="market-risk scenarios (preset: 256 / 8192)")
    parser.add_argument("--portfolio-copies", type=int, help="groups of seven trades with varied values (preset: 1 / 4)")
    parser.add_argument("--date-grid", choices=("annual", "demo"), help="3 annual dates or the shared 8-date demo grid")
    parser.add_argument("--n-per-std", type=int, help="option state-grid resolution (preset: 8 / 30)")
    parser.add_argument("--exercise-steps", type=int, help="American exercise steps per year (preset: 1 / 24)")
    parser.add_argument("--baseline", action="store_true", help="time an additional untraced warm repeat")
    parser.add_argument("--model", choices=("HullWhite", "LGM"), default="HullWhite")
    parser.add_argument("--greeks", choices=("AD", "Bump"), default="AD", help="Bump can be much slower")
    parser.add_argument("--output", default=".profile-out-python", help="trace root; runs get unique subdirectories")
    args = parser.parse_args(argv)
    # Explicit overrides win over the preset. Keep the resolved settings in summary.json.
    large = args.preset == "representative"
    defaults = {"paths": 4096 if large else 64, "risk_scenarios": 8192 if large else 256,
                "portfolio_copies": 4 if large else 1, "date_grid": "demo" if large else "annual",
                "n_per_std": 30 if large else 8, "exercise_steps": 24 if large else 1}
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.paths < 1 or args.paths & (args.paths - 1):
        parser.error("--paths must be a positive power of two")
    if args.risk_scenarios < 2:
        parser.error("--risk-scenarios must be at least 2")
    if min(args.portfolio_copies, args.n_per_std, args.exercise_steps) < 1:
        parser.error("--portfolio-copies, --n-per-std and --exercise-steps must be positive")
    if args.cold and args.baseline:
        parser.error("--baseline needs a warm run; omit --cold")
    if args.device == "gpu" and sys.platform == "win32":
        parser.exit(1, "GPU tracing requires Linux/WSL2 with the gpu extra. Run this demo in WSL, "
                       "or use --device cpu for a Windows check. See docs/concepts/profiling-python.md.\n")

    # Set the backend before importing JAX or the engine. Never silently fall back to CPU.
    os.environ["JAX_PLATFORMS"] = "cuda" if args.device == "gpu" else "cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import numpy as np
    import ORE

    if __package__:
        from demos.demo_scenarios import DATES, EVAL_DATE, demo_market, demo_simulation
    else:
        from demo_scenarios import DATES, EVAL_DATE, demo_market, demo_simulation
    from engine.instruments.american_swaption import AmericanSwaptionConfig
    from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
    from engine.instruments.european_swaption import SwaptionConfig
    from engine.instruments.swap import SwapConfig
    from engine.instruments.treasury import BondConfig, CouponPeriod
    from engine.risk.market import MarketRiskRequest, RateRiskFactors, monte_carlo_scenarios, run_market_risk
    from engine.run import GreeksConfig, LgmSwaptionEngineConfig, PortfolioRequest, PricingConfig, RunConfig, price_portfolio
    from engine.run.trace import phase, traced_phase

    try:
        devices = jax.local_devices()
    except (RuntimeError, AssertionError) as exc:
        parser.exit(1, f"Cannot initialize {args.device}: {exc}\nFor GPU use Linux/WSL2 with the gpu extra; "
                       "for a CPU check use --device cpu. See docs/concepts/profiling-python.md.\n")
    if any(device.platform != args.device for device in devices):
        parser.exit(1, f"Requested {args.device}, got {devices}\n")
    print("Devices:", ", ".join(f"{d}: {d.device_kind}" for d in devices), flush=True)
    if args.cold:
        jax.config.update("jax_enable_compilation_cache", False)

    # 1. Inputs: two calibrated currencies, correlated IR/FX/equity paths, every trade type.
    # Separate the contracts' annual payment/exercise dates from the simulation grid.
    market = demo_market(("USD", "EUR"))
    dates = tuple(EVAL_DATE + ORE.Period(y, ORE.Years) for y in (1, 2, 3))
    simulation = demo_simulation(args.model, samples=args.paths, dates=DATES if args.date_grid == "demo" else dates)
    simulation = dataclasses.replace(simulation, ir={
        ccy: dataclasses.replace(ir, calibration_expiries=("1Y", "2Y"), calibration_terms=("2Y", "1Y"))
        for ccy, ir in simulation.ir.items()
    })
    grid = LgmSwaptionEngineConfig(n_per_std=args.n_per_std, std_devs=5.0,
                                  exercise_time_steps_per_year=args.exercise_steps)
    pricing = PricingConfig(bermudan=grid, american=grid)
    trades = [
        SwapConfig(notional=2_000_000.0, fixed_rate=0.032, payer=True, swap_tenor="3Y",
                   evaluation_date=EVAL_DATE, trade_id="usd_swap"),
        SwapConfig(notional=1_000_000.0, fixed_rate=0.025, payer=True, swap_tenor="3Y", currency="EUR",
                   evaluation_date=EVAL_DATE, trade_id="eur_swap"),
        SwaptionConfig(notional=1_500_000.0, fixed_rate=0.036, payer=True, swap_tenor="2Y",
                       forward_start=ORE.Period(1, ORE.Years), evaluation_date=EVAL_DATE, trade_id="european"),
        BermudanSwaptionConfig(notional=1_000_000.0, fixed_rate=0.036, payer=True, swap_tenor="3Y",
                               exercise_dates=list(dates[:2]), evaluation_date=EVAL_DATE, trade_id="bermudan"),
        AmericanSwaptionConfig(notional=800_000.0, fixed_rate=0.035, payer=False, swap_tenor="3Y",
                               first_exercise_date=dates[0], last_exercise_date=dates[1],
                               evaluation_date=EVAL_DATE, trade_id="american"),
        BondConfig(face_amount=1_000_000.0, maturity_date=dates[1], evaluation_date=EVAL_DATE, trade_id="bill"),
        BondConfig(face_amount=1_000_000.0, maturity_date=dates[2], coupon_rate=0.0375,
                   coupon_schedule=tuple(CouponPeriod(start, end)
                                         for start, end in zip((EVAL_DATE,) + dates[:-1], dates)),
                   evaluation_date=EVAL_DATE, trade_id="note"),
    ]
    # Same contract shapes, different notionals, rates and option sides. This tests the
    # engine's per-trade work and realistic reuse of compiled programs for shared shapes.
    # It is a synthetic mix, not a measured production book or extra contract-shape diversity.
    templates = trades
    trades = []
    for copy in range(args.portfolio_copies):
        for trade in templates:
            changes = {"trade_id": trade.trade_id if args.portfolio_copies == 1 else f"{trade.trade_id}_{copy:03d}"}
            if copy:
                if isinstance(trade, BondConfig):
                    changes["face_amount"] = trade.face_amount * (1.0 + 0.1 * copy)
                    if trade.coupon_schedule:
                        changes["coupon_rate"] = trade.coupon_rate + 0.0001 * copy
                else:
                    changes.update(notional=trade.notional * (1.0 + 0.1 * copy),
                                   fixed_rate=trade.fixed_rate + 0.0001 * copy, payer=bool(copy % 2) != trade.payer)
            trades.append(dataclasses.replace(trade, **changes))
    request = PortfolioRequest(market=market, trades=trades, compute_greeks=True,
                               config=RunConfig(simulation=simulation, pricing=pricing,
                                                greeks=GreeksConfig(method=args.greeks)))
    targets = ["calibration", "simulation", "pricing", "exposure", "greeks", "market_risk"]
    available = ["portfolio", *targets, *[f"greeks/trade{i}/{type(t).__name__}" for i, t in enumerate(trades)]]
    if args.phase and args.phase not in available:
        parser.error(f"unknown --phase {args.phase!r}; choose from {', '.join(available)}")
    targets = targets if args.split else [args.phase or "full"]
    run_dir = Path(args.output).resolve() / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%fZ}-pid-{os.getpid()}"
    run_dir.mkdir(parents=True)
    print(f"Workload: {len(trades)} trades, {args.paths} paths x {len(simulation.dates)} dates, "
          f"{args.risk_scenarios} risk scenarios, {args.model}, {args.greeks} Greeks", flush=True)
    print(f"Option grid: n_per_std={grid.n_per_std}, std_devs={grid.std_devs}, "
          f"American exercise steps/year={grid.exercise_time_steps_per_year}", flush=True)
    print(f"Trace directory: {run_dir}", flush=True)

    # 2. Python calls stay ON for both whole-engine and separate-phase captures.
    options = jax.profiler.ProfileOptions()
    options.python_tracer_level = 1
    options.host_tracer_level = 2
    options.raise_error_on_start_failure = True
    # Device tracing uses JAX's default CUPTI collection; stderr remains visible.

    # 3. Warm up once, then capture all work, or repeat the real pipeline for each phase.
    # The engine calls remain here so it is clear what executes under the profiler.
    passes = ([] if args.cold else ["warmup"] + (["baseline"] if args.baseline else [])) + targets
    timings = {}
    for target in passes:
        untraced = target in ("warmup", "baseline")
        directory = run_dir / target.replace("/", "_")
        if not untraced:
            directory.mkdir()
        print(f"\n{target}: starting", flush=True)
        jax.block_until_ready(jax.live_arrays())
        if untraced or target == "full":
            window = nullcontext()
        else:
            # These callbacks only start/stop the profiler. PhaseWindow synchronizes
            # device work at both boundaries, so this phase owns its queued kernels.
            window = traced_phase(
                target, lambda: jax.profiler.start_trace(str(directory), profiler_options=options),
                jax.profiler.stop_trace,
            )
        started = time.perf_counter()
        with window as selected:
            if target == "full":
                jax.profiler.start_trace(str(directory), profiler_options=options)
            capture_started = time.perf_counter()
            try:
                with phase("portfolio"):
                    result = price_portfolio(request)
                    jax.block_until_ready(jax.live_arrays())

                # Short-horizon market risk is a separate engine entry point. Include
                # shock generation, every trade's revaluation, and VaR/ES in its region.
                with phase("market_risk"):
                    factors = RateRiskFactors.from_market(market)
                    pillar = np.arange(factors.size)
                    covariance = np.exp(-np.abs(pillar[:, None] - pillar[None, :]) / 3.0) * 0.0008**2 * 10
                    shocks = monte_carlo_scenarios(factors, covariance, horizon_days=10,
                                                   num_scenarios=args.risk_scenarios, seed=1)
                    risk = run_market_risk(MarketRiskRequest(trades=trades, market=market, scenarios=shocks,
                                                            pricing=pricing, quantiles=(0.99, 0.975)))
                    jax.block_until_ready(jax.live_arrays())
                if target == "full":
                    with phase("capture_end"):
                        pass
            finally:
                capture_seconds = time.perf_counter() - capture_started
                if target == "full":
                    jax.profiler.stop_trace()
        wall = time.perf_counter() - started
        if not untraced and target != "full":
            if selected.wall_seconds is None:
                raise RuntimeError(f"Phase {target!r} did not execute; no trace was captured")
            capture_seconds = selected.wall_seconds
        print(f"  completed in {wall:.2f}s; portfolio NPV {result.base_npv:,.2f} USD; "
              f"cube {result.npv_cube.shape}; VaR_99 {risk.risk['VaR_99']:,.2f}; "
              f"ES_97.5 {risk.risk['ES_97.5']:,.2f}", flush=True)
        if any(d.platform != args.device for d in result.npv_cube.devices() | risk.portfolio_pnl.devices()):
            raise RuntimeError("Result arrays are on a different backend than requested")
        if untraced:
            timings[target] = wall
            (run_dir / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
        else:
            write_summary(directory, target, capture_seconds, devices[0], args, timings)

    for warning in risk.warnings:
        print(f"Market-risk note: {warning}")
    viewer = Path(sys.executable).with_name("xprof.exe" if os.name == "nt" else "xprof")
    launch = "& " if os.name == "nt" else ""
    print(f'\nView: {launch}"{viewer}" --port 8791 --logdir "{run_dir}"'
          '\nOpen http://localhost:8791 and select trace_viewer.')
    print("Read the XPlane trace in XProf; the JSON export can omit late events. "
          "See docs/concepts/profiling-python.md.")


if __name__ == "__main__":
    main()
