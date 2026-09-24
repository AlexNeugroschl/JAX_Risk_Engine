"""
A deliberately SMALL portfolio, sized so its profiler trace is something you
can actually open -- while still exercising the full end-to-end path
`demo_structured.py` does: calibration -> simulation -> all four instrument
pricers -> risk -> Greeks, over the real HTTP API, in a real pool worker,
under `jax.profiler.trace`.

**Why this exists.** `demo_structured.py`'s portfolio (4096 scenarios, a
9-point time grid, `n_per_std=64`, 4 exercise dates) produces a trace that is
a perfectly good representation of a real pricing job -- it is just an awkward
thing to load, scrub through, and re-run while you are actually learning the
timeline. This script trades portfolio realism for trace ergonomics and
nothing else: same code path, same profiler configuration, same instrument
coverage, a fraction of the trace.

**What it produces:** ~41 MB, ~25 s, one `pid-<pid>/` directory under
`.profile-out-small`. The timeline is labelled by phase (calibration /
simulation / pricing / base_npv / risk / greeks, plus one region per trade
inside greeks) -- see `docs/concepts/profiling.md` for how those annotations
work and how to read the result.

**Sizing note, since it is counter-intuitive.** For TRACE SIZE, the obvious
"make the numbers smaller" knobs barely matter: scenario count, time-grid
length, tree resolution, exercise count and curve pillar count change how big
each XLA program's ARRAYS are, not how MANY programs get compiled, and it is
the program count the trace records. Greeks is the one knob that genuinely
moves it (~5x here) -- which is exactly why this demo leaves Greeks ON: the
Greeks path is the interesting part of the timeline, and a trace that omits
it is not representative of what this engine actually spends its time on.
`docs/concepts/profiling.md` has the full per-knob measurements.

Run with:  .venv/Scripts/python.exe demos/demo_profile_small.py
View with: xprof --port 8791 .profile-out-small

Delete `.profile-out-small` between runs when comparing: the profiler writes
a new `pid-<pid>/` each time and never cleans up after itself.
"""
import os
import subprocess
import sys
import time

import httpx

# =============================================================================
# STAGE 1 -- GIVEN INPUTS (small, but the same KINDS of input as demo_structured)
# =============================================================================

EVALUATION_DATE = "2026-07-30"

FLAT_RATE = 0.03
# 3 pillars rather than demo_structured's 6. Pillar count turned out not to
# matter for trace size (see module docstring's table -- 6 -> 2 changed
# nothing), so this is kept at a readable 3 purely so the printed Greeks
# vectors are short, not as a size optimization.
ZERO_CURVE_TIMES = [0.0, 1.0, 3.0]
ZERO_CURVE_RATES = [FLAT_RATE] * len(ZERO_CURVE_TIMES)

HW_MEAN_REVERSION = 0.03
HW_SHORT_RATE_VOL = 0.01

# 256 scenarios / 4 time points, vs demo_structured's 4096 / 9. Barely moves
# the trace (see docstring) but does cut wall time and keeps the printed NPV
# table small enough to read at a glance. 256 is still a power of two, which
# keeps the Sobol' engine's balance property (engine.simulation.market_model
# warns when it isn't).
NUM_SCENARIOS = 256
TIME_GRID_YEARS = [0.0, 1.0, 2.0, 3.0]

# One of every instrument type this engine prices -- deliberately unchanged
# in KIND from demo_structured.py, only in size. The two tree-priced trades
# stay UNCALIBRATED (hw_sigma left out) so the calibration stage below is
# genuinely exercised rather than skipped.
#
# Everything is 3Y-tenor with 2 exercise dates (vs 5Y / 4 dates), and
# n_per_std=16 (vs 64 -- a 33-node state grid instead of 769). Again: this
# is for wall time and memory, not trace size.
PORTFOLIO_TRADES = [
    {
        "trade_type": "swap",
        "notional": 2_000_000.0, "fixed_rate": 0.032, "payer": True,
        "discount_curve_index": 0, "forward_curve_index": 0, "swap_tenor": "3Y",
    },
    {
        "trade_type": "european_swaption",
        "notional": 1_500_000.0, "fixed_rate": 0.031, "payer": True,
        "rate_factor_index": 0, "hw_a": HW_MEAN_REVERSION, "hw_sigma": HW_SHORT_RATE_VOL,
        "swap_tenor": "2Y", "forward_start": "1Y",
    },
    {
        "trade_type": "bermudan_swaption",
        "notional": 1_000_000.0, "fixed_rate": 0.030, "payer": True,
        "rate_factor_index": 0, "hw_a": HW_MEAN_REVERSION, "hw_sigma": None,
        "exercise_dates": ["2027-07-30", "2028-07-30"], "swap_tenor": "3Y",
        "n_per_std": 16, "std_devs": 6.0,
    },
    {
        "trade_type": "american_swaption",
        "notional": 800_000.0, "fixed_rate": 0.029, "payer": False,
        "rate_factor_index": 0, "hw_a": HW_MEAN_REVERSION, "hw_sigma": None,
        "first_exercise_date": "2027-07-30", "last_exercise_date": "2028-07-30",
        "exercise_time_steps_per_year": 1,
        "n_per_std": 16, "std_devs": 6.0,
    },
]

# Calibration basket: 2 co-terminal quotes for the 2 exercise dates above.
CALIBRATION_EXERCISE_TIMES = [1.0, 2.0]
CALIBRATION_FINAL_MATURITY = 3.0
CALIBRATION_MARKET_VOLS = [0.0080, 0.0090]

RISK_PERCENTILES = [0.95, 0.99]


# =============================================================================
# STAGE 2 -- SERVER SETUP (identical mechanics to demo_structured.py)
# =============================================================================

API_BASE = "http://127.0.0.1:8000"
_MANAGE_SERVER = os.environ.get("JAX_RISK_ENGINE_DEMO_SKIP_SERVER") != "1"

# A separate directory from demo_structured.py's own `.profile-out`, so the
# two demos' traces never land in the same place and get confused for one
# another (xprof lists every run dir it finds under the path you point it at).
# Inherited by the spawned uvicorn (and its pool workers) via os.environ.copy()
# in start_server below -- that variable is what arms the profiler hook in
# engine/portfolio/worker_pool.py::_run_pricing_job.
PROFILE_DIR = ".profile-out-small"

# WARM CACHE: run the job once and throw it away BEFORE opening the trace, so
# the traced run hits already-populated XLA compilation caches. Comment this
# line out to go back to a cold-start trace.
#
#   warm (this line active)    -- "what does a REPEAT of this job cost?"
#       Measured on this portfolio: the discarded run absorbs 208 of the 239
#       XLA compilations, and the traced run does 31. Wall time ~26s -> ~18s.
#
#   cold (this line commented) -- "what does this job cost from scratch?"
#       All 208 compilations land inside the trace. For a portfolio this small
#       that is ~98% of the wall time -- a real result, not a defect, since
#       compilation is per-program and amortized over every later call.
#
# **Warmup does NOT make this an execution-only trace, and the difference is
# worth understanding.** Compilation still visibly dominates the warm timeline
# (~27k MLIR pass events vs ~630 ThunkExecutor::Execute). Two reasons, both
# real:
#
#   1. Those 31 residual compiles are genuine. `engine.risk.greeks` builds a
#      FRESH price_fn closure per call, and jax.jit keys its cache on function
#      identity -- so the Greeks programs recompile even for an identical
#      trade. Documented in engine/risk/greeks.py::_grad_and_hessian_diagonal
#      and docs/concepts/profiling.md; pinned by a test so it cannot silently
#      regress.
#   2. 31 compilations of LARGE fused programs still emit far more trace
#      events than ~630 kernel executions on 256 scenarios. Event count is not
#      proportional to time spent.
#
# So: use warm to see what a steady-state repeat costs, not to make
# compilation disappear. If you want compilation to actually vanish from the
# timeline, the residual-recompile issue above has to be fixed first.
WARM_CACHE = True


def wait_until_healthy(timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if httpx.get(f"{API_BASE}/health", timeout=2.0).status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"server at {API_BASE} did not become healthy within {timeout_s}s")


def start_server() -> subprocess.Popen:
    env = os.environ.copy()
    env["JAX_RISK_PROFILE_DIR"] = PROFILE_DIR
    # globals().get(...) rather than a bare `WARM_CACHE` reference so that
    # COMMENTING OUT the constant above is a valid way to turn warmup off,
    # rather than a NameError. Absent -> off, which is also the profiler
    # hook's own default (worker_pool._run_pricing_job).
    if globals().get("WARM_CACHE", False):
        env["JAX_RISK_PROFILE_WARMUP"] = "1"
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "engine.api.app:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=env,
    )
    print(f"launched uvicorn (pid {process.pid}), waiting for {API_BASE}/health ...")
    try:
        wait_until_healthy()
    except Exception:
        process.terminate()
        raise
    print("server is up")
    return process


def stop_server(process: subprocess.Popen) -> None:
    process.terminate()
    process.wait(timeout=10)
    print(f"stopped uvicorn (pid {process.pid})")


# =============================================================================
# STAGE 3 -- SERVER INPUTS
# =============================================================================

def build_zero_curve_schema() -> dict:
    return {"times": ZERO_CURVE_TIMES, "rates": ZERO_CURVE_RATES}


def build_market_schema(zero_curve: dict) -> dict:
    return {
        "time_grid": TIME_GRID_YEARS,
        "scenarios": NUM_SCENARIOS,
        "equities": {"initial_prices": [100.0], "dividend_yields": [0.0], "rate_mapping": [[0.0]]},
        "rates": {
            "initial_rates": [FLAT_RATE], "theta": [FLAT_RATE], "mean_reversion": [HW_MEAN_REVERSION],
            "initial_zero_curves": [zero_curve],
        },
        "joint_covariance": [[0.04, 0.0], [0.0, HW_SHORT_RATE_VOL ** 2]],
    }


def build_trades_schema(zero_curve: dict) -> list:
    trades = []
    for trade in PORTFOLIO_TRADES:
        trade = dict(trade)
        if trade["trade_type"] != "swap":
            trade["initial_zero_curve"] = zero_curve
        trades.append(trade)
    return trades


def build_calibration_basket_schema() -> dict:
    return {
        "exercise_times": CALIBRATION_EXERCISE_TIMES,
        "final_maturity_time": CALIBRATION_FINAL_MATURITY,
        "notional": 1_000_000.0,
        "payer": True,
        "market_vols": CALIBRATION_MARKET_VOLS,
    }


def build_portfolio_request_schema() -> dict:
    zero_curve = build_zero_curve_schema()
    return {
        "evaluation_date": EVALUATION_DATE,
        "market": build_market_schema(zero_curve),
        "trades": build_trades_schema(zero_curve),
        "percentiles": RISK_PERCENTILES,
        "calibration_basket": build_calibration_basket_schema(),
        # Always on -- see module docstring: the Greeks path is the
        # interesting part of this timeline, and a trace without it is not
        # representative of where this engine spends its time.
        "compute_greeks": True,
    }


# =============================================================================
# STAGE 4 -- SUBMIT AND PRINT
# =============================================================================

def submit_and_wait(request_body: dict) -> dict:
    submit = httpx.post(f"{API_BASE}/portfolio/price", json=request_body, timeout=30.0)
    if submit.status_code != 202:
        raise RuntimeError(f"submission failed ({submit.status_code}): {submit.text}")
    job_id = submit.json()["job_id"]
    print(f"job_id: {job_id} (202 Accepted -- pricing is running in the background)")

    start = time.time()
    while True:
        poll = httpx.get(f"{API_BASE}/portfolio/price/{job_id}", timeout=30.0)
        poll.raise_for_status()
        status = poll.json()
        print(f"  [{time.time() - start:6.1f}s] status: {status['status']}")
        if status["status"] in ("done", "failed"):
            break
        time.sleep(1.0)

    if status["status"] == "failed":
        raise RuntimeError(f"pricing job failed:\n{status['error']}")
    return status["result"]


def print_result(result: dict) -> None:
    trade_names = [t["trade_type"] for t in PORTFOLIO_TRADES]
    npv_cube = result["npv_cube"]
    num_scenarios = len(npv_cube)

    print("\nmean NPV across scenarios, at each simulated time step:")
    print("  time   " + "".join(f"{n:>18}" for n in trade_names))
    for i, t in enumerate(TIME_GRID_YEARS[1:]):
        means = [
            sum(npv_cube[s][i][j] for s in range(num_scenarios)) / num_scenarios
            for j in range(len(trade_names))
        ]
        print(f"  {t:>4.2f}  " + "".join(f"{m:>18,.0f}" for m in means))

    print(f"\nbaseline portfolio NPV: {result['base_npv']:,.2f}")
    if result["warnings"]:
        print(f"warnings: {len(result['warnings'])}")
        for message in result["warnings"]:
            print(f"  - {message}")

    print("\nrisk:")
    risk = result["risk"]["values"]
    # Column width is derived from the longest metric NAME rather than fixed:
    # compute_risk_metrics reports Monte Carlo diagnostics alongside the
    # headline VaR/ES (e.g. "ES_95_standardError", 19 chars), which a
    # hardcoded width silently runs together into an unreadable header.
    width = max(12, max(len(m) for m in risk) + 2)
    print("  time  " + "".join(f"{m:>{width}}" for m in risk))
    for i, t in enumerate(TIME_GRID_YEARS[1:]):
        row = "".join(
            f"{risk[m][i]:>{width},.0f}" if risk[m][i] is not None else f"{'nan':>{width}}"
            for m in risk
        )
        print(f"  {t:>4.2f} " + row)

    print("\nGreeks (per trade index):")
    for idx, greeks in sorted(result["greeks"].items(), key=lambda kv: int(kv[0])):
        name = trade_names[int(idx)]
        values = greeks["values"]
        # A swap reports discount_delta/forward_delta (one per curve -- it
        # has two); every option type reports a single delta against its one
        # calibration curve. Print whichever this trade has rather than
        # assuming "delta": engine.portfolio.request's _compute_all_greeks
        # covers swaps too, so both shapes occur in this very portfolio.
        shown = [k for k in ("delta", "discount_delta", "forward_delta") if k in values]
        deltas = " ".join(f"{k}={[round(v, 2) for v in values[k]]}" for k in shown)
        print(f"  [{idx}] {name:>18}: {deltas} theta={greeks['theta']:,.2f}")


def report_trace_size() -> None:
    """Prints what the run actually wrote, and -- more importantly -- whether
    the trace covers the whole job. The profiler's event buffer is a fixed
    ~1M-event cap with no backpressure and no warning: once it fills, the
    remaining events are silently dropped and a truncated trace looks exactly
    like a complete one. Comparing the captured events' own timestamp span
    against the job's wall time is the cheap way to catch that, so this demo
    does it every run rather than leaving it to be noticed later."""
    if not os.path.isdir(PROFILE_DIR):
        return

    # Size is reported PER RUN (per pid- subdirectory), not for the whole
    # directory. Each run writes a new `pid-<pid>/`, and the profiler never
    # cleans up after itself, so summing the directory reports the total of
    # every run ever done into it -- which reads as a trace that grows every
    # time you run the demo, and hid a genuine size REDUCTION behind three
    # older runs the first time this was measured.
    runs = [
        os.path.join(PROFILE_DIR, name) for name in os.listdir(PROFILE_DIR)
        if name.startswith("pid-") and os.path.isdir(os.path.join(PROFILE_DIR, name))
    ]
    if not runs:
        return
    this_run = max(runs, key=os.path.getmtime)

    total = 0
    newest = None
    for root, _dirs, files in os.walk(this_run):
        for name in files:
            path = os.path.join(root, name)
            total += os.path.getsize(path)
            if name.endswith(".trace.json.gz") and (newest is None or os.path.getmtime(path) > os.path.getmtime(newest)):
                newest = path
    print(f"\ntrace written to {this_run!r}: {total / 1e6:.1f} MB")
    if len(runs) > 1:
        print(f"  ({len(runs) - 1} older run(s) also in {PROFILE_DIR!r}"
              f" -- delete it between runs for a clean comparison)")

    if newest is None:
        return
    try:
        import gzip
        import json
        events = json.load(gzip.open(newest, "rt"))["traceEvents"]
    except Exception as exc:  # a partially-flushed trace shouldn't fail the demo
        print(f"  (could not read {os.path.basename(newest)}: {exc})")
        return
    stamps = [e["ts"] for e in events if "ts" in e]
    python_frames = sum(1 for e in events if str(e.get("name", "")).startswith("$"))
    if stamps:
        print(f"  {len(events):,} events spanning {(max(stamps) - min(stamps)) / 1e6:.1f}s"
              f" ({python_frames:,} CPython frames)")
    if len(events) > 950_000:
        print("  WARNING: near the profiler's ~1M-event cap -- this trace is"
              " probably truncated. Shrink the portfolio.")
    print(f"  view with: xprof --port 8791 {PROFILE_DIR}")


def main() -> None:
    print("=== stage 1: given inputs (small portfolio) ===")
    print(f"{NUM_SCENARIOS:,} scenarios, {len(PORTFOLIO_TRADES)} trades "
          f"(one of each type), {len(TIME_GRID_YEARS) - 1} simulated steps, "
          f"greeks=ON")

    print("\n=== stage 2: server setup ===")
    server_process = start_server() if _MANAGE_SERVER else None
    if server_process is None:
        wait_until_healthy()
        print(f"reusing an already-running server at {API_BASE}")
        print("note: profiling is only active if THAT server was itself started "
              "with JAX_RISK_PROFILE_DIR set")
    else:
        warm = globals().get("WARM_CACHE", False)
        print(f"pricing-job profiler ON -> traces in {PROFILE_DIR!r}")
        if warm:
            print("  cache: WARM -- job runs twice, first run discarded; the trace "
                  "shows what a REPEAT costs")
            print("         (compilation is reduced, NOT eliminated -- see WARM_CACHE's comment)")
        else:
            print("  cache: COLD -- the trace includes all XLA lowering/compilation")

    try:
        print("\n=== stage 3: server inputs ===")
        request_body = build_portfolio_request_schema()
        print(f"built a PortfolioRequestSchema body: {len(request_body['trades'])} trades, "
              f"calibration_basket for {len(CALIBRATION_EXERCISE_TIMES)} market vol quotes")

        print("\n=== stage 4: submit and print ===")
        result = submit_and_wait(request_body)
        print_result(result)
        report_trace_size()
    finally:
        if server_process is not None:
            stop_server(server_process)


if __name__ == "__main__":
    main()
