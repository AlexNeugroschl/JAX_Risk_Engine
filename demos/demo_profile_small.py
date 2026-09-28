"""
A small portfolio sized for a profiler trace that is practical to open, on the same path as
`demo_structured.py`: calibration -> simulation -> all four instrument pricers -> exposure
-> Greeks, over the HTTP API, in a pool worker, under `jax.profiler.trace`.

Output (as measured): about 41 MB and 25 s, one `pid-<pid>/` directory under
`.profile-out-small`, with the timeline labelled by phase (calibration / simulation /
pricing / base_npv / exposure / greeks, plus one region per trade inside greeks); see
`docs/concepts/profiling.md`.

Trace size follows the number of compiled XLA programs, not array sizes, so scenario count,
grid length, tree resolution, exercise count and pillar count barely change it. Greeks do
(about 5x here), and are left on because they are a large part of the real timeline.
`docs/concepts/profiling.md` has the per-knob measurements.

Run with:  .venv/Scripts/python.exe demos/demo_profile_small.py
View with: xprof --port 8791 .profile-out-small

Delete `.profile-out-small` between runs when comparing: each run adds a `pid-<pid>/`.
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
# 3 pillars rather than demo_structured's 6, only to keep the printed Greeks vectors short
# (pillar count does not affect trace size).
ZERO_CURVE_TIMES = [0.0, 1.0, 3.0]
ZERO_CURVE_RATES = [FLAT_RATE] * len(ZERO_CURVE_TIMES)

HW_MEAN_REVERSION = 0.03
HW_SHORT_RATE_VOL = 0.01

# 256 scenarios / 4 time points (vs 4096 / 9): cuts wall time, not trace size. 256 is a
# power of two, which Sobol' balance needs (engine.simulation.market_model warns otherwise).
NUM_SCENARIOS = 256
TIME_GRID_YEARS = [0.0, 1.0, 2.0, 3.0]

# One of every instrument type, as in demo_structured.py but smaller. The two tree-priced
# trades are uncalibrated (no hw_sigma) so the calibration stage runs. 3Y tenor, 2 exercise
# dates and n_per_std=16 (a 33-node grid vs 769 at 64), for wall time and memory.
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
        "swap_tenor": "5Y", "exercise_time_steps_per_year": 1,
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

# A separate directory from demo_structured.py's `.profile-out`. Passed to uvicorn and its
# pool workers via os.environ.copy() in start_server; it arms the profiler hook in
# engine/portfolio/worker_pool.py::_run_pricing_job.
PROFILE_DIR = ".profile-out-small"

# WARM CACHE: run the job once untraced so the traced run reuses the XLA compilation caches.
# Comment this line out for a cold-start trace.
#
#   warm: what a repeat costs. Measured here: the discarded run does 208 of 239
#         compilations, the traced run 31; wall time ~26s -> ~18s.
#   cold: what the job costs from scratch. All compilations land in the trace, ~98% of wall
#         time for a portfolio this small.
#
# Compilation still dominates the warm trace's event count: the Greeks recompile on every
# call because `engine.risk.greeks` builds a fresh price_fn closure each time (I-21; see
# _grad_and_hessian_diagonal), and a compilation emits many more events than a kernel
# execution. Event count is not time.
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
    # globals().get so that commenting out WARM_CACHE turns warmup off instead of raising
    # NameError.
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
        "pfe_quantiles": RISK_PERCENTILES,
        "calibration_basket": build_calibration_basket_schema(),
        # Always on (see the module docstring).
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

    print("\nexposure profile:")
    exposure = result["exposure"]
    columns = ["epe", "ene", "ee_b"] + list(exposure["pfe"])
    print("  time  " + "".join(f"{c.upper():>12}" for c in columns))
    for i, t in enumerate(exposure["times"]):
        values = [exposure[c][i] if c in exposure else exposure["pfe"][c][i] for c in columns]
        print(f"  {t:>4.2f}" + "".join(f"{v:>12,.0f}" for v in values))
    print("(netting set; EPE/ENE/PFE discounted to today, EE_B undiscounted -- ORE's definitions)")

    print("\nGreeks (per trade index):")
    for idx, greeks in sorted(result["greeks"].items(), key=lambda kv: int(kv[0])):
        name = trade_names[int(idx)]
        values = greeks["values"]
        # A swap reports discount_delta/forward_delta (two curves); option types report a
        # single delta. Print whichever this trade has.
        shown = [k for k in ("delta", "discount_delta", "forward_delta") if k in values]
        deltas = " ".join(f"{k}={[round(v, 2) for v in values[k]]}" for k in shown)
        print(f"  [{idx}] {name:>18}: {deltas} theta={greeks['theta']:,.2f}")


def report_trace_size() -> None:
    """Print the run's trace size, and whether the trace covers the whole job: the
    profiler's event buffer (about 1M events) silently drops events once full, so the
    captured timestamp span is compared with the job's wall time."""
    if not os.path.isdir(PROFILE_DIR):
        return

    # Size is per run (per pid- subdirectory); the directory accumulates every run.
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
