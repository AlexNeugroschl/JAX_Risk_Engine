"""
A small portfolio sized for a profiler trace that is practical to open, on the same path as
`demo_structured.py`: calibration -> simulation -> all four instrument pricers -> exposure
-> Greeks, over the HTTP API, in the engine worker, under `jax.profiler.trace`.

Output: one `pid-<pid>/` directory under `.profile-out-small`, with the timeline labelled by
phase (calibration / simulation / pricing / base_npv / exposure / greeks, plus one region per
trade inside greeks); see `docs/concepts/profiling.md`. Measured 2026-10-01: the job takes
about 112 s and the trace is TRUNCATED at the profiler's ~1M-event cap (160 MB): the
Bermudan's and American's recalibration on every path date dispatches that many events in
its first 10 s (I-53). Without those two trades the whole job is 205k events, 12.8 MB.

Trace size follows the number of dispatched XLA programs, not array sizes. Greeks are left on
(AD) because they are a large part of the real timeline.

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

# The USD market of demo_structured.py on 3 pillars rather than 6, only to keep the printed
# Greeks vectors short (pillar count does not affect trace size).
CURVE_TIMES = [0.0, 1.0, 3.0]
DISCOUNT_RATES = [0.030, 0.030, 0.034]
INDEX_RATES = [0.034, 0.034, 0.038]
INDEX_NAME = "USD-SIMINDEX-6M"
VOLS = {"option_tenors": ["1Y", "2Y"], "swap_tenors": ["1Y", "5Y"], "vols": [[0.0080, 0.0088], [0.0085, 0.0091]]}

# The Hull-White model for USD, calibrated to a 1Y/2Y co-terminal basket.
MODEL = "HullWhite"
MEAN_REVERSION = 0.03
CALIBRATION_EXPIRIES, CALIBRATION_TERMS = ["1Y", "2Y"], ["2Y", "1Y"]

# 256 paths on 3 dates (vs 512 on 8): cuts wall time, not trace size. A power of two, which
# Sobol' balance needs (engine.simulation.random warns otherwise).
NUM_PATHS = 256

# AD: each trade's Greeks are a few fused programs. Bump reprices every trade under ~40 shifts,
# recalibrating each option under each, and overflows the profiler's ~1M-event cap (I-53).
GREEKS_METHOD = "AD"
SIMULATION_DATES = ["2027-07-30", "2028-07-30", "2029-07-30"]

# One of every trade type, as in demo_structured.py but smaller: 3Y tenors, 2 exercise dates,
# and a 33-node Bermudan/American grid (n_per_std=16), for wall time and memory.
PORTFOLIO_TRADES = [
    {"trade_type": "swap", "trade_id": "swap", "notional": 2_000_000.0, "fixed_rate": 0.032, "payer": True,
     "swap_tenor": "3Y"},
    {"trade_type": "european_swaption", "trade_id": "european", "notional": 1_500_000.0, "fixed_rate": 0.036,
     "payer": True, "swap_tenor": "2Y", "forward_start": "1Y"},
    {"trade_type": "bermudan_swaption", "trade_id": "bermudan", "notional": 1_000_000.0, "fixed_rate": 0.036,
     "payer": True, "exercise_dates": ["2027-07-30", "2028-07-30"], "swap_tenor": "3Y"},
    {"trade_type": "american_swaption", "trade_id": "american", "notional": 800_000.0, "fixed_rate": 0.035,
     "payer": False, "first_exercise_date": "2027-07-30", "last_exercise_date": "2028-07-30", "swap_tenor": "3Y"},
    {"trade_type": "bond", "trade_id": "bill", "face_amount": 1_000_000.0, "maturity_date": "2028-01-31"},
]
ENGINE = {"n_per_std": 16, "std_devs": 6.0, "exercise_time_steps_per_year": 1}

RISK_PERCENTILES = [0.95, 0.99]


# =============================================================================
# STAGE 2 -- SERVER SETUP (identical mechanics to demo_structured.py)
# =============================================================================

API_BASE = "http://127.0.0.1:8000"
_MANAGE_SERVER = os.environ.get("JAX_RISK_ENGINE_DEMO_SKIP_SERVER") != "1"

# A separate directory from demo_structured.py's `.profile-out`. Passed to uvicorn and its
# engine worker via os.environ.copy() in start_server; it arms the profiler hook in
# engine/api/worker.py::_profiled.
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

def build_portfolio_request() -> dict:
    curve = lambda rates: {"times": CURVE_TIMES, "rates": rates}  # noqa: E731
    return {
        "market": {"asof": EVALUATION_DATE, "currencies": {"USD": {
            "discount_curve": curve(DISCOUNT_RATES), "index_curves": {INDEX_NAME: curve(INDEX_RATES)},
            "swaption_vols": VOLS}}},
        "trades": PORTFOLIO_TRADES,
        "simulation": {"dates": SIMULATION_DATES, "base_currency": "USD", "samples": NUM_PATHS,
                       "ir": {"USD": {"model": MODEL, "reversion": MEAN_REVERSION,
                                      "calibration_expiries": CALIBRATION_EXPIRIES,
                                      "calibration_terms": CALIBRATION_TERMS}}},
        "pricing": {"bermudan": ENGINE, "american": ENGINE},
        "pfe_quantiles": RISK_PERCENTILES,
        # Always on (see the module docstring).
        "compute_greeks": True,
        "greeks": {"method": GREEKS_METHOD},
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
    trade_ids = result["trade_ids"]
    npv_cube = result["npv_cube"]
    num_paths = len(npv_cube)

    print("\nmean NPV across paths, at each simulation date:")
    print("  time  " + "".join(f"{n:>12}" for n in trade_ids))
    for i, t in enumerate(result["exposure"]["times"][1:]):  # the cube's dates (times[0] is t=0)
        means = [sum(npv_cube[s][i][j] for s in range(num_paths)) / num_paths for j in range(len(trade_ids))]
        print(f"  {t:>4.2f}  " + "".join(f"{m:>12,.0f}" for m in means))
    print(f"\nportfolio NPV today: {result['base_npv']:,.2f}")

    print("\nexposure profile:")
    exposure = result["exposure"]
    columns = ["epe", "ene", "ee_b"] + list(exposure["pfe"])
    print("  time  " + "".join(f"{c.upper():>12}" for c in columns))
    for i, t in enumerate(exposure["times"]):
        values = [exposure[c][i] if c in exposure else exposure["pfe"][c][i] for c in columns]
        print(f"  {t:>4.2f}" + "".join(f"{v:>12,.0f}" for v in values))

    print("\nGreeks (discount-curve Delta per tenor, Theta):")
    for idx, greeks in sorted(result["greeks"].items(), key=lambda kv: int(kv[0])):
        delta = [round(v, 2) for v in greeks["values"]["delta:discount:USD"]]
        print(f"  {trade_ids[int(idx)]:>10}: {delta} theta={greeks['theta']:,.2f}")


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
    print(f"{NUM_PATHS:,} paths, {len(PORTFOLIO_TRADES)} trades (one of each type), "
          f"{len(SIMULATION_DATES)} simulation dates, the {MODEL} model, greeks=ON")

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
        request_body = build_portfolio_request()
        print(f"built the portfolio request: {len(request_body['trades'])} trades")

        print("\n=== stage 4: submit and print ===")
        result = submit_and_wait(request_body)
        print_result(result)
        report_trace_size()
    finally:
        if server_process is not None:
            stop_server(server_process)


if __name__ == "__main__":
    main()
