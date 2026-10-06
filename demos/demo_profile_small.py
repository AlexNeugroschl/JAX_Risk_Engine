"""
A small portfolio sized for a profiler trace that is practical to open, on the same path as
`demo_structured.py`: calibration -> simulation -> all four instrument pricers -> exposure
-> Greeks, over the HTTP API, in the engine worker, under `jax.profiler.trace`.

Output: one `pid-<pid>/` directory under `.profile-out-small`, with the timeline labelled by
phase (calibration / simulation / pricing / exposure / greeks, and one region per trade inside
greeks, `greeks/trade<i>/<type>`), and beside each trace the worker's summary of it
(`<run>.summary.json`: wall time, compiles, events, time per phase), which this script prints;
see `docs/concepts/profiling.md` §2.

Measured 2026-10-05 on CPU (roadmap 2.1; the traced run, the server's start-up excluded):

    mode                                 wall    compiles  events     phases: pricing / greeks
    --cold --no-disk-cache (scratch)     43.1 s  269       1,499,875  9.0 s / 30.4 s
    --cold (disk cache read back)        18.8 s  269       916,387    3.6 s / 13.6 s
    default (warm: the repeat)            2.8 s    9       830,171    0.8 s /  0.9 s

Every trace is whole: it spans the run and holds every phase. xprof reads it all from the
`.xplane.pb`; the `.trace.json.gz` beside it keeps only the ~1M events that start first, so a
cold trace is partial in viewers that read that file (the summary warns). Most events are
executed XLA kernels, not compiles: 517k of a cold trace's 1.5M run inside "pricing", most
of them the 60-step bisection of `_bootstrap_bucket`, the Bermudan's and American's
recalibration on every path date, one event per op per iteration (I-53).

Run with:  .venv/Scripts/python.exe demos/demo_profile_small.py [--cold] [--no-disk-cache]
View with: xprof --port 8791 .profile-out-small

On a GPU (roadmap 2.2: Linux or WSL2 with the `gpu` extra, docs/getting-started/user-guide.md)
the same command runs the job on the GPU; the result's `ran on:` line names the device the
engine worker used. Measured on an RTX 5060 in `docs/concepts/profiling.md` §2.0.

Delete `.profile-out-small` between runs when comparing: each run adds a `pid-<pid>/`.
"""
import argparse
import glob
import json
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
# recalibrating each option under each, many times the work and the events (I-53).
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

# Cache modes (command-line switches, see `parse_args`):
#
#   default:          warm. The worker runs the job once untraced, then traced: the trace shows
#                     what a REPEAT costs, the summary also gives the untraced first run.
#   --cold:           one traced run; with the worker's persistent compilation cache populated
#                     by an earlier run (a restarted worker), compiles read it back.
#   --no-disk-cache:  turns that cache off, so a --cold trace is the job from scratch.


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


def start_server(cold: bool, disk_cache: bool) -> subprocess.Popen:
    env = os.environ.copy()
    env["JAX_RISK_PROFILE_DIR"] = PROFILE_DIR
    if not cold:
        env["JAX_RISK_PROFILE_WARMUP"] = "1"
    if not disk_cache:
        env["JAX_COMPILATION_CACHE_DIR"] = ""  # empty turns the worker's disk cache off
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
    # Where the engine worker ran the job (its precision report), not the API process.
    print(f"ran on: {result['precision']['backend']} ({', '.join(result['precision']['devices'])})")

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


def report_trace() -> None:
    """Print the newest trace's summary, which the engine worker writes beside it
    (`engine.api.worker._record_trace`): the traced run's wall time and compiles (and the
    untraced warm-up's), the trace's events and size, the share of the run it covers, and the
    time per phase."""
    summaries = glob.glob(os.path.join(PROFILE_DIR, "pid-*", "*.summary.json"))
    if not summaries:
        print(f"\nno trace summary in {PROFILE_DIR!r} (was the server started with profiling?)")
        return
    with open(max(summaries, key=os.path.getmtime), encoding="utf-8") as handle:
        summary = json.load(handle)
    traced, warmup = summary["traced"], summary["warmup"]
    print(f"\ntrace: {summary['path']}")
    if len(glob.glob(os.path.join(PROFILE_DIR, "pid-*"))) > 1:
        print(f"  (older runs also in {PROFILE_DIR!r} -- delete it between runs for a clean comparison)")
    if warmup:
        print(f"  untraced first run: {warmup['wall_seconds']:.1f}s, {warmup['compiles']} compiles")
    print(f"  traced run:         {traced['wall_seconds']:.1f}s, {traced['compiles']} compiles")
    print(f"  {summary['events']:,} events, {summary['bytes'] / 1e6:.1f} MB, spanning {summary['span_seconds']:.1f}s"
          f" = {summary['coverage']:.0%} of the traced run")
    for thread, count in list(summary["threads"].items())[:3]:
        print(f"    {count:>9,} events on {thread or 'the job thread'}")
    print("  time per phase (host time; device work lands in the phase that waits for it):")
    for name, seconds in summary["phases"].items():
        print(f"    {name:<40}{seconds:>8.2f}s")
    if summary["warning"]:
        print(f"  WARNING: {summary['warning']}")
    print(f"  view with: xprof --port 8791 {PROFILE_DIR}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cold", action="store_true",
                        help="trace the first run, compiles included (default: run once untraced, trace the repeat)")
    parser.add_argument("--no-disk-cache", dest="disk_cache", action="store_false",
                        help="turn the worker's persistent compilation cache off, so --cold compiles everything")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    print("=== stage 1: given inputs (small portfolio) ===")
    print(f"{NUM_PATHS:,} paths, {len(PORTFOLIO_TRADES)} trades (one of each type), "
          f"{len(SIMULATION_DATES)} simulation dates, the {MODEL} model, greeks=ON")

    print("\n=== stage 2: server setup ===")
    server_process = start_server(args.cold, args.disk_cache) if _MANAGE_SERVER else None
    if server_process is None:
        wait_until_healthy()
        print(f"reusing an already-running server at {API_BASE}")
        print("note: profiling is only active if THAT server was itself started "
              "with JAX_RISK_PROFILE_DIR set")
    else:
        print(f"pricing-job profiler ON -> traces in {PROFILE_DIR!r}")
        if args.cold:
            print("  cache: COLD -- the trace includes every compile"
                  + ("" if args.disk_cache else " (disk cache off)"))
        else:
            print("  cache: WARM -- job runs twice, first run untraced; the trace shows what a REPEAT costs")

    try:
        print("\n=== stage 3: server inputs ===")
        request_body = build_portfolio_request()
        print(f"built the portfolio request: {len(request_body['trades'])} trades")

        print("\n=== stage 4: submit and print ===")
        result = submit_and_wait(request_body)
        print_result(result)
        report_trace()
    finally:
        if server_process is not None:
            stop_server(server_process)


if __name__ == "__main__":
    main()
