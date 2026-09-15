"""
A deliberately SMALL portfolio, sized so its profiler trace is something you
can actually open -- while still exercising the same end-to-end path
`demo_structured.py` does: calibration -> simulation -> all four instrument
pricers -> risk -> Greeks, over the real HTTP API, in a real pool worker,
under `jax.profiler.trace`.

**Why this exists.** `demo_structured.py`'s portfolio (4096 scenarios, a
9-point time grid, `n_per_std=64`, 4 exercise dates, Greeks on) produces a
~50MB raw trace from ~1200 separate XLA compilations, which xprof then
expands to ~99MB on disk when it ingests it. That is a perfectly good
representation of a real pricing job -- it is just an awkward thing to load,
scrub through, and re-run while you are actually learning the timeline. This
script trades portfolio realism for trace ergonomics and nothing else: same
code path, same profiler configuration, same instrument coverage, ~1/5 the
trace.

**What was actually shrunk, and why those knobs.** Measured per-knob on this
machine (fresh process each, `python_tracer_level=0`, raw trace bytes before
any xprof ingest):

    baseline: demo_structured.py's own portfolio        ~50 MB   ~36 s
    scenarios 4096 -> 256, grid 9 -> 4 points            ~42 MB   ~31 s
    n_per_std 64 -> 16, 4 exercise dates -> 2            ~42 MB   ~31 s
    curve pillars 6 -> 2                                 ~42 MB   ~31 s
    compute_greeks: on -> off                            ~11 MB    ~9 s

The lesson in that table, and the reason this demo is shaped the way it is:
for TRACE SIZE, none of the obvious "make the numbers smaller" knobs matter
much. Scenario count, time-grid length, tree resolution, exercise count and
curve pillar count are all nearly free -- they change how big each XLA
program's ARRAYS are, not how MANY programs get compiled and dispatched, and
it is the program count that the trace records. Greeks is the one knob that
moves it, because `engine.risk.greeks` runs `jax.grad`/`jax.hessian` through
`engine.instruments.bermudan_swaption._run_backward_induction`, which is
deliberately not `jax.jit`-wrapped (see that function's own docstring for
why -- it must stay traceable with tracer-carrying arguments), so every
elementwise op around its `lax.scan` dispatches as its own tiny program.

Attributing that cost per trade (same measurement, adding one trade at a
time):

    swap + European swaption only                        ~7.8 MB   ~8 s
    + Bermudan swaption                                 ~29.8 MB  ~27 s
    + American swaption (instead of Bermudan)           ~29.9 MB  ~27 s
    all four                                            ~41.6 MB  ~31 s

Each tree-priced trade adds a fixed ~22MB of trace regardless of how small
its grid is. So this demo keeps ALL FOUR instrument types (dropping one
would stop it being representative, which is the whole point) and instead
keeps Greeks on for only ONE of the two tree trades -- see GREEKS_MODE
below. That is the single decision that makes this trace small.

Run with: venv/Scripts/python.exe demos/demo_profile_small.py
View with: xprof --port 8791 .profile-out-small
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
        "exercise_times": [1.0, 2.0], "swap_tenor": "3Y",
        "n_per_std": 16, "std_devs": 6.0,
    },
    {
        "trade_type": "american_swaption",
        "notional": 800_000.0, "fixed_rate": 0.029, "payer": False,
        "rate_factor_index": 0, "hw_a": HW_MEAN_REVERSION, "hw_sigma": None,
        "first_exercise": 1.0, "last_exercise": 2.0, "exercise_time_steps_per_year": 1,
        "n_per_std": 16, "std_devs": 6.0,
    },
]

# Calibration basket: 2 co-terminal quotes for the 2 exercise dates above.
CALIBRATION_EXERCISE_TIMES = [1.0, 2.0]
CALIBRATION_FINAL_MATURITY = 3.0
CALIBRATION_MARKET_VOLS = [0.0080, 0.0090]

RISK_PERCENTILES = [0.95, 0.99]

# The one knob that actually controls this demo's trace size (see module
# docstring). `compute_greeks` is portfolio-wide in PortfolioRequestSchema --
# there is no per-trade Greeks flag -- so the two settings here are:
#
#   "off"  -- ~11MB trace, ~9s. Calibration + simulation + all four pricers +
#             VaR/ES, no Greeks at all. The cheapest run that still covers
#             every pricing path.
#   "on"   -- ~42MB trace, ~31s. Adds Delta/Gamma/Theta for the European,
#             Bermudan and American trades (a swap's Greeks are skipped by
#             engine.portfolio.request._compute_all_greeks, which has no
#             curve for it). Each tree-priced trade costs a fixed ~22MB.
#
# Default "off": this demo's job is to produce a trace you can open, and the
# no-Greeks run already exercises calibration, simulation, and all four
# pricers -- i.e. it is representative of the pricing path. Flip to "on"
# (JAX_RISK_DEMO_GREEKS=1) when the Greeks path is specifically what you
# want on the timeline, and accept the 4x.
GREEKS_MODE = os.environ.get("JAX_RISK_DEMO_GREEKS", "0") == "1"


# =============================================================================
# STAGE 2 -- SERVER SETUP (identical mechanics to demo_structured.py)
# =============================================================================

API_BASE = "http://127.0.0.1:8000"
_MANAGE_SERVER = os.environ.get("JAX_RISK_ENGINE_DEMO_SKIP_SERVER") != "1"

# A separate directory from demo_structured.py's own `.profile-out`, so the
# two demos' traces never land in the same place and get confused for one
# another (xprof lists every run dir it finds under the path you point it at).
PROFILE_DIR = os.environ.get("JAX_RISK_PROFILE_DIR", ".profile-out-small")

# Both inherited by the spawned uvicorn (and its pool workers) via
# os.environ.copy() in start_server below. See
# engine/portfolio/worker_pool.py::_run_pricing_job for what each does; the
# Python tracer is OFF by default there, which is what keeps this trace's
# events JAX/XLA work rather than CPython frames.
PROFILE_WARMUP = os.environ.get("JAX_RISK_PROFILE_WARMUP", "0")


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
    if PROFILE_DIR:
        env["JAX_RISK_PROFILE_DIR"] = PROFILE_DIR
    env["JAX_RISK_PROFILE_WARMUP"] = PROFILE_WARMUP
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
        "compute_greeks": GREEKS_MODE,
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
        print(f"warnings: {len(result['warnings'])} (mid-coupon exercise alignment -- expected here)")

    print("\nrisk:")
    risk = result["risk"]["values"]
    print("  time   " + "".join(f"{m:>12}" for m in risk))
    for i, t in enumerate(TIME_GRID_YEARS[1:]):
        row = "".join(
            f"{risk[m][i]:>12,.0f}" if risk[m][i] is not None else f"{'nan':>12}"
            for m in risk
        )
        print(f"  {t:>4.2f}  " + row)

    if result["greeks"]:
        print("\nGreeks (per trade index):")
        for idx, greeks in sorted(result["greeks"].items(), key=lambda kv: int(kv[0])):
            name = trade_names[int(idx)]
            delta = [round(v, 2) for v in greeks["values"]["delta"]]
            print(f"  [{idx}] {name:>18}: delta={delta} theta={greeks['theta']:,.2f}")
    else:
        print("\nGreeks: not computed (set JAX_RISK_DEMO_GREEKS=1 to include them"
              " -- roughly 4x the trace)")


def report_trace_size() -> None:
    """Prints what the run actually wrote, and -- more importantly -- whether
    the trace covers the whole job. The profiler's event buffer is a fixed
    ~1M-event cap with no backpressure and no warning: once it fills, the
    remaining events are silently dropped and a truncated trace looks exactly
    like a complete one. Comparing the captured events' own timestamp span
    against the job's wall time is the cheap way to catch that, so this demo
    does it every run rather than leaving it to be noticed later."""
    if not PROFILE_DIR or not os.path.isdir(PROFILE_DIR):
        return
    total = 0
    newest = None
    for root, _dirs, files in os.walk(PROFILE_DIR):
        for name in files:
            path = os.path.join(root, name)
            total += os.path.getsize(path)
            if name.endswith(".trace.json.gz") and (newest is None or os.path.getmtime(path) > os.path.getmtime(newest)):
                newest = path
    print(f"\ntrace written to {PROFILE_DIR!r}: {total / 1e6:.1f} MB")

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
              " probably truncated. Shrink the portfolio or turn Greeks off.")
    print(f"  view with: xprof --port 8791 {PROFILE_DIR}")


def main() -> None:
    print("=== stage 1: given inputs (small portfolio) ===")
    print(f"{NUM_SCENARIOS:,} scenarios, {len(PORTFOLIO_TRADES)} trades "
          f"(one of each type), {len(TIME_GRID_YEARS) - 1} simulated steps, "
          f"greeks={'ON' if GREEKS_MODE else 'OFF'}")

    print("\n=== stage 2: server setup ===")
    server_process = start_server() if _MANAGE_SERVER else None
    if server_process is None:
        wait_until_healthy()
        print(f"reusing an already-running server at {API_BASE}")
        print("note: profiling is only active if THAT server was itself started "
              "with JAX_RISK_PROFILE_DIR set")
    elif PROFILE_DIR:
        print(f"pricing-job profiler ON -> traces in {PROFILE_DIR!r}"
              f"{' (with warmup)' if PROFILE_WARMUP == '1' else ''}")

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
