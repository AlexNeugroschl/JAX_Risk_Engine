"""
The portfolio of `demo_api.py`, split into the four stages an integrator meets:

  1. GIVEN INPUTS     market and portfolio facts (curves, trades, vol quotes); plain Python
                      values with no server or schema concepts.
  2. SERVER SETUP     starting an engine.api server, with the pricing-job profiler on
                      (JAX_RISK_PROFILE_DIR; see PROFILE_DIR below). Same for any portfolio.
  3. SERVER INPUTS    reshaping stage 1 into `engine.api.schemas.PortfolioRequestSchema`
                      JSON.
  4. SUBMIT AND PRINT send, poll, print.

Meant as a template for a real integration: it shows what a caller must supply versus what
is deployment or translation.

Run with: .venv/Scripts/python.exe demos/demo_structured.py
"""
import os
import subprocess
import sys
import time

import httpx

# =============================================================================
# STAGE 1 -- GIVEN INPUTS
# Market and portfolio facts, independent of how the server runs or what its API expects.
# =============================================================================

EVALUATION_DATE = "2026-07-30"          # ISO date -- "today" for this pricing run

# Today's market: a single flat 3% zero curve, one interest-rate factor.
FLAT_RATE = 0.03
ZERO_CURVE_TIMES = [0.0, 1.0, 2.0, 5.0, 10.0, 30.0]
ZERO_CURVE_RATES = [FLAT_RATE] * len(ZERO_CURVE_TIMES)

# Hull-White/LGM model parameters for that one rate factor.
HW_MEAN_REVERSION = 0.03                # "a" -- mean-reversion speed
HW_SHORT_RATE_VOL = 0.01                # flat short-rate vol (simulation input)

# Monte Carlo simulation settings: how many alternate futures, and at which
# points in time to evaluate the portfolio in each of them.
NUM_SCENARIOS = 4096
TIME_GRID_YEARS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

# The portfolio: one of each instrument type. The Bermudan and American are uncalibrated
# (no hw_sigma); their vol is calibrated server-side to the basket quotes below.
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
        "swap_tenor": "3Y", "forward_start": "2Y",
        # initial_zero_curve is filled in at stage 3 (every trade shares the one curve).
    },
    {
        "trade_type": "bermudan_swaption",
        "notional": 1_000_000.0, "fixed_rate": 0.030, "payer": True,
        "rate_factor_index": 0, "hw_a": HW_MEAN_REVERSION, "hw_sigma": None,
        "exercise_dates": ["2027-07-30", "2028-07-30", "2029-07-30", "2030-07-30"], "swap_tenor": "5Y",
        "n_per_std": 64, "std_devs": 6.0,
    },
    {
        "trade_type": "american_swaption",
        "notional": 800_000.0, "fixed_rate": 0.029, "payer": False,
        "rate_factor_index": 0, "hw_a": HW_MEAN_REVERSION, "hw_sigma": None,
        "first_exercise_date": "2027-07-30", "last_exercise_date": "2030-07-30",
        "swap_tenor": "5Y", "exercise_time_steps_per_year": 2,
        "n_per_std": 64, "std_devs": 6.0,
    },
]

# Market swaption vol quotes for calibrating the Bermudan/American: one per exercise date,
# co-terminal at the Bermudan's final maturity (5Y).
CALIBRATION_EXERCISE_TIMES = [1.0, 2.0, 3.0, 4.0]
CALIBRATION_FINAL_MATURITY = 5.0
CALIBRATION_MARKET_VOLS = [0.0080, 0.0088, 0.0095, 0.0100]

# Risk parameters: which PFE quantiles the exposure profile reports, and
# whether to also compute Greeks.
RISK_PERCENTILES = [0.95, 0.99]
COMPUTE_GREEKS = True


# =============================================================================
# STAGE 2 -- SERVER SETUP
# Infrastructure only; nothing here reads stage 1. See docs/reference/http-api.md, and
# docs/getting-started/user-guide.md#running-the-api to run a server by hand.
# =============================================================================

API_BASE = "http://127.0.0.1:8000"
_MANAGE_SERVER = os.environ.get("JAX_RISK_ENGINE_DEMO_SKIP_SERVER") != "1"

# The profiler is on by default: the server starts with JAX_RISK_PROFILE_DIR set, so each job
# the worker pool runs is wrapped in jax.profiler.trace
# (engine/portfolio/worker_pool.py::_run_pricing_job), one pid-<worker-pid>/ subdirectory
# per worker. Needs the `profiling` extra (`pip install -e .[api,profiling]`). View with
# `xprof --port 8791 <dir>`. Override from the environment, or set "" to opt out.
#
# Two more knobs, read by _run_pricing_job and passed through below:
#   JAX_RISK_PROFILE_WARMUP=1        run the job once untraced first, so the trace shows
#       warm execution rather than compilation (the job runs twice). Off by default.
#   JAX_RISK_PROFILE_PYTHON_TRACER=1 turn on JAX's Python tracer (JAX defaults it on; off
#       here). Off still keeps compilation, dispatch, tracing and execution; it drops only
#       CPython frames. Measured on this portfolio: on, CPython frames were 97% of events
#       and the capture truncated to the first 1.6s of a ~90s job (469MB vs 50MB).
PROFILE_DIR = os.environ.get("JAX_RISK_PROFILE_DIR", ".profile-out")


def wait_until_healthy(timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if httpx.get(f"{API_BASE}/health", timeout=2.0).status_code == 200:
                return
        except httpx.TransportError:
            # Not listening yet, or still importing JAX/ORE: expected during startup.
            pass
        time.sleep(0.5)
    raise RuntimeError(f"server at {API_BASE} did not become healthy within {timeout_s}s")


def start_server() -> subprocess.Popen:
    # Pass the profiler dir explicitly so PROFILE_DIR's default applies even when the caller
    # did not set JAX_RISK_PROFILE_DIR; the pool workers inherit it from uvicorn.
    env = os.environ.copy()
    if PROFILE_DIR:
        env["JAX_RISK_PROFILE_DIR"] = PROFILE_DIR
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
# Stage 1 reshaped into PortfolioRequestSchema JSON
# (docs/reference/http-api.md#request-schema-portfoliorequestschema); no new facts.
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
            # maturities left unset: the server derives the swap's cashflow pillars.
        },
        "joint_covariance": [[0.04, 0.0], [0.0, HW_SHORT_RATE_VOL ** 2]],
    }


def build_trades_schema(zero_curve: dict) -> list:
    """Put the shared zero curve on every trade (the schema carries it per trade)."""
    trades = []
    for trade in PORTFOLIO_TRADES:
        trade = dict(trade)
        if trade["trade_type"] != "swap":
            trade["initial_zero_curve"] = zero_curve
        trades.append(trade)
    return trades


def build_calibration_basket_schema() -> dict:
    """Resolves the trades with hw_sigma=null: the server builds a co-terminal basket from
    these vols on the first uncalibrated Bermudan/American's curve, evaluation date and
    index tenor, fits a piecewise Sigma, and uses it for every rate factor that needs
    calibration."""
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
        "compute_greeks": COMPUTE_GREEKS,
    }


# =============================================================================
# STAGE 4 -- SUBMIT AND PRINT
# Send stage 3's request to stage 2's server, poll the async job
# (docs/reference/http-api.md#why-async-not-sync), and print the result.
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
        time.sleep(2.0)

    if status["status"] == "failed":
        raise RuntimeError(f"pricing job failed:\n{status['error']}")
    return status["result"]


def print_result(result: dict) -> None:
    trade_names = [t["trade_type"] for t in PORTFOLIO_TRADES]
    npv_cube = result["npv_cube"]  # [Scenarios, TimeSteps, Trades]
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
        print(f"warnings: {result['warnings']}")

    print("\nexposure profile:")
    exposure = result["exposure"]
    columns = ["epe", "ene", "ee_b"] + list(exposure["pfe"])
    print("  time  " + "".join(f"{c.upper():>12}" for c in columns))
    for i, t in enumerate(exposure["times"]):
        values = [exposure[c][i] if c in exposure else exposure["pfe"][c][i] for c in columns]
        print(f"  {t:>4.2f}" + "".join(f"{v:>12,.0f}" for v in values))
    print("(netting set; EPE/ENE/PFE discounted to today, EE_B undiscounted -- ORE's definitions)")

    if result["greeks"] is not None:
        bermudan_index = str(trade_names.index("bermudan_swaption"))
        greeks = result["greeks"][bermudan_index]
        print("\nBermudan Greeks:")
        print(f"  delta per pillar: {[round(v, 2) for v in greeks['values']['delta']]}")
        print(f"  gamma per pillar: {[round(v, 4) for v in greeks['values']['gamma']]}")
        print(f"  theta (1-day decay): {greeks['theta']:,.2f}")


# =============================================================================
# Run the four stages, in order.
# =============================================================================

def main() -> None:
    print("=== stage 1: given inputs ===")
    print(f"{NUM_SCENARIOS:,} scenarios, {len(PORTFOLIO_TRADES)} trades, "
          f"flat {FLAT_RATE:.2%} curve, evaluation date {EVALUATION_DATE}")

    print("\n=== stage 2: server setup ===")
    server_process = start_server() if _MANAGE_SERVER else None
    if server_process is None:
        wait_until_healthy()
        print(f"reusing an already-running server at {API_BASE}")
        if PROFILE_DIR:
            print("note: profiling is only active if THAT server was itself "
                  "started with JAX_RISK_PROFILE_DIR set -- this script can't "
                  "set the environment of a server it didn't launch")
    elif PROFILE_DIR:
        print(f"pricing-job profiler ON -> traces in {PROFILE_DIR!r} "
              f"(view: xprof --port 8791 {PROFILE_DIR})")

    try:
        print("\n=== stage 3: server inputs ===")
        request_body = build_portfolio_request_schema()
        print(f"built a PortfolioRequestSchema body: {len(request_body['trades'])} trades, "
              f"calibration_basket for {len(CALIBRATION_EXERCISE_TIMES)} market vol quotes")

        print("\n=== stage 4: submit and print ===")
        result = submit_and_wait(request_body)
        print_result(result)
    finally:
        if server_process is not None:
            stop_server(server_process)


if __name__ == "__main__":
    main()
