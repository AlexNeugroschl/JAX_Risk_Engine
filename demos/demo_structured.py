"""
The same portfolio as `demo_api.py`, restructured into four clearly separated
stages, in the order a real integrator actually encounters them:

  1. GIVEN INPUTS       -- raw market/portfolio facts nobody derives; they
                            come from wherever real data comes from (a
                            market-data feed, a trade blotter, a calibration
                            desk). Plain Python values, no server/schema
                            concepts at all.
  2. SERVER SETUP        -- getting a running engine.api server to talk to.
                            Infrastructure, not data -- would look identical
                            for a completely different portfolio. This demo
                            launches that server with the pricing-job
                            profiler ON by default (JAX_RISK_PROFILE_DIR set),
                            so each priced job writes an XProf/TensorBoard-
                            profiler trace -- see PROFILE_DIR below.
  3. SERVER INPUTS        -- translating stage 1's given inputs into the
                            exact JSON shape engine.api.schemas.
                            PortfolioRequestSchema expects. This is where
                            "market fact" becomes "wire format."
  4. SUBMIT AND PRINT     -- send stage 3's payload to stage 2's server, poll
                            for completion, and print the result.

`demo.py` (direct Python call) and `demo_api.py` (a runnable, single-flow
HTTP walkthrough) already cover "how do I call this system." This script's
own purpose is different: it's about *where the line falls* between what a
caller must supply (stage 1), what's just deployment mechanics (stage 2),
and what's translation boilerplate (stage 3) -- useful as a template to copy
from when wiring up a real integration, not just a demo to run once.

Run with: venv/Scripts/python.exe demos/demo_structured.py
"""
import os
import subprocess
import sys
import time

import httpx

# =============================================================================
# STAGE 1 -- GIVEN INPUTS
#
# Everything below this line is a market/portfolio FACT: something a real
# caller would receive from a market-data feed, a trade blotter, or a
# calibration desk -- not something this script derives or that depends in
# any way on how the server is run or what shape its API expects. If you
# swapped the HTTP layer for a direct Python call (demo.py) or a message
# queue, every value in this section would stay exactly the same.
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

# The portfolio itself: one swap, one European swaption, one Bermudan
# swaption, one American swaption -- one of every instrument type this
# engine prices. The two swaption-family trades below are deliberately
# UNCALIBRATED (hw_sigma is left out / null): their volatility comes from
# market swaption quotes, calibrated server-side -- see the calibration
# basket further down, which is exactly the "market quotes" input that
# calibration step needs.
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
        # initial_zero_curve filled in at stage 3 -- every trade here shares
        # the one curve defined above, so it's not repeated per-trade in the
        # "given inputs" section.
    },
    {
        "trade_type": "bermudan_swaption",
        "notional": 1_000_000.0, "fixed_rate": 0.030, "payer": True,
        "rate_factor_index": 0, "hw_a": HW_MEAN_REVERSION, "hw_sigma": None,
        "exercise_times": [1.0, 2.0, 3.0, 4.0], "swap_tenor": "5Y",
        "n_per_std": 64, "std_devs": 6.0,
    },
    {
        "trade_type": "american_swaption",
        "notional": 800_000.0, "fixed_rate": 0.029, "payer": False,
        "rate_factor_index": 0, "hw_a": HW_MEAN_REVERSION, "hw_sigma": None,
        "first_exercise": 1.0, "last_exercise": 4.0, "exercise_time_steps_per_year": 2,
        "n_per_std": 64, "std_devs": 6.0,
    },
]

# Market swaption volatility quotes -- what a calibration desk hands over to
# fit the Bermudan/American trades' own hw_sigma above. One quote per
# exercise date, co-terminal at the Bermudan's own final maturity (5Y).
CALIBRATION_EXERCISE_TIMES = [1.0, 2.0, 3.0, 4.0]
CALIBRATION_FINAL_MATURITY = 5.0
CALIBRATION_MARKET_VOLS = [0.0080, 0.0088, 0.0095, 0.0100]

# Risk parameters: which VaR/ES confidence levels to report, and whether to
# also compute Greeks.
RISK_PERCENTILES = [0.95, 0.99]
COMPUTE_GREEKS = True


# =============================================================================
# STAGE 2 -- SERVER SETUP
#
# Pure infrastructure: getting an engine.api server up and reachable. Would
# be identical for any other portfolio -- nothing below this line reads any
# of stage 1's values. See docs/reference/http-api.md for what the server
# actually does; see docs/getting-started/user-guide.md#running-the-api for
# running one by hand instead of letting this script launch/manage it.
# =============================================================================

API_BASE = "http://127.0.0.1:8000"
_MANAGE_SERVER = os.environ.get("JAX_RISK_ENGINE_DEMO_SKIP_SERVER") != "1"

# This demo turns the pricing-job profiler ON by default: the uvicorn server it
# launches below is started with JAX_RISK_PROFILE_DIR set, so every job the
# worker pool runs is wrapped in jax.profiler.trace (see
# engine/portfolio/worker_pool.py::_run_pricing_job) and writes an
# XProf/TensorBoard-profiler trace here -- one pid-<worker-pid>/ subdir per pool
# worker. Requires the `profiling` extra (`pip install -e .[api,profiling]`).
# Open the timeline with:  xprof --port 8791 <dir>
# Override the location from the environment, or set it to "" to opt out.
PROFILE_DIR = os.environ.get("JAX_RISK_PROFILE_DIR", ".profile-out")


def wait_until_healthy(timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if httpx.get(f"{API_BASE}/health", timeout=2.0).status_code == 200:
                return
        except httpx.TransportError:
            # "nothing listening yet" or "still importing JAX/ORE/routing" --
            # both expected while uvicorn is starting up.
            pass
        time.sleep(0.5)
    raise RuntimeError(f"server at {API_BASE} did not become healthy within {timeout_s}s")


def start_server() -> subprocess.Popen:
    # Pass the profiler dir explicitly to the server's environment (rather than
    # relying on implicit inheritance) so PROFILE_DIR's own default applies even
    # when this script's caller didn't set JAX_RISK_PROFILE_DIR themselves. The
    # spawned ProcessPoolExecutor workers inherit it from uvicorn in turn.
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
#
# Translating stage 1's given inputs into the exact JSON shape
# engine.api.schemas.PortfolioRequestSchema expects (see
# docs/reference/http-api.md#request-schema-portfoliorequestschema). This is
# the only stage where "a market fact" becomes "a field in a wire-format
# request" -- everything here is mechanical reshaping, nothing here is a new
# fact about the market or the portfolio.
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
            # maturities left unset -- the server derives the swap's real
            # cashflow-pillar set automatically from the trades below.
        },
        "joint_covariance": [[0.04, 0.0], [0.0, HW_SHORT_RATE_VOL ** 2]],
    }


def build_trades_schema(zero_curve: dict) -> list:
    """Stamps every trade with the one shared zero_curve from stage 1 --
    PortfolioRequestSchema wants it inline per-trade even though, for this
    portfolio, every trade happens to share the same curve."""
    trades = []
    for trade in PORTFOLIO_TRADES:
        trade = dict(trade)
        if trade["trade_type"] != "swap":
            trade["initial_zero_curve"] = zero_curve
        trades.append(trade)
    return trades


def build_calibration_basket_schema() -> dict:
    """Resolves the two trades above with hw_sigma=null: the server builds a
    co-terminal calibration basket from these market vols against the first
    uncalibrated trade's own curve/hw_a, fits a piecewise Sigma, and reuses
    it for every trade sharing that rate factor."""
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
        "compute_greeks": COMPUTE_GREEKS,
    }


# =============================================================================
# STAGE 4 -- SUBMIT AND PRINT
#
# Send stage 3's request to stage 2's server, poll POST /portfolio/price's
# async job to completion (see docs/reference/http-api.md#why-async-not-sync
# for why this polls instead of blocking), and print the result.
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

    print("\nrisk:")
    risk = result["risk"]["values"]
    print("  time   " + "".join(f"{m:>12}" for m in risk))
    for i, t in enumerate(TIME_GRID_YEARS[1:]):
        row = "".join(
            f"{risk[m][i]:>12,.0f}" if risk[m][i] is not None else f"{'nan':>12}"
            for m in risk
        )
        print(f"  {t:>4.2f}  " + row)
    print("(nan = the loss tail was empty at that step, matching ORE's own edge case)")

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
