"""
The portfolio of `demo_api.py`, split into the four stages an integrator meets:

  1. GIVEN INPUTS     market and portfolio facts (curves, trades, vol quotes); plain Python
                      values with no server or schema concepts.
  2. SERVER SETUP     starting an engine.api server, with the pricing-job profiler on
                      (JAX_RISK_PROFILE_DIR; see PROFILE_DIR below). Same for any portfolio.
  3. SERVER INPUTS    reshaping stage 1 into the portfolio request's JSON
                      (`engine.api.requests.MarketPortfolioRequestSchema`).
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

# Today's USD market: zero curves for discounting and for the 6M index's forwards (rising
# from 3% to 5%), and ATM normal swaption volatilities. The same numbers as demo_scenarios.py.
CURVE_TIMES = [0.0, 1.0, 2.0, 5.0, 10.0, 30.0]
DISCOUNT_RATES = [0.030, 0.030, 0.034, 0.040, 0.046, 0.050]
INDEX_RATES = [0.034, 0.034, 0.038, 0.044, 0.049, 0.052]
INDEX_NAME = "USD-SIMINDEX-6M"
VOL_OPTION_TENORS = ["1Y", "2Y", "5Y", "10Y"]
VOL_SWAP_TENORS = ["1Y", "5Y", "10Y"]
VOLS = [[0.0080, 0.0088, 0.0090], [0.0085, 0.0091, 0.0093], [0.0090, 0.0093, 0.0094], [0.0092, 0.0094, 0.0096]]

# The model of USD rates: Hull-White, mean reversion 3%, its short-rate volatility calibrated
# to a 1Y/2Y/5Y co-terminal basket of those swaption quotes (ORE's CalibrationSwaptions).
MODEL = "HullWhite"
MEAN_REVERSION = 0.03
CALIBRATION_EXPIRIES = ["1Y", "2Y", "5Y"]
CALIBRATION_TERMS = ["9Y", "8Y", "5Y"]

# Monte Carlo settings: how many alternate futures, and at which dates to value the portfolio.
NUM_PATHS = 128
SIMULATION_DATES = ["2026-10-30", "2027-01-30", "2027-04-30", "2027-07-30", "2028-07-30", "2029-07-30",
                    "2030-07-30", "2031-07-30"]

# The portfolio: one of each trade type, each with an id. A trade names its currency (USD by
# default); the curves and the model are the market's and the run's, not the trade's.
PORTFOLIO_TRADES = [
    {"trade_type": "swap", "trade_id": "swap", "notional": 2_000_000.0, "fixed_rate": 0.036, "payer": True,
     "swap_tenor": "3Y"},
    {"trade_type": "european_swaption", "trade_id": "european", "notional": 1_500_000.0, "fixed_rate": 0.042,
     "payer": True, "swap_tenor": "3Y", "forward_start": "2Y"},
    {"trade_type": "bermudan_swaption", "trade_id": "bermudan", "notional": 1_000_000.0, "fixed_rate": 0.042,
     "payer": True, "swap_tenor": "5Y",
     "exercise_dates": ["2027-07-30", "2028-07-30", "2029-07-30", "2030-07-30"]},
    {"trade_type": "american_swaption", "trade_id": "american", "notional": 800_000.0, "fixed_rate": 0.040,
     "payer": False, "swap_tenor": "5Y", "first_exercise_date": "2027-07-30", "last_exercise_date": "2030-07-30"},
]

# Risk parameters: which PFE quantiles the exposure profile reports, whether to also compute
# Greeks, and how: "AD" (ORE's "Bump", the default, recalibrates the options under every bump,
# which takes minutes for this portfolio: I-53).
RISK_PERCENTILES = [0.95, 0.99]
COMPUTE_GREEKS = True
GREEKS_METHOD = "AD"


# =============================================================================
# STAGE 2 -- SERVER SETUP
# Infrastructure only; nothing here reads stage 1. See docs/reference/http-api.md, and
# docs/getting-started/user-guide.md#running-the-api to run a server by hand.
# =============================================================================

API_BASE = "http://127.0.0.1:8000"
_MANAGE_SERVER = os.environ.get("JAX_RISK_ENGINE_DEMO_SKIP_SERVER") != "1"

# The profiler is on by default: the server starts with JAX_RISK_PROFILE_DIR set, so each job
# the engine worker runs is wrapped in jax.profiler.trace (engine/api/worker.py::_profiled),
# in a pid-<worker-pid>/ subdirectory. Needs the `profiling` extra (`pip install -e .[api,profiling]`). View with
# `xprof --port 8791 <dir>`. Override from the environment, or set "" to opt out.
#
# Three more knobs, read by _profiled and passed through below:
#   JAX_RISK_PROFILE_WARMUP=1        run the job once untraced first, so the trace shows
#       warm execution rather than compilation (the job runs twice). Off by default.
#   JAX_RISK_PROFILE_PHASE=<phase>   trace only that phase of the job (e.g. pricing, or
#       greeks/trade3/AmericanSwaptionConfig), the rest untraced. Unset by default.
#   JAX_RISK_PROFILE_PYTHON_TRACER=1 turn on JAX's Python tracer (JAX defaults it on; off
#       here). Off still keeps compilation, dispatch, tracing and execution; it drops only
#       CPython frames. Measured on this portfolio: on, CPython frames were 97% of events
#       and the trace's .trace.json.gz covered the first 1.6s of a ~90s job (469MB vs 50MB).
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
    # did not set JAX_RISK_PROFILE_DIR; the engine worker inherits it from uvicorn.
    env = os.environ.copy()
    if PROFILE_DIR:
        env["JAX_RISK_PROFILE_DIR"] = PROFILE_DIR
    # On a GPU, XLA takes 75% of the card in every process that opens it. On a developer's
    # machine the engine worker shares the card (with a notebook, the tests), so it grows on
    # demand instead, unless your environment says otherwise. A deployment with one engine
    # worker per GPU keeps XLA's default. The worker sets its other device settings itself.
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
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
# Stage 1 reshaped into the portfolio request (docs/reference/http-api.md); no new facts.
# =============================================================================

def build_market() -> dict:
    curve = lambda rates: {"times": CURVE_TIMES, "rates": rates}  # noqa: E731
    return {"asof": EVALUATION_DATE, "currencies": {"USD": {
        "discount_curve": curve(DISCOUNT_RATES), "index_curves": {INDEX_NAME: curve(INDEX_RATES)},
        "swaption_vols": {"option_tenors": VOL_OPTION_TENORS, "swap_tenors": VOL_SWAP_TENORS, "vols": VOLS}}}}


def build_simulation() -> dict:
    return {"dates": SIMULATION_DATES, "base_currency": "USD", "samples": NUM_PATHS,
            "ir": {"USD": {"model": MODEL, "reversion": MEAN_REVERSION, "calibration_expiries": CALIBRATION_EXPIRIES,
                           "calibration_terms": CALIBRATION_TERMS}}}


def build_portfolio_request() -> dict:
    coarse_grid = {"n_per_std": 16}  # a quicker Bermudan/American grid than the default 30
    return {
        "market": build_market(),
        "trades": PORTFOLIO_TRADES,
        "simulation": build_simulation(),
        "pricing": {"bermudan": coarse_grid, "american": coarse_grid},
        "pfe_quantiles": RISK_PERCENTILES,
        "compute_greeks": COMPUTE_GREEKS,
        "greeks": {"method": GREEKS_METHOD},
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
        poll = httpx.get(f"{API_BASE}/jobs/{job_id}", timeout=30.0)
        poll.raise_for_status()
        status = poll.json()
        print(f"  [{time.time() - start:6.1f}s] status: {status['status']}")
        if status["status"] in ("done", "failed", "interrupted"):  # the final statuses
            break
        time.sleep(2.0)

    if status["status"] != "done":
        raise RuntimeError(f"pricing job {status['status']}:\n{status['error']}")
    return status["result"]


def print_result(result: dict) -> None:
    rows = result["trades"]  # one row per trade, in request order: the cube's trade axis
    trade_ids = [row["trade_id"] for row in rows]
    npv_cube = result["npv_cube"]  # [Paths, Dates, Trades]
    num_paths = len(npv_cube)

    print("\nmean NPV across paths, at each simulation date:")
    print("  time  " + "".join(f"{n:>12}" for n in trade_ids))
    print("  0.00  " + "".join(f"{row['base_npv']:>12,.0f}" for row in rows))
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
    print(f"(netting set; ORE's ExposureCalculator definitions; measure: {result['measure']})")

    greeks = next(row for row in rows if row["trade_id"] == "bermudan")["greeks"]
    if greeks is not None:
        print(f"\nBermudan Greeks ({GREEKS_METHOD}):")
        for key, values in greeks["values"].items():
            print(f"  {key}: {[round(v, 2) for v in values]}")
        print(f"  theta (1-day): {greeks['theta']:,.2f}")


# =============================================================================
# Run the four stages, in order.
# =============================================================================

def main() -> None:
    print("=== stage 1: given inputs ===")
    print(f"{NUM_PATHS:,} paths, {len(PORTFOLIO_TRADES)} trades, the {MODEL} model for USD, "
          f"evaluation date {EVALUATION_DATE}")

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
        request_body = build_portfolio_request()
        print(f"built the portfolio request: {len(request_body['trades'])} trades, "
              f"{MODEL} calibrated to {len(CALIBRATION_EXPIRIES)} swaption quotes")

        print("\n=== stage 4: submit and print ===")
        result = submit_and_wait(request_body)
        print_result(result)
    finally:
        if server_process is not None:
            stop_server(server_process)


if __name__ == "__main__":
    main()
