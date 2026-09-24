"""
End-to-end walkthrough of the same portfolio `demo.py` prices, but over the
real HTTP API instead of calling `engine.portfolio.price_portfolio` directly
in-process -- exercises exactly what an external caller (e.g. TraderX) would
actually do: build a JSON request matching `PortfolioRequestSchema`, submit
it to a running server, poll the async job until it completes, and read the
result back out of `PortfolioResultSchema`'s JSON shape.

Portfolio: one swap, one European swaption, one Bermudan swaption, one
American swaption -- the same instruments and market data as `demo.py`,
so the two demos' printed NPVs/risk/Greeks are directly comparable.

This script starts its own `uvicorn` server as a subprocess so it can be run
standalone; if a server is already running at `API_BASE`, set
`JAX_RISK_ENGINE_DEMO_SKIP_SERVER=1` to reuse it instead of starting a new
one (useful when iterating with `--reload` already running in another
terminal).

See docs/reference/http-api.md for the full endpoint reference and the
async job pattern's reasoning (why this polls instead of blocking).

Run with: .venv/Scripts/python.exe demos/demo_api.py
"""
import os
import subprocess
import sys
import time

import httpx

API_BASE = "http://127.0.0.1:8000"
_START_SERVER = os.environ.get("JAX_RISK_ENGINE_DEMO_SKIP_SERVER") != "1"


def section(title: str) -> None:
    print(f"\n--- {title} ---")


def _wait_for_server(timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(f"{API_BASE}/health", timeout=2.0)
            if r.status_code == 200:
                return
        except httpx.TransportError:
            # Covers both "nothing listening yet" (ConnectError) and "server
            # accepted the TCP connection but hasn't finished importing JAX/
            # ORE/routing yet" (ConnectTimeout/ReadTimeout) -- both are
            # expected while uvicorn is still starting up.
            pass
        time.sleep(0.5)
    raise RuntimeError(f"server at {API_BASE} did not become healthy within {timeout_s}s")


# =============================================================================
# Start the server (unless the caller already has one running).
# =============================================================================
server_process = None
if _START_SERVER:
    section("Starting the API server")
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "engine.api.app:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"launched uvicorn (pid {server_process.pid}), waiting for {API_BASE}/health ...")
    try:
        _wait_for_server()
    except Exception:
        server_process.terminate()
        raise
    print("server is up")
else:
    section("Reusing an already-running server")
    _wait_for_server()
    print(f"found a healthy server at {API_BASE}")

try:
    # =========================================================================
    # /version -- confirm what we're actually talking to.
    # =========================================================================
    section("Version")
    version = httpx.get(f"{API_BASE}/version", timeout=30.0).json()
    print(f"engine {version['engine_version']}, JAX backend {version['jax_backend']}, "
          f"commit {version['git_commit'] or '(not a git checkout)'}")

    # =========================================================================
    # Build the request body: the same market/portfolio as demo.py, expressed
    # as PortfolioRequestSchema JSON instead of Python dataclasses.
    # =========================================================================
    section("Building the request")

    EVAL_DATE = "2026-07-30"
    FLAT_RATE = 0.03
    HW_A = 0.03
    HW_SIGMA = 0.01

    zero_curve = {"times": [0.0, 1.0, 2.0, 5.0, 10.0, 30.0], "rates": [FLAT_RATE] * 6}

    request_body = {
        "evaluation_date": EVAL_DATE,
        "market": {
            "time_grid": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
            "scenarios": 4096,
            "equities": {"initial_prices": [100.0], "dividend_yields": [0.0], "rate_mapping": [[0.0]]},
            "rates": {
                "initial_rates": [FLAT_RATE], "theta": [FLAT_RATE], "mean_reversion": [HW_A],
                "initial_zero_curves": [zero_curve],
                # maturities left unset -- the server derives the swap's real
                # cashflow-pillar set automatically, same as demo.py.
            },
            "joint_covariance": [[0.04, 0.0], [0.0, HW_SIGMA ** 2]],
        },
        "trades": [
            {
                "trade_type": "swap",
                "notional": 2_000_000.0, "fixed_rate": 0.032, "payer": True,
                "discount_curve_index": 0, "forward_curve_index": 0, "swap_tenor": "3Y",
            },
            {
                "trade_type": "european_swaption",
                "notional": 1_500_000.0, "fixed_rate": 0.031, "payer": True,
                "rate_factor_index": 0, "hw_a": HW_A, "hw_sigma": HW_SIGMA,
                "initial_zero_curve": zero_curve, "swap_tenor": "3Y", "forward_start": "2Y",
            },
            {
                "trade_type": "bermudan_swaption",
                "notional": 1_000_000.0, "fixed_rate": 0.030, "payer": True,
                "rate_factor_index": 0, "hw_a": HW_A, "hw_sigma": None,  # null -> server calibrates
                "initial_zero_curve": zero_curve, "exercise_dates": ["2027-07-30", "2028-07-30", "2029-07-30", "2030-07-30"],
                "swap_tenor": "5Y", "n_per_std": 64, "std_devs": 6.0,
            },
            {
                "trade_type": "american_swaption",
                "notional": 800_000.0, "fixed_rate": 0.029, "payer": False,
                "rate_factor_index": 0, "hw_a": HW_A, "hw_sigma": None,
                "initial_zero_curve": zero_curve,
                "first_exercise_date": "2027-07-30", "last_exercise_date": "2030-07-30",
                "exercise_time_steps_per_year": 2, "n_per_std": 64, "std_devs": 6.0,
            },
        ],
        "percentiles": [0.95, 0.99],
        # Resolves both trades' hw_sigma=null above: the server builds a
        # co-terminal calibration basket from these inputs (the same
        # inputs build_coterminal_basket itself takes -- see
        # CalibrationBasketRequestSchema) against the first uncalibrated
        # trade's own curve/hw_a, fits a piecewise Sigma to market_vols,
        # and reuses it for every trade sharing that rate factor.
        "calibration_basket": {
            "exercise_times": [1.0, 2.0, 3.0, 4.0],
            "final_maturity_time": 5.0,
            "notional": 1_000_000.0,
            "payer": True,
            "market_vols": [0.0080, 0.0088, 0.0095, 0.0100],
        },
        "compute_greeks": True,
    }
    print("4 trades: swap, european_swaption, bermudan_swaption, american_swaption "
          "(Bermudan/American hw_sigma=null -> server-side calibration via calibration_basket)")

    # =========================================================================
    # Calibration preview via the standalone /calibration/lgm endpoint --
    # same basket inputs as request_body["calibration_basket"] above, fetched
    # standalone so the fitted Sigma is visible before submitting the full
    # portfolio request. (Fitting happens again, redundantly, inside
    # price_portfolio when the portfolio request below is submitted --
    # this call is purely illustrative, not required.)
    # =========================================================================
    section("Calibration (standalone /calibration/lgm preview)")

    calibration_body = {
        "evaluation_date": EVAL_DATE,
        "exercise_times": [1.0, 2.0, 3.0, 4.0],
        "final_maturity_time": 5.0,
        "notional": 1_000_000.0,
        "payer": True,
        "market_vols": [0.0080, 0.0088, 0.0095, 0.0100],
        "zero_curve": zero_curve,
        "hw_a": HW_A,
    }
    calibration = httpx.post(f"{API_BASE}/calibration/lgm", json=calibration_body, timeout=30.0)
    calibration.raise_for_status()
    calibration_result = calibration.json()
    print(f"calibrated sigma per bucket: {[round(v, 5) for v in calibration_result['sigma_values']]}")
    print(f"reprice RMSE (should be ~0): {calibration_result['rmse']:.2e}")

    # =========================================================================
    # Submit the portfolio and poll until it's done.
    # =========================================================================
    section("Submitting the portfolio")

    submit = httpx.post(f"{API_BASE}/portfolio/price", json=request_body, timeout=30.0)
    if submit.status_code != 202:
        raise RuntimeError(f"submission failed ({submit.status_code}): {submit.text}")
    job_id = submit.json()["job_id"]
    print(f"job_id: {job_id} (202 Accepted -- pricing is running in the background)")

    section("Polling for the result")
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

    result = status["result"]

    # =========================================================================
    # Print the result -- same shape as demo.py's own output.
    # =========================================================================
    section("Pricing result")

    TRADE_NAMES = ["swap", "european", "bermudan", "american"]
    TIME_GRID = request_body["market"]["time_grid"]
    npv_cube = result["npv_cube"]  # [Scenarios, TimeSteps, Trades]

    print("mean NPV across scenarios, at each simulated time step:")
    print("  time   " + "".join(f"{n:>12}" for n in TRADE_NAMES))
    num_scenarios = len(npv_cube)
    for i, t in enumerate(TIME_GRID[1:]):
        means = [
            sum(npv_cube[s][i][j] for s in range(num_scenarios)) / num_scenarios
            for j in range(len(TRADE_NAMES))
        ]
        print(f"  {t:>4.2f}  " + "".join(f"{m:>12,.0f}" for m in means))

    print(f"\nbaseline portfolio NPV: {result['base_npv']:,.2f}")
    if result["warnings"]:
        print(f"warnings: {result['warnings']}")

    section("Risk aggregation")
    risk = result["risk"]["values"]
    print("  time   " + "".join(f"{m:>12}" for m in risk))
    for i, t in enumerate(TIME_GRID[1:]):
        row = "".join(
            f"{risk[m][i]:>12,.0f}" if risk[m][i] is not None else f"{'nan':>12}"
            for m in risk
        )
        print(f"  {t:>4.2f}  " + row)
    print("(nan = the loss tail was empty at that step, matching ORE's own edge case)")

    section("Greeks")
    bermudan_index = str(TRADE_NAMES.index("bermudan"))
    bermudan_greeks = result["greeks"][bermudan_index]
    print(f"delta per pillar: {[round(v, 2) for v in bermudan_greeks['values']['delta']]}")
    print(f"gamma per pillar: {[round(v, 4) for v in bermudan_greeks['values']['gamma']]}")
    print(f"theta (1-day decay): {bermudan_greeks['theta']:,.2f}")

finally:
    if server_process is not None:
        section("Shutting down the API server")
        server_process.terminate()
        server_process.wait(timeout=10)
        print(f"stopped uvicorn (pid {server_process.pid})")
