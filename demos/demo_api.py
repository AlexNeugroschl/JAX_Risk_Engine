"""
The portfolio of `demo.py` priced over the HTTP API: build the portfolio request as JSON (today's
market, trades naming their currency, the run configuration with the Hull-White model per
currency), submit it, poll the async job, and read back a `PortfolioResultSchema`. Same
instruments, market and configuration as `demo.py`, so the printed numbers are comparable.

Starts its own `uvicorn` server as a subprocess; set `JAX_RISK_ENGINE_DEMO_SKIP_SERVER=1` to
use one already running at `API_BASE`. Endpoint reference: docs/reference/http-api.md.

Run with: .venv/Scripts/python.exe demos/demo_api.py
"""
import os
import subprocess
import sys
import time

import httpx

from demo_scenarios import demo_market_json, demo_simulation_json

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
            # ConnectError (nothing listening) or a timeout (still importing JAX/ORE)
            # are expected while uvicorn starts.
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
    # Build the request body: the same market, portfolio and configuration as demo.py,
    # as JSON instead of Python dataclasses.
    # =========================================================================
    section("Building the request")

    engine = {"n_per_std": 16}
    request_body = {
        "market": demo_market_json(("USD",)),
        "trades": [
            {"trade_type": "swap", "trade_id": "swap", "notional": 2_000_000.0, "fixed_rate": 0.036,
             "payer": True, "swap_tenor": "3Y"},
            {"trade_type": "european_swaption", "trade_id": "european", "notional": 1_500_000.0,
             "fixed_rate": 0.042, "payer": True, "swap_tenor": "3Y", "forward_start": "2Y"},
            {"trade_type": "bermudan_swaption", "trade_id": "bermudan", "notional": 1_000_000.0,
             "fixed_rate": 0.042, "payer": True, "swap_tenor": "5Y",
             "exercise_dates": ["2027-07-30", "2028-07-30", "2029-07-30", "2030-07-30"]},
            {"trade_type": "american_swaption", "trade_id": "american", "notional": 800_000.0,
             "fixed_rate": 0.040, "payer": False, "swap_tenor": "5Y",
             "first_exercise_date": "2027-07-30", "last_exercise_date": "2030-07-30"},
            {"trade_type": "bond", "trade_id": "note", "face_amount": 1_000_000.0, "maturity_date": "2029-02-15",
             "coupon_rate": 0.0375,
             "coupon_schedule": [{"start_date": s, "end_date": e} for s, e in (
                 ("2026-02-15", "2026-08-15"), ("2026-08-15", "2027-02-15"), ("2027-02-15", "2027-08-15"),
                 ("2027-08-15", "2028-02-15"), ("2028-02-15", "2028-08-15"), ("2028-08-15", "2029-02-15"))]},
        ],
        # The run configuration: the Hull-White model for USD, calibrated to the market's
        # swaption volatilities; ORE's default engines (a coarser Bermudan/American grid);
        # Greeks by AD (ORE's bump-and-revalue, the default, recalibrates the options under
        # every bump, which takes minutes here: I-53).
        "simulation": demo_simulation_json("HullWhite", samples=128, currencies=("USD",), calibrated=True),
        "pricing": {"bermudan": engine, "american": engine},
        "greeks": {"method": "AD"},
        "pfe_quantiles": [0.95, 0.99],
        "compute_greeks": True,
    }
    print(", ".join(t["trade_id"] for t in request_body["trades"]) +
          f"; model {request_body['simulation']['ir']['USD']['model']}")

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

    trade_ids = result["trade_ids"]
    npv_cube = result["npv_cube"]  # [Scenarios, Dates, Trades]
    num_paths = len(npv_cube)
    print("mean NPV across paths, at each simulation date:")
    print("  time  " + "".join(f"{n:>12}" for n in trade_ids))
    print("  0.00  " + "".join(f"{v:>12,.0f}" for v in result["base_npv_per_trade"]))
    for i, t in enumerate(result["exposure"]["times"][1:]):  # the cube's dates (times[0] is t=0)
        means = [sum(npv_cube[s][i][j] for s in range(num_paths)) / num_paths for j in range(len(trade_ids))]
        print(f"  {t:>4.2f}  " + "".join(f"{m:>12,.0f}" for m in means))
    print(f"\nportfolio NPV today: {result['base_npv']:,.2f}")

    section("Exposure profile")
    exposure = result["exposure"]
    columns = ["epe", "ene", "ee_b"] + list(exposure["pfe"])
    print("  time  " + "".join(f"{c.upper():>12}" for c in columns))
    for i, t in enumerate(exposure["times"]):
        values = [exposure[c][i] if c in exposure else exposure["pfe"][c][i] for c in columns]
        print(f"  {t:>4.2f}" + "".join(f"{v:>12,.0f}" for v in values))
    print(f"(netting set; ORE's ExposureCalculator definitions; measure: {result['measure']})")

    section("Greeks (Bermudan, automatic differentiation)")
    bermudan = result["greeks"][str(trade_ids.index("bermudan"))]
    for key, values in bermudan["values"].items():
        print(f"  {key}: {[round(v, 2) for v in values]}")
    print(f"  theta: {bermudan['theta']:,.2f}")

finally:
    if server_process is not None:
        section("Shutting down the API server")
        server_process.terminate()
        server_process.wait(timeout=10)
        print(f"stopped uvicorn (pid {server_process.pid})")
