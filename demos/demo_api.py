"""
The portfolio of `demo.py` over the HTTP API, through every route: build the portfolio request
as JSON (today's market, trades naming their currency, the run configuration with the Hull-White
model per currency); calibrate the cross-asset model (`POST /calibration/cam`, synchronous);
submit the portfolio (`POST /portfolio/price`), poll the job (`GET /jobs/{job_id}`) and read back
a `PortfolioResultSchema`, its cube by reference (each chunk fetched and its hash checked); then
the same trades' market-risk VaR and ES (`POST /portfolio/market-risk`). Same instruments,
market and configuration as `demo.py`, so the printed numbers are comparable.

Starts its own `uvicorn` server as a subprocess; set `JAX_RISK_ENGINE_DEMO_SKIP_SERVER=1` to
use one already running at `API_BASE`. Endpoint reference: docs/reference/http-api.md.

Run with: .venv/Scripts/python.exe demos/demo_api.py
"""
import hashlib
import os
import struct
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
    # On a GPU, XLA takes 75% of the card in every process that opens it; on a developer's
    # machine the engine worker shares it, so it grows on demand instead, unless your
    # environment says otherwise. The worker sets its other device settings itself.
    server_env = os.environ.copy()
    server_env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "engine.api.app:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=server_env,
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

def submit_and_wait(route: str, request_body: dict) -> dict:
    """Submit a job (`POST` to its route, `202` with a `job_id`) and poll `GET /jobs/{job_id}`
    until it reaches a final status; the result of a done job."""
    submit = httpx.post(f"{API_BASE}{route}", json=request_body, timeout=30.0)
    if submit.status_code != 202:
        raise RuntimeError(f"submission failed ({submit.status_code}): {submit.text}")
    job_id = submit.json()["job_id"]
    print(f"job_id: {job_id} (202 Accepted -- running in the background)")
    start = time.time()
    while True:
        poll = httpx.get(f"{API_BASE}/jobs/{job_id}", timeout=30.0)
        poll.raise_for_status()
        status = poll.json()
        print(f"  [{time.time() - start:6.1f}s] {status['kind']} job: {status['status']}")
        if status["status"] in ("done", "failed", "interrupted"):  # the final statuses
            break
        time.sleep(2.0)
    if status["status"] != "done":
        raise RuntimeError(f"{status['kind']} job {status['status']}:\n{status['error']}")
    return status["result"]


def read_artifact(reference: dict) -> list:
    """An array returned by reference: fetch each chunk, check its sha256 and the whole array's,
    and unpack the little-endian float64 values in C order (nested lists of `shape`)."""
    whole, values = hashlib.sha256(), []
    for chunk in reference["chunks"]:
        data = httpx.get(f"{API_BASE}{chunk['url']}", timeout=30.0).content
        if hashlib.sha256(data).hexdigest() != chunk["sha256"]:
            raise RuntimeError(f"chunk {chunk['url']} does not match its sha256")
        whole.update(data)
        values += struct.unpack(f"<{len(data) // 8}d", data)
    if whole.hexdigest() != reference["sha256"] or reference["dtype"] != "float64":
        raise RuntimeError(f"{reference['name']} does not match its reference")
    for size in reversed(reference["shape"][1:]):
        values = [values[i:i + size] for i in range(0, len(values), size)]
    return values


def factor_labels(market: dict) -> list:
    """The market's risk factors, the order the scenarios' covariance follows: per currency its
    discount curve's pillars, then each index curve's by name, as `"<curve>/<pillar time>y"`."""
    labels = []
    for code, data in market["currencies"].items():
        curves = [(f"discount:{code}", data["discount_curve"])]
        curves += [(f"index:{name}", data["index_curves"][name]) for name in sorted(data["index_curves"])]
        labels += [f"{name}/{t:g}y" for name, curve in curves for t in curve["times"]]
    return labels


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
    market = demo_market_json(("USD",))
    trades = [
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
    ]
    request_body = {
        "market": market,
        "trades": trades,
        # The run configuration: the Hull-White model for USD, calibrated to the market's
        # swaption volatilities; ORE's default engines (a coarser Bermudan/American grid);
        # Greeks by AD (ORE's bump-and-revalue, the default, recalibrates the options under
        # every bump, which takes minutes here: I-53).
        "simulation": demo_simulation_json("HullWhite", samples=128, currencies=("USD",), calibrated=True),
        "pricing": {"bermudan": engine, "american": engine},
        "greeks": {"method": "AD"},
        "pfe_quantiles": [0.95, 0.99],
        "compute_greeks": True,
        # The cube by reference rather than inline: chunks of bytes, each hashed (decision A-17).
        "cube_output": "artifact",
    }
    print(", ".join(t["trade_id"] for t in trades) + f"; model {request_body['simulation']['ir']['USD']['model']}")

    # =========================================================================
    # The cross-asset model's calibration: what the simulation below will run with.
    # =========================================================================
    section("Calibrating the cross-asset model (POST /calibration/cam)")
    calibration = httpx.post(f"{API_BASE}/calibration/cam", json={"market": market,
                                                                 "ir": request_body["simulation"]["ir"]},
                             timeout=120.0)
    if calibration.status_code != 200:
        raise RuntimeError(f"calibration failed ({calibration.status_code}): {calibration.text}")
    for code, fitted in calibration.json()["currencies"].items():
        print(f"{code} {fitted['model']}, reversion {fitted['reversion']}: sigma {fitted['sigma_values']} "
              f"after {fitted['sigma_times']}")
        for helper in fitted["helpers"]:
            print(f"  {helper['expiry']:>3} x {helper['term']:<3} market {helper['market_value']:.8f}, "
                  f"model {helper['model_value']:.8f}")

    # =========================================================================
    # Submit the portfolio and poll until it's done.
    # =========================================================================
    section("Pricing the portfolio (POST /portfolio/price)")
    result = submit_and_wait("/portfolio/price", request_body)

    # =========================================================================
    # Print the result -- same shape as demo.py's own output.
    # =========================================================================
    section("Pricing result")

    rows = result["trades"]  # one row per trade, in request order: the cube's trade axis
    trade_ids = [row["trade_id"] for row in rows]
    npv_cube = read_artifact(result["npv_cube_artifact"])  # [Scenarios, Dates, Trades]
    num_paths = len(npv_cube)
    print(f"the cube by reference: {result['npv_cube_artifact']['shape']}, "
          f"{len(result['npv_cube_artifact']['chunks'])} chunk(s), every sha256 checked")
    print("mean NPV across paths, at each simulation date:")
    print("  time  " + "".join(f"{n:>12}" for n in trade_ids))
    print("  0.00  " + "".join(f"{row['base_npv']:>12,.0f}" for row in rows))
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
    bermudan = next(row for row in rows if row["trade_id"] == "bermudan")["greeks"]
    for key, values in bermudan["values"].items():
        print(f"  {key}: {[round(v, 2) for v in values]}")
    print(f"  theta: {bermudan['theta']:,.2f}")

    # =========================================================================
    # Market risk: 10-day VaR and ES by full revaluation under Monte Carlo shocks of every
    # curve pillar, the covariance given in the market's factor order.
    # =========================================================================
    section("Market risk (POST /portfolio/market-risk)")
    labels = factor_labels(market)
    daily_vol, horizon_days = 0.0008, 10
    covariance = [[daily_vol ** 2 * horizon_days * 0.9 ** abs(i - j) for j in range(len(labels))]
                  for i in range(len(labels))]
    risk = submit_and_wait("/portfolio/market-risk", {
        "market": market, "trades": [t for t in trades if t["trade_type"] in ("swap", "european_swaption", "bond")],
        "scenarios": {"source": "monte-carlo", "factors": labels, "covariance": covariance,
                      "horizon_days": horizon_days, "num_scenarios": 1024, "seed": 1},
        "quantiles": [0.99, 0.975], "pnl_output": "none"})
    print(f"{risk['num_scenarios']} {risk['source']} scenarios of {len(risk['risk_factors'])} factors, "
          f"{risk['horizon_days']}-day horizon ({risk['measure']})")
    for key in ("VaR_99", "ES_97.5", "ES_97.5_standardError"):
        print(f"  {key}: {risk['risk'][key]:,.2f}")
    for warning in risk["warnings"]:
        print(f"  note: {warning}")

finally:
    if server_process is not None:
        section("Shutting down the API server")
        server_process.terminate()
        server_process.wait(timeout=10)
        print(f"stopped uvicorn (pid {server_process.pid})")
