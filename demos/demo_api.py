"""
The portfolio of `demo.py` over the HTTP API's one run request (`POST /runs`): build the request
as JSON (today's market, trades naming their currency, and a `config` whose sections follow ORE's
files, with the Hull-White model per currency); run today's values, the exposure, the Greeks and
the cross-asset model's calibration as one run, polled with `GET /runs/{run_id}?wait=30`, its cube
by reference (each chunk fetched and its hash checked); then the linear trades' market-risk VaR and
ES as a second run, submitted with an idempotency key so a retried submission is the same run.
Same instruments, market and configuration as `demo.py`, so the printed numbers are comparable.

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
import uuid

import httpx

from demo_scenarios import demo_config_json, demo_market_json

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


def run_and_wait(request_body: dict) -> dict:
    """Submit a run (`POST /runs`, `202` with a `run_id`) and long-poll `GET /runs/{run_id}?wait=30`
    until it is finished; the result of a done run. A pending run's `error` says why no engine
    worker is running it, if one cannot start."""
    submit = httpx.post(f"{API_BASE}/runs", json=request_body, timeout=30.0)
    if submit.status_code != 202:
        raise RuntimeError(f"submission refused ({submit.status_code}): {submit.text}")
    run_id = submit.json()["run_id"]
    print(f"run_id: {run_id} (202 Accepted -- running in the background)")
    start = time.time()
    while True:
        poll = httpx.get(f"{API_BASE}/runs/{run_id}", params={"wait": 30}, timeout=60.0)
        poll.raise_for_status()
        status = poll.json()
        print(f"  [{time.time() - start:6.1f}s] {status['status']}" + (f": {status['error']}" if status["error"] and
                                                                         status["status"] == "pending" else ""))
        if status["status"] not in ("pending", "running"):
            break
    if status["status"] != "done":
        raise RuntimeError(f"run {status['status']}:\n{status['error']}")
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
    engine_identity = httpx.get(f"{API_BASE}/version", timeout=30.0).json()["engine"]
    print(f"engine {engine_identity['version']} at {engine_identity['revision'] or '(not a git checkout)'}, "
          f"jax {engine_identity['jax']}, ORE {engine_identity['ore']}")

    # =========================================================================
    # Build the run request: the same market, portfolio and configuration as demo.py, as JSON.
    # =========================================================================
    section("Building the run request")

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
        "portfolio": trades,
        # What to compute: today's values, the exposure simulation, the Greeks, and the cross-asset
        # model's calibration (the one the simulation runs with), as ORE's ore.xml lists analytics.
        "analytics": ["npv", "exposure", "sensitivities", "calibration"],
        # How: the Hull-White model for USD, calibrated to the market's swaption volatilities
        # (`models`) on the simulation grid (`simulation`); ORE's default engines (a coarser
        # Bermudan/American grid); Greeks by AD (ORE's bump-and-revalue, the default, recalibrates
        # the options under every bump, which takes minutes here: I-53).
        "config": {**demo_config_json("HullWhite", samples=128, currencies=("USD",), calibrated=True),
                   "pricing": {"bermudan": engine, "american": engine},
                   "sensitivity": {"method": "AD"},
                   "exposure": {"pfe_quantiles": [0.95, 0.99]}},
        # The cube by reference (the default since the run request, decision A-17): chunks of
        # bytes, each hashed.
        "outputs": {"npv_cube": "artifact"},
    }
    print(", ".join(t["trade_id"] for t in trades) + f"; analytics {request_body['analytics']}; "
          f"model {request_body['config']['models']['ir']['USD']['model']}")

    # =========================================================================
    # Submit the run and wait until it's done.
    # =========================================================================
    section("Running (POST /runs)")
    result = run_and_wait(request_body)
    print(f"computed by engine {result['engine']['revision'] or result['engine']['version']} "
          f"(jax {result['engine']['jax']}); every default of the configuration as run is in result['config']")

    # =========================================================================
    # Print the result -- same shape as demo.py's own output.
    # =========================================================================
    section("The cross-asset model's calibration")
    for code, fitted in result["calibration"]["currencies"].items():
        print(f"{code} {fitted['model']}, reversion {fitted['reversion']}: sigma {fitted['sigma_values']} "
              f"after {fitted['sigma_times']} (error {fitted['error']:.2e})")
        for helper in fitted["helpers"]:
            print(f"  {helper['expiry']:>3} x {helper['term']:<3} market {helper['market_value']:.8f}, "
                  f"model {helper['model_value']:.8f}")

    section("Pricing result")
    rows = result["trades"]  # one row per trade, in request order: the cube's trade axis
    exposure = result["exposure"]
    cube = exposure["npv_cube"]
    npv_cube = read_artifact(cube)  # [scenario, date, trade]
    num_paths = len(npv_cube)
    print(f"the cube by reference: {cube['shape']} on axes {cube['axes']}, {len(cube['chunks'])} chunk(s), "
          f"every sha256 checked; trades {cube['items']['ids']}")
    print("mean NPV across paths, at each simulation date:")
    print("        date  " + "".join(f"{row['trade_id']:>12}" for row in rows))
    print(f"  {result['asof']}  " + "".join(f"{row['npv']:>12,.0f}" for row in rows))
    for i, date in enumerate(exposure["dates"]):
        means = [sum(npv_cube[s][i][j] for s in range(num_paths)) / num_paths for j in range(len(rows))]
        print(f"  {date}  " + "".join(f"{m:>12,.0f}" for m in means))
    print(f"\nportfolio NPV today: {result['npv']:,.2f} {result['base_currency']}")

    section("Exposure profile")
    netting_set = exposure["netting_set"]
    columns = ["epe", "ene", "ee_b"] + list(netting_set["pfe"])
    print("  time  " + "".join(f"{c.upper():>12}" for c in columns))
    for i, t in enumerate(netting_set["times"]):
        values = [netting_set[c][i] if c in netting_set else netting_set["pfe"][c][i] for c in columns]
        print(f"  {t:>4.2f}" + "".join(f"{v:>12,.0f}" for v in values))
    print(f"(netting set; ORE's ExposureCalculator definitions; measure: {exposure['measure']})")

    section(f"Greeks (Bermudan, {result['sensitivities']['method']})")
    factors = result["sensitivities"]["factors"]
    bermudan = next(row for row in rows if row["trade_id"] == "bermudan")["greeks"]
    for key, values in bermudan["values"].items():
        print(f"  {key}: " + ", ".join(f"{label} {v:,.2f}" for label, v in zip(factors[key], values)))
    print(f"  theta: {bermudan['theta']:,.2f}")

    # =========================================================================
    # Market risk: 10-day VaR and ES by full revaluation under Monte Carlo shocks of every
    # curve pillar, the covariance given in the market's factor order. Submitted with an
    # idempotency key: a client that loses the 202 and submits again gets the same run.
    # =========================================================================
    section("Market risk (POST /runs, analytics ['market_risk'])")
    labels = factor_labels(market)
    daily_vol, horizon_days = 0.0008, 10
    covariance = [[daily_vol ** 2 * horizon_days * 0.9 ** abs(i - j) for j in range(len(labels))]
                  for i in range(len(labels))]
    risk_body = {
        "market": market, "portfolio": [t for t in trades if t["trade_type"] in ("swap", "european_swaption", "bond")],
        "analytics": ["market_risk"],
        "config": {"market_risk": {"scenarios": {"source": "monte-carlo", "factors": labels,
                                                 "covariance": covariance, "horizon_days": horizon_days,
                                                 "num_scenarios": 1024, "seed": 1},
                                   "quantiles": [0.99, 0.975]}},
        "outputs": {"pnl": "none"},
        "idempotency_key": f"demo-market-risk-{uuid.uuid4()}",
    }
    first = httpx.post(f"{API_BASE}/runs", json=risk_body, timeout=30.0).json()["run_id"]
    retried = httpx.post(f"{API_BASE}/runs", json=risk_body, timeout=30.0).json()["run_id"]
    same = "the same run" if first == retried else "two runs"
    print(f"submitted twice with one idempotency key: {first} and {retried} ({same})")
    result = run_and_wait(risk_body)
    risk = result["market_risk"]
    print(f"{risk['num_scenarios']} {risk['source']} scenarios of {len(risk['risk_factors'])} factors, "
          f"{risk['horizon_days']}-day horizon ({risk['measure']}), in {result['base_currency']}")
    for key in ("VaR_99", "ES_97.5", "ES_97.5_standardError"):
        print(f"  {key}: {risk['risk'][key]:,.2f}" if risk["risk"][key] is not None else f"  {key}: unavailable")
    for warning in result["warnings"]:
        print(f"  note: {warning}")

finally:
    if server_process is not None:
        section("Shutting down the API server")
        server_process.terminate()
        server_process.wait(timeout=10)
        print(f"stopped uvicorn (pid {server_process.pid})")
