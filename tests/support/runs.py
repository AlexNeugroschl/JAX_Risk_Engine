"""
Run requests in the tests (I-87): bodies on the shared portfolio's market, submitted to
`POST /runs` and polled at `GET /runs/{run_id}` through an in-process `TestClient`, and the same
body run in-process (`engine.run.run`) to compare with.

    body = run_body(["npv", "exposure"], ["swap-payer"], simulation=simulation(), models=models())
    result = run_over_http(client, body)        # the result document, once done
    direct = run_in_process(body)               # engine.run.RunResult
"""
import copy
import time
from typing import Dict, Optional, Sequence

import ORE

from engine.api.requests import RunRequestSchema
from engine.run import RunRequest, RunResult, run
from tests.support import portfolio as shared


def models(model: str = "LGM", reversion: float = 0.03, volatility: float = 0.01, **fields) -> dict:
    """`config.models` with one USD model."""
    return {"ir": {"USD": {"model": model, "reversion": reversion, "volatility": volatility, **fields}}}


def simulation(samples: int = 64, months: Sequence[int] = (6, 12, 24), seed: int = 3, **fields) -> dict:
    """`config.simulation` on dates `months` after the shared as-of date."""
    return {"dates": [(shared.ASOF + ORE.Period(m, ORE.Months)).ISO() for m in months], "samples": samples,
            "seed": seed, **fields}


def run_body(analytics: Sequence[str], trades: Sequence = (), outputs: Optional[Dict[str, str]] = None,
             market: Optional[dict] = None, **config) -> dict:
    """A run request: `trades` the shared portfolio's by name, or bodies; `config` its sections.
    A copy, so a test may change it without changing what it was built from."""
    portfolio = [shared.trades_json()[t] if isinstance(t, str) else t for t in trades]
    body = {"market": market or shared.market_json(), "portfolio": portfolio, "analytics": list(analytics),
            "config": config}
    if outputs is not None:
        body["outputs"] = outputs
    return copy.deepcopy(body)


def submit(client, body: dict) -> str:
    """The `run_id` of `body` submitted to `POST /runs`, which must accept it."""
    response = client.post("/runs", json=body)
    assert response.status_code == 202, response.text
    return response.json()["run_id"]


def poll(client, run_id: str, timeout_s: float = 600.0) -> dict:
    """The run's status once it is finished (`?wait=` long polls, so no sleep loop)."""
    deadline = time.monotonic() + timeout_s
    while True:
        response = client.get(f"/runs/{run_id}", params={"wait": 30})
        assert response.status_code == 200, response.text
        status = response.json()
        if status["status"] not in ("pending", "running") or time.monotonic() > deadline:
            return status


def run_over_http(client, body: dict, timeout_s: float = 600.0) -> dict:
    """The result document of `body`, which must run to `done`."""
    status = poll(client, submit(client, body), timeout_s)
    assert status["status"] == "done", status.get("error") or status
    return status["result"]


def request_of(body: dict) -> RunRequest:
    """The `RunRequest` the API builds from `body`."""
    return RunRequestSchema.model_validate(body).to_dataclass()


def run_in_process(body: dict) -> RunResult:
    """`engine.run.run` on the request the API builds from `body`."""
    return run(request_of(body))
