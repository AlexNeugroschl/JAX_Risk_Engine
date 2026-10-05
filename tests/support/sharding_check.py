"""
Runs in its own process with several XLA host devices (`XLA_FLAGS`), for
tests/test_sharding.py: the same portfolio and market-risk runs with the scenario axis on one
device (`JAX_RISK_SCENARIO_DEVICES=1`) and split across every device, printed as JSON so the
test compares them. The device count is fixed when JAX starts, so this cannot run in the
test's own process.

    python -m tests.support.sharding_check
"""
import json
import os
import sys

import numpy as np


def _portfolio_runs():
    import ORE

    from engine.api.market_schemas import MarketPortfolioRequestSchema
    from engine.portfolio import price_portfolio
    from tests.support import portfolio as shared

    dates = [(shared.ASOF + ORE.Period(m, ORE.Months)).ISO() for m in (6, 12)]
    trades = [shared.trades_json()[n] for n in ("swap-payer", "european-payer", "bermudan-payer-physical", "bond")]
    fp8 = {"storage": "float8_e4m3fn", "compute": "float64", "accumulate": "float64"}
    bodies = {
        "lgm": {"simulation": {"samples": 64, "seed": 5}, "precision": {"paired_fraction": 0.25}},
        "fp8-stochastic": {"simulation": {"samples": 64, "seed": 5},
                           "precision": {"market": fp8, "pricing": fp8, "rounding": "stochastic"}},
    }
    for name, extra in bodies.items():
        body = {"market": shared.market_json(), "trades": trades,
                "pricing": {"bermudan": {"n_per_std": 12, "std_devs": 4.0}},
                "simulation": {"dates": dates, "base_currency": "USD", "ir": {"USD": {"model": "LGM", "reversion": 0.03}},
                               **extra["simulation"]},
                "precision": extra["precision"]}
        request = MarketPortfolioRequestSchema.model_validate(body).to_dataclass()
        yield name, lambda request=request: _portfolio_figures(price_portfolio(request))


def _portfolio_figures(result) -> dict:
    exposure = result.exposure
    return {"cube": np.asarray(result.npv_cube).tolist(), "epe": np.asarray(exposure.epe).tolist(),
            "ene": np.asarray(exposure.ene).tolist(), "devices": list(result.precision.devices),
            "cube_devices": len(result.npv_cube.devices())}


def _market_risk_run():
    from engine.market_risk import MarketRiskRequest, monte_carlo_scenarios, run_market_risk
    from tests import market_risk_support as m

    scenarios = monte_carlo_scenarios(m.factors(), m.covariance(horizon_days=10), 10, 64, seed=3)
    trades = [m.swap(), m.european(), m.bermudan(), m.bond()]
    request = MarketRiskRequest(trades, m.market(), scenarios, m.PRICING, batch_size=8)
    result = run_market_risk(request)
    return {"pnl": np.asarray(result.pnl).tolist(), "risk": result.risk, "devices": list(result.precision.devices),
            "pnl_devices": len(result.pnl.devices())}


def main() -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    runs = dict(_portfolio_runs())
    runs["market-risk"] = _market_risk_run
    out = {"device_count": len(jax.local_devices())}
    for name, run in runs.items():
        os.environ["JAX_RISK_SCENARIO_DEVICES"] = "1"
        one = run()
        del os.environ["JAX_RISK_SCENARIO_DEVICES"]
        out[name] = {"one": one, "split": run()}
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
