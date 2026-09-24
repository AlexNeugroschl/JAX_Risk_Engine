"""
Short-horizon market risk: VaR and Expected Shortfall by full revaluation of
the portfolio at t=0 under shocked curves.

    factors = RateRiskFactors.from_curves([ois, libor], names=["OIS", "LIBOR"])
    scenarios = monte_carlo_scenarios(factors, covariance, horizon_days=10,
                                      num_scenarios=2**14, seed=1)
    #   or    = historical_scenarios(factors, history, horizon_days=10)
    result = run_market_risk(MarketRiskRequest(trades, scenarios))
    result.risk["VaR_99"], result.risk["ES_97.5"]

This is the engine's market-risk measure. `engine.portfolio.price_portfolio`
simulates a multi-step risk-neutral cube, which gives exposure profiles
(`engine.risk.exposure`), not VaR. See docs/risk/market-risk.md.
"""
from engine.market_risk.factors import RateRiskFactors
from engine.market_risk.run import MarketRiskRequest, MarketRiskResult, run_market_risk
from engine.market_risk.scenarios import (
    SOURCE_HISTORICAL,
    SOURCE_MONTE_CARLO,
    ShockScenarios,
    covariance_from_history,
    historical_scenarios,
    horizon_moves,
    monte_carlo_scenarios,
)

__all__ = [
    "MarketRiskRequest",
    "MarketRiskResult",
    "RateRiskFactors",
    "SOURCE_HISTORICAL",
    "SOURCE_MONTE_CARLO",
    "ShockScenarios",
    "covariance_from_history",
    "historical_scenarios",
    "horizon_moves",
    "monte_carlo_scenarios",
    "run_market_risk",
]
