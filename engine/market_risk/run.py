"""
`run_market_risk`: short-horizon VaR and Expected Shortfall of a portfolio,
by full revaluation at t=0 under shocked curves.

    scenarios (Monte Carlo or historical)
        -> shocked curves at t=0
        -> every trade repriced under every scenario   (revaluation)
        -> P&L per scenario                             [S]
        -> VaR / ES, with tail counts and MC standard errors

The statistics are `engine.risk.var_es` -- ORE's `RiskStatistics`
conventions, already pinned against ORE -- applied to a one-date P&L sample.
This is the engine's market-risk measure. The multi-step simulation in
`engine.portfolio` is an exposure profile, not a VaR (audit finding R-1).

Precision (`engine.precision`), the portfolio run's stages on this pipeline: the shifts are
rounded to `simulation.compute` and stored at `simulation.storage`; each trade is revalued, and
its P&L computed, at the compute format of its own pricing stage
(`Precision.precision_for(trade)`: an override for its id, else for its product, else
`pricing`; decision A-15) and its P&L stored at that stage's storage format; VaR/ES load every
P&L at float64 (decision A-10). A trade's base value is its revaluation of the unshocked
curves at its compute format, the anchor its P&L is measured from, so a zero shift is exactly
zero P&L at every precision.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import jax.numpy as jnp
import numpy as np

from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.market import Market
from engine.market_risk.factors import RateRiskFactors
from engine.market_risk.revaluation import factor_indices, revalue_trade
from engine.market_risk.scenarios import ShockScenarios
from engine.precision import Precision, load, require_precision, store
from engine.risk.var_es import compute_risk_metrics
from engine.valuation.config import PricingConfig
from engine.valuation.portfolio import require_unique_ids, validate_trades

OPTION_TYPES = (SwaptionConfig, BermudanSwaptionConfig, AmericanSwaptionConfig)

_CURVE_TOLERANCE = 1e-12


@dataclass
class MarketRiskRequest:
    """A portfolio, today's market, and the shock scenarios to revalue it under.

    trades: any mix of `SwapConfig`, `SwaptionConfig`, `BermudanSwaptionConfig`,
        `AmericanSwaptionConfig` and `BondConfig`, each valued on `market.asof`, with unique
        `trade_id`s.
    market: today's market: the curves the factors shock, and the volatilities held fixed.
    scenarios: `ShockScenarios` on factors named after the market's curves
        (`RateRiskFactors.from_market`), each equal to the market's curve of its name. Every
        curve a trade reads must be a factor.
    pricing: the engine per product (`PricingConfig`), as in a portfolio run: a European's
        engine, and a Bermudan's/American's LGM engine, calibrated on today's market and held
        fixed under every scenario.
    quantiles: VaR/ES confidence levels, e.g. 0.99 for VaR and 0.975 for
        Basel's ES.
    precision: the `Precision` of the run (see the module docstring); float64 by default.
        `by_product` and `by_trade` set single trades' revaluation.
    batch_size: scenarios vmapped at once inside the revaluation loop; lower
        it if a large Bermudan runs out of memory.
    """
    trades: List
    market: Market
    scenarios: ShockScenarios
    pricing: PricingConfig = field(default_factory=PricingConfig)
    quantiles: Sequence[float] = (0.99, 0.975)
    precision: Precision = field(default_factory=Precision)
    batch_size: int = 256


@dataclass
class MarketRiskResult:
    """VaR/ES of one run, with everything needed to reproduce and audit it.

    risk: `VaR_99`, `ES_97.5`, ... as positive losses, plus
        `ES_<q>_tailCount` (observations the ES averaged) and
        `ES_<q>_standardError` (its Monte Carlo standard error); NaN where a
        tail is empty.
    pnl: `[S, N]` P&L of each trade under each scenario, in request order, loaded at float64
        from its storage format.
    portfolio_pnl: `[S]` the sum across trades, which the statistics use (float64).
    """
    base_npv: float
    base_npv_per_trade: List[float]
    pnl: jnp.ndarray
    portfolio_pnl: jnp.ndarray
    risk: Dict[str, float]
    measure: str
    source: str
    horizon_days: int
    num_scenarios: int
    risk_factors: List[str]
    warnings: List[str] = field(default_factory=list)


def run_market_risk(request: MarketRiskRequest) -> MarketRiskResult:
    """Revalue `request.trades` under every scenario and take VaR/ES of the
    portfolio P&L. See the module docstring for the pipeline."""
    _validate(request)
    scenarios = request.scenarios
    precision = request.precision
    shifts = store(jnp.asarray(scenarios.shifts, dtype=precision.simulation.compute_dtype),
                   precision.simulation.storage)
    base, columns = [], []
    for cfg in request.trades:
        stage = precision.precision_for(cfg)
        moves = load(shifts, stage.compute_dtype)
        value, shocked = revalue_trade(cfg, request.market, scenarios.factors, moves, request.pricing,
                                       request.batch_size)
        base.append(value)
        columns.append(load(store(shocked - jnp.asarray(value, dtype=moves.dtype), stage.storage), jnp.float64))
    pnl = jnp.stack(columns, axis=-1)
    portfolio_pnl = jnp.sum(pnl, axis=-1)

    metrics = compute_risk_metrics(pnl[:, None, :], 0.0, percentiles=request.quantiles)
    risk = {key: _scalar(value) for key, value in metrics.items()}

    return MarketRiskResult(
        base_npv=float(np.sum(base)),
        base_npv_per_trade=[float(v) for v in base],
        pnl=pnl,
        portfolio_pnl=portfolio_pnl,
        risk=risk,
        measure=scenarios.measure,
        source=scenarios.source,
        horizon_days=scenarios.horizon_days,
        num_scenarios=scenarios.num_scenarios,
        risk_factors=scenarios.factors.labels(),
        warnings=_warnings(request),
    )


def _scalar(value) -> float:
    return float(np.asarray(value).reshape(()))


def _validate(request: MarketRiskRequest) -> None:
    if not request.trades:
        raise ValueError("a market-risk run needs at least one trade")
    require_precision("MarketRiskRequest.precision", request.precision)
    if request.batch_size < 1:
        raise ValueError(f"batch_size must be at least 1; got {request.batch_size}")
    for q in request.quantiles:
        if not 0.0 < q < 1.0:
            raise ValueError(f"quantile must lie in (0, 1); got {q}")
    if not isinstance(request.pricing, PricingConfig):
        raise TypeError(f"pricing must be a PricingConfig; got {type(request.pricing).__name__}")
    require_unique_ids(request.trades)
    validate_trades(request.trades, request.market, request.pricing, request.precision)

    factors = request.scenarios.factors
    market_curves = _market_curves(request.market)
    for name, curve in zip(factors.names, factors.curves):
        if name not in market_curves:
            raise ValueError(f"risk factor {name!r} is not a curve of the market (have {sorted(market_curves)}); "
                             f"build the factors with RateRiskFactors.from_market")
        if not _same_curve(curve, market_curves[name]):
            raise ValueError(f"risk factor {name!r} differs from the market's curve of that name; its trades would "
                             f"be shocked from a base they are not priced on")
    for cfg in request.trades:
        try:
            factor_indices(cfg, factors)
        except KeyError as exc:
            raise ValueError(f"trade {cfg.trade_id!r}: {exc.args[0]}") from None


def _market_curves(market: Market):
    every = RateRiskFactors.from_market(market)
    return dict(zip(every.names, every.curves))


def _same_curve(a, b) -> bool:
    if list(a.times) != list(b.times) or len(a.rates) != len(b.rates):
        return False
    return all(abs(x - y) <= _CURVE_TOLERANCE for x, y in zip(a.rates, b.rates))


def _warnings(request: MarketRiskRequest) -> List[str]:
    out = []
    options = [cfg.trade_id for cfg in request.trades if isinstance(cfg, OPTION_TYPES)]
    if options:
        out.append(
            f"trades {options} are options priced with fixed volatility (the swaption volatility surface, "
            f"a calibrated LGM or the Jamshidian model): only curve pillar rates are shocked, so volatility "
            f"risk is not in this VaR/ES."
        )
    for q in request.quantiles:
        tail = int(np.floor(request.scenarios.num_scenarios * (1.0 - q)))
        if tail < 10:
            out.append(
                f"only {request.scenarios.num_scenarios} scenarios: the {q:.1%} tail holds about "
                f"{tail} observations, too few for a stable estimate."
            )
    return out
