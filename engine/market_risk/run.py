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
"""
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.market_risk.revaluation import OPTION_TYPES, curve_indices, revalue
from engine.market_risk.scenarios import ShockScenarios
from engine.portfolio.validation import validate_single_evaluation_date
from engine.risk.var_es import compute_risk_metrics

_CURVE_TOLERANCE = 1e-12


@dataclass
class MarketRiskRequest:
    """A portfolio and the shock scenarios to revalue it under.

    trades: any mix of `SwapConfig`, `SwaptionConfig`,
        `BermudanSwaptionConfig`, `AmericanSwaptionConfig` and `BondConfig`,
        all on one evaluation date. Curve references index into
        `scenarios.factors.curves`; a trade that carries its own
        `initial_zero_curve` must carry exactly that curve. Bermudan and
        American trades need a calibrated (non-`None`) `hw_sigma`.
    scenarios: `ShockScenarios` built for the same curves.
    quantiles: VaR/ES confidence levels, e.g. 0.99 for VaR and 0.975 for
        Basel's ES.
    precision: 64 or 32 -- the dtype of the revaluation and the statistics.
    batch_size: scenarios vmapped at once inside the revaluation loop; lower
        it if a large Bermudan runs out of memory.
    """
    trades: List
    scenarios: ShockScenarios
    quantiles: Sequence[float] = (0.99, 0.975)
    precision: int = 64
    batch_size: int = 256


@dataclass
class MarketRiskResult:
    """VaR/ES of one run, with everything needed to reproduce and audit it.

    risk: `VaR_99`, `ES_97.5`, ... as positive losses, plus
        `ES_<q>_tailCount` (observations the ES averaged) and
        `ES_<q>_standardError` (its Monte Carlo standard error); NaN where a
        tail is empty.
    pnl: `[S, N]` P&L of each trade under each scenario, in request order.
    portfolio_pnl: `[S]` the sum across trades, which the statistics use.
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
    dtype = jnp.float64 if request.precision == 64 else jnp.float32

    # Revaluation at float32 still needs x64 enabled to build the float64
    # arrays a 64-bit request uses; it never changes a float32 array.
    jax.config.update("jax_enable_x64", True)
    base, shocked = revalue(request.trades, scenarios.factors, scenarios.shifts, dtype, request.batch_size)
    pnl = shocked - jnp.asarray(base, dtype=dtype)[None, :]
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
    if request.precision not in (32, 64):
        raise ValueError(f"precision must be 32 or 64; got {request.precision!r}")
    if request.batch_size < 1:
        raise ValueError(f"batch_size must be at least 1; got {request.batch_size}")
    for q in request.quantiles:
        if not 0.0 < q < 1.0:
            raise ValueError(f"quantile must lie in (0, 1); got {q}")
    validate_single_evaluation_date(request.trades)

    factors = request.scenarios.factors
    for i, cfg in enumerate(request.trades):
        label = f"trade[{i}] ({type(cfg).__name__}, notional={cfg.notional})"
        indices = curve_indices(cfg)
        for index in indices:
            if not 0 <= index < len(factors.curves):
                raise ValueError(
                    f"{label}: curve index {index} is out of range for {len(factors.curves)} "
                    f"risk-factor curves"
                )
        own_curve = getattr(cfg, "initial_zero_curve", None)
        if own_curve is not None and not _same_curve(own_curve, factors.curves[indices[0]]):
            raise ValueError(
                f"{label}: initial_zero_curve does not match risk-factor curve "
                f"{factors.names[indices[0]]!r} (index {indices[0]}); the trade would be shocked "
                f"from a base it is not priced on"
            )
        if isinstance(cfg, (BermudanSwaptionConfig, AmericanSwaptionConfig)) and cfg.hw_sigma is None:
            raise ValueError(f"{label}: hw_sigma is None; calibrate it before a market-risk run")


def _same_curve(a, b) -> bool:
    if list(a.times) != list(b.times) or len(a.rates) != len(b.rates):
        return False
    return all(abs(x - y) <= _CURVE_TOLERANCE for x, y in zip(a.rates, b.rates))


def _warnings(request: MarketRiskRequest) -> List[str]:
    out = []
    options = [i for i, cfg in enumerate(request.trades) if isinstance(cfg, OPTION_TYPES)]
    if options:
        out.append(
            f"trades {options} are options priced with fixed model volatility (hw_sigma): only "
            f"curve pillar rates are shocked, so volatility risk is not in this VaR/ES."
        )
    for q in request.quantiles:
        tail = int(np.floor(request.scenarios.num_scenarios * (1.0 - q)))
        if tail < 10:
            out.append(
                f"only {request.scenarios.num_scenarios} scenarios: the {q:.1%} tail holds about "
                f"{tail} observations, too few for a stable estimate."
            )
    return out
