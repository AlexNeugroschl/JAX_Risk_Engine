"""
`run_market_risk`: short-horizon VaR and Expected Shortfall of a portfolio,
by full revaluation at t=0 under shocked curves.

    scenarios (Monte Carlo or historical)
        -> shocked curves at t=0
        -> every trade repriced under every scenario   (revaluation)
        -> P&L per scenario                             [S]
        -> VaR / ES, with tail counts and dependence-aware error estimates

The statistics are `engine.risk.market.var_es` -- ORE's `RiskStatistics`
conventions, already pinned against ORE -- applied to a one-date P&L sample.
This is the engine's market-risk measure. The multi-step simulation in
`engine.run` is an exposure profile, not a VaR (audit finding R-1).

Currency: each trade is revalued in its own currency, and its base value and P&L are converted
into the run's reporting currency (`MarketRiskRequest.base_currency`, USD by default, as a
portfolio run without a simulation) at the market's as-of FX spot, in float64 after the P&L is
loaded, so the portfolio P&L adds amounts in one unit. A trade with no spot into it is refused
before any work. Only rate pillars are shocked, so the spot is the same in every scenario: FX
risk is not in the VaR/ES, and the result says so when a trade was converted.

Precision (`engine.precision`), the portfolio run's stages on this pipeline: the shifts are
rounded to the accumulate format of `simulation` and stored at `simulation.storage` (they are
drawn before the run, so the simulation stage has no product to compute); each trade is
revalued, and its P&L computed, in the arithmetic of its own pricing stage
(`Precision.precision_for(trade)`: an override for its id, else for its product, else
`pricing`; decision A-15): the shifts loaded at its accumulate format, its products in its
compute format, in difference form for swaps, bonds and Europeans
(`engine.risk.market.revaluation`); its P&L stored at that stage's storage format; VaR/ES load
every P&L at float64 (decision A-10). A scaled storage format keeps block scales along the
scenarios; the shifts are named `"shocks"` and each P&L `"values/<trade id>"` for a
stochastic rounding (`Precision.store`). A trade's base value is its value on the unshocked
curves in its accumulate format; a zero shift is exactly zero P&L at every precision, in every
storage format.

With `precision.paired_fraction > 0` a prefix within each scramble is revalued at float64 (the
paired sample, decision A-13), and the result's `PrecisionReport` measures each VaR and ES on
it: the figure at the run's precision and at float64 on the same scenarios. They are quantiles,
so the reported VaR/ES are the run's own, not corrected. The result's `warnings` name every
figure whose precision has no passing row of the evidence table under the run's pricing settings
(`run_figures`, `pricing_settings`, `engine.precision.evidence`, I-55, I-114; a float64 run has none).
"""
from dataclasses import dataclass, field
import dataclasses
from typing import Dict, List, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.market_data.market import Market
from engine.market_simulation.sharding import shard_scenarios
from engine.precision import (
    Arithmetic, Precision, PrecisionReport, RunFigures, evidence_warnings, format_name, load,
    paired_quantile, require_precision,
)
from engine.precision.estimate import paired_count
from engine.pricing.config import PricingConfig
from engine.pricing.options import bound_option_pnl, position_sign
from engine.pricing.cube import (
    DEFAULT_REPORTING_CURRENCY, reporting_fx_spots, require_unique_ids, validate_trades,
)
from engine.risk.market.factors import RateRiskFactors
from engine.risk.market.revaluation import factor_indices, revalue_trade, scenario_batch_size
from engine.risk.market.scenarios import ShockScenarios
from engine.risk.market.scenarios import SOURCE_HISTORICAL
from engine.risk.market.var_es import (
    compute_risk_metrics, expected_shortfall, historical_es_error, quantile_label, replicated_es_error, tail_sample_size,
    validate_quantiles, value_at_risk, var_rank,
)

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
        engine, and a Bermudan's/American's LGM engine, its calibration redone on every
        scenario's curves as ORE's historical P&L redoes it (I-112), the volatilities held; a
        scenario whose calibration misses its basket is refused, naming the trade.
    quantiles: VaR/ES confidence levels in (0, 1), each once, e.g. 0.99 for VaR and 0.975 for
        Basel's ES (ORE's statistics take [0.9, 1); the rank rule is the same).
    precision: the `Precision` of the run (see the module docstring); float64 by default.
        `by_product` and `by_trade` set single trades' revaluation.
    batch_size: scenarios vmapped at once inside the revaluation loop; lower
        it if a large Bermudan runs out of memory.
    base_currency: the reporting currency (ORE's `baseCurrency`) every base value, P&L and
        figure is in; each trade in another currency converted at the market's as-of FX spot,
        which must be given (`Market.fx_spots`, either way round).
    """
    trades: List
    market: Market
    scenarios: ShockScenarios
    pricing: PricingConfig = field(default_factory=PricingConfig)
    quantiles: Sequence[float] = (0.99, 0.975)
    precision: Precision = field(default_factory=Precision)
    batch_size: int = 256
    base_currency: str = DEFAULT_REPORTING_CURRENCY


@dataclass
class MarketRiskResult:
    """VaR/ES of one run, with everything needed to reproduce and audit it. Every amount is in
    `base_currency`.

    risk: `VaR_99`, `ES_97.5`, ... as positive losses, plus
        `ES_<q>_tailCount` and `ES_<q>_standardError`: delete-one-scramble jackknife
        for RQMC, conditional moving-block bootstrap for history. NaN with one scramble
        or an unestimable tail. `uncertainty` records the method and its settings.
    pnl: `[S, N]` P&L of each trade under each scenario, in request order, loaded at float64
        from its storage format and converted at its FX spot.
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
    base_currency: str
    warnings: List[str] = field(default_factory=list)
    #: The precision as run, read from the run's arrays, and with a paired float64 sample each
    #: VaR/ES measured on it ("portfolio/VaR_99", "portfolio/ES_97.5"); see `PrecisionReport`.
    precision: Optional[PrecisionReport] = None
    uncertainty: Dict = field(default_factory=dict)


def run_market_risk(request: MarketRiskRequest) -> MarketRiskResult:
    """Revalue `request.trades` under every scenario and take VaR/ES of the
    portfolio P&L. See the module docstring for the pipeline."""
    validate_request(request)
    scenarios = request.scenarios
    precision = request.precision
    spots = reporting_fx_spots(request.trades, request.market, request.base_currency, "base_currency")
    shifts = precision.store(shard_scenarios(precision.simulation.arithmetic.cast(scenarios.shifts), axis=0),
                             precision.simulation.storage, "shocks")
    base, columns, stored = [], [], []
    for cfg, spot in zip(request.trades, spots):
        stage = precision.precision_for(cfg)
        arithmetic = precision.arithmetic_for(stage, f"values/{cfg.trade_id}")
        value, trade_pnl = _trade_pnl(cfg, request, load(shifts, arithmetic.dtype), arithmetic)
        base.append(value * spot)
        stored.append(precision.store(trade_pnl, stage.storage, f"values/{cfg.trade_id}"))
        column = load(stored[-1], jnp.float64)
        if isinstance(cfg, OPTION_TYPES):
            column = bound_option_pnl(column, value, position_sign(cfg))
        columns.append(column)
    pnl, portfolio_pnl = _in_reporting_currency(tuple(columns), jnp.asarray(spots, dtype=jnp.float64))

    metrics = compute_risk_metrics(pnl[:, None, :], 0.0, percentiles=request.quantiles, include_diagnostics=False)
    risk = {key: _scalar(value) for key, value in metrics.items()}
    for q in request.quantiles:
        risk[f"ES_{quantile_label(q)}_tailCount"] = float(tail_sample_size(portfolio_pnl[:, None], q)[0])
        if scenarios.source == SOURCE_HISTORICAL:
            error = historical_es_error(portfolio_pnl, jax.random.key(scenarios.bootstrap_seed), q,
                                        _block_length(scenarios), scenarios.bootstrap_replicates)
        else:
            error = replicated_es_error(portfolio_pnl, q, scenarios.replicates)
        risk[f"ES_{quantile_label(q)}_standardError"] = float(error)

    paired = paired_count(scenarios.num_scenarios, precision.paired_fraction, scenarios.replicates)
    figures = _paired_figures(request, spots, portfolio_pnl, paired) if paired else {}
    realized = {"shocks": format_name(shifts),
                **{f"values/{cfg.trade_id}": format_name(s) for cfg, s in zip(request.trades, stored)}}

    report = dataclasses.replace(PrecisionReport.of(precision, request.trades, realized, stored,
                                 scenarios.num_scenarios, paired, figures,
                                 [precision.precision_for(cfg).compute for cfg in request.trades]),
                                 replicates=scenarios.replicates, uncertainty_method=(
                                     "moving_block_bootstrap" if scenarios.source == SOURCE_HISTORICAL else
                                     "replicate_jackknife" if scenarios.replicates > 1 else "unavailable"))
    uncertainty = ({"method": "moving_block_bootstrap", "replicates": scenarios.bootstrap_replicates,
                    "block_length": _block_length(scenarios), "seed": scenarios.bootstrap_seed}
                   if scenarios.source == SOURCE_HISTORICAL else
                   {"method": report.uncertainty_method, "replicates": scenarios.replicates})
    uncertainty["interval"] = dataclasses.asdict(report.interval)

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
        base_currency=request.base_currency,
        warnings=_warnings(request),
        precision=report,
        uncertainty=uncertainty,
    )


def _trade_pnl(cfg, request: MarketRiskRequest, moves, arithmetic: Optional[Arithmetic] = None):
    """One trade's base value and P&L `[S]` under `moves`, in `arithmetic` (the moves' own
    format by default; `revalue_trade`)."""
    return revalue_trade(cfg, request.market, request.scenarios.factors, moves, request.pricing, request.batch_size,
                         arithmetic)


@jax.jit
def _in_reporting_currency(columns, spots):
    """Each trade's float64 P&L column `[S]` times its FX spot into the reporting currency:
    the `[S, N]` P&L and its sum over trades `[S]`. A spot of 1.0 leaves a column's bits as
    they are. One program per scenario and trade count."""
    pnl = jnp.stack(columns, axis=-1) * spots
    return pnl, jnp.sum(pnl, axis=-1)


def _paired_figures(request: MarketRiskRequest, spots: Sequence[float], portfolio_pnl, paired: int) -> Dict:
    """The paired sample (decision A-13): the first `paired` scenarios revalued at float64,
    converted at the run's FX `spots`, and each VaR and ES at the run's precision and at float64
    on them.

    Each trade is revalued in the batches a float64 run uses (`scenario_batch_size`): whole
    batches from the first scenario, then cut to `paired`. A batch of another shape may round
    differently (XLA vectorizes by shape), so this is what makes the paired values exactly the
    float64 run's, and it reuses that run's compiled programs."""
    moves = jnp.asarray(request.scenarios.shifts, dtype=jnp.float64)
    total = moves.shape[0]

    replicates = request.scenarios.replicates
    per_rep, paired_rep = total // replicates, paired // replicates

    def float64_pnl(cfg):
        batch = scenario_batch_size(cfg, request.pricing, request.batch_size, moves.dtype.itemsize)
        size = min(per_rep, -(-paired_rep // batch) * batch)
        return jnp.concatenate([_trade_pnl(cfg, request, moves[i:i + size])[1][:paired_rep]
                                for i in range(0, total, per_rep)])

    _, paired_pnl = _in_reporting_currency(tuple(float64_pnl(cfg) for cfg in request.trades),
                                           jnp.asarray(spots, dtype=jnp.float64))
    figures = {}
    for q in request.quantiles:
        for name, statistic in (("VaR", value_at_risk), ("ES", expected_shortfall)):
            figures[f"portfolio/{name}_{quantile_label(q)}"] = paired_quantile(
                lambda sample, statistic=statistic, q=q: statistic(sample[:, None], q)[0], portfolio_pnl, paired_pnl,
                replicates)
    return figures


def _scalar(value) -> float:
    return float(np.asarray(value).reshape(()))


def validate_request(request: MarketRiskRequest) -> None:
    """Refuse a request `run_market_risk` cannot run, naming the field, before any work."""
    validate_portfolio(request.trades, request.market, request.pricing, request.precision, request.quantiles,
                       request.batch_size, request.base_currency)
    validate_factors(request.trades, request.market, request.scenarios.factors)
    scenarios = request.scenarios
    request.precision.validate_sampling(scenarios.num_scenarios, scenarios.replicates, "scenarios")


def validate_factors(trades, market: Market, factors: RateRiskFactors) -> None:
    """Every factor is the market's curve of its name, and every curve a trade reads is a
    factor."""
    market_curves = _market_curves(market)
    for name, curve in zip(factors.names, factors.curves):
        if name not in market_curves:
            raise ValueError(f"risk factor {name!r} is not a curve of the market (have {sorted(market_curves)}); "
                             f"build the factors with RateRiskFactors.from_market")
        if not _same_curve(curve, market_curves[name]):
            raise ValueError(f"risk factor {name!r} differs from the market's curve of that name; its trades would "
                             f"be shocked from a base they are not priced on")
    for cfg in trades:
        try:
            factor_indices(cfg, factors)
        except KeyError as exc:
            raise ValueError(f"trade {cfg.trade_id!r}: {exc.args[0]}") from None


def validate_portfolio(trades, market: Market, pricing: PricingConfig, precision: Precision,
                       quantiles: Sequence[float], batch_size: int, base_currency: str) -> None:
    """Everything of a market-risk request but its scenarios: the HTTP route checks this before
    queueing the job, then the scenarios' inputs without drawing them (no JAX work:
    `engine.risk.market.scenarios.validate_monte_carlo`, `historical_scenarios`) and
    `validate_factors`."""
    if not trades:
        raise ValueError("a market-risk run needs at least one trade")
    require_precision("MarketRiskRequest.precision", precision)
    for name in ("simulation", "market"):
        if getattr(precision, name).compute_rounding != "nearest":
            raise ValueError(f"precision.{name}.compute_rounding is not read by market risk: that stage has no products")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1; got {batch_size}")
    validate_quantiles("quantiles", quantiles)
    if not isinstance(pricing, PricingConfig):
        raise TypeError(f"pricing must be a PricingConfig; got {type(pricing).__name__}")
    require_unique_ids(trades)
    validate_trades(trades, market, pricing, precision)
    reporting_fx_spots(trades, market, base_currency, "base_currency")


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
            f"trades {options} are options priced with their volatilities held (the swaption volatility surface, "
            f"to which a calibrated LGM is recalibrated in every scenario, a fixed LGM or the Jamshidian model): "
            f"only curve pillar rates are shocked, so volatility risk is not in this VaR/ES."
        )
    converted = [cfg.trade_id for cfg in request.trades if cfg.currency != request.base_currency]
    if converted:
        out.append(
            f"trades {converted} are converted into the reporting currency {request.base_currency} at today's FX "
            f"spot, held fixed: only curve pillar rates are shocked, so FX risk is not in this VaR/ES."
        )
    for q in request.quantiles:
        tail = var_rank(request.scenarios.num_scenarios, q)
        if tail < 10:
            out.append(
                f"only {request.scenarios.num_scenarios} scenarios: the {q:.1%} tail holds about "
                f"{tail} observations, too few for a stable estimate."
            )
    if request.scenarios.source == SOURCE_HISTORICAL:
        out.append("ES standard errors use a conditional moving-block bootstrap of historical horizon windows; "
                   "they depend on block_length and stationarity, and do not measure forecast/model error (I-111).")
        if 2 * _block_length(request.scenarios) > request.scenarios.num_scenarios:
            out.append("ES uncertainty is unavailable: history is too short for two dependence blocks (I-111).")
    elif request.scenarios.replicates == 1:
        out.append("ES standard errors are unavailable with one Sobol scramble; set scenarios.replicates to at "
                   "least 2 independent scrambles (8 or more recommended; I-111).")
    else:
        out.append("ES standard errors use independent-scramble deletion (variance diagnostics only; I-111).")
    out.append("Sampling confidence intervals are unavailable: no validated tail bound for these estimators "
               "(I-111). Standard errors do not certify nominal coverage, including when zero; see "
               "docs/reference/sampling-uncertainty.md.")
    return out + evidence_warnings(request.precision, request.trades, request.scenarios.num_scenarios,
                                   run_figures(request.quantiles),
                                   settings=pricing_settings(request.pricing, request.trades),
                                   replicates=request.scenarios.replicates)


def _block_length(scenarios: ShockScenarios) -> int:
    return scenarios.block_length or max(scenarios.horizon_days, int(np.ceil(scenarios.num_scenarios ** (1 / 3))))


def pricing_settings(pricing: PricingConfig, trades: Sequence) -> Dict[str, str]:
    """The pricing settings of each product of `trades` that change its kernel's arithmetic, as
    the evidence table keys a market-risk run's rows (`PricingConfig.kernel_settings`, I-114): a
    calibrated Bermudan or American is recalibrated under each scenario, its own induction per
    scenario (I-112); a fixed one shares one model."""
    return pricing.kernel_settings({t.product for t in trades}, recalibrated=True)


def run_figures(quantiles: Sequence[float]) -> RunFigures:
    """The figures a market-risk run reports, as the evidence table names them (I-55): each
    trade's mean P&L, and the portfolio's with its VaR and ES at `quantiles`, all the run's own
    (a paired sample measures VaR and ES, and corrects nothing). The shifts are the simulation
    stage and the revaluation the pricing stage; there is no scenario market."""
    labels = [quantile_label(q) for q in quantiles]
    return RunFigures(stages=("simulation", "pricing"), aggregate="portfolio", trade=("PnL",),
                      total=("PnL",) + tuple(f"{name}_{label}" for label in labels for name in ("VaR", "ES")),
                      unit="scenarios", legend="PnL here is the mean P&L over the scenarios.")
