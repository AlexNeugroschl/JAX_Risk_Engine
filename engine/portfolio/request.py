"""
The engine's portfolio entry point: `PortfolioRequest` (trades, today's market, run
configuration, analytics) in, `PortfolioResult` (base NPVs, scenario cube, exposure, Greeks)
out, via `price_portfolio`.

Pure dataclasses and JAX; the HTTP layer (`engine/api/`) wraps these types.

Every model, engine, Greeks method and precision choice is in `request.config`, a `RunConfig`
(`engine.portfolio.config`) whose defaults are ORE's. One pipeline prices every
configuration (`engine.portfolio.market_path`): the cross-asset model with each currency's
model (`config.simulation.ir`: the LGM by default, or Hull-White), then each trade on every path
with its configured engine, exercise and fixings as ORE handles them. Before roadmap 1.3 the
Hull-White model was a second pipeline selected by passing a `SimulationConfig` as the market;
that shape is retired.

Concurrency: `price_portfolio` may be called from several threads at once. Every precision is
an explicit dtype of the run's own arrays (`engine.precision`) and `jax_enable_x64` is set once,
when `engine` is imported, never per run; the pipeline keeps no module-level state and never
reads ORE's global evaluation date (trades carry their own, I-64). Roadmap 1.4 removed the lock
that serialized runs while the flag was switched per precision.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import jax

from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig
from engine.instruments.treasury import BondConfig
from engine.market import Market
from engine.portfolio.config import RunConfig
from engine.risk.exposure import ExposureProfile

# Re-exported so `engine.portfolio._validate_common_fields` resolves (implemented in
# validation.py to avoid an import cycle).
from engine.portfolio.validation import _validate_common_fields, _validate_tenor  # noqa: F401

TradeConfig = Union[
    SwapConfig, SwaptionConfig, BermudanSwaptionConfig, AmericanSwaptionConfig, BondConfig,
]


@dataclass
class PortfolioRequest:
    """
    Input to `price_portfolio`.

    market: today's `Market` (curves per currency and index, swaption volatilities, FX and
        equity spots).
    trades: any mix of trade types, each valued on `market.asof` and named by its `trade_id`
        (unique in the portfolio, as ORE requires); results come back in this order.
    config: the run configuration (`RunConfig`): simulation and model per currency, engine
        per product, Greeks method and settings, precision per stage, reporting currency.
        The default is ORE's.
    pfe_quantiles: PFE quantiles for the exposure profiles.
    compute_greeks: also compute the Greeks per trade, by `config.greeks.method`.
    scenario_risk: whether to simulate and build `npv_cube` and the exposure profiles (needs
        `config.simulation`). Without it the result has an empty `npv_cube` and no exposure
        (absent, not zero), and `scenario_risk_available` says so.

    The cube is a multi-step risk-neutral simulation, used for exposure profiles
    (`engine.risk.exposure`). Short-horizon VaR/ES is `engine.market_risk.run_market_risk`.
    """
    market: Market
    trades: List[TradeConfig]
    config: RunConfig = field(default_factory=RunConfig)
    pfe_quantiles: Sequence[float] = (0.95, 0.99)
    compute_greeks: bool = False
    scenario_risk: bool = True

    def __post_init__(self):
        if not isinstance(self.market, Market):
            raise TypeError(f"market must be today's Market; got {type(self.market).__name__} (the Hull-White "
                            f"model is a HullWhiteConfig in config.simulation.ir, not a SimulationConfig market)")
        if not isinstance(self.config, RunConfig):
            raise TypeError(f"config must be a RunConfig; got {type(self.config).__name__}")
        ids = [cfg.trade_id for cfg in self.trades]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"trade ids must be unique in a portfolio; repeated: {duplicates}")


@dataclass
class PortfolioResult:
    """Output of `price_portfolio`."""
    base_npv: float
    npv_cube: jax.Array                                   # [Scenarios, TimeSteps, Trades]
    #: The whole portfolio as one netting set, no collateral. `None` without scenario risk.
    exposure: Optional[ExposureProfile] = None
    #: Standalone exposure per trade, in request order. Empty without scenario risk.
    trade_exposures: List[ExposureProfile] = field(default_factory=list)
    greeks: Optional[Dict[int, Dict[str, jax.Array]]] = None  # trade index (in request.trades order) -> greeks dict
    warnings: List[str] = field(default_factory=list)
    # t=0 NPV per trade, in request order; `base_npv` is their sum.
    base_npv_per_trade: List[float] = field(default_factory=list)
    # Whether `npv_cube`/`exposure` were computed (`scenario_risk`). When False, `exposure`
    # is None and `npv_cube` has no time steps. Carried on the result for consumers that
    # never see the request.
    scenario_risk_available: bool = True
    # The measure of the exposure figures: `ENGINE_RISK_MEASURE` (risk-neutral-pricing), or
    # None without scenario risk (I-11).
    measure: Optional[str] = None
    # Every trade's `trade_id`, in request order: the key of each per-trade figure (I-10).
    trade_ids: List[str] = field(default_factory=list)


def price_portfolio(request: PortfolioRequest) -> PortfolioResult:
    """Price a `PortfolioRequest` (`engine.portfolio.market_path.price_on_market`): validate
    the configuration and trades before any JAX work, calibrate and simulate the cross-asset
    model, value every trade today and on every path, build the exposure profiles, and
    optionally the Greeks. The result carries the trades' ids."""
    from engine.portfolio.market_path import price_on_market
    result = price_on_market(request)
    result.trade_ids = [cfg.trade_id for cfg in request.trades]
    return result
