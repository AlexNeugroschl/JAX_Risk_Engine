"""
The run request (I-87, decision A-2): one request for every analytic, as ORE runs a portfolio,
its market data and its configuration files with a list of analytics. `run(RunRequest)` is
the engine's one way in, from Python and, through `engine.api`, over HTTP.

    RunRequest
      market        today's `Market`
      portfolio     the trades, each valued on `market.asof`
      analytics     what to compute: "npv", "exposure", "sensitivities", "market_risk", "calibration"
      config        `RunRequestConfig`, how to compute it, one section per ORE file:
        base_currency   ore.xml's `baseCurrency`: the reporting currency, the simulation's domestic one
        models          simulation.xml's `CrossAssetModel`: the model per currency, FX and equity
                        volatilities, correlations, the bootstrap tolerance (`ModelsConfig`)
        simulation      simulation.xml's `Parameters` and `Market`: dates, samples, seed, replicates,
                        the scenario market's tenors, the swaption volatilities' decay
        pricing         pricingengine.xml: the engine per product (`PricingConfig`)
        sensitivity     sensitivity.xml, and the Greeks method (`GreeksConfig`)
        exposure        the exposure analytic's PFE quantiles (`ExposureConfig`)
        market_risk     the shock scenarios, VaR/ES quantiles and batch size (`MarketRiskConfig`)
        precision       storage, compute and accumulate formats per stage (`Precision`; not ORE's,
                        the research axis)
      include_npv_cube  whether the exposure analytic assembles its float64 cube

**What runs.** `npv`, `exposure` and `sensitivities` are one `price_portfolio` (today's values
are in every one of them); `market_risk` is `run_market_risk`; `calibration` is `calibrate_cam`,
the cross-asset model's calibration per currency that `exposure` simulates with. Each is the
engine's own entry point, unchanged, so a run of one analytic is that entry point's direct call
and a run of several is each alone. Sharing work between analytics (the market build, the
calibration) is a later optimization, not a requirement (details/configurable-engine.md).

**Checked before any work** (`validate_run`, no JAX work, so the HTTP API answers a refusal
with a `400` and queues nothing):

  - each analytic named once, from `ANALYTICS`;
  - every section an analytic needs given (`exposure`: models and simulation; `market_risk`: its
    section; `calibration`: models with a calibration basket);
  - nothing accepted and then ignored (I-59's rule): a section, a portfolio or a setting that
    differs from its default where no requested analytic reads it is refused, naming it. The
    precision is read by `exposure` and `market_risk` alone (today's values, Greeks and
    calibration are float64, decision A-10), and each of them refuses a precision setting it
    does not read (I-95): market risk has no scenario market (`precision.market`) and its
    shifts no products (`precision.simulation.compute` other than its accumulate format);
  - then each analytic's own validation (`engine.run.pipeline.validate_request`,
    `engine.risk.market.run.validate_portfolio` and `validate_scenarios`), on scenarios not yet
    drawn (`MonteCarloScenarios`).

Every refusal is a `ValueError`, `KeyError` or `TypeError` naming the section, field or trade.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from engine.calibration.cam import CamCalibration, calibrate_cam
from engine.calibration.lgm import BOOTSTRAP_TOLERANCE, check_tolerance
from engine.market_data.market import Market, index_name
from engine.market_simulation.config import (
    DEFAULT_CURVE_TENORS, CamConfig, IrModelConfig, LgmConfig, Volatility,
)
from engine.market_simulation.sharding import whole_on_every_host
from engine.precision import Precision, StagePrecision, require_precision
from engine.pricing.config import PricingConfig
from engine.pricing.cube import DEFAULT_REPORTING_CURRENCY, require_unique_ids
from engine.risk.market import MarketRiskRequest, MarketRiskResult, run_market_risk
from engine.risk.market.run import validate_portfolio, validate_scenarios
from engine.risk.market.scenarios import MonteCarloScenarios, ShockScenarios, shock_scenarios
from engine.risk.market.var_es import validate_quantiles
from engine.run.config import GreeksConfig, RunConfig
from engine.run.identity import EngineIdentity, engine_identity
from engine.run.request import PortfolioRequest, PortfolioResult, TradeConfig, price_portfolio
from engine.run.trace import phase

#: The analytics a run request names, as ORE's `ore.xml` lists its analytics.
ANALYTICS = ("npv", "exposure", "sensitivities", "market_risk", "calibration")
#: The analytics of one `price_portfolio`.
PORTFOLIO_ANALYTICS = ("npv", "exposure", "sensitivities")
#: The analytics that price the portfolio.
TRADE_ANALYTICS = PORTFOLIO_ANALYTICS + ("market_risk",)


@dataclass(frozen=True)
class ModelsConfig:
    """simulation.xml's `CrossAssetModel`, the models the `exposure` analytic simulates and the
    `calibration` analytic calibrates (`engine.market_simulation.config.CamConfig`'s model half):

    ir: currency -> its model, `LgmConfig` (the default) or `HullWhiteConfig`; one with a
        calibration basket is bootstrapped to it on today's market.
    fx_volatilities: foreign currency -> the Black-Scholes volatility of its FX rate against
        `base_currency`, for every other currency of `ir`.
    equity_volatilities: equity name -> its Black-Scholes volatility.
    correlations: `{(factor, factor): rho}` (`IR:USD`, `FX:EURUSD`, `EQ:SP5`), unlisted pairs 0.
    bootstrap_tolerance: ORE's `BootstrapTolerance`, one for the model (I-73).
    """
    ir: Mapping[str, IrModelConfig]
    fx_volatilities: Mapping[str, Volatility] = field(default_factory=dict)
    equity_volatilities: Mapping[str, Volatility] = field(default_factory=dict)
    correlations: Mapping[Tuple[str, str], float] = field(default_factory=dict)
    bootstrap_tolerance: float = BOOTSTRAP_TOLERANCE

    def __post_init__(self):
        if not self.ir:
            raise ValueError("config.models.ir: give at least one currency's model")
        for currency, model in self.ir.items():
            if not isinstance(model, LgmConfig):
                raise TypeError(f"config.models.ir[{currency!r}] must be an LgmConfig or a HullWhiteConfig; "
                                f"got {type(model).__name__}")
        check_tolerance(self.bootstrap_tolerance, "config.models.bootstrap_tolerance")

    @property
    def calibrated(self) -> Tuple[str, ...]:
        """The currencies with a calibration basket."""
        return tuple(c for c, model in self.ir.items() if model.calibrated)


@dataclass(frozen=True)
class SimulationParameters:
    """simulation.xml's `Parameters` and `Market`, the exposure simulation's grid and market
    (`CamConfig`'s other half): `dates` ascending after the as-of date, `samples` paths in
    `replicates` independent scrambles from `seed`, the scenario market's `curve_tenors`, and
    the swaption volatilities' `ReactionToTimeDecay` (`swaption_vol_decay`). Checked whole when
    the cross-asset configuration is built (`RunRequestConfig.cam`)."""
    dates: Tuple
    samples: int = 1000
    seed: int = 42
    replicates: int = 1
    curve_tenors: Tuple[str, ...] = DEFAULT_CURVE_TENORS
    swaption_vol_decay: str = "ForwardVariance"


@dataclass(frozen=True)
class ExposureConfig:
    """The exposure analytic's PFE quantiles (ORE's `PfeQuantile`, one per figure here)."""
    pfe_quantiles: Tuple[float, ...] = (0.95, 0.99)

    def __post_init__(self):
        object.__setattr__(self, "pfe_quantiles", tuple(self.pfe_quantiles))
        validate_quantiles("config.exposure.pfe_quantiles", self.pfe_quantiles)


@dataclass(frozen=True)
class MarketRiskConfig:
    """The market-risk analytic (`engine.risk.market.MarketRiskRequest`): the shock `scenarios`,
    drawn (`ShockScenarios`, e.g. `historical_scenarios`) or drawn when the run starts
    (`MonteCarloScenarios`), the VaR/ES `quantiles`, and the scenarios revalued at once
    (`batch_size`)."""
    scenarios: Union[ShockScenarios, MonteCarloScenarios]
    quantiles: Tuple[float, ...] = (0.99, 0.975)
    batch_size: int = 256

    def __post_init__(self):
        object.__setattr__(self, "quantiles", tuple(self.quantiles))
        if not isinstance(self.scenarios, (ShockScenarios, MonteCarloScenarios)):
            raise TypeError(f"config.market_risk.scenarios must be ShockScenarios or MonteCarloScenarios; "
                            f"got {type(self.scenarios).__name__}")


@dataclass(frozen=True)
class RunRequestConfig:
    """How a run computes its analytics, one section per ORE file (see the module docstring).
    `RunRequestConfig()` is ORE's defaults; `models`, `simulation` and `market_risk` have none,
    being the inputs of the analytics that need them."""
    base_currency: str = DEFAULT_REPORTING_CURRENCY
    models: Optional[ModelsConfig] = None
    simulation: Optional[SimulationParameters] = None
    pricing: PricingConfig = field(default_factory=PricingConfig)
    sensitivity: GreeksConfig = field(default_factory=GreeksConfig)
    exposure: ExposureConfig = field(default_factory=ExposureConfig)
    market_risk: Optional[MarketRiskConfig] = None
    precision: Precision = field(default_factory=Precision)

    def __post_init__(self):
        for name, expected in (("models", ModelsConfig), ("simulation", SimulationParameters),
                               ("market_risk", MarketRiskConfig)):
            value = getattr(self, name)
            if value is not None and not isinstance(value, expected):
                raise TypeError(f"config.{name} must be a {expected.__name__}; got {type(value).__name__}")
        for name, expected in (("pricing", PricingConfig), ("sensitivity", GreeksConfig),
                               ("exposure", ExposureConfig)):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"config.{name} must be a {expected.__name__}; got {type(getattr(self, name)).__name__}")
        require_precision("config.precision", self.precision)

    def cam(self) -> CamConfig:
        """The cross-asset model's configuration of the `exposure` analytic: `models` simulated
        on `simulation`'s grid, `base_currency` its domestic currency."""
        models, simulation = self.models, self.simulation
        try:
            return CamConfig(dates=tuple(simulation.dates), base_currency=self.base_currency, ir=dict(models.ir),
                             fx_volatilities=dict(models.fx_volatilities),
                             equity_volatilities=dict(models.equity_volatilities),
                             correlations=dict(models.correlations), curve_tenors=tuple(simulation.curve_tenors),
                             samples=simulation.samples, seed=simulation.seed,
                             swaption_vol_decay=simulation.swaption_vol_decay,
                             bootstrap_tolerance=models.bootstrap_tolerance, replicates=simulation.replicates)
        except (ValueError, TypeError) as exc:
            raise type(exc)(f"config.models, config.simulation and config.base_currency make the cross-asset "
                            f"model: {exc}") from None


@dataclass(frozen=True)
class RunRequest:
    """One run (see the module docstring). `portfolio` is empty only when no analytic prices
    trades (calibration alone)."""
    market: Market
    analytics: Tuple[str, ...]
    portfolio: Tuple[TradeConfig, ...] = ()
    config: RunRequestConfig = field(default_factory=RunRequestConfig)
    include_npv_cube: bool = True

    def __post_init__(self):
        if not isinstance(self.market, Market):
            raise TypeError(f"market must be today's Market; got {type(self.market).__name__}")
        if not isinstance(self.config, RunRequestConfig):
            raise TypeError(f"config must be a RunRequestConfig; got {type(self.config).__name__}")
        object.__setattr__(self, "analytics", tuple(self.analytics))
        object.__setattr__(self, "portfolio", tuple(self.portfolio))
        require_unique_ids(self.portfolio)

    def wants(self, *analytics: str) -> bool:
        """Whether any of `analytics` is requested."""
        return any(a in self.analytics for a in analytics)


@dataclass
class RunResult:
    """The result of `run`: each requested analytic's, from its entry point. `request` is the
    request as run, every default filled in, so a result reproduces itself; `engine` what ran it.

    portfolio: `price_portfolio`'s result, with `npv`, `exposure` or `sensitivities` (today's
        values always; the cube and exposure with `exposure`; the Greeks with `sensitivities`).
    market_risk: `run_market_risk`'s, with `market_risk`.
    calibration: `calibrate_cam`'s per calibrated currency, with `calibration`.
    """
    request: RunRequest
    engine: EngineIdentity
    portfolio: Optional[PortfolioResult] = None
    market_risk: Optional[MarketRiskResult] = None
    calibration: Optional[Dict[str, CamCalibration]] = None

    @property
    def trade_ids(self) -> List[str]:
        """Every trade's id, in request order: the key of every per-trade figure."""
        return [t.trade_id for t in self.request.portfolio]


def run(request: RunRequest) -> RunResult:
    """Run every analytic of `request` (see the module docstring), each by the engine's entry
    point, after `validate_run`; on several hosts (a pod slice) every array of the result whole on
    each (`engine.market_simulation.sharding.whole_on_every_host`)."""
    validate_run(request)
    result = RunResult(request=request, engine=engine_identity())
    if request.wants(*PORTFOLIO_ANALYTICS):
        result.portfolio = price_portfolio(portfolio_request(request))
    if request.wants("market_risk"):
        with phase("scenarios"):
            scenarios = shock_scenarios(request.config.market_risk.scenarios)
        result.market_risk = run_market_risk(market_risk_request(request, scenarios))
    if request.wants("calibration"):
        models = request.config.models
        with phase("calibration"):
            result.calibration = calibrate_cam(request.market, models.ir, models.bootstrap_tolerance)
    return whole_on_every_host(result)


def portfolio_request(request: RunRequest) -> PortfolioRequest:
    """The `price_portfolio` request of `request`'s `npv`, `exposure` and `sensitivities`."""
    config, exposure = request.config, request.wants("exposure")
    return PortfolioRequest(
        market=request.market, trades=list(request.portfolio),
        config=RunConfig(simulation=config.cam() if exposure else None, pricing=config.pricing,
                         greeks=config.sensitivity, precision=config.precision, base_currency=config.base_currency),
        pfe_quantiles=config.exposure.pfe_quantiles, compute_greeks=request.wants("sensitivities"),
        scenario_risk=exposure, include_npv_cube=request.include_npv_cube)


def market_risk_request(request: RunRequest, scenarios: ShockScenarios) -> MarketRiskRequest:
    """The `run_market_risk` request of `request`'s `market_risk`, on its drawn `scenarios`."""
    config, risk = request.config, request.config.market_risk
    return MarketRiskRequest(trades=list(request.portfolio), market=request.market, scenarios=scenarios,
                             pricing=config.pricing, quantiles=risk.quantiles, precision=config.precision,
                             batch_size=risk.batch_size, base_currency=config.base_currency)


# --- the check before any work ------------------------------------------------------------------

#: Which analytics read each section of the configuration (beyond those that need it).
_READERS = {
    "base_currency": TRADE_ANALYTICS,
    "models": ("exposure", "calibration"),
    "simulation": ("exposure",),
    "pricing": TRADE_ANALYTICS,
    "sensitivity": ("sensitivities",),
    "exposure": ("exposure",),
    "market_risk": ("market_risk",),
    "precision": ("exposure", "market_risk"),
}


def validate_run(request: RunRequest) -> None:
    """Refuse, before any work, a request `run` cannot run as given (see the module docstring)."""
    _check_analytics(request.analytics)
    _check_sections(request)
    config = request.config
    if request.wants(*PORTFOLIO_ANALYTICS):
        from engine.run.pipeline import validate_request

        validate_request(portfolio_request(request))
    if request.wants("market_risk"):
        risk = config.market_risk
        validate_portfolio(list(request.portfolio), request.market, config.pricing, config.precision, risk.quantiles,
                           risk.batch_size, config.base_currency)
        validate_scenarios(list(request.portfolio), request.market, config.precision, risk.scenarios)
    if request.wants("calibration"):
        _check_calibration(request)


def _check_analytics(analytics: Sequence[str]) -> None:
    if not analytics:
        raise ValueError(f"analytics: name at least one of {list(ANALYTICS)}")
    unknown = [a for a in analytics if a not in ANALYTICS]
    if unknown:
        raise ValueError(f"analytics: {unknown} are not analytics; choose from {list(ANALYTICS)}")
    repeated = sorted({a for a in analytics if list(analytics).count(a) > 1})
    if repeated:
        raise ValueError(f"analytics: {repeated} named more than once")


def _check_sections(request: RunRequest) -> None:
    """Every section an analytic needs is given, and nothing is given that no requested analytic
    reads (see the module docstring)."""
    config, defaults = request.config, RunRequestConfig()
    needs = {"models": ("exposure", "calibration"), "simulation": ("exposure",), "market_risk": ("market_risk",)}
    for section, analytics in needs.items():
        for analytic in analytics:
            if request.wants(analytic) and getattr(config, section) is None:
                raise ValueError(f"config.{section}: the {analytic} analytic needs it")
    for section, readers in _READERS.items():
        if getattr(config, section) != getattr(defaults, section) and not request.wants(*readers):
            raise ValueError(_unread(f"config.{section}", readers, request.analytics))
    if request.portfolio and not request.wants(*TRADE_ANALYTICS):
        raise ValueError(_unread("portfolio", TRADE_ANALYTICS, request.analytics))
    if not request.include_npv_cube and not request.wants("exposure"):
        raise ValueError(_unread("include_npv_cube (the NPV cube)", ("exposure",), request.analytics))
    if config.models is not None and not request.wants("exposure"):
        _check_models_for_calibration(config.models, request.analytics)
    if request.wants("market_risk"):
        _check_market_risk_precision(config.precision)


def _unread(what: str, readers: Sequence[str], analytics: Sequence[str]) -> str:
    return (f"{what}: no requested analytic reads it (read by {', '.join(readers)}; requested: "
            f"{', '.join(analytics)}); remove it or request an analytic that reads it")


def _check_models_for_calibration(models: ModelsConfig, analytics: Sequence[str]) -> None:
    """The calibration analytic alone reads each calibrated currency's model and the tolerance:
    a currency without a basket, FX and equity volatilities, and correlations are the
    simulation's (`exposure`)."""
    uncalibrated = [c for c in models.ir if c not in models.calibrated]
    if uncalibrated:
        raise ValueError(_unread(f"config.models.ir{uncalibrated} (no calibration basket, so nothing to calibrate)",
                                 ("exposure",), analytics))
    defaults = ModelsConfig(ir=models.ir)
    for name in ("fx_volatilities", "equity_volatilities", "correlations"):
        if getattr(models, name) != getattr(defaults, name):
            raise ValueError(_unread(f"config.models.{name}", ("exposure",), analytics))


def _check_market_risk_precision(precision: Precision) -> None:
    """Market risk reads the simulation stage's storage and accumulate formats (its shifts) and
    the pricing stage with its overrides (I-95)."""
    if precision.market != StagePrecision():
        raise ValueError("config.precision.market is not read by market risk, which has no scenario market (its "
                         "trades are revalued on shocked curves today); leave it at float64")
    simulation = precision.simulation
    if simulation.compute != simulation.accumulate:
        raise ValueError(f"config.precision.simulation.compute={simulation.compute!r} is not read by market risk: "
                         f"its shifts are drawn before the run and have no products, rounded to the accumulate "
                         f"format {simulation.accumulate!r}; give compute equal to accumulate")


def _check_calibration(request: RunRequest) -> None:
    """At least one currency to calibrate, and the market data each one's basket reads."""
    models, market = request.config.models, request.market
    if not models.calibrated:
        raise ValueError("config.models.ir: no currency has a calibration basket (calibration_expiries and "
                         "calibration_terms), so the calibration analytic has nothing to calibrate")
    for currency in models.calibrated:
        model = models.ir[currency]
        try:
            market.currency(currency)
            market.index_curve(currency, index_name(currency, model.swap_index.index_tenor_months))
            market.swaption_vols(currency)
        except KeyError as exc:
            raise KeyError(f"config.models.ir[{currency!r}]'s calibration basket: {exc.args[0]}") from None


__all__ = [
    "ANALYTICS", "ExposureConfig", "MarketRiskConfig", "ModelsConfig", "RunRequest", "RunRequestConfig", "RunResult",
    "SimulationParameters", "market_risk_request", "portfolio_request", "run", "validate_run",
]
