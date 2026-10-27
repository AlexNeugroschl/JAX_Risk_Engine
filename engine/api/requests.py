"""
The run request on the wire (decision A-2, I-87): `RunRequestSchema` (`POST /runs`) mirrors
`engine.run.RunRequest`, one request for every analytic, reaching every setting the engine has
(`tests/test_api_completeness.py` fails on a setting without a field, and on a field whose value
does not arrive in the engine's configuration):

    {"market": ..., "portfolio": [...], "analytics": ["npv", "exposure", ...],
     "config": {"base_currency", "models", "simulation", "pricing", "sensitivity", "exposure",
                "market_risk", "precision"},
     "outputs": {"npv_cube", "exercised", "pnl"}, "idempotency_key": ..., "keep": false}

  * `market`: today's market (curves per currency and index, swaption volatilities, FX and
    equity spots);
  * `portfolio`: trades that name their currency and index but carry no model or curve of their
    own (audit A-3, I-63), each valued on `market.asof`;
  * `config`: one section per ORE file (`engine.run.RunRequestConfig`): the model per currency
    (`"model": "LGM"`, the default, or `"HullWhite"`) in `models`, the simulation grid in
    `simulation`, the engine per product, the Greeks method and ORE's sensitivity settings, the
    PFE quantiles, the market-risk scenarios (Monte Carlo from a covariance, or historical from
    a history of the factors), and the precision;
  * `outputs`: how the result carries each large array: `artifact` (a chunked, hashed reference,
    `engine.api.artifacts`, the default for the cube and the P&L since this revision of the
    contract, decision A-17), `inline`, or `none` (the default for the exercise decisions, which
    are returned on request);
  * `idempotency_key`: a retried submission with the same key and body is the same run (I-89);
  * `keep`: the run is kept by the job queue's retention policy indefinitely (I-76).

Conventions: ISO dates, ORE periods ("5Y"), fixings `{"YYYY-MM-DD": rate}`. Unknown fields are
refused (422) in every model, so a trade carrying `hw_sigma` or a curve is not silently stripped
of its model (I-103). A value the engine refuses when it is built is a 400 naming its field
(`market.currencies.USD.discount_curve: ...`).
"""
import dataclasses
import hashlib
import json
from typing import Annotated, Callable, ClassVar, Dict, List, Literal, Optional, Tuple, TypeVar, Union

import jax.numpy as jnp
import numpy as np
import ORE
from pydantic import BaseModel, ConfigDict, Field, model_validator

from engine.api.artifacts import ARRAY_OUTPUTS
from engine.calibration.lgm import BOOTSTRAP_TOLERANCE, SwapIndexConventions
from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig
from engine.instruments.treasury import BondConfig, CouponPeriod
from engine.market_data.market import CurrencyMarket, EquityMarket, Market, SwaptionVolSurface, ZeroCurveConfig
from engine.market_simulation.config import DEFAULT_CURVE_TENORS, HullWhiteConfig, LgmConfig
from engine.models.lgm import Sigma
from engine.precision import (
    FORMAT_NAMES, OFFSETS, OVERRIDES, RETIRED_SHAPE as RETIRED_PRECISION_SHAPE, ROUNDINGS, STAGES, Precision,
    StagePrecision, ExecutionPolicy, InternalPrecision,
)
from engine.pricing.config import JamshidianEngineConfig, LgmSwaptionEngineConfig, PricingConfig
from engine.pricing.cube import DEFAULT_REPORTING_CURRENCY, PRODUCTS
from engine.risk.market import MonteCarloScenarios, RateRiskFactors, historical_scenarios
from engine.risk.greeks.bump import SensitivityConfig
from engine.run import (
    ANALYTICS, ExposureConfig, GreeksConfig, MarketRiskConfig, ModelsConfig, RunRequest, RunRequestConfig,
    SimulationParameters, validate_run,
)
from engine.solvers.roots import DEFAULT_SOLVER, SOLVERS

_Built = TypeVar("_Built")


def _named(where: str, build: Callable[[], _Built]) -> _Built:
    """`build()`, a refusal of the engine's own validation naming the wire field `where`."""
    try:
        return build()
    except (ValueError, TypeError) as exc:
        raise type(exc)(f"{where}: {exc}") from None


def _parse_ore_date(value: str) -> ORE.Date:
    try:
        return ORE.DateParser.parseISO(value)
    except Exception as exc:
        raise ValueError(f"not a valid ISO date (YYYY-MM-DD): {value!r} ({exc})") from exc


def _parse_ore_period(value: str) -> ORE.Period:
    try:
        return ORE.Period(value)
    except Exception as exc:
        raise ValueError(f"not a valid ORE period string (e.g. '5Y', '18M'): {value!r} ({exc})") from exc


def _parse_optional_date(value: Optional[str]) -> Optional[ORE.Date]:
    return _parse_ore_date(value) if value is not None else None


def _parse_fixings(fixings: Dict[str, float]) -> Dict[ORE.Date, float]:
    return {_parse_ore_date(date): rate for date, rate in fixings.items()}


class _Strict(BaseModel):
    """Refuses unknown fields (see the module docstring)."""
    model_config = ConfigDict(extra="forbid")


class ZeroCurveConfigSchema(_Strict):
    """`engine.market_data.market.ZeroCurveConfig`: pillar times and continuously compounded zero
    rates, checked when built (non-empty, equal lengths, finite, times non-negative and strictly
    increasing; I-103)."""
    times: List[float]
    rates: List[float]

    def to_dataclass(self, where: str = "zero curve") -> ZeroCurveConfig:
        return _named(where, lambda: ZeroCurveConfig(times=self.times, rates=self.rates))


class CouponPeriodSchema(_Strict):
    """One explicit coupon period of a bond; enumerated, not generated (see
    `engine.instruments.treasury.CouponPeriod`)."""
    start_date: str
    end_date: str
    #: Defaults to `end_date`.
    payment_date: Optional[str] = None

    def to_dataclass(self) -> CouponPeriod:
        return CouponPeriod(_parse_ore_date(self.start_date), _parse_ore_date(self.end_date),
                            _parse_optional_date(self.payment_date))


#: A number format of `engine.precision.FORMATS`, by name.
FormatName = Literal[FORMAT_NAMES]

#: A rounding into a scaled storage format (`engine.precision.ROUNDINGS`).
RoundingName = Literal[ROUNDINGS]

#: A scaled storage format's block layout (`engine.precision.OFFSETS`).
OffsetName = Literal[OFFSETS]

#: A product the pipeline prices (`engine.pricing.cube.PRODUCTS`, the trades' `trade_type`).
ProductName = Literal[PRODUCTS]

#: A root solver of calibrations and exercise boundaries (`engine.solvers.roots.SOLVERS`, decision A-21).
SolverName = Literal[SOLVERS]

#: How a result carries a large array (`engine.api.artifacts.ARRAY_OUTPUTS`, decision A-17).
ArrayOutputName = Literal[ARRAY_OUTPUTS]

#: An analytic of the run request (`engine.run.ANALYTICS`).
AnalyticName = Literal[ANALYTICS]


class StagePrecisionSchema(_Strict):
    """`engine.precision.StagePrecision`; validated by the dataclass."""
    storage: FormatName = "float64"
    compute: FormatName = "float64"
    accumulate: FormatName = "float64"
    compute_rounding: RoundingName = "nearest"

    @classmethod
    def from_dataclass(cls, stage: StagePrecision) -> "StagePrecisionSchema":
        return cls(**dataclasses.asdict(stage))

    def to_dataclass(self, where: str = "precision stage") -> StagePrecision:
        return _named(where, lambda: StagePrecision(**self.model_dump()))


class ExecutionPolicySchema(_Strict):
    backend: Literal["default", "cpu", "gpu", "tpu"] = "default"
    mode: Literal["native", "allow_emulation"] = "native"

    def to_dataclass(self) -> ExecutionPolicy:
        return ExecutionPolicy(**self.model_dump())


class InternalPrecisionSchema(_Strict):
    """Requirements of existing kernels; FP32 implementations are not available yet."""
    calibration: Literal["float64"] = "float64"
    levels: Literal["float64"] = "float64"
    initial_values: Literal["float64"] = "float64"
    reductions: Literal["float64"] = "float64"
    greeks: Literal["float64"] = "float64"
    output: Literal["float64"] = "float64"

    def to_dataclass(self) -> InternalPrecision:
        return InternalPrecision(**self.model_dump())


class PrecisionSchema(_Strict):
    """`engine.precision.Precision`: storage, compute and accumulate per adjustable stage, each
    float64 by default, and the pricing stage overridden per product (`by_product`, keyed by
    `trade_type`) and per trade (`by_trade`, keyed by `trade_id`); the rounding into a scaled
    storage format, its seed, and its block layout (`offset`: an offset beside each block
    scale, or none; I-75). The 32/64 shape retired on 2026-10-01 is refused, naming the
    replacement (decision A-12). `paired_fraction`: the share of paths also run at float64, whose
    estimates the result's `precision` report carries (decision A-13)."""
    simulation: StagePrecisionSchema = Field(default_factory=StagePrecisionSchema)
    market: StagePrecisionSchema = Field(default_factory=StagePrecisionSchema)
    pricing: StagePrecisionSchema = Field(default_factory=StagePrecisionSchema)
    by_product: Dict[ProductName, StagePrecisionSchema] = Field(default_factory=dict)
    by_trade: Dict[str, StagePrecisionSchema] = Field(default_factory=dict)
    rounding: RoundingName = "nearest"
    rounding_seed: int = Field(default=0, ge=0)
    offset: OffsetName = "block"
    paired_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    internal: InternalPrecisionSchema = Field(default_factory=InternalPrecisionSchema)
    execution: ExecutionPolicySchema = Field(default_factory=ExecutionPolicySchema)

    @model_validator(mode="before")
    @classmethod
    def _refuse_the_retired_shape(cls, data):
        """The 32/64 shape: a `risk` or `calibration` field, or a stage given as a bit count."""
        if isinstance(data, dict) and (any(k in data for k in ("risk", "calibration"))
                                       or any(isinstance(data.get(s), int) for s in STAGES)):
            raise ValueError(RETIRED_PRECISION_SHAPE)
        return data

    def to_dataclass(self, where: str = "precision") -> Precision:
        stages = {stage: getattr(self, stage).to_dataclass(f"{where}.{stage}") for stage in STAGES}
        overrides = {name: {key: value.to_dataclass(f"{where}.{name}[{key!r}]")
                            for key, value in getattr(self, name).items()} for name in OVERRIDES}
        return _named(where, lambda: Precision(**stages, **overrides, rounding=self.rounding,
                                               rounding_seed=self.rounding_seed, offset=self.offset,
                                               paired_fraction=self.paired_fraction,
                                               internal=self.internal.to_dataclass(),
                                               execution=self.execution.to_dataclass()))

    @classmethod
    def from_dataclass(cls, precision: Precision) -> "PrecisionSchema":
        stage = StagePrecisionSchema.from_dataclass
        return cls(**{s: stage(getattr(precision, s)) for s in STAGES},
                   **{name: {k: stage(v) for k, v in getattr(precision, name).items()} for name in OVERRIDES},
                   rounding=precision.rounding, rounding_seed=precision.rounding_seed, offset=precision.offset,
                   paired_fraction=precision.paired_fraction,
                   internal=InternalPrecisionSchema(**dataclasses.asdict(precision.internal)),
                   execution=ExecutionPolicySchema(**dataclasses.asdict(precision.execution)))


# --- today's market -----------------------------------------------------------------------------

class SwaptionVolSurfaceSchema(_Strict):
    """ATM normal volatilities, `vols[i][j]` for `option_tenors[i]` x `swap_tenors[j]`."""
    option_tenors: List[str]
    swap_tenors: List[str]
    vols: List[List[float]]

    def to_dataclass(self, where: str = "swaption_vols") -> SwaptionVolSurface:
        return _named(where, lambda: SwaptionVolSurface(tuple(self.option_tenors), tuple(self.swap_tenors),
                                                        tuple(tuple(row) for row in self.vols)))


class CurrencyMarketSchema(_Strict):
    discount_curve: ZeroCurveConfigSchema
    #: Forwarding curve per index name, e.g. {"USD-SIMINDEX-6M": {...}}.
    index_curves: Dict[str, ZeroCurveConfigSchema] = Field(default_factory=dict)
    swaption_vols: Optional[SwaptionVolSurfaceSchema] = None

    def to_dataclass(self, where: str = "currency market") -> CurrencyMarket:
        return CurrencyMarket(
            discount_curve=self.discount_curve.to_dataclass(f"{where}.discount_curve"),
            index_curves={k: v.to_dataclass(f"{where}.index_curves.{k}") for k, v in self.index_curves.items()},
            swaption_vols=self.swaption_vols.to_dataclass(f"{where}.swaption_vols") if self.swaption_vols else None)


class EquityMarketSchema(_Strict):
    currency: str
    spot: float
    dividend_curve: Optional[ZeroCurveConfigSchema] = None

    def to_dataclass(self, where: str = "equity") -> EquityMarket:
        return EquityMarket(self.currency, self.spot,
                            self.dividend_curve.to_dataclass(f"{where}.dividend_curve") if self.dividend_curve else None)


class MarketSchema(_Strict):
    """`engine.market_data.market.Market`. `fx_spots` are keyed `<foreign><domestic>`, e.g. "EURUSD"."""
    asof: str
    currencies: Dict[str, CurrencyMarketSchema]
    fx_spots: Dict[str, float] = Field(default_factory=dict)
    equities: Dict[str, EquityMarketSchema] = Field(default_factory=dict)

    def to_dataclass(self) -> Market:
        currencies = {k: v.to_dataclass(f"market.currencies.{k}") for k, v in self.currencies.items()}
        equities = {k: v.to_dataclass(f"market.equities.{k}") for k, v in self.equities.items()}
        return _named("market", lambda: Market(asof=_parse_ore_date(self.asof), currencies=currencies,
                                               fx_spots=dict(self.fx_spots), equities=equities))


# --- models -------------------------------------------------------------------------------------

class SwapIndexConventionsSchema(_Strict):
    fixed_tenor: str = "1Y"
    fixed_day_counter: str = "ACT/365"
    index_tenor_months: int = 6

    def to_dataclass(self) -> SwapIndexConventions:
        return SwapIndexConventions(**self.model_dump())


class PiecewiseVolatilitySchema(_Strict):
    """`engine.models.lgm.Sigma`: a piecewise-constant volatility, `values[i]` on bucket `i` of
    `[0, times[0]), [times[0], times[1]), ..., [times[-1], inf)` (ORE's `VolatilityTimes` and
    `Volatility`), so one more value than times."""
    times: List[float]
    values: List[float]

    @model_validator(mode="after")
    def _buckets(self):
        times, values = np.asarray(self.times, dtype=np.float64), np.asarray(self.values, dtype=np.float64)
        if len(values) != len(times) + 1:
            raise ValueError(f"a piecewise volatility has one more value than times; got {len(times)} times and "
                             f"{len(values)} values")
        if not np.all(np.isfinite(times)) or np.any(times <= 0.0) or np.any(np.diff(times) <= 0.0):
            raise ValueError("a piecewise volatility's times must be finite, positive and increasing")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("a piecewise volatility's values must be finite and non-negative")
        return self

    def to_dataclass(self) -> Sigma:
        return Sigma(jnp.asarray(self.times, dtype=jnp.float64), jnp.asarray(self.values, dtype=jnp.float64))


#: A volatility: a constant, or piecewise constant (`engine.market_simulation.config.Volatility`).
VolatilitySchema = Union[float, PiecewiseVolatilitySchema]


def _volatility(value: VolatilitySchema):
    return value.to_dataclass() if isinstance(value, PiecewiseVolatilitySchema) else value


class LgmConfigSchema(_Strict):
    """One currency's CAM LGM (`LgmConfig`, the default model): constant reversion, and a
    volatility (constant or piecewise) bootstrapped to the `calibration_expiries` x
    `calibration_terms` basket when one is given."""
    model: Literal["LGM"] = "LGM"
    reversion: float
    volatility: VolatilitySchema = 0.01
    calibration_expiries: List[str] = Field(default_factory=list)
    calibration_terms: List[str] = Field(default_factory=list)
    swap_index: SwapIndexConventionsSchema = Field(default_factory=SwapIndexConventionsSchema)
    #: The bootstrap's root solver (decision A-21).
    solver: SolverName = DEFAULT_SOLVER

    #: The configuration type this schema builds.
    _config = LgmConfig

    def to_dataclass(self) -> LgmConfig:
        return self._config(self.reversion, _volatility(self.volatility), tuple(self.calibration_expiries),
                            tuple(self.calibration_terms), self.swap_index.to_dataclass(), self.solver)


class HullWhiteConfigSchema(LgmConfigSchema):
    """One currency's Hull-White model (`HullWhiteConfig`): `volatility` is the short rate's;
    the same calibration basket and conventions as the LGM."""
    model: Literal["HullWhite"]

    _config = HullWhiteConfig


#: One currency's model in `config.models.ir`.
IrModelSchema = Union[LgmConfigSchema, HullWhiteConfigSchema]


class CorrelationSchema(_Strict):
    """One instantaneous correlation between two factors (`IR:USD`, `FX:EURUSD`, `EQ:SP5`)."""
    factor1: str
    factor2: str
    value: float


class ModelsSchema(_Strict):
    """`engine.run.ModelsConfig`, simulation.xml's `CrossAssetModel`: the model per currency, the
    FX volatility of every currency but `config.base_currency`, the equities', the correlations,
    and ORE's `BootstrapTolerance` (one for the model: a currency's calibration whose error, the
    root mean square of its helpers' relative price errors, is not below it is refused, I-73)."""
    ir: Dict[str, IrModelSchema]
    fx_volatilities: Dict[str, VolatilitySchema] = Field(default_factory=dict)
    equity_volatilities: Dict[str, VolatilitySchema] = Field(default_factory=dict)
    correlations: List[CorrelationSchema] = Field(default_factory=list)
    bootstrap_tolerance: float = BOOTSTRAP_TOLERANCE

    def to_dataclass(self) -> ModelsConfig:
        ir = {k: _named(f"config.models.ir.{k}", v.to_dataclass) for k, v in self.ir.items()}
        return _named("config.models", lambda: ModelsConfig(
            ir=ir, fx_volatilities={k: _volatility(v) for k, v in self.fx_volatilities.items()},
            equity_volatilities={k: _volatility(v) for k, v in self.equity_volatilities.items()},
            correlations={(c.factor1, c.factor2): c.value for c in self.correlations},
            bootstrap_tolerance=self.bootstrap_tolerance))


class SimulationSchema(_Strict):
    """`engine.run.SimulationParameters`, simulation.xml's `Parameters` and `Market`: the grid's
    `dates`, `samples` paths in `replicates` independent scrambles from `seed`, the scenario
    market's `curve_tenors`, and the swaption volatilities' `ReactionToTimeDecay`."""
    dates: List[str]
    samples: int = 1000
    seed: int = 42
    replicates: int = Field(default=1, ge=1, strict=True)
    curve_tenors: List[str] = Field(default_factory=lambda: list(DEFAULT_CURVE_TENORS))
    swaption_vol_decay: Literal["ForwardVariance", "ConstantVariance"] = "ForwardVariance"

    def to_dataclass(self) -> SimulationParameters:
        return SimulationParameters(dates=tuple(_parse_ore_date(d) for d in self.dates), samples=self.samples,
                                    seed=self.seed, replicates=self.replicates, curve_tenors=tuple(self.curve_tenors),
                                    swaption_vol_decay=self.swaption_vol_decay)


# --- pricing and sensitivities ------------------------------------------------------------------

class LgmEngineSchema(_Strict):
    """`engine.pricing.config.LgmSwaptionEngineConfig` (ORE's LGM Grid engine)."""
    reversion: float = 0.0
    volatility: float = 0.01
    calibration: Literal["Bootstrap", "None"] = "Bootstrap"
    strategy: Literal["CoterminalDealStrike", "CoterminalATM"] = "CoterminalDealStrike"
    #: ORE's `Tolerance`: a calibration whose error (the root mean square of its helpers' relative
    #: price errors) is not below it is refused, naming the trade (I-73).
    bootstrap_tolerance: float = BOOTSTRAP_TOLERANCE
    reference_calibration_grid: str = "400,3M"
    #: ORE's `ShiftHorizon`: the fraction of the time to the underlying's maturity where the LGM's
    #: H is shifted to vanish (any value >= 0; 0 is the unshifted model).
    shift_horizon: float = 0.5
    n_per_std: int = 30
    std_devs: float = 5.0
    exercise_time_steps_per_year: int = 24
    swap_index: SwapIndexConventionsSchema = Field(default_factory=SwapIndexConventionsSchema)
    #: The root solver of the calibration and of each helper's exercise boundary (decision A-21).
    solver: SolverName = DEFAULT_SOLVER

    def to_dataclass(self) -> LgmSwaptionEngineConfig:
        fields = self.model_dump(exclude={"swap_index"})
        return LgmSwaptionEngineConfig(**fields, swap_index=self.swap_index.to_dataclass())


class JamshidianEngineSchema(_Strict):
    """`engine.pricing.config.JamshidianEngineConfig`: the Jamshidian engine's Hull-White model."""
    reversion: float
    volatility: float
    #: The root solver of the exercise boundary x* (decision A-21).
    solver: SolverName = DEFAULT_SOLVER

    def to_dataclass(self) -> JamshidianEngineConfig:
        return JamshidianEngineConfig(self.reversion, self.volatility, self.solver)


class PricingConfigSchema(_Strict):
    """`engine.pricing.config.PricingConfig`, pricingengine.xml: the engine per product.
    `jamshidian` is required with `european: "Jamshidian"` and refused without it."""
    european: Literal["Bachelier", "Jamshidian"] = "Bachelier"
    jamshidian: Optional[JamshidianEngineSchema] = None
    bermudan: LgmEngineSchema = Field(default_factory=LgmEngineSchema)
    american: LgmEngineSchema = Field(default_factory=LgmEngineSchema)

    def to_dataclass(self) -> PricingConfig:
        jamshidian = _named("config.pricing.jamshidian", self.jamshidian.to_dataclass) if self.jamshidian else None
        bermudan = _named("config.pricing.bermudan", self.bermudan.to_dataclass)
        american = _named("config.pricing.american", self.american.to_dataclass)
        return _named("config.pricing", lambda: PricingConfig(european=self.european, jamshidian=jamshidian,
                                                             bermudan=bermudan, american=american))


class SensitivitySchema(_Strict):
    """`engine.run.GreeksConfig`, sensitivity.xml and the Greeks method in one section: ORE's
    bump-and-revalue (`Bump`, the default) or `AD`, on the sensitivity market's curve tenors
    (also the shift tenors), with absolute shift sizes and the Theta horizon
    (`engine.risk.greeks.bump.SensitivityConfig`)."""
    method: Literal["Bump", "AD"] = "Bump"
    curve_tenors: List[str] = Field(default_factory=lambda: list(SensitivityConfig().curve_tenors))
    curve_shift: float = 1e-4
    vol_shift: float = 1e-4
    theta_days: int = 1

    def to_dataclass(self) -> GreeksConfig:
        settings = SensitivityConfig(curve_tenors=tuple(self.curve_tenors), curve_shift=self.curve_shift,
                                     vol_shift=self.vol_shift, theta_days=self.theta_days)
        return _named("config.sensitivity", lambda: GreeksConfig(method=self.method, sensitivity=settings))


class ExposureSchema(_Strict):
    """`engine.run.ExposureConfig`: the PFE quantiles, each in (0, 1) and given once."""
    pfe_quantiles: List[float] = Field(default_factory=lambda: list(ExposureConfig().pfe_quantiles))

    def to_dataclass(self) -> ExposureConfig:
        return ExposureConfig(pfe_quantiles=tuple(self.pfe_quantiles))


# --- trades -------------------------------------------------------------------------------------

class _Trade(_Strict):
    #: The trade's id (ORE's `<Trade id>`), unique in the request and echoed on every per-trade
    #: figure of the result (I-10). Give one on every trade or on none; none numbers them
    #: `trade-0`, `trade-1`, ... in request order.
    trade_id: Optional[str] = None


class _SwapTerms(_Trade):
    """What every swap-bearing trade names: currency, schedule, index, spread."""
    notional: float
    fixed_rate: float
    payer: bool
    currency: str = "USD"
    effective_date: Optional[str] = None
    maturity_date: Optional[str] = None
    swap_tenor: Optional[str] = None
    index_tenor_months: int = 6
    floating_spread: float = 0.0

    def _common(self, evaluation_date, trade_id):
        return dict(notional=self.notional, fixed_rate=self.fixed_rate, payer=self.payer, currency=self.currency,
                    effective_date=_parse_optional_date(self.effective_date),
                    maturity_date=_parse_optional_date(self.maturity_date), swap_tenor=self.swap_tenor,
                    index_tenor_months=self.index_tenor_months, floating_spread=self.floating_spread,
                    evaluation_date=evaluation_date, trade_id=trade_id)


class SwapTradeSchema(_SwapTerms):
    trade_type: Literal["swap"] = "swap"
    accrual_day_count: str = "ACT/365"
    fixings: Dict[str, float] = Field(default_factory=dict)

    def to_dataclass(self, evaluation_date, trade_id) -> SwapConfig:
        return SwapConfig(**self._common(evaluation_date, trade_id), accrual_day_count=self.accrual_day_count,
                          fixings=_parse_fixings(self.fixings))


class _OptionTerms(_SwapTerms):
    """What every swaption names beyond its swap: ORE's `OptionData` `Settlement` and
    `LongShort`. A short option is `Short` on a positive notional; a zero or negative option
    notional is refused, naming the trade (I-101)."""
    settlement: Literal["Physical", "Cash"] = "Physical"
    long_short: Literal["Long", "Short"] = "Long"

    def _common(self, evaluation_date, trade_id):
        return dict(super()._common(evaluation_date, trade_id), settlement=self.settlement,
                    long_short=self.long_short)


class EuropeanTradeSchema(_OptionTerms):
    trade_type: Literal["european_swaption"] = "european_swaption"
    exercise_date: Optional[str] = None
    forward_start: Optional[str] = None
    exercise_lag_days: Optional[int] = None

    def to_dataclass(self, evaluation_date, trade_id) -> SwaptionConfig:
        return SwaptionConfig(**self._common(evaluation_date, trade_id),
                              exercise_date=_parse_optional_date(self.exercise_date),
                              forward_start=_parse_ore_period(self.forward_start) if self.forward_start else None,
                              exercise_lag_days=self.exercise_lag_days)


class BermudanTradeSchema(_OptionTerms):
    trade_type: Literal["bermudan_swaption"] = "bermudan_swaption"
    exercise_dates: List[str]
    fixings: Dict[str, float] = Field(default_factory=dict)

    def to_dataclass(self, evaluation_date, trade_id) -> BermudanSwaptionConfig:
        return BermudanSwaptionConfig(**self._common(evaluation_date, trade_id),
                                      exercise_dates=[_parse_ore_date(d) for d in self.exercise_dates],
                                      fixings=_parse_fixings(self.fixings))


class AmericanTradeSchema(_OptionTerms):
    trade_type: Literal["american_swaption"] = "american_swaption"
    first_exercise_date: str
    last_exercise_date: str
    fixings: Dict[str, float] = Field(default_factory=dict)

    def to_dataclass(self, evaluation_date, trade_id) -> AmericanSwaptionConfig:
        return AmericanSwaptionConfig(**self._common(evaluation_date, trade_id),
                                      first_exercise_date=_parse_ore_date(self.first_exercise_date),
                                      last_exercise_date=_parse_ore_date(self.last_exercise_date),
                                      fixings=_parse_fixings(self.fixings))


class BondTradeSchema(_Trade):
    """A Treasury bill (no coupon schedule) or note, discounted on its currency's curve."""
    trade_type: Literal["bond"] = "bond"
    face_amount: float
    maturity_date: str
    currency: str = "USD"
    coupon_rate: float = 0.0
    coupon_schedule: List[CouponPeriodSchema] = Field(default_factory=list)
    redemption_fraction: float = 1.0
    accrual_day_count: str = "ACT/ACT (ICMA)"

    def to_dataclass(self, evaluation_date, trade_id) -> BondConfig:
        return BondConfig(
            face_amount=self.face_amount, maturity_date=_parse_ore_date(self.maturity_date),
            evaluation_date=evaluation_date, trade_id=trade_id, currency=self.currency, coupon_rate=self.coupon_rate,
            coupon_schedule=tuple(p.to_dataclass() for p in self.coupon_schedule),
            redemption_fraction=self.redemption_fraction, accrual_day_count=self.accrual_day_count)


MarketTradeSchema = Annotated[
    Union[SwapTradeSchema, EuropeanTradeSchema, BermudanTradeSchema, AmericanTradeSchema, BondTradeSchema],
    Field(discriminator="trade_type"),
]


def _trades(trades: List[MarketTradeSchema], market: Market) -> list:
    """The trade configs, each valued on the market's date. `trade_id` on every trade or on none;
    none numbers them `trade-0`, `trade-1`, ... in request order. A trade the engine refuses
    when it is built is named by its position and id."""
    ids = [t.trade_id for t in trades]
    if any(i is not None for i in ids) and None in ids:
        raise ValueError("portfolio: give trade_id on every trade or on none")
    if not ids or ids[0] is None:
        ids = [f"trade-{i}" for i in range(len(trades))]
    return [_named(f"portfolio[{n}] ({i!r})", lambda t=t, i=i: t.to_dataclass(market.asof, i))
            for n, (t, i) in enumerate(zip(trades, ids))]


# --- market risk --------------------------------------------------------------------------------

class _ScenariosSchema(_Strict):
    """What both scenario sources share. `factors` names the risk factors, the pillar zero rates of
    market curves, in the order of the covariance's or the history's columns: each curve's labels
    `"<curve>/<pillar time>y"` for every pillar in the market's order, curve after curve
    (`RateRiskFactors.labels`, e.g. `"discount:USD/1y"`, `"index:USD-SIMINDEX-6M/1y"`). Any curves
    of the market, in any order; every curve a trade reads must be among them."""
    factors: List[str]
    horizon_days: int

    def risk_factors(self, market: Market) -> RateRiskFactors:
        """The market's curves `factors` names, checked pillar by pillar."""
        every = RateRiskFactors.from_market(market)
        curves = dict(zip(every.names, every.curves))
        names = list(dict.fromkeys(label.rpartition("/")[0] for label in self.factors))
        unknown = [n for n in names if n not in curves]
        if unknown:
            raise ValueError(f"config.market_risk.scenarios.factors: {unknown} are not curves of the market; its "
                             f"curves are {list(every.names)}")
        factors = RateRiskFactors.from_curves([curves[n] for n in names], names)
        if factors.labels() != list(self.factors):
            raise ValueError(f"config.market_risk.scenarios.factors must give each curve's pillars in the market's "
                             f"order, curve after curve: {factors.labels()}; got {list(self.factors)}")
        return factors


class MonteCarloScenariosSchema(_ScenariosSchema):
    """`engine.risk.market.MonteCarloScenarios`: `num_scenarios` zero-mean Gaussian moves over
    `horizon_days` with the `covariance` of the factors (`[F][F]`, in `factors`' order), scrambled
    Sobol normals seeded by `seed`, in `replicates` independent scrambles; drawn when the run
    starts, in the engine worker."""
    source: Literal["monte-carlo"]
    covariance: List[List[float]]
    num_scenarios: int
    seed: int = 42
    replicates: int = Field(default=1, ge=1, strict=True)

    def to_dataclass(self, market: Market) -> MonteCarloScenarios:
        factors = self.risk_factors(market)
        return _named("config.market_risk.scenarios", lambda: MonteCarloScenarios(
            factors, self.covariance, self.horizon_days, self.num_scenarios, self.seed, self.replicates))


class HistoricalScenariosSchema(_ScenariosSchema):
    """`engine.risk.market.historical_scenarios`: one scenario per overlapping `horizon_days`
    window of `history` (`[D][F]` factor levels on consecutive business dates, oldest first, in
    `factors`' order), each labelled with its window when `dates` (`D` labels) are given; the ES
    standard error by a moving-block bootstrap of `bootstrap_replicates` resamples of blocks of
    `block_length` from `bootstrap_seed`."""
    source: Literal["historical"]
    history: List[List[float]]
    dates: Optional[List[str]] = None
    bootstrap_replicates: int = Field(default=128, ge=2, strict=True)
    block_length: Optional[int] = Field(default=None, ge=1, strict=True)
    bootstrap_seed: int = Field(default=42, ge=0, strict=True)

    def to_dataclass(self, market: Market):
        factors = self.risk_factors(market)
        return _named("config.market_risk.scenarios", lambda: historical_scenarios(
            factors, self.history, self.horizon_days, self.dates,
            bootstrap_replicates=self.bootstrap_replicates, block_length=self.block_length,
            bootstrap_seed=self.bootstrap_seed))


ScenariosSchema = Annotated[Union[MonteCarloScenariosSchema, HistoricalScenariosSchema],
                            Field(discriminator="source")]


class MarketRiskSchema(_Strict):
    """`engine.run.MarketRiskConfig`: the shock `scenarios`, the VaR/ES `quantiles`, and the
    scenarios revalued at once (`batch_size`; lower it if a large Bermudan runs out of memory)."""
    scenarios: ScenariosSchema
    quantiles: List[float] = Field(default_factory=lambda: [0.99, 0.975])
    batch_size: int = 256

    def to_dataclass(self, market: Market) -> MarketRiskConfig:
        scenarios = self.scenarios.to_dataclass(market)
        return MarketRiskConfig(scenarios=scenarios, quantiles=tuple(self.quantiles), batch_size=self.batch_size)


# --- the run request ----------------------------------------------------------------------------

class RunRequestConfigSchema(_Strict):
    """`engine.run.RunRequestConfig`: how the run computes its analytics, one section per ORE file
    (see the module docstring). Every section is optional; the defaults are ORE's
    (`GET /config/defaults`). A section, or a setting differing from its default, that no
    requested analytic reads is refused (400), naming it."""
    base_currency: str = DEFAULT_REPORTING_CURRENCY
    models: Optional[ModelsSchema] = None
    simulation: Optional[SimulationSchema] = None
    pricing: PricingConfigSchema = Field(default_factory=PricingConfigSchema)
    sensitivity: SensitivitySchema = Field(default_factory=SensitivitySchema)
    exposure: ExposureSchema = Field(default_factory=ExposureSchema)
    market_risk: Optional[MarketRiskSchema] = None
    precision: PrecisionSchema = Field(default_factory=PrecisionSchema)

    def to_dataclass(self, market: Market) -> RunRequestConfig:
        sections = dict(
            models=self.models.to_dataclass() if self.models else None,
            simulation=_named("config.simulation", self.simulation.to_dataclass) if self.simulation else None,
            pricing=self.pricing.to_dataclass(), sensitivity=self.sensitivity.to_dataclass(),
            exposure=_named("config.exposure", self.exposure.to_dataclass),
            market_risk=self.market_risk.to_dataclass(market) if self.market_risk else None,
            precision=self.precision.to_dataclass("config.precision"))
        return RunRequestConfig(base_currency=self.base_currency, **sections)


class OutputsSchema(_Strict):
    """How the result carries each large array (decision A-17, revised with this contract): the
    exposure analytic's NPV cube `[scenario, date, trade]` (`npv_cube`) and each option's exercise
    decisions `[scenario, date, option]` (`exercised`), the market-risk P&L `[scenario, trade]`
    (`pnl`). `artifact` is a chunked, hashed reference read at `GET /runs/{run_id}/artifacts/...`;
    `none` leaves the array out (and the exposure analytic then never assembles its float64 cube,
    I-105). A setting other than its default for an analytic not requested is refused."""
    npv_cube: ArrayOutputName = "artifact"
    exercised: ArrayOutputName = "none"
    pnl: ArrayOutputName = "artifact"

    #: The analytic that produces each array.
    PRODUCED_BY: ClassVar[Dict[str, str]] = {"npv_cube": "exposure", "exercised": "exposure", "pnl": "market_risk"}


class RunRequestSchema(_Strict):
    """The run request (see the module docstring): `engine.run.RunRequest` on the wire, plus what
    only the job queue reads (`idempotency_key`, `keep`)."""
    market: MarketSchema
    portfolio: List[MarketTradeSchema] = Field(default_factory=list)
    analytics: List[AnalyticName]
    config: RunRequestConfigSchema = Field(default_factory=RunRequestConfigSchema)
    outputs: OutputsSchema = Field(default_factory=OutputsSchema)
    #: A caller's key for this submission (I-89): the same key with the same body is the run
    #: already submitted, whose id is returned again; with another body it is refused (409).
    idempotency_key: Optional[str] = Field(default=None, min_length=1, max_length=256)
    #: Keep the run, its result and its artifacts whatever the queue's retention policy (I-76),
    #: as a regulatory run must be kept.
    keep: bool = False

    def to_dataclass(self) -> RunRequest:
        market = self.market.to_dataclass()
        portfolio, config = tuple(_trades(self.portfolio, market)), self.config.to_dataclass(market)
        return _named("portfolio", lambda: RunRequest(market=market, analytics=tuple(self.analytics),
                                                      portfolio=portfolio, config=config,
                                                      include_npv_cube=self.outputs.npv_cube != "none"))

    def check(self) -> RunRequest:
        """Refuse what `engine.run.run` would refuse, before the run is queued (`validate_run`: no
        JAX work, the Monte Carlo scenarios not drawn), and an output of an analytic not
        requested; the request, built."""
        request = self.to_dataclass()
        validate_run(request)
        default = OutputsSchema()
        for name, analytic in OutputsSchema.PRODUCED_BY.items():
            if getattr(self.outputs, name) != getattr(default, name) and analytic not in self.analytics:
                raise ValueError(f"outputs.{name}: the {analytic} analytic, which produces it, is not requested")
        return request

    def body_sha256(self) -> str:
        """The request's identity for `idempotency_key` (I-89): the SHA-256 of its canonical JSON,
        every default filled in, the key itself left out, so a retry with its fields reordered or
        a default spelt out is the same body."""
        canonical = json.dumps(self.model_dump(mode="json", exclude={"idempotency_key"}), sort_keys=True,
                               separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("ascii")).hexdigest()


__all__: Tuple[str, ...] = ("RunRequestConfigSchema", "RunRequestSchema")
