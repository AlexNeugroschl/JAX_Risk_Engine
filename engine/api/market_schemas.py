"""
The portfolio request: one shape for every configuration of the engine (decision A-2).

Today's market (curves per currency and index, swaption volatilities, FX and equity spots),
trades that name their currency and index but carry no model or curve of their own (audit
A-3, I-63), and the run configuration (`engine.portfolio.RunConfig`): the simulation with the
model of each currency (`"model": "LGM"`, the default, or `"HullWhite"`), the engine per
product (the European's `Bachelier` or `Jamshidian` with its model), the Greeks method and
ORE's sensitivity settings, precision and reporting currency.

`POST /portfolio/price` and `POST /v2/portfolio/price` both take it; the `/v2` and
`schema_version: "2"` are historical names, not versions (docs/reference/http-api.md; roadmap
4.1 retires them). Until roadmap 1.3 `POST /portfolio/price` took the Hull-White model's own
request (a `SimulationConfig` market, model parameters on the trades); that shape is refused
with a message naming its replacement.

Conventions: ISO dates, ORE periods ("5Y"), fixings `{"YYYY-MM-DD": rate}`. Unknown fields are
refused (422), so a trade carrying `hw_sigma` or a curve is not silently stripped of its model.
"""
from typing import Annotated, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from engine.api.schemas import (
    CouponPeriodSchema, PrecisionSchema, ZeroCurveConfigSchema, _parse_fixings, _parse_optional_date,
    _parse_ore_date, _parse_ore_period,
)
from engine.calibration.ore_lgm import SwapIndexConventions
from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig
from engine.instruments.treasury import BondConfig, CouponPeriod
from engine.market import CurrencyMarket, EquityMarket, Market, SwaptionVolSurface
from engine.portfolio import GreeksConfig, PortfolioRequest, RunConfig, SensitivityConfig
from engine.simulation.config import CamConfig, HullWhiteConfig, LgmConfig
from engine.valuation.config import JamshidianEngineConfig, LgmSwaptionEngineConfig, PricingConfig


class _Strict(BaseModel):
    """Refuses unknown fields (see the module docstring)."""
    model_config = ConfigDict(extra="forbid")


class SwaptionVolSurfaceSchema(_Strict):
    """ATM normal volatilities, `vols[i][j]` for `option_tenors[i]` x `swap_tenors[j]`."""
    option_tenors: List[str]
    swap_tenors: List[str]
    vols: List[List[float]]

    def to_dataclass(self) -> SwaptionVolSurface:
        return SwaptionVolSurface(tuple(self.option_tenors), tuple(self.swap_tenors),
                                  tuple(tuple(row) for row in self.vols))


class CurrencyMarketSchema(_Strict):
    discount_curve: ZeroCurveConfigSchema
    #: Forwarding curve per index name, e.g. {"USD-SIMINDEX-6M": {...}}.
    index_curves: Dict[str, ZeroCurveConfigSchema] = Field(default_factory=dict)
    swaption_vols: Optional[SwaptionVolSurfaceSchema] = None

    def to_dataclass(self) -> CurrencyMarket:
        return CurrencyMarket(
            discount_curve=self.discount_curve.to_dataclass(),
            index_curves={k: v.to_dataclass() for k, v in self.index_curves.items()},
            swaption_vols=self.swaption_vols.to_dataclass() if self.swaption_vols else None)


class EquityMarketSchema(_Strict):
    currency: str
    spot: float
    dividend_curve: Optional[ZeroCurveConfigSchema] = None

    def to_dataclass(self) -> EquityMarket:
        return EquityMarket(self.currency, self.spot,
                            self.dividend_curve.to_dataclass() if self.dividend_curve else None)


class MarketSchema(_Strict):
    """`engine.market.Market`. `fx_spots` are keyed `<foreign><domestic>`, e.g. "EURUSD"."""
    asof: str
    currencies: Dict[str, CurrencyMarketSchema]
    fx_spots: Dict[str, float] = Field(default_factory=dict)
    equities: Dict[str, EquityMarketSchema] = Field(default_factory=dict)

    def to_dataclass(self) -> Market:
        return Market(asof=_parse_ore_date(self.asof),
                      currencies={k: v.to_dataclass() for k, v in self.currencies.items()},
                      fx_spots=dict(self.fx_spots),
                      equities={k: v.to_dataclass() for k, v in self.equities.items()})


class SwapIndexConventionsSchema(_Strict):
    fixed_tenor: str = "1Y"
    fixed_day_counter: str = "ACT/365"
    index_tenor_months: int = 6

    def to_dataclass(self) -> SwapIndexConventions:
        return SwapIndexConventions(**self.model_dump())


class LgmConfigSchema(_Strict):
    """One currency's CAM LGM (`LgmConfig`, the default model): constant reversion, and a
    volatility bootstrapped to the `calibration_expiries` x `calibration_terms` basket when one
    is given."""
    model: Literal["LGM"] = "LGM"
    reversion: float
    volatility: float = 0.01
    calibration_expiries: List[str] = Field(default_factory=list)
    calibration_terms: List[str] = Field(default_factory=list)
    swap_index: SwapIndexConventionsSchema = Field(default_factory=SwapIndexConventionsSchema)

    #: The configuration type this schema builds.
    _config = LgmConfig

    def to_dataclass(self) -> LgmConfig:
        return self._config(self.reversion, self.volatility, tuple(self.calibration_expiries),
                            tuple(self.calibration_terms), self.swap_index.to_dataclass())


class HullWhiteConfigSchema(LgmConfigSchema):
    """One currency's Hull-White model (`HullWhiteConfig`): `volatility` is the short rate's;
    the same calibration basket and conventions as the LGM."""
    model: Literal["HullWhite"]

    _config = HullWhiteConfig


class CorrelationSchema(_Strict):
    """One instantaneous correlation between two factors (`IR:USD`, `FX:EURUSD`, `EQ:SP5`)."""
    factor1: str
    factor2: str
    value: float


class CamConfigSchema(_Strict):
    """`engine.simulation.config.CamConfig`."""
    dates: List[str]
    base_currency: str
    ir: Dict[str, Union[LgmConfigSchema, HullWhiteConfigSchema]]
    fx_volatilities: Dict[str, float] = Field(default_factory=dict)
    equity_volatilities: Dict[str, float] = Field(default_factory=dict)
    correlations: List[CorrelationSchema] = Field(default_factory=list)
    curve_tenors: Optional[List[str]] = None
    samples: int = 1000
    seed: int = 42
    swaption_vol_decay: Literal["ForwardVariance", "ConstantVariance"] = "ForwardVariance"

    def to_dataclass(self) -> CamConfig:
        extra = {"curve_tenors": tuple(self.curve_tenors)} if self.curve_tenors else {}
        return CamConfig(
            dates=tuple(_parse_ore_date(d) for d in self.dates), base_currency=self.base_currency,
            ir={k: v.to_dataclass() for k, v in self.ir.items()}, fx_volatilities=dict(self.fx_volatilities),
            equity_volatilities=dict(self.equity_volatilities),
            correlations={(c.factor1, c.factor2): c.value for c in self.correlations},
            samples=self.samples, seed=self.seed, swaption_vol_decay=self.swaption_vol_decay, **extra)


class LgmEngineSchema(_Strict):
    """`engine.valuation.config.LgmSwaptionEngineConfig` (ORE's LGM Grid engine)."""
    reversion: float = 0.0
    volatility: float = 0.01
    calibration: Literal["Bootstrap", "None"] = "Bootstrap"
    strategy: Literal["CoterminalDealStrike", "CoterminalATM"] = "CoterminalDealStrike"
    reference_calibration_grid: str = "400,3M"
    n_per_std: int = 30
    std_devs: float = 5.0
    exercise_time_steps_per_year: int = 24
    swap_index: SwapIndexConventionsSchema = Field(default_factory=SwapIndexConventionsSchema)

    def to_dataclass(self) -> LgmSwaptionEngineConfig:
        fields = self.model_dump(exclude={"swap_index"})
        return LgmSwaptionEngineConfig(**fields, swap_index=self.swap_index.to_dataclass())


class JamshidianEngineSchema(_Strict):
    """`engine.valuation.config.JamshidianEngineConfig`: the Jamshidian engine's Hull-White model."""
    reversion: float
    volatility: float

    def to_dataclass(self) -> JamshidianEngineConfig:
        return JamshidianEngineConfig(self.reversion, self.volatility)


class PricingConfigSchema(_Strict):
    """`engine.valuation.config.PricingConfig`: the engine per product. `jamshidian` is required
    with `european: "Jamshidian"` and refused without it."""
    european: Literal["Bachelier", "Jamshidian"] = "Bachelier"
    jamshidian: Optional[JamshidianEngineSchema] = None
    bermudan: LgmEngineSchema = Field(default_factory=LgmEngineSchema)
    american: LgmEngineSchema = Field(default_factory=LgmEngineSchema)
    recalibrate: bool = True

    def to_dataclass(self) -> PricingConfig:
        return PricingConfig(european=self.european,
                             jamshidian=self.jamshidian.to_dataclass() if self.jamshidian else None,
                             bermudan=self.bermudan.to_dataclass(), american=self.american.to_dataclass(),
                             recalibrate=self.recalibrate)


class SensitivityConfigSchema(_Strict):
    """`engine.risk.sensitivities.SensitivityConfig` (ORE's `sensitivity.xml`)."""
    curve_tenors: Optional[List[str]] = None
    curve_shift: float = 1e-4
    vol_shift: float = 1e-4
    theta_days: int = 1
    swaption_vol_decay: Literal["ForwardVariance", "ConstantVariance"] = "ForwardVariance"

    def to_dataclass(self) -> SensitivityConfig:
        fields = self.model_dump(exclude={"curve_tenors"})
        if self.curve_tenors:
            fields["curve_tenors"] = tuple(self.curve_tenors)
        return SensitivityConfig(**fields)


class GreeksConfigSchema(_Strict):
    """`engine.portfolio.GreeksConfig`: ORE's bump-and-revalue (`Bump`, the default) or `AD`."""
    method: Literal["Bump", "AD"] = "Bump"
    sensitivity: SensitivityConfigSchema = Field(default_factory=SensitivityConfigSchema)

    def to_dataclass(self) -> GreeksConfig:
        return GreeksConfig(method=self.method, sensitivity=self.sensitivity.to_dataclass())


class _Trade(_Strict):
    #: The trade's id (ORE's `<Trade id>`), unique in the request and echoed as `trade_ids` on
    #: the result (I-10). Give one on every trade or on none; none numbers them `trade-0`,
    #: `trade-1`, ... in request order.
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


class EuropeanTradeSchema(_SwapTerms):
    trade_type: Literal["european_swaption"] = "european_swaption"
    exercise_date: Optional[str] = None
    forward_start: Optional[str] = None
    exercise_lag_days: Optional[int] = None
    settlement: Literal["Physical", "Cash"] = "Physical"

    def to_dataclass(self, evaluation_date, trade_id) -> SwaptionConfig:
        return SwaptionConfig(**self._common(evaluation_date, trade_id),
                              exercise_date=_parse_optional_date(self.exercise_date),
                              forward_start=_parse_ore_period(self.forward_start) if self.forward_start else None,
                              exercise_lag_days=self.exercise_lag_days, settlement=self.settlement)


class BermudanTradeSchema(_SwapTerms):
    trade_type: Literal["bermudan_swaption"] = "bermudan_swaption"
    exercise_dates: List[str]
    settlement: Literal["Physical", "Cash"] = "Physical"
    fixings: Dict[str, float] = Field(default_factory=dict)

    def to_dataclass(self, evaluation_date, trade_id) -> BermudanSwaptionConfig:
        return BermudanSwaptionConfig(**self._common(evaluation_date, trade_id),
                                      exercise_dates=[_parse_ore_date(d) for d in self.exercise_dates],
                                      settlement=self.settlement, fixings=_parse_fixings(self.fixings))


class AmericanTradeSchema(_SwapTerms):
    trade_type: Literal["american_swaption"] = "american_swaption"
    first_exercise_date: str
    last_exercise_date: str
    settlement: Literal["Physical", "Cash"] = "Physical"
    fixings: Dict[str, float] = Field(default_factory=dict)

    def to_dataclass(self, evaluation_date, trade_id) -> AmericanSwaptionConfig:
        return AmericanSwaptionConfig(**self._common(evaluation_date, trade_id),
                                      first_exercise_date=_parse_ore_date(self.first_exercise_date),
                                      last_exercise_date=_parse_ore_date(self.last_exercise_date),
                                      settlement=self.settlement, fixings=_parse_fixings(self.fixings))


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
            coupon_schedule=tuple(CouponPeriod(_parse_ore_date(p.start_date), _parse_ore_date(p.end_date),
                                               _parse_optional_date(p.payment_date)) for p in self.coupon_schedule),
            redemption_fraction=self.redemption_fraction, accrual_day_count=self.accrual_day_count)


MarketTradeSchema = Annotated[
    Union[SwapTradeSchema, EuropeanTradeSchema, BermudanTradeSchema, AmericanTradeSchema, BondTradeSchema],
    Field(discriminator="trade_type"),
]


#: Fields of the Hull-White request shape retired by roadmap 1.3, at the top of the body and in
#: its market; a body carrying one is refused with `RETIRED_SHAPE`. Only fields the current
#: shape does not have: its market's `equities` is the current market's too.
RETIRED_FIELDS = ("evaluation_date", "calibration_basket")
RETIRED_MARKET_FIELDS = ("time_grid", "rates", "joint_covariance")
RETIRED_SHAPE = (
    "the Hull-White request shape (a SimulationConfig market with time_grid/rates/joint_covariance, model "
    "parameters on the trades, calibration_basket) was retired by roadmap 1.3. Send the portfolio request: "
    "today's market (market.asof, currencies), trades naming their currency and index, and the Hull-White "
    "model per currency in simulation.ir, e.g. {\"USD\": {\"model\": \"HullWhite\", \"reversion\": 0.03, "
    "\"volatility\": 0.01}}; Jamshidian Europeans are pricing.european=\"Jamshidian\" with pricing.jamshidian "
    "(docs/reference/http-api.md)")


class MarketPortfolioRequestSchema(_Strict):
    """The portfolio request (see the module docstring). Every trade is valued on
    `market.asof`.
    `simulation` is required with `scenario_risk`. `base_currency` is the reporting currency:
    omitted, the simulation's base currency (USD without a simulation); one contradicting the
    simulation's is refused. `simulation`, `pricing`, `greeks`, `base_currency` and
    `precision` are the run configuration (`engine.portfolio.RunConfig`)."""
    schema_version: Literal["2"] = "2"
    market: MarketSchema
    trades: List[MarketTradeSchema]
    simulation: Optional[CamConfigSchema] = None
    pricing: PricingConfigSchema = Field(default_factory=PricingConfigSchema)
    greeks: GreeksConfigSchema = Field(default_factory=GreeksConfigSchema)
    base_currency: Optional[str] = None
    pfe_quantiles: List[float] = Field(default_factory=lambda: [0.95, 0.99])
    compute_greeks: bool = False
    scenario_risk: bool = True
    precision: PrecisionSchema = Field(default_factory=PrecisionSchema)

    @model_validator(mode="before")
    @classmethod
    def _refuse_the_retired_hull_white_shape(cls, data):
        if isinstance(data, dict):
            market = data.get("market")
            if any(f in data for f in RETIRED_FIELDS) or (
                    isinstance(market, dict) and any(f in market for f in RETIRED_MARKET_FIELDS)):
                raise ValueError(RETIRED_SHAPE)
        return data

    def to_dataclass(self) -> PortfolioRequest:
        market = self.market.to_dataclass()
        ids = [t.trade_id for t in self.trades]
        if any(i is not None for i in ids) and None in ids:
            raise ValueError("give trade_id on every trade or on none")
        if not ids or ids[0] is None:
            ids = [f"trade-{i}" for i in range(len(self.trades))]
        return PortfolioRequest(
            market=market, trades=[t.to_dataclass(market.asof, i) for t, i in zip(self.trades, ids)],
            config=RunConfig(simulation=self.simulation.to_dataclass() if self.simulation else None,
                             pricing=self.pricing.to_dataclass(), greeks=self.greeks.to_dataclass(),
                             base_currency=self.base_currency,
                             precision=self.precision.to_dataclass()),
            pfe_quantiles=tuple(self.pfe_quantiles), compute_greeks=self.compute_greeks,
            scenario_risk=self.scenario_risk)


__all__: Tuple[str, ...] = ("MarketPortfolioRequestSchema",)
