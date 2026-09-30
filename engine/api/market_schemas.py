"""
The portfolio request for the market path (`engine.portfolio.market_path`), the default
model.

Today's market (curves per currency and index, swaption volatilities, FX and equity spots),
ORE's simulation configuration and pricing engines, and trades that name their currency and
index but carry no model or curve of their own (audit A-3). `POST /v2/portfolio/price`
takes it; the result is `PortfolioResultSchema`, shared with the Hull-White request.

The Hull-White model's request (`engine.api.schemas.PortfolioRequestSchema`,
`POST /portfolio/price`) is the other request shape: trade-level model parameters. The two
shapes are models, not versions. The `/v2` in its route and its `schema_version: "2"` are historical names, not versions (docs/reference/http-api.md); they are to become one configurable
request (compliance/decisions.md A-2).

Conventions shared with the Hull-White request: ISO dates, ORE periods ("5Y"), fixings
`{"YYYY-MM-DD": rate}`. Unknown fields are refused (422), so a Hull-White-shaped trade carrying
`hw_sigma` or a curve is not silently stripped of its model (audit A-3).
"""
from typing import Annotated, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

from engine.api.schemas import (
    CouponPeriodSchema, PrecisionConfigSchema, ZeroCurveConfigSchema, _parse_fixings, _parse_optional_date,
    _parse_ore_date, _parse_ore_period,
)
from engine.calibration.ore_lgm import SwapIndexConventions
from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig
from engine.instruments.treasury import BondConfig, CouponPeriod
from engine.market import CurrencyMarket, EquityMarket, Market, SwaptionVolSurface
from engine.portfolio import PortfolioRequest, PrecisionConfig
from engine.simulation.config import CamConfig, LgmConfig
from engine.valuation.config import LgmSwaptionEngineConfig, PricingConfig


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
    """One currency's CAM LGM: constant reversion, and a volatility bootstrapped to the
    `calibration_expiries` x `calibration_terms` basket when one is given."""
    reversion: float
    volatility: float = 0.01
    calibration_expiries: List[str] = Field(default_factory=list)
    calibration_terms: List[str] = Field(default_factory=list)
    swap_index: SwapIndexConventionsSchema = Field(default_factory=SwapIndexConventionsSchema)

    def to_dataclass(self) -> LgmConfig:
        return LgmConfig(self.reversion, self.volatility, tuple(self.calibration_expiries),
                         tuple(self.calibration_terms), self.swap_index.to_dataclass())


class CorrelationSchema(_Strict):
    """One instantaneous correlation between two factors (`IR:USD`, `FX:EURUSD`, `EQ:SP5`)."""
    factor1: str
    factor2: str
    value: float


class CamConfigSchema(_Strict):
    """`engine.simulation.config.CamConfig`."""
    dates: List[str]
    base_currency: str
    ir: Dict[str, LgmConfigSchema]
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


class PricingConfigSchema(_Strict):
    bermudan: LgmEngineSchema = Field(default_factory=LgmEngineSchema)
    american: LgmEngineSchema = Field(default_factory=LgmEngineSchema)
    recalibrate: bool = True

    def to_dataclass(self) -> PricingConfig:
        return PricingConfig(self.bermudan.to_dataclass(), self.american.to_dataclass(), self.recalibrate)


class _Trade(_Strict):
    #: The caller's id for the trade, echoed as `trade_ids` on the result (I-10). Give one on
    #: every trade or on none.
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

    def _common(self, evaluation_date):
        return dict(notional=self.notional, fixed_rate=self.fixed_rate, payer=self.payer, currency=self.currency,
                    effective_date=_parse_optional_date(self.effective_date),
                    maturity_date=_parse_optional_date(self.maturity_date), swap_tenor=self.swap_tenor,
                    index_tenor_months=self.index_tenor_months, floating_spread=self.floating_spread,
                    evaluation_date=evaluation_date)


class SwapTradeSchema(_SwapTerms):
    trade_type: Literal["swap"] = "swap"
    accrual_day_count: str = "ACT/365"
    fixings: Dict[str, float] = Field(default_factory=dict)

    def to_dataclass(self, evaluation_date) -> SwapConfig:
        return SwapConfig(**self._common(evaluation_date), accrual_day_count=self.accrual_day_count,
                          fixings=_parse_fixings(self.fixings))


class EuropeanTradeSchema(_SwapTerms):
    trade_type: Literal["european_swaption"] = "european_swaption"
    exercise_date: Optional[str] = None
    forward_start: Optional[str] = None
    exercise_lag_days: Optional[int] = None
    settlement: Literal["Physical", "Cash"] = "Physical"

    def to_dataclass(self, evaluation_date) -> SwaptionConfig:
        return SwaptionConfig(**self._common(evaluation_date), exercise_date=_parse_optional_date(self.exercise_date),
                              forward_start=_parse_ore_period(self.forward_start) if self.forward_start else None,
                              exercise_lag_days=self.exercise_lag_days, settlement=self.settlement)


class BermudanTradeSchema(_SwapTerms):
    trade_type: Literal["bermudan_swaption"] = "bermudan_swaption"
    exercise_dates: List[str]
    settlement: Literal["Physical", "Cash"] = "Physical"
    fixings: Dict[str, float] = Field(default_factory=dict)

    def to_dataclass(self, evaluation_date) -> BermudanSwaptionConfig:
        return BermudanSwaptionConfig(**self._common(evaluation_date),
                                      exercise_dates=[_parse_ore_date(d) for d in self.exercise_dates],
                                      settlement=self.settlement, fixings=_parse_fixings(self.fixings))


class AmericanTradeSchema(_SwapTerms):
    trade_type: Literal["american_swaption"] = "american_swaption"
    first_exercise_date: str
    last_exercise_date: str
    settlement: Literal["Physical", "Cash"] = "Physical"
    fixings: Dict[str, float] = Field(default_factory=dict)

    def to_dataclass(self, evaluation_date) -> AmericanSwaptionConfig:
        return AmericanSwaptionConfig(**self._common(evaluation_date),
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

    def to_dataclass(self, evaluation_date) -> BondConfig:
        return BondConfig(
            face_amount=self.face_amount, maturity_date=_parse_ore_date(self.maturity_date),
            evaluation_date=evaluation_date, currency=self.currency, coupon_rate=self.coupon_rate,
            coupon_schedule=tuple(CouponPeriod(_parse_ore_date(p.start_date), _parse_ore_date(p.end_date),
                                               _parse_optional_date(p.payment_date)) for p in self.coupon_schedule),
            redemption_fraction=self.redemption_fraction, accrual_day_count=self.accrual_day_count)


MarketTradeSchema = Annotated[
    Union[SwapTradeSchema, EuropeanTradeSchema, BermudanTradeSchema, AmericanTradeSchema, BondTradeSchema],
    Field(discriminator="trade_type"),
]


class MarketPortfolioRequestSchema(_Strict):
    """The market path's request (see the module docstring). Every trade is valued on
    `market.asof`.
    `simulation` is required with `scenario_risk`; without it `base_currency` sets the
    reporting currency."""
    schema_version: Literal["2"] = "2"
    market: MarketSchema
    trades: List[MarketTradeSchema]
    simulation: Optional[CamConfigSchema] = None
    pricing: PricingConfigSchema = Field(default_factory=PricingConfigSchema)
    base_currency: str = "USD"
    pfe_quantiles: List[float] = Field(default_factory=lambda: [0.95, 0.99])
    compute_greeks: bool = False
    scenario_risk: bool = True
    precision: Optional[PrecisionConfigSchema] = None

    def to_dataclass(self) -> PortfolioRequest:
        market = self.market.to_dataclass()
        ids = [t.trade_id for t in self.trades]
        if any(i is not None for i in ids) and None in ids:
            raise ValueError("give trade_id on every trade or on none")
        return PortfolioRequest(
            trade_ids=ids if ids and ids[0] is not None else None,
            market=market, trades=[t.to_dataclass(market.asof) for t in self.trades],
            simulation=self.simulation.to_dataclass() if self.simulation else None,
            pricing=self.pricing.to_dataclass(), base_currency=self.base_currency,
            pfe_quantiles=tuple(self.pfe_quantiles), compute_greeks=self.compute_greeks,
            scenario_risk=self.scenario_risk,
            precision=self.precision.to_dataclass() if self.precision else PrecisionConfig())


__all__: Tuple[str, ...] = ("MarketPortfolioRequestSchema",)
