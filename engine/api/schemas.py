"""
Pydantic request/response schemas for the HTTP API, mirroring the engine's dataclasses
field for field. Each schema converts with `.to_dataclass()` / `.from_dataclass()`; the
dataclasses remain the source of truth, and nothing below `engine/api/` imports Pydantic.

ORE types travel as strings: dates as ISO `YYYY-MM-DD` (`ORE.DateParser.parseISO`), periods
in ORE syntax (`"5Y"`, `"18M"`), fixings as `{"YYYY-MM-DD": rate}`.

A trade's schedule is `effective_date`/`maturity_date` (plus a European's
`exercise_date`), or `swap_tenor` (plus a European's `forward_start`/`exercise_lag_days`)
resolved on the trade's evaluation date. There is no default tenor.
"""
import dataclasses
from typing import Annotated, Dict, List, Literal, Optional, Union

import numpy as np
import ORE
from pydantic import BaseModel, Field

from engine.simulation.market_model import (
    EquityConfig, RatesConfig, SimulationConfig, ZeroCurveConfig,
)
from engine.instruments.swap import SwapConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.treasury import BondConfig, CouponPeriod
from engine.calibration.basket import build_coterminal_basket
from engine.models.hull_white import ZeroCurve as _HwZeroCurve
from engine.risk.exposure import ExposureProfile
from engine.portfolio import (
    HULL_WHITE_CONFIG, PortfolioRequest, PortfolioResult, PrecisionConfig, PricingPrecisionOverride,
    RiskPrecisionOverride,
)


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


class ZeroCurveConfigSchema(BaseModel):
    times: List[float]
    rates: List[float]

    def to_dataclass(self) -> ZeroCurveConfig:
        return ZeroCurveConfig(times=self.times, rates=self.rates)

    @classmethod
    def from_dataclass(cls, cfg: ZeroCurveConfig) -> "ZeroCurveConfigSchema":
        return cls(times=list(cfg.times), rates=list(cfg.rates))


class EquityConfigSchema(BaseModel):
    initial_prices: List[float]
    dividend_yields: List[float]
    rate_mapping: List[List[float]]

    def to_dataclass(self) -> EquityConfig:
        return EquityConfig(
            initial_prices=self.initial_prices, dividend_yields=self.dividend_yields,
            rate_mapping=self.rate_mapping,
        )


class RatesConfigSchema(BaseModel):
    initial_rates: List[float]
    theta: List[float]
    mean_reversion: List[float]
    maturities: Optional[List[float]] = None
    initial_zero_curves: Optional[List[ZeroCurveConfigSchema]] = None

    def to_dataclass(self) -> RatesConfig:
        return RatesConfig(
            initial_rates=self.initial_rates, theta=self.theta, mean_reversion=self.mean_reversion,
            maturities=self.maturities,
            initial_zero_curves=(
                [c.to_dataclass() for c in self.initial_zero_curves]
                if self.initial_zero_curves is not None else None
            ),
        )


class SimulationConfigSchema(BaseModel):
    time_grid: List[float]
    equities: EquityConfigSchema
    rates: RatesConfigSchema
    joint_covariance: List[List[float]]
    scenarios: int = 10000
    seed: int = 42

    def to_dataclass(self) -> SimulationConfig:
        return SimulationConfig(
            time_grid=self.time_grid, equities=self.equities.to_dataclass(),
            rates=self.rates.to_dataclass(), joint_covariance=self.joint_covariance,
            scenarios=self.scenarios, seed=self.seed,
        )


class SwapConfigSchema(BaseModel):
    trade_type: Literal["swap"] = "swap"
    notional: float
    fixed_rate: float
    payer: bool
    discount_curve_index: int
    forward_curve_index: int
    effective_date: Optional[str] = None
    maturity_date: Optional[str] = None
    swap_tenor: Optional[str] = None
    index_tenor_months: int = 6
    floating_spread: float = 0.0
    evaluation_date: Optional[str] = None
    fixings: Dict[str, float] = Field(default_factory=dict)

    def to_dataclass(self, default_evaluation_date: ORE.Date) -> SwapConfig:
        return SwapConfig(
            notional=self.notional, fixed_rate=self.fixed_rate, payer=self.payer,
            discount_curve_index=self.discount_curve_index, forward_curve_index=self.forward_curve_index,
            effective_date=_parse_optional_date(self.effective_date),
            maturity_date=_parse_optional_date(self.maturity_date),
            swap_tenor=self.swap_tenor, index_tenor_months=self.index_tenor_months,
            floating_spread=self.floating_spread,
            evaluation_date=_parse_ore_date(self.evaluation_date) if self.evaluation_date else default_evaluation_date,
            fixings=_parse_fixings(self.fixings),
        )


class SwaptionConfigSchema(BaseModel):
    trade_type: Literal["european_swaption"] = "european_swaption"
    notional: float
    fixed_rate: float
    payer: bool
    rate_factor_index: int
    hw_a: float
    hw_sigma: float
    initial_zero_curve: ZeroCurveConfigSchema
    exercise_date: Optional[str] = None
    effective_date: Optional[str] = None
    maturity_date: Optional[str] = None
    swap_tenor: Optional[str] = None
    index_tenor_months: int = 6
    floating_spread: float = 0.0
    forward_start: Optional[str] = None
    exercise_lag_days: Optional[int] = None
    evaluation_date: Optional[str] = None

    def to_dataclass(self, default_evaluation_date: ORE.Date) -> SwaptionConfig:
        return SwaptionConfig(
            notional=self.notional, fixed_rate=self.fixed_rate, payer=self.payer,
            rate_factor_index=self.rate_factor_index, hw_a=self.hw_a, hw_sigma=self.hw_sigma,
            initial_zero_curve=self.initial_zero_curve.to_dataclass(),
            exercise_date=_parse_optional_date(self.exercise_date),
            effective_date=_parse_optional_date(self.effective_date),
            maturity_date=_parse_optional_date(self.maturity_date),
            swap_tenor=self.swap_tenor, index_tenor_months=self.index_tenor_months,
            floating_spread=self.floating_spread,
            forward_start=_parse_ore_period(self.forward_start) if self.forward_start is not None else None,
            exercise_lag_days=self.exercise_lag_days,
            evaluation_date=_parse_ore_date(self.evaluation_date) if self.evaluation_date else default_evaluation_date,
        )


class BermudanSwaptionConfigSchema(BaseModel):
    trade_type: Literal["bermudan_swaption"] = "bermudan_swaption"
    notional: float
    fixed_rate: float
    payer: bool
    rate_factor_index: int
    hw_a: float
    hw_sigma: Optional[float] = None  # None -> uncalibrated, filled in by price_portfolio
    initial_zero_curve: ZeroCurveConfigSchema
    exercise_dates: List[str]  # ISO dates, ascending -- ORE's exercise contract
    effective_date: Optional[str] = None
    maturity_date: Optional[str] = None
    swap_tenor: Optional[str] = None
    index_tenor_months: int = 6
    floating_spread: float = 0.0
    n_per_std: int = 48
    std_devs: float = 6.0
    evaluation_date: Optional[str] = None
    fixings: Dict[str, float] = Field(default_factory=dict)

    def to_dataclass(self, default_evaluation_date: ORE.Date) -> BermudanSwaptionConfig:
        return BermudanSwaptionConfig(
            notional=self.notional, fixed_rate=self.fixed_rate, payer=self.payer,
            rate_factor_index=self.rate_factor_index, hw_a=self.hw_a, hw_sigma=self.hw_sigma,
            initial_zero_curve=self.initial_zero_curve.to_dataclass(),
            exercise_dates=[_parse_ore_date(d) for d in self.exercise_dates],
            effective_date=_parse_optional_date(self.effective_date),
            maturity_date=_parse_optional_date(self.maturity_date), swap_tenor=self.swap_tenor,
            index_tenor_months=self.index_tenor_months, floating_spread=self.floating_spread,
            n_per_std=self.n_per_std, std_devs=self.std_devs,
            evaluation_date=_parse_ore_date(self.evaluation_date) if self.evaluation_date else default_evaluation_date,
            fixings=_parse_fixings(self.fixings),
        )


class AmericanSwaptionConfigSchema(BaseModel):
    trade_type: Literal["american_swaption"] = "american_swaption"
    notional: float
    fixed_rate: float
    payer: bool
    rate_factor_index: int
    hw_a: float
    hw_sigma: Optional[float] = None
    initial_zero_curve: ZeroCurveConfigSchema
    first_exercise_date: str  # ISO date: first day of the exercise window
    last_exercise_date: str   # ISO date: last day of the exercise window
    effective_date: Optional[str] = None
    maturity_date: Optional[str] = None
    swap_tenor: Optional[str] = None
    index_tenor_months: int = 6
    floating_spread: float = 0.0
    exercise_time_steps_per_year: int = 24
    n_per_std: int = 48
    std_devs: float = 6.0
    evaluation_date: Optional[str] = None
    fixings: Dict[str, float] = Field(default_factory=dict)

    def to_dataclass(self, default_evaluation_date: ORE.Date) -> AmericanSwaptionConfig:
        return AmericanSwaptionConfig(
            notional=self.notional, fixed_rate=self.fixed_rate, payer=self.payer,
            rate_factor_index=self.rate_factor_index, hw_a=self.hw_a, hw_sigma=self.hw_sigma,
            initial_zero_curve=self.initial_zero_curve.to_dataclass(),
            first_exercise_date=_parse_ore_date(self.first_exercise_date),
            last_exercise_date=_parse_ore_date(self.last_exercise_date),
            effective_date=_parse_optional_date(self.effective_date),
            maturity_date=_parse_optional_date(self.maturity_date),
            swap_tenor=self.swap_tenor, index_tenor_months=self.index_tenor_months,
            floating_spread=self.floating_spread, exercise_time_steps_per_year=self.exercise_time_steps_per_year,
            n_per_std=self.n_per_std, std_devs=self.std_devs,
            evaluation_date=_parse_ore_date(self.evaluation_date) if self.evaluation_date else default_evaluation_date,
            fixings=_parse_fixings(self.fixings),
        )


class CouponPeriodSchema(BaseModel):
    """One explicit coupon period of a `BondConfigSchema`; enumerated, not generated (see
    `engine.instruments.treasury.CouponPeriod`)."""
    start_date: str
    end_date: str
    #: Defaults to `end_date`.
    payment_date: Optional[str] = None


class BondConfigSchema(BaseModel):
    """A Treasury bill or note (`engine.instruments.treasury.BondConfig`).

    A bill omits `coupon_schedule` and leaves `coupon_rate` at 0. `face_amount` is signed
    (negative for a short); there is no separate sign field. The bond carries its own
    `initial_zero_curve`.

    A portfolio containing a bond must set `scenario_risk: false`: a bond has no scenario
    NPV, and the request is refused otherwise (I-24).
    """
    trade_type: Literal["bond"] = "bond"
    face_amount: float
    maturity_date: str
    initial_zero_curve: ZeroCurveConfigSchema
    #: Annual coupon rate as a decimal (0.04 == 4%).
    coupon_rate: float = 0.0
    coupon_schedule: List[CouponPeriodSchema] = Field(default_factory=list)
    redemption_fraction: float = 1.0
    accrual_day_count: str = "ACT/ACT (ICMA)"
    evaluation_date: Optional[str] = None

    def to_dataclass(self, default_evaluation_date: ORE.Date) -> BondConfig:
        return BondConfig(
            face_amount=self.face_amount,
            maturity_date=_parse_ore_date(self.maturity_date),
            evaluation_date=(
                _parse_ore_date(self.evaluation_date) if self.evaluation_date
                else default_evaluation_date
            ),
            initial_zero_curve=self.initial_zero_curve.to_dataclass(),
            coupon_rate=self.coupon_rate,
            coupon_schedule=tuple(
                CouponPeriod(
                    start_date=_parse_ore_date(p.start_date),
                    end_date=_parse_ore_date(p.end_date),
                    payment_date=_parse_ore_date(p.payment_date) if p.payment_date else None,
                )
                for p in self.coupon_schedule
            ),
            redemption_fraction=self.redemption_fraction,
            accrual_day_count=self.accrual_day_count,
        )


TradeSchema = Annotated[
    Union[
        SwapConfigSchema, SwaptionConfigSchema, BermudanSwaptionConfigSchema,
        AmericanSwaptionConfigSchema, BondConfigSchema,
    ],
    Field(discriminator="trade_type"),
]


class CalibrationBasketRequestSchema(BaseModel):
    """Inputs of `build_coterminal_basket`, except the curve, evaluation date and index
    tenor, which `PortfolioRequestSchema.to_dataclass()` takes from the first uncalibrated
    Bermudan/American trade. Required when any such trade has `hw_sigma: null`."""
    exercise_times: List[float]
    final_maturity_time: float
    notional: float
    payer: bool
    market_vols: List[float]


class PricingPrecisionOverrideSchema(BaseModel):
    """`engine.portfolio.PricingPrecisionOverride`; validated by the dataclass."""
    default: int = 64
    swap: Optional[int] = None
    european_swaption: Optional[int] = None
    bermudan_swaption: Optional[int] = None
    american_swaption: Optional[int] = None

    def to_dataclass(self) -> PricingPrecisionOverride:
        return PricingPrecisionOverride(**self.model_dump())


class RiskPrecisionOverrideSchema(BaseModel):
    """`engine.portfolio.RiskPrecisionOverride`; validated by the dataclass."""
    default: int = 64
    delta_gamma: Optional[int] = None
    theta: Optional[int] = None
    vega: Optional[int] = None
    exposure: Optional[int] = None

    def to_dataclass(self) -> RiskPrecisionOverride:
        return RiskPrecisionOverride(**self.model_dump())


class PrecisionConfigSchema(BaseModel):
    """`engine.portfolio.PrecisionConfig`; validated by the dataclass. `pricing`/`risk`
    take an int or an override object."""
    simulation: int = 64
    pricing: Union[int, PricingPrecisionOverrideSchema] = 64
    risk: Union[int, RiskPrecisionOverrideSchema] = 64
    calibration: int = 64

    def to_dataclass(self) -> PrecisionConfig:
        pricing = self.pricing if isinstance(self.pricing, int) else self.pricing.to_dataclass()
        risk = self.risk if isinstance(self.risk, int) else self.risk.to_dataclass()
        return PrecisionConfig(simulation=self.simulation, pricing=pricing, risk=risk, calibration=self.calibration)


class PortfolioRequestSchema(BaseModel):
    """`engine.portfolio.PortfolioRequest` on the Hull-White model. `evaluation_date` is the
    default for trades that do not give their own (there is no ambient ORE evaluation date
    across requests). The run configuration is `HULL_WHITE_CONFIG` (the engines this model
    implements) with `precision`; `precision` omitted means `PrecisionConfig()`."""
    evaluation_date: str = Field(..., description="ISO date (YYYY-MM-DD), e.g. '2026-07-30'")
    market: SimulationConfigSchema
    trades: List[TradeSchema]
    pfe_quantiles: List[float] = Field(default_factory=lambda: [0.95, 0.99])
    calibration_basket: Optional[CalibrationBasketRequestSchema] = None
    compute_greeks: bool = False
    precision: Optional[PrecisionConfigSchema] = None
    #: Whether to build `npv_cube` and the exposure profiles. Must be false for a portfolio
    #: containing a bond; the response then has an empty `npv_cube`, no exposure, and
    #: `scenario_risk_available: false` (I-24).
    scenario_risk: bool = True

    def to_dataclass(self) -> PortfolioRequest:
        eval_date = _parse_ore_date(self.evaluation_date)
        trades = [t.to_dataclass(eval_date) for t in self.trades]

        calibration_targets = None
        if self.calibration_basket is not None:
            # The basket is built on the first uncalibrated Bermudan/American's curve and
            # evaluation date, and serves every rate factor that needs calibration (as
            # engine.portfolio.request._fill_calibrated_sigma assumes).
            first_uncalibrated = next(
                (t for t in trades if isinstance(t, (BermudanSwaptionConfig, AmericanSwaptionConfig)) and t.hw_sigma is None),
                None,
            )
            if first_uncalibrated is None:
                raise ValueError(
                    "calibration_basket was supplied but no trade has hw_sigma=null "
                    "(uncalibrated) to calibrate it for"
                )
            curve_jax = _HwZeroCurve.from_config(first_uncalibrated.initial_zero_curve)
            calibration_targets = build_coterminal_basket(
                exercise_times=self.calibration_basket.exercise_times,
                final_maturity_time=self.calibration_basket.final_maturity_time,
                notional=self.calibration_basket.notional,
                payer=self.calibration_basket.payer,
                market_vols=self.calibration_basket.market_vols,
                zero_curve=curve_jax,
                evaluation_date=first_uncalibrated.evaluation_date,
                index_tenor_months=first_uncalibrated.index_tenor_months,
            )

        return PortfolioRequest(
            market=self.market.to_dataclass(), trades=trades,
            pfe_quantiles=tuple(self.pfe_quantiles), calibration_targets=calibration_targets,
            compute_greeks=self.compute_greeks,
            config=dataclasses.replace(
                HULL_WHITE_CONFIG,
                precision=self.precision.to_dataclass() if self.precision is not None else PrecisionConfig()),
            scenario_risk=self.scenario_risk,
        )


class RiskMetricsSchema(BaseModel):
    """A `compute_risk_metrics` dict on the wire: key -> values, NaN as null."""
    values: Dict[str, List[Optional[float]]]

    @classmethod
    def from_dataclass(cls, risk: Dict[str, "np.ndarray"]) -> "RiskMetricsSchema":
        out = {}
        for key, arr in risk.items():
            arr_np = np.atleast_1d(np.asarray(arr))
            out[key] = [None if np.isnan(v) else float(v) for v in arr_np.tolist()]
        return cls(values=out)


class ExposureProfileSchema(BaseModel):
    """`engine.risk.exposure.ExposureProfile`; every list is indexed like `times`, starting
    at t=0."""
    times: List[float]
    epe: List[float]
    ene: List[float]
    ee_b: List[float]
    eee_b: List[float]
    pfe: Dict[str, List[float]]
    #: ORE's time-weighted EPE_B / EEPE_B profiles.
    epe_b: Optional[List[float]] = None
    eepe_b: Optional[List[float]] = None
    #: ORE's Basel EPE_B / EEPE_B at the one-year horizon; null on the Hull-White path.
    basel_epe: Optional[float] = None
    basel_eepe: Optional[float] = None

    @classmethod
    def from_dataclass(cls, profile: ExposureProfile) -> "ExposureProfileSchema":
        as_list = lambda values: [float(v) for v in np.asarray(values).tolist()]  # noqa: E731
        optional = lambda values: None if values is None else as_list(values)  # noqa: E731
        return cls(
            times=as_list(profile.times), epe=as_list(profile.epe), ene=as_list(profile.ene),
            ee_b=as_list(profile.ee_b), eee_b=as_list(profile.eee_b),
            pfe={key: as_list(values) for key, values in profile.pfe.items()},
            epe_b=optional(profile.epe_b), eepe_b=optional(profile.eepe_b),
            basel_epe=profile.basel_epe, basel_eepe=profile.basel_eepe,
        )


class GreeksSchema(BaseModel):
    """Each Greek flattened row-major into `values`; a Greek of more than one dimension (the
    market path's `vega:<ccy>`, option tenors x swap tenors) also has its shape in `shapes`."""
    values: Dict[str, List[float]] = Field(default_factory=dict)
    shapes: Dict[str, List[int]] = Field(default_factory=dict)
    theta: Optional[float] = None

    @classmethod
    def from_dataclass(cls, greeks: Dict[str, "np.ndarray"]) -> "GreeksSchema":
        values, shapes = {}, {}
        theta = None
        for key, val in greeks.items():
            if key == "theta":
                theta = float(val)
                continue
            array = np.asarray(val)
            # ravel: bond Greeks are scalars, and a 0-d array's .tolist() is a float, not a
            # list (I-25).
            values[key] = [float(v) for v in array.ravel().tolist()]
            if array.ndim > 1:
                shapes[key] = list(array.shape)
        return cls(values=values, shapes=shapes, theta=theta)


class PortfolioResultSchema(BaseModel):
    base_npv: float
    npv_cube: List[List[List[float]]]  # [Scenarios, TimeSteps, Trades]
    #: The whole portfolio as one netting set; null without scenario risk.
    exposure: Optional[ExposureProfileSchema] = None
    #: Standalone exposure per trade, in request order.
    trade_exposures: List[ExposureProfileSchema] = Field(default_factory=list)
    greeks: Optional[Dict[int, GreeksSchema]] = None
    warnings: List[str] = Field(default_factory=list)
    # Per-trade t=0 NPV in request order; `base_npv` is their sum.
    base_npv_per_trade: List[float] = Field(default_factory=list)
    # Whether `npv_cube`/`exposure` were computed; false means absent, not zero (I-24).
    scenario_risk_available: bool = True
    # Measure of the exposure (`risk-neutral-pricing`), or null (I-11).
    measure: Optional[str] = None
    # The request's trade ids in request order, or null if it gave none (I-10).
    trade_ids: Optional[List[str]] = None

    @classmethod
    def from_dataclass(cls, result: PortfolioResult) -> "PortfolioResultSchema":
        return cls(
            base_npv=result.base_npv,
            base_npv_per_trade=[float(v) for v in result.base_npv_per_trade],
            npv_cube=np.asarray(result.npv_cube).tolist(),
            exposure=(
                ExposureProfileSchema.from_dataclass(result.exposure)
                if result.exposure is not None else None
            ),
            trade_exposures=[ExposureProfileSchema.from_dataclass(p) for p in result.trade_exposures],
            greeks=(
                {i: GreeksSchema.from_dataclass(g) for i, g in result.greeks.items()}
                if result.greeks is not None else None
            ),
            warnings=list(result.warnings),
            scenario_risk_available=result.scenario_risk_available,
            measure=result.measure,
            trade_ids=result.trade_ids,
        )


class JobStatusSchema(BaseModel):
    status: Literal["pending", "running", "done", "failed"]
    result: Optional[PortfolioResultSchema] = None
    error: Optional[str] = None


class HealthSchema(BaseModel):
    status: Literal["ok"] = "ok"


class VersionSchema(BaseModel):
    engine_version: str
    jax_backend: str
    git_commit: Optional[str] = None


class CalibrationRequestSchema(BaseModel):
    """Inputs to `build_coterminal_basket` + `calibrate_lgm_sigma`, for a caller that wants
    a fitted `Sigma` without a full portfolio request."""
    evaluation_date: str
    exercise_times: List[float]
    final_maturity_time: float
    notional: float
    payer: bool
    market_vols: List[float]
    zero_curve: ZeroCurveConfigSchema
    hw_a: float
    index_tenor_months: int = 6


class CalibrationResultSchema(BaseModel):
    sigma_times: List[float]
    sigma_values: List[float]
    market_prices: List[float]
    model_prices: List[float]
    rmse: float
