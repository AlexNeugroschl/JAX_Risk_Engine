"""
Pydantic v2 request/response schemas -- the HTTP boundary's own copy of
`engine.portfolio`/`engine/instruments/*.py`'s dataclasses, field-for-field.

**Wrap, not replace** (see docs/concepts/architecture.md's "Typed
configuration" section): the dataclasses stay the single source of truth
for the engine's own internal shape. Every schema here has a
`.to_dataclass()` method converting to the real engine dataclass, and every
result schema has a `.from_dataclass()` classmethod for the reverse
direction. Pydantic exists ONLY in this HTTP boundary layer -- nothing under
`engine/portfolio/` or below imports Pydantic or FastAPI, preserving the
existing principle that core simulation/risk functionality shouldn't require
the heavy optional `api` dependency extra.

`ORE.Date`/`ORE.Period` fields (SWIG-bound types, not natively
Pydantic-serializable) are represented here as plain strings: dates as ISO
`YYYY-MM-DD` (parsed via `ORE.DateParser.parseISO`), tenors/periods as ORE's
own period-string syntax (e.g. `"5Y"`, `"18M"`, `"0D"`), parsed via
`ORE.Period(str)` -- the exact same parse `engine.portfolio.validation.
_validate_tenor` already validates for the underlying dataclasses.
"""
from typing import Annotated, Dict, List, Literal, Optional, Sequence, Union

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
from engine.portfolio import (
    PortfolioRequest, PortfolioResult, PrecisionConfig, PricingPrecisionOverride, RiskPrecisionOverride,
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

    def to_dataclass(self) -> SimulationConfig:
        return SimulationConfig(
            time_grid=self.time_grid, equities=self.equities.to_dataclass(),
            rates=self.rates.to_dataclass(), joint_covariance=self.joint_covariance,
            scenarios=self.scenarios,
        )


class SwapConfigSchema(BaseModel):
    trade_type: Literal["swap"] = "swap"
    notional: float
    fixed_rate: float
    payer: bool
    discount_curve_index: int
    forward_curve_index: int
    swap_tenor: str = "5Y"
    index_tenor_months: int = 6
    floating_spread: float = 0.0
    evaluation_date: Optional[str] = None

    def to_dataclass(self, default_evaluation_date: ORE.Date) -> SwapConfig:
        return SwapConfig(
            notional=self.notional, fixed_rate=self.fixed_rate, payer=self.payer,
            discount_curve_index=self.discount_curve_index, forward_curve_index=self.forward_curve_index,
            swap_tenor=self.swap_tenor, index_tenor_months=self.index_tenor_months,
            floating_spread=self.floating_spread,
            evaluation_date=_parse_ore_date(self.evaluation_date) if self.evaluation_date else default_evaluation_date,
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
    swap_tenor: str = "5Y"
    index_tenor_months: int = 6
    floating_spread: float = 0.0
    forward_start: str = "0D"
    exercise_lag_days: int = 2
    evaluation_date: Optional[str] = None

    def to_dataclass(self, default_evaluation_date: ORE.Date) -> SwaptionConfig:
        return SwaptionConfig(
            notional=self.notional, fixed_rate=self.fixed_rate, payer=self.payer,
            rate_factor_index=self.rate_factor_index, hw_a=self.hw_a, hw_sigma=self.hw_sigma,
            initial_zero_curve=self.initial_zero_curve.to_dataclass(),
            swap_tenor=self.swap_tenor, index_tenor_months=self.index_tenor_months,
            floating_spread=self.floating_spread, forward_start=_parse_ore_period(self.forward_start),
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
    exercise_times: List[float]
    swap_tenor: str = "5Y"
    index_tenor_months: int = 6
    floating_spread: float = 0.0
    n_per_std: int = 48
    std_devs: float = 6.0
    evaluation_date: Optional[str] = None

    def to_dataclass(self, default_evaluation_date: ORE.Date) -> BermudanSwaptionConfig:
        return BermudanSwaptionConfig(
            notional=self.notional, fixed_rate=self.fixed_rate, payer=self.payer,
            rate_factor_index=self.rate_factor_index, hw_a=self.hw_a, hw_sigma=self.hw_sigma,
            initial_zero_curve=self.initial_zero_curve.to_dataclass(),
            exercise_times=self.exercise_times, swap_tenor=self.swap_tenor,
            index_tenor_months=self.index_tenor_months, floating_spread=self.floating_spread,
            n_per_std=self.n_per_std, std_devs=self.std_devs,
            evaluation_date=_parse_ore_date(self.evaluation_date) if self.evaluation_date else default_evaluation_date,
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
    first_exercise: float
    last_exercise: float
    swap_tenor: str = "5Y"
    index_tenor_months: int = 6
    floating_spread: float = 0.0
    exercise_time_steps_per_year: int = 24
    n_per_std: int = 48
    std_devs: float = 6.0
    evaluation_date: Optional[str] = None

    def to_dataclass(self, default_evaluation_date: ORE.Date) -> AmericanSwaptionConfig:
        return AmericanSwaptionConfig(
            notional=self.notional, fixed_rate=self.fixed_rate, payer=self.payer,
            rate_factor_index=self.rate_factor_index, hw_a=self.hw_a, hw_sigma=self.hw_sigma,
            initial_zero_curve=self.initial_zero_curve.to_dataclass(),
            first_exercise=self.first_exercise, last_exercise=self.last_exercise,
            swap_tenor=self.swap_tenor, index_tenor_months=self.index_tenor_months,
            floating_spread=self.floating_spread, exercise_time_steps_per_year=self.exercise_time_steps_per_year,
            n_per_std=self.n_per_std, std_devs=self.std_devs,
            evaluation_date=_parse_ore_date(self.evaluation_date) if self.evaluation_date else default_evaluation_date,
        )


class CouponPeriodSchema(BaseModel):
    """One explicit coupon period of a `BondConfigSchema`.

    Enumerated rather than generated from a frequency, matching
    `engine.integration.note`'s rule: a schedule this engine derived by
    stepping back from maturity that disagreed with the booked one would
    silently reprice every coupon.
    """
    start_date: str
    end_date: str
    #: Defaults to `end_date` when absent -- an unadjusted schedule has
    #: them equal. A present-but-unparseable value still raises.
    payment_date: Optional[str] = None


class BondConfigSchema(BaseModel):
    """A Treasury bill or note (W1.5).

    **A bill is simply `coupon_schedule` omitted with `coupon_rate` left at
    0.0** -- the degenerate case of the same type, not a separate one.

    `face_amount` is **signed**: a short position is a negative face. There
    is no separate sign field, and adding one would risk the double-sign
    bug TraderX flagged in their v3 §2.

    **A bond carries its own `initial_zero_curve`** rather than an index
    into the simulation's curves, like the swaption family and unlike
    `SwapConfig`. See `engine.instruments.treasury`: that is what makes
    I-01's silent-skip class unreachable for this type.

    ⚠ **A portfolio containing a bond must set `scenario_risk: false`.** A
    bond has no scenario NPV, so it cannot appear in `npv_cube` and has no
    VaR/ES. Submitting one with `scenario_risk: true` (the default) is
    **refused** with an explicit message rather than served a fabricated
    zero -- see `PortfolioRequestSchema.scenario_risk` and I-24.
    """
    trade_type: Literal["bond"] = "bond"
    face_amount: float
    maturity_date: str
    initial_zero_curve: ZeroCurveConfigSchema
    #: Annual coupon rate as a DECIMAL (0.04 == 4%), not a percent --
    #: the same unit `engine.instruments.treasury.BondConfig` states.
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
    """Mirrors `engine.calibration.basket.build_coterminal_basket`'s own
    inputs (minus `zero_curve`/`evaluation_date`/`index_tenor_months`,
    which `PortfolioRequestSchema.to_dataclass()` fills in from whichever
    Bermudan/American trade's own `initial_zero_curve`/`evaluation_date`
    actually needs the resulting basket -- see that method for why).
    Supplying this on a `PortfolioRequestSchema` is what makes
    `hw_sigma: null` on a Bermudan/American trade actually work end-to-end
    over HTTP: without it, `price_portfolio` raises `"calibration_targets
    was not supplied"` (mirroring `engine.portfolio.PortfolioRequest.
    calibration_targets`'s own dataclass-level requirement)."""
    exercise_times: List[float]
    final_maturity_time: float
    notional: float
    payer: bool
    market_vols: List[float]


class PricingPrecisionOverrideSchema(BaseModel):
    """Mirrors `engine.portfolio.PricingPrecisionOverride` field-for-field --
    optional per-instrument-type drill-down for `PrecisionConfigSchema.
    pricing`, validated by the dataclass's own `__post_init__` once
    `.to_dataclass()` constructs it."""
    default: int = 64
    swap: Optional[int] = None
    european_swaption: Optional[int] = None
    bermudan_swaption: Optional[int] = None
    american_swaption: Optional[int] = None

    def to_dataclass(self) -> PricingPrecisionOverride:
        return PricingPrecisionOverride(**self.model_dump())


class RiskPrecisionOverrideSchema(BaseModel):
    """Mirrors `engine.portfolio.RiskPrecisionOverride` field-for-field --
    optional per-Greek/per-metric drill-down for `PrecisionConfigSchema.
    risk`, validated by the dataclass's own `__post_init__` once
    `.to_dataclass()` constructs it."""
    default: int = 64
    delta_gamma: Optional[int] = None
    theta: Optional[int] = None
    vega: Optional[int] = None
    var_es: Optional[int] = None

    def to_dataclass(self) -> RiskPrecisionOverride:
        return RiskPrecisionOverride(**self.model_dump())


class PrecisionConfigSchema(BaseModel):
    """Mirrors `engine.portfolio.PrecisionConfig` field-for-field --
    independent simulation/pricing/risk/calibration dtype control (32 or
    64), each validated by `PrecisionConfig.__post_init__` itself once
    `.to_dataclass()` constructs it (no duplicate Pydantic-level validator
    needed here). `pricing`/`risk` each additionally accept a structured
    override object instead of a flat int, for optional per-instrument-type/
    per-Greek drill-down -- Pydantic v2 resolves `Union[int, ...Schema]`
    natively from the request JSON shape, no explicit discriminator needed."""
    simulation: int = 64
    pricing: Union[int, PricingPrecisionOverrideSchema] = 64
    risk: Union[int, RiskPrecisionOverrideSchema] = 64
    calibration: int = 64

    def to_dataclass(self) -> PrecisionConfig:
        pricing = self.pricing if isinstance(self.pricing, int) else self.pricing.to_dataclass()
        risk = self.risk if isinstance(self.risk, int) else self.risk.to_dataclass()
        return PrecisionConfig(simulation=self.simulation, pricing=pricing, risk=risk, calibration=self.calibration)


class PortfolioRequestSchema(BaseModel):
    """Mirrors `engine.portfolio.PortfolioRequest` field-for-field.
    `evaluation_date` is a request-scoped default applied to any trade that
    doesn't specify its own (matching every dataclass's own
    `ORE.Settings.instance().evaluationDate`-defaulting `field`, made
    explicit here since there's no ambient global evaluation date to fall
    back on across HTTP requests).

    `precision` is `Optional`, not a populated default -- makes "no
    `precision` key sent" and "explicit all-64 sent" behave identically
    (both resolve to `PrecisionConfig()`), and is more accurate in the
    generated OpenAPI schema than a default that looks like it was always
    required."""
    evaluation_date: str = Field(..., description="ISO date (YYYY-MM-DD), e.g. '2026-07-30'")
    market: SimulationConfigSchema
    trades: List[TradeSchema]
    percentiles: List[float] = Field(default_factory=lambda: [0.95, 0.99])
    calibration_basket: Optional[CalibrationBasketRequestSchema] = None
    compute_greeks: bool = False
    precision: Optional[PrecisionConfigSchema] = None
    #: Whether to build `npv_cube` and derive VaR/ES. `true` (the default)
    #: is the pre-W1.5 behaviour exactly.
    #:
    #: **Must be `false` for a portfolio containing a bond**, which has no
    #: scenario representation. The response then carries an empty
    #: `npv_cube` and an empty `risk`, with `scenario_risk_available:
    #: false` saying so -- absent rather than a fabricated zero (I-24).
    scenario_risk: bool = True

    def to_dataclass(self) -> PortfolioRequest:
        eval_date = _parse_ore_date(self.evaluation_date)
        trades = [t.to_dataclass(eval_date) for t in self.trades]

        calibration_targets = None
        if self.calibration_basket is not None:
            # build_coterminal_basket needs ONE curve/a to build the
            # basket's own ATM strikes against -- engine.portfolio.request.
            # _fill_calibrated_sigma already assumes a single shared basket
            # applies uniformly per rate_factor_index that needs
            # calibration (see its own docstring), so this mirrors that:
            # the first uncalibrated Bermudan/American trade's own curve/a/
            # evaluation_date is what the basket (and therefore every
            # calibration derived from it) is built against.
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
            percentiles=tuple(self.percentiles), calibration_targets=calibration_targets,
            compute_greeks=self.compute_greeks,
            precision=self.precision.to_dataclass() if self.precision is not None else PrecisionConfig(),
            scenario_risk=self.scenario_risk,
        )


class RiskMetricsSchema(BaseModel):
    values: Dict[str, List[Optional[float]]]

    @classmethod
    def from_dataclass(cls, risk: Dict[str, "np.ndarray"]) -> "RiskMetricsSchema":
        out = {}
        for key, arr in risk.items():
            arr_np = np.asarray(arr)
            out[key] = [None if np.isnan(v) else float(v) for v in arr_np.tolist()]
        return cls(values=out)


class GreeksSchema(BaseModel):
    values: Dict[str, List[float]] = Field(default_factory=dict)
    theta: Optional[float] = None

    @classmethod
    def from_dataclass(cls, greeks: Dict[str, "np.ndarray"]) -> "GreeksSchema":
        values = {}
        theta = None
        for key, val in greeks.items():
            if key == "theta":
                theta = float(val)
            else:
                # A 0-d array's `.tolist()` returns a bare Python float, not
                # a list, so iterating it raises `TypeError: 'float' object
                # is not iterable`. Every rate-derivative Greek is a
                # per-pillar VECTOR, so this never arose before W1.5 -- but
                # a BondConfig's delta/gamma are scalars (one parallel bump
                # against a single curve), and serializing one crashed here.
                # `np.atleast_1d` normalizes the scalar case to a
                # one-element list, keeping `values` uniformly a list-per-
                # Greek rather than sometimes a float. Pinned by
                # `TestBondGreeksSerializeOverHttp`.
                values[key] = [float(v) for v in np.atleast_1d(np.asarray(val)).tolist()]
        return cls(values=values, theta=theta)


class PortfolioResultSchema(BaseModel):
    base_npv: float
    npv_cube: List[List[List[float]]]  # [Scenarios, TimeSteps, Trades]
    risk: RiskMetricsSchema
    greeks: Optional[Dict[int, GreeksSchema]] = None
    warnings: List[str] = Field(default_factory=list)
    # Per-trade t=0 NPV in the request's own `trades` order; `base_npv` is
    # their sum. Lets a caller reconcile the portfolio total against
    # identified positions/contracts instead of only seeing an aggregate.
    base_npv_per_trade: List[float] = Field(default_factory=list)
    # Whether `npv_cube`/`risk` were actually computed. `false` means they
    # are EMPTY because the run was `scenario_risk: false` -- the VaR/ES
    # numbers are absent, not zero. Without this field an empty `risk` is
    # ambiguous between "not requested" and "computed as nothing" (I-24).
    scenario_risk_available: bool = True

    @classmethod
    def from_dataclass(cls, result: PortfolioResult) -> "PortfolioResultSchema":
        return cls(
            base_npv=result.base_npv,
            base_npv_per_trade=[float(v) for v in result.base_npv_per_trade],
            npv_cube=np.asarray(result.npv_cube).tolist(),
            risk=RiskMetricsSchema.from_dataclass(result.risk),
            greeks=(
                {i: GreeksSchema.from_dataclass(g) for i, g in result.greeks.items()}
                if result.greeks is not None else None
            ),
            warnings=list(result.warnings),
            scenario_risk_available=result.scenario_risk_available,
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
    """Inputs to `engine.calibration.basket.build_coterminal_basket` +
    `engine.calibration.lgm.calibrate_lgm_sigma` -- a caller wanting a
    fitted `Sigma` back without submitting a full portfolio request."""
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
