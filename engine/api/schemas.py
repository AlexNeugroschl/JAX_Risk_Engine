"""
Pydantic schemas shared by the HTTP API: the wire forms of curves, coupon periods, precision,
results, job status, health and version, and the standalone calibration route's request.
The portfolio request itself is `engine.api.market_schemas`. Each schema converts with
`.to_dataclass()` / `.from_dataclass()`; the dataclasses remain the source of truth, and
nothing below `engine/api/` imports Pydantic.

ORE types travel as strings: dates as ISO `YYYY-MM-DD` (`ORE.DateParser.parseISO`), periods
in ORE syntax (`"5Y"`, `"18M"`), fixings as `{"YYYY-MM-DD": rate}`.
"""
from typing import Dict, List, Literal, Optional

import numpy as np
import ORE
from pydantic import BaseModel, ConfigDict, Field, model_validator

from engine.market import ZeroCurveConfig
from engine.risk.exposure import ExposureProfile
from engine.portfolio import PortfolioResult
from engine.precision import FORMAT_NAMES, OVERRIDES, RETIRED_SHAPE, STAGES, Precision, StagePrecision
from engine.valuation.portfolio import PRODUCTS


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


class CouponPeriodSchema(BaseModel):
    """One explicit coupon period of a bond; enumerated, not generated (see
    `engine.instruments.treasury.CouponPeriod`)."""
    start_date: str
    end_date: str
    #: Defaults to `end_date`.
    payment_date: Optional[str] = None


#: A number format of `engine.precision.FORMATS`, by name.
FormatName = Literal[FORMAT_NAMES]

#: A product the pipeline prices (`engine.valuation.portfolio.PRODUCTS`, the trades' `trade_type`).
ProductName = Literal[PRODUCTS]


class StagePrecisionSchema(BaseModel):
    """`engine.precision.StagePrecision`; validated by the dataclass."""
    model_config = ConfigDict(extra="forbid")
    storage: FormatName = "float64"
    compute: FormatName = "float64"
    accumulate: FormatName = "float64"


class PrecisionSchema(BaseModel):
    """`engine.precision.Precision`: storage, compute and accumulate per adjustable stage, each
    float64 by default, and the pricing stage overridden per product (`by_product`, keyed by
    `trade_type`) and per trade (`by_trade`, keyed by `trade_id`). The 32/64 shape before
    roadmap 1.4 is refused, naming the replacement (decision A-12)."""
    model_config = ConfigDict(extra="forbid")
    simulation: StagePrecisionSchema = Field(default_factory=StagePrecisionSchema)
    market: StagePrecisionSchema = Field(default_factory=StagePrecisionSchema)
    pricing: StagePrecisionSchema = Field(default_factory=StagePrecisionSchema)
    by_product: Dict[ProductName, StagePrecisionSchema] = Field(default_factory=dict)
    by_trade: Dict[str, StagePrecisionSchema] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _refuse_the_retired_shape(cls, data):
        if isinstance(data, dict) and (any(k in data for k in ("risk", "calibration"))
                                       or any(isinstance(v, int) for v in data.values())):
            raise ValueError(RETIRED_SHAPE)
        return data

    def to_dataclass(self) -> Precision:
        stages = {stage: _stage(f"precision.{stage}", getattr(self, stage)) for stage in STAGES}
        overrides = {name: {key: _stage(f"precision.{name}[{key!r}]", value)
                            for key, value in getattr(self, name).items()} for name in OVERRIDES}
        return Precision(**stages, **overrides)


def _stage(where: str, schema: StagePrecisionSchema) -> StagePrecision:
    """The dataclass of one `StagePrecisionSchema`, its refusal naming the wire field."""
    try:
        return StagePrecision(**schema.model_dump())
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from None


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
    #: ORE's Basel EPE_B / EEPE_B at the one-year horizon.
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
    # Every trade's id, in request order: the key of each per-trade figure (I-10).
    trade_ids: List[str] = Field(default_factory=list)

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
