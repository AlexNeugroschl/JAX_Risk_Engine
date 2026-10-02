"""
Pydantic schemas shared by the HTTP API: the wire forms of curves, coupon periods, precision,
results, job status, health and version, and the standalone calibration route's request.
The portfolio request itself is `engine.api.market_schemas`. Each schema converts with
`.to_dataclass()` / `.from_dataclass()`; the dataclasses remain the source of truth, and
nothing below `engine/api/` imports Pydantic.

ORE types travel as strings: dates as ISO `YYYY-MM-DD` (`ORE.DateParser.parseISO`), periods
in ORE syntax (`"5Y"`, `"18M"`), fixings as `{"YYYY-MM-DD": rate}`.
"""
import dataclasses
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import ORE
from pydantic import BaseModel, ConfigDict, Field, model_validator

from engine.market import ZeroCurveConfig
from engine.risk.exposure import ExposureProfile
from engine.portfolio import PortfolioResult
from engine.precision import (
    FORMAT_NAMES, OVERRIDES, RETIRED_SHAPE, ROUNDINGS, STAGES, MeanEstimate, Precision, PrecisionReport,
    StagePrecision,
)
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

#: A rounding into a scaled storage format (`engine.precision.ROUNDINGS`).
RoundingName = Literal[ROUNDINGS]

#: A product the pipeline prices (`engine.valuation.portfolio.PRODUCTS`, the trades' `trade_type`).
ProductName = Literal[PRODUCTS]


class StagePrecisionSchema(BaseModel):
    """`engine.precision.StagePrecision`; validated by the dataclass."""
    model_config = ConfigDict(extra="forbid")
    storage: FormatName = "float64"
    compute: FormatName = "float64"
    accumulate: FormatName = "float64"

    @classmethod
    def from_dataclass(cls, stage: StagePrecision) -> "StagePrecisionSchema":
        return cls(**dataclasses.asdict(stage))


class PrecisionSchema(BaseModel):
    """`engine.precision.Precision`: storage, compute and accumulate per adjustable stage, each
    float64 by default, and the pricing stage overridden per product (`by_product`, keyed by
    `trade_type`) and per trade (`by_trade`, keyed by `trade_id`); the rounding into a scaled
    storage format and its seed. The 32/64 shape before roadmap 1.4 is refused, naming the
    replacement (decision A-12). `paired_fraction`: the share of paths also run at float64, whose
    estimates the result's `precision` report carries (decision A-13)."""
    model_config = ConfigDict(extra="forbid")
    simulation: StagePrecisionSchema = Field(default_factory=StagePrecisionSchema)
    market: StagePrecisionSchema = Field(default_factory=StagePrecisionSchema)
    pricing: StagePrecisionSchema = Field(default_factory=StagePrecisionSchema)
    by_product: Dict[ProductName, StagePrecisionSchema] = Field(default_factory=dict)
    by_trade: Dict[str, StagePrecisionSchema] = Field(default_factory=dict)
    rounding: RoundingName = "nearest"
    rounding_seed: int = Field(default=0, ge=0)
    paired_fraction: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _refuse_the_retired_shape(cls, data):
        """The 32/64 shape: a `risk` or `calibration` field, or a stage given as a bit count."""
        if isinstance(data, dict) and (any(k in data for k in ("risk", "calibration"))
                                       or any(isinstance(data.get(s), int) for s in STAGES)):
            raise ValueError(RETIRED_SHAPE)
        return data

    def to_dataclass(self) -> Precision:
        stages = {stage: _stage(f"precision.{stage}", getattr(self, stage)) for stage in STAGES}
        overrides = {name: {key: _stage(f"precision.{name}[{key!r}]", value)
                            for key, value in getattr(self, name).items()} for name in OVERRIDES}
        return Precision(**stages, **overrides, rounding=self.rounding, rounding_seed=self.rounding_seed,
                         paired_fraction=self.paired_fraction)

    @classmethod
    def from_dataclass(cls, precision: Precision) -> "PrecisionSchema":
        stage = StagePrecisionSchema.from_dataclass
        return cls(**{s: stage(getattr(precision, s)) for s in STAGES},
                   **{name: {k: stage(v) for k, v in getattr(precision, name).items()} for name in OVERRIDES},
                   rounding=precision.rounding, rounding_seed=precision.rounding_seed,
                   paired_fraction=precision.paired_fraction)


def _stage(where: str, schema: StagePrecisionSchema) -> StagePrecision:
    """The dataclass of one `StagePrecisionSchema`, its refusal naming the wire field."""
    try:
        return StagePrecision(**schema.model_dump())
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from None


def _floats(values) -> List[Optional[float]]:
    """An array (or a scalar) as a flat list of floats, NaN as null."""
    return [None if np.isnan(v) else float(v) for v in np.atleast_1d(np.asarray(values, dtype=np.float64)).ravel()]


class MeanEstimateSchema(BaseModel):
    """`engine.precision.MeanEstimate`: a mean figure's two-level estimate, each array a list
    (per simulation date for an exposure figure), NaN as null."""
    kind: Literal["mean"] = "mean"
    value: List[Optional[float]]
    uncorrected: List[Optional[float]]
    correction: List[Optional[float]]
    standard_error: List[Optional[float]]
    uncorrected_standard_error: List[Optional[float]]
    correction_standard_error: List[Optional[float]]
    max_difference: List[Optional[float]]
    paths: int
    paired_paths: int


class QuantileEstimateSchema(BaseModel):
    """`engine.precision.QuantileEstimate`: a quantile figure measured on the paired sample."""
    kind: Literal["quantile"] = "quantile"
    value: List[Optional[float]]
    paired: List[Optional[float]]
    paired_float64: List[Optional[float]]
    difference: List[Optional[float]]
    paths: int
    paired_paths: int


def _estimate_schema(estimate) -> Union[MeanEstimateSchema, QuantileEstimateSchema]:
    schema = MeanEstimateSchema if isinstance(estimate, MeanEstimate) else QuantileEstimateSchema
    fields = {f.name: getattr(estimate, f.name) for f in dataclasses.fields(estimate)}
    return schema(**{k: v if k.endswith("paths") else _floats(v) for k, v in fields.items()})


class PrecisionReportSchema(BaseModel):
    """`engine.precision.PrecisionReport`: the precision as run, read from the run's arrays in
    the worker that ran it (I-12): the policy, each trade's pricing stage, the realized format
    of every stored array, the devices and backend, and with a paired float64 sample each
    figure's estimate, keyed `"netting_set/EPE"`, `"trades/<trade id>/PFE_95"`."""
    policy: PrecisionSchema
    trades: Dict[str, StagePrecisionSchema]
    realized: Dict[str, str]
    devices: List[str]
    backend: str
    jax_version: str
    paths: int
    paired_paths: int
    figures: Dict[str, Union[MeanEstimateSchema, QuantileEstimateSchema]] = Field(default_factory=dict)

    @classmethod
    def from_dataclass(cls, report: PrecisionReport) -> "PrecisionReportSchema":
        return cls(policy=PrecisionSchema.from_dataclass(report.policy),
                   trades={k: StagePrecisionSchema.from_dataclass(v) for k, v in report.trades.items()},
                   realized=dict(report.realized), devices=list(report.devices), backend=report.backend,
                   jax_version=report.jax_version, paths=report.paths, paired_paths=report.paired_paths,
                   figures={k: _estimate_schema(v) for k, v in report.figures.items()})


class RiskMetricsSchema(BaseModel):
    """A `compute_risk_metrics` dict on the wire: key -> values, NaN as null."""
    values: Dict[str, List[Optional[float]]]

    @classmethod
    def from_dataclass(cls, risk: Dict[str, "np.ndarray"]) -> "RiskMetricsSchema":
        return cls(values={key: _floats(values) for key, values in risk.items()})


class ExposureProfileSchema(BaseModel):
    """`engine.risk.exposure.ExposureProfile`; every list is indexed like `times`, starting
    at t=0. Its paired-sample `estimates` travel in the result's `precision.figures`."""
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
    # The precision as run, read in the worker that ran the job (roadmap 1.7, I-12).
    precision: Optional[PrecisionReportSchema] = None

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
            precision=(PrecisionReportSchema.from_dataclass(result.precision)
                       if result.precision is not None else None),
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
