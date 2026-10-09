"""
The results of the HTTP API: a portfolio's, a market-risk run's, the cross-asset model's
calibration and the standalone LGM calibration's, with the precision report and the array
artifacts they carry; job status, health and version. The requests, and the wire forms the
results echo (the precision policy), are `engine.api.requests`. Each schema converts with `.to_dataclass()` /
`.from_dataclass()`; the dataclasses remain the source of truth, and nothing below `engine/api/`
imports Pydantic.

Every per-trade figure of a result is a row carrying its `trade_id` (I-10), in request order,
which is also the trade axis of the NPV cube and of the P&L matrix.

ORE types travel as strings: dates as ISO `YYYY-MM-DD` (`ORE.DateParser.parseISO`), periods
in ORE syntax (`"5Y"`, `"18M"`), fixings as `{"YYYY-MM-DD": rate}`.
"""
import dataclasses
from typing import Dict, List, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, Field

from engine.api.requests import PrecisionSchema, StagePrecisionSchema
from engine.calibration.cam import CamCalibration
from engine.precision import MeanEstimate, PrecisionReport
from engine.risk.counterparty.exposure import ExposureProfile
from engine.risk.market import MarketRiskResult
from engine.run import PortfolioResult


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
    """`engine.risk.counterparty.exposure.ExposureProfile`; every list is indexed like `times`, starting
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
    `vega:<ccy>`, option tenors x swap tenors) also has its shape in `shapes`."""
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


class ArrayChunkSchema(BaseModel):
    """One chunk of an array artifact: rows `[start, stop)` of its first axis."""
    url: str
    rows: List[int]
    bytes: int
    sha256: str


class ItemOrderSchema(BaseModel):
    """The order of one axis's items (trade ids), hashed (`engine.api.artifacts.items_record`)."""
    axis: str
    ids: List[str]
    sha256: str


class ArrayArtifactSchema(BaseModel):
    """A large array returned by reference (`engine.api.artifacts`, decision A-17): read each
    chunk from its `url`, check its `sha256`, and concatenate them in order into a C-ordered
    array of `dtype` and `shape` (`engine.api.artifacts.read_array` does all of it)."""
    name: str
    dtype: str
    byte_order: Literal["little"]
    shape: List[int]
    axes: List[str]
    chunks: List[ArrayChunkSchema]
    sha256: str
    items: ItemOrderSchema


class TradeValueSchema(BaseModel):
    """One trade's row: its id and its t=0 value."""
    trade_id: str
    base_npv: float


class TradeResultSchema(TradeValueSchema):
    """One trade's row of a portfolio result: its standalone exposure (null without scenario
    risk) and its Greeks (null unless requested)."""
    exposure: Optional[ExposureProfileSchema] = None
    greeks: Optional[GreeksSchema] = None


class PortfolioResultSchema(BaseModel):
    """`engine.run.PortfolioResult` on the wire. `base_npv` is the sum of the trades'.
    The cube, `[Scenarios, Dates, Trades]` with the trade axis in `trades`' order, is inline
    (`npv_cube`), by reference (`npv_cube_artifact`) or absent, as the request's `cube_output`
    asked."""
    base_npv: float
    #: Every trade's figures, in request order (I-10).
    trades: List[TradeResultSchema]
    #: The whole portfolio as one netting set; null without scenario risk.
    exposure: Optional[ExposureProfileSchema] = None
    npv_cube: Optional[List[List[List[float]]]] = None
    npv_cube_artifact: Optional[ArrayArtifactSchema] = None
    warnings: List[str] = Field(default_factory=list)
    # Whether the cube and the exposure were computed; false means absent, not zero (I-24).
    scenario_risk_available: bool = True
    # Measure of the exposure (`risk-neutral-pricing`), or null (I-11).
    measure: Optional[str] = None
    # The precision as run, read in the worker that ran the job (I-12).
    precision: Optional[PrecisionReportSchema] = None

    @classmethod
    def from_dataclass(cls, result: PortfolioResult, cube_output: str = "inline",
                       cube_artifact: Optional[Dict] = None) -> "PortfolioResultSchema":
        exposures = result.trade_exposures or [None] * len(result.trade_ids)
        greeks = result.greeks or {}
        return cls(
            base_npv=result.base_npv,
            trades=[TradeResultSchema(
                trade_id=trade_id, base_npv=float(npv),
                exposure=ExposureProfileSchema.from_dataclass(profile) if profile is not None else None,
                greeks=GreeksSchema.from_dataclass(greeks[i]) if i in greeks else None)
                for i, (trade_id, npv, profile) in enumerate(zip(result.trade_ids, result.base_npv_per_trade,
                                                                 exposures))],
            exposure=(
                ExposureProfileSchema.from_dataclass(result.exposure)
                if result.exposure is not None else None
            ),
            npv_cube=np.asarray(result.npv_cube).tolist() if cube_output == "inline" else None,
            npv_cube_artifact=cube_artifact,
            warnings=list(result.warnings),
            scenario_risk_available=result.scenario_risk_available,
            measure=result.measure,
            precision=(PrecisionReportSchema.from_dataclass(result.precision)
                       if result.precision is not None else None),
        )


class MarketRiskResultSchema(BaseModel):
    """`engine.risk.market.MarketRiskResult` on the wire: VaR/ES of the portfolio P&L
    (`risk`: `VaR_99`, `ES_97.5`, `ES_97.5_tailCount`, `ES_97.5_standardError`, ...; NaN as
    null), each trade's t=0 value, the scenarios' description, and the P&L matrix
    `[Scenarios, Trades]` (trade axis in `trades`' order) inline (`pnl`), by reference
    (`pnl_artifact`) or absent, as the request's `pnl_output` asked. The portfolio P&L the
    statistics use is the sum of each scenario's row."""
    base_npv: float
    trades: List[TradeValueSchema]
    risk: Dict[str, Optional[float]]
    measure: str
    source: str
    horizon_days: int
    num_scenarios: int
    #: The factors shocked, `"<curve>/<pillar time>y"` in the scenarios' order.
    risk_factors: List[str]
    pnl: Optional[List[List[float]]] = None
    pnl_artifact: Optional[ArrayArtifactSchema] = None
    warnings: List[str] = Field(default_factory=list)
    precision: Optional[PrecisionReportSchema] = None

    @classmethod
    def from_dataclass(cls, result: MarketRiskResult, trade_ids: List[str], pnl_output: str = "inline",
                       pnl_artifact: Optional[Dict] = None) -> "MarketRiskResultSchema":
        return cls(
            base_npv=result.base_npv,
            trades=[TradeValueSchema(trade_id=i, base_npv=float(v)) for i, v in zip(trade_ids,
                                                                                      result.base_npv_per_trade)],
            risk={key: _floats(value)[0] for key, value in result.risk.items()},
            measure=result.measure, source=result.source, horizon_days=result.horizon_days,
            num_scenarios=result.num_scenarios, risk_factors=list(result.risk_factors),
            pnl=np.asarray(result.pnl, dtype=np.float64).tolist() if pnl_output == "inline" else None,
            pnl_artifact=pnl_artifact, warnings=list(result.warnings),
            precision=(PrecisionReportSchema.from_dataclass(result.precision)
                       if result.precision is not None else None),
        )


class CalibrationHelperSchema(BaseModel):
    """One helper of a calibration basket: its expiry and term, and its value in the market
    (Bachelier on the ATM volatility) and in the calibrated model."""
    expiry: str
    term: str
    market_value: float
    model_value: float


class CurrencyCalibrationSchema(BaseModel):
    """One currency's calibrated volatility, in its model's parametrization (the LGM's alpha, or
    the Hull-White short rate's sigma): `sigma_values[i]` on bucket `i` of `[0, sigma_times[0]),
    ..., [sigma_times[-1], inf)`, the reversion it was calibrated with, and each helper."""
    model: Literal["LGM", "HullWhite"]
    reversion: float
    sigma_times: List[float]
    sigma_values: List[float]
    helpers: List[CalibrationHelperSchema]

    @classmethod
    def from_dataclass(cls, calibration: CamCalibration, model) -> "CurrencyCalibrationSchema":
        return cls(model="HullWhite" if model.volatility_type == "HullWhite" else "LGM",
                   reversion=float(model.reversion), sigma_times=_floats(calibration.sigma.times),
                   sigma_values=_floats(calibration.sigma.values),
                   helpers=[CalibrationHelperSchema(expiry=e, term=t, market_value=float(m), model_value=float(v))
                            for e, t, m, v in zip(model.calibration_expiries, model.calibration_terms,
                                                  calibration.market, calibration.model)])


class CamCalibrationResultSchema(BaseModel):
    """`engine.calibration.cam.calibrate_cam`'s result: each requested currency's calibration,
    the one a portfolio run with this market and these models simulates with."""
    currencies: Dict[str, CurrencyCalibrationSchema]


class JobStatusSchema(BaseModel):
    """A job's row in the job queue (`engine.api.job_queue`): its `kind` (`portfolio` or
    `market-risk`) and status, "pending" (queued), "running", "done" with the result (the kind's
    result schema), "failed" with a failure class and the worker's traceback, or "interrupted"
    (the engine worker stopped during the job; submit it again)."""
    kind: Literal["portfolio", "market-risk"]
    status: Literal["pending", "running", "done", "failed", "interrupted"]
    result: Optional[Union[PortfolioResultSchema, MarketRiskResultSchema]] = None
    error: Optional[str] = None
    #: Set when "failed": bad-terms, missing-market-data, unsupported-product,
    #: numerical-failure or infrastructure (I-08).
    failure_class: Optional[Literal[
        "bad-terms", "missing-market-data", "unsupported-product", "numerical-failure", "infrastructure"]] = None


class HealthSchema(BaseModel):
    status: Literal["ok"] = "ok"


class VersionSchema(BaseModel):
    engine_version: str
    jax_backend: str
    git_commit: Optional[str] = None


class CalibrationResultSchema(BaseModel):
    sigma_times: List[float]
    sigma_values: List[float]
    market_prices: List[float]
    model_prices: List[float]
    rmse: float
