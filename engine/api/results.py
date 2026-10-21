"""
The result of a run request on the wire (I-87): `RunResultSchema`, one document for every
analytic of the run, which says what each figure is without the request beside it: the
configuration as run (every default filled in), the as-of date, the simulation dates, the
reporting currency, the trade ids on every per-trade axis, each Greek's factor labels, the
pricing settings and calibration policy each analytic ran with, and the engine's identity (its
git revision, and the versions of jax, jaxlib and ORE). Also the run's status, health and
version. The requests, and the wire forms the results echo (the configuration, the precision
policy), are `engine.api.requests`. The dataclasses remain the source of truth, and nothing
below `engine/api/` imports Pydantic.

Every per-trade figure is a row carrying its `trade_id` (I-10), in request order, which is also
the trade axis of the NPV cube and of the P&L. Every large array is inline or by reference
(`outputs`, decision A-17) with the same description: its name, dtype, shape, the name of each
axis and the hashed order of its trade (or option) axis.

ORE types travel as strings: dates as ISO `YYYY-MM-DD` (`ORE.DateParser.parseISO`), periods
in ORE syntax (`"5Y"`, `"18M"`), fixings as `{"YYYY-MM-DD": rate}`.
"""
import dataclasses
import datetime
from typing import Annotated, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

from engine.api.artifacts import array_artifact, items_record
from engine.api.requests import PrecisionSchema, RunRequestConfigSchema, RunRequestSchema, StagePrecisionSchema
from engine.calibration.cam import CamCalibration
from engine.precision import IntervalAvailability, MeanEstimate, PrecisionReport
from engine.pricing.cube import PRODUCTS
from engine.risk.counterparty.exposure import ExposureProfile
from engine.run import EngineIdentity, RunResult


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
    of every stored array, every compute format whose matrix products ran and whether the
    device ran it natively or the engine emulated it (`products`), the devices and backend, and
    with a paired float64 sample each figure's estimate, keyed `"netting_set/EPE"`,
    `"trades/<trade id>/PFE_95"`."""
    policy: PrecisionSchema
    trades: Dict[str, StagePrecisionSchema]
    realized: Dict[str, str]
    products: Dict[str, Literal["native", "emulated"]] = Field(default_factory=dict)
    devices: List[str]
    backend: str
    jax_version: str
    paths: int
    paired_paths: int
    figures: Dict[str, Union[MeanEstimateSchema, QuantileEstimateSchema]] = Field(default_factory=dict)
    replicates: int = 1
    uncertainty_method: str = "unavailable"
    interval: IntervalAvailability = Field(default_factory=IntervalAvailability)
    qualification_scope: str = PrecisionReport.__dataclass_fields__["qualification_scope"].default

    @classmethod
    def from_dataclass(cls, report: PrecisionReport) -> "PrecisionReportSchema":
        return cls(policy=PrecisionSchema.from_dataclass(report.policy),
                   trades={k: StagePrecisionSchema.from_dataclass(v) for k, v in report.trades.items()},
                   realized=dict(report.realized), products=dict(report.products), devices=list(report.devices),
                   backend=report.backend, jax_version=report.jax_version, paths=report.paths,
                   paired_paths=report.paired_paths, figures={k: _estimate_schema(v) for k, v in report.figures.items()},
                   replicates=report.replicates, uncertainty_method=report.uncertainty_method,
                   interval=report.interval, qualification_scope=report.qualification_scope)


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




# --- arrays -------------------------------------------------------------------------------------

class ItemOrderSchema(BaseModel):
    """The order of one axis's items (trade ids), hashed (`engine.api.artifacts.items_record`)."""
    axis: str
    ids: List[str]
    sha256: str


class _ArrayDescription(BaseModel):
    """What every large array of a result says of itself, inline or by reference: C-ordered (the
    last axis fastest), each axis named, the trade (or option) axis's order hashed."""
    name: str
    dtype: str
    shape: List[int]
    axes: List[str]
    items: ItemOrderSchema


class InlineArraySchema(_ArrayDescription):
    """An array in the result: `values` nested lists of `shape`."""
    form: Literal["inline"] = "inline"
    values: list


class ArrayChunkSchema(BaseModel):
    """One chunk of an array artifact: rows `[start, stop)` of its first axis."""
    url: str
    rows: List[int]
    bytes: int
    sha256: str


class ArrayArtifactSchema(_ArrayDescription):
    """An array by reference (`engine.api.artifacts`, decision A-17): read each chunk from its
    `url`, check its `sha256`, and concatenate them in order into a C-ordered array of `dtype`
    and `shape` (`engine.api.artifacts.read_array` does all of it)."""
    form: Literal["artifact"] = "artifact"
    byte_order: Literal["little"]
    chunks: List[ArrayChunkSchema]
    sha256: str


#: A large array as the request's `outputs` asked: inline or by reference.
ArrayOutputSchema = Annotated[Union[InlineArraySchema, ArrayArtifactSchema], Field(discriminator="form")]


def array_output(array, name: str, axes: Sequence[str], items_axis: str, ids: Sequence[str], output: str,
                 url: str) -> Tuple[Optional[Union[InlineArraySchema, ArrayArtifactSchema]], List[bytes]]:
    """`array` as `output` asks (`inline`, `artifact` with its chunks served under `url`, or
    `none`), its `items_axis` named by `ids`; and the chunks to store with the result."""
    if output == "none":
        return None, []
    values = np.asarray(array)
    items = items_record(items_axis, ids)
    if output == "artifact":
        reference, chunks = array_artifact(values, name, url, axes, items)
        return ArrayArtifactSchema(**reference), chunks
    if len(axes) != values.ndim or len(items["ids"]) != values.shape[list(axes).index(items_axis)]:
        raise ValueError(f"{name}: axes {list(axes)} and {len(ids)} {items_axis} ids for shape {values.shape}")
    return InlineArraySchema(name=name, dtype=values.dtype.name, shape=list(values.shape), axes=list(axes),
                             items=items, values=values.tolist()), []


# --- the run result -----------------------------------------------------------------------------

class TradeRowSchema(BaseModel):
    """One trade's row: its id and product, its value today (with `npv`, `exposure` or
    `sensitivities`), its standalone exposure (with `exposure`) and its Greeks (with
    `sensitivities`); null where its analytic was not requested."""
    trade_id: str
    product: Literal[PRODUCTS]
    npv: Optional[float] = None
    exposure: Optional[ExposureProfileSchema] = None
    greeks: Optional[GreeksSchema] = None


class ExposureSectionSchema(BaseModel):
    """The exposure analytic: the netting set's profile (the whole portfolio, no collateral),
    each list indexed like `times`, whose first entry is the as-of date and the rest `dates`;
    the cube `[scenario, date, trade]` on `dates` and each option's exercise decisions
    `[scenario, date, option]` (whether the holder has exercised on the path on or before the
    date) as `outputs` asked; the measure (I-11), the precision as run (I-12), and each product's
    pricing settings (`model=fixed`: an option keeps the model calibrated when it was built, A-24)."""
    dates: List[str]
    netting_set: ExposureProfileSchema
    npv_cube: Optional[ArrayOutputSchema] = None
    exercised: Optional[ArrayOutputSchema] = None
    measure: str
    precision: PrecisionReportSchema
    pricing_settings: Dict[str, str]


class SensitivitiesSectionSchema(BaseModel):
    """The sensitivities analytic: the method as run (`Bump`, ORE's sensitivity analysis, or
    `AD`), each Greek's factors in its row's flattened order (`delta:<curve>` and `gamma:<curve>`
    per curve tenor, `vega:<ccy>` per `<option tenor>x<swap tenor>` row-major), and each product's
    pricing settings (`model=recalibrated`: a calibrated option recalibrated under every shift)."""
    method: Literal["Bump", "AD"]
    factors: Dict[str, List[str]]
    pricing_settings: Dict[str, str]


class TradeValueSchema(BaseModel):
    """One trade's value on today's curves, as the market-risk analytic revalues it."""
    trade_id: str
    base_npv: float


class MarketRiskSectionSchema(BaseModel):
    """The market-risk analytic (`engine.risk.market.MarketRiskResult`): VaR/ES of the portfolio
    P&L (`risk`: `VaR_99`, `ES_97.5`, `ES_97.5_tailCount`, `ES_97.5_standardError`, ...; NaN as
    null), each trade's value on today's curves, the scenarios' description, the factors shocked
    (`"<curve>/<pillar time>y"` in the scenarios' order), the P&L `[scenario, trade]` as `outputs`
    asked (the portfolio P&L the statistics use is the sum of each scenario's row), the precision
    as run, the uncertainty method, and each product's pricing settings (`model=recalibrated`: a
    calibrated option recalibrated under every scenario, I-112)."""
    base_npv: float
    trades: List[TradeValueSchema]
    risk: Dict[str, Optional[float]]
    measure: str
    source: str
    horizon_days: int
    num_scenarios: int
    risk_factors: List[str]
    pnl: Optional[ArrayOutputSchema] = None
    precision: Optional[PrecisionReportSchema] = None
    uncertainty: Dict[str, Union[str, int, Dict[str, Optional[str]]]] = Field(default_factory=dict)
    pricing_settings: Dict[str, str]


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
    ..., [sigma_times[-1], inf)`, the reversion it was calibrated with, each helper, and the
    calibration's `error` as ORE measures it against `config.models.bootstrap_tolerance` (the
    root mean square of the helpers' relative price errors)."""
    model: Literal["LGM", "HullWhite"]
    reversion: float
    sigma_times: List[float]
    sigma_values: List[float]
    helpers: List[CalibrationHelperSchema]
    error: float

    @classmethod
    def from_dataclass(cls, calibration: CamCalibration, model) -> "CurrencyCalibrationSchema":
        return cls(model="HullWhite" if model.volatility_type == "HullWhite" else "LGM",
                   reversion=float(model.reversion), sigma_times=_floats(calibration.sigma.times),
                   sigma_values=_floats(calibration.sigma.values),
                   helpers=[CalibrationHelperSchema(expiry=e, term=t, market_value=float(m), model_value=float(v))
                            for e, t, m, v in zip(model.calibration_expiries, model.calibration_terms,
                                                  calibration.market, calibration.model)],
                   error=float(calibration.error))


class CalibrationSectionSchema(BaseModel):
    """The calibration analytic (`engine.calibration.cam.calibrate_cam`): each calibrated
    currency's model, the one a run's exposure analytic simulates with on this market."""
    currencies: Dict[str, CurrencyCalibrationSchema]


class EngineIdentitySchema(BaseModel):
    """`engine.run.EngineIdentity`: what computed the result. `revision` is the engine's git
    commit (null outside a checkout), `modified` whether its tracked files differed from it."""
    version: str
    revision: Optional[str] = None
    modified: Optional[bool] = None
    jax: str
    jaxlib: str
    ore: str

    @classmethod
    def from_dataclass(cls, identity: EngineIdentity) -> "EngineIdentitySchema":
        return cls(**dataclasses.asdict(identity))


class RunResultSchema(BaseModel):
    """`engine.run.RunResult` on the wire (see the module docstring): one section per requested
    analytic, null for the others; `npv` the portfolio's value today, the sum of the rows';
    every amount in `base_currency`; `warnings` every analytic's."""
    run_id: str
    analytics: List[str]
    asof: str
    base_currency: str
    engine: EngineIdentitySchema
    config: RunRequestConfigSchema
    trades: List[TradeRowSchema]
    npv: Optional[float] = None
    exposure: Optional[ExposureSectionSchema] = None
    sensitivities: Optional[SensitivitiesSectionSchema] = None
    market_risk: Optional[MarketRiskSectionSchema] = None
    calibration: Optional[CalibrationSectionSchema] = None
    warnings: List[str] = Field(default_factory=list)


def run_result_document(result: RunResult, schema: RunRequestSchema, run_id: str,
                        artifacts_url: str) -> Tuple[RunResultSchema, Dict[str, List[bytes]]]:
    """The wire result of `result`, the run of `schema` (the request as the API parsed it, whose
    `outputs` say how each array travels and whose `config` is the configuration as run), and the
    chunks of each array sent by reference, served under `artifacts_url/<name>/<chunk>`."""
    request, chunks = result.request, {}

    def output(array, name, axes, axis, ids):
        described, data = array_output(array, name, axes, axis, ids, getattr(schema.outputs, name),
                                       f"{artifacts_url}/{name}")
        if data:
            chunks[name] = data
        return described

    portfolio = result.portfolio
    rows = [TradeRowSchema(trade_id=t.trade_id, product=t.product) for t in request.portfolio]
    sections, warnings = {}, []
    if portfolio is not None:
        warnings += portfolio.warnings
        greeks = portfolio.greeks or {}
        exposures = portfolio.trade_exposures or [None] * len(rows)
        for i, (row, npv, profile) in enumerate(zip(rows, portfolio.base_npv_per_trade, exposures)):
            row.npv = float(npv)
            row.exposure = ExposureProfileSchema.from_dataclass(profile) if profile is not None else None
            row.greeks = GreeksSchema.from_dataclass(greeks[i]) if i in greeks else None
        if request.wants("exposure"):
            sections["exposure"] = _exposure_section(result, output)
        if request.wants("sensitivities"):
            sections["sensitivities"] = _sensitivities_section(result)
    if result.market_risk is not None:
        warnings += result.market_risk.warnings
        sections["market_risk"] = _market_risk_section(result, output)
    if result.calibration is not None:
        models = request.config.models.ir
        sections["calibration"] = CalibrationSectionSchema(currencies={
            currency: CurrencyCalibrationSchema.from_dataclass(calibration, models[currency])
            for currency, calibration in result.calibration.items()})
    document = RunResultSchema(
        run_id=run_id, analytics=list(request.analytics), asof=request.market.asof.ISO(),
        base_currency=request.config.base_currency, engine=EngineIdentitySchema.from_dataclass(result.engine),
        config=schema.config, trades=rows, npv=float(portfolio.base_npv) if portfolio is not None else None,
        warnings=list(dict.fromkeys(warnings)), **sections)
    return document, chunks


def _products(request) -> List[str]:
    return [t.product for t in request.portfolio]


def _exposure_section(result: RunResult, output) -> ExposureSectionSchema:
    request, portfolio = result.request, result.portfolio
    ids = result.trade_ids
    options = [t for t in ids if t in portfolio.exercised]
    paths, dates = request.config.simulation.samples, len(request.config.simulation.dates)
    exercised = (np.stack([np.asarray(portfolio.exercised[t]) for t in options], axis=-1) if options
                 else np.zeros((paths, dates, 0), dtype=bool))
    return ExposureSectionSchema(
        dates=[d.ISO() for d in request.config.simulation.dates],
        netting_set=ExposureProfileSchema.from_dataclass(portfolio.exposure),
        npv_cube=output(portfolio.npv_cube, "npv_cube", ("scenario", "date", "trade"), "trade", ids),
        exercised=output(exercised, "exercised", ("scenario", "date", "option"), "option", options),
        measure=portfolio.measure, precision=PrecisionReportSchema.from_dataclass(portfolio.precision),
        pricing_settings=request.config.pricing.kernel_settings(_products(request), recalibrated=False))


def _sensitivities_section(result: RunResult) -> SensitivitiesSectionSchema:
    """The factor labels of every Greek the rows carry (see `SensitivitiesSectionSchema`)."""
    request = result.request
    tenors = list(request.config.sensitivity.sensitivity.curve_tenors)
    factors: Dict[str, List[str]] = {}
    for greeks in (result.portfolio.greeks or {}).values():
        for key in greeks:
            if key == "theta" or key in factors:
                continue
            if key.startswith("vega:"):
                surface = request.market.swaption_vols(key.partition(":")[2])
                factors[key] = [f"{o}x{s}" for o in surface.option_tenors for s in surface.swap_tenors]
            else:
                factors[key] = tenors
    return SensitivitiesSectionSchema(
        method=request.config.sensitivity.method, factors=dict(sorted(factors.items())),
        pricing_settings=request.config.pricing.kernel_settings(_products(request), recalibrated=True))


def _market_risk_section(result: RunResult, output) -> MarketRiskSectionSchema:
    request, risk = result.request, result.market_risk
    ids = result.trade_ids
    return MarketRiskSectionSchema(
        base_npv=risk.base_npv,
        trades=[TradeValueSchema(trade_id=i, base_npv=float(v)) for i, v in zip(ids, risk.base_npv_per_trade)],
        risk={key: _floats(value)[0] for key, value in risk.risk.items()}, measure=risk.measure,
        source=risk.source, horizon_days=risk.horizon_days, num_scenarios=risk.num_scenarios,
        risk_factors=list(risk.risk_factors),
        pnl=output(np.asarray(risk.pnl, dtype=np.float64), "pnl", ("scenario", "trade"), "trade", ids),
        precision=PrecisionReportSchema.from_dataclass(risk.precision) if risk.precision is not None else None,
        uncertainty=risk.uncertainty,
        pricing_settings=request.config.pricing.kernel_settings(_products(request), recalibrated=True))


# --- status, health, version --------------------------------------------------------------------

#: A run's statuses (`engine.api.job_queue.STATUSES`).
RunStatus = Literal["pending", "running", "done", "failed", "interrupted", "cancelled"]
#: Why a run failed (`engine.api.job_queue.FAILURE_CLASSES`, I-08).
FailureClass = Literal["bad-terms", "missing-market-data", "unsupported-product", "numerical-failure",
                       "infrastructure"]


def iso_time(seconds: Optional[float]) -> Optional[str]:
    """A queue timestamp (seconds since the epoch) as ISO 8601 UTC, None as None."""
    if seconds is None:
        return None
    return datetime.datetime.fromtimestamp(seconds, tz=datetime.timezone.utc).isoformat(timespec="milliseconds")


class RunStatusSchema(BaseModel):
    """A run's row in the job queue (`engine.api.job_queue`): "pending" (queued), "running",
    "done" with its `result`, "failed" with a failure class and the worker's traceback,
    "interrupted" (the engine worker stopped during the run; submit it again) or "cancelled".
    While pending, `error` says why no engine worker is running it, if one cannot (I-77, I-106).
    `cancel_requested` is set once a cancel reached a running run, which stops at its next phase
    boundary (I-91). `kind` is `run`, or for a job a version-2 queue held, the kind of the route
    it was submitted to (its result that route's document). Times are ISO 8601 UTC."""
    run_id: str
    kind: Literal["run", "portfolio", "market-risk"] = "run"
    status: RunStatus
    result: Optional[RunResultSchema] = None
    error: Optional[str] = None
    failure_class: Optional[FailureClass] = None
    cancel_requested: bool = False
    keep: bool = False
    submitted: Optional[str] = None
    started: Optional[str] = None
    finished: Optional[str] = None


class WorkerHealthSchema(BaseModel):
    """The engine worker as this API process sees it (`engine.api.supervisor`): `mode` (`spawn`:
    this process starts it; `external`: something else does), whether a worker serves this
    process's queue, how many times running it has exited without serving a run (I-77), and why
    no worker can serve the queue, if that is known (a worker that cannot start, or this host's
    worker serving another queue, I-106)."""
    mode: Literal["spawn", "external"]
    serving: bool
    failed_starts: int = 0
    error: Optional[str] = None


class HealthSchema(BaseModel):
    """The API process is up; and its engine worker's state."""
    status: Literal["ok"] = "ok"
    worker: Optional[WorkerHealthSchema] = None


class VersionSchema(BaseModel):
    """The engine's identity in the API process (`EngineIdentitySchema`), and its JAX backend,
    which is not where runs execute: each result names its own engine, and its precision report
    the devices that ran it."""
    engine: EngineIdentitySchema
    jax_backend: str
