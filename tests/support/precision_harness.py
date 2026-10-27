"""
The precision harness (I-75; docs/planning/details/precision.md §15.4), built as the storage
harness and measuring compute formats too since the kernel rewrite (F-07, 2026-10-10): how far
each precision policy moves every figure of a run from float64, measured on the run's own
paths, per case, product and path count, over independent Sobol seeds. Rerunnable on any kernel
change; the campaigns of record are `tests/support/precision_campaigns.py`, which also runs
them.

    Campaign: a portfolio run (`PortfolioCampaign`) or a market-risk run
              (`MarketRiskCampaign`), its cases (a labelled `Precision` each), path counts, seeds
      --measure--> Measurements, per path count:
          FigureRow       per case and figure, per entry (a date, or one for a scalar): the
                          precision error (the case's figure less the float64 run's on the
                          same paths) as its mean over seeds, the bias, with its standard
                          error, and its root mean square; and the figure's statistical error,
                          its float64 spread across seeds
          CoverageRow     per paired case and mean figure: how often the paired estimator's
                          stated standard errors cover what they claim to (decision A-13)
          AttributionRow  per market-risk case: the P&L attribution statistics (Spearman's
                          correlation, the Kolmogorov-Smirnov statistic) of its P&L against
                          float64's (decision A-11; the thresholds are the evidence table's, I-55)
          ExerciseRow     per portfolio case and option: the share of its paths whose exercise
                          decisions the case flips, per date (decision A-24: a cash-settled
                          option's flipped exercise moves it by its whole value, a precision
                          result, not a rounding)
          CaseRun         per case: XLA programs compiled after its first seed, warm seconds
      --save / summary--> every number (gzip-compressed JSON), and a markdown summary of it

Each path count's runs share their Sobol seeds (`CamConfig.seed`, the market-risk scenarios'
seed): seeds scramble the sequence independently, so the seeds are independent replicates of
each figure and their spread is its statistical error at that path count. A case's stochastic
rounding takes the seed as its `rounding_seed`. The float64 reference is `Precision()` on the
same paths: a case's error is its precision error alone.

Two ratios summarize a figure, each the worst over its entries: the relative error, the root
mean square precision error in units of the statistical error (what the Basel plan's P6.2 rule
bounds, A-11), and the bias's significance, the bias in units of its own standard error (a
systematic shift where it is large: I-75's, which more paths do not remove). An entry float64
holds fixed has no statistical error to measure against: a netting set's ENE on a date where
every path is in the money, an EEE_B or EEPE_B held at its t=0 value, a matured trade's value
(their spread over the seeds is 0, or float64's rounding, below `FIXED` of the figure's scale,
its largest float64 value). Such entries are left out of both ratios, and what a case moves
them by is the fixed error, relative to the figure's scale, so it is never hidden. On Sobol paths the
statistical error of a smooth figure falls nearly as 1/N, while a rounding error that varies
from path to path averages out as 1/sqrt(N), like Monte Carlo noise: so the relative error of
an unbiased format grows with the path count, and the path count at which it reaches a rule is
that format's path ceiling (the evidence table's, I-55).

The figures are what a result reports, plus the mean of the stored values per date
(`trades/<id>/NPV`, the class I-75 is about): for a portfolio, the netting set's and each
trade's EPE, ENE, EE_B, EEE_B, EPE_B, EEPE_B and PFE per simulation date and their Basel EPE
and EEPE; for market risk, the portfolio's VaR and ES and each trade's mean P&L. Coverage, per
paired case, per seed and date of each two-level estimate (`MeanEstimate`):

  * precision: |correction - the other seeds' mean full-draw correction| within the critical
    multiple of `sqrt(correction_standard_error^2 + s_c^2 / (m - 1))`, where s_c is the
    independent outer correction's spread;
  * total: |value - the other seeds' mean float64 figure| within the critical multiple of
    `sqrt(standard_error^2 + s^2 / (m - 1))`, s the float64 figure's spread over the m seeds:
    the report's standard error against an independent estimate of the float64 expectation.

The critical multiple is Student t with R-1 degrees of freedom for R independent inner
scrambles; a single scramble has unavailable errors and no coverage checks. There is no
IID finite-population adjustment. These are diagnostic 95% intervals, never available
production confidence intervals. Rows are per entry/date so pooling cannot hide a
failure. Zero-width misses beyond numerical roundoff count as infinite standardized
errors; numerical equality is recorded separately and never proves a constant.
"""
import dataclasses
import datetime
import gzip
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import jax
import numpy as np
from scipy import stats

from engine.api.requests import PrecisionSchema
from engine.market_data.market import Market
from engine.market_simulation.config import CamConfig
from engine.precision import FORMATS, OFFSETS, ROUNDINGS, STAGES, MeanEstimate, Precision, StagePrecision
from engine.pricing.config import PricingConfig
from engine.risk.counterparty.exposure import BASEL_FIGURES, PROFILE_FIGURES
from engine.risk.market import MarketRiskRequest, RateRiskFactors, monte_carlo_scenarios, run_market_risk
from engine.risk.market.run import pricing_settings as market_risk_pricing_settings
from engine.run import PortfolioRequest, RunConfig, price_portfolio
from engine.run.pipeline import pricing_settings as portfolio_pricing_settings
from tests.support.compiles import count_compiles

#: The scaled storage formats, widest first.
SCALED = ("float16", "bfloat16", "float8_e4m3fn", "float8_e5m2")

#: An entry whose statistical error is below this share of its figure's scale is held fixed by
#: float64 (its spread over the seeds is float64's rounding, or 0): see the module docstring.
FIXED = 1e-9

#: The two-sided 95% quantile of the standard normal.
Z95 = 1.959963984540054

#: Where the repository is (the commit the measurements ran on).
_REPOSITORY = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Case:
    """One precision policy to measure, under a label that names it in every row."""
    label: str
    precision: Precision


def storage_cases(stages: Sequence[str] = STAGES, formats: Sequence[str] = ("float32",) + SCALED,
                  roundings: Sequence[str] = ROUNDINGS, offsets: Sequence[str] = OFFSETS,
                  paired_fraction: float = 0.0) -> Tuple[Case, ...]:
    """A case per stage, storage format, rounding and block layout: that stage stored in the
    format and computed in float64, every other stage float64, so the storage is the only
    difference from the reference. A plain format (float32) has one case per stage, rounding
    and layout being for the scaled ones. Labels: `"<stage>:<format>[:<rounding>:<offset>]"`,
    and `":paired"` after them with a paired sample."""
    cases = []
    for stage in stages:
        for name in formats:
            variants = [(r, o) for r in roundings for o in offsets] if name in SCALED else [("nearest", "block")]
            for rounding, offset in variants:
                label = f"{stage}:{name}" + (f":{rounding}:{offset}" if name in SCALED else "")
                cases.append(Case(label + (":paired" if paired_fraction else ""), Precision(
                    **{stage: StagePrecision(name)}, rounding=rounding, offset=offset, paired_fraction=paired_fraction)))
    return tuple(cases)


def throughout_case(name: str, paired_fraction: float = 0.0) -> Case:
    """Every adjustable stage computed in `name`, stored and accumulated in it where it can be
    (`Precision.throughout`: a format below float32 accumulates in float32, a product format is
    stored in float32)."""
    return Case(f"throughout:{name}" + (":paired" if paired_fraction else ""),
                dataclasses.replace(Precision.throughout(name), paired_fraction=paired_fraction,
                                    execution=Precision.research().execution if FORMATS[name].below_float32
                                    else Precision().execution))


#: The compute formats from float32 down: float32's own row is the floor every other one is
#: read against, its accumulator being theirs; each format below float32 whose product is one
#: pass carries the operand every path shares in two terms, and its single pass (`<name>_1x`)
#: follows it (decision A-23).
COMPUTE = ("float32", "bfloat16_3x", "tensorfloat32", "tensorfloat32_1x", "float16", "float16_1x", "bfloat16",
           "bfloat16_1x", "float8_e4m3fn", "float8_e4m3fn_1x", "float8_e5m2", "float8_e5m2_1x")


def compute_cases(stages: Sequence[str] = STAGES, formats: Sequence[str] = COMPUTE) -> Tuple[Case, ...]:
    """A case per stage and compute format: that stage's matrix products in the format,
    accumulated and stored in float32, every other stage float64, so the products and the
    float32 accumulator are the only differences from the reference (the `float32` case is the
    accumulator's share). Labels: `"<stage>:compute:<format>"`."""
    return tuple(Case(f"{stage}:compute:{name}", Precision.research(**{stage: StagePrecision("float32", name, "float32")}))
                 for stage in stages for name in formats)


# ---------------------------------------------------------------------------
# Campaigns
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PortfolioCampaign:
    """`price_portfolio` of `trades` on `market` under `simulation` (whose `samples` and `seed`
    the harness sets), with `pricing`, at each case's precision. `market_name` names the market
    in every row measured on it (the evidence table's, I-55)."""
    name: str
    description: str
    market_name: str
    market: Market
    trades: Tuple
    simulation: CamConfig
    pricing: PricingConfig
    cases: Tuple[Case, ...]
    paths: Tuple[int, ...]
    seeds: Tuple[int, ...]
    pfe_quantiles: Tuple[float, ...] = (0.95,)
    kind = "portfolio"

    def scenarios(self, paths: int, seed: int) -> CamConfig:
        return dataclasses.replace(self.simulation, samples=paths, seed=seed)

    def run(self, scenarios: CamConfig, precision: Precision):
        return price_portfolio(PortfolioRequest(
            market=self.market, trades=list(self.trades), pfe_quantiles=self.pfe_quantiles,
            config=RunConfig(simulation=scenarios, pricing=self.pricing, precision=precision)))

    def settings(self) -> Dict[str, str]:
        """Each product's pricing settings, as the portfolio run keys its evidence (I-114)."""
        return portfolio_pricing_settings(self.pricing, self.trades)

    def dates(self) -> Tuple[str, ...]:
        """The simulation dates, the entries of a profile figure."""
        return tuple(d.ISO() for d in self.simulation.dates)

    def figures(self, result) -> Dict[str, np.ndarray]:
        """Every figure of a `PortfolioResult` (see the module docstring) as float64: a profile
        per simulation date, t=0 excluded (it is float64 at every precision); a Basel figure
        0-dimensional."""
        cube = np.asarray(result.npv_cube, dtype=np.float64)
        out = {"netting_set/NPV": cube.sum(axis=-1).mean(axis=0)}
        out.update({f"trades/{t.trade_id}/NPV": cube[:, :, i].mean(axis=0) for i, t in enumerate(self.trades)})
        scopes = [("netting_set", result.exposure)] + [(f"trades/{t.trade_id}", p)
                                                      for t, p in zip(self.trades, result.trade_exposures)]
        for scope, profile in scopes:
            for name in PROFILE_FIGURES:
                out[f"{scope}/{name}"] = np.asarray(getattr(profile, name.lower()), dtype=np.float64)[1:]
            out.update({f"{scope}/{k}": np.asarray(v, dtype=np.float64)[1:] for k, v in profile.pfe.items()})
            for name in BASEL_FIGURES:
                out[f"{scope}/{name}"] = np.float64(getattr(profile, name.lower()))
        return out

    def pnl(self, result) -> Optional[np.ndarray]:
        return None

    def decisions(self, result) -> Dict[str, np.ndarray]:
        """Each option's exercise state `[paths, dates]` (`PortfolioResult.exercised`)."""
        return {trade: np.asarray(state, dtype=bool) for trade, state in result.exercised.items()}


@dataclass(frozen=True)
class MarketRiskCampaign:
    """`run_market_risk` of `trades` on `market` under Monte Carlo shifts of every curve
    pillar (`covariance` of the `horizon_days` moves, in `RateRiskFactors.from_market`'s
    order), `paths` scenarios each, at each case's precision. `market_name` as a portfolio
    campaign's."""
    name: str
    description: str
    market_name: str
    market: Market
    trades: Tuple
    covariance: np.ndarray
    horizon_days: int
    pricing: PricingConfig
    cases: Tuple[Case, ...]
    paths: Tuple[int, ...]
    seeds: Tuple[int, ...]
    quantiles: Tuple[float, ...] = (0.99, 0.975)
    replicates: int = 1
    kind = "market_risk"

    def scenarios(self, paths: int, seed: int):
        return monte_carlo_scenarios(RateRiskFactors.from_market(self.market), self.covariance, self.horizon_days,
                                     paths, seed=seed, replicates=self.replicates)

    def run(self, scenarios, precision: Precision):
        return run_market_risk(MarketRiskRequest(list(self.trades), self.market, scenarios, self.pricing,
                                                 quantiles=self.quantiles, precision=precision))

    def settings(self) -> Dict[str, str]:
        """Each product's pricing settings, as the market-risk run keys its evidence (I-114)."""
        return market_risk_pricing_settings(self.pricing, self.trades)

    def dates(self) -> Tuple[str, ...]:
        """None: every market-risk figure is a scalar."""
        return ()

    def figures(self, result) -> Dict[str, np.ndarray]:
        """The portfolio's VaR and ES, and the mean P&L of the portfolio and of each trade, each
        0-dimensional float64."""
        pnl = np.asarray(result.pnl, dtype=np.float64)
        out = {f"portfolio/{k}": np.float64(v) for k, v in result.risk.items()
               if k.startswith(("VaR_", "ES_")) and k.count("_") == 1}
        out["portfolio/PnL"] = np.float64(pnl.sum(axis=-1).mean())
        out.update({f"trades/{t.trade_id}/PnL": np.float64(pnl[:, i].mean()) for i, t in enumerate(self.trades)})
        return out

    def pnl(self, result) -> Optional[np.ndarray]:
        return np.asarray(result.portfolio_pnl, dtype=np.float64)

    def snapshot(self, result) -> Dict[str, np.ndarray]:
        """Required observable outputs, with trade identities preserved.

        Scenario PV is the reported arithmetic-consistent P&L anchor plus P&L, not an
        independently repriced FP64 today's value. Do not change the engine's anchor.
        """
        pnl = np.asarray(result.pnl, dtype=np.float64)
        base = np.asarray(result.base_npv_per_trade, dtype=np.float64)
        out = {"portfolio/PnL": np.asarray(result.portfolio_pnl),
               "portfolio/scenario_PV": np.asarray(result.portfolio_pnl) + result.base_npv,
               "portfolio/base_PV": np.asarray(result.base_npv)}
        for i, trade in enumerate(self.trades):
            out.update({f"trades/{trade.trade_id}/PnL": pnl[:, i],
                        f"trades/{trade.trade_id}/scenario_PV": base[i] + pnl[:, i],
                        f"trades/{trade.trade_id}/base_PV": np.asarray(base[i])})
        out.update({k: v for k, v in self.figures(result).items() if k.startswith("portfolio/")
                    and k != "portfolio/PnL"})
        return out

    def decisions(self, result) -> Dict[str, np.ndarray]:
        """None: market risk revalues today, where no option is exercised."""
        return {}


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FigureRow:
    """One figure of one case at one path count, per entry (see the module docstring)."""
    case: str
    paths: int
    figure: str
    product: str
    level: Tuple[float, ...]
    statistical_error: Tuple[float, ...]
    bias: Tuple[float, ...]
    bias_error: Tuple[float, ...]
    rms_error: Tuple[float, ...]
    max_error: float

    @property
    def scale(self) -> float:
        """The figure's largest float64 value, in magnitude, over its entries."""
        return max(abs(v) for v in self.level)

    @property
    def fixed(self) -> Tuple[bool, ...]:
        """Per entry, whether float64 holds it fixed: its statistical error below `FIXED` of
        the scale (see the module docstring)."""
        return tuple(not s > FIXED * self.scale for s in self.statistical_error)

    @property
    def relative_error(self) -> float:
        """The largest root mean square precision error / statistical error over the entries
        float64 does not hold fixed (0 if it holds them all)."""
        return _worst_ratio(*self._varying(self.rms_error))

    @property
    def relative_bias(self) -> float:
        """The largest |bias| / statistical error over the entries float64 does not hold fixed."""
        return _worst_ratio(*self._varying(self.bias))

    @property
    def significance(self) -> float:
        """The largest |bias| / its standard error over the entries: how far the measured bias
        is from zero in its own noise."""
        return _worst_ratio(self.bias, self.bias_error)

    @property
    def fixed_error(self) -> float:
        """The largest root mean square precision error over the entries float64 holds fixed,
        relative to the scale (0 without such entries; inf if the scale is 0 and they move)."""
        moved = [e for e, f in zip(self.rms_error, self.fixed) if f]
        return _worst_ratio(moved, [self.scale] * len(moved))

    def _varying(self, numerator):
        keep = [not f for f in self.fixed]
        return ([n for n, k in zip(numerator, keep) if k],
                [s for s, k in zip(self.statistical_error, keep) if k])


@dataclass(frozen=True)
class CoverageRow:
    """The paired estimator's coverage for one mean figure of one paired case (see the module
    docstring), for each interval: the checks made (seeds x dates where it has a positive
    width), how many it covered, and the root mean square of the standardized errors (1 if
    the standard error is right)."""
    case: str
    paths: int
    figure: str
    product: str
    precision_checks: int
    precision_covered: int
    precision_z_rms: float
    total_checks: int
    total_covered: int
    total_z_rms: float
    #: Empty on historical records, which pooled dates and omitted zero-width misses.
    entry: str = ""
    interval_status: str = "legacy_diagnostic"
    interval_unavailable: int = 0
    precision_zero_width_equal: int = 0
    total_zero_width_equal: int = 0

    @property
    def precision_passes(self) -> bool:
        return _coverage_passes(self.precision_checks, self.precision_covered, self.precision_z_rms)

    @property
    def total_passes(self) -> bool:
        return _coverage_passes(self.total_checks, self.total_covered, self.total_z_rms)


def _coverage_passes(checks: int, covered: int, rms: float) -> bool:
    """I-111's predeclared diagnostic criteria, unchanged; no checks is not a pass."""
    return checks > 0 and covered / checks >= .90 and .75 <= rms <= 1.33


@dataclass(frozen=True)
class AttributionRow:
    """The P&L attribution statistics of one market-risk case against float64, per seed."""
    case: str
    paths: int
    spearman: Tuple[float, ...]
    kolmogorov_smirnov: Tuple[float, ...]


@dataclass(frozen=True)
class ExerciseRow:
    """The exercise decisions one case flips on one option at one path count: per simulation
    date, the share of the paths whose exercise state (exercised on or before the date, ORE's
    `OptionWrapper`) differs from the float64 run's on the same paths, its mean over the seeds
    (`flipped`) and its largest (`worst`)."""
    case: str
    paths: int
    trade: str
    product: str
    flipped: Tuple[float, ...]
    worst: Tuple[float, ...]


@dataclass(frozen=True)
class CaseRun:
    """XLA programs compiled by a case's runs after its first seed (0 when every shape is
    compiled once), and the median wall time of those warm runs."""
    case: str
    paths: int
    recompiles: int
    seconds: float


@dataclass
class Measurements:
    """A campaign's measurements (see the module docstring); `merge` joins path counts.

    market: the campaign's `market_name`. cases: each case's policy, in its wire form
        (`PrecisionSchema`); a stochastic rounding's seed is each run's Sobol seed. dates: the
        entries of a profile figure; a scalar figure (`scalars`) has one. products: each
        trade's product. settings: each product's pricing settings that change its kernel's
        arithmetic, as the campaign's analytic ran them (`campaign.settings()`, the evidence
        table's key, I-114). environment: what they ran on (`environment()`)."""
    campaign: str
    kind: str
    description: str
    market: str
    cases: Dict[str, dict]
    seeds: Tuple[int, ...]
    dates: Tuple[str, ...]
    scalars: Tuple[str, ...]
    products: Dict[str, str]
    settings: Dict[str, str]
    environment: Dict[str, str]
    figures: List[FigureRow] = field(default_factory=list)
    coverage: List[CoverageRow] = field(default_factory=list)
    attribution: List[AttributionRow] = field(default_factory=list)
    runs: List[CaseRun] = field(default_factory=list)
    exercise: List[ExerciseRow] = field(default_factory=list)
    replicates: int = 1

    @property
    def paths(self) -> Tuple[int, ...]:
        return tuple(sorted({r.paths for r in self.runs}))

    def merge(self, other: "Measurements") -> "Measurements":
        """These measurements and `other`'s (the same campaign at other path counts)."""
        if (other.campaign, other.market, other.cases, other.seeds, other.replicates, other.settings) != (
                self.campaign, self.market, self.cases, self.seeds, self.replicates, self.settings):
            raise ValueError("only measurements of one campaign, its cases, seeds and pricing settings merge")
        if self.products != other.products or self.dates != other.dates or any(
                self.environment.get(k, "") != other.environment.get(k, "")
                for k in ("engine", "jax", "backend", "devices", "product_emulation", "xla_flags")):
            raise ValueError("measurement implementations, devices, products or dates differ")
        overlap = set(self.paths) & set(other.paths)
        if overlap:
            raise ValueError(f"path counts {sorted(overlap)} measured twice")
        return dataclasses.replace(self, figures=self.figures + other.figures, coverage=self.coverage + other.coverage,
                                   attribution=self.attribution + other.attribution, runs=self.runs + other.runs,
                                   exercise=self.exercise + other.exercise)

    def row(self, case: str, paths: int, figure: str) -> FigureRow:
        """The row of one figure of one case at one path count."""
        for r in self.figures:
            if (r.case, r.paths, r.figure) == (case, paths, figure):
                return r
        raise KeyError((case, paths, figure))

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), separators=(",", ":"))

    def save(self, path: Path) -> None:
        """Every number to `path` as gzip-compressed JSON (no timestamp in the header, so the
        same measurements give the same bytes)."""
        Path(path).write_bytes(gzip.compress(self.to_json().encode("utf-8"), compresslevel=9, mtime=0))

    @classmethod
    def load(cls, path: Path) -> "Measurements":
        return cls.from_json(gzip.decompress(Path(path).read_bytes()).decode("utf-8"))

    @classmethod
    def from_json(cls, text: str) -> "Measurements":
        data = json.loads(text)
        rows = {"figures": FigureRow, "coverage": CoverageRow, "attribution": AttributionRow, "runs": CaseRun,
                "exercise": ExerciseRow}
        for name, row in rows.items():
            # A campaign measured before the exercise rows (2026-10-13) has none.
            data[name] = [row(**{k: tuple(v) if isinstance(v, list) else v for k, v in r.items()})
                          for r in data.get(name, [])]
        for name in ("seeds", "dates", "scalars"):
            data[name] = tuple(data[name])
        return cls(**data)


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------
def measure(campaign, paths: int, cases: Optional[Sequence[Case]] = None,
            progress: Callable[[str], None] = lambda line: None) -> Measurements:
    """Every case of `campaign` (or `cases`, a subset) at `paths` paths, over the campaign's
    seeds, against the float64 run on the same paths (see the module docstring)."""
    cases = tuple(campaign.cases if cases is None else cases)
    seeds = tuple(campaign.seeds)
    if len(seeds) < 2:
        raise ValueError("the statistical error of a figure is its spread over seeds: give at least two")
    if len({c.label for c in cases}) != len(cases):
        raise ValueError("case labels name rows: they must be unique")
    if len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2 ** 32 for s in seeds):
        raise ValueError("independent seed identities must be distinct uint32 values")
    if not cases:
        raise ValueError("at least one precision case is required")
    from types import SimpleNamespace

    # Retain only per-seed summaries. Holding every seed's scenarios and complete reference/
    # candidate cubes simultaneously made the large-count experiment scale in memory with
    # both path count AND seed count. Each seed now needs one reference and one candidate.
    reference_figures = {}
    values = {case.label: {} for case in cases}
    reports = {case.label: {} for case in cases}
    attribution = {case.label: [] for case in cases}
    exercise = {case.label: [] for case in cases}
    recompiles = {case.label: 0 for case in cases}
    warm = {case.label: [] for case in cases}
    started = time.perf_counter()
    for i, seed in enumerate(seeds):
        inputs = campaign.scenarios(paths, seed)
        reference = campaign.run(inputs, Precision())
        reference_figures[seed] = campaign.figures(reference)
        for case in cases:
            precision = case.precision
            if precision.rounding == "stochastic" or any(s.compute_rounding == "stochastic" for s in precision._stages()):
                precision = dataclasses.replace(precision, rounding_seed=seed)
            tick = time.perf_counter()
            if i == 0:
                result = campaign.run(inputs, precision)
            else:
                with count_compiles() as counter:
                    result = campaign.run(inputs, precision)
                recompiles[case.label] += sum(counter.values())
            values[case.label][seed] = campaign.figures(result)
            reports[case.label][seed] = SimpleNamespace(precision=result.precision)
            if i:
                warm[case.label].append(time.perf_counter() - tick)
            row = _attribution_row(campaign, case, paths, {seed: reference}, {seed: result})
            if row is not None:
                attribution[case.label].append(row)
            exercise[case.label].append(_exercise_rows(campaign, case, paths, {seed: reference}, {seed: result}))
            del result
        del reference, inputs
        progress(f"{campaign.name} {paths} paths: seed {seed} ({i + 1}/{len(seeds)}), "
                 f"{time.perf_counter() - started:.1f}s")
    figures = reference_figures[seeds[0]]
    out = Measurements(
        campaign=campaign.name, kind=campaign.kind, description=campaign.description, market=campaign.market_name,
        cases={c.label: PrecisionSchema.from_dataclass(c.precision).model_dump() for c in cases}, seeds=seeds,
        dates=campaign.dates(), scalars=tuple(k for k, v in figures.items() if np.ndim(v) == 0),
        products={t.trade_id: t.product for t in campaign.trades}, settings=campaign.settings(),
        environment=environment(),
        replicates=campaign.simulation.replicates if campaign.kind == "portfolio" else campaign.replicates)
    for case in cases:
        label = case.label
        out.runs.append(CaseRun(label, paths, recompiles[label], statistics.median(warm[label])))
        out.figures.extend(_figure_rows(campaign, case, paths, reference_figures, values[label]))
        out.coverage.extend(_coverage_rows(campaign, case, paths, reference_figures, reports[label]))
        rows = attribution[label]
        if rows:
            out.attribution.append(AttributionRow(label, paths,
                tuple(v for row in rows for v in row.spearman),
                tuple(v for row in rows for v in row.kolmogorov_smirnov)))
        for group in zip(*exercise[label]):
            shares = np.asarray([r.flipped for r in group])
            out.exercise.append(dataclasses.replace(group[0], flipped=_floats(shares.mean(axis=0)),
                                                     worst=_floats(shares.max(axis=0))))
    return out


def _product(campaign, figure: str) -> str:
    """The product a figure belongs to: its trade's, else its scope ("netting_set", "portfolio")."""
    scope, *rest = figure.split("/")
    if scope != "trades":
        return scope
    return {t.trade_id: t.product for t in campaign.trades}[rest[0]]


def _figure_rows(campaign, case: Case, paths: int, reference: Mapping[int, Mapping[str, np.ndarray]],
                 results: Mapping[int, Mapping[str, np.ndarray]]) -> List[FigureRow]:
    seeds = sorted(reference)
    rows = []
    for figure in reference[seeds[0]]:
        ref = np.stack([reference[s][figure] for s in seeds])
        error = np.stack([results[s][figure] for s in seeds]) - ref
        rows.append(FigureRow(
            case=case.label, paths=paths, figure=figure, product=_product(campaign, figure),
            level=_floats(ref.mean(axis=0)), statistical_error=_floats(ref.std(axis=0, ddof=1)),
            bias=_floats(error.mean(axis=0)), bias_error=_floats(error.std(axis=0, ddof=1) / math.sqrt(len(seeds))),
            rms_error=_floats(np.sqrt(np.mean(error ** 2, axis=0))), max_error=float(np.max(np.abs(error)))))
    return rows


def _coverage_rows(campaign, case: Case, paths: int, reference: Mapping[int, Mapping[str, np.ndarray]],
                   results: Mapping[int, object]) -> List[CoverageRow]:
    """Coverage of every two-level estimate of a paired case (none without a paired sample)."""
    seeds = sorted(results)
    estimates = {s: {k: v for k, v in results[s].precision.figures.items() if isinstance(v, MeanEstimate)}
                 for s in seeds}
    rows = []
    for figure in estimates[seeds[0]]:
        ref = np.stack([reference[s][figure] for s in seeds])                      # [m, D] float64 runs
        corrections = np.stack([ref[i] - np.asarray(estimates[s][figure].uncorrected)
                                 for i, s in enumerate(seeds)])
        precision_errors, total_errors, precision_errors_se, total_errors_se = [], [], [], []
        for i, s in enumerate(seeds):
            e = estimates[s][figure]
            value = np.asarray(e.value, dtype=np.float64)
            replicates = results[s].precision.replicates
            # The reported correction SE estimates error against the expected correction,
            # not against a dependent full-draw correction. Estimate that expectation on
            # independent outer seeds, including its own uncertainty in the comparison.
            other_corrections = np.delete(corrections, i, axis=0)
            # Two outer seeds suffice for matched-draw precision errors, but not
            # the independent reference variance of a coverage comparison.
            correction_variance = (other_corrections.var(axis=0, ddof=1) / len(other_corrections)
                                   if len(seeds) >= 3 else np.full(ref.shape[1:], np.nan))
            precision_se = np.sqrt(np.asarray(e.correction_standard_error) ** 2 + correction_variance)
            others = np.delete(ref, i, axis=0)
            reference_variance = (others.var(axis=0, ddof=1) / len(others)
                                  if len(seeds) >= 3 else np.full(ref.shape[1:], np.nan))
            total_se = np.sqrt(np.asarray(e.standard_error) ** 2 + reference_variance)
            precision_errors.append(np.asarray(e.correction) - other_corrections.mean(axis=0))
            total_errors.append(value - others.mean(axis=0))
            precision_errors_se.append(precision_se)
            total_errors_se.append(total_se)
        critical = stats.t.ppf(.975, replicates - 1) if replicates > 1 else Z95
        scale = float(np.max(np.abs(ref)))
        matrices = [np.asarray(a).reshape(len(seeds), -1) for a in
                    (precision_errors, total_errors, precision_errors_se, total_errors_se)]
        entries = campaign.dates() if ref.ndim > 1 else ("scalar",)
        for entry, pe, te, ps, ts in zip(entries, *(a.T for a in matrices)):
            precision_z = standardized_errors(pe, ps, scale)
            total_z = standardized_errors(te, ts, scale)
            rows.append(CoverageRow(
                case=case.label, paths=paths, figure=figure, product=_product(campaign, figure), entry=entry,
                precision_checks=int(precision_z.size), precision_covered=int(np.sum(np.abs(precision_z) <= critical)),
                precision_z_rms=_rms(precision_z), total_checks=int(total_z.size),
                total_covered=int(np.sum(np.abs(total_z) <= critical)), total_z_rms=_rms(total_z),
                interval_status=results[seeds[0]].precision.interval.status, interval_unavailable=len(seeds),
                precision_zero_width_equal=int(np.sum((ps == 0) & (np.abs(pe) <= _roundoff(scale)))),
                total_zero_width_equal=int(np.sum((ts == 0) & (np.abs(te) <= _roundoff(scale))))))
    return rows


def _roundoff(scale: float) -> float:
    # Numerical equality allowance only, not a statistical threshold or tail bound.
    return 32 * np.finfo(np.float64).eps * abs(scale)


def standardized_errors(error: np.ndarray, standard_error: np.ndarray, scale: float = 0.) -> np.ndarray:
    """Diagnostic z scores including zero-width misses (I-111).

    NaN/nonfinite observations are unavailable; zero-width numerical equalities are
    uninformative, not successful intervals. No empirical equality certifies a known
    constant. A nonzero residual beyond roundoff with zero SE is an infinite miss.
    """
    error, standard_error = np.broadcast_arrays(error, standard_error)
    keep = np.isfinite(error) & np.isfinite(standard_error) & (standard_error >= 0)
    keep &= (standard_error > 0) | (np.abs(error) > _roundoff(scale))
    error, standard_error = error[keep], standard_error[keep]
    z = np.copysign(np.full(error.shape, np.inf), error)
    return np.divide(error, standard_error, out=z, where=standard_error > 0)


def _attribution_row(campaign, case: Case, paths: int, reference, results) -> Optional[AttributionRow]:
    seeds = sorted(results)
    pnl = {s: campaign.pnl(results[s]) for s in seeds}
    if pnl[seeds[0]] is None:
        return None
    ref = {s: campaign.pnl(reference[s]) for s in seeds}
    return AttributionRow(
        case=case.label, paths=paths,
        spearman=tuple(float(stats.spearmanr(pnl[s], ref[s]).statistic) for s in seeds),
        kolmogorov_smirnov=tuple(float(stats.ks_2samp(pnl[s], ref[s]).statistic) for s in seeds))


def _exercise_rows(campaign, case: Case, paths: int, reference, results) -> List[ExerciseRow]:
    """Per option, the share of its paths whose exercise state the case flips, per date (none
    without an option)."""
    seeds = sorted(results)
    exact = {s: campaign.decisions(reference[s]) for s in seeds}
    found = {s: campaign.decisions(results[s]) for s in seeds}
    rows = []
    for trade in exact[seeds[0]]:
        shares = np.stack([np.mean(found[s][trade] != exact[s][trade], axis=0) for s in seeds])   # [seeds, dates]
        rows.append(ExerciseRow(case=case.label, paths=paths, trade=trade,
                                product=_product(campaign, f"trades/{trade}"), flipped=_floats(shares.mean(axis=0)),
                                worst=_floats(shares.max(axis=0))))
    return rows


def _floats(a) -> Tuple[float, ...]:
    return tuple(float(x) for x in np.atleast_1d(a))


def _rms(z: np.ndarray) -> float:
    return float(np.sqrt(np.mean(z ** 2))) if z.size else math.nan


def _worst_ratio(numerator: Sequence[float], denominator: Sequence[float]) -> float:
    worst = 0.0
    for n, d in zip(numerator, denominator):
        n = abs(n)
        if math.isnan(n) or math.isnan(d):
            return math.nan
        worst = max(worst, n / d if d > 0 else (math.inf if n > 0 else 0.0))
    return worst


def environment() -> Dict[str, str]:
    """What the measurements ran on: the commit (and whether the tree had changes), the engine's
    source digest (`engine_digest`: exact even when the tree had changes), JAX, the backend and
    devices, the platform, and when."""
    def git(*args) -> str:
        try:
            return subprocess.run(["git", *args], cwd=_REPOSITORY, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    # Runtime code is engine/ plus the installed ORE library, not reference submodules'
    # working files. Preserve changed gitlinks but avoid scanning their large file trees
    # for every timing point (particularly costly on a WSL mounted checkout).
    return {"commit": git("rev-parse", "HEAD"),
            "changes": "yes" if git("status", "--porcelain", "--ignore-submodules=dirty") else "no",
            "engine": engine_digest(),
            "jax": jax.__version__, "backend": jax.default_backend(),
            "devices": ", ".join(sorted({d.device_kind for d in jax.devices()})),
            "product_emulation": os.environ.get("JAX_RISK_EMULATE_PRODUCTS", ""),
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "platform": platform.platform(), "python": platform.python_version(),
            "measured": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}


def engine_digest() -> str:
    """The first 16 hex digits of a SHA-256 over the engine package's Python sources (each
    file's path and its bytes with line endings as LF, in path order): the code that priced a
    measurement, whatever the commit and the checkout's line endings."""
    digest = hashlib.sha256()
    root = _REPOSITORY / "engine"
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()[:16]


# ---------------------------------------------------------------------------
# The summary
# ---------------------------------------------------------------------------
def summary(m: Measurements) -> str:
    """A markdown summary: per path count and case, the worst relative error over the figures
    and where it is, by kind of figure, the worst relative bias and bias significance, the
    coverage of paired cases, recompiles and warm seconds; then the cases that differ in one
    setting only (the block layout, the rounding) side by side, and the P&L attribution
    statistics of market-risk cases."""
    lines = [f"# Precision measurements: {m.campaign}", "", m.description, "",
             f"Kind: {m.kind}. Market: {m.market}. Seeds: {len(m.seeds)} ({m.seeds[0]}..{m.seeds[-1]}). "
             f"Commit {m.environment.get('commit', '?')[:12]} (local changes: {m.environment.get('changes', '?')}; "
             f"engine digest {m.environment.get('engine', '?')}), "
             f"JAX {m.environment.get('jax', '?')} on {m.environment.get('backend', '?')} "
             f"({m.environment.get('devices', '?')}), {m.environment.get('measured', '?')}.", "",
             "Pricing settings (the evidence table's key, I-114): "
             + "; ".join(f"`{product}` {value}" for product, value in sorted(m.settings.items())) + ".", "",
             "Each figure's precision error is its value less the float64 run's on the same paths, per seed. "
             "Relative error: its root mean square over the seeds in units of the figure's statistical error (the "
             "float64 figure's spread over the seeds). Relative bias: its mean over the seeds in the same units. "
             "Significance: that mean in units of its own standard error. Each the worst over the figures and their "
             "dates. Entries float64 holds fixed (a statistical error below 1e-9 of the figure's largest value: a "
             "netting set's ENE where every path is in the money, an EEE_B at its t=0 value) are left out of the "
             "ratios; Fixed moved is the largest error on them relative to the figure's largest value. Generated "
             "by `tests/support/precision_harness.py`; the module docstring defines every quantity.", "",
             "Coverage below is diagnostic Student t coverage, not available confidence bounds. "
             "Current runs report nominal intervals unavailable for every figure and entry (I-111). "
             "The per-entry audit retains the predeclared >=0.90 coverage and 0.75–1.33 standardized-error "
             "RMS criteria; unavailable or uninformative diagnostics never count as passes. A diagnostic pass "
             "does not make a nominal interval available. Historical rows "
             "without an entry label pooled dates and omitted zero-width misses.", ""]
    for paths in m.paths:
        lines += [f"## {paths} paths", "",
                  "| Case | Relative error | Where | Means | Quantiles | Relative bias | Significance | Fixed moved | "
                  "Coverage, precision / total | Recompiles | Warm s |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for case in m.cases:
            rows = [r for r in m.figures if r.case == case and r.paths == paths]
            if not rows:
                continue
            worst = max(rows, key=lambda r: _sortable(r.relative_error))
            by_kind = {kind: _worst(rows, "relative_error", lambda r: _is_quantile(r.figure) == kind)
                       for kind in (False, True)}
            run = next(r for r in m.runs if r.case == case and r.paths == paths)
            lines.append(f"| {case} | {_fmt(worst.relative_error)} | {worst.figure} | {_fmt(by_kind[False])} | "
                         f"{_fmt(by_kind[True])} | {_fmt(_worst(rows, 'relative_bias'))} | "
                         f"{_fmt(_worst(rows, 'significance'))} | {_fmt(_worst(rows, 'fixed_error'))} | "
                         f"{_coverage(m, case, paths)} | {run.recompiles} | "
                         f"{run.seconds:.2f} |")
        lines.append("")
        audited = [r for r in m.coverage if r.paths == paths and r.entry]
        if audited:
            lines += [f"### Diagnostic intervals by entry, {paths} paths", "",
                      "Zero-width residuals beyond 32 float64 epsilons of the figure scale are infinite misses. "
                      "Zero-width numerical equalities are counted separately and do not certify constants. "
                      "No nominal production interval is eligible; the unavailable column counts outer seeds.", "",
                      "| Case | Figure | Entry | Precision covered/checks | Precision RMS | Total covered/checks | "
                      "Total RMS | Diagnostic criteria, precision / total | Zero-width equal, precision / total | "
                      "Nominal intervals unavailable |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            for r in audited:
                verdict = lambda passed: "pass" if passed else "fail/uninformative"  # noqa: E731
                lines.append(f"| {r.case} | {r.figure} | {r.entry} | {r.precision_covered}/{r.precision_checks} | "
                             f"{_fmt(r.precision_z_rms)} | {r.total_covered}/{r.total_checks} | {_fmt(r.total_z_rms)} | "
                             f"{verdict(r.precision_passes)} / {verdict(r.total_passes)} | "
                             f"{r.precision_zero_width_equal} / {r.total_zero_width_equal} | "
                             f"{r.interval_unavailable} |")
            lines.append("")
        for setting in ("offset", "rounding"):
            pairs = _pairs(m, setting)
            if not pairs:
                continue
            values = sorted({v for a, b in pairs for v in (m.cases[a][setting], m.cases[b][setting])})
            lines += [f"### By {setting}, {paths} paths", "",
                      f"Worst relative error, and in brackets the worst bias significance, of cases that differ in "
                      f"`{setting}` only.", "",
                      "| Case | " + " | ".join(f"`{v}`" for v in values) + " |",
                      "|---|" + "---|" * len(values)]
            for a, b in pairs:
                cells = {}
                for c in (a, b):
                    rows = [r for r in m.figures if r.case == c and r.paths == paths]
                    cells[m.cases[c][setting]] = (f"{_fmt(_worst(rows, 'relative_error'))} "
                                                  f"({_fmt(_worst(rows, 'significance'))})")
                lines.append(f"| {_without(a, m.cases[a][setting])} | "
                             + " | ".join(cells.get(v, "") for v in values) + " |")
            lines.append("")
        lines += _exercise_table(m, paths)
        attribution = [r for r in m.attribution if r.paths == paths]
        if attribution:
            lines += [f"### P&L attribution, {paths} scenarios", "",
                      "Against the float64 P&L on the same scenarios, over the seeds (A-11's statistics; the "
                      "thresholds are the evidence table's).", "",
                      "| Case | Spearman, smallest | Kolmogorov-Smirnov, largest |", "|---|---|---|"]
            lines += [f"| {r.case} | {min(r.spearman):.6f} | {max(r.kolmogorov_smirnov):.6f} |" for r in attribution]
            lines.append("")
    return "\n".join(lines)


def _exercise_table(m: Measurements, paths: int) -> List[str]:
    """The exercise decisions each case flips per option at `paths` (none without an option)."""
    rows = [r for r in m.exercise if r.paths == paths]
    if not rows:
        return []
    trades = list(dict.fromkeys(r.trade for r in rows))
    lines = [f"### Exercise decisions flipped, {paths} paths", "",
             "Per option, the share of its paths whose exercise state on a date (exercised on or before it) differs "
             "from float64's on the same paths: the worst date's mean over the seeds, and in brackets the largest at "
             "any seed and date. A cash-settled option's flipped exercise moves it by its whole value (decision A-24).",
             "", "| Case | " + " | ".join(f"`{t}`" for t in trades) + " |", "|---|" + "---|" * len(trades)]
    for case in m.cases:
        mine = {r.trade: r for r in rows if r.case == case}
        if mine:
            lines.append(f"| {case} | " + " | ".join(
                f"{max(mine[t].flipped):.2%} ({max(mine[t].worst):.2%})" if t in mine else "" for t in trades) + " |")
    return lines + [""]


def _worst(rows, ratio: str, keep=lambda row: True) -> float:
    """The largest of `ratio` (a `FigureRow` property) over the rows `keep` keeps."""
    return max((getattr(r, ratio) for r in rows if keep(r)), default=math.nan, key=_sortable)


def _pairs(m: Measurements, setting: str) -> List[Tuple[str, str]]:
    """Cases whose policies differ in `setting` only, as (first, second) in case order."""
    seen, pairs = {}, []
    for case, policy in m.cases.items():
        key = json.dumps({k: v for k, v in policy.items() if k != setting}, sort_keys=True)
        if key in seen:
            pairs.append((seen[key], case))
        else:
            seen[key] = case
    return pairs


def _without(label: str, value: str) -> str:
    """A case label with one setting's value left out (it is the table's column)."""
    return ":".join(part for part in label.split(":") if part != value)


def _coverage(m: Measurements, case: str, paths: int) -> str:
    """The pooled coverage of a paired case's two intervals, out of how many checks."""
    rows = [r for r in m.coverage if r.case == case and r.paths == paths]
    if not rows:
        return "—"
    rate = lambda covered, checks: f"{covered / checks:.3f}" if checks else "—"  # noqa: E731
    return (f"{rate(sum(r.precision_covered for r in rows), sum(r.precision_checks for r in rows))} / "
            f"{rate(sum(r.total_covered for r in rows), sum(r.total_checks for r in rows))} "
            f"of {sum(r.total_checks for r in rows)}")


def _is_quantile(figure: str) -> bool:
    return figure.split("/")[-1].startswith(("PFE_", "VaR_", "ES_"))


def _sortable(x: float) -> float:
    return -math.inf if math.isnan(x) else x


def _fmt(x: float) -> str:
    return "nan" if math.isnan(x) else "inf" if math.isinf(x) else f"{x:.3g}"
