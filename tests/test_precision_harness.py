"""
The storage harness (`tests/support/precision_harness.py`, I-75; docs/planning/details/
precision.md §15.4) and the closure of I-75 on it.

  * Fast: the case grid; on a tiny portfolio and a tiny market-risk run, every figure is
    measured against the float64 run on the same paths (a float64 case measures exactly zero),
    a case compiles nothing after its first seed, a paired case's coverage is counted, the
    attribution statistics are recorded, and the measurements survive JSON and merge by path
    count.
  * Slow, I-75: rounded to nearest in FP8, a bond's cube column (near 1e6 with a spread near
    1e4) without block offsets is biased, by more than its Monte Carlo standard error and
    detectably at the harness's seeds; with them (the default) its bias is undetectable and
    below its Monte Carlo standard error, at 1,024 and 4,096 paths. The thresholds were fixed
    before the campaigns were measured (precision.md §15.4).
"""
import dataclasses
import gzip
import json
import math

import jax
import numpy as np
import ORE
import pytest

from engine.market_simulation.config import CamConfig, LgmConfig
from engine.precision import OFFSETS, ROUNDINGS, STAGES, Precision, StagePrecision
from engine.pricing.config import LgmSwaptionEngineConfig, PricingConfig
from engine.risk.market.run import run_figures as market_risk_figures
from engine.run.pipeline import run_figures as portfolio_figures
from tests import market_risk_support as mr
from tests.support import portfolio as shared
from tests.support.precision_campaigns import CAMPAIGNS, SIMULATION
from tests.support.precision_harness import (
    SCALED, Case, FigureRow, MarketRiskCampaign, Measurements, PortfolioCampaign, measure, storage_cases, summary,
    throughout_case,
)

FP8 = StagePrecision("float8_e5m2")
DATES = tuple(shared.ASOF + ORE.Period(m, ORE.Months) for m in (3, 12))


def _portfolio(cases, seeds=(1, 2, 3), names=("swap-payer", "bond")) -> PortfolioCampaign:
    return PortfolioCampaign(
        name="tiny", description="Two of the shared portfolio's trades on two dates.", market_name="shared",
        market=shared.market(),
        trades=tuple(shared.trades()[n] for n in names),
        simulation=CamConfig(dates=DATES, base_currency="USD", ir={"USD": LgmConfig(0.03, 0.01, ("1Y",), ("9Y",))}),
        pricing=PricingConfig(), cases=tuple(cases), paths=(64,), seeds=seeds)


@pytest.fixture(scope="module")
def tiny():
    """The tiny portfolio at 64 paths, three seeds: a float64 case, FP8 storage of the cube, and
    the same with a paired sample of half the paths."""
    cases = (Case("float64", Precision()), Case("pricing:fp8", Precision(pricing=FP8)),
             Case("pricing:fp8:paired", Precision(pricing=FP8, paired_fraction=0.5)))
    campaign = _portfolio(cases)
    return campaign, measure(campaign, 64)


class TestCases:
    def test_one_case_per_stage_format_rounding_and_layout(self):
        """A plain format has one case per stage; each case stores one stage, computed in
        float64, and leaves the rest of the policy at its default."""
        cases = storage_cases()
        assert len(cases) == len(STAGES) * (1 + len(SCALED) * len(ROUNDINGS) * len(OFFSETS))
        assert len({c.label for c in cases}) == len(cases)
        for case in cases:
            stage, name, *layout = case.label.split(":")
            assert getattr(case.precision, stage) == StagePrecision(name)
            others = {s: getattr(case.precision, s) for s in STAGES if s != stage}
            assert set(others.values()) == {StagePrecision()}
            if layout:
                assert layout == [case.precision.rounding, case.precision.offset]
        assert storage_cases(stages=("pricing",), formats=("float16",), roundings=("nearest",), offsets=("block",),
                             paired_fraction=0.25)[0] == Case(
            "pricing:float16:nearest:block:paired", Precision(pricing=StagePrecision("float16"), paired_fraction=0.25))
        assert throughout_case("float32").precision == Precision.throughout("float32")

    def test_the_campaigns_of_record_are_well_formed(self):
        """Unique labels; the linear campaign covers every stage x format x rounding x layout
        and pairs every scaled format; seeds enough for a statistical error."""
        for build in CAMPAIGNS.values():
            campaign = build()
            assert len({c.label for c in campaign.cases}) == len(campaign.cases) and len(campaign.seeds) >= 8
        linear = CAMPAIGNS["linear"]()
        assert set(storage_cases()) <= set(linear.cases)
        assert {c.label.split(":")[1] for c in linear.cases if c.precision.paired_fraction} == set(SCALED)
        assert linear.simulation == SIMULATION and linear.paths == (1024, 4096, 16384)

    def test_the_calibrated_market_risk_campaign_differs_in_the_bermudans_model_alone(self):
        """`market_risk_calibrated` measures the kernel the fixed-LGM campaigns do not (I-114):
        the same trades, market, shifts and grid, the Bermudan recalibrated under each scenario,
        and every combination the three market-risk campaigns measure on the fixed LGM."""
        fixed, calibrated = CAMPAIGNS["market_risk"](), CAMPAIGNS["market_risk_calibrated"]()
        assert (calibrated.trades, calibrated.paths, calibrated.seeds) == (fixed.trades, fixed.paths, fixed.seeds)
        assert calibrated.pricing.bermudan.calibrates and not fixed.pricing.bermudan.calibrates
        assert dataclasses.replace(calibrated.pricing.bermudan, calibration="None", volatility=0.0) == \
            dataclasses.replace(fixed.pricing.bermudan, volatility=0.0)
        differ = {p for p, s in calibrated.settings().items() if fixed.settings()[p] != s}
        assert differ == {"bermudan_swaption"} and calibrated.settings()["bermudan_swaption"].endswith(
            "model=recalibrated")
        measured = {c.precision for name in ("market_risk", "compute_market_risk", "rounding_market_risk")
                    for c in CAMPAIGNS[name]().cases}
        assert measured == {c.precision for c in calibrated.cases}


class TestMeasure:
    def test_a_float64_case_measures_exactly_zero(self, tiny):
        """The reference is the float64 run on the same paths: a case at float64 differs from
        it nowhere, so its every error is exactly 0 and its statistical error is the spread over
        the seeds."""
        _, m = tiny
        rows = [r for r in m.figures if r.case == "float64"]
        assert rows and all(r.max_error == 0 and set(r.bias) == {0.0} and set(r.rms_error) == {0.0} for r in rows)
        assert all(r.relative_error == 0 for r in rows)
        npv = m.row("float64", 64, "trades/bond/NPV")
        assert len(npv.level) == len(DATES) and all(e > 0 for e in npv.statistical_error)

    def test_every_figure_of_every_scope_is_measured(self, tiny):
        campaign, m = tiny
        figures = {r.figure for r in m.figures if r.case == "pricing:fp8"}
        for scope in ["netting_set"] + [f"trades/{t.trade_id}" for t in campaign.trades]:
            for name in ("NPV", "EPE", "ENE", "EE_B", "EEE_B", "EPE_B", "EEPE_B", "PFE_95", "Basel_EPE", "Basel_EEPE"):
                assert f"{scope}/{name}" in figures
        assert set(m.scalars) == {f for f in figures if f.endswith(("Basel_EPE", "Basel_EEPE"))}
        assert m.dates == tuple(d.ISO() for d in DATES) and m.products == {"swap-payer": "swap", "bond": "bond"}
        assert m.settings == {"bond": "engine=Discounting", "swap": "engine=Discounting"}  # the evidence's key, I-114
        assert m.row("pricing:fp8", 64, "trades/bond/NPV").product == "bond"
        assert m.row("pricing:fp8", 64, "netting_set/EPE").product == "netting_set"

    def test_the_figures_measured_are_those_the_engine_warns_of(self, tiny):
        """The evidence table's rows are the harness's figures, and a run's warnings name the
        engine's (`run_figures`, I-55): the two lists are one."""
        campaign, m = tiny
        names = portfolio_figures(campaign.pfe_quantiles)
        kinds = {scope: {r.figure.rsplit("/", 1)[1] for r in m.figures if r.figure.startswith(scope + "/")}
                 for scope in ["netting_set"] + [f"trades/{t.trade_id}" for t in campaign.trades]}
        assert kinds["netting_set"] == set(names.total) and names.aggregate == "netting_set"
        assert all(found == set(names.trade) for scope, found in kinds.items() if scope.startswith("trades/"))

    def test_fp8_storage_moves_the_figures_and_the_bias_is_the_mean_error(self, tiny):
        campaign, m = tiny
        row = m.row("pricing:fp8", 64, "trades/bond/NPV")
        assert row.max_error > 0 and all(e > 0 for e in row.rms_error)
        assert all(abs(b) <= r for b, r in zip(row.bias, row.rms_error))
        assert row.relative_error == max(r / s for r, s in zip(row.rms_error, row.statistical_error))

    def test_a_case_compiles_nothing_after_its_first_seed(self, tiny):
        _, m = tiny
        assert [r.recompiles for r in m.runs] == [0, 0, 0] and all(r.seconds > 0 for r in m.runs)

    def test_a_paired_case_counts_its_coverage(self, tiny):
        """Each two-level estimate (EPE, ENE of the netting set and each trade) is checked on
        every seed and date where its interval has a width; an unpaired case has none."""
        campaign, m = tiny
        rows = [r for r in m.coverage if r.case == "pricing:fp8:paired"]
        assert {r.figure for r in rows} == {f"{s}/{n}" for s in ["netting_set", "trades/swap-payer", "trades/bond"]
                                            for n in ("EPE", "ENE")}
        assert sum(r.total_checks for r in rows) == 0  # one scramble: uncertainty is unavailable
        assert {r.entry for r in rows} == {d.ISO() for d in DATES}
        assert all(r.interval_status == "unavailable" and r.interval_unavailable == len(campaign.seeds) for r in rows)
        assert not any(r.total_passes or r.precision_passes for r in rows)
        for r in rows:
            assert 0 <= r.precision_covered <= r.precision_checks <= len(campaign.seeds) * len(DATES)
            assert 0 <= r.total_covered <= r.total_checks <= len(campaign.seeds) * len(DATES)
        assert not [r for r in m.coverage if r.case != "pricing:fp8:paired"]

    def test_the_measurements_survive_json_and_merge_by_path_count(self, tiny, tmp_path):
        campaign, m = tiny
        again = Measurements.from_json(m.to_json())
        path = tmp_path / "m.json.gz"
        m.save(path)
        assert gzip.decompress(path.read_bytes()).decode() == m.to_json() == Measurements.load(path).to_json()
        assert again.to_json() == m.to_json() and again.figures[0] == m.figures[0]  # NaN != NaN: compare the text
        json.loads(m.to_json())
        other = measure(campaign, 32, cases=campaign.cases[:1])
        with pytest.raises(ValueError, match="only measurements of one campaign"):
            m.merge(other)
        other = dataclasses.replace(other, cases=m.cases)
        with pytest.raises(ValueError, match="only measurements of one campaign"):
            m.merge(dataclasses.replace(other, settings={**other.settings, "bond": "engine=Other"}))
        merged = m.merge(other)
        assert merged.paths == (32, 64) and len(merged.figures) == len(m.figures) + len(other.figures)
        with pytest.raises(ValueError, match="measured twice"):
            merged.merge(other)

    @pytest.mark.parametrize("seeds", [(1, 2), (1, 2, 3, 4)])
    def test_replicated_diagnostics_are_per_date_and_never_certify_empirical_constants(self, seeds):
        """On the CPU, whose paired float64 sample is the full run's bit for bit: on a GPU a prefix of
        the paths is an ulp from them (the batch shape), which is not what this tests."""
        import warnings
        campaign = _portfolio((Case("paired", Precision(paired_fraction=.5)),), seeds=seeds, names=("bond",))
        campaign = dataclasses.replace(campaign, simulation=dataclasses.replace(campaign.simulation, replicates=8))
        with warnings.catch_warnings(), jax.default_device(jax.devices("cpu")[0]):
            warnings.simplefilter("error", RuntimeWarning)
            measurements = measure(campaign, 256)
        rows = measurements.coverage
        assert len(rows) == 8  # two dates, two figures, trade and netting set
        assert all(r.interval_unavailable == len(seeds) and r.interval_status == "unavailable" for r in rows)
        assert not any(r.precision_passes for r in rows)  # paired float64 is identically equal
        assert all(r.precision_checks == 0 for r in rows)
        if len(seeds) >= 3:
            assert all(r.precision_zero_width_equal == len(seeds) for r in rows)
            assert any(r.total_checks == len(seeds) for r in rows)
        else:
            assert all(r.total_checks == 0 for r in rows)  # reference variance unavailable
        text = summary(measurements)
        assert "Diagnostic intervals by entry" in text and "unavailable" in text
        assert Measurements.from_json(measurements.to_json()).to_json() == measurements.to_json()

    def test_the_summary_names_every_case_and_compares_layouts(self, tiny):
        campaign = _portfolio((Case("block", Precision(pricing=FP8)),
                               Case("none", Precision(pricing=FP8, offset="none"))), seeds=(1, 2))
        text = summary(measure(campaign, 64))
        assert "## 64 paths" in text and "| block |" in text and "| none |" in text
        assert "### By offset, 64 paths" in text

    def test_each_options_flipped_exercise_decisions_are_measured(self):
        """Decision A-24: per option and date, the share of its paths whose exercise state a
        case flips against float64's on the same paths, its mean over the seeds and its largest;
        none at float64, none for a swap; the rows survive JSON and the summary shows them."""
        names = ("swap-payer", "european-receiver-otm-cash", "bermudan-receiver-cash")
        cases = (Case("float64", Precision()), Case("market:fp8", Precision(market=FP8)))
        coarse = LgmSwaptionEngineConfig(n_per_std=8, std_devs=3.0)
        dates = tuple(shared.ASOF + ORE.Period(m, ORE.Months) for m in (12, 24, 36, 48))  # the options' exercises
        campaign = _portfolio(cases, names=names)
        campaign = dataclasses.replace(campaign, pricing=PricingConfig(bermudan=coarse),
                                       simulation=dataclasses.replace(campaign.simulation, dates=dates))
        m = measure(campaign, 64)
        rows = {(r.case, r.trade): r for r in m.exercise}
        assert set(rows) == {(c.label, t) for c in cases for t in names[1:]}
        assert all(r.paths == 64 and len(r.flipped) == len(r.worst) == len(dates) for r in rows.values())
        assert all(set(rows["float64", t].flipped) == {0.0} for t in names[1:])
        fp8 = [rows["market:fp8", t] for t in names[1:]]
        assert all(0.0 <= f <= w <= 1.0 for r in fp8 for f, w in zip(r.flipped, r.worst))
        assert any(w > 0.0 for r in fp8 for w in r.worst)
        assert rows["market:fp8", "bermudan-receiver-cash"].product == "bermudan_swaption"
        assert Measurements.from_json(m.to_json()).exercise == m.exercise
        assert "### Exercise decisions flipped, 64 paths" in summary(m)

    def test_misuse_is_refused(self):
        with pytest.raises(ValueError, match="at least two"):
            measure(_portfolio([Case("a", Precision())], seeds=(1,)), 64)
        with pytest.raises(ValueError, match="unique"):
            measure(_portfolio([Case("a", Precision())] * 2), 64)


class TestRatios:
    @staticmethod
    def _row(level, statistical_error, error, bias_error=None):
        return FigureRow("c", 1, "f", "p", level=level, statistical_error=statistical_error, bias=error,
                         bias_error=bias_error or statistical_error, rms_error=tuple(abs(e) for e in error),
                         max_error=0.0)

    @pytest.mark.parametrize("error, statistical_error, worst", [
        ((0.0, 1.0), (1.0, 4.0), 0.25), ((-2.0, 0.0), (1.0, 1.0), 2.0), ((math.nan, 0.0), (1.0, 1.0), math.nan),
    ])
    def test_the_worst_ratio_over_the_entries(self, error, statistical_error, worst):
        row = self._row((10.0, 10.0), statistical_error, error)
        for value in (row.relative_error, row.relative_bias, row.significance):
            assert value == worst or (math.isnan(value) and math.isnan(worst))
        assert row.fixed == (False, False) and row.fixed_error == 0.0

    def test_an_entry_float64_holds_fixed_is_measured_against_the_scale(self):
        """A netting set's ENE that is 0 on every path at the first date, and an EEE_B whose
        spread is float64's rounding: left out of the ratios, their movement relative to the
        figure's largest value."""
        row = self._row((0.0, 2e5, 1e6), (0.0, 50.0, 1e-10), (3.0, 10.0, 7.0))
        assert row.fixed == (True, False, True) and row.scale == 1e6
        assert row.relative_error == row.relative_bias == 0.2 and row.fixed_error == 7e-6
        assert self._row((0.0,), (0.0,), (0.0,)).fixed_error == 0.0
        assert self._row((0.0,), (0.0,), (1e-9,)).fixed_error == math.inf


class TestMarketRisk:
    def test_market_risk_figures_and_attribution(self):
        """VaR, ES and the mean P&L are measured; each case's P&L carries Spearman's correlation
        and the Kolmogorov-Smirnov statistic against float64's, per seed."""
        campaign = MarketRiskCampaign(
            name="tiny-risk", description="", market_name="market risk", market=mr.market(),
            trades=(mr.swap(), mr.bond()),
            covariance=mr.covariance(), horizon_days=10, pricing=mr.PRICING,
            cases=(Case("float64", Precision()), Case("pricing:fp8", Precision(pricing=FP8))), paths=(256,),
            seeds=(1, 2))
        m = measure(campaign, 256)
        figures = {r.figure for r in m.figures if r.case == "pricing:fp8"}
        assert figures == {"portfolio/VaR_99", "portfolio/ES_99", "portfolio/VaR_97.5", "portfolio/ES_97.5",
                           "portfolio/PnL", "trades/swap/PnL", "trades/bond/PnL"} == set(m.scalars)
        names = market_risk_figures(campaign.quantiles)  # the figures the engine warns of (I-55)
        assert figures == ({f"portfolio/{k}" for k in names.total} | {f"trades/{t.trade_id}/{k}"
                                                                       for t in campaign.trades for k in names.trade})
        exact, fp8 = m.attribution
        assert exact.spearman == pytest.approx((1.0, 1.0), abs=1e-15) and exact.kolmogorov_smirnov == (0.0, 0.0)
        assert all(0.9 < s < 1.0 for s in fp8.spearman) and all(0 <= k < 0.1 for k in fp8.kolmogorov_smirnov)
        assert m.dates == () and not m.coverage


# ---------------------------------------------------------------------------
# I-75, closed: a bond's FP8 cube column is unbiased with block offsets
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("paths", [1024, 4096])
@pytest.mark.parametrize("name", ["float8_e4m3fn", "float8_e5m2"])
def test_a_bonds_fp8_cube_column_is_unbiased_with_block_offsets(name, paths):
    """The thresholds fixed before measuring (precision.md §15.4): with block offsets, at every
    date the bond column's bias is within three of its standard errors of zero (undetectable at
    16 seeds) and below the Monte Carlo standard error of the column's mean (its float64 spread
    over the paths / sqrt(paths)); without them, at some date it is beyond both, as I-75 found.
    The swap shares the netting set, as in the shared portfolio."""
    cases = tuple(Case(o, Precision(pricing=StagePrecision(name), offset=o)) for o in OFFSETS)
    campaign = dataclasses.replace(CAMPAIGNS["linear"](), trades=(shared.trades()["swap-payer"],
                                                                  shared.trades()["bond"]), cases=cases)
    m = measure(campaign, paths)
    reference = campaign.run(campaign.scenarios(paths, campaign.seeds[0]), Precision())
    column = np.asarray(reference.npv_cube)[:, :, 1]
    monte_carlo = column.std(axis=0, ddof=1) / math.sqrt(paths)
    block, none = (m.row(o, paths, "trades/bond/NPV") for o in OFFSETS)
    bias, bias_error = np.abs(block.bias), np.asarray(block.bias_error)
    matured = (bias == 0) & (monte_carlo == 0)  # its last date: the bond has paid out on every path
    assert np.all(bias <= 3 * bias_error) and np.all((bias < monte_carlo) | matured), (bias, bias_error, monte_carlo)
    bias, bias_error = np.abs(none.bias), np.asarray(none.bias_error)
    assert np.any((bias > 3 * bias_error) & (bias > monte_carlo)), (bias, bias_error, monte_carlo)
