"""
The precision evidence table (I-55; docs/planning/details/precision.md §10.3): the acceptance
standard as cited data (A-11), the rows the generator reads off the precision harness's
measurements (`tests/support/precision_evidence.py`), and the warnings the engine gives a
result whose figures have no passing row (`engine.precision.evidence`; D-9: never a refusal).

  * The standard: every rule cites its source, MAR32.42's green zone is strict.
  * The rows: a scope's worst trade decides, the statistical and fixed rules and, for market
    risk, the P&L attribution test all must hold, NaN fails, a ceiling needs every smaller path
    count to pass, a float64 case is the reference.
  * The engine's lookup: combination, paired share, path ceiling, an aggregate's products, a
    mix of pricing stages, and the warning's text.
  * Pricing settings (I-114): the key of each product's kernel, and that rows measured under one
    European engine, one rollback grid or one fixed model validate no other, at the lookup and
    end to end through `price_portfolio` and `run_market_risk` on the shipped table.
  * The files of record are what the measurements give (`--check`), and the engine reads them.
  * End to end: a float64 run warns of nothing, a reduced one names its figures, and the paired
    sample changes exactly the figures the table says it corrects.
"""
import dataclasses
import json
import math

import numpy as np
import ORE
import pytest

from engine.api.requests import PrecisionSchema
from engine.market_simulation.config import CamConfig, LgmConfig
from engine.precision import Precision, StagePrecision, evidence_warnings, paired_error_warning
from engine.precision import evidence as evidence_module
from engine.precision.evidence import TABLE, Evidence, Row, combination, evidence_json, shipped
from engine.pricing.config import JamshidianEngineConfig, LgmSwaptionEngineConfig, PricingConfig
from engine.pricing.cube import PRODUCTS
from engine.risk.counterparty.exposure import CORRECTED_FIGURES
from engine.risk.market.run import pricing_settings as market_risk_settings
from engine.risk.market.run import run_figures as market_risk_figures
from engine.run.pipeline import pricing_settings as portfolio_settings
from engine.run.pipeline import run_figures as portfolio_figures
from tests.support import portfolio as shared
from tests.support import precision_campaigns as campaigns
from tests.support import precision_evidence as generator
from tests.support.precision_campaigns import CAMPAIGNS
from tests.support.precision_harness import AttributionRow, CaseRun, FigureRow, Measurements, PortfolioCampaign

F16 = StagePrecision("float16")
F32 = StagePrecision("float32", "float32", "float32")
PORTFOLIO = portfolio_figures((0.95,))
MARKET_RISK = market_risk_figures((0.99,))

#: Pricing settings of the products the lookup tests use.
SETTINGS = {"swap": "engine=Discounting", "bond": "engine=Discounting",
            "bermudan_swaption": "engine=LgmGrid nx=8 sx=3 model=fixed"}


def _settings(*products):
    """The sorted (product, settings) pairs of a row whose campaign held `products`."""
    return tuple(sorted((p, SETTINGS[p]) for p in products))


@dataclasses.dataclass(frozen=True)
class Trade:
    trade_id: str
    product: str


# ---------------------------------------------------------------------------
# The standard
# ---------------------------------------------------------------------------
class TestStandard:
    def test_the_standard_of_record_cites_every_rule(self):
        standard = generator.Standard.load()
        assert (standard.statistical, standard.fixed, standard.spearman_above, standard.ks_below) == (
            0.01, 1e-6, 0.8, 0.09)
        assert "MAR32.42" in standard.text["pla"]["cite"] and "P6.2" in standard.text["statistical"]["cite"]
        assert "bis.org" in standard.text["pla"]["source"]

    def test_a_rule_without_a_citation_is_refused(self, tmp_path):
        data = json.loads(generator.ACCEPTANCE.read_text(encoding="utf-8"))
        data["fixed"]["cite"] = ""
        path = tmp_path / "acceptance.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="'fixed' rule has no citation"):
            generator.Standard.load(path)

    @pytest.mark.parametrize("spearman, ks, green", [
        ((0.81, 0.99), (0.01, 0.08), True), ((0.80,), (0.01,), False), ((0.9,), (0.09,), False),
        ((math.nan,), (0.01,), False), ((0.9,), (math.nan,), False), ((0.9, 0.79), (0.0, 0.0), False),
    ])
    def test_the_green_zone_is_strict_at_every_seed(self, spearman, ks, green):
        """MAR32.42(1): the correlation *above* 0.80 and the statistic *below* 0.09."""
        assert generator.Standard.load().green(spearman, ks) is green


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------
def _figure(case, paths, figure, product, relative, fixed=None):
    """A harness row whose varying entry has `relative` statistical errors of root mean square
    error, and, with `fixed`, an entry float64 holds fixed moved by `fixed` of the scale (1)."""
    level, statistical, rms = [1.0], [1.0], [relative]
    if fixed is not None:
        level, statistical, rms = level + [1.0], statistical + [0.0], rms + [fixed]
    return FigureRow(case, paths, figure, product, level=tuple(level), statistical_error=tuple(statistical),
                     bias=tuple(rms), bias_error=tuple(statistical), rms_error=tuple(rms), max_error=max(rms))


def _measurements(kind, cases, figures, attribution=(), paths=(1024, 4096)):
    products = {"payer": "swap", "receiver": "swap", "bond": "bond"}
    return Measurements(
        campaign="c", kind=kind, description="", market="m",
        cases={label: PrecisionSchema.from_dataclass(p).model_dump() for label, p in cases.items()}, seeds=(1, 2),
        dates=("d",), scalars=(), products=products, settings={"swap": SETTINGS["swap"], "bond": SETTINGS["bond"]},
        environment={"commit": "abc", "changes": "no", "engine": "e"},
        figures=list(figures), attribution=list(attribution),
        runs=[CaseRun(label, n, 0, 1.0) for label in cases for n in paths])


class TestRows:
    def test_a_products_worst_trade_decides_and_its_ceiling_needs_every_smaller_count(self):
        m = _measurements("portfolio", {"f64": Precision(), "c": Precision(pricing=F16)}, [
            _figure("f64", 1024, "trades/payer/EPE", "swap", 0.0),
            _figure("c", 1024, "trades/payer/EPE", "swap", 0.001), _figure("c", 4096, "trades/payer/EPE", "swap", 0.002),
            _figure("c", 1024, "trades/receiver/EPE", "swap", 0.005),
            _figure("c", 4096, "trades/receiver/EPE", "swap", 0.02),
            _figure("c", 1024, "trades/bond/EPE", "bond", 0.02), _figure("c", 4096, "trades/bond/EPE", "bond", 0.001),
            _figure("c", 1024, "netting_set/EPE", "netting_set", 0.0, fixed=2e-6),
            _figure("c", 4096, "netting_set/EPE", "netting_set", 0.0, fixed=1e-7),
        ])
        rows = {(r.scope, r.figure): r for r in generator.rows_of(m, generator.Standard.load())}
        assert set(rows) == {("swap", "EPE"), ("bond", "EPE"), ("netting_set", "EPE")}  # the float64 case gives none
        swap = rows["swap", "EPE"]
        assert swap.ceiling == 1024 and [p.worst for p in swap.points] == ["trades/receiver/EPE"] * 2
        assert rows["bond", "EPE"].ceiling is None                 # fails at 1,024, so 4,096's pass is not enough
        assert rows["netting_set", "EPE"].ceiling is None          # the fixed rule: 2e-6 of its scale
        assert swap.combination == combination(Precision(pricing=F16), PORTFOLIO.stages, F16)

    def test_nan_fails(self):
        m = _measurements("portfolio", {"c": Precision(pricing=F16)}, [
            _figure("c", 1024, "trades/bond/NPV", "bond", math.nan),
            _figure("c", 1024, "trades/payer/NPV", "swap", 0.0)], paths=(1024,))
        rows = {r.scope: r for r in generator.rows_of(m, generator.Standard.load())}
        assert rows["bond"].ceiling is None and rows["swap"].ceiling == 1024

    def test_market_risk_needs_the_attribution_test_green(self):
        cases = {"green": Precision(pricing=F16), "red": Precision(pricing=StagePrecision("bfloat16"))}
        figures = [_figure(c, 1024, "portfolio/VaR_99", "portfolio", 0.0) for c in cases]
        attribution = [AttributionRow("green", 1024, (0.99, 0.99), (0.01, 0.01)),
                       AttributionRow("red", 1024, (0.99, 0.79), (0.01, 0.01))]
        rows = {r.case: r for r in generator.rows_of(
            _measurements("market_risk", cases, figures, attribution, paths=(1024,)), generator.Standard.load())}
        assert rows["green"].ceiling == 1024 and rows["red"].ceiling is None
        assert rows["green"].combination == "simulation=float64/float64/float64 pricing=float16/float64/float64 " \
                                            "rounding=nearest offset=block"

    def test_only_a_corrected_figure_keeps_the_paired_share(self):
        m = _measurements("portfolio", {"c": Precision(pricing=F16, paired_fraction=0.125)}, [
            _figure("c", 1024, f"trades/bond/{k}", "bond", 0.0) for k in ("EPE", "PFE_95", "NPV")], paths=(1024,))
        rows = {r.figure: r.paired for r in generator.rows_of(m, generator.Standard.load())}
        assert rows == {"EPE": 0.125, "PFE_95": 0.0, "NPV": 0.0}

    def test_the_evidence_file_carries_every_ceiling_back(self):
        m = _measurements("portfolio", {"c": Precision(pricing=F16, paired_fraction=0.125)}, [
            _figure("c", 1024, f"{scope}/{k}", "x", 0.0) for scope in ("trades/bond", "netting_set")
            for k in ("EPE", "NPV")], paths=(1024,))
        _, text = generator.generate([m], generator.Standard.load())
        rows = {(r.scope, r.figure): r for r in Evidence.from_json(text).rows}
        assert rows["bond", "EPE"] == Row("c", "c", combination(Precision(pricing=F16), PORTFOLIO.stages, F16), 0.125,
                                          "bond", "EPE", 1024, _settings("bond", "swap"))
        assert rows["bond", "NPV"].paired == 0.0 and rows["netting_set", "NPV"].settings == _settings("bond", "swap")

    def test_a_campaign_whose_settings_miss_a_product_is_refused(self):
        m = _measurements("portfolio", {"c": Precision(pricing=F16)}, [], paths=(1024,))
        m = dataclasses.replace(m, settings={"swap": SETTINGS["swap"]})
        with pytest.raises(ValueError, match=r"pricing settings for \['swap'\], but its trades are \['bond', 'swap'\]"):
            generator.generate([m], generator.Standard.load())


# ---------------------------------------------------------------------------
# The engine's lookup and warnings
# ---------------------------------------------------------------------------
def _evidence(*rows):
    return Evidence(rows)


KEY16 = combination(Precision(pricing=F16), PORTFOLIO.stages, F16)


def _row(case, scope, figure, ceiling, paired=0.0, campaign="c", products=("bond", "swap")):
    """A row of `campaign`, which held `products` under `SETTINGS`, at `KEY16`."""
    return Row(campaign, case, KEY16, paired, scope, figure, ceiling, _settings(*products))


def _warnings(precision, trades, paths, figures, evidence, settings=SETTINGS):
    return evidence_warnings(precision, trades, paths, figures, evidence, settings=settings)


class TestCombination:
    def test_float64_is_the_reference(self):
        assert combination(Precision(), PORTFOLIO.stages, StagePrecision()) is None
        assert combination(Precision(market=F16), MARKET_RISK.stages, StagePrecision()) is None  # no market stage

    def test_rounding_and_layout_count_only_where_a_stage_stores_scaled(self):
        assert combination(Precision(pricing=F32), PORTFOLIO.stages, F32) == (
            "simulation=float64/float64/float64 market=float64/float64/float64 pricing=float32/float32/float32")
        stochastic = Precision(pricing=F16, rounding="stochastic")
        assert combination(stochastic, PORTFOLIO.stages, F16).endswith("rounding=stochastic offset=block")
        assert combination(stochastic, PORTFOLIO.stages, StagePrecision()) is None  # a float64 trade of that run


class TestLookup:
    def test_a_ceiling_bounds_the_paths_from_above(self):
        evidence = _evidence(_row("x", "bond", "EPE", 4096))
        trades = [Trade("b", "bond")]
        figures = dataclasses.replace(PORTFOLIO, trade=("EPE",), total=())
        assert _warnings(Precision(pricing=F16), trades, 4096, figures, evidence) == []
        assert _warnings(Precision(pricing=F16), trades, 512, figures, evidence) == []
        [warning] = _warnings(Precision(pricing=F16), trades, 16384, figures, evidence)
        assert "trades ['b'] (bond, engine=Discounting) at 16,384 paths" in warning
        assert "EPE validated up to 4,096 paths" in warning and TABLE in warning and "not refused" in warning

    def test_statuses_name_what_is_known(self):
        evidence = _evidence(_row("x", "bond", "EPE", None))
        figures = dataclasses.replace(PORTFOLIO, trade=("EPE", "NPV"), total=())
        [warning] = _warnings(Precision(pricing=F16), [Trade("b", "bond")], 1024, figures, evidence)
        assert "EPE fails at every path count measured; NPV not measured" in warning
        assert warning.endswith(PORTFOLIO.legend) and "today's NPV is float64" in warning

    def test_the_paired_share_of_a_corrected_figure(self):
        """A corrected figure takes the rows of the largest paired share at most the run's; a
        figure the sample does not correct, the rows without one."""
        evidence = _evidence(_row("a", "bond", "EPE", 16384), _row("b", "bond", "EPE", 1024, 0.0625),
                             _row("c", "bond", "EPE", 4096, 0.125), _row("d", "bond", "PFE_95", 4096))
        bond = {"bond": SETTINGS["bond"]}
        assert evidence.ceiling(KEY16, "bond", "EPE", bond) == (True, 16384)
        assert evidence.ceiling(KEY16, "bond", "EPE", bond, 0.1) == (True, 1024)
        assert evidence.ceiling(KEY16, "bond", "EPE", bond, 0.5) == (True, 4096)
        assert evidence.ceiling(KEY16, "bond", "EPE", bond, 0.03) == (False, None)
        assert evidence.measured_elsewhere(KEY16, "bond", "EPE", bond, 0.03) == ()  # the share, not the settings
        figures = dataclasses.replace(PORTFOLIO, trade=("EPE", "PFE_95"), total=())
        paired = Precision(pricing=F16, paired_fraction=0.5)
        assert _warnings(paired, [Trade("b", "bond")], 4096, figures, evidence) == []
        [warning] = _warnings(paired, [Trade("b", "bond")], 8192, figures, evidence)
        assert "with a paired sample of 0.5 of the paths" in warning

    def test_rows_of_two_campaigns_give_the_smaller_ceiling(self):
        bond = {"bond": SETTINGS["bond"]}
        evidence = _evidence(_row("x", "bond", "EPE", 4096), _row("x", "bond", "EPE", 1024, campaign="d"))
        assert evidence.ceiling(KEY16, "bond", "EPE", bond) == (True, 1024)
        evidence = _evidence(_row("x", "bond", "EPE", 4096), _row("x", "bond", "EPE", None, campaign="d"))
        assert evidence.ceiling(KEY16, "bond", "EPE", bond) == (True, None)

    def test_an_aggregate_needs_a_campaign_with_every_product(self):
        evidence = _evidence(_row("x", "netting_set", "EPE", 4096))
        figures = dataclasses.replace(PORTFOLIO, trade=(), total=("EPE",))
        swaps = [Trade("s", "swap"), Trade("b", "bond")]
        assert _warnings(Precision(pricing=F16), swaps, 1024, figures, evidence) == []
        [warning] = _warnings(Precision(pricing=F16), swaps + [Trade("o", "bermudan_swaption")], 1024, figures,
                              evidence)
        assert ("the netting set (products ['bermudan_swaption', 'bond', 'swap']; bermudan_swaption: engine=LgmGrid "
                "nx=8 sx=3 model=fixed; bond: engine=Discounting; swap: engine=Discounting)") in warning
        assert ("EPE not measured with these products and pricing settings (measured with bond: engine=Discounting; "
                "swap: engine=Discounting)") in warning

    def test_a_mix_of_pricing_stages_has_no_aggregate_row(self):
        precision = Precision(by_trade={"b": F16})
        figures = dataclasses.replace(PORTFOLIO, trade=(), total=("EPE",))
        [warning] = _warnings(precision, [Trade("s", "swap"), Trade("b", "bond")], 1024, figures, _evidence())
        assert "pricing stages differ" in warning and "netting set" in warning

    def test_float64_warns_of_nothing(self):
        trades = [Trade("s", "swap"), Trade("b", "bond")]
        assert _warnings(Precision(), trades, 10 ** 6, PORTFOLIO, _evidence()) == []
        assert _warnings(Precision(paired_fraction=0.5), trades, 10 ** 6, PORTFOLIO, _evidence()) == []

    def test_a_float64_trade_of_a_reduced_run_is_exact(self):
        """Only the reduced trade and the aggregate are warned of; the swap's stages are all
        float64."""
        warnings = _warnings(Precision(by_trade={"b": F16}), [Trade("s", "swap"), Trade("b", "bond")], 1024,
                             PORTFOLIO, _evidence())
        assert len(warnings) == 2 and "trades ['b'] (bond, " in warnings[0] and "swap" not in warnings[0]

    def test_market_risk_counts_scenarios(self):
        [trade, total] = _warnings(Precision(pricing=F16), [Trade("s", "swap")], 2048, MARKET_RISK, _evidence())
        assert "at 2,048 scenarios" in trade and "PnL not measured" in trade
        assert "the portfolio (products ['swap']; swap: engine=Discounting)" in total and "VaR_99, ES_99" in total


# ---------------------------------------------------------------------------
# Pricing settings (I-114)
# ---------------------------------------------------------------------------
JAMSHIDIAN = PricingConfig(european="Jamshidian", jamshidian=JamshidianEngineConfig(0.03, 0.01))
COARSE = LgmSwaptionEngineConfig(n_per_std=8, std_devs=3.0)


class TestKernelSettings:
    def test_each_product_is_keyed_by_what_changes_its_kernel(self):
        assert PricingConfig().kernel_settings(PRODUCTS, recalibrated=False) == {
            "american_swaption": "engine=LgmGrid nx=30 sx=5 steps_per_year=24 model=fixed",
            "bermudan_swaption": "engine=LgmGrid nx=30 sx=5 model=fixed", "bond": "engine=Discounting",
            "european_swaption": "engine=Bachelier", "swap": "engine=Discounting"}
        settings = dataclasses.replace(JAMSHIDIAN, jamshidian=JamshidianEngineConfig(0.03, 0.01, "Bisection"),
                                       bermudan=COARSE, american=dataclasses.replace(
                                           COARSE, std_devs=3.5, exercise_time_steps_per_year=12))
        assert settings.kernel_settings(PRODUCTS, recalibrated=True) == {
            "american_swaption": "engine=LgmGrid nx=8 sx=3.5 steps_per_year=12 model=recalibrated",
            "bermudan_swaption": "engine=LgmGrid nx=8 sx=3 model=recalibrated", "bond": "engine=Discounting",
            "european_swaption": "engine=Jamshidian solver=Bisection", "swap": "engine=Discounting"}

    def test_only_a_calibrated_model_is_recalibrated(self):
        fixed = PricingConfig(bermudan=dataclasses.replace(COARSE, calibration="None"))
        assert fixed.kernel_settings(["bermudan_swaption"], recalibrated=True) == {
            "bermudan_swaption": "engine=LgmGrid nx=8 sx=3 model=fixed"}

    def test_the_models_parameters_are_not_the_kernels(self):
        """Reversion, volatility, strategy, tolerance, shift horizon and the calibration's
        solver change the numbers a kernel rounds, as the market does, not the kernel."""
        other = dataclasses.replace(COARSE, reversion=0.05, volatility=0.02, strategy="CoterminalATM",
                                    bootstrap_tolerance=1e-6, shift_horizon=0.0, solver="Bisection")
        for recalibrated in (False, True):
            assert (PricingConfig(bermudan=other).kernel_settings(["bermudan_swaption"], recalibrated=recalibrated)
                    == PricingConfig(bermudan=COARSE).kernel_settings(["bermudan_swaption"],
                                                                      recalibrated=recalibrated))

    def test_each_analytic_states_whether_it_recalibrates(self):
        """The exposure simulation keeps an option's model on every path (A-24); market risk
        recalibrates a calibrated one under each scenario (I-112)."""
        trades = [Trade("o", "bermudan_swaption"), Trade("s", "swap")]
        pricing = PricingConfig(bermudan=COARSE)
        assert portfolio_settings(pricing, trades) == {"bermudan_swaption": "engine=LgmGrid nx=8 sx=3 model=fixed",
                                                       "swap": "engine=Discounting"}
        assert market_risk_settings(pricing, trades)["bermudan_swaption"] == (
            "engine=LgmGrid nx=8 sx=3 model=recalibrated")

    def test_an_unknown_product_is_refused(self):
        with pytest.raises(ValueError, match="no pricing engine for product 'equity'"):
            PricingConfig().kernel_settings(["equity"], recalibrated=False)


class TestSettingsLookup:
    """One engine, grid or calibration cannot validate another (I-114)."""

    @pytest.mark.parametrize("measured, run", [
        (PricingConfig(), JAMSHIDIAN),
        (JAMSHIDIAN, dataclasses.replace(JAMSHIDIAN, jamshidian=JamshidianEngineConfig(0.03, 0.01, "Bisection"))),
        (PricingConfig(bermudan=COARSE), PricingConfig()),
        (PricingConfig(american=COARSE), PricingConfig(american=dataclasses.replace(
            COARSE, exercise_time_steps_per_year=12))),
    ], ids=["Bachelier-Jamshidian", "Jamshidian-solver", "grid", "american-steps"])
    def test_rows_of_other_pricing_settings_validate_nothing(self, measured, run):
        trades = [Trade("e", "european_swaption"), Trade("b", "bermudan_swaption"), Trade("a", "american_swaption")]
        campaign = portfolio_settings(measured, trades)
        evidence = _evidence(*(Row("c", "x", KEY16, 0.0, t.product, "EPE", 16384, tuple(sorted(campaign.items())))
                               for t in trades))
        figures = dataclasses.replace(PORTFOLIO, trade=("EPE",), total=())
        precision = Precision(pricing=F16)
        assert _warnings(precision, trades, 1024, figures, evidence, settings=campaign) == []
        settings = portfolio_settings(run, trades)
        [warning] = _warnings(precision, trades, 1024, figures, evidence, settings=settings)
        [product] = [p for p in campaign if campaign[p] != settings[p]]
        assert f"({product}, {settings[product]}) at 1,024 paths" in warning
        assert f"EPE not measured with these pricing settings (measured with {campaign[product]})" in warning

    def test_a_fixed_model_cannot_validate_a_recalibrated_one(self):
        trades = [Trade("b", "bermudan_swaption")]
        pricing = PricingConfig(bermudan=COARSE)
        fixed = market_risk_settings(PricingConfig(bermudan=dataclasses.replace(COARSE, calibration="None")), trades)
        key = combination(Precision(pricing=F16), MARKET_RISK.stages, F16)
        evidence = _evidence(Row("c", "x", key, 0.0, "bermudan_swaption", "PnL", 8192, tuple(fixed.items())))
        figures = dataclasses.replace(MARKET_RISK, total=())
        assert _warnings(Precision(pricing=F16), trades, 2048, figures, evidence, settings=fixed) == []
        [warning] = _warnings(Precision(pricing=F16), trades, 2048, figures, evidence,
                              settings=market_risk_settings(pricing, trades))
        assert "model=recalibrated) at 2,048 scenarios" in warning
        assert ("PnL not measured with these pricing settings (measured with engine=LgmGrid nx=8 sx=3 model=fixed)"
                in warning)

    def test_an_aggregate_needs_each_products_settings(self):
        trades = [Trade("s", "swap"), Trade("b", "bermudan_swaption")]
        evidence = _evidence(_row("x", "netting_set", "EPE", 4096, products=("bermudan_swaption", "swap")))
        figures = dataclasses.replace(PORTFOLIO, trade=(), total=("EPE",))
        assert _warnings(Precision(pricing=F16), trades, 1024, figures, evidence) == []
        other = {**SETTINGS, "bermudan_swaption": "engine=LgmGrid nx=30 sx=5 model=fixed"}
        [warning] = _warnings(Precision(pricing=F16), trades, 1024, figures, evidence, settings=other)
        assert "EPE not measured with these products and pricing settings" in warning

    def test_a_product_without_settings_is_refused(self):
        with pytest.raises(ValueError, match=r"no pricing settings for products \['european_swaption'\]"):
            _warnings(Precision(pricing=F16), [Trade("e", "european_swaption")], 1024, PORTFOLIO, _evidence())


class TestPairedErrorLabel:
    @pytest.mark.parametrize("precision, labelled", [
        (Precision(simulation=F16, paired_fraction=0.1), True), (Precision(market=F16, paired_fraction=0.1), True),
        (Precision(pricing=F16, paired_fraction=0.1), True), (Precision(simulation=F16), False),
        (Precision(simulation=F32, paired_fraction=0.1), True),
    ])
    def test_a_single_scramble_cannot_measure_error_in_any_storage_format(self, precision, labelled):
        assert (paired_error_warning(precision) is not None) is labelled
        warning = paired_error_warning(precision, replicates=8)
        assert warning is None if not precision.paired_fraction else "independent scrambles" in warning


# ---------------------------------------------------------------------------
# The files of record
# ---------------------------------------------------------------------------
class TestRecord:
    def test_the_files_of_record_are_what_the_measurements_give(self):
        """`compliance/precision-evidence.md` and `engine/precision/evidence.json` are generated:
        after a campaign is rerun, `python -m tests.support.precision_evidence` regenerates both."""
        assert generator.main(["--check"]) == 0

    def test_the_engine_reads_its_table(self):
        evidence = shipped()
        assert evidence.rows and set(evidence.campaigns) == set(CAMPAIGNS)
        for name, campaign in evidence.campaigns.items():
            assert campaign["corrected"] == (sorted(CORRECTED_FIGURES) if campaign["kind"] == "portfolio" else [])
            assert campaign["commit"] != "unknown" and campaign["engine"] != "unknown" and campaign["seeds"] >= 8
        assert all(r.ceiling is None or r.ceiling in evidence.campaigns[r.campaign]["paths"] for r in evidence.rows)

    def test_each_record_holds_its_campaigns_pricing_settings(self):
        """A campaign's definition and its record agree on every product's pricing settings: a
        definition changed without measuring the campaign again fails here (I-114)."""
        for name, campaign in shipped().campaigns.items():
            assert campaign["settings"] == CAMPAIGNS[name]().settings(), name

    def test_the_file_layout_round_trips(self):
        campaigns = {"c": {"aggregate": "portfolio", "settings": {"swap": SETTINGS["swap"]}, "corrected": []}}
        cases = [{"campaign": "c", "case": "x", "combination": "k", "paired": 0.0,
                  "ceilings": {"swap": {"PnL": 2048}, "portfolio": {"VaR_99": None}}}]
        rows = Evidence.from_json(evidence_json(campaigns, cases)).rows
        assert set(rows) == {Row("c", "x", "k", 0.0, "swap", "PnL", 2048, _settings("swap")),
                             Row("c", "x", "k", 0.0, "portfolio", "VaR_99", None, _settings("swap"))}

    def test_a_file_without_pricing_settings_is_refused(self):
        """Rows of unknown settings would validate any: an evidence file of the layout before
        I-114 does not load."""
        campaigns = {"c": {"aggregate": "portfolio", "products": ["swap"], "corrected": []}}
        cases = [{"campaign": "c", "case": "x", "combination": "k", "paired": 0.0, "ceilings": {"swap": {"PnL": 1}}}]
        with pytest.raises(KeyError, match="settings"):
            Evidence.from_json(evidence_json(campaigns, cases))


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_runs():
    """Two of the shared portfolio's trades on two dates at 64 paths: float64; FP8 cube storage,
    alone and with a paired sample of half the paths."""
    campaign = PortfolioCampaign(
        name="tiny", description="", market_name="shared", market=shared.market(),
        trades=(shared.trades()["swap-payer"], shared.trades()["bond"]),
        simulation=CamConfig(dates=tuple(shared.ASOF + ORE.Period(m, ORE.Months) for m in (3, 12)),
                             base_currency="USD", ir={"USD": LgmConfig(0.03, 0.01, ("1Y",), ("9Y",))}),
        pricing=PricingConfig(), cases=(), paths=(64,), seeds=(1, 2))
    scenarios = campaign.scenarios(64, 1)
    fp8 = StagePrecision("float8_e4m3fn")
    return campaign, {name: campaign.run(scenarios, p) for name, p in (
        ("float64", Precision()), ("fp8", Precision(pricing=fp8)),
        ("paired", Precision(pricing=fp8, paired_fraction=0.5)))}


class TestEndToEnd:
    def test_a_float64_run_warns_of_nothing_and_a_reduced_one_names_its_figures(self, tiny_runs):
        campaign, runs = tiny_runs
        assert runs["float64"].warnings == []
        text = "\n".join(runs["fp8"].warnings)
        assert len(runs["fp8"].warnings) == 3  # the swap, the bond, the netting set
        assert "(swap, engine=Discounting)" in text and "(bond, engine=Discounting)" in text
        assert "the netting set" in text and "at 64 paths" in text
        assert "pricing=float8_e4m3fn/float64/float64" in text

    def test_the_paired_sample_corrects_exactly_the_figures_the_table_says(self, tiny_runs):
        """`CORRECTED_FIGURES` (the rows that keep their paired share) are the figures a paired
        sample changes; the cube's mean and PFE are the run's own."""
        campaign, runs = tiny_runs
        alone, paired = campaign.figures(runs["fp8"]), campaign.figures(runs["paired"])
        changed = {name.rsplit("/", 1)[1] for name in alone if not np.array_equal(alone[name], paired[name])}
        assert changed == set(CORRECTED_FIGURES)

    def test_a_market_risk_run_warns_in_scenarios(self):
        m = dataclasses.replace(campaigns.market_risk(), trades=campaigns.market_risk().trades[:2])
        result = m.run(m.scenarios(256, 1), Precision(pricing=F16))
        precision = [w for w in result.warnings if w.startswith("precision:")]
        assert len(precision) == 2 and all("at 256 scenarios" in w for w in precision)
        assert any("ES standard errors" in w for w in result.warnings)
        float64 = m.run(m.scenarios(256, 1), Precision())
        assert not [w for w in float64.warnings if w.startswith("precision:")]


# ---------------------------------------------------------------------------
# Pricing settings end to end, on the shipped table (I-114)
# ---------------------------------------------------------------------------
def _trade_warnings(result, trade_id):
    return [w for w in result.warnings if w.startswith("precision:") and f"'{trade_id}'" in w]


def _settings_portfolio(name: str, pricing: PricingConfig) -> PortfolioCampaign:
    return PortfolioCampaign(
        name="settings", description="", market_name="shared", market=shared.market(), trades=(shared.trades()[name],),
        simulation=CamConfig(dates=tuple(shared.ASOF + ORE.Period(m, ORE.Months) for m in (3, 12)),
                             base_currency="USD", ir={"USD": LgmConfig(0.03, 0.01, ("1Y",), ("9Y",))}),
        pricing=pricing, cases=(), paths=(64,), seeds=(1, 2))


class TestSettingsEndToEnd:
    """Each red before I-114: the shipped table's rows, measured with Bachelier, on the coarse
    grid and on a fixed LGM, validated a Jamshidian European, ORE's default grid and a Bermudan
    recalibrated under each scenario. The precisions are ones those rows validate."""

    @pytest.mark.parametrize("trade, precision, measured, unmeasured, settings", [
        ("european-payer", Precision(pricing=StagePrecision("float32")), PricingConfig(), JAMSHIDIAN,
         "engine=Jamshidian solver=Newton"),
        ("bermudan-payer-physical", Precision(pricing=F32), PricingConfig(bermudan=COARSE), PricingConfig(),
         "engine=LgmGrid nx=30 sx=5 model=fixed"),
    ], ids=["european-engine", "rollback-grid"])
    def test_a_portfolio_run_is_validated_only_under_its_measured_settings(self, trade, precision, measured,
                                                                           unmeasured, settings):
        runs = {}
        for name, pricing in (("measured", measured), ("unmeasured", unmeasured)):
            campaign = _settings_portfolio(trade, pricing)
            runs[name] = campaign.run(campaign.scenarios(64, 1), precision)
        trade_id = shared.trades()[trade].trade_id
        assert _trade_warnings(runs["measured"], trade_id) == []
        [warning] = _trade_warnings(runs["unmeasured"], trade_id)
        assert f", {settings}) at 64 paths" in warning and "not measured with these pricing settings" in warning

    def test_a_recalibrated_bermudan_is_not_validated_by_fixed_model_rows(self, monkeypatch):
        """The market-risk campaigns before 2026-10-18 measured one fixed LGM shared by every
        scenario; with only their rows, a Bermudan recalibrated under each scenario (its own
        induction per scenario, I-112) has none."""
        fixed_only = Evidence([r for r in shipped().rows if r.campaign != "market_risk_calibrated"])
        monkeypatch.setattr(evidence_module, "shipped", lambda: fixed_only)
        base = campaigns.market_risk()
        calibrated = dataclasses.replace(base.pricing, bermudan=dataclasses.replace(
            base.pricing.bermudan, calibration="Bootstrap"))
        precision = Precision(pricing=StagePrecision("float32"))
        for pricing, warned in ((base.pricing, False), (calibrated, True)):
            m = dataclasses.replace(base, trades=(base.trades[3],), pricing=pricing)
            warnings = _trade_warnings(m.run(m.scenarios(256, 1), precision), "bermudan")
            assert bool(warnings) is warned
        [warning] = warnings
        assert "model=recalibrated) at 256 scenarios" in warning
        assert "PnL not measured with these pricing settings (measured with engine=LgmGrid nx=8 sx=3 model=fixed)" \
            in warning

    def test_the_shipped_table_validates_a_recalibrated_bermudan_by_its_own_campaign(self):
        """Every figure of the recalibrated Bermudan and of a portfolio holding it reads the
        `market_risk_calibrated` campaign's rows alone."""
        campaign = campaigns.market_risk_calibrated()
        settings = market_risk_settings(campaign.pricing, campaign.trades)
        own = Evidence([r for r in shipped().rows if r.campaign == campaign.name])
        checked = 0
        for row in own.rows:
            if row.scope in ("bermudan_swaption", "portfolio"):
                key = {row.scope: settings[row.scope]} if row.scope in settings else settings
                assert shipped().ceiling(row.combination, row.scope, row.figure, key) == own.ceiling(
                    row.combination, row.scope, row.figure, key)
                checked += 1
        assert checked == len(campaign.cases) * (1 + len(market_risk_figures(campaign.quantiles).total))
