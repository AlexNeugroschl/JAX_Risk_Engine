"""
The paired float64 sample, the two-level estimator and the precision report (roadmap 1.7,
decision A-13, I-12; docs/planning/details/precision.md §9, §13.6).

  * The estimator needs no market (§13.6): identical samples give the plain mean and a zero
    correction; a known bias is removed within the estimate's standard error, and the
    uncorrected mean is flagged by its correction's; over many seeds the corrected estimate's
    95% interval holds the float64 figure at its stated rate, which needs the covariance term.
  * Pipelines: at float64 against float64 every paired difference is exactly 0, which holds
    only if the paired paths are the run's own (§9.2); with every path paired the estimate is
    the float64 run's figure; a stored-cube bias is corrected within its standard error.
  * The report reads the realized format of every stored array (§13.3), the devices and the
    backend from the run (over HTTP, the worker's: tests/test_api_market_path.py, I-12), and a
    repeated run compiles nothing.
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.api.schemas import PrecisionReportSchema, PrecisionSchema
from engine.market_risk import MarketRiskRequest, monte_carlo_scenarios, run_market_risk
from engine.portfolio import (
    CamConfig, HullWhiteConfig, LgmConfig, LgmSwaptionEngineConfig, PortfolioRequest, PricingConfig, RunConfig,
    price_portfolio,
)
from engine.precision import (
    BLOCK, FORMAT_NAMES, MeanEstimate, Precision, QuantileEstimate, StagePrecision, paired_paths,
    paired_quantile, two_level_mean,
)
from engine.precision import estimate as estimate_module
from engine.risk.exposure import exposure_profile, netting_set_profile
from tests import market_risk_support as mr
from tests.support import portfolio as shared
from tests.support.compiles import count_compiles

FAST = LgmSwaptionEngineConfig(n_per_std=8, std_devs=3.0)
PRICING = PricingConfig(bermudan=FAST, american=FAST)
DATES = tuple(shared.ASOF + ORE.Period(m, ORE.Months) for m in (3, 12, 24))
#: One trade of each kind of kernel: legs, a European, a Bermudan's per-path recalibration, a bond.
TRADES = ("swap-payer", "european-payer", "bermudan-payer-physical", "bond")
#: Cheap kernels only.
CHEAP = ("swap-payer", "european-payer", "bond")


def _simulation(samples=96, seed=3, model=LgmConfig) -> CamConfig:
    return CamConfig(dates=DATES, base_currency="USD", ir={"USD": model(0.03, 0.01, ("1Y", "2Y"), ("9Y", "8Y"))},
                     samples=samples, seed=seed)


def _price(policy, names=TRADES, **simulation):
    request = PortfolioRequest(market=shared.market(), trades=[shared.trades()[n] for n in names],
                               config=RunConfig(simulation=_simulation(**simulation), pricing=PRICING,
                                                precision=policy))
    return price_portfolio(request)


def _same(a, b, err_msg=""):
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=err_msg)


# ---------------------------------------------------------------------------
# The paired sample's size and the policy field
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("paths, fraction, expected", [
    (1000, 0.0, 0), (1000, 0.001, BLOCK), (1000, 0.02, BLOCK), (1000, 0.1, 4 * BLOCK), (128, 0.25, BLOCK),
    (128, 0.26, 2 * BLOCK), (1000, 1.0, 1000), (10, 0.1, 10), (100, 0.9, 3 * BLOCK), (100, 0.97, 100),
])
def test_the_paired_sample_is_whole_blocks_of_the_first_paths(paths, fraction, expected):
    assert paired_paths(paths, fraction) == expected


class TestPolicy:
    def test_no_paired_sample_by_default(self):
        assert Precision().paired_fraction == 0.0

    @pytest.mark.parametrize("value", [0, 1, 0.02, np.float64(0.5)])
    def test_a_fraction_in_the_unit_interval_is_kept_as_a_float(self, value):
        policy = Precision(paired_fraction=value)
        assert type(policy.paired_fraction) is float and policy.paired_fraction == float(value)
        assert hash(policy) == hash(Precision(paired_fraction=float(value)))

    @pytest.mark.parametrize("value, error", [
        (-0.1, ValueError), (1.5, ValueError), (float("nan"), TypeError), (float("inf"), TypeError),
        (True, TypeError), ("0.1", TypeError), (None, TypeError),
    ])
    def test_anything_else_is_refused_naming_the_field(self, value, error):
        with pytest.raises(error, match="Precision.paired_fraction"):
            Precision(paired_fraction=value)

    def test_float64_throughout_may_be_paired_as_a_check(self):
        """D-9: never refused; it measures zero (see the pipeline tests)."""
        assert Precision(paired_fraction=0.5).paired_fraction == 0.5

    def test_the_wire_form_round_trips_and_refuses_out_of_range(self):
        policy = Precision(market=StagePrecision("float16", "float32", "float32"),
                           by_product={"bond": StagePrecision("float32")}, by_trade={"s1": StagePrecision()},
                           rounding="stochastic", rounding_seed=4, paired_fraction=0.02)
        assert PrecisionSchema.from_dataclass(policy).to_dataclass() == policy
        assert PrecisionSchema.model_validate({"paired_fraction": 0.25}).to_dataclass() == Precision(paired_fraction=0.25)
        for bad in (-0.5, 1.01):
            with pytest.raises(ValueError, match="paired_fraction"):
                PrecisionSchema.model_validate({"paired_fraction": bad})


# ---------------------------------------------------------------------------
# The estimator (no market)
# ---------------------------------------------------------------------------
class TestTwoLevelMean:
    def test_identical_samples_give_the_plain_mean_and_no_correction(self):
        """float64 against float64 (§13.6): the estimate is the uncorrected mean, bit for bit."""
        low = jnp.asarray(np.random.default_rng(1).standard_normal((256, 3)))
        got = two_level_mean(low, low[:64])
        _same(got.value, got.uncorrected)
        _same(got.uncorrected, jnp.mean(low, axis=0))
        _same(got.correction, 0.0)
        _same(got.max_difference, 0.0)
        _same(got.correction_standard_error, 0.0)
        _same(got.standard_error, got.uncorrected_standard_error)
        np.testing.assert_allclose(got.uncorrected_standard_error, np.std(np.asarray(low), axis=0, ddof=1) / 16,
                                   rtol=1e-14)
        assert (got.paths, got.paired_paths) == (256, 64)

    def test_a_known_bias_is_removed_within_its_standard_error(self):
        """A 'format' that adds 0.3 and noise to every value: the uncorrected mean is many
        standard errors off and its correction flags it; the estimate is within three of the
        float64 figure (0)."""
        rng = np.random.default_rng(2)
        exact = rng.standard_normal(4096)
        low = exact + 0.3 + 0.05 * rng.standard_normal(4096)
        got = two_level_mean(low, exact[:512])
        assert abs(float(got.uncorrected)) > 10 * float(got.uncorrected_standard_error)
        assert abs(float(got.correction)) > 50 * float(got.correction_standard_error)
        assert abs(float(got.correction) + 0.3) < 3 * float(got.correction_standard_error)
        assert abs(float(got.value)) < 3 * float(got.standard_error)

    def test_with_every_path_paired_the_estimate_is_the_float64_mean(self):
        rng = np.random.default_rng(3)
        exact = rng.standard_normal((300, 2))
        got = two_level_mean(exact + 0.1 * rng.standard_normal((300, 2)) + 0.2, exact)
        np.testing.assert_allclose(got.value, exact.mean(axis=0), rtol=0, atol=1e-14)
        np.testing.assert_allclose(got.standard_error, exact.std(axis=0, ddof=1) / np.sqrt(300), rtol=1e-12)

    @pytest.mark.parametrize("correlation", [0.0, -0.9, 0.9])
    def test_the_95_percent_interval_covers_at_its_rate(self, correlation):
        """Coverage over 400 seeds, with the difference correlated with the value: dropping
        the covariance term would cover 99.9% at -0.9 and 87% at 0.9."""
        num, paired, bias = 512, 256, 0.25
        covered = 0
        for seed in range(400):
            rng = np.random.default_rng(seed)
            low = rng.standard_normal(num)
            exact = low[:paired] + correlation * low[:paired] + bias + 0.1 * rng.standard_normal(paired)
            got = two_level_mean(low, exact)
            covered += abs(float(got.value) - bias) <= 1.96 * float(got.standard_error)
        assert 0.92 <= covered / 400 <= 0.98, covered / 400

    @pytest.mark.parametrize("low, high", [
        (np.zeros((8, 2)), np.zeros((9, 2))), (np.zeros((8, 2)), np.zeros((4, 3))), (np.zeros((8,)), np.zeros((0,))),
        (np.zeros(()), np.zeros(())),
    ])
    def test_a_sample_that_cannot_pair_is_refused(self, low, high):
        with pytest.raises(ValueError, match="pairs the first n"):
            two_level_mean(low, high)

    def test_it_compiles_once_per_pair_of_shapes(self):
        rng = np.random.default_rng(4)
        two_level_mean(rng.standard_normal((64, 3)), rng.standard_normal((32, 3)))
        compiled = estimate_module._two_level._cache_size()
        for _ in range(3):
            two_level_mean(rng.standard_normal((64, 3)), rng.standard_normal((32, 3)))
        assert estimate_module._two_level._cache_size() == compiled


def test_a_quantile_is_measured_on_the_paired_paths_not_corrected():
    rng = np.random.default_rng(5)
    exact = rng.standard_normal((200, 2))
    low = exact + 0.05
    statistic = lambda x: jnp.sort(x, axis=0)[int(0.9 * (x.shape[0] - 1))]  # noqa: E731
    got = paired_quantile(statistic, low, exact[:64])
    assert isinstance(got, QuantileEstimate) and (got.paths, got.paired_paths) == (200, 64)
    _same(got.value, statistic(jnp.asarray(low)))
    _same(got.paired, statistic(jnp.asarray(low[:64])))
    _same(got.paired_float64, statistic(jnp.asarray(exact[:64])))
    np.testing.assert_allclose(got.difference, 0.05, rtol=1e-12)


class TestExposureWithAPairedSample:
    """`exposure_profile(..., paired=...)` on synthetic paths."""

    @staticmethod
    def _paths(seed=6, num=128):
        rng = np.random.default_rng(seed)
        npv = jnp.asarray(rng.standard_normal((num, 4)) * 100.0)
        numeraire = jnp.asarray(1.0 + 0.1 * rng.random((num, 4)))
        return npv, numeraire, jnp.asarray([0.99, 0.97, 0.95, 0.9]), [0.5, 1.0, 2.0, 3.0]

    def test_pairing_the_run_with_itself_changes_no_figure(self):
        npv, numeraire, discount, times = self._paths()
        plain = exposure_profile(npv, 3.0, numeraire, discount, times)
        paired = exposure_profile(npv, 3.0, numeraire, discount, times, paired=(npv[:32], numeraire[:32]))
        assert plain.estimates == {} and set(paired.estimates) == {"EPE", "ENE", "PFE_95", "PFE_99"}
        for name in ("epe", "ene", "ee_b", "eee_b", "epe_b", "eepe_b"):
            _same(getattr(paired, name), getattr(plain, name), err_msg=name)
        for key, figure in paired.estimates.items():
            assert figure.paths == 128 and figure.paired_paths == 32
            if isinstance(figure, MeanEstimate):
                _same(figure.correction, 0.0, err_msg=key)
            else:
                _same(figure.difference, 0.0, err_msg=key)
                _same(figure.value, plain.pfe[key][1:], err_msg=key)

    def test_the_means_are_corrected_and_the_basel_figures_follow(self):
        """A low-precision cube biased by +5 on every path: EPE/ENE are corrected by the
        paired paths (here every path), EE_B and its successors follow from the corrected EPE."""
        npv, numeraire, discount, times = self._paths()
        exact = exposure_profile(npv, 3.0, numeraire, discount, times)
        got = exposure_profile(npv + 5.0, 3.0, numeraire, discount, times, paired=(npv, numeraire))
        for name in ("epe", "ene", "ee_b", "eee_b", "epe_b", "eepe_b"):
            np.testing.assert_allclose(getattr(got, name), getattr(exact, name), rtol=1e-13, err_msg=name)
        assert float(got.estimates["EPE"].correction[0]) < -1.0  # the bias was there

    def test_a_netting_set_pairs_its_netted_paths(self):
        npv, numeraire, discount, times = self._paths()
        cube = jnp.stack([npv, -0.5 * npv], axis=-1)
        netted = netting_set_profile(cube, [1.0, 2.0], numeraire, discount, times, paired=(cube[:32], numeraire[:32]))
        alone = exposure_profile(0.5 * npv, 3.0, numeraire, discount, times, paired=(0.5 * npv[:32], numeraire[:32]))
        _same(netted.estimates["EPE"].value, alone.estimates["EPE"].value)

    def test_a_paired_sample_of_another_shape_is_refused(self):
        npv, numeraire, discount, times = self._paths()
        with pytest.raises(ValueError, match="paired npv"):
            exposure_profile(npv, 0.0, numeraire, discount, times, paired=(npv[:32, :3], numeraire[:32, :3]))


# ---------------------------------------------------------------------------
# The portfolio pipeline
# ---------------------------------------------------------------------------
def _all_figures_exactly_zero(report):
    for key, figure in report.figures.items():
        if isinstance(figure, MeanEstimate):
            _same(figure.correction, 0.0, err_msg=key)
            _same(figure.max_difference, 0.0, err_msg=key)
            _same(figure.value, figure.uncorrected, err_msg=key)
        else:
            _same(figure.paired, figure.paired_float64, err_msg=key)
            _same(figure.difference, 0.0, err_msg=key)


class TestPortfolioPairedSample:
    @pytest.mark.parametrize("model", [LgmConfig, HullWhiteConfig])
    def test_float64_against_float64_every_paired_difference_is_exactly_zero(self, model):
        """The paired paths are the run's own: same Sobol points, same kernels, bit for bit,
        including a Bermudan recalibrated on every path; and the figures are those of the run
        without a paired sample."""
        plain = _price(Precision(), model=model)
        paired = _price(Precision(paired_fraction=0.5), model=model)
        report = paired.precision
        assert (report.paths, report.paired_paths) == (96, 64)
        expected = {f"{scope}/{f}" for scope in ["netting_set"] + [f"trades/{n}" for n in TRADES]
                    for f in ("EPE", "ENE", "PFE_95", "PFE_99")}
        assert set(report.figures) == expected
        _all_figures_exactly_zero(report)
        _same(paired.npv_cube, plain.npv_cube)
        for name in ("epe", "ene", "ee_b", "eee_b", "epe_b", "eepe_b"):
            _same(getattr(paired.exposure, name), getattr(plain.exposure, name), err_msg=name)
            for i in range(len(TRADES)):
                _same(getattr(paired.trade_exposures[i], name), getattr(plain.trade_exposures[i], name))
        assert paired.exposure.basel_eepe == plain.exposure.basel_eepe

    def test_with_every_path_paired_the_figures_are_the_float64_runs(self):
        """An FP8 cube with every path paired: EPE/ENE equal the float64 run's to rounding,
        though the uncorrected means are not; PFE stays the run's own, measured against the
        float64 run's."""
        exact = _price(Precision(), CHEAP)
        policy = Precision(pricing=StagePrecision("float8_e4m3fn"), paired_fraction=1.0)
        got = _price(policy, CHEAP)
        figures = got.precision.figures
        assert got.precision.paired_paths == 96
        scale = float(jnp.max(jnp.abs(exact.exposure.epe)))
        for key, profile, reference in [("netting_set", got.exposure, exact.exposure),
                                        *[(f"trades/{n}", p, e) for n, p, e in zip(CHEAP, got.trade_exposures,
                                                                                     exact.trade_exposures)]]:
            for name in ("epe", "ene", "ee_b", "eee_b", "epe_b", "eepe_b"):
                np.testing.assert_allclose(getattr(profile, name), getattr(reference, name), rtol=0,
                                           atol=1e-12 * scale, err_msg=f"{key} {name}")
            _same(figures[f"{key}/PFE_95"].value, profile.pfe["PFE_95"][1:])
            _same(figures[f"{key}/PFE_95"].paired_float64, reference.pfe["PFE_95"][1:])
        assert float(jnp.max(jnp.abs(figures["netting_set/EPE"].correction))) > 1e-4 * scale

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_a_stored_cube_bias_is_corrected_within_its_standard_error(self, seed):
        """FP8 storage, rounded to nearest, of a concentrated column (a bond's cube, I-75) is a
        bias. Against the float64 run on the same paths, the estimate's error is the paired
        differences' mean over the other paths, whose standard error is the correction's
        times sqrt(1 - n/N)."""
        names = ("bond", "swap-payer")
        exact = _price(Precision(), names, samples=256, seed=seed)
        got = _price(Precision(pricing=StagePrecision("float8_e5m2"), paired_fraction=0.25), names, samples=256,
                     seed=seed)
        for key, reference in (("trades/bond/EPE", exact.trade_exposures[0]), ("netting_set/EPE", exact.exposure)):
            figure = got.precision.figures[key]
            error = np.abs(np.asarray(figure.value) - np.asarray(reference.epe[1:]))
            bound = 3.0 * np.asarray(figure.correction_standard_error) * np.sqrt(1 - 64 / 256)
            assert np.all(error <= bound), (key, error, bound)

    def test_a_repeated_paired_run_compiles_nothing(self):
        policy = Precision(simulation=StagePrecision("float32", "float32", "float32"),
                           pricing=StagePrecision("float16", "float32", "float32"), paired_fraction=0.3)
        _price(policy, TRADES)
        with count_compiles() as counter:
            _price(policy, TRADES)
        assert sum(counter.values()) == 0, counter

    def test_a_paired_float32_run_never_mixes_dtypes(self):
        """Under strict promotion: the float32 run, the float64 paired run and the estimates."""
        f32 = StagePrecision("float32", "float32", "float32")
        with jax.numpy_dtype_promotion("strict"):
            result = _price(Precision(simulation=f32, market=f32, pricing=f32, paired_fraction=0.5), TRADES)
        assert all(np.all(np.isfinite(np.asarray(f.value))) for f in result.precision.figures.values())


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------
def _expected_realized(policy, names):
    sim, mkt = policy.simulation.storage, policy.market.storage
    trades = [shared.trades()[n] for n in names]
    return {"shocks": sim, "states": sim, "market": mkt,
            **{f"values/{t.trade_id}": policy.precision_for(t).storage for t in trades}}


class TestReport:
    @pytest.mark.parametrize("name", FORMAT_NAMES)
    @pytest.mark.parametrize("stage", ["simulation", "market", "pricing"])
    def test_the_realized_format_of_every_class_is_the_policys(self, stage, name):
        """§13.3: each class × enabled format, read from the arrays; one trade overridden."""
        compute = "float64" if name == "float64" else "float32"
        policy = dataclasses.replace(Precision(by_trade={"bond": StagePrecision("float32")}),
                                     **{stage: StagePrecision(name, compute, compute)})
        report = _price(policy, CHEAP, samples=32).precision
        assert report.realized == _expected_realized(policy, CHEAP)
        assert report.policy == policy
        assert report.trades == {n: policy.precision_for(shared.trades()[n]) for n in CHEAP}

    def test_it_names_the_device_and_backend_the_run_used(self):
        result = _price(Precision(), CHEAP, samples=32)
        report = result.precision
        device = next(iter(result.npv_cube.devices()))
        assert report.devices == (f"{device.platform}:{device.id} ({device.device_kind})",)
        assert report.backend == jax.default_backend() and report.jax_version == jax.__version__
        assert (report.paths, report.paired_paths, report.figures) == (32, 0, {})

    def test_a_run_without_paths_reports_its_policy_and_no_stored_array(self):
        policy = Precision(pricing=StagePrecision("float32"), paired_fraction=0.5)
        request = PortfolioRequest(market=shared.market(), trades=[shared.trades()["swap-payer"]],
                                   config=RunConfig(precision=policy), scenario_risk=False)
        report = price_portfolio(request).precision
        assert report.policy == policy and report.realized == {} and report.devices
        assert (report.paths, report.paired_paths, report.figures) == (0, 0, {})

    def test_the_report_crosses_the_wire_with_its_figures(self):
        report = _price(Precision(pricing=StagePrecision("bfloat16", "float32", "float32"), paired_fraction=0.5),
                        CHEAP).precision
        wire = PrecisionReportSchema.from_dataclass(report).model_dump()
        assert wire["realized"] == report.realized and wire["policy"]["paired_fraction"] == 0.5
        assert wire["figures"]["netting_set/EPE"]["kind"] == "mean"
        assert wire["figures"]["netting_set/PFE_99"]["kind"] == "quantile"
        np.testing.assert_array_equal(wire["figures"]["trades/bond/ENE"]["correction"],
                                      np.asarray(report.figures["trades/bond/ENE"].correction))
        assert PrecisionReportSchema.model_validate(wire).figures["netting_set/EPE"].kind == "mean"


# ---------------------------------------------------------------------------
# Market risk
# ---------------------------------------------------------------------------
class TestMarketRisk:
    @staticmethod
    def _run(policy, num=256):
        scenarios = monte_carlo_scenarios(mr.factors(), mr.covariance(), 10, num, seed=3)
        trades = [mr.swap(), mr.european(), mr.bermudan(), mr.bond()]
        return run_market_risk(MarketRiskRequest(trades, mr.market(), scenarios, mr.PRICING, quantiles=(0.95, 0.9),
                                                 precision=policy))

    def test_float64_against_float64_every_paired_difference_is_exactly_zero(self):
        """The paired scenarios are revalued in the run's batches (here one of 256): in a batch of
        their own (128) a vectorized kernel rounds an ES differently by an ulp."""
        plain, paired = self._run(Precision()), self._run(Precision(paired_fraction=0.5))
        report = paired.precision
        assert (report.paths, report.paired_paths) == (256, 128)
        assert set(report.figures) == {"portfolio/VaR_95", "portfolio/ES_95", "portfolio/VaR_90", "portfolio/ES_90"}
        _all_figures_exactly_zero(report)
        assert paired.risk == plain.risk
        for key, figure in report.figures.items():
            assert float(figure.value) == plain.risk[key.split("/")[1]], key

    def test_with_every_scenario_paired_the_float64_figure_is_measured(self):
        exact = self._run(Precision())
        policy = Precision(simulation=StagePrecision("float32", "float32", "float32"),
                           pricing=StagePrecision("float8_e4m3fn", "float32", "float32"), paired_fraction=1.0)
        got = self._run(policy)
        for key, figure in got.precision.figures.items():
            name = key.split("/")[1]
            assert float(figure.value) == got.risk[name] and float(figure.paired_float64) == exact.risk[name], key
        assert any(float(f.difference) != 0.0 for f in got.precision.figures.values())

    def test_a_repeated_paired_run_compiles_nothing(self):
        policy = Precision(pricing=StagePrecision("float32", "float32", "float32"), paired_fraction=0.3)
        self._run(policy, num=96)
        with count_compiles() as counter:
            self._run(policy, num=96)
        assert sum(counter.values()) == 0, counter

    def test_the_report_reads_the_shifts_and_each_pnl_from_the_arrays(self):
        policy = Precision(simulation=StagePrecision("float16", "float32", "float32"),
                           by_trade={"bermudan": StagePrecision("float32")})
        report = self._run(policy, num=64).precision
        assert report.realized == {"shocks": "float16", "values/swap": "float64", "values/european": "float64",
                                   "values/bermudan": "float32", "values/bond": "float64"}
        assert (report.paths, report.paired_paths, report.figures) == (64, 0, {})
