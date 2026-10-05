"""
The run configuration (`engine.portfolio.config.RunConfig`, roadmap 1.2 and 1.3, I-68): one
value naming the model per currency and simulation, the engine per product, the Greeks method
and settings, and the precision per stage, with ORE's defaults.

  * The defaults are ORE's, and a request that spells them out prices bit for bit as one that
    omits the configuration.
  * Every model runs with every engine and Greeks method (1.3): the Hull-White model is
    `HullWhiteConfig` in `CamConfig.ir`, on the same pipeline as the LGM. Before 1.3 it was a
    second pipeline selected by the market's type, with its own engines and Greeks, and each
    pipeline refused the other's options.
  * What the pipeline does not implement yet is refused before any work, naming the field:
    today a precision format before the roadmap step that enables it (tests/test_precision.py),
    and what the Jamshidian engine cannot price.
  * The bump-and-revalue settings (`GreeksConfig.sensitivity`, ORE's sensitivity.xml) reach
    the Greeks.
"""
import dataclasses

import numpy as np
import ORE
import pytest

from engine.portfolio import (
    CamConfig, GreeksConfig, HullWhiteConfig, JamshidianEngineConfig, LgmConfig, LgmSwaptionEngineConfig,
    PortfolioRequest, Precision, PricingConfig, RunConfig, SensitivityConfig, StagePrecision, price_portfolio,
)
from engine.portfolio.market_path import validate_request
from engine.risk.sensitivities import portfolio_sensitivities
from tests.support import portfolio as shared

FAST = LgmSwaptionEngineConfig(n_per_std=12, std_devs=4.0)
JAMSHIDIAN = PricingConfig(european="Jamshidian", jamshidian=JamshidianEngineConfig(0.03, 0.01))
MODELS = {"LGM": LgmConfig, "HullWhite": HullWhiteConfig}


def _cam(base="USD", model="LGM"):
    return CamConfig(dates=(shared.ASOF + ORE.Period(1, ORE.Years),), base_currency=base,
                     ir={base: MODELS[model](0.03, 0.01)}, samples=8)


def _market_request(names=("swap-payer", "european-payer"), **kwargs):
    trades = [shared.trades()[n] for n in names]
    return PortfolioRequest(market=shared.market(), trades=trades, scenario_risk=False, **kwargs)


class TestDefaults:
    def test_the_defaults_are_ores(self):
        config = RunConfig()
        assert config.simulation is None
        assert config.pricing == PricingConfig() and config.pricing.european == "Bachelier"
        assert config.pricing.jamshidian is None
        assert config.greeks.method == "Bump" and config.greeks.sensitivity == SensitivityConfig()
        assert config.precision == Precision.throughout("float64") == Precision()
        assert PortfolioRequest(market=shared.market(), trades=[]).config == config

    def test_the_reporting_currency_is_the_simulations_else_usd(self):
        assert RunConfig().reporting_currency == "USD"
        assert RunConfig(base_currency="EUR").reporting_currency == "EUR"
        assert RunConfig(simulation=_cam("EUR")).reporting_currency == "EUR"
        assert RunConfig(simulation=_cam("EUR"), base_currency="EUR").reporting_currency == "EUR"

    def test_explicit_defaults_price_bit_for_bit_as_omitted_ones(self):
        explicit = RunConfig(simulation=None, pricing=PricingConfig(bermudan=LgmSwaptionEngineConfig()),
                             greeks=GreeksConfig("Bump", SensitivityConfig()), precision=Precision(),
                             base_currency="USD")
        a = price_portfolio(_market_request(compute_greeks=True))
        b = price_portfolio(_market_request(compute_greeks=True, config=explicit))
        assert a.base_npv_per_trade == b.base_npv_per_trade
        for i in a.greeks:
            assert a.greeks[i].keys() == b.greeks[i].keys()
            for key in a.greeks[i]:
                np.testing.assert_array_equal(a.greeks[i][key], b.greeks[i][key])


class TestConfigurationValues:
    @pytest.mark.parametrize("build, error, match", [
        (lambda: RunConfig(precision=64), TypeError, "RunConfig.precision must be an engine.precision.Precision"),
        (lambda: RunConfig(pricing={"european": "Bachelier"}), TypeError, "RunConfig.pricing"),
        (lambda: RunConfig(greeks="Bump"), TypeError, "RunConfig.greeks"),
        (lambda: GreeksConfig(method="Adjoint"), ValueError, "GreeksConfig.method"),
        (lambda: GreeksConfig(sensitivity={}), TypeError, "GreeksConfig.sensitivity"),
        (lambda: PricingConfig(european="Black"), ValueError, "european must be one of"),
        (lambda: RunConfig(simulation=_cam("USD"), base_currency="EUR"), ValueError, "contradicts"),
    ])
    def test_a_malformed_configuration_is_refused_by_name(self, build, error, match):
        with pytest.raises(error, match=match):
            build()

    def test_the_market_is_the_market_and_the_model_is_in_the_simulation(self):
        """The Hull-White model is `HullWhiteConfig` in `CamConfig.ir`; neither a `Market` as the
        simulation nor a `CamConfig` as the market is accepted."""
        with pytest.raises(TypeError, match="HullWhiteConfig in CamConfig.ir"):
            RunConfig(simulation=shared.market())
        with pytest.raises(TypeError, match="market must be today's Market"):
            PortfolioRequest(market=_cam(), trades=[])
        with pytest.raises(TypeError, match="config must be a RunConfig"):
            PortfolioRequest(market=shared.market(), trades=[], config=PricingConfig())

    def test_the_jamshidian_engine_needs_its_model_and_only_it_takes_one(self):
        with pytest.raises(ValueError, match="PricingConfig.jamshidian"):
            PricingConfig(european="Jamshidian")
        with pytest.raises(ValueError, match="PricingConfig.jamshidian"):
            PricingConfig(jamshidian=JamshidianEngineConfig(0.03, 0.01))

    @pytest.mark.parametrize("build", [
        lambda: JamshidianEngineConfig(0.0, 0.01), lambda: JamshidianEngineConfig(0.03, -0.01),
        lambda: JamshidianEngineConfig(float("nan"), 0.01),
    ])
    def test_a_jamshidian_model_is_positive_and_finite(self, build):
        with pytest.raises(ValueError):
            build()


class TestWhatThePipelineDoesNotImplementIsRefused:
    def test_the_jamshidian_engine_refuses_what_quantlibs_refuses_naming_the_trade(self):
        request = _market_request(("european-receiver-otm-cash",), config=RunConfig(pricing=JAMSHIDIAN))
        with pytest.raises(ValueError, match="european-receiver-otm-cash.*floating_spread"):
            validate_request(request)

    def test_an_engine_is_checked_only_where_the_run_uses_it(self):
        validate_request(_market_request(("swap-payer", "european-receiver-otm-cash"),
                                         config=RunConfig(greeks=GreeksConfig(method="AD"))))
        validate_request(_market_request(("swap-payer",), config=RunConfig(pricing=JAMSHIDIAN)))

    @pytest.mark.parametrize("model", MODELS)
    def test_every_enabled_precision_is_accepted_under_every_model(self, model):
        """Since roadmap 1.4 every stage is adjustable under either model; before it, a stage
        other than the simulation below float64 was refused here."""
        for precision in (Precision.throughout("float32"), Precision(pricing=StagePrecision("float32"))):
            request = _market_request(("swap-payer",), config=RunConfig(simulation=_cam(model=model),
                                                                        precision=precision))
            validate_request(dataclasses.replace(request, scenario_risk=True))

    def test_calibration_targets_are_not_a_request_field(self):
        """The basket is the model's (the `LgmConfig`/`HullWhiteConfig` tenors), per currency;
        before 1.3 the Hull-White request took its own `calibration_targets`."""
        with pytest.raises(TypeError, match="calibration_targets"):
            PortfolioRequest(market=shared.market(), trades=[], calibration_targets=[])


class TestEveryModelRunsWithEveryEngineAndMethod:
    """I-68: the model is an option of the run, not a pipeline. Every combination prices
    today, on the paths and its Greeks; before 1.3 the Hull-White request refused the Bachelier
    engine and AD Greeks' market path, and the market path refused Jamshidian and AD."""

    @pytest.mark.parametrize("model", MODELS)
    @pytest.mark.parametrize("pricing", [PricingConfig(), JAMSHIDIAN], ids=["Bachelier", "Jamshidian"])
    @pytest.mark.parametrize("method", ["Bump", "AD"])
    def test_prices_with_greeks_and_scenario_risk(self, model, pricing, method):
        config = RunConfig(simulation=_cam(model=model), pricing=pricing, greeks=GreeksConfig(method=method))
        request = dataclasses.replace(_market_request(config=config, compute_greeks=True), scenario_risk=True)
        result = price_portfolio(request)
        assert np.all(np.isfinite(np.asarray(result.npv_cube)))
        assert result.exposure is not None and set(result.greeks) == {0, 1}
        assert result.trade_ids == ["swap-payer", "european-payer"]
        assert all(np.all(np.isfinite(v)) for g in result.greeks.values() for v in g.values())

    def test_the_today_value_does_not_depend_on_the_model(self):
        """The model simulates; today's values are the engines' on today's market."""
        values = [price_portfolio(dataclasses.replace(_market_request(config=RunConfig(simulation=_cam(model=m))),
                                                      scenario_risk=True)).base_npv_per_trade for m in MODELS]
        assert values[0] == values[1]


class TestTheSensitivitySettingsReachTheGreeks:
    def test_greeks_use_the_configured_tenors_and_theta_horizon(self):
        settings = SensitivityConfig(curve_tenors=("1Y", "5Y", "10Y"), theta_days=3)
        request = _market_request(compute_greeks=True, config=RunConfig(greeks=GreeksConfig(sensitivity=settings)))
        greeks = price_portfolio(request).greeks
        direct = portfolio_sensitivities(request.trades, request.market, "USD", PricingConfig(), settings)
        default = portfolio_sensitivities(request.trades, request.market, "USD")
        assert greeks[0]["delta:discount:USD"].shape == (3,)
        for i in direct:
            for key in direct[i]:
                np.testing.assert_array_equal(greeks[i][key], direct[i][key])
        assert float(greeks[0]["theta"]) != pytest.approx(float(default[0]["theta"]), rel=1e-3)
