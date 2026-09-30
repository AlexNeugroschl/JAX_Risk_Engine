"""
The run configuration (`engine.portfolio.config.RunConfig`, roadmap 1.2, I-68): one value
naming the model per currency and simulation, the engine per product, the Greeks method and
settings, and the precision per stage, with ORE's defaults.

  * The defaults are ORE's, and a request that spells them out prices bit for bit as one that
    omits the configuration. (The one-off check that the shared portfolio prices bit for bit
    as before 1.2 is recorded in docs/planning/known-issues.md, "Verification status".)
  * Each model refuses, before any work and naming the field, a configured option it does not
    implement or read. Before 1.2 the market path silently ignored `precision.pricing`/`risk`/
    `calibration`, `calibration_targets`, and a `base_currency` contradicting the simulation.
  * The bump-and-revalue settings (`GreeksConfig.sensitivity`, ORE's sensitivity.xml) reach
    the market path's Greeks; before 1.2 they were not settable from a request.
  * The configuration survives the trip to a pricing worker.
"""
import dataclasses
import pickle

import numpy as np
import ORE
import pytest

from engine.instruments.european_swaption import SwaptionConfig
from engine.portfolio import (
    HULL_WHITE_CONFIG, CamConfig, GreeksConfig, LgmConfig, LgmSwaptionEngineConfig, PortfolioRequest,
    PrecisionConfig, PricingConfig, RiskPrecisionOverride, RunConfig, SensitivityConfig, price_portfolio,
)
from engine.portfolio.market_path import validate_market_request
from engine.portfolio.request import validate_hull_white_request
from engine.portfolio.worker_pool import _freeze_trade, _thaw_trade
from engine.risk.sensitivities import portfolio_sensitivities
from tests.support import portfolio as shared

FAST = LgmSwaptionEngineConfig(n_per_std=12, std_devs=4.0)


def _cam(base="USD"):
    return CamConfig(dates=(shared.ASOF + ORE.Period(1, ORE.Years),), base_currency=base,
                     ir={base: LgmConfig(0.03, 0.01)}, samples=8)


def _market_request(names=("swap-payer", "european-payer"), **kwargs):
    trades = [shared.trades()[n] for n in names]
    return PortfolioRequest(market=shared.market(), trades=trades, scenario_risk=False, **kwargs)


def _hull_white_request(portfolio_request, **kwargs):
    """The conftest Hull-White request plus a European on its one rate factor."""
    rates = portfolio_request.market.rates
    european = SwaptionConfig(notional=1e6, fixed_rate=0.03, payer=True, rate_factor_index=0,
                              hw_a=rates.mean_reversion[0], hw_sigma=0.01,
                              initial_zero_curve=rates.initial_zero_curves[0], swap_tenor="1Y",
                              forward_start=ORE.Period(1, ORE.Years),
                              evaluation_date=portfolio_request.trades[0].evaluation_date)
    return dataclasses.replace(portfolio_request, trades=[*portfolio_request.trades, european], **kwargs)


class TestDefaults:
    def test_the_defaults_are_ores(self):
        config = RunConfig()
        assert config.simulation is None
        assert config.pricing == PricingConfig() and config.pricing.european == "Bachelier"
        assert config.greeks.method == "Bump" and config.greeks.sensitivity == SensitivityConfig()
        assert config.precision == PrecisionConfig(simulation=64, pricing=64, risk=64, calibration=64)
        assert PortfolioRequest(market=shared.market(), trades=[]).config == config

    def test_the_reporting_currency_is_the_simulations_else_usd(self):
        assert RunConfig().reporting_currency == "USD"
        assert RunConfig(base_currency="EUR").reporting_currency == "EUR"
        assert RunConfig(simulation=_cam("EUR")).reporting_currency == "EUR"
        assert RunConfig(simulation=_cam("EUR"), base_currency="EUR").reporting_currency == "EUR"

    def test_explicit_defaults_price_bit_for_bit_as_omitted_ones(self):
        explicit = RunConfig(simulation=None, pricing=PricingConfig(bermudan=LgmSwaptionEngineConfig()),
                             greeks=GreeksConfig("Bump", SensitivityConfig()), precision=PrecisionConfig(),
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
        (lambda: RunConfig(precision=64), TypeError, "RunConfig.precision must be a PrecisionConfig"),
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

    def test_a_hull_white_simulation_is_the_market_not_the_simulation(self, portfolio_request):
        with pytest.raises(TypeError, match="request's market"):
            RunConfig(simulation=portfolio_request.market)

    def test_the_request_checks_its_market_and_config_types(self):
        with pytest.raises(TypeError, match="market must be a Market"):
            PortfolioRequest(market=_cam(), trades=[])
        with pytest.raises(TypeError, match="config must be a RunConfig"):
            PortfolioRequest(market=shared.market(), trades=[], config=PricingConfig())


class TestTheMarketPathRefusesWhatItDoesNotImplement:
    @pytest.mark.parametrize("kwargs, field", [
        (dict(config=RunConfig(pricing=PricingConfig(european="Jamshidian"))), "config.pricing.european"),
        (dict(config=RunConfig(greeks=GreeksConfig(method="AD")), compute_greeks=True), "config.greeks.method"),
        (dict(config=RunConfig(precision=PrecisionConfig(pricing=32))), "config.precision.pricing"),
        (dict(config=RunConfig(precision=PrecisionConfig(risk=RiskPrecisionOverride(vega=32)))),
         "config.precision.risk"),
        (dict(config=RunConfig(precision=PrecisionConfig(calibration=32))), "config.precision.calibration"),
        (dict(calibration_targets=[]), "calibration_targets"),
    ])
    def test_refused_before_any_work_naming_the_field(self, kwargs, field):
        with pytest.raises(ValueError, match=field.replace(".", r"\.")):
            validate_market_request(_market_request(**kwargs))

    def test_an_engine_or_method_is_checked_only_where_the_run_uses_it(self):
        swap_only = _market_request(("swap-payer",), config=RunConfig(pricing=PricingConfig(european="Jamshidian"),
                                                                      greeks=GreeksConfig(method="AD")))
        validate_market_request(swap_only)

    def test_the_simulation_precision_is_accepted(self):
        request = _market_request(("swap-payer",), config=RunConfig(simulation=_cam(),
                                                                    precision=PrecisionConfig(simulation=32)))
        validate_market_request(dataclasses.replace(request, scenario_risk=True))


class TestTheSensitivitySettingsReachTheMarketPath:
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


class TestTheHullWhiteModelRefusesWhatItDoesNotImplement:
    def test_its_own_engines_are_accepted(self, portfolio_request):
        validate_hull_white_request(_hull_white_request(portfolio_request, config=HULL_WHITE_CONFIG,
                                                        compute_greeks=True))

    def test_a_request_without_europeans_or_greeks_needs_no_configuration(self, portfolio_request):
        validate_hull_white_request(portfolio_request)

    @pytest.mark.parametrize("config, compute_greeks, field", [
        (RunConfig(greeks=GreeksConfig(method="AD")), False, "config.pricing.european"),
        (RunConfig(pricing=PricingConfig(european="Jamshidian")), True, "config.greeks.method"),
        (dataclasses.replace(HULL_WHITE_CONFIG, simulation=_cam()), False, "config.simulation"),
        (dataclasses.replace(HULL_WHITE_CONFIG, base_currency="USD"), False, "config.base_currency"),
        (dataclasses.replace(HULL_WHITE_CONFIG, pricing=PricingConfig(european="Jamshidian", bermudan=FAST)),
         False, "config.pricing.bermudan"),
        (dataclasses.replace(HULL_WHITE_CONFIG, pricing=PricingConfig(european="Jamshidian", american=FAST)),
         False, "config.pricing.american"),
        (dataclasses.replace(HULL_WHITE_CONFIG, pricing=PricingConfig(european="Jamshidian", recalibrate=False)),
         False, "config.pricing.recalibrate"),
        (dataclasses.replace(HULL_WHITE_CONFIG, greeks=GreeksConfig("AD", SensitivityConfig(theta_days=2))),
         True, "config.greeks.sensitivity"),
    ])
    def test_refused_before_any_work_naming_the_field(self, portfolio_request, config, compute_greeks, field):
        request = _hull_white_request(portfolio_request, config=config, compute_greeks=compute_greeks)
        with pytest.raises(ValueError, match=field.replace(".", r"\.")):
            validate_hull_white_request(request)
        with pytest.raises(ValueError, match=field.replace(".", r"\.")):
            price_portfolio(request)


def test_the_configuration_survives_the_trip_to_a_worker():
    """The whole request is frozen (ORE dates in the CamConfig and the market as text) and
    rebuilt in the worker, re-running every validation."""
    config = RunConfig(simulation=_cam(), pricing=PricingConfig(bermudan=FAST, recalibrate=False),
                       greeks=GreeksConfig(sensitivity=SensitivityConfig(curve_tenors=("1Y", "2Y"))),
                       precision=PrecisionConfig(simulation=32, risk=RiskPrecisionOverride(theta=32)))
    request = _market_request(config=config, trade_ids=["a", "b"])
    thawed = _thaw_trade(pickle.loads(pickle.dumps(_freeze_trade(request))))
    assert isinstance(thawed, PortfolioRequest)
    assert thawed.config == config
    assert thawed.trade_ids == ["a", "b"] and thawed.market.asof == shared.ASOF
    assert [type(t) for t in thawed.trades] == [type(t) for t in request.trades]
