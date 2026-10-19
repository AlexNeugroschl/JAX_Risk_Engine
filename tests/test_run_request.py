"""
The run request in Python (`engine.run.run`, I-87): one request for every analytic.

  * A run of one analytic equals the engine entry point's direct call, bit for bit: `npv`,
    `exposure` and `sensitivities` `price_portfolio`, `market_risk` `run_market_risk` (on the
    Monte Carlo scenarios drawn as `monte_carlo_scenarios` draws them), `calibration`
    `calibrate_cam`; a run of several equals each alone, and compiles nothing they would not.
  * The check before any work (`validate_run`): each analytic once; the sections an analytic
    needs; nothing accepted and then ignored (a section, a portfolio, a precision setting no
    requested analytic reads, I-59, I-95).
  * The phases of a run are the boundaries at which it can stop (`at_phase_boundaries`, I-91).
"""
import dataclasses

import jax
import numpy as np
import ORE
import pytest

from engine.calibration.cam import calibrate_cam
from engine.precision import Precision, StagePrecision
from engine.risk.market import MarketRiskRequest, MonteCarloScenarios, RateRiskFactors, monte_carlo_scenarios
from engine.risk.market import run_market_risk
from engine.run import (
    CamConfig, ExposureConfig, GreeksConfig, JamshidianEngineConfig, LgmConfig, MarketRiskConfig, ModelsConfig, PortfolioRequest,
    PricingConfig, RunConfig, RunRequest, RunRequestConfig, SimulationParameters, engine_identity, price_portfolio,
    run, validate_run,
)
from engine.run.trace import PHASES, PRICING_TRADE_PREFIX, at_phase_boundaries
from tests.support import portfolio as shared

BASKET = dict(calibration_expiries=("1Y", "2Y"), calibration_terms=("4Y", "3Y"))
TRADES = ("swap-payer", "european-payer", "bond")


def _trades(names=TRADES):
    return tuple(shared.trades()[n] for n in names)


def _models(**fields):
    return ModelsConfig(ir={"USD": LgmConfig(0.03, 0.01, **fields)})


def _simulation(samples=32):
    return SimulationParameters(dates=tuple(shared.ASOF + ORE.Period(m, ORE.Months) for m in (6, 12)),
                                samples=samples, seed=5)


def _scenarios(num=64):
    factors = RateRiskFactors.from_market(shared.market())
    return MonteCarloScenarios(factors, np.eye(factors.size) * 0.002 ** 2, 10, num, seed=3)


def _config(**sections):
    return RunRequestConfig(models=_models(**BASKET), simulation=_simulation(),
                            market_risk=MarketRiskConfig(_scenarios()), **sections)


def _request(analytics, names=TRADES, config=None, **fields):
    portfolio = _trades(names) if any(a != "calibration" for a in analytics) else ()
    config = config or _config()
    return RunRequest(shared.market(), analytics, portfolio, _only_read(config, analytics), **fields)


def _only_read(config, analytics):
    """`config` without the sections `analytics` do not read (a run refuses them)."""
    drop = {}
    if "exposure" not in analytics and "calibration" not in analytics:
        drop["models"] = None
    if "exposure" not in analytics:
        drop["simulation"] = None
    if "market_risk" not in analytics:
        drop["market_risk"] = None
    return dataclasses.replace(config, **drop)


def _cam():
    return _config().cam()


class TestOneAnalyticIsItsEntryPoint:
    def test_npv_is_price_portfolio_without_scenarios(self):
        result = run(_request(("npv",)))
        direct = price_portfolio(PortfolioRequest(shared.market(), list(_trades()), RunConfig(), scenario_risk=False))
        assert result.portfolio.base_npv_per_trade == direct.base_npv_per_trade
        assert result.portfolio.greeks is None and result.market_risk is None and result.calibration is None
        assert result.trade_ids == list(TRADES) and result.engine == engine_identity()

    def test_exposure_is_price_portfolio_with_scenarios(self):
        result = run(_request(("exposure",)))
        direct = price_portfolio(PortfolioRequest(shared.market(), list(_trades()), RunConfig(simulation=_cam())))
        np.testing.assert_array_equal(np.asarray(result.portfolio.npv_cube), np.asarray(direct.npv_cube))
        np.testing.assert_array_equal(np.asarray(result.portfolio.exposure.epe), np.asarray(direct.exposure.epe))
        assert result.portfolio.base_npv_per_trade == direct.base_npv_per_trade
        assert result.portfolio.warnings == direct.warnings

    def test_sensitivities_are_price_portfolios_greeks(self):
        result = run(_request(("sensitivities",), names=("swap-payer", "european-payer")))
        direct = price_portfolio(PortfolioRequest(shared.market(), list(_trades(("swap-payer", "european-payer"))),
                                                  RunConfig(), compute_greeks=True, scenario_risk=False))
        for i in (0, 1):
            for key, value in direct.greeks[i].items():
                np.testing.assert_array_equal(np.asarray(result.portfolio.greeks[i][key]), np.asarray(value))

    def test_market_risk_is_run_market_risk_on_the_scenarios_drawn(self):
        result = run(_request(("market_risk",)))
        scenarios = _scenarios()
        drawn = monte_carlo_scenarios(scenarios.factors, scenarios.covariance, 10, 64, seed=3)
        np.testing.assert_array_equal(scenarios.draw().shifts, drawn.shifts)
        direct = run_market_risk(MarketRiskRequest(list(_trades()), shared.market(), drawn))
        np.testing.assert_array_equal(np.asarray(result.market_risk.pnl), np.asarray(direct.pnl))
        np.testing.assert_equal(result.market_risk.risk, direct.risk)
        assert result.portfolio is None

    def test_calibration_is_calibrate_cam(self):
        result = run(_request(("calibration",)))
        direct = calibrate_cam(shared.market(), _models(**BASKET).ir)
        np.testing.assert_array_equal(np.asarray(result.calibration["USD"].sigma.values),
                                      np.asarray(direct["USD"].sigma.values))
        assert result.calibration["USD"].error == direct["USD"].error < 1e-4 and result.trade_ids == []


@pytest.mark.slow
def test_a_run_of_several_analytics_equals_each_alone_and_compiles_nothing_more():
    """The same programs: a combined run after its analytics alone builds no program (each
    analytic is its own entry point's call, `engine.run.run_request`), and a repeated run none."""
    import jax.monitoring

    every = ("npv", "exposure", "sensitivities", "market_risk", "calibration")
    alone = {a: run(_request((a,))) for a in every}
    compiles = []
    listener = lambda event, duration, **kw: compiles.append(event)  # noqa: E731
    jax.monitoring.register_event_duration_secs_listener(listener)
    try:
        together = run(_request(every))
        again = run(_request(every))
    finally:
        jax.monitoring.unregister_event_duration_listener(listener)
    assert [e for e in compiles if e == "/jax/core/compile/backend_compile_duration"] == []
    assert together.portfolio.base_npv_per_trade == alone["npv"].portfolio.base_npv_per_trade
    np.testing.assert_array_equal(np.asarray(together.portfolio.npv_cube), np.asarray(alone["exposure"].portfolio.npv_cube))
    for i, greeks in alone["sensitivities"].portfolio.greeks.items():
        for key, value in greeks.items():
            np.testing.assert_array_equal(np.asarray(together.portfolio.greeks[i][key]), np.asarray(value))
    np.testing.assert_array_equal(np.asarray(together.market_risk.pnl), np.asarray(alone["market_risk"].market_risk.pnl))
    np.testing.assert_array_equal(np.asarray(together.calibration["USD"].sigma.values),
                                  np.asarray(alone["calibration"].calibration["USD"].sigma.values))
    np.testing.assert_array_equal(np.asarray(again.portfolio.npv_cube), np.asarray(together.portfolio.npv_cube))


class TestTheCheckBeforeAnyWork:
    @pytest.mark.parametrize("analytics, message", [
        ((), "name at least one"),
        (("xva",), "['xva'] are not analytics"),
        (("npv", "npv"), "['npv'] named more than once"),
    ])
    def test_the_analytics_are_named_once_from_the_list(self, analytics, message):
        with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
            validate_run(RunRequest(shared.market(), analytics, _trades()))

    @pytest.mark.parametrize("analytic, section", [
        ("exposure", "models"), ("exposure", "simulation"), ("market_risk", "market_risk"), ("calibration", "models"),
    ])
    def test_a_section_an_analytic_needs_is_required(self, analytic, section):
        config = dataclasses.replace(_only_read(_config(), (analytic,)), **{section: None})
        with pytest.raises(ValueError, match=f"config.{section}: the {analytic} analytic needs it"):
            validate_run(RunRequest(shared.market(), (analytic,), _trades() if analytic != "calibration" else (),
                                    config))

    @pytest.mark.parametrize("analytics, change, field", [
        (("npv",), dict(models=_models()), "config.models"),
        (("npv",), dict(simulation=_simulation()), "config.simulation"),
        (("npv",), dict(market_risk=MarketRiskConfig(_scenarios())), "config.market_risk"),
        (("npv",), dict(sensitivity=GreeksConfig(method="AD")), "config.sensitivity"),
        (("npv",), dict(exposure=ExposureConfig((0.9,))), "config.exposure"),
        (("npv",), dict(precision=Precision.throughout("float32")), "config.precision"),
        (("sensitivities",), dict(precision=Precision(pricing=StagePrecision("float32"))), "config.precision"),
        (("calibration",), dict(models=_models(**BASKET), base_currency="EUR"), "config.base_currency"),
        (("calibration",), dict(models=_models(**BASKET), pricing=PricingConfig(
            european="Jamshidian", jamshidian=JamshidianEngineConfig(0.03, 0.01))), "config.pricing"),
    ])
    def test_a_section_no_requested_analytic_reads_is_refused_naming_it(self, analytics, change, field):
        """I-59's rule: an input accepted and then ignored is a defect. A precision with only
        today's values or Greeks (float64 by decision A-10) is one (I-95)."""
        config = dataclasses.replace(RunRequestConfig(), **change)
        portfolio = _trades() if analytics != ("calibration",) else ()
        with pytest.raises(ValueError, match=f"{field}: no requested analytic reads it"):
            validate_run(RunRequest(shared.market(), analytics, portfolio, config))

    def test_a_portfolio_or_a_cube_no_analytic_reads_is_refused(self):
        calibration = RunRequestConfig(models=_models(**BASKET))
        with pytest.raises(ValueError, match="portfolio: no requested analytic reads it"):
            validate_run(RunRequest(shared.market(), ("calibration",), _trades(), calibration))
        with pytest.raises(ValueError, match="include_npv_cube"):
            validate_run(RunRequest(shared.market(), ("npv",), _trades(), include_npv_cube=False))

    def test_calibration_alone_reads_only_the_calibrated_models(self):
        """The FX and equity volatilities and the correlations are the simulation's; a currency
        without a basket has nothing to calibrate."""
        ir = {"USD": LgmConfig(0.03, 0.01, **BASKET), "EUR": LgmConfig(0.02)}
        for models, field in ((ModelsConfig(ir=ir), "config.models.ir\\['EUR'\\]"),
                              (ModelsConfig(ir={"USD": ir["USD"]}, correlations={("IR:USD", "IR:EUR"): 0.3}),
                               "config.models.correlations")):
            with pytest.raises(ValueError, match=field):
                validate_run(RunRequest(shared.market(), ("calibration",), (), RunRequestConfig(models=models)))
        with pytest.raises(ValueError, match="no currency has a calibration basket"):
            validate_run(RunRequest(shared.market(), ("exposure", "calibration"), _trades(),
                                    _only_read(dataclasses.replace(_config(), models=_models()),
                                               ("exposure", "calibration"))))

    @pytest.mark.parametrize("precision, message", [
        (Precision(market=StagePrecision("float32", "float32", "float32")), "config.precision.market is not read"),
        (Precision(simulation=StagePrecision("float32", "float32", "float64")),
         "config.precision.simulation.compute='float32' is not read by market risk"),
    ])
    def test_market_risk_refuses_the_precision_it_does_not_read(self, precision, message):
        """I-95: no scenario market; shifts with no products. Refused also beside the exposure
        analytic, which reads both: market risk's precision report would state a policy it did not
        run."""
        for analytics in (("market_risk",), ("exposure", "market_risk")):
            config = _only_read(dataclasses.replace(_config(), precision=precision), analytics)
            with pytest.raises(ValueError, match=message.replace("(", r"\(")):
                validate_run(RunRequest(shared.market(), analytics, _trades(), config))

    def test_the_precision_each_analytic_reads_is_accepted(self):
        f32 = StagePrecision("float32", "float32", "float32")
        validate_run(_request(("exposure",), config=dataclasses.replace(_config(), precision=Precision.throughout("float32"))))
        validate_run(_request(("market_risk",), config=dataclasses.replace(
            _config(), precision=Precision(simulation=f32, pricing=f32))))

    def test_the_cross_asset_models_refusal_names_the_sections(self):
        config = dataclasses.replace(_config(), base_currency="EUR")
        with pytest.raises(ValueError, match="config.models, config.simulation and config.base_currency"):
            validate_run(_request(("exposure",), config=config))

    def test_a_pfe_quantile_outside_the_unit_interval_or_repeated_is_refused(self):
        """Unchecked before 2026-10-18 on the direct path: a quantile is a figure's key."""
        with pytest.raises(ValueError, match="config.exposure.pfe_quantiles"):
            ExposureConfig((0.95, 1.5))
        request = PortfolioRequest(shared.market(), list(_trades()), RunConfig(simulation=_cam()),
                                   pfe_quantiles=(0.95, 0.95))
        with pytest.raises(ValueError, match="pfe_quantiles: 0.95 is given more than once"):
            price_portfolio(request)

    def test_the_monte_carlo_scenarios_are_checked_undrawn(self):
        factors = RateRiskFactors.from_market(shared.market())
        with pytest.raises(ValueError, match="positive semi-definite"):
            MonteCarloScenarios(factors, -np.eye(factors.size), 10, 64)
        with pytest.raises(TypeError, match="ShockScenarios or MonteCarloScenarios"):
            MarketRiskConfig(scenarios=np.zeros((2, 2)))


class TestThePhasesAreBoundaries:
    def test_a_run_passes_its_phases_and_each_trades_pricing(self):
        seen = []
        with at_phase_boundaries(seen.append):
            run(_request(("exposure", "market_risk"), names=("swap-payer", "bond")))
        assert seen[:4] == ["calibration", "simulation", "pricing", f"{PRICING_TRADE_PREFIX}trade0/SwapConfig"]
        assert "scenarios" in seen and "revaluation" in seen and f"{PRICING_TRADE_PREFIX}trade1/BondConfig" in seen
        assert all(name in PHASES or name.startswith(PRICING_TRADE_PREFIX) for name in seen)

    def test_a_check_that_raises_stops_the_run_there(self):
        class Stop(Exception):
            pass

        def stop_at_simulation(name):
            if name == "simulation":
                raise Stop(name)

        with at_phase_boundaries(stop_at_simulation), pytest.raises(Stop, match="simulation"):
            run(_request(("exposure",)))

    def test_outside_the_block_nothing_is_called(self):
        seen = []
        with at_phase_boundaries(seen.append):
            pass
        run(_request(("npv",)))
        assert seen == []


def test_the_engine_identity_names_the_libraries():
    identity = engine_identity()
    assert identity.jax == jax.__version__ and identity.ore == ORE.__version__
    assert identity.revision is None or len(identity.revision) == 40
    assert identity.modified is (None if identity.revision is None else identity.modified)


def test_a_run_request_holds_trades_and_a_config_of_the_engines_types():
    with pytest.raises(TypeError, match="config must be a RunRequestConfig"):
        RunRequest(shared.market(), ("npv",), _trades(), config=RunConfig())
    with pytest.raises(TypeError, match="config.simulation must be a SimulationParameters"):
        RunRequestConfig(simulation=CamConfig(dates=_simulation().dates, base_currency="USD",
                                              ir={"USD": LgmConfig(0.03)}))
    with pytest.raises(ValueError, match="unique"):
        RunRequest(shared.market(), ("npv",), _trades(("swap-payer", "swap-payer")))
