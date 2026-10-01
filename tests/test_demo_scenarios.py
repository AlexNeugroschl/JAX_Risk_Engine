"""
The shared demo scenarios (`demos.demo_scenarios`): the market and the simulation the demos
and many tests build on, checked directly rather than only as inputs to downstream pricers.
The market is sloped (a flat curve hides drift and convexity errors, I-42); every
configuration builds and simulates under either model, and the simulated numeraire reprices
today's discount curve.
"""
import numpy as np
import ORE
import pytest

from demos.demo_scenarios import DATES, EVAL_DATE, MODELS, PILLARS, USD_6M, demo_market, demo_simulation
from engine.models.curves import ZeroCurve, discount
from engine.simulation.config import HullWhiteConfig, LgmConfig, build_cross_asset_model, simulate


class TestDemoMarket:
    def test_both_currencies_with_their_curves_vols_fx_and_equity(self):
        market = demo_market()
        assert set(market.currencies) == {"USD", "EUR"} and market.asof == EVAL_DATE
        assert market.fx_spot("EUR", "USD") == pytest.approx(1.10)
        assert market.index_curve("USD", USD_6M).times == PILLARS
        market.swaption_vols("EUR")
        assert "AAPL" in market.equities

    def test_one_currency_has_no_fx_or_equity(self):
        market = demo_market(("USD",))
        assert set(market.currencies) == {"USD"} and not market.fx_spots and not market.equities

    @pytest.mark.parametrize("currency", ["USD", "EUR"])
    def test_every_curve_is_sloped(self, currency):
        """No flat curve: a flat curve hides model bugs (I-42)."""
        data = demo_market().currency(currency)
        for curve in (data.discount_curve, *data.index_curves.values()):
            assert len(set(curve.rates)) > 2 and curve.rates[-1] > curve.rates[0]


class TestDemoSimulation:
    @pytest.mark.parametrize("model", MODELS)
    def test_every_currency_gets_the_named_model(self, model):
        config = demo_simulation(model, samples=8)
        assert config.dates == DATES and config.base_currency == "USD"
        assert all(type(c) is MODELS[model] for c in config.ir.values())
        assert MODELS == {"HullWhite": HullWhiteConfig, "LGM": LgmConfig}

    def test_a_one_currency_simulation_has_no_cross_asset_terms(self):
        config = demo_simulation(currencies=("USD",), samples=8)
        assert set(config.ir) == {"USD"} and not config.fx_volatilities and not config.correlations

    @pytest.mark.parametrize("model", MODELS)
    def test_a_calibrated_simulation_bootstraps_each_currency(self, model):
        market = demo_market()
        fixed = build_cross_asset_model(market, demo_simulation(model, samples=8))
        calibrated = build_cross_asset_model(market, demo_simulation(model, samples=8, calibrated=True))
        for f, c in zip(fixed.ir, calibrated.ir):
            assert len(np.asarray(c.sigma.values)) == 3
            assert not np.allclose(np.asarray(c.sigma.values), np.asarray(f.sigma.values))

    @pytest.mark.parametrize("model", MODELS)
    def test_the_numeraire_reprices_todays_discount_curve(self, model):
        """E[1/N(t)] = P(0, t) on the sloped curve, for both models (the drift and the
        numeraire are curve-fitted)."""
        market = demo_market(("USD",))
        scenarios = simulate(market, demo_simulation(model, samples=2 ** 14, currencies=("USD",)))
        p0 = np.asarray(discount(ZeroCurve.from_config(market.currency("USD").discount_curve),
                                 np.asarray(scenarios.times)))
        deflated = np.mean(1.0 / np.asarray(scenarios.numeraire), axis=0)
        np.testing.assert_allclose(deflated, p0, rtol=2e-3)

    def test_the_dates_are_after_the_evaluation_date(self):
        assert all(d > EVAL_DATE for d in DATES)
        assert DATES[-1] == EVAL_DATE + ORE.Period(60, ORE.Months)


class TestTheHttpForms:
    """The HTTP demos send `demo_market_json`/`demo_simulation_json`: the same market and
    simulation as the dataclass demos, through the request schema."""

    def test_the_market_round_trips(self):
        from demos.demo_scenarios import demo_market_json
        from engine.api.market_schemas import MarketSchema
        parsed = MarketSchema.model_validate(demo_market_json(("USD", "EUR"))).to_dataclass()
        expected = demo_market()
        assert parsed.asof == expected.asof and parsed.fx_spots == expected.fx_spots
        for code in ("USD", "EUR"):
            assert parsed.currency(code) == expected.currency(code)
        assert parsed.equities == expected.equities

    @pytest.mark.parametrize("model", MODELS)
    def test_the_simulation_round_trips(self, model):
        from demos.demo_scenarios import demo_simulation_json
        from engine.api.market_schemas import CamConfigSchema
        body = demo_simulation_json(model, 8, currencies=("USD", "EUR"), calibrated=True)
        assert CamConfigSchema.model_validate(body).to_dataclass() == \
            demo_simulation(model, 8, currencies=("USD", "EUR"), calibrated=True)
