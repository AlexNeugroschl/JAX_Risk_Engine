"""
`engine.portfolio.price_portfolio` equals the same pipeline orchestrated by hand (calibrate
and simulate the cross-asset model, value every trade today and on every path, deflate into
exposure profiles), trade by trade, for every instrument type under the Hull-White model; its
results follow the request's trade order; the model is calibrated per currency; the precision
of the simulation is honoured; concurrent callers are serialized.
"""
import dataclasses
import threading

import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.models.curves import ZeroCurve, discount
from engine.portfolio import (
    CamConfig, HullWhiteConfig, LgmSwaptionEngineConfig, PortfolioRequest, PortfolioResult, PrecisionConfig,
    PricingConfig, RunConfig, price_portfolio,
)
from engine.risk.exposure import netting_set_profile
from engine.simulation.config import build_cross_asset_model, simulate
from engine.valuation.portfolio import value_portfolio
from tests.support import portfolio as shared

FAST = LgmSwaptionEngineConfig(n_per_std=12, std_devs=4.0)
PRICING = PricingConfig(bermudan=FAST, american=FAST)
NAMES = ("swap-payer", "european-payer", "bermudan-payer-physical", "american-payer", "bond")
DATES = tuple(shared.ASOF + ORE.Period(m, ORE.Months) for m in (6, 12, 24))


def _simulation(samples=64, calibrated=False) -> CamConfig:
    basket = (("1Y", "2Y", "5Y"), ("9Y", "8Y", "5Y")) if calibrated else ((), ())
    return CamConfig(dates=DATES, base_currency="USD", ir={"USD": HullWhiteConfig(0.03, 0.01, *basket)},
                     samples=samples, seed=11)


def _request(names=NAMES, precision=PrecisionConfig(), **kwargs) -> PortfolioRequest:
    trades = [shared.trades()[n] for n in names]
    config = RunConfig(simulation=_simulation(), pricing=PRICING, precision=precision)
    return PortfolioRequest(market=shared.market(), trades=trades, config=config, **kwargs)


class TestPortfolioRequestFixture:
    """The conftest fixture other tests build on prices."""

    def test_fixture_prices_successfully(self, portfolio_request):
        result = price_portfolio(portfolio_request)
        assert isinstance(result, PortfolioResult)
        assert np.isfinite(result.base_npv) and result.trade_ids == ["swap-2y"]


@pytest.mark.slow
class TestPricePortfolioMatchesHandOrchestration:
    """Every trade type under the Hull-White model, through the entry point and by hand."""

    @pytest.fixture(scope="class")
    def via_entrypoint(self):
        return price_portfolio(_request())

    @pytest.fixture(scope="class")
    def manual(self):
        request = _request()
        market, simulation = request.market, request.config.simulation
        scenarios = simulate(market, simulation, build_cross_asset_model(market, simulation))
        valuation = value_portfolio(request.trades, market, scenarios, "USD", PRICING, simulation.swaption_vol_decay)
        p0 = discount(ZeroCurve.from_config(market.currency("USD").discount_curve), jnp.asarray(scenarios.times))
        exposure = netting_set_profile(valuation.cube, valuation.today, numeraire=jnp.asarray(scenarios.numeraire),
                                       discount=p0, times=scenarios.times, quantiles=request.pfe_quantiles,
                                       dates=scenarios.dates, asof=market.asof)
        return valuation, exposure

    def test_npv_cube_shape(self, via_entrypoint):
        assert np.asarray(via_entrypoint.npv_cube).shape == (64, len(DATES), len(NAMES))

    def test_npv_cube_matches_trade_by_trade(self, via_entrypoint, manual):
        for i, name in enumerate(NAMES):
            np.testing.assert_array_equal(np.asarray(via_entrypoint.npv_cube[:, :, i]),
                                          np.asarray(manual[0].cube[:, :, i]), err_msg=name)

    def test_base_npv_matches(self, via_entrypoint, manual):
        assert via_entrypoint.base_npv_per_trade == list(manual[0].today)
        assert via_entrypoint.base_npv == float(np.sum(manual[0].today))

    def test_exposure_matches(self, via_entrypoint, manual):
        for field in ("epe", "ene", "ee_b", "eee_b", "epe_b", "eepe_b"):
            np.testing.assert_array_equal(np.asarray(getattr(via_entrypoint.exposure, field)),
                                          np.asarray(getattr(manual[1], field)), err_msg=field)
        for q in via_entrypoint.exposure.pfe:
            np.testing.assert_array_equal(np.asarray(via_entrypoint.exposure.pfe[q]), np.asarray(manual[1].pfe[q]))

    def test_one_standalone_exposure_per_trade(self, via_entrypoint):
        assert len(via_entrypoint.trade_exposures) == len(NAMES)

    def test_no_warnings(self, via_entrypoint):
        """The Hull-White model's cube warnings (aged swaps, options expiring in the horizon)
        went with its pipeline: what they warned about is priced correctly now (I-04, I-43)."""
        assert via_entrypoint.warnings == []

    def test_greeks_are_none_when_not_requested(self, via_entrypoint):
        assert via_entrypoint.greeks is None


@pytest.mark.slow
class TestPricePortfolioReorderingIndependence:
    def test_shuffled_trade_order_reorders_every_output_identically(self):
        """Each column, today's value and id follow its trade (the simulation does not
        depend on the trades)."""
        names = ("swap-payer", "european-payer", "bond")
        shuffled = (names[2], names[0], names[1])
        a, b = price_portfolio(_request(names)), price_portfolio(_request(shuffled))
        for j, name in enumerate(shuffled):
            i = names.index(name)
            np.testing.assert_array_equal(np.asarray(b.npv_cube[:, :, j]), np.asarray(a.npv_cube[:, :, i]))
            assert b.base_npv_per_trade[j] == a.base_npv_per_trade[i] and b.trade_ids[j] == name


@pytest.mark.slow
class TestPricePortfolioCalibration:
    """A Hull-White model with a basket is calibrated to the market's swaption volatilities
    (the CAM's per-currency calibration, as for the LGM), not to a request field."""

    def test_the_basket_calibrates_the_model(self):
        request = _request(("swap-payer",))
        market = request.market
        calibrated = build_cross_asset_model(market, _simulation(calibrated=True))
        assert calibrated.ir[0].volatility_type == "HullWhite"
        assert not np.allclose(np.asarray(calibrated.ir[0].sigma.values), 0.01)

    def test_a_calibrated_run_prices_finite_and_moves_the_cube(self):
        base = _request(("swap-payer", "european-payer"))
        calibrated = dataclasses.replace(base, config=dataclasses.replace(
            base.config, simulation=_simulation(calibrated=True)))
        a, b = price_portfolio(base), price_portfolio(calibrated)
        assert np.all(np.isfinite(np.asarray(b.npv_cube)))
        assert a.base_npv_per_trade == b.base_npv_per_trade, "today's values do not depend on the model"
        assert not np.array_equal(np.asarray(a.npv_cube), np.asarray(b.npv_cube))


@pytest.mark.slow
class TestPricePortfolioGreeks:
    def test_compute_greeks_true_returns_per_trade_dict(self):
        result = price_portfolio(_request(("european-payer", "swap-payer"), compute_greeks=True))
        assert set(result.greeks) == {0, 1}
        assert "vega:USD" in result.greeks[0] and "vega:USD" not in result.greeks[1]


class TestPricePortfolioPrecision:
    """Until roadmap 1.4 only the simulation's precision is adjustable (I-55)."""

    def test_default_precision_is_float64(self):
        assert price_portfolio(_request(("swap-payer",))).npv_cube.dtype == jnp.float64

    @pytest.mark.slow
    def test_simulation_32_simulates_in_float32_and_still_prices(self):
        request = _request(("swap-payer", "european-payer"), precision=PrecisionConfig(simulation=32))
        scenarios = simulate(request.market, request.config.simulation, dtype=jnp.float32)
        assert scenarios.numeraire.dtype == jnp.float32
        a = price_portfolio(request)
        b = price_portfolio(_request(("swap-payer", "european-payer")))
        assert a.base_npv_per_trade == b.base_npv_per_trade, "today's values are float64 either way"
        np.testing.assert_allclose(np.asarray(a.npv_cube), np.asarray(b.npv_cube), rtol=1e-3, atol=5.0)

    @pytest.mark.parametrize("stage", ["pricing", "risk", "calibration"])
    def test_other_stages_below_64_are_refused(self, stage):
        with pytest.raises(ValueError, match=rf"config\.precision\.{stage}"):
            price_portfolio(_request(("swap-payer",), precision=PrecisionConfig(**{stage: 32})))


class TestPrecisionOverrideValidation:
    """The override classes validate every field as `PrecisionConfig` does."""

    def test_pricing_override_rejects_invalid_bits(self):
        from engine.portfolio import PricingPrecisionOverride
        with pytest.raises(ValueError):
            PricingPrecisionOverride(swap=48)

    def test_risk_override_rejects_invalid_bits(self):
        from engine.portfolio import RiskPrecisionOverride
        with pytest.raises(ValueError):
            RiskPrecisionOverride(vega=48)


@pytest.mark.slow
class TestPricePortfolioConcurrency:
    """Two requests at different simulation precisions on two threads each get their own
    values. `_PRICING_LOCK` serializes them (the x64 flag is process-global until roadmap
    1.4); a `threading.Barrier` forces overlap, and the body repeats to make a race likely."""

    NUM_REPETITIONS = 4

    def test_two_precisions_concurrently_each_get_their_own_values(self):
        names = ("swap-payer", "european-payer")
        precisions = {"a": PrecisionConfig(simulation=64), "b": PrecisionConfig(simulation=32)}
        refs = {k: price_portfolio(_request(names, precision=p)) for k, p in precisions.items()}
        assert not np.array_equal(np.asarray(refs["a"].npv_cube), np.asarray(refs["b"].npv_cube))

        for rep in range(self.NUM_REPETITIONS):
            barrier = threading.Barrier(2)
            results, errors = {}, {}

            def run(key):
                try:
                    barrier.wait(timeout=30)
                    results[key] = price_portfolio(_request(names, precision=precisions[key]))
                except Exception as exc:  # pragma: no cover - failure path
                    errors[key] = exc

            threads = [threading.Thread(target=run, args=(k,)) for k in precisions]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=120)
            assert not errors, f"rep {rep}: concurrent price_portfolio raised: {errors}"
            for key in precisions:
                np.testing.assert_array_equal(np.asarray(results[key].npv_cube), np.asarray(refs[key].npv_cube),
                                              err_msg=f"rep {rep}: thread {key} diverged")
                assert results[key].base_npv == refs[key].base_npv


class TestSingleEvaluationDate:
    """The simulation has one t=0, the market's as-of date; a trade dated differently would be
    valued on a shifted time axis. Refused before any pricing runs, naming the trade."""

    def test_a_trade_off_the_market_date_is_refused(self):
        request = _request(("swap-payer", "bond"))
        shifted = dataclasses.replace(request.trades[1], evaluation_date=shared.ASOF + 1)
        with pytest.raises(ValueError, match=r"'bond'.*not the market's as-of date"):
            price_portfolio(dataclasses.replace(request, trades=[request.trades[0], shifted]))

    def test_the_market_date_is_accepted(self):
        assert np.isfinite(price_portfolio(_request(("swap-payer", "bond"))).base_npv)
