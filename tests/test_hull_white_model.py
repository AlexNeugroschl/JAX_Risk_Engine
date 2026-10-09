"""
The Hull-White model as an option of the run configuration (decision A-1): a
`HullWhiteConfig` per currency in `CamConfig.ir` (ORE's `<LGM>` with `VolatilityType
HullWhite`), simulated by the cross-asset model and valued by the same pipeline as the default
LGM model. The model-level identities (bond prices against QuantLib's `HullWhite`, zeta,
martingales for both parametrizations) are in tests/test_cam.py; this module holds its
calibration and the regressions of the defects the Hull-White model had before, each on a
curve rising from 3% to 5%.
"""
import dataclasses

import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.calibration.cam import calibrate_cam
from engine.calibration.ore_lgm import basket_vols, build_basket, price_pair
from engine.market_data.curves import ZeroCurve
from engine.market_data.market import CurrencyMarket, Market, SwaptionVolSurface, ZeroCurveConfig, index_name
from engine.market_simulation.config import HullWhiteConfig, LgmConfig
from engine.models.lgm import hull_white_zeta, zeta as hagan_zeta

ASOF = ORE.Date(30, 7, 2026)
PILLARS = [0.0, 1.0, 2.0, 5.0, 10.0, 30.0]
#: 3% -> 5%: the curve on which the Hull-White model's old simulation missed by 4-9% (I-42).
DISC = [0.030, 0.030, 0.034, 0.040, 0.046, 0.050]
INDEX_RATES = [0.034, 0.034, 0.038, 0.044, 0.049, 0.052]
INDEX = index_name("USD", 6)
VOLS = SwaptionVolSurface(("1Y", "2Y", "5Y", "10Y"), ("1Y", "5Y", "10Y"),
                          ((0.0080, 0.0088, 0.0090), (0.0085, 0.0091, 0.0093), (0.0090, 0.0093, 0.0094),
                           (0.0092, 0.0094, 0.0096)))
BASKET = (("1Y", "2Y", "5Y"), ("9Y", "8Y", "5Y"))


def market() -> Market:
    return Market(ASOF, {"USD": CurrencyMarket(ZeroCurveConfig(PILLARS, DISC),
                                               {INDEX: ZeroCurveConfig(PILLARS, INDEX_RATES)}, VOLS)})


class TestCalibration:
    """A Hull-White currency is bootstrapped to the same `CalibrationSwaptions` helpers as an
    LGM one; its short-rate volatility reprices every helper (ORE's bootstrap of the adaptor
    solves exactly that)."""

    @pytest.mark.parametrize("reversion", [0.03, 0.1, 0.0])
    def test_every_helper_is_repriced_by_the_short_rate_volatility(self, reversion):
        calibrated = calibrate_cam(market(), {"USD": HullWhiteConfig(reversion, 0.01, *BASKET)})["USD"]
        basket = build_basket(ASOF, list(BASKET[0]), list(BASKET[1]))
        disc, index = ZeroCurve.from_config(market().currency("USD").discount_curve), \
            ZeroCurve.from_config(market().index_curve("USD", INDEX))
        vols = basket_vols(basket, VOLS, ASOF)
        for helper, vol in zip(basket, vols):
            zeta = hull_white_zeta(reversion, calibrated.sigma, jnp.asarray(helper.expiry_time))
            market_value, model_value = price_pair(helper, disc, index, vol, reversion, zeta)
            assert float(model_value) == pytest.approx(float(market_value), rel=1e-12)

    def test_it_has_the_lgm_calibrations_zeta_at_every_expiry_and_another_volatility(self):
        hull_white = calibrate_cam(market(), {"USD": HullWhiteConfig(0.05, 0.01, *BASKET)})["USD"].sigma
        hagan = calibrate_cam(market(), {"USD": LgmConfig(0.05, 0.01, *BASKET)})["USD"].sigma
        expiries = jnp.asarray([b.expiry_time for b in build_basket(ASOF, list(BASKET[0]), list(BASKET[1]))])
        np.testing.assert_allclose(np.asarray(hull_white_zeta(0.05, hull_white, expiries)),
                                   np.asarray(hagan_zeta(hagan, expiries)), rtol=1e-13)
        np.testing.assert_array_equal(np.asarray(hull_white.times), np.asarray(hagan.times))
        # The short rate's volatility decays against alpha: sigma = alpha e^{-a t} bucket-wise.
        assert np.all(np.asarray(hull_white.values)[1:] < np.asarray(hagan.values)[1:])

    def test_at_zero_reversion_it_is_the_lgm_calibration(self):
        hull_white = calibrate_cam(market(), {"USD": HullWhiteConfig(0.0, 0.01, *BASKET)})["USD"].sigma
        hagan = calibrate_cam(market(), {"USD": LgmConfig(0.0, 0.01, *BASKET)})["USD"].sigma
        np.testing.assert_allclose(np.asarray(hull_white.values), np.asarray(hagan.values), rtol=1e-15)


# =============================================================================
# Regressions of the defects the Hull-White model had as a separate pipeline (before
# 2026-10-01), on the 3% -> 5% curve. Each was measured on the code before then
# (docs/planning/known-issues.md, ledger); the per-path ORE comparisons of the shared pricers
# on Hull-White paths are tests/test_pricing.py (parametrized over both models).
# =============================================================================
from engine.instruments.european_swaption import SwaptionConfig  # noqa: E402
from engine.instruments.swap import SwapConfig  # noqa: E402
from engine.instruments.treasury import BondConfig  # noqa: E402
from engine.market_data.curves import discount  # noqa: E402
from engine.market_simulation.config import CamConfig, build_cross_asset_model, simulate  # noqa: E402
from engine.pricing.context import from_market  # noqa: E402
from engine.pricing.cube import value_portfolio, value_today  # noqa: E402
from engine.pricing.european import european_value  # noqa: E402
from engine.risk.market.var_es import compute_risk_metrics  # noqa: E402
from engine.run import (  # noqa: E402
    JamshidianEngineConfig, PortfolioRequest, PricingConfig, RunConfig, price_portfolio,
)

HW = HullWhiteConfig(0.03, 0.01)
DATES = tuple(ASOF + ORE.Period(y, ORE.Years) for y in (1, 2, 3))
#: Monthly simulation-market tenors to 11Y. A path curve exists at its tenors only,
#: log-linear in between, as ORE's `ScenarioSimMarket` holds it; with the default tenors (7Y,
#: 10Y, ...) a 10Y bond seen from t = 1y is read between nodes and its deflated price carries
#: that interpolation (0.3% here), which is not the model's.
MONTHLY = tuple(f"{m}M" for m in range(1, 133)) + ("15Y", "20Y", "30Y")


def hull_white_run(samples=2 ** 13, model=HW, dates=DATES, tenors=MONTHLY):
    config = CamConfig(dates=dates, base_currency="USD", ir={"USD": model}, samples=samples, seed=29,
                       curve_tenors=tenors)
    return config, simulate(market(), config)


def p0(times):
    return np.asarray(discount(ZeroCurve.from_config(market().currency("USD").discount_curve), jnp.asarray(times)))


def assert_martingale(deflated, today):
    """E[V(t)/N(t)] = V(0) on every date, within four standard errors."""
    deflated = np.asarray(deflated)
    error = np.abs(deflated.mean(axis=0) - today)
    standard_error = deflated.std(axis=0, ddof=1) / np.sqrt(deflated.shape[0])
    assert np.all(error <= 4 * standard_error + 1e-12 * abs(today)), (error, standard_error)


class TestCurveFittedDrift:
    """I-42: the old simulation evolved the short rate toward a constant theta while pricing
    bonds with the curve-fitted A(t,T); deflated bond prices missed today's curve by 4.2% (5y)
    and 8.8% (10y) at t = 2y on this curve. The model is now ORE's LGM-form Hull-White,
    fitted to the curve by construction."""

    @pytest.fixture(scope="class")
    def run(self):
        return hull_white_run()

    @pytest.mark.parametrize("maturity_years", [5, 10])
    def test_deflated_zero_bonds_are_martingales(self, run, maturity_years):
        config, scenarios = run
        bill = BondConfig(face_amount=1.0, maturity_date=ASOF + ORE.Period(maturity_years, ORE.Years),
                          evaluation_date=ASOF, trade_id="zero")
        valuation = value_portfolio([bill], market(), scenarios, "USD")
        assert_martingale(np.asarray(valuation.cube[:, :, 0]) / np.asarray(scenarios.numeraire), valuation.today[0])

    def test_a_deflated_forward_starting_swap_is_a_martingale(self, run):
        """A trade with no flow before the horizon: its whole value is carried by the curve."""
        config, scenarios = run
        swap = SwapConfig(notional=1e6, fixed_rate=0.045, payer=True, effective_date=ORE.Date(2, 8, 2029),
                          maturity_date=ORE.Date(2, 8, 2036), evaluation_date=ASOF, trade_id="forward-swap")
        valuation = value_portfolio([swap], market(), scenarios, "USD")
        assert_martingale(np.asarray(valuation.cube[:, :, 0]) / np.asarray(scenarios.numeraire), valuation.today[0])


class TestExactNumeraire:
    """I-45: the old numeraire was a bank account accrued at the left point of each step,
    biased by the rate's change over the step. It is now the model's exact LGM numeraire."""

    def test_the_numeraire_is_the_models_closed_form_on_every_path(self):
        config, scenarios = hull_white_run(samples=64)
        model = build_cross_asset_model(market(), config).ir[0]
        t = np.asarray(scenarios.times)
        z = np.asarray(scenarios.states[:, :, 0])
        a, sigma = model.reversion, float(np.asarray(model.sigma.values)[0])
        H = (1 - np.exp(-a * t)) / a
        zeta = sigma ** 2 * np.expm1(2 * a * t) / (2 * a)
        expected = np.exp(H * z + 0.5 * H ** 2 * zeta) / p0(t)
        np.testing.assert_allclose(np.asarray(scenarios.numeraire), expected, rtol=1e-13)

    def test_the_expected_inverse_numeraire_is_todays_discount_factor(self):
        """E[1/N(t)] = P(0, t), whatever the grid spacing (one-year steps here)."""
        _config, scenarios = hull_white_run()
        inverse = np.mean(1.0 / np.asarray(scenarios.numeraire), axis=0)
        np.testing.assert_allclose(inverse, p0(scenarios.times), rtol=2e-3)


class TestEuropeansOnTheMarketVolatility:
    """I-46: the Hull-White pipeline priced Europeans with Jamshidian on the trade's own
    `hw_sigma`, so its NPV was not the market's and it had no Vega. The default European
    engine is ORE's Bachelier on the market's volatility, whichever model simulates; the
    Jamshidian engine is a configured option with its own model."""

    EUROPEAN = SwaptionConfig(notional=1e6, fixed_rate=0.044, payer=True, swap_tenor="5Y",
                              forward_start=ORE.Period(1, ORE.Years), evaluation_date=ASOF, trade_id="european")

    def _run(self, pricing=PricingConfig(), compute_greeks=False):
        config, _ = hull_white_run(samples=64)
        return price_portfolio(PortfolioRequest(market=market(), trades=[self.EUROPEAN], compute_greeks=compute_greeks,
                                                config=RunConfig(simulation=config, pricing=pricing)))

    def test_the_default_engine_is_bachelier_on_the_market_vols_with_vega(self):
        result = self._run(compute_greeks=True)
        assert result.base_npv_per_trade[0] == float(european_value(self.EUROPEAN, from_market(market())))
        assert np.any(np.asarray(result.greeks[0]["vega:USD"]) > 0)

    def test_jamshidian_is_an_option_with_its_own_model(self):
        jamshidian = PricingConfig(european="Jamshidian", jamshidian=JamshidianEngineConfig(0.03, 0.01))
        assert self._run(jamshidian).base_npv_per_trade[0] != self._run().base_npv_per_trade[0]

    def test_jamshidian_on_its_own_model_is_a_martingale(self):
        """The Jamshidian engine with the simulated model's (a, sigma) is that model's own
        price on each path, so deflated it is a martingale before expiry (3y here)."""
        option = SwaptionConfig(notional=1e6, fixed_rate=0.044, payer=True, swap_tenor="5Y",
                                forward_start=ORE.Period(4, ORE.Years), evaluation_date=ASOF, trade_id="european")
        pricing = PricingConfig(european="Jamshidian", jamshidian=JamshidianEngineConfig(HW.reversion, HW.volatility))
        _config, scenarios = hull_white_run()
        valuation = value_portfolio([option], market(), scenarios, "USD", pricing)
        assert_martingale(np.asarray(valuation.cube[:, :, 0]) / np.asarray(scenarios.numeraire), valuation.today[0])


class TestOneModelOnePipeline:
    """I-44, I-47, I-62, I-68: the Hull-White model only simulates; every trade is valued by
    the shared engines (a Bermudan by the LGM grid engine calibrated to its own co-terminal
    basket, on device, per path), so today's values do not depend on the model and the
    choice is one field of the run configuration."""

    def test_todays_values_do_not_depend_on_the_model(self):
        from tests.support import portfolio as shared
        trades = list(shared.trades().values())
        fast = PricingConfig(bermudan=dataclasses.replace(shared.ENGINE, n_per_std=16),
                             american=dataclasses.replace(shared.ENGINE, n_per_std=16))
        values = []
        for model in (LgmConfig(0.03, 0.01), HullWhiteConfig(0.03, 0.01)):
            config = CamConfig(dates=(shared.ASOF + ORE.Period(1, ORE.Years),), base_currency="USD",
                               ir={"USD": model}, samples=4)
            request = PortfolioRequest(market=shared.market(), trades=trades,
                                       config=RunConfig(simulation=config, pricing=fast))
            values.append(price_portfolio(request).base_npv_per_trade)
        assert values[0] == values[1] == value_today(trades, shared.market(), "USD", fast)


class TestPaidFlowsAndMaturity:
    """I-04 (the Hull-White half): the old cube kept every paid flow and grew it, so a 3Y
    payer swap had a mean NPV of -9,852 at t = 4y, after maturity. On the shared legs a paid
    flow drops out and a matured trade is worth exactly 0 on every path."""

    def test_a_matured_swap_is_worth_zero_on_every_path(self):
        swap = SwapConfig(notional=1e6, fixed_rate=0.035, payer=True, swap_tenor="2Y", evaluation_date=ASOF,
                          trade_id="swap")
        _config, scenarios = hull_white_run(samples=256)
        cube = np.asarray(value_portfolio([swap], market(), scenarios, "USD").cube[:, :, 0])
        assert np.all(cube[:, 2] == 0.0) and np.all(cube[:, 0] != 0.0)


class TestBondsOnEveryPath:
    """I-24: bonds were refused on the Hull-White model with scenario risk (a broadcast of the
    t=0 value gave VaR 0.00 and ES NaN for a $100k bill). A bond is priced on every path."""

    def test_a_bonds_var_and_es_are_positive_and_finite(self):
        """At one year (later the bill is worth more than today on every path: pull to par,
        so no loss and VaR 0 is right there)."""
        bill = BondConfig(face_amount=100_000.0, maturity_date=ASOF + ORE.Period(5, ORE.Years), evaluation_date=ASOF,
                          trade_id="bill")
        config, _ = hull_white_run(samples=1024)
        result = price_portfolio(PortfolioRequest(market=market(), trades=[bill], config=RunConfig(simulation=config)))
        metrics = compute_risk_metrics(result.npv_cube[:, :1], result.base_npv, percentiles=(0.99,))
        assert float(metrics["VaR_99"][0]) > 0 and np.isfinite(float(metrics["ES_99"][0]))
        assert float(metrics["ES_99"][0]) >= float(metrics["VaR_99"][0])
