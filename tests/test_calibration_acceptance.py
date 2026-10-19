"""
ORE's calibration acceptance on every LGM bootstrap (I-73; `engine.calibration.lgm`):
`LgmBuilder::calibrate` computes the calibration error, the root mean square of the helpers'
relative price errors |market - model| / market (`getCalibrationError`), and fails the build
when it is not below the tolerance (a trade's `Tolerance`, the cross-asset model's
`BootstrapTolerance`), unless `ContinueOnCalibrationError`.

  * A volatility dip after a spike, which no non-negative variance fits (its bucket at the
    bracket's floor), is refused today naming the trade (any market of the as-of date: the
    pipeline's base value, its simulation market's build, a bump), and in the cross-asset model
    naming the currency; a tolerance above the misfit accepts it, as ORE's would.
  * A zero ATM volatility: the helper's market value is QuantLib's, 0 (it was NaN), and the
    calibration is refused, its error not finite.
  * On the paths nothing is calibrated: ORE's exposure simulation keeps the model it built on
    the as-of date (I-49), so a market whose basket could not be fitted from a later date prices.
  * The tolerances are settings with their API fields, one for each option engine (ORE's
    `Tolerance`) and one for the cross-asset model (ORE's `BootstrapTolerance`, decision A-24),
    refused when not finite and positive.

Red first (2026-10-12, on the code before): the dip was accepted at t=0 (its bucket at the floor,
unflagged), the zero volatility gave NaN market values and a σ at an end of the bracket.
"""
import dataclasses

import numpy as np
import ORE
import pytest

from engine.api.requests import LgmEngineSchema, RunRequestSchema
from engine.calibration.cam import calibrate_cam
from engine.calibration.lgm import BOOTSTRAP_TOLERANCE, CalibrationError, build_basket, market_price
from engine.market_data.curves import ZeroCurve
from engine.market_data.market import CurrencyMarket, Market, SwaptionVolSurface
from engine.market_simulation.config import CamConfig, LgmConfig, build_cross_asset_model
from engine.pricing.bermudan import bermudan_value
from engine.pricing.config import LgmSwaptionEngineConfig, PricingConfig
from engine.pricing.context import from_market
from engine.run import PortfolioRequest, RunConfig, price_portfolio
from tests.support import portfolio as shared

ENGINE = LgmSwaptionEngineConfig(n_per_std=12, std_devs=4.0)
BERMUDAN = shared.trades()["bermudan-payer-physical"]
#: The cross-asset model's tenor basket, expiries 1Y to 3Y.
BASKET = dict(calibration_expiries=("1Y", "2Y", "3Y"), calibration_terms=("5Y", "4Y", "3Y"))

#: A 2% volatility to two years, then 0.2%: the variance to three years is below the variance to
#: two, which no non-negative volatility after two years reaches.
DIP = SwaptionVolSurface(("1Y", "2Y", "3Y", "10Y"), ("1Y", "10Y"),
                         ((0.02, 0.02), (0.02, 0.02), (0.002, 0.002), (0.002, 0.002)))
ZERO = SwaptionVolSurface(("1Y", "10Y"), ("1Y", "10Y"), ((0.0, 0.0), (0.0, 0.0)))
#: Flat 0.9% from one year on, 5% at six months: every helper of the shared Bermudan (expiries
#: from 1.5 years) reads 0.9% today; a year on, read at the same time to expiry
#: (`ConstantVariance`), its first helper would read about 5% and the next 0.9%, a dip.
SPIKE = SwaptionVolSurface(("6M", "1Y", "2Y", "10Y"), ("1Y", "10Y"),
                           ((0.05, 0.05), (0.009, 0.009), (0.009, 0.009), (0.009, 0.009)))


def _market(vols: SwaptionVolSurface) -> Market:
    usd = shared.market().currency("USD")
    return Market(shared.ASOF, {"USD": CurrencyMarket(usd.discount_curve, usd.index_curves, vols)})


def _run(market: Market, trades, pricing: PricingConfig, **simulation) -> PortfolioRequest:
    config = CamConfig(**{**dict(dates=(shared.ASOF + ORE.Period(12, ORE.Months),), base_currency="USD",
                                 ir={"USD": LgmConfig(0.03, 0.01)}, samples=8), **simulation})
    return PortfolioRequest(market, trades, RunConfig(simulation=config, pricing=pricing))


class TestToday:
    """A calibration on a market of the as-of date that misses its basket is refused, naming
    the trade, as ORE's `LgmBuilder` fails without `ContinueOnCalibrationError`."""

    def test_the_shared_market_is_accepted(self):
        assert np.isfinite(bermudan_value(BERMUDAN, ENGINE, from_market(shared.market())))

    def test_a_dip_after_a_spike_is_refused_naming_the_trade(self):
        with pytest.raises(CalibrationError, match=r"bermudan-payer-physical.*not below the tolerance 0\.0001"):
            bermudan_value(BERMUDAN, ENGINE, from_market(_market(DIP)))

    def test_the_pipeline_refuses_it_before_pricing_a_path(self):
        request = _run(_market(DIP), [BERMUDAN], PricingConfig(bermudan=ENGINE))
        with pytest.raises(CalibrationError, match="bermudan-payer-physical"):
            price_portfolio(request)

    def test_a_tolerance_above_the_misfit_accepts_it(self):
        """The setting reaches the rule: ORE accepts any error below its `Tolerance`."""
        loose = dataclasses.replace(ENGINE, bootstrap_tolerance=1e6)
        assert np.isfinite(bermudan_value(BERMUDAN, loose, from_market(_market(DIP))))

    def test_a_volatility_beyond_the_bracket_is_refused_whatever_the_tolerance(self):
        """A bucket at the bracket's ceiling (2000 bp normal) is where ORE's unbounded optimizer
        would go on: a miss even under a tolerance that accepts its error, said so alone."""
        loose = dataclasses.replace(ENGINE, bootstrap_tolerance=1e6)
        beyond = SwaptionVolSurface(("1Y", "10Y"), ("1Y", "10Y"), ((0.5, 0.5), (0.5, 0.5)))
        with pytest.raises(CalibrationError, match=r"above the bracket's 0\.2") as refusal:
            bermudan_value(BERMUDAN, loose, from_market(_market(beyond)))
        assert "not below the tolerance" not in str(refusal.value)

    def test_a_zero_volatility_is_refused_its_error_not_finite(self):
        with pytest.raises(CalibrationError, match="not finite"):
            bermudan_value(BERMUDAN, ENGINE, from_market(_market(ZERO)))


class TestTheHelpersMarketValue:
    def test_a_zero_volatility_is_quantlibs_intrinsic_value_not_nan(self):
        """`bachelierBlackFormula` at a zero standard deviation: the intrinsic value, 0 for a
        helper struck at the money or out of it (it was 0/0 before 2026-10-12)."""
        curve = ZeroCurve.from_config(shared.market().currency("USD").discount_curve)
        for strike in (None, 0.02, 0.08):
            helper = build_basket(shared.ASOF, ["2Y"], ["5Y"], deal_strikes=[strike])[0]
            assert float(market_price(helper, curve, curve, 0.0)) == 0.0
            assert float(market_price(helper, curve, curve, 0.009)) > 0.0


def _simulation(**fields) -> CamConfig:
    return CamConfig(dates=(ORE.Date(30, 7, 2027),), base_currency="USD", ir={"USD": LgmConfig(0.03, 0.01, **BASKET)},
                     **fields)


class TestTheCrossAssetModel:
    def test_a_dip_is_refused_naming_the_currency(self):
        with pytest.raises(CalibrationError, match=r"the USD model of the cross-asset model"):
            build_cross_asset_model(_market(DIP), _simulation())

    def test_a_zero_volatility_is_refused(self):
        with pytest.raises(CalibrationError, match="not finite"):
            build_cross_asset_model(_market(ZERO), _simulation())

    def test_a_tolerance_above_the_misfit_accepts_it(self):
        """One tolerance for the model, as ORE's `CrossAssetModelData` holds it (A-24), in a
        run and in the calibration on its own."""
        build_cross_asset_model(_market(DIP), _simulation(bootstrap_tolerance=1e6))
        assert calibrate_cam(_market(DIP), _simulation().ir, 1e6)["USD"].sigma is not None
        with pytest.raises(CalibrationError):
            calibrate_cam(_market(DIP), _simulation().ir)


class TestThePaths:
    def test_ores_model_as_built_is_kept_so_a_later_date_needs_no_fit(self):
        """`SPIKE` fits today; a year on, read at the same time to expiry, it could not be fitted.
        ORE keeps the model it built on the as-of date (I-49), and so does the engine: the paths
        price, and nothing warns."""
        request = _run(_market(SPIKE), [BERMUDAN], PricingConfig(bermudan=ENGINE),
                       swaption_vol_decay="ConstantVariance")
        result = price_portfolio(request)
        assert np.all(np.isfinite(np.asarray(result.npv_cube))) and not result.warnings


class TestTheSetting:
    @pytest.mark.parametrize("value", [0.0, -1e-4, float("nan"), float("inf")])
    def test_a_tolerance_not_finite_and_positive_is_refused_by_name(self, value):
        with pytest.raises(ValueError, match="LgmSwaptionEngineConfig.bootstrap_tolerance"):
            LgmSwaptionEngineConfig(bootstrap_tolerance=value)
        with pytest.raises(ValueError, match="CamConfig.bootstrap_tolerance"):
            _simulation(bootstrap_tolerance=value)

    def test_the_default_is_ores_example_configurations(self):
        """ORE has no default; 1e-4 is its shared example configuration's and every example
        simulation's (decision A-24)."""
        assert LgmSwaptionEngineConfig().bootstrap_tolerance == _simulation().bootstrap_tolerance == 1e-4
        assert BOOTSTRAP_TOLERANCE == 1e-4

    def test_the_api_fields_reach_the_configurations(self):
        """A trade's engine's, and the cross-asset model's, which its exposure simulation and its
        calibration analytic both read (`config.models.bootstrap_tolerance`)."""
        assert LgmEngineSchema(bootstrap_tolerance=0.2).to_dataclass().bootstrap_tolerance == 0.2
        body = {"market": shared.market_json(), "analytics": ["exposure", "calibration"],
                "portfolio": [shared.trades_json()["swap-payer"]],
                "config": {"models": {"ir": {"USD": {"reversion": 0.03, "calibration_expiries": ["1Y"],
                                                     "calibration_terms": ["5Y"]}}, "bootstrap_tolerance": 0.2},
                           "simulation": {"dates": ["2027-07-30"]}}}
        request = RunRequestSchema.model_validate(body).to_dataclass()
        assert request.config.models.bootstrap_tolerance == request.config.cam().bootstrap_tolerance == 0.2
