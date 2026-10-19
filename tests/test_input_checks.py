"""
Every input checked before any work, on the host (I-103), each refusal naming its field or trade.

Red before 2026-10-18, measured on the shared portfolio's market: a discount curve with pillars
[5, 1, 10] priced the European payer at 50,679.68 instead of 51,133.41 with no error; a NaN or
negative-time pillar priced; three times and two rates failed inside the interpolation after
the job had started; an expiry of 2028-12-01 on a 2027-01-15 -> 2029-01-15 swap passed the
configuration and raised a bare `StopIteration` when priced; and four API models dropped unknown
fields silently.

  * A curve's pillars: non-empty, as many rates as times, finite, times non-negative and
    strictly increasing; negative rates allowed (`ZeroCurveConfig`).
  * A live European's exercise leaves a whole period of each leg to enter; an expired one is
    worth nothing, as ORE builds no option for it. A Bermudan's exercise into no whole period is
    ORE's (its builder warns and prices the empty underlying, worth nothing), so it is accepted.
  * Every model of the API refuses an unknown field, TraderX's submission included.
  (The standalone calibration's notional, I-103's fourth input, went with the standalone
  calibration, I-94.)
"""
import ORE
import pytest
from pydantic import BaseModel, ValidationError

from engine.api import requests as wire
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.market_data.market import ZeroCurveConfig
from engine.pricing.cube import validate_trades
from engine.run import CamConfig, LgmConfig, PortfolioRequest, RunConfig, price_portfolio
from tests.support import portfolio as shared


class TestCurves:
    @pytest.mark.parametrize("times, rates, message", [
        ([], [], "at least one pillar"),
        ([1.0, 2.0, 5.0], [0.03, 0.034], "as many rates as times"),
        ([0.0, 1.0, 5.0], [0.03, float("nan"), 0.04], "must be finite"),
        ([0.0, 1.0, float("inf")], [0.03, 0.03, 0.04], "must be finite"),
        ([5.0, 1.0, 10.0], [0.04, 0.03, 0.046], "strictly increasing"),
        ([1.0, 1.0, 5.0], [0.03, 0.031, 0.04], "strictly increasing"),
        ([-1.0, 1.0, 5.0], [0.03, 0.03, 0.04], "non-negative"),
    ], ids=["empty", "lengths", "nan-rate", "inf-time", "unsorted", "repeated", "negative-time"])
    def test_a_curve_the_interpolation_cannot_read_is_refused_when_built(self, times, rates, message):
        with pytest.raises(ValueError, match=message):
            ZeroCurveConfig(times, rates)

    def test_negative_rates_and_a_single_pillar_are_curves(self):
        ZeroCurveConfig([0.0, 1.0, 30.0], [-0.005, -0.002, 0.01])
        ZeroCurveConfig((0.0,), (0.03,))

    def test_the_wire_names_the_curve(self):
        market = shared.market_json()
        market["currencies"]["USD"]["index_curves"][shared.INDEX]["times"] = [0.0, 2.0, 1.0, 5.0, 10.0, 30.0]
        with pytest.raises(ValueError, match=f"market.currencies.USD.index_curves.{shared.INDEX}: a zero curve"):
            wire.MarketSchema.model_validate(market).to_dataclass()


def _european(exercise, effective="2027-01-15", maturity="2029-01-15"):
    date = ORE.DateParser.parseISO
    return SwaptionConfig(notional=1e6, fixed_rate=0.03, payer=True, exercise_date=date(exercise),
                          effective_date=date(effective), maturity_date=date(maturity), trade_id="late",
                          evaluation_date=shared.ASOF)


class TestAnExpiryWithNoWholePeriod:
    def test_a_live_european_entering_no_whole_period_is_refused_naming_it_and_its_dates(self):
        """The fixed leg is annual: an expiry after its last accrual start enters nothing."""
        with pytest.raises(ValueError, match=r"trade 'late': exercise date 2028-12-01 leaves no whole period of its "
                                             r"fixed leg to exercise into \(the last accrues from 2028-01-17 to "
                                             r"2029-01-15\)"):
            validate_trades([_european("2028-12-01")], shared.market())

    def test_one_leg_without_a_whole_period_is_enough(self):
        """2028-03-01 leaves the semi-annual floating leg a whole period (from 2028-07-17), not
        the annual fixed one."""
        with pytest.raises(ValueError, match="no whole period of its fixed leg"):
            validate_trades([_european("2028-03-01")], shared.market())
        validate_trades([_european("2028-01-10")], shared.market())

    def test_it_is_a_refusal_before_any_work_not_a_stop_iteration(self):
        request = PortfolioRequest(shared.market(), [_european("2028-12-01")], RunConfig(), scenario_risk=False)
        with pytest.raises(ValueError, match="no whole period"):
            price_portfolio(request)

    @pytest.mark.slow
    def test_an_expired_european_is_worth_nothing_on_every_path(self):
        """ORE builds no option once no exercise is left (`Swaption::build`, step 6), and the
        engine reads no exercise-into coupon of it, so an expiry that entered no whole period is
        no refusal once it has passed."""
        expired = _european("2026-06-01", effective="2026-01-15", maturity="2026-07-15")
        dates = tuple(shared.ASOF + ORE.Period(m, ORE.Months) for m in (3, 6))
        simulation = CamConfig(dates=dates, base_currency="USD", ir={"USD": LgmConfig(0.03, 0.01)}, samples=8)
        result = price_portfolio(PortfolioRequest(shared.market(), [expired], RunConfig(simulation=simulation)))
        assert result.base_npv == 0.0 and float(abs(result.npv_cube).max()) == 0.0

    @pytest.mark.slow
    def test_a_bermudan_exercising_into_no_whole_period_is_oress_and_worth_nothing(self):
        """ORE's `Swaption::buildUnderlyingSwaps` warns of the empty leg and prices it: such an
        exercise is worth nothing, so a Bermudan with no other is."""
        date = ORE.DateParser.parseISO
        late = BermudanSwaptionConfig(notional=1e6, fixed_rate=0.03, payer=True, exercise_dates=[date("2028-12-01")],
                                      effective_date=date("2027-01-15"), maturity_date=date("2029-01-15"),
                                      trade_id="late-bermudan", evaluation_date=shared.ASOF)
        result = price_portfolio(PortfolioRequest(shared.market(), [late], RunConfig(), scenario_risk=False))
        assert result.base_npv == 0.0


class TestEveryModelIsStrict:
    def test_every_request_model_refuses_an_unknown_field(self):
        """Red before 2026-10-18: `ZeroCurveConfigSchema`, `CouponPeriodSchema` and TraderX's
        `EodSubmissionSchema` (and the retired standalone calibration's) dropped one."""
        from engine.api.traderx_routes import EodSubmissionSchema

        models = [cls for cls in vars(wire).values() if isinstance(cls, type) and issubclass(cls, BaseModel)
                  and cls.__module__ == wire.__name__] + [EodSubmissionSchema]
        lax = sorted(cls.__name__ for cls in models if cls.model_config.get("extra") != "forbid")
        assert lax == []

    @pytest.mark.parametrize("schema, fields", [
        (wire.ZeroCurveConfigSchema, {"times": [0.0], "rates": [0.03], "day_count": "ACT/360"}),
        (wire.CouponPeriodSchema, {"start_date": "2026-07-30", "end_date": "2027-01-30", "paymnet_date": "x"}),
    ])
    def test_an_unknown_field_is_named(self, schema, fields):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            schema.model_validate(fields)

    def test_a_traderx_submission_with_an_unknown_field_is_a_422(self, test_client):
        r = test_client.post("/eod/price", json={"bundlePath": "nowhere", "reuseExisting": False})
        assert r.status_code == 422 and "reuseExisting" in r.text

