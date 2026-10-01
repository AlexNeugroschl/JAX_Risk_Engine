"""
What every trade config is, whatever its type (roadmap 1.3):

  * I-10: it names itself. `trade_id` (ORE's `<Trade id>`) is required and non-empty, travels
    with the trade through the worker pool, and a portfolio refuses a repeated id, so results
    are never keyed by position alone.
  * I-64: it names its valuation date. `evaluation_date` is required: a trade never takes ORE's
    thread-local evaluation date, which defaults to the wall clock on a fresh thread.
  * I-63: it carries no model and no curve. The curves are the market's (named by currency
    and index) and the models the run configuration's; a trade given model parameters is a
    `TypeError`, not a second copy of a fact that can drift from the first.
"""
import dataclasses

import ORE
import pytest

from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig
from engine.instruments.treasury import BondConfig
from engine.portfolio import PortfolioRequest
from engine.portfolio.worker_pool import _freeze_trade, _thaw_trade
from tests.support import portfolio

TODAY = ORE.Date(30, 7, 2026)

#: One minimal booking per trade type, without its identity and date.
BOOKINGS = {
    SwapConfig: dict(notional=1e6, fixed_rate=0.03, payer=True, swap_tenor="5Y"),
    SwaptionConfig: dict(notional=1e6, fixed_rate=0.03, payer=True, swap_tenor="5Y",
                         forward_start=ORE.Period(1, ORE.Years)),
    BermudanSwaptionConfig: dict(notional=1e6, fixed_rate=0.03, payer=True, swap_tenor="5Y",
                                 exercise_dates=[TODAY + 400, TODAY + 800]),
    AmericanSwaptionConfig: dict(notional=1e6, fixed_rate=0.03, payer=True, swap_tenor="5Y",
                                 first_exercise_date=TODAY + 400, last_exercise_date=TODAY + 800),
    BondConfig: dict(face_amount=1e6, maturity_date=ORE.Date(30, 7, 2028)),
}
TYPES = list(BOOKINGS)

#: The fields a trade carried before roadmap 1.3: a model, or a curve by index or by value.
RETIRED_FIELDS = {"hw_a": 0.03, "hw_sigma": 0.01, "rate_factor_index": 0, "discount_curve_index": 0,
                  "forward_curve_index": 1, "initial_zero_curve": None, "index_zero_curve": None,
                  "curve_index": 0, "n_per_std": 48, "std_devs": 6.0, "exercise_time_steps_per_year": 24}


def _build(cls, **fields):
    return cls(**{**BOOKINGS[cls], **fields})


@pytest.mark.parametrize("cls", TYPES, ids=lambda c: c.__name__)
class TestEveryTradeNamesItselfAndItsDate:
    def test_without_either_it_is_refused(self, cls):
        """Before roadmap 1.3 this booked a trade on whatever ORE's global evaluation date was,
        with no identity."""
        with pytest.raises(TypeError, match="trade_id|evaluation_date"):
            _build(cls)

    def test_without_an_evaluation_date_it_is_refused(self, cls):
        ORE.Settings.instance().evaluationDate = TODAY
        with pytest.raises(TypeError, match="evaluation_date"):
            _build(cls, trade_id="t")

    def test_without_an_id_it_is_refused(self, cls):
        with pytest.raises(TypeError, match="trade_id"):
            _build(cls, evaluation_date=TODAY)

    @pytest.mark.parametrize("bad", ["", None, 7])
    def test_an_empty_or_non_string_id_is_refused(self, cls, bad):
        with pytest.raises(ValueError, match="trade_id"):
            _build(cls, evaluation_date=TODAY, trade_id=bad)

    def test_the_id_and_date_are_the_booking(self, cls):
        cfg = _build(cls, evaluation_date=TODAY, trade_id="the-trade")
        assert cfg.trade_id == "the-trade" and cfg.evaluation_date == TODAY

    def test_the_id_survives_the_worker_pool(self, cls):
        cfg = _build(cls, evaluation_date=TODAY, trade_id="the-trade")
        assert _thaw_trade(_freeze_trade(cfg)) == cfg

    @pytest.mark.parametrize("field", RETIRED_FIELDS)
    def test_a_model_or_curve_on_the_trade_is_refused(self, cls, field):
        """The market and the run configuration supply them (I-63)."""
        with pytest.raises(TypeError, match=field):
            _build(cls, evaluation_date=TODAY, trade_id="t", **{field: RETIRED_FIELDS[field]})


def test_a_portfolio_refuses_a_repeated_id():
    trades = list(portfolio.trades().values())
    twin = dataclasses.replace(trades[1], trade_id=trades[0].trade_id)
    with pytest.raises(ValueError, match=f"repeated: \\['{trades[0].trade_id}'\\]"):
        PortfolioRequest(market=portfolio.market(), trades=[trades[0], twin], scenario_risk=False)
