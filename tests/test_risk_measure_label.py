"""
I-11 regressions: `PortfolioResult.measure` (and `PortfolioResultSchema.measure`) labels the
risk figures of a direct `price_portfolio` call, and is set exactly when there are risk
figures. (I-28's demo test is in tests/test_demos.py.)
"""
import json

import ORE

from demos.demo_scenarios import EVAL_DATE, demo_market, demo_simulation
from engine.api.results import PortfolioResultSchema
from engine.instruments.swap import SwapConfig
from engine.instruments.treasury import BondConfig
from engine.risk.market.var_es import ENGINE_RISK_MEASURE, RISK_MEASURE_RISK_NEUTRAL, RISK_MEASURES
from engine.run import PortfolioRequest, RunConfig, price_portfolio


def _swap() -> SwapConfig:
    return SwapConfig(notional=1_000_000.0, fixed_rate=0.03, payer=True, swap_tenor="2Y",
                      evaluation_date=EVAL_DATE, trade_id="swap")


def _bill() -> BondConfig:
    return BondConfig(face_amount=100_000.0, maturity_date=EVAL_DATE + ORE.Period(6, ORE.Months),
                      evaluation_date=EVAL_DATE, trade_id="bill")


def _request(trades, scenario_risk: bool = True) -> PortfolioRequest:
    dates = tuple(EVAL_DATE + ORE.Period(m, ORE.Months) for m in (6, 12))
    simulation = demo_simulation("HullWhite", samples=32, dates=dates, currencies=("USD",))
    return PortfolioRequest(market=demo_market(("USD",)), trades=list(trades), scenario_risk=scenario_risk,
                            config=RunConfig(simulation=simulation if scenario_risk else None))


class TestPortfolioResultStatesItsMeasure:
    """I-11: the label travels with the number on the direct path too."""

    def test_a_scenario_run_is_labelled_risk_neutral(self):
        result = price_portfolio(_request([_swap()]))
        assert result.exposure is not None
        assert result.measure == RISK_MEASURE_RISK_NEUTRAL

    def test_the_label_is_the_engines_own_statement_not_a_literal(self):
        """The result follows `ENGINE_RISK_MEASURE` and stays in the vocabulary."""
        result = price_portfolio(_request([_swap()]))
        assert result.measure == ENGINE_RISK_MEASURE
        assert result.measure in RISK_MEASURES

    def test_no_risk_figures_means_no_label(self):
        """With `scenario_risk=False` there are no risk figures, so no label."""
        result = price_portfolio(_request([_bill()], scenario_risk=False))
        assert result.exposure is None
        assert result.measure is None

    def test_the_label_is_serialized_over_http(self):
        result = price_portfolio(_request([_swap()]))
        body = json.loads(PortfolioResultSchema.from_dataclass(result).model_dump_json())
        assert body["measure"] == RISK_MEASURE_RISK_NEUTRAL

    def test_an_absent_label_serializes_as_null(self):
        result = price_portfolio(_request([_bill()], scenario_risk=False))
        body = json.loads(PortfolioResultSchema.from_dataclass(result).model_dump_json())
        assert "measure" in body
        assert body["measure"] is None
