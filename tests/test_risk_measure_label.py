"""
I-11 and I-28 regression tests.

**I-11 -- the measure label reaches `price_portfolio`.** W0.6 put `measure`
on the EOD path's `RiskResult`, but `PortfolioResult.risk` stayed a bare
dict of `VaR_95`/`ES_95` keys, so a direct Python or HTTP caller could not
tell a risk-neutral exposure from a loss forecast. `PortfolioResult.measure`
(and `PortfolioResultSchema.measure`) now carry it, set exactly when there
are risk figures for it to describe.

**I-28 -- `python -m engine.risk.var_es` runs.** Its demo built a
`SwapConfig` without `evaluation_date`, so the swap scheduled off ORE's
wall-clock "today" while `SWAP_DEMO_MATURITIES` stayed pinned to
`EVAL_DATE`; the pillar-alignment check refused it on any day but
2026-07-30.

Both classes fail against the pre-fix code.
"""
import json
import subprocess
import sys
from pathlib import Path

from engine.api.schemas import PortfolioResultSchema
from engine.instruments.swap import SwapConfig
from engine.instruments.treasury import BondConfig
from engine.portfolio import PortfolioRequest, derive_maturity_pillars, price_portfolio
from engine.risk.var_es import ENGINE_RISK_MEASURE, RISK_MEASURE_RISK_NEUTRAL, RISK_MEASURES
from engine.simulation.demo_scenarios import EVAL_DATE
from engine.simulation.market_model import EquityConfig, RatesConfig, SimulationConfig, ZeroCurveConfig

import ORE

REPO_ROOT = Path(__file__).resolve().parent.parent
FLAT_3PCT = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[0.03] * 6)


def _swap() -> SwapConfig:
    return SwapConfig(
        notional=1_000_000.0, fixed_rate=0.03, payer=True,
        discount_curve_index=0, forward_curve_index=0,
        swap_tenor="2Y", evaluation_date=EVAL_DATE,
    )


def _bill() -> BondConfig:
    return BondConfig(
        face_amount=100_000.0, maturity_date=EVAL_DATE + ORE.Period(6, ORE.Months),
        initial_zero_curve=FLAT_3PCT, evaluation_date=EVAL_DATE,
    )


def _request(trades, **kwargs) -> PortfolioRequest:
    trades = list(trades)
    market = SimulationConfig(
        time_grid=[0.0, 0.5, 1.0],
        scenarios=32,
        equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0],
                              rate_mapping=[[0.0]]),
        rates=RatesConfig(
            initial_rates=[0.03], theta=[0.03], mean_reversion=[0.03],
            maturities=derive_maturity_pillars(trades, EVAL_DATE),
            initial_zero_curves=[FLAT_3PCT],
        ),
        joint_covariance=[[0.04, 0.0], [0.0, 0.01 ** 2]],
    )
    return PortfolioRequest(market=market, trades=trades, **kwargs)


class TestPortfolioResultStatesItsMeasure:
    """I-11: the label travels with the number on the direct path too."""

    def test_a_scenario_run_is_labelled_risk_neutral(self):
        result = price_portfolio(_request([_swap()]))
        assert "VaR_95" in result.risk
        assert result.measure == RISK_MEASURE_RISK_NEUTRAL

    def test_the_label_is_the_engines_own_statement_not_a_literal(self):
        """If the engine ever gains a second measure, this result must follow
        `ENGINE_RISK_MEASURE`, and it must stay inside the vocabulary."""
        result = price_portfolio(_request([_swap()]))
        assert result.measure == ENGINE_RISK_MEASURE
        assert result.measure in RISK_MEASURES

    def test_no_risk_figures_means_no_label(self):
        """`scenario_risk=False` returns an empty `risk`; a measure label
        would then describe numbers that do not exist."""
        result = price_portfolio(_request([_bill()], scenario_risk=False))
        assert result.risk == {}
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


class TestVarEsDemoRuns:
    """I-28: the documented command runs to completion.

    The pre-fix demo passes only when the wall clock reads 2026-07-30, so
    this fails against it on every other day.
    """

    def test_python_m_engine_risk_var_es_exits_cleanly(self):
        proc = subprocess.run(
            [sys.executable, "-m", "engine.risk.var_es"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert "Base (t=0) NPV:" in proc.stdout
        assert "VaR_95:" in proc.stdout
