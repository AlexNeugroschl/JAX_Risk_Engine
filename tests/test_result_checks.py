"""
A run request's check of its results for NaN and infinite values (`config.checks`,
`engine.run.checks`, I-127).

Unscaled FP8 products (`ProductOptions()`, the default the engine keeps as the format behaves)
turn a notional-sized operand into NaN: E4M3 holds magnitudes up to 448. Before the check, such a
run returned NaN figures without a word, and a P&L only partly NaN gave a finite VaR that was too
small. Now every run counts them per array, refuses them by default, and warns or allows them on
request.
"""
import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from engine.precision import ExecutionPolicy, Precision, StagePrecision
from engine.risk.market import RateRiskFactors
from engine.risk.market.scenarios import MonteCarloScenarios
from engine.risk.market.var_es import value_at_risk
from engine.run import MarketRiskConfig, NonFiniteResultError, ResultChecks, RunRequest, RunRequestConfig, run
from engine.run.checks import nonfinite_counts
from tests.support import portfolio as shared

TRADE = "swap-payer"


def _request(checks=ResultChecks(), product=None):
    factors = RateRiskFactors.from_market(shared.market())
    scenarios = MonteCarloScenarios(factors, np.eye(factors.size) * 0.002 ** 2, 10, 64, seed=3)
    pricing = StagePrecision(compute="float8_e4m3fn", accumulate="float32", storage="float32")
    if product is not None:
        pricing = dataclasses.replace(pricing, product=product)
    precision = Precision(pricing=pricing, execution=ExecutionPolicy(mode="allow_emulation"))
    config = RunRequestConfig(market_risk=MarketRiskConfig(scenarios), precision=precision, checks=checks)
    return RunRequest(shared.market(), ("market_risk",), (shared.trades()[TRADE],), config)


@pytest.fixture(scope="module")
def allowed():
    return run(_request(ResultChecks("allow")))


def test_a_run_with_nan_results_is_refused_by_default_naming_each_array():
    with pytest.raises(NonFiniteResultError, match=rf"market_risk\.pnl\[{TRADE}\]: 64.*config\.checks\.nonfinite"):
        run(_request())


def test_allowed_they_are_returned_and_counted(allowed):
    assert np.isnan(np.asarray(allowed.market_risk.pnl)).all()
    assert allowed.nonfinite == {f"market_risk.pnl[{TRADE}]": 64, "market_risk.portfolio_pnl": 64}
    assert allowed.warnings == []


def test_warned_they_are_returned_with_the_message():
    result = run(_request(ResultChecks("warn")))
    assert result.nonfinite == {f"market_risk.pnl[{TRADE}]": 64, "market_risk.portfolio_pnl": 64}
    assert len(result.warnings) == 1 and "128 NaN or infinite values" in result.warnings[0]


def test_finite_results_are_counted_as_none():
    from engine.precision import ProductOptions
    result = run(_request(product=ProductOptions.compensated("float8_e4m3fn")))
    assert result.nonfinite == {} and result.warnings == []
    assert np.isfinite(np.asarray(result.market_risk.pnl)).all()


def test_a_partly_nan_pnl_gives_a_finite_var_which_is_why_the_pnl_is_counted(allowed):
    """The sort puts a NaN after every gain, so the VaR of a P&L whose worst scenarios are NaN
    reads a smaller loss, finite."""
    pnl = jnp.asarray([-50.0, -40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0])
    hidden = pnl.at[:2].set(jnp.nan)
    assert float(value_at_risk(pnl[:, None], 0.9)[0]) == 50.0
    assert float(value_at_risk(hidden[:, None], 0.9)[0]) == 30.0
    result = dataclasses.replace(allowed, market_risk=dataclasses.replace(
        allowed.market_risk, pnl=hidden[:, None], portfolio_pnl=hidden))
    assert nonfinite_counts(result) == {f"market_risk.pnl[{TRADE}]": 2, "market_risk.portfolio_pnl": 2}


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="config.checks.nonfinite must be one of"):
        ResultChecks("ignore")
