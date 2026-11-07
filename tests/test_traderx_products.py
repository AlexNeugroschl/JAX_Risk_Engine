"""
The TraderX fork's products and fixes (R1-R6 of TraderX's
`coordination/alex-engine-requests-20261106.md`), each against a value the engine did not
compute: the requests' ORE/QuantLib expected values (`expected_values.py` beside that file, run
in this environment's ORE wheel; their numbers are pinned here) or QuantLib built inline.

The market is the requests' synthetic one: flat continuous ACT/365F zero rates, USD discount
4.0%, index 4.1%, as of 2026-10-26.
"""
import time

import numpy as np
import ORE
import pytest

from engine.risk.market.run import _in_reporting_currency
from tests.support.runs import request_of, run_in_process

ASOF = "2026-10-26"
PILLARS = [0.0, 1.0, 2.0, 5.0, 10.0, 30.0]
TENORS = ["1Y", "2Y", "3Y", "5Y", "7Y", "10Y"]

#: `expected_values.py` (TraderX coordination/alex-engine-requests-20261106/), ORE 1.8.16.0.
R1_EQUITY_NPV = 15000.0
R1_EQUITY_VAR99_10D = 1946.9173
R2_OPTION_NPV_ONE_CONTRACT = 720.17779
R3_SOFR_OIS_NPV = -32453.801
R4_CORPORATE_DIRTY_NPV = 1049185.363
R4_TREASURY_DISCOUNTED_NPV = 1145157.509


def flat(rate):
    return {"times": PILLARS, "rates": [rate] * len(PILLARS)}


def market(**currency):
    usd = {"discount_curve": flat(0.04), "index_curves": {"USD-SIMINDEX-12M": flat(0.041)},
           "swaption_vols": {"option_tenors": TENORS, "swap_tenors": TENORS, "vols": [[0.009] * 6] * 6}}
    return {"asof": ASOF, "currencies": {"USD": {**usd, **currency}},
            "equities": {"AAPL": {"currency": "USD", "spot": 150.0, "volatility": 0.30,
                                  "dividend_curve": {"times": [0.0, 1.0], "rates": [0.005, 0.005]}}}}


EQUITY = {"trade_type": "equity", "trade_id": "eq", "symbol": "AAPL", "quantity": 100}
OPTION = {"trade_type": "equity_option", "trade_id": "opt", "symbol": "AAPL", "option_type": "Call", "strike": 150,
          "expiry_date": "2026-12-18", "quantity": 1, "multiplier": 100}
SOFR = {"trade_type": "swap", "trade_id": "sofr", "notional": 1e7, "fixed_rate": 0.042, "payer": True,
        "effective_date": "2026-10-28", "maturity_date": "2031-10-28", "index_tenor_months": 12,
        "accrual_day_count": "ACT/360", "overnight_index": "USD-SOFR"}


def corporate(**fields):
    dates, d = [], ORE.Date(24, 1, 2036)
    while d >= ORE.Date(24, 7, 2026):
        dates.append(d.ISO())
        d = d - ORE.Period(6, ORE.Months)
    dates.reverse()
    return {"trade_type": "bond", "trade_id": "gs", "face_amount": 1e6, "maturity_date": "2036-01-24",
            "coupon_rate": 0.0575, "coupon_schedule": [{"start_date": a, "end_date": b} for a, b in zip(dates, dates[1:])],
            "accrual_day_count": "30/360", "credit_spread": 0.012, **fields}


def npv(*trades, **config):
    body = {"market": market(), "portfolio": list(trades), "analytics": ["npv"]}
    if config:
        body["config"] = config
    return run_in_process(body).portfolio.base_npv_per_trade


def ql_setup():
    asof = ORE.Date(26, 10, 2026)
    ORE.Settings.instance().evaluationDate = asof
    a365 = ORE.Actual365Fixed()
    curve = lambda r: ORE.YieldTermStructureHandle(ORE.FlatForward(asof, r, a365, ORE.Continuous))  # noqa: E731
    return asof, a365, curve


# --- R1: cash equities ----------------------------------------------------------------------------

def test_r1_equity_npv_is_units_times_spot():
    assert npv(EQUITY)[0] == pytest.approx(R1_EQUITY_NPV, rel=1e-12)


def test_r1_equity_var_is_the_lognormal_spot_shock():
    factors = ["equity:AAPL/spot"] + [f"discount:USD/{t:g}y" for t in PILLARS]
    covariance = np.zeros((len(factors), len(factors)))
    covariance[0, 0] = 0.30 ** 2 * 10 / 252
    body = {"market": {**market(), "currencies": {"USD": {"discount_curve": flat(0.04)}}},
            "portfolio": [EQUITY], "analytics": ["market_risk"], "outputs": {"pnl": "inline"},
            "config": {"market_risk": {"scenarios": {"source": "monte-carlo", "factors": factors,
                                                     "covariance": covariance.tolist(), "horizon_days": 10,
                                                     "num_scenarios": 8192, "seed": 11}}}}
    result = run_in_process(body).market_risk
    assert result.risk_factors[0] == "equity:AAPL/spot"
    assert result.risk["VaR_99"] == pytest.approx(R1_EQUITY_VAR99_10D, rel=0.01)


def test_r1_an_equity_the_market_lacks_is_refused():
    with pytest.raises(KeyError, match="no equity 'MSFT'"):
        npv({**EQUITY, "symbol": "MSFT"})


def test_r1_an_equity_spot_factor_missing_from_the_scenarios_is_refused():
    body = {"market": market(), "portfolio": [EQUITY], "analytics": ["market_risk"],
            "config": {"market_risk": {"scenarios": {"source": "monte-carlo",
                                                     "factors": [f"discount:USD/{t:g}y" for t in PILLARS],
                                                     "covariance": np.diag([1e-8] * 6).tolist(), "horizon_days": 10,
                                                     "num_scenarios": 64}}}}
    from engine.run.run_request import validate_run
    with pytest.raises(ValueError, match="equity:AAPL"):
        validate_run(request_of(body))


def test_equities_are_not_priced_on_paths_or_as_greeks():
    from engine.run.run_request import validate_run
    body = {"market": market(), "portfolio": [EQUITY], "analytics": ["sensitivities"]}
    with pytest.raises(ValueError, match="npv and market_risk only"):
        validate_run(request_of(body))


# --- R2: listed equity options --------------------------------------------------------------------

def test_r2_american_call_matches_quantlib_finite_differences():
    assert npv(OPTION)[0] == pytest.approx(R2_OPTION_NPV_ONE_CONTRACT, rel=2e-4)


@pytest.mark.parametrize("option_type,exercise", [("Put", "American"), ("Put", "European"), ("Call", "European")])
def test_r2_against_quantlib(option_type, exercise):
    asof, a365, curve = ql_setup()
    process = ORE.BlackScholesMertonProcess(
        ORE.QuoteHandle(ORE.SimpleQuote(150.0)), curve(0.005), curve(0.04),
        ORE.BlackVolTermStructureHandle(ORE.BlackConstantVol(asof, ORE.NullCalendar(), 0.30, a365)))
    expiry = ORE.Date(18, 12, 2026)
    kind = ORE.Option.Call if option_type == "Call" else ORE.Option.Put
    if exercise == "American":
        option = ORE.VanillaOption(ORE.PlainVanillaPayoff(kind, 150.0), ORE.AmericanExercise(asof, expiry))
        option.setPricingEngine(ORE.FdBlackScholesVanillaEngine(process, 2000, 2000))
    else:
        option = ORE.VanillaOption(ORE.PlainVanillaPayoff(kind, 150.0), ORE.EuropeanExercise(expiry))
        option.setPricingEngine(ORE.AnalyticEuropeanEngine(process))
    trade = {**OPTION, "option_type": option_type, "exercise": exercise, "quantity": -3}
    assert npv(trade)[0] == pytest.approx(-300 * option.NPV(), rel=1e-4)


def test_r2_option_on_its_expiry_date_is_intrinsic_and_after_it_nothing():
    assert npv({**OPTION, "strike": 140, "expiry_date": ASOF})[0] == pytest.approx(1000.0)
    assert npv({**OPTION, "expiry_date": "2026-10-23"})[0] == 0.0


def test_r2_option_var_moves_with_spot_and_holds_the_volatility():
    factors = ["discount:USD/0y", "discount:USD/1y", "discount:USD/2y", "discount:USD/5y", "discount:USD/10y",
               "discount:USD/30y", "equity:AAPL/spot"]
    covariance = np.diag([1e-6] * 6 + [0.30 ** 2 * 10 / 252])
    body = {"market": {**market(), "currencies": {"USD": {"discount_curve": flat(0.04)}}},
            "portfolio": [OPTION, EQUITY], "analytics": ["npv", "market_risk"], "outputs": {"pnl": "inline"},
            "config": {"market_risk": {"scenarios": {"source": "monte-carlo", "factors": factors,
                                                     "covariance": covariance.tolist(), "horizon_days": 10,
                                                     "num_scenarios": 256}}}}
    result = run_in_process(body)
    mr = result.market_risk
    assert mr.base_npv_per_trade == pytest.approx(result.portfolio.base_npv_per_trade, rel=1e-12)
    pnl = np.asarray(mr.pnl)
    # a long call gains with the stock: its P&L ranks as the shares' do
    assert np.corrcoef(pnl[:, 0], pnl[:, 1])[0, 1] > 0.95
    assert any("volatility risk is not in this VaR/ES" in w and "opt" in w for w in mr.warnings)


# --- R3: overnight-index swaps and swaptions ------------------------------------------------------

def test_r3_sofr_ois_matches_quantlib():
    assert npv(SOFR)[0] == pytest.approx(R3_SOFR_OIS_NPV, abs=0.01)


def test_r3_act360_without_an_overnight_index_is_refused():
    with pytest.raises(ValueError, match="overnight_index"):
        npv({k: v for k, v in SOFR.items() if k != "overnight_index"})


def test_r3_an_index_in_another_currency_is_refused():
    with pytest.raises(ValueError, match="EUR index"):
        npv({**SOFR, "overnight_index": "EUR-ESTR"})


def test_r3_a_coupon_in_progress_is_refused():
    with pytest.raises(ValueError, match="in progress"):
        npv({**SOFR, "effective_date": "2026-09-28", "maturity_date": "2031-09-29"})


def _ql_ois(effective, maturity, index):
    _, _, curve = ql_setup()
    calendar = ORE.UnitedStates(ORE.UnitedStates.GovernmentBond)
    schedule = ORE.Schedule(effective, maturity, ORE.Period(1, ORE.Years), calendar, ORE.ModifiedFollowing,
                            ORE.ModifiedFollowing, ORE.DateGeneration.Forward, False)
    swap = ORE.OvernightIndexedSwap(ORE.OvernightIndexedSwap.Payer, 1e7, schedule, 0.042, ORE.Actual360(),
                                    ORE.Sofr(curve(0.041)), 0.0, 2, ORE.ModifiedFollowing, calendar)
    swap.setPricingEngine(ORE.DiscountingSwapEngine(curve(0.04)))
    return swap, curve


def test_r3_sofr_european_swaption_matches_quantlib_bachelier():
    swap, curve = _ql_ois(ORE.Date(28, 10, 2027), ORE.Date(28, 10, 2032), "USD-SOFR")
    option = ORE.Swaption(swap, ORE.EuropeanExercise(ORE.Date(26, 10, 2027)))
    option.setPricingEngine(ORE.BachelierSwaptionEngine(curve(0.04), ORE.QuoteHandle(ORE.SimpleQuote(0.009))))
    trade = {**SOFR, "trade_type": "european_swaption", "exercise_date": "2027-10-26", "effective_date": "2027-10-28",
             "maturity_date": "2032-10-28"}
    trade.pop("accrual_day_count")
    assert npv(trade)[0] == pytest.approx(option.NPV(), rel=1e-10)


def test_r3_sofr_bermudan_calibrates_on_its_own_index_and_bounds_the_european():
    terms = {k: v for k, v in SOFR.items() if k != "accrual_day_count"}
    terms.update(effective_date="2027-10-28", maturity_date="2032-10-28")
    european = {**terms, "trade_type": "european_swaption", "trade_id": "e", "exercise_date": "2027-10-26"}
    bermudan = {**terms, "trade_type": "bermudan_swaption", "trade_id": "b",
                "exercise_dates": ["2027-10-26", "2028-10-30", "2029-10-29", "2030-10-28", "2031-10-28"]}
    engine = {"swap_index_from_trade": True}
    e, b = npv(european, bermudan, pricing={"bermudan": engine})
    assert e < b < 1.6 * e


# --- R4: corporate bonds --------------------------------------------------------------------------

def test_r4_corporate_on_the_spreaded_curve_matches_quantlib():
    spread, treasury = npv(corporate(), corporate(trade_id="tsy", credit_spread=0.0))
    assert spread == pytest.approx(R4_CORPORATE_DIRTY_NPV, abs=0.01)
    assert treasury == pytest.approx(R4_TREASURY_DISCOUNTED_NPV, abs=0.01)


def test_r4_a_spread_bond_is_not_simulated():
    from engine.run.run_request import validate_run
    body = {"market": market(), "portfolio": [corporate()], "analytics": ["sensitivities"]}
    with pytest.raises(ValueError, match="credit spread"):
        validate_run(request_of(body))


# --- R5: an American is never below its Bermudan --------------------------------------------------

def test_r5_american_on_period_starts_is_at_least_the_bermudan():
    terms = {"notional": 5e6, "fixed_rate": 0.045, "payer": True, "effective_date": "2027-10-28",
             "maturity_date": "2032-10-28", "index_tenor_months": 12}
    starts = ["2027-10-26", "2028-10-30", "2029-10-29", "2030-10-28", "2031-10-28"]
    bermudan = {"trade_type": "bermudan_swaption", "trade_id": "b", "exercise_dates": starts, **terms}
    american = {"trade_type": "american_swaption", "trade_id": "a", "first_exercise_date": starts[0],
                "last_exercise_date": starts[-1], **terms}
    on_trade = {"swap_index_from_trade": True}
    b, a = npv(bermudan, american, pricing={"bermudan": on_trade,
                                            "american": {**on_trade, "exercise_on_period_starts": True}})
    _, ore_grid = npv(bermudan, american, pricing={"bermudan": on_trade, "american": on_trade})
    assert a > b
    assert a > ore_grid


# --- R6: assembling the portfolio P&L does not compile per trade count ----------------------------

def test_r6_reporting_currency_assembly_is_on_the_host():
    rng = np.random.default_rng(3)
    columns = tuple(rng.normal(size=256) for _ in range(6000))
    spots = np.linspace(0.5, 2.0, len(columns))
    start = time.perf_counter()
    pnl, total = _in_reporting_currency(columns, spots)
    assert time.perf_counter() - start < 5.0  # the jitted stack took 47 s to compile at 4,000 columns
    assert np.asarray(pnl).shape == (256, len(columns))
    np.testing.assert_allclose(np.asarray(total), (np.stack(columns, axis=-1) * spots).sum(axis=-1), rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(pnl)[:, 0], columns[0] * spots[0])


def test_r2_positions_in_one_contract_share_one_revaluation_bit_for_bit():
    from engine.risk.market.revaluation import revalue_trade
    factors = ["discount:USD/0y", "discount:USD/1y", "discount:USD/2y", "discount:USD/5y", "discount:USD/10y",
               "discount:USD/30y", "equity:AAPL/spot"]
    covariance = np.diag([1e-6] * 6 + [0.30 ** 2 * 10 / 252])
    short = {**OPTION, "trade_id": "opt-short", "quantity": -7}
    body = {"market": {**market(), "currencies": {"USD": {"discount_curve": flat(0.04)}}},
            "portfolio": [OPTION, short], "analytics": ["market_risk"], "outputs": {"pnl": "inline"},
            "config": {"market_risk": {"scenarios": {"source": "monte-carlo", "factors": factors,
                                                     "covariance": covariance.tolist(), "horizon_days": 10,
                                                     "num_scenarios": 64}}}}
    request = request_of(body)
    from engine.run.run_request import market_risk_request, run
    from engine.risk.market.scenarios import ShockScenarios  # noqa: F401
    result = run(request).market_risk
    pnl = np.asarray(result.pnl)
    from engine.risk.market import MonteCarloScenarios  # noqa: F401
    scenarios = request.config.market_risk.scenarios.draw()
    mr = market_risk_request(request, scenarios)
    _, alone = revalue_trade(mr.trades[1], mr.market, scenarios.factors, scenarios.shifts, mr.pricing)
    np.testing.assert_array_equal(pnl[:, 1], np.asarray(alone))
