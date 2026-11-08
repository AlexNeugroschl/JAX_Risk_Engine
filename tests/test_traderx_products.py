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


def test_r3_a_day_count_contradicting_the_index_is_refused():
    with pytest.raises(ValueError, match="contradicts overnight_index"):
        npv({**SOFR, "accrual_day_count": "30/360"})


def test_r3_an_index_in_another_currency_is_refused():
    with pytest.raises(ValueError, match="EUR index"):
        npv({**SOFR, "overnight_index": "EUR-ESTR"})


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


# --- R3 completed: every TraderX convention, and seasoned OIS coupons -----------------------------

#: The R3 convention set (TraderX `SwapConventions`): index, QuantLib's index, schedule and payment
#: calendar, payment lag, day count of both legs.
QL_OIS = {
    "USD-SOFR": (lambda h: ORE.Sofr(h), ORE.UnitedStates(ORE.UnitedStates.GovernmentBond), 2, ORE.Actual360()),
    "EUR-ESTR": (lambda h: ORE.Estr(h), ORE.TARGET(), 1, ORE.Actual360()),
    "GBP-SONIA": (lambda h: ORE.Sonia(h), ORE.UnitedKingdom(), 0, ORE.Actual365Fixed()),
    "JPY-TONA": (lambda h: ORE.Tonar(h), ORE.Japan(), 2, ORE.Actual365Fixed()),
}


def ql_ois_npv(name, effective, maturity, months, fixings=()):
    """QuantLib's `OvernightIndexedSwap` on the convention set, payer of 4.2% on 10mm, its index's
    past `fixings` `[(ORE.Date, rate)]` added to (and cleared from) QuantLib's index history."""
    _, _, curve = ql_setup()
    make_index, calendar, lag, day_count = QL_OIS[name]
    index = make_index(curve(0.041))
    schedule = ORE.Schedule(effective, maturity, ORE.Period(months, ORE.Months), calendar, ORE.ModifiedFollowing,
                            ORE.ModifiedFollowing, ORE.DateGeneration.Forward, False)
    swap = ORE.OvernightIndexedSwap(ORE.OvernightIndexedSwap.Payer, 1e7, schedule, 0.042, day_count, index, 0.0, lag,
                                    ORE.ModifiedFollowing, calendar)
    swap.setPricingEngine(ORE.DiscountingSwapEngine(curve(0.04)))
    try:
        for d, r in fixings:
            index.addFixing(d, r)
        return swap.NPV()
    finally:
        index.clearFixings()


def every_currency_market():
    usd = market()["currencies"]["USD"]
    curves = {f"{c}-SIMINDEX-{m}M": flat(0.041) for c in ("USD", "EUR", "GBP", "JPY") for m in (3, 12)}
    one = lambda c: {**usd, "index_curves": {k: v for k, v in curves.items() if k.startswith(c)}}  # noqa: E731
    return {**market(), "currencies": {c: one(c) for c in ("USD", "EUR", "GBP", "JPY")}}


def daily_fixings(name, start, today):
    """Made-up daily fixings on the index's own fixing calendar, every day from `start` to before `today`."""
    calendar = {"USD-SOFR": ORE.Sofr(), "EUR-ESTR": ORE.Estr(), "GBP-SONIA": ORE.Sonia(),
                "JPY-TONA": ORE.Tonar()}[name].fixingCalendar()
    out, d, i = [], start, 0
    while d < today:
        out.append((d, 0.043 + 0.002 * np.sin(i / 7.0)))
        d, i = calendar.advance(d, 1, ORE.Days), i + 1
    return out


def ois(name, effective, maturity, months, **fields):
    currency = name[:3]
    return {"trade_type": "swap", "trade_id": f"{name}-{effective}", "currency": currency, "notional": 1e7,
            "fixed_rate": 0.042, "payer": True, "effective_date": effective, "maturity_date": maturity,
            "index_tenor_months": months, "overnight_index": name, **fields}


def iso(d):
    return ORE.Date(int(d[8:]), int(d[5:7]), int(d[:4]))


@pytest.mark.parametrize("name,months", [("USD-SOFR", 3), ("EUR-ESTR", 12), ("GBP-SONIA", 12), ("JPY-TONA", 12)])
def test_r3_every_convention_matches_quantlib(name, months):
    trade = ois(name, "2026-10-28", "2031-10-28", months)
    body = {"market": every_currency_market(), "portfolio": [trade], "analytics": ["npv"],
            "config": {"base_currency": trade["currency"]}}
    value = run_in_process(body).portfolio.base_npv_per_trade[0]
    assert value == pytest.approx(ql_ois_npv(name, iso("2026-10-28"), iso("2031-10-28"), months), abs=0.01)


@pytest.mark.parametrize("effective,maturity,months", [
    ("2026-04-28", "2031-04-28", 12),   # an annual coupon half accrued
    ("2026-09-28", "2031-09-29", 3),    # a quarterly coupon in progress
    ("2025-10-23", "2030-10-23", 12),   # a coupon accrued in full on Friday, paid on Tuesday
])
def test_r3_a_seasoned_ois_pays_its_realised_compounding(effective, maturity, months):
    past = daily_fixings("USD-SOFR", iso(effective), iso(ASOF))
    fixings = {d.ISO(): r for d, r in past}
    seasoned = lambda history: run_in_process({  # noqa: E731
        "market": every_currency_market(), "analytics": ["npv"],
        "portfolio": [ois("USD-SOFR", effective, maturity, months, overnight_fixings=history)]}).portfolio.base_npv_per_trade[0]
    value = seasoned(fixings)
    expected = ql_ois_npv("USD-SOFR", iso(effective), iso(maturity), months, past)
    assert value == pytest.approx(expected, abs=0.01)
    # negative control: the realised compounding matters (the fixings are 20-40bp off the curve)
    flat_history = {d: 0.041 for d in fixings}
    assert abs(seasoned(flat_history) - expected) > 100


def test_r3_a_seasoned_ois_missing_a_past_fixing_is_refused():
    past = daily_fixings("USD-SOFR", iso("2026-04-28"), iso(ASOF))
    fixings = {d.ISO(): r for d, r in past[:-1]}
    with pytest.raises(ValueError, match="missing overnight fixing for October 23rd, 2026"):
        npv(ois("USD-SOFR", "2026-04-28", "2031-04-28", 12, overnight_fixings=fixings))


def test_r3_a_spread_on_a_coupon_in_progress_is_refused():
    fixings = {d.ISO(): r for d, r in daily_fixings("USD-SOFR", iso("2026-04-28"), iso(ASOF))}
    with pytest.raises(ValueError, match="floating spread"):
        npv(ois("USD-SOFR", "2026-04-28", "2031-04-28", 12, overnight_fixings=fixings, floating_spread=0.001))


def test_r2_delta_matches_quantlib():
    asof, a365, curve = ql_setup()
    spot = ORE.SimpleQuote(150.0)
    process = ORE.BlackScholesMertonProcess(
        ORE.QuoteHandle(spot), curve(0.005), curve(0.04),
        ORE.BlackVolTermStructureHandle(ORE.BlackConstantVol(asof, ORE.NullCalendar(), 0.30, a365)))
    option = ORE.VanillaOption(ORE.PlainVanillaPayoff(ORE.Option.Call, 150.0),
                               ORE.AmericanExercise(asof, ORE.Date(18, 12, 2026)))
    option.setPricingEngine(ORE.FdBlackScholesVanillaEngine(process, 2000, 2000))
    assert option.delta() == pytest.approx(0.5401, abs=1e-4)  # the request's FD(400, 400) value
    h = 0.01 * 150.0
    up, down = ({**market(), "equities": {"AAPL": {**market()["equities"]["AAPL"], "spot": 150.0 + s}}} for s in (h, -h))
    share = {**OPTION, "multiplier": 1}
    bumped = [run_in_process({"market": m, "portfolio": [share], "analytics": ["npv"]}).portfolio.base_npv_per_trade[0]
              for m in (up, down)]
    assert (bumped[0] - bumped[1]) / (2 * h) == pytest.approx(option.delta(), abs=2e-3)


def test_r3_a_bermudan_past_its_first_exercise_prices_as_its_remaining_option():
    """Past its first exercise date, with its underlying accruing, a Bermudan still exercises only
    into coupons starting on a later exercise date: its value is the forward-starting Bermudan on
    the remaining dates (calibrated on the same basket; that pricer's ORE parity is
    `tests/test_ore_bermudan_oracle.py`). It was refused while its underlying had a coupon in progress."""
    terms = {"notional": 5e6, "fixed_rate": 0.045, "payer": True, "currency": "GBP", "overnight_index": "GBP-SONIA",
             "index_tenor_months": 12, "maturity_date": "2031-04-28"}
    mid = {"trade_type": "bermudan_swaption", "trade_id": "mid", "effective_date": "2026-04-28",
           "exercise_dates": ["2026-04-24", "2027-04-28", "2028-04-28", "2029-04-30", "2030-04-29"], **terms}
    forward = {**mid, "trade_id": "forward", "effective_date": "2027-04-28", "exercise_dates": mid["exercise_dates"][1:]}
    body = {"market": every_currency_market(), "portfolio": [mid, forward], "analytics": ["npv"],
            "config": {"base_currency": "GBP", "pricing": {"bermudan": {"swap_index_from_trade": True}}}}
    values = run_in_process(body).portfolio.base_npv_per_trade
    assert values[0] > 0
    assert values[0] == pytest.approx(values[1], rel=1e-12)
