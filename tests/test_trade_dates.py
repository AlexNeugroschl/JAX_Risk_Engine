"""
Trades are defined by absolute dates, not relative to the evaluation date
(audit M-4, closed; docs/planning/known-issues.md).

A trade config holds its booked schedule -- `effective_date`/`maturity_date`,
and a European swaption's `exercise_date` -- so the same config is the same
trade on every evaluation date. A tenor is a booking convenience, resolved
once, at construction, with ORE's own rule.

What is checked, and against what:

  * **Booking by tenor** reproduces `ORE.MakeVanillaSwap`'s own tenor path
    coupon for coupon, over a grid of trade dates (weekends, holidays, month
    ends), tenors and forward starts.
  * **Seasoned trades** -- priced on a date after they started -- equal ORE:
    swaps against `ORE.DiscountingSwapEngine` with ORE's own fixing history
    (by the valuation pipeline and by the differentiable price function),
    European swaptions against `ORE.JamshidianSwaptionEngine` (the Jamshidian
    engine), Bermudans and Americans against ORE's own
    `NumericLgmMultiLegOptionEngine` (tests/support/ore_lgm_oracle.py). Paid
    cashflows drop out, a coupon fixed before the evaluation date pays its
    historical fixing, a missing fixing is refused (as ORE refuses it), and an
    expired option is worth 0.
  * **Theta ages the booked trade** (audit M-5), as ORE's sensitivity analysis
    defines it: the same schedule valued one calendar day later on the Theta
    market, with the fixings printed in between backfilled at the index's
    forecast, plus the flows paid in between -- against ORE's
    `DiscountingSwapEngine` on the same Theta market.

Curves are sloped, and each pillar is a whole number of ACT/365 days so ORE
and the engine read identical dates. The Bermudan/American curve is flat up
to its first non-zero pillar: ORE's zero-curve build moves the as-of zero to
`z(1e-4)`, which only a flat first segment leaves unchanged (see the oracle's
module docstring and I-34 in docs/planning/known-issues.md).
"""
import dataclasses

import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import (
    BermudanSwaptionConfig,
    _build_ore_swap as _bermudan_underlying,
    exercisable_dates,
)
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig, _build_ore_swap
from engine.market import CurrencyMarket, Market, SwaptionVolSurface, ZeroCurveConfig, index_name
from engine.models.curves import ZeroCurve
from engine.models.ore_builders import (
    SUPPORTED_ACCRUAL_DAY_COUNTS,
    TIME_AXIS_DAY_COUNTER,
    MissingFixingError,
    build_vanilla_swap,
    resolve_accrual_day_count,
    resolve_swap_dates,
)
from engine.risk.greeks import curve_greeks, portfolio_greeks
from engine.risk.price_functions import curves_of, trade_price_function
from engine.risk.sensitivities import SensitivityConfig, portfolio_sensitivities, sensitivity_context, theta_context
from engine.valuation.config import JamshidianEngineConfig, LgmSwaptionEngineConfig, PricingConfig
from engine.valuation.context import PricingContext
from engine.valuation.european import european_terms
from engine.valuation.jamshidian import jamshidian_value
from engine.valuation.portfolio import value_today
from tests.support.lgm_engine import grid_npv
from tests.support.ore_lgm_oracle import ore_lgm_swaption_npv

TODAY = ORE.Date(30, 7, 2026)  # a Thursday
DC = TIME_AXIS_DAY_COUNTER
NOTIONAL = 1_000_000.0
INDEX = index_name("USD", 6)

# Whole ACT/365 days, so an ORE curve built from dates has these exact times.
PILLARS = [0.0, 73 / 365, 1.0, 2.0, 5.0, 10.0, 30.0]
DISC_RATES = [0.020, 0.021, 0.024, 0.028, 0.034, 0.038, 0.040]
FWD_RATES = [0.025, 0.027, 0.031, 0.034, 0.039, 0.042, 0.043]
# Flat to the first non-zero pillar (see module docstring), sloped after it.
LGM_RATES = [0.030, 0.030, 0.032, 0.035, 0.040, 0.042, 0.045]
FLAT_RATE = 0.03
VOLS = SwaptionVolSurface(("1Y", "5Y", "10Y"), ("1Y", "5Y", "10Y"),
                          ((0.0080, 0.0088, 0.0090), (0.0090, 0.0093, 0.0094), (0.0092, 0.0094, 0.0096)))
JAMSHIDIAN = PricingConfig(european="Jamshidian", jamshidian=JamshidianEngineConfig(0.03, 0.01))


def _ore_zero_curve(asof, rates):
    dates = [asof + round(t * 365) for t in PILLARS]
    return ORE.YieldTermStructureHandle(ORE.ZeroCurve(dates, rates, DC))


def _market(asof, disc=DISC_RATES, fwd=FWD_RATES) -> Market:
    return Market(asof, {"USD": CurrencyMarket(ZeroCurveConfig(PILLARS, disc), {INDEX: ZeroCurveConfig(PILLARS, fwd)},
                                               VOLS)})


def _swap_cfg(payer=True, **overrides):
    fields = dict(notional=NOTIONAL, fixed_rate=0.032, payer=payer, swap_tenor="5Y", evaluation_date=TODAY,
                  trade_id="swap")
    fields.update(overrides)
    return SwapConfig(**fields)


def _floating_fixing_dates(swap):
    return [ORE.as_floating_rate_coupon(cf).fixingDate() for cf in swap.floatingLeg()]


def _history(swap, before):
    """A plausible fixing for every coupon fixing before `before`, distinct
    per date so a mix-up cannot cancel out."""
    dates = [d for d in _floating_fixing_dates(swap) if d < before]
    return {d: 0.021 + 0.0007 * i for i, d in enumerate(dates)}


# =============================================================================
# BOOKING: a tenor resolves to ORE's own dates, once
# =============================================================================
def _coupons(swap):
    fixed = [(c.accrualStartDate(), c.accrualEndDate(), c.date(), c.amount())
             for c in map(ORE.as_fixed_rate_coupon, swap.fixedLeg())]
    floating = [(c.accrualStartDate(), c.accrualEndDate(), c.date(), c.fixingDate(), c.accrualPeriod())
                for c in map(ORE.as_floating_rate_coupon, swap.floatingLeg())]
    return fixed, floating


FORWARD_STARTS = [ORE.Period(0, ORE.Days), ORE.Period(1, ORE.Months), ORE.Period(18, ORE.Months),
                  ORE.Period(3, ORE.Years)]


@pytest.mark.parametrize("tenor", ["1Y", "18M", "5Y", "30Y"])
def test_booking_by_tenor_reproduces_make_vanilla_swap(tenor):
    """`resolve_swap_dates` + a dated build == `MakeVanillaSwap(tenor)` on
    the trade date, coupon for coupon, across a year of trade dates (so
    weekends, TARGET holidays and month ends all occur) and forward starts."""
    index = ORE.IborIndex("SimIndex", ORE.Period(6, ORE.Months), 2, ORE.USDCurrency(), ORE.TARGET(),
                          ORE.ModifiedFollowing, False, DC, ORE.YieldTermStructureHandle())
    for day in range(0, 366, 5):
        trade_date = ORE.Date(1, 1, 2026) + day
        for forward_start in FORWARD_STARTS:
            ORE.Settings.instance().evaluationDate = trade_date
            reference = ORE.MakeVanillaSwap(ORE.Period(tenor), index, 0.03, forward_start, nominal=NOTIONAL,
                                             fixedLegDayCount=DC, floatingLegDayCount=DC)
            effective, maturity = resolve_swap_dates(trade_date, tenor, forward_start)
            booked = build_vanilla_swap(NOTIONAL, 0.03, True, effective, maturity, 6, 0.0)
            assert _coupons(booked) == _coupons(reference), (trade_date, forward_start)


def _european_cfg(payer=True, **overrides):
    fields = dict(notional=NOTIONAL, fixed_rate=0.031, payer=payer, swap_tenor="5Y",
                  forward_start=ORE.Period(2, ORE.Years), evaluation_date=TODAY, trade_id="european")
    fields.update(overrides)
    return SwaptionConfig(**fields)


def test_european_booking_by_tenor_places_exercise_on_the_forward_start_point():
    cfg = _european_cfg()
    forward_start_point = ORE.TARGET().advance(TODAY, ORE.Period(2, ORE.Years))
    assert cfg.exercise_date == ORE.TARGET().advance(forward_start_point, 2, ORE.Days)
    assert (cfg.effective_date, cfg.maturity_date) == resolve_swap_dates(TODAY, "5Y", ORE.Period(2, ORE.Years))


class TestBookedDatesAreTheTrade:
    """The audit's evidence, inverted: one config, one trade."""

    def test_a_later_evaluation_date_keeps_the_schedule(self):
        cfg = _swap_cfg()
        later = dataclasses.replace(cfg, evaluation_date=TODAY + 30)
        assert (later.effective_date, later.maturity_date) == (cfg.effective_date, cfg.maturity_date)
        assert _coupons(_build_ore_swap(later)) == _coupons(_build_ore_swap(cfg))
        assert cfg.maturity_date == ORE.Date(3, 8, 2031)

    def test_a_swaption_expiry_approaches(self):
        cfg = _european_cfg(forward_start=ORE.Period(3, ORE.Years))
        for days in (0, 100, 500):
            later = dataclasses.replace(cfg, evaluation_date=TODAY + days)
            assert european_terms(later, later.evaluation_date).expiry_time == DC.yearFraction(
                TODAY + days, cfg.exercise_date)

    def test_explicit_dates_equal_the_tenor_booking(self):
        by_tenor = _swap_cfg()
        by_dates = SwapConfig(notional=NOTIONAL, fixed_rate=0.032, payer=True, effective_date=by_tenor.effective_date,
                              maturity_date=by_tenor.maturity_date, evaluation_date=TODAY, trade_id="swap")
        assert by_dates == by_tenor


class TestConfigValidation:
    def test_tenor_and_dates_together_are_refused(self):
        with pytest.raises(ValueError, match="either swap_tenor or"):
            _swap_cfg(effective_date=TODAY + 5, maturity_date=TODAY + 400)

    def test_no_schedule_is_refused(self):
        with pytest.raises(ValueError, match="needs its dates"):
            _swap_cfg(swap_tenor=None)

    def test_one_date_alone_is_refused(self):
        with pytest.raises(TypeError, match="maturity_date"):
            _swap_cfg(swap_tenor=None, effective_date=TODAY)

    def test_dates_out_of_order_are_refused(self):
        with pytest.raises(ValueError, match="before maturity_date"):
            _swap_cfg(swap_tenor=None, effective_date=TODAY + 10, maturity_date=TODAY + 10)

    def test_a_non_date_is_refused(self):
        with pytest.raises(TypeError, match="effective_date"):
            _swap_cfg(swap_tenor=None, effective_date="2026-08-03", maturity_date=TODAY + 400)

    def test_an_unparseable_tenor_is_refused(self):
        with pytest.raises(ValueError, match="swap_tenor"):
            _swap_cfg(swap_tenor="five years")

    def test_forward_start_without_a_tenor_is_refused(self):
        with pytest.raises(ValueError, match="forward_start"):
            _european_cfg(swap_tenor=None, exercise_date=TODAY + 300, effective_date=TODAY + 302,
                          maturity_date=TODAY + 2000)

    def test_exercise_after_maturity_is_refused(self):
        with pytest.raises(ValueError, match="exercise_date"):
            _european_cfg(swap_tenor=None, forward_start=None, exercise_date=TODAY + 2000, effective_date=TODAY + 2,
                          maturity_date=TODAY + 2000)

    @pytest.mark.parametrize("fixings, error", [
        ({"2026-07-30": 0.02}, TypeError),
        ({TODAY: float("nan")}, ValueError),
    ])
    def test_malformed_fixings_are_refused(self, fixings, error):
        with pytest.raises(error):
            _swap_cfg(fixings=fixings)


# =============================================================================
# SEASONED SWAPS == ORE.DiscountingSwapEngine
# =============================================================================
def _ore_swap(cfg: SwapConfig, disc, fwd, fixings=None, flows_after=None):
    """The booked trade in ORE on `cfg.evaluation_date` over the ORE curves `disc`/`fwd`, with
    `fixings` (default: the trade's) as fixing history. Returns (NPV, the flows paid in
    (flows_after, evaluation date], signed for the trade), the flows read while the history is
    loaded."""
    ORE.Settings.instance().evaluationDate = cfg.evaluation_date
    index = ORE.IborIndex("SimIndex", ORE.Period(6, ORE.Months), 2, ORE.USDCurrency(), ORE.TARGET(),
                          ORE.ModifiedFollowing, False, DC, fwd)
    index.clearFixings()
    try:
        for date, rate in (cfg.fixings if fixings is None else fixings).items():
            index.addFixing(date, rate, True)
        swap = ORE.MakeVanillaSwap(
            ORE.Period(0, ORE.Days), index, cfg.fixed_rate, nominal=cfg.notional,
            swapType=ORE.VanillaSwap.Payer if cfg.payer else ORE.VanillaSwap.Receiver,
            effectiveDate=cfg.effective_date, terminationDate=cfg.maturity_date,
            fixedLegDayCount=resolve_accrual_day_count(cfg.accrual_day_count),
            floatingLegDayCount=resolve_accrual_day_count(cfg.accrual_day_count))
        swap.setPricingEngine(ORE.DiscountingSwapEngine(disc))
        flows = 0.0 if flows_after is None else _ore_flows_between(swap, cfg.payer, flows_after, cfg.evaluation_date)
        return swap.NPV(), flows
    finally:
        index.clearFixings()


def _ore_swap_npv(cfg: SwapConfig) -> float:
    return _ore_swap(cfg, _ore_zero_curve(cfg.evaluation_date, DISC_RATES),
                     _ore_zero_curve(cfg.evaluation_date, FWD_RATES))[0]


def _engine_swap_npvs(cfg: SwapConfig):
    """The engine's t=0 price by both of its routes on the market dated the trade's evaluation
    date: the valuation pipeline, and the differentiable price function (AD Greeks, market
    risk)."""
    market = _market(cfg.evaluation_date)
    by_pipeline = value_today([cfg], market, "USD")[0]
    fn = trade_price_function(cfg, market)
    by_function = float(fn.price(*curves_of(fn, market, jnp.float64)))
    return by_pipeline, by_function


# Evaluation dates across the swap booked on TODAY (effective 2026-08-03,
# floating fixings 2026-07-30, 2027-02-01, ...), each exercising one rule.
SEASONED_SWAP_DATES = {
    "fixed-not-started": TODAY + 1,               # coupon 1 fixed yesterday, starts in 3 days
    "accrual-starts-today": ORE.Date(3, 8, 2026),
    "mid-coupon": ORE.Date(15, 10, 2026),
    "fixes-today": ORE.Date(1, 2, 2027),           # coupon 2 fixes today: ORE forecasts it
    "paid-today": ORE.Date(3, 2, 2027),            # coupon 1 pays today: already occurred
    "mid-life-on-a-payment-date": ORE.Date(3, 8, 2028),
    "last-coupon": ORE.Date(1, 6, 2031),
}


@pytest.mark.parametrize("payer", [True, False], ids=["payer", "receiver"])
@pytest.mark.parametrize("date_id", SEASONED_SWAP_DATES)
def test_seasoned_swap_equals_ore(date_id, payer):
    booked = _swap_cfg(payer=payer)
    later = SEASONED_SWAP_DATES[date_id]
    cfg = dataclasses.replace(booked, evaluation_date=later, fixings=_history(_build_ore_swap(booked), later))
    ore = _ore_swap_npv(cfg)
    for engine in _engine_swap_npvs(cfg):
        assert engine == pytest.approx(ore, rel=1e-10, abs=1e-6)


@pytest.mark.parametrize("day_count", sorted(SUPPORTED_ACCRUAL_DAY_COUNTS))
@pytest.mark.parametrize("date_id", ["fixed-not-started", "mid-coupon", "last-coupon"])
def test_any_leg_day_count_equals_ore(date_id, day_count):
    """The at-par forecast is annualized by the INDEX day count's spanning time and paid
    over the LEG's accrual (I-36). Before the fix the engine divided by the leg's accrual,
    which is right only for an ACT/365 leg: a 1mm 5Y ACT/ACT (ICMA) payer was 161 off."""
    booked = _swap_cfg(accrual_day_count=day_count)
    later = SEASONED_SWAP_DATES[date_id]
    cfg = dataclasses.replace(booked, evaluation_date=later, fixings=_history(_build_ore_swap(booked), later))
    ore = _ore_swap_npv(cfg)
    for engine in _engine_swap_npvs(cfg):
        assert engine == pytest.approx(ore, rel=1e-10, abs=1e-6)


def test_a_fixing_supplied_for_today_is_used():
    """ORE's `InterestRateIndex::fixing(today)` takes a stored fixing over
    its forecast; so does the engine, and it changes the price."""
    booked = _swap_cfg()
    today = SEASONED_SWAP_DATES["fixes-today"]
    swap = _build_ore_swap(booked)
    forecast = dataclasses.replace(booked, evaluation_date=today, fixings=_history(swap, today))
    stored = dataclasses.replace(forecast, fixings={**forecast.fixings, today: 0.05})
    assert _engine_swap_npvs(stored)[0] == pytest.approx(_ore_swap_npv(stored), rel=1e-10)
    assert abs(_engine_swap_npvs(stored)[0] - _engine_swap_npvs(forecast)[0]) > 1_000.0


def test_a_missing_fixing_is_refused_like_ore():
    cfg = dataclasses.replace(_swap_cfg(), evaluation_date=SEASONED_SWAP_DATES["mid-coupon"])
    with pytest.raises(MissingFixingError, match="2026"):
        _engine_swap_npvs(cfg)
    with pytest.raises(RuntimeError, match="Missing .* fixing"):
        _ore_swap_npv(cfg)


def test_a_matured_swap_is_worth_zero():
    cfg = dataclasses.replace(_swap_cfg(), evaluation_date=ORE.Date(4, 8, 2031))
    assert _engine_swap_npvs(cfg) == (0.0, 0.0)
    assert _ore_swap_npv(cfg) == 0.0


# =============================================================================
# SEASONED EUROPEAN SWAPTIONS == ORE.JamshidianSwaptionEngine
# =============================================================================
def _ore_european_npv(cfg: SwaptionConfig) -> float:
    ORE.Settings.instance().evaluationDate = cfg.evaluation_date
    curve = ORE.YieldTermStructureHandle(ORE.FlatForward(cfg.evaluation_date, FLAT_RATE, DC))
    index = ORE.IborIndex("SimIndex", ORE.Period(6, ORE.Months), 2, ORE.USDCurrency(), ORE.TARGET(),
                          ORE.ModifiedFollowing, False, DC, curve)
    underlying = ORE.MakeVanillaSwap(
        ORE.Period(0, ORE.Days), index, cfg.fixed_rate, nominal=cfg.notional,
        swapType=ORE.VanillaSwap.Payer if cfg.payer else ORE.VanillaSwap.Receiver,
        effectiveDate=cfg.effective_date, terminationDate=cfg.maturity_date,
        fixedLegDayCount=DC, floatingLegDayCount=DC)
    swaption = ORE.Swaption(underlying, ORE.EuropeanExercise(cfg.exercise_date))
    swaption.setPricingEngine(ORE.JamshidianSwaptionEngine(ORE.HullWhite(curve, 0.03, 0.01), curve))
    return swaption.NPV()


def _engine_european_npv(cfg: SwaptionConfig) -> float:
    """The Jamshidian engine on the flat curve, on the trade's evaluation date."""
    flat = ZeroCurve.flat(FLAT_RATE, PILLARS)
    return float(jamshidian_value(cfg, PricingContext(cfg.evaluation_date, {"USD": flat}, {}, None),
                                  JAMSHIDIAN.jamshidian))


@pytest.mark.parametrize("payer", [True, False], ids=["payer", "receiver"])
@pytest.mark.parametrize("days", [0, 100, 365, 700])
def test_seasoned_european_equals_ore(days, payer):
    """To QuantLib's Brent tolerance on its root (tests/test_jamshidian.py)."""
    cfg = dataclasses.replace(_european_cfg(payer), evaluation_date=TODAY + days)
    ore = _ore_european_npv(cfg)
    assert ore > 100.0
    assert _engine_european_npv(cfg) == pytest.approx(ore, rel=5e-6)


@pytest.mark.parametrize("offset", [0, 1, 30], ids=["on-exercise-date", "day-after", "month-after"])
def test_an_expired_european_is_worth_zero_like_ore(offset):
    booked = _european_cfg()
    cfg = dataclasses.replace(booked, evaluation_date=booked.exercise_date + offset)
    assert cfg.is_expired()
    assert _ore_european_npv(cfg) == 0.0
    assert _engine_european_npv(cfg) == 0.0
    for pricing in (PricingConfig(), JAMSHIDIAN):
        market = _market(cfg.evaluation_date, [FLAT_RATE] * len(PILLARS), [FLAT_RATE] * len(PILLARS))
        assert value_today([cfg], market, "USD", pricing) == [0.0]
        assert all(np.all(v == 0.0) for v in curve_greeks(cfg, market, pricing, 1e-4).values())


# =============================================================================
# SEASONED BERMUDANS AND AMERICANS == ORE's NumericLgmMultiLegOptionEngine
# =============================================================================
MODEL = dict(a=0.03, sigma=0.01, curve=ZeroCurveConfig(PILLARS, LGM_RATES), n_per_std=48, std_devs=6.0)
AMERICAN_WINDOW = (ORE.Date(3, 8, 2027), ORE.Date(3, 8, 2029))


def _bermudan_cfg(payer=True):
    booked = BermudanSwaptionConfig(notional=NOTIONAL, fixed_rate=0.034, payer=payer, exercise_dates=[TODAY + 1],
                                    swap_tenor="5Y", evaluation_date=TODAY, trade_id="bermudan")
    # A standard coterminal Bermudan: exercisable on each fixed accrual start after the first.
    return dataclasses.replace(booked, exercise_dates=exercisable_dates(booked)[1:])


def _american_cfg(payer=True):
    return AmericanSwaptionConfig(notional=NOTIONAL, fixed_rate=0.034, payer=payer,
                                  first_exercise_date=AMERICAN_WINDOW[0], last_exercise_date=AMERICAN_WINDOW[1],
                                  swap_tenor="5Y", evaluation_date=TODAY, trade_id="american")


def _ore_lgm_npv(cfg) -> float:
    american = isinstance(cfg, AmericanSwaptionConfig)
    return ore_lgm_swaption_npv(
        evaluation_date=cfg.evaluation_date, curve_times=PILLARS, curve_rates=LGM_RATES,
        swap=_bermudan_underlying(cfg), notional=cfg.notional, fixed_rate=cfg.fixed_rate, payer=cfg.payer,
        floating_spread=0.0, index_tenor_months=6, style="American" if american else "Bermudan",
        exercise_dates=[cfg.first_exercise_date, cfg.last_exercise_date] if american else cfg.exercise_dates,
        hw_a=MODEL["a"], hw_sigma=MODEL["sigma"], n_per_std=MODEL["n_per_std"], std_devs=MODEL["std_devs"],
        exercise_time_steps_per_year=24, fixings=cfg.fixings,
    ).npv


def _seasoned(cfg, date):
    return dataclasses.replace(cfg, evaluation_date=date, fixings=_history(_bermudan_underlying(cfg), date))


LGM_CASES = {
    # coupon 1 fixed yesterday and can still be exercised into (belongs until its start)
    "bermudan-historical-fixing-in-play": (_bermudan_cfg, TODAY + 1),
    "bermudan-mid-period": (_bermudan_cfg, ORE.Date(15, 10, 2026)),
    # coupon 2 fixed on 2027-02-01, starts 2027-02-03
    "bermudan-between-fixing-and-start": (_bermudan_cfg, ORE.Date(2, 2, 2027)),
    "bermudan-first-exercise-passed": (_bermudan_cfg, ORE.Date(1, 9, 2027)),
    "american-before-window": (_american_cfg, ORE.Date(2, 2, 2027)),
    # Inside the window: exercisable from tomorrow, never today (ORE's
    # ExerciseBuilder), into the broken current coupon.
    "american-inside-window": (_american_cfg, ORE.Date(15, 10, 2027)),
    "american-window-opens-on-the-evaluation-date": (
        lambda payer: dataclasses.replace(_american_cfg(payer), first_exercise_date=TODAY), TODAY),
}


@pytest.mark.parametrize("payer", [True, False], ids=["payer", "receiver"])
@pytest.mark.parametrize("case_id", LGM_CASES)
def test_seasoned_bermudan_and_american_equal_ore(case_id, payer):
    make, date = LGM_CASES[case_id]
    cfg = _seasoned(make(payer), date)
    ore = _ore_lgm_npv(cfg)
    assert ore > 100.0
    assert grid_npv(cfg, **MODEL) == pytest.approx(ore, rel=1e-10, abs=1e-7)


@pytest.mark.parametrize("make", [_bermudan_cfg, _american_cfg], ids=["bermudan", "american"])
def test_an_expired_option_is_worth_zero(make):
    """ORE builds a swaption with no exercise date after today as a zero
    cashflow (OREData/ored/portfolio/swaption.cpp, step 6: "if we do not
    have an active exercise as of today ... we only build unconditional
    premiums"). The oracle cannot show it: that trade has no engine to run."""
    booked = make()
    last = booked.exercise_dates[-1] if isinstance(booked, BermudanSwaptionConfig) else booked.last_exercise_date
    cfg = _seasoned(booked, last + 1)
    assert cfg.is_expired()
    assert not dataclasses.replace(cfg, evaluation_date=last - 1).is_expired()
    assert grid_npv(cfg, **MODEL) == 0.0
    engine = LgmSwaptionEngineConfig(reversion=0.03, volatility=0.01, calibration="None")
    market = _market(cfg.evaluation_date, LGM_RATES, LGM_RATES)
    fn = trade_price_function(cfg, market, PricingConfig(bermudan=engine, american=engine))
    assert float(fn.price(*curves_of(fn, market, jnp.float64))) == 0.0


def test_a_bermudan_missing_a_fixing_it_needs_is_refused():
    cfg = dataclasses.replace(_bermudan_cfg(), evaluation_date=ORE.Date(2, 2, 2027))
    with pytest.raises(MissingFixingError):
        grid_npv(cfg, **MODEL)


def test_a_bermudan_needs_no_fixing_for_a_coupon_it_can_no_longer_enter():
    """ORE never values a coupon whose accrual started before the
    evaluation date (Bermudan exercise enters whole periods), so its fixing
    is not needed -- mid-period there is nothing to supply."""
    cfg = dataclasses.replace(_bermudan_cfg(), evaluation_date=ORE.Date(15, 10, 2026))
    assert grid_npv(cfg, **MODEL) == pytest.approx(_ore_lgm_npv(cfg), rel=1e-10)


# =============================================================================
# THETA AGES THE BOOKED TRADE (audit M-5), as ORE's sensitivity analysis defines it
# =============================================================================
FRIDAY = ORE.Date(31, 7, 2026)
CONFIG = SensitivityConfig()


def _ore_curve_of(context_curve, reference: ORE.Date):
    """A sensitivity or Theta market's curve (log-linear discount factors at whole-day tenor
    points from `reference`) as an ORE `DiscountCurve`."""
    times = np.asarray(context_curve.times)
    dates = [reference + int(round(t * 365)) for t in times]
    handle = ORE.YieldTermStructureHandle(ORE.DiscountCurve(dates, list(np.exp(np.asarray(context_curve.log_discounts))),
                                                            DC))
    handle.enableExtrapolation()
    return handle


def _ore_flows_between(swap, payer: bool, start: ORE.Date, end: ORE.Date) -> float:
    """The swap's cashflows paid in (start, end], signed for the fixed payer when `payer`."""
    fixed = sum(c.amount() for c in swap.fixedLeg() if start < c.date() <= end)
    floating = sum(c.amount() for c in swap.floatingLeg() if start < c.date() <= end)
    return (floating - fixed) if payer else (fixed - floating)


@pytest.mark.parametrize("base", [TODAY, ORE.Date(2, 2, 2027), FRIDAY],
                         ids=["booking-date", "day-before-a-payment", "friday"])
def test_swap_theta_equals_ore(base):
    """Theta = NPV(Theta date, Theta market) - NPV(base date, sensitivity market) + flows paid
    in between, everything but the markets priced by ORE: the booked schedule one calendar day
    older (a Friday rolls to the Saturday, I-38), the fixings printed in between backfilled at
    the index's forecast on the Theta market (ORE's `FixingManager`). On 2027-02-02 the first
    floating coupon pays the next day, so the flow term is material. Both Greeks methods give
    this Theta."""
    booked = _swap_cfg()
    swap = _build_ore_swap(booked)
    cfg = dataclasses.replace(booked, evaluation_date=base, fixings=_history(swap, base))
    market = _market(base)
    base_ctx, theta_ctx = sensitivity_context(market, CONFIG), theta_context(market, CONFIG)
    theta_date = theta_ctx.date
    assert theta_date == base + 1

    base_npv, _ = _ore_swap(cfg, _ore_curve_of(base_ctx.discount["USD"], base),
                            _ore_curve_of(base_ctx.index[INDEX], base))
    aged = dataclasses.replace(cfg, evaluation_date=theta_date)
    fixings = {**cfg.fixings, **theta_ctx.fixings[INDEX]}
    theta_npv, flows = _ore_swap(aged, _ore_curve_of(theta_ctx.discount["USD"], theta_date),
                                 _ore_curve_of(theta_ctx.index[INDEX], theta_date), fixings, flows_after=base)
    expected = theta_npv - base_npv + flows

    for greeks in (portfolio_sensitivities, portfolio_greeks):
        assert float(greeks([cfg], market, "USD", config=CONFIG)[0]["theta"]) == pytest.approx(
            expected, rel=1e-9, abs=1e-6)
    if base == ORE.Date(2, 2, 2027):
        assert abs(flows) > 1_000.0


def test_theta_rolls_one_calendar_day_as_ore():
    """ORE's `thetaDate = asof + thetaPeriod` (I-38): Friday -> Saturday, not Monday."""
    assert theta_context(_market(FRIDAY), CONFIG).date == ORE.Date(1, 8, 2026)
    assert theta_context(_market(TODAY), SensitivityConfig(theta_days=3)).date == TODAY + 3


# =============================================================================
# THE PORTFOLIO AND HTTP BOUNDARIES CARRY BOOKED DATES AND FIXINGS
# =============================================================================
def test_price_portfolio_prices_a_seasoned_swap_like_ore():
    """End to end on either model: the base NPV is ORE's, and the scenario cube takes the aged
    schedule (the accruing coupon at its fixing, no paid flow; tests/test_valuation.py)."""
    from engine.portfolio import PortfolioRequest, RunConfig, price_portfolio
    from engine.simulation.config import CamConfig, HullWhiteConfig, LgmConfig

    later = SEASONED_SWAP_DATES["mid-coupon"]
    booked = _swap_cfg()
    cfg = dataclasses.replace(booked, evaluation_date=later, fixings=_history(_build_ore_swap(booked), later))
    for model in (LgmConfig, HullWhiteConfig):
        simulation = CamConfig(dates=(later + 91,), base_currency="USD", ir={"USD": model(0.03, 0.01)}, samples=64)
        result = price_portfolio(PortfolioRequest(market=_market(later), trades=[cfg],
                                                  config=RunConfig(simulation=simulation)))
        assert result.base_npv_per_trade[0] == pytest.approx(_ore_swap_npv(cfg), rel=1e-10)
        assert result.warnings == []
        assert bool(jnp.all(jnp.isfinite(result.npv_cube)))


def test_http_schema_takes_booked_dates_and_fixings():
    from engine.api.market_schemas import EuropeanTradeSchema, SwapTradeSchema

    swap = SwapTradeSchema(notional=NOTIONAL, fixed_rate=0.03, payer=True, effective_date="2026-08-03",
                           maturity_date="2031-08-03", fixings={"2026-07-30": 0.025}).to_dataclass(TODAY + 10, "s")
    assert (swap.effective_date, swap.maturity_date) == (ORE.Date(3, 8, 2026), ORE.Date(3, 8, 2031))
    assert swap.fixings == {TODAY: 0.025}

    swaption = EuropeanTradeSchema(notional=NOTIONAL, fixed_rate=0.031, payer=True, swap_tenor="5Y",
                                   forward_start="2Y").to_dataclass(TODAY, "e")
    assert swaption.exercise_date == _european_cfg().exercise_date


def test_http_schema_without_a_schedule_is_refused():
    """No default tenor: a trade without dates is an error, not a 5Y swap."""
    from engine.api.market_schemas import SwapTradeSchema

    with pytest.raises(ValueError, match="needs its dates"):
        SwapTradeSchema(notional=NOTIONAL, fixed_rate=0.03, payer=True).to_dataclass(TODAY, "s")
