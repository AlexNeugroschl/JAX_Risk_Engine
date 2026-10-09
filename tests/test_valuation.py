"""
Valuation with ORE's semantics (`engine.valuation`), trade by trade, against ORE's own
engines -- today and on simulated paths (plan 6.2, L2 on scenario curves).

On a path, each simulated curve is handed to ORE as an `ORE.DiscountCurve` on the same
tenor dates (log-linear in the discount factor with a flat-forward extrapolation: the
scenario market's curve exactly), with the evaluation date moved to the simulation date and
the fixings `FixingManager` would have stored put in ORE's fixing history. ORE then prices the
trade with its default engine, and that is the reference for the cube.

Every path test runs under both interest-rate models (`MODELS`): the pricers read only the
path curves, so the Hull-White model's cube is ORE's on its own paths exactly as the LGM's is
(2026-10-01: I-43 exercised options, I-44 one model, I-24 bonds, I-04 paid flows).
"""
import dataclasses

import numpy as np
import ORE
import pytest

from engine.instruments.swap import SwapConfig, _build_ore_swap
from engine.market import CurrencyMarket, Market, ZeroCurveConfig, index_name
from engine.models.curves import ZeroCurve
from engine.models.ore_builders import TIME_AXIS_DAY_COUNTER as DC, ibor_index, resolve_accrual_day_count
from engine.simulation.config import CamConfig, HullWhiteConfig, LgmConfig, simulate
from engine.valuation.legs import legs_cube, legs_of, path_fixings, path_schedule, today_npv

ASOF = ORE.Date(30, 7, 2026)
PILLARS = [0.0, 73 / 365, 1.0, 2.0, 5.0, 10.0, 30.0]
DISC_RATES = [0.020, 0.021, 0.024, 0.028, 0.034, 0.038, 0.040]
INDEX_RATES = [0.025, 0.027, 0.031, 0.034, 0.039, 0.042, 0.043]
INDEX = index_name("USD", 6)
TENORS = ("3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y")
#: The interest-rate models every path test runs under.
MODELS = {"LGM": LgmConfig, "HullWhite": HullWhiteConfig}


def _market():
    return Market(ASOF, {"USD": CurrencyMarket(
        discount_curve=ZeroCurveConfig(PILLARS, DISC_RATES),
        index_curves={INDEX: ZeroCurveConfig(PILLARS, INDEX_RATES)})})


def _ore_zero_curve(rates, pillars=PILLARS):
    handle = ORE.YieldTermStructureHandle(ORE.ZeroCurve([ASOF + round(t * 365) for t in pillars], rates, DC))
    handle.enableExtrapolation()
    return handle


def _tenor_dates(date):
    """Where ORE's simulation market holds its tenor points on `date`: at each tenor's time from
    the as-of date (`ScenarioSimMarket::addYieldCurve` builds the curve on those times once, on a
    reference date that moves with the evaluation date), so `date` plus the as-of tenor's days."""
    return [date + ((ASOF + ORE.Period(t)) - ASOF) for t in TENORS]


def _ore_path_curve(date, tenor_times, log_discounts):
    """The scenario curve of one path and date as ORE's simulation market holds it."""
    dates = [date] + _tenor_dates(date)
    np.testing.assert_allclose([DC.yearFraction(date, d) for d in dates], tenor_times, rtol=0, atol=1e-15)
    curve = ORE.DiscountCurve(dates, [float(np.exp(v)) for v in log_discounts], DC)
    curve.enableExtrapolation()
    return ORE.YieldTermStructureHandle(curve)


def _ore_swap_npv(cfg, evaluation_date, disc, fwd, fixings):
    """ORE's `DiscountingSwapEngine` on the booked trade at `evaluation_date`."""
    ORE.Settings.instance().evaluationDate = evaluation_date
    index = ibor_index(cfg.index_tenor_months, fwd)
    index.clearFixings()
    try:
        for date, rate in fixings.items():
            index.addFixing(date, float(rate), True)
        day_count = resolve_accrual_day_count(cfg.accrual_day_count)
        swap = ORE.MakeVanillaSwap(
            ORE.Period(0, ORE.Days), index, cfg.fixed_rate, nominal=cfg.notional,
            swapType=ORE.VanillaSwap.Payer if cfg.payer else ORE.VanillaSwap.Receiver,
            effectiveDate=cfg.effective_date, terminationDate=cfg.maturity_date,
            fixedLegDayCount=day_count, floatingLegDayCount=day_count,
            floatingLegSpread=cfg.floating_spread)
        swap.setPricingEngine(ORE.DiscountingSwapEngine(disc))
        return swap.NPV()
    finally:
        index.clearFixings()
        ORE.Settings.instance().evaluationDate = ASOF


def _history(cfg):
    """A plausible historical fixing for every coupon fixed before the as-of date."""
    swap = _build_ore_swap(cfg)
    dates = [ORE.as_floating_rate_coupon(c).fixingDate() for c in swap.floatingLeg()]
    return {d: 0.021 + 0.0007 * i for i, d in enumerate(d for d in dates if d < ASOF)}


def _swap(**overrides):
    fields = dict(notional=1e6, fixed_rate=0.032, payer=True, effective_date=ORE.Date(3, 2, 2025),
                  maturity_date=ORE.Date(3, 2, 2032), evaluation_date=ASOF, floating_spread=0.0015, trade_id="swap")
    fields.update(overrides)
    cfg = SwapConfig(**fields)
    return dataclasses.replace(cfg, fixings=_history(cfg))


SWAPS = {
    "seasoned-payer": {},
    "seasoned-receiver": {"payer": False},
    "act-act-legs": {"accrual_day_count": "ACT/ACT (ICMA)"},
    "forward-starting": {"effective_date": ORE.Date(4, 8, 2027), "maturity_date": ORE.Date(4, 8, 2033),
                         "floating_spread": 0.0},
}


@pytest.mark.parametrize("name", SWAPS)
def test_today_equals_ores_discounting_swap_engine(name):
    cfg = _swap(**SWAPS[name])
    legs = legs_of(_build_ore_swap(cfg), cfg.payer, ASOF, cfg.fixings)
    ours = float(today_npv(legs, ASOF, ZeroCurve.from_config(ZeroCurveConfig(PILLARS, DISC_RATES)),
                           ZeroCurve.from_config(ZeroCurveConfig(PILLARS, INDEX_RATES))))
    ore = _ore_swap_npv(cfg, ASOF, _ore_zero_curve(DISC_RATES), _ore_zero_curve(INDEX_RATES), cfg.fixings)
    assert ours == pytest.approx(ore, rel=1e-10, abs=1e-6)


@pytest.fixture(scope="module", params=MODELS)
def scenarios(request):
    """A few paths over dates that fall between fixing and payment dates, on a holiday
    (Good Friday 2027: FixingManager fixes on the next business day), and after maturity."""
    dates = (ORE.Date(15, 10, 2026), ORE.Date(2, 2, 2027), ORE.Date(26, 3, 2027), ORE.Date(5, 8, 2027),
             ORE.Date(1, 2, 2029), ORE.Date(3, 8, 2031), ORE.Date(10, 2, 2032))
    config = CamConfig(dates=dates, base_currency="USD", ir={"USD": MODELS[request.param](0.03, 0.012)},
                       curve_tenors=TENORS, samples=4, seed=7)
    return simulate(_market(), config)


def test_path_fixings_are_ores_index_forecast_on_the_path_curve(scenarios):
    fixings = np.asarray(path_fixings(6, ASOF, scenarios.dates, scenarios.times, scenarios.index[INDEX]))
    for j, date in enumerate(scenarios.dates):
        for s in range(scenarios.num_paths):
            curve = _ore_path_curve(date, np.asarray(scenarios.index[INDEX].tenor_times[j]),
                                    np.asarray(scenarios.index[INDEX].log_discounts[s, j]))
            ORE.Settings.instance().evaluationDate = date
            index = ibor_index(6, curve)
            index.clearFixings()
            expected = index.fixing(ORE.TARGET().adjust(date, ORE.Following))
            assert fixings[s, j] == pytest.approx(expected, rel=1e-12)
    ORE.Settings.instance().evaluationDate = ASOF


@pytest.mark.parametrize("name", SWAPS)
def test_every_path_and_date_equals_ores_discounting_swap_engine(name, scenarios):
    """Paid cashflows drop out, historical fixings stay, and every fixing between the as-of
    date and the simulation date pays FixingManager's path value."""
    cfg = _swap(**SWAPS[name])
    legs = legs_of(_build_ore_swap(cfg), cfg.payer, ASOF, cfg.fixings)
    schedule = path_schedule(legs, ASOF, scenarios.dates)
    fixings = path_fixings(6, ASOF, scenarios.dates, scenarios.times, scenarios.index[INDEX])
    cube = np.asarray(legs_cube(legs, schedule, scenarios.times, scenarios.discount["USD"],
                                scenarios.index[INDEX], fixings))
    swap = _build_ore_swap(cfg)
    fixing_dates = [ORE.as_floating_rate_coupon(c).fixingDate() for c in swap.floatingLeg()]
    serials = [d.serialNumber() for d in scenarios.dates]
    for j, date in enumerate(scenarios.dates):
        for s in range(scenarios.num_paths):
            history = dict(cfg.fixings)
            for f in fixing_dates:
                if ASOF <= f < date:
                    history[f] = float(fixings[s, np.searchsorted(serials, f.serialNumber(), side="right")])
            disc = _ore_path_curve(date, np.asarray(scenarios.discount["USD"].tenor_times[j]),
                                   np.asarray(scenarios.discount["USD"].log_discounts[s, j]))
            fwd = _ore_path_curve(date, np.asarray(scenarios.index[INDEX].tenor_times[j]),
                                  np.asarray(scenarios.index[INDEX].log_discounts[s, j]))
            if date >= cfg.maturity_date:
                assert cube[s, j] == 0.0
                continue
            ore = _ore_swap_npv(cfg, date, disc, fwd, history)
            assert cube[s, j] == pytest.approx(ore, rel=1e-10, abs=1e-6), (date, s)


# =============================================================================
# European swaptions: ORE's default BlackMultiLegOptionEngine (T-12)
# =============================================================================
from engine.instruments.european_swaption import SwaptionConfig  # noqa: E402
from engine.market import SwaptionVolSurface  # noqa: E402
from engine.valuation.context import from_market  # noqa: E402
from engine.valuation.european import (  # noqa: E402
    european_cube, european_terms, european_value, variance_on_path, volatility_on_path,
)
from tests.support.ore_lgm_oracle import ore_lgm_swaption_npv  # noqa: E402

VOLS = SwaptionVolSurface(
    ("6M", "1Y", "2Y", "5Y", "10Y"), ("1Y", "2Y", "5Y", "10Y"),
    ((0.0070, 0.0080, 0.0085, 0.0090), (0.0080, 0.0085, 0.0090, 0.0092), (0.0085, 0.0088, 0.0091, 0.0093),
     (0.0090, 0.0092, 0.0093, 0.0095), (0.0092, 0.0093, 0.0094, 0.0096)))
# Flat to the first non-zero pillar, as written before the oracle's fix of I-34 (no longer needed).
ORACLE_PILLARS = [0.0, 1.0, 2.0, 5.0, 10.0, 30.0]
ORACLE_DISC = [0.020, 0.020, 0.025, 0.030, 0.035, 0.040]
ORACLE_INDEX = [0.025, 0.025, 0.031, 0.036, 0.040, 0.044]


def _ore_vol_matrix():
    matrix = ORE.Matrix(len(VOLS.option_tenors), len(VOLS.swap_tenors))
    for i, row in enumerate(VOLS.vols):
        for j, v in enumerate(row):
            matrix[i][j] = v
    return ORE.SwaptionVolatilityMatrix(
        ORE.TARGET(), ORE.Following, [ORE.Period(t) for t in VOLS.option_tenors],
        [ORE.Period(t) for t in VOLS.swap_tenors], matrix, ORE.Actual365Fixed(), True, ORE.Normal)


def test_the_vol_surface_equals_quantlibs_swaption_volatility_matrix():
    ORE.Settings.instance().evaluationDate = ASOF
    ore = _ore_vol_matrix()
    ore.enableExtrapolation()
    for t in [0.1, 0.5, 0.77, 1.0, 3.3, 7.0, 12.0]:
        for length in [0.5, 1.0, 1.5, 4.0, 10.0, 20.0]:
            assert float(VOLS.volatility(ASOF, t, length)) == pytest.approx(ore.volatility(t, length, 0.03), rel=1e-14)
            assert float(VOLS.black_variance(ASOF, t, length)) == pytest.approx(
                ore.blackVariance(t, length, 0.03), rel=1e-14)


def _european(**overrides):
    fields = dict(notional=1e6, fixed_rate=0.033, payer=True, swap_tenor="5Y",
                  forward_start=ORE.Period(2, ORE.Years), evaluation_date=ASOF, trade_id="european")
    fields.update(overrides)
    return SwaptionConfig(**fields)


EUROPEANS = {
    "payer": {},
    "receiver": {"payer": False},
    "payer-with-spread": {"floating_spread": 0.0025},
    "receiver-otm-short-expiry": {"payer": False, "fixed_rate": 0.02, "forward_start": ORE.Period(6, ORE.Months),
                                  "swap_tenor": "2Y"},
}


@pytest.mark.parametrize("name", EUROPEANS)
def test_european_today_equals_ores_default_engine(name):
    """ORE's `EuropeanSwaptionEngineBuilder` (`BlackMultiLegOptionEngine`) through the
    OREApp oracle, on two curves and the vol matrix; a spread is supported (I-37)."""
    from engine.instruments.european_swaption import _build_ore_swap as build_underlying
    cfg = _european(**EUROPEANS[name])
    ours = float(european_value(cfg, from_market(_oracle_market(VOLS))))
    ore = ore_lgm_swaption_npv(
        evaluation_date=ASOF, curve_times=ORACLE_PILLARS, curve_rates=ORACLE_DISC, swap=build_underlying(cfg),
        notional=cfg.notional, fixed_rate=cfg.fixed_rate, payer=cfg.payer, floating_spread=cfg.floating_spread,
        index_tenor_months=6, style="European", exercise_dates=[cfg.exercise_date], hw_a=0.03, hw_sigma=0.01,
        n_per_std=48, std_devs=6.0, index_curve_rates=ORACLE_INDEX, swaption_vols=VOLS).npv
    assert ours == pytest.approx(ore, rel=1e-10)


CASH_EUROPEANS = {
    "cash-payer": {"settlement": "Cash"},
    "cash-receiver": {"payer": False, "settlement": "Cash"},
    "cash-receiver-otm-short-expiry": {**EUROPEANS["receiver-otm-short-expiry"], "settlement": "Cash"},
}


def _ql_swaption(cfg, index, disc, vol):
    """QuantLib's `Swaption` on the engine's underlying schedules with `BachelierSwaptionEngine`
    (`vol` a quote or a volatility structure); cash settlement by `ParYieldCurve`, ORE's
    default method for it. The underlying's fixed leg is annual, so QuantLib's cash annuity
    (compounding at the fixed frequency) is ORE's (always annual)."""
    from engine.instruments.european_swaption import _build_ore_swap as build_underlying
    underlying = build_underlying(cfg)
    assert underlying.fixedSchedule().tenor() == ORE.Period(1, ORE.Years)
    swap = ORE.VanillaSwap(
        ORE.VanillaSwap.Payer if cfg.payer else ORE.VanillaSwap.Receiver, cfg.notional,
        underlying.fixedSchedule(), cfg.fixed_rate, DC, underlying.floatingSchedule(), index,
        cfg.floating_spread, DC)
    settlement = ((ORE.Settlement.Cash, ORE.Settlement.ParYieldCurve) if cfg.settlement == "Cash"
                  else (ORE.Settlement.Physical, ORE.Settlement.PhysicalOTC))
    swaption = ORE.Swaption(swap, ORE.EuropeanExercise(cfg.exercise_date), *settlement)
    if isinstance(vol, float):
        vol = ORE.QuoteHandle(ORE.SimpleQuote(vol))
    swaption.setPricingEngine(ORE.BachelierSwaptionEngine(disc, vol))
    return swaption


@pytest.mark.parametrize("name", CASH_EUROPEANS)
def test_a_cash_settled_european_uses_the_par_yield_annuity(name):
    """ORE prices a cash-settled European with `ParYieldCurve` settlement by default
    (`defaultSettlementMethod`); QuantLib's Bachelier engine is the same formula. Red first:
    with the physical annuity the engine was 1.1% to 2% off (measured 2026-09-29); now
    ~2e-14."""
    cfg = _european(**CASH_EUROPEANS[name])
    ORE.Settings.instance().evaluationDate = ASOF
    oracle_curve = lambda rates: _ore_zero_curve(rates, ORACLE_PILLARS)  # noqa: E731
    vols = _ore_vol_matrix()
    vols.enableExtrapolation()
    reference = _ql_swaption(cfg, ibor_index(6, oracle_curve(ORACLE_INDEX)), oracle_curve(ORACLE_DISC),
                             ORE.SwaptionVolatilityStructureHandle(vols)).NPV()
    ours = float(european_value(cfg, from_market(_oracle_market(VOLS))))
    physical = float(european_value(dataclasses.replace(cfg, settlement="Physical"),
                                    from_market(_oracle_market(VOLS))))
    assert ours == pytest.approx(reference, rel=1e-12)
    assert abs(physical / reference - 1.0) > 5e-3


@pytest.mark.parametrize("name", {**EUROPEANS, **CASH_EUROPEANS})
def test_european_on_every_path_equals_quantlibs_bachelier_engine(name, scenarios):
    """On each path and date before expiry: QuantLib's `BachelierSwaptionEngine` (the same
    formula as ORE's engine for a swap starting after expiry) on the path curves, with the
    volatility the time-decayed surface gives that date and the fixings FixingManager stored
    (a short-expiry option's first coupon can fix before its expiry)."""
    from engine.instruments.european_swaption import _build_ore_swap as build_underlying
    cfg = _european(**{**EUROPEANS, **CASH_EUROPEANS}[name])
    terms = european_terms(cfg, ASOF)
    variances = [variance_on_path(terms, VOLS, ASOF, d, "ForwardVariance") for d in scenarios.dates]
    fixings = path_fixings(6, ASOF, scenarios.dates, scenarios.times, scenarios.index[INDEX])
    cube = np.asarray(european_cube(terms, path_schedule(terms.legs, ASOF, scenarios.dates), scenarios.times,
                                    scenarios.discount["USD"], scenarios.index[INDEX], fixings, variances))
    serials = [d.serialNumber() for d in scenarios.dates]
    underlying = build_underlying(cfg)
    for j, date in enumerate(scenarios.dates):
        if not cfg.exercise_date > date:
            assert np.all(cube[:, j] == 0.0)
            continue
        for s in range(scenarios.num_paths):
            disc = _ore_path_curve(date, np.asarray(scenarios.discount["USD"].tenor_times[j]),
                                   np.asarray(scenarios.discount["USD"].log_discounts[s, j]))
            fwd = _ore_path_curve(date, np.asarray(scenarios.index[INDEX].tenor_times[j]),
                                  np.asarray(scenarios.index[INDEX].log_discounts[s, j]))
            ORE.Settings.instance().evaluationDate = date
            index = ibor_index(6, fwd)
            index.clearFixings()
            for c in underlying.floatingLeg():
                f = ORE.as_floating_rate_coupon(c).fixingDate()
                if ASOF <= f < date:
                    index.addFixing(f, float(fixings[s, np.searchsorted(serials, f.serialNumber(), side="right")]),
                                    True)
            vol = float(np.sqrt(variances[j] / DC.yearFraction(date, cfg.exercise_date)))
            swaption = _ql_swaption(cfg, index, disc, vol)
            try:
                assert cube[s, j] == pytest.approx(swaption.NPV(), rel=1e-10), (date, s)
            finally:
                index.clearFixings()
    ORE.Settings.instance().evaluationDate = ASOF


def test_forward_variance_decay_keeps_the_variance_to_expiry():
    """ForwardVariance: seen from any date the remaining variance is V(T) - V(t) on the t=0
    surface, so today's variance splits exactly into what has passed and what remains."""
    cfg = _european()
    terms = european_terms(cfg, ASOF)
    date = ORE.Date(3, 5, 2027)
    tf, tau = DC.yearFraction(ASOF, date), DC.yearFraction(date, cfg.exercise_date)
    passed = float(VOLS.black_variance(ASOF, tf, terms.swap_length))
    today = float(VOLS.black_variance(ASOF, DC.yearFraction(ASOF, cfg.exercise_date), terms.swap_length))
    assert variance_on_path(terms, VOLS, ASOF, date, "ForwardVariance") + passed == pytest.approx(today, rel=1e-12)
    constant = volatility_on_path(VOLS, ASOF, date, tau, terms.swap_length, "ConstantVariance")
    assert constant == pytest.approx(float(VOLS.volatility(ASOF, tau, terms.swap_length)), rel=1e-15)
    assert variance_on_path(terms, VOLS, ASOF, cfg.exercise_date, "ForwardVariance") == 0.0


# =============================================================================
# Bermudan / American swaptions: the trade's own calibrated LGM (T-13, V-5, I-47)
# =============================================================================
from engine.instruments.american_swaption import AmericanSwaptionConfig  # noqa: E402
from engine.instruments.bermudan_swaption import (  # noqa: E402
    BermudanSwaptionConfig, _build_ore_swap as build_option_underlying, exercisable_dates,
)
from engine.valuation.bermudan import bermudan_cube, bermudan_value  # noqa: E402
from engine.valuation.config import LgmSwaptionEngineConfig  # noqa: E402
from tests.support.ore_lgm_oracle import OreCalibration, OreDiscountCurves  # noqa: E402

ENGINE = LgmSwaptionEngineConfig()   # ORE's example settings (reversion 0, 30 points, 5 std devs)
FLAT_VOLS = SwaptionVolSurface(("1Y",), ("1Y",), ((0.0095,),))


def _oracle_market(vols, pillars=ORACLE_PILLARS, disc=ORACLE_DISC, index=ORACLE_INDEX):
    return Market(ASOF, {"USD": CurrencyMarket(
        discount_curve=ZeroCurveConfig(pillars, disc), index_curves={INDEX: ZeroCurveConfig(pillars, index)},
        swaption_vols=vols)})


def _bermudan_trade(**overrides):
    fields = dict(notional=1e6, fixed_rate=0.031, payer=True, exercise_dates=[ASOF + 400],
                  effective_date=ORE.Date(3, 2, 2027), maturity_date=ORE.Date(3, 2, 2033), evaluation_date=ASOF,
                  trade_id="bermudan")
    fields.update(overrides)
    booked = BermudanSwaptionConfig(**fields)
    if "exercise_dates" in overrides:
        return booked
    return dataclasses.replace(booked, exercise_dates=exercisable_dates(booked)[:-1])


def _american_trade(**overrides):
    fields = dict(notional=1e6, fixed_rate=0.031, payer=True, first_exercise_date=ORE.Date(3, 2, 2027),
                  last_exercise_date=ORE.Date(3, 2, 2031), effective_date=ORE.Date(3, 2, 2027),
                  maturity_date=ORE.Date(3, 2, 2033), evaluation_date=ASOF, trade_id="american")
    fields.update(overrides)
    return AmericanSwaptionConfig(**fields)


def _ore_option(cfg, evaluation_date, vols, discount_curves=None, fixings=None):
    american = isinstance(cfg, AmericanSwaptionConfig)
    return ore_lgm_swaption_npv(
        evaluation_date=evaluation_date, curve_times=ORACLE_PILLARS, curve_rates=ORACLE_DISC,
        swap=build_option_underlying(cfg), notional=cfg.notional, fixed_rate=cfg.fixed_rate, payer=cfg.payer,
        floating_spread=cfg.floating_spread, index_tenor_months=6, style="American" if american else "Bermudan",
        exercise_dates=[cfg.first_exercise_date, cfg.last_exercise_date] if american else cfg.exercise_dates,
        hw_a=ENGINE.reversion, hw_sigma=ENGINE.volatility, n_per_std=ENGINE.n_per_std, std_devs=ENGINE.std_devs,
        exercise_time_steps_per_year=ENGINE.exercise_time_steps_per_year, index_curve_rates=ORACLE_INDEX,
        swaption_vols=vols, calibration=OreCalibration("Bootstrap", ENGINE.strategy, 1e-8),
        discount_curves=discount_curves, fixings=fixings).npv


OPTIONS = {
    "bermudan-payer": lambda: _bermudan_trade(),
    "bermudan-receiver": lambda: _bermudan_trade(payer=False),
    "bermudan-with-spread": lambda: _bermudan_trade(floating_spread=0.002),
    "american-payer": lambda: _american_trade(),
}


@pytest.mark.parametrize("name", OPTIONS)
def test_option_today_equals_ores_calibrated_grid_engine(name):
    """ORE builds the co-terminal basket from the trade, bootstraps its LGM to the vol matrix
    on two curves (CoterminalDealStrike) and prices on the grid; so does the engine."""
    cfg = OPTIONS[name]()
    ours = bermudan_value(cfg, ENGINE, from_market(_oracle_market(VOLS)))
    assert ours == pytest.approx(_ore_option(cfg, ASOF, VOLS), rel=1e-9)


@pytest.fixture(scope="module", params=MODELS)
def option_scenarios(request):
    dates = (ORE.Date(15, 12, 2026), ORE.Date(10, 2, 2027), ORE.Date(3, 8, 2028))
    config = CamConfig(dates=dates, base_currency="USD", ir={"USD": MODELS[request.param](0.03, 0.012)},
                       curve_tenors=TENORS, samples=3, seed=11)
    return simulate(_oracle_market(FLAT_VOLS), config)


def _path_oracle_inputs(cfg, sm, j, s, fixings):
    date = sm.dates[j]
    tenor_dates = _tenor_dates(date)
    curves = OreDiscountCurves(
        tenor_dates, np.exp(np.asarray(sm.discount["USD"].log_discounts[s, j, 1:])),
        tenor_dates, np.exp(np.asarray(sm.index[INDEX].log_discounts[s, j, 1:])))
    serials = [d.serialNumber() for d in sm.dates]
    fixing_dates = [ORE.as_floating_rate_coupon(c).fixingDate() for c in build_option_underlying(cfg).floatingLeg()]
    history = {f: float(fixings[s, np.searchsorted(serials, f.serialNumber(), side="right")])
               for f in fixing_dates if ASOF <= f < date}
    return curves, history


def test_bermudan_on_every_path_equals_ore_recalibrated_on_the_path_curves(option_scenarios):
    """On each path and date, ORE recalibrates the trade's LGM to its basket on the path's own
    curves (handed over as log-linear discount factors) and prices on the grid, with
    FixingManager's fixings. A flat vol surface makes the time-decay rule immaterial."""
    cfg = OPTIONS["bermudan-payer"]()
    sm = option_scenarios
    fixings = path_fixings(6, ASOF, sm.dates, sm.times, sm.index[INDEX])
    cube = np.asarray(bermudan_cube(cfg, ENGINE, _oracle_market(FLAT_VOLS), sm, fixings, "ForwardVariance"))
    for j, date in enumerate(sm.dates):
        for s in range(sm.num_paths):
            curves, history = _path_oracle_inputs(cfg, sm, j, s, fixings)
            ore = _ore_option(dataclasses.replace(cfg, evaluation_date=date), date, FLAT_VOLS, curves, history)
            assert cube[s, j] == pytest.approx(ore, rel=1e-8), (date, s)


def test_american_on_every_path_equals_ores_grid_engine_on_the_path_curves(option_scenarios):
    """An American's basket expiries are the reference-grid dates ORE's builder generated on
    the as-of date (it builds the trade once, at t=0), so a trade built afresh on the path date
    -- the oracle -- calibrates to other expiries. Here ORE prices with the engine's own path
    calibration (`Calibration=None`): the rollback on the path curves, its exercise grid from
    the path date, and the fixings."""
    from engine.valuation.bermudan import path_sigmas
    from engine.models.lgm import Sigma
    cfg = OPTIONS["american-payer"]()
    sm = option_scenarios
    fixings = path_fixings(6, ASOF, sm.dates, sm.times, sm.index[INDEX])
    cube = np.asarray(bermudan_cube(cfg, ENGINE, _oracle_market(FLAT_VOLS), sm, fixings, "ForwardVariance"))
    sigmas = path_sigmas(cfg, ENGINE, _oracle_market(FLAT_VOLS), sm, "ForwardVariance")
    for j, date in enumerate(sm.dates):
        sigma = sigmas[j]
        for s in range(sm.num_paths):
            curves, history = _path_oracle_inputs(cfg, sm, j, s, fixings)
            ore = ore_lgm_swaption_npv(
                evaluation_date=date, curve_times=ORACLE_PILLARS, curve_rates=ORACLE_DISC,
                swap=build_option_underlying(cfg), notional=cfg.notional, fixed_rate=cfg.fixed_rate, payer=True,
                floating_spread=0.0, index_tenor_months=6, style="American",
                exercise_dates=[cfg.first_exercise_date, cfg.last_exercise_date], hw_a=ENGINE.reversion,
                hw_sigma=Sigma(sigma.times, sigma.values[s]), n_per_std=ENGINE.n_per_std, std_devs=ENGINE.std_devs,
                exercise_time_steps_per_year=ENGINE.exercise_time_steps_per_year, index_curve_rates=ORACLE_INDEX,
                swaption_vols=FLAT_VOLS, discount_curves=curves, fixings=history).npv
            assert cube[s, j] == pytest.approx(ore, rel=1e-10), (date, s)


def test_an_americans_basket_keeps_the_as_of_grid_on_later_dates():
    """ORE's builder sets an American's basket expiries once, from the reference grid on the
    as-of date; on a later date only those still ahead remain."""
    from engine.valuation.bermudan import basket_expiries
    from engine.valuation.config import reference_grid_dates
    cfg = OPTIONS["american-payer"]()
    later = ORE.Date(15, 12, 2026)
    grid = reference_grid_dates(ASOF, ENGINE.reference_calibration_grid)
    expected = [d for d in grid if cfg.first_exercise_date <= d < cfg.last_exercise_date and d > later]
    assert list(basket_expiries(cfg, ENGINE, later, ASOF)) == expected
    assert list(basket_expiries(cfg, ENGINE, ASOF, ASOF))[0] == grid[[d >= cfg.first_exercise_date for d in grid].index(True)]


# =============================================================================
# The portfolio on the scenario market: ORE's ValuationEngine::buildCube (T-8, T-11)
# =============================================================================
from engine.calibration.cam import calibrate_cam  # noqa: E402
from engine.instruments.treasury import BondConfig, CouponPeriod, _remaining_cashflows  # noqa: E402
from engine.market import EquityMarket  # noqa: E402
from engine.valuation.config import PricingConfig  # noqa: E402
from engine.valuation.portfolio import value_portfolio, value_today  # noqa: E402

FAST_ENGINE = LgmSwaptionEngineConfig(n_per_std=12, std_devs=4.0)
FAST_PRICING = PricingConfig(bermudan=FAST_ENGINE, american=FAST_ENGINE)
EUR_INDEX = index_name("EUR", 6)


def _two_currency_market():
    usd = CurrencyMarket(ZeroCurveConfig(ORACLE_PILLARS, ORACLE_DISC),
                         {INDEX: ZeroCurveConfig(ORACLE_PILLARS, ORACLE_INDEX)}, VOLS)
    eur = CurrencyMarket(ZeroCurveConfig(ORACLE_PILLARS, [0.010, 0.010, 0.014, 0.020, 0.025, 0.030]),
                         {EUR_INDEX: ZeroCurveConfig(ORACLE_PILLARS, [0.013, 0.013, 0.017, 0.023, 0.028, 0.032])}, VOLS)
    return Market(ASOF, {"USD": usd, "EUR": eur}, fx_spots={"EURUSD": 1.1},
                  equities={"SP5": EquityMarket("USD", 100.0)})


def _bond():
    periods = tuple(CouponPeriod(ORE.Date(15, m, y), ORE.Date(15, m + 6 if m == 2 else 2, y if m == 2 else y + 1))
                    for y in (2026, 2027) for m in (2, 8))
    return BondConfig(trade_id="bond-L513", face_amount=100_000.0, maturity_date=ORE.Date(15, 2, 2028), evaluation_date=ASOF,
                      coupon_rate=0.04, coupon_schedule=periods)


def _portfolio():
    expiry_in = ORE.Period(3, ORE.Months)
    return [
        _swap(),
        _european(fixed_rate=0.005, forward_start=expiry_in),                       # deep ITM payer, physical
        _european(fixed_rate=0.005, forward_start=expiry_in, settlement="Cash"),    # the same, cash settled
        _bermudan_trade(),
        _american_trade(),
        _bond(),
        _swap(currency="EUR"),
    ]


@pytest.fixture(scope="module", params=MODELS)
def portfolio_run(request):
    market = _two_currency_market()
    dates = (ORE.Date(15, 9, 2026), ORE.Date(15, 12, 2026), ORE.Date(3, 8, 2028), ORE.Date(1, 3, 2033))
    model = MODELS[request.param]
    config = CamConfig(dates=dates, base_currency="USD",
                       ir={"USD": model(0.03, 0.01, ("1Y", "2Y", "5Y"), ("9Y", "8Y", "5Y")),
                           "EUR": model(0.02, 0.009)},
                       fx_volatilities={"EUR": 0.1}, correlations={("IR:USD", "FX:EURUSD"): 0.2},
                       curve_tenors=TENORS, samples=8, seed=5)
    scenarios = simulate(market, config)
    return market, config, scenarios, value_portfolio(_portfolio(), market, scenarios, "USD", FAST_PRICING)


def test_todays_npvs_are_each_pricers_own(portfolio_run):
    market, _, _, result = portfolio_run
    trades = _portfolio()
    assert result.today == pytest.approx(value_today(trades, market, "USD", FAST_PRICING), rel=1e-12)
    bond = trades[5]
    discount = _ore_zero_curve(ORACLE_DISC, ORACLE_PILLARS)
    expected = sum(bond.face_amount * a * discount.discount(d) for d, a in _remaining_cashflows(bond))
    assert result.today[5] == pytest.approx(expected, rel=1e-12)
    eur_in_eur = value_today([trades[6]], market, "EUR", FAST_PRICING)[0]
    assert result.today[6] == pytest.approx(1.1 * eur_in_eur, rel=1e-12)


def test_an_exercised_physical_option_becomes_its_swap_and_a_cash_one_leaves(portfolio_run):
    """The deep ITM European expires between the first two dates: on the second every path
    exercises; physical settlement carries the swap, cash settlement nothing."""
    market, _, sm, result = portfolio_run
    cube = np.asarray(result.cube)
    trade = _portfolio()[1]
    assert sm.dates[0] < trade.exercise_date <= sm.dates[1]
    underlying = SwapConfig(trade_id="swap-L562", notional=trade.notional, fixed_rate=trade.fixed_rate, payer=True,
                            effective_date=trade.effective_date, maturity_date=trade.maturity_date,
                            evaluation_date=ASOF)
    swap_cube = np.asarray(value_portfolio([underlying], market, sm, "USD", FAST_PRICING).cube[:, :, 0])
    assert np.all(cube[:, 0, 1] > 0) and np.all(cube[:, 0, 2] > 0)
    np.testing.assert_allclose(cube[:, 1:, 1], swap_cube[:, 1:], rtol=1e-12)
    assert np.all(cube[:, 1:, 2] == 0.0)


def test_a_bond_on_every_path_discounts_its_remaining_flows(portfolio_run):
    _, _, sm, result = portfolio_run
    cube = np.asarray(result.cube[:, :, 5])
    bond = _bond()
    from engine.instruments.treasury import _remaining_cashflows
    for j, date in enumerate(sm.dates):
        for s in range(sm.num_paths):
            curve = _ore_path_curve(date, np.asarray(sm.discount["USD"].tenor_times[j]),
                                    np.asarray(sm.discount["USD"].log_discounts[s, j]))
            expected = sum(bond.face_amount * a * curve.discount(d) for d, a in _remaining_cashflows(bond) if d > date)
            assert cube[s, j] == pytest.approx(expected, rel=1e-12, abs=1e-9)


def test_a_foreign_trade_is_converted_with_the_path_fx(portfolio_run):
    market, _, sm, result = portfolio_run
    eur_only = value_portfolio([_portfolio()[6]], market, sm, "USD", FAST_PRICING)
    np.testing.assert_allclose(np.asarray(result.cube[:, :, 6]), np.asarray(eur_only.cube[:, :, 0]), rtol=1e-14)
    swap = _portfolio()[6]
    legs = legs_of(_build_ore_swap(swap), True, ASOF, swap.fixings)
    fixings = path_fixings(6, ASOF, sm.dates, sm.times, sm.index[EUR_INDEX])
    in_eur = legs_cube(legs, path_schedule(legs, ASOF, sm.dates), sm.times, sm.discount["EUR"], sm.index[EUR_INDEX],
                       fixings)
    np.testing.assert_allclose(np.asarray(result.cube[:, :, 6]), np.asarray(in_eur * sm.fx["EUR"]), rtol=1e-14)


def test_the_cam_is_calibrated_to_its_basket(portfolio_run):
    market, config, _, _ = portfolio_run
    calibration = calibrate_cam(market, config.ir)
    assert set(calibration) == {"USD"}
    np.testing.assert_allclose(calibration["USD"].model, calibration["USD"].market, rtol=1e-12)


class TestMarketPathRefusals:
    def test_a_trade_carries_no_model(self):
        """The model is the pricing configuration's (I-63): a trade given one is a TypeError
        at construction (tests/test_trade_configs.py for every type and field)."""
        with pytest.raises(TypeError, match="hw_a"):
            _european(hw_a=0.03)

    def test_a_trade_valued_on_another_date_is_refused(self):
        with pytest.raises(ValueError, match="as-of"):
            value_today([_swap(evaluation_date=ASOF + 1)], _two_currency_market(), "USD")

    def test_a_missing_index_curve_is_refused_not_replaced(self):
        with pytest.raises(KeyError, match="SIMINDEX-3M"):
            value_today([_swap(index_tenor_months=3)], _two_currency_market(), "USD")

    def test_a_missing_currency_is_refused(self):
        with pytest.raises(KeyError, match="GBP"):
            value_today([_swap(currency="GBP")], _two_currency_market(), "USD")
