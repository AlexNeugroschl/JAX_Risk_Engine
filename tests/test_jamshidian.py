"""
The Jamshidian European engine (`engine.valuation.jamshidian`, `PricingConfig.european =
"Jamshidian"`) against QuantLib's `JamshidianSwaptionEngine` on `HullWhite(curve, a, sigma)`, and
`SwaptionConfig`'s own validation.

The engine is QuantLib's formula in the LGM form of the Hull-White model, so it needs only a
curve's discount factors: today's, a bumped one or a simulated path's. It equals QuantLib's
formula evaluated with an exact root to 1e-10 (`TestAgainstQuantLibWithAnExactRoot`); QuantLib's
engine itself solves its root r* by Brent to 1e-8 in r, which moves its price by up to 1.8e-6
relative, so the direct comparisons allow 5e-6. `TestOnALaterDatesCurve` checks the path case,
pricing on a later date's curve as QuantLib rebuilt there does.
"""
import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.instruments.european_swaption import SwaptionConfig
from engine.market import CurrencyMarket, Market, ZeroCurveConfig, index_name
from engine.models.curves import DiscountCurve, ZeroCurve
from engine.models.hull_white import bond_call as _bond_call, bond_put as _bond_put
from engine.valuation.config import JamshidianEngineConfig, PricingConfig
from engine.valuation.context import PricingContext
from engine.valuation.european import european_terms
from engine.valuation.jamshidian import jamshidian_value
from engine.valuation.portfolio import validate_trades

TODAY = ORE.Date(30, 7, 2026)
HW_A = 0.03
HW_SIGMA = 0.01
FLAT_RATE = 0.03
ZERO_CURVE = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[FLAT_RATE] * 6)
MODEL = JamshidianEngineConfig(HW_A, HW_SIGMA)
DC = ORE.Actual365Fixed()
#: QuantLib's engine solves r* by Brent to 1e-8 in r, the engine exactly: measured up to
#: 1.8e-6 relative (a 2Y spot-starting ATM payer). `TestAgainstQuantLibWithAnExactRoot`
#: removes QuantLib's root error and agrees to 1e-10.
RTOL = 5e-6


def _context(curve, date=TODAY) -> PricingContext:
    return PricingContext(date=date, discount={"USD": curve}, index={}, volatility=None)


def _price_at_t0(cfg: SwaptionConfig, model: JamshidianEngineConfig = MODEL, curve: ZeroCurveConfig = ZERO_CURVE) -> float:
    return float(jamshidian_value(cfg, _context(ZeroCurve.from_config(curve)), model))


def _make_cfg(fixed_rate: float, payer: bool, tenor: str = "5Y", forward_start_years: int = 0, **fields) -> SwaptionConfig:
    return SwaptionConfig(
        notional=1_000_000.0, fixed_rate=fixed_rate, payer=payer, swap_tenor=tenor,
        forward_start=ORE.Period(forward_start_years, ORE.Years) if forward_start_years else ORE.Period(0, ORE.Days),
        evaluation_date=TODAY, trade_id="european", **fields)


def _ore_swaption_npv(fixed_rate, payer, tenor, forward_start=None, a=HW_A, sigma=HW_SIGMA, rate=FLAT_RATE,
                      exercise_lag=2):
    """QuantLib's `JamshidianSwaptionEngine` on `HullWhite(FlatForward(rate), a, sigma)` for the
    swaption `_make_cfg` books (same index and ACT/365 legs as `build_vanilla_swap`)."""
    ORE.Settings.instance().evaluationDate = TODAY
    curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, rate, DC))
    index = ORE.IborIndex("SimIndex", ORE.Period(6, ORE.Months), 2, ORE.USDCurrency(), ORE.TARGET(),
                          ORE.ModifiedFollowing, False, DC, curve)
    forward_start = forward_start or ORE.Period(0, ORE.Days)
    swap = ORE.MakeVanillaSwap(ORE.Period(tenor), index, fixed_rate, nominal=1_000_000.0,
                               swapType=ORE.VanillaSwap.Payer if payer else ORE.VanillaSwap.Receiver,
                               fixedLegDayCount=DC, floatingLegDayCount=DC, forwardStart=forward_start)
    exercise_date = ORE.TARGET().advance(ORE.TARGET().advance(TODAY, forward_start), exercise_lag, ORE.Days)
    swaption = ORE.Swaption(swap, ORE.EuropeanExercise(exercise_date))
    swaption.setPricingEngine(ORE.JamshidianSwaptionEngine(ORE.HullWhite(curve, a, sigma), curve))
    return swaption.NPV()


def _ore_years(fixed_rate, payer, tenor, forward_start_years=0, **kw):
    start = ORE.Period(forward_start_years, ORE.Years) if forward_start_years else None
    return _ore_swaption_npv(fixed_rate, payer, tenor, start, **kw)


def _ore_swap_npv(fixed_rate, payer, tenor="5Y"):
    ORE.Settings.instance().evaluationDate = TODAY
    curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, FLAT_RATE, DC))
    index = ORE.IborIndex("SimIndex", ORE.Period(6, ORE.Months), 2, ORE.USDCurrency(), ORE.TARGET(),
                          ORE.ModifiedFollowing, False, DC, curve)
    swap = ORE.MakeVanillaSwap(ORE.Period(tenor), index, fixed_rate, nominal=1_000_000.0,
                               swapType=ORE.VanillaSwap.Payer if payer else ORE.VanillaSwap.Receiver,
                               fixedLegDayCount=DC, floatingLegDayCount=DC)
    swap.setPricingEngine(ORE.DiscountingSwapEngine(curve))
    return swap.NPV()


class TestAgainstQuantLibJamshidianEngine:
    @pytest.mark.parametrize("fixed_rate,payer,tenor", [
        (0.03, True, "5Y"),   # ATM payer
        (0.03, False, "5Y"),  # ATM receiver
        (0.02, True, "10Y"),  # ITM payer
        (0.05, True, "2Y"),   # deep OTM payer
        (0.05, False, "2Y"),  # ITM receiver
        (0.02, False, "10Y"),  # deep OTM receiver
    ])
    def test_matches_quantlib_spot_starting(self, fixed_rate, payer, tenor):
        mine = _price_at_t0(_make_cfg(fixed_rate, payer, tenor))
        assert mine == pytest.approx(_ore_years(fixed_rate, payer, tenor), rel=RTOL, abs=1e-6)

    @pytest.mark.parametrize("fixed_rate,payer,tenor,forward_years", [
        (0.03, True, "2Y", 3), (0.03, False, "2Y", 3), (0.025, True, "5Y", 5),
    ])
    def test_matches_quantlib_forward_starting(self, fixed_rate, payer, tenor, forward_years):
        """The forward-start decomposition: strikes relative to the bond maturing on the
        swap's start, bond options from it (`discountBondOption(..., bondStart, ...)`)."""
        mine = _price_at_t0(_make_cfg(fixed_rate, payer, tenor, forward_years))
        assert mine == pytest.approx(_ore_years(fixed_rate, payer, tenor, forward_years), rel=RTOL)

    def test_deep_otm_is_near_zero(self):
        mine = _price_at_t0(_make_cfg(0.20, True, "5Y"))
        assert mine >= 0.0
        assert mine == pytest.approx(_ore_years(0.20, True, "5Y"), abs=1e-6)

    @pytest.mark.parametrize("fixed_rate", [0.01, 0.02, 0.03, 0.04, 0.06])
    @pytest.mark.parametrize("payer", [True, False])
    @pytest.mark.parametrize("tenor,forward_years", [("2Y", 0), ("5Y", 0), ("10Y", 2), ("5Y", 5)])
    def test_grid(self, fixed_rate, payer, tenor, forward_years):
        mine = _price_at_t0(_make_cfg(fixed_rate, payer, tenor, forward_years))
        assert mine == pytest.approx(_ore_years(fixed_rate, payer, tenor, forward_years), rel=RTOL, abs=1e-6)

    @pytest.mark.parametrize("a,sigma", [
        (0.01, 0.005), (0.01, 0.02), (0.05, 0.005), (0.05, 0.02), (0.1, 0.01), (0.3, 0.01), (1.0, 0.01),
        (2.0, 0.01),
    ])
    @pytest.mark.parametrize("fixed_rate,payer", [(0.02, True), (0.03, False), (0.04, True)])
    def test_grid_across_model_parameters(self, a, sigma, fixed_rate, payer):
        mine = _price_at_t0(_make_cfg(fixed_rate, payer), JamshidianEngineConfig(a, sigma))
        assert mine == pytest.approx(_ore_years(fixed_rate, payer, "5Y", a=a, sigma=sigma), rel=RTOL, abs=1e-6)

    @pytest.mark.parametrize("rate", [-0.005, 1e-6])
    def test_negative_and_near_zero_rates(self, rate):
        """Hull-White is a normal model: negative rates price, as in QuantLib."""
        curve = ZeroCurveConfig(times=ZERO_CURVE.times, rates=[rate] * 6)
        mine = _price_at_t0(_make_cfg(rate, True), curve=curve)
        assert mine == pytest.approx(_ore_years(rate, True, "5Y", rate=rate), rel=RTOL)

    def test_deeply_negative_rates_price_finitely(self):
        curve = ZeroCurveConfig(times=ZERO_CURVE.times, rates=[-0.10] * 6)
        mine = _price_at_t0(_make_cfg(-0.10, True), curve=curve)
        assert np.isfinite(mine) and mine >= -1e-6

    @pytest.mark.parametrize("fixed_rate,payer", [(1.5, True), (1.5, False), (-0.99, False)])
    def test_extreme_moneyness(self, fixed_rate, payer):
        """Far from the money the root lies far out; the window is shifted until it brackets
        it. (The -99% payer, where QuantLib's own engine loses 1.3e-3 to its Brent root, is in
        `TestAgainstQuantLibWithAnExactRoot`.)"""
        mine = _price_at_t0(_make_cfg(fixed_rate, payer))
        assert np.isfinite(mine) and mine >= -1e-6
        assert mine == pytest.approx(_ore_years(fixed_rate, payer, "5Y"), rel=RTOL, abs=1e-6)

    def test_very_short_and_very_long_expiries(self):
        short = SwaptionConfig(notional=1_000_000.0, fixed_rate=0.03, payer=True, swap_tenor="1Y",
                               forward_start=ORE.Period(1, ORE.Months), evaluation_date=TODAY, trade_id="european")
        assert _price_at_t0(short) == pytest.approx(
            _ore_swaption_npv(0.03, True, "1Y", ORE.Period(1, ORE.Months)), rel=RTOL)
        assert _price_at_t0(_make_cfg(0.03, True, "5Y", 20)) == pytest.approx(
            _ore_years(0.03, True, "5Y", 20), rel=RTOL)

    @pytest.mark.parametrize("payer, fixed_rate", [(True, 0.03), (False, 0.025)])
    def test_single_coupon_underlying(self, payer, fixed_rate):
        """A 1Y swap has one fixed coupon: the smallest decomposition."""
        cfg = _make_cfg(fixed_rate, payer, "1Y")
        assert european_terms(cfg, TODAY).legs.fixed_pay.size == 1
        assert _price_at_t0(cfg) == pytest.approx(_ore_years(fixed_rate, payer, "1Y"), rel=RTOL)

    def test_zero_exercise_lag(self):
        cfg = SwaptionConfig(notional=1_000_000.0, fixed_rate=0.03, payer=True, swap_tenor="5Y", exercise_lag_days=0,
                             evaluation_date=TODAY, trade_id="european")
        assert _price_at_t0(cfg) == pytest.approx(_ore_swaption_npv(0.03, True, "5Y", exercise_lag=0), rel=RTOL)


def _quantlib_with_exact_root(cfg: SwaptionConfig, a=HW_A, sigma=HW_SIGMA) -> float:
    """QuantLib's `JamshidianSwaptionEngine::calculate` step by step on its own `HullWhite`:
    `rStarFinder` on `discountBond`, solved to 1e-14 instead of QuantLib's Brent tolerance of
    1e-8, and each coupon's `discountBondOption(type, strike, maturity, bondStart,
    bondMaturity)` (not bound in Python: its formula, from ql/models/shortrate/
    onefactormodels/hullwhite.cpp, on QuantLib's discount factors)."""
    from scipy.optimize import brentq
    from scipy.stats import norm

    ORE.Settings.instance().evaluationDate = TODAY
    curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, FLAT_RATE, DC))
    hw = ORE.HullWhite(curve, a, sigma)
    terms = european_terms(cfg, TODAY)
    maturity, start = terms.expiry_time, terms.start_time
    times = list(terms.legs.fixed_pay)
    amounts = list(terms.legs.fixed_amount)
    amounts[-1] += terms.nominal

    def finder(r):
        b = hw.discountBond(maturity, start, r)
        return terms.nominal - sum(c * hw.discountBond(maturity, t, r) / b for c, t in zip(amounts, times))

    r_star = brentq(finder, -10.0, 10.0, xtol=1e-14, rtol=1e-15)
    b = hw.discountBond(maturity, start, r_star)
    w = -1.0 if cfg.payer else 1.0   # put for a payer, as QuantLib
    value = 0.0
    for c, t in zip(amounts, times):
        e = np.exp
        spread = (e(-2 * a * (start - maturity)) - e(-2 * a * start) - 2 * (e(-a * (start + t - 2 * maturity))
                  - e(-a * (start + t))) + e(-2 * a * (t - maturity)) - e(-2 * a * t))
        v = sigma / (a * np.sqrt(2 * a)) * np.sqrt(max(spread, 0.0))
        f, k = curve.discount(t), curve.discount(start) * hw.discountBond(maturity, t, r_star) / b
        d1 = np.log(f / k) / v + 0.5 * v
        value += c * w * (f * norm.cdf(w * d1) - k * norm.cdf(w * (d1 - v)))
    return value


class TestAgainstQuantLibWithAnExactRoot:
    """With QuantLib's root solved exactly, the engine is QuantLib's formula to rounding: the
    differences from QuantLib's engine above are its Brent tolerance on r*."""

    @pytest.mark.parametrize("fixed_rate,payer,tenor,forward_years", [
        (0.02, True, "10Y", 0), (0.03, True, "2Y", 0), (0.03, False, "5Y", 0), (0.03, True, "2Y", 3),
        (0.05, False, "5Y", 5), (0.01, True, "10Y", 2), (1.5, False, "5Y", 0),
    ])
    def test_matches(self, fixed_rate, payer, tenor, forward_years):
        cfg = _make_cfg(fixed_rate, payer, tenor, forward_years)
        assert _price_at_t0(cfg) == pytest.approx(_quantlib_with_exact_root(cfg), rel=1e-10)

    def test_a_fixed_rate_of_minus_99_percent(self):
        """A payer at -99%: the root's window [-2, 2] must shift to bracket it (it once
        returned the edge, an NPV wrong in sign and size). The fixed coupons (~ -990,000 each)
        nearly cancel the notional, so the formula carries ~7 digits: the engine and QuantLib's
        formula at the exact root agree to 1.2e-7, and QuantLib's own engine, whose Brent root
        cannot resolve the cancellation, is 1.3e-3 away."""
        cfg = _make_cfg(-0.99, True)
        exact = _quantlib_with_exact_root(cfg)
        assert _price_at_t0(cfg) == pytest.approx(exact, rel=1e-6)
        assert abs(_ore_years(-0.99, True, "5Y") / exact - 1.0) > 1e-4

    def test_quantlibs_engine_is_off_by_its_root_tolerance(self):
        """The 2Y spot-starting ATM payer: QuantLib's engine is 1.8e-6 away from its own
        formula at the exact root, the engine 1e-12."""
        cfg = _make_cfg(0.03, True, "2Y")
        exact = _quantlib_with_exact_root(cfg)
        assert abs(_ore_years(0.03, True, "2Y") / exact - 1.0) > 1e-7
        assert _price_at_t0(cfg) == pytest.approx(exact, rel=1e-10)


class TestLimits:
    def test_payer_minus_receiver_is_the_swap(self):
        """Put-call parity, model free: payer - receiver = the underlying swap."""
        payer, receiver = _price_at_t0(_make_cfg(0.03, True)), _price_at_t0(_make_cfg(0.03, False))
        assert payer - receiver == pytest.approx(_ore_swap_npv(0.03, True), rel=1e-9, abs=1e-6)

    @pytest.mark.parametrize("fixed_rate", [0.02, 0.05])
    def test_vanishing_volatility_is_the_intrinsic_value(self, fixed_rate):
        mine = _price_at_t0(_make_cfg(fixed_rate, True), JamshidianEngineConfig(HW_A, 1e-10))
        assert mine == pytest.approx(max(_ore_swap_npv(fixed_rate, True), 0.0), abs=1e-2)

    def test_small_volatility_is_near_the_vanishing_one(self):
        small = _price_at_t0(_make_cfg(0.02, True), JamshidianEngineConfig(HW_A, 1e-5))
        vanishing = _price_at_t0(_make_cfg(0.02, True), JamshidianEngineConfig(HW_A, 1e-9))
        assert small == pytest.approx(vanishing, rel=1e-2)

    @pytest.mark.parametrize("a", [1e-8, 1e-6, 1e-4])
    def test_near_zero_reversion_is_finite_and_near_its_limit(self, a):
        """The engine's H and zeta use expm1, so a small reversion loses no digits (QuantLib's
        bond option volatility `sigma/(a sqrt(2a)) sqrt(c)` subtracts nearly equal exponentials
        and is 6.5e-5 off at a = 1e-4, which is why the QuantLib grid stops at 0.01)."""
        small = _price_at_t0(_make_cfg(0.03, True), JamshidianEngineConfig(a, HW_SIGMA))
        limit = _price_at_t0(_make_cfg(0.03, True), JamshidianEngineConfig(1e-9, HW_SIGMA))
        assert np.isfinite(small)
        assert small == pytest.approx(limit, rel=10 * a)

    def test_zero_notional_prices_to_zero(self):
        cfg = SwaptionConfig(notional=0.0, fixed_rate=0.03, payer=True, swap_tenor="5Y", evaluation_date=TODAY,
                             trade_id="european")
        assert _price_at_t0(cfg) == pytest.approx(0.0, abs=1e-6)

    def test_expired_is_worth_zero(self):
        cfg = _make_cfg(0.03, True, "5Y", 2)
        after = _context(ZeroCurve.from_config(ZERO_CURVE), cfg.exercise_date)
        assert float(jamshidian_value(cfg, after, MODEL)) == 0.0

    def test_bond_options_at_zero_volatility_are_intrinsic(self):
        """sigma_p = 0 at expiry: the Black-on-bond formula takes its intrinsic branch, with
        no NaN from 1/sigma_p even beside a positive sigma_p."""
        p_start, p_bond, k = jnp.array([0.95] * 3), jnp.array([0.90, 0.80, 0.70]), jnp.array([0.90] * 3)
        zero = jnp.zeros(3)
        np.testing.assert_allclose(np.asarray(_bond_call(p_start, p_bond, k, zero)),
                                   np.maximum(np.asarray(p_bond - k * p_start), 0.0), atol=1e-12)
        np.testing.assert_allclose(np.asarray(_bond_put(p_start, p_bond, k, zero)),
                                   np.maximum(np.asarray(k * p_start - p_bond), 0.0), atol=1e-12)
        assert bool(jnp.all(jnp.isfinite(_bond_call(p_start[:2], p_bond[:2], k[:2], jnp.array([0.0, 0.05])))))


# ---------------------------------------------------------------------------
# On a later date's curve: what the engine does on every simulated path
# ---------------------------------------------------------------------------
CURVES = {
    "flat": [FLAT_RATE] * 6,
    "upward": [0.010, 0.015, 0.020, 0.030, 0.040, 0.050],
    "inverted": [0.050, 0.045, 0.040, 0.030, 0.025, 0.020],
}
DATES = {"t0.50": ORE.Date(29, 1, 2027), "t1.51": ORE.Date(31, 1, 2028), "t2.25": ORE.Date(30, 10, 2028)}
# (short rate at the date, payer, fixed rate, tenor, forward start in years)
TRADES = {
    "r1%-payer": (0.010, True, 0.03, "5Y", 3),
    "r3.5%-receiver": (0.035, False, 0.03, "5Y", 3),
    "r6%-payer": (0.060, True, 0.03, "5Y", 3),
    "r3%-itm-payer-10Y": (0.030, True, 0.02, "10Y", 3),
}


def _later_curve(curve: str, date: ORE.Date, r: float):
    """A later date's curve: QuantLib's `HullWhite(curve).discountBond(t, T, r)` on daily
    dates, as (dates, discount factors). The Hull-White model conditioned on its short rate
    there, which is what a simulated path's curve is."""
    ORE.Settings.instance().evaluationDate = TODAY
    pillars = ZERO_CURVE.times
    today = ORE.YieldTermStructureHandle(ORE.ZeroCurve([TODAY + int(round(p * 365)) for p in pillars],
                                                       CURVES[curve], DC))
    hw = ORE.HullWhite(today, HW_A, HW_SIGMA)
    t = DC.yearFraction(TODAY, date)
    dates = [date + k for k in range(0, 365 * 16)]
    return dates, [1.0] + [hw.discountBond(t, DC.yearFraction(TODAY, d), r) for d in dates[1:]]


class TestOnALaterDatesCurve:
    """The engine on a later date's curve (measured from that date, log-linear as a path's
    scenario curve) equals QuantLib's engine rebuilt on that date on the same curve: pricing a
    European on a path. Three curve shapes, three dates, payers and receivers."""

    @pytest.mark.parametrize("curve", CURVES)
    @pytest.mark.parametrize("date_id", DATES)
    @pytest.mark.parametrize("trade_id", TRADES)
    def test_equals_quantlib_rebuilt_on_the_date(self, curve, date_id, trade_id):
        r, payer, fixed_rate, tenor, forward_years = TRADES[trade_id]
        date = DATES[date_id]
        dates, discounts = _later_curve(curve, date, r)
        cfg = _make_cfg(fixed_rate, payer, tenor, forward_years)
        assert cfg.exercise_date > date

        times = jnp.asarray([DC.yearFraction(date, d) for d in dates])
        path = DiscountCurve(times=times, log_discounts=jnp.log(jnp.asarray(discounts)))
        mine = float(jamshidian_value(cfg, _context(path, date), MODEL))

        ORE.Settings.instance().evaluationDate = date
        try:
            handle = ORE.YieldTermStructureHandle(ORE.DiscountCurve(dates, discounts, DC))
            index = ORE.IborIndex("SimIndex", ORE.Period(6, ORE.Months), 2, ORE.USDCurrency(), ORE.TARGET(),
                                  ORE.ModifiedFollowing, False, DC, handle)
            swap = ORE.VanillaSwap(ORE.VanillaSwap.Payer if payer else ORE.VanillaSwap.Receiver, 1_000_000.0,
                                   ORE.Schedule(cfg.effective_date, cfg.maturity_date, ORE.Period(1, ORE.Years),
                                                ORE.TARGET(), ORE.ModifiedFollowing, ORE.ModifiedFollowing,
                                                ORE.DateGeneration.Forward, False),
                                   fixed_rate, DC,
                                   ORE.Schedule(cfg.effective_date, cfg.maturity_date, ORE.Period(6, ORE.Months),
                                                ORE.TARGET(), ORE.ModifiedFollowing, ORE.ModifiedFollowing,
                                                ORE.DateGeneration.Forward, False),
                                   index, 0.0, DC)
            swaption = ORE.Swaption(swap, ORE.EuropeanExercise(cfg.exercise_date))
            swaption.setPricingEngine(ORE.JamshidianSwaptionEngine(ORE.HullWhite(handle, HW_A, HW_SIGMA), handle))
            ore = swaption.NPV()
        finally:
            ORE.Settings.instance().evaluationDate = TODAY
        assert mine == pytest.approx(ore, rel=RTOL, abs=1e-6)

    @pytest.mark.parametrize("payer", [True, False])
    def test_monotone_in_the_curves_level(self, payer):
        """On a later date a payer gains and a receiver loses as that date's curve rises."""
        cfg = _make_cfg(0.03, payer, "5Y", 3)
        date = DATES["t1.51"]
        values = []
        for r in np.linspace(-0.03, 0.10, 14):
            dates, discounts = _later_curve("flat", date, r)
            times = jnp.asarray([DC.yearFraction(date, d) for d in dates])
            values.append(float(jamshidian_value(
                cfg, _context(DiscountCurve(times, jnp.log(jnp.asarray(discounts))), date), MODEL)))
        steps = np.diff(values)
        assert np.all(steps >= -1e-6) if payer else np.all(steps <= 1e-6)


# ---------------------------------------------------------------------------
# Configuration: the engine's model, and what it refuses as QuantLib refuses it (I-37, I-41, I-52)
# ---------------------------------------------------------------------------
def _market():
    return Market(TODAY, {"USD": CurrencyMarket(ZERO_CURVE, {index_name("USD", 6): ZERO_CURVE})})


class TestConfiguration:
    def test_the_engine_needs_its_model(self):
        with pytest.raises(ValueError, match="PricingConfig.jamshidian"):
            PricingConfig(european="Jamshidian")

    def test_a_model_without_the_engine_is_refused(self):
        with pytest.raises(ValueError, match="does not read"):
            PricingConfig(jamshidian=MODEL)

    @pytest.mark.parametrize("a", [0.0, -0.01, float("nan")])
    def test_non_positive_reversion_is_refused(self, a):
        """QuantLib's `HullWhite` holds `a` positive; at a = 0 the bond option volatility would
        divide by zero and price at intrinsic (I-41)."""
        with pytest.raises(ValueError, match="reversion"):
            JamshidianEngineConfig(a, HW_SIGMA)

    @pytest.mark.parametrize("sigma", [0.0, -0.01, float("inf")])
    def test_non_positive_volatility_is_refused(self, sigma):
        with pytest.raises(ValueError, match="volatility"):
            JamshidianEngineConfig(HW_A, sigma)

    def test_quantlib_hull_white_refuses_zero_reversion(self):
        curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, FLAT_RATE, DC))
        with pytest.raises(RuntimeError, match="invalid value"):
            ORE.HullWhite(curve, 0.0, HW_SIGMA)

    def test_a_spread_is_refused_naming_the_trade(self):
        """The engine values the floating leg at par; a 100bp spread once priced exactly like
        none (I-37)."""
        cfg = _make_cfg(0.03, True, "5Y", 2, floating_spread=0.01)
        with pytest.raises(ValueError, match="'european'.*floating_spread"):
            validate_trades([cfg], _market(), PricingConfig(european="Jamshidian", jamshidian=MODEL))

    def test_quantlib_refuses_a_spread_too(self):
        ORE.Settings.instance().evaluationDate = TODAY
        curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, FLAT_RATE, DC))
        index = ORE.IborIndex("SimIndex", ORE.Period(6, ORE.Months), 2, ORE.USDCurrency(), ORE.TARGET(),
                              ORE.ModifiedFollowing, False, DC, curve)
        swap = ORE.MakeVanillaSwap(ORE.Period("5Y"), index, 0.03, forwardStart=ORE.Period(2, ORE.Years),
                                   floatingLegSpread=0.01)
        swaption = ORE.Swaption(swap, ORE.EuropeanExercise(TODAY + ORE.Period(2, ORE.Years)))
        swaption.setPricingEngine(ORE.JamshidianSwaptionEngine(ORE.HullWhite(curve, HW_A, HW_SIGMA), curve))
        with pytest.raises(RuntimeError, match="non zero spread"):
            swaption.NPV()

    def test_cash_settlement_is_refused(self):
        """ORE's cash method for a European is `ParYieldCurve`, which QuantLib's engine
        refuses; the Bachelier engine prices it (I-52)."""
        cfg = _make_cfg(0.03, True, "5Y", 2, settlement="Cash")
        with pytest.raises(ValueError, match="settlement"):
            validate_trades([cfg], _market(), PricingConfig(european="Jamshidian", jamshidian=MODEL))

    def test_quantlib_refuses_par_yield_cash_settlement_too(self):
        ORE.Settings.instance().evaluationDate = TODAY
        curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, FLAT_RATE, DC))
        index = ORE.IborIndex("SimIndex", ORE.Period(6, ORE.Months), 2, ORE.USDCurrency(), ORE.TARGET(),
                              ORE.ModifiedFollowing, False, DC, curve)
        swap = ORE.MakeVanillaSwap(ORE.Period("5Y"), index, 0.03, forwardStart=ORE.Period(2, ORE.Years))
        swaption = ORE.Swaption(swap, ORE.EuropeanExercise(TODAY + ORE.Period(2, ORE.Years)),
                                ORE.Settlement.Cash, ORE.Settlement.ParYieldCurve)
        swaption.setPricingEngine(ORE.JamshidianSwaptionEngine(ORE.HullWhite(curve, HW_A, HW_SIGMA), curve))
        with pytest.raises(RuntimeError, match="ParYieldCurve"):
            swaption.NPV()

    def test_the_engine_reads_no_volatility_surface(self):
        """A market without swaption volatilities is enough for the Jamshidian engine."""
        validate_trades([_make_cfg(0.03, True)], _market(), PricingConfig(european="Jamshidian", jamshidian=MODEL))
        with pytest.raises(KeyError, match="swaption volatilities"):
            validate_trades([_make_cfg(0.03, True)], _market())


class TestSwaptionConfigValidation:
    """`SwaptionConfig.__post_init__` rejects non-finite notional/fixed_rate and unparseable
    tenors. Zero notional is valid."""

    def test_nan_notional_rejected(self):
        with pytest.raises(ValueError, match="notional"):
            SwaptionConfig(notional=float("nan"), fixed_rate=0.03, payer=True, swap_tenor="5Y",
                           evaluation_date=TODAY, trade_id="european")

    def test_inf_fixed_rate_rejected(self):
        with pytest.raises(ValueError, match="fixed_rate"):
            SwaptionConfig(notional=1e6, fixed_rate=float("inf"), payer=True, swap_tenor="5Y",
                           evaluation_date=TODAY, trade_id="european")

    def test_unparseable_swap_tenor_rejected(self):
        with pytest.raises(ValueError, match="swap_tenor"):
            SwaptionConfig(notional=1e6, fixed_rate=0.03, payer=True, swap_tenor="bogus",
                           evaluation_date=TODAY, trade_id="european")

    def test_custom_exercise_lag_moves_the_exercise(self):
        assert _make_cfg(0.03, True, exercise_lag_days=5).exercise_date > _make_cfg(0.03, True).exercise_date

    def test_valid_config_constructs_without_error(self):
        _make_cfg(0.03, True, "5Y")  # must not raise
