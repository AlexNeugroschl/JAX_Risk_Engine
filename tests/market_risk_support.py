"""
Shared fixtures for the market-risk tests: a sloped two-curve market, a
mixed portfolio on it, and independent ORE pricers for each instrument
under an arbitrary (shocked) set of pillar rates.

The ORE side never reuses engine code for pricing. Curves are
`ORE.ZeroCurve` (linear in the continuously compounded ACT/365 zero rate,
the engine's convention) with pillars on whole days, so ORE's and the
engine's year fractions are identical. ORE extrapolates a zero curve
linearly past its last pillar where the engine holds it flat, so the last
pillar (30y) lies beyond every cashflow.
"""
import numpy as np
import ORE

from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig
from engine.instruments.treasury import BondConfig, CouponPeriod
from engine.market_risk import RateRiskFactors
from engine.simulation.market_model import ZeroCurveConfig

TODAY = ORE.Date(30, 7, 2026)
DAY_COUNTER = ORE.Actual365Fixed()
PILLAR_DAYS = [0, 365, 730, 1095, 1825, 2555, 3650, 5475, 7300, 10950]
PILLAR_TIMES = [d / 365 for d in PILLAR_DAYS]
OIS = ZeroCurveConfig(PILLAR_TIMES, [0.030, 0.031, 0.032, 0.033, 0.035, 0.037, 0.039, 0.041, 0.042, 0.043])
IBOR = ZeroCurveConfig(PILLAR_TIMES, [r + 0.004 for r in OIS.rates])
HW_A, HW_SIGMA = 0.03, 0.01


def factors() -> RateRiskFactors:
    return RateRiskFactors.from_curves([OIS, IBOR], ["OIS", "IBOR"])


def covariance(daily_vol: float = 0.0008, horizon_days: int = 10, decay: float = 4.0) -> np.ndarray:
    """Horizon covariance of absolute pillar moves: equal vols, correlation
    decaying with pillar distance, the two curves' matching pillars highly
    correlated."""
    n = len(PILLAR_TIMES)
    idx = np.arange(2 * n)
    pillar, curve = idx % n, idx // n
    corr = np.exp(-np.abs(pillar[:, None] - pillar[None, :]) / decay)
    corr = corr * np.where(curve[:, None] == curve[None, :], 1.0, 0.95)
    np.fill_diagonal(corr, 1.0)
    return corr * (daily_vol ** 2) * horizon_days


def swap() -> SwapConfig:
    return SwapConfig(notional=10e6, fixed_rate=0.036, payer=True, discount_curve_index=0,
                      forward_curve_index=1, swap_tenor="10Y", evaluation_date=TODAY)


def european() -> SwaptionConfig:
    return SwaptionConfig(notional=5e6, fixed_rate=0.037, payer=False, rate_factor_index=0, hw_a=HW_A,
                          hw_sigma=HW_SIGMA, initial_zero_curve=OIS, swap_tenor="5Y",
                          forward_start=ORE.Period(2, ORE.Years), evaluation_date=TODAY)


def bermudan(n_per_std: int = 16, std_devs: float = 5.0) -> BermudanSwaptionConfig:
    return BermudanSwaptionConfig(notional=3e6, fixed_rate=0.035, payer=True, rate_factor_index=0, hw_a=HW_A,
                                  hw_sigma=HW_SIGMA, initial_zero_curve=OIS,
                                  exercise_dates=[TODAY + ORE.Period(y, ORE.Years) for y in (1, 2, 3, 4)],
                                  swap_tenor="5Y", evaluation_date=TODAY, n_per_std=n_per_std, std_devs=std_devs)


def bond() -> BondConfig:
    periods = tuple(CouponPeriod(ORE.Date(30, 7, 2026 + k), ORE.Date(30, 7, 2027 + k)) for k in range(3))
    return BondConfig(face_amount=2e6, maturity_date=ORE.Date(30, 7, 2029), evaluation_date=TODAY,
                      initial_zero_curve=OIS, coupon_rate=0.04, coupon_schedule=periods, curve_index=0)


# ---------------------------------------------------------------------------
# ORE pricers, each a function of the two curves' pillar rates.
# ---------------------------------------------------------------------------
def ore_curve(rates) -> "ORE.YieldTermStructureHandle":
    dates = [TODAY + d for d in PILLAR_DAYS]
    curve = ORE.ZeroCurve(dates, [float(r) for r in rates], DAY_COUNTER)
    curve.enableExtrapolation()
    return ORE.YieldTermStructureHandle(curve)


def _index(forwarding) -> "ORE.IborIndex":
    return ORE.IborIndex("SimIndex", ORE.Period(6, ORE.Months), 2, ORE.USDCurrency(), ORE.TARGET(),
                         ORE.ModifiedFollowing, False, DAY_COUNTER, forwarding)


def ore_swap_npv(cfg: SwapConfig, disc_rates, fwd_rates) -> float:
    ORE.Settings.instance().evaluationDate = TODAY
    disc, fwd = ore_curve(disc_rates), ore_curve(fwd_rates)
    swap_type = ORE.VanillaSwap.Payer if cfg.payer else ORE.VanillaSwap.Receiver
    instrument = ORE.MakeVanillaSwap(
        ORE.Period(cfg.swap_tenor), _index(fwd), cfg.fixed_rate, nominal=cfg.notional, swapType=swap_type,
        fixedLegDayCount=DAY_COUNTER, floatingLegDayCount=DAY_COUNTER,
    )
    instrument.setPricingEngine(ORE.DiscountingSwapEngine(disc))
    return instrument.NPV()


def ore_european_npv(cfg: SwaptionConfig, rates) -> float:
    ORE.Settings.instance().evaluationDate = TODAY
    curve = ore_curve(rates)
    swap_type = ORE.VanillaSwap.Payer if cfg.payer else ORE.VanillaSwap.Receiver
    underlying = ORE.MakeVanillaSwap(
        ORE.Period(cfg.swap_tenor), _index(curve), cfg.fixed_rate, nominal=cfg.notional, swapType=swap_type,
        fixedLegDayCount=DAY_COUNTER, floatingLegDayCount=DAY_COUNTER, forwardStart=cfg.forward_start,
    )
    start = ORE.TARGET().advance(TODAY, cfg.forward_start)
    exercise = ORE.EuropeanExercise(ORE.TARGET().advance(start, ORE.Period(cfg.exercise_lag_days, ORE.Days)))
    swaption = ORE.Swaption(underlying, exercise)
    swaption.setPricingEngine(ORE.JamshidianSwaptionEngine(ORE.HullWhite(curve, cfg.hw_a, cfg.hw_sigma), curve))
    return swaption.NPV()


def ore_bond_npv(cfg: BondConfig, rates) -> float:
    """ORE's own fixed-rate bond (ACT/ACT ISMA coupons, zero settlement days)
    under its discounting engine: the dirty value of every remaining flow."""
    ORE.Settings.instance().evaluationDate = TODAY
    dates = [p.start_date for p in cfg.coupon_schedule] + [cfg.coupon_schedule[-1].end_date]
    schedule = ORE.Schedule(dates, ORE.NullCalendar(), ORE.Unadjusted)
    instrument = ORE.FixedRateBond(0, cfg.face_amount, schedule, [cfg.coupon_rate],
                                   ORE.ActualActual(ORE.ActualActual.ISMA))
    instrument.setPricingEngine(ORE.DiscountingBondEngine(ore_curve(rates)))
    return instrument.NPV()
