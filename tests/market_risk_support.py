"""
Shared fixtures for the market-risk tests: a sloped two-curve USD market (an OIS discount
curve and the 6M index's forwarding curve), a mixed portfolio on it, the engines it is priced
with, and independent ORE pricers for each instrument under an arbitrary (shocked) set of
pillar rates.

The ORE side never reuses engine code for pricing. Curves are `ORE.ZeroCurve` (linear in the
continuously compounded ACT/365 zero rate, the engine's convention) with pillars on whole days,
so ORE's and the engine's year fractions are identical. The last pillar (30y) lies beyond every
cashflow, so extrapolation is not exercised.

The European is priced by ORE's default Bachelier engine on `VOLS` (`PricingConfig()`) or by
the Jamshidian engine on a Hull-White model (`JAMSHIDIAN`); the Bermudan by the LGM grid
engine, uncalibrated at `ENGINE`'s reversion and volatility (`PRICING`) so that ORE's engine
can be given the same model under every shocked curve.
"""
import numpy as np
import ORE

from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig
from engine.instruments.treasury import BondConfig, CouponPeriod
from engine.market import CurrencyMarket, Market, SwaptionVolSurface, ZeroCurveConfig, index_name
from engine.market_risk import RateRiskFactors
from engine.valuation.config import JamshidianEngineConfig, LgmSwaptionEngineConfig, PricingConfig

TODAY = ORE.Date(30, 7, 2026)
DAY_COUNTER = ORE.Actual365Fixed()
PILLAR_DAYS = [0, 365, 730, 1095, 1825, 2555, 3650, 5475, 7300, 10950]
PILLAR_TIMES = [d / 365 for d in PILLAR_DAYS]
OIS = ZeroCurveConfig(PILLAR_TIMES, [0.030, 0.031, 0.032, 0.033, 0.035, 0.037, 0.039, 0.041, 0.042, 0.043])
IBOR = ZeroCurveConfig(PILLAR_TIMES, [r + 0.004 for r in OIS.rates])
INDEX = index_name("USD", 6)
HW_A, HW_SIGMA = 0.03, 0.01
#: ATM normal swaption volatilities.
VOLS = SwaptionVolSurface(("1Y", "2Y", "5Y", "10Y"), ("1Y", "5Y", "10Y"),
                          ((0.0080, 0.0088, 0.0090), (0.0085, 0.0091, 0.0093), (0.0090, 0.0093, 0.0094),
                           (0.0092, 0.0094, 0.0096)))
#: The Bermudan's engine: ORE's LGM grid at a fixed model (no calibration).
ENGINE = LgmSwaptionEngineConfig(reversion=HW_A, volatility=HW_SIGMA, calibration="None", n_per_std=16,
                                 std_devs=5.0)
PRICING = PricingConfig(bermudan=ENGINE)
JAMSHIDIAN = PricingConfig(european="Jamshidian", jamshidian=JamshidianEngineConfig(HW_A, HW_SIGMA), bermudan=ENGINE)


def market() -> Market:
    return Market(TODAY, {"USD": CurrencyMarket(OIS, {INDEX: IBOR}, VOLS)})


def factors() -> RateRiskFactors:
    """The market's curves, `discount:USD` (OIS) then `index:USD-SIMINDEX-6M` (IBOR)."""
    return RateRiskFactors.from_market(market())


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
    return SwapConfig(notional=10e6, fixed_rate=0.036, payer=True, swap_tenor="10Y", evaluation_date=TODAY,
                      trade_id="swap")


def european() -> SwaptionConfig:
    return SwaptionConfig(notional=5e6, fixed_rate=0.037, payer=False, swap_tenor="5Y",
                          forward_start=ORE.Period(2, ORE.Years), evaluation_date=TODAY, trade_id="european")


def bermudan() -> BermudanSwaptionConfig:
    return BermudanSwaptionConfig(notional=3e6, fixed_rate=0.035, payer=True,
                                  exercise_dates=[TODAY + ORE.Period(y, ORE.Years) for y in (1, 2, 3, 4)],
                                  swap_tenor="5Y", evaluation_date=TODAY, trade_id="bermudan")


def bond() -> BondConfig:
    periods = tuple(CouponPeriod(ORE.Date(30, 7, 2026 + k), ORE.Date(30, 7, 2027 + k)) for k in range(3))
    return BondConfig(face_amount=2e6, maturity_date=ORE.Date(30, 7, 2029), evaluation_date=TODAY,
                      coupon_rate=0.04, coupon_schedule=periods, trade_id="bond")


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
        ORE.Period(0, ORE.Days), _index(fwd), cfg.fixed_rate, nominal=cfg.notional, swapType=swap_type,
        effectiveDate=cfg.effective_date, terminationDate=cfg.maturity_date,
        fixedLegDayCount=DAY_COUNTER, floatingLegDayCount=DAY_COUNTER,
    )
    instrument.setPricingEngine(ORE.DiscountingSwapEngine(disc))
    return instrument.NPV()


def _ore_swaption(cfg: SwaptionConfig, curve) -> "ORE.Swaption":
    ORE.Settings.instance().evaluationDate = TODAY
    swap_type = ORE.VanillaSwap.Payer if cfg.payer else ORE.VanillaSwap.Receiver
    underlying = ORE.MakeVanillaSwap(
        ORE.Period(0, ORE.Days), _index(curve), cfg.fixed_rate, nominal=cfg.notional, swapType=swap_type,
        effectiveDate=cfg.effective_date, terminationDate=cfg.maturity_date,
        fixedLegDayCount=DAY_COUNTER, floatingLegDayCount=DAY_COUNTER,
    )
    return ORE.Swaption(underlying, ORE.EuropeanExercise(cfg.exercise_date))


def ore_european_npv(cfg: SwaptionConfig, disc_rates, fwd_rates, engine: str = "Bachelier") -> float:
    """QuantLib's `BachelierSwaptionEngine` on ORE's own `SwaptionVolatilityMatrix` of `VOLS`
    (the formula of ORE's `BlackMultiLegOptionEngine` for a swap starting after expiry), the
    index forwarding on `fwd_rates`; or `JamshidianSwaptionEngine` on `HullWhite(HW_A,
    HW_SIGMA)`, which is single-curve (the floating leg at par on the discount curve)."""
    disc = ore_curve(disc_rates)
    if engine == "Bachelier":
        swaption = _ore_swaption(cfg, ore_curve(fwd_rates))
        swaption.setPricingEngine(ORE.BachelierSwaptionEngine(disc, ORE.SwaptionVolatilityStructureHandle(_ore_vols())))
    else:
        swaption = _ore_swaption(cfg, disc)
        swaption.setPricingEngine(ORE.JamshidianSwaptionEngine(ORE.HullWhite(disc, HW_A, HW_SIGMA), disc))
    return swaption.NPV()


def _ore_vols() -> "ORE.SwaptionVolatilityMatrix":
    matrix = ORE.Matrix(len(VOLS.option_tenors), len(VOLS.swap_tenors))
    for i, row in enumerate(VOLS.vols):
        for j, v in enumerate(row):
            matrix[i][j] = v
    vols = ORE.SwaptionVolatilityMatrix(
        ORE.TARGET(), ORE.Following, [ORE.Period(t) for t in VOLS.option_tenors],
        [ORE.Period(t) for t in VOLS.swap_tenors], matrix, DAY_COUNTER, True, ORE.Normal)
    vols.enableExtrapolation()
    return vols


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
