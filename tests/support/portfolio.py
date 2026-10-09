"""
The test portfolio shared across the ORE alignment's test layers (plan §6.3): one sloped
two-curve USD market and one trade of each kind the market path prices, so a discrepancy can
be followed from the HTTP result (L6) down to one trade against ORE (L2).

    swap-payer                     ACT/365, 5Y, payer
    swap-receiver-seasoned-icma    ACT/ACT (ICMA) legs, started before the as-of date, with
                                   its past fixing in `fixings`
    european-payer                 near the money, physically settled
    european-receiver-otm-cash     out of the money, 25bp spread, cash settled (ParYieldCurve)
    bermudan-payer-physical        exercise into whole periods
    bermudan-receiver-cash         cash settled
    american-payer
    bond                           a Treasury note with a coupon inside the first simulation step

Not included: an equity position (the engine has no equity trade yet; the CAM's equity
component is tested in tests/test_cam.py).

`trades()` are the configs (each named by its `trade_id`) and `trades_json()` the same trades
as the HTTP request takes them; `ore_npv(cfg)` is ORE's t=0 value of one trade, from ORE's
engines:
`DiscountingSwapEngine`, QuantLib's `BachelierSwaptionEngine` (ORE's European formula for a
swap starting after expiry, cash settled by `ParYieldCurve`), the OREApp oracle's calibrated
`NumericLgmMultiLegOptionEngine` (a cash-settled Bermudan/American is priced there as the
physical one: ORE approximates it by `CollateralizedCashPrice`, swaption.cpp), and
`DiscountingBondEngine`.

The curves are flat to their first non-zero pillar, as they were written before the oracle
handed ORE the as-of quote its curve rebuild needs (I-34, fixed 2026-10-07); a slope there
would now reach ORE unchanged too.
"""
import ORE

from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig, _build_ore_swap as option_underlying
from engine.instruments.european_swaption import SwaptionConfig, _build_ore_swap as european_underlying
from engine.instruments.swap import SwapConfig, _build_ore_swap as swap_underlying
from engine.instruments.treasury import BondConfig, CouponPeriod
from engine.market import CurrencyMarket, Market, SwaptionVolSurface, ZeroCurveConfig, index_name
from engine.models.ore_builders import TIME_AXIS_DAY_COUNTER as DC, ibor_index
from engine.valuation.config import LgmSwaptionEngineConfig
from tests.support.ore_lgm_oracle import OreCalibration, ore_lgm_swaption_npv

ASOF = ORE.Date(30, 7, 2026)
ASOF_ISO = "2026-07-30"
PILLARS = [0.0, 1.0, 2.0, 5.0, 10.0, 30.0]
DISCOUNT = [0.030, 0.030, 0.034, 0.040, 0.046, 0.050]
FORWARDING = [0.034, 0.034, 0.038, 0.044, 0.049, 0.052]
INDEX = index_name("USD", 6)
VOLS = SwaptionVolSurface(("1Y", "2Y", "5Y", "10Y"), ("1Y", "5Y", "10Y"),
                          ((0.0080, 0.0088, 0.0090), (0.0085, 0.0091, 0.0093), (0.0090, 0.0093, 0.0094),
                           (0.0092, 0.0094, 0.0096)))
#: The Bermudan/American engine as ORE's example configuration (the market path's default).
ENGINE = LgmSwaptionEngineConfig()

#: The seasoned swap's one past fixing (its first coupon fixed on 2026-04-29).
SEASONED_FIXINGS = {"2026-04-29": 0.0331}


def market() -> Market:
    return Market(ASOF, {"USD": CurrencyMarket(ZeroCurveConfig(PILLARS, DISCOUNT),
                                               {INDEX: ZeroCurveConfig(PILLARS, FORWARDING)}, VOLS)})


def market_json() -> dict:
    curve = lambda rates: {"times": PILLARS, "rates": rates}  # noqa: E731
    return {"asof": ASOF_ISO, "currencies": {"USD": {
        "discount_curve": curve(DISCOUNT), "index_curves": {INDEX: curve(FORWARDING)},
        "swaption_vols": {"option_tenors": list(VOLS.option_tenors), "swap_tenors": list(VOLS.swap_tenors),
                          "vols": [list(row) for row in VOLS.vols]}}}}


def _bermudan_exercises():
    """Every fixed-coupon start of the 2027-02-03 x 6Y underlying but the first and last."""
    return ["2028-02-03", "2029-02-05", "2030-02-04", "2031-02-03", "2032-02-03"]


def trades_json() -> dict:
    """Name -> the trade as the HTTP request takes it, `trade_id` its name."""
    return {name: {"trade_id": name, **trade} for name, trade in _trades_json().items()}


def _trades_json() -> dict:
    underlying = {"notional": 1e6, "effective_date": "2027-02-03", "maturity_date": "2033-02-03"}
    return {
        "swap-payer": {"trade_type": "swap", "notional": 1e7, "fixed_rate": 0.042, "payer": True,
                       "swap_tenor": "5Y"},
        "swap-receiver-seasoned-icma": {
            "trade_type": "swap", "notional": 5e6, "fixed_rate": 0.039, "payer": False,
            "effective_date": "2026-05-01", "maturity_date": "2031-05-01", "accrual_day_count": "ACT/ACT (ICMA)",
            "fixings": SEASONED_FIXINGS},
        "european-payer": {"trade_type": "european_swaption", "notional": 2e6, "fixed_rate": 0.044,
                           "payer": True, "swap_tenor": "5Y", "forward_start": "1Y"},
        "european-receiver-otm-cash": {"trade_type": "european_swaption", "notional": 2e6, "fixed_rate": 0.035,
                                       "payer": False, "swap_tenor": "4Y", "forward_start": "2Y",
                                       "floating_spread": 0.0025, "settlement": "Cash"},
        "bermudan-payer-physical": {"trade_type": "bermudan_swaption", "fixed_rate": 0.045, "payer": True,
                                    "exercise_dates": _bermudan_exercises(), **underlying},
        "bermudan-receiver-cash": {"trade_type": "bermudan_swaption", "fixed_rate": 0.040, "payer": False,
                                   "exercise_dates": _bermudan_exercises(), "settlement": "Cash", **underlying},
        "american-payer": {"trade_type": "american_swaption", "fixed_rate": 0.045, "payer": True,
                           "first_exercise_date": "2027-02-03", "last_exercise_date": "2031-02-03", **underlying},
        "bond": {"trade_type": "bond", "face_amount": 1e6, "maturity_date": "2029-08-15", "coupon_rate": 0.0375,
                 "coupon_schedule": [{"start_date": s, "end_date": e} for s, e in _BOND_PERIODS]},
    }


_BOND_PERIODS = [(f"{y}-{a}-15", f"{y + (b == '02')}-{b}-15") for y in (2026, 2027, 2028, 2029)
                 for a, b in (("02", "08"), ("08", "02"))][:7]


def _date(iso: str) -> ORE.Date:
    return ORE.DateParser.parseISO(iso)


def trades() -> dict:
    """Name -> the trade config, `trade_id` its name (the same trades as `trades_json`)."""
    underlying = dict(notional=1e6, effective_date=ORE.Date(3, 2, 2027), maturity_date=ORE.Date(3, 2, 2033),
                      evaluation_date=ASOF)
    exercises = [_date(d) for d in _bermudan_exercises()]
    return {
        "swap-payer": SwapConfig(notional=1e7, fixed_rate=0.042, payer=True, swap_tenor="5Y", evaluation_date=ASOF,
                                 trade_id="swap-payer"),
        "swap-receiver-seasoned-icma": SwapConfig(
            notional=5e6, fixed_rate=0.039, payer=False, effective_date=ORE.Date(1, 5, 2026),
            maturity_date=ORE.Date(1, 5, 2031), accrual_day_count="ACT/ACT (ICMA)", evaluation_date=ASOF,
            fixings={_date(d): r for d, r in SEASONED_FIXINGS.items()}, trade_id="swap-receiver-seasoned-icma"),
        "european-payer": SwaptionConfig(notional=2e6, fixed_rate=0.044, payer=True, swap_tenor="5Y",
                                         forward_start=ORE.Period(1, ORE.Years), evaluation_date=ASOF,
                                         trade_id="european-payer"),
        "european-receiver-otm-cash": SwaptionConfig(
            notional=2e6, fixed_rate=0.035, payer=False, swap_tenor="4Y", forward_start=ORE.Period(2, ORE.Years),
            floating_spread=0.0025, settlement="Cash", evaluation_date=ASOF, trade_id="european-receiver-otm-cash"),
        "bermudan-payer-physical": BermudanSwaptionConfig(fixed_rate=0.045, payer=True, exercise_dates=exercises,
                                                          trade_id="bermudan-payer-physical", **underlying),
        "bermudan-receiver-cash": BermudanSwaptionConfig(fixed_rate=0.040, payer=False, exercise_dates=exercises,
                                                         settlement="Cash", trade_id="bermudan-receiver-cash",
                                                         **underlying),
        "american-payer": AmericanSwaptionConfig(fixed_rate=0.045, payer=True, first_exercise_date=ORE.Date(3, 2, 2027),
                                                 last_exercise_date=ORE.Date(3, 2, 2031), trade_id="american-payer",
                                                 **underlying),
        "bond": BondConfig(face_amount=1e6, maturity_date=ORE.Date(15, 8, 2029), evaluation_date=ASOF,
                           coupon_rate=0.0375, trade_id="bond",
                           coupon_schedule=tuple(CouponPeriod(_date(s), _date(e)) for s, e in _BOND_PERIODS)),
    }


# ---------------------------------------------------------------------------
# ORE's t=0 values
# ---------------------------------------------------------------------------
def ore_curve(rates) -> "ORE.YieldTermStructureHandle":
    """The engine's zero curve in ORE: linear in the ACT/365 zero rate on whole-day pillars."""
    handle = ORE.YieldTermStructureHandle(ORE.ZeroCurve([ASOF + round(t * 365) for t in PILLARS], rates, DC))
    handle.enableExtrapolation()
    return handle


def ore_vols() -> "ORE.SwaptionVolatilityStructureHandle":
    matrix = ORE.Matrix(len(VOLS.option_tenors), len(VOLS.swap_tenors))
    for i, row in enumerate(VOLS.vols):
        for j, v in enumerate(row):
            matrix[i][j] = v
    surface = ORE.SwaptionVolatilityMatrix(
        ORE.TARGET(), ORE.Following, [ORE.Period(t) for t in VOLS.option_tenors],
        [ORE.Period(t) for t in VOLS.swap_tenors], matrix, DC, True, ORE.Normal)
    surface.enableExtrapolation()
    return ORE.SwaptionVolatilityStructureHandle(surface)


def ore_npv(cfg) -> float:
    """ORE's t=0 value of one trade of `trades()` (see the module docstring)."""
    ORE.Settings.instance().evaluationDate = ASOF
    if isinstance(cfg, SwapConfig):
        return _ore_swap(cfg)
    if isinstance(cfg, SwaptionConfig):
        return _ore_european(cfg)
    if isinstance(cfg, BondConfig):
        return _ore_bond(cfg)
    return _ore_lgm_option(cfg)


def _ore_swap(cfg: SwapConfig) -> float:
    index = ibor_index(cfg.index_tenor_months, ore_curve(FORWARDING))
    index.clearFixings()
    try:
        for date, rate in cfg.fixings.items():
            index.addFixing(date, rate, True)
        booked = swap_underlying(cfg)
        day_count = ORE.as_fixed_rate_coupon(booked.fixedLeg()[0]).dayCounter()
        swap = ORE.VanillaSwap(ORE.VanillaSwap.Payer if cfg.payer else ORE.VanillaSwap.Receiver, cfg.notional,
                               booked.fixedSchedule(), cfg.fixed_rate, day_count, booked.floatingSchedule(), index,
                               cfg.floating_spread, day_count)
        swap.setPricingEngine(ORE.DiscountingSwapEngine(ore_curve(DISCOUNT)))
        return swap.NPV()
    finally:
        index.clearFixings()


def _ore_european(cfg: SwaptionConfig) -> float:
    booked = european_underlying(cfg)
    swap = ORE.VanillaSwap(ORE.VanillaSwap.Payer if cfg.payer else ORE.VanillaSwap.Receiver, cfg.notional,
                           booked.fixedSchedule(), cfg.fixed_rate, DC, booked.floatingSchedule(),
                           ibor_index(cfg.index_tenor_months, ore_curve(FORWARDING)), cfg.floating_spread, DC)
    settlement = ((ORE.Settlement.Cash, ORE.Settlement.ParYieldCurve) if cfg.settlement == "Cash"
                  else (ORE.Settlement.Physical, ORE.Settlement.PhysicalOTC))
    swaption = ORE.Swaption(swap, ORE.EuropeanExercise(cfg.exercise_date), *settlement)
    swaption.setPricingEngine(ORE.BachelierSwaptionEngine(ore_curve(DISCOUNT), ore_vols()))
    return swaption.NPV()


def _ore_lgm_option(cfg) -> float:
    american = isinstance(cfg, AmericanSwaptionConfig)
    return ore_lgm_swaption_npv(
        evaluation_date=ASOF, curve_times=PILLARS, curve_rates=DISCOUNT, swap=option_underlying(cfg),
        notional=cfg.notional, fixed_rate=cfg.fixed_rate, payer=cfg.payer, floating_spread=cfg.floating_spread,
        index_tenor_months=cfg.index_tenor_months, style="American" if american else "Bermudan",
        exercise_dates=[cfg.first_exercise_date, cfg.last_exercise_date] if american else cfg.exercise_dates,
        hw_a=ENGINE.reversion, hw_sigma=ENGINE.volatility, n_per_std=ENGINE.n_per_std, std_devs=ENGINE.std_devs,
        exercise_time_steps_per_year=ENGINE.exercise_time_steps_per_year, index_curve_rates=FORWARDING,
        swaption_vols=VOLS, calibration=OreCalibration("Bootstrap", ENGINE.strategy, 1e-8)).npv


def _ore_bond(cfg: BondConfig) -> float:
    if not cfg.coupon_schedule:  # a bill: its redemption, discounted
        return cfg.face_amount * cfg.redemption_fraction * ore_curve(DISCOUNT).discount(cfg.maturity_date)
    dates = [p.start_date for p in cfg.coupon_schedule] + [cfg.coupon_schedule[-1].end_date]
    schedule = ORE.Schedule(dates, ORE.NullCalendar(), ORE.Unadjusted)
    bond = ORE.FixedRateBond(0, cfg.face_amount, schedule, [cfg.coupon_rate], ORE.ActualActual(ORE.ActualActual.ISMA))
    bond.setPricingEngine(ORE.DiscountingBondEngine(ore_curve(DISCOUNT)))
    return bond.NPV()
