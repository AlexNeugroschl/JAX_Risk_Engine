"""
Seeded random cases for the ORE parity sweeps (tests/test_ore_parity_sweep.py; I-116): a market,
a portfolio of every product, the pricing and sensitivity settings, a Theta roll and market-risk
scenarios, each drawn from one seed, so a failure is reproduced by its seed alone:

    from tests.support.parity_sweep import case
    print(case(17).describe())          # what seed 17 is
    pytest tests/test_ore_parity_sweep.py -k "seed-17"

**What a case draws** (`case(seed)`):

  * **The market** (`SHAPES`): a discount curve sloped, inverted, negative, flat, steep in its
    first segment or humped, on four to nine random pillars between t=0 and 30Y (whole ACT/365
    days, as ORE's curves take them; the last pillar beyond every flow); one Ibor index (3M or
    6M) on a basis over it; an ATM normal swaption matrix on one of two grids, its total variance
    increasing in the expiry, so every co-terminal basket can be fitted.
  * **The as-of date**: a TARGET business day of 2026-2030, a Friday for a Theta across the
    weekend (on a holiday ORE's simulation market reads its swaption volatilities from the next
    business day, inconsistently with its own sensitivity scenarios; I-117).
  * **The portfolio**: one trade of each product and, in the full sweep, up to three more of
    random products. Strikes up to 300bp in or out of the money; tenors and forward starts
    from a few days (a European near its expiry) to years; seasoned swaps, Bermudans,
    Americans and bonds, each past fixing given (the as-of date's sometimes); payer and
    receiver, long and short, physical and cash settlement, spreads, every accrual day count,
    Bermudan exercises on and inside accrual periods (some already past), Americans whose window
    opened before the as-of date, bonds annual or semi-annual and bills.
  * **The pricing configuration** (`PricingConfig`): each option engine calibrated or not, each
    strategy, reversion (zero, positive, negative), volatility, `ShiftHorizon`, grid, an
    American's exercise grid, reference calibration grid, calibration tolerance, root solver and
    swap-index conventions (one swap index for both engines, on the case's index, as ORE builds
    every helper on one). The European engine is Bachelier, ORE's (no ORE analytic prices with
    Jamshidian's).
  * **The sensitivity configuration** (`SensitivityConfig`): four to thirteen of ORE's tenors,
    the curve and volatility shift sizes (a negative curve shift too; volatility shifts of at
    most 2bp, I-117), and `theta_days` by the
    roll (`ROLLS`): one day, across a weekend, or onto the portfolio's next coupon payment,
    fixing or exercise, or a long roll of three to twelve months, so a coupon fixes and pays
    inside it.
  * **The market-risk scenarios** (`MarketRiskCase`): Monte Carlo (Sobol, a random horizon
    covariance of every pillar) or historical (a random walk of every pillar over consecutive
    TARGET business days, windows of 1, 5 or 10 days), and the VaR/ES quantiles.

**The fixed sample** is seeds `0 .. FIXED_SAMPLE - 1`: for them each enumerated choice is taken in
turn rather than drawn (`_Draw.pick`), so the fast tier's few cases cover every choice of every
drawn setting, and their grids are small. The full sweep is seeds `0 .. FULL_SWEEP - 1`.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import ORE

from engine.calibration.ore_lgm import SwapIndexConventions
from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig, exercisable_dates, underlying_swap
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.schedules import SWAP_CALENDAR
from engine.instruments.swap import SwapConfig, underlying_swap as swap_underlying
from engine.instruments.treasury import BondConfig, CouponPeriod
from engine.market_data.day_counts import SUPPORTED_ACCRUAL_DAY_COUNTS
from engine.market_data.market import CurrencyMarket, Market, SwaptionVolSurface, ZeroCurveConfig, index_name
from engine.market_simulation.config import DEFAULT_CURVE_TENORS
from engine.pricing.config import CALIBRATION_STRATEGIES, LgmSwaptionEngineConfig, PricingConfig, reference_grid_dates
from engine.risk.greeks.bump import SensitivityConfig
from engine.risk.market import RateRiskFactors, ShockScenarios, historical_scenarios, monte_carlo_scenarios
from engine.solvers.roots import SOLVERS
from tests.support.ore_market_risk_oracle import business_days_before

SHAPES = ("sloped", "inverted", "negative", "flat", "steep first segment", "humped")
ROLLS = ("one day", "weekend", "coupon", "fixing", "exercise", "long")
#: The fast tier's cases (see the module docstring) and the full sweep's.
FIXED_SAMPLE = 6
FULL_SWEEP = 40
#: Pillars in days from the as-of date: t=0, then four to nine of these, then 30Y.
_PILLAR_DAYS = (91, 182, 365, 730, 1095, 1825, 2555, 3650, 5475, 7300)
_LAST_PILLAR_DAYS = 10950
_VOL_GRIDS = ((("1Y", "2Y", "5Y", "10Y"), ("1Y", "5Y", "10Y")),
              (("6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y"), ("1Y", "2Y", "5Y", "10Y", "20Y")))
_PRODUCTS = ("swap", "european_swaption", "bermudan_swaption", "american_swaption", "bond")
#: The longest Theta roll drawn (`_sensitivity`), in days.
_LONGEST_ROLL = 370


class _Draw:
    """The case's random draws: `pick` takes the choices in turn in the fixed sample (each draw
    site offset by its own counter, so the sample's few cases vary every choice together) and
    draws them otherwise."""

    def __init__(self, seed: int):
        self.seed, self.rng, self.fixed, self._site = seed, np.random.default_rng(seed), seed < FIXED_SAMPLE, 0
        self.roll = ""  # the case's Theta roll, which the trades are drawn to reach
        #: The index's fixings, one per date for every trade of the case, as ORE holds them.
        self.history: Dict[ORE.Date, float] = {}
        self.pricing = PricingConfig()  # the case's engines, which an American's window is drawn for

    def pick(self, choices: Sequence):
        self._site += 1
        if self.fixed:
            return choices[(self.seed + self._site) % len(choices)]
        return choices[int(self.rng.integers(len(choices)))]

    def uniform(self, low: float, high: float) -> float:
        return float(self.rng.uniform(low, high))

    def integer(self, low: int, high: int) -> int:
        """In [low, high]."""
        return int(self.rng.integers(low, high + 1))

    def chance(self, p: float) -> bool:
        return bool(self.rng.random() < p)


@dataclass(frozen=True)
class MarketRiskCase:
    """A case's market-risk scenarios: the engine's (`scenarios`) and ORE's history of the same
    moves (`history` on `dates`, windows of `horizon_days`), and the figures' quantiles."""
    scenarios: ShockScenarios
    history: np.ndarray
    dates: Tuple[ORE.Date, ...]
    horizon_days: int
    quantiles: Tuple[float, ...]


@dataclass(frozen=True)
class SweepCase:
    """One seed's case (see the module docstring)."""
    seed: int
    shape: str
    roll: str
    market: Market
    trades: Tuple
    pricing: PricingConfig
    sensitivity: SensitivityConfig
    market_risk: MarketRiskCase

    @property
    def id(self) -> str:
        return f"seed-{self.seed}"

    @property
    def fixed_sample(self) -> bool:
        return self.seed < FIXED_SAMPLE

    def describe(self) -> str:
        """What the case is, for a failure's message."""
        data = self.market.currency("USD")
        index = next(iter(data.index_curves))
        lines = [f"{self.id}: {self.shape} market on {_iso(self.market.asof)} ({_weekday(self.market.asof)}), "
                 f"pillars {[round(t * 365) for t in data.discount_curve.times]} days, index {index}",
                 f"  discount {np.round(data.discount_curve.rates, 5).tolist()}",
                 f"  index    {np.round(data.index_curves[index].rates, 5).tolist()}",
                 f"  Theta: {self.roll} roll of {self.sensitivity.theta_days} days; tenors "
                 f"{list(self.sensitivity.curve_tenors)}, shifts {self.sensitivity.curve_shift}, "
                 f"{self.sensitivity.vol_shift}",
                 f"  Bermudan engine {self.pricing.bermudan}", f"  American engine {self.pricing.american}",
                 f"  scenarios: {self.market_risk.scenarios.source}, {self.market_risk.scenarios.num_scenarios} of "
                 f"{self.market_risk.horizon_days} days, quantiles {self.market_risk.quantiles}"]
        lines += [f"  {_trade_text(cfg)}" for cfg in self.trades]
        return "\n".join(lines)


def case(seed: int) -> SweepCase:
    """The case of `seed` (see the module docstring)."""
    draw = _Draw(seed)
    shape, roll = draw.pick(SHAPES), draw.pick(ROLLS)
    draw.roll = roll
    asof = _asof(draw, roll)
    index_months = draw.pick((6, 3, 6))
    market = _market(draw, asof, shape, index_months)
    pricing = draw.pricing = _pricing(draw, index_months)
    products = list(_PRODUCTS) + ([] if draw.fixed else [draw.pick(_PRODUCTS) for _ in range(draw.integer(0, 3))])
    trades = tuple(_TRADES[product](draw, market, f"{product}-{i}") for i, product in enumerate(products))
    sensitivity = _sensitivity(draw, roll, asof, trades)
    return SweepCase(seed, shape, roll, market, trades, pricing, sensitivity, _market_risk(draw, market))


def fixed_sample() -> List[SweepCase]:
    return [case(seed) for seed in range(FIXED_SAMPLE)]


def full_sweep() -> List[SweepCase]:
    return [case(seed) for seed in range(FIXED_SAMPLE, FULL_SWEEP)]


# --- the market ---------------------------------------------------------------------------------

def _asof(draw: _Draw, roll: str) -> ORE.Date:
    """A TARGET business day (an end-of-day run's), a Friday for the weekend roll."""
    day = ORE.Date(1, 1, 2026) + draw.integer(0, 4 * 365)
    while not ORE.TARGET().isBusinessDay(day) or (roll == "weekend") != (day.weekday() == ORE.Friday):
        day = day + 1
    return day


def _zero_rates(draw: _Draw, shape: str, t: np.ndarray) -> np.ndarray:
    tau = draw.uniform(1.0, 6.0)
    if shape == "flat":
        return np.full_like(t, draw.uniform(-0.005, 0.05))
    if shape == "humped":
        return draw.uniform(0.01, 0.03) + draw.uniform(0.005, 0.02) * (t / tau) * np.exp(1.0 - t / tau)
    if shape == "negative":
        level, slope = draw.uniform(-0.008, -0.001), draw.uniform(-0.002, 0.008)
    elif shape == "inverted":
        level, slope = draw.uniform(0.03, 0.06), -draw.uniform(0.005, 0.025)
    else:
        level, slope = draw.uniform(0.005, 0.04), draw.uniform(0.005, 0.03)
    rates = level + slope * (1.0 - np.exp(-t / tau))
    if shape == "steep first segment":
        rates[0] -= draw.uniform(0.005, 0.015)
    return rates


def _market(draw: _Draw, asof: ORE.Date, shape: str, index_months: int) -> Market:
    inner = sorted(int(d) for d in draw.rng.choice(_PILLAR_DAYS, size=draw.integer(4, 9), replace=False))
    days = [0] + inner + [_LAST_PILLAR_DAYS]
    t = np.asarray(days, dtype=np.float64) / 365.0
    discount = _zero_rates(draw, shape, t)
    basis = draw.uniform(0.0, 0.004) + draw.uniform(-0.001, 0.003) * (1.0 - np.exp(-t / 3.0))
    expiries, terms = draw.pick(_VOL_GRIDS)
    level, decay, term_slope = draw.uniform(0.005, 0.011), draw.uniform(-0.15, 0.3), draw.uniform(-0.1, 0.1)
    vols = tuple(tuple(level * (1.0 + decay * np.exp(-_years(e) / 2.0)) * (1.0 + term_slope * np.log(_years(s)) / 3.0)
                       for s in terms) for e in expiries)
    curve = lambda rates: ZeroCurveConfig(list(map(float, t)), list(map(float, rates)))  # noqa: E731
    return Market(asof, {"USD": CurrencyMarket(curve(discount), {index_name("USD", index_months): curve(discount + basis)},
                                               SwaptionVolSurface(expiries, terms, vols))})


def _years(tenor: str) -> float:
    period = ORE.Period(tenor)
    return period.length() / (12.0 if period.units() == ORE.Months else 1.0)


def _index_rate(market: Market, years: float) -> float:
    """The index curve's zero rate at `years`, the level a strike is drawn around."""
    data = market.currency("USD")
    curve = next(iter(data.index_curves.values()))
    return float(np.interp(years, curve.times, curve.rates))


# --- the trades ---------------------------------------------------------------------------------

def _index_months(market: Market) -> int:
    return int(next(iter(market.currency("USD").index_curves)).rsplit("-", 1)[1][:-1])


def _strike(draw: _Draw, market: Market, years: float) -> float:
    return _index_rate(market, years) + draw.pick((0.0, 1.0, -1.0)) * draw.uniform(0.0, 0.03)


def _fixings(draw: _Draw, market: Market, swap: ORE.VanillaSwap) -> Dict[ORE.Date, float]:
    """A fixing for every floating coupon fixed before the as-of date, and sometimes the
    as-of date's own, each near the index's level and the case's one fixing of its date."""
    asof, out = market.asof, {}
    for cf in swap.floatingLeg():
        fixing = ORE.as_floating_rate_coupon(cf).fixingDate()
        if fixing not in draw.history and (fixing < asof or (fixing == asof and draw.chance(0.5))):
            draw.history[fixing] = _index_rate(market, 0.5) + draw.uniform(-0.005, 0.005)
        if fixing in draw.history and fixing <= asof:
            out[fixing] = draw.history[fixing]
    return out


def _start(draw: _Draw, asof: ORE.Date, seasoned: bool, years_back: float) -> ORE.Date:
    if seasoned:
        return SWAP_CALENDAR.adjust(asof - draw.integer(20, max(21, int(years_back * 365))))
    return SWAP_CALENDAR.advance(asof, draw.integer(2, 700), ORE.Days)


def _swap(draw: _Draw, market: Market, trade_id: str) -> SwapConfig:
    asof, tenor = market.asof, draw.pick((1, 2, 3, 5, 7, 10))
    seasoned = draw.roll in ("coupon", "fixing") or draw.pick((True, False))
    effective = _start(draw, asof, seasoned, tenor - 0.5)
    terms = dict(notional=10 ** draw.uniform(5.0, 7.5), fixed_rate=_strike(draw, market, tenor / 2.0),
                 payer=draw.pick((True, False)), effective_date=effective,
                 maturity_date=effective + ORE.Period(tenor, ORE.Years), index_tenor_months=_index_months(market),
                 floating_spread=draw.pick((0.0, draw.uniform(-0.005, 0.01))),
                 accrual_day_count=draw.pick(sorted(SUPPORTED_ACCRUAL_DAY_COUNTS)), evaluation_date=asof,
                 trade_id=trade_id)
    cfg = SwapConfig(**terms, fixings={})
    return dataclasses.replace(cfg, fixings=_fixings(draw, market, swap_underlying(cfg)))


def _option_terms(draw: _Draw, market: Market, tenor: int) -> dict:
    return dict(notional=10 ** draw.uniform(5.0, 7.0), fixed_rate=_strike(draw, market, tenor / 2.0 + 1.0),
                payer=draw.pick((True, False)), index_tenor_months=_index_months(market),
                floating_spread=draw.pick((0.0, draw.uniform(-0.003, 0.006))),
                settlement=draw.pick(("Physical", "Cash")), long_short=draw.pick(("Long", "Short")),
                evaluation_date=market.asof)


def _european(draw: _Draw, market: Market, trade_id: str) -> SwaptionConfig:
    asof, tenor = market.asof, draw.pick((1, 2, 5, 10))
    days = draw.integer(1, 20) if draw.roll == "exercise" else         draw.pick((draw.integer(1, 20), draw.integer(30, 400), draw.integer(400, 1800)))
    exercise = SWAP_CALENDAR.adjust(asof + days)
    effective = SWAP_CALENDAR.advance(exercise, 2, ORE.Days)
    return SwaptionConfig(**_option_terms(draw, market, tenor), exercise_date=exercise, effective_date=effective,
                          maturity_date=effective + ORE.Period(tenor, ORE.Years), trade_id=trade_id)


def _seasonable_option(draw: _Draw, market: Market, cls, trade_id: str):
    """A Bermudan or American on an underlying starting ahead or already running."""
    asof, tenor = market.asof, draw.pick((3, 5, 7, 10))
    seasoned = draw.pick((False, True))
    effective = _start(draw, asof, seasoned, tenor - 1.5)
    maturity = effective + ORE.Period(tenor, ORE.Years)
    terms = dict(_option_terms(draw, market, tenor), effective_date=effective, maturity_date=maturity, fixings={},
                 trade_id=trade_id)
    engine = draw.pricing.american if cls is AmericanSwaptionConfig else draw.pricing.bermudan
    if engine.strategy == "CoterminalDealStrike" and engine.swap_index.fixed_day_counter == "ACT/ACT (ICMA)":
        # Near the money: ORE's sensitivity analytic is inconsistent with itself for a deal strike
        # its fallback moves to the ATM band's edge on an ACT/ACT (ICMA) swap index (I-117).
        terms["fixed_rate"] = _index_rate(market, tenor / 2.0 + 1.0) + draw.uniform(-0.005, 0.005)
    probe = BermudanSwaptionConfig(**terms, exercise_dates=[maturity - 30])
    fixings = _fixings(draw, market, underlying_swap(probe))
    starts = [d for d in exercisable_dates(probe)][1:]
    if cls is AmericanSwaptionConfig:
        first = (asof - draw.integer(5, 200)) if seasoned and draw.chance(0.5) else starts[0]
        last = starts[-1] if draw.chance(0.5) else starts[max(1, len(starts) // 2)]
        # A calibrated American needs a reference grid date in its window, else ORE refuses it
        # (and the engine, tests/test_ore_parity_sweep.py): the window is drawn to hold one from
        # the as-of date and from any Theta date (`_LONGEST_ROLL`), where ORE values a trade it
        # cannot build at 0 and the engine refuses the run (I-117).
        grid = [d for d in reference_grid_dates(asof + _LONGEST_ROLL, draw.pricing.american.reference_calibration_grid)
                if d >= max(asof + 1, first)]
        return AmericanSwaptionConfig(**dict(terms, fixings=fixings), first_exercise_date=first,
                                      last_exercise_date=max(last, grid[0] + 1))
    live = [d for d in starts if d > asof]
    chosen = sorted({d for d in live if draw.chance(0.6)} | {live[-1]})
    if draw.chance(0.4):  # some exercises inside an accrual period: they enter the next whole period
        chosen = sorted({d + draw.integer(10, 100) if draw.chance(0.5) else d for d in chosen})
        chosen = [d for d in chosen if d < maturity - 40] or [live[-1]]
    past = [d for d in starts if d <= asof]
    if past and draw.chance(0.5):  # an exercise already past: not an opportunity any more
        chosen = [past[-1]] + chosen
    return BermudanSwaptionConfig(**dict(terms, fixings=fixings), exercise_dates=chosen)


def _bermudan(draw: _Draw, market: Market, trade_id: str) -> BermudanSwaptionConfig:
    return _seasonable_option(draw, market, BermudanSwaptionConfig, trade_id)


def _american(draw: _Draw, market: Market, trade_id: str) -> AmericanSwaptionConfig:
    return _seasonable_option(draw, market, AmericanSwaptionConfig, trade_id)


def _bond(draw: _Draw, market: Market, trade_id: str) -> BondConfig:
    asof = market.asof
    face = 10 ** draw.uniform(5.0, 7.0)
    if draw.pick((False, False, True)):  # a bill
        return BondConfig(face_amount=face, maturity_date=asof + draw.integer(7, 360), evaluation_date=asof,
                          trade_id=trade_id)
    months = draw.pick((12, 6))
    start = asof - draw.integer(0, 400)
    count = draw.integer(1, 20 if months == 6 else 10)
    dates = [start + ORE.Period(months * k, ORE.Months) for k in range(count + 1)]
    while dates[-1] <= asof:
        dates.append(dates[-1] + ORE.Period(months, ORE.Months))
    periods = tuple(CouponPeriod(a, b) for a, b in zip(dates, dates[1:]))
    return BondConfig(face_amount=face, maturity_date=dates[-1], coupon_rate=draw.uniform(0.0, 0.07),
                      coupon_schedule=periods, accrual_day_count=draw.pick(sorted(SUPPORTED_ACCRUAL_DAY_COUNTS)),
                      evaluation_date=asof, trade_id=trade_id)


_TRADES = {"swap": _swap, "european_swaption": _european, "bermudan_swaption": _bermudan,
           "american_swaption": _american, "bond": _bond}


# --- the settings -------------------------------------------------------------------------------

def _engine(draw: _Draw, swap_index: SwapIndexConventions) -> LgmSwaptionEngineConfig:
    return LgmSwaptionEngineConfig(
        reversion=draw.pick((0.0, 0.01, 0.03, 0.05, -0.005)), volatility=draw.uniform(0.004, 0.012),
        calibration=draw.pick(("Bootstrap", "None", "Bootstrap")), strategy=draw.pick(CALIBRATION_STRATEGIES),
        bootstrap_tolerance=draw.pick((1e-4, 1e-3)), reference_calibration_grid=draw.pick(("400,3M", "400,1M", "100,6M")),
        shift_horizon=draw.pick((0.0, 0.25, 0.5, 1.0, 1.5)),
        # At most 24 points (ORE's default 30 is the shared portfolio's, compared there) and 36 steps
        # a year: a 32-helper American recalibrated under 128 scenarios took ORE 12 minutes at 30 and 52.
        n_per_std=draw.pick((8, 12) if draw.fixed else (8, 12, 16, 24)), std_devs=draw.pick((3.0, 4.0, 5.0, 6.0)),
        exercise_time_steps_per_year=draw.pick((12, 24, 36)), swap_index=swap_index, solver=draw.pick(SOLVERS))


def _pricing(draw: _Draw, index_months: int) -> PricingConfig:
    swap_index = SwapIndexConventions(fixed_tenor=draw.pick(("1Y", "6M")),
                                      fixed_day_counter=draw.pick(sorted(SUPPORTED_ACCRUAL_DAY_COUNTS)),
                                      index_tenor_months=index_months)
    return PricingConfig(bermudan=_engine(draw, swap_index), american=_engine(draw, swap_index))


def _events(trades, asof: ORE.Date) -> Dict[str, List[ORE.Date]]:
    """Every coupon payment, fixing and exercise of `trades` after `asof`, by kind."""
    out = {"coupon": [], "fixing": [], "exercise": []}
    for cfg in trades:
        if isinstance(cfg, BondConfig):
            out["coupon"] += [p.end_date for p in cfg.coupon_schedule]
            continue
        if isinstance(cfg, (SwapConfig,)):
            swap = swap_underlying(cfg)
            out["coupon"] += [cf.date() for leg in (swap.fixedLeg(), swap.floatingLeg()) for cf in leg]
            out["fixing"] += [ORE.as_floating_rate_coupon(cf).fixingDate() for cf in swap.floatingLeg()]
        if isinstance(cfg, BermudanSwaptionConfig):
            out["exercise"] += list(cfg.exercise_dates)
        if isinstance(cfg, AmericanSwaptionConfig):
            out["exercise"] += [cfg.first_exercise_date]
        if isinstance(cfg, SwaptionConfig):
            out["exercise"] += [cfg.exercise_date]
    return {kind: sorted(d for d in dates if d > asof) for kind, dates in out.items()}


def _sensitivity(draw: _Draw, roll: str, asof: ORE.Date, trades) -> SensitivityConfig:
    count = draw.integer(4, len(DEFAULT_CURVE_TENORS))
    tenors = tuple(t for t in DEFAULT_CURVE_TENORS
                   if t in set(draw.rng.choice(DEFAULT_CURVE_TENORS, size=count, replace=False)))
    days = 1
    if roll == "weekend":
        days = draw.pick((1, 2, 3))
    elif roll == "long":
        days = draw.pick((92, 190, 370))
    elif roll in ("coupon", "fixing", "exercise"):
        upcoming = _events(trades, asof)[roll]
        days = (upcoming[0] - asof) + draw.pick((0, 0, 1)) if upcoming else 1
    return SensitivityConfig(curve_tenors=tenors, curve_shift=draw.pick((1e-4, 5e-5, 2.5e-4, -1e-4)),
                             # A 5bp volatility shift can leave ORE's optimiser short of a large basket's fit,
                             # which the engine's roots reach (I-117): ORE then fails the whole analytic.
                             vol_shift=draw.pick((1e-4, 2e-4, 1e-5)), theta_days=int(days))


def _market_risk(draw: _Draw, market: Market) -> MarketRiskCase:
    factors = RateRiskFactors.from_market(market)
    size = factors.size
    pillar = np.concatenate([np.arange(n) for n in factors.sizes])
    curve = np.repeat(np.arange(len(factors.sizes)), factors.sizes)
    vols = draw.uniform(4e-4, 1.2e-3) * (1.0 + draw.uniform(-0.3, 0.3) * np.exp(-pillar / 3.0))
    corr = np.exp(-np.abs(pillar[:, None] - pillar[None, :]) / draw.uniform(2.0, 6.0))
    corr = corr * np.where(curve[:, None] == curve[None, :], 1.0, draw.uniform(0.8, 0.98))
    np.fill_diagonal(corr, 1.0)
    daily = corr * vols[:, None] * vols[None, :]
    # Powers of two (balanced Sobol); at most 128, where a 32-helper American takes ORE 9 minutes.
    count = draw.pick((16, 32) if draw.fixed else (32, 64, 128))
    quantiles = draw.pick(((0.99, 0.975), (0.95,), (0.99, 0.975, 0.9)))
    base = factors.base_rates()
    if draw.pick(("monte-carlo", "historical")) == "monte-carlo":
        scenarios = monte_carlo_scenarios(factors, daily * 10.0, horizon_days=10, num_scenarios=count,
                                          seed=draw.integer(0, 10 ** 6))
        # ORE reads scenarios as a history: each move is one day's, from the level before it.
        history = base[None, :] + np.concatenate([np.zeros((1, size)), np.cumsum(scenarios.shifts, axis=0)])
        return MarketRiskCase(scenarios, history, tuple(business_days_before(market.asof, count + 1)), 1, quantiles)
    horizon = draw.pick((1, 5, 10))
    steps = draw.rng.multivariate_normal(np.zeros(size), daily, size=count + horizon)
    history = base[None, :] + np.cumsum(steps, axis=0) - np.sum(steps, axis=0)[None, :]
    return MarketRiskCase(historical_scenarios(factors, history, horizon), history,
                          tuple(business_days_before(market.asof, len(history))), horizon, quantiles)


# --- descriptions -------------------------------------------------------------------------------

def _iso(d: ORE.Date) -> str:
    return f"{d.year():04d}-{d.month():02d}-{d.dayOfMonth():02d}"


def _weekday(d: ORE.Date) -> str:
    return ("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")[d.weekday() - 1]


def _trade_text(cfg) -> str:
    fields = {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)
              if f.name not in ("evaluation_date", "trade_id", "coupon_schedule")}
    shown = {k: (_iso(v) if isinstance(v, ORE.Date) else [_iso(d) for d in v] if k == "exercise_dates"
                 else {_iso(d): round(r, 6) for d, r in v.items()} if k == "fixings" else v)
             for k, v in fields.items()}
    if isinstance(cfg, BondConfig):
        shown["coupon_schedule"] = [(_iso(p.start_date), _iso(p.end_date)) for p in cfg.coupon_schedule]
    return f"{cfg.trade_id}: {shown}"
