"""
ORE's inputs built in memory, and an `OREApp` run over them in this process: the conventions,
curves, today's market configuration and pricing engines as XML strings, the trades as ORE's
trade XML, and the market data and fixings as ORE's text lines. Shared by the two oracles:
tests/support/ore_lgm_oracle.py (one swaption's NPV) and tests/support/ore_xva_oracle.py (an
exposure simulation of a portfolio).

Test tooling: the engine never imports it. One currency (`CCY`) with its discount curve, one
Ibor index's forwarding curve and an ATM normal swaption matrix: the market of the parity tests
and of the shared portfolio (tests/support/portfolio.py).

What is built:
  * Trades: the engine's own trades, each leg's schedule passed as explicit `<Dates>` from the
    engine's booked `ORE` instrument (`trade_xml`), so ORE prices exactly the engine's
    coupons. The index is a convention-defined `USD-SIMINDEX-<N>M` with `build_vanilla_swap`'s
    `SimIndex` terms (2 settlement days, TARGET, MF, no EOM, ACT/365).
  * Curves: date-quoted continuous ACT/365 zero rates, linear in the zero rate, as the engine's
    `ZeroCurve`; pillars must be whole ACT/365 days from the as-of date (checked) and start at
    t=0. ORE's zero-curve build re-reads the t=0 rate as the rate at `t = 1e-4`
    (`YieldCurve::flattenPiecewiseCurve` reads `zeroRate(asof)`, which QuantLib's
    `YieldTermStructure::zeroRate` takes at `dt = 1e-4`), so a curve sloped in its first segment
    would be tilted there. The as-of quote handed to ORE is the one that rebuild maps onto the
    engine's t=0 rate (`_as_of_quote`, I-34): every curve, flat or sloped, is the engine's.
    Alternatively a curve is given as discount factors on dates, log-linear between them
    (`OreDiscountCurves`): how a simulated path's curves reach ORE.

Two inputs ORE requires that do not affect a price:
  * A swap index pair (`USD-CMS-1Y`/`USD-CMS-30Y`). `IrModelBuilder` takes the LGM term
    structure from the swap index's discounting curve and would otherwise fall back to a flat
    1%; it is mapped to the discount curve through `DISCOUNT_INDEX`.
  * An ATM swaption vol quote, read only when calibrating (`PLACEHOLDER_VOLS`).

Log file: `OREApp` needs a real log path and keeps the file open, so one scratch directory per
process is reused (`_scratch_dir`).

Global state: `OREApp` sets ORE's evaluation date (restored by `run_app`) and its process-wide
`InstrumentConventions` (no SWIG accessor to restore). Another reason not to run it inside a
pricing process.
"""
from __future__ import annotations

import functools
import tempfile
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import ORE

from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig, underlying_swap as option_underlying
from engine.instruments.european_swaption import SwaptionConfig, underlying_swap as european_underlying
from engine.instruments.swap import SwapConfig, underlying_swap as swap_underlying
from engine.instruments.treasury import BondConfig
from engine.market_data.day_counts import resolve_accrual_day_count
from engine.market_data.market import SwaptionVolSurface
from engine.models.lgm import Sigma

CCY = "USD"
CURVE_ID = "SIMCURVE"
INDEX_CURVE_ID = "SIMINDEXCURVE"
#: An index whose forwarding curve is the discount curve: ORE's swap index curves name an
#: index for discounting, the LGM builder takes the model's term structure from it, and a
#: Treasury discounts on it (its `ReferenceCurveId`).
DISCOUNT_INDEX = f"{CCY}-SIMDISC-6M"
#: The vol quote used when a caller gives no surface (read only when calibrating).
PLACEHOLDER_VOLS = SwaptionVolSurface(option_tenors=("1Y",), swap_tenors=("1Y",), vols=((0.01,),))
#: QuantLib's `dt` in `YieldTermStructure::zeroRate` at t=0 (ql/termstructures/yieldtermstructure.cpp).
ZERO_RATE_DT = 1e-4
NETTING_SET = "NS"


@dataclass(frozen=True)
class OreDiscountCurves:
    """Discount and index curves as discount factors on dates after the as-of date,
    log-linear between them: how a simulated path's scenario curves reach ORE."""
    discount_dates: Sequence[ORE.Date]
    discount_factors: Sequence[float]
    index_dates: Sequence[ORE.Date]
    index_factors: Sequence[float]


@dataclass(frozen=True)
class OreCurves:
    """The curves of one ORE run: zero rates on `times` (the discount curve's, and the index's
    on the same pillars, the discount curve's if `index_rates` is None), or discount factors
    (`discount_factors`, which then replaces both)."""
    times: Sequence[float] = ()
    rates: Sequence[float] = ()
    index_rates: Optional[Sequence[float]] = None
    discount_factors: Optional[OreDiscountCurves] = None

    @property
    def separate_index_curve(self) -> bool:
        return self.index_rates is not None or self.discount_factors is not None


@functools.lru_cache(maxsize=1)
def _scratch_dir() -> str:
    return tempfile.mkdtemp(prefix="ore_oracle_")


def iso(d: ORE.Date) -> str:
    return f"{d.year():04d}-{d.month():02d}-{d.dayOfMonth():02d}"


def index_name(index_tenor_months: int) -> str:
    return f"{CCY}-SIMINDEX-{index_tenor_months}M"


def curve_dates(asof: ORE.Date, times: Sequence[float]) -> list:
    dates = []
    for t in times:
        days = round(t * 365)
        if abs(days / 365 - t) > 1e-12:
            raise ValueError(f"curve pillar {t} is not a whole number of ACT/365 days from the as-of date")
        dates.append(asof + days)
    return dates


def _as_of_quote(times: Sequence[float], rates: Sequence[float]) -> float:
    """The t=0 quote whose rebuilt rate at `ZERO_RATE_DT` (linear in the zero rate up to the
    first pillar after t=0) is the engine's t=0 rate (I-34)."""
    if times[0] != 0.0 or rates[0] == rates[1]:  # nothing to tilt: the quote as given, bit for bit
        return float(rates[0])
    t1, eps = float(times[1]), ZERO_RATE_DT
    return (float(rates[0]) - float(rates[1]) * eps / t1) / (1.0 - eps / t1)


def _number(value) -> str:
    """A float as ORE parses it: full precision, never `np.float64(...)`."""
    return repr(float(value))


# --- configuration ------------------------------------------------------------------------------

def conventions_xml(index: str) -> str:
    return f"""<Conventions>
  <Zero>
    <Id>SIM-ZERO</Id>
    <TenorBased>false</TenorBased>
    <DayCounter>A365</DayCounter>
    <Compounding>Continuous</Compounding>
  </Zero>
  <IborIndex>
    <Id>{index}</Id>
    <FixingCalendar>TARGET</FixingCalendar>
    <DayCounter>A365</DayCounter>
    <SettlementDays>2</SettlementDays>
    <BusinessDayConvention>MF</BusinessDayConvention>
    <EndOfMonth>false</EndOfMonth>
  </IborIndex>
  <IborIndex>
    <Id>{DISCOUNT_INDEX}</Id>
    <FixingCalendar>TARGET</FixingCalendar>
    <DayCounter>A365</DayCounter>
    <SettlementDays>2</SettlementDays>
    <BusinessDayConvention>MF</BusinessDayConvention>
    <EndOfMonth>false</EndOfMonth>
  </IborIndex>
  <Swap>
    <Id>SIM-SWAP-CONVENTIONS</Id>
    <FixedCalendar>TARGET</FixedCalendar>
    <FixedFrequency>Annual</FixedFrequency>
    <FixedConvention>MF</FixedConvention>
    <FixedDayCounter>A365</FixedDayCounter>
    <Index>{index}</Index>
  </Swap>
  <SwapIndex><Id>{CCY}-CMS-1Y</Id><Conventions>SIM-SWAP-CONVENTIONS</Conventions></SwapIndex>
  <SwapIndex><Id>{CCY}-CMS-30Y</Id><Conventions>SIM-SWAP-CONVENTIONS</Conventions></SwapIndex>
</Conventions>"""


def _yield_curve_xml(curve_id: str, asof: ORE.Date, times: Sequence[float]) -> str:
    quotes = "\n".join(
        f"          <Quote>ZERO/RATE/{CCY}/{curve_id}/A365/{iso(d)}</Quote>" for d in curve_dates(asof, times))
    return f"""    <YieldCurve>
      <CurveId>{curve_id}</CurveId>
      <CurveDescription>engine zero curve</CurveDescription>
      <Currency>{CCY}</Currency>
      <DiscountCurve/>
      <Segments>
        <Direct>
          <Type>Zero</Type>
          <Quotes>
{quotes}
          </Quotes>
          <Conventions>SIM-ZERO</Conventions>
        </Direct>
      </Segments>
      <InterpolationVariable>Zero</InterpolationVariable>
      <InterpolationMethod>Linear</InterpolationMethod>
      <YieldCurveDayCounter>A365</YieldCurveDayCounter>
      <Extrapolation>true</Extrapolation>
    </YieldCurve>"""


def _discount_curve_xml(curve_id: str, dates: Sequence[ORE.Date]) -> str:
    """A curve given by discount factors, log-linear between them (ORE keeps them as given
    when the interpolation variable is Discount): a simulated path's scenario curve."""
    quotes = "\n".join(f"          <Quote>DISCOUNT/RATE/{CCY}/{curve_id}/{iso(d)}</Quote>" for d in dates)
    return f"""    <YieldCurve>
      <CurveId>{curve_id}</CurveId>
      <CurveDescription>scenario curve</CurveDescription>
      <Currency>{CCY}</Currency>
      <DiscountCurve/>
      <Segments>
        <Direct>
          <Type>Discount</Type>
          <Quotes>
{quotes}
          </Quotes>
          <Conventions>SIM-ZERO</Conventions>
        </Direct>
      </Segments>
      <InterpolationVariable>Discount</InterpolationVariable>
      <InterpolationMethod>LogLinear</InterpolationMethod>
      <YieldCurveDayCounter>A365</YieldCurveDayCounter>
      <Extrapolation>true</Extrapolation>
    </YieldCurve>"""


def curveconfig_xml(asof: ORE.Date, curves: OreCurves, vols: SwaptionVolSurface) -> str:
    if curves.discount_factors is None:
        xml = _yield_curve_xml(CURVE_ID, asof, curves.times)
        if curves.separate_index_curve:
            xml += "\n" + _yield_curve_xml(INDEX_CURVE_ID, asof, curves.times)
    else:
        xml = (_discount_curve_xml(CURVE_ID, curves.discount_factors.discount_dates) + "\n"
               + _discount_curve_xml(INDEX_CURVE_ID, curves.discount_factors.index_dates))
    return f"""<CurveConfiguration>
  <YieldCurves>
{xml}
  </YieldCurves>
  <SwaptionVolatilities>
    <SwaptionVolatility>
      <CurveId>SIMVOL</CurveId>
      <CurveDescription>ATM normal swaption volatilities</CurveDescription>
      <Dimension>ATM</Dimension>
      <VolatilityType>Normal</VolatilityType>
      <Extrapolation>Flat</Extrapolation>
      <DayCounter>A365</DayCounter>
      <Calendar>TARGET</Calendar>
      <BusinessDayConvention>Following</BusinessDayConvention>
      <OptionTenors>{",".join(vols.option_tenors)}</OptionTenors>
      <SwapTenors>{",".join(vols.swap_tenors)}</SwapTenors>
      <ShortSwapIndexBase>{CCY}-CMS-1Y</ShortSwapIndexBase>
      <SwapIndexBase>{CCY}-CMS-30Y</SwapIndexBase>
    </SwaptionVolatility>
  </SwaptionVolatilities>
</CurveConfiguration>"""


def todaysmarket_xml(index: str, separate_index_curve: bool) -> str:
    index_curve = INDEX_CURVE_ID if separate_index_curve else CURVE_ID
    return f"""<TodaysMarket>
  <Configuration id="default">
    <DiscountingCurvesId>default</DiscountingCurvesId>
    <IndexForwardingCurvesId>default</IndexForwardingCurvesId>
    <SwapIndexCurvesId>default</SwapIndexCurvesId>
    <SwaptionVolatilitiesId>default</SwaptionVolatilitiesId>
  </Configuration>
  <DiscountingCurves id="default">
    <DiscountingCurve currency="{CCY}">Yield/{CCY}/{CURVE_ID}</DiscountingCurve>
  </DiscountingCurves>
  <IndexForwardingCurves id="default">
    <Index name="{index}">Yield/{CCY}/{index_curve}</Index>
    <Index name="{DISCOUNT_INDEX}">Yield/{CCY}/{CURVE_ID}</Index>
  </IndexForwardingCurves>
  <SwapIndexCurves id="default">
    <SwapIndex name="{CCY}-CMS-1Y"><Discounting>{DISCOUNT_INDEX}</Discounting></SwapIndex>
    <SwapIndex name="{CCY}-CMS-30Y"><Discounting>{DISCOUNT_INDEX}</Discounting></SwapIndex>
  </SwapIndexCurves>
  <SwaptionVolatilities id="default">
    <SwaptionVolatility currency="{CCY}">SwaptionVolatility/{CCY}/SIMVOL</SwaptionVolatility>
  </SwaptionVolatilities>
</TodaysMarket>"""


# --- pricing engines ----------------------------------------------------------------------------

@dataclass(frozen=True)
class OreCalibration:
    """ORE's LGM calibration settings (`LGMGridSwaptionEngineBuilder` model parameters).
    `method="None"` prices with the given volatility; `"Bootstrap"` calibrates a piecewise
    volatility to the trade's co-terminal basket (`strategy` `CoterminalATM` or
    `CoterminalDealStrike`), starting from it. `tolerance` is the error above which ORE's
    builder reports the calibration as failed (it does not change the calibration)."""
    method: str = "None"
    strategy: str = "None"
    tolerance: float = 1e-4


@dataclass(frozen=True)
class OreFdSolver:
    """ORE's FD solver (`LGMFDSwaptionEngineBuilder` -> `LgmFdSolver`) instead of the Grid
    solver the engine reproduces. Defaults are ORE's example American settings
    (Examples/Products/Input/pricingengine.xml)."""
    scheme: str = "Douglas"
    state_grid_points: int = 64
    time_steps_per_year: int = 24
    mesher_epsilon: float = 1e-4


@dataclass(frozen=True)
class OreLgmEngine:
    """ORE's LGM swaption engine and its model: the reversion and volatility (constant or a
    piecewise `Sigma`, mapped to `VolatilityTimes`/`Volatility` bucket for bucket), Hagan
    volatility with HullWhite (constant) reversion as `engine.models.lgm`, the calibration, the
    Grid solver's `nx`/`sx` (or ORE's FD solver), an American's exercise grid,
    `ShiftHorizon` and `ReferenceCalibrationGrid`."""
    reversion: float
    volatility: Union[float, Sigma]
    n_per_std: int
    std_devs: float
    exercise_time_steps_per_year: int = 24
    shift_horizon: float = 0.0
    reference_calibration_grid: str = "400,3M"
    calibration: OreCalibration = OreCalibration()
    fd_solver: Optional[OreFdSolver] = None

    @classmethod
    def of(cls, config, tolerance: float = 1e-4) -> "OreLgmEngine":
        """The engine of an `engine.pricing.config.LgmSwaptionEngineConfig`."""
        strategy = config.strategy if config.calibration == "Bootstrap" else "None"
        return cls(config.reversion, config.volatility, config.n_per_std, config.std_devs,
                   config.exercise_time_steps_per_year, config.shift_horizon, config.reference_calibration_grid,
                   OreCalibration(config.calibration, strategy, tolerance))


def _volatility_parameters(volatility) -> str:
    if isinstance(volatility, Sigma):
        times = ",".join(_number(t) for t in np.asarray(volatility.times))
        values = ",".join(_number(v) for v in np.asarray(volatility.values))
    else:
        times, values = "", _number(volatility)
    return (f'<Parameter name="Volatility">{values}</Parameter>\n'
            f'      <Parameter name="VolatilityTimes">{times}</Parameter>')


def _solver_xml(engine: OreLgmEngine) -> str:
    if engine.fd_solver is None:
        return f"""<Engine>Grid</Engine>
    <EngineParameters>
      <Parameter name="sy">{engine.std_devs!r}</Parameter>
      <Parameter name="ny">{engine.n_per_std}</Parameter>
      <Parameter name="sx">{engine.std_devs!r}</Parameter>
      <Parameter name="nx">{engine.n_per_std}</Parameter>
    </EngineParameters>"""
    fd = engine.fd_solver
    return f"""<Engine>FD</Engine>
    <EngineParameters>
      <Parameter name="Scheme">{fd.scheme}</Parameter>
      <Parameter name="StateGridPoints">{fd.state_grid_points}</Parameter>
      <Parameter name="TimeStepsPerYear">{fd.time_steps_per_year}</Parameter>
      <Parameter name="MesherEpsilon">{fd.mesher_epsilon!r}</Parameter>
    </EngineParameters>"""


def _lgm_product_xml(engine: OreLgmEngine) -> str:
    calibration = engine.calibration
    return f"""
    <Model>LGM</Model>
    <ModelParameters>
      <Parameter name="Calibration">{calibration.method}</Parameter>
      <Parameter name="CalibrationStrategy">{calibration.strategy}</Parameter>
      <Parameter name="Reversion">{_number(engine.reversion)}</Parameter>
      <Parameter name="ReversionType">HullWhite</Parameter>
      {_volatility_parameters(engine.volatility)}
      <Parameter name="VolatilityType">Hagan</Parameter>
      <Parameter name="ShiftHorizon">{_number(engine.shift_horizon)}</Parameter>
      <Parameter name="Tolerance">{calibration.tolerance!r}</Parameter>
      <Parameter name="ExerciseTimeStepsPerYear">{engine.exercise_time_steps_per_year}</Parameter>
      <Parameter name="ReferenceCalibrationGrid">{engine.reference_calibration_grid}</Parameter>
    </ModelParameters>
    {_solver_xml(engine)}"""


def pricingengine_xml(bermudan: OreLgmEngine, american: Optional[OreLgmEngine] = None) -> str:
    """ORE's default engines (discounting swaps and bonds, Bachelier Europeans) and the LGM
    engine of Bermudans and Americans (`american` defaults to `bermudan`'s)."""
    return f"""<PricingEngines>
  <Product type="Swap">
    <Model>DiscountedCashflows</Model>
    <ModelParameters/>
    <Engine>DiscountingSwapEngine</Engine>
    <EngineParameters/>
  </Product>
  <Product type="EuropeanSwaption">
    <Model>BlackBachelier</Model>
    <ModelParameters/>
    <Engine>BlackBachelierSwaptionEngine</Engine>
    <EngineParameters/>
  </Product>
  <Product type="Bond">
    <Model>DiscountedCashflows</Model>
    <ModelParameters/>
    <Engine>DiscountingRiskyBondEngine</Engine>
    <EngineParameters><Parameter name="TimestepPeriod">6M</Parameter></EngineParameters>
  </Product>
  <Product type="BermudanSwaption">{_lgm_product_xml(bermudan)}
  </Product>
  <Product type="AmericanSwaption">{_lgm_product_xml(american or bermudan)}
  </Product>
</PricingEngines>"""


# --- trades -------------------------------------------------------------------------------------

def _schedule_xml(dates, calendar: str = "TARGET", convention: str = "MF") -> str:
    inner = "\n".join(f"              <Date>{iso(d)}</Date>" for d in dates)
    return f"""<ScheduleData>
            <Dates>
              <Calendar>{calendar}</Calendar>
              <Convention>{convention}</Convention>
              <Dates>
{inner}
              </Dates>
            </Dates>
          </ScheduleData>"""


def _leg_dates(leg, as_coupon) -> list:
    coupons = [as_coupon(cf) for cf in leg]
    return [coupons[0].accrualStartDate()] + [c.accrualEndDate() for c in coupons]


def _envelope() -> str:
    return (f"<Envelope><CounterParty>CP</CounterParty><NettingSetId>{NETTING_SET}</NettingSetId>"
            f"<AdditionalFields/></Envelope>")


def _legs_xml(swap, notional, fixed_rate, payer, floating_spread, index, day_counter="A365") -> str:
    """The fixed and floating legs of the engine's booked `ORE.VanillaSwap`, on its own dates."""
    return f"""<LegData>
        <LegType>Fixed</LegType>
        <Payer>{str(payer).lower()}</Payer>
        <Currency>{CCY}</Currency>
        <Notionals><Notional>{_number(notional)}</Notional></Notionals>
        <DayCounter>{day_counter}</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FixedLegData><Rates><Rate>{_number(fixed_rate)}</Rate></Rates></FixedLegData>
        {_schedule_xml(_leg_dates(swap.fixedLeg(), ORE.as_fixed_rate_coupon))}
      </LegData>
      <LegData>
        <LegType>Floating</LegType>
        <Payer>{str(not payer).lower()}</Payer>
        <Currency>{CCY}</Currency>
        <Notionals><Notional>{_number(notional)}</Notional></Notionals>
        <DayCounter>{day_counter}</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>{index}</Index>
          <Spreads><Spread>{_number(floating_spread)}</Spread></Spreads>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
        {_schedule_xml(_leg_dates(swap.floatingLeg(), ORE.as_floating_rate_coupon))}
      </LegData>"""


def swaption_trade_xml(trade_id, swap, notional, fixed_rate, payer, floating_spread, index, style,
                       exercise_dates, mid_coupon_exercise=False, settlement="Physical") -> str:
    """A long swaption on the engine's underlying `swap`: `style` "European", "Bermudan" (one
    date per exercise) or "American" (the window's first and last day)."""
    exercise = "\n".join(f"          <ExerciseDate>{iso(d)}</ExerciseDate>" for d in exercise_dates)
    mid = "<MidCouponExercise>true</MidCouponExercise>" if mid_coupon_exercise else ""
    return f"""  <Trade id="{trade_id}">
    <TradeType>Swaption</TradeType>
    {_envelope()}
    <SwaptionData>
      <OptionData>
        <LongShort>Long</LongShort>
        <OptionType>Call</OptionType>
        <Style>{style}</Style>
        <Settlement>{settlement}</Settlement>
        <PayOffAtExpiry>false</PayOffAtExpiry>
        {mid}
        <ExerciseDates>
{exercise}
        </ExerciseDates>
      </OptionData>
      {_legs_xml(swap, notional, fixed_rate, payer, floating_spread, index)}
    </SwaptionData>
  </Trade>"""


def _day_counter_name(name: str) -> str:
    """An engine accrual day count as ORE's parser names it."""
    return resolve_accrual_day_count(name).name()


def trade_xml(cfg) -> str:
    """One engine trade config as ORE's trade XML, its id the trade's."""
    index = index_name(cfg.index_tenor_months) if not isinstance(cfg, BondConfig) else None
    if isinstance(cfg, SwapConfig):
        return f"""  <Trade id="{cfg.trade_id}">
    <TradeType>Swap</TradeType>
    {_envelope()}
    <SwapData>
      {_legs_xml(swap_underlying(cfg), cfg.notional, cfg.fixed_rate, cfg.payer, cfg.floating_spread, index,
                 _day_counter_name(cfg.accrual_day_count))}
    </SwapData>
  </Trade>"""
    if isinstance(cfg, SwaptionConfig):
        return swaption_trade_xml(cfg.trade_id, european_underlying(cfg), cfg.notional, cfg.fixed_rate, cfg.payer,
                                  cfg.floating_spread, index, "European", [cfg.exercise_date],
                                  settlement=cfg.settlement)
    if isinstance(cfg, AmericanSwaptionConfig):
        return swaption_trade_xml(cfg.trade_id, option_underlying(cfg), cfg.notional, cfg.fixed_rate, cfg.payer,
                                  cfg.floating_spread, index, "American",
                                  [cfg.first_exercise_date, cfg.last_exercise_date], settlement=cfg.settlement)
    if isinstance(cfg, BermudanSwaptionConfig):
        return swaption_trade_xml(cfg.trade_id, option_underlying(cfg), cfg.notional, cfg.fixed_rate, cfg.payer,
                                  cfg.floating_spread, index, "Bermudan", cfg.exercise_dates, settlement=cfg.settlement)
    if isinstance(cfg, BondConfig) and cfg.coupon_schedule and cfg.redemption_fraction == 1.0:
        periods = cfg.coupon_schedule
        if any(p.payment_date not in (None, p.end_date) for p in periods):
            raise NotImplementedError(f"bond {cfg.trade_id!r}: a payment date off the period end")
        dates = [periods[0].start_date] + [p.end_date for p in periods]
        return f"""  <Trade id="{cfg.trade_id}">
    <TradeType>Bond</TradeType>
    {_envelope()}
    <BondData>
      <IssuerId>TREASURY</IssuerId>
      <SecurityId>{cfg.trade_id}</SecurityId>
      <ReferenceCurveId>{DISCOUNT_INDEX}</ReferenceCurveId>
      <SettlementDays>0</SettlementDays>
      <Calendar>NullCalendar</Calendar>
      <IssueDate>{iso(dates[0])}</IssueDate>
      <LegData>
        <LegType>Fixed</LegType>
        <Payer>false</Payer>
        <Currency>{CCY}</Currency>
        <Notionals><Notional>{_number(cfg.face_amount)}</Notional></Notionals>
        <DayCounter>{_day_counter_name(cfg.accrual_day_count)}</DayCounter>
        <PaymentConvention>Unadjusted</PaymentConvention>
        <FixedLegData><Rates><Rate>{_number(cfg.coupon_rate)}</Rate></Rates></FixedLegData>
        {_schedule_xml(dates, "NullCalendar", "Unadjusted")}
      </LegData>
    </BondData>
  </Trade>"""
    raise NotImplementedError(f"trade {cfg.trade_id!r} ({type(cfg).__name__}): the oracle builds swaps, swaptions "
                              f"and coupon bonds redeemed at par")


def portfolio_xml(trades: Iterable[str]) -> str:
    return "<Portfolio>\n" + "\n".join(trades) + "\n</Portfolio>"


def netting_xml() -> str:
    return (f"<NettingSetDefinitions><NettingSet><NettingSetId>{NETTING_SET}</NettingSetId>"
            f"<ActiveCSAFlag>false</ActiveCSAFlag></NettingSet></NettingSetDefinitions>")


# --- market data --------------------------------------------------------------------------------

def market_lines(asof: ORE.Date, curves: OreCurves, vols: SwaptionVolSurface) -> list:
    """ORE's market data lines: the curves' quotes (the as-of zero quote solved for, I-34) and
    the swaption volatilities."""
    stamp = iso(asof).replace("-", "")
    if curves.discount_factors is None:
        dates = curve_dates(asof, curves.times)

        def zero(curve_id, rates):
            quotes = [_as_of_quote(curves.times, rates)] + list(rates[1:])
            return [f"{stamp} ZERO/RATE/{CCY}/{curve_id}/A365/{iso(d)} {_number(r)}" for d, r in zip(dates, quotes)]

        lines = zero(CURVE_ID, curves.rates)
        if curves.separate_index_curve:
            lines += zero(INDEX_CURVE_ID, curves.index_rates)
    else:
        given = curves.discount_factors
        lines = [f"{stamp} DISCOUNT/RATE/{CCY}/{curve_id}/{iso(d)} {_number(v)}"
                 for curve_id, dates, values in ((CURVE_ID, given.discount_dates, given.discount_factors),
                                                 (INDEX_CURVE_ID, given.index_dates, given.index_factors))
                 for d, v in zip(dates, values)]
    return lines + [f"{stamp} SWAPTION/RATE_NVOL/{CCY}/{e}/{t}/ATM {_number(v)}"
                    for e, row in zip(vols.option_tenors, vols.vols) for t, v in zip(vols.swap_tenors, row)]


def fixing_lines(index: str, fixings: Optional[Mapping[ORE.Date, float]]) -> list:
    return [f"{iso(d).replace('-', '')} {index} {_number(r)}" for d, r in (fixings or {}).items()]


# --- the run ------------------------------------------------------------------------------------

def inputs(asof: ORE.Date, index: str, curves: OreCurves, vols: SwaptionVolSurface, engines: str,
           portfolio: str) -> "ORE.InputParameters":
    """The `InputParameters` every run shares; the caller adds its analytics."""
    parameters = ORE.InputParameters()
    parameters.setAsOfDate(iso(asof))
    parameters.setBaseCurrency(CCY)
    parameters.setResultsPath(_scratch_dir())
    parameters.setEntireMarket(True)
    parameters.setAllFixings(True)
    parameters.setBuildFailedTrades(False)
    parameters.setConventions(conventions_xml(index))
    parameters.setCurveConfigs(curveconfig_xml(asof, curves, vols))
    parameters.setTodaysMarketParams(todaysmarket_xml(index, curves.separate_index_curve))
    parameters.setPricingEngine(engines)
    parameters.setPortfolio(portfolio)
    return parameters


def run_app(parameters, evaluation_date: ORE.Date, market: Sequence[str], fixings: Sequence[str],
            required_report: str) -> "ORE.OREApp":
    """Run `OREApp` over `parameters`, ORE's evaluation date restored afterwards; raises
    `RuntimeError` with ORE's errors and the log's alerts if `required_report` was not written."""
    previous = ORE.Settings.instance().evaluationDate
    log_file = f"{_scratch_dir()}/log.txt"
    try:
        ORE.Settings.instance().evaluationDate = evaluation_date
        app = ORE.OREApp(parameters, log_file, 31, False)
        app.run(ORE.StrVector(list(market)), ORE.StrVector(list(fixings)))
        if required_report not in app.getReportNames():
            with open(log_file, encoding="utf-8", errors="replace") as log:
                alerts = [line.strip() for line in log if "ALERT" in line or "ERROR" in line]
            raise RuntimeError(f"ORE produced no {required_report} report:\n" + "\n".join(list(app.getErrors()) + alerts))
        return app
    finally:
        ORE.Settings.instance().evaluationDate = previous


def report_columns(app, name: str) -> dict:
    """An ORE report as `{column: values}`: real columns as float arrays, string columns as
    lists; other column types (dates) are left out."""
    report = app.getReport(name)
    columns = {}
    for i in range(report.columns()):
        try:
            columns[report.header(i)] = np.asarray(report.dataAsReal(i), dtype=np.float64)
        except RuntimeError:
            try:
                columns[report.header(i)] = [str(v) for v in report.dataAsString(i)]
            except RuntimeError:
                pass
    return columns
