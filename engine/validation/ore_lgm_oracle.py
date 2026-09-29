"""
ORE's own Bermudan/American swaption engine, run in-process: the reference
`engine.instruments.bermudan_swaption` is validated against (tests/test_ore_lgm_parity.py).

Validation tooling, not a pricer: nothing in the pricing path imports it. Each call runs a
full `OREApp` (about a second) and changes process-wide ORE state (see below).

`NumericLgmMultiLegOptionEngine` has no SWIG constructor, so it is reached the way ORE users
reach it: an `OREApp` NPV run over trade XML, with every input in memory. (QuantLib's
Hull-White tree/FD engines, used by tests/test_ore_bermudan_oracle.py, are a different
model realization and agree only to a few percent.)

What is built:
  * Underlying: the engine's own `ORE.VanillaSwap` schedule, passed as explicit `<Dates>`.
    The index is a convention-defined `USD-SIMINDEX-<N>M` with `build_vanilla_swap`'s
    `SimIndex` terms (2 settlement days, TARGET, MF, no EOM, ACT/365).
  * Curve: date-quoted continuous ACT/365 zero rates, linear in the zero rate, as the
    engine's `ZeroCurve`. Pillars must be whole ACT/365 days from the as-of date (checked)
    and should start at t=0. Exact only where the curve is flat up to its first non-zero
    pillar: ORE's zero-curve build re-reads the t=0 rate as `zeroRate(1e-4)`
    (OREData/ored/marketdata/yieldcurve.cpp), tilting a sloped first segment by ~1e-6
    relative (I-34).
  * Model: `Calibration=None`, `ReversionType=HullWhite`, `VolatilityType=Hagan`, i.e.
    constant reversion `hw_a` and `zeta(t) = integral sigma^2`, as `engine.models.lgm`. A
    piecewise `Sigma` maps to `VolatilityTimes`/`Volatility` bucket for bucket. Defaults
    are the Grid solver and `ShiftHorizon=0` (what the engine reproduces); ORE's other
    settings are available (see `ore_lgm_swaption_npv`).

Two inputs ORE requires that do not affect the price:
  * A swap index pair (`USD-CMS-1Y`/`USD-CMS-30Y`). Required: `IrModelBuilder` takes the
    LGM term structure from the swap index's discounting curve and would otherwise fall
    back to a flat 1%. It is mapped to the engine's curve.
  * An ATM swaption vol quote, read only when calibrating.

Log file: `OREApp` needs a real log path and keeps the file open, so one scratch directory
per process is reused (`_scratch_dir`).

Global state: `OREApp` sets ORE's evaluation date (restored on exit) and its process-wide
`InstrumentConventions` (no SWIG accessor to restore). Another reason not to run it inside
a pricing process.
"""
from __future__ import annotations

import functools
import tempfile
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import ORE

from engine.market import SwaptionVolSurface
from engine.models.lgm import Sigma

CCY = "USD"
CURVE_ID = "SIMCURVE"
INDEX_CURVE_ID = "SIMINDEXCURVE"
#: An index whose forwarding curve is the discount curve: ORE's swap index curves name an
#: index for discounting, and the LGM builder takes the model's term structure from it.
DISCOUNT_INDEX = f"{CCY}-SIMDISC-6M"
#: The vol quote used when a caller gives no surface (read only when calibrating).
PLACEHOLDER_VOLS = SwaptionVolSurface(option_tenors=("1Y",), swap_tenors=("1Y",), vols=((0.01,),))


@dataclass(frozen=True)
class OreLgmResult:
    npv: float


@dataclass(frozen=True)
class OreDiscountCurves:
    """Discount and index curves as discount factors on dates after the as-of date,
    log-linear between them: how a simulated path's scenario curves reach ORE."""
    discount_dates: Sequence[ORE.Date]
    discount_factors: Sequence[float]
    index_dates: Sequence[ORE.Date]
    index_factors: Sequence[float]


@dataclass(frozen=True)
class OreCalibration:
    """ORE's LGM calibration settings (`LGMGridSwaptionEngineBuilder` model parameters).
    `method="None"` prices with the given volatility; `"Bootstrap"` calibrates a piecewise
    volatility to the trade's co-terminal basket (`strategy` `CoterminalATM` or
    `CoterminalDealStrike`), starting from it."""
    method: str = "None"
    strategy: str = "None"
    tolerance: float = 1e-4


@functools.lru_cache(maxsize=1)
def _scratch_dir() -> str:
    return tempfile.mkdtemp(prefix="ore_lgm_oracle_")


def _iso(d: ORE.Date) -> str:
    return f"{d.year():04d}-{d.month():02d}-{d.dayOfMonth():02d}"


def _index_name(index_tenor_months: int) -> str:
    return f"{CCY}-SIMINDEX-{index_tenor_months}M"


def _curve_dates(asof: ORE.Date, times: Sequence[float]) -> list:
    dates = []
    for t in times:
        days = round(t * 365)
        if abs(days / 365 - t) > 1e-12:
            raise ValueError(f"curve pillar {t} is not a whole number of ACT/365 days from the as-of date")
        dates.append(asof + days)
    return dates


def _conventions_xml(index: str) -> str:
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
        f"          <Quote>ZERO/RATE/{CCY}/{curve_id}/A365/{_iso(d)}</Quote>" for d in _curve_dates(asof, times)
    )
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
    quotes = "\n".join(f"          <Quote>DISCOUNT/RATE/{CCY}/{curve_id}/{_iso(d)}</Quote>" for d in dates)
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


def _curveconfig_xml(asof: ORE.Date, times: Sequence[float], separate_index_curve: bool,
                     vols: SwaptionVolSurface, discount_curves: Optional["OreDiscountCurves"] = None) -> str:
    if discount_curves is None:
        curves = _yield_curve_xml(CURVE_ID, asof, times)
        if separate_index_curve:
            curves += "\n" + _yield_curve_xml(INDEX_CURVE_ID, asof, times)
    else:
        curves = (_discount_curve_xml(CURVE_ID, discount_curves.discount_dates) + "\n"
                  + _discount_curve_xml(INDEX_CURVE_ID, discount_curves.index_dates))
    return f"""<CurveConfiguration>
  <YieldCurves>
{curves}
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


def _todaysmarket_xml(index: str, separate_index_curve: bool) -> str:
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


def _volatility_parameters(hw_sigma) -> str:
    if isinstance(hw_sigma, Sigma):
        times = ",".join(repr(float(t)) for t in np.asarray(hw_sigma.times))
        values = ",".join(repr(float(v)) for v in np.asarray(hw_sigma.values))
    else:
        times, values = "", repr(float(hw_sigma))
    return (f'<Parameter name="Volatility">{values}</Parameter>\n'
            f'      <Parameter name="VolatilityTimes">{times}</Parameter>')


@dataclass(frozen=True)
class OreFdSolver:
    """ORE's FD solver (`LGMFDSwaptionEngineBuilder` -> `LgmFdSolver`) instead of the Grid
    solver the engine reproduces. Defaults are ORE's example American settings
    (Examples/Products/Input/pricingengine.xml)."""
    scheme: str = "Douglas"
    state_grid_points: int = 64
    time_steps_per_year: int = 24
    mesher_epsilon: float = 1e-4


def _engine_xml(n_per_std, std_devs, fd_solver: Optional[OreFdSolver]) -> str:
    if fd_solver is None:
        return f"""<Engine>Grid</Engine>
    <EngineParameters>
      <Parameter name="sy">{std_devs!r}</Parameter>
      <Parameter name="ny">{n_per_std}</Parameter>
      <Parameter name="sx">{std_devs!r}</Parameter>
      <Parameter name="nx">{n_per_std}</Parameter>
    </EngineParameters>"""
    return f"""<Engine>FD</Engine>
    <EngineParameters>
      <Parameter name="Scheme">{fd_solver.scheme}</Parameter>
      <Parameter name="StateGridPoints">{fd_solver.state_grid_points}</Parameter>
      <Parameter name="TimeStepsPerYear">{fd_solver.time_steps_per_year}</Parameter>
      <Parameter name="MesherEpsilon">{fd_solver.mesher_epsilon!r}</Parameter>
    </EngineParameters>"""


def _pricingengine_xml(hw_a, hw_sigma, n_per_std, std_devs, exercise_time_steps_per_year,
                       shift_horizon, fd_solver, calibration: "OreCalibration") -> str:
    lgm = f"""
    <Model>LGM</Model>
    <ModelParameters>
      <Parameter name="Calibration">{calibration.method}</Parameter>
      <Parameter name="CalibrationStrategy">{calibration.strategy}</Parameter>
      <Parameter name="Reversion">{hw_a!r}</Parameter>
      <Parameter name="ReversionType">HullWhite</Parameter>
      {_volatility_parameters(hw_sigma)}
      <Parameter name="VolatilityType">Hagan</Parameter>
      <Parameter name="ShiftHorizon">{shift_horizon!r}</Parameter>
      <Parameter name="Tolerance">{calibration.tolerance!r}</Parameter>
      <Parameter name="ExerciseTimeStepsPerYear">{exercise_time_steps_per_year}</Parameter>
      <Parameter name="ReferenceCalibrationGrid">400,3M</Parameter>
    </ModelParameters>
    {_engine_xml(n_per_std, std_devs, fd_solver)}"""
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
  <Product type="BermudanSwaption">{lgm}
  </Product>
  <Product type="AmericanSwaption">{lgm}
  </Product>
</PricingEngines>"""


def _schedule_xml(dates) -> str:
    inner = "\n".join(f"              <Date>{_iso(d)}</Date>" for d in dates)
    return f"""<ScheduleData>
            <Dates>
              <Calendar>TARGET</Calendar>
              <Convention>MF</Convention>
              <Dates>
{inner}
              </Dates>
            </Dates>
          </ScheduleData>"""


def _leg_dates(leg, as_coupon) -> list:
    coupons = [as_coupon(cf) for cf in leg]
    return [coupons[0].accrualStartDate()] + [c.accrualEndDate() for c in coupons]


def _portfolio_xml(swap, notional, fixed_rate, payer, floating_spread, index,
                   style, exercise_dates, mid_coupon_exercise) -> str:
    exercise = "\n".join(f"          <ExerciseDate>{_iso(d)}</ExerciseDate>" for d in exercise_dates)
    mid = "<MidCouponExercise>true</MidCouponExercise>" if mid_coupon_exercise else ""
    fixed_dates = _leg_dates(swap.fixedLeg(), ORE.as_fixed_rate_coupon)
    float_dates = _leg_dates(swap.floatingLeg(), ORE.as_floating_rate_coupon)
    return f"""<Portfolio>
  <Trade id="T">
    <TradeType>Swaption</TradeType>
    <Envelope><CounterParty>CP</CounterParty><NettingSetId>NS</NettingSetId><AdditionalFields/></Envelope>
    <SwaptionData>
      <OptionData>
        <LongShort>Long</LongShort>
        <OptionType>Call</OptionType>
        <Style>{style}</Style>
        <Settlement>Physical</Settlement>
        <PayOffAtExpiry>false</PayOffAtExpiry>
        {mid}
        <ExerciseDates>
{exercise}
        </ExerciseDates>
      </OptionData>
      <LegData>
        <LegType>Fixed</LegType>
        <Payer>{str(payer).lower()}</Payer>
        <Currency>{CCY}</Currency>
        <Notionals><Notional>{notional!r}</Notional></Notionals>
        <DayCounter>A365</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FixedLegData><Rates><Rate>{fixed_rate!r}</Rate></Rates></FixedLegData>
        {_schedule_xml(fixed_dates)}
      </LegData>
      <LegData>
        <LegType>Floating</LegType>
        <Payer>{str(not payer).lower()}</Payer>
        <Currency>{CCY}</Currency>
        <Notionals><Notional>{notional!r}</Notional></Notionals>
        <DayCounter>A365</DayCounter>
        <PaymentConvention>MF</PaymentConvention>
        <FloatingLegData>
          <Index>{index}</Index>
          <Spreads><Spread>{floating_spread!r}</Spread></Spreads>
          <IsInArrears>false</IsInArrears>
          <FixingDays>2</FixingDays>
        </FloatingLegData>
        {_schedule_xml(float_dates)}
      </LegData>
    </SwaptionData>
  </Trade>
</Portfolio>"""


def ore_lgm_swaption_npv(
    *,
    evaluation_date: ORE.Date,
    curve_times: Sequence[float],
    curve_rates: Sequence[float],
    swap: ORE.VanillaSwap,
    notional: float,
    fixed_rate: float,
    payer: bool,
    floating_spread: float,
    index_tenor_months: int,
    style: str,
    exercise_dates: Sequence[ORE.Date],
    hw_a: float,
    hw_sigma: Union[float, Sigma],
    n_per_std: int,
    std_devs: float,
    exercise_time_steps_per_year: int = 24,
    mid_coupon_exercise: bool = False,
    shift_horizon: float = 0.0,
    fd_solver: Optional[OreFdSolver] = None,
    fixings: Optional[Mapping[ORE.Date, float]] = None,
    index_curve_rates: Optional[Sequence[float]] = None,
    swaption_vols: Optional[SwaptionVolSurface] = None,
    calibration: OreCalibration = OreCalibration(),
    discount_curves: Optional[OreDiscountCurves] = None,
) -> OreLgmResult:
    """NPV of a long, physically settled swaption on `swap`, by ORE's
    `NumericLgmMultiLegOptionEngine`.

    `style`: "Bermudan" (one date per exercise) or "American" (two dates: the window's
    first and last day). `swap` must be the engine's own underlying (normally
    `build_vanilla_swap` with the same terms); only its schedule is used. Raises
    `RuntimeError` with ORE's messages if the market or trade fails to build.

    Defaults match how the engine prices (Grid solver at `n_per_std`/`std_devs`,
    `shift_horizon=0`). `shift_horizon` (ORE's builder default is 0.5) and `fd_solver`
    measure the distance to ORE's other settings (I-32).

    `fixings`: historical index fixings `{ORE.Date: rate}`, for a seasoned trade.

    `index_curve_rates`: a forwarding curve for the Ibor index on the same pillars (default:
    the discount curve). `swaption_vols`: ATM normal swaption volatilities (default: one
    placeholder quote). `calibration`: ORE's LGM calibration settings. `discount_curves`:
    both curves as log-linear discount factors instead (`curve_times`/`curve_rates` are then
    unused). `style="European"`
    prices with ORE's default European engine (`BlackMultiLegOptionEngine` on the vols);
    the LGM arguments are then unused.
    """
    vols = swaption_vols or PLACEHOLDER_VOLS
    separate_index_curve = index_curve_rates is not None or discount_curves is not None
    index = _index_name(index_tenor_months)
    previous_evaluation_date = ORE.Settings.instance().evaluationDate
    out_dir = _scratch_dir()
    log_file = f"{out_dir}/log.txt"
    try:
        ORE.Settings.instance().evaluationDate = evaluation_date
        inputs = ORE.InputParameters()
        inputs.setAsOfDate(_iso(evaluation_date))
        inputs.setBaseCurrency(CCY)
        inputs.setResultsPath(out_dir)
        inputs.setEntireMarket(True)
        inputs.setAllFixings(True)
        inputs.setBuildFailedTrades(False)
        inputs.setConventions(_conventions_xml(index))
        inputs.setCurveConfigs(_curveconfig_xml(evaluation_date, curve_times, separate_index_curve, vols,
                                                discount_curves))
        inputs.setTodaysMarketParams(_todaysmarket_xml(index, separate_index_curve))
        inputs.setPricingEngine(_pricingengine_xml(
            hw_a, hw_sigma, n_per_std, std_devs, exercise_time_steps_per_year, shift_horizon, fd_solver,
            calibration))
        inputs.setPortfolio(_portfolio_xml(
            swap, notional, fixed_rate, payer, floating_spread, index,
            style, exercise_dates, mid_coupon_exercise))
        inputs.insertAnalytic("NPV")

        stamp = _iso(evaluation_date).replace("-", "")
        # float(r)!r: full precision, and a plain number for a numpy scalar (ORE cannot
        # parse "np.float64(...)").
        if discount_curves is None:
            dates = _curve_dates(evaluation_date, curve_times)
            market = [f"{stamp} ZERO/RATE/{CCY}/{CURVE_ID}/A365/{_iso(d)} {float(r)!r}"
                      for d, r in zip(dates, curve_rates)]
            if separate_index_curve:
                market += [f"{stamp} ZERO/RATE/{CCY}/{INDEX_CURVE_ID}/A365/{_iso(d)} {float(r)!r}"
                           for d, r in zip(dates, index_curve_rates)]
        else:
            market = [f"{stamp} DISCOUNT/RATE/{CCY}/{curve_id}/{_iso(d)} {float(v)!r}"
                      for curve_id, dates, values in (
                          (CURVE_ID, discount_curves.discount_dates, discount_curves.discount_factors),
                          (INDEX_CURVE_ID, discount_curves.index_dates, discount_curves.index_factors))
                      for d, v in zip(dates, values)]
        market += [f"{stamp} SWAPTION/RATE_NVOL/{CCY}/{e}/{t}/ATM {float(v)!r}"
                   for e, row in zip(vols.option_tenors, vols.vols) for t, v in zip(vols.swap_tenors, row)]

        fixing_lines = [f"{_iso(d).replace('-', '')} {index} {float(r)!r}" for d, r in (fixings or {}).items()]

        app = ORE.OREApp(inputs, log_file, 31, False)
        app.run(ORE.StrVector(market), ORE.StrVector(fixing_lines))
        if "npv" not in app.getReportNames():
            with open(log_file, encoding="utf-8", errors="replace") as log:
                alerts = [line.strip() for line in log if "ALERT" in line or "ERROR" in line]
            raise RuntimeError("ORE produced no npv report:\n" + "\n".join(list(app.getErrors()) + alerts))
        report = app.getReport("npv")
        columns = {report.header(i): i for i in range(report.columns())}
        return OreLgmResult(npv=float(report.dataAsReal(columns["NPV"])[0]))
    finally:
        ORE.Settings.instance().evaluationDate = previous_evaluation_date
