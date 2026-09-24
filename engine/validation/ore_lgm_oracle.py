"""
ORE's own Bermudan/American swaption engine, driven in-process -- the
reference `engine.instruments.bermudan_swaption` is validated against
(tests/test_ore_lgm_parity.py), and a tool for reconciling any Bermudan or
American price with ORE.

VALIDATION TOOLING, NOT A PRICER. Nothing in the pricing path imports this
module. It runs a full `OREApp` per call (about a second) and changes
process-wide ORE state (see GLOBAL STATE below), so it belongs in tests,
reconciliation jobs and investigations, never inside a pricing request.

WHY THIS EXISTS. ORE prices Bermudan and American swaptions with
`QuantExt::NumericLgmMultiLegOptionEngine` (built by
`LGMGridSwaptionEngineBuilder`). Its SWIG class has no bound constructor, so
`tests/test_ore_bermudan_oracle.py` falls back to QuantLib's Hull-White tree
and FD engines -- a different model realization, good only to a few percent.
This module reaches the real engine by the route ORE users take: an
`OREApp` run of the `NPV` analytic over a trade XML, with every input held in
memory. The engine it builds is the same LGM model, the same convolution
solver and the same exercise logic this codebase reproduces, so agreement is
expected at numerical, not model, level.

WHAT IS BUILT, and why each piece is exactly the engine's:

  * **The underlying.** Schedule dates are read off the engine's own
    `ORE.VanillaSwap` and passed to ORE as explicit `<Dates>`, so both sides
    price the same accrual periods by construction. The floating index is a
    convention-defined `USD-SIMINDEX-<N>M` carrying `build_vanilla_swap`'s
    `SimIndex` terms (2 settlement days, TARGET, ModifiedFollowing, no EOM,
    ACT/365).
  * **The curve.** Date-quoted continuous ACT/365 zero rates, linearly
    interpolated in the zero rate -- the engine's `ZeroCurve`. Pillars must
    be whole numbers of ACT/365 days (checked), and the first pillar should
    be at t=0: ORE otherwise inserts a flat point at the as-of date.
  * **The model.** `Calibration=None`, `ReversionType=HullWhite`,
    `VolatilityType=Hagan`: ORE's LGM with constant reversion `hw_a` and
    Hagan alpha `hw_sigma`, i.e. `zeta(t) = hw_sigma^2 * t`, the
    parametrization `engine.models.lgm` implements. A piecewise
    `engine.models.lgm.Sigma` maps onto ORE's `VolatilityTimes`/`Volatility`
    pair bucket for bucket (same `[times[i-1], times[i])` convention).
    By default `ShiftHorizon=0` and the Grid solver, which is what the engine
    reproduces; both can be changed to ORE's other settings (see
    `ore_lgm_swaption_npv`).

TWO INPUTS ORE REQUIRES THAT DO NOT AFFECT THE PRICE, supplied only so the
model builder does not fall back to dummies:

  * A swap index pair (`USD-CMS-1Y`/`USD-CMS-30Y`). **Not optional, and not
    inert:** `IrModelBuilder` takes the LGM's own term structure from the
    swap index's *discounting* curve, so a fallback would silently replace
    the model curve with a flat 1%. It is mapped to the engine's curve.
  * An ATM swaption vol quote. Read only when calibrating; unused here.

ORE'S LOG FILE. `OREApp` refuses an empty log path, and its logger keeps the
file open after the run, so a per-call temporary directory cannot be
deleted. One scratch directory per process (`_scratch_dir`) is reused
instead; it holds only ORE's log and an empty results folder.

GLOBAL STATE. `OREApp` sets ORE's evaluation date and its process-wide
`InstrumentConventions`. The evaluation date is restored on exit. The
conventions registry has no SWIG accessor to restore it with; nothing in
the pricing path parses ORE conventions, so that residue is inert for it --
but it is a reason this must not run inside a pricing process.
"""
from __future__ import annotations

import functools
import tempfile
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import ORE

from engine.models.lgm import Sigma

CCY = "USD"
CURVE_ID = "SIMCURVE"


@dataclass(frozen=True)
class OreLgmResult:
    npv: float


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


def _curveconfig_xml(asof: ORE.Date, times: Sequence[float]) -> str:
    quotes = "\n".join(
        f"          <Quote>ZERO/RATE/{CCY}/{CURVE_ID}/A365/{_iso(d)}</Quote>" for d in _curve_dates(asof, times)
    )
    return f"""<CurveConfiguration>
  <YieldCurves>
    <YieldCurve>
      <CurveId>{CURVE_ID}</CurveId>
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
    </YieldCurve>
  </YieldCurves>
  <SwaptionVolatilities>
    <SwaptionVolatility>
      <CurveId>SIMVOL</CurveId>
      <CurveDescription>required by the LGM builder; never read with Calibration=None</CurveDescription>
      <Dimension>ATM</Dimension>
      <VolatilityType>Normal</VolatilityType>
      <Extrapolation>Flat</Extrapolation>
      <DayCounter>A365</DayCounter>
      <Calendar>TARGET</Calendar>
      <BusinessDayConvention>Following</BusinessDayConvention>
      <OptionTenors>1Y</OptionTenors>
      <SwapTenors>1Y</SwapTenors>
      <ShortSwapIndexBase>{CCY}-CMS-1Y</ShortSwapIndexBase>
      <SwapIndexBase>{CCY}-CMS-30Y</SwapIndexBase>
    </SwaptionVolatility>
  </SwaptionVolatilities>
</CurveConfiguration>"""


def _todaysmarket_xml(index: str) -> str:
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
    <Index name="{index}">Yield/{CCY}/{CURVE_ID}</Index>
  </IndexForwardingCurves>
  <SwapIndexCurves id="default">
    <SwapIndex name="{CCY}-CMS-1Y"><Discounting>{index}</Discounting></SwapIndex>
    <SwapIndex name="{CCY}-CMS-30Y"><Discounting>{index}</Discounting></SwapIndex>
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
    """ORE's alternative "FD" solver (`LGMFDSwaptionEngineBuilder` ->
    `LgmFdSolver`) in place of the "Grid" convolution solver this engine
    reproduces. The defaults are ORE's shipped American swaption settings
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
                       shift_horizon, fd_solver) -> str:
    lgm = f"""
    <Model>LGM</Model>
    <ModelParameters>
      <Parameter name="Calibration">None</Parameter>
      <Parameter name="CalibrationStrategy">None</Parameter>
      <Parameter name="Reversion">{hw_a!r}</Parameter>
      <Parameter name="ReversionType">HullWhite</Parameter>
      {_volatility_parameters(hw_sigma)}
      <Parameter name="VolatilityType">Hagan</Parameter>
      <Parameter name="ShiftHorizon">{shift_horizon!r}</Parameter>
      <Parameter name="Tolerance">0.0001</Parameter>
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
) -> OreLgmResult:
    """NPV of a long physically-settled swaption on `swap`, priced by ORE's
    `NumericLgmMultiLegOptionEngine` (Grid solver).

    `style` is `"Bermudan"` (one date per exercise opportunity) or
    `"American"` (exactly two dates: the window's first and last day).
    `swap` must be the engine's own underlying -- normally
    `engine.models.ore_builders.build_vanilla_swap` with the same terms --
    so its schedule is what ORE prices; its index's forward curve is never
    read. Raises `RuntimeError` with ORE's own messages if ORE fails to
    build the market or the trade, rather than returning a partial result.

    The defaults configure ORE exactly as this engine prices: the Grid solver
    at the engine's own `n_per_std`/`std_devs`, and `shift_horizon=0`. Two
    settings ORE's own example configs use instead are available for
    measuring how far the engine sits from them: `shift_horizon` (ORE's
    `ShiftHorizon`, a fraction of the trade's maturity; ORE's builder default
    is 0.5) and `fd_solver` (ORE's FD solver; `n_per_std`/`std_devs` are then
    unused by ORE).
    """
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
        inputs.setCurveConfigs(_curveconfig_xml(evaluation_date, curve_times))
        inputs.setTodaysMarketParams(_todaysmarket_xml(index))
        inputs.setPricingEngine(_pricingengine_xml(
            hw_a, hw_sigma, n_per_std, std_devs, exercise_time_steps_per_year, shift_horizon, fd_solver))
        inputs.setPortfolio(_portfolio_xml(
            swap, notional, fixed_rate, payer, floating_spread, index,
            style, exercise_dates, mid_coupon_exercise))
        inputs.insertAnalytic("NPV")

        stamp = _iso(evaluation_date).replace("-", "")
        market = [f"{stamp} ZERO/RATE/{CCY}/{CURVE_ID}/A365/{_iso(d)} {r!r}"
                  for d, r in zip(_curve_dates(evaluation_date, curve_times), curve_rates)]
        market.append(f"{stamp} SWAPTION/RATE_NVOL/{CCY}/1Y/1Y/ATM 0.01")

        app = ORE.OREApp(inputs, log_file, 31, False)
        app.run(ORE.StrVector(market), ORE.StrVector([]))
        if "npv" not in app.getReportNames():
            with open(log_file, encoding="utf-8", errors="replace") as log:
                alerts = [line.strip() for line in log if "ALERT" in line or "ERROR" in line]
            raise RuntimeError("ORE produced no npv report:\n" + "\n".join(list(app.getErrors()) + alerts))
        report = app.getReport("npv")
        columns = {report.header(i): i for i in range(report.columns())}
        return OreLgmResult(npv=float(report.dataAsReal(columns["NPV"])[0]))
    finally:
        ORE.Settings.instance().evaluationDate = previous_evaluation_date
