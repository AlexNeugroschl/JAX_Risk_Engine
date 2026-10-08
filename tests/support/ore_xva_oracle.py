"""
ORE's exposure simulation of a portfolio, run in-process: the reference the engine's assembled
simulation, NPV cube and exposure are validated against (tests/test_ore_xva_parity.py; roadmap
3.2, I-50; docs/planning/details/ore-parity-validation.md, layers L3 and L4, gate V-4).

Test tooling, not a pricer: the engine never imports it. One `OREApp` run of ORE's `EXPOSURE`
and `PFE` analytics (`XvaAnalytic`) over in-memory inputs (`tests.support.ore_inputs`): the
engine's market and trades, ORE's `simulation.xml` written from the engine's `CamConfig`
(`simulation_xml`) and the engine's pricing configuration. What it returns (`OreXvaRun`):

  * ORE's NPV cube: each trade's value on every path and simulation date divided by the
    numeraire, in double precision (`XvaUseDoublePrecisionCubes`: ORE's default stores single
    precision, which would cap every comparison near 1e-7), and its `T0`: each trade's value
    on the simulation market of the as-of date;
  * ORE's scenario dump (`writeScenarios`): the numeraire and each simulated curve's discount
    factors at the simulation tenors, on every path and date;
  * ORE's exposure reports, per trade and for the one netting set (no collateral): EPE, ENE,
    PFE at one quantile, Basel EE and EEE, time-weighted Basel EPE and EEPE, from t=0.

`implied_states` reads the cross-asset model's state on each path back from the numeraire
(gate V-4), so the engine can price on ORE's own paths (L3).

The simulation (`simulation_xml`): the engine's dates as a grid of day tenors with no calendar
and ACT/365 times (the engine's time axis), Sobol with a Brownian bridge (independent of the
engine's: decision X-1), the LGM measure and exact discretization; per currency ORE's `<LGM>`
with constant HullWhite reversion, the volatility `Hagan` (`LgmConfig`) or `HullWhite`
(`HullWhiteConfig`), constant or piecewise, or bootstrapped (`Calibration Bootstrap`, ORE's
time grid the basket's expiries) to the tenor basket at ATM; the simulation market's curves at
the configured tenors, log-linear, flat forward, and the swaption volatilities not simulated
but decayed (`ReactionToTimeDecay`). ORE values every trade with the engine's configured
engine and recalibrates options on each path and date. One currency, no FX or equity
(the two-currency end-to-end test is roadmap 4.7); what the oracle cannot express is refused
by name.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import ORE

from engine.market import Market
from engine.models.curves import log_discount
from engine.models.lgm import H as lgm_H, Sigma
from engine.models.ore_builders import TIME_AXIS_DAY_COUNTER
from engine.simulation.config import CamConfig, LgmConfig
from engine.valuation.config import PricingConfig
from tests.support.ore_inputs import (
    CCY, DISCOUNT_INDEX, NETTING_SET, OreCurves, OreLgmEngine, fixing_lines, index_name, inputs, market_lines,
    netting_xml, portfolio_xml, pricingengine_xml, report_columns, run_app, trade_xml,
)

#: ORE's exposure report columns the oracle returns, each from t=0.
EXPOSURE_COLUMNS = ("Time", "EPE", "ENE", "PFE", "BaselEE", "BaselEEE", "TimeWeightedBaselEPE",
                    "TimeWeightedBaselEEPE")
#: The key of the netting set's profile in `OreXvaRun.exposure`.
NETTING_SET_PROFILE = "netting_set"


@dataclass(frozen=True)
class OreXvaRun:
    """One ORE exposure simulation (see the module docstring). Trade axes follow `trade_ids`,
    the request's order; curve axes the simulation tenors."""
    trade_ids: Tuple[str, ...]
    dates: Tuple[ORE.Date, ...]
    times: np.ndarray                 # [D] ACT/365 from the as-of date
    cube: np.ndarray                  # [S, D, N] NPV / numeraire
    t0: np.ndarray                    # [N] NPV on the simulation market of the as-of date
    numeraire: np.ndarray             # [S, D]
    discount: np.ndarray              # [S, D, K] discount factors at the tenors
    index: Dict[str, np.ndarray]      # index name -> [S, D, K]
    exposure: Dict[str, Dict[str, np.ndarray]]  # trade id or NETTING_SET_PROFILE -> column -> [D + 1]

    @property
    def num_paths(self) -> int:
        return self.cube.shape[0]


def _volatility_xml(lgm: LgmConfig) -> str:
    """ORE's `<Volatility>` block of one currency's model."""
    if lgm.calibrated:
        start = lgm.volatility.values[0] if isinstance(lgm.volatility, Sigma) else lgm.volatility
        param, times, values, calibrate = "Piecewise", "", repr(float(start)), "Y"
    elif isinstance(lgm.volatility, Sigma):
        param, calibrate = "Piecewise", "N"
        times = ", ".join(repr(float(t)) for t in np.asarray(lgm.volatility.times))
        values = ", ".join(repr(float(v)) for v in np.asarray(lgm.volatility.values))
    else:
        param, times, values, calibrate = "Constant", "", repr(float(lgm.volatility)), "N"
    return f"""<Volatility>
          <Calibrate>{calibrate}</Calibrate>
          <VolatilityType>{lgm.volatility_type}</VolatilityType>
          <ParamType>{param}</ParamType>
          <TimeGrid>{times}</TimeGrid>
          <InitialValue>{values}</InitialValue>
        </Volatility>"""


def simulation_xml(asof: ORE.Date, config: CamConfig, indices: Sequence[str], vol_expiries: Sequence[str],
                   vol_terms: Sequence[str]) -> str:
    """ORE's `simulation.xml` (Parameters, CrossAssetModel, Market) for `config`."""
    _require_supported(config)
    lgm = config.ir[CCY]
    grid = ",".join(f"{d - asof}D" for d in config.dates)
    basket = (f"<Expiries>{','.join(lgm.calibration_expiries)}</Expiries><Terms>{','.join(lgm.calibration_terms)}"
              f"</Terms>") if lgm.calibrated else "<Expiries>1Y</Expiries><Terms>1Y</Terms>"
    index_xml = "".join(f"<Index>{name}</Index>" for name in indices)
    return f"""<Simulation>
  <Parameters>
    <Grid>{grid}</Grid>
    <Calendar>NullCalendar</Calendar>
    <DayCounter>A365</DayCounter>
    <Sequence>SobolBrownianBridge</Sequence>
    <Scenario>Simple</Scenario>
    <Seed>{config.seed}</Seed>
    <Samples>{config.samples}</Samples>
    <Ordering>Steps</Ordering>
    <DirectionIntegers>JoeKuoD7</DirectionIntegers>
  </Parameters>
  <CrossAssetModel>
    <DomesticCcy>{CCY}</DomesticCcy>
    <Currencies><Currency>{CCY}</Currency></Currencies>
    <BootstrapTolerance>0.0001</BootstrapTolerance>
    <Measure>LGM</Measure>
    <Discretization>Exact</Discretization>
    <InterestRateModels>
      <LGM key="{CCY}">
        <CalibrationType>{"Bootstrap" if lgm.calibrated else "None"}</CalibrationType>
        {_volatility_xml(lgm)}
        <Reversion>
          <Calibrate>N</Calibrate>
          <ReversionType>HullWhite</ReversionType>
          <ParamType>Constant</ParamType>
          <TimeGrid/>
          <InitialValue>{float(lgm.reversion)!r}</InitialValue>
        </Reversion>
        <CalibrationSwaptions>{basket}<Strikes/></CalibrationSwaptions>
        <ParameterTransformation><ShiftHorizon>0.0</ShiftHorizon><Scaling>1.0</Scaling></ParameterTransformation>
      </LGM>
    </InterestRateModels>
    <ForeignExchangeModels/>
    <InstantaneousCorrelations/>
  </CrossAssetModel>
  <Market>
    <BaseCurrency>{CCY}</BaseCurrency>
    <Currencies><Currency>{CCY}</Currency></Currencies>
    <YieldCurves>
      <Configuration>
        <Tenors>{",".join(config.curve_tenors)}</Tenors>
        <Interpolation>LogLinear</Interpolation>
        <Extrapolation>FlatFwd</Extrapolation>
      </Configuration>
    </YieldCurves>
    <Indices>{index_xml}</Indices>
    <SwapIndices>
      <SwapIndex><Name>{CCY}-CMS-1Y</Name><DiscountingIndex>{DISCOUNT_INDEX}</DiscountingIndex></SwapIndex>
      <SwapIndex><Name>{CCY}-CMS-30Y</Name><DiscountingIndex>{DISCOUNT_INDEX}</DiscountingIndex></SwapIndex>
    </SwapIndices>
    <SwaptionVolatilities>
      <Simulate>false</Simulate>
      <ReactionToTimeDecay>{config.swaption_vol_decay}</ReactionToTimeDecay>
      <Keys><Key>{CCY}</Key></Keys>
      <Expiries>{",".join(vol_expiries)}</Expiries>
      <Terms>{",".join(vol_terms)}</Terms>
    </SwaptionVolatilities>
    <AggregationScenarioDataCurrencies><Currency>{CCY}</Currency></AggregationScenarioDataCurrencies>
  </Market>
</Simulation>"""


def _require_supported(config: CamConfig) -> None:
    if set(config.ir) != {CCY} or config.fx_volatilities or config.equity_volatilities or config.correlations:
        raise NotImplementedError(f"the XVA oracle simulates {CCY} alone (no FX, equity or correlations); the "
                                  f"two-currency test is roadmap 4.7")
    conventions = config.ir[CCY].swap_index
    if (conventions.fixed_tenor, conventions.fixed_day_counter) != ("1Y", "ACT/365"):
        raise NotImplementedError(f"the XVA oracle's swap index is annual ACT/365; got {conventions}")


def ore_xva_run(market: Market, trades: Sequence, simulation: CamConfig, pricing: PricingConfig = PricingConfig(),
                pfe_quantile: float = 0.95) -> OreXvaRun:
    """ORE's exposure simulation of `trades` on `market` under `simulation` and `pricing` (see
    the module docstring). Raises `RuntimeError` with ORE's messages if a trade or the model
    fails to build."""
    if pricing.european != "Bachelier" or not pricing.recalibrate:
        raise NotImplementedError("the XVA oracle prices Europeans by ORE's Bachelier engine and recalibrates "
                                  "options on every path (PricingConfig's defaults)")
    data = market.currency(CCY)
    index = index_name(simulation.ir[CCY].swap_index.index_tenor_months)
    if set(data.index_curves) != {index} or list(data.index_curves[index].times) != list(data.discount_curve.times):
        raise NotImplementedError(f"the XVA oracle's market is {CCY}'s discount curve and {index}'s, on one set of "
                                  f"pillars")
    curves = OreCurves(data.discount_curve.times, data.discount_curve.rates, data.index_curves[index].rates)
    vols = data.swaption_vols
    engines = pricingengine_xml(OreLgmEngine.of(pricing.bermudan), OreLgmEngine.of(pricing.american))
    parameters = inputs(market.asof, index, curves, vols, engines, portfolio_xml(trade_xml(t) for t in trades))
    sim = simulation_xml(market.asof, simulation, (index, DISCOUNT_INDEX), vols.option_tenors, vols.swap_tenors)
    parameters.setExposureSimMarketParams(sim)
    parameters.setScenarioGeneratorData(sim)
    parameters.setCrossAssetModelData(sim)
    parameters.setSimulationPricingEngine(engines)
    parameters.setNettingSetManager(netting_xml())
    parameters.setExposureBaseCurrency(CCY)
    parameters.setXvaBaseCurrency(CCY)
    parameters.setExposureObservationModel("Disable")
    parameters.setExposureProfiles(True)
    parameters.setExposureProfilesByTrade(True)
    parameters.setPfeQuantile(pfe_quantile)
    parameters.setWriteScenarios(True)
    parameters.setWriteCube(True)
    parameters.setXvaUseDoublePrecisionCubes(True)
    parameters.insertAnalytic("EXPOSURE")
    parameters.insertAnalytic("PFE")
    fixings = [line for t in trades for line in fixing_lines(index, getattr(t, "fixings", None))]
    app = run_app(parameters, market.asof, market_lines(market.asof, curves, vols), fixings, "scenario")
    return _read(app, market.asof, simulation, [t.trade_id for t in trades])


def _read(app, asof: ORE.Date, simulation: CamConfig, trade_ids) -> OreXvaRun:
    dates, samples = simulation.dates, simulation.samples
    cube = app.getCube("cube")
    ore_ids = list(cube.ids())
    positions = [ore_ids.index(i) for i in trade_ids]
    values = np.asarray([[[cube.get(j, d, s, 0) for j in positions] for d in range(len(dates))]
                         for s in range(samples)])

    scenario = report_columns(app, "scenario")
    # ScenarioWriter: one row per path and date, path-major; paths are labelled from 1.
    labels = np.asarray(scenario["Scenario"], dtype=np.int64).reshape(samples, len(dates))
    assert np.array_equal(labels, np.broadcast_to(np.arange(1, samples + 1)[:, None], labels.shape))

    def curve(prefix):
        keys = sorted((k for k in scenario if k.startswith(prefix)), key=lambda k: int(k.rsplit("/", 1)[1]))
        return np.stack([scenario[k].reshape(samples, len(dates)) for k in keys], axis=-1)

    names = {k.split("/")[1] for k in scenario if k.startswith("IndexCurve/")}
    exposure = {i: _profile(app, f"exposure_trade_{i}") for i in trade_ids}
    exposure[NETTING_SET_PROFILE] = _profile(app, f"exposure_nettingset_{NETTING_SET}")
    return OreXvaRun(trade_ids=tuple(trade_ids), dates=tuple(dates),
                     times=np.asarray([TIME_AXIS_DAY_COUNTER.yearFraction(asof, d) for d in dates]),
                     cube=values, t0=np.asarray([cube.getT0(j, 0) for j in positions]),
                     numeraire=scenario["Numeraire"].reshape(samples, len(dates)),
                     discount=curve(f"DiscountCurve/{CCY}/"), index={n: curve(f"IndexCurve/{n}/") for n in names},
                     exposure=exposure)


def _profile(app, report: str) -> Dict[str, np.ndarray]:
    columns = report_columns(app, report)
    return {name: columns[name] for name in EXPOSURE_COLUMNS}


def implied_states(model, times: np.ndarray, numeraire: np.ndarray) -> np.ndarray:
    """`[S, D, 1]`: the domestic LGM state on each path and date whose numeraire is ORE's,
    N(t, z) = exp(H(t) z + 1/2 H(t)^2 zeta(t)) / P(0, t) solved for z on the engine's model
    (an `engine.simulation.cam.CrossAssetModel` of one currency). Every other quantity of the
    simulation market follows from the state (gate V-4)."""
    ir = model.ir[0]
    h = np.asarray(lgm_H(ir.reversion, times))
    zeta = np.asarray(ir.zeta(times))
    p0 = np.exp(np.asarray(log_discount(ir.curve, times)))
    return ((np.log(numeraire * p0[None, :]) - 0.5 * h ** 2 * zeta) / h)[:, :, None]
