"""
ORE's sensitivity analysis of a portfolio, run in-process: the reference the engine's Greeks are
validated against (tests/test_sensitivity_ore_parity.py; I-51).

Test tooling, not a pricer: the engine never imports it. An `OREApp` run of ORE's `SENSITIVITY`
analytic (`SensitivityAnalysis`, OREAnalytics/orea/engine/sensitivityanalysis.cpp) over
in-memory inputs (`tests.support.ore_inputs`): the engine's market and trades, its pricing
configuration, and ORE's `sensitivity.xml` and sensitivity simulation market written from the
engine's `SensitivityConfig` (`sensitivity_xml`, `ore_inputs.sim_market_xml`):

  * every curve sampled at `curve_tenors`, log-linear in the discount factor, flat forward
    beyond; each tenor's zero rate shifted by `curve_shift`, absolute, forward scheme, the shift
    tenors the curve tenors; Gamma computed;
  * the ATM swaption volatilities simulated (a risk factor) at the surface's own expiries and
    terms, each shifted by `vol_shift`, absolute, forward scheme;
  * Theta over `theta_days` calendar days (`computeTheta`, `thetaPeriod`);
  * the option models recalibrated under every shift (`recalibrateModels`, ORE's default).

What it returns (`OreSensitivityRun`) is ORE's report read into the engine's keys
(`engine.risk.greeks.bump.portfolio_sensitivities`), per trade: `delta:<kind>:<name>` and
`gamma:<kind>:<name>` per curve tenor, `vega:<ccy>` per quote, `theta`, and ORE's base NPV
(each trade on the sensitivity market). A factor ORE omits (its Delta is 0) reads 0.

**One engine curve is two ORE factors.** ORE's LGM model builder takes its term structure and
its calibration helpers' discounting from the swap index's discounting curve
(`IrModelBuilder`: `swapIndex_->discountingTermStructure()`), and a bond discounts on its
`ReferenceCurveId`; in ORE's sensitivity market each is an index curve
(`IndexCurve/<DISCOUNT_INDEX>`), a risk factor of its own beside `DiscountCurve/<ccy>`, even
where both come from one market curve. The engine has one discount curve per currency, which
the option's model and the bond read too. So the oracle shifts both ORE factors, with their
cross gammas, and reads the engine's discount curve as both moved together, each tenor `k`:

    NPV(both up) - NPV(base) = delta_D + delta_S + crossGamma_DS,

ORE's own forward difference of the joint move (its cross scenario is both shifts at once,
`SensitivityScenarioGenerator`), so the joint Delta is exact. ORE's cross gamma has no
down-down counterpart, so the joint Gamma `NPV(both up) - 2 NPV(base) + NPV(both down)` takes
a second run at the negated shift: the sum of the two runs' joint differences. (Approximating
the down-down cross term by the up-up one is not enough: a recalibrated Bermudan on its grid is
not smooth at the shift's scale, and the approximation missed by a third at some tenors.)

ORE's cross-gamma filter names an index curve by its currency alone (`ScenarioDescription::
keyName`: `IndexCurve/USD` for every USD index), so the filter asks for the discount curve's
cross gammas with every index curve, and the oracle reads those with `DISCOUNT_INDEX` at the
same tenor.

One currency, one Ibor index; what the oracle cannot express is refused by name.
"""
from __future__ import annotations

import dataclasses
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import ORE

from engine.market_data.market import Market
from engine.pricing.config import PricingConfig
from engine.risk.greeks.bump import SensitivityConfig
from tests.support.ore_inputs import (
    CCY, DISCOUNT_INDEX, OreCurves, OreLgmEngine, fixing_lines, index_name, inputs, market_lines, portfolio_xml,
    pricingengine_xml, report_columns, run_app, sim_market_xml, trade_xml,
)

#: ORE's names of the two factors that are the engine's discount curve.
_DISCOUNT, _SWAP_DISCOUNT = f"DiscountCurve/{CCY}", f"IndexCurve/{DISCOUNT_INDEX}"


@dataclass(frozen=True)
class OreSensitivityRun:
    """ORE's sensitivity report in the engine's keys (see the module docstring), by trade id."""
    base_npv: Dict[str, float]
    greeks: Dict[str, Dict[str, np.ndarray]]


def _curve_shift_xml(tag: str, attribute: str, name: str, config: SensitivityConfig) -> str:
    return f"""<{tag} {attribute}="{name}">
      <ShiftType>Absolute</ShiftType>
      <ShiftSize>{config.curve_shift!r}</ShiftSize>
      <ShiftScheme>Forward</ShiftScheme>
      <ShiftTenors>{",".join(config.curve_tenors)}</ShiftTenors>
    </{tag}>"""


def sensitivity_xml(index: str, vols, config: SensitivityConfig, discount_only: bool = False) -> str:
    """ORE's `sensitivity.xml` for `config` (see the module docstring); with `discount_only`, the
    two discount factors and their cross gammas alone (the negated run's)."""
    others = "" if discount_only else f"""
    {_curve_shift_xml("IndexCurve", "index", index, config)}"""
    vol = "" if discount_only else f"""
  <SwaptionVolatilities>
    <SwaptionVolatility ccy="{CCY}">
      <ShiftType>Absolute</ShiftType>
      <ShiftSize>{config.vol_shift!r}</ShiftSize>
      <ShiftScheme>Forward</ShiftScheme>
      <ShiftExpiries>{",".join(vols.option_tenors)}</ShiftExpiries>
      <ShiftStrikes/>
      <ShiftTerms>{",".join(vols.swap_tenors)}</ShiftTerms>
    </SwaptionVolatility>
  </SwaptionVolatilities>"""
    return f"""<SensitivityAnalysis>
  <DiscountCurves>{_curve_shift_xml("DiscountCurve", "ccy", CCY, config)}</DiscountCurves>
  <IndexCurves>{others}
    {_curve_shift_xml("IndexCurve", "index", DISCOUNT_INDEX, config)}
  </IndexCurves>{vol}
  <CrossGammaFilter><Pair>{_DISCOUNT},IndexCurve/{CCY}</Pair></CrossGammaFilter>
  <ComputeGamma>true</ComputeGamma>
</SensitivityAnalysis>"""


def ore_sensitivity_run(market: Market, trades: Sequence, pricing: PricingConfig = PricingConfig(),
                        config: SensitivityConfig = SensitivityConfig()) -> OreSensitivityRun:
    """ORE's sensitivity analysis of `trades` on `market` (see the module docstring): two
    `OREApp` runs. Raises `RuntimeError` with ORE's messages if a trade or the market fails to
    build."""
    if pricing.european != "Bachelier":
        raise NotImplementedError("the sensitivity oracle prices Europeans by ORE's Bachelier engine "
                                  "(PricingConfig's default)")
    if set(market.currencies) != {CCY}:
        raise NotImplementedError(f"the sensitivity oracle's market is {CCY} alone")
    data = market.currency(CCY)
    tenors = {t.index_tenor_months for t in trades if hasattr(t, "index_tenor_months")}
    index = index_name(tenors.pop() if len(tenors) == 1 else 6)
    if tenors or set(data.index_curves) != {index} or \
            list(data.index_curves[index].times) != list(data.discount_curve.times):
        raise NotImplementedError(f"the sensitivity oracle's market is {CCY}'s discount curve and one index's, on one "
                                  f"set of pillars")
    up = _run(market, trades, pricing, config, index, discount_only=False)
    down = _run(market, trades, pricing, dataclasses.replace(config, curve_shift=-config.curve_shift), index,
                discount_only=True)
    greeks = {}
    for cfg in trades:
        trade = cfg.trade_id
        moved, ore = up.joint[trade], up.greeks[trade]
        out = {f"delta:discount:{CCY}": moved, f"gamma:discount:{CCY}": moved + down.joint[trade]}
        if hasattr(cfg, "index_tenor_months"):
            out[f"delta:index:{index}"] = ore[("IndexCurve", index, "delta")]
            out[f"gamma:index:{index}"] = ore[("IndexCurve", index, "gamma")]
        if trade in up.vega:
            out[f"vega:{CCY}"] = up.vega[trade]
        out["theta"] = np.asarray(up.theta[trade])
        greeks[trade] = out
    return OreSensitivityRun(up.base_npv, greeks)


@dataclass(frozen=True)
class _Report:
    """One run's report: per trade, ORE's Delta and Gamma per factor (`(kind, name, "delta" |
    "gamma")` -> `[K]`), the joint forward difference of the two discount factors `[K]`, Vega,
    Theta and the base NPV."""
    base_npv: Dict[str, float]
    greeks: Dict[str, Dict[tuple, np.ndarray]]
    joint: Dict[str, np.ndarray]
    vega: Dict[str, np.ndarray]
    theta: Dict[str, float]


def _run(market: Market, trades: Sequence, pricing: PricingConfig, config: SensitivityConfig, index: str,
         discount_only: bool) -> _Report:
    data = market.currency(CCY)
    curves = OreCurves(data.discount_curve.times, data.discount_curve.rates, data.index_curves[index].rates)
    vols = data.swaption_vols
    engines = pricingengine_xml(OreLgmEngine.of(pricing.bermudan), OreLgmEngine.of(pricing.american))
    parameters = inputs(market.asof, index, curves, vols, engines, portfolio_xml(trade_xml(t) for t in trades))
    parameters.setSensiSimMarketParams(
        f"<Simulation>{sim_market_xml(config.curve_tenors, (index, DISCOUNT_INDEX), vols, simulate_vols=True)}"
        f"</Simulation>")
    parameters.setSensiScenarioData(sensitivity_xml(index, vols, config, discount_only))
    parameters.setSensiRecalibrateModels(True)
    parameters.setSensiThreshold(0.0)
    parameters.setComputeTheta(not discount_only)
    parameters.setThetaPeriod(ORE.Period(config.theta_days, ORE.Days))
    parameters.insertAnalytic("SENSITIVITY")
    fixings = [line for t in trades for line in fixing_lines(index, getattr(t, "fixings", None))]
    app = run_app(parameters, market.asof, market_lines(market.asof, curves, vols), fixings, "sensitivity")
    return _read(report_columns(app, "sensitivity"), len(config.curve_tenors), vols)


def _read(report, tenors: int, vols) -> _Report:
    columns = len(vols.swap_tenors)
    greeks: Dict[str, Dict[tuple, np.ndarray]] = defaultdict(lambda: defaultdict(lambda: np.zeros(tenors)))
    cross: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(tenors))
    vega: Dict[str, np.ndarray] = {}
    theta, base = {}, {}
    rows = zip(report["TradeId"], report["Factor_1"], report["Factor_2"], report["Base NPV"], report["Delta"],
               report["Gamma"])
    for trade, factor, other, npv, delta, gamma in rows:
        base[trade] = float(npv)
        kind, name, position = (factor.split("/") + ["", ""])[:3]
        if other:   # a cross gamma, in Gamma
            if factor.startswith(_DISCOUNT + "/") and other.startswith(_SWAP_DISCOUNT + "/") \
                    and position == other.split("/")[2]:
                cross[trade][int(position)] = gamma
        elif kind == "Theta":
            theta[trade] = float(delta)
        elif kind == "SwaptionVolatility":
            vega.setdefault(trade, np.zeros((len(vols.option_tenors), columns)))[divmod(int(position), columns)] = delta
        elif kind in ("DiscountCurve", "IndexCurve"):
            greeks[trade][(kind, name, "delta")][int(position)] = delta
            greeks[trade][(kind, name, "gamma")][int(position)] = gamma
        else:
            raise ValueError(f"unexpected ORE sensitivity factor {factor!r}")
    joint = defaultdict(lambda: np.zeros(tenors))
    joint.update({trade: (factors[("DiscountCurve", CCY, "delta")] + factors[("IndexCurve", DISCOUNT_INDEX, "delta")]
                          + cross[trade]) for trade, factors in greeks.items()})
    return _Report(base, greeks, joint, vega, theta)
