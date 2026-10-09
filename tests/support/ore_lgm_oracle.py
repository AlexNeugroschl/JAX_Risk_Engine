"""
ORE's own Bermudan/American swaption engine, run in-process: the reference
`engine.pricing.lgm_grid` is validated against (tests/test_ore_lgm_parity.py).

Test tooling, not a pricer: the engine never imports it. Each call runs a full `OREApp`
(about a second) over the in-memory inputs of `tests.support.ore_inputs`, which describes what
is built and the process-wide state it changes.

`NumericLgmMultiLegOptionEngine` has no SWIG constructor, so it is reached the way ORE users
reach it: an `OREApp` NPV run over trade XML. (QuantLib's Hull-White tree/FD engines, used by
tests/test_ore_bermudan_oracle.py, are a different model realization and agree only to a few
percent.) The model: `Calibration=None`, `ReversionType=HullWhite`, `VolatilityType=Hagan`,
i.e. constant reversion `hw_a` and `zeta(t) = integral sigma^2`, as `engine.models.lgm`; or
calibrated (`OreCalibration`). Defaults are the Grid solver and `ShiftHorizon=0` (what the
engine reproduces); ORE's other settings are available (see `ore_lgm_swaption_npv`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Union

import ORE

from engine.market_data.market import SwaptionVolSurface
from engine.models.lgm import Sigma
from tests.support.ore_inputs import (  # noqa: F401  (re-exported: the tests import them from here)
    PLACEHOLDER_VOLS, OreCalibration, OreCurves, OreDiscountCurves, OreFdSolver, OreLgmEngine, fixing_lines,
    index_name, inputs, market_lines, portfolio_xml, pricingengine_xml, report_columns, run_app, swaption_trade_xml,
)


@dataclass(frozen=True)
class OreLgmResult:
    npv: float


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
    unused). `style="European"` prices with ORE's default European engine
    (`BlackMultiLegOptionEngine` on the vols); the LGM arguments are then unused.
    """
    vols = swaption_vols or PLACEHOLDER_VOLS
    index = index_name(index_tenor_months)
    curves = OreCurves(curve_times, curve_rates, index_curve_rates, discount_curves)
    engine = OreLgmEngine(hw_a, hw_sigma, n_per_std, std_devs, exercise_time_steps_per_year, shift_horizon,
                          calibration=calibration, fd_solver=fd_solver)
    trade = swaption_trade_xml("T", swap, notional, fixed_rate, payer, floating_spread, index, style, exercise_dates,
                               mid_coupon_exercise)
    parameters = inputs(evaluation_date, index, curves, vols, pricingengine_xml(engine), portfolio_xml([trade]))
    parameters.insertAnalytic("NPV")
    app = run_app(parameters, evaluation_date, market_lines(evaluation_date, curves, vols),
                  fixing_lines(index, fixings), "npv")
    return OreLgmResult(npv=float(report_columns(app, "npv")["NPV"][0]))
