"""
Bermudan and American swaption prices must equal ORE's own
`NumericLgmMultiLegOptionEngine`, to floating-point precision.

The oracle (`engine/validation/ore_lgm_oracle.py`) runs ORE's real pricing path --
trade XML -> `LGMGridSwaptionEngineBuilder` -> `NumericLgmMultiLegOptionEngine`
-- on this engine's own underlying swap, curve and LGM parameters. The
engine runs ORE's own backward loop (same model, same convolution rollback,
same cashflow bookkeeping, same grid), so at the same grid settings it
reproduces ORE's NUMBERS, not only their converged limit. Measured worst
case across every case here: 8.7e-12 relative. Hence a 1e-10 tolerance,
not the few-percent model gap `test_ore_bermudan_oracle.py` has to allow
against QuantLib's Hull-White engines: any change to the algorithm, however
small, shows up.

WHAT EACH CASE GROUP PINS, and the ORE source that defines it:

  * aligned Bermudan -- exercise on fixed accrual starts. Exercises every
    floating coupon's projection, including coupons whose index fixing
    period differs from their accrual period (I-31):
    `LgmVectorised::fixing` projects over `[valueDate(fix), maturityDate]`.
  * mid-period Bermudan -- exercise 100 days into a period. ORE exercises
    into the next WHOLE period (`belongsToUnderlyingMaxTime_ =
    accrualStart` for Bermudan), so the in-progress coupon is excluded.
  * American -- ORE keeps a coupon until its accrual END and credits
    `couponRatio(t)` of it (I-06). High strike at low vol is where the
    engine used to be furthest off (6x).
  * American with a fractional step count >= 0.5 -- ORE sizes the grid with
    a truncating `static_cast<Size>`, not rounding.
  * zero volatility -- the option collapses to its best intrinsic value,
    a direct check of the underlying's cashflow valuation.
  * piecewise Sigma -- ORE's `VolatilityTimes`/`Volatility`.
"""
from dataclasses import dataclass
from typing import Callable, Union

import jax
import numpy as np
import ORE
import pytest

from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig, price_bermudan_swaption_base
from engine.models.lgm import Sigma
from engine.models.ore_builders import build_vanilla_swap
from engine.simulation.market_model import ZeroCurveConfig
from engine.validation.ore_lgm_oracle import ore_lgm_swaption_npv

jax.config.update("jax_enable_x64", True)

EVAL_DATE = ORE.Date(15, 6, 2025)
NOTIONAL = 1_000_000.0
HW_A = 0.03
CURVE_TIMES = [0.0, 1.0, 2.0, 5.0, 10.0, 30.0]
CURVE_RATES = [0.02, 0.025, 0.03, 0.035, 0.04, 0.04]
SWAP_TENOR = "5Y"
INDEX_TENOR_MONTHS = 6
N_PER_STD = 48
STD_DEVS = 6.0
STEPS_PER_YEAR = 24

AMERICAN_START = ORE.Date(20, 6, 2026)
AMERICAN_END = ORE.Date(20, 6, 2029)
# (end - start) * 24 = 1105 / 365 * 24 = 72.66: ORE truncates to 72 steps,
# rounding would give 73.
AMERICAN_FRACTIONAL_END = AMERICAN_START + 1105

# Measured worst case 8.7e-12 relative (floating-point summation order).
RTOL = 1e-10
ATOL = 1e-7  # currency units on a 1e6 notional, for small out-of-the-money values


def _swap(fixed_rate: float, payer: bool) -> ORE.VanillaSwap:
    return build_vanilla_swap(NOTIONAL, fixed_rate, payer, SWAP_TENOR, INDEX_TENOR_MONTHS, 0.0, EVAL_DATE)


def _fixed_starts(swap) -> list:
    return [ORE.as_fixed_rate_coupon(cf).accrualStartDate() for cf in swap.fixedLeg()]


def _aligned(swap) -> list:
    return _fixed_starts(swap)[1:]


def _mid_period(swap) -> list:
    return [d + 100 for d in _fixed_starts(swap)[1:]]


@dataclass(frozen=True)
class Case:
    id: str
    style: str                       # "Bermudan" | "American"
    payer: bool
    fixed_rate: float
    sigma: Union[float, Sigma]
    exercise: Callable               # swap -> list of ORE.Date (American: [first, last])


def _american(end=AMERICAN_END):
    return lambda swap: [AMERICAN_START, end]


PIECEWISE = Sigma(times=np.array([1.0, 3.0]), values=np.array([0.008, 0.011, 0.009]))

CASES = [
    *[Case(f"bermudan-aligned-{'payer' if p else 'receiver'}-{k}", "Bermudan", p, k, 0.01, _aligned)
      for p in (True, False) for k in (0.02, 0.03, 0.04)],
    *[Case(f"bermudan-mid-period-{'payer' if p else 'receiver'}-{k}", "Bermudan", p, k, 0.01, _mid_period)
      for p in (True, False) for k in (0.02, 0.04)],
    *[Case(f"american-{'payer' if p else 'receiver'}-{k}", "American", p, k, 0.01, _american())
      for p in (True, False) for k in (0.02, 0.03, 0.04)],
    Case("american-payer-high-strike-low-vol", "American", True, 0.04, 0.002, _american()),
    Case("american-receiver-low-vol", "American", False, 0.03, 0.002, _american()),
    Case("american-fractional-step-count", "American", True, 0.03, 0.01, _american(AMERICAN_FRACTIONAL_END)),
    Case("bermudan-piecewise-sigma", "Bermudan", True, 0.03, PIECEWISE, _aligned),
    Case("american-piecewise-sigma", "American", False, 0.03, PIECEWISE, _american()),
]


def _engine_npv(case: Case, exercise_dates: list) -> float:
    """The one place this file touches the engine's config API."""
    common = dict(
        notional=NOTIONAL, fixed_rate=case.fixed_rate, payer=case.payer, rate_factor_index=0,
        hw_a=HW_A, hw_sigma=case.sigma,
        initial_zero_curve=ZeroCurveConfig(times=CURVE_TIMES, rates=CURVE_RATES),
        swap_tenor=SWAP_TENOR, index_tenor_months=INDEX_TENOR_MONTHS,
        n_per_std=N_PER_STD, std_devs=STD_DEVS, evaluation_date=EVAL_DATE,
    )
    if case.style == "Bermudan":
        return price_bermudan_swaption_base(BermudanSwaptionConfig(exercise_dates=exercise_dates, **common))
    first, last = exercise_dates
    return price_bermudan_swaption_base(AmericanSwaptionConfig(
        first_exercise_date=first, last_exercise_date=last,
        exercise_time_steps_per_year=STEPS_PER_YEAR, **common))


def _ore_npv(case: Case, exercise_dates: list) -> float:
    return ore_lgm_swaption_npv(
        evaluation_date=EVAL_DATE, curve_times=CURVE_TIMES, curve_rates=CURVE_RATES,
        swap=_swap(case.fixed_rate, case.payer), notional=NOTIONAL, fixed_rate=case.fixed_rate,
        payer=case.payer, floating_spread=0.0, index_tenor_months=INDEX_TENOR_MONTHS,
        style=case.style, exercise_dates=exercise_dates, hw_a=HW_A, hw_sigma=case.sigma,
        n_per_std=N_PER_STD, std_devs=STD_DEVS, exercise_time_steps_per_year=STEPS_PER_YEAR,
    ).npv


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_engine_equals_ore_lgm_engine(case):
    exercise_dates = case.exercise(_swap(case.fixed_rate, case.payer))
    ore = _ore_npv(case, exercise_dates)
    engine = _engine_npv(case, exercise_dates)
    assert ore > 0.0, "a case must price something to test"
    assert engine == pytest.approx(ore, rel=RTOL, abs=ATOL)


@pytest.mark.parametrize("payer", [True, False], ids=["payer", "receiver"])
def test_zero_volatility_equals_ore_intrinsic(payer):
    """At vanishing vol the option is its best exercise's intrinsic value, so
    this isolates the underlying's cashflow valuation from the rollback."""
    case = Case("zero-vol", "Bermudan", payer, 0.03, 1e-7, _aligned)
    exercise_dates = case.exercise(_swap(case.fixed_rate, case.payer))
    assert _engine_npv(case, exercise_dates) == pytest.approx(_ore_npv(case, exercise_dates), rel=RTOL, abs=ATOL)
