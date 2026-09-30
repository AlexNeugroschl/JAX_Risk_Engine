"""
American swaption pricing on the Bermudan engine in `engine.instruments.bermudan_swaption`.

ORE's `NumericLgmMultiLegOptionEngineBase::calculate()`
(QuantExt/qle/pricingengines/numericlgmmultilegoptionengine.cpp) treats an American
differently from a Bermudan in two places, both reproduced here:

  1. Option times: for the window `[first, last]`, `t1 = max(0, t(first))`,
     `t2 = max(t1, t(last))`, `steps = max(1, Size((t2 - t1) * ExerciseTimeStepsPerYear))`
     (truncated, not rounded), and times `t1 + i*(t2-t1)/steps` for `i = 0..steps`
     (`AmericanSwaptionConfig.option_times`).
  2. Coupon membership: a coupon belongs until its accrual end and is credited
     `couponRatio(t) = (accrualEnd - t) / (accrualEnd - accrualStart)`
     (`ExerciseStyle.AMERICAN`, applied in `bermudan_swaption._build_grid_schedule`).

An `AmericanSwaptionConfig` is priced by the same functions as a `BermudanSwaptionConfig`.
Checked against ORE's engine by tests/test_ore_lgm_parity.py.
"""
from dataclasses import InitVar, dataclass, field
from typing import Dict, List, Optional, Union

import jax
import ORE

from engine.instruments.bermudan_swaption import ExerciseStyle, price_bermudan_swaptions
from engine.models.lgm import Sigma
from engine.models.ore_builders import book_swap_dates, is_live, time_from_reference, validate_fixings
from engine.simulation.market_model import ZeroCurveConfig
from engine.instruments._validation import _validate_common_fields, _validate_hw_sigma, _validate_settlement


@dataclass
class AmericanSwaptionConfig:
    """
    One American swaption, exercisable on any day in
    `[first_exercise_date, last_exercise_date]`, represented as ORE does: a uniform grid of
    `exercise_time_steps_per_year` option times per year, each exercising into the
    remaining swap with the current coupon credited pro rata.

    exercise_time_steps_per_year: ORE's `ExerciseTimeStepsPerYear`. The default, 24, is
        what ORE's example config (Examples/Products/Input/pricingengine.xml) uses.
    Other fields are as in `BermudanSwaptionConfig`.
    """
    notional: float
    fixed_rate: float
    payer: bool
    rate_factor_index: Optional[int] = None
    hw_a: Optional[float] = None
    hw_sigma: Optional[Union[float, Sigma]] = None
    initial_zero_curve: Optional[ZeroCurveConfig] = None
    first_exercise_date: Optional[ORE.Date] = None
    last_exercise_date: Optional[ORE.Date] = None
    effective_date: Optional[ORE.Date] = None
    maturity_date: Optional[ORE.Date] = None
    swap_tenor: InitVar[Optional[str]] = None
    index_tenor_months: int = 6
    floating_spread: float = 0.0
    exercise_time_steps_per_year: int = 24
    n_per_std: int = 48
    std_devs: float = 6.0
    evaluation_date: ORE.Date = field(default_factory=lambda: ORE.Settings.instance().evaluationDate)
    fixings: Dict[ORE.Date, float] = field(default_factory=dict)
    index_zero_curve: Optional[ZeroCurveConfig] = None
    currency: str = "USD"
    settlement: str = "Physical"

    exercise_style = ExerciseStyle.AMERICAN

    def option_times(self) -> List[float]:
        """ORE's American `optionTimes`, with its truncating step count and the same
        arithmetic (`t1 + i * (t2 - t1) / steps`, left to right as in C++).

        The window starts no earlier than the day after the evaluation date, as ORE's
        trade builder sets it (`ExerciseBuilder`, OREData/ored/portfolio/optiondata.cpp:
        `max(today + 1, first)`)."""
        first = max(self.evaluation_date + 1, self.first_exercise_date)
        t1 = max(0.0, time_from_reference(self.evaluation_date, first))
        t2 = max(t1, time_from_reference(self.evaluation_date, self.last_exercise_date))
        steps = max(1, int((t2 - t1) * float(self.exercise_time_steps_per_year)))
        return sorted({t1} | {t1 + float(i) * (t2 - t1) / float(steps) for i in range(steps + 1)})

    def is_expired(self) -> bool:
        """The window's last day is on or before the evaluation date (ORE's
        `isExpired`)."""
        return not is_live(self.last_exercise_date, self.evaluation_date)

    def __post_init__(self, swap_tenor: Optional[str]) -> None:
        _validate_common_fields(self.notional, self.fixed_rate, self.evaluation_date)
        book_swap_dates(self, swap_tenor)
        validate_fixings(self.fixings)
        # hw_sigma=None means "uncalibrated" (see BermudanSwaptionConfig).
        _validate_hw_sigma(self.hw_sigma)
        _validate_settlement(self.settlement)
        for name in ("first_exercise_date", "last_exercise_date"):
            if not isinstance(getattr(self, name), ORE.Date):
                raise TypeError(f"{name} must be an ORE.Date; got {getattr(self, name)!r}")
        if self.last_exercise_date < self.first_exercise_date:
            raise ValueError(
                f"first_exercise_date ({self.first_exercise_date}) must be on or before "
                f"last_exercise_date ({self.last_exercise_date})"
            )
        if self.exercise_time_steps_per_year < 1:
            raise ValueError(
                f"exercise_time_steps_per_year must be >= 1; got {self.exercise_time_steps_per_year}"
            )


def price_american_swaptions(
    american_configs: List[AmericanSwaptionConfig],
    hw_paths: jax.Array,
    step_times: jax.Array,
) -> jax.Array:
    """NPV cube for American swaptions (the shared Bermudan pricer)."""
    return price_bermudan_swaptions(american_configs, hw_paths, step_times)
