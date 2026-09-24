"""
American swaption pricing, on engine.instruments.bermudan_swaption's numeric
LGM backward-induction engine.

**How ORE prices an American, and so how this module does.**
`QuantExt::NumericLgmMultiLegOptionEngineBase::calculate()`
(QuantExt/qle/pricingengines/numericlgmmultilegoptionengine.cpp) differs from
its Bermudan path in exactly two places, and this module reproduces both:

  1. **The option times.** The window `[first, last]` becomes
     `t1 = max(0, t(first))`, `t2 = max(t1, t(last))`,
     `steps = max(1, static_cast<Size>((t2 - t1) * ExerciseTimeStepsPerYear))`
     -- a TRUNCATION, not a rounding -- and option times `t1 + i*(t2-t1)/steps`
     for `i = 0..steps` (lines 494-505). `AmericanSwaptionConfig.option_times`.
  2. **Which coupons an exercise enters.** An American exercise can land
     inside an accrual period, so a coupon keeps belonging to the
     exercised-into swap until its accrual END and is credited
     `couponRatio(t) = (accrualEnd - t) / (accrualEnd - accrualStart)` of its
     value (`buildCashflowInfo`, lines 107-109: "american exercise implies
     that we can exercise into broken periods"). Carried by
     `ExerciseStyle.AMERICAN` and applied in
     `bermudan_swaption._exercise_value_at_nodes`.

Everything else -- the LGM state grid, Hagan's convolution, the
numeraire-deflated `max(exercise, continuation)` induction -- is shared with
the Bermudan and lives in engine.instruments.bermudan_swaption. An
`AmericanSwaptionConfig` is priced by the same functions a
`BermudanSwaptionConfig` is (`prepare_bermudan`, `price_bermudan_swaption_base`,
`price_bermudan_swaptions`); there is no conversion step between the two.
Verified against ORE's own engine by tests/test_ore_lgm_parity.py.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Union

import jax
import numpy as np
import ORE

from engine.instruments.bermudan_swaption import ExerciseStyle, price_bermudan_swaptions
from engine.models.lgm import Sigma
from engine.models.ore_builders import time_from_reference
from engine.simulation.market_model import ZeroCurveConfig
from engine.portfolio.validation import _validate_common_fields, _validate_tenor


@dataclass
class AmericanSwaptionConfig:
    """
    One American swaption: exercisable on any day in the window
    `[first_exercise_date, last_exercise_date]`, represented exactly as ORE
    represents it -- `exercise_time_steps_per_year` option times per year
    across the window (see the module docstring), each exercising into the
    remaining swap with the in-progress coupon credited pro rata.

    exercise_time_steps_per_year: ORE's own `ExerciseTimeStepsPerYear`
    model parameter; ORE's shipped example config
    (Examples/Products/Input/pricingengine.xml) uses 24 (~monthly), which is
    this field's default.

    hw_sigma accepts either a plain float or an `engine.models.lgm.Sigma`
    (a piecewise-constant term structure, e.g. from `engine.calibration`),
    exactly as `BermudanSwaptionConfig` does.
    """
    notional: float
    fixed_rate: float
    payer: bool
    rate_factor_index: int
    hw_a: float
    hw_sigma: Optional[Union[float, Sigma]]
    initial_zero_curve: ZeroCurveConfig
    first_exercise_date: ORE.Date
    last_exercise_date: ORE.Date
    swap_tenor: str = "5Y"
    index_tenor_months: int = 6
    floating_spread: float = 0.0
    exercise_time_steps_per_year: int = 24
    n_per_std: int = 48
    std_devs: float = 6.0
    evaluation_date: ORE.Date = field(default_factory=lambda: ORE.Settings.instance().evaluationDate)

    exercise_style = ExerciseStyle.AMERICAN

    def option_times(self) -> List[float]:
        """ORE's American `optionTimes` (`calculate()`, lines 494-505),
        including its truncating step count and its exact arithmetic
        (`t1 + i * (t2 - t1) / steps`, evaluated left to right as in C++)."""
        t1 = max(0.0, time_from_reference(self.evaluation_date, self.first_exercise_date))
        t2 = max(t1, time_from_reference(self.evaluation_date, self.last_exercise_date))
        steps = max(1, int((t2 - t1) * float(self.exercise_time_steps_per_year)))
        return sorted({t1} | {t1 + float(i) * (t2 - t1) / float(steps) for i in range(steps + 1)})

    def __post_init__(self) -> None:
        _validate_common_fields(self.notional, self.fixed_rate, self.evaluation_date)
        _validate_tenor(self.swap_tenor, "swap_tenor")
        # None is a valid sentinel meaning "uncalibrated" -- see
        # BermudanSwaptionConfig.__post_init__'s identical handling.
        if self.hw_sigma is not None:
            sigma_values = self.hw_sigma.values if isinstance(self.hw_sigma, Sigma) else [self.hw_sigma]
            if any(v != v or v in (float("inf"), float("-inf")) for v in np.asarray(sigma_values, dtype=np.float64).tolist()):
                raise ValueError(f"hw_sigma must be finite; got {self.hw_sigma}")
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
    """The NPV cube for a list of American swaptions -- the shared
    backward-induction pricer, which reads each config's own option times
    and exercise style (see module docstring)."""
    return price_bermudan_swaptions(american_configs, hw_paths, step_times)


# =============================================================================
# EXECUTION DEMONSTRATION
# =============================================================================
if __name__ == "__main__":
    import jax.numpy as jnp

    from engine.instruments.bermudan_swaption import price_bermudan_swaption_base
    from engine.simulation.market_model import generate_paths
    from engine.simulation.demo_scenarios import EVAL_DATE, swaption_demo_config

    config = swaption_demo_config()
    market_cubes = generate_paths(config)
    step_times = jnp.array(config.time_grid[1:], dtype=jnp.float64)

    zero_curve = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[0.03] * 6)

    american_cfg = AmericanSwaptionConfig(
        notional=1_000_000.0,
        fixed_rate=0.030,
        payer=True,
        rate_factor_index=0,
        hw_a=config.rates.mean_reversion[0],
        hw_sigma=float(np.sqrt(config.joint_covariance[1][1])),
        initial_zero_curve=zero_curve,
        first_exercise_date=EVAL_DATE + ORE.Period(1, ORE.Years),
        last_exercise_date=EVAL_DATE + ORE.Period(4, ORE.Years),
        swap_tenor="5Y",
        evaluation_date=EVAL_DATE,
    )
    print("American exercise opportunities:", len(american_cfg.option_times()))
    print("American t=0 NPV:", price_bermudan_swaption_base(american_cfg))

    npv_cube = price_american_swaptions([american_cfg], market_cubes["rates"], step_times)
    print("American NPV cube shape:", npv_cube.shape)
    for i, t in enumerate(config.time_grid[1:]):
        print(f"  t={t:.2f}: mean NPV across scenarios = {float(jnp.mean(npv_cube[:, i, 0])):.2f}")
