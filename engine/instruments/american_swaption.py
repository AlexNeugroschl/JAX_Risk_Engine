"""
American swaption pricing -- a thin wrapper around
engine.instruments.bermudan_swaption's numeric LGM backward-induction
engine.

**American exercise is NOT priced continuously.** ORE itself does not do
this either: `QuantExt::NumericLgmMultiLegOptionEngineBase::calculate()`
(QuantExt/qle/pricingengines/numericlgmmultilegoptionengine.cpp) discretizes
an American exercise window `[t1, t2]` into
`steps = round((t2 - t1) * americanExerciseTimeStepsPerYear)` equally spaced
exercise dates and then runs the exact same discrete-date backward induction
Bermudan uses -- confirmed by reading ORE's own trade-builder source
(`OREData/ored/portfolio/builders/swaption.hpp`/`.cpp`): both
`BermudanSwaption` and `AmericanSwaption` route to the same
`LGMSwaptionEngineBuilder`, and `NumericLgmMultiLegOptionEngineBase` has no
separate code path for American beyond this initial discretization step.

This module reproduces that literally: `AmericanSwaptionConfig` takes an
`exercise_time_steps_per_year` parameter (ORE's own `ExerciseTimeStepsPerYear`
model parameter, defaulting to 24 -- ~monthly -- in ORE's own shipped example
configs) and an exercise window `[first_exercise, last_exercise]`, and builds
the discretized date list exactly as ORE's C++ does
(`AmericanSwaptionConfig.to_bermudan()`). "American" is therefore, here as in
ORE, simply "Bermudan with a very fine, evenly-spaced exercise schedule" --
`price_american_swaptions` expands into a
`bermudan_swaption.BermudanSwaptionConfig` and delegates entirely to
`bermudan_swaption.price_bermudan_swaptions`; every algorithmic detail (the
LGM state grid, Hagan's quadrature convolution, numeraire-deflated backward
induction, early-exercise comparison) lives in that module -- see
`engine.instruments.bermudan_swaption`'s own docstring and
docs/10-american-swaptions.md for the full algorithm writeup.
"""
from dataclasses import dataclass, field
from typing import List

import jax
import ORE

from engine.instruments.bermudan_swaption import BermudanSwaptionConfig, price_bermudan_swaptions
from engine.simulation import ZeroCurveConfig


@dataclass
class AmericanSwaptionConfig:
    """
    One American swaption: exercisable at any time in a continuous window
    `[first_exercise, last_exercise]` -- represented, exactly as ORE itself
    represents it, by discretizing that window into
    `exercise_time_steps_per_year`-spaced exercise dates and pricing as a
    (very fine) Bermudan (see module docstring, and
    `NumericLgmMultiLegOptionEngineBase::calculate()`'s American branch,
    `QuantExt/qle/pricingengines/numericlgmmultilegoptionengine.cpp` lines
    ~494-505).

    exercise_time_steps_per_year: ORE's own `ExerciseTimeStepsPerYear`
    model parameter; ORE's own shipped example config
    (Examples/Products/Input/pricingengine.xml) uses 24 (~monthly) for
    American swaptions, which is this field's default.

    **Known limitation: no mid-coupon proration.** ORE's own American
    engine supports exercise landing INSIDE an accrual period (a "broken"
    coupon), prorating that period's payment via `couponRatio` (see module
    docstring's ORE citation, `belongsToUnderlyingMaxTime_` using
    `accrualEndDate()` for American specifically). This module's
    (`bermudan_swaption._hw_swap_value_at_nodes`) uses a simpler, coarser
    rule (any coupon whose accrual has already STARTED at the exercise time
    is excluded entirely from the remaining swap value -- see that
    function's docstring) which is exact when every exercise date coincides
    with a reset date (BermudanSwaptionConfig's own documented scope) but,
    for a mid-coupon American exercise date, understates the true value
    slightly (the holder forfeits the ALREADY-ACCRUED portion of the
    in-progress coupon entirely, rather than receiving its prorated share
    as ORE's engine would). This is a conservative (understating, not
    overstating) approximation, not a silent/dangerous error -- verified
    not to produce nonsensical output (see
    tests/test_bermudan_swaption.py's TestMidCouponKnownLimitation), and
    `exercise_time_steps_per_year` values that evenly divide the
    underlying's own reset frequency (e.g. a semi-annual-reset swap with
    `exercise_time_steps_per_year` a multiple of 2) avoid it entirely by
    construction.
    """
    notional: float
    fixed_rate: float
    payer: bool
    rate_factor_index: int
    hw_a: float
    hw_sigma: float
    initial_zero_curve: ZeroCurveConfig
    first_exercise: float
    last_exercise: float
    swap_tenor: str = "5Y"
    index_tenor_months: int = 6
    floating_spread: float = 0.0
    exercise_time_steps_per_year: int = 24
    n_per_std: int = 48
    std_devs: float = 6.0
    evaluation_date: ORE.Date = field(default_factory=lambda: ORE.Settings.instance().evaluationDate)

    def to_bermudan(self) -> BermudanSwaptionConfig:
        """
        Expand the continuous exercise window into ORE's own discretized
        exercise-date grid: `steps = round((t2-t1) * stepsPerYear)` equally
        spaced dates over `[t1, t2]`, INCLUDING both endpoints -- the exact
        construction in `NumericLgmMultiLegOptionEngineBase::calculate()`
        (`optionTimes.insert(t1); for i in 0..steps: optionTimes.insert(t1 +
        i*(t2-t1)/steps)`, which inserts `t1` itself twice into a std::set,
        a no-op, and always includes `t2` at `i=steps`).
        """
        t1, t2 = self.first_exercise, self.last_exercise
        steps = max(1, round((t2 - t1) * self.exercise_time_steps_per_year))
        exercise_times = sorted(set(
            [t1] + [t1 + i * (t2 - t1) / steps for i in range(steps + 1)]
        ))
        return BermudanSwaptionConfig(
            notional=self.notional, fixed_rate=self.fixed_rate, payer=self.payer,
            rate_factor_index=self.rate_factor_index, hw_a=self.hw_a, hw_sigma=self.hw_sigma,
            initial_zero_curve=self.initial_zero_curve, exercise_times=exercise_times,
            swap_tenor=self.swap_tenor, index_tenor_months=self.index_tenor_months,
            floating_spread=self.floating_spread, n_per_std=self.n_per_std, std_devs=self.std_devs,
            evaluation_date=self.evaluation_date,
        )


def price_american_swaptions(
    american_configs: List[AmericanSwaptionConfig],
    hw_paths: jax.Array,
    step_times: jax.Array,
) -> jax.Array:
    """Convenience wrapper: expand each AmericanSwaptionConfig into its
    discretized-exercise-window BermudanSwaptionConfig (see
    AmericanSwaptionConfig.to_bermudan, ORE's own American-as-fine-Bermudan
    convention) and price through the identical Bermudan machinery
    (engine.instruments.bermudan_swaption.price_bermudan_swaptions)."""
    return price_bermudan_swaptions([cfg.to_bermudan() for cfg in american_configs], hw_paths, step_times)


# =============================================================================
# EXECUTION DEMONSTRATION
# =============================================================================
if __name__ == "__main__":
    import numpy as np
    import jax.numpy as jnp

    from engine.instruments.bermudan_swaption import price_bermudan_swaption_base
    from engine.simulation import generate_paths
    from engine.scenarios import EVAL_DATE, swaption_demo_config

    config = swaption_demo_config()
    market_cubes = generate_paths(config)
    step_times = jnp.array(config.time_grid[1:], dtype=jnp.float64)

    zero_curve = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[0.03] * 6)

    # Exercisable any time in [1Y, 4Y], discretized every 6 months --
    # reset-aligned for this scenario's semiannual floating leg, avoiding
    # the mid-coupon limitation documented above.
    american_cfg = AmericanSwaptionConfig(
        notional=1_000_000.0,
        fixed_rate=0.030,
        payer=True,
        rate_factor_index=0,
        hw_a=config.rates.mean_reversion[0],
        hw_sigma=float(np.sqrt(config.joint_covariance[1][1])),
        initial_zero_curve=zero_curve,
        first_exercise=1.0,
        last_exercise=4.0,
        swap_tenor="5Y",
        exercise_time_steps_per_year=2,
    )
    berm_equivalent = american_cfg.to_bermudan()
    print("American discretized into", len(berm_equivalent.exercise_times), "exercise dates")
    print("American t=0 NPV:", price_bermudan_swaption_base(berm_equivalent))

    npv_cube = price_american_swaptions([american_cfg], market_cubes["rates"], step_times)
    print("American NPV cube shape:", npv_cube.shape)
    for i, t in enumerate(config.time_grid[1:]):
        print(f"  t={t:.2f}: mean NPV across scenarios = {float(jnp.mean(npv_cube[:, i, 0])):.2f}")
