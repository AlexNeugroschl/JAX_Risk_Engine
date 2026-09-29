"""
What an option is worth on a simulated path around its exercise: ORE's `OptionWrapper`
(OREData/ored/portfolio/optionwrapper.cpp), which `Swaption::build` wraps every swaption in
as a `BermudanOptionWrapper`, one underlying swap per exercise date (plan T-11, I-43).

  * A contract exercise date after the as-of date and within the grid takes effect on the
    first grid date on or after it (`initialise`: `lower_bound`); on that date the option's
    own price is its continuation value, the exercise itself having passed.
  * On an effective date a path that has not exercised yet exercises into that date's
    underlying when `underlying NPV - option NPV > 0` (the first exercise date mapping to the
    grid date decides).
  * After exercise a physically settled option is its underlying swap on every later date;
    a cash-settled one is 0, since its settlement date is the exercise date, which has passed
    by the effective date (ORE settles on the exercise date without `PaymentData`).
  * The underlying of exercise date e is the swap from the coupon before the first one
    accruing from e (`buildUnderlyingSwaps`: `lower_bound` on the accrual start, then one
    back), on both legs.
"""
from typing import List, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import ORE


def effective_steps(exercise_dates: Sequence[ORE.Date], asof: ORE.Date,
                    grid: Sequence[ORE.Date]) -> List[Optional[int]]:
    """For each contract exercise date, the index of the grid date it takes effect on, or
    None if it is not after the as-of date or falls after the grid."""
    serials = np.array([d.serialNumber() for d in grid], dtype=np.int64)
    steps = []
    for e in exercise_dates:
        inside = asof < e and serials.size > 0 and e.serialNumber() <= serials[-1]
        steps.append(int(np.searchsorted(serials, e.serialNumber(), side="left")) if inside else None)
    return steps


def underlying_start(accrual_starts: Sequence[ORE.Date], exercise_date: ORE.Date) -> int:
    """Index of the first coupon of the swap exercise date `exercise_date` enters."""
    first = next((i for i, d in enumerate(accrual_starts) if not d < exercise_date), len(accrual_starts))
    return max(first - 1, 0)


def wrap(option: jax.Array, underlyings: Sequence[jax.Array], steps: Sequence[Optional[int]],
         physical: bool) -> jax.Array:
    """`[S, D]` NPV of the wrapped option from the option's own cube `option` and each
    exercise's underlying cube (`underlyings[i]` for contract date i), as `OptionWrapper::NPV`
    walks the grid. A long position."""
    num_dates = option.shape[1]
    after = jnp.arange(num_dates)
    value = option
    exercised = jnp.zeros(option.shape[0], dtype=bool)
    seen = set()
    for underlying, step in zip(underlyings, steps):
        if step is None or step in seen:
            continue
        seen.add(step)
        exercise = ~exercised & (underlying[:, step] - option[:, step] > 0.0)
        settled = underlying if physical else jnp.zeros_like(underlying)
        value = jnp.where(exercise[:, None] & (after[None, :] >= step), settled, value)
        exercised = exercised | exercise
    return value
