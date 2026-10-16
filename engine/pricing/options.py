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
  * The position (`OptionData/LongShort`) multiplies the whole value, the option's and, once
    exercised, the underlying's (`multiplier2_`: 1 long, -1 short), but not the exercise
    decision, which compares the long option with its underlying: the holder exercises, so a
    short option is minus the long one on every path and date (I-101).

The engines (`engine.pricing.european`, `jamshidian`, `bermudan`) value the long option, as
ORE's pricing engines know nothing of the position; `position_sign` is applied where the trade's
value is formed, as ORE's wrapper applies it: `wrap` on the paths, and today
`engine.pricing.cube.value_on` and the price functions of the Greeks and market risk
(`engine.risk.greeks.price_functions`).
"""
from functools import partial
from typing import List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import ORE


def position_sign(cfg) -> float:
    """An option's position as ORE's `OptionWrapper` multiplies by it: 1.0 for `Long`, -1.0
    for `Short` (`long_short`)."""
    return -1.0 if cfg.long_short == "Short" else 1.0


@jax.jit
def bound_option_pnl(pnl, base, sign):
    """Keep a standalone option's signed value within its bound (I-113).

    Project the change directly: reconstructing and subtracting a large base would
    lose small moves. Used before storage and after decoding, which can cross the
    bound again. This is never applied to a delivered physical underlying.
    """
    base, sign = jnp.asarray(base, dtype=pnl.dtype), jnp.asarray(sign, dtype=pnl.dtype)
    return sign * jnp.maximum(sign * pnl, -sign * base)


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
         physical: bool, sign: float) -> Tuple[jax.Array, jax.Array]:
    """`(value, exercised)`: the `[S, D]` NPV of the wrapped option from the long option's own
    cube `option` and each exercise's underlying cube (`underlyings[i]` for contract date i), as
    `OptionWrapper::NPV` walks the grid: the holder's exercise, the position's `sign`
    (`position_sign`) on the value; and the wrapper's state on each path and date, `[S, D]` bool,
    whether the holder has exercised on or before the date (ORE's `exercised_`): its exercise
    decisions, which a precision policy can flip (a cash-settled option's flipped exercise moves
    it by its whole value; the precision harness measures the share each format flips, decision
    A-24). The sign is data, so a long and a short option share one program."""
    return _wrap(option, tuple(underlyings), tuple(steps), physical, jnp.asarray(sign, dtype=option.dtype))


@partial(jax.jit, static_argnums=(2, 3))
def _wrap(option: jax.Array, underlyings, steps, physical: bool, sign: jax.Array):
    # A reconstructed long continuation can cross zero by its rounding error (I-113).
    # Project before exercise and LongShort; delivered physical swaps retain their sign.
    option = jnp.maximum(option, 0.0)
    num_dates = option.shape[1]
    after = jnp.arange(num_dates)
    value = option
    exercised = jnp.zeros(option.shape[0], dtype=bool)
    state = jnp.zeros(option.shape, dtype=bool)
    seen = set()
    for underlying, step in zip(underlyings, steps):
        if step is None or step in seen:
            continue
        seen.add(step)
        exercise = ~exercised & (underlying[:, step] - option[:, step] > 0.0)
        from_now = exercise[:, None] & (after[None, :] >= step)
        settled = underlying if physical else jnp.zeros_like(underlying)
        value = jnp.where(from_now, settled, value)
        state = state | from_now
        exercised = exercised | exercise
    return sign * value, state
