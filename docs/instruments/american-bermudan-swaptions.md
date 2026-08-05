# Instruments: American & Bermudan Swaptions

## Plain-language summary

A **European swaption** (see [European Swaptions](european-swaptions.md)) gives its holder exactly
one date on which to decide whether to enter a swap. A **Bermudan swaption** gives the
holder several such dates — say, once a year for five years — and lets them pick the best
one. An **American swaption** goes further and lets the holder exercise on *any* day within
a window, not just a fixed list.

Having more chances to exercise makes the option strictly more valuable (or, at worst, no
less valuable) than having only one — an extra choice is never a disadvantage. But it also
makes the option much harder to price: with only one exercise date, there's a closed-form
shortcut (Jamshidian's trick). With several dates, the holder's decision at an early date
depends on how much the option would be worth if they *didn't* exercise and waited — which
itself depends on the model, not on any formula. There's no shortcut left; the only way to
price it correctly is to work backward through time on a grid of possible interest-rate
outcomes, comparing "exercise now" against "wait" at every point, exactly the way ORE's own
engine does it.

This module builds that backward-working grid.

## Why not Jamshidian's decomposition

[European Swaptions](european-swaptions.md#why-its-built-this-way-jamshidians-trick) explains
Jamshidian's trick: a European swaption is equivalent to an option on a coupon-bearing
bond, and that bond option can be split into a handful of independent zero-coupon bond
options, each of which has a textbook closed form under a one-factor model. This works
because the whole trick hinges on there being a single critical short rate `r*` at the
single exercise date, below (or above) which exercising is optimal.

Once there is more than one exercise date, that single critical rate no longer exists —
whether exercising at date 1 is optimal depends on what the option would be worth if held
until date 2, which itself depends on rates at date 2, which are still random as of date 1.
This is a genuinely different (numerically harder) problem, and reading ORE's own source
confirms it *is* solved differently there: `OREData/ored/portfolio/builders/swaption.hpp`
routes `EuropeanSwaption` trades to `BlackBachelierSwaptionEngine` (closed-form), but both
`BermudanSwaption` and `AmericanSwaption` trades to `LGMSwaptionEngineBuilder`, which
constructs `QuantExt::NumericLgmMultiLegOptionEngine` — a genuinely numeric backward-
induction engine. Neither `QuantLib::TreeSwaptionEngine` nor
`QuantLib::JamshidianSwaptionEngine` (plain QuantLib classes that *do* exist in the
`reference/ORE/QuantLib` submodule) are referenced anywhere in `OREData` or `QuantExt` —
confirmed by a repository-wide search — so ORE's own production code path for Bermudan and
American swaptions genuinely is this numeric engine, not a tree or Jamshidian's trick.

## American exercise is not priced continuously — in ORE either

A continuous exercise window (any instant in `[t1, t2]`) can't be represented on a finite
computer, so ORE — and this module, matching it exactly — discretizes it into a finite list
of exercise *dates*, then prices it as an (unusually fine) Bermudan swaption.

**ORE's own C++ source**, `QuantExt::NumericLgmMultiLegOptionEngineBase::calculate()`
(`QuantExt/qle/pricingengines/numericlgmmultilegoptionengine.cpp`, lines ~494-505):

```cpp
} else if (exercise_->type() == Exercise::American) {
    QL_REQUIRE(exercise_->dates().size() == 2, ...);
    Real t1 = std::max(0.0, ts->timeFromReference(exercise_->dates().front()));
    Real t2 = std::max(t1, ts->timeFromReference(exercise_->dates().back()));
    Size steps = std::max<Size>(1, static_cast<Size>((t2 - t1) * americanExerciseTimeStepsPerYear_));
    optionTimes.insert(t1);
    for (Size i = 0; i <= steps; ++i) {
        optionTimes.insert(t1 + static_cast<Real>(i) * (t2 - t1) / static_cast<Real>(steps));
    }
}
```

`americanExerciseTimeStepsPerYear_` is ORE's own `ExerciseTimeStepsPerYear` model
parameter (its own shipped example config, `Examples/Products/Input/pricingengine.xml`,
uses `24` — roughly monthly — for American swaptions). From this point on, the exercise
window is just a set of dates indistinguishable from a Bermudan's own explicit list — there
is no separate code path for American exercise beyond this discretization step.

`AmericanSwaptionConfig.to_bermudan()` (`engine/instruments/american_swaption.py`)
reproduces this exact construction: `steps = round((t2-t1) * exercise_time_steps_per_year)`,
then `steps+1` equally-spaced dates including both endpoints, expanded into a
`BermudanSwaptionConfig`. `price_american_swaptions` is a one-line wrapper around
`price_bermudan_swaptions` using that expansion — "American," in both ORE and here, really
does mean "Bermudan with a lot of dates."

## The pipeline, step by step

### 1. Describing a trade: `BermudanSwaptionConfig` / `AmericanSwaptionConfig`

`BermudanSwaptionConfig` takes an explicit `exercise_times` list (year-fractions from
`evaluation_date`); `AmericanSwaptionConfig` takes a `first_exercise`/`last_exercise`
window plus `exercise_time_steps_per_year`, and exposes `.to_bermudan()` to expand into the
former.

Both configs require `exercise_times` to coincide with the underlying swap's own reset
dates (see "Known limitation" below) — the standard "coterminal" structure ORE's own
calibration machinery (`SwaptionEngineBuilder::model()`) is itself built around for
Bermudan/American baskets.

### 2. Building the trade and extracting cashflows: `prepare_bermudan()`

Unlike the European module, Jamshidian's telescoping-notional shortcut for the floating leg
(see [European Swaptions](european-swaptions.md#5-why-t_start-matters-the-floating-legs-notional-timing))
cannot be used here: early exercise means the continuation value at each node needs the
*actual* remaining swap value, not just a closed-form identity valid only for the full,
unexercised swap. `prepare_bermudan()` therefore extracts every fixed and floating coupon's
full schedule (accrual start/end, payment date, amount/accrual fraction) from the real ORE
trade — the same `MakeVanillaSwap`-built swap `swap.py` and
`european_swaption.py` use, so date generation and day-count accrual again match ORE
exactly, not a reimplementation.

### 3. The model: LGM, not plain Hull-White — and why that distinction matters here

Every other pricer in this codebase (`simulation.py`, `swap.py`,
`european_swaption.py`) is built on this codebase's own direct short-rate closed form,
`compute_hw_A`/`_hw_B`, live-verified against `QuantLib::HullWhite` (see
[ORE Parity](../reference/ore-parity.md)). This module deliberately does **not** reuse that
formula, using instead a *separate* closed form, `_lgm_bond`, parametrized directly in
`QuantExt`'s own LGM state variable `x`:

```
P(t,T,x) = [P(0,T)/P(0,t)] * exp(-0.5*(H(T)^2 - H(t)^2)*zeta(t)) * exp(-(H(T)-H(t))*x)
```

with `H(t) = (1-exp(-a*t))/a` and `zeta(t) = sigma^2*t` — `QuantExt::LinearGaussMarkovModel::
discountBond` (`QuantExt/qle/models/lgm.hpp`, lines 252-280).

**This split exists because of a finding made while building this module, not stylistic
preference.** `ORE.HullWhite` (`QuantLib::HullWhite`) and `ORE.LinearGaussMarkovModel`
(`QuantExt::CrossAssetModel`'s own rates leg) were assumed, going into this task, to be two
equivalent parametrizations of the *same* model — [ORE Parity](../reference/ore-parity.md#a-parametrization-note-lgm-vs-plain-hull-white)
documents exactly that equivalence claim, verified at `t=0`. Building this module's
backward induction required evaluating both classes at `t>0`, and a live, direct comparison
showed they are **not** numerically the same model realization there:

```python
>>> hw.discountBond(t=3, T=5, r=0.03)       # r = f(0,t), HullWhite's own "no shock" point
0.9393234598794674
>>> lgm.discountBond(t=3, T=5, x=0.0)       # x = 0, LGM's own "no shock" point
0.9337296209777532
```

a genuine ~0.6% difference at `t=3y` (a=0.03, sigma=0.02) — not a rounding artifact. Both
classes were checked and are individually self-consistent affine short-rate models (each
satisfies its own `-d/dT log P(t,T)|_{T=t} == r` identity exactly, live-verified via finite
difference), and every individual building block along the way — `H(t)`, `zeta(t)`,
`H'(t)`, `f(0,t)`, the short-rate identity `r(t,x) = f(0,t) + x*H'(t) + zeta(t)*H'(t)*H(t)`
(itself confirmed via finite difference directly on `_lgm_bond`), and `A(t,T)`/`B(t,T)`
(confirmed exactly against `ORE.HullWhite.discountBond` for arbitrary `r`) — checked out
individually correct. The two models are simply calibrated/parametrized differently for
`t>0`, in a way this investigation did not fully resolve to a root cause but did concretely
measure and confirm is real, not a bug in either formula.

Since ORE's actual Bermudan/American engine is built on `LinearGaussMarkovModel`
(`NumericLgmMultiLegOptionEngine`'s constructor takes an `IrModel`, and
`LGMGridSwaptionEngineBuilder`/`LGMFDSwaptionEngineBuilder` both build an
`IrLgm1fConstantParametrization`/`LinearGaussMarkovModel`), **this module matches that
model exclusively** — `_lgm_bond` is used for every discount factor computed here;
`compute_hw_A`/`_hw_B` are never imported. `_lgm_bond` itself is live-verified to machine
precision (~1e-16 relative) against `ORE.LinearGaussMarkovModel.discountBond` directly
(`tests/test_bermudan_swaption.py::TestLgmClosedFormsAgainstORE`).

`r(t,x)` (`_r_from_x`) and its exact inverse `_x_from_r` are still used, but only to convert
between LGM's state `x` and the literal short rate `r` this codebase's Monte Carlo
simulation (`simulation.py`) produces directly — needed to condition the
backward-induction result on a simulated path, not to compute bond prices.

### 4. The state grid and Hagan's quadrature convolution

`QuantExt::LgmConvolutionSolver2` (`QuantExt/qle/models/lgmconvolutionsolver2.hpp/.cpp`,
citing Hagan's paper *"Methodology for callable swaps and Bermudan exercise into
swaptions"*) is ORE's own "Grid" backward-induction scheme (as opposed to the alternative
"FD" finite-difference solver, `LgmFdSolver` — both plug into the identical
`max(intrinsic, continuation)` loop, so choosing between them is a numerical-implementation
detail, not a modeling one; this module implements Grid, the simpler of the two to
reproduce exactly since it's a closed-form quadrature rather than a PDE scheme with its own
truncation error).

`_state_grid(sigma, t, n_per_std, std_devs)` builds a symmetric grid of `x` values at time
`t`, spaced `dx = sqrt(zeta(t))/n_per_std` apart, spanning `+/- std_devs` standard
deviations — exactly `LgmConvolutionSolver2::stateGrid`'s construction. At `t=0`, `zeta(0)=0`
and the grid collapses to the single point `x=0`, matching ORE's own `t=0` special case.

`_hagan_quadrature_weights(n_per_std, std_devs)` precomputes a fixed set of quadrature
weights on a standardized grid, derived from integrating a **piecewise-linear**
interpolation of the value function against the exact Gaussian transition density in
closed form (the "trapezoid-of-normal-density" weights in Hagan's paper) — reproducing
`LgmConvolutionSolver2`'s constructor term-for-term, including its boundary special cases
(the first/last node has only one neighbor, so the outer half-interval is treated as flat).
This closed-form weight vector was independently verified (not just transcribed) by
checking it against known Gaussian expectation identities — `E[X]=0`, `E[X^2]=1`, and
`E[max(X-k,0)]` matching the standard normal's known closed form — before it was ever used
in the pricer itself.

### 5. Numeraire deflation — the step that makes the rollback mathematically valid

`_rollback_one_step` computes `E[values(x_from) | x_to]` by convolving the quadrature
weights against a linearly-interpolated value function, using `x`'s own driftless Gaussian
transition law (`E[x_from | x_to] = x_to`, `Var[x_from | x_to] = zeta(t_from) - zeta(t_to)`
— LGM's state variable is driftless by construction, confirmed directly from
`QuantExt::IrLgm1fStateProcess::expectation()` returning its input unchanged; see
[ORE Parity section 3a](../reference/ore-parity.md#3a-short-rate-transition-monte-carlo-step)).

This convolution is only a valid way to compute a conditional expectation if what's being
rolled back is itself a **Q-martingale** under `x`'s own transition law. A raw bond or swap
price is *not* a martingale on its own — only the price **divided by the model's numeraire**
is (`N(t,x) = exp(0.5*H(t)^2*zeta(t) + H(t)*x) / P(0,t)`,
`QuantExt::LinearGaussMarkovModel::numeraire`). This was discovered directly, not assumed
from a textbook: an early version of this module rolled back *raw* (non-deflated) bond
prices and swap values, and a direct test comparing a rolled-back zero-coupon bond price
against the same bond's closed-form value at the target time showed a persistent,
non-shrinking (i.e. not a discretization-error) ~3% mismatch. Dividing every value by its
own time's numeraire before rolling back, and multiplying the result back by the target
time's numeraire afterward, reproduced the closed form to a genuine, grid-resolution-
shrinking numerical error instead. `_run_backward_induction` therefore runs entirely in
these "reduced" (numeraire-deflated) units — exactly matching
`QuantExt::LinearGaussMarkovModel::reducedDiscountBond`'s own reason for existing.

### 6. The underlying swap's own value: `_hw_swap_value_at_nodes`

At each grid node, the underlying swap's remaining value (fixed leg minus floating leg,
payer-signed, matching `swap.py`'s own sign convention) is computed directly
from the extracted cashflow schedule via `_lgm_bond`. A coupon is included only if its own
accrual has not yet started as of the evaluation time `t` (`start_time >= t`, using the same
`1e-9` tolerance throughout the module, including inside `_discount_at_nodes`'s own P(t,T,x)
computation — an earlier version of this function used a stricter, inconsistent zero-
tolerance comparison there, which silently zeroed out a coupon's own discount factor
whenever its start time landed *exactly* on the exercise time — the common case for a
reset-aligned exercise date, not an edge case — corrupting that coupon's forward rate; found
and fixed via `tests/test_bermudan_swaption.py`'s exact-reset-date test cases). This is
exact for any exercise/conditioning time that coincides with a reset date; see "Known
limitation" below for what happens otherwise.

### 7. Backward induction and early exercise: `_run_backward_induction`

Starting from the underlying's final maturity and walking backward to `t=0`, at each grid
time in the union of `{0, final_maturity} ∪ exercise_times ∪ condition_times`:

1. Roll the (deflated) value function back one step from the previous (later) grid time via
   the quadrature convolution.
2. If this time is an exercise date, compute the underlying swap's own (deflated) remaining
   value at every node and take `max(continuation, intrinsic)` — exactly
   `NumericLgmMultiLegOptionEngineBase::calculate()`'s rule:
   ```cpp
   optionNpv = max(optionNpv, underlyingNpv + provisionalNpv + ... + rebateNpv);
   ```
3. Re-inflate by the numeraire and, if this time was requested by the caller as a
   `condition_time` (for scenario-conditional pricing — see below), snapshot the (raw,
   re-inflated) value function, converted from `x` to `r` via `_r_from_x`, so it can later
   be interpolated against a simulated short rate directly.

At `t=0` the grid collapses to `x=0`, and reading off that single node gives the base-case
NPV — `price_bermudan_swaption_base`.

### 8. Conditional (scenario-cube) pricing: `price_bermudan_swaptions`

Unlike the European module, an American/Bermudan swaption's value at some future step
depends on its *entire remaining* exercise schedule — it cannot be evaluated at an
arbitrary future time from a single t=0 backward induction the way Jamshidian's closed form
can. `price_bermudan_swaptions` therefore re-runs the backward induction once per trade,
snapshotting the value function at every requested `step_time` before its own last exercise
date as the walk passes through, then interpolates each scenario's simulated short rate
(`hw_paths`) against the appropriate snapshot's `r`-grid — the same Markov-conditioning
principle [European Swaptions](european-swaptions.md#6-conditional-future-time-pricing) uses, applied
to a numerically-rolled-back value function instead of a closed form. Steps at or after a
trade's last exercise date are priced as exactly `0`, matching this codebase's (and ORE's
`Instrument.NPV()`'s) convention for an already-lapsed option.

## Known limitation: no mid-coupon proration

ORE's own American engine supports exercise landing *inside* an accrual period, prorating
that period's payment via a `couponRatio` computed from how far into the period the
exercise date falls (`belongsToUnderlyingMaxTime_` using `accrualEndDate()` specifically
for American exercise, `QuantExt/qle/pricingengines/numericlgmmultilegoptionengine.cpp`).

This module's `_hw_swap_value_at_nodes` uses a coarser rule instead: any coupon whose
accrual has already started by the exercise time is excluded **entirely**, not prorated.
This is exact whenever every exercise date coincides with a reset date (the scope
`BermudanSwaptionConfig`'s own docstring documents), but for a genuinely mid-coupon
American exercise date it forfeits the holder's entire already-accrued claim on that
period's payment rather than crediting a prorated share — a conservative (understating, not
overstating) approximation, verified not to produce nonsensical output
(`tests/test_bermudan_swaption.py::TestMidCouponKnownLimitation`), but a real, measurable
gap (observed to shift the priced value by an amount comparable to a full coupon's PV, not
merely a few days' accrual — this is NOT a small effect, and callers choosing
`exercise_time_steps_per_year` values that don't evenly divide the underlying's own reset
frequency should expect it). Choosing `exercise_time_steps_per_year` values that evenly
divide the reset frequency (verified concretely, not just estimated: for a semiannual-reset
swap, only step counts giving a `0.5`-year-multiple grid spacing land on true reset dates —
step counts as coarse as 2 avoid the limitation entirely, but the arithmetic doesn't scale
linearly with `exercise_time_steps_per_year`, so this must be checked per swap structure,
not assumed) avoids it entirely.

Full mid-coupon proration (matching ORE's own `couponRatio` construction) is intentionally
out of scope for this module, following this project's established pattern of documenting
known gaps as explicit, tested limitations rather than leaving them silent (see
`swap.py`'s own aged-swap limitation, [Instruments: Interest Rate Swaps](swaps.md)).

## Tested by

`tests/test_bermudan_swaption.py` — the underlying engine (`bermudan_swaption.py`),
which American exercise is priced through:

- `TestLgmClosedFormsAgainstORE` — every closed-form primitive (`H`, `zeta`, `_lgm_bond`,
  the Hagan quadrature weights) checked directly against live `ORE.IrLgm1fConstantParametrization`
  / `ORE.LinearGaussMarkovModel` objects, plus an explicit regression test documenting the
  `HullWhite` vs. `LinearGaussMarkovModel` divergence for `t>0` described above.
- `TestSingleExerciseMatchesLgmJamshidian` — the core correctness check: a Bermudan with
  exactly one exercise date must reproduce an independent from-scratch Jamshidian-style
  decomposition built on `_lgm_bond` (not `european_swaption.py`'s own HullWhite-
  parametrized Jamshidian pricer, confirmed to be a different model realization), across
  payer/receiver and several exercise dates, plus a grid-convergence check.
- `TestMonotonicity` — model-independent no-arbitrage bounds (more exercise dates, higher
  volatility, deeper ITM all strictly increase or preserve value).
- `TestPortfolioAndShape`, `TestEdgeCases`, `TestMidCouponKnownLimitation` — shape/portfolio
  correctness, negative rates, zero notional, and the documented mid-coupon behavior.

`tests/test_american_swaption.py` — the American-specific wrapper
(`american_swaption.py`):

- `TestAmericanAsFineBermudan` — the American-exercise discretization matches ORE's own
  construction exactly, converges as it's refined, and a reset-aligned American exactly
  reproduces the equivalent explicit Bermudan.
