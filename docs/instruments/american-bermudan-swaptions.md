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

This module builds that backward-working grid, and it builds it the way ORE does, step for
step: at the same grid settings it returns the same numbers as ORE's own engine, to about
1e-11 (see [How it is checked against ORE](#how-it-is-checked-against-ore)).

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

## American exercise: ORE's grid, and ORE's broken periods

A continuous exercise window (any instant in `[t1, t2]`) can't be represented on a finite
computer, so ORE discretizes it. **ORE's own C++ source**,
`QuantExt::NumericLgmMultiLegOptionEngineBase::calculate()`
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

`americanExerciseTimeStepsPerYear_` is ORE's `ExerciseTimeStepsPerYear` model parameter
(its shipped example config uses `24`, roughly monthly). Note the `static_cast<Size>`: the
step count is **truncated**, not rounded. `AmericanSwaptionConfig.option_times()` reproduces
this line for line, including the truncation and the order of the floating-point operations.

**An American is not "a Bermudan with a lot of dates".** This page used to say it was, and
the engine priced it that way. ORE does not. The second difference is in which coupons an
exercise enters (`buildCashflowInfo`, lines 107-115):

```cpp
if (exerciseType == Exercise::American) {
    // american exercise implies that we can exercise into broken periods
    info.belongsToUnderlyingMaxTime_ = timeFromReference(cpn->accrualEndDate());
} else {
    // bermudan exercise implies that we always exercise into whole periods
    info.belongsToUnderlyingMaxTime_ = timeFromReference(
        midCouponExercise ? noticeCalendar.advance(cpn->accrualEndDate(), -noticePeriod, ...)
                          : cpn->accrualStartDate());
}
```

and each coupon is credited `couponRatio(t) = clamp((accrualEnd − t − lag) / (accrualEnd −
accrualStart), 0, 1)` of its value (`lag` is the notice period, zero here). So an American
exercised in the middle of a period enters that broken period and is credited the unexpired
share of its coupons; a Bermudan exercised on the same day enters the next whole period.
Treating the American as a Bermudan dropped the broken coupon, which overstated payers by up
to 6x and understated receivers down to 0.09x against ORE ([I-06](../known-issues.md#i-06)).
The engine now carries the rule as `ExerciseStyle` and applies ORE's `couponRatio`.

Both `BermudanSwaptionConfig` and `AmericanSwaptionConfig` are priced by the same functions
(`prepare_bermudan`, `price_bermudan_swaption_base`, `price_bermudan_swaptions`); there is no
conversion from one to the other. Each config supplies its own option times and exercise
style.

## The pipeline, step by step

### 1. Describing a trade: `BermudanSwaptionConfig` / `AmericanSwaptionConfig`

Exercise is specified in **dates**, as in ORE: `BermudanSwaptionConfig.exercise_dates` (a
list of `ORE.Date`), `AmericanSwaptionConfig.first_exercise_date`/`last_exercise_date` plus
`exercise_time_steps_per_year`. Over HTTP these are ISO date strings. Times are derived from
the dates with the curve's own day counter (`time_from_reference`), exactly as ORE derives
`optionTimes`, so an exercise date equal to an accrual date maps to the bit-identical time.
Bermudan dates on or before the evaluation date are not exercise opportunities (ORE: `if (d >
refDate)`); an American window that is already open starts at `t = 0`.

Any exercise date is legitimate. A Bermudan date inside an accrual period exercises into the
next whole period, as in ORE (see [Which coupons an exercise enters](#which-coupons-an-exercise-enters)).
`exercisable_dates(cfg)` lists the underlying's own accrual start dates, the exercise dates
of a standard coterminal Bermudan.

`hw_sigma` is typed `Union[float, engine.models.lgm.Sigma]` on both configs: a flat scalar
volatility, or a calibrated piecewise `Sigma` term structure produced by
`engine.calibration.lgm.calibrate_lgm_sigma` (see [Calibration](../reference/calibration.md)).
No code in this module branches on which case it received: every downstream formula calls
`engine.models.lgm.zeta(sigma, t)`, which handles both via `as_sigma`'s upgrade of a bare
float to a one-bucket `Sigma` (see
[Models & Trades](../reference/models-and-trades.md#sigma-a-piecewise-constant-volatility-term-structure)).
A piecewise `Sigma` corresponds exactly to ORE's `VolatilityTimes`/`Volatility` parameters.

### 2. Building the trade and extracting cashflows: `prepare_bermudan()`

Unlike the European module, Jamshidian's telescoping-notional shortcut for the floating leg
(see [European Swaptions](european-swaptions.md#5-why-t_start-matters-the-floating-legs-notional-timing))
cannot be used here: early exercise means the value at each node needs the *actual*
remaining swap, not an identity valid only for the full, unexercised swap.
`prepare_bermudan()` therefore resolves, per coupon, everything ORE's `CashflowInfo` holds:
pay time, accrual start and end, the time the coupon stops belonging to the exercised-into
swap (set by the exercise style), and for a floating coupon the **index's own fixing
period** `[valueDate(fixingDate), maturityDate(valueDate)]` and its day count fraction. All of
it is read from the real ORE trade, the same `MakeVanillaSwap`-built swap `swap.py` and
`european_swaption.py` use.

The index period matters because ORE's LGM engine projects an Ibor rate over it
(`LgmVectorised::fixing`), not over the coupon's accrual period. The two usually coincide,
but not always: the schedule is generated backward from maturity, while an index period is
rolled forward from its own start date, and they can end a business day apart. Projecting
over the accrual period, as this engine once did, was worth about 2e-4 of the price on
ordinary trades ([I-31](../known-issues.md#i-31)).

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
`LgmConvolutionSolver2`'s constructor term-for-term, including its boundary special case
(ORE gives the first and last node the same weight, computed from `y_0`), its clamping of
a rounding-negative weight to zero, and its grid size `floor(sx·nx)` points either side of
zero.
This closed-form weight vector was independently verified (not just transcribed) by
checking it against known Gaussian expectation identities — `E[X]=0`, `E[X^2]=1`, and
`E[max(X-k,0)]` matching the standard normal's known closed form — before it was ever used
in the pricer itself.

### The `_state_grid`/`sqrt` gradient bug

`_state_grid(sigma, t, n_per_std, std_devs)` and the backward induction's own per-step
`std_step` computation (`_run_backward_induction`) both compute `sqrt(zeta(...))` — the
grid spacing/transition standard deviation is, by definition, the square root of a
variance. `sqrt`'s own derivative, `1/(2*sqrt(x))`, is a `0/0` indeterminate form exactly
at `x=0` — which genuinely happens at `t=0`, where `zeta(0)=0` by construction (see
`engine.models.lgm.zeta`). The *forward* value was always correct (`sqrt(0) = 0` is
perfectly well-defined), but `jax.grad`/`jax.hessian` through either function produced
`NaN`, since JAX still backpropagates through both branches of any computation that feeds
into a value used downstream, even a value that is numerically `0`.

This was found while building Bermudan Greeks (see
[Delta, Gamma, and Theta](../risk/greeks.md)): `d(NPV)/d(sigma_values[0])` came back `NaN`
until this guard was added, the first caller in this codebase to differentiate through
`_state_grid` at all — no forward-only pricing call had ever needed a gradient through it
before. Fixed via the same standard branch-free `jnp.where` pattern already used throughout
`engine.models.hull_white`/`engine.models.lgm` for their own removable singularities
(evaluate `sqrt` on a safe placeholder that is never actually `0`, select the correct
branch with `jnp.where`, discard the placeholder) — the same general pattern, just applied
to a different singularity (`sqrt` at `0`, rather than division by `a` at `0`).

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

### 6. Each cashflow's value: `_cashflow_values_at_nodes`

ORE's `CashflowInfo::pv` calculators, at every grid node, signed by the holder's side of each
leg:

- fixed coupon: `amount · P(t, pay; x)`;
- Ibor coupon: `(fixing(t, x) + spread) · accrual · notional · P(t, pay; x)`, with
  `fixing(t, x) = (P(t,T1)/P(t,T2) − 1) / dcf(d1, d2)`, `T1 = max(t, d1)`,
  `T2 = max(T1, d2)` over the index period `[d1, d2]`. A fixing dated on the evaluation date
  is deterministic in ORE (`index->fixing(today)`, forecast off today's curve), and is so here.

Once `t` is past `d1` the clamp projects only the remaining stub, which `couponRatio` then
scales again. As read, that shortens a broken American floating coupon twice. It is ORE's
behaviour and is kept deliberately.

### 7. Backward induction: ORE's own loop, `_backward_induction_arrays`

The grid times are `{0} ∪ optionTimes ∪ condition_times`, deduplicated exactly (ORE's
`std::set<Real> timeGrid`, never rounded, so an option time stays bit-identical to the
accrual time it names). Walking them from the latest to `t = 0`, at each time, exactly as
`NumericLgmMultiLegOptionEngineBase::calculate()` does (lines 543-619):

1. Roll the carried option value, the carried `underlyingNpv` and each cached cashflow back
   from the previous (later) grid time via the convolution. ORE's rollback is a no-op
   between times that are `close_enough`, and so is this one.
2. Apply ORE's cashflow bookkeeping. A cashflow that still belongs to the underlying is:
   added to `underlyingNpv` at the latest grid time its amount can be estimated (**Done**);
   or, if the exercise would enter it mid-period, cached and credited
   `cache · couponRatio(t)` (**Cached**), moving into `underlyingNpv` once it is whole
   again; or, for an Ibor coupon past its fixing, valued afresh at each time and credited
   `pv · couponRatio(t)`.
3. At an option time, `option = max(option, underlyingNpv + provisionalNpv +
   provisionalNpvNonCached)` — ORE's exercise rule, in numeraire-deflated units.

Every branch of step 2 depends only on the grid time and the cashflow's own times, never on
the model state, so the whole Open → Cached → Done history is computed once in Python
(`_GridSchedule`) and replayed inside one `jax.lax.scan` as 0/1 masks.

**Why replay ORE's loop, rather than evaluate the exercise value in closed form?** Both have
the same mathematical limit, because rolling a cashflow's value back is exactly its
conditional expectation. But they differ numerically: at a 48-point grid the closed form
differs from ORE by up to ~1e-4 on an American, and the gap shrinks about 4x per grid
doubling. The replayed loop matches ORE at any grid, to ~1e-11. The whole point of this
module is to produce ORE's numbers, so it reproduces ORE's algorithm, not only ORE's
mathematics.

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

## Delta, Gamma, Theta, and Vega

Bermudan/American swaptions have full Greeks support — `engine.risk.greeks.
bermudan_delta_gamma`, `bermudan_theta`, and `bermudan_vega`. `_run_backward_induction` is
implemented via `jax.lax.scan`, so it is differentiable end-to-end, exactly like the swap
and European swaption pricers, and `bermudan_delta_gamma`/`bermudan_theta` follow the
identical pattern/units as their swap/European counterparts (per-pillar $-per-1bp
Delta/Gamma, a 1-day-repricing-difference Theta).

Vega required a second prerequisite beyond JAX-nativeness: a real market-vol-to-model
relationship, supplied by [`engine/calibration/`](../reference/calibration.md).
`bermudan_vega` differentiates `d(NPV)/d(market_vol_i)` for each basket instrument through
`calibrate_lgm_sigma`'s own bootstrap via the implicit function theorem, rather than
literally re-running calibration once per bumped market vol. See
[Delta, Gamma, and Theta: Vega](../risk/greeks.md#vega-bermudanamerican-only) for the full
derivation, including two real bugs (a missing cross-bucket Jacobian term in `bermudan_
vega` itself, and the `_bisect_xstar` gradient bug documented in
[Calibration](../reference/calibration.md#the-_bisect_xstar-gradient-bug)) found and fixed
while building it.

American swaptions have no separate Greeks function: `bermudan_delta_gamma`,
`bermudan_theta` and `bermudan_vega` take an `AmericanSwaptionConfig` directly, since both
configs run through the same backward induction. Theta reprices the same trade one day on:
its exercise *dates* stay put and every time is re-derived from the new evaluation date, as
ORE does. (While exercise was given in year fractions, Theta silently moved every exercise
opportunity a day later too.)

## Which coupons an exercise enters

| Exercise | A coupon belongs to the swap entered while | Credited |
|---|---|---|
| Bermudan (ORE's default) | `t ≤ accrualStart` | the whole coupon |
| American | `t ≤ accrualEnd` | `couponRatio(t)` of it |

For a Bermudan exercise date inside an accrual period this means the holder enters **each
leg from its own next accrual start**: on an annual-fixed, semi-annual-floating swap, one
annual fixed coupon drops out while the second semi-annual floating coupon of that year stays
in. That is ORE's contract, not an approximation, and it moves the price in the direction of
the trade: up for a payer (a payment dropped), down for a receiver (a receipt dropped).

> **Corrected twice.** Until 2026-09-18 this page called that difference a "conservative"
> understatement; measured, it is not, since for a payer it is an overstatement (up to
> 7.4x against the aligned price). The 2026-09-18 correction then called it a mispricing to be
> fixed by prorating Bermudan coupons. Against ORE's own engine (2026-09-23) it is neither:
> it is what ORE prices, and prorating would have moved the engine away from ORE. The
> mispricing that did exist was the **American**, which the engine used to price with the
> Bermudan rule ([I-06](../known-issues.md#i-06)).

ORE also supports `midCouponExercise=true` Bermudans (coupons belong until `accrualEnd −
noticePeriod`, credited `couponRatio`), and notice periods generally. Neither is exposed by
these configs; the coupon model handles them by construction if they ever are.

## Exercise is specified by date

Exercise used to be given as year fractions, and the coupon-membership test was a float
comparison. A **rounded literal** was dangerous out of all proportion to the rounding:
`2.0137` for a true accrual start of `2.0136986301369864` landed 1.4e-6 after it, the coupon
starting that day read as already elapsed, and the zero-vol price came out 14,336.12 instead
of about 1,211 — a ~12x overstatement, silent and finite
([I-29](../known-issues.md#i-29)). The fix at the time snapped near-misses onto the schedule
within a tolerance band, and exempted Americans from it with a flag.

Both are gone. Exercise is now given in dates, as ORE takes it. An exercise date equal to an
accrual date maps to the identical time, so there is nothing to snap and no band to get
wrong. Coupon membership then uses QuantLib's own `close_enough`, as ORE does. A year
fraction passed as an exercise date is refused with a `TypeError`.

## How it is checked against ORE

ORE's `NumericLgmMultiLegOptionEngine` has no bound constructor in the Python bindings, so it
can't be built directly. `engine/validation/ore_lgm_oracle.py` reaches it by the route ORE users take: an
in-process `OREApp` run of the `NPV` analytic over a trade XML, through
`LGMGridSwaptionEngineBuilder`. It uses the engine's own underlying (explicit schedule dates),
a convention-defined copy of its `SimIndex`, its zero curve (date-quoted, linear in zero
rate), and its LGM parameters with calibration off. Two inputs ORE requires but never reads
for pricing are supplied only so the model builder does not fall back to dummies. One of
them, the swap index, is **not** inert: the LGM's own term structure is taken from its
discounting curve, so it is mapped to the engine's curve.

`tests/test_ore_lgm_parity.py` then requires equality to 1e-10 relative across aligned and
mid-period Bermudans, Americans (including high strike at low vol, and a window whose step
count truncates), piecewise volatility and the zero-vol limit. Measured worst case: 8.7e-12.
Before the changes on this page, 22 of those 23 cases failed.

**That parity is with ORE's Grid solver at `ShiftHorizon=0`**, the configuration this engine
reproduces. Under ORE's own defaults the engine is close but not identical: `ShiftHorizon=0.5`
(ORE's builder default) moves an American by up to 1.5e-4, and ORE's FD solver (used in its
shipped American config) by up to 1.6e-3. See [I-32](../known-issues.md#i-32). The oracle
takes both settings as parameters, so the gap can be re-measured at any time.

## Tested by

`tests/test_ore_lgm_parity.py` (23 tests) — equality with ORE's own LGM engine, as above.
The authoritative check.

`tests/test_bermudan_swaption.py` (63 tests) — the shared engine:

- `TestLgmClosedFormsAgainstORE` — every closed-form primitive (`H`, `zeta`, `_lgm_bond`,
  the Hagan quadrature weights) checked directly against live `ORE.IrLgm1fConstantParametrization`
  / `ORE.LinearGaussMarkovModel` objects, plus an explicit regression test documenting the
  `HullWhite` vs. `LinearGaussMarkovModel` divergence for `t>0` described above.
- `TestSingleExerciseMatchesDirectIntegration` — with one exercise date the value is a
  single Gaussian expectation, computed independently of the grid and the bookkeeping by
  `tests/bermudan_references.py`; agreement to 2e-5, plus a grid-convergence check. (This
  replaced a Jamshidian decomposition, which needs the floating leg to telescope and so is
  invalid once coupons are projected over their index periods.)
- `TestMonotonicity` — model-independent no-arbitrage bounds.
- `TestMidPeriodBermudanExercise` — the whole-period membership rule and its direction.
- `TestPortfolioAndShape`, `TestEdgeCases`, `TestBermudanSwaptionConfigValidation` —
  shape/portfolio correctness, negative rates, zero notional, and date validation.

`tests/test_american_swaption.py` (24 tests) — the American-specific parts: ORE's option
times (including the truncating step count), the broken-period `couponRatio`, and the one
case where the two styles must coincide exactly (a single exercise on an accrual start).

`tests/test_ore_bermudan_oracle.py` (51 tests) — against QuantLib's Hull-White tree and FD
engines, a model-level comparison of a few percent, plus `TestExerciseDatesAreExact`.

`tests/test_greeks_bermudan.py` (26 tests) — Delta/Gamma/Theta/Vega for this module's own
pricer, via `engine.risk.greeks`: see
[Delta, Gamma, and Theta: Tested by](../risk/greeks.md#tested-by) for the full breakdown,
including the Gamma finite-difference methodology (finite-differencing the *gradient*
rather than the price, since a naive price-level central difference is numerically
unreliable for this pricer at a realistic bump size) and the Vega cross-check against a
literal finite-difference recalibration.

`tests/test_calibration_integration.py` (6 tests) — a calibrated `Sigma` from
`engine.calibration.lgm.calibrate_lgm_sigma` fed into `BermudanSwaptionConfig.hw_sigma`
and priced end-to-end through this module, confirming the `Union[float, Sigma]` interface
works identically to a flat scalar throughout the full pipeline.
