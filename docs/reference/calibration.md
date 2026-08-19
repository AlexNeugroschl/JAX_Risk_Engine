# Calibration: Fitting LGM's Sigma to Market Swaption Volatilities

**Module:** [`engine/calibration/`](../../engine/calibration) —
[`basket.py`](../../engine/calibration/basket.py),
[`lgm.py`](../../engine/calibration/lgm.py)

## Plain-language summary

Every pricer in this codebase up to this point took its interest-rate-model volatility
(`hw_sigma`) directly as a config input — a number the caller just... supplies. A real
trading desk doesn't do that: it starts from prices quoted in the market for simpler,
liquid options (European swaptions), and works backward to find the single-factor model
parameter that reproduces those prices — a process called **calibration**. Only then is
that fitted parameter used to price the more complex trade (a Bermudan swaption) that
doesn't trade liquidly enough to have its own quoted price.

This module is that missing step: given a handful of market-quoted European swaption
volatilities, it finds the piecewise-constant LGM volatility term structure that exactly
reprices every one of them. It exists specifically because
[Bermudan/American Vega](../risk/greeks.md#vega-bermudanamerican-only) has no meaning
without it — "how much does this Bermudan's value change if a market quote moves" is only
a well-posed question once there's an actual market-quote-to-model relationship to
differentiate.

## Why bootstrap, not joint least-squares

ORE supports two calibration strategies (`ore::data::LgmBuilder`'s `CalibrationType`
enum): `Bootstrap` and `BestFit`. This module implements `Bootstrap` exclusively, matching
`LgmBuilder::calibrate()`'s default path
(`OREData/ored/model/lgmbuilder.cpp`, lines 209-212):

```cpp
if (calibrationType == CalibrationType::Bootstrap) {
    ...
    lgmModel->calibrateVolatilitiesIterative(basket_, *optimizationMethod_, endCriteria_);
```

`calibrateVolatilitiesIterative` (`QuantLib::CalibratedModel::calibrateIterative`) doesn't
run one joint multi-parameter optimization over the whole basket at once — it calibrates
one instrument at a time, in sequence, each time solving for exactly one new free
parameter while holding every earlier one fixed. This module's own bootstrap is a direct
JAX-native analogue of that same per-instrument sequence, not a simplification of it: the
underlying construction makes this the *mathematically correct* thing to do, not just a
convenient shortcut, because of how ORE sets up the piecewise sigma's bucket breakpoints.

### The `aTimes = swaptionExpiries[:-1]` convention, and why it makes calibration triangular

`LgmBuilder::initParametrization` sets the piecewise sigma's bucket breakpoints (`aTimes`)
to the calibration basket's own swaption expiry times, dropping the basket's *last*
expiry: `N` expiries produce `N-1` interior breakpoints for `N` buckets — exactly
`engine.models.lgm.Sigma`'s own `len(values) == len(times) + 1` invariant (see
[Models & Trades](models-and-trades.md#sigma-a-piecewise-constant-volatility-term-structure)).

This single convention is what makes the whole calibration triangular, and therefore
solvable one root-find at a time: the `i`-th co-terminal swaption's price depends on
`zeta(t)` only up to its own expiry `T0_i` (`zeta` is a cumulative integral —
see `engine.models.lgm.zeta`), which in turn depends **only** on sigma buckets `0..i` —
buckets `i+1..N-1` cover times strictly after `T0_i`, so they contribute nothing to
`zeta(T0_i)` no matter what value they eventually take. Each new basket instrument
therefore pins down exactly one new free sigma value, holding every earlier bucket fixed
at whatever it was already calibrated to. This is precisely what
`calibrateVolatilitiesIterative` does, called per-instrument in basket order — reading it
this way confirms the bootstrap isn't a simplified stand-in for `BestFit`, it's ORE's own
default algorithm for exactly this reason.

`engine.calibration.lgm.calibrate_lgm_sigma` (`engine/calibration/lgm.py`) implements this
directly: for each target `i` (targets must be supplied in increasing expiry order,
asserted explicitly), the bucket breakpoints so far are `targets[0].expiry_time, ...,
targets[i-1].expiry_time`, and bucket `i`'s sigma value is the root of

```
price_lgm_swaption(Sigma(times=[T0_0..T0_{i-1}], values=[s_0..s_{i-1}, s_i]), targets[i])
    == bachelier_swaption_price(targets[i])
```

found via a 60-iteration scalar bisection (`_bisect_bucket_sigma`) over `sigma in [1e-6,
0.20]` — a fixed (not dynamically expanded) bracket, deliberately: a piecewise LGM sigma
bucket calibrating outside a 1bp-20% annual vol range indicates a misconfigured basket
(e.g. a market vol far outside realistic rates levels), not a case this bootstrap should
silently paper over the way `european_swaption._solve_rstar`'s dynamically-expanding
bracket does for its own, different reason (see
[ORE Parity](ore-parity.md#6-european-swaption-pricing-jamshidians-decomposition)).

An exact bootstrap reprices every basket instrument exactly (RMSE ~`1e-10`, limited only
by the bisection's own root-find tolerance) — unlike a joint least-squares/`BestFit`
calibration, whose RMSE is generally nonzero even at convergence, since it's trading off
fit quality across every instrument simultaneously rather than fitting each one exactly in
turn.

### Why a joint optimizer was never built

`engine/calibration/lgm.py`'s own module docstring notes a general Levenberg-Marquardt
joint optimizer was considered during this module's design but turned out unnecessary: the
bootstrap is a sequence of independent 1D root-finds, each with a closed, monotonic
relationship between one sigma bucket and one target price (see "Monotonicity" below), not
a joint multi-parameter fit. A JAX-native LM implementation would only be needed for an
ORE-style `BestFit` calibration mode — genuinely fitting several sigma buckets (or sigma
and mean reversion together) jointly against a basket where no exact reprice is expected —
which is out of this module's scope. No file exists for this; it's a documented "not
needed here," not a stub or an unfinished component.

### Why mean reversion is never calibrated

`a` (mean reversion / kappa) is always an input to `calibrate_lgm_sigma`, never an output.
This matches ORE's own default: `LgmData::calibrateH() == false` unless a trade
configuration explicitly opts in, meaning ORE itself calibrates only the piecewise sigma
term structure by default and treats mean reversion as a fixed, separately-chosen
parameter. This isn't an arbitrary scope cut mirroring ORE's default for its own sake —
the one-parameter-per-instrument bootstrap this module implements structurally requires
it: jointly calibrating both `a` and `sigma` from the same basket is a fundamentally
different, non-bootstrap problem (ORE's own `calibrate()` routes that case to `BestFit`,
not `Bootstrap`, precisely because a joint `(a, sigma...)` fit has no natural triangular
ordering the way sigma-only calibration does). Supporting it would mean building the joint
optimizer described above, which this module deliberately does not do.

## The co-terminal calibration basket

**This engine:** `engine.calibration.basket.build_coterminal_basket`.

**ORE:** `ore::data::IrModelBuilder::buildSwaptionBasket()`
(`OREData/ored/model/irmodelbuilder.cpp`).

Builds one `CalibrationTarget` (one European swaption) per Bermudan/American exercise
date, each into a swap that matures on the **same final date** as the trade being
calibrated for — ORE's own "diagonal"/co-terminal basket convention (as opposed to a
"co-initial" basket, which ORE also supports but does not use by default). An exercise
schedule `[1Y, 2Y, 3Y, 4Y]` against a trade maturing in `5Y` produces basket instruments
`1Yx4Y, 2Yx3Y, 3Yx2Y, 4Yx1Y` (expiry x tenor-to-the-shared-final-maturity).

This is the natural choice, not an arbitrary one: exercising a Bermudan at `t_i` always
means entering the *same* underlying swap that runs to the trade's own final maturity,
just with a shorter remaining tenor — a co-terminal basket calibrates the model against
instruments that are economically the closest available liquid proxy for each of the
Bermudan's own exercise decisions.

Each underlying swap is struck **at-the-money** — its own par rate under the supplied
`ZeroCurve`, computed from the swap's own annuity/discount factors (the standard
par-swap-rate identity: `par_rate = (P(0,T_start) - P(0,T_end)) / annuity`) — matching
`ore::data::IrModelBuilder::getStrike` returning `Null<Real>()`, ORE's own sentinel for
"use the ATM strike," whenever no explicit strike is configured (the common case this
module targets). Both the swap schedule and the par rate are computed from a real
`ORE.VanillaSwap`/`ORE.MakeVanillaSwap` build (via
[`engine.trades.ore_builders`](models-and-trades.md#enginetradesore_builderspy)) against
the *caller-supplied* `ZeroCurve`, not ORE's own discount curve — deliberately, so
calibration prices consistently against the same curve `engine.models.lgm` itself
discounts with.

`build_coterminal_basket` does not fetch a live market volatility surface — `market_vols`
is a caller-supplied list, one normal (basis-point) volatility per exercise date, in the
same order as `exercise_times`. This module's normal-vol (Bachelier) convention matches
[Bermudan Vega](../risk/greeks.md#vega-bermudanamerican-only)'s own convention throughout.

## `price_lgm_swaption`: LGM's own closed-form European swaption pricer

**This engine:** `engine.calibration.basket.price_lgm_swaption`.

**ORE:** the LGM analogue of `QuantExt::AnalyticLgmSwaptionEngine` — Jamshidian's
bond-option decomposition expressed in LGM's own `H`/`zeta`/`x` state variables
(`engine.models.lgm`), rather than Hull-White's `A`/`B`/`r` state variables
(`engine.models.hull_white`, which `engine.instruments.european_swaption` uses — see
[Models & Trades](models-and-trades.md) for why these are not interchangeable for `t>0`).

This is a genuinely separate closed-form pricer from `european_swaption.py`'s own
Jamshidian decomposition, not a duplicate of it, for two reasons: (1) it prices under a
different model realization (LGM, not HW1F — see the divergence finding documented in
[ORE Parity](ore-parity.md#a-parametrization-note-lgm-vs-plain-hull-white) and
[Models & Trades](models-and-trades.md)), and (2) its whole purpose is to be evaluated
repeatedly with a *trial* `Sigma` during calibration — `european_swaption.py`'s HW1F-only
formulas only support a constant sigma (see
[Models & Trades](models-and-trades.md#enginemodelshull_whitepy)), which cannot express
the piecewise structure calibration itself is fitting.

**`QuantExt::AnalyticLgmSwaptionEngine` could not be used directly.** Its constructor is
not exposed through this codebase's installed ORE Python bindings — confirmed directly by
reading `ORE-SWIG/QuantExt-SWIG/SWIG/qle_pricingengines.i`, which declares only
`enableCache`/`clearCache`/`setZetaShift`/`resetZetaShift` for `ORE.AnalyticLgmSwaptionEngine`,
with no `%extend` constructor. Building an independent closed-form pricer directly against
`engine.models.lgm`'s own already-verified primitives was the only route available, which
is why this module's two-route verification (below) matters more here than it might
otherwise — there's no way to simply instantiate ORE's own engine and compare NPVs
directly, the way `european_swaption.py` compares against
`ORE.JamshidianSwaptionEngine`.

### Two-route verification

**Route 1: every individual formula piece, checked to machine precision.** `bond_price`,
`bond_option_sigma`, and `numeraire` (`engine.models.lgm`) are each independently
live-verified against `ORE.LinearGaussMarkovModel`'s own exposed methods
(`discountBond`, `numeraire`, `zeta`) — see
[Models & Trades](models-and-trades.md#enginemodelslgmpy) and
`tests/test_models_piecewise_sigma.py`. This confirms every building block
`price_lgm_swaption` is assembled from is individually correct, but not that the assembly
(Jamshidian's decomposition itself, applied to these particular building blocks) is.

**Route 2: the full price, checked against a numeraire-deflated Monte Carlo simulation.**
`price_lgm_swaption`'s complete output is cross-checked against a Monte Carlo simulation of
`x(T0) ~ N(0, zeta(T0))` — LGM's own exact terminal distribution (`x` is driftless, so its
distribution at any time is Gaussian with variance `zeta`, per
`QuantExt::IrLgm1fStateProcess::variance`) — pricing the same swaption's payoff at each
sampled `x(T0)` and averaging.

**The key requirement: LGM's own measure is not the T0-forward measure.** LGM's state
variable `x` is defined under LGM's own measure, not the `T0`-forward measure — a payoff
sampled from `x(T0)`'s own distribution must be deflated by the model's own numeraire,
`N(T0,x) = exp(0.5*H(T0)^2*zeta(T0) + H(T0)*x) / P(0,T0)` (`engine.models.lgm.numeraire`,
the same numeraire
[`bermudan_swaption.py`'s backward induction](../instruments/american-bermudan-swaptions.md#5-numeraire-deflation--the-step-that-makes-the-rollback-mathematically-valid)
deflates by, for exactly the same underlying reason), not simply discounted by `P(0,T0)`.
Dividing each sampled payoff by `N(T0,x)` matches the closed form to ~0.05% — well within
Monte Carlo standard error for `N=3,000,000` paths.

This is the same "raw values aren't martingales, only numeraire-deflated ones are" lesson
`bermudan_swaption.py`'s own backward induction ran into independently while first being
built (see
[American & Bermudan Swaptions](../instruments/american-bermudan-swaptions.md#5-numeraire-deflation--the-step-that-makes-the-rollback-mathematically-valid))
— surfacing twice, in two different pieces of this codebase's LGM machinery, from two
different kinds of mistake (a rollback recursion vs. a Monte Carlo discounting convention),
is a useful independent confirmation that it's a genuine property of the model, not a
one-off implementation slip.

### Monotonicity

`price_lgm_swaption` is monotonically increasing in the newest (most recently added)
sigma bucket's own value — a swaption is long volatility, and raising the newest bucket's
sigma strictly increases the model's total `zeta(T0_i)` for every `t` in that bucket. This
property (confirmed directly,
`tests/test_calibration_basket.py::TestPriceLgmSwaptionSanity::test_higher_sigma_gives_higher_price`)
is exactly what makes each bucket's calibration a simple, well-posed scalar bisection
rather than something needing a more general root-finder.

## The `_bisect_xstar` gradient bug

**This engine:** `engine.calibration.basket._bisect_xstar`/`_bisect_xstar_raw`.

`price_lgm_swaption` finds the exercise boundary — the state `x*` at which the signed
coupon bond (every fixed cashflow, the final notional, minus the notional received back at
the swap's own accrual start) is worth exactly `0` — via a 100-iteration bisection,
`_bisect_xstar_raw`. This is the LGM analogue of
`european_swaption._solve_rstar`'s own bisection for Jamshidian's critical short rate
`r*` (see
[ORE Parity](ore-parity.md#6-european-swaption-pricing-jamshidians-decomposition)), and it
had the identical bug, for the identical underlying reason.

**The bug.** `_bisect_xstar_raw`'s bisection loop narrows a bracket using
`jnp.where(val > 0.0, ...)` at every iteration. `val > 0.0`'s comparison has zero gradient
everywhere — differentiating straight through the unrolled loop with plain `jax.grad`
therefore captures only `price_lgm_swaption`'s *direct* dependence on `sigma` (through `K`,
`sigma_p`, and `P0_Ti`, each evaluated *at* a fixed `x*`), completely dropping the
*indirect* contribution `d(price)/d(x*) * d(x*)/d(sigma)` — the effect of `sigma` moving
`x*` itself, which then moves every downstream quantity evaluated at `x*`. This isn't a
precision shortfall; it silently omits an entire term of the true derivative, giving a
gradient that is **wrong**, not merely approximate.

Concretely: this produced a systematic ~6% error in `price_lgm_swaption`'s own `jax.grad`
with respect to `sigma`, caught by
[`engine.risk.greeks.bermudan_vega`](../risk/greeks.md#vega-bermudanamerican-only)'s own
finite-difference cross-check against a literal recalibration (bump one market vol, rerun
`calibrate_lgm_sigma`, reprice) — not by any test inside this module itself, since the
*forward* value of `price_lgm_swaption` was always correct; only its gradient was wrong,
and nothing calling this module for a forward price alone (i.e. `calibrate_lgm_sigma`
itself, whose bisections never need `price_lgm_swaption`'s gradient) could have noticed.

**The fix.** `_bisect_xstar` wraps `_bisect_xstar_raw` with the same
[implicit function theorem](https://en.wikipedia.org/wiki/Implicit_function_theorem)
correction `european_swaption._solve_rstar` already uses (see
[Delta, Gamma, and Theta](../risk/greeks.md#two-real-bugs-this-module-found-and-fixed)) —
a `jax.custom_jvp` implementing, at a root of `f(x*, params) = 0`:

```
dx*/dparams · v = -(df/dparams · v) / (df/dx)
```

for any tangent direction `v`, computed via one `jax.grad` (for `df/dx`, at the
stop-gradient'd converged root) and one `jax.jvp` (for the directional derivative
`df/dparams · v`) — cheap relative to the 100-iteration bisection itself, and exact rather
than approximate. This requires `coupon_bond_value_fn`'s parameters to be passed as an
explicit pytree (`params = (a, sigma)`) rather than only captured in a Python closure,
since `jax.custom_jvp` needs an explicit primal argument to attach a JVP rule to — the
same structural requirement `_solve_rstar`'s own docstring explains.

**A second, related fix this bug's investigation surfaced: `Sigma` needed pytree
registration.** The implicit-function-theorem correction above only works if a tangent can
actually reach `Sigma.values` when `Sigma` is nested inside a larger `params` tuple. Before
`engine.models.lgm.Sigma` was registered with `@jax.tree_util.register_pytree_node_class`,
an unregistered dataclass was treated by `jax.tree_util` as an opaque leaf — `jax.jvp`
would differentiate `price_lgm_swaption` with respect to `sigma` as a single indivisible
object, contributing zero tangent through its `.values` field, silently truncating the
correction term above even after `_bisect_xstar`'s own `custom_jvp` was in place. Both
fixes were required together; either alone still produced a systematically wrong gradient.
See [Models & Trades](models-and-trades.md#the-pytree-registration-bug) for the full
account of this second bug, which is really about `Sigma`'s own JAX integration rather
than about calibration specifically, but was only ever exposed by this calibration code
path — nothing outside `engine/calibration/` passes a `Sigma` inside a larger
differentiated pytree.

Both bugs are fixed now, cross-checked to within ~0.005%-0.03% of a literal
finite-difference recalibration — see
`tests/test_calibration_basket.py`'s gradient-correctness tests and
`tests/test_greeks_bermudan.py::TestBermudanVega`.

## Tested by

- `tests/test_calibration_basket.py` (16 tests) — `build_coterminal_basket`'s schedule/par-
  rate construction, `price_lgm_swaption`'s two-route verification described above (formula
  pieces against live `ORE.LinearGaussMarkovModel` objects, full price against the
  numeraire-deflated Monte Carlo), monotonicity, and the `_bisect_xstar` gradient-
  correctness regression tests (value-level, not just sign/finiteness, cross-checks against
  finite difference).
- `tests/test_calibration_lgm.py` (9 tests) — `calibrate_lgm_sigma`'s bootstrap: exact
  reprice of every basket instrument (RMSE ~`1e-10`), the triangular
  `aTimes = swaptionExpiries[:-1]` bucket construction, ordering assertions, and
  `CalibrationResult`'s diagnostic fields.
- `tests/test_calibration_integration.py` (3 tests) — end-to-end: build a basket, calibrate
  a `Sigma`, feed it into `BermudanSwaptionConfig.hw_sigma`, price the Bermudan, confirming
  the calibrated `Sigma` behaves correctly as a drop-in replacement for a flat scalar
  throughout the full pricing pipeline.
- `tests/test_greeks_bermudan.py::TestBermudanVega` — the `_bisect_xstar` fix's
  real-world consequence: Vega matches a literal finite-difference recalibration to within
  ~0.005% for every bucket in a 4-instrument basket (see
  [Delta, Gamma, and Theta](../risk/greeks.md#vega-bermudanamerican-only)).
