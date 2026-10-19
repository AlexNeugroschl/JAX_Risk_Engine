# Calibration: Fitting LGM's Sigma to Market Swaption Volatilities

**Module:** [`engine/calibration/`](../../engine/calibration) —
[`lgm.py`](../../engine/calibration/lgm.py) (ORE's `LgmBuilder` bootstrap, every LGM calibration
the engine makes: each currency's in the cross-asset model and each Bermudan's/American's),
[`cam.py`](../../engine/calibration/cam.py) (the cross-asset model's, per currency); the root
solver, [`engine/solvers/roots.py`](../../engine/solvers/roots.py)

Until 2026-10-18 a second, standalone bootstrap (`calibrate_lgm_sigma`, with its own co-terminal
basket and Bachelier helper prices, behind `POST /calibration/lgm`) calibrated an LGM to a
caller-given curve and volatilities. It is retired (I-94): the run request's `calibration`
analytic returns what it returned, from ORE's bootstrap, which is compared with ORE itself.

## Plain-language summary

A model volatility could be taken directly as a configuration input — a number the caller
just... supplies (`calibration="None"` still does). A real
trading desk doesn't do that: it starts from prices quoted in the market for simpler,
liquid options (European swaptions), and works backward to find the single-factor model
parameter that reproduces those prices — a process called **calibration**. Only then is
that fitted parameter used to price the more complex trade (a Bermudan swaption) that
doesn't trade liquidly enough to have its own quoted price.

This module is that missing step: given a basket of market-quoted European swaption
volatilities, it finds the piecewise-constant LGM volatility term structure that reprices every
one of them, as ORE's `LgmBuilder` does. It exists specifically because
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
parameter while holding every earlier one fixed. `engine.calibration.lgm.bootstrap_sigma` is
the same per-instrument sequence in JAX, not a simplification of it: the underlying
construction makes this the *mathematically correct* thing to do, not just a convenient
shortcut, because of how ORE sets up the piecewise sigma's bucket breakpoints.

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

`engine.calibration.lgm.bootstrap_sigma` implements this directly: the helpers in increasing
expiry order, bucket `i` ending at helper `i`'s expiry (the last open), each bucket's volatility
the root, with the earlier buckets fixed, of the helper's model value less its market value
(below), found by the configured root solver ([below](#the-root-solver)) on `sigma in
[1e-6, 0.20]` — a fixed (not expanded) bracket, deliberately: a piecewise LGM sigma bucket
calibrating outside a 1bp-20% annual vol range indicates a misconfigured basket (e.g. a market
vol far outside realistic rates levels), not a case this bootstrap should silently paper over
the way an exercise boundary's widened window does for its own, different reason (an exercise
boundary has no natural bound). A bucket that ends at an edge is a miss of the calibration
([Acceptance](#acceptance)).

An exact bootstrap reprices every basket instrument exactly (to the rounding of the
instrument's price) — unlike a joint least-squares/`BestFit`
calibration, whose RMSE is generally nonzero even at convergence, since it's trading off
fit quality across every instrument simultaneously rather than fitting each one exactly in
turn.

### Why a joint optimizer was never built

A general Levenberg-Marquardt joint optimizer was considered and turned out unnecessary: the
bootstrap is a sequence of independent 1D root-finds, each with a closed, monotonic
relationship between one sigma bucket and one helper's price (see "Monotonicity" below), not
a joint multi-parameter fit. A JAX-native LM implementation would only be needed for an
ORE-style `BestFit` calibration mode — genuinely fitting several sigma buckets (or sigma
and mean reversion together) jointly against a basket where no exact reprice is expected —
which is out of this module's scope. No file exists for this; it's a documented "not
needed here," not a stub or an unfinished component.

### Why mean reversion is never calibrated

`a` (mean reversion / kappa) is always an input to the bootstrap (the model's `reversion`),
never an output.
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

## The calibration basket

**This engine:** `engine.calibration.lgm.build_basket`, each helper QuantLib's own
`SwaptionHelper` read into arrays (`BasketInstrument`, a pytree, so one jitted bootstrap serves
every basket of a shape).

**ORE:** `ore::data::IrModelBuilder::buildSwaptionBasket()`
(`OREData/ored/model/irmodelbuilder.cpp`), on the currency's swap index conventions
(`SwapIndexConventions`: the fixed leg's tenor and day counter, the Ibor index's tenor).

Two baskets, as ORE builds them:

- **The cross-asset model's** (`engine.calibration.cam`): ORE's `CalibrationSwaptions`, an
  expiry and a term per helper given as tenors (`LgmConfig.calibration_expiries`,
  `calibration_terms`), struck at the money, on today's market.
- **A Bermudan's or American's** (`engine.pricing.bermudan.calibration_basket`): ORE's
  co-terminal basket, one helper per exercise date into the swap to the trade's maturity
  (`CoterminalDealStrike`, struck at the deal's strike, or `CoterminalATM`), a deal strike
  further from the money than 3 ATM standard deviations moved there (`createSwaptionHelper`'s
  fallback rule 1); an American's on the reference calibration grid inside its window.

This is the natural choice, not an arbitrary one: exercising a Bermudan at `t_i` always
means entering the *same* underlying swap that runs to the trade's own final maturity,
just with a shorter remaining tenor — a co-terminal basket calibrates the model against
instruments that are economically the closest available liquid proxy for each of the
Bermudan's own exercise decisions.

Each helper's volatility is read from the market's ATM normal surface where ORE reads it
(`basket_vols`: its option date and swap length), in the normal (Bachelier) convention of
[Bermudan Vega](../risk/greeks.md#vega-bermudanamerican-only).

## The helpers' values

**Market value** (`market_price`): the helper's `BachelierSwaptionEngine` value on its own ATM
forward and annuity (nominal 1), out of the money (receiver when the strike is at or below the
forward), checked against QuantLib's own `SwaptionHelper.marketValue()`
(`tests/test_ore_lgm_calibration.py`).

**Model value** (`price_pair`): ORE's `AnalyticLgmSwaptionEngine` with its default
`FloatSpreadMapping::proRata` (the swaption priced on the discount curve alone, each floating
coupon's difference from its discount-curve amount moved onto the fixed coupons and the start),
the exercise boundary y\* the root of `yStarHelper`. `AnalyticLgmSwaptionEngine` has no Python
constructor, so it is checked through the end-to-end NPV: a Bermudan calibrated to its basket and
priced on the grid equals ORE's `LGMGridSwaptionEngineBuilder` with `Calibration=Bootstrap`, where
any error in the model value would move the calibrated volatility and the price.

### Monotonicity

The model value is monotonically increasing in the newest (most recently added) sigma bucket's
own value — a swaption is long volatility, and raising the newest bucket's sigma strictly
increases the model's total `zeta(T0_i)` for every `t` in that bucket. This is exactly what
makes each bucket's calibration a simple, well-posed scalar root: bracketed, monotone, with one
root when the market value is attainable.

## The root solver

**This engine:** `engine.solvers.roots` (decision A-21), for every root the
engine solves: each bucket of ORE's bootstrap (`engine.calibration.lgm`: the CAM's
calibration and every Bermudan's/American's, today, under every bump and scenario), each
helper's exercise boundary y\*, and the Jamshidian European engine's x\*.

| Setting | Where | Default |
|---|---|---|
| `LgmSwaptionEngineConfig.solver` (`pricing.bermudan.solver`, `pricing.american.solver`) | the option's calibration and its helpers' y\* | `"Newton"` |
| `LgmConfig.solver`, `HullWhiteConfig.solver` (`models.ir.<ccy>.solver`) | the CAM's calibration | `"Newton"` |
| `JamshidianEngineConfig.solver` (`pricing.jamshidian.solver`) | x\* | `"Newton"` |

Two solvers, both a fixed number of steps on every backend (a data-dependent stop would make
a GPU report to the host each step; a fixed count is one loop on the device):

- **`"Newton"`**, a safeguarded Newton method: each step evaluates the function and its
  derivative (`jax.jvp`), narrows the bracket to the side of the root the function's sign
  shows, and takes the Newton step where it stays inside the bracket, a bisection step where
  it would leave it (a zero or non-finite derivative included), so it is never less robust
  than bisection. Where the slope has the sign monotony gives it, the side of the root is the
  Newton step's direction (the residual's sign, when the residual is computed once; XLA may
  compute it twice, rounded differently, and the step then fall just outside a bracket moved
  by the other rounding: seen in float32 on 128 paths, where a bucket's volatility ended 74%
  off), so a root once found is kept to its rounding however many steps follow. Where it is not monotone across the bracket (a Jamshidian European struck far below
  the money: its coupons and its nominal cancel), Newton can step against the bracket and
  falls back to bisecting, which its count covers. `rtsafe`'s second safeguard (bisect unless
  a step halves an earlier one) is not used: it suits a loop that stops at a tolerance, and at
  a fixed count the root's own rounding-level steps, which need not halve, would set off
  bisection steps across a bracket still wide on one side (tried: a bucket's volatility
  halved). A bucket starts at its helper's market normal volatility, close to the bucket's in
  level and never at an edge of the bracket (the bucket before it can sit at the floor, where
  the slope vanishes and Newton would spend its steps bisecting); each y\* starts from 0, near
  which a helper's lies (it is struck within 3 ATM standard deviations), not from the y\* of
  the previous evaluation, which can sit far up the boundary's steep side (tried: in float32
  on a path, a helper's model value ended 2% off its market value). Steps: 7 per bucket, 5 per y\*, 26 per
  Jamshidian x\*.
- **`"Bisection"`**, the reference: the engine's halvings before 2.5 (60 per bucket, 100 per
  y\* and x\*), reproducing its numbers bit for bit. It keeps one path date per calibration
  call: under Newton a Bermudan's/American's path dates of one basket shape are calibrated in
  one call over dates and paths, which a bisection cannot share bit for bit (XLA vectorizes
  another shape, and the bisection's last comparisons move with the residual's last bit; the
  roots move within their resolution, 1.7e-13).

Newton's counts are measured (2026-10-07, the shared sloped market today and on 256 LGM and
Hull-White paths, and the shared European struck from 0% to 15%): a bucket reaches the
rounding of its residual in 5 steps, a y\* from 0 in 3, Jamshidian's x\* in 5 (in 24 for a
strike of -99%, where its coupons cancel the nominal and the coupon bond is not monotone across
the window), and each count keeps two steps of margin, which `tests/test_root_solvers.py` checks (two
steps fewer give the same roots, one step does not). Both solvers reach the same root: on those
markets a bucket's volatility under one equals the other's to 7e-13 and a Jamshidian NPV to
3e-16 of its nominal (a root is resolvable only to its residual's rounding over its slope), and
both reprice every helper to its rounding (at worst 7.5e-12 of the market value, a helper far
out of the money on a path).

**Against ORE.** ORE solves the same equations — each bucket's model value equals its market
value; `yStarHelper(y*) = 0` — with Levenberg-Marquardt to
`EndCriteria(1000, 500, 1e-8, 1e-8, 1e-8)` per bucket (`irmodelbuilder.cpp`) and Brent at
accuracy 1e-6 for y\* (`analyticlgmswaptionengine.cpp`). Both of the engine's solvers go to
float64 rounding, so the gap to ORE is ORE's stopping error, which is why calibrated
Bermudans agree with ORE to about 4e-11, under either solver. ORE's algorithms are not offered:
their per-element branching does not batch over paths.

**Brackets.** A bucket's bracket is fixed: a volatility beyond it ends at the edge (Newton
returns the edge itself) and is flagged (`hit_ceiling`), a miss of the calibration (below).
An exercise boundary has none: its starting window (`[-1, 1]` for y\*, `[-2, 2]` for x\*) is
widened side by side to 2, 8, 128, 32768, 2^31 or 2^63 times itself, the first that holds the
sign change, all candidates evaluated at once (one vectorized evaluation, not a loop).

**Derivatives.** A root carries no derivative of its own (`solve` stops its inputs'
tangents). Where a price depends on its root — x\* here and in the Jamshidian engine, and the
ORE helper's y\* — `implicit_root` gives the root its derivative by the implicit function
theorem, below; the helper's price is stationary in y\*, so only its second derivatives need
y\*'s. A whole calibration's buckets get theirs from the basket's system of residuals
(`implicit_solution`, `engine.calibration.lgm.calibrated_values`): what an option
calibrated inside its price function responds with ([Greeks](../risk/greeks.md), market risk's
recalibration per scenario).

<a id="acceptance"></a>
## Acceptance: when a calibration misses its basket

**This engine:** `engine.calibration.lgm.BootstrapResult.misses` and `require_accepted`, on
every ORE-style bootstrap: a Bermudan's/American's (`engine.pricing.bermudan.calibrate_on`) and
each currency's in the cross-asset model (`engine.calibration.cam`). Since 2026-10-12 (I-73).

ORE's `LgmBuilder::calibrate` computes the calibration error, the root mean square over the
basket of each helper's relative price error |market − model| / market (`getCalibrationError`
with `RelativePriceError`), and when it is not below the tolerance — a trade's `Tolerance`,
the cross-asset model's `BootstrapTolerance` — logs a structured warning and fails, unless the
global `ContinueOnCalibrationError` is set. The engine applies the same rule with
`LgmSwaptionEngineConfig.bootstrap_tolerance` and `CamConfig.bootstrap_tolerance`, one for the
model as ORE holds it (1e-4 by default: ORE has none, and 1e-4 is its shared example
configuration's and every example simulation's; decision A-24), and counts as a miss
also a bucket at the bracket's ceiling, where ORE's unbounded optimizer would go on. What
misses:

- **A dip after a spike.** A helper worth less than the model gives at the bracket's floor (the
  variance to its expiry below the variance the earlier buckets already carry) cannot be fitted
  by any non-negative volatility; the bucket ends at the floor and its relative error is large.
- **A zero volatility.** The helper's market value is then its intrinsic value, 0 at or out of
  the money (QuantLib's `bachelierBlackFormula`, `engine.models.bachelier`), so its relative
  error is not finite.
- **A volatility beyond the bracket** (2000 bp normal), where the bucket ends at the ceiling.

A miss on a market of the as-of date — today's, the simulation market where ORE builds the
trade, a bumped one — is refused with a `CalibrationError` naming the trade or the currency and
the date, as ORE fails without `ContinueOnCalibrationError`, which is not offered (A-24). On
the paths nothing is calibrated: ORE's exposure simulation keeps the model it built (I-49).

<a id="the-_bisect_xstar-gradient-bug"></a>
## The x\* gradient bug

**This engine:** `engine.solvers.roots.implicit_root`, which the Jamshidian engine's x\* and
ORE's helpers' y\* are found with, and `implicit_solution` (`engine.calibration.lgm.
calibrated_values`), which gives a whole calibration's buckets their derivatives.

**The bug, found in the retired standalone bootstrap's exercise boundary x\*.** A
bisection narrows a bracket with `jnp.where(val > 0.0, ...)` at every iteration, and that
comparison has zero gradient everywhere: differentiating straight through the unrolled loop
captures only a price's *direct* dependence on its parameters, at a fixed root, and drops the
*indirect* contribution `d(price)/d(x*) * d(x*)/d(sigma)` — the effect of `sigma` moving `x*`
itself. Not a precision shortfall: it silently omits a whole term of the derivative (a
systematic ~6% error in Bermudan Vega, caught by its finite-difference cross-check against a
literal recalibration). The Hull-White European engine's bisection for Jamshidian's critical
short rate `r*` (see
[ORE Parity](ore-parity.md#6-european-swaption-pricing-jamshidians-decomposition)) had the
identical bug, for the identical underlying reason.

**The fix.** A root carries the derivative of the
[implicit function theorem](https://en.wikipedia.org/wiki/Implicit_function_theorem) (see
[Delta, Gamma, and Theta](../risk/greeks.md#differentiating-through-bisection-root-finds)) — a
`jax.custom_jvp` implementing, at a root of `f(x*, params) = 0`:

```
dx*/dparams · v = -(df/dparams · v) / (df/dx)
```

for any tangent direction `v`, from one derivative of `f` in `x` at the converged root and one
`jax.jvp` in the parameters — cheap relative to the solve itself, and exact rather than
approximate. The parameters must be an explicit pytree (`params = (a, sigma)`), since
`jax.custom_jvp` attaches its rule to an explicit primal argument. Newton's iterations would
carry a derivative of their own, but only of the steps taken, not of the root: the rule is the
same for either solver (`implicit_root`). ORE's helpers' prices are stationary in y\*, so only
their second derivatives need y\*'s.

**A second, related fix: `Sigma` needed pytree registration.** The correction only works if a
tangent can reach `Sigma.values` when `Sigma` is nested inside a larger `params` tuple: before
`engine.models.lgm.Sigma` was registered with `@jax.tree_util.register_pytree_node_class`, an
unregistered dataclass was an opaque leaf to `jax.tree_util`, which contributed zero tangent
through its `.values` field and silently truncated the correction above. Both fixes were
required together. See [Models & Trades](models-and-trades.md#the-pytree-registration-bug).

Cross-checked against a literal finite-difference recalibration
(`tests/test_greeks_bermudan.py::TestBermudanVega`) and against ORE's sensitivity analytic, which
recalibrates under every bump (`tests/test_sensitivity_ore_parity.py`).

## Over HTTP

The run request's `calibration` analytic (`POST /runs` with `"analytics": ["calibration"]`)
returns the cross-asset model's calibration per currency of `config.models.ir` that has a basket
(`engine.calibration.cam.calibrate_cam`, the bootstrap of each currency's LGM or Hull-White
volatility to its tenor basket on today's market): each currency's sigma (times and values, in
its model's parametrization), the reversion it was calibrated with, each helper's market and model
value, and the calibration's error as ORE measures it against `config.models.bootstrap_tolerance`.
It is the calibration a run's exposure analytic simulates with on the same market and models,
which was checked on 2026-10-07 against ORE's own `CrossAssetModelBuilder` through its simulation
(`tests/test_ore_xva_parity.py`). It replaced `POST /calibration/cam` and the standalone
`POST /calibration/lgm` on 2026-10-18 ([HTTP API](http-api.md#the-run-request-post-runs)).

## Tested by

- `tests/test_ore_lgm_calibration.py` — against ORE: the helpers' market values against
  QuantLib's `SwaptionHelper.marketValue()` (tenor- and date-based helpers, ATM and deal
  strikes, a sloped two-curve market); the bootstrap reprices every helper; end to end, a
  Bermudan calibrated to its co-terminal basket and priced on the grid equals ORE's
  `LGMGridSwaptionEngineBuilder` with `Calibration=Bootstrap` (`CoterminalATM` and
  `CoterminalDealStrike`, a strike far enough out that the fallback rule moves it); batched
  curves calibrate as one curve at a time.
- `tests/test_calibration_acceptance.py` — ORE's acceptance (I-73): the misses, the tolerance
  settings and their API fields.
- `tests/test_cam.py`, `tests/test_hull_white_model.py::TestCalibration` — the cross-asset
  model's calibration, the Hull-White model's by matching zeta bucket by bucket.
- `tests/test_root_solvers.py` — the solver (2026-10-07): both solvers on known roots,
  brackets, widening, Newton's fallback, the implicit derivatives (first and second); on the
  parity markets every root under Newton equal to Bisection's and at rounding, with two
  steps of margin; the setting and its API fields; the path dates batched.
- `tests/test_greeks_bermudan.py::TestBermudanVega` — Vega through the calibration matches a
  literal finite-difference recalibration (see
  [Delta, Gamma, and Theta](../risk/greeks.md#vega-bermudanamerican-only)).
- `tests/test_api.py::TestTheCalibrationAnalytic` — the calibration analytic over HTTP is the
  calibration a run simulates with.
