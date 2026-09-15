# Risk: Delta, Gamma, Vega, and Theta

**Module:** [`engine/risk/greeks.py`](../../engine/risk/greeks.py)
**Public entry points:** `swap_delta_gamma`, `swap_theta`, `swaption_delta_gamma`,
`swaption_theta`, `bermudan_delta_gamma`, `bermudan_theta`, `bermudan_vega`

## Plain-language summary

VaR and Expected Shortfall (see [VaR & Expected Shortfall](var_es.md)) answer "how much
could we lose across thousands of simulated futures?" **Greeks** answer a different,
complementary question: "if today's market moves by a small, specific amount, how much
does this one trade's value change?" A bank's trading desk uses Greeks constantly — to
hedge (buy or sell something else to offset the risk), to understand which market moves
actually matter for a given position, and to explain day-to-day P&L.

This module computes three of the most standard Greeks:

- **Delta** — how much a trade's value changes for a small move in a specific point on
  the interest rate curve (e.g. "if the 5-year rate rises by 0.01%, this trade gains
  $150"). Reported per curve pillar, not as one aggregate number, since a real trade is
  usually more sensitive to some maturities than others.
- **Gamma** — how much *Delta itself* changes as rates move; a measure of how curved
  (non-linear) a trade's value is. A plain swap has very little Gamma (its value is
  almost a straight line against rates); a swaption has meaningful Gamma (that curvature
  is exactly what optionality is).
- **Theta** — how much a trade's value changes purely from one day passing, with the
  market held completely still. Every trade has *some* Theta even in a frozen market,
  because moving one day closer to maturity changes discounting and, for options,
  changes how much time is left for the market to move before the exercise decision.

## Scope: every instrument in this codebase

This module computes Delta/Gamma/Theta for [interest rate swaps](../instruments/swaps.md),
[European swaptions](../instruments/european-swaptions.md), and
[Bermudan/American swaptions](../instruments/american-bermudan-swaptions.md), plus Vega
for Bermudan/American swaptions (see [Vega](#vega-bermudanamerican-only) below).

- **Bermudan/American Greeks** work because `bermudan_swaption.py`'s backward-induction
  engine (see [American & Bermudan Swaptions](../instruments/american-bermudan-swaptions.md))
  is implemented via `jax.lax.scan` — a genuine JAX computational graph end-to-end, so
  `jax.grad`/`jax.hessian` work exactly as they do for the swap/European swaption pricers.
- **Vega** is well-defined because [`engine/calibration/`](../reference/calibration.md)
  provides a real market-vol-to-model-parameter calibration step (see
  [Vega](#vega-bermudanamerican-only) below for why that's the required prerequisite).

## Why it's built this way: matching ORE's exact convention, via autodiff instead of finite differences

ORE computes Delta and Gamma via **bump-and-revalue**: its
`OREAnalytics::SensitivityAnalysis`/`SensitivityScenarioGenerator` classes
(`OREAnalytics/orea/engine/sensitivityanalysis.cpp`,
`OREAnalytics/orea/scenario/sensitivityscenariogenerator.cpp`) bump one yield-curve
pillar at a time — ORE's own example configuration
(`Examples/MarketRisk/Input/sensitivity.xml`) uses a `1 basis point` (`0.0001`) absolute
zero-rate move — reprice the whole portfolio under each bumped market scenario, and
finite-difference the resulting NPVs
(`OREAnalytics/orea/cube/sensitivitycube.cpp`):

```
delta = NPV(curve bumped up) − NPV(base)
gamma = NPV(curve bumped up) − 2×NPV(base) + NPV(curve bumped down)
```

**This module computes the mathematically identical quantity a different way.** Rather
than literally perturbing a curve and re-running the pricer twice per pillar, it uses
`jax.grad`/`jax.hessian` — automatic differentiation — to compute the exact derivative of
NPV with respect to each curve pillar's zero rate, then scales that exact derivative by
the same 1 basis point ORE uses. The result is ORE's own "dollar Delta/Gamma for a 1bp
move," computed with no finite-difference truncation error and no arbitrary step-size
choice. This mirrors a design decision ORE itself makes: alongside its production
bump-and-revalue framework, ORE also maintains closed-form
`DiscountingSwapEngineDeltaGamma`/`BlackSwaptionEngineDeltaGamma` engines
(`QuantExt/qle/pricingengines/discountingswapenginedeltagamma.hpp`,
`blackswaptionenginedeltagamma.hpp`) purely to cross-check its own finite-difference
numbers (`OREAnalytics/test/sensitivityvsanalytic.cpp`) — this module goes one step
further and uses the closed-form (autodiff) route as the primary implementation, since it
is exact rather than approximate.

**Bucketed per curve pillar, with the same triangular interpolation shape ORE uses — not
a single parallel shift.** ORE's `ShiftScenarioGenerator::applyShift`
(`OREAnalytics/orea/scenario/shiftscenariogenerator.cpp`) bumps one pillar at a time with
a triangular ("tent") weight: the bump's effect ramps from 0 at the neighboring pillars up
to full strength at the bumped pillar itself, and is flat-extrapolated beyond the first
and last pillar. This module's own zero curve interpolation
(linear on zero rates, flat at the ends — the same convention
`european_swaption.compute_hw_A` already uses) has exactly that same piecewise-linear
support, so differentiating NPV with respect to a single pillar's rate automatically
produces the identical triangular sensitivity ORE's explicit bump shape encodes — no
separate bump-shape logic is needed here.

## Rho

ORE has no separate "Rho" concept for interest-rate-sensitive instruments — its
`RiskFactorKey::KeyType` enum has no rho-specific entry, and its sensitivity reports emit
only "Delta"/"Gamma" columns for whatever risk factor was bumped, curves included. An
interest-rate-curve Delta (as this module computes it) **is** ORE's own equivalent of a
textbook option "Rho." There is no separate Rho function in this module.

## Vega (Bermudan/American only)

**Why Vega needed a calibration engine first.** ORE's swaption Vega bumps the
market-quoted implied-volatility surface used to **calibrate** the model
(`SensitivityScenarioGenerator::generateSwaptionVolScenarios`) — `hw_sigma`/the LGM
`Sigma` term structure is a calibration *output*, never an independent risk factor in its
own right; there is no `RiskFactorKey::KeyType` anywhere in ORE for a raw model
parameter. Before [`engine/calibration/`](../reference/calibration.md) existed, this
codebase's swaption pricers took `hw_sigma` directly as a config input with no
market-vol-to-model calibration step anywhere in the pipeline — so `d(NPV)/d(hw_sigma)`
was a real, computable number, but a genuinely *different* quantity from ORE's Vega (a
raw model-parameter sensitivity, not a market-vol sensitivity), and reporting it under the
name "Vega" would have misrepresented what it means. `engine.calibration.lgm.
calibrate_lgm_sigma` (a bootstrap fit of a piecewise `Sigma` to a co-terminal basket of
market swaption vols, matching `ore::data::LgmBuilder::calibrate()`'s own bootstrap path —
see [Calibration](../reference/calibration.md)) supplies exactly the missing
market-vol-to-model relationship, making a genuine, ORE-equivalent Vega possible for the
first time.

**Definition.** `bermudan_vega` computes `d(NPV)/d(market_vol_i)` for each basket
instrument `i` — the dollar NPV change for a 1bp move in *one* market swaption's own
quoted normal volatility, holding every other market quote fixed, exactly ORE's own bump
definition. It is **not** implemented by literally re-running `calibrate_lgm_sigma` once
per bumped market vol (which is what ORE itself does) — that would work, but would pay
for the bootstrap's own root-find tolerance and finite-difference truncation error on top
of the Bermudan pricer's own cost, repeated once per basket instrument. Instead,
`bermudan_vega` differentiates straight through the bootstrap's own root-find via the
implicit function theorem, giving an exact closed-form Vega at roughly the cost of one
Bermudan pricing call plus a small (`O(n_buckets^2)`) amount of extra `jax.grad` work.

**The chain rule has two links, both computed via the implicit function theorem, not
autodiff through Python control flow.** `calibrate_lgm_sigma`'s bootstrap loop runs on
the CPU (each bucket's calibration builds a fresh ORE swap via `build_coterminal_basket`),
so `jax.grad` cannot trace through it directly. Instead:

1. **`d(NPV)/d(sigma_j)`** — one `jax.grad` of the Bermudan price with respect to the full
   calibrated `Sigma.values` vector (this is exactly what Delta/Gamma-style
   differentiation already gives, extended to a new argument).
2. **`d(sigma_j)/d(market_vol_i)`** — the harder link. The bootstrap calibrates each
   bucket `s_j` as the root of `g_j(s_0,...,s_j; v_j) = 0` (model price minus market
   price, using only buckets `0..j` and only target `j`'s own market vol). A bump to
   `v_i` moves `s_i` directly, and *through* `s_i`, moves every **later** bucket `s_j`
   (`j > i`) too, since `g_j` depends on every earlier bucket's own value. This is a
   genuinely triangular (lower-triangular, not diagonal) system — `bermudan_vega` builds
   the full Jacobian `d(s_j)/d(v_i)` by forward substitution over `j`, using one
   `jax.grad` of `price_lgm_swaption` per bucket to get each row's own partial
   derivatives, then combines it with step 1's vector via a single dot product to get
   the final Vega for every basket instrument at once.

**Why the full Jacobian matters, not just the diagonal.** `d(s_j)/d(v_i) = 0` for `j < i`
(the bootstrap is triangular forward in time; an earlier bucket cannot depend on a later
target), but is generally **nonzero** for `j > i` — a change to an earlier bucket's
calibrated sigma cascades forward into every later bucket's own calibration equation.
Assuming a diagonal-only Jacobian understates Vega by 45-78% in every bucket except the
last. `price_lgm_swaption`'s exercise-boundary root-find (`_bisect_xstar`) uses the same
[implicit-function-theorem `custom_jvp` pattern](#differentiating-through-bisection-root-finds)
as `_solve_rstar` so its gradient with respect to sigma is exact, and
`engine.models.lgm.Sigma` is registered as a proper JAX pytree so tangents propagate
through its `values` field when nested inside a larger argument tuple. See
[Calibration](../reference/calibration.md) for the full derivation, and
`tests/test_calibration_basket.py`/`tests/test_greeks_bermudan.py` for the
finite-difference regression tests (matching to within ~0.005%-0.03% of a literal
finite-difference recalibration).

**`cfg.hw_sigma` must be the exact `Sigma` `calibrate_lgm_sigma` produced** from the same
`calibration_targets` list, in the same order — `bermudan_vega` does not re-run
calibration itself, only differentiates through the relationship between each target's
own market vol and that already-calibrated `Sigma`.

## Why no Vega for swaps or European swaptions

A linear swap has no volatility exposure at all (no optionality — Vega is meaningless).
`SwaptionConfig` (the European swaption pricer's config) was never migrated to accept a
calibrated `engine.models.lgm.Sigma` the way `BermudanSwaptionConfig` was — it still takes
a flat `hw_sigma` directly, with no calibration step behind it — so there is no
market-vol-to-model relationship to differentiate through for a European swaption yet,
for the same "calibration output, not an independent risk factor" reason described above.

## Theta: advance the evaluation date, hold the market fixed

ORE's Theta (`SensitivityAnalysis::generateSensitivities`) advances the evaluation date
by a configured period (its own default is 1 day), holds every market quote's own
shape/level completely fixed (no re-simulation, no re-fitting — the *same* curve object,
just read from a later reference date, which changes its implied discount factors purely
through the passage of time), reprices, and adds back any cashflow paid in the interim so
a coupon payment isn't misread as a pure valuation loss:

```
Theta = NPV(today + 1 day, SAME curve) − NPV(today) + cashflow paid in between
```

This module reproduces that definition exactly. Unlike Delta/Gamma, Theta is **not** an
autodiff computation — an evaluation date has no meaningful continuous derivative to take;
it's a literal forward difference along the time axis, exactly like ORE's own approach.

**A European swaption's Theta has no interim-cashflow term.** The underlying swap's
cashflows only matter *at* exercise (decomposed into the option's payoff by Jamshidian's
trick — see [European Swaptions](../instruments/european-swaptions.md)), not paid
independently before then, so `swaption_theta` is a pure repricing difference with no
`+ cashflow` term, unlike `swap_theta`.

**A swap's Theta only accounts for the fixed leg's cashflows in the window, not the
floating leg's.** The floating leg's rate for a period landing inside the (very short,
1-day-by-default) Theta window would depend on a fixing that hasn't happened yet as of
today — this module doesn't simulate that fixing. This is a documented simplification, not
a silent one: a swap's floating leg pays only on its own reset dates (typically monthly or
longer), so a floating payment landing within a single day of today is the rare exception,
not the common case this simplification needs to handle exactly.

## A JAX-differentiable curve: `ZeroCurve`

Both pricers' main entry points (`price_swaps`, `price_swaptions`) read today's curve
through plain NumPy interpolation (`np.interp`) — correct and fast for their own one-time,
CPU-side setup, but not something `jax.grad` can differentiate through (NumPy code has no
JAX computational graph). `ZeroCurve` is this module's differentiable stand-in: the same
zero-rate-pillar structure, but as genuine `jax.Array`s, interpolated via `jnp.interp`
(mathematically identical to `np.interp`, just traceable). Every Greek this module
computes is a derivative with respect to `ZeroCurve.pillar_rates`.

## Bermudan/American: Delta, Gamma, Theta

`bermudan_delta_gamma`/`bermudan_theta` follow the identical pattern and units as the
swap/European swaption functions above — per-pillar dollar Delta/Gamma for a 1bp curve
move, and a 1-day repricing-difference Theta — built directly on top of
`bermudan_swaption.py`'s own `_run_backward_induction`, which is fully JAX-native as of
the port to `jax.lax.scan` (see
[American & Bermudan Swaptions](../instruments/american-bermudan-swaptions.md)). No
separate JAX reimplementation of the pricing formula was needed here (unlike
`swaption_delta_gamma`, which originally needed its own JAX twin of a then-NumPy-only
Jamshidian formula) — `_bermudan_price_fn` simply substitutes differentiable JAX values
for the prepared trade's own `zero_rates`/`hw_sigma` fields via `dataclasses.replace` and
calls the existing backward induction directly.

**Gamma's finite-difference cross-check needs a different methodology than Delta's.** A
direct central finite-difference of the *price* (`(NPV_up - 2·NPV_base + NPV_down) /
bump²`) is numerically unreliable for the Bermudan pricer at a realistic 1bp-scale bump —
the NPV's own magnitude (~$10⁴) swamps the true second-order signal in float64
cancellation error at that bump size, and the naive finite difference swings by orders of
magnitude (and even flips sign) across different bump sizes while the autodiff Hessian
stays fixed. The numerically sound check instead finite-differences the *gradient itself*
(`(grad(rate+eps) − grad(rate−eps)) / (2·eps)`), which has no such cancellation problem —
see `tests/test_greeks_bermudan.py`'s own docstring and `TestBermudanDeltaGamma` for the
full methodology and the convergence check that confirms this is a numerical artifact of
the cross-check, not a bug in the autodiff Hessian itself.

**American swaptions have no separate Greeks function** — `AmericanSwaptionConfig.
to_bermudan()` expands into a `BermudanSwaptionConfig`, so `bermudan_delta_gamma(cfg.
to_bermudan(), curve)` covers both, matching `american_swaption.py`'s own "American is
just a finely-discretized Bermudan" design.

## Differentiating through bisection root-finds

Naively differentiating through a bisection-based root-find gives a silently wrong (not
merely imprecise) gradient, because a bisection's comparison (`jnp.where(val > 0.0, ...)`)
has zero gradient everywhere — `jax.grad` straight through the unrolled loop ignores how
the converged root actually moves with the function's own inputs. Two root-finds in this
codebase need a gradient through them and both use the same fix:

**1. `european_swaption._solve_rstar`** — the vectorized bisection that finds
Jamshidian's critical exercise-boundary short rate `r*`. Uses `jax.custom_jvp`,
implementing the
[implicit function theorem](https://en.wikipedia.org/wiki/Implicit_function_theorem)
directly: at a root of `f(r*, params) = 0`, `d(r*)/d(params) = −(∂f/∂params) / (∂f/∂r)`,
computed cheaply relative to the 100-iteration bisection itself (one `jax.grad` and one
`jax.jvp` call). `_solve_rstar`'s signature takes the values Delta/Gamma need to
differentiate with respect to as an explicit `params` argument rather than capturing them
in a Python closure — `jax.custom_jvp` requires an explicit primal argument to attach a
gradient rule to. See `_solve_rstar`'s own docstring in `european_swaption.py` for the
full explanation.

**2. `engine.calibration.basket._bisect_xstar`** — the LGM analogue of `_solve_rstar`,
used by `bermudan_vega` (see [Vega](#vega-bermudanamerican-only) above). Uses the
identical `custom_jvp`/implicit-function-theorem pattern, plus
`engine.models.lgm.Sigma` is registered as a proper JAX pytree
(`@register_pytree_node_class`) so a tangent can propagate into its `values` field when
`Sigma` is nested inside a larger `params` tuple/pytree rather than passed as a bare
array — an unregistered dataclass is treated as an opaque leaf by `jax.tree_util`, which
would otherwise block any tangent from reaching its fields.

## The functions

```python
def swap_delta_gamma(cfg: SwapConfig, disc_curve: ZeroCurve, fwd_curve: ZeroCurve, bump_size=DEFAULT_RATE_BUMP) -> Dict[str, jax.Array]:
```
Per-pillar Delta/Gamma of one swap's t=0 NPV, with respect to its own discount and
forward curves independently. Returns `"discount_delta"`, `"discount_gamma"`,
`"forward_delta"`, `"forward_gamma"`, each shaped `[len(pillar_rates)]`.

```python
def swap_theta(cfg: SwapConfig, disc_curve: ZeroCurve, fwd_curve: ZeroCurve, theta_days=DEFAULT_THETA_DAYS) -> float:
```
A single number: the swap's 1-day (by default) time-decay.

```python
def swaption_delta_gamma(cfg: SwaptionConfig, curve: ZeroCurve, bump_size=DEFAULT_RATE_BUMP) -> Dict[str, jax.Array]:
```
Per-pillar Delta/Gamma of one European swaption's t=0 NPV, with respect to its own
Hull-White calibration curve. Returns `"delta"`, `"gamma"`, each shaped
`[len(curve.pillar_rates)]`.

```python
def swaption_theta(cfg: SwaptionConfig, curve: ZeroCurve, theta_days=DEFAULT_THETA_DAYS) -> float:
```
A single number: the swaption's 1-day (by default) time-decay.

```python
def bermudan_delta_gamma(cfg: BermudanSwaptionConfig, curve: ZeroCurve, bump_size=DEFAULT_RATE_BUMP) -> Dict[str, jax.Array]:
```
Per-pillar Delta/Gamma of one Bermudan/American swaption's t=0 NPV, with respect to its
own LGM calibration curve. Returns `"delta"`, `"gamma"`, each shaped
`[len(curve.pillar_rates)]`.

```python
def bermudan_theta(cfg: BermudanSwaptionConfig, curve: ZeroCurve, theta_days=DEFAULT_THETA_DAYS) -> float:
```
A single number: the Bermudan/American swaption's 1-day (by default) time-decay.

```python
def bermudan_vega(cfg: BermudanSwaptionConfig, curve: ZeroCurve, calibration_targets: List[CalibrationTarget], market_vol_bump: float = 0.0001) -> jax.Array:
```
Per-basket-instrument Vega: dollar NPV change for a 1bp move in each of
`calibration_targets`' own market vol, in the same order as `calibration_targets`.
`cfg.hw_sigma` must be the `Sigma` `calibrate_lgm_sigma` produced from
`calibration_targets`. Shaped `[len(calibration_targets)]`.

**Gamma is the diagonal only, not a full cross-pillar Hessian.** ORE's own
`SensitivityCube::gamma` is a cross-*scenario* second difference at one pillar, and so
only ever reports this same-pillar term — never a genuine cross-pillar second derivative
(how Delta at pillar A changes as pillar B moves). This module matches that scope.

It also *computes* only that diagonal. An earlier version built the full `jax.hessian`
and returned `jnp.diagonal` of it, which meant forward-over-reverse-differentiating the
whole pricer `n` times and discarding `n² − n` of the results. `_grad_and_hessian_diagonal`
now gets each diagonal entry from one Hessian-vector product against a basis vector
(`hvp(f, x, eᵢ)[i] == ∂²f/∂xᵢ²`), batched under `vmap`. The two are mathematically
identical, and `tests/test_profiling_and_jit.py::TestHessianDiagonalEquivalence` pins that
they agree numerically for all three instrument types as well as for an analytic case with
a known closed-form answer. See [Profiling & the Tracer](../concepts/profiling.md) for why
this mattered.

> If cross-pillar curvature (curve-twist risk) is ever wanted, the full Hessian is still
> one `jax.hessian` call away — the diagonal-only choice is ORE parity, not a limitation
> of the autodiff.

## Tested by

- `tests/test_greeks.py::TestSolveRstarGradientCorrectness` — `_solve_rstar`'s
  `custom_jvp` gradient rule, tested directly against toy root-finding problems with
  known closed-form derivatives (both first and second order), independent of the
  swaption pricer itself.
- `TestSwapDeltaGamma`/`TestSwaptionDeltaGamma` — direct comparison against literal
  finite-difference bump-and-revalue (the same computation ORE itself performs), across
  payer/receiver, deep ITM/OTM, and a spread of Hull-White parameters.
- `TestSwapDeltaGammaAgainstORE`/`TestSwaptionDeltaGammaAgainstORE` — an independent cross-
  check against a real `ORE.VanillaSwap`/`ORE.Swaption` priced twice under a directly
  bumped `ORE.FlatForward` curve, isolating the parallel (whole-curve) sensitivity.
- `TestSwapTheta`/`TestSwaptionTheta` — Theta matches a from-scratch manual reprice-
  difference computed from the same building blocks, plus finiteness/magnitude sanity
  checks and a zero-horizon no-op check.
- `TestComputeHwAJaxMatchesNumpy` — the JAX-native curve-interpolation twin of
  `compute_hw_A` matches the original NumPy formula exactly (not approximately) across a
  grid of `(t, T, a, sigma)` combinations.
- `TestSwaptionPriceFnMatchesMainPricer` — this module's own from-scratch t=0 swaption
  pricing path reproduces `price_swaptions`' actual output exactly.
- `tests/test_greeks_bermudan.py::TestBermudanDeltaGamma` — Delta cross-checked against a
  direct price-level finite difference; Gamma cross-checked against a finite difference
  *of the gradient* (see this doc's own explanation of why a price-level check is
  numerically unreliable here).
- `TestBermudanTheta` — finiteness/magnitude sanity checks and a zero-horizon no-op check.
- `TestBermudanVega` — the core correctness check: Vega against a literal
  finite-difference recalibration (bump one basket instrument's market vol, rerun
  `calibrate_lgm_sigma`, reprice — exactly what ORE itself does), matching to within
  ~0.005% for every bucket in a 4-instrument basket; plus positivity checks (payer and
  receiver both long-vol) and a bucket-count-mismatch guard test.
- `TestAmericanSwaptionSharesTheSameGreeksPath` — confirms `AmericanSwaptionConfig.
  to_bermudan()` feeds `bermudan_delta_gamma` correctly (no separate American-specific
  Greeks function exists).
- `tests/test_profiling_and_jit.py::TestHessianDiagonalEquivalence` — the HVP-based Gamma
  equals `jnp.diagonal(jax.hessian(...))` for swap, European swaption and Bermudan, and
  equals a known analytic second derivative on a closed-form case.
- `TestGradientsSurviveTheJitBoundary` — guards the failure mode the `_PreparedBermudan`
  pytree split could introduce: a differentiable field placed in *static* aux data, for
  which JAX does not raise but silently returns a **zero** gradient.
- `tests/test_calibration_basket.py::TestPriceLgmSwaptionSanity::
  test_gradient_wrt_sigma_matches_finite_difference_value`/
  `test_gradient_wrt_piecewise_sigma_bucket_matches_finite_difference` — value-level (not
  just sign/finiteness) cross-checks of `_bisect_xstar`'s gradient and `Sigma`'s pytree
  registration against finite difference.
