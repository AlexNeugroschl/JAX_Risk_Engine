# Models & Trades: The Shared Foundation Layer

**Modules:** [`engine/models/`](../../engine/models/) — [`curves.py`](../../engine/models/curves.py),
[`lgm.py`](../../engine/models/lgm.py), [`hull_white.py`](../../engine/models/hull_white.py),
[`ore_builders.py`](../../engine/models/ore_builders.py)

## Plain-language summary

Every stage of the engine needs a few things that have nothing to do with what makes a trade
distinctive: today's curves as differentiable arrays, the interest-rate model's closed-form
formulas, and a real ORE trade object with a real payment schedule. This layer holds one
implementation of each, used by the simulation, the valuation, the calibration and the
Greeks, so no formula or schedule loop is written twice.

## `engine/models/curves.py`

The curve primitives: `ZeroCurve` (today's market curve as `jax.Array`s, linear in the zero
rate between pillars and QuantLib's flat forward beyond the last, differentiable in its
pillar rates; `ZeroCurve.from_config`) with `zero_rate`, `log_discount`, `discount`,
`forward_rate`; and `DiscountCurve` (a batched curve of log discount factors at tenor times,
log-linear between them: a scenario market's curve on one date, for every path at once).
Every pricer reads one of the two, so the pricers do not know which model produced a curve.

## `engine/models/lgm.py`

The interest-rate model's closed forms. ORE's cross-asset model and its Bermudan engine are
both the **Linear Gauss-Markov model** (`QuantExt::LinearGaussMarkovModel`), with one state
`x` per currency:

```
H(t)      = (1 - exp(-a*t)) / a
P(t,T,x)  = P(0,T)/P(0,t) * exp(-(H(T)-H(t)) x - 1/2 (H(T)^2 - H(t)^2) zeta(t))
N(t,x)    = exp(H(t) x + 1/2 H(t)^2 zeta(t)) / P(0,t)
r(t,x)    = f(0,t) + H'(t) x + zeta(t) H(t) H'(t)
```

ORE parametrizes the volatility two ways (`LgmData::VolatilityType`), and both are here:

- **Hagan** (the default): α piecewise constant, `zeta(t) = ∫ α²` (`zeta`).
- **Hull-White** (`IrLgm1fPiecewiseConstantHullWhiteAdaptor`): the short rate's σ piecewise
  constant, α(t) = σ(t)e^{at}, `zeta(t) = ∫ σ² e^{2as} ds` (`hull_white_zeta`, exact per
  bucket). This is the Hull-White model itself: with the short rate above, every bond price
  equals QuantLib's `HullWhite::discountBond(t, T, r)` (`tests/test_cam.py`).
  `hull_white_matching_zeta(a, hagan, last)` converts a Hagan calibration to the Hull-White σ
  with the same ζ at every bucket end, which is exact for a calibration (helper prices
  depend on the model only through ζ at their expiry and H).

The same numbers in the two parametrizations are different models: `ORE.HullWhite(a, σ)`
and the Hagan LGM with α = σ differ for t > 0 (the reason an early comparison here found a
0.6% gap at 3y). See [Market Simulation](../concepts/market-simulation.md#the-configuration).

| Name | Notes |
|---|---|
| `Sigma`, `as_sigma` | A piecewise-constant volatility (below). |
| `H`, `H_prime` | The state-space function and its derivative; independent of the volatility. |
| `zeta`, `hull_white_zeta`, `hull_white_matching_zeta` | The variance in each parametrization, and the conversion. |
| `bond_price`, `numeraire`, `r_from_x`, `x_from_r`, `bond_option_sigma` | The closed forms above (Hagan), checked against `ORE.LinearGaussMarkovModel`. |

### `Sigma`: a piecewise-constant volatility term structure

`sigma(s) = values[i]` for `s` in bucket `i`, where the buckets are `[0, times[0]),
[times[0], times[1]), ..., [times[-1], inf)` — exactly ORE's own `alphaTimes`/`alpha` pair
(`QuantExt::Lgm1fPiecewiseConstantParametrization`'s constructor,
`QuantExt/qle/models/irlgm1fpiecewiseconstantparametrization.hpp`). `len(values) ==
len(times) + 1` always (one more bucket than interior breakpoints). `times` may be empty,
in which case `values` has exactly one entry — a flat sigma for all `t >= 0` — which is
what `Sigma.flat(sigma)` builds, and what a plain `float` volatility is automatically
upgraded to via `as_sigma`, so every model formula in this module has exactly one code path
regardless of which case a caller is in.

**`H(t)` is completely unaffected by piecewise sigma.** Confirmed directly from ORE's own
source: `Lgm1fPiecewiseConstantParametrization::H` delegates to a helper built purely from
the (constant, uncalibrated — see [Calibration](calibration.md#why-mean-reversion-is-never-calibrated))
reversion parameter, never referencing alpha/sigma at all. Only `zeta`'s own computation
changes when sigma is piecewise; the bond price/numeraire/short-rate formulas above keep
their exact same structure regardless of which `zeta` feeds them.

`zeta` itself is vectorized (no Python loop over buckets) via a cumulative sum over
full-bucket contributions, then a `jnp.searchsorted` to find which bucket `t` falls in
(matching `std::upper_bound`'s semantics exactly — ORE's own bucket-lookup rule) plus that
bucket's own partial contribution — live-verified against
`ORE.IrLgm1fPiecewiseConstantParametrization.zeta` directly
(`tests/test_models_piecewise_sigma.py`).

### The pytree registration bug

`Sigma` is registered as a JAX pytree (`@jax.tree_util.register_pytree_node_class`), with
both `times` and `values` as children (not static/auxiliary data). This is required for
`jax.jvp`/`jax.custom_jvp`/`jax.grad` to differentiate correctly whenever a `Sigma` is
passed as part of a **larger** pytree argument — e.g.
`engine.calibration.basket.price_lgm_swaption`'s x* params, the `(a, sigma)` tuple — rather than
having its own `.values` array unpacked and passed directly as a bare `jax.Array`.

**This was a real, not merely theoretical, gap.** An unregistered `@dataclass` is treated
by `jax.tree_util` as an opaque leaf, silently preventing any tangent from propagating into
its fields at all — no error is raised; the gradient with respect to `Sigma`'s contents
simply comes back as zero (or, worse, partially correct in a way that's hard to distinguish
from a legitimately small effect). Before this registration was added, `_bisect_xstar`'s
own implicit-function-theorem correction (see [Calibration](calibration.md#the-_bisect_xstar-gradient-bug))
produced a systematic ~6% wrong gradient even *with* that correction in place, because
`jax.jvp` was differentiating `price_lgm_swaption` with respect to `sigma` as an opaque
object, contributing zero tangent through `Sigma.values`, silently understating the
correction term. Both fixes — the `custom_jvp` implicit-function-theorem correction and
this pytree registration — were required together; either one alone still produced a
systematically wrong gradient. `times` is registered as a child too, even though no caller
in this codebase currently differentiates with respect to bucket *breakpoints* (only
`values`, mirroring `ZeroCurve.pillar_times` never being a Greek's own target either) —
registering it as a child regardless is the conservative choice, since a differentiable
field mistakenly marked static would silently drop its own gradient, while a
non-differentiable field marked as a child costs nothing (its cotangent is simply unused).

See [Calibration: the x* gradient bug](calibration.md#the-_bisect_xstar-gradient-bug)
and [Delta, Gamma, and Theta: two real bugs](../risk/greeks.md#differentiating-through-bisection-root-finds)
for the full incident this bug was caught inside.

## `engine/models/hull_white.py`

`bond_call` / `bond_put`: Black's formula on a zero-coupon bond option, the building block of
the Jamshidian engine (`engine/valuation/jamshidian.py`). The Hull-White model's own curves and
numeraire are the LGM's in the Hull-White parametrization (above); its earlier affine
`A(t,T)`/`B(t,T)` implementation went with the separate Hull-White pipeline on 2026-10-01.

## `engine/models/ore_builders.py`

**The single source of truth for turning a trade into a real ORE object and its schedule.**

| Name | What it does |
|---|---|
| `TIME_AXIS_DAY_COUNTER` | `ORE.Actual365Fixed()` — the **time axis**. Permanently ACT/365, not configurable. `DAY_COUNTER` is a deprecated alias. |
| `SUPPORTED_ACCRUAL_DAY_COUNTS`, `resolve_accrual_day_count(name)` | The **instrument accrual** allowlist (`ACT/365`, the default, and `ACT/ACT (ICMA)`) and its resolution, refusing anything else (`UnsupportedDayCountError`). Defined in [`engine/day_count.py`](../../engine/day_count.py), re-exported here. |
| `build_vanilla_swap(...)` | A real `ORE.VanillaSwap` via `ORE.MakeVanillaSwap` from the booked `effective_date`/`maturity_date`, independent of any evaluation date (audit M-4). |
| `resolve_swap_dates(trade_date, swap_tenor, forward_start=None)`, `book_swap_dates` | A tenor-quoted swap's dates by `MakeVanillaSwap`'s own rule, resolved once at booking. |
| `ibor_index(tenor_months, curve=None)` | The `SimIndex` Ibor index the swaps and baskets are built on. |
| `par_coupon_forecast_period(coupon)` | The Ibor index's own fixing period a floating coupon is forecast over (I-31). |
| `known_fixing(fixing_date, today, fixings)`, `MissingFixingError` | ORE's `InterestRateIndex::fixing`: forecast after today, today's supplied-or-forecast, an earlier one must be supplied. |
| `is_live(cashflow_date, today)` | ORE's `hasOccurred` with default settings: a cashflow paid on `today` has occurred. |
| `time_from_reference`, `validate_tenor`, `validate_fixings` | Shared helpers. |
| `fixed_leg_cashflows`, `LegCashflows` | A swap's remaining fixed coupons, for the standalone calibration route's basket. |

The trades' legs as arrays, valued on any date and path, are `engine/valuation/legs.py`
([Interest Rate Swaps](../instruments/swaps.md)).

### Where the accrual vocabulary lives (moved in W1.3)

`SUPPORTED_ACCRUAL_DAY_COUNTS`, `DEFAULT_ACCRUAL_DAY_COUNT`, `resolve_accrual_day_count` and
`UnsupportedDayCountError` are **defined in [`engine/day_count.py`](../../engine/day_count.py)**
and re-exported from this module, so every existing import and all 27 of
`tests/test_day_count_roles.py` are unchanged.

**Why they moved.** W1.3's note pricer
([`engine/integration/note.py`](../../engine/integration/note.py)) needs ACT/ACT (ICMA), but
`engine/integration/` is **forbidden** to import `engine.models` — this module is where
`build_vanilla_swap` lives, the exact object the EOD boundary's convention refusal exists to
keep unreachable ([I-05](../planning/known-issues.md#i-05)). Importing it just to borrow a dictionary
would put that builder one attribute access from the refusal boundary. The dictionary moved to
a leaf module that imports only `ORE` and can therefore pull nothing in behind it.

**`TIME_AXIS_DAY_COUNTER` deliberately did *not* move.** It is a property of this engine's
simulated curve cube, not of any contract, so keeping the two roles in separate modules makes
the distinction below structural rather than a naming convention.

### The two roles `Actual/365Fixed` plays — and why they are named apart (W1.1)

One day count was doing two unrelated jobs under one name, which is what made "make the day
count per-instrument" look like a one-line change when it is not:

| Role | Constant | Configurable? |
|---|---|---|
| **Time axis** — dates → year-fractions for the curves, the model, the simulation dates and every cashflow | `TIME_AXIS_DAY_COUNTER` | **No, permanently ACT/365.** The model's time, the scenario market's tenor times and every pricer's cashflow times must be measured alike (ORE's model day counter); changing one silently desynchronizes the others — no error, just wrong discount factors. |
| **Instrument accrual** — the day count a contract's coupons accrue on | `accrual_day_count` | **Yes, per-instrument.** A property of the booking, not the engine: the TraderX note is ACT/ACT (ICMA), a USD-SOFR swap is ACT/360. |

Tracing every use: **49 are the time axis, 2 are the accrual**
([`ore_builders.py:90-91`](../../engine/models/ore_builders.py#L90-L91)'s
`fixedLegDayCount`/`floatingLegDayCount`). So the risky part of this change — what a contract
accrues on — is two lines; everything else is a rename that must not move a number.

Note what does *not* take `accrual_day_count`: the `SimIndex` index's own day count, which
fixes the index's forecast period and its year fraction (`par_coupon_forecast_period`).

**Defaults are byte-identical.** `accrual_day_count` defaults to ACT/365, which is what every
caller got before it existed. This is load-bearing: **38 tests across 10 files pin
`ORE.Actual365Fixed()` directly**, and the full suite passing unchanged is the acceptance
test for the rename.

**An unsupported day count is refused, never defaulted** — the same refuse-don't-infer rule
as [I-05](../planning/known-issues.md#i-05), one layer down. `ACT/360` raises
`UnsupportedDayCountError` at `SwapConfig` construction, where the offending trade is
identifiable, rather than deep inside ORE at pricing time. A day count silently replaced by
ACT/365 shifts every accrual by 1.389%.

**Why the day count is forced explicitly at all, rather than left to `MakeVanillaSwap`'s
defaults.** `ORE.MakeVanillaSwap` has implicit per-index defaults that differ unpredictably
by index/currency (e.g. Euribor6M defaults to 30/360 fixed vs. Act/360 float). Both roles are
therefore always set here deliberately, rather than inherited by accident from whatever a
given index happens to default to.

**There is exactly one `TIME_AXIS_DAY_COUNTER`**, defined in
[`ore_builders.py`](../../engine/models/ore_builders.py) and *imported* everywhere else (the
market, the simulation, the valuation, the calibration).

Those two modules previously constructed their own `ORE.Actual365Fixed()`, which was
described here as being "for import-cycle reasons" — that was not accurate. Both already
imported `ore_builders` for `build_vanilla_swap`, so no cycle ever required it; the
duplication was incidental. Three equal-but-distinct objects are a real hazard for a value
whose defining property is that it is *not configurable*: a change to one would leave the
others silently on the old value, and the tests of the day could not tell the difference
(`TestTimeAxisConstantsAgree` checks each constant's `.name()` independently, which three
separate ACT/365 objects satisfy just as well as one shared one).

`TestTimeAxisIsOneObject` now pins **identity** across all three modules, and additionally
AST-scans `engine/` to fail if any module re-introduces a local construction.

## Tested by

- `tests/test_cam.py` — the closed forms in both parametrizations against ORE's
  `LinearGaussMarkovModel` and QuantLib's `HullWhite`; `TestHullWhiteParametrization`.
- `tests/test_models_piecewise_sigma.py` — `Sigma`: flat compatibility, ζ against
  `ORE.IrLgm1fPiecewiseConstantParametrization`, H's independence of σ, bond price and
  numeraire, gradients.
- `tests/test_hull_white_model.py::TestCalibration` — the Hull-White conversion of a
  calibration reprices every helper exactly.
- `tests/test_curves.py` — `ZeroCurve` against QuantLib's interpolation and flat-forward
  extrapolation; forward-rate precision.
- `tests/test_day_count_roles.py` — the two day-count roles, the allowlist, one time-axis
  object.
- `tests/test_bermudan_swaption.py::TestLgmClosedFormsAgainstORE` — the grid engine's
  primitives.
