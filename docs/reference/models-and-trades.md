# Models & Trades: The Shared Foundation Layer

**Modules:** [`engine/models/`](../../engine/models) —
[`hull_white.py`](../../engine/models/hull_white.py), [`lgm.py`](../../engine/models/lgm.py),
[`ore_builders.py`](../../engine/models/ore_builders.py)

## Plain-language summary

Every instrument pricer in this codebase (`swap.py`, `european_swaption.py`,
`bermudan_swaption.py`) needs the same two kinds of thing: (1) closed-form interest-rate
model mathematics — bond prices, bond options, discount factors — and (2) a real ORE
trade object, with a real ORE-generated payment schedule, to price against. Before
`engine/models/` held all of it, both were implemented **separately inside each instrument
file** — the same Hull-White formula written four times with slightly different call
shapes, the same ORE swap-building code written three times with nearly identical bodies.
`engine/models/` is the result of pulling all of that out into one place, so there is
exactly one implementation of each formula and each piece of trade-building machinery,
used by every pricer that needs it.

This split also did real, not just cosmetic, work: extracting the model math into its own
module made it possible to make sigma piecewise (`Sigma`, below) in exactly one place and
have it work for every downstream caller with no other code changes — the calibration
engine (`engine/calibration/`) and Bermudan Greeks would not have been possible without
this consolidation happening first.

## `engine/models/hull_white.py`

**The single source of truth for this codebase's Hull-White 1-Factor (HW1F) closed-form
math.** Before this module existed, the same formulas were implemented four separate
times: `engine/simulation/market_model.py::compute_hw_A_matrix` (NumPy, pillar-grid), `engine/
instruments/european_swaption.py::compute_hw_A`/`_hw_B` (NumPy `A`, JAX `B`, arbitrary
`(t,T)` pairs), and `engine/risk/greeks.py::_compute_hw_A_jax` (a JAX transliteration of
the NumPy version, built only because `np.interp` isn't traceable and Delta/Gamma need
`jax.grad` through the curve). Three of the four original docstrings said so explicitly
("identical formula to `simulation.compute_hw_A_matrix`", "JAX-differentiable
reimplementation of `european_swaption.compute_hw_A` — the SAME closed form"). This module
replaces all four with one JAX-native implementation (via `jnp.interp`, which *is*
traceable), used everywhere: by `engine.simulation.market_model` for the pillar-grid yield-curve cube,
by `engine.instruments.swap`/`european_swaption` for arbitrary-`(t,T)` bond/bond-option
pricing, and by `engine.risk.greeks` for autodiff Delta/Gamma.

**Formulas** (Brigo-Mercurio, live-verified against `ORE.HullWhite`/`QuantLib::HullWhite`
throughout this codebase's test suite — see
[ORE Parity](ore-parity.md#3-interest-rate-model-hull-white-1-factor) for the line-by-line
C++ correspondence):

```
B(t,T)   = (1 - exp(-a*(T-t))) / a
A(t,T)   = [P(0,T)/P(0,t)] * exp(B(t,T)*f(0,t) - (sigma^2/4a)*(1-exp(-2at))*B(t,T)^2)
P(t,T,r) = A(t,T) * exp(-B(t,T)*r)
```

where `f(0,t) = -d/dt ln P(0,t)` is today's instantaneous forward rate, read off the
caller's `ZeroCurve`.

**Contents:**

| Name | What it does |
|---|---|
| `ZeroCurve` | Today's market zero curve as `jax.Array`s (`pillar_times`/`pillar_rates`) — differentiable end-to-end via `jnp.interp`, unlike `engine.simulation.market_model.ZeroCurveConfig`'s plain Python lists. `ZeroCurve.from_config` builds one from any object with `.times`/`.rates`; `ZeroCurve.flat` builds a flat curve for demos/tests. |
| `zero_rate`, `log_discount`, `discount`, `forward_rate` | Curve interpolation and the derived discount/forward quantities every formula below needs. |
| `B(t, T, a)` | `(1-exp(-a*(T-t)))/a`, guarded at `a==0` (see below). |
| `A(curve, t, T, a, sigma, B_override=None)` | The today's-curve calibration term. `B_override` exists solely so `engine.simulation.market_model.compute_hw_A_matrix` can reproduce its own long-standing "clamped `B` for an aged pillar" convention bit-for-bit — ordinary callers never pass it. |
| `bond_price`, `bond_option_sigma`, `bond_call`, `bond_put` | The full affine bond price, the HW1F bond-option volatility (Brigo-Mercurio 3.41), and the Black-formula-on-a-bond payoff, each live-verified against `ORE.HullWhite`'s equivalent methods. |

**Constant sigma only — deliberately, not an oversight.** Unlike `engine.models.lgm.Sigma`
(below), this module has no piecewise-volatility support: `sigma` is always a plain
`float`. This is a direct consequence of reading ORE's own source, not an arbitrary scope
cut: `QuantLib::HullWhite` itself (`QuantLib/ql/models/shortrate/onefactormodels/
hullwhite.hpp`) has no piecewise-constant-volatility variant anywhere in ORE — confirmed
by reading the class hierarchy directly. There is no ORE counterpart this module would be
approximating by staying constant-only; matching `QuantLib::HullWhite` exactly means
staying constant-only. `swap.py`, `european_swaption.py`, `engine.simulation.market_model`, and
`engine.risk.greeks` all use this module and therefore all stay constant-sigma; only
`bermudan_swaption.py` (via `engine.models.lgm`) supports a genuine term structure, because
only `QuantExt::LinearGaussMarkovModel`/`Lgm1fPiecewiseConstantParametrization` supports one
in ORE.

**The `a==0`/`sqrt`-type removable-singularity guards.** Both `B` and `A` contain
`1/a` terms that are a removable `0/0` singularity at `a=0` (the mathematically valid
arithmetic-Brownian-motion limit of Ornstein-Uhlenbeck mean reversion, with a known
analytic limit — `B(t,T) -> T-t` by L'Hopital/first-order Taylor expansion). Both are
guarded via the same branch-free JAX pattern: evaluate the formula on a safe placeholder
value of `a` that is never actually `0`, select the correct branch with `jnp.where`, and
discard the placeholder result — both branches are always computed (required for
`jax.grad`/`jax.jit` tracing to work at all; a Python `if a == 0` would fail under
tracing), so there's no reliance on short-circuiting. This exact pattern recurs in
`engine.models.lgm.H` for its own analogous singularity, and in `bermudan_swaption.py`'s
`_state_grid`/backward-induction `std_step` computation for a *different* singularity
(`sqrt` at `0`, not division by `a` at `0`) — see
[American & Bermudan Swaptions](../instruments/american-bermudan-swaptions.md#the-_state_gridsqrt-gradient-bug).

## `engine/models/lgm.py`

**A different model from `engine.models.hull_white`, not an alternate parametrization of
the same one.** This is the single most important thing to understand about this module,
and it was not the original assumption going into building it. `ORE.HullWhite`
(`QuantLib::HullWhite`, what `hull_white.py` implements) and `ORE.LinearGaussMarkovModel`
(`QuantExt::CrossAssetModel`'s own rates leg, what this module implements) were assumed,
going into the original build of this codebase's Bermudan engine, to be two equivalent
parametrizations of the same underlying short-rate model —
[ORE Parity](ore-parity.md#a-parametrization-note-lgm-vs-plain-hull-white) documents
exactly that equivalence claim, verified at `t=0`. Building the Bermudan engine required
evaluating both classes at `t>0`, and a live, direct comparison against the installed ORE
package showed they are **not** the same numerical model realization there, despite being
constructed with identical `(a, sigma)` and sharing the same `t=0` curve:

```python
>>> hw.discountBond(t=3, T=5, r=0.03)       # r = f(0,t), HullWhite's own "no shock" point
0.9393234598794674
>>> lgm.discountBond(t=3, T=5, x=0.0)       # x = 0, LGM's own "no shock" point
0.9337296209777532
```

a genuine ~0.6% difference at `t=3y` (`a=0.03`, `sigma=0.02`) — not a rounding artifact.
Both classes are individually self-consistent affine short-rate models (each satisfies its
own `-d/dT log P(t,T)|_{T=t} == r` identity exactly, live-verified via finite difference),
so this is a real parametrization/calibration difference between the two ORE classes, not
a bug in either formula, and this investigation did not fully resolve it to a root cause —
it concretely measured and confirmed the divergence is real, without explaining exactly
why ORE's own two classes disagree there. Since ORE's actual Bermudan/American engine
(`QuantExt::NumericLgmMultiLegOptionEngine`) is built on `LinearGaussMarkovModel`, not
`HullWhite`, `bermudan_swaption.py` uses **only** this module's formulas for every discount
factor its backward induction computes — `engine.models.hull_white` is never imported
there.

**Formulas** (`QuantExt::Lgm1fParametrization`/`LinearGaussMarkovModel`,
`QuantExt/qle/models/lgm.hpp` lines 227-280, live-verified to machine precision against
`ORE.LinearGaussMarkovModel.discountBond` throughout this codebase's test suite):

```
H(t)      = (1 - exp(-a*t)) / a                (== hull_white.B(0, t, a), a live-verified identity —
                                                   NOT evidence the two models coincide for t>0)
zeta(t)   = integral_0^t sigma(s)^2 ds          (sigma^2 * t for constant sigma)
P(t,T,x)  = [P(0,T)/P(0,t)] * exp(-0.5*(H(T)^2-H(t)^2)*zeta(t)) * exp(-(H(T)-H(t))*x)
N(t,x)    = exp(0.5*H(t)^2*zeta(t) + H(t)*x) / P(0,t)     (the model's numeraire)
r(t,x)    = f(0,t) + x*H'(t) + zeta(t)*H'(t)*H(t)          (LGM's own short rate at state x)
```

`x(t)` is the model's own state variable: driftless under its own transition law
(confirmed directly from `QuantExt::IrLgm1fStateProcess::expectation()`, which returns its
input unchanged — no drift term at all), which is what makes Hagan's quadrature
convolution (`bermudan_swaption.py`'s backward-induction rollback) valid at all — see
[American & Bermudan Swaptions](../instruments/american-bermudan-swaptions.md#5-numeraire-deflation--the-step-that-makes-the-rollback-mathematically-valid).

**Contents:**

| Name | What it does |
|---|---|
| `Sigma` | Piecewise-constant volatility term structure (below). |
| `as_sigma` | Upgrades a plain `float`/`jax.Array` to a one-bucket `Sigma`; every function below calls this first. |
| `zeta(sigma, t)` | Accumulated variance, vectorized over buckets (no Python loop). |
| `H(a, t)`, `H_prime(a, t)` | `(1-exp(-a*t))/a` and its derivative `exp(-a*t)`. |
| `bond_price`, `numeraire`, `r_from_x`, `x_from_r` | The formulas above, plus `x_from_r`'s closed-form (not root-found — `r` is affine in `x`) inverse of `r_from_x`, used to convert a simulated `hw_paths` short rate into LGM's own state `x` for conditioning. |
| `bond_option_sigma` | The LGM analogue of `hull_white.bond_option_sigma` — read directly off `bond_price`'s own exponent, since `ln P(T_opt,S,x)` is affine in the Gaussian state `x(T_opt)`. |

`hull_white.bond_call`/`bond_put` (the Black-on-bond payoff formula itself) are reused
as-is by LGM's own swaption pricer (`engine.calibration.basket.price_lgm_swaption`) rather
than reimplemented — that formula is model-agnostic (it consumes only `P_t_Topt`, `P_t_S`,
`K`, `sigma_p`); only the volatility term (`bond_option_sigma`) differs between the two
models.

### `Sigma`: a piecewise-constant volatility term structure

`sigma(s) = values[i]` for `s` in bucket `i`, where the buckets are `[0, times[0]),
[times[0], times[1]), ..., [times[-1], inf)` — exactly ORE's own `alphaTimes`/`alpha` pair
(`QuantExt::Lgm1fPiecewiseConstantParametrization`'s constructor,
`QuantExt/qle/models/irlgm1fpiecewiseconstantparametrization.hpp`). `len(values) ==
len(times) + 1` always (one more bucket than interior breakpoints). `times` may be empty,
in which case `values` has exactly one entry — a flat sigma for all `t >= 0` — which is
what `Sigma.flat(sigma)` builds, and what every existing caller passing a plain `float` for
`hw_sigma` is automatically upgraded to via `as_sigma`, so every model formula in this
module has exactly one code path regardless of which case a caller is in.

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
`engine.calibration.basket._bisect_xstar`'s `params = (a, sigma)` tuple — rather than
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

See [Calibration: the `_bisect_xstar` gradient bug](calibration.md#the-_bisect_xstar-gradient-bug)
and [Delta, Gamma, and Theta: two real bugs](../risk/greeks.md#two-real-bugs-this-module-found-and-fixed)
for the full incident this bug was caught inside.

## `engine/models/ore_builders.py`

**The single source of truth for turning a trade config into a real ORE object and its
cashflow schedule.** Before this module existed, `_build_ore_swap` was defined three
separate times — in `swap.py`, `european_swaption.py`, and `bermudan_swaption.py` — with
identical bodies except that `european_swaption.py`'s version additionally passed
`forwardStart` to `ORE.MakeVanillaSwap` (`bermudan_swaption.py`'s own docstring said so
explicitly: "identical pattern to `swap._build_ore_swap` / `european_swaption._build_ore_
swap`"). The per-leg cashflow-extraction loop (iterate `swap.fixedLeg()`/`floatingLeg()`,
pull `payment`/`accrualStart`/`accrualEnd` dates and `accrualPeriod()` off each ORE coupon)
was likewise hand-copied four times across those same modules — fixed and floating legs in
`swap.py` and `bermudan_swaption.py`, fixed leg only in `european_swaption.py` (which
collapses the floating leg to a telescoping-notional identity instead — see
[European Swaptions](../instruments/european-swaptions.md#5-why-t_start-matters-the-floating-legs-notional-timing)
for why that shortcut is valid there specifically, and nowhere else). This module replaces
all of it with one implementation, used by every instrument pricer and by
`engine.calibration.basket.build_coterminal_basket`.

**Contents:**

| Name | What it does |
|---|---|
| `TIME_AXIS_DAY_COUNTER` | `ORE.Actual365Fixed()` — the **simulation time axis**. Permanently ACT/365, not configurable. |
| `DAY_COUNTER` | Deprecated alias for `TIME_AXIS_DAY_COUNTER`. Kept so existing imports work; it always meant the time axis. |
| `SUPPORTED_ACCRUAL_DAY_COUNTS` | The **instrument accrual** allowlist: `ACT/365` (default) and `ACT/ACT (ICMA)`. **Defined in [`engine/day_count.py`](../../engine/day_count.py) and re-exported here** — see below. |
| `resolve_accrual_day_count(name)` | Name → `ORE.DayCounter`, raising `UnsupportedDayCountError` for anything outside the allowlist. Also re-exported from `engine/day_count.py`. |
| `build_vanilla_swap(...)` | Builds a real `ORE.VanillaSwap` via `ORE.MakeVanillaSwap` — schedules, day counts, and conventions all come from ORE's own machinery, not a reimplementation. `forward_start` (an `ORE.Period` delaying the first accrual beyond the standard spot lag) defaults to `None`; `european_swaption.py` and `engine.calibration.basket.build_coterminal_basket` are the callers that pass a non-default value. `accrual_day_count` defaults to ACT/365. |
| `LegCashflows` | One leg's schedule as year-fractions from `today`: `payment_times`, `accrual_start_times`, `accrual_end_times`, `accrual_fractions`, `notional`. |
| `fixed_leg_cashflows`, `floating_leg_cashflows` | Extract a `LegCashflows` from a real `ORE.VanillaSwap`'s fixed/floating leg. |

### Where the accrual vocabulary lives (moved in W1.3)

`SUPPORTED_ACCRUAL_DAY_COUNTS`, `DEFAULT_ACCRUAL_DAY_COUNT`, `resolve_accrual_day_count` and
`UnsupportedDayCountError` are **defined in [`engine/day_count.py`](../../engine/day_count.py)**
and re-exported from this module, so every existing import and all 27 of
`tests/test_day_count_roles.py` are unchanged.

**Why they moved.** W1.3's note pricer
([`engine/integration/note.py`](../../engine/integration/note.py)) needs ACT/ACT (ICMA), but
`engine/integration/` is **forbidden** to import `engine.models` — this module is where
`build_vanilla_swap` lives, the exact object the EOD boundary's convention refusal exists to
keep unreachable ([I-05](../known-issues.md#i-05)). Importing it just to borrow a dictionary
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
| **Simulation time axis** — dates → year-fractions indexing `time_grid`, `maturities`, `hw_paths` | `TIME_AXIS_DAY_COUNTER` | **No, permanently ACT/365.** Every pricer's cashflow times are looked up against the simulated cube's own axis; [`_maturity_indices`](../../engine/instruments/swap.py) requires each to land *exactly* on a maturity pillar. Changing it silently desynchronizes every pricer — no error, just wrong discount factors. |
| **Instrument accrual** — the day count a contract's coupons accrue on | `accrual_day_count` | **Yes, per-instrument.** A property of the booking, not the engine: the TraderX note is ACT/ACT (ICMA), a USD-SOFR swap is ACT/360. |

Tracing every use: **49 are the time axis, 2 are the accrual**
([`ore_builders.py:90-91`](../../engine/models/ore_builders.py#L90-L91)'s
`fixedLegDayCount`/`floatingLegDayCount`). So the risky part of this change — what a contract
accrues on — is two lines; everything else is a rename that must not move a number.

Note what does *not* take `accrual_day_count`: the `SimIndex` index's own day count and the
dummy forward curve's. The index day count feeds ORE's forward-rate calculation, and this
engine never reads ORE's forwards — it reprices against the JAX-simulated cube.

**Defaults are byte-identical.** `accrual_day_count` defaults to ACT/365, which is what every
caller got before it existed. This is load-bearing: **38 tests across 10 files pin
`ORE.Actual365Fixed()` directly**, and the full suite passing unchanged is the acceptance
test for the rename.

**An unsupported day count is refused, never defaulted** — the same refuse-don't-infer rule
as [I-05](../known-issues.md#i-05), one layer down. `ACT/360` raises
`UnsupportedDayCountError` at `SwapConfig` construction, where the offending trade is
identifiable, rather than deep inside ORE at pricing time. A day count silently replaced by
ACT/365 shifts every accrual by 1.389%.

**Why the day count is forced explicitly at all, rather than left to `MakeVanillaSwap`'s
defaults.** `ORE.MakeVanillaSwap` has implicit per-index defaults that differ unpredictably
by index/currency (e.g. Euribor6M defaults to 30/360 fixed vs. Act/360 float). Both roles are
therefore always set here deliberately, rather than inherited by accident from whatever a
given index happens to default to.

**There is exactly one `TIME_AXIS_DAY_COUNTER`**, defined in
[`ore_builders.py`](../../engine/models/ore_builders.py) and *imported* by
[`bermudan_swaption.py`](../../engine/instruments/bermudan_swaption.py) and
[`greeks.py`](../../engine/risk/greeks.py).

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

**Why `floating_leg_cashflows` never reads ORE's own fixing.** `accrual_start`/
`accrual_end` times are what forward rates get computed from downstream, in whichever
pricer calls this function — ORE's own (single, deterministic) fixing history is never
read, since the entire point of every pricer in this codebase is repricing a trade's
cashflows under many JAX-simulated scenarios, not reproducing ORE's own one deterministic
valuation.

## How the pieces fit together

```
engine/models/hull_white.py  ──┬──►  engine/instruments/swap.py
                                ├──►  engine/instruments/european_swaption.py
                                ├──►  engine/simulation/market_model.py (yield-curve cube)
                                └──►  engine/risk/greeks.py (swap/European swaption Delta/Gamma)

engine/models/lgm.py          ──┬──►  engine/instruments/bermudan_swaption.py
                                 ├──►  engine/calibration/basket.py, lgm.py
                                 └──►  engine/risk/greeks.py (Bermudan Delta/Gamma/Theta/Vega)

engine/models/ore_builders.py ──┬──►  engine/instruments/swap.py
                                 ├──►  engine/instruments/european_swaption.py
                                 ├──►  engine/instruments/bermudan_swaption.py
                                 └──►  engine/calibration/basket.py
```

No instrument file imports another instrument file's model math directly any more — every
shared formula and every shared piece of ORE trade-building now has exactly one home. See
[Architecture](../concepts/architecture.md#the-repository-layout) for how this fits into
the repository layout as a whole.

## Tested by

- `tests/test_models_piecewise_sigma.py` (44 tests) — every `Sigma`/piecewise-`zeta`
  formula in `engine.models.lgm`, live-verified against
  `ORE.IrLgm1fPiecewiseConstantParametrization`/`ORE.LinearGaussMarkovModel` directly,
  plus the flat-scalar-upgrade path (`as_sigma`) and the pytree-registration gradient
  tests.
- `tests/test_bermudan_swaption.py::TestLgmClosedFormsAgainstORE` — `engine.models.lgm`'s
  closed forms checked directly against live `ORE.LinearGaussMarkovModel` objects, plus the
  explicit regression test documenting the `HullWhite` vs. `LinearGaussMarkovModel`
  divergence for `t>0` described above.
- `tests/test_day_count_roles.py` (27 tests) — the W1.1 time-axis/accrual split:
  **`TestOnlyTheAccrualRoleIsConfigurable`** (changing the accrual must move accrual
  fractions and leave cashflow *times* exactly where they were),
  `TestDefaultsAreByteIdentical`, `TestActActIcmaIsSupported`,
  `TestUnsupportedDayCountIsRefused`, `TestTimeAxisConstantsAgree`. Verified to fail
  against the dangerous wrong fix — making the time axis follow the instrument accrual.
- `tests/test_swap.py`, `tests/test_european_swaption.py` — indirectly exercise
  `engine.models.hull_white` and `engine.models.ore_builders` through the pricers built on
  them; see [ORE Parity](ore-parity.md) for the specific formula-level correspondences
  these tests check.
- `tests/test_calibration_basket.py` — `engine.models.ore_builders.build_vanilla_swap`'s
  reuse inside `build_coterminal_basket`, and the `Sigma` pytree registration's real-world
  consequence (the `_bisect_xstar` gradient fix — see [Calibration](calibration.md)).
