# Instruments: European Swaptions

**Module:** [`engine/instruments/european_swaption.py`](../../engine/instruments/european_swaption.py)
**Public entry point:** `price_swaptions(hw_paths, step_times, swaption_configs)`

## Plain-language summary

A **swaption** ("swap option") is the *right, but not the obligation,* to enter into an
[interest rate swap](swaps.md) at a fixed rate agreed today, on a specific future
date. A **European swaption** can only be exercised on that one specific date — not any
time before it (that's a *Bermudan* swaption — see
[American & Bermudan Swaptions](american-bermudan-swaptions.md)).

Whoever holds a swaption will only choose to exercise it if doing so is worth more than
not doing so — e.g. the holder of a *payer* swaption (the right to enter a swap paying
fixed) will only exercise if the fixed rate they locked in is now *below* where the
market actually ended up, so they come out ahead. That "only exercise if favorable"
feature is exactly what makes a swaption a genuinely different, harder pricing problem
than the [underlying swap](swaps.md) itself: a swap's value is just its expected
future cashflows, but a swaption's value also has to account for the *option* to walk
away, which requires reasoning about the probability that exercising will actually be
worthwhile.

This module answers the same kind of question the [swap pricing module](swaps.md) does — *"given
thousands of simulated alternate futures, what is this specific trade worth in each one,
at each point in time?"* — but for a swaption instead of a plain swap, producing the same
kind of NPV cube risk aggregation (see [VaR & Expected Shortfall](../risk/statistics.md)) needs.

## Why it's built this way: Jamshidian's trick

Pricing an option generally requires either a closed-form formula (fast, but only exists
for specific, simple cases) or running a *second*, nested simulation from every point a
decision might be made (always works, but is extremely slow). This project's root
[README.md](../../README.md) calls for **Jamshidian's trick** — a clever closed-form
shortcut that applies specifically to swaptions priced under a
[Hull-White 1-Factor model](../concepts/market-simulation.md#phase-2--the-cross-asset-model-engine)
(the same model the [market simulation module](../concepts/market-simulation.md) already simulates interest rates
with), avoiding the need for any nested simulation at all.

**The key mathematical insight:** a swap's fixed leg (plus the notional exchanged with
the floating leg) can be treated as a single bond with several coupons. Under a
Hull-White model, every zero-coupon bond's price is a strictly *monotonic* function of
the current short-term interest rate — meaning there's exactly one interest rate at which
that bond is worth precisely the swap's notional. Above that rate, exercising the
swaption isn't worthwhile; below it, it is (or vice versa, depending on payer/receiver).
That single critical rate lets the whole multi-coupon option be broken apart into a
portfolio of much simpler options, each on a single zero-coupon bond — and *those* have
an exact closed-form price. No simulation-within-a-simulation needed.

This is the same approach ORE's own `ORE.JamshidianSwaptionEngine` uses. **The formula in
this module was not written from a textbook — it was reverse-engineered and verified by
directly testing the installed ORE package**, the same methodology used throughout this
codebase for every ORE-parity claim (see
[Architecture: ORE as a dependency](../concepts/architecture.md#ore-as-a-dependency)). An
independent implementation was checked against `ORE.JamshidianSwaptionEngine.NPV()`
across payer and receiver swaptions, in-the-money/at-the-money/out-of-the-money cases,
and several tenors and forward-start dates, matching to a relative precision of
1e-6 or better.

## The pipeline, step by step

### 1. Describing a swaption: `SwaptionConfig`

A `SwaptionConfig` describes one swaption: the underlying swap's terms (`notional`,
`fixed_rate`, `payer`, `swap_tenor` — the same meaning as
[`SwapConfig`](swaps.md#1-describing-a-swap-swapconfig)), which simulated
Hull-White rate factor prices it (`rate_factor_index`), that factor's own model
parameters (`hw_a`, `hw_sigma`, `initial_zero_curve` — see
[below](#why-a-swaption-needs-its-own-copy-of-the-models-parameters)), and, optionally,
`forward_start` — how far in the future the option can first be exercised. See
[API Reference](../reference/api-reference.md#swaptionconfig) for every field.

**Why a swaption needs its own copy of the model's parameters.** Unlike the linear swap
pricer, which only needs *discount factors* (already baked into the yield curve cube),
Jamshidian's trick needs the underlying Hull-White model's own `a` (mean reversion speed)
and `sigma` (volatility) parameters directly — they determine how much uncertainty there
is left between now and the exercise date, which is exactly what the option's value
depends on. There is no way to recover these from the simulated rate paths alone, so they
must be passed in explicitly, matching the same values used to configure that rate
factor in the simulation's own `RatesConfig` (see
[Market Simulation](../concepts/market-simulation.md)).

**Single-model pricing, not multi-curve.** Unlike the swap pricer's independent
`discount_curve_index`/`forward_curve_index` split, Jamshidian's trick prices the
underlying swap and the option itself off **one** Hull-White factor
(`rate_factor_index`). This isn't a simplification — `ORE.JamshidianSwaptionEngine`
itself only ever takes a single `ShortRateModel`, since the bond-option decomposition is
intrinsically tied to one model's own affine bond-price dynamics. There's no
multi-curve version of this formula in ORE either.

### 2. Building the real trade: `_build_ore_swap()`, `prepare_swaption()`

Exactly like [the swap pricer](swaps.md#2-building-the-real-trade-_build_ore_swap-prepare_swap),
the underlying swap's schedule, day-count accrual, and coupon amounts are built with
ORE's own `MakeVanillaSwap` machinery rather than reimplemented — see
[Instruments: using ORE for the fiddly parts](swaps.md#why-its-built-this-way-using-ore-for-the-fiddly-parts)
for why. `prepare_swaption()` additionally extracts two dates unique to a swaption:

- **The exercise date `T0`** — when the option holder must decide whether to exercise.
  Conventionally 2 business days before the underlying swap's own accrual begins (the
  same spot-lag convention `MakeVanillaSwap` itself applies), computed from
  `evaluation_date + forward_start`, not by working backwards from the underlying swap's
  own (business-day-adjusted) start date — see the next point for why that distinction
  matters.
- **The underlying swap's own accrual start date `T_start`.**

**A bug this distinction caught during development.** An earlier version of this module
assumed `T_start` always equals `T0` — true only for a swaption with no `forward_start`
(where the 2-day spot lag and the "exercise lag" happen to coincide), but false for any
genuinely forward-starting swaption, where `T_start` is 2 business days *after* `T0`, not
equal to it. That assumption showed up as an ~1% NPV mismatch against
`ORE.JamshidianSwaptionEngine` for a forward-starting test case — traced down to the fact
that the underlying's floating leg doesn't redeem its notional exactly at `T0`, but at
`T_start`, which is a real (if small) discount factor away, not an identity. See
[the mathematics section below](#why-t_start-matters-the-floating-legs-notional-timing)
and `tests/test_european_swaption.py::TestAgainstOREJamshidianEngine::test_matches_ore_forward_starting`
for the regression coverage.

### 3. The closed-form building blocks: `compute_hw_A()`, `_hw_B()`, `_bond_option_sigma()`, `_bond_call()`/`_bond_put()`

These implement the Hull-White 1-Factor closed forms Jamshidian's trick is built from —
the same `A(t,T)`/`B(t,T)` affine bond-price formula
[Market Simulation](../concepts/market-simulation.md#phase-3--yield-curve-reconstruction) already
uses to reconstruct discount factors, generalized here to *any* `(t,T)` pair (not just
pre-tabulated maturity pillars), since Jamshidian's trick needs it evaluated at two
distinct anchor points per swaption (today → exercise date, and exercise date → each
coupon date):

```
P(t,T) = A(t,T) · exp(−B(t,T) · r(t))                         [affine bond price]

B(t,T) = (1 − exp(−a·(T−t))) / a

σ_P(t,T,S) = σ · B(T,S) · sqrt((1 − exp(−2a·(T−t))) / (2a))    [bond-option volatility]

BondCall(t,T,S,K) = P(t,S)·Φ(h) − K·P(t,T)·Φ(h − σ_P)
    where h = (1/σ_P)·ln(P(t,S) / (P(t,T)·K)) + σ_P/2
```

`BondCall`/`BondPut` (put via put-call parity) are the standard Black-formula-on-a-bond
closed form — live-verified bit-for-bit against `ORE.HullWhite.discountBondOption` in
this module's tests, not derived independently.

**A zero-volatility edge case this module handles explicitly.** `σ_P` is exactly zero
whenever a bond's own maturity `S` coincides with the option's expiry `T` — which is
*always true* of the notional-redemption leg for a non-forward-starting swaption (see
above: `T_start == T0` there). A naive Black formula divides by `σ_P`, producing `NaN` at
exactly this point. `_bond_call`/`_bond_put` handle this directly (not left to the
caller) by falling back to the deterministic zero-volatility limit,
`max(P(t,S) − K·P(t,T), 0)` — the option's plain intrinsic value, since there's no time
left for uncertainty to matter.

### 4. Solving the exercise boundary: `_solve_rstar()`

```python
def _solve_rstar(coupon_bond_value_fn, t_shape, iterations=100) -> jax.Array:
```

Finds Jamshidian's critical short rate `r*` — the rate at which the (signed) coupon bond
is worth exactly zero — via **vectorized bisection**, run once per `(scenario, time
step)` pair simultaneously under `jax.jit`, directly implementing the root's
[README.md](../../README.md) goal of "vectorize the calculation of the exercise boundary."
Bisection (rather than Newton's method) is used because it needs no derivative and
converges within a fixed, data-independent number of iterations — which is what makes it
`jax.jit`-friendly (JAX requires a fixed amount of work per compiled call, not a
data-dependent "stop when converged" loop).

### 5. Why T_start matters: the floating leg's notional timing

The par-floating-leg identity this module relies on (see
[Instruments: multi-curve discounting](swaps.md#2-building-the-real-trade-_build_ore_swap-prepare_swap))
is, conditional on the exercise date `T0`:

```
FloatLegPV(T0) = notional · ( P(T0, T_start) − P(T0, T_last) )
```

— receive the notional at the swap's own accrual start `T_start`, pay it back at the
final date `T_last`, with every intermediate reset cancelling out. Jamshidian's exercise
equation is then: the fixed leg's coupon bond (every fixed cashflow, plus the final
notional) is worth exactly the *received* notional leg at the exercise boundary:

```
Σᵢ cᵢ·P(T0,Tᵢ;r*)  +  notional·P(T0,T_last;r*)  =  notional·P(T0,T_start;r*)
```

This module solves this directly by folding the right-hand side into the same signed sum
as every other leg — the `T_start` notional receipt enters with a **negative** cashflow
amount, flowing through the exact same bond-option machinery used for every other leg,
rather than as a special case. See `_price_one_swaption`'s docstring for the full
derivation and `prepare_swaption`'s docstring for why `T_start` can't be assumed equal to
`T0`.

### 6. Conditional (future-time) pricing

Like every other pricer in this codebase, `price_swaptions` must return a
`[Scenarios, TimeSteps, Trades]` NPV cube — the trade's value in *every* simulated
future, not just today. This means Jamshidian's formula needs to be evaluated not just at
`t=0`, but conditional on every simulated `(scenario, time step)`'s own short rate.

This falls directly out of the Hull-White model's **Markov property**: the same
`P(t,T) = A(t,T)·exp(−B(t,T)·r(t))` formula above holds with *any* `t` (not just `t=0`)
as the conditioning point, using that scenario/step's own simulated short rate `r(t)` in
place of today's. Jamshidian's decomposition applies verbatim — solve `r*` at the
exercise date `T0` exactly as before (this doesn't depend on `t`, since it's a property
of the exercise boundary itself), then price every bond-option leg *as seen from* `t`
using its own conditional bond prices.

**This generalization was itself live-verified against ORE**, not just assumed correct
by mathematical argument: by rebuilding ORE's own evaluation date and implied yield curve
at a later point in time and re-pricing a still-alive swaption with a fresh
`JamshidianSwaptionEngine`, matching this module's conditional formula to the same
~1e-6 relative precision as the `t=0` case. See
`tests/test_european_swaption.py::TestConditionalPricingAndExpiry::test_conditional_pricing_matches_ore_rebuilt_at_later_date`.

**Once `t` reaches the option's own exercise time `T0`, NPV is reported as exactly 0** for
that `(scenario, step)` — a European option carries no remaining value after its own
expiry (it has either been exercised or has lapsed). This mirrors ORE's own
`Instrument.NPV()` convention of reporting 0 post-expiry, rather than raising an error.

## Tested by

- `tests/test_european_swaption.py::TestAgainstOREJamshidianEngine` — direct numeric
  comparison against `ORE.JamshidianSwaptionEngine.NPV()`, covering payer/receiver,
  ITM/ATM/OTM, several tenors, and both spot-starting and forward-starting swaptions
  (the case that caught the `T_start` bug described above).
- `TestZeroVolatilityLimit` — the `σ_P == 0` edge case's fallback to the intrinsic-value
  formula, tested directly against the closed-form building blocks.
- `TestZeroVolatilityCollapsesToIntrinsic` — an end-to-end check that a near-zero-volatility
  swaption collapses to `max(swap NPV, 0)`, live-verified against ORE.
- `TestPayerReceiverParity` — the model-independent put-call-parity identity
  (`payer − receiver == forward swap value`), an independent sanity check that doesn't
  rely on the Jamshidian formula itself being correct.
- `TestConditionalPricingAndExpiry` — the future-time conditional pricing generalization
  described above, plus the post-expiry-is-zero and still-alive-is-positive properties.
- `TestPriceSwaptionsShape` — output shape correctness and that `rate_factor_index`
  genuinely selects the right simulated rate factor.
- `TestMonotonicRStarSolve` — confirms the vectorized bisection converges to a sane,
  finite result across a spread of tenors and forward-start dates.
