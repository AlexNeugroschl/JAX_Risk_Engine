# Instruments: European Swaptions

**Modules:** [`engine/instruments/european_swaption.py`](../../engine/instruments/european_swaption.py)
(the trade), [`engine/pricing/european.py`](../../engine/pricing/european.py) (Bachelier,
the default engine), [`engine/pricing/jamshidian.py`](../../engine/pricing/jamshidian.py)
(Jamshidian, the configured alternative)
**Entry points:** `price_portfolio`, or `value_today` / `value_portfolio`; the engine is
`PricingConfig.european`

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

The engine answers the same kind of question as for a [swap](swaps.md) — *"given thousands
of simulated alternate futures, what is this specific trade worth in each one, at each point
in time?"* — producing its column of the same NPV cube.

## Two engines

- **Bachelier on the market volatility** (`european="Bachelier"`, the default): ORE's
  `EuropeanSwaptionEngineBuilder` → `BlackMultiLegOptionEngine` on the market's ATM normal
  swaption volatilities, today and on every path. It supports a floating spread (folded into
  the strike) and cash settlement (ORE's `ParYieldCurve` annuity), and has Vega on every quote
  it reads.
- **Jamshidian** (`european="Jamshidian"` with `jamshidian=JamshidianEngineConfig(a, σ)`):
  QuantLib's `JamshidianSwaptionEngine` on a Hull-White model with the configured reversion
  and volatility. ORE has no builder for it; it is kept as an option (decision A-1). It
  refuses what QuantLib's engine refuses: a floating spread ([I-37](../planning/known-issues.md#i-37)),
  a non-positive reversion ([I-41](../planning/known-issues.md#i-41)), cash settlement
  ([I-52](../planning/known-issues.md#i-52)). It reads no volatility quote, so it has no Vega.

Both price whichever interest-rate model simulates the paths: they read only the path's
curves. The Jamshidian engine's Hull-White model is its own; when it equals the simulated
Hull-White model, the engine's value on a path is that model's own conditional price, and the
deflated value is a martingale (`tests/test_hull_white_model.py::TestEuropeansOnTheMarketVolatility`).

## The pipeline, step by step

### 1. Describing a swaption: `SwaptionConfig`

The trade: `trade_id`, `evaluation_date`, the underlying swap's terms as for a
[swap](swaps.md#1-describing-a-swap-swapconfig) (`notional`, `fixed_rate`, `payer`, `currency`,
`index_tenor_months`, `floating_spread`, the schedule), the exercise (`exercise_date`, or
`forward_start` with an exercise lag), and `settlement` (`"Physical"` or `"Cash"`). It
carries no curve, volatility or model: those are the market's and the configuration's.

### 2. Building the real trade: `underlying_swap()`, `european_terms()`

`underlying_swap(cfg)` builds the underlying with ORE's `MakeVanillaSwap`.
`european_terms(cfg, date)` keeps the coupons `BlackMultiLegOptionEngine` reads: those paying
after expiry and accruing from it, with the swap's start time Tv and its nominal.

### 3. Bachelier (`engine/pricing/european.py`)

```
annuity  = |fixed BPS| = Σ_i N τ_i P(T_i)
forward  = Σ_k N F_k δ_k P(T_k) / annuity            (float leg without spread)
strike   = K − spread · (Σ_k N δ_k P(T_k)) / annuity
variance = vol(expiry, swap length)² · t(expiry)
NPV      = annuity · Bachelier(call for a payer, put for a receiver)
```

A cash-settled option uses the `ParYieldCurve` annuity: the fixed leg discounted at the
forward swap rate from the earliest accrual start. On a path the volatility is today's
surface seen from the path date (`DynamicSwaptionVolatilityMatrix`, `ForwardVariance` by
default).

## Why it's built this way: Jamshidian's trick

Pricing an option generally requires either a closed-form formula (fast, but only exists
for specific, simple cases) or a nested simulation from every point a decision might be made
(always works, but extremely slow). Under a one-factor model like Hull-White, **Jamshidian's
trick** gives a closed form for a European swaption: there is one critical state x* at which
the swap's fixed coupons, as a bond, are worth exactly its nominal; exercising is worthwhile
on one side of x* only, so the option on the whole coupon bond splits into a portfolio of
options on single zero-coupon bonds, each struck at that bond's value at x*, each with a
closed-form Black price.

### 4. The Jamshidian engine (`engine/pricing/jamshidian.py`)

The engine is QuantLib's decomposition written in the LGM form of the Hull-White model
(H(t) = (1 − e^{−at})/a, ζ(t) = σ²(e^{2at} − 1)/(2a)): on a valuation date, with every time
measured from it and P(T) the curve's discount factor,

```
Σ_i c_i K_i(x*) = N,    K_i(x) = P(T_i)/P(Tv) · exp(−(H_i − H_v) x − ½ (H_i² − H_v²) ζ(T0))
```

and each c_i is priced as an option on the forward bond P(T_i)/P(Tv) struck at K_i, Black on
the forward with standard deviation |H_i − H_v| √ζ(T0): puts for a payer, calls for a
receiver. This is QuantLib's formula term by term (its `discountBond(T0, T, r)` carries the
forward f(0, T0) into K_i's numerator and denominator alike), so the engine reads only
discount factors and prices on any curve: today's, a bumped one, a path's.

x* is solved to float64 rounding by the engine's root solver (`JamshidianEngineConfig.solver`:
`"Newton"` by default, `"Bisection"` the reference; [the root solver](../reference/calibration.md#the-root-solver)),
where QuantLib's Brent stops at 1e-8. A solver's iterations carry no derivative of the root, so
its tangent is the implicit function theorem's (`engine.solvers.roots.implicit_root`, a
`jax.custom_jvp`), and AD Greeks differentiate through the root.

### 5. Why T_start matters: the floating leg's notional timing

The floating leg of the swap entered at exercise is worth par on a single curve: N at its
start Tv, less N at its end, with every intermediate reset cancelling out. So the coupon bond
the exercise boundary compares with the nominal is measured in units of the bond maturing at
Tv, not at the exercise date T0: K_i above is P(T0, T_i)/P(T0, Tv). Using T0 instead of Tv
(a forward start's two settlement days) gave a ~1% mismatch against QuantLib before it was
found. QuantLib's engine is single-curve: the floating leg is at par on the discount curve,
and the index's forwarding curve is not read.

### 6. Conditional (future-time) pricing

Every pricer must give the trade's value on every path and date, not only today: on a path
the engine is evaluated on that path's scenario curve, with every time measured from the
path date. The Hull-White model is time-homogeneous with a constant volatility, so the
decomposition applies verbatim on the later date; once the exercise date has passed the
option is 0 and, if physically settled, the path carries the swap it entered
(`engine.pricing.options`, ORE's `OptionWrapper`).

## Tested by

- `tests/test_pricing.py::test_european_today_equals_ores_default_engine` and
  `::test_european_on_every_path_equals_quantlibs_bachelier_engine` — Bachelier against ORE
  (1e-10) and QuantLib's `BachelierSwaptionEngine` on every path, under both models;
  `::test_a_cash_settled_european_uses_the_par_yield_annuity` (2e-14).
- `tests/test_jamshidian.py` — the Jamshidian engine against `ORE.JamshidianSwaptionEngine`
  (`TestAgainstQuantLibJamshidianEngine`, within QuantLib's Brent tolerance on its root, 5e-6),
  against QuantLib's formula at the exact root (`TestAgainstQuantLibWithAnExactRoot`, 1e-10),
  limits, a later date's curve (`TestOnALaterDatesCurve`), the configuration and refusals
  (`TestConfiguration`), trade validation.
- `tests/test_end_to_end.py` — the Hull-White simulation's paths priced by QuantLib's
  Jamshidian engine on each path's curve, pathwise and in VaR/ES.
- `tests/test_greeks.py::TestJamshidianRootDerivative` — the implicit-function derivative of
  x*, first and second order.
