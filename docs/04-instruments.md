# Instruments: Interest Rate Swaps

**Module:** [`engine/instruments/swap.py`](../engine/instruments/swap.py)
**Public entry point:** `price_swaps(yield_curves, maturities, swap_configs)`

## Plain-language summary

An **interest rate swap** is one of the most common trades in finance: two parties agree
to exchange interest payments on some notional amount of money for a set period, where
one side pays a **fixed** rate (agreed today, never changes) and the other pays a
**floating** rate (reset periodically based on where market interest rates actually end
up). Neither side ever exchanges the underlying notional amount itself — only the
interest payments. It's essentially a bet on which direction interest rates move: if
rates rise above what was fixed, the floating-rate payer comes out ahead; if they fall,
the fixed-rate payer does.

This module answers: *"given [thousands of simulated alternate futures for interest
rates](03-market-simulation.md), what is this specific swap worth in each of them, at
each point in time?"* The output is the **NPV cube** — a big table of "what this trade is
worth," organized by scenario and by time. That's the raw material [Stage 3](05-risk-statistics.md)
needs to compute risk numbers.

## Why it's built this way: using ORE for the fiddly parts

Pricing a swap correctly requires getting two very fiddly things exactly right, for
every single payment:
1. **Exactly which calendar dates does each side pay on?** ("Every 6 months, but
   adjusted if that date falls on a weekend or holiday, using this particular calendar
   and rounding rule...")
2. **Exactly how much time elapsed between two dates, for interest-accrual purposes?**
   This sounds simple but isn't — different markets use different conventions (e.g.
   "count every month as exactly 30 days" vs. "count the actual number of calendar
   days"), and getting the wrong convention produces a real, model-independent pricing
   error.

Neither of these is a place where "reimplement it in JAX for GPU speed" makes sense —
they run **once per trade**, not once per simulated scenario, so there's no performance
benefit to a from-scratch implementation, and they're exactly the kind of thing where a
subtle bug would produce numbers that are wrong in a way that's hard to detect just by
looking at them. So this module uses [ORE](https://www.opensourcerisk.org/)'s own
trade-building code directly (`ORE.MakeVanillaSwap`, `ORE.Actual365Fixed`, and related
classes) to build the schedule and compute accrual fractions — see
[Architecture: ORE as a dependency](02-architecture.md#ore-as-a-dependency) for the
broader design rationale. Only the actual "add up the simulated cashflows" math is
custom, GPU-accelerated JAX code.

## The pipeline, step by step

### 1. Describing a swap: `SwapConfig`

A `SwapConfig` describes one trade: how much money (`notional`), what fixed rate is
being paid, whether *this side* of the deal is paying fixed or receiving it (`payer`),
how long the swap runs (`swap_tenor`), and — importantly — *which* of the simulated
interest rate curves from [Stage 1](03-market-simulation.md) should be used to discount
this swap's cashflows versus to figure out its floating payments (`discount_curve_index`,
`forward_curve_index`). See [API Reference](07-api-reference.md#swapconfig) for every
field.

**Why two separate curve indices?** In modern practice, the interest rate used to
*discount* a cashflow back to today (usually an overnight/OIS rate) is not necessarily
the same rate used to figure out what a *floating* payment will actually be (usually
tied to a specific lending benchmark). This is called **multi-curve discounting**, and
it's standard in real trading desks since the 2008 financial crisis. This module supports
it directly by letting a swap point at two different entries in the simulated yield curve
cube's rate-factor axis — mirroring ORE's own `DiscountingSwapEngine` (one discount
curve) plus `IborIndex` (its own, separate forwarding curve) split exactly. If a swap
should use the same curve for both, `discount_curve_index` and `forward_curve_index` are
simply set to the same value.

### 2. Building the real trade: `_build_ore_swap()`, `prepare_swap()`

```python
def _build_ore_swap(cfg: SwapConfig) -> ORE.VanillaSwap:
```
Constructs an actual `ORE.VanillaSwap` object via `ORE.MakeVanillaSwap` — this is what
generates the real payment schedule (which dates, how many payments) using ORE's own
calendar/schedule logic. Both legs are explicitly set to use the `Actual/365 Fixed`
day-count convention (rather than leaving it to whatever a given interest rate index's
implicit default happens to be, which varies unpredictably by currency/index — e.g. some
default conventions use `30/360` for one leg and `Actual/360` for the other). Making
this an explicit, deliberate choice avoids surprising behavior.

```python
def _fixed_leg_cashflows(...) / _floating_leg_cashflows(...) -> _LegCashflows
```
Walks the ORE-built schedule and extracts, per payment: when it happens (converted into
"years from today," to match the simulation's own time axis) and how much time it
accrues over (again, using ORE's own `accrualPeriod()` — the convention-correct answer,
not a hand-rolled one).

```python
def prepare_swap(cfg: SwapConfig, maturities: np.ndarray) -> _PreparedSwap:
```
Ties the above together and resolves every cashflow date onto a specific index into the
simulation's `maturities` array (see
[the maturity-pillar-alignment requirement](#a-known-limitation-maturity-pillar-alignment)
below). This whole step is a one-time, CPU-only, per-trade setup cost — it does **not**
run per scenario, which is why it's fine for it to call into ORE (a regular Python
library, not a GPU-friendly one).

### 3. Actually pricing it: `_price_one_swap()`, `price_swaps()`

This is the only part of the module that touches the simulated data — and it's a pure,
vectorized JAX computation, with no ORE calls and no Python loops over scenarios.

**Fixed leg** — straightforward: for each payment, `notional × fixed_rate × accrual
fraction`, discounted back to today using the appropriate entry in the yield curve cube,
summed across all payments:

```
Fixed leg value(t) = Σᵢ  notional · fixed_rate · accrualᵢ · P_discount(t, Tᵢ)
```

**Floating leg** — the payment amount isn't known in advance; it depends on what the
simulated interest rate actually turns out to be between two dates. This is computed
directly from the simulated forwarding curve, using the standard **simple forward rate**
formula:

```
F(t; Tᵢ₋₁, Tᵢ) = ( P_forward(t, Tᵢ₋₁) / P_forward(t, Tᵢ)  −  1 )  /  accrualᵢ

Float leg value(t) = Σᵢ  notional · (F(t; Tᵢ₋₁, Tᵢ) + spread) · accrualᵢ · P_discount(t, Tᵢ)
```

This uses one forward rate per accrual period ("at-par" coupon pricing), matching ORE's
own default convention (`IborCoupon.usingAtParCoupons()`).

**Combining the two legs:**
```
NPV(t)  =  Float leg value(t) − Fixed leg value(t)      if paying fixed
        = −(Float leg value(t) − Fixed leg value(t))    if receiving fixed
```

Because a swap has no optionality — nobody gets to choose whether to exercise it — its
value at any future scenario/step is just the expected value of its remaining cashflows
computed with *that* scenario's simulated curve. No nested simulation is needed (unlike,
say, an option, where you'd need to simulate what happens *after* a decision point too).
This is also why this module doesn't use the "numéraire" that
[Stage 1](03-market-simulation.md#phase-2--the-cross-asset-model-engine) produces —
discounting directly off the simulated yield curve cube is simpler and exactly
equivalent for a linear instrument like this.

```python
def price_swaps(yield_curves, maturities, swap_configs: List[SwapConfig]) -> jax.Array:
```
The public entry point: prepares and prices every swap in the list, and stacks the
results into one NPV cube shaped `[Scenarios, TimeSteps, Trades]` — the standard shape
[Stage 3](05-risk-statistics.md) expects from *any* pricer, not just this one.

## A known limitation: maturity-pillar alignment

The simulated yield curve cube ([Stage 1](03-market-simulation.md)) only contains
discount factors for a specific, fixed list of future dates (`maturities`) — it doesn't
support asking for an arbitrary date that wasn't explicitly requested when the simulation
was configured. This means **every one of a swap's payment/accrual dates must land
exactly on one of the simulation's configured `maturities`** — there's no curve
interpolation between pillars yet. `_maturity_indices()` enforces this strictly: it
raises a clear `ValueError` (rather than silently using the nearest available date, which
would be a subtle pricing error) if a swap's dates don't line up.

In practice, this means: when setting up a simulation that's meant to price a specific
swap, `config.rates.maturities` must be set to the union of every payment/accrual date
that swap will need. See the [User Guide](06-user-guide.md#pricing-a-swap) for a worked
example, and
[`engine/scenarios.py`](../engine/scenarios.py)'s `SWAP_DEMO_MATURITIES` for a concrete
instance of this being done correctly.

**A subtle correctness detail this limitation caught:** the original lookup logic
checked whether a date was valid using a *clipped* array index, but then returned the
*unclipped* one — meaning a cashflow date landing just barely past the last available
pillar (e.g. from ordinary floating-point rounding) could silently pass validation while
still producing an out-of-bounds index, which JAX does not raise an error for by default
(it silently clips to the nearest valid index instead). This has been fixed, and
`tests/test_swap.py::TestMaturityIndicesBounds` locks the fix in.

## Tested by

- `tests/test_swap.py::TestPriceSwapsAgainstORE` — builds the *same* swap
  two ways (once through this module, once through `ORE.VanillaSwap` +
  `ORE.DiscountingSwapEngine` directly) and asserts the resulting NPVs match to `1e-6`
  relative tolerance. Covers a payer swap, a receiver swap, a swap priced at its exact
  par rate (which should be worth ~0), and confirms paying vs. receiving are exact
  negations of each other.
- `tests/test_swap.py::TestPriceSwapsShape` — output shape correctness and
  the maturity-mismatch error path.
- `tests/test_swap.py::TestMaturityIndicesBounds` — the out-of-bounds-index
  regression described above.
