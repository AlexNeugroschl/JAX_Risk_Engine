# Instruments: Interest Rate Swaps

**Modules:** [`engine/instruments/swap.py`](../../engine/instruments/swap.py) (the trade),
[`engine/pricing/legs.py`](../../engine/pricing/legs.py) (its valuation)
**Entry points:** `price_portfolio` (with every other trade), or `value_today` /
`value_portfolio` (`engine/pricing/cube.py`)

## Plain-language summary

An **interest rate swap** is one of the most common trades in finance: two parties agree
to exchange interest payments on some notional amount of money for a set period, where
one side pays a **fixed** rate (agreed today, never changes) and the other pays a
**floating** rate (reset periodically based on where market interest rates actually end
up). Neither side ever exchanges the underlying notional amount itself — only the
interest payments. It's essentially a bet on which direction interest rates move: if
rates rise above what was fixed, the floating-rate payer comes out ahead; if they fall,
the fixed-rate payer does.

The engine answers: *"given [thousands of simulated alternate futures for interest
rates](../concepts/market-simulation.md), what is this specific swap worth in each of them, at
each point in time?"* — and what it is worth today. The output is its column of the **NPV
cube**, a big table of "what this trade is worth," organized by scenario and by date. That's the raw material risk aggregation (see
[VaR & Expected Shortfall](../risk/var_es.md)) needs to compute risk numbers.

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

Neither of these is a place where "reimplement it in JAX for accelerator speed" makes sense —
they run **once per trade**, not once per simulated scenario, so there's no performance
benefit to a from-scratch implementation, and they're exactly the kind of thing where a
subtle bug would produce numbers that are wrong in a way that's hard to detect just by
looking at them. So this module uses [ORE](https://www.opensourcerisk.org/)'s own
trade-building code directly (`ORE.MakeVanillaSwap`, `ORE.Actual365Fixed`, and related
classes) to build the schedule and compute accrual fractions — see
[Architecture: ORE as a dependency](../concepts/architecture.md#ore-as-a-dependency) for the
broader design rationale. Only the "discount the cashflows on each path's curves" math is custom, accelerator-run
(GPU/TPU) JAX code.

## The pipeline, step by step

### 1. Describing a swap: `SwapConfig`

The trade, as ORE's trade XML holds it: `trade_id`, `evaluation_date` (the market's as-of
date), `notional`, `fixed_rate`, `payer` (pays fixed), `currency` and `index_tenor_months`
(which name its discount curve and its Ibor index's forwarding curve in the `Market`),
`floating_spread`, `accrual_day_count`, the schedule (`swap_tenor` from spot, or
`effective_date`/`maturity_date`), and `fixings` (a coupon fixed before the evaluation date
needs its rate, as ORE requires). It carries no curve or model. See
[API Reference](../reference/api-reference.md#swapconfig-engineinstrumentsswap) for every field.

The dates are the trade: `swap_tenor="5Y"` is resolved once to the booked
`effective_date`/`maturity_date`, so `dataclasses.replace(swap, evaluation_date=later)` prices
the same swap on a later day, with the coupons it has already paid gone.

### 2. Building the real trade: `underlying_swap()`, `legs_of()`

`underlying_swap(cfg)` builds the swap with ORE's own `MakeVanillaSwap`
(`engine/instruments/schedules.py`), so the payment dates, accrual fractions and fixing dates
are exactly ORE's. `legs_of(swap, payer, asof, fixings)` turns its coupons into arrays: each
fixed coupon's pay time and amount, each floating coupon's pay time, accrual, its index's
forecast period (`par_coupon_forecast_period`) and fixing date, and any historical fixing.
This runs once per trade.

### 3. Pricing it: `legs_npv()`, `today_npv()`, `legs_cube()`

ORE's `DiscountingSwapEngine` with at-par Ibor coupons: each remaining coupon discounted on
the currency's discount curve, each floating coupon's rate forecast on the index's forwarding
curve. On a valuation date d (today, or a simulation date on one path):

- a flow paid on or before d has occurred and is dropped (`CashFlow::hasOccurred`);
- a floating coupon whose fixing date is before the as-of date pays its historical fixing;
- one that fixed during the simulation (between the as-of date and d) pays the rate
  `FixingManager` stored on that path when the simulation stepped over its fixing date
  (`path_fixings`);
- one fixing on or after d is projected off d's forwarding curve.

`today_npv` values today's market; `legs_cube` values every path and date at once, vmapped
over the dates. After maturity a swap is worth exactly 0. Under either interest-rate model
the curves are the scenario market's, so the pricer does not depend on the model.

## Tested by

- `tests/test_pricing.py::test_today_equals_ores_discounting_swap_engine` — today's value
  against ORE's `DiscountingSwapEngine`, including seasoned and ICMA swaps.
- `tests/test_pricing.py::test_every_path_and_date_equals_ores_discounting_swap_engine` and
  `::test_path_fixings_are_ores_index_forecast_on_the_path_curve` — every path and date
  against ORE on that path's curve, with `FixingManager`'s fixings (dates between fixing and
  payment, on a holiday, after maturity), under both interest-rate models.
- `tests/test_swap.py` — `TestAgainstORE` (payer, receiver, at par, spreads, to 1e-12),
  `TestNegativeRates`, `TestPortfolios`, `TestSwapConfigValidation`.
- `tests/test_trade_dates.py` — seasoned trades, historical fixings and Theta against ORE.
- `tests/test_hull_white_model.py::TestPaidFlowsAndMaturity` — a matured swap is worth 0 on
  every path.
