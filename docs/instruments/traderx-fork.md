# The TraderX Fork: Products, Conventions and Remaining Gaps

**Branch:** `traderx/engine-fork`, cut from upstream `17ac757` with Alex's permission
(2026-11-07). **Tests:** [`tests/test_traderx_products.py`](../../tests/test_traderx_products.py).
Every expected value there is QuantLib's or ORE's (the ORE wheel this environment pins), never
the engine's own output.

TraderX's end-of-day VaR/ES sends one `POST /runs` (`npv` + `market_risk`) with every
position and OTC contract it books. Its November 6 requests R1–R6 named what this engine could
not price. This page lists what the fork added for them, the input each one takes, and every
TraderX class or field still refused or mapped lossily.

## What the fork adds

| Request | Input | Priced by | Reference (test) |
|---|---|---|---|
| R1 cash equity / ETF | `{"trade_type": "equity", "symbol", "quantity", "multiplier", "currency"}`; spot from `market.equities[symbol].spot` | `units x spot`; market-risk factor `equity:<SYM>/spot`, a move of the log spot | NPV 15,000; `VaR_99` 1,946.92 (closed form) |
| R2 listed equity option | `{"trade_type": "equity_option", "symbol", "option_type", "strike", "expiry_date", "exercise", "quantity", "multiplier"}`; `market.equities[symbol].volatility` and `dividend_curve` | Crank-Nicolson on log spot (`engine/pricing/equity.py`), American or European; one grid run per listed contract per run | QuantLib `FdBlackScholesVanillaEngine` (American), `AnalyticEuropeanEngine`; 720.18 per contract, delta 0.5401 |
| R3 OIS swaps and swaptions | `overnight_index`: `USD-SOFR`, `EUR-ESTR`, `GBP-SONIA`, `JPY-TONA`, on swaps and all three swaption styles; `index_tenor_months` 12 or 3 | Both legs on the index's day count, schedule on its calendar (Modified Following, forward), paid `payment_lag_days` late, coupon compounded in arrears (no lookback or lockout) | QuantLib `OvernightIndexedSwap` per convention; SOFR 5Y −32,453.80; `BachelierSwaptionEngine` on the OIS |
| R3 seasoned OIS | `overnight_fixings: {"YYYY-MM-DD": rate}` on a swap: the index's daily fixings, every past day of the coupon accruing on the evaluation date | The coupon in progress pays the realised compounding times the forecast from the first unfixed day (`schedules.build_overnight_swap`) | QuantLib `OvernightIndexedSwap` with the same fixings in its index history: in progress (annual and quarterly) and accrued but unpaid |
| R3 mid-life swaptions | none | A Bermudan or American past its first exercise date exercises into the coupons after a later exercise date, as before; it is no longer refused while its underlying accrues | Equal to the forward-starting option on the remaining dates |
| R4 corporate bond | `credit_spread` (flat, continuous) on `bond`; `accrual_day_count: "30/360"` (US bond basis) | Discount curve plus the spread on every flow | QuantLib: dirty 1,049,185.36; 1,145,157.51 on the Treasury curve |
| R5 American at least the Bermudan | `config.pricing.american.exercise_on_period_starts: true` | The American's exercise grid also holds every underlying period start in its window | The Bermudan on those dates bounds it from below (ORE's default grid stays the default) |
| R6 compile time | none | Market risk assembles the `[scenario, trade]` P&L on the host; the exposure cube is filled column by column into a preallocated host buffer | See [R6 timings](#r6-timings) |
| Batching (TraderX's own, not an R request) | none | On an accelerator, market risk revalues trades of one shape together: their terms stacked, each program mapped over them, one call and one host sync per group of up to `trades_per_call`; a CPU still takes one trade a call | See [Same-shape trades in one call](#same-shape-trades-in-one-call) |
| — | `config.pricing.{bermudan,american}.swap_index_from_trade: true` | Each option calibrates on its own underlying's conventions | — |

ACT/360 and 30/360 are accrual day counts of every product (`SUPPORTED_ACCRUAL_DAY_COUNTS`),
so the existing ORE-parity matrix and sweep cover them on term-index swaps, bonds and
calibration helpers too. An `overnight_index` swap accrues on its index's day count; naming a
different one is refused.

## R6 timings

Measured 2026-11-07 on an 11-core Mac (CPU, JAX 0.10.2, float64), each figure a fresh process
with the persistent compilation cache off; another engine run held about 1.7 cores throughout.

Joining N `[256]` float64 P&L columns, first call and cached:

| Operands | Upstream `jnp.stack`, first | cached | Fork (host), first | cached |
|---|---|---|---|---|
| 1,000 | 3.30 s | 0.003 s | 0.010 s | 0.004 s |
| 2,000 | 12.8 s | 0.010 s | 0.010 s | 0.003 s |
| 4,000 | 62.1 s | 0.016 s | 0.025 s | 0.016 s |
| 8,000 | 358 s (XLA's "Very slow compile" alarm) | 0.035 s | 0.022 s | 0.011 s |

One in-process `npv` + `market_risk` run (256 Monte Carlo scenarios) of TraderX's large book
tiled x10 (bonds, GBP/JPY swaps, European swaptions), its first N trades:

| Trades | Wall, cold | Peak RSS |
|---|---|---|
| 1,000 | 9.0 s | 605 MiB |
| 2,000 | 24.6 s | 866 MiB |
| 4,000 | 26.9 s | 878 MiB |
| 8,000 | 33.4 s | 900 MiB |
| 26,220 | 72.6 s | 845 MiB |

The 26,220-trade run was still compiling after 22 minutes upstream. Its `VaR_99`,
449,939,764.42, is the one the same request gave through the HTTP API on the first fork
commit. The exposure cube's assembly (`engine.pricing.cube._output_cube`) had the same jitted
stack of T operands and is filled the same way now.

## Same-shape trades in one call

`engine.risk.market.revaluation.revalue_trades` (and `run_market_risk`, through it) groups trades
by `engine.risk.greeks.scenario_pnl.group_key`: one product, pricer, curves and arithmetic, and
terms of one pytree structure and leaf shape. `price_function_pnls` stacks a group's terms on the
host, padded to a power of two with copies of the last trade, and maps the P&L program over the
trades (`jax.vmap` inside its `jax.jit`), and the price too where the product does not
recalibrate. A recalibrated option's price and recalibration stay per trade: see the A100
measurement below. A Python-scalar term (a
Bermudan's `float_spread`) is stacked weakly typed, so a float32 option stays float32. An equity
option is still revalued once per listed contract. A trade with stochastic rounding is revalued
alone (its key is its own).

**Why.** On the A100 (2026-11-08, fork `a5bd7c3`), the warm book spent its time in a per-trade
loop: a program call and a host sync for each of about 8,100 revaluations, with each
swaption's induction a few milliseconds on the device. Grouped, TraderX's 11,360-trade book
is 486 groups (Bermudans 75, Americans 55, swaps 90, bonds 29, cash equities 50, Europeans
50, equity-option contracts 39).

**A CPU takes one trade a call** (`trades_per_call`). On an 11-core Mac, four Bermudans
recalibrated under 256 scenarios in one call took 2.4 times the CPU time of four calls, and
four calls' time at 64 scenarios a batch: the call's working set, not the program, set the cost.
A CPU's per-call cost on the host is small, so a group has nothing to save there. A group of
one runs the trade's own programs, unmapped: mapped over one trade, a recalibrated Bermudan's
P&L took twice the CPU time (2.5 s against 5.0 s, warm). So a CPU run's programs are the fork's
before batching, and its answers are `a5bd7c3`'s bit for bit: every per-trade P&L, base value
and `VaR_99` of 20 trades of each of the seven classes TraderX books, at 256 scenarios. On an accelerator a call takes up to `MODELS_PER_CALL` (16,384) trade-scenario
pairs, or what the grid engine's per-device induction budget allows (16 Bermudans or Americans,
64 equity options a call on an A100 40 GB).

**On an A100** (2026-11-08, fork `640a989`, warm, 256 scenarios). One trade revalued as a group
against the same number of calls:

| Product | Group of 4 | 4 calls | Group of 16 | 16 calls |
|---|---|---|---|---|
| Bermudan (recalibrated) | 0.19 s | 0.41 s | 0.43 s | 1.55 s |
| American (recalibrated) | 1.99 s | 3.41 s | 6.67 s | 13.56 s |
| Equity option | 0.16 s | 0.55 s | 0.39 s | 2.18 s |

Three distinct Bermudans of one shape: 0.18 s as a group, 0.28 s as three calls. The mapped
induction pays on the GPU, though not on a CPU.

**A run's second pass compiled again,** fixed since. With the price and the recalibration
mapped outside a jit (`jax.vmap` called eagerly, as the recalibration is a program per bucket,
never one program), the 20 Bermudans' second `run` on the A100 compiled for 117 s (533 s cold,
136 s second, 4.1 s third). The recompiled programs were exactly those the eager map reached
(`_bootstrap_bucket`, `_induction`, `_misses`), with the signatures of the first run's. Now a
recalibrated option prices and recalibrates per trade and its P&L alone is mapped; another
product's price is mapped inside a jit (`_prices`). With grouping forced on the Mac, the
sample's second run compiled for 28–39 s before the fix and not at all after. Three Bermudans
do not reproduce it (the first build kept or freed), so no unit test guards it: the check is the
20-trade run, `run` three times with the compile events counted.

The book itself is not yet measured grouped on a GPU.

**Same answers in a group.** Against fork `a5bd7c3`, on 20 trades of each class at 256 scenarios (the
seven classes TraderX books) revalued in groups on the CPU (before it took one trade a call),
every per-trade P&L agrees within 5e-11 of that trade's largest
(Americans; Bermudans 3e-14, the linear classes 7e-16, cash equities and swaps bit for bit) and
`VaR_99` within 2e-13. `TestGroupedRevaluation` checks a group against each trade revalued alone
at float64 and float32, a short Bermudan among long ones.

## Remaining gaps

What TraderX books (its risk extract: positions `EQUITY`, `OPTION`, `TREASURY`, `CORPORATE`;
contracts `SWAP` and `SWAPTION` on five conventions, three exercise styles), and what of it the
engine still refuses or holds fixed.

| TraderX class or field | Status | Why not closed |
|---|---|---|
| Equity option volatility and dividend yield in VaR/ES | Held fixed: only the spot and the discount curve move (a run warning names the trades) | The engine's market risk shocks curve pillars and spots only; swaption volatilities are held the same way upstream. A volatility factor needs a volatility history TraderX does not have (its market is synthetic) |
| Corporate credit spread in VaR/ES | Held fixed (a run warning names the trades) | No credit-spread history to shock; TraderX sends one flat synthetic spread per corporate. An issuer curve is not in the extract either |
| Equities, equity options and spread bonds on simulated paths (`exposure`) or as Greeks (`sensitivities`) | Refused (`t0_only`) | TraderX's EOD run asks for `npv` and `market_risk` only. Needs an equity path pricer and a credit model on paths |
| Seasoned OIS daily fixings | Priced when sent; refused, naming the first missing day, when not | TraderX has no overnight-fixings source yet: its side must send them (a feed, or flat synthetic fixings for the proof) |
| Floating spread on an OIS coupon in progress | Refused | TraderX books no spread. Carrying it needs a per-coupon spread, which the legs do not hold |
| Discrete dividends on listed options | Not an input (continuous yield only) | The extract has no dividend schedule; TraderX sends a flat yield |
| `counterpartyId`, `nettingSetId` | Not read by `npv`/`market_risk` | Exposure and XVA inputs; TraderX does not request them |
| Short swaptions | Supported (`long_short`) | TraderX's `/swaptions` has no short side; it sends `Long` |
| American exercise grid | ORE's default (24 a year) unless `exercise_on_period_starts` | Kept so upstream's ORE parity holds; TraderX should set the flag |

The extract's classes and fields not listed here price exactly as booked.

## Found on the way (upstream, not the fork)

- **Two American swaption parity misses on upstream `17ac757`.** Letting the ORE-parity sweep
  draw ACT/360 and 30/360 reaches two cases, `seed-24` (Greeks, an index delta) and `seed-39`
  (market-risk P&L), where an American swaption misses ORE by about 6e-5 relative, outside the
  sweep's tolerance. Both reproduce bit for bit on upstream with only its day-count list extended,
  and neither touches a fork product: an American exercise-grid tolerance edge upstream's own
  sample does not reach. The sweep keeps upstream's day counts (`SWEEP_DAY_COUNTS`), so its
  sample is upstream's.
- **`test_sharding`'s four-device run** fails on upstream `17ac757` itself: its FP8 stage asks for
  native execution on a CPU, which the precision policy refuses.
