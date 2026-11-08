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
| — | `config.pricing.{bermudan,american}.swap_index_from_trade: true` | Each option calibrates on its own underlying's conventions | — |

ACT/360 and 30/360 are accrual day counts of every product (`SUPPORTED_ACCRUAL_DAY_COUNTS`),
so the existing ORE-parity matrix and sweep cover them on term-index swaps, bonds and
calibration helpers too. An `overnight_index` swap accrues on its index's day count; naming a
different one is refused.

## R6 timings

R6_TIMINGS

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
