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
| Scenario draw (TraderX's own) | none | Monte Carlo scenarios are the Sobol normals times the covariance's symmetric root, so one request draws the same scenarios on every host | See [One draw on every host](#one-draw-on-every-host) |
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

**The book on an A100** (2026-11-08, fork `3e1d1d9`, TraderX's 11,360 trades at 256 scenarios,
one process, `npv` + `market_risk`), against the same book ungrouped that morning (`a5bd7c3`):

| Run | Ungrouped | Grouped |
|---|---|---|
| Cold | 1,571 s, about 983 s of it compile | 2,001 s, 1,444 s of it compile |
| Second | 575 s | 530 s (1 s of compile) |
| `VaR_99` | 73,560,770.0634258 | 73,560,770.0634258 (to 1e-15) |

Grouped, the book is 8% faster warm and 27% slower cold: a group compiles a program per padded
size as well as per shape, and the warm time left is mostly outside the grouped revaluation
(the per-trade work around it: building each trade's price function, the `npv` pass, each
column's storage and bounds).

**Where the warm time goes on an A100** (2026-11-08, fork `3e1d1d9`, every tenth trade of the
book, 1,136 trades, warm 62.9 s; each part blocking on its own device work):

| Part | Time | Share |
|---|---|---|
| `npv` pass (`base_npv`) | 30.6 s | 49% |
| of it, equity options valued one trade at a time (348 grid runs, 38 ms each) | 17.7 s | |
| of it, the curves read outside a jit (`log_discount`, 764 calls) | 8.4 s | |
| Revaluation | 31.9 s | 51% |
| of it, recalibration (46 calls, 0.23 s each; Bermudans 10.5 s of their 11.8 s) | 10.5 s | 17% |
| of it, building price functions (the market's context rebuilt per trade, `from_market`, 1,007 calls, 7.0 s) | 7.8 s | 12% |
| of it, the inductions of Americans and the equity options' grid runs | 10.0 s | |
| of it, every linear class | 0.9 s | |

The same sample on the Mac spends 97% of its 416.5 s in the P&L programs. What is slow on the
GPU is the per-trade work outside a jit, each step a kernel launch and a sync: the `npv`
pass values every equity option position by position (market risk values each listed contract
once), the recalibration's buckets are many small programs per trade, and each trade's price
function rebuilds the market's curves.

**Two of them fixed since** (fork `c0abc66` → this commit): the `npv` pass values each listed
equity-option contract once, one share's value times each trade's shares, as market risk does
(`engine.pricing.cube._equity_value`); and `engine.pricing.context.from_market` keeps the last
market's context per dtype, so a run builds the market's curves once, not once per trade.
Every npv, P&L and `VaR_99` of the 1,136-trade sample is the same bit for bit on the Mac, where
the `npv` pass fell from 3.5 s to 2.1 s. On the A100 (fork `2a3bed2`, 2026-11-09) the sample's
warm run fell from 62.9 s to 40.9 s: the `npv` pass from 30.6 s to 11.0 s, the revaluation from
31.9 s to 29.5 s. The engine's
whole suite passes but `test_sharding` (4,339 passed, 18 skipped), which fails on upstream
`17ac757` too (below).

**The whole book on an A100 with both** (2026-11-09, fork `2a3bed2`, 11,360 trades, one
process), against the c4d-standard-32 CPU pod on fork `a5bd7c3` (November 8):

| Run | c4d CPU | A100 | Faster |
|---|---|---|---|
| 256 scenarios, cold | — | 1,698 s, 1,417 s of it compile | |
| 256 scenarios, warm | 3,926 s | 251.5 s (`npv` 69 s, revaluation 179 s) | 15.6x |
| 1,024 scenarios, same process | 14,961 s | 558.3 s (`npv` 68 s, revaluation 481 s, 24 s compile) | 26.8x |

`VaR_99` is the CPU pod's to 1e-15 at both counts (73,560,770.0634258; 77,759,354.61007; the
old draw on both, so the figures compare). More scenarios reuse the programs compiled at 256:
a scenario count is a number of 256-scenario batches. The revaluation is about 78 s a run plus
101 s per 256 scenarios, so at 1,024 scenarios four fifths of the run scales with the count.

**A new process, with JAX's persistent compilation cache** (`JAX_COMPILATION_CACHE_DIR`, every
program cached, as the engine's HTTP worker sets it): the 1,136-trade sample took 848.6 s in a
first process (792 programs written, 74 MB) and 212.5 s in a second (816 of 816 read from the
cache), against 40.9 s warm in one process. The cache skips XLA's compile, not the tracing
(58 s), the lowering to MLIR (52 s) or the reads (52 s).

**Recalibration is the group's too** (2026-11-09). Market risk recalibrates a group's trades
under a batch of scenarios with one program per bucket (`ScenarioCalibration.group`,
`engine.calibration.lgm.bootstrap_group`: the trades and the scenarios mapped inside
`_bootstrap_bucket_group`), a group of one included, on a CPU as on an accelerator. Trade by
trade it was `jax.vmap` called outside a jit on a program per bucket, traced again each call:
about 200 kernel launches a trade and batch on an A100, 0.23 s, 10.5 s of the sample's 31.9 s
of revaluation. Where the scenarios move the quotes (a Vega bump) it is still per trade.

- Its volatilities are the per-trade path's within the root solver's resolution (2e-12 of a
  bucket's; never bit for bit), and the 20-Bermudan sample's P&L within 5.9e-15 of each trade's
  largest.
- A run's second and third pass compile nothing (that sample, grouping forced, the compile
  events counted): the map is inside the jit.
- On a CPU a call costs what it did (0.010 s against 0.012 s for one Bermudan), so a CPU run's
  recalibrated options are no longer `a5bd7c3`'s bit for bit, only within the solver's
  resolution; every other class still is.
- Not yet measured on a GPU.

**The A100's `VaR_99` of the sample is not the Mac's, and that is the scenarios, not the
device.** The 1,136-trade sample gives `VaR_99` 59,274,718.60 on the A100 and 59,689,775.14 on
the Mac, 0.7% apart. Every npv and every rates trade's P&L agree; every cash equity's and
equity option's P&L differ, by 30–40% of their largest. The pod drew other equity shocks than
the Mac from the same request, fixed since: see [One draw on every host](#one-draw-on-every-host). On the
A100, grouped and ungrouped agree within 2.5e-12 of each trade's largest P&L (`VaR_99` 2.4e-14
apart); run to run, its Bermudans' and Americans' P&L move by up to 1.1e-15 (the GPU's
summation order). On the Mac, grouping forced at the A100's group sizes reproduces the
ungrouped P&L within 2.6e-12.

**Same answers in a group.** Against fork `a5bd7c3`, on 20 trades of each class at 256 scenarios (the
seven classes TraderX books) revalued in groups on the CPU (before it took one trade a call),
every per-trade P&L agrees within 5e-11 of that trade's largest
(Americans; Bermudans 3e-14, the linear classes 7e-16, cash equities and swaps bit for bit) and
`VaR_99` within 2e-13. `TestGroupedRevaluation` checks a group against each trade revalued alone
at float64 and float32, a short Bermudan among long ones.

## One draw on every host

**The defect** (upstream's, found 2026-11-09). `MonteCarloScenarios.draw` took the covariance's
root as `V sqrt(L)` from `np.linalg.eigh`. That is a root for every choice of eigenvectors `V`,
and the choice is not unique: each eigenvector's sign is free, and a repeated eigenvalue's
eigenvectors are any basis of its space. TraderX's covariance (140 factors, its 50 equity spots
equally correlated) has 48 pairs of eigenvalues within 1e-10 of each other. numpy's eigenvectors
there differ by BLAS and by CPU: on the book's request the Mac (Accelerate), a Linux arm64
container, a Linux amd64 container and the A100 pod (all OpenBLAS, numpy 2.4.6) drew four
different sets of equity shocks, up to 0.29 apart, of one distribution. The sample's `VaR_99` was
59,689,775 on the Mac and 59,274,719 on the pod.

**The fix.** The draw is the Sobol normals times the covariance's symmetric root,
`V sqrt(L) V^T` (`engine.risk.market.scenarios.symmetric_root`). A positive semi-definite matrix
has exactly one symmetric positive semi-definite root, whatever `V` the decomposition returns,
so the scenarios are a function of the request alone. On the book's covariance the three numpy
builds above agree within 1.5e-15 on every shock.

**Why this and not another mechanism.**

| Mechanism | Why not |
|---|---|
| Cholesky | Unique only for a positive definite matrix. The engine accepts singular covariances (perfectly correlated pillars, `test_singular_covariance_is_accepted`); a jitter to admit them changes the distribution, and pivoting breaks ties by rounding, host by host |
| Fixing the eigenvectors (signs, a canonical basis in each repeated eigenvalue's space) | Needs a tolerance to say which eigenvalues are equal, and the draw would jump where a gap crosses it; the book's pairs are 1e-10 apart, not equal |
| Storing the drawn shocks with the run | Makes a replay of that run reproducible; a fresh run of the same request on another host still draws others. It can be added on top |

**What changes.** The distribution does not: the scenarios have the requested covariance
exactly as before (`R R^T = V L V^T`). The realizations do, on every factor, so every Monte
Carlo figure moves once, by sampling noise; historical scenarios and scenarios given as shifts
are untouched. Before, Sobol dimension `k` drove eigenvector `k` (smallest eigenvalue first, so
the leading dimensions carried the least variance); now it is centred on factor `k`. Measured on
the Mac:

| Request | `VaR_99` before | after | `ES_97.5` before | after |
|---|---|---|---|---|
| TraderX's VaR/ES proof, 17 trades, 1,024 scenarios | 373,857.16 | 381,036.79 (+1.9%) | 381,254.68 | 379,224.28 (−0.5%) |
| Every tenth trade of the book, 1,136 trades, 256 scenarios | 59,689,775.14 (Mac), 59,274,718.60 (pod) | 63,607,091.37 (+6.6%, +7.3%) | — | 58,760,256.65 |

The proof's covariance (61 factors, one equity) has no repeated eigenvalue, which is why the
Mac and the pods agreed on it before. No test and no TraderX check holds a Monte Carlo figure
taken from the engine: the suite's expected values are closed forms and ORE's, and they pass
unchanged.

**The test** (`TestMonteCarloScenarios::test_the_draw_does_not_depend_on_the_eigenvectors`)
needs one machine. On an equally correlated covariance it checks the draw against the normals
times the symmetric root in closed form, then again with `np.linalg.eigh` returning a rotated
basis of the repeated eigenvalue's space and a flipped sign. Both fail on the draw as it was.

**End to end, the Mac against the A100 pod** (2026-11-09, fork `4b2fd3a`; Accelerate against
OpenBLAS, CPU against GPU), the same requests:

| Request | `VaR_99`, Mac | A100 pod | Apart | `ES_97.5` apart |
|---|---|---|---|---|
| The proof, 17 trades, 1,024 scenarios | 381,036.79285857355 | 381,036.79285857925 | 1.5e-14 | 1.2e-15 |
| The sample, 1,136 trades, 256 scenarios | 63,607,091.3670907 | 63,607,091.367090955 | 4.0e-15 | 1.9e-14 |

The sample is the request whose `VaR_99` was 0.7% apart before. Its shocks on the two hosts are
within 2.0e-15 of each other, the proof's within 6.6e-17.

The fix is to rounding, not to the bit: hosts' roots differ in the last place (1.5e-15 here).

**One ORE-parity sweep case moved, and it is ORE's side.** The sweep's scenarios are drawn
by the engine, so its sample changed. In `test_market_risk_is_ores[seed-18]`, under scenario 9
of 32, a recalibrated American (32 calibration helpers) is 9.1e-5 off ORE's P&L: 3.3e-10 of its
notional, 8.4e-8 of the P&L, where the sweep allowed 1e-11 of the notional and every other
scenario of the case agrees within 1e-12. On that scenario's market:

- the engine fits every helper to 1.5e-14 of its market value, as on the other scenarios, and
  its value is the same on either root solver within 5e-17 of the notional;
- the engine's market risk and its own valuation of the shocked market agree within 3.4e-16;
- ORE's historical simulation and its NPV analytic on the shocked market agree with each other
  within 1.8e-15, and are 3.2e-10 from the engine.

So each side is consistent with itself, the engine's calibration is exact to rounding, and the
difference is ORE's calibration on that market: its optimiser stopping at its tolerance, which
the sweep already records for today's values (2.7e-12) and for 32-helper baskets (I-117).
ORE's log at the oracle's level does not print its calibration error, so that last step is
inferred, not read. The pricing is untouched by the fix: the same move gives the same
difference on the code before, whose sample did not reach it.

The sweep's tolerances follow its measurement, as they were set: a Bermudan's or American's
P&L under a scenario 1e-9 of its notional (the rest 1e-11, as before), and VaR and ES 1e-9
relative plus the largest difference of a scenario's portfolio P&L from ORE's in that run (an
order statistic, and a mean of them, cannot agree better than the P&L does). The bugs the
sweep found before were 1e-7 of a notional and larger. With them the suite passes but
`test_sharding` (as upstream): 4,339 passed on the fix, and the 145 market-risk parity cases
again on these tolerances.

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

- **A Monte Carlo draw depends on the host's BLAS** on upstream `17ac757`
  (`MonteCarloScenarios.draw`, `engine/risk/market/scenarios.py`): fixed in the fork, see
  [One draw on every host](#one-draw-on-every-host). The two Linux pods (c4d and A100) drew the
  same shocks as each other. The Mac's November 7 book run agreed with them to 1e-15 although
  its shocks would differ by the same mechanism; which numpy that run used is not recorded.

- **Two American swaption parity misses on upstream `17ac757`.** Letting the ORE-parity sweep
  draw ACT/360 and 30/360 reaches two cases, `seed-24` (Greeks, an index delta) and `seed-39`
  (market-risk P&L), where an American swaption misses ORE by about 6e-5 relative, outside the
  sweep's tolerance. Both reproduce bit for bit on upstream with only its day-count list extended,
  and neither touches a fork product: an American exercise-grid tolerance edge upstream's own
  sample does not reach. The sweep keeps upstream's day counts (`SWEEP_DAY_COUNTS`), so its
  sample is upstream's.
- **`test_sharding`'s four-device run** fails on upstream `17ac757` itself: its FP8 stage asks for
  native execution on a CPU, which the precision policy refuses.
