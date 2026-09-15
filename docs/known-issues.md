# Known Issues and Limitations Register

**Purpose.** One authoritative list of every known defect and scope gap in this engine, what
each one does to a number a user would see, and what closing it actually requires. Written
during the TraderX EOD integration review (see
[EOD Contract Proposal](planning/eod-contract-proposal.md)), which is where several of these
were first identified.

**Why this file exists.** Every issue below was, at the time it was found, *invisible from
the outside*: the engine returned a plausible, finite, confidently-formatted number (or
silently returned nothing at all) with no indication anything was wrong. A passing test
suite did not surface them — in one case a test actively asserted the buggy behavior was
"by design". A register is the mitigation for that class of problem.

**Status vocabulary — used precisely:**

| Status | Meaning |
|---|---|
| **FIXED** | Defect removed. Regression test verified to fail against the pre-fix code. |
| **FLAGGED** | Inaccuracy **unchanged**. The engine now warns instead of staying silent. Not a fix. |
| **OPEN** | Not addressed. Numbers are wrong or absent today. |

Last full verification: **824 tests passing** (baseline before this work: 802).

---

## Summary

| ID | Issue | Severity | Status |
|---|---|---|---|
| [I-01](#i-01) | Swap Delta/Gamma/Theta silently absent from portfolio results | High | ✅ FIXED |
| [I-02](#i-02) | Bermudan Vega never computed | Medium | ✅ FIXED |
| [I-03](#i-03) | No per-instrument NPV; totals unattributable | Medium | ✅ FIXED |
| [I-04](#i-04) | Aged swaps mispriced at every step past first accrual | **High** | ⚠️ FLAGGED |
| [I-05](#i-05) | No faithful USD-SOFR/ACT360 swap construction | **High** | ❌ OPEN |
| [I-06](#i-06) | Mid-coupon Bermudan/American exercise understates value | Medium | ⚠️ FLAGGED |
| [I-07](#i-07) | No bond, equity, or listed-option pricer | Medium | ❌ OPEN |
| [I-08](#i-08) | Job store is in-process; lost on restart | Medium | ❌ OPEN |
| [I-09](#i-09) | Whole scenario cube serialized into JSON responses | Medium | ❌ OPEN |
| [I-10](#i-10) | No trade identity; results keyed by array position | Medium | ❌ OPEN |
| [I-11](#i-11) | Risk measure unlabelled; no Monte Carlo error reported | Medium | ❌ OPEN |
| [I-12](#i-12) | `/version` reports dispatcher backend, not worker device | Low | ❌ OPEN |

**The two that matter most for financial correctness are [I-04](#i-04) and [I-05](#i-05).**
Both are unfixed. Both need inputs or decisions that do not exist yet — not more engineering
time on the current code.

---

## FIXED

### I-01 — Swap Delta/Gamma/Theta were silently absent {#i-01}

**Severity:** High · **Status:** ✅ FIXED

**Symptom.** `compute_greeks=True` on a portfolio containing swaps returned a `greeks` dict
with **no entry for any swap**, no error, no warning. A swap-only portfolio returned `{}`.

**Cause.** `engine.risk.greeks.swap_delta_gamma`/`swap_theta` were fully implemented and
unit-tested against finite differences. But a `SwapConfig` carries curve *indices*
(`discount_curve_index`/`forward_curve_index`) rather than its own `ZeroCurveConfig`, and
`_compute_all_greeks` never received the `SimulationConfig` those indices resolve against.
Rather than guess a placeholder curve, it skipped every swap — a defensible local decision
that became a silent data loss at the portfolio boundary.

**Why it went unnoticed.** `tests/test_portfolio_scale_and_edge_cases.py` asserted
`set(result.greeks) == set(range(3, 12))` — i.e. that swaps 0–2 have no Greeks — with the
comment *"which skips SwapConfig by design."* **The test pinned the bug in place and
described it as intentional.**

**Fix.** `_compute_all_greeks` now takes `market_config` and resolves each swap's curves via
`_swap_curve_configs`, which **raises** on an out-of-range index rather than falling back to
curve 0. Pricing math unchanged — orchestration only.

**Verified.** `tests/test_portfolio_gap_fixes.py::TestSwapGreeksReachThePortfolioPath` (7
tests). Reverting the wiring fails 5. A naive "default to curve 0" fix fails
`test_uses_the_curve_its_indexes_name_not_curve_zero`, which compares against a deliberately
different curve — **this negative check is the important one**, since a wrong-curve
sensitivity is a plausible-looking number.

---

### I-02 — Bermudan Vega was never computed {#i-02}

**Severity:** Medium · **Status:** ✅ FIXED

**Symptom.** No Bermudan/American trade ever reported Vega through `price_portfolio` or the
HTTP API.

**Cause.** `engine.risk.greeks.bermudan_vega` was implemented and unit-tested (including a
subtle cross-bucket Jacobian correction) but **no portfolio-level caller ever invoked it**.

**Fix.** Wired into `_compute_all_greeks`, guarded to the only case where it is well-defined:
a genuinely *calibrated* `Sigma` produced from `calibration_targets`. A flat hand-set
`hw_sigma` has no market quote to be sensitive to, so Vega is **omitted rather than
fabricated** — and omitting it does not suppress the Greeks that are well-defined.

**Verified.** `tests/test_portfolio_gap_fixes.py::TestBermudanVegaReachesThePortfolioPath`.

---

### I-03 — No per-instrument NPV {#i-03}

**Severity:** Medium · **Status:** ✅ FIXED

**Symptom.** `PortfolioResult.base_npv` was a single float. A portfolio total could not be
reconciled against the positions that produced it — a blocker for any integration that must
attribute value to an account or contract.

**Fix.** `PortfolioResult.base_npv_per_trade` returns each trade's t=0 NPV in request order,
with `base_npv` defined as `sum(...)` of that list. **The total and the breakdown are the
same numbers**, computed once, so they cannot drift apart. Exposed through
`PortfolioResultSchema` too.

**Verified.** `tests/test_portfolio_gap_fixes.py::TestPerTradeBaseNpv` (5 tests), including
sign/order attribution via a payer/receiver mirror pair.

---

## FLAGGED — inaccuracy unchanged, silence removed

> These are **not fixes.** The numbers are as wrong as they were before. What changed is that
> the engine now says so.

### I-04 — Aged swaps are mispriced at every step past first accrual {#i-04}

**Severity:** High · **Status:** ⚠️ FLAGGED (inaccuracy unchanged)

**What is wrong.** `price_swaps` has no representation of an already-fixed floating coupon.
At any simulated time `t` past a swap's first accrual start, the elapsed period is discounted
using `P(t, accrual_start)` for `accrual_start < t` — which is not a discount factor at all,
but a clamped, meaningless value (see
[`reconstruct_yield_curves`](../engine/simulation/market_model.py)'s `B(t,T)` clamp).

**Blast radius — this is the widest of any issue here:**

- t=0 base NPV — **unaffected and exact**.
- Every `npv_cube` value at every step past first accrual — **inaccurate**.
- **Every VaR/ES number derived from that cube — inaccurate**, since they aggregate it.
- Any exposure profile, XVA-style calculation, or multi-day experiment — inaccurate.
- Theta past the first reset — inaccurate.

Measured divergence against an ORE reference at a future evaluation date is ~1e-4 to 1e-3
relative, **growing** with distance past the aged dates
(`tests/test_swap.py::TestAgedSwapKnownLimitation`).

**Scope note.** `SwapConfig` has no `forward_start` field, so **every swap in this engine is
spot-starting**. Therefore *every* multi-step swap portfolio is affected. This is not an edge
case; it is the default path.

**What changed.** `price_portfolio` now emits a warning per affected swap into
`PortfolioResult.warnings`, naming the trade, its first accrual start, how many steps are
affected, and explicitly that t=0 base NPV is unaffected. Warnings cross the worker-process
boundary into the HTTP result. **The pricing is unchanged.**

**What closing it requires** — two things, neither of which exists today:

1. **Engine work:** track already-fixed rates per scenario/step inside the pricing kernel, or
   exclude elapsed cashflows from the sum. This is real work in
   [`engine/instruments/swap.py`](../engine/instruments/swap.py), not orchestration.
2. **Data that does not exist:** historical published fixings for each floating index, back
   to each live trade's effective date. **TraderX does not currently export these** — it is
   the `pastFixings` field requested in
   [the proposal §2.2](planning/eod-contract-proposal.md). Without them there is nothing to
   populate a fixed coupon *with*.

Item 2 is the binding constraint. Engine work alone cannot close this.

**Verified (the warning, not the fix):**
`tests/test_portfolio_gap_fixes.py::TestAgedSwapWarningIsNotSilent` (5 tests) and
`tests/test_api.py::TestGapFixesSurviveTheHttpBoundary`.

---

### I-06 — Mid-coupon Bermudan/American exercise understates value {#i-06}

**Severity:** Medium · **Status:** ⚠️ FLAGGED

**What is wrong.** Bermudan/American pricing is exact **only** when exercise dates are
reset-aligned with the underlying's accrual schedule. For an exercise date falling inside an
accrual period, the in-progress coupon is **excluded entirely rather than prorated** — a
deliberate, conservative (value-understating) approximation. The error is an entire coupon's
PV, not a few days' accrual.

**What changed.** `validate_portfolio_against_simulation` warns per misaligned trade, and
`price_portfolio` collects these into `PortfolioResult.warnings`. (This predates the current
review; recorded here for completeness.)

**What closing it requires.** Proration logic in `_hw_swap_value_at_nodes`
([`engine/instruments/bermudan_swaption.py`](../engine/instruments/bermudan_swaption.py)),
plus agreement on the settlement convention for a mid-period exercise — see decision **D16**
in the TraderX pack. Engine-side work; no external data dependency.

**Documented by.** `tests/test_bermudan_swaption.py::TestMidCouponKnownLimitation`.

---

## OPEN — not addressed

### I-05 — No faithful USD-SOFR / ACT-360 swap construction {#i-05}

**Severity:** High · **Status:** ❌ OPEN — **blocked on external agreement, not effort**

**What is wrong.** [`build_vanilla_swap`](../engine/models/ore_builders.py) constructs, for
*every* swap, a generic term-IBOR swap. Verified empirically against the live builder:

| Property | This engine builds | A USD-SOFR booking is |
|---|---|---|
| Index | `SimIndex6M` (term IBOR) | USD-SOFR (overnight) |
| Fixed leg day count | `Actual/365 (Fixed)` | `ACT/360` |
| Float leg day count | `Actual/365 (Fixed)` | `ACT/360` |
| Calendar | `TARGET` (European) | `US-SIFMA` / FedFunds |
| Compounding | none (term rate) | daily compounded in arrears |
| Schedule source | tenor string (`"5Y"`) | explicit effective/maturity dates |
| Lookback / lockout / payment lag | not represented | contractual, per booking |

**Why the ACT/365 choice is not itself a bug.** It is deliberate and documented — it keeps
day count consistent with the simulation's own year-fraction time axis. It is a sound
*internal* decision that becomes an *external* incompatibility the moment a real SOFR
contract arrives.

**Magnitude — this is not rounding.** ACT/360 vs ACT/365 changes every accrual factor by
`365/360 - 1` = **1.389%**. On a $1mm 5Y fixed leg at 3% that is **~$1,906**, roughly **46x a
1bp DV01**. A wrong-convention swap prices confidently and wrongly.

**Current behavior with a SOFR booking:** it would produce a number, and **no test in this
repository would catch it**, because every test builds its inputs with the same generic
builder. There is no cross-check against a real booked contract.

**What closing it requires:**

1. **Agreement first (blocking):** the full convention set — fixed/float day counts,
   compounding method, lookback, lockout, payment lag, calendar, business-day convention,
   roll convention, stub handling, separate fixed/float frequencies. These are decisions
   **D03/D04** in the TraderX pack and
   [proposal §2.2](planning/eod-contract-proposal.md#22-usd-sofr-swap--the-w2-blocker-set).
   *Guessing them produces confident wrong numbers, which is exactly this issue's failure
   mode.*
2. **A new builder alongside the existing one** — not a modification of it. Every current
   swaption pricer depends on `build_vanilla_swap`'s ACT/365 consistency with the simulation
   time axis, and the full test suite pins that behavior.
3. **A hard refusal path:** any booking whose conventions fall outside the supported subset
   must be returned as explicitly *unsupported with a reason*, never approximated by the
   generic builder.
4. **Acceptance against a same-terms ORE reference** — not this engine's own test suite.

**Interim mitigation (recommended, not yet implemented).** Until the builder exists, reject
any trade carrying SOFR/overnight-index or ACT/360 metadata rather than routing it through
the generic path. This cannot be implemented today because `SwapConfig` has no field in which
such metadata could arrive — which is itself part of the work.

---

### I-07 — No bond, equity, or listed-option pricer {#i-07}

**Severity:** Medium · **Status:** ❌ OPEN

`engine/instruments/` contains exactly four modules, all rate derivatives (swap, European /
Bermudan / American swaption). There is **no pricer** for Treasuries, corporate bonds, cash
equities/ETFs, or listed options — all of which appear in TraderX's schema-3 position export.

**Important distinction:** `SimulationConfig.equities` drives correlated equity *risk-factor
paths*. It is **not** an equity position pricer — nothing takes a signed share count and
returns a position value.

**What closing it requires.** Per instrument: a config dataclass, a pricer, ORE parity tests.
A fixed-rate Treasury is the cheapest (deterministic discounted cashflows, no Monte Carlo, no
calibration) and already has a written plan —
[traderx-bond-integration-roadmap.md](planning/traderx-bond-integration-roadmap.md). Corporate
bonds additionally need a credit/spread model; **a Treasury-discounted corporate is not credit
pricing** and should be refused rather than approximated.

---

### I-08 — Job store is in-process and lost on restart {#i-08}

**Severity:** Medium · **Status:** ❌ OPEN

`_JOBS` in [`engine/api/routes.py`](../engine/api/routes.py) is a plain Python dict in the
dispatcher process. A restart loses every job id; a second uvicorn worker would 404 on ids
issued by the first. Job states are `pending`/`running`/`done`/`failed` only — there is no
`partial`, no `superseded`, no attempt history, and no structured failure classification.

**Consequence for a batch caller:** a lost in-memory job is indistinguishable from a
computation failure, and a coordinator cannot tell which failures are worth retrying.

**What closing it requires.** Either a durable store, or — preferred, per **D13** — accept
that the coordinator owns the durable logical job while this engine owns only the computation
*attempt*. That needs: a worker boot epoch exposed so restarts are detectable, idempotency so
re-submitting identical immutable inputs is safe, and structured failure classes
(`bad-terms` / `missing-market-data` / `unsupported-product` / `numerical-failure` /
`infrastructure`) so only retryable failures are retried. See
[proposal §6.3](planning/eod-contract-proposal.md).

---

### I-09 — Whole scenario cube serialized into JSON responses {#i-09}

**Severity:** Medium · **Status:** ❌ OPEN

`PortfolioResultSchema.npv_cube` serializes `[Scenarios, TimeSteps, Trades]` as nested JSON.
At a realistic 4096 × 24 × 211 that is ~20M floats in a single HTTP body — unusable through a
browser or a control message, and a memory risk on both ends.

**What closing it requires.** Write the cube to a chunked artifact (shape, dtype, axis
ordering, hash, and an instrument-id ordering file) and return a *reference* plus compact
summaries. See [proposal §6.4](planning/eod-contract-proposal.md).

---

### I-10 — No trade identity; results keyed by array position {#i-10}

**Severity:** Medium · **Status:** ❌ OPEN

`PortfolioRequest.trades` is positional; `PortfolioResult.greeks` and
`base_npv_per_trade` are keyed/ordered by index. There is no opaque
account/position/contract id anywhere in the request or result, so **array position is
load-bearing** for joining a result back to a booking.

Ordering is correct and tested today — but any reordering, filtering, or partial-coverage
response would silently misattribute results. Identity should not depend on list order.

**What closing it requires.** An `instrumentId`/`accountId` pair on every trade config,
echoed on every result row. Mechanically small; touches request, result, and schema layers.

---

### I-11 — Risk measure unlabelled; no Monte Carlo error reported {#i-11}

**Severity:** Medium · **Status:** ❌ OPEN

`PortfolioResult.risk` returns keys like `VaR_95` with **no statement of what measure they
are**. A risk-neutral exposure simulation is *not* a calibrated forecast of tomorrow's loss,
and nothing in the result distinguishes the two. No effective sample size, Monte Carlo
standard error, or convergence diagnostic is reported alongside the tail statistic, so a
sparse-tail estimate is indistinguishable from a well-converged one.

**What closing it requires.** An explicit `measure` label
(`risk-neutral-pricing` / `historical-forecast` / `deterministic-stress`), plus effective
sample size and MC standard error on every tail statistic. Small change; prevents a whole
category of misreading. See [proposal §3.6/§4](planning/eod-contract-proposal.md).

---

### I-12 — `/version` reports dispatcher backend, not worker device {#i-12}

**Severity:** Low · **Status:** ❌ OPEN

`GET /version` reports `jax.default_backend()` of the **dispatcher** process, which performs
no JAX work. Actual pricing runs in a separate `worker_pool` process with its own JAX runtime.
On a single-CPU dev machine these agree, so the discrepancy is invisible; on a multi-TPU host
it would not be — and a precision/hardware study that trusted this field would draw wrong
conclusions about which device produced a result.

**What closing it requires.** Report device and actual per-stage precision **from the worker**,
on the result itself, rather than from the dispatcher.

---

## How to use this register

- **Adding an issue:** give it the next `I-NN`, state the *symptom a user sees* before the
  cause, and be explicit about FIXED vs FLAGGED. "We warn about it now" is never FIXED.
- **Closing an issue:** a regression test must be verified to **fail against the pre-fix
  code** before the status changes. A test that passes either way proves nothing. Where a
  plausible-but-wrong fix exists (e.g. substituting a default curve), add a test that fails
  against *that* too.
- **Related reading:**
  [EOD Contract Proposal](planning/eod-contract-proposal.md) (integration context and the
  full field-level requirements), [HTTP API](reference/http-api.md),
  [The Portfolio Entry Point](reference/portfolio-entrypoint.md).
