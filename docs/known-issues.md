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

Last full verification (2026-09-16, after the **I-19** and **I-20** fixes from TraderX's v5
review): **1,452 passed, 1 failed**. Previously 1,384/2, then 1,324/2 after W1.3, and
1,217/2 on 2026-09-15 after W1.2.

**The single failure is a timing-sensitive concurrency test, not a code defect.**
`tests/test_worker_pool.py::TestWorkerPoolConcurrency::test_cross_tier_jobs_correct_and_concurrent`
asserts that two jobs' wall-clock intervals genuinely *overlap*. Under full-suite load the OS
can serialize them, so the assertion fails while every correctness assertion in the same test
(dtypes, NPV parity against a reference run) passes. Verified: it passes 3/3 in isolation, and
it also passes against stashed pre-fix code, so it is unrelated to the v5 fixes. This is
[I-15](#i-15)'s failure mode in a sibling test — I-15 fixed
`test_same_tier_jobs_also_overlap_across_pool_workers`, and `test_cross_tier_jobs...` carries
the same wall-clock premise. Tracked as a follow-up rather than silently re-run until green.

(The 2 pydantic failures recorded previously are resolved — that dependency is now installed.)

**The 2 failures are an environment gap, not a code defect.** Both are in
`tests/test_var_es_diagnostics.py::TestDiagnosticsReachTheHttpBoundary` and fail with
`ModuleNotFoundError: pydantic` — a dependency this project *declares*
(`pyproject.toml`: `pydantic>=2`, `fastapi`) but which is not installed in this environment.
`tests/test_api.py` does not collect for the same reason. Verified pre-existing by stashing
the fixes and re-running: identical failures. `pip install -e .` resolves both.

Engine-side, every test passes.

**This line reports what a full run actually produces.** Earlier figures here (824, then
1175) were stale or recorded a passing count a full run did not reproduce. A register whose
own header overstates its verification undermines every status in it.

---

## Summary

| ID | Issue | Severity | Status |
|---|---|---|---|
| [I-01](#i-01) | Swap Delta/Gamma/Theta silently absent from portfolio results | High | ✅ FIXED |
| [I-02](#i-02) | Bermudan Vega never computed | Medium | ✅ FIXED |
| [I-03](#i-03) | No per-instrument NPV; totals unattributable | Medium | ✅ FIXED |
| [I-04](#i-04) | Aged swaps mispriced at every step past first accrual | **High** | ⚠️ FLAGGED |
| [I-05](#i-05) | No faithful USD-SOFR/ACT360 swap construction | **High** | ❌ OPEN — refusal path landed (W0.4) |
| [I-06](#i-06) | Mid-coupon Bermudan/American exercise understates value | Medium | ⚠️ FLAGGED |
| [I-07](#i-07) | No bond, equity, or listed-option pricer | Medium | ❌ OPEN — both Treasury pricers landed (W1.2 bill, W1.3 note) |
| [I-08](#i-08) | Job store is in-process; lost on restart | Medium | ❌ OPEN — attempt semantics + 4 lookup states landed on the EOD path (W1.6.4) |
| [I-09](#i-09) | Whole scenario cube serialized into JSON responses | Medium | ❌ OPEN |
| [I-10](#i-10) | No trade identity; results keyed by array position | Medium | ❌ OPEN — closed at the EOD boundary (W0.7) |
| [I-11](#i-11) | Risk measure unlabelled; no Monte Carlo error reported | Medium | ❌ OPEN — measure + MC diagnostics landed (W0.6) |
| [I-12](#i-12) | `/version` reports dispatcher backend, not worker device | Low | ❌ OPEN |
| [I-13](#i-13) | Negative curve index silently prices against the wrong curve | **High** | ✅ FIXED |
| [I-14](#i-14) | `generate_paths(precision=32)` leaks `jax_enable_x64=False`; float64 silently truncates | **High** | ✅ FIXED |
| [I-15](#i-15) | Worker-pool concurrency test could not observe concurrency | Low | ✅ FIXED |
| [I-16](#i-16) | `rateSensitivity` is parallel-only; no per-pillar decomposition | Medium | ❌ OPEN — labelled honestly, blocked on a real curve |
| [I-17](#i-17) | A malformed note date failed the entire bundle, not just its row | Medium | ✅ FIXED |
| [I-18](#i-18) | No equity spot or FX source; equity positions are refused, not valued | Medium | ❌ OPEN — refusal path landed (W1.4) |
| [I-19](#i-19) | Accrual tolerance rounded the bound it exists to enforce | Medium | ✅ FIXED |
| [I-20](#i-20) | Impossible calendar dates aborted the whole bundle | **High** | ✅ FIXED |
| [I-21](#i-21) | Greeks recompile 23 XLA programs on every call (fresh closures) | Medium | ❌ OPEN |
| [I-22](#i-22) | Calibration recompiles 8 XLA programs per call (baked-in constants) | Low | ❌ OPEN |

**The two that matter most for financial correctness are [I-04](#i-04) and [I-05](#i-05).**
Both are unfixed. Both need inputs or decisions that do not exist yet — not more engineering
time on the current code.

**[I-13](#i-13) and [I-14](#i-14) were found on 2026-09-15** and are the first defects in
this register found by *reading source and predicting a failure* rather than by running the
suite — I-13 by TraderX's review of a pushed commit, I-14 while reproducing it. Both produced
plausible wrong numbers with no error. **Both are now fixed**, each with a regression test
verified to fail against the pre-fix code.

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
   generic builder. — ✅ **Done at the EOD integration boundary (W0.4).**
4. **Acceptance against a same-terms ORE reference** — not this engine's own test suite.

**Interim mitigation — ✅ implemented for the EOD path (W0.4).**
[`engine/integration/conventions.py`](../engine/integration/conventions.py) refuses any
booking whose conventions fall outside an explicit **positive** allowlist (today: generic
`SimIndex*` term IBOR, ACT/365 legs, no overnight compounding), returning
`CONVENTION_NOT_SUPPORTED` with the offending fields named — **before any pricing object is
constructed**, which is the point at which the wrong conventions would otherwise be applied.
A booking that states *no* conventions is refused too, never defaulted into the generic
builder. The TraderX SOFR fixture now returns an identified refusal naming all 13 of its
`missingTerms`. See [the EOD integration boundary](reference/eod-integration.md#w04--convention-allowlist-and-refusal--closes-part-of-i-05).

**Scope of that mitigation, stated precisely.** It covers bookings arriving through
`engine/integration/` — the TraderX EOD path. It does **not** change
`build_vanilla_swap`, and it does **not** guard a caller who constructs a `SwapConfig`
directly in Python: `SwapConfig` still has no field in which convention metadata could
arrive, so there is nothing there to refuse on. The W0.4 allowlist is a gate on the external
boundary, not a property of the pricer. **The underlying defect is unchanged** — this engine
still cannot faithfully price USD-SOFR — which is why this issue stays **OPEN** rather than
moving to FLAGGED or FIXED.

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

**Partially closed (W1.2, W1.3) — both Treasury shapes, and nothing else.**
[`engine/integration/bill.py`](../engine/integration/bill.py) prices a bill as a single
discounted cashflow; [`engine/integration/note.py`](../engine/integration/note.py) prices a
coupon-bearing note as a fixed-coupon strip plus bullet redemption. Both are verified to
**zero difference** against an independent ORE valuation — the bill against
`ORE.CashFlows.npv`, the note against a real `ORE.FixedRateBond` + `DiscountingBondEngine`.
Long and short on the delivered fixtures return exact mirrors (bill +98,507.15 / −98,507.15;
note +103,308.33 / −103,308.33).

**The note additionally reconciles to TraderX's own books.** Its accrued interest agrees
with their exported `0.018571` of par, at the §2 derived tolerance rather than as an exact
equality — the exporter rounds HALF_EVEN at 6 decimals, so the two monetary paths differ by
$0.04 on $100k by construction. Both paths travel in the published payload under an explicit
`accrualSource` label, and a disagreement beyond tolerance is **refused, not warned about**.

**Scope of that, stated precisely.** It covers a Treasury arriving through
`engine/integration/` in a **v2** bundle, and nothing else:

- an **equity position** is now understood, identified and validated, but **refused** for
  want of a spot source (W1.4 — see [I-18](#i-18)). Its multiplier, sign and currency are
  read and echoed; what is missing is market data, not engine code;
- **listed options** are untouched;
- a **corporate bond** is still refused. It shares every column with a Treasury and a
  Treasury-discounted corporate is *not* credit pricing;
- **`rateSensitivity` is available for the note only** (bumped revaluation, 1bp, parallel —
  see [I-16](#i-16)). The **bill still has none**: W1.3 earned the note's with a parity test
  and earned nothing for the bill;
- **`rateGamma`/`theta` are `unsupported` for every instrument**, including both priced
  Treasuries;
- `engine/instruments/` is **unchanged** — still exactly four rate-derivative modules. A
  direct Python caller of `engine.portfolio` has no bond pricer, because both bond pricers
  live at the integration boundary rather than in the instrument layer (W1.5 is the
  wire-through).

Status stays **OPEN**: the issue is "no bond, equity, or listed-option pricer". Treasuries
now price; an equity is refused for a *market-data* reason rather than a missing pricer
([I-18](#i-18)); corporate bonds and listed options are still absent entirely.

---

### I-16 — `rateSensitivity` is parallel-only; no per-pillar decomposition {#i-16}

**Severity:** Medium · **Status:** ❌ OPEN — labelled honestly, not silently approximated

**Found:** 2026-09-16, while implementing W1.3.

The note pricer returns a `rateSensitivity`, and the plan (§W1.3) asked for "per-pillar
`rateSensitivity` **non-zero across the curve**". What is delivered is a **parallel** shift
of the whole zero curve, not a per-pillar decomposition.

**Why, and why it is not a defect in the pricer.** The only market input this boundary can
be handed today is a registered *assumed profile*, and every registered profile is a **flat
constant** — `flat-3pct-v1` is one rate materialized onto pillars that all carry the same
value. There is no per-pillar structure to shift independently, so a per-pillar sensitivity
against it would be arithmetic theatre: it would either report the parallel number once per
pillar, or attribute the whole move to one arbitrary pillar. Both are worse than saying
"parallel", because both look like a decomposition a consumer could hedge against.

**What the engine does instead.** The number is labelled for exactly what it is, in the
published payload:

```json
"shockedFactor": "zero-curve-parallel",
"method": "bumped-revaluation",
"bump": 0.0001
```

`shockedFactor` deliberately names a *parallel shift* rather than a pillar. A consumer
reading the payload cannot mistake it for a bucketed sensitivity, which is the whole
mitigation — this is the "no silent approximation" rule applied to a *label* rather than to
a refusal.

**A consumer relying on bucketed rate risk does not have it.** Parallel DV01 is a correct
aggregate and a poor hedge instruction: it cannot distinguish a 2y position from a 10y one
with the same duration, and it says nothing about curve-shape risk.

**What closing it requires.** A curve with genuine pillar structure, which means
`marketInputs.mode: "package"` — observed market data with bootstrapped pillars. That is
**W2** and is blocked on the D03/D04 convention agreement, the same external dependency as
[I-05](#i-05). Once such a curve exists, the bump loop shifts one pillar at a time and
`shockedFactor` names the pillar; the pricer itself needs no change, because it already
re-prices through its own public path rather than differentiating a closed form.

**Do not close this by bumping the flat profile per-pillar.** That is the plausible-wrong
fix: it produces a full-looking bucketed vector whose entries are either all equal or all
but one zero, and it would pass any test that only checks the vector's shape.

---

### I-18 — No equity spot or FX source; equity positions are refused {#i-18}

**Severity:** Medium · **Status:** ❌ OPEN — refusal path landed (W1.4), valuation blocked on
market data

**Found:** 2026-09-16, while implementing W1.4.

A cash equity position is worth

```
signedQuantity × contractMultiplier × spot × fx
```

and **two of those four factors have no source at this boundary**.
[`engine/integration/market_inputs.py`](../engine/integration/market_inputs.py) registers
*flat interest-rate profiles* and nothing else — an `AssumedProfile` is a single
`flat_rate`. There is no equity spot in it, no FX rate, and no `mode` that supplies either.

**`SimulationConfig.equities` is not a substitute**, and the distinction is the same one
[I-07](#i-07) draws: it drives correlated risk-factor *paths* for a Monte Carlo. Nothing in
it takes a signed share count and returns a position value. It also lives in
`engine.simulation`, which `engine/integration/` is forbidden to import.

**What the engine does.** It reads the position — quantity, multiplier, currency — validates
it, and refuses with the missing input named:

| Condition | Reason |
|---|---|
| USD position | `SPOT_SOURCE_NOT_SUPPLIED` |
| Non-USD position | `FX_SOURCE_NOT_SUPPLIED` — supplying a spot alone would still not price it |
| Malformed quantity/multiplier | `TERMS_INCOMPLETE` — a broken row, not missing market data |

The refusal carries `signedQuantity`, `contractMultiplier` and `multipliedQuantity`, so a
consumer can confirm the engine read the position correctly even though it would not value
it.

**The tempting wrong fix is `closingMark`.** The positions extract carries one, and
`quantity × closingMark × contractMultiplier` reproduces the exporter's own `marketValue`
column **exactly**. That is what makes it dangerous:

- it is an **echo, not a valuation**. The engine would hand TraderX their own number back as
  though it had priced it, and any reconciliation against it would always agree — proving
  nothing while looking like independent confirmation;
- `closingMark` is an **observation at the session cut**, not a curve this run was priced
  against. Publishing it under `npv` with a `marketProvenance` derived from the requested
  *rate* profile would label an observed number with a provenance it does not have;
- it silently answers a **different question** than every other `npv` in the result. The
  bill and note NPVs are present values off an explicitly requested curve; an equity "NPV"
  taken from the mark is a mark. Summing them into one portfolio total would mix two
  incompatible quantities under one heading.

`tests/test_integration_equity.py::TestDoesNotEchoTheExportedMark` is the guard, and it was
verified to fail against exactly that implementation — patched in at both the pricer and the
pipeline level, 20 tests failed.

**What closing it requires — a market-data decision, not engine work.** Either
`marketInputs` grows a registered spot/FX surface (extending the W0.6 contract, with the
same "named, versioned, requested by id" discipline the rate profiles already have), or
TraderX supplies observed spots in the bundle. The pricer itself is four multiplications;
the arithmetic is not what is missing.

**Advertised, not hidden.** `capabilities()` reports equity `npv` under
`blockedOnMarketInput` rather than as an absent calculation, so a coordinator can tell
"wait for a release" from "send me a spot" — only the second is something they can act on.

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

**Partially mitigated by W1.6.4 (2026-09-16), on the EOD path only.**
[`engine/integration/workload.py`](../engine/integration/workload.py) adds the
*attempt*-ownership half of the preferred design: a canonical **workload key** over every
input that can change a number, **idempotent submission** (a repeated `submissionId` recovers
the same attempt rather than starting a second), **immutable terminal attempts** (a second
attempt cannot overwrite a first's outcome), and the **four distinguishable lookup states** —
so an accepted-but-running job no longer looks like an unknown one, which is what previously
invited a duplicate overnight batch.

**What is still open, and why the status has not changed.** The store is still an in-process
dict, so a restart still loses *running*-state knowledge. The crash-safety design in
[plan §W0.8](planning/traderx-integration-plan.md) — publish via a content-addressed manifest,
with lookup falling back to a **scan** rather than trusting a pointer — is deliberately not
built, because there is no persistent artifact store to scan. This is the difference between
*fixed* and *mitigated* (working rule 5): the state machine is right, and nothing durable
backs it yet. Note also that the portfolio path's `_JOBS` dict in
[`engine/api/routes.py`](../engine/api/routes.py) is **untouched** by this — the mitigation is
EOD-only.

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

**Partially closed (W0.7) — at the EOD boundary only.**
[`engine/integration/identity.py`](../engine/integration/identity.py) gives every row
reaching the TraderX EOD path an opaque, reproducible `itemId` plus a source identity block
(`{kind, accountId, security | contractId}` + `clusterEpoch`), carried on **refused rows
too**, with item ordering published as its own hashed artifact rather than inferred from
array position. `ItemResult` cannot be constructed without an identity, so an unidentified
row is unrepresentable rather than merely discouraged.

**This does not close the issue.** `PortfolioRequest.trades` and `PortfolioResult.greeks`
are unchanged and still positional — the W0.7 identity lives in a separate result type
(`engine.integration.result.RiskResult`) that does not yet flow through `price_portfolio`.
A direct Python caller of `engine.portfolio` still has no trade identity. Status stays
**OPEN** until `instrumentId`/`accountId` reach the trade configs themselves.

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

**Substantially addressed (W0.6), but not closed.** Both halves now exist:

- **The `measure` label.** `RISK_MEASURE_*` in
  [`engine/risk/var_es.py`](../engine/risk/var_es.py) defines the three-value vocabulary, and
  `ENGINE_RISK_MEASURE` records what this engine actually produces
  (`risk-neutral-pricing`). `engine.integration.result.RiskResult` carries it on every
  published result, and `capabilities()` advertises it so a consumer knows *before*
  submitting.
- **Convergence diagnostics.** `compute_risk_metrics` now returns `ES_<p>_tailCount`
  (effective sample size — the observations the ES mean actually averaged) and
  `ES_<p>_standardError` (`s/sqrt(n)`, `ddof=1`) beside every tail statistic. Purely
  additive: existing keys and values are untouched, and `include_diagnostics=False` returns
  the prior key set exactly. `standardError` is **NaN, never 0.0**, when `n < 2` — 0.0 would
  read as "perfectly converged" for the least trustworthy case.

**Why it stays OPEN.** `PortfolioResult.risk` is still a bare `Dict[str, jax.Array]` with no
`measure` field of its own — the label lives on `RiskResult`, which only the TraderX EOD path
produces. A direct Python caller of `price_portfolio` still gets unlabelled `VaR_95` keys,
which is exactly what this issue reports. Closing it means putting `measure` on
`PortfolioResult` itself.

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

### I-21 — Greeks recompile 23 XLA programs on every call {#i-21}

**Severity:** Medium · **Status:** ❌ OPEN — **performance only; every number is correct**

**Symptom.** A second, byte-identical `price_portfolio(request)` call in the same warm
process recompiles 31 XLA programs (23 of them in `engine/risk/greeks.py`) instead of
reusing cached ones. Nothing is *wrong* with the output — this costs wall time and makes a
profiler trace look compile-bound even after warmup.

Measured, three consecutive identical calls on the 4-trade demo portfolio:

| Run | Compilations | Wall |
|---|---:|---:|
| 1 (cold) | 208 | 26.5 s |
| 2 | **31** | 17.8 s |
| 3 | **31** | 16.0 s |

**The 23 Greeks recompiles, with exact callsites** (instrumented at
`jax._src.compiler.backend_compile_and_load`, the same event an xprof trace labels as XLA
compilation; cache hits do not reach it):

| Count | Program | Callsite |
|---:|---|---|
| 10 | `jit_price_fn` | `greeks.py:390,400` (`swap_theta`), `:588,600` (`swaption_theta`), `:708,720` (`bermudan_theta`), `:813` (`bermudan_vega`) |
| 5 | `jit_combined` | `greeks.py:222` (`_grad_and_hessian_diagonal`) |
| 4 | `jit_model_price_wrt_prefix` | `greeks.py:842` (`bermudan_vega`) |
| 4 | `jit_market_price_wrt_v_j` | `greeks.py:849` (`bermudan_vega`) |

**Cause — one mechanism, seven sites.** `jax.jit` keys its cache on **function identity**,
and every one of these jits a **closure built fresh on each call**. `_swap_price_fn`,
`_swaption_price_fn` and `_bermudan_price_fn` each return a new function object that has
captured that trade's prepared structure; `jax.jit(that_new_object)` is, as far as JAX is
concerned, a function it has never seen. Demonstrated in isolation:

```
fresh closure + jax.jit each call : 5 compiles for 3 calls
stable fn, constant as argument   : 1 compile  for 3 calls
ONE jitted closure, reused        : 1 compile  for 3 calls
```

This was introduced *by* the jitting work that removed ~600 eager dispatches
(see [Profiling & the Tracer](concepts/profiling.md) §3) — a large net win that left this
residue behind. It is recorded here rather than silently accepted because it is the only
thing now standing between this engine and an execution-dominated profile.

**What closing it requires — memoize the jitted wrapper, keyed on prepared structure.**

Cache `jax.jit(price_fn)` in a module-level dict keyed on the *prepared trade* rather than
on the closure's identity, so two calls with the same economics reuse one compiled program.
The key must be `static_key(prepared)` — the codebase's existing by-value normalizer
([`engine/models/static_key.py`](../engine/models/static_key.py)) — plus the curve's shape
and dtype.

Prototyped and verified on the European swaption path:

| | Call 1 | Call 2 | Call 3 | Result |
|---|---:|---:|---:|---|
| today | 11 | 1 | 1 | 10273.553365459014 |
| memoized | 1 | **0** | **0** | 10273.553365459014 |

Bit-identical output, and steady-state recompiles reach **zero**.

**The risk this must not introduce, and why the design avoids it.** A memo that returns a
program compiled for a *different* trade is silently wrong numbers — far worse than the
slowness it fixes. Two properties make that safe:

1. **The key must distinguish everything economically meaningful.** Verified directly
   against `static_key(prepare_swaption(cfg))`: `notional`, `fixed_rate`, `payer`,
   `swap_tenor`, `hw_sigma`, `hw_a` and `forward_start` each produce a *different* key,
   while an identical config reproduces the same one. No collisions. This works because
   `static_key` hashes NumPy arrays by **content** (`tobytes()`), not identity — the same
   property that already lets `_Prepared*` objects be `jax.jit` static arguments.
2. **Key on the PREPARED object, never the config.** `prepare_*` is what resolves a config
   into the schedule the compiled program actually depends on. Keying on the raw config
   would miss anything ORE's date generation derives (holiday rolls, accrual fractions), and
   those genuinely change the program.

**Bounded growth.** The cache must be an LRU (`functools.lru_cache`, or an explicit dict
with a cap), not an unbounded dict: one entry retains a compiled XLA executable, and a
long-lived server pricing thousands of distinct trades would otherwise leak. A pool worker
is long-lived by design, so this is a real constraint, not a theoretical one. `jax` exposes
`jax.clear_caches()` and each wrapper a `_clear_cache()` if an explicit eviction hook is
wanted.

**What it must not do.** It must not key on `id()` (each `prepare_*` call returns a fresh
object — every lookup would miss, and recycled ids could collide), must not be keyed on
anything mutable, and must not be applied to `bermudan_vega`'s per-bucket closures without
the same content-based key (they capture `bucket_times`/`bucket_values` prefixes that differ
per bucket, and must *not* share a program).

**Regression test.** Assert the steady-state recompile count is **0** for a repeated
identical Greeks call, and — the important negative — that a config differing only in
`notional`, `fixed_rate` or `swap_tenor` still produces its own correct, *different* answer.
A test that only checks the count would pass against a broken always-hit cache.
`tests/test_profiling_and_jit.py::TestCompileCounts::test_repeated_greeks_call_costs_one_compile_not_zero`
currently pins the *present* behavior and must be updated, not deleted, when this lands.

---

### I-22 — Calibration recompiles 8 XLA programs per call {#i-22}

**Severity:** Low · **Status:** ❌ OPEN — **performance only; every number is correct**

**Symptom.** The remaining 8 of I-21's 31 steady-state recompiles are in
`engine/calibration/lgm.py`:

| Count | Program | Callsite |
|---:|---|---|
| 6 | `jit__lambda` | `lgm.py:173`, `:189`, `:192` (`calibrate_lgm_sigma`) |
| 2 | `jit_scan` | `lgm.py:98` (`_bisect_bucket_sigma`) |

**Cause — a DIFFERENT mechanism from I-21, which is why it needs a different fix.** These
are not merely fresh closures; they bake **Python float constants** into the traced program:

- `_bisect_bucket_sigma` closes over `market_price` as a concrete `float`, so every bucket
  and every call traces a structurally identical `lax.scan` with a different embedded
  constant.
- `calibrate_lgm_sigma`'s `_jit_over_target` closures capture `target` and `final_sigma`
  the same way.

Confirmed in isolation — the distinction is exactly constant-vs-argument:

```
market_price baked in as a constant : 4, 1, 1 compiles across 3 differing calls
market_price as a traced argument   : 1, 0, 0
```

**A memo (I-21's fix) would NOT help here** and would actively hurt: the constants differ
legitimately per bucket, so a content-keyed cache would simply miss every time while adding
lookup cost and retention. Applying I-21's fix mechanically to this file would be the wrong
call.

**What closing it requires — promote the constants to traced arguments.**

Make `_bisect_bucket_sigma` take `market_price` as a JAX array argument rather than closing
over a float, and give it a stable (module-level, `@partial(jax.jit, static_argnums=...)`)
identity so the `lax.scan` compiles once and is reused across buckets and calls. Same for
the diagnostics repricing: pass the target's arrays in rather than capturing them.

**The constraint that makes this non-trivial, and must not be broken.** `price_fn` itself is
genuinely different per bucket — bucket *j*'s pricer depends on the `[s_0..s_{j-1}]` prefix
already calibrated, which is the whole structure of a bootstrap. So `price_fn` cannot become
a traced argument; it has to stay a static one, and only `market_price` moves. That caps the
achievable win at **one compile per distinct bucket count**, not zero. Realistically this
takes 8 → ~2.

**Why this is Low and I-21 is Medium.** Calibration runs once per distinct `rate_factor_index`
per job; Greeks run per trade. On the demo portfolio calibration is ~1.3 s against Greeks'
~18 s. Fix I-21 first — and note the two are independent, so I-21 can land alone.

**Do not "fix" this by raising the bisection tolerance or lowering `iterations`.** The 60
iterations are a correctness property (`rmse < 1e-8` is asserted); trading calibration
accuracy for compile count would be a real regression disguised as an optimization.

---

## FIXED — found during the TraderX EOD exchange (2026-09-15)

> Kept in ID order here rather than moved up into the FIXED section above, so the
> register reads chronologically and the I-NN anchors stay stable. All four are
> **FIXED**, each with a regression test verified to fail against the pre-fix code.
>
> **I-17 was found on 2026-09-16 during W1.3** and is listed here rather than in a new
> section because it belongs to the same integration work.

### I-13 — A negative curve index silently prices against the wrong curve {#i-13}

**Severity:** High · **Status:** ✅ FIXED · **Found:** 2026-09-15, by TraderX source review

**Symptom.** A `SwapConfig` with `discount_curve_index=-1` or `forward_curve_index=-1` prices
**cleanly, finitely, and wrongly** — against a curve the trade was never booked against. No
error, no warning. Only the `compute_greeks=False` path is affected, which is the default and
the path an EOD batch takes.

**Cause.** [`_base_npv_per_trade`](../engine/portfolio/request.py#L840-L841) indexes the curve
list directly:

```python
disc_curve = market_config.rates.initial_zero_curves[cfg.discount_curve_index]
fwd_curve  = market_config.rates.initial_zero_curves[cfg.forward_curve_index]
```

Python's negative indexing wraps `-1` to the **last** curve rather than raising.
`_swap_curve_configs` — which validates and raises — guards only the Greeks path, and runs
at [request.py:691](../engine/portfolio/request.py#L691), *after* base pricing at
[request.py:670](../engine/portfolio/request.py#L670).

**Measured, not inferred.** With two deliberately different curves:

| Indices | `compute_greeks=False` | `compute_greeks=True` |
|---|---|---|
| `fwd=7` (out of range) | `IndexError` | `IndexError` |
| `fwd=-1` (negative) | **prices silently** ❌ | `ValueError`, named |
| `disc=-1` (negative) | **prices silently** ❌ | `ValueError`, named |

```
fwd_idx= 0 (booked, flat curve) : -5,913.9266
fwd_idx= 1 (steep curve)        : -5,857.0074
fwd_idx=-1 (INVALID)            : -5,857.0074   ← identical to curve 1
magnitude of silent error       :     56.92 USD on 2,000,000 notional
```

The 56.92 figure is **not a bound** — it reflects how far apart the two test curves happen to
be. The error scales with curve separation and has no ceiling.

**The irony worth recording.** `_swap_curve_configs`'s own docstring states that a hard
failure is deliberate because silently substituting *some* curve "would produce a
plausible-looking sensitivity computed against a curve the trade was never booked against."
That is a precise description of what the base pricing path does two hundred lines earlier.

**Why the suite missed it.** `tests/test_portfolio_gap_fixes.py:237` calls
`_swap_curve_configs` **directly** and asserts it raises. Nothing routes a bad index through
`price_portfolio`. **The validator was tested; the caller that skips it was not.** A unit test
on a guard proves nothing about paths that never reach the guard. 1,138 tests passed with this
defect live.

**Fix.** `_validate_swap_curve_indices` in
[`engine/portfolio/request.py`](../engine/portfolio/request.py) range-checks every
`SwapConfig`'s two curve indices, called from `validate_portfolio_against_simulation` —
which `price_portfolio` already ran **before any JAX work**, so one check now covers every
pricing path (base NPV, the cube, Greeks) rather than each indexing site having to remember
to guard itself. That is the structural point: the omission *was* a per-site guard being
forgotten, so the fix removes the need for per-site guards.

Placed alongside the existing `rate_factor_index` range check in the same function — a swap
carries curve indices instead of a `rate_factor_index`, so it was the one trade type that
validator skipped entirely.

**Verified.** `tests/test_portfolio_gap_fixes.py::TestCurveIndexValidatedBeforeAllPricing`
(12 tests) drives the full 2×2 **through `price_portfolio`**, not through the helper.
Against the pre-fix code **8 of the 12 fail** — including both negative cases at
`compute_greeks=False`, which returned a plausible NPV instead of raising. The 4 that pass
pre-fix are the valid-index controls and the two negative cases at `compute_greeks=True`
(already guarded), which is exactly the asymmetry the issue describes.

All four invalid cases now raise the same named `ValueError` regardless of `compute_greeks`,
and the bare `IndexError` is gone. The valid-index NPV is unchanged (`-5913.926602071378`).

---

### I-14 — `generate_paths(precision=32)` leaks `jax_enable_x64=False` {#i-14}

**Severity:** High · **Status:** ✅ FIXED · **Found:** 2026-09-15

**Symptom.** A float64 request silently produces **float32** results. JAX emits only a
`UserWarning`:

```
UserWarning: Explicitly requested dtype float64 requested in asarray is not available,
and will be truncated to dtype float32.
```

The returned arrays are finite, plausible, and wrong in the last several digits — and the
`PortfolioResult` carries nothing recording that the requested precision was not honored.

**Cause.** `generate_paths` set `jax_enable_x64` (a **process-global** JAX/XLA flag) to
`precision == 64` on entry and **never restored it**. Its own docstring claimed it toggled
the flag "for the duration of this call" — the documented contract and the behavior
disagreed, and the docstring was the accurate description of what callers needed.

Reproduced directly:

```
initial x64: True
after generate_paths(precision=32), x64: False
float64 request yields: float32      ← silent truncation
```

**Scope, stated precisely.** `price_portfolio` was **never** affected: it re-enables x64
immediately after calling `generate_paths`, deliberately and with a comment explaining why.
The exposure was every *direct* caller — demos, tests, every `engine/instruments/*`
`__main__` block, and any future caller of `engine.simulation` — plus anything running
afterward in the same process.

**This corrects an earlier reading of this issue**, which attributed the leak to
`price_portfolio` and to sequential jobs of differing precision. Probing showed
`price_portfolio` leaves x64 `True` on both paths; the actual leak is one function lower.

**Observed in the test suite.** Two tests fail in a full run and pass in isolation:

| Test | Behavior |
|---|---|
| `test_portfolio_entrypoint.py::...::test_risk_var_es_override_recasts_npv_cube_for_risk_only` | Passes alone, and with its own file (29 passed); fails in the full run |
| `test_var_es.py::...::test_synthetic_arbitrary_shaped_cube` | Passes alone; fails in the full run |

The first surfaces as `AssertionError: risk metric 'ES_95_tailCount' not float32` /
`dtype('int64') == float32`. **That assertion is not the bug** — the test already excludes
`_tailCount` keys deliberately ([line 515](../tests/test_portfolio_entrypoint.py#L515)) and
documents why counts stay integral. The failure is the *downstream symptom* of x64 having
been turned off by an earlier test, which changes what the risk dict contains. Diagnosing it
as a bad assertion and "fixing" the test would have buried the real defect.

**Fix.** `generate_paths` now scopes the flag with an `_x64_enabled` context manager that
restores the **prior** value in a `finally`, making the docstring's "for the duration of this
call" true. Restoring the prior value rather than unconditionally re-enabling is deliberate:
a caller legitimately running an all-float32 process should be left in that state, not
quietly promoted to float64 by having called a simulation.

The pipeline body moved to `_generate_paths_inner` **only** so the flag could be scoped
without re-indenting ~175 lines; no pipeline logic changed.

**Verified.** `tests/test_market_model.py::TestGeneratePathsEdgeCases::
test_precision_32_restores_the_global_x64_flag`, confirmed to fail against the pre-fix code.
It asserts on the **ambient flag and a plain float64 array**, not on the simulation's own
output — the neighboring
`test_sequential_precision_switches_produce_correct_dtype_each_time` passes either way,
because every call re-sets the flag on entry and is therefore self-correcting. Only code
requesting float64 *without* going through `generate_paths` first ever saw the leak, which
is why that existing test never caught it.

**Residual, not closed by this fix.** `PortfolioResult` still carries no record of the
*realized* dtype, so a truncation from any other cause would remain invisible in the result.
Recording realized precision on the result composes with **I-12**'s worker-device reporting
and is the remaining work; it is not required to close this leak.

---

### I-15 — Worker-pool concurrency test could not observe concurrency {#i-15}

**Severity:** Low · **Status:** ✅ FIXED · **Found:** 2026-09-15

`tests/test_worker_pool.py::TestWorkerPoolConcurrency::test_same_tier_jobs_also_overlap_across_pool_workers`
failed intermittently on `assert overlap_count >= 1`.

**Cause — the probe, not the pool.** The test primed the pool, then submitted the same tiny
portfolio the other tests use. Instrumenting the worker PIDs and durations showed why:

```
prime pid: 58060  duration: 2.531s      ← cold JIT
round 0: pids=58060,58060  dur1=0.015  dur2=0.000  overlap=False
round 1: pids=58060,58060  dur1=0.000  dur2=0.016  overlap=False
round 2: pids=58060,58060  dur1=0.015  dur2=0.016  overlap=False
```

After priming, the JIT cache is warm and each job takes **~15ms** — job 1 finished before the
executor handed job 2 to a worker, so both ran on **one** process every round. **Two jobs
that never coexist in time cannot demonstrate concurrency at all**, so the test could only
pass by luck. The pool itself was correct throughout (`max_workers=2`, and a plain
`ProcessPoolExecutor` probe confirmed it does spread work across two workers).

**Fix.** The test now occupies workers with a deliberately slow `_sleep_job` (1.0s) long
enough that a second worker is genuinely required, and asserts **both** halves: distinct
worker PIDs (the dispatch mechanism) *and* wall-clock overlap (its observable payoff). Both
are now deterministic rather than racing the scheduler. It sleeps rather than prices because
the pool's *dispatch* is what is under test; `test_cross_tier_jobs_correct_and_concurrent`
already covers concurrency with real pricing work plus numerical correctness.

**Verified.** Passes 3/3 consecutive runs, and the file dropped from ~16s to ~2.3s.

**A note on why this was worth fixing rather than muting.** An intermittently red suite
trains people to ignore red — which is how I-13 survived 1,138 passing tests. The first
attempted fix (assert distinct PIDs only) *also* failed, and that failure is what surfaced
the real cause: a genuine signal that would have been lost by simply relaxing the assertion.

---

### I-17 — A malformed note date failed the entire bundle {#i-17}

**Severity:** Medium · **Status:** ✅ FIXED · **Found:** 2026-09-16, while writing W1.3's
own test suite

**Symptom a consumer would see.** One Treasury note with a malformed or absent date in its
reference terms — a single bad `schedule[i].endDate`, or a missing `maturityDate` — would
fail the **whole job** with an unhandled `BillPricingError`, rather than returning a refused
item alongside everyone else's results. A 200-row bundle would lose all 200 results to one
bad row.

This directly violates the contract both pricers' docstrings state: *"one unpriceable row
must not fail the other 200."*

**Cause.** `engine/integration/note.py` reused `bill._parse_date` rather than defining its
own. That helper raises `BillPricingError`, but the pipeline's note path catches
`NotePricingError`:

```python
except NotePricingError as exc:          # never matched a BillPricingError
    refusal = CalculationOutcome.unsupported(...)
```

so the exception propagated straight through `_note_outcomes` → `_build_item` →
`price_bundle`. The refusal machinery was correct; the exception simply never reached it.

Two smaller faults rode along: the refusal message read *"A bill cannot be priced without
its maturity"* for a **note**, which sends a consumer looking at the wrong instrument; and
the reason code, had it been caught, would have been the bill module's constant rather than
the note's.

**Why no existing test caught it.** Every W1.2 test exercised the *bill* path, where the
exception type matches by construction. The note path had no malformed-date test because the
note path did not exist. This is working rule 9 in miniature: a green suite was evidence
about the tests, not the code.

**Fix.** `note.py` defines its own `_parse_date` raising `NotePricingError`. The four
duplicated lines are deliberate and documented in the function's docstring — sharing the
helper is precisely what coupled the two modules' failure semantics. `_iso`, a pure
formatter with no exception behaviour, is still shared.

**Regression tests** — `tests/test_integration_note.py::TestRefusalsAreNotePricingErrors`,
verified to fail against the pre-fix code. Three pin the exception type and the wording; the
fourth pins the consequence that actually mattered, asserting through the pipeline that a
malformed row comes back as a refused *item* rather than taking the bundle down with it.

---

### I-19 — Accrual tolerance rounded the bound it exists to enforce {#i-19}

**Severity:** Medium · **Status:** ✅ FIXED · **Found:** 2026-09-16 by TraderX's v5 review

**Symptom a consumer would see.** A perfectly good note bundle refused with
`ACCRUAL_MISMATCH` at certain face amounts, reporting a reconciliation failure where none
existed.

**Cause.** `accrual_mismatch_tolerance` computed

```python
round(0.5 * 10**-fraction_decimals * abs(face), 2) + 0.01   # wrong
```

The first term is the exporter's worst-case HALF_EVEN rounding error. Rounding *that* to
cents truncates the very quantity the tolerance exists to admit, making the bound **tighter
than agreed**. At 124,000 face the true error is 0.062, so the agreed tolerance is 0.072 —
but rounding gave 0.06 and a tolerance of 0.07.

Because `ACCRUAL_MISMATCH` is a hard refusal, a too-tight tolerance turns a safety check into
an outage: it rejects data that is within the exporter's own stated rounding.

**Why no existing test caught it.** Every test used the delivered fixture's **$100,000**
face, where the rounding error is exactly 0.05 — already a whole number of cents, so
`round(0.05, 2) == 0.05` and both formulas agree at 0.06. The bug was invisible at the one
face amount the suite ever used. `test_covers_the_exporters_worst_case_rounding_at_every_size`
parametrised over face sizes but compared against `>= 0.5e-6 * face`, omitting the `+0.01`,
so it passed too.

**Fix.** Return `rounding_error + 0.01`, unrounded.

**Regression tests** — `tests/test_integration_note.py::TestToleranceIsDerivedNotConstant`:
`test_the_rounding_bound_is_not_itself_rounded` (the 124,000 case, asserting both the correct
0.072 and the absence of the wrong 0.07), `test_the_fixture_face_is_unchanged_by_the_fix`,
and a parametrised `test_never_narrower_than_the_exporters_rounding_error`. Verified to fail
against the pre-fix code.

---

### I-20 — Impossible calendar dates aborted the whole bundle {#i-20}

**Severity:** **High** · **Status:** ✅ FIXED · **Found:** 2026-09-16 by TraderX's v5 review

**Symptom a consumer would see.** A single instrument whose terms carried a date that
*parses* as three integers but names a day that cannot exist — `2025-02-30`, `2025-13-01`,
`2025-02-29` — failed the **entire job** with an unhandled `RuntimeError`. A 200-row bundle
returned nothing at all because of one typo.

**Cause.** `ORE.Date(30, 2, 2025)` raises **`RuntimeError`** ("day outside month (2)
day-range [1,28]") — SWIG surfacing QuantLib's C++ `std::runtime_error`. Both date parsers
caught only `(ValueError, TypeError)`:

```python
except (ValueError, TypeError) as exc:   # never matched ORE's RuntimeError
```

So `not-a-date` was handled correctly (it fails at `int()`, a `ValueError`) while
`2025-02-30` escaped every handler — the pricer's, the pipeline's, and `price_bundle`'s.

**Relationship to [I-17](#i-17) — the same symptom, a different cause.** I-17 was an
exception *type* mismatch (a note raising `BillPricingError`); this is an exception *class*
gap (an exception neither module anticipated). **The I-17 fix could not have prevented it**,
and its regression test did not catch it, because that test constructs a note whose date is
merely *absent* rather than impossible. Two independent routes to the same contract
violation, found six hours apart.

**Why no existing test caught it.** Every malformed-date test in the suite used
`"not-a-date"` — a string that fails at `int()` and so takes the `ValueError` path. Nothing
tested a date that was numerically well-formed but calendrically impossible, which is the
only input that reaches `ORE.Date` and raises.

**Fix.** Three layers, because the date parser was one instance of a general hazard rather
than the whole of it:

1. `bill._parse_date` and `note._parse_date` catch `RuntimeError` alongside the Python date
   exceptions, refusing with `TERMS_INCOMPLETE`.
2. All three pipeline handlers (`_bill_outcomes`, `_note_outcomes`, `_equity_outcomes`) catch
   `RuntimeError` too. **Every pricer calls into ORE**, so any ORE precondition failure —
   not just a date — arrives as `RuntimeError`; the narrow handler would have let all of
   them abort the bundle.

**Regression tests** — `tests/test_integration_note.py::TestImpossibleCalendarDates`, 20+
cases exercised **through `price_bundle`** as TraderX asked: the bundle still returns, both
rows survive, the outcome is an identified item-level refusal naming the offending value, and
coverage still sums. The bill path is covered too — it had the identical parser and the
identical gap; the reviewer happened to try the note. Verified to fail against the pre-fix
code.

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
  [The Portfolio Entry Point](reference/portfolio-entrypoint.md),
  [Profiling & the Tracer](concepts/profiling.md) (how to measure where a job's time
  actually goes, and the known cost characteristics of the Greeks path).
