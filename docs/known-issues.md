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
| **ASSUMPTION** | Nothing known to be broken. The engine acts on an **unconfirmed reading** of an external contract, and the reading may be wrong. Registered so a deliberate interpretation does not pass for a settled fact. |
| **PARTIAL** | Closed on one path and open on another. Used only where the split is real and nameable — not as a softer word for OPEN. [I-08](#i-08) is the current case: the EOD path is durable, the portfolio path is not. |

**On ASSUMPTION, added 2026-09-16 with [I-23](#i-23).** The other three statuses all describe
something the code gets wrong. This one describes a decision made in the absence of an answer
— where the risk is not a wrong number but a **wrong premise**, invisible precisely because
the code behaves exactly as designed. An entry here is a standing question to an external
party, not a bug queue item, and it closes when the question is answered rather than when
code changes.

Last full verification (2026-09-17, after **W0.8**): **1,777 passed, 0 failed** (16m55s) —
the complete suite, nothing excluded, summary line printed, exit code 0. That is 1,718 at the
previous commit plus W0.8's 59 new tests (51 in `tests/test_integration_publication.py`, and
8 net added to `tests/test_integration_eod_routes.py`, which goes 56 → 64), so the delta
reconciles exactly: 1,718 + 51 + 8 = 1,777.

**Two corrections to figures previously recorded here**, both found by re-collecting rather
than re-reading:

1. The total here briefly read **1,770**, reconciled as "52 new tests (44 + 8)". Both halves
   were wrong. `tests/test_integration_publication.py` collects **51**, not 44 — the 44 was a
   count of `def test_` lines, which undercounts every parameterized case. And 1,770 was
   taken before the last of the `eod_routes` tests landed.
2. Before that it read **1,716 after W1.5**, reconciled as "1,618 + the 98 W1.5 tests".
   Re-collecting that commit (`pytest --collect-only` at `1e078f3`) yields **1,718** — so
   that total and its arithmetic were off by two.

Counts here are now taken from `pytest --collect-only`, never transcribed from a remembered
summary line and never derived by grepping for `def test_`. A register whose own header
overstates its verification undermines every status in it. Earlier figures, for history:
1,618/0 after W1.6, 1,452/1 after the v5 fixes, 1,384/2 after W1.4, 1,324/2 after W1.3,
1,217/2 after W1.2.

**One caveat, and it is about the runner rather than the code.** [I-27](#i-27) makes a
whole-suite run *intermittently* hard-abort inside XLA compilation, killing the process with
no summary at all. This run completed cleanly; an earlier identical one did not — and during
this same session a run interrupted by an unrelated `git stash` of the working tree also
terminated without a summary, which is a reminder that the tree must be left alone for the
duration of a run. So a green result here is real but **not reliably repeatable on demand** —
always confirm a summary line was actually printed before calling a run green.

A green suite is evidence about the *tests*, not proof about the *code* — working rule 9,
which this register exists to embody. Of the defects found during this integration, three came
from TraderX reading source or independently reproducing numbers ([I-13](#i-13),
[I-19](#i-19), [I-20](#i-20)), three from reviewing or reasoning about my own code
([I-25](#i-25), [I-26](#i-26) and the W1.6 `submissionId` bug), and **none from running this
suite**.

Two previously-recorded header caveats are now resolved:

- The `pydantic` environment gap is gone — the dependency is installed (2.13.5),
  `tests/test_api.py` collects, and the run contains zero `ModuleNotFoundError`. (The old
  header both claimed this was resolved *and* described the failures as current; that
  contradiction is removed.)

  > **⚠ Always run `.venv/Scripts/python.exe -m pytest`, never the bare `python`.** During
  > W1.5 the system interpreter was used by mistake, where `pydantic` and `jsonschema` are
  > absent. That made `tests/test_api.py` and `tests/test_integration_schema.py`
  > **uncollectable — 77 tests silently missing at today's counts** (31 + 46) — and produced
  > a full-suite count of 1,663 against the venv's 1,709, a discrepancy that looked like a
  > regression and was purely environmental. A count taken from the wrong interpreter is not
  > comparable to anything recorded here. See [I-25](#i-25). (This bullet previously said
  > "46", which was `test_integration_schema.py`'s share alone.)
- `test_cross_tier_jobs_correct_and_concurrent` — the long-running flake — **failed again**.
  W0.8 ran the full suite twice: run 1 passed it (1,770 / 0), run 2 failed it
  (1,771 passed, 1 failed). An earlier version of this bullet claimed it had "passed again …
  its second consecutive clean full-suite pass" — that was written from run 1 and **run 2
  falsified it**. The intermittency is the whole point: it passes in isolation (12.28s,
  re-run immediately after run 2), it failed in every full run from W1.2 through W1.5, and it
  has now passed and failed in full runs of the *same* code an hour apart. A **third** full
  run, taken during the documentation pass that added [I-28](#i-28), passed it again
  (1,777 / 0, 16m55s) — so the tally across W0.8 stands at two passes and one failure, which
  changes nothing. That is consistent with a load-dependent wall-clock overlap assertion
  rather than a defect, and it is a standing warning that **a single green run of this test
  means nothing in either direction**. It shares [I-15](#i-15)'s premise and remains a
  follow-up.

**This line reports what a full run actually produces.** Earlier figures here (824, 1092,
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
| [I-08](#i-08) | Job store is in-process; lost on restart | Medium | ⚠️ PARTIAL — EOD path durable (W0.8); the portfolio path's `_JOBS` dict is unchanged |
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
| [I-23](#i-23) | `accrualBasis` strictness is an **assumption** on an unanswered question | Medium | ⚠️ ASSUMPTION — may refuse bundles TraderX considers valid |
| [I-24](#i-24) | Bonds have no scenario NPV, so no VaR/ES — refused, not approximated | Medium | ❌ OPEN — refusal path landed (W1.5) |
| [I-25](#i-25) | A **scalar** Greek crashed the HTTP result serializer | Medium | ✅ FIXED |
| [I-26](#i-26) | Greeks for a bond maturing **tomorrow** crashed on the theta reprice | Low | ✅ FIXED |
| [I-27](#i-27) | Long full-suite runs **hard-abort inside XLA compilation**, with no summary line | Medium | ❌ OPEN — located, not root-caused |
| [I-28](#i-28) | `python -m engine.risk.var_es`'s **own demo crashes**: it omits `evaluation_date`, so its swap schedules off today | Low | ❌ OPEN |

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

**As originally written (pre-W1.2):** `engine/instruments/` contained exactly four modules,
all rate derivatives (swap, European / Bermudan / American swaption), and there was **no
pricer** for Treasuries, corporate bonds, cash equities/ETFs, or listed options — all of
which appear in TraderX's schema-3 position export.

**Today** Treasuries price on both paths (W1.2/W1.3 at the integration boundary, W1.5 as
`engine/instruments/treasury.py`, now a fifth module). Corporate bonds, cash equities and
listed options remain unpriced — see "Scope of that, stated precisely" below for exactly
which of those is missing a *pricer* versus missing *market data*.

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
- ~~`engine/instruments/` is **unchanged**~~ — **closed by W1.5 (2026-09-17).**
  `engine/instruments/treasury.py` adds `BondConfig`, and a direct Python caller of
  `price_portfolio` now gets a bond's t=0 NPV, its per-trade breakdown entry, and
  Delta/Gamma/Theta. **With one bounded exception:** a bond has no scenario NPV, so no
  VaR/ES — refused explicitly rather than approximated, tracked as [I-24](#i-24).

Status stays **OPEN**: the issue is "no bond, equity, or listed-option pricer". Treasuries
now price *and* are reachable from the portfolio path; an equity is refused for a
*market-data* reason rather than a missing pricer ([I-18](#i-18)); corporate bonds and
listed options are still absent entirely.

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

**Severity:** Medium · **Status:** ⚠️ PARTIAL — **EOD path durable (W0.8, 2026-09-17); the
portfolio path's `_JOBS` dict is unchanged**

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

**Closed on the EOD path by W0.8's second half (2026-09-17).**
[`engine/integration/publication.py`](../engine/integration/publication.py) adds the durable
store the earlier mitigation was missing, and with it the crash-safety design from
[plan §W0.8](planning/traderx-integration-plan.md):

- **The four-step publication protocol** — stage to a temp path, verify the hash of what was
  *actually written* (not what was meant to be), atomically publish the manifest, then advance
  the pointer. Every crash window leaves a coherent store: nothing partial is ever
  discoverable.
- **The manifest is the commit point; the pointer is a cache.** Lookup falls back to a
  **scan** over published manifests whenever the pointer is missing, torn, or behind, and
  reconciles the pointer as a side effect. This closes the window TraderX found in v3 — a
  crash between publish and pointer advance previously left a complete result that no lookup
  could find.
- **Completed and failed attempts survive a restart**, addressable by `attemptId`, and
  **idempotent submission survives it too**: a coordinator retrying a lost response after a
  bounce recovers its original attempt rather than starting a duplicate overnight batch.

**What remains true, and is deliberate.** A *running* attempt is still memory-only and is
still lost on restart — it is never published, because writing one would make an in-flight
computation discoverable as a finished answer. After a restart such a job reports as unknown,
the coordinator resubmits, and the workload key makes the recomputation identical. That is an
infrastructure event, not a financial one.

**One ordering defect was found after the fact and fixed.** Publication originally ran
*after* the in-memory transition, so a store failure left an attempt `completed` in memory
with nothing on disk — a result this process reported as finished and no restart could find,
and which the immutability guard then refused to let anyone retry. The transition now happens
only if the manifest lands, applying the store's own commit-first rule to the in-memory
attempt. The *failure* path deliberately keeps the opposite ordering: `fail()` marks the
attempt failed whether or not publication succeeds, because leaving it `running` would report
an in-flight job to a coordinator that would wait forever, which is worse than the
`UNKNOWN_WORKLOAD` a restart yields.

**Still open, and why this issue is not fully closed.** The portfolio path's `_JOBS` dict in
[`engine/api/routes.py`](../engine/api/routes.py) is **untouched** — this work is EOD-only.
The store is also single-machine: it is thread-safe within a process (an unguarded sequence
counter was measured issuing 2 distinct values across 30 concurrent publications, now locked),
but two engines on two machines do not coordinate.

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

### I-24 — A bond has no scenario NPV, so no VaR/ES {#i-24}

**Severity:** Medium · **Status:** ❌ OPEN — **refusal path landed (W1.5); the model has not**

**Symptom.** A `BondConfig` in a `PortfolioRequest` cannot produce VaR or ES. Submitting one
with the default `scenario_risk=True` is **refused** with `ScenarioPricingNotSupported`,
naming the trade. The caller must set `scenario_risk=False`, which returns real
`base_npv`/`base_npv_per_trade`/`greeks` alongside an **empty** `risk` dict and a
zero-width `npv_cube`.

**Cause.** `price_portfolio`'s `npv_cube` is `[Scenarios, TimeSteps, Trades]` — each column
is a trade's *conditional* NPV at each simulated future step, and `engine.risk.var_es` turns
those columns into VaR/ES. The four rate-derivative types fill their columns from simulated
Hull-White paths. A bond, as priced by `engine.instruments.treasury`, is closed-form
arithmetic against **one deterministic curve**: no stochastic driver, no time evolution, and
therefore nothing to vary across a scenario axis.

**Why this is a refusal and not a zero — measured, not argued.** The only way to fill the
column without a model is to broadcast one t=0 number across every entry. That was
implemented and run through the real `price_portfolio`, on a $100,000 bill priced at
$98,401.95:

| Metric | Broadcast-constant column |
|---|---|
| `VaR_95` / `VaR_99` | **0.00** |
| `ES_95` / `ES_99` | **NaN** |

A consumer reading `VaR_95 = 0.00` concludes the position carries no risk. It is not
conservative, not approximate, and not labelled — the risk is **absent, wearing the shape of
a measurement**. (The NaN does *not* propagate to the portfolio aggregate, which was also
checked: ES differences the P&L across trades first, so a constant column cancels. That
makes the failure quieter, not safer — the per-trade number is the one that misleads.)

**Why `risk` is empty rather than zero-filled.** An empty dict asserts nothing; a `VaR` key
holding 0.00 asserts a *measured absence of risk*. Only the first is true.
`PortfolioResult.scenario_risk_available` carries the distinction onto the **result**, since
a consumer holding a result object has no access to the request that produced it — without
it, an empty `risk` is ambiguous between "not requested" and "computed and found to be
nothing".

**What closing it requires — a bond scenario model, not plumbing.** Each simulated scenario's
rate state must be repriced through the bond's own schedule: build a zero curve per
`[scenario, step]` from the Hull-White state, then rerun the discounting. That is genuine
modelling work with its own validation burden (a bond repriced off an HW short rate needs
its discount curve reconstructed consistently with how the swap pricer does it, or the two
instruments carry incompatible risk in one portfolio total).

**Do not close it by broadcasting, zero-filling, or defaulting `scenario_risk` to `False`.**
The first two produce the table above. The third would silently strip VaR/ES from every
existing swap portfolio that never asked for it — turning a bond-shaped gap into a
portfolio-wide regression.

**Verified.** `tests/test_treasury_instrument.py::TestScenarioPricingIsRefused` (4 tests,
one of which *measures* the VaR-0/ES-NaN outcome so the justification is pinned rather than
remembered) and `tests/test_portfolio_bond_wire_through.py::TestScenarioRiskIsRefusedForBonds`
(5 tests). The broadcast implementation was patched in and **4 of 5 fail against it**
(working rule 3).

---

### I-25 — A scalar Greek crashed the HTTP result serializer {#i-25}

**Severity:** Medium · **Status:** ✅ FIXED · **Found:** 2026-09-17, during W1.5

**Symptom.** `GET`ting a completed job whose portfolio contained a bond with
`compute_greeks=True` raised `TypeError: 'float' object is not iterable` inside
`PortfolioResultSchema.from_dataclass`. The job **priced correctly** — the failure was
purely in serializing the answer, so the work was done and then thrown away with a 500.

**Cause.** `engine/api/schemas.py`'s `GreeksSchema.from_dataclass` converted every non-theta
Greek with:

```python
values[key] = [float(v) for v in np.asarray(val).tolist()]
```

A 0-dimensional array's `.tolist()` returns a **bare Python float**, not a list, so the
comprehension tries to iterate a scalar.

**Why it went unnoticed until W1.5.** Every pre-existing Greek is a per-pillar **vector** —
`swap_delta_gamma` returns `discount_delta`/`forward_delta` arrays, one entry per curve
pillar. The unconditional iteration was correct for all four rate-derivative types. A
`BondConfig` prices off a single curve with one parallel bump, so its `delta`/`gamma` are
genuine **scalars** — the first 0-d Greek in the codebase.

**Fix.** `np.atleast_1d` before `.tolist()`, normalizing the scalar case to a one-element
list so `values` stays uniformly a list-per-Greek rather than sometimes a float. One line;
the vector path is byte-identical.

**Verified.** `tests/test_api_bond_schemas.py::TestBondGreeksSerializeOverHttp` (5 tests).
**3 fail against the pre-fix code**, including `test_a_full_bond_result_serializes_end_to_end`
which drives the real `price_portfolio` → `from_dataclass` → `model_dump_json` path.
`test_a_vector_greek_is_unchanged` is the negative control confirming the fix did not alter
the existing per-pillar behaviour.

**Note on how this was found.** By reading the conversion code while adding the bond schema
and predicting that a 0-d array would break it — then confirming it in one line before
writing any test. No existing test could have caught it: the bond is the first scalar Greek,
so there was nothing to exercise the path.

> **⚠ Process finding, worth more than the bug.** This was found while running the **system
> Python**, where `tests/test_api.py` was uncollectable for want of `pydantic`. The project
> has a **`.venv/`** that has always had it — so the HTTP tests were passing there all along,
> and the "uncollectable" state was an artifact of the wrong interpreter, not a real gap.
>
> The same mistake hid 46 further tests (`tests/test_integration_schema.py`, which needs
> `jsonschema`) and produced a full-suite count of **1,663** against the venv's **1,709** —
> a 46-test discrepancy that looked like a regression and was purely environmental.
> **Always run `.venv/Scripts/python.exe -m pytest`, not the system `python`.** A suite that
> cannot import a module reports nothing for it, and a count taken from the wrong interpreter
> is not comparable to the recorded baseline (working rules 9 and 10).

---

### I-26 — Greeks for a bond maturing tomorrow crashed on the theta reprice {#i-26}

**Severity:** Low · **Status:** ✅ FIXED · **Found:** 2026-09-17, during W1.5

**Symptom.** `compute_greeks=True` on a portfolio containing a bond whose maturity is
**exactly one day** after the evaluation date raised:

```
BondPricingError: maturity_date 2025-06-03 is not after evaluation_date 2025-06-03.
A matured bond has no remaining cashflow to discount ...
```

The bond was **not** matured, priced perfectly well in the same run, and the error named a
date the caller never supplied. `base_npv` succeeded; only the Greeks call died — so a
single near-maturity position failed the whole portfolio's Greeks.

**Cause.** `_bond_greeks` computes theta by repricing with `evaluation_date + 1`. For a bond
maturing tomorrow that lands **exactly on** maturity, which `BondConfig.__post_init__`
refuses to construct — correctly, since a bond with no remaining cashflow is a settlement
question rather than a pricing one. The refusal is right for the reprice and wrong as a
failure of the entire Greeks call.

**Fix.** Guard the reprice on `maturity_date > evaluation_date + 1`. Delta and Gamma are
unaffected and still reported; **theta is omitted**, not zeroed. There is no next day on
which the instrument still exists, so its decay is *undefined*, not nil — and `0.0` would
assert a measured absence of time decay on precisely the bond that decays fastest. Same
reasoning as the omitted Vega.

**Verified.** `tests/test_portfolio_bond_wire_through.py::TestBondGreeksReachThePortfolioPath`
— `test_a_bond_maturing_tomorrow_does_not_crash_the_greeks` and
`test_theta_is_omitted_not_zeroed_at_the_maturity_boundary`, **both verified to fail against
the pre-fix code**. `test_theta_is_present_one_day_the_other_side_of_the_boundary` is the
control: a bond maturing in *two* days still has theta, so an unconditional omission would
not pass.

**How it was found.** By asking what `replace(cfg, evaluation_date=+1)` does at the edge of
the constructor's own validity, and checking — not by a failing test. No fixture had a bond
that close to maturity.

---

### I-27 — Long full-suite runs hard-abort inside XLA compilation {#i-27}

**Severity:** Medium · **Status:** ❌ OPEN — **located, not yet root-caused**
**Found:** 2026-09-17, while verifying W1.5

**Symptom.** A long `pytest tests/` run dies with `Fatal Python error: Aborted` and
**no summary line at all**. There is no failure report — the process is gone. Separately,
`tests/test_bermudan_swaption.py` has been seen to fail intermittently
(`3 failed, 52 passed`, then `4 failed`, then clean) without aborting.

**Where the abort actually is.** The faulthandler traceback puts the crashing thread inside
**JAX's XLA compiler**, not in any pricer:

```
jax/_src/compiler.py:353  backend_compile_and_load
jax/_src/pjit.py:1175     _pjit_call_impl_python
engine/simulation/market_model.py:689  _generate_paths_inner
engine/portfolio/request.py:729        price_portfolio
tests/test_api_bond_schemas.py:173     test_a_full_bond_result_serializes_end_to_end
```

Other threads sit in `concurrent/futures/process.py` and `multiprocessing/queues.py` — i.e.
**a `ProcessPoolExecutor` is alive while the parent process compiles an XLA program**.

**The leading hypothesis, and its limits.** `engine/portfolio/worker_pool.py` caches pools in
a module-level `_POOLS` dict that lives for the interpreter's lifetime.
`tests/test_worker_pool.py` tears its pools down in an autouse fixture whose own comment says
it exists *"so later test modules don't inherit idle worker processes"* — but
**`tests/test_api.py` creates pools and never calls `shutdown_pools`**. A later in-process
XLA compile then runs with live worker children attached.

**That hypothesis is not proven.** Pairing the modules directly does *not* reproduce it:

| Attempted reproduction | Result |
|---|---|
| `test_api.py` + `test_api_bond_schemas.py`, 3× | **passed** (52 each time) |
| `test_worker_pool.py` + `test_api_bond_schemas.py` | passed (that module cleans up) |
| `test_api_bond_schemas.py` alone, repeatedly | passed (21) |
| `test_bermudan_swaption.py` alone, 6× consecutively | passed (55 each) |
| Same, under deliberate CPU contention from 2 concurrent JAX pytest processes | passed |
| Full suite (~1,100 tests in, Bermudan file *excluded*) | **ABORTED** |
| The *same* full-suite command, rerun | **1,661 passed, 0 failed** (10m55s), clean summary |
| Full suite again, Bermudan file **included** | **1,716 passed, 0 failed** (11m24s), clean summary |

So it needs accumulated whole-suite state, not any two modules — and even then it is
**intermittent**: the identical command that aborted later completed cleanly end to end.
**The abort is not specific to the Bermudan file** — it happened with that file excluded
entirely, in `tests/test_api_bond_schemas.py`.

**Not caused by W1.5.** `git stash` of all W1.5 work reproduced the Bermudan failures on
pristine code. W1.5's only contact with Bermudan code is adding `BondConfig` to the
`TradeConfig` union plus two comments. The W1.5 test named in the traceback is simply the
*victim* — it is the point where a fresh XLA compile happens late in a long run.

**Why Medium.** Two subsequent full runs completed cleanly (1,661 and 1,716, both with real
summary lines), so the suite *is* green — but it makes that result **not reliably obtainable
on demand**, and it fails
in the most deceptive way available: a dead process with no summary. That is how a run dying
at 4% was briefly taken for a pass — the shell's `echo EXIT=$?` had captured the redirect
rather than pytest (pytest's real exit was `3`). **Any future "the suite is green" claim must
confirm a summary line was actually printed**, not infer it from an exit code.

**What would characterize it.** Add a `shutdown_pools()` autouse fixture to
`tests/test_api.py` mirroring `tests/test_worker_pool.py`'s, then run the full suite
repeatedly and see whether the abort stops. That is a cheap, low-risk experiment — but it
is an *experiment*, and it was deliberately not applied as a "fix" here.

> **The intermittency is precisely why.** The same full-suite command that aborted later
> passed 1,661/1,661 with no change at all. Had the fixture been added first, that green run
> would have looked like proof it worked — and the register would now carry a "FIXED" entry
> resting on a coincidence. Any candidate fix for this needs *repeated* clean full runs
> against a known-bad baseline, not one. Also worth checking whether XLA's on-disk compilation cache is shared
unsafely across the parent and its spawned workers.

**Related:** [I-15](#i-15) and `test_cross_tier_jobs_correct_and_concurrent` share the
worker-pool/timing premise. Whether they are the same underlying problem is **not**
established.

---

### I-28 — The `var_es` module demo crashes on a date that moved {#i-28}

**Severity:** Low · **Status:** ❌ OPEN — **root-caused, one-line fix, not applied here**
**Found:** 2026-09-17, while verifying that every command in
[the User Guide](getting-started/user-guide.md#running-the-demos) actually runs.

`python -m engine.risk.var_es` — a documented command — aborts before printing anything:

```
ValueError: Swap cashflow times must be a subset of the simulation's rates.maturities
pillars; got cashflow times [0.5095890410958904, 1.010958904109589, 1.5095890410958903,
2.0136986301369864] against maturities [0.010958904109589041, 0.5150684931506849,
1.010958904109589, 1.515068493150685, 2.0136986301369864]
```

**The cause is one missing keyword argument.** The demo block at
[`engine/risk/var_es.py:321`](../engine/risk/var_es.py) builds its `SwapConfig` without an
`evaluation_date`, so the field falls back to its default —
`ORE.Settings.instance().evaluationDate`, i.e. *today*. It then prices that swap against
`SWAP_DEMO_MATURITIES`, which is pinned to `EVAL_DATE = ORE.Date(30, 7, 2026)` in
`engine/simulation/demo_scenarios.py`. Once the wall clock left 2026-07-30 the two stopped
agreeing, and the maturity-pillar-alignment check in
[`engine/instruments/swap.py:165`](../engine/instruments/swap.py) correctly refused the
mismatch. Adding `evaluation_date=EVAL_DATE` to that config — which the other module demos
already pass, e.g. `engine/instruments/swap.py:300` — makes it run; that was confirmed
directly rather than assumed.

**Why it is filed rather than fixed here.** This register entry came out of a documentation
pass, and the fix is a code change. It is recorded so the documented command and the
register agree about reality in the meantime.

**Two things worth drawing out of it.**

- **The failure is the guardrail working.** This is the maturity-pillar-alignment
  constraint the [User Guide](getting-started/user-guide.md#pricing-a-swap) and
  [Instruments: swaps](instruments/swaps.md#a-known-limitation-maturity-pillar-alignment)
  both warn about, doing exactly what it exists to do. A loud `ValueError` naming both lists
  is the good outcome; silently discounting a cashflow against the nearest pillar is the bad
  one.
- **It is a time bomb by construction, and only this demo carries it.** A default that reads
  the wall clock, combined with a constant pinned to a fixed date, is a test that passes
  until a date passes. The rest of the suite is immune because `tests/conftest.py` and
  `demo_scenarios.py` thread `EVAL_DATE` explicitly — which is why 1,777 tests stay green
  while a documented demo does not. The lesson is the one the guide already gives for
  user-written configs: pass `evaluation_date` explicitly rather than inheriting ORE's
  global.

**Related:** the same alignment rule is discussed at
[Instruments: swaps](instruments/swaps.md#a-known-limitation-maturity-pillar-alignment).

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

### I-23 — The `accrualBasis` strictness rule is an assumption, not a confirmed contract {#i-23}

**Severity:** Medium · **Status:** ⚠️ ASSUMPTION — unconfirmed · **Raised:** 2026-09-16 with W1.6.1

**This entry is a different kind from the others.** Everything above records something the
code *does wrong*. This records something the code does **deliberately, on an unverified
reading of an unanswered question** — which is worth registering precisely because it will
otherwise look settled. Nothing here is known to be broken; what is unknown is whether the
interpretation matches TraderX's intent.

**The question, asked and never answered.** [Response v4](planning/eod-contract-response-v4.md)
§1.3 asked: when a real settlement calendar arrives, does `traderx.accrual-basis.v1` **gain
new `dateBasis` / `settlementAdjustment` values**, or does it become `accrual-basis.v2`? That
went out on 2026-09-16 and TraderX's v5 reply did not address it. It was asked again in
[response v6](planning/eod-contract-response-v6.md) §2.3.

**What W1.6.1 implemented in the absence of an answer.**
[`engine/integration/terms.py`](../engine/integration/terms.py) pins each enum to an exact
accepted set and **refuses everything else**:

| Field | Accepted |
|---|---|
| `dateBasis` | `SESSION_DATE` |
| `settlementAdjustment` | `NONE` |
| `rounding` | `HALF_EVEN` |
| `accrualBasis.schema` | `traderx.accrual-basis.v1` |

**Why this direction was chosen.** The alternative — parsing an unrecognized basis
optimistically — means reconciling accrued interest against a date convention this engine
does not actually understand, with no error anywhere. That is the silent-approximation
failure the whole boundary exists to prevent (working rule 1), and it is unrecoverable after
the fact. Refusing is loud, and widening a tuple later is a one-line change.

**What could be wrong about it, stated plainly.** If TraderX intends to add values **in
place** — keeping `accrual-basis.v1` and extending its enums — then this engine will
**refuse bundles they consider valid**, on the day they first export a real calendar. The
refusal would be correct by this engine's stated rule and wrong by their intent. That is a
false rejection, not a wrong number, so it fails safe; but it is an operational break that
will look like a defect to whoever is on call, and it will arrive without warning.

**What a consumer should know now.** A `TermsJoinError` naming
`accrualBasis.dateBasis` / `.settlementAdjustment` / `.rounding` is **not necessarily a bad
bundle**. It may be this engine's allowlist being narrower than the exporter's current
vocabulary. Check the accepted set above before treating it as an export defect.

**What closing it requires — one of two answers from TraderX:**

1. *"New values force a new schema version."* → The implementation is already correct; this
   entry closes with no code change.
2. *"Values are added in place."* → Pin the expanded set in
   `SUPPORTED_DATE_BASES` / `SUPPORTED_SETTLEMENT_ADJUSTMENTS` / `SUPPORTED_ACCRUAL_ROUNDING`,
   and add a test per newly-accepted value. Still an allowlist — the change is *which* values
   are on it, never *whether* there is one.

**Related.** The same strictness refuses a **v1-labelled artifact carrying an
`accrualBasis`** at all, on the grounds that the document has contradicted its own version
marker. That reading is firmer — a self-contradictory document cannot be parsed on the
assumptions its version marker implies — but it shares this entry's root cause: the version
semantics were specified by one side and never jointly confirmed.

**No regression test can close this**, which is why it is an assumption rather than a bug.
The behaviour is fully tested
([`tests/test_integration_terms_v2.py::TestUnrecognizedValuesAreRefused`](../tests/test_integration_terms_v2.py),
6 cases, verified to fail against a lenient parser). What the tests cannot establish is
whether the rule they pin is the *agreed* one.

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
