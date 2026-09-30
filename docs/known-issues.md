# Known Issues and Limitations Register

**Purpose.** One authoritative list of every known defect and scope gap in this engine, what
each one does to a number a user would see, and what closing it actually requires. Written
during the TraderX EOD integration review (see
[EOD Contract Proposal](planning/traderX_integration/eod-contract-proposal.md)), which is where several of these
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
| *Difference from ORE* | A tag on an OPEN entry, not a status: the engine computes something ORE computes differently, by design rather than by slip, and the difference moves a number enough that it should be resolved. Minor, deliberate differences (derivatives instead of finite-difference Greeks, bisection instead of an optimizer) are noted in the code, not here. |
| **PARTIAL** | Closed on one path and open on another. Used only where the split is real and nameable — not as a softer word for OPEN. [I-08](#i-08): the EOD path is durable, the portfolio path is not. [I-24](#i-24) and [I-42](#i-42) to [I-47](#i-47): closed on the market path (the default model), open on the Hull-White model, which stays a supported, non-default option (decision A-1). |

**On ASSUMPTION, added 2026-09-16 with [I-23](#i-23).** The other three statuses all describe
something the code gets wrong. This one describes a decision made in the absence of an answer
— where the risk is not a wrong number but a **wrong premise**, invisible precisely because
the code behaves exactly as designed. An entry here is a standing question to an external
party, not a bug queue item, and it closes when the question is answered rather than when
code changes.

## Verification status

Last full verification (2026-09-29, after the [ORE alignment](planning/ore-alignment-plan.md)):
**2,320 passed, 0 failed** on Windows (46m00s) and **2,319 passed, 1 skipped, 0 failed** in a
Linux `python:3.11` container on 4 cores (50m09s; the skip is `reference/traderX`, absent in
the container, as on 2026-09-24). Both are the complete suite
(`.venv/Scripts/python.exe -m pytest tests/`), 2,320 collected, summary line printed, exit
code 0, zero `FAILED`/`ERROR` lines, on the same final code, run at the same time. The count
reconciles against the 2,159 below: the collected count rose by 161, all from tests added or
parametrized by the alignment (new files `test_cam.py`, `test_curves.py`, `test_valuation.py`,
`test_ore_lgm_calibration.py`, `test_sensitivities.py`, `test_portfolio_market_path.py`,
`test_api_market_path.py`, `test_shared_portfolio.py`, `test_import_layering.py`, plus new
cases in existing files).

How it got there. The first full runs on the aligned code (Windows 2,318 passed, 2 failed;
Linux 2,317 passed, 1 skipped, 2 failed) failed the same two tests,
`test_profiling_and_jit.py::TestPreparedBermudanPytree`. They pinned the prepared Bermudan's
old `zero_rates` field, which the market path replaced with a `curve` pytree; the tests now
assert the new field. A re-run of the touched files then failed
`TestCompileCounts::test_bermudan_theta_compiles_few_programs` (27 programs, limit 15). This was
a real regression, hidden in the full runs because earlier tests had already compiled the same
eager ops. With the calendar-day Theta ([I-38](#i-38)) a fixing can fall inside the Theta
period, and its forecast was looked up eagerly. Alone the test fails on the pre-fix code and
passes on the baseline; the lookups in `engine.risk.greeks` are now one jitted program, and
the Greeks, Theta and profiling files pass (176 tests). The Phase 2 fixes were shown red first
against a baseline worktree at `fc7cd3e`, whose own full run was 2,158 passed, 1 skipped,
0 failed (22m54s).

The run before it (2026-09-25, after [engine audit M-4/M-5](planning/engine-audit.md#m-4),
trade dates and theta, and the [I-35](#i-35) fix): **2,159 passed, 0 failed** (23m23s) — the
complete suite (`.venv/Scripts/python.exe -m pytest tests/`), 2,159 collected, summary line
printed, exit code 0, zero `FAILED`/`ERROR` lines. The count reconciles against the 2,087
below: +72 from the new `tests/test_trade_dates.py`, and no other file's count changed. The run
before it, same code except one test, was 2,158 passed, 1 failed: the failure was
`test_american_swaption.py::TestOptionTimes`, which pinned the pre-I-35 behaviour and was
rewritten to ORE's rule. Slowest test 42.5s, no outlier. Windows only; not yet run on Linux.

The run before it (2026-09-24, after pinning dependencies and adding CI,
[audit Q-2](planning/engine-audit.md#q-2), and the [I-33](#i-33) fix): **2,087 passed,
0 failed** (21m50s) — the complete suite (`.venv/Scripts/python.exe -m pytest tests/
--durations=25`), 2,087 collected, summary line printed, exit code 0, zero
`FAILED`/`ERROR` lines. The count reconciles against the 1,966 below, per file, from
`--collect-only` in worktrees at each commit:

| Δ | Source |
|---:|---|
| +11 | `784b51a` (audit fixes), never recorded here: `test_market_model.py` +4, `test_calibration_lgm.py` +2, `test_greeks.py` +2, `test_portfolio_entrypoint.py` +2, `test_worker_pool.py` +1 |
| +91 | `91e3832` (market risk): `test_market_risk.py` +53 (new), `test_exposure.py` +21 (new), `test_market_risk_ore_parity.py` +9 (new), `test_portfolio.py` +7, `test_portfolio_entrypoint.py` +1 |
| +18 | `tests/test_environment.py` (new): installed numerics match `constraints.txt` |
| +1 | `tests/test_worker_pool.py` — [I-33](#i-33) |

1,966 + 11 + 91 + 18 + 1 = 2,087. The same code before these changes ran 2,068 passed,
0 failed (27m21s). Slowest test 44.4s, no outlier.

**Two tiers since this run.** `-m "not slow"` is the fast tier (1,986 tests, about 9.4
minutes of the 21m50s here); 101 tests are marked `slow`. **A fast-tier count is not a
full verification** and never goes in this section. The fast tier also passed on Linux
(`python:3.11` container, 4 cores, pinned environment): 1,985 passed, 1 skipped
(`reference/traderX` absent), 0 failed in 9m32s, and the slow tier 101 passed in 12m58s.
That is the first recorded Linux run, and it found [I-33](#i-33).

The run before it (2026-09-24, after the I-11/I-28 fixes): **1,966 passed, 0 failed**
(25m12s) — the complete suite (`.venv/Scripts/python.exe -m pytest tests/ --durations=25`),
nothing excluded, summary line printed, exit code 0, zero `FAILED`/`ERROR` lines. The
count reconciles against the 1,960 below: +6 from the new
`tests/test_risk_measure_label.py` (5 for [I-11](#i-11), 1 for [I-28](#i-28)), and no other
file changed. Slowest test 58.3s
(`test_all_four_instrument_types_each_represented_multiple_times`, 41–42s in the two runs
before). No single test explains the extra six minutes of wall clock; record it and watch
the next run rather than read anything into one.

The run before it (2026-09-23, after the I-30 fix): **1,960 passed, 0 failed**
(19m29s), under the same conditions. That count reconciles against the 1,897 below, per file
against `HEAD` (`--collect-only` in a separate worktree):

| Δ | Source |
|---:|---|
| +60 | `tests/test_european_swaption.py` 131 → 191 — the conditional-pricing grid ([I-30](#i-30)) |
| +3 | `tests/test_ore_coverage_hardening.py` 36 → 39 — every grid point catches every mutation |

1,897 + 60 + 3 = 1,960. Slowest test 42.1s; no outlier.

The run before it (2026-09-23, after the I-06/I-31 fixes): **1,897 passed, 0 failed**
(19m12s), under the same conditions. That count reconciles against the previous confirmed
collection of 1,879, per file against the previous commit:

| Δ | Source |
|---:|---|
| +23 | `tests/test_ore_lgm_parity.py` (new) — parity with ORE's own LGM engine |
| +2 | `tests/test_bermudan_swaption.py` 61 → 63 — date validation |
| +2 | `tests/test_american_swaption.py` 22 → 24 — rewritten around ORE's American rules |
| −9 | `tests/test_ore_bermudan_oracle.py` 60 → 51 — 12 snap tests removed with the snap, 3 added |

1,879 + 23 + 2 + 2 − 9 = 1,897. The 1,879 itself was never run in full (see history); this run
covers it.

Earlier figures, for history: 1,863/0 on 2026-09-18 (4h08m); 1,777/0 after W0.8 (16m55s),
1,718/0 at `1e078f3`, 1,618/0 after W1.6, 1,452/1 after the v5 fixes, 1,384/2 after W1.4,
1,324/2 after W1.3, 1,217/2 after W1.2.

**Four standing rules for any figure recorded here**, each one written after it was violated:

1. **Run `.venv/Scripts/python.exe -m pytest`, never the bare `python`.** The system
   interpreter lacks `pydantic` and `jsonschema`, which makes `tests/test_api.py` and
   `tests/test_integration_schema.py` silently uncollectable — 77 tests at today's counts.
   A count from the wrong interpreter is not comparable to anything here. See [I-25](#i-25).
2. **Take counts from `pytest --collect-only`** — never from a remembered summary line, and
   never by grepping `def test_`, which undercounts every parameterized case.
3. **Confirm a summary line was actually printed.** [I-27](#i-27) can kill the process with
   no summary at all, and an exit code read from the wrong place has been mistaken for a
   pass. Leave the working tree alone for the duration of a run — an unrelated `git stash`
   mid-run also terminates it without a summary.
4. **A count that does not reconcile is a signal to find out why, not a new number to
   record.** Both hazards above have fired and both were caught by arithmetic rather than by
   the runner.

**The 4h08m of 2026-09-18 did not recur**: this run took 19m12s, in line with the 16m55s
before it, on a larger suite and with `--durations=25` recorded. The slowest test was 41s
(`test_all_four_instrument_types_each_represented_multiple_times`), and the 25 slowest sum
to ~7 minutes, so no single test explains four hours. **The user reports the machine
crashed during that run**, which would account for the wall clock: it was not a property of
the suite. (That run still printed a full summary and exit code, so whatever interrupted it
the pytest process survived; the pass count stands.) Keep recording durations.

**The long-running flake.** `test_cross_tier_jobs_correct_and_concurrent` has now passed
six full runs and failed one, on identical code (passed in both 2026-09-23 runs). It passes in isolation (12.28s) and failed
every full run from W1.2 through W1.5. The 2026-09-18 pass came during the *slowest* run on
record (4h08m), which is mild evidence *against* the load-dependence hypothesis, since a
wall-clock overlap assertion should be most likely to fail under exactly those conditions.
**A single green run of this test means nothing in either direction.** Shares
[I-15](#i-15)'s premise; still not enough to reclassify.

**How the defects here were actually found — none by running this suite where it was written.** A green suite is
evidence about the *tests*, not proof about the *code* (working rule 9), which is the premise
this register exists to embody. Of the defects found during this integration:

| Route | Issues |
|---|---|
| TraderX reading source or reproducing numbers | [I-13](#i-13), [I-19](#i-19), [I-20](#i-20) |
| Reviewing or reasoning about my own code | [I-25](#i-25), [I-26](#i-26), the W1.6 `submissionId` bug |
| Building a new external oracle | [I-29](#i-29) (fixed 2026-09-18) |
| Mutation-testing the suite's own tolerances | [I-30](#i-30) (fixed 2026-09-23) |
| **Measuring a claim the docs made but no test asserted** | **[I-06](#i-06)'s error direction** |
| **Pricing against the engine ORE actually uses** | **[I-06](#i-06) rescoped to American, [I-31](#i-31)** |
| Running the test suite on the platform it was written on | **none** |
| **Running it on a second platform** (the Linux CI container, Q-2) | **[I-33](#i-33)** |

**The newest route is the cheapest, and it found the worst result.** [I-06](#i-06) had been
described as a "conservative (value-understating)" approximation here, in a module
docstring and in the instruments doc. Nothing measured it — `TestMidCouponKnownLimitation`
asserted only that the price was finite, non-negative and within one order of magnitude.
Actually pricing a payer and a receiver either side of an accrual boundary took one script
and showed the approximation **overstates a payer up to 7.4x**. A documented property that
no test asserts is a hypothesis wearing the costume of a fact; this register now carries
its own instance of the thing it was created to catch.

**And the 7.4x was then shown to be measured against the wrong baseline** (2026-09-23).
It compared a mid-period Bermudan with the engine's *own* aligned price. ORE's source says
a mid-period Bermudan exercises into the next whole period, so that difference is ORE's
behaviour, not an error. Pricing against ORE's own LGM engine, now reachable in-process
through `engine/validation/ore_lgm_oracle.py`, moved the real deviation to **American** exercise (up
to 6.0x) and turned up [I-31](#i-31), which the engine's self-consistent tests could never
see. A self-comparison can show a direction; only an external oracle shows the right
reference.

The last two routes are new as of 2026-09-18. The Bermudan engine was cross-checked end to
end against real `ORE.TreeSwaptionEngine` / `ORE.FdHullWhiteSwaptionEngine` objects for the
first time (see [ore-parity.md §7a](reference/ore-parity.md#7a-an-external-multi-exercise-oracle-does-exist-correction-2026-09-18),
which corrects this project's standing claim that no such oracle existed) and **no pricing
defect was found** — every apparent discrepancy resolved to the new test being wrong or to
the already-documented HW/LGM parametrization difference. What it surfaced instead was one
silent input sensitivity ([I-29](#i-29)) and one place where the suite's own `rtol=1e-4`
cannot see a deleted term of the core bond-price formula ([I-30](#i-30)).

---

## Priority order for fixing

Ranked by **criticality** (how wrong is a number a user would act on?) against **difficulty**
(can someone start today, or is it blocked on an input that does not exist?). FIXED entries
are excluded. The tiers are the unit of decision here — within a tier, order is a judgement
call and the rank column should not be read as precise.

**The one-line read:** Tier 1 is blocked on someone else. Of the work that can start, the
validation gaps in Tier 3 ([I-50](#i-50), [I-51](#i-51)) come first. The market path's
components each equal ORE, but nothing yet compares its assembled cube, exposure and Greeks
with an ORE run. ([I-36](#i-36) to [I-41](#i-41), [I-48](#i-48) and [I-52](#i-52) were fixed on
2026-09-29 by the [ORE alignment](planning/ore-alignment-plan.md); [I-24](#i-24) and
[I-42](#i-42) to [I-47](#i-47) are closed on its market path, the default, and remain on the
Hull-White model.)

**Owner decisions of 2026-09-30** ([compliance/decisions.md](../compliance/decisions.md)) shape this order. The engine is to be configurable,
with ORE's defaults, and the Hull-White model stays as an option, so its defects are to be
fixed rather than retired. [I-49](#i-49) and [I-54](#i-54) are to be closed. Precision stays
freely adjustable, with warnings where a combination is unproven ([I-55](#i-55)).

### Tier 1 — Highest criticality, blocked on external input

Neither can be closed by engineering effort alone. **Chase the dependency, not the code.**

| # | Issue | Severity | Difficulty | What actually unblocks it |
|---:|---|---|---|---|
| 1 | [I-04](#i-04) — aged swaps need past fixings | High | **Blocked** | Historical `pastFixings` from TraderX, which they do not export. The market path prices a seasoned swap correctly when given them and refuses one without them; the Hull-White model is still flagged. |
| 2 | [I-05](#i-05) — no faithful USD-SOFR/ACT-360 construction | High | **Blocked** + moderate | The D03/D04 convention agreement. *Guessing the conventions reproduces exactly this issue's failure mode.* |

### Tier 2 — Real correctness exposure, unblocked

| # | Issue | Severity | Difficulty | Why it ranks here |
|---:|---|---|---|---|
| 3 | [I-42](#i-42)–[I-47](#i-47) — the Hull-White model's defects | High | Moderate–hard | Closed on the market path (the default). The Hull-White model stays a supported option (decision A-1), so they are to be fixed within it: plan 9.3. Until then anyone choosing that model gets these numbers. |
| 4 | [I-49](#i-49) — per-path recalibration differs from ORE in two known details | Medium | Moderate, needs [I-50](#i-50)'s oracle | Every Bermudan/American value past t=0 on the market path; unmeasured. Decided: close it (plan 10.1). |
| 5 | [I-54](#i-54) — no swaption smile | Medium | Moderate–hard | An option away from the money reads the ATM volatility. Decided: close it (plan 10.2). |
| 6 | [I-32](#i-32) — Bermudan engine only at `ShiftHorizon = 0`, Grid solver | Medium | Moderate | Decided: configurable, with ORE's default 0.5 once parity there is proven (plan 9.6). Not urgent. |
| 7 | [I-10](#i-10) — no trade identity on the configs | Medium | Low-moderate | Optional `trade_ids` are now echoed; the configs themselves carry no identity. |

### Tier 3 — Blocks confidence in the numbers or the suite

| # | Issue | Severity | Difficulty | Note |
|---:|---|---|---|---|
| 8 | [I-50](#i-50) — no L3/L4 parity against an ORE simulation | Medium | **Hard** | Every exposure figure on the market path rests on component parity only. Needs the oracle generalized to an OREApp XVA run. |
| 9 | [I-51](#i-51) — Greeks not checked against ORE's sensitivity analytic | Medium | Moderate | Same oracle work, sensitivity analytic instead of XVA. |
| 10 | [I-55](#i-55) — unproven precision combinations are not flagged; precision is switched per process | Medium | Moderate–hard | Any combination may be run (decision D-9), so a result must say when its precision is unproven for that figure. The switching mechanism is to be replaced, not removed (decision A-9; plan 9.4, 9.5). |
| 11 | [I-27](#i-27) — full-suite runs hard-abort inside XLA | Medium | **Hard to diagnose** | Intermittent; no abort in the runs recorded under Verification status. |
| 12 | [I-34](#i-34) — the ORE oracle's curve differs before the first pillar | Low | Moderate | Validation tooling only; parity tests use a flat first segment. |

### Tier 4 — Scope gaps, correctly refused rather than approximated

None of these produces a wrong number. **Priority here is driven by demand, not by risk.**

| # | Issue | Severity | Difficulty | Blocked on |
|---:|---|---|---|---|
| 13 | [I-18](#i-18) — no equity spot or FX source for EOD equity positions | Medium | **Blocked**, then trivial | Market data. *Do not close it with `closingMark`.* |
| 14 | [I-16](#i-16) — `rateSensitivity` parallel-only | Medium | **Blocked** | A curve with genuine pillar structure (W2). |
| 15 | [I-07](#i-07) — no corporate bond / equity / listed-option pricer | Medium | Moderate–hard | Corporate bonds need a credit model. FX and equity trades on the market path are decided, not urgent (plan 10.4). |
| 16 | [I-24](#i-24) — bonds refused in the Hull-White model's cube | Medium | Moderate | Priced on every path on the market path. On the Hull-White model, part of plan 9.3. |
| 17 | [I-08](#i-08) — portfolio path's `_JOBS` dict still in-process | Medium | Moderate | Port `publication.py`'s design to the portfolio path. |
| 18 | [I-09](#i-09) — whole scenario cube serialized into JSON | Medium | Moderate | A chunked artifact plus a reference. |
| 19 | [I-56](#i-56) — the API cannot reach every setting; two routes named like versions | Medium | Moderate | Decided: one route and one request reaching every setting (decision A-2, plan 9.2). Unreachable settings are absent, not wrong. |

### Tier 5 — Performance and cosmetic

Every number is correct. Plan Phase 7 (performance) has not started: its rule is that every
change leaves the parity tests bit-identical, and I-49 to I-51 may still move numbers.

| # | Issue | Severity | Difficulty | Note |
|---:|---|---|---|---|
| 20 | [I-53](#i-53) — the market path is slow (full suite 23 → 46 min) | Medium | Moderate | Profile first; recalibration per path and date and the bump loops are the suspects. |
| 21 | [I-21](#i-21) — Greeks recompile 23 XLA programs per call | Medium | **Moderate, fully designed** | Key the memo on `static_key(prepared)`, never the config, never `id()`. The market path's bump Greeks revalue in Python loops and are slower still. |
| 22 | [I-22](#i-22) — calibration recompiles 8 programs per call | Low | Moderate | A different mechanism from I-21. |
| 23 | [I-12](#i-12) — `/version` reports dispatcher, not worker device | Low | Low | Invisible on a single-CPU box. |

### Tier 6 — Awaiting an answer, not an engineer

| # | Issue | Severity | Difficulty | Note |
|---:|---|---|---|---|
| 24 | [I-23](#i-23) — `accrualBasis` strictness is an assumption | Medium | **Not a code task** | Closes when TraderX answers. |

### What the ordering deliberately does not do

- **It does not rank by severity alone.** [I-04](#i-04) is High and cannot be started.
- **It does not treat "refused" as "broken".** Tier 4 entries return an identified refusal
  rather than a number.
- **It does not treat component parity as end-to-end parity.** Tier 3 exists because every
  piece of the market path equalling ORE does not prove the assembly does.

---

## Summary

Sorted by ID. For **what to fix first**, see
[Priority order for fixing](#priority-order-for-fixing) above; the rank column below points
back into it.

| ID | Issue | Severity | Status | Rank |
|---|---|---|---|---|
| [I-01](#i-01) | Swap Delta/Gamma/Theta silently absent from portfolio results | High | ✅ FIXED | — |
| [I-02](#i-02) | Bermudan Vega never computed | Medium | ✅ FIXED | — |
| [I-03](#i-03) | No per-instrument NPV; totals unattributable | Medium | ✅ FIXED | — |
| [I-04](#i-04) | Aged swaps mispriced at every step past first accrual | **High** | ⚠️ FLAGGED — correct given fixings on the market path; the fixings are not exported | **1** |
| [I-05](#i-05) | No faithful USD-SOFR/ACT360 swap construction | **High** | ❌ OPEN — refusal path landed (W0.4) | **2** |
| [I-06](#i-06) | American exercise ignored ORE's broken-period proration | **High** | ✅ FIXED | — |
| [I-07](#i-07) | No bond, equity, or listed-option pricer | Medium | ❌ OPEN — Treasury pricers landed | 15 |
| [I-08](#i-08) | Job store is in-process; lost on restart | Medium | ⚠️ PARTIAL — EOD path durable (W0.8) | 17 |
| [I-09](#i-09) | Whole scenario cube serialized into JSON responses | Medium | ❌ OPEN | 18 |
| [I-10](#i-10) | No trade identity; results keyed by array position | Medium | ❌ OPEN — closed at the EOD boundary; optional `trade_ids` echoed | 7 |
| [I-11](#i-11) | Risk measure unlabelled; no Monte Carlo error reported | Medium | ✅ FIXED | — |
| [I-12](#i-12) | `/version` reports dispatcher backend, not worker device | Low | ❌ OPEN | 23 |
| [I-13](#i-13) | Negative curve index silently prices against the wrong curve | **High** | ✅ FIXED | — |
| [I-14](#i-14) | `generate_paths(precision=32)` leaks `jax_enable_x64=False` | **High** | ✅ FIXED | — |
| [I-15](#i-15) | Worker-pool concurrency test could not observe concurrency | Low | ✅ FIXED | — |
| [I-16](#i-16) | `rateSensitivity` is parallel-only | Medium | ❌ OPEN — blocked on a real curve | 14 |
| [I-17](#i-17) | A malformed note date failed the entire bundle | Medium | ✅ FIXED | — |
| [I-18](#i-18) | No equity spot or FX source; equity positions are refused | Medium | ❌ OPEN — refusal path landed (W1.4) | 13 |
| [I-19](#i-19) | Accrual tolerance rounded the bound it exists to enforce | Medium | ✅ FIXED | — |
| [I-20](#i-20) | Impossible calendar dates aborted the whole bundle | **High** | ✅ FIXED | — |
| [I-21](#i-21) | Greeks recompile 23 XLA programs on every call | Medium | ❌ OPEN | 21 |
| [I-22](#i-22) | Calibration recompiles 8 XLA programs per call | Low | ❌ OPEN | 22 |
| [I-23](#i-23) | `accrualBasis` strictness is an **assumption** | Medium | ⚠️ ASSUMPTION | 24 |
| [I-24](#i-24) | Bonds have no scenario NPV on the Hull-White path | Medium | ⚠️ PARTIAL — priced on every path on the market path | 16 |
| [I-25](#i-25) | A **scalar** Greek crashed the HTTP result serializer | Medium | ✅ FIXED | — |
| [I-26](#i-26) | Greeks for a bond maturing **tomorrow** crashed on the theta reprice | Low | ✅ FIXED | — |
| [I-27](#i-27) | Long full-suite runs **hard-abort inside XLA compilation** | Medium | ❌ OPEN — located, not root-caused | 11 |
| [I-28](#i-28) | `python -m engine.risk.var_es`'s own demo crashed | Low | ✅ FIXED | — |
| [I-29](#i-29) | A rounded exercise time silently drops a whole coupon | Medium | ✅ FIXED | — |
| [I-30](#i-30) | The `A(t,T)` variance term was nearly uncovered at `t=0` | Medium | ✅ FIXED | — |
| [I-31](#i-31) | Bermudan/American floating coupons projected over the wrong period | Medium | ✅ FIXED | — |
| [I-32](#i-32) | Parity with ORE holds only for its Grid solver at `ShiftHorizon=0` | Medium | ❌ OPEN — decided: configurable, ORE's default 0.5 | 6 |
| [I-33](#i-33) | On Linux, worker-pool jobs **hung** once the parent had run JAX | High | ✅ FIXED | — |
| [I-34](#i-34) | The ORE oracle's curve differs from the engine's before the first pillar | Low | ❌ OPEN | 12 |
| [I-35](#i-35) | An American already in its window could be exercised on the evaluation date | Medium | ✅ FIXED | — |
| [I-36](#i-36) | A non-ACT/365 floating leg was projected with the wrong forward | Medium | ✅ FIXED | — |
| [I-37](#i-37) | A European swaption silently ignored `floating_spread` | **High** | ✅ FIXED — refused (Hull-White), priced (market path) | — |
| [I-38](#i-38) | Theta rolled a business day; ORE rolls a calendar day | Medium | ✅ FIXED | — |
| [I-39](#i-39) | Bond Theta had no add-back for a coupon paid inside the period | Medium | ✅ FIXED | — |
| [I-40](#i-40) | The note's `rateSensitivity` ignored the declared `fractionDecimals` | Low | ✅ FIXED | — |
| [I-41](#i-41) | A European swaption at zero mean reversion priced at intrinsic value | Low | ✅ FIXED — refused | — |
| [I-42](#i-42) | Simulated curves are not arbitrage-free against the input curve (audit M-1) | **High** | ⚠️ PARTIAL — closed on the market path | 3 |
| [I-43](#i-43) | Options worth zero after expiry instead of becoming the swap (audit M-3) | **High** | ⚠️ PARTIAL — closed on the market path | 3 |
| [I-44](#i-44) | Scenario pricing mixes Hull-White and LGM (audit A-2) | Medium | ⚠️ PARTIAL — closed on the market path | 3 |
| [I-45](#i-45) | Numeraire is a discretely accrued bank account, not ORE's LGM numeraire | Medium | ⚠️ PARTIAL — closed on the market path | 3 |
| [I-46](#i-46) | Europeans priced off the model vol, not the market vol | Medium | ⚠️ PARTIAL — closed on the market path and in market risk | 3 |
| [I-47](#i-47) | The calibration basket is not the one ORE builds for the trade | Medium | ⚠️ PARTIAL — closed on the market path | 3 |
| [I-48](#i-48) | Zero curves extrapolated a flat zero rate; ORE a flat forward | Low | ✅ FIXED | — |
| [I-49](#i-49) | Per-path recalibration differs from ORE's in two known details | Medium | ❌ OPEN · *Difference from ORE* | 4 |
| [I-50](#i-50) | No path-level or distribution-level parity test against an ORE simulation | Medium | ❌ OPEN (validation gap) | 8 |
| [I-51](#i-51) | Reported sensitivities not checked against ORE's sensitivity analytic | Medium | ❌ OPEN (validation gap) | 9 |
| [I-52](#i-52) | Cash settlement was priced as physical | Medium | ✅ FIXED | — |
| [I-53](#i-53) | The market path is slow: full suite about 46 minutes, from about 23 | Medium | ❌ OPEN (performance) | 20 |
| [I-54](#i-54) | No swaption smile: an option away from the money reads the ATM volatility | Medium | ❌ OPEN · *Difference from ORE* | 5 |
| [I-55](#i-55) | Unproven precision combinations are not flagged; precision is switched per process | Medium | ❌ OPEN | 10 |
| [I-56](#i-56) | The API cannot reach every setting, and two routes are named like versions | Medium | ❌ OPEN | 19 |

**Counts:** 56 issues — 27 FIXED, 19 OPEN, 8 PARTIAL, 1 FLAGGED, 1 ASSUMPTION. The 29
unfixed entries are ranked above.

**For financial correctness:** [I-04](#i-04) and [I-05](#i-05) remain blocked on external input.
On the market path the model defects the audit found ([I-42](#i-42) to [I-47](#i-47)) are closed
by ORE's own design; on the Hull-White model, which stays an option, they are still to be fixed. What is not yet shown is that the assembled simulation, exposure and
Greeks equal an ORE run ([I-50](#i-50), [I-51](#i-51)).

---

## FIXED — defect removed, regression test verified against the pre-fix code

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

### I-06 — American exercise ignored ORE's broken-period proration {#i-06}

**Severity:** **High** · **Status:** ✅ FIXED (2026-09-23) · **Found:** first as a Bermudan
"approximation" (2026-09-18, wrongly); correctly scoped 2026-09-23 against ORE's own engine

**What was wrong.** The engine priced every Bermudan/American exercise with one rule: a
coupon belongs to the exercised-into swap only if its accrual has not started. That is
ORE's rule for a **Bermudan** ("bermudan exercise implies that we always exercise into whole
periods", `NumericLgmMultiLegOptionEngineBase::buildCashflowInfo`). It is not ORE's rule for
an **American**: there a coupon belongs until its accrual *end* and is credited
`couponRatio(t) = clamp((accrualEnd − t − lag)/(accrualEnd − accrualStart), 0, 1)`. So every
American option time inside a period, which is nearly all of them since the grid isn't
aligned, dropped a coupon ORE credits pro rata. Measured against ORE's engine: payers
overstated up to **6.0x** (high strike, low vol), receivers understated to **0.09x**.

> **This entry was wrong twice before it was right, and both corrections are the record.**
> Until 2026-09-18 it called the approximation "conservative (value-understating)" — false
> for a payer. On 2026-09-18 it measured a mid-period *Bermudan* against the engine's own
> aligned price, called the resulting 7.4x a mispricing, and prescribed prorating Bermudan
> coupons. That was the wrong **reference**: against ORE, the mid-period Bermudan was
> already right, and the prescribed fix would have moved it away from ORE. The first
> correction was found by measuring a direction nobody had tested; the second only by an
> external oracle. A self-comparison can reveal a direction, but not which baseline is correct.

**Fix.** Coupon membership is now a property of the exercise style, taken from ORE rather
than hard-coded. `ExerciseStyle` (BERMUDAN / AMERICAN) sets each coupon's belongs-until time
in `prepare_bermudan`, and the backward induction replays ORE's own cashflow bookkeeping
(`_GridSchedule`, `_backward_induction_arrays`): cashflows go into a rolled-back
`underlyingNpv` at the latest time they can be estimated, broken coupons are cached and
credited `couponRatio`, and the exercise value is `underlyingNpv + provisionalNpv +
provisionalNpvNonCached`. `AmericanSwaptionConfig` is priced directly: `to_bermudan()` is
gone, and so is the flag that exempted its grid from snapping. Its option times follow
ORE's construction exactly, including a **truncating** step count (`static_cast<Size>`),
which `to_bermudan` used to round.

Replaying the loop, rather than evaluating the exercise value in closed form, is
deliberate. Both converge to the same limit, but the closed form differs from ORE by up to
1e-4 at a 48-point grid (shrinking ~4x per doubling, measured). The replay matches ORE's
number at any grid.

**D16** (settlement convention for a mid-period exercise) is answered by the same source:
ORE's convention, as implemented.

**Verified.** `tests/test_ore_lgm_parity.py` (23 tests) prices through ORE's own
`NumericLgmMultiLegOptionEngine` in-process (`engine/validation/ore_lgm_oracle.py`: `OREApp` → trade XML →
`LGMGridSwaptionEngineBuilder`) and asserts equality to **1e-10**. Measured worst case
**8.7e-12**. Run against the pre-fix engine through an adapter to the old year-fraction API,
**22 of the 23 fail** (every American case among them; the one pass is a zero-vol receiver
whose best exercise involves none of the affected coupons). American-specific mechanics are
pinned in `tests/test_american_swaption.py` (option times, truncation, `couponRatio`, and the
one case where the styles must coincide exactly). The Bermudan direction tests stay, now
documented as ORE's contract rather than an error:
`tests/test_bermudan_swaption.py::TestMidPeriodBermudanExercise`.

**Not closed, deliberately:** ORE's `midCouponExercise=true` Bermudans and notice periods
are not exposed by these configs. The coupon model handles both by construction.

---

### I-11 — Risk measure unlabelled on the direct portfolio path {#i-11}

**Severity:** Medium · **Status:** ✅ FIXED (2026-09-24) · **Found:** during the TraderX EOD
integration review

**What was wrong.** `PortfolioResult.risk` returned keys like `VaR_95` with **no statement of
what measure they are**. A risk-neutral exposure simulation is *not* a calibrated forecast of
tomorrow's loss, and nothing in the result distinguished the two. It was also silent about
convergence, so a sparse-tail estimate could not be told apart from a well-converged one.

**Fixed in two steps.**

- **W0.6 — vocabulary and diagnostics, EOD path only.** `RISK_MEASURE_*` in
  [`engine/risk/var_es.py`](../engine/risk/var_es.py) defines the three-value vocabulary
  (`risk-neutral-pricing` / `historical-forecast` / `deterministic-stress`), and
  `ENGINE_RISK_MEASURE` records what this engine actually produces (`risk-neutral-pricing`).
  `engine.integration.result.RiskResult` carries it, and `capabilities()` advertises it.
  `compute_risk_metrics` returns `ES_<p>_tailCount` (effective sample size) and
  `ES_<p>_standardError` (`s/sqrt(n)`, `ddof=1`, **NaN, never 0.0**, when `n < 2`) beside every
  tail statistic. That left the issue OPEN: the label lived only on `RiskResult`, so a direct
  `price_portfolio` caller still got unlabelled `VaR_95` keys.
- **2026-09-24 — the label reaches `PortfolioResult`.** `PortfolioResult.measure` is set to
  `ENGINE_RISK_MEASURE` whenever `risk` was computed, and to `None` when
  `scenario_risk=False` left `risk` empty. A label with no figures would describe numbers that
  do not exist. This is the same rule the EOD pipeline already applies to `RiskResult.measure`.
  `PortfolioResultSchema.measure` carries it over HTTP (`null` when absent).

**Superseded 2026-09-24 (engine audit R-1).** `PortfolioResult.risk` no longer exists: the
portfolio path's cube statistics are now exposure profiles (`PortfolioResult.exposure`,
[Exposure](risk/exposure.md)), still labelled `risk-neutral-pricing`. Short-horizon VaR/ES
is `engine.market_risk`, labelled `historical-forecast` ([Market Risk](risk/market-risk.md)).
The label rule above is unchanged.

**Verified.** `tests/test_risk_measure_label.py::TestPortfolioResultStatesItsMeasure`
(5 tests): risk-neutral on a scenario run, taken from `ENGINE_RISK_MEASURE` and inside
`RISK_MEASURES`, `None` on a `scenario_risk=False` run, and both cases serialized over HTTP.
**All 5 fail against the pre-fix code.**

---

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

### I-28 — The `var_es` module demo crashed on a date that moved {#i-28}

**Severity:** Low · **Status:** ✅ FIXED (2026-09-24) · **Found:** 2026-09-17, while verifying
that every command in [the User Guide](getting-started/user-guide.md#running-the-demos)
actually runs

**What was wrong.** `python -m engine.risk.var_es`, a documented command, aborted before
printing anything:

```
ValueError: Swap cashflow times must be a subset of the simulation's rates.maturities
pillars; got cashflow times [0.5095890410958904, 1.010958904109589, 1.5095890410958903,
2.0136986301369864] against maturities [0.010958904109589041, 0.5150684931506849,
1.010958904109589, 1.515068493150685, 2.0136986301369864]
```

The demo's `SwapConfig` omitted `evaluation_date`, so the swap took ORE's wall-clock *today*
while `SWAP_DEMO_MATURITIES` stayed pinned to `EVAL_DATE = ORE.Date(30, 7, 2026)`. Once the
clock left 2026-07-30 the two disagreed, and the maturity-pillar-alignment check in
[`engine/instruments/swap.py`](../engine/instruments/swap.py) correctly refused the mismatch.
The failure was the guardrail working: a loud `ValueError` naming both lists, not a cashflow
silently discounted off the nearest pillar.

**Fix.** Pass `evaluation_date=EVAL_DATE`, as the other module demos already do.

**Verified.** `tests/test_risk_measure_label.py::TestVarEsDemoRuns` runs the documented
command in a subprocess and asserts a clean exit and printed VaR output. It fails against the
pre-fix code with the `ValueError` above. Before this there was no test at all, because a
`__main__` block is not reachable by importing the module.

**The lesson still stands.** A default that reads the wall clock, combined with a constant
pinned to a fixed date, is a test that passes until a date passes. Pass `evaluation_date`
explicitly rather than inheriting ORE's global.

---

### I-29 — A rounded exercise time silently drops a whole coupon {#i-29}

> **Superseded by design (2026-09-23).** Exercise is now specified by **date**, as ORE takes
> it; an exercise date equal to an accrual date maps to the bit-identical time, and a year
> fraction is refused with `TypeError`. The snap tolerance, the American exemption flag and
> `exercisable_times` described below are removed, and their 12 tests replaced by 3
> (`TestExerciseDatesAreExact`) plus `test_year_fraction_exercise_rejected`. The defect
> stays FIXED, now by construction, not by repair. The residual this entry describes
> ("every late-landing unaligned time drops a coupon") was ORE's own Bermudan rule all along
> ([I-06](#i-06)).

**Severity:** Medium · **Status:** ✅ FIXED (2026-09-18) · **Found:** 2026-09-18, while
building the external Bermudan oracle (`tests/test_ore_bermudan_oracle.py`)

**What was wrong.** `BermudanSwaptionConfig.exercise_times` are year-fractions supplied by
the caller, and the engine did **not** snap them onto the underlying's actual accrual
schedule. `_hw_swap_value_at_nodes` decides which coupons are still alive at exercise with
`fixed_start_times >= t - 1e-9`. A caller who wrote a *rounded* exercise time — `2.0137`
for a true accrual start of `2.0136986301369864` — landed **1.4e-6 late**, which is ~1400x
that 1e-9 tolerance. The coupon starting on that very date then read as already-elapsed
and was dropped from the exercise value entirely.

Measured: at `sigma -> 1e-6`, where the Bermudan must collapse to its intrinsic value of
**1211.47**, the rounded input instead priced **14336.12** — an ~12x overstatement,
independent of volatility. The engine returns a plausible, finite, confidently-formatted
number with no warning, which is exactly the invisible-from-outside class of problem this
register exists for.

**Fix.** `_snap_exercise_times` in
[`engine/instruments/bermudan_swaption.py`](../engine/instruments/bermudan_swaption.py),
called from `prepare_bermudan` — which already builds the ORE swap and reads
`fixed_start_times` off it, so the schedule is in hand at no extra cost. Any exercise time
within `EXERCISE_SNAP_TOLERANCE` (**1e-4** years, ~53 minutes) of a fixed accrual start is
replaced by that accrual start exactly; anything further away is passed through untouched.
The rounded input now prices **1211.47**, bit-identical to the exact one.

`__post_init__` was rejected as the site: it would catch the mistake earlier but force an
`ORE.MakeVanillaSwap` call on every config construction, including the `hw_sigma=None`
"uncalibrated" configs that are built and passed around before they are ever priceable.

**The original plan said "snap or raise". Raising was wrong, and the tests said so.**
This entry previously proposed refusing any time that is not within tolerance of an accrual
boundary. Implemented that way it **failed 66 tests** across
`tests/test_bermudan_swaption.py`, `test_american_swaption.py`, `test_greeks_bermudan.py`
and `test_calibration_lgm.py` — correctly, because a genuinely mid-period exercise date is
**in scope** for this engine rather than invalid input: `TestSingleExerciseMatchesLgmJamshidian`
prices at 1.0/2.5/4.0 against an independent closed form that applies the same liveness
rule, and `TestMidCouponKnownLimitation` exercises it deliberately. Refusing would have
converted a documented approximation into a hard failure and deleted a working capability.

**That does not mean those prices are right.** Investigating this fix's residual is what
showed [I-06](#i-06)'s approximation is **not** the "conservative understatement" this
register had claimed — it overstates a payer up to 7.4x. So the correct reading is: a
mid-period date must keep *pricing* (many callers legitimately supply one), and what it
prices is separately wrong and now tracked at the top of the actionable list.

So the fix repairs a damaged *spelling* of an accrual date and changes nothing else. That
is a narrower claim than "the alignment contract is now enforced", and it is the true one.

**The tolerance is chosen from measured data, not picked.** It separates two populations
with wide margins on both sides: a 4-, 5- or 6-decimal year fraction is off by at most
1.4e-6 (~70x inside the band), while one calendar day is 2.74e-3 (~27x outside it) and the
mid-period times used in the tests above sit ~1e-2 away. Fixed accrual starts are ≥0.99
years apart, so a snap can never be ambiguous between two boundaries.

> **⚠ Correction (2026-09-23).** Some of the reasoning below turned out to be wrong, and
> [I-06](#i-06) now carries the evidence. For a **Bermudan**, a late-landing exercise time
> dropping the coupon that has already started is **ORE's own rule**: a mid-period date
> exercises into the next whole period. So "prorating the in-progress coupon" is *not* how
> the residual closes. What is actually non-ORE is the input: ORE takes exercise **dates**
> and derives times with the curve's day counter, so a year fraction 1.4e-6 past an accrual
> start cannot arise there at all. The snap tolerance exists only because this engine takes
> year fractions. Read "wrong" below as "differs from what the caller meant", not "differs
> from ORE".

**What is still wrong, stated at full scope.** The residual is **not** a precision
question, and an earlier draft of this entry framed it too narrowly as "3-decimal
rounding". Measured by sweeping the offset around a 2.0136986301369864 accrual start:

| Offset from the accrual start | Price | |
|---|---:|---|
| −1e-2, −1e-3, −3e-4, −1.1e-4 | 1211.47 | correct |
| 0 (exact), ±within 1e-4 | 1211.47 | correct (snapped) |
| **+1.1e-4, +3e-4, +1e-3, +1e-2** | **14336.12** | **coupon dropped, ~12x** |

**The failure is entirely one-sided.** An exercise time landing *before* an accrual start
keeps that coupon alive and prices correctly at any distance. One landing *after* it — by
any amount past the snap band — drops the coupon. So what remains is not "coarse
roundings"; it is **every late-landing unaligned exercise time**, which is precisely the
[I-06](#i-06) mid-coupon case seen from the input side. A 3-decimal rounding is just the
cheapest way to stumble into it.

This is why the fix is scoped to near-misses and why widening the band is not the answer:
the band cannot grow far enough to cover a genuine mid-period date without *moving* one,
and moving it would answer a different question than the caller asked. Closing the residual
means **prorating the in-progress coupon** (I-06), not snapping harder.

Pinned by `test_a_coarsely_rounded_time_is_still_not_repaired`, which asserts the
overstatement survives — so the day I-06 is fixed, that test fails and says so.

**New helper: `exercisable_times(cfg)`.** The valid exercise times are a property of the
ORE-generated schedule, so a caller had no way to ask for them without already having one.
(The oracle test reached for them by preparing a throwaway config with a dummy
`exercise_times=[0.0]`.) It returns the underlying's fixed accrual starts, and is now the
documented way to build a config.

**A second, quieter defect found while fixing this one.**
`_warn_if_not_reset_aligned` in [`engine/portfolio/request.py`](../engine/portfolio/request.py)
warns that a misaligned trade *"will use the documented mid-coupon approximation"*, and it
matched accrual dates by **exact set membership** (`round(t, 9) in reset_dates`). Once
near-misses are snapped, that warning fired on trades the engine now prices **exactly** —
reporting an approximation that no longer happens, on the very input the fix repairs. It
now matches on the same `EXERCISE_SNAP_TOLERANCE`, so the warning and the pricer agree.
Genuinely mid-period trades still warn, which is verified in both directions by
`test_the_portfolio_warning_agrees_with_what_is_priced`. Found by running `demos/demo.py`
and reading its warnings against what the pricer had just been changed to do.

**The American path is exempt, by a flag on the config.**
`AmericanSwaptionConfig.to_bermudan` discretizes a *continuous* exercise window onto a
uniform grid (ORE's own construction), where a grid point landing near an accrual start is
a coincidence of the spacing rather than a damaged date — snapping it would silently *move*
an exercise opportunity. It sets `exercise_times_are_discretized=True` on the config it
builds. The provenance rides on the config rather than a parameter each pricing call passes,
because a `to_bermudan()` result is priced at ~14 call sites across engine, tests and demos,
and a flag every one of them had to remember would eventually be forgotten at one.

**Verified.** `tests/test_ore_bermudan_oracle.py::TestExerciseTimeAlignment` — **12 tests**,
replacing the 2 that pinned the defect, so that file goes **50 → 60** collected
(`pytest --collect-only -q`).
**5 fail against the pre-fix code** — the three rounding cases plus the two snapping-mechanism
tests. The rest assert behavior the fix must *not* change (mid-period pass-through, the
American exemption, the coarse-rounding limit) and so pass either way by design; the
exemption test was separately confirmed to fail when the exemption alone is disabled.

The twelve suites touching Bermudan/American pricing, calibration and the portfolio path
run **327 passed, 0 failed** (7m59s, summary line printed, exit code 0) — including the
four the raising version broke. `demos/demo.py`, which prices a Bermudan and an American
through `price_portfolio`, runs end to end, as do both module demos. **A full suite has not
been re-run**; see the caveat in the header.

---

---

### I-30 — The `A(t,T)` variance term was nearly uncovered at `t=0` {#i-30}

**Severity:** Medium · **Status:** ✅ FIXED (2026-09-23) · **Found:** 2026-09-18, by
mutation-testing the ORE comparisons (`tests/test_ore_coverage_hardening.py`)

**What was wrong.** A gap in the **tests**, not in the code. It was recorded because this
register's premise is that a green suite is evidence about the tests, not proof about the
code.

The variance term of the Hull-White `A(t,T)` carries a factor `(1 - exp(-2at))` that is
**identically zero at `t=0`**. So at `t=0` the term contributes nothing, and deleting it
outright changes an ATM swaption price by ~**7e-6** relative — an order of magnitude
*inside* the `rtol=1e-4` that this suite's ORE swaption comparisons assert. Most of those
comparisons price at `t=0`. Deleting the entire term failed exactly **one** test in
`tests/test_european_swaption.py` (131 tests):
`TestConditionalPricingAndExpiry::test_conditional_pricing_matches_ore_rebuilt_at_later_date`.
That single test carried the whole suite's coverage of a term of the core bond-price
formula.

**The formula itself was never in doubt.** It is independently verified against
QuantLib's C++ in [ore-parity.md](reference/ore-parity.md) §3b, against the algebraic
identity `0.25*(sigma*B(t,T))^2*B(0,2t) == (sigma^2/4a)*(1-exp(-2at))*B(t,T)^2` in
`tests/test_ore_parity.py`, and against live `ORE.HullWhite.discountBond` at `t>0`. No
pricing changed in this fix.

**Fix.** A 60-point conditional-pricing grid against ORE's own `JamshidianSwaptionEngine`,
`tests/test_european_swaption.py::TestConditionalPricingAndExpiry::test_conditional_pricing_matches_ore_across_t_and_r`:
four dates from 0.5Y to 2.25Y (all off the curve pillars, because of the kink pinned in
`test_pillar_times_differ_by_the_interpolation_kink`), short rates from 1% to 6%, payer and
receiver, a 5Y and a 10Y underlying at two strikes, and **flat, upward and inverted
curves**. Before this, every conditional check ran on a flat curve. Tolerance is the suite's
own `rtol=1e-4, atol=1e-2`. Measured worst case is **2.1e-6** relative.

**The reference had to be fixed before it could be trusted.** The existing test rebuilds
ORE's market at `t` as a discount curve sampled at annual pillars, which is adequate on its
flat curve. On a sloped curve, even monthly pillars disagreed with the engine by up to
**7.6e-4**, which is outside the tolerance. The cause was not the engine: at `t=0`, priced directly on the
same curves, the two agree to ~5e-7. Log-linear interpolation between pillars gives the
rebuilt ORE model a stepwise instantaneous forward. With **daily** pillars the gap falls to
2.1e-6. The reference also builds the swap **once, at today's date**, exactly as
`prepare_swaption` does. Rebuilding it at `t` would move schedule dates across weekends and
compare two different swaps. (`_reference_ore_conditional_npv` documents all this.)

**Verified — the analogue of "fails against the pre-fix code" for a coverage gap.** Each
mutation was applied to the real `engine/models/hull_white.py`, and the suite run against it:

| Mutation of `A(t,T)` | `test_european_swaption.py` failures before | after |
|---|---:|---:|
| Delete the variance term | 1 of 131 | **61 of 191** (the whole grid plus the original test) |
| Flip its sign | — | **65 of 191** |

`tests/test_ore_coverage_hardening.py::TestVarianceTermIsActuallyChecked::test_every_conditional_grid_point_catches_the_mutation`
(3 cases) keeps this true. For every grid point and every mutation in that file, it asserts
the price moves by at least **10x** the grid's tolerance, so each point fails on its own,
not only the grid as a whole. The measured minimum is ~30x. The existing
`test_variance_mutation_is_invisible_at_t0` still pins why `t=0` comparisons cannot do this
job.

---

### I-31 — Bermudan/American floating coupons were projected over the wrong period {#i-31}

**Severity:** Medium · **Status:** ✅ FIXED (2026-09-23) · **Found:** 2026-09-23, by the
first head-to-head against ORE's own LGM engine

**What was wrong.** The engine projected each floating coupon's rate over the coupon's
**accrual** period. ORE's LGM engine projects it over the **index's** fixing period
`[valueDate(fixingDate), maturityDate(valueDate)]`, with the index day count, and then
multiplies by the coupon's accrual (`LgmVectorised::fixing`). The periods usually coincide
but not always: the schedule is generated backward from maturity, while an index period
is rolled forward from its own start date (e.g. accrual 2027-12-20 → 2028-06-19, index
2027-12-20 → 2028-06-20). Measured on a fully aligned 5Y Bermudan: a constant −5.853 at zero
vol (grid-independent, so not numerical), 0.9–2.1e-4 relative across strikes, sign
following the trade. Recomputing by hand with index-period forwards gave ORE's number to
every printed digit.

**Fix.** `prepare_bermudan` carries each floating coupon's index fixing period, day count
fraction and fixing time, and `_cashflow_values_at_nodes` projects with ORE's clamps
(`T1 = max(t, d1)`, `T2 = max(T1, d2)`). A fixing dated on the evaluation date is
deterministic, as in ORE (`index->fixing(today)`). A fixing before it is refused: it needs
a historical fixing this engine does not hold (I-04).

**A consequence worth knowing: ORE's two engines disagree here.** QuantLib's
`DiscountingSwapEngine` with default "at par" Ibor coupons projects over the accrual period;
ORE's LGM engine over the index period. So a zero-vol Bermudan no longer equals the
at-par discounting value of its forward swap (1211.47) but the **indexed**-coupon one
(1214.23, what ORE's LGM engine returns). `test_ore_bermudan_oracle.py`'s zero-vol anchor
now builds its swap with `IborCoupon.createIndexedCoupons()`. `engine/instruments/swap.py`
still projects over the accrual period, which is correct for the ORE engine it reproduces
(the discounting one). Each pricer matches the ORE engine it stands in for.

**Verified.** Same parity suite as [I-06](#i-06); every aligned-Bermudan case there failed
before this fix and was the whole of the gap. Two pinned values moved by exactly this:
`tests/test_profiling_and_jit.py` (8521.0223 → 8522.4605, and 5x that).

---

### I-33 — On Linux, worker-pool jobs hung once the parent process had run JAX {#i-33}

**Severity:** High · **Status:** ✅ FIXED (2026-09-24) · **Found:** 2026-09-24, running the
new CI fast tier ([engine audit Q-2](planning/engine-audit.md#q-2)) in a Linux
`python:3.11` container before enabling it

**What was wrong.** `_pool_for` in
[`engine/portfolio/worker_pool.py`](../engine/portfolio/worker_pool.py) built its
`ProcessPoolExecutor` without an `mp_context`, so it used the platform default: spawn on
Windows, **fork on Linux**. The module docstring already said Linux "*should* still use
spawn/forkserver deliberately", because forking a process that has initialized a JAX/XLA
client can hang, but the code never did. On Linux, a job submitted after the parent had run
any JAX work never completed, and the HTTP poll saw `pending` until it gave up:

```
FAILED tests/test_api.py::TestPortfolioPriceHappyPath::test_valid_portfolio_returns_202_then_done
    assert 'pending' == 'done'
3 failed, 4 passed   (tests/test_american_swaption.py, then TestPortfolioPriceHappyPath)
```

The same HTTP test passed on its own, because then the parent had not yet run JAX when it
forked. Linux is the deployment target (Cloud TPU VMs), so every portfolio job over HTTP
after the server's first in-process JAX call would have hung.

**Why the suite never saw it.** Every recorded run was on Windows, which cannot fork.

**Fix.** `mp_context=multiprocessing.get_context("spawn")`, on every platform, matching the
Windows behavior the design already assumed.

**Verified.**
`tests/test_worker_pool.py::TestPoolsSpawnOnEveryPlatform` stubs the executor, asserts the
pool gets a spawn context, and fails against the pre-fix code on any platform. It starts no
process, so it runs in the fast tier. On Linux, the reproduction above went from 3 failed to
7 passed (240 s → 70 s).

**Not shown to be related to [I-27](#i-27).** I-27's abort happens on Windows, where
pools always spawned.

---

### I-35 — An American already in its window could be exercised on the evaluation date {#i-35}

**Severity:** Medium · **Status:** ✅ FIXED (2026-09-25) · **Found:** 2026-09-25, pricing
seasoned Americans against ORE's own engine while implementing
[engine audit M-4](planning/engine-audit.md#m-4) (trade dates)

**What was wrong.** `AmericanSwaptionConfig.option_times` started the window at
`t1 = max(0, t(first_exercise_date))`, reproducing ORE's engine
(`NumericLgmMultiLegOptionEngineBase::calculate()`) but not ORE's trade builder in front of
it. `ExerciseBuilder` (OREData/ored/portfolio/optiondata.cpp) first moves an American's
first date to `max(today + 1, first)` — "keep two alive notice dates always for american
style exercise" — so ORE never exercises an American on the evaluation date. The engine
did, whenever the window was already open: `t1 = 0` became an option time. Measured
against ORE's engine through `engine/validation/ore_lgm_oracle.py`, for a 1e6 payer whose
window opens on the evaluation date: engine 38,703.37, ORE 38,729.27 (−6.7e-4 relative).

**Why the suite never saw it.** It did, and asserted it:
`tests/test_american_swaption.py::TestOptionTimes` pinned "a window opening in the past
starts at time zero" as ORE's rule, reading only the engine and not the trade builder in
front of it. No ORE comparison covered an open window: before M-4 a trade could not age into
one, and every parity case booked the window to open about a year out. That test now asserts
ORE's rule (`..._starts_tomorrow`).

**Fix.** `option_times` uses `max(evaluation_date + 1, first_exercise_date)`, ORE's rule.

**Verified.** `tests/test_trade_dates.py::test_seasoned_bermudan_and_american_equal_ore`,
cases `american-window-opens-on-the-evaluation-date` and `american-inside-window`, now
agree with ORE to ~2e-12. The first fails against the pre-fix rule (−6.7e-4, far outside
its 1e-10 tolerance). Nothing that already agreed with ORE moved:
`tests/test_ore_lgm_parity.py` is unchanged, since its windows open in the future.

---

### I-36 — A non-ACT/365 floating leg was projected with the wrong forward {#i-36}

**Severity:** Medium · **Status:** ✅ FIXED (2026-09-29) · **Found:** 2026-09-28, during the
comment review ([engine audit Q-1](planning/engine-audit.md#q-1))

**What was wrong.** The swap pricers computed a floating coupon as `N * (F + s) * accrual`
with `F = (P(start)/P(end) - 1) / accrual`, the leg's accrual fraction. ORE's par coupon
(`IborCouponPricer::initializeCachedData`, QuantLib `couponpricer.cpp`) forecasts over the
index's fixing period and annualizes by the **index** day count's spanning time, then
multiplies by the leg's accrual. They agree only when leg and index share a day count: a 1mm
5Y payer with `accrual_day_count="ACT/ACT (ICMA)"` gave 34,471.06 against
`DiscountingSwapEngine`'s 34,309.58.

**Fix.** `engine.models.ore_builders.par_coupon_forecast_period(coupon)` returns each coupon's
forecast start, end and spanning time as QuantLib computes them. The swap kernel
(`engine.instruments.swap`, used by `engine.risk.price_functions` too), the Greeks' Theta
forecast and the market path (`engine.valuation.legs`) divide by the spanning time.

**Verified.** `tests/test_trade_dates.py::test_any_leg_day_count_equals_ore`, ACT/ACT (ICMA)
cases `fixed-not-started` and `mid-coupon`, agree with `DiscountingSwapEngine` to 1e-10; both
fail against the baseline worktree at `fc7cd3e` (the code before the ORE alignment) with the new tests copied in, 2026-09-29. On the market path `tests/test_shared_portfolio.py`
(`swap-receiver-seasoned-icma`) agrees to 1e-12.

---
### I-37 — A European swaption silently ignored `floating_spread` {#i-37}

**Severity:** **High** · **Status:** ✅ FIXED (2026-09-29): refused on the Hull-White path,
priced on the market path · **Found:** 2026-09-28, during the comment review

**What was wrong.** The Jamshidian pricer valued the floating leg at par, so a spread was
accepted and had no effect (a 1mm 2Y-into-5Y payer priced 21,520.364 with 0 and with 100bp).
QuantLib's `JamshidianSwaptionEngine` refuses a spread.

**Fix.** `prepare_swaption` refuses a non-zero spread, as QuantLib does. On the market path
(the default) a European is priced with ORE's `BlackMultiLegOptionEngine`, which folds the
spread into the strike (`fairRateFromNpvBps`); `engine.valuation.european`.

**Verified.** `tests/test_european_swaption.py::TestJamshidianRefusals::test_nonzero_floating_spread_is_refused_not_ignored`
fails against the baseline worktree at `fc7cd3e` (the code before the ORE alignment) with the new tests copied in, 2026-09-29; `test_quantlib_refuses_a_nonzero_spread_too` pins the reference.
`tests/test_valuation.py::test_european_today_equals_ores_default_engine[payer-with-spread]`
equals ORE's engine (OREApp) to 1e-10, and the path case equals QuantLib's Bachelier engine.

---
### I-38 — Theta rolled to the next business day; ORE rolls one calendar day {#i-38}

**Severity:** Medium · **Status:** ✅ FIXED (2026-09-29) · **Found:** 2026-09-28, during the
comment review

**What was wrong.** `engine.risk.greeks` moved the date with
`TARGET().advance(t, theta_days, Days)`; ORE's sensitivity analysis uses
`asof + thetaPeriod` (`sensitivityanalysis.cpp`). From a Friday the engine measured three
days of carry: −78.04 against ORE's −25.94 on a new swap.

**Fix.** `engine.risk.greeks.theta_date_of` is `evaluation_date + theta_days`, used by swap,
European and Bermudan/American Theta. The market path's Theta (`engine.risk.sensitivities`)
rolls the same way and rebuilds the market on that date (T-19).

**Verified.** `tests/test_trade_dates.py::test_swap_theta_equals_ore[friday]`,
`test_bermudan_theta_equals_ore[friday]` and `test_theta_rolls_one_calendar_day_as_ore` fail
against the baseline worktree at `fc7cd3e` (the code before the ORE alignment) with the new tests copied in, 2026-09-29. `tests/test_sensitivities.py::test_theta_rolls_one_calendar_day_from_a_friday`
covers the market path.

---
### I-39 — Bond Theta had no add-back for a coupon paid inside the period {#i-39}

**Severity:** Medium · **Status:** ✅ FIXED (2026-09-29) · **Found:** 2026-09-28, during the
comment review

**What was wrong.** `engine.portfolio.request._bond_greeks` computed Theta as
`NPV(t + 1 day) - NPV(t)`; a coupon paid in `(t, t + 1]` dropped out and was not added back,
unlike ORE's Theta. The day before a coupon the bond reported about minus the coupon.

**Fix.** `_bond_greeks` adds back the flows paid in `(t, t + 1]`. On the market path Theta adds
every trade's flows paid in `(asof, thetaDate]` (`engine.risk.sensitivities._period_flows`).

**Verified.** `tests/test_portfolio_bond_wire_through.py::TestBondGreeksReachThePortfolioPath::test_theta_adds_back_a_coupon_paid_the_next_day`
failed against the pre-fix code (Theta −1,991.58 where the one-day reprice plus the coupon is
+8.42). `tests/test_sensitivities.py::test_theta_adds_back_a_bond_coupon_paid_on_the_theta_date`
and `test_theta_adds_back_a_coupon_paid_on_the_theta_date` (a swap) cover the market path.

---
### I-40 — The note's `rateSensitivity` ignored the declared `fractionDecimals` {#i-40}

**Severity:** Low · **Status:** ✅ FIXED (2026-09-29) · **Found:** 2026-09-28, during the
comment review

**What was wrong.** `engine.integration.note.rate_sensitivity` re-priced without the terms'
`fraction_decimals`, so it reconciled accrued at 6 decimals, and `pipeline._note_outcomes`
wrapped NPV and sensitivity in one `try`: a sensitivity refusal refused the NPV too.

**Fix.** `rate_sensitivity` takes and passes `fraction_decimals`; `_note_outcomes` attempts
each outcome separately (`_note_attempt`).

**Verified.** `tests/test_integration_note.py::TestSensitivityUsesTheDeclaredFractionDecimals::test_sensitivity_reconciles_at_the_same_precision_as_the_npv`
fails against the baseline worktree at `fc7cd3e` (the code before the ORE alignment) with the new tests copied in, 2026-09-29.

---
### I-41 — A European swaption at zero mean reversion priced at intrinsic value {#i-41}

**Severity:** Low · **Status:** ✅ FIXED (2026-09-29): refused · **Found:** 2026-09-28, during
the comment review

**What was wrong.** `hull_white.bond_option_sigma` divides by `2a`; at `a = 0` it returned NaN,
every bond option fell back to intrinsic value, and the swaption was priced without
volatility (0.0 against 23,957.83 at `a = 1e-8`). ORE's `HullWhite` refuses `a = 0`.

**Fix.** `_validate_hw_a` (`engine.instruments._validation`) refuses `hw_a <= 0` or NaN in
`prepare_swaption`, as ORE does. On the market path a European does not use Hull-White.

**Verified.** `tests/test_european_swaption.py::TestJamshidianRefusals::test_non_positive_mean_reversion_is_refused`
(`0.0`, `-0.01`, `nan`) fails against the baseline worktree at `fc7cd3e` (the code before the ORE alignment) with the new tests copied in, 2026-09-29; `test_quantlib_hull_white_refuses_zero_mean_reversion`
pins the reference.

---
### I-48 — Zero curves extrapolated a flat zero rate; ORE extrapolates a flat forward {#i-48}

**Severity:** Low · **Status:** ✅ FIXED (2026-09-29) · **Found:** 2026-09-28, during the
comment review

**What was wrong.** `zero_rate` held the zero rate flat past the last pillar (`jnp.interp`);
QuantLib's `InterpolatedZeroCurve::zeroYieldImpl` extrapolates the last pillar's
instantaneous forward (`ContinuousForward`).

**Fix.** The curve primitives moved to `engine.models.curves` (re-exported by
`engine.models.hull_white`) and `zero_rate` extrapolates as QuantLib does. Every pricer on a
`ZeroCurve`, Hull-White path included, uses it.

**Verified.** `tests/test_curves.py` (zero rates and discount factors against `ORE.ZeroCurve`
inside and beyond the pillars, on upward, humped and inverted curves) and
`tests/test_treasury_instrument.py::TestCurveInterpolation::test_beyond_the_last_pillar_extrapolates_as_ore`
fail against the baseline worktree at `fc7cd3e` (the code before the ORE alignment) with the new tests copied in, 2026-09-29. The tests that pinned the flat-zero rule
(`test_greeks.py::TestZeroCurve::test_flat_extrapolation_beyond_pillars`, three classes in
`test_market_model.py`) now assert ORE's rule, and fail against the old code.

---
### I-52 — Cash settlement was priced as physical {#i-52}

**Severity:** Medium · **Status:** ✅ FIXED (2026-09-29) · **Found:** 2026-09-29, building the
shared test portfolio (plan §6.3). Both halves were in code written during the ORE alignment
and never released.

**What was wrong.** Swaption configs gained a `settlement` field for the market path.
(1) The Hull-White Jamshidian pricer ignored it and priced a cash-settled European as the
physical one. QuantLib's `JamshidianSwaptionEngine` refuses ORE's cash method,
`ParYieldCurve`. (2) The market path's European priced cash settlement with the physical
annuity `|fixed BPS|`. ORE's default method for a cash-settled European is `ParYieldCurve`
(`defaultSettlementMethod`, swaption.cpp), and `BlackMultiLegOptionEngine` then discounts the
fixed leg at the forward swap rate, `P(start) * sum N tau_i (1 + F)^(-yf(start, T_i))`. Measured:
1.1% to 2% off.

**Fix.** (1) `prepare_swaption` refuses anything but physical settlement. (2)
`engine.valuation.european` uses the par-yield annuity for cash settlement (`EuropeanTerms.par_yield`), at
t=0, on paths and in `engine.market_risk`. A cash-settled Bermudan/American stays priced as
the physical one at t=0. ORE's LGM engine does the same, approximating `ParYieldCurve` by
`CollateralizedCashPrice` with a warning.

**Verified.** `tests/test_european_swaption.py::TestJamshidianRefusals::test_cash_settlement_is_refused_not_priced_as_physical`
failed against the pre-fix code; `test_quantlib_refuses_par_yield_cash_settlement_too` pins
the reference. `tests/test_valuation.py::test_a_cash_settled_european_uses_the_par_yield_annuity`
equals QuantLib's `BachelierSwaptionEngine` with `ParYieldCurve` to about 2e-14, and asserts that
the physical annuity misses it. The `cash-*` cases of
`test_european_on_every_path_equals_quantlibs_bachelier_engine` cover the paths.

---

## FLAGGED — inaccuracy unchanged, silence removed

> These are **not fixes.** The numbers are as wrong as they were before. What changed is that
> the engine now says so. ([I-30](#i-30), a gap in the *tests* rather than the code, was
> filed here until it was closed on 2026-09-23.)

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

**Scope note.** Swaps booked by tenor (`swap_tenor`) start at spot, and nearly every swap in
the demos and tests is booked that way. So nearly *every* multi-step swap portfolio is
affected. This is not an edge case; it is the default path. (Since
[audit M-4](planning/engine-audit.md#m-4), a swap can also be booked forward-starting or in
the past, with explicit dates.)

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
   [the proposal §2.2](planning/traderX_integration/eod-contract-proposal.md). Without them there is nothing to
   populate a fixed coupon *with*.

**Update 2026-09-25 ([audit M-4](planning/engine-audit.md#m-4)).** Item 2 now has a place to
go. Every trade config takes `fixings` (`{ORE.Date: rate}`), and a coupon fixed before the
evaluation date is priced off it at t=0, matching ORE (`tests/test_trade_dates.py`). Without
the fixing, pricing stops with `MissingFixingError` rather than guessing, as ORE stops. That
closes the *pre-t=0* half at t=0. What remains is the simulated steps: a coupon that fixes
**during** the simulation needs the path's own fixing, not data. That is engine work, scoped
as [audit M-2](planning/engine-audit.md#m-2). The TraderX feed still has no `pastFixings`,
so a seasoned TraderX swap would be refused rather than priced.

**Verified (the warning, not the fix):**
`tests/test_portfolio_gap_fixes.py::TestAgedSwapWarningIsNotSilent` (5 tests) and
`tests/test_api.py::TestGapFixesSurviveTheHttpBoundary`.

**On the market path (2026-09-29).** The kernel half (audit M-2) is closed there. A coupon
that fixed before the as-of date pays its historical fixing from the trade's `fixings`, and a
missing one is refused, as ORE refuses it. Coupons fixing during the simulation take
`FixingManager`'s path fixing, and paid coupons drop out
(`tests/test_valuation.py::test_every_path_and_date_equals_ores_discounting_swap_engine`).
What stays blocked is the data: TraderX does not export past fixings.

---

## OPEN — not addressed
> Ordered by ID. For what to tackle first, see
> [the priority order](#priority-order-for-fixing). [I-08](#i-08) is **PARTIAL** — closed on
> the EOD path, open on the portfolio path — and is filed here because the open half is real
> work.

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
| Schedule source | explicit effective/maturity dates (since [M-4](planning/engine-audit.md#m-4); a tenor is resolved to them at booking) | explicit effective/maturity dates |
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
   [proposal §2.2](planning/traderX_integration/eod-contract-proposal.md#22-usd-sofr-swap--the-w2-blocker-set).
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
[traderx-bond-integration-roadmap.md](planning/traderX_integration/traderx-bond-integration-roadmap.md). Corporate
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
[proposal §6.3](planning/traderX_integration/eod-contract-proposal.md).

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
[plan §W0.8](planning/traderX_integration/traderx-integration-plan.md):

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
summaries. See [proposal §6.4](planning/traderX_integration/eod-contract-proposal.md).

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

**Progress (2026-09-29), not closure.** `PortfolioRequest.trade_ids` (optional, one unique
id per trade) is echoed as `PortfolioResult.trade_ids` on both pricing paths. Over HTTP v2
each trade may carry a `trade_id`, on every trade or on none (`tests/test_portfolio_market_path.py`,
`tests/test_api_market_path.py`). Greeks stay keyed by position, and the ids are the caller's
labels, not the `instrumentId`/`accountId` pair on the configs that closure requires. Status
stays OPEN.

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

**Severity:** Medium · **Status:** ⚠️ PARTIAL — **priced on every path on the market path (2026-09-29); still refused on the Hull-White path**

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

**Closed on the market path (`price_portfolio` on a `Market`, `engine.portfolio.market_path`; HTTP `POST /v2/portfolio/price`).** A bond is its remaining flows as a received fixed leg,
discounted on each path's scenario curve. Paid flows drop out, as for any leg. This is ORE's
`DiscountingRiskyBondEngine` with no credit curve and no security spread (plan V-9;
`engine.valuation.portfolio.bond_legs`). It has a cube column and exposure like any trade.
`tests/test_valuation.py::test_a_bond_on_every_path_discounts_its_remaining_flows`,
`tests/test_portfolio_market_path.py::test_a_bills_expected_exposure_is_its_forward_value`,
and `tests/test_shared_portfolio.py` (t=0 against `DiscountingBondEngine`). The Hull-White path
still refuses a bond with `scenario_risk=True`, for the reasons above.

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

### I-32 — Parity with ORE holds only for its Grid solver at `ShiftHorizon=0` {#i-32}

**Severity:** Medium · **Status:** ❌ OPEN · **Found:** 2026-09-23, by pricing against ORE's
engine under the settings ORE's own example configs use

**What is wrong.** [I-06](#i-06)'s parity (1e-11) is with ORE configured the way this engine
prices: the `Grid` convolution solver, `ShiftHorizon=0`. ORE's own shipped American config
(`Examples/Products/Input/pricingengine.xml`) uses neither: it uses the **FD** solver
(`LgmFdSolver`, Douglas scheme, 64 points, 24 steps/year), and its LGM builder defaults
`ShiftHorizon` to **0.5** of the trade's maturity. Measured with
`engine/validation/ore_lgm_oracle.py` (`shift_horizon=`, `fd_solver=`), 5Y trades, engine at
its default 48-point grid:

| ORE setting | Bermudan | Mid-period Bermudan | American |
|---|---:|---:|---:|
| Grid, `ShiftHorizon=0` (what the engine reproduces) | ~1e-13 | ~1e-13 | ~1e-12 |
| Grid, `ShiftHorizon=0.5` (ORE's builder default) | 1.6e-6 | 2.7e-5 | 2.7e-5 – 1.5e-4 |
| FD, shipped settings (64 pts, 24/yr) | 3.8e-4 | 1.0e-3 | 5.7e-4 – 1.6e-3 |
| FD, fine (400 pts, 200/yr) | 3.2e-6 | 4.2e-4 | 1.1e-4 – 1.0e-3 |

**What each gap is.**

- **The shift horizon** is an exact invariance of the LGM: shifting `H(t)` by a constant
  leaves every price unchanged in exact arithmetic, but it moves the state grid, so the
  discretization error changes. That gap is ORE's discretization, and it is small and
  shrinks with the grid. To produce ORE's numbers under its default configuration the engine
  would need the shift (`H → H + shift`, with the state grid built in the shifted variable).
  Moderate work, all inside `engine.models.lgm` and `_state_grid`.
- **The FD solver** is a different numerical scheme. At fine settings its gap to the Grid
  solver is still up to 1e-3 on Americans and mid-period Bermudans while an aligned Bermudan
  converges to 3e-6. So ORE's two solvers do not agree with each other on broken-period
  exercise, and the engine agrees with the Grid one. Reproducing FD would mean porting
  `LgmFdSolver`, which is substantial. It should only be done if ORE runs here use FD,
  and even then the Grid-vs-FD disagreement inside ORE is worth understanding first.

**Not affected:** the I-06 and I-31 fixes themselves. Those were errors of up to 6x and
2e-4 that no solver setting explains; this entry is the remaining numerical-configuration
distance to ORE.

**What closing it requires.** Decide which ORE configuration is the reference. If ORE is run
with its defaults, implement the shift horizon and add `shift_horizon=0.5` cases to
`tests/test_ore_lgm_parity.py`. Port the FD solver only if the reference uses FD.

**Decided (2026-09-30).** Both are to be configurable, with ORE's defaults: `ShiftHorizon`
defaults to 0.5 (ORE's builder default, `OREData/ored/portfolio/builders/swaption.cpp`), and ORE's
FD solver is added as an option beside Grid ([compliance/decisions.md](../compliance/decisions.md) A-3, D-10; plan 9.6). Until the shift is
implemented and parity at 0.5 proven, the engine accepts `ShiftHorizon = 0` only, and the
difference to ORE's default stands as measured above. Not urgent.

---

### I-34 — The ORE oracle's curve differs from the engine's before the first pillar {#i-34}

**Severity:** Low · **Status:** ❌ OPEN · **Found:** 2026-09-25, pricing seasoned trades
against ORE's engine while implementing [engine audit M-4](planning/engine-audit.md#m-4)

**What is wrong.** This concerns the validation tooling, not the pricer.
`engine/validation/ore_lgm_oracle.py` hands ORE the engine's zero curve as date-quoted
zero rates, linearly interpolated. ORE does not keep them as given: its zero-curve build
(`YieldCurve::buildZeroCurve`, OREData/ored/marketdata/yieldcurve.cpp) re-reads every
pillar's rate off a temporary curve, and QuantLib reads the one at t=0 as
`zeroRate(1e-4)`. So ORE's as-of zero becomes `z0 + slope · 1e-4`, where `slope` is the
curve's slope in its first segment. Only that segment tilts: `ORE.ZeroCurve` built from
`[0.03, 0.032]` at `[asof, asof + 1Y]` reports 0.0300002 at the as-of date.

**Size.** On a curve rising 3% → 3.2% over its first year, a Bermudan whose exercise and
cashflows fall inside that year differs from ORE by up to 2.4e-6 relative (at zero
volatility too, so it is the curve and not the model). Exercise after the first pillar
agrees to ~1e-12. A curve that is flat up to its first non-zero pillar shows no difference
at all.

**Why it went unseen.** `tests/test_ore_lgm_parity.py` uses a sloped first segment, but
every exercise it prices is after the first pillar. Dates inside the first segment only
appeared once trades could age (M-4).

**Current handling.** Parity tests that need 1e-10 use a curve flat to its first non-zero
pillar (`tests/test_trade_dates.py`), and the oracle's docstring states the limit. The
engine's own curve is its input and is not changed to imitate ORE's rebuild.

**What closing it requires.** A way to give ORE the engine's curve without the rebuild —
for example a curve segment type that keeps the quotes as given, if ORE has one, or quoting
the as-of zero so that ORE's rebuild lands on `z0` (a fixed point to solve, then a check
that it holds). Until then, keep oracle checks off sloped first segments.

---







### I-42 — Simulated curves are not arbitrage-free against the input curve {#i-42}

**Severity:** **High** · **Status:** ⚠️ PARTIAL — closed on the market path (2026-09-29); the Hull-White path is unchanged · **Found:** 2026-09-24, [engine audit
M-1](planning/engine-audit.md#m-1) (registered here 2026-09-28)

**What is wrong.** The simulation evolves the short rate toward a constant `theta` from a
configured `initial_rates`, while its discount factors use the Hull-White `A(t,T)` fitted
to the input curve, which assumes the curve-fitted drift θ(t). ORE simulates the LGM state
with the curve-fitted model. The two halves agree only on a flat curve with `theta` and
`initial_rates` at its level.

**Size.** From the audit: martingale error `E[P(t,T)/N(t)] / P(0,T) - 1` at t=2y is 0.06%
(5y) and 0.14% (10y) on a flat 3% curve, and 4.2% and 8.8% on a curve rising 3% → 5%.

**Reach.** Every `npv_cube` value past t=0 on a non-flat curve, every exposure and VaR/ES
figure derived from it, and the Bermudan/American scenario pricers, which condition on the
simulated short rate. t=0 prices are unaffected.

**What closing it requires.** See the audit: simulate the zero-mean state and add the
curve-fitted drift, or simulate the LGM state directly; add the martingale check on a
sloped curve as a permanent test.

**Closed on the market path (`price_portfolio` on a `Market`, `engine.portfolio.market_path`; HTTP `POST /v2/portfolio/price`).** The simulation is ORE's cross-asset model: an LGM state per
currency, exact discretization, scenario curves implied by the model and the input curve
(`engine.simulation.cam`, `scenario_market`). `tests/test_cam.py` checks the martingale
`E[P(t,T)/N(t)] = P(0,T)` exactly (analytically, by integrating over the state's normal
distribution) and by Monte Carlo in FP64 and FP32 on sloped curves, and checks the model
against ORE's `LinearGaussMarkovModel`, `IrLgm1fStateProcess` and Cholesky. **Still open on the Hull-White model**, which stays a supported, non-default option (decision A-1); closing it there is plan 9.3. There, a
curve-fitted drift makes the simulation arbitrage-free against the input curve.

---

### I-43 — Options are worth zero after expiry instead of becoming the swap {#i-43}

**Severity:** **High** · **Status:** ⚠️ PARTIAL — closed on the market path (2026-09-29); the Hull-White path is unchanged · **Found:** 2026-09-24, [engine audit
M-3](planning/engine-audit.md#m-3) (registered here 2026-09-28)

**What is wrong.** A European's scenario NPV is 0 at every step after expiry, and a
Bermudan's or American's after its last exercise date. A physically settled option that
was exercised becomes the underlying swap; the engine does not track exercise, so every
path reports 0.

**Size.** From the audit: in `demos/demo.py`, VaR_99 at t=2..5 is 49,820 to 53,207, close to
the portfolio's base NPV of 43,055, with an ES standard error near 0. The "risk" is the
options disappearing on every path at once.

**Current handling.** `engine.portfolio.request` warns when an option expires inside the
simulated horizon.

**What closing it requires.** Record the exercise decision per path at each exercise date
(the backward induction already has both values) and carry the underlying swap's value
afterwards.

**Closed on the market path (`price_portfolio` on a `Market`, `engine.portfolio.market_path`; HTTP `POST /v2/portfolio/price`).** Options are wrapped as ORE's `OptionWrapper` wraps them
(`engine.valuation.options`): exercise at the first grid date on or after each exercise date
when the underlying is worth more than the option. After that a physical option carries the
swap it entered (from `buildUnderlyingSwaps`' first coupon) and a cash-settled one leaves.
`tests/test_valuation.py::test_an_exercised_physical_option_becomes_its_swap_and_a_cash_one_leaves`.
**Still open on the Hull-White model**, which stays a supported, non-default option (decision A-1); closing it there is plan 9.3.

---

### I-44 — Scenario pricing mixes Hull-White and LGM with the same `(a, sigma)` {#i-44}

**Severity:** Medium · **Status:** ⚠️ PARTIAL — closed on the market path (2026-09-29) · **Difference from ORE** on the Hull-White path · **Found:**
2026-09-24, [engine audit A-2](planning/engine-audit.md#a-2) (registered here 2026-09-28)

**What differs.** ORE simulates and prices rates with one model, LGM (the cross-asset
model and `NumericLgmMultiLegOptionEngine`). This engine simulates a Hull-White short rate
(`engine.models.hull_white`) and prices Bermudans/Americans with LGM
(`engine.models.lgm`), then conditions the LGM rollback on the simulated rate through a
state conversion (`r_from_x`). The two are fed the same numbers:
`validate_portfolio_against_simulation` requires a Bermudan's flat `hw_sigma` to equal the
factor's Hull-White volatility, but LGM reads it as the volatility of x, whose short-rate
equivalent is `sigma * exp(-a t)`. With the same `(a, sigma)` the models agree at t=0 and
diverge after; at a=3% the LGM short-rate vol is 14% below the Hull-White one at 5y.
(`docs/reference/ore-parity.md` calls the two "provably equivalent"; they are only under a
reparametrization the engine does not apply.)

**Reach.** Bermudan/American scenario values at every step past t=0, and the exposure and
VaR/ES built on them. t=0 prices equal ORE's (I-31/I-35 parity).

**What closing it requires.** Simulate the LGM state, as ORE does, and price every rates
instrument from it; the audit pairs this with [I-42](#i-42) (M-1).

**Closed on the market path (`price_portfolio` on a `Market`, `engine.portfolio.market_path`; HTTP `POST /v2/portfolio/price`).** One model, LGM, simulates and prices. Each Bermudan/American is
repriced on every path with its own LGM recalibrated to the path's curves, as ORE's
`ValuationEngine` does with `recalibrate = true`. `tests/test_valuation.py::test_bermudan_on_every_path_equals_ore_recalibrated_on_the_path_curves`
agrees with ORE's engine to 1e-8. The recalibration's mechanics are not yet confirmed by an ORE
simulation ([I-49](#i-49)). **Still open on the Hull-White model**, which stays a supported, non-default option (decision A-1); closing it there is plan 9.3.

---

### I-45 — The simulation numeraire is a discretely accrued bank account on factor 0 {#i-45}

**Severity:** Medium · **Status:** ⚠️ PARTIAL — closed on the market path (2026-09-29) · **Difference from ORE** on the Hull-White path · **Found:**
2026-09-28, during the comment review

**What differs.** `engine.simulation.market_model` accrues the numeraire as
`N(t_{i+1}) = N(t_i) * exp(r(t_i) * dt)` (left-point rule on the simulation grid) off rate
factor 0. ORE's numeraire is the LGM numeraire `N(t, x)`, exact at each date given the
state, in the base currency's model. The left-point rule is biased by the rate's change
over each step, so the error grows with the grid spacing (exposure grids are often yearly).

**Reach.** Every exposure profile in `engine.risk.exposure` (EPE, ENE, EE_B, EEE_B, PFE are
all computed on NPV / N), and anything built on them (CVA, EEPE). t=0 prices and the
market-risk path (`engine.market_risk`) do not use it.

**What closing it requires.** Use the model's exact numeraire at each date (the LGM
numeraire, once [I-44](#i-44) moves the simulation to LGM), in the reporting currency.

**Closed on the market path (`price_portfolio` on a `Market`, `engine.portfolio.market_path`; HTTP `POST /v2/portfolio/price`).** The numeraire is the base currency's LGM numeraire
`N(t, x)`, exact at each date (`scenario_market.lgm_numeraire`). The cube stores NPVs and
exposure deflates them by `N`. `tests/test_portfolio_market_path.py::test_a_bills_expected_exposure_is_its_forward_value`
checks an identity that holds only if numeraire and deflation are right (Monte Carlo error
1.4e-5 at 512 paths, tolerance 1e-4). **Still open on the Hull-White model**, which stays a supported, non-default option (decision A-1); closing it there is plan 9.3.

---

### I-46 — European swaptions are priced off the model vol, not the market vol {#i-46}

**Severity:** Medium · **Status:** ⚠️ PARTIAL — closed on the market path and in `engine.market_risk` (2026-09-29) · **Difference from ORE** on the Hull-White path · **Found:**
2026-09-28, during the comment review

**What differs.** ORE's default European swaption engine (`EuropeanSwaptionEngineBuilder`)
is Black/Bachelier on the market swaption volatility surface, so a European's NPV is its
market price. This engine prices Europeans with Hull-White Jamshidian on the trade's
`hw_sigma`, which must equal the simulation's factor volatility and is not calibrated to
the European's own vol. It matches `ORE.JamshidianSwaptionEngine`, not ORE's default.

**Reach.** Every European NPV, and its Greeks: they are model values for the configured
sigma, and differ from the market price unless that sigma happens to reprice the trade.
There is no Vega for Europeans.

**What closing it requires.** A market-vol (Bachelier) European pricer as the default for
t=0 valuation, as ORE has, keeping Jamshidian for scenario pricing; or calibrating the
model to each European's own expiry and tenor.

**Closed on the market path (`price_portfolio` on a `Market`, `engine.portfolio.market_path`; HTTP `POST /v2/portfolio/price`).** Europeans are priced with ORE's `BlackMultiLegOptionEngine`
(Bachelier on the market's normal swaption volatilities) at t=0 and on every path. The
surface is seen from each path date as `DynamicSwaptionVolatilityMatrix` sees it
(`engine.valuation.european`), and Vega is reported. `engine.market_risk` does the same for a
European without Hull-White parameters (plan 6.4), and its ORE-parity test uses QuantLib's
Bachelier engine as reference (measured 1.6e-14). **Still open on the Hull-White model**, which stays a supported, non-default option (decision A-1); closing it there is plan 9.3. Jamshidian stays available there
as a configurable European engine; market-vol Bachelier becomes that model's default too.

---

### I-47 — The calibration basket is not the one ORE builds for the trade {#i-47}

**Severity:** Medium · **Status:** ⚠️ PARTIAL — closed on the market path (2026-09-29) · **Difference from ORE** on the Hull-White path · **Found:**
2026-09-28, during the comment review

**What differs.** ORE's LGM builder calibrates each Bermudan/American to a co-terminal
basket derived from that trade's own exercise dates and underlying. Here the caller
supplies exercise and maturity times as year fractions (`engine.calibration.basket`); each
is rounded to whole months, turned into a tenor-quoted swap starting on the evaluation
date, with expiry two TARGET business days before its first accrual date. One basket, built
on the first uncalibrated trade's curve, evaluation date and index tenor, then serves every
trade and every rate factor (`engine.portfolio.request._fill_calibrated_sigma`).

**Reach.** Every calibrated Bermudan/American: its `Sigma`, and therefore its NPV, Greeks
and Vega, differ from ORE's calibrated price whenever the trade's exercise dates or tenor
differ from the supplied basket, or several trades share it. Parity with ORE's engine
(I-31) holds for a given `Sigma`, not end to end through calibration.

**What closing it requires.** Build each trade's basket from its own exercise dates and
underlying, as ORE does, and calibrate per trade; add an end-to-end parity test against
ORE with `Calibration=Bootstrap`.

**Closed on the market path (`price_portfolio` on a `Market`, `engine.portfolio.market_path`; HTTP `POST /v2/portfolio/price`).** Each trade gets ORE's basket from its own exercise dates and
underlying (`engine.valuation.bermudan.calibration_basket`, `LgmBuilder` and
`IrModelBuilder::buildSwaptionBasket`), built through `ORE.SwaptionHelper` and bootstrapped
per trade (`engine.calibration.ore_lgm`). `tests/test_ore_lgm_calibration.py` prices calibrated
Bermudans equal to ORE's `Calibration=Bootstrap` end to end, measured 2e-11
(tolerance 1e-9). **Still open on the Hull-White model**, which stays a supported, non-default option (decision A-1); closing it there is plan 9.3.

---


### I-49 — Per-path recalibration differs from ORE's in two known details, and is unconfirmed by an ORE run {#i-49}

**Severity:** Medium · **Status:** ❌ OPEN · **Difference from ORE** · **Found:** 2026-09-29,
implementing plan 5.3 (gate V-1)

**What differs.** On the market path every Bermudan/American is recalibrated on each path and
date, as ORE's `ValuationEngine` does with `recalibrate = true`
(`engine.valuation.bermudan`). Two details of ORE's recalibration, read from its source, are
not reproduced:

- ORE keeps the model parametrization's time grid from the first (as-of) build. The engine
  measures each date's bucket times from that date.
- ORE still passes helpers whose expiry has passed on a later date. The engine's basket on a
  date holds only the exercise dates after it.

Separately, the volatility the basket reads on a path date transcribes
`DynamicSwaptionVolatilityMatrix` (`ForwardVariance`). That transcription is checked against
the formula, not against ORE running it: the class has no Python constructor.

**Size.** Not measured. It needs ORE's own cube for a Bermudan on the same scenarios
([I-50](#i-50)). At t=0 and on a single path's curves with a given volatility, the engine
equals ORE (`tests/test_valuation.py`, 1e-8 recalibrated, 1e-10 with a given sigma).

**Reach.** Bermudan/American values past t=0 on the market path, and the exposure built on
them.

**What closing it requires.** An OREApp XVA run with a Bermudan and NPV cube output through
the in-process oracle, compared path by path (with ORE's scenarios, V-4) or in distribution;
then reproduce the two details.

**Decided (2026-09-30): close it** ([compliance/decisions.md](../compliance/decisions.md) X-9; plan 10.1, after [I-50](#i-50)).

---

### I-50 — No path-level or distribution-level parity test against an ORE simulation {#i-50}

**Severity:** Medium · **Status:** ❌ OPEN (validation gap) · **Found:** 2026-09-29, plan §6.2
layers L3 and L4

**What is missing.** The plan's L3 test feeds ORE's dumped scenarios to the engine and compares
the NPV cube cell by cell. Its L4 test compares exposure profiles (EPE, ENE, EE_B, EEE_B,
EPE_B, EEPE_B, PFE) against ORE's XVA analytic statistically. Neither is in the suite. Gate
V-4 (can ORE export per-path scenarios completely enough) is not closed, and the in-process
oracle (`engine.validation.ore_lgm_oracle`) runs pricing analytics only, not a simulation.

**What is checked instead.** Every component against ORE, on a path's curves: model analytics
(ζ, H, bond prices, numeraire, state process, Cholesky) to 1e-12. Each pricer on scenario
curves against the matching ORE engine to 1e-8 to 1e-12. Fixings, cash flows and exercise
rules by ORE's rules. The martingale property exactly and by Monte Carlo. The exposure
numeraire by an exact identity. Assembly errors across components would pass all of these.

**What closing it requires.** Generalize the oracle to an OREApp XVA run (simulation.xml from
a `CamConfig`, the portfolio, cube and exposure reports). Then L4 with independent random
numbers; then, once V-4 closes, L3 on ORE's scenarios.

---

### I-51 — Reported sensitivities are not checked against ORE's sensitivity analytic {#i-51}

**Severity:** Medium · **Status:** ❌ OPEN (validation gap) · **Found:** 2026-09-29, plan §6.2
layer L5

**What is missing.** The market path's Greeks (`engine.risk.sensitivities`) implement ORE's
definitions, read from `sensitivityanalysis.cpp` and `sensitivitycube.cpp`: absolute
zero-rate shifts at the curve tenors, forward-difference Delta, `up − 2·base + down` Gamma,
Vega per quote, and Theta on the rolled market. No test compares them with an OREApp
sensitivity run.

**What is checked instead.** Delta and Gamma equal the AD derivatives to the bump's order.
Vega adds up to a parallel bump. Theta rolls one calendar day, backfills the as-of fixing and
adds back paid flows (`tests/test_sensitivities.py`). A different shift convention in ORE's
simulation market (for example on how a shifted tenor point interpolates) would pass all of
these.

**What closing it requires.** Run ORE's sensitivity analytic through the in-process oracle on
the shared test portfolio (`tests/support/portfolio.py`) and compare per trade, factor and
tenor to 1e-8 relative, as plan §6.2 L5 sets.

---

### I-53 — The market path is slow: the full suite went from about 23 to about 46 minutes {#i-53}

**Severity:** Medium · **Status:** ❌ OPEN (performance) · **Found:** 2026-09-29, the first
full runs after the ORE alignment

**What is slow.** The final full runs took 46:00 on Windows and 50:09 in a 4-core Linux
container, run at the same time on one 24-thread machine. The baseline (the code before the
alignment) took 22:54. Two earlier runs that shared the machine with more work took 1h30m and
1h24m, so wall time here is sensitive to load. The slowest test,
`tests/test_api_market_path.py::test_result_matches_direct_price_portfolio_call`, took 275s
(Windows) and 308s (Linux) in the final runs. It prices the 8-trade shared portfolio on 128 paths and 4 dates
twice: once in a fresh worker process, once directly. The next slowest are the module
fixtures of `test_portfolio_market_path.py` and `test_valuation.py`, about 110 to 130s each.

**Likely cause, not yet profiled.** On each path date every Bermudan/American rebuilds its
basket through `ORE.SwaptionHelper` and bootstraps, then rolls back on its grid (per-path
recalibration, ORE's `recalibrate = true`). Sensitivities revalue in Python loops, one
bump at a time. Neither is jitted end to end, and each fresh process recompiles.

**What closing it requires.** Profile a market-path job (`JAX_RISK_PROFILE_DIR`,
[profiling](concepts/profiling.md)), then the plan's Phase 7 once I-49 to I-51 have frozen
the numbers. `PricingConfig(recalibrate=False)` is available where ORE's own semantics are
not needed.

---

### I-54 — No swaption smile: an option away from the money reads the ATM volatility {#i-54}

**Severity:** Medium · **Status:** ❌ OPEN · **Difference from ORE** · **Found:** 2026-09-29,
plan X-5; decided 2026-09-30

**What differs.** The market's swaption volatilities are an ATM normal matrix
(`engine.market.SwaptionVolSurface`, expiry × swap tenor, no strike axis). ORE reads a
volatility cube or a SABR smile at each option's strike. Here every option reads the ATM
volatility for its expiry and tenor: Europeans, and the calibration helpers of
Bermudans/Americans, which `CoterminalDealStrike` strikes at the deal rate.

**Size.** None when the market itself is ATM-only, as every market given to the engine so far
is. With a smile, the error grows with the distance from the money and the steepness of the
smile. Not measured.

**Reach.** European NPVs and Vega away from the money; calibrated Bermudan/American volatility,
hence their NPVs, Greeks and exposure.

**What closing it requires.** A strike axis in the market's volatilities, read at each option's
and helper's strike as ORE reads its cube; SABR later as an option. **Decided (2026-09-30): close
it** ([compliance/decisions.md](../compliance/decisions.md) X-5; plan 10.2).

---

### I-55 — Unproven precision combinations are not flagged; precision is switched per process {#i-55}

**Severity:** Medium · **Status:** ❌ OPEN · **Found:** 2026-09-24 as [engine audit
A-1](planning/engine-audit.md#a-1) (the mechanism); registered 2026-09-30 with the owner's
decisions D-9 and A-9

**What is missing.** Two parts:

1. **No warning for an unproven combination.** Precision can be set per stage
   (`PrecisionConfig`: simulation, pricing, risk, calibration). Decision D-9 allows any
   combination for any calculation, regulatory figures included. But nothing records which
   combinations have been shown adequate for which figure, or at how many paths, so a
   float32 VaR or exposure profile comes back looking exactly like a validated one.
2. **The mechanism.** `jax_enable_x64` is process-global. The engine toggles it
   (`generate_paths`, `price_portfolio`), serializes threads with `_PRICING_LOCK`, and keeps
   one worker-process pool per precision tier. As the audit found, `price_portfolio` turns x64
   back on in every job, so the tiers do not isolate what they were built to isolate. The
   market-path code already takes explicit dtypes.

**Reach.** Every run below FP64: the numbers may be adequate, but the result does not say
whether they are.

**What closing it requires.** Decided (2026-09-30; [compliance/decisions.md](../compliance/decisions.md)
A-9, D-9): keep adjustable precision and replace the mechanism before removing it. x64 is
enabled once per process and each stage takes an explicit dtype (plan 9.4). A table of
evidence per figure and precision drives a warning on any result whose combination is
unproven, stating what is validated and at how many paths (plan 9.5).

---

### I-56 — The API cannot reach every setting, and two routes are named like versions {#i-56}

**Severity:** Medium · **Status:** ❌ OPEN · **Found:** 2026-09-30, the owner's review of the API
(decision A-2)

**What is wrong.** Three things:

1. **Names that look like versions.** The market path is served at `POST /v2/portfolio/price`,
   its body carries `schema_version: "2"`, and earlier documents called the two request shapes
   "version 1" and "version 2". They are not versions. They are two models: the Hull-White
   model at `POST /portfolio/price`, the market path (the default model) at `/v2`. A caller can
   reasonably read `/v2` as the successor of `/portfolio/price`, or `schema_version` as a
   revision of the contract.
2. **The model is chosen by the request's shape**, not by a setting, so no request can mix
   options across the two (for example the Hull-White simulation with ORE's sensitivities).
3. **Settings the API cannot reach** (compared field by field, 2026-09-30):
   - the sensitivity settings (curve tenors, shift sizes, Theta horizon, the vol decay on the
     Theta date: `engine.risk.sensitivities.SensitivityConfig`). `price_portfolio` always uses
     their defaults, so they are unreachable from Python's portfolio entry point too;
   - market-risk VaR/ES (`engine.market_risk.run_market_risk`): no route at all;
   - the market path's calibrations (`engine.calibration.cam`, `engine.calibration.ore_lgm`)
     as standalone runs. `POST /calibration/lgm` serves only the Hull-White path's shared basket;
   - trade ids on the Hull-White request shape (the market path's has `trade_id`, [I-10](#i-10)).

   Every other setting of the two request shapes is reachable. The Bermudan/American grid
   settings that the market path's trade schemas lack belong to its engine configuration,
   which the request carries.

**Reach.** Nothing is priced wrongly: an unreachable setting runs at its default, and the
default is documented. But the API does not yet do what the engine can, and its names
mislead.

**What closing it requires.** Decided (2026-09-30, [compliance/decisions.md](../compliance/decisions.md)
A-2): one route and one request whose configuration reaches every setting, validated before
any job starts, with names that say what they are and no version-like names except for real
contract revisions. Today's routes keep answering, translated into that request. A
completeness test fails when a configuration setting has no API field (plan 9.2).

---

## ASSUMPTION — nothing known to be broken, premise unconfirmed

> The code behaves exactly as designed. What is unverified is whether the design reads an
> external contract correctly. Closes when the question is answered, not when code changes.

### I-23 — The `accrualBasis` strictness rule is an assumption, not a confirmed contract {#i-23}

**Severity:** Medium · **Status:** ⚠️ ASSUMPTION — unconfirmed · **Raised:** 2026-09-16 with W1.6.1

**This entry is a different kind from the others.** Everything above records something the
code *does wrong*. This records something the code does **deliberately, on an unverified
reading of an unanswered question** — which is worth registering precisely because it will
otherwise look settled. Nothing here is known to be broken; what is unknown is whether the
interpretation matches TraderX's intent.

**The question, asked and never answered.** [Response v4](planning/traderX_integration/eod-contract-response-v4.md)
§1.3 asked: when a real settlement calendar arrives, does `traderx.accrual-basis.v1` **gain
new `dateBasis` / `settlementAdjustment` values**, or does it become `accrual-basis.v2`? That
went out on 2026-09-16 and TraderX's v5 reply did not address it. It was asked again in
[response v6](planning/traderX_integration/eod-contract-response-v6.md) §2.3.

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
  [EOD Contract Proposal](planning/traderX_integration/eod-contract-proposal.md) (integration context and the
  full field-level requirements), [HTTP API](reference/http-api.md),
  [The Portfolio Entry Point](reference/portfolio-entrypoint.md),
  [Profiling & the Tracer](concepts/profiling.md) (how to measure where a job's time
  actually goes, and the known cost characteristics of the Greeks path).
