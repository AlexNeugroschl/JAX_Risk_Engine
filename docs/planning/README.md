# Planning

Everything the project still has to fix or build, and the order to do it in. Three
documents carry the whole plan; `details/` holds the longer write-ups they point to.

| Document | Holds | Question it answers |
|---|---|---|
| [known-issues.md](known-issues.md) | Open defects and important shortcomings, `I-NN` | What is wrong or missing in what the engine claims to do today? |
| [features.md](features.md) | Additive work, `F-NN` | What new capability should the engine gain? |
| [roadmap.md](roadmap.md) | The order of work, in stages | What do I do next, and why that first? |
| [details/](details/) | Design and background for large items | How exactly is a large item done? |

A reader should be able to take the next step from these three files alone. A `details/`
document explains *how*; it never adds work the three files do not list.

## Where an item goes

**known-issues.md** if a user of today's engine can hit it:

- a number that is wrong, unverified against ORE, or silently less precise than stated;
- a crash, a hang, or an input accepted and then ignored;
- a refusal of something a current consumer sends (a scope gap), or a stated project goal
  that the code does not meet (for example: runs on one device only);
- a gap in the tests or tooling that lets a defect of the kinds above through.

**features.md** if it adds capability beyond today's scope, and the engine is correct and
honest without it. The project goals in the root [README](../../README.md) decide the line.
ORE parity, running accurately and fast on TPU and GPU clusters, the precision research on those
clusters, and one run request for every caller are the current scope. Regulatory figures, new
asset classes and new precision formats are features.

If an item is both (a feature that also closes a shortcoming), it lives in known-issues.md
and the feature entry links to it.

## Entries

**known-issues.md** entry, in this order and no longer than needed:

```markdown
<a id="i-nn"></a>
### I-NN — What a user sees, stated as the defect

**Severity:** High | Medium | Low · **Status:** OPEN | PARTIAL | FLAGGED | ASSUMPTION ·
**Category:** Correctness | Validation | Performance | API | Scope | Tooling · **Found:** date, how

**What is wrong.** The symptom, then the cause.
**Reach.** Which numbers or callers it affects, and which it does not.
**Current handling.** Warning, refusal, or nothing.
**To close.** The change and the test that proves it; link `details/` if long.
```

| Status | Meaning |
|---|---|
| OPEN | Not addressed. |
| PARTIAL | Closed in one named part, open in another (for example: fixed on the trade configs, open on the results). |
| FLAGGED | The number is unchanged; the engine now warns. Never a softer FIXED. |
| ASSUMPTION | Nothing known to be broken, but the code acts on an unconfirmed reading of an external contract. Closes when the other party answers. |

Severity: **High** — a wrong number on an ordinary input, or another caller's result
served. **Medium** — wrong or unverified on a common path, or a stated goal unmet.
**Low** — narrow, cosmetic, or tooling.

Anchors are HTML (`<a id="i-nn"></a>` on the line above the heading), which render on
GitHub and in editors alike; an entry that absorbed a retired ID carries that ID's anchor too
(`<a id="m-1"></a>`), so old links keep landing on it.

**features.md** entry: `F-NN`, a title, the value to the project, dependencies (issues,
stages, external inputs), size (S ≤ 3 days, M ≤ 2 weeks, L longer), and a `details/` link
if it has one.

## IDs

- IDs are permanent. Never renumber, never reuse. `I-NN` IDs are cited from code, tests and
  other docs, and from the retired `/eod` path's published capability document
  (`engine/traderx/capabilities.py`) until [I-120](known-issues.md#i-120) removes it, so a
  renumbering breaks those citations.
- New issues take the next free `I-NN`; new features the next free `F-NN`.
- Roadmap step numbers are not IDs. They follow the order of work and are renumbered when it
  changes, so only roadmap.md uses them: code, error messages, tests and the other documents
  cite the item, the decision or a date (`tests/test_import_layering.py` checks the code).
- Retired ID schemes (the 2026-09-24 audit's `M-`, `P-`, `A-`, `Q-`, `R-` items, the ORE
  alignment plan's phases, the TraderX plan's `W` tasks) still appear in code comments. The
  [ID map](#retired-ids) below says where each went. Do not create new IDs in those schemes.
- Owner decisions (`A-n`, `D-n`, `X-n`, `T-n`) live in
  [compliance/decisions.md](../../compliance/decisions.md), which is a decision record, not a
  plan. A plan entry cites the decision it implements.

## Lifecycle

1. **Open.** Add the entry to known-issues.md or features.md, and place it in a stage of
   roadmap.md in the same change. An item not in the roadmap is not planned.
2. **Work.** Update the entry when the understanding changes; delete statements that stop
   being true rather than appending "Update:" paragraphs. An entry says what is true now.
3. **Close an issue.** The regression test must be shown failing against the pre-fix code
   (red first). Then delete the entry and add one row to the
   [closed ledger](known-issues.md#closed) with the ID, one line, and the test. The
   narrative goes in the commit message, where it is permanently findable with
   `git log --grep "I-NN"`.
4. **Finish a feature.** Delete the entry and its roadmap line; document the capability in
   the user-facing docs (`docs/reference`, `docs/risk`, ...). Features have no ledger.
5. **Finish a stage.** Remove it from roadmap.md. The roadmap lists only work to do.
6. **Details.** Delete a `details/` document when nothing open links to it.

### Why fixed bugs keep one line and nothing more

Full write-ups of fixed bugs made the old register 176 KB, mostly history. The history is
in git and the protection is in the regression tests. The one-line ledger stays because
code, tests and older docs cite fixed IDs (`I-06`, `I-29`, ...): the ledger keeps each ID
resolvable, names the test that guards it, and stops the ID being reused.

## Verification rules

Any statement here that the suite passed follows these, each written after it was
violated once:

1. Run `.venv/Scripts/python.exe -m pytest`. The system interpreter lacks `pydantic` and
   `jsonschema`, so the API and schema tests silently fail to collect.
2. Take counts from `pytest --collect-only -q tests/`. Never grep for `def test_`
   (misses parametrized cases), never collect outside `tests/` (`reference/` has its own
   test trees).
3. A run counts as green only if the summary line was printed and there are no
   `FAILED`/`ERROR` lines. An exit code alone is not evidence ([I-27](known-issues.md#i-27)
   can kill the process with no summary). Leave the working tree alone during a run.
4. A count that does not reconcile with the previous one is a signal to find out why.
5. Changes to processes, file paths or platform defaults also run on Linux (Docker
   `python:3.11`, as CI does).
6. A green suite is evidence about the tests, not proof about the code. Parity claims need
   an ORE reference, not a re-derivation.

Record only the latest full run, in [known-issues.md](known-issues.md#verification-status).

## Retired IDs

Where the IDs of retired planning documents went. Their full text is in git history
(last present at commit `8306073`).

| Retired ID | Now |
|---|---|
| Audit M-1 | [I-42](known-issues.md#i-42) (fixed) |
| Audit M-2 | [I-04](known-issues.md#i-04) (the Hull-White half fixed on 2026-10-01; the TraderX half retired with the `/eod` path, I-120) |
| Audit M-3 | [I-43](known-issues.md#i-43) (fixed) |
| Audit M-4, M-5, R-1, P-3, A-5, A-7, Q-1 | Fixed; [closed ledger](known-issues.md#closed) |
| Audit P-1 | [I-61](known-issues.md#i-61) |
| Audit P-2 | [I-62](known-issues.md#i-62) (fixed) |
| Audit A-1 | [I-55](known-issues.md#i-55) |
| Audit A-2 | [I-44](known-issues.md#i-44) (fixed) |
| Audit A-3 | [I-63](known-issues.md#i-63) (fixed) |
| Audit A-4 | [I-64](known-issues.md#i-64) (fixed) |
| Audit A-6 | [I-65](known-issues.md#i-65) |
| Audit Q-2 (lint, types) | [I-66](known-issues.md#i-66) |
| Audit Q-3, cleanup plan Phase 3 | [I-67](known-issues.md#i-67) |
| Audit Q-4 | Done by this reorganization |
| ORE alignment plan Phases 0–6, 8 | Done; remaining items are issues |
| ORE alignment plan Phase 7 (performance) | [I-53](known-issues.md#i-53); done 2026-10-06 and 2026-10-07 (the profiling demo on a local GPU) |
| ORE alignment plan Phase 9 (configurable engine) | Done 2026-10-04; [details/configurable-engine.md](details/configurable-engine.md) |
| ORE alignment plan Phase 10.1, 10.2 | [I-49](known-issues.md#i-49) (closed 2026-10-12), [I-54](known-issues.md#i-54) |
| ORE alignment plan Phase 10.3, 10.4 | [F-04](features.md#f-04) |
| ORE alignment plan §1.5, §6 (test layers, shared portfolio), gates V-n | [details/ore-parity-validation.md](details/ore-parity-validation.md) |
| TraderX plan W0, W1 | Done |
| TraderX plan W2 | [I-05](known-issues.md#i-05); the rest retired with the `/eod` path ([I-120](known-issues.md#i-120)) |
| TraderX response v7 A-02 to A-05 | [I-57](known-issues.md#i-57) (fixed); the rest retired with the `/eod` path ([I-120](known-issues.md#i-120)) |
| I-04, I-08, I-16, I-18, I-23, I-58, I-59, I-60, I-93, I-107, I-108, I-109 (2026-10-19) | Retired unfixed with the `/eod` path, since TraderX submits through the run request: [I-120](known-issues.md#i-120), which carries their anchors |
| F-08, reporting currencies for TraderX (2026-10-19) | Retired with the `/eod` path. The run request already reports in a chosen currency, converting each trade at the as-of FX spot |
| Roadmap phase 10 (XVA) | [F-06](features.md#f-06) |
| Roadmap phase 12 (lower-precision formats) | [F-07](features.md#f-07) |
| Basel plan P0–P7 | [F-05](features.md#f-05); [details/basel-iii.md](details/basel-iii.md) |
| Precision research, `details/sub-fp32-precision.md` | [F-07](features.md#f-07); [details/precision.md](details/precision.md) (its findings, corrected, in §15) |
| Project audit of 2026-10-10, AUD-01 to AUD-51 | Each finding's item, or the reason it was set aside: [below](#the-project-audit-of-2026-10-10) |

<a id="the-project-audit-of-2026-10-10"></a>
### The project audit of 2026-10-10

The audit was a static review of the code at `53f85b9`, checked against the code at `c6e59da`
on 2026-10-11 and merged into the plan that day. Severities are this register's, so some differ
from the audit's own. Its report was deleted on 2026-10-19, after every finding had been merged;
its full text is in git history (`git log -- docs/planning/project-audit.md`). Code and test
comments cite its `AUD-NN` findings.

| Findings | Now |
|---|---|
| AUD-01, AUD-03, AUD-04 (VaR rank, quantile keys, confidence domain) | [I-102](known-issues.md#i-102) (fixed; the broader domain is documented as broader than ORE's) |
| AUD-02 (currencies in market risk) | [I-100](known-issues.md#i-100) (fixed) |
| AUD-05, AUD-10, AUD-35, AUD-40 (input validation, late exercise, preflight, strict schemas) | [I-103](known-issues.md#i-103), [I-87](known-issues.md#i-87), [I-88](known-issues.md#i-88), [I-95](known-issues.md#i-95) (fixed) |
| AUD-06, AUD-07, AUD-13 (calibration acceptance, zero volatility, the solver's fit check) | [I-73](known-issues.md#i-73) (fixed) |
| AUD-08, AUD-11, AUD-14 (zero-variance European, delayed last payment, H at small a·t) | [I-104](known-issues.md#i-104) |
| AUD-09 (zero and negative option notionals) | [I-101](known-issues.md#i-101) (fixed) |
| AUD-12 (Theta's add-back of fixings) | [I-116](known-issues.md#i-116) (fixed: no add-back, as ORE) |
| AUD-15, AUD-46 (qualification matrix, stale documentation) | The parity matrix ([I-116](known-issues.md#i-116), fixed); [I-110](known-issues.md#i-110) |
| AUD-16, AUD-17 (Bermudan paths, AD against bump Greeks) | [I-32](known-issues.md#i-32), [I-49](known-issues.md#i-49), [I-51](known-issues.md#i-51), [I-78](known-issues.md#i-78) (fixed) |
| AUD-18 (generic index conventions) | [I-05](known-issues.md#i-05), [I-54](known-issues.md#i-54) |
| AUD-19, AUD-20, AUD-21 (standard errors, qualification on results, empty warnings) | [I-55](known-issues.md#i-55), [I-111](known-issues.md#i-111) (fixed); [F-10](features.md#f-10) |
| AUD-22 to AUD-34 (the TraderX path's identity, rows, persistence, settings, pricers, file paths) | Retired with the `/eod` path, [I-120](known-issues.md#i-120); AUD-34's API limits are [I-90](known-issues.md#i-90) |
| AUD-36, AUD-38 (queue migration, one worker per queue file) | [I-106](known-issues.md#i-106) (fixed) |
| AUD-37 (admission and lifecycle) | [I-76](known-issues.md#i-76), [I-77](known-issues.md#i-77), [I-89](known-issues.md#i-89), [I-91](known-issues.md#i-91) (fixed) |
| AUD-39 (result provenance) | [I-87](known-issues.md#i-87) (fixed); Basel's run manifest ([F-05](features.md#f-05)) |
| AUD-41, AUD-42 (rollback memory, the float64 cube) | [I-83](known-issues.md#i-83), [I-105](known-issues.md#i-105) (fixed) |
| AUD-43 (host work and eager programs) | [I-81](known-issues.md#i-81), [I-110](known-issues.md#i-110) |
| AUD-45 (sharding by divisors, block alignment) | [I-61](known-issues.md#i-61) (the warning fixed; alignment on the devices open) |
| AUD-47 (CI coverage) | [I-27](known-issues.md#i-27), [I-110](known-issues.md#i-110); independent bookings with [F-09](features.md#f-09), mixed currencies with [F-04](features.md#f-04)'s L6 test |
| AUD-48 (duplicate algorithms) | [I-94](known-issues.md#i-94) (fixed); TraderX's half [I-120](known-issues.md#i-120) |
| AUD-49 (dependency hygiene, licence) | [I-66](known-issues.md#i-66) (`pandas`); the licence is waiting on the owner ([roadmap](roadmap.md#waiting-on-others)) |
| The architecture review's advice to check the target hardware before the research timing | [I-121](known-issues.md#i-121) |

Set aside, with the reason:

- **AUD-13, in part:** a general convergence status from a fixed-step solver. By its design
  (A-21), a root beyond a bounded bracket is that bracket's end, and the calibration's acceptance
  judges it. Derivatives at a clipped root are also set aside: the unbounded problems widen
  to 2^63, and bounded bootstrap roots carry no tangent.
- **AUD-40, in part:** deep immutability of Python inputs. A Python caller owns its objects, and
  HTTP jobs are rebuilt from JSON in the worker.
- **AUD-43, in part:** the Brownian-bridge matrix is `[T, T]` over the time steps, a few hundred
  of them, which is negligible.
- **AUD-44:** not a defect. The Hessian diagonal needs one Hessian-vector product per factor,
  each product yields its whole column anyway, and the factors are tens of pillars, so the
  `F × F` block is kilobytes.
- **AUD-49, in part:** broad ranges in the package metadata. `constraints.txt` is the qualified
  environment, and `tests/test_environment.py` checks it.
- **AUD-50:** the profiling demo served the 2026-10-07 milestone and is not the research's
  timing instrument; the research result states its own conditions (F-07).
- **AUD-51:** resolved before the merge: `products._quantize` gives no scale to formats with
  float32's range.
