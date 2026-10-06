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
honest without it. The project goals in the root [README](../../README.md) decide the line:
ORE parity, the precision research on multiple TPUs, and the TraderX integration are the
current scope; regulatory figures, new asset classes and new precision formats are features.

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

- IDs are permanent. Never renumber, never reuse. `I-NN` IDs are cited from code, tests,
  other docs, and the published EOD capability document
  (`engine/integration/capabilities.py`), so a renumbering breaks an external contract.
- New issues take the next free `I-NN`; new features the next free `F-NN`.
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
| Audit M-2 | [I-04](known-issues.md#i-04) (the Hull-White half fixed by roadmap 1.3; the TraderX half open) |
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
| ORE alignment plan Phase 7 (performance) | [I-53](known-issues.md#i-53), roadmap [2.4 and 2.5](roadmap.md#stage-2--the-demo-on-a-local-gpu) |
| ORE alignment plan Phase 9 (configurable engine) | Roadmap [stage 1](roadmap.md#stage-1--structure); [details/configurable-engine.md](details/configurable-engine.md) |
| ORE alignment plan Phase 10.1, 10.2 | [I-49](known-issues.md#i-49), [I-54](known-issues.md#i-54) |
| ORE alignment plan Phase 10.3, 10.4 | [F-04](features.md#f-04) |
| ORE alignment plan §1.5, §6 (test layers, shared portfolio), gates V-n | [details/ore-parity-validation.md](details/ore-parity-validation.md) |
| TraderX plan W0, W1 | Done |
| TraderX plan W2 | [I-05](known-issues.md#i-05), [I-16](known-issues.md#i-16); [details/traderx-integration.md](details/traderx-integration.md) |
| TraderX response v7 A-02 to A-05 | [I-57](known-issues.md#i-57), [I-58](known-issues.md#i-58), [I-59](known-issues.md#i-59) |
| Roadmap phase 10 (XVA) | [F-06](features.md#f-06) |
| Roadmap phase 12 (lower-precision formats) | [F-07](features.md#f-07) |
| Basel plan P0–P7 | [F-05](features.md#f-05); [details/basel-iii.md](details/basel-iii.md) |
| Precision research, `details/sub-fp32-precision.md` | [F-07](features.md#f-07); [details/precision.md](details/precision.md) (its findings, corrected, in §15) |
