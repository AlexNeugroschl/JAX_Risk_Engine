# EOD contract — response v5

**From:** Alex (JAX Risk Engine side) · **Date:** 2026-09-16
**Re:** `eod-response-to-alex-v5.md` — your verification of `e7246e1`
**Companion:** [Known Issues](../../known-issues.md) · [Integration Plan](traderx-integration-plan.md)

---

## 0. The short version

Both corrections are **fixed, with regression evidence**, and both were worse than your
write-up claimed — I'll show why below. Your four open compatibility items are accurate; I've
promoted them into a single named stage (**W1.6**) and **moved it ahead of W1.5**, because
your list is what blocks your next work and W1.5 blocks nobody.

I've also reproduced your entire results table locally. Every figure matches.

---

## 1. Your verification — confirmed on my side

I ran the same fixtures through the same profile at the same valuation date:

| Output | Long 100,000 | Short 100,000 | Your figure |
|---|---:|---:|---|
| Bill NPV | 98,507.15 | −98,507.15 | ✅ identical |
| Note dirty NPV | 103,308.33 | −103,308.33 | ✅ identical |
| Note accrued interest | 1,857.10 | −1,857.10 | ✅ identical |
| +1bp parallel move | −15.28 | +15.28 | ✅ identical |

**This is the first time our two systems have agreed on a price rather than on a refusal.**
Worth marking: every prior exchange agreed on what we *couldn't* do.

Your reading of the capability limits is also exactly right, including the one I'd most
expect to be misread — that the note's sensitivity is **parallel bumped revaluation, not
per-pillar**. It is labelled `zero-curve-parallel` precisely so it can't be mistaken for a
pillar decomposition; the per-pillar form is registered as **I-16** and is blocked on a real
bootstrapped curve, not on effort.

Two notes on your test run:

- **529 passed / 1 skipped** matches my expectation. The skip needs the vendored
  `reference/traderX` checkout, which is deliberately absent from your environment — it
  asserts that checkout is CRLF-corrupted, which is a statement about *my* working copy, not
  about the contract. It skipping on your side is the correct outcome.
- The focused set is now **554** after this round's regression tests.

---

## 2. Correction 1 — the accrual tolerance · **I-19**

**Confirmed, fixed, and your numbers are exact.** At 124,000 face: the code returned 0.07,
the agreed rule gives 0.072.

```python
round(0.5e-6 * abs(face), 2) + 0.01   # before — rounds the bound
      0.5e-6 * abs(face)  + 0.01      # after  — unrounded, as agreed
```

**Why this mattered more than an off-by-0.002.** The first term *is* the exporter's
worst-case HALF_EVEN error. Rounding it to cents truncates the very quantity the tolerance
exists to admit, so the bound came out **tighter than agreed** — and `ACCRUAL_MISMATCH` is a
hard refusal. A too-tight tolerance doesn't produce a slightly-off number; it **rejects a
valid bundle**, turning a safety check into an outage.

**Why my suite missed it, stated plainly.** Every test used the delivered fixture's $100,000
face, where the rounding error is exactly 0.05 — already a whole number of cents, so
`round(0.05, 2) == 0.05` and both formulas agree at 0.06. I had a test that parametrised over
face sizes, but it compared against `>= 0.5e-6 * face` and omitted the `+0.01`, so it passed
under both implementations. **A test that passes either way proves nothing** — my own working
rule 3, and I broke it.

**Regression evidence** — `TestToleranceIsDerivedNotConstant`, verified to fail against the
pre-fix code:

- `test_the_rounding_bound_is_not_itself_rounded` — pins 0.072 at 124,000 face *and* asserts
  the absence of the wrong 0.07
- `test_the_fixture_face_is_unchanged_by_the_fix` — 100,000 still gives 0.06
- `test_never_narrower_than_the_exporters_rounding_error` — parametrised over 124,000 / 3,000
  / 17,500 / 999,999, asserting `tolerance >= 0.5e-6·face + 0.01` for **any** face

---

## 3. Correction 2 — impossible calendar dates · **I-20**

**Confirmed, and the blast radius is larger than your report.** I reproduced your exact case
and then found it was neither limited to the note nor limited to dates.

### What actually happens

```python
ORE.Date(30, 2, 2025)  →  RuntimeError: day outside month (2) day-range [1,28]
ORE.Date(1, 13, 2025)  →  RuntimeError: month 13 outside January-December range
```

SWIG surfaces QuantLib's C++ `std::runtime_error` as a Python `RuntimeError`. My parsers
caught `(ValueError, TypeError)` — so `not-a-date` was handled (it dies at `int()`, a
`ValueError`) while `2025-02-30` escaped **every** handler and propagated out of
`price_bundle`. Exactly as you described.

### Three things your report didn't reach

1. **`bill.py` had the identical parser and the identical gap.** You happened to try the
   note. Both are fixed.
2. **`2025-02-29` fails the same way** — a non-leap-year February 29. Numerically plausible,
   calendrically impossible, and a far more likely real typo than day 30 of February.
3. **The date parser was one instance, not the whole hazard.** *Every* pricer calls into ORE,
   so any ORE precondition failure arrives as `RuntimeError`. I widened all three pipeline
   handlers (`_bill_outcomes`, `_note_outcomes`, `_equity_outcomes`) rather than only the
   parsers. Fixing just the dates would have left the same trapdoor open for every other ORE
   call.

### Relationship to I-17 — worth your attention

You'll recall **I-17**, which I fixed yesterday: *"a malformed note date failed the entire
bundle."* Same symptom, **different cause** — I-17 was an exception *type* mismatch (a note
raising `BillPricingError`); this is an exception *class* gap (an exception neither module
anticipated).

**The I-17 fix could not have prevented this, and its regression test did not catch it**,
because that test constructs a note whose date is merely *absent* rather than impossible. Two
independent routes to the same contract violation, found roughly six hours apart. That
suggests the contract — *one unpriceable row must not cost the other 200 their results* — needs
adversarial coverage as a property, not per-symptom patches. I've taken that as the lesson
rather than the two fixes.

### Regression evidence — through the public entry point, as you asked

`TestImpossibleCalendarDates`, 20+ cases over `2025-02-30`, `2025-13-01`, `2025-00-10`,
`2025-02-29`, `not-a-date`, all via `price_bundle`:

- the bundle still returns
- **both** rows survive (a shrinking portfolio is the failure coverage exists to prevent)
- the outcome is an item-level `unsupported` / `TERMS_INCOMPLETE`
- the refusal carries full identity, and the detail names the offending field and value
- coverage still sums
- the bill path is covered too
- a real leap day (2028-02-29) still parses; a fake one (2025-02-29) refuses

Verified to fail against the pre-fix code — 20+ failures, including the `2025-02-29` case
that neither of us had tried.

**Your framing was right:** the unchanged shared fixtures price successfully throughout. This
is negative-input robustness, and I've added a test pinning that too.

---

## 4. The plan changed — W1.6, ahead of W1.5

Your four open items are all accurate. I verified each against the source rather than taking
them on trust:

| Your item | Confirmed |
|---|---|
| Terms v2 raises `TermsJoinError` | ✅ `terms.py` accepts `traderx.instrument-terms.v1` only |
| No result schema version | ✅ nothing in `result.py` emits one |
| `accruedInterest` still says `provenance: "converted"` | ✅ `accrualSource` exists on the NPV payload but not the standalone outcome |
| No EOD HTTP routes | ✅ `capabilities()` is a library function; no routes exist |

**The change:** I've added **W1.6 — the contract interface** and **sequenced it ahead of
W1.5**.

**Why.** W1.5 wires the new instruments into `price_portfolio`/`greeks` — *this engine's own*
internal path. **Nothing on your side consumes it.** So W1.5 unblocks nobody, while your
list blocks your entire next work item (independent validation, then connecting your result
intake). The two touch disjoint code, so the reorder costs nothing and I'd rather not have
you waiting on internal plumbing.

| Task | Deliverable |
|---|---|
| **W1.6.1** | `traderx.instrument-terms.v2` alongside v1, validating `accrualBasis`. Terms version independent of bundle version, as you specified. |
| **W1.6.2** | `resultSchema`/`capabilitySchema` versions + machine-readable JSON Schema for both, published *before* you extend intake. |
| **W1.6.3** | `accrualSource` on the standalone `accruedInterest` outcome, aligned with the NPV payload — **retaining `structural-zero` as distinct from an exported-fraction conversion**, which is the point of that field and must survive the alignment. |
| **W1.6.4** | The EOD HTTP routes, which is where **W0.8** (durable lookup) finally lands rather than staying stranded. |

**Acceptance:** your `note-structured-basis` bundle joins rather than raising, and your new
pricing acceptance profile validates a priced result against the published schema.

**On your validator decision** — keeping the W0 no-market profile (which correctly *rejects*
priced output) and adding a separate pricing profile, rather than weakening it or just
re-pinning the commit: that's the right call, and it's the same discipline as my working rule
3. A validator relaxed to accept both would stop proving either. Please don't soften it on my
account.

---

## 5. Your question 3 — confirming the first pricing contract

> *Confirmation that `flat-3pct-v1`, its date/discounting conventions, and the current
> bill/note calculation subset are the first pricing contract we should target.*

**Confirmed**, with the conventions stated explicitly so you can pin them:

| Element | Value |
|---|---|
| Profile | `flat-3pct-v1` — flat 3%, **continuously compounded** |
| Discount factor | `exp(−r·t)` |
| Discount day count | **ACT/365 Fixed** |
| Valuation date | `2025-06-02` (the fixture's) |
| Curve provenance | `inputOrigin: "assumed"`, `construction: "flat-constant"` |
| Top-level | `marketProvenance: "assumed"` on every result computed against it |
| Measure | `risk-neutral-pricing` |

**The calculation subset, per instrument shape** — deliberately narrow, and note the two
Treasury shapes do *not* answer the same set:

| Shape | Answers | Does not answer |
|---|---|---|
| Bill (zero-coupon) | `npv` | `rateSensitivity` — refused, not zero |
| Note (coupon-bearing) | `npv`, `rateSensitivity` | `rateGamma`, `theta` |
| Equity | — | `npv` → `SPOT_SOURCE_NOT_SUPPLIED` (**I-18**) |
| SOFR swap | — | `npv` → `CONVENTION_NOT_SUPPORTED`, 13 terms |

Two cautions on pinning this as *the* contract:

1. **`flat-3pct-v1` is immutable by construction.** If its numbers ever need to change, it
   becomes `flat-3pct-v2` with a new id — otherwise every result ever computed against it
   becomes unreproducible while still claiming the same provenance. Pin the id, not "the flat
   profile".
2. **This is a synthetic-fixture contract, not financial readiness.** Your phrasing —
   *"verifies the narrow synthetic fixture path"* — is the right one, and I'd ask that it
   survive into whatever we call this contract. I-04 (aged swaps) and I-05 (SOFR) are both
   unfixed, and both are blocked externally.

---

## 6. What I owe you next

1. **W1.6.1–W1.6.4**, in that order, with the terms-v2 join first since it unblocks your
   `note-structured-basis` bundle soonest.
2. The published JSON Schemas before you extend intake.

### Suite status, stated precisely

Full run after these fixes: **1,452 passed, 1 failed.** The focused set you ran is now
**554** (from 529).

The one failure is
`test_worker_pool.py::TestWorkerPoolConcurrency::test_cross_tier_jobs_correct_and_concurrent`,
and it is **not** a code defect. It asserts that two jobs' wall-clock intervals genuinely
*overlap*; under full-suite load the OS can serialize them, so the timing assertion fails
while every correctness assertion in the same test — dtypes, NPV parity against a reference
run — passes. Verified three ways: it passes 3/3 in isolation, it passes against stashed
pre-fix code, and it is the same failure mode as **I-15** in a sibling test.

I'm reporting it rather than re-running until green, because a suite status that quietly
excludes its own failures is worth nothing to you. It's tracked as a follow-up.

The two `pydantic` failures I reported in v4 are **resolved** — that dependency is now
installed here.

**Still blocked on you:** D03/D04 for W2. The 13 `missingTerms` the SOFR refusal emits remain
that agenda verbatim.

**Not blocking anything:** the `.gitattributes` work. I still can't independently verify it
without a pushed commit SHA, but my side is protected by my own `.gitattributes` and a test
that asserts my fixtures are committed with LF.

---

## 7. One request

Your last two reviews have each found a real defect by trying an input I hadn't — a different
face amount, then an impossible date. Both were in code with what I considered thorough test
coverage, and in both cases my suite passed against the bug.

If you're willing, **keep doing exactly that**, and prefer inputs my fixtures don't contain.
The $100,000 face and the well-formed date were load-bearing assumptions I didn't know I was
making. I'd rather you find the next one than a production run.
