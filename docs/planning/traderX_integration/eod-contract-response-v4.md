# Reply to TraderX v3 — W0 shipped, SOFR refusal attached, and your source finding is worse than you thought

**Date:** 2026-09-15
**From:** Alex (JAX Risk Engine) · **To:** Yaakov (TraderX)
**Re:** `eod-response-to-alex-v3.md` (2026-09-15)
**Prior:** [proposal](eod-contract-proposal.md) → [v2](eod-contract-response-v2.md) → [v3](eod-contract-response-v3.md)
**Status:** W0 is **shipped** — named tests below. W1 is not started. Everything else is proposal.

---

## 0. Headline: your source-review finding is a live silent mispricing

You flagged this as "a source-level concern, not an executed reproduction." I executed it.
**You under-called it.** You predicted a missing validation; what's actually there is a
silent wrong number.

You asked me to test negative and out-of-range values with `compute_greeks` both true and
false. That 2×2 is exactly the right experiment, and it splits:

| Indices | `compute_greeks=False` | `compute_greeks=True` |
|---|---|---|
| `fwd=7` (out of range) | `IndexError` | `IndexError` |
| `fwd=-1` (negative) | **prices silently** ❌ | `ValueError`, named |
| `disc=-1` (negative) | **prices silently** ❌ | `ValueError`, named |

The out-of-range case crashes either way — ugly, but loud. **The negative case is the real
defect**, and only the `compute_greeks=False` path is exposed, which is the default and
the path an EOD batch actually takes.

Python's negative indexing wraps `-1` to the *last* curve. So the trade prices — cleanly,
finitely, with no warning — against a curve it was never booked against:

```
fwd_idx= 0 (booked, flat curve) : -5,913.9266
fwd_idx= 1 (steep curve)        : -5,857.0074
fwd_idx=-1 (INVALID)            : -5,857.0074   ← identical to curve 1

-1 silently priced as curve 1 : True
magnitude of silent error     : 56.92 USD on 2,000,000 notional
```

The number is small only because my test curves are close together. The mechanism has no
bound: it's "priced against an arbitrary other curve," and the error scales with how far
apart the two curves are.

What makes this the bad kind of bug is that [`_swap_curve_configs`](../../../engine/portfolio/request.py#L885)
has a docstring explicitly promising this cannot happen — it says a hard failure is
deliberate because silently substituting *some* curve "would produce a plausible-looking
sensitivity computed against a curve the trade was never booked against." That is a precise
description of what the base pricing path does two hundred lines earlier at
[request.py:840-841](../../../engine/portfolio/request.py#L840-L841), which indexes the list raw:

```python
disc_curve = market_config.rates.initial_zero_curves[cfg.discount_curve_index]
fwd_curve  = market_config.rates.initial_zero_curves[cfg.forward_curve_index]
```

Base pricing runs at [request.py:670](../../../engine/portfolio/request.py#L670); the validating
Greeks path runs at [request.py:691](../../../engine/portfolio/request.py#L691) — *after*, and only
when the flag is set.

**Why my 800-test suite missed it:** the existing coverage calls `_swap_curve_configs`
directly ([test_portfolio_gap_fixes.py:237](../../../tests/test_portfolio_gap_fixes.py#L237)) and
asserts it raises. It never routes a bad index through `price_portfolio`. The validator was
tested; the pricing path's *use* of it was not. A unit test on a guard proves nothing about
callers that skip the guard.

**Fix, W0.10, ahead of W1:** hoist validation to request admission — before any pricing,
independent of `compute_greeks` — and reject negatives explicitly rather than relying on
range arithmetic. Regression test is the 2×2 above driven through `price_portfolio`, not
through the helper.

This is the second real defect your review process has found in my engine, and the first one
that produces a wrong number rather than a failed verification. Worth noting what found it:
reading source and predicting a failure mode, then handing me the exact experiment. **Please
keep doing that.**

---

## 1. Your four implementations — accepted, with one I can't verify yet

| Item | Position |
|---|---|
| `.gitattributes` scoped to the fixture dir | **Accepted** — better than my proposal (§1.1) |
| `note-missing-accrual/` negative fixture | **Accepted**; two-boundary arrangement **confirmed** (§1.2) |
| `traderx.instrument-terms.v2` + `accrualBasis` | **Accepted**; field shape **confirmed, freeze it** (§1.3) |
| `synthetic` distinct from `assumed` | **Accepted** — already implemented, §3 |

### 1.1 The scoped `.gitattributes` is the right call — but it isn't in my checkout

Scoping to the fixture directory rather than my blunt repo-wide `*.json -text` is correct:
it pins exactly the hash-critical bytes and leaves unrelated CSV/JSON behavior alone. Your
point that hashed cut/preimage files need it too is one I'd missed.

**However — I cannot confirm any of it.** My `reference/traderX` submodule is still pinned at
`a102e498` / `bc30ed66`. There is no `.gitattributes` at that commit, and
`scripts/test-state-YU18-checkout.py` does not exist in my tree:

```
$ ls reference/traderX/.gitattributes
ls: cannot access '...': No such file or directory
```

So your "99 Python tests passed" and the Git checkout-filter proof are **your verification,
not a shared one**. I'm not doubting them; I'm flagging that neither of us should record them
as jointly confirmed until I can re-run them. **Please push, and tell me the commit SHA** —
I'll bump the submodule and independently re-run the CRLF proof the same way I independently
re-ran the golden hashes, rather than taking the pass on report.

Your note about restoring translated working files before updating attributes is well taken
and applies to me directly: my working tree is the CRLF-translated one from §0 of v3.

### 1.2 Two-boundary test arrangement — confirmed, exactly as you built it

Confirmed, and your framing is the one I want on record:

- **Your side:** blanked accrual on a coupon-bearing bond is **invalid input, rejected before
  bundle publication.** Your valid-extract rule is unchanged.
- **My side:** the same bytes are a **negative test** for my mapper —
  `unavailable` / `ACCRUED_NOT_SUPPLIED`, never structural zero.

That the fixture has no manifest and no completion receipt is the right design: it's mutated
raw input, and it must not be mistakable for a publishable bundle. **I am not asking you to
make missing accrual publishable.** If that ever becomes necessary it's a separate,
explicitly negotiated change to your validation contract — not something to smuggle in
through a test fixture.

**Your correction accepted, and my v3 §3.3 was wrong.** I wrote that v1 bill and note rows
with blank accrued are "indistinguishable." They aren't — v1 rows already carry coupon and
maturity. My claim confused *"terms artifact absent"* with *"no information available,"*
which your CSV disproves.

The rule I implemented is nevertheless the strict one — **require terms v2 before
normalizing accrual, even though v1 coupon/maturity would often be sufficient.** I'm keeping
that as deliberate worker policy, per your "fine as an explicit worker policy," because
inferring from coupon≠0 is exactly the inference-from-partial-data habit this contract
exists to prevent. Noting plainly that this is *policy*, not *necessity* — the v1 fields
would work; I'm declining to use them.

### 1.3 `accrualBasis` — freeze it, with one question about future-proofing

The field shape is right and **I'm freezing my consumer against it.** Specifically good:
`dateBasis: SESSION_DATE` makes the exporter's date convention explicit rather than
conventional, and `fractionDecimals: 6` + `rounding: HALF_EVEN` is what makes §2's tolerance
derivable rather than negotiated.

That validation rejects a conflict with the current NONE-calendar/unadjusted/same-day
fixture is the important part — it means a real-market settlement basis **cannot silently
inherit synthetic semantics.** That was my v3 §3.1 request and you've implemented it more
strictly than I asked.

Accepted without qualification: this describes the **exporter's** accrual basis and
rounding. It does not make synthetic reference assumptions into observed facts, and it does
not imply any pricing model supports them. My result vocabulary keeps those separate (§3).

**One question, non-blocking:** `settlementAdjustment: "NONE"` and `dateBasis:
"SESSION_DATE"` read as enums. When a real calendar arrives, does `accrual-basis.v1` gain
values, or does it become `accrual-basis.v2`? I'd prefer **a new schema version for any new
value**, so my allowlist can refuse an unrecognized basis rather than parse it optimistically.
If you'd rather add values in place, tell me and I'll pin the exact value set I accept and
refuse everything else.

---

## 2. Accrual reconciliation — your rounding correction is right, and it changes my number

**You are correct and my v3 §2.2 figure was wrong.** I reported +1,857.14 / −1,857.14 as
though it were *the* accrued value. It's the recomputed-unrounded figure. Reproducing both
paths with `HALF_EVEN`:

| Path | Long 100,000 | Short 100,000 |
|---|---|---|
| Exported fraction `0.018571` × signed face | `+1,857.10` | `−1,857.10` |
| Recomputed unrounded, then rounded to cents | `+1,857.14` | `−1,857.14` |
| **Difference** | **0.04** | **0.04** |

Your $0.05 bound is right: half an ulp at six decimals is `0.0000005 × 100,000 = $0.05`.
Observed 0.04, bounded 0.05.

**Labelling — my answer:** the returned value is **taken from the export**, labelled
`accrualSource: "exported-fraction"`. Rationale: your export is the authority on your
position, and recomputation would silently substitute my schedule interpretation for yours.
I'll carry `accrualSource: "recomputed-schedule"` as the alternative so the label is always
present rather than implied by a default.

**Tolerance rule, not a fixed number** — comparisons use half-ulp of the exported fraction
scaled by face, plus one cent for currency rounding:

```
tolerance = round(0.5 × 10^(−fractionDecimals) × |faceAmount|, 2) + 0.01
```

| Face | Tolerance |
|---|---|
| 100,000 | 0.06 USD |
| 1,000,000 | 0.51 USD |
| 5,000,000 | 2.51 USD |

`fractionDecimals` comes from `accrualBasis`, so if you ever export more precision the
tolerance tightens automatically instead of staying at a stale constant. **Agreed: the two
monetary values must never be compared as exact equals.**

**Signed face — confirmed, and already implemented that way.**
[`normalize.py:159-174`](../../../engine/integration/normalize.py#L159-L174) multiplies
`fraction × signed_face` in one step, and the docstring states there is no separate `sign()`
factor. The double-sign bug you warned about — which would flip a short to positive — is
structurally impossible here rather than merely avoided.

---

## 3. W0 is shipped — 295 integration tests, and the SOFR refusal is real

This is the deliverable you asked for. **It executes against your actual fixtures**, and
below is genuine output, not a schema sketch.

`engine/integration/` — nine modules, **295 tests passing in under a second.** It
deliberately **contains no pricing math and imports no pricer**, so contract bugs and pricing
bugs can never be debugged simultaneously.

### 3.1 The identified SOFR refusal (your W0 blocking item)

Against `sofr/v2`, all 13 missing terms echoed verbatim:

```jsonc
{
  "bundleId": "db954a3ce11b0cfe112c3ac9a081609dc2d8541fface10419bcb3bd982a44ab5",
  "clusterEpoch": "synthetic-shared-examples-v1",
  "engineVersion": "0.1.0",
  "itemOrder": {
    "scheme": "traderx-item-v1", "itemCount": 1,
    "itemIds": ["f81d34ed56ee436b86c726d9305fd984"],
    "sha256": "ad25220808a81a7cc0bf8496587eee8805dcd17d7422df63c9a130b26cbceb76"
  },
  "items": [{
    "itemId": "f81d34ed56ee436b86c726d9305fd984",
    "currency": "USD",
    "sourceIdentity": {
      "kind": "contract", "contractId": "SW-3",
      "clusterEpoch": "synthetic-shared-examples-v1", "accountId": "22214"
    },
    "refusal": {
      "reason": "CONVENTION_NOT_SUPPORTED",
      "missingTerms": ["businessDayAdjustment", "calendar",
        "fixedPaymentLagBusinessDays", "fixedSchedule", "fixingCalendar",
        "fixingHistoryReference", "floatingDayCount",
        "floatingPaymentLagBusinessDays", "floatingSchedule",
        "lockoutBusinessDays", "lookbackBusinessDays", "observationShift",
        "overnightCompounding"],
      "detail": "instrument terms are incomplete: the export enumerates 13 required term(s) it does not supply. An incompletely specified instrument is refused, not approximated from the terms that were supplied."
    }
  }]
}
```

**Your W0 identity requirement is met:** every refusal echoes bundle identity, `contractId`,
`clusterEpoch` and `accountId`. Array order is not load-bearing — and the hashed item-order
artifact is already in, rather than deferred to W1 as you allowed.

**v1 refuses too, with a different reason** — `TERMS_NOT_SUPPLIED` /
`"no instrument terms available for this row (NO_TERMS_ARTIFACT); conventions cannot be
established and will not be inferred."` Both refuse; the reasons are not interchangeable.

### 3.2 Coverage arithmetic, from that same run

```jsonc
"coverage": {
  "itemCount": 1,
  "byCalculation": {
    "npv":             {"ok":0,"unsupported":1,"unavailable":0,"failed":0,"notApplicable":0},
    "accruedInterest": {"ok":0,"unsupported":1,"unavailable":0,"failed":0,"notApplicable":0},
    "rateSensitivity": {"ok":0,"unsupported":1,"unavailable":0,"failed":0,"notApplicable":0},
    "vega":            {"ok":0,"unsupported":0,"unavailable":0,"failed":0,"notApplicable":1},
    "varEs":           {"ok":0,"unsupported":0,"unavailable":0,"failed":0,"notApplicable":1}
  },
  "allOutcomesAccountedFor": true,
  "allApplicableComputed": false
}
```

**This is the case your warning was about:** `allOutcomesAccountedFor: true` while *nothing
was priced*. It asserts statuses sum to `itemCount` — arithmetic only. `allApplicableComputed:
false` is the flag that carries financial meaning. **Please do recompute these from per-item
outcomes and reject inconsistent summaries**; I want that check adversarial, not trusting.

Agreed on non-applicability: `vega`/`varEs` here are `notApplicable` on a
capability/semantic basis — a single refused swap has no calibration basket and no scenario
cube — not to flatter coverage.

### 3.3 `synthetic` vs `assumed` — implemented, distinct

Confirmed and shipped. `provenance.origin` stays `synthetic` | `supplied` on your side; my
curve vocabulary keeps `inputOrigin` separate. Synthetic reference terms and an assumed
pricing curve never collapse into one value, and `supplied` is not treated as evidence of
market observation.

`GET /capabilities` reports `"deliveryStage": "W0"`, `"calculations.mode":
"refusal-only"`, and `conventions.swap.overnightCompounding: []` — an explicitly empty
list, so you can see SOFR is unsupported without inferring it from a refusal.

### 3.4 Named tests

| Component | Tests |
|---|---|
| Bundle load + hash verify (v1/v2) | `test_integration_bundle.py` |
| Terms join, `missingTerms` as refusal input | `test_integration_terms.py` |
| Normalization, signed face, accrual branches | `test_integration_normalize.py` |
| Convention allowlist + refusal | `test_integration_conventions.py` |
| Result shape + coverage arithmetic | `test_integration_result.py` |
| `itemId` + source identity | `test_integration_identity.py` |
| End-to-end against your fixtures | `test_integration_pipeline.py` |

**Caveat, deliberately:** this is transport, identity, and refusal. **Nothing is priced.**
Every calculation for every fixture is currently a refusal or a non-applicability. That's
W0's entire point, and no part of it should be read as progress on bill/note pricing.

---

## 4. Worker semantics — your four gaps

### 4.1 Publication vs pointer advancement

You found a real gap. My v3 §4 two-phase commit made the *manifest* the commit point, then
said the pointer advances after — without specifying the window between.

**Adopting the reconciliation approach.** The result manifest remains the commit point, and
discoverability no longer depends on the pointer: lookup falls back to a scan/reconcile over
published manifests when the pointer is behind, and advances it as a side effect. A crash
between publication and pointer advancement therefore loses no completed result — it costs
one scan on the next lookup. A published manifest is the durable record; the pointer is a
cache.

### 4.2 Unknown workload vs accepted-but-running — you're right, 404 is wrong

**Accepted without reservation**, including your framing: *a process-memory loss must not
automatically become a financial failure.* Collapsing "never heard of it" and "running" into
one 404 invites a duplicate overnight batch.

Distinct states, and lookup never returns a bare 404 for work that was accepted:

| State | Response |
|---|---|
| Never submitted | `404 UNKNOWN_WORKLOAD` |
| Accepted, running | `200 {"state": "running", "attemptId": ...}` |
| Accepted, failed | `200 {"state": "failed", "attemptId": ..., "reason": ...}` |
| Completed | `200 {"state": "completed", ...result...}` |

**Durability caveat, on the record:** the job table is still an in-process dict
(**I-08**, open). Until it's durable, a restart can still lose *running*-state knowledge —
which is precisely why §4.1's manifest scan matters: recovery goes through content-addressed
artifacts, never my process memory. The idempotent submission key in §4.3 is what closes the
remaining window, and I'd rather name that ordering than imply I-08 is fixed.

### 4.3 Fresh retries need submission identity — accepted

Your distinction is one I'd missed, and it's the right one:

- **`submissionId`** — caller-generated, idempotent. Retrying a lost submission response
  with the same `submissionId` **recovers the same attempt**, never starts a second one.
- **`reuseExistingResult: false`** — "don't serve me a cached result," *not* "make a new
  attempt every time you see this request."
- **A deliberate second benchmark repetition uses a new `submissionId`.** That's the caller
  stating intent explicitly rather than the worker inferring it from a retry.

Without this, a dropped response during a benchmark either silently duplicates expensive work
or silently returns the prior attempt — and you can't tell which from the outside.

### 4.4 Coverage — agreed, and please verify adversarially

Covered in §3.2. Your rule that `allOutcomesAccountedFor=true` must never be presented as
proof of successful pricing is exactly right, and §3.1 is a working demonstration of the
distinction.

---

## 5. Sequencing

**W0 — shipped**, §3. Plus one addition:

- **W0.10 (new, ahead of W1):** hoist curve-index validation to request admission, reject
  negatives explicitly, regression-test the 2×2 through `price_portfolio` (§0).

**W1 — not started.** Unchanged and correctly ordered:

1. **Day count as a per-instrument input** — `ore_builders.DAY_COUNTER` is still a module
   constant (`ORE.Actual365Fixed()`). No caller can supply ACT/ACT (ICMA). Prerequisite for
   everything below.
2. Bill pricer — single discounted cashflow.
3. Note pricer — schedule, accrual, clean/dirty reconciliation against `0.018571` at §2's
   tolerance.
4. Equity position pricer.

**W2 — faithful USD-SOFR builder.** Blocked on D03/D04.

**Unchanged from v3, and I want it to stay unambiguous:** priced bill/note results land at
the end of W1. **Not on receipt of the fixtures.** Your v3 already reflects this — "the
bill/note prices are acceptance targets after the new pricers exist" — and I'm restating it
so no schedule reads the shipped W0 as progress toward it.

---

## 6. What I need from you

1. **Push the `.gitattributes` / compatibility work and send the commit SHA** (§1.1). It's
   the only item blocking independent verification. My submodule is at `a102e498`.
2. **Answer the `accrual-basis` versioning question** (§1.3) — new values in place, or a new
   schema version? Affects whether my allowlist refuses or parses.
3. **Confirm `accrualSource: "exported-fraction"` is what you expect me to return** (§2).
4. **Review the W0 result shape in §3.1** before I freeze it — particularly whether
   `sourceIdentity` carries everything your validator needs to join back.

**Not blocking:** nothing. W0.10 and W1 item 1 start now.

---

## 7. State of play

**Verified this session, executed not asserted:**

- Your source-review finding reproduced, and it's a **silent mispricing** on the default
  `compute_greeks=False` path, not the missing-validation crash you predicted (§0).
- Your rounding correction confirmed to the cent — $0.04 observed against your $0.05 bound.
  **My v3 accrued figure was mislabelled** (§2).
- W0 shipped: **295 integration tests**, identified SOFR refusal against your real fixture
  with all 13 missing terms and full source identity (§3).

**Corrected in my own prior letters:** the v1 "indistinguishable rows" claim (§1.2) and the
accrued-interest value (§2). Both were mine; both were caught by you.

**Not verified, and not claimed:** your 99 tests and the Git checkout proof. Yours alone
until §6.1 lands.

**My suite is not green, and you should have the number rather than a rounded-up one.**
Full run: **1,138 passed, 3 failed** (900s). Plus `test_api.py`, which doesn't collect at
all — `ModuleNotFoundError: pydantic`, an uninstalled optional dependency, not a code fault.
None of the three touch the integration boundary or anything in §3, and all three are
test-side defects rather than engine regressions:

| Failure | Nature |
|---|---|
| `test_risk_var_es_override_recasts_npv_cube_for_risk_only` | Test asserts every risk metric is float32, but `ES_95_tailCount` is a **count** — correctly `int64`. Bad assertion, not a bad cast. |
| `test_same_tier_jobs_also_overlap_across_pool_workers` | Timing-sensitive concurrency assertion; observed zero wall-clock overlap. Flaky by construction. |
| `test_synthetic_arbitrary_shaped_cube` | **Passes in isolation, fails in the full run** — order-dependent, so shared state leaks between tests. |

I'm reporting these rather than quietly running the subset that's green. The third is the
one that actually bothers me: an order-dependent failure means some test mutates state
another test reads, and that's the kind of thing that hides a real defect behind a green
badge. All three are mine to fix and none block your §6 items.

**Not close:** priced bill/note. W1 items 1–3, gated on day count becoming a per-instrument
input.

One observation to close. Every number our systems have exchanged so far has been either a
refusal or a reconciliation of a convention — and two of the three defects found to date
were found by reading source and predicting a failure, not by running a suite. **1,138 of my
tests passed while §0 was sitting in the default pricing path** — not one of them caught it;
your read of a docstring did. Worth remembering when W1 starts producing numbers that look
plausible.
