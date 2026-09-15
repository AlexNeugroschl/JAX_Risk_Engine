# Reply to TraderX v2 — package received, hashes verified, schema diff

**Date:** 2026-09-15
**From:** Alex (JAX Risk Engine) · **To:** Yaakov (TraderX)
**Re:** `eod-response-to-alex-v2.md` (2026-09-15)
**Prior:** [proposal](eod-contract-proposal.md) → [response v2](eod-contract-response-v2.md)
**Status:** Proposal. Nothing is implemented unless marked **shipped** with a named test.

---

## 0. Package received — deliverable #3 is done

Your commits are in my checkout (`a102e498`, `bc30ed66`) and I've verified YU18 end to end.

**All 14 golden-vector hash checks verify independently**, computed with my own stdlib
implementation of your canonicalization rather than by running your verifier:

| Case | Check | Value | Match |
|---|---|---|---|
| basic | `manifestFileSha256` | `c3aa19bb8a9f1d86…9b9f7674` | ✅ |
| basic | `bundleId` | `9a0cb99550dd2e18…ec7c5d36` | ✅ |
| basic | positions / contracts | `659086e4…` / `d542657b…` | ✅ |
| basic | workload local / http | `e1be064d…` / `2330fab4…` | ✅ |
| exchange | `manifestFileSha256` | `13228b65a0af01d1…4137111c` | ✅ |
| exchange | `bundleId` | `de73a94990867ef8…f885d5288` | ✅ |
| exchange | positions / contracts | `07416900…` / `d542657b…` | ✅ |
| exchange | workload local / http | `ed041040…` / `14706eb6…` | ✅ |

Canonical form confirmed: sorted keys, two-space indent, ASCII escaping, one trailing
newline; `bundleId` over the manifest body **excluding** `bundleId`; manifest-file hash over
the final bytes **including** it. The two hash scopes are distinct and both reproduce.
**No preimage disagreements — nothing to report under your "or the exact preimage where they
differ."**

### One finding worth acting on: CRLF silently breaks your vectors

Running your `verify_golden.py` against my working tree **fails immediately**:

```
AssertionError  at:  assert raw == encoded(manifest)
```

Cause is not your spec. `reference/traderX` has `core.autocrlf=true` and **no
`.gitattributes`**, so git rewrote the committed LF bytes to CRLF on checkout. The blob is
clean (`{\n` = `7b0a`); my working tree had `{\r\n`. My verification above reads
`git show HEAD:<path>` blobs, which bypasses the smudge filter — that's why it passes.

**To be unambiguous that this is not a defect in your work:** I exported the golden directory
at pristine committed bytes and ran your `verify_golden.py` **unmodified** against it:

```
$ python verify_golden.py <pristine-LF-export>
PASS: 2 fixed v1 golden cases; both workload profiles verified     (exit 0)
```

Your verifier and vectors are correct. Only the checkout translation breaks them.

**This is precisely the silent canonicalization divergence I asked to catch in W0, and it
found a real one on contact.** Every hash in the package is unverifiable from a default
Windows checkout, and the failure mode is a bare `AssertionError` that looks like a bad
vector rather than a line-ending problem.

**Recommended fix on your side** — add to the repo root:

```gitattributes
*.json  -text
*.csv   -text
```

That pins hash-critical artifacts to byte-exact checkout on every platform. I'd rather this
live in your repo than in each consumer's local git config, since any future consumer hits it
identically. Worth a line in `golden-v1.md` too: *hashes are over committed bytes; verify with
line-ending translation disabled.*

---

## 1. Your six answers — accepted

| # | Your answer | My position |
|---|---|---|
| 1 | `unsupported` + `CONVENTION_NOT_SUPPORTED` even if it covers every OTC row | **Accepted.** Your point that the mock prices nothing today, so this is a step forward not a regression, resolves my concern |
| 2 | New `traderx.eod-bundle.v2` + hash-pinned `instrument-terms.json` | **Accepted**; consumption path in §3.1 |
| 3 | Bill first, then note, same wave, separate acceptance checks | **Accepted** — exactly the split I asked for |
| 4 | Accrued signed for the position | **Accepted**, and independently cross-validated — see §2.2 |
| 5 | Deterministic lookup + immutable per-attempt results | **Accepted**; semantics in §4 |
| 6 | D03/D04 next, no date until terms agreed | **Accepted.** Correct not to guess a date |

Also accepted: one stable calculation name, `ad-first-order` / `bumped-revaluation` method
values, scaled-second-derivative gamma excluding cross terms, five-value coverage status,
per-curve provenance plus top-level indicator, your workload-identity additions, the hashed
item-order artifact, and source identity on **unsupported rows too**.

---

## 2. Your package answered two of my open questions

### 2.1 `closingMark` is clean — question withdrawn

I asked you to confirm this. Your terms artifact already states it:
`"priceBasis": "clean-fraction-of-par"`, alongside
`"quantityUnit": "signed-currency-face"`. Both my §2.4 questions are answered by the data.
**Withdrawn — no reply needed.**

This is the right pattern: the convention travels as a field rather than as an agreement we
each remember separately.

### 2.2 Note fixture — representable, and our accrued interest already agrees

You asked me to confirm my builder can represent the exact fixture. **Yes**, and I verified
rather than asserting it.

**Schedule** — ORE reproduces your four periods exactly:

```
ORE.Schedule(2024-12-15 → 2026-12-15, 6M, NullCalendar, Unadjusted, Backward)
  → 2024-12-15, 2025-06-15, 2025-12-15, 2026-06-15, 2026-12-15
MATCHES FIXTURE: True
```

**Accrued interest — independent cross-validation.** Computing ACT/ACT (ICMA) from your terms
at valuation date 2025-06-02, with no reference to your exported value:

| | |
|---|---|
| Accrual fraction (2024-12-15 → 2025-06-02) | `0.4642857143` |
| × 4% coupon → fraction of par | **`0.0185714286`** |
| Your exported `accruedInterestFraction` | **`0.018571`** |
| **Agreement at exported precision** | ✅ |

Signed per position, per your §1.4: long 100,000 face → **+1,857.14 USD**; short →
**−1,857.14 USD**.

**This is the first number our two systems have independently agreed on.** It's a small one,
but it validates the accrual basis, the day-count interpretation, the schedule, and the
sign convention simultaneously — which is more than a schema check does.

**One caveat, deliberately:** this is ORE-via-Python agreeing with your exporter. It is
**not** my engine pricing your bond — see §3.2. I don't want the agreement above read as more
than it is.

---

## 3. Corrections and the schema diff (your #1 and #2)

### 3.1 How my worker consumes the v2 terms artifact

Agreed: pinned artifact, never a mutable `latest`. My adapter will:

1. Verify `instrument-terms.json` against the **manifest's own hash entry** before parsing —
   the terms file is hash-pinned exactly like `positions.csv`/`contracts.csv`.
2. Join **securities by security identity** (shared across long/short accounts) and **OTC by
   `contractId` + `clusterEpoch`**, per your §1.2.
3. Treat a non-empty **`missingTerms`** as an authoritative refusal input, not a hint: any
   entry with a non-empty list yields `unsupported` for every calculation requiring those
   terms, listing them verbatim.
4. Include the terms-file hash in the workload key, so a changed terms file is a changed
   workload. (You already made this true at bundle level; I'm making it explicit at mine.)

**Field-name changes requested: none.** Your terms vocabulary maps cleanly onto what I asked
for in the proposal. Two additions I'd like, both small:

- **`accrualBasisNote`** (or reuse `provenance.description`) on the *security* entry, so the
  "calendar NONE / unadjusted / same-day settlement matches exporter session-date accrual
  basis" caveat is machine-readable rather than prose. When real reference data arrives with a
  real calendar, I want a structural signal that these values changed meaning.
- **`origin: "synthetic"`** promoted to a first-class enum alongside `observed`/`assumed` in
  my curve-provenance vocabulary (§2.6 of my v2), so synthetic reference terms and assumed
  market inputs stay distinguishable in the result. They're different kinds of "not real".

### 3.2 Correction to your §3 acceptance table

Your table lists, for the bill: *"Identified NPV and method-tagged rate sensitivity."* I need
to be direct that **I cannot return that on receipt of the fixtures.**

**My engine has no bond pricer.** `engine/instruments/` contains exactly four modules, all
rate derivatives (swap, European/Bermudan/American swaption). No bill pricer, no note pricer,
no equity position pricer. That's **I-07** in [Known Issues](../known-issues.md).

There's a second, more specific blocker I found while checking your fixture:

```python
# engine/models/ore_builders.py:55
DAY_COUNTER = ORE.Actual365Fixed()   # module constant, applied to both legs of everything
```

**There is currently no path by which a caller supplies ACT/ACT (ICMA).** Making day count a
per-instrument input is a prerequisite for the bond work, not part of it. Sequencing in §5.

I'm flagging this because your table reads as something I could hit on delivery. The ORE
verification in §2.2 proves the *conventions are representable*; it does not mean the engine
prices them.

### 3.3 Zero-coupon normalization — your fixture confirms the hazard

Your exporter detail was right, and the bill fixture demonstrates it concretely:

```
UST-BILL-20251202 ... coupon=0, lastCouponDate=<blank>, accruedInterestFraction=<blank>
UST-NOTE-20261215 ... coupon=4.0, lastCouponDate=2024-12-15, accruedInterestFraction=0.018571
```

A naive `blank → 0.0` is **correct** for the bill and **silently wrong** for a coupon-bearing
note whose accrued is blank because it's *missing*. Same bytes, opposite meanings.

**Good news:** v2 disambiguates it — the bill's terms carry `couponFrequency: "NONE"`,
`dayCount: "NOT_APPLICABLE"`, `schedule: []`. So the rule keys on **instrument terms, never
on the blank**:

| Case | Rule |
|---|---|
| Terms say zero-coupon (`couponFrequency: NONE`), accrued blank | `accruedInterest = 0.0`, `provenance: "structural-zero"` |
| Terms say coupon-bearing, accrued blank | **`status: "unavailable"`, `reason: "ACCRUED_NOT_SUPPLIED"`** — never `0.0` |
| Terms say coupon-bearing, accrued present | `fraction × faceAmount × sign(position)` |
| **Terms absent entirely** | `unavailable` — the blank is uninterpretable without them |

The last row matters for v1 bundles, which have no terms artifact: a v1 bill and a v1 note
with blank accrued are **indistinguishable**, so neither gets `0.0`.

**Fixture request stands:** a coupon-bearing note with `accruedInterestFraction` deliberately
blank, expecting `unavailable`. Your current note has it populated, so nothing exercises the
wrong-branch case — and it's the one that would pass a long-only, bill-only test suite while
being wrong.

### 3.4 `pv01` vs `rateSensitivity` — my inconsistency, corrected

You're right that I mixed them. **Settling on `rateSensitivity`**, method/factor/bump in the
payload. `pv01` is withdrawn: it implies a unit and a method in the name, the same mistake as
`pvChangeForPlus1bp`.

Frozen calculation names: `npv`, `accruedInterest`, `rateSensitivity`, `rateGamma`, `theta`,
`vega`, `varEs`.

### 3.5 Coverage counts — explicit arithmetic

Accepting your "accounted for" vs "applicable computed" distinction:

```jsonc
"coverage": {
  "byCalculation": {
    "npv":             {"ok": 211, "unsupported": 3, "unavailable": 0, "failed": 0, "notApplicable": 0},
    "rateSensitivity": {"ok": 208, "unsupported": 3, "unavailable": 3, "failed": 0, "notApplicable": 0},
    "vega":            {"ok": 12,  "unsupported": 0, "unavailable": 0, "failed": 0, "notApplicable": 202}
  },
  "allOutcomesAccountedFor": true,
  "allApplicableComputed": false
}
```

`allOutcomesAccountedFor` asserts statuses sum to `itemCount` per requested calculation — a
structural check you can verify independently, `true` even when everything failed.
`allApplicableComputed` is `true` only when `unsupported`/`unavailable`/`failed` are all zero;
`notApplicable` never counts against it.

Requested vega without a basket → `unavailable` + `CALIBRATION_BASKET_NOT_SUPPLIED`, never
silently omitted. **Cross-currency:** per-currency aggregates only, with
`reportingCurrencyConversion: "NOT_APPLIED"`, until a conversion policy exists.

### 3.6 Assumed curves must be *requested*, never fallen back to

Your correction accepted. `MARKET_PACKAGE_NOT_SUPPLIED` does **not** trigger an assumed curve:

```jsonc
"marketInputs": {"mode": "assumed-profile", "assumedProfileId": "flat-3pct-v1"}
```

Absent `marketInputs`, or `mode: "package"` with no package → job **fails** with
`MARKET_INPUTS_NOT_SUPPLIED`. No curve is ever substituted.

Your origin/transformation split accepted — sharper than my single field:

```jsonc
{"curveId": "USD-DISC", "inputOrigin": "assumed", "construction": "bootstrap-v2",
 "inputHashes": ["sha256:..."], "assumedProfileId": "flat-3pct-v1"}
```

`derived` dropped — it said nothing about whether inputs were observed. `mixed` retained for a
curve bootstrapped from partly observed, partly assumed pillars.

**Disclosure:** my `ZeroCurveConfig` is bare `times`/`rates` with **no metadata field at
all** — assumed and observed curves are currently indistinguishable inside my engine.
Carrying `inputOrigin` end-to-end is W0 work, not a field I can echo today.

---

## 4. Durable lookup and attempt semantics (your #5)

**Which attempt does lookup return?** The **most recent successfully completed** attempt for
that workload key. Never failed, never partial, never in-flight.

**Second attempt must not overwrite the first.** Each attempt writes to its own immutable path
keyed by `attemptId`. The workload lookup is a **pointer**, advanced only after artifacts are
fully written and verified. Prior attempts stay addressable by `attemptId` permanently.

**Interrupted write must not become discoverable.** Two-phase: write artifacts to a temporary
path → verify hashes → atomically publish the result manifest → then advance the pointer.
**The result manifest is the commit point.** Crash before it → orphaned artifacts no lookup
can reach. Crash after → complete, discoverable result. No window where a partial write is
reachable.

**Benchmark fresh attempt:**

```jsonc
{"execution": {"reuseExistingResult": false}}
```

Forces a new `attemptId` and fresh computation, bypassing lookup, **without overwriting**.
Published alongside; pointer advances only on success. Agreed hardware comparisons must run
fresh attempts and record the actual device used.

**Caveat on the record:** my dispatcher's job table is still an in-process dict (**I-08**).
The durable lookup above is what makes that acceptable — recovery goes through
content-addressed artifacts, not my process memory. A lost in-memory job is an
**infrastructure** event, never a financial failure; retry against identical immutable inputs
is safe.

---

## 5. What I'll deliver, in order

**W0 — unblocked, starting now:**

| Item | Detail |
|---|---|
| Request/result/capability schema + fixtures | §3.4 names, §3.5 coverage, §3.6 market inputs |
| v2 terms consumption | §3.1 — hash-verify, join, `missingTerms` as refusal input |
| Convention allowlist + refusal path | Field-level `CONVENTION_NOT_SUPPORTED` before any pricing object |
| **SOFR `unsupported` result (#4b)** | Against your `sofr` case, echoing all 13 `missingTerms` |
| `GET /capabilities` | Supported (product × convention × calculation) matrix |
| Adapter mapping + `mappingVersion` | §3.3 rules; long **and** short tested |
| Durable result lookup | §4 semantics |
| `inputOrigin` through curve config | Currently absent entirely |

The SOFR refusal is genuinely near-term: it requires no pricer, only the terms join and the
allowlist. I expect that to be the **first real result I return you.**

**W1 — the build work:**

1. **Day count as a per-instrument input** (removes `ore_builders.DAY_COUNTER` as a constant)
   — prerequisite for everything below.
2. Bill pricer — single discounted cashflow; transport/identity smoke test.
3. Note pricer — schedule generation, accrual, clean/dirty reconciliation against your
   `0.018571`.
4. Equity position pricer.
5. Identity plumbing — `itemId`, source identity, hashed item-order artifact (**I-10**).

Each with ORE parity at matched terms. **Priced bill/note results (#4a) land at the end of
this, not on receipt of fixtures.**

**W2 — faithful USD-SOFR builder.** Blocked on D03/D04. A separate builder alongside the
generic one, which stays untouched (every swaption pricer depends on its ACT/365 consistency
with my simulation's time axis).

**Already shipped** — verified this session, **824 tests passing**:

| Item | Test |
|---|---|
| Per-instrument NPV; total is the sum of parts | `TestPerTradeBaseNpv` |
| Swap Delta/Gamma/Theta through the portfolio path | `TestSwapGreeksReachThePortfolioPath` |
| Bermudan vega for calibrated sigma | `TestBermudanVegaReachesThePortfolioPath` |
| Aged-swap approximation warned, not silent | `TestAgedSwapWarningIsNotSilent` |

Aged-swap is **flagged, not fixed** — inaccuracy unchanged, needs fixing history to close.
Full register: [Known Issues](../known-issues.md).

---

## 6. What I need from you

**Blocking: nothing.** The package unblocked everything; §5's W0 work starts now.

**Wanted, non-blocking:**

1. **`.gitattributes` with `*.json -text` / `*.csv -text`** (§0) — highest value, smallest
   change. Without it your vectors are unverifiable from a default Windows checkout.
2. **Blank-accrued coupon-bearing note fixture** (§3.3) — expects `unavailable`, not `0.0`.
3. **Machine-readable accrual-basis caveat** on security terms (§3.1).
4. Confirm `origin: "synthetic"` as a distinct value from `assumed` (§3.1).

**Answered by your package, withdrawn:** `closingMark` clean vs dirty; `faceDenomination` /
`signedFaceAmount` split. Both settled by fields already in the artifact.

---

## 7. State of play

**Verified working:** all 14 golden hashes reproduce independently; ORE reproduces your note
schedule exactly; our accrued-interest figures agree to exported precision.

**Found:** CRLF breaks your own verifier on a default Windows checkout (§0).

**Agreed and settled:** ownership, dispatch, refusal semantics, units, coverage model, curve
provenance, workload identity, attempt/lookup semantics, W1 scope, terms consumption.

**Next real result from me:** the SOFR `CONVENTION_NOT_SUPPORTED` output — no pricer needed,
so it comes out of W0.

**Not close:** priced bill/note. That needs W0 plus W1 items 1–3, gated on making day count a
per-instrument input. Your stated end-to-end target — *one exported Treasury case consumed and
returned as a validated identified financial result* — is that whole path. I'd rather name it
honestly than let the fixtures' arrival imply it's near.
