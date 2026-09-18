# EOD contract — response v6

**From:** Alex (JAX Risk Engine side) · **Date:** 2026-09-16
**Re:** W1.6 delivered — the contract interface, in answer to your four open items
**Companion:** [Known Issues](../../known-issues.md) · [Integration Plan](traderx-integration-plan.md) · [Boundary reference](../../reference/eod-integration.md)

---

## 0. The short version

**All four of your open compatibility items are implemented.** Terms v2 with a validated
`accrualBasis`, versioned result and capability documents with machine-readable JSON Schema,
`accrualSource` aligned onto the standalone accrued outcome, and the EOD HTTP routes — which
also land W0.9's capability function (unreachable code since W0) and the *semantics* half of
W0.8's lookup.

**One caveat up front, so it isn't buried in §4:** the attempt store is **in-process**. The
four lookup states, the workload key, idempotent submission and attempt immutability are all
real; **durability is not**. A restart still loses running-state knowledge. Please do not
build a recovery path that assumes the lookup survives a process restart —
[I-08](../../known-issues.md#i-08) stays open, and §4 says exactly what is and is not there.

**One thing needs an answer from you**, and it is now load-bearing rather than
theoretical — §2.3.

**Two things found in my own review**, reported rather than left for you to find: a
`submissionId` reused across bundles could return **another bundle's priced numbers** (§5.1),
and a first implementation of the `accrualBasis` work passed **59 of 59 tests while being
wrong** (§5.2).

---

## 1. What you can act on immediately

### The service is reachable

```
GET  /eod/capabilities              the W0.9 document, finally routed
GET  /eod/schemas/result            machine-readable JSON Schema (Draft 2020-12)
GET  /eod/schemas/capabilities
POST /eod/price                     submit a bundle
GET  /eod/results/by-workload/{key} the W0.8 lookup, four states
GET  /eod/attempts/{attemptId}      one attempt, permanently addressable
```

A worked submission against the note fixture:

```jsonc
POST /eod/price
{"bundlePath": "…/note/v2",
 "marketInputs": {"mode": "assumed-profile", "assumedProfileId": "flat-3pct-v1"},
 "submissionId": "eod-2025-06-02-001"}

→ {"workloadKey": "sha256:7daed17b49a440beabef702f7306145d44625951185c13912dd9b1fd45e0dfb4",
   "attemptId": "b4863ccd-…", "state": "completed", "reused": false,
   "result": {"resultSchema": "jaxrisk.eod-result.v1", …}}
```

The numbers are unchanged from the ones you verified: **±103,308.33** dirty, **±1,857.10**
accrued, **∓15.28** at +1bp.

### Your validator can pin a version today

Every published document now carries its own:

```jsonc
{"resultSchema": "jaxrisk.eod-result.v1", …}
{"capabilitySchema": "jaxrisk.eod-capabilities.v1", …}
```

Emitted **before** I extend intake, per your own sequencing point — a version that was never
published cannot be pinned, so shipping the field while the value is still `.v1` is what
makes the *next* change safe rather than breaking.

They are **separately versioned on purpose**: a new pricer changes the capability document
without touching the result's shape. One shared version would force a lockstep neither of us
wants.

---

## 2. Terms v2 — accepted, with the strict reading

### 2.1 Terms version is independent of bundle version

As you specified. A `traderx.eod-bundle.v2` bundle may carry either `instrument-terms.v1` or
`.v2`, and both pairings are tested. `capabilities()` now advertises `termsSchemas` separately
from `bundleSchemas` so this is visible before you submit rather than inferred from a refusal.

### 2.2 `accrualBasis` is validated, and `fractionDecimals` now does real work

The field shape you froze is accepted as-is. What matters is that `fractionDecimals` is
**threaded into the reconciliation tolerance** rather than parsed and shelved — your
"tolerance tightens automatically instead of staying at a stale constant" is now literally
true:

| Declared decimals | Tolerance at 100,000 face | Note outcome |
|---|---|---|
| 4 | 5.01 | prices |
| **6** (your actual) | **0.06** | prices — the delivered result |
| 8 | 0.0105 | **refused**, `ACCRUAL_MISMATCH` |

At 8 declared decimals your own $0.04 rounding difference exceeds tolerance, so the note is
refused rather than priced. That is the correct behaviour and it is the observable proof the
field is wired through.

### 2.3 **The question I need answered — and it is now blocking-ish**

My v4 §1.3 asked: when a real calendar arrives, does `accrual-basis.v1` **gain values**, or
does it become `accrual-basis.v2`? That never came back.

**I have implemented the strict reading**, which is what my plan's settled row asked for:

- `dateBasis` ∈ `{SESSION_DATE}`
- `settlementAdjustment` ∈ `{NONE}`
- `rounding` ∈ `{HALF_EVEN}`
- schema ∈ `{traderx.accrual-basis.v1}`

**Anything else is refused**, including a future `accrual-basis.v2`. The reasoning is the one
this whole contract rests on: an unrecognized basis may describe an accrual computed to a
different date, and parsing it on v1 assumptions would silently reconcile against the wrong
one. Widening a tuple later is a one-line change; recovering from months of
optimistically-parsed wrong accruals is not.

**What I need from you:** if you intend to add values in place, say so and I will pin the
exact expanded set. If you intend a new schema version per change, nothing needs to happen —
that is what I have built for. **Either answer is fine; silence is the one that eventually
produces a refusal you did not expect.**

**I have registered my own uncertainty here as [I-23](../../known-issues.md#i-23)**, under a new
`ASSUMPTION` status created for it — my register previously had no way to record "the code is
working as designed, and the design rests on a premise nobody confirmed". The entry says
plainly that if you add values in place, **this engine will refuse bundles you consider
valid**, and that a `dateBasis`/`settlementAdjustment`/`rounding` refusal is not by itself
proof of a bad export. I would rather carry that as a named open assumption than let a
one-sided interpretation harden into an apparent agreement.

One related strictness you should know about: a **v1-labelled artifact carrying an
`accrualBasis`** is refused outright. The document has contradicted its own version marker,
and that marker is what every other parsing decision keys on.

---

## 3. `accrualSource` — aligned, with the distinction preserved

You asked for the standalone `accruedInterest` outcome to carry the same label as the NPV
payload. It does. What I want to be explicit about is what I **did not** collapse:

| Instrument | `provenance` | `accrualSource` |
|---|---|---|
| Note (your exported fraction, converted) | `converted` | `exported-fraction` |
| Bill (no coupon schedule) | `structural-zero` | **`structural-zero`** |

A bill's zero is not an exported fraction that happened to be zero. Labelling it
`exported-fraction` would erase the exact distinction the W0.3 rule exists to hold — the one
between a bill and a coupon-bearing note whose accrual was omitted. Folding them together
fails 4 tests, one of which asserts it negatively by name.

An `unavailable` accrual carries **no** label at all. There is no source, and a label would
imply one.

---

## 4. Transport semantics you'll want to code against

### A refusal is a `200`

| Condition | Status |
|---|---|
| Instrument refused (SOFR, equity) | **`200`** — the refusal is the answer |
| Bundle fails hash verification, **or is missing** | `422` |
| Terms artifact structurally unusable | `422` |
| Market inputs unresolvable | `400` |
| `submissionId` reused for a **different** workload | `409` — see §5.1 |
| Workload never submitted | `404` — the only one |

An HTTP error for a refusal would make *"we correctly declined to guess"* indistinguishable
from *"we broke"*. Note the second row: a **missing** bundle is a `422`, not a `404`, because
W0.1 already established that a missing artifact is an integrity failure rather than an
absence — the same rule that distinguishes your empty contracts file from a missing one.

### Four lookup states, and `submissionId`

| State | Response |
|---|---|
| Never submitted | `404 UNKNOWN_WORKLOAD` |
| Accepted, running | `200 {"state": "running"}` |
| Accepted, failed | `200 {"state": "failed", "reason": …}` |
| Completed | `200 {"state": "completed", "result": …}` |

Lookup returns the most recent **successful** attempt — a later failure never hides an earlier
success. Attempts are immutable once terminal.

`submissionId` is **not** part of the workload key, deliberately: it identifies a *request*,
the key identifies a *computation*. Retrying a lost response with the same id recovers the
same attempt; a deliberate repeat uses a new one. `reuseExistingResult: false` still means
"don't serve me a cache" rather than "always start a new attempt".

### What is still not durable — [I-08](../../known-issues.md#i-08)

The attempt store is **in-process**. A restart still loses running-state knowledge. What
landed is the state machine and the key; the manifest-scan recovery in plan §W0.8 needs a
persistent artifact store that does not exist yet. I'd rather say that plainly than let the
four-state lookup imply durability it does not have.

---

## 5. Two things found in my own review, reported rather than buried

### 5.1 A `submissionId` could return another bundle's numbers

Found reviewing W1.6.4 before it shipped. `AttemptStore.start` honoured **any** repeated
`submissionId` without checking the workload matched:

```
submit(bundle=note, submissionId="dup")   → 103,308.33
submit(bundle=bill, submissionId="dup")   → 103,308.33   ← the NOTE's number
```

A bill submission came back carrying the note's priced result, under a bundle id that did not
produce it. **This is exactly the silently-wrong-number failure this contract exists to
prevent**, arrived at through the idempotency path rather than through a pricer — and it
would have reconciled against nothing on your side, because the number was real, just for a
different instrument.

**Now a `409 SUBMISSION_ID_CONFLICT`.** Idempotency means *"this exact request, again"*; it
cannot mean *"whatever I sent last time under this name"*. Four regression tests verified to
fail against the pre-fix code, plus a fifth that passes both ways by design — confirming the
guard did not break the retry-a-lost-response case it sits next to.

**What this means for your coordinator:** a new `submissionId` per distinct workload. Reusing
one for a retry of the *same* submission is still correct and still recovers the same
attempt.

### 5.2 Something my own testing got wrong

Working rule 3 says regression tests must be verified to fail against the wrong
implementation. I ran six wrong implementations against the W1.6 tests. Five were caught.

**One was not**, and it is the interesting one: an implementation that parsed `accrualBasis`,
validated every enum, refused every bad value — and then **never used `fractionDecimals`**
passed **59 of 59 tests**.

Every test of the *parser* still held. What nothing asserted was that the value reached the
tolerance it exists to derive. That is a bug of **omission**, and it produces no wrong output
anywhere a parser test can look.

The fix was a test driving an *end-to-end consequence* — 8 declared decimals must refuse the
note with `ACCRUAL_MISMATCH` — verified to fail against that implementation. §2.2's table is
that test.

I'm flagging it because it is the same shape as the defects you found in my source: the suite
was green and the code was wrong. Your source reviews have caught two real bugs that my tests
did not; this one I caught only because I went looking for it deliberately.

---

## 6. Open items

**On your side:**

1. **The `accrual-basis` versioning question** (§2.3) — now load-bearing.
2. **Push the compatibility work and send the commit SHA.** My submodule is still pinned at
   `a102e498`, which has no `.gitattributes` and no `scripts/test-state-YU18-checkout.py`.
   Your 99 tests and the checkout-filter proof remain *your* verification rather than a shared
   one until I can re-run them.
3. **Review the result shape against the published schema.** It is now machine-readable at
   `GET /eod/schemas/result` — I would rather you find a gap against the schema than against
   a document.

**On mine:**

- **W1.5**, the internal wire-through — no effect on this interface.
- **W2 / USD-SOFR** still blocked on D03/D04. The 13-term refusal is unchanged.
- Equity valuation still blocked on a spot/FX source ([I-18](../../known-issues.md#i-18)).

**Unchanged and worth restating:** nothing here priced anything new. W1.6 is contract surface
over the same two Treasury pricers you already verified.
