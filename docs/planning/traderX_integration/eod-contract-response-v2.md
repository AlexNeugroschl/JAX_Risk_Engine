# Reply to TraderX on the EOD contract response

**Date:** 2026-09-15
**From:** Alex (JAX Risk Engine) · **To:** Yaakov (TraderX)
**Re:** `eod-response-to-alex.md` (2026-09-14), replying to
[EOD Contract Proposal](eod-contract-proposal.md)
**Status:** Proposal for agreement. Nothing below is implemented unless explicitly marked
"shipped", and shipped items name the test that proves it.

---

## 0. Summary

**Accepted as written:** dispatch ownership (§4), the local-fixtures-first acceptance package
(§5), per-calculation coverage, identity echo requirements, the unit corrections, and the
ownership split. You were right on every correction you raised, including two where my
proposal was **wrong rather than merely unclear** — the `pvChangeForPlus1bp` field name
(§2.1) and the gamma convention (§2.2). Both are corrected below.

**Pushing back on two things:**

- **Treasury bills before fixed-rate notes (§3.1).** I'd rather do a short fixed-rate note
  first, or run both together. A bill exercises almost none of the machinery that the rest
  of the roadmap depends on, so "bill first" front-loads the easy half and defers every
  question that actually blocks W2.
- **"Attempt state may remain in memory initially" (§4.2).** I want to accept this *less*
  than you've offered. Deterministic result lookup by workload identity is cheap and I'd
  rather build it in W0 than retrofit durability later.

**One thing I need to flag that your response assumes and I cannot yet deliver:** a
"USD-SOFR booking before W2 support → explicit unsupported row" fixture (your §5) **cannot be
satisfied by my current code**, and the failure mode is silent. Details in §5.1 — this is the
single most important item in this document.

---

## 1. Accepted without modification

Recording these so they're settled and we don't relitigate them:

| Your §  | Item | My position |
|---|---|---|
| §2 | Ownership table | Accepted exactly as written |
| §3 | Return opaque result-item ID **and** source identity (kind + `accountId` + `security`/`contractId`) | Accepted; see §3.3 |
| §3 | Echo market package hash, reference-data identity, cut identity, input valuation time | Accepted |
| §3 | Session date / valuation time / completion time are three separate fields | Accepted — this is the same four-clocks separation your architecture pack already requires |
| §3 | EOD valuation time stays fixed even if the worker finishes next morning | Accepted, and it falls out of the design: valuation time is an *input*, never `now()` |
| §3 | Per-calculation coverage, not just per-instrument | Accepted; see §2.4 |
| §3 | Explicitly requested assumed curve OK; silent synthetic replacement not | Accepted; see §2.5 |
| §4 | Single dispatch owner; TraderX coordinator discovers and submits | Accepted. My worker will **not** independently discover bundles |
| §4 | Polling sufficient initially; notifications are accelerators | Accepted |
| §4 | Workload key separate from attempt ID; precision change = new workload | Accepted; see §4.2 |
| §4 | New manifest version for market/reference additions; separate result schema | Accepted |
| §4 | Name the hash-byte definitions separately (`bundleId` vs manifest-file hash) | Accepted — see §4.3, I want to go further |
| §5 | Local files/HTTP first, GCS transport after | Accepted |
| §6 | The three-way ownership split of next steps | Accepted |

---

## 2. Corrections — you were right

### 2.1 The sensitivity field name was wrong, not just ambiguous

You caught a real error. My proposal named the field `pvChangeForPlus1bp`, which reads as
`PV(r+0.0001) − PV(r)` — an actual bumped revaluation. **My implementation produces the
first-order estimate**, confirmed in
[`engine/risk/greeks.py`](../../../engine/risk/greeks.py):

```python
grad_disc, grad_fwd = jax.grad(price_fn, argnums=(0, 1))(...)
return {"discount_delta": grad_disc * bump_size, ...}   # bump_size = 0.0001
```

That is `dPV/dr × 0.0001`. The name promised something the code does not do.

**Accepted. New field shape**, with the method carried in the payload rather than implied by
a name:

```jsonc
{
  "name": "rateSensitivity",
  "method": "analytic-first-order",       // | "bumped-revaluation"
  "derivative": "dPV/dr",
  "shockedFactor": {"curveId": "USD-SOFR-DISC", "pillarDate": "2027-09-13"},
  "bump": {"type": "absolute", "size": 0.0001, "unit": "decimal-rate"},
  "value": -41.88,
  "currency": "USD"
}
```

`method` is populated from what actually ran, never from configuration intent. If I later add
a true bumped revaluation it appears as `bumped-revaluation` on the same field, and the two
are then directly comparable.

**Worth stating plainly:** for a linear swap these agree to well below any tolerance we'd
set. For a Bermudan near its exercise boundary they do **not**, and the first-order number is
the less meaningful one. So the label is load-bearing exactly where the instrument is most
interesting — which is why I'm glad you pushed.

### 2.2 Gamma — scaled second derivative, no one-half factor

Confirmed in the same file:

```python
"discount_gamma": jnp.diagonal(hess_disc_full) * bump_size ** 2
```

So it is **`d²PV/dr² × (0.0001)²`** — the scaled second derivative, **not** the second-order
P&L contribution (no `½`). Consumer applying it to a shock of size `k` bumps:
`ΔPV ≈ delta·k + ½·gamma·k²`. **The one-half is the consumer's to apply; I will not fold it
in.**

Two further points I should have stated in the original proposal:

- It is the **diagonal** of the Hessian only — same-pillar curvature, no cross-pillar terms.
  This matches ORE's `SensitivityCube::gamma`, which is a cross-scenario second difference
  and so only ever reports the same-pillar term. **Cross-gamma is not available**, and
  summing diagonal gammas is not a portfolio convexity.
- Field will carry `"convention": "scaled-second-derivative"` and
  `"crossTerms": "excluded"` explicitly.

### 2.3 Normal vega — already a numeric bump

Agreed that "one volatility point" is too loose. The implementation already takes a numeric
`market_vol_bump: float = 0.0001` (absolute normal/Bachelier vol), so this is a documentation
fix, not a code change. The field will carry the same `bump` object as §2.1:
`{"type": "absolute", "size": 0.0001, "unit": "normal-vol-decimal"}`.

One scope limit to record now rather than surprise you at W3: vega is only well-defined for a
Bermudan/American whose sigma was **calibrated from a supplied basket**. A flat hand-set sigma
has no market quote to be sensitive to, so vega is **omitted** for those trades rather than
fabricated. (Shipped and tested:
`tests/test_portfolio_gap_fixes.py::TestBermudanVegaReachesThePortfolioPath`.)

### 2.4 Per-calculation coverage — accepted, and it changes my result shape

You're right that per-instrument coverage is insufficient, and your example is exactly the
realistic case: a row can have a valid NPV while theta is unavailable. My §6.4 shape implied
one status per row. Corrected:

```jsonc
{
  "itemId": "r-0001",
  "source": {"kind": "otc", "accountId": "ACC-7", "contractId": "SW-00412"},
  "calculations": {
    "npv":   {"status": "ok", "value": -184203.11, "currency": "USD"},
    "pv01":  {"status": "ok", "value": -41.88, "currency": "USD"},
    "theta": {"status": "unavailable", "reason": "AGED_SWAP_FIXINGS_NOT_SUPPLIED"},
    "vega":  {"status": "not-applicable", "reason": "NO_OPTIONALITY"}
  }
}
```

`not-applicable` and `unavailable` are deliberately distinct: vega on a vanilla swap is not a
gap in coverage, but theta blocked on missing fixings is. Conflating them would make a
coverage percentage meaningless.

**Aggregates carry the covered subset explicitly:**

```jsonc
{
  "scope": {"accountId": "ACC-7"},
  "calculation": "npv",
  "value": 1204881.22, "currency": "USD",
  "coveredItemCount": 211, "totalItemCount": 214,
  "excludedItems": [{"itemId": "r-0207", "reason": "CORPORATE_SPREAD_MODEL_NOT_IMPLEMENTED"}],
  "materialWarnings": ["AGED_SWAP_APPROXIMATION"],
  "complete": false
}
```

**`complete: false` is mandatory whenever `coveredItemCount < totalItemCount`.** A partially
covered aggregate must be structurally distinguishable from a complete one — not merely
accompanied by a warning a UI might not render.

### 2.5 Units — accepted, with the conversion tested in both directions

All three accepted:

- **Bond coupon is annual percent** (`4.5` → `0.045`). My adapter converts; your bytes stay
  untouched.
- **`accruedInterestFraction` is a fraction of par**, not currency.
- **Normalization lives in my receiving adapter, with a versioned mapping.** Agreed —
  `mappingVersion` will be echoed in result provenance so a number is always traceable to the
  conversion that produced it.

**Answering your direct question on accrued-interest sign:** I propose accrued interest be
returned **signed for the position**, consistent with NPV. A short bond position has negative
accrued. Rationale: every other monetary field in the result is signed to the account, and a
positive-per-instrument convention would make accrued the single field a consumer must
special-case before summing. I'll include both a long and a **short** position in the W1
fixtures, per your request — and I'd note the short case is the one that would silently pass
a long-only test suite, which is presumably why you asked.

---

## 3. Where I'd push back

### 3.1 Treasury bills first — I'd rather not

You propose an explicitly specified **Treasury bill** first, then fixed-rate notes/bonds once
reference terms are agreed. I understand the motivation — a bill has no coupon schedule, so
it sidesteps the calendar/settlement fields your export lacks.

**My concern:** a bill is a single zero-coupon cashflow. Pricing one exercises almost none of
the machinery everything downstream depends on:

| Exercised by | Bill | Fixed-rate note |
|---|---|---|
| Single discounted cashflow | ✅ | ✅ |
| Coupon schedule generation | ❌ | ✅ |
| Day-count/accrual conventions | minimal | ✅ |
| Clean vs dirty price reconciliation | ❌ (no accrued) | ✅ |
| Settlement-date handling | minimal | ✅ |
| Curve pillars beyond one point | ❌ | ✅ |
| **Meaningful rate sensitivity** | one pillar | **full curve** |

If W1 ships bills only, then at W1 exit we still have **zero** evidence about schedule
generation, accrual conventions, or clean/dirty reconciliation — which are precisely the
things that must be right before W2's SOFR schedules, and precisely the things your export is
missing fields for. That defers the risk instead of retiring it.

**Counter-proposal — both, in one wave, sequenced within it:**

1. **Bill first as the transport/identity smoke test** (days, not weeks). It proves the
   bundle → adapter → result → coverage path end to end with trivial pricing math. Genuinely
   useful, and I agree with that much of your reasoning.
2. **One fixed-rate note immediately after**, against an explicitly enumerated term set.

For (2) I need these, and I'd rather discover a missing field now than at W2:
`issueDate`, `maturityDate`, `coupon` (percent, per your §3), `couponFrequency`, `dayCount`,
`calendar`, `businessDayConvention`, `firstCouponDate`, `penultimateCouponDate` (stubs),
`settlementDays`, `faceAmount`, `cleanPrice` (fraction of par), `accruedInterestFraction`.

**If your export cannot supply some of these, that is exactly what I want to learn in W1**,
while the pricing math is still simple enough that the conversation is about data rather than
about my model.

Agreed unreservedly on the rest: **no silent assumption of absent terms, and no inferring
TIPS/FRN support from a generic Treasury label.** TIPS and FRNs will be explicit
`unsupported-product` rows until separately specified.

### 3.2 In-memory attempt state — I want to accept less than you offered

You wrote that my attempt state "may remain in memory initially," provided successful result
artifacts survive a restart and there's a deterministic lookup by workload identity.

**I'd rather build the deterministic lookup in W0 than take the allowance.** Reasoning:

- The durable half is the *artifact write plus a content-addressed key*. That is cheap.
- The in-memory half is [`engine/api/routes.py`](../../../engine/api/routes.py)'s `_JOBS` dict,
  which today is a genuine gap (**I-08** in [Known Issues](../../known-issues.md)): a restart
  loses every job id, and a lost job is currently indistinguishable from a computation
  failure.
- "Temporarily in memory" tends to become load-bearing the moment anything is built on it.

**Proposal:** before publishing any result artifact I write it at a deterministic path
derived from the workload key, so `GET /risk/results/by-workload/{workloadKey}` answers
correctly after a crash, a restart, or a lost notification — without any durable state in my
dispatcher process. Your coordinator keeps owning the durable logical job; I own the attempt
and the content-addressed artifact. That satisfies your crash-after-publication-before-
notification case by construction rather than by retry policy.

I'll still expose a worker boot epoch so restarts are *detectable*, but I agree with you that
it is not a result store and I'm not proposing it as one.

### 3.3 Identity — accepted, with one addition

Accepted in full: opaque `itemId` plus source identity, `clusterEpoch` echoed, same `itemId`
used for cube ordering. Your point that a security ID alone can't identify holdings across
accounts is correct, and your "same security in two accounts" fixture is the right test.

**One addition:** I also need the item ordering published as **its own artifact** alongside
any array output, rather than only being implied by result-row order:

```jsonc
{"kind": "item-order", "uri": "gs://.../order.json",
 "itemIds": ["r-0001", "r-0002", "..."], "hash": "sha256:..."}
```

A cube axis is resolvable to identities only if the mapping is a first-class, hashed artifact.
Otherwise a filtered or partially-covered result silently misattributes every column — and it
would misattribute *quietly*, which is the failure mode I most want designed out.

This matters to me more than usual because my engine currently has **no identity fields at
all** — trades are positional, results keyed by array index (**I-10**). Ordering is correct
and tested today, but it is correct by construction rather than by identity, and I don't want
the contract to inherit that weakness.

---

## 4. Points needing tighter definition

### 4.1 Assumed curves must be structurally marked, not labelled by convention

Agreed that an explicitly requested assumed curve is acceptable and silent synthetic
replacement is not. To make that enforceable rather than aspirational, I propose the
**provenance travels on the curve object itself**, not alongside it:

```jsonc
{
  "curveId": "USD-SOFR-DISC",
  "provenance": "assumed",        // "market-derived" | "assumed" | "derived"
  "assumedReason": "MARKET_PACKAGE_NOT_SUPPLIED",
  "constructionVersion": "flat-3pct-v1"
}
```

Every result computed against any `assumed` curve carries a top-level
`"marketProvenance": "assumed"`. **A consumer must not have to join across objects to
discover that a number is not market-derived.**

My own motivation: my `ZeroCurveConfig` is currently bare `times`/`rates` with no metadata
field at all, so today an assumed curve and a market-derived one are **indistinguishable
inside my engine**. I'd rather fix that at the contract boundary now than let it become
another silent gap.

### 4.2 Workload key — proposed exact contents

Accepting your framing; proposing the concrete list so we can diff it rather than discover a
mismatch later. Workload key = canonical hash of:

1. Portfolio bundle hash
2. Market package hash + reference-data identity
3. Requested calculation set (sorted)
4. Reporting currency
5. **Coverage policy** (your point — it changes the output, so it changes the workload)
6. Model + calibration versions
7. Precision profile (per-stage: simulation / pricing / risk)
8. Scenario definition: set ID + hash if supplied; **generator version + seed + scenario
   count** if engine-generated
9. Adapter `mappingVersion` (§2.5)
10. Engine version

Items 8–10 are ones I want explicitly in scope: a generator change, a unit-mapping change, or
an engine upgrade all change the number, so none may reuse a cached result.

**Attempt ID** is separate and always fresh — benchmark repetitions create new attempts of the
same workload, which is exactly the behavior my precision research needs.

### 4.3 Hash canonicalization — I want to go further than naming them

Agreed on naming `bundleId` (manifest body without `bundleId`) separately from a hash of the
final manifest file. Beyond naming, I'd like to pin:

- **Exact byte scope** of each hash, stated as a rule a third implementation could follow.
- **Canonical JSON form** for anything hashed after serialization: UTF-8, sorted keys, no
  insignificant whitespace, and an explicit number-formatting rule.
- **Large integers as decimal strings** across the boundary (`consensusSequence`,
  sequence numbers) — JavaScript truncates past 2^53, and your console is the consumer.
- **A fixture per hash kind**, with expected values exchanged, so we detect a canonicalization
  divergence in W0 rather than through a mysterious mismatch later.

Float formatting in hashed payloads is the classic silent divergence between a JVM producer
and a Python consumer. I'd rather over-specify it once.

---

## 5. The one blocker your fixture list assumes

### 5.1 I cannot currently produce "SOFR → explicit unsupported row"

Your §5 lists: *"USD-SOFR booking before W2 support → explicit unsupported row, never generic
IBOR substitution."* **I agree with the requirement completely and cannot satisfy it today.**

The problem is structural, not a missing `if`:

- [`build_vanilla_swap`](../../../engine/models/ore_builders.py) constructs **one** kind of swap:
  a generic term-IBOR index (`SimIndex6M`), **ACT/365 on both legs**, TARGET calendar,
  schedule derived from a **tenor string** (`"5Y"`).
- `SwapConfig` has **no field** for index family, day count, calendar, compounding, or
  explicit effective/maturity dates.

So a SOFR booking arriving today isn't rejected — **there is nowhere for its SOFR-ness to be
represented**. The adapter would map it to a tenor and it would price as generic IBOR,
confidently and wrongly.

**Magnitude, so this isn't hand-waving:** ACT/360 vs ACT/365 changes every accrual factor by
`365/360 − 1` = **1.389%**. On a $1mm 5Y fixed leg at 3%, ≈ **$1,906** — roughly **46× a 1bp
DV01**. And **no test in my repository would catch it**, because every test builds its inputs
with that same builder. This is **I-05** in [Known Issues](../../known-issues.md).

**What I'll do in W0, before any pricing work:**

1. Add the convention fields to the trade schema — `floatingIndex`, `fixedLegDayCount`,
   `floatingLegDayCount`, `compounding`, `calendar`, `businessDayConvention`,
   `effectiveDate`, `maturityDate` — so a booking's conventions **can be represented even
   though they can't yet be priced**.
2. Add a **supported-convention allowlist**. Anything outside it returns
   `status: "unsupported"` with `reason: "CONVENTION_NOT_SUPPORTED"` and the specific
   offending field, before any pricing runs.
3. **Refuse to infer.** A booking with no stated conventions is `unsupported`, **not**
   defaulted to the generic builder.

Point 3 is the one I want explicit agreement on, because it has a cost you should weigh: if
your current `contracts.csv` doesn't carry these fields, then **every OTC row becomes
`unsupported` at W0** rather than silently producing numbers. I think that's correct — a
visibly empty result is recoverable, a silently wrong one is not — but it will look like a
regression against the mock, and I'd rather agree on it now than have it appear as a surprise.

**This is why I want the faithful SOFR builder to be its own wave.** It is not one flag on the
existing builder: every current swaption pricer depends on that builder's ACT/365 consistency
with my simulation's year-fraction time axis. The SOFR path is a **separate** builder, gated
on D03/D04, with the generic one untouched.

### 5.2 Fixture list — accepted, with two additions

Your eight cases are accepted as the acceptance package. Two more, both targeting failures
that pass silently:

| Case | Expected outcome | Why |
|---|---|---|
| **Short bond position** | Correct signs on NPV **and** accrued interest | §2.5 — the case a long-only suite passes |
| **Same contract, two precision profiles** | Distinct workload keys, no cache reuse, comparable results | §4.2 item 7 — a cached fp64 result returned for an fp32 request would be invisible |

---

## 6. What I'm committing to

**W0 (unblocked — starting now):**

- Result schema + fixtures per §2.1/§2.2/§2.4 (method-tagged sensitivities, per-calculation
  coverage, explicit aggregate completeness).
- Convention fields on the trade schema + allowlist + `unsupported` path (§5.1).
- `GET /capabilities` returning the supported (product × convention × calculation) matrix.
- Adapter unit mapping with `mappingVersion`, tested long **and** short (§2.5).
- Deterministic result lookup by workload key (§3.2).

**W1:** equity positions; Treasury bill as transport smoke test, then one fixed-rate note
against the §3.1 term set; identity plumbing (`itemId` + source identity, item-order artifact).

**W2:** faithful USD-SOFR builder — **blocked on D03/D04**, and I won't start it on assumed
conventions.

**Already shipped** (verified this session, 824 tests passing):

| Item | Test |
|---|---|
| Per-instrument NPV; total is the sum of the parts | `TestPerTradeBaseNpv` |
| Swap Delta/Gamma/Theta through the portfolio path | `TestSwapGreeksReachThePortfolioPath` |
| Bermudan vega for calibrated sigma | `TestBermudanVegaReachesThePortfolioPath` |
| Aged-swap approximation warned, not silent | `TestAgedSwapWarningIsNotSilent` |

The aged-swap item is **flagged, not fixed** — the inaccuracy is unchanged, and it needs the
`pastFixings` history to actually close. Full register: [Known Issues](../../known-issues.md).

---

## 7. Decisions I need back from you

| # | Question | Blocks |
|---|---|---|
| 1 | **Agree that unmapped-convention OTC rows return `unsupported` at W0**, even if that means every current OTC row, rather than being priced generically? | §5.1 — the whole W0 exercise |
| 2 | Can `contracts.csv` carry the §5.1 convention fields, or do they need a new artifact? | W0 schema freeze |
| 3 | Bill **and** note in W1 (§3.1), or bill only? If note: can you supply the term set? | W1 scope |
| 4 | **Accrued interest signed for the position** (§2.5)? | Unit freeze |
| 5 | Deterministic result lookup in W0 (§3.2) rather than the in-memory allowance? | Job protocol |
| 6 | Timeline for D03/D04 SOFR conventions? | W2 start |

Questions 1 and 6 are the ones I'd most like answered first — 1 because it changes what W0's
output looks like and I don't want it to read as a regression, and 6 because W2 is the
highest-value wave and I can't begin it on guessed conventions.

Everything else in your response I'm treating as agreed. Happy to turn this into a schema
diff against your v1 manifest as soon as 1–4 are settled.
