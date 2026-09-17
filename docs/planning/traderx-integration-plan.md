# TraderX Integration — Actionable Plan

**Date:** 2026-09-15 (last revised 2026-09-16) · **Owner:** Alex (JAX Risk Engine side)
**Status:** Working plan. Supersedes nothing; it consolidates the agreed outcome of the
four-document exchange into executable tasks.

**Source documents (the negotiation, in order):**

1. [EOD Contract Proposal](eod-contract-proposal.md) — my opening position
2. `firstEODProposal.md` → TraderX's opening
3. [Response v2](eod-contract-response-v2.md) — my reply to their corrections
4. `eod-response-to-alex-v2.md` → their answers + the YU18 package
5. [Response v3](eod-contract-response-v3.md) — my reply; hashes verified
6. `eod-response-to-alex-v3.md` → their compatibility work + the source review that found **I-13**
7. [Response v4](eod-contract-response-v4.md) — my reply; W0 shipped, I-13 reproduced
8. `eod-response-to-alex-v5.md` → their **independent verification of the priced results**,
   plus two reproducible defects (**I-19**, **I-20**)
9. [Response v5](eod-contract-response-v5.md) — my reply; both fixed, W1.6 inserted ahead of W1.5
10. [Response v6](eod-contract-response-v6.md) — my reply; **W1.6 delivered**, all four of
    their compatibility items implemented, plus the `accrual-basis` versioning question
    restated as load-bearing

**Companion:** [Known Issues](../known-issues.md) — the defect register. Task IDs below
reference issue IDs (`I-NN`) where they close one.

---

## 0. The one-paragraph version

TraderX exports an immutable, hash-pinned EOD bundle (positions + OTC contracts + an
`instrument-terms.json` reference artifact). Their coordinator discovers it, records a durable
logical job, and calls my API. My worker verifies hashes, joins terms, refuses what it cannot
faithfully price, prices what it can against an **explicitly requested** curve, and publishes
immutable identified results that their coordinator discovers and reconciles. **Nothing is
ever silently approximated** — that principle is what most of the design below exists to
enforce.

**Delivery order: W0 → W1 → W2.** W0 is contract and refusal machinery (no pricing). W1 adds
the first real pricers. W2 adds faithful USD-SOFR and is gated on an external decision.

---

## 1. Where things actually stand

### Settled — do not reopen

| Topic | Agreement |
|---|---|
| Bundle transport | Immutable GCS bundle + versioned manifest; pull-discoverable, notifications are accelerators only |
| Job ownership | TraderX owns the durable logical job + retry decisions; I own computation attempts + result publication |
| Refusal policy | Unmapped conventions → `unsupported` + `CONVENTION_NOT_SUPPORTED`, **even if that covers every OTC row today** |
| Market data | TraderX supplies dated observations; I own curve construction and return repricing diagnostics |
| Assumed curves | Must be **explicitly requested** by profile ID. Missing market data → job **fails**, never falls back |
| Units | Coupon = annual percent; marks/accrued = fraction of par; quantity = signed face. Normalization in my adapter, `mappingVersion` echoed |
| Accrued sign | Signed for the position (short → negative), consistent with NPV |
| Sensitivities | `rateSensitivity` with `method` (`ad-first-order` \| `bumped-revaluation`), explicit numeric bump, factor, units |
| Gamma | Scaled second derivative, **no ½ factor**, diagonal only, cross terms excluded |
| Coverage | Per **calculation** per item; 5 statuses; `allOutcomesAccountedFor` vs `allApplicableComputed` |
| Identity | Opaque `itemId` + source identity (kind/account/security-or-contract) + `clusterEpoch`, **including on unsupported rows** |
| Cube output | Artifact reference with shape/dtype/ordering/hash + separate hashed item-order file — never inlined JSON |
| Attempt semantics | Lookup returns most recent **successful** attempt; attempts immutable; result manifest is the commit point |
| W1 scope | Bill **and** note, separate acceptance checks, plus equity positions |
| Accrued source label | Value is taken from the **export** (`accrualSource: "exported-fraction"`); `"recomputed-schedule"` is the alternative, so the label is always explicit |
| Accrual tolerance | `round(0.5 × 10^(−fractionDecimals) × \|face\|, 2) + 0.01` — derived from `accrualBasis`, never a fixed constant. **Never compare the two monetary paths as exact equals** |
| Accrued sign, restated | `fraction × signed face` in **one step**. No separate `sign()` factor — a second multiplication makes a short position positive |
| `accrualBasis` artifact | `traderx.accrual-basis.v1` **field shape** confirmed and frozen. **Its value *vocabulary* is not** — whether new enum values force a new schema version is unanswered, and W1.6.1 refuses unrecognized ones on an unconfirmed reading. Registered as **[I-23](../known-issues.md#i-23)**, status ASSUMPTION |
| Missing-accrual fixture | Two boundaries: TraderX **rejects before publication**; my mapper treats the same bytes as a **negative test** → `unavailable` / `ACCRUED_NOT_SUPPLIED` |
| `synthetic` vs `assumed` | Distinct values. Their `provenance.origin` stays `synthetic`\|`supplied`; my curve `inputOrigin` is separate. `supplied` ≠ observed |
| Lookup states | Never a bare 404 for accepted work: `UNKNOWN_WORKLOAD` / `running` / `failed` / `completed` are four distinct responses |
| Submission identity | Caller-generated idempotent `submissionId`; retrying recovers the same attempt. `reuseExistingResult:false` means "don't serve cache", **not** "always start a new attempt" |
| Publication recovery | Manifest is the commit point **and** discoverability no longer depends on the pointer — lookup falls back to a scan/reconcile. The pointer is a cache |

### Verified working (evidence, not claims)

- **All 14 golden-vector hashes reproduce** from an independent stdlib implementation.
- **ORE reproduces the note fixture schedule exactly** (4 periods, ACT/ACT ICMA, unadjusted).
- **Accrued interest agrees to exported precision**: mine `0.0185714286` vs theirs `0.018571`.
  **Monetary value corrected in v4** — see the accrual-rounding row below.
- **The SOFR refusal runs against the real fixture**, echoing all 13 `missingTerms` plus
  `contractId` / `clusterEpoch` / `accountId`. v1 refuses distinctly with
  `TERMS_NOT_SUPPLIED`.
- **The bill prices, exactly** (W1.2): **+98,507.15 / −98,507.15** long/short, **zero
  difference** against an independent ORE valuation at machine precision.
- **The note prices, exactly** (W1.3): **+103,308.33 / −103,308.33** long/short, **zero
  difference** against an independent `ORE.FixedRateBond`, with accrued interest
  reconciling to their exported `0.018571` at the §2 derived tolerance. This is the
  half of the W1 exit criterion where the two systems' numbers actually had to agree.
  **692 integration tests** pass in ~2.3s (491 at W1.3; W1.6 added 165).

### Suite status — stated honestly

Full run 2026-09-17, **after W1.5**: **1,716 passed, 0 failed** (11m24s) — the complete
suite, nothing excluded. That is 1,618 + the **98** W1.5 tests (40 treasury, 37 wire-through,
21 API schemas), so the delta reconciles exactly.

**A caveat about the runner, not the code:** an earlier identical invocation **hard-aborted**
inside XLA compilation with no summary line at all
(**[I-27](../known-issues.md#i-27)**). A green full run is therefore real when it happens but
**not reliably repeatable on demand**.

Previously 1,618 / 0 after W1.6, 1,452 / 1 after the v5 fixes, 1,384 / 2 after W1.4,
1,324 / 2 after W1.3, 1,217 / 2 after W1.2.

> ⚠ **Two verification hazards found during W1.5, both now guarded against.**
> 1. **Run `.venv/Scripts/python.exe -m pytest`, never the bare `python`.** The system
>    interpreter lacks `pydantic` and `jsonschema`, making 46 tests **silently
>    uncollectable** and yielding a count that looked like a regression.
> 2. **A shell exit code is not a pass.** A run that hard-aborted at 4% was briefly taken
>    for green because `echo EXIT=$?` captured a redirect rather than pytest (real exit: 3).
>    **Confirm a summary line was printed.**

**This is the first fully green full run in this exchange.** Worth stating plainly, because
working rule 9 cuts the other way too: a green suite is evidence about the tests, not proof
about the code. Two of the three defects found in this exchange came from reading source, and
the W1.6 `submissionId` bug came from reviewing my own code — none came from running this.

**Two failures were seen during W1.6 and both were resolved rather than waived:**

1. `test_integration_equity.py::TestCapabilitiesAdvertiseW14::test_stage_is_w14` pinned the
   literal `"W1.4"`, so the W1.6 stage bump broke it. That is a **test design defect** — it
   fails on every future stage bump regardless of whether the equity contract changed. Now
   asserts the stage has *reached* W1.4, and verified to still fail if the stage regresses
   below it. The equity contract itself is pinned by the four other tests in that class.
2. `test_integration_terms.py::test_unsupported_terms_schema_is_rejected` used `.v2` as its
   example of an unsupported schema — correct at W0.2, and precisely what W1.6.1 is chartered
   to change. Now uses `.v99`; the contract it protects is unchanged.

**On the flaky concurrency test.** `test_cross_tier_jobs_correct_and_concurrent` failed in
every full run from W1.2 through the v5 fixes, and **passed** in the W1.6 runs. That is
consistent with the recorded diagnosis — a wall-clock overlap assertion that the OS can
serialize under load, not a code defect (it passes 3/3 in isolation and against stashed
pre-fix code). It is load-dependent, so a green run is not proof it is fixed; the underlying
assertion is still timing-sensitive and remains a follow-up.

The 2 previously-recorded `pydantic` failures are **resolved** — the dependency is now
installed (2.13.5), and the run contains zero `ModuleNotFoundError`.

Earlier figures in this exchange (824, 1092, 1175) were stale or unreproducible.

### Blocked, and on what

| Blocked | On | Can I start? |
|---|---|---|
| Faithful USD-SOFR pricing | D03/D04 convention agreement | ❌ External |
| Aged-swap correctness (**I-04**) | Historical fixings TraderX doesn't export | ❌ External |
| Independent check of TraderX's `.gitattributes` work | Their push + commit SHA | ❌ External |
| ~~Priced **bill** results~~ | — | ✅ **Done** (W1.2) |
| ~~Priced **note** results~~ | — | ✅ **Done** (W1.3) |
| ~~I-13 / I-14 fixes (W0.10)~~ | — | ✅ **Done** |
| ~~Terms v2 / schema versions / `accrualSource` / HTTP~~ | — | ✅ **Done** (W1.6) |
| Everything else | Nothing | ✅ Start now |

### Corrections I owe, or have made

| Correction | Source |
|---|---|
| v3 §3.3 claimed v1 bill/note rows are indistinguishable — **wrong**, v1 carries coupon and maturity | Their v3; conceded in v4 §1.2 |
| v3 §2.2 reported accrued as `+1,857.14` as though it were *the* value — it is the **recomputed-unrounded** path; the export path gives `1,857.10` | Their v3; conceded in v4 §2 |
| `_swap_curve_configs` validates only the Greeks path | Their v3 source review → **I-13**, now fixed |
| I-14 first diagnosed as `price_portfolio` leaking x64 across jobs — **wrong**; it re-enables the flag deliberately. The leak was one level down, in `generate_paths` | Probed while fixing → entry corrected |
| I-15 first diagnosed as a flaky timing assertion — **incomplete**; the test's premise was unsound (warm-JIT jobs take ~15ms and never coexist) | Surfaced when the first fix also failed |
| **Accrual tolerance rounded the rounding bound** — `round(err, 2) + 0.01` truncates the quantity it exists to bound. At 124,000 face the agreed rule gives 0.072, the code gave 0.07: **tighter than agreed**, so it could refuse a reconciliation inside the exporter's own stated rounding error. The 100,000 fixture hides it exactly | Their v5 §3; **fixed** → **I-19** |
| **Impossible calendar dates aborted the whole bundle** — `ORE.Date` raises `RuntimeError` (SWIG over C++ `std::runtime_error`) for `2025-02-30`, and the parsers caught only `(ValueError, TypeError)`. One typo'd date returned *nothing* for 200 good rows, breaking this boundary's core contract | Their v5 §3; **fixed** → **I-20** |

---

## 2. W0 — Contract and refusal machinery

> **Status as of 2026-09-15: exit criterion met.** Every W0 task except **W0.8** and the
> newly-added **W0.10** is implemented, with 318 tests (295 in `engine/integration/`, 23 for
> the tail diagnostics in `engine/risk/var_es.py`). The SOFR case returns
> `CONVENTION_NOT_SUPPORTED` naming all 13 missing terms, and bill/note return structurally
> valid results with `npv: unsupported`. See the per-task markers below and
> [the boundary's own doc](../reference/eod-integration.md).
>
> **W0.10 was added after the exit criterion was met**, from TraderX's v3 source review. It
> does not gate the SOFR refusal (already delivered) but **does run ahead of W1**: it is the
> only open item that produces a wrong number today.
>
> Seven plausible-but-wrong implementations were patched in and verified to fail the new
> tests (working rule 3): naive `blank accrued → 0.0`; USD-SOFR/ACT-360 added to the
> allowlist; absent conventions defaulted into the generic builder; the loader's
> required-column check removed (surfaced as a bare `KeyError`); missing market inputs
> falling back to a default profile; ES standard error returning `0.0` at `n < 2`; the
> measure vocabulary drifting between its two definitions.

**Goal:** return a correct, identified, *refusing* result for every fixture without pricing
anything. **Exit criterion:** the SOFR case returns `CONVENTION_NOT_SUPPORTED` naming all 13
missing terms, and bill/note return structurally valid results with `npv: unsupported`.

**Why this ordering:** it proves the entire transport → identity → coverage → publication path
while pricing math is still out of scope, so contract bugs and pricing bugs never get debugged
simultaneously.

---

### W0.1 — Bundle ingestion and hash verification · ✅ DONE

> `engine/integration/bundle.py`. The CRLF warning below was **confirmed live in this
> checkout**: all 15 `reference/traderX` fixture artifacts fail as-is and reproduce their
> manifest hashes exactly after LF normalization. Root `.gitattributes` added; LF-exact
> fixtures vendored to `tests/fixtures/traderx-eod/`.

**Deliverable:** `engine/integration/bundle.py` — read a v1/v2 bundle, verify every artifact
against its own hash, parse preambles, reject tampering.

**Steps:**
1. Parse `manifest.json`; assert schema is `traderx.eod-bundle.v1` or `.v2`.
2. Verify **each** artifact against its **own** `sha256` — positions, contracts, and (v2)
   `instrument-terms.json`. Never compare a CSV against `cutSha256`.
3. Validate declared `rows=` counts and shared cut/date/version preamble fields.
4. Distinguish **empty contracts file** (valid, zero OTC coverage) from **missing** file
   (integrity failure).
5. Read bytes in **binary**, never text mode.

> **⚠ CRLF — non-negotiable.** Hashes are over committed bytes. Windows checkouts with
> `core.autocrlf=true` silently rewrite LF→CRLF and **every hash fails**. This already broke
> TraderX's own verifier in my checkout. Read binary; never let git or Python translate.
> Add `*.json -text` / `*.csv -text` to `.gitattributes` for any vendored fixtures.

**Tests:** happy path v1 + v2; tampered byte → integrity failure; wrong row count → failure;
empty vs missing contracts distinguished; **a CRLF-translated fixture fails loudly** (not
silently passing).

---

### W0.2 — Terms artifact join · ✅ DONE

> `engine/integration/terms.py`.

**Deliverable:** join `instrument-terms.json` onto positions/contracts.

**Steps:**
1. Verify the terms file hash **before parsing** (part of W0.1).
2. Join **securities by security identity** (shared across long/short accounts); **OTC by
   `contractId` + `clusterEpoch`**.
3. Treat non-empty **`missingTerms` as an authoritative refusal input**, not a hint — carry
   the list verbatim into the refusal reason.
4. Reject duplicate identities, wrong epoch, unjoinable rows.
5. **v1 bundles have no terms artifact** — every instrument needing terms is `unsupported`.

**Tests:** same security across two accounts joins to one terms entry with correct per-account
signs; wrong epoch rejected; duplicate identity rejected; v1 bundle → terms-dependent
calculations `unsupported`.

---

### W0.3 — Unit normalization adapter · ✅ DONE

> `engine/integration/normalize.py`. All four rows of the zero-coupon table are pinned, and
> the naive `blank → 0.0` implementation was verified to fail exactly 4 tests while passing
> the other 20 — the "bill-only suite" failure mode this section warns about.

**Deliverable:** `engine/integration/normalize.py` + `mappingVersion` echoed in every result.

| Source | Unit | Normalized |
|---|---|---|
| `coupon` | annual **percent** (`4.0`) | decimal (`0.04`) |
| `closingMark` | clean, **fraction of par** | keep as fraction; echo `observedCleanPrice` |
| `accruedInterestFraction` | fraction of par | `× faceAmount × sign(position)` → signed currency |
| `quantity` | signed face | `signedFaceAmount` |

**The zero-coupon rule — key on terms, never on the blank:**

| Terms say | Accrued field | Result |
|---|---|---|
| Zero-coupon (`couponFrequency: NONE`) | blank | `0.0`, `provenance: structural-zero` |
| Coupon-bearing | blank | **`unavailable` + `ACCRUED_NOT_SUPPLIED`** — never `0.0` |
| Coupon-bearing | present | convert |
| **No terms artifact** | blank | **`unavailable`** — uninterpretable without terms |

> **Why this matters:** a bill and a coupon-bearing note both show a blank accrued field. The
> naive `blank → 0.0` is correct for one and silently wrong for the other. A bill-only,
> long-only test suite passes while the note case is wrong.

**Tests:** long **and short** note (signs opposite, equal magnitude); bill → structural zero;
**coupon-bearing with blank accrued → `unavailable`, asserted NOT `0.0`**; v1 bundle →
`unavailable`; round-trip `mappingVersion` present.

---

### W0.4 — Convention allowlist and refusal path · closes part of **I-05** · ✅ DONE

> `engine/integration/conventions.py`. Allowlist is a positive list; refusal happens before
> any pricing object is constructed. Scope: the EOD boundary only — `build_vanilla_swap` and
> `SwapConfig` are untouched, so I-05 stays OPEN.

**Deliverable:** refuse before constructing any pricing object.

**Steps:**
1. Add convention fields to the trade schema so a booking's conventions **can be represented
   even while unpriceable**: `floatingIndex`, `fixedLegDayCount`, `floatingLegDayCount`,
   `compounding`, `calendar`, `businessDayConvention`, `effectiveDate`, `maturityDate`,
   `lookback`, `lockout`, `paymentLag`.
2. Define the allowlist. **Today it contains only the generic term-IBOR/ACT-365 profile the
   engine actually implements.**
3. Anything outside → `unsupported` + `CONVENTION_NOT_SUPPORTED` + the offending fields.
4. **Refuse to infer.** A booking with no stated conventions is `unsupported`, *never*
   defaulted into the generic builder.

> **⚠ The bug this prevents.** `build_vanilla_swap` produces `SimIndex6M`, ACT/365 both legs,
> TARGET calendar, tenor-derived schedule. A USD-SOFR booking routed through it prices
> **confidently and wrongly**: ACT/360 vs ACT/365 shifts every accrual by 1.389% ≈ **$1,906 on
> a $1mm 5Y leg ≈ 46× a 1bp DV01** — and **no existing test catches it**, because every test
> builds inputs with that same builder.

**Tests:** SOFR fixture → `unsupported` naming all **13** `missingTerms`; a booking with
absent conventions → `unsupported`, **not** generically priced; an allowlisted generic swap
still prices.

---

### W0.5 — Result schema and coverage model · ✅ DONE

> `engine/integration/result.py`.

**Deliverable:** `RiskResult` schema + per-calculation coverage.

Frozen calculation names: `npv`, `accruedInterest`, `rateSensitivity`, `rateGamma`, `theta`,
`vega`, `varEs`.

Per-item, per-calculation status: `ok` | `unsupported` | `unavailable` | `failed` |
`not-applicable`.

```jsonc
"coverage": {
  "byCalculation": {
    "npv": {"ok": 211, "unsupported": 3, "unavailable": 0, "failed": 0, "notApplicable": 0}
  },
  "allOutcomesAccountedFor": true,   // statuses sum to itemCount — true even if all failed
  "allApplicableComputed": false     // true only if unsupported+unavailable+failed == 0
}
```

**Rules:** `not-applicable` never counts against coverage (vega on a vanilla swap is not a
gap). Aggregates carry `coveredItemCount`, `totalItemCount`, `excludedItems[]`, and
**`complete: false` whenever they differ**. Cross-currency → per-currency aggregates with
`reportingCurrencyConversion: "NOT_APPLIED"`.

**Sensitivity payload** carries `method`, `derivative`, `shockedFactor`, numeric `bump`,
`value`, `currency` — never a method implied by a field name.

**Tests:** statuses sum to item count; `complete:false` on partial aggregate; mixed-currency
returns per-currency, not a blended scalar.

---

### W0.6 — Market input selection · closes part of **I-11** · ✅ DONE

> `engine/integration/market_inputs.py`. Fail-on-missing-inputs with no fallback on any
> branch; `ZeroCurveConfig.provenance` added as an **optional** field (all 69 existing
> construction sites untouched); top-level `marketProvenance`; `measure` label; and
> `tailCount`/`standardError` convergence diagnostics on every tail statistic, added
> additively so no existing VaR/ES key or value moves.
>
> **Not yet consumed by a pricer**, because W0 prices nothing: `resolve_market_inputs`
> returns the profile and its provenance, but materializing it into a curve a pricer
> discounts against is W1.2's first task.

**Deliverable:** explicit market-input mode; no silent fallback.

```jsonc
"marketInputs": {"mode": "assumed-profile", "assumedProfileId": "flat-3pct-v1"}
```

Absent, or `mode: "package"` with no package → **job fails** with `MARKET_INPUTS_NOT_SUPPLIED`.

Add curve provenance — **`ZeroCurveConfig` currently has no metadata field at all**, so
assumed and observed curves are indistinguishable inside the engine:

```jsonc
{"curveId": "...", "inputOrigin": "observed|assumed|mixed|synthetic",
 "construction": "bootstrap-v2", "inputHashes": ["sha256:..."]}
```

Every result computed against any assumed curve carries top-level
`"marketProvenance": "assumed"`. Also add `measure` (`risk-neutral-pricing` |
`historical-forecast` | `deterministic-stress`) plus effective sample size and MC standard
error on tail statistics.

**Tests:** missing market inputs → fail, no curve substituted; assumed profile → provenance on
curve **and** top level; `measure` present on every risk figure.

---

### W0.7 — Identity plumbing · closes **I-10** · ✅ DONE at the EOD boundary

> `engine/integration/identity.py`. Steps 2–4 are complete for rows arriving through
> `engine/integration/`. **Step 1 is not done**: `instrumentId`/`accountId` are still absent
> from the trade configs themselves, so `price_portfolio` remains positional and I-10 stays
> OPEN.

**Deliverable:** opaque `itemId` + source identity end to end.

1. `instrumentId`/`accountId` on every trade config, echoed on every result row.
2. Source identity block: `{kind, accountId, security|contractId}` + `clusterEpoch`.
3. **Unsupported rows carry full identity too** — an unidentified refusal is useless.
4. Publish item ordering as its **own hashed artifact**; never rely on array position.

**Tests:** same security in two accounts → distinct rows, correct signs; unsupported row has
identity; item-order artifact hash matches result ordering.

---

### W0.8 — Durable result lookup · mitigates **I-08** · ⚠️ **PARTIAL** (absorbed into W1.6.4)

> **Landed in W1.6.4:** the HTTP endpoint (`GET /eod/results/by-workload/{key}`), the
> canonical **workload key** (`engine/integration/workload.py`), **idempotent submission**
> via `submissionId`, **immutable terminal attempts**, and the **four distinguishable lookup
> states** — so an accepted-but-running job no longer looks like an unknown one.
>
> **Still not built:** the publication protocol and manifest-scan recovery below. Those need
> a persistent artifact store to scan, and there is none — so the store is still an
> in-process dict and a restart still loses *running*-state knowledge. The state machine is
> right; nothing durable backs it yet. **Mitigated, not fixed** (working rule 5).

**Deliverable:** `GET /risk/results/by-workload/{workloadKey}`.

**Workload key** = canonical hash of: portfolio bundle hash · market package hash +
reference-data identity · **terms-file hash** · calculation set (sorted) · reporting currency ·
**coverage policy** · model + calibration versions · precision profile · scenario definition
(set ID+hash, or generator version+seed+count) · adapter `mappingVersion` · engine version ·
valuation time.

**Publication protocol — the result manifest is the commit point:**
1. Write artifacts to a temporary path
2. Verify hashes
3. Atomically publish the result manifest
4. **Then** advance the workload pointer

Crash before (3) → orphaned artifacts no lookup reaches. Crash after → complete discoverable
result. **No window where a partial write is reachable.**

> **Gap TraderX found in v3, closed in v4 §4.1.** The protocol above left the window
> *between* (3) and (4) unspecified: a crash there published a complete result that no
> lookup could find. **Discoverability no longer depends on the pointer** — lookup falls
> back to a scan/reconcile over published manifests when the pointer is behind, and advances
> it as a side effect. A published manifest is the durable record; **the pointer is a cache.**

**Lookup must distinguish four states** (v4 §4.2 — a bare 404 for both "unknown" and
"running" invites a duplicate overnight batch):

| State | Response |
|---|---|
| Never submitted | `404 UNKNOWN_WORKLOAD` |
| Accepted, running | `200 {"state": "running", "attemptId": ...}` |
| Accepted, failed | `200 {"state": "failed", "attemptId": ..., "reason": ...}` |
| Completed | `200 {"state": "completed", ...result...}` |

**Submission identity** (v4 §4.3): a caller-generated idempotent `submissionId`. Retrying a
lost submission response **recovers the same attempt**; a deliberate second benchmark
repetition uses a **new** `submissionId`. `reuseExistingResult:false` means "don't serve me a
cached result" — **not** "start a new attempt every time you see this request."

> **I-08 caveat, on the record.** The job table is still an in-process dict, so a restart can
> lose *running*-state knowledge. That is precisely why the manifest scan matters: recovery
> goes through content-addressed artifacts, **never** process memory. A lost in-memory job is
> an infrastructure event, not a financial failure.

Lookup returns the most recent **successful** attempt — never failed, partial, or in-flight.
Attempts are immutable and permanently addressable by `attemptId`.

**Tests:** kill between artifact write and manifest publish → lookup finds nothing (no partial);
kill after publish → lookup finds complete result; **kill between publish and pointer advance →
lookup still finds it via scan**; four lookup states distinguished; same `submissionId` recovers
one attempt, not two; second attempt doesn't overwrite first; **different precision →
different workload key** (no cache reuse).

---

### W0.9 — `GET /capabilities` · ✅ DONE and **routed** (W1.6.4)

> `engine/integration/capabilities.py::capabilities()` returns the document, derived from the
> allowlist rather than hand-maintained. **Served at `GET /eod/capabilities`** since W1.6.4 —
> it had existed as unreachable code since W0.

Return the supported (product × convention × calculation) matrix, precision/device profiles,
engine/model/build versions, and known-limitation flags, so the coordinator can determine
*before submitting* whether a bundle is priceable. This is what makes "no silent exclusions"
enforceable rather than aspirational.

Now reports `"deliveryStage": "W1.6"` and `"calculations.mode": "partial"` (it read `"W0"` /
`"refusal-only"` while nothing priced). `conventions.swap.overnightCompounding: []` is still
an explicitly empty list, so a consumer sees SOFR is unsupported without inferring it from a
refusal. W1.6.2 added `capabilitySchema`, `termsSchemas`, `accrualBasisSchemas` and a
`schemas` block naming both document versions and where to fetch them.

---

### W0.10 — Validate curve indices before **all** pricing · closes **I-13** and **I-14** · ✅ DONE

> **Ran ahead of W1**, as planned — this was the one open item producing a **wrong number
> today**, and it was found by TraderX reading a pushed commit, not by my suite.
>
> `_validate_swap_curve_indices` in `engine/portfolio/request.py`, called from the existing
> `validate_portfolio_against_simulation` (which already ran before any JAX work), so **one
> check covers every pricing path** rather than each indexing site having to guard itself —
> the omission that caused I-13 in the first place.
>
> **I-14 was also fixed, and its cause turned out to be different from the original
> diagnosis.** `price_portfolio` was never affected (it re-enables x64 deliberately); the
> leak was in `generate_paths`, which set the process-global flag and never restored it,
> contradicting its own docstring. Now scoped by an `_x64_enabled` context manager that
> restores the prior value.
>
> **I-15 fixed too**, and the investigation is the interesting part: the first attempted fix
> (assert distinct worker PIDs) *also* failed, which revealed the test's premise was unsound
> — after priming, jobs take ~15ms and both ran on one worker, so it could never observe
> concurrency. Now uses a 1.0s job and asserts mechanism **and** overlap, deterministically.

**Deliverable:** no pricing path can index a curve list without validation. ✅

**Steps:**
1. ✅ Curve-index validation hoisted to **request admission** — before any pricing,
   independent of `compute_greeks`.
2. ✅ Negatives rejected explicitly, so `-1` can no longer wrap to the last curve.
3. ✅ **I-14** closed at its actual source (`generate_paths` restoring the x64 flag).
   **Residual, deliberately not done:** `PortfolioResult` still carries no *realized* dtype,
   so a truncation from some other cause would stay invisible in the result. That belongs
   with **I-12**'s worker-device reporting, not here — doing it now would be scope creep on
   a fix that is already complete.

> **⚠ The bug this closes.** `fwd_idx=-1` prices silently against the **last** curve:
> measured `-5,857.0074` instead of the booked `-5,913.9266` — a 56.92 USD error on 2mm
> notional, **unbounded in principle** since it scales with curve separation. The failure is
> invisible: finite, plausible, no warning. `_swap_curve_configs`'s own docstring promises
> this cannot happen.

**Tests:** ✅ `TestCurveIndexValidatedBeforeAllPricing` (12 tests) drives the full **2×2** —
negative and out-of-range × `compute_greeks` both ways — **through `price_portfolio`**, not
through the helper. **8 of 12 fail against the pre-fix code**, including both negative cases
at `compute_greeks=False` (which returned a plausible NPV instead of raising); the 4 that
pass pre-fix are the valid-index controls and the two cases the Greeks path already guarded.

Plus `TestGeneratePathsEdgeCases::test_precision_32_restores_the_global_x64_flag` (I-14) and
the rewritten same-tier concurrency test (I-15), both verified to fail pre-fix.

---

## 3. W1 — First real pricers

**Exit criterion:** one exported Treasury case consumed end-to-end and returned as a
validated, identified financial result — TraderX's stated target.

> **Met by W1.2 and W1.3, for Treasuries.** Both cases now do exactly this: consumed from
> the hash-verified bundle, identified, priced, and returned with a reconcilable payload,
> each exact against an independent ORE valuation.
>
> **The note was the harder half, and it is the one that closed the criterion.** It needed
> the coupon schedule and the ACT/ACT (ICMA) accrual, and it is the case where our two
> systems' numbers actually had to agree — their `0.018571`, at the §2 derived tolerance,
> which it does. A bill agreeing on a single discount factor was a weaker claim than it
> looked, which is precisely why the plan ordered it first.
>
> **What remains in W1 is not this criterion:** the wire-through to the portfolio path
> (W1.5). Equity positions (W1.4) are resolved to a refusal — valuing them needs a spot
> source, not engine work (**I-18**).

---

### W1.1 — Day count as a per-instrument input ⚠ **prerequisite for everything below** · ✅ DONE

> `TIME_AXIS_DAY_COUNTER` (permanently ACT/365) split from a per-instrument
> `accrual_day_count`, defaulting to ACT/365 so the whole suite is byte-identical.
> `ACT/ACT (ICMA)` added; anything else — `ACT/360` included — is **refused** at
> `SwapConfig` construction, never defaulted. 27 tests in
> `tests/test_day_count_roles.py`, verified to fail against the dangerous wrong fix
> (making the time axis follow the instrument accrual).
>
> **Two corrections to the analysis below.** There are **three** `DAY_COUNTER` constants,
> not two — `engine/risk/greeks.py:138` is the third — used across **seven** modules, not
> five. All three are the time-axis role. And tracing every use: **49 are the time axis,
> only 2 are the accrual** (`ore_builders.py`'s `fixedLegDayCount`/`floatingLegDayCount`),
> so the risky part of this change was two lines.

**The subtlety that makes this more than adding a parameter.** `Actual365Fixed` is used in
**two distinct roles**, and there are **two independent `DAY_COUNTER` constants**
(`ore_builders.py:55` and `bermudan_swaption.py:173`), used across five modules:

| Role | Must it stay ACT/365? |
|---|---|
| **Simulation time axis** — converting dates to year-fractions matching `time_grid`, `maturities`, `hw_paths` | **YES — do not touch.** Changing it silently desynchronizes every pricer from the simulated curve cube |
| **Instrument accrual** — the coupon day count of the contract | **NO — must become per-instrument** (ACT/ACT ICMA for the note) |

**Steps:**
1. **Rename to separate the roles**: `TIME_AXIS_DAY_COUNTER` (stays ACT/365, used for all
   date→year-fraction conversions) vs a per-instrument `accrual_day_count` field.
2. Thread `accrual_day_count` through instrument configs → ORE builders.
3. Default to ACT/365 so **every existing test is byte-identical** (several pin
   `ORE.Actual365Fixed()` directly).
4. Only then add ACT/ACT (ICMA) as a supported value.

**Tests:** full suite byte-identical with defaults; ACT/ACT ICMA note reproduces the fixture
schedule; a trade requesting an unsupported day count is **refused**, not silently defaulted.

---

### W1.2 — Bill pricer · ✅ DONE

> `engine/integration/bill.py` — **the first number this boundary returns.**
> `signedFace × redemptionFraction × P(valuationDate, maturity)`. Long/short on the delivered
> fixture: **+98,507.15 / −98,507.15**, summing to zero.
>
> **ORE parity is exact** — zero difference at machine precision against an independent
> `ORE.FlatForward` + `ORE.CashFlows.npv` valuation, built from ORE's own machinery rather
> than by re-deriving `exp(-rt)` (which would restate the implementation and pass even if
> both were wrong together).
>
> **Four plausible-but-wrong implementations** were patched in and verified to fail (working
> rule 3): simple instead of continuous discounting (4 tests), a separate position sign on
> top of signed face (5), pricing a matured bill instead of refusing (2), and **treating any
> Treasury as a bill** — a note priced with the wrong model (3). The last is the dangerous
> one: a confident, plausible number for the wrong instrument.
>
> **Scope held deliberately narrow.** A priced `npv` does **not** make the other calculations
> answerable: `rateSensitivity`/`rateGamma`/`theta` still return `unsupported`, and
> `capabilities()` advertises `TREASURY: ["npv"]` and nothing more. W1.2 delivers a price,
> not a sensitivity.

Single discounted cashflow: `faceAmount × redemptionFraction × P(valuationDate, maturity)`.
No coupon schedule, no Monte Carlo, no calibration. Serves as the transport/identity smoke
test with trivial pricing math.

**Conventions, stated rather than assumed:** ACT/365 Fixed discounting (matching the
simulation time axis and the W0.4 allowlist), and **continuous** compounding to match the
assumed profiles' own stated convention — simple discounting would shift the price ~$3.6 per
$100k face, small enough to read as rounding.

**Refuses, each naming a reason:** `NOT_A_BILL` (coupon-bearing or scheduled),
`INSTRUMENT_MATURED` (maturity on/before valuation — **not** priced at face),
`TERMS_INCOMPLETE`. A v1 bundle and a request with no `marketInputs` both price nothing.

**Tests:** ✅ `tests/test_integration_bill.py` (27 tests) — ORE parity at matched terms;
long/short signs; **maturity-date boundary**; accrued = structural zero; plus the pipeline
end-to-end, the v1 refusal, and proof the note and SOFR refusals are unchanged.

---

### W1.3 — Note pricer · ✅ DONE

> `engine/integration/note.py`. Fixed coupons + bullet principal, discounted off the
> requested curve. Long/short on the delivered fixture: **+103,308.33 / −103,308.33**,
> summing to zero.
>
> **ORE parity is exact** — zero difference at machine precision against an independent
> `ORE.FixedRateBond` + `ORE.DiscountingBondEngine`, built from ORE's own bond machinery
> rather than by re-deriving `Σ cᵢ·exp(−r·tᵢ)`. Every coupon is checked against ORE's own
> cashflows one by one, so a dropped coupon and a compensating discount-factor error cannot
> cancel.
>
> **Accrued reconciles to TraderX's `0.018571`.** Both paths are carried: the
> `exported-fraction` ($1,857.10) is what the result *reports*, and the
> `recomputed-schedule` ($1,857.14) is the cross-check. They differ by $0.04 — the
> exporter's own HALF_EVEN rounding — so they are compared at the §2 derived tolerance,
> never as exact equals. Beyond tolerance is a **refusal**, not a warning.
>
> **This exercised the ACT/ACT (ICMA) half of W1.1 that the bill never touched.** Pricing
> the note on the engine's ACT/365 default instead shifts accrued by **$5.09 on $100k** —
> about 85× the $0.06 tolerance — so the accrual check catches that bug too, which a test
> pins.
>
> **`rateSensitivity` is delivered for the note, and labelled honestly.** A bumped
> revaluation at an explicit 1bp, `shockedFactor: "zero-curve-parallel"`. The plan asked for
> *per-pillar*; every registered assumed profile is a flat constant with no pillar structure
> to shift, so a per-pillar vector would be arithmetic theatre — recorded as **I-16**,
> closed by W2's `mode: "package"`. **The bill still has no sensitivity**, and
> `capabilities()` now reports per *shape* so that stays visible.
>
> **Two things this work found.** **I-17**: reusing the bill's date parser meant a malformed
> note date raised `BillPricingError`, which the pipeline's `except NotePricingError` never
> caught — one bad row failed the **whole bundle**. Fixed, with four regression tests.
> And the `engine.models` import ban fired on the ACT/ACT (ICMA) lookup: rather than relax
> it, the day-count vocabulary moved to the leaf module `engine/day_count.py`, and a new
> test closes the AST guard's transitive blind spot.

**Tests:** ✅ `tests/test_integration_note.py` (140 tests; recorded as 104 at W1.3 — stale,
re-counted 2026-09-16) — ORE parity including every
coupon and discount factor; accrued reconciliation to `0.018571`; the derived tolerance;
clean vs dirty; long/short mirrors; `rateSensitivity` sign, magnitude and maturity scaling;
schedule validation (gap, overlap, zero-length, maturity disagreement); every refusal;
plus the pipeline end-to-end, the v1 refusal, and proof the bill and SOFR paths are
unchanged.

---

### W1.4 — Equity position pricer · ✅ DONE (as a **refusal**)

> `engine/integration/equity.py`. `signedQuantity × multiplier × spot × fx` — and **two of
> those four factors have no source at this boundary**, so W1.4 ships an honest refusal
> rather than a number. `marketInputs` registers flat *interest-rate* profiles only;
> `SimulationConfig.equities` drives correlated risk-factor *paths*, takes no share count,
> and lives in a package this one may not import. Recorded as **I-18**.
>
> **The tempting wrong answer was `closingMark`.** `quantity × closingMark × multiplier`
> reproduces the exporter's own `marketValue` column *exactly* — which is what makes it
> dangerous. It would be an **echo, not a valuation**: TraderX's own number handed back as
> though the engine had priced it, reconciling perfectly and proving nothing, under a
> provenance derived from a *rate* curve it was never computed against. Patched in at both
> the pricer and pipeline level, it fails **20 of 60** tests (working rule 3).
>
> **Three distinct refusals, because they have different fixes.**
> `SPOT_SOURCE_NOT_SUPPLIED` (USD: send a spot), `FX_SOURCE_NOT_SUPPLIED` (non-USD: a spot
> alone still would not price it), `TERMS_INCOMPLETE` (a broken row, not missing market
> data). An absent currency is treated as **foreign, not assumed USD**.
>
> **The refusal is diagnosable.** It carries `signedQuantity`, `contractMultiplier` and
> `multipliedQuantity` — multiplier applied **exactly once** — so sign and size are already
> correct the day a spot arrives.
>
> **`EQUITY` joined the W0.4 allowlist deliberately.** It previously refused as
> `INSTRUMENT_TYPE_NOT_SUPPORTED` ("outside this engine's scope"), which is the wrong fact:
> a cash equity *is* in scope and fully understood, and needs one market input nobody
> supplied. `capabilities()` now reports `blockedOnMarketInput` as a third state alongside
> "priced" and "no pricer" — the only one the **consumer** can clear.
>
> **Fixtures:** `equity/v1` is **vendored from TraderX's own `golden-v1/basic`**, LF-exact
> (the source was CRLF — the W0.1 trap) and verifies against their published `bundleId`;
> `equity/v2` is authored and clearly labelled synthetic, carrying a long/short USD pair
> plus a EUR row.

**Tests:** ✅ `tests/test_integration_equity.py` (60 tests) — long/short mirrors; multiplier
applied exactly once (asserted against a multiplier that is **not 1**, the only way the test
can fail); currency/FX handling including the absent-currency case; the mark-echo guard;
malformed rows; I-17's exception-type regression class; plus the pipeline end-to-end in both
bundle versions and proof the Treasury pricers are unchanged.

---

### W1.5 — Wire the new instruments through the portfolio path · ✅ **DONE** (2026-09-17)

> **Delivered, with one deliberate deviation from the task list below.** Four of the five
> named targets landed as written. **`_price_by_type` did not**, and the reason is the
> substantive finding of this stage — see "The scenario dimension" below.
>
> **`engine/instruments/treasury.py`** — a new `BondConfig` covering both Treasuries, with
> a bill as the degenerate `coupon_schedule=()` case rather than a second type. It carries
> **its own `ZeroCurveConfig`** (like the swaption family, unlike `SwapConfig`'s curve
> indexes), which is what makes I-01's failure mode *structurally* unreachable: there is no
> index to resolve, so there is no resolution step to forget.
>
> **Why a new module rather than reusing `engine.integration.bill`/`note`.** Dependency
> direction: `integration` imports `instruments`, never the reverse. Importing the
> integration pricers into the instrument layer to share ~20 lines of `exp(-r*t)` would
> couple `engine/instruments/` to the TraderX bundle format permanently.
> `TestAgreesWithTheIntegrationPricers` prices the *same* instrument through both paths and
> asserts agreement to the cent — verified **exact to the bit** on both the bill NPV and the
> note's dirty NPV, and on the ACT/ACT (ICMA) accrued (`1,857.142857`). A drift fails a test.
>
> **⚠ The scenario dimension is refused, not broadcast.** `npv_cube` is
> `[Scenarios, TimeSteps, Trades]` and feeds VaR/ES. A bond priced against one deterministic
> curve has **no such column**: the only way to fill it is one t=0 number broadcast across
> every entry. That was implemented and measured through the real `price_portfolio` — on a
> $100k bill it returns **VaR 0.00 and ES NaN**, a position reading as risk-measured whose
> risk was never modelled. So `_price_by_type` **raises** naming the trade, and
> `scenario_risk=False` gives a caller real `base_npv`/`base_npv_per_trade`/`greeks` with
> `risk` **empty** (not zero-filled) and `PortfolioResult.scenario_risk_available` saying so
> on the result. Registered as **[I-24](../known-issues.md#i-24)**.
>
> **One real bug found and fixed:** **[I-25](../known-issues.md#i-25)** — a **scalar** Greek
> crashed the HTTP serializer (`TypeError: 'float' object is not iterable`). Every pre-W1.5
> Greek is a per-pillar *vector*; a bond's delta/gamma are the first 0-d arrays in the
> codebase. The job priced correctly and then 500'd on the way out. Found by reading the
> conversion code, not by a test — nothing existed to exercise the path.
>
> ⚠ **Process finding: run the venv, not the system Python.** Much of this stage was run
> against the system interpreter, where `tests/test_api.py` and
> `tests/test_integration_schema.py` are uncollectable (no `pydantic`, no `jsonschema`) —
> **46 tests silently absent**, and a full-suite count of 1,663 against the venv's 1,709 that
> looked like a regression and was purely environmental. `.venv/` has always had both.
> Recorded in [I-25](../known-issues.md#i-25) because the lesson outlives the bug.
>
> **Four wrong implementations were patched in and verified to fail** (working rule 3):
> deleting the Greeks branch (**19 of 19** I-01 tests fail, and the broken version raises
> *no error* — exactly I-01's signature); broadcasting a constant column (**4 of 5** fail);
> placeholder `gamma`/`theta = 0.0`; and returning the **clean** instead of dirty NPV.
>
> **The last two each found a gap in my own tests rather than the code** — the same shape as
> W1.6's decorative-`fractionDecimals` bug. The placeholder Greeks passed **27 of 28**, and
> the clean/dirty swap passed **67 of 68** (caught only by the integration cross-check, not
> by any bond-only test). Both gaps were closed by pinning each quantity to an *independent
> recomputation* rather than to a sign or a presence check, and the new tests verified to
> fail against those implementations.

**Original plan text follows.**

Add to the `TradeConfig` union, `_price_by_type`, `_base_npv_per_trade`, `_compute_all_greeks`,
and the API schemas.

> **⚠ Regression class to avoid — this exact bug already happened (I-01).** A new instrument
> type reaching `_compute_all_greeks` without its curve resolution gets **silently skipped**,
> returning no Greeks and no error. The prior test even asserted the skip was "by design."
> **For every new type, add a test asserting its Greeks are present**, and verify that test
> fails if the branch is removed.

---

### W1.6 — The contract interface · ✅ **DONE** (added and delivered 2026-09-16)

> **All four sub-tasks landed, ahead of W1.5 as planned.** The boundary is now
> reachable over HTTP: `GET /eod/capabilities` finally routes W0.9's function,
> and W0.8's durable lookup has an endpoint with its four states distinguished.
>
> **W1.6.1** `engine/integration/terms.py` — `traderx.instrument-terms.v2`
> accepted alongside v1, **checked independently of the bundle version**. The
> optional `accrualBasis` block is parsed and its enums pinned to an exact
> accepted set; an unrecognized `dateBasis`/`settlementAdjustment`/`rounding`,
> a future `accrual-basis.v2`, or a v1 artifact carrying the block at all is
> **refused**. `fractionDecimals` is threaded into the reconciliation tolerance,
> so the exporter's declared precision now *derives* the check rather than a
> constant standing in for it.
>
> ⚠️ **The strictness rule rests on an unanswered question and is registered as
> [I-23](../known-issues.md#i-23) (status ASSUMPTION).** If TraderX adds enum
> values in place rather than versioning the schema, this refuses bundles they
> consider valid. It fails safe, but it is not a settled contract and must not
> be recorded as one.
>
> **W1.6.2** `schema.py` + `schema_version.py` — `resultSchema` and
> `capabilitySchema` on every published document, plus machine-readable JSON
> Schema (Draft 2020-12) **derived from the frozen vocabulary** rather than
> hand-written. The two versions are separate because a new pricer changes the
> capability document without touching the result's shape. `schema_version.py`
> is a dependency-free leaf breaking the `result` ↔ `schema` cycle at its
> narrowest point — the same shape as `engine/day_count.py` in W1.3.
>
> **W1.6.3** `pipeline.py` — `accrualSource` on the standalone `accruedInterest`
> outcome, matching the NPV payload's label. **`structural-zero` stays distinct**
> from an exported-fraction conversion; folding them together fails 4 tests.
>
> **W1.6.4** `engine/api/eod_routes.py` + `workload.py` — six routes, the
> canonical workload key, and an immutable attempt store. `engine.api` imports
> `engine.integration`, never the reverse, so the no-pricer invariant survives.
>
> **Six plausible-but-wrong implementations** were patched in and verified to
> fail (working rule 3) — and **one of them found a gap in the tests rather than
> the code**: an implementation that parsed and validated `accrualBasis` and
> then *ignored* it passed 59 of 59. A bug of omission produces no wrong output
> anywhere a parser test can see it.
> `TestFractionDecimalsActuallyReachesTheTolerance` was written afterwards,
> driving an end-to-end consequence (8 declared decimals must *refuse* the note
> with `ACCRUAL_MISMATCH`), and verified to fail against it.
>
> **One stale test was updated, not deleted.**
> `test_unsupported_terms_schema_is_rejected` used `.v2` as its example of an
> unsupported schema — correct at W0.2, and exactly what W1.6.1 is chartered to
> change. It now uses `.v99`, so the contract it protects is unchanged.

**Original plan text follows.**

**Why this was inserted, and why it jumps the queue.** TraderX's v5 review independently
reproduced every priced number on the shared fixtures, and their remaining blockers are
*all* interface, not pricing: terms v2, a versioned result/capability schema, an
`accrualSource` alignment, and the HTTP service. Their own next step — independent
validation, then connecting their local result intake — is **blocked on the contract
surface, not on more instruments.**

W1.5 is internal plumbing: it wires the new instruments into `price_portfolio`/`greeks` for
*this engine's own* callers. Nothing on TraderX's side consumes that path. So W1.5 delivers
no unblocking to the counterparty, while W1.6 unblocks their entire next work item.
**Sequencing W1.6 first is the higher-value ordering**, and it does not make W1.5 harder —
the two touch disjoint code.

| Task | Deliverable |
|---|---|
| **W1.6.1** | Accept `traderx.instrument-terms.v2` alongside v1, validating `accrualBasis`. **Terms version is independent of bundle version** — a v2 bundle may carry either terms version. |
| **W1.6.2** | `resultSchema`/`capabilitySchema` version on every published document, plus machine-readable JSON Schema for both. Emit the version *before* extending intake, so their validator can pin it. |
| **W1.6.3** | Carry `accrualSource` onto the standalone `accruedInterest` outcome, aligned with the NPV payload's label. **Keep `structural-zero` distinct from an exported-fraction conversion** — that distinction is W0.3's whole point and must survive the alignment. |
| **W1.6.4** | The EOD HTTP routes (`GET /capabilities`, submission) — this is where **W0.8** and **W0.9**'s unrouted function finally land. Fold W0.8 in here rather than leaving it stranded. |

**Acceptance:** TraderX's `note-structured-basis` bundle joins rather than raising
`TermsJoinError`, and their new pricing acceptance profile validates a priced result against
the published schema.

> **Note on their W0 validator.** They will keep the existing no-market profile (which
> correctly *rejects* priced output) and add a separate pricing profile rather than weakening
> it. That is the right call and matches working rule 3 — a validator relaxed to accept both
> would stop proving either.

---

## 4. W2 — Faithful USD-SOFR · **I-05** · blocked on D03/D04

**Do not start on assumed conventions.** Guessing produces confident wrong numbers — the exact
failure this whole design prevents.

**Required first (D03/D04):** both legs' schedules, explicit dates, fixed/float day counts,
calendars, business-day adjustment, payment lags, overnight compounding method, lookback,
lockout, observation shift, fixing calendar, fixing history reference. TraderX's SOFR fixture
already enumerates **13 missing terms** — that list *is* the agenda.

**Build as a SEPARATE builder.** Do not modify `build_vanilla_swap`: every swaption pricer
depends on its ACT/365 consistency with the simulation time axis, and the full suite pins that
behavior.

**Acceptance:** parity against a same-terms **ORE reference**, not my own test suite.

---

## 5. Deferred — explicitly out of scope, with reasons

| Item | Issue | Blocked on |
|---|---|---|
| Aged-swap correctness | **I-04** | Historical fixings not exported. Currently **flagged, not fixed** — warning only; exposure/VaR at aged steps carry known inaccuracy |
| Mid-coupon exercise proration | **I-06** | D16 settlement convention |
| Corporate bonds | I-07 | Credit/spread model. Treasury-discounted corporate is **not** credit pricing — refuse it |
| Listed options | I-07 | Vol surface + settlement/deliverable terms |
| TIPS / FRNs | — | Explicitly unsupported until separately specified |
| Cube artifact offload | **I-09** | Not blocking W0/W1 correctness |
| Worker device reporting | **I-12** | Low severity |

---

## 6. Execution order

```
W0.1 bundle+hash ─┬─ W0.2 terms join ─┬─ W0.3 normalize ─┐
                  │                   └─ W0.4 allowlist ─┤
                  └─ W0.7 identity ───────────────────── ├─→ SOFR refusal ★ first real result
                     W0.5 result schema ─────────────────┤
                     W0.6 market inputs ─────────────────┘
                     W0.8 durable lookup  (independent)
                     W0.9 capabilities    (independent)

W0.10 curve-index validation ✅ I-13/I-14 (ran ahead of W1, as planned)

W1.1 day count ✅ ─→ W1.2 bill ✅ ─→ W1.3 note ✅ ─┬─→ W1.6 contract interface ✅ DONE
                     W1.4 equity ✅ (refusal) ─────────┤    (terms v2, schema versions,
                                                      │     accrualSource, HTTP + W0.8)
                                                      └─→ W1.5 wire-through ✅ DONE (internal)
                                                           (BondConfig; no VaR/ES — I-24)

W2  ⛔ blocked on D03/D04
```

**W1.6 now precedes W1.5** (2026-09-16). TraderX's v5 review reproduced every priced number
and found no pricing defects; their remaining blockers are entirely interface. W1.5 wires
instruments into *this engine's own* `price_portfolio` path, which nothing on their side
consumes — so it unblocks nobody, while W1.6 unblocks their whole next work item. The two
touch disjoint code, so the reorder costs nothing.

**★ First real result — delivered.** The SOFR `CONVENTION_NOT_SUPPORTED` output needed no
pricer, only W0.1/0.2/0.4/0.5/0.7. It runs against the real fixture and is the substance of
[response v4](eod-contract-response-v4.md) §3.

**★ Second real result — delivered.** The bill prices: **+98,507.15 / −98,507.15** on the
delivered fixture, exact against ORE, with a fully reconcilable payload. This is the first
result carrying a *number* rather than a refusal, and the first thing TraderX can validate
against their own model rather than merely parse.

**★ Third real result — delivered.** The note prices: **+103,308.33 / −103,308.33** on the
delivered fixture, exact against an independent `ORE.FixedRateBond`, with accrued interest
reconciling to TraderX's own `0.018571` at the §2 derived tolerance. This completes the
W1 exit criterion's harder half — the case where our two systems' numbers actually had to
agree, rather than a single discount factor.

**★ W1.4 delivered as a refusal.** The equity case is resolved: understood, identified,
validated, and refused with the missing market input named. It is the first task in this
plan whose *correct* deliverable was "no number", and the reasoning is in **I-18**.

**★ v5 — priced results independently verified by TraderX.** Running our adapter against
their own files, they reproduced **every** number: bill ±98,507.15, note dirty ±103,308.33,
accrued ±1,857.10, +1bp sensitivity ∓15.28, with 529 focused tests passing. This is the first
time our two systems have agreed on a *price* rather than on a refusal. They also found two
real defects, both now fixed with regression evidence (§7).

**★ W1.6 delivered — the boundary is reachable.** Six HTTP routes under `/eod`, terms v2 with
a validated `accrualBasis` whose declared precision now derives the reconciliation tolerance,
versioned result/capability documents with machine-readable JSON Schema, and `accrualSource`
aligned across both places it appears — with `structural-zero` still distinct. W0.9's
capability function and W0.8's lookup, both stranded since W0, finally have endpoints.

**★ W1.5 delivered — bonds reach the portfolio path.** `engine/instruments/treasury.py`'s
`BondConfig` is in the `TradeConfig` union, `_base_npv_per_trade`, `_compute_all_greeks` and
the API schemas, pinned bit-exact against the integration pricers. **Bonds have no VaR/ES**
— a deterministic instrument has no scenario column, and the refusal is explicit rather than
a broadcast zero ([I-24](../known-issues.md#i-24)). Two bugs found and fixed on the way
([I-25](../known-issues.md#i-25) — a scalar Greek crashed the HTTP result serializer).

**Next:** W2 / USD-SOFR, still blocked externally on D03/D04. Equity *valuation* remains
blocked on a spot/FX source, which is a market-data decision rather than engine work. The
nearest unblocked engine work is a **bond scenario model** (I-24) — that is what would give
a bond VaR.

**Still open with TraderX, and now load-bearing:** the `accrual-basis` versioning question
from v4 §1.3 (new values in place, or a new schema version?). W1.6.1 implements the strict
reading — the exact value set is pinned and anything else is refused — so if they intend to
add values in place, that is a one-line widening they need to tell me about rather than
discover through a refusal.

---

## 7. Open items with TraderX

**All four v3 asks were answered in their v3 — closed:**

| Ask | Outcome |
|---|---|
| `.gitattributes` for hash-critical bytes | Implemented, **scoped to the fixture directory** — better than my repo-wide proposal. Pins exactly the hashed bytes, including cut/preimage files |
| Blank-accrued note fixture | Supplied as `compatibility/note-missing-accrual/`, long **and** short |
| Machine-readable accrual basis | Implemented as `traderx.instrument-terms.v2` + `traderx.accrual-basis.v1` |
| `synthetic` distinct from `assumed` | Confirmed |

**Open, asked in v4:**

1. **Push the compatibility work and send the commit SHA.** My submodule is pinned at
   `a102e498`; there is no `.gitattributes` at that commit and
   `scripts/test-state-YU18-checkout.py` is absent from my tree. **Their 99 tests and the Git
   checkout-filter proof are therefore their verification, not a shared one** — I will re-run
   both independently, as I did the golden hashes, rather than recording a pass on report.
2. **`accrual-basis` versioning — still unanswered, and now load-bearing.** Do new
   `dateBasis`/`settlementAdjustment` values land in `v1`, or force a `v2`? **W1.6.1 shipped
   the strict reading**: the exact value set is pinned and anything outside it — including a
   future `accrual-basis.v2` — is refused. If they add values in place, that is a one-line
   widening they must tell me about rather than discover through a refusal. Restated in
   [v6](eod-contract-response-v6.md) §2.3.
3. ~~**Confirm `accrualSource: "exported-fraction"`**~~ — **implemented** (W1.6.3), aligned
   across the standalone outcome and the NPV payload, with `structural-zero` kept distinct.
4. **Review the result shape** — now **machine-readable** at `GET /eod/schemas/result`
   (W1.6.2), so this can be a schema check rather than a document review.

**Blocking W2 only:** D03/D04 SOFR conventions.

**Answered by their package — closed:** `closingMark` is clean (`priceBasis:
clean-fraction-of-par`); `faceDenomination` / `signedFaceAmount` split confirmed.

---

## 8. Working rules

These are what the exchange actually established. They apply to every task above.

1. **Never silently approximate.** Refuse loudly; an explicit `unsupported` is recoverable, a
   plausible wrong number is not.
2. **Key decisions on terms, not on blanks.** A missing field and a structural zero look
   identical in CSV and mean opposite things.
3. **Regression tests must be verified to fail against the pre-fix code.** A test that passes
   either way proves nothing. Where a plausible-but-wrong fix exists (defaulting a curve,
   defaulting a day count), add a test that fails against *that* too.
4. **Distinguish "ORE can represent it" from "my engine prices it."** Conflating them is the
   overclaim to avoid.
5. **Fixed ≠ flagged.** A warning improves honesty, not accuracy. **I-04** stays FLAGGED.
6. **Read bytes in binary for anything hashed.** Line-ending translation silently breaks every
   hash.
7. **Identity never rides on array position.**
8. **Test the caller, not just the guard.** A unit test asserting a validator raises proves
   nothing about paths that never call it. **I-13** lived behind a passing test of exactly
   this shape: `_swap_curve_configs` was tested directly and correctly, while the pricing path
   that skipped it went unexercised. Every guard needs at least one test that reaches it
   *through the public entry point*.
9. **A green suite is evidence about the tests, not about the code.** 1,138 tests passed
   while I-13 sat in the default pricing path. Two of the three defects found in this exchange
   came from reading source and predicting a failure mode; none came from running the suite.
10. **Never report a stale or rounded-up test count.** This document has now carried three
    (824, 1092, 1175). Numbers that describe verification must be reproducible by running the
    thing, and a red suite is reported as red — an intermittently red suite trains people to
    ignore red, which is how I-13 survived.
