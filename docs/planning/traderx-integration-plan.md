# TraderX Integration — Actionable Plan

**Date:** 2026-09-15 · **Owner:** Alex (JAX Risk Engine side)
**Status:** Working plan. Supersedes nothing; it consolidates the agreed outcome of the
four-document exchange into executable tasks.

**Source documents (the negotiation, in order):**

1. [EOD Contract Proposal](eod-contract-proposal.md) — my opening position
2. `firstEODProposal.md` → TraderX's opening
3. [Response v2](eod-contract-response-v2.md) — my reply to their corrections
4. `eod-response-to-alex-v2.md` → their answers + the YU18 package
5. [Response v3](eod-contract-response-v3.md) — my reply; hashes verified

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

### Verified working (evidence, not claims)

- **All 14 golden-vector hashes reproduce** from an independent stdlib implementation.
- **ORE reproduces the note fixture schedule exactly** (4 periods, ACT/ACT ICMA, unadjusted).
- **Accrued interest agrees to exported precision**: mine `0.0185714286` vs theirs `0.018571`.
- **1092 engine tests passed** at the W0.1–W0.5/0.7/0.9 milestone. 856 were passing before
  the W0 integration work (the "824" recorded earlier in this exchange was already stale).
  W0.6 adds a further 83, for 1175. Full suite ~13 min.

### Blocked, and on what

| Blocked | On | Can I start? |
|---|---|---|
| Faithful USD-SOFR pricing | D03/D04 convention agreement | ❌ External |
| Aged-swap correctness (**I-04**) | Historical fixings TraderX doesn't export | ❌ External |
| Priced bill/note results | W0 + W1.1–W1.3 build work | ✅ Mine |
| Everything else | Nothing | ✅ Start now |

---

## 2. W0 — Contract and refusal machinery

> **Status as of 2026-09-15: exit criterion met.** Every W0 task except **W0.8** is
> implemented, with 318 tests (295 in `engine/integration/`, 23 for the tail diagnostics in
> `engine/risk/var_es.py`). The SOFR case returns `CONVENTION_NOT_SUPPORTED` naming all 13
> missing terms, and bill/note return structurally valid results with `npv: unsupported`.
> See the per-task markers below and
> [the boundary's own doc](../reference/eod-integration.md).
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

### W0.8 — Durable result lookup · mitigates **I-08** · ❌ NOT STARTED

> Needs a persistent store plus an HTTP endpoint. The crash-safety semantics are the
> substance here and cannot be meaningfully tested against the current in-process job store.

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

Lookup returns the most recent **successful** attempt — never failed, partial, or in-flight.
Attempts are immutable and permanently addressable by `attemptId`.
`{"execution": {"reuseExistingResult": false}}` forces a fresh attempt **without overwriting**.

**Tests:** kill between artifact write and manifest publish → lookup finds nothing (no partial);
kill after publish → lookup finds complete result; second attempt doesn't overwrite first;
`reuseExistingResult:false` creates a new attempt, both retained; **different precision →
different workload key** (no cache reuse).

---

### W0.9 — `GET /capabilities` · ✅ DONE (as a function; not yet routed)

> `engine/integration/capabilities.py::capabilities()` returns the document, derived from the
> allowlist rather than hand-maintained. Wiring it to an actual HTTP route belongs with
> W0.8's endpoint work.

Return the supported (product × convention × calculation) matrix, precision/device profiles,
engine/model/build versions, and known-limitation flags, so the coordinator can determine
*before submitting* whether a bundle is priceable. This is what makes "no silent exclusions"
enforceable rather than aspirational.

---

## 3. W1 — First real pricers

**Exit criterion:** one exported Treasury case consumed end-to-end and returned as a
validated, identified financial result — TraderX's stated target.

---

### W1.1 — Day count as a per-instrument input ⚠ **prerequisite for everything below**

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

### W1.2 — Bill pricer

Single discounted cashflow: `faceAmount × redemptionFraction × P(valuationDate, maturity)`.
No coupon schedule, no Monte Carlo, no calibration. Serves as the transport/identity smoke
test with trivial pricing math.

**Tests:** ORE parity at matched terms; long/short signs; **maturity-date boundary**; accrued
= structural zero.

---

### W1.3 — Note pricer

Fixed coupons + bullet principal, discounted off the requested curve.

**Tests:** ORE parity; **accrued reconciles to TraderX's `0.018571`** (already independently
verified in Python — this test moves it into the engine); clean vs dirty reconciliation;
long/short; per-pillar `rateSensitivity` non-zero across the curve.

---

### W1.4 — Equity position pricer

`signedQuantity × multiplier × spot × fx`. **Note:** `SimulationConfig.equities` drives
correlated risk-factor *paths* — it is **not** a position pricer. This is new code.

**Tests:** long/short; multiplier applied **exactly once**; currency/FX handling.

---

### W1.5 — Wire the new instruments through the portfolio path

Add to the `TradeConfig` union, `_price_by_type`, `_base_npv_per_trade`, `_compute_all_greeks`,
and the API schemas.

> **⚠ Regression class to avoid — this exact bug already happened (I-01).** A new instrument
> type reaching `_compute_all_greeks` without its curve resolution gets **silently skipped**,
> returning no Greeks and no error. The prior test even asserted the skip was "by design."
> **For every new type, add a test asserting its Greeks are present**, and verify that test
> fails if the branch is removed.

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

W1.1 day count ⚠ ─→ W1.2 bill ─→ W1.3 note ─→ W1.5 wire-through ─→ ★ Treasury end-to-end
                    W1.4 equity ────────────┘

W2  ⛔ blocked on D03/D04
```

**★ First real result:** the SOFR `CONVENTION_NOT_SUPPORTED` output — needs no pricer, only
W0.1/0.2/0.4/0.5/0.7. Deliver this before any pricing work.

---

## 7. Open items with TraderX

**Wanted (non-blocking):**

1. **`.gitattributes` with `*.json -text` / `*.csv -text`** — highest value, smallest change.
   Without it their vectors are unverifiable from a default Windows checkout.
2. **Blank-accrued coupon-bearing note fixture** — expects `unavailable`, not `0.0`. Nothing
   currently exercises the wrong branch.
3. **Machine-readable accrual-basis caveat** on security terms, so fixture assumptions are
   structurally distinguishable from real conventions.
4. Confirm `origin: "synthetic"` as distinct from `assumed`.

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
