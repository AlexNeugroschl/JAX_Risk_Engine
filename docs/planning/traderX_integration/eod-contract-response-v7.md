# EOD contract — response v7

**From:** Alex (JAX Risk Engine side) · **Date:** 2026-09-17
**Re:** `docs/risk-integration/spec-kit-draft/` — the integration spec kit, reviewed
**Companion:** [Known Issues](../../known-issues.md) · [Integration Plan](traderx-integration-plan.md) · [Boundary reference](../../reference/eod-integration.md)

---

## 0. The short version

**I reproduced your four failures on my own checkout, at your commit, and they are all real
engine bugs.** Not harness artifacts, not environment differences. `check_current_api.py` ran
against `e4ca50b` here: 1 passed, 4 failed, same four. Thank you — A-02 and A-03 are the two
worst defects anyone has found in this boundary, and both are mine.

**Your spec kit is right about the requirements and wrong about the architecture.** FR-01
through FR-12 are a better statement of this boundary's obligations than anything in my own
planning docs, and I'm adopting them. But the *pack* is built as if the engine had no
integration layer, and it has had one since W0. The result is that TraderX has grown a second
workload key, a second attempt store, a second result document and a second pricer — and the
spec kit's architecture diagram blesses that arrangement rather than ending it.

**The core disagreement, stated once:** the kit proposes a "producer adapter" that constructs
canonical requests, and a "consumer" that validates returned identity and coverage. Those two
boxes already exist **inside my engine** as `engine/integration/`, and they are the tested
part. What TraderX needs is not to rebuild them on your side — it is an HTTP client with a
`while` loop.

**What I'm asking for:** delete `pricing_result.py`, `w0_result.py` and `fake_worker.py` from
the TraderX runtime overrides, and let `coordinator.py` call my `/eod` routes. I'll fix
FR-06/07/08 first so that call is worth making. Concrete sequence in §6.

---

## 1. The four failures — confirmed, diagnosed, and worse than your write-up says

I ran your starter suite unmodified:

```
$ python reference/traderX/docs/risk-integration/spec-kit-draft/acceptance/check_current_api.py \
    --engine C:/PROJECTS/JAX_Risk_Engine
Engine revision: e4ca50bd3f5179e230e589275c3a09a5a88422c8
test_A01_healthy_assumed_bill ... ok
test_A02_cached_destination_still_checks_submission_binding ... FAIL
test_A03_overlapping_retry_has_one_execution_owner ... FAIL
test_A04_USD_profile_rejects_EUR ... FAIL
test_A05_unknown_calculation_is_rejected ... FAIL
Ran 5 tests in 0.086s
FAILED (failures=4)
```

Your characterizations are accurate. Here is what each one actually is in my source, because
two of them are not the bugs they look like.

### 1.1 A-02 — the cache bypasses submission binding entirely (FR-07)

**This is the same class of defect as the one I self-reported in v6 §5.1, in a code path I
did not fix.** In v6 I fixed `AttemptStore.start` to raise `SubmissionIdConflict` when a
submission id is reused against a different workload. That guard works. The problem is that
[`eod_routes.py:251`](../../../engine/api/eod_routes.py#L251) **returns before ever reaching it**:

```python
if request.reuseExistingResult:
    existing = STORE.lookup(key)
    if existing is not None and existing.state == STATE_COMPLETED:
        return {... "reused": True, "result": existing.result}

try:
    attempt = STORE.start(key, submission_id=request.submissionId)   # the guard lives here
except SubmissionIdConflict as exc:
    raise HTTPException(409, ...)
```

The cache lookup is keyed on the **workload**, and the conflict guard is keyed on the
**submission id**. When the bill is already cached, the first branch hits and the second never
runs. So the defect is not "the guard is wrong" — the guard is fine. It is that **a cache hit
is served before identity is checked at all**, which is precisely your FR-07 wording: *"A
submission ID binds to one workload before any cache return."* You wrote the requirement to
catch exactly this ordering, and it caught it.

Severity: a coordinator that reuses a submission id across bundles gets **another bundle's
priced numbers** under its own id, with `"reused": true` and a `workloadKey` that does not
match what it asked for. The response is self-describing enough that a strict consumer would
catch it — but "the consumer might notice" is not a control.

**Fix:** move the binding check above the cache read. `STORE.start` must be consulted (or a
`check_binding(key, submission_id)` extracted from it) before `STORE.lookup` is allowed to
return. I'd rather extract the check than reorder, so the invariant is structural.

### 1.2 A-03 — two threads, one workload, two executions (FR-08)

Also real, and the diagnosis matters. `AttemptStore` is thread-safe: `start()` holds
`self._lock`, and the docstring claims *"two concurrent submissions of the same `submissionId`
must resolve to one attempt."* That claim is true and insufficient.

Your test submits twice with `reuseExistingResult=False` and **no submission id**. With no
submission id there is nothing to deduplicate on — `start()` mints a fresh `uuid4()` attempt
each time, both threads get distinct running attempts for the **same workload key**, and both
enter `price_bundle`. The lock protects the dictionary, not the computation.

So the real gap is that **the attempt store has no concept of an execution owner.** It tracks
"which attempts exist for this workload" (`_by_workload` is a *list*, deliberately, so a
second attempt never overwrites a first) but nothing anywhere asks "is one already running?"

This is the requirement I most want to get right, because your FR-08 wording is careful in a
way that matters: *"One execution owner prices an attempt. Overlapping retries do not execute
that same attempt again. This is not a promise of exactly-once execution across arbitrary
crashes."* I can deliver that. What I cannot deliver, and will not claim, is cross-process
exclusion — see §4.2.

**Fix:** a per-workload running-attempt index in `AttemptStore`, consulted under the existing
lock. A second submission for a workload with a live running attempt joins that attempt
rather than starting one. I'll return `200` with the same `attemptId` when the first finishes
in time, `202` with `state: "running"` when it does not — your test already accepts both,
which tells me you thought about the same case.

**Your test's own caveat is correct and I want it preserved:** the bounded observation
interval proves one schedule, not all of them. I'll add the deterministic-barrier variant on
my side as a regression test rather than relying on timing.

### 1.3 A-04 and A-05 — one bug, not two (FR-06)

These are the same defect and it is embarrassingly simple. `EodSubmissionSchema` accepts
`calculations` and `reportingCurrency`. `_key_for` faithfully threads both into the workload
key. And then:

```python
result = price_bundle(bundle, request.marketInputs)
```

`price_bundle` **takes no such parameters.** Grepping `engine/integration/pipeline.py` for
`reporting_currency` returns nothing; `calculations` appears only in the internal
`_refused_calculations` / `_unpriced_calculations` helpers, which are about *coverage
accounting*, not request filtering. Both options are accepted, hashed into the cache key, and
dropped on the floor.

The consequence is worse than "ignored". Because they are in the workload key, a `EUR` request
and a `USD` request produce **different cache entries containing byte-identical USD results**.
The cache faithfully partitions on a distinction the pricer does not honour. Your A-04 output
shows it: `"currency": "USD"` throughout, `200 OK`, no warning, no refusal.

This one is straightforwardly my fault and it is a **fifth defect found by source review that
my green suite did not catch** — the same shape as v6 §5.2. My tests asserted that the fields
parse. Nothing asserted a consequence.

**Fix, and a question.** Rejecting is easy; rejecting *correctly* needs one decision from you.
My capability document currently advertises `calculations.names`, `calculations.statuses` and
`calculations.mode: "partial"` — but **no currency field at all**. There is nothing for a
consumer to pin. So:

- `calculations`: reject any name not in `CALCULATIONS` with `400 UNKNOWN_CALCULATION`. The
  allowlist already exists in `engine/integration/result.py`; this is wiring.
- `reportingCurrency`: I'll add `reportingCurrencies: ["USD"]` to the capability document and
  reject anything else with `400 UNSUPPORTED_REPORTING_CURRENCY`. **Your D-02 proposes exactly
  this and I'm accepting it** — but note that this makes the capability document the thing
  your validator pins for currency, so it needs to land before you write that check.

A genuinely honest alternative is to **remove both fields from the request schema** until the
engine can honour them, since an option that is always rejected is an option that does not
exist. I prefer advertising USD-only, because it gives you a versioned place to watch for the
day a second currency appears. Flagging the alternative because your FR-06 would be satisfied
by either.

---

## 2. Where the spec kit is right

I want to be specific here rather than gracious, because several of these are things I should
have written and did not.

**FR-01 through FR-12 are adopted.** They are a sharper statement of this boundary than my own
planning docs. Three in particular:

- **FR-04** (*"refusal, missing input and failure cannot disappear into an aggregate"*) is the
  principle behind my coverage block, stated better than my docstring states it.
- **FR-05** (*"Missing observations never trigger silent substitution"*) is the rule
  `marketInputs` exists to enforce, and `fallbackOnMissingInputs: False` is my machine-readable
  answer to it. A-01 passing is the evidence.
- **FR-11** (*"Cache identity includes every input or implementation version that changes
  results"*) is what `workload_key` was built for, and A-04 shows the inverse failure mode you
  did not name: **the key can be more discriminating than the pricer**, which turns a correct
  cache into a liar. Worth adding to FR-11 as a second clause.

**NFR-01 — "no producer-specific imports inside canonical pricing/risk code" — is already an
enforced invariant, not an aspiration.** `engine/integration/` imports no FastAPI, no Pydantic,
no JAX and no simulation pricer, and
`tests/test_integration_pipeline.py::TestPackageImportsNoSimulationPricer` fails the build if
that changes. This is why the HTTP routes live in `engine/api/` rather than in the integration
package. You can pin this one as satisfied today.

**NFR-02 — "matching two paths that share a bug is insufficient numerical evidence" — is the
best sentence in the pack** and it is the argument against the current arrangement. The
±103,308.33 / ±1,857.10 / ∓15.28 agreement we reached in v5 is *two independent
implementations agreeing*, which is real evidence. The moment `pricing_result.py` becomes a
runtime fallback rather than a test reference, that evidence evaporates. Your architecture doc
says this explicitly — *"Independent test formulas are intentional and must never become a
runtime fallback"* — and I'm holding you to it in §3.

**Your process discipline is right and I'm matching it.** "Known defects FAIL, never xfail or
silently skip"; results record the tested commit; no relaxing expected values to match current
bugs. All four failures above stay failing in my tree until fixed, and I'll attach
failing-before/passing-after output per your tasks list.

**D-01 and D-04 are accepted as proposed.** One authoritative pack, TraderX pins a revision and
owns its producer profile. Local single-worker server-path submission for acceptance; remote
staging, auth and byte limits are separate work that this draft does not authorize. Agreed on
both, including the explicit non-authorization — nobody is bringing up GKE off this document.

---

## 3. Where it doesn't work — the duplication is the actual problem

### 3.1 What the architecture diagram legitimizes

Your diagram has four boxes between producer and engine models: *producer adapter*,
*canonical engine request/service*, *identity/validation*, *result envelope / attempt store*.
Read as a description of responsibilities, it is correct. Read as a map of **where code
lives**, it has already produced this on the TraderX side:

| TraderX override | Lines | What it duplicates in my engine |
|---|---:|---|
| `pricing_result.py` | 211 | A second Treasury pricer, with its own `Decimal` bill/note economics |
| `w0_result.py` | 169 | A second result-document definition (`CALCULATIONS`, `STATUSES`) |
| `fake_worker.py` | 162 | A second attempt store, with its own submit/lookup lifecycle |
| `http_adapter.py` | 130 | A second `workloadKey` derivation (`bundleId` + profile, sha256) |
| `coordinator.py` | 366 | Submission, retry and reconciliation orchestration |
| `instrument_terms.py`, `market_inputs.py`, `bundle.py` | 570 | Terms decode, market-input handling, bundle/manifest verification |

That is ~1,600 lines reimplementing what `engine/integration/` already does in 5,400 tested
lines. The two are not equivalent and cannot be made equivalent by review, because **they are
different programs**. Two concrete divergences, both live today:

- **`workloadKey` means two different things.** Mine
  ([`workload.py:87`](../../../engine/integration/workload.py#L87)) is a canonical hash over bundle
  id, cluster epoch, session date, resolved market inputs, calculations, reporting currency,
  mapping version, engine version, result schema, precision and terms hash. Yours
  (`http_adapter.py:18`) is `digest({bundleId, profile})`. Yours cannot distinguish two
  submissions of the same bundle against different curves. Mine can. Neither is wrong for its
  own purpose; having both **named the same thing** on either side of one wire is how a cache
  hit ends up meaning different things to the two parties.
- **`pricing_result.py` hardcodes two bundle hashes** as `SUPPORTED_BUNDLES` and reproduces my
  exact strings — the `NOT_SUPPLIED` warning text, `UNSUPPORTED_DETAIL`, the `flat-3pct-v1`
  provenance block — as literals. Every one of those is a copy that drifts the next time I
  change a message. It is a faithful golden fixture for `e7246e1`, and as a **test reference**
  that is exactly right. As anything else it is a fork.

Your own architecture doc forbids this: *"An adapter may convert a documented percent into a
decimal or map an enum. It must not invent missing financial terms or maintain an alternative
schedule/pricer."* `pricing_result.py` maintains an alternative pricer. **I agree with your
rule and I'm pointing out that your own overrides break it.**

### 3.2 The reuse audit asks the wrong side to do the work

`architecture.md` has a five-row table of "Audit required" and assigns it to me, tracing
`engine/integration/pipeline.py`, `engine/instruments/treasury.py`,
`engine/portfolio/request.py` and `engine/api/eod_routes.py`. I'll answer it directly, because
the answer is short and it redirects the audit:

| Operation | Canonical symbol | Duplicate? |
|---|---|---|
| Bill valuation | `engine/integration/bill.py` | No — single implementation |
| Note schedule + valuation | `engine/integration/note.py` | No — single implementation |
| Accrued reconciliation | `engine/integration/note.py` (exported vs. structural-zero, separated) | No |
| Rate sensitivity | `engine/integration/note.py`, bumped revaluation, `zero-curve-parallel` | No |
| Market resolution | `engine/integration/market_inputs.py` | No — one provenance-preserving path |

**There is no duplication inside my engine along the EOD path.** `engine/portfolio/request.py`
and `engine/api/routes.py` are the *Monte Carlo portfolio* path — a different service
(`/portfolio/price`, async, 4096 scenarios, ~52s) that shares no code with `/eod` and prices
no Treasuries. It is not a second implementation of the EOD economics; it is a different
product. `engine/instruments/treasury.py` belongs to that path and is **not** what `/eod`
calls.

So A-07 (*"trace both entry points to canonical service"*) is testing a hypothesis that is
false. The duplication your kit correctly senses is real, but it is **across the boundary, not
inside my engine** — it is the table in §3.1. I'd rewrite A-07 as: *prove the TraderX
coordinator obtains every financial number from the engine service, and that no TraderX module
outside `tests/` computes one.* That is a grep, and it is enforceable in CI.

### 3.3 The kit is written as though `engine/integration/` did not exist

`contracts.md` says *"Reuse and pin the engine's published result/capability schemas after
review. Do not invent competing JSON shapes in this draft"* — correct, and already satisfiable:
`GET /eod/schemas/result` and `GET /eod/schemas/capabilities` serve machine-readable Draft
2020-12 JSON Schema, versioned `jaxrisk.eod-result.v1` and `jaxrisk.eod-capabilities.v1`. You
name both identifiers, so you found them. But the kit then specifies a *release manifest*
pinning "schema bytes/hash, engine build identity, mapping version" as new work, when
`capabilities()` already returns `engineVersion`, `mappingVersion`, `deliveryStage`,
`bundleSchemas`, `termsSchemas`, `accrualBasisSchemas` and the schema versions **derived from
the running engine on every call**, so it cannot go stale. The manifest you want is one
`GET` away from existing. What's genuinely missing is a *pinning* step on your side and the
additive-field policy you raise — which is a real gap, see §5.

Same pattern in `acceptance.md`: A-08 (enumerate every item/calculation including mixed
good/refused cases) is the coverage block, which ships today with `allOutcomesAccountedFor`
and `allApplicableComputed` and is validated in my suite. A-09's durability half —
completed and failed attempts surviving restart — landed in W1.6.4 via `ResultStore`; it is
the *running*-state half that is open ([I-08](../../known-issues.md#i-08)). A-10 is what
`workload_key` is for. These aren't "planned"; they're delivered and untested **by you**,
which is a different and much cheaper task.

**The distinction I'm drawing:** your kit treats the boundary as unbuilt and specifies it from
scratch. The boundary is built and has four bugs. Those are very different pieces of work, and
conflating them is what produced 1,600 lines of parallel implementation on your side.

---

## 4. Two places I won't follow the kit

### 4.1 The "consumer validates returned identity and coverage" box — yes, but not by rebuilding

`architecture.md` assigns the consumer *"Validate returned identity/schema/coverage and preserve
received bytes."* I agree completely, and it is the one new thing TraderX genuinely needs to
build. But it is a **validator**, not an engine: fetch `/eod/schemas/result`, pin the version,
assert the received document conforms, assert `workloadKey` matches what you submitted, assert
`coverage.allOutcomesAccountedFor`, retain the original bytes. That is maybe 80 lines and it
imports no financial logic. `pricing_result.py`'s 211 lines of `Decimal` cashflow arithmetic
are not that, and should not be reached from a runtime path.

Keep it as the **acceptance reference** it was built to be — A-12's "independent bill/note
references" is a legitimate and valuable role, and it is the reason our v5 numerical agreement
means something. Move it under `tests/`, mark it fixture-bound to `e7246e1`, and let it fail
loudly when it diverges. What it must not be is the thing the coordinator calls when the
engine is unreachable.

### 4.2 D-03, active-attempt recovery — I'll implement a policy, not a claim

Your D-03 asks for *"explicit interrupted/recoverable status or boot/lease protocol"* and
notes that terminal persistence alone is insufficient. Correct, and I want to be precise about
what I will and won't deliver, because this is where over-claiming is tempting.

Today: terminal attempts (completed and failed) are durable through `ResultStore` and survive
restart. **Running state is memory-only and deliberately so** — publishing an in-flight
computation would make it discoverable as a finished answer. After a restart, a job that was
running reports `UNKNOWN_WORKLOAD`.

Your requirement is that this must not be indistinguishable from "never submitted". Agreed —
that's a real hole, and `UNKNOWN_WORKLOAD` for interrupted work invites exactly the duplicate
overnight batch the four-state lookup exists to prevent. I'll add a durable
**accepted-attempt record** written at `start()` (identity only, never a result) and a boot
sweep that marks orphaned running attempts `interrupted` with a distinct lookup state.

**What I will not claim:** `contracts.md` says *"Do not claim distributed exactly-once
semantics from a process-local lock or filesystem rename."* That is the right warning and it
binds my §1.2 fix too. The FR-08 fix is a `threading.Lock` in one process. It is **not**
cross-process exclusion, and two engine processes against one store will still double-execute.
Single-worker is the only configuration I'll certify, and I'd like that written into the
profile rather than left as a footnote — per your D-04, which already scopes it that way.

---

## 5. What I need from you

1. **The `accrual-basis` versioning question is still open.** Third time asking (v4 §1.3, v6
   §2.3). When a real calendar arrives, does `accrual-basis.v1` **gain values** or become
   `.v2`? I've implemented the strict reading — anything unrecognized is refused, including a
   future `.v2`. Your `traderx-profile.md` says *"new settlement/calendar semantics require an
   agreed version rather than silently changing an enum's interpretation,"* which reads like
   agreement with the strict reading. If that's a yes, say so and I'll close it.

2. **The additive-field policy — you're right and I need to answer it.** `contracts.md`:
   *"Additive-field policy must be explicit; do not assume our existing exact-shape validator
   accepts additions."* My published result schema does not currently state whether unknown
   fields are permitted, and your validator does exact-shape matching. That is a live
   incompatibility: the next field I add breaks you. **My proposal:** the result schema
   declares `additionalProperties: false` at the document root and consumers may pin
   exact-shape, meaning **any new field is a schema version bump**. Conservative, and it makes
   your existing validator correct rather than lucky. Say if you'd rather have additive
   tolerance instead — but we need one answer, not two assumptions.

3. **Still waiting on the submodule SHA.** My pin is `a102e498`. Unchanged from v6 §6.

4. **Where does the pack live?** D-01 says one authoritative repository. My view: the
   *requirements* (spec.md, contracts.md) are genuinely joint and belong wherever we both pin
   them. The *schemas* must be engine-owned and served, because a schema that is copied is a
   schema that drifts — that is the §3.1 lesson in miniature. The TraderX profile is yours.
   If you want the pack in TraderX, the schemas stay served from `/eod/schemas/*` and the pack
   references them by URL and version rather than embedding them.

---

## 6. Proposed sequence

Ordered so that each step makes the next one worth doing. Steps 1–3 are mine and unblock
everything; nothing on your side needs to wait on step 4.

| # | Step | Owner | Done when |
|---|---|---|---|
| 1 | Fix FR-06 — reject unknown calculations; add `reportingCurrencies: ["USD"]` to capabilities and reject non-USD | Alex | A-04, A-05 pass |
| 2 | Fix FR-07 — submission binding checked before cache return | Alex | A-02 passes |
| 3 | Fix FR-08 — per-workload execution owner; second submission joins the running attempt | Alex | A-03 passes, plus a deterministic-barrier regression test |
| 4 | Answer §5.1 (accrual-basis versioning) and §5.2 (additive-field policy) | Yaakov | Recorded as decisions with date |
| 5 | Durable accepted-attempt record + boot sweep + `interrupted` lookup state | Alex | A-09 passes against a true process restart |
| 6 | `coordinator.py` calls `/eod/price` and `/eod/results/by-workload/{key}`; retire `fake_worker.py` and `http_adapter.py`'s `workload()` | Yaakov | One workload key on the wire |
| 7 | Move `pricing_result.py` / `w0_result.py` under `tests/`, fixture-bound, no runtime import path | Yaakov | CI greps prove no TraderX runtime module computes a price |
| 8 | Consumer-side validator: pin `jaxrisk.eod-result.v1`, assert conformance, `workloadKey` match, coverage completeness, retain bytes | Yaakov | A-08, A-11 pass against the served schema |

**Steps 1–3 first, deliberately.** Every one of them is a bug you found, and asking you to
rewire `coordinator.py` onto an interface that ignores request options and double-executes
would be asking you to build on a floor I know is broken.

I'll send failing-before/passing-after output for 1–3 against a named commit, and I'll add
your four cases to my own suite so they gate my builds rather than only yours — a test that
only the other party runs is a test I can break without noticing.

---

## 7. Restating the disagreement in one paragraph

Your kit and my engine agree on almost everything that matters: the requirements, the refusal
semantics, the provenance rules, the discipline about evidence. Where we differ is that the kit
specifies a boundary to be built, and the boundary exists — 5,400 tested lines of it, with four
real bugs. Building it a second time on the TraderX side has already produced a second workload
key, a second result document and a second Treasury pricer, and those will drift no matter how
carefully they are reviewed, because they are separate programs maintained by separate people.
The fix is not more specification. It is: I fix the four bugs, you delete the second engine and
call the first one, and `pricing_result.py` goes back to being the independent reference that
makes our numerical agreement mean something.
