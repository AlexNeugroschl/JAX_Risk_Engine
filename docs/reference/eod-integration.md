# The TraderX EOD Integration Boundary (`engine/integration/`)

**Status: W0 delivered.** Contract and refusal machinery only — **this layer prices
nothing.** That is deliberate, not a gap: see [Why W0 prices nothing](#why-w0-prices-nothing).

Implements W0 of the [TraderX Integration Plan](../planning/traderx-integration-plan.md).

---

## Plain-language summary

TraderX sends this engine an end-of-day snapshot of their book: what they hold, what OTC
contracts they've booked, and a reference file describing each instrument's terms. The whole
package is *hash-pinned* — every file comes with a fingerprint, so any change to any byte is
detectable.

This layer is the front door. It:

1. **Checks the fingerprints.** If a single byte differs from what was sent, the bundle is
   rejected. No partial acceptance.
2. **Looks up each instrument's terms** and attaches them to the right positions.
3. **Converts units** — their coupon is a percentage, the engine wants a decimal; their
   accrued interest is a fraction of par, the engine wants signed currency.
4. **Refuses what it cannot faithfully price**, loudly and by name.
5. **Returns an identified result** for *every* row — including the refused ones.

Point 5 is the one that matters most. A refusal that doesn't say *which* position it refers
to is useless, and a refusal that doesn't say *why* can't be acted on. So every row comes
back with a stable id, its source identity, and — where it was refused — the exact list of
terms that were missing or unsupported.

**The governing rule: nothing is ever silently approximated.** An explicit "I can't price
this" is recoverable. A plausible wrong number is not.

---

## Why W0 prices nothing

The plan orders the work W0 → W1 → W2 for a specific reason: it "proves the entire
transport → identity → coverage → publication path while pricing math is still out of scope,
so contract bugs and pricing bugs never get debugged simultaneously."

So every model-driven calculation in a W0 result comes back `unsupported`. The machinery is
real and tested; the pricers arrive in W1.

**The one exception is `accruedInterest`**, and it is deliberate. Accrued interest at this
stage is a *unit conversion of an exported value*, not a model output — TraderX supplies
`accruedInterestFraction` and the terms supply enough to interpret it. So the engine can
answer it honestly, and does.

---

## The delivered result

Running the SOFR fixture through the pipeline produces the plan's "★ first real result":

```python
from engine.integration import price_bundle

result = price_bundle("tests/fixtures/traderx-eod/sofr/v2")
```

```jsonc
{
  "itemId": "f81d34ed56ee436b86c726d9305fd984",
  "sourceIdentity": {
    "kind": "contract",
    "accountId": "22214",
    "clusterEpoch": "synthetic-shared-examples-v1",
    "contractId": "SW-3"
  },
  "refusal": {
    "reason": "CONVENTION_NOT_SUPPORTED",
    "detail": "instrument terms are incomplete: the export enumerates 13 required term(s) it does not supply...",
    "missingTerms": [
      "businessDayAdjustment", "calendar", "fixedPaymentLagBusinessDays",
      "fixedSchedule", "fixingCalendar", "fixingHistoryReference",
      "floatingDayCount", "floatingPaymentLagBusinessDays", "floatingSchedule",
      "lockoutBusinessDays", "lookbackBusinessDays", "observationShift",
      "overnightCompounding"
    ]
  },
  "calculations": { "npv": { "status": "unsupported", "reason": "CONVENTION_NOT_SUPPORTED", ... } }
}
```

That list of 13 is not a diagnostic this engine composed — it is TraderX's own
`missingTerms`, carried through verbatim. Per the plan, **that list is W2's agenda**.

---

## Module map

| Module | Task | Responsibility |
|---|---|---|
| [`bundle.py`](../../engine/integration/bundle.py) | W0.1 | Read + hash-verify a v1/v2 bundle |
| [`terms.py`](../../engine/integration/terms.py) | W0.2 | Join `instrument-terms.json` onto rows |
| [`normalize.py`](../../engine/integration/normalize.py) | W0.3 | Source units → engine units |
| [`conventions.py`](../../engine/integration/conventions.py) | W0.4 | Allowlist + refusal path |
| [`result.py`](../../engine/integration/result.py) | W0.5 | `RiskResult` + per-calculation coverage |
| [`market_inputs.py`](../../engine/integration/market_inputs.py) | W0.6 | Explicit market-input mode; no silent fallback |
| [`identity.py`](../../engine/integration/identity.py) | W0.7 | Opaque `itemId` + source identity |
| [`capabilities.py`](../../engine/integration/capabilities.py) | W0.9 | Supported matrix |
| [`pipeline.py`](../../engine/integration/pipeline.py) | — | Composition of the above |

**This package imports no pricer, no ORE builder, and no curve** — in fact no ORE, JAX, or
even NumPy at all. That is enforced by a test
(`TestPackageImportsNoPricer::test_no_pricer_ore_or_jax_import`), not just asserted here:
refusal has to happen *before* any pricing object is constructed, because constructing one is
what applies the wrong conventions. An import of `engine.models.ore_builders` would mean that
ordering is no longer structurally guaranteed.

A side benefit of the same constraint: all 295 tests in this layer run in well under a
second, because none of them loads a numerical runtime. (The 23 tail-diagnostic tests live
in `tests/test_var_es_diagnostics.py` instead, since they exercise `engine/risk/var_es.py`
and do need JAX.)

---

## W0.1 — Bundle ingestion and hash verification

Each artifact is verified against **its own** `sha256`. The manifest's `cut.cutSha256` is a
different thing — it identifies the consensus cut — and a CSV is never compared against it.

**Empty is not missing.** A `contracts.csv` with `rows=0` and only a preamble is a valid
bundle carrying zero OTC coverage (the bill and note fixtures are exactly this). A
`contracts.csv` absent from disk is an integrity failure.

### ⚠ The CRLF trap

**This is live in this repository right now.** Hashes are over *committed bytes*. A Windows
checkout with `core.autocrlf=true` (Git for Windows' installer default) silently rewrites
LF → CRLF for anything it considers text — including `.json` and `.csv`. Every artifact hash
then fails, for a reason nothing in a bare error message points at.

It already broke TraderX's own verifier, and it has corrupted the vendored
`reference/traderX/` checkout here: every one of its 15 fixture artifacts fails to verify
as-is, and every one reproduces its manifest hash exactly after LF normalization.

Two decisions, pulling in opposite directions on purpose:

1. **Bytes are verified exactly as read, in binary.** No normalization, and deliberately no
   `tolerate_crlf` escape hatch. A file whose committed bytes were LF and whose on-disk
   bytes are CRLF is *not* the artifact the manifest pins.
2. **The failure explains itself.** On mismatch the loader tests whether LF-normalized bytes
   *would* have matched, and if so names CRLF translation and gives the fix. The normalized
   digest is used only to explain; it never satisfies the check.

The second is what makes the first survivable. Strictness without a diagnostic is what
produced the original confusion.

**The repository-level fix**, per the plan's §7 "highest value, smallest change", is
[`.gitattributes`](../../.gitattributes):

```gitattributes
*.json -text
*.csv  -text
```

Test fixtures live in `tests/fixtures/traderx-eod/` with committed LF bytes, verified by
`test_our_own_fixtures_are_committed_with_lf`. Note `.gitignore` has blanket `*.csv`/`*.json`
rules, so that directory carries explicit negation rules — without them the fixtures would be
silently untracked and a fresh clone would fail.

---

## W0.2 — Terms join

Two different join keys, because securities and contracts have different identity:

- **Securities join by security identity alone.** One `UST-NOTE-20261215` terms entry serves
  both the long account's row and the short account's — the terms describe the *instrument*,
  while amounts and signs stay in the position rows.
- **Contracts join by `contractId` + `clusterEpoch`.** A contract id is unique only *within*
  its epoch; joining on the id alone would silently match a contract from a different epoch.

**`missingTerms` is an authoritative refusal input, not a hint.** It is carried verbatim, in
supplied order. This layer does not judge whether the listed terms matter, does not fill
them, and does not reorder them.

**v1 bundles have no terms artifact.** Not a defect — the older version. Every row comes back
unjoined with reason `NO_TERMS_ARTIFACT`, and every instrument needing terms is `unsupported`.

---

## W0.3 — Unit normalization

| Source field | Source unit | Normalized |
|---|---|---|
| `coupon` | annual **percent** (`4.0`) | decimal (`0.04`) |
| `closingMark` | clean, fraction of par | kept as fraction, echoed as `observedCleanPrice` |
| `accruedInterestFraction` | fraction of par | `× signed face` → signed currency |
| `quantity` | signed face | `signedFaceAmount` |

`faceDenomination` (100 in these fixtures) **does not rescale** the CSV quantity, which is
already signed currency face. Applying it would be a 100× error.

### The zero-coupon rule — key on terms, never on the blank

A bill and a coupon-bearing note both show a **blank** accrued field. They mean opposite
things:

| Terms say | Accrued field | Result |
|---|---|---|
| Zero-coupon (`couponFrequency: NONE`) | blank | `0.0`, `provenance: structural-zero` |
| Coupon-bearing | blank | **`unavailable` + `ACCRUED_NOT_SUPPLIED`** — never `0.0` |
| Coupon-bearing | present | convert |
| **No terms artifact** | blank | **`unavailable`** — uninterpretable without terms |

The naive `blank → 0.0` is correct for the bill and **silently wrong** for the note. A
bill-only, long-only test suite passes while the note case is wrong — which is why
`test_coupon_bearing_blank_is_not_zero` exists and was
[verified to fail](#regression-tests-verified-against-the-wrong-implementation) against that
implementation.

The fourth row is the subtle one: without terms there is nothing to key on, so the blank is
*uninterpretable*, not zero. A v1 bundle yields `unavailable` even for what is in fact a
bill, because the engine cannot know that from a v1 bundle.

**Accrued interest is signed for the position** — short → negative, consistent with NPV.
Multiplying by the *signed* face does sign and scaling in one step, so there is no separate
`sign()` factor to get wrong.

---

## W0.4 — Convention allowlist and refusal · closes part of [I-05](../known-issues.md#i-05)

### The bug this prevents

`engine/models/ore_builders.py::build_vanilla_swap` produces a generic term-IBOR swap:
`SimIndex6M`, ACT/365 on both legs, TARGET calendar, schedule derived from a tenor *string*.
A USD-SOFR booking is none of those — it is an overnight index, ACT/360, compounded daily in
arrears, on a US calendar, with explicit dates plus lookback/lockout/payment-lag terms that
signature cannot even express.

Routing it through that builder prices it **confidently and wrongly**. ACT/360 vs ACT/365
alone shifts every accrual factor by 1.389% — roughly **$1,906 on a $1mm 5Y fixed leg, about
46× a 1bp DV01**. And **no pre-existing test catches it**, because every test in this
repository builds its inputs with that same builder, so the wrong convention is applied
identically on both sides of every comparison and cancels out.

That is why the allowlist is an explicit **positive** list of what the engine implements,
not a blocklist of what it doesn't. A blocklist silently admits everything nobody thought to
add to it.

### Today's allowlist

| Dimension | Supported |
|---|---|
| `floatIndex` | `SimIndex*` (the generic term IBOR) — **USD-SOFR deliberately absent** |
| Leg day counts | `ACT/365` only |
| Overnight compounding | *nothing* — not implemented at all |
| Instrument types | `SWAP`, `TREASURY` |

Adding an entry here is a **financial assertion** and should come with a parity test against
an independent reference, not against this engine's own suite.

### Refuse to infer

A booking with no stated conventions is `unsupported` — *never* defaulted into the generic
builder. This is the case that most wants a default, because the builder would happily accept
it and return a plausible number.

### Accepted ≠ priced

`check_conventions` returning `None` means only that the *conventions* are supported. It is
not a statement that a pricer exists. A note passes the convention check and still reports
`npv: unsupported` with reason `NO_PRICER_AT_THIS_STAGE` — a different reason from
`CONVENTION_NOT_SUPPORTED`, because they are closed by different work. This is the plan's
working rule 4: "ORE can represent it" is not "my engine prices it".

---

## W0.5 — Result schema and coverage

Frozen calculation names: `npv`, `accruedInterest`, `rateSensitivity`, `rateGamma`, `theta`,
`vega`, `varEs`. Every item carries an explicit outcome for **every** one — an omitted
calculation would be indistinguishable from a forgotten one.

### The five statuses

| Status | Meaning |
|---|---|
| `ok` | Computed. A number is present. |
| `unsupported` | Cannot be faithfully priced. Refused, not attempted. |
| `unavailable` | An input the calculation needs was not supplied. |
| `failed` | Attempted and errored. |
| `not-applicable` | Meaningless for this instrument. |

`unsupported` vs `unavailable` carries the most weight downstream: the first is closed by
engine work or a convention agreement, the second by a better export. Conflating them tells
the coordinator to retry something that will never succeed, or to give up on something a
resend would fix.

**W0 never reports `failed`** — nothing is attempted, so nothing can error.

### The two flags answer different questions

- **`allOutcomesAccountedFor`** — did every item get *some* verdict? **True even if
  everything failed.** It is an internal-consistency check on the document: false means the
  engine lost track of an item.
- **`allApplicableComputed`** — did everything that *could* have a number get one? True only
  when `unsupported + unavailable + failed == 0`.

Every W0 result is `allOutcomesAccountedFor: true, allApplicableComputed: false`.

**`not-applicable` never counts against coverage.** Vega on a vanilla swap is not a gap — a
swap has no optionality, so there is no vega to miss. Counting it would make a complete
result look incomplete and train consumers to ignore the coverage block.

### Aggregates

Aggregates carry `coveredItemCount`, `totalItemCount`, `excludedItems[]` and **`complete:
false` whenever those differ**. A missing value is *excluded*, never summed as zero — a total
over a subset presented as a total is the most dangerous number in a risk report.

Cross-currency portfolios get per-currency aggregates with
`reportingCurrencyConversion: "NOT_APPLIED"` — never a blended scalar. The engine is given no
FX rates, and inventing them would be exactly the silent approximation this design refuses.

---

## W0.6 — Market input selection · closes part of [I-11](../known-issues.md#i-11)

**Explicit mode; no silent fallback.** A job says where its market data comes from:

```jsonc
"marketInputs": {"mode": "assumed-profile", "assumedProfileId": "flat-3pct-v1"}
```

Absent, or `mode: "package"` with no package → the **job fails** with
`MARKET_INPUTS_NOT_SUPPLIED`. It does not fall back to a default curve, a flat curve, or the
last curve it saw.

### Why this fails the job rather than refusing the item

Every other gap in this boundary produces a per-item refusal: an unmapped convention makes
*that instrument* `unsupported` and the rest of the portfolio still prices. Market inputs are
different in kind — they are not a property of any one instrument but the shared basis every
price is measured against. A run with no curve has nothing to price *anything* against, so
there is no partial result worth publishing, and publishing one would invite a consumer to
reconcile a total that was never computed.

`MarketInputsNotSupplied` is therefore deliberately **not** a `ConventionRefusal`.

The most dangerous fallback in the system would be `mode: "package"` with no package quietly
becoming an assumed curve: the result would carry **observed provenance it does not have**.
That branch raises, and a test asserts the message says so.

### Assumed profiles are named, versioned and registered

`assumedProfileId` resolves against a registry. A caller cannot hand over arbitrary curve
numbers — an assumed curve has to be a *thing with a name* that appears verbatim in the
result, so "what was this priced against?" is answerable from the published artifact alone.
A bare rate would make every assumed run look identical in the output while being different
in fact.

| Profile | Rate | Construction |
|---|---|---|
| `flat-3pct-v1` | 3% | `flat-constant` |

Changing a profile's numbers under a stable id is **not allowed** — add a bumped id instead,
or every result ever computed against the old one becomes unreproducible while still claiming
the same provenance. The `-v1` suffix is the guard, and a test pins it.

An unregistered id is refused, never resolved to a nearest match — which matters most while
exactly one profile is registered and "just use the only one" is the tempting shortcut.

### Curve provenance

`ZeroCurveConfig` gained an **optional** `provenance` field:

```jsonc
{"curveId": "flat-3pct-v1", "inputOrigin": "assumed",
 "construction": "flat-constant", "inputHashes": []}
```

Before it, that dataclass carried times and rates and nothing else — a flat 3% assumption and
a bootstrapped market curve were *the same object*, and no downstream code could tell them
apart. The field is metadata only: `ZeroCurve.from_config` reads times and rates, so
attaching provenance cannot move a number, and a test asserts that.

It defaults to `None` across all **69 existing construction sites in 24 files**, which are
untouched. `None` means *unstated*, which is honestly different from `inputOrigin: "observed"`
— nothing infers observedness from a missing provenance.

`inputOrigin` is one of `observed | assumed | mixed | synthetic`. **`synthetic` is kept
distinct from `assumed`** (open item §7.4 with TraderX): an assumed curve is a deliberate
modelling choice standing in for an observation, a synthetic one is fabricated test data.
Both are "not observed", but collapsing them loses the difference.

### `marketProvenance` is top-level

Every result computed against any non-observed curve carries top-level
`"marketProvenance": "assumed"` — not buried per-curve, because a consumer reading only the
summary must not be able to miss it. **One assumed curve makes the whole result assumed**: a
consumer cannot act on "mostly observed", so the conservative label is the honest one.

Omitting `market_inputs` is legal *only* at W0, which prices nothing. The result then reports
`marketProvenance: null` and warns — never `"observed"`, which would claim a market basis
that was never consulted.

### `measure` — the label a risk number is unactionable without

`VaR_95 = 2.1mm` means materially different things depending on what produced it, and nothing
in a bare float distinguishes them:

| Measure | What it is |
|---|---|
| `risk-neutral-pricing` | Exposure under the pricing measure. Correct for CVA/exposure/limits. **Not** a forecast of tomorrow's P&L. |
| `historical-forecast` | A calibrated real-world forecast of realised loss. What a capital process wants — **this engine does not produce it.** |
| `deterministic-stress` | A prescribed scenario's revaluation. No probability attaches to it. |

Everything this engine computes is `risk-neutral-pricing`. Reporting a risk-neutral exposure
where a consumer expects a historical forecast is a category error that no amount of numerical
accuracy fixes, which is why the label travels with the number.

The three constants are defined in **both** `engine/risk/var_es.py` and
`engine/integration/market_inputs.py`. That duplication is deliberate: `engine/integration/`
must not import `engine.risk`, which pulls in JAX and would break the
[imports-no-pricer invariant](#module-map). `TestMeasureVocabularyMatchesVarEs` pins the two
definitions together so they cannot drift.

### Tail statistics carry their own convergence diagnostics

`compute_risk_metrics` now returns two extra keys per percentile:

| Key | Meaning |
|---|---|
| `ES_99_tailCount` | Effective sample size — how many observations the ES mean actually averaged |
| `ES_99_standardError` | Monte Carlo standard error of that mean, `s / sqrt(n)` with `ddof=1` |

`ES_99 = 1,240,000` reads as a precise figure. With a standard error of 380,000 it is not one,
and before this nothing in the result said so. **At 99% over 10,000 scenarios roughly 100
observations carry the estimate** — 1% of the sample, all from the part of the distribution
the Monte Carlo sampled least.

`tailCount` counts the **strict value-based tail** (`pnl < -VaR`), i.e. exactly the
observations `expected_shortfall` averages — not a positional `floor(N*(1-p))` slice, which
disagrees whenever there are ties at the VaR boundary (the same distinction the module
already documents for ES itself).

**`standardError` is NaN, never 0.0, when `n < 2`.** With one tail observation the ES is
defined but its spread is not; returning 0.0 would read as "perfectly converged" for exactly
the case where the estimate is least trustworthy.

**This is purely additive.** Every pre-existing `VaR_*`/`ES_*` key and value is unchanged, so
`engine/api/schemas.py`, `engine/portfolio/request.py` and the ~8 consuming test files keep
working untouched. `include_diagnostics=False` returns exactly the pre-W0.6 key set. The
diagnostics reach the HTTP boundary with no schema change, since `RiskMetricsSchema` is a
generic `Dict[str, List[Optional[float]]]` that already maps NaN → `null`.

**What this does not do:** it does not make any estimate better. It makes the uncertainty
visible, which is the difference between a number a reader can weigh and one they must simply
trust.

---

## W0.7 — Identity · closes [I-10](../known-issues.md#i-10) at this boundary

Two things travel on every row — **both, not either**:

- an opaque **`itemId`** (SHA-256 of the canonical identity, truncated to 128 bits), stable
  across runs, processes and machines. Not Python's `hash()`, which is `PYTHONHASHSEED`-salted
  and differs per process;
- the **source identity block** it derives from — `{kind, accountId, security | contractId}`
  plus `clusterEpoch`.

An opaque id alone is unreconcilable without a side channel; a structured identity alone
invites consumers to re-derive keys with their own subtly different rules.

**Unsupported rows carry full identity too.** An unidentified refusal is useless. `ItemResult`
cannot be constructed without an identity, so this is structurally enforced rather than
merely intended.

**Item ordering is published as its own hashed artifact** (`itemOrder`), never inferred from
array position — the plan's working rule 7. An array's position carries no integrity
guarantee of its own.

---

## W0.9 — `capabilities()`

Reports the supported (product × convention × calculation) matrix so a coordinator can
determine **before submitting** whether a bundle is priceable. This is what makes "no silent
exclusions" enforceable rather than aspirational.

**Derived from the allowlist, never hand-maintained** — a stale capability document makes a
promise the engine no longer keeps. Known limitations (I-04, I-05, I-07) are advertised as
part of the capability surface: a consumer weighing an exposure profile needs I-04 *before*
submitting, not after reconciling.

---

## Not yet implemented from W0

| Task | Status | Why |
|---|---|---|
| **W0.8** durable result lookup | Not started | Needs a persistent store + HTTP endpoint; the crash-safety semantics are the substance and can't be meaningfully tested against the in-process job store ([I-08](../known-issues.md#i-08)). |

Unblocked — sequencing, not dependency.

**W0.6 caveat.** The refusal path, curve provenance, `marketProvenance`, `measure` and the
tail diagnostics are all in place, but no pricer consumes a resolved `MarketInputs` yet
because W0 prices nothing. `resolve_market_inputs` returns the profile and its provenance;
materializing it into a `ZeroCurveConfig` that a pricer discounts against is W1.2's first
task, and is the point at which `market_inputs` stops being an optional argument to
`price_bundle`.

---

## Tested by

| Test file | Covers |
|---|---|
| [`tests/test_integration_bundle.py`](../../tests/test_integration_bundle.py) | W0.1 — `TestHappyPath`, `TestEachArtifactAgainstItsOwnHash`, `TestCrlfTranslation`, `TestRowCountValidation`, `TestEmptyVersusMissing`, `TestManifestValidation`, `TestRequiredColumns`, `TestPreambleAgreesWithManifest` |
| [`tests/test_integration_terms.py`](../../tests/test_integration_terms.py) | W0.2 — `TestSecurityJoinAcrossAccounts`, `TestContractJoin`, `TestMissingTermsIsAuthoritative`, `TestDuplicateAndUnjoinable`, `TestV1HasNoTermsArtifact` |
| [`tests/test_integration_normalize.py`](../../tests/test_integration_normalize.py) | W0.3 — `TestUnitConversions`, `TestAccruedSignAndMagnitude`, **`TestZeroCouponRule`**, `TestMalformedInput`, `TestRoundTripThroughTheResult` |
| [`tests/test_integration_conventions.py`](../../tests/test_integration_conventions.py) | W0.4 — `TestSofrFixtureIsRefused`, **`TestRefusesToInfer`**, `TestDayCountAllowlist`, `TestFloatIndexAllowlist`, `TestAllowlistedSwapIsAccepted` |
| [`tests/test_integration_result.py`](../../tests/test_integration_result.py) | W0.5 — `TestCoverageSumsToItemCount`, `TestNotApplicableNeverCountsAgainstCoverage`, `TestAggregatesAreCompleteOrSayOtherwise`, `TestCrossCurrency`, `TestSensitivityPayload` |
| [`tests/test_integration_market_inputs.py`](../../tests/test_integration_market_inputs.py) | W0.6 — **`TestNoSilentFallback`**, `TestAssumedProfileResolves`, `TestCurveProvenance`, `TestTopLevelMarketProvenance`, `TestMeasureIsReported`, `TestMeasureVocabularyMatchesVarEs`, `TestBundleDeclaredMarketStatus` |
| [`tests/test_var_es_diagnostics.py`](../../tests/test_var_es_diagnostics.py) | W0.6 tail diagnostics — `TestTailSampleSize`, `TestExpectedShortfallStandardError`, **`TestAdditiveOnly`**, `TestDiagnosticsReachTheHttpBoundary`, `TestRiskMeasureVocabulary` |
| [`tests/test_integration_identity.py`](../../tests/test_integration_identity.py) | W0.7 — `TestSameSecurityInTwoAccounts`, `TestItemIdProperties`, `TestUnsupportedRowsCarryIdentity`, `TestItemOrderArtifact`, `TestIdentityIsNotArrayPosition` |
| [`tests/test_integration_pipeline.py`](../../tests/test_integration_pipeline.py) | Exit criterion + W0.9 — **`TestExitCriterion`**, `TestNothingIsPricedAtW0`, `TestAccruedInterestIsAnsweredForReal`, `TestCapabilities` |

**318 tests** — 295 in `engine/integration/` plus 23 for the tail diagnostics in
`engine/risk/var_es.py`. The integration tests run against the real delivered TraderX YU18
fixtures (bill, note, sofr — each in v1 and v2).

### Regression tests verified against the wrong implementation

The plan's working rule 3: *"Regression tests must be verified to fail against the pre-fix
code. A test that passes either way proves nothing."* Seven plausible-but-wrong
implementations were patched in and confirmed to fail:

| Wrong implementation | Tests that caught it |
|---|---|
| `blank accrued → 0.0` (the naive zero-coupon rule) | 4 failed, **20 passed** — the 20 are exactly the "bill-only suite" the plan warns about |
| USD-SOFR + ACT/360 added to the allowlist | 8 failed, incl. the `test_sofr_is_absent_from_the_allowlist` tripwire |
| Absent conventions defaulted into the generic builder profile | 8 failed, the whole `TestRefusesToInfer` class |
| Required-column check removed from the loader | 5 failed, surfacing as a bare `KeyError: 'security'` from `terms.py` — the opaque crash the check exists to prevent |
| Missing market inputs falling back to `flat-3pct-v1` | 5 failed, incl. `test_no_branch_ever_returns_a_curve` and `test_failure_is_not_a_convention_refusal` |
| ES standard error returning `0.0` for `n < 2`, and using `ddof=0` | 4 failed, incl. `test_single_observation_is_nan_not_zero` and the sample-vs-population distinction |
| The measure vocabulary drifting between `var_es.py` and `market_inputs.py` | 2 failed, incl. `TestMeasureVocabularyMatchesVarEs` — which is what makes the deliberate duplication safe |
