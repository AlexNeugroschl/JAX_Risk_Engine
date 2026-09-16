# The TraderX EOD Integration Boundary (`engine/integration/`)

**Status: W0 delivered, plus W1.2–W1.4.** Contract and refusal machinery, Treasuries that
return **real numbers**, and an equity case resolved to a *precise refusal* rather than a
guess. Everything else still refuses.

Implements W0, W1.2, W1.3 and W1.4 of the
[TraderX Integration Plan](../planning/traderx-integration-plan.md).

| | |
|---|---|
| **Prices today** | `npv` for **both Treasury shapes** in a **v2** bundle, against an explicitly requested curve — zero-coupon ([W1.2](#w12--the-bill-pricer)) and coupon-bearing ([W1.3](#w13--the-note-pricer)) |
| **Plus one sensitivity** | `rateSensitivity` for a **note only**, as a labelled 1bp parallel bump ([I-16](../known-issues.md#i-16)) |
| **Answers without a model** | `accruedInterest` — a unit conversion of an exported value, not a model output |
| **Refuses, naming what it needs** | A cash equity, for want of a spot/FX source ([W1.4](#w14--the-equity-position-pricer-which-refuses), [I-18](../known-issues.md#i-18)) |
| **Still refuses** | Everything else: corporate bonds, listed options, `rateGamma`/`theta`, a bill's sensitivity, and any unsupported convention |

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

## Why W0 priced nothing

The plan orders the work W0 → W1 → W2 for a specific reason: it "proves the entire
transport → identity → coverage → publication path while pricing math is still out of scope,
so contract bugs and pricing bugs never get debugged simultaneously."

So every model-driven calculation in a W0 result came back `unsupported`. The machinery was
real and tested before any pricer existed — which is what made W1.2 a small, checkable
change rather than a new subsystem.

**The one exception was `accruedInterest`**, and it is deliberate. Accrued interest is a
*unit conversion of an exported value*, not a model output — TraderX supplies
`accruedInterestFraction` and the terms supply enough to interpret it. So the engine can
answer it honestly, and does.

**W1.2 adds the second thing the engine can answer: a bill's NPV.** The ordering paid off
exactly as intended — the pricer is ~60 lines of arithmetic, and because the transport was
already proven, an ORE disagreement could only have been the pricing math.

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
| [`bill.py`](../../engine/integration/bill.py) | W1.2 | Zero-coupon Treasury NPV — the first pricer |
| [`note.py`](../../engine/integration/note.py) | W1.3 | Coupon-bearing Treasury NPV + `rateSensitivity` |
| [`equity.py`](../../engine/integration/equity.py) | W1.4 | Cash equity — a refusal naming the missing spot/FX |
| [`pipeline.py`](../../engine/integration/pipeline.py) | — | Composition of the above |

**This package imports no simulation pricer, no ORE builder, and no curve construction.**
At W0 the ban was total — no ORE, JAX or NumPy at all. W1.2 narrowed it: `bill.py` genuinely
needs `ORE` for date and day-count arithmetic, so `ORE` alone is now permitted.

What has *not* changed is the guarantee the ban exists for. Both pricers are closed-form
discounted cashflows — dates, a day count, some `exp()`s. Neither touches the Monte Carlo
simulation, the JAX kernels, or `build_vanilla_swap`, the last of which is precisely what
W0.4's refusal path keeps away from a booking with unsupported conventions
([I-05](../known-issues.md#i-05)). Refusal must still happen *before* any such pricing object
is constructed, because constructing one is what applies the wrong conventions.

**W1.3 tested this ban rather than theorising about it.** The note pricer needs the ACT/ACT
(ICMA) day count, which lived in `engine/models/ore_builders.py` — so the guard fired. The
vocabulary moved out to the leaf module `engine/day_count.py` (ORE only, no pricer behind it)
instead of the ban being relaxed.

Enforced by tests, not just asserted here:
`TestPackageImportsNoSimulationPricer::test_no_simulation_or_model_pricer_import` bans the
`engine.models`/`engine.simulation`/`engine.portfolio`/JAX layer by AST;
`test_bill_pricer_does_not_reach_the_swap_builder` and
`test_note_pricer_does_not_reach_the_swap_builder` state the specific cases; and
`test_importing_the_package_does_not_pull_in_the_model_layer` closes the AST test's blind
spot by asserting the **transitive** property — importing `engine.integration` in a clean
interpreter must not load the model layer through *any* chain of leaves.

A side benefit of the same constraint: all 491 tests in this layer run in well under a
second, because none of them loads a numerical runtime — ORE's date arithmetic is cheap and
JAX is still absent. (The 23 tail-diagnostic tests live in
`tests/test_var_es_diagnostics.py` instead, since they exercise `engine/risk/var_es.py` and
do need JAX.)

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

### Precision: `standardError` follows the override, `tailCount` does not

`RiskPrecisionOverride(var_es=32)` is applied by casting the P&L cube, so a statistic's
**output dtype is how a caller observes the override**. `standardError` is a statistic and
honours it; `tailCount` is a *count* and stays integral at every precision — float32 cannot
represent integers exactly above 2²⁴, so following the override would let a large-scenario
count silently round.

> **A real bug lived here.** The first implementation promoted float32 P&L to a float64
> standard error, because `jnp.maximum(count, 2)` is integer-typed and the Bessel-correction
> arithmetic promoted the whole expression under `jax_enable_x64`. That silently defeated the
> `var_es=32` override for the one new statistic. It was caught by the *existing*
> `TestPricePortfolioPrecision`, which sweeps every key in `result.risk` — a test written long
> before these keys existed. `TestDiagnosticsRespectInputPrecision` now pins it directly, and
> fails against the buggy version in float32 only; float64 passes either way, which is exactly
> why it hid.

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

## W1.2 — The bill pricer

**The first number this boundary returns.** A Treasury bill is a single cashflow, so its
present value is the whole model:

```
NPV = signedFace × redemptionFraction × P(valuationDate, maturityDate)
```

```python
from engine.integration import price_bundle

result = price_bundle(
    "tests/fixtures/traderx-eod/bill/v2",
    market_inputs={"mode": "assumed-profile", "assumedProfileId": "flat-3pct-v1"},
)
```

```jsonc
"npv": {
  "status": "ok",
  "value": 98507.14563826029,
  "method": "discounted-cashflow",
  "signedFaceAmount": 100000.0,
  "redemptionFraction": 1.0,
  "discountFactor": 0.9850714563826029,
  "yearFraction": 0.5013698630136987,
  "dayCount": "ACT/365 (Fixed)",
  "maturityDate": "2025-12-02",
  "valuationDate": "2025-06-02",
  "curveProvenance": {"curveId": "flat-3pct-v1", "inputOrigin": "assumed", ...}
}
```

The long and short positions come back as exact mirrors (`+98,507.15` / `−98,507.15`,
summing to zero).

### Verified against ORE, exactly

`price_bill` agrees with an independent `ORE.FlatForward` + `ORE.CashFlows.npv` valuation to
**zero difference at machine precision** — not a tolerance, an exact match. The reference is
built from ORE's own term-structure machinery rather than by re-deriving `exp(-rt)`, which
would merely restate the implementation and pass even if both were wrong together.

### The payload is reconcilable, deliberately

A bare NPV is unreconcilable: when TraderX's number disagrees, nothing says whether the
curve, the day count, or the face amount was the cause. So every input to the arithmetic
travels with the answer, and a test asserts
`signedFaceAmount × redemptionFraction × discountFactor == value`.

### Conventions, stated rather than assumed

| Choice | Value | Why |
|---|---|---|
| Discounting day count | **ACT/365 Fixed** | Matches the engine's simulation time axis and the W0.4 allowlist. This is the *discounting* convention — distinct from an instrument's *accrual* convention (the W1.1 split) |
| Compounding | **Continuous** | The assumed profiles are continuously-compounded zero curves. Simple discounting would shift the price by ~$3.6 per $100k face — small enough to read as rounding, large enough to be wrong |

### What it refuses

Each refusal names a reason; none falls back to a default.

| Condition | Reason |
|---|---|
| Coupon-bearing, or a non-empty schedule | `NOT_A_BILL` |
| Maturity on or before the valuation date | `INSTRUMENT_MATURED` — **not** priced at face; that is a settlement question |
| Missing maturity, or unparseable redemption | `TERMS_INCOMPLETE` |
| A **v1** bundle | `NO_PRICER_AT_THIS_STAGE` — without terms the engine cannot establish the row *is* a bill, and will not infer it from a zero coupon column |
| No `marketInputs` requested | `NO_PRICER_AT_THIS_STAGE` — no curve is ever substituted |

**A priced NPV does not make the other calculations answerable.** A priced bill still returns
`rateSensitivity`/`rateGamma`/`theta` as `unsupported`. W1.2 delivers a price, not a
sensitivity, and reporting a zero would be the silent approximation this boundary exists to
prevent. `capabilities()` advertises `TREASURY: ["npv"]` and nothing more.

### Verified against four plausible-but-wrong implementations

Per the plan's working rule 3, each was patched in and confirmed to fail the new tests:

| Wrong implementation | Caught by |
|---|---|
| Simple instead of continuous discounting | 4 tests |
| A separate position sign on top of signed face (short → positive) | 5 tests |
| Pricing a matured bill instead of refusing | 2 tests |
| **Treating any Treasury as a bill** — a note priced with the wrong model | 3 tests |

The last is the dangerous one: it produces a confident, plausible number for the wrong
instrument.

---

## W1.3 — The note pricer

**The first instrument here with a schedule.** A Treasury note is a strip of fixed coupons
plus a bullet redemption:

```
NPV = signedFace × [ Σᵢ cᵢ × P(tᵢ) + redemptionFraction × P(T) ]
```

where `cᵢ` is period `i`'s accrual under the **instrument's own** day count, and `P(·)` is
the discount factor off the explicitly requested curve.

```python
result = price_bundle(
    "tests/fixtures/traderx-eod/note/v2",
    market_inputs={"mode": "assumed-profile", "assumedProfileId": "flat-3pct-v1"},
)
```

On the delivered fixture the long and short positions return **+103,308.33 / −103,308.33**,
summing to zero.

### Three things are new, and each is where a plausible wrong answer lives

| New | The wrong version, and what it would cost |
|---|---|
| **A coupon schedule** | Regenerating it by stepping back from maturity instead of using the exporter's. A plausible schedule produces a plausible price, and silently reprices every coupon whenever the two disagree |
| **A per-instrument accrual day count** | Using the engine's ACT/365 default instead of the note's ACT/ACT (ICMA). The fixture's periods are 182 and 183 days: ICMA makes both exactly 0.5, ACT/365 gives 0.4986 and 0.5014 — a 0.27% error on each coupon, invisible in isolation |
| **Accrued interest from two paths** | Reporting only one of them. See below |

This is the half of W1.1 the bill never exercised: the bill's terms say
`dayCount: NOT_APPLICABLE`, so ACT/ACT (ICMA) had been implemented but never actually used
to price anything until now.

### Accrued interest: the export is authoritative, the schedule is the check

There are two ways to know this note's accrued interest, and they **must not be compared as
exact equals** — the exporter rounds HALF_EVEN at 6 decimals, so they differ by
construction:

| Path | Value on the fixture | Role |
|---|---|---|
| `exported-fraction` | `0.018571` → **$1,857.10** | **What the result reports.** It is the value TraderX's books carry, and what a reconciliation is against |
| `recomputed-schedule` | `0.0185714286` → $1,857.14 | The cross-check, recomputed here from the schedule and day count |

Both travel in the payload with their difference and the tolerance, under an explicit
`accrualSource` label:

```jsonc
"accrualReconciliation": {
  "accrualSource": "exported-fraction",
  "exportedFraction": 0.018571,
  "recomputedFraction": 0.018571428571428572,
  "difference": -0.042857142857136155,
  "tolerance": 0.060000000000000005
}
```

Reporting only one would lose the check; reporting the *recomputed* one as **the** value
would publish a number TraderX's books do not contain — the correction owed in plan §1
(`+1,857.14` vs `1,857.10`).

**The tolerance is derived, never a constant:**

```
round(0.5 × 10^−fractionDecimals × |face|, 2) + 0.01
```

A fixed tolerance is either useless on a $1 position or vacuous on a $1bn one. A difference
**beyond** it is a **refusal**, not a warning — the two paths disagreeing means the schedule
this engine priced is not the schedule the exporter accrued against, so every discounted
coupon is suspect, not just the accrued figure.

> **A useful accident.** This check also catches the wrong-day-count bug: pricing the note
> on ACT/365 shifts accrued by **$5.09 on $100k**, about 85× the $0.06 tolerance. A test
> pins that.

### Verified against ORE, exactly

`price_note` agrees with an independent **`ORE.FixedRateBond`** + `ORE.DiscountingBondEngine`
valuation to **zero difference at machine precision**. The reference is a real ORE bond over
the fixture's own schedule, not a re-derived `Σ cᵢ·exp(−r·tᵢ)` — the ways a bond
pricer goes wrong (a dropped coupon, a double-counted one, an accrual on the wrong basis)
all survive a test that merely restates the implementation.

Parity is asserted on the intermediates too: **every coupon** against ORE's own cashflows one
by one (so a dropped coupon and a compensating discount-factor error cannot cancel), every
discount factor against ORE's curve, and accrued against `accruedAmount`.

### Clean vs dirty

`npv` reports the **dirty** (full) present value, labelled `"priceType": "dirty"`.
`cleanNpv` is carried alongside it, because the exported `closingMark` is a *clean* price and
a consumer reconciling against the extract compares like with like. A "bond NPV" that
silently meant clean would be off by the accrued interest — $1,857 here, large enough to
matter and small enough to look like a curve difference.

### `rateSensitivity` — and what it honestly is

The note is the first instrument here to answer **two** calculations. The sensitivity is a
**bumped revaluation** at an explicit 1bp, re-priced through the same public path rather
than differentiating a closed form:

```jsonc
"rateSensitivity": {
  "status": "ok",
  "value": -15.283220896104467,
  "method": "bumped-revaluation",
  "derivative": "dNPV/dZeroRate",
  "shockedFactor": "zero-curve-parallel",
  "bump": 0.0001
}
```

**It is a parallel shift, not a per-pillar decomposition, and the label says so.** The plan
asked for "per-pillar `rateSensitivity`", but every registered assumed profile is a *flat
constant* — one rate, no pillar structure to shift independently. A per-pillar vector against
it would be arithmetic theatre. This is recorded as **[I-16](../known-issues.md#i-16)** and
closes when `mode: "package"` lands a bootstrapped curve (W2).

### What it refuses

| Condition | Reason |
|---|---|
| Zero-coupon, or no explicit schedule | `NOT_A_NOTE` |
| Maturity on or before the valuation date | `INSTRUMENT_MATURED` |
| Accrued paths disagree beyond tolerance | `ACCRUAL_MISMATCH` — **refused, not warned** |
| `dayCount` absent, or outside the W1.1 allowlist | `TERMS_INCOMPLETE` / `DAY_COUNT_NOT_SUPPORTED` — never defaulted |
| Schedule gapped, overlapping, zero-length, or disagreeing with `maturityDate` | `SCHEDULE_INCONSISTENT` — refused rather than bridged |
| `settlementDays` not 0 | `SETTLEMENT_CONVENTION_NOT_SUPPORTED` — ignoring it would discount on one date and accrue to another |
| Missing coupon rate or maturity | `TERMS_INCOMPLETE` |

`redemptionFraction` absent is the one legitimate default (par) — distinguished from
*present but unparseable*, which is a malformed artifact and refused.

### The bill still has no sensitivity

`capabilities()` reports this **per shape**, because the two Treasury shapes no longer answer
the same set:

```jsonc
"TREASURY": {
  "calculations": ["npv", "rateSensitivity"],
  "byShape": {
    "zero-coupon":     ["npv"],
    "coupon-bearing":  ["npv", "rateSensitivity"]
  }
}
```

W1.3 earned the note's sensitivity with a parity test and earned nothing for the bill.
Collapsing them into one list would advertise a bill sensitivity that does not exist.

### An architectural constraint this forced

The note needs ACT/ACT (ICMA), which lived in `engine/models/ore_builders.py` — a module
`engine/integration/` is **forbidden** to import, because it is where `build_vanilla_swap`
lives, the exact object W0.4's refusal keeps unreachable
([I-05](../known-issues.md#i-05)).

The guard caught the import. Rather than relax it, the day-count vocabulary moved to a new
leaf module **`engine/day_count.py`** that imports only `ORE`; `ore_builders` re-exports it so
every existing caller and W1.1's 27 tests are untouched. The *time axis* role deliberately
did **not** move — it is a property of the simulated curve cube, not of any contract.

A new test also closes the gap the guard had: it was AST-based and saw only *direct* imports,
so it would have missed `integration → leaf → engine.models`.
`test_importing_the_package_does_not_pull_in_the_model_layer` now asserts the real property —
after importing `engine.integration` in a clean interpreter, the model and simulation
modules are not loaded.

### A bug this found — [I-17](../known-issues.md#i-17)

Reusing `bill._parse_date` in `note.py` meant a malformed note date raised
`BillPricingError`, which the pipeline's `except NotePricingError` never caught — so **one
bad date failed the entire bundle** instead of refusing one row. Fixed with its own parser;
four regression tests, verified to fail against the pre-fix code.

---

## W1.4 — The equity position pricer, which refuses

**The simplest arithmetic in this package, and the one thing it cannot honestly compute.**

```
NPV = signedQuantity × contractMultiplier × spot × fx
```

Two of those four factors have no source at this boundary:

| Factor | Source | Status |
|---|---|---|
| `signedQuantity` | positions CSV `quantity` | ✅ present |
| `contractMultiplier` | terms / CSV `contractMultiplier` | ✅ present |
| **`spot`** | a market-data input | ❌ **none exists** |
| **`fx`** | a market-data input | ❌ **none exists** |

`marketInputs` registers flat *interest-rate* profiles and nothing else. `SimulationConfig.equities`
is not a substitute — it drives correlated risk-factor *paths* for a Monte Carlo, takes no
share count, returns no position value, and lives in `engine.simulation`, which this package
may not import.

So W1.4 delivers a **refusal**, and [I-18](../known-issues.md#i-18) records it.

### Why not just use `closingMark`?

The extract carries one, and `quantity × closingMark × contractMultiplier` reproduces the
exporter's own `marketValue` column **exactly**. That is precisely what makes it dangerous:

- it is an **echo, not a valuation**. The engine would hand TraderX their own number back as
  though it had priced it — and a reconciliation against it would *always* agree, proving
  nothing while looking like independent confirmation;
- `closingMark` is an **observation at the session cut**, not a curve this run was priced
  against. Publishing it under `npv` with a provenance derived from the requested *rate*
  profile would label an observed number with a provenance it does not have;
- it silently answers a **different question** than every other `npv` here. The bill and note
  NPVs are present values off a requested curve; an equity "NPV" from the mark is a mark.
  Summing them into one total mixes two incompatible quantities under one heading.

This is working rule 1 applied where returning *a* number would have been trivially easy —
which is exactly when the rule earns its keep.

### What the refusal carries

```jsonc
"npv": {
  "status": "unsupported",
  "reason": "SPOT_SOURCE_NOT_SUPPLIED",
  "intendedMethod": "spot-revaluation",
  "signedQuantity": 1000.0,
  "contractMultiplier": 1.0,
  "multipliedQuantity": 1000.0,
  "currency": "USD",
  "missingInputs": ["spot"]
}
```

A refusal that says only "no" is hard to act on, so the engine reports everything it *could*
establish. The multiplier is applied **exactly once**, in `multipliedQuantity` — so the sign
and size are already correct the day a spot arrives.

### Three refusals, not one

| Condition | Reason | Fixed by |
|---|---|---|
| USD position | `SPOT_SOURCE_NOT_SUPPLIED` | sending a spot |
| Non-USD position | `FX_SOURCE_NOT_SUPPLIED` | sending a spot **and** an FX rate |
| Malformed quantity/multiplier | `TERMS_INCOMPLETE` | re-exporting the row |

The middle one matters: telling a coordinator "send a spot" for a EUR position would be
wrong, because it still would not price. The third is kept distinct because a broken row and
missing market data have different remedies.

An **absent currency is treated as foreign**, not assumed USD — the same refuse-to-infer rule
the rest of the boundary follows. Assuming it would value a foreign position at parity.

### `EQUITY` joined the convention allowlist — deliberately

Before W1.4 an equity refused as `INSTRUMENT_TYPE_NOT_SUPPORTED`: *"outside this engine's
scope"*. That was the wrong fact. A cash equity **is** in scope and fully understood; it
needs one market input nobody has supplied. Those two refusals point at different remedies,
and conflating them tells a coordinator to give up when it should be sending data.

`capabilities()` therefore reports it under its own heading:

```jsonc
"EQUITY": {
  "priced": false,
  "calculations": [],
  "blockedOnMarketInput": {
    "npv": {
      "reason": "SPOT_SOURCE_NOT_SUPPLIED",
      "requires": ["spot", "fx (non-USD positions only)"]
    }
  }
}
```

"Blocked on a market input" is a third state alongside "priced" and "no pricer", and it is
the only one the *consumer* can clear.

### Fixtures

| Bundle | Origin | Exercises |
|---|---|---|
| `equity/v1` | **Vendored from TraderX's `golden-v1/basic`**, LF-exact | The real delivered bytes; verifies against their published `bundleId`. No terms artifact, so it refuses with `TERMS_NOT_SUPPLIED` — a more fundamental gap than the missing spot. Also carries a USD-SOFR swap, so the I-05 refusal is re-checked |
| `equity/v2` | Authored, synthetic, clearly labelled | The priced path: a long/short USD pair plus a EUR row, so both market-data refusals and the long/short mirror are exercised end-to-end |

The v1 source files were **CRLF** in the reference checkout and were normalized to LF on
vendoring — the exact trap [W0.1](#w01--bundle-ingestion-and-hash-verification) warns about.
They reproduce TraderX's hashes only as LF.

### Verified against the wrong implementation

Per working rule 3, the mark-echo was patched in at **both** the pricer and the pipeline
level and confirmed to fail: **20 of 60 tests**, including the dedicated
`TestDoesNotEchoTheExportedMark` guard. A test that passes either way would prove nothing
here, because the wrong answer is a plausible, well-formed, perfectly reconciling number.

---

## Not yet implemented

| Task | Status | Why |
|---|---|---|
| **W0.8** durable result lookup | Not started | Needs a persistent store + HTTP endpoint; the crash-safety semantics are the substance and can't be meaningfully tested against the in-process job store ([I-08](../known-issues.md#i-08)). |
| **W1.5** wire-through to the portfolio path | Not started | Both bond pricers live at this boundary; `engine/instruments/` is still four rate-derivative modules. |
| Equity **valuation** | Blocked | The refusal path landed (W1.4); pricing needs a spot/FX source ([I-18](../known-issues.md#i-18)). |
| Per-pillar `rateSensitivity` | Blocked | Needs a curve with pillar structure - `mode: "package"`, i.e. W2 ([I-16](../known-issues.md#i-16)). |

Unblocked — sequencing, not dependency.

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
| [`tests/test_integration_pipeline.py`](../../tests/test_integration_pipeline.py) | Exit criterion + W0.9 — **`TestExitCriterion`**, `TestNothingIsPricedAtW0`, `TestAccruedInterestIsAnsweredForReal`, `TestCapabilities`, `TestPackageImportsNoSimulationPricer` |
| [`tests/test_integration_bill.py`](../../tests/test_integration_bill.py) | W1.2 — **`TestOreParity`**, `TestSigns`, **`TestMaturityBoundary`**, `TestRefusesWhatItCannotPrice`, `TestThroughTheBundlePipeline` |
| [`tests/test_integration_note.py`](../../tests/test_integration_note.py) | W1.3 — **`TestOreParity`**, **`TestAccruedReconcilesToTraderX`**, `TestToleranceIsDerivedNotConstant`, `TestCleanDirtyReconciliation`, `TestLongShort`, `TestRateSensitivity`, **`TestWrongDayCountIsCaught`**, `TestAccrualMismatchIsRefused`, `TestScheduleIsUsedNotRegenerated`, `TestRefusals`, **`TestRefusalsAreNotePricingErrors`**, `TestIsNote`, `TestPipelineEndToEnd`, **`TestBillIsUnchangedByW13`**, `TestCapabilitiesAdvertiseW13` |
| [`tests/test_integration_equity.py`](../../tests/test_integration_equity.py) | W1.4 — `TestRefusesRatherThanPrices`, **`TestDoesNotEchoTheExportedMark`**, `TestLongShort`, **`TestMultiplierAppliedExactlyOnce`**, `TestCurrencyAndFx`, `TestIsEquity`, `TestMalformedRows`, **`TestRefusalsAreEquityPricingErrors`**, `TestPipelineEndToEnd`, **`TestTreasuriesAreUnchangedByW14`**, `TestCapabilitiesAdvertiseW14` |
| [`tests/test_day_count_roles.py`](../../tests/test_day_count_roles.py) | W1.1 — the two day-count roles; 27 tests, unchanged by W1.3's move of the accrual vocabulary to `engine/day_count.py` |

**491 tests in `engine/integration/`** — 104 for W1.3's note pricer and 60 for W1.4's equity refusal, plus 23 for the tail
diagnostics in `engine/risk/var_es.py` and 27 for the W1.1 day-count split. The integration
tests run against the real delivered TraderX YU18 fixtures (bill, note, sofr — each in v1
and v2), and complete in under a second.

Full suite, run 2026-09-16 after W1.4: **1,384 passed, 2 failed**. The 2 failures are the
documented `pydantic` environment gap (`tests/test_var_es_diagnostics.py::TestDiagnosticsReachTheHttpBoundary`),
a declared dependency that is not installed here — not a code defect, and confirmed
pre-existing. `tests/test_api.py` does not collect for the same reason. Engine-side, every
test passes.

### Regression tests verified against the wrong implementation

The plan's working rule 3: *"Regression tests must be verified to fail against the pre-fix
code. A test that passes either way proves nothing."* Eleven plausible-but-wrong
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
| ES standard error promoting float32 → float64 (**a real bug this caught**) | 2 failed in `TestDiagnosticsRespectInputPrecision`, float32 only — float64 passes either way, which is why it hid |
| Pricing the note on **ACT/365** instead of ACT/ACT (ICMA) | Caught by the accrual reconciliation itself — the error is $5.09 on $100k against a $0.06 derived tolerance, ~85× |
| A note's refusal raised as a **`BillPricingError`** (**a real bug this caught** — [I-17](../known-issues.md#i-17)) | 4 failed in `TestRefusalsAreNotePricingErrors`; the one that mattered asserts a malformed row does not take the whole bundle down |
| **Echoing `closingMark` as an equity `npv`** | **20 of 60** failed, incl. the dedicated `TestDoesNotEchoTheExportedMark` — patched in at both the pricer and the pipeline level. The dangerous one: it reconciles perfectly against TraderX because it *is* TraderX's number |
