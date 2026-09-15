# Response Proposal: TraderX → JAX Risk Engine EOD Contracts

**Status:** Proposal, for reconciliation against `reference/traderX/docs/risk-integration/`
(01-architecture, 02-contracts-and-data, 05-alex-backlog, 07-baseline-and-decisions) and the
TraderX EOD handoff proposal.

**Grounding rule for this document.** Every "supported today" claim below names the module and
test file that backs it in this repository. Every "would implement" item is explicitly marked
as not built. The distinction the handoff asked for — *what the engine already supports vs.
what I would add* — is the organizing principle of the whole document, not a closing section.

---

## 0. Executive summary

I accept the bundle-and-manifest delivery shape, with four amendments (§5). I propose we
split the work by **what the engine can price correctly today** rather than by what it can
parse.

The honest headline: **the engine's pricing strength (rate derivatives under Hull-White/LGM
with Monte Carlo exposure cubes) and TraderX's export coverage overlap less than the artifact
table suggests.** Schema 2 exports swaps and swaptions, which is my strong suit — but as
*generic tenor-based bookings*, while my builder constructs a generic IBOR/ACT365 swap
(`engine/models/ore_builders.py:41`). Neither side's "swap" is a USD-SOFR swap yet. Meanwhile
schema 3's equities and Treasuries are trivially priceable financially but have **no pricer in
this repo at all**.

So I propose an ordering that inverts the apparent difficulty:

| Wave | Content | Why first |
|---|---|---|
| **W0** | Contract freeze + fixtures, no new pricing | Both repos validate the same bytes before any number is called correct |
| **W1** | Treasuries + cash equities | New but *small* pricers; deterministic, no Monte Carlo, no calibration — fastest path to a real number tied to a real booked position |
| **W2** | USD-SOFR swaps (explicit dates, ACT/360) | Highest-value, but blocked on D03/D04 conventions **and** a builder replacement; the existing swap pricer is not reusable as-is for SOFR |
| **W3** | European swaptions on booked terms; scenario/VaR on the joint portfolio | Reuses W2's schedule work |
| **W4** | Bermudan/American, corporate spread, listed options | Each gated on a capability-profile flag, not implied by W3 |

W1 before W2 is deliberate and contradicts the intuition that I should "start where the engine
is strongest." Rationale in §3.1.

---

## 1. Capability matrix: what the engine supports today

This is the machine-readable answer to the handoff's question 1, and it is what I propose to
serve from a `GET /capabilities` endpoint (§6.1) so coverage is negotiated rather than assumed.

### 1.1 Instruments

| TraderX artifact | Instrument | Engine status today | Module / evidence |
|---|---|---|---|
| Schema 2 OTC | Vanilla swap | **Prices — but generic IBOR/ACT365/TARGET, tenor-based** | `engine/instruments/swap.py`, `ore_builders.build_vanilla_swap` |
| Schema 2 OTC | European swaption | **Prices — Hull-White analytic (Jamshidian)** | `engine/instruments/european_swaption.py` |
| Schema 2 OTC | Bermudan swaption | **Prices — LGM backward induction** | `engine/instruments/bermudan_swaption.py` |
| Schema 2 OTC | American swaption | **Prices — discretized Bermudan** | `engine/instruments/american_swaption.py` |
| Schema 3 positions | Treasury / fixed-rate bond | **Not implemented** — planned only | `docs/planning/traderx-bond-integration-roadmap.md` |
| Schema 3 positions | Corporate bond | **Not implemented** — no spread/credit model exists | — |
| Schema 3 positions | Cash equity / ETF | **Not implemented as a pricer** | see §1.4 caveat |
| Schema 3 positions | Listed option | **Not implemented** | — |

### 1.2 Calculations

| Calculation | Status | Evidence |
|---|---|---|
| Base NPV (t=0), per portfolio | Supported | `PortfolioResult.base_npv` |
| Per-trade base NPV | **Supported** (closed — was aggregate-only) | `PortfolioResult.base_npv_per_trade`; `tests/test_portfolio_gap_fixes.py::TestPerTradeBaseNpv` |
| Exposure cube `[Scenarios, TimeSteps, Trades]` | Supported | `PortfolioResult.npv_cube` |
| VaR / ES at requested percentiles | Supported | `engine/risk/var_es.py` |
| Delta / Gamma — swaptions (all 3 types) | Supported, ORE bump units | `engine/risk/greeks.py:437,566` |
| Delta / Gamma — **swaps** | **Supported** (closed — was implemented but unreachable) | `swap_delta_gamma` wired via `_swap_curve_configs`; `tests/test_portfolio_gap_fixes.py::TestSwapGreeksReachThePortfolioPath` |
| Theta — all four types | Supported (swaps included) | `greeks.py:279,474,594` |
| Vega — Bermudan | **Supported for a calibrated `Sigma`** (closed — was never called) | `tests/test_portfolio_gap_fixes.py::TestBermudanVegaReachesThePortfolioPath` |
| LGM sigma calibration from a swaption basket | Supported | `engine/calibration/lgm.py`, `POST /calibration/lgm` |
| Externally supplied scenario sets | **Not implemented** — engine generates its own paths | §4.3, backlog A-R01 |
| Deterministic stress | **Not implemented** | backlog A-R06 |

### 1.3 Numerics and execution

| Capability | Status |
|---|---|
| Per-stage precision (simulation / pricing / risk), fp32 or fp64 | Supported — `PrecisionConfig`, `engine/portfolio/request.py:169` |
| Per-instrument-type and per-Greek precision drill-down | Supported — `PricingPrecisionOverride`, `RiskPrecisionOverride` |
| Process-isolated concurrent precision tiers | Supported — `engine/portfolio/worker_pool.py` |
| CPU / GPU / TPU backends | Supported by JAX; **TPU fp64 is not equivalent to CPU fp64** and must never be assumed so |
| Async job submit + poll | Supported — `POST /portfolio/price` → 202 + `job_id`; `GET /portfolio/price/{job_id}` |
| Durable job store | **Not implemented** — in-process dict (`routes.py` `_JOBS`); §6.3 |
| PSD validation of the covariance matrix | Supported, with opt-in repair — `validate_joint_covariance` / `nearest_psd` |
| Cross-field consistency validation | Supported — `validate_portfolio_against_simulation` |
| Auto-derived curve pillars from trade schedules | Supported — `derive_maturity_pillars` |

### 1.4 Three caveats I want on the record before any scoping conversation

**(a) An equity simulation factor is not an equity pricer.** `SimulationConfig.equities`
(`initial_prices`, `dividend_yields`, `rate_mapping`) drives correlated *risk-factor paths*.
There is no instrument that takes a signed share count and returns a position value. Backlog
A-P01 is correctly stated; I am confirming it, not disputing it. This is genuinely small work
(§3.1) but it is work.

**(b) The swap builder cannot accept a TraderX SOFR booking.** `build_vanilla_swap` takes a
*tenor string* (`"5Y"`) and constructs an `ORE.IborIndex` literally named `"SimIndex"` with
`ORE.Period(index_tenor_months, Months)`, `USDCurrency`, `TARGET()` calendar,
`ModifiedFollowing`, and **`Actual365Fixed` forced on both legs**
(`ore_builders.py:41-77`). The module docstring states the ACT/365 choice is deliberate — it
keeps day count consistent with the simulation's own year-fraction time axis. That is a
sound internal decision and a **blocking external incompatibility**: TraderX books USD-SOFR
on ACT/360 with explicit effective/maturity dates and an overnight index, not a term IBOR
tenor. 07-baseline's assessment is correct. Feeding a SOFR contract through this builder
would produce a number, and that number would be wrong in a way no test here would catch.
I am treating "refuse the booking" as the required near-term behavior and the faithful
builder as W2 work.

**(c) The aged-swap limitation bounds every multi-day product.** `swap.py`'s conditional
pricing at any simulated time past a swap's first accrual date does not represent an
already-fixed floating coupon (documented, tested as `TestAgedSwapKnownLimitation`). t=0
valuation is unaffected. **Any exposure profile, any multi-day experiment, and any theta
interpretation past the first reset inherits this gap.** It is not a rounding concern; it
is a structural one, and it is the single largest financial-correctness item standing
between this engine and M3.

**This gap is no longer silent.** `price_portfolio` now emits a warning into
`PortfolioResult.warnings` naming every swap that will be aged at one or more simulated
steps, how many steps are affected, and that t=0 base NPV is unaffected
(`_warn_if_aged_swap_exposure`; `tests/test_portfolio_gap_fixes.py::TestAgedSwapWarningIsNotSilent`).
The underlying pricing inaccuracy is **unchanged** — closing it needs per-scenario fixings
that neither this engine nor the current TraderX export has (§2.2 `pastFixings`). I propose
these warnings map to structured `warnings[]` codes in the result contract (§6.4), so a
console can badge an approximated exposure rather than displaying it as exact.

---

## 2. Required additional fields, with units

Answering handoff question 2. Organized by what breaks without them.

### 2.1 Universal — required on every row of both artifacts

| Field | Type / unit | Why the engine needs it |
|---|---|---|
| `instrumentId` | opaque string | Result join key. Array index is not an identity (§3.4) |
| `accountId` | opaque string | Account-level aggregation and sign attribution |
| `currency` | ISO 4217 | Refuse cross-currency aggregation without synchronized FX |
| `assetClass` / `productType` | controlled enum | Capability-profile routing; drives the supported/unsupported decision |
| `direction` / signed quantity | explicit sign convention | Must not be inferred from notional sign |

### 2.2 USD-SOFR swap — the W2 blocker set

Ranked by what silently mis-prices without it. `paymentFrequency` in TraderX describes the
**fixed leg** and must not be read as the floating index tenor (02-contracts §"Alex's current
generic IBOR" — confirmed, and it is exactly the mistake my current builder invites).

| Field | Unit / type | Consequence if absent |
|---|---|---|
| `effectiveDate`, `maturityDate` | explicit ISO dates | **Must not be rounded to whole years.** Tenor-derived dates are not the booked contract |
| `fixedLegDayCount` | enum (expect `ACT/360`) | Wrong accrual on every fixed coupon |
| `floatingIndex` | enum (`USD-SOFR`) | Index identity, not a name-only string map |
| `floatingLegDayCount` | enum (`ACT/360`) | |
| `floatingLegCompounding` | enum: compounded / simple average | Materially different NPV on an OIS leg |
| `lookback`, `lockout`, `paymentLag` | integer business days | SOFR-specific; no default is safe |
| `fixedLegFrequency`, `floatingLegFrequency` | enum, **separately** | Single `paymentFrequency` is ambiguous |
| `calendar`, `businessDayConvention` | enum (expect `US-SIFMA`/`FedFunds`, not `TARGET`) | My builder currently hardcodes `TARGET()` |
| `rollConvention`, stub type/dates | enum + dates | Irregular first/last periods |
| `pastFixings` | array of (date, rate as decimal) | **Required to close the aged-swap gap (§1.4c).** Without it, no correct t>0 valuation |
| `notionalSchedule` | array or "constant" | Amortizers refused explicitly rather than assumed bullet |

### 2.3 Swaption — beyond the current export

07-baseline notes the export has expiry/style but **the direction field describes the
underlying fixed leg, not option holder long/short**. I need both, separately:

| Field | Unit / type | Note |
|---|---|---|
| `optionHolderDirection` | long / short | **Distinct from underlying payer/receiver.** Sign of the whole position |
| `settlementType` | cash / physical | Cash settlement changes the payoff, not just the plumbing |
| `cashSettlementMethod` | enum | If cash: collateralized-cash-price vs par-yield materially differ |
| `exerciseSchedule` | **explicit array of dates** | `exerciseStyle="Bermudan"` alone is insufficient (02-contracts §product table — agreed) |
| `exerciseWindowStart/End` | dates | American |
| `premium`, `premiumDate` | amount + date | Or explicitly "not modelled" |
| Underlying full term set | §2.2 | A swaption is only as well-specified as its underlying |

**Mid-coupon exercise caveat:** Bermudan/American pricing here is exact only when exercise
dates are reset-aligned with the underlying accrual schedule; otherwise it is a documented,
conservative (understating) approximation. `validate_portfolio_against_simulation` already
emits a `UserWarning` per misaligned trade and `price_portfolio` collects these into
`PortfolioResult.warnings`. **I propose these surface as structured `warnings[]` entries in
the result contract (§6.2), not as free text**, so the console can badge an approximated
contract rather than displaying it as an exact value.

### 2.4 Bonds

| Field | Unit / type | Note |
|---|---|---|
| `faceAmount` | currency amount | TraderX quantities are face amounts — confirmed |
| `coupon` | **decimal fraction** | Explicitly not percent |
| `couponFrequency`, `dayCount`, `calendar` | enums | ACT/ACT-ICMA for Treasuries |
| `firstCouponDate`, `penultimateCouponDate` | dates | Stub resolution |
| `cleanPrice` | **fraction of par** | Confirmed convention |
| `accruedInterest` | currency amount | TraderX market value is clean; accrued is separate |
| `settlementConvention` | T+n | Required to reconcile dirty value vs discounted cashflows |
| `creditSpread` / `recoveryRate` | decimal / decimal | Corporates only. **A Treasury-discounted corporate is not credit pricing** — I will refuse corporates until W4 rather than emit a misleadingly precise number |

### 2.5 Market data — what I need from the proposed pricing bundle

The handoff correctly flags this as proposed work. Answering "curves, fixings, vol,
historical windows, observations vs. constructed":

**I want observations, and I own the construction.** This matches D05 and 02-contracts
§"Alex owns the accepted curve bootstrap." One validated numerical definition, on my side,
so curve construction is not silently duplicated in two languages.

| Package | Content | Units | Wave |
|---|---|---|---|
| **USD-SOFR curve inputs** | Dated OIS quotes / futures / deposit rates with instrument terms | decimal rates | W2 |
| **Treasury curve inputs** | On-the-run prices **or** yields with full instrument terms | fractions of par / decimal | W1 |
| **SOFR fixings history** | Daily published fixings back to each live trade's effective date | decimal, dated | W2 — **hard blocker for aged swaps** |
| **Swaption vol** | Normal (Bachelier) vols, expiry × tenor grid, with quote convention stated | decimal, absolute | W3 |
| **Equity spot** | Per-security close, with source/time | currency | W1 |
| **Corporate actions / dividends** | Dated, with treatment | — | W1 (multi-day), W4 (options) |
| **FX** | Spot per reporting pair, synchronized to valuation time | rate | Only when non-USD appears |
| **Historical panels** | Aligned factor returns, with missingness flags | decimal returns | W3 (covariance) |

**Explicitly not usable as-is:** the FRED CMT reader's 11 interpolated yield points are
**neither zero-rate pillars nor a SOFR curve** (07-baseline — agreed). I will not ingest
interpolated CMT yields as zero rates. Percent-vs-decimal units and each observation's own
date must be preserved.

**Curve package metadata I require** (mirroring 02-contracts §market-data): currency, role
(discount vs projection), index/collateral assumption, valuation date, pillar dates,
zero-vs-discount-factor representation, compounding, day count, interpolation,
extrapolation, calibration input IDs, and repricing diagnostics. **Bootstrapping must
reprice its own calibration instruments within an agreed tolerance, and I will return that
diagnostic in the result** rather than asserting the curve is good.

**Initial labelled test curve.** For W0/W1 I propose a flat or simply-shaped **explicitly
assumed** curve so both repos can run end-to-end before real market packaging exists. It
must be labelled `assumed` in the manifest and carried into the result's provenance block.
A number computed on an assumed curve must never be displayable as market-derived.

---

## 3. Engine-side work I propose to implement

Answering the "what would you add" half of the handoff, in wave order.

### 3.1 W1 — Treasuries and cash equities (why these come first)

The counterintuitive ordering is deliberate:

1. **They need no conventions negotiation.** Bond and equity conventions are standard and
   already in the export. SOFR conventions are D03/D04 — genuinely open, and I should not
   block the first real number on a decision that needs both of us.
2. **They need no Monte Carlo.** A fixed-rate Treasury is a deterministic discounted
   cashflow; an equity position is `quantity × multiplier × spot × fx`. No simulation, no
   calibration, no JIT-compilation latency, no precision tiering. This means **W1 results
   return in milliseconds**, which makes them viable for the live desk (M2) in a way a
   4096-scenario cube is not.
3. **They exercise the entire contract path** — bundle fetch, hash verification, identity
   join, per-instrument results, coverage reporting, result publication — with the pricing
   math held trivially simple. Contract bugs and pricing bugs get debugged separately
   instead of simultaneously.
4. **The bond plan already exists here.** `docs/planning/traderx-bond-integration-roadmap.md`
   treats a fixed-rate Treasury as a swap's fixed leg plus bullet principal, reusing
   `ore_builders.py`'s existing cashflow machinery — one thin instrument module plus one
   curve bootstrap module.

Deliverables: `BondConfig` + pricer; `EquityPositionConfig` + pricer; a Treasury zero-curve
bootstrap validated by repricing its own inputs; both joined into `PortfolioRequest.trades`.

### 3.2 W2 — a faithful USD-SOFR builder

**A new builder alongside `build_vanilla_swap`, not a modification of it.** The existing
generic builder stays exactly as it is — every current swaption pricer depends on its
ACT/365 consistency with the simulation time axis, and the full existing test suite pins
that behavior. The new path takes explicit dates, an overnight index, ACT/360, real
calendars, and compounding/lookback/lockout, and it is selected by the *booked contract's
own terms*, never by a fallback.

**Unmapped conventions raise, never degrade.** A booking whose convention set is not in the
supported subset comes back as an explicitly unsupported item with a reason, and never as a
generic-IBOR approximation. This is the single most important behavioral commitment in this
document.

Also in W2: **past fixings ingestion**, which is what actually closes §1.4(c) and unlocks
multi-day work.

### 3.3 ~~W2~~ — swap Greeks through the portfolio path ✅ **done**

`swap_delta_gamma` was implemented and tested but the portfolio dispatcher skipped
`SwapConfig`, because a swap indexes into the simulation's curves rather than carrying its
own `ZeroCurveConfig` and the code refused to guess a placeholder. **Fixed:**
`_compute_all_greeks` now receives the `SimulationConfig` and resolves each swap's
`discount_curve_index`/`forward_curve_index` through `_swap_curve_configs`, which **raises**
on an out-of-range index rather than falling back to curve 0. `bermudan_vega` is likewise
wired, for a genuinely calibrated `Sigma` only.

Orchestration only — no pricing math changed. Regression-verified: reverting the wiring
fails 5 tests, and a naive "default to curve 0" fix fails
`test_uses_the_curve_its_indexes_name_not_curve_zero`.

### 3.4 W1 — identity and per-instrument results

Two changes, both required before any result can be joined back to a booking:

- **Opaque IDs carried end to end.** `PortfolioRequest.trades` is positional and
  `PortfolioResult.greeks` is keyed by trade index. I propose adding an
  `instrumentId`/`accountId` pair to every trade config, echoed on every result row. Array
  position stops being load-bearing.
- **Per-instrument base NPV** ✅ **done.** `base_npv` was a single portfolio float.
  `PortfolioResult.base_npv_per_trade` now returns each trade's own t=0 NPV in request
  order, with `base_npv` defined as their sum — the total and the breakdown are the same
  numbers and cannot drift apart. Exposed through the HTTP result schema too.

### 3.5 W3 — externally supplied scenario sets

The engine generates its own risk-neutral paths. For stress and historical VaR I need to
accept **immutable, externally supplied scenario sets with stable IDs and hashes** and
reprice against them. This also decouples my precision/hardware research from path
generation — the same frozen scenario set across devices is what makes a numerical
comparison meaningful rather than a resampling artifact.

### 3.6 Measure labelling (W1, cheap, and I consider it non-negotiable)

Every risk number gets an explicit `measure` label: `risk-neutral-pricing`,
`historical-forecast`, or `deterministic-stress`. **A risk-neutral exposure simulation is not
a calibrated forecast of tomorrow's loss**, and the current `VaR_95` key in
`PortfolioResult.risk` does not say which it is. With ~5 months of daily observations, a 99%
empirical tail is sparse; I will report effective sample size and Monte Carlo standard error
alongside every tail statistic rather than emitting a bare quantile.

---

## 4. Financial semantics I propose to fix (D08)

02-contracts assigns me the lead here. My proposed definitions:

**NPV** — signed value to the named account, in the stated currency, at the stated valuation
time. Observed execution price, observed mark, and model value stay three separate fields and
are never coalesced.

**Rate sensitivity** — I will publish `pvChangeForPlus1bp`: the signed P&L from a +0.0001
**absolute zero-rate** move. This matches the engine's existing internal convention exactly —
`greeks.py` computes AD gradients scaled by a 1bp bump (`DEFAULT_RATE_BUMP`, replicating ORE's
`ShiftType=Absolute, ShiftSize=0.0001`). **I propose we avoid the name "DV01" in the contract
entirely**; if the console displays DV01, it publishes its own sign convention next to it.

Per-pillar deltas carry `curveId` + `pillarDate` + bump definition. **Discount and forward
contributions are summed only for a specifically defined joint shock**, never by default.

**Gamma** is in units of the bump squared (`greeks.py` returns `hess × bump²`) — stated
explicitly, because a gamma whose scaling is ambiguous is worse than no gamma.

**Vega** — per 1 **absolute** volatility point on the normal/Bachelier quote, stated in the
field name, since the engine calibrates to normal vols.

**Theta** — holds contractual identity fixed, defines curve roll explicitly, and includes
intervening cashflows. Rebuilding a new spot-starting trade tomorrow is not ageing the
original contract, and this is the definition the aged-swap gap (§1.4c) currently threatens;
until W2's fixings work lands, **theta on an aged swap is a flagged approximation**.

**Aggregation** — requires common reporting currency, synchronized FX, and compatible factor
definitions. The engine will **refuse** to sum incompatible vegas, unrelated curve pillars, or
raw deltas across factors. Portfolio VaR/ES comes from aggregating scenario P&L *within each
scenario first*, then taking tail statistics. **Summing standalone VaRs is not portfolio risk**
and the API will not offer it.

**Non-finite values** — never emitted as bare JSON `NaN`/`Infinity`, never reinterpreted as
zero. An empty-tail ES serializes as `null` with a status code explaining why (already the
behavior in the existing response schema).

---

## 5. Response to the proposed delivery bundle

I accept GCS + immutable versioned manifest. Four amendments:

**(1) Pull, not push-notified-only.** The current readiness announcement is a core NATS
publish; a recovering subscriber cannot retrieve a missed announcement. I propose the bundle
be **discoverable by listing immutable unprocessed bundles**, with the notification as an
optimization. My worker must be able to restart and find work without a replayed message.
This is D02 and I am agreeing with 02-contracts' framing, not proposing something new.

**(2) Verify each hash against its own bytes.** The CSV hash, contracts hash, and cut hash
cover different bytes. I will verify each artifact against its corresponding hash and
separately validate shared cut/date/version fields and declared row counts. **A cut hash
alone does not pin every mark or static term joined during rendering** — so I need the
reference-data version in the manifest too, and I will echo all of them.

**(3) An empty contracts file is not a missing one.** Zero OTC contracts is a valid EOD state
and must produce a complete result with zero OTC rows, not a partial-coverage warning. I need
these distinguishable at the manifest level.

**(4) Market bundle referenced by identity, not by "latest."** The manifest's market reference
must name an immutable versioned package. I will refuse a request whose market reference
resolves differently on retry — otherwise idempotency is not real.

**Idempotency key.** I propose it covers exactly: portfolio bundle hash + market package hash
+ calculation set + model version + calibration version + scenario-set ID + precision profile.
An identical logical request returns the existing successful result. **Changed bytes under an
existing immutable ID is an error, not an update.** Benchmark repetitions of the same workload
get distinct attempt IDs while sharing a workload key — I need this for the precision research
lane, where re-running the identical workload is the entire point.

---

## 6. Proposed interfaces

### 6.1 `GET /capabilities` — new

Returns §1's matrix as structured data: supported (product × convention × calculation)
combinations, precision/device profiles, engine/model/build versions, and known-limitation
flags. **This is the mechanism that makes "no silent exclusions" enforceable** — the
coordinator can determine before submitting whether every instrument in a bundle is priceable,
and I can reject a bundle containing products I never claimed.

### 6.2 `POST /risk/jobs` — the EOD entry point

Extends the existing `POST /portfolio/price` (which stays for direct/dataclass callers). Body:

```jsonc
{
  "requestId": "...",              // caller's idempotency key
  "portfolioBundle": { "uri": "gs://...", "manifestHash": "sha256:..." },
  "marketPackage":   { "id": "...", "hash": "sha256:...", "label": "assumed|market-derived" },
  "calculations":    ["npv", "pv01", "gamma", "theta", "var_es"],
  "measure":         "risk-neutral-pricing",
  "reportingCurrency": "USD",
  "valuationDate":   "2026-09-14",
  "scenarioSet":     { "id": "...", "hash": "sha256:..." },   // optional; else engine-generated
  "precision":       { "simulation": 64, "pricing": 64, "risk": 64 },
  "coveragePolicy":  "partial-allowed"                         // or "all-or-nothing"
}
```

Returns `202` + `jobId`. Validation is synchronous and cheap (the existing validators do no
JAX work), so bad terms fail immediately with the validator's own message — the current
`400`-vs-`422` split already works this way and I propose keeping it.

### 6.3 Job lifecycle

Current: in-process dict, statuses `pending`/`running`/`done`/`failed`, lost on restart.

Proposed: `queued | running | succeeded | partial | failed | cancelled | superseded`, with
**every attempt and failure reason preserved**. Two additions matter most:

- **`partial` is a first-class success.** A bundle with three unsupported corporates returns
  `partial` with results for everything else plus explicit unsupported rows. It is not a
  failure and not a silent exclusion.
- **Structured failure classes**, so the coordinator retries only what is retryable:
  `bad-terms` (never retry), `missing-market-data` (retry after data lands),
  `unsupported-product` (never retry — capability gap), `numerical-failure` (retry with
  different precision may help), `infrastructure` (retry freely).

**On job ownership (D13): TraderX's coordinator owns the durable logical job; I own the
computation attempt.** My `jobId` is an attempt identity, not a system of record. My process
restart must be *detectable* — I propose a worker boot epoch in `/version` and on every result
— and recoverable by the coordinator re-submitting immutable inputs under the same idempotency
key. **A lost in-memory job is an infrastructure event, never a financial failure**, and I
would rather the coordinator retry me than that I build a competing durable store.

### 6.4 `GET /risk/results/{jobId}` — the morning contract

Answering handoff question 4. Compact by default; arrays referenced, not inlined.

```jsonc
{
  "jobId": "...", "requestId": "...", "status": "partial",
  "inputs": {                                  // echoed, for audit
    "portfolioBundleHash": "sha256:...", "marketPackageId": "...", "marketLabel": "assumed",
    "referenceDataVersion": "...", "scenarioSetId": "...", "cutHash": "...",
    "sessionDate": "2026-09-13", "consensusSequence": "48211903"   // decimal STRING
  },
  "execution": {
    "engineVersion": "0.1.0", "modelVersions": {"rates": "hull-white-1f", "bermudan": "lgm"},
    "calibrationVersion": "...", "device": "tpu-v5e", "workerBootEpoch": "...",
    "actualPrecision": {"simulation": 32, "pricing": 32, "risk": 64},
    "timings": {"prepareMs": 812, "compileMs": 41903, "computeMs": 9122, "serializeMs": 210}
  },
  "valuation": { "valuationTime": "2026-09-13T21:00:00Z", "reportingCurrency": "USD" },
  "results": [
    { "instrumentId": "SW-00412", "accountId": "ACC-7", "status": "ok",
      "npv": {"value": -184203.11, "currency": "USD"},
      "sensitivities": [
        {"name": "pvChangeForPlus1bp", "curveId": "USD-SOFR-DISC", "pillarDate": "2027-09-13",
         "value": -41.88, "currency": "USD", "bump": {"type": "absolute", "size": 0.0001}}
      ],
      "warnings": [{"code": "AGED_SWAP_FIXING_APPROXIMATION", "severity": "material"}] },
    { "instrumentId": "CB-0007", "accountId": "ACC-7", "status": "unsupported",
      "reason": "CORPORATE_SPREAD_MODEL_NOT_IMPLEMENTED",
      "grossNotional": {"value": 5000000.0, "currency": "USD"} }
  ],
  "aggregates": [
    { "scope": {"accountId": "ACC-7"}, "npv": {"value": 1204881.22, "currency": "USD"},
      "measure": "risk-neutral-pricing",
      "var": {"confidence": 0.99, "horizonDays": 1, "value": 88213.4,
              "effectiveSampleSize": 4096, "monteCarloStdError": 1204.7,
              "lossSign": "positive-is-loss"} }
  ],
  "coverage": {
    "instrumentsRequested": 214, "priced": 211, "unsupported": 3, "failed": 0,
    "unpricedGrossNotional": {"value": 15000000.0, "currency": "USD"}
  },
  "artifacts": [
    { "kind": "npv_cube", "uri": "gs://...", "shape": [4096, 24, 211],
      "dtype": "float32", "ordering": ["scenario", "timeStep", "instrument"],
      "instrumentIdOrder": "gs://.../order.json", "hash": "sha256:..." }
  ],
  "diagnostics": {
    "curveRepricingErrors": [{"curveId": "USD-SOFR-DISC", "maxAbsBp": 0.03}],
    "calibrationRmse": 1.8e-10
  }
}
```

Five deliberate choices:

- **Unsupported items appear as rows with gross notional**, so coverage is auditable from the
  result alone. They never vanish from the denominator.
- **`actualPrecision` reports what ran**, not what was requested — a downgrade must be
  visible. (The known `/version` nuance applies: the dispatcher's backend is not proof of the
  worker's device, so device is reported *by the worker*, per backlog A-O03.)
- **The scenario cube is an artifact reference with shape/dtype/ordering/hash**, never inlined.
  The current `npv_cube` serializes `[Scenarios, TimeSteps, Trades]` into the JSON body; at
  4096 × 24 × 211 that is unusable through a browser or control message.
- **Instrument ordering for the cube is published as its own file**, so an array axis is
  resolvable to opaque IDs without relying on request order.
- **Large integers are decimal strings** (`consensusSequence`) to survive JavaScript's
  integer truncation in the console.

### 6.5 Completion signalling

Answering handoff question 3. I publish `risk.result.ready` / `risk.result.failed` carrying
`{jobId, requestId, status, resultUri, inputHashes}` — **and treat it as an optimization, not
the contract**. The coordinator must be able to poll `GET /risk/results/{jobId}` and reach the
same conclusion. Symmetric with my amendment (1): neither side should depend on a best-effort
publish for correctness.

**Completion order is not portfolio order.** A result carries its input identity; the console
must compare source versions before replacing a displayed result. A successfully computed old
result is still historical.

---

## 7. Open decisions, with my positions

| ID | My position |
|---|---|
| D01 | Agreed — TraderX publishes the versioned contract package; I pin a released hash. I need an unsupported-schema rule that **fails loudly** rather than ignoring unknown fields |
| D03 | **Blocking for W2.** Need the full convention set of §2.2 before I write the builder; I will not infer them |
| D04 | Explicit booked dates preserved exactly; documented supported-convention subset initially, everything outside it refused |
| D05 | **Agreed as split in 02-contracts**: TraderX supplies dated observations, I own curve construction and return repricing diagnostics |
| D06 | **I propose amending the default.** 02-contracts suggests cash equities + one USD-SOFR product. I propose **cash equities + Treasuries** for W1 (both deterministic, no open conventions), with USD-SOFR as W2's dedicated focus once D03 closes — see §3.1 |
| D08 | Positions in §4. `pvChangeForPlus1bp`, absolute-bump gamma scaling, normal vega, no "DV01" in the contract |
| D12 | I propose per-product absolute + relative tolerances against an ORE reference at matched terms, **plus** separately reported Monte Carlo standard error — numerical error and sampling error must never be reported as one number |
| D13 | Agreed: coordinator owns the durable job, I own the attempt. My `jobId` is not a system of record |
| D16 | **Blocking for swaptions (W3/W4).** Need holder direction and settlement type as distinct fields (§2.3) |

---

## 8. What I am explicitly not claiming

Stated plainly, because 07-baseline's "stale assumptions" section is right that my own docs
can read as stronger than the evidence supports:

- Passing unit tests here establish **internal** consistency and ORE parity on **my** terms.
  They establish nothing about TraderX's actual booked contracts.
- The engine has **never** priced a real TraderX position. No combined run has been performed.
- Existing swap/swaption pricers do **not** imply the booked contracts are representable.
  Schema coverage is not valuation readiness — the handoff says this about its own exports and
  it is equally true of my pricers.
- The aged-swap gap (§1.4c) is now **flagged**, not **fixed**. A warning improves honesty, not
  accuracy: exposure and VaR/ES at aged steps carry the same inaccuracy they did before.
- A seed is **not** a cross-device bitwise guarantee. Reproducibility within declared numerical
  tolerance and byte-identical replay are different properties, and only the latter is
  appropriate for anything entering consensus.
- fp32 and TPU results are research outputs until the error-vs-decision analysis (A-N06/A-N07)
  shows they do not change a hedge or limit decision. **Faster is not a financial argument.**

---

## 9. Proposed next steps

1. **Exchange fixtures before schemas are final.** TraderX emits one equity position, one
   Treasury, and one USD-SOFR swap as real bytes from the real exporter; I emit a result
   document for each. Disagreements surface in hours rather than after an adapter is built.
2. **Close D03/D04.** They block W2, and W2 is where most of the value is.
3. **I build W1** against the fixtures — Treasury + equity pricers, opaque IDs, and
   `/capabilities`. These are unblocked by any open decision. (Per-instrument NPV, swap
   Greeks and Bermudan Vega are already done — see §3.3/§3.4.)
4. **Agree the §6.4 result shape**, which the console can build against before any pricer
   behind it is finished.
5. **Then W2**, with a same-terms ORE reference comparison as the acceptance gate — not my own
   test suite.
