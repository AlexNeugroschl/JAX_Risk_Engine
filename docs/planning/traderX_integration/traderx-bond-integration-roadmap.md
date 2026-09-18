# TraderX ↔ JAX Risk Engine: Bond Integration Roadmap

**Status:** Planning document for the JAX side. Nothing described in section 3 onward has been
implemented in this repository. The TraderX side described in section 2 is **not** planning —
`YU16-cdm-instruments` is implemented, proven on a live `kind` cluster, and its own acceptance
suite passed clean (21/0; see `specs/YU16-cdm-instruments/generation/implementation-status.md`
on that branch) before this document was written. This plan builds **against `YU16-cdm-
instruments` as the base branch**, not against a hoped-for future state — every TraderX-side
fact below (schema, conventions, validation rules, live proof output) is what that branch
already does today, not what it is expected to do eventually.

**Supersedes:** nothing — this is additive to
[`traderx-integration.md`](traderx-integration.md), which covers hardening the engine's
existing config surface (PSD validation, cross-field consistency, pillar assembly) for
arbitrary portfolios in general. This document is the concrete instrument-and-transport plan
for the specific opportunity described in the integration-status memo: TraderX now holds five
fixed-rate U.S. Treasuries, which puts it on the same risk factor — the zero curve — that this
engine is built around. Read `traderx-integration.md` first if the goal is "harden the engine
for any portfolio"; read this one for "price TraderX's actual bonds, on the branch that already
has them."

**Grounding.** Every TraderX fact below was checked against `reference/traderX`'s
`origin/YU16-cdm-instruments` branch tip (`1829fc70`, "final suite result — 21/0 clean") —
specs, ADRs, data model, sample fixtures, the branch's own `implementation-status.md` and its
live-proof transcripts — and `origin/YU15-eod-risk-extract`'s consumer guide, not against the
integration-status memo's prose alone. Every JAX-engine fact was checked against the current
`engine/` tree. Where the two sources disagree in a small way (e.g. exact column order), this
document follows the repo, not the memo.

---

## 1. The one-sentence plan

Treat a fixed-rate Treasury as **a swap's fixed leg plus a bullet principal repayment**,
discounted off a zero curve calibrated to the same five Treasuries' own market prices — reusing
`engine/models/ore_builders.py`'s cashflow-schedule machinery and `engine/models/hull_white.py`'s
discounting, adding one new thin instrument module and one new curve-bootstrapping module, and
wiring a batch ingestion path that reads TraderX's EOD risk extract (schema 2) directly.

No optionality, no Monte Carlo, no Bermudan-style backward induction, and — this matters for the
"how much precision/parallelization work" question the user asked about — **no meaningful
Monte Carlo simulation burden at all** for the bonds themselves. That is the central finding of
section 4.

---

## 2. What TraderX actually holds and hands off (verified, not assumed)

### 2.1 The five Treasuries

From `specs/YU16-cdm-instruments/data-model.md` on `origin/YU16-cdm-instruments`:

| `instrumentKey` | coupon % | maturity | term | quoted clean % | stored fraction |
|---|---|---|---|---|---|
| `UST-20280630` | 4.125 | 2028-06-30 | 2Y | 99.878 | 0.998780 |
| `UST-20310630` | 4.125 | 2031-06-30 | 5Y | 99.665 | 0.996650 |
| `UST-20360515` | 4.375 | 2036-05-15 | 10Y | 99.257 | 0.992570 |
| `UST-20460515` | 5.000 | 2046-05-15 | 20Y | 98.481 | 0.984810 |
| `UST-20560515` | 5.000 | 2056-05-15 | 30Y | 99.293 | 0.992930 |

Fixed coupon, semiannual, bullet redemption at par, `US_TREASURY_NOTE`/`US_TREASURY_BOND` in
CDM terms. Nothing here has an exercise decision, a floating leg, or a volatility surface.

### 2.2 The fraction-of-par convention (ADR-057) — binding on the JAX side too

TraderX stores every bond price as a **fraction of par**, never a percentage: `99.886%` is
`0.998860`. This is a TraderX-internal decision (it keeps their deterministic matching core's
integer contract-multiplier arithmetic correct) but it is also **exactly what a bond pricer
needs anyway** — `price = clean_price_fraction × face` is the natural unit for a discounted-
cashflow calculation. No conversion is needed on ingest beyond parsing a `DECIMAL(18,6)` string
as a float; the convention is a gift, not friction.

### 2.3 The EOD risk extract (schema 2) — the actual transport

From `origin/YU15-eod-risk-extract`'s `docs/engineering/risk-extract-consumer-guide.md` and
`specs/YU16-cdm-instruments/data-model.md`:

- **Cadence:** one file per end-of-day session close, not a live feed. Delivery is announced on
  NATS subject `risk.extract.ready` with a JSON payload carrying `schema`, `uri`,
  `consensusSequence`, `sessionDate`, `priceSnapshotVersion`, `rows`, `sha256`, `cutSha256`.
- **Transport:** a write-once object — either `file://` (local dev, no cloud) or
  `gs://traderx-<project>-risk-extracts/<sessionDate>/v<priceSnapshotVersion>/seq-<N>.csv`, with
  a `.cut` sidecar for integrity verification.
- **Schema 2 columns** (verified against the sample fixture and ADR-059):
  ```
  accountId,security,instrumentType,quantity,contractMultiplier,costBasis,closingMark,
  markSource,markQuality,marketValue,unrealizedPnl,currency,counterpartyId,nettingSetId,
  coupon,maturityDate
  ```
- **Treasury-specific semantics:**
  - `instrumentType = TREASURY`, set by TraderX joining the extract cut against instrument
    reference data — never by parsing the ticker.
  - `quantity` is **face amount in USD**, not shares or contracts.
  - `contractMultiplier` is `1` (ADR-057's whole point — a bond is arithmetically an equity).
  - `costBasis`/`closingMark` are clean-price fractions of par, six decimals.
  - `coupon` (annual %, fixed, semiannual) and `maturityDate` are populated only for Treasury
    rows, empty otherwise.
  - **Accrued interest and dirty settlement value are deliberately excluded** — TraderX's
    memo says this explicitly and the schema confirms it: there is no accrual-fraction or
    dirty-price column anywhere in schema 2. `closingMark` is a clean price.
- **Fails closed:** the guide is explicit that a file which exists is complete — no partial
  rows, no zero-filled rows, no silently-skipped positions. The JAX side can trust `rows` in
  the header and every row's shape without defensive per-row validation of *completeness*
  (per-row *value* validation is still the JAX side's job — see section 6).

This is the single most load-bearing fact for the roadmap: **the integration is batch, not
streaming**, and the batch is a small, complete, self-describing, checksummed file — not a
live position feed the JAX engine needs to reconcile incrementally.

### 2.4 A real row, from the branch's own live proof — not a constructed example

`YU16-cdm-instruments`'s own live-`kind` proof (`scripts/proofs/yu15-risk-extract.sh` run with a
bond held, recorded verbatim in `implementation-status.md`) produced this schema-2 row:

```
22214,UST-20360515,TREASURY,100000,1,0.992570,0.992620,EOD_SNAPSHOT,OK,99262.000000,5.000000,USD,CPTY-CASCADE-AM,NS-CASC-ISDA-01,4.375,2036-05-15
```

Every field this document relies on is visible directly: `TREASURY` classification, 100,000 USD
face, multiplier `1`, `costBasis 0.992570` and `closingMark 0.992620` as six-decimal fractions
of par, `marketValue = 100000 × 0.992620 = 99,262.000000` exactly, `coupon 4.375` and
`maturityDate 2036-05-15` populated by the join. This is not a schema the JAX side needs to
prepare for — it is a schema the JAX side can parse against today, byte-for-byte, on the base
branch as it stands.

### 2.5 Order-entry constraints worth knowing before building an ingestion/round-trip layer

From the branch's `system/architecture.md` (Node Catalog, `gateway`): a Treasury order's face
amount is rejected pre-consensus unless it is **at least 100 and an exact multiple of 100** —
confirmed live in the proof suite (face 50 and face 150 both return 422; the cluster's applied
sequence never advances for a rejected order). This matters less for the read-only extract-
ingestion path this roadmap centers on, but it is directly relevant if the JAX side or a future
phase ever writes anything back toward TraderX (e.g. a what-if trade ticket) — `BondConfig.
face_amount` should validate this same rule at construction time rather than let a malformed
face amount reach TraderX's gateway and bounce.

---

## 3. What the JAX engine needs to build

### 3.1 New instrument: `engine/instruments/bond.py`

A fixed-rate bullet bond, built the same way every other instrument module in this codebase is
built — thin CPU-side ORE trade construction, JAX-native discounting.

**Design, following the existing pattern exactly** (`engine/instruments/swap.py` as the
template):

```python
@dataclass
class BondConfig:
    """A fixed-rate bullet Treasury: a fixed coupon leg (ORE.MakeVanillaSwap's
    fixed leg, reused via engine.models.ore_builders) plus principal
    redemption at maturity -- no floating leg, no optionality."""
    face_amount: float
    coupon_rate: float          # annual, e.g. 0.04125 for 4.125%
    coupon_frequency_months: int = 6   # semiannual, matches every UST issue
    discount_curve_index: int = 0
    maturity_date: ORE.Date = ...      # bullet-dated, not tenor-dated (bonds
                                        # have a fixed maturity, not a rolling
                                        # tenor from evaluation_date the way a
                                        # demo swap does)
    evaluation_date: ORE.Date = field(default_factory=...)
```

**Why not just call it a swap with `payer=False` and a zero float leg?** Because a bond has no
floating leg at all — building one via `SwapConfig` would mean threading a phantom
`forward_curve_index` and relying on the floating leg happening to net to zero, which is
fragile and undocumented. A dedicated `BondConfig`/`_build_ore_bond` (via
`ORE.FixedRateBond` or, more simply, `ORE.MakeVanillaSwap`'s fixed leg alone plus a manually
appended redemption cashflow) is a few dozen lines and avoids leaning on an accident of a
different instrument's math. `engine/models/ore_builders.py`'s `fixed_leg_cashflows` and
`DAY_COUNTER` are reused unchanged — this is exactly the kind of shared building block that
module exists for.

**Pricing formula** (closed form, no root-find, no Monte Carlo):

```
NPV(t) = sum_i  coupon_amount_i * P(t, T_i)   +   face * P(t, T_maturity)
```

using `engine.models.hull_white.discount`/`ZeroCurve` for `P(t, T)` — the same discounting
primitive `swap.py` already calls. There is no bond-option, no exercise boundary, no `_solve_
rstar`-style root-find anywhere in this instrument. It is the simplest pricer in the codebase.

**Maturity-dated, not tenor-dated.** Every existing `*Config` in this codebase (`SwapConfig`,
`SwaptionConfig`, ...) specifies `swap_tenor: str` (e.g. `"5Y"`) relative to `evaluation_date`,
because demo/test trades are conveniently spot-starting. A real Treasury has a **fixed calendar
maturity** (`2036-05-15`) that does not move as `evaluation_date` advances day to day — `BondConfig`
must take `maturity_date: ORE.Date` directly, not a tenor string, and the cashflow schedule must
be built backward from that fixed date (`ORE.MakeSchedule(evaluation_date, maturity_date, ...)`
or equivalent), not forward from `evaluation_date` by a tenor. This is a genuine, new
requirement — no existing config in this codebase takes an absolute maturity date, only a
relative tenor — and it is worth flagging early because it touches the same "date arithmetic"
territory the existing e2e test suite's own docstrings warn about repeatedly (calendar-day vs.
business-day advance, `ORE.Period` fractional parsing — a known trap in `engine/calibration/basket.py`,
worth re-checking for at `BondConfig` build time).

**Accrued interest — build it, don't inherit the exclusion.** TraderX's own extract excludes
accrued interest by explicit design (`closingMark` is a clean price). That is the right choice
*for TraderX's matching/booking layer*, where clean-price arithmetic keeps the deterministic
core simple. It is not automatically the right choice for a *risk engine* computing NPV,
sensitivities, or a curve calibration — a clean-price-only bond pricer is fine for relative
comparisons but understates true economic value by the accrued coupon between reset dates, and
a curve bootstrap fit to clean prices without accounting for the settlement-date accrual
convention introduces a small, systematic bias that grows with coupon size and shrinks near
a coupon date. `BondConfig`/`price_bonds` should compute **both** `clean_npv` and
`dirty_npv = clean_npv + accrued_interest`, where accrued interest is a standard
`day_count.yearFraction(last_coupon_date, settlement_date) × coupon_amount` calculation ORE's
own `DAY_COUNTER` already supports — a small addition, not a new subsystem, and it should be
exposed as an explicit extra field rather than silently only doing one or the other.

### 3.2 New module: `engine/calibration/treasury_curve.py`

The five Treasuries are not just tradeable instruments — they are exactly the market data a zero
curve needs. **This closes a gap the integration-status memo correctly flags as belonging to
neither system**: TraderX produces bond prices but no curve; the JAX engine consumes curves but
(today) has no curve-construction step, only hand-specified `ZeroCurveConfig` demo curves.

**Bootstrap, not calibration-in-the `engine/calibration/` sense.** This is a different problem
from `engine/calibration/lgm.py`'s job (fitting a *volatility* term structure to swaption
prices, holding the curve fixed). Here the curve *itself* is the unknown: given five bond
prices at five maturities, find the zero rates that reprice all five exactly. This is the
classic bootstrap problem — solved sequentially, shortest maturity first, each new pillar's
zero rate found by a 1D root-find that holds every earlier (shorter) pillar fixed, exactly
mirroring the triangular structure `engine/calibration/lgm.py`'s own docstring already explains
for a different (volatility, not rate) unknown. The same "why bootstrap, not joint
least-squares" reasoning documented in [`docs/reference/calibration.md`](../../reference/calibration.md)
applies verbatim, with "sigma bucket" replaced by "zero rate pillar."

```python
def bootstrap_treasury_curve(
    bonds: List[BondConfig],       # in increasing maturity order
    clean_prices: List[float],     # fraction of par, from the extract's closingMark
    evaluation_date: ORE.Date,
) -> ZeroCurve:
    """One zero rate pillar per bond, solved shortest-maturity-first, each
    new pillar's rate found by a 1D bisection holding every earlier pillar
    fixed at its own already-bootstrapped value -- reprices every input
    bond exactly (to root-find tolerance), matching the standard par/zero
    curve bootstrap used throughout fixed income."""
```

With only **five** points, this produces a coarse curve — fine for pricing the five bonds
themselves (which is circular but useful as a sanity check and for accrued-interest/dirty-price
work) but too sparse to be a trustworthy discounting curve for *other* instruments (a swap or
swaption priced off a 5-point Treasury curve would have large, unrealistic interpolation gaps
between 2Y and 5Y, and nothing beyond 30Y). **Scope this explicitly**: the bootstrap module is
for pricing/reconciling the Treasuries themselves and for internal consistency checks, not
positioned as a production discounting curve for the rest of the book unless TraderX's own
instrument universe grows enough pillars to support one.

### 3.3 New module: `engine/ingestion/traderx_extract.py`

A pure-parsing, no-network module that turns a downloaded extract CSV + `.cut` sidecar into
this engine's own config objects. Four responsibilities, each independently testable:

1. **Parse and verify.** Read the `#`-comment header (schema, `rows`, `cutSha256`), verify row
   count matches the declared `rows`, verify the `.cut` sidecar's SHA-256 against the header's
   `cutSha256` if the sidecar is available — exactly the integrity check the consumer guide's
   own reference Python snippet performs. Refuse to proceed on any mismatch; this is a direct,
   cheap adoption of TraderX's own "fail closed, so a file that exists is complete" guarantee —
   the JAX side should trust *completeness* once these checks pass, and spend its own validation
   budget on *value* sanity instead (section 6).
2. **Classify.** Split rows by `instrumentType`. `TREASURY` rows feed `BondConfig` construction
   (`face_amount = quantity`, `coupon_rate = coupon / 100`, `maturity_date` parsed from
   `maturityDate`). `EQUITY`/`OPTION` rows are explicitly out of scope for this engine (see
   section 5) — the parser should surface them as a labeled, skipped category in its return
   value, not silently drop them, so a caller can see the extract's full shape even though only
   the Treasury slice is priced.
3. **Convention translation.** `quantity` on a Treasury row is face amount in USD, already the
   unit `BondConfig.face_amount` wants — no conversion needed (this is the fraction-of-par
   convention paying off, per section 2.2). `costBasis`/`closingMark` are already fractions of
   par at 6dp, directly usable as clean prices.
4. **Netting awareness, explicitly not applied.** The consumer guide is emphatic that extract
   rows are **un-netted at `(accountId, security)` grain** — this engine's own VaR/ES machinery
   (`engine/risk/var_es.py`) operates on a portfolio of trades and produces one NPV cube; whether
   that portfolio is "one row per account" or "netted per `nettingSetId`" is a modeling decision
   the ingestion layer should make explicit and configurable (a `netting_mode` parameter),
   not an implicit choice buried in a for-loop. Match the extract's own posture: raw by default,
   netted only if the caller opts in.

### 3.4 What does NOT need to be built

- **No new Greeks machinery.** `engine/risk/greeks.py`'s existing `swap_delta_gamma`/
  `swap_theta` pattern (per-curve-pillar `jax.grad`/`jax.hessian`, 1bp-scaled, ORE-convention
  matching) applies to a bond's NPV function unchanged — a bond's delta with respect to a curve
  pillar is exactly the same kind of derivative a swap's fixed leg already produces. A thin
  `bond_delta_gamma`/`bond_theta` pair, mirroring `swap_delta_gamma`/`swap_theta` almost line
  for line, is the right shape — not a new Greeks paradigm.
- **No Vega.** A fixed-rate bond has no optionality; Vega is not a meaningful quantity for it,
  the same way `engine/risk/greeks.py`'s own docstring already explains for a plain swap.
- **No calibration engine changes.** `engine/calibration/lgm.py`/`basket.py` exist to fit a
  *volatility* term structure for Bermudan/American pricing. A bond needs none of that
  machinery — the new `treasury_curve.py` bootstrap (section 3.2) is a different, much simpler
  problem and deliberately lives in its own module rather than overloading `engine/calibration/`'s
  existing meaning.
- **No new simulation/scenario machinery.** `engine/simulation/market_model.py`'s `generate_paths` already
  produces a `[Scenarios, TimeSteps, NumRates, 2]` yield-curve cube; conditional bond pricing
  at a future simulated time slots into that cube exactly the way `swap.py`'s `_price_one_swap`
  already does (with the same aged-instrument caveat noted in section 6).

---

## 4. Precision, parallelization, and Monte Carlo: how much does a bond actually need?

The user's question deserves a direct, honest answer rather than a generic "GPUs help"
statement, because the honest answer is **"almost none, and that is the finding, not a
limitation."**

### 4.1 t=0 / single-date pricing: zero Monte Carlo, embarrassingly parallel across trades

A fixed-rate bond's NPV is a **closed-form sum of discounted cashflows** — no stochastic
simulation of any kind is needed to price one bond at one valuation date. This is qualitatively
different from every optionable instrument already in this engine:

| Instrument | Needs Monte Carlo / simulated paths? | Why |
|---|---|---|
| Swap | No (closed form) | Deterministic cashflows given a curve |
| European swaption | No (closed form, Jamshidian) | Analytic bond-option formula |
| Bermudan/American swaption | No paths, but yes to a numeric grid (Hagan quadrature convolution) | Early-exercise backward induction needs a state-space discretization, not Monte Carlo, in this engine's own design |
| **Fixed-rate bond** | **No — pure discounted cashflow sum** | No optionality, no path-dependence, nothing stochastic in the payoff at all |

So for pricing the five Treasuries at today's close, there is **no simulation to parallelize**
in the Monte Carlo sense. What *does* parallelize, and where JAX still earns its keep:

- **Vectorizing across the bond portfolio.** If TraderX's book eventually holds more than five
  Treasuries (or the same five held across many accounts, per the extract's un-netted grain —
  section 3.3), pricing `N` bonds simultaneously is a single `jax.vmap`'d discounted-cashflow
  sum, not `N` sequential Python calls — the same pattern `price_swaps` already uses for
  swap portfolios in `engine/instruments/swap.py`.
  - **Scale, honestly assessed:** with TraderX's current five Treasuries, `N` is small enough
    (single digits to low hundreds once per-account rows are un-netted) that vectorization is a
    correctness/consistency nicety, not a performance requirement — a Python loop over five
    bonds would run in microseconds either way. This becomes a real parallelization concern only
    if the account/position universe grows by orders of magnitude, which is a TraderX-side
    scaling question, not a bond-pricing-math one.
- **Autodiff Greeks, which the existing engine already treats as "free" parallelization.**
  `bond_delta_gamma` (section 3.4) computing key-rate DV01 across every curve pillar for every
  bond in the portfolio is one `jax.vmap(jax.grad(...))` call — this is where accelerator
  (GPU/TPU) throughput genuinely matters, especially once the calculation is repeated across the full simulated
  scenario grid for a VaR run (next point).

### 4.2 VaR/ES over simulated scenarios: this is where the real parallel workload is — and it is the SAME workload the engine already runs, not a new kind of one

The memo's own diagram is right that this is a large dense tensor operation with no branching —
but it is worth being precise about *what* is being parallelized, because it is not "many Monte
Carlo paths per bond" (there is no path-dependence to simulate) — it is **"many independent
scenarios of the curve, each producing one deterministic bond reprice."**

Concretely, using this engine's existing `generate_paths`/`compute_risk_metrics` pipeline
unchanged in shape:

1. `engine.simulation.market_model.generate_paths` produces `[Scenarios, TimeSteps, NumRates, 2]` — e.g.
   `50,000` correlated Hull-White scenarios of the Treasury curve's own short rate(s), sampled
   via Sobol quasi-Monte Carlo (this is the one place "Monte Carlo" genuinely applies — it is
   simulating the **future curve**, not the bond's payoff).
2. For each `(scenario, step)` pair, every bond's NPV is the SAME closed-form discounted-
   cashflow sum from section 4.1, evaluated against that scenario's own conditional discount
   factors — `50,000 × TimeSteps × 5 bonds` independent, branchless evaluations of one formula.
3. `compute_risk_metrics` (`engine/risk/var_es.py`) reduces the resulting NPV cube to VaR/ES
   percentiles across the scenario axis.

This is precisely the shape `price_swaps`/`price_swaptions` already run today for the demo
portfolios — a bond adds **zero new computational pattern**, only a new (simpler) per-scenario
formula slotted into an existing vectorized pipeline. The "precision research" framing in the
user's question is worth answering directly: **there is no new precision/convergence question
to research for the bonds themselves** (a closed-form sum has no Monte Carlo standard error,
no convergence rate, no path-count-vs-accuracy tradeoff to tune) — the precision questions that
*do* exist belong entirely to the **curve simulation** (scenario count for VaR/ES convergence,
already a solved, tested part of this engine — see `engine/risk/var_es.py` and its existing
test suite) and to the **calibration bootstrap** (section 3.2's root-find tolerance, a solved
problem pattern directly reused from `engine/calibration/lgm.py`'s own bisection convergence,
already verified to ~1e-10 RMSE in that module's tests).

**Bottom line for the roadmap:** budget essentially zero engineering time on
Monte-Carlo-precision tuning for the bond pricer itself. Budget the *existing* scenario-count/
convergence testing pattern (`tests/test_var_es.py`'s methodology) applied to a bond-inclusive
portfolio, which is a test-writing task, not a numerical-methods research task.

---

## 5. What stays explicitly out of scope

Matching the memo's own "what is still missing" list, made concrete against the repo:

- **Equities, ETFs, and listed options.** TraderX's `EQUITY`/`OPTION` rows have no pricer in
  this engine and none is proposed here — `engine/simulation/market_model.py`'s equity legs are GBM paths
  used only as a *background correlated factor* for rates (via the `rate_mapping` UIP drift),
  never priced as their own instrument. Extending this engine to price listed equity options
  would be a materially different, unrelated project (an equity-vol surface, an American-equity-
  option pricer) and is out of scope for a "bonds are the easy win" roadmap.
- **Swaps and floating-rate instruments on TraderX's side.** The overlap today is
  factor-and-math (both live on the zero curve), not instrument identity — TraderX holds no
  swap. Nothing in this roadmap requires TraderX to add one.
- **XVA (CVA/DVA).** The extract already carries `counterpartyId`/`nettingSetId`, so the data
  is ready whenever this engine builds XVA — genuinely a "later" item, not a blocker for the
  bond work above.
- **Live/streaming integration.** Section 2.3 established the transport is batch (one file per
  EOD close via NATS announcement + object store), not a live position feed. A "consume
  `risk.extract.ready` and process automatically" pipeline (section 8, item 6) is a reasonable
  automation target, but there is no case here for a continuously-updating intraday risk
  calculation — that would be a different, larger integration than what TraderX currently
  produces (see also section 7.5).

---

## 6. Known limitations to carry forward, not silently inherit

- **The aged-instrument discounting gap applies to bonds too.** `engine/instruments/swap.py`'s
  documented limitation — conditional pricing at a simulated time past a trade's first accrual
  date doesn't correctly represent an already-fixed coupon, because the yield-curve cube's
  `P(t, T)` is only meaningful for `T ≥ t` — applies identically to a bond's own coupon
  schedule once VaR/ES conditions on future simulated dates. This is not a new bug to introduce;
  it is an existing, tested, documented gap (`TestAgedSwapKnownLimitation`) that the bond pricer
  inherits by construction (same discounting primitive) and should inherit the same explicit
  documentation for, not a silent one.
- **`ORE.Period` fractional-string parsing.** `engine/calibration/basket.py` hit and fixed a
  real bug where `ORE.Period("0.75Y")` silently parses to `0Y` instead of raising. `BondConfig`'s
  maturity-date-based schedule construction (section 3.1) sidesteps this specific failure mode
  by using an absolute date rather than a tenor string, but any *other* period arithmetic
  introduced during bond-schedule construction (e.g. deriving "years to maturity" for display
  or for the bootstrap's own pillar ordering) should be built with this exact failure mode in
  mind and tested against it directly, the way `tests/test_calibration_edge_cases.py`'s
  `TestSubYearAndFractionalTenors` class already does for the calibration engine.
- **Accrued interest is a genuine gap on both sides today**, not just a TraderX omission (see
  section 3.1). Building it into the JAX bond pricer closes it on this side; TraderX's own
  extract will still report clean prices only unless a future TraderX-side change adds it —
  this is the first, and highest-value, item in section 7 below.
- **The Treasury curve bootstrap (section 3.2) is coarse by construction (5 points).** Do not
  let it become an implicit "the" discounting curve for other instruments without a deliberate
  decision to do so, documented the way `engine/models/lgm.py`'s own docstring documents the
  HW1F-vs-LGM non-equivalence finding — a scope boundary worth stating in code, not just here.

---

## 7. What TraderX-side people could do to bridge the gap further

Everything in sections 3-6 is buildable entirely on the JAX side against `YU16-cdm-instruments`
exactly as it stands today — none of it is blocked on TraderX changing anything. This section is
a separate list: asks that, if TraderX's own team is willing to pick them up, would make the
integration meaningfully better rather than merely possible. Ordered by value delivered per unit
of TraderX-side effort, cheapest/highest-value first. Every item here is scoped to be additive
to `YU16-cdm-instruments` the same way `YU16` was additive to `YU15` (per ADR-058's own
"additive, never a rename" precedent) — nothing below proposes changing extract schema 1/2
column semantics that already ship.

### 7.1 Publish the accrued-interest inputs the extract already has upstream of it (cheap, high value)

The extract's own header states accrued interest is *excluded*, not that the data to compute it
doesn't exist — `DebtEconomics.fixedInterest.couponFrequency` and `issueDate` are already on the
instrument static (`reference-data/instruments.csv`, per the YU16 data model), and every Treasury
position's `costBasis`/`closingMark` already carry a clean price. The only missing piece is the
**last coupon date** (or, equivalently, the day-count fraction since it) as of the extract's own
`sessionDate` — a pure function of `issueDate`, `couponFrequency`, and `sessionDate`, computable
entirely from data the extract producer already has in hand at render time, needing no new
market data source. Two ways to expose it, in order of preference:

- **Add a `lastCouponDate` (or `accruedInterestFraction`) column to schema 3**, following the
  exact precedent ADR-059 already set for schema 2 (a derived-by-join column, appended after the
  existing ones, with every prior column's name/position/meaning preserved) — the cheapest,
  most consistent-with-precedent option, and the one this document recommends.
- **Alternatively**, expose it as a field on the `/instruments/{key}` CDM record instead of the
  extract (since it's a function of static data, not position data) — workable, but less useful
  to a risk consumer than having it land row-by-row in the same file the position already comes
  from.

This single addition lets the JAX side drop the "build accrued interest itself" work in section
3.1 down to a formatting/validation step rather than a from-scratch day-count calculation,
and — more importantly — gives both sides the same number to check against, rather than the
JAX side computing accrued interest independently and hoping it agrees with whatever TraderX's
own booking/settlement layer would compute if it ever needed to.

### 7.2 A `priceProvenance`-shaped column (or sidecar) for the CURRENT day, not just the seed

`DebtEconomics.priceProvenance` (data model section, TraderX side) already carries the
TreasuryDirect auction provenance for the **seed** price at issue — genuinely useful for the
build-order item 1 cross-check in section 8, and the verification-plan cross-check in section
9, but a one-time, issue-date snapshot, not a running one. A risk engine benefits from knowing, for the price actually marking a position *today*,
whether that mark came from `price-publisher`'s own simulated walk (which it does, per section
2 — TraderX's own memo and this document are both explicit that current prices are synthetic,
not market-sourced) versus some future state where a real feed exists. The extract's existing
`markSource`/`markQuality` columns already carry exactly this kind of provenance for the *general*
mechanism (`EOD_SNAPSHOT` vs `CLUSTER_LAST_TRADE_AT_N`, `OK`/`OVERRIDDEN`/`STALE`/`LAST_TRADE`) —
no new column is needed here, only a documentation confirmation (in the consumer guide) that
these same enums apply identically to Treasury rows, since today's guide text and worked
examples are equity/option-flavored throughout. **This is a documentation ask, not a code
change** — cheap enough to bundle with item 7.1 rather than schedule separately.

### 7.3 A settlement-date/trade-date distinction on Treasury rows

Treasuries conventionally settle T+1 (T+2 in some markets); the current extract's `costBasis`
is a trade-price average with no settlement-date field, which is fine for the un-netted,
trade-date position view the extract already documents itself as being, but becomes relevant
the moment accrued interest (item 7.1) is added — accrued interest is conventionally computed to
**settlement date**, not trade date, and the two can differ by the coupon-fraction of a day or
two. This is a smaller, second-order ask relative to 7.1: worth raising once 7.1 is in flight,
not before, since accrued interest needs to exist before its settlement-date sensitivity matters
at all. If TraderX's own booking model doesn't track a distinct settlement date today (plausible,
since the memo describes settlement mechanics as explicitly out of scope for the current state),
the honest, lower-effort answer may simply be "the extract computes accrued interest to trade
date and says so in the header" — a documented convention choice, in the same spirit as ADR-057's
"the percentage is display only" framing, rather than a new subsystem.

### 7.4 More than five points on the curve, if and when the instrument universe grows

Section 3.2 is explicit that a 5-point Treasury curve is too coarse to be a production
discounting curve for anything beyond the five bonds themselves. This is not something to ask
TraderX to fix directly (adding Treasury issues purely to densify a curve for a downstream
consumer would be a strange reason to grow TraderX's own instrument universe) — it is listed
here only so that if TraderX's own roadmap independently adds more Treasury maturities, on-the-
run/off-the-run pairs, or a second currency's sovereign curve for some other reason, the JAX
side should treat that as a trigger to revisit the "coarse bootstrap, explicitly not a
production curve" scope boundary in section 3.2 and section 6, not as a coincidence to ignore.

### 7.5 A push notification path, if the batch cadence ever stops being enough

Section 5 already scoped continuous/intraday risk as out of scope for what TraderX currently
produces, and section 8 (build order) already proposes a `risk.extract.ready` NATS subscriber as
an *optional* automation step entirely buildable from the JAX side against the existing
mechanism. Nothing further is asked of TraderX here — the durable, watermarked NATS channel
already exists and already does the job. This item exists in this list only to say explicitly:
if a future need for tighter latency arises, the right question to ask TraderX's team is "can
the EOD chain fire more than once a day," not "can you build a new streaming feed" — the
watermarked-consumer mechanism this repo already has (`risk.extract.ready`) generalizes to a
higher cadence without a new integration pattern, only a scheduling change on TraderX's side.

---

## 8. Suggested build order

1. **`engine/instruments/bond.py`** (section 3.1) — standalone, no dependency on the others,
   directly testable against the five known Treasury prices/coupons/maturities as fixtures.
   Verify clean NPV at issue-date-adjacent prices against the TreasuryDirect auction provenance
   TraderX's own data model already carries (`priceProvenance.officialCleanPrice` — a free,
   independent cross-check with no new data-sourcing work).
2. **`engine/ingestion/traderx_extract.py`** (section 3.3) — standalone parsing/verification
   logic, testable directly against the schema-1 sample fixture in `reference/traderX`'s
   `specs/YU15-eod-risk-extract/contracts/sample/risk-extract.csv` and a schema-2 fixture built
   from the ACTUAL row reproduced in section 2.4 above (a live-proven artifact, not a
   constructed guess at the shape) — no live TraderX deployment needed to build or test this
   (the consumer guide itself makes this point: producing a fresh real extract only needs
   Docker+kind locally, and the recorded live row is sufficient for the parser's own test suite
   either way).
3. **`bond_delta_gamma`/`bond_theta` in `engine/risk/greeks.py`** (section 3.4) — depends on
   item 1 existing; mirrors `swap_delta_gamma`/`swap_theta` closely enough that it should be a
   short addition once the pattern is followed, not a new design.
4. **Wire ingestion → pricing → VaR/ES end to end**: `traderx_extract.py`'s parsed `BondConfig`
   list feeds directly into the existing `engine.simulation.market_model.generate_paths` +
   `engine.risk.var_es.compute_risk_metrics` pipeline (section 4.2), no new orchestration
   pattern needed beyond what `tests/test_diverse_portfolio_e2e.py` already exercises for other
   instrument types.
5. **`engine/calibration/treasury_curve.py`** (section 3.2) — depends on item 1's `BondConfig`
   existing; lowest priority of the five since it is not required to price the bonds against a
   hand-specified `ZeroCurveConfig` (which is sufficient for items 1-4) — it becomes valuable
   once there is a reason to derive the curve from TraderX's own marks rather than an externally
   supplied one.
6. **(Optional, automation) A `risk.extract.ready` NATS subscriber** that triggers steps 2-4
   automatically on each EOD announcement, rather than a manually-invoked batch script — a
   reasonable "phase 2" once the manual pipeline (items 1-4) is proven correct, matching the
   consumer guide's own suggestion ("if you want your engine to react to extracts rather than
   poll for them, subscribe to that subject").
7. **(Parallel track, not sequential)** Raise section 7's items 7.1/7.2 with TraderX's team as
   soon as item 1's accrued-interest implementation is underway — the ask lands with more
   weight once the JAX side can point at its own working (if independently-derived) accrued-
   interest calculation and ask "can we cross-check this against your own numbers instead of
   each deriving it separately."

---

## 9. Verification plan

- **Instrument-level:** `tests/test_bond.py` (new) — clean NPV against the five real Treasury
  fixtures at multiple valuation dates, dirty NPV/accrued-interest correctness against a
  hand-computed reference, degenerate cases (valuation date exactly at a coupon date, valuation
  date at/after maturity → NPV collapses to the final redemption or 0), and a face-amount/
  notional-invariance check mirroring the pattern already used in
  `tests/test_calibration_edge_cases.py::TestNotionalAndScaleInvariance`.
- **Ingestion-level:** `tests/test_traderx_extract_ingestion.py` (new) — parses both the real
  schema-1 sample fixture (`reference/traderX`) and a constructed schema-2 fixture matching the
  YU16 data model exactly (header comments, column order, six-decimal fraction-of-par values);
  asserts the `rows`/`cutSha256` integrity checks both pass on well-formed input and reject
  truncated/tampered input; asserts `EQUITY`/`OPTION` rows are surfaced as skipped, not dropped
  silently.
- **Greeks-level:** finite-difference cross-checks for `bond_delta_gamma`/`bond_theta`, following
  the exact methodology already established in `tests/test_greeks.py`'s `TestSwapDeltaGamma`
  class (this engine has hit real autodiff-through-a-root-find gradient bugs twice already —
  see `docs/risk/greeks.md`'s "two real bugs" section — so a value-level finite-difference
  check, not just a finiteness/sign check, is the standard this codebase holds itself to).
- **Curve bootstrap:** `tests/test_treasury_curve.py` (new) — reprices all five input bonds
  exactly off the bootstrapped curve (RMSE near float tolerance, matching the exact-reprice
  standard `engine/calibration/lgm.py`'s own bootstrap is held to), plus a degenerate-input
  test (e.g. an inverted or unrealistic input price set) documenting the failure mode explicitly
  rather than leaving it undiscovered, following the precedent set by
  `tests/test_calibration_edge_cases.py::TestExtremeMarketVols::
  test_large_dip_after_a_spike_cannot_reprice_exactly`.
- **End-to-end:** extend `tests/test_diverse_portfolio_e2e.py` (or add a sibling
  `tests/test_traderx_bond_portfolio_e2e.py`) with a portfolio built directly from a parsed
  extract fixture — ingestion → bond pricing → simulated-scenario VaR/ES → sanity checks on
  the resulting distribution (finite, appropriately signed, consistent with the aged-instrument
  caveat noted in section 6) — the same "diverse portfolio" rigor this codebase already applies
  to swaps/swaptions, extended to cover the actual TraderX data shape rather than a hand-built
  one.
- No changes to any existing pricer, calibration routine, or Greeks function are required by
  this plan — the full existing test suite must continue to pass unchanged throughout, exactly
  the standard `traderx-integration.md` already holds itself to.
