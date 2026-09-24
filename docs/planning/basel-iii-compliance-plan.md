# Basel III Compliance Plan

**Status:** proposed · **Written:** 2026-09-24 · **Scope:** market risk (FRTB), with
counterparty credit and CVA as later phases

This plan covers three things: what "Basel III compliant" can honestly mean for this
engine, the work needed to get there, and how to test and prove each regulatory number it
emits. Every task names the files it touches, the test that proves it, the oracle that
test compares against, and the exit criterion that closes it.

---

## 1. What "compliant" can and cannot mean

A bank is compliant with Basel III, not a piece of software. Supervisors approve a bank's
models, governance, data and controls. An engine can do three things:

1. **Compute each regulatory figure exactly as the Basel Framework specifies it**, for a
   stated set of instruments and risk classes.
2. **Refuse, with a reason, anything outside that set**, and never approximate it.
3. **Produce evidence** that lets a validator or supervisor confirm (1) and (2) without
   trusting the authors.

So the target claim for this project is:

> For portfolios inside the supported scope, every regulatory figure the engine emits is
> computed as the Basel Framework specifies. Each figure is traceable to the paragraph
> that defines it, reproduced by an independent oracle within a stated tolerance, and
> shipped with a reproducible evidence pack. Anything outside scope is refused with an
> identified reason.

The engine cannot produce the bank-side items: trading/banking book designation (MAR12),
desk structure approval, the independent risk control function, senior management
oversight, or supervisory approval of the IMA. The engine **carries** the metadata those
processes need (desk, book, trade identity), but it does not decide them. Section 10
lists these explicitly so nobody reads them as covered.

### Reference text

Unless a decision below changes it, the reference text is the **BCBS consolidated Basel
Framework** as currently published at bis.org, specifically:

| Chapter | Subject | Phase |
|---|---|---|
| MAR10–12 | Scope, definitions, trading/banking book boundary | P0 (metadata only) |
| MAR20–23 | Standardised approach: sensitivities-based method (SBM), default risk charge (DRC), residual risk add-on (RRAO) | **P1** |
| MAR30–33 | Internal models approach (IMA): ES, stressed calibration, NMRF, DRC, capital | P2–P4 |
| MAR32, MAR99 | Backtesting, P&L attribution, traffic light | P4 |
| CRE52 | SA-CCR | P5 |
| CRE53 | Internal model method (IMM) for CCR | P5 (optional) |
| MAR50 | CVA risk (BA-CVA, SA-CVA) | P5 |

Jurisdictional implementations (US, EU CRR3, UK PRA) differ in parameters, timing, and
some rules, and they change. They are handled as **regulatory profiles**: versioned
parameter files layered on the BCBS base (§5.3). The BCBS profile is built first, and a
jurisdiction profile is added only when someone needs it.

---

## 2. Where the engine stands today

The engine is accurate for what it computes: prices match ORE to ~1e-11 for Bermudans and
1e-3 per scenario end to end. **But what it computes is not what Basel asks for.** The
central gap is one of measure, not precision:

- `generate_paths` simulates under the **risk-neutral pricing measure**
  ([`engine/risk/var_es.py`](../../engine/risk/var_es.py), `ENGINE_RISK_MEASURE =
  "risk-neutral-pricing"`). Basel market risk capital needs **historical, real-world
  10-day P&L**. No amount of accuracy turns one into the other. The I-11 label already
  says so; the plan makes it enforceable (P0.5).
- The risk-neutral exposure cube `[Scenarios, TimeSteps, Trades]` is the right object for
  **counterparty exposure** (EPE/EEPE, CRE53). That makes P5 cheaper than it looks.

### Gap analysis

| Basel requirement | Today | Gap |
|---|---|---|
| SBM delta at prescribed GIRR tenors (0.25y … 30y) | AD delta per *engine* pillar, scaled to 1bp | Wrong vertices. AD derivative, not Basel's one-sided difference (§6, P1.2) |
| SBM vega to implied vol × vol | Bermudan vega per basket swaption, 1bp normal vol; no European swaption vega | Vertices, the × σ scaling, European coverage |
| SBM curvature (full reval at ±RW shift) | None | New |
| SBM aggregation, 3 correlation scenarios | None | New |
| DRC (SA) for Treasuries | None | New |
| RRAO | None | New |
| ES 97.5%, 10-day, liquidity-horizon cascade | VaR/ES at any percentile of a risk-neutral cube | Measure, horizon, LH cascade |
| Stressed calibration (reduced set, since 2007) | No market history at all | Data plus new code |
| RFET, NMRF, SES | None | Data plus new code |
| Backtesting 97.5%/99%, 250 days, traffic light, multiplier | None; no P&L time series | New |
| P&L attribution (Spearman, KS) | None; no HPL/RTPL | Needs an independent front-office P&L (D-7) |
| IMA capital (60-day averages, m_c) | None | New |
| SA-CCR | None | New; ORE oracle available |
| BA-CVA | None | New; ORE oracle available |
| Audit trail, reproducibility | EOD bundles hash-verified; portfolio path job store in-process | I-08, I-10 |

### Existing register items that block a compliance claim

Everything in [known-issues.md](../known-issues.md) that affects a regulatory number is a
precondition, not a side issue:

| Issue | Why it blocks | Blocks |
|---|---|---|
| [I-05](../known-issues.md#i-05) no faithful USD-SOFR swap construction | Every USD swap sensitivity rests on it | P1 for USD swaps (other books can proceed) |
| [I-04](../known-issues.md#i-04) aged swaps mispriced past first accrual | Wrong exposure at every step past first accrual | P5 (IMM, exposure). **Not** P1–P4, which revalue at t=0 |
| [I-10](../known-issues.md#i-10) no trade identity on the portfolio path | Desk attribution, backtesting per desk, audit | P0 |
| [I-24](../known-issues.md#i-24) bonds have no scenario NPV | Treasuries excluded from ES and curvature | P1 curvature, P3 |
| [I-18](../known-issues.md#i-18) no equity spot/FX | EQ risk class impossible | EQ only; refused until fixed |
| [I-27](../known-issues.md#i-27) full-suite runs abort inside XLA | An evidence pack needs a complete, reproducible suite run | P7 |
| [I-08](../known-issues.md#i-08) portfolio job store in-process | Regulatory runs must survive restart and stay retrievable | P0 |
| [I-32](../known-issues.md#i-32) ORE parity only at one solver config | The reference configuration must be fixed before it is cited as an oracle | P1 vega/curvature for Americans |

---

## 3. Decisions needed

Each decision has a recommended default. **Work can start on the defaults**; each
decision is recorded in `compliance/decisions.md` (P0.1) with date and rationale, and
changing one later is a tracked change with a re-run of the affected evidence.

| # | Decision | Recommended default | Why |
|---|---|---|---|
| D-1 | Reference text | BCBS consolidated framework; jurisdictions as profiles | Jurisdiction-neutral; overlays are data |
| D-2 | Frameworks in scope | **Must:** FRTB-SA (SBM, DRC, RRAO). **Should:** FRTB-IMA, backtesting, PLA, SA-CCR, BA-CVA. **Could:** IMM, SA-CVA. **Won't (now):** IMA-DRC | SA is mandatory for every bank, including IMA banks (fallback, PLA surcharge, output floor). It is fully specified, so it is the most testable. IMA-DRC needs a credit portfolio model the engine lacks |
| D-3 | GIRR delta risk factor | **Zero rates** at Basel vertices | Native to the engine's curves. Par-rate sensitivities can come later, with ORE's `ParSensitivityAnalysis` as oracle |
| D-4 | Treatment of US Treasuries | GIRR on the risk-free curve, plus CSR (sovereign bucket) on the Treasury–risk-free spread | The conventional reading. The alternative (Treasury curve as the GIRR curve, no CSR) must be a documented choice, not an accident |
| D-5 | ES/VaR estimator on 250 observations | Keep ORE's lower order statistic and strict tail (current `var_es.py`) and report the interpolated estimator beside it as a model-risk diagnostic | Basel does not prescribe the estimator. Parity with ORE is already proven. The difference is material at 6.25 tail observations, so it must be visible |
| D-6 | Historical shock type | Absolute for rates and vols; log-returns for FX and equity | Rates can be negative; relative shocks are undefined near zero |
| D-7 | Source of hypothetical/actual P&L for backtesting and PLA | TraderX front-office valuations (HPL, APL); engine produces RTPL | **If the engine produces both HPL and RTPL, PLA passes trivially and proves nothing.** An independent valuation is required for PLA to mean anything |
| D-8 | Market-data history | Public US Treasury constant-maturity yields (daily, back to before 2007) for the reduced risk-factor set and stress-period search; SOFR history (2018+) for the current period; swaption vols from a named vendor or declared NMRF | Only freely available history reaching 2007–2009. Vol history is the hardest item; without it, vol factors are NMRFs |
| D-9 | Precision of regulatory runs | FP64 only, until the FP32 gate (P6) passes for a given figure | Capital numbers need a precision guarantee. The precision research gives a path to lift this per figure |
| D-10 | ORE reference config for Americans (I-32) | Grid solver, `ShiftHorizon=0` (current), cited explicitly | It is the configuration parity is proven for. Revisit if a validator requires ORE's defaults |

---

## 4. Working rules for this plan

These extend the rules the known-issues register already follows.

1. **Parameters are data with citations.** No risk weight, correlation, threshold or
   horizon appears as a literal in code. Each lives in the profile file (§5.3) with the
   paragraph it came from.
2. **Double-entry transcription.** The profile is transcribed from the Basel text twice,
   independently, and the two files are diffed. Transcription is where capital
   calculators most often go wrong, and a single transcriber cannot catch their own
   misreading. The constants in Appendix B of this plan are **orientation only**, recalled
   rather than transcribed. They are not a source.
3. **Every regulatory function has an oracle that is not itself.** In order of preference:
   ORE running in-process; an independent reference implementation written from the text
   by someone who has not read the JAX code; a hand-computed golden case.
4. **Red first.** Every test is shown failing against a deliberately broken or pre-fix
   implementation before it counts as evidence, as the register already requires.
5. **Refuse rather than approximate.** An instrument, risk class, or risk factor outside
   scope produces an identified refusal (the existing `CalculationOutcome.unsupported`
   pattern), never a zero.
6. **The measure travels with the number.** A regulatory figure carries its measure,
   profile version, and run manifest. The engine refuses to compute an IMA figure from a
   risk-neutral cube (P0.5).
7. **A green suite is evidence about the tests, not proof about the code** (register rule
   9). Proof comes from oracles, mutation testing, and independent validation (§8).

---

## 5. Architecture

### 5.1 Package layout

```
engine/regulatory/
  __init__.py
  profile.py                 # load + validate a regulatory profile; exposes typed params
  profiles/
    bcbs.yaml                # BCBS base parameters, each with a paragraph citation
  measures.py                # new measure labels + guards (P0.5)
  manifest.py                # run manifest: git SHA, dirty flag, versions, input hashes
  sa/
    risk_factors.py          # Basel vertices, bucket/currency mapping
    sensitivities.py         # delta/vega per Basel definition (one-sided bump)
    curvature.py             # CVR up/down via full revaluation
    sbm.py                   # within-bucket, across-bucket, 3 correlation scenarios
    drc.py                   # SA default risk charge
    rrao.py                  # residual risk add-on
    capital.py               # SA total = max over scenarios of SBM + DRC + RRAO
  history/
    timeseries.py            # risk-factor time series store (Parquet), provenance
    shocks.py                # 10-day overlapping shocks, absolute/log per D-6
    scenarios.py             # HistoricalScenarioSet -> shocked curves at t=0
  ima/
    revaluation.py           # vmap full revaluation of the portfolio under a scenario set
    es.py                    # ES 97.5%, liquidity-horizon cascade
    stressed.py              # stress-period search, reduced-set ratio (floored at 1)
    imcc.py                  # IMCC = rho*unconstrained + (1-rho)*sum of risk classes
    rfet.py                  # risk-factor eligibility test
    nmrf.py                  # stressed ES per NMRF, SES aggregation
    capital.py               # CA = max(prev day, m_c * 60-day avg) + SES + DRC
  backtest/
    pnl.py                   # RTPL from the engine; HPL/APL ingest from TraderX
    backtesting.py           # exceptions at 97.5% and 99%, desk and bank level
    traffic_light.py         # zones and plus factor
    pla.py                   # Spearman, KS, zones
  ccr/
    saccr.py
    imm.py                   # EE, EEE, EPE, EEPE from the existing exposure cube
  cva/
    ba_cva.py
engine/validation/
  ore_app_oracle.py          # generalises ore_lgm_oracle.py to any OREApp analytic
compliance/
  requirements.yaml          # the requirement catalogue (Appendix A seeds it)
  decisions.md               # D-1..D-n, dated
  golden/                    # hand-computed cases (inputs + expected outputs + worksheet)
tests/regulatory/            # all tests for the above, marked @pytest.mark.basel(...)
```

### 5.2 Data flow

```
                 today's market ──► pricing curves (Basel vertices re-pillared)
                        │
       ┌────────────────┼──────────────────────────────┐
       ▼                ▼                              ▼
  SA: bump at      IMA: historical            CCR: risk-neutral paths
  Basel vertices   scenario set (t=0)         (generate_paths, existing)
       │                │                              │
  sensitivities    vmap full reval               exposure cube [S,T,N]
  + curvature      P&L vectors per LH/class            │
       │                │                        EPE / EEPE / SA-CCR
  sbm.py           es.py → imcc.py → capital.py        │
       │                │                              ▼
       └────────► RegulatoryResult (figure, measure, profile hash, manifest)
                                │
                        evidence pack (P7)
```

Every pricer already exposes a pure function `V(pillar_rates, …)` at t=0 for the greeks
(`_swap_price_fn`, `_swaption_price_fn`, `_bermudan_price_fn` in
[`greeks.py`](../../engine/risk/greeks.py)). That function is exactly what SA bumps,
curvature shifts, and historical full revaluation need: `jax.vmap` it over a batch of
curves. No pricer needs rewriting. This is also where the project's TPU research goal
pays off: stress-period search is thousands of full revaluations.

### 5.3 Regulatory profile format

```yaml
profile: bcbs
version: 2026.1           # bumped on any change; hash recorded in every result
source: "Basel Framework, MAR21 (as published at bis.org, retrieved YYYY-MM-DD)"
girr:
  delta_tenors_years: {value: [0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30], cite: "MAR21.8"}
  delta_risk_weights: {value: [...], cite: "MAR21.42"}
  specified_currency_rw_divisor: {value: 1.41421356, currencies: [...], cite: "..."}
  tenor_correlation_theta: {value: 0.03, cite: "..."}
  tenor_correlation_floor: {value: 0.40, cite: "..."}
  ...
```

`profile.py` validates the schema, rejects any key without a `cite`, and exposes frozen
dataclasses. Paragraph numbers are filled during P0.3 transcription. The ones above
illustrate the format and must not be copied.

### 5.4 New measure labels

Extend the vocabulary in `var_es.py` rather than overloading it:

| Label | Produced by |
|---|---|
| `risk-neutral-pricing` (existing) | `generate_paths` exposure runs; CCR |
| `historical-forecast` (existing, first producer) | `ima/` full revaluation over historical scenarios |
| `deterministic-stress` (existing) | curvature shifts, stress tests |
| `regulatory-standardised` (new) | SA capital: a formula over sensitivities, not a distribution |

---

## 6. Phases and tasks

Sizing is rough, in engineer-weeks for one engineer, and excludes time spent blocked on
external inputs. Each phase has an exit criterion. A phase is done when that criterion
is met, not when its tasks are merged.

### Phase 0 — Foundations (≈2 weeks)

| ID | Task | Files | Test / proof | Exit |
|---|---|---|---|---|
| P0.1 | Record D-1…D-10 with dates | `compliance/decisions.md` | Review | Owner has signed off each default or its replacement |
| P0.2 | Download the Basel chapters in §1 into `reference/basel/` with retrieval date and SHA-256 | `reference/basel/README.md` | Hash check test | Text pinned; every citation resolves to a pinned file |
| P0.3 | Transcribe the BCBS profile twice independently; diff; resolve against the text | `engine/regulatory/profiles/bcbs.yaml`, `profile.py` | `test_profile.py`: schema, every key cited, the two transcriptions agree | Zero diff between transcriptions; every value has a paragraph |
| P0.4 | Requirement catalogue plus traceability check | `compliance/requirements.yaml`, `tests/regulatory/test_traceability.py`, `basel` marker in `conftest.py` | The check fails if a requirement has no test, a test cites an unknown requirement, or a requirement's status is `implemented` with no passing oracle test. Include a negative test that feeds it a broken catalogue | Catalogue seeded from Appendix A; check green; negative test red on a broken catalogue |
| P0.5 | Measure guards: IMA functions accept only `HistoricalScenarioSet`; passing a risk-neutral cube raises | `measures.py` | Test that `ima.es` given a `generate_paths` cube raises with a message naming the measure | Guard in place, red-first shown |
| P0.6 | Close I-10 on the portfolio path: trade ID, desk, book, currency on every trade and result | `portfolio/request.py`, `api/schemas.py` | The I-10 closing tests the register already specifies | I-10 FIXED in the register |
| P0.7 | Run manifest on every regulatory result | `manifest.py` | Test: manifest has git SHA, dirty flag, package versions, JAX backend and dtype, profile hash, input hashes; two identical runs give identical output hashes | Deterministic reruns proven byte-identical on CPU FP64 |
| P0.8 | Generalise the ORE oracle to any OREApp analytic (sensitivity, stress, SA-CCR, BA-CVA, HistSimVaR, backtest) | `validation/ore_app_oracle.py` | Smoke test per analytic against an ORE Example's `ExpectedOutput` | Each analytic reproduces its ORE example output |
| P0.9 | Port the durable job store to the portfolio path (I-08) | `portfolio/`, `api/routes.py` | Existing I-08 closing criteria | I-08 FIXED |

### Phase 1 — FRTB standardised approach (≈6 weeks)

The capital figure every bank must produce. Covers GIRR fully, FX if any non-reporting-
currency position exists, CSR for Treasuries per D-4. EQ is refused until I-18 closes.

| ID | Task | Detail | Oracle | Exit |
|---|---|---|---|---|
| P1.1 | Risk-factor mapping | Re-pillar each pricing curve onto the union of its market pillars and the Basel vertices, so a vertex bump is an exact risk-factor move rather than an interpolated one. Map each curve to a GIRR bucket (currency) and curve type | Independent check that the re-pillared curve reprices the portfolio to 1e-12 | Base NPV unchanged by re-pillaring (1e-12 relative) |
| P1.2 | GIRR delta **per Basel's definition** | Basel defines `s = (V(r + 1bp) − V(r)) / 0.0001`, a one-sided difference. The engine's AD delta equals `∂V/∂r`, which differs from that by `½·Γ·1bp`. Compute the literal definition by vmapping `price_fn` over the bumped curves. Keep AD as a cross-check whose gap must equal `½·Γ·1bp` | ORE `SensitivityAnalysis` via `ore_app_oracle` with shift tenors at the Basel vertices, absolute zero shift of 1bp, forward scheme | Matches ORE to 1e-8 relative per vertex; AD gap equals `½Γh` to 1e-6 |
| P1.3 | GIRR vega | `s = (∂V/∂σ)·σ` at option-maturity × underlying-maturity vertices. Bermudan/American: from `bermudan_vega` (already by implicit function through calibration) mapped to vertices. European swaptions: add a vega from a market implied vol input (the Jamshidian pricer takes HW sigma today, so route through calibration as the Bermudan does) | ORE `SensitivityAnalysis` swaption vol shifts | 1e-6 relative to ORE per vertex |
| P1.4 | GIRR curvature | `CVR± = V(r ± RW_curv) − V(r) ∓ Σ RW·s` with a parallel shift of all vertices; `CVR = −min(CVR+, CVR−)` per bucket. Needs full reval: bonds included (closed-form, so I-24 does not block t=0 shifts) | ORE `StressTest` with parallel shifts; independent reference implementation | 1e-10 relative |
| P1.5 | SBM aggregation | Weighted sensitivities, within-bucket `K_b`, across-bucket with the alternative `S_b` when the square root argument is negative, curvature ρ² and the ψ indicator, **three correlation scenarios** (high/medium/low), total = max over scenarios of the sum across risk classes | Independent numpy implementation (§7.3); hand-computed golden cases | Agrees with the reference to 1e-12; all golden cases pass |
| P1.6 | FX delta (conditional) | Only if a position is not in the reporting currency. `s = (V(FX·1.01) − V(FX)) / 0.01` | ORE sensitivity, FX spot shift | 1e-8 relative |
| P1.7 | CSR for Treasuries (per D-4) | Sovereign spread = Treasury curve − risk-free curve, bumped at CSR vertices | ORE sensitivity on a spread curve; golden case | 1e-8 relative |
| P1.8 | SA-DRC | Jump-to-default per issuer, net long/short by seniority, hedge benefit ratio, risk weights by credit quality. Treasuries are the only in-scope issuer today | Golden cases; independent implementation | 1e-12 |
| P1.9 | RRAO | Gross notional × weight for flagged instruments. **Open question:** whether Bermudan/American swaptions attract RRAO. Resolve from the MAR23 text and BCBS FAQs before coding, and record the answer as a decision | Golden cases | Decision recorded; tests encode it |
| P1.10 | SA capital and API | `SaCapitalResult` with per-risk-class, per-bucket, per-scenario breakdown, measure `regulatory-standardised`, profile hash, manifest. Exposed via `/regulatory/sa` | Round-trip schema tests | Endpoint live; breakdown sums reconcile to the total exactly |

**Phase 1 exit:** a mixed portfolio (swaps in two currencies, European, Bermudan and
American swaptions, bills, notes) produces an SA capital figure that (a) matches the
independent reference implementation to 1e-12 given the same sensitivities, (b) takes
sensitivities that match ORE per vertex within the tolerances above, and (c) passes every
golden case. USD swaps are included only once I-05 closes; until then they are refused,
not approximated.

### Phase 2 — Market-data history and historical scenarios (≈4 weeks + data acquisition)

| ID | Task | Detail | Test | Exit |
|---|---|---|---|---|
| P2.1 | Time-series store | Parquet per risk factor: date, value, source, retrieval timestamp, provenance (`observed`/`proxied`/`filled`). Reuse `CurveProvenance` | Round-trip, provenance required, no silent forward-fill | Store holds D-8 series from 2007 onward |
| P2.2 | Data quality | Gap and stale detection, outlier flags, a log of every fill or proxy. Basel requires current-period data updated at least monthly | Tests with injected gaps and stale runs | Every fill is logged and surfaces in the manifest |
| P2.3 | Shock generation | 10-day overlapping shocks from daily data, absolute or log per D-6, applied to today's curve at each vertex | Hand-checked small series; property: zero history gives zero shocks | Exact against a golden series |
| P2.4 | Scenario set | `HistoricalScenarioSet(period, shocks, measure="historical-forecast")` feeding `ima/revaluation.py` | Measure guard (P0.5) applies | Guard test green |
| P2.5 | Full revaluation | `vmap(price_fn)` over the scenario set for the whole portfolio, including bonds | Per scenario, equals a looped single-scenario reprice to 1e-12; ORE `HistoricalSimulationVaR` P&L vector on the same scenarios to 1e-6 | Matches ORE P&L vectors |

### Phase 3 — IMA expected shortfall (≈4 weeks)

| ID | Task | Detail | Oracle | Exit |
|---|---|---|---|---|
| P3.1 | Base ES | ES at 97.5% of the 10-day P&L, estimator per D-5, reusing `var_es.expected_shortfall` | ORE `HistoricalSimulationVaR` ES; independent implementation | 1e-10 relative |
| P3.2 | Liquidity-horizon cascade | Assign each risk factor a liquidity horizon from the profile. `ES = sqrt(ES_T(P)² + Σ_{j≥2} (ES_T(P,j)·sqrt((LH_j − LH_{j−1})/T))²)`, where `ES_T(P,j)` shocks only the factors with LH ≥ LH_j | Independent implementation; golden case | 1e-12 |
| P3.3 | Stressed calibration | Choose a reduced factor set that explains ≥75% of full-set current ES. Search every 250-day window since 2007 for maximum reduced-set ES. `ES = ES_R,S · max(ES_F,C / ES_R,C, 1)` | Brute-force search on a small case equals the optimised search; golden case | The window found matches brute force; the 75% test is enforced and reported |
| P3.4 | IMCC | `IMCC = ρ·IMCC(C) + (1 − ρ)·Σ_i IMCC(C_i)`, ρ from the profile | Independent implementation | 1e-12 |
| P3.5 | RFET | Real price observations: ≥24 in 12 months with no 90-day window holding fewer than 4, or ≥100 in 12 months (verify against P0.3's transcription). Needs an observation feed, not just end-of-day marks | Golden observation calendars at each boundary (23/24 observations, a 90-day window with 3/4) | Boundary cases exact |
| P3.6 | NMRF and SES | Stressed ES per NMRF at LH = max(LH, 20). Aggregate with zero correlation for idiosyncratic credit and the profile's ρ for the rest | Independent implementation; golden case | 1e-12 |
| P3.7 | ES estimator diagnostic | Report the interpolated-quantile ES beside the ORE-convention ES, plus the existing MC standard error and tail count | Test that both are present and labelled | Present on every IMA result |

**Phase 3 exit:** IMCC and SES for the Phase 1 portfolio, matched to ORE where ORE
computes the same quantity and to the independent implementation everywhere else.

### Phase 4 — Backtesting, P&L attribution, IMA capital (≈3 weeks + D-7 dependency)

| ID | Task | Detail | Oracle | Exit |
|---|---|---|---|---|
| P4.1 | P&L series | RTPL: engine revaluation of yesterday's portfolio under today's modelled risk factors. HPL and APL: ingested from TraderX with trade IDs (needs I-10) | Reconciliation test: the per-trade sum equals the desk total exactly | Daily series stored with provenance |
| P4.2 | VaR backtesting | 1-day VaR at 97.5% and 99%, exceptions against both HPL and APL, desk and bank level, last 250 days. Desk thresholds (12 at 99%, 30 at 97.5%) from the profile | ORE `MarketRiskBacktest` | Exception counts exact |
| P4.3 | Traffic light and multiplier | Zone from the 99% exception count; plus factor from the profile table; `m_c = 1.5 + plus` | Basel's published table (golden, integer-exact); ORE `stopLightBounds` with RAG levels 0.95/0.9999; exact binomial CDF from scipy | Zones equal on 0…250 exceptions |
| P4.4 | PLA test | Spearman correlation and KS statistic between HPL and RTPL over 250 days; zones from the profile thresholds | `scipy.stats.spearmanr` and `ks_2samp` to 1e-12; golden boundary cases at each threshold | Exact zone assignment at every boundary |
| P4.5 | IMA capital | `CA = max(IMCC_{t−1} + SES_{t−1}, m_c·IMCC_avg + SES_avg)` over 60 business days, plus DRC (SA-DRC per D-2 until IMA-DRC is in scope), plus the capital surcharge for amber PLA desks | Golden case over a synthetic 60-day history | Exact |

**The honest limit of P4.** If D-7 is not met and HPL comes from this engine, the PLA
test compares the engine with itself. Report it as `pla_status: "not-independent"`, never
as green.

### Phase 5 — Counterparty credit and CVA (≈4 weeks; IMM needs I-04)

| ID | Task | Detail | Oracle | Exit |
|---|---|---|---|---|
| P5.1 | SA-CCR | Replacement cost, PFE multiplier (5% floor), IR hedging sets by currency, maturity buckets with the profile's correlations, supervisory duration, supervisory delta for swaptions, α = 1.4. Needs netting set and collateral inputs | ORE `SaccrCalculator` (`Examples/CreditRisk/run_saccr.py`) | 1e-10 relative per netting set |
| P5.2 | BA-CVA (reduced) | Counterparty-level SCVA from SA-CCR EAD, supervisory discount factor, discount scalar, ρ | ORE `BaCvaCalculator` | 1e-10 |
| P5.3 | IMM exposure (optional) | EE, Effective EE (non-decreasing), EPE, EEPE over the first year from the existing `npv_cube`. Stressed calibration, α. **Blocked on I-04**: every exposure past first accrual is wrong today | ORE exposure simulation (`Examples/Exposure`) on the same model | EEPE within MC error of ORE; I-04 FIXED first |
| P5.4 | IMM backtesting (optional) | Exposure-model backtesting against realised MtM paths, as CRE53 requires | Statistical tests (§7.5) | Documented test passes over the history available |

### Phase 6 — Precision gate for regulatory figures (≈1–2 weeks)

The project researches whether FP32 can match FP64. For capital, that must be settled
per figure before FP32 is allowed.

| ID | Task | Exit |
|---|---|---|
| P6.1 | Run every regulatory figure at FP64 and FP32 on the Phase 1 portfolio and on a large synthetic one | Table of relative differences |
| P6.2 | Acceptance rule per figure: FP32 is allowed only when its difference is below 1% of the figure's own statistical error (for ES) or below 1e-6 relative (for SA, which is deterministic) | Rule in the profile; `RegulatoryResult` refuses an FP32 run for any figure that fails its gate |
| P6.3 | Record the realised dtype on every regulatory result (composes with I-12, I-14) | Present in the manifest |

### Phase 7 — Proof: evidence pack and independent validation (≈2–3 weeks, then continuous)

Detailed in §8. Tasks:

| ID | Task | Exit |
|---|---|---|
| P7.1 | `python -m engine.regulatory.evidence --portfolio … --out evidence/<date>/` | Produces the pack in §8.2 |
| P7.2 | CI job builds the pack on every change under `engine/regulatory/`, `engine/instruments/`, `engine/models/` | Pack attached to every CI run |
| P7.3 | Mutation testing on `sa/sbm.py`, `ima/es.py`, `backtest/traffic_light.py`, `backtest/pla.py` | Mutation score ≥ 90%; survivors triaged into tests or documented equivalents |
| P7.4 | Independent validation report by someone who did not write the code | Report in `compliance/validation/`; every finding resolved or recorded in the register |
| P7.5 | Close I-27 so the pack rests on a complete suite run | I-27 FIXED, shown by repeated clean runs against a known-bad baseline |

---

## 7. Test strategy

### 7.1 Layers

| Layer | What it proves | Where |
|---|---|---|
| Spec unit tests | Each formula matches the text on small inputs | `tests/regulatory/test_<module>.py` |
| Golden cases | Hand-computed, small portfolios, with a worksheet showing every step | `compliance/golden/` + `test_golden.py` |
| Oracle parity | Same quantity as ORE on the same inputs | `test_ore_<analytic>_parity.py` |
| Independent reference | A second implementation from the text agrees | `tests/regulatory/reference/` |
| Properties | Invariants hold across randomised inputs (§7.4) | `test_properties.py` (hypothesis) |
| Refusals | Out-of-scope inputs are refused, never zero | `test_refusals.py` |
| Boundaries | Every threshold tested one step either side | inside each module's tests |
| End-to-end | A full regulatory run on the reference portfolio reproduces the stored pack | `test_regulatory_e2e.py` |
| Mutation | The tests fail when the code is wrong | P7.3 |

### 7.2 Tolerances

| Comparison | Tolerance | Rationale |
|---|---|---|
| SBM, IMCC, LH cascade, SES, DRC vs independent reference | 1e-12 relative | Same arithmetic in FP64 |
| Traffic light, PLA zones, exception counts, RFET | Exact | Integers and categories |
| Delta vs ORE bump (same one-sided definition) | 1e-8 relative | Same definition; residual from curve arithmetic |
| Vega vs ORE (through recalibration) | 1e-6 relative | ORE recalibrates numerically; the engine uses the implicit function theorem |
| Curvature vs ORE stress | 1e-10 relative | Full revaluation both sides |
| Historical P&L vectors vs ORE | 1e-6 relative | Existing end-to-end parity is 1e-3 per scenario on simulated paths; t=0 revaluation should be tighter. **Measure first, then set.** If it cannot reach 1e-6, record why before loosening |
| SA-CCR, BA-CVA vs ORE | 1e-10 relative | Closed-form both sides |
| EEPE vs ORE | Within 3 MC standard errors | Different random numbers |

A tolerance is never loosened to make a test pass. Loosening needs a register entry
explaining the measured cause, as the project already does for ORE parity.

### 7.3 The independent reference implementation

Written in plain NumPy from the transcribed profile and the Basel text, in
`tests/regulatory/reference/`, by someone who has not read `engine/regulatory/`. Plain
loops over buckets and pairs, no vectorisation, no shared helpers. It exists to disagree
with the engine. If the only author available has written both, record that in the
validation report as a limitation.

### 7.4 Properties to test

- SBM: capital ≥ 0; scaling every position by k scales delta and vega capital by |k|;
  a perfect hedge (equal and opposite positions on the same factor) gives zero delta
  capital; the high-correlation scenario is not below medium for a same-sign book.
- Correlation scenarios: high = min(1.25ρ, 1), low = max(2ρ − 1, 0.75ρ), elementwise.
- ES ≥ VaR at the same level; ES is monotone in the LH cascade; the stressed ratio is
  floored at 1.
- IMCC lies between the fully diversified and undiversified sums.
- The traffic-light zone is monotone in the exception count.
- Re-pillaring does not change NPV; the order of trades does not change any figure
  (catches I-10-class misattribution).

### 7.5 Statistical validation of the ES model (model validation, not a Basel formula)

Run over the P2 history as a rolling out-of-sample backtest and report, without gating
on them: Kupiec proportion of failures and Christoffersen independence at 97.5% and 99%;
an ES backtest (Acerbi–Székely Z2). These go in the validation report as evidence that
the model is fit for purpose, which the supervisory qualitative standards expect.

---

## 8. Proof

### 8.1 Traceability

`compliance/requirements.yaml` holds one entry per Basel requirement the engine claims:

```yaml
- id: SA-GIRR-DELTA-DEF
  cite: "MAR21.xx"                 # filled at P0.3
  text: "GIRR delta sensitivity is the one-sided 1bp difference divided by 0.0001"
  status: implemented               # planned | implemented | refused | not-applicable
  code: [engine/regulatory/sa/sensitivities.py::girr_delta]
  oracle: ore-sensitivity
  tests: [tests/regulatory/test_sensitivities.py::test_girr_delta_matches_ore]
```

Tests reference requirements with `@pytest.mark.basel("SA-GIRR-DELTA-DEF")`.
`test_traceability.py` enforces the catalogue in both directions (P0.4). `refused` and
`not-applicable` are first-class statuses. The catalogue states what is **not** claimed
as plainly as what is.

### 8.2 Evidence pack contents

```
evidence/2026-MM-DD/
  manifest.json          # git SHA, dirty flag (a dirty tree fails the build),
                         # package versions incl. ORE, JAX backend and dtype,
                         # profile name + version + SHA-256, input bundle hashes
  results/               # every regulatory figure with its full breakdown
  traceability.html      # requirement → code → test → last result → oracle diff
  oracle-diffs.csv       # per comparison: max abs, max rel, tolerance, pass
  golden-results.csv
  junit.xml              # full suite run, with the collected count reconciled
                         # against the previous pack
  mutation-report.html
  decisions.md           # copy at the time of the run
  limitations.md         # generated from refused / not-applicable requirements
                         # and open register items that touch a claimed figure
```

The pack is reproducible: rerunning `evidence` on the same SHA and inputs must give
byte-identical `results/` (on CPU FP64, the reference platform). Packs are archived as
CI artifacts, not committed, except one small reference pack under
`compliance/golden/` that `test_regulatory_e2e.py` reproduces.

### 8.3 What counts as proof, and what does not

Counts:
- Agreement with ORE on the same quantity and inputs.
- Agreement with an independent implementation from the text.
- Hand-computed golden cases with a worksheet.
- A red-first record for each test.
- A mutation score showing the tests fail when the code is wrong.
- An independent validation report.

Does not count:
- A green suite on its own (register rule 9).
- Agreement between two code paths inside this engine (for example AD vs bump using the
  same pricer), except as a consistency check.
- PLA computed from engine-produced HPL (P4).
- Any figure computed with a profile value that has not passed double-entry
  transcription.

---

## 9. Sequencing

```
P0 ──► P1 ─────────────────────────► P6 ──► P7 (continuous from P1)
  │                                   ▲
  └──► P2 ──► P3 ──► P4 ─────────────┘
                     ▲
                 D-7 (TraderX HPL/APL)

P5.1, P5.2 (SA-CCR, BA-CVA): after P0, independent of P1–P4
P5.3, P5.4 (IMM):            after I-04
USD swaps in any phase:      after I-05
```

**Start here:** P0.1–P0.4 (decisions, pinned text, profile, traceability) in week 1. Then
P1.1–P1.2 (re-pillaring and literal GIRR delta against ORE), which proves the oracle
pipeline end to end on the smallest real piece. P2 data acquisition should begin in
parallel on day one: it is calendar time, not engineering time.

Rough total: ≈26 engineer-weeks for P0–P7 on one engineer, excluding blocked waits
(I-04, I-05, D-7, vol history).

---

## 10. Risks and limits

| Risk | Consequence | Mitigation |
|---|---|---|
| Transcription error in a risk weight or correlation | Wrong capital, passing tests (tests and code share the error) | Double-entry transcription (P0.3); golden cases computed from the text, not the profile |
| Engine-vs-engine PLA | Meaningless green | D-7; `not-independent` status |
| No swaption vol history | Vol factors become NMRFs; large SES | Declared in D-8; SES reported separately so its size is visible |
| USD swap construction (I-05) stays blocked | No USD swap figures | Refuse; other books proceed |
| ES estimator choice at 250 observations | Capital differs by estimator | D-5; both reported |
| Jurisdictional divergence | BCBS figures differ from a local rule | Profiles; a jurisdiction profile is its own double-entry task |
| Regulation changes | Profile goes stale | Pinned text with retrieval date; profile version in every result; review on each BCBS publication |
| One author writes both implementations | The reference shares the engine's misreading | Recorded as a limitation in the validation report |

**Outside what the engine can prove** (bank responsibilities, listed so nobody assumes
them covered): trading/banking book designation, desk definitions and approval,
independent risk control and model validation functions, senior management and board
oversight, supervisory approval, use-test evidence, the stress-testing programme's
governance, operational controls around market data.

---

## 11. Definition of done

The engine may claim Basel conformance for a figure when all of these hold:

1. Every requirement behind the figure is `implemented` in the catalogue, with passing
   oracle and golden tests.
2. Each of those tests has a red-first record.
3. The profile values it uses passed double-entry transcription against pinned text.
4. Mutation score ≥ 90% on its module.
5. The FP32 gate is recorded (or the figure is FP64-only).
6. The latest evidence pack reproduces it byte for byte.
7. The independent validation report covers it with no open finding.
8. No open register item touches it, or the item is listed in `limitations.md` with its
   effect bounded.

"Fully compliant" for the project means every figure in the D-2 **Must** and **Should**
scope meets this, and everything else in the catalogue is `refused` or
`not-applicable` with a reason.

---

## Appendix A — Seed requirement catalogue

Chapter-level. Paragraph numbers are filled at P0.3 from the pinned text.

| ID | Source | Requirement | Phase |
|---|---|---|---|
| SCOPE-BOOK | MAR12 | Each trade carries book (trading/banking) and desk; the engine refuses a trade without them on the regulatory path | P0 |
| SA-GIRR-VERTICES | MAR21 | GIRR delta risk factors at the prescribed tenors, per currency and curve | P1 |
| SA-GIRR-DELTA-DEF | MAR21 | Delta = one-sided 1bp difference / 0.0001 | P1 |
| SA-GIRR-VEGA-DEF | MAR21 | Vega = ∂V/∂σ × σ at option and underlying maturity vertices | P1 |
| SA-CURV-DEF | MAR21 | Curvature from up/down shifts net of delta; parallel shift for GIRR | P1 |
| SA-WS | MAR21 | Weighted sensitivity = RW × s | P1 |
| SA-KB | MAR21 | Within-bucket aggregation with floor at 0 | P1 |
| SA-ACROSS | MAR21 | Across-bucket aggregation with the alternative S_b | P1 |
| SA-CORR-SCEN | MAR21 | Three correlation scenarios; capital is the maximum | P1 |
| SA-CURV-AGG | MAR21 | Curvature correlations squared; ψ indicator for negative pairs | P1 |
| SA-FX | MAR21 | FX delta for non-reporting-currency positions | P1 |
| SA-CSR-SOV | MAR21 | Sovereign CSR per D-4 | P1 |
| SA-DRC | MAR22 | Jump-to-default, netting, hedge benefit ratio, risk weights | P1 |
| SA-RRAO | MAR23 | Residual risk add-on for flagged instruments | P1 |
| IMA-ES-DEF | MAR33 | ES 97.5%, one-tailed, 10-day base horizon | P3 |
| IMA-LH | MAR33 | Liquidity-horizon cascade | P3 |
| IMA-STRESS | MAR33 | Stressed period since 2007, reduced set ≥75%, ratio floored at 1 | P3 |
| IMA-DATA | MAR33 | Current-period data updated at least monthly | P2 |
| IMA-IMCC | MAR33 | Constrained/unconstrained combination with ρ | P3 |
| IMA-RFET | MAR31 | Risk-factor eligibility test | P3 |
| IMA-SES | MAR33 | NMRF stressed ES and aggregation | P3 |
| IMA-CA | MAR33 | Capital: max of previous day and 60-day average × m_c, plus SES, plus DRC | P4 |
| BT-DESK | MAR32 | Desk backtesting at 97.5% and 99% over 250 days, against HPL and APL | P4 |
| BT-TL | MAR99 | Bank-level traffic light and plus factor | P4 |
| PLA | MAR32 | Spearman and KS between HPL and RTPL; zones; amber surcharge | P4 |
| CCR-SACCR | CRE52 | SA-CCR EAD | P5 |
| CCR-IMM | CRE53 | EEPE, stressed calibration, α, backtesting | P5 |
| CVA-BA | MAR50 | BA-CVA reduced | P5 |
| IMA-DRC | MAR33 | IMA default risk charge | `not-applicable` (D-2) |
| SA-EQ | MAR21 | Equity risk class | `refused` until I-18 |

## Appendix B — Orientation values (not a source)

Recalled for sizing and review only. **Use the transcribed profile, never this table.**

- ES 97.5%; base horizon T = 10 days; liquidity horizons 10/20/40/60/120 days; IMCC
  ρ = 0.5; 60-day averaging; stressed window since 2007; reduced set ≥ 75%.
- Backtesting: 250 days; desk thresholds 12 exceptions at 99% and 30 at 97.5%; bank
  traffic light green 0–4, amber 5–9, red 10+; m_c = 1.5 plus 0 / 0.20 / 0.26 / 0.33 /
  0.38 / 0.42 / 0.50 for ≤4 / 5 / 6 / 7 / 8 / 9 / ≥10 exceptions.
- PLA: green if Spearman > 0.80 and KS < 0.09; red if Spearman < 0.70 or KS > 0.12.
- RFET: ≥ 24 real price observations in 12 months with no 90-day period under 4, or
  ≥ 100 in 12 months.
- GIRR delta tenors 0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30 years; risk weights around
  1.1%–1.7% by tenor, reduced by √2 for specified currencies; tenor correlation
  max(exp(−θ·|Tk − Tl| / min(Tk, Tl)), 40%) with θ = 3%; 99.9% between curves at the same
  tenor; 50% across currencies.
- Correlation scenarios: high = min(1.25ρ, 1); low = max(2ρ − 1, 0.75ρ).
- SA-CCR: α = 1.4; IR supervisory factor 0.5%; supervisory duration
  (e^(−0.05S) − e^(−0.05E)) / 0.05; maturity buckets <1y, 1–5y, >5y with 70% between
  adjacent buckets and 30% between the outer two; PFE multiplier floor 5%.
- BA-CVA: ρ = 50%; discount scalar 0.65.
