# Configurable engine

Design for roadmap [stage 1](../roadmap.md#stage-1--structure) and step
[4.1](../roadmap.md#stage-4--api-robustness): one run configuration whose options are
models, engines, methods and precision, as ORE configures a run. Implements owner decisions
A-1 to A-9 ([compliance/decisions.md](../../../compliance/decisions.md) §1–2).

## Principles

1. **ORE is the specification.** New code cites the ORE/QuantLib source it reproduces (file
   and function) in its docstring; "ORE" means the vendored `reference/ORE/`.
2. **Defaults are ORE's defaults.** Where ORE has no code default and its configuration must
   name a value, the default is what ORE's `Examples/` use, and the code says so.
3. **Options are added, not swapped.** Nothing that works is removed without a replacement;
   an option leaves the default when something better arrives, but stays available.
4. **One request shape for every configuration.** No API versions per model.
5. **Red first, on sloped curves.** Every fix is shown failing against the pre-fix code, on a
   curve rising 3% → 5% as well as flat; flat curves make drift and convexity terms cancel.

## Today

Every choice is one `RunConfig` (`engine/portfolio/config.py`) on
`PortfolioRequest.config`, and every run is one pipeline (`engine.portfolio.market_path`):

| | |
|---|---|
| Entry | `price_portfolio(PortfolioRequest(market=Market(...), trades=[...], config=RunConfig(...)))` |
| HTTP | `POST /portfolio/price` (also served as `/v2/portfolio/price`), one request shape |
| Simulation | `config.simulation` (`CamConfig`), `engine.simulation.cam`: per currency `LgmConfig` or `HullWhiteConfig`, exact step moments, LGM numeraire |
| Valuation | `engine.valuation`: every trade by its t=0 engine on every path (scenario market, legs, `OptionWrapper`, bond legs, per-trade basket) |
| European engine | `Bachelier` (ORE's default) or `Jamshidian` with `PricingConfig.jamshidian` |
| Greeks | `Bump` (`engine.risk.sensitivities`, settings `config.greeks.sensitivity`) or `AD` (`engine.risk.greeks`) |
| Market risk | `engine.market_risk.run_market_risk` on a `Market` with the same `PricingConfig` (A-8) |
| Precision | `config.precision` (`engine.precision.Precision`, step 1.4): storage, compute and accumulate per adjustable stage (simulation, market, pricing), float64 or float32 compute, storage down to FP8 with block scales and nearest or stochastic rounding since step 1.6; per product and per trade since step 1.5; the paired float64 sample and the precision report on every result since step 1.7 ([precision.md](precision.md)) |

## Step 1.2 — the run configuration (I-68) — done

| Component | Field | Options | Default |
|---|---|---|---|
| Model per currency | `simulation.ir[ccy]` (`CamConfig`, ORE's `CrossAssetModelData`) | `LgmConfig`; `HullWhiteConfig` (step 1.3) | none: named per currency |
| Simulation | `simulation` | Classic revaluation; AMC is [F-03](../features.md#f-03) | Classic |
| Engine per product | `pricing` (`PricingConfig`) | Swap: discounting. European: `Bachelier`, `Jamshidian`. Bermudan/American: `LgmSwaptionEngineConfig` (FD solver in F-01) | ORE's builder defaults |
| Greeks method | `greeks.method` | `Bump`; `AD` | `Bump` |
| Sensitivity settings | `greeks.sensitivity` (`SensitivityConfig`) | Tenors, shifts, Theta horizon, vol decay | ORE's |
| Precision per stage | `precision` (`Precision`, step 1.4) | Storage, compute and accumulate format per adjustable stage | FP64 |
| Swaption vol decay | `simulation.swaption_vol_decay` | `ForwardVariance`; `ConstantVariance` (A-4) | `ForwardVariance` |
| Reporting currency | `base_currency` | Any market currency; `None` is the simulation's, else USD | `None` |

Rules the implementation follows, which steps 1.3, 1.4 and 4.1 keep:

- **An option the pipeline does not implement is refused, never substituted.** The
  configuration's own validation (a precision format before its step, `engine.precision`)
  and `validate_trades` run before any work and name the field (since step 1.3 one check for
  both models). An engine is checked where the run uses it (the Jamshidian engine's
  refusals only for a European on it).
- **One fact, one field.** The reporting currency is the simulation's; a `base_currency`
  contradicting it is refused (before 1.2 it was silently ignored).
- **The request travels whole.** The engine worker parses the HTTP body exactly as the
  route did, so every configuration component reaches it and is validated again there
  (until roadmap 1.8 a worker pool froze the request into a picklable form).

Evidence that the defaults reproduce the market path bit for bit: the shared portfolio
(8 trades, scenario risk, exposure, bump Greeks), an FP32-simulation run and the Hull-White
model, compared array for array against the code before the change (114 arrays, all
identical: [verification status](../known-issues.md#verification-status)), and the parity
suites in the full run.

## Step 1.3 — the Hull-White model on the shared pipeline — done

The Hull-White model is a model per currency (`HullWhiteConfig` in `CamConfig.ir`, `"model":
"HullWhite"` over HTTP) on the same pipeline as the LGM; the separate Hull-White pipeline
(`SimulationConfig` market, `engine.simulation.market_model`, the `engine.instruments`
scenario pricers, `HULL_WHITE_CONFIG`) is removed. Trades carry no model, name themselves and
their date. What closed which issue:

| Change | Closed |
|---|---|
| The model is ORE's `<LGM>` with `ReversionType`/`VolatilityType` `HullWhite` (`IrLgm1fPiecewiseConstantHullWhiteAdaptor`): α(t) = σ(t)e^{at}, ζ(t) = ∫σ²e^{2as}ds, H(t) = (1 − e^{−at})/a, simulated exactly under the LGM measure by the CAM; its curves are its own bond prices, fitted to today's curve by construction | [I-42](../known-issues.md#i-42), [I-44](../known-issues.md#i-44) |
| The model's exact LGM numeraire | [I-45](../known-issues.md#i-45) |
| Every trade valued by `engine.valuation` on the paths: legs (paid flows drop out, path fixings), `OptionWrapper`, bond legs, per-trade basket recalibrated per path, vectorized on device | [I-04](../known-issues.md#i-04) (model half), [I-43](../known-issues.md#i-43), [I-24](../known-issues.md#i-24), [I-47](../known-issues.md#i-47), [I-62](../known-issues.md#i-62) |
| Bachelier on the market volatility is the default European engine for every model; Jamshidian is an option with its own Hull-White model | [I-46](../known-issues.md#i-46) |
| Trades name currency and index only; a model or curve field on a trade is refused | [I-63](../known-issues.md#i-63) |
| `evaluation_date` and `trade_id` required keyword fields on every trade config | [I-64](../known-issues.md#i-64), [I-10](../known-issues.md#i-10) (configs) |
| The model is a field of the configuration, not the market's type | [I-68](../known-issues.md#i-68) |

Design decisions taken in the step, with their reasons:

- **LGM-measure form, not the bank-account drift.** The plan named Brigo–Mercurio's
  r(t) = x(t) + α(t) with the curve-fitted drift. ORE does not simulate that: its Hull-White
  model is the LGM adaptor above, under the LGM measure with the LGM numeraire. The two are the
  same model (the short rate is r = f(0,t) + H′(t)z + ζ(t)H(t)H′(t); the path curve is
  QuantLib's `HullWhite::discountBond(t, T, r)` to 1e-12, `tests/test_cam.py`), so the CAM's
  exact step moments, numeraire and scenario market serve both models unchanged, and ORE
  parity holds by construction.
- **Calibration converts the Hagan bootstrap bucket by bucket.** A helper's price depends on
  the model only through ζ at its expiry and H, so the Hull-White σ is the one with the
  LGM's ζ at every bucket end (`engine.models.lgm.hull_white_matching_zeta`); exact, and the
  same calibration ORE's bootstrap of the adaptor reaches (`tests/test_hull_white_model.py::
  TestCalibration`).
- **Jamshidian's model in `PricingConfig.jamshidian`.** ORE has no Jamshidian builder, so the
  engine's Hull-White (a, σ) is configured with it (`JamshidianEngineConfig`, both positive,
  as QuantLib's `HullWhite` requires); required with `european="Jamshidian"`, refused
  without. The engine is QuantLib's decomposition in discount factors only, so it prices on
  any curve: today's, a bump, a path's.
- **`trade_id`, not "instrument id".** Named as ORE's `<Trade id>`; unique in a portfolio,
  echoed as `PortfolioResult.trade_ids`. Over HTTP, give it on every trade or on none (none
  numbers them `trade-0`, ...).
- **ORE globals.** The engine sets no ORE global (every date is passed explicitly; the
  calibration helpers take their dates from their own curve's reference date), so there is
  nothing to scope.
- **Precision narrowed until 1.4.** The Hull-White pipeline took `pricing`/`risk` below 64;
  the shared pipeline computed those stages in float64 and refused less (I-55). Kept rather
  than ported: 1.4 made them adjustable for both models at once.

Evidence: the shared portfolio's market-path numbers (t=0, an FP64 and an FP32-simulation
scenario run, bump Greeks) bit for bit before and after, 103 of 103 arrays; each closed
Hull-White defect measured on the code before the step and asserted after, on a 3% → 5%
curve; every per-path ORE comparison of `tests/test_valuation.py` run under both models;
`tests/test_end_to_end.py` prices the Hull-White simulation's paths in QuantLib
([verification status](../known-issues.md#verification-status)).

## Steps 1.4 to 1.8 — precision and the engine worker (I-55, I-12, I-72); done 2026-10-01 to 10-04

Designed in [precision.md](precision.md): a `Precision` with storage, compute and accumulate
per adjustable stage, overridable per product and trade (A-10, A-15); the old configuration
refused (A-12); five cast points with inputs following dtype; storage down to FP8; the paired
float64 sample, the two-level estimator and the precision report (A-13); then one engine
worker process per host behind a durable job queue (A-14). Every step keeps the default bit
for bit. Adjustable precision stays available throughout.

## Step 2.5 — `ShiftHorizon` (I-32)

`H → H + shift`, with the state grid built in the shifted variable (`engine.models.lgm`,
`_state_grid`). Add `shift_horizon=0.5` cases to `tests/test_ore_lgm_parity.py`, then make 0.5
the default (ORE's builder default, `OREData/ored/portfolio/builders/swaption.cpp`).

## Steps 2.7 and 2.8 — precision evidence and low-precision kernels (I-55, F-07)

In [precision.md](precision.md) §8 and §10: the evidence table per figure and precision
against the acceptance standard (A-11), shared with Basel P6; then the kernels in difference
form, one implementation for every precision (A-16).

## Step 4.1 — one request (I-56)

- One route. Since 1.3 one request shape (`MarketPortfolioRequestSchema`) reaches the model
  per currency, the engines, the Greeks method and sensitivity settings, precision and the
  reporting currency; still missing: market-risk runs (`engine.market_risk.run_market_risk`),
  the CAM calibration as a standalone run, and `shift_horizon` (I-32).
- Validated before any job starts: types, unknown fields refused, cross-field checks, each
  refusal naming its field.
- Names say what they are: `/v2` and `schema_version: "2"` go; a version marks a revision of
  the contract, never a model.
- `POST /portfolio/price` and `POST /v2/portfolio/price` keep answering. (The Hull-White
  request shape retired by 1.3 is refused with a 422 naming its replacement, not translated:
  its trades carried model copies the new request has no place for.)
- A completeness test compares the Python configuration types with the request schema and
  fails on any setting without an API field.
- Every per-trade result row carries its `trade_id` (today a list beside position-keyed rows,
  I-10), and the cube returns as a chunked artifact reference (shape, dtype, axis order, hash,
  item order) instead of nested JSON (I-09).
