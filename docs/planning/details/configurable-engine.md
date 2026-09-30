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

Every choice is one `RunConfig` (`engine/portfolio/config.py`, step 1.2), on
`PortfolioRequest.config`. The model is still chosen by the type of `market`:

| | Market path (default) | Hull-White model |
|---|---|---|
| Entry | `price_portfolio(PortfolioRequest(market=Market(...), config=RunConfig(...)))`, `engine.portfolio.market_path` | `price_portfolio(PortfolioRequest(market=SimulationConfig(...), config=HULL_WHITE_CONFIG))`, `engine.portfolio.request` |
| HTTP | `POST /v2/portfolio/price`, `schema_version: "2"` | `POST /portfolio/price` (translated with `HULL_WHITE_CONFIG`) |
| Simulation | `config.simulation`, `engine.simulation.cam` (LGM per currency, exact) | the `market` itself, `engine.simulation.market_model` (Hull-White, constant θ) |
| Valuation | `engine.valuation` (every trade by its t=0 engine per path) | `engine.instruments.*` scenario pricers |
| European engine | `Bachelier` | `Jamshidian` |
| Greeks | `Bump`, `engine.risk.sensitivities`, settings `config.greeks.sensitivity` | `AD`, `engine.risk.greeks` |
| Precision | `simulation` only; other stages refused below 64 (I-55, step 1.4) | every stage, process-global x64 toggle |

`engine.market_risk` still picks the European engine from the trade's fields (A-8, F-01).

## Step 1.2 — the run configuration (I-68) — done

| Component | Field | Options | Default |
|---|---|---|---|
| Model per currency | `simulation.ir[ccy]` (`CamConfig`, ORE's `CrossAssetModelData`) | `LgmConfig`; Hull-White in step 1.3 | LGM |
| Simulation | `simulation` | Classic revaluation; AMC is [F-03](../features.md#f-03) | Classic |
| Engine per product | `pricing` (`PricingConfig`) | Swap: discounting. European: `Bachelier`, `Jamshidian`. Bermudan/American: `LgmSwaptionEngineConfig` (FD solver in F-01) | ORE's builder defaults |
| Greeks method | `greeks.method` | `Bump`; `AD` | `Bump` |
| Sensitivity settings | `greeks.sensitivity` (`SensitivityConfig`) | Tenors, shifts, Theta horizon, vol decay | ORE's |
| Precision per stage | `precision` (`PrecisionConfig`) | 32 or 64 per stage, overrides per type and metric | FP64 |
| Swaption vol decay | `simulation.swaption_vol_decay` | `ForwardVariance`; `ConstantVariance` (A-4) | `ForwardVariance` |
| Reporting currency | `base_currency` | Any market currency; `None` is the simulation's, else USD | `None` |

Rules the implementation follows, which steps 1.3, 1.4 and 4.1 keep:

- **An option a model does not implement is refused, never substituted.** `check_market_path`
  and `check_hull_white` run before any work and name the field. An engine or method is
  checked where the run uses it (a European engine only with a European, a Greeks method
  only with `compute_greeks`); a setting a model does not read at all is refused when
  changed from its default.
- **One fact, one field.** The reporting currency is the simulation's; a `base_currency`
  contradicting it is refused (before 1.2 it was silently ignored).
- **The request travels whole.** The worker pool freezes the entire request, so every
  configuration component reaches the worker and is validated again there.

Evidence that the defaults reproduce the market path bit for bit: the shared portfolio
(8 trades, scenario risk, exposure, bump Greeks), an FP32-simulation run and the Hull-White
model, compared array for array against the code before the change (114 arrays, all
identical: [verification status](../known-issues.md#verification-status)), and the parity
suites in the full run.

## Step 1.3 — the Hull-White model on the shared pipeline

Rebuild the Hull-White model as an option of the configuration, reusing the market path's
valuation layer instead of `engine.instruments`' scenario pricers. Each item closes an issue
and is tested as on the market path (sloped curves, red first):

| Change | Closes |
|---|---|
| Simulate the zero-mean OU state and add the curve-fitted drift, `r(t) = x(t) + α(t)`, `α(t) = f(0,t) + σ²/(2a²)(1 − e^{−at})²` (Brigo–Mercurio 3.36); `theta` and `initial_rates` become derived, not inputs. Permanent martingale test `E[P(t,T)/N(t)] = P(0,T)` on a sloped curve | [I-42](../known-issues.md#i-42) |
| Scenario curves from the model's own bond prices; no `x_from_r` conversion to LGM | [I-44](../known-issues.md#i-44), [I-62](../known-issues.md#i-62) |
| The model's exact numeraire in the reporting currency | [I-45](../known-issues.md#i-45) |
| `engine.valuation.legs` for paid flows and path fixings | [I-04](../known-issues.md#i-04) (model half) |
| `engine.valuation.options` (`OptionWrapper`) for exercise | [I-43](../known-issues.md#i-43) |
| Market-vol Bachelier as the default European engine; Jamshidian as an option | [I-46](../known-issues.md#i-46) |
| Per-trade basket and bootstrap (`engine.valuation.bermudan.calibration_basket`, `engine.calibration.ore_lgm`); the shared-basket policy leaves `engine/api/schemas.py` and `_fill_calibrated_sigma` | [I-47](../known-issues.md#i-47) |
| Bond legs on the model's scenario curves | [I-24](../known-issues.md#i-24) |
| Trades name curves and index only; model parameters and calibrated σ from the market and configuration | [I-63](../known-issues.md#i-63) |
| `evaluation_date` required on every trade config; any ORE global set inside a restoring context manager | [I-64](../known-issues.md#i-64) |
| An instrument id on every trade config | [I-10](../known-issues.md#i-10) (configs) |

The Hull-White warnings in `engine.portfolio.request` (aged swaps, expiry, curve
consistency) are removed with the defects they describe.

## Step 1.4 — the precision mechanism (I-55)

x64 is enabled once per process; every stage takes its dtype from the configuration and
every array is created with one (find each array created without a dtype;
`compute_hw_A_matrix` hard-codes float64). Then remove the `jax_enable_x64` toggling in
`generate_paths` and `price_portfolio`, `_PRICING_LOCK`, and the per-precision worker pools.
Adjustable precision stays available throughout the change.

## Step 2.5 — `ShiftHorizon` (I-32)

`H → H + shift`, with the state grid built in the shifted variable (`engine.models.lgm`,
`_state_grid`). Add `shift_horizon=0.5` cases to `tests/test_ore_lgm_parity.py`, then make 0.5
the default (ORE's builder default, `OREData/ored/portfolio/builders/swaption.cpp`).

## Step 2.7 — precision evidence (I-55)

A table per figure (NPV, exposure profile, VaR/ES, each Greek, calibration) and precision
combination: what was validated, how, at how many paths. A result whose combination is not
validated for a figure it reports carries a warning naming the evidence and what is
missing. Any combination may still be run (D-9). The same table is Basel P6's precision gate
and F-07's validation bar.

## Step 4.1 — one request (I-56)

- One route and one request whose configuration reaches every setting of step 1.2,
  including the sensitivity settings, market-risk runs (`engine.market_risk.run_market_risk`)
  and the market path's calibrations as standalone runs.
- Validated before any job starts: types, unknown fields refused, cross-field checks, each
  refusal naming its field.
- Names say what they are: `/v2` and `schema_version: "2"` go; a version marks a revision of
  the contract, never a model.
- `POST /portfolio/price` and `POST /v2/portfolio/price` keep answering, translated into the
  new request, so no caller breaks.
- A completeness test compares the Python configuration types with the request schema and
  fails on any setting without an API field.
- Results echo instrument ids and return the cube as a chunked artifact reference (shape,
  dtype, axis order, hash, item order) instead of nested JSON (I-09).
