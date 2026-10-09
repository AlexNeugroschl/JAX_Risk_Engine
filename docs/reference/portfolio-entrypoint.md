# The Portfolio Entry Point

**Modules:** [`engine/portfolio/request.py`](../../engine/portfolio/request.py), the pipeline in
[`engine/portfolio/market_path.py`](../../engine/portfolio/market_path.py), and the run
configuration in [`engine/portfolio/config.py`](../../engine/portfolio/config.py)
**Public entry point:** `price_portfolio(request: PortfolioRequest) -> PortfolioResult`

## Plain-language summary

`engine.portfolio` answers the question "what should a caller of the whole system hand
over, and what do they get back?" with two dataclasses — `PortfolioRequest` in,
`PortfolioResult` out — and one function, `price_portfolio`, that does everything in
between: validate, calibrate and simulate the cross-asset model, price every trade today and
on every path, and summarise exposure and Greeks. The model, engines, Greeks method and
precision it uses are the request's `RunConfig`; the curves and volatilities are the
request's `Market`; the trades name their currency and index and carry no model.

**This is the exposure path.** Its simulation runs months to years forward under the
risk-neutral measure, so its risk output is an exposure profile (EPE, ENE, PFE through
time — [Exposure](../risk/exposure.md)), not a VaR. Short-horizon market-risk VaR/ES is
`engine.market_risk.run_market_risk` ([Market Risk](../risk/market-risk.md)).

## `PortfolioRequest`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `market` | `Market` | *required* | Today's market: curves per currency and index, swaption volatilities, FX and equity spots (`engine/market.py`). Anything else is a `TypeError` (the Hull-White model is a `HullWhiteConfig` in `config.simulation.ir`, not a market type). |
| `trades` | `List[TradeConfig]` | *required* | Any mix of `SwapConfig`, `SwaptionConfig`, `BermudanSwaptionConfig`, `AmericanSwaptionConfig`, `BondConfig`. Each has a `trade_id`, unique in the portfolio (a repeat is refused), and is valued on `market.asof` (its `evaluation_date`). |
| `config` | `RunConfig` | ORE's defaults | See below. |
| `pfe_quantiles` | `Sequence[float]` | `(0.95, 0.99)` | The exposure profiles' PFE quantiles. |
| `compute_greeks` | `bool` | `False` | Also compute every trade's Greeks, by `config.greeks.method`. |
| `scenario_risk` | `bool` | `True` | Simulate and build `npv_cube` and the exposure profiles (needs `config.simulation`). `False` prices today's values (and Greeks) only. |

## `RunConfig`

`engine/portfolio/config.py`. One value for every choice a run makes, as ORE's run is
configured by its files; `RunConfig()` is ORE's defaults.

| Field | Type | Default | ORE | Meaning |
|---|---|---|---|---|
| `simulation` | `Optional[CamConfig]` | `None` | `simulation.xml` | The date grid, the model per currency (`ir[ccy]`: `LgmConfig`, or `HullWhiteConfig`), FX/equity volatilities, correlations, simulation-market tenors, samples, seed, swaption vol decay. Required for scenario risk. |
| `pricing` | `PricingConfig` | ORE's builders | `pricingengine.xml` | The engine per product: `european` (`"Bachelier"`, ORE's default, or `"Jamshidian"` with its Hull-White model in `jamshidian`), `bermudan` and `american` (`LgmSwaptionEngineConfig`), `recalibrate` (per path, as ORE). |
| `greeks` | `GreeksConfig` | `Bump`, ORE's settings | `sensitivity.xml` | `method` (`"Bump"`, ORE's, or `"AD"`) and `sensitivity` (`SensitivityConfig`: curve tenors, shift sizes, Theta horizon, vol decay on the Theta date). |
| `precision` | `Precision` | float64 everywhere | none | Storage, compute and accumulate format per adjustable stage; see "`Precision`" below. |
| `base_currency` | `Optional[str]` | `None` | `baseCurrency` | The reporting currency. `None` means the simulation's base currency, or USD without a simulation; a value contradicting the simulation's is refused. |

Every option runs with every other: the models differ only in the simulation, and the
engines and Greeks methods price whatever it produced. What the pipeline does not implement
yet is refused **before any work**, with a `ValueError` naming the field: today a precision
format before the roadmap step that enables it (`engine.precision`). An engine refuses a trade it
cannot price, naming the trade (`validate_trades`; e.g. the Jamshidian engine refuses a
floating spread and cash settlement, as QuantLib's does). Nothing is priced with another
engine than the one configured.

```python
from engine.portfolio import (
    CamConfig, GreeksConfig, HullWhiteConfig, JamshidianEngineConfig, PortfolioRequest, PricingConfig, RunConfig,
    SensitivityConfig, price_portfolio,
)

config = RunConfig(
    simulation=CamConfig(dates=dates, base_currency="USD", ir={"USD": HullWhiteConfig(reversion=0.03, volatility=0.01)}),
    pricing=PricingConfig(european="Jamshidian", jamshidian=JamshidianEngineConfig(0.03, 0.01)),
    greeks=GreeksConfig(method="AD", sensitivity=SensitivityConfig(curve_tenors=("1Y", "2Y", "5Y", "10Y"))),
)
result = price_portfolio(PortfolioRequest(market=market, trades=trades, config=config, compute_greeks=True))
```

### Bonds

`BondConfig` (Treasury bills and notes) prices like every other trade: today and on every
path, by discounting its remaining flows on its currency's curve (ORE's
`DiscountingRiskyBondEngine` without credit), with exposure, Greeks (discount curve only, no
Vega) and scenario risk. (Until 2026-10-01 the Hull-White model refused a bond with
scenario risk, [I-24](../planning/known-issues.md#i-24).)

## `Precision`

On `RunConfig.precision` (`engine.precision`, exported by `engine.portfolio` too; design:
[details/precision.md](../planning/details/precision.md)). A `StagePrecision(storage, compute,
accumulate)` of format names per adjustable stage, each `"float64"` by default:

| Field | Governs |
|---|---|
| `simulation` | The Sobol shocks and the model states (`simulate`). |
| `market` | The scenario market built from the states: curves, numeraire, FX and equity spots. |
| `pricing` | Every trade on every path (Bermudan/American per-path recalibration included) and the cube it stores. |
| `by_product` | `{product: StagePrecision}`: replaces `pricing` for every trade of a product. The products are `"swap"`, `"european_swaption"`, `"bermudan_swaption"`, `"american_swaption"` and `"bond"` (each trade config's `product`). |
| `by_trade` | `{trade_id: StagePrecision}`: replaces `by_product` and `pricing` for one trade. |
| `rounding` | `"nearest"` (default) or `"stochastic"`: how values are rounded into a storage format below 32 bits. Stochastic rounding without such a format is refused. |
| `rounding_seed` | Non-negative integer, 0 by default: the seed of the stochastic rounding. |
| `paired_fraction` | In [0, 1], 0 by default: the share of paths also simulated and priced at float64 throughout (rounded up to whole blocks of 32 paths), to measure the run against float64 (2026-10-02). |

`compute` is the format a stage computes in, `storage` the format its output is kept in until
the next stage reads it (no wider than `compute`), `accumulate` the format its sums
accumulate in (equal to `compute` until F-07's compute formats). `compute` is `"float64"` or
`"float32"`; `storage` may also be `"float16"`, `"bfloat16"`, `"float8_e4m3fn"` or
`"float8_e5m2"` (2026-10-02), while compute in them is refused, naming F-07.
`Precision.throughout("float32")` sets every stage to float32.

Storage below 32 bits keeps a float32 power-of-two scale per block of 32 paths that brings the
block's largest value to the format's maximum, so the format's range never limits the values;
its precision does (FP8 e4m3 keeps 4 significant bits). Each value is rounded to the format's
nearest value, or with `rounding="stochastic"` up or down in proportion to the distance, so
rounding errors average out over the paths instead of biasing a mean; the draws come from
`rounding_seed` and the array's name (the trade id for a cube column), so runs reproduce. The
scenario market's tenor grid is not stored narrower than `market.compute`. Curves stored below
32 bits lose forward rates to cancellation ([I-75](../planning/known-issues.md#i-75)).

A trade is priced at `Precision.precision_for(trade)`: its `by_trade` entry, else its
product's `by_product` entry, else `pricing` (decision A-15). Each trade's cube column is
computed at that `compute` and stored at that `storage`, exactly as if the trade were priced
alone with that `pricing`. An override that names no trade of the request, or no product, is
refused before any work:

```python
f32 = StagePrecision("float32", "float32", "float32")
Precision(simulation=f32, market=f32, pricing=f32,
          by_product={"bermudan_swaption": StagePrecision()},   # Bermudans in float64
          by_trade={"swap-7": StagePrecision(storage="float32")})  # one swap: float64, stored float32
```

Calibration, today's values, Greeks and every reduction over paths (the exposure profiles)
are float64 whatever the policy says (decision A-10): `base_npv_per_trade` is float64, and
`npv_cube` is the stored cube read back at float64, so its values are float32 numbers when
its `storage` is `"float32"`, and FP8 numbers times their block scales when it is FP8. The
32/64 shape before 2026-10-01 (`PrecisionConfig` and its override classes) is refused,
naming the replacement (decision A-12).

**The paired sample and the precision report** (decision A-13). With
`paired_fraction > 0` the first paths are simulated and priced again at float64 throughout (the
same Sobol points, so the same paths a float64 run would give, bit for bit), and every figure is
measured against float64 on them. Means are corrected by the two-level estimator,
`mean over the run's paths + mean over the paired paths of (float64 − run)`, which turns a
precision bias into variance: `exposure.epe`/`ene` and everything computed from EPE (EE_B,
EEE_B, EPE_B, EEPE_B, the Basel figures) are the corrected values. PFE is a quantile: it stays
the run's own, measured on the pair. Every result carries `precision`, a `PrecisionReport`:

| Field | Meaning |
|---|---|
| `policy` | The `Precision` as run. |
| `trades` | `{trade_id: StagePrecision}`: each trade's resolved pricing stage. |
| `realized` | `{array: format}`, read from the stored arrays: `"shocks"`, `"states"`, `"market"` and `"values/<trade_id>"` (a market-risk run: `"shocks"` and the P&L per trade). |
| `devices`, `backend`, `jax_version` | The devices holding the stored arrays (`"cpu:0 (cpu)"`), the JAX backend and version of the process that ran the job. |
| `paths`, `paired_paths` | The run's paths (scenarios), and how many were re-run at float64. |
| `figures` | With a paired sample: `"netting_set/EPE"`, `"trades/<trade_id>/ENE"`, `"netting_set/PFE_95"`, ... (market risk: `"portfolio/VaR_99"`, `"portfolio/ES_97.5"`). A mean (`MeanEstimate`): `value` (the reported, corrected figure), `uncorrected`, `correction`, `standard_error`, `uncorrected_standard_error`, `correction_standard_error`, `max_difference`. A quantile (`QuantileEstimate`): `value` (the run's own), `paired`, `paired_float64`, `difference`. Arrays per simulation date (t=0 excluded). |

```python
result = price_portfolio(PortfolioRequest(market, trades, RunConfig(
    simulation=sim, precision=Precision(pricing=StagePrecision("float8_e4m3fn"), paired_fraction=0.05))))
epe = result.precision.figures["netting_set/EPE"]
epe.correction, epe.correction_standard_error   # the FP8 cube's bias on EPE, and its error
```

At float64 everywhere a paired sample measures exactly zero. Whether a combination is
validated for a figure is the evidence table of [I-55](../planning/known-issues.md#i-55).

## `PortfolioResult`

| Field | Type | Meaning |
|---|---|---|
| `base_npv` | `float` | The whole portfolio's value today, in the reporting currency (the sum of `base_npv_per_trade`). |
| `npv_cube` | `jax.Array` | `[Scenarios, Dates, Trades]`, one column per trade in `request.trades`' order, in the reporting currency. Zero-width without scenario risk. |
| `exposure` | `Optional[ExposureProfile]` | The whole portfolio as one netting set: `times`, `epe`, `ene`, `ee_b`, `eee_b`, `epe_b`, `eepe_b`, `basel_epe`, `basel_eepe` and `pfe` (`"PFE_95"` → profile), with one entry per date including t=0. `None` without scenario risk. See [Exposure](../risk/exposure.md). |
| `trade_exposures` | `List[ExposureProfile]` | Each trade's standalone exposure, in `request.trades` order. Empty without scenario risk. |
| `greeks` | `Optional[Dict[int, Dict[str, jax.Array]]]` | `None` unless `request.compute_greeks=True`. Keyed by each trade's index in `request.trades`; see [Greeks](../risk/greeks.md) for the keys. |
| `warnings` | `List[str]` | Run warnings (none are emitted today). |
| `base_npv_per_trade` | `List[float]` | Each trade's value today, in `request.trades` order. |
| `trade_ids` | `List[str]` | Each trade's `trade_id`, in `request.trades` order: the names of every per-trade row ([I-10](../planning/known-issues.md#i-10)). |
| `scenario_risk_available` | `bool` | `False` when the run was `scenario_risk=False`, meaning `exposure` is **absent** and `npv_cube` zero-width. |
| `measure` | `Optional[str]` | The exposure's measure: `"risk-neutral-pricing"` (`engine.risk.var_es.ENGINE_RISK_MEASURE`) whenever it was computed, `None` without scenario risk. An exposure under the pricing measure, **not** a forecast of tomorrow's loss ([I-11](../planning/known-issues.md#i-11)). |
| `precision` | `PrecisionReport` | The precision as run, read from the run's arrays, and with a paired sample each figure's estimate ([above](#precision)). |

## `price_portfolio(request: PortfolioRequest) -> PortfolioResult`

`engine.portfolio.market_path.price_on_market` (safe to call from several threads at once):

1. **Validate before any JAX work** (`validate_request`): the configuration
   (validated when it is built), scenario risk needs a simulation, every trade valued on the market's date
   with every curve and volatility it reads present and its engine's refusals, precision
   overrides that name a trade or product (`validate_trades`, naming the trade or override),
   the reporting currency in the market. The HTTP
   route runs the same check synchronously, so such a request is a 400, not a failed job.
2. **Calibrate and simulate** (with scenario risk): `build_cross_asset_model` calibrates
   each currency with a basket to the market's swaption volatilities, and `simulate` builds
   the scenario market at `precision.simulation` and `precision.market`.
3. **Value** every trade today and on every path with its configured engine
   (`engine.valuation.portfolio.value_portfolio`, each at `precision.precision_for(trade)`), converting
   foreign trades at the spot today and at the path FX on paths; without scenario risk,
   today only (`value_today`).
4. **Exposure**: the netting set and each trade, deflated by the LGM numeraire, from the
   cube and numeraire loaded at float64; with `precision.paired_fraction`, first the paired
   float64 sample (`paired_sample`), and the exposure means corrected by it.
5. **Greeks** (with `compute_greeks`): `portfolio_sensitivities` (Bump) or
   `portfolio_greeks` (AD).

The result carries the trades' ids. Each stage is labelled for the profiler (`calibration`,
`simulation`, `pricing`, `paired_sample`, `exposure`, or `base_npv` without scenario risk,
then `greeks`;
[Profiling](../concepts/profiling.md)).

### Automatic calibration

There is no request-level calibration: each currency's model is calibrated to its own basket
when `CamConfig.ir[ccy]` names one (`calibration_expiries` × `calibration_terms`), and each
Bermudan/American's engine to the trade's own co-terminal basket (ORE's `LgmBuilder`, per
trade, recalibrated on every path). Until 2026-10-01 the Hull-White model took one shared
basket for every uncalibrated trade (`calibration_targets`,
[I-47](../planning/known-issues.md#i-47)).

### Known-limitation flagging

(Section title kept for links.) The Hull-White model's cube warnings (aged swaps, options
expiring in the horizon, a curve inconsistent with its short rate) went with that pipeline on
2026-10-01: what they warned about is priced correctly now. A run's known limitations are in
[known issues](../planning/known-issues.md).

## Greeks

`compute_greeks=True` gives every trade's Greeks by `config.greeks.method`: ORE's
bump-and-revalue sensitivities (`Bump`, the default) or automatic differentiation (`AD`),
with the same keys — Delta and Gamma per curve tenor (Bump) or pillar (AD) of each curve the
trade reads, Vega per swaption quote for a trade whose engine reads them, and ORE's Theta.
See [Greeks](../risk/greeks.md).

## Tested by

- `tests/test_portfolio_market_path.py` — the result is the valuation layer's, exposure
  identities (a bill's EE is its forward value), ORE's time-weighted and Basel profiles,
  trade ids.
- `tests/test_portfolio_entrypoint.py` — `price_portfolio` equals the pipeline orchestrated
  by hand for every trade type under the Hull-White model; reordering, calibration,
  precision, concurrency, one evaluation date.
- `tests/test_run_config.py` — the defaults, every model with every engine and Greeks method,
  refusals naming the field, the sensitivity settings reaching the Greeks.
- `tests/test_engine_worker.py` — the HTTP path's engine worker: the request it prices is the
  one the route validated; jobs queued together equal jobs run alone and the direct call.
- `tests/test_portfolio_gap_fixes.py` — regressions for I-01/I-02/I-03/I-13.
- `tests/test_portfolio_bond_wire_through.py` — bonds through `price_portfolio`: today,
  Greeks (I-26, I-70), every path (I-24).
- `tests/test_portfolio_scale_and_edge_cases.py`, `tests/test_diverse_portfolio_e2e.py` —
  scale, currencies, degenerate inputs, breadth against ORE.
- `tests/test_hull_white_model.py` — the Hull-White model's regressions.
