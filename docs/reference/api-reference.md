# API Reference

Exact inputs and outputs for every public function and configuration dataclass. For the
*why* behind these shapes, see the per-stage deep dives
([Market Simulation](../concepts/market-simulation.md), [Instruments](../instruments/swaps.md),
[Greeks](../risk/greeks.md), [Risk Statistics](../risk/var_es.md)). For runnable examples,
see the [User Guide](../getting-started/user-guide.md).

**Notation:** `[Scenarios, Dates, ...]` describes an array's shape. `Scenarios` is the
number of simulated paths (`CamConfig.samples`); `Dates` is the number of simulation dates
(after the as-of date; t=0 is not on the cube's date axis, while an exposure profile starts
at t=0).

---

## `engine.market`

Today's market (ORE's `TodaysMarket`), what every trade is valued against.

### `Market`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `asof` | `ORE.Date` | *required* | The valuation date; every trade's `evaluation_date` must equal it. |
| `currencies` | `Dict[str, CurrencyMarket]` | *required* | Per currency code. |
| `fx_spots` | `Dict[str, float]` | `{}` | Keyed `<foreign><domestic>`, e.g. `"EURUSD"`; both currencies must be in the market. |
| `equities` | `Dict[str, EquityMarket]` | `{}` | Equity spots (the cross-asset model simulates them; no equity trade yet). |

Methods: `currency(code)`, `index_curve(currency, index)`, `swaption_vols(currency)` (each
raising `KeyError` naming what is missing), `fx_spot(foreign, domestic)`.

### `CurrencyMarket`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `discount_curve` | `ZeroCurveConfig` | *required* | The currency's discount curve. |
| `index_curves` | `Dict[str, ZeroCurveConfig]` | `{}` | Forwarding curves keyed by index name (`index_name("USD", 6)` = `"USD-SIMINDEX-6M"`). |
| `swaption_vols` | `Optional[SwaptionVolSurface]` | `None` | The ATM normal swaption matrix (needed by a Bachelier European and a calibrated Bermudan/American). |

### `ZeroCurveConfig`

`times` (ACT/365 year fractions from the as-of date, starting at 0) and `rates` (continuously
compounded zero rates): linear in the zero rate between pillars, QuantLib's flat forward beyond
the last. `provenance` is optional metadata.

### `SwaptionVolSurface`

`option_tenors`, `swap_tenors` (ORE period strings) and `vols[i][j]` (normal volatilities),
bilinear between nodes as QuantLib's `SwaptionVolatilityMatrix`. `volatility(reference,
option_time, swap_length)` reads it; `weights(reference, option_time, swap_length)` is the
`[option tenors, swap tenors]` weight of each quote in that read (the AD Vega's chain rule).

### `EquityMarket`

`currency`, `spot`, optional `dividend_curve`.

---

## `engine.simulation.config`

The simulation: ORE's `simulation.xml`. See [Market Simulation](../concepts/market-simulation.md).

### `CamConfig`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `dates` | `Tuple[ORE.Date, ...]` | *required* | Simulation dates, after the market's as-of date, increasing. |
| `base_currency` | `str` | *required* | The domestic currency of the model and the reporting currency. |
| `ir` | `Dict[str, LgmConfig \| HullWhiteConfig]` | *required* | The model of each simulated currency. |
| `fx_volatilities` | `Dict[str, float]` | `{}` | Black-Scholes FX volatility per foreign currency. |
| `equity_volatilities` | `Dict[str, float]` | `{}` | Per equity. |
| `correlations` | `Dict[Tuple[str, str], float]` | `{}` | Between factors named `IR:USD`, `FX:EURUSD`, `EQ:SP5`. |
| `curve_tenors` | `Tuple[str, ...]` | ORE's 13, `3M` … `30Y` | The scenario market's curve tenors. |
| `samples` | `int` | `1000` | Paths (a power of two keeps Sobol balanced). |
| `seed` | `int` | `42` | Sobol scrambling seed. |
| `swaption_vol_decay` | `str` | `"ForwardVariance"` | How the swaption volatilities are seen from a path date (`"ConstantVariance"` too). |

### `LgmConfig` / `HullWhiteConfig`

One currency's `<LGM>`: `reversion` (*required*), `volatility` (`0.01`), `calibration_expiries`
and `calibration_terms` (tenor tuples of equal length; non-empty means bootstrap the
volatility to that co-terminal basket of the market's swaption volatilities), `swap_index`
(`SwapIndexConventions`: the helpers' fixed tenor, fixed day counter and index tenor).
`HullWhiteConfig` is the same with ORE's Hull-White volatility parametrization: `volatility`
is the short rate's.

### `simulate(market, config, model=None, precision=Precision()) -> ScenarioMarket`

The scenario market: `numeraire [S, D]`, `discount[ccy]`/`index[name]` (`ScenarioCurves`:
`log_discounts [S, D, K+1]` at `tenor_times [D, K+1]`), `fx[ccy]`, `equity[name]`,
`states [S, D, factors]`, `dates`, `times`. `model` defaults to
`build_cross_asset_model(market, config)` (which calibrates). `precision.simulation` sets the
shocks and states, `precision.market` the market; every path array is returned in
`precision.market.storage`. `ScenarioMarket.map_arrays(fn)` applies `fn` to every path array.

### `build_cross_asset_model(market, config, sigmas=None) -> CrossAssetModel`

The cross-asset model, each currency with a basket bootstrapped first
(`engine.calibration.cam.calibrate_cam`).

---

## Trade configs (`engine.instruments`)

What ORE's trade XML holds. Every config has two **required keyword fields**: `trade_id`
(ORE's `<Trade id>`, a non-empty string, unique in a portfolio and echoed on every result)
and `evaluation_date` (an `ORE.Date`; on a `Market` it must be the market's as-of date).
None carries a curve or a model: `currency` (default `"USD"`) and `index_tenor_months`
(default `6`) name the curves in the `Market`, and the engines' models come from the run
configuration. A model or curve field is a `TypeError`.

**The dates are the trade (audit M-4).** A schedule is given **either** as
`effective_date`/`maturity_date` **or** as `swap_tenor` (an ORE period string, booking
convenience, resolved once to spot-starting dates on `evaluation_date`), never both and
never neither. `dataclasses.replace(cfg, evaluation_date=d)` is the same trade on another
date: paid flows drop out, a coupon fixed before `d` pays its historical fixing from
`fixings` (`{ORE.Date: rate}`; missing is `MissingFixingError`, as ORE refuses it).

### `SwapConfig` (`engine.instruments.swap`)

`notional`, `fixed_rate`, `payer` (pays fixed), the schedule, `index_tenor_months`,
`floating_spread` (`0.0`), `accrual_day_count` (`"ACT/365"`, an allowlisted name), `fixings`,
`currency`, `trade_id`, `evaluation_date`.

### `SwaptionConfig` (`engine.instruments.european_swaption`)

The underlying swap's terms as `SwapConfig` (no `accrual_day_count`), plus the exercise:
`exercise_date`, or with `swap_tenor` the booking conveniences `forward_start` (an
`ORE.Period` beyond the spot lag) and `exercise_lag_days` (`2`); and `settlement`
(`"Physical"` \| `"Cash"`). On or after `exercise_date` it has expired (`is_expired()`) and
is worth 0.

### `BermudanSwaptionConfig` (`engine.instruments.bermudan_swaption`)

The underlying swap's terms, `exercise_dates` (ascending `ORE.Date`s; a year fraction is a
`TypeError`; dates on or before `evaluation_date` are not exercise opportunities), `fixings`,
`settlement`. `exercisable_dates(cfg)` lists the underlying's accrual starts.

### `AmericanSwaptionConfig` (`engine.instruments.american_swaption`)

As the Bermudan, with `first_exercise_date`/`last_exercise_date` instead of `exercise_dates`.
`option_times(exercise_time_steps_per_year)` is ORE's uniform option-time grid over the
window (the step count from the engine's `LgmSwaptionEngineConfig`, `>= 1`).

### The grid engine (`engine.instruments.bermudan_swaption`)

| Function | Notes |
|---|---|
| `prepare_bermudan(cfg, *, reversion, sigma, n_per_std, std_devs, exercise_time_steps_per_year, curve=None, index_curve=None)` | The trade's prepared structure for the engine, with an explicit LGM (`sigma`: float or `Sigma`). |
| `grid_value(prepared) -> jax.Array` | ORE's `NumericLgmMultiLegOptionEngine` value on the prepared curves. |

### `BondConfig`, `CouponPeriod` (`engine.instruments.treasury`)

A Treasury bill or note, discounted on its currency's curve (ORE's `DiscountingRiskyBondEngine`
without credit; `engine.valuation.portfolio.bond_legs`), today and on every path.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `face_amount` | `float` | *required* | **Signed** — a short position is a negative face. |
| `maturity_date` | `ORE.Date` | *required* | Strictly after `evaluation_date`, else `BondPricingError`. |
| `coupon_rate` | `float` | `0.0` | Annual coupon as a decimal. |
| `coupon_schedule` | `Tuple[CouponPeriod, ...]` | `()` | Empty ⇒ a **bill**. `CouponPeriod(start_date, end_date, payment_date=None)`, supplied, never generated. |
| `redemption_fraction` | `float` | `1.0` | Non-negative. |
| `accrual_day_count` | `str` | `"ACT/ACT (ICMA)"` | Resolved at construction and refused if unsupported. |
| `currency`, `trade_id`, `evaluation_date` | | | As above. |

`is_bill`; `accrued_interest(cfg)` (recomputed from the schedule, position-signed).
Construction refuses a maturity at or before the evaluation date, a negative redemption, a
schedule with a zero coupon and a coupon without a schedule (`BondPricingError`).

---

## `engine.valuation`

ORE's `ValuationEngine`: every trade with its configured engine, today and on every path.
See [Instruments](../instruments/swaps.md).

### `PricingConfig` (`engine.valuation.config`)

| Field | Type | Default | Meaning |
|---|---|---|---|
| `european` | `str` | `"Bachelier"` | `"Bachelier"` (ORE's default, on the market volatility) or `"Jamshidian"`. |
| `jamshidian` | `Optional[JamshidianEngineConfig]` | `None` | The Jamshidian engine's Hull-White model, `(reversion, volatility)`, both finite and > 0. Required with `european="Jamshidian"`, refused without. |
| `bermudan`, `american` | `LgmSwaptionEngineConfig` | ORE's builder defaults | `reversion` (`0.0`), `volatility` (`0.01`, the start or the fixed model), `calibration` (`"Bootstrap"` \| `"None"`), `strategy` (`"CoterminalDealStrike"`), `reference_calibration_grid` (`"400,3M"`), `shift_horizon` (`0.0`, the only value implemented, I-32), `n_per_std` (`30`), `std_devs` (`5.0`), `exercise_time_steps_per_year` (`24`), `swap_index`. |
| `recalibrate` | `bool` | `True` | Recalibrate each Bermudan/American on every path date, as ORE's `ValuationEngine`. |

### `engine.valuation.portfolio`

| Function | Returns |
|---|---|
| `value_today(trades, market, base_currency, pricing=PricingConfig()) -> List[float]` | Today's values in the base currency. |
| `value_portfolio(trades, market, scenarios, base_currency, pricing=PricingConfig(), decay="ForwardVariance", precision=Precision()) -> PortfolioValuation` | `today [T]` (float64) and `columns` (each trade's `[S, D]`) in the base currency; each trade priced on the scenarios loaded at its `precision.precision_for(trade).compute`, its column stored at that `storage`. `PortfolioValuation.cube` is `[S, D, T]`, in the columns' shared format, or float64 when they differ. |
| `value_on(cfg, context, pricing) -> float` | One trade on a `PricingContext` (a date's curves, volatilities, fixings). |
| `validate_trades(trades, market, pricing=PricingConfig(), precision=Precision())` | Refuses, naming the trade, a trade off the market's date, a curve or volatility the market lacks, or an engine's refusal; and a precision override naming no trade or product (`Precision.check_overrides`). |
| `PRODUCTS` | The products the pipeline prices (each trade config's `product`): the keys of `Precision.by_product`. |
| `reads_swaption_vols(cfg, pricing) -> bool` | Whether the trade's engine reads the market's swaption volatilities. |

---

## `engine.risk.var_es`

Every function here is instrument-agnostic — see
[Risk Statistics](../risk/var_es.md) and
[Architecture: modules agree on shapes, not code](../concepts/architecture.md#design-principle-modules-agree-on-shapes-not-code).

### `portfolio_pnl(npv_cube: jax.Array, base_npv: float) -> jax.Array`

**Parameters**
- `npv_cube` — `[Scenarios, TimeSteps, Trades]`, from any pricer.
- `base_npv` — the portfolio's value today (t=0), from a separate zero-shock
  revaluation. See [Risk Statistics: the P&L baseline](../risk/var_es.md#the-pl-baseline-what-are-gainslosses-measured-against).

**Returns** `[Scenarios, TimeSteps]` — `sum(npv_cube, axis=Trades) - base_npv`.

### `value_at_risk(pnl: jax.Array, percentile: float) -> jax.Array`

**Parameters**
- `pnl` — `[Scenarios, TimeSteps]`, typically from `portfolio_pnl`.
- `percentile` — e.g. `0.99` for 99% VaR.

**Returns** `[TimeSteps]` — always `>= 0`.

### `expected_shortfall(pnl: jax.Array, percentile: float) -> jax.Array`

Same signature as `value_at_risk`. **Returns `NaN`** for any time step whose loss tail
(strictly worse than that step's VaR) is empty — see
[Risk Statistics: the formulas](../risk/var_es.md#the-formulas). Callers must check
for this explicitly.

### `compute_risk_metrics(npv_cube, base_npv, percentiles=(0.95, 0.99), include_diagnostics=True) -> Dict[str, jax.Array]`

The main entry point — combines the three functions above.

**Parameters**
- `npv_cube` — `[Scenarios, TimeSteps, Trades]`.
- `base_npv` — see `portfolio_pnl` above.
- `percentiles` — which confidence levels to compute VaR/ES at. Default `(0.95, 0.99)`.
- `include_diagnostics` — attach per-percentile convergence diagnostics. Default `True`;
  `False` returns exactly the pre-W0.6 key set.

**Returns** a `dict`, every value shaped `[TimeSteps]`, with these keys per entry in
`percentiles` (shown for `0.99`):

| Key | Meaning |
|---|---|
| `VaR_99` | Value at Risk. |
| `ES_99` | Expected Shortfall. NaN where the strict tail is empty. |
| `ES_99_tailCount` | **Effective sample size** — how many observations the ES mean actually averaged. At 99% over 10,000 scenarios this is ~100, so the estimate rests on 1% of the sample. |
| `ES_99_standardError` | Monte Carlo standard error of the ES mean (`s/√n`, `ddof=1`). **NaN, never `0.0`, when `n < 2`** — `0.0` would read as "perfectly converged" for the least trustworthy case. |

The two diagnostic keys are additive (W0.6, part of [I-11](../planning/known-issues.md#i-11)); the
`VaR_*`/`ES_*` keys and values are unchanged. See
[EOD Integration: tail diagnostics](eod-integration.md#tail-statistics-carry-their-own-convergence-diagnostics).

### Risk measure constants

`RISK_MEASURES` = `("risk-neutral-pricing", "historical-forecast", "deterministic-stress")`,
and `ENGINE_RISK_MEASURE` = `"risk-neutral-pricing"` — what this engine actually produces. A
risk-neutral exposure is **not** a calibrated forecast of tomorrow's loss; reporting one where
the other is expected is a category error no numerical accuracy fixes.

---

## `engine.risk.sensitivities`, `engine.risk.greeks`, `engine.risk.price_functions`

The Greeks — see [Greeks](../risk/greeks.md) for the keys and the two methods.

| Name | Returns |
|---|---|
| `SensitivityConfig` | ORE's `sensitivity.xml`: `curve_tenors` (ORE's 13), `curve_shift` (`1e-4`), `vol_shift` (`1e-4`), `theta_days` (`1`), `swaption_vol_decay` (`"ForwardVariance"`). |
| `portfolio_sensitivities(trades, market, base_currency, pricing=PricingConfig(), config=SensitivityConfig())` | Bump Greeks: `{trade index: {key: array}}`. |
| `portfolio_greeks(trades, market, base_currency, pricing=PricingConfig(), config=SensitivityConfig())` | AD Greeks, the same keys. |
| `curve_greeks(cfg, market, pricing, shift)` | AD Delta/Gamma per pillar of each curve the trade reads, in its currency. |
| `vega_greek(cfg, market, pricing, shift)` | AD Vega `[option tenors, swap tenors]`, or `None`. |
| `trade_theta(value, base, cfg, theta_context, fx)` | ORE's Theta, shared by both methods. |
| `trade_price_function(cfg, market, pricing=PricingConfig(), dtype=jnp.float64)` | `TradePriceFunction(curves, price)`: the trade's t=0 price as a JAX function of its curves' pillar rates; `curves` are `(kind, name)` keys (`("discount", "USD")`, `("index", "USD-SIMINDEX-6M")`). Shared by the AD Greeks and market-risk revaluation. |
| `bermudan_price_function(cfg, market, pricing, dtype)` | `BermudanPriceFunction(price, sigma, calibration)`: the grid engine on today's calibration, `price(disc, index, sigma_values=None)`. |

## `engine.risk.exposure`

ORE's `ExposureCalculator` statistics over a simulated cube — see [Exposure](../risk/exposure.md).

| Function | Returns |
|---|---|
| `exposure_profile(npv [S,D], npv0, numeraire, discount, times, quantiles, maturity=None, dates=None, asof=None)` | `ExposureProfile` for one trade: `times` (t=0 first), `epe`, `ene`, `ee_b`, `eee_b`, `epe_b`, `eepe_b`, `basel_epe`, `basel_eepe`, `pfe` (`"PFE_95"` → `[D+1]`) |
| `netting_set_profile(npv_cube [S,D,N], npv0_per_trade, numeraire, discount, times, quantiles, dates=None, asof=None)` | The same for the netted sum of `N` trades |

## `engine.market_risk`

Short-horizon VaR/ES by full revaluation at t=0 — see [Market Risk](../risk/market-risk.md).

| Name | Kind | Summary |
|---|---|---|
| `RateRiskFactors.from_market(market)` | class | Every curve of the market, named `discount:<ccy>` / `index:<name>`; also `from_curves(curves, names)`, `index_of(name)`, `size`, `slice_of(i)`, `base_rates()`, `labels()` |
| `monte_carlo_scenarios(factors, covariance, horizon_days, num_scenarios, seed=42)` | function | Gaussian horizon moves via scrambled Sobol → `ShockScenarios` |
| `historical_scenarios(factors, history, horizon_days, dates=None)` | function | Overlapping horizon moves of a `[dates, factors]` history → `ShockScenarios` |
| `covariance_from_history(history, horizon_days)` | function | Sample covariance of those moves, `[F, F]` |
| `horizon_moves(history, horizon_days)` | function | The overlapping moves themselves, `[D-h, F]` |
| `ShockScenarios` | dataclass | `factors`, `shifts [S, F]`, `horizon_days`, `source`, `measure`, `windows` |
| `MarketRiskRequest(trades, market, scenarios, pricing=PricingConfig(), quantiles=(0.99, 0.975), precision=Precision(), batch_size=256)` | dataclass | Every factor must be the market's curve of its name; every curve a trade reads must be a factor. `precision.simulation` rounds and stores the shifts, `precision.precision_for(trade)` sets each trade's revaluation and stored P&L; VaR/ES are float64. Trade ids are unique |
| `run_market_risk(request) -> MarketRiskResult` | function | `base_npv`, `base_npv_per_trade` (each trade's revaluation at its pricing compute precision), `pnl [S, N]` (float64-loaded), `portfolio_pnl [S]`, `risk` (`VaR_99`, `ES_97.5`, tail counts, standard errors), `measure`, `source`, `horizon_days`, `num_scenarios`, `risk_factors`, `warnings` |

---

## `engine.calibration`

| Module | Contents |
|---|---|
| `engine.calibration.cam` | `calibrate_cam(market, ir_configs) -> Dict[str, Calibration]`: each currency with a basket bootstrapped to the market's swaption volatilities, in its own parametrization (a Hull-White currency's σ by `hull_white_matching_zeta`). |
| `engine.calibration.ore_lgm` | ORE's `LgmBuilder`: `build_basket(reference, expiries, terms, conventions, deal_strikes)` (QuantLib `SwaptionHelper`s), `basket_vols`, `price_pair(helper, disc, index, vol, reversion, zeta)` (market and model value), the bootstrap, batched over path curves. Used by the CAM's and every option's calibration. |
| `engine.calibration.basket`, `engine.calibration.lgm` | The standalone `POST /calibration/lgm` route's co-terminal basket (`build_coterminal_basket`, `CalibrationTarget`) and Hagan bootstrap (`calibrate_lgm_sigma -> CalibrationResult`: `sigma`, `market_prices`, `model_prices`, `rmse`). See [Calibration](calibration.md). |

---

## `engine.models`

The shared foundation — see [Models & Trades](models-and-trades.md).

### `engine.models.curves`

| Name | Notes |
|---|---|
| `ZeroCurve` | JAX pytree: `pillar_times`, `pillar_rates`; `ZeroCurve.from_config(config, dtype)`. Linear zero rate, QuantLib's flat forward beyond the last pillar. |
| `zero_rate`, `log_discount`, `discount`, `forward_rate` | On a `ZeroCurve`. |
| `DiscountCurve`, `loglinear_log_discount` | A batched curve of log discount factors at tenor times, log-linear between them (a scenario market's curve). |

### `engine.models.lgm`

| Name | Notes |
|---|---|
| `Sigma` | JAX pytree: `times`, `values` (`len(values) == len(times) + 1`); `Sigma.flat(sigma)`. |
| `as_sigma` | Upgrades a scalar to a one-bucket `Sigma`. |
| `H` / `H_prime` | `(a, t)`: the LGM's state-space function and its derivative. |
| `zeta` | `(sigma, t)`: Hagan's ζ(t) = ∫α². |
| `hull_white_zeta` | `(a, sigma, t)`: the Hull-White parametrization's ζ(t) = ∫σ²e^{2as}ds, piecewise exact. |
| `hull_white_matching_zeta` | `(a, hagan: Sigma, last) -> Sigma`: the Hull-White σ with Hagan's ζ at every bucket end (exact calibration conversion). |
| `bond_price`, `numeraire`, `bond_option_sigma`, `r_from_x`, `x_from_r` | The LGM's closed forms (Hagan parametrization). |

### `engine.models.hull_white`

`bond_call` / `bond_put`: Black on a bond, the Jamshidian engine's building block.

### `engine.models.ore_builders`

ORE trade building and the time axis: `build_vanilla_swap` (from booked dates, via
`MakeVanillaSwap`), `resolve_swap_dates` / `book_swap_dates` (`MakeVanillaSwap`'s tenor rule),
`ibor_index`, `par_coupon_forecast_period` (the Ibor index's own fixing period, I-31),
`known_fixing` / `MissingFixingError` (ORE's `InterestRateIndex::fixing`), `is_live` (ORE's
`hasOccurred`), `time_from_reference` and `TIME_AXIS_DAY_COUNTER` (ACT/365), `validate_tenor`,
`validate_fixings`, `fixed_leg_cashflows` / `LegCashflows` (the standalone basket's fixed leg).

---

## `engine.portfolio`

The top-level entry point — see [The Portfolio Entry Point](portfolio-entrypoint.md) for the
full write-up (this is the quick reference).

### `PortfolioRequest`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `market` | `Market` | *required* | Today's market. |
| `trades` | `List[SwapConfig \| SwaptionConfig \| BermudanSwaptionConfig \| AmericanSwaptionConfig \| BondConfig]` | *required* | Any mix and order; `trade_id`s unique. |
| `config` | `RunConfig` | `RunConfig()` | The run configuration (below). |
| `pfe_quantiles` | `Sequence[float]` | `(0.95, 0.99)` | Quantiles of the PFE profiles. |
| `compute_greeks` | `bool` | `False` | By `config.greeks.method`. |
| `scenario_risk` | `bool` | `True` | Build `npv_cube` and the exposure profiles (needs `config.simulation`). |

### `RunConfig` (`engine/portfolio/config.py`)

| Field | Type | Default | Meaning |
|---|---|---|---|
| `simulation` | `Optional[CamConfig]` | `None` | ORE's `simulation.xml`; the model per currency is `ir[ccy]`. Required for scenario risk. |
| `pricing` | `PricingConfig` | `PricingConfig()` | Engine per product. |
| `greeks` | `GreeksConfig` | `GreeksConfig()` | `method` (`"Bump"` \| `"AD"`), `sensitivity` (`SensitivityConfig`). |
| `precision` | `Precision` | float64 everywhere | Storage, compute and accumulate format per adjustable stage (`simulation`, `market`, `pricing`); see `engine.precision` below. |
| `base_currency` | `Optional[str]` | `None` | Reporting currency; `None` is the simulation's, else USD. |

`reporting_currency` (property) resolves `base_currency`.

### `PortfolioResult`

| Field | Type | Meaning |
|---|---|---|
| `base_npv` | `float` | Today's portfolio value in the reporting currency. |
| `npv_cube` | `jax.Array` | `[Scenarios, Dates, Trades]`, the request's `trades` order; float64, loaded from the cube stored at `precision.pricing.storage`. |
| `exposure` | `Optional[ExposureProfile]` | Netting-set exposure; `None` without scenario risk. |
| `trade_exposures` | `List[ExposureProfile]` | Standalone exposure per trade. |
| `greeks` | `Optional[Dict[int, Dict[str, jax.Array]]]` | Keyed by trade index in `request.trades`. |
| `warnings` | `List[str]` | Run warnings. |
| `base_npv_per_trade` | `List[float]` | Today's value per trade. |
| `trade_ids` | `List[str]` | Each trade's id, in request order. |
| `scenario_risk_available` | `bool` | Whether `exposure`/`npv_cube` were computed. |
| `measure` | `Optional[str]` | `"risk-neutral-pricing"`, or `None` without scenario risk. |

### `price_portfolio(request: PortfolioRequest) -> PortfolioResult`

The main entry point — see [The Portfolio Entry Point](portfolio-entrypoint.md#price_portfoliorequest-portfoliorequest---portfolioresult).
`engine.portfolio.worker_pool.submit_pricing_job(request) -> Future[PortfolioResult]` runs it
in a worker process (the HTTP route's path).

---

## `engine.precision`

The precision of a run ([details/precision.md](../planning/details/precision.md)); imports
nothing from the pipeline. `Precision` and `StagePrecision` are also exported by
`engine.portfolio`.

| Name | Kind | Summary |
|---|---|---|
| `StagePrecision(storage="float64", compute="float64", accumulate="float64")` | frozen dataclass | One stage's formats, by name. Refused, naming the field: a name outside the table; a format before the roadmap step that enables it (compute below float32 or `accumulate != compute`: 2.8); `storage` wider than `compute`; `accumulate` narrower than `compute`. `storage_dtype`, `compute_dtype`, `scaled_storage` |
| `Precision(simulation, market, pricing, by_product={}, by_trade={}, rounding="nearest", rounding_seed=0)` | frozen dataclass | A `StagePrecision` per adjustable stage, each float64 by default, and the pricing stage's overrides by product and by trade id (kept as an immutable `Overrides`); the rounding into a scaled storage format (`"nearest"` or `"stochastic"`, refused when nothing is stored scaled) and its seed. `Precision.throughout(name)` sets all three stages to `name`; `precision_for(trade)` is a trade's pricing stage (`by_trade`, else `by_product`, else `pricing`); `check_overrides(trades, products)` refuses a key that names nothing; `store(x, storage, stream, axis=0)` stores with the policy's rounding, `stream` naming the array (it seeds a stochastic rounding) |
| `FORMATS`, `FORMAT_NAMES`, `Format` | table | name -> dtype, bits, mantissa bits, minimum normal exponent, max, scaled, enabling steps: `float64`, `float32`, `float16`, `bfloat16`, `float8_e4m3fn`, `float8_e5m2` |
| `format_of(name)`, `dtype_of(name)`, `name_of(dtype)` | functions | Table lookups; an unknown name is refused |
| `store(x, name, rounding="nearest", key=None, axis=0)`, `load(x, dtype)` | functions | The only casts between stages; at the array's own dtype both return it unchanged. A scaled format returns a `Stored`; `stochastic` needs a `key` (`rounding_key(seed, stream)`) |
| `Stored(values, scales, format, axis)` | pytree | A scaled format's values (the array's shape) and float32 power-of-two scales, one per `BLOCK` (32) entries along `axis`, the scenario axis; `shape`, `dtype` (the format's), `nbytes` (scales included) |
| `BLOCK`, `ROUNDINGS` | constants | 32; `("nearest", "stochastic")` |
| `require_precision(owner, value)` | function | Refuses anything but a `Precision`, naming the replacement of the retired 32/64 shape (`RETIRED_SHAPE`, decision A-12) |

---

## `demos.demo_scenarios`

Demo and test scenarios in `demos/`, outside the `engine` package: the demos and tests use
them; the engine never imports them. See
[Architecture: demos/demo_scenarios.py](../concepts/architecture.md#demosdemo_scenariospy-shared-example-configurations).

| Name | Meaning |
|---|---|
| `EVAL_DATE` | The scenarios' as-of date (2026-07-30). |
| `PILLARS`, `USD_DISCOUNT`, `USD_INDEX`, `EUR_DISCOUNT`, `EUR_INDEX`, `VOLS`, `DATES` | The data. |
| `demo_market(currencies=("USD", "EUR")) -> Market` | USD (3% → 5%) and EUR curves, 6M index curves, swaption volatilities; with EUR, the EURUSD spot and an equity. |
| `demo_simulation(model="HullWhite", samples=4096, dates=DATES, currencies=("USD", "EUR"), calibrated=False) -> CamConfig` | The simulation, `model` (`"HullWhite"` \| `"LGM"`) for every currency; two currencies add FX, equity and correlations. |
| `demo_market_json(...)`, `demo_simulation_json(...)` | The same as the HTTP request's JSON. |

---

## `engine.api`

The HTTP boundary over `engine.portfolio.price_portfolio` — see [HTTP API](http-api.md).
Requires the `api` extra (`pip install -e .[api]`); `engine.portfolio` and everything below it
has no dependency on this package.

| Module | Contents |
|---|---|
| `engine.api.app` | `create_app() -> FastAPI` / `app`. Run with `uvicorn engine.api.app:app`. |
| `engine.api.routes` | `GET /health`, `GET /version`, `POST /portfolio/price` (also `/v2/portfolio/price`), `GET /portfolio/price/{job_id}`, `POST /calibration/lgm`. |
| `engine.api.market_schemas` | `MarketPortfolioRequestSchema` and its parts (market, trades, `CamConfigSchema` with `LgmConfigSchema`/`HullWhiteConfigSchema`, `PricingConfigSchema`, `GreeksConfigSchema`), each with `.to_dataclass()`. Refuses unknown fields and the retired Hull-White shape. |
| `engine.api.schemas` | Shared schemas (curves, coupon periods, precision) and the results: `PortfolioResultSchema`, `GreeksSchema`, `ExposureProfileSchema`, `JobStatusSchema`, the calibration route's schemas. |
| `engine.api.eod_routes` | `router` (prefix `/eod`) — the TraderX EOD contract, plain dicts under a published JSON Schema. See [EOD Integration](eod-integration.md#w164--the-eod-http-routes). |

---

## `engine.integration`

The TraderX EOD boundary: a hash-verified bundle in, an **identified** risk result out. Its
governing rule is that **nothing is ever silently approximated** — an explicit `unsupported`
is recoverable, a plausible wrong number is not.

Documented in full in [The EOD Integration Boundary](eod-integration.md); this table is the
module index. **This package imports no FastAPI, no Pydantic, no JAX, and no simulation
pricer** — only `ORE`, for date and day-count arithmetic. The dependency runs
`engine.api` → `engine.integration`, never the reverse, and it is enforced by
`tests/test_integration_pipeline.py::TestPackageImportsNoSimulationPricer`.

| Module | Task | Contents |
|---|---|---|
| `bundle` | W0.1 | `load_bundle(root) -> Bundle`, `Bundle`, `BundleIntegrityError`. Reads and hash-verifies every artifact **in binary, exactly as read** — see the CRLF trap. |
| `terms` | W0.2 / W1.6.1 | `join_terms`, `TermsEntry`, `AccrualBasis`, `TermsJoinError`, `SUPPORTED_TERMS_SCHEMAS`. |
| `normalize` | W0.3 | `normalize_position`, `NormalizedPosition`, `Quantity`, `MAPPING_VERSION`. |
| `conventions` | W0.4 | `check_conventions`, `ConventionRefusal` — a positive **allowlist**, applied before any pricing object is constructed ([I-05](../planning/known-issues.md#i-05)). |
| `result` | W0.5 | `RiskResult`, `ItemResult`, `Coverage`, `CALCULATIONS`, `STATUSES`. |
| `market_inputs` | W0.6 | `resolve_market_inputs`, `MarketInputs`, `MarketInputsNotSupplied`, `ASSUMED_PROFILES`, `CurveProvenance`, `ENGINE_RISK_MEASURE`. No silent fallback ([I-11](../planning/known-issues.md#i-11)). |
| `identity` | W0.7 | `item_id`, `ItemIdentity` — an opaque, stable id plus the source identity block, on **every** row including refusals ([I-10](../planning/known-issues.md#i-10)). |
| `publication` | W0.8 | `ResultStore`, `PublicationError`, `default_store_root` — the crash-safe durable result store ([I-08](../planning/known-issues.md#i-08)). |
| `capabilities` | W0.9 | `capabilities()` — the supported (product × convention × calculation) matrix. |
| `bill` | W1.2 | `price_bill`, `BillPrice`, `is_bill`, `BillPricingError`. |
| `note` | W1.3 | `price_note`, `NotePrice`, `rate_sensitivity`, `AccruedReconciliation`, `accrual_mismatch_tolerance`, `is_note`. |
| `equity` | W1.4 | `price_equity`, `EquityPosition`, `read_position`, `is_equity` — a **refusal** naming the missing spot/FX source ([I-18](../planning/known-issues.md#i-18)). |
| `schema_version` | W1.6.2 | `RESULT_SCHEMA_VERSION`, `CAPABILITY_SCHEMA_VERSION` — a dependency-free leaf that breaks the `result` ↔ `schema` cycle. |
| `schema` | W1.6.2 | `result_schema()`, `capability_schema()` — JSON Schema **derived** from the frozen vocabulary, never hand-written. |
| `workload` | W1.6.4 / W0.8 | `workload_key`, `AttemptStore`, `Attempt`, `UNKNOWN_WORKLOAD` — the attempt state machine and the four lookup states. |
| `pipeline` | — | `price_bundle(bundle_or_path, market_inputs=None) -> RiskResult` — the composition of all of the above. Accepts a loaded `Bundle` or a path to load one from. |
