# Risk: Delta, Gamma, Vega, and Theta

Two methods, chosen by the run configuration's `greeks` (decision A-5,
[compliance/decisions.md](../../compliance/decisions.md)), for every trade type and either
simulation model, with the same keys:

```python
config = RunConfig(greeks=GreeksConfig(method="Bump",            # or "AD"
                                       sensitivity=SensitivityConfig(curve_tenors=("1Y", "5Y", "10Y"),
                                                                     theta_days=3)))
greeks = price_portfolio(PortfolioRequest(market=market, trades=trades, config=config, scenario_risk=False,
                                          compute_greeks=True)).greeks
```

**Modules:** [`engine/risk/sensitivities.py`](../../engine/risk/sensitivities.py) (`Bump`,
`portfolio_sensitivities`, and Theta for both), [`engine/risk/greeks.py`](../../engine/risk/greeks.py)
(`AD`, `portfolio_greeks`), [`engine/risk/price_functions.py`](../../engine/risk/price_functions.py)
(each trade's price as a JAX function of its market curves)

## Plain-language summary

VaR and Expected Shortfall (see [VaR & Expected Shortfall](var_es.md)) answer "how much
could we lose across thousands of simulated futures?" **Greeks** answer a different,
complementary question: "if today's market moves by a small, specific amount, how much
does this one trade's value change?" A bank's trading desk uses Greeks constantly — to
hedge, to understand which market moves matter for a position, and to explain day-to-day
P&L.

- **Delta** — how much a trade's value changes for a small move in one point of one interest
  rate curve (e.g. "if the 5-year rate rises by 0.01%, this trade gains $150"). Reported per
  point, since a real trade is more sensitive to some maturities than others.
- **Gamma** — how much *Delta itself* changes as rates move: how curved a trade's value is.
  A plain swap has very little Gamma; a swaption has meaningful Gamma.
- **Vega** — how much an option's value changes when one quoted swaption volatility moves.
- **Theta** — how much a trade's value changes purely from one day passing, with the market
  held still.

## The keys

Per trade (by request index), in the reporting currency:

| Key | Bump (ORE's defaults) | AD |
|---|---|---|
| `delta:discount:<ccy>`, `delta:index:<name>` | `NPV(up) − NPV(base)`, an absolute 1bp zero-rate shift of one curve tenor (`ShiftScheme::Forward`), per `SensitivityConfig.curve_tenors` tenor | `dNPV/dz_i · shift` per pillar of the market curve |
| `gamma:discount:<ccy>`, `gamma:index:<name>` | `NPV(up) − 2·NPV(base) + NPV(down)` | `d²NPV/dz_i² · shift²` (the diagonal, as ORE computes no cross-gammas by default) |
| `vega:<ccy>` `[option tenors, swap tenors]` | `NPV(up) − NPV(base)` per swaption quote, absolute 1bp normal-vol shift | `dNPV/dquote · vol_shift` |
| `theta` | `NPV(thetaDate) − NPV(base) + flows paid in (asof, thetaDate]`, `thetaDate = asof + theta_days` calendar days | the same function |

Vega is present for a trade whose engine reads the swaption volatilities: a European on
Bachelier, a Bermudan/American calibrated to them. A swap, a bond, a Jamshidian European and
an uncalibrated option have none (omitted, not zero). A bond reads its discount curve only.

## Scope: every rate-derivative instrument in this codebase

Swaps, Europeans (either engine), Bermudans, Americans and bonds, with either simulation
model (the Greeks are of today's value; the model only simulates).

## Bump: ORE's sensitivity analysis

`portfolio_sensitivities` reproduces ORE's `SensitivityAnalysis` (`sensitivityanalysis.cpp`,
`sensitivitycube.cpp`): each trade is priced with its t=0 engine on ORE's sensitivity
simulation market — every curve sampled at the curve tenors, log-linear between them,
flat-forward beyond — with one tenor's zero rate (or one volatility quote) shifted at a time.
A Bermudan/American is recalibrated under every shift, as ORE does. These follow ORE's source
(gates V-2, V-3 in [ORE Parity](../reference/ore-parity.md#verification-gates)) but are not yet
checked against an OREApp sensitivity run ([I-51](../planning/known-issues.md#i-51)).

## Why it's built this way: matching ORE's exact convention, via autodiff instead of finite differences

The AD method gives the derivatives exactly, in one pass per trade, where the bump method
reprices once per tenor and quote. Each trade's price is a pure JAX function of its market
curves' pillar rates (`trade_price_function`: discounting legs, Bachelier or Jamshidian, the
grid engine on the calibrated LGM, discounted bond flows), and `curve_greeks` takes its
gradient and the diagonal of its Hessian — one linearization of the gradient over all the
trade's curves, then one Hessian-vector product per pillar, batched with `vmap`, never the
full Hessian (`_gradients_and_hessian_diagonals`). Scaled by the shift (Delta) and its square
(Gamma), they are the shift → 0 limit of ORE's numbers. Each trade's Delta and Gamma are one
compiled program, and its Vega gradient another, shared by every trade of the same product
and shape ([profiling §3.8](../concepts/profiling.md#38-the-ad-greeks-as-one-program-per-product-2026-10-06-roadmap-24)).

**How the two methods differ, beyond the shift size.**

- *Axis.* Bump Deltas are per sensitivity tenor on ORE's resampled curve; AD Deltas per pillar
  of the market's own curve. Summed, both are the parallel Delta.
- *Curvature.* ORE's Delta is a forward difference, `Δh + ½Γh²`; with the curvature removed
  (Delta − Gamma/2) the parallel Deltas agree to O(h³): 1e-4 on flat curves
  (`tests/test_greeks.py::TestAgainstTheBumpMethod`). Uncorrected, a European's differs by
  0.6%.
- *Curve representation.* On a sloped curve the sensitivity market (log-linear in the
  discount factor between tenors) is not the market's curve (linear in the zero rate between
  pillars), so the two methods differentiate slightly different curves: a near-par swap's
  small discount Delta differed by 2%, a European's Vega by 0.9%.
- *Bermudans/Americans.* AD Delta and Gamma hold the calibrated volatility fixed; the bump
  method recalibrates under each bump. AD Vega moves the volatility through the calibration
  (below), as the bump method's recalibration does.

Showing the two agree as the bump halves, beyond flat curves, is
[F-01](../planning/features.md#f-01).

## Vega (Bermudan/American only)

(Section title kept for links; a European on Bachelier has Vega too.) A European's Vega is
`dNPV/dσ` times the weight of each quote in the volatility it reads: the surface is bilinear
in its quotes, so `SwaptionVolSurface.weights` gives `dσ/dquote` exactly.

A Bermudan's/American's volatility comes from its calibration: the bootstrap fits bucket j's
σ_j so that the model reprices helper j at its market volatility v_j, with ζ_j = Σ_{k≤j} σ_k²
dt_k. A quote moves the helpers' volatilities (by the same bilinear weights), which move every
later bucket. By the implicit function theorem, row by row,

```
J[j] = −(Σ_{k<j} ∂g_j/∂σ_k · J[k] + e_j ∂g_j/∂v_j) / (∂g_j/∂σ_j),     g_j = model_j(ζ_j) − market_j(v_j)
```

(`_bootstrap_jacobian`; `∂g_j/∂v_j` includes the model's side, since a deal strike beyond 3
ATM standard deviations is clipped there and moves with its volatility), and
`Vega = dNPV/dσ · J · weights · vol_shift`. Checked against central differences of the full
recalibrating pricing to 1e-4 (`tests/test_greeks_bermudan.py`).

## Why no Vega for swaps or European swaptions

A swap reads no volatility. A European on the Jamshidian engine reads its configured
Hull-White model, not the market's quotes, so it has no Vega either; on Bachelier it has.

## Theta: advance the evaluation date, hold the market fixed

Both methods report ORE's Theta (`trade_theta`): the trade repriced on the market rebuilt at
`thetaDate = asof + theta_days` calendar days ([I-38](../planning/known-issues.md#i-38)) —
each curve's tenor points take the original curve's discount factor at `thetaDate + tenor`,
fixed in dates and not renormalized; swaption volatilities are today's surface seen from the
Theta date; fixings between the dates backfilled with the index's forecast — plus the
cashflows paid in `(asof, thetaDate]` ([I-39](../planning/known-issues.md#i-39)). A bond
maturing on the Theta date is worth 0 there and its redemption is a paid flow
([I-70](../planning/known-issues.md#i-70)). On a flat curve, curves fixed in dates mean a
bill's Theta is 0, not a pull to par.

## Differentiating through bisection root-finds

Two prices depend on a root found by bisection, which has no derivative: the Jamshidian
engine's critical state x*, and each calibration bucket's σ. Both get the implicit function
theorem's derivative instead: the Jamshidian root through a `jax.custom_jvp`
(`_solve_decreasing_root`: `dx = −(∂g/∂p · dp) / (∂g/∂x)`, itself differentiable, so Gamma is
right too), the calibration through `_bootstrap_jacobian` above. Before these, a bisection's
gradient was silently zero.

## The functions

| Function | Returns |
|---|---|
| `engine.risk.sensitivities.portfolio_sensitivities(trades, market, base_currency, pricing, config)` | Bump Greeks per trade |
| `engine.risk.greeks.portfolio_greeks(trades, market, base_currency, pricing, config)` | AD Greeks per trade |
| `engine.risk.greeks.curve_greeks(cfg, market, pricing, shift)` | AD Delta/Gamma per pillar of each curve a trade reads, in its currency |
| `engine.risk.greeks.vega_greek(cfg, market, pricing, shift)` | AD Vega per quote, or None |
| `engine.risk.price_functions.trade_price_function(cfg, market, pricing, dtype)` | `TradePriceFunction(curves, pricer, terms, times)`: the trade's price as a JAX function of its curves' pillar rates, `.price(*rates)` (shared with market risk); `pricer(terms, *curves)` is the same function as data, a module-level pricer and the trade's terms as a pytree, which the AD Greeks jit once per product and shape |

## Tested by

- `tests/test_greeks.py` — AD Delta per market pillar against central differences of the
  engine's own price (swaps, Europeans on both engines, bonds, 1e-6), Gamma against
  differences of the AD Delta, Vega per quote, agreement with the bump method (keys, Theta,
  parallel Deltas to O(h³), Vega), EUR trades in USD, and the Jamshidian root's derivatives.
- `tests/test_greeks_bermudan.py` — a Bermudan's Delta/Gamma with the volatility held, Vega
  through the recalibration, an American.
- `tests/test_sensitivities.py`, `tests/test_trade_dates.py` — the bump method and Theta
  against ORE's definitions (calendar-day roll, paid flows, seasoned trades).
- `tests/test_portfolio_gap_fixes.py`, `tests/test_portfolio_bond_wire_through.py` — every
  trade type gets its Greeks through `price_portfolio` (I-01, I-02, I-26).
- `tests/test_profiling_and_jit.py` — the Hessian-diagonal route equals the full Hessian,
  gradients survive the jit boundary, compile counts.
