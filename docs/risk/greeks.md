# Risk: Delta, Gamma, Vega, and Theta

Two methods, chosen by the run configuration's `greeks` (decision A-5,
[compliance/decisions.md](../../compliance/decisions.md)), for every trade type and either
simulation model, with the same keys on the same factors:

```python
config = RunConfig(greeks=GreeksConfig(method="Bump",            # or "AD"
                                       sensitivity=SensitivityConfig(curve_tenors=("1Y", "5Y", "10Y"),
                                                                     theta_days=3)))
greeks = price_portfolio(PortfolioRequest(market=market, trades=trades, config=config, scenario_risk=False,
                                          compute_greeks=True)).greeks
```

**Modules:** [`engine/risk/greeks/bump.py`](../../engine/risk/greeks/bump.py) (`Bump`,
`portfolio_sensitivities`, and Theta for both), [`engine/risk/greeks/ad.py`](../../engine/risk/greeks/ad.py)
(`AD`, `portfolio_greeks`, `trade_greeks`), [`engine/risk/greeks/price_functions.py`](../../engine/risk/greeks/price_functions.py)
(each trade's price as a JAX function of its market: its curves' node values and its volatility
quotes)

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

Per trade (by request index), in the reporting currency, every curve on ORE's sensitivity
market (each curve sampled at `SensitivityConfig.curve_tenors`, log-linear in the discount
factor, flat forward beyond):

| Key | Bump (ORE's sensitivity analysis) | AD |
|---|---|---|
| `delta:discount:<ccy>`, `delta:index:<name>` `[tenors]` | `NPV(up) − NPV(base)`, an absolute zero-rate shift of `curve_shift` at one tenor (`ShiftScheme::Forward`) | `dNPV/dz_k · curve_shift` in the tenor's zero rate |
| `gamma:discount:<ccy>`, `gamma:index:<name>` `[tenors]` | `NPV(up) − 2·NPV(base) + NPV(down)` | `d²NPV/dz_k² · curve_shift²` (the diagonal, as ORE computes no cross-gammas by default) |
| `vega:<ccy>` `[option tenors, swap tenors]` | `NPV(up) − NPV(base)` per swaption quote, an absolute normal-vol shift of `vol_shift` | `dNPV/dquote · vol_shift` |
| `theta` | `NPV(thetaDate) − NPV(base)`, `thetaDate = asof + theta_days` calendar days | the same function |

Vega is present for a trade whose engine reads the swaption volatilities: a European on
Bachelier, a Bermudan/American calibrated to them. A swap, a bond, a Jamshidian European and
an uncalibrated option have none (omitted, not zero). A bond reads its discount curve only.

## Scope: every rate-derivative instrument in this codebase

Swaps, Europeans (either engine), Bermudans, Americans and bonds, with either simulation
model (the Greeks are of today's value; the model only simulates).

## Bump: ORE's sensitivity analysis

`portfolio_sensitivities` reproduces ORE's `SensitivityAnalysis` (`sensitivityanalysis.cpp`,
`sensitivitycube.cpp`): each trade is priced with its t=0 engine on ORE's sensitivity
simulation market, with one tenor's zero rate (or one volatility quote) shifted at a time. A
Bermudan/American is recalibrated under every shift, as ORE does (`recalibrateModels`, its
default). **It equals ORE's own sensitivity analytic**, run in-process on the shared
portfolio's sloped two-curve market (`tests/test_sensitivity_ore_parity.py`, the oracle
`tests/support/ore_sensitivity_oracle.py`; I-51, 2026-10-17): every Delta, Gamma, Vega and
Theta of every trade, calibrated Bermudans and the American included, within 2e-14 of the
trade's NPV.

One engine curve is two of ORE's factors: ORE's LGM builder takes its model's and its helpers'
discount curve from the swap index's discounting index, and a bond discounts on its reference
curve, each an index curve of its own in ORE's sensitivity market beside the currency's discount
curve. The engine has one discount curve per currency, which the option's calibration and the
bond read too, so its discount Delta is ORE's two factors moved together (the oracle reads
ORE's cross gammas to put them together exactly).

## AD: ORE's Greek in the limit of a vanishing shift

ORE computes AD sensitivities in its sensitivity analysis itself (`UseAD` with a
`SensitivityDelta` run: the engine's gradient in its market's parameters, taken once on the
sensitivity market, each shift scenario's NPV then the base plus the gradient times the move).
The engine's AD Greeks are the same thing: each trade's price is a pure JAX function of its
market on ORE's sensitivity market (`trade_price_function(..., tenors=...)`: discounting legs,
Bachelier or Jamshidian, the grid engine, discounted bond flows; its base value is the bump
method's base), differentiated in what ORE shifts, each tenor's zero rate and each volatility
quote. So ORE's forward-difference Delta is `Delta + Gamma/2 + O(h³)` of the AD Greeks and its
Gamma `Gamma + O(h⁴)`.

Measured on the shared portfolio's sloped market (2026-10-17, I-78): for swaps, Europeans and
the bond, ORE's Delta less half its Gamma is the AD Delta within 2e-5 of each curve's largest
Delta, and the Gammas agree within 4e-6 (against ORE itself, `tests/test_sensitivity_ore_parity.py`,
and against the bump method, `tests/test_greeks.py`); ORE's Vega, a forward difference, is
`Vega + O(h)`, and extrapolated from two shifts (Richardson) it is the AD Vega to 1e-5.

**A calibrated Bermudan/American calibrates inside its price function**, so its Delta and Gamma
move the calibration with the curve, as ORE's recalibrating bumps do, and its Vega moves the
calibration with the quotes. The calibration's derivatives are the implicit function theorem's
on the whole basket (below).

**Where AD is not ORE's Greek: the grid engine.** A Bermudan's/American's price on the LGM grid
(ORE's own engine, to 1e-11) is piecewise smooth in the rates: its exercise switches on the state
grid's nodes, so the price has many small kinks. Its AD Greeks are the derivatives of that price
at the market, exactly (central differences at steps small enough to stay on one smooth piece,
1e-7, agree to 1e-9 for Delta); ORE's 1bp bump is a secant across the kinks. On the shared
portfolio the AD Delta was within 0.4% of each curve's largest bump Delta less half its Gamma,
and the bump Gamma, a second difference across kinks, is not a curvature (at one tenor it moved
from 1.9e8 to 3.9e9 as the step shrank from 1e-4 to 1e-6, against a smooth curvature of 8e6;
`tests/test_greeks_bermudan.py`). An AD run with such a trade says so in its warnings; the bump
method gives ORE's numbers.

**One program per product and shape.** A trade's Delta, Gamma and Vega are one compiled program
(`_derivatives`), shared by every trade of the same product and shape: forward differentiation
along every node value (a forward pass of a forward pass gives the first and second directional
derivative) and every quote, the directions batched with `vmap` and the price itself computed
once. Forward differentiation holds no tape: an option recalibrated under the shift moves its
LGM's grid and rollback operator with the curve, and reverse mode held their tangents on every
row of the induction for every direction (13 GB for an American on ORE's 301-node grid, against
2 GB forward). ([profiling §3.8](../concepts/profiling.md#38-the-ad-greeks-as-one-program-per-product-2026-10-06))

## Vega (Bermudan/American only)

(Section title kept for links; a European on Bachelier has Vega too.) A European's volatility is
read on the quotes through the surface's bilinear weights (`SwaptionVolSurface.weights`), so its
price is a function of the quotes and its Vega a derivative in them.

A Bermudan's/American's volatility comes from its calibration: the bootstrap fits bucket j's
σ_j so that the model reprices helper j at its market volatility v_j (read on the quotes the
same way), with ζ_j = Σ_{k≤j} σ_k² dt_k. Together the buckets solve the lower-triangular system
`G(σ; curves, quotes) = model(ζ(σ)) − market = 0` (`engine.calibration.lgm.basket_residuals`),
and by the implicit function theorem

```
dσ = −(∂G/∂σ)⁻¹ (∂G/∂(curves, quotes) · d(curves, quotes))
```

(`calibrated_values`, `engine.solvers.roots.implicit_solution`: ∂G/∂σ by forward
differentiation, the triangular system by forward substitution). The bootstrap's own root
finding is never differentiated; the rule is itself differentiable, so the second derivatives
(Gamma through the recalibration) are right too. Checked against central differences of the
full recalibrating price (`tests/test_greeks_bermudan.py`).

## Why no Vega for swaps or European swaptions

A swap reads no volatility. A European on the Jamshidian engine reads its configured
Hull-White model, not the market's quotes, so it has no Vega either; on Bachelier it has.

## Theta: advance the evaluation date, hold the market fixed

Both methods report ORE's Theta (`trade_theta`), equal to ORE's sensitivity analytic's: the
trade repriced on the sensitivity market rebuilt on `thetaDate = asof + theta_days` calendar
days ([I-38](../planning/known-issues.md#i-38)), less its value there. A flow paid in the roll
leaves Theta as it leaves the NPV: ORE adds back the flows paid in `(asof, thetaDate]`, but reads
them from the trade's cash flows on the Theta date without its past ones, where every flow of the
roll has occurred, so with ORE's settings it adds back nothing (until 2026-10-18 the engine added
them back, [I-39](../planning/known-issues.md#i-39), which ORE's analytic disproved,
[I-116](../planning/known-issues.md#i-116)); the owner kept ORE's definition (decision A-26,
2026-10-18). For a carry with the cash received, add the flows paid in the roll, which are to be
reported beside Theta as a figure of their own ([F-11](../planning/features.md#f-11)). ORE's
simulation-market curves hold their tenor
points at the tenors' times from the as-of date, with the as-of curve's discount factors there,
and take their reference date from the evaluation date (`InterpolatedDiscountCurve2`, the same
rule as on a simulated path, I-84): on the Theta date each curve is the sensitivity market's own
as a function of the time from its reference date, and the ATM volatility matrix, a simulated
factor, floats with it: its option dates roll from its reference date, the first business day on
or after the Theta date, where each option's variance starts (a Theta across a weekend reads the
volatilities from the Monday). Fixings between the dates
are backfilled with the index's forecast there. A bond maturing on the Theta date is worth 0
there ([I-70](../planning/known-issues.md#i-70)), its redemption having occurred, so its Theta is
minus its value. Until
2026-10-17 the engine's Theta market kept each curve's discount factors fixed in dates, off
ORE's by 60–290% of each trade's Theta (I-51).

<a id="differentiating-through-bisection-root-finds"></a>
## Differentiating through root-finds

Prices depend on roots found by the engine's root solver
([the root solver](../reference/calibration.md#the-root-solver)), whose iterations carry no
derivative of the root: the Jamshidian engine's critical state x*, each ORE helper's y*, and
the calibration's buckets. Each gets the implicit function theorem's derivative instead, itself
differentiable, so Gamma is right too: x* and y* elementwise (`engine.solvers.roots.implicit_root`:
`dx = −(∂g/∂p · dp) / (∂g/∂x)`, ∂g/∂x a forward derivative), the calibration as one system
(`implicit_solution`, above). The helper's price is stationary in its y*, so first derivatives
need none of y*'s; a calibration's second derivatives do.

## The functions

| Function | Returns |
|---|---|
| `engine.risk.greeks.bump.portfolio_sensitivities(trades, market, base_currency, pricing, config)` | Bump Greeks per trade |
| `engine.risk.greeks.ad.portfolio_greeks(trades, market, base_currency, pricing, config)` | AD Greeks per trade |
| `engine.risk.greeks.ad.trade_greeks(cfg, market, pricing, config)` | AD Delta/Gamma per tenor of each curve a trade reads and Vega per quote, in its currency |
| `engine.risk.greeks.ad.ad_warnings(trades)` | The label an AD run carries for its grid-priced options |
| `engine.risk.greeks.price_functions.trade_price_function(cfg, market, pricing, dtype, tenors)` | `TradePriceFunction`: the trade's price as a JAX function of its curves' node values (on ORE's sensitivity market at `tenors`, or on the market's pillars, market risk's) and its quotes, `.price(*rates, quotes=None)`; `pricer(terms, curves, quotes)` is the same function as data, a module-level pricer and the trade's terms as a pytree, which the AD Greeks jit once per product and shape |

## Tested by

- `tests/test_sensitivity_ore_parity.py` — the bump Greeks against ORE's sensitivity analytic,
  every trade, factor and tenor of the shared portfolio, Theta included; the AD Greeks of its
  smooth trades against ORE's as the shift vanishes.
- `tests/test_greeks.py` — the AD Greeks as derivatives of the price on the sensitivity market
  (central differences: swaps, Europeans on both engines, bonds), Vega per quote, and against the
  bump method on the sloped market per trade, factor and tenor (Delta and Gamma to O(h³), Vega by
  Richardson extrapolation), EUR trades in USD.
- `tests/test_greeks_bermudan.py` — a Bermudan's Delta, Gamma and Vega through its
  recalibration against central differences of the recalibrating price; its forward differences
  are the bump method's; the shared options within the grid's gap of ORE's bump Delta; an
  American.
- `tests/test_sensitivities.py`, `tests/test_trade_dates.py` — the bump method and Theta
  against ORE's definitions (calendar-day roll, paid flows, seasoned trades).
- `tests/test_portfolio_gap_fixes.py`, `tests/test_portfolio_bond_wire_through.py` — every
  trade type gets its Greeks through `price_portfolio` (I-01, I-02, I-26).
- `tests/test_profiling_and_jit.py` — the forward derivatives equal the gradient and the full
  Hessian's diagonal, gradients survive the jit boundary, compile counts.
