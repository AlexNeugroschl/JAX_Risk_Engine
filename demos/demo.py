"""
End-to-end walkthrough: today's market -> calibrate and simulate -> exposure -> Greeks ->
market risk.

Portfolio: one swap, one European swaption, one Bermudan swaption, one American swaption and
one Treasury note -- one of each trade type the engine prices -- on the demo USD market (a
curve rising from 3% to 5%, demos/demo_scenarios.py).

The run configuration names every choice (engine.portfolio.RunConfig): here the Hull-White
model for USD, calibrated to the market's swaption volatilities, ORE's default engines, and
Greeks by automatic differentiation (ORE's bump-and-revalue is the default; for the options
it recalibrates under every bump, which takes minutes here, I-53). Switching the model is one
field; the last exposure table shows the LGM beside it.

The engine produces two different risk measures, and this demo shows both:

  * an EXPOSURE PROFILE (EPE/ENE/PFE through time) from the multi-step risk-neutral
    simulation, `price_portfolio`;
  * short-horizon MARKET RISK (10-day VaR/ES) by revaluing the portfolio at t=0 under shocked
    curves, `run_market_risk`.

docs/reference/portfolio-entrypoint.md describes what `price_portfolio` does internally.

Run with: .venv/Scripts/python.exe demos/demo.py
"""
import dataclasses

import numpy as np
import ORE

from demo_scenarios import EVAL_DATE, demo_market, demo_simulation
from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig
from engine.instruments.treasury import BondConfig, CouponPeriod
from engine.market_risk import MarketRiskRequest, RateRiskFactors, monte_carlo_scenarios, run_market_risk
from engine.portfolio import (
    GreeksConfig, LgmSwaptionEngineConfig, PortfolioRequest, PricingConfig, RunConfig, price_portfolio,
)
from engine.simulation.config import build_cross_asset_model


def section(title: str) -> None:
    print(f"\n--- {title} ---")


# =============================================================================
# Today's market and the run configuration.
# =============================================================================
section("Market and configuration")

TODAY = EVAL_DATE
market = demo_market(("USD",))
usd = market.currency("USD")
print(f"as of {TODAY.ISO()}: USD discount curve {usd.discount_curve.rates} at {usd.discount_curve.times}")

# The Hull-White model for USD, calibrated to a 1Y/2Y/5Y co-terminal basket of the market's
# swaption volatilities. 128 paths: every Bermudan/American is recalibrated on each path and
# date as ORE does, which is the slow part of a run (I-53).
simulation = demo_simulation("HullWhite", samples=128, currencies=("USD",), calibrated=True)
# A coarser Bermudan/American grid than the default (n_per_std=30), for a quick demo.
engine = LgmSwaptionEngineConfig(n_per_std=16)
config = RunConfig(simulation=simulation, pricing=PricingConfig(bermudan=engine, american=engine),
                   greeks=GreeksConfig(method="AD"))


# =============================================================================
# Calibration: what the model is fitted to.
# =============================================================================
section("Calibration")

model = build_cross_asset_model(market, simulation).ir[0]
print(f"Hull-White a = {model.reversion}, short-rate volatility per bucket "
      f"{np.round(np.asarray(model.sigma.values), 5)} (times {np.asarray(model.sigma.times).round(3)})")


# =============================================================================
# The portfolio: one of each trade type, each named by its id.
# =============================================================================
section("Portfolio")

note_periods = tuple(CouponPeriod(ORE.Date(15, m, y), ORE.Date(15, 8 if m == 2 else 2, y if m == 2 else y + 1))
                     for y in (2026, 2027, 2028) for m in (2, 8))
trades = [
    SwapConfig(notional=2_000_000.0, fixed_rate=0.036, payer=True, swap_tenor="3Y", evaluation_date=TODAY,
               trade_id="swap"),
    SwaptionConfig(notional=1_500_000.0, fixed_rate=0.042, payer=True, swap_tenor="3Y",
                   forward_start=ORE.Period(2, ORE.Years), evaluation_date=TODAY, trade_id="european"),
    BermudanSwaptionConfig(notional=1_000_000.0, fixed_rate=0.042, payer=True, swap_tenor="5Y",
                           exercise_dates=[TODAY + ORE.Period(y, ORE.Years) for y in (1, 2, 3, 4)],
                           evaluation_date=TODAY, trade_id="bermudan"),
    AmericanSwaptionConfig(notional=800_000.0, fixed_rate=0.040, payer=False, swap_tenor="5Y",
                           first_exercise_date=TODAY + ORE.Period(1, ORE.Years),
                           last_exercise_date=TODAY + ORE.Period(4, ORE.Years), evaluation_date=TODAY,
                           trade_id="american"),
    BondConfig(face_amount=1_000_000.0, maturity_date=ORE.Date(15, 2, 2029), coupon_rate=0.0375,
               coupon_schedule=note_periods, evaluation_date=TODAY, trade_id="note"),
]
print(", ".join(t.trade_id for t in trades))


# =============================================================================
# Price the whole portfolio in one call.
# =============================================================================
section("Pricing the whole portfolio")

result = price_portfolio(PortfolioRequest(market=market, trades=trades, config=config, pfe_quantiles=(0.95, 0.99),
                                          compute_greeks=True))
times = np.asarray(result.exposure.times)     # t=0, then the simulation dates (the cube's axis)
print("mean NPV across paths, at each simulation date:")
print("  time  " + "".join(f"{n:>12}" for n in result.trade_ids))
print("  0.00  " + "".join(f"{v:>12,.0f}" for v in result.base_npv_per_trade))
for i, t in enumerate(times[1:]):
    print(f"  {t:>4.2f}  " + "".join(f"{float(result.npv_cube[:, i, j].mean()):>12,.0f}"
                                    for j in range(len(trades))))
print(f"\nportfolio NPV today: {result.base_npv:,.2f}")
print("(a matured swap is worth 0; an exercised physical option carries the swap it entered)")


# =============================================================================
# Exposure: what the simulated cube measures.
# =============================================================================
section("Exposure profile (the whole portfolio as one netting set)")

exposure = result.exposure
columns = {"EPE": exposure.epe, "ENE": exposure.ene, "EE_B": exposure.ee_b, **exposure.pfe}
print("  time  " + "".join(f"{name:>12}" for name in columns))
for i, t in enumerate(times):
    print(f"  {t:>4.2f}" + "".join(f"{float(values[i]):>12,.0f}" for values in columns.values()))
print(f"Basel EPE {exposure.basel_epe:,.0f}, EEPE {exposure.basel_eepe:,.0f} "
      f"(ORE's ExposureCalculator definitions; measure: {result.measure})")


# =============================================================================
# Greeks on the Bermudan, by AD: per market-curve pillar, Vega per swaption quote through the
# calibration.
# =============================================================================
section("Greeks (Bermudan, automatic differentiation)")

greeks = result.greeks[result.trade_ids.index("bermudan")]
for key, value in greeks.items():
    value = np.asarray(value)
    print(f"  {key}: {round(float(value), 2) if value.ndim == 0 else np.round(value, 2).tolist()}")


# =============================================================================
# The model is one field: the same run under the LGM.
# =============================================================================
section("Exposure under each model (EPE)")

lgm = dataclasses.replace(config, simulation=demo_simulation("LGM", samples=128, currencies=("USD",), calibrated=True))
lgm_result = price_portfolio(PortfolioRequest(market=market, trades=trades, config=lgm))
print("  time   Hull-White         LGM")
for i, t in enumerate(times):
    print(f"  {t:>4.2f}  {float(exposure.epe[i]):>10,.0f}  {float(lgm_result.exposure.epe[i]):>10,.0f}")
print("(the same t=0 values; the calibrated models differ in their dynamics beyond the basket)")


# =============================================================================
# Market risk: 10-day VaR and ES by full revaluation at t=0.
# =============================================================================
section("Market risk (10-day, Monte Carlo)")

# The risk factors are the pillar zero rates of the market's curves (USD discount and its 6M
# index), named as the trades read them. Their 10-day moves are drawn from a Gaussian: 8bp
# daily vol per pillar, correlation decaying with pillar distance. A real run would estimate
# this from history (engine.market_risk.covariance_from_history) or use historical_scenarios.
factors = RateRiskFactors.from_market(market)
pillar_index = np.arange(factors.size) % len(usd.discount_curve.times)
correlation = np.exp(-np.abs(pillar_index[:, None] - pillar_index[None, :]) / 3.0)
covariance = correlation * 0.0008 ** 2 * 10
scenarios = monte_carlo_scenarios(factors, covariance, horizon_days=10, num_scenarios=4096, seed=1)
market_risk = run_market_risk(MarketRiskRequest(trades=trades, market=market, scenarios=scenarios,
                                                pricing=config.pricing, quantiles=(0.99, 0.975)))
print(f"{market_risk.num_scenarios:,} scenarios over {market_risk.horizon_days} days "
      f"({market_risk.source}, measure: {market_risk.measure})")
for q in ("99", "97.5"):
    print(f"  VaR {q:>4}%: {market_risk.risk[f'VaR_{q}']:>12,.0f}    "
          f"ES {q:>4}%: {market_risk.risk[f'ES_{q}']:>12,.0f}  "
          f"(+/- {market_risk.risk[f'ES_{q}_standardError']:,.0f}, "
          f"{int(market_risk.risk[f'ES_{q}_tailCount'])} tail scenarios)")
for message in market_risk.warnings:
    print(f"  note: {message}")
