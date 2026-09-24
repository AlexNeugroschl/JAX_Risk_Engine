"""
End-to-end walkthrough: calibrate -> simulate exposure -> Greeks -> market risk.

Portfolio: one swap, one European swaption, one Bermudan swaption, one
American swaption -- one of each rate-derivative type this engine prices.

The engine produces two different risk measures, and this demo shows both:

  * an EXPOSURE PROFILE (EPE/ENE/PFE through time) from the multi-step
    risk-neutral simulation, `price_portfolio`;
  * short-horizon MARKET RISK (10-day VaR/ES) by revaluing the portfolio at
    t=0 under shocked curves, `run_market_risk`.

This demo used to hand-orchestrate every stage (build a swap first to get
real maturity pillars, call generate_paths, calibrate a Sigma, call each
pricer separately, sum base NPVs by hand, then call compute_risk_metrics)
-- ~200 lines duplicating exactly what engine.portfolio.price_portfolio now
does as a single call. That duplication would immediately drift out of sync
with price_portfolio's own logic, exactly the failure mode
engine/simulation/demo_scenarios.py was created to avoid for shared example
configs -- see docs/reference/portfolio-entrypoint.md for what happens
"under the hood" of the one call below.

Run with: .venv/Scripts/python.exe demos/demo.py
"""
import dataclasses

import numpy as np
import ORE

from engine.simulation.market_model import EquityConfig, RatesConfig, SimulationConfig, ZeroCurveConfig
from engine.simulation.demo_scenarios import EVAL_DATE

from engine.instruments.swap import SwapConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.american_swaption import AmericanSwaptionConfig

from engine.calibration.basket import build_coterminal_basket
from engine.calibration.lgm import calibrate_lgm_sigma
from engine.models.hull_white import ZeroCurve

from engine.market_risk import MarketRiskRequest, RateRiskFactors, monte_carlo_scenarios, run_market_risk
from engine.portfolio import PortfolioRequest, price_portfolio


def section(title: str) -> None:
    print(f"\n--- {title} ---")


# =============================================================================
# Today's market: a flat 3% curve, one rate factor.
# =============================================================================
section("Market data")

TODAY = EVAL_DATE
ORE.Settings.instance().evaluationDate = TODAY
FLAT_RATE = 0.03
HW_A = 0.03      # mean-reversion speed
HW_SIGMA = 0.01  # flat short-rate vol, used until calibration produces a real one

zero_curve_config = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[FLAT_RATE] * 6)
zero_curve_jax = ZeroCurve.flat(FLAT_RATE, zero_curve_config.times)

print(f"flat rate {FLAT_RATE:.2%}")


# =============================================================================
# Calibrate a volatility term structure to market swaption quotes.
# =============================================================================
section("Calibration")

CALIBRATION_EXERCISE_TIMES = [1.0, 2.0, 3.0, 4.0]
CALIBRATION_FINAL_MATURITY = 5.0
MARKET_NORMAL_VOLS = [0.0080, 0.0088, 0.0095, 0.0100]

basket = build_coterminal_basket(
    exercise_times=CALIBRATION_EXERCISE_TIMES,
    final_maturity_time=CALIBRATION_FINAL_MATURITY,
    notional=1_000_000.0, payer=True, market_vols=MARKET_NORMAL_VOLS,
    zero_curve=zero_curve_jax, evaluation_date=TODAY,
)
calibration = calibrate_lgm_sigma(basket, zero_curve_jax, a=HW_A)
CALIBRATED_SIGMA = calibration.sigma

print(f"calibrated sigma per bucket: {np.round(np.asarray(calibration.sigma.values), 5)}")
print(f"reprice RMSE (should be ~0): {calibration.rmse:.2e}")


# =============================================================================
# Describe the portfolio: one of each instrument type.
# =============================================================================
section("Portfolio")

swap_cfg = SwapConfig(
    notional=2_000_000.0, fixed_rate=0.032, payer=True,
    discount_curve_index=0, forward_curve_index=0,
    swap_tenor="3Y", evaluation_date=TODAY,
)
european_cfg = SwaptionConfig(
    notional=1_500_000.0, fixed_rate=0.031, payer=True, rate_factor_index=0,
    hw_a=HW_A, hw_sigma=HW_SIGMA, initial_zero_curve=zero_curve_config,
    swap_tenor="3Y", forward_start=ORE.Period(2, ORE.Years), evaluation_date=TODAY,
)
bermudan_cfg = BermudanSwaptionConfig(
    notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
    hw_a=HW_A, hw_sigma=CALIBRATED_SIGMA, initial_zero_curve=zero_curve_config,
    # Exercisable on the 1Y..4Y anniversaries -- the calibration basket's own expiries.
    exercise_dates=[TODAY + ORE.Period(years, ORE.Years) for years in (1, 2, 3, 4)], swap_tenor="5Y",
    evaluation_date=TODAY, n_per_std=64, std_devs=6.0,
)
american_cfg = AmericanSwaptionConfig(
    notional=800_000.0, fixed_rate=0.029, payer=False, rate_factor_index=0,
    hw_a=HW_A, hw_sigma=CALIBRATED_SIGMA, initial_zero_curve=zero_curve_config,
    first_exercise_date=TODAY + ORE.Period(1, ORE.Years), last_exercise_date=TODAY + ORE.Period(4, ORE.Years),
    exercise_time_steps_per_year=2,
    evaluation_date=TODAY, n_per_std=64, std_devs=6.0,
)

NUM_SCENARIOS = 4096
TIME_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

sim_config = SimulationConfig(
    time_grid=TIME_GRID,
    scenarios=NUM_SCENARIOS,
    equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[0.0]]),
    rates=RatesConfig(
        initial_rates=[FLAT_RATE], theta=[FLAT_RATE], mean_reversion=[HW_A],
        initial_zero_curves=[zero_curve_config],
        # maturities left unset -- price_portfolio derives the swap's real
        # cashflow-pillar set automatically (engine.portfolio.
        # derive_maturity_pillars), rather than this demo hand-computing
        # them the way it used to.
    ),
    joint_covariance=[[0.04, 0.0], [0.0, HW_SIGMA ** 2]],
)

print(f"{NUM_SCENARIOS:,} scenarios requested, one swap / European / Bermudan / American swaption")


# =============================================================================
# Price the whole portfolio in one call: simulate -> validate -> price ->
# aggregate risk -- everything engine.portfolio.price_portfolio does under
# the hood is documented in docs/reference/portfolio-entrypoint.md.
# =============================================================================
section("Pricing the whole portfolio")

request = PortfolioRequest(
    market=sim_config,
    trades=[swap_cfg, european_cfg, bermudan_cfg, american_cfg],
    pfe_quantiles=(0.95, 0.99),
    compute_greeks=True,
)
result = price_portfolio(request)

TRADE_NAMES = ["swap", "european", "bermudan", "american"]
print("mean NPV across scenarios, at each simulated time step:")
print("  time   " + "".join(f"{n:>12}" for n in TRADE_NAMES))
for i, t in enumerate(TIME_GRID[1:]):
    row = "".join(f"{float(result.npv_cube[:, i, j].mean()):>12,.0f}" for j in range(len(TRADE_NAMES)))
    print(f"  {t:>4.2f}  " + row)

print(f"\nbaseline portfolio NPV: {result.base_npv:,.2f}")
if result.warnings:
    print(f"warnings: {result.warnings}")


# =============================================================================
# Exposure: what the simulated cube measures (computed inside price_portfolio).
# =============================================================================
section("Exposure profile (the whole portfolio as one netting set)")

exposure = result.exposure
columns = {"EPE": exposure.epe, "ENE": exposure.ene, "EE_B": exposure.ee_b, **exposure.pfe}
print("  time  " + "".join(f"{name:>12}" for name in columns))
for i, t in enumerate(exposure.times):
    print(f"  {t:>4.2f}" + "".join(f"{float(values[i]):>12,.0f}" for values in columns.values()))
print("(EPE/ENE/PFE discounted to today, EE_B undiscounted -- ORE's ExposureCalculator definitions)")


# =============================================================================
# Greeks on the Bermudan trade -- sensitivity to a 1bp market move.
# =============================================================================
section("Greeks")

bermudan_index = TRADE_NAMES.index("bermudan")
bermudan_greeks = result.greeks[bermudan_index]
print(f"delta per pillar: {np.round(np.asarray(bermudan_greeks['delta']), 2)}")
print(f"gamma per pillar: {np.round(np.asarray(bermudan_greeks['gamma']), 4)}")
print(f"theta (1-day decay): {bermudan_greeks['theta']:,.2f}")


# =============================================================================
# Market risk: 10-day VaR and ES by full revaluation at t=0.
# =============================================================================
section("Market risk (10-day, Monte Carlo)")

# The risk factors are the curve's pillar zero rates. Their 10-day moves are
# drawn from a Gaussian: 8bp daily vol per pillar, correlation decaying with
# pillar distance. A real run would estimate this from history
# (engine.market_risk.covariance_from_history) or use historical_scenarios.
factors = RateRiskFactors.from_curves([zero_curve_config], names=["USD"])
pillar_index = np.arange(factors.size)
correlation = np.exp(-np.abs(pillar_index[:, None] - pillar_index[None, :]) / 3.0)
covariance = correlation * 0.0008 ** 2 * 10
scenarios = monte_carlo_scenarios(factors, covariance, horizon_days=10, num_scenarios=4096, seed=1)

# Every scenario is a full revaluation. The Bermudan and American were priced
# above on a fine exercise grid (n_per_std=64); for 4,096 revaluations they
# use a risk-sized grid instead, and the base values show what that costs.
risk_grid = dict(n_per_std=16, std_devs=5.0)
risk_trades = [swap_cfg, european_cfg,
               dataclasses.replace(bermudan_cfg, **risk_grid), dataclasses.replace(american_cfg, **risk_grid)]
market_risk = run_market_risk(MarketRiskRequest(
    trades=risk_trades, scenarios=scenarios, quantiles=(0.99, 0.975),
))
for name, fine, coarse in zip(TRADE_NAMES[2:], result.base_npv_per_trade[2:], market_risk.base_npv_per_trade[2:]):
    print(f"  {name} base value: fine grid {fine:,.2f}, risk grid {coarse:,.2f} ({coarse / fine - 1:+.2e})")
print(f"{market_risk.num_scenarios:,} scenarios over {market_risk.horizon_days} days "
      f"({market_risk.source}, measure: {market_risk.measure})")
for q in ("99", "97.5"):
    print(f"  VaR {q:>4}%: {market_risk.risk[f'VaR_{q}']:>12,.0f}    "
          f"ES {q:>4}%: {market_risk.risk[f'ES_{q}']:>12,.0f}  "
          f"(+/- {market_risk.risk[f'ES_{q}_standardError']:,.0f}, "
          f"{int(market_risk.risk[f'ES_{q}_tailCount'])} tail scenarios)")
for message in market_risk.warnings:
    print(f"  note: {message}")
