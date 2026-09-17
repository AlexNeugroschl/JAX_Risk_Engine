"""
End-to-end walkthrough: simulate -> calibrate -> price -> aggregate risk.

Portfolio: one swap, one European swaption, one Bermudan swaption, one
American swaption -- one of each instrument type this engine prices.

This demo used to hand-orchestrate every stage (build a swap first to get
real maturity pillars, call generate_paths, calibrate a Sigma, call each
pricer separately, sum base NPVs by hand, then call compute_risk_metrics)
-- ~200 lines duplicating exactly what engine.portfolio.price_portfolio now
does as a single call. That duplication would immediately drift out of sync
with price_portfolio's own logic, exactly the failure mode
engine/simulation/demo_scenarios.py was created to avoid for shared example
configs -- see docs/reference/portfolio-entrypoint.md for what happens
"under the hood" of the one call below.

Run with: .venv/Scripts/python.exe demo.py
"""
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
    exercise_times=CALIBRATION_EXERCISE_TIMES, swap_tenor="5Y",
    evaluation_date=TODAY, n_per_std=64, std_devs=6.0,
)
american_cfg = AmericanSwaptionConfig(
    notional=800_000.0, fixed_rate=0.029, payer=False, rate_factor_index=0,
    hw_a=HW_A, hw_sigma=CALIBRATED_SIGMA, initial_zero_curve=zero_curve_config,
    first_exercise=1.0, last_exercise=4.0, exercise_time_steps_per_year=2,
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
    percentiles=(0.95, 0.99),
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
# Risk aggregation: VaR and Expected Shortfall (computed inside price_portfolio).
# =============================================================================
section("Risk aggregation")

print("  time   " + "".join(f"{m:>12}" for m in result.risk))
for i, t in enumerate(TIME_GRID[1:]):
    row = "".join(
        f"{float(result.risk[m][i]):>12,.0f}" if not np.isnan(float(result.risk[m][i])) else f"{'nan':>12}"
        for m in result.risk
    )
    print(f"  {t:>4.2f}  " + row)
print("(nan = the loss tail was empty at that step, matching ORE's own edge case)")


# =============================================================================
# Greeks on the Bermudan trade -- sensitivity to a 1bp market move.
# =============================================================================
section("Greeks")

bermudan_index = TRADE_NAMES.index("bermudan")
bermudan_greeks = result.greeks[bermudan_index]
print(f"delta per pillar: {np.round(np.asarray(bermudan_greeks['delta']), 2)}")
print(f"gamma per pillar: {np.round(np.asarray(bermudan_greeks['gamma']), 4)}")
print(f"theta (1-day decay): {bermudan_greeks['theta']:,.2f}")
