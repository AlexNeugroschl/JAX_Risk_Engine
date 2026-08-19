"""
End-to-end walkthrough: simulate -> calibrate -> price -> aggregate risk.

Portfolio: one swap, one European swaption, one Bermudan swaption, one
American swaption -- one of each instrument type this engine prices.

Run with: venv/Scripts/python.exe demo.py
"""
import jax.numpy as jnp
import numpy as np
import ORE

from engine.simulation.market_model import (
    EquityConfig, RatesConfig, SimulationConfig, ZeroCurveConfig, generate_paths,
)
from engine.simulation.demo_scenarios import EVAL_DATE, flat_yield_curves

from engine.instruments.swap import SwapConfig, price_swaps, _build_ore_swap
from engine.instruments.european_swaption import SwaptionConfig, price_swaptions
from engine.instruments.bermudan_swaption import (
    BermudanSwaptionConfig, price_bermudan_swaptions, price_bermudan_swaption_base,
)
from engine.instruments.american_swaption import AmericanSwaptionConfig, price_american_swaptions
from engine.trades.ore_builders import DAY_COUNTER

from engine.models.hull_white import ZeroCurve
from engine.calibration.basket import build_coterminal_basket
from engine.calibration.lgm import calibrate_lgm_sigma

from engine.risk.var_es import compute_risk_metrics
from engine.risk.greeks import bermudan_delta_gamma, bermudan_theta, bermudan_vega


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
HW_SIGMA = 0.01  # flat short-rate vol, used until Step 2 calibrates a real one

# The swap pricer requires every cashflow date to land exactly on a
# simulation pillar, and real ORE dates aren't round numbers (spot lag,
# weekends) -- so build the swap first and use its own dates as pillars.
swap_cfg = SwapConfig(
    notional=2_000_000.0, fixed_rate=0.032, payer=True,
    discount_curve_index=0, forward_curve_index=0,
    swap_tenor="3Y", evaluation_date=TODAY,
)
_swap = _build_ore_swap(swap_cfg)
_pillars = {0.0}
for cf in _swap.fixedLeg():
    c = ORE.as_fixed_rate_coupon(cf)
    _pillars.update({DAY_COUNTER.yearFraction(TODAY, c.date()), DAY_COUNTER.yearFraction(TODAY, c.accrualStartDate())})
for cf in _swap.floatingLeg():
    c = ORE.as_floating_rate_coupon(cf)
    _pillars.update({
        DAY_COUNTER.yearFraction(TODAY, c.date()),
        DAY_COUNTER.yearFraction(TODAY, c.accrualStartDate()),
        DAY_COUNTER.yearFraction(TODAY, c.accrualEndDate()),
    })
PILLAR_TIMES = sorted(_pillars)

zero_curve_config = ZeroCurveConfig(times=PILLAR_TIMES, rates=[FLAT_RATE] * len(PILLAR_TIMES))
zero_curve_jax = ZeroCurve.flat(FLAT_RATE, PILLAR_TIMES)

print(f"flat rate {FLAT_RATE:.2%}, {len(PILLAR_TIMES)} curve pillars from the swap's real cashflow dates")


# =============================================================================
# Simulate the market: thousands of alternate future rate paths.
# =============================================================================
section("Market simulation")

NUM_SCENARIOS = 4096
TIME_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

sim_config = SimulationConfig(
    time_grid=TIME_GRID,
    scenarios=NUM_SCENARIOS,
    equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[0.0]]),
    rates=RatesConfig(
        initial_rates=[FLAT_RATE], theta=[FLAT_RATE], mean_reversion=[HW_A],
        maturities=PILLAR_TIMES, initial_zero_curves=[zero_curve_config],
    ),
    joint_covariance=[[0.04, 0.0], [0.0, HW_SIGMA ** 2]],
)
market = generate_paths(sim_config)

last_step = np.asarray(market["rates"][:, -1, 0])
print(f"{NUM_SCENARIOS:,} scenarios simulated; short rate at t={TIME_GRID[-1]}: "
      f"mean={last_step.mean():.2%}, std={last_step.std():.2%}")


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
# Price the portfolio: one of each instrument type.
# =============================================================================
section("Pricing")

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

step_times = jnp.array(TIME_GRID[1:], dtype=jnp.float64)

swap_cube = price_swaps(market["yield_curves"], PILLAR_TIMES, [swap_cfg])
european_cube = price_swaptions(market["rates"], step_times, [european_cfg])
bermudan_cube = price_bermudan_swaptions([bermudan_cfg], market["rates"], step_times)
american_cube = price_american_swaptions([american_cfg], market["rates"], step_times)

print("mean NPV across scenarios, at each simulated time step:")
print("  time   " + "".join(f"{n:>12}" for n in ["swap", "european", "bermudan", "american"]))
for i, t in enumerate(TIME_GRID[1:]):
    cubes = [swap_cube, european_cube, bermudan_cube, american_cube]
    print(f"  {t:>4.2f}  " + "".join(f"{float(jnp.mean(c[:, i, 0])):>12,.0f}" for c in cubes))


# =============================================================================
# Aggregate into portfolio risk: VaR and Expected Shortfall.
# =============================================================================
section("Risk aggregation")

portfolio_cube = jnp.concatenate([swap_cube, european_cube, bermudan_cube, american_cube], axis=-1)

# Baseline NPV: today's actual (zero-shock) value, not read off the cube.
base_curve = flat_yield_curves(disc_rate=FLAT_RATE, fwd_rate=FLAT_RATE, maturities=PILLAR_TIMES, eval_date=TODAY)
r0_path = jnp.array([[[FLAT_RATE]]])
base_npv = (
    float(price_swaps(base_curve, PILLAR_TIMES, [swap_cfg])[0, 0, 0])
    + float(price_swaptions(r0_path, jnp.array([0.0]), [european_cfg])[0, 0, 0])
    + price_bermudan_swaption_base(bermudan_cfg)
    + price_bermudan_swaption_base(american_cfg.to_bermudan())
)
print(f"baseline portfolio NPV: {base_npv:,.2f}")

risk = compute_risk_metrics(portfolio_cube, base_npv, percentiles=(0.95, 0.99))
print("  time   " + "".join(f"{m:>12}" for m in risk))
for i, t in enumerate(TIME_GRID[1:]):
    row = "".join(f"{float(risk[m][i]):>12,.0f}" if not np.isnan(float(risk[m][i])) else f"{'nan':>12}" for m in risk)
    print(f"  {t:>4.2f}  " + row)
print("(nan = the loss tail was empty at that step, matching ORE's own edge case)")


# =============================================================================
# Greeks on the Bermudan trade -- sensitivity to a 1bp market move.
# =============================================================================
section("Greeks")

greeks = bermudan_delta_gamma(bermudan_cfg, zero_curve_jax)
theta = bermudan_theta(bermudan_cfg, zero_curve_jax)
vega = bermudan_vega(bermudan_cfg, zero_curve_jax, basket)

print(f"delta per pillar: {np.round(np.asarray(greeks['delta']), 2)}")
print(f"gamma per pillar: {np.round(np.asarray(greeks['gamma']), 4)}")
print(f"theta (1-day decay): {theta:,.2f}")
print(f"vega per calibration bucket: {np.round(np.asarray(vega), 2)}")
