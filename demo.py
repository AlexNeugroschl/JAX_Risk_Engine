"""
End-to-end walkthrough of this engine, front to back, on a small but
genuinely diverse portfolio: a swap, a European swaption, a Bermudan
swaption, and an American swaption -- one of each instrument type this
codebase prices, so every different calculation style shows up at least
once.

Run it with:

    venv/Scripts/python.exe demo.py        (Windows, this repo's own venv)
    python demo.py                         (any environment with the
                                             dependencies from requirements.txt
                                             installed)

Nothing here is new machinery -- every function called below already
exists in engine/. This script's only job is to call them in pipeline
order (simulate -> calibrate -> price -> aggregate into risk) and print
the intermediate state at each step, so you can see what's actually
flowing through the pipeline instead of only the final numbers.
"""
import time

import jax.numpy as jnp
import numpy as np
import ORE

from engine.simulation.market_model import (
    EquityConfig,
    RatesConfig,
    SimulationConfig,
    ZeroCurveConfig,
    generate_paths,
)
from engine.simulation.demo_scenarios import EVAL_DATE

from engine.instruments.swap import SwapConfig, price_swaps, _build_ore_swap
from engine.instruments.european_swaption import SwaptionConfig, price_swaptions
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig, price_bermudan_swaptions
from engine.instruments.american_swaption import AmericanSwaptionConfig, price_american_swaptions
from engine.trades.ore_builders import DAY_COUNTER

from engine.models.hull_white import ZeroCurve
from engine.calibration.basket import build_coterminal_basket
from engine.calibration.lgm import calibrate_lgm_sigma

from engine.risk.var_es import compute_risk_metrics
from engine.risk.greeks import bermudan_delta_gamma, bermudan_theta, bermudan_vega


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# =============================================================================
# STEP 0 -- today's market: one flat 3% USD curve.
#
# Everything downstream -- the simulation's own starting point, every
# instrument's discounting, and the calibration basket -- reads off this
# SAME curve, exactly the way a real desk prices everything off one
# morning's market data snapshot.
# =============================================================================
section("STEP 0 -- Today's market data")

FLAT_RATE = 0.03
HW_A = 0.03       # Hull-White / LGM mean-reversion speed
HW_SIGMA = 0.01   # flat short-rate volatility (1% annualized) for the swap/European/simulation

TODAY = EVAL_DATE
ORE.Settings.instance().evaluationDate = TODAY

print(f"Evaluation date : {TODAY}")
print(f"Flat zero rate  : {FLAT_RATE:.2%}")
print(f"Mean reversion a: {HW_A}")
print(f"Short-rate vol  : {HW_SIGMA:.2%} (flat, before calibration)")


# =============================================================================
# Pin down the swap's own real cashflow dates BEFORE building the
# simulation config: the swap pricer requires every cashflow's year-
# fraction to land EXACTLY on one of the simulation's output maturity
# pillars (no curve interpolation there -- see engine/instruments/swap.py's
# own "Known limitation" docstring), and a swap's real dates (from ORE's
# own schedule generator, including weekend/holiday adjustment and the
# standard 2-day spot lag) do NOT fall on round numbers like exactly
# "1.0" or "2.0" years. Building the real ORE trade once, here, and
# reading its own dates off is the same approach
# engine/simulation/demo_scenarios.py's own SWAP_DEMO_MATURITIES uses.
# =============================================================================
swap_cfg = SwapConfig(
    notional=2_000_000.0, fixed_rate=0.032, payer=True,
    discount_curve_index=0, forward_curve_index=0,
    swap_tenor="3Y", evaluation_date=TODAY,
)
_real_swap = _build_ore_swap(swap_cfg)
_pillar_set = {0.0}
for cf in _real_swap.fixedLeg():
    c = ORE.as_fixed_rate_coupon(cf)
    _pillar_set.add(DAY_COUNTER.yearFraction(TODAY, c.date()))
    _pillar_set.add(DAY_COUNTER.yearFraction(TODAY, c.accrualStartDate()))
for cf in _real_swap.floatingLeg():
    c = ORE.as_floating_rate_coupon(cf)
    _pillar_set.add(DAY_COUNTER.yearFraction(TODAY, c.date()))
    _pillar_set.add(DAY_COUNTER.yearFraction(TODAY, c.accrualStartDate()))
    _pillar_set.add(DAY_COUNTER.yearFraction(TODAY, c.accrualEndDate()))
PILLAR_TIMES = sorted(_pillar_set)
zero_curve_config = ZeroCurveConfig(times=PILLAR_TIMES, rates=[FLAT_RATE] * len(PILLAR_TIMES))
zero_curve_jax = ZeroCurve.flat(FLAT_RATE, PILLAR_TIMES)

print(f"Curve pillars   : {[round(t, 4) for t in PILLAR_TIMES]}  (years from today)")
print("  (these are the swap's own real ORE cashflow dates, not round numbers --")
print("   every simulated discount-curve pillar must land exactly on a cashflow date)")


# =============================================================================
# STEP 1 -- MARKET SIMULATION
#
# One equity/FX factor is required by generate_paths even though nothing
# in this portfolio is equity-linked -- give it a zero drift-coupling so
# it's simulated but has no effect on anything else (the same "placeholder
# equity" pattern engine/simulation/demo_scenarios.py itself uses).
# =============================================================================
section("STEP 1 -- Market simulation (generate_paths)")

NUM_SCENARIOS = 4096
TIME_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

sim_config = SimulationConfig(
    time_grid=TIME_GRID,
    scenarios=NUM_SCENARIOS,
    equities=EquityConfig(
        initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[0.0]],
    ),
    rates=RatesConfig(
        initial_rates=[FLAT_RATE],
        theta=[FLAT_RATE],
        mean_reversion=[HW_A],
        maturities=PILLAR_TIMES,
        initial_zero_curves=[zero_curve_config],
    ),
    joint_covariance=[
        [0.04, 0.0],               # placeholder equity: 20% vol, uncorrelated with rates
        [0.0, HW_SIGMA ** 2],      # the one rate factor: 1% vol
    ],
)

print(f"Scenarios       : {NUM_SCENARIOS:,}")
print(f"Time grid       : {TIME_GRID}")
print("Running Sobol quasi-Monte Carlo -> Brownian bridge -> Hull-White/GBM paths ...")

t0 = time.perf_counter()
market = generate_paths(sim_config)
elapsed = time.perf_counter() - t0

print(f"Done in {elapsed:.2f}s.")
print("\nWhat came back (a dict of JAX arrays):")
for key, arr in market.items():
    print(f"  {key:<12} shape={tuple(arr.shape)}  dtype={arr.dtype}")

simulated_rates = market["rates"][:, :, 0]  # [Scenarios, TimeSteps]
print("\nSimulated short rate, scenario 0, across the time grid:")
print("  " + ", ".join(f"t={t:.2f}: {r:.4%}" for t, r in zip(TIME_GRID[1:], simulated_rates[0])))
print(f"\nAcross all {NUM_SCENARIOS:,} scenarios at the LAST time step (t={TIME_GRID[-1]}):")
last_step_rates = np.asarray(simulated_rates[:, -1])
print(f"  mean={last_step_rates.mean():.4%}  std={last_step_rates.std():.4%}  "
      f"min={last_step_rates.min():.4%}  max={last_step_rates.max():.4%}")
print("  (should cluster around the 3% long-run mean -- mean reversion pulling it back)")


# =============================================================================
# STEP 2 -- CALIBRATION
#
# Before pricing the Bermudan/American swaptions below, fit a genuine
# volatility TERM STRUCTURE (one value per exercise date) to a handful of
# market-quoted swaption volatilities -- rather than just reusing the flat
# HW_SIGMA above. This is what a real desk would do: back out the model
# parameter the market is actually implying, instead of guessing a number.
# =============================================================================
section("STEP 2 -- Calibrating a volatility term structure to market quotes")

CALIBRATION_EXERCISE_TIMES = [1.0, 2.0, 3.0, 4.0]
CALIBRATION_FINAL_MATURITY = 5.0
MARKET_NORMAL_VOLS = [0.0080, 0.0088, 0.0095, 0.0100]  # rising term structure, in decimal (80bp, 88bp, ...)

print("Market-quoted (normal/Bachelier) volatilities, by exercise date:")
for t, vol in zip(CALIBRATION_EXERCISE_TIMES, MARKET_NORMAL_VOLS):
    print(f"  {t:.0f}Y-into-{CALIBRATION_FINAL_MATURITY - t:.0f}Y swaption: {vol:.2%}")

basket = build_coterminal_basket(
    exercise_times=CALIBRATION_EXERCISE_TIMES,
    final_maturity_time=CALIBRATION_FINAL_MATURITY,
    notional=1_000_000.0,
    payer=True,
    market_vols=MARKET_NORMAL_VOLS,
    zero_curve=zero_curve_jax,
    evaluation_date=TODAY,
)
print(f"\nBuilt a co-terminal calibration basket: {len(basket)} instruments, "
      f"all maturing at year {CALIBRATION_FINAL_MATURITY:.0f}.")

calibration = calibrate_lgm_sigma(basket, zero_curve_jax, a=HW_A)

print("\nBootstrapped piecewise sigma (one value per exercise date):")
print(f"  bucket breakpoints (years): {np.asarray(calibration.sigma.times)}")
print(f"  bucket values             : {np.asarray(calibration.sigma.values)}")
print(f"  RMSE (model price vs market price, should be ~0): {calibration.rmse:.2e}")
print("\nPer-instrument check -- model price should equal market price after calibration:")
for i, (mkt, mdl) in enumerate(zip(np.asarray(calibration.market_prices), np.asarray(calibration.model_prices))):
    print(f"  bucket {i}: market={mkt:,.2f}  model={mdl:,.2f}")

CALIBRATED_SIGMA = calibration.sigma  # feeds the Bermudan/American trades below


# =============================================================================
# STEP 3 -- INSTRUMENT PRICING
#
# Four trades, one of each type this engine prices, each requiring a
# genuinely different calculation:
#   - swap:              linear cashflow sum, no optionality
#   - European swaption:  closed-form (Jamshidian decomposition)
#   - Bermudan swaption:  numeric backward induction (early exercise, a
#                          handful of discrete dates) -- uses the calibrated
#                          sigma from Step 2
#   - American swaption:  the same backward induction, on a much finer
#                          discretized exercise schedule
# =============================================================================
section("STEP 3 -- Pricing a diverse portfolio")

# swap_cfg was already built above (Step 0) -- its own real cashflow dates
# are what defined PILLAR_TIMES in the first place.

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

print("Portfolio:")
print(f"  1. Swap             : {swap_cfg.swap_tenor} payer, notional ${swap_cfg.notional:,.0f}, fixed {swap_cfg.fixed_rate:.2%}")
print(f"  2. European swaption: {european_cfg.swap_tenor} payer 2Y-forward, "
      f"notional ${european_cfg.notional:,.0f}, strike {european_cfg.fixed_rate:.2%}")
print(f"  3. Bermudan swaption: {bermudan_cfg.swap_tenor} payer, "
      f"notional ${bermudan_cfg.notional:,.0f}, exercises at {bermudan_cfg.exercise_times}")
american_dates = american_cfg.to_bermudan().exercise_times
print(f"  4. American swaption: receiver window [{american_cfg.first_exercise:.0f}Y, "
      f"{american_cfg.last_exercise:.0f}Y], notional ${american_cfg.notional:,.0f}, "
      f"discretized into {len(american_dates)} exercise dates")

step_times = jnp.array(TIME_GRID[1:], dtype=jnp.float64)

swap_cube = price_swaps(market["yield_curves"], PILLAR_TIMES, [swap_cfg])
european_cube = price_swaptions(market["rates"], step_times, [european_cfg])
bermudan_cube = price_bermudan_swaptions([bermudan_cfg], market["rates"], step_times)
american_cube = price_american_swaptions([american_cfg], market["rates"], step_times)

print("\nEach pricer returns an NPV cube shaped [Scenarios, TimeSteps, Trades]:")
print(f"  swap_cube      : {tuple(swap_cube.shape)}")
print(f"  european_cube  : {tuple(european_cube.shape)}")
print(f"  bermudan_cube  : {tuple(bermudan_cube.shape)}")
print(f"  american_cube  : {tuple(american_cube.shape)}")

print("\nMean NPV across all scenarios, at each simulated time step:")
header = "  time  " + "".join(f"{'swap':>14}{'european':>14}{'bermudan':>14}{'american':>14}")
print(header)
for i, t in enumerate(TIME_GRID[1:]):
    row = (
        f"  {t:>4.2f}  "
        f"{float(jnp.mean(swap_cube[:, i, 0])):>14,.2f}"
        f"{float(jnp.mean(european_cube[:, i, 0])):>14,.2f}"
        f"{float(jnp.mean(bermudan_cube[:, i, 0])):>14,.2f}"
        f"{float(jnp.mean(american_cube[:, i, 0])):>14,.2f}"
    )
    print(row)
print("  (the European/Bermudan/American columns drop to 0 once every remaining")
print("   exercise opportunity for that trade has passed)")


# =============================================================================
# STEP 4 -- RISK AGGREGATION
#
# Combine every trade's NPV cube into ONE portfolio NPV cube, then compute
# VaR/Expected Shortfall from the resulting spread of simulated outcomes.
# risk/var_es.py never looks at what kind of trade produced a cube -- it
# only ever sees numbers shaped [Scenarios, TimeSteps, Trades].
# =============================================================================
section("STEP 4 -- Risk aggregation: VaR and Expected Shortfall")

portfolio_cube = jnp.concatenate(
    [swap_cube, european_cube, bermudan_cube, american_cube], axis=-1,
)  # [Scenarios, TimeSteps, 4]
print(f"Combined portfolio cube: {tuple(portfolio_cube.shape)}  (4 trades, stacked on the last axis)")

# base_npv: today's ACTUAL portfolio value -- a separate, zero-shock
# revaluation at t=0, NOT read off the simulated cube (which only has
# values at t=0.5 onward; the simulation's own time_grid starts stepping
# forward from there, per generate_paths' own [TimeSteps] convention).
from engine.simulation.demo_scenarios import flat_yield_curves
from engine.instruments.bermudan_swaption import price_bermudan_swaption_base

base_yield_curves = flat_yield_curves(
    disc_rate=FLAT_RATE, fwd_rate=FLAT_RATE, maturities=PILLAR_TIMES, eval_date=TODAY,
)
base_swap_npv = float(price_swaps(base_yield_curves, PILLAR_TIMES, [swap_cfg])[0, 0, 0])

# European swaption's own t=0 baseline: one deterministic "scenario" at
# r(0) = today's flat short rate, priced at step_times=[0.0] -- Jamshidian's
# formula is exact at t=0, so this needs no simulation, just a single
# evaluation of the same pricer used for the conditional cube above.
r0_path = jnp.array([[[FLAT_RATE]]])  # [1 scenario, 1 step, 1 rate factor]
base_european_npv = float(price_swaptions(r0_path, jnp.array([0.0]), [european_cfg])[0, 0, 0])

base_bermudan_npv = price_bermudan_swaption_base(bermudan_cfg)
base_american_npv = price_bermudan_swaption_base(american_cfg.to_bermudan())

base_npv = base_swap_npv + base_european_npv + base_bermudan_npv + base_american_npv

print(f"\nToday's actual (zero-shock) baseline NPV:")
print(f"  swap     : {base_swap_npv:>14,.2f}")
print(f"  european : {base_european_npv:>14,.2f}")
print(f"  bermudan : {base_bermudan_npv:>14,.2f}")
print(f"  american : {base_american_npv:>14,.2f}")
print(f"  total base NPV used for P&L: {base_npv:,.2f}")

risk = compute_risk_metrics(portfolio_cube, base_npv, percentiles=(0.95, 0.99))

print("\nVaR / Expected Shortfall at each simulated time step:")
metric_names = list(risk.keys())
print("  time   " + "".join(f"{m:>12}" for m in metric_names))
for i, t in enumerate(TIME_GRID[1:]):
    row = f"  {t:>4.2f}  " + "".join(
        f"{float(risk[m][i]):>12,.0f}" if not np.isnan(float(risk[m][i])) else f"{'nan':>12}"
        for m in metric_names
    )
    print(row)
print("\n  VaR_95 : the loss so bad only 5% of scenarios were worse")
print("  ES_95  : the AVERAGE loss among that worst 5% (a stricter, tail-aware number)")
print("  nan    : expected, not a bug -- ES is undefined when every observation in the")
print("           worst-5%/1% tail is EXACTLY tied at the VaR cutoff (no observation is")
print("           strictly worse than VaR to average). ORE's own RiskStatistics raises")
print("           in this case; this engine reports NaN instead, since JAX code can't")
print("           raise mid-computation -- callers are expected to check for it.")


# =============================================================================
# STEP 5 -- GREEKS (Delta / Gamma / Theta / Vega) on the Bermudan trade
#
# A different question than VaR: not "what's the spread of outcomes across
# many scenarios," but "if today's curve moves by exactly 1bp, how much
# does THIS ONE trade's value change, right now?" Computed via JAX
# automatic differentiation straight through the pricing formula -- no
# bump-and-revalue, no finite-difference step-size choice.
# =============================================================================
section("STEP 5 -- Greeks on the Bermudan swaption (autodiff, not bump-and-revalue)")

greeks = bermudan_delta_gamma(bermudan_cfg, zero_curve_jax)
theta = bermudan_theta(bermudan_cfg, zero_curve_jax)
vega = bermudan_vega(bermudan_cfg, zero_curve_jax, basket)

print("Delta (dollar NPV change per 1bp move), per curve pillar:")
for t, d in zip(PILLAR_TIMES, np.asarray(greeks["delta"])):
    print(f"  {t:>7.4f}Y pillar: {d:>10.2f}")

print("\nGamma (how much Delta itself changes per 1bp), per curve pillar:")
for t, g in zip(PILLAR_TIMES, np.asarray(greeks["gamma"])):
    print(f"  {t:>7.4f}Y pillar: {g:>10.4f}")

print(f"\nTheta (1-day time decay, market held still): {theta:,.2f}")

print("\nVega (dollar NPV change per 1bp move in EACH basket instrument's own market vol):")
for t, v in zip(CALIBRATION_EXERCISE_TIMES, np.asarray(vega)):
    print(f"  {t:.0f}Y exercise bucket: {v:>10.2f}")


section("Done")
print("Every number above came from calling the same functions in engine/ that")
print("the test suite and the individual modules' own __main__ demos call --")
print("this script just walks them in pipeline order with the intermediate")
print("state printed at each stage.")
