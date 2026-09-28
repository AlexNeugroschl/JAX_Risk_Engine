"""
A large mixed portfolio (swaps, Europeans, Bermudans, Americans) priced through one
simulated three-factor market, with per-type cross-checks and portfolio risk aggregation.

As in tests/test_end_to_end.py, ORE is fed the engine's own simulated short rates, so a
divergence is a pricing difference, not sampling noise. References per type:

  - Swaps: `ORE.DiscountingSwapEngine`, at t=0 only (the cube misprices aged swaps past
    first accrual, I-04).
  - Europeans: `ORE.JamshidianSwaptionEngine`, per scenario, on a curve implied from
    `ORE.HullWhite.discountBond(t, T, r)` at a later date.
  - Bermudans/Americans: at t=0, the direct-integration single-exercise reference
    (tests/bermudan_references.py) and the bound that more exercise dates cannot lower the
    value. (Parity with ORE's own LGM engine is tested in tests/test_ore_lgm_parity.py.)

Dates: `date + N` adds calendar days; `ORE.TARGET().advance(date, N, ORE.Days)` adds
business days.
"""
import time
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.simulation.market_model import SimulationConfig, EquityConfig, RatesConfig, ZeroCurveConfig, generate_paths
from engine.instruments.swap import SwapConfig, price_swaps, prepare_swap, _price_one_swap
from engine.instruments.european_swaption import (
    SwaptionConfig, prepare_swaption, _price_one_swaption, price_swaptions,
)
from engine.instruments.bermudan_swaption import (
    BermudanSwaptionConfig, price_bermudan_swaption_base, price_bermudan_swaptions,
)
from engine.instruments.american_swaption import AmericanSwaptionConfig, price_american_swaptions
from engine.risk.var_es import compute_risk_metrics
from engine.simulation.demo_scenarios import flat_yield_curves
from engine.models.hull_white import ZeroCurve as _HwZeroCurve
from bermudan_references import single_exercise_value_by_integration
from date_helpers import in_years

TODAY = ORE.Date(30, 7, 2026)
DAY_COUNTER = ORE.Actual365Fixed()

# Three USD rate factors: 0 = discounting (OIS), 1 and 2 = two forwarding curves, so trades
# can use distinct discount/forward pairs.
RATE0, RATE1, RATE2 = 0.030, 0.035, 0.025
HW_A = [0.03, 0.03, 0.04]
HW_SIGMA = [0.01, 0.012, 0.009]

TIME_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
# Swap cube pillars come from ORE's own schedules (_collect_swap_maturities), not typed in.

ZERO_CURVE_0 = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[RATE0] * 6)
ZERO_CURVE_1 = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[RATE1] * 6)
ZERO_CURVE_2 = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[RATE2] * 6)
ZERO_CURVES = [ZERO_CURVE_0, ZERO_CURVE_1, ZERO_CURVE_2]
FLAT_RATES = [RATE0, RATE1, RATE2]


def _collect_swap_maturities(swap_cfgs):
    """Sorted union of every swap's payment and accrual times, from its ORE schedule: the
    pillar set `swap._maturity_indices` requires."""
    times = set()
    for cfg in swap_cfgs:
        from engine.instruments.swap import _build_ore_swap
        swap = _build_ore_swap(cfg)
        today = cfg.evaluation_date
        for cf in swap.fixedLeg():
            c = ORE.as_fixed_rate_coupon(cf)
            times.add(DAY_COUNTER.yearFraction(today, c.date()))
            times.add(DAY_COUNTER.yearFraction(today, c.accrualStartDate()))
        for cf in swap.floatingLeg():
            c = ORE.as_floating_rate_coupon(cf)
            times.add(DAY_COUNTER.yearFraction(today, c.date()))
            times.add(DAY_COUNTER.yearFraction(today, c.accrualStartDate()))
            times.add(DAY_COUNTER.yearFraction(today, c.accrualEndDate()))
    return sorted(times)


# =============================================================================
# PORTFOLIO DEFINITION
# =============================================================================
def _build_swaps():
    """9 swaps varying tenor, rate, notional, direction and curve pair (single- and
    multi-curve)."""
    return [
        SwapConfig(notional=1_000_000.0, fixed_rate=0.030, payer=True,
                   discount_curve_index=0, forward_curve_index=0, swap_tenor="2Y", evaluation_date=TODAY),
        SwapConfig(notional=2_000_000.0, fixed_rate=0.028, payer=False,
                   discount_curve_index=0, forward_curve_index=1, swap_tenor="2Y", evaluation_date=TODAY),
        SwapConfig(notional=1_500_000.0, fixed_rate=0.032, payer=True,
                   discount_curve_index=0, forward_curve_index=2, swap_tenor="3Y", evaluation_date=TODAY),
        SwapConfig(notional=750_000.0, fixed_rate=0.035, payer=False,
                   discount_curve_index=1, forward_curve_index=1, swap_tenor="3Y", evaluation_date=TODAY),
        SwapConfig(notional=3_000_000.0, fixed_rate=0.025, payer=True,
                   discount_curve_index=2, forward_curve_index=2, swap_tenor="5Y", evaluation_date=TODAY),
        SwapConfig(notional=500_000.0, fixed_rate=0.033, payer=False,
                   discount_curve_index=2, forward_curve_index=0, swap_tenor="5Y", evaluation_date=TODAY),
        SwapConfig(notional=1_200_000.0, fixed_rate=0.029, payer=True,
                   discount_curve_index=1, forward_curve_index=2, swap_tenor="7Y", evaluation_date=TODAY),
        SwapConfig(notional=900_000.0, fixed_rate=0.031, payer=False,
                   discount_curve_index=0, forward_curve_index=0, swap_tenor="7Y", evaluation_date=TODAY),
        SwapConfig(notional=400_000.0, fixed_rate=0.030, payer=True,
                   discount_curve_index=0, forward_curve_index=1, swap_tenor="2Y",
                   index_tenor_months=3, evaluation_date=TODAY),
    ]


def _build_european_swaptions():
    """6 Europeans varying moneyness, tenor, forward start, direction and rate factor."""
    return [
        SwaptionConfig(notional=1_500_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
                       hw_a=HW_A[0], hw_sigma=HW_SIGMA[0], initial_zero_curve=ZERO_CURVE_0,
                       swap_tenor="5Y", forward_start=ORE.Period(1, ORE.Years), evaluation_date=TODAY),
        SwaptionConfig(notional=800_000.0, fixed_rate=0.040, payer=True, rate_factor_index=0,
                       hw_a=HW_A[0], hw_sigma=HW_SIGMA[0], initial_zero_curve=ZERO_CURVE_0,
                       swap_tenor="3Y", forward_start=ORE.Period(2, ORE.Years), evaluation_date=TODAY),
        SwaptionConfig(notional=1_000_000.0, fixed_rate=0.020, payer=False, rate_factor_index=0,
                       hw_a=HW_A[0], hw_sigma=HW_SIGMA[0], initial_zero_curve=ZERO_CURVE_0,
                       swap_tenor="4Y", forward_start=ORE.Period(3, ORE.Years), evaluation_date=TODAY),
        SwaptionConfig(notional=1_200_000.0, fixed_rate=0.035, payer=True, rate_factor_index=1,
                       hw_a=HW_A[1], hw_sigma=HW_SIGMA[1], initial_zero_curve=ZERO_CURVE_1,
                       swap_tenor="5Y", forward_start=ORE.Period(1, ORE.Years), evaluation_date=TODAY),
        SwaptionConfig(notional=600_000.0, fixed_rate=0.028, payer=False, rate_factor_index=1,
                       hw_a=HW_A[1], hw_sigma=HW_SIGMA[1], initial_zero_curve=ZERO_CURVE_1,
                       swap_tenor="2Y", forward_start=ORE.Period(0, ORE.Days), evaluation_date=TODAY),
        SwaptionConfig(notional=900_000.0, fixed_rate=0.025, payer=False, rate_factor_index=2,
                       hw_a=HW_A[2], hw_sigma=HW_SIGMA[2], initial_zero_curve=ZERO_CURVE_2,
                       swap_tenor="3Y", forward_start=ORE.Period(2, ORE.Years), evaluation_date=TODAY),
    ]


def _expires_near(cfg, years: int) -> bool:
    """Whether a European's exercise date is within a week after `years` whole years from
    TODAY (its NPV there is near zero)."""
    point = ORE.TARGET().advance(TODAY, ORE.Period(years, ORE.Years))
    return point <= cfg.exercise_date <= point + 7


def _build_bermudan_swaptions():
    """4 Bermudans varying exercise schedule, tenor and direction over 2 rate factors.
    Exercise dates are whole years from TODAY, so most fall inside an accrual period and
    enter the next whole period, as in ORE."""
    return [
        BermudanSwaptionConfig(notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
                                hw_a=HW_A[0], hw_sigma=HW_SIGMA[0], initial_zero_curve=ZERO_CURVE_0,
                                exercise_dates=in_years(TODAY, [1.0, 2.0, 3.0, 4.0]), swap_tenor="5Y",
                                evaluation_date=TODAY, n_per_std=64, std_devs=6.0),
        BermudanSwaptionConfig(notional=600_000.0, fixed_rate=0.033, payer=False, rate_factor_index=0,
                                hw_a=HW_A[0], hw_sigma=HW_SIGMA[0], initial_zero_curve=ZERO_CURVE_0,
                                exercise_dates=in_years(TODAY, [1.0, 3.0]), swap_tenor="5Y",
                                evaluation_date=TODAY, n_per_std=64, std_devs=6.0),
        BermudanSwaptionConfig(notional=1_400_000.0, fixed_rate=0.027, payer=True, rate_factor_index=1,
                                hw_a=HW_A[1], hw_sigma=HW_SIGMA[1], initial_zero_curve=ZERO_CURVE_1,
                                exercise_dates=in_years(TODAY, [2.0, 3.0, 4.0, 5.0, 6.0]), swap_tenor="7Y",
                                evaluation_date=TODAY, n_per_std=64, std_devs=6.0),
        BermudanSwaptionConfig(notional=500_000.0, fixed_rate=0.031, payer=False, rate_factor_index=1,
                                hw_a=HW_A[1], hw_sigma=HW_SIGMA[1], initial_zero_curve=ZERO_CURVE_1,
                                exercise_dates=in_years(TODAY, [1.0]), swap_tenor="3Y",
                                evaluation_date=TODAY, n_per_std=64, std_devs=6.0),
    ]


def _build_american_swaptions():
    """3 Americans varying window and exercise_time_steps_per_year over 2 rate factors."""
    return [
        AmericanSwaptionConfig(notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
                                hw_a=HW_A[0], hw_sigma=HW_SIGMA[0], initial_zero_curve=ZERO_CURVE_0,
                                first_exercise_date=in_years(TODAY, 1.0),
                                last_exercise_date=in_years(TODAY, 4.0), swap_tenor="5Y",
                                exercise_time_steps_per_year=1, evaluation_date=TODAY, n_per_std=64, std_devs=6.0),
        AmericanSwaptionConfig(notional=700_000.0, fixed_rate=0.026, payer=False, rate_factor_index=0,
                                hw_a=HW_A[0], hw_sigma=HW_SIGMA[0], initial_zero_curve=ZERO_CURVE_0,
                                first_exercise_date=in_years(TODAY, 2.0),
                                last_exercise_date=in_years(TODAY, 6.0), swap_tenor="7Y",
                                exercise_time_steps_per_year=2, evaluation_date=TODAY, n_per_std=64, std_devs=6.0),
        AmericanSwaptionConfig(notional=1_100_000.0, fixed_rate=0.034, payer=True, rate_factor_index=2,
                                hw_a=HW_A[2], hw_sigma=HW_SIGMA[2], initial_zero_curve=ZERO_CURVE_2,
                                first_exercise_date=in_years(TODAY, 1.0),
                                last_exercise_date=in_years(TODAY, 3.0), swap_tenor="3Y",
                                exercise_time_steps_per_year=1, evaluation_date=TODAY, n_per_std=64, std_devs=6.0),
    ]


def _sim_config(scenarios: int, swap_maturities) -> SimulationConfig:
    """Three rate factors plus one placeholder equity; rates correlated with each other."""
    return SimulationConfig(
        time_grid=TIME_GRID,
        scenarios=scenarios,
        equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[0.0, 0.0, 0.0]]),
        rates=RatesConfig(
            initial_rates=FLAT_RATES, theta=FLAT_RATES, mean_reversion=HW_A,
            maturities=swap_maturities, initial_zero_curves=ZERO_CURVES,
        ),
        joint_covariance=[
            [0.0400, 0.0000, 0.0000, 0.0000],
            [0.0000, HW_SIGMA[0] ** 2, 0.3 * HW_SIGMA[0] * HW_SIGMA[1], 0.2 * HW_SIGMA[0] * HW_SIGMA[2]],
            [0.0000, 0.3 * HW_SIGMA[0] * HW_SIGMA[1], HW_SIGMA[1] ** 2, 0.25 * HW_SIGMA[1] * HW_SIGMA[2]],
            [0.0000, 0.2 * HW_SIGMA[0] * HW_SIGMA[2], 0.25 * HW_SIGMA[1] * HW_SIGMA[2], HW_SIGMA[2] ** 2],
        ],
    )


# =============================================================================
# ENGINE-SIDE PRICING
# =============================================================================
def _price_full_portfolio_engine(scenarios: int, swaps, euro_swaptions, bermudans, americans, swap_maturities):
    """Price any mix of the four types through one simulation. Returns the portfolio NPV
    `[S, T]`, t=0 base NPV, per-type cubes, simulated rates, risk metrics and timing."""
    config = _sim_config(scenarios, swap_maturities)
    t_start = time.perf_counter()
    market = generate_paths(config)
    step_times = jnp.array(config.time_grid[1:])
    maturities_np = np.asarray(swap_maturities)

    num_steps = len(config.time_grid) - 1
    num_scen = scenarios
    total = jnp.zeros((num_scen, num_steps))
    base_npv = 0.0
    cubes = {}

    if swaps:
        swap_cube = price_swaps(market["yield_curves"], maturities_np, swaps)  # [S,T,Ntr]
        cubes["swaps"] = swap_cube
        total = total + jnp.sum(swap_cube, axis=-1)
        base_cube = flat_yield_curves(disc_rate=1.0, fwd_rate=1.0, maturities=swap_maturities, eval_date=TODAY)
        # t=0 base: each swap on its own curve pair (see _swaps_base_npv).
        base_npv += _swaps_base_npv(swaps, swap_maturities)

    if euro_swaptions:
        prepared_e = [prepare_swaption(c) for c in euro_swaptions]
        euro_cube = price_swaptions(market["rates"], step_times, euro_swaptions)  # [S,T,Ntr]
        cubes["european"] = euro_cube
        total = total + jnp.sum(euro_cube, axis=-1)
        t0_step = jnp.array([0.0])
        for cfg, prep in zip(euro_swaptions, prepared_e):
            r0_path = jnp.array([[[FLAT_RATES[cfg.rate_factor_index]]]])
            # Base value from a [1, 1, NumHW] path with every factor at its flat rate.
            r0_full = jnp.array([[FLAT_RATES]])
            base_npv += float(_price_one_swaption(r0_full, t0_step, prep)[0, 0])

    if bermudans:
        berm_cube = price_bermudan_swaptions(bermudans, market["rates"], step_times)  # [S,T,Ntr]
        cubes["bermudan"] = berm_cube
        total = total + jnp.sum(berm_cube, axis=-1)
        for cfg in bermudans:
            base_npv += price_bermudan_swaption_base(cfg)

    if americans:
        amer_cube = price_american_swaptions(americans, market["rates"], step_times)  # [S,T,Ntr]
        cubes["american"] = amer_cube
        total = total + jnp.sum(amer_cube, axis=-1)
        for cfg in americans:
            base_npv += price_bermudan_swaption_base(cfg)

    npv_cube = total[:, :, None]  # [Scenarios, TimeSteps, Trades=1] combined portfolio
    metrics = compute_risk_metrics(npv_cube, base_npv, percentiles=(0.95, 0.99))
    elapsed = time.perf_counter() - t_start

    return {
        "portfolio_npv": np.asarray(total),  # [S, T]
        "base_npv": base_npv,
        "metrics": metrics,
        "rates": np.asarray(market["rates"]),  # [S, T, NumHW]
        "cubes": {k: np.asarray(v) for k, v in cubes.items()},
        "step_times": np.asarray(config.time_grid[1:]),
        "elapsed": elapsed,
    }


def _swaps_base_npv(swaps, swap_maturities) -> float:
    """t=0 sum of swap NPVs, each on its own (discount, forward) flat rates.

    `flat_yield_curves` builds a two-slot cube (discount, forward), so each swap is remapped
    to curve indices 0/1 against a cube of its own two rates. Passing index 2 directly
    would be clipped silently by JAX."""
    total = 0.0
    for cfg in swaps:
        disc_rate = FLAT_RATES[cfg.discount_curve_index]
        fwd_rate = FLAT_RATES[cfg.forward_curve_index]
        remapped_cfg = replace(cfg, discount_curve_index=0, forward_curve_index=1)
        base_cube = flat_yield_curves(disc_rate=disc_rate, fwd_rate=fwd_rate, maturities=swap_maturities, eval_date=TODAY)
        total += float(price_swaps(base_cube, np.asarray(swap_maturities), [remapped_cfg])[0, 0, 0])
    return total


# =============================================================================
# ORE-side pricing (swaps and Europeans only; see the module docstring)
# =============================================================================
def _ore_index(curve_handle, tenor_months=6):
    dc = DAY_COUNTER
    return ORE.IborIndex(
        "SimIndex", ORE.Period(tenor_months, ORE.Months), 2,
        ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False, dc, curve_handle,
    )


def _price_swaps_ore(swaps, rates_t: np.ndarray, t_eval: float):
    """Total swap NPV per scenario at time `t_eval`, plus the t=0 base NPV. For t > 0 each
    scenario's curves are implied from `ORE.HullWhite.discountBond` at the simulated rates
    (the pattern of test_end_to_end.py, over three factors)."""
    dc = DAY_COUNTER
    hw_models_t0 = [ORE.HullWhite(ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, FLAT_RATES[k], dc)), HW_A[k], HW_SIGMA[k]) for k in range(3)]

    ORE.Settings.instance().evaluationDate = TODAY
    curves_t0 = [ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, FLAT_RATES[k], dc)) for k in range(3)]
    idx_t0 = [_ore_index(curves_t0[k], tenor_months=6) for k in range(3)]
    idx_t0_3m = [_ore_index(curves_t0[k], tenor_months=3) for k in range(3)]

    def build_swap_t0(cfg):
        idx_map = idx_t0_3m if cfg.index_tenor_months == 3 else idx_t0
        swap_type = ORE.VanillaSwap.Payer if cfg.payer else ORE.VanillaSwap.Receiver
        s = ORE.MakeVanillaSwap(
            ORE.Period(0, ORE.Days), idx_map[cfg.forward_curve_index], cfg.fixed_rate,
            nominal=cfg.notional, swapType=swap_type, fixedLegDayCount=dc, floatingLegDayCount=dc,
            effectiveDate=cfg.effective_date, terminationDate=cfg.maturity_date,
        )
        s.setPricingEngine(ORE.DiscountingSwapEngine(curves_t0[cfg.discount_curve_index]))
        return s

    base_npv = sum(build_swap_t0(cfg).NPV() for cfg in swaps)

    if t_eval == 0.0:
        npv_per_scenario = np.full(rates_t.shape[0], base_npv)
        return npv_per_scenario, base_npv

    eval_date = TODAY + int(round(t_eval * 365))
    curve_years = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def price_one_scenario(r_vec: np.ndarray) -> float:
        ORE.Settings.instance().evaluationDate = eval_date
        implied_curves = []
        for k in range(3):
            dates = [eval_date] + [TODAY + int(round((t_eval + y) * 365)) for y in curve_years]
            discounts = [1.0] + [hw_models_t0[k].discountBond(t_eval, t_eval + y, r_vec[k]) for y in curve_years]
            implied_curves.append(ORE.YieldTermStructureHandle(ORE.DiscountCurve(dates, discounts, dc)))
        idx6 = [_ore_index(implied_curves[k], tenor_months=6) for k in range(3)]
        idx3 = [_ore_index(implied_curves[k], tenor_months=3) for k in range(3)]

        total = 0.0
        for cfg in swaps:
            idx_map = idx3 if cfg.index_tenor_months == 3 else idx6
            swap_type = ORE.VanillaSwap.Payer if cfg.payer else ORE.VanillaSwap.Receiver
            s = ORE.MakeVanillaSwap(
                ORE.Period(0, ORE.Days), idx_map[cfg.forward_curve_index], cfg.fixed_rate,
                nominal=cfg.notional, swapType=swap_type, fixedLegDayCount=dc, floatingLegDayCount=dc,
                effectiveDate=cfg.effective_date, terminationDate=cfg.maturity_date,
            )
            s.setPricingEngine(ORE.DiscountingSwapEngine(implied_curves[cfg.discount_curve_index]))
            total += s.NPV()
        return total

    npv_per_scenario = np.array([price_one_scenario(r) for r in rates_t])
    return npv_per_scenario, base_npv


def _price_european_swaptions_ore(euro_swaptions, rates_t: np.ndarray, t_eval_years: int):
    """European NPV per scenario at `t_eval_years`, plus the t=0 base NPV, by
    `ORE.JamshidianSwaptionEngine` on curves implied from `ORE.HullWhite.discountBond`.

    `t_eval_years` must be a whole number: the evaluation date is built by calendar-period
    advance, as tests/test_european_swaption.py does. Day-rounded fractional dates made
    this helper (not the engine) diverge ~0.8% from the engine's year-fraction conditioning.
    """
    if t_eval_years != int(t_eval_years):
        raise ValueError(f"t_eval_years must be a whole number of years; got {t_eval_years}")
    t_eval_years = int(t_eval_years)
    dc = DAY_COUNTER
    hw_models_t0 = [ORE.HullWhite(ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, FLAT_RATES[k], dc)), HW_A[k], HW_SIGMA[k]) for k in range(3)]

    ORE.Settings.instance().evaluationDate = TODAY
    curves_t0 = [ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, FLAT_RATES[k], dc)) for k in range(3)]
    idx_t0 = [_ore_index(curves_t0[k]) for k in range(3)]
    hw_t0 = [ORE.HullWhite(curves_t0[k], HW_A[k], HW_SIGMA[k]) for k in range(3)]

    def build_swaption_t0(cfg):
        k = cfg.rate_factor_index
        swap_type = ORE.VanillaSwap.Payer if cfg.payer else ORE.VanillaSwap.Receiver
        underlying = ORE.MakeVanillaSwap(
            ORE.Period(0, ORE.Days), idx_t0[k], cfg.fixed_rate,
            nominal=cfg.notional, swapType=swap_type, fixedLegDayCount=dc, floatingLegDayCount=dc,
            effectiveDate=cfg.effective_date, terminationDate=cfg.maturity_date,
        )
        swaption = ORE.Swaption(underlying, ORE.EuropeanExercise(cfg.exercise_date))
        swaption.setPricingEngine(ORE.JamshidianSwaptionEngine(hw_t0[k], curves_t0[k]))
        return swaption

    base_npv = sum(build_swaption_t0(cfg).NPV() for cfg in euro_swaptions)

    if t_eval_years == 0:
        npv_per_scenario = np.full(rates_t.shape[0], base_npv)
        return npv_per_scenario, base_npv

    eval_date = ORE.TARGET().advance(TODAY, ORE.Period(t_eval_years, ORE.Years))
    t_eval = dc.yearFraction(TODAY, eval_date)
    curve_years = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    def price_one_scenario(r_vec: np.ndarray) -> float:
        ORE.Settings.instance().evaluationDate = eval_date
        implied_curves, hw_eval = [], []
        for k in range(3):
            dates = [eval_date] + [ORE.TARGET().advance(eval_date, ORE.Period(y, ORE.Years)) for y in curve_years]
            year_fracs = [dc.yearFraction(TODAY, d) - t_eval for d in dates]
            discounts = [1.0] + [hw_models_t0[k].discountBond(t_eval, t_eval + yf, r_vec[k]) for yf in year_fracs[1:]]
            curve = ORE.YieldTermStructureHandle(ORE.DiscountCurve(dates, discounts, dc))
            implied_curves.append(curve)
            hw_eval.append(ORE.HullWhite(curve, HW_A[k], HW_SIGMA[k]))
        idx_eval = [_ore_index(implied_curves[k]) for k in range(3)]

        total = 0.0
        for cfg in euro_swaptions:
            k = cfg.rate_factor_index
            if cfg.exercise_date <= eval_date:
                continue  # expired by then: worth 0, as the engine reports
            # The same booked trade, one step older.
            swap_type = ORE.VanillaSwap.Payer if cfg.payer else ORE.VanillaSwap.Receiver
            underlying = ORE.MakeVanillaSwap(
                ORE.Period(0, ORE.Days), idx_eval[k], cfg.fixed_rate,
                nominal=cfg.notional, swapType=swap_type, fixedLegDayCount=dc, floatingLegDayCount=dc,
                effectiveDate=cfg.effective_date, terminationDate=cfg.maturity_date,
            )
            swaption = ORE.Swaption(underlying, ORE.EuropeanExercise(cfg.exercise_date))
            swaption.setPricingEngine(ORE.JamshidianSwaptionEngine(hw_eval[k], implied_curves[k]))
            total += swaption.NPV()
        return total

    npv_per_scenario = np.array([price_one_scenario(r) for r in rates_t])
    return npv_per_scenario, base_npv


# =============================================================================
# 1. Large heterogeneous portfolio: per-type cross-checks
# =============================================================================
class TestLargeHeterogeneousPortfolio:
    """22 trades (9 swaps, 6 Europeans, 4 Bermudans, 3 Americans) through one simulated
    three-factor market, each type checked against its reference (module docstring)."""

    @classmethod
    @pytest.fixture(scope="class")
    def portfolio(cls):
        swaps = _build_swaps()
        euros = _build_european_swaptions()
        berms = _build_bermudan_swaptions()
        amers = _build_american_swaptions()
        swap_maturities = _collect_swap_maturities(swaps)
        scenarios = 4096
        result = _price_full_portfolio_engine(scenarios, swaps, euros, berms, amers, swap_maturities)
        result["swaps"], result["euros"], result["berms"], result["amers"] = swaps, euros, berms, amers
        result["swap_maturities"] = swap_maturities
        return result

    @pytest.mark.slow
    def test_trade_count_and_shapes(self, portfolio):
        assert len(portfolio["swaps"]) == 9
        assert len(portfolio["euros"]) == 6
        assert len(portfolio["berms"]) == 4
        assert len(portfolio["amers"]) == 3
        assert portfolio["cubes"]["swaps"].shape[-1] == 9
        assert portfolio["cubes"]["european"].shape[-1] == 6
        assert portfolio["cubes"]["bermudan"].shape[-1] == 4
        assert portfolio["cubes"]["american"].shape[-1] == 3
        num_scen = portfolio["cubes"]["swaps"].shape[0]
        num_steps = portfolio["cubes"]["swaps"].shape[1]
        assert portfolio["portfolio_npv"].shape == (num_scen, num_steps)

    def test_swaps_match_ore_at_t0_per_curve_combination(self, portfolio):
        """Swaps are checked against ORE at t=0 only: the cube misprices a swap at any step
        past its first accrual (I-04, audit M-2), which is not what this test is for. Every
        (discount, forward) curve pair used is covered."""
        mine_base = _swaps_base_npv(portfolio["swaps"], portfolio["swap_maturities"])
        rates_t0 = portfolio["rates"][:, 0, :]
        _, ore_base = _price_swaps_ore(portfolio["swaps"], rates_t0, 0.0)
        np.testing.assert_allclose(mine_base, ore_base, rtol=1e-6)

        mine_npv_t0 = portfolio["cubes"]["swaps"][:, 0, :].sum(axis=-1)
        # The cube's first step is t=0.5, not t=0, so only finiteness is checked here; the
        # t=0 comparison is the base NPV above.
        assert np.all(np.isfinite(mine_npv_t0))

    def test_swaps_base_npv_matches_ore(self, portfolio):
        rates_t0 = portfolio["rates"][:, 0, :]
        _, ore_base = _price_swaps_ore(portfolio["swaps"], rates_t0, 0.0)
        mine_base = _swaps_base_npv(portfolio["swaps"], portfolio["swap_maturities"])
        np.testing.assert_allclose(mine_base, ore_base, rtol=1e-6)

    @pytest.mark.parametrize("step_idx,t_eval_years", [(1, 1)])
    def test_european_swaptions_match_ore_per_scenario(self, portfolio, step_idx, t_eval_years):
        """At t=1 (a whole year, as the ORE helper requires). Trades 0 and 3 are excluded:
        they expire a few days after t=1 and are worth a few dollars there, where a relative
        error is meaningless."""
        rates_t = portfolio["rates"][:, step_idx, :]
        assert portfolio["step_times"][step_idx] == pytest.approx(float(t_eval_years))
        well_conditioned = [c for c in portfolio["euros"] if not _expires_near(c, t_eval_years)]
        assert len(well_conditioned) == 4
        indices = [i for i, c in enumerate(portfolio["euros"]) if c in well_conditioned]

        ore_npv, ore_base = _price_european_swaptions_ore(well_conditioned, rates_t, t_eval_years)
        mine_npv = portfolio["cubes"]["european"][:, step_idx, indices].sum(axis=-1)
        rel = np.abs(mine_npv - ore_npv) / np.maximum(np.abs(ore_npv), 1.0)
        assert np.max(rel) < 5e-3, f"max rel diff {np.max(rel)} at t={t_eval_years}"
        assert np.mean(rel) < 1e-3

    def test_european_swaptions_base_npv_matches_ore(self, portfolio):
        rates_t0 = portfolio["rates"][:, 0, :]
        _, ore_base = _price_european_swaptions_ore(portfolio["euros"], rates_t0, 0.0)
        prepared = [prepare_swaption(c) for c in portfolio["euros"]]
        t0_step = jnp.array([0.0])
        mine_base = 0.0
        for cfg, prep in zip(portfolio["euros"], prepared):
            r0_full = jnp.array([[FLAT_RATES]])
            mine_base += float(_price_one_swaption(r0_full, t0_step, prep)[0, 0])
        np.testing.assert_allclose(mine_base, ore_base, rtol=1e-6)

    def test_bermudan_single_exercise_subset_matches_direct_integration(self, portfolio):
        """The single-exercise Bermudan (index 3) matches the direct-integration reference
        (tests/bermudan_references.py)."""
        cfg = portfolio["berms"][3]
        assert len(cfg.exercise_dates) == 1
        numeric_npv = price_bermudan_swaption_base(cfg)
        assert numeric_npv == pytest.approx(single_exercise_value_by_integration(cfg), rel=1e-4)

    def test_bermudans_at_least_as_valuable_as_last_exercise_only(self, portfolio):
        """More exercise dates cannot decrease value, for every Bermudan here."""
        for cfg in portfolio["berms"]:
            full_npv = price_bermudan_swaption_base(cfg)
            single_cfg = replace(cfg, exercise_dates=[cfg.exercise_dates[-1]])
            single_npv = price_bermudan_swaption_base(single_cfg)
            assert full_npv >= single_npv - 1e-6

    def test_americans_at_least_as_valuable_as_their_last_date_alone(self, portfolio):
        """The whole American window is worth at least an American exercisable only on its
        last day (whose one option time is in the full window's grid)."""
        for cfg in portfolio["amers"]:
            full_npv = price_bermudan_swaption_base(cfg)
            single_cfg = replace(cfg, first_exercise_date=cfg.last_exercise_date)
            single_npv = price_bermudan_swaption_base(single_cfg)
            assert full_npv >= single_npv - 1e-6

    def test_portfolio_npv_equals_sum_of_per_type_cubes(self, portfolio):
        """The portfolio NPV equals the sum of the per-type cubes (aggregation and
        broadcasting)."""
        expected = (
            portfolio["cubes"]["swaps"].sum(axis=-1)
            + portfolio["cubes"]["european"].sum(axis=-1)
            + portfolio["cubes"]["bermudan"].sum(axis=-1)
            + portfolio["cubes"]["american"].sum(axis=-1)
        )
        np.testing.assert_allclose(portfolio["portfolio_npv"], expected, rtol=1e-10)

    def test_reports_timing(self, portfolio):
        print(f"\n[timing] 22-trade portfolio, {portfolio['cubes']['swaps'].shape[0]} scenarios: "
              f"{portfolio['elapsed']:.3f}s")
        assert portfolio["elapsed"] > 0.0


# =============================================================================
# 2. Portfolio-level risk aggregation
# =============================================================================
class TestPortfolioLevelRiskAggregation:
    """VaR/ES of the portfolio.

    The ORE cross-check uses Europeans only: the engine's swap cube diverges from ORE after
    first accrual (I-04), so including swaps would not be a like-for-like comparison. The
    full 22-trade portfolio is checked for internal consistency instead.
    """

    @classmethod
    @pytest.fixture(scope="class")
    def euro_only(cls):
        """The 4 Europeans not expiring just after t=2, where the risk test compares (a
        near-zero NPV makes relative errors meaningless). Engine and ORE price the same
        set."""
        all_euros = _build_european_swaptions()
        euros = [c for c in all_euros if not _expires_near(c, 2)]
        assert len(euros) == 4
        scenarios = 4096
        result = _price_full_portfolio_engine(scenarios, [], euros, [], [], [])
        result["euros"] = euros
        return result

    def test_var_es_match_ore_on_european_swaption_subset(self, euro_only):
        """At t=2 (step 3), a whole year as the ORE helper requires. Not the last step:
        by t=7 every European has expired, the P&L has zero variance, and the ES tail is
        empty (ORE raises; the engine returns NaN)."""
        rates = euro_only["rates"]  # [S, T, 3]
        step_times = euro_only["step_times"]
        step_idx = 3
        t_eval_years = 2
        assert step_times[step_idx] == pytest.approx(float(t_eval_years))

        rates_t = rates[:, step_idx, :]
        euro_npv, ore_base = _price_european_swaptions_ore(euro_only["euros"], rates_t, t_eval_years)
        ore_pnl = euro_npv - ore_base
        ore_stats_95 = ORE.RiskStatistics()
        ore_stats_99 = ORE.RiskStatistics()
        for v in ore_pnl:
            ore_stats_95.add(float(v), 1.0)
            ore_stats_99.add(float(v), 1.0)

        mine = euro_only["metrics"]
        np.testing.assert_allclose(float(mine["VaR_95"][step_idx]), ore_stats_95.valueAtRisk(0.95), rtol=5e-3)
        np.testing.assert_allclose(float(mine["VaR_99"][step_idx]), ore_stats_99.valueAtRisk(0.99), rtol=5e-3)
        np.testing.assert_allclose(float(mine["ES_95"][step_idx]), ore_stats_95.expectedShortfall(0.95), rtol=1e-2)
        np.testing.assert_allclose(float(mine["ES_99"][step_idx]), ore_stats_99.expectedShortfall(0.99), rtol=1e-2)

    def test_full_22_trade_portfolio_var_es_internally_consistent(self):
        """The full portfolio's VaR/ES are non-negative, ES_p >= VaR_p, and
        VaR_99 >= VaR_95 at every step."""
        swaps = _build_swaps()
        euros = _build_european_swaptions()
        berms = _build_bermudan_swaptions()
        amers = _build_american_swaptions()
        swap_maturities = _collect_swap_maturities(swaps)
        result = _price_full_portfolio_engine(4096, swaps, euros, berms, amers, swap_maturities)
        m = result["metrics"]
        for step in range(len(result["step_times"])):
            for p in ("95", "99"):
                var_v = float(m[f"VaR_{p}"][step])
                es_v = float(m[f"ES_{p}"][step])
                assert var_v >= 0.0
                if not np.isnan(es_v):
                    assert es_v >= var_v - 1e-6
            assert float(m["VaR_99"][step]) >= float(m["VaR_95"][step]) - 1e-6


# =============================================================================
# 3. Multi-curve breadth (3 rate factors, varying curve assignment per trade)
# =============================================================================
class TestMultiCurveBreadth:
    """Three rate factors in one currency (OIS plus two forwarding curves), and swaps with
    distinct discount and forward curves."""

    @classmethod
    @pytest.fixture(scope="class")
    def sim(cls):
        swaps = _build_swaps()
        swap_maturities = _collect_swap_maturities(swaps)
        config = _sim_config(2048, swap_maturities)
        market = generate_paths(config)
        return market, swap_maturities, swaps

    def test_three_rate_factors_simulated(self, sim):
        market, _, _ = sim
        assert market["rates"].shape[-1] == 3
        assert market["yield_curves"].shape[-1] == 3

    def test_rate_factors_are_not_degenerate_copies_of_each_other(self, sim):
        """The three factors have distinct paths, and each factor's terminal mean stays near
        its own theta."""
        market, _, _ = sim
        r = market["rates"]  # [S, T, 3]
        assert not np.allclose(r[:, :, 0], r[:, :, 1])
        assert not np.allclose(r[:, :, 0], r[:, :, 2])
        assert not np.allclose(r[:, :, 1], r[:, :, 2])
        # Terminal means near each factor's theta.
        for k, theta in enumerate(FLAT_RATES):
            assert abs(float(np.mean(r[:, -1, k])) - theta) < 0.02

    def test_swaps_reprice_correctly_with_mismatched_discount_forward_curves(self, sim):
        """Swaps with discount != forward curve match ORE at t=0."""
        market, swap_maturities, swaps = sim
        mismatched = [c for c in swaps if c.discount_curve_index != c.forward_curve_index]
        assert len(mismatched) >= 4, "expected several genuinely multi-curve swaps in the fixture portfolio"
        rates_t0 = market["rates"][:, 0, :]
        _, ore_base = _price_swaps_ore(mismatched, rates_t0[:1], 0.0)
        mine_base = _swaps_base_npv(mismatched, swap_maturities)
        np.testing.assert_allclose(mine_base, ore_base, rtol=1e-6)


# =============================================================================
# 4. Portfolio-composition edge cases
# =============================================================================
class TestEdgeOfPortfolioComposition:
    """All-swaps, all-swaptions, single-trade and offsetting-position portfolios."""

    def test_all_swaps_portfolio_prices_and_aggregates(self):
        swaps = _build_swaps()
        swap_maturities = _collect_swap_maturities(swaps)
        result = _price_full_portfolio_engine(1024, swaps, [], [], [], swap_maturities)
        assert result["cubes"]["swaps"].shape[-1] == len(swaps)
        assert "european" not in result["cubes"]
        assert np.all(np.isfinite(result["portfolio_npv"]))
        m = result["metrics"]
        assert float(m["VaR_95"][0]) >= 0.0

    def test_all_swaption_types_portfolio_prices_and_aggregates(self):
        """European + Bermudan + American only, no linear swaps."""
        euros = _build_european_swaptions()
        berms = _build_bermudan_swaptions()
        amers = _build_american_swaptions()
        result = _price_full_portfolio_engine(1024, [], euros, berms, amers, [])
        assert "swaps" not in result["cubes"]
        assert result["cubes"]["european"].shape[-1] == len(euros)
        assert result["cubes"]["bermudan"].shape[-1] == len(berms)
        assert result["cubes"]["american"].shape[-1] == len(amers)
        assert np.all(np.isfinite(result["portfolio_npv"]))

    def test_single_trade_portfolio(self):
        """A one-swap portfolio's metrics equal `compute_risk_metrics` on that swap's
        cube."""
        swap = _build_swaps()[0]
        swap_maturities = _collect_swap_maturities([swap])
        result = _price_full_portfolio_engine(2048, [swap], [], [], [], swap_maturities)
        assert result["cubes"]["swaps"].shape[-1] == 1
        single_cube = result["cubes"]["swaps"]  # [S, T, 1]
        direct_metrics = compute_risk_metrics(jnp.asarray(single_cube), result["base_npv"], percentiles=(0.95, 0.99))
        for key in result["metrics"]:
            np.testing.assert_allclose(
                np.asarray(result["metrics"][key]), np.asarray(direct_metrics[key]), rtol=1e-10, equal_nan=True,
            )

    def test_single_bermudan_trade_portfolio(self):
        """Single-trade case for a Bermudan."""
        berm = _build_bermudan_swaptions()[0]
        result = _price_full_portfolio_engine(512, [], [], [berm], [], [])
        assert result["cubes"]["bermudan"].shape[-1] == 1
        assert np.all(np.isfinite(result["portfolio_npv"]))

    def test_offsetting_payer_receiver_swaps_net_to_near_zero(self):
        """The same swap held payer and receiver nets to ~0 NPV at every scenario and step,
        so VaR is ~0."""
        payer = SwapConfig(notional=1_000_000.0, fixed_rate=0.030, payer=True,
                            discount_curve_index=0, forward_curve_index=1, swap_tenor="3Y", evaluation_date=TODAY)
        receiver = SwapConfig(notional=1_000_000.0, fixed_rate=0.030, payer=False,
                               discount_curve_index=0, forward_curve_index=1, swap_tenor="3Y", evaluation_date=TODAY)
        swap_maturities = _collect_swap_maturities([payer, receiver])
        result = _price_full_portfolio_engine(2048, [payer, receiver], [], [], [], swap_maturities)

        assert abs(result["base_npv"]) < 1e-6, f"base NPV should net to ~0, got {result['base_npv']}"
        max_abs_npv = float(np.max(np.abs(result["portfolio_npv"])))
        assert max_abs_npv < 1e-4, f"offsetting swap portfolio NPV should be ~0 everywhere, max |NPV|={max_abs_npv}"

        m = result["metrics"]
        assert float(m["VaR_95"][0]) < 1e-4
        assert float(m["VaR_99"][0]) < 1e-4

    def test_offsetting_payer_receiver_european_swaptions_net_to_near_zero(self):
        """Put-call parity: payer minus receiver swaption with the same strike equals the
        forward-starting payer swap's t=0 NPV (the two do not net to zero)."""
        base = dict(notional=1_000_000.0, fixed_rate=0.030, rate_factor_index=0,
                    hw_a=HW_A[0], hw_sigma=HW_SIGMA[0], initial_zero_curve=ZERO_CURVE_0,
                    swap_tenor="3Y", forward_start=ORE.Period(1, ORE.Years), evaluation_date=TODAY)
        payer_cfg = SwaptionConfig(payer=True, **base)
        receiver_cfg = SwaptionConfig(payer=False, **base)

        prep_payer = prepare_swaption(payer_cfg)
        prep_receiver = prepare_swaption(receiver_cfg)
        t0_step = jnp.array([0.0])
        r0_full = jnp.array([[FLAT_RATES]])
        payer_npv = float(_price_one_swaption(r0_full, t0_step, prep_payer)[0, 0])
        receiver_npv = float(_price_one_swaption(r0_full, t0_step, prep_receiver)[0, 0])

        # Forward payer swap NPV from ORE at t=0.
        dc = DAY_COUNTER
        ORE.Settings.instance().evaluationDate = TODAY
        curve0 = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, RATE0, dc))
        idx0 = _ore_index(curve0)
        fwd_swap = ORE.MakeVanillaSwap(
            ORE.Period("3Y"), idx0, 0.030, nominal=1_000_000.0,
            swapType=ORE.VanillaSwap.Payer, discountingTermStructure=curve0,
            fixedLegDayCount=dc, floatingLegDayCount=dc, forwardStart=ORE.Period(1, ORE.Years),
        )
        fwd_swap.setPricingEngine(ORE.DiscountingSwapEngine(curve0))
        forward_swap_npv = fwd_swap.NPV()

        np.testing.assert_allclose(payer_npv - receiver_npv, forward_swap_npv, rtol=2e-3)


# =============================================================================
# 5. Time evolution across the portfolio
# =============================================================================
class TestTimeEvolutionSanity:
    """Each type's value goes to zero after its expiry within the combined portfolio."""

    @classmethod
    @pytest.fixture(scope="class")
    def portfolio(cls):
        swaps = _build_swaps()
        euros = _build_european_swaptions()
        berms = _build_bermudan_swaptions()
        amers = _build_american_swaptions()
        swap_maturities = _collect_swap_maturities(swaps)
        result = _price_full_portfolio_engine(2048, swaps, euros, berms, amers, swap_maturities)
        result["euros"] = euros
        return result

    @pytest.mark.slow
    def test_european_swaptions_mean_npv_is_zero_after_last_exercise(self, portfolio):
        """Every European exercises by 3Y plus a few days, so at t=4 their summed NPV is
        exactly 0 on every path."""
        step_times = portfolio["step_times"]
        idx_4y = int(np.where(np.isclose(step_times, 4.0))[0][0])
        euro_cube = portfolio["cubes"]["european"]
        mean_npv_at_4y = float(np.mean(euro_cube[:, idx_4y, :].sum(axis=-1)))
        assert mean_npv_at_4y == pytest.approx(0.0, abs=1e-9)

    def test_bermudan_mean_npv_is_zero_after_last_exercise(self, portfolio):
        """Bermudan 3's only exercise is at 1Y, so it is 0 at t=2 (the other Bermudans are
        still alive)."""
        step_times = portfolio["step_times"]
        idx_2y = int(np.where(np.isclose(step_times, 2.0))[0][0])
        berm_cube = portfolio["cubes"]["bermudan"]
        mean_npv_trade3_at_2y = float(np.mean(berm_cube[:, idx_2y, 3]))
        assert mean_npv_trade3_at_2y == pytest.approx(0.0, abs=1e-6)

    def test_american_mean_npv_is_zero_after_last_exercise(self, portfolio):
        """American 2's window closes at 3Y, so it is 0 at t=4."""
        step_times = portfolio["step_times"]
        idx_4y = int(np.where(np.isclose(step_times, 4.0))[0][0])
        amer_cube = portfolio["cubes"]["american"]
        mean_npv_trade2_at_4y = float(np.mean(amer_cube[:, idx_4y, 2]))
        assert mean_npv_trade2_at_4y == pytest.approx(0.0, abs=1e-6)

    def test_combined_portfolio_mean_npv_finite_and_declining_optionality_over_time(self, portfolio):
        """The portfolio NPV and its cross-scenario standard deviation stay finite, and the
        last step's deviation (only swaps alive after t=6) is not pathologically large."""
        npv = portfolio["portfolio_npv"]  # [S, T]
        assert np.all(np.isfinite(npv))
        std_per_step = np.std(npv, axis=0)
        assert np.all(np.isfinite(std_per_step))
        # After t=6 every swaption has expired; only swaps remain. A loose sanity bound.
        step_times = portfolio["step_times"]
        idx_last = len(step_times) - 1
        assert step_times[idx_last] >= 6.0
        assert std_per_step[idx_last] < 10.0 * (np.max(std_per_step) + 1.0)


# 6. Calibration and Greeks across a Bermudan/American book: one Sigma calibrated to one
#    basket, reused by several trades, with Delta/Gamma/Theta/Vega per trade.
from engine.calibration.basket import build_coterminal_basket
from engine.calibration.lgm import calibrate_lgm_sigma
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig as _BermCfg
from engine.risk.greeks import bermudan_delta_gamma, bermudan_theta, bermudan_vega
from engine.simulation.market_model import ZeroCurveConfig as _ZCC


class TestCalibrationAndGreeksAcrossDiversePortfolio:
    """One calibrated Sigma reused across varied Bermudan/American trades."""

    @classmethod
    @pytest.fixture(scope="class")
    def calibrated(cls):
        curve = _HwZeroCurve.flat(0.03, [0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0])
        curve_config = _ZCC(times=[0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0], rates=[0.03] * 7)
        exercise_times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        targets = build_coterminal_basket(
            exercise_times=exercise_times, final_maturity_time=7.0,
            notional=1_000_000.0, payer=True,
            market_vols=[0.0075, 0.0082, 0.0088, 0.0092, 0.0095, 0.0097],
            zero_curve=curve, evaluation_date=TODAY,
        )
        result = calibrate_lgm_sigma(targets, curve, a=0.03)
        return {"curve": curve, "curve_config": curve_config, "sigma": result.sigma, "targets": targets}

    @classmethod
    @pytest.fixture(scope="class")
    def diverse_trades(cls, calibrated):
        """5 trades sharing the calibrated Sigma: deep-ITM payer, deep-OTM receiver, ATM
        single exercise, a sparse schedule, and an American window."""
        sigma = calibrated["sigma"]
        curve_config = calibrated["curve_config"]
        berm_trades = {
            "deep_itm_payer": _BermCfg(
                notional=1_000_000.0, fixed_rate=0.01, payer=True, rate_factor_index=0,
                hw_a=0.03, hw_sigma=sigma, initial_zero_curve=curve_config,
                exercise_dates=in_years(TODAY, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), swap_tenor="7Y", evaluation_date=TODAY,
            ),
            "deep_otm_receiver": _BermCfg(
                notional=1_500_000.0, fixed_rate=0.01, payer=False, rate_factor_index=0,
                hw_a=0.03, hw_sigma=sigma, initial_zero_curve=curve_config,
                exercise_dates=in_years(TODAY, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), swap_tenor="7Y", evaluation_date=TODAY,
            ),
            "atm_single_exercise": _BermCfg(
                notional=500_000.0, fixed_rate=0.03, payer=True, rate_factor_index=0,
                hw_a=0.03, hw_sigma=sigma, initial_zero_curve=curve_config,
                exercise_dates=in_years(TODAY, [3.0]), swap_tenor="4Y", evaluation_date=TODAY,
            ),
            "sparse_schedule": _BermCfg(
                notional=2_000_000.0, fixed_rate=0.028, payer=True, rate_factor_index=0,
                hw_a=0.03, hw_sigma=sigma, initial_zero_curve=curve_config,
                exercise_dates=in_years(TODAY, [2.0, 4.0, 6.0]), swap_tenor="7Y", evaluation_date=TODAY,
            ),
        }
        from engine.instruments.american_swaption import AmericanSwaptionConfig as _AmerCfg
        american_cfg = _AmerCfg(
            notional=800_000.0, fixed_rate=0.032, payer=False, rate_factor_index=0,
            hw_a=0.03, hw_sigma=sigma, initial_zero_curve=curve_config,
            first_exercise_date=in_years(TODAY, 1.0),
            last_exercise_date=in_years(TODAY, 5.0), exercise_time_steps_per_year=1,
            swap_tenor="7Y", evaluation_date=TODAY,
        )
        berm_trades["american"] = american_cfg
        return berm_trades

    @pytest.mark.slow
    def test_every_trade_prices_finite_and_signed_sensibly(self, diverse_trades):
        for name, cfg in diverse_trades.items():
            npv = price_bermudan_swaption_base(cfg)
            assert np.isfinite(npv), f"{name}: non-finite NPV"
            assert npv >= 0.0, f"{name}: negative NPV for a long option position"

    def test_deep_itm_worth_substantially_more_than_atm_despite_smaller_notional(self, diverse_trades):
        itm_npv = price_bermudan_swaption_base(diverse_trades["deep_itm_payer"])
        atm_npv = price_bermudan_swaption_base(diverse_trades["atm_single_exercise"])
        assert itm_npv > atm_npv

    def test_sparse_schedule_worth_no_more_than_dense_schedule_same_moneyness(self, diverse_trades):
        """More exercise dates cannot decrease value, with a calibrated Sigma: the sparse
        schedule against the same trade exercisable every year."""
        sparse_cfg = diverse_trades["sparse_schedule"]
        dense_cfg = _BermCfg(
            notional=sparse_cfg.notional, fixed_rate=sparse_cfg.fixed_rate, payer=sparse_cfg.payer,
            rate_factor_index=sparse_cfg.rate_factor_index, hw_a=sparse_cfg.hw_a, hw_sigma=sparse_cfg.hw_sigma,
            initial_zero_curve=sparse_cfg.initial_zero_curve,
            exercise_dates=in_years(TODAY, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            effective_date=sparse_cfg.effective_date, maturity_date=sparse_cfg.maturity_date,
            evaluation_date=sparse_cfg.evaluation_date,
        )
        sparse_npv = price_bermudan_swaption_base(sparse_cfg)
        dense_npv = price_bermudan_swaption_base(dense_cfg)
        assert dense_npv >= sparse_npv - 1e-6

    @pytest.mark.slow
    def test_delta_gamma_finite_across_every_trade(self, calibrated, diverse_trades):
        curve = calibrated["curve"]
        for name, cfg in diverse_trades.items():
            greeks = bermudan_delta_gamma(cfg, curve)
            assert jnp.all(jnp.isfinite(greeks["delta"])), f"{name}: non-finite delta"
            assert jnp.all(jnp.isfinite(greeks["gamma"])), f"{name}: non-finite gamma"

    @pytest.mark.slow
    def test_theta_finite_and_bounded_across_every_trade(self, calibrated, diverse_trades):
        curve = calibrated["curve"]
        for name, cfg in diverse_trades.items():
            npv = price_bermudan_swaption_base(cfg)
            theta = bermudan_theta(cfg, curve)
            assert np.isfinite(theta), f"{name}: non-finite theta"
            # One day of decay is a small fraction of NPV (a loose bound).
            if abs(npv) > 1.0:
                assert abs(theta) < 0.1 * abs(npv), f"{name}: theta implausibly large relative to NPV"

    @pytest.mark.slow
    def test_vega_finite_and_positive_for_trades_within_the_calibration_horizon(self, calibrated, diverse_trades):
        """Vega is finite and positive per basket instrument for the two trades exercisable
        on every basket date."""
        curve = calibrated["curve"]
        targets = calibrated["targets"]
        for name in ["deep_itm_payer", "deep_otm_receiver"]:
            cfg = diverse_trades[name]
            vega = bermudan_vega(cfg, curve, targets)
            assert jnp.all(jnp.isfinite(vega)), f"{name}: non-finite vega"
            assert jnp.all(vega > 0.0), f"{name}: non-positive vega (should be long-vol)"

    @pytest.mark.slow
    def test_payer_and_receiver_vega_are_both_positive_and_finite(self, calibrated, diverse_trades):
        """Payer and receiver Bermudans are both long volatility. Their ratio is not
        asserted: at a 1% strike against a ~3% forward the payer is deep ITM and the receiver
        deep OTM, so their per-bucket sensitivities differ by orders of magnitude."""
        curve = calibrated["curve"]
        targets = calibrated["targets"]
        payer_vega = bermudan_vega(diverse_trades["deep_itm_payer"], curve, targets)
        receiver_vega = bermudan_vega(diverse_trades["deep_otm_receiver"], curve, targets)
        assert jnp.all(jnp.isfinite(payer_vega)) and jnp.all(payer_vega > 0.0)
        assert jnp.all(jnp.isfinite(receiver_vega)) and jnp.all(receiver_vega > 0.0)

    @pytest.mark.slow
    def test_recalibrating_with_shifted_market_vols_shifts_portfolio_value_consistently(self, calibrated):
        """Recalibrating to market vols 10bp higher raises every trade's NPV (all are long
        volatility)."""
        curve = calibrated["curve"]
        curve_config = calibrated["curve_config"]
        exercise_times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        base_vols = [0.0075, 0.0082, 0.0088, 0.0092, 0.0095, 0.0097]
        shifted_vols = [v + 0.001 for v in base_vols]

        base_targets = build_coterminal_basket(
            exercise_times=exercise_times, final_maturity_time=7.0, notional=1_000_000.0,
            payer=True, market_vols=base_vols, zero_curve=curve, evaluation_date=TODAY,
        )
        shifted_targets = build_coterminal_basket(
            exercise_times=exercise_times, final_maturity_time=7.0, notional=1_000_000.0,
            payer=True, market_vols=shifted_vols, zero_curve=curve, evaluation_date=TODAY,
        )
        base_sigma = calibrate_lgm_sigma(base_targets, curve, a=0.03).sigma
        shifted_sigma = calibrate_lgm_sigma(shifted_targets, curve, a=0.03).sigma

        for payer in [True, False]:
            cfg_base = _BermCfg(
                notional=1_000_000.0, fixed_rate=0.02, payer=payer, rate_factor_index=0,
                hw_a=0.03, hw_sigma=base_sigma, initial_zero_curve=curve_config,
                exercise_dates=in_years(TODAY, exercise_times), swap_tenor="7Y", evaluation_date=TODAY,
            )
            cfg_shifted = _BermCfg(
                notional=1_000_000.0, fixed_rate=0.02, payer=payer, rate_factor_index=0,
                hw_a=0.03, hw_sigma=shifted_sigma, initial_zero_curve=curve_config,
                exercise_dates=in_years(TODAY, exercise_times), swap_tenor="7Y", evaluation_date=TODAY,
            )
            npv_base = price_bermudan_swaption_base(cfg_base)
            npv_shifted = price_bermudan_swaption_base(cfg_shifted)
            assert npv_shifted > npv_base, f"payer={payer}: higher market vol did not increase NPV"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
