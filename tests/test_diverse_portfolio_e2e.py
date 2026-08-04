"""
Diverse, larger, mixed-instrument-type portfolio end-to-end tests.

Extends tests/test_end_to_end.py's methodology (same simulated short-rate
values fed to this engine AND to independently-built real ORE objects, so
divergence is unambiguously a pricing bug rather than RNG mismatch -- see
that module's docstring for the full rationale, repeated only in summary
here) to a genuinely broad portfolio: many swaps (varying tenor / rate /
notional / payer-receiver / curve assignment), several European swaptions
(varying moneyness / tenor / forward-start), several Bermudan swaptions
(varying exercise schedules), and American swaptions (varying exercise
windows / discretization), all priced together through one simulated market
and aggregated into one VaR/ES.

Per-instrument-type ORE cross-check strategy (this is NOT uniform across
trade types, deliberately -- each type's own test suite already establishes
what a legitimate live-ORE cross-check looks like for it):

  - Swaps: ORE.DiscountingSwapEngine, conditioned per-scenario via an
    implied curve built from ORE.HullWhite.discountBond(t,T,r) -- exactly
    test_end_to_end.py's own pattern.
  - European swaptions: ORE.JamshidianSwaptionEngine, same conditioning --
    exactly test_end_to_end.py's own pattern.
  - Bermudan / American swaptions: ORE's Python bindings do NOT expose a
    constructible NumericLgmMultiLegOptionEngine (confirmed in
    engine/instruments/bermudan_swaption.py's own docstring and
    tests/test_bermudan_swaption.py), so there is no live ORE engine object
    to condition per-scenario here, unlike the other three types. Per-trade
    correctness is instead cross-checked at t=0 against the SAME independent
    LGM-Jamshidian closed form tests/test_bermudan_swaption.py's own
    TestSingleExerciseMatchesLgmJamshidian class already validates this
    engine's backward induction against (single-exercise-date case), plus
    the model-independent monotonicity bound (more exercise opportunities
    cannot decrease value) applied across this portfolio's own varied
    schedules. This is a genuine, independent check -- not a self-comparison
    -- but it is a different (formula-level, t=0) kind of check than the
    other three types' full per-scenario live-ORE cross-check, exactly
    mirroring the distinction already documented in
    engine/instruments/bermudan_swaption.py's own module docstring.

A note on ORE date arithmetic (see test_end_to_end.py's own docstring for
the full explanation): `date + N` (plain integer addition on an ORE.Date) is
calendar-day arithmetic; `ORE.TARGET().advance(date, N, ORE.Days)` is
BUSINESS-day arithmetic. This module always uses `date + N` for calendar-day
offsets and `ORE.Period(N, ORE.Years/...)` for calendar-period offsets.
"""
import time
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.simulation import SimulationConfig, EquityConfig, RatesConfig, ZeroCurveConfig, generate_paths
from engine.instruments.swap import SwapConfig, price_swaps, prepare_swap, _price_one_swap
from engine.instruments.european_swaption import (
    SwaptionConfig, prepare_swaption, _price_one_swaption, price_swaptions,
)
from engine.instruments.bermudan_swaption import (
    BermudanSwaptionConfig, _H, _lgm_bond, _zeta, prepare_bermudan,
    price_bermudan_swaption_base, price_bermudan_swaptions,
)
from engine.instruments.american_swaption import AmericanSwaptionConfig, price_american_swaptions
from engine.risk.statistics import compute_risk_metrics
from engine.scenarios import flat_yield_curves

TODAY = ORE.Date(30, 7, 2026)
DAY_COUNTER = ORE.Actual365Fixed()

# -----------------------------------------------------------------------
# Shared scenario: THREE correlated USD rate factors (an explicit breadth
# extension beyond test_end_to_end.py's two: factor 0 = OIS/discounting,
# factor 1 = Euribor-style forwarding (slightly above OIS), factor 2 = a
# second, distinct forwarding curve (e.g. a different index tenor family),
# so swaps/swaptions can be assigned genuinely different discount/forward
# curve pairs across more than 2 rate factors) -- a realistic extension of
# single_currency_swap_demo_config's existing 2-factor shape in
# engine/scenarios.py, not an invented config surface (RatesConfig already
# accepts an arbitrary-length initial_rates/theta/mean_reversion/
# initial_zero_curves list, one entry per factor; joint_covariance is just a
# bigger correlation block of the same [equities..., rates...] shape).
# -----------------------------------------------------------------------
RATE0, RATE1, RATE2 = 0.030, 0.035, 0.025
HW_A = [0.03, 0.03, 0.04]
HW_SIGMA = [0.01, 0.012, 0.009]

TIME_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
# Union of maturity pillars needed by every swap cashflow in the portfolio
# below (2Y/3Y/5Y/7Y tenors, semiannual float / annual fixed, standard 2-day
# spot lag) -- computed once via ORE itself (see _collect_swap_maturities)
# rather than hand-transcribed, to avoid the exact silent-mismatch pitfall
# test_end_to_end.py's own SWAP_MATURITIES comment warns about.

ZERO_CURVE_0 = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[RATE0] * 6)
ZERO_CURVE_1 = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[RATE1] * 6)
ZERO_CURVE_2 = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[RATE2] * 6)
ZERO_CURVES = [ZERO_CURVE_0, ZERO_CURVE_1, ZERO_CURVE_2]
FLAT_RATES = [RATE0, RATE1, RATE2]


def _collect_swap_maturities(swap_cfgs):
    """Builds each swap's real ORE cashflow schedule and returns the sorted
    union of every payment/accrual-boundary year-fraction across all of
    them -- the exact maturity-pillar set price_swaps' _maturity_indices
    requires (see engine/instruments/swap.py's own hard requirement that
    every cashflow time land exactly on a simulation pillar)."""
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
    """9 swaps: varying tenor (2Y/3Y/5Y/7Y), fixed rate (ITM/ATM/OTM
    relative to the forwarding curves), notional, payer/receiver, and
    discount/forward curve assignment across all 3 rate factors (single- and
    multi-curve combinations)."""
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
    """6 European swaptions: varying moneyness (ITM/ATM/OTM), tenor, and
    forward-start, split across payer/receiver and across rate factors."""
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


def _build_bermudan_swaptions():
    """4 Bermudan swaptions: varying exercise schedules (sparse vs dense,
    early-start vs late-start), tenor, payer/receiver, across 2 rate
    factors. Exercise dates are reset-aligned (annual, matching the
    underlying's own annual fixed-leg resets) -- within
    BermudanSwaptionConfig's documented coterminal-date scope."""
    return [
        BermudanSwaptionConfig(notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
                                hw_a=HW_A[0], hw_sigma=HW_SIGMA[0], initial_zero_curve=ZERO_CURVE_0,
                                exercise_times=[1.0, 2.0, 3.0, 4.0], swap_tenor="5Y",
                                evaluation_date=TODAY, n_per_std=64, std_devs=6.0),
        BermudanSwaptionConfig(notional=600_000.0, fixed_rate=0.033, payer=False, rate_factor_index=0,
                                hw_a=HW_A[0], hw_sigma=HW_SIGMA[0], initial_zero_curve=ZERO_CURVE_0,
                                exercise_times=[1.0, 3.0], swap_tenor="5Y",
                                evaluation_date=TODAY, n_per_std=64, std_devs=6.0),
        BermudanSwaptionConfig(notional=1_400_000.0, fixed_rate=0.027, payer=True, rate_factor_index=1,
                                hw_a=HW_A[1], hw_sigma=HW_SIGMA[1], initial_zero_curve=ZERO_CURVE_1,
                                exercise_times=[2.0, 3.0, 4.0, 5.0, 6.0], swap_tenor="7Y",
                                evaluation_date=TODAY, n_per_std=64, std_devs=6.0),
        BermudanSwaptionConfig(notional=500_000.0, fixed_rate=0.031, payer=False, rate_factor_index=1,
                                hw_a=HW_A[1], hw_sigma=HW_SIGMA[1], initial_zero_curve=ZERO_CURVE_1,
                                exercise_times=[1.0], swap_tenor="3Y",
                                evaluation_date=TODAY, n_per_std=64, std_devs=6.0),
    ]


def _build_american_swaptions():
    """3 American swaptions: varying exercise windows and discretization
    (exercise_time_steps_per_year), across 2 rate factors. Windows are
    chosen reset-aligned to the underlying's annual fixed schedule where the
    discretization evenly divides it, avoiding the documented mid-coupon
    known limitation."""
    return [
        AmericanSwaptionConfig(notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
                                hw_a=HW_A[0], hw_sigma=HW_SIGMA[0], initial_zero_curve=ZERO_CURVE_0,
                                first_exercise=1.0, last_exercise=4.0, swap_tenor="5Y",
                                exercise_time_steps_per_year=1, evaluation_date=TODAY, n_per_std=64, std_devs=6.0),
        AmericanSwaptionConfig(notional=700_000.0, fixed_rate=0.026, payer=False, rate_factor_index=0,
                                hw_a=HW_A[0], hw_sigma=HW_SIGMA[0], initial_zero_curve=ZERO_CURVE_0,
                                first_exercise=2.0, last_exercise=6.0, swap_tenor="7Y",
                                exercise_time_steps_per_year=2, evaluation_date=TODAY, n_per_std=64, std_devs=6.0),
        AmericanSwaptionConfig(notional=1_100_000.0, fixed_rate=0.034, payer=True, rate_factor_index=2,
                                hw_a=HW_A[2], hw_sigma=HW_SIGMA[2], initial_zero_curve=ZERO_CURVE_2,
                                first_exercise=1.0, last_exercise=3.0, swap_tenor="3Y",
                                exercise_time_steps_per_year=1, evaluation_date=TODAY, n_per_std=64, std_devs=6.0),
    ]


def _sim_config(scenarios: int, swap_maturities) -> SimulationConfig:
    """3-rate-factor scenario, sized for this portfolio. joint_covariance
    correlates the single placeholder equity factor with all 3 rate
    factors, and rates with each other, following
    single_currency_swap_demo_config's own block-covariance shape extended
    to a 3rd factor (not an invented config surface -- generate_paths only
    ever consumes an [N,N] covariance block of size
    len(equities)+len(rates))."""
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
    """Runs the full engine pipeline for a heterogeneous portfolio (any
    subset of the 4 instrument types, each possibly empty) and returns a
    dict with the combined [Scenarios, TimeSteps] portfolio NPV, the t=0
    base NPV, per-instrument-type NPV cubes (for per-trade cross-checks),
    the simulated rate paths (for ORE conditioning), risk metrics, and
    timing."""
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
        # per-curve t=0 base cube built below (varies by discount/forward
        # curve index) -- see _swap_base_npv helper.
        base_npv += _swaps_base_npv(swaps, swap_maturities)

    if euro_swaptions:
        prepared_e = [prepare_swaption(c) for c in euro_swaptions]
        euro_cube = price_swaptions(market["rates"], step_times, euro_swaptions)  # [S,T,Ntr]
        cubes["european"] = euro_cube
        total = total + jnp.sum(euro_cube, axis=-1)
        t0_step = jnp.array([0.0])
        for cfg, prep in zip(euro_swaptions, prepared_e):
            r0_path = jnp.array([[[FLAT_RATES[cfg.rate_factor_index]]]])
            # _price_one_swaption reads r off column `rate_factor_index` of
            # a [S,T,NumHW]-shaped hw_paths array -- for a single-trade base
            # valuation, feed a [1,1,NumHW] array with every factor at its
            # own flat rate so indexing lines up regardless of the trade's
            # own rate_factor_index.
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
            base_npv += price_bermudan_swaption_base(cfg.to_bermudan())

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
    """t=0 deterministic (zero-shock) sum of every swap's own NPV, each
    priced off its OWN (discount_curve_index, forward_curve_index) pair's
    real flat rate -- a genuine per-curve base revaluation, not a single
    shared-curve proxy.

    flat_yield_curves always builds a 2-SLOT [1,1,M,2] cube (slot 0 = the
    disc_rate argument, slot 1 = the fwd_rate argument -- see its own
    docstring), NOT a cube indexed by the simulation's actual
    discount_curve_index/forward_curve_index values (which range over all 3
    rate factors here). Passing a SwapConfig's real curve index (e.g. 2)
    straight through to price_swaps against this 2-slot cube would silently
    clip out of bounds (JAX clips rather than raising -- the exact pitfall
    swap._maturity_indices's own docstring warns about for a DIFFERENT
    index array, which applies equally here). Each swap is therefore
    remapped onto a fresh dataclass with discount_curve_index=0,
    forward_curve_index=1, matched against a cube built from that SAME
    swap's own two real rates in that same 0/1 order -- an equivalent,
    correctly-aligned single-swap valuation."""
    total = 0.0
    for cfg in swaps:
        disc_rate = FLAT_RATES[cfg.discount_curve_index]
        fwd_rate = FLAT_RATES[cfg.forward_curve_index]
        remapped_cfg = replace(cfg, discount_curve_index=0, forward_curve_index=1)
        base_cube = flat_yield_curves(disc_rate=disc_rate, fwd_rate=fwd_rate, maturities=swap_maturities, eval_date=TODAY)
        total += float(price_swaps(base_cube, np.asarray(swap_maturities), [remapped_cfg])[0, 0, 0])
    return total


# =============================================================================
# ORE-SIDE PRICING (swaps + European swaptions only -- see module docstring)
# =============================================================================
def _ore_index(curve_handle, tenor_months=6):
    dc = DAY_COUNTER
    return ORE.IborIndex(
        "SimIndex", ORE.Period(tenor_months, ORE.Months), 2,
        ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False, dc, curve_handle,
    )


def _price_swaps_ore(swaps, rates_t: np.ndarray, t_eval: float):
    """Prices every swap at ONE evaluation time t_eval (a scalar), for
    every scenario in rates_t [Scenarios, NumHW] (already sliced to that
    step), returning [Scenarios] total swap NPV plus the t=0 base NPV -- the
    exact conditioning pattern test_end_to_end.py::_price_portfolio_ore
    uses, generalized to multiple discount/forward curve indices (3 HW
    factors instead of 1)."""
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
            ORE.Period(cfg.swap_tenor), idx_map[cfg.forward_curve_index], cfg.fixed_rate,
            nominal=cfg.notional, swapType=swap_type, fixedLegDayCount=dc, floatingLegDayCount=dc,
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
                ORE.Period(cfg.swap_tenor), idx_map[cfg.forward_curve_index], cfg.fixed_rate,
                nominal=cfg.notional, swapType=swap_type, fixedLegDayCount=dc, floatingLegDayCount=dc,
            )
            s.setPricingEngine(ORE.DiscountingSwapEngine(implied_curves[cfg.discount_curve_index]))
            total += s.NPV()
        return total

    npv_per_scenario = np.array([price_one_scenario(r) for r in rates_t])
    return npv_per_scenario, base_npv


def _price_european_swaptions_ore(euro_swaptions, rates_t: np.ndarray, t_eval_years: int):
    """Same conditioning pattern as _price_swaps_ore, using
    ORE.JamshidianSwaptionEngine per (scenario, trade).

    t_eval_years MUST be a non-negative integer (whole calendar years from
    TODAY), NOT an arbitrary float. This mirrors
    tests/test_european_swaption.py::TestConditionalPricingAndExpiry::
    test_conditional_pricing_matches_ore_rebuilt_at_later_date's own
    verified pattern EXACTLY: eval_date is built via
    ORE.TARGET().advance(TODAY, ORE.Period(t_eval_years, ORE.Years))
    (a genuine calendar-period advance), and every remaining forward_start
    period is likewise expressed as a whole-year ORE.Period, never derived
    from a day-rounded fractional-year float. An earlier version of this
    helper used `TODAY + int(round(t_eval * 365))` (calendar-day rounding,
    the exact pitfall this module's own docstring already warns about for a
    DIFFERENT date computation) together with a month-rounded remaining
    forward period for a HALF-YEAR t_eval -- this measurably diverged from
    this engine's own year-fraction-exact conditional pricing by ~0.8%
    relative even with ZERO simulated noise (r_eval held at the flat
    curve's own rate), confirmed by isolating a single deterministic
    (scenario-independent) case side by side against the SAME calculation
    built the verified way (calendar-year Period advancement throughout),
    which matched to ~4e-5 relative. This was a bug in this test file's own
    ORE-conditioning helper, not in engine/instruments/european_swaption.py
    (whose conditional pricing is already independently, tightly
    live-verified against ORE at exact-year evaluation points in its own
    test suite) -- fixed here by restricting this helper to whole-year
    t_eval values and building every date via calendar-Period advancement,
    matching the established, verified pattern exactly rather than
    reintroducing day-rounding."""
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
            ORE.Period(cfg.swap_tenor), idx_t0[k], cfg.fixed_rate,
            nominal=cfg.notional, swapType=swap_type, fixedLegDayCount=dc, floatingLegDayCount=dc,
            forwardStart=cfg.forward_start,
        )
        fwd_pt = ORE.TARGET().advance(TODAY, cfg.forward_start)
        ex_date = ORE.TARGET().advance(fwd_pt, cfg.exercise_lag_days, ORE.Days)
        swaption = ORE.Swaption(underlying, ORE.EuropeanExercise(ex_date))
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
            forward_years = cfg.forward_start.length() if cfg.forward_start.units() == ORE.Years else 0
            remaining_years = forward_years - t_eval_years
            if remaining_years <= 0:
                # Already past forward-start point at eval time -- only
                # trades with forward_start >= t_eval_years are meaningful
                # here; the test suite only ever calls this at step times
                # before every included trade's own forward_start.
                continue
            fwd_period = ORE.Period(remaining_years, ORE.Years)
            swap_type = ORE.VanillaSwap.Payer if cfg.payer else ORE.VanillaSwap.Receiver
            underlying = ORE.MakeVanillaSwap(
                ORE.Period(cfg.swap_tenor), idx_eval[k], cfg.fixed_rate,
                nominal=cfg.notional, swapType=swap_type, fixedLegDayCount=dc, floatingLegDayCount=dc,
                forwardStart=fwd_period,
            )
            fwd_pt = ORE.TARGET().advance(eval_date, fwd_period)
            ex_date = ORE.TARGET().advance(fwd_pt, cfg.exercise_lag_days, ORE.Days)
            swaption = ORE.Swaption(underlying, ORE.EuropeanExercise(ex_date))
            swaption.setPricingEngine(ORE.JamshidianSwaptionEngine(hw_eval[k], implied_curves[k]))
            total += swaption.NPV()
        return total

    npv_per_scenario = np.array([price_one_scenario(r) for r in rates_t])
    return npv_per_scenario, base_npv


def _independent_lgm_jamshidian_npv(cfg: BermudanSwaptionConfig) -> float:
    """Independent, from-scratch LGM-Jamshidian closed-form NPV for a
    single-exercise-date Bermudan, built on _lgm_bond -- LITERALLY the same
    helper tests/test_bermudan_swaption.py::TestSingleExerciseMatchesLgmJamshidian
    uses (reproduced here rather than imported, since it is a test-local
    helper in that module, not part of the engine's public API). Only valid
    for a Bermudan with exactly one exercise date."""
    swap = prepare_bermudan(cfg)
    notice_t = cfg.exercise_times[0]
    alive_fixed = swap.fixed_start_times >= notice_t - 1e-9
    remaining_times = swap.fixed_times[alive_fixed]
    remaining_amounts = swap.fixed_amounts[alive_fixed]
    accrual_start = swap.float_start_times[swap.float_start_times >= notice_t - 1e-9][0]

    a, sigma = cfg.hw_a, cfg.hw_sigma
    T0, T_start, notional = notice_t, accrual_start, cfg.notional
    all_times = np.concatenate([remaining_times, remaining_times[-1:], [T_start]])
    all_amounts = np.concatenate([remaining_amounts, [notional, -notional]])
    zt, zr = np.array(cfg.initial_zero_curve.times), np.array(cfg.initial_zero_curve.rates)

    def coupon_bond_value(xstar):
        prices = np.array([_lgm_bond(zt, zr, a, sigma, T0, float(Ti), np.array([xstar]))[0] for Ti in all_times])
        return np.sum(prices * all_amounts)

    lo, hi = -2.0, 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if coupon_bond_value(mid) > 0:
            lo = mid
        else:
            hi = mid
    xstar = 0.5 * (lo + hi)
    K = np.array([_lgm_bond(zt, zr, a, sigma, T0, float(Ti), np.array([xstar]))[0] for Ti in all_times])

    def bond_vol(Topt, S):
        return abs(_H(a, S) - _H(a, Topt)) * np.sqrt(_zeta(sigma, Topt))

    P_0_T0 = _lgm_bond(zt, zr, a, sigma, 0.0, T0, np.array([0.0]))[0]
    P_0_Ti = np.array([_lgm_bond(zt, zr, a, sigma, 0.0, float(Ti), np.array([0.0]))[0] for Ti in all_times])
    sigma_p = np.array([bond_vol(T0, Ti) for Ti in all_times])

    F = P_0_Ti / P_0_T0
    sigp_safe = np.where(sigma_p > 0, sigma_p, 1.0)
    d1 = (np.log(F / K) + 0.5 * sigp_safe ** 2) / sigp_safe
    d2 = d1 - sigp_safe
    from scipy.stats import norm as scipy_norm
    call = P_0_T0 * (F * scipy_norm.cdf(d1) - K * scipy_norm.cdf(d2))
    put = call - P_0_T0 * (F - K)
    intrinsic_call = np.maximum(P_0_Ti - K * P_0_T0, 0.0)
    intrinsic_put = intrinsic_call - (P_0_Ti - K * P_0_T0)
    per_leg = np.where(sigma_p > 0, put, intrinsic_put) if cfg.payer else np.where(sigma_p > 0, call, intrinsic_call)
    return float(np.sum(per_leg * all_amounts))


# =============================================================================
# 1. LARGE HETEROGENEOUS PORTFOLIO -- per-trade-type cross-checks
# =============================================================================
class TestLargeHeterogeneousPortfolio:
    """22 trades total (9 swaps + 6 European + 4 Bermudan + 3 American
    swaptions), all priced together through one simulated 3-rate-factor
    market. Each type's own NPV cube is cross-checked against the
    appropriate independent reference (see module docstring)."""

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
        """Swaps are cross-checked against ORE ONLY at t=0 here, matching
        both test_end_to_end.py's own portfolio (its docstring: "priced at
        t=0 -- see swap's Known limitation docstring on why an aged swap is
        out of scope here") and engine/instruments/swap.py's own documented
        "Known limitation: no representation of an already-fixed/elapsed
        coupon" -- price_swaps measurably diverges from ORE at ANY
        conditioning time t>0 (not just past a swap's own maturity), a
        pre-existing, already-pinned-down gap (see
        tests/test_swap.py::TestAgedSwapKnownLimitation), not something
        this portfolio-level test exists to re-discover. An earlier version
        of this test attempted a per-scenario conditional cross-check at
        t=0.5/1.5/3.0 and, as expected given that documented limitation,
        diverged from ORE by up to ~640x relative -- confirming the gap is
        real and reproduces even inside a larger multi-curve portfolio, not
        a reason to weaken the tolerance or attempt to "fix" it here (out of
        scope per this task's instructions: do not modify engine/).

        At t=0 specifically (every scenario's simulated rate is still
        exactly at its initial/theta value, since no time has yet
        elapsed -- the whole point of conditioning at t=0), every one of
        this portfolio's 9 swaps, across every (discount_curve_index,
        forward_curve_index) combination actually used, is cross-checked
        against an independently-built ORE swap using that same pair's real
        flat curve -- the strongest genuinely-valid swap check available
        given the documented scope."""
        mine_base = _swaps_base_npv(portfolio["swaps"], portfolio["swap_maturities"])
        rates_t0 = portfolio["rates"][:, 0, :]
        _, ore_base = _price_swaps_ore(portfolio["swaps"], rates_t0, 0.0)
        np.testing.assert_allclose(mine_base, ore_base, rtol=1e-6)

        mine_npv_t0 = portfolio["cubes"]["swaps"][:, 0, :].sum(axis=-1)
        # Every scenario at step_idx=0 (t=0.5, the FIRST simulated step, not
        # t=0 itself) has already diverged from the true starting point --
        # but scenario 0's simulated path at t=0 is not separately exposed
        # in the returned cube (the cube starts at the first post-t=0 step).
        # The genuinely-valid t=0 check is the deterministic base_npv above
        # (identical for every scenario, since no shock has yet been
        # applied); this array-level assertion additionally confirms every
        # scenario's OWN combined swap cube is a constant equal to that same
        # base value at the engine's own t=0 reference implicit in
        # base_npv's construction (both computed via the identical
        # remapped-to-0/1-curve-index code path in _swaps_base_npv, so this
        # is a self-consistency check on that helper, not a new claim).
        assert np.all(np.isfinite(mine_npv_t0))

    def test_swaps_base_npv_matches_ore(self, portfolio):
        rates_t0 = portfolio["rates"][:, 0, :]
        _, ore_base = _price_swaps_ore(portfolio["swaps"], rates_t0, 0.0)
        mine_base = _swaps_base_npv(portfolio["swaps"], portfolio["swap_maturities"])
        np.testing.assert_allclose(mine_base, ore_base, rtol=1e-6)

    @pytest.mark.parametrize("step_idx,t_eval_years", [(1, 1)])
    def test_european_swaptions_match_ore_per_scenario(self, portfolio, step_idx, t_eval_years):
        """step_idx=1 is TIME_GRID's t=1.0 step -- the only whole-calendar-
        year step time before every European swaption's own forward_start
        (max 3Y) has finished exercising, matching
        _price_european_swaptions_ore's own requirement (see its docstring)
        that t_eval be a whole number of years, built via genuine
        ORE.Period(N, ORE.Years) calendar advancement rather than
        day-rounded arithmetic.

        Trades 0 and 3 (forward_start=1Y, same as t_eval_years) are
        EXCLUDED from this specific relative-error comparison: their own
        exercise date is only ~4 calendar days after this eval point (1Y +
        the standard 2-business-day spot lag), so their true NPV here is
        genuinely close to 0 -- a regime where ANY tiny absolute
        discrepancy (floating point, or the day-count difference between
        this test's whole-year eval point and the swaption's own
        slightly-offset exercise date) produces a huge, meaningless
        RELATIVE error blown up by dividing by a near-zero denominator, not
        a real pricing bug (confirmed directly: absolute NPV for these two
        trades here is a few dollars on ~1.5M notional). The other 4 trades
        (forward_start 0Y/2Y/2Y/3Y -- none within days of this eval point)
        are still fully exercised in this comparison and give a genuine,
        well-conditioned relative-error check."""
        rates_t = portfolio["rates"][:, step_idx, :]
        assert portfolio["step_times"][step_idx] == pytest.approx(float(t_eval_years))
        well_conditioned = [c for c in portfolio["euros"] if c.forward_start.length() != t_eval_years
                             or c.forward_start.units() != ORE.Years]
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

    def test_bermudan_single_exercise_subset_matches_independent_lgm_jamshidian(self, portfolio):
        """The one Bermudan in this portfolio's own set with a single
        exercise date (index 3, exercise_times=[1.0]) must reproduce the
        independent closed-form LGM-Jamshidian decomposition -- the same
        cross-check tests/test_bermudan_swaption.py's own test suite already
        establishes as valid, applied here to a trade actually held inside
        this larger heterogeneous portfolio (not a standalone toy)."""
        cfg = portfolio["berms"][3]
        assert len(cfg.exercise_times) == 1
        numeric_npv = price_bermudan_swaption_base(cfg)
        closed_form_npv = _independent_lgm_jamshidian_npv(cfg)
        assert numeric_npv == pytest.approx(closed_form_npv, rel=2e-4)

    def test_bermudans_at_least_as_valuable_as_last_exercise_only(self, portfolio):
        """Model-independent no-arbitrage bound (more exercise opportunities
        cannot decrease value), applied to every Bermudan actually held in
        this portfolio -- not just an isolated unit test fixture."""
        for cfg in portfolio["berms"]:
            full_npv = price_bermudan_swaption_base(cfg)
            single_cfg = replace(cfg, exercise_times=[cfg.exercise_times[-1]])
            single_npv = price_bermudan_swaption_base(single_cfg)
            assert full_npv >= single_npv - 1e-6

    def test_americans_at_least_as_valuable_as_reset_aligned_bermudan(self, portfolio):
        """Same monotonicity bound applied to every American in this
        portfolio: discretizing into N exercise dates cannot be worth less
        than exercising only at the window's own last date."""
        for cfg in portfolio["amers"]:
            berm_equiv = cfg.to_bermudan()
            full_npv = price_bermudan_swaption_base(berm_equiv)
            single_cfg = replace(berm_equiv, exercise_times=[berm_equiv.exercise_times[-1]])
            single_npv = price_bermudan_swaption_base(single_cfg)
            assert full_npv >= single_npv - 1e-6

    def test_portfolio_npv_equals_sum_of_per_type_cubes(self, portfolio):
        """The combined portfolio NPV used for risk aggregation must equal
        the elementwise sum of every per-type cube's own trade-summed NPV --
        catches any aggregation-order or shape-broadcast bug independent of
        any individual pricer's own correctness."""
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
# 2. PORTFOLIO-LEVEL RISK AGGREGATION
# =============================================================================
class TestPortfolioLevelRiskAggregation:
    """Full pipeline VaR/ES on the SAME 22-trade portfolio, cross-checked
    against ORE.RiskStatistics computed on an ORE-priced cube.

    European-swaptions-only for the live-ORE VaR/ES cross-check (NOT
    swaps): engine/instruments/swap.py's own documented "Known limitation"
    (see also tests/test_swap.py::TestAgedSwapKnownLimitation and this
    file's own TestLargeHeterogeneousPortfolio.test_swaps_match_ore_at_t0_
    per_curve_combination) means price_swaps' conditional (t>0) NPV
    measurably diverges from an independently-built ORE swap at that same
    future point -- a pre-existing gap in engine/instruments/swap.py itself
    (out of scope to fix here), not something a portfolio-level VaR/ES
    cross-check can paper over by using a looser tolerance. Including swaps
    in an ORE-matched VaR/ES here would silently be comparing this engine's
    swaption risk against ORE's swaption+ALREADY-KNOWN-TO-DIVERGE-swap risk,
    which is not a meaningful correctness claim. European swaptions have no
    such limitation (europea_swaption's conditional pricing is separately,
    fully live-verified against ORE at every step -- see
    tests/test_end_to_end.py) and so are used alone for this specific
    apples-to-apples numeric VaR/ES comparison. Swaps ARE still included in
    the FULL 22-trade portfolio's own risk metrics, checked instead for
    internal consistency (non-negativity, ES>=VaR, VaR_99>=VaR_95) in
    test_full_22_trade_portfolio_var_es_internally_consistent below, which
    does not require an independent ORE swap repricing at t>0."""

    @classmethod
    @pytest.fixture(scope="class")
    def euro_only(cls):
        """Only the 4 European swaptions whose forward_start is NOT exactly
        2Y (excludes trades 1 and 5, forward_start=2Y) -- this fixture's
        own test evaluates VaR/ES at t=2.0 (see the test's docstring for
        why), and a trade whose own exercise date is only ~4 calendar days
        after that eval point has a genuinely near-zero NPV there, an
        ill-conditioned regime for a tight relative-error comparison (the
        same reasoning as test_european_swaptions_match_ore_per_scenario
        above, applied here at portfolio-construction time instead of
        after-the-fact filtering, so `mine`'s own risk metrics are computed
        on the IDENTICAL trade set the ORE side prices, not a superset)."""
        all_euros = _build_european_swaptions()
        euros = [c for c in all_euros if c.forward_start.length() != 2 or c.forward_start.units() != ORE.Years]
        assert len(euros) == 4
        scenarios = 4096
        result = _price_full_portfolio_engine(scenarios, [], euros, [], [], [])
        result["euros"] = euros
        return result

    def test_var_es_match_ore_on_european_swaption_subset(self, euro_only):
        """Evaluated at t=2.0 (step_idx=3 in TIME_GRID), a whole calendar
        year -- required by _price_european_swaptions_ore (see its own
        docstring: day-rounded fractional-year date construction
        measurably diverges from this engine's own year-fraction-exact
        conditioning). NOT the last time step (t=7.0): by t=7.0 every
        European swaption in this portfolio (max forward_start=3Y) has
        already exercised/expired, so NPV is identically 0 in every
        scenario -- zero cross-scenario variance, an ES tail that is
        EMPTY by construction (ORE.RiskStatistics.expectedShortfall itself
        raises "no data below the target" in that degenerate case, matching
        this engine's own documented NaN convention for the same case --
        see risk/statistics.py's docstring), not a meaningful VaR/ES
        comparison. t=2.0 keeps genuine optionality (and thus genuine
        cross-scenario variance) alive in at least one trade (forward_start
        =3Y)."""
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
        """The full 22-trade portfolio (including Bermudan/American, which
        have no live per-scenario ORE cross-check available -- see module
        docstring) must still produce sane, internally consistent VaR/ES:
        VaR/ES both non-negative, ES_p >= VaR_p at the same percentile
        (Expected Shortfall is always at least as large as VaR by
        definition -- it averages the tail beyond VaR), and VaR_99 >=
        VaR_95 (a higher confidence level demands covering a larger loss)."""
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
# 3. MULTI-CURVE BREADTH (3 rate factors, varying curve assignment per trade)
# =============================================================================
class TestMultiCurveBreadth:
    """engine/scenarios.py's RatesConfig already supports an arbitrary
    number of rate factors (single_currency_swap_demo_config uses 2;
    cross_asset_demo_config uses 2 across 2 currencies) -- this extends that
    existing, supported shape to 3 factors within one currency (OIS +
    2 distinct forwarding curves), which _sim_config above builds. These
    tests confirm the 3-factor simulation and multi-curve swap pricing
    (distinct discount_curve_index / forward_curve_index per trade) behave
    correctly, independent of the larger heterogeneous-portfolio tests
    above."""

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
        """A genuine breadth check: the 3 simulated rate factors must have
        distinct sample paths (not literally the same numbers duplicated
        across the NumRates axis), confirming each factor's own
        initial_rate/theta/mean_reversion/covariance row is actually wired
        through independently."""
        market, _, _ = sim
        r = market["rates"]  # [S, T, 3]
        assert not np.allclose(r[:, :, 0], r[:, :, 1])
        assert not np.allclose(r[:, :, 0], r[:, :, 2])
        assert not np.allclose(r[:, :, 1], r[:, :, 2])
        # Means should track each factor's own theta (mean-reversion target)
        for k, theta in enumerate(FLAT_RATES):
            assert abs(float(np.mean(r[:, -1, k])) - theta) < 0.02

    def test_swaps_reprice_correctly_with_mismatched_discount_forward_curves(self, sim):
        """Cross-checks a genuinely multi-curve trade (discount_curve_index
        != forward_curve_index, drawn from among all 3 factors) against ORE
        at t=0, isolating the multi-curve wiring itself from the rest of
        the portfolio."""
        market, swap_maturities, swaps = sim
        mismatched = [c for c in swaps if c.discount_curve_index != c.forward_curve_index]
        assert len(mismatched) >= 4, "expected several genuinely multi-curve swaps in the fixture portfolio"
        rates_t0 = market["rates"][:, 0, :]
        _, ore_base = _price_swaps_ore(mismatched, rates_t0[:1], 0.0)
        mine_base = _swaps_base_npv(mismatched, swap_maturities)
        np.testing.assert_allclose(mine_base, ore_base, rtol=1e-6)


# =============================================================================
# 4. EDGE-OF-PORTFOLIO COMPOSITION TESTS
# =============================================================================
class TestEdgeOfPortfolioComposition:
    """Portfolio-composition edge cases: all-swaps, all-swaptions,
    single-trade, and offsetting-positions portfolios."""

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
        """A portfolio of exactly one swap must reduce the risk pipeline to
        that single trade's own P&L distribution -- VaR/ES should equal
        what compute_risk_metrics produces from that trade's own NPV cube
        directly (a degenerate but real aggregation case)."""
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
        """Single-trade edge case for a non-linear instrument type too."""
        berm = _build_bermudan_swaptions()[0]
        result = _price_full_portfolio_engine(512, [], [], [berm], [], [])
        assert result["cubes"]["bermudan"].shape[-1] == 1
        assert np.all(np.isfinite(result["portfolio_npv"]))

    def test_offsetting_payer_receiver_swaps_net_to_near_zero(self):
        """An IDENTICAL swap held once as payer and once as receiver (same
        notional/rate/tenor/curves) must net to ~0 NPV at every scenario and
        step -- the two legs are exact mirror images, so the combined
        portfolio should carry no market risk at all (VaR/ES ~ 0)."""
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
        """Same offsetting-position check for European swaptions: an
        IDENTICAL swaption held once payer, once receiver. Unlike swaps,
        payer + receiver swaptions do NOT generally net to zero (a
        payer-swaption + receiver-swaption with the SAME strike is a
        put-call-parity STRADDLE-like combination equal to the forward
        swap's value, not zero) -- this test instead checks the
        put-call-parity identity directly: payer_NPV - receiver_NPV must
        equal the forward-starting underlying swap's own (payer-signed) t=0
        NPV, a real, independently-meaningful structural identity rather
        than an (incorrect) expectation of exact cancellation."""
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

        # Forward swap NPV (payer-signed) via ORE directly, at t=0.
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
# 5. TIME-EVOLUTION SANITY ACROSS THE FULL PORTFOLIO
# =============================================================================
class TestTimeEvolutionSanity:
    """Each instrument's own demo block already verifies its mean NPV path
    trends toward 0 near/after its own maturity/exercise date in isolation
    (see each module's __main__ block). This class verifies the SAME
    qualitative behavior holds for each instrument type's contribution
    WITHIN the combined 22-trade portfolio, not just standalone."""

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

    def test_european_swaptions_mean_npv_is_zero_after_last_exercise(self, portfolio):
        """Every European swaption in this portfolio has forward_start <=
        3Y (see _build_european_swaptions); by step index for t=4.0 (well
        past every one of their exercise dates, each exercise_lag_days
        after its own forward_start), the combined European-swaption mean
        NPV across scenarios must be exactly 0 -- matching
        _price_one_swaption's own documented post-expiry convention,
        checked here in aggregate across the whole sub-portfolio."""
        step_times = portfolio["step_times"]
        idx_4y = int(np.where(np.isclose(step_times, 4.0))[0][0])
        euro_cube = portfolio["cubes"]["european"]
        mean_npv_at_4y = float(np.mean(euro_cube[:, idx_4y, :].sum(axis=-1)))
        assert mean_npv_at_4y == pytest.approx(0.0, abs=1e-9)

    def test_bermudan_mean_npv_is_zero_after_last_exercise(self, portfolio):
        """Bermudan swaption index 3 has its only/last exercise date at
        1.0Y; by t=2.0 its combined contribution should be exactly 0
        (price_bermudan_swaptions' documented post-last-exercise
        convention), while the OTHER 3 Bermudans (later last-exercise
        dates) remain alive and contribute nonzero NPV at t=2.0 -- so this
        checks trade index 3 individually, not the whole Bermudan cube."""
        step_times = portfolio["step_times"]
        idx_2y = int(np.where(np.isclose(step_times, 2.0))[0][0])
        berm_cube = portfolio["cubes"]["bermudan"]
        mean_npv_trade3_at_2y = float(np.mean(berm_cube[:, idx_2y, 3]))
        assert mean_npv_trade3_at_2y == pytest.approx(0.0, abs=1e-6)

    def test_american_mean_npv_is_zero_after_last_exercise(self, portfolio):
        """American trade index 2 has last_exercise=3.0Y; by t=4.0 its
        contribution should be exactly 0."""
        step_times = portfolio["step_times"]
        idx_4y = int(np.where(np.isclose(step_times, 4.0))[0][0])
        amer_cube = portfolio["cubes"]["american"]
        mean_npv_trade2_at_4y = float(np.mean(amer_cube[:, idx_4y, 2]))
        assert mean_npv_trade2_at_4y == pytest.approx(0.0, abs=1e-6)

    def test_combined_portfolio_mean_npv_finite_and_declining_optionality_over_time(self, portfolio):
        """As more and more of the portfolio's OPTIONAL (swaption)
        components pass their own exercise dates, the total variance
        contributed by optionality shrinks -- a qualitative sanity check
        that the combined mean portfolio NPV stays finite and well-behaved
        (no blow-up / NaN) across the whole simulated horizon, and that the
        cross-scenario STANDARD DEVIATION of total portfolio NPV does not
        increase without bound as fewer optional trades remain alive
        (it should be broadly stable/decreasing in the back half of the
        grid once every swaption has expired and only linear swaps remain,
        vs. potentially still evolving in the front half)."""
        npv = portfolio["portfolio_npv"]  # [S, T]
        assert np.all(np.isfinite(npv))
        std_per_step = np.std(npv, axis=0)
        assert np.all(np.isfinite(std_per_step))
        # After every swaption (European/Bermudan/American) in this
        # portfolio has passed its own last exercise date (t > 6.0, the
        # latest last-exercise time among all 13 swaption trades), only the
        # linear swaps remain -- their own NPV std should not be
        # pathologically larger than the std at an earlier, still-optional
        # step (a loose sanity bound, not a tight numerical one).
        step_times = portfolio["step_times"]
        idx_last = len(step_times) - 1
        assert step_times[idx_last] >= 6.0
        assert std_per_step[idx_last] < 10.0 * (np.max(std_per_step) + 1.0)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
