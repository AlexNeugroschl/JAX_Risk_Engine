"""
End to end: a mixed portfolio (swaps and European swaptions) through `generate_paths`,
`price_swaps`/`price_swaptions` and `compute_risk_metrics`, and independently through ORE
(`DiscountingSwapEngine`, `JamshidianSwaptionEngine`, `RiskStatistics`) conditioned on the
same simulated short rates; NPVs and VaR/ES compared, with timings.

Sharing the simulated rates (rather than two independent Monte Carlo runs with different
generators) makes a difference a pricing difference, not sampling noise. ORE prices each
scenario on a curve implied from `ORE.HullWhite.discountBond(t, T, r)`.

Dates: `ORE.TARGET().advance(date, N, ORE.Days)` adds business days (365 of them is ~1.4
years); `date + N` adds calendar days. This file uses `date + N` and `ORE.Period`.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.simulation.market_model import SimulationConfig, EquityConfig, RatesConfig, ZeroCurveConfig, generate_paths
from engine.instruments.swap import SwapConfig, price_swaps
from engine.instruments.european_swaption import SwaptionConfig, prepare_swaption, _price_one_swaption, price_swaptions
from engine.risk.var_es import compute_risk_metrics
from demos.demo_scenarios import flat_yield_curves

TODAY = ORE.Date(30, 7, 2026)
FLAT_RATE = 0.03
HW_A = 0.03
HW_SIGMA = 0.01
DAY_COUNTER = ORE.Actual365Fixed()

# The 2Y swap's payment and accrual times (years from TODAY): the pillars for the t=0
# revaluation.
SWAP_MATURITIES = [
    0.010958904109589041, 0.5150684931506849, 1.010958904109589,
    1.515068493150685, 2.0136986301369864,
]

ZERO_CURVE = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[FLAT_RATE] * 6)


def _build_portfolio():
    """A 2Y payer swap (checked at t=0 only; aged swaps are I-04) and two forward-starting
    swaptions (5Y payer exercisable in 3Y, 7Y receiver in 2Y), both alive at t=1."""
    swap = SwapConfig(
        notional=1_000_000.0, fixed_rate=0.03, payer=True,
        discount_curve_index=0, forward_curve_index=0,
        swap_tenor="2Y", evaluation_date=TODAY,
    )
    swaption_a = SwaptionConfig(
        notional=1_500_000.0, fixed_rate=0.03, payer=True,
        rate_factor_index=0, hw_a=HW_A, hw_sigma=HW_SIGMA,
        initial_zero_curve=ZERO_CURVE, swap_tenor="5Y",
        forward_start=ORE.Period(3, ORE.Years), evaluation_date=TODAY,
    )
    swaption_b = SwaptionConfig(
        notional=800_000.0, fixed_rate=0.028, payer=False,
        rate_factor_index=0, hw_a=HW_A, hw_sigma=HW_SIGMA,
        initial_zero_curve=ZERO_CURVE, swap_tenor="7Y",
        forward_start=ORE.Period(2, ORE.Years), evaluation_date=TODAY,
    )
    return swap, swaption_a, swaption_b


def _sim_config(scenarios: int) -> SimulationConfig:
    return SimulationConfig(
        time_grid=[0.0, 1.0],
        scenarios=scenarios,
        equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[0.0]]),
        rates=RatesConfig(initial_rates=[FLAT_RATE], theta=[FLAT_RATE], mean_reversion=[HW_A]),
        joint_covariance=[[0.0400, 0.0000], [0.0000, HW_SIGMA ** 2]],
    )


def _price_portfolio_engine(scenarios: int):
    """The engine pipeline: (portfolio NPV at t=1 [S], base NPV, risk metrics, simulated
    short rates r_t [S] for the ORE side, timings)."""
    swap, swaption_a, swaption_b = _build_portfolio()
    config = _sim_config(scenarios)

    t_start = time.perf_counter()
    market = generate_paths(config)
    step_times = jnp.array(config.time_grid[1:])

    base_cube = flat_yield_curves(disc_rate=FLAT_RATE, fwd_rate=FLAT_RATE, maturities=SWAP_MATURITIES, eval_date=TODAY)
    swap_npv_t0 = float(price_swaps(base_cube, np.array(SWAP_MATURITIES), [swap])[0, 0, 0])

    swaption_npv = price_swaptions(market["rates"], step_times, [swaption_a, swaption_b])
    portfolio_at_t1 = swap_npv_t0 + jnp.sum(swaption_npv[:, 0, :], axis=-1)  # [Scenarios]

    # t=0 baseline: a deterministic revaluation of the whole portfolio (the P&L baseline;
    # docs/risk/var_es.md).
    prep_a = prepare_swaption(swaption_a)
    prep_b = prepare_swaption(swaption_b)
    t0_step = jnp.array([0.0])
    r0_path = jnp.array([[[FLAT_RATE]]])
    swaption_a_t0 = float(_price_one_swaption(r0_path, t0_step, prep_a)[0, 0])
    swaption_b_t0 = float(_price_one_swaption(r0_path, t0_step, prep_b)[0, 0])
    base_npv = swap_npv_t0 + swaption_a_t0 + swaption_b_t0

    npv_cube = portfolio_at_t1[:, None, None]  # [Scenarios, TimeSteps=1, Trades=1]
    metrics = compute_risk_metrics(npv_cube, base_npv, percentiles=(0.95, 0.99))
    elapsed = time.perf_counter() - t_start

    r_t = np.asarray(market["rates"][:, 0, 0])
    return np.asarray(portfolio_at_t1), base_npv, metrics, r_t, elapsed


def _price_portfolio_ore(r_t: np.ndarray):
    """The same portfolio on the same r_t by ORE's engines, one implied curve per scenario
    from `ORE.HullWhite.discountBond(t_eval, T, r)`. Returns (portfolio NPV [S], base NPV,
    seconds)."""
    dc = DAY_COUNTER
    ORE.Settings.instance().evaluationDate = TODAY
    curve0 = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, FLAT_RATE, dc))
    hw0 = ORE.HullWhite(curve0, HW_A, HW_SIGMA)

    idx0 = ORE.IborIndex(
        "SimIndex", ORE.Period(6, ORE.Months), 2,
        ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False, dc, curve0,
    )
    ore_swap = ORE.MakeVanillaSwap(
        ORE.Period("2Y"), idx0, 0.03, nominal=1_000_000.0,
        swapType=ORE.VanillaSwap.Payer, discountingTermStructure=curve0,
        fixedLegDayCount=dc, floatingLegDayCount=dc,
    )
    ore_swap.setPricingEngine(ORE.DiscountingSwapEngine(curve0))
    swap_npv_t0 = ore_swap.NPV()

        # Exercise date = forward-start point + 2 business days, as SwaptionConfig books it
        # (omitting the lag was once a ~0.1% bug in this test).
    fwd_a_t0 = ORE.TARGET().advance(TODAY, ORE.Period(3, ORE.Years))
    ex_a_t0 = ORE.EuropeanExercise(ORE.TARGET().advance(fwd_a_t0, ORE.Period(2, ORE.Days)))
    swap_a_t0 = ORE.MakeVanillaSwap(
        ORE.Period("5Y"), idx0, 0.03, nominal=1_500_000.0,
        swapType=ORE.VanillaSwap.Payer, fixedLegDayCount=dc, floatingLegDayCount=dc,
        forwardStart=ORE.Period(3, ORE.Years),
    )
    swaption_a_t0 = ORE.Swaption(swap_a_t0, ex_a_t0)
    swaption_a_t0.setPricingEngine(ORE.JamshidianSwaptionEngine(hw0, curve0))
    va_t0 = swaption_a_t0.NPV()

    fwd_b_t0 = ORE.TARGET().advance(TODAY, ORE.Period(2, ORE.Years))
    ex_b_t0 = ORE.EuropeanExercise(ORE.TARGET().advance(fwd_b_t0, ORE.Period(2, ORE.Days)))
    swap_b_t0 = ORE.MakeVanillaSwap(
        ORE.Period("7Y"), idx0, 0.028, nominal=800_000.0,
        swapType=ORE.VanillaSwap.Receiver, fixedLegDayCount=dc, floatingLegDayCount=dc,
        forwardStart=ORE.Period(2, ORE.Years),
    )
    swaption_b_t0 = ORE.Swaption(swap_b_t0, ex_b_t0)
    swaption_b_t0.setPricingEngine(ORE.JamshidianSwaptionEngine(hw0, curve0))
    vb_t0 = swaption_b_t0.NPV()

    base_npv = swap_npv_t0 + va_t0 + vb_t0

    t_eval = 1.0
    eval_date = TODAY + int(round(t_eval * 365))  # calendar days -- see module docstring
    curve_years = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def price_one_scenario(r: float) -> float:
        dates = [eval_date] + [TODAY + int(round((t_eval + y) * 365)) for y in curve_years]
        discounts = [1.0] + [hw0.discountBond(t_eval, t_eval + y, r) for y in curve_years]
        ORE.Settings.instance().evaluationDate = eval_date
        implied_curve = ORE.YieldTermStructureHandle(ORE.DiscountCurve(dates, discounts, dc))
        hw_eval = ORE.HullWhite(implied_curve, HW_A, HW_SIGMA)
        idx = ORE.IborIndex(
            "SimIndex", ORE.Period(6, ORE.Months), 2,
            ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False, dc, implied_curve,
        )

        # forward_start counts from TODAY; one year has passed at eval_date.
        swa = ORE.MakeVanillaSwap(
            ORE.Period("5Y"), idx, 0.03, nominal=1_500_000.0,
            swapType=ORE.VanillaSwap.Payer, fixedLegDayCount=dc, floatingLegDayCount=dc,
            forwardStart=ORE.Period(2, ORE.Years),
        )
        fwd_a = ORE.TARGET().advance(eval_date, ORE.Period(2, ORE.Years))
        exa = ORE.EuropeanExercise(ORE.TARGET().advance(fwd_a, ORE.Period(2, ORE.Days)))
        swpta = ORE.Swaption(swa, exa)
        swpta.setPricingEngine(ORE.JamshidianSwaptionEngine(hw_eval, implied_curve))
        va = swpta.NPV()

        swb = ORE.MakeVanillaSwap(
            ORE.Period("7Y"), idx, 0.028, nominal=800_000.0,
            swapType=ORE.VanillaSwap.Receiver, fixedLegDayCount=dc, floatingLegDayCount=dc,
            forwardStart=ORE.Period(1, ORE.Years),
        )
        fwd_b = ORE.TARGET().advance(eval_date, ORE.Period(1, ORE.Years))
        exb = ORE.EuropeanExercise(ORE.TARGET().advance(fwd_b, ORE.Period(2, ORE.Days)))
        swptb = ORE.Swaption(swb, exb)
        swptb.setPricingEngine(ORE.JamshidianSwaptionEngine(hw_eval, implied_curve))
        vb = swptb.NPV()

        return swap_npv_t0 + va + vb

    t_start = time.perf_counter()
    portfolio_npv = np.array([price_one_scenario(float(r)) for r in r_t])
    elapsed = time.perf_counter() - t_start

    return portfolio_npv, base_npv, elapsed


class TestEndToEndEngineVsORE:
    """Same rate paths, same trades, priced by each side; NPV and VaR/ES compared."""

    @classmethod
    @pytest.fixture(scope="class")
    def comparison(cls):
        scenarios = 8192
        mine_npv, mine_base, mine_metrics, r_t, engine_time = _price_portfolio_engine(scenarios)
        ore_npv, ore_base, ore_time = _price_portfolio_ore(r_t)
        return {
            "scenarios": scenarios,
            "mine_npv": mine_npv, "mine_base": mine_base, "mine_metrics": mine_metrics,
            "ore_npv": ore_npv, "ore_base": ore_base,
            "engine_time": engine_time, "ore_time": ore_time,
        }

    def test_t0_base_npv_matches_ore(self, comparison):
        np.testing.assert_allclose(comparison["mine_base"], comparison["ore_base"], rtol=1e-6)

    def test_per_scenario_npv_matches_ore(self, comparison):
        """Every scenario's NPV matches (pathwise, not just summary statistics)."""
        diff = comparison["mine_npv"] - comparison["ore_npv"]
        rel = np.abs(diff) / np.maximum(np.abs(comparison["ore_npv"]), 1.0)
        assert np.max(rel) < 1e-3
        assert np.mean(rel) < 1e-4

    def test_var_es_match_ore(self, comparison):
        ore_pnl = comparison["ore_npv"] - comparison["ore_base"]
        ore_stats = ORE.RiskStatistics()
        for v in ore_pnl:
            ore_stats.add(float(v), 1.0)

        mine = comparison["mine_metrics"]
        np.testing.assert_allclose(float(mine["VaR_95"][0]), ore_stats.valueAtRisk(0.95), rtol=1e-3)
        np.testing.assert_allclose(float(mine["VaR_99"][0]), ore_stats.valueAtRisk(0.99), rtol=1e-3)
        np.testing.assert_allclose(float(mine["ES_95"][0]), ore_stats.expectedShortfall(0.95), rtol=1e-3)
        np.testing.assert_allclose(float(mine["ES_99"][0]), ore_stats.expectedShortfall(0.99), rtol=1e-3)

    def test_reports_timing(self, comparison):
        """Prints the timings (no assertion): the engine has a fixed compile/dispatch cost, so
        ORE's loop can be faster at small scenario counts (see TestEndToEndScaling)."""
        print(
            f"\n[timing] {comparison['scenarios']} scenarios: "
            f"engine={comparison['engine_time']:.3f}s, ORE={comparison['ore_time']:.3f}s, "
            f"ratio={comparison['ore_time'] / comparison['engine_time']:.2f}x"
        )
        assert comparison["engine_time"] > 0.0
        assert comparison["ore_time"] > 0.0


class TestEndToEndScaling:
    """Timings at several scenario counts, printed: the engine's fixed overhead against its
    vectorized scaling."""

    @pytest.mark.slow
    @pytest.mark.parametrize("scenarios", [512, 32768])
    def test_timing_at_scale(self, scenarios):
        mine_npv, mine_base, mine_metrics, r_t, engine_time = _price_portfolio_engine(scenarios)
        ore_npv, ore_base, ore_time = _price_portfolio_ore(r_t)

        rel = np.abs(mine_npv - ore_npv) / np.maximum(np.abs(ore_npv), 1.0)
        assert np.max(rel) < 1e-3, "Accuracy must hold at every scale tested, not just the default."

        speedup = ore_time / engine_time
        print(
            f"\n[timing] {scenarios} scenarios: engine={engine_time:.3f}s, "
            f"ORE={ore_time:.3f}s, speedup={speedup:.2f}x "
            f"({'engine faster' if speedup > 1 else 'ORE faster (fixed-overhead regime)'})"
        )
