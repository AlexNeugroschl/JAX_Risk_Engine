"""
End-to-end validation: prices a MIXED portfolio (interest rate swaps +
European swaptions) through this engine's full pipeline
(generate_paths -> price_swaps/price_swaptions -> compute_risk_metrics),
and independently through real ORE objects (ORE.DiscountingSwapEngine +
ORE.JamshidianSwaptionEngine + ORE.RiskStatistics) conditioned on the
EXACT SAME simulated short-rate values, then compares both NPVs and
VaR/ES numerically, and reports wall-clock timing for both paths.

Why "same simulated short-rate values" rather than two independent Monte
Carlo runs: the goal here is to validate PRICING/RISK FORMULA correctness
and PERFORMANCE at scale, not random-number-generator equivalence (this
engine uses Sobol quasi-random sequences + a Brownian bridge; ORE's own
path generators use a different RNG entirely -- comparing two independently
-generated samples would only ever be "close", never a precise correctness
check, and any observed difference would be ambiguous between "pricing bug"
and "different random draws"). Instead, this engine's own real
generate_paths() output supplies the short-rate path; ORE prices the exact
same trades conditional on those exact same rate values (via a
YieldTermStructureHandle built from ORE.HullWhite.discountBond(t, T, r) at
each simulated r) -- the same live-testing methodology already used
throughout engine/instruments/european_swaption.py's own test suite,
extended here to a full multi-trade portfolio and a real VaR/ES
aggregation.

A note on ORE date arithmetic (a real pitfall this test's own development
hit): `ORE.TARGET().advance(date, N, ORE.Days)` with a plain Days unit
treats N as BUSINESS days (skipping weekends/holidays via the TARGET
calendar), NOT calendar days -- `date + N` (plain integer addition on an
ORE.Date) is calendar-day arithmetic. Conflating the two silently produces
a "1 year forward" date that's actually ~1.4 calendar years out, which
looks exactly like a pricing bug (a growing, non-random NPV divergence)
until traced back to the date construction itself. This module always uses
`date + N` for calendar-day offsets and `ORE.Period(N, ORE.Years/...)` for
calendar-period offsets, never the ambiguous integer-Days advance form.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.market_simulations import SimulationConfig, EquityConfig, RatesConfig, ZeroCurveConfig, generate_paths
from engine.instruments.interest_rate_swap import SwapConfig, price_swaps
from engine.instruments.european_swaption import SwaptionConfig, prepare_swaption, _price_one_swaption, price_swaptions
from engine.aggregate_statistics.risk_statistics import compute_risk_metrics
from engine.scenarios import flat_yield_curves

TODAY = ORE.Date(30, 7, 2026)
FLAT_RATE = 0.03
HW_A = 0.03
HW_SIGMA = 0.01
DAY_COUNTER = ORE.Actual365Fixed()

# The 2Y swap's own real cashflow/accrual-boundary dates (absolute
# year-fractions from TODAY) -- required maturity pillars for
# flat_yield_curves' t=0 deterministic revaluation.
SWAP_MATURITIES = [
    0.010958904109589041, 0.5150684931506849, 1.010958904109589,
    1.515068493150685, 2.0136986301369864,
]

ZERO_CURVE = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[FLAT_RATE] * 6)


def _build_portfolio():
    """One 2Y payer swap (priced at t=0 -- see interest_rate_swap's "Known
    limitation" docstring on why an aged swap is out of scope here) plus
    two forward-starting swaptions (a 5Y payer exercisable in 3Y, a 7Y
    receiver exercisable in 2Y) -- both genuinely still alive at the
    simulated t=1.0 evaluation point used throughout this file, so their
    conditional pricing is on the same solid ground the swaption module's
    own test suite already validates."""
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
    """Runs the full engine pipeline for the mixed portfolio and returns
    (portfolio_npv_at_t1 [Scenarios], base_npv, risk_metrics dict, r_t
    [Scenarios] the simulated short rates -- handed to the ORE path so
    both price conditional on the exact same numbers, timings dict)."""
    swap, swaption_a, swaption_b = _build_portfolio()
    config = _sim_config(scenarios)

    t_start = time.perf_counter()
    market = generate_paths(config)
    step_times = jnp.array(config.time_grid[1:])

    base_cube = flat_yield_curves(disc_rate=FLAT_RATE, fwd_rate=FLAT_RATE, maturities=SWAP_MATURITIES, eval_date=TODAY)
    swap_npv_t0 = float(price_swaps(base_cube, np.array(SWAP_MATURITIES), [swap])[0, 0, 0])

    swaption_npv = price_swaptions(market["rates"], step_times, [swaption_a, swaption_b])
    portfolio_at_t1 = swap_npv_t0 + jnp.sum(swaption_npv[:, 0, :], axis=-1)  # [Scenarios]

    # t=0 baseline: swap's own t=0 value + each swaption's own t=0 value
    # (a real, deterministic zero-shock revaluation of the whole
    # portfolio, not a proxy) -- see risk_statistics' P&L baseline
    # convention (docs/05-risk-statistics.md).
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
    """Prices the SAME portfolio, conditional on the SAME r_t values, using
    real ORE pricing engines throughout -- one implied curve rebuilt per
    scenario via ORE.HullWhite.discountBond(t_eval, T, r), exactly as
    engine/instruments/european_swaption.py's own conditional-pricing test
    does. Returns (portfolio_npv [Scenarios], base_npv, elapsed_seconds)."""
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

    # Exercise date = forward_start point + 2-business-day spot lag,
    # matching european_swaption.SwaptionConfig's own convention exactly
    # (see prepare_swaption's docstring) -- omitting the spot lag here
    # was an earlier version of this test's own bug, caught by this exact
    # base_npv cross-check diverging from the engine by ~0.1%.
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

        # forward_start is always relative to TODAY, not eval_date -- 1y
        # has already elapsed by eval_date, so only (3-1)=2y / (2-1)=1y
        # remain from eval_date's own perspective.
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
    """The full pipeline, cross-checked against ORE end-to-end: same
    simulated rate paths, same trades, priced independently by each side,
    compared on NPV and on the VaR/ES computed from them."""

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
        """Every single simulated scenario's portfolio NPV, not just the
        mean -- this is the strongest possible correctness claim: pathwise
        agreement across the entire distribution, not merely matching
        summary statistics that could hide offsetting errors."""
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
        """Not a pass/fail assertion on speed (that would make the test
        flaky across machines/load) -- just surfaces the measured timing so
        a human reviewing test output can see the actual tradeoff: this
        engine pays a fixed JIT-compilation/dispatch cost per call, so
        ORE's plain Python loop can win at small scenario counts, while the
        engine's vectorized pricing wins decisively as scenario count
        grows (see TestEndToEndScaling below for the crossover)."""
        print(
            f"\n[timing] {comparison['scenarios']} scenarios: "
            f"engine={comparison['engine_time']:.3f}s, ORE={comparison['ore_time']:.3f}s, "
            f"ratio={comparison['ore_time'] / comparison['engine_time']:.2f}x"
        )
        assert comparison["engine_time"] > 0.0
        assert comparison["ore_time"] > 0.0


class TestEndToEndScaling:
    """Demonstrates (does not merely assert) the actual performance
    crossover: at small scenario counts, this engine's fixed per-call
    overhead (JIT compilation, Python-level dispatch, JAX device transfer)
    can make ORE's plain Python per-scenario loop faster in absolute terms;
    at large scenario counts, this engine's vectorized tensor pricing wins.
    Reporting both regimes honestly (not cherry-picking a favorable scale)
    is the point of this test -- see its printed output.
    """

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
