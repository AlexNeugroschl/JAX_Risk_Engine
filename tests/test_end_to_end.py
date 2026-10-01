"""
End to end against ORE on the same simulated paths: a portfolio (a forward-starting swap and
two forward-starting European swaptions on the Jamshidian engine) through `price_portfolio`
with the Hull-White model, and independently through QuantLib/ORE (`DiscountingSwapEngine`,
`JamshidianSwaptionEngine`, `RiskStatistics`) on each path's curve; NPVs and VaR/ES compared,
with timings.

Sharing the simulated states (rather than two Monte Carlo runs with different generators)
makes a difference a pricing difference, not sampling noise. Each path's state z at t = 1 is
mapped to its short rate, r = f(0, t) + H'(t) z + zeta(t) H(t) H'(t) (tests/test_cam.py
shows the simulated curve is then QuantLib's `HullWhite.discountBond(t, T, r)` to 1e-12), and
ORE prices on that path's curve as ORE's `ScenarioSimMarket` holds it: QuantLib's Hull-White
discount factors at the simulation-market tenors (`CamConfig.curve_tenors`), log-linear in
between (`ORE.DiscountCurve`). On a sloped curve (a flat one hides drift errors, I-42).

Dates: `date + N` adds calendar days; the simulation date is `TODAY + 365`, t = 1 exactly on
the ACT/365 time axis.
"""
import time

import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.instruments.european_swaption import SwaptionConfig, _build_ore_swap as european_underlying
from engine.instruments.swap import SwapConfig, _build_ore_swap as swap_underlying
from engine.market import CurrencyMarket, Market, ZeroCurveConfig, index_name
from engine.models.ore_builders import ibor_index
from engine.portfolio import (
    CamConfig, HullWhiteConfig, JamshidianEngineConfig, PortfolioRequest, PricingConfig, RunConfig, price_portfolio,
)
from engine.risk.var_es import compute_risk_metrics
from engine.simulation.config import DEFAULT_CURVE_TENORS, simulate

TODAY = ORE.Date(30, 7, 2026)
HW_A, HW_SIGMA = 0.03, 0.01
DC = ORE.Actual365Fixed()
PILLARS = [0.0, 1.0, 2.0, 5.0, 10.0, 30.0]
RATES = [0.030, 0.030, 0.034, 0.040, 0.046, 0.050]
CURVE = ZeroCurveConfig(PILLARS, RATES)
T_EVAL = 1.0
EVAL_DATE = TODAY + 365
PRICING = PricingConfig(european="Jamshidian", jamshidian=JamshidianEngineConfig(HW_A, HW_SIGMA))


def _trades():
    """A swap starting after the simulation date and two swaptions alive at it (5Y payer
    exercisable in 3Y, 7Y receiver in 2Y)."""
    swap = SwapConfig(notional=1_000_000.0, fixed_rate=0.036, payer=True, effective_date=ORE.Date(2, 8, 2028),
                      maturity_date=ORE.Date(2, 8, 2030), evaluation_date=TODAY, trade_id="swap")
    payer = SwaptionConfig(notional=1_500_000.0, fixed_rate=0.042, payer=True, swap_tenor="5Y",
                           forward_start=ORE.Period(3, ORE.Years), evaluation_date=TODAY, trade_id="payer")
    receiver = SwaptionConfig(notional=800_000.0, fixed_rate=0.036, payer=False, swap_tenor="7Y",
                              forward_start=ORE.Period(2, ORE.Years), evaluation_date=TODAY, trade_id="receiver")
    return [swap, payer, receiver]


def _request(samples: int) -> PortfolioRequest:
    # One curve for discounting and forwarding: QuantLib's Jamshidian engine is single-curve.
    market = Market(TODAY, {"USD": CurrencyMarket(CURVE, {index_name("USD", 6): CURVE})})
    simulation = CamConfig(dates=(EVAL_DATE,), base_currency="USD", ir={"USD": HullWhiteConfig(HW_A, HW_SIGMA)},
                           samples=samples, seed=17)
    return PortfolioRequest(market=market, trades=_trades(), config=RunConfig(simulation=simulation, pricing=PRICING),
                            pfe_quantiles=(0.95,))


def _price_engine(samples: int):
    """The engine: (per-trade NPVs at t = 1 [S, N], t=0 NPVs [N], metrics, states z [S], seconds)."""
    request = _request(samples)
    start = time.perf_counter()
    result = price_portfolio(request)
    values = np.asarray(result.npv_cube[:, 0, :])
    metrics = compute_risk_metrics(jnp.asarray(values.sum(axis=1))[:, None, None], result.base_npv,
                                   percentiles=(0.95, 0.99))
    elapsed = time.perf_counter() - start
    scenarios = simulate(request.market, request.config.simulation)
    return values, np.asarray(result.base_npv_per_trade), metrics, np.asarray(scenarios.states[:, 0, 0]), elapsed


def _ore_curve(dates, discounts) -> "ORE.YieldTermStructureHandle":
    return ORE.YieldTermStructureHandle(ORE.DiscountCurve(dates, discounts, DC))


def _ore_instruments(curve: "ORE.YieldTermStructureHandle", model: "ORE.HullWhite"):
    """ORE's swap and swaptions on `curve` (forwarding and discounting), the schedules as
    ORE builds the trades' underlyings: `(instrument, its swap, exercise date or None)`."""
    index = ibor_index(6, curve)
    out = []
    for cfg in _trades():
        booked = (swap_underlying if isinstance(cfg, SwapConfig) else european_underlying)(cfg)
        side = ORE.VanillaSwap.Payer if cfg.payer else ORE.VanillaSwap.Receiver
        swap = ORE.VanillaSwap(side, cfg.notional, booked.fixedSchedule(), cfg.fixed_rate, DC,
                               booked.floatingSchedule(), index, 0.0, DC)
        if isinstance(cfg, SwapConfig):
            swap.setPricingEngine(ORE.DiscountingSwapEngine(curve))
            out.append((swap, swap, None))
        else:
            swaption = ORE.Swaption(swap, ORE.EuropeanExercise(cfg.exercise_date))
            swaption.setPricingEngine(ORE.JamshidianSwaptionEngine(model, curve))
            out.append((swaption, swap, cfg.exercise_date))
    return out


def _price_ore(states: np.ndarray):
    """The same trades priced by ORE on each path's Hull-White curve at t = 1. Returns
    (per-trade NPVs [S, N], t=0 NPVs [N], seconds)."""
    ORE.Settings.instance().evaluationDate = TODAY
    curve0 = ORE.YieldTermStructureHandle(ORE.ZeroCurve([TODAY + round(t * 365) for t in PILLARS], RATES, DC))
    curve0.enableExtrapolation()
    hw0 = ORE.HullWhite(curve0, HW_A, HW_SIGMA)
    base = np.array([i.NPV() for i, _, _ in _ore_instruments(curve0, hw0)])

    a = HW_A
    H, Hp = (1 - np.exp(-a * T_EVAL)) / a, np.exp(-a * T_EVAL)
    zeta = HW_SIGMA ** 2 * np.expm1(2 * a * T_EVAL) / (2 * a)
    f0 = curve0.forwardRate(T_EVAL, T_EVAL, ORE.Continuous, ORE.NoFrequency, True).rate()

    nodes = [EVAL_DATE + ORE.Period(tenor) for tenor in DEFAULT_CURVE_TENORS]
    times = [T_EVAL + DC.yearFraction(EVAL_DATE, d) for d in nodes]
    start = time.perf_counter()
    values = []
    for z in states:
        r = f0 + Hp * float(z) + zeta * H * Hp
        ORE.Settings.instance().evaluationDate = TODAY
        discounts = [1.0] + [hw0.discountBond(T_EVAL, t, r) for t in times]
        ORE.Settings.instance().evaluationDate = EVAL_DATE
        curve = _ore_curve([EVAL_DATE] + nodes, discounts)
        curve.enableExtrapolation()
        values.append([i.NPV() for i, _, _ in _ore_instruments(curve, ORE.HullWhite(curve, HW_A, HW_SIGMA))])
    elapsed = time.perf_counter() - start
    ORE.Settings.instance().evaluationDate = TODAY
    return np.asarray(values), base, elapsed


@pytest.fixture(scope="module")
def comparison():
    samples = 2048
    values, base, metrics, states, engine_time = _price_engine(samples)
    ore_values, ore_base, ore_time = _price_ore(states)
    return dict(samples=samples, values=values, base=base, metrics=metrics, ore_values=ore_values,
                ore_base=ore_base, engine_time=engine_time, ore_time=ore_time)


class TestEndToEndEngineVsORE:
    """Same paths, same trades, priced by each side; NPV and VaR/ES compared."""

    def test_t0_values_match_ore(self, comparison):
        """The swap to machine precision; the swaptions within QuantLib's Brent tolerance on
        its root (tests/test_jamshidian.py)."""
        assert comparison["base"][0] == pytest.approx(comparison["ore_base"][0], rel=1e-12)
        np.testing.assert_allclose(comparison["base"][1:], comparison["ore_base"][1:], rtol=5e-6)

    def test_every_paths_value_matches_ore(self, comparison):
        """Pathwise, per trade, not only in summary statistics."""
        scale = np.maximum(np.abs(comparison["ore_values"]), 1.0)
        rel = np.abs(comparison["values"] - comparison["ore_values"]) / scale
        assert np.max(rel[:, 0]) < 1e-10, "the swap on a path's curve"
        assert np.max(rel[:, 1:]) < 5e-6, "the swaptions on a path's curve"

    def test_var_es_match_ore(self, comparison):
        ore_pnl = comparison["ore_values"].sum(axis=1) - comparison["ore_base"].sum()
        stats = ORE.RiskStatistics()
        stats.add(ORE.DoubleVector([float(v) for v in ore_pnl]))
        mine = comparison["metrics"]
        for q in (95, 99):
            assert float(mine[f"VaR_{q}"][0]) == pytest.approx(stats.valueAtRisk(q / 100), rel=1e-5)
            assert float(mine[f"ES_{q}"][0]) == pytest.approx(stats.expectedShortfall(q / 100), rel=1e-5)

    def test_reports_timing(self, comparison):
        """Prints the timings (no assertion on the ratio): the engine's time includes its
        compiles and the whole portfolio run, ORE's only the per-path loop."""
        print(f"\n[timing] {comparison['samples']} paths: engine={comparison['engine_time']:.3f}s, "
              f"ORE={comparison['ore_time']:.3f}s")
        assert comparison["engine_time"] > 0.0 and comparison["ore_time"] > 0.0


@pytest.mark.slow
class TestEndToEndScaling:
    """Accuracy holds at a larger path count; timings printed."""

    def test_timing_at_scale(self):
        values, _base, _metrics, states, engine_time = _price_engine(16384)
        ore_values, _ore_base, ore_time = _price_ore(states)
        rel = np.abs(values - ore_values) / np.maximum(np.abs(ore_values), 1.0)
        assert np.max(rel) < 5e-6
        print(f"\n[timing] 16384 paths: engine={engine_time:.3f}s, ORE={ore_time:.3f}s, "
              f"speedup={ore_time / engine_time:.2f}x")
