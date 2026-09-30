"""
`price_portfolio` (and, where noted, the HTTP API) at 1, ~12 and ~50 trades, with several
rate factors, and with inputs an external caller could submit: empty portfolios,
single-type portfolios at scale, duplicate and degenerate trades, extreme notionals, and
offsetting positions. (Per-pricer breadth is tests/test_diverse_portfolio_e2e.py.)
"""
import time

import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.simulation.market_model import (
    EquityConfig, RatesConfig, SimulationConfig, ZeroCurveConfig,
)
from engine.instruments.swap import SwapConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.portfolio import HULL_WHITE_CONFIG, PortfolioRequest, derive_maturity_pillars, price_portfolio

TODAY = ORE.Date(30, 7, 2026)
FLAT_RATE = 0.03
HW_A = 0.03
HW_SIGMA = 0.01
ZERO_CURVE = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[FLAT_RATE] * 6)
TIME_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def _swap(i: int, **overrides) -> SwapConfig:
    defaults = dict(
        notional=1_000_000.0 * (1 + i % 5), fixed_rate=0.028 + 0.001 * (i % 7),
        payer=(i % 2 == 0), discount_curve_index=0, forward_curve_index=0,
        swap_tenor=f"{2 + (i % 4)}Y", evaluation_date=TODAY,
    )
    defaults.update(overrides)
    return SwapConfig(**defaults)


def _swaption(i: int, **overrides) -> SwaptionConfig:
    defaults = dict(
        notional=500_000.0 * (1 + i % 3), fixed_rate=0.030 + 0.0005 * (i % 5),
        payer=(i % 2 == 0), rate_factor_index=0, hw_a=HW_A, hw_sigma=HW_SIGMA,
        initial_zero_curve=ZERO_CURVE, swap_tenor="2Y",
        forward_start=ORE.Period(1, ORE.Years), evaluation_date=TODAY,
    )
    defaults.update(overrides)
    return SwaptionConfig(**defaults)


def _bermudan(i: int, **overrides) -> BermudanSwaptionConfig:
    defaults = dict(
        notional=800_000.0 * (1 + i % 3), fixed_rate=0.029, payer=(i % 2 == 0),
        rate_factor_index=0, hw_a=HW_A, hw_sigma=HW_SIGMA, initial_zero_curve=ZERO_CURVE,
        exercise_dates=[ORE.Date(3, 8, 2027), ORE.Date(3, 8, 2028)], swap_tenor="3Y",
        evaluation_date=TODAY, n_per_std=32, std_devs=6.0,
    )
    defaults.update(overrides)
    return BermudanSwaptionConfig(**defaults)


def _american(i: int, **overrides) -> AmericanSwaptionConfig:
    defaults = dict(
        notional=600_000.0 * (1 + i % 3), fixed_rate=0.0295, payer=(i % 2 == 1),
        rate_factor_index=0, hw_a=HW_A, hw_sigma=HW_SIGMA, initial_zero_curve=ZERO_CURVE,
        first_exercise_date=ORE.Date(3, 8, 2027), last_exercise_date=ORE.Date(3, 8, 2028),
        swap_tenor="5Y", exercise_time_steps_per_year=1, evaluation_date=TODAY, n_per_std=32, std_devs=6.0,
    )
    defaults.update(overrides)
    return AmericanSwaptionConfig(**defaults)


def _sim_config(trades, scenarios=64, rates_overrides=None) -> SimulationConfig:
    pillars = derive_maturity_pillars(trades, TODAY)
    rates = dict(
        initial_rates=[FLAT_RATE], theta=[FLAT_RATE], mean_reversion=[HW_A],
        maturities=pillars, initial_zero_curves=[ZERO_CURVE],
    )
    if rates_overrides:
        rates.update(rates_overrides)
    return SimulationConfig(
        time_grid=TIME_GRID, scenarios=scenarios,
        equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[0.0]]),
        rates=RatesConfig(**rates),
        joint_covariance=[[0.04, 0.0], [0.0, HW_SIGMA ** 2]],
    )


def _assert_finite_result(result, expected_trades):
    assert result.npv_cube.shape[-1] == expected_trades
    assert bool(jnp.all(jnp.isfinite(result.npv_cube)))
    assert np.isfinite(result.base_npv)
    profile = result.exposure
    for name, arr in [("epe", profile.epe), ("ene", profile.ene), ("ee_b", profile.ee_b),
                      ("eee_b", profile.eee_b), *profile.pfe.items()]:
        assert np.all(np.isfinite(np.asarray(arr))), f"{name} produced a non-finite value"


# =============================================================================
# 1. PORTFOLIO SIZE SCALING
# =============================================================================
class TestPortfolioSizeScaling:
    """Increasing trade counts of mixed types: no shape errors, truncation or NaN."""

    def test_single_trade_portfolio(self):
        trades = [_swap(0)]
        sim = _sim_config(trades)
        result = price_portfolio(PortfolioRequest(market=sim, trades=trades, config=HULL_WHITE_CONFIG))
        _assert_finite_result(result, 1)

    @pytest.mark.slow
    def test_dozen_trade_mixed_portfolio(self):
        trades = [_swap(i) for i in range(4)] + [_swaption(i) for i in range(4)] + \
            [_bermudan(i) for i in range(2)] + [_american(i) for i in range(2)]
        sim = _sim_config(trades)
        result = price_portfolio(PortfolioRequest(market=sim, trades=trades, config=HULL_WHITE_CONFIG))
        _assert_finite_result(result, 12)

    @pytest.mark.slow
    def test_fifty_trade_mixed_portfolio(self):
        """50 trades of all four types with varied notional, tenor and direction price and
        aggregate end to end."""
        trades = (
            [_swap(i) for i in range(20)]
            + [_swaption(i) for i in range(15)]
            + [_bermudan(i) for i in range(8)]
            + [_american(i) for i in range(7)]
        )
        assert len(trades) == 50
        sim = _sim_config(trades, scenarios=128)
        result = price_portfolio(PortfolioRequest(market=sim, trades=trades, pfe_quantiles=(0.95, 0.99),
                                                  config=HULL_WHITE_CONFIG))
        _assert_finite_result(result, 50)

    def test_result_size_scales_linearly_in_trade_axis_only(self):
        """Doubling the trades doubles only the trade axis of the cube."""
        small = [_swap(i) for i in range(3)]
        large = [_swap(i) for i in range(6)]
        sim_small = _sim_config(small, scenarios=32)
        sim_large = _sim_config(large, scenarios=32)
        result_small = price_portfolio(PortfolioRequest(market=sim_small, trades=small, config=HULL_WHITE_CONFIG))
        result_large = price_portfolio(PortfolioRequest(market=sim_large, trades=large, config=HULL_WHITE_CONFIG))
        assert result_small.npv_cube.shape[0] == result_large.npv_cube.shape[0]
        assert result_small.npv_cube.shape[1] == result_large.npv_cube.shape[1]
        assert result_large.npv_cube.shape[2] == 2 * result_small.npv_cube.shape[2]


# =============================================================================
# 2. MULTI-RATE-FACTOR PORTFOLIOS THROUGH THE ENTRY POINT
# =============================================================================
class TestMultiRateFactorPortfolios:
    """Several rate factors through `price_portfolio`'s own validation, routing and base-NPV
    layers, which index per factor."""

    CURVE_A = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[0.030] * 6)
    CURVE_B = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[0.025] * 6)
    CURVE_C = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[0.035] * 6)

    def _three_factor_sim(self, trades, scenarios=64):
        pillars = derive_maturity_pillars(trades, TODAY)
        return SimulationConfig(
            time_grid=TIME_GRID, scenarios=scenarios,
            equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[0.0, 0.0, 0.0]]),
            rates=RatesConfig(
                initial_rates=[0.030, 0.025, 0.035], theta=[0.030, 0.025, 0.035],
                mean_reversion=[0.03, 0.025, 0.04], maturities=pillars,
                initial_zero_curves=[self.CURVE_A, self.CURVE_B, self.CURVE_C],
            ),
            joint_covariance=[
                [0.04, 0.0, 0.0, 0.0],
                [0.0, 0.03 ** 2, 0.0, 0.0],
                [0.0, 0.0, 0.028 ** 2, 0.0],
                [0.0, 0.0, 0.0, 0.033 ** 2],
            ],
        )

    def test_swaps_across_all_three_curves_price_finite(self):
        swaps = [
            SwapConfig(notional=1_000_000.0, fixed_rate=0.030, payer=True,
                       discount_curve_index=0, forward_curve_index=0, swap_tenor="2Y", evaluation_date=TODAY),
            SwapConfig(notional=1_000_000.0, fixed_rate=0.025, payer=True,
                       discount_curve_index=1, forward_curve_index=1, swap_tenor="2Y", evaluation_date=TODAY),
            SwapConfig(notional=1_000_000.0, fixed_rate=0.035, payer=False,
                       discount_curve_index=2, forward_curve_index=2, swap_tenor="2Y", evaluation_date=TODAY),
            SwapConfig(notional=500_000.0, fixed_rate=0.030, payer=True,
                       discount_curve_index=0, forward_curve_index=2, swap_tenor="2Y", evaluation_date=TODAY),
        ]
        sim = self._three_factor_sim(swaps)
        result = price_portfolio(PortfolioRequest(market=sim, trades=swaps, config=HULL_WHITE_CONFIG))
        _assert_finite_result(result, 4)

    def test_swaptions_on_each_distinct_rate_factor_price_finite(self):
        swaptions = [
            SwaptionConfig(notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
                            hw_a=0.03, hw_sigma=0.03, initial_zero_curve=self.CURVE_A,
                            swap_tenor="2Y", forward_start=ORE.Period(1, ORE.Years), evaluation_date=TODAY),
            SwaptionConfig(notional=1_000_000.0, fixed_rate=0.025, payer=True, rate_factor_index=1,
                            hw_a=0.025, hw_sigma=0.028, initial_zero_curve=self.CURVE_B,
                            swap_tenor="2Y", forward_start=ORE.Period(1, ORE.Years), evaluation_date=TODAY),
            SwaptionConfig(notional=1_000_000.0, fixed_rate=0.035, payer=False, rate_factor_index=2,
                            hw_a=0.04, hw_sigma=0.033, initial_zero_curve=self.CURVE_C,
                            swap_tenor="2Y", forward_start=ORE.Period(1, ORE.Years), evaluation_date=TODAY),
        ]
        sim = self._three_factor_sim(swaptions)
        result = price_portfolio(PortfolioRequest(market=sim, trades=swaptions, config=HULL_WHITE_CONFIG))
        _assert_finite_result(result, 3)

    def test_mismatched_curve_on_wrong_factor_is_rejected_even_with_multiple_factors(self):
        """A swaption on rate factor 1 carrying factor 0's curve is rejected (the check
        indexes the right factor)."""
        bad = SwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.025, payer=True, rate_factor_index=1,
            hw_a=0.025, hw_sigma=0.028, initial_zero_curve=self.CURVE_A,  # wrong curve for factor 1
            swap_tenor="2Y", forward_start=ORE.Period(1, ORE.Years), evaluation_date=TODAY,
        )
        sim = self._three_factor_sim([bad])
        with pytest.raises(ValueError, match="initial_zero_curve"):
            price_portfolio(PortfolioRequest(market=sim, trades=[bad], config=HULL_WHITE_CONFIG))


# =============================================================================
# 3. COMPOSITION EDGE CASES THROUGH THE ENTRY POINT
# =============================================================================
class TestCompositionEdgeCases:
    def test_empty_portfolio_returns_empty_but_well_formed_result(self):
        """An empty portfolio gives a zero-width cube and a base NPV of exactly 0.0."""
        sim = _sim_config([])
        result = price_portfolio(PortfolioRequest(market=sim, trades=[], config=HULL_WHITE_CONFIG))
        assert result.npv_cube.shape[-1] == 0
        assert result.base_npv == 0.0

    def test_single_instrument_type_at_scale_swaps_only(self):
        trades = [_swap(i) for i in range(25)]
        sim = _sim_config(trades, scenarios=64)
        result = price_portfolio(PortfolioRequest(market=sim, trades=trades, config=HULL_WHITE_CONFIG))
        _assert_finite_result(result, 25)

    @pytest.mark.slow
    def test_single_instrument_type_at_scale_swaptions_only(self):
        trades = [_swaption(i) for i in range(20)]
        sim = _sim_config(trades, scenarios=64)
        result = price_portfolio(PortfolioRequest(market=sim, trades=trades, config=HULL_WHITE_CONFIG))
        _assert_finite_result(result, 20)

    def test_identical_duplicate_trades_price_identically(self):
        """Identical trades give identical columns."""
        trades = [_swap(0) for _ in range(10)]
        sim = _sim_config(trades, scenarios=64)
        result = price_portfolio(PortfolioRequest(market=sim, trades=trades, config=HULL_WHITE_CONFIG))
        cube = np.asarray(result.npv_cube)
        for j in range(1, 10):
            np.testing.assert_allclose(cube[:, :, j], cube[:, :, 0], rtol=1e-12)

    def test_zero_notional_trade_prices_to_zero_and_does_not_poison_others(self):
        """A zero-notional trade prices to ~0 without NaN affecting the rest."""
        trades = [_swap(0, notional=0.0), _swap(1, notional=2_000_000.0)]
        sim = _sim_config(trades, scenarios=64)
        result = price_portfolio(PortfolioRequest(market=sim, trades=trades, config=HULL_WHITE_CONFIG))
        cube = np.asarray(result.npv_cube)
        np.testing.assert_allclose(cube[:, :, 0], 0.0, atol=1e-6)
        assert np.all(np.isfinite(cube[:, :, 1]))
        assert np.all(np.isfinite(np.asarray(result.exposure.pfe["PFE_95"])))

    def test_negative_notional_swap_is_the_mirror_image_of_positive(self):
        """A negative-notional swap is the exact mirror of the positive one."""
        positive = _swap(0, notional=1_000_000.0)
        negative = _swap(0, notional=-1_000_000.0)
        sim = _sim_config([positive, negative], scenarios=64)
        result = price_portfolio(PortfolioRequest(market=sim, trades=[positive, negative], config=HULL_WHITE_CONFIG))
        cube = np.asarray(result.npv_cube)
        np.testing.assert_allclose(cube[:, :, 1], -cube[:, :, 0], rtol=1e-9)

    def test_very_large_notional_does_not_overflow_or_lose_precision(self):
        """A 5e10 notional prices finitely and scales correctly."""
        small = _swap(0, notional=1_000_000.0)
        huge = _swap(0, notional=5.0e10)
        sim = _sim_config([small, huge], scenarios=64)
        result = price_portfolio(PortfolioRequest(market=sim, trades=[small, huge], config=HULL_WHITE_CONFIG))
        cube = np.asarray(result.npv_cube)
        assert np.all(np.isfinite(cube))
        np.testing.assert_allclose(cube[:, :, 1], cube[:, :, 0] * 5.0e4, rtol=1e-6)

    def test_offsetting_large_portfolio_nets_close_to_zero_risk(self):
        """20 identical payer/receiver pairs net to ~0 NPV and small VaR/ES at every step."""
        payers = [_swap(i, payer=True, notional=1_000_000.0, fixed_rate=0.03, swap_tenor="3Y") for i in range(20)]
        receivers = [_swap(i, payer=False, notional=1_000_000.0, fixed_rate=0.03, swap_tenor="3Y") for i in range(20)]
        trades = payers + receivers
        sim = _sim_config(trades, scenarios=256)
        result = price_portfolio(PortfolioRequest(market=sim, trades=trades, pfe_quantiles=(0.95,),
                                                  config=HULL_WHITE_CONFIG))
        cube = np.asarray(result.npv_cube)
        portfolio_npv_per_scenario = cube.sum(axis=-1)
        assert np.max(np.abs(portfolio_npv_per_scenario)) < 1e-6
        assert abs(result.base_npv) < 1e-6

    @pytest.mark.slow
    def test_all_four_instrument_types_each_represented_multiple_times(self):
        """Several trades of each type with compute_greeks: every trade, swaps included,
        has Greeks. (This once asserted swaps were skipped "by design", pinning I-01; see
        tests/test_portfolio_gap_fixes.py::TestSwapGreeksReachThePortfolioPath.)"""
        swaps = [_swap(i) for i in range(3)]
        swaptions = [_swaption(i) for i in range(3)]
        bermudans = [_bermudan(i) for i in range(3)]
        americans = [_american(i) for i in range(3)]
        trades = swaps + swaptions + bermudans + americans
        sim = _sim_config(trades, scenarios=64)
        result = price_portfolio(PortfolioRequest(market=sim, trades=trades, compute_greeks=True,
                                                  config=HULL_WHITE_CONFIG))
        _assert_finite_result(result, 12)
        assert result.greeks is not None
        # Every trade, swaps (0-2) included, has Greeks.
        assert set(result.greeks.keys()) == set(range(12))
        for idx in range(12):
            assert np.isfinite(result.greeks[idx]["theta"])
        # Swaps report per-curve deltas; the swaption family one.
        for idx in range(3):
            assert "discount_delta" in result.greeks[idx]
            assert "forward_delta" in result.greeks[idx]


# =============================================================================
# 4. Timing sanity at scale
# =============================================================================
class TestPortfolioTimingSanity:
    """A 50-trade portfolio finishes within a generous bound (120s), to catch a
    super-linear regression that correctness tests would not notice."""

    def test_fifty_trade_portfolio_completes_within_a_generous_bound(self):
        trades = (
            [_swap(i) for i in range(20)]
            + [_swaption(i) for i in range(15)]
            + [_bermudan(i) for i in range(8)]
            + [_american(i) for i in range(7)]
        )
        sim = _sim_config(trades, scenarios=64)
        start = time.time()
        price_portfolio(PortfolioRequest(market=sim, trades=trades, config=HULL_WHITE_CONFIG))
        elapsed = time.time() - start
        assert elapsed < 120.0, f"50-trade portfolio took {elapsed:.1f}s -- investigate for a scaling regression"
