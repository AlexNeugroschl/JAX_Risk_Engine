"""
`price_portfolio` at 1, ~12 and ~50 trades, across currencies, and with inputs an external
caller could submit: empty portfolios, single-type portfolios at scale, duplicate and
degenerate trades, extreme notionals, and offsetting positions. The simulation is the
Hull-White model on the demo market (USD and EUR). (Per-pricer breadth is
tests/test_diverse_portfolio_e2e.py.)
"""
import time

import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from demos.demo_scenarios import EVAL_DATE, demo_market, demo_simulation
from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig
from engine.run import (
    LgmSwaptionEngineConfig, PortfolioRequest, PricingConfig, RunConfig, price_portfolio,
)

FAST = LgmSwaptionEngineConfig(n_per_std=12, std_devs=4.0)
DATES = tuple(EVAL_DATE + ORE.Period(m, ORE.Months) for m in (6, 12, 24, 36))


def _swap(i: int, currency: str = "USD", **overrides) -> SwapConfig:
    fields = dict(notional=1_000_000.0 * (1 + i % 5), fixed_rate=0.028 + 0.001 * (i % 7), payer=(i % 2 == 0),
                  swap_tenor=f"{2 + (i % 4)}Y", currency=currency, evaluation_date=EVAL_DATE,
                  trade_id=f"swap-{currency}-{i}")
    fields.update(overrides)
    return SwapConfig(**fields)


def _swaption(i: int, currency: str = "USD") -> SwaptionConfig:
    return SwaptionConfig(notional=500_000.0 * (1 + i % 3), fixed_rate=0.030 + 0.0005 * (i % 5), payer=(i % 2 == 0),
                          swap_tenor="2Y", forward_start=ORE.Period(1, ORE.Years), currency=currency,
                          evaluation_date=EVAL_DATE, trade_id=f"european-{currency}-{i}")


def _bermudan(i: int) -> BermudanSwaptionConfig:
    return BermudanSwaptionConfig(notional=800_000.0 * (1 + i % 3), fixed_rate=0.029, payer=(i % 2 == 0),
                                  exercise_dates=[ORE.Date(3, 8, 2027), ORE.Date(3, 8, 2028)], swap_tenor="3Y",
                                  evaluation_date=EVAL_DATE, trade_id=f"bermudan-{i}")


def _american(i: int) -> AmericanSwaptionConfig:
    return AmericanSwaptionConfig(notional=600_000.0 * (1 + i % 3), fixed_rate=0.0295, payer=(i % 2 == 1),
                                  first_exercise_date=ORE.Date(3, 8, 2027), last_exercise_date=ORE.Date(3, 8, 2028),
                                  swap_tenor="5Y", evaluation_date=EVAL_DATE, trade_id=f"american-{i}")


def _price(trades, samples=64, compute_greeks=False, pfe_quantiles=(0.95, 0.99)):
    simulation = demo_simulation("HullWhite", samples=samples, dates=DATES)
    config = RunConfig(simulation=simulation, pricing=PricingConfig(bermudan=FAST, american=FAST))
    return price_portfolio(PortfolioRequest(market=demo_market(), trades=list(trades), config=config,
                                            compute_greeks=compute_greeks, pfe_quantiles=pfe_quantiles))


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
        _assert_finite_result(_price([_swap(0)]), 1)

    @pytest.mark.slow
    def test_dozen_trade_mixed_portfolio(self):
        trades = [_swap(i) for i in range(4)] + [_swaption(i) for i in range(4)] + \
            [_bermudan(i) for i in range(2)] + [_american(i) for i in range(2)]
        _assert_finite_result(_price(trades), 12)

    @pytest.mark.slow
    def test_fifty_trade_mixed_portfolio(self):
        """50 trades of all four types in two currencies, with varied notional, tenor and
        direction, price and aggregate end to end."""
        trades = ([_swap(i) for i in range(10)] + [_swap(i, "EUR") for i in range(10)]
                  + [_swaption(i) for i in range(8)] + [_swaption(i, "EUR") for i in range(8)]
                  + [_bermudan(i) for i in range(7)] + [_american(i) for i in range(7)])
        assert len(trades) == 50
        _assert_finite_result(_price(trades, samples=128), 50)

    def test_result_size_scales_linearly_in_trade_axis_only(self):
        small = _price([_swap(i) for i in range(3)], samples=32)
        large = _price([_swap(i) for i in range(6)], samples=32)
        assert small.npv_cube.shape[:2] == large.npv_cube.shape[:2]
        assert large.npv_cube.shape[2] == 2 * small.npv_cube.shape[2]


# =============================================================================
# 2. SEVERAL CURRENCIES THROUGH THE ENTRY POINT
# =============================================================================
class TestMultiCurrencyPortfolios:
    """Trades in each currency through `price_portfolio`'s validation, valuation and FX
    conversion, which work per currency (before 2026-10-01 the Hull-White model indexed
    rate factors and curves by position instead)."""

    def test_swaps_in_both_currencies_price_finite(self):
        _assert_finite_result(_price([_swap(0), _swap(1, "EUR"), _swap(2, "EUR", payer=False)]), 3)

    def test_swaptions_in_both_currencies_price_finite(self):
        _assert_finite_result(_price([_swaption(0), _swaption(1, "EUR")]), 2)

    def test_a_eur_trade_is_reported_in_usd_at_the_spot(self):
        """Today's value of a EUR trade is its EUR value at the EURUSD spot."""
        from engine.pricing.cube import value_today
        eur = _swap(1, "EUR")
        result = _price([eur])
        in_eur = value_today([eur], demo_market(), "EUR")[0]
        assert result.base_npv_per_trade[0] == pytest.approx(in_eur * 1.10, rel=1e-14)


# =============================================================================
# 3. COMPOSITION EDGE CASES THROUGH THE ENTRY POINT
# =============================================================================
class TestCompositionEdgeCases:
    def test_empty_portfolio_returns_empty_but_well_formed_result(self):
        result = _price([])
        assert result.npv_cube.shape[-1] == 0
        assert result.base_npv == 0.0

    def test_single_instrument_type_at_scale_swaps_only(self):
        _assert_finite_result(_price([_swap(i) for i in range(25)]), 25)

    @pytest.mark.slow
    def test_single_instrument_type_at_scale_swaptions_only(self):
        _assert_finite_result(_price([_swaption(i) for i in range(20)]), 20)

    def test_identical_duplicate_trades_price_identically(self):
        """Identical trades (distinct ids) give identical columns."""
        trades = [_swap(0, trade_id=f"copy-{j}") for j in range(10)]
        cube = np.asarray(_price(trades).npv_cube)
        for j in range(1, 10):
            np.testing.assert_array_equal(cube[:, :, j], cube[:, :, 0])

    def test_zero_notional_trade_prices_to_zero_and_does_not_poison_others(self):
        result = _price([_swap(0, notional=0.0), _swap(1, notional=2_000_000.0)])
        cube = np.asarray(result.npv_cube)
        np.testing.assert_allclose(cube[:, :, 0], 0.0, atol=1e-6)
        assert np.all(np.isfinite(cube[:, :, 1]))
        assert np.all(np.isfinite(np.asarray(result.exposure.pfe["PFE_95"])))

    def test_negative_notional_swap_is_the_mirror_image_of_positive(self):
        positive = _swap(0, notional=1_000_000.0, trade_id="long")
        negative = _swap(0, notional=-1_000_000.0, trade_id="short")
        cube = np.asarray(_price([positive, negative]).npv_cube)
        np.testing.assert_allclose(cube[:, :, 1], -cube[:, :, 0], rtol=1e-12)

    def test_very_large_notional_does_not_overflow_or_lose_precision(self):
        small = _swap(0, notional=1_000_000.0, trade_id="small")
        huge = _swap(0, notional=5.0e10, trade_id="huge")
        cube = np.asarray(_price([small, huge]).npv_cube)
        assert np.all(np.isfinite(cube))
        np.testing.assert_allclose(cube[:, :, 1], cube[:, :, 0] * 5.0e4, rtol=1e-9)

    def test_offsetting_large_portfolio_nets_close_to_zero_risk(self):
        """20 identical payer/receiver pairs net to ~0 on every path and date."""
        pairs = [_swap(i, payer=p, notional=1_000_000.0, fixed_rate=0.03, swap_tenor="3Y",
                       trade_id=f"{'payer' if p else 'receiver'}-{i}") for i in range(20) for p in (True, False)]
        result = _price(pairs, samples=256, pfe_quantiles=(0.95,))
        assert np.max(np.abs(np.asarray(result.npv_cube).sum(axis=-1))) < 1e-6
        assert abs(result.base_npv) < 1e-6

    @pytest.mark.slow
    def test_all_four_instrument_types_each_represented_multiple_times(self):
        """Several trades of each type with compute_greeks: every trade, swaps included,
        has Greeks. (This once asserted swaps were skipped "by design", pinning I-01; see
        tests/test_portfolio_gap_fixes.py::TestSwapGreeksReachThePortfolioPath.)"""
        trades = ([_swap(i) for i in range(2)] + [_swaption(i) for i in range(2)]
                  + [_bermudan(i) for i in range(2)] + [_american(i) for i in range(2)])
        result = _price(trades, compute_greeks=True)
        _assert_finite_result(result, 8)
        assert set(result.greeks) == set(range(8))
        for idx in range(8):
            assert np.isfinite(result.greeks[idx]["theta"])
            assert "delta:discount:USD" in result.greeks[idx]


# =============================================================================
# 4. Timing sanity at scale
# =============================================================================
@pytest.mark.slow
class TestPortfolioTimingSanity:
    """A 50-trade portfolio of the path-vectorized types finishes within a generous bound
    (120 s), to catch a super-linear regression that correctness tests would not notice.
    Bermudans and Americans are left out: each recalibrates on every path date, as ORE's
    `recalibrate = true` does, at about 45 s per option today (I-53)."""

    def test_fifty_trade_portfolio_completes_within_a_generous_bound(self):
        trades = ([_swap(i) for i in range(15)] + [_swap(i, "EUR") for i in range(15)]
                  + [_swaption(i) for i in range(10)] + [_swaption(i, "EUR") for i in range(10)])
        start = time.time()
        _price(trades)
        elapsed = time.time() - start
        assert elapsed < 120.0, f"50-trade portfolio took {elapsed:.1f}s -- investigate for a scaling regression"
