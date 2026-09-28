"""
Regression tests for gaps where `price_portfolio` returned a quietly incomplete result
(found while writing docs/planning/traderX_integration/eod-contract-proposal.md). Each
class fails against the pre-fix code.

1. Swap Greeks were skipped (I-01): `_compute_all_greeks` could not resolve a swap's curve
   indices and skipped every swap without error.
2. Bermudan Vega was never called from the portfolio path (I-02).
3. There was no per-trade base NPV (I-03).
4. The aged-swap inaccuracy (I-04) is not fixed, but is no longer silent: a warning is
   collected into `PortfolioResult.warnings`.

Plus I-13: curve indices were validated only on the Greeks path.
"""
import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.models.ore_builders import time_from_reference
from engine.simulation.market_model import (
    EquityConfig, RatesConfig, SimulationConfig, ZeroCurveConfig,
)
from engine.instruments.swap import SwapConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.calibration.basket import build_coterminal_basket
from engine.models.hull_white import ZeroCurve as HwZeroCurve
from engine.risk import greeks as _greeks
from engine.portfolio import PortfolioRequest, derive_maturity_pillars, price_portfolio
from engine.portfolio.request import _swap_curve_configs

TODAY = ORE.Date(30, 7, 2026)
FLAT_RATE = 0.03
HW_A = 0.03
HW_SIGMA = 0.01
ZERO_CURVE = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[FLAT_RATE] * 6)

# A curve different from ZERO_CURVE, as rate factor 1, to show a swap's Greeks use the curve
# its indices name.
STEEP_CURVE = ZeroCurveConfig(
    times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0],
    rates=[0.020, 0.025, 0.030, 0.040, 0.045, 0.050],
)

# A short grid ending before the swap's first accrual end (fast); aged swaps have their own
# grids below.
TIME_GRID = [0.0, 0.5, 1.0]


def _swap(notional=2_000_000.0, fixed_rate=0.032, payer=True,
          disc_idx=0, fwd_idx=0, tenor="2Y"):
    return SwapConfig(
        notional=notional, fixed_rate=fixed_rate, payer=payer,
        discount_curve_index=disc_idx, forward_curve_index=fwd_idx,
        swap_tenor=tenor, evaluation_date=TODAY,
    )


def _sim_config(trades, curves=None, time_grid=None, n_factors=1):
    """A `SimulationConfig` with `n_factors` rate factors and maturity pillars derived from
    the trades."""
    curves = curves or [ZERO_CURVE]
    pillars = derive_maturity_pillars(trades, TODAY)
    # joint_covariance is [equities + rates]; one equity here.
    dim = 1 + n_factors
    cov = np.zeros((dim, dim))
    cov[0, 0] = 0.04
    for k in range(n_factors):
        cov[1 + k, 1 + k] = HW_SIGMA ** 2
    return SimulationConfig(
        time_grid=time_grid or TIME_GRID,
        scenarios=64,
        equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0],
                              rate_mapping=[[0.0] * n_factors]),
        rates=RatesConfig(
            initial_rates=[FLAT_RATE] * n_factors, theta=[FLAT_RATE] * n_factors,
            mean_reversion=[HW_A] * n_factors, maturities=pillars,
            initial_zero_curves=curves,
        ),
        joint_covariance=cov.tolist(),
    )


# =============================================================================
# Gap 1: swap Greeks were silently skipped (I-01)
# =============================================================================
class TestSwapGreeksReachThePortfolioPath:
    """`compute_greeks=True` returns Delta/Gamma/Theta for swaps (the pre-fix code skipped
    them and returned no swap entries, without error)."""

    def test_swap_greeks_are_present_not_skipped(self):
        """A swap-only portfolio with compute_greeks=True returned `{}` before the fix."""
        trades = [_swap()]
        request = PortfolioRequest(
            market=_sim_config(trades), trades=trades, compute_greeks=True,
        )
        result = price_portfolio(request)

        assert result.greeks is not None
        assert 0 in result.greeks, (
            "swap at trade index 0 has no Greeks entry -- the dispatcher "
            "skipped it (the exact pre-fix bug this test pins closed)"
        )
        for key in ("discount_delta", "discount_gamma",
                    "forward_delta", "forward_gamma", "theta"):
            assert key in result.greeks[0], f"missing {key!r} for the swap"

    def test_swap_greeks_are_finite_and_nonzero(self):
        """A 2Y payer swap's Greeks are finite and non-zero (not skipped-then-zeroed)."""
        trades = [_swap()]
        result = price_portfolio(PortfolioRequest(
            market=_sim_config(trades), trades=trades, compute_greeks=True,
        ))
        g = result.greeks[0]
        for key in ("discount_delta", "forward_delta"):
            arr = np.asarray(g[key])
            assert np.all(np.isfinite(arr)), f"{key} contains non-finite values"
            assert np.any(np.abs(arr) > 0), f"{key} is identically zero"
        assert np.isfinite(float(g["theta"]))

    def test_portfolio_swap_greeks_equal_direct_greeks_call(self):
        """Greeks through `price_portfolio` equal a direct `engine.risk.greeks` call on the
        curves the swap's indices name."""
        trades = [_swap()]
        sim = _sim_config(trades)
        result = price_portfolio(PortfolioRequest(
            market=sim, trades=trades, compute_greeks=True,
        ))

        disc = HwZeroCurve.from_config(sim.rates.initial_zero_curves[0])
        fwd = HwZeroCurve.from_config(sim.rates.initial_zero_curves[0])
        expected = _greeks.swap_delta_gamma(trades[0], disc, fwd)
        expected_theta = _greeks.swap_theta(trades[0], disc, fwd)

        for key in ("discount_delta", "discount_gamma",
                    "forward_delta", "forward_gamma"):
            np.testing.assert_allclose(
                np.asarray(result.greeks[0][key]), np.asarray(expected[key]),
                rtol=1e-12, atol=0.0,
                err_msg=f"{key} routed through price_portfolio differs from a direct call",
            )
        np.testing.assert_allclose(
            float(result.greeks[0]["theta"]), float(expected_theta), rtol=1e-12,
        )

    def test_uses_the_curve_its_indexes_name_not_curve_zero(self):
        """A swap on rate factor 1 is differentiated against factor 1's (steep) curve;
        defaulting to curve 0 would fail here."""
        trades = [_swap(disc_idx=1, fwd_idx=1)]
        sim = _sim_config(trades, curves=[ZERO_CURVE, STEEP_CURVE], n_factors=2)
        result = price_portfolio(PortfolioRequest(
            market=sim, trades=trades, compute_greeks=True,
        ))

        steep = HwZeroCurve.from_config(STEEP_CURVE)
        expected = _greeks.swap_delta_gamma(trades[0], steep, steep)
        np.testing.assert_allclose(
            np.asarray(result.greeks[0]["discount_delta"]),
            np.asarray(expected["discount_delta"]), rtol=1e-12, atol=0.0,
        )

        flat = HwZeroCurve.from_config(ZERO_CURVE)
        wrong = _greeks.swap_delta_gamma(trades[0], flat, flat)
        assert not np.allclose(
            np.asarray(expected["discount_delta"]),
            np.asarray(wrong["discount_delta"]), rtol=1e-6,
        ), ("STEEP_CURVE and ZERO_CURVE must produce materially different "
            "deltas, or this test cannot detect a wrong-curve substitution")

    def test_mixed_portfolio_keys_every_trade_by_its_own_index(self):
        """In a mixed portfolio every trade index gets its own entry."""
        swap_cfg = _swap()
        swaption_cfg = SwaptionConfig(
            notional=1_500_000.0, fixed_rate=0.031, payer=True, rate_factor_index=0,
            hw_a=HW_A, hw_sigma=HW_SIGMA, initial_zero_curve=ZERO_CURVE,
            swap_tenor="1Y", forward_start=ORE.Period(1, ORE.Years),
            evaluation_date=TODAY,
        )
        # The swap is second, so appending rather than keying by index would misalign.
        trades = [swaption_cfg, swap_cfg]
        result = price_portfolio(PortfolioRequest(
            market=_sim_config(trades), trades=trades, compute_greeks=True,
        ))

        assert set(result.greeks) == {0, 1}
        assert "discount_delta" in result.greeks[1], "index 1 is the swap"
        assert "delta" in result.greeks[0], "index 0 is the swaption"

    def test_out_of_range_curve_index_raises_naming_the_trade(self):
        """An unresolvable curve index raises rather than being clamped."""
        trades = [_swap()]
        sim = _sim_config(trades)
        bad = SwapConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True,
            discount_curve_index=0, forward_curve_index=7,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        with pytest.raises(ValueError, match=r"forward_curve_index=7.*out of range"):
            _swap_curve_configs(bad, sim, trade_index=3)

    def test_greeks_absent_when_not_requested(self):
        """compute_greeks=False still returns None."""
        trades = [_swap()]
        result = price_portfolio(PortfolioRequest(
            market=_sim_config(trades), trades=trades, compute_greeks=False,
        ))
        assert result.greeks is None


# =============================================================================
# Gap 1b (I-13): curve indices were validated only on the Greeks path
# =============================================================================
class TestCurveIndexValidatedBeforeAllPricing:
    """An invalid curve index fails on every pricing path, not only with
    compute_greeks=True.

    I-13: only `_compute_all_greeks` validated indices, so with the default
    compute_greeks=False a negative index wrapped (-1 -> last curve) and the trade priced
    against the wrong curve. These go through `price_portfolio`, where a real caller enters;
    the helper's own test passed throughout.
    """

    @staticmethod
    def _price(disc_idx, fwd_idx, compute_greeks, n_curves=1):
        curves = [ZERO_CURVE, STEEP_CURVE][:n_curves]
        trades = [_swap(disc_idx=disc_idx, fwd_idx=fwd_idx)]
        return price_portfolio(PortfolioRequest(
            market=_sim_config(trades, curves=curves, n_factors=n_curves),
            trades=trades, compute_greeks=compute_greeks,
        ))

    @pytest.mark.parametrize("compute_greeks", [False, True])
    @pytest.mark.parametrize("disc_idx,fwd_idx,offender", [
        (0, -1, "forward_curve_index=-1"),
        (-1, 0, "discount_curve_index=-1"),
        (0, 7, "forward_curve_index=7"),
        (7, 0, "discount_curve_index=7"),
    ])
    def test_invalid_index_raises_on_both_paths(self, disc_idx, fwd_idx,
                                                offender, compute_greeks):
        """Negative and out-of-range indices, with compute_greeks True and False. Before the
        fix, negative indices at compute_greeks=False returned a plausible NPV and
        out-of-range ones a bare IndexError."""
        with pytest.raises(ValueError, match=rf"{offender}.*out of range"):
            self._price(disc_idx, fwd_idx, compute_greeks)

    def test_error_names_the_trade_and_the_field(self):
        """The error names the trade and which index."""
        with pytest.raises(ValueError) as exc:
            self._price(0, -1, compute_greeks=False)
        message = str(exc.value)
        assert "trade[0]" in message
        assert "forward_curve_index=-1" in message
        assert "SwapConfig" in message

    def test_negative_index_does_not_price_against_the_wrapped_curve(self):
        """With two different curves, fwd_idx=-1 once returned curve 1's NPV silently. The
        valid index still prices and the invalid one raises instead."""
        booked = self._price(0, 0, compute_greeks=False, n_curves=2)
        wrapped = self._price(0, 1, compute_greeks=False, n_curves=2)
        booked_npv = booked.base_npv_per_trade[0]
        wrapped_npv = wrapped.base_npv_per_trade[0]

        assert abs(booked_npv - wrapped_npv) > 1e-6, (
            "test setup is degenerate: the two curves must price differently "
            "for the wrap to be detectable at all"
        )
        with pytest.raises(ValueError):
            self._price(0, -1, compute_greeks=False, n_curves=2)

    @pytest.mark.parametrize("compute_greeks", [False, True])
    def test_valid_indices_are_unaffected(self, compute_greeks):
        """Every in-range index, including the last curve, still prices."""
        result = self._price(0, 1, compute_greeks, n_curves=2)
        assert np.isfinite(result.base_npv_per_trade[0])
        assert result.base_npv_per_trade[0] != 0.0


# =============================================================================
# Gap 2: Bermudan Vega was never called (I-02)
# =============================================================================
class TestBermudanVegaReachesThePortfolioPath:
    """The portfolio path now calls `bermudan_vega`."""

    @staticmethod
    def _calibrated_request():
        exercise_dates = [ORE.Date(3, 8, 2027), ORE.Date(3, 8, 2028)]
        bermudan = BermudanSwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
            hw_a=HW_A, hw_sigma=None, initial_zero_curve=ZERO_CURVE,
            exercise_dates=exercise_dates, swap_tenor="3Y",
            evaluation_date=TODAY, n_per_std=32, std_devs=5.0,
        )
        targets = build_coterminal_basket(
            evaluation_date=TODAY, exercise_times=[time_from_reference(TODAY, d) for d in exercise_dates],
            final_maturity_time=3.0, notional=1_000_000.0, payer=True,
            market_vols=[0.008, 0.0088],
            zero_curve=HwZeroCurve.from_config(ZERO_CURVE),
        )
        trades = [bermudan]
        return PortfolioRequest(
            market=_sim_config(trades), trades=trades, compute_greeks=True,
            calibration_targets=targets,
        ), targets

    @pytest.mark.slow
    def test_vega_present_for_a_calibrated_bermudan(self):
        request, targets = self._calibrated_request()
        result = price_portfolio(request)

        assert "vega" in result.greeks[0], (
            "a calibrated Bermudan reported no Vega -- bermudan_vega was "
            "never invoked by the portfolio path (the pre-fix bug)"
        )
        vega = np.asarray(result.greeks[0]["vega"])
        assert vega.shape == (len(targets),), (
            "Vega must carry one entry per calibration-basket instrument"
        )
        assert np.all(np.isfinite(vega))
        assert np.any(np.abs(vega) > 0), "Vega is identically zero"

    @pytest.mark.slow
    def test_no_vega_for_a_flat_uncalibrated_sigma(self):
        """A flat hand-set hw_sigma has no market quote to be sensitive to: Vega is omitted,
        not fabricated."""
        bermudan = BermudanSwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
            hw_a=HW_A, hw_sigma=HW_SIGMA, initial_zero_curve=ZERO_CURVE,
            exercise_dates=[ORE.Date(3, 8, 2027), ORE.Date(3, 8, 2028)],
            swap_tenor="3Y", evaluation_date=TODAY, n_per_std=32, std_devs=5.0,
        )
        trades = [bermudan]
        result = price_portfolio(PortfolioRequest(
            market=_sim_config(trades), trades=trades, compute_greeks=True,
        ))
        assert "vega" not in result.greeks[0]
        # Delta/Gamma/Theta are still present.
        assert "delta" in result.greeks[0]
        assert "theta" in result.greeks[0]


# =============================================================================
# Gap 3: per-trade base NPV (I-03)
# =============================================================================
class TestPerTradeBaseNpv:
    """`base_npv_per_trade` lets the total be reconciled against its positions."""

    def test_per_trade_values_are_returned_one_per_trade(self):
        trades = [_swap(notional=2_000_000.0), _swap(notional=500_000.0, tenor="3Y")]
        result = price_portfolio(PortfolioRequest(
            market=_sim_config(trades), trades=trades,
        ))
        assert len(result.base_npv_per_trade) == len(trades)
        assert all(np.isfinite(v) for v in result.base_npv_per_trade)

    def test_total_equals_sum_of_parts_exactly(self):
        """The total is the sum of the per-trade values (not a separate computation)."""
        trades = [_swap(notional=2_000_000.0), _swap(notional=500_000.0, tenor="3Y")]
        result = price_portfolio(PortfolioRequest(
            market=_sim_config(trades), trades=trades,
        ))
        assert result.base_npv == pytest.approx(
            float(sum(result.base_npv_per_trade)), rel=0.0, abs=1e-9,
        )

    def test_values_are_attributable_to_the_right_trade(self):
        """Order follows `request.trades`: a payer and an identical receiver have opposite
        signs, so misordering fails."""
        payer = _swap(notional=2_000_000.0, fixed_rate=0.05, payer=True)
        receiver = _swap(notional=2_000_000.0, fixed_rate=0.05, payer=False)
        trades = [payer, receiver]
        result = price_portfolio(PortfolioRequest(
            market=_sim_config(trades), trades=trades,
        ))
        first, second = result.base_npv_per_trade
        assert first == pytest.approx(-second, rel=1e-9), (
            "a payer and its mirrored receiver must have opposite base NPVs"
        )
        assert first < 0 < second, (
            "a 5% payer against a 3% curve is out-of-the-money; sign/order "
            "attribution is wrong"
        )

    def test_single_trade_total_matches_its_only_row(self):
        trades = [_swap()]
        result = price_portfolio(PortfolioRequest(
            market=_sim_config(trades), trades=trades,
        ))
        assert len(result.base_npv_per_trade) == 1
        assert result.base_npv == pytest.approx(result.base_npv_per_trade[0], rel=0.0, abs=1e-12)

    def test_empty_portfolio_has_empty_breakdown(self):
        result = price_portfolio(PortfolioRequest(
            market=_sim_config([]), trades=[],
        ))
        assert result.base_npv_per_trade == []
        assert result.base_npv == 0.0


# =============================================================================
# Gap 4: the aged-swap limitation was silent (I-04)
# =============================================================================
class TestAgedSwapWarningIsNotSilent:
    """The aged-swap inaccuracy is not fixed (see `engine.instruments.swap` and
    tests/test_swap.py::TestAgedSwapKnownLimitation), but a grid passing a swap's first
    accrual start now produces a warning."""

    def test_spot_starting_swap_over_a_long_grid_warns(self):
        """A spot-starting swap over a long grid warns."""
        trades = [_swap(tenor="2Y")]
        sim = _sim_config(trades, time_grid=[0.0, 0.5, 1.0, 1.5])
        result = price_portfolio(PortfolioRequest(market=sim, trades=trades))

        aged = [w for w in result.warnings if "already started accruing" in w]
        assert aged, (
            f"expected an aged-swap warning in PortfolioResult.warnings, got {result.warnings!r}"
        )
        assert "trade[0]" in aged[0], "the warning must name the offending trade"
        assert "t=0 base NPV is unaffected" in aged[0], (
            "the warning must scope the inaccuracy, so a reader does not "
            "discard an exact t=0 valuation"
        )

    def test_grid_stopping_before_first_accrual_does_not_warn(self):
        """A grid whose last step is before the swap's first accrual start (~2 business days
        out for a spot start) is exact and does not warn."""
        trades = [_swap(tenor="2Y")]
        sim = _sim_config(trades, time_grid=[0.0, 0.002])
        result = price_portfolio(PortfolioRequest(market=sim, trades=trades))
        assert not [w for w in result.warnings if "already started accruing" in w], (
            f"warned about a swap that is never aged in this grid: {result.warnings!r}"
        )

    def test_warning_names_every_aged_swap_separately(self):
        trades = [_swap(notional=1_000_000.0), _swap(notional=7_500_000.0, tenor="3Y")]
        sim = _sim_config(trades, time_grid=[0.0, 0.5, 1.0, 1.5])
        result = price_portfolio(PortfolioRequest(market=sim, trades=trades))

        aged = [w for w in result.warnings if "already started accruing" in w]
        assert len(aged) == 2
        assert any("trade[0]" in w for w in aged)
        assert any("trade[1]" in w for w in aged)

    def test_non_swap_trades_do_not_trigger_the_swap_warning(self):
        """A swaption does not produce the swap warning."""
        swaption = SwaptionConfig(
            notional=1_500_000.0, fixed_rate=0.031, payer=True, rate_factor_index=0,
            hw_a=HW_A, hw_sigma=HW_SIGMA, initial_zero_curve=ZERO_CURVE,
            swap_tenor="1Y", forward_start=ORE.Period(1, ORE.Years),
            evaluation_date=TODAY,
        )
        trades = [swaption]
        sim = _sim_config(trades, time_grid=[0.0, 0.5, 1.0, 1.5])
        result = price_portfolio(PortfolioRequest(market=sim, trades=trades))
        assert not [w for w in result.warnings if "already started accruing" in w]

    def test_warning_does_not_block_pricing(self):
        """The warning does not block pricing: the result is complete and finite."""
        trades = [_swap(tenor="2Y")]
        sim = _sim_config(trades, time_grid=[0.0, 0.5, 1.0, 1.5])
        result = price_portfolio(PortfolioRequest(market=sim, trades=trades))
        assert np.isfinite(result.base_npv)
        assert bool(jnp.all(jnp.isfinite(result.npv_cube)))
