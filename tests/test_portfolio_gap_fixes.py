"""
Regression tests for three integration-boundary gaps found while writing
docs/planning/eod-contract-proposal.md, each of which caused
`price_portfolio` to return a result that was *quietly* incomplete rather
than failing:

1. **Swap Greeks were silently skipped.** `engine.risk.greeks.
   swap_delta_gamma`/`swap_theta` were fully implemented and unit-tested,
   but `engine.portfolio.request._compute_all_greeks` had no access to the
   `SimulationConfig` a `SwapConfig`'s `discount_curve_index`/
   `forward_curve_index` resolve against, so it `continue`d past every
   swap. `compute_greeks=True` on a swap portfolio returned a `greeks`
   dict with no swap entries and no error.

2. **Bermudan Vega was never called.** `engine.risk.greeks.bermudan_vega`
   existed and was unit-tested, but nothing in the portfolio path invoked
   it, so a calibrated Bermudan never reported Vega through
   `price_portfolio` or the HTTP API.

3. **Per-trade base NPV did not exist.** `PortfolioResult.base_npv` was a
   single portfolio float, so a total could not be reconciled against the
   individual positions/contracts that produced it.

Plus a fourth, different in kind: the documented **aged-swap limitation**
(`engine.instruments.swap`'s module docstring,
`tests/test_swap.py::TestAgedSwapKnownLimitation`) is a real pricing
inaccuracy at every simulated step past a swap's first accrual date. It is
NOT fixed here (it needs per-scenario fixings that neither this engine nor
the TraderX export currently has). What IS fixed is its silence: it now
emits a warning that `price_portfolio` collects into
`PortfolioResult.warnings`.

Each test class below is written to FAIL against the pre-fix code, so it
pins the gap closed rather than merely exercising the new path.
"""
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

# A curve deliberately DIFFERENT from ZERO_CURVE, used as rate factor 1 so a
# test can prove a swap's Greeks are computed against the curve its own
# indexes name -- not merely against "some" curve that happens to be present.
STEEP_CURVE = ZeroCurveConfig(
    times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0],
    rates=[0.020, 0.025, 0.030, 0.040, 0.045, 0.050],
)

# Short grid whose steps all sit before the swap's first accrual end, keeping
# these tests fast; aged-swap behavior gets its own explicit grids below.
TIME_GRID = [0.0, 0.5, 1.0]


def _swap(notional=2_000_000.0, fixed_rate=0.032, payer=True,
          disc_idx=0, fwd_idx=0, tenor="2Y"):
    return SwapConfig(
        notional=notional, fixed_rate=fixed_rate, payer=payer,
        discount_curve_index=disc_idx, forward_curve_index=fwd_idx,
        swap_tenor=tenor, evaluation_date=TODAY,
    )


def _sim_config(trades, curves=None, time_grid=None, n_factors=1):
    """A `SimulationConfig` sized to `n_factors` rate factors, with maturity
    pillars derived from the trades themselves (never hand-picked)."""
    curves = curves or [ZERO_CURVE]
    pillars = derive_maturity_pillars(trades, TODAY)
    # joint_covariance is [equities + rates]; one equity factor here.
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
# GAP 1: SWAP GREEKS WERE SILENTLY SKIPPED
# =============================================================================
class TestSwapGreeksReachThePortfolioPath:
    """`compute_greeks=True` must return Delta/Gamma/Theta for SWAPS.

    Every test here fails against the pre-fix `_compute_all_greeks`, which
    `continue`d past `SwapConfig` and produced an empty (or swap-less)
    greeks dict with no error raised.
    """

    def test_swap_greeks_are_present_not_skipped(self):
        """The headline regression: a swap-only portfolio with
        compute_greeks=True returned `{}` before the fix."""
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
        """A skipped-then-defaulted Greek would read as zeros; a real
        sensitivity for a 2Y payer swap is neither zero nor NaN."""
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
        """The wiring must not alter the numbers: Greeks routed through
        `price_portfolio` must equal calling `engine.risk.greeks` directly
        with the curves the swap's own indexes name."""
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
        """A swap pointing at rate factor 1 must be differentiated against
        factor 1's own (deliberately different) curve.

        This is the test that would catch a "just default to curve 0" fix:
        against STEEP_CURVE the sensitivities differ materially from the
        flat-curve ones, so silently substituting curve 0 fails here.
        """
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
        """Swap Greeks must not disturb the existing index keying: in a
        mixed portfolio every trade index gets its own entry."""
        swap_cfg = _swap()
        swaption_cfg = SwaptionConfig(
            notional=1_500_000.0, fixed_rate=0.031, payer=True, rate_factor_index=0,
            hw_a=HW_A, hw_sigma=HW_SIGMA, initial_zero_curve=ZERO_CURVE,
            swap_tenor="1Y", forward_start=ORE.Period(1, ORE.Years),
            evaluation_date=TODAY,
        )
        # Swap deliberately placed SECOND, so a fix that appended swap
        # Greeks rather than keying them by index would misalign here.
        trades = [swaption_cfg, swap_cfg]
        result = price_portfolio(PortfolioRequest(
            market=_sim_config(trades), trades=trades, compute_greeks=True,
        ))

        assert set(result.greeks) == {0, 1}
        assert "discount_delta" in result.greeks[1], "index 1 is the swap"
        assert "delta" in result.greeks[0], "index 0 is the swaption"

    def test_out_of_range_curve_index_raises_naming_the_trade(self):
        """An unresolvable curve index must fail loudly rather than being
        clamped to an existing curve -- the silent-mispricing class this
        whole file exists to prevent."""
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
        """compute_greeks=False must still return None -- the fix must not
        make Greeks unconditional (they are expensive)."""
        trades = [_swap()]
        result = price_portfolio(PortfolioRequest(
            market=_sim_config(trades), trades=trades, compute_greeks=False,
        ))
        assert result.greeks is None


# =============================================================================
# GAP 1b (I-13): CURVE INDICES WERE VALIDATED ONLY ON THE GREEKS PATH
# =============================================================================
class TestCurveIndexValidatedBeforeAllPricing:
    """An invalid curve index must fail loudly on EVERY pricing path, not
    only when `compute_greeks=True`.

    **The bug (I-13), found by TraderX source review.** `_swap_curve_configs`
    validates, but only `_compute_all_greeks` called it, and that runs after
    `_base_npv_per_trade` and only when Greeks are requested. Base pricing
    indexed the curve list directly, so with the DEFAULT
    `compute_greeks=False` a negative index silently wrapped (-1 -> last
    curve) and the trade priced against a curve it was never booked against.

    Every test here is driven through `price_portfolio`, deliberately.
    `TestSwapGreeksReachThePortfolioPath::
    test_out_of_range_curve_index_raises_naming_the_trade` already covers
    the helper directly and passed throughout -- testing the guard proved
    nothing about the caller that skipped it. That is the whole lesson of
    this issue, so these tests enter where a real caller enters.
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
        """The full 2x2 TraderX asked for: negative AND out-of-range, with
        compute_greeks both True and False.

        Against the pre-fix code the two NEGATIVE cases at
        compute_greeks=False do not raise at all -- they return a plausible
        NPV. The out-of-range cases raise a bare `IndexError` from list
        indexing, which names neither the trade nor the field.
        """
        with pytest.raises(ValueError, match=rf"{offender}.*out of range"):
            self._price(disc_idx, fwd_idx, compute_greeks)

    def test_error_names_the_trade_and_the_field(self):
        """A caller with hundreds of trades needs to know WHICH trade and
        WHICH of its two indices -- an `IndexError` from list subscripting
        says neither."""
        with pytest.raises(ValueError) as exc:
            self._price(0, -1, compute_greeks=False)
        message = str(exc.value)
        assert "trade[0]" in message
        assert "forward_curve_index=-1" in message
        assert "SwapConfig" in message

    def test_negative_index_does_not_price_against_the_wrapped_curve(self):
        """The specific silent mispricing, pinned.

        With two DIFFERENT curves, `fwd_idx=-1` wrapped to curve 1 and
        returned curve 1's NPV -- finite, plausible, no warning. Pinning
        that the valid index still prices, and that the invalid one raises
        INSTEAD OF returning that value, is what makes this a regression
        test rather than a restatement of the one above.
        """
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
        """The guard must not become an obstacle: every in-range index,
        including the last curve named POSITIVELY, still prices."""
        result = self._price(0, 1, compute_greeks, n_curves=2)
        assert np.isfinite(result.base_npv_per_trade[0])
        assert result.base_npv_per_trade[0] != 0.0


# =============================================================================
# GAP 2: BERMUDAN VEGA WAS NEVER CALLED
# =============================================================================
class TestBermudanVegaReachesThePortfolioPath:
    """`bermudan_vega` existed and was tested, but no portfolio-level caller
    ever invoked it."""

    @staticmethod
    def _calibrated_request():
        exercise_times = [1.010958904109589, 2.0136986301369864]
        bermudan = BermudanSwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
            hw_a=HW_A, hw_sigma=None, initial_zero_curve=ZERO_CURVE,
            exercise_times=exercise_times, swap_tenor="3Y",
            evaluation_date=TODAY, n_per_std=32, std_devs=5.0,
        )
        targets = build_coterminal_basket(
            evaluation_date=TODAY, exercise_times=exercise_times,
            final_maturity_time=3.0, notional=1_000_000.0, payer=True,
            market_vols=[0.008, 0.0088],
            zero_curve=HwZeroCurve.from_config(ZERO_CURVE),
        )
        trades = [bermudan]
        return PortfolioRequest(
            market=_sim_config(trades), trades=trades, compute_greeks=True,
            calibration_targets=targets,
        ), targets

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

    def test_no_vega_for_a_flat_uncalibrated_sigma(self):
        """A flat hand-set hw_sigma has no market quote to be sensitive to;
        Vega must be omitted rather than fabricated from a meaningless
        bump."""
        bermudan = BermudanSwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
            hw_a=HW_A, hw_sigma=HW_SIGMA, initial_zero_curve=ZERO_CURVE,
            exercise_times=[1.010958904109589, 2.0136986301369864],
            swap_tenor="3Y", evaluation_date=TODAY, n_per_std=32, std_devs=5.0,
        )
        trades = [bermudan]
        result = price_portfolio(PortfolioRequest(
            market=_sim_config(trades), trades=trades, compute_greeks=True,
        ))
        assert "vega" not in result.greeks[0]
        # Delta/Gamma/Theta must still be there -- omitting Vega must not
        # suppress the Greeks that ARE well-defined.
        assert "delta" in result.greeks[0]
        assert "theta" in result.greeks[0]


# =============================================================================
# GAP 3: PER-TRADE BASE NPV DID NOT EXIST
# =============================================================================
class TestPerTradeBaseNpv:
    """`PortfolioResult.base_npv` alone cannot be reconciled against the
    positions that produced it."""

    def test_per_trade_values_are_returned_one_per_trade(self):
        trades = [_swap(notional=2_000_000.0), _swap(notional=500_000.0, tenor="3Y")]
        result = price_portfolio(PortfolioRequest(
            market=_sim_config(trades), trades=trades,
        ))
        assert len(result.base_npv_per_trade) == len(trades)
        assert all(np.isfinite(v) for v in result.base_npv_per_trade)

    def test_total_equals_sum_of_parts_exactly(self):
        """The reported total and the reported breakdown must be the same
        numbers, not two independent computations that could drift."""
        trades = [_swap(notional=2_000_000.0), _swap(notional=500_000.0, tenor="3Y")]
        result = price_portfolio(PortfolioRequest(
            market=_sim_config(trades), trades=trades,
        ))
        assert result.base_npv == pytest.approx(
            float(sum(result.base_npv_per_trade)), rel=0.0, abs=1e-9,
        )

    def test_values_are_attributable_to_the_right_trade(self):
        """Ordering must follow `request.trades`, so a row joins to the
        contract that produced it. A payer and an otherwise-identical
        receiver have opposite-signed NPVs -- a swapped/misordered
        breakdown fails here."""
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
# GAP 4: THE AGED-SWAP LIMITATION WAS SILENT
# =============================================================================
class TestAgedSwapWarningIsNotSilent:
    """The aged-swap inaccuracy itself is NOT fixed here (it needs
    per-scenario fixings). Its SILENCE is: a portfolio whose simulated grid
    advances past a swap's first accrual start must say so in
    `PortfolioResult.warnings`.

    See `engine.instruments.swap`'s module docstring and
    `tests/test_swap.py::TestAgedSwapKnownLimitation` for the underlying
    pricing gap this warning advertises.
    """

    def test_spot_starting_swap_over_a_long_grid_warns(self):
        """A spot-starting swap is aged at every step past its first
        accrual date -- the common case, previously silent."""
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
        """A swap that is never aged WITHIN the simulated grid must not
        warn -- otherwise the warning is noise that trains users to ignore
        it, and stops distinguishing exact results from approximated ones.

        `_warn_if_aged_swap_exposure` compares the last simulated step
        against the swap's own first accrual start, so a grid whose final
        step lands before that boundary is exact throughout. (A spot-
        starting swap's first accrual begins ~2 business days out, so this
        uses a very short grid to stay strictly inside it.)
        """
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
        """The aged-swap warning is specific to `SwapConfig`; a swaption
        must not produce it."""
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
        """It is a warning, not an error: the result must still be a
        complete, finite, usable valuation."""
        trades = [_swap(tenor="2Y")]
        sim = _sim_config(trades, time_grid=[0.0, 0.5, 1.0, 1.5])
        result = price_portfolio(PortfolioRequest(market=sim, trades=trades))
        assert np.isfinite(result.base_npv)
        assert bool(jnp.all(jnp.isfinite(result.npv_cube)))
