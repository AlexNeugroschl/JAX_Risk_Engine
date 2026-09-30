import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.instruments.swap import SwapConfig, price_swaps, _maturity_indices
from demos.demo_scenarios import EVAL_DATE, SWAP_DEMO_MATURITIES

TODAY = EVAL_DATE
MATURITIES = np.array(SWAP_DEMO_MATURITIES)
DISCOUNT_RATE = 0.030
FORWARD_RATE = 0.035


def _single_scenario_yield_curves(make_flat_yield_curves) -> jnp.ndarray:
    """[1, 1, Maturities, 2] cube from two flat `FlatForward` curves: the deterministic t=0
    case, comparable with `ORE.VanillaSwap.NPV()`."""
    return make_flat_yield_curves(DISCOUNT_RATE, FORWARD_RATE)


def _reference_ore_swap(payer: bool, notional: float, fixed_rate: float):
    """The same swap in ORE as `_build_ore_swap` builds (same index, ACT/365 legs)."""
    ORE.Settings.instance().evaluationDate = TODAY
    dc = ORE.Actual365Fixed()
    fwd_curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, FORWARD_RATE, dc))
    disc_curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, DISCOUNT_RATE, dc))
    idx = ORE.IborIndex(
        "SimIndex", ORE.Period(6, ORE.Months), 2,
        ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False,
        dc, fwd_curve,
    )
    swap_type = ORE.VanillaSwap.Payer if payer else ORE.VanillaSwap.Receiver
    swap = ORE.MakeVanillaSwap(
        ORE.Period("2Y"), idx, fixed_rate,
        nominal=notional,
        swapType=swap_type,
        discountingTermStructure=disc_curve,
        fixedLegDayCount=dc,
        floatingLegDayCount=dc,
    )
    engine = ORE.DiscountingSwapEngine(disc_curve)
    swap.setPricingEngine(engine)
    return swap


class TestPriceSwapsAgainstORE:
    def test_matches_ore_npv_payer(self, make_flat_yield_curves):
        notional = 1_000_000.0
        fixed_rate = 0.03
        cfg = SwapConfig(
            notional=notional, fixed_rate=fixed_rate, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        npv_cube = price_swaps(_single_scenario_yield_curves(make_flat_yield_curves), MATURITIES, [cfg])
        our_npv = float(npv_cube[0, 0, 0])

        ore_swap = _reference_ore_swap(payer=True, notional=notional, fixed_rate=fixed_rate)
        ore_npv = ore_swap.NPV()

        np.testing.assert_allclose(our_npv, ore_npv, rtol=1e-6)

    def test_matches_ore_npv_receiver(self, make_flat_yield_curves):
        notional = 1_000_000.0
        fixed_rate = 0.03
        cfg = SwapConfig(
            notional=notional, fixed_rate=fixed_rate, payer=False,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        npv_cube = price_swaps(_single_scenario_yield_curves(make_flat_yield_curves), MATURITIES, [cfg])
        our_npv = float(npv_cube[0, 0, 0])

        ore_swap = _reference_ore_swap(payer=False, notional=notional, fixed_rate=fixed_rate)
        ore_npv = ore_swap.NPV()

        np.testing.assert_allclose(our_npv, ore_npv, rtol=1e-6)

    def test_par_rate_prices_near_zero(self, make_flat_yield_curves):
        notional = 1_000_000.0
        ore_swap = _reference_ore_swap(payer=True, notional=notional, fixed_rate=0.03)
        fair_rate = ore_swap.fairRate()

        cfg = SwapConfig(
            notional=notional, fixed_rate=fair_rate, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        npv_cube = price_swaps(_single_scenario_yield_curves(make_flat_yield_curves), MATURITIES, [cfg])
        np.testing.assert_allclose(float(npv_cube[0, 0, 0]), 0.0, atol=1.0)

    def test_payer_receiver_are_negations(self, make_flat_yield_curves):
        notional = 1_000_000.0
        fixed_rate = 0.03
        cube = _single_scenario_yield_curves(make_flat_yield_curves)
        payer_cfg = SwapConfig(
            notional=notional, fixed_rate=fixed_rate, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        receiver_cfg = SwapConfig(
            notional=notional, fixed_rate=fixed_rate, payer=False,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        payer_npv = float(price_swaps(cube, MATURITIES, [payer_cfg])[0, 0, 0])
        receiver_npv = float(price_swaps(cube, MATURITIES, [receiver_cfg])[0, 0, 0])
        np.testing.assert_allclose(payer_npv, -receiver_npv, rtol=1e-9)


class TestPriceSwapsShape:
    def test_output_shape_multi_trade_multi_scenario(self, make_flat_yield_curves):
        cube = jnp.tile(_single_scenario_yield_curves(make_flat_yield_curves), (8, 3, 1, 1))  # [S=8, T=3, M, 2]
        cfg = SwapConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        npv_cube = price_swaps(cube, MATURITIES, [cfg, cfg])
        assert npv_cube.shape == (8, 3, 2)

    def test_mismatched_maturities_raises(self, make_flat_yield_curves):
        cube = _single_scenario_yield_curves(make_flat_yield_curves)
        cfg = SwapConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        bad_maturities = np.array([1.0, 2.0])  # doesn't include the real cashflow dates
        with pytest.raises(ValueError):
            price_swaps(cube, bad_maturities, [cfg])


class TestMaturityIndicesBounds:
    """Regression: a cashflow time past the last pillar produced an out-of-bounds index
    that was validated against a clipped copy but returned unclipped; JAX then clipped it on
    use and priced off the wrong pillar silently. Times within 1e-6 of a pillar on either
    side match (TestMaturityIndicesFloatRoundoff); these use offsets beyond that."""

    def test_time_past_last_pillar_raises(self):
        maturities = np.array([1.0, 2.0, 5.0, 10.0])
        with pytest.raises(ValueError):
            _maturity_indices(np.array([10.1]), maturities)

    def test_time_below_first_pillar_raises(self):
        maturities = np.array([1.0, 2.0, 5.0, 10.0])
        with pytest.raises(ValueError):
            _maturity_indices(np.array([0.5]), maturities)

    def test_exact_pillar_matches_succeed(self):
        maturities = np.array([1.0, 2.0, 5.0, 10.0])
        np.testing.assert_array_equal(
            _maturity_indices(np.array([1.0, 5.0, 10.0]), maturities), [0, 2, 3]
        )

    def test_time_between_two_pillars_raises(self):
        """A time strictly between two pillars is rejected (the closeness check, not only
        the bounds check)."""
        maturities = np.array([1.0, 2.0, 5.0, 10.0])
        with pytest.raises(ValueError):
            _maturity_indices(np.array([3.0]), maturities)

    def test_empty_times_returns_empty(self):
        """An empty input returns an empty result."""
        maturities = np.array([1.0, 2.0, 5.0, 10.0])
        result = _maturity_indices(np.array([]), maturities)
        assert result.shape == (0,)

    def test_duplicate_cashflow_times_both_resolve(self):
        """Two cashflows on the same pillar both resolve to it."""
        maturities = np.array([1.0, 2.0, 5.0, 10.0])
        result = _maturity_indices(np.array([2.0, 2.0]), maturities)
        np.testing.assert_array_equal(result, [1, 1])

    def test_single_pillar_maturities_array(self):
        maturities = np.array([1.0])
        result = _maturity_indices(np.array([1.0]), maturities)
        np.testing.assert_array_equal(result, [0])
        with pytest.raises(ValueError):
            _maturity_indices(np.array([1.1]), maturities)


class TestPriceSwapsAdditionalOREChecks:
    """Beyond the par 2Y swap: a floating spread, single-curve discounting, another index
    tenor, and a far off-market fixed rate."""

    def _reference_ore_swap_with_spread(self, payer: bool, notional: float, fixed_rate: float, spread: float):
        ORE.Settings.instance().evaluationDate = TODAY
        dc = ORE.Actual365Fixed()
        fwd_curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, FORWARD_RATE, dc))
        disc_curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, DISCOUNT_RATE, dc))
        idx = ORE.IborIndex(
            "SimIndex", ORE.Period(6, ORE.Months), 2,
            ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False,
            dc, fwd_curve,
        )
        swap_type = ORE.VanillaSwap.Payer if payer else ORE.VanillaSwap.Receiver
        swap = ORE.MakeVanillaSwap(
            ORE.Period("2Y"), idx, fixed_rate,
            nominal=notional,
            swapType=swap_type,
            floatingLegSpread=spread,
            discountingTermStructure=disc_curve,
            fixedLegDayCount=dc,
            floatingLegDayCount=dc,
        )
        swap.setPricingEngine(ORE.DiscountingSwapEngine(disc_curve))
        return swap

    def test_matches_ore_with_floating_spread(self, make_flat_yield_curves):
        notional, fixed_rate, spread = 1_000_000.0, 0.03, 0.005
        cfg = SwapConfig(
            notional=notional, fixed_rate=fixed_rate, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="2Y", floating_spread=spread, evaluation_date=TODAY,
        )
        npv_cube = price_swaps(_single_scenario_yield_curves(make_flat_yield_curves), MATURITIES, [cfg])
        our_npv = float(npv_cube[0, 0, 0])

        ore_swap = self._reference_ore_swap_with_spread(True, notional, fixed_rate, spread)
        np.testing.assert_allclose(our_npv, ore_swap.NPV(), rtol=1e-6)

    def test_matches_ore_single_curve_discounting(self, make_flat_yield_curves):
        """Discount and forward curve the same factor matches ORE on one curve."""
        notional, fixed_rate = 1_000_000.0, 0.03
        cube = make_flat_yield_curves(disc_rate=DISCOUNT_RATE, fwd_rate=DISCOUNT_RATE)
        cfg = SwapConfig(
            notional=notional, fixed_rate=fixed_rate, payer=True,
            discount_curve_index=0, forward_curve_index=0,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        npv_cube = price_swaps(cube, MATURITIES, [cfg])
        our_npv = float(npv_cube[0, 0, 0])

        ORE.Settings.instance().evaluationDate = TODAY
        dc = ORE.Actual365Fixed()
        single_curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, DISCOUNT_RATE, dc))
        idx = ORE.IborIndex(
            "SimIndex", ORE.Period(6, ORE.Months), 2,
            ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False,
            dc, single_curve,
        )
        swap = ORE.MakeVanillaSwap(
            ORE.Period("2Y"), idx, fixed_rate,
            nominal=notional, swapType=ORE.VanillaSwap.Payer,
            discountingTermStructure=single_curve,
            fixedLegDayCount=dc, floatingLegDayCount=dc,
        )
        swap.setPricingEngine(ORE.DiscountingSwapEngine(single_curve))
        np.testing.assert_allclose(our_npv, swap.NPV(), rtol=1e-6)

    def test_matches_ore_deeply_off_market_fixed_rate(self, make_flat_yield_curves):
        """A fixed rate far from the forward rate (large NPV) matches ORE."""
        notional, fixed_rate = 1_000_000.0, 0.15
        cfg = SwapConfig(
            notional=notional, fixed_rate=fixed_rate, payer=False,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        npv_cube = price_swaps(_single_scenario_yield_curves(make_flat_yield_curves), MATURITIES, [cfg])
        our_npv = float(npv_cube[0, 0, 0])

        ore_swap = _reference_ore_swap(payer=False, notional=notional, fixed_rate=fixed_rate)
        np.testing.assert_allclose(our_npv, ore_swap.NPV(), rtol=1e-6)

    def test_multi_trade_portfolio_sums_correctly(self, make_flat_yield_curves):
        """N trades in one call equal N single-trade calls."""
        cube = _single_scenario_yield_curves(make_flat_yield_curves)
        cfg_a = SwapConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        cfg_b = SwapConfig(
            notional=2_500_000.0, fixed_rate=0.04, payer=False,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        combined = price_swaps(cube, MATURITIES, [cfg_a, cfg_b])
        individual_a = price_swaps(cube, MATURITIES, [cfg_a])
        individual_b = price_swaps(cube, MATURITIES, [cfg_b])
        np.testing.assert_allclose(float(combined[0, 0, 0]), float(individual_a[0, 0, 0]), rtol=1e-12)
        np.testing.assert_allclose(float(combined[0, 0, 1]), float(individual_b[0, 0, 0]), rtol=1e-12)

    def test_zero_notional_prices_to_zero(self, make_flat_yield_curves):
        cfg = SwapConfig(
            notional=0.0, fixed_rate=0.03, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        npv_cube = price_swaps(_single_scenario_yield_curves(make_flat_yield_curves), MATURITIES, [cfg])
        np.testing.assert_allclose(float(npv_cube[0, 0, 0]), 0.0, atol=1e-9)

    def test_empty_portfolio_raises_rather_than_silently_misbehaving(self, make_flat_yield_curves):
        """An empty list raises (`jnp.stack([])`), as for `price_swaptions`."""
        cube = _single_scenario_yield_curves(make_flat_yield_curves)
        with pytest.raises(ValueError):
            price_swaps(cube, MATURITIES, [])


class TestAgedSwapKnownLimitation:
    """Pins I-04 (FLAGGED): `price_swaps` has no representation of an elapsed floating
    coupon, so at a simulated step past the first accrual date (every step after t=0 for a
    spot-starting swap) the elapsed period uses a clamped P(t, T) with T < t. Against ORE at
    a later date the error is ~1e-4 to 1e-3 relative, growing with time past the aged
    dates."""

    def test_t0_pricing_is_exact_no_aging_effect(self, make_flat_yield_curves):
        """At t=0 every cashflow is in the future: exact (the baseline)."""
        notional, fixed_rate = 1_000_000.0, 0.03
        cfg = SwapConfig(
            notional=notional, fixed_rate=fixed_rate, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        npv_cube = price_swaps(_single_scenario_yield_curves(make_flat_yield_curves), MATURITIES, [cfg])
        ore_swap = _reference_ore_swap(payer=True, notional=notional, fixed_rate=fixed_rate)
        np.testing.assert_allclose(float(npv_cube[0, 0, 0]), ore_swap.NPV(), rtol=1e-6)

    def test_pricing_after_first_accrual_date_diverges_from_ore(self):
        """Past the first accrual boundary (~0.011y), `price_swaps` diverges from ORE priced
        on an equivalent implied curve at that date. This documents the gap; if it starts
        matching tightly, I-04 is fixed and this test and the docs should change.

        MATURITIES serves as both the cube's maturity axis and the reference for ORE's
        implied curve, so both sides use the same maturity definitions.
        """
        a, sigma = 0.03, 0.01
        flat_rate = 0.03
        # Between MATURITIES[0] (~0.011y) and MATURITIES[1] (~0.515y): past the first
        # boundary, and not on any pillar's date.
        step_time = (MATURITIES[0] + MATURITIES[1]) / 2.0
        r_eval = flat_rate  # condition on the flat curve's own rate (no MC noise)

        ORE.Settings.instance().evaluationDate = TODAY
        dc = ORE.Actual365Fixed()
        curve0 = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, flat_rate, dc))
        hw0 = ORE.HullWhite(curve0, a, sigma)

        t_eval_date = TODAY + int(round(step_time * 365))
        # Extend past the last cashflow (~2.01y) so ORE never extrapolates.
        curve_maturities_abs = list(MATURITIES[1:]) + [3.0]
        dates = [t_eval_date] + [TODAY + int(round(T * 365)) for T in curve_maturities_abs]
        discounts = [1.0] + [hw0.discountBond(step_time, T, r_eval) for T in curve_maturities_abs]

        ORE.Settings.instance().evaluationDate = t_eval_date
        implied_curve = ORE.YieldTermStructureHandle(ORE.DiscountCurve(dates, discounts, dc))
        index = ORE.IborIndex(
            "SimIndex", ORE.Period(6, ORE.Months), 2,
            ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False,
            dc, implied_curve,
        )
        swap = ORE.MakeVanillaSwap(
            ORE.Period("2Y"), index, 0.03,
            nominal=1_000_000.0, swapType=ORE.VanillaSwap.Payer,
            discountingTermStructure=implied_curve,
            fixedLegDayCount=dc, floatingLegDayCount=dc,
        )
        swap.setPricingEngine(ORE.DiscountingSwapEngine(implied_curve))
        ore_npv = swap.NPV()

        # The conditional discount cube `reconstruct_yield_curves` would give at step_time
        # for r_eval.
        from engine.simulation.market_model import compute_hw_A_matrix, ZeroCurveConfig
        zero_curves = [
            ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[flat_rate] * 6),
            ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[flat_rate] * 6),
        ]
        step_times = np.array([step_time])
        hw_a_arr = np.array([a, a])
        B = (1.0 - np.exp(-hw_a_arr[None, None, :] *
             np.maximum(MATURITIES[None, :, None] - step_times[:, None, None], 0.0))) / hw_a_arr[None, None, :]
        A = compute_hw_A_matrix(zero_curves, hw_a_arr, np.array([sigma, sigma]), step_times, MATURITIES, B)
        disc = A[0, :, 0] * np.exp(-B[0, :, 0] * r_eval)
        cube = jnp.asarray(np.stack([disc, disc], axis=-1)[None, None, :, :], dtype=jnp.float64)

        cfg = SwapConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        mine = float(price_swaps(cube, MATURITIES, [cfg])[0, 0, 0])

        rel_diff = abs(mine - ore_npv) / abs(ore_npv)
        # The gap exists and is small, not a total mispricing. Not a correctness claim.
        assert rel_diff > 1e-5, (
            "This limitation appears to have been fixed (divergence from "
            "ORE is now below the documented gap's typical size) -- update "
            "this test and the module's 'Known limitation' docstring."
        )
        assert rel_diff < 0.5  # sanity: not a catastrophic mispricing


# Edge-case tenors, rates and notionals; more than two curves; large portfolios; negative
# rates; `_maturity_indices` round-off.


def _ore_cashflow_pillars(payer: bool, notional: float, fixed_rate: float, swap_tenor: str,
                           index_tenor_months: int = 6, floating_spread: float = 0.0,
                           evaluation_date=TODAY) -> np.ndarray:
    """The swap's ORE schedule (built as `_build_ore_swap` would) as a sorted, de-duplicated
    array of payment and accrual times on both legs: the cube pillars for any tenor."""
    ORE.Settings.instance().evaluationDate = evaluation_date
    dc = ORE.Actual365Fixed()
    dummy_fwd = ORE.YieldTermStructureHandle(ORE.FlatForward(evaluation_date, 0.0, dc))
    idx = ORE.IborIndex(
        "SimIndex", ORE.Period(index_tenor_months, ORE.Months), 2,
        ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False,
        dc, dummy_fwd,
    )
    swap_type = ORE.VanillaSwap.Payer if payer else ORE.VanillaSwap.Receiver
    swap = ORE.MakeVanillaSwap(
        ORE.Period(swap_tenor), idx, fixed_rate,
        nominal=notional, swapType=swap_type,
        floatingLegSpread=floating_spread,
        fixedLegDayCount=dc, floatingLegDayCount=dc,
    )
    times = set()
    for cf in swap.fixedLeg():
        c = ORE.as_fixed_rate_coupon(cf)
        times.add(dc.yearFraction(evaluation_date, c.date()))
        times.add(dc.yearFraction(evaluation_date, c.accrualStartDate()))
        times.add(dc.yearFraction(evaluation_date, c.accrualEndDate()))
    for cf in swap.floatingLeg():
        c = ORE.as_floating_rate_coupon(cf)
        times.add(dc.yearFraction(evaluation_date, c.date()))
        times.add(dc.yearFraction(evaluation_date, c.accrualStartDate()))
        times.add(dc.yearFraction(evaluation_date, c.accrualEndDate()))
    return np.array(sorted(times))


def _flat_cube_for_pillars(maturities: np.ndarray, disc_rate: float, fwd_rate: float,
                            evaluation_date=TODAY) -> jnp.ndarray:
    """`flat_yield_curves` for an arbitrary pillar array."""
    dc = ORE.Actual365Fixed()
    disc_curve = ORE.YieldTermStructureHandle(ORE.FlatForward(evaluation_date, disc_rate, dc))
    fwd_curve = ORE.YieldTermStructureHandle(ORE.FlatForward(evaluation_date, fwd_rate, dc))
    disc = np.array([disc_curve.discount(evaluation_date + int(round(t * 365))) for t in maturities])
    fwd = np.array([fwd_curve.discount(evaluation_date + int(round(t * 365))) for t in maturities])
    cube = np.stack([disc, fwd], axis=-1)
    return jnp.asarray(cube[None, None, :, :], dtype=jnp.float64)


def _reference_ore_swap_generic(payer: bool, notional: float, fixed_rate: float, swap_tenor: str,
                                 disc_rate: float, fwd_rate: float, index_tenor_months: int = 6,
                                 floating_spread: float = 0.0, evaluation_date=TODAY) -> ORE.VanillaSwap:
    """`_reference_ore_swap` for any tenor and index tenor (ACT/365 legs)."""
    ORE.Settings.instance().evaluationDate = evaluation_date
    dc = ORE.Actual365Fixed()
    fwd_curve = ORE.YieldTermStructureHandle(ORE.FlatForward(evaluation_date, fwd_rate, dc))
    disc_curve = ORE.YieldTermStructureHandle(ORE.FlatForward(evaluation_date, disc_rate, dc))
    idx = ORE.IborIndex(
        "SimIndex", ORE.Period(index_tenor_months, ORE.Months), 2,
        ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False,
        dc, fwd_curve,
    )
    swap_type = ORE.VanillaSwap.Payer if payer else ORE.VanillaSwap.Receiver
    swap = ORE.MakeVanillaSwap(
        ORE.Period(swap_tenor), idx, fixed_rate,
        nominal=notional, swapType=swap_type,
        floatingLegSpread=floating_spread,
        discountingTermStructure=disc_curve,
        fixedLegDayCount=dc, floatingLegDayCount=dc,
    )
    swap.setPricingEngine(ORE.DiscountingSwapEngine(disc_curve))
    return swap


class TestSwapTenorEdgeCases:
    """Tenor, notional and rate extremes."""

    def test_single_cashflow_short_tenor(self):
        """A 6M swap on a 6M index: one cashflow per leg."""
        notional, fixed_rate, tenor = 1_000_000.0, 0.03, "6M"
        pillars = _ore_cashflow_pillars(True, notional, fixed_rate, tenor)
        cube = _flat_cube_for_pillars(pillars, DISCOUNT_RATE, FORWARD_RATE)
        cfg = SwapConfig(
            notional=notional, fixed_rate=fixed_rate, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor=tenor, evaluation_date=TODAY,
        )
        our_npv = float(price_swaps(cube, pillars, [cfg])[0, 0, 0])
        ore_swap = _reference_ore_swap_generic(True, notional, fixed_rate, tenor, DISCOUNT_RATE, FORWARD_RATE)
        np.testing.assert_allclose(our_npv, ore_swap.NPV(), rtol=1e-6)

    def test_very_long_tenor_30y(self):
        """A 30Y swap: dozens of cashflows per leg."""
        notional, fixed_rate, tenor = 1_000_000.0, 0.04, "30Y"
        pillars = _ore_cashflow_pillars(True, notional, fixed_rate, tenor)
        cube = _flat_cube_for_pillars(pillars, DISCOUNT_RATE, FORWARD_RATE)
        cfg = SwapConfig(
            notional=notional, fixed_rate=fixed_rate, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor=tenor, evaluation_date=TODAY,
        )
        our_npv = float(price_swaps(cube, pillars, [cfg])[0, 0, 0])
        ore_swap = _reference_ore_swap_generic(True, notional, fixed_rate, tenor, DISCOUNT_RATE, FORWARD_RATE)
        np.testing.assert_allclose(our_npv, ore_swap.NPV(), rtol=1e-6)

    def test_extremely_high_fixed_rate(self, make_flat_yield_curves):
        """A 50% fixed rate (large NPV) matches ORE."""
        notional, fixed_rate = 1_000_000.0, 0.50
        cfg = SwapConfig(
            notional=notional, fixed_rate=fixed_rate, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        npv_cube = price_swaps(_single_scenario_yield_curves(make_flat_yield_curves), MATURITIES, [cfg])
        our_npv = float(npv_cube[0, 0, 0])
        ore_swap = _reference_ore_swap(payer=True, notional=notional, fixed_rate=fixed_rate)
        np.testing.assert_allclose(our_npv, ore_swap.NPV(), rtol=1e-6)

    def test_extremely_low_negative_fixed_rate(self, make_flat_yield_curves):
        notional, fixed_rate = 1_000_000.0, -0.10
        cfg = SwapConfig(
            notional=notional, fixed_rate=fixed_rate, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        npv_cube = price_swaps(_single_scenario_yield_curves(make_flat_yield_curves), MATURITIES, [cfg])
        our_npv = float(npv_cube[0, 0, 0])
        ore_swap = _reference_ore_swap(payer=True, notional=notional, fixed_rate=fixed_rate)
        np.testing.assert_allclose(our_npv, ore_swap.NPV(), rtol=1e-6)

    def test_zero_notional_with_extreme_rate(self, make_flat_yield_curves):
        """Zero notional prices to exactly zero whatever the rate."""
        cfg = SwapConfig(
            notional=0.0, fixed_rate=5.0, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        npv_cube = price_swaps(_single_scenario_yield_curves(make_flat_yield_curves), MATURITIES, [cfg])
        assert np.isfinite(float(npv_cube[0, 0, 0]))
        np.testing.assert_allclose(float(npv_cube[0, 0, 0]), 0.0, atol=1e-9)

    def test_negative_notional_matches_ore_and_negates_positive(self, make_flat_yield_curves):
        """A negative notional matches ORE and is the exact negation of the positive one."""
        fixed_rate = 0.03
        cube = _single_scenario_yield_curves(make_flat_yield_curves)
        cfg_pos = SwapConfig(
            notional=1_000_000.0, fixed_rate=fixed_rate, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        cfg_neg = SwapConfig(
            notional=-1_000_000.0, fixed_rate=fixed_rate, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        npv_pos = float(price_swaps(cube, MATURITIES, [cfg_pos])[0, 0, 0])
        npv_neg = float(price_swaps(cube, MATURITIES, [cfg_neg])[0, 0, 0])
        np.testing.assert_allclose(npv_neg, -npv_pos, rtol=1e-9)

        ore_swap = _reference_ore_swap_generic(True, -1_000_000.0, fixed_rate, "2Y", DISCOUNT_RATE, FORWARD_RATE)
        np.testing.assert_allclose(npv_neg, ore_swap.NPV(), rtol=1e-6)

    def test_stub_period_tenor_matches_ore(self):
        """A 15M swap on a 6M index has a stub period; it matches ORE."""
        notional, fixed_rate, tenor = 1_000_000.0, 0.03, "15M"
        pillars = _ore_cashflow_pillars(True, notional, fixed_rate, tenor)
        cube = _flat_cube_for_pillars(pillars, DISCOUNT_RATE, FORWARD_RATE)
        cfg = SwapConfig(
            notional=notional, fixed_rate=fixed_rate, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor=tenor, evaluation_date=TODAY,
        )
        our_npv = float(price_swaps(cube, pillars, [cfg])[0, 0, 0])
        ore_swap = _reference_ore_swap_generic(True, notional, fixed_rate, tenor, DISCOUNT_RATE, FORWARD_RATE)
        np.testing.assert_allclose(our_npv, ore_swap.NPV(), rtol=1e-6)


class TestMultiCurveManyDistinctCurves:
    """Four distinct flat curves in one [1, 1, Maturities, 4] cube, with trades on different
    (discount, forward) pairs, each checked against ORE."""

    RATES = [0.010, 0.025, 0.040, 0.060]  # four genuinely distinct flat curves

    def _cube4(self) -> jnp.ndarray:
        dc = ORE.Actual365Fixed()
        curves = []
        for r in self.RATES:
            h = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, r, dc))
            curves.append(np.array([h.discount(TODAY + int(round(t * 365))) for t in MATURITIES]))
        cube = np.stack(curves, axis=-1)  # [Maturities, 4]
        return jnp.asarray(cube[None, None, :, :], dtype=jnp.float64)

    def test_four_distinct_curve_pairs_each_match_ore(self):
        cube4 = self._cube4()
        # Every curve in each role at least once, including crossed pairs.
        pairs = [(0, 1), (1, 0), (2, 3), (3, 2), (0, 3), (2, 0)]
        notional, fixed_rate = 1_000_000.0, 0.03
        cfgs = [
            SwapConfig(
                notional=notional, fixed_rate=fixed_rate, payer=True,
                discount_curve_index=d, forward_curve_index=f,
                swap_tenor="2Y", evaluation_date=TODAY,
            )
            for d, f in pairs
        ]
        npv_cube = price_swaps(cube4, MATURITIES, cfgs)
        assert npv_cube.shape == (1, 1, len(pairs))

        for i, (d, f) in enumerate(pairs):
            ore_swap = _reference_ore_swap_generic(
                True, notional, fixed_rate, "2Y",
                disc_rate=self.RATES[d], fwd_rate=self.RATES[f],
            )
            np.testing.assert_allclose(
                float(npv_cube[0, 0, i]), ore_swap.NPV(), rtol=1e-6,
                err_msg="mismatch for (discount_idx=%d, forward_idx=%d)" % (d, f),
            )

    def test_mixed_portfolio_distinct_curves_sums_independently(self):
        """Trades on different curve pairs in one call are independent."""
        cube4 = self._cube4()
        pairs = [(0, 2), (1, 3), (3, 0)]
        notionals = [1_000_000.0, 2_000_000.0, 500_000.0]
        rates = [0.02, 0.05, 0.035]
        payers = [True, False, True]
        cfgs = [
            SwapConfig(
                notional=n, fixed_rate=r, payer=p,
                discount_curve_index=d, forward_curve_index=f,
                swap_tenor="2Y", evaluation_date=TODAY,
            )
            for (d, f), n, r, p in zip(pairs, notionals, rates, payers)
        ]
        combined = price_swaps(cube4, MATURITIES, cfgs)
        for i, cfg in enumerate(cfgs):
            individual = price_swaps(cube4, MATURITIES, [cfg])
            np.testing.assert_allclose(
                float(combined[0, 0, i]), float(individual[0, 0, 0]), rtol=1e-12
            )
            ore_swap = _reference_ore_swap_generic(
                cfg.payer, cfg.notional, cfg.fixed_rate, "2Y",
                disc_rate=self.RATES[cfg.discount_curve_index],
                fwd_rate=self.RATES[cfg.forward_curve_index],
            )
            np.testing.assert_allclose(float(combined[0, 0, i]), ore_swap.NPV(), rtol=1e-6)


class TestLargeHeterogeneousPortfolio:
    """Forty 2Y swaps on the demo pillars, varying rate, notional, direction and curve
    pair, priced in one call and each checked against ORE."""

    def test_forty_trade_portfolio_matches_independent_ore_pricing(self, make_flat_yield_curves):
        rng = np.random.default_rng(20260730)
        n_trades = 40
        cube = make_flat_yield_curves(DISCOUNT_RATE, FORWARD_RATE)

        notionals = rng.uniform(-5_000_000.0, 5_000_000.0, n_trades)
        fixed_rates = rng.uniform(-0.02, 0.20, n_trades)
        payers = rng.integers(0, 2, n_trades).astype(bool)
        spreads = rng.uniform(-0.01, 0.01, n_trades)

        cfgs = [
            SwapConfig(
                notional=float(notionals[i]), fixed_rate=float(fixed_rates[i]), payer=bool(payers[i]),
                discount_curve_index=0, forward_curve_index=1,
                swap_tenor="2Y", floating_spread=float(spreads[i]), evaluation_date=TODAY,
            )
            for i in range(n_trades)
        ]

        npv_cube = price_swaps(cube, MATURITIES, cfgs)
        assert npv_cube.shape == (1, 1, n_trades)

        ore_npvs = []
        for cfg in cfgs:
            dc = ORE.Actual365Fixed()
            ORE.Settings.instance().evaluationDate = TODAY
            fwd_curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, FORWARD_RATE, dc))
            disc_curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, DISCOUNT_RATE, dc))
            idx = ORE.IborIndex(
                "SimIndex", ORE.Period(6, ORE.Months), 2,
                ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False,
                dc, fwd_curve,
            )
            swap_type = ORE.VanillaSwap.Payer if cfg.payer else ORE.VanillaSwap.Receiver
            swap = ORE.MakeVanillaSwap(
                ORE.Period("2Y"), idx, cfg.fixed_rate,
                nominal=cfg.notional, swapType=swap_type,
                floatingLegSpread=cfg.floating_spread,
                discountingTermStructure=disc_curve,
                fixedLegDayCount=dc, floatingLegDayCount=dc,
            )
            swap.setPricingEngine(ORE.DiscountingSwapEngine(disc_curve))
            ore_npvs.append(swap.NPV())

        our_npvs = np.asarray(npv_cube[0, 0, :])
        np.testing.assert_allclose(our_npvs, np.array(ore_npvs), rtol=1e-6, atol=1e-3)

        # The random portfolio must contain both directions and both NPV signs.
        assert np.any(our_npvs > 0) and np.any(our_npvs < 0)
        assert np.any(payers) and np.any(~payers)

    def test_portfolio_sum_equals_sum_of_independent_single_trade_calls(self, make_flat_yield_curves):
        """One call equals the sum of single-trade calls (no shared state between trades)."""
        rng = np.random.default_rng(7)
        n_trades = 25
        cube = make_flat_yield_curves(DISCOUNT_RATE, FORWARD_RATE)
        notionals = rng.uniform(100_000.0, 10_000_000.0, n_trades)
        fixed_rates = rng.uniform(0.001, 0.10, n_trades)
        payers = rng.integers(0, 2, n_trades).astype(bool)

        cfgs = [
            SwapConfig(
                notional=float(notionals[i]), fixed_rate=float(fixed_rates[i]), payer=bool(payers[i]),
                discount_curve_index=0, forward_curve_index=1,
                swap_tenor="2Y", evaluation_date=TODAY,
            )
            for i in range(n_trades)
        ]
        combined = np.asarray(price_swaps(cube, MATURITIES, cfgs)[0, 0, :])
        individual = np.array([
            float(price_swaps(cube, MATURITIES, [cfg])[0, 0, 0]) for cfg in cfgs
        ])
        np.testing.assert_allclose(combined, individual, rtol=1e-12)


class TestNegativeRateNumericalStability:
    """An entirely negative curve (discount factors above 1) prices finitely and matches
    ORE."""

    NEG_DISC, NEG_FWD = -0.005, -0.002

    def test_negative_flat_curve_matches_ore_payer(self):
        notional, fixed_rate = 1_000_000.0, -0.003
        pillars = MATURITIES
        cube = _flat_cube_for_pillars(pillars, self.NEG_DISC, self.NEG_FWD)
        cfg = SwapConfig(
            notional=notional, fixed_rate=fixed_rate, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        our_npv = float(price_swaps(cube, pillars, [cfg])[0, 0, 0])
        assert np.isfinite(our_npv)
        ore_swap = _reference_ore_swap_generic(True, notional, fixed_rate, "2Y", self.NEG_DISC, self.NEG_FWD)
        np.testing.assert_allclose(our_npv, ore_swap.NPV(), rtol=1e-6)

    def test_negative_flat_curve_matches_ore_receiver(self):
        notional, fixed_rate = 1_000_000.0, -0.003
        pillars = MATURITIES
        cube = _flat_cube_for_pillars(pillars, self.NEG_DISC, self.NEG_FWD)
        cfg = SwapConfig(
            notional=notional, fixed_rate=fixed_rate, payer=False,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        our_npv = float(price_swaps(cube, pillars, [cfg])[0, 0, 0])
        assert np.isfinite(our_npv)
        ore_swap = _reference_ore_swap_generic(False, notional, fixed_rate, "2Y", self.NEG_DISC, self.NEG_FWD)
        np.testing.assert_allclose(our_npv, ore_swap.NPV(), rtol=1e-6)

    def test_negative_forward_curve_produces_negative_implied_forward_rate(self):
        """A negative forward curve gives a negative forward rate (not floored), matching
        ORE's floating leg."""
        pillars = MATURITIES
        cube = _flat_cube_for_pillars(pillars, self.NEG_DISC, self.NEG_FWD)
        cfg = SwapConfig(
            notional=1_000_000.0, fixed_rate=0.0, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        # fixed_rate=0 isolates the floating leg (payer NPV = float - fixed).
        our_npv = float(price_swaps(cube, pillars, [cfg])[0, 0, 0])
        ore_swap = _reference_ore_swap_generic(True, 1_000_000.0, 0.0, "2Y", self.NEG_DISC, self.NEG_FWD)
        assert our_npv < 0  # negative forward rate -> negative floating leg PV -> negative NPV vs a 0% fixed leg
        np.testing.assert_allclose(our_npv, ore_swap.NPV(), rtol=1e-6)

    def test_negative_rates_across_large_portfolio_match_ore(self):
        """A small portfolio under a negative curve matches ORE."""
        rng = np.random.default_rng(99)
        n_trades = 12
        pillars = MATURITIES
        cube = _flat_cube_for_pillars(pillars, self.NEG_DISC, self.NEG_FWD)
        notionals = rng.uniform(-2_000_000.0, 2_000_000.0, n_trades)
        fixed_rates = rng.uniform(-0.01, 0.01, n_trades)
        payers = rng.integers(0, 2, n_trades).astype(bool)
        cfgs = [
            SwapConfig(
                notional=float(notionals[i]), fixed_rate=float(fixed_rates[i]), payer=bool(payers[i]),
                discount_curve_index=0, forward_curve_index=1,
                swap_tenor="2Y", evaluation_date=TODAY,
            )
            for i in range(n_trades)
        ]
        npv_cube = price_swaps(cube, pillars, cfgs)
        our_npvs = np.asarray(npv_cube[0, 0, :])
        assert np.all(np.isfinite(our_npvs))

        ore_npvs = [
            _reference_ore_swap_generic(
                cfg.payer, cfg.notional, cfg.fixed_rate, "2Y", self.NEG_DISC, self.NEG_FWD
            ).NPV()
            for cfg in cfgs
        ]
        np.testing.assert_allclose(our_npvs, np.array(ore_npvs), rtol=1e-6, atol=1e-3)

    def test_near_zero_rate_curve_matches_ore(self):
        """Rates very close to zero match ORE."""
        notional, fixed_rate = 1_000_000.0, 1e-6
        tiny = 1e-7
        pillars = MATURITIES
        cube = _flat_cube_for_pillars(pillars, tiny, -tiny)
        cfg = SwapConfig(
            notional=notional, fixed_rate=fixed_rate, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        our_npv = float(price_swaps(cube, pillars, [cfg])[0, 0, 0])
        assert np.isfinite(our_npv)
        ore_swap = _reference_ore_swap_generic(True, notional, fixed_rate, "2Y", tiny, -tiny)
        np.testing.assert_allclose(our_npv, ore_swap.NPV(), rtol=1e-6, atol=1e-6)


class TestMaturityIndicesFloatRoundoff:
    """`_maturity_indices` uses `np.isclose(atol=1e-6)` with numpy's default rtol=1e-5:
    tight enough to reject real errors, loose enough for round-off from computing the same
    time two ways."""

    def test_ulp_level_roundoff_below_pillar_still_matches(self):
        """A pillar time a few ULPs low still matches."""
        maturities = np.array([1.0, 2.0136986301369864, 5.0, 10.0])
        eps = np.finfo(np.float64).eps
        perturbed = maturities[1] - 4 * eps * abs(maturities[1])
        result = _maturity_indices(np.array([perturbed]), maturities)
        np.testing.assert_array_equal(result, [1])

    def test_tolerance_boundary_just_inside_atol_below_pillar_matches(self):
        """Just inside atol=1e-6 below the pillar matches."""
        maturities = np.array([1.0, 2.0, 5.0, 10.0])
        perturbed = 5.0 - 9e-7
        result = _maturity_indices(np.array([perturbed]), maturities)
        np.testing.assert_array_equal(result, [2])

    def test_tolerance_boundary_just_outside_atol_raises(self):
        """Well outside the tolerance is rejected."""
        maturities = np.array([1.0, 2.0, 5.0, 10.0])
        perturbed = 5.0 - 1e-3  # well past atol + rtol*5.0 = 5.1e-5, below the pillar
        with pytest.raises(ValueError):
            _maturity_indices(np.array([perturbed]), maturities)

    def test_tolerance_is_symmetric_around_a_pillar(self):
        """Regression: `np.searchsorted` (side='left') sent a time a hair above a pillar to
        the next pillar's index, so 5.0000009 was rejected while 4.9999991 matched. The
        nearer of the two neighbours is now compared, so both directions match."""
        maturities = np.array([1.0, 2.0, 5.0, 10.0])
        just_below = 5.0 - 9e-7  # inside atol -> matches
        just_above = 5.0 + 9e-7  # inside atol by the same margin -> now also matches

        result_below = _maturity_indices(np.array([just_below]), maturities)
        np.testing.assert_array_equal(result_below, [2])

        result_above = _maturity_indices(np.array([just_above]), maturities)
        np.testing.assert_array_equal(result_above, [2])

    def test_relative_tolerance_at_large_maturity_below_pillar(self):
        """`np.isclose`'s rtol makes the tolerance atol + rtol*30 ~= 3.01e-4 at a 30Y pillar,
        ~300x looser than atol alone: a time 2e-4 (about 1.75 hours) off matches."""
        maturities = np.array([1.0, 2.0, 5.0, 30.0])
        near_miss = 30.0 - 2e-4  # inside atol + rtol*30.0 ~= 3.01e-4, below the pillar
        result = _maturity_indices(np.array([near_miss]), maturities)
        np.testing.assert_array_equal(result, [3])

        genuinely_off = 30.0 - 1.0  # far outside any plausible tolerance
        with pytest.raises(ValueError):
            _maturity_indices(np.array([genuinely_off]), maturities)

    def test_many_trades_different_maturity_subsets_share_pillar_array_correctly(self):
        """One pillar array reused across many calls with overlapping subsets (as
        `prepare_swap` does per leg and trade); calls do not affect each other."""
        maturities = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 20.0, 30.0])
        rng = np.random.default_rng(3)
        results = []
        for _ in range(30):
            subset = rng.choice(maturities, size=rng.integers(1, 5), replace=True)
            results.append((subset, _maturity_indices(subset, maturities)))
        for subset, idx in results:
            expected = np.searchsorted(maturities, subset)
            np.testing.assert_array_equal(idx, expected)
            np.testing.assert_allclose(maturities[idx], subset)

    def test_float32_vs_float64_time_representation_edge_case(self):
        """A float32-rounded time against float64 pillars. Documents the observed behaviour:
        the isclose tolerance covers the ~1e-7 float32 error here."""
        maturities = np.array([1.0, 2.0136986301369864, 5.0, 10.0], dtype=np.float64)
        as_f32 = np.float32(maturities[1])
        back_to_f64 = np.array([np.float64(as_f32)])
        f32_roundoff = abs(float(back_to_f64[0]) - maturities[1])
        result = _maturity_indices(back_to_f64, maturities)
        np.testing.assert_array_equal(result, [1])
        # The tolerance at this magnitude covers float32 round-off.
        assert f32_roundoff < 1e-6 + 1e-5 * maturities[1]


class TestSwapConfigValidation:
    """`SwapConfig.__post_init__` rejects non-finite notional/fixed_rate and unparseable
    tenors before any ORE call. Zero and negative values are valid."""

    def _base_kwargs(self):
        return dict(
            notional=1_000_000.0, fixed_rate=0.03, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="2Y", evaluation_date=TODAY,
        )

    def test_nan_notional_rejected(self):
        kwargs = self._base_kwargs()
        kwargs["notional"] = float("nan")
        with pytest.raises(ValueError, match="notional"):
            SwapConfig(**kwargs)

    def test_inf_notional_rejected(self):
        kwargs = self._base_kwargs()
        kwargs["notional"] = float("inf")
        with pytest.raises(ValueError, match="notional"):
            SwapConfig(**kwargs)

    def test_nan_fixed_rate_rejected(self):
        kwargs = self._base_kwargs()
        kwargs["fixed_rate"] = float("nan")
        with pytest.raises(ValueError, match="fixed_rate"):
            SwapConfig(**kwargs)

    def test_negative_inf_fixed_rate_rejected(self):
        kwargs = self._base_kwargs()
        kwargs["fixed_rate"] = float("-inf")
        with pytest.raises(ValueError, match="fixed_rate"):
            SwapConfig(**kwargs)

    def test_unparseable_swap_tenor_rejected(self):
        kwargs = self._base_kwargs()
        kwargs["swap_tenor"] = "not-a-tenor"
        with pytest.raises(ValueError, match="swap_tenor"):
            SwapConfig(**kwargs)

    def test_zero_and_negative_notional_still_valid(self):
        """Zero and negative notionals are still accepted."""
        kwargs = self._base_kwargs()
        kwargs["notional"] = 0.0
        SwapConfig(**kwargs)  # must not raise
        kwargs["notional"] = -1_000_000.0
        SwapConfig(**kwargs)  # must not raise

    def test_valid_config_constructs_without_error(self):
        SwapConfig(**self._base_kwargs())
