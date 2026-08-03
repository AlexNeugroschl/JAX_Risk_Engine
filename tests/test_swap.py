import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.instruments.swap import SwapConfig, price_swaps, _maturity_indices
from engine.scenarios import EVAL_DATE, SWAP_DEMO_MATURITIES

TODAY = EVAL_DATE
MATURITIES = np.array(SWAP_DEMO_MATURITIES)
DISCOUNT_RATE = 0.030
FORWARD_RATE = 0.035


def _single_scenario_yield_curves(make_flat_yield_curves) -> jnp.ndarray:
    """A [1, 1, Maturities, 2] cube built from two flat FlatForward curves,
    reproducing the deterministic t=0 case exactly (no Monte Carlo noise) so
    price_swaps can be cross-checked against ORE.VanillaSwap.NPV() directly."""
    return make_flat_yield_curves(DISCOUNT_RATE, FORWARD_RATE)


def _reference_ore_swap(payer: bool, notional: float, fixed_rate: float):
    """Mirrors engine.instruments.swap._build_ore_swap exactly
    (same custom IborIndex, same explicit Act/365Fixed on both legs) so this
    is a true apples-to-apples cross-check, not just "some ORE swap"."""
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
    """Regression coverage for a bug where a cashflow time landing just past
    the last maturity pillar produced an out-of-bounds np.searchsorted index
    that the validation silently accepted (checked against a *clipped*
    index) while returning the *unclipped* one -- which JAX then silently
    clips again on use instead of raising, so the cashflow would price off
    the wrong pillar with no error at all."""

    def test_time_past_last_pillar_raises(self):
        maturities = np.array([1.0, 2.0, 5.0, 10.0])
        with pytest.raises(ValueError):
            _maturity_indices(np.array([10.0000001]), maturities)

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
        """A cashflow time that falls strictly between two valid pillars
        (not past the last one, not before the first) must also be
        rejected -- searchsorted's in-bounds check alone isn't sufficient,
        the closeness check must independently catch this."""
        maturities = np.array([1.0, 2.0, 5.0, 10.0])
        with pytest.raises(ValueError):
            _maturity_indices(np.array([3.0]), maturities)

    def test_empty_times_returns_empty(self):
        """A swap with no cashflows on one side (not realistic for a vanilla
        swap, but the function itself should degrade gracefully) shouldn't
        crash on an empty input array."""
        maturities = np.array([1.0, 2.0, 5.0, 10.0])
        result = _maturity_indices(np.array([]), maturities)
        assert result.shape == (0,)

    def test_duplicate_cashflow_times_both_resolve(self):
        """Two distinct cashflows landing on the exact same pillar (e.g. a
        fixed and floating payment coinciding) must both resolve to that
        pillar's index, not error or silently drop one."""
        maturities = np.array([1.0, 2.0, 5.0, 10.0])
        result = _maturity_indices(np.array([2.0, 2.0]), maturities)
        np.testing.assert_array_equal(result, [1, 1])

    def test_single_pillar_maturities_array(self):
        maturities = np.array([1.0])
        result = _maturity_indices(np.array([1.0]), maturities)
        np.testing.assert_array_equal(result, [0])
        with pytest.raises(ValueError):
            _maturity_indices(np.array([1.0000001]), maturities)


class TestPriceSwapsAdditionalOREChecks:
    """Assumptions TestPriceSwapsAgainstORE's happy-path (par-ish 2Y swap,
    zero spread) doesn't exercise: a floating spread, single-curve
    discounting (discount_curve_index == forward_curve_index), a
    non-default index tenor, and a deeply off-market fixed rate (large
    positive NPV magnitude, not just near-zero)."""

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
        """discount_curve_index == forward_curve_index (both pointing at
        the SAME rate factor) must match ORE with a single flat curve used
        for both discounting and forwarding -- the multi-curve split is
        optional, not load-bearing."""
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
        """A fixed rate far from the forwarding curve's own rate (large
        NPV magnitude, not a near-par case) -- guards against an error that
        only shows up away from the small-NPV regime every other test
        happens to use."""
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
        """price_swaps' output for N trades must equal pricing each trade
        individually -- no cross-contamination between trades stacked in
        one call vs. priced one-at-a-time."""
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
        """Same pre-existing jnp.stack([]) limitation documented in
        engine.instruments.european_swaption's equivalent test -- a caller
        passing an empty swap_configs list gets a clear error, not a
        silently zero-trades NPV cube."""
        cube = _single_scenario_yield_curves(make_flat_yield_curves)
        with pytest.raises(ValueError):
            price_swaps(cube, MATURITIES, [])


class TestAgedSwapKnownLimitation:
    """Documents and pins down a real, previously-undetected limitation
    (see swap module docstring): price_swaps has no
    representation of an already-fixed/elapsed floating coupon. Pricing a
    swap at a simulated step_time AFTER its own first accrual date (true of
    EVERY step beyond t=0 for a spot-starting swap -- i.e. every existing
    demo/test scenario's swap, at every step index > 0) silently uses a
    meaningless clamped "discount factor" for the already-elapsed first
    floating period instead of excluding it or raising.

    This was found while building an end-to-end cross-check against ORE at
    a future evaluation date: direct comparison showed a real, non-noise
    discrepancy (~1e-4 to 1e-3 relative, growing with how far past the
    aged period's dates the evaluation point is) that traced back to
    exactly this cause -- confirmed by isolating the affected discount
    factor entries (P(t, T) for T < t) directly.
    """

    def test_t0_pricing_is_exact_no_aging_effect(self, make_flat_yield_curves):
        """At t=0, every cashflow is still in the future -- this is the
        unaffected, already-well-tested case, included here as the
        contrast baseline for the aged case below."""
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
        """Conditional on a future evaluation TIME past the swap's own
        first floating accrual boundary (T=~0.011y, i.e. any step_time >=
        the swap demo's SECOND time-grid point onward), price_swaps'
        output measurably diverges from ORE's own real swap object priced
        with an equivalent implied curve at that same future point -- the
        gap this test exists to document, not assert away. If this ever
        starts passing at a tight tolerance, the underlying limitation has
        been fixed and this test (and its docstring, and the module
        docstring's "Known limitation" section) should be updated/removed
        accordingly rather than left stale.

        Uses MATURITIES (the module's real 2Y-swap cashflow pillars,
        absolute time-from-TODAY) directly as both the yield_curves cube's
        maturity axis AND the conditioning reference for ORE's own implied
        curve, so there is no risk of the two paths silently using
        different maturity definitions.
        """
        a, sigma = 0.03, 0.01
        flat_rate = 0.03
        # Strictly between MATURITIES[0] (~0.011y, the first floating
        # accrual boundary) and MATURITIES[1] (~0.515y) -- past the first
        # boundary (the aged case) while not colliding with any pillar's
        # own (rounded-to-day) date.
        step_time = (MATURITIES[0] + MATURITIES[1]) / 2.0
        r_eval = flat_rate  # condition on the flat curve's own rate (no MC noise)

        ORE.Settings.instance().evaluationDate = TODAY
        dc = ORE.Actual365Fixed()
        curve0 = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, flat_rate, dc))
        hw0 = ORE.HullWhite(curve0, a, sigma)

        t_eval_date = TODAY + int(round(step_time * 365))
        # Extend well past the swap's own last cashflow (~2.01y from
        # TODAY) so ORE's curve never needs to extrapolate.
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

        # Build the SAME conditional discount cube this module's own
        # reconstruct_yield_curves formula would produce for MATURITIES at
        # step_time, conditional on r_eval.
        from engine.simulation import compute_hw_A_matrix, ZeroCurveConfig
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
        # documents the gap exists and is small-but-real, not catastrophic
        # or a total mispricing -- NOT a correctness assertion to preserve.
        assert rel_diff > 1e-5, (
            "This limitation appears to have been fixed (divergence from "
            "ORE is now below the documented gap's typical size) -- update "
            "this test and the module's 'Known limitation' docstring."
        )
        assert rel_diff < 0.5  # sanity: not a catastrophic mispricing
