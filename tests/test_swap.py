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
    the wrong pillar with no error at all.

    Times within atol=1e-6 of the last pillar (in either direction) are
    expected to MATCH, not raise -- see TestMaturityIndicesFloatRoundoff for
    the regression coverage on that symmetric-tolerance behavior. This class
    now uses a genuinely out-of-tolerance offset to test the true
    past-last-pillar rejection path."""

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
            _maturity_indices(np.array([1.1]), maturities)


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


# =============================================================================
# Additional coverage: edge-case tenors/rates/notionals, >2-curve portfolios,
# large heterogeneous portfolio diversity checks, negative-rate numerical
# stability, and _maturity_indices float-roundoff robustness.
# =============================================================================


def _ore_cashflow_pillars(payer: bool, notional: float, fixed_rate: float, swap_tenor: str,
                           index_tenor_months: int = 6, floating_spread: float = 0.0,
                           evaluation_date=TODAY) -> np.ndarray:
    """Builds the SAME ORE trade _build_ore_swap would (mirrored, not
    imported, since engine/instruments/swap.py is off-limits to import
    private helpers from beyond what's already exposed) and returns every
    cashflow-related year-fraction (payment/accrual-start/accrual-end, both
    legs) as a sorted, deduped maturity-pillar array -- the same recipe used
    to hand-derive SWAP_DEMO_MATURITIES in engine.scenarios, generalized to
    any tenor/index-tenor so edge-tenor swaps (single cashflow, 30Y+) get a
    correctly-sized yield_curves cube instead of hand-guessed pillars."""
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
    """Same recipe as engine.scenarios.flat_yield_curves, but for an
    arbitrary pillar array (not just SWAP_DEMO_MATURITIES) -- needed for
    edge-tenor swaps (6M, 30Y) whose cashflow dates don't land on the 2Y
    demo's pillars."""
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
    """Generalization of _reference_ore_swap for an arbitrary tenor/rate
    pair, matching _build_ore_swap's conventions (explicit Act/365Fixed both
    legs, given index tenor)."""
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
    """Tenor/notional/rate extremes not exercised by the fixed 2Y demo
    swap used everywhere else in this file."""

    def test_single_cashflow_short_tenor(self):
        """A 6M swap with a 6M index tenor produces exactly one cashflow
        per leg -- the minimal-degenerate-case boundary for the
        vectorized per-cashflow sum (a sum of one term)."""
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
        """A 30Y+ swap has dozens of cashflows per leg -- exercises the
        vectorized tensordot/sum over a much larger cashflow axis than the
        2Y demo (5 fixed / 4 floating)."""
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
        """A far-off-market, very high (50%) fixed rate -- large-magnitude
        NPV, still must match ORE exactly (no overflow/precision issue at
        this scale)."""
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
        """notional=0 combined with a large nonzero fixed rate must still
        price to exactly zero (rate doesn't matter when scaled by zero
        notional) -- guards against a latent NaN/inf from e.g. a
        div-by-zero on the notional somewhere in the pipeline."""
        cfg = SwapConfig(
            notional=0.0, fixed_rate=5.0, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        npv_cube = price_swaps(_single_scenario_yield_curves(make_flat_yield_curves), MATURITIES, [cfg])
        assert np.isfinite(float(npv_cube[0, 0, 0]))
        np.testing.assert_allclose(float(npv_cube[0, 0, 0]), 0.0, atol=1e-9)

    def test_negative_notional_matches_ore_and_negates_positive(self, make_flat_yield_curves):
        """A negative notional (short position) must match ORE's own
        negative-nominal swap directly, and must equal the exact negation
        of the same trade with a positive notional (linear in notional)."""
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
        """An odd (non-integer-multiple-of-index-tenor) swap tenor, e.g.
        15M with a 6M index, forces MakeVanillaSwap to generate a short
        stub period on at least one leg -- exercises the accrual-fraction
        extraction machinery on a non-regular period, not just the neat
        2Y/6M demo case."""
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
    """The existing suite only ever exercises exactly two distinct curves
    (discount_curve_index=0, forward_curve_index=1) or the single-curve
    (0, 0) case. Real multi-curve books mix several trades across several
    genuinely distinct discount/forward curve pairs in one portfolio --
    this class builds a [1, 1, Maturities, 4] cube from FOUR flat curves at
    different rates and prices several trades that each pick a different
    (discount, forward) pair, cross-checked per-trade against an
    independently constructed ORE swap using that exact pair."""

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
        # (discount_idx, forward_idx) pairs covering every curve at least
        # once on each role, including reused/crossed combinations.
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
        """A portfolio where each trade uses a DIFFERENT curve pair must
        still price each trade independently of the others in the same
        call -- no cross-talk introduced by the extra curve axis width."""
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
    """A strong end-to-end diversity check: dozens of swaps spanning varied
    tenor, fixed rate, notional, sign (payer/receiver), and curve pair, all
    priced together through price_swaps in ONE call, each verified against
    an independently-built single-swap ORE NPV. Because every trade shares
    the same maturity pillar set (2Y, semi-annual index -- SWAP_DEMO_MATURITIES),
    this also stresses _maturity_indices being reused correctly across many
    trades with the SAME cashflow times but different (notional, rate, sign,
    curve) combinations."""

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

        # Diversity sanity: the random portfolio must actually contain both
        # signs of NPV and both payer/receiver trades -- otherwise this
        # "diversity" test would be silently degenerate (e.g. rng seed
        # producing all-payer or all-positive-NPV trades).
        assert np.any(our_npvs > 0) and np.any(our_npvs < 0)
        assert np.any(payers) and np.any(~payers)

    def test_portfolio_sum_equals_sum_of_independent_single_trade_calls(self, make_flat_yield_curves):
        """Beyond matching ORE, price_swaps' per-trade output for a large
        portfolio priced in one call must equal the SUM one would get by
        calling price_swaps independently, trade-by-trade -- no shared
        state (e.g. an accidentally-reused index array) leaking between
        trades stacked in one call."""
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
    """ORE and real markets both support negative-rate environments (EUR/
    CHF/JPY post-2015). The pricer's forward-rate formula
    F = (P_start/P_end - 1) / accrual and its discount-factor lookups must
    stay well-behaved (finite, matching ORE) when the ENTIRE curve is
    negative, not just the fixed rate -- a negative zero curve produces
    discount factors ABOVE 1.0, which is the actual numerically-sensitive
    regime (accumulation, not decay)."""

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
        """Sanity anchor for the cross-check above: a negative forward
        curve must actually imply a negative simulated forward rate (not
        get silently floored/clamped at zero somewhere), matching ORE's
        own fairRate() for the floating leg under the same negative curve."""
        pillars = MATURITIES
        cube = _flat_cube_for_pillars(pillars, self.NEG_DISC, self.NEG_FWD)
        cfg = SwapConfig(
            notional=1_000_000.0, fixed_rate=0.0, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        # fixed_rate=0 isolates the floating leg's PV (NPV = floatLegPV since payer NPV = float - fixed)
        our_npv = float(price_swaps(cube, pillars, [cfg])[0, 0, 0])
        ore_swap = _reference_ore_swap_generic(True, 1_000_000.0, 0.0, "2Y", self.NEG_DISC, self.NEG_FWD)
        assert our_npv < 0  # negative forward rate -> negative floating leg PV -> negative NPV vs a 0% fixed leg
        np.testing.assert_allclose(our_npv, ore_swap.NPV(), rtol=1e-6)

    def test_negative_rates_across_large_portfolio_match_ore(self):
        """Extends the single-trade negative-curve check to a small
        diverse portfolio (varied rate/notional/sign) under a negative
        curve environment, verified against independent ORE swaps."""
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
        """Rates extremely close to (but not exactly) zero -- the boundary
        between the positive- and negative-rate regimes, where a naive
        log/exp-based discount-factor formula could be most exposed to
        cancellation error."""
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
    """_maturity_indices' closeness check uses np.isclose(..., atol=1e-6)
    with default rtol=1e-5 (numpy's default). This class probes whether
    that tolerance is appropriately tight (rejects genuinely-off times a
    real bug could produce) while still being loose enough to tolerate
    ULP-level float roundoff on times computed two different ways (e.g. via
    day-count year-fraction arithmetic vs. a hand-built maturities array),
    plus reuse of the same maturity set across many trades with different
    (but overlapping) cashflow-time subsets."""

    def test_ulp_level_roundoff_below_pillar_still_matches(self):
        """A pillar time perturbed DOWNWARD by a few ULPs (float64 machine
        epsilon magnitude) -- the kind of noise that arises from computing
        'the same' year-fraction via two different but mathematically
        equivalent call sequences -- must still resolve successfully.
        Perturbed downward specifically: see
        test_tolerance_is_asymmetric_due_to_searchsorted_side below for why
        an upward ULP perturbation is NOT equivalent here."""
        maturities = np.array([1.0, 2.0136986301369864, 5.0, 10.0])
        eps = np.finfo(np.float64).eps
        perturbed = maturities[1] - 4 * eps * abs(maturities[1])
        result = _maturity_indices(np.array([perturbed]), maturities)
        np.testing.assert_array_equal(result, [1])

    def test_tolerance_boundary_just_inside_atol_below_pillar_matches(self):
        """A perturbation just inside the documented atol=1e-6 absolute
        tolerance, BELOW the pillar, must match (see the asymmetry test
        below for why "below" is the operative word here, not just
        "close")."""
        maturities = np.array([1.0, 2.0, 5.0, 10.0])
        perturbed = 5.0 - 9e-7
        result = _maturity_indices(np.array([perturbed]), maturities)
        np.testing.assert_array_equal(result, [2])

    def test_tolerance_boundary_just_outside_atol_raises(self):
        """A perturbation well outside atol=1e-6 (and outside isclose's
        rtol=1e-5 relative term at this magnitude) must be rejected --
        pins down that the tolerance has a real, finite edge rather than
        silently matching anything close enough by eye."""
        maturities = np.array([1.0, 2.0, 5.0, 10.0])
        perturbed = 5.0 - 1e-3  # well past atol + rtol*5.0 = 5.1e-5, below the pillar
        with pytest.raises(ValueError):
            _maturity_indices(np.array([perturbed]), maturities)

    def test_tolerance_is_symmetric_around_a_pillar(self):
        """Regression test for a fixed bug: _maturity_indices used to have
        an ASYMMETRIC tolerance around a pillar despite using np.isclose
        (which is itself symmetric), because plain np.searchsorted's
        default side='left' sends any time even a single ULP GREATER than
        a pillar to the NEXT pillar's index, so the closeness check
        compared against the wrong neighbor and rejected times that were
        genuinely within atol=1e-6 -- while the same-magnitude downward
        perturbation matched fine. Concretely, 5.0000009 (up by 9e-7) used
        to be rejected while 4.9999991 (down by the same 9e-7) was
        accepted, even though both are within the documented tolerance.

        Fixed by comparing against BOTH of searchsorted's neighboring
        pillars and picking whichever is actually closer before the
        tolerance check, so a roundoff-perturbed time matches its intended
        pillar regardless of which side it lands on. This test now checks
        both directions match symmetrically.
        """
        maturities = np.array([1.0, 2.0, 5.0, 10.0])
        just_below = 5.0 - 9e-7  # inside atol -> matches
        just_above = 5.0 + 9e-7  # inside atol by the same margin -> now also matches

        result_below = _maturity_indices(np.array([just_below]), maturities)
        np.testing.assert_array_equal(result_below, [2])

        result_above = _maturity_indices(np.array([just_above]), maturities)
        np.testing.assert_array_equal(result_above, [2])

    def test_relative_tolerance_at_large_maturity_below_pillar(self):
        """Documents a specific, possibly-surprising consequence of using
        np.isclose (atol + rtol combined) rather than a pure atol: at a
        large pillar value (e.g. 30.0, a realistic long-tenor maturity),
        the effective tolerance is atol + rtol*30.0 = 1e-6 + 3e-4 =
        ~3.01e-4 -- roughly 300x looser than the atol=1e-6 alone would
        suggest. A cashflow time off by 2e-4 (about 1.75 hours in
        year-fraction terms) BELOW this maturity currently matches the
        pillar rather than raising. This is not necessarily wrong (day-
        count roundoff at 30Y could plausibly reach this size), but it is
        a real, non-obvious behavior worth pinning down explicitly rather
        than leaving implicit in np.isclose's default rtol. Uses a
        downward perturbation per the asymmetry documented above (an
        upward one of the same magnitude would raise instead)."""
        maturities = np.array([1.0, 2.0, 5.0, 30.0])
        near_miss = 30.0 - 2e-4  # inside atol + rtol*30.0 ~= 3.01e-4, below the pillar
        result = _maturity_indices(np.array([near_miss]), maturities)
        np.testing.assert_array_equal(result, [3])

        genuinely_off = 30.0 - 1.0  # far outside any plausible tolerance
        with pytest.raises(ValueError):
            _maturity_indices(np.array([genuinely_off]), maturities)

    def test_many_trades_different_maturity_subsets_share_pillar_array_correctly(self):
        """Reuses ONE maturities pillar array across many _maturity_indices
        calls with different (but overlapping) cashflow-time subsets --
        mirrors how prepare_swap calls _maturity_indices independently per
        leg per trade against the SAME shared maturities array in a
        portfolio. No call should be affected by any other call's input
        (the function is pure / stateless)."""
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
        """A cashflow time computed/stored at float32 precision (e.g. if
        an upstream array were accidentally narrowed) then compared
        against a float64 pillar array -- checks whether the float32
        rounding error (~1e-7 relative, can exceed atol=1e-6 in absolute
        terms for large-ish maturities) is still within the combined
        isclose tolerance, or whether it would cause a spurious rejection
        of a legitimately-matching pillar. This documents actual observed
        behavior rather than asserting a specific desired outcome is
        correct -- either outcome is informative here."""
        maturities = np.array([1.0, 2.0136986301369864, 5.0, 10.0], dtype=np.float64)
        as_f32 = np.float32(maturities[1])
        back_to_f64 = np.array([np.float64(as_f32)])
        f32_roundoff = abs(float(back_to_f64[0]) - maturities[1])
        result = _maturity_indices(back_to_f64, maturities)
        np.testing.assert_array_equal(result, [1])
        # combined isclose tolerance at this magnitude comfortably covers
        # float32 roundoff (~1e-7), consistent with the match above.
        assert f32_roundoff < 1e-6 + 1e-5 * maturities[1]
