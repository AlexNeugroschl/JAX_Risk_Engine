import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.instruments.interest_rate_swap import SwapConfig, price_swaps, _maturity_indices
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
    """Mirrors engine.instruments.interest_rate_swap._build_ore_swap exactly
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
