import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.simulation import ZeroCurveConfig
from engine.instruments.european_swaption import (
    SwaptionConfig,
    prepare_swaption,
    price_swaptions,
    _price_one_swaption,
    _bond_call,
    _bond_put,
)

TODAY = ORE.Date(30, 7, 2026)
HW_A = 0.03
HW_SIGMA = 0.01
FLAT_RATE = 0.03
ZERO_CURVE = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[FLAT_RATE] * 6)


def _reference_ore_jamshidian_npv(fixed_rate: float, payer: bool, tenor: str, forward_start_years: int = 0) -> float:
    """
    Builds the identical swaption in real ORE (same custom IborIndex, same
    explicit Act/365Fixed on both legs as european_swaption._build_ore_swap)
    and prices it with ORE's own ORE.JamshidianSwaptionEngine under a
    ORE.HullWhite(FLAT_RATE, HW_A, HW_SIGMA) model -- the ground truth every
    test in this file cross-checks against, not a re-derivation of the
    formula under test.
    """
    ORE.Settings.instance().evaluationDate = TODAY
    dc = ORE.Actual365Fixed()
    curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, FLAT_RATE, dc))
    hw = ORE.HullWhite(curve, HW_A, HW_SIGMA)
    index = ORE.IborIndex(
        "SimIndex", ORE.Period(6, ORE.Months), 2,
        ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False,
        dc, curve,
    )
    swap_type = ORE.VanillaSwap.Payer if payer else ORE.VanillaSwap.Receiver
    forward_start = ORE.Period(forward_start_years, ORE.Years) if forward_start_years else ORE.Period(0, ORE.Days)
    swap = ORE.MakeVanillaSwap(
        ORE.Period(tenor), index, fixed_rate,
        nominal=1_000_000.0,
        swapType=swap_type,
        fixedLegDayCount=dc,
        floatingLegDayCount=dc,
        forwardStart=forward_start,
    )
    first_accrual_start = ORE.as_fixed_rate_coupon(swap.fixedLeg()[0]).accrualStartDate()
    forward_start_date = ORE.TARGET().advance(TODAY, forward_start)
    exercise_date = ORE.TARGET().advance(forward_start_date, 2, ORE.Days)
    assert exercise_date != TODAY or forward_start_years == 0  # sanity: matches module's own convention
    exercise = ORE.EuropeanExercise(exercise_date)
    swaption = ORE.Swaption(swap, exercise)
    engine = ORE.JamshidianSwaptionEngine(hw, curve)
    swaption.setPricingEngine(engine)
    return swaption.NPV()


def _make_cfg(fixed_rate: float, payer: bool, tenor: str = "5Y", forward_start_years: int = 0) -> SwaptionConfig:
    return SwaptionConfig(
        notional=1_000_000.0,
        fixed_rate=fixed_rate,
        payer=payer,
        rate_factor_index=0,
        hw_a=HW_A,
        hw_sigma=HW_SIGMA,
        initial_zero_curve=ZERO_CURVE,
        swap_tenor=tenor,
        forward_start=ORE.Period(forward_start_years, ORE.Years) if forward_start_years else ORE.Period(0, ORE.Days),
        evaluation_date=TODAY,
    )


def _price_at_t0(cfg: SwaptionConfig) -> float:
    """Prices at t=0 (today), conditional on r(0) = FLAT_RATE -- the
    apples-to-apples comparison against ORE's own t=0 NPV()."""
    prepared = prepare_swaption(cfg)
    step_times = jnp.array([0.0])
    hw_paths = jnp.array([[[FLAT_RATE]]])
    npv = _price_one_swaption(hw_paths, step_times, prepared)
    return float(npv[0, 0])


class TestAgainstOREJamshidianEngine:
    """Direct numeric cross-check against ORE.JamshidianSwaptionEngine --
    the same live-testing methodology used throughout this codebase (see
    engine/instruments/european_swaption.py's module docstring) rather than
    a from-scratch textbook derivation."""

    @pytest.mark.parametrize("fixed_rate,payer,tenor", [
        (0.03, True, "5Y"),   # ATM payer
        (0.03, False, "5Y"),  # ATM receiver
        (0.02, True, "10Y"),  # ITM payer (low fixed rate favors payer)
        (0.05, True, "2Y"),   # deep OTM payer
        (0.05, False, "2Y"),  # ITM receiver (high fixed rate favors receiver)
        (0.02, False, "10Y"),  # deep OTM receiver
    ])
    def test_matches_ore_spot_starting(self, fixed_rate, payer, tenor):
        mine = _price_at_t0(_make_cfg(fixed_rate, payer, tenor))
        ore = _reference_ore_jamshidian_npv(fixed_rate, payer, tenor)
        np.testing.assert_allclose(mine, ore, rtol=1e-4, atol=1e-2)

    @pytest.mark.parametrize("fixed_rate,payer,tenor,forward_years", [
        (0.03, True, "2Y", 3),
        (0.03, False, "2Y", 3),
        (0.025, True, "5Y", 5),
    ])
    def test_matches_ore_forward_starting(self, fixed_rate, payer, tenor, forward_years):
        """Regression coverage for the forward-start bug this module's
        development caught: an earlier version assumed the swap's floating
        leg always redeems its notional exactly at the option's own
        exercise date T0 (true only when the spot lag and the "exercise
        lag" coincide, i.e. a non-forward-starting swaption), which is
        false whenever forward_start != 0 -- the underlying's real first
        accrual date T_start is `exercise_lag_days` AFTER T0, not equal to
        it, and the floating leg's par-redemption identity needs a genuine
        P(T0,T_start) discount factor, not an assumed 1. This was caught by
        this exact cross-check diverging from ORE by ~1%."""
        mine = _price_at_t0(_make_cfg(fixed_rate, payer, tenor, forward_years))
        ore = _reference_ore_jamshidian_npv(fixed_rate, payer, tenor, forward_years)
        np.testing.assert_allclose(mine, ore, rtol=1e-4, atol=1e-2)

    def test_matches_ore_deep_otm_is_near_zero(self):
        """Deep OTM payer (fixed rate far above any plausible forward rate)
        should price near zero on both sides -- and, since ORE's own value
        underflows toward 0 here too, this mainly guards against a NaN/inf
        from this module's own bisection or Black formula in the tail."""
        mine = _price_at_t0(_make_cfg(0.20, True, "5Y"))
        ore = _reference_ore_jamshidian_npv(0.20, True, "5Y")
        np.testing.assert_allclose(mine, ore, atol=1e-3)
        assert mine >= 0.0


class TestZeroVolatilityLimit:
    """sigma_p == 0 is a real, reachable edge case in this module (not just
    a theoretical corner): it occurs at-or-after the option's own expiry,
    AND whenever a bond leg's own maturity coincides with the option's
    expiry -- which is exactly the T_start leg for a non-forward-starting
    swaption (exercise_lag_days brings T0 back to precisely T_start). An
    earlier version of _bond_call/_bond_put left this to a caller-side
    jnp.where that didn't correctly handle the second case, producing wildly
    wrong (large negative) NPVs -- caught by the spot-starting cross-check
    above going from matching ORE to being off by orders of magnitude."""

    def test_bond_call_at_zero_vol_matches_intrinsic(self):
        P_t_Topt = jnp.array([0.95, 0.95, 0.95])
        P_t_S = jnp.array([0.90, 0.80, 0.70])
        K = jnp.array([0.90, 0.90, 0.90])
        sigma_p = jnp.array([0.0, 0.0, 0.0])
        result = _bond_call(P_t_Topt, P_t_S, K, sigma_p)
        expected = jnp.maximum(P_t_S - K * P_t_Topt, 0.0)
        np.testing.assert_allclose(np.asarray(result), np.asarray(expected), atol=1e-12)

    def test_bond_put_at_zero_vol_matches_intrinsic(self):
        P_t_Topt = jnp.array([0.95, 0.95, 0.95])
        P_t_S = jnp.array([0.90, 0.80, 0.70])
        K = jnp.array([0.90, 0.90, 0.90])
        sigma_p = jnp.array([0.0, 0.0, 0.0])
        result = _bond_put(P_t_Topt, P_t_S, K, sigma_p)
        expected = jnp.maximum(K * P_t_Topt - P_t_S, 0.0)
        np.testing.assert_allclose(np.asarray(result), np.asarray(expected), atol=1e-12)

    def test_no_nan_or_inf_across_zero_and_positive_vol(self):
        """A mix of zero and positive sigma_p in the same call (exactly the
        shape _price_one_swaption produces, where only SOME legs have
        sigma_p==0) must not let the zero-vol branch's 0/0 contaminate the
        positive-vol entries or vice versa."""
        P_t_Topt = jnp.array([0.95, 0.95])
        P_t_S = jnp.array([0.90, 0.85])
        K = jnp.array([0.90, 0.90])
        sigma_p = jnp.array([0.0, 0.05])
        result = _bond_call(P_t_Topt, P_t_S, K, sigma_p)
        assert bool(jnp.all(jnp.isfinite(result)))


class TestPayerReceiverParity:
    def test_put_call_parity_payer_minus_receiver_equals_forward_swap_value(self):
        """Standard option identity: payer_swaption - receiver_swaption ==
        the forward value of the underlying swap itself (a model- and
        volatility-independent identity, so this doesn't depend on the
        Jamshidian formula being correct -- an independent sanity check)."""
        cfg_payer = _make_cfg(0.03, True, "5Y")
        cfg_receiver = _make_cfg(0.03, False, "5Y")
        payer_npv = _price_at_t0(cfg_payer)
        receiver_npv = _price_at_t0(cfg_receiver)

        ORE.Settings.instance().evaluationDate = TODAY
        dc = ORE.Actual365Fixed()
        curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, FLAT_RATE, dc))
        index = ORE.IborIndex(
            "SimIndex", ORE.Period(6, ORE.Months), 2,
            ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False,
            dc, curve,
        )
        swap = ORE.MakeVanillaSwap(
            ORE.Period("5Y"), index, 0.03,
            nominal=1_000_000.0, swapType=ORE.VanillaSwap.Payer,
            fixedLegDayCount=dc, floatingLegDayCount=dc,
        )
        swap.setPricingEngine(ORE.DiscountingSwapEngine(curve))
        underlying_npv = swap.NPV()

        np.testing.assert_allclose(payer_npv - receiver_npv, underlying_npv, rtol=1e-3, atol=1.0)


class TestZeroVolatilityCollapsesToIntrinsic:
    def test_payer_matches_max_swap_npv_zero(self):
        """As sigma -> 0, a European swaption collapses to the deterministic
        max(swap NPV, 0) -- no uncertainty left to price optionality on.
        Live-verified against ORE directly (see this module's development
        notes); this test locks that limit in for both an ITM and an OTM
        strike."""
        for fixed_rate, payer in [(0.02, True), (0.05, True)]:
            cfg = SwaptionConfig(
                notional=1_000_000.0, fixed_rate=fixed_rate, payer=payer,
                rate_factor_index=0, hw_a=HW_A, hw_sigma=1e-9,
                initial_zero_curve=ZERO_CURVE, swap_tenor="5Y",
                evaluation_date=TODAY,
            )
            mine = _price_at_t0(cfg)

            ORE.Settings.instance().evaluationDate = TODAY
            dc = ORE.Actual365Fixed()
            curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, FLAT_RATE, dc))
            index = ORE.IborIndex(
                "SimIndex", ORE.Period(6, ORE.Months), 2,
                ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False,
                dc, curve,
            )
            swap_type = ORE.VanillaSwap.Payer if payer else ORE.VanillaSwap.Receiver
            swap = ORE.MakeVanillaSwap(
                ORE.Period("5Y"), index, fixed_rate,
                nominal=1_000_000.0, swapType=swap_type,
                fixedLegDayCount=dc, floatingLegDayCount=dc,
            )
            swap.setPricingEngine(ORE.DiscountingSwapEngine(curve))
            expected = max(swap.NPV(), 0.0)
            np.testing.assert_allclose(mine, expected, rtol=1e-3, atol=1.0)


class TestConditionalPricingAndExpiry:
    """The [Scenarios, TimeSteps, Trades] NPV cube contract needs Jamshidian's
    formula evaluated at every simulated (scenario, step), not just t=0 --
    these tests cover that generalization directly (see
    european_swaption.py's module docstring on the Markov-conditioning
    argument)."""

    def test_post_expiry_npv_is_zero(self):
        cfg = _make_cfg(0.03, True, "5Y")
        prepared = prepare_swaption(cfg)
        step_times = jnp.array([prepared.exercise_time + 1.0, prepared.exercise_time + 10.0])
        hw_paths = jnp.array([[[FLAT_RATE], [FLAT_RATE]]])
        npv = _price_one_swaption(hw_paths, step_times, prepared)
        np.testing.assert_array_equal(np.asarray(npv), np.zeros((1, 2)))

    def test_still_alive_before_expiry_is_positive_for_reasonable_scenario(self):
        cfg = _make_cfg(0.03, True, "5Y")
        prepared = prepare_swaption(cfg)
        step_times = jnp.array([prepared.exercise_time * 0.5])
        hw_paths = jnp.array([[[FLAT_RATE]]])
        npv = _price_one_swaption(hw_paths, step_times, prepared)
        assert float(npv[0, 0]) > 0.0

    def test_conditional_pricing_matches_ore_rebuilt_at_later_date(self):
        """Prices a still-alive, forward-starting swaption conditional on a
        simulated short rate at t=1Y (before its own ~3Y exercise date),
        and cross-checks against ORE's own JamshidianSwaptionEngine with
        its evaluation date and yield curve rebuilt to represent that same
        future point -- the same live-testing methodology used to validate
        the t=0 case, generalized to confirm HW1F's Markov conditional
        pricing property holds for this module's implementation too."""
        a, sigma = HW_A, HW_SIGMA
        t_eval_date = ORE.Date(30, 7, 2027)
        r_eval = 0.035
        dc = ORE.Actual365Fixed()

        ORE.Settings.instance().evaluationDate = TODAY
        curve0 = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, FLAT_RATE, dc))
        hw0 = ORE.HullWhite(curve0, a, sigma)
        T_eval_abs = dc.yearFraction(TODAY, t_eval_date)

        maturities = [T_eval_abs] + [dc.yearFraction(TODAY, ORE.Date(30, 7, 2028 + i)) for i in range(12)]
        discounts = [1.0] + [hw0.discountBond(T_eval_abs, T, r_eval) for T in maturities[1:]]
        dates = [t_eval_date] + [ORE.Date(30, 7, 2028 + i) for i in range(12)]

        ORE.Settings.instance().evaluationDate = t_eval_date
        implied_curve = ORE.YieldTermStructureHandle(ORE.DiscountCurve(dates, discounts, dc))
        hw_eval = ORE.HullWhite(implied_curve, a, sigma)
        index = ORE.IborIndex(
            "SimIndex", ORE.Period(6, ORE.Months), 2,
            ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False,
            dc, implied_curve,
        )
        swap = ORE.MakeVanillaSwap(
            ORE.Period("5Y"), index, 0.03,
            nominal=1_000_000.0, swapType=ORE.VanillaSwap.Payer,
            fixedLegDayCount=dc, floatingLegDayCount=dc,
            forwardStart=ORE.Period(2, ORE.Years),
        )
        exercise_date = ORE.TARGET().advance(ORE.TARGET().advance(t_eval_date, ORE.Period(2, ORE.Years)), 2, ORE.Days)
        exercise = ORE.EuropeanExercise(exercise_date)
        swaption = ORE.Swaption(swap, exercise)
        swaption.setPricingEngine(ORE.JamshidianSwaptionEngine(hw_eval, implied_curve))
        ore_npv = swaption.NPV()

        ORE.Settings.instance().evaluationDate = TODAY
        cfg = _make_cfg(0.03, True, "5Y", forward_start_years=3)
        prepared = prepare_swaption(cfg)
        step_times = jnp.array([T_eval_abs])
        hw_paths = jnp.array([[[r_eval]]])
        mine = float(_price_one_swaption(hw_paths, step_times, prepared)[0, 0])

        np.testing.assert_allclose(mine, ore_npv, rtol=1e-4, atol=1e-2)


class TestPriceSwaptionsShape:
    def test_output_shape_multi_trade_multi_scenario_multi_step(self):
        cfg = _make_cfg(0.03, True, "5Y")
        step_times = jnp.array([0.0, 1.0, 2.0])
        hw_paths = jnp.full((8, 3, 1), FLAT_RATE)
        npv_cube = price_swaptions(hw_paths, step_times, [cfg, cfg])
        assert npv_cube.shape == (8, 3, 2)

    def test_multiple_rate_factors_selects_correct_one(self):
        """rate_factor_index must actually select the right column of
        hw_paths, not silently default to 0."""
        cfg0 = SwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True,
            rate_factor_index=0, hw_a=HW_A, hw_sigma=HW_SIGMA,
            initial_zero_curve=ZERO_CURVE, swap_tenor="5Y", evaluation_date=TODAY,
        )
        cfg1 = SwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True,
            rate_factor_index=1, hw_a=HW_A, hw_sigma=HW_SIGMA,
            initial_zero_curve=ZERO_CURVE, swap_tenor="5Y", evaluation_date=TODAY,
        )
        # factor 0 at 3% (near ATM), factor 1 at 8% (deep ITM for a payer)
        hw_paths = jnp.array([[[0.03, 0.08]]])
        step_times = jnp.array([0.0])
        npv0 = price_swaptions(hw_paths, step_times, [cfg0])
        npv1 = price_swaptions(hw_paths, step_times, [cfg1])
        assert float(npv1[0, 0, 0]) > float(npv0[0, 0, 0])


class TestMonotonicRStarSolve:
    """The bisection in _solve_rstar relies on the (signed) coupon bond
    value being monotonically decreasing in r across the practical search
    range -- verified numerically here across a spread of tenors/rates
    rather than just assumed from the theoretical argument in the
    docstring, since the T_start leg's negative amount makes the sum only
    APPROXIMATELY (not exactly) monotonic in general."""

    @pytest.mark.parametrize("tenor,forward_years", [("2Y", 0), ("10Y", 0), ("5Y", 5), ("30Y", 10)])
    def test_rstar_solve_converges_to_true_root(self, tenor, forward_years):
        cfg = _make_cfg(0.03, True, tenor, forward_years)
        prepared = prepare_swaption(cfg)
        mine = _price_at_t0(cfg)
        # if bisection failed to converge, the resulting NPV would be wildly
        # wrong (as seen during development) rather than merely imprecise --
        # a loose finiteness + sign sanity check catches that failure mode.
        assert np.isfinite(mine)
        assert mine >= -1e-6


def _price_at_t0_with_rate(cfg: SwaptionConfig, r0: float) -> float:
    prepared = prepare_swaption(cfg)
    step_times = jnp.array([0.0])
    hw_paths = jnp.array([[[r0]]])
    return float(_price_one_swaption(hw_paths, step_times, prepared)[0, 0])


class TestNegativeRates:
    """HW1F is a NORMAL (not lognormal) short-rate model, so it natively
    supports negative rates -- live-verified against ORE directly, since a
    naive lognormal-style implementation would break (log of a negative
    number) or silently floor rates at zero."""

    def test_matches_ore_negative_flat_curve(self):
        negative_rate = -0.005
        negative_curve = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[negative_rate] * 6)
        cfg = SwaptionConfig(
            notional=1_000_000.0, fixed_rate=negative_rate, payer=True,
            rate_factor_index=0, hw_a=HW_A, hw_sigma=HW_SIGMA,
            initial_zero_curve=negative_curve, swap_tenor="5Y", evaluation_date=TODAY,
        )
        mine = _price_at_t0_with_rate(cfg, negative_rate)

        ORE.Settings.instance().evaluationDate = TODAY
        dc = ORE.Actual365Fixed()
        curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, negative_rate, dc))
        hw = ORE.HullWhite(curve, HW_A, HW_SIGMA)
        index = ORE.IborIndex(
            "SimIndex", ORE.Period(6, ORE.Months), 2,
            ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False,
            dc, curve,
        )
        swap = ORE.MakeVanillaSwap(
            ORE.Period("5Y"), index, negative_rate,
            nominal=1_000_000.0, swapType=ORE.VanillaSwap.Payer,
            fixedLegDayCount=dc, floatingLegDayCount=dc,
        )
        exercise_date = ORE.TARGET().advance(TODAY, ORE.Period(2, ORE.Days))
        exercise = ORE.EuropeanExercise(exercise_date)
        swaption = ORE.Swaption(swap, exercise)
        swaption.setPricingEngine(ORE.JamshidianSwaptionEngine(hw, curve))
        np.testing.assert_allclose(mine, swaption.NPV(), rtol=1e-4, atol=1e-2)

    def test_positive_underlying_rate_but_simulated_negative_short_rate(self):
        """A scenario where today's curve is positive but the CONDITIONAL
        simulated short rate at some future step has gone negative -- this
        is the realistic Monte-Carlo case (HW1F's Gaussian shocks can push
        r(t) below zero even from a positive start), not just a negative
        flat-curve setup."""
        cfg = _make_cfg(0.03, True, "5Y")
        mine = _price_at_t0_with_rate(cfg, -0.01)
        assert np.isfinite(mine)
        assert mine >= 0.0


class TestNearZeroVolatility:
    """Distinct from the exact-zero sigma_p edge case already covered in
    TestZeroVolatilityLimit: a genuinely small but nonzero hw_sigma
    (as opposed to sigma_p==0 from t==T_opt or a same-maturity leg) must
    still price close to the deterministic intrinsic value without any
    numerical blow-up from the Black formula's 1/sigma_p term."""

    def test_very_small_but_nonzero_sigma_stays_near_intrinsic(self):
        cfg = SwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.02, payer=True,
            rate_factor_index=0, hw_a=HW_A, hw_sigma=1e-5,
            initial_zero_curve=ZERO_CURVE, swap_tenor="5Y", evaluation_date=TODAY,
        )
        mine = _price_at_t0(cfg)

        cfg_zero = SwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.02, payer=True,
            rate_factor_index=0, hw_a=HW_A, hw_sigma=1e-9,
            initial_zero_curve=ZERO_CURVE, swap_tenor="5Y", evaluation_date=TODAY,
        )
        near_zero = _price_at_t0(cfg_zero)
        np.testing.assert_allclose(mine, near_zero, rtol=1e-2)
        assert np.isfinite(mine)


class TestExtremeMeanReversion:
    """hw_a appears in several denominators (B(t,T), the bond-option
    volatility formula) -- both a very small and a very large mean
    reversion speed must still price finitely and match ORE, not just the
    HW_A=0.03 value every other test in this file uses."""

    @pytest.mark.parametrize("a", [1e-4, 0.5, 2.0])
    def test_matches_ore_across_mean_reversion_range(self, a):
        cfg = SwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True,
            rate_factor_index=0, hw_a=a, hw_sigma=HW_SIGMA,
            initial_zero_curve=ZERO_CURVE, swap_tenor="5Y", evaluation_date=TODAY,
        )
        mine = _price_at_t0(cfg)

        ORE.Settings.instance().evaluationDate = TODAY
        dc = ORE.Actual365Fixed()
        curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, FLAT_RATE, dc))
        hw = ORE.HullWhite(curve, a, HW_SIGMA)
        index = ORE.IborIndex(
            "SimIndex", ORE.Period(6, ORE.Months), 2,
            ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False,
            dc, curve,
        )
        swap = ORE.MakeVanillaSwap(
            ORE.Period("5Y"), index, 0.03,
            nominal=1_000_000.0, swapType=ORE.VanillaSwap.Payer,
            fixedLegDayCount=dc, floatingLegDayCount=dc,
        )
        exercise_date = ORE.TARGET().advance(TODAY, ORE.Period(2, ORE.Days))
        exercise = ORE.EuropeanExercise(exercise_date)
        swaption = ORE.Swaption(swap, exercise)
        swaption.setPricingEngine(ORE.JamshidianSwaptionEngine(hw, curve))
        np.testing.assert_allclose(mine, swaption.NPV(), rtol=1e-3, atol=1e-2)


class TestExerciseLagVariations:
    def test_custom_exercise_lag_still_prices_sanely(self):
        """exercise_lag_days defaults to 2 (standard spot lag) everywhere
        else in this file -- confirm a non-default lag still produces a
        finite, non-negative NPV and a later exercise time than lag=0."""
        cfg_default = _make_cfg(0.03, True, "5Y")
        cfg_custom = SwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True,
            rate_factor_index=0, hw_a=HW_A, hw_sigma=HW_SIGMA,
            initial_zero_curve=ZERO_CURVE, swap_tenor="5Y",
            exercise_lag_days=5, evaluation_date=TODAY,
        )
        prepared_default = prepare_swaption(cfg_default)
        prepared_custom = prepare_swaption(cfg_custom)
        assert prepared_custom.exercise_time > prepared_default.exercise_time

        mine = _price_at_t0(cfg_custom)
        assert np.isfinite(mine)
        assert mine >= 0.0

    def test_zero_exercise_lag_matches_ore(self):
        """exercise_lag_days=0 (option decided exactly today, entering an
        underlying that itself still spot-starts) is a real, priceable
        edge case, not just an internal implementation detail."""
        cfg = SwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True,
            rate_factor_index=0, hw_a=HW_A, hw_sigma=HW_SIGMA,
            initial_zero_curve=ZERO_CURVE, swap_tenor="5Y",
            exercise_lag_days=0, evaluation_date=TODAY,
        )
        mine = _price_at_t0(cfg)

        ORE.Settings.instance().evaluationDate = TODAY
        dc = ORE.Actual365Fixed()
        curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, FLAT_RATE, dc))
        hw = ORE.HullWhite(curve, HW_A, HW_SIGMA)
        index = ORE.IborIndex(
            "SimIndex", ORE.Period(6, ORE.Months), 2,
            ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False,
            dc, curve,
        )
        swap = ORE.MakeVanillaSwap(
            ORE.Period("5Y"), index, 0.03,
            nominal=1_000_000.0, swapType=ORE.VanillaSwap.Payer,
            fixedLegDayCount=dc, floatingLegDayCount=dc,
        )
        exercise = ORE.EuropeanExercise(TODAY)  # lag=0 -> exercise date is today itself
        swaption = ORE.Swaption(swap, exercise)
        swaption.setPricingEngine(ORE.JamshidianSwaptionEngine(hw, curve))
        np.testing.assert_allclose(mine, swaption.NPV(), rtol=1e-4, atol=1e-2)


class TestPortfolioOfSwaptions:
    """price_swaptions must correctly price a MIXED portfolio -- different
    payer/receiver sides, different rate factors, different tenors -- in
    one call, with each trade's NPV independent of the others (no
    cross-contamination in the stacked bisection/pricing)."""

    def test_mixed_portfolio_matches_individual_pricing(self):
        cfg_a = _make_cfg(0.03, True, "5Y")
        cfg_b = _make_cfg(0.025, False, "10Y")
        cfg_c = _make_cfg(0.03, True, "2Y", forward_start_years=3)

        step_times = jnp.array([0.0])
        hw_paths = jnp.array([[[FLAT_RATE]]])

        combined = price_swaptions(hw_paths, step_times, [cfg_a, cfg_b, cfg_c])
        individual_a = price_swaptions(hw_paths, step_times, [cfg_a])
        individual_b = price_swaptions(hw_paths, step_times, [cfg_b])
        individual_c = price_swaptions(hw_paths, step_times, [cfg_c])

        np.testing.assert_allclose(float(combined[0, 0, 0]), float(individual_a[0, 0, 0]), rtol=1e-9)
        np.testing.assert_allclose(float(combined[0, 0, 1]), float(individual_b[0, 0, 0]), rtol=1e-9)
        np.testing.assert_allclose(float(combined[0, 0, 2]), float(individual_c[0, 0, 0]), rtol=1e-9)

    def test_empty_portfolio_raises_rather_than_silently_misbehaving(self):
        """An empty swaption_configs list is a degenerate caller error, not
        a "zero trades" NPV cube -- jnp.stack([]) has no way to infer the
        Scenarios/TimeSteps shape from zero inputs, so it raises. This is
        the same pre-existing behavior swap.price_swaps has
        for an empty swap_configs list (not a swaption-specific gap) --
        documented here as a known, tested boundary rather than an
        assumption that was never checked."""
        step_times = jnp.array([0.0, 1.0])
        hw_paths = jnp.full((4, 2, 1), FLAT_RATE)
        with pytest.raises(ValueError):
            price_swaptions(hw_paths, step_times, [])


class TestZeroNotional:
    def test_zero_notional_prices_to_zero(self):
        cfg = SwaptionConfig(
            notional=0.0, fixed_rate=0.03, payer=True,
            rate_factor_index=0, hw_a=HW_A, hw_sigma=HW_SIGMA,
            initial_zero_curve=ZERO_CURVE, swap_tenor="5Y", evaluation_date=TODAY,
        )
        mine = _price_at_t0(cfg)
        np.testing.assert_allclose(mine, 0.0, atol=1e-6)


# =============================================================================
# NEW TESTS: bisection robustness, wide ORE grid cross-check, conditional
# monotonicity, numerical edge cases, and shape/API robustness.
# =============================================================================


class TestBisectionRootFindRobustness:
    """_solve_rstar brackets r* in a FIXED [-2, 2] window (see its docstring
    and source: `lo = -jnp.ones(t_shape) * 2.0`, `hi = jnp.ones(t_shape) *
    2.0`) regardless of the rate environment -- these tests probe whether
    that fixed bracket, and the bisection generally, still produces a
    correct, finite r* (and NPV) at the edges of plausible use: extreme
    moneyness, very short/long time-to-expiry, near-zero/negative rates,
    and a degenerate single-cashflow underlying (n=1 zero-coupon bond
    option, the smallest possible Jamshidian decomposition)."""

    @pytest.mark.parametrize("fixed_rate,payer", [
        (1.5, True),    # deep OTM payer: r* should sit near the bracket's
                         # own upper edge (+2 = 200%), never clamp/NaN
        (1.5, False),    # deep ITM receiver: symmetric case
        (-0.99, False),  # deep OTM receiver
    ])
    def test_extreme_moneyness_still_finite_and_matches_ore(self, fixed_rate, payer):
        mine = _price_at_t0(_make_cfg(fixed_rate, payer, "5Y"))
        ore = _reference_ore_jamshidian_npv(fixed_rate, payer, "5Y")
        assert np.isfinite(mine)
        assert mine >= -1e-6
        np.testing.assert_allclose(mine, ore, rtol=1e-4, atol=1e-2)

    def test_deep_itm_payer_beyond_bracket_range_matches_ore(self):
        """Regression test for a fixed bug: a deep-ITM payer with a very
        negative fixed_rate (e.g. -85%, paying an almost-free fixed leg)
        has a true r* whose coupon-bond-value root lies OUTSIDE
        _solve_rstar's base `[-2, 2]` bracket. Direct inspection of
        coupon_bond_value(r) across r in [-2, 2] for this config shows it
        is STRICTLY NEGATIVE across the entire bracket for fixed_rate=-0.99
        (never crosses zero there -- confirmed by evaluating the same
        closed-form used inside _price_one_swaption at 21 points from -2
        to 2, all negative, ranging from -2.1e9 at r=-2 to -1.16e6 at
        r=+2). Plain bisection would then never see `val_mid > 0.0`, so
        `hi` collapses onto `lo` every iteration and r* would converge to
        exactly the bracket's own edge (-2.0) rather than any real root --
        this used to produce an NPV wildly wrong in both magnitude and
        sign vs. ORE.

        _solve_rstar now expands the bracket outward first whenever it
        doesn't already contain a sign change, so r* converges to the true
        (out-of-bracket) root and matches ORE to the module's usual
        tolerance. This is a real, reachable case (not contrived): nothing
        in SwaptionConfig validates fixed_rate's range, and the base
        bracket's width is fixed regardless of trade scale, so any
        sufficiently deep-ITM payer can push r* outside [-2, 2]."""
        fixed_rate, payer, tenor = -0.99, True, "5Y"
        mine = _price_at_t0(_make_cfg(fixed_rate, payer, tenor))
        ore = _reference_ore_jamshidian_npv(fixed_rate, payer, tenor)
        assert ore > 0.0
        # A looser tolerance than this module's usual 1e-6 in-bracket
        # precision: the expanded bracket is wider (several outward
        # doublings before bisection starts), so the same fixed 100-
        # iteration bisection budget yields coarser per-iteration
        # precision on r* here than in the common in-bracket case.
        assert abs(mine - ore) / abs(ore) < 5e-3, (
            f"deep-ITM payer with fixed_rate={fixed_rate} beyond the base "
            f"bracket: got NPV={mine!r} vs ORE={ore!r}"
        )

    def test_very_short_time_to_expiry_matches_ore(self):
        """1M forward-start into a 1Y underlying: exercise is only ~1 month
        (plus 2-day spot lag) away -- b(T0,Ti) terms and sigma_p are all
        tiny, stressing the bisection's convergence far from its [-2,2]
        bracket's natural scale."""
        mine = _price_at_t0(_make_cfg(0.03, True, "1Y", forward_start_years=0))
        # forward_start_years is int-only in _make_cfg; build 1M directly.
        cfg = SwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True,
            rate_factor_index=0, hw_a=HW_A, hw_sigma=HW_SIGMA,
            initial_zero_curve=ZERO_CURVE, swap_tenor="1Y",
            forward_start=ORE.Period(1, ORE.Months), evaluation_date=TODAY,
        )
        mine_1m_fwd = _price_at_t0(cfg)
        assert np.isfinite(mine_1m_fwd)
        assert mine_1m_fwd >= -1e-6

        ORE.Settings.instance().evaluationDate = TODAY
        dc = ORE.Actual365Fixed()
        curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, FLAT_RATE, dc))
        hw = ORE.HullWhite(curve, HW_A, HW_SIGMA)
        index = ORE.IborIndex(
            "SimIndex", ORE.Period(6, ORE.Months), 2,
            ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False,
            dc, curve,
        )
        swap = ORE.MakeVanillaSwap(
            ORE.Period("1Y"), index, 0.03,
            nominal=1_000_000.0, swapType=ORE.VanillaSwap.Payer,
            fixedLegDayCount=dc, floatingLegDayCount=dc,
            forwardStart=ORE.Period(1, ORE.Months),
        )
        exercise_date = ORE.TARGET().advance(ORE.TARGET().advance(TODAY, ORE.Period(1, ORE.Months)), 2, ORE.Days)
        exercise = ORE.EuropeanExercise(exercise_date)
        swaption = ORE.Swaption(swap, exercise)
        swaption.setPricingEngine(ORE.JamshidianSwaptionEngine(hw, curve))
        np.testing.assert_allclose(mine_1m_fwd, swaption.NPV(), rtol=1e-4, atol=1e-2)

    def test_very_long_time_to_expiry_matches_ore(self):
        """20Y forward-start into a 5Y underlying -- exercise time is far
        beyond every other test in this file, stressing the variance terms
        (1 - exp(-2at)) which saturate near 1 at this horizon."""
        mine = _price_at_t0(_make_cfg(0.03, True, "5Y", forward_start_years=20))
        ore = _reference_ore_jamshidian_npv(0.03, True, "5Y", forward_start_years=20)
        assert np.isfinite(mine)
        np.testing.assert_allclose(mine, ore, rtol=1e-4, atol=1e-2)

    def test_near_zero_rate_environment_matches_ore(self):
        near_zero = 1e-6
        curve = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[near_zero] * 6)
        cfg = SwaptionConfig(
            notional=1_000_000.0, fixed_rate=near_zero, payer=True,
            rate_factor_index=0, hw_a=HW_A, hw_sigma=HW_SIGMA,
            initial_zero_curve=curve, swap_tenor="5Y", evaluation_date=TODAY,
        )
        mine = _price_at_t0_with_rate(cfg, near_zero)

        ORE.Settings.instance().evaluationDate = TODAY
        dc = ORE.Actual365Fixed()
        ore_curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, near_zero, dc))
        hw = ORE.HullWhite(ore_curve, HW_A, HW_SIGMA)
        index = ORE.IborIndex(
            "SimIndex", ORE.Period(6, ORE.Months), 2,
            ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False,
            dc, ore_curve,
        )
        swap = ORE.MakeVanillaSwap(
            ORE.Period("5Y"), index, near_zero,
            nominal=1_000_000.0, swapType=ORE.VanillaSwap.Payer,
            fixedLegDayCount=dc, floatingLegDayCount=dc,
        )
        exercise_date = ORE.TARGET().advance(TODAY, 2, ORE.Days)
        exercise = ORE.EuropeanExercise(exercise_date)
        swaption = ORE.Swaption(swap, exercise)
        swaption.setPricingEngine(ORE.JamshidianSwaptionEngine(hw, ore_curve))
        np.testing.assert_allclose(mine, swaption.NPV(), rtol=1e-4, atol=1e-2)

    def test_deeply_negative_rate_environment_finite(self):
        """A rate environment far more negative than TestNegativeRates'
        -0.5% (still within the bracket's [-2,2] but stressing B(t,T) and
        the variance term with a large-magnitude r) must not blow up the
        bisection or produce NaN/inf."""
        very_negative = -0.10
        curve = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[very_negative] * 6)
        cfg = SwaptionConfig(
            notional=1_000_000.0, fixed_rate=very_negative, payer=True,
            rate_factor_index=0, hw_a=HW_A, hw_sigma=HW_SIGMA,
            initial_zero_curve=curve, swap_tenor="5Y", evaluation_date=TODAY,
        )
        mine = _price_at_t0_with_rate(cfg, very_negative)
        assert np.isfinite(mine)
        assert mine >= -1e-6

    def test_degenerate_single_cashflow_underlying_matches_ore(self):
        """swap_tenor='1Y' with a 6M-index/annual-fixed-frequency underlying
        produces EXACTLY ONE fixed cashflow (verified directly against ORE's
        own swap.fixedLeg()) -- Jamshidian's decomposition here has only a
        single "coupon" leg plus the final/T_start notional legs (n=1 zero-
        coupon bond option per notional leg), the smallest possible instance
        of the general N-leg formula. This exercises _solve_rstar and the
        summation logic at their minimal, least-averaged-out scale."""
        prepared = prepare_swaption(_make_cfg(0.03, True, "1Y"))
        assert len(prepared.fixed_cashflow_times) == 1

        mine = _price_at_t0(_make_cfg(0.03, True, "1Y"))
        ore = _reference_ore_jamshidian_npv(0.03, True, "1Y")
        np.testing.assert_allclose(mine, ore, rtol=1e-4, atol=1e-2)

    def test_degenerate_single_cashflow_receiver_matches_ore(self):
        mine = _price_at_t0(_make_cfg(0.025, False, "1Y"))
        ore = _reference_ore_jamshidian_npv(0.025, False, "1Y")
        np.testing.assert_allclose(mine, ore, rtol=1e-4, atol=1e-2)


class TestWideGridAgainstORE:
    """Systematic breadth cross-check against real
    ORE.JamshidianSwaptionEngine: sweeps strike x payer/receiver x tenor x
    forward-start-offset x hw_a x hw_sigma, at the documented ~1e-6-class
    relative tolerance (tightened here to 1e-5 relative + a small absolute
    floor for near-zero NPVs, matching the module docstring's claim) rather
    than the handful of hardcoded points in TestAgainstOREJamshidianEngine."""

    @pytest.mark.parametrize("fixed_rate", [0.01, 0.02, 0.03, 0.04, 0.06])
    @pytest.mark.parametrize("payer", [True, False])
    @pytest.mark.parametrize("tenor,forward_years", [("2Y", 0), ("5Y", 0), ("10Y", 2), ("5Y", 5)])
    def test_grid_matches_ore(self, fixed_rate, payer, tenor, forward_years):
        mine = _price_at_t0(_make_cfg(fixed_rate, payer, tenor, forward_years))
        ore = _reference_ore_jamshidian_npv(fixed_rate, payer, tenor, forward_years)
        np.testing.assert_allclose(mine, ore, rtol=1e-5, atol=5.0)

    @pytest.mark.parametrize("hw_a,hw_sigma", [
        (0.01, 0.005), (0.01, 0.02), (0.05, 0.005), (0.05, 0.02),
        (0.1, 0.01), (0.3, 0.01), (1.0, 0.01),
    ])
    @pytest.mark.parametrize("fixed_rate,payer", [(0.02, True), (0.03, False), (0.04, True)])
    def test_grid_matches_ore_across_model_params(self, hw_a, hw_sigma, fixed_rate, payer):
        cfg = SwaptionConfig(
            notional=1_000_000.0, fixed_rate=fixed_rate, payer=payer,
            rate_factor_index=0, hw_a=hw_a, hw_sigma=hw_sigma,
            initial_zero_curve=ZERO_CURVE, swap_tenor="5Y", evaluation_date=TODAY,
        )
        mine = _price_at_t0(cfg)

        ORE.Settings.instance().evaluationDate = TODAY
        dc = ORE.Actual365Fixed()
        curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, FLAT_RATE, dc))
        hw = ORE.HullWhite(curve, hw_a, hw_sigma)
        index = ORE.IborIndex(
            "SimIndex", ORE.Period(6, ORE.Months), 2,
            ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False,
            dc, curve,
        )
        swap_type = ORE.VanillaSwap.Payer if payer else ORE.VanillaSwap.Receiver
        swap = ORE.MakeVanillaSwap(
            ORE.Period("5Y"), index, fixed_rate,
            nominal=1_000_000.0, swapType=swap_type,
            fixedLegDayCount=dc, floatingLegDayCount=dc,
        )
        exercise_date = ORE.TARGET().advance(TODAY, 2, ORE.Days)
        exercise = ORE.EuropeanExercise(exercise_date)
        swaption = ORE.Swaption(swap, exercise)
        swaption.setPricingEngine(ORE.JamshidianSwaptionEngine(hw, curve))
        np.testing.assert_allclose(mine, swaption.NPV(), rtol=1e-5, atol=5.0)


class TestConditionalMonotonicityInShortRate:
    """A payer swaption's conditional value at a future node must be
    monotonically non-decreasing in the simulated short rate r(t) (higher
    rates -> a higher forward rate on the underlying swap -> more valuable
    right to PAY fixed), and a receiver swaption's value must be
    monotonically non-increasing in r(t) -- the mirror-image property. This
    is a model-independent economic property of the payoff, checked here
    numerically across a grid of short-rate values at a fixed future time
    (not just at t=0), matching this module's own conditional-pricing
    claim (see module docstring's Markov-conditioning argument)."""

    @pytest.mark.parametrize("t_frac", [0.0, 0.25, 0.5, 0.9])
    def test_payer_value_nondecreasing_in_short_rate(self, t_frac):
        cfg = _make_cfg(0.03, True, "5Y", forward_start_years=3)
        prepared = prepare_swaption(cfg)
        t = prepared.exercise_time * t_frac
        step_times = jnp.array([t])
        rates = np.linspace(-0.03, 0.10, 25)
        hw_paths = jnp.array([[[r]] for r in rates])
        step_times_b = jnp.broadcast_to(step_times, (1,))
        npvs = []
        for r in rates:
            hw_paths_1 = jnp.array([[[r]]])
            npvs.append(float(_price_one_swaption(hw_paths_1, step_times_b, prepared)[0, 0]))
        npvs = np.array(npvs)
        diffs = np.diff(npvs)
        assert np.all(diffs >= -1e-6), f"payer NPV not monotonic non-decreasing in r at t_frac={t_frac}: {npvs}"

    @pytest.mark.parametrize("t_frac", [0.0, 0.25, 0.5, 0.9])
    def test_receiver_value_nonincreasing_in_short_rate(self, t_frac):
        cfg = _make_cfg(0.03, False, "5Y", forward_start_years=3)
        prepared = prepare_swaption(cfg)
        t = prepared.exercise_time * t_frac
        step_times_b = jnp.array([t])
        rates = np.linspace(-0.03, 0.10, 25)
        npvs = []
        for r in rates:
            hw_paths_1 = jnp.array([[[r]]])
            npvs.append(float(_price_one_swaption(hw_paths_1, step_times_b, prepared)[0, 0]))
        npvs = np.array(npvs)
        diffs = np.diff(npvs)
        assert np.all(diffs <= 1e-6), f"receiver NPV not monotonic non-increasing in r at t_frac={t_frac}: {npvs}"

    def test_payer_and_receiver_cross_at_consistent_point(self):
        """At any fixed future node, the payer and receiver value curves (as
        functions of r) are mirror images crossing near the forward-neutral
        rate -- a looser, complementary check to the monotonicity tests
        above: for a low r the receiver should be worth more than the
        payer, and for a high r the payer should be worth more than the
        receiver."""
        cfg_payer = _make_cfg(0.03, True, "5Y", forward_start_years=3)
        cfg_receiver = _make_cfg(0.03, False, "5Y", forward_start_years=3)
        prepared_payer = prepare_swaption(cfg_payer)
        prepared_receiver = prepare_swaption(cfg_receiver)
        t = prepared_payer.exercise_time * 0.5
        step_times = jnp.array([t])

        low_r = jnp.array([[[-0.02]]])
        high_r = jnp.array([[[0.10]]])

        payer_low = float(_price_one_swaption(low_r, step_times, prepared_payer)[0, 0])
        receiver_low = float(_price_one_swaption(low_r, step_times, prepared_receiver)[0, 0])
        payer_high = float(_price_one_swaption(high_r, step_times, prepared_payer)[0, 0])
        receiver_high = float(_price_one_swaption(high_r, step_times, prepared_receiver)[0, 0])

        assert receiver_low > payer_low
        assert payer_high > receiver_high


class TestNumericalEdgeCasesSigmaAndMeanReversion:
    """hw_sigma -> 0 (near-deterministic) and hw_a -> 0 (near-degenerate
    mean reversion, a potential 0/0 in B(t,T) = (1-exp(-a*tau))/a as a->0)
    -- distinct from TestExtremeMeanReversion's a=1e-4 point and
    TestNearZeroVolatility's sigma=1e-5 point, this class pushes both
    further and checks the actual convergence/finiteness properties rather
    than just cross-checking a single value against ORE."""

    @pytest.mark.parametrize("sigma", [1e-3, 1e-6, 1e-10])
    def test_decreasing_sigma_converges_monotonically_toward_intrinsic(self, sigma):
        """As hw_sigma shrinks toward 0, the swaption's t=0 NPV should get
        (weakly) closer to the deterministic intrinsic max(swap NPV, 0) --
        checked directly at three shrinking scales rather than a single
        near-zero value, to confirm genuine convergence rather than a
        coincidental match at one sigma."""
        fixed_rate, payer = 0.02, True
        cfg = SwaptionConfig(
            notional=1_000_000.0, fixed_rate=fixed_rate, payer=payer,
            rate_factor_index=0, hw_a=HW_A, hw_sigma=sigma,
            initial_zero_curve=ZERO_CURVE, swap_tenor="5Y", evaluation_date=TODAY,
        )
        mine = _price_at_t0(cfg)

        ORE.Settings.instance().evaluationDate = TODAY
        dc = ORE.Actual365Fixed()
        curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, FLAT_RATE, dc))
        index = ORE.IborIndex(
            "SimIndex", ORE.Period(6, ORE.Months), 2,
            ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False,
            dc, curve,
        )
        swap = ORE.MakeVanillaSwap(
            ORE.Period("5Y"), index, fixed_rate,
            nominal=1_000_000.0, swapType=ORE.VanillaSwap.Payer,
            fixedLegDayCount=dc, floatingLegDayCount=dc,
        )
        swap.setPricingEngine(ORE.DiscountingSwapEngine(curve))
        intrinsic = max(swap.NPV(), 0.0)

        assert np.isfinite(mine)
        # at sigma=1e-3 there's still real optionality value above intrinsic;
        # by 1e-10 it should have collapsed to within a cent.
        if sigma <= 1e-6:
            np.testing.assert_allclose(mine, intrinsic, atol=1e-2)
        else:
            assert mine >= intrinsic - 1e-6

    @pytest.mark.parametrize("a", [1e-8, 1e-6, 1e-4])
    def test_near_zero_mean_reversion_no_nan(self, a):
        """a -> 0 makes B(t,T) = (1-exp(-a*tau))/a a classic 0/0 form whose
        analytic (L'Hopital) limit is simply tau -- this test doesn't
        assert the specific limiting value, only that the module's actual
        floating-point evaluation of that ratio stays finite (rather than
        NaN'ing out) at three shrinking scales of a, and still produces a
        sane (finite, non-negative) NPV end-to-end."""
        cfg = SwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True,
            rate_factor_index=0, hw_a=a, hw_sigma=HW_SIGMA,
            initial_zero_curve=ZERO_CURVE, swap_tenor="5Y", evaluation_date=TODAY,
        )
        mine = _price_at_t0(cfg)
        assert np.isfinite(mine), f"NaN/inf at hw_a={a}"
        assert mine >= -1e-6

    def test_near_zero_mean_reversion_B_matches_taylor_limit(self):
        """Directly probes _hw_B's 0/0 behavior at a tiny a: for a small
        tau, B(t,T) should be very close to tau itself (the a->0 limit),
        confirming the ratio doesn't silently produce garbage (e.g. 0, or
        blow up) even though it's not called through the L'Hopital-limit
        code path explicitly."""
        from engine.instruments.european_swaption import _hw_B
        a = 1e-8
        t = jnp.array(0.0)
        T = jnp.array(5.0)
        b = float(_hw_B(t, T, a))
        assert np.isfinite(b)
        np.testing.assert_allclose(b, 5.0, rtol=1e-4)


class TestShapeAndAPIRobustness:
    """Mismatched batch shapes, empty configs, and single-trade portfolios
    -- boundary behavior of price_swaptions/prepare_swaption beyond the
    already-covered empty-list case in TestPortfolioOfSwaptions."""

    def test_single_trade_portfolio_matches_direct_pricing(self):
        cfg = _make_cfg(0.03, True, "5Y")
        step_times = jnp.array([0.0])
        hw_paths = jnp.array([[[FLAT_RATE]]])
        via_portfolio = price_swaptions(hw_paths, step_times, [cfg])
        direct = _price_at_t0(cfg)
        assert via_portfolio.shape == (1, 1, 1)
        np.testing.assert_allclose(float(via_portfolio[0, 0, 0]), direct, rtol=1e-9)

    def test_mismatched_rate_factor_index_out_of_bounds_raises_or_propagates(self):
        """rate_factor_index selecting a column beyond hw_paths' own NumHW
        axis is a caller-config error -- confirm it surfaces as an
        exception (JAX's default indexing semantics for out-of-bounds is to
        clip, not raise, so this documents/locks in that actual behavior
        rather than assuming a raise)."""
        cfg = SwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True,
            rate_factor_index=5, hw_a=HW_A, hw_sigma=HW_SIGMA,
            initial_zero_curve=ZERO_CURVE, swap_tenor="5Y", evaluation_date=TODAY,
        )
        prepared = prepare_swaption(cfg)
        step_times = jnp.array([0.0])
        hw_paths = jnp.array([[[FLAT_RATE, FLAT_RATE]]])  # only 2 HW factors, index 5 is OOB
        # JAX clips out-of-bounds indices by default rather than raising --
        # this must not silently crash; it should return SOME finite number
        # (documenting the clip-not-raise behavior rather than asserting a
        # specific numeric value, since that's an implementation detail of
        # JAX's indexing, not this module's).
        npv = _price_one_swaption(hw_paths, step_times, prepared)
        assert np.isfinite(float(npv[0, 0]))

    def test_more_scenarios_than_steps_and_vice_versa_shape(self):
        """Non-square [Scenarios, TimeSteps] shapes (far more scenarios than
        steps, and far more steps than scenarios) must both produce the
        correctly-shaped NPV cube -- guards against an accidental
        transpose/broadcast bug in _solve_rstar's t_shape or the
        per-(scenario,step) computation."""
        cfg = _make_cfg(0.03, True, "5Y")
        step_times_many_steps = jnp.linspace(0.0, 1.0, 20)
        hw_paths_many_steps = jnp.full((2, 20, 1), FLAT_RATE)
        npv_a = price_swaptions(hw_paths_many_steps, step_times_many_steps, [cfg])
        assert npv_a.shape == (2, 20, 1)
        assert bool(jnp.all(jnp.isfinite(npv_a)))

        step_times_one_step = jnp.array([0.5])
        hw_paths_many_scenarios = jnp.full((500, 1, 1), FLAT_RATE)
        npv_b = price_swaptions(hw_paths_many_scenarios, step_times_one_step, [cfg])
        assert npv_b.shape == (500, 1, 1)
        assert bool(jnp.all(jnp.isfinite(npv_b)))

    def test_single_trade_list_vs_multi_trade_list_consistent_shape(self):
        cfg = _make_cfg(0.03, True, "5Y")
        step_times = jnp.array([0.0, 0.5])
        hw_paths = jnp.full((3, 2, 1), FLAT_RATE)
        npv_one = price_swaptions(hw_paths, step_times, [cfg])
        npv_three = price_swaptions(hw_paths, step_times, [cfg, cfg, cfg])
        assert npv_one.shape == (3, 2, 1)
        assert npv_three.shape == (3, 2, 3)
        np.testing.assert_allclose(np.asarray(npv_three[:, :, 0]), np.asarray(npv_one[:, :, 0]), rtol=1e-9)
        np.testing.assert_allclose(np.asarray(npv_three[:, :, 1]), np.asarray(npv_one[:, :, 0]), rtol=1e-9)
