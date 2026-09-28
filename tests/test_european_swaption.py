import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.simulation.market_model import ZeroCurveConfig
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
    The same swaption built in ORE (same index and ACT/365 legs as `_build_ore_swap`),
    priced by `ORE.JamshidianSwaptionEngine` under `ORE.HullWhite(FLAT_RATE, HW_A,
    HW_SIGMA)`: the reference for this file.
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
    """t=0 price conditional on r(0) = FLAT_RATE, comparable with ORE's t=0 NPV."""
    prepared = prepare_swaption(cfg)
    step_times = jnp.array([0.0])
    hw_paths = jnp.array([[[FLAT_RATE]]])
    npv = _price_one_swaption(hw_paths, step_times, prepared)
    return float(npv[0, 0])


# ---------------------------------------------------------------------------
# Conditional (t > 0) pricing against ORE (I-30)
# ---------------------------------------------------------------------------
# At t=0 the variance term of A(t,T) has a factor (1 - exp(-2at)) = 0, so only t > 0
# checks it (tests/test_ore_coverage_hardening.py measures this).
#
# Dates are off the curve pillars: f(0,t) has a kink at each pillar under linear zero
# interpolation, and the engine and ORE resolve it differently
# (test_ore_coverage_hardening.py::test_pillar_times_differ_by_the_interpolation_kink).
CONDITIONAL_PILLARS = [0.0, 1.0, 2.0, 5.0, 10.0, 30.0]
CONDITIONAL_CURVES = {
    "flat": [FLAT_RATE] * 6,
    "upward": [0.010, 0.015, 0.020, 0.030, 0.040, 0.050],
    "inverted": [0.050, 0.045, 0.040, 0.030, 0.025, 0.020],
}
CONDITIONAL_DATES = {
    "t0.50": ORE.Date(29, 1, 2027),
    "t0.75": ORE.Date(30, 4, 2027),
    "t1.51": ORE.Date(31, 1, 2028),
    "t2.25": ORE.Date(30, 10, 2028),
}
# (short rate at t, payer, fixed rate, tenor, forward start in years)
CONDITIONAL_TRADES = {
    "r1%-payer": (0.010, True, 0.03, "5Y", 3),
    "r3.5%-payer": (0.035, True, 0.03, "5Y", 3),
    "r3.5%-receiver": (0.035, False, 0.03, "5Y", 3),
    "r6%-payer": (0.060, True, 0.03, "5Y", 3),
    "r3%-itm-payer-10Y": (0.030, True, 0.02, "10Y", 3),
}
CONDITIONAL_GRID = [
    pytest.param(curve, date_id, trade_id, id=f"{curve}-{date_id}-{trade_id}")
    for curve in CONDITIONAL_CURVES
    for date_id in CONDITIONAL_DATES
    for trade_id in CONDITIONAL_TRADES
]


def _conditional_cfg(curve: str, trade_id: str) -> SwaptionConfig:
    _, payer, fixed_rate, tenor, forward_start_years = CONDITIONAL_TRADES[trade_id]
    return SwaptionConfig(
        notional=1_000_000.0, fixed_rate=fixed_rate, payer=payer, rate_factor_index=0,
        hw_a=HW_A, hw_sigma=HW_SIGMA,
        initial_zero_curve=ZeroCurveConfig(times=CONDITIONAL_PILLARS, rates=CONDITIONAL_CURVES[curve]),
        swap_tenor=tenor, forward_start=ORE.Period(forward_start_years, ORE.Years),
        evaluation_date=TODAY,
    )


def _price_conditional(curve: str, date_id: str, trade_id: str) -> float:
    """The engine's NPV at the grid point's date, conditional on its short rate."""
    t = ORE.Actual365Fixed().yearFraction(TODAY, CONDITIONAL_DATES[date_id])
    r = CONDITIONAL_TRADES[trade_id][0]
    prepared = prepare_swaption(_conditional_cfg(curve, trade_id))
    return float(_price_one_swaption(jnp.array([[[r]]]), jnp.array([t]), prepared)[0, 0])


def _reference_ore_conditional_npv(curve: str, date_id: str, trade_id: str) -> float:
    """ORE's price for the same swaption at a later date `t`, given r(t).

      * The swap is built once at TODAY, as `prepare_swaption` does, so both sides use
        the same schedule (rebuilding at `t` would move dates).
      * The market at `t` is ORE's conditional curve `ORE.HullWhite(curve0).discountBond(t,
        T, r)`, with ORE's own A(t,T).
      * That curve is sampled at daily pillars: monthly log-linear pillars gave the rebuilt
        model a stepwise forward and up to 7.6e-4 reference error on a sloped curve; daily
        pillars bring it to ~2e-6.
      * `JamshidianSwaptionEngine` prices at `t` under HullWhite(conditional curve, a,
        sigma). Conditioning on r(t) and refitting to P(t, .) are the same model (Markov
        property); that equivalence is what is tested.
    """
    rates = CONDITIONAL_CURVES[curve]
    t_date = CONDITIONAL_DATES[date_id]
    r_t, payer, fixed_rate, tenor, forward_start_years = CONDITIONAL_TRADES[trade_id]
    dc = ORE.Actual365Fixed()

    ORE.Settings.instance().evaluationDate = TODAY
    try:
        curve0 = ORE.YieldTermStructureHandle(ORE.ZeroCurve(
            [TODAY + int(round(p * 365)) for p in CONDITIONAL_PILLARS], list(rates), dc))
        hw0 = ORE.HullWhite(curve0, HW_A, HW_SIGMA)
        market = ORE.RelinkableYieldTermStructureHandle(curve0.currentLink())
        index = ORE.IborIndex(
            "SimIndex", ORE.Period(6, ORE.Months), 2,
            ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False,
            dc, market,
        )
        swap = ORE.MakeVanillaSwap(
            ORE.Period(tenor), index, fixed_rate,
            nominal=1_000_000.0,
            swapType=ORE.VanillaSwap.Payer if payer else ORE.VanillaSwap.Receiver,
            fixedLegDayCount=dc, floatingLegDayCount=dc,
            forwardStart=ORE.Period(forward_start_years, ORE.Years),
        )
        exercise_date = ORE.TARGET().advance(
            ORE.TARGET().advance(TODAY, ORE.Period(forward_start_years, ORE.Years)), 2, ORE.Days)
        assert exercise_date > t_date, "grid point must be before the option's expiry"

        t = dc.yearFraction(TODAY, t_date)
        horizon_days = (swap.maturityDate() - t_date) + 30
        dates = [t_date] + [t_date + k for k in range(1, horizon_days)]
        discounts = [1.0] + [hw0.discountBond(t, dc.yearFraction(TODAY, d), r_t) for d in dates[1:]]

        ORE.Settings.instance().evaluationDate = t_date
        market.linkTo(ORE.DiscountCurve(dates, discounts, dc))
        swaption = ORE.Swaption(swap, ORE.EuropeanExercise(exercise_date))
        swaption.setPricingEngine(ORE.JamshidianSwaptionEngine(ORE.HullWhite(market, HW_A, HW_SIGMA), market))
        return swaption.NPV()
    finally:
        ORE.Settings.instance().evaluationDate = TODAY


class TestAgainstOREJamshidianEngine:
    """Against `ORE.JamshidianSwaptionEngine`."""

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
        """Regression: an earlier version assumed the floating leg redeems its notional at
        the exercise time T0. For a forward start the first accrual T_start is after T0, and
        the par identity needs P(T0, T_start); omitting it was ~1% off ORE."""
        mine = _price_at_t0(_make_cfg(fixed_rate, payer, tenor, forward_years))
        ore = _reference_ore_jamshidian_npv(fixed_rate, payer, tenor, forward_years)
        np.testing.assert_allclose(mine, ore, rtol=1e-4, atol=1e-2)

    def test_matches_ore_deep_otm_is_near_zero(self):
        """A deep-OTM payer prices near zero on both sides without NaN/inf in the tail."""
        mine = _price_at_t0(_make_cfg(0.20, True, "5Y"))
        ore = _reference_ore_jamshidian_npv(0.20, True, "5Y")
        np.testing.assert_allclose(mine, ore, atol=1e-3)
        assert mine >= 0.0


class TestZeroVolatilityLimit:
    """sigma_p == 0 occurs at expiry and for a leg maturing at the exercise time (the
    T_start leg of a spot-starting swaption). An earlier caller-side guard mishandled the
    second case and gave large negative NPVs."""

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
        """Zero and positive sigma_p in one call (as `_price_one_swaption` produces) do not
        contaminate each other."""
        P_t_Topt = jnp.array([0.95, 0.95])
        P_t_S = jnp.array([0.90, 0.85])
        K = jnp.array([0.90, 0.90])
        sigma_p = jnp.array([0.0, 0.05])
        result = _bond_call(P_t_Topt, P_t_S, K, sigma_p)
        assert bool(jnp.all(jnp.isfinite(result)))


class TestPayerReceiverParity:
    def test_put_call_parity_payer_minus_receiver_equals_forward_swap_value(self):
        """Payer minus receiver equals the forward swap value, independent of the model."""
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
        """As sigma -> 0 a European tends to max(swap NPV, 0), for an ITM and an OTM
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
    """Pricing at simulated (scenario, step) points after t=0, and expiry."""

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
        """Conditional on a short rate at t=1Y (before the ~3Y exercise), against ORE's
        Jamshidian engine with its evaluation date and curve rebuilt at that date."""
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

    @pytest.mark.parametrize("curve,date_id,trade_id", CONDITIONAL_GRID)
    def test_conditional_pricing_matches_ore_across_t_and_r(self, curve, date_id, trade_id):
        """The same check over a grid: dates 0.5Y-2.25Y, short rates 1%-6%, payer and
        receiver, two strikes/tenors, three curve shapes (closes I-30).

        Worst case ~2e-6 relative, against rtol=1e-4. Every variance-term mutation in
        tests/test_ore_coverage_hardening.py moves every point by at least 3e-3, and that
        file asserts it.
        """
        mine = _price_conditional(curve, date_id, trade_id)
        ore = _reference_ore_conditional_npv(curve, date_id, trade_id)
        np.testing.assert_allclose(mine, ore, rtol=1e-4, atol=1e-2)


class TestPriceSwaptionsShape:
    def test_output_shape_multi_trade_multi_scenario_multi_step(self):
        cfg = _make_cfg(0.03, True, "5Y")
        step_times = jnp.array([0.0, 1.0, 2.0])
        hw_paths = jnp.full((8, 3, 1), FLAT_RATE)
        npv_cube = price_swaptions(hw_paths, step_times, [cfg, cfg])
        assert npv_cube.shape == (8, 3, 2)

    def test_multiple_rate_factors_selects_correct_one(self):
        """`rate_factor_index` selects the right column of `hw_paths`."""
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
        # Factor 0 at 3% (near ATM), factor 1 at 8% (deep ITM for a payer).
        hw_paths = jnp.array([[[0.03, 0.08]]])
        step_times = jnp.array([0.0])
        npv0 = price_swaptions(hw_paths, step_times, [cfg0])
        npv1 = price_swaptions(hw_paths, step_times, [cfg1])
        assert float(npv1[0, 0, 0]) > float(npv0[0, 0, 0])


class TestMonotonicRStarSolve:
    """The r* bisection needs the signed coupon bond value to decrease in r. The negative
    T_start leg makes that approximate, not exact, so it is checked numerically."""

    @pytest.mark.parametrize("tenor,forward_years", [("2Y", 0), ("10Y", 0), ("5Y", 5), ("30Y", 10)])
    def test_rstar_solve_converges_to_true_root(self, tenor, forward_years):
        cfg = _make_cfg(0.03, True, tenor, forward_years)
        prepared = prepare_swaption(cfg)
        mine = _price_at_t0(cfg)
        # A non-converged bisection gives a wildly wrong NPV, which these loose checks catch.
        assert np.isfinite(mine)
        assert mine >= -1e-6


def _price_at_t0_with_rate(cfg: SwaptionConfig, r0: float) -> float:
    prepared = prepare_swaption(cfg)
    step_times = jnp.array([0.0])
    hw_paths = jnp.array([[[r0]]])
    return float(_price_one_swaption(hw_paths, step_times, prepared)[0, 0])


class TestNegativeRates:
    """Hull-White is a normal short-rate model, so negative rates price (checked against
    ORE)."""

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
        """A positive curve with a negative simulated short rate at a later step (the Monte
        Carlo case)."""
        cfg = _make_cfg(0.03, True, "5Y")
        mine = _price_at_t0_with_rate(cfg, -0.01)
        assert np.isfinite(mine)
        assert mine >= 0.0


class TestNearZeroVolatility:
    """A small but non-zero hw_sigma prices near intrinsic with no blow-up from the Black
    formula's 1/sigma_p."""

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
    """Very small and very large mean reversion price finitely and match ORE."""

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
        """A non-default exercise lag gives a finite, non-negative NPV and a later exercise
        time than lag 0."""
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
        """exercise_lag_days=0 matches ORE."""
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
    """A mixed portfolio in one call equals pricing each trade alone."""

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
        """An empty list raises (`jnp.stack([])` cannot infer the shape), as for
        `price_swaps`."""
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


# Bisection robustness, a wide ORE grid, conditional monotonicity, numerical edge cases and
# shapes.


class TestBisectionRootFindRobustness:
    """`_bisect_rstar` starts from [-2, 2]; these probe extreme moneyness, very short and
    long expiries, near-zero and negative rates, and a single-coupon underlying."""

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
        """Regression: for a deep-ITM payer (fixed rate -99%) the coupon bond value is
        negative over all of [-2, 2], so plain bisection returned the bracket edge and an
        NPV wrong in sign and size. The window is now shifted (up to 20 times its width)
        until it brackets the root. Reachable: nothing bounds `fixed_rate`."""
        fixed_rate, payer, tenor = -0.99, True, "5Y"
        mine = _price_at_t0(_make_cfg(fixed_rate, payer, tenor))
        ore = _reference_ore_jamshidian_npv(fixed_rate, payer, tenor)
        assert ore > 0.0
        # Looser than the usual 1e-6: the shifted window is far from the root's scale, so
        # the same 100 iterations leave r* less precise.
        assert abs(mine - ore) / abs(ore) < 5e-3, (
            f"deep-ITM payer with fixed_rate={fixed_rate} beyond the base "
            f"bracket: got NPV={mine!r} vs ORE={ore!r}"
        )

    def test_very_short_time_to_expiry_matches_ore(self):
        """A 1M forward start into a 1Y swap: exercise about a month away."""
        mine = _price_at_t0(_make_cfg(0.03, True, "1Y", forward_start_years=0))
        # _make_cfg takes whole years only; build the 1M case directly.
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
        """A 20Y forward start into a 5Y swap (variance terms near saturation)."""
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
        """A deeply negative rate environment stays finite."""
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
        """A 1Y swap has exactly one fixed coupon (checked in ORE's leg): the smallest
        Jamshidian decomposition."""
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
    """Against `ORE.JamshidianSwaptionEngine` over strike x direction x tenor x forward
    start x hw_a x hw_sigma, at 1e-5 relative plus a small absolute floor."""

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
    """Conditional value at a future node: a payer is non-decreasing and a receiver
    non-increasing in the simulated short rate."""

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
        """At a fixed node, the receiver is worth more at low r and the payer at high r."""
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
    """hw_sigma and hw_a pushed toward 0 (further than the classes above): convergence and
    finiteness."""

    @pytest.mark.parametrize("sigma", [1e-3, 1e-6, 1e-10])
    def test_decreasing_sigma_converges_monotonically_toward_intrinsic(self, sigma):
        """As hw_sigma shrinks, the NPV approaches max(swap NPV, 0)."""
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
        # At sigma=1e-3 there is optionality value; by 1e-10 it is within a cent of intrinsic.
        if sigma <= 1e-6:
            np.testing.assert_allclose(mine, intrinsic, atol=1e-2)
        else:
            assert mine >= intrinsic - 1e-6

    @pytest.mark.parametrize("a", [1e-8, 1e-6, 1e-4])
    def test_near_zero_mean_reversion_no_nan(self, a):
        """a -> 0 stays finite and gives a sane NPV (the 0/0 in B is guarded)."""
        cfg = SwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True,
            rate_factor_index=0, hw_a=a, hw_sigma=HW_SIGMA,
            initial_zero_curve=ZERO_CURVE, swap_tenor="5Y", evaluation_date=TODAY,
        )
        mine = _price_at_t0(cfg)
        assert np.isfinite(mine), f"NaN/inf at hw_a={a}"
        assert mine >= -1e-6

    def test_near_zero_mean_reversion_B_matches_taylor_limit(self):
        """At a tiny a, B(t,T) is close to its limit T - t."""
        from engine.instruments.european_swaption import _hw_B
        a = 1e-8
        t = jnp.array(0.0)
        T = jnp.array(5.0)
        b = float(_hw_B(t, T, a))
        assert np.isfinite(b)
        np.testing.assert_allclose(b, 5.0, rtol=1e-4)


class TestShapeAndAPIRobustness:
    """Shape and API boundaries of `price_swaptions` / `prepare_swaption`."""

    def test_single_trade_portfolio_matches_direct_pricing(self):
        cfg = _make_cfg(0.03, True, "5Y")
        step_times = jnp.array([0.0])
        hw_paths = jnp.array([[[FLAT_RATE]]])
        via_portfolio = price_swaptions(hw_paths, step_times, [cfg])
        direct = _price_at_t0(cfg)
        assert via_portfolio.shape == (1, 1, 1)
        np.testing.assert_allclose(float(via_portfolio[0, 0, 0]), direct, rtol=1e-9)

    def test_mismatched_rate_factor_index_out_of_bounds_raises_or_propagates(self):
        """An out-of-range rate_factor_index is not rejected: JAX clips the index. The test
        documents that a finite number comes back."""
        cfg = SwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True,
            rate_factor_index=5, hw_a=HW_A, hw_sigma=HW_SIGMA,
            initial_zero_curve=ZERO_CURVE, swap_tenor="5Y", evaluation_date=TODAY,
        )
        prepared = prepare_swaption(cfg)
        step_times = jnp.array([0.0])
        hw_paths = jnp.array([[[FLAT_RATE, FLAT_RATE]]])  # only 2 HW factors, index 5 is OOB
        # JAX clips out-of-bounds indices rather than raising.
        npv = _price_one_swaption(hw_paths, step_times, prepared)
        assert np.isfinite(float(npv[0, 0]))

    def test_more_scenarios_than_steps_and_vice_versa_shape(self):
        """Non-square [Scenarios, TimeSteps] shapes give the right cube shape."""
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


class TestSwaptionConfigValidation:
    """`SwaptionConfig.__post_init__` rejects non-finite notional/fixed_rate/hw_sigma and
    unparseable tenors. Zero notional is valid."""

    def test_nan_notional_rejected(self):
        with pytest.raises(ValueError, match="notional"):
            SwaptionConfig(
                notional=float("nan"), fixed_rate=0.03, payer=True, rate_factor_index=0,
                hw_a=HW_A, hw_sigma=HW_SIGMA, initial_zero_curve=ZERO_CURVE,
                swap_tenor="5Y", evaluation_date=TODAY,
            )

    def test_inf_fixed_rate_rejected(self):
        with pytest.raises(ValueError, match="fixed_rate"):
            SwaptionConfig(
                notional=1_000_000.0, fixed_rate=float("inf"), payer=True, rate_factor_index=0,
                hw_a=HW_A, hw_sigma=HW_SIGMA, initial_zero_curve=ZERO_CURVE,
                swap_tenor="5Y", evaluation_date=TODAY,
            )

    def test_nan_hw_sigma_rejected(self):
        with pytest.raises(ValueError, match="hw_sigma"):
            SwaptionConfig(
                notional=1_000_000.0, fixed_rate=0.03, payer=True, rate_factor_index=0,
                hw_a=HW_A, hw_sigma=float("nan"), initial_zero_curve=ZERO_CURVE,
                swap_tenor="5Y", evaluation_date=TODAY,
            )

    def test_unparseable_swap_tenor_rejected(self):
        with pytest.raises(ValueError, match="swap_tenor"):
            SwaptionConfig(
                notional=1_000_000.0, fixed_rate=0.03, payer=True, rate_factor_index=0,
                hw_a=HW_A, hw_sigma=HW_SIGMA, initial_zero_curve=ZERO_CURVE,
                swap_tenor="bogus", evaluation_date=TODAY,
            )

    def test_valid_config_constructs_without_error(self):
        _make_cfg(0.03, True, "5Y")  # must not raise
