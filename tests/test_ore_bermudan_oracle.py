"""
Bermudan/American backward induction (`engine.pricing.lgm_grid`) against
ORE's constructible multi-exercise engines, `ORE.TreeSwaptionEngine` (Hull-White trinomial
tree) and `ORE.FdHullWhiteSwaptionEngine` (Hull-White finite differences).

These are Hull-White engines, while the engine prices under LGM, a different model for the
same (a, sigma) (see `engine.models.lgm`). So the agreement here is a model-level few
percent, not numerical parity; `TestHullWhiteVersusLgmBondPrices` measures the model
difference at the bond level. Three controls attribute the gap to the model, not the
induction:

  1. ORE's tree and FD engines agree with each other to ~1e-3, so the target is sound.
  2. The engine is grid-converged: n_per_std 48 -> 384 moves the price ~1e-5 relative.
  3. The gap is no larger with four exercise dates than with one, where the engine matches
     a direct integration (tests/test_bermudan_swaption.py).

Parity with ORE's own LGM engine, at 1e-10, is tests/test_ore_lgm_parity.py.

Construction notes:
  * The ORE-side underlying gets its own index on a real forwarding curve.
    `build_vanilla_swap`'s index has no forwarding curve (the engine never reads ORE's
    forecasts), so its swap cannot be priced by an ORE engine.
  * Exercise dates are the underlying's fixed-leg accrual starts (`exercisable_dates`),
    given to both sides as the same `ORE.Date`s. (Year-fraction exercise, now removed, once
    silently dropped a coupon when rounded: I-29.)
"""
import numpy as np
import ORE
import pytest

from engine.instruments.bermudan_swaption import BermudanSwaptionConfig, exercisable_dates
from engine.market_data.day_counts import TIME_AXIS_DAY_COUNTER
from engine.market_data.market import ZeroCurveConfig
from tests.support.lgm_engine import grid_npv, prepared

EVAL_DATE = ORE.Date(30, 7, 2026)
DC = TIME_AXIS_DAY_COUNTER
NOTIONAL = 1_000_000.0

# Grid resolution for every comparison: past convergence (test_engine_is_grid_converged).
N_PER_STD = 128
STD_DEVS = 8.0

# ORE engine resolutions, also past convergence (test_ore_tree_and_fd_engines_agree).
ORE_TREE_STEPS = 800
ORE_FD_GRID = 800


def _flat_curve(rate: float) -> ZeroCurveConfig:
    return ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[rate] * 6)


def _ore_underlying(flat_rate, fixed_rate, payer, tenor, notional=NOTIONAL):
    """The ORE-side underlying, with its own index on a real forwarding curve at
    `flat_rate` (not `build_vanilla_swap`; see the module docstring). Index tenor, spot
    lag, calendar, roll convention and day counts match that builder."""
    ORE.Settings.instance().evaluationDate = EVAL_DATE
    curve = ORE.YieldTermStructureHandle(ORE.FlatForward(EVAL_DATE, flat_rate, DC))
    index = ORE.IborIndex(
        "SimIndex", ORE.Period(6, ORE.Months), 2,
        ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False, DC, curve,
    )
    swap = ORE.MakeVanillaSwap(
        ORE.Period(tenor), index, fixed_rate, nominal=notional,
        swapType=ORE.VanillaSwap.Payer if payer else ORE.VanillaSwap.Receiver,
        fixedLegDayCount=DC, floatingLegDayCount=DC,
    )
    return curve, swap


def _ore_bermudan_npv(flat_rate, hw_a, hw_sigma, fixed_rate, payer, tenor,
                      exercise_indices, engine, notional=NOTIONAL):
    """Bermudan NPV by one of ORE's multi-exercise engines. `exercise_indices` pick the
    underlying's fixed-leg accrual starts, the same dates the engine side gets."""
    curve, swap = _ore_underlying(flat_rate, fixed_rate, payer, tenor, notional)
    hw = ORE.HullWhite(curve, hw_a, hw_sigma)
    starts = [ORE.as_fixed_rate_coupon(cf).accrualStartDate() for cf in swap.fixedLeg()]
    swaption = ORE.Swaption(swap, ORE.BermudanExercise([starts[i] for i in exercise_indices]))
    if engine == "tree":
        swaption.setPricingEngine(ORE.TreeSwaptionEngine(hw, ORE_TREE_STEPS))
    elif engine == "fd":
        swaption.setPricingEngine(ORE.FdHullWhiteSwaptionEngine(hw, ORE_FD_GRID, ORE_FD_GRID))
    else:  # pragma: no cover - guards a typo'd parametrization
        raise ValueError(f"unknown ORE engine {engine!r}")
    return swaption.NPV()


def _engine_cfg(flat_rate, hw_a, hw_sigma, fixed_rate, payer, tenor,
                exercise_dates, n_per_std=N_PER_STD, std_devs=STD_DEVS,
                notional=NOTIONAL):
    """(trade, model): the engine's trade and the LGM it is priced with."""
    trade = BermudanSwaptionConfig(
        notional=notional, fixed_rate=fixed_rate, payer=payer, exercise_dates=exercise_dates, swap_tenor=tenor,
        evaluation_date=EVAL_DATE, trade_id="bermudan",
    )
    return trade, dict(a=hw_a, sigma=hw_sigma, curve=_flat_curve(flat_rate), n_per_std=n_per_std, std_devs=std_devs)


def _npv(case) -> float:
    trade, model = case
    return grid_npv(trade, **model)


def _prepared(case):
    trade, model = case
    return prepared(trade, **model)


def _engine_exercise_dates(flat_rate, hw_a, hw_sigma, fixed_rate, payer, tenor,
                           exercise_indices, notional=NOTIONAL):
    """The underlying's fixed-leg accrual starts at `exercise_indices`, from the engine's
    schedule."""
    probe = _engine_cfg(flat_rate, hw_a, hw_sigma, fixed_rate, payer, tenor,
                        exercise_dates=[EVAL_DATE + 1], notional=notional)
    starts = exercisable_dates(probe[0])
    return [starts[i] for i in exercise_indices]


def _both_sides(flat_rate, hw_a, hw_sigma, fixed_rate, payer, tenor, exercise_indices):
    """(engine NPV, ORE tree NPV, ORE FD NPV) for one trade."""
    dates = _engine_exercise_dates(flat_rate, hw_a, hw_sigma, fixed_rate, payer,
                                   tenor, exercise_indices)
    cfg = _engine_cfg(flat_rate, hw_a, hw_sigma, fixed_rate, payer, tenor, dates)
    mine = _npv(cfg)
    tree = _ore_bermudan_npv(flat_rate, hw_a, hw_sigma, fixed_rate, payer, tenor,
                             exercise_indices, "tree")
    fd = _ore_bermudan_npv(flat_rate, hw_a, hw_sigma, fixed_rate, payer, tenor,
                           exercise_indices, "fd")
    return mine, tree, fd


# The grid: moneyness against a 3% curve, volatility, direction, mean reversion, curve
# level, and the number and spacing of exercise dates.
CASES = [
    # (id, flat, a, sigma, fixed_rate, payer, tenor, exercise_indices)
    ("atm-payer-lowvol",    0.03, 0.03, 0.005, 0.03, True,  "5Y", [1, 2, 3]),
    ("atm-payer",           0.03, 0.03, 0.01,  0.03, True,  "5Y", [1, 2, 3]),
    ("atm-payer-highvol",   0.03, 0.03, 0.02,  0.03, True,  "5Y", [1, 2, 3]),
    ("itm-payer",           0.03, 0.03, 0.01,  0.02, True,  "5Y", [1, 2, 3]),
    ("otm-payer",           0.03, 0.03, 0.01,  0.04, True,  "5Y", [1, 2, 3]),
    ("atm-receiver",        0.03, 0.03, 0.01,  0.03, False, "5Y", [1, 2, 3]),
    ("itm-receiver",        0.03, 0.03, 0.01,  0.04, False, "5Y", [1, 2, 3]),
    ("otm-receiver",        0.03, 0.03, 0.01,  0.02, False, "5Y", [1, 2, 3]),
    ("high-mean-reversion", 0.03, 0.10, 0.01,  0.03, True,  "5Y", [1, 2, 3]),
    ("low-curve",           0.01, 0.03, 0.01,  0.03, True,  "5Y", [1, 2, 3]),
    ("two-exercises",       0.03, 0.03, 0.01,  0.03, True,  "5Y", [1, 3]),
    ("four-exercises",      0.03, 0.03, 0.01,  0.03, True,  "5Y", [1, 2, 3, 4]),
]
CASE_IDS = [c[0] for c in CASES]

# Measured worst case over CASES: 1.51e-1 (deep-OTM receiver worth ~830 on 1e6, where a
# small absolute difference is a large relative one); ATM cases ~3e-2. Set just above the
# worst case as a regression bound on a known model difference.
MAX_MODEL_RELATIVE_GAP = 0.16

# ORE's two engines are independent schemes for the same model and agree much more tightly
# (measured worst case 1.34e-3).
MAX_ORE_INTERNAL_GAP = 2e-3


class TestOracleIsSound:
    """ORE's multi-exercise engines are a sound target."""

    @pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
    def test_ore_tree_and_fd_engines_agree(self, case):
        """ORE's tree and FD engines agree on every case to ~1e-3, so "the ORE value" is
        well defined."""
        _id, flat, a, sigma, rate, payer, tenor, ex = case
        tree = _ore_bermudan_npv(flat, a, sigma, rate, payer, tenor, ex, "tree")
        fd = _ore_bermudan_npv(flat, a, sigma, rate, payer, tenor, ex, "fd")
        rel = abs(tree - fd) / max(abs(fd), 1.0)
        assert rel < MAX_ORE_INTERNAL_GAP, (
            f"ORE's own tree and FD engines disagree by {rel:.2e} on {_id}; "
            f"tree={tree:.4f} fd={fd:.4f}. The oracle is not converged, so "
            f"comparisons against it are meaningless until this is resolved."
        )

    def test_oracle_underlying_swap_is_not_the_dummy_curve_swap(self):
        """The oracle's underlying has a live floating leg (non-zero first coupon, fair
        rate near the curve), so it is never replaced by the shared builder's swap."""
        curve, swap = _ore_underlying(0.03, 0.03, True, "5Y")
        first_float = ORE.as_floating_rate_coupon(swap.floatingLeg()[0])
        assert first_float.amount() > 0.0, (
            "oracle's floating leg is identically zero -- the underlying was "
            "built on a dummy 0% forward curve (see module docstring)"
        )
        swap.setPricingEngine(ORE.DiscountingSwapEngine(curve))
        assert swap.fairRate() == pytest.approx(0.03, abs=1e-3)


class TestEngineMatchesOreBermudanEngines:
    """The induction agrees with ORE's multi-exercise engines within the Hull-White/LGM
    model difference."""

    @pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
    def test_matches_ore_within_model_difference(self, case):
        _id, flat, a, sigma, rate, payer, tenor, ex = case
        mine, tree, fd = _both_sides(flat, a, sigma, rate, payer, tenor, ex)
        denom = max(abs(fd), 1.0)
        assert abs(mine - fd) / denom < MAX_MODEL_RELATIVE_GAP, (
            f"{_id}: engine={mine:.4f} ORE_fd={fd:.4f} ORE_tree={tree:.4f}; "
            f"relative gap {abs(mine - fd) / denom:.2e} exceeds the "
            f"{MAX_MODEL_RELATIVE_GAP} model-difference bound"
        )

    @pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
    def test_engine_does_not_exceed_ore_hull_white_value(self, case):
        """The gap is one-sided: across all cases the LGM engine prices at or below the
        Hull-White engines. A change pushing it above ORE would be a different discrepancy.
        The small slack covers deep-ITM cases where both are near intrinsic."""
        _id, flat, a, sigma, rate, payer, tenor, ex = case
        mine, _tree, fd = _both_sides(flat, a, sigma, rate, payer, tenor, ex)
        assert mine <= fd * 1.01 + 1.0, (
            f"{_id}: engine ({mine:.4f}) now prices ABOVE ORE ({fd:.4f}); the "
            f"documented HW-vs-LGM gap has always had the opposite sign"
        )

    def test_more_exercise_dates_never_decreases_value_in_both_engines(self):
        """More exercise dates never decrease value, on both sides (a mis-wired schedule on
        one side would break this there)."""
        common = (0.03, 0.03, 0.01, 0.03, True, "5Y")
        two_mine, _, two_fd = _both_sides(*common, [1, 3])
        four_mine, _, four_fd = _both_sides(*common, [1, 2, 3, 4])
        assert four_mine >= two_mine
        assert four_fd >= two_fd


class TestGapIsTheParametrizationNotTheInduction:
    """The controls attributing the gap to the model difference (module docstring)."""

    def test_engine_is_grid_converged(self):
        """Refining the grid does not move the price beyond the 5th significant figure, so
        the gap is not discretization error.

        The fine grid is 192 nodes per standard deviation over 8: measured 2026-10-05, it is
        within 1.6e-6 of 384 over 10 (8 or 10 standard deviations agree to 1e-11), at a peak of
        3.3 GB against 11.5 GB, which with three other test processes ran CI's 16 GB runner out
        of memory (I-27)."""
        flat, a, sigma, rate, payer, tenor, ex = 0.03, 0.03, 0.01, 0.03, True, "5Y", [1, 2, 3]
        times = _engine_exercise_dates(flat, a, sigma, rate, payer, tenor, ex)
        coarse = _npv(
            _engine_cfg(flat, a, sigma, rate, payer, tenor, times, n_per_std=48, std_devs=6.0))
        fine = _npv(
            _engine_cfg(flat, a, sigma, rate, payer, tenor, times, n_per_std=192, std_devs=8.0))
        assert abs(coarse - fine) / abs(fine) < 1e-4, (
            f"engine not grid-converged: coarse={coarse:.4f} fine={fine:.4f}"
        )
        # And refining does not move toward ORE's value.
        fd = _ore_bermudan_npv(flat, a, sigma, rate, payer, tenor, ex, "fd")
        assert abs(fine - fd) / abs(fd) > 1e-3

    def test_single_exercise_gap_matches_multi_exercise_gap(self):
        """With one exercise date the engine matches a direct integration to 2e-5
        (tests/test_bermudan_swaption.py::TestSingleExerciseMatchesDirectIntegration), so the
        gap there is the model. The multi-exercise gap is no larger."""
        flat, a, sigma, rate, payer, tenor = 0.03, 0.03, 0.01, 0.03, True, "5Y"

        single_times = _engine_exercise_dates(flat, a, sigma, rate, payer, tenor, [2])
        single_mine = _npv(
            _engine_cfg(flat, a, sigma, rate, payer, tenor, single_times))
        single_fd = _ore_bermudan_npv(flat, a, sigma, rate, payer, tenor, [2], "fd")
        single_gap = abs(single_mine - single_fd) / abs(single_fd)

        multi_mine, _, multi_fd = _both_sides(flat, a, sigma, rate, payer, tenor, [1, 2, 3])
        multi_gap = abs(multi_mine - multi_fd) / abs(multi_fd)

        assert multi_gap <= single_gap * 1.5 + 5e-3, (
            f"multi-exercise gap ({multi_gap:.2e}) is materially worse than the "
            f"single-exercise gap ({single_gap:.2e}), which would indicate error "
            f"in the early-exercise logic rather than in the parametrization"
        )

    def test_zero_vol_collapses_to_intrinsic(self):
        """Model-free anchor: at sigma -> 0 the option is worth the forward-starting swap,
        valued by ORE with a discounting engine and no model.

        The swap uses QuantLib indexed coupons, projected over each index fixing period, as
        ORE's LGM engine does (`LgmVectorised::fixing`, I-31). Default at-par coupons project
        over the accrual period and differ here by 2.3e-3 (1211.47 vs 1214.23).
        """
        flat, a, rate, payer, tenor = 0.03, 0.03, 0.03, True, "5Y"
        times = _engine_exercise_dates(flat, a, 1e-6, rate, payer, tenor, [2])
        mine = _npv(
            _engine_cfg(flat, a, 1e-6, rate, payer, tenor, times,
                        n_per_std=160, std_devs=9.0))

        # The swap entered on exercise: forward-starting over the remaining term.
        ORE.Settings.instance().evaluationDate = EVAL_DATE
        curve = ORE.YieldTermStructureHandle(ORE.FlatForward(EVAL_DATE, flat, DC))
        index = ORE.IborIndex(
            "SimIndex", ORE.Period(6, ORE.Months), 2,
            ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False, DC, curve,
        )
        was_at_par = ORE.IborCoupon.usingAtParCoupons()
        ORE.IborCoupon.createIndexedCoupons()  # process-global QuantLib setting
        try:
            forward_swap = ORE.MakeVanillaSwap(
                ORE.Period("3Y"), index, rate, nominal=NOTIONAL,
                swapType=ORE.VanillaSwap.Payer, fixedLegDayCount=DC, floatingLegDayCount=DC,
                forwardStart=ORE.Period(2, ORE.Years),
            )
            forward_swap.setPricingEngine(ORE.DiscountingSwapEngine(curve))
            intrinsic = forward_swap.NPV()
        finally:
            if was_at_par:
                ORE.IborCoupon.createAtParCoupons()

        assert mine == pytest.approx(intrinsic, rel=1e-3), (
            f"at sigma->0 the Bermudan must equal its intrinsic value: "
            f"engine={mine:.4f} intrinsic={intrinsic:.4f}"
        )


class TestHullWhiteVersusLgmBondPrices:
    """The Hull-White/LGM difference at the discount-bond level, using only ORE's own
    `HullWhite` and `LinearGaussMarkovModel` (no engine code): the evidence that a
    percent-level Bermudan gap is expected."""

    HW_A = 0.03
    HW_SIGMA = 0.01

    @staticmethod
    def _models(flat_rate=0.03, a=HW_A, sigma=HW_SIGMA):
        ORE.Settings.instance().evaluationDate = EVAL_DATE
        curve = ORE.YieldTermStructureHandle(ORE.FlatForward(EVAL_DATE, flat_rate, DC))
        hw = ORE.HullWhite(curve, a, sigma)
        param = ORE.IrLgm1fConstantParametrization(ORE.USDCurrency(), curve, sigma, a)
        return hw, ORE.LinearGaussMarkovModel(param)

    @pytest.mark.parametrize("T", [1.0, 5.0, 10.0])
    def test_models_agree_exactly_at_t0(self, T):
        """At t=0 both reduce to today's curve and agree to machine precision (so the
        divergence below is not a units mismatch)."""
        hw, lgm = self._models()
        np.testing.assert_allclose(hw.discountBond(0.0, T, 0.03), lgm.discountBond(0.0, T, 0.0),
                                   rtol=1e-10)

    @pytest.mark.parametrize("t,T", [(1.0, 5.0), (2.0, 5.0), (3.0, 5.0), (2.0, 3.0)])
    def test_models_diverge_for_t_greater_than_zero(self, t, T):
        """For t > 0 they diverge, by a pinned amount with a lower bound as well: if a
        future ORE made them agree, the tolerances in this file should be tightened."""
        hw, lgm = self._models()
        hw_price = hw.discountBond(t, T, 0.03)
        lgm_price = lgm.discountBond(t, T, 0.0)
        rel = abs(hw_price - lgm_price) / lgm_price
        assert 1e-5 < rel < 1e-2, (
            f"HW/LGM bond-price divergence at t={t}, T={T} is {rel:.2e}, outside "
            f"the measured range this file's tolerances are calibrated against"
        )


class TestExerciseDatesAreExact:
    """Exercise is given as dates (I-29).

    A rounded year fraction (2.0137 for 2.0136986301369864) once landed after the accrual
    start it meant, dropped a coupon and overstated the zero-vol price ~12x. A date equal
    to an accrual date maps to the identical time, so nothing needs snapping. Year fractions
    are refused (tests/test_bermudan_swaption.py::TestBermudanSwaptionConfigValidation).
    """

    # The 5Y annual-fixed underlying used throughout; index 2 is the accrual start from the
    # original I-29 report.
    ARGS = (0.03, 0.03, 1e-6, 0.03, True, "5Y")

    def test_an_accrual_date_is_the_identical_exercise_time(self):
        date = _engine_exercise_dates(*self.ARGS, [2])[0]
        prepared = _prepared(_engine_cfg(*self.ARGS, [date]))
        assert float(prepared.exercise_times[0]) == float(prepared.fixed_start_times[2])

    def test_every_exercisable_date_is_an_exercise_time_unchanged(self):
        dates = exercisable_dates(_engine_cfg(*self.ARGS, [EVAL_DATE + 1])[0])
        prepared = _prepared(_engine_cfg(*self.ARGS, dates[1:]))
        assert prepared.exercise_times.tolist() == prepared.fixed_start_times[1:].tolist()

    def test_exact_accrual_start_prices_its_intrinsic_at_zero_vol(self):
        """The I-29 case (once 14336.12): the zero-vol price is the intrinsic value. ORE's LGM
        engine gives 1214.2313035805 (tests/support/ore_lgm_oracle.py); I-29's 1211.47
        was the at-par-coupon value, before I-31."""
        date = _engine_exercise_dates(*self.ARGS, [2])[0]
        npv = _npv(_engine_cfg(*self.ARGS, [date], n_per_std=160, std_devs=9.0))
        assert npv == pytest.approx(1214.2313035805, rel=1e-9)
