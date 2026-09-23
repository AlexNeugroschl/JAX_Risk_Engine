"""
External-oracle tests for the Bermudan/American backward-induction engine
(`engine.instruments.bermudan_swaption`), against ORE's own multi-exercise
swaption engines.

WHY THIS FILE EXISTS. `tests/test_bermudan_swaption.py`'s own docstring
records that ORE's Python bindings expose no constructible
`NumericLgmMultiLegOptionEngine`, and concludes that the full backward
induction therefore "cannot be cross-checked against a live ORE engine
object end-to-end". The first half of that is true and still true --
verified again while writing this file: both `ORE.NumericLgmMultiLegOptionEngine`
and `ORE.AnalyticLgmSwaptionEngine` have no usable `__init__`. The second
half does not follow. `ORE.TreeSwaptionEngine` (Hull-White trinomial tree)
and `ORE.FdHullWhiteSwaptionEngine` (Hull-White finite differences) are
both fully constructible, both price a genuine `ORE.BermudanExercise`, and
so between them supply the external multi-exercise oracle that file says
does not exist.

WHAT THIS CAN AND CANNOT PROVE -- read before tightening any tolerance
here. ORE's two engines are Hull-White-parametrized; this engine's
Bermudan pricer is LGM-parametrized (`engine.models.lgm`), and those are
NOT the same numerical model realization for t>0 -- a fact this project
already documents (`docs/reference/ore-parity.md`, "A parametrization
note: LGM vs. plain Hull-White") and which `TestHullWhiteVersusLgmBondPrices`
below re-measures directly rather than taking on trust. So the agreement
this file asserts is a MODEL-LEVEL agreement of a few percent, not the
1e-4-style numerical parity the European-swaption tests get against
`ORE.JamshidianSwaptionEngine` (where both sides are the same
parametrization). Three separate facts establish that the residual gap is
the parametrization and not an error in the backward induction:

  1. ORE's two engines, built on completely different numerical schemes
     (tree vs. PDE), agree with EACH OTHER to ~1e-3 or better
     (`test_ore_tree_and_fd_engines_agree`) -- so the oracle itself is
     sound and the target value is not in doubt.
  2. This engine is fully grid-converged at the resolutions used here:
     refining `n_per_std` from 48 to 384 moves the price by ~1e-5
     relative (`test_engine_is_grid_converged`), so the residual gap is
     not discretization error that a finer grid would remove.
  3. The same-sized gap appears in the SINGLE-exercise case
     (`test_single_exercise_gap_matches_multi_exercise_gap`), where
     `tests/test_bermudan_swaption.py::TestSingleExerciseMatchesLgmJamshidian`
     already proves this engine matches its own LGM closed form to 2e-4.
     A discrepancy that is present with one exercise date and no larger
     with four is not coming from the early-exercise logic.

Together those pin the gap to the HW/LGM parametrization difference, which
is exactly what `TestHullWhiteVersusLgmBondPrices` quantifies at the
bond-price level. What this file DOES prove, and what nothing in the suite
proved before it: the multi-exercise backward induction produces values a
real, independent, multi-exercise ORE engine also produces, to within that
known model difference -- and it would fail loudly on any change that
moved a Bermudan price by more than a few percent.

TWO CONSTRUCTION TRAPS, both of which silently produce a green-looking
wrong answer and both of which were hit while writing this file:

  * `engine.models.ore_builders.build_vanilla_swap` deliberately builds
    its `ORE.IborIndex` on a **0% dummy forward curve** -- correct for this
    engine, which never reads ORE's forwards and reprices the floating leg
    off its own simulated/LGM curve, but fatal for an ORE PRICING engine,
    which does read it. Reusing that helper to build the oracle's
    underlying gives a swap whose floating leg is identically zero
    (`fairRate()` == 0.0, first coupon amount 0.00) and Bermudan prices
    ~40x too small. The ORE side here therefore builds its own index on a
    REAL forwarding curve; `test_oracle_underlying_swap_is_not_the_dummy_curve_swap`
    is a standing guard that this distinction is never "simplified" away.
  * `exercise_times` are matched to the underlying's fixed-leg accrual
    starts by reading them back off the schedule (`exercisable_times`),
    never by writing a rounded literal. An exercise time of 2.0137 instead
    of the true 2.0136986301369864 is 1.4e-6 late, which falls outside
    `_hw_swap_value_at_nodes`' own `>= t - 1e-9` liveness tolerance and
    silently dropped that date's entire fixed coupon from the exercise
    value -- a 12x overstatement at low vol.

    **That trap is now closed** (I-29): `prepare_bermudan` snaps a
    near-miss onto the accrual start it names, so the rounded literal above
    prices identically to the exact one. `TestExerciseTimeAlignment` is the
    regression test, and it also pins the two things the fix deliberately
    does NOT change -- a genuinely mid-period date and an American's
    discretized grid are both left alone. Reading times off the schedule
    remains the right habit here regardless: it keeps the engine and ORE
    sides on the same calendar dates even if ORE's schedule generation ever
    shifts one by a business day, which snapping would silently follow.
"""
import warnings

import numpy as np
import ORE
import pytest

from engine.simulation.market_model import ZeroCurveConfig
from engine.instruments.bermudan_swaption import (
    EXERCISE_SNAP_TOLERANCE,
    BermudanSwaptionConfig,
    exercisable_times,
    prepare_bermudan,
    price_bermudan_swaption_base,
)
from engine.models.ore_builders import TIME_AXIS_DAY_COUNTER
from engine.portfolio.request import _warn_if_not_reset_aligned

EVAL_DATE = ORE.Date(30, 7, 2026)
DC = TIME_AXIS_DAY_COUNTER
NOTIONAL = 1_000_000.0

# Grid resolution used for every comparison below. Deliberately well past
# the point of convergence (test_engine_is_grid_converged measures it) so
# no assertion here is a statement about discretization.
N_PER_STD = 128
STD_DEVS = 8.0

# ORE engine resolutions. Also past convergence -- test_ore_tree_and_fd_engines_agree
# is what establishes that, and it is the reason these are not tuning knobs.
ORE_TREE_STEPS = 800
ORE_FD_GRID = 800


def _flat_curve(rate: float) -> ZeroCurveConfig:
    return ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[rate] * 6)


def _ore_underlying(flat_rate, fixed_rate, payer, tenor, notional=NOTIONAL):
    """The ORE side's underlying vanilla swap, built on a REAL forwarding
    curve at `flat_rate`.

    Deliberately NOT `engine.models.ore_builders.build_vanilla_swap`: see
    this module's docstring: that helper's 0% dummy forward curve is
    correct for this engine and silently wrong for an ORE pricing engine.
    Everything else here (index tenor, spot lag, calendar, roll convention,
    both legs' day count) is kept identical to that helper so the two sides
    price the same instrument.
    """
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
    """Prices a Bermudan swaption with one of ORE's own multi-exercise
    engines. `exercise_indices` index the underlying's own fixed-leg
    accrual-start dates -- ORE's standard coterminal Bermudan convention,
    and the same set of dates the engine side is handed (as year fractions)
    by `_engine_exercise_times`."""
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
                exercise_times, n_per_std=N_PER_STD, std_devs=STD_DEVS,
                notional=NOTIONAL):
    return BermudanSwaptionConfig(
        notional=notional, fixed_rate=fixed_rate, payer=payer, rate_factor_index=0,
        hw_a=hw_a, hw_sigma=hw_sigma, initial_zero_curve=_flat_curve(flat_rate),
        exercise_times=exercise_times, swap_tenor=tenor, evaluation_date=EVAL_DATE,
        n_per_std=n_per_std, std_devs=std_devs,
    )


def _engine_exercise_times(flat_rate, hw_a, hw_sigma, fixed_rate, payer, tenor,
                           exercise_indices, notional=NOTIONAL):
    """The EXACT year fractions of the underlying's fixed-leg accrual
    starts, read off the engine's own schedule rather than written as
    rounded literals.

    Still the right way to build these even though `prepare_bermudan` now
    snaps a near-miss onto the schedule (`_snap_exercise_times`, I-29):
    reading them back guarantees the engine and ORE sides are talking about
    the same calendar dates even if ORE's schedule generation ever shifts
    one by a business day, which snapping alone would not catch -- it would
    follow the shift silently.

    Uses the public `exercisable_times` rather than a throwaway
    `prepare_bermudan` call: the old form passed a dummy
    `exercise_times=[0.0]` purely to reach `fixed_start_times`, and that
    dummy is now correctly refused as unaligned.
    """
    probe = _engine_cfg(flat_rate, hw_a, hw_sigma, fixed_rate, payer, tenor,
                        exercise_times=[0.0], notional=notional)
    starts = exercisable_times(probe)
    return [starts[i] for i in exercise_indices]


def _both_sides(flat_rate, hw_a, hw_sigma, fixed_rate, payer, tenor, exercise_indices):
    """(engine NPV, ORE tree NPV, ORE FD NPV) for one trade."""
    times = _engine_exercise_times(flat_rate, hw_a, hw_sigma, fixed_rate, payer,
                                   tenor, exercise_indices)
    cfg = _engine_cfg(flat_rate, hw_a, hw_sigma, fixed_rate, payer, tenor, times)
    mine = price_bermudan_swaption_base(cfg)
    tree = _ore_bermudan_npv(flat_rate, hw_a, hw_sigma, fixed_rate, payer, tenor,
                             exercise_indices, "tree")
    fd = _ore_bermudan_npv(flat_rate, hw_a, hw_sigma, fixed_rate, payer, tenor,
                           exercise_indices, "fd")
    return mine, tree, fd


# The comparison grid. Sweeps moneyness (fixed_rate against a 3% curve),
# volatility, payer/receiver, mean reversion, curve level, and the number
# and spacing of exercise dates -- every axis the backward induction's own
# behavior could plausibly depend on.
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

# Measured worst case across CASES is 1.51e-1 (deep-OTM receiver, where the
# option is worth ~830 on a 1e6 notional and a small absolute model
# difference is a large relative one); the ATM cases sit near 3e-2. This
# bound is deliberately just above the measured worst case rather than
# round: it is a REGRESSION bound on a known, explained model difference,
# and anything that widens it is a real change that should be looked at.
MAX_MODEL_RELATIVE_GAP = 0.16

# ORE's two engines are independent numerical schemes for the SAME model,
# so they agree far more tightly than either agrees with this engine.
# Measured worst case across CASES is 1.34e-3.
MAX_ORE_INTERNAL_GAP = 2e-3


class TestOracleIsSound:
    """Establishes that ORE's own multi-exercise engines are a trustworthy
    target BEFORE anything is compared against them. If these fail, no
    other assertion in this file means anything."""

    @pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
    def test_ore_tree_and_fd_engines_agree(self, case):
        """ORE's trinomial tree and its Hull-White PDE solver are entirely
        different numerical schemes. Their agreeing on every case to ~1e-3
        is what makes 'the ORE number' well-defined at all -- and it is the
        control that separates 'this engine differs from ORE' from 'ORE's
        own engines have not converged'."""
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
        """Guards the first construction trap in this module's docstring.

        `build_vanilla_swap`'s 0% dummy forwarding curve makes every
        floating coupon zero, which an ORE pricing engine faithfully
        prices -- yielding a wrong Bermudan value that looks entirely
        plausible. This asserts the oracle's own underlying has a LIVE
        floating leg (non-zero first coupon, fair rate near the curve),
        so that a future refactor 'simplifying' `_ore_underlying` into a
        call to the shared builder fails here rather than silently
        weakening every comparison in the file.
        """
        curve, swap = _ore_underlying(0.03, 0.03, True, "5Y")
        first_float = ORE.as_floating_rate_coupon(swap.floatingLeg()[0])
        assert first_float.amount() > 0.0, (
            "oracle's floating leg is identically zero -- the underlying was "
            "built on a dummy 0% forward curve (see module docstring)"
        )
        swap.setPricingEngine(ORE.DiscountingSwapEngine(curve))
        assert swap.fairRate() == pytest.approx(0.03, abs=1e-3)


class TestEngineMatchesOreBermudanEngines:
    """The claim this file exists to make: the multi-exercise backward
    induction agrees with a real, independent, multi-exercise ORE engine
    to within the documented HW/LGM parametrization difference."""

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
        """The gap is ONE-SIDED, and that directionality is itself a
        checkable fact rather than an incidental observation: across every
        case measured, the LGM-parametrized engine prices at or below the
        HW-parametrized ORE engines. A future change that pushed this
        engine ABOVE ORE would be a qualitatively different discrepancy
        from the one analyzed in this module's docstring, and should not
        pass quietly just because its magnitude happens to sit inside the
        relative bound above.

        The small positive slack absorbs the two cases where the gap is
        near zero (deep-ITM, where both models price close to intrinsic).
        """
        _id, flat, a, sigma, rate, payer, tenor, ex = case
        mine, _tree, fd = _both_sides(flat, a, sigma, rate, payer, tenor, ex)
        assert mine <= fd * 1.01 + 1.0, (
            f"{_id}: engine ({mine:.4f}) now prices ABOVE ORE ({fd:.4f}); the "
            f"documented HW-vs-LGM gap has always had the opposite sign"
        )

    def test_more_exercise_dates_never_decreases_value_in_both_engines(self):
        """A no-arbitrage property both sides must independently satisfy --
        a cross-check on the comparison itself rather than on either
        engine, since a mis-wired exercise schedule on ONE side (the most
        likely way this file could silently compare two different trades)
        would break the monotonicity on that side alone."""
        common = (0.03, 0.03, 0.01, 0.03, True, "5Y")
        two_mine, _, two_fd = _both_sides(*common, [1, 3])
        four_mine, _, four_fd = _both_sides(*common, [1, 2, 3, 4])
        assert four_mine >= two_mine
        assert four_fd >= two_fd


class TestGapIsTheParametrizationNotTheInduction:
    """The three controls that attribute the residual gap to the HW/LGM
    model difference rather than to the backward induction. Without these,
    `MAX_MODEL_RELATIVE_GAP` would just be a tolerance wide enough to hide
    a bug."""

    def test_engine_is_grid_converged(self):
        """If the gap against ORE were discretization error, refining the
        state grid would shrink it. It does not move the price at all
        beyond the 5th significant figure, so it cannot be."""
        flat, a, sigma, rate, payer, tenor, ex = 0.03, 0.03, 0.01, 0.03, True, "5Y", [1, 2, 3]
        times = _engine_exercise_times(flat, a, sigma, rate, payer, tenor, ex)
        coarse = price_bermudan_swaption_base(
            _engine_cfg(flat, a, sigma, rate, payer, tenor, times, n_per_std=48, std_devs=6.0))
        fine = price_bermudan_swaption_base(
            _engine_cfg(flat, a, sigma, rate, payer, tenor, times, n_per_std=384, std_devs=10.0))
        assert abs(coarse - fine) / abs(fine) < 1e-4, (
            f"engine not grid-converged: coarse={coarse:.4f} fine={fine:.4f}"
        )
        # And the converged value is still meaningfully away from ORE's --
        # i.e. refining does not walk toward it.
        fd = _ore_bermudan_npv(flat, a, sigma, rate, payer, tenor, ex, "fd")
        assert abs(fine - fd) / abs(fd) > 1e-3

    def test_single_exercise_gap_matches_multi_exercise_gap(self):
        """The decisive control. With ONE exercise date, this engine is
        already known to match its own LGM closed form to 2e-4
        (tests/test_bermudan_swaption.py::TestSingleExerciseMatchesLgmJamshidian),
        so any gap against ORE there is definitionally the parametrization
        and not the early-exercise logic. If the multi-exercise gap is no
        larger than the single-exercise gap, the induction is adding no
        error of its own."""
        flat, a, sigma, rate, payer, tenor = 0.03, 0.03, 0.01, 0.03, True, "5Y"

        single_times = _engine_exercise_times(flat, a, sigma, rate, payer, tenor, [2])
        single_mine = price_bermudan_swaption_base(
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
        """A model-free anchor: as sigma -> 0 the option must be worth
        exactly the forward-starting underlying swap, which ORE values
        with a plain discounting engine and no model at all. This is the
        one check in the file that is NOT subject to the HW/LGM difference
        (both parametrizations agree at zero vol), so it holds to 1e-3."""
        flat, a, rate, payer, tenor = 0.03, 0.03, 0.03, True, "5Y"
        times = _engine_exercise_times(flat, a, 1e-6, rate, payer, tenor, [2])
        mine = price_bermudan_swaption_base(
            _engine_cfg(flat, a, 1e-6, rate, payer, tenor, times,
                        n_per_std=160, std_devs=9.0))

        # The swap you receive on exercising at that date: a forward-starting
        # swap over the remaining term, priced off today's curve.
        ORE.Settings.instance().evaluationDate = EVAL_DATE
        curve = ORE.YieldTermStructureHandle(ORE.FlatForward(EVAL_DATE, flat, DC))
        index = ORE.IborIndex(
            "SimIndex", ORE.Period(6, ORE.Months), 2,
            ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False, DC, curve,
        )
        forward_swap = ORE.MakeVanillaSwap(
            ORE.Period("3Y"), index, rate, nominal=NOTIONAL,
            swapType=ORE.VanillaSwap.Payer, fixedLegDayCount=DC, floatingLegDayCount=DC,
            forwardStart=ORE.Period(2, ORE.Years),
        )
        forward_swap.setPricingEngine(ORE.DiscountingSwapEngine(curve))
        intrinsic = forward_swap.NPV()

        assert mine == pytest.approx(intrinsic, rel=1e-3), (
            f"at sigma->0 the Bermudan must equal its intrinsic value: "
            f"engine={mine:.4f} intrinsic={intrinsic:.4f}"
        )


class TestHullWhiteVersusLgmBondPrices:
    """Quantifies the HW/LGM parametrization difference at the level it
    originates -- the discount bond -- rather than leaving it as a single
    remembered '~0.6% at t=3y' figure in prose
    (docs/reference/ore-parity.md). Both sides here are ORE's OWN objects,
    so this measures the difference between two ORE models and involves
    this engine not at all: it is the independent evidence that a
    percent-level Bermudan gap is the expected consequence of the
    parametrization choice."""

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
        """At t=0 both parametrizations reduce to today's curve, so they
        must agree to machine precision. This is the control proving the
        divergence measured below is genuinely about time evolution and
        not a units/convention mismatch in how this test calls them."""
        hw, lgm = self._models()
        np.testing.assert_allclose(hw.discountBond(0.0, T, 0.03), lgm.discountBond(0.0, T, 0.0),
                                   rtol=1e-10)

    @pytest.mark.parametrize("t,T", [(1.0, 5.0), (2.0, 5.0), (3.0, 5.0), (2.0, 3.0)])
    def test_models_diverge_for_t_greater_than_zero(self, t, T):
        """The divergence is real and one-directional, and this pins its
        measured size. Asserting a LOWER bound as well as an upper one is
        deliberate: if a future ORE version made these two agree, every
        percent-level tolerance in this file would be unjustified and
        should be tightened -- so that change must fail here loudly rather
        than silently leaving the tolerances too loose."""
        hw, lgm = self._models()
        hw_price = hw.discountBond(t, T, 0.03)
        lgm_price = lgm.discountBond(t, T, 0.0)
        rel = abs(hw_price - lgm_price) / lgm_price
        assert 1e-5 < rel < 1e-2, (
            f"HW/LGM bond-price divergence at t={t}, T={T} is {rel:.2e}, outside "
            f"the measured range this file's tolerances are calibrated against"
        )


class TestExerciseTimeAlignment:
    """The I-29 fix: an exercise time that is a near-miss SPELLING of an
    accrual date is snapped onto it; everything else is left alone.

    This class previously PINNED the defect -- a rounded exercise time
    silently dropped a coupon and overstated the zero-vol price ~12x -- with
    a note saying that any change which snapped exercise times should delete
    that test and tighten this one. That is what happened; the tests below
    are the tightened form, and `test_rounded_exercise_time_matches_exact`
    is the direct inversion of the deleted `..._drops_a_coupon`.

    The scope is deliberately narrow, and half these tests defend that
    narrowness rather than the repair: a genuinely mid-period exercise date
    is a SUPPORTED trade (its understatement is the documented I-06), and an
    American's discretized grid is unaligned by construction. Neither may be
    touched.
    """

    # The 5Y annual-fixed underlying used throughout: accrual starts at
    # ~0.011, 1.011, 2.014, 3.014, 4.019. Index 2 is the one the original
    # defect report used (true value 2.0136986301369864).
    ARGS = (0.03, 0.03, 1e-6, 0.03, True, "5Y")

    def _price(self, times):
        return price_bermudan_swaption_base(
            _engine_cfg(*self.ARGS, times, n_per_std=160, std_devs=9.0))

    def test_exact_accrual_start_matches_intrinsic_at_zero_vol(self):
        """With the exact accrual-start time, the zero-vol price is the
        intrinsic value (see TestGapIsTheParametrizationNotTheInduction::
        test_zero_vol_collapses_to_intrinsic for the ORE-side value)."""
        times = _engine_exercise_times(*self.ARGS, [2])
        assert self._price(times) == pytest.approx(1211.47, rel=1e-3)

    @pytest.mark.parametrize("decimals", [4, 5, 6])
    def test_rounded_exercise_time_matches_exact(self, decimals):
        """THE REGRESSION TEST FOR I-29. A rounded literal now prices
        identically to the exact accrual start, because it is snapped onto
        it before the liveness test ever sees it.

        Against the pre-fix code the 4-decimal case priced 14336.12 against
        an exact 1211.47 -- an ~12x overstatement -- so this fails loudly
        without the fix rather than merely losing precision.
        """
        exact = _engine_exercise_times(*self.ARGS, [2])[0]
        rounded = round(exact, decimals)
        assert rounded != exact, "rounding must actually perturb the input"
        # Bit-identical, not just close: snapping replaces the caller's
        # value with the schedule's own float, so the two prices come from
        # numerically identical inputs.
        assert self._price([rounded]) == self._price([exact])

    def test_snapping_restores_the_schedule_value_exactly(self):
        """The snapped time is the schedule's own float, not the caller's
        rounded one -- which is what makes the prices above bit-identical
        rather than merely close."""
        exact = _engine_exercise_times(*self.ARGS, [2])[0]
        prepared = prepare_bermudan(_engine_cfg(*self.ARGS, [round(exact, 4)]))
        assert float(prepared.exercise_times[0]) == exact

    def test_a_genuinely_mid_period_date_is_passed_through_untouched(self):
        """THE NEGATIVE CONTROL, and the one that matters most: snapping
        must repair a damaged spelling of an accrual date and NOTHING else.

        A true mid-period exercise is a supported trade whose
        value-understating approximation is the documented I-06 -- not an
        error. Moving it to a neighbouring accrual date would answer a
        different question than the caller asked, and refusing it would
        delete a working capability (tried: it broke 66 tests).
        """
        prepared = prepare_bermudan(_engine_cfg(*self.ARGS, [2.5]))
        assert float(prepared.exercise_times[0]) == 2.5

    def test_the_tolerance_band_is_where_it_is_documented_to_be(self):
        """Just inside snaps; just outside is left exactly as written. Pins
        the actual boundary so a future widening is a deliberate, visible
        edit rather than a drift -- the tolerance is a contract, not a
        tuning knob."""
        exact = _engine_exercise_times(*self.ARGS, [2])[0]

        inside = exact + 0.9 * EXERCISE_SNAP_TOLERANCE
        prepared = prepare_bermudan(_engine_cfg(*self.ARGS, [inside]))
        assert float(prepared.exercise_times[0]) == exact

        outside = exact + 1.1 * EXERCISE_SNAP_TOLERANCE
        prepared = prepare_bermudan(_engine_cfg(*self.ARGS, [outside]))
        assert float(prepared.exercise_times[0]) == outside

    def test_a_coarsely_rounded_time_is_still_not_repaired(self):
        """THE HONEST LIMIT OF THIS FIX, pinned rather than left to be
        rediscovered. A 3-decimal exercise time (2.014 for 2.01369...) is
        3e-4 out -- a tenth of a day, outside the snap band -- so it is left
        as written and still drops the coupon, pricing ~12x its intrinsic.

        That is deliberate: at that coarseness the engine cannot tell a typo
        from an intentional mid-period date, and guessing would be the silent
        approximation the band exists to avoid. The defence is
        `exercisable_times`, not a wider tolerance -- widening it to cover
        this would start moving genuine mid-period dates (I-06).
        """
        exact = _engine_exercise_times(*self.ARGS, [2])[0]
        coarse = round(exact, 3)
        assert abs(coarse - exact) > EXERCISE_SNAP_TOLERANCE
        assert prepare_bermudan(
            _engine_cfg(*self.ARGS, [coarse])).exercise_times[0] == coarse
        assert self._price([coarse]) > self._price([exact]) * 5

    def test_an_american_grid_is_exempt_from_snapping(self):
        """The discretization exemption, asserted on a grid point placed
        deliberately inside the snap tolerance of an accrual start. Without
        the exemption this would be moved, silently shifting one of the
        exercise opportunities the discretization is made of."""
        exact = _engine_exercise_times(*self.ARGS, [2])[0]
        near = exact + 0.5 * EXERCISE_SNAP_TOLERANCE

        cfg = _engine_cfg(*self.ARGS, [near])
        cfg.exercise_times_are_discretized = True
        assert float(prepare_bermudan(cfg).exercise_times[0]) == near

    def test_tolerance_cannot_reach_a_neighbouring_accrual_date(self):
        """The safety property behind the chosen tolerance: it is far
        smaller than the gap between accrual starts, so snapping can never
        be ambiguous between two boundaries. If a future schedule change
        (or a wider tolerance) broke this, snapping could silently retarget
        a trade to the wrong date."""
        starts = np.asarray(exercisable_times(_engine_cfg(*self.ARGS, [0.0])))
        min_gap = float(np.min(np.diff(starts)))
        assert min_gap > 100 * EXERCISE_SNAP_TOLERANCE

    def test_every_exercisable_time_is_accepted_unchanged(self):
        """`exercisable_times` is the documented way to build a config, so
        every value it returns must survive `prepare_bermudan` untouched."""
        cfg = _engine_cfg(*self.ARGS, [0.0])
        starts = exercisable_times(cfg)
        prepared = prepare_bermudan(_engine_cfg(*self.ARGS, starts))
        assert prepared.exercise_times.tolist() == starts

    def test_the_portfolio_warning_agrees_with_what_is_priced(self):
        """The mid-coupon warning must describe what the pricer DOES.

        `price_portfolio` warns that a misaligned trade "will use the
        documented mid-coupon approximation". Once a near-miss is snapped
        that is no longer true of it, so the warning has to match on the
        same tolerance rather than on exact equality -- otherwise a trade
        that is now priced exactly still reports an approximation it does
        not use, which is its own species of misleading output.

        Both directions are asserted: silent for a snapped time, still
        warning for a genuine mid-period one.
        """
        exact = _engine_exercise_times(*self.ARGS, [2])[0]

        for time, expected in ((exact, 0), (round(exact, 4), 0), (2.5, 1)):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _warn_if_not_reset_aligned(
                    "trade[0]", _engine_cfg(*self.ARGS, [time]), 0)
            aligned = [w for w in caught if "not reset-aligned" in str(w.message)]
            assert len(aligned) == expected, f"exercise_time={time!r}"
