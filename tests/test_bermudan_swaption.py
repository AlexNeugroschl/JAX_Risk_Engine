"""
`engine.instruments.bermudan_swaption`, the LGM backward induction (also used for
Americans, whose own tests are in tests/test_american_swaption.py), with an explicit model
(`tests.support.lgm_engine`: the trade carries none).

The authoritative check is tests/test_ore_lgm_parity.py (ORE's own
`NumericLgmMultiLegOptionEngine`, ~1e-11). These tests are independent of ORE's engine:

  - the LGM closed forms (`H`, `zeta`, bond price) against `ORE.IrLgm1fConstantParametrization`
    and `ORE.LinearGaussMarkovModel`;
  - the single-exercise case against a direct integration of the Gaussian expectation
    (tests/bermudan_references.py);
  - model-independent properties: monotonicity in exercise dates, direction, moneyness,
    grid convergence.
"""
import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from bermudan_references import single_exercise_value_by_integration
from date_helpers import in_years
from engine.instruments.bermudan_swaption import (
    BermudanSwaptionConfig,
    _hagan_quadrature_weights,
    exercisable_dates,
    _state_grid,
)
from engine.instruments.european_swaption import SwaptionConfig
from engine.market import ZeroCurveConfig
from engine.models.hull_white import ZeroCurve as HwZeroCurve
from engine.models.lgm import H as _H, bond_price as _lgm_bond_price, zeta as _zeta
from engine.valuation.config import JamshidianEngineConfig
from engine.valuation.european import european_terms
from engine.valuation.jamshidian import jamshidian_npv
from tests.support.lgm_engine import grid_npv, prepared


def _lgm_bond(zero_times, zero_rates, a, sigma, t, T, x):
    """`engine.models.lgm.bond_price` in the older NumPy call shape
    `(zero_times, zero_rates, a, sigma, t, T, x)`."""
    curve = HwZeroCurve(pillar_times=jnp.asarray(zero_times), pillar_rates=jnp.asarray(zero_rates))
    return np.asarray(_lgm_bond_price(curve, a, sigma, jnp.asarray(t), jnp.asarray(T), jnp.asarray(x)))

FLAT_CURVE = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[0.03] * 6)
EVAL_DATE = ORE.Date(30, 7, 2026)


def _in_years(years):
    return in_years(EVAL_DATE, years)


#: The engine's model and grid; a case overrides any of them by keyword.
MODEL = dict(a=0.03, sigma=0.02, curve=FLAT_CURVE, n_per_std=96, std_devs=7.0)


def _make_bermudan(**overrides):
    """(trade, model): the trade's fields and the model's (`MODEL`'s keys) in one call."""
    trade = dict(notional=1_000_000.0, fixed_rate=0.030, payer=True, exercise_dates=_in_years([1.0, 2.0, 3.0, 4.0]),
                 swap_tenor="5Y", evaluation_date=EVAL_DATE, trade_id="bermudan")
    model = dict(MODEL)
    for key, value in overrides.items():
        (model if key in MODEL else trade)[key] = value
    return BermudanSwaptionConfig(**trade), model


def _npv(case) -> float:
    cfg, model = case
    return grid_npv(cfg, **model)


def _prepared(case):
    cfg, model = case
    return prepared(cfg, **model)


class TestLgmClosedFormsAgainstORE:
    """The LGM closed forms against ORE's LGM objects."""

    def test_H_matches_ore_parametrization(self):
        today = EVAL_DATE
        ORE.Settings.instance().evaluationDate = today
        dc = ORE.Actual365Fixed()
        curve = ORE.YieldTermStructureHandle(ORE.FlatForward(today, 0.03, dc))
        param = ORE.IrLgm1fConstantParametrization(ORE.USDCurrency(), curve, 0.02, 0.03)
        for t in [0.5, 1.0, 3.0, 7.5]:
            assert _H(0.03, t) == pytest.approx(param.H(t), abs=1e-12)

    def test_zeta_matches_ore_parametrization(self):
        today = EVAL_DATE
        ORE.Settings.instance().evaluationDate = today
        dc = ORE.Actual365Fixed()
        curve = ORE.YieldTermStructureHandle(ORE.FlatForward(today, 0.03, dc))
        param = ORE.IrLgm1fConstantParametrization(ORE.USDCurrency(), curve, 0.02, 0.03)
        for t in [0.5, 1.0, 3.0, 7.5]:
            assert _zeta(0.02, t) == pytest.approx(param.zeta(t), abs=1e-12)

    def test_lgm_bond_matches_ore_linear_gauss_markov_model(self):
        today = EVAL_DATE
        ORE.Settings.instance().evaluationDate = today
        dc = ORE.Actual365Fixed()
        curve = ORE.YieldTermStructureHandle(ORE.FlatForward(today, 0.03, dc))
        param = ORE.IrLgm1fConstantParametrization(ORE.USDCurrency(), curve, 0.02, 0.03)
        lgm = ORE.LinearGaussMarkovModel(param)

        zero_times = np.array(FLAT_CURVE.times)
        zero_rates = np.array(FLAT_CURVE.rates)
        for t, T, x in [
            (0.0, 5.0, 0.0), (1.0, 3.0, 0.05), (3.0, 5.01643836, 0.05),
            (3.0, 5.01643836, -0.1), (2.0, 2.0001, 0.02),
        ]:
            mine = _lgm_bond(zero_times, zero_rates, 0.03, 0.02, t, T, np.array([x]))[0]
            ore_val = lgm.discountBond(t, T, x)
            assert mine == pytest.approx(ore_val, rel=1e-9)

    def test_lgm_bond_differs_from_hullwhite_for_t_greater_than_zero(self):
        """`ORE.HullWhite(a, sigma)` and ORE's constant-parametrization LGM with alpha = sigma
        are different models for t > 0: the same number is the short rate's volatility in one
        and the LGM state's in the other (zeta = sigma^2 (e^{2at} - 1)/(2a) against
        alpha^2 t), and the state x = 0 is r = f(0,t) + zeta H H', not r = f(0,t). ORE's
        Bermudan engine is the LGM, so these are its formulas. The LGM in the Hull-White
        parametrization is the Hull-White model exactly
        (tests/test_cam.py::test_hull_white_path_curves_equal_quantlibs_hull_white)."""
        today = EVAL_DATE
        ORE.Settings.instance().evaluationDate = today
        dc = ORE.Actual365Fixed()
        curve = ORE.YieldTermStructureHandle(ORE.FlatForward(today, 0.03, dc))
        hw = ORE.HullWhite(curve, 0.03, 0.02)
        param = ORE.IrLgm1fConstantParametrization(ORE.USDCurrency(), curve, 0.02, 0.03)
        lgm = ORE.LinearGaussMarkovModel(param)

        t, T = 3.0, 5.0
        hw_bond = hw.discountBond(t, T, 0.03)  # r = f(0,t) = 0.03 (flat curve)
        lgm_bond = lgm.discountBond(t, T, 0.0)  # x = 0
        # Different models: not expected to match; the gap is pinned.
        assert abs(hw_bond - lgm_bond) / lgm_bond > 1e-4

    def test_state_grid_collapses_to_single_zero_at_t0(self):
        grid = _state_grid(0.02, 0.0, 32, 5.0)
        assert np.all(grid == 0.0)
        assert grid.shape[0] == 2 * 32 * 5 + 1

    def test_hagan_quadrature_weights_are_valid_probability_measure(self):
        w = _hagan_quadrature_weights(64, 7.0)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.all(w >= 0.0)

    def test_hagan_quadrature_first_and_second_moments(self):
        n_per_std, std_devs = 64, 7.0
        h = 1.0 / n_per_std
        my = int(round(std_devs * n_per_std))
        y = h * np.arange(-my, my + 1)
        w = _hagan_quadrature_weights(n_per_std, std_devs)
        assert np.sum(w * y) == pytest.approx(0.0, abs=1e-6)
        assert np.sum(w * y ** 2) == pytest.approx(1.0, abs=1e-3)


class TestSingleExerciseMatchesDirectIntegration:
    """With one exercise date the value is a single Gaussian expectation, computed without
    the grid by tests/bermudan_references.py."""

    @staticmethod
    def _direct_integration(case):
        cfg, model = case
        return single_exercise_value_by_integration(cfg, **model)

    @pytest.mark.slow
    @pytest.mark.parametrize("payer", [True, False])
    @pytest.mark.parametrize("exercise_time", [1.0, 2.5, 4.0])
    def test_matches_direct_integration(self, payer, exercise_time):
        cfg = _make_bermudan(payer=payer, exercise_dates=_in_years([exercise_time]), n_per_std=192, std_devs=9.0)
        # Measured 0.7-5e-6: the rollback's discretization at this grid.
        assert _npv(cfg) == pytest.approx(self._direct_integration(cfg), rel=2e-5)

    def test_grid_convergence_toward_direct_integration(self):
        cfg_coarse = _make_bermudan(exercise_dates=_in_years([3.0]), n_per_std=48, std_devs=6.0)
        cfg_fine = _make_bermudan(exercise_dates=_in_years([3.0]), n_per_std=256, std_devs=9.0)
        reference = self._direct_integration(cfg_fine)
        err_coarse = abs(_npv(cfg_coarse) - reference)
        err_fine = abs(_npv(cfg_fine) - reference)
        assert err_fine < err_coarse


class TestMonotonicity:
    """More exercise dates can never decrease a Bermudan's value."""

    def test_bermudan_at_least_as_valuable_as_either_single_exercise(self):
        euro_first = _npv(_make_bermudan(exercise_dates=_in_years([1.0])))
        euro_last = _npv(_make_bermudan(exercise_dates=_in_years([4.0])))
        bermudan_2 = _npv(_make_bermudan(exercise_dates=_in_years([1.0, 4.0])))
        assert bermudan_2 >= max(euro_first, euro_last) - 1e-6

    def test_more_exercise_dates_never_decreases_value(self):
        bermudan_2 = _npv(_make_bermudan(exercise_dates=_in_years([1.0, 4.0])))
        bermudan_4 = _npv(_make_bermudan(exercise_dates=_in_years([1.0, 2.0, 3.0, 4.0])))
        assert bermudan_4 >= bermudan_2 - 1e-6

    def test_itm_payer_worth_more_than_otm_payer(self):
        itm = _npv(_make_bermudan(fixed_rate=0.01, payer=True))
        otm = _npv(_make_bermudan(fixed_rate=0.08, payer=True))
        assert itm > otm

    def test_itm_receiver_worth_more_than_otm_receiver(self):
        itm = _npv(_make_bermudan(fixed_rate=0.08, payer=False))
        otm = _npv(_make_bermudan(fixed_rate=0.01, payer=False))
        assert itm > otm

    def test_higher_volatility_increases_value(self):
        low_vol = _npv(_make_bermudan(sigma=0.005))
        high_vol = _npv(_make_bermudan(sigma=0.04))
        assert high_vol > low_vol


class TestShape:
    def test_zero_notional_prices_to_zero(self):
        cfg = _make_bermudan(notional=0.0)
        assert _npv(cfg) == pytest.approx(0.0, abs=1e-8)


class TestEdgeCases:
    def test_rejects_exercise_time_at_or_after_final_maturity(self):
        with pytest.raises(ValueError):
            _prepared(_make_bermudan(exercise_dates=_in_years([10.0])))

    def test_exercise_date_exactly_at_final_reset_prices_finite(self):
        cfg = _make_bermudan(exercise_dates=_in_years([4.0]))
        npv = _npv(cfg)
        assert np.isfinite(npv)
        assert npv >= 0.0

    def test_negative_rate_curve_prices_finite(self):
        neg_curve = ZeroCurveConfig(times=FLAT_CURVE.times, rates=[-0.005] * 6)
        cfg = _make_bermudan(curve=neg_curve, fixed_rate=-0.005)
        npv = _npv(cfg)
        assert np.isfinite(npv)
        assert npv >= 0.0

    def test_near_zero_volatility_collapses_toward_intrinsic(self):
        cfg_low_vol = _make_bermudan(sigma=1e-6, exercise_dates=_in_years([4.0]))
        npv = _npv(cfg_low_vol)
        assert np.isfinite(npv)
        assert npv >= -1e-3

    def test_single_reset_bermudan_matches_prepare_bermudan_final_maturity(self):
        swap = _prepared(_make_bermudan())
        assert swap.final_maturity == pytest.approx(swap.fixed_times[-1])


class TestMidPeriodBermudanExercise:
    """A Bermudan exercise date inside an accrual period enters each leg from its own next
    accrual start, as ORE does ("bermudan exercise implies that we always exercise into
    whole periods", `buildCashflowInfo`). The value is pinned against ORE in
    tests/test_ore_lgm_parity.py; these pin the mechanism."""

    def test_a_bermudan_coupon_belongs_only_until_its_accrual_start(self):
        swap = _prepared(_make_bermudan())
        assert np.array_equal(swap.fixed_belongs_until, swap.fixed_start_times)
        assert np.array_equal(swap.float_belongs_until, swap.float_start_times)

    def test_mid_period_exercise_does_not_enter_the_coupon_in_progress(self):
        from engine.instruments.bermudan_swaption import _build_grid_schedule
        starts = exercisable_dates(_make_bermudan()[0])
        swap = _prepared(_make_bermudan(exercise_dates=[starts[1] + 90]))
        t = float(swap.exercise_times[0])
        schedule = _build_grid_schedule(swap)
        row = int(np.nonzero(schedule.times == t)[0][0])
        num_fixed = len(swap.fixed_times)
        in_progress = int(np.nonzero((swap.fixed_start_times < t) & (swap.fixed_end_times > t))[0][0])
        acted_on = (schedule.add_pv | schedule.start_cache | schedule.from_cache | schedule.non_cached)[row]
        assert not acted_on[in_progress]
        assert acted_on[in_progress + 1]  # the next whole fixed period is entered
        assert not np.any(acted_on[num_fixed:][swap.float_start_times < t])

    def test_mid_coupon_exercise_still_prices_finite_and_nonnegative(self):
        cfg = _make_bermudan(exercise_dates=_in_years([1.25]))
        npv = _npv(cfg)
        assert np.isfinite(npv)
        assert npv >= 0.0

    @pytest.mark.parametrize("payer,fixed_rate", [
        (True, 0.02), (True, 0.03), (True, 0.04),
        (False, 0.02), (False, 0.03), (False, 0.04),
    ])
    def test_the_value_difference_follows_the_trade_direction(self, payer, fixed_rate):
        """Exercising a day into the period enters the annual fixed leg a whole period later
        but the semi-annual floating leg only one half-year later, so one fixed coupon drops
        out. For a payer that coupon was a payment, so the value rises; for a receiver it
        falls. This is ORE's behaviour (once mistaken for an error to prorate away; I-06)."""
        reset = exercisable_dates(_make_bermudan(payer=payer, fixed_rate=fixed_rate)[0])[2]

        def price(exercise_date):
            return _npv(_make_bermudan(
                payer=payer, fixed_rate=fixed_rate, sigma=0.005, exercise_dates=[exercise_date]))

        aligned, next_day = price(reset), price(reset + 1)
        if payer:
            assert next_day > aligned
        else:
            assert next_day < aligned


class TestStateGridAndScheduleEdgeCases:
    """One, two and many exercise dates, and grid resolution and width."""

    def test_single_exercise_date_prices_finite(self):
        cfg = _make_bermudan(exercise_dates=_in_years([2.5]))
        npv = _npv(cfg)
        assert np.isfinite(npv)
        assert npv >= 0.0

    def test_two_exercise_dates_at_least_as_valuable_as_either_alone(self):
        v1 = _npv(_make_bermudan(exercise_dates=_in_years([1.5])))
        v2 = _npv(_make_bermudan(exercise_dates=_in_years([3.5])))
        v_both = _npv(_make_bermudan(exercise_dates=_in_years([1.5, 3.5])))
        assert v_both >= max(v1, v2) - 1e-6

    def test_dense_monthly_schedule_over_long_tenor_prices_finite_and_consistent(self):
        # ~Monthly exercise over 9Y on a 10Y swap: many grid times.
        dense_times = [round(i / 12.0, 6) for i in range(1, 12 * 9)]
        cfg = _make_bermudan(exercise_dates=_in_years(dense_times), swap_tenor="10Y", n_per_std=32, std_devs=6.0)
        npv_dense = _npv(cfg)
        assert np.isfinite(npv_dense)
        assert npv_dense >= 0.0
        # Monotonicity against a sparse subset of the same dates.
        sparse_cfg = _make_bermudan(exercise_dates=_in_years([dense_times[0], dense_times[-1]]),
                                     swap_tenor="10Y", n_per_std=32, std_devs=6.0)
        npv_sparse = _npv(sparse_cfg)
        assert npv_dense >= npv_sparse - 1e-6

    @pytest.mark.slow
    def test_n_per_std_convergence_is_monotone_and_shrinking(self):
        # Successive n_per_std refinements move the price by shrinking amounts toward the
        # finest grid's value.
        ns = [8, 16, 32, 64, 128, 256]
        prices = [_npv(_make_bermudan(n_per_std=n, std_devs=6.0)) for n in ns]
        finest = prices[-1]
        errors = [abs(p - finest) for p in prices[:-1]]
        # Each refinement does not increase the error against the finest grid (small slack).
        for e_coarser, e_finer in zip(errors, errors[1:]):
            assert e_finer <= e_coarser + 1e-6

    def test_std_devs_too_small_understates_or_matches_wider_grid(self):
        # A narrow grid clips the tails; widening std_devs at fixed resolution should not
        # lower the price appreciably, and converges.
        narrow = _npv(_make_bermudan(n_per_std=48, std_devs=2.0))
        medium = _npv(_make_bermudan(n_per_std=48, std_devs=5.0))
        wide = _npv(_make_bermudan(n_per_std=48, std_devs=9.0))
        assert np.isfinite(narrow) and np.isfinite(medium) and np.isfinite(wide)
        assert abs(wide - medium) <= abs(medium - narrow) + 1e-6

    def test_near_zero_mean_reversion_prices_finite_and_consistent(self):
        # a -> 0 is a 0/0 in H(t); it stays finite and close to a small non-zero a.
        v_tiny = _npv(_make_bermudan(a=1e-6))
        v_small = _npv(_make_bermudan(a=1e-4))
        v_normal = _npv(_make_bermudan(a=0.03))
        assert np.isfinite(v_tiny) and np.isfinite(v_small) and np.isfinite(v_normal)
        assert v_tiny == pytest.approx(v_small, rel=1e-2)
        assert v_tiny > 0.0

    def test_high_volatility_prices_finite_and_increasing(self):
        # A large sigma stresses the grid's absolute span: no blow-up, value still rising.
        vols = [0.02, 0.05, 0.10, 0.20]
        prices = [
            _npv(_make_bermudan(sigma=s, std_devs=9.0, n_per_std=96))
            for s in vols
        ]
        assert all(np.isfinite(p) for p in prices)
        for p_lo, p_hi in zip(prices, prices[1:]):
            assert p_hi > p_lo


class TestConvergenceToAmericanAcrossConfigs:
    """A denser exercise schedule is worth at least a sparser one, across several
    underlyings."""

    @pytest.mark.slow
    @pytest.mark.parametrize("swap_tenor,fixed_rate,payer", [
        ("5Y", 0.03, True),
        ("5Y", 0.03, False),
        ("10Y", 0.02, True),
        ("2Y", 0.04, False),
    ])
    def test_dense_schedule_worth_at_least_sparse_schedule(self, swap_tenor, fixed_rate, payer):
        tenor_years = float(swap_tenor[:-1])
        sparse = [1.0] if tenor_years <= 2.0 else [1.0, round(tenor_years - 1.0, 2)]
        dense = [round(i * 0.25, 4) for i in range(1, int(4 * (tenor_years - 0.25)))]
        v_sparse = _npv(
            _make_bermudan(swap_tenor=swap_tenor, fixed_rate=fixed_rate, payer=payer, exercise_dates=_in_years(sparse))
        )
        v_dense = _npv(
            _make_bermudan(swap_tenor=swap_tenor, fixed_rate=fixed_rate, payer=payer, exercise_dates=_in_years(dense))
        )
        assert v_dense >= v_sparse - 1e-6


class TestDegenerateSingleExerciseCases:
    """Single-exercise Bermudans: against the European pricer, and deep OTM/ITM."""

    def test_single_exercise_vs_independent_european_pricer_same_order_of_magnitude(self):
        # The European engine is Jamshidian on a Hull-White model and this is the LGM with a
        # Hagan volatility, different models for t > 0 with the same (a, sigma), so only a
        # loose check: both finite, positive and within 2x of each other. (The tight
        # single-exercise check is TestSingleExerciseMatchesDirectIntegration.)
        euro_cfg = SwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.030, payer=True, swap_tenor="5Y",
            forward_start=ORE.Period(3, ORE.Years), evaluation_date=EVAL_DATE, trade_id="european",
        )
        curve = HwZeroCurve(pillar_times=jnp.asarray(FLAT_CURVE.times), pillar_rates=jnp.asarray(FLAT_CURVE.rates))
        euro_npv = float(jamshidian_npv(european_terms(euro_cfg, EVAL_DATE), JamshidianEngineConfig(0.03, 0.02),
                                        curve, 0.0))

        # The same underlying: the European's forward-starting 5Y swap, exercised on its date.
        berm_npv = _npv(_make_bermudan(exercise_dates=[euro_cfg.exercise_date], swap_tenor=None,
                                       effective_date=euro_cfg.effective_date, maturity_date=euro_cfg.maturity_date,
                                       n_per_std=192, std_devs=9.0))

        assert np.isfinite(euro_npv) and euro_npv > 0.0
        assert np.isfinite(berm_npv) and berm_npv > 0.0
        ratio = berm_npv / euro_npv
        assert 0.5 < ratio < 2.0

    def test_deeply_otm_single_exercise_is_near_zero(self):
        cfg = _make_bermudan(fixed_rate=0.30, payer=True, exercise_dates=_in_years([3.0]))
        npv = _npv(cfg)
        assert np.isfinite(npv)
        assert npv == pytest.approx(0.0, abs=1.0)

    def test_deeply_itm_single_exercise_approximates_discounted_intrinsic(self):
        # Exercise nearly certain: close to the swap's discounted value at exercise on
        # today's curve (a loose bound; some time value remains).
        cfg = _make_bermudan(fixed_rate=0.001, payer=True, exercise_dates=_in_years([3.0]))
        npv = _npv(cfg)
        swap = _prepared(cfg)
        from engine.instruments.bermudan_swaption import _cashflow_values_at_nodes, _zero_curve_of
        curve = _zero_curve_of(swap)
        t = float(swap.exercise_times[0])
        values = np.asarray(_cashflow_values_at_nodes(swap, curve, jnp.array([0.0]), jnp.array(t))[0])
        belongs = np.concatenate([swap.fixed_belongs_until, swap.float_belongs_until]) >= t
        intrinsic_at_exercise = float(np.sum(values[belongs]))
        discounted_intrinsic = intrinsic_at_exercise * np.exp(-0.03 * t)
        assert np.isfinite(npv)
        assert npv == pytest.approx(discounted_intrinsic, rel=0.1)

    def test_deeply_otm_receiver_is_near_zero(self):
        # Deep OTM for a receiver: a very low (negative) fixed rate against the 3% curve.
        cfg = _make_bermudan(fixed_rate=-0.10, payer=False, exercise_dates=_in_years([3.0]))
        npv = _npv(cfg)
        assert np.isfinite(npv)
        assert npv == pytest.approx(0.0, abs=1.0)


class TestPayerReceiver:
    """Payer/receiver sanity."""

    def test_payer_and_receiver_both_positive_and_comparable_near_atm(self):
        # Not a symmetry claim: both positive and within a broad band of each other.
        payer = _npv(_make_bermudan(payer=True, fixed_rate=0.03, exercise_dates=_in_years([3.0])))
        receiver = _npv(_make_bermudan(payer=False, fixed_rate=0.03, exercise_dates=_in_years([3.0])))
        assert payer > 0.0 and receiver > 0.0
        assert 0.5 < payer / receiver < 2.0

class TestBermudanSwaptionConfigValidation:
    """`BermudanSwaptionConfig.__post_init__` rejects non-finite notional/fixed_rate,
    unparseable tenors, and empty, unsorted or non-date exercise_dates. Zero notional is
    valid. The model's volatility is the engine configuration's, validated there
    (tests/test_run_config.py)."""

    def test_nan_notional_rejected(self):
        with pytest.raises(ValueError, match="notional"):
            _make_bermudan(notional=float("nan"))

    def test_inf_fixed_rate_rejected(self):
        with pytest.raises(ValueError, match="fixed_rate"):
            _make_bermudan(fixed_rate=float("inf"))

    def test_unparseable_swap_tenor_rejected(self):
        with pytest.raises(ValueError, match="swap_tenor"):
            _make_bermudan(swap_tenor="not-a-tenor")

    def test_empty_exercise_dates_rejected(self):
        with pytest.raises(ValueError, match="exercise_dates"):
            _make_bermudan(exercise_dates=[])

    def test_unsorted_exercise_dates_rejected(self):
        with pytest.raises(ValueError, match="exercise_dates"):
            _make_bermudan(exercise_dates=_in_years([2.0, 1.0, 3.0]))

    def test_year_fraction_exercise_rejected(self):
        """Exercise is given by date, as in ORE; a year fraction once dropped a coupon when
        rounded (I-29)."""
        with pytest.raises(TypeError, match="exercise_dates"):
            _make_bermudan(exercise_dates=[1.0, 2.0])

    def test_exercise_dates_on_or_before_the_evaluation_date_are_not_opportunities(self):
        """ORE: `if (d > refDate) optionTimes.insert(...)`."""
        cfg = _make_bermudan(exercise_dates=[EVAL_DATE] + _in_years([1.0, 2.0]))
        assert _prepared(cfg).exercise_times.tolist() == pytest.approx([1.0, 2.0])

    def test_valid_config_constructs_without_error(self):
        _make_bermudan()  # must not raise
