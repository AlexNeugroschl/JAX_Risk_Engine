"""
Tests for engine.instruments.american_swaption.

An American differs from a Bermudan in exactly two places, both taken from
`NumericLgmMultiLegOptionEngineBase` and both pinned here:

  * its option times -- ORE's uniform grid over the window, with a
    TRUNCATED step count (`AmericanSwaptionConfig.option_times`), on the engine's
    `ExerciseTimeStepsPerYear`;
  * which coupons an exercise enters -- a coupon belongs until its accrual
    END and is credited `couponRatio(t)` (`ExerciseStyle.AMERICAN`).

Everything else is the shared backward induction (tests/test_bermudan_swaption.py), priced here
with an explicit model (`tests.support.lgm_engine`). The prices themselves are pinned against
ORE's own engine in tests/test_ore_lgm_parity.py.
"""
import numpy as np
import ORE
import pytest

from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig, _build_grid_schedule, exercisable_dates
from engine.market import ZeroCurveConfig
from date_helpers import in_years
from tests.support.lgm_engine import grid_npv, prepared

FLAT_CURVE = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[0.03] * 6)
EVAL_DATE = ORE.Date(30, 7, 2026)
MODEL = dict(a=0.03, sigma=0.02, curve=FLAT_CURVE, n_per_std=96, std_devs=7.0, steps_per_year=24)


def _make_american(**overrides) -> AmericanSwaptionConfig:
    defaults = dict(
        notional=1_000_000.0, fixed_rate=0.030, payer=True,
        first_exercise_date=in_years(EVAL_DATE, 1.0), last_exercise_date=in_years(EVAL_DATE, 4.0),
        swap_tenor="5Y", evaluation_date=EVAL_DATE, trade_id="american",
    )
    defaults.update(overrides)
    return AmericanSwaptionConfig(**defaults)


def _make_bermudan(**overrides) -> BermudanSwaptionConfig:
    defaults = dict(
        notional=1_000_000.0, fixed_rate=0.030, payer=True,
        exercise_dates=in_years(EVAL_DATE, [1.0, 2.0, 3.0, 4.0]), swap_tenor="5Y",
        evaluation_date=EVAL_DATE, trade_id="bermudan",
    )
    defaults.update(overrides)
    return BermudanSwaptionConfig(**defaults)


def _npv(cfg, **model) -> float:
    return grid_npv(cfg, **{**MODEL, **model})


def _prepared(cfg, **model):
    return prepared(cfg, **{**MODEL, **model})


class TestOptionTimes:
    """ORE's American grid (`calculate()`, lines 494-505)."""

    def test_window_endpoints_are_both_option_times(self):
        times = _make_american().option_times(4)
        assert times[0] == pytest.approx(1.0)
        assert times[-1] == pytest.approx(4.0)

    def test_step_count_on_a_whole_number_of_steps(self):
        # (4 - 1) * 4 = 12 steps -> 13 option times, endpoints inclusive.
        assert len(_make_american().option_times(4)) == 13

    def test_step_count_is_truncated_not_rounded(self):
        """`static_cast<Size>((t2 - t1) * stepsPerYear)`: 1105 days at 24 per
        year is 72.66 steps, which ORE truncates to 72 (rounding would give
        73)."""
        first = in_years(EVAL_DATE, 1.0)
        cfg = _make_american(first_exercise_date=first, last_exercise_date=first + 1105)
        assert len(cfg.option_times(24)) == 72 + 1

    def test_zero_width_window_is_a_single_option_time_at_any_resolution(self):
        """t2 == t1 gives `steps = max(1, 0) = 1`, and both grid points are
        the same time."""
        date = in_years(EVAL_DATE, 2.0)
        cfg = _make_american(first_exercise_date=date, last_exercise_date=date)
        for steps_per_year in (1, 24, 500):
            assert cfg.option_times(steps_per_year) == [pytest.approx(2.0)]

    def test_a_window_opening_in_the_past_starts_tomorrow(self):
        """An American window already open is exercisable from the day after
        the evaluation date, never on it: ORE's `ExerciseBuilder` moves the
        first date to `max(today + 1, first)` before the engine sees it
        (I-35). Checked against ORE's own engine in
        tests/test_trade_dates.py."""
        cfg = _make_american(first_exercise_date=EVAL_DATE - 30, last_exercise_date=in_years(EVAL_DATE, 1.0))
        assert cfg.option_times(24)[0] == 1.0 / 365.0

    def test_the_step_count_comes_from_the_engine(self):
        """ORE's `ExerciseTimeStepsPerYear` is an engine parameter (`pricingengine.xml`), so the
        grid is the engine configuration's: the same trade on another engine setting has
        another grid."""
        cfg = _make_american()
        assert len(_prepared(cfg, steps_per_year=4).exercise_times) == 13
        assert len(_prepared(cfg, steps_per_year=2).exercise_times) == 7

    def test_a_non_positive_step_count_is_refused(self):
        with pytest.raises(ValueError, match="exercise_time_steps_per_year"):
            _make_american().option_times(0)


class TestBrokenPeriodExercise:
    """An American exercise can land inside an accrual period and credits the
    coupon in progress its unexpired share (`buildCashflowInfo`, lines
    107-109) -- the difference from a Bermudan that I-06 was about."""

    def test_an_american_coupon_belongs_until_its_accrual_end(self):
        swap = _prepared(_make_american())
        assert np.array_equal(swap.fixed_belongs_until, swap.fixed_end_times)
        assert np.array_equal(swap.float_belongs_until, swap.float_end_times)

    def test_the_coupon_in_progress_is_credited_its_unexpired_share(self):
        swap = _prepared(_make_american(), steps_per_year=12)
        schedule = _build_grid_schedule(swap)
        t = float(swap.exercise_times[3])
        row = int(np.nonzero(schedule.times == t)[0][0])
        i = int(np.nonzero((swap.fixed_start_times < t) & (swap.fixed_end_times > t))[0][0])
        expected = (swap.fixed_end_times[i] - t) / (swap.fixed_end_times[i] - swap.fixed_start_times[i])
        assert 0.0 < expected < 1.0
        assert schedule.coupon_ratio[row, i] == pytest.approx(expected, rel=1e-15)
        assert schedule.from_cache[row, i]

    def test_single_exercise_on_an_accrual_start_equals_the_bermudan(self):
        """On an accrual start the coupon ratio is 1 for the coupon starting
        and 0 for the one ending, so the two exercise styles enter the same
        swap and must agree to rounding."""
        date = exercisable_dates(_make_bermudan())[2]
        american = _make_american(first_exercise_date=date, last_exercise_date=date)
        bermudan = _make_bermudan(exercise_dates=[date])
        assert _npv(american) == pytest.approx(_npv(bermudan), rel=1e-12)

    @pytest.mark.parametrize("payer", [True, False], ids=["payer", "receiver"])
    def test_single_mid_period_exercise_differs_from_the_bermudan(self, payer):
        """Mid-period, the Bermudan skips to the next whole periods while the
        American enters the broken one, so the two must differ."""
        date = exercisable_dates(_make_bermudan())[2] + 100
        american = _npv(_make_american(payer=payer, first_exercise_date=date, last_exercise_date=date))
        bermudan = _npv(_make_bermudan(payer=payer, exercise_dates=[date]))
        assert abs(american - bermudan) / bermudan > 1e-3


class TestMoreOpportunitiesNeverLoseValue:
    """Doubling the steps per year over a whole-year window doubles the
    step count exactly, so each grid is a superset of the coarser one and the
    value cannot fall -- a model-independent bound."""

    @pytest.mark.slow
    @pytest.mark.parametrize("swap_tenor,fixed_rate,payer,last_years", [
        ("5Y", 0.03, True, 4.0),
        ("5Y", 0.03, False, 4.0),
        ("2Y", 0.03, True, 1.9),
    ])
    def test_finer_grid_never_decreases_value(self, swap_tenor, fixed_rate, payer, last_years):
        cfg = _make_american(swap_tenor=swap_tenor, fixed_rate=fixed_rate, payer=payer,
                             last_exercise_date=in_years(EVAL_DATE, last_years))
        npvs = [_npv(cfg, steps_per_year=k) for k in (2, 4, 8)]
        assert npvs[1] >= npvs[0] - 1e-6
        assert npvs[2] >= npvs[1] - 1e-6

    @pytest.mark.slow
    def test_very_dense_grid_prices_finite_and_above_the_coarsest(self):
        cfg = _make_american(first_exercise_date=in_years(EVAL_DATE, 1.0), last_exercise_date=in_years(EVAL_DATE, 1.5))
        assert len(cfg.option_times(1000)) > 100
        dense_npv, coarse_npv = _npv(cfg, steps_per_year=1000), _npv(cfg, steps_per_year=1)
        assert np.isfinite(dense_npv) and dense_npv >= coarse_npv - 1e-6


class TestAmericanSwaptionConfigValidation:
    """AmericanSwaptionConfig.__post_init__ -- rejects non-finite notional/fixed_rate, an
    unparseable swap_tenor, a window that ends before it starts and non-date window bounds.
    The grid resolution is the engine's (see `TestOptionTimes`)."""

    def test_nan_notional_rejected(self):
        with pytest.raises(ValueError, match="notional"):
            _make_american(notional=float("nan"))

    def test_inf_fixed_rate_rejected(self):
        with pytest.raises(ValueError, match="fixed_rate"):
            _make_american(fixed_rate=float("inf"))

    def test_unparseable_swap_tenor_rejected(self):
        with pytest.raises(ValueError, match="swap_tenor"):
            _make_american(swap_tenor="bogus")

    def test_window_ending_before_it_starts_rejected(self):
        with pytest.raises(ValueError, match="first_exercise_date"):
            _make_american(first_exercise_date=in_years(EVAL_DATE, 4.0), last_exercise_date=in_years(EVAL_DATE, 1.0))

    def test_year_fraction_window_rejected(self):
        with pytest.raises(TypeError, match="first_exercise_date"):
            _make_american(first_exercise_date=1.0)

    def test_zero_width_window_is_valid(self):
        date = in_years(EVAL_DATE, 2.0)
        _make_american(first_exercise_date=date, last_exercise_date=date)  # must not raise

    def test_valid_config_constructs_without_error(self):
        _make_american()  # must not raise
