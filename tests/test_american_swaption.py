"""
Tests for engine.instruments.american_swaption.

An American differs from a Bermudan in exactly two places, both taken from
`NumericLgmMultiLegOptionEngineBase` and both pinned here:

  * its option times -- ORE's uniform grid over the window, with a
    TRUNCATED step count (`AmericanSwaptionConfig.option_times`);
  * which coupons an exercise enters -- a coupon belongs until its accrual
    END and is credited `couponRatio(t)` (`ExerciseStyle.AMERICAN`).

Everything else is the shared backward induction (tests/test_bermudan_swaption.py).
The prices themselves are pinned against ORE's own engine in
tests/test_ore_lgm_parity.py.
"""
import numpy as np
import ORE
import pytest

from engine.simulation.market_model import ZeroCurveConfig
from engine.instruments.bermudan_swaption import (
    BermudanSwaptionConfig,
    _build_grid_schedule,
    exercisable_dates,
    prepare_bermudan,
    price_bermudan_swaption_base,
)
from engine.instruments.american_swaption import AmericanSwaptionConfig, price_american_swaptions
from date_helpers import in_years

FLAT_CURVE = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[0.03] * 6)
EVAL_DATE = ORE.Date(30, 7, 2026)


def _make_american(**overrides) -> AmericanSwaptionConfig:
    defaults = dict(
        notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
        hw_a=0.03, hw_sigma=0.02, initial_zero_curve=FLAT_CURVE,
        first_exercise_date=in_years(EVAL_DATE, 1.0), last_exercise_date=in_years(EVAL_DATE, 4.0),
        swap_tenor="5Y", evaluation_date=EVAL_DATE, n_per_std=96, std_devs=7.0,
    )
    defaults.update(overrides)
    return AmericanSwaptionConfig(**defaults)


def _make_bermudan(**overrides) -> BermudanSwaptionConfig:
    defaults = dict(
        notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
        hw_a=0.03, hw_sigma=0.02, initial_zero_curve=FLAT_CURVE,
        exercise_dates=in_years(EVAL_DATE, [1.0, 2.0, 3.0, 4.0]), swap_tenor="5Y",
        evaluation_date=EVAL_DATE, n_per_std=96, std_devs=7.0,
    )
    defaults.update(overrides)
    return BermudanSwaptionConfig(**defaults)


class TestOptionTimes:
    """ORE's American grid (`calculate()`, lines 494-505)."""

    def test_window_endpoints_are_both_option_times(self):
        times = _make_american(exercise_time_steps_per_year=4).option_times()
        assert times[0] == pytest.approx(1.0)
        assert times[-1] == pytest.approx(4.0)

    def test_step_count_on_a_whole_number_of_steps(self):
        # (4 - 1) * 4 = 12 steps -> 13 option times, endpoints inclusive.
        assert len(_make_american(exercise_time_steps_per_year=4).option_times()) == 13

    def test_step_count_is_truncated_not_rounded(self):
        """`static_cast<Size>((t2 - t1) * stepsPerYear)`: 1105 days at 24 per
        year is 72.66 steps, which ORE truncates to 72 (rounding would give
        73)."""
        first = in_years(EVAL_DATE, 1.0)
        cfg = _make_american(first_exercise_date=first, last_exercise_date=first + 1105,
                             exercise_time_steps_per_year=24)
        assert len(cfg.option_times()) == 72 + 1

    def test_zero_width_window_is_a_single_option_time_at_any_resolution(self):
        """t2 == t1 gives `steps = max(1, 0) = 1`, and both grid points are
        the same time."""
        date = in_years(EVAL_DATE, 2.0)
        for steps_per_year in (1, 24, 500):
            cfg = _make_american(first_exercise_date=date, last_exercise_date=date,
                                 exercise_time_steps_per_year=steps_per_year)
            assert cfg.option_times() == [pytest.approx(2.0)]

    def test_a_window_opening_in_the_past_starts_at_time_zero(self):
        """`t1 = max(0, t(first))`: unlike a Bermudan date, an American
        window already open is exercisable today."""
        cfg = _make_american(first_exercise_date=EVAL_DATE - 30, last_exercise_date=in_years(EVAL_DATE, 1.0))
        assert cfg.option_times()[0] == 0.0


class TestBrokenPeriodExercise:
    """An American exercise can land inside an accrual period and credits the
    coupon in progress its unexpired share (`buildCashflowInfo`, lines
    107-109) -- the difference from a Bermudan that I-06 was about."""

    def test_an_american_coupon_belongs_until_its_accrual_end(self):
        swap = prepare_bermudan(_make_american())
        assert np.array_equal(swap.fixed_belongs_until, swap.fixed_end_times)
        assert np.array_equal(swap.float_belongs_until, swap.float_end_times)

    def test_the_coupon_in_progress_is_credited_its_unexpired_share(self):
        swap = prepare_bermudan(_make_american(exercise_time_steps_per_year=12))
        schedule = _build_grid_schedule(swap, [])
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
        assert price_bermudan_swaption_base(american) == pytest.approx(
            price_bermudan_swaption_base(bermudan), rel=1e-12)

    @pytest.mark.parametrize("payer", [True, False], ids=["payer", "receiver"])
    def test_single_mid_period_exercise_differs_from_the_bermudan(self, payer):
        """Mid-period, the Bermudan skips to the next whole periods while the
        American enters the broken one, so the two must differ."""
        date = exercisable_dates(_make_bermudan())[2] + 100
        american = price_bermudan_swaption_base(
            _make_american(payer=payer, first_exercise_date=date, last_exercise_date=date))
        bermudan = price_bermudan_swaption_base(_make_bermudan(payer=payer, exercise_dates=[date]))
        assert abs(american - bermudan) / bermudan > 1e-3


class TestMoreOpportunitiesNeverLoseValue:
    """Doubling the steps per year over a whole-year window doubles the
    step count exactly, so each grid is a superset of the coarser one and the
    value cannot fall -- a model-independent bound."""

    @pytest.mark.parametrize("swap_tenor,fixed_rate,payer,last_years", [
        ("5Y", 0.03, True, 4.0),
        ("5Y", 0.03, False, 4.0),
        ("2Y", 0.03, True, 1.9),
    ])
    def test_finer_grid_never_decreases_value(self, swap_tenor, fixed_rate, payer, last_years):
        npvs = [
            price_bermudan_swaption_base(_make_american(
                swap_tenor=swap_tenor, fixed_rate=fixed_rate, payer=payer,
                last_exercise_date=in_years(EVAL_DATE, last_years), exercise_time_steps_per_year=k))
            for k in (2, 4, 8)
        ]
        assert npvs[1] >= npvs[0] - 1e-6
        assert npvs[2] >= npvs[1] - 1e-6

    def test_very_dense_grid_prices_finite_and_above_the_coarsest(self):
        window = dict(first_exercise_date=in_years(EVAL_DATE, 1.0), last_exercise_date=in_years(EVAL_DATE, 1.5))
        dense = _make_american(exercise_time_steps_per_year=1000, **window)
        assert len(dense.option_times()) > 100
        dense_npv = price_bermudan_swaption_base(dense)
        coarse_npv = price_bermudan_swaption_base(_make_american(exercise_time_steps_per_year=1, **window))
        assert np.isfinite(dense_npv) and dense_npv >= coarse_npv - 1e-6


class TestAmericanPortfolio:
    """A mixed American portfolio through the conditioning pricer: output
    shape and per-trade independence."""

    def test_mixed_portfolio_shape_and_per_trade_independence(self):
        import jax.numpy as jnp
        from engine.simulation.market_model import generate_paths
        from engine.simulation.demo_scenarios import swaption_demo_config

        config = swaption_demo_config()
        cubes = generate_paths(config)
        step_times = jnp.array(config.time_grid[1:], dtype=jnp.float64)
        common = dict(
            hw_a=config.rates.mean_reversion[0],
            hw_sigma=float(np.sqrt(config.joint_covariance[1][1])),
            n_per_std=48, std_devs=6.0,
        )
        window = lambda first, last: dict(  # noqa: E731
            first_exercise_date=in_years(EVAL_DATE, first), last_exercise_date=in_years(EVAL_DATE, last))
        configs = [
            _make_american(payer=True, fixed_rate=0.02, swap_tenor="5Y", exercise_time_steps_per_year=2,
                           **window(1.0, 4.0), **common),
            _make_american(payer=False, fixed_rate=0.04, swap_tenor="5Y", exercise_time_steps_per_year=2,
                           **window(0.5, 4.5), **common),
            _make_american(payer=True, fixed_rate=0.03, swap_tenor="2Y", exercise_time_steps_per_year=4,
                           **window(0.5, 1.9), **common),
        ]
        cube = price_american_swaptions(configs, cubes["rates"], step_times)
        assert cube.shape == (config.scenarios, len(config.time_grid) - 1, 3)
        assert np.all(np.isfinite(np.asarray(cube)))

        for i, cfg in enumerate(configs):
            solo_cube = price_american_swaptions([cfg], cubes["rates"], step_times)
            assert np.allclose(np.asarray(cube[:, :, i]), np.asarray(solo_cube[:, :, 0]), rtol=1e-10, atol=1e-8)

        assert not np.allclose(np.asarray(cube[:, :, 0]), np.asarray(cube[:, :, 1]))
        assert not np.allclose(np.asarray(cube[:, :, 1]), np.asarray(cube[:, :, 2]))


class TestAmericanSwaptionConfigValidation:
    """AmericanSwaptionConfig.__post_init__ -- rejects non-finite notional/
    fixed_rate/hw_sigma, an unparseable swap_tenor, a window that ends
    before it starts, non-date window bounds and a non-positive resolution."""

    def test_nan_notional_rejected(self):
        with pytest.raises(ValueError, match="notional"):
            _make_american(notional=float("nan"))

    def test_inf_fixed_rate_rejected(self):
        with pytest.raises(ValueError, match="fixed_rate"):
            _make_american(fixed_rate=float("inf"))

    def test_nan_hw_sigma_rejected(self):
        with pytest.raises(ValueError, match="hw_sigma"):
            _make_american(hw_sigma=float("nan"))

    def test_unparseable_swap_tenor_rejected(self):
        with pytest.raises(ValueError, match="swap_tenor"):
            _make_american(swap_tenor="bogus")

    def test_window_ending_before_it_starts_rejected(self):
        with pytest.raises(ValueError, match="first_exercise_date"):
            _make_american(first_exercise_date=in_years(EVAL_DATE, 4.0), last_exercise_date=in_years(EVAL_DATE, 1.0))

    def test_year_fraction_window_rejected(self):
        with pytest.raises(TypeError, match="first_exercise_date"):
            _make_american(first_exercise_date=1.0)

    def test_non_positive_resolution_rejected(self):
        with pytest.raises(ValueError, match="exercise_time_steps_per_year"):
            _make_american(exercise_time_steps_per_year=0)

    def test_zero_width_window_is_valid(self):
        date = in_years(EVAL_DATE, 2.0)
        _make_american(first_exercise_date=date, last_exercise_date=date)  # must not raise

    def test_valid_config_constructs_without_error(self):
        _make_american()  # must not raise
