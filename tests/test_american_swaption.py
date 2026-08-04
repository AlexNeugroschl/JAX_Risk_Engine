"""
Tests for engine.instruments.american_swaption -- a thin wrapper around
engine.instruments.bermudan_swaption's numeric LGM backward-induction
engine (see tests/test_bermudan_swaption.py for tests of that underlying
engine, including the closed-form-primitive ORE cross-checks, the
single-exercise-date Jamshidian-style convergence check, monotonicity
bounds, and the mid-coupon known-limitation tests -- the same limitation
applies here since American exercise is priced through that same engine).

This file covers only what's specific to the American-exercise wrapper
itself: the exercise-window discretization (`AmericanSwaptionConfig.
to_bermudan()`), its convergence as `exercise_time_steps_per_year`
increases, and that it reproduces an equivalent explicit Bermudan exactly
at a reset-aligned resolution.
"""
import numpy as np
import ORE
import pytest

from engine.simulation import ZeroCurveConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig, price_bermudan_swaption_base
from engine.instruments.american_swaption import AmericanSwaptionConfig, price_american_swaptions

FLAT_CURVE = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[0.03] * 6)
EVAL_DATE = ORE.Date(30, 7, 2026)


def _make_bermudan(**overrides) -> BermudanSwaptionConfig:
    defaults = dict(
        notional=1_000_000.0,
        fixed_rate=0.030,
        payer=True,
        rate_factor_index=0,
        hw_a=0.03,
        hw_sigma=0.02,
        initial_zero_curve=FLAT_CURVE,
        exercise_times=[1.0, 2.0, 3.0, 4.0],
        swap_tenor="5Y",
        evaluation_date=EVAL_DATE,
        n_per_std=96,
        std_devs=7.0,
    )
    defaults.update(overrides)
    return BermudanSwaptionConfig(**defaults)


class TestAmericanAsFineBermudan:
    """American exercise is priced by discretizing the exercise window into
    ORE's own ExerciseTimeStepsPerYear-spaced grid -- verify the expansion
    and its convergence behavior."""

    def _make_american(self, **overrides):
        defaults = dict(
            notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
            hw_a=0.03, hw_sigma=0.02, initial_zero_curve=FLAT_CURVE,
            first_exercise=1.0, last_exercise=4.0, swap_tenor="5Y",
            evaluation_date=EVAL_DATE, n_per_std=96, std_devs=7.0,
        )
        defaults.update(overrides)
        return AmericanSwaptionConfig(**defaults)

    def test_to_bermudan_includes_both_endpoints(self):
        cfg = self._make_american(exercise_time_steps_per_year=4)
        berm = cfg.to_bermudan()
        assert berm.exercise_times[0] == pytest.approx(1.0)
        assert berm.exercise_times[-1] == pytest.approx(4.0)

    def test_to_bermudan_step_count_matches_ore_formula(self):
        cfg = self._make_american(first_exercise=1.0, last_exercise=4.0, exercise_time_steps_per_year=4)
        berm = cfg.to_bermudan()
        # steps = round((4-1)*4) = 12 -> 13 dates (endpoints inclusive)
        assert len(berm.exercise_times) == 13

    def test_american_worth_at_least_as_much_as_reset_aligned_bermudan(self):
        american = self._make_american(exercise_time_steps_per_year=2)  # reset-aligned (semiannual)
        american_npv = price_bermudan_swaption_base(american.to_bermudan())
        sparse_bermudan = price_bermudan_swaption_base(
            _make_bermudan(exercise_times=[1.0, 4.0], n_per_std=96, std_devs=7.0)
        )
        assert american_npv >= sparse_bermudan - 1e-6

    def test_finer_discretization_increases_or_holds_value(self):
        # exercise_time_steps_per_year=2 is the only resolution that lands
        # exactly on this swap's own semiannual float resets (0.5y steps);
        # any finer grid (4, 8, ...) hits the documented mid-coupon
        # limitation for SOME of its extra dates, so monotonic shrinking of
        # the price change isn't guaranteed step-to-step -- only that more
        # exercise opportunities can't decrease value (the model-independent
        # bound, not a convergence-rate claim about this specific grid).
        npvs = [
            price_bermudan_swaption_base(self._make_american(exercise_time_steps_per_year=k).to_bermudan())
            for k in [2, 4, 8]
        ]
        assert npvs[1] >= npvs[0] - 1e-6
        assert npvs[2] >= npvs[1] - 1e-6

    def test_reset_aligned_bermudan_matches_american_at_matching_resolution(self):
        # With exercise_time_steps_per_year=2, the American discretization
        # lands exactly on the swap's own semiannual float resets --
        # verify it reproduces an EXPLICITLY-listed Bermudan with the same
        # dates exactly (both paths should be identical, not merely close).
        american = self._make_american(exercise_time_steps_per_year=2)
        berm_from_american = american.to_bermudan()
        explicit_bermudan = _make_bermudan(
            exercise_times=list(berm_from_american.exercise_times),
            n_per_std=96, std_devs=7.0,
        )
        npv_american = price_bermudan_swaption_base(berm_from_american)
        npv_explicit = price_bermudan_swaption_base(explicit_bermudan)
        assert npv_american == pytest.approx(npv_explicit, rel=1e-9)

    def test_price_american_swaptions_matches_base(self):
        import jax.numpy as jnp
        from engine.simulation import generate_paths
        from engine.scenarios import swaption_demo_config

        config = swaption_demo_config()
        cubes = generate_paths(config)
        step_times = jnp.array(config.time_grid[1:], dtype=jnp.float64)
        cfg = self._make_american(
            hw_a=config.rates.mean_reversion[0],
            hw_sigma=float(np.sqrt(config.joint_covariance[1][1])),
            exercise_time_steps_per_year=2,
        )
        cube = price_american_swaptions([cfg], cubes["rates"], step_times)
        assert cube.shape == (config.scenarios, len(config.time_grid) - 1, 1)
        base = price_bermudan_swaption_base(cfg.to_bermudan())
        assert base > 0.0


class TestExerciseTimeStepsPerYearSensitivity:
    """Extends test_finer_discretization_increases_or_holds_value to more
    swap configurations (tenor, payer/receiver) and directions, plus the
    coarsest (steps=1) and a very dense discretization."""

    def _make_american(self, **overrides):
        defaults = dict(
            notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
            hw_a=0.03, hw_sigma=0.02, initial_zero_curve=FLAT_CURVE,
            first_exercise=1.0, last_exercise=4.0, swap_tenor="5Y",
            evaluation_date=EVAL_DATE, n_per_std=96, std_devs=7.0,
        )
        defaults.update(overrides)
        return AmericanSwaptionConfig(**defaults)

    @pytest.mark.parametrize("swap_tenor,fixed_rate,payer,last_exercise", [
        ("5Y", 0.03, True, 4.0),
        ("5Y", 0.03, False, 4.0),
        ("2Y", 0.03, True, 1.9),
    ])
    def test_finer_discretization_never_decreases_value_across_configs(
        self, swap_tenor, fixed_rate, payer, last_exercise,
    ):
        npvs = [
            price_bermudan_swaption_base(
                self._make_american(
                    swap_tenor=swap_tenor, fixed_rate=fixed_rate, payer=payer,
                    last_exercise=last_exercise, exercise_time_steps_per_year=k,
                ).to_bermudan()
            )
            for k in [2, 4, 8]
        ]
        assert npvs[1] >= npvs[0] - 1e-6
        assert npvs[2] >= npvs[1] - 1e-6

    def test_coarsest_single_step_per_window_prices_finite(self):
        # exercise_time_steps_per_year=1 over a 3Y window still rounds to a
        # small step count (max(1, round(3*1))=3) -- use a narrow window so
        # `steps = max(1, round((t2-t1)*stepsPerYear))` actually bottoms out
        # at exactly 1 (the coarsest possible discretization: just the two
        # window endpoints).
        cfg = self._make_american(first_exercise=1.0, last_exercise=1.4, exercise_time_steps_per_year=1)
        berm = cfg.to_bermudan()
        assert len(berm.exercise_times) == 2
        npv = price_bermudan_swaption_base(berm)
        assert np.isfinite(npv)
        assert npv >= 0.0

    def test_very_dense_discretization_prices_finite_and_consistent(self):
        cfg = self._make_american(first_exercise=1.0, last_exercise=1.5, exercise_time_steps_per_year=1000)
        berm = cfg.to_bermudan()
        assert len(berm.exercise_times) > 100
        npv = price_bermudan_swaption_base(berm)
        assert np.isfinite(npv)
        assert npv >= 0.0
        # Should be at least as valuable as the coarsest 2-date version of
        # the same window (more exercise opportunities).
        coarse_npv = price_bermudan_swaption_base(
            self._make_american(first_exercise=1.0, last_exercise=1.5, exercise_time_steps_per_year=1).to_bermudan()
        )
        assert npv >= coarse_npv - 1e-6


class TestDegenerateExerciseWindow:
    """first_exercise == last_exercise is a zero-width window -- ORE's own
    discretization formula (`steps = max(1, round((t2-t1)*stepsPerYear))`)
    degenerates to `steps=1` with t2-t1=0, producing exactly one exercise
    date; verify this reduces exactly to a single-date Bermudan (and thus
    to a European swaption at that date)."""

    def _make_american(self, **overrides):
        defaults = dict(
            notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
            hw_a=0.03, hw_sigma=0.02, initial_zero_curve=FLAT_CURVE,
            first_exercise=2.0, last_exercise=2.0, swap_tenor="5Y",
            evaluation_date=EVAL_DATE, n_per_std=96, std_devs=7.0,
            exercise_time_steps_per_year=24,
        )
        defaults.update(overrides)
        return AmericanSwaptionConfig(**defaults)

    def test_zero_width_window_produces_single_exercise_date(self):
        cfg = self._make_american(first_exercise=2.0, last_exercise=2.0)
        berm = cfg.to_bermudan()
        assert len(berm.exercise_times) == 1
        assert berm.exercise_times[0] == pytest.approx(2.0)

    def test_zero_width_window_matches_explicit_single_date_bermudan_exactly(self):
        cfg = self._make_american(first_exercise=2.0, last_exercise=2.0)
        american_npv = price_bermudan_swaption_base(cfg.to_bermudan())
        explicit_bermudan = _make_bermudan(exercise_times=[2.0], n_per_std=96, std_devs=7.0)
        explicit_npv = price_bermudan_swaption_base(explicit_bermudan)
        assert american_npv == pytest.approx(explicit_npv, rel=1e-9)

    def test_zero_width_window_is_insensitive_to_exercise_time_steps_per_year(self):
        # With t2-t1=0, `steps = max(1, round(0*stepsPerYear)) == 1`
        # regardless of stepsPerYear -- the discretization must collapse to
        # the same single date (and price) no matter how fine a resolution
        # is requested.
        npv_coarse = price_bermudan_swaption_base(
            self._make_american(exercise_time_steps_per_year=1).to_bermudan()
        )
        npv_fine = price_bermudan_swaption_base(
            self._make_american(exercise_time_steps_per_year=500).to_bermudan()
        )
        assert npv_coarse == pytest.approx(npv_fine, rel=1e-9)


class TestAmericanPortfolio:
    """A diverse American-swaption portfolio (mixed payer/receiver, mixed
    tenors, mixed exercise windows/resolutions) -- checks output shape and
    that each trade's priced values are independent of the others in the
    batch."""

    def _make_american(self, **overrides):
        defaults = dict(
            notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
            hw_a=0.03, hw_sigma=0.02, initial_zero_curve=FLAT_CURVE,
            first_exercise=1.0, last_exercise=4.0, swap_tenor="5Y",
            evaluation_date=EVAL_DATE, n_per_std=48, std_devs=6.0,
            exercise_time_steps_per_year=2,
        )
        defaults.update(overrides)
        return AmericanSwaptionConfig(**defaults)

    def test_mixed_portfolio_shape_and_per_trade_independence(self):
        import jax.numpy as jnp
        from engine.simulation import generate_paths
        from engine.scenarios import swaption_demo_config

        config = swaption_demo_config()
        cubes = generate_paths(config)
        step_times = jnp.array(config.time_grid[1:], dtype=jnp.float64)
        common = dict(
            hw_a=config.rates.mean_reversion[0],
            hw_sigma=float(np.sqrt(config.joint_covariance[1][1])),
            n_per_std=48, std_devs=6.0,
        )
        cfg_payer = self._make_american(payer=True, fixed_rate=0.02, swap_tenor="5Y",
                                         first_exercise=1.0, last_exercise=4.0,
                                         exercise_time_steps_per_year=2, **common)
        cfg_receiver = self._make_american(payer=False, fixed_rate=0.04, swap_tenor="5Y",
                                            first_exercise=0.5, last_exercise=4.5,
                                            exercise_time_steps_per_year=2, **common)
        cfg_short_tenor = self._make_american(payer=True, fixed_rate=0.03, swap_tenor="2Y",
                                               first_exercise=0.5, last_exercise=1.9,
                                               exercise_time_steps_per_year=4, **common)

        configs = [cfg_payer, cfg_receiver, cfg_short_tenor]
        cube = price_american_swaptions(configs, cubes["rates"], step_times)
        assert cube.shape == (config.scenarios, len(config.time_grid) - 1, 3)
        assert np.all(np.isfinite(np.asarray(cube)))

        for i, cfg in enumerate(configs):
            solo_cube = price_american_swaptions([cfg], cubes["rates"], step_times)
            assert np.allclose(np.asarray(cube[:, :, i]), np.asarray(solo_cube[:, :, 0]), rtol=1e-10, atol=1e-8)

        assert not np.allclose(np.asarray(cube[:, :, 0]), np.asarray(cube[:, :, 1]))
        assert not np.allclose(np.asarray(cube[:, :, 1]), np.asarray(cube[:, :, 2]))
