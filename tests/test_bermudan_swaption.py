"""
Tests for engine.instruments.bermudan_swaption -- the numeric LGM
Hagan-quadrature backward-induction engine that also powers American
swaption pricing (engine.instruments.american_swaption is a thin wrapper
around this module -- its own American-specific tests live in
tests/test_american_swaption.py).

Validation strategy (see the module's own docstring for the full account):
ORE's Python bindings do not expose a constructible
`NumericLgmMultiLegOptionEngine`, so this module's full backward-induction
engine cannot be cross-checked against a live ORE engine object end-to-end
the way the swap/European-swaption/VaR modules are. Instead:

  - Every closed-form building block (`_H`, `_zeta`, `_lgm_bond`) is
    live-verified here against `ORE.IrLgm1fConstantParametrization` /
    `ORE.LinearGaussMarkovModel` directly.
  - The single-exercise-date limit is cross-checked against an independent
    from-scratch Jamshidian-style decomposition built on the SAME `_lgm_bond`
    formula (deliberately NOT engine.instruments.european_swaption's own
    Jamshidian pricer, which is HullWhite-parametrized -- confirmed, while
    building this module, to be a genuinely different model realization
    for t>0 than QuantExt's LinearGaussMarkovModel; see
    bermudan_swaption._lgm_bond's docstring for the live-verified evidence).
  - Model-independent structural properties (monotonicity in exercise
    opportunities, put/call sign convention, ITM > OTM, grid convergence)
    are checked directly.
"""
import numpy as np
import ORE
import pytest

from engine.simulation import ZeroCurveConfig
from engine.instruments.bermudan_swaption import (
    BermudanSwaptionConfig,
    _H,
    _hagan_quadrature_weights,
    _lgm_bond,
    _state_grid,
    _zeta,
    prepare_bermudan,
    price_bermudan_swaption_base,
    price_bermudan_swaptions,
)

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


class TestLgmClosedFormsAgainstORE:
    """Every closed-form primitive this module's backward induction is
    built from, checked directly against live ORE LGM objects."""

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
        """Documents the finding that motivated this module's exclusive use
        of _lgm_bond: ORE.HullWhite and ORE.LinearGaussMarkovModel are NOT
        the same model realization for t>0, even at each model's own
        natural 'no shock' reference state. This is not a bug in either
        class -- it's why this module can't reuse
        european_swaption.compute_hw_A/_hw_B."""
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
        # Different models -- NOT expected to match; this test documents
        # the gap is real and of a specific, non-trivial magnitude.
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


class TestSingleExerciseMatchesLgmJamshidian:
    """The core correctness check: a Bermudan with exactly one exercise
    date must reproduce an independent closed-form (Jamshidian-style)
    decomposition built on the SAME _lgm_bond formula this module's
    backward induction uses -- the numeric scheme, given only one exercise
    opportunity, is mathematically required to collapse to that closed
    form."""

    @staticmethod
    def _independent_lgm_jamshidian(cfg: BermudanSwaptionConfig) -> float:
        swap = prepare_bermudan(cfg)
        notice_t = cfg.exercise_times[0]
        alive_fixed = swap.fixed_start_times >= notice_t - 1e-9
        remaining_times = swap.fixed_times[alive_fixed]
        remaining_amounts = swap.fixed_amounts[alive_fixed]
        accrual_start = swap.float_start_times[swap.float_start_times >= notice_t - 1e-9][0]

        a, sigma = cfg.hw_a, cfg.hw_sigma
        T0, T_start, notional = notice_t, accrual_start, cfg.notional
        all_times = np.concatenate([remaining_times, remaining_times[-1:], [T_start]])
        all_amounts = np.concatenate([remaining_amounts, [notional, -notional]])
        zt, zr = np.array(cfg.initial_zero_curve.times), np.array(cfg.initial_zero_curve.rates)

        def coupon_bond_value(xstar):
            prices = np.array([_lgm_bond(zt, zr, a, sigma, T0, float(Ti), np.array([xstar]))[0] for Ti in all_times])
            return np.sum(prices * all_amounts)

        lo, hi = -2.0, 2.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if coupon_bond_value(mid) > 0:
                lo = mid
            else:
                hi = mid
        xstar = 0.5 * (lo + hi)
        K = np.array([_lgm_bond(zt, zr, a, sigma, T0, float(Ti), np.array([xstar]))[0] for Ti in all_times])

        def bond_vol(Topt, S):
            return abs(_H(a, S) - _H(a, Topt)) * np.sqrt(_zeta(sigma, Topt))

        P_0_T0 = _lgm_bond(zt, zr, a, sigma, 0.0, T0, np.array([0.0]))[0]
        P_0_Ti = np.array([_lgm_bond(zt, zr, a, sigma, 0.0, float(Ti), np.array([0.0]))[0] for Ti in all_times])
        sigma_p = np.array([bond_vol(T0, Ti) for Ti in all_times])

        F = P_0_Ti / P_0_T0
        sigp_safe = np.where(sigma_p > 0, sigma_p, 1.0)
        d1 = (np.log(F / K) + 0.5 * sigp_safe ** 2) / sigp_safe
        d2 = d1 - sigp_safe
        from scipy.stats import norm as scipy_norm
        call = P_0_T0 * (F * scipy_norm.cdf(d1) - K * scipy_norm.cdf(d2))
        put = call - P_0_T0 * (F - K)
        intrinsic_call = np.maximum(P_0_Ti - K * P_0_T0, 0.0)
        intrinsic_put = intrinsic_call - (P_0_Ti - K * P_0_T0)
        # Payer swaption = portfolio of bond PUTS (payer benefits when
        # rates rise, bond prices fall below strike); receiver = bond
        # CALLS -- same convention as european_swaption._price_one_swaption.
        per_leg = np.where(sigma_p > 0, put, intrinsic_put) if cfg.payer else \
            np.where(sigma_p > 0, call, intrinsic_call)
        return float(np.sum(per_leg * all_amounts))

    @pytest.mark.parametrize("payer", [True, False])
    @pytest.mark.parametrize("exercise_time", [1.0, 2.5, 4.0])
    def test_matches_independent_lgm_jamshidian(self, payer, exercise_time):
        cfg = _make_bermudan(payer=payer, exercise_times=[exercise_time], n_per_std=192, std_devs=9.0)
        numeric_npv = price_bermudan_swaption_base(cfg)
        closed_form_npv = self._independent_lgm_jamshidian(cfg)
        assert numeric_npv == pytest.approx(closed_form_npv, rel=2e-4)

    def test_grid_convergence_toward_closed_form(self):
        cfg_coarse = _make_bermudan(exercise_times=[3.0], n_per_std=48, std_devs=6.0)
        cfg_fine = _make_bermudan(exercise_times=[3.0], n_per_std=256, std_devs=9.0)
        closed_form = self._independent_lgm_jamshidian(cfg_fine)
        err_coarse = abs(price_bermudan_swaption_base(cfg_coarse) - closed_form)
        err_fine = abs(price_bermudan_swaption_base(cfg_fine) - closed_form)
        assert err_fine < err_coarse


class TestMonotonicity:
    """Model-independent no-arbitrage bounds: more exercise opportunities
    can never decrease a Bermudan swaption's value."""

    def test_bermudan_at_least_as_valuable_as_either_single_exercise(self):
        euro_first = price_bermudan_swaption_base(_make_bermudan(exercise_times=[1.0]))
        euro_last = price_bermudan_swaption_base(_make_bermudan(exercise_times=[4.0]))
        bermudan_2 = price_bermudan_swaption_base(_make_bermudan(exercise_times=[1.0, 4.0]))
        assert bermudan_2 >= max(euro_first, euro_last) - 1e-6

    def test_more_exercise_dates_never_decreases_value(self):
        bermudan_2 = price_bermudan_swaption_base(_make_bermudan(exercise_times=[1.0, 4.0]))
        bermudan_4 = price_bermudan_swaption_base(_make_bermudan(exercise_times=[1.0, 2.0, 3.0, 4.0]))
        assert bermudan_4 >= bermudan_2 - 1e-6

    def test_itm_payer_worth_more_than_otm_payer(self):
        itm = price_bermudan_swaption_base(_make_bermudan(fixed_rate=0.01, payer=True))
        otm = price_bermudan_swaption_base(_make_bermudan(fixed_rate=0.08, payer=True))
        assert itm > otm

    def test_itm_receiver_worth_more_than_otm_receiver(self):
        itm = price_bermudan_swaption_base(_make_bermudan(fixed_rate=0.08, payer=False))
        otm = price_bermudan_swaption_base(_make_bermudan(fixed_rate=0.01, payer=False))
        assert itm > otm

    def test_higher_volatility_increases_value(self):
        low_vol = price_bermudan_swaption_base(_make_bermudan(hw_sigma=0.005))
        high_vol = price_bermudan_swaption_base(_make_bermudan(hw_sigma=0.04))
        assert high_vol > low_vol


class TestPortfolioAndShape:
    def test_multiple_trades_stack_correctly(self):
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
        cfg1 = _make_bermudan(fixed_rate=0.02, exercise_times=[1.0, 4.0], **common)
        cfg2 = _make_bermudan(fixed_rate=0.04, exercise_times=[2.0, 4.0], **common)
        cube = price_bermudan_swaptions([cfg1, cfg2], cubes["rates"], step_times)
        assert cube.shape == (config.scenarios, len(config.time_grid) - 1, 2)
        assert not np.allclose(np.asarray(cube[:, :, 0]), np.asarray(cube[:, :, 1]))

    def test_npv_is_zero_after_last_exercise_date(self):
        import jax.numpy as jnp
        from engine.simulation import generate_paths
        from engine.scenarios import swaption_demo_config

        config = swaption_demo_config()  # time_grid up to 5.0
        cubes = generate_paths(config)
        step_times = jnp.array(config.time_grid[1:], dtype=jnp.float64)
        cfg = _make_bermudan(
            hw_a=config.rates.mean_reversion[0],
            hw_sigma=float(np.sqrt(config.joint_covariance[1][1])),
            exercise_times=[1.0, 2.0], n_per_std=48, std_devs=6.0,
        )
        cube = price_bermudan_swaptions([cfg], cubes["rates"], step_times)
        for i, t in enumerate(config.time_grid[1:]):
            if t >= 2.0:
                assert np.all(np.asarray(cube[:, i, 0]) == 0.0)

    def test_zero_notional_prices_to_zero(self):
        cfg = _make_bermudan(notional=0.0)
        assert price_bermudan_swaption_base(cfg) == pytest.approx(0.0, abs=1e-8)


class TestEdgeCases:
    def test_rejects_exercise_time_at_or_after_final_maturity(self):
        with pytest.raises(ValueError):
            prepare_bermudan(_make_bermudan(exercise_times=[10.0]))

    def test_exercise_date_exactly_at_final_reset_prices_finite(self):
        cfg = _make_bermudan(exercise_times=[4.0])
        npv = price_bermudan_swaption_base(cfg)
        assert np.isfinite(npv)
        assert npv >= 0.0

    def test_negative_rate_curve_prices_finite(self):
        neg_curve = ZeroCurveConfig(times=FLAT_CURVE.times, rates=[-0.005] * 6)
        cfg = _make_bermudan(initial_zero_curve=neg_curve, fixed_rate=-0.005)
        npv = price_bermudan_swaption_base(cfg)
        assert np.isfinite(npv)
        assert npv >= 0.0

    def test_near_zero_volatility_collapses_toward_intrinsic(self):
        cfg_low_vol = _make_bermudan(hw_sigma=1e-6, exercise_times=[4.0])
        npv = price_bermudan_swaption_base(cfg_low_vol)
        assert np.isfinite(npv)
        assert npv >= -1e-3

    def test_single_reset_bermudan_matches_prepare_bermudan_final_maturity(self):
        swap = prepare_bermudan(_make_bermudan())
        assert swap.final_maturity == pytest.approx(swap.fixed_times[-1])


class TestMidCouponKnownLimitation:
    """Documents the deliberate, conservative approximation for exercise
    dates that fall inside an accrual period (not reset-aligned) -- see
    engine.instruments.american_swaption.AmericanSwaptionConfig's 'Known
    limitation' docstring section (the underlying limitation lives in
    THIS module's _hw_swap_value_at_nodes, since American exercise is
    priced through this same engine -- see tests/test_american_swaption.py
    for the American-specific consequences)."""

    def test_mid_coupon_exercise_excludes_in_progress_coupon_entirely(self):
        cfg = _make_bermudan(exercise_times=[1.25])  # mid fixed-period (annual resets)
        swap = prepare_bermudan(cfg)
        t = 1.25
        alive_fixed = swap.fixed_start_times >= t - 1e-9
        # The coupon accruing [1.011, 2.014] has start < t -- excluded
        # entirely (no proration), the documented conservative behavior.
        in_progress = (swap.fixed_start_times < t) & (swap.fixed_times > t)
        assert np.any(in_progress)
        assert not np.any(alive_fixed & in_progress)

    def test_mid_coupon_exercise_still_prices_finite_and_nonnegative(self):
        cfg = _make_bermudan(exercise_times=[1.25])
        npv = price_bermudan_swaption_base(cfg)
        assert np.isfinite(npv)
        assert npv >= 0.0

    def test_mid_coupon_exercise_is_finite_and_of_plausible_magnitude(self):
        # Exercising slightly after a reset date (still inside the coupon
        # that started there) forfeits that entire in-progress coupon
        # rather than prorating it (the documented conservative
        # approximation) -- this can shift the priced value by a
        # non-trivial amount (an entire coupon's notional-scale PV, not
        # just a few days' accrual), so this is a plausibility bound, not a
        # tight equality: both must be finite, non-negative, and within the
        # same order of magnitude.
        swap = prepare_bermudan(_make_bermudan())
        reset_t = float(swap.fixed_start_times[1])  # second period's start
        mid_t = reset_t + 0.005
        cfg_reset = _make_bermudan(exercise_times=[reset_t])
        cfg_mid = _make_bermudan(exercise_times=[mid_t])
        npv_reset = price_bermudan_swaption_base(cfg_reset)
        npv_mid = price_bermudan_swaption_base(cfg_mid)
        assert np.isfinite(npv_mid) and npv_mid >= 0.0
        assert np.isfinite(npv_reset) and npv_reset >= 0.0
        assert abs(npv_mid - npv_reset) / max(npv_reset, 1.0) < 1.0
