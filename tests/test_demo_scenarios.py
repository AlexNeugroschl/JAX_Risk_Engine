"""
The demo `SimulationConfig` builders in `engine.simulation.demo_scenarios`
(`cross_asset_demo_config`, `single_currency_swap_demo_config`, `swaption_demo_config`) and
`flat_yield_curves`, checked directly rather than only as inputs to downstream pricers:
each factor's `initial_zero_curves` agrees with its `initial_rates`/theta, and the flat
curves' discount factors are correct.
"""
import numpy as np
import pytest

import ORE

from engine.simulation.demo_scenarios import (
    EVAL_DATE,
    SWAP_DEMO_MATURITIES,
    cross_asset_demo_config,
    flat_yield_curves,
    single_currency_swap_demo_config,
    swaption_demo_config,
)
from engine.simulation.market_model import generate_paths


class TestFlatYieldCurves:
    def test_matches_direct_ore_flatforward_discount(self):
        """Discount factors equal an independent `ORE.FlatForward`'s."""
        disc_rate, fwd_rate = 0.03, 0.035
        cube = flat_yield_curves(disc_rate, fwd_rate)
        assert cube.shape == (1, 1, len(SWAP_DEMO_MATURITIES), 2)

        dc = ORE.Actual365Fixed()
        disc_curve = ORE.YieldTermStructureHandle(ORE.FlatForward(EVAL_DATE, disc_rate, dc))
        fwd_curve = ORE.YieldTermStructureHandle(ORE.FlatForward(EVAL_DATE, fwd_rate, dc))
        expected_disc = np.array([disc_curve.discount(EVAL_DATE + int(round(t * 365)))
                                   for t in SWAP_DEMO_MATURITIES])
        expected_fwd = np.array([fwd_curve.discount(EVAL_DATE + int(round(t * 365)))
                                  for t in SWAP_DEMO_MATURITIES])

        np.testing.assert_allclose(np.asarray(cube[0, 0, :, 0]), expected_disc, atol=1e-12)
        np.testing.assert_allclose(np.asarray(cube[0, 0, :, 1]), expected_fwd, atol=1e-12)

    def test_zero_rate_curve_is_all_ones(self):
        """A 0% curve discounts to exactly 1.0 at every pillar."""
        cube = flat_yield_curves(0.0, 0.0)
        np.testing.assert_allclose(np.asarray(cube), 1.0, atol=1e-12)

    def test_higher_rate_gives_smaller_discount_factor(self):
        cube_low = flat_yield_curves(0.01, 0.01)
        cube_high = flat_yield_curves(0.10, 0.10)
        assert np.all(np.asarray(cube_high) < np.asarray(cube_low))

    def test_custom_maturities_and_eval_date_are_honored(self):
        """The `maturities` and `eval_date` overrides are applied, not ignored."""
        custom_maturities = [0.5, 1.5, 3.0]
        custom_eval_date = ORE.Date(1, 1, 2027)
        cube = flat_yield_curves(0.02, 0.02, maturities=custom_maturities, eval_date=custom_eval_date)
        assert cube.shape == (1, 1, 3, 2)

        dc = ORE.Actual365Fixed()
        curve = ORE.YieldTermStructureHandle(ORE.FlatForward(custom_eval_date, 0.02, dc))
        expected = np.array([curve.discount(custom_eval_date + int(round(t * 365)))
                              for t in custom_maturities])
        np.testing.assert_allclose(np.asarray(cube[0, 0, :, 0]), expected, atol=1e-12)


class TestDemoConfigInternalConsistency:
    """Each config's `initial_zero_curves` agrees with its own `initial_rates`/theta, so the
    t=0 repricing in `generate_paths` reprices the curve the short rate starts from (a
    shape check would not catch a mismatch)."""

    @pytest.mark.parametrize("config_fn", [cross_asset_demo_config, single_currency_swap_demo_config])
    def test_zero_curve_short_end_matches_initial_rates_and_theta(self, config_fn):
        cfg = config_fn()
        rates_cfg = cfg.rates
        assert rates_cfg.initial_zero_curves is not None
        assert len(rates_cfg.initial_zero_curves) == len(rates_cfg.initial_rates)

        for k, zc in enumerate(rates_cfg.initial_zero_curves):
            short_end_rate = np.interp(0.0, zc.times, zc.rates)
            np.testing.assert_allclose(
                short_end_rate, rates_cfg.initial_rates[k], atol=1e-9,
                err_msg=f"rate factor {k}: initial_zero_curves' t=0 pillar rate "
                        f"does not match initial_rates[{k}] -- generate_paths' "
                        f"yield_curves output would not reprice the rate this "
                        f"factor is actually initialized to."
            )
            np.testing.assert_allclose(
                short_end_rate, rates_cfg.theta[k], atol=1e-9,
                err_msg=f"rate factor {k}: initial_zero_curves' t=0 pillar rate "
                        f"does not match theta[{k}] -- every existing test in "
                        f"this suite that relies on theta==initial_rates as a "
                        f"fixed point (see TestHullWhiteMeanReversionTransition) "
                        f"assumes this holds for the demo configs too."
            )

    @pytest.mark.parametrize("config_fn", [cross_asset_demo_config, single_currency_swap_demo_config])
    def test_joint_covariance_is_square_and_matches_factor_count(self, config_fn):
        cfg = config_fn()
        num_eq = len(cfg.equities.initial_prices)
        num_hw = len(cfg.rates.initial_rates)
        n = num_eq + num_hw
        cov = np.asarray(cfg.joint_covariance)
        assert cov.shape == (n, n)
        np.testing.assert_allclose(cov, cov.T, atol=1e-12, err_msg="joint_covariance must be symmetric")
        # A valid (PSD) covariance: eigenvalues non-negative.
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals >= -1e-10), f"joint_covariance is not PSD: eigenvalues={eigvals}"

    def test_equity_rate_mapping_row_count_matches_num_equities_and_col_count_matches_num_rates(self):
        for cfg in [cross_asset_demo_config(), single_currency_swap_demo_config(), swaption_demo_config()]:
            num_eq = len(cfg.equities.initial_prices)
            num_hw = len(cfg.rates.initial_rates)
            mapping = cfg.equities.rate_mapping
            assert len(mapping) == num_eq
            assert all(len(row) == num_hw for row in mapping)

    def test_swap_demo_maturities_align_with_single_currency_swap_config(self):
        cfg = single_currency_swap_demo_config()
        assert cfg.rates.maturities == SWAP_DEMO_MATURITIES

    def test_swaption_demo_config_has_no_maturities_or_zero_curves(self):
        """Swaption pricing uses `hw_paths`, not the yield-curve cube, so `maturities` stays
        unset (per the builder's docstring)."""
        cfg = swaption_demo_config()
        assert cfg.rates.maturities is None
        assert cfg.rates.initial_zero_curves is None


class TestDemoConfigsRunEndToEnd:
    """Each config runs through `generate_paths` at a small scenario count."""

    def test_cross_asset_demo_config_runs_and_reprices_both_curves_at_t0(self):
        """With a first step of 1e-8 (as in TestHullWhiteAMatrix) the simulated curves equal
        the initial curves to within 1e-4, using the real demo config."""
        import dataclasses
        cfg = cross_asset_demo_config()
        cfg = dataclasses.replace(cfg, time_grid=[0.0, 1e-8], scenarios=8)
        result = generate_paths(cfg)
        yc = np.asarray(result["yield_curves"])  # [S, 1, Maturities, NumRates]
        assert bool(np.all(np.isfinite(yc)))

        maturities = np.array(cfg.rates.maturities)
        for k, zc in enumerate(cfg.rates.initial_zero_curves):
            expected = np.exp(-np.interp(maturities, zc.times, zc.rates) * maturities)
            # Every scenario agrees (the shock over 1e-8 is negligible).
            for s in range(yc.shape[0]):
                np.testing.assert_allclose(yc[s, 0, :, k], expected, atol=1e-4)

    def test_single_currency_swap_demo_config_runs(self):
        cfg = single_currency_swap_demo_config()
        import dataclasses
        result = generate_paths(dataclasses.replace(cfg, scenarios=64))
        assert bool(np.all(np.isfinite(np.asarray(result["yield_curves"]))))
        assert result["yield_curves"].shape[2] == len(SWAP_DEMO_MATURITIES)

    def test_swaption_demo_config_runs(self):
        cfg = swaption_demo_config()
        import dataclasses
        result = generate_paths(dataclasses.replace(cfg, scenarios=64))
        assert "yield_curves" not in result
        assert bool(np.all(np.isfinite(np.asarray(result["rates"]))))
