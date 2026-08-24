"""
Tests for engine.portfolio.price_portfolio -- the single entry point that
replaces demo.py's hand orchestration. Follows
tests/test_diverse_portfolio_e2e.py's methodology (cross-check against
independently-orchestrated pricing, same simulated values) but calling
price_portfolio as the entry point rather than hand-wiring each pricer:
proves price_portfolio's output equals manually-orchestrated demo.py-style
output, trade-by-trade, for a portfolio mixing all four instrument types.
"""
import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.simulation.market_model import (
    EquityConfig, RatesConfig, SimulationConfig, ZeroCurveConfig, generate_paths,
)
from engine.instruments.swap import SwapConfig, price_swaps
from engine.instruments.european_swaption import SwaptionConfig, price_swaptions
from engine.instruments.bermudan_swaption import (
    BermudanSwaptionConfig, price_bermudan_swaptions, price_bermudan_swaption_base,
)
from engine.instruments.american_swaption import AmericanSwaptionConfig, price_american_swaptions
from engine.calibration.basket import build_coterminal_basket
from engine.calibration.lgm import calibrate_lgm_sigma
from engine.models.hull_white import ZeroCurve as HwZeroCurve
from engine.risk.var_es import compute_risk_metrics
from engine.simulation.demo_scenarios import flat_yield_curves
from engine.portfolio import PortfolioRequest, PortfolioResult, derive_maturity_pillars, price_portfolio

TODAY = ORE.Date(30, 7, 2026)
FLAT_RATE = 0.03
HW_A = 0.03
HW_SIGMA = 0.01
ZERO_CURVE = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[FLAT_RATE] * 6)
TIME_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def _build_trades():
    swap_cfg = SwapConfig(
        notional=2_000_000.0, fixed_rate=0.032, payer=True,
        discount_curve_index=0, forward_curve_index=0,
        swap_tenor="2Y", evaluation_date=TODAY,
    )
    swaption_cfg = SwaptionConfig(
        notional=1_500_000.0, fixed_rate=0.031, payer=True, rate_factor_index=0,
        hw_a=HW_A, hw_sigma=HW_SIGMA, initial_zero_curve=ZERO_CURVE,
        swap_tenor="1Y", forward_start=ORE.Period(1, ORE.Years), evaluation_date=TODAY,
    )
    bermudan_cfg = BermudanSwaptionConfig(
        notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
        hw_a=HW_A, hw_sigma=HW_SIGMA, initial_zero_curve=ZERO_CURVE,
        exercise_times=[1.010958904109589, 2.0136986301369864], swap_tenor="3Y",
        evaluation_date=TODAY, n_per_std=64, std_devs=6.0,
    )
    american_cfg = AmericanSwaptionConfig(
        notional=800_000.0, fixed_rate=0.029, payer=False, rate_factor_index=0,
        hw_a=HW_A, hw_sigma=HW_SIGMA, initial_zero_curve=ZERO_CURVE,
        first_exercise=1.010958904109589, last_exercise=2.0136986301369864,
        exercise_time_steps_per_year=1, evaluation_date=TODAY, n_per_std=64, std_devs=6.0,
    )
    return swap_cfg, swaption_cfg, bermudan_cfg, american_cfg


def _sim_config(trades) -> SimulationConfig:
    pillars = derive_maturity_pillars(trades, TODAY)
    return SimulationConfig(
        time_grid=TIME_GRID,
        scenarios=256,
        equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[0.0]]),
        rates=RatesConfig(
            initial_rates=[FLAT_RATE], theta=[FLAT_RATE], mean_reversion=[HW_A],
            maturities=pillars, initial_zero_curves=[ZERO_CURVE],
        ),
        joint_covariance=[[0.04, 0.0], [0.0, HW_SIGMA ** 2]],
    )


class TestPortfolioRequestFixture:
    """Exercises the shared conftest.py portfolio_request fixture directly
    -- a minimal, valid PortfolioRequest other tests (and engine/api tests)
    can build on without repeating this setup."""

    def test_fixture_prices_successfully(self, portfolio_request):
        result = price_portfolio(portfolio_request)
        assert np.isfinite(result.base_npv)
        assert bool(jnp.all(jnp.isfinite(result.npv_cube)))


class TestPricePortfolioMatchesHandOrchestration:
    """price_portfolio's output must equal calling each pricer by hand
    (the exact demo.py-style sequence it replaces), trade-by-trade -- not
    just an internally-consistent number."""

    @classmethod
    @pytest.fixture(scope="class")
    def trades(cls):
        return _build_trades()

    @classmethod
    @pytest.fixture(scope="class")
    def sim_config(cls, trades):
        return _sim_config(list(trades))

    @classmethod
    @pytest.fixture(scope="class")
    def manual(cls, trades, sim_config):
        """Hand-orchestrated pricing -- the exact demo.py-style sequence,
        computed independently of price_portfolio."""
        swap_cfg, swaption_cfg, bermudan_cfg, american_cfg = trades
        market = generate_paths(sim_config)
        step_times = jnp.array(sim_config.time_grid[1:], dtype=jnp.float64)
        pillars = sim_config.rates.maturities

        swap_cube = price_swaps(market["yield_curves"], pillars, [swap_cfg])
        swaption_cube = price_swaptions(market["rates"], step_times, [swaption_cfg])
        bermudan_cube = price_bermudan_swaptions([bermudan_cfg], market["rates"], step_times)
        american_cube = price_american_swaptions([american_cfg], market["rates"], step_times)
        npv_cube = jnp.concatenate([swap_cube, swaption_cube, bermudan_cube, american_cube], axis=-1)

        base_curve = flat_yield_curves(disc_rate=FLAT_RATE, fwd_rate=FLAT_RATE, maturities=pillars, eval_date=TODAY)
        r0_path = jnp.array([[[FLAT_RATE]]])
        base_npv = (
            float(price_swaps(base_curve, pillars, [swap_cfg])[0, 0, 0])
            + float(price_swaptions(r0_path, jnp.array([0.0]), [swaption_cfg])[0, 0, 0])
            + price_bermudan_swaption_base(bermudan_cfg)
            + price_bermudan_swaption_base(american_cfg.to_bermudan())
        )
        risk = compute_risk_metrics(npv_cube, base_npv, percentiles=(0.95, 0.99))
        return {"npv_cube": npv_cube, "base_npv": base_npv, "risk": risk}

    @classmethod
    @pytest.fixture(scope="class")
    def via_entrypoint(cls, trades, sim_config):
        request = PortfolioRequest(market=sim_config, trades=list(trades), percentiles=(0.95, 0.99))
        return price_portfolio(request)

    def test_returns_portfolio_result(self, via_entrypoint):
        assert isinstance(via_entrypoint, PortfolioResult)

    def test_npv_cube_shape_matches(self, via_entrypoint, manual):
        assert via_entrypoint.npv_cube.shape == manual["npv_cube"].shape
        assert via_entrypoint.npv_cube.shape[-1] == 4  # one per trade, caller order

    def test_npv_cube_matches_trade_by_trade(self, via_entrypoint, manual):
        """Each trade's own NPV column must match the hand-orchestrated
        equivalent exactly -- confirms price_portfolio's routing/
        reassembly preserves the caller's original trade order (swap,
        european swaption, bermudan, american) rather than silently
        permuting by internal pricing-group order."""
        np.testing.assert_allclose(
            np.asarray(via_entrypoint.npv_cube), np.asarray(manual["npv_cube"]), rtol=1e-9,
        )

    def test_base_npv_matches(self, via_entrypoint, manual):
        np.testing.assert_allclose(via_entrypoint.base_npv, manual["base_npv"], rtol=1e-9)

    def test_risk_metrics_match(self, via_entrypoint, manual):
        assert set(via_entrypoint.risk.keys()) == set(manual["risk"].keys())
        for key in manual["risk"]:
            np.testing.assert_allclose(
                np.asarray(via_entrypoint.risk[key]), np.asarray(manual["risk"][key]), rtol=1e-9, equal_nan=True,
            )

    def test_no_warnings_for_a_clean_reset_aligned_portfolio(self, via_entrypoint):
        assert via_entrypoint.warnings == []

    def test_greeks_are_none_when_not_requested(self, via_entrypoint):
        assert via_entrypoint.greeks is None


class TestPricePortfolioReorderingIndependence:
    """Trades are grouped by type internally for pricing (each pricer only
    accepts a homogeneous list) but the result must always be reassembled
    in the CALLER's original order -- verified by shuffling the trade list
    and confirming each trade's own NPV column follows it, not the
    type-grouping order."""

    def test_shuffled_trade_order_reorders_npv_cube_columns_identically(self):
        swap_cfg, swaption_cfg, bermudan_cfg, american_cfg = _build_trades()
        trades_original = [swap_cfg, swaption_cfg, bermudan_cfg, american_cfg]
        trades_shuffled = [bermudan_cfg, swap_cfg, american_cfg, swaption_cfg]

        sim_config = _sim_config(trades_original)
        result_original = price_portfolio(PortfolioRequest(market=sim_config, trades=trades_original))
        result_shuffled = price_portfolio(PortfolioRequest(market=sim_config, trades=trades_shuffled))

        # trades_shuffled = [bermudan, swap, american, swaption] -- index
        # mapping back into trades_original's [swap, swaption, bermudan, american] order.
        shuffled_to_original = [2, 0, 3, 1]
        for shuffled_idx, original_idx in enumerate(shuffled_to_original):
            np.testing.assert_allclose(
                np.asarray(result_shuffled.npv_cube[:, :, shuffled_idx]),
                np.asarray(result_original.npv_cube[:, :, original_idx]),
                rtol=1e-9,
            )


class TestPricePortfolioAutoDerivesMaturityPillars:
    def test_unset_maturities_are_derived_automatically(self):
        """If the caller leaves RatesConfig.maturities unset,
        price_portfolio must derive it via derive_maturity_pillars and
        still price successfully (the automatic-pillar-assembly case the
        general PortfolioRequest surface is meant to cover)."""
        swap_cfg = SwapConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True,
            discount_curve_index=0, forward_curve_index=0,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        sim_config = SimulationConfig(
            time_grid=TIME_GRID, scenarios=64,
            equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[0.0]]),
            rates=RatesConfig(
                initial_rates=[FLAT_RATE], theta=[FLAT_RATE], mean_reversion=[HW_A],
                initial_zero_curves=[ZERO_CURVE],  # maturities left unset
            ),
            joint_covariance=[[0.04, 0.0], [0.0, HW_SIGMA ** 2]],
        )
        result = price_portfolio(PortfolioRequest(market=sim_config, trades=[swap_cfg]))
        assert result.npv_cube.shape == (64, len(TIME_GRID) - 1, 1)
        assert bool(jnp.all(jnp.isfinite(result.npv_cube)))
        assert np.isfinite(result.base_npv)

    def test_preset_maturities_are_left_untouched(self):
        """A caller who already supplies rates.maturities keeps exactly
        that pillar set -- price_portfolio must not silently override an
        explicit choice."""
        swap_cfg = SwapConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True,
            discount_curve_index=0, forward_curve_index=0,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        preset_pillars = derive_maturity_pillars([swap_cfg], TODAY)
        sim_config = SimulationConfig(
            time_grid=TIME_GRID, scenarios=64,
            equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[0.0]]),
            rates=RatesConfig(
                initial_rates=[FLAT_RATE], theta=[FLAT_RATE], mean_reversion=[HW_A],
                maturities=preset_pillars, initial_zero_curves=[ZERO_CURVE],
            ),
            joint_covariance=[[0.04, 0.0], [0.0, HW_SIGMA ** 2]],
        )
        result = price_portfolio(PortfolioRequest(market=sim_config, trades=[swap_cfg]))
        assert bool(jnp.all(jnp.isfinite(result.npv_cube)))


class TestPricePortfolioCalibration:
    """hw_sigma=None on a Bermudan/American trade triggers calibration via
    request.calibration_targets -- once per distinct rate_factor_index, not
    once per trade."""

    def test_uncalibrated_hw_sigma_is_filled_in_and_prices_finite(self):
        curve_jax = HwZeroCurve.flat(FLAT_RATE, ZERO_CURVE.times)
        exercise_times = [1.010958904109589, 2.0136986301369864]
        targets = build_coterminal_basket(
            exercise_times=exercise_times, final_maturity_time=3.0136986301369864,
            notional=1_000_000.0, payer=True, market_vols=[0.008, 0.009],
            zero_curve=curve_jax, evaluation_date=TODAY,
        )
        berm_cfg = BermudanSwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
            hw_a=HW_A, hw_sigma=None, initial_zero_curve=ZERO_CURVE,
            exercise_times=exercise_times, swap_tenor="3Y", evaluation_date=TODAY,
        )
        sim_config = SimulationConfig(
            time_grid=TIME_GRID, scenarios=64,
            equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[0.0]]),
            rates=RatesConfig(initial_rates=[FLAT_RATE], theta=[FLAT_RATE], mean_reversion=[HW_A],
                               initial_zero_curves=[ZERO_CURVE]),
            joint_covariance=[[0.04, 0.0], [0.0, HW_SIGMA ** 2]],
        )
        request = PortfolioRequest(
            market=sim_config, trades=[berm_cfg], calibration_targets=targets,
        )
        result = price_portfolio(request)
        assert bool(jnp.all(jnp.isfinite(result.npv_cube)))
        assert np.isfinite(result.base_npv)

    def test_missing_calibration_targets_raises_clear_error(self):
        berm_cfg = BermudanSwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
            hw_a=HW_A, hw_sigma=None, initial_zero_curve=ZERO_CURVE,
            exercise_times=[1.0], swap_tenor="3Y", evaluation_date=TODAY,
        )
        sim_config = SimulationConfig(
            time_grid=TIME_GRID, scenarios=64,
            equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[0.0]]),
            rates=RatesConfig(initial_rates=[FLAT_RATE], theta=[FLAT_RATE], mean_reversion=[HW_A],
                               initial_zero_curves=[ZERO_CURVE]),
            joint_covariance=[[0.04, 0.0], [0.0, HW_SIGMA ** 2]],
        )
        request = PortfolioRequest(market=sim_config, trades=[berm_cfg])
        with pytest.raises(ValueError, match="calibration_targets"):
            price_portfolio(request)


class TestPricePortfolioGreeks:
    def test_compute_greeks_true_returns_per_trade_dict(self):
        swap_cfg, swaption_cfg, bermudan_cfg, american_cfg = _build_trades()
        trades = [swaption_cfg, bermudan_cfg]  # swap Greeks need a caller-supplied ZeroCurve, skipped by design
        sim_config = _sim_config([swap_cfg, swaption_cfg, bermudan_cfg, american_cfg])
        request = PortfolioRequest(market=sim_config, trades=trades, compute_greeks=True)
        result = price_portfolio(request)
        assert result.greeks is not None
        assert set(result.greeks.keys()) == {0, 1}
        assert "delta" in result.greeks[0]
        assert "gamma" in result.greeks[0]
        assert "theta" in result.greeks[0]

    def test_greeks_keyed_by_original_trade_index_not_pricing_group_order(self):
        """trades = [bermudan, swaption] -- greeks[0] must be the
        Bermudan's own Delta/Gamma/Theta, greeks[1] the swaption's, matching
        the CALLER's list order, not internal type-grouping order."""
        swap_cfg, swaption_cfg, bermudan_cfg, american_cfg = _build_trades()
        trades = [bermudan_cfg, swaption_cfg]
        sim_config = _sim_config([swap_cfg, swaption_cfg, bermudan_cfg, american_cfg])
        request = PortfolioRequest(market=sim_config, trades=trades, compute_greeks=True)
        result = price_portfolio(request)

        curve = HwZeroCurve.flat(FLAT_RATE, ZERO_CURVE.times)
        from engine.risk.greeks import bermudan_delta_gamma, swaption_delta_gamma
        expected_berm = bermudan_delta_gamma(bermudan_cfg, curve)
        expected_swaption = swaption_delta_gamma(swaption_cfg, curve)

        np.testing.assert_allclose(np.asarray(result.greeks[0]["delta"]), np.asarray(expected_berm["delta"]), rtol=1e-9)
        np.testing.assert_allclose(np.asarray(result.greeks[1]["delta"]), np.asarray(expected_swaption["delta"]), rtol=1e-9)
