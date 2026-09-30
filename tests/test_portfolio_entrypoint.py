"""
`engine.portfolio.price_portfolio` equals the same pricing orchestrated by hand (the
demo.py sequence), trade by trade, for a portfolio of all four instrument types, on the same
simulated values.
"""
import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.models.ore_builders import time_from_reference
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
from engine.models.hull_white import discount as hw_discount
from engine.risk.exposure import netting_set_profile
from demos.demo_scenarios import flat_yield_curves
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
        exercise_dates=[ORE.Date(3, 8, 2027), ORE.Date(3, 8, 2028)], swap_tenor="3Y",
        evaluation_date=TODAY, n_per_std=64, std_devs=6.0,
    )
    american_cfg = AmericanSwaptionConfig(
        notional=800_000.0, fixed_rate=0.029, payer=False, rate_factor_index=0,
        hw_a=HW_A, hw_sigma=HW_SIGMA, initial_zero_curve=ZERO_CURVE,
        first_exercise_date=ORE.Date(3, 8, 2027), last_exercise_date=ORE.Date(3, 8, 2028),
        swap_tenor="5Y", exercise_time_steps_per_year=1, evaluation_date=TODAY, n_per_std=64, std_devs=6.0,
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
    """The shared conftest `portfolio_request` fixture prices."""

    def test_fixture_prices_successfully(self, portfolio_request):
        result = price_portfolio(portfolio_request)
        assert np.isfinite(result.base_npv)
        assert bool(jnp.all(jnp.isfinite(result.npv_cube)))


class TestPricePortfolioMatchesHandOrchestration:
    """`price_portfolio` equals each pricer called by hand, trade by trade."""

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
        """Hand-orchestrated pricing, independent of `price_portfolio`."""
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
            + price_bermudan_swaption_base(american_cfg)
        )
        # Netting-set exposure from the simulation's numeraire and rate factor 0's curve.
        discount = hw_discount(HwZeroCurve.from_config(sim_config.rates.initial_zero_curves[0]), step_times)
        exposure = netting_set_profile(
            npv_cube, [base_npv], market["numeraire"], discount,
            np.asarray(sim_config.time_grid[1:]), quantiles=(0.95, 0.99),
        )
        return {"npv_cube": npv_cube, "base_npv": base_npv, "exposure": exposure}

    @classmethod
    @pytest.fixture(scope="class")
    def via_entrypoint(cls, trades, sim_config):
        request = PortfolioRequest(market=sim_config, trades=list(trades), pfe_quantiles=(0.95, 0.99))
        return price_portfolio(request)

    @pytest.mark.slow
    def test_returns_portfolio_result(self, via_entrypoint):
        assert isinstance(via_entrypoint, PortfolioResult)

    def test_npv_cube_shape_matches(self, via_entrypoint, manual):
        assert via_entrypoint.npv_cube.shape == manual["npv_cube"].shape
        assert via_entrypoint.npv_cube.shape[-1] == 4  # one per trade, caller order

    def test_npv_cube_matches_trade_by_trade(self, via_entrypoint, manual):
        """Each trade's column matches, in the caller's order (swap, European, Bermudan,
        American), not the internal pricing-group order."""
        np.testing.assert_allclose(
            np.asarray(via_entrypoint.npv_cube), np.asarray(manual["npv_cube"]), rtol=1e-9,
        )

    def test_base_npv_matches(self, via_entrypoint, manual):
        np.testing.assert_allclose(via_entrypoint.base_npv, manual["base_npv"], rtol=1e-9)

    def test_exposure_matches(self, via_entrypoint, manual):
        got, expected = via_entrypoint.exposure, manual["exposure"]
        np.testing.assert_array_equal(got.times, expected.times)
        for name in ("epe", "ene", "ee_b", "eee_b"):
            np.testing.assert_allclose(
                np.asarray(getattr(got, name)), np.asarray(getattr(expected, name)), rtol=1e-9, atol=1e-9,
            )
        assert set(got.pfe) == set(expected.pfe) == {"PFE_95", "PFE_99"}
        for key in expected.pfe:
            np.testing.assert_allclose(np.asarray(got.pfe[key]), np.asarray(expected.pfe[key]), rtol=1e-9, atol=1e-9)

    def test_one_standalone_exposure_per_trade(self, via_entrypoint):
        assert len(via_entrypoint.trade_exposures) == 4
        for i, profile in enumerate(via_entrypoint.trade_exposures):
            npv0 = via_entrypoint.base_npv_per_trade[i]
            assert float(profile.epe[0]) == pytest.approx(max(npv0, 0.0))
            assert float(profile.ene[0]) == pytest.approx(max(-npv0, 0.0))

    def test_no_warnings_other_than_the_aged_swap_one(self, via_entrypoint):
        """The option trades raise no warning: their exercise is priced as ORE prices it.
        The aged-swap warning (the swap is spot-starting on a multi-step grid; I-04) is
        expected and excluded."""
        # The expiry warning (audit M-3) is excluded too: it concerns exposure after the
        # last exercise, not how exercise is priced.
        unrelated = [w for w in via_entrypoint.warnings
                     if "already started accruing" not in w and "last exercise" not in w]
        assert unrelated == []

    def test_greeks_are_none_when_not_requested(self, via_entrypoint):
        assert via_entrypoint.greeks is None


class TestPricePortfolioReorderingIndependence:
    """Trades are grouped by type for pricing, but results come back in the caller's order
    (checked by shuffling the input)."""

    def test_shuffled_trade_order_reorders_npv_cube_columns_identically(self):
        swap_cfg, swaption_cfg, bermudan_cfg, american_cfg = _build_trades()
        trades_original = [swap_cfg, swaption_cfg, bermudan_cfg, american_cfg]
        trades_shuffled = [bermudan_cfg, swap_cfg, american_cfg, swaption_cfg]

        sim_config = _sim_config(trades_original)
        result_original = price_portfolio(PortfolioRequest(market=sim_config, trades=trades_original))
        result_shuffled = price_portfolio(PortfolioRequest(market=sim_config, trades=trades_shuffled))

        # trades_shuffled = [bermudan, swap, american, swaption], mapped back to the original
        # [swap, swaption, bermudan, american] order.
        shuffled_to_original = [2, 0, 3, 1]
        for shuffled_idx, original_idx in enumerate(shuffled_to_original):
            np.testing.assert_allclose(
                np.asarray(result_shuffled.npv_cube[:, :, shuffled_idx]),
                np.asarray(result_original.npv_cube[:, :, original_idx]),
                rtol=1e-9,
            )


class TestPricePortfolioAutoDerivesMaturityPillars:
    def test_unset_maturities_are_derived_automatically(self):
        """Unset `RatesConfig.maturities` are derived with `derive_maturity_pillars`."""
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
        """Maturities the caller supplies are kept unchanged."""
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
    """hw_sigma=None on a Bermudan/American is calibrated from
    `request.calibration_targets`, once per rate factor."""

    @pytest.mark.slow
    def test_uncalibrated_hw_sigma_is_filled_in_and_prices_finite(self):
        curve_jax = HwZeroCurve.flat(FLAT_RATE, ZERO_CURVE.times)
        exercise_dates = [ORE.Date(3, 8, 2027), ORE.Date(3, 8, 2028)]
        targets = build_coterminal_basket(
            exercise_times=[time_from_reference(TODAY, d) for d in exercise_dates],
            final_maturity_time=3.0136986301369864,
            notional=1_000_000.0, payer=True, market_vols=[0.008, 0.009],
            zero_curve=curve_jax, evaluation_date=TODAY,
        )
        berm_cfg = BermudanSwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
            hw_a=HW_A, hw_sigma=None, initial_zero_curve=ZERO_CURVE,
            exercise_dates=exercise_dates, swap_tenor="3Y", evaluation_date=TODAY,
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
            exercise_dates=[TODAY + 365], swap_tenor="3Y", evaluation_date=TODAY,
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
    @pytest.mark.slow
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

    @pytest.mark.slow
    def test_greeks_keyed_by_original_trade_index_not_pricing_group_order(self):
        """With trades [bermudan, swaption], greeks[0] is the Bermudan's and greeks[1] the
        swaption's (caller order)."""
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


class TestPricePortfolioPrecision:
    """`PrecisionConfig`'s four knobs (simulation, pricing, risk, calibration), each 32 or 64,
    with optional per-type/per-metric overrides for pricing and risk. The default reproduces
    all-64 behaviour; each knob at 32 shows up in the right output's dtype without affecting
    the others."""

    def _request(self, precision=None, compute_greeks=False):
        from engine.portfolio import PrecisionConfig
        swap_cfg, swaption_cfg, bermudan_cfg, american_cfg = _build_trades()
        trades = [swap_cfg, swaption_cfg, bermudan_cfg, american_cfg]
        sim_config = _sim_config(trades)
        kwargs = dict(market=sim_config, trades=trades, compute_greeks=compute_greeks)
        if precision is not None:
            kwargs["precision"] = precision
        return PortfolioRequest(**kwargs)

    def test_default_precision_matches_pre_feature_behavior(self):
        """No `precision` gives all-64 and plain ints for pricing/risk (not override
        objects)."""
        request = self._request()
        assert request.precision.simulation == 64
        assert request.precision.pricing == 64
        assert request.precision.risk == 64
        assert request.precision.calibration == 64
        assert isinstance(request.precision.pricing, int)
        assert isinstance(request.precision.risk, int)
        result = price_portfolio(request)
        assert result.npv_cube.dtype == jnp.float64
        assert bool(jnp.all(jnp.isfinite(result.npv_cube)))

    @pytest.mark.slow
    def test_simulation_32_produces_float32_market_data_and_still_prices(self):
        from engine.portfolio import PrecisionConfig
        request = self._request(precision=PrecisionConfig(simulation=32))
        result = price_portfolio(request)
        # simulation=32 gives float32 market data; npv_cube's dtype follows `pricing` (64
        # here). The point is that simulation=32 alone does not break pricing.
        assert bool(jnp.all(jnp.isfinite(result.npv_cube)))
        assert np.isfinite(result.base_npv)

    def test_pricing_32_produces_float32_npv_cube_and_base_npv(self):
        """pricing=32 makes npv_cube and the base NPV computation float32, across all four
        trade types."""
        from engine.portfolio import PrecisionConfig
        request = self._request(precision=PrecisionConfig(pricing=32))
        result = price_portfolio(request)
        assert result.npv_cube.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(result.npv_cube)))
        assert np.isfinite(result.base_npv)

    def test_pricing_64_vs_32_base_npv_numerically_close(self):
        """float32 pricing gives a base NPV close to float64's."""
        from engine.portfolio import PrecisionConfig
        request64 = self._request(precision=PrecisionConfig(pricing=64))
        request32 = self._request(precision=PrecisionConfig(pricing=32))
        result64 = price_portfolio(request64)
        result32 = price_portfolio(request32)
        assert result32.base_npv == pytest.approx(result64.base_npv, rel=1e-3)

    @pytest.mark.slow
    def test_risk_32_changes_greeks_dtype_for_swaption_and_bermudan(self):
        """risk=32 reaches the European's and the Bermudan's Greeks through
        `price_portfolio`. (Vega needs a calibrated Sigma; it is tested in
        test_greeks_bermudan.py.)"""
        from engine.portfolio import PrecisionConfig
        request = self._request(precision=PrecisionConfig(risk=32), compute_greeks=True)
        result = price_portfolio(request)
        assert result.greeks is not None
        # trades = [swap, swaption, bermudan, american]; the options are checked here.
        for idx in (1, 2, 3):
            assert idx in result.greeks
            for key, val in result.greeks[idx].items():
                if key == "theta":
                    continue  # a plain Python float (forward-difference NPV), not a JAX array
                assert jnp.asarray(val).dtype == jnp.float32, f"trade {idx} greek {key!r} not float32"

    @pytest.mark.slow
    def test_mixed_precision_each_stage_independent(self):
        """simulation=64, pricing=32, risk=32 in one request, each taking effect."""
        from engine.portfolio import PrecisionConfig
        request = self._request(
            precision=PrecisionConfig(simulation=64, pricing=32, risk=32), compute_greeks=True,
        )
        result = price_portfolio(request)
        assert result.npv_cube.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(result.npv_cube)))
        assert result.greeks is not None
        for key, val in result.greeks[1].items():  # swaption
            if key == "theta":
                continue
            assert jnp.asarray(val).dtype == jnp.float32

    def test_pricing_per_instrument_type_override(self):
        """With only bermudan_swaption=32, that bucket is float32 internally, but npv_cube
        takes the widest dtype present (`jnp.stack` promotion)."""
        from engine.portfolio import PrecisionConfig, PricingPrecisionOverride

        override = PricingPrecisionOverride(default=64, bermudan_swaption=32)
        request32 = self._request(precision=PrecisionConfig(pricing=override))
        request64 = self._request(precision=PrecisionConfig(pricing=64))
        result32 = price_portfolio(request32)
        result64 = price_portfolio(request64)

        assert result32.npv_cube.dtype == jnp.float64
        assert bool(jnp.all(jnp.isfinite(result32.npv_cube)))
        assert result32.base_npv == pytest.approx(result64.base_npv, rel=1e-3)

    @pytest.mark.slow
    def test_risk_per_greek_override(self):
        """theta=32 with default 64: delta_gamma and theta resolve to different dtypes
        (Delta/Gamma stay float64; theta is a Python float, so its resolution is checked
        directly)."""
        from engine.portfolio import PrecisionConfig, RiskPrecisionOverride
        from engine.portfolio.request import _resolve_risk_dtype

        override = RiskPrecisionOverride(default=64, theta=32)
        assert _resolve_risk_dtype(override, "delta_gamma") == jnp.float64
        assert _resolve_risk_dtype(override, "theta") == jnp.float32

        request = self._request(precision=PrecisionConfig(risk=override), compute_greeks=True)
        result = price_portfolio(request)
        assert result.greeks is not None
        for idx in (1, 2, 3):
            for key, val in result.greeks[idx].items():
                if key == "theta":
                    continue
                assert jnp.asarray(val).dtype == jnp.float64, f"trade {idx} greek {key!r} not float64"

    def test_risk_exposure_override_recasts_npv_cube_for_exposure_only(self):
        """exposure=32 casts only the copy fed to the exposure statistics; npv_cube keeps
        the pricing dtype."""
        from engine.portfolio import PrecisionConfig, RiskPrecisionOverride

        override = RiskPrecisionOverride(default=64, exposure=32)
        request = self._request(precision=PrecisionConfig(pricing=64, risk=override))
        result = price_portfolio(request)
        assert result.npv_cube.dtype == jnp.float64

        profiles = [result.exposure] + list(result.trade_exposures)
        for profile in profiles:
            arrays = [profile.epe, profile.ene, profile.ee_b, profile.eee_b] + list(profile.pfe.values())
            for arr in arrays:
                assert jnp.asarray(arr).dtype == jnp.float32

    def test_calibration_precision_flows_through_lgm_bootstrap(self):
        """calibration=32 vs 64 gives a different-dtype calibrated Sigma and close
        prices."""
        from dataclasses import replace as _replace
        from engine.portfolio import PrecisionConfig
        from engine.portfolio.request import _fill_calibrated_sigma
        from engine.calibration.basket import build_coterminal_basket
        from engine.models.hull_white import ZeroCurve as HwZeroCurve

        swap_cfg, swaption_cfg, bermudan_cfg, american_cfg = _build_trades()
        uncalibrated = _replace(bermudan_cfg, hw_sigma=None)
        trades = [uncalibrated]
        sim_config = _sim_config([swap_cfg, swaption_cfg, bermudan_cfg, american_cfg])

        curve64 = HwZeroCurve.from_config(uncalibrated.initial_zero_curve)
        targets = build_coterminal_basket(
            exercise_times=[time_from_reference(uncalibrated.evaluation_date, d) for d in uncalibrated.exercise_dates],
            final_maturity_time=3.0,
            notional=uncalibrated.notional, payer=uncalibrated.payer,
            market_vols=[0.01, 0.01], zero_curve=curve64,
            evaluation_date=uncalibrated.evaluation_date, index_tenor_months=3,
        )

        filled_64 = _fill_calibrated_sigma(trades, targets, sim_config, PrecisionConfig(calibration=64))
        filled_32 = _fill_calibrated_sigma(trades, targets, sim_config, PrecisionConfig(calibration=32))

        sigma_64 = filled_64[0].hw_sigma
        sigma_32 = filled_32[0].hw_sigma
        assert sigma_64.values.dtype == jnp.float64
        assert sigma_32.values.dtype == jnp.float32
        assert np.asarray(sigma_32.values) == pytest.approx(np.asarray(sigma_64.values), rel=1e-3)

    @pytest.mark.slow
    def test_precision_knobs_fully_independent_four_way(self):
        """All four knobs set independently at once, checked through the resolvers."""
        from engine.portfolio import PrecisionConfig, PricingPrecisionOverride, RiskPrecisionOverride
        from engine.portfolio.request import _resolve_pricing_dtype, _resolve_risk_dtype

        pricing = PricingPrecisionOverride(default=64, bermudan_swaption=32)
        risk = RiskPrecisionOverride(default=64, vega=32)
        precision = PrecisionConfig(simulation=64, pricing=pricing, risk=risk, calibration=32)

        assert precision.simulation == 64
        assert precision.calibration == 32
        assert _resolve_pricing_dtype(precision.pricing, BermudanSwaptionConfig) == jnp.float32
        assert _resolve_pricing_dtype(precision.pricing, SwapConfig) == jnp.float64
        assert _resolve_risk_dtype(precision.risk, "vega") == jnp.float32
        assert _resolve_risk_dtype(precision.risk, "delta_gamma") == jnp.float64

        request = self._request(precision=precision, compute_greeks=True)
        result = price_portfolio(request)
        assert bool(jnp.all(jnp.isfinite(result.npv_cube)))
        assert np.isfinite(result.base_npv)


class TestPrecisionOverrideValidation:
    """The override classes validate every field as `PrecisionConfig` does."""

    def test_pricing_override_rejects_invalid_bits(self):
        from engine.portfolio import PricingPrecisionOverride
        with pytest.raises(ValueError):
            PricingPrecisionOverride(swap=48)

    def test_risk_override_rejects_invalid_bits(self):
        from engine.portfolio import RiskPrecisionOverride
        with pytest.raises(ValueError):
            RiskPrecisionOverride(vega=48)


class TestPricePortfolioConcurrency:
    """Two requests at different precisions on two threads each get their own dtype and
    values. `generate_paths` toggles the process-global `jax_enable_x64`, so without
    `_PRICING_LOCK` one thread could corrupt the other mid-flight. A `threading.Barrier`
    forces overlap, and the body repeats to make a race likely."""

    NUM_REPETITIONS = 8

    def _make_request(self, precision):
        swap_cfg, swaption_cfg, bermudan_cfg, american_cfg = _build_trades()
        trades = [swap_cfg, swaption_cfg]
        sim_config = _sim_config([swap_cfg, swaption_cfg, bermudan_cfg, american_cfg])
        return PortfolioRequest(market=sim_config, trades=trades, precision=precision)

    def test_two_different_precisions_concurrently_each_get_their_own_dtype(self):
        from engine.portfolio import PrecisionConfig
        import threading

        precision_a = PrecisionConfig(simulation=64, pricing=64, risk=64)
        precision_b = PrecisionConfig(simulation=32, pricing=32, risk=32)

        # Sequential reference results, outside any threading, to catch wrong values as well
        # as wrong dtypes.
        ref_a = price_portfolio(self._make_request(precision_a))
        ref_b = price_portfolio(self._make_request(precision_b))

        for rep in range(self.NUM_REPETITIONS):
            barrier = threading.Barrier(2)
            results = {}
            errors = {}

            def run(key, precision):
                try:
                    barrier.wait(timeout=30)
                    results[key] = price_portfolio(self._make_request(precision))
                except Exception as exc:  # pragma: no cover - failure path
                    errors[key] = exc

            t_a = threading.Thread(target=run, args=("a", precision_a))
            t_b = threading.Thread(target=run, args=("b", precision_b))
            t_a.start()
            t_b.start()
            t_a.join(timeout=60)
            t_b.join(timeout=60)

            assert not errors, f"rep {rep}: concurrent price_portfolio raised: {errors}"
            assert "a" in results and "b" in results, f"rep {rep}: a thread failed to complete"

            result_a, result_b = results["a"], results["b"]
            assert result_a.npv_cube.dtype == jnp.float64, f"rep {rep}: thread A got the wrong dtype"
            assert result_b.npv_cube.dtype == jnp.float32, f"rep {rep}: thread B got the wrong dtype"

            np.testing.assert_allclose(
                np.asarray(result_a.npv_cube), np.asarray(ref_a.npv_cube), rtol=1e-9,
                err_msg=f"rep {rep}: thread A's values diverged from the sequential reference",
            )
            np.testing.assert_allclose(
                np.asarray(result_b.npv_cube), np.asarray(ref_b.npv_cube), rtol=1e-3,
                err_msg=f"rep {rep}: thread B's values diverged from the sequential reference",
            )
            np.testing.assert_allclose(
                result_a.base_npv, ref_a.base_npv, rtol=1e-9,
                err_msg=f"rep {rep}: thread A's base_npv diverged from the sequential reference",
            )
            np.testing.assert_allclose(
                result_b.base_npv, ref_b.base_npv, rtol=1e-3,
                err_msg=f"rep {rep}: thread B's base_npv diverged from the sequential reference",
            )


class TestSingleEvaluationDate:
    """The simulation has one t=0; a trade dated differently would be
    priced on a shifted time axis. Refused before any pricing runs."""

    def test_mixed_evaluation_dates_are_refused(self):
        import dataclasses

        from engine.portfolio import validate_portfolio_against_simulation

        trades = _build_trades()
        shifted = dataclasses.replace(trades[1], evaluation_date=trades[1].evaluation_date + 1)
        mixed = [trades[0], shifted] + list(trades[2:])
        with pytest.raises(ValueError, match="one evaluation_date"):
            validate_portfolio_against_simulation(_sim_config(trades), mixed)

    def test_shared_evaluation_date_is_accepted(self):
        from engine.portfolio import validate_portfolio_against_simulation

        trades = _build_trades()
        validate_portfolio_against_simulation(_sim_config(trades), trades)
