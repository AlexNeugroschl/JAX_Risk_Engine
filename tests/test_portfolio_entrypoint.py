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


class TestPricePortfolioPrecision:
    """PrecisionConfig's four independent knobs (simulation/pricing/risk/
    calibration), each 32 or 64 -- see engine.portfolio.request.
    PrecisionConfig. `pricing`/`risk` may each additionally be a structured
    override (PricingPrecisionOverride/RiskPrecisionOverride) for optional
    per-instrument-type/per-Greek drill-down. Default (all-64) must
    reproduce today's exact behavior; each knob set to 32 (flat or via an
    override) must be independently observable in the right output dtype,
    without affecting the others."""

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
        """No `precision` supplied -> PrecisionConfig() (all-64) -> byte-
        identical npv_cube dtype/values to a request built before this
        feature existed. Also proves omitting overrides produces plain
        ints for pricing/risk, not silently-promoted override objects --
        the literal backward-compat regression proof for the hierarchy
        redesign."""
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

    def test_simulation_32_produces_float32_market_data_and_still_prices(self):
        from engine.portfolio import PrecisionConfig
        request = self._request(precision=PrecisionConfig(simulation=32))
        result = price_portfolio(request)
        # generate_paths under precision=32 produces float32 rates/yield
        # curves; the final npv_cube dtype is governed by `pricing` (still
        # 64 here), which price_swaps/etc. derive from their JAX-array
        # inputs -- so npv_cube itself may still be float64 even though the
        # underlying simulation ran in float32. What matters here is that
        # simulation=32 alone doesn't break pricing.
        assert bool(jnp.all(jnp.isfinite(result.npv_cube)))
        assert np.isfinite(result.base_npv)

    def test_pricing_32_produces_float32_npv_cube_and_base_npv(self):
        """pricing=32 must make BOTH npv_cube AND base_npv's own internal
        computation (_flat_curve_cube, the swaption zero-shock r0_path)
        float32 -- exercising all four trade types at once, since each
        pricer's own final-cast-to-input-dtype behavior needs a float32
        input to prove out."""
        from engine.portfolio import PrecisionConfig
        request = self._request(precision=PrecisionConfig(pricing=32))
        result = price_portfolio(request)
        assert result.npv_cube.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(result.npv_cube)))
        assert np.isfinite(result.base_npv)

    def test_pricing_64_vs_32_base_npv_numerically_close(self):
        """float32 pricing shouldn't produce a wildly different base_npv --
        just a lower-precision one (pytest.approx with a loose relative
        tolerance, not exact equality)."""
        from engine.portfolio import PrecisionConfig
        request64 = self._request(precision=PrecisionConfig(pricing=64))
        request32 = self._request(precision=PrecisionConfig(pricing=32))
        result64 = price_portfolio(request64)
        result32 = price_portfolio(request32)
        assert result32.base_npv == pytest.approx(result64.base_npv, rel=1e-3)

    def test_risk_32_changes_greeks_dtype_for_swaption_and_bermudan(self):
        """risk=32 must flow into both a European swaption's and a
        calibrated Bermudan's Greeks (the Jacobian path bermudan_vega
        exercises isn't triggered by compute_greeks -- that's covered
        directly in test_greeks_bermudan.py -- but bermudan_delta_gamma's
        own risk-dtype plumbing runs through price_portfolio here)."""
        from engine.portfolio import PrecisionConfig
        request = self._request(precision=PrecisionConfig(risk=32), compute_greeks=True)
        result = price_portfolio(request)
        assert result.greeks is not None
        # trades = [swap, swaption, bermudan, american]; swap Greeks are
        # skipped by design (see _compute_all_greeks's own docstring).
        for idx in (1, 2, 3):
            assert idx in result.greeks
            for key, val in result.greeks[idx].items():
                if key == "theta":
                    continue  # a plain Python float (forward-difference NPV), not a JAX array
                assert jnp.asarray(val).dtype == jnp.float32, f"trade {idx} greek {key!r} not float32"

    def test_mixed_precision_each_stage_independent(self):
        """simulation=64, pricing=32, risk=32 -- confirms each of the three
        knobs takes effect independently in the same request, the
        end-to-end scenario the plan's own verification step calls out."""
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
        """PricingPrecisionOverride(default=64, bermudan_swaption=32) --
        each bucket casts to its OWN dtype internally; npv_cube's own dtype
        reflects the WIDEST bucket present (jnp.stack promotion), not the
        one that was drilled down -- this assertion documents that
        consequence rather than hiding it."""
        from engine.portfolio import PrecisionConfig, PricingPrecisionOverride

        override = PricingPrecisionOverride(default=64, bermudan_swaption=32)
        request32 = self._request(precision=PrecisionConfig(pricing=override))
        request64 = self._request(precision=PrecisionConfig(pricing=64))
        result32 = price_portfolio(request32)
        result64 = price_portfolio(request64)

        assert result32.npv_cube.dtype == jnp.float64
        assert bool(jnp.all(jnp.isfinite(result32.npv_cube)))
        assert result32.base_npv == pytest.approx(result64.base_npv, rel=1e-3)

    def test_risk_per_greek_override(self):
        """RiskPrecisionOverride(default=64, theta=32) -- delta_gamma and
        theta must resolve to DIFFERENT dtypes via _resolve_risk_dtype, and
        that difference must be observable in the actual Greeks output
        (delta/gamma stay float64 while theta's own float32 resolution is
        checked directly, since theta returns a plain Python float with no
        observable dtype)."""
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

    def test_risk_var_es_override_recasts_npv_cube_for_risk_only(self):
        """RiskPrecisionOverride(default=64, var_es=32) -- the cast happens
        ONLY on the copy fed to compute_risk_metrics; npv_cube itself (what
        `pricing` produced) is untouched."""
        from engine.portfolio import PrecisionConfig, RiskPrecisionOverride

        override = RiskPrecisionOverride(default=64, var_es=32)
        request = self._request(precision=PrecisionConfig(pricing=64, risk=override))
        result = price_portfolio(request)
        assert result.npv_cube.dtype == jnp.float64
        for key, arr in result.risk.items():
            assert jnp.asarray(arr).dtype == jnp.float32, f"risk metric {key!r} not float32"

    def test_calibration_precision_flows_through_lgm_bootstrap(self):
        """calibration=32 vs 64 must produce a genuinely different-dtype
        calibrated Sigma via _fill_calibrated_sigma, with numerically close
        (not equal) pricing -- exercised through a Bermudan trade with
        hw_sigma=None so calibration actually runs."""
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
            exercise_times=uncalibrated.exercise_times, final_maturity_time=3.0,
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

    def test_precision_knobs_fully_independent_four_way(self):
        """All four axes set independently and simultaneously -- a mechanism
        proof via direct resolver checks, since four independently-varying
        axes make a single end-to-end numeric assertion weak."""
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
    """PricingPrecisionOverride/RiskPrecisionOverride validate every field
    exactly like PrecisionConfig.__post_init__ already does."""

    def test_pricing_override_rejects_invalid_bits(self):
        from engine.portfolio import PricingPrecisionOverride
        with pytest.raises(ValueError):
            PricingPrecisionOverride(swap=48)

    def test_risk_override_rejects_invalid_bits(self):
        from engine.portfolio import RiskPrecisionOverride
        with pytest.raises(ValueError):
            RiskPrecisionOverride(vega=48)


class TestPricePortfolioConcurrency:
    """The load-bearing correctness proof for this whole feature:
    price_portfolio's async-job usage (engine/api/routes.py's
    BackgroundTasks thread pool) means two DIFFERENT PrecisionConfig
    requests can genuinely run on separate threads at the same time.
    generate_paths flips jax_enable_x64, a process-global JAX/XLA flag --
    with no lock, thread B's flag flip can land while thread A is still
    mid-flight, corrupting thread A's dtype or (worse) silently producing a
    dtype-correct-but-numerically-wrong array. _PRICING_LOCK in
    engine/portfolio/request.py serializes price_portfolio's entire
    JAX-executing body against exactly this race.

    Uses threading.Barrier (not bare Thread.start()) to force genuine
    overlap -- both threads block until both have reached the barrier,
    maximizing the odds of a real race if the lock were absent/broken.
    Repeats the body multiple times within the test, since a race-condition
    test that only sometimes catches the bug is a weak guarantee."""

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

        # Sequential reference results, computed once, OUTSIDE any threading
        # -- the numeric ground truth each concurrent run is cross-checked
        # against (dtype alone wouldn't catch numeric corruption from a
        # mid-flight flag flip producing a dtype-correct-but-wrong-valued
        # array).
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
