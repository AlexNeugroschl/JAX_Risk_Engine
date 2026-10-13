"""
Tests for engine.risk.market: risk factors, shock scenarios, revaluation and
the VaR/ES run. ORE parity is in tests/test_market_risk_ore_parity.py.
"""
import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from engine.market_data.market import ZeroCurveConfig
from engine.precision import Precision, StagePrecision
from engine.pricing.config import PricingConfig
from engine.pricing.cube import value_today
from engine.risk.greeks.ad import curve_greeks
from engine.risk.greeks.price_functions import curves_of, trade_price_function
from engine.risk.market import (
    MarketRiskRequest,
    SOURCE_HISTORICAL,
    RateRiskFactors,
    ShockScenarios,
    covariance_from_history,
    historical_scenarios,
    horizon_moves,
    monte_carlo_scenarios,
    run_market_risk,
)
from engine.risk.market.revaluation import revalue, revalue_trade
from engine.risk.market.var_es import RISK_MEASURE_HISTORICAL, compute_risk_metrics
from tests import market_risk_support as m

FACTORS = m.factors()


def _scenarios(num=256, seed=3, horizon_days=10):
    return monte_carlo_scenarios(FACTORS, m.covariance(horizon_days=horizon_days), horizon_days, num, seed=seed)


# ---------------------------------------------------------------------------
# Risk factors
# ---------------------------------------------------------------------------
class TestRateRiskFactors:
    def test_vector_layout(self):
        n = len(m.PILLAR_TIMES)
        assert FACTORS.sizes == (n, n)
        assert FACTORS.size == 2 * n
        assert FACTORS.slice_of(1) == slice(n, 2 * n)
        np.testing.assert_array_equal(FACTORS.base_rates()[FACTORS.slice_of(1)], m.IBOR.rates)

    def test_labels_name_curve_and_pillar(self):
        labels = FACTORS.labels()
        assert labels[0] == "discount:USD/0y" and labels[1] == "discount:USD/1y"
        assert labels[len(m.PILLAR_TIMES)] == f"index:{m.INDEX}/0y"

    def test_from_market_takes_every_curve_by_its_name(self):
        """Each currency's discount curve, then its index curves: the names trades read."""
        assert FACTORS.names == ("discount:USD", f"index:{m.INDEX}")
        assert FACTORS.curves == (m.OIS, m.IBOR)
        assert FACTORS.index_of(f"index:{m.INDEX}") == 1
        with pytest.raises(KeyError, match="not a risk factor"):
            FACTORS.index_of("discount:EUR")

    def test_default_names(self):
        assert RateRiskFactors.from_curves([m.OIS]).names == ("curve0",)

    @pytest.mark.parametrize("curves, names, match", [
        ([], [], "at least one curve"),
        ([m.OIS], ["a", "b"], "names for"),
        ([m.OIS, m.IBOR], ["a", "a"], "unique"),
        ([ZeroCurveConfig([0.0, 2.0, 1.0], [0.01] * 3)], ["x"], "strictly increasing"),
        ([ZeroCurveConfig([0.0, 1.0], [0.01])], ["x"], "same length"),
        ([ZeroCurveConfig([0.0, 1.0], [0.01, float("nan")])], ["x"], "finite"),
    ])
    def test_rejects_malformed_curves(self, curves, names, match):
        with pytest.raises(ValueError, match=match):
            RateRiskFactors.from_curves(curves, names)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
class TestMonteCarloScenarios:
    def test_sample_covariance_matches_the_target(self):
        target = m.covariance()
        shifts = _scenarios(num=2 ** 14).shifts
        sample = np.cov(shifts, rowvar=False)
        scale = np.max(np.diag(target))
        assert np.max(np.abs(sample - target)) < 0.03 * scale

    def test_moves_are_centred(self):
        shifts = _scenarios(num=2 ** 14).shifts
        assert np.max(np.abs(shifts.mean(axis=0))) < 0.02 * np.sqrt(np.max(np.diag(m.covariance())))

    def test_seed_reproduces_and_varies(self):
        np.testing.assert_array_equal(_scenarios(seed=5).shifts, _scenarios(seed=5).shifts)
        assert not np.allclose(_scenarios(seed=5).shifts, _scenarios(seed=6).shifts)

    def test_singular_covariance_is_accepted(self):
        """Perfectly correlated pillars make the covariance singular; the
        eigendecomposition factor handles it where Cholesky would not."""
        vol = 0.003
        singular = np.full((FACTORS.size, FACTORS.size), vol ** 2)
        shifts = monte_carlo_scenarios(FACTORS, singular, 10, 64, seed=1).shifts
        np.testing.assert_allclose(shifts, np.repeat(shifts[:, :1], FACTORS.size, axis=1), atol=1e-15)

    def test_labels_and_measure(self):
        scenarios = _scenarios()
        assert scenarios.source == "monte-carlo"
        assert scenarios.measure == RISK_MEASURE_HISTORICAL
        assert scenarios.windows is None

    @pytest.mark.parametrize("mutate, match", [
        (lambda c: c[:-1, :-1], r"\[20, 20\]"),
        (lambda c: c + np.triu(np.full_like(c, 1e-6), 1), "symmetric"),
        (lambda c: c - np.eye(len(c)) * 1e-3, "positive semi-definite"),
    ], ids=["shape", "asymmetric", "not-psd"])
    def test_rejects_an_invalid_covariance(self, mutate, match):
        with pytest.raises(ValueError, match=match):
            monte_carlo_scenarios(FACTORS, mutate(m.covariance()), 10, 64)


class TestHistoricalScenarios:
    HISTORY = np.cumsum(np.arange(1, 13, dtype=np.float64)[:, None] * np.ones((1, 2)), axis=0) * 1e-4

    def test_moves_are_overlapping_differences(self):
        moves = horizon_moves(self.HISTORY, 3)
        assert moves.shape == (9, 2)
        np.testing.assert_allclose(moves[0], self.HISTORY[3] - self.HISTORY[0])
        np.testing.assert_allclose(moves[-1], self.HISTORY[11] - self.HISTORY[8])

    def test_windows_are_labelled(self):
        factors = RateRiskFactors.from_curves([ZeroCurveConfig([0.0, 1.0], [0.01, 0.02])])
        dates = [f"2026-01-{d:02d}" for d in range(1, 13)]
        scenarios = historical_scenarios(factors, self.HISTORY, 3, dates=dates)
        assert scenarios.source == "historical"
        assert scenarios.windows[0] == ("2026-01-01", "2026-01-04")
        assert len(scenarios.windows) == scenarios.num_scenarios == 9

    def test_covariance_from_history_is_the_sample_covariance_of_the_moves(self):
        rng = np.random.default_rng(0)
        history = np.cumsum(rng.normal(0, 1e-4, size=(200, 3)), axis=0)
        expected = np.cov(history[10:] - history[:-10], rowvar=False, ddof=1)
        np.testing.assert_allclose(covariance_from_history(history, 10), expected)

    def test_covariance_of_one_factor_is_a_matrix(self):
        history = np.linspace(0.01, 0.02, 30)[:, None]
        assert covariance_from_history(history, 5).shape == (1, 1)

    @pytest.mark.parametrize("history, horizon, match", [
        (np.zeros((5,)), 1, "dates, factors"),
        (np.zeros((4, 2)), 3, "fewer than two"),
        (np.zeros((10, 2)), 0, "at least 1"),
    ])
    def test_rejects_unusable_history(self, history, horizon, match):
        with pytest.raises(ValueError, match=match):
            horizon_moves(history, horizon)

    def test_rejects_a_history_for_other_factors(self):
        with pytest.raises(ValueError, match="factors"):
            historical_scenarios(FACTORS, np.zeros((30, 3)), 10)


class TestShockScenariosValidation:
    @pytest.mark.parametrize("shifts, kwargs, match", [
        (np.zeros((4, 3)), {}, "shifts must be"),
        (np.zeros((1, 20)), {}, "at least two"),
        (np.full((4, 20), np.inf), {}, "finite"),
        (np.zeros((4, 20)), {"horizon_days": 0}, "horizon_days"),
        (np.zeros((4, 20)), {"source": "guess"}, "source"),
    ])
    def test_rejects(self, shifts, kwargs, match):
        args = dict(factors=FACTORS, shifts=shifts, horizon_days=10, source="monte-carlo")
        args.update(kwargs)
        with pytest.raises(ValueError, match=match):
            ShockScenarios(**args)


# ---------------------------------------------------------------------------
# Revaluation
# ---------------------------------------------------------------------------
class TestRevaluation:
    def test_zero_shift_is_zero_pnl(self):
        trades = [m.swap(), m.european(), m.bermudan(), m.bond()]
        base, shocked = revalue(trades, m.market(), FACTORS, np.zeros((3, FACTORS.size)), m.PRICING)
        np.testing.assert_allclose(np.asarray(shocked), np.tile(base, (3, 1)), rtol=1e-13)

    def test_the_base_is_the_portfolio_runs_value(self):
        """One pricer: the base equals `value_today` with the same engines."""
        trades = [m.swap(), m.european(), m.bermudan(), m.bond()]
        base, _ = revalue(trades, m.market(), FACTORS, np.zeros((2, FACTORS.size)), m.PRICING)
        np.testing.assert_allclose(base, value_today(trades, m.market(), "USD", m.PRICING), rtol=1e-13)

    def test_batching_does_not_change_the_answer(self):
        trades = [m.swap(), m.bermudan()]
        shifts = _scenarios(num=20).shifts
        _, one_by_one = revalue(trades, m.market(), FACTORS, shifts, m.PRICING, batch_size=1)
        _, batched = revalue(trades, m.market(), FACTORS, shifts, m.PRICING, batch_size=7)
        np.testing.assert_allclose(np.asarray(one_by_one), np.asarray(batched), rtol=1e-13)

    def test_a_parallel_shift_is_the_value_on_the_shifted_market(self):
        bond = m.bond()
        shift = np.zeros((2, FACTORS.size))
        shift[:, FACTORS.slice_of(0)] = 0.0025
        _, shocked = revalue([bond], m.market(), FACTORS, shift, m.PRICING)
        moved = ZeroCurveConfig(m.PILLAR_TIMES, [r + 0.0025 for r in m.OIS.rates])
        market = m.market()
        usd = dataclasses.replace(market.currency("USD"), discount_curve=moved)
        expected = value_today([bond], dataclasses.replace(market, currencies={"USD": usd}), "USD")[0]
        assert float(shocked[0, 0]) == pytest.approx(expected, rel=1e-14)

    def test_swap_pnl_is_first_order_its_delta(self):
        """A 0.1bp parallel move of the discount curve: P&L equals the summed AD discount
        Delta (per 1bp) scaled by 0.1, to second order."""
        swap = m.swap()
        shift = np.zeros((2, FACTORS.size))
        shift[:, FACTORS.slice_of(0)] = 1e-5
        base, shocked = revalue([swap], m.market(), FACTORS, shift, m.PRICING)
        pnl = float(shocked[0, 0]) - base[0]
        delta = curve_greeks(swap, m.market(), m.PRICING, 1e-4)["delta:discount:USD"]
        assert pnl == pytest.approx(0.1 * float(np.sum(delta)), rel=1e-4)

    def test_the_european_engine_is_the_configured_one(self):
        """Bachelier by default, Jamshidian when configured: not chosen by the trade (A-8)."""
        trades = [m.european()]
        zero = np.zeros((2, FACTORS.size))
        bachelier, _ = revalue(trades, m.market(), FACTORS, zero, m.PRICING)
        jamshidian, _ = revalue(trades, m.market(), FACTORS, zero, m.JAMSHIDIAN)
        assert bachelier[0] == pytest.approx(value_today(trades, m.market(), "USD")[0], rel=1e-13)
        assert jamshidian[0] == pytest.approx(value_today(trades, m.market(), "USD", m.JAMSHIDIAN)[0], rel=1e-13)
        assert bachelier[0] != pytest.approx(jamshidian[0], rel=1e-3)

    def test_grid_pricers_get_a_memory_bounded_batch(self):
        """A fine-grid Bermudan must not be vmapped hundreds of scenarios at a time: at
        n_per_std=64 one scenario's rollback holds ~80 MB, and a batch of 256 thrashed a
        32 GB machine."""
        from engine.risk.market.revaluation import BATCH_MEMORY_BUDGET, scenario_batch_size

        fine = PricingConfig(bermudan=dataclasses.replace(m.ENGINE, n_per_std=64, std_devs=6.0))
        batch = scenario_batch_size(m.bermudan(), fine, 256, itemsize=8)
        assert 1 <= batch < 16
        nodes = 2 * int(64 * 6.0 + 0.5) + 1
        assert batch * nodes * nodes * 8 <= BATCH_MEMORY_BUDGET
        assert scenario_batch_size(m.swap(), fine, 256, itemsize=8) == 256
        assert scenario_batch_size(m.bermudan(), m.PRICING, 64, itemsize=8) == 64

    @pytest.mark.parametrize("pricing", [m.PRICING, m.JAMSHIDIAN], ids=["bachelier", "jamshidian"])
    def test_the_european_price_function_keeps_float32(self, pricing):
        """Regression: dtype-less constants inside a European price function promoted a
        float32 curve to float64, so float32 market risk silently ran partly in float64."""
        fn32 = trade_price_function(m.european(), m.market(), pricing, jnp.float32)
        value32 = fn32.price(*curves_of(fn32, m.market(), jnp.float32))
        assert value32.dtype == jnp.float32
        fn64 = trade_price_function(m.european(), m.market(), pricing, jnp.float64)
        assert float(value32) == pytest.approx(float(fn64.price(*curves_of(fn64, m.market(), jnp.float64))), rel=1e-5)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mixed_run():
    trades = [m.swap(), m.european(), m.bermudan(), m.bond()]
    return trades, run_market_risk(MarketRiskRequest(trades, m.market(), _scenarios(num=512), m.PRICING))


class TestRun:
    @pytest.mark.slow
    def test_statistics_are_var_es_of_the_portfolio_pnl(self, mixed_run):
        _, result = mixed_run
        expected = compute_risk_metrics(np.asarray(result.pnl)[:, None, :], 0.0, percentiles=(0.99, 0.975))
        assert set(result.risk) == set(expected)
        for key, value in expected.items():
            assert result.risk[key] == pytest.approx(float(np.asarray(value)[0]), rel=1e-12)

    def test_basel_quantile_is_labelled_97_5(self, mixed_run):
        _, result = mixed_run
        assert {"VaR_99", "ES_99", "VaR_97.5", "ES_97.5"} <= set(result.risk)

    def test_portfolio_pnl_is_the_sum_of_trade_pnl(self, mixed_run):
        _, result = mixed_run
        np.testing.assert_allclose(np.asarray(result.portfolio_pnl), np.asarray(result.pnl).sum(axis=1), rtol=1e-12)
        assert result.base_npv == pytest.approx(sum(result.base_npv_per_trade))

    def test_describes_itself(self, mixed_run):
        _, result = mixed_run
        assert result.measure == RISK_MEASURE_HISTORICAL
        assert result.source == "monte-carlo"
        assert result.horizon_days == 10
        assert result.num_scenarios == 512
        assert result.risk_factors == FACTORS.labels()

    def test_es_exceeds_var(self, mixed_run):
        _, result = mixed_run
        assert result.risk["ES_99"] >= result.risk["VaR_99"] > 0.0

    def test_warns_that_option_vol_is_not_shocked(self, mixed_run):
        _, result = mixed_run
        assert any("volatility risk is not in this VaR/ES" in w and "['european', 'bermudan']" in w
                   for w in result.warnings)
        assert not any("I-112" in w for w in result.warnings), "an uncalibrated LGM is ORE's in every scenario"

    def test_warns_that_a_calibrated_lgm_is_held_where_ore_recalibrates(self):
        """I-112: ORE's historical P&L recalibrates a calibrated LGM to each scenario's curves;
        the revaluation holds today's calibration, and the result says so."""
        calibrated = PricingConfig(bermudan=dataclasses.replace(m.ENGINE, calibration="Bootstrap"))
        result = run_market_risk(MarketRiskRequest([m.bermudan()], m.market(), _scenarios(num=64), calibrated,
                                                   quantiles=(0.99,)))
        assert any("I-112" in w and "['bermudan']" in w for w in result.warnings)

    def test_warns_about_a_thin_tail(self):
        result = run_market_risk(MarketRiskRequest([m.swap()], m.market(), _scenarios(num=64), quantiles=(0.99,)))
        assert any("too few for a stable estimate" in w for w in result.warnings)

    @pytest.mark.slow
    def test_float32_run_is_float32_and_close(self, mixed_run):
        """Float32 throughout: the revaluation and P&L are computed and stored in float32 (the
        float32 revaluation's P&L, bit for bit: in difference form for the swaps, the bond and
        the European, the shocked values less the base for the Bermudan) and VaR/ES reduce them
        in float64."""
        trades, result64 = mixed_run
        scenarios = _scenarios(num=512)
        result32 = run_market_risk(MarketRiskRequest(trades, m.market(), scenarios, m.PRICING,
                                                     precision=Precision.throughout("float32")))
        moves = jnp.asarray(scenarios.shifts, jnp.float32)
        pnl = [revalue_trade(cfg, m.market(), FACTORS, moves, m.PRICING)[1] for cfg in trades]
        assert all(p.dtype == jnp.float32 for p in pnl) and result32.pnl.dtype == jnp.float64
        np.testing.assert_array_equal(np.asarray(result32.pnl), np.stack([np.asarray(p, np.float64) for p in pnl], -1))
        assert result32.risk["VaR_99"] == pytest.approx(result64.risk["VaR_99"], rel=1e-4)

    @pytest.mark.parametrize("precision", [Precision(), Precision.throughout("float32"),
                                           Precision(pricing=StagePrecision("float32", "float64", "float64"))])
    def test_a_zero_shift_is_zero_pnl_at_every_precision(self, precision):
        """The base is the revaluation at the pricing compute, so P&L has no precision bias at
        zero."""
        zero = ShockScenarios(FACTORS, np.zeros((4, FACTORS.size)), 10, SOURCE_HISTORICAL)
        result = run_market_risk(MarketRiskRequest([m.swap(), m.bond()], m.market(), zero, m.PRICING,
                                                   precision=precision))
        assert np.all(np.asarray(result.pnl) == 0.0)

    def test_storage_narrower_than_compute_rounds_only_the_stored_pnl(self):
        """`pricing` stored float32 and computed float64: the float64 P&L rounded once."""
        scenarios = _scenarios(num=64)
        exact = run_market_risk(MarketRiskRequest([m.swap()], m.market(), scenarios))
        stored = run_market_risk(MarketRiskRequest([m.swap()], m.market(), scenarios, precision=Precision(
            pricing=StagePrecision("float32", "float64", "float64"))))
        np.testing.assert_array_equal(np.asarray(stored.pnl), np.asarray(exact.pnl).astype(np.float32))

    @pytest.mark.slow
    def test_a_calibrated_bermudan_is_held_at_todays_calibration(self):
        """With the default engine the Bermudan is calibrated on today's market and that
        model held under every scenario: the base is its portfolio value, and a zero shock
        is zero P&L."""
        cfg = m.bermudan()
        shocks = _scenarios(num=16)
        result = run_market_risk(MarketRiskRequest([cfg], m.market(), shocks))
        assert result.base_npv_per_trade[0] == pytest.approx(value_today([cfg], m.market(), "USD")[0], rel=1e-12)
        assert np.all(np.isfinite(np.asarray(result.pnl)))


class TestRunValidation:
    def _request(self, trades, **kwargs):
        return MarketRiskRequest(trades, m.market(), _scenarios(num=16), **kwargs)

    def test_no_trades(self):
        with pytest.raises(ValueError, match="at least one trade"):
            run_market_risk(self._request([]))

    @pytest.mark.parametrize("old", [32, 64])
    def test_the_retired_integer_precision_is_refused_naming_the_replacement(self, old):
        """Refused, not translated (decision A-12)."""
        with pytest.raises(TypeError, match=r"MarketRiskRequest\.precision.*retired on 2026-10-01"):
            run_market_risk(self._request([m.swap()], precision=old))

    @pytest.mark.parametrize("kwargs, match", [
        ({"batch_size": 0}, "batch_size"),
        ({"quantiles": (1.0,)}, "quantile"),
        ({"quantiles": (float("nan"),)}, "quantile"),
        ({"quantiles": (0.99, 0.975, 0.99)}, r"quantiles: 0\.99 is given more than once"),
    ])
    def test_bad_settings(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            run_market_risk(self._request([m.swap()], **kwargs))

    def test_close_quantiles_keep_their_own_figures(self):
        """I-102: 0.99 and 0.9900000001 were both keyed `99`, the second overwriting the first in
        `risk` and in the precision report's figures."""
        close = (0.99, 0.9900000001)
        result = run_market_risk(self._request([m.swap()], quantiles=close, precision=Precision(paired_fraction=0.5)))
        for label in ("99", "99.00000001"):
            assert {f"VaR_{label}", f"ES_{label}", f"ES_{label}_tailCount", f"ES_{label}_standardError"} <= set(result.risk)
        assert {"portfolio/VaR_99", "portfolio/VaR_99.00000001"} <= set(result.precision.figures)

    def test_a_curve_the_market_lacks_is_refused_naming_the_trade(self):
        eur = dataclasses.replace(m.swap(), currency="EUR")
        with pytest.raises(KeyError, match="'swap'.*EUR"):
            run_market_risk(self._request([eur]))

    def test_a_factor_must_be_the_market_curve_of_its_name(self):
        other = ZeroCurveConfig(m.PILLAR_TIMES, [r + 0.001 for r in m.OIS.rates])
        factors = RateRiskFactors.from_curves([other, m.IBOR], FACTORS.names)
        shocks = monte_carlo_scenarios(factors, m.covariance(), 10, 16)
        with pytest.raises(ValueError, match="differs from the market's curve"):
            run_market_risk(MarketRiskRequest([m.swap()], m.market(), shocks))

    def test_a_factor_must_be_a_market_curve(self):
        factors = RateRiskFactors.from_curves([m.OIS, m.IBOR], ["discount:USD", "IBOR"])
        shocks = monte_carlo_scenarios(factors, m.covariance(), 10, 16)
        with pytest.raises(ValueError, match="'IBOR' is not a curve of the market"):
            run_market_risk(MarketRiskRequest([m.bond()], m.market(), shocks))

    def test_every_curve_a_trade_reads_must_be_a_factor(self):
        factors = RateRiskFactors.from_curves([m.OIS], ["discount:USD"])
        shocks = monte_carlo_scenarios(factors, m.covariance()[:10, :10], 10, 16)
        run_market_risk(MarketRiskRequest([m.bond()], m.market(), shocks))  # a bond reads the discount curve only
        with pytest.raises(ValueError, match="'swap'.*not a risk factor"):
            run_market_risk(MarketRiskRequest([m.swap()], m.market(), shocks))

    def test_a_bachelier_european_needs_swaption_vols(self):
        market = m.market()
        usd = dataclasses.replace(market.currency("USD"), swaption_vols=None)
        shocks = _scenarios(num=16)
        with pytest.raises(KeyError, match="swaption volatilities"):
            run_market_risk(MarketRiskRequest([m.european()], dataclasses.replace(market, currencies={"USD": usd}),
                                              shocks))

    def test_one_evaluation_date(self):
        shifted = dataclasses.replace(m.swap(), evaluation_date=m.TODAY + 1, trade_id="later")
        with pytest.raises(ValueError, match="'later'.*not the market's as-of date"):
            run_market_risk(self._request([m.swap(), shifted]))

    def test_a_reporting_currency_the_market_lacks_is_refused(self):
        with pytest.raises(KeyError, match="base_currency='EUR': no market for currency 'EUR'"):
            run_market_risk(self._request([m.swap()], base_currency="EUR"))


# ---------------------------------------------------------------------------
# The reporting currency (I-100)
# ---------------------------------------------------------------------------
def _two_currency_run(trades, fx_spots, num=256, **kwargs):
    market = m.two_currency_market(fx_spots)
    shocks = monte_carlo_scenarios(RateRiskFactors.from_market(market), m.two_currency_covariance(), 10, num, seed=5)
    return run_market_risk(MarketRiskRequest(trades, market, shocks, m.PRICING, **kwargs))


@pytest.fixture(scope="module")
def alone():
    """Each swap's run on its own, in its own currency."""
    return {cfg.trade_id: _two_currency_run([cfg], {}, base_currency=cfg.currency) for cfg in (m.swap(), m.eur_swap())}


class TestReportingCurrency:
    """Every amount of a run is in its reporting currency, `base_currency` (USD by default): a
    trade in another currency is revalued in its own and converted at the as-of FX spot. Until
    2026-10-11 a EUR trade's P&L was added to a USD trade's as if both were dollars."""

    @pytest.mark.parametrize("spot", [1.10, 1.30])
    def test_a_two_currency_var_moves_with_the_spot_as_the_converted_pnls_say(self, alone, spot):
        result = _two_currency_run([m.swap(), m.eur_swap()], {"EURUSD": spot})
        usd, eur = alone["swap"], alone["swap-eur"]
        expected = np.stack([np.asarray(usd.pnl[:, 0]), spot * np.asarray(eur.pnl[:, 0])], axis=-1)
        assert result.base_currency == "USD"
        np.testing.assert_allclose(np.asarray(result.pnl), expected, rtol=1e-14, atol=0.0)
        np.testing.assert_allclose(np.asarray(result.portfolio_pnl), expected.sum(axis=1), rtol=1e-14, atol=1e-9)
        assert result.base_npv_per_trade == pytest.approx([usd.base_npv, spot * eur.base_npv], rel=1e-14)
        statistics = compute_risk_metrics(expected[:, None, :], 0.0, percentiles=(0.99, 0.975))
        for key in ("VaR_99", "ES_99", "VaR_97.5", "ES_97.5"):
            assert result.risk[key] == pytest.approx(float(statistics[key][0]), rel=1e-12)

    def test_the_spot_moves_the_figures(self):
        """The two spots' VaRs differ, so the test above is not met by an unconverted sum."""
        low, high = (_two_currency_run([m.swap(), m.eur_swap()], {"EURUSD": s}) for s in (1.10, 1.30))
        assert abs(high.risk["VaR_99"] - low.risk["VaR_99"]) > 1e-3 * low.risk["VaR_99"]

    def test_the_inverse_quote_converts_alike(self):
        quoted = _two_currency_run([m.swap(), m.eur_swap()], {"EURUSD": 1.25})
        inverse = _two_currency_run([m.swap(), m.eur_swap()], {"USDEUR": 0.8})
        np.testing.assert_allclose(np.asarray(inverse.pnl), np.asarray(quoted.pnl), rtol=1e-15, atol=0.0)

    def test_reporting_in_the_trades_currency_needs_no_spot_and_converts_nothing(self, alone):
        eur = alone["swap-eur"]
        assert eur.base_currency == "EUR"
        assert not any("FX" in w for w in eur.warnings)
        np.testing.assert_array_equal(np.asarray(eur.pnl[:, 0]), np.asarray(eur.portfolio_pnl))

    def test_warns_that_fx_is_not_shocked_naming_the_converted_trades(self):
        result = _two_currency_run([m.swap(), m.eur_swap()], {"EURUSD": 1.1}, num=64)
        assert any("['swap-eur']" in w and "FX risk is not in this VaR/ES" in w for w in result.warnings)

    def test_a_missing_spot_is_refused_before_any_work_naming_the_trade_and_the_pair(self, monkeypatch):
        def no_work(*args, **kwargs):
            raise AssertionError("revalued before the request was checked")

        monkeypatch.setattr("engine.risk.market.run.revalue_trade", no_work)
        with pytest.raises(KeyError, match=r"'swap-eur' is in EUR and the reporting currency is USD \(base_currency\), "
                                           r"but the market has no EURUSD spot"):
            _two_currency_run([m.swap(), m.eur_swap()], {})

    def test_the_paired_sample_is_converted_too(self):
        """At float64 the paired scenarios' figures at the run's precision and at float64 are the
        same numbers: both converted."""
        result = _two_currency_run([m.swap(), m.eur_swap()], {"EURUSD": 1.3}, num=128,
                                   precision=Precision(paired_fraction=0.5))
        for key in ("portfolio/VaR_99", "portfolio/ES_97.5"):
            figure = result.precision.figures[key]
            np.testing.assert_allclose(np.asarray(figure.paired), np.asarray(figure.paired_float64), rtol=1e-14)
