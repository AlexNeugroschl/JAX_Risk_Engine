"""
Tests for engine.market_risk: risk factors, shock scenarios, revaluation and
the VaR/ES run. ORE parity is in tests/test_market_risk_ore_parity.py.
"""
import dataclasses

import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.instruments.treasury import price_bond_base
from engine.market_risk import (
    MarketRiskRequest,
    RateRiskFactors,
    ShockScenarios,
    covariance_from_history,
    historical_scenarios,
    horizon_moves,
    monte_carlo_scenarios,
    run_market_risk,
)
from engine.market_risk.revaluation import revalue
from engine.models.hull_white import ZeroCurve
from engine.risk.greeks import swap_delta_gamma
from engine.risk.price_functions import bond_price_function, swaption_price_function
from engine.risk.var_es import RISK_MEASURE_HISTORICAL, compute_risk_metrics
from engine.simulation.market_model import ZeroCurveConfig
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
        assert labels[0] == "OIS/0y" and labels[1] == "OIS/1y"
        assert labels[len(m.PILLAR_TIMES)] == "IBOR/0y"

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
        base, shocked = revalue(trades, FACTORS, np.zeros((3, FACTORS.size)))
        np.testing.assert_allclose(np.asarray(shocked), np.tile(base, (3, 1)), rtol=1e-13)

    def test_batching_does_not_change_the_answer(self):
        trades = [m.swap(), m.bermudan()]
        shifts = _scenarios(num=20).shifts
        _, one_by_one = revalue(trades, FACTORS, shifts, batch_size=1)
        _, batched = revalue(trades, FACTORS, shifts, batch_size=7)
        np.testing.assert_allclose(np.asarray(one_by_one), np.asarray(batched), rtol=1e-13)

    def test_bond_equals_the_float_pricer_including_a_parallel_shift(self):
        bond = m.bond()
        f = bond_price_function(bond)
        rates = jnp.asarray(bond.initial_zero_curve.rates)
        assert float(f(rates)) == pytest.approx(price_bond_base(bond), rel=1e-14)
        assert float(f(rates + 0.0025)) == pytest.approx(price_bond_base(bond, rate_shift=0.0025), rel=1e-14)

    def test_swap_pnl_is_first_order_its_delta(self):
        """A 0.1bp parallel move of the discount curve: P&L equals the summed
        AD discount Delta (per 1bp) scaled by 0.1, to second order."""
        swap = m.swap()
        shift = np.zeros((2, FACTORS.size))
        shift[:, FACTORS.slice_of(0)] = 1e-5
        base, shocked = revalue([swap], FACTORS, shift)
        pnl = float(shocked[0, 0]) - base[0]
        delta = swap_delta_gamma(swap, ZeroCurve.from_config(m.OIS), ZeroCurve.from_config(m.IBOR))
        assert pnl == pytest.approx(0.1 * float(jnp.sum(delta["discount_delta"])), rel=1e-4)

    def test_grid_pricers_get_a_memory_bounded_batch(self):
        """A fine-grid Bermudan must not be vmapped hundreds of scenarios at a
        time: at n_per_std=64 one scenario's rollback holds ~80 MB, and a
        batch of 256 thrashed a 32 GB machine."""
        from engine.market_risk.revaluation import BATCH_MEMORY_BUDGET, scenario_batch_size

        fine = m.bermudan(n_per_std=64, std_devs=6.0)
        batch = scenario_batch_size(fine, 256, itemsize=8)
        assert 1 <= batch < 16
        nodes = 2 * int(64 * 6.0 + 0.5) + 1
        assert batch * nodes * nodes * 8 <= BATCH_MEMORY_BUDGET
        assert scenario_batch_size(m.swap(), 256, itemsize=8) == 256
        assert scenario_batch_size(m.bermudan(n_per_std=16, std_devs=5.0), 64, itemsize=8) == 64

    def test_swaption_price_function_keeps_float32(self):
        """Regression: two dtype-less constants inside the European price
        function promoted a float32 curve to float64, so `risk=32` swaption
        Greeks and float32 market risk silently ran partly in float64."""
        f = swaption_price_function(m.european(), ZeroCurve.from_config(m.OIS, dtype=jnp.float32))
        assert f(jnp.asarray(m.OIS.rates, dtype=jnp.float32)).dtype == jnp.float32


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mixed_run():
    trades = [m.swap(), m.european(), m.bermudan(), m.bond()]
    return trades, run_market_risk(MarketRiskRequest(trades, _scenarios(num=512)))


class TestRun:
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
        assert any("volatility risk is not in this VaR/ES" in w and "[1, 2]" in w for w in result.warnings)

    def test_warns_about_a_thin_tail(self):
        result = run_market_risk(MarketRiskRequest([m.swap()], _scenarios(num=64), quantiles=(0.99,)))
        assert any("too few for a stable estimate" in w for w in result.warnings)

    def test_float32_run_is_float32_and_close(self, mixed_run):
        trades, result64 = mixed_run
        result32 = run_market_risk(MarketRiskRequest(trades, _scenarios(num=512), precision=32))
        assert result32.pnl.dtype == jnp.float32
        assert result32.risk["VaR_99"] == pytest.approx(result64.risk["VaR_99"], rel=1e-4)


class TestRunValidation:
    def _request(self, trades, **kwargs):
        return MarketRiskRequest(trades, _scenarios(num=16), **kwargs)

    def test_no_trades(self):
        with pytest.raises(ValueError, match="at least one trade"):
            run_market_risk(self._request([]))

    @pytest.mark.parametrize("kwargs, match", [
        ({"precision": 16}, "precision"),
        ({"batch_size": 0}, "batch_size"),
        ({"quantiles": (1.0,)}, "quantile"),
    ])
    def test_bad_settings(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            run_market_risk(self._request([m.swap()], **kwargs))

    def test_curve_index_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            run_market_risk(self._request([dataclasses.replace(m.swap(), forward_curve_index=2)]))

    def test_trade_curve_must_be_the_factor_curve(self):
        other = ZeroCurveConfig(m.PILLAR_TIMES, [r + 0.001 for r in m.OIS.rates])
        with pytest.raises(ValueError, match="does not match risk-factor curve 'OIS'"):
            run_market_risk(self._request([dataclasses.replace(m.european(), initial_zero_curve=other)]))

    def test_bond_must_name_its_curve(self):
        with pytest.raises(ValueError, match="curve_index"):
            run_market_risk(self._request([dataclasses.replace(m.bond(), curve_index=None)]))

    def test_bermudan_must_be_calibrated(self):
        with pytest.raises(ValueError, match="calibrate"):
            run_market_risk(self._request([dataclasses.replace(m.bermudan(), hw_sigma=None)]))

    def test_one_evaluation_date(self):
        shifted = dataclasses.replace(m.swap(), evaluation_date=m.TODAY + 1)
        with pytest.raises(ValueError, match="one evaluation_date"):
            run_market_risk(self._request([m.swap(), shifted]))
