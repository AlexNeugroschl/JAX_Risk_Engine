"""Rounding, option bounds and independent uncertainty (I-98, I-111, I-113)."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from engine.market_simulation.sobol import generate_sobol_normals
from engine.precision import Arithmetic, StagePrecision
from engine.precision.estimate import paired_indices, replicate_standard_error, two_level_mean
from engine.pricing.options import wrap
from tests.support.compiles import count_compiles


def test_long_option_bound_precedes_position_and_physical_delivery():
    option = jnp.asarray([[-1e-5, -2e-5]])
    long, _ = wrap(option, (), (), False, 1.)
    short, _ = wrap(option, (), (), False, -1.)
    np.testing.assert_array_equal(long, 0.)
    np.testing.assert_array_equal(short, -long)
    # An exercised physical underlying can legitimately become negative.
    delivered, _ = wrap(jnp.zeros((1, 2)), (jnp.asarray([[1., -3.]]),), (0,), True, 1.)
    np.testing.assert_array_equal(delivered, [[1., -3.]])


def test_market_risk_cannot_revalue_a_long_option_below_zero():
    from engine.risk.market.revaluation import revalue_trade
    from tests import market_risk_support as mr
    moves = np.linspace(-.15, .15, 128)[:, None] * np.ones((1, mr.factors().size))
    base, pnl = revalue_trade(mr.bermudan(), mr.market(), mr.factors(), moves, mr.PRICING, 128,
                              Arithmetic("bfloat16", "float32", execution="allow_emulation"))
    assert np.min(base + np.asarray(pnl)) >= 0.


def test_option_bound_survives_pnl_storage_and_position():
    from engine.precision import Precision
    from engine.risk.market import MarketRiskRequest, run_market_risk
    from engine.risk.market.scenarios import ShockScenarios
    from tests import market_risk_support as mr
    moves = np.linspace(-.15, .15, 128)[:, None] * np.ones((1, mr.factors().size))
    scenarios = ShockScenarios(mr.factors(), moves, 10, "monte-carlo")
    long = mr.bermudan()
    for position, sign in (("Long", 1.), ("Short", -1.)):
        trade = dataclasses.replace(long, long_short=position)
        result = run_market_risk(MarketRiskRequest([trade], mr.market(), scenarios, mr.PRICING,
                                 precision=Precision.research(pricing=StagePrecision("float8_e4m3fn"))))
        assert np.min(sign * (result.base_npv + np.asarray(result.portfolio_pnl))) >= 0.


@pytest.mark.parametrize("name", ["float16", "bfloat16", "float8_e4m3fn", "float8_e5m2", "tensorfloat32"])
def test_stochastic_product_has_no_binade_rounding_bias(name):
    rng = np.random.default_rng(147)
    values = jnp.asarray(rng.normal(1., .3, (65536, 3)), dtype=jnp.float32)
    arithmetic = Arithmetic(name, "float32", rounding="stochastic", key=jax.random.key(719), execution="allow_emulation")
    got = jax.jit(lambda x, a: a.matmul(x, jnp.ones((3, 1)), shared="right"))(values, arithmetic)[:, 0]
    errors = np.asarray(got) - np.asarray(values).sum(axis=1)
    assert abs(errors.mean()) < 4 * errors.std(ddof=1) / np.sqrt(len(errors)) + 1e-6


def test_stochastic_keys_are_data_and_sample_prefixes_reproduce():
    stage = StagePrecision("float32", "float8_e4m3fn", "float32", compute_rounding="stochastic")
    arithmetic = dataclasses.replace(stage.arithmetic, key=jax.random.key(9), execution="allow_emulation")
    product = jax.jit(lambda x, a: a.matmul(x, jnp.asarray([[.17], [.43]]), shared="right"))
    values = jnp.asarray(np.random.default_rng(13).normal(size=(128, 2)), dtype=jnp.float32)
    full = product(values, arithmetic)
    np.testing.assert_array_equal(product(values[:32], arithmetic), full[:32])
    # Stable global path identities also survive batching.
    tail = dataclasses.replace(arithmetic, sample_ids=jnp.arange(64, 128, dtype=jnp.uint32))
    np.testing.assert_array_equal(product(values[64:], tail), full[64:])
    changed = dataclasses.replace(arithmetic, key=jax.random.key(10))
    with count_compiles() as compiled:
        other = product(values, changed)
        other.block_until_ready()
    assert not compiled
    assert np.any(np.asarray(other) != np.asarray(full))


@pytest.mark.parametrize("kernel", ["legs", "jamshidian"])
def test_adjacent_calendar_dates_have_distinct_rounding_streams(monkeypatch, kernel):
    from engine.market_data.curves import ZeroCurve
    from engine.pricing.config import JamshidianEngineConfig
    from engine.pricing.european import european_terms
    from engine.pricing.jamshidian import jamshidian_discounts, jamshidian_on_discounts
    from engine.pricing.legs import Deviations, leg_values
    from engine.risk.greeks.price_functions import trade_price_function
    from tests import market_risk_support as mr
    terms = (trade_price_function(mr.swap(), mr.market()).terms if kernel == "legs" else
             european_terms(mr.european(), mr.TODAY))
    curve = ZeroCurve.flat(.03, [0., 1., 5., 10.])
    moved = jnp.zeros((2, 4), dtype=jnp.float32)
    deviations = Deviations(moved, moved, jnp.zeros((2, len(terms.legs.float_pay)), dtype=jnp.float32))
    arithmetic = Arithmetic("float16", "float32", "stochastic", jax.random.key(71), execution="allow_emulation")
    fold_in, dates = Arithmetic.fold_in, []

    def record(self, data):
        if isinstance(data, jax.Array):
            dates.append(int(data))
        return fold_in(self, data)

    monkeypatch.setattr(Arithmetic, "fold_in", record)
    # 3 / 365 * 365 is slightly below 3 in float64; truncation collides with day 2.
    for day in (2, 3):
        if kernel == "legs":
            leg_values(terms.legs, curve, curve, day / 365., terms.alive_fixed, terms.alive_float,
                       terms.projected, terms.known_rates, deviations, arithmetic=arithmetic)
        else:
            p_bond, p_start = jamshidian_discounts(terms, curve, day / 365., moved)
            jamshidian_on_discounts(terms, JamshidianEngineConfig(.03, .01), p_bond, p_start, day / 365., arithmetic)
    assert dates == [2, 3]


def test_sobol_replicates_pair_a_prefix_in_each_scramble():
    full = generate_sobol_normals(256, 2, 1, jnp.float64, seed=18, replicates=4)
    sample = generate_sobol_normals(64, 2, 1, jnp.float64, seed=18, replicates=4)
    np.testing.assert_array_equal(np.asarray(full).reshape(2, 4, 64, 1)[:, :, :16],
                                  np.asarray(sample).reshape(2, 4, 16, 1))
    assert not np.array_equal(full[:, :64], full[:, 64:128])


def test_replicate_mean_pairs_each_scramble_and_counts_covariance():
    rng = np.random.default_rng(814)
    low = rng.normal(size=(8, 64, 3)) + np.arange(8)[:, None, None]
    high = low[:, :32] + .1 * rng.normal(size=(8, 32, 3))
    estimate = two_level_mean(low.reshape(-1, 3), high.reshape(-1, 3), replicates=8)
    means, corrections = low.mean(axis=1), (high - low[:, :32]).mean(axis=1)
    np.testing.assert_allclose(estimate.value, (means + corrections).mean(axis=0))
    np.testing.assert_allclose(estimate.standard_error, (means + corrections).std(axis=0, ddof=1) / np.sqrt(8))
    np.testing.assert_allclose(estimate.correction_standard_error, corrections.std(axis=0, ddof=1) / np.sqrt(8))
    np.testing.assert_array_equal(paired_indices(512, 256, 8),
                                  (np.arange(8)[:, None] * 64 + np.arange(32)).ravel())


def test_one_scramble_cannot_report_an_rqmc_standard_error():
    estimate = two_level_mean(jnp.arange(64.), jnp.arange(32.), replicates=1)
    assert np.isnan(estimate.standard_error)
    assert np.isnan(estimate.correction_standard_error)


def test_rqmc_mean_intervals_cover_across_independent_outer_seeds():
    from scipy.stats import t
    estimates, errors = [], []
    # E[Z^2] = 1 is an analytic reference; within-scramble paths are dependent.
    for seed in range(128):
        z = np.asarray(generate_sobol_normals(1024, 1, 1, jnp.float64, seed=seed, replicates=8))[0, :, 0]
        means = (z.reshape(8, -1) ** 2).mean(axis=1)
        estimates.append(means.mean())
        errors.append(float(replicate_standard_error(means)))
    covered = np.abs(np.asarray(estimates) - 1) <= t.ppf(.975, 7) * np.asarray(errors)
    assert .89 <= covered.mean() <= 1.


def test_replicate_controls_reject_invalid_groups():
    for bad in (0, -1, True, 1.5, 3):
        with pytest.raises(ValueError, match="replicates"):
            generate_sobol_normals(128, 1, 1, jnp.float64, replicates=bad)


def test_sampling_wire_controls_refuse_bool_and_ignored_settings():
    from engine.api.requests import HistoricalScenariosSchema, MonteCarloScenariosSchema, SimulationSchema
    from engine.risk.market.scenarios import ShockScenarios
    from tests import market_risk_support as mr
    for cls, body in (
        (SimulationSchema, {"dates": ["2027-01-01"], "replicates": True}),
        (MonteCarloScenariosSchema, {"source": "monte-carlo", "factors": [], "horizon_days": 10,
                                    "covariance": [], "num_scenarios": 128, "replicates": True}),
        (HistoricalScenariosSchema, {"source": "historical", "factors": [], "horizon_days": 10,
                                    "history": [], "bootstrap_replicates": True}),
    ):
        with pytest.raises(ValueError, match="replicates"):
            cls.model_validate(body)
    with pytest.raises(ValueError, match="only to historical"):
        ShockScenarios(mr.factors(), np.zeros((128, mr.factors().size)), 10, "monte-carlo", block_length=32)


def test_a_scalar_paired_sample_is_refused_by_shape():
    with pytest.raises(ValueError, match="paired sample"):
        two_level_mean(jnp.arange(10.), 1.)


def test_replicated_portfolio_pairs_all_scrambles_and_reuses_programs():
    from engine.run import CamConfig, LgmConfig, PortfolioRequest, RunConfig, price_portfolio
    from engine.precision import Precision
    from tests.support import portfolio as shared
    import ORE
    simulation = CamConfig(dates=(shared.ASOF + ORE.Period("1Y"),), base_currency="USD",
                           ir={"USD": LgmConfig(.03)}, samples=256, replicates=4, seed=13)
    # Retain the measured scaled/residual profile while testing uncertainty and reuse.
    precision = Precision.research(pricing=StagePrecision.compensated("float32", "float8_e4m3fn", "float32", "stochastic"),
                          paired_fraction=.25)
    request = PortfolioRequest(shared.market(), [shared.trades()["swap-payer"]],
                               RunConfig(simulation=simulation, precision=precision))
    result = price_portfolio(request)
    assert result.precision.paired_paths == 128  # at least one storage block in each scramble
    assert result.precision.replicates == 4
    from engine.api.results import PrecisionReportSchema
    assert result.precision.interval.status == "unavailable"
    wire = PrecisionReportSchema.from_dataclass(result.precision).model_dump(mode="json")
    assert wire["interval"] == dataclasses.asdict(result.precision.interval)
    assert np.isfinite(result.precision.figures["netting_set/EPE"].standard_error).all()
    request.config = dataclasses.replace(request.config, simulation=dataclasses.replace(simulation, seed=14),
                                        precision=dataclasses.replace(precision, rounding_seed=9))
    with count_compiles() as compiled:
        price_portfolio(request)
    assert not compiled


def test_market_risk_reports_replicate_and_historical_uncertainty_on_wire():
    from pydantic import TypeAdapter

    from engine.api.results import MarketRiskSectionSchema
    from engine.risk.market import MarketRiskRequest, monte_carlo_scenarios, run_market_risk
    from engine.risk.market.scenarios import historical_scenarios
    from tests import market_risk_support as mr
    scenarios = monte_carlo_scenarios(mr.factors(), mr.covariance(), 10, 1024, seed=93, replicates=8)
    request = MarketRiskRequest([mr.swap()], mr.market(), scenarios, quantiles=(.95,))
    with jax.numpy_dtype_promotion("strict"):
        result = run_market_risk(request)
    assert result.risk["ES_95_standardError"] > 0
    assert result.uncertainty == {"method": "replicate_jackknife", "replicates": 8,
                                  "interval": dataclasses.asdict(result.precision.interval)}
    wire = TypeAdapter(MarketRiskSectionSchema.model_fields["uncertainty"].annotation)
    assert wire.validate_python(result.uncertainty) == result.uncertainty
    history = np.cumsum(np.random.default_rng(6).normal(size=(1024, mr.factors().size)) * 1e-4, axis=0)
    request.scenarios = historical_scenarios(mr.factors(), history, 10, block_length=32, bootstrap_replicates=64)
    with jax.numpy_dtype_promotion("strict"):
        historical = run_market_risk(request)
        request.scenarios = dataclasses.replace(request.scenarios, bootstrap_seed=43)
        with count_compiles() as compiled:
            run_market_risk(request)
        assert not compiled
    assert historical.risk["ES_95_standardError"] > 0
    assert historical.uncertainty["block_length"] == 32
    assert historical.uncertainty["method"] == "moving_block_bootstrap"
    assert historical.uncertainty["interval"] == result.uncertainty["interval"]


def test_replicate_tail_error_recomputes_threshold_in_each_deletion():
    from engine.risk.market.var_es import expected_shortfall, replicated_es_error
    from engine.precision.uncertainty import jackknife_error
    pnl = np.random.default_rng(45).normal(size=(8, 128)) + np.arange(8)[:, None] * .2
    deleted = [float(expected_shortfall(jnp.asarray(np.delete(pnl, i, axis=0).reshape(-1, 1)), .95)[0])
               for i in range(8)]
    np.testing.assert_allclose(replicated_es_error(jnp.asarray(pnl.ravel()), .95, 8), jackknife_error(jnp.asarray(deleted)))


@pytest.mark.slow
def test_es_intervals_cover_independent_scrambles_and_overlapping_history():
    from scipy.stats import norm, t
    from engine.risk.market.var_es import expected_shortfall, replicated_es_error, historical_es_error
    target = norm.pdf(norm.ppf(.95)) / .05
    rqmc, historical = [], []
    for seed in range(96):
        pnl = generate_sobol_normals(4096, 1, 1, jnp.float64, seed=seed, replicates=8)[0, :, 0]
        estimate = float(expected_shortfall(pnl[:, None], .95)[0])
        error = float(replicated_es_error(pnl, .95, 8))
        rqmc.append(abs(estimate - target) <= t.ppf(.975, 7) * error)
        # Overlapping five-day horizons have a known marginal N(0, 5) distribution.
        daily = np.random.default_rng(seed).normal(size=2053)
        history = np.cumsum(daily)
        moves = jnp.asarray(history[5:] - history[:-5])
        estimate = float(expected_shortfall(moves[:, None], .95)[0])
        error = float(historical_es_error(moves, jax.random.key(seed), .95, 32, 128))
        historical.append(abs(estimate - np.sqrt(5) * target) <= 1.96 * error)
    assert .88 <= np.mean(rqmc) <= 1., np.mean(rqmc)
    assert .85 <= np.mean(historical) <= 1., np.mean(historical)


def test_stochastic_option_rollback_runs_in_blocks_without_changing_rounding(monkeypatch):
    from engine.run import PortfolioRequest, RunConfig, price_portfolio
    from engine.precision import Precision
    from tests.support import precision_campaigns as campaigns
    from engine.pricing import lgm_grid
    campaign = campaigns.options()
    simulation = dataclasses.replace(campaign.simulation, samples=64)
    # This checks batching of the historical stochastic rollback profile.
    policy = Precision.research(pricing=StagePrecision.compensated("float32", "float16", "float32", "stochastic"))
    request = PortfolioRequest(campaign.market, list(campaign.trades[:1]),
                               RunConfig(simulation=simulation, precision=policy, pricing=campaign.pricing))
    full = price_portfolio(request)
    monkeypatch.setattr(lgm_grid, "INDUCTION_MEMORY_BUDGET", 1)
    split = price_portfolio(request)
    np.testing.assert_allclose(full.npv_cube, split.npv_cube, rtol=2e-6, atol=.02)
    np.testing.assert_array_equal(full.exercised[request.trades[0].trade_id], split.exercised[request.trades[0].trade_id])
