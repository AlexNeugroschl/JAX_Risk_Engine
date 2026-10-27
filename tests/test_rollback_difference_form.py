"""
The Bermudan/American induction as matrix products in difference form (F-07, I-83;
`engine.pricing.lgm_grid`, docs/planning/details/precision.md §8.2): its building blocks against
independent references, each path's value against the induction run on that path's own curves,
the padding that gives a cube's dates one program, its memory and the blocks of paths that bound
it, and its compute formats below float32. ORE parity is tests/test_ore_lgm_parity.py (today)
and tests/test_ore_xva_parity.py (on ORE's paths).
"""
import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from bermudan_references import cashflow_values_at_nodes
from engine.instruments.bermudan_swaption import underlying_swap
from engine.market_data.curves import DiscountCurve, ZeroCurve, discount
from engine.market_data.market import index_name
from engine.market_simulation.config import CamConfig, LgmConfig, simulate
from engine.models.lgm import horizon_shift, numeraire, zeta
from engine.precision import Arithmetic, Precision, StagePrecision, add_level
from engine.pricing.bermudan import bermudan_cube, built_sigma, prepared_option
from engine.pricing.config import LgmSwaptionEngineConfig, PricingConfig
from engine.pricing.cube import value_portfolio
from engine.pricing.legs import Deviations, path_fixings
from engine.pricing import lgm_grid
from engine.pricing.lgm_grid import (
    _basis, _Coefficients, _hagan_quadrature_weights, _quadrature_nodes, _rollback_operator, grid_value,
    induction_rows, one_shape, option_values, path_blocks, working_bytes_per_path,
)
from tests.support import portfolio as shared
from tests.support.compiles import count_compiles

#: A coarse grid, as the precision tests use, and the demo's (I-83's measurements).
COARSE = LgmSwaptionEngineConfig(n_per_std=8, std_devs=3.0)
DEMO_GRID = LgmSwaptionEngineConfig(n_per_std=16, std_devs=6.0)
OPTIONS = ("bermudan-payer-physical", "bermudan-receiver-cash", "american-payer")
DATES = tuple(shared.ASOF + ORE.Period(m, ORE.Months) for m in (6, 18, 36))
SIMULATION = CamConfig(dates=DATES, base_currency="USD", ir={"USD": LgmConfig(0.03, 0.01, ("1Y", "2Y"), ("9Y", "8Y"))},
                       samples=8, seed=5)


def _today(name, engine=COARSE):
    """A shared option prepared on today's curves (its LGM the engine's flat volatility)."""
    market = shared.market()
    disc = ZeroCurve.from_config(market.currency("USD").discount_curve)
    cfg = shared.trades()[name]
    index = ZeroCurve.from_config(market.index_curve("USD", index_name("USD", cfg.index_tenor_months)))
    return prepared_option(cfg, engine, engine.volatility, disc, index)


@pytest.fixture(scope="module")
def scenarios():
    return simulate(shared.market(), SIMULATION)


class TestRollbackOperator:
    """One step's rollback is one `[nodes, nodes]` operator, shared by every path and column."""

    @pytest.mark.parametrize("std_ratio", [0.3, 0.9, 3.0])
    def test_it_is_ores_interpolated_convolution(self, std_ratio):
        """`R @ f` is `sum_i w_i f(x_to + y_i std)`, f interpolated linearly on the grid it is
        given on and flat beyond it (`LgmConvolutionSolver2::rollback`), for any f: here a
        smooth one and noise, with queries off the grid's ends (`std_ratio` 3)."""
        n_per_std, std_devs = 8, 3.0
        mx = 24
        y, w = jnp.asarray(_quadrature_nodes(n_per_std, std_devs)), jnp.asarray(_hagan_quadrature_weights(n_per_std,
                                                                                                          std_devs))
        dx_from, dx_to = 0.0125, 0.009
        nodes = jnp.arange(-mx, mx + 1, dtype=jnp.float64)
        x_from, x_to = dx_from * nodes, dx_to * nodes
        std = std_ratio * dx_from * n_per_std
        operator = _rollback_operator(jnp.asarray(dx_from), x_to, jnp.asarray(std), y, w)
        for f in (jnp.exp(3.0 * x_from) - 1.0, jax.random.normal(jax.random.key(1), x_from.shape)):
            query = x_to[:, None] + y[None, :] * std
            reference = jnp.sum(jnp.interp(query, x_from, f, left=f[0], right=f[-1]) * w[None, :], axis=-1)
            np.testing.assert_allclose(np.asarray(operator @ f), np.asarray(reference), rtol=0, atol=1e-14)

    def test_without_variance_it_is_interpolation_alone(self):
        """Where no variance lies between the two times (a volatility of 0), ORE interpolates:
        each point with weight 1, not the quadrature's mass."""
        y, w = jnp.asarray(_quadrature_nodes(8, 3.0)), jnp.asarray(_hagan_quadrature_weights(8, 3.0))
        nodes = jnp.arange(-24, 25, dtype=jnp.float64)
        x_from, x_to = 0.01 * nodes, 0.007 * nodes
        f = jnp.sin(40.0 * x_from)
        operator = _rollback_operator(jnp.asarray(0.01), x_to, jnp.asarray(0.0), y, w)
        np.testing.assert_allclose(np.asarray(operator @ f), np.asarray(jnp.interp(x_to, x_from, f)), atol=1e-15)


class TestCashflowBasis:
    """A cashflow's value on the grid is a coefficient (the path's) times a function of the
    state (the model's), so every path's cashflows are one product with shared functions."""

    @pytest.mark.parametrize("name", ["bermudan-payer-physical", "american-payer"])
    def test_coefficients_times_functions_are_the_lgm_cashflow_values(self, name):
        """Against the LGM's bond prices directly (`tests/bermudan_references.py`), with every
        kind of floating coupon: forecast, known (a fixing) and fixed today, on rows before and
        after an index start (where `T1 = max(t, start)` moves the forecast)."""
        swap = _today(name)
        n = len(swap.float_pay_times)
        known, today = np.arange(n) % 3 == 1, np.arange(n) % 3 == 2
        swap = dataclasses.replace(swap, float_is_known=known, float_fixed_today=today & ~known,
                                   float_known_rates=np.where(known, 0.041, 0.0))
        shift = horizon_shift(swap.reversion, swap.shift_horizon)
        coefficients = _Coefficients.of(swap, swap.curve, swap.index_curve, jnp.float64)
        nf = len(swap.fixed_times)
        for t in (0.25, 1.3, float(swap.float_index_start_times[n // 2]) + 0.01):
            x = jnp.linspace(-0.04, 0.05, 7)
            start, end, basis = _basis(swap, jnp.asarray(t), x, zeta(swap.sigma, t), shift)
            ratio = discount(swap.index_curve, start) / discount(swap.index_curve, end)
            terms = coefficients.at_row(ratio)[:, None] * basis                                   # [J, N]
            values = jnp.concatenate([terms[:nf], terms[nf:nf + n] + terms[nf + n:]], axis=0).T  # [N, C]
            values = values * numeraire(swap.curve, swap.reversion, swap.sigma, jnp.asarray(t), x, shift)[:, None]
            reference = cashflow_values_at_nodes(swap, x, jnp.asarray(t), shift)
            np.testing.assert_allclose(np.asarray(values), np.asarray(reference), rtol=1e-12,
                                       atol=1e-12 * float(jnp.max(jnp.abs(reference))))


class TestCacheColumns:
    def test_a_bermudan_caches_nothing_and_an_american_few_coupons(self):
        """A Bermudan enters whole periods, so no coupon is ever cached; an American caches the
        coupon in progress on each leg, a column reused once its coupon has moved, far fewer
        than its coupons (each one a column per path until 2026-10-13)."""
        assert induction_rows(_today("bermudan-payer-physical")).shape[1] == 0
        american = _today("american-payer")
        slots = induction_rows(american).shape[1]
        assert 0 < slots <= 4 < len(american.fixed_times) + len(american.float_pay_times)


class TestDifferenceForm:
    """Each path's value is the level's plus its deviation, at float64 the induction on the
    path's own curves to rounding."""

    @pytest.mark.parametrize("name", OPTIONS)
    def test_each_scenario_is_the_induction_on_its_shifted_curves(self, name):
        """Market risk's case: today's curves shifted pillar by pillar (`ZeroCurve`s)."""
        swap = _today(name)
        k = swap.curve.pillar_rates.shape[0]
        shifts = 0.004 * jax.random.normal(jax.random.key(7), (2, 6, k))
        deviations = Deviations(shifts[0], shifts[1], jnp.zeros((6, len(swap.float_pay_times))))
        level, moved = option_values(swap, None, deviations)
        for s in range(6):
            shifted = dataclasses.replace(
                swap, curve=ZeroCurve(swap.curve.pillar_times, swap.curve.pillar_rates + shifts[0, s]),
                index_curve=ZeroCurve(swap.index_curve.pillar_times, swap.index_curve.pillar_rates + shifts[1, s]))
            expected = float(grid_value(shifted))
            assert float(add_level(level, moved[s])) == pytest.approx(expected, rel=1e-12, abs=1e-12 * abs(level))

    @pytest.mark.parametrize("name", OPTIONS)
    def test_a_scenario_that_moves_nothing_moves_the_value_by_exactly_nothing(self, name):
        swap = _today(name)
        k = swap.curve.pillar_rates.shape[0]
        zeros = Deviations(jnp.zeros((3, k)), jnp.zeros((3, k)), jnp.zeros((3, len(swap.float_pay_times))))
        for arithmetic in (Arithmetic(), Arithmetic.compensated("float8_e4m3fn", "float32", execution="allow_emulation")):
            deviations = jax.tree_util.tree_map(lambda a: a.astype(arithmetic.dtype), zeros)
            _, moved = option_values(swap, None, deviations, arithmetic)
            np.testing.assert_array_equal(np.asarray(moved), 0.0)

    @pytest.mark.parametrize("name", OPTIONS)
    def test_each_path_is_the_induction_on_its_own_curves_and_fixings(self, name, scenarios):
        """The path cube (`bermudan_cube`) against `grid_value` of the trade on each path's own
        curves on each date, its path fixings given to it as fixings (FixingManager's, read
        off the path), as the engine computed it until 2026-10-13 (per path)."""
        market = shared.market()
        cfg = shared.trades()[name]
        index = index_name(cfg.currency, cfg.index_tenor_months)
        fixings = path_fixings(cfg.index_tenor_months, market.asof, scenarios.dates, scenarios.times,
                               scenarios.index[index])
        cube = np.asarray(bermudan_cube(cfg, COARSE, market, scenarios, fixings))
        sigma = built_sigma(cfg, COARSE, market, scenarios)
        serials = np.array([d.serialNumber() for d in scenarios.dates])
        rates = np.asarray(fixings.rates)
        for j, date in enumerate(scenarios.dates):
            for s in (0, 3, 7):
                path_fixed = {}
                for coupon in underlying_swap(cfg).floatingLeg():
                    f = ORE.as_floating_rate_coupon(coupon).fixingDate()
                    if market.asof <= f < date:
                        step = min(int(np.searchsorted(serials, f.serialNumber(), side="right")), len(serials) - 1)
                        path_fixed[f] = float(rates[s, step])
                dated = dataclasses.replace(cfg, evaluation_date=date, fixings={**cfg.fixings, **path_fixed})
                curves = [DiscountCurve(c.tenor_times[j], c.log_discounts[s, j])
                          for c in (scenarios.discount["USD"], scenarios.index[index])]
                expected = float(grid_value(prepared_option(dated, COARSE, sigma, *curves, built_on=market.asof)))
                assert cube[s, j] == pytest.approx(expected, rel=1e-12, abs=1e-12 * np.max(np.abs(cube)))


class TestOneProgramPerTrade:
    def test_a_trade_padded_to_another_shape_keeps_its_value(self):
        """Coupons, rows and cache columns that do nothing change no value."""
        cfg = shared.trades()["american-payer"]
        curve = _today("american-payer").curve
        earlier = _today("american-payer")
        fixing_dates = (ORE.as_floating_rate_coupon(c).fixingDate() for c in underlying_swap(cfg).floatingLeg())
        fixed = {f: 0.041 for f in fixing_dates if f < DATES[-1]}
        later = prepared_option(dataclasses.replace(cfg, evaluation_date=DATES[-1], fixings=fixed), COARSE,
                                COARSE.volatility, curve, curve, built_on=cfg.evaluation_date)
        (_, _), (padded, rows) = one_shape([earlier, later])
        assert rows.shape == induction_rows(earlier).shape[:1] + (rows.shape[1],)
        assert float(grid_value(padded, rows)) == pytest.approx(float(grid_value(later)), rel=1e-14)

    def test_a_cube_compiles_its_induction_once_for_every_date(self, scenarios):
        """Every date of a cube is the trade padded to one shape (`one_shape`) and the date a
        traced index: one program (the induction inside it), where each date's own shape
        (exercises and coupons running out) compiled its own; a second trade of the shape compiles
        nothing at all."""
        market = shared.market()
        cfg = shared.trades()["bermudan-payer-physical"]
        fixings = path_fixings(cfg.index_tenor_months, market.asof, scenarios.dates, scenarios.times,
                               scenarios.index[index_name("USD", cfg.index_tenor_months)])
        jax.clear_caches()
        with count_compiles() as counter:
            bermudan_cube(cfg, COARSE, market, scenarios, fixings)
        assert counter["jit__date_column"] == 1 and counter["jit__induction"] == 0, dict(counter)
        other = dataclasses.replace(cfg, fixed_rate=0.05, notional=2e6, trade_id="other")
        with count_compiles() as counter:
            bermudan_cube(other, COARSE, market, scenarios, fixings)
        assert sum(counter.values()) == 0, dict(counter)


class TestPathBlocks:
    def test_the_fewest_equal_blocks_within_the_budget(self, monkeypatch):
        """Each device's share of the paths split into the fewest equal blocks whose induction
        fits the budget; one block when every path fits."""
        monkeypatch.setattr(lgm_grid, "induction_memory_budget", lambda: 1000)
        assert path_blocks(64, 10) == 1
        assert path_blocks(64, 100) == 8 and path_blocks(64, 100, devices=4) == 2
        assert path_blocks(96, 100) == 12
        assert path_blocks(7, 100) == 1
        assert path_blocks(7, 2000) == 7  # not even one path fits: one per block

    @pytest.mark.parametrize("name", ["bermudan-receiver-cash", "american-payer"])
    def test_a_cube_priced_in_blocks_is_the_cube_priced_at_once(self, name, scenarios, monkeypatch):
        """A budget a quarter of what the paths need prices each date in four blocks of strided
        paths: the same values, to the order of a product's sums."""
        market = shared.market()
        cfg = shared.trades()[name]
        index = index_name(cfg.currency, cfg.index_tenor_months)
        fixings = path_fixings(cfg.index_tenor_months, market.asof, scenarios.dates, scenarios.times,
                               scenarios.index[index])
        whole = np.asarray(bermudan_cube(cfg, COARSE, market, scenarios, fixings))
        per_path = working_bytes_per_path(prepared_option(cfg, COARSE, COARSE.volatility), 8)
        monkeypatch.setattr(lgm_grid, "induction_memory_budget", lambda: 2 * per_path)
        assert path_blocks(scenarios.num_paths, per_path) == 4
        in_blocks = np.asarray(bermudan_cube(cfg, COARSE, market, scenarios, fixings))
        np.testing.assert_allclose(in_blocks, whole, rtol=0, atol=1e-12 * np.max(np.abs(whole)))


class TestMemory:
    @pytest.mark.parametrize("name", OPTIONS)
    @pytest.mark.parametrize("compute", ["float64", "float32", "float8_e4m3fn"])
    def test_the_induction_holds_a_few_columns_per_path_not_an_operator_per_step(self, name, compute):
        """I-83: each path held `[steps, nodes, nodes]` when the rollback interpolated per path
        (0.3 MB per path and step at the demo's grid; 277 GB at 65,536 paths). The induction
        holds a few copies of its columns `[columns, nodes]` per path, whatever the steps: XLA's
        memory analysis of the compiled program, per path, within `working_bytes_per_path`,
        which sizes market risk's batches."""
        swap = _today(name, DEMO_GRID)
        rows = induction_rows(swap)
        arithmetic = Arithmetic.compensated(compute, "float64" if compute == "float64" else "float32", execution="allow_emulation")
        run = jax.jit(partial(option_values, arithmetic=arithmetic))
        k = swap.curve.pillar_rates.shape[0]

        def bytes_at(paths):
            deviations = Deviations(*(jnp.zeros((paths, n), arithmetic.dtype)
                                      for n in (k, k, len(swap.float_pay_times))))
            analysis = run.lower(swap, rows, deviations).compile().memory_analysis()
            return analysis.temp_size_in_bytes + analysis.argument_size_in_bytes + analysis.output_size_in_bytes

        per_path = (bytes_at(2048) - bytes_at(512)) / 1536
        nodes = 2 * 96 + 1
        steps = rows.shape[0]
        assert per_path <= working_bytes_per_path(swap, jnp.dtype(arithmetic.dtype).itemsize)
        assert per_path < 0.05 * steps * nodes * nodes * jnp.dtype(arithmetic.dtype).itemsize


#: The options' largest error per value and largest bias (the mean over the paths, per date),
#: relative to the float64 column's largest value, per compute format, accumulated in float32:
#: about four times those measured on 256 paths on 2026-10-13. Regression guards; the precision
#: harness measures each format against the statistical error (the `options` campaign).
_COMPUTE_BOUNDS = {"float32": (2e-6, 1e-7), "bfloat16_3x": (1e-4, 1e-6), "tensorfloat32": (5e-3, 4e-5),
                   "float16": (5e-3, 4e-5), "bfloat16": (5e-2, 5e-4), "float8_e4m3fn": (1.0, 1.5e-2),
                   "float8_e5m2": (2.5, 1.5e-2)}


@pytest.mark.slow
@pytest.mark.parametrize("compute", sorted(_COMPUTE_BOUNDS))
def test_every_compute_format_prices_the_options(compute):
    """The options' pricing stage in every compute format (refused until 2026-10-13, when the
    rollback was not in difference form): finite under strict dtype promotion, today's values
    float64, each path's error and the paths' bias within the format's bound."""
    market = shared.market()
    trades = [shared.trades()[n] for n in OPTIONS]
    simulation = dataclasses.replace(SIMULATION, samples=256, seed=1)
    scenarios = simulate(market, simulation)
    pricing = PricingConfig(bermudan=COARSE, american=COARSE)
    exact = value_portfolio(trades, market, scenarios, "USD", pricing)
    with jax.numpy_dtype_promotion("strict"):
        run = value_portfolio(trades, market, scenarios, "USD", pricing,
                              precision=Precision.research(pricing=StagePrecision.compensated("float32", compute, "float32")))
    assert run.today == exact.today
    error, bias = _COMPUTE_BOUNDS[compute]
    for column, reference in zip(run.columns, exact.columns):
        values, reference = np.asarray(column, dtype=np.float64), np.asarray(reference)
        scale = np.max(np.abs(reference))
        assert np.all(np.isfinite(values))
        assert np.max(np.abs(values - reference)) <= error * scale
        assert np.max(np.abs(np.mean(values - reference, axis=0))) <= bias * scale
