"""
The root solver (`engine.solvers.roots`, decision A-21) and the roots the engine
solves with it.

  * The solvers on functions with known roots: both to rounding, elementwise over a batch,
    increasing and decreasing; a bounded root beyond an end is that end; an unbounded one far
    outside its starting window is found; Newton falls back to bisection where its step would
    leave the bracket; state carried between evaluations; float32 stays float32; a fixed
    number of steps (no data-dependent loop).
  * `implicit_root`'s first and second derivatives on roots with known derivatives.
  * On the parity markets (the shared sloped portfolio today and on simulated LGM paths, and
    the ORE-calibration tests' market): every bootstrap bucket reprices its helper to
    rounding under Newton and equals Bisection's; Jamshidian's x* and the standalone
    `/calibration/lgm` bootstrap likewise. Newton's step counts have two steps of margin: two
    fewer reach the same roots.
  * The setting on every configuration that solves a root, with its API field, refused by
    name when unknown; ORE parity at t=0 under the reference solver.
  * The path dates batched (`path_sigmas`): equal to one date at a time; one program per
    basket shape and power of two of dates; a date with an exercise left and no helper.
"""
import dataclasses
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import ORE
import pytest
from pydantic import ValidationError

import engine.calibration.basket as basket_module
import engine.calibration.lgm as lgm_module
import engine.calibration.ore_lgm as ore_lgm
import engine.pricing.jamshidian as jamshidian
from engine.api.requests import (
    HullWhiteConfigSchema, JamshidianEngineSchema, LgmConfigSchema, LgmEngineSchema,
)
from engine.calibration.basket import build_coterminal_basket
from engine.calibration.lgm import calibrate_lgm_sigma
from engine.calibration.ore_lgm import basket_vols, bootstrap_sigma, build_basket
from engine.market_data.curves import ZeroCurve
from engine.market_data.curves import ZeroCurve as HwZeroCurve
from engine.market_data.market import index_name
from engine.market_simulation.config import CamConfig, HullWhiteConfig, LgmConfig, simulate
from engine.pricing.bermudan import bermudan_cube, calibration_basket, contract_exercise_dates, path_sigmas
from engine.pricing.config import JamshidianEngineConfig, LgmSwaptionEngineConfig, PricingConfig
from engine.pricing.cube import value_today
from engine.pricing.european import european_terms, volatility_on_path
from engine.pricing.legs import path_fixings
from engine.run import PortfolioRequest, RunConfig, price_portfolio
from engine.solvers import roots
from engine.solvers.roots import SOLVERS, Steps, implicit_root, solve
from tests.support import portfolio as shared
from tests.support.compiles import count_compiles

STEPS = Steps(bisection=100, newton=12)
EPS = float(np.finfo(np.float64).eps)


def _solve(f, params, start, solver, **kw):
    return solve(f, params, jnp.asarray(start, dtype=jnp.float64), solver=solver, steps=STEPS, **kw)


# =============================================================================
# The solvers
# =============================================================================
@pytest.mark.parametrize("solver", SOLVERS)
class TestSolve:
    def test_increasing_and_decreasing_roots_to_rounding_over_a_batch(self, solver):
        c = jnp.asarray([-7.0, -0.2, 0.0, 0.5, 3.0])
        up = _solve(lambda x, c: jnp.sinh(x) - c, c, jnp.zeros(5), solver, increasing=True, window=1.0)
        down = _solve(lambda x, c: c - jnp.sinh(x), c, jnp.zeros(5), solver, increasing=False, window=1.0)
        for root in (up, down):
            np.testing.assert_allclose(np.asarray(root), np.arcsinh(np.asarray(c)), rtol=4 * EPS, atol=4 * EPS)

    def test_a_bounded_root_beyond_an_end_is_that_end(self, solver):
        """A bootstrap bucket out of reach ends at the bracket's edge (and is flagged there);
        Newton returns the edge itself, bisection converges to it."""
        c = jnp.asarray([-1.0, 0.25, 2.0])
        root = np.asarray(_solve(lambda x, c: x - c, c, jnp.full(3, 0.5), solver, increasing=True, bracket=(0.0, 1.0)))
        np.testing.assert_allclose(root, [0.0, 0.25, 1.0], atol=1e-15)
        if solver == "Newton":
            assert root[0] == 0.0 and root[2] == 1.0

    @pytest.mark.parametrize("target", [25.0, -1e6, 3e12])
    def test_an_unbounded_root_far_outside_the_window_is_bracketed(self, solver, target):
        root = _solve(lambda x, c: c - x, jnp.asarray(target), 0.0, solver, increasing=False, window=2.0)
        assert float(root) == pytest.approx(target, rel=1e-13)

    def test_float32_stays_float32_under_strict_promotion(self, solver):
        c = jnp.asarray([0.1, 0.7], dtype=jnp.float32)
        with jax.numpy_dtype_promotion("strict"):
            root = solve(lambda x, c: jnp.sinh(x) - c, c, jnp.zeros(2, dtype=jnp.float32), solver=solver,
                         steps=Steps(40, 8), increasing=True, window=1.0)
        assert root.dtype == jnp.float32
        np.testing.assert_allclose(np.asarray(root), np.arcsinh([0.1, 0.7]), rtol=4e-7)

    def test_a_fixed_number_of_steps_and_no_data_dependent_loop(self, solver):
        """A data-dependent stop is a `while` whose predicate a GPU reads on the host each
        step (decision A-21); the solver's loop is a fixed-length scan."""
        jaxpr = str(jax.make_jaxpr(lambda c: _solve(lambda x, c: x ** 3 - c, c, 0.0, solver, increasing=True,
                                                    window=1.0))(jnp.asarray(0.5)))
        assert "scan" in jaxpr and "while" not in jaxpr

    def test_the_root_has_no_derivative(self, solver):
        """`solve` stops the tangents of its inputs: a nested solve under an outer Newton's
        `jvp` (y* inside a bucket's solve) computes no tangents in its loop."""
        grad = jax.grad(lambda c: _solve(lambda x, c: x - c, c, 0.0, solver, increasing=True, window=1.0))(0.4)
        assert float(grad) == 0.0


class TestNewton:
    def test_it_bisects_where_its_step_would_leave_the_bracket(self):
        """arctan's Newton steps diverge from |x - c| > 1.39; safeguarded, it converges."""
        f = lambda x, c: jnp.arctan(x - c)  # noqa: E731
        x = 15.0
        for _ in range(3):   # unsafeguarded: each step lands further out
            x = x - float(f(x, 0.3)) * (1.0 + (x - 0.3) ** 2)
        assert abs(x) > 1e3
        root = _solve(f, 0.3, 15.0, "Newton", increasing=True, bracket=(-20.0, 20.0))
        assert float(root) == pytest.approx(0.3, abs=4 * EPS)

    def test_more_steps_never_lose_a_converged_root(self):
        """y*'s shape (a coupon bond less a strike: exponentials of several rates, its root near
        0 in a window of 1), for 4,001 strikes: a root once found stays at its rounding whatever
        the count. Red under `rtsafe`'s second safeguard (bisect unless a step halves an earlier
        one), tried on 2026-10-07: at the root a rounding-level step need not halve, and the
        bisection step across the window, still wide on one side, carried the iterate away (the
        worst root 0.5 off after 5 steps, 3e-11 after 40)."""
        rates = jnp.arange(1.0, 11.0)
        amounts = jnp.full(10, 0.03).at[-1].add(1.0)

        def f(x, k):
            return jnp.sum(amounts * jnp.exp(-rates * x[..., None] - 0.5e-4 * rates ** 2), axis=-1) - k

        strikes = jnp.linspace(1.15, 1.35, 4001)
        reference = _solve(f, strikes, jnp.zeros_like(strikes), "Bisection", increasing=False, window=1.0)
        for n in (5, 9, 40):
            root = solve(f, strikes, jnp.zeros_like(strikes), solver="Newton", steps=Steps(0, n), increasing=False,
                         window=1.0)
            np.testing.assert_allclose(np.asarray(root), np.asarray(reference), rtol=0, atol=4 * EPS, err_msg=n)

    def test_a_zero_derivative_is_a_bisection_step(self):
        root = _solve(lambda x, c: x ** 3 - c, 1e-3, 0.0, "Newton", increasing=True, window=1.0)
        assert float(root) == pytest.approx(0.1, rel=4 * EPS)

    def test_a_non_finite_start_starts_in_the_bracket(self):
        root = _solve(lambda x, c: x - c, 0.3, jnp.nan, "Newton", increasing=True, bracket=(0.0, 1.0))
        assert float(root) == pytest.approx(0.3, abs=4 * EPS)


def test_an_unknown_solver_is_refused_by_name():
    with pytest.raises(ValueError, match="solver must be one of"):
        roots.check_solver("Brent")
    with pytest.raises(ValueError, match="bracket or a window"):
        solve(lambda x, c: x - c, 0.0, jnp.asarray(0.0), solver="Newton", steps=STEPS, increasing=True)


@pytest.mark.parametrize("solver", SOLVERS)
class TestImplicitRoot:
    """A root's derivative in its parameters by the implicit function theorem (Jamshidian's
    x*, the co-terminal basket's x*); on roots with closed-form derivatives."""

    @staticmethod
    def root(g, params, solver):
        return implicit_root(g, params, jnp.zeros(jnp.shape(params)), solver=solver, steps=STEPS, increasing=False,
                             window=2.0)

    def test_linear_root(self, solver):
        """g(x, c) = c - x: x* = c, dx*/dc = 1."""
        g = lambda x, c: c - x  # noqa: E731
        c = jnp.asarray(0.5)
        assert float(self.root(g, c, solver)) == pytest.approx(0.5, abs=1e-12)
        assert float(jax.grad(lambda p: self.root(g, p, solver))(c)) == pytest.approx(1.0, abs=1e-12)

    def test_cubic_root_first_and_second_derivative(self, solver):
        """g(x, c) = c - x^3: x* = c^(1/3), with the closed-form first and second derivatives
        (the rule is itself differentiable, which Gamma needs)."""
        g = lambda x, c: c - x ** 3  # noqa: E731
        c = jnp.asarray(0.5)
        root = lambda p: self.root(g, p, solver)  # noqa: E731
        assert float(root(c)) == pytest.approx(0.5 ** (1 / 3), rel=1e-12)
        assert float(jax.grad(root)(c)) == pytest.approx(1 / (3 * 0.5 ** (2 / 3)), rel=1e-10)
        assert float(jax.hessian(root)(c)) == pytest.approx(-2 / 9 * 0.5 ** (-5 / 3), rel=1e-8)

    def test_a_batch_of_roots_each_with_its_own_derivative(self, solver):
        """Elementwise over a batch (paths): each root depends on its own parameter only."""
        g = lambda x, c: c - jnp.sinh(x)  # noqa: E731
        c = jnp.asarray([-1.0, 0.2, 3.0])
        jac = jax.jacobian(lambda p: self.root(g, p, solver))(c)
        np.testing.assert_allclose(np.diag(jac), 1 / np.sqrt(1 + np.asarray(c) ** 2), rtol=1e-10)
        np.testing.assert_allclose(jac - np.diag(np.diag(jac)), 0.0, atol=1e-14)

    def test_a_root_far_outside_the_first_window_is_bracketed(self, solver):
        g = lambda x, c: c - x  # noqa: E731
        assert float(self.root(g, jnp.asarray(25.0), solver)) == pytest.approx(25.0, abs=1e-10)


# =============================================================================
# The engine's roots on the parity markets
# =============================================================================
ASOF = shared.ASOF
OPTIONS = [cfg for cfg in shared.trades().values() if type(cfg).__name__ in ("BermudanSwaptionConfig",
                                                                             "AmericanSwaptionConfig")]
DATES = tuple(ASOF + ORE.Period(m, ORE.Months) for m in (3, 6, 12, 24, 36))

#: Measured on these markets and on Hull-White paths (2026-10-07): the largest gap between a
#: bucket's model and market value is 7.5e-12 of the market value under either solver (a helper
#: far out of the money on a path, whose value is small beside its terms' rounding), and Newton's
#: volatilities equal Bisection's to 7e-13 (a root is resolvable to the residual's rounding over
#: its slope). Newton's largest residual is 1.7 times Bisection's (2.8 with two steps fewer):
#: both are rounding; this factor bounds it, above this floor.
RESIDUAL_FACTOR, RESIDUAL_FLOOR = 4.0, 1e-13
AGREEMENT = 4e-12


@pytest.fixture(scope="module")
def problems():
    """`(label, basket, disc, index, vols)` of every bootstrap on the parity markets: each
    option of the shared portfolio today and on 64 LGM paths on every path date, and the
    ORE-calibration tests' tenor basket."""
    market = shared.market()
    surface = market.swaption_vols("USD")
    disc = ZeroCurve.from_config(market.currency("USD").discount_curve)
    index = ZeroCurve.from_config(market.index_curve("USD", index_name("USD", 6)))
    sm = simulate(market, CamConfig(dates=DATES, base_currency="USD", samples=64, seed=3,
                                    ir={"USD": LgmConfig(0.03, 0.01, ("1Y", "2Y", "5Y"), ("9Y", "8Y", "5Y"))}))
    out = []
    for cfg in OPTIONS:
        engine = LgmSwaptionEngineConfig()
        basket = calibration_basket(cfg, engine, ASOF, ASOF)
        out.append((f"{cfg.trade_id} today", basket, disc, index, basket_vols(basket, surface, ASOF)))
        for j, date in enumerate(sm.dates):
            basket = calibration_basket(cfg, engine, date, ASOF)
            if basket:
                vols = [volatility_on_path(surface, ASOF, date, b.vol_option_time, b.vol_swap_length, "ForwardVariance")
                        for b in basket]
                out.append((f"{cfg.trade_id} on {date}", basket, sm.discount["USD"].on_date(j),
                            sm.index[index_name("USD", 6)].on_date(j), np.asarray(vols)))
    tenors = build_basket(ASOF, ["1Y", "2Y", "3Y", "5Y"], ["9Y", "8Y", "7Y", "5Y"])
    out.append(("tenor basket today", tenors, disc, index, basket_vols(tenors, surface, ASOF)))
    return out


@contextmanager
def newton_steps(monkeypatch, **counts):
    """Each `module.CONSTANT` of `counts` (`"ore_lgm.SIGMA_STEPS"`: a Newton count, or a change
    of it when negative) set for the block, with JAX's caches cleared on entry and exit: jitted
    programs read the counts when traced, and wrappers of one function share their traces."""
    modules = {"ore_lgm": ore_lgm, "basket": basket_module, "lgm": lgm_module, "jamshidian": jamshidian}
    for name, count in counts.items():
        module, constant = name.split("__", 1)
        steps = getattr(modules[module], constant)
        monkeypatch.setattr(modules[module], constant, Steps(steps.bisection, steps.newton + count if count < 0
                                                              else count))
    jax.clear_caches()
    try:
        yield
    finally:
        monkeypatch.undo()
        jax.clear_caches()


def _bootstraps(problems, solver):
    return [bootstrap_sigma(basket, d, i, v, 0.0, solver) for _, basket, d, i, v in problems]


@pytest.fixture(scope="module")
def references(problems):
    return _bootstraps(problems, "Bisection")


def _gap(result):
    return np.max(np.abs(np.asarray(result.model) - np.asarray(result.market)) / np.asarray(result.market))


class TestBootstrapRoots:
    def test_newton_reprices_every_helper_to_rounding(self, problems, references):
        """As closely as the reference does: its residual is the rounding of the helper's price."""
        for (label, *_), result, reference in zip(problems, _bootstraps(problems, "Newton"), references):
            assert _gap(result) <= max(RESIDUAL_FACTOR * _gap(reference), RESIDUAL_FLOOR), label
            assert not np.any(np.asarray(result.hit_ceiling)), label

    def test_newton_s_volatilities_are_bisection_s(self, problems, references):
        for (label, *_), newton, bisection in zip(problems, _bootstraps(problems, "Newton"), references):
            np.testing.assert_allclose(np.asarray(newton.values), np.asarray(bisection.values), rtol=AGREEMENT,
                                       err_msg=label)

    def test_two_newton_steps_fewer_reach_the_same_roots(self, problems, monkeypatch):
        """The step counts' margin: with two fewer steps for the bucket and for y*, every
        bucket is already at its full count's root (and with one step each it is not, which
        shows the counts were changed)."""
        full = _bootstraps(problems, "Newton")
        with newton_steps(monkeypatch, ore_lgm__SIGMA_STEPS=-2, ore_lgm__Y_STAR_STEPS=-2):
            fewer = _bootstraps(problems, "Newton")
        with newton_steps(monkeypatch, ore_lgm__SIGMA_STEPS=1, ore_lgm__Y_STAR_STEPS=1):
            one = _bootstraps(problems, "Newton")
        for (label, *_), a, b in zip(problems, full, fewer):
            np.testing.assert_allclose(np.asarray(b.values), np.asarray(a.values), rtol=AGREEMENT, err_msg=label)
        assert max(np.max(np.abs(np.asarray(b.values) / np.asarray(a.values) - 1)) for a, b in zip(full, one)) > 1e-6

    def test_an_unattainable_volatility_ends_at_the_ceiling_under_newton(self):
        basket = build_basket(ASOF, ["1Y", "2Y"], ["9Y", "8Y"])
        market = shared.market()
        disc = ZeroCurve.from_config(market.currency("USD").discount_curve)
        result = bootstrap_sigma(basket, disc, disc, np.array([0.9, 0.9]), 0.0, "Newton")
        assert np.all(np.asarray(result.hit_ceiling))
        assert np.all(np.asarray(result.values) == ore_lgm.SIGMA_BRACKET[1])


@pytest.fixture(scope="module")
def jamshidian_cases():
    """The shared European as payer and receiver, struck in, at and out of the money, on
    today's curve and on 64 LGM paths at every date before its expiry."""
    market = shared.market()
    european = next(t for t in shared.trades().values() if type(t).__name__ == "SwaptionConfig")
    disc = ZeroCurve.from_config(market.currency("USD").discount_curve)
    sm = simulate(market, CamConfig(dates=DATES, base_currency="USD", samples=64, seed=3,
                                    ir={"USD": LgmConfig(0.03, 0.01)}))
    cases = []
    for payer in (True, False):
        for rate in (0.01, 0.04, 0.08):
            cfg = dataclasses.replace(european, payer=payer, fixed_rate=rate, settlement="Physical",
                                      floating_spread=0.0)
            terms = european_terms(cfg, ASOF)
            cases.append((terms, disc, 0.0))
            cases += [(terms, sm.discount["USD"].on_date(j), t) for j, t in enumerate(sm.times)
                      if terms.expiry_time > t]
    return cases


#: A Jamshidian NPV is a sum of terms of the nominal's size, so it carries their rounding, which
#: x*'s rounding moves at first order (the price is not stationary in x*): measured, the two
#: solvers' NPVs differ by at most 6e-16 of the nominal (on 256 paths, strikes from 0% to 15%;
#: where an option's value is 1e-10 of its terms, Newton's x* has residual 0 and Bisection's
#: 2.5e-16 of them).
NPV_AGREEMENT = 1e-14


def test_jamshidian_s_x_star_is_the_same_under_both_solvers(jamshidian_cases):
    npv = jax.jit(jamshidian.jamshidian_npv, static_argnums=1)
    for terms, disc, t in jamshidian_cases:
        values = {s: np.asarray(npv(terms, JamshidianEngineConfig(0.03, 0.01, s), disc, t)) for s in SOLVERS}
        np.testing.assert_allclose(values["Newton"], values["Bisection"], atol=NPV_AGREEMENT * terms.nominal, rtol=0)


@pytest.mark.parametrize("tenor, forward", [("5Y", 0), ("10Y", 2), ("30Y", 0)])
@pytest.mark.parametrize("payer", [True, False], ids=["payer", "receiver"])
def test_jamshidian_s_x_star_at_extreme_strikes(tenor, forward, payer):
    """A trade's strike is unbounded: from -50% (coupons of the nominal's sign cancelling it, a
    coupon bond not monotone across the window) to 200%, Newton's x* is Bisection's to the
    formula's own resolution (1e-11 of the nominal at worst, where it cancels; -99% is
    tests/test_jamshidian.py's, against the exact root)."""
    from tests.test_jamshidian import _make_cfg, _price_at_t0
    for rate in (-0.5, -0.2, -0.05, 0.0, 0.03, 0.1, 0.6, 2.0):
        cfg = _make_cfg(rate, payer, tenor, forward)
        if rate == -0.5 and payer and tenor == "30Y":
            continue  # the formula cancels away: Bisection's value is itself 3% off the exact root's
        values = {s: _price_at_t0(cfg, JamshidianEngineConfig(0.03, 0.01, s)) for s in SOLVERS}
        assert abs(values["Newton"] - values["Bisection"]) <= 1e-11 * cfg.notional, (rate, values)


def _standalone_baskets():
    curves = [HwZeroCurve(jnp.asarray([0.0, 1.0, 5.0, 10.0]), jnp.asarray([0.02, 0.025, 0.035, 0.04])),
              HwZeroCurve(jnp.asarray([0.0, 30.0]), jnp.asarray([0.03, 0.03]))]
    return [(curve, build_coterminal_basket([1.0, 2.0, 3.0, 4.0], 5.0, 1e6, payer, vols, curve, ASOF))
            for curve in curves for payer in (True, False)
            for vols in ([0.008, 0.0085, 0.009, 0.0088], [0.02, 0.005, 0.012, 0.01])]


def _gap_of(result):
    return np.max(np.abs(np.asarray(result.model_prices) - np.asarray(result.market_prices))
                  / np.asarray(result.market_prices))


def test_the_standalone_bootstrap_is_the_same_under_both_solvers():
    """`POST /calibration/lgm`'s bootstrap (x* inside each bucket's price), on a sloped and a
    flat curve, payer and receiver, a smooth and a spiked volatility term structure."""
    for curve, targets in _standalone_baskets():
        newton, bisection = (calibrate_lgm_sigma(targets, curve, 0.03, s) for s in ("Newton", "Bisection"))
        np.testing.assert_allclose(np.asarray(newton.sigma.values), np.asarray(bisection.sigma.values),
                                   rtol=AGREEMENT)
        assert _gap_of(newton) <= max(RESIDUAL_FACTOR * _gap_of(bisection), RESIDUAL_FLOOR)


def test_the_x_star_step_counts_have_two_steps_of_margin(jamshidian_cases, monkeypatch):
    """Two fewer steps for x* (and the standalone bootstrap's buckets) reach the same roots;
    one step does not, which shows the counts were changed."""
    model = JamshidianEngineConfig(0.03, 0.01)

    def roots_now():
        npv = jax.jit(jamshidian.jamshidian_npv, static_argnums=1)
        return ([np.asarray(calibrate_lgm_sigma(t, c, 0.03, "Newton").sigma.values) for c, t in _standalone_baskets()],
                [np.asarray(npv(terms, model, disc, t)) for terms, disc, t in jamshidian_cases])

    full = roots_now()
    with newton_steps(monkeypatch, basket__X_STAR_STEPS=-2, lgm___SIGMA_STEPS=-2, jamshidian__X_STAR_STEPS=-2):
        fewer = roots_now()
    with newton_steps(monkeypatch, basket__X_STAR_STEPS=1, lgm___SIGMA_STEPS=1, jamshidian__X_STAR_STEPS=1):
        one = roots_now()
    for a, b in zip(full[0], fewer[0]):
        np.testing.assert_allclose(b, a, rtol=AGREEMENT)
    for a, b, (terms, _, _) in zip(full[1], fewer[1], jamshidian_cases):
        np.testing.assert_allclose(b, a, atol=NPV_AGREEMENT * terms.nominal, rtol=0)
    assert any(not np.allclose(a, b, rtol=1e-8, atol=0) for a, b in zip(full[0] + full[1], one[0] + one[1]))


@pytest.mark.parametrize("name", [cfg.trade_id for cfg in OPTIONS])
def test_the_reference_solver_keeps_ore_parity_today(name):
    """The calibrated Bermudans and the American equal ORE at t=0 under `"Bisection"` too
    (tests/test_shared_portfolio.py checks the default)."""
    cfg = shared.trades()[name]
    engine = LgmSwaptionEngineConfig(solver="Bisection")
    ours = value_today([cfg], shared.market(), "USD", PricingConfig(bermudan=engine, american=engine))[0]
    assert ours == pytest.approx(shared.ore_npv(cfg), rel=1e-9)


# =============================================================================
# The setting
# =============================================================================
class TestTheSetting:
    def test_newton_is_every_configuration_s_default(self):
        assert roots.DEFAULT_SOLVER == "Newton"
        assert LgmSwaptionEngineConfig().solver == "Newton"
        assert JamshidianEngineConfig(0.03, 0.01).solver == "Newton"
        assert LgmConfig(0.03).solver == HullWhiteConfig(0.03).solver == "Newton"

    @pytest.mark.parametrize("build, field", [
        (lambda: LgmSwaptionEngineConfig(solver="Brent"), "LgmSwaptionEngineConfig.solver"),
        (lambda: JamshidianEngineConfig(0.03, 0.01, "Brent"), "JamshidianEngineConfig.solver"),
        (lambda: LgmConfig(0.03, solver="Brent"), "LgmConfig.solver"),
        (lambda: HullWhiteConfig(0.03, solver="Brent"), "HullWhiteConfig.solver"),
        (lambda: calibrate_lgm_sigma([], None, 0.03, "Brent"), "solver"),
    ])
    def test_an_unknown_solver_is_refused_naming_the_field(self, build, field):
        with pytest.raises(ValueError, match=f"{field} must be one of"):
            build()

    def test_every_api_field_reaches_its_configuration(self):
        assert LgmEngineSchema(solver="Bisection").to_dataclass().solver == "Bisection"
        assert JamshidianEngineSchema(reversion=0.03, volatility=0.01, solver="Bisection").to_dataclass().solver \
            == "Bisection"
        assert LgmConfigSchema(reversion=0.03, solver="Bisection").to_dataclass().solver == "Bisection"
        hull_white = HullWhiteConfigSchema(model="HullWhite", reversion=0.03, solver="Bisection").to_dataclass()
        assert isinstance(hull_white, HullWhiteConfig) and hull_white.solver == "Bisection"
        assert LgmEngineSchema().to_dataclass().solver == "Newton"
        with pytest.raises(ValidationError):
            LgmEngineSchema(solver="Brent")

    def test_the_standalone_route_takes_the_solver(self, test_client):
        body = {"evaluation_date": shared.ASOF_ISO, "exercise_times": [1.0, 2.0, 3.0, 4.0], "final_maturity_time": 5.0,
                "notional": 1_000_000.0, "payer": True, "market_vols": [0.0080, 0.0088, 0.0095, 0.0100],
                "zero_curve": {"times": [0.0, 1.0, 5.0, 10.0], "rates": [0.02, 0.025, 0.035, 0.04]}, "hw_a": 0.03}
        values = {s: test_client.post("/calibration/lgm", json={**body, "solver": s}).json()["sigma_values"]
                  for s in SOLVERS}
        np.testing.assert_allclose(values["Newton"], values["Bisection"], rtol=AGREEMENT)
        assert test_client.post("/calibration/lgm", json={**body, "solver": "Brent"}).status_code == 422


# =============================================================================
# The path dates batched
# =============================================================================
ENGINE = LgmSwaptionEngineConfig(n_per_std=8, std_devs=4.0)
PATH_DATES = tuple(ASOF + ORE.Period(m, ORE.Months) for m in (1, 2, 3, 6, 9, 12, 15, 18, 24))


@pytest.fixture(scope="module")
def path_market():
    market = shared.market()
    return market, simulate(market, CamConfig(dates=PATH_DATES, base_currency="USD", samples=16, seed=5,
                                              ir={"USD": LgmConfig(0.03, 0.01)}))


@pytest.mark.parametrize("solver", SOLVERS)
def test_dates_batched_equal_one_date_at_a_time(path_market, solver):
    """Every date's volatilities equal its own bootstrap on that date's path curves: under
    Newton, batched over dates, to the root's resolution (XLA vectorizes another shape); under
    the reference, which keeps one date per call, bit for bit."""
    market, sm = path_market
    engine = dataclasses.replace(ENGINE, solver=solver)
    surface = market.swaption_vols("USD")
    for cfg in OPTIONS:
        sigmas = path_sigmas(cfg, engine, market, sm, "ForwardVariance")
        assert sorted(sigmas) == [j for j, d in enumerate(sm.dates) if any(e > d for e in contract_exercise_dates(cfg))]
        for j, sigma in sigmas.items():
            basket = calibration_basket(cfg, engine, sm.dates[j], ASOF)
            vols = [volatility_on_path(surface, ASOF, sm.dates[j], b.vol_option_time, b.vol_swap_length,
                                       "ForwardVariance") for b in basket]
            one = bootstrap_sigma(basket, sm.discount["USD"].on_date(j), sm.index[index_name("USD", 6)].on_date(j),
                                  np.asarray(vols), engine.reversion, solver)
            np.testing.assert_allclose(np.asarray(sigma.times), one.times, rtol=0, atol=0)
            np.testing.assert_allclose(np.asarray(sigma.values), np.asarray(one.values),
                                       rtol=AGREEMENT if solver == "Newton" else 0,
                                       err_msg=f"{cfg.trade_id} on {sm.dates[j]}")


def test_one_bootstrap_program_per_basket_shape_and_power_of_two(path_market):
    """A Bermudan whose dates group differently, but into the same powers of two, compiles no
    new bootstrap; the dates are not a program each."""
    market, sm = path_market
    cfg = next(c for c in OPTIONS if type(c).__name__ == "BermudanSwaptionConfig")
    path_sigmas(cfg, ENGINE, market, sm, "ForwardVariance")
    later = dataclasses.replace(sm, dates=sm.dates[1:], times=sm.times[1:],
                                discount={k: _drop_first_date(v) for k, v in sm.discount.items()},
                                index={k: _drop_first_date(v) for k, v in sm.index.items()})
    with count_compiles() as counter:
        path_sigmas(cfg, ENGINE, market, later, "ForwardVariance")
    assert counter["jit__bootstrap_bucket"] == 0, dict(counter)


def _drop_first_date(curves):
    return dataclasses.replace(curves, tenor_times=curves.tenor_times[1:], log_discounts=curves.log_discounts[:, 1:])


def test_a_date_with_an_exercise_left_and_no_helper_keeps_the_engine_volatility():
    """An American on a date after its window's last reference-grid date (2031-01-30) and
    before its last exercise (2031-02-03): no helper is left. It raised `ValueError: Need at
    least one array to stack` before 2026-10-07; it prices on the engine's volatility, as a
    calibration today with no helper does."""
    cfg = shared.trades()["american-payer"]
    market = shared.market()
    date = ORE.Date(1, 2, 2031)
    assert not calibration_basket(cfg, ENGINE, date, ASOF) and cfg.last_exercise_date > date
    sm = simulate(market, CamConfig(dates=(ORE.Date(30, 7, 2027), date), base_currency="USD", samples=16,
                                    ir={"USD": LgmConfig(0.03, 0.01)}))
    sigmas = path_sigmas(cfg, ENGINE, market, sm, "ForwardVariance")
    assert np.asarray(sigmas[1].values).tolist() == [ENGINE.volatility]
    fixings = path_fixings(6, ASOF, sm.dates, sm.times, sm.index[index_name("USD", 6)])
    cube = np.asarray(bermudan_cube(cfg, ENGINE, market, sm, fixings, "ForwardVariance"))
    assert cube.shape == (16, 2) and np.all(np.isfinite(cube)) and np.all(cube[:, 1] >= 0.0)
    result = price_portfolio(PortfolioRequest(market=market, trades=[cfg], config=RunConfig(
        simulation=CamConfig(dates=sm.dates, base_currency="USD", samples=16, ir={"USD": LgmConfig(0.03, 0.01)}),
        pricing=PricingConfig(american=ENGINE))))
    assert np.all(np.isfinite(np.asarray(result.npv_cube)))
