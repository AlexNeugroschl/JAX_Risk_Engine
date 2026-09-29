"""
The cross-asset simulation (`engine.simulation.cam`, `scenario_market`, `config`) against
ORE and against the identities an arbitrage-free model must satisfy.

  L1 (exact): the step moments make every traded asset deflated by the LGM numeraire a
     martingale -- domestic and foreign zero bonds, FX, domestic and foreign equities -- as an
     analytic Gaussian expectation, with no Monte Carlo error; `z_0` has variance zeta.
  L1 (statistical): the same on simulated paths, within 4 standard errors, on sloped curves
     (plan 6.2), in float64 and float32.
  L2: the path curves and numeraire against `ORE.LinearGaussMarkovModel` (`discountBond`
     with and without a target curve, `numeraire`); the one-currency step against
     `ORE.IrLgm1fStateProcess`; the square root against QuantLib's `CholeskyDecomposition`.
  Independent derivation: every covariance block equals the integral of the products of the
     states' Brownian loadings, a different route to the same numbers than ORE's expanded
     `CrossAssetAnalytics` formulas.

The CAM's own state process has no usable Python binding (`CrossAssetModel.stateProcess()`
returns an unwrapped pointer), so multi-asset step moments cannot be compared with ORE
directly; the exact martingale identities are what pins them.
"""
import jax
import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.market import CurrencyMarket, EquityMarket, Market, ZeroCurveConfig
from engine.models.curves import ZeroCurve, discount, log_discount
from engine.models.lgm import Sigma
from engine.simulation.cam import (
    CrossAssetModel, EqComponent, FxComponent, IrComponent, _H, _integral_of_square, _piecewise,
    flexible_cholesky, step_moments,
)
from engine.simulation.config import CamConfig, LgmConfig, build_cross_asset_model, simulate
from engine.simulation.scenario_market import DISCOUNT_FLOOR, implied_log_discounts, lgm_numeraire

ASOF = ORE.Date(30, 7, 2026)
PILLARS = [0.0, 1.0, 2.0, 5.0, 10.0, 30.0]
CURVE_SHAPES = {
    "upward": ([0.03, 0.035, 0.04, 0.045, 0.05, 0.05], [0.01, 0.012, 0.018, 0.025, 0.03, 0.03]),
    "inverted": ([0.05, 0.045, 0.04, 0.035, 0.03, 0.03], [0.03, 0.028, 0.02, 0.015, 0.01, 0.01]),
    "flat": ([0.03] * 6, [0.02] * 6),
}
GRID = np.array([0.0, 0.25, 0.7, 1.5, 2.0, 3.2, 4.0])
CORRELATION = np.array([
    [1.0, 0.5, 0.2, 0.3, 0.1],
    [0.5, 1.0, -0.1, 0.1, 0.25],
    [0.2, -0.1, 1.0, 0.15, -0.2],
    [0.3, 0.1, 0.15, 1.0, 0.4],
    [0.1, 0.25, -0.2, 0.4, 1.0],
])


def _sigma(times, values):
    return Sigma(jnp.asarray(times, dtype=jnp.float64), jnp.asarray(values, dtype=jnp.float64))


def _curve(rates):
    return ZeroCurve(jnp.asarray(PILLARS), jnp.asarray(rates))


def _model(shape="upward"):
    """USD (domestic), EUR, EURUSD, a USD and a EUR equity: every covariance block present."""
    usd, eur = (_curve(r) for r in CURVE_SHAPES[shape])
    dividend = ZeroCurve(jnp.asarray([0.0, 30.0]), jnp.asarray([0.015, 0.02]))
    return CrossAssetModel(
        ir=(IrComponent("USD", usd, 0.03, _sigma([1.0, 3.0], [0.010, 0.012, 0.009])),
            IrComponent("EUR", eur, 0.02, _sigma([2.0], [0.008, 0.007]))),
        fx=(FxComponent("EUR", 1.10, _sigma([1.5], [0.10, 0.12])),),
        eq=(EqComponent("SP5", "USD", 100.0, _sigma([], [0.20]), usd, dividend),
            EqComponent("DAX", "EUR", 50.0, _sigma([2.5], [0.25, 0.22]), eur, dividend)),
        correlation=CORRELATION,
    )


def _propagate(model, grid):
    """Mean and covariance of the state at grid[-1] from the exact step moments."""
    moments = step_moments(model, grid)
    mean, cov = model.initial_state(), np.zeros((model.dimension, model.dimension))
    for M, b, C in zip(moments.transition, moments.drift, moments.covariance):
        mean = M @ mean + b
        cov = M @ cov @ M.T + C
    return mean, cov


def _lp(curve, t):
    return float(log_discount(curve, jnp.asarray(t)))


class _Deflated:
    """E[exp(c . X + d)] for the Gaussian state at t, and the log-affine coefficients of
    the traded assets deflated by the domestic numeraire."""

    def __init__(self, model, grid):
        self.model, self.t = model, grid[-1]
        self.mean, self.cov = _propagate(model, grid)

    def expect(self, c, d):
        return np.exp(c @ self.mean + d + 0.5 * c @ self.cov @ c)

    def _lgm(self, i):
        c = self.model.ir[i]
        return _H(c.reversion, self.t), float(_integral_of_square(c.sigma, self.t))

    def numeraire(self):
        """ln N(t) = H z_0 + 1/2 H^2 zeta - ln P(0,t): returns (coefficient on z_0, constant)."""
        H, zeta = self._lgm(0)
        return H, 0.5 * H ** 2 * zeta - _lp(self.model.ir[0].curve, self.t)

    def bond(self, i, T):
        """ln P_i(t, T | z_i): (coefficient on z_i, constant)."""
        c = self.model.ir[i]
        (Ht, zeta), HT = self._lgm(i), _H(c.reversion, T)
        return -(HT - Ht), _lp(c.curve, T) - _lp(c.curve, self.t) - 0.5 * (HT ** 2 - Ht ** 2) * zeta


# =============================================================================
# L1, exact
# =============================================================================
@pytest.mark.parametrize("shape", CURVE_SHAPES)
@pytest.mark.parametrize("maturity_after", [0.5, 3.0, 10.0])
def test_zero_bonds_deflated_by_the_numeraire_are_martingales(shape, maturity_after):
    model = _model(shape)
    x = _Deflated(model, GRID)
    hN, dN = x.numeraire()
    T = x.t + maturity_after
    for i, spot in ((0, 1.0), (1, model.fx[0].spot)):
        b, d = x.bond(i, T)
        c = np.zeros(model.dimension)
        c[i] += b
        c[0] -= hN
        if i > 0:
            c[model.fx_state(0)] = 1.0   # converted to the domestic currency
        expected = spot * np.exp(_lp(model.ir[i].curve, T))
        assert x.expect(c, d - dN) == pytest.approx(expected, rel=2e-15)


@pytest.mark.parametrize("shape", CURVE_SHAPES)
def test_equities_deflated_by_the_numeraire_are_martingales(shape):
    """E[FX(t) S(t) / N(t)] = FX(0) S(0) q(0, t) for a domestic (FX = 1) and a foreign
    equity whose forecast curve is its currency's discount curve."""
    model = _model(shape)
    x = _Deflated(model, GRID)
    hN, dN = x.numeraire()
    for k, eq in enumerate(model.eq):
        c = np.zeros(model.dimension)
        c[model.eq_state(k)] = 1.0
        c[0] -= hN
        fx = 1.0
        if eq.currency != model.domestic_currency:
            c[model.fx_state(model.ir_index(eq.currency) - 1)] = 1.0
            fx = model.fx[model.ir_index(eq.currency) - 1].spot
        expected = fx * eq.spot * np.exp(_lp(eq.dividend_curve, x.t))
        assert x.expect(c, -dN) == pytest.approx(expected, rel=2e-15)


def test_the_domestic_state_is_driftless_with_variance_zeta():
    model = _model()
    mean, cov = _propagate(model, GRID)
    assert mean[0] == 0.0
    assert cov[0, 0] == pytest.approx(float(_integral_of_square(model.ir[0].sigma, GRID[-1])), rel=1e-14)


def test_every_covariance_block_equals_the_integral_of_the_brownian_loadings():
    """Over one step [t0, t1], each state's stochastic part is the integral of a loading
    vector g(s) against the correlated Brownians: z_i has alpha_i e_i; the FX rate of
    currency i adds (H_0(t1)-H_0(s)) alpha_0 e_0 - (H_i(t1)-H_i(s)) alpha_i e_i to its own
    sigma; an equity in currency i adds (H_i(t1)-H_i(s)) alpha_i e_i. So
    Cov = int g_k(s)' rho g_l(s) ds, integrated here by Simpson's rule on each piece between
    the volatility breakpoints (where the integrand is smooth)."""
    model = _model()
    t0, t1 = 0.7, 3.2
    edges = [t0, 1.0, 1.5, 2.0, 2.5, 3.0, t1]
    expected = sum(_loading_covariance(model, a, b, t1) for a, b in zip(edges[:-1], edges[1:]))
    actual = step_moments(model, [0.0, t0, t1]).covariance[1]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-18)


def _loading_covariance(model, a, b, t1):
    """int_a^b g(s)' rho g(s) ds on a piece where every volatility is constant."""
    from scipy.integrate import simpson

    s = np.linspace(a, b, 4001)
    constant = lambda sigma: np.full(s.size, _piecewise(sigma, 0.5 * (a + b)))  # noqa: E731
    d = model.dimension
    G = np.zeros((s.size, d, d))  # G[:, state, brownian]
    alpha = [constant(c.sigma) for c in model.ir]
    H = [_H(c.reversion, s) for c in model.ir]
    H1 = [_H(c.reversion, t1) for c in model.ir]
    for i in range(len(model.ir)):
        G[:, i, i] = alpha[i]
    for j, fx in enumerate(model.fx):
        row, i = model.fx_state(j), j + 1
        G[:, row, 0] += (H1[0] - H[0]) * alpha[0]
        G[:, row, i] -= (H1[i] - H[i]) * alpha[i]
        G[:, row, row] += constant(fx.sigma)
    for k, eq in enumerate(model.eq):
        row, i = model.eq_state(k), model.ir_index(eq.currency)
        G[:, row, i] += (H1[i] - H[i]) * alpha[i]
        G[:, row, row] += constant(eq.sigma)
    return simpson(np.einsum("sab,bc,sdc->sad", G, model.correlation, G), x=s, axis=0)


# =============================================================================
# L2 against ORE
# =============================================================================
def _ore_lgm(curve_rates, reversion, sigma_times, sigma_values):
    ORE.Settings.instance().evaluationDate = ASOF
    dates = [ASOF + round(t * 365) for t in PILLARS]
    handle = ORE.YieldTermStructureHandle(ORE.ZeroCurve(dates, curve_rates, ORE.Actual365Fixed()))
    handle.enableExtrapolation()
    p = ORE.IrLgm1fPiecewiseConstantParametrization(
        ORE.USDCurrency(), handle, ORE.Array(sigma_times), ORE.Array(sigma_values), ORE.Array([]),
        ORE.Array([reversion]))
    return ORE.LinearGaussMarkovModel(p), p


def _ore_handle(rates):
    dates = [ASOF + round(t * 365) for t in PILLARS]
    handle = ORE.YieldTermStructureHandle(ORE.ZeroCurve(dates, rates, ORE.Actual365Fixed()))
    handle.enableExtrapolation()
    return handle


def test_path_curves_and_numeraire_equal_ore_lgm():
    """`discountBond(t, T, x)` for the discount curve, `discountBond(t, T, x, target)` for an
    index curve (ORE's `ModelImpliedYtsFwdFwdCorrected`), and `numeraire(t, x)`."""
    rates, index_rates = CURVE_SHAPES["upward"][0], [0.032, 0.037, 0.043, 0.048, 0.052, 0.053]
    lgm, _ = _ore_lgm(rates, 0.03, [1.0, 3.0], [0.010, 0.012, 0.009])
    sigma = _sigma([1.0, 3.0], [0.010, 0.012, 0.009])
    t = np.array([0.4, 1.7, 4.2])
    taus = np.array([[0.0, 0.5, 2.0, 12.0]] * 3)
    z = jnp.asarray([[-0.03, 0.01, 0.05], [0.02, -0.04, 0.0]])
    for target_rates, handle in ((rates, None), (index_rates, _ore_handle(index_rates))):
        ours = np.asarray(implied_log_discounts(_curve(target_rates), 0.03, sigma, t, taus, z))
        for s in range(z.shape[0]):
            for j in range(t.size):
                for k in range(taus.shape[1]):
                    args = (t[j], t[j] + taus[j, k], float(z[s, j]))
                    ore = lgm.discountBond(*args) if handle is None else lgm.discountBond(*args, handle)
                    assert np.exp(ours[s, j, k]) == pytest.approx(ore, rel=1e-13)
    N = np.asarray(lgm_numeraire(_curve(rates), 0.03, sigma, t, z))
    for s in range(z.shape[0]):
        for j in range(t.size):
            assert N[s, j] == pytest.approx(lgm.numeraire(t[j], float(z[s, j])), rel=1e-13)


def test_path_curves_are_floored_at_ores_minimum_discount_factor():
    sigma = _sigma([], [0.01])
    ours = implied_log_discounts(_curve([0.03] * 6), 0.03, sigma, np.array([1.0]),
                                 np.array([[0.0, 30.0]]), jnp.asarray([[100.0]]))
    assert float(ours[0, 0, 1]) == pytest.approx(np.log(DISCOUNT_FLOOR), rel=1e-15)


def test_one_currency_step_equals_ores_lgm_state_process():
    lgm, _ = _ore_lgm(CURVE_SHAPES["upward"][0], 0.03, [1.0, 3.0], [0.010, 0.012, 0.009])
    process = ORE.IrLgm1fStateProcess(lgm.parametrization())
    usd = IrComponent("USD", _curve(CURVE_SHAPES["upward"][0]), 0.03, _sigma([1.0, 3.0], [0.010, 0.012, 0.009]))
    model = CrossAssetModel(ir=(usd,), fx=(), eq=(), correlation=np.eye(1))
    moments = step_moments(model, GRID)
    for i, (t0, t1) in enumerate(zip(GRID[:-1], GRID[1:])):
        assert moments.drift[i, 0] + 0.37 * moments.transition[i, 0, 0] == pytest.approx(
            process.expectation(t0, 0.37, t1 - t0), abs=1e-16)
        assert moments.covariance[i, 0, 0] == pytest.approx(process.variance(t0, 0.37, t1 - t0), rel=1e-12)


def test_the_square_root_is_quantlibs_flexible_cholesky():
    C = step_moments(_model(), GRID).covariance[3]
    ore = ORE.CholeskyDecomposition(ORE.Matrix([list(map(float, row)) for row in C]), True)
    np.testing.assert_allclose(flexible_cholesky(C), [[ore[i][j] for j in range(C.shape[1])]
                                                      for i in range(C.shape[0])], rtol=1e-13, atol=1e-17)


def test_a_zero_volatility_component_gets_a_zero_column_not_nan():
    C = np.diag([1e-4, 0.0, 4e-4])
    L = flexible_cholesky(C)
    assert np.all(np.isfinite(L)) and L[1, 1] == 0.0
    np.testing.assert_allclose(L @ L.T, C, atol=1e-18)


# =============================================================================
# Simulated paths (L1, statistical) and the configuration
# =============================================================================
PILLAR_CURVES = {"USD": [0.03, 0.035, 0.04, 0.045, 0.05, 0.05], "EUR": [0.01, 0.012, 0.018, 0.025, 0.03, 0.03]}
USD_INDEX = [0.032, 0.037, 0.042, 0.047, 0.052, 0.052]


def _market():
    zc = lambda rates: ZeroCurveConfig(PILLARS, rates)  # noqa: E731
    return Market(
        ASOF,
        {"USD": CurrencyMarket(zc(PILLAR_CURVES["USD"]), {"USD-SIMINDEX-6M": zc(USD_INDEX)}),
         "EUR": CurrencyMarket(zc(PILLAR_CURVES["EUR"]))},
        fx_spots={"EURUSD": 1.1}, equities={"SP5": EquityMarket("USD", 100.0)},
    )


def _config(**overrides):
    fields = dict(
        dates=tuple(ASOF + ORE.Period(k, ORE.Years) for k in (1, 2, 5)), base_currency="USD",
        ir={"USD": LgmConfig(0.03, 0.01), "EUR": LgmConfig(0.02, 0.008)},
        fx_volatilities={"EUR": 0.1}, equity_volatilities={"SP5": 0.2},
        correlations={("IR:USD", "IR:EUR"): 0.5, ("IR:USD", "FX:EURUSD"): 0.2, ("EQ:SP5", "IR:USD"): 0.3},
        samples=4096)
    fields.update(overrides)
    return CamConfig(**fields)


@pytest.mark.parametrize("dtype, se_bound", [(jnp.float64, 4.0), (jnp.float32, 4.0)])
def test_simulated_assets_are_martingales_on_sloped_curves(dtype, se_bound):
    """The permanent check audit M-1 asked for: E[P(t,T)/N(t)] = P(0,T) (the Hull-White
    simulation missed by 4-9% at t=2y on a 3% -> 5% curve), plus the foreign bond and the
    equity, within `se_bound` standard errors."""
    market = _market()
    sm = simulate(market, _config(), dtype=dtype)
    assert sm.numeraire.dtype == dtype and sm.discount["USD"].log_discounts.dtype == dtype
    N = np.asarray(sm.numeraire, dtype=np.float64)

    def check(samples, expected):
        se = samples.std() / np.sqrt(samples.size)
        assert abs(samples.mean() - expected) <= se_bound * se + 1e-6 * abs(expected)

    for j, t in enumerate(sm.times):
        for k in (2, 6, 12):
            tau = float(sm.discount["USD"].tenor_times[j, k])
            ln = lambda ccy: np.asarray(sm.discount[ccy].log_discounts[:, j, k], dtype=np.float64)  # noqa: E731
            check(np.exp(ln("USD")) / N[:, j], float(discount(_curve(PILLAR_CURVES["USD"]), t + tau)))
            fx = np.asarray(sm.fx["EUR"][:, j], dtype=np.float64)
            check(fx * np.exp(ln("EUR")) / N[:, j], 1.1 * float(discount(_curve(PILLAR_CURVES["EUR"]), t + tau)))
        check(np.asarray(sm.equity["SP5"][:, j], dtype=np.float64) / N[:, j], 100.0)


def test_the_index_basis_is_deterministic_on_every_path():
    """ORE's `ModelImpliedYtsFwdFwdCorrected`: index and discount curves move with the same
    state, so ln P_idx(t, t+tau) - ln P(t, t+tau) is the t=0 forward-forward basis on every
    path."""
    sm = simulate(_market(), _config(samples=64))
    usd, idx = _curve(PILLAR_CURVES["USD"]), _curve(USD_INDEX)
    for j, t in enumerate(sm.times):
        tau = np.asarray(sm.discount["USD"].tenor_times[j])
        basis = (np.asarray(log_discount(idx, t + tau)) - float(log_discount(idx, t))
                 - np.asarray(log_discount(usd, t + tau)) + float(log_discount(usd, t)))
        spread = np.asarray(sm.index["USD-SIMINDEX-6M"].log_discounts[:, j] - sm.discount["USD"].log_discounts[:, j])
        np.testing.assert_allclose(spread, np.broadcast_to(basis, spread.shape), atol=1e-14)


def test_zero_volatility_paths_are_todays_forward_curves():
    sm = simulate(_market(), _config(ir={"USD": LgmConfig(0.03, 0.0), "EUR": LgmConfig(0.02, 0.0)}, samples=8))
    usd = _curve(PILLAR_CURVES["USD"])
    for j, t in enumerate(sm.times):
        tau = np.asarray(sm.discount["USD"].tenor_times[j])
        expected = np.asarray(log_discount(usd, t + tau)) - float(log_discount(usd, t))
        np.testing.assert_allclose(np.asarray(sm.discount["USD"].log_discounts[:, j]),
                                   np.broadcast_to(expected, (8, tau.size)), atol=1e-15)
        np.testing.assert_allclose(np.asarray(sm.numeraire[:, j]), np.exp(-float(log_discount(usd, t))), rtol=1e-15)


def test_tenor_times_are_measured_from_each_date_with_period_arithmetic():
    sm = simulate(_market(), _config(dates=(ORE.Date(28, 2, 2027),), samples=4))
    one_year = float(sm.discount["USD"].tenor_times[0, 3])  # "1Y" is the third configured tenor
    assert one_year == pytest.approx((ORE.Date(28, 2, 2028) - ORE.Date(28, 2, 2027)) / 365.0, rel=1e-15)


class TestConfigurationRefusals:
    def test_fx_volatilities_must_cover_every_foreign_currency(self):
        with pytest.raises(ValueError, match="fx_volatilities"):
            _config(fx_volatilities={})

    def test_the_base_currency_needs_a_model(self):
        with pytest.raises(ValueError, match="base_currency"):
            _config(base_currency="GBP")

    def test_dates_must_increase(self):
        with pytest.raises(ValueError, match="increase"):
            _config(dates=(ASOF + 365, ASOF + 100))

    def test_dates_must_be_after_the_as_of_date(self):
        with pytest.raises(ValueError, match="after the as-of date"):
            simulate(_market(), _config(dates=(ASOF,)))

    def test_a_correlation_on_an_unknown_factor_is_refused(self):
        with pytest.raises(KeyError, match="IR:GBP"):
            build_cross_asset_model(_market(), _config(correlations={("IR:GBP", "IR:USD"): 0.1}))

    def test_a_non_psd_correlation_is_refused(self):
        bad = {("IR:USD", "IR:EUR"): 0.99, ("IR:USD", "FX:EURUSD"): 0.99, ("IR:EUR", "FX:EURUSD"): -0.99}
        with pytest.raises(ValueError, match="positive semi-definite"):
            build_cross_asset_model(_market(), _config(correlations=bad))

    def test_fx_components_must_follow_the_ir_currencies(self):
        m = _model()
        with pytest.raises(ValueError, match="FX components"):
            CrossAssetModel(ir=m.ir, fx=(), eq=(), correlation=np.eye(2))
