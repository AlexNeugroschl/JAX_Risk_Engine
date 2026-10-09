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
     `ORE.IrLgm1fStateProcess`; the square root against QuantLib's `CholeskyDecomposition`;
     a Hull-White currency's path curves against QuantLib's `HullWhite.discountBond`.

Every model-level identity runs for both IR parametrizations, the Hagan LGM and the Hull-White
model (I-42: the Hull-White model's simulated curves are its own bond prices).
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
from engine.models.lgm import Sigma, hull_white_zeta, zeta as hagan_zeta
from engine.precision import Precision
from engine.simulation.cam import (
    CrossAssetModel, EqComponent, FxComponent, IrComponent, _H, _integral_of_square, _ir_alpha, _ir_zeta,
    _piecewise, flexible_cholesky, step_moments,
)
from engine.simulation.config import CamConfig, HullWhiteConfig, LgmConfig, build_cross_asset_model, simulate
from engine.simulation.scenario_market import (
    DISCOUNT_FLOOR, as_of_tenor_times, implied_log_discounts, lgm_numeraire, tenor_times,
)


def _value_times(sm):
    """`[D, K+1]`: the times each date's curve values are computed at (the tenors from that
    date, `nextPath`), not the times the market holds them at (`ScenarioCurves.tenor_times`, the
    as-of date's, I-84)."""
    return tenor_times(sm.dates, _config().curve_tenors)

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


VOLATILITY_TYPES = ("Hagan", "HullWhite")


def _model(shape="upward", volatility_type="Hagan"):
    """USD (domestic), EUR, EURUSD, a USD and a EUR equity: every covariance block present.
    Both currencies take `volatility_type`'s parametrization."""
    usd, eur = (_curve(r) for r in CURVE_SHAPES[shape])
    dividend = ZeroCurve(jnp.asarray([0.0, 30.0]), jnp.asarray([0.015, 0.02]))
    return CrossAssetModel(
        ir=(IrComponent("USD", usd, 0.03, _sigma([1.0, 3.0], [0.010, 0.012, 0.009]), volatility_type),
            IrComponent("EUR", eur, 0.02, _sigma([2.0], [0.008, 0.007]), volatility_type)),
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
        return _H(c.reversion, self.t), float(_ir_zeta(c, self.t))

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
@pytest.mark.parametrize("volatility_type", VOLATILITY_TYPES)
@pytest.mark.parametrize("shape", CURVE_SHAPES)
@pytest.mark.parametrize("maturity_after", [0.5, 3.0, 10.0])
def test_zero_bonds_deflated_by_the_numeraire_are_martingales(shape, maturity_after, volatility_type):
    model = _model(shape, volatility_type)
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


@pytest.mark.parametrize("volatility_type", VOLATILITY_TYPES)
@pytest.mark.parametrize("shape", CURVE_SHAPES)
def test_equities_deflated_by_the_numeraire_are_martingales(shape, volatility_type):
    """E[FX(t) S(t) / N(t)] = FX(0) S(0) q(0, t) for a domestic (FX = 1) and a foreign
    equity whose forecast curve is its currency's discount curve."""
    model = _model(shape, volatility_type)
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


@pytest.mark.parametrize("volatility_type", VOLATILITY_TYPES)
def test_the_domestic_state_is_driftless_with_variance_zeta(volatility_type):
    model = _model(volatility_type=volatility_type)
    mean, cov = _propagate(model, GRID)
    assert mean[0] == 0.0
    assert cov[0, 0] == pytest.approx(float(_ir_zeta(model.ir[0], GRID[-1])), rel=1e-14)


@pytest.mark.parametrize("volatility_type", VOLATILITY_TYPES)
def test_every_covariance_block_equals_the_integral_of_the_brownian_loadings(volatility_type):
    """Over one step [t0, t1], each state's stochastic part is the integral of a loading
    vector g(s) against the correlated Brownians: z_i has alpha_i e_i; the FX rate of
    currency i adds (H_0(t1)-H_0(s)) alpha_0 e_0 - (H_i(t1)-H_i(s)) alpha_i e_i to its own
    sigma; an equity in currency i adds (H_i(t1)-H_i(s)) alpha_i e_i. So
    Cov = int g_k(s)' rho g_l(s) ds, integrated here by Simpson's rule on each piece between
    the volatility breakpoints (where the integrand is smooth)."""
    model = _model(volatility_type=volatility_type)
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
    # An IR state's loading is alpha(s): constant on a piece for the Hagan LGM, sigma e^{a s}
    # for the Hull-White model.
    alpha = [constant(c.sigma) * (np.exp(c.reversion * s) if c.volatility_type == "HullWhite" else 1.0)
             for c in model.ir]
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
    usd = IrComponent("USD", _curve(rates), 0.03, sigma)
    for target_rates, handle in ((rates, None), (index_rates, _ore_handle(index_rates))):
        ours = np.asarray(implied_log_discounts(_curve(target_rates), usd, t, taus, z))
        for s in range(z.shape[0]):
            for j in range(t.size):
                for k in range(taus.shape[1]):
                    args = (t[j], t[j] + taus[j, k], float(z[s, j]))
                    ore = lgm.discountBond(*args) if handle is None else lgm.discountBond(*args, handle)
                    assert np.exp(ours[s, j, k]) == pytest.approx(ore, rel=1e-13)
    N = np.asarray(lgm_numeraire(usd, t, z))
    for s in range(z.shape[0]):
        for j in range(t.size):
            assert N[s, j] == pytest.approx(lgm.numeraire(t[j], float(z[s, j])), rel=1e-13)


def _short_rate(hw, component, t, z):
    """r(t, z) = f(0, t) + H'(t) z + zeta(t) H(t) H'(t), the LGM's short rate at state z, with
    f(0, t) read off QuantLib's own term structure so the comparison is of the model, not of
    two numerical forward rates."""
    f = hw.termStructure().forwardRate(t, t, ORE.Continuous, ORE.NoFrequency, True).rate()
    a = component.reversion
    H, Hp = _H(a, t), np.exp(-a * t)
    return f + Hp * z + float(_ir_zeta(component, t)) * H * Hp


@pytest.mark.parametrize("a, sigma", [(0.03, 0.01), (0.1, 0.02), (0.005, 0.008)])
def test_hull_white_path_curves_equal_quantlibs_hull_white(a, sigma):
    """A Hull-White currency's path curve is QuantLib's `HullWhite::discountBond(t, T, r)` at
    the state's short rate, on a sloped curve: the simulated curves are the Hull-White
    model's own bond prices, fitted to today's curve (I-42, I-44)."""
    rates = CURVE_SHAPES["upward"][0]
    hw = ORE.HullWhite(_ore_handle(rates), a, sigma)
    usd = IrComponent("USD", _curve(rates), a, _sigma([], [sigma]), "HullWhite")
    t = np.array([0.4, 1.7, 4.2, 9.0])
    taus = np.array([[0.0, 0.5, 2.0, 12.0]] * t.size)
    z = jnp.asarray([[-0.03, 0.01, 0.05, 0.02], [0.02, -0.04, 0.0, -0.06]])
    ours = np.asarray(implied_log_discounts(_curve(rates), usd, t, taus, z))
    for s in range(z.shape[0]):
        for j in range(t.size):
            r = _short_rate(hw, usd, t[j], float(z[s, j]))
            for k in range(1, taus.shape[1]):
                assert np.exp(ours[s, j, k]) == pytest.approx(hw.discountBond(t[j], t[j] + taus[j, k], r), rel=1e-12)


class TestHullWhiteParametrization:
    """ORE's `Lgm1fPiecewiseConstantHullWhiteAdaptor` with constant reversion: alpha(t) =
    sigma(t) e^{a t} and zeta(t) = int_0^t sigma(s)^2 e^{2 a s} ds, on the host (step moments)
    and on the device (scenario market), piecewise and at a = 0."""
    SIGMA = ([1.0, 3.0], [0.010, 0.012, 0.009])
    T = np.array([0.0, 0.3, 1.0, 2.2, 3.0, 7.5])

    @pytest.mark.parametrize("a", [0.03, -0.02, 0.4])
    def test_zeta_is_the_integral_of_alpha_squared(self, a):
        from scipy.integrate import quad

        component = IrComponent("USD", _curve([0.03] * 6), a, _sigma(*self.SIGMA), "HullWhite")
        for t in self.T:
            pieces = [0.0] + [b for b in self.SIGMA[0] if b < t] + [t]
            expected = sum(quad(lambda s: float(_ir_alpha(component, s)) ** 2, lo, hi, epsabs=0, epsrel=1e-13)[0]
                           for lo, hi in zip(pieces[:-1], pieces[1:]))
            assert float(_ir_zeta(component, t)) == pytest.approx(expected, rel=1e-12, abs=1e-18)
            assert float(component.zeta(jnp.asarray(t))) == pytest.approx(expected, rel=1e-12, abs=1e-18)

    def test_alpha_is_the_short_rate_volatility_over_h_prime(self):
        component = IrComponent("USD", _curve([0.03] * 6), 0.05, _sigma(*self.SIGMA), "HullWhite")
        for t in self.T:
            assert float(_ir_alpha(component, t)) == pytest.approx(
                float(_piecewise(component.sigma, t)) * np.exp(0.05 * t), rel=1e-15)

    def test_at_zero_reversion_it_is_the_hagan_lgm(self):
        sigma = _sigma(*self.SIGMA)
        np.testing.assert_allclose(np.asarray(hull_white_zeta(0.0, sigma, jnp.asarray(self.T))),
                                   np.asarray(hagan_zeta(sigma, jnp.asarray(self.T))), rtol=1e-15, atol=0)

    def test_an_unknown_volatility_type_is_refused(self):
        with pytest.raises(ValueError, match="volatility_type"):
            IrComponent("USD", _curve([0.03] * 6), 0.03, _sigma([], [0.01]), "Black")


def test_path_curves_are_floored_at_ores_minimum_discount_factor():
    usd = IrComponent("USD", _curve([0.03] * 6), 0.03, _sigma([], [0.01]))
    ours = implied_log_discounts(_curve([0.03] * 6), usd, np.array([1.0]), np.array([[0.0, 30.0]]),
                                 jnp.asarray([[100.0]]))
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


@pytest.mark.parametrize("model", [LgmConfig, HullWhiteConfig])
@pytest.mark.parametrize("fmt, se_bound", [("float64", 4.0), ("float32", 4.0)])
def test_simulated_assets_are_martingales_on_sloped_curves(fmt, se_bound, model):
    """The permanent check audit M-1 asked for (I-42): E[P(t,T)/N(t)] = P(0,T) on a 3% -> 5%
    curve (the Hull-White simulation before 2026-10-01, a constant-theta short rate, missed by
    4.2% and 8.8% at t=2y for 5y and 10y bonds), plus the foreign bond and the equity, within
    `se_bound` standard errors, for either IR model."""
    market = _market()
    sm = simulate(market, _config(ir={"USD": model(0.03, 0.01), "EUR": model(0.02, 0.008)}),
                  precision=Precision.throughout(fmt))
    dtype = jnp.dtype(fmt)
    assert sm.numeraire.dtype == dtype and sm.discount["USD"].log_discounts.dtype == dtype
    N = np.asarray(sm.numeraire, dtype=np.float64)

    def check(samples, expected):
        se = samples.std() / np.sqrt(samples.size)
        assert abs(samples.mean() - expected) <= se_bound * se + 1e-6 * abs(expected)

    taus = _value_times(sm)
    for j, t in enumerate(sm.times):
        for k in (2, 6, 12):
            tau = float(taus[j, k])
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
        tau = _value_times(sm)[j]
        basis = (np.asarray(log_discount(idx, t + tau)) - float(log_discount(idx, t))
                 - np.asarray(log_discount(usd, t + tau)) + float(log_discount(usd, t)))
        spread = np.asarray(sm.index["USD-SIMINDEX-6M"].log_discounts[:, j] - sm.discount["USD"].log_discounts[:, j])
        np.testing.assert_allclose(spread, np.broadcast_to(basis, spread.shape), atol=1e-14)


@pytest.mark.parametrize("model", [LgmConfig, HullWhiteConfig])
def test_zero_volatility_paths_are_todays_forward_curves(model):
    sm = simulate(_market(), _config(ir={"USD": model(0.03, 0.0), "EUR": model(0.02, 0.0)}, samples=8))
    usd = _curve(PILLAR_CURVES["USD"])
    for j, t in enumerate(sm.times):
        tau = _value_times(sm)[j]
        expected = np.asarray(log_discount(usd, t + tau)) - float(log_discount(usd, t))
        np.testing.assert_allclose(np.asarray(sm.discount["USD"].log_discounts[:, j]),
                                   np.broadcast_to(expected, (8, tau.size)), atol=1e-15)
        np.testing.assert_allclose(np.asarray(sm.numeraire[:, j]), np.exp(-float(log_discount(usd, t))), rtol=1e-15)


def test_values_are_computed_at_each_dates_tenors_and_held_at_the_as_of_dates():
    """ORE: `nextPath` computes a curve's discount factors at the tenors from the simulation
    date, by period arithmetic (1Y from 2027-03-01 is 366 days), and `ScenarioSimMarket` holds them
    at the tenors from the as-of date (1Y from 2026-07-30 is 365 days) on every date, its curve
    built once with a moving reference date (`addYieldCurve`; I-84, found against ORE's own
    simulation on 2026-10-07)."""
    date = ORE.Date(1, 3, 2027)
    sm = simulate(_market(), _config(dates=(date,), samples=4, ir={"USD": LgmConfig(0.03, 0.0),
                                                                   "EUR": LgmConfig(0.02, 0.0)}))
    one_year = 3  # "1Y" is the third configured tenor
    assert tenor_times([date], _config().curve_tenors)[0, one_year] == 366 / 365
    held = np.asarray(sm.discount["USD"].tenor_times[0])
    np.testing.assert_array_equal(held, as_of_tenor_times(ASOF, _config().curve_tenors))
    assert held[one_year] == 365 / 365
    # At zero volatility the value held at 365 days is today's forward over 366.
    t = float(sm.times[0])
    usd = _curve(PILLAR_CURVES["USD"])
    expected = float(log_discount(usd, t + 366 / 365) - log_discount(usd, t))
    assert float(sm.discount["USD"].log_discounts[0, 0, one_year]) == pytest.approx(expected, rel=1e-14)


def test_a_currency_takes_the_model_of_its_configuration():
    """`CamConfig.ir[ccy]` chooses the model (decision A-1): the same volatility number is the
    LGM's alpha for an `LgmConfig` and the short rate's for a `HullWhiteConfig`."""
    model = build_cross_asset_model(_market(), _config(ir={"USD": HullWhiteConfig(0.03, 0.01),
                                                             "EUR": LgmConfig(0.02, 0.008)}))
    assert [c.volatility_type for c in model.ir] == ["HullWhite", "Hagan"]
    assert float(_ir_alpha(model.ir[0], 2.0)) == pytest.approx(0.01 * np.exp(0.06), rel=1e-15)
    assert float(_ir_alpha(model.ir[1], 2.0)) == pytest.approx(0.008, rel=1e-15)


class TestConfigurationRefusals:
    def test_fx_volatilities_must_cover_every_foreign_currency(self):
        with pytest.raises(ValueError, match="fx_volatilities"):
            _config(fx_volatilities={})

    def test_the_base_currency_needs_a_model(self):
        with pytest.raises(ValueError, match="base_currency"):
            _config(base_currency="GBP")

    def test_a_currency_model_must_be_a_model_configuration(self):
        with pytest.raises(TypeError, match=r"ir\['USD'\] must be an LgmConfig or a HullWhiteConfig"):
            _config(ir={"USD": {"reversion": 0.03}, "EUR": LgmConfig(0.02)})

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
