"""
ORE's cross-asset model (CAM): one LGM interest-rate component per currency, Black-Scholes FX
and equity components, constant instantaneous correlations, simulated under the domestic LGM
measure with the exact discretization.

    ORE: `QuantExt::CrossAssetModel` (QuantExt/qle/models/crossassetmodel.hpp),
         `CrossAssetStateProcess::ExactDiscretization` (qle/processes/crossassetstateprocess.cpp),
         `CrossAssetAnalytics` (qle/models/crossassetanalytics.cpp).

State, in ORE's order (`pIdx`): the LGM states z_0..z_{n-1} (z_0 domestic), the log FX rates
x_1..x_{n-1} (domestic units per unit of currency i), then the log equity spots s_k. One
Brownian per state; `correlation` is over the same order.

Exact discretization (ORE's default `Discretization::Exact`): given the state at t0, the
state at t0 + dt is Gaussian with mean `transition @ x + drift` and covariance `covariance`.
The transition carries ORE's state-dependent drift parts (`ir/fx/eq_expectation_2`: identity
plus the H-increment couplings of FX and EQ to their rates); `drift` is the state-independent
part (`*_expectation_1`); the covariance is ORE's `covarianceImpl`. They depend only on the
time grid and the parameters, so they are computed once on the host in float64 and the path
recursion `x_{i+1} = M_i x_i + b_i + L_i Z_i` runs on device. `L_i` is the Cholesky factor of
the step covariance: ORE's `pseudoSqrt` with the CAM's default `SalvagingAlgorithm::None`
is `CholeskyDecomposition(cov, flexible = true)`, reproduced by `flexible_cholesky`.

An IR component is an LGM with constant reversion and one of ORE's two volatility
parametrizations (`LgmData::VolatilityType`): `Hagan` (the LGM's own volatility alpha,
`IrLgm1fPiecewiseConstantParametrization`) or `HullWhite` (the short rate's volatility,
`IrLgm1fPiecewiseConstantHullWhiteAdaptor`: the Hull-White model with its curve-fitted drift,
alpha(t) = sigma(t) exp(a t); see `engine.models.lgm`). Only alpha and zeta differ; every
formula below reads them through `_Analytics.az`/`zetaz`.

Scope, refused rather than approximated: the LGM measure only (ORE's default; `BA` is not
implemented), LGM components with constant reversion, and Black-Scholes FX/EQ with
piecewise-constant volatility. Integrals are exact up to rounding (Gauss-Legendre on each
interval between parameter breakpoints, where every integrand is a product of constants and
exponentials); ORE integrates the same functions with a `SimpsonIntegral(1e-8, 100)`, so the
two agree to ORE's integration tolerance.
"""
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from engine.models.curves import ZeroCurve, log_discount
from engine.models.lgm import VOLATILITY_TYPES, Sigma, as_sigma, hull_white_zeta, zeta as hagan_zeta

#: Gauss-Legendre nodes per interval between breakpoints. The integrands are smooth there
#: (products of constants and exponentials), and 20 nodes are exact to double precision for
#: any interval a simulation grid produces.
_GAUSS_LEGENDRE_NODES = 20
_GL_X, _GL_W = np.polynomial.legendre.leggauss(_GAUSS_LEGENDRE_NODES)


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IrComponent:
    """One currency's LGM with constant reversion: `sigma` is the LGM's alpha with
    `volatility_type="Hagan"` (`IrLgm1fPiecewiseConstantParametrization`), the short rate's
    volatility with `"HullWhite"` (the Hull-White adaptor; see the module docstring). `curve`
    is the model's term structure P(0, t), the currency's discount curve."""
    currency: str
    curve: ZeroCurve
    reversion: float
    sigma: Sigma
    volatility_type: str = "Hagan"

    def __post_init__(self):
        if self.volatility_type not in VOLATILITY_TYPES:
            raise ValueError(f"volatility_type must be one of {VOLATILITY_TYPES}; got {self.volatility_type!r}")

    def zeta(self, t) -> jax.Array:
        """zeta(t) on the device (in `t`'s dtype), for the scenario market."""
        if self.volatility_type == "HullWhite":
            return hull_white_zeta(self.reversion, self.sigma, t)
        return hagan_zeta(self.sigma, t)


@dataclass(frozen=True)
class FxComponent:
    """FX rate of `currency` against the domestic currency, in domestic units per unit of
    `currency` (`FxBsPiecewiseConstantParametrization`). The simulated state is its log."""
    currency: str
    spot: float
    sigma: Sigma


@dataclass(frozen=True)
class EqComponent:
    """An equity in `currency` (`EqBsParametrization`): spot, volatility, and the two
    curves ORE's drift reads, `equityIrCurveToday` (`forecast_curve`) and
    `equityDivYieldCurveToday` (`dividend_curve`)."""
    name: str
    currency: str
    spot: float
    sigma: Sigma
    forecast_curve: ZeroCurve
    dividend_curve: ZeroCurve


@dataclass(frozen=True)
class CrossAssetModel:
    """The CAM's components and correlation. `ir[0]` is the domestic (base) currency, and
    `fx[i]` is the currency of `ir[i + 1]`, as ORE requires."""
    ir: Tuple[IrComponent, ...]
    fx: Tuple[FxComponent, ...]
    eq: Tuple[EqComponent, ...]
    correlation: np.ndarray

    def __post_init__(self):
        if not self.ir:
            raise ValueError("a cross-asset model needs at least one IR component")
        currencies = [c.currency for c in self.ir]
        if len(set(currencies)) != len(currencies):
            raise ValueError(f"IR components must have distinct currencies; got {currencies}")
        if [c.currency for c in self.fx] != currencies[1:]:
            raise ValueError(
                f"FX components must be the non-domestic IR currencies, in the same order "
                f"({currencies[1:]}); got {[c.currency for c in self.fx]}")
        for eq in self.eq:
            if eq.currency not in currencies:
                raise ValueError(f"equity {eq.name!r} is in {eq.currency}, which has no IR component")
        corr = np.asarray(self.correlation, dtype=np.float64)
        d = self.dimension
        if corr.shape != (d, d):
            raise ValueError(f"correlation must be {d}x{d} (factors {self.factor_names}); got {corr.shape}")
        if not np.allclose(corr, corr.T, rtol=0.0, atol=1e-12):
            raise ValueError("correlation must be symmetric")
        if not np.allclose(np.diag(corr), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("correlation must have a unit diagonal")
        if np.linalg.eigvalsh(corr)[0] < -1e-12:
            raise ValueError(f"correlation must be positive semi-definite (factors {self.factor_names})")
        object.__setattr__(self, "correlation", corr)

    @property
    def dimension(self) -> int:
        return len(self.ir) + len(self.fx) + len(self.eq)

    @property
    def domestic_currency(self) -> str:
        return self.ir[0].currency

    @property
    def factor_names(self) -> Tuple[str, ...]:
        """ORE's factor keys, in state order: `IR:<ccy>`, `FX:<ccy><domestic>`, `EQ:<name>`."""
        dom = self.domestic_currency
        return (tuple(f"IR:{c.currency}" for c in self.ir)
                + tuple(f"FX:{c.currency}{dom}" for c in self.fx)
                + tuple(f"EQ:{c.name}" for c in self.eq))

    def ir_index(self, currency: str) -> int:
        return [c.currency for c in self.ir].index(currency)

    def fx_state(self, j: int) -> int:
        return len(self.ir) + j

    def eq_state(self, k: int) -> int:
        return len(self.ir) + len(self.fx) + k

    def initial_state(self) -> np.ndarray:
        """ORE's `initialValues`: z = 0, ln FX spot, ln equity spot."""
        return np.concatenate([
            np.zeros(len(self.ir)),
            np.log([c.spot for c in self.fx]) if self.fx else np.zeros(0),
            np.log([c.spot for c in self.eq]) if self.eq else np.zeros(0),
        ])


# ---------------------------------------------------------------------------
# Model functions on the host (float64)
# ---------------------------------------------------------------------------
def _piecewise(sigma, t: np.ndarray) -> np.ndarray:
    """A piecewise-constant function at `t`, ORE's `PiecewiseConstantHelper` lookup
    (`upper_bound`, clamped to the last value)."""
    s = as_sigma(sigma)
    times, values = np.asarray(s.times, dtype=np.float64), np.asarray(s.values, dtype=np.float64)
    return values[np.minimum(np.searchsorted(times, t, side="right"), values.size - 1)]


def _integral_of_square(sigma, t: np.ndarray, reversion: float = 0.0) -> np.ndarray:
    """int_0^t sigma(s)^2 exp(2 reversion s) ds: with reversion 0 the Hagan LGM's zeta or an
    FX/EQ variance; otherwise the Hull-White adaptor's zeta (`engine.models.lgm.
    hull_white_zeta`, on the host)."""
    s = as_sigma(sigma)
    times, values = np.asarray(s.times, dtype=np.float64), np.asarray(s.values, dtype=np.float64)
    t = np.maximum(np.asarray(t, dtype=np.float64), 0.0)
    edges = np.concatenate([[0.0], times, [np.inf]])
    lo, hi = edges[:-1], edges[1:]
    covered = np.clip(t[..., None] - lo, 0.0, hi - lo)
    if reversion == 0.0:
        return covered @ (values ** 2)
    # integral_lo^(lo + covered) exp(2 a s) ds, with expm1 for accuracy at small 2 a covered.
    rate = 2.0 * reversion
    return (np.exp(rate * lo) * np.expm1(rate * covered) / rate) @ (values ** 2)


def _ir_alpha(component: "IrComponent", t: np.ndarray) -> np.ndarray:
    """The LGM's alpha(t): sigma(t) (Hagan), or sigma(t) exp(a t) (Hull-White: ORE's
    `hullWhiteSigma(t) / Hprime(t)`)."""
    alpha = _piecewise(component.sigma, t)
    if component.volatility_type == "HullWhite":
        alpha = alpha * np.exp(component.reversion * np.asarray(t, dtype=np.float64))
    return alpha


def _ir_zeta(component: "IrComponent", t: np.ndarray) -> np.ndarray:
    """The LGM's zeta(t) = int_0^t alpha(s)^2 ds on the host."""
    reversion = component.reversion if component.volatility_type == "HullWhite" else 0.0
    return _integral_of_square(component.sigma, t, reversion)


def _H(reversion: float, t: np.ndarray) -> np.ndarray:
    """LGM H(t) = (1 - exp(-a t)) / a (t when a = 0), as `engine.models.lgm.H`."""
    t = np.asarray(t, dtype=np.float64)
    if reversion == 0.0:
        return t
    return -np.expm1(-reversion * t) / reversion


def _breakpoints(model: CrossAssetModel) -> np.ndarray:
    """Every parameter breakpoint; integrals split there (ORE's `PiecewiseIntegral`)."""
    sigmas = ([c.sigma for c in model.ir] + [c.sigma for c in model.fx] + [c.sigma for c in model.eq])
    times = [np.asarray(as_sigma(s).times, dtype=np.float64) for s in sigmas]
    return np.unique(np.concatenate(times)) if times else np.zeros(0)


class _Analytics:
    """ORE's `CrossAssetAnalytics` building blocks for one model: `az` (IR alpha), `Hz`,
    `zetaz`, `sx`/`vx` (FX sigma/variance), `ss`/`vs` (EQ), and `integral`."""

    def __init__(self, model: CrossAssetModel):
        self.model = model
        self.breaks = _breakpoints(model)
        self.rho = model.correlation

    def integral(self, f: Callable[[np.ndarray], np.ndarray], t0: float, t1: float) -> float:
        """int_t0^t1 f(t) dt, split at the parameter breakpoints."""
        inner = self.breaks[(self.breaks > t0) & (self.breaks < t1)]
        edges = np.concatenate([[t0], inner, [t1]])
        total = 0.0
        for a, b in zip(edges[:-1], edges[1:]):
            half = 0.5 * (b - a)
            nodes = a + half * (_GL_X + 1.0)
            total += half * float(np.dot(_GL_W, f(nodes)))
        return total

    def az(self, i, t):
        return _ir_alpha(self.model.ir[i], t)

    def Hz(self, i, t):
        return _H(self.model.ir[i].reversion, t)

    def zetaz(self, i, t):
        return _ir_zeta(self.model.ir[i], t)

    def sx(self, j, t):
        return _piecewise(self.model.fx[j].sigma, t)

    def vx(self, j, t):
        return _integral_of_square(self.model.fx[j].sigma, t)

    def ss(self, k, t):
        return _piecewise(self.model.eq[k].sigma, t)

    def vs(self, k, t):
        return _integral_of_square(self.model.eq[k].sigma, t)

    # correlations by component index
    def rzz(self, i, j):
        return self.rho[i, j]

    def rzx(self, i, j):
        return self.rho[i, self.model.fx_state(j)]

    def rxx(self, i, j):
        return self.rho[self.model.fx_state(i), self.model.fx_state(j)]

    def rzs(self, i, k):
        return self.rho[i, self.model.eq_state(k)]

    def rxs(self, j, k):
        return self.rho[self.model.fx_state(j), self.model.eq_state(k)]

    def rss(self, k, l):
        return self.rho[self.model.eq_state(k), self.model.eq_state(l)]


def _log_discount(curve: ZeroCurve, t: float) -> float:
    return float(log_discount(curve, jnp.asarray(t, dtype=jnp.float64)))


# ---------------------------------------------------------------------------
# Step moments (ORE's CrossAssetAnalytics, LGM measure)
# ---------------------------------------------------------------------------
def _ir_expectation_1(x: _Analytics, i: int, t0: float, dt: float) -> float:
    """`ir_expectation_1`: 0 for the domestic currency; for a foreign one the drift from
    the change of measure to the domestic LGM measure."""
    if i == 0:
        return 0.0
    I, t1 = x.integral, t0 + dt
    return (-I(lambda t: x.Hz(i, t) * x.az(i, t) ** 2, t0, t1)
            - x.rzx(i, i - 1) * I(lambda t: x.az(i, t) * x.sx(i - 1, t), t0, t1)
            + x.rzz(0, i) * I(lambda t: x.Hz(0, t) * x.az(0, t) * x.az(i, t), t0, t1))


def _fx_expectation_1(x: _Analytics, j: int, t0: float, dt: float) -> float:
    """`fx_expectation_1` (LGM measure) for the FX rate of IR component i = j + 1."""
    i, I, t1 = j + 1, x.integral, t0 + dt
    m = x.model
    H0a, H0b, Hia, Hib = x.Hz(0, t0), x.Hz(0, t1), x.Hz(i, t0), x.Hz(i, t1)
    z0a, z0b, zia, zib = x.zetaz(0, t0), x.zetaz(0, t1), x.zetaz(i, t0), x.zetaz(i, t1)
    res = (_log_discount(m.ir[i].curve, t1) - _log_discount(m.ir[i].curve, t0)
           + _log_discount(m.ir[0].curve, t0) - _log_discount(m.ir[0].curve, t1))
    res -= 0.5 * (x.vx(j, t1) - x.vx(j, t0))
    res += 0.5 * (H0b ** 2 * z0b - H0a ** 2 * z0a - I(lambda t: x.Hz(0, t) ** 2 * x.az(0, t) ** 2, t0, t1))
    res -= 0.5 * (Hib ** 2 * zib - Hia ** 2 * zia - I(lambda t: x.Hz(i, t) ** 2 * x.az(i, t) ** 2, t0, t1))
    res += x.rzx(0, j) * I(lambda t: x.Hz(0, t) * x.az(0, t) * x.sx(j, t), t0, t1)
    res -= Hib * (-I(lambda t: x.Hz(i, t) * x.az(i, t) ** 2, t0, t1)
                  + x.rzz(0, i) * I(lambda t: x.Hz(0, t) * x.az(0, t) * x.az(i, t), t0, t1)
                  - x.rzx(i, j) * I(lambda t: x.az(i, t) * x.sx(j, t), t0, t1))
    res += (-I(lambda t: x.Hz(i, t) ** 2 * x.az(i, t) ** 2, t0, t1)
            + x.rzz(0, i) * I(lambda t: x.Hz(0, t) * x.Hz(i, t) * x.az(0, t) * x.az(i, t), t0, t1)
            - x.rzx(i, j) * I(lambda t: x.Hz(i, t) * x.az(i, t) * x.sx(j, t), t0, t1))
    return res


def _eq_expectation_1(x: _Analytics, k: int, t0: float, dt: float) -> float:
    """`eq_expectation_1` (LGM measure) for equity k in the currency of IR component i."""
    m = x.model
    eq, I, t1 = m.eq[k], x.integral, t0 + dt
    i = m.ir_index(eq.currency)
    Hia, Hib, zia, zib = x.Hz(i, t0), x.Hz(i, t1), x.zetaz(i, t0), x.zetaz(i, t1)
    res = (_log_discount(eq.dividend_curve, t1) - _log_discount(eq.dividend_curve, t0)
           + _log_discount(eq.forecast_curve, t0) - _log_discount(eq.forecast_curve, t1))
    res -= 0.5 * (x.vs(k, t1) - x.vs(k, t0))
    res += 0.5 * (Hib ** 2 * zib - Hia ** 2 * zia - I(lambda t: x.Hz(i, t) ** 2 * x.az(i, t) ** 2, t0, t1))
    res += x.rzs(0, k) * I(lambda t: x.Hz(0, t) * x.az(0, t) * x.ss(k, t), t0, t1)
    if i > 0:
        res -= x.rxs(i - 1, k) * I(lambda t: x.sx(i - 1, t) * x.ss(k, t), t0, t1)
        res += Hib * (-I(lambda t: x.Hz(i, t) * x.az(i, t) ** 2, t0, t1)
                      - x.rzx(i, i - 1) * I(lambda t: x.sx(i - 1, t) * x.az(i, t), t0, t1)
                      + x.rzz(0, i) * I(lambda t: x.az(i, t) * x.az(0, t) * x.Hz(0, t), t0, t1))
        res -= (-I(lambda t: x.Hz(i, t) ** 2 * x.az(i, t) ** 2, t0, t1)
                - x.rzx(i, i - 1) * I(lambda t: x.Hz(i, t) * x.sx(i - 1, t) * x.az(i, t), t0, t1)
                + x.rzz(0, i) * I(lambda t: x.Hz(i, t) * x.az(i, t) * x.az(0, t) * x.Hz(0, t), t0, t1))
    return res


def _transition(x: _Analytics, t0: float, dt: float) -> np.ndarray:
    """The state-dependent part of the conditional mean, as a matrix M with
    `E[X(t0+dt) | X(t0)] = M X(t0) + drift` (`ir/fx/eq_expectation_2`)."""
    m, t1 = x.model, t0 + dt
    M = np.eye(m.dimension)
    dH = [float(x.Hz(i, t1) - x.Hz(i, t0)) for i in range(len(m.ir))]
    for j in range(len(m.fx)):
        row = m.fx_state(j)
        M[row, 0] += dH[0]
        M[row, j + 1] -= dH[j + 1]
    for k, eq in enumerate(m.eq):
        i = m.ir_index(eq.currency)
        M[m.eq_state(k), i] += dH[i]
    return M


def _covariance(x: _Analytics, t0: float, dt: float) -> np.ndarray:
    """`ExactDiscretization::covarianceImpl`: IR-IR, IR-FX, FX-FX, EQ-EQ, IR-EQ, FX-EQ."""
    m, I, t1 = x.model, x.integral, t0 + dt
    n, nfx, neq = len(m.ir), len(m.fx), len(m.eq)
    C = np.zeros((m.dimension, m.dimension))

    def put(r, c, v):
        C[r, c] = C[c, r] = v

    H0b = float(x.Hz(0, t1))
    for i in range(n):
        for j in range(i, n):
            put(i, j, x.rzz(i, j) * I(lambda t: x.az(i, t) * x.az(j, t), t0, t1))
    for i in range(n):          # ir_fx_covariance
        for j in range(nfx):
            Hj1 = float(x.Hz(j + 1, t1))
            put(i, m.fx_state(j), I(lambda t: (
                (H0b - x.Hz(0, t)) * x.az(0, t) * x.az(i, t) * x.rzz(0, i)
                - (Hj1 - x.Hz(j + 1, t)) * x.az(j + 1, t) * x.az(i, t) * x.rzz(j + 1, i)
                + x.az(i, t) * x.sx(j, t) * x.rzx(i, j)), t0, t1))
    for a in range(nfx):        # fx_fx_covariance
        for b in range(a, nfx):
            Ha1, Hb1 = float(x.Hz(a + 1, t1)), float(x.Hz(b + 1, t1))

            def integrand(t, a=a, b=b, Ha1=Ha1, Hb1=Hb1):
                a0, aa, ab = x.az(0, t), x.az(a + 1, t), x.az(b + 1, t)
                g0, ga, gb = H0b - x.Hz(0, t), Ha1 - x.Hz(a + 1, t), Hb1 - x.Hz(b + 1, t)
                sa, sb = x.sx(a, t), x.sx(b, t)
                return (g0 * g0 * a0 * a0
                        - g0 * gb * a0 * ab * x.rzz(0, b + 1)
                        - g0 * ga * a0 * aa * x.rzz(0, a + 1)
                        + g0 * a0 * sb * x.rzx(0, b)
                        + g0 * a0 * sa * x.rzx(0, a)
                        - ga * aa * sb * x.rzx(a + 1, b)
                        - gb * ab * sa * x.rzx(b + 1, a)
                        + ga * gb * aa * ab * x.rzz(a + 1, b + 1)
                        + sa * sb * x.rxx(a, b))
            put(m.fx_state(a), m.fx_state(b), I(integrand, t0, t1))
    for k in range(neq):        # eq_eq_covariance
        for l in range(k, neq):
            i, j = m.ir_index(m.eq[k].currency), m.ir_index(m.eq[l].currency)
            Hib, Hjb = float(x.Hz(i, t1)), float(x.Hz(j, t1))
            put(m.eq_state(k), m.eq_state(l), I(lambda t: (
                x.rss(k, l) * x.ss(k, t) * x.ss(l, t)
                + (Hjb - x.Hz(j, t)) * x.rzs(j, k) * x.az(j, t) * x.ss(k, t)
                + (Hib - x.Hz(i, t)) * x.rzs(i, l) * x.az(i, t) * x.ss(l, t)
                + (Hib - x.Hz(i, t)) * (Hjb - x.Hz(j, t)) * x.rzz(i, j) * x.az(i, t) * x.az(j, t)), t0, t1))
    for j in range(n):          # ir_eq_covariance
        for k in range(neq):
            i = m.ir_index(m.eq[k].currency)
            Hib = float(x.Hz(i, t1))
            put(j, m.eq_state(k), I(lambda t: (
                (Hib - x.Hz(i, t)) * x.rzz(i, j) * x.az(i, t) * x.az(j, t)
                + x.rzs(j, k) * x.az(j, t) * x.ss(k, t)), t0, t1))
    for j in range(nfx):        # fx_eq_covariance
        for k in range(neq):
            i, jl = m.ir_index(m.eq[k].currency), j + 1
            Hib, Hjb = float(x.Hz(i, t1)), float(x.Hz(jl, t1))
            put(m.fx_state(j), m.eq_state(k), I(lambda t: (
                (Hib - x.Hz(i, t)) * (H0b - x.Hz(0, t)) * x.rzz(0, i) * x.az(0, t) * x.az(i, t)
                - (Hib - x.Hz(i, t)) * (Hjb - x.Hz(jl, t)) * x.rzz(jl, i) * x.az(jl, t) * x.az(i, t)
                + (Hib - x.Hz(i, t)) * x.rzx(i, j) * x.sx(j, t) * x.az(i, t)
                + (H0b - x.Hz(0, t)) * x.rzs(0, k) * x.az(0, t) * x.ss(k, t)
                - (Hjb - x.Hz(jl, t)) * x.rzs(jl, k) * x.az(jl, t) * x.ss(k, t)
                + x.rxs(j, k) * x.sx(j, t) * x.ss(k, t)), t0, t1))
    return C


def flexible_cholesky(S: np.ndarray) -> np.ndarray:
    """QuantLib's `CholeskyDecomposition(S, flexible = true)`
    (ql/math/matrixutilities/choleskydecomposition.cpp): the lower-triangular factor, with a
    zero column where a pivot is not positive (a degenerate, e.g. zero-volatility,
    component) instead of failing."""
    n = S.shape[0]
    L = np.zeros_like(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i, j] - np.dot(L[i, :i], L[j, :i])
            if i == j:
                L[i, i] = np.sqrt(s) if s > 0.0 else 0.0
            else:
                L[j, i] = 0.0 if L[i, i] == 0.0 else s / L[i, i]
    return L


@dataclass(frozen=True)
class StepMoments:
    """Per step i (from `times[i]` to `times[i+1]`, with `times[0] = 0`), the exact
    conditional law `X_{i+1} | X_i ~ N(transition[i] @ X_i + drift[i], covariance[i])`, and
    `cholesky[i]` with `cholesky[i] @ cholesky[i].T == covariance[i]`."""
    transition: np.ndarray  # [T, d, d]
    drift: np.ndarray       # [T, d]
    covariance: np.ndarray  # [T, d, d]
    cholesky: np.ndarray    # [T, d, d]


def step_moments(model: CrossAssetModel, times: Sequence[float]) -> StepMoments:
    """Exact step moments on the grid `0 = times[0] < times[1] < ...` (float64, host)."""
    times = np.asarray(times, dtype=np.float64)
    if times[0] != 0.0 or np.any(np.diff(times) <= 0.0):
        raise ValueError(f"times must start at 0 and increase strictly; got {times.tolist()}")
    x = _Analytics(model)
    transitions, drifts, covariances = [], [], []
    for t0, t1 in zip(times[:-1], times[1:]):
        dt = t1 - t0
        drift = np.zeros(model.dimension)
        for i in range(len(model.ir)):
            drift[i] = _ir_expectation_1(x, i, t0, dt)
        for j in range(len(model.fx)):
            drift[model.fx_state(j)] = _fx_expectation_1(x, j, t0, dt)
        for k in range(len(model.eq)):
            drift[model.eq_state(k)] = _eq_expectation_1(x, k, t0, dt)
        transitions.append(_transition(x, t0, dt))
        drifts.append(drift)
        covariances.append(_covariance(x, t0, dt))
    covariances = np.asarray(covariances)
    return StepMoments(
        transition=np.asarray(transitions), drift=np.asarray(drifts), covariance=covariances,
        cholesky=np.asarray([flexible_cholesky(c) for c in covariances]),
    )


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
@jax.jit
def evolve_states(x0: jax.Array, moments: Tuple[jax.Array, jax.Array, jax.Array], normals: jax.Array) -> jax.Array:
    """States `[S, T, d]` at `times[1:]` from independent standard normals `[T, S, d]`:
    `x_{i+1} = M_i x_i + b_i + L_i z_i` (ORE's `StochasticProcess::evolve` with the exact
    discretization). `moments` is `(transition, drift, cholesky)` in the normals' dtype."""
    transition, drift, cholesky = moments

    def step(x, inputs):
        M, b, L, z = inputs
        x_next = x @ M.T + b + z @ L.T
        return x_next, x_next

    initial = jnp.broadcast_to(x0, (normals.shape[1], x0.shape[0]))
    _, states = jax.lax.scan(step, initial, (transition, drift, cholesky, normals))
    return jnp.transpose(states, (1, 0, 2))
