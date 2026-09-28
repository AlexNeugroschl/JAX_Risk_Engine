"""
One-factor Linear Gauss-Markov (LGM) model, in ORE's state-variable parametrization.

This is the model behind ORE's `NumericLgmMultiLegOptionEngine` and the model
`bermudan_swaption.py` rolls back on. Formulas follow `QuantExt::Lgm1fParametrization` and
`LinearGaussMarkovModel` (QuantExt/qle/models/lgm.hpp):

    H(t)      = (1 - exp(-a*t)) / a
    zeta(t)   = integral_0^t sigma(s)^2 ds
    P(t,T,x)  = [P(0,T)/P(0,t)] * exp(-0.5*(H(T)^2-H(t)^2)*zeta(t) - (H(T)-H(t))*x)
    N(t,x)    = exp(0.5*H(t)^2*zeta(t) + H(t)*x) / P(0,t)       (numeraire)
    r(t,x)    = f(0,t) + x*H'(t) + zeta(t)*H'(t)*H(t)

The state x(t) is driftless (`IrLgm1fStateProcess::expectation` returns its input), with
variance zeta(t).

Not the same model as `engine.models.hull_white` for the same `(a, sigma)`. Here sigma is
the volatility of x, so the equivalent Hull-White short-rate volatility is
`sigma * exp(-a*t)`; `QuantLib::HullWhite` uses a constant `sigma`. The two agree at t=0 and
diverge after. Bermudan/American pricing uses only this module, as ORE does.

`Sigma` is piecewise constant (`Lgm1fPiecewiseConstantParametrization`); a flat sigma is
the one-bucket case. Only zeta depends on sigma; H depends only on the reversion `a`.

`bond_option_sigma` is the LGM zero-bond option volatility. It is fed to the model-free
Black-on-bond formula in `hull_white.bond_call`/`bond_put`, which `engine.calibration`
uses to price co-terminal Europeans as `AnalyticLgmSwaptionEngine` does.
"""
from dataclasses import dataclass
from typing import Union

import jax
import jax.numpy as jnp

from engine.models.hull_white import ZeroCurve, discount, forward_rate


@jax.tree_util.register_pytree_node_class
@dataclass
class Sigma:
    """Piecewise-constant volatility: `sigma(s) = values[i]` on bucket `i`.

    Buckets are `[0, times[0]), [times[0], times[1]), ..., [times[-1], inf)`, as ORE's
    `alphaTimes`/`alpha`, so `len(values) == len(times) + 1`. Empty `times` means flat.

    Registered as a pytree with both fields as children, so tangents propagate when a
    `Sigma` is nested inside a larger argument (e.g. `(a, sigma)` in
    `engine.calibration.basket`). As an unregistered leaf it would silently get a zero
    tangent.
    """
    times: jax.Array    # [B-1], strictly increasing, B = number of buckets
    values: jax.Array   # [B]

    @staticmethod
    def flat(sigma: float, dtype=jnp.float64) -> "Sigma":
        # Reshape rather than `jnp.asarray([sigma])`: a Python list around a tracer
        # fails under jit/grad ("No constant handler for type: DynamicJaxprTracer").
        return Sigma(
            times=jnp.zeros((0,), dtype=dtype),
            values=jnp.reshape(jnp.asarray(sigma, dtype=dtype), (1,)),
        )

    def tree_flatten(self):
        return (self.times, self.values), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        times, values = children
        return cls(times=times, values=values)


def as_sigma(sigma: Union[float, jax.Array, "Sigma"]) -> Sigma:
    """Wrap a scalar sigma as a one-bucket `Sigma`; pass a `Sigma` through."""
    if isinstance(sigma, Sigma):
        return sigma
    return Sigma.flat(sigma)


def zeta(sigma: Union[float, jax.Array, "Sigma"], t: jax.Array) -> jax.Array:
    """
    zeta(t) = integral_0^t sigma(s)^2 ds, the variance of x(t).

    `sigma^2 * t` when flat; otherwise the sum of whole buckets plus the partial bucket
    containing `t`, as `QuantExt::PiecewiseConstantHelper1::int_y_sqr`. Bucket lookup
    uses `searchsorted(side="right")`, matching ORE's `std::upper_bound`.
    """
    s = as_sigma(sigma)
    t = jnp.maximum(t, 0.0)
    if s.times.shape[0] == 0:
        return (s.values[0] ** 2) * t

    boundaries = jnp.concatenate([jnp.zeros((1,), dtype=s.times.dtype), s.times])  # [B], left edge of each bucket
    full_durations = s.times - boundaries[:-1]              # [B-1], duration of every bucket EXCEPT the last (open-ended) one
    full_contrib = s.values[:-1] ** 2 * full_durations       # [B-1]
    cum_before = jnp.concatenate([jnp.zeros((1,), dtype=full_contrib.dtype), jnp.cumsum(full_contrib)])  # [B]

    i = jnp.clip(jnp.searchsorted(s.times, t, side="right"), 0, s.values.shape[0] - 1)
    bucket_start = jnp.where(i == 0, 0.0, boundaries[jnp.clip(i, 1, boundaries.shape[0] - 1)])
    partial = s.values[i] ** 2 * jnp.maximum(t - bucket_start, 0.0)
    return cum_before[i] + partial


def H(a: float, t: jax.Array) -> jax.Array:
    """H(t) = (1-exp(-a*t))/a, with the a -> 0 limit H(t) = t."""
    a_safe = jnp.where(a == 0.0, 1.0, a)
    return jnp.where(t > 0.0, jnp.where(a == 0.0, t, (1.0 - jnp.exp(-a_safe * t)) / a_safe), 0.0)


def H_prime(a: float, t: jax.Array) -> jax.Array:
    """H'(t) = exp(-a*t)."""
    return jnp.exp(-a * t)


def bond_price(curve: ZeroCurve, a: float, sigma: Union[float, Sigma], t: jax.Array, T: jax.Array, x: jax.Array) -> jax.Array:
    """
    Zero-bond price P(t,T) at LGM state x:

        P(t,T,x) = [P(0,T)/P(0,t)] * exp(-0.5*(H(T)^2-H(t)^2)*zeta(t) - (H(T)-H(t))*x)

    Matches `ORE.LinearGaussMarkovModel.discountBond` to ~1e-16 relative.
    """
    P0T = discount(curve, T)
    P0t = jnp.where(t > 0.0, discount(curve, jnp.maximum(t, 1e-12)), 1.0)
    Ht, HT = H(a, t), H(a, T)
    return (P0T / P0t) * jnp.exp(-0.5 * (HT ** 2 - Ht ** 2) * zeta(sigma, t)) * jnp.exp(-(HT - Ht) * x)


def numeraire(curve: ZeroCurve, a: float, sigma: Union[float, Sigma], t: jax.Array, x: jax.Array) -> jax.Array:
    """
    N(t,x) = exp(0.5*H(t)^2*zeta(t) + H(t)*x) / P(0,t), as
    `LinearGaussMarkovModel::numeraire`. The backward induction deflates by this, so the
    rolled-back quantity is a martingale under x's driftless law.
    """
    P0t = jnp.where(t > 0.0, discount(curve, jnp.maximum(t, 1e-12)), 1.0)
    Ht = H(a, t)
    return jnp.where(t > 0.0, jnp.exp(0.5 * Ht ** 2 * zeta(sigma, t) + Ht * x) / P0t, jnp.ones_like(x))


def r_from_x(curve: ZeroCurve, a: float, sigma: Union[float, Sigma], t: jax.Array, x: jax.Array) -> jax.Array:
    """
    Short rate at LGM state x: r(t,x) = f(0,t) + x*H'(t) + zeta(t)*H'(t)*H(t),
    i.e. -d/dT log P(t,T,x) at T=t.
    """
    f0t = forward_rate(curve, t)
    Hp = H_prime(a, t)
    return f0t + x * Hp + zeta(sigma, t) * Hp * H(a, t)


def x_from_r(curve: ZeroCurve, a: float, sigma: Union[float, Sigma], t: jax.Array, r: jax.Array) -> jax.Array:
    """Inverse of `r_from_x` (affine in x). Returns 0 at t <= 0."""
    f0t = forward_rate(curve, t)
    Hp = H_prime(a, t)
    x = (r - f0t - zeta(sigma, t) * Hp * H(a, t)) / Hp
    return jnp.where(t > 0.0, x, jnp.zeros_like(r))


def bond_option_sigma(a: float, sigma: Union[float, Sigma], T_opt: jax.Array, S: jax.Array, t: jax.Array) -> jax.Array:
    """
    Volatility, seen from `t`, of the zero bond P(T_opt, S, x):

        |H(S) - H(T_opt)| * sqrt(zeta(T_opt) - zeta(t))

    Read off `bond_price`'s exponent, since ln P is affine in the Gaussian state x.
    """
    H_diff = H(a, S) - H(a, T_opt)
    variance = jnp.clip(zeta(sigma, T_opt) - zeta(sigma, t), 0.0, None)
    return jnp.abs(H_diff) * jnp.sqrt(variance)
