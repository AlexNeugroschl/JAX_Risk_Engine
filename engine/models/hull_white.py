"""
Hull-White one-factor closed forms (`QuantLib::HullWhite`), plus the zero curve type every
pricer uses.

    B(t,T)   = (1 - exp(-a*(T-t))) / a
    A(t,T)   = [P(0,T)/P(0,t)] * exp(B(t,T)*f(0,t) - (sigma^2/4a)*(1-exp(-2at))*B(t,T)^2)
    P(t,T,r) = A(t,T) * exp(-B(t,T)*r)

f(0,t) is today's instantaneous forward, and P(0,t) comes from the zero curve. Checked
against `ORE.HullWhite` in the test suite.

Used by the swap pricer, the European (Jamshidian) swaption pricer and the simulation's
yield-curve cube. Bermudans and Americans use `engine.models.lgm` instead, as ORE does. The
two models differ for t>0 with the same `(a, sigma)` (see that module); do not mix them.
"""
from dataclasses import dataclass
from typing import List

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm


@jax.tree_util.register_pytree_node_class
@dataclass
class ZeroCurve:
    """Today's zero curve as JAX arrays: continuously compounded zero rates at pillar
    times (year fractions).

    Greeks differentiate with respect to `pillar_rates`, never `pillar_times`, which
    matches ORE's sensitivity framework bumping rates, not times. Registered as a pytree
    so it can pass through `jax.jit`/`jax.grad` boundaries.
    """
    pillar_times: jax.Array
    pillar_rates: jax.Array

    def tree_flatten(self):
        return (self.pillar_times, self.pillar_rates), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        pillar_times, pillar_rates = children
        return cls(pillar_times=pillar_times, pillar_rates=pillar_rates)

    @staticmethod
    def from_config(config, dtype=jnp.float64) -> "ZeroCurve":
        return ZeroCurve(
            pillar_times=jnp.asarray(config.times, dtype=dtype),
            pillar_rates=jnp.asarray(config.rates, dtype=dtype),
        )

    @staticmethod
    def flat(rate: float, pillar_times: List[float], dtype=jnp.float64) -> "ZeroCurve":
        """A curve with the same rate at every pillar."""
        n = len(pillar_times)
        return ZeroCurve(
            pillar_times=jnp.asarray(pillar_times, dtype=dtype),
            pillar_rates=jnp.asarray([rate] * n, dtype=dtype),
        )


def zero_rate(curve: ZeroCurve, t: jax.Array) -> jax.Array:
    """Zero rate at `t`: linear between pillars, flat beyond both ends (`jnp.interp`).

    Differs from ORE past the last pillar: QuantLib's `InterpolatedZeroCurve` extrapolates a
    flat instantaneous forward there, not a flat zero rate. They agree wherever the pillars
    extend past every cashflow (I-48).

    Differentiating through linear interpolation with respect to a pillar gives the
    triangular bump shape ORE's `ShiftScenarioGenerator` applies, so `jax.grad` yields
    ORE-style per-pillar sensitivities directly."""
    return jnp.interp(t, curve.pillar_times, curve.pillar_rates)


def log_discount(curve: ZeroCurve, t: jax.Array) -> jax.Array:
    """Continuously-compounded log discount factor ln P(0,t)."""
    return -zero_rate(curve, t) * t


def discount(curve: ZeroCurve, t: jax.Array) -> jax.Array:
    """P(0,t) = exp(-zero_rate(t) * t)."""
    return jnp.exp(log_discount(curve, t))


def forward_rate(curve: ZeroCurve, t: jax.Array) -> jax.Array:
    """f(0,t) = -d/dt ln P(0,t) = z(t) + t*z'(t), computed exactly from the linear
    interpolant.

    `z'(t)` is the slope of the segment to the right of `t` (a pillar takes the slope of
    the segment it starts) and 0 in the flat extrapolation regions. An exact slope is used
    rather than a finite difference because a finite difference on ln P is cancellation
    noise in float32 (forward errors of up to ~2 percentage points).
    """
    times, rates = curve.pillar_times, curve.pillar_rates
    num_pillars = times.shape[0]
    right = jnp.clip(jnp.searchsorted(times, t, side="right"), 1, num_pillars - 1)
    left = right - 1
    slope = (rates[right] - rates[left]) / (times[right] - times[left])
    inside = (t >= times[0]) & (t < times[-1])
    slope = jnp.where(inside, slope, jnp.zeros_like(slope))
    return zero_rate(curve, t) + t * slope


def B(t: jax.Array, T: jax.Array, a: float) -> jax.Array:
    """B(t,T) = (1 - exp(-a*(T-t))) / a, with the a -> 0 limit B = T - t.

    Both branches are evaluated (as `jnp.where` requires under jit/grad); the division
    uses a placeholder `a` so the unused branch never produces NaN."""
    a_safe = jnp.where(a == 0.0, 1.0, a)
    return jnp.where(a == 0.0, T - t, (1.0 - jnp.exp(-a_safe * (T - t))) / a_safe)


def A(curve: ZeroCurve, t: jax.Array, T: jax.Array, a: float, sigma: float, B_override: jax.Array = None) -> jax.Array:
    """
    A(t,T) = [P(0,T)/P(0,t)] * exp(B(t,T)*f(0,t) - (sigma^2/4a)*(1-exp(-2at))*B(t,T)^2)

    As `QuantLib::HullWhite::A`. At t=0 it reproduces the curve's P(0,T).
    `t`, `T` are broadcastable arrays of year fractions.

    `B_override` replaces B(t,T) in the exponent only. `engine.simulation` uses it to pass
    its own B matrix, which is clamped at 0 for pillars already in the past, so that A and
    the `exp(-B*r)` factor applied later use the same B. Pricers never pass it.
    """
    log_P0_t = log_discount(curve, t)
    log_P0_T = log_discount(curve, T)
    fwd_0_t = forward_rate(curve, t)
    ratio = jnp.exp(log_P0_T - log_P0_t)
    B_t_T = B(t, T, a) if B_override is None else B_override

    # a -> 0 limit of (sigma^2/4a)*(1-exp(-2at)) is sigma^2*t/2.
    a_safe = jnp.where(a == 0.0, 1.0, a)
    variance_term = jnp.where(
        a == 0.0,
        0.5 * sigma ** 2 * t,
        (sigma ** 2 / (4.0 * a_safe)) * (1.0 - jnp.exp(-2.0 * a_safe * t)),
    )
    return ratio * jnp.exp(B_t_T * fwd_0_t - variance_term * B_t_T ** 2)


def bond_price(curve: ZeroCurve, t: jax.Array, T: jax.Array, r: jax.Array, a: float, sigma: float) -> jax.Array:
    """P(t,T) = A(t,T) * exp(-B(t,T)*r), given the short rate `r` at `t`."""
    return A(curve, t, T, a, sigma) * jnp.exp(-B(t, T, a) * r)


def bond_option_sigma(T_opt: jax.Array, S: jax.Array, t: jax.Array, a: float, sigma: float) -> jax.Array:
    """Volatility, seen from t, of the zero bond P(T_opt, S): the standard HW1F
    bond-option volatility (Brigo-Mercurio 3.41), as `ORE.HullWhite.discountBondOption`.

    Known issue (I-41): unlike `B` and `A` there is no a = 0 guard, so a = 0 gives NaN,
    which `bond_call` then treats as zero volatility (intrinsic value). ORE raises."""
    B_Topt_S = B(T_opt, S, a)
    return sigma * B_Topt_S * jnp.sqrt(jnp.clip(1.0 - jnp.exp(-2.0 * a * (T_opt - t)), 0.0, None) / (2.0 * a))


def bond_call(P_t_Topt: jax.Array, P_t_S: jax.Array, K: jax.Array, sigma_p: jax.Array) -> jax.Array:
    """Black call on a zero bond, as `HullWhite::discountBondOption`.

    When `sigma_p == 0` (bond maturing at expiry, or pricing at expiry) this returns the
    intrinsic value max(P_S - K*P_T, 0). The Black branch is evaluated on a placeholder
    sigma so it cannot produce NaN in the gradient."""
    sigma_p_safe = jnp.where(sigma_p > 0.0, sigma_p, 1.0)
    h = (1.0 / sigma_p_safe) * jnp.log(P_t_S / (P_t_Topt * K)) + sigma_p_safe / 2.0
    black = P_t_S * norm.cdf(h) - K * P_t_Topt * norm.cdf(h - sigma_p_safe)
    intrinsic = jnp.maximum(P_t_S - K * P_t_Topt, 0.0)
    return jnp.where(sigma_p > 0.0, black, intrinsic)


def bond_put(P_t_Topt: jax.Array, P_t_S: jax.Array, K: jax.Array, sigma_p: jax.Array) -> jax.Array:
    """Put-call parity on `bond_call` above."""
    call = bond_call(P_t_Topt, P_t_S, K, sigma_p)
    return call - (P_t_S - K * P_t_Topt)
