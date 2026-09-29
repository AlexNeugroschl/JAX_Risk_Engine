"""
Hull-White one-factor closed forms (`QuantLib::HullWhite`).

    B(t,T)   = (1 - exp(-a*(T-t))) / a
    A(t,T)   = [P(0,T)/P(0,t)] * exp(B(t,T)*f(0,t) - (sigma^2/4a)*(1-exp(-2at))*B(t,T)^2)
    P(t,T,r) = A(t,T) * exp(-B(t,T)*r)

f(0,t) is today's instantaneous forward, and P(0,t) comes from the zero curve
(`engine.models.curves`, re-exported here for existing callers). Checked against
`ORE.HullWhite` in the test suite.

Used by the swap pricer, the European (Jamshidian) swaption pricer and the simulation's
yield-curve cube. Bermudans and Americans use `engine.models.lgm` instead, as ORE does. The
two models differ for t>0 with the same `(a, sigma)` (see that module); do not mix them.
"""
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from engine.models.curves import (  # noqa: F401  (re-exports)
    ZeroCurve,
    discount,
    forward_rate,
    log_discount,
    zero_rate,
)


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

    Requires `a > 0`, as QuantLib's `HullWhite` does; `prepare_swaption` refuses anything
    else (I-41). Unlike `B` and `A` there is no `a -> 0` limit here."""
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
