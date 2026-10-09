"""
Black's formula on zero-coupon bonds, as `QuantLib::HullWhite::discountBondOption` evaluates
it once the bond's volatility is known: shared by the Jamshidian European engine
(`engine.valuation.jamshidian`) and the standalone calibration's LGM swaption pricer
(`engine.calibration.basket`). The curve primitives are re-exported here for existing callers.

The Hull-White model itself is the LGM with the Hull-White volatility parametrization
(`engine.models.lgm`: `hull_white_zeta`), simulated by the cross-asset model
(`engine.simulation.config.HullWhiteConfig`). Its closed forms in short-rate form (A(t,T),
B(t,T)) belonged to the Hull-White simulation retired on 2026-10-01.
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


def bond_call(P_t_Topt: jax.Array, P_t_S: jax.Array, K: jax.Array, sigma_p: jax.Array) -> jax.Array:
    """Black call on a zero bond, as `HullWhite::discountBondOption`: the forward P_t_S, the
    strike K P_t_Topt, the standard deviation `sigma_p`.

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
