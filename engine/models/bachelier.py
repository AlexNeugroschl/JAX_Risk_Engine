"""
The normal (Bachelier) option formula, QuantLib's `bachelierBlackFormula` without the discount:
shared by the European engine (`engine.pricing.european`) and the market value of an LGM
calibration helper (`engine.calibration.lgm`), so both read a zero volatility as QuantLib
does, as the option's intrinsic value.
"""
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm


def bachelier(forward, strike, std_dev, call) -> jax.Array:
    """w (F - K) N(w d) + s n(d), d = (F - K) / s, w = 1 for a call and -1 for a put; the
    intrinsic value max(w (F - K), 0) when the standard deviation `std_dev` is 0. `call` is a
    bool or a boolean array broadcasting with the others."""
    w = jnp.where(call, 1.0, -1.0)
    safe = jnp.where(std_dev > 0.0, std_dev, 1.0)
    d = (forward - strike) / safe
    value = w * (forward - strike) * norm.cdf(w * d) + safe * norm.pdf(d)
    return jnp.where(std_dev > 0.0, value, jnp.maximum(w * (forward - strike), 0.0))
