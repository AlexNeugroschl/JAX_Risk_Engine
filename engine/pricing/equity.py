"""
Cash equities and listed equity options on today's market (TraderX fork, R1 and R2).

    cash equity   units x spot                      (units = quantity x multiplier)
    option        units x V(spot, r, q, sigma)      V by finite differences on Black-Scholes

`option_value` solves Black-Scholes in log spot on a uniform grid of `SPACE_NODES` nodes
spanning +-`WIDTH_STDEVS` standard deviations of ln S at expiry around today's spot (so today's
spot is the centre node and no interpolation is needed), stepped back from expiry in
`TIME_STEPS` steps of Crank-Nicolson after four implicit half steps (Rannacher's start, which
damps the payoff's kink). An American option is projected onto its exercise value after every
step; a European is not. The boundaries carry the option's asymptotes. The constant
coefficients are read off the curves at expiry: `r` the discount curve's zero rate, `q` the
dividend curve's (0 without one), `sigma` the equity's flat Black volatility. The grid is
finer in space than QuantLib's `FdBlackScholesVanillaEngine(400, 400)` (whose mesh concentrates
at the strike): 2e-5 of the analytic European price, 5e-5 of QuantLib's American
(`tests/test_traderx_products.py`).

Everything is computed in float64, as a model input (decision A-10), and the value is rounded
once to the caller's dtype. On the expiry date the option is worth its intrinsic value.
"""
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from engine.market_data.curves import ZeroCurve, log_discount
from engine.market_data.day_counts import TIME_AXIS_DAY_COUNTER

SPACE_NODES = 801
TIME_STEPS = 400
WIDTH_STDEVS = 5.0
RANNACHER_HALF_STEPS = 4


class OptionTerms(NamedTuple):
    """An equity option's data as arrays (`[]` each): shares controlled, strike, expiry time
    (ACT/365 from the as-of date), dividend yield and Black volatility to expiry."""
    units: np.ndarray
    strike: np.ndarray
    expiry: np.ndarray
    dividend: np.ndarray
    volatility: np.ndarray


def option_terms(cfg, market, dtype=np.float64) -> OptionTerms:
    """`OptionTerms` of `cfg` on `market` (its equity's dividend curve and volatility)."""
    equity = market.equities[cfg.symbol]
    expiry = TIME_AXIS_DAY_COUNTER.yearFraction(market.asof, cfg.expiry_date)
    dividend = 0.0
    if equity.dividend_curve is not None and expiry > 0:
        dividend = -float(log_discount(ZeroCurve.from_config(equity.dividend_curve), jnp.asarray(expiry))) / expiry
    as_array = lambda v: np.asarray(v, dtype=dtype)  # noqa: E731
    return OptionTerms(as_array(cfg.units), as_array(cfg.strike), as_array(expiry), as_array(dividend),
                       as_array(equity.volatility))


def payoff(spot, strike, call: bool):
    return jnp.maximum(spot - strike, 0.0) if call else jnp.maximum(strike - spot, 0.0)


@partial(jax.jit, static_argnames=("call", "american"))
def option_value(spot, rate, terms: OptionTerms, call: bool, american: bool):
    """One share's value (see the module docstring), float64; `spot` and `rate` scalars. One
    program per kind of option, whatever its terms."""
    f64 = jnp.float64
    spot, rate = jnp.asarray(spot, f64), jnp.asarray(rate, f64)
    strike, expiry = jnp.asarray(terms.strike, f64), jnp.asarray(terms.expiry, f64)
    q, sigma = jnp.asarray(terms.dividend, f64), jnp.asarray(terms.volatility, f64)
    width = WIDTH_STDEVS * sigma * jnp.sqrt(expiry)
    x = jnp.log(spot) + jnp.linspace(-width, width, SPACE_NODES, dtype=f64)
    h = 2.0 * width / (SPACE_NODES - 1)
    s = jnp.exp(x)
    exercise = payoff(s, strike, call)
    a, mu = 0.5 * sigma ** 2, rate - q - 0.5 * sigma ** 2
    lower, centre, upper = a / h ** 2 - mu / (2 * h), -2 * a / h ** 2 - rate, a / h ** 2 + mu / (2 * h)

    def boundary(tau):
        """The values at the grid's two ends `tau` before expiry: deep in or out of the money."""
        forward = s[jnp.array([0, -1])] * jnp.exp(-q * tau) - strike * jnp.exp(-rate * tau)
        value = jnp.maximum(forward, 0.0) if call else jnp.maximum(-forward, 0.0)
        return jnp.maximum(value, exercise[jnp.array([0, -1])]) if american else value

    def apply(v, ends):
        """L v on the interior nodes, the ends' values given."""
        full = jnp.concatenate([ends[:1], v, ends[1:]])
        return lower * full[:-2] + centre * full[1:-1] + upper * full[2:]

    def step(carry, item):
        v, tau = carry
        theta, dt = item
        ends_old, ends_new = boundary(tau), boundary(tau + dt)
        rhs = v + (1 - theta) * dt * apply(v, ends_old)
        rhs = rhs.at[0].add(theta * dt * lower * ends_new[0]).at[-1].add(theta * dt * upper * ends_new[1])
        m = SPACE_NODES - 2
        dl = jnp.full((m,), -theta * dt * lower, f64).at[0].set(0.0)
        du = jnp.full((m,), -theta * dt * upper, f64).at[-1].set(0.0)
        d = jnp.full((m,), 1.0 - theta * dt * centre, f64)
        v = jax.lax.linalg.tridiagonal_solve(dl, d, du, rhs[:, None])[:, 0]
        if american:
            v = jnp.maximum(v, exercise[1:-1])
        return (v, tau + dt), None

    dt = expiry / TIME_STEPS
    thetas = np.concatenate([np.ones(RANNACHER_HALF_STEPS), np.full(TIME_STEPS - RANNACHER_HALF_STEPS // 2, 0.5)])
    steps = jnp.concatenate([jnp.full((RANNACHER_HALF_STEPS,), dt / 2, f64),
                             jnp.full((TIME_STEPS - RANNACHER_HALF_STEPS // 2,), dt, f64)])
    (v, _), _ = jax.lax.scan(step, (exercise[1:-1], jnp.zeros((), f64)), (jnp.asarray(thetas, f64), steps))
    return v[SPACE_NODES // 2 - 1]


@dataclass(frozen=True)
class OptionPrice:
    """`(terms, curves, quotes) -> NPV` of an equity option: `curves` its equity's log spot
    `[1]` and its currency's discount curve. Equal for equal flags, so one compiled program
    serves every option of a kind. `intrinsic` is an option on its expiry date."""
    call: bool
    american: bool
    intrinsic: bool = False

    def value(self, log_spot, rate, terms: OptionTerms):
        spot = jnp.exp(jnp.asarray(log_spot, jnp.float64))
        if self.intrinsic:
            return payoff(spot, jnp.asarray(terms.strike, jnp.float64), self.call)
        return option_value(spot, rate, terms, self.call, self.american)

    def __call__(self, terms: OptionTerms, curves, _quotes):
        log_spot, disc = curves
        rate = -log_discount(disc, jnp.asarray(terms.expiry, disc.pillar_rates.dtype)) / jnp.maximum(terms.expiry, 1e-12)
        return (terms.units * self.value(log_spot[0], rate, terms)).astype(log_spot.dtype)
