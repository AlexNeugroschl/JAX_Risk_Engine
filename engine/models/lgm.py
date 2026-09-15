"""
Linear Gauss-Markov (LGM) model primitives, in ORE's own state-variable
parametrization -- the model `bermudan_swaption.py`'s numeric backward
induction is built on.

**A different model from `engine.models.hull_white`, not an alternate
parametrization of the same one.** `ORE.HullWhite` (`QuantLib::HullWhite`,
the model `engine.models.hull_white` implements) and
`ORE.LinearGaussMarkovModel` (`QuantExt::CrossAssetModel`'s own rates leg,
the model this module implements) were assumed, going into the original
build of this codebase's Bermudan engine, to be two equivalent
parametrizations of the same underlying short-rate model. A live,
directwcomparison against the installed ORE package showed they are
**not** the same numerical model realization for t>0, despite being
constructed with identical `(a, sigma)` and sharing the same t=0 curve:

    >>> hw.discountBond(t=3, T=5, r=0.03)       # r = f(0,t), HullWhite's own "no shock" point
    0.9393234598794674
    >>> lgm.discountBond(t=3, T=5, x=0.0)       # x = 0, LGM's own "no shock" point
    0.9337296209777532

a genuine ~0.6% difference at t=3y -- not a rounding artifact. Both classes
are individually self-consistent affine short-rate models (each satisfies
its own `-d/dT log P(t,T)|_{T=t} == r` identity exactly, live-verified via
finite difference), so this is a real parametrization/calibration
difference between the two ORE classes, not a bug in either formula. Since
ORE's actual Bermudan/American engine
(`QuantExt::NumericLgmMultiLegOptionEngine`) is built on
`LinearGaussMarkovModel`, `bermudan_swaption.py` uses ONLY this module's
formulas -- never `engine.models.hull_white` -- for every discount factor
its backward induction computes. See docs/reference/ore-parity.md's "A
parametrization note: LGM vs. plain Hull-White" for the full investigation
and every individual building block's independent verification.

Formulas (`QuantExt::Lgm1fParametrization`/`LinearGaussMarkovModel`,
`QuantExt/qle/models/lgm.hpp` lines 227-280, live-verified to machine
precision against `ORE.LinearGaussMarkovModel.discountBond` throughout
this codebase's test suite):

    H(t)      = (1 - exp(-a*t)) / a          (== hull_white.B(0, t, a))
    zeta(t)   = integral_0^t sigma(s)^2 ds    (sigma^2 * t for constant sigma)
    P(t,T,x)  = [P(0,T)/P(0,t)] * exp(-0.5*(H(T)^2-H(t)^2)*zeta(t)) * exp(-(H(T)-H(t))*x)
    N(t,x)    = exp(0.5*H(t)^2*zeta(t) + H(t)*x) / P(0,t)     (the model's numeraire)
    r(t,x)    = f(0,t) + x*H'(t) + zeta(t)*H'(t)*H(t)          (LGM's own short rate at state x)

`x(t)` is the model's own state variable: driftless under its own
transition law (confirmed directly from
`QuantExt::IrLgm1fStateProcess::expectation()`, which returns its input
unchanged -- no drift term at all), which is what makes Hagan's quadrature
convolution (`bermudan_swaption.py`'s backward-induction rollback) valid.

**Piecewise-constant sigma(t), not just a flat scalar.** ORE's own
calibration produces a genuine term structure -- `QuantExt::
Lgm1fPiecewiseConstantParametrization`, `QuantExt/qle/models/
irlgm1fpiecewiseconstantparametrization.hpp` -- not a single number: sigma
is held constant between consecutive "alpha times" (in practice, the
calibration basket's own swaption expiry times -- see
`engine/calibration/`) and zeta(t) is the accumulated piecewise integral,
not `sigma^2*t`. Crucially, **H(t) is completely unaffected** -- confirmed
directly from ORE's own source: `Lgm1fPiecewiseConstantParametrization::H`
delegates to a helper built purely from the (constant, uncalibrated -- see
`engine/calibration/`'s own docstring on why reversion stays fixed)
reversion parameter, never referencing alpha/sigma at all. Only `zeta`'s
own computation changes; the bond price/numeraire/short-rate formulas
above keep their exact same structure regardless of which `zeta` feeds
them. `Sigma` below represents both the flat and piecewise cases through
one uniform type -- a flat scalar is simply a `Sigma` with zero interior
breakpoints (one "bucket" spanning `[0, inf)`) -- so every formula in this
module only ever needs to call `zeta(sigma, t)`, never branch on which
case it's handling.

**Zero-coupon bond options under LGM** (`bond_call`/`bond_put` below):
since `ln P(T_opt,S,x)` is affine in the Gaussian state `x`, its variance
as seen from `t` is `(H(S)-H(T_opt))^2 * (zeta(T_opt)-zeta(t))` -- read
directly off `bond_price`'s own exponent. This is the LGM analogue of
`engine.models.hull_white.bond_option_sigma`; the Black-on-bond payoff
formula itself (`hull_white.bond_call`/`bond_put`) is model-agnostic (it
only consumes `P_t_Topt`, `P_t_S`, `K`, `sigma_p`), so it is reused as-is
here rather than reimplemented -- only the volatility term differs between
the two models. `engine/calibration/` builds a co-terminal European
swaption price from these primitives (LGM's own analogue of
`QuantExt::AnalyticLgmSwaptionEngine`) as the calibration target function.
"""
from dataclasses import dataclass
from typing import Union

import jax
import jax.numpy as jnp

from engine.models.hull_white import ZeroCurve, discount, forward_rate


@jax.tree_util.register_pytree_node_class
@dataclass
class Sigma:
    """A piecewise-constant Hull-White/LGM volatility term structure:
    `sigma(s) = values[i]` for `s` in bucket `i`, where the buckets are
    `[0, times[0])`, `[times[0], times[1])`, ..., `[times[-1], inf)` --
    exactly ORE's own `alphaTimes`/`alpha` pair
    (`Lgm1fPiecewiseConstantParametrization`'s constructor). `len(values)
    == len(times) + 1` always (one more bucket than interior breakpoints).

    `times` may be empty, in which case `values` has exactly one entry --
    a flat sigma for all `t >= 0` -- which is what `Sigma.flat(sigma)`
    builds, and what every existing caller passing a plain `float` for
    `hw_sigma` is automatically upgraded to (see `as_sigma` below) so
    every model formula has exactly one code path regardless of which
    case a caller is in.

    **Registered as a JAX pytree** (`@register_pytree_node_class`), with
    both `times` and `values` as children (not static/auxiliary data) --
    required for `jax.jvp`/`jax.custom_jvp`/`jax.grad` to differentiate
    correctly whenever a `Sigma` is passed as part of a LARGER pytree
    argument (e.g. `engine.calibration.basket._bisect_xstar`'s `params =
    (a, sigma)` tuple) rather than having its own `.values` array unpacked
    and passed directly -- an UNregistered dataclass is treated by
    `jax.tree_util` as an opaque leaf, silently preventing any tangent
    from propagating into its fields. Confirmed as a real, not merely
    theoretical, gap: `_bisect_xstar`'s implicit-function-theorem
    correction produced a systematic ~6% wrong Vega before this
    registration was added (`jax.jvp` was differentiating `price_lgm_
    swaption` with respect to `sigma` as an opaque object, contributing
    zero tangent through `Sigma.values`, silently understating the
    correction term) -- see `engine/calibration/basket.py`'s own
    docstring for the full incident. `times` is registered as a child too
    (not auxiliary/static data), even though no caller in this codebase
    currently differentiates with respect to bucket BREAKPOINTS (only
    `values`, mirroring `ZeroCurve.pillar_times` never being a Greek's
    target either) -- registering it as a child is the conservative
    choice (a differentiable field mistakenly marked static would silently
    drop its own gradient; a non-differentiable field marked as a child
    costs nothing, since its cotangent is simply unused).
    """
    times: jax.Array    # [B-1], strictly increasing, B = number of buckets
    values: jax.Array   # [B]

    @staticmethod
    def flat(sigma: float, dtype=jnp.float64) -> "Sigma":
        # `jnp.reshape(jnp.asarray(sigma), (1,))` rather than
        # `jnp.asarray([sigma])`: wrapping a value in a Python LIST forces JAX
        # to treat that list as a constant to be materialized, which fails
        # outright for a traced value ("No constant handler for type:
        # DynamicJaxprTracer") whenever this is reached from inside a
        # jax.jit/jax.grad trace with a scalar sigma -- e.g. differentiating
        # `price_lgm_swaption` with respect to a plain float sigma. Converting
        # the scalar first and then reshaping keeps a tracer a tracer, and is
        # exactly equivalent for a concrete input.
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
    """Upgrades a plain scalar (the common case for every pricer/test
    written before piecewise-sigma support existed) into a one-bucket
    `Sigma` -- the normalization point every function in this module
    calls before touching `sigma`, so callers never need to construct a
    `Sigma` by hand just to keep using a flat volatility."""
    if isinstance(sigma, Sigma):
        return sigma
    return Sigma.flat(sigma)


def zeta(sigma: Union[float, jax.Array, "Sigma"], t: jax.Array) -> jax.Array:
    """
    Accumulated variance of the driftless LGM state x(t):
    `zeta(t) = integral_0^t sigma(s)^2 ds` -- for a flat scalar sigma,
    this is `sigma^2 * t`
    (`QuantExt::IrLgm1fConstantParametrization::zeta`); for a piecewise
    `Sigma`, this is the running sum of each FULLY-elapsed bucket's own
    `value^2 * duration` plus one partial-bucket term for whichever bucket
    `t` currently falls in -- exactly
    `QuantExt::PiecewiseConstantHelper1::int_y_sqr`
    (`QuantExt/qle/models/piecewiseconstanthelper.hpp`), live-verified
    against `ORE.IrLgm1fPiecewiseConstantParametrization.zeta` directly
    (`tests/test_models_piecewise_sigma.py`).

    Vectorized (no Python loop over buckets) via a cumulative sum over
    full-bucket contributions, then a `jnp.searchsorted` to find which
    bucket `t` itself falls in (matching `std::upper_bound`'s semantics
    exactly -- ORE's own bucket-lookup rule) and adding that bucket's own
    partial contribution.
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
    """H(t) = (1-exp(-a*t))/a -- QuantExt::Lgm1fParametrization's H(t),
    identical in shape to `engine.models.hull_white.B` with its first
    argument fixed at 0 (a live-verified identity, NOT evidence the two
    models coincide for t>0 -- see this module's own docstring)."""
    a_safe = jnp.where(a == 0.0, 1.0, a)
    return jnp.where(t > 0.0, jnp.where(a == 0.0, t, (1.0 - jnp.exp(-a_safe * t)) / a_safe), 0.0)


def H_prime(a: float, t: jax.Array) -> jax.Array:
    """H'(t) = exp(-a*t)."""
    return jnp.exp(-a * t)


def bond_price(curve: ZeroCurve, a: float, sigma: Union[float, Sigma], t: jax.Array, T: jax.Array, x: jax.Array) -> jax.Array:
    """
    P(t,T,x) directly in LGM's own state variable, with no detour through
    `engine.models.hull_white`'s r(t)-parametrized formulas (see module
    docstring for why the two are not interchangeable):

        P(t,T,x) = [P(0,T)/P(0,t)] * exp(-0.5*(H(T)^2-H(t)^2)*zeta(t))
                                    * exp(-(H(T)-H(t))*x)

    Live-verified to machine precision (~1e-16 relative) against
    `ORE.LinearGaussMarkovModel.discountBond(t,T,x)`.
    """
    P0T = discount(curve, T)
    P0t = jnp.where(t > 0.0, discount(curve, jnp.maximum(t, 1e-12)), 1.0)
    Ht, HT = H(a, t), H(a, T)
    return (P0T / P0t) * jnp.exp(-0.5 * (HT ** 2 - Ht ** 2) * zeta(sigma, t)) * jnp.exp(-(HT - Ht) * x)


def numeraire(curve: ZeroCurve, a: float, sigma: Union[float, Sigma], t: jax.Array, x: jax.Array) -> jax.Array:
    """
    N(t,x) = exp(0.5*H(t)^2*zeta(t) + H(t)*x) / P(0,t) --
    QuantExt::LinearGaussMarkovModel::numeraire. Every payoff in
    `bermudan_swaption.py`'s backward induction is divided by this before
    rolling back, so the rolled-back quantity is a true Q-martingale under
    x(t)'s own driftless transition law (a plain, non-deflated rollback of
    P(t,T) under x's own transition does NOT reproduce P(t_to,T) --
    verified directly while originally building the Bermudan engine).
    """
    P0t = jnp.where(t > 0.0, discount(curve, jnp.maximum(t, 1e-12)), 1.0)
    Ht = H(a, t)
    return jnp.where(t > 0.0, jnp.exp(0.5 * Ht ** 2 * zeta(sigma, t) + Ht * x) / P0t, jnp.ones_like(x))


def r_from_x(curve: ZeroCurve, a: float, sigma: Union[float, Sigma], t: jax.Array, x: jax.Array) -> jax.Array:
    """
    The short rate implied by LGM's own state x(t) and bond formula (NOT
    `engine.models.hull_white`'s short rate -- see module docstring):

        r(t,x) = f(0,t) + x*H'(t) + zeta(t)*H'(t)*H(t)

    Live-verified via central finite difference directly on `bond_price`
    (`-d/dT log P(t,T,x) |_{T=t}`, matching to ~1e-12 relative) -- LGM's
    own genuine short rate at state x, used only to convert a simulated
    `hw_paths` short rate (from `engine.simulation`, calibrated with the
    SAME (a, sigma) this module receives) into the corresponding LGM state
    x for conditioning (see `x_from_r`, its exact inverse).
    """
    f0t = forward_rate(curve, t)
    Hp = H_prime(a, t)
    return f0t + x * Hp + zeta(sigma, t) * Hp * H(a, t)


def x_from_r(curve: ZeroCurve, a: float, sigma: Union[float, Sigma], t: jax.Array, r: jax.Array) -> jax.Array:
    """Exact inverse of `r_from_x` (r(t,x) is affine/linear in x, so this
    is a closed-form solve, not a numerical root-find): converts a short
    rate (e.g. a simulated `hw_paths` value) into the corresponding LGM
    state x at the same time t, for interpolating into an x-indexed value
    function."""
    f0t = forward_rate(curve, t)
    Hp = H_prime(a, t)
    x = (r - f0t - zeta(sigma, t) * Hp * H(a, t)) / Hp
    return jnp.where(t > 0.0, x, jnp.zeros_like(r))


def bond_option_sigma(a: float, sigma: Union[float, Sigma], T_opt: jax.Array, S: jax.Array, t: jax.Array) -> jax.Array:
    """
    sigma_p: the volatility (as seen from `t`) of the LGM zero-coupon bond
    price `P(T_opt,S,x)` -- the LGM analogue of
    `engine.models.hull_white.bond_option_sigma`. Read directly off
    `bond_price`'s own exponent: `ln P(T_opt,S,x)` is affine in the
    Gaussian state `x(T_opt)`, with coefficient `-(H(S)-H(T_opt))`, so its
    variance conditional on information at `t` is
    `(H(S)-H(T_opt))^2 * Var[x(T_opt)|x(t)] = (H(S)-H(T_opt))^2 *
    (zeta(T_opt)-zeta(t))` (x is driftless, so its variance simply
    accumulates zeta -- see module docstring).
    """
    H_diff = H(a, S) - H(a, T_opt)
    variance = jnp.clip(zeta(sigma, T_opt) - zeta(sigma, t), 0.0, None)
    return jnp.abs(H_diff) * jnp.sqrt(variance)
