"""
Hull-White 1-Factor (HW1F) closed-form model primitives: the zero curve
representation, the affine bond-price formula `A(t,T)*exp(-B(t,T)*r)`, and
the analytic bond-option (Black-on-bond) formula Jamshidian's decomposition
is built from.

**The single source of truth for this codebase's Hull-White math.**
Before this module existed, the SAME closed forms were implemented four
separate times: `engine/simulation/market_model.py::compute_hw_A_matrix` (NumPy,
pillar-grid), `engine/instruments/european_swaption.py::compute_hw_A`/
`_hw_B` (NumPy A, JAX B, arbitrary (t,T) pairs), and
`engine/risk/greeks.py::_compute_hw_A_jax` (a JAX transliteration of the
NumPy version, built only because `np.interp` isn't traceable and Delta/
Gamma need `jax.grad` through the curve). All four were the same
mathematics with different call shapes and different NumPy/JAX splits, and
three of the four docstrings said so explicitly ("identical formula to
simulation.compute_hw_A_matrix", "JAX-differentiable reimplementation of
european_swaption.compute_hw_A -- the SAME closed form"). This module
replaces all four: one JAX-native (via `jnp.interp`, which IS traceable)
implementation, used everywhere -- by `engine.simulation` for the pillar-
grid yield-curve cube, by `engine.instruments.swap`/`european_swaption`
for arbitrary-(t,T) bond/bond-option pricing, and by `engine.risk.greeks`
for autodiff Delta/Gamma/Vega -- with no NumPy/JAX fork and no duplicated
formula anywhere.

Formulas (Brigo-Mercurio, and live-verified against `ORE.HullWhite`/
`QuantLib::HullWhite` throughout this codebase's test suite, not assumed
from a textbook):

    B(t,T)   = (1 - exp(-a*(T-t))) / a
    A(t,T)   = [P(0,T)/P(0,t)] * exp(B(t,T)*f(0,t) - (sigma^2/4a)*(1-exp(-2at))*B(t,T)^2)
    P(t,T,r) = A(t,T) * exp(-B(t,T)*r)

where f(0,t) = -d/dt ln P(0,t) is today's instantaneous forward rate, and
P(0,t) is read off the caller's zero curve.

**Not the same model as `engine.models.lgm`.** `ORE.HullWhite` and
`ORE.LinearGaussMarkovModel`, despite sharing (a, sigma) and today's curve,
are live-verified (see `engine/models/lgm.py`'s own docstring) to NOT be
the same numerical model realization for t>0 -- a genuine ORE
parametrization difference, not a bug in either formula. This module
implements the `HullWhite`-parametrized family exclusively (used by
`swap.py`'s discounting, `european_swaption.py`'s Jamshidian decomposition,
and `simulation.py`'s yield-curve cube); `engine.models.lgm` implements the
`LinearGaussMarkovModel`-parametrized family exclusively (used by
`bermudan_swaption.py`). Do not mix the two.
"""
from dataclasses import dataclass
from typing import List

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm


@jax.tree_util.register_pytree_node_class
@dataclass
class ZeroCurve:
    """Today's market zero curve, as JAX arrays -- differentiable
    end-to-end via `jnp.interp` (unlike `engine.simulation.ZeroCurveConfig`,
    whose `times`/`rates` are plain Python lists, fine for one-time config
    but not traceable by `jax.grad`). `pillar_times` is curve structure
    (never itself a Greek's differentiation target -- ORE's own sensitivity
    framework bumps a pillar's RATE, never its time); `pillar_rates` is
    what Delta/Gamma/Vega differentiate with respect to.

    `from_config` builds one from a plain `engine.simulation.ZeroCurveConfig`
    (or any object with `.times`/`.rates`) for the common case of starting
    from a non-differentiable config and only later needing gradients.

    **Registered as a JAX pytree** (`@register_pytree_node_class`), exactly
    as `engine.models.lgm.Sigma` is and for the same reason: both fields are
    JAX arrays, so a `ZeroCurve` can be passed straight through a
    `jax.jit`/`jax.grad` boundary as an ordinary traced argument (JAX
    flattens it to its two arrays and rebuilds it on the other side).
    Without this, any jitted function taking a `ZeroCurve` fails with
    "Error interpreting argument ... as an abstract array". This does not
    change how the class is constructed or used anywhere -- it only teaches
    JAX how to look inside it.
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
        """A flat curve (the same rate at every pillar) -- the common
        case for a demo/test base curve."""
        n = len(pillar_times)
        return ZeroCurve(
            pillar_times=jnp.asarray(pillar_times, dtype=dtype),
            pillar_rates=jnp.asarray([rate] * n, dtype=dtype),
        )


def zero_rate(curve: ZeroCurve, t: jax.Array) -> jax.Array:
    """Linear interpolation on zero rates, flat-extrapolated at the curve
    ends -- `jnp.interp`'s own behavior, the standard "bootstrap zero
    curve" interpolation convention used throughout this codebase (matches
    `np.interp`'s identical behavior in every prior NumPy implementation
    this module replaces). This piecewise-linear shape is also exactly
    ORE's own `ShiftScenarioGenerator::applyShift` triangular sensitivity-
    bump shape (see `engine/risk/greeks.py`'s module docstring) --
    differentiating through this interpolation via `jax.grad` reproduces
    ORE's own per-pillar bump sensitivity with no separate bump-shape code
    needed anywhere."""
    return jnp.interp(t, curve.pillar_times, curve.pillar_rates)


def log_discount(curve: ZeroCurve, t: jax.Array) -> jax.Array:
    """Continuously-compounded log discount factor ln P(0,t)."""
    return -zero_rate(curve, t) * t


def discount(curve: ZeroCurve, t: jax.Array) -> jax.Array:
    """P(0,t) = exp(-zero_rate(t) * t)."""
    return jnp.exp(log_discount(curve, t))


def forward_rate(curve: ZeroCurve, t: jax.Array, eps: float = 1e-6) -> jax.Array:
    """f(0,t) = -d/dt ln P(0,t), today's instantaneous forward rate at t,
    via central-difference-free forward finite difference (matching every
    prior implementation's own `eps=1e-6` convention exactly, for bit-
    identical output to the NumPy versions this replaces)."""
    return -(log_discount(curve, t + eps) - log_discount(curve, t)) / eps


def B(t: jax.Array, T: jax.Array, a: float) -> jax.Array:
    """B(t,T) = (1 - exp(-a*(T-t))) / a.

    a==0.0 (arithmetic Brownian motion, the mathematically valid a->0 limit
    of Ornstein-Uhlenbeck mean reversion) is a removable 0/0 singularity in
    this formula as literally written; its analytic limit is B(t,T) = T-t
    (L'Hopital / first-order Taylor expansion of the exponential). Guarded
    via the standard JAX branch-free pattern: evaluate on a safe
    placeholder `a` (never actually 0), then select the correct branch with
    `jnp.where` -- both branches are always computed (required for
    `jax.grad`/`jax.jit` tracing), the placeholder result is simply
    discarded when `a != 0`."""
    a_safe = jnp.where(a == 0.0, 1.0, a)
    return jnp.where(a == 0.0, T - t, (1.0 - jnp.exp(-a_safe * (T - t))) / a_safe)


def A(curve: ZeroCurve, t: jax.Array, T: jax.Array, a: float, sigma: float, B_override: jax.Array = None) -> jax.Array:
    """
    A(t,T) = [P(0,T)/P(0,t)] * exp(B(t,T)*f(0,t) - (sigma^2/4a)*(1-exp(-2at))*B(t,T)^2)

    The term that calibrates the affine bond-price family to today's
    actual market curve: plugging in t=0 reproduces the curve's own
    P(0,T) exactly (B(0,T)*f(0,0) - variance_term(0) both vanish at t=0 as
    written). Matches `QuantLib::HullWhite::A`/`Vasicek::B` exactly (see
    docs/reference/ore-parity.md section 3b for the line-by-line C++
    correspondence) -- live-verified to ~1e-12 relative precision against
    `ORE.HullWhite.discountBond` across many (t,T,r) combinations and both
    flat and sloped input curves throughout this codebase's test suite.

    t, T: broadcastable arrays of times (year-fractions from today).

    B_override: use this value in place of `B(t,T,a)` in the exponent
    term, while still using the raw (possibly T<t) `t`/`T` for the
    `ratio`/`f(0,t)` terms above. Exists solely so `engine.simulation.
    compute_hw_A_matrix` can reproduce its own long-standing yield-curve-
    cube convention exactly: `generate_paths` computes ITS OWN `B_matrix`
    once with `T_minus_t` clamped at 0 (a cashflow pillar earlier than the
    current simulated step -- an "aged" pillar, see `engine.instruments.
    swap`'s "Known limitation: no representation of an already-fixed/
    elapsed coupon" docstring), and `reconstruct_yield_curves` downstream
    combines THAT SAME clamped `B_matrix` with whatever `A` this function
    returns as `A * exp(-B_matrix*r)` -- so `A`'s own internal exponent
    term must use the identical clamped `B`, not silently recompute an
    unclamped one, or the two would disagree on what `B` means for an
    aged pillar and the combined discount factor would be inconsistent
    with itself. This is a pre-existing, documented approximation this
    function's job is to reproduce bit-for-bit, not a new correctness
    requirement -- ordinary callers (European/Bermudan swaption pricing,
    Greeks) never pass `B_override` and get the plain, self-consistent
    `B(t,T,a)` computed from the same `t`,`T` throughout.
    """
    log_P0_t = log_discount(curve, t)
    log_P0_T = log_discount(curve, T)
    fwd_0_t = forward_rate(curve, t)
    ratio = jnp.exp(log_P0_T - log_P0_t)
    B_t_T = B(t, T, a) if B_override is None else B_override

    # Same a==0 removable-singularity guard as B() above: the analytic
    # a->0 limit of (sigma^2/4a)*(1-exp(-2at)) is sigma^2*t/2.
    a_safe = jnp.where(a == 0.0, 1.0, a)
    variance_term = jnp.where(
        a == 0.0,
        0.5 * sigma ** 2 * t,
        (sigma ** 2 / (4.0 * a_safe)) * (1.0 - jnp.exp(-2.0 * a_safe * t)),
    )
    return ratio * jnp.exp(B_t_T * fwd_0_t - variance_term * B_t_T ** 2)


def bond_price(curve: ZeroCurve, t: jax.Array, T: jax.Array, r: jax.Array, a: float, sigma: float) -> jax.Array:
    """P(t,T) = A(t,T) * exp(-B(t,T)*r) -- the full HW1F affine bond price,
    conditional on a short rate `r` observed at time `t` (r(0) = today's
    curve's own short end recovers the deterministic t=0 price)."""
    return A(curve, t, T, a, sigma) * jnp.exp(-B(t, T, a) * r)


def bond_option_sigma(T_opt: jax.Array, S: jax.Array, t: jax.Array, a: float, sigma: float) -> jax.Array:
    """sigma_p: the volatility (as seen from t) of the zero-coupon bond
    price P(T_opt,S) -- the standard HW1F bond-option volatility (Brigo-
    Mercurio 3.41), live-verified bit-for-bit against
    `ORE.HullWhite.discountBondOption`."""
    B_Topt_S = B(T_opt, S, a)
    return sigma * B_Topt_S * jnp.sqrt(jnp.clip(1.0 - jnp.exp(-2.0 * a * (T_opt - t)), 0.0, None) / (2.0 * a))


def bond_call(P_t_Topt: jax.Array, P_t_S: jax.Array, K: jax.Array, sigma_p: jax.Array) -> jax.Array:
    """Black-formula call on a zero-coupon bond -- ORE's
    `HullWhite::discountBondOption` closed form, live-verified bit-for-bit
    against the installed ORE package.

    sigma_p == 0 occurs whenever a bond's own maturity S coincides with the
    option's expiry T_opt (always true of a swaption's notional-received-
    at-accrual-start leg for a non-forward-starting trade), or when pricing
    exactly at expiry (t == T_opt). Both collapse to the deterministic
    intrinsic payoff max(P_S - K*P_T, 0) in the zero-vol limit -- guarded
    here directly (not left to the caller) so every call site is correct
    by construction, using jnp.where to stay branch-free/jit-friendly (the
    Black-formula branch is still evaluated on a safe placeholder sigma_p
    to avoid a 0/0 NaN contaminating the gradient-safe branch, then
    discarded)."""
    sigma_p_safe = jnp.where(sigma_p > 0.0, sigma_p, 1.0)
    h = (1.0 / sigma_p_safe) * jnp.log(P_t_S / (P_t_Topt * K)) + sigma_p_safe / 2.0
    black = P_t_S * norm.cdf(h) - K * P_t_Topt * norm.cdf(h - sigma_p_safe)
    intrinsic = jnp.maximum(P_t_S - K * P_t_Topt, 0.0)
    return jnp.where(sigma_p > 0.0, black, intrinsic)


def bond_put(P_t_Topt: jax.Array, P_t_S: jax.Array, K: jax.Array, sigma_p: jax.Array) -> jax.Array:
    """Put-call parity on `bond_call` above."""
    call = bond_call(P_t_Topt, P_t_S, K, sigma_p)
    return call - (P_t_S - K * P_t_Topt)
