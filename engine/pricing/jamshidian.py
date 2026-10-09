"""
European swaptions with QuantLib's `JamshidianSwaptionEngine`
(ql/pricingengines/swaption/jamshidianswaptionengine.cpp) on a Hull-White model
(`HullWhite(termStructure, a, sigma)`): the configurable alternative to ORE's default Bachelier
engine, `PricingConfig.european = "Jamshidian"` with the model in `PricingConfig.jamshidian`.
ORE has no builder for it; it is kept as an option (decision A-1: options are added, not
swapped), and it is the same engine whichever model simulates the paths.

On a valuation date at model time t, with every time measured from it and P(T) the curve's
discount factor (today's curve, a bumped one, or a simulated path's): the swap entered on
exercise at T0 starts at Tv and pays fixed amounts c_i at T_i, with the nominal N added to the
last. The engine finds the state x* at which the coupon bond, in units of the bond maturing at
Tv, is worth the nominal (`rStarFinder`):

    sum_i c_i K_i(x*) = N,   K_i(x) = P(T0, T_i | x) / P(T0, Tv | x)
                                    = P(T_i)/P(Tv) exp(-(H_i - H_v) x - 1/2 (H_i^2 - H_v^2) zeta(T0)),

and prices each c_i as an option on the forward bond P(T0, T_i)/P(T0, Tv) struck at K_i
(`HullWhite::discountBondOption(type, strike, maturity, bondStart, bondMaturity)`): Black on
the forward P(T_i) with strike K_i P(Tv) and standard deviation |H_i - H_v| sqrt(zeta(T0)); puts
for a payer, calls for a receiver.

It is written in the LGM form of the Hull-White model (`engine.models.lgm`: H(t) = (1 - e^{-at})/a,
zeta(t) = sigma^2 (e^{2at} - 1)/(2a)), which is QuantLib's formula term by term: its
`discountBond(T0, T, r)` carries the forward f(0, T0) into K_i's numerator and denominator
alike, and its bond option volatility `sigma/(a sqrt(2a)) sqrt(c)` equals |H_i - H_v|
sqrt(zeta(T0)). So the engine reads only discount factors and prices on any curve. The Hull-White
model is time-homogeneous with a constant volatility, so the price on a path's curve is the
model's conditional price there.

Differs from QuantLib: x* is solved to float64 rounding (by the configured solver,
`JamshidianEngineConfig.solver`, decision A-21; differentiated by the implicit function
theorem) where QuantLib's Brent stops at 1e-8; and the exercise-into swap is
the coupons `BlackMultiLegOptionEngine` reads (paying after expiry and accruing from it,
`engine.pricing.european.european_terms`), which is QuantLib's whole underlying for any
European exercising on or before its first accrual start. The floating leg is valued at par on
the discount curve (QuantLib's engine is single-curve): the index's forwarding curve is not
read. Refused as QuantLib refuses them (`validate_jamshidian`): a floating spread and cash
settlement.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from engine.instruments.european_swaption import SwaptionConfig
from engine.market_data.market import index_name
from engine.models.hull_white import bond_call, bond_put
from engine.pricing.config import JamshidianEngineConfig
from engine.pricing.context import PricingContext
from engine.pricing.european import EuropeanTerms, european_terms
from engine.pricing.legs import PathSchedule, discount_from, on_every_date
from engine.solvers.roots import Steps, implicit_root


#: Steps of x* (`engine.solvers.roots`): Bisection's are the count before Newton (A-21),
#: Newton's measured (2026-10-07): x* from 0 reaches its rounding in 5 steps for the shared
#: European struck from 0% to 15%, payer and receiver, today and on 256 LGM paths under two
#: Hull-White models (the NPV then within 6e-16 of the nominal of Bisection's), and in 24 for
#: the hardest strike the formula still resolves, -99% (its coupons cancel the nominal, and the
#: coupon bond is not monotone across the window, so Newton bisects its way in; scanned from
#: -99% to 200%, tenors 2Y to 30Y; tests/test_root_solvers.py, tests/test_jamshidian.py). A
#: trade's strike is unbounded, unlike a helper's, so the count covers the hardest case, with
#: two steps of margin.
X_STAR_STEPS = Steps(bisection=100, newton=26)

#: x*'s starting window, widened when it does not hold the root.
X_STAR_WINDOW = 2.0


def validate_jamshidian(cfg: SwaptionConfig) -> None:
    """Refuse what QuantLib's `JamshidianSwaptionEngine` refuses."""
    if cfg.floating_spread != 0.0:
        raise ValueError(f"trade {cfg.trade_id!r}: floating_spread={cfg.floating_spread}: the Jamshidian engine "
                         f"values the floating leg at par and cannot price a spread (QuantLib's refuses it too, I-37)")
    if cfg.settlement != "Physical":
        raise ValueError(f"trade {cfg.trade_id!r}: settlement={cfg.settlement!r}: the Jamshidian engine prices "
                         f"physical settlement only (QuantLib's refuses ORE's cash method, ParYieldCurve; I-52)")


def jamshidian_npv(terms: EuropeanTerms, model: JamshidianEngineConfig, disc, t) -> jax.Array:
    """The engine's NPV on a valuation date at model time `t`, on the discount curve `disc`
    measured from it (any batch axes, e.g. paths). The option must not have expired
    (expiry after the date).

    x* is the root of the coupon bond less the nominal, by `model.solver`, with its derivative
    in the curve by the implicit function theorem. Not jitted on its own: x*'s derivative rule
    closes over this function's intermediates, which `jax.grad` through a jit boundary cannot
    carry (the AD Greeks differentiate it). Its path cube is jitted (`_jamshidian_cube`)."""
    legs = terms.legs
    a, sigma = model.reversion, model.volatility
    p_bond = discount_from(disc, t, legs.fixed_pay)                  # [..., n] P(T_i)
    p_start = discount_from(disc, t, terms.start_time)               # [...]    P(Tv)
    dtype = p_bond.dtype
    H = lambda s: -jnp.expm1(-a * s) / a                             # noqa: E731
    expiry = jnp.maximum(jnp.asarray(terms.expiry_time - t, dtype=dtype), 0.0)
    zeta = sigma ** 2 * jnp.expm1(2.0 * a * expiry) / (2.0 * a)
    H_i = H(jnp.asarray(legs.fixed_pay - t, dtype=dtype))
    H_v = H(jnp.asarray(terms.start_time - t, dtype=dtype))
    dH, convexity = H_i - H_v, 0.5 * (H_i ** 2 - H_v ** 2) * zeta
    amounts = jnp.asarray(legs.fixed_amount, dtype=dtype)
    amounts = amounts.at[-1].add(jnp.asarray(terms.nominal, dtype=dtype))
    nominal = jnp.asarray(terms.nominal, dtype=dtype)

    def forward_bonds(x, ratio):
        return ratio * jnp.exp(-dH * x[..., None] - convexity)

    def coupon_bond(x, ratio):
        return jnp.sum(amounts * forward_bonds(x, ratio), axis=-1) - nominal

    ratio = p_bond / p_start[..., None]
    x_star = implicit_root(coupon_bond, ratio, jnp.zeros(p_start.shape, dtype=dtype), solver=model.solver,
                           steps=X_STAR_STEPS, increasing=False, window=X_STAR_WINDOW)
    strikes = forward_bonds(x_star, ratio)
    option = bond_put if legs.payer else bond_call
    std_dev = jnp.abs(dH) * jnp.sqrt(zeta)
    return jnp.sum(amounts * option(p_start[..., None], p_bond, strikes, std_dev), axis=-1)


def jamshidian_value(cfg: SwaptionConfig, context: PricingContext, model: JamshidianEngineConfig) -> jax.Array:
    """The engine's NPV on the context's date (0 once expired: exercise on or before the
    date)."""
    if not cfg.exercise_date > context.date:
        return jnp.zeros(())
    extra = context.fixings.get(index_name(cfg.currency, cfg.index_tenor_months), {})
    terms = european_terms(cfg, context.date, extra)
    return jamshidian_npv(terms, model, context.discount[cfg.currency], 0.0)


def jamshidian_cube(terms: EuropeanTerms, model: JamshidianEngineConfig, schedule: PathSchedule, times: np.ndarray,
                    disc, index, fixings: jax.Array, alive: np.ndarray) -> jax.Array:
    """`[S, D]` NPVs on every path and date (0 from expiry on), in the curves' dtype; `alive`
    `[D]` says the option has not expired on each date."""
    return _jamshidian_cube(terms.astype(disc.log_discounts.dtype), model, schedule, times, disc, index, fixings,
                            alive)


@partial(jax.jit, static_argnums=1)
def _jamshidian_cube(terms: EuropeanTerms, model: JamshidianEngineConfig, schedule: PathSchedule, times, disc, index,
                     fixings: jax.Array, alive) -> jax.Array:
    def value(disc_j, _index, t_j, _fixed, _float, _projected, _known, alive_j):
        return jnp.where(alive_j, jamshidian_npv(terms, model, disc_j, t_j), 0.0)

    return on_every_date(value, terms.legs, schedule, times, disc, index, fixings, alive)
