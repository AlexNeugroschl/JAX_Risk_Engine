"""
t=0 price of each trade as a pure JAX function of its curves' pillar zero rates.

`engine.risk.greeks` differentiates these (Delta, Gamma, Vega) and `engine.market_risk`
evaluates them under shocked curves. Each builder does the CPU trade setup once and returns
a closure of pillar-rate arrays only, so it can be `jax.grad`-ed, `jax.vmap`-ed and jitted.

    swap_price_function(cfg, disc, fwd)  -> f(disc_rates, fwd_rates)
    swaption_price_function(cfg, curve)  -> f(rates)
    bermudan_price_function(cfg, curve)  -> (f(rates, sigma_values), sigma_values)
    bond_price_function(cfg)             -> f(rates)   (engine.instruments.treasury)

The curve fixes the pillar times and the working dtype; its rates are only the base point.
An option expired on its evaluation date prices to a constant 0 (ORE's `isExpired`), so
its sensitivities are 0.
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from engine.instruments.swap import SwapConfig, _price_one_swap, prepare_swap, swap_schedule
from engine.instruments.european_swaption import (
    SwaptionConfig,
    _bond_call,
    _bond_option_sigma,
    _bond_put,
    _hw_B,
    _solve_rstar,
    prepare_swaption,
)
from engine.instruments.bermudan_swaption import (
    BermudanSwaptionConfig,
    _run_backward_induction,
    prepare_bermudan,
)
from engine.instruments.treasury import bond_price_function  # noqa: F401  (re-export)
from engine.models.hull_white import A as _hw_A, ZeroCurve, discount as _discount_at, zero_rate as _zero_rate_at
from engine.models.lgm import Sigma, as_sigma
from engine.market import VOL_DAY_COUNTER, SwaptionVolSurface
from engine.valuation.european import black_multileg_npv, european_terms
from engine.valuation.legs import today_schedule


def _yield_curves_from_zero_curves(
    disc_curve: ZeroCurve, fwd_curve: ZeroCurve, maturities: jax.Array,
) -> jax.Array:
    """`[1, 1, Maturities, 2]` yield-curve tensor for `_price_one_swap`, built directly
    from two curves: index 0 is discount, index 1 forward."""
    disc = _discount_at(disc_curve, maturities)
    fwd = _discount_at(fwd_curve, maturities)
    cube = jnp.stack([disc, fwd], axis=-1)  # [Maturities, 2]
    return cube[None, None, :, :]


def swap_price_function(cfg: SwapConfig, disc_curve: ZeroCurve, fwd_curve: ZeroCurve):
    """`(disc_rates, fwd_rates) -> t=0 NPV`, differentiable.

    Uses the cube kernel `_price_one_swap`, but with the swap's own cashflow times as the
    "pillars" and the curves interpolated at them, so there is no pillar-alignment
    constraint. The ORE trade is built once, outside the closure."""
    # `dataclasses.replace` keeps every field (a hand-written copy once dropped
    # `accrual_day_count`). Curve indices are set to this function's local 0/1 layout.
    local_cfg = dataclasses.replace(cfg, discount_curve_index=0, forward_curve_index=1)

    # The swap's cashflow times serve as both the pillar array and the query points.
    maturities = swap_schedule(local_cfg).pillar_times()
    prepared_swap = prepare_swap(local_cfg, np.asarray(maturities))
    # Use the curve's dtype: under jax_enable_x64, mixing in a float64 array would
    # upcast a float32 risk computation.
    maturities_jax = jnp.asarray(maturities, dtype=disc_curve.pillar_rates.dtype)

    def price_fn(disc_rates: jax.Array, fwd_rates: jax.Array) -> jax.Array:
        dc = ZeroCurve(pillar_times=disc_curve.pillar_times, pillar_rates=disc_rates)
        fc = ZeroCurve(pillar_times=fwd_curve.pillar_times, pillar_rates=fwd_rates)
        yield_curves = _yield_curves_from_zero_curves(dc, fc, maturities_jax)
        npv = _price_one_swap(yield_curves, prepared_swap)
        return npv[0, 0]

    return price_fn


def bachelier_swaption_price_function(cfg: SwaptionConfig, curve: ZeroCurve, surface: SwaptionVolSurface):
    """`curve_rates -> t=0 NPV` for one European swaption with ORE's default engine
    (`BlackMultiLegOptionEngine`, `engine.valuation.european`) on one curve for discounting
    and forwarding, the normal volatility read once from `surface` and held fixed. The trade must carry no Hull-White parameters. Computed in the curve's
    dtype."""
    dtype = curve.pillar_rates.dtype
    asof = cfg.evaluation_date
    if not cfg.exercise_date > asof:
        return lambda pillar_rates: jnp.zeros((), dtype=pillar_rates.dtype) * jnp.sum(pillar_rates)
    terms = european_terms(cfg, asof)
    t = VOL_DAY_COUNTER.yearFraction(asof, cfg.exercise_date)
    variance = float(surface.volatility(asof, t, terms.swap_length)) ** 2 * t
    known = today_schedule(terms.legs, asof)
    terms = terms.astype(dtype)
    known_rates = np.asarray(known.known_rates, dtype=dtype)

    def price(pillar_rates):
        on = ZeroCurve(pillar_times=curve.pillar_times, pillar_rates=pillar_rates)
        return black_multileg_npv(terms, on, on, 0.0, variance, known.projected, known_rates)

    return price


def swaption_price_function(cfg: SwaptionConfig, curve: ZeroCurve):
    """`curve_rates -> t=0 NPV` for one European swaption, differentiable.

    The t=0 case of `_price_one_swaption`, with A(t,T) computed from the traced curve
    (`hull_white.A`) and the same building blocks (`_solve_rstar`, bond options). The
    trade is prepared once, outside the closure."""
    if cfg.is_expired():
        return lambda pillar_rates: jnp.zeros((), dtype=pillar_rates.dtype) * jnp.sum(pillar_rates)
    swaption = prepare_swaption(cfg)
    a = swaption.hw_a
    sigma = swaption.hw_sigma
    T0 = swaption.exercise_time
    T_start = swaption.accrual_start_time
    cf_times = swaption.fixed_cashflow_times
    notional = swaption.notional
    # Use the curve's dtype (see swap_price_function).
    _dtype = curve.pillar_rates.dtype
    all_times = jnp.asarray(
        np.concatenate([cf_times, cf_times[-1:], [T_start]]), dtype=_dtype
    )
    all_amounts = jnp.concatenate([
        jnp.asarray(swaption.fixed_cashflow_amounts, dtype=_dtype),
        jnp.asarray([notional, -notional], dtype=_dtype),
    ])
    B_T0_Ti = _hw_B(T0, all_times, a)  # [N+1]

    def coupon_bond_value(rstar, params):
        # rstar: scalar. A_T0_Ti is passed explicitly so _solve_rstar's custom_jvp can
        # differentiate it (see european_swaption._solve_rstar).
        A_T0_Ti = params
        prices = A_T0_Ti * jnp.exp(-B_T0_Ti * rstar)
        return jnp.sum(prices * all_amounts)

    def price_fn(pillar_rates: jax.Array) -> jax.Array:
        curve_local = ZeroCurve(pillar_times=curve.pillar_times, pillar_rates=pillar_rates)
        A_T0_Ti = _hw_A(curve_local, jnp.full_like(all_times, T0), all_times, a, sigma)

        rstar = _solve_rstar(coupon_bond_value, A_T0_Ti, ())
        K = A_T0_Ti * jnp.exp(-B_T0_Ti * rstar)

        # t=0 bond prices P(0,S) = A(0,S)*exp(-B(0,S)*r0). r0 is taken as the zero rate at
        # t=1e-6, while hull_white.A uses the exact f(0,0) = z(0); on a curve sloped in
        # its first segment P(0,S) therefore differs from the curve by a factor
        # exp(B(0,S)*(z(0) - z(1e-6))), of order 1e-8 relative.
        r0 = _zero_rate_at(curve_local, jnp.asarray(1e-6, dtype=_dtype))
        B_0_Ti = _hw_B(0.0, all_times, a)
        B_0_T0 = _hw_B(0.0, T0, a)
        A_0_Ti = _hw_A(curve_local, jnp.zeros_like(all_times), all_times, a, sigma)
        A_0_T0 = _hw_A(curve_local, jnp.zeros((), dtype=_dtype), T0, a, sigma)
        P_0_Ti = A_0_Ti * jnp.exp(-B_0_Ti * r0)
        P_0_T0 = A_0_T0 * jnp.exp(-B_0_T0 * r0)

        sigma_p = _bond_option_sigma(T0, all_times, 0.0, a, sigma)
        bond_fn = _bond_put if swaption.payer else _bond_call
        per_leg = bond_fn(P_0_T0, P_0_Ti, K, sigma_p)
        return jnp.sum(per_leg * all_amounts)

    return price_fn


def bermudan_price_function(cfg: BermudanSwaptionConfig, curve: ZeroCurve):
    """
    `(pillar_rates, sigma_values) -> t=0 NPV` for one Bermudan/American swaption,
    differentiable in the pillar rates (Delta/Gamma) and the sigma bucket values (Vega).

    Reuses `prepare_bermudan`/`_run_backward_induction` directly, substituting the traced
    values into `_PreparedBermudan.curve` and `hw_sigma`. `sigma_values` is
    `cfg.hw_sigma.values` for a `Sigma`, or a one-element array for a flat sigma. Bucket
    times are not differentiated, as ORE's sensitivities bump values, not times.

    `curve` is not read: the pillar times are `cfg.initial_zero_curve.times`, and
    `pillar_rates` must be given on them.
    """
    from dataclasses import replace

    base_sigma = as_sigma(cfg.hw_sigma)
    sigma_times = base_sigma.times
    if cfg.is_expired():
        def expired_fn(pillar_rates, sigma_values):
            return jnp.zeros((), dtype=pillar_rates.dtype) * (jnp.sum(pillar_rates) + jnp.sum(sigma_values))
        return expired_fn, base_sigma.values
    swap = prepare_bermudan(cfg)

    def price_fn(pillar_rates: jax.Array, sigma_values: jax.Array) -> jax.Array:
        local_swap = replace(
            swap,
            curve=ZeroCurve(pillar_times=swap.curve.pillar_times, pillar_rates=pillar_rates),
            hw_sigma=Sigma(times=sigma_times, values=sigma_values),
        )
        result = _run_backward_induction(local_swap, condition_times=[])
        return result.value_at_t0

    return price_fn, base_sigma.values
