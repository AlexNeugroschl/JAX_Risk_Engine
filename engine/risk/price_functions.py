"""
Today's (t=0) price of each trade as a pure JAX function of its curves'
pillar zero rates.

These are what `engine.risk.greeks` differentiates (Delta, Gamma, Vega) and
what `engine.market_risk` evaluates under shocked curves. Each builder does
the CPU-side trade construction once (ORE schedules, cashflow times) and
returns a closure whose only inputs are pillar-rate arrays, so it can be
`jax.grad`-ed, `jax.vmap`-ed over scenarios and jitted.

    swap_price_function(cfg, disc, fwd)  -> f(disc_rates, fwd_rates)
    swaption_price_function(cfg, curve)  -> f(rates)
    bermudan_price_function(cfg, curve)  -> (f(rates, sigma_values), sigma_values)
    bond_price_function(cfg)             -> f(rates)   (engine.instruments.treasury)

The curve passed in fixes the pillar TIMES (and, through its dtype, the
working precision); the rates it carries are only the base point.
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from engine.instruments.swap import SwapConfig, _build_ore_swap, _price_one_swap, prepare_swap
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
from engine.models.ore_builders import fixed_leg_cashflows, floating_leg_cashflows
from engine.models.hull_white import A as _hw_A, ZeroCurve, discount as _discount_at, zero_rate as _zero_rate_at
from engine.models.lgm import Sigma, as_sigma


def _yield_curves_from_zero_curves(
    disc_curve: ZeroCurve, fwd_curve: ZeroCurve, maturities: jax.Array,
) -> jax.Array:
    """Builds the `[1, 1, Maturities, 2]` yield_curves tensor
    `_price_one_swap` expects, directly from two `ZeroCurve`s -- a
    deterministic (no simulated noise), differentiable stand-in for
    `generate_paths(...)["yield_curves"]`'s zero-shock t=0 slice. Curve
    index 0 is always the discount curve, index 1 the forward curve, in
    this tensor -- a fixed local convention for this Greeks module only
    (independent of whatever discount_curve_index/forward_curve_index a
    caller's SwapConfig was built with against some larger simulation
    cube), which is why `swap_delta_gamma`/`swap_theta` below construct
    their own single-swap `SwapConfig` copy with indices forced to
    (0, 1)."""
    disc = _discount_at(disc_curve, maturities)
    fwd = _discount_at(fwd_curve, maturities)
    cube = jnp.stack([disc, fwd], axis=-1)  # [Maturities, 2]
    return cube[None, None, :, :]


def swap_price_function(cfg: SwapConfig, disc_curve: ZeroCurve, fwd_curve: ZeroCurve):
    """Builds a `(disc_rates, fwd_rates) -> t=0 NPV` closure, differentiable
    end-to-end via `jax.grad`/`jax.hessian`. `cfg`'s own trade structure
    (schedule, day counts) is resolved via ORE ONCE, outside the returned
    closure -- only the curve-to-NPV tensor math is retraced per gradient
    evaluation, matching every other pricer's own CPU-setup/GPU-math split
    (see swap.py's module docstring).

    Unlike the main pricer (which requires every cashflow date to land
    EXACTLY on a pre-tabulated `maturities` pillar -- see
    swap.py's maturity-pillar-alignment limitation), this function uses
    the swap's own real cashflow dates AS the "maturities" array and
    evaluates the curve continuously at exactly those dates via
    `_discount_at`'s interpolation -- there is no pillar-mismatch
    constraint to satisfy here, since the yield-curve tensor is built
    fresh (via interpolation, not lookup) for exactly the dates this one
    trade needs."""
    # `dataclasses.replace`, not a hand-copied constructor call: a copy that
    # lists fields explicitly silently drops any it forgets, which is how
    # `accrual_day_count` used to fall back to ACT/365 on this path.
    local_cfg = dataclasses.replace(cfg, discount_curve_index=0, forward_curve_index=1)
    swap = _build_ore_swap(local_cfg)
    today = local_cfg.evaluation_date
    fixed = fixed_leg_cashflows(swap, today)
    floating = floating_leg_cashflows(swap, today)

    # The swap's own cashflow dates, deduplicated -- exactly the set of
    # times the curve needs to be evaluated at (via interpolation, not a
    # fixed pillar lookup), used both as the "maturities" pillar array
    # AND as the query points for the differentiable curve.
    maturities = sorted(set(
        fixed.payment_times.tolist() + floating.payment_times.tolist()
        + floating.accrual_start_times.tolist() + floating.accrual_end_times.tolist()
    ))
    prepared_swap = prepare_swap(local_cfg, np.asarray(maturities))
    # Derived from disc_curve's own dtype (not hardcoded) -- disc_curve/
    # fwd_curve carry whatever dtype the caller built them at (governed by
    # PrecisionConfig.risk when reached via engine.portfolio.request), and
    # mixing a hardcoded-float64 array with a float32 curve inside price_fn
    # below would silently upcast the curve back to float64 under
    # jax_enable_x64=True (confirmed: jnp.interp promotes a float32/float64
    # mix to float64 whenever x64 is enabled, regardless of which operand is
    # which dtype) -- see this module's docstring and engine.portfolio.
    # request's "Concurrency" section on why jax_enable_x64 being process-
    # global makes this a genuine, not theoretical, correctness gap.
    maturities_jax = jnp.asarray(maturities, dtype=disc_curve.pillar_rates.dtype)

    def price_fn(disc_rates: jax.Array, fwd_rates: jax.Array) -> jax.Array:
        dc = ZeroCurve(pillar_times=disc_curve.pillar_times, pillar_rates=disc_rates)
        fc = ZeroCurve(pillar_times=fwd_curve.pillar_times, pillar_rates=fwd_rates)
        yield_curves = _yield_curves_from_zero_curves(dc, fc, maturities_jax)
        npv = _price_one_swap(yield_curves, prepared_swap)
        return npv[0, 0]

    return price_fn


def swaption_price_function(cfg: SwaptionConfig, curve: ZeroCurve):
    """Builds a `curve_rates -> t=0 NPV` closure for a single European
    swaption, differentiable via `jax.grad`/`jax.hessian` with respect to
    `curve.pillar_rates` -- reimplements `_price_one_swaption`'s t=0 path
    using `engine.models.hull_white.A` (the SAME shared, JAX-native
    formula `price_swaptions`' own NumPy-facing `compute_hw_A` wraps) in
    place of a separately-maintained JAX twin, and
    otherwise reusing the SAME JAX building blocks the main pricer already
    uses (`_hw_B`, `_bond_option_sigma`, `_bond_call`/`_bond_put`,
    `_solve_rstar`) -- only the today's-curve lookup changes, not the
    option-pricing formulas themselves. Trade structure (cashflow times/
    amounts, exercise time) is resolved via `prepare_swaption` ONCE,
    outside the returned closure, exactly like `_swap_price_fn` above."""
    swaption = prepare_swaption(cfg)
    a = swaption.hw_a
    sigma = swaption.hw_sigma
    T0 = swaption.exercise_time
    T_start = swaption.accrual_start_time
    cf_times = swaption.fixed_cashflow_times
    notional = swaption.notional
    # Derived from curve's own dtype (not hardcoded) -- see
    # _swap_price_fn's maturities_jax comment above for why a hardcoded
    # dtype here would silently upcast curve.pillar_rates back to float64
    # under jax_enable_x64=True whenever a caller requests risk=32.
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
        # rstar: scalar. params = A_T0_Ti -- the explicit pytree
        # _solve_rstar's jax.custom_jvp differentiates with respect to
        # (see european_swaption._solve_rstar's docstring on why this
        # must be explicit, not a closure).
        A_T0_Ti = params
        prices = A_T0_Ti * jnp.exp(-B_T0_Ti * rstar)
        return jnp.sum(prices * all_amounts)

    def price_fn(pillar_rates: jax.Array) -> jax.Array:
        curve_local = ZeroCurve(pillar_times=curve.pillar_times, pillar_rates=pillar_rates)
        A_T0_Ti = _hw_A(curve_local, jnp.full_like(all_times, T0), all_times, a, sigma)

        rstar = _solve_rstar(coupon_bond_value, A_T0_Ti, ())
        K = A_T0_Ti * jnp.exp(-B_T0_Ti * rstar)

        # t=0 conditioning: P(t,S) = A(t,S)*exp(-B(t,S)*r(t)) for ANY t,
        # including t=0 -- B(0,S) is NOT zero (only t==S makes B(t,S)==0),
        # so r(0) genuinely matters here, exactly as it does at every other
        # step in _price_one_swaption. r(0), the model's own "no shock"
        # short rate, is the curve's own initial instantaneous forward
        # rate f(0,0) -- the same quantity _price_one_swaption reads off
        # hw_paths[:, 0, :] at the simulation's own first step (by
        # construction, RatesConfig.initial_rates is chosen to equal the
        # calibration curve's own short end in every scenario in this
        # codebase). Computed here as the curve's own zero rate at the
        # (numerically tiny but nonzero) limit t->0, matching
        # compute_hw_A's own f(0,t) finite-difference convention.
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
    Builds a `(pillar_rates, sigma_values) -> t=0 NPV` closure for a single
    Bermudan/American swaption, differentiable via `jax.grad`/`jax.hessian`
    with respect to BOTH the curve's own pillar rates (Delta/Gamma) and the
    calibrated Sigma's own bucket values (Vega) -- `_run_backward_
    induction` is already fully JAX-native end-to-end as of Phase 2 (no
    NumPy/Python control flow standing between these inputs and the
    output), so this closure needs no separate JAX reimplementation of the
    backward induction the way `_swaption_price_fn` needed for Jamshidian's
    (originally NumPy) formula -- it reuses `prepare_bermudan`/
    `_run_backward_induction` directly, only substituting differentiable
    values for `_PreparedBermudan.zero_rates`/`hw_sigma` via
    `dataclasses.replace`.

    `sigma_values` is always a flat JAX array -- `len(cfg.hw_sigma.values)`
    if `cfg.hw_sigma` is already a `Sigma` (the calibrated-Vega case), or a
    single-element array wrapping a flat scalar otherwise (Delta/Gamma-only
    callers, or a flat-sigma trade with no calibration behind it). Bucket
    BREAKPOINTS (`Sigma.times`) are never a differentiation target (mirrors
    `ZeroCurve.pillar_times` never being one for Delta/Gamma -- ORE's own
    sensitivity framework bumps a bucket's VALUE, never its own time grid).

    `cfg.hw_sigma` (NOT this closure's own `curve` argument) determines the
    initial zero curve `_PreparedBermudan.zero_times` uses -- unlike
    `_swaption_price_fn`, whose caller passes the curve as a wholly
    separate argument, `cfg.initial_zero_curve` is what `prepare_bermudan`
    reads to set `zero_times`; the `curve` parameter here supplies the
    PILLAR TIMES (`curve.pillar_times` must equal `cfg.initial_zero_curve.
    times` -- the same "same curve, JAX-native copy" convention
    `swaption_delta_gamma`'s own docstring documents) while `pillar_rates`
    becomes the differentiable rate values substituted in.
    """
    from dataclasses import replace

    swap = prepare_bermudan(cfg)
    base_sigma = as_sigma(cfg.hw_sigma)
    sigma_times = base_sigma.times

    def price_fn(pillar_rates: jax.Array, sigma_values: jax.Array) -> jax.Array:
        local_swap = replace(
            swap,
            zero_rates=pillar_rates,
            hw_sigma=Sigma(times=sigma_times, values=sigma_values),
        )
        result = _run_backward_induction(local_swap, condition_times=[])
        return result.value_at_t0

    return price_fn, base_sigma.values
