"""
Independent reference values for single-exercise Bermudan swaptions.

With one exercise date `T0`, a Bermudan's value is exactly

    V(0) = E[ max(U(T0, x), 0) / N(T0, x) ],   x ~ Normal(0, zeta(T0)),

where `U` is the value of the swap exercised into and `N` the LGM numeraire
(`N(0, 0) = 1`), both in the prepared trade's model, shifted at its horizon (the state is
driftless in the shifted model as in the unshifted one). `single_exercise_value_by_integration` computes that
expectation by a fine direct integral over x: no Hagan grid, no convolution
rollback and none of ORE's cashflow bookkeeping, so agreement tests exactly
the machinery the backward induction adds. The cashflow values (`cashflow_values_at_nodes`)
are the LGM's bond prices as `engine.models.lgm` writes them, independent of the engine's own
decomposition into each path's coefficients and functions of the state
(`engine.pricing.lgm_grid`), which `tests/test_bermudan_swaption.py` holds to them; the engine is
pinned separately against ORE in tests/test_ore_lgm_parity.py.

A Jamshidian decomposition is not used: it needs the floating leg to telescope to
`notional * (P(T_start) - P(T_end))`, which fails once each coupon is projected over its
index fixing period, as ORE projects it (I-31).
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import norm as scipy_norm

from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.models.lgm import bond_price, horizon_shift, numeraire as _numeraire, zeta as _zeta
from tests.support.lgm_engine import prepared


def _bond_prices_at_nodes(curve, a: float, sigma, t, maturities, x_nodes, shift) -> jax.Array:
    """P(t, T; x) for every state node and maturity, `[nodes, maturities]`."""
    return jax.vmap(lambda T: bond_price(curve, a, sigma, t, T, x_nodes, shift))(maturities).T


def cashflow_values_at_nodes(swap, x_nodes, t, shift) -> jax.Array:
    """
    Every cashflow's value at time `t` on every state node, signed for the option holder:
    ORE's `CashflowInfo::pv`, in the model shifted by `shift`, `[nodes, C]`, fixed leg first,
    then floating, from the LGM's bond prices directly:

      * fixed coupon: `amount * P(t, pay; x)`;
      * Ibor coupon: `(fixing(t, x) + spread) * accrual * notional * P(t, pay; x)`, with
        `LgmVectorised::fixing` projecting over the index period [d1, d2] on the index's
        forwarding curve: `(P_idx(t,T1)/P_idx(t,T2) - 1) / dcf(d1, d2)`, `T1 = max(t, d1)`,
        `T2 = max(T1, d2)`; a known fixing its value, one fixed today the forecast off today's
        curve.
    """
    curve = swap.curve
    index_curve = swap.curve if swap.index_curve is None else swap.index_curve
    a, sigma = swap.reversion, swap.sigma
    fixed = (_bond_prices_at_nodes(curve, a, sigma, t, jnp.asarray(swap.fixed_times), x_nodes, shift)
             * jnp.asarray(swap.fixed_amounts)[None, :])
    T1 = jnp.maximum(t, jnp.asarray(swap.float_index_start_times))
    T2 = jnp.maximum(T1, jnp.asarray(swap.float_index_end_times))
    dcf = jnp.asarray(swap.float_index_dcf)
    projected = (_bond_prices_at_nodes(index_curve, a, sigma, t, T1, x_nodes, shift)
                 / _bond_prices_at_nodes(index_curve, a, sigma, t, T2, x_nodes, shift) - 1.0) / dcf
    zero = jnp.zeros(())
    start, end = jnp.asarray(swap.float_index_start_times), jnp.asarray(swap.float_index_end_times)
    today = (bond_price(index_curve, a, sigma, zero, start, zero, shift)
             / bond_price(index_curve, a, sigma, zero, end, zero, shift) - 1.0) / dcf
    fixing = jnp.where(jnp.asarray(swap.float_fixed_today)[None, :], today[None, :], projected)
    fixing = jnp.where(jnp.asarray(swap.float_is_known)[None, :], jnp.asarray(swap.float_known_rates)[None, :], fixing)
    floating = (swap.notional * (fixing + swap.float_spread) * jnp.asarray(swap.float_accrual)[None, :]
                * _bond_prices_at_nodes(curve, a, sigma, t, jnp.asarray(swap.float_pay_times), x_nodes, shift))
    sign = 1.0 if swap.payer else -1.0
    return jnp.concatenate([-sign * fixed, sign * floating], axis=1)


def single_exercise_value_by_integration(cfg: BermudanSwaptionConfig, **model) -> float:
    """`model` as `tests.support.lgm_engine.prepared`."""
    swap = prepared(cfg, **model)
    assert swap.exercise_times.shape == (1,), "only defined for a single exercise date"
    t = float(swap.exercise_times[0])
    shift = horizon_shift(swap.reversion, swap.shift_horizon)
    belongs = np.concatenate([swap.fixed_belongs_until, swap.float_belongs_until]) >= t

    z = np.linspace(-12.0, 12.0, 240_001)
    x = jnp.asarray(z * np.sqrt(float(_zeta(swap.sigma, t))))
    exercised = np.asarray(cashflow_values_at_nodes(swap, x, jnp.asarray(t), shift))[:, belongs].sum(axis=1)
    deflated = np.maximum(exercised, 0.0) / np.asarray(
        _numeraire(swap.curve, swap.reversion, swap.sigma, jnp.asarray(t), x, shift))
    return float(np.trapezoid(deflated * scipy_norm.pdf(z), z))
