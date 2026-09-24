"""
Independent reference values for single-exercise Bermudan swaptions.

With one exercise date `T0`, a Bermudan's value is exactly

    V(0) = E[ max(U(T0, x), 0) / N(T0, x) ],   x ~ Normal(0, zeta(T0)),

where `U` is the value of the swap exercised into and `N` the LGM numeraire
(`N(0, 0) = 1`). `single_exercise_value_by_integration` computes that
expectation by a fine direct integral over x: no Hagan grid, no convolution
rollback and none of ORE's cashflow bookkeeping, so agreement tests exactly
the machinery the backward induction adds. The cashflow values are the
engine's own (`_cashflow_values_at_nodes`), pinned separately against ORE in
tests/test_ore_lgm_parity.py.

This replaced a Jamshidian decomposition, which needs the floating leg to
telescope to `notional * (P(T_start) - P(T_end))`. It does not once each
coupon is projected over its index fixing period, as ORE projects it (I-31).
"""
import jax.numpy as jnp
import numpy as np
from scipy.stats import norm as scipy_norm

from engine.instruments.bermudan_swaption import (
    BermudanSwaptionConfig,
    _cashflow_values_at_nodes,
    _zero_curve_of,
    prepare_bermudan,
)
from engine.models.lgm import numeraire as _numeraire, zeta as _zeta


def single_exercise_value_by_integration(cfg: BermudanSwaptionConfig) -> float:
    swap = prepare_bermudan(cfg)
    assert swap.exercise_times.shape == (1,), "only defined for a single exercise date"
    t = float(swap.exercise_times[0])
    curve = _zero_curve_of(swap)
    belongs = np.concatenate([swap.fixed_belongs_until, swap.float_belongs_until]) >= t

    z = np.linspace(-12.0, 12.0, 240_001)
    x = jnp.asarray(z * np.sqrt(float(_zeta(cfg.hw_sigma, t))))
    exercised = np.asarray(_cashflow_values_at_nodes(swap, curve, x, jnp.asarray(t)))[:, belongs].sum(axis=1)
    deflated = np.maximum(exercised, 0.0) / np.asarray(_numeraire(curve, cfg.hw_a, cfg.hw_sigma, jnp.asarray(t), x))
    return float(np.trapezoid(deflated * scipy_norm.pdf(z), z))
