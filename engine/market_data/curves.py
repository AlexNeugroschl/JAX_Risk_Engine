"""
Yield-curve primitives shared by every pricer and model.

`ZeroCurve` is today's curve as continuously compounded zero rates at pillar times, linear
in the zero rate, as QuantLib's `InterpolatedZeroCurve<Linear>` (ORE's `ZeroCurve`):

  * between pillars: linear in the zero rate;
  * past the last pillar: a flat instantaneous forward, QuantLib's default
    `Extrapolation::ContinuousForward` (ql/termstructures/yield/zerocurve.hpp,
    `zeroYieldImpl`):
        z(t) = (z_n t_n + f_n (t - t_n)) / t,   f_n = z_n + t_n * slope of the last segment
    so ln P(0, t) = -(z_n t_n + f_n (t - t_n)) is linear in t there (I-48);
  * before the first pillar: flat. QuantLib curves start at the reference date, so a
    curve whose first pillar is t = 0 never uses this branch.

Greeks differentiate with respect to `pillar_rates`, never `pillar_times`, as ORE's
sensitivity framework bumps rates, not times.

`DiscountCurve` is a curve on a simulated path: discount factors at the simulation-market
tenors, log-linear between them with flat-forward extrapolation, as the scenario market's
`QuantExt::InterpolatedDiscountCurve` (ORE's `ScenarioSimMarketParameters` defaults
`Interpolation = LogLinear`, `Extrapolation = FlatFwd`). Its values may carry leading batch
axes (paths), sharing one tenor grid.
"""
from dataclasses import dataclass
from typing import List

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
@dataclass
class ZeroCurve:
    """Continuously compounded zero rates at pillar times (year fractions). A pytree, so it
    passes through `jax.jit`/`jax.grad` boundaries."""
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
        """From anything with `times` and `rates` (e.g. `ZeroCurveConfig`)."""
        return ZeroCurve(
            pillar_times=jnp.asarray(config.times, dtype=dtype),
            pillar_rates=jnp.asarray(config.rates, dtype=dtype),
        )

    @staticmethod
    def flat(rate: float, pillar_times: List[float], dtype=jnp.float64) -> "ZeroCurve":
        """A curve with the same rate at every pillar."""
        return ZeroCurve(
            pillar_times=jnp.asarray(pillar_times, dtype=dtype),
            pillar_rates=jnp.full((len(pillar_times),), rate, dtype=dtype),
        )


def _segment_slopes(curve: ZeroCurve, t: jax.Array) -> jax.Array:
    """Slope of the linear segment starting at or before `t` (a pillar takes the slope of
    the segment it starts); the last segment's slope at and beyond the last pillar."""
    times, rates = curve.pillar_times, curve.pillar_rates
    n = times.shape[0]
    if n < 2:
        return jnp.zeros_like(t)
    right = jnp.clip(jnp.searchsorted(times, t, side="right"), 1, n - 1)
    left = right - 1
    return (rates[right] - rates[left]) / (times[right] - times[left])


def _last_forward(curve: ZeroCurve) -> jax.Array:
    """f_n = z_n + t_n * (slope of the last segment): the instantaneous forward at the last
    pillar, held flat beyond it."""
    t_n, z_n = curve.pillar_times[-1], curve.pillar_rates[-1]
    return z_n + t_n * _segment_slopes(curve, t_n)


def zero_rate(curve: ZeroCurve, t: jax.Array) -> jax.Array:
    """z(t) (see the module docstring for the three regions)."""
    t = jnp.asarray(t)
    t_n, z_n = curve.pillar_times[-1], curve.pillar_rates[-1]
    beyond = t > t_n
    # Both branches are evaluated under jit/grad; divide by a safe t off the region.
    t_safe = jnp.where(beyond, t, jnp.ones_like(t))
    extrapolated = (z_n * t_n + _last_forward(curve) * (t - t_n)) / t_safe
    return jnp.where(beyond, extrapolated, jnp.interp(t, curve.pillar_times, curve.pillar_rates))


def log_discount(curve, t: jax.Array) -> jax.Array:
    """ln P(0,t): -z(t) t for a `ZeroCurve`; the log-linear interpolant for a
    `DiscountCurve` (whose batch axes, if any, lead the result)."""
    if isinstance(curve, DiscountCurve):
        return loglinear_log_discount(curve, t)
    return -zero_rate(curve, t) * t


def discount(curve, t: jax.Array) -> jax.Array:
    """P(0,t) = exp(ln P(0,t)), for either curve type."""
    return jnp.exp(log_discount(curve, t))


def forward_rate(curve: ZeroCurve, t: jax.Array) -> jax.Array:
    """Instantaneous forward f(0,t) = -d/dt ln P(0,t), computed exactly from the
    interpolant: z(t) + t z'(t) inside the pillars, z_0 before the first one, and the flat
    f_n from the last pillar on (continuous there).

    Exact rather than a finite difference, which is cancellation noise in float32 (forward
    errors of up to ~2 percentage points)."""
    t = jnp.asarray(t)
    times = curve.pillar_times
    inside = (t >= times[0]) & (t < times[-1])
    slope = jnp.where(inside, _segment_slopes(curve, t), jnp.zeros_like(t))
    within = zero_rate(curve, t) + t * slope
    return jnp.where(t >= times[-1], _last_forward(curve), within)


@jax.tree_util.register_pytree_node_class
@dataclass
class DiscountCurve:
    """Log discount factors `log_discounts[..., k]` at `times[k]`, with `times[0] == 0` and
    `log_discounts[..., 0] == 0`. Leading axes of `log_discounts` are batch axes (e.g.
    paths) that share the tenor grid."""
    times: jax.Array          # [K]
    log_discounts: jax.Array  # [..., K]

    def tree_flatten(self):
        return (self.times, self.log_discounts), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        times, log_discounts = children
        return cls(times=times, log_discounts=log_discounts)


def loglinear_log_discount(curve: DiscountCurve, t: jax.Array) -> jax.Array:
    """ln P(t) for query times `t`, as `QuantExt::InterpolatedDiscountCurve` (`discountImpl`,
    logLinear, flatFwd): linear in ln P between tenors, and the last segment continued
    linearly (a flat forward) past the last tenor. Returns the curve's batch axes followed
    by `t`'s shape."""
    times = curve.times
    t = jnp.asarray(t, dtype=curve.log_discounts.dtype)
    # std::upper_bound, clamped to the last segment (which then extrapolates).
    right = jnp.clip(jnp.searchsorted(times, t, side="right"), 1, times.shape[0] - 1)
    left = right - 1
    weight = (times[right] - t) / (times[right] - times[left])
    values = curve.log_discounts
    return (1.0 - weight) * values[..., right] + weight * values[..., left]


def node_values(curve) -> jax.Array:
    """The values a curve's log discount factors are linear in: a `DiscountCurve`'s log
    discount factors, a `ZeroCurve`'s pillar rates (`log_discount_deviation`)."""
    return curve.log_discounts if isinstance(curve, DiscountCurve) else curve.pillar_rates


def log_discount_deviation(curve, deviation: jax.Array, t: jax.Array) -> jax.Array:
    """`[..., *t.shape]`: how far ln P(t) moves when the curve's node values (`node_values`)
    move by `deviation` `[..., K]` (any batch axes leading it). ln P is linear in the node values
    for both curve types (a `DiscountCurve` is log-linear between its tenors and continues its
    last segment; a `ZeroCurve` is linear in the zero rate, flat before its first pillar, and its
    flat-forward extrapolation is linear in its last two rates), so this is the curve's own
    two-node interpolation applied to the deviations: a gather and two multiplications per time,
    in the deviation's dtype. A path's deviation of the curve from a level is carried to every
    flow this way (F-07's difference form, `engine.pricing.legs`)."""
    left, right, w_left, w_right = _stencil(curve, jnp.asarray(t, dtype=deviation.dtype))
    return deviation[..., left] * w_left + deviation[..., right] * w_right


def _stencil(curve, t):
    """ln P(t) as `values[left] * w_left + values[right] * w_right` of the curve's node values
    (see `log_discount_deviation`): the two nodes and their weights, each of `t`'s shape. A
    `DiscountCurve`'s are `loglinear_log_discount`'s; a `ZeroCurve`'s are `-t` times
    `zero_rate`'s, with its extrapolation's coefficients past the last pillar."""
    times = (curve.times if isinstance(curve, DiscountCurve) else curve.pillar_times).astype(t.dtype)
    n = times.shape[0]
    if n < 2:
        zero = jnp.zeros(t.shape, dtype=jnp.int32)
        return zero, zero, jnp.zeros_like(t), (-t if isinstance(curve, ZeroCurve) else jnp.ones_like(t))
    right = jnp.clip(jnp.searchsorted(times, t, side="right"), 1, n - 1)
    left = right - 1
    if isinstance(curve, DiscountCurve):
        weight = (times[right] - t) / (times[right] - times[left])
        return left, right, weight, 1.0 - weight
    share = jnp.clip((t - times[left]) / (times[right] - times[left]), 0.0, 1.0)
    t_n, gap = times[-1], times[-1] - times[-2]
    past = t - t_n
    beyond = t > t_n
    return (jnp.where(beyond, n - 2, left), jnp.where(beyond, n - 1, right),
            jnp.where(beyond, past * t_n / gap, -t * (1.0 - share)),
            jnp.where(beyond, -(t_n + past * (1.0 + t_n / gap)), -t * share))


def curve_dtype(curve):
    """The working dtype of a `ZeroCurve` or a `DiscountCurve`: kernels that read a curve
    compute in it (docs/planning/details/precision.md §6.4)."""
    return (curve.log_discounts if isinstance(curve, DiscountCurve) else curve.pillar_rates).dtype
