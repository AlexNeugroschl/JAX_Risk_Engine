"""
The state paths of the cross-asset model (`engine.models.cam`): the exact discretization's
recursion `x_{i+1} = M_i x_i + b_i + L_i z_i`, run on the device on every path at once, in
difference form (F-07; docs/planning/details/precision.md §8.2).

The step moments `(M_i, b_i, L_i)` and the path moments (the mean path `m_i` and the scale
`s_i`, a power of two per component near its standard deviation) are the model's
(`engine.models.cam.step_moments`, `path_moments`), computed once on the host in float64. A
path is its mean plus a deviation in units of the scale, `x_i = m_i + s_i * y_i`, and the
deviation follows

    y_{i+1} = M~_i y_i + L~_i z_i,   M~_i = S_{i+1}^-1 M_i S_i,   L~_i = S_{i+1}^-1 L_i,

`S_i = diag(s_i)`, the drift and the mean having cancelled. Each step is one matrix product,
`[y_i, z_i] @ [M~_i, L~_i]'`, in the simulation stage's compute format; the deviation is its
accumulator, carried in the accumulate format; the state comes out as `m_i + s_i y_i` in that
format, the mean (float64) added in two terms of that format (`engine.precision.add_level`), so
no rounding of the mean is shared by every path. Everything multiplied is near 1: a path's
deviation in units of its own scale, the normals, and the standardized moments (a scale's
ratio, a correlation).
"""
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from engine.models.cam import PathMoments, StepMoments
from engine.precision import Arithmetic, add_level


def standardized_steps(moments: StepMoments, path: PathMoments) -> np.ndarray:
    """`[T, 2d, d]`: each step's `[M~_i, L~_i]'` (see the module docstring), float64. Exact:
    the scales are powers of two."""
    inverse = 1.0 / path.scale[1:, :, None]                                    # S_{i+1}^-1, [T, d, 1]
    transition = inverse * moments.transition * path.scale[:-1, None, :]     # S_{i+1}^-1 M_i S_i
    noise = inverse * moments.cholesky                                       # S_{i+1}^-1 L_i
    return np.concatenate([np.swapaxes(transition, 1, 2), np.swapaxes(noise, 1, 2)], axis=1)


@partial(jax.jit, static_argnames="arithmetic")
def evolve_states(path: Tuple[jax.Array, jax.Array], steps: jax.Array, normals: jax.Array,
                  arithmetic: Arithmetic) -> jax.Array:
    """States `[S, T, d]` at `times[1:]` from independent standard normals `[T, S, d]`
    (ORE's `StochasticProcess::evolve` with the exact discretization), in `arithmetic`'s
    accumulate format: `path` is the mean and the scale `[T+1, d]` (`PathMoments`), `steps` the
    standardized moments `[T, 2d, d]` (`standardized_steps`)."""
    mean, scale = jnp.asarray(path[0], dtype=jnp.float64), arithmetic.cast(path[1])
    normals = arithmetic.cast(normals)

    def step(deviation, inputs):
        moments, z = inputs
        deviation = arithmetic.matmul(jnp.concatenate([deviation, z], axis=-1), moments)
        return deviation, deviation

    start = jnp.zeros(normals.shape[1:], dtype=arithmetic.dtype)
    _, deviations = jax.lax.scan(step, start, (steps, normals))               # [T, S, d]
    states = add_level(mean[1:, None, :], scale[1:, None, :] * deviations)
    return jnp.transpose(states, (1, 0, 2))
