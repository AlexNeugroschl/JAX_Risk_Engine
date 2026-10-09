"""
The state paths of the cross-asset model (`engine.models.cam`): the exact discretization's
recursion `x_{i+1} = M_i x_i + b_i + L_i z_i`, run on the device on every path at once.

The step moments `(M_i, b_i, L_i)` are the model's (`engine.models.cam.step_moments`, computed
once on the host in float64); `engine.market_simulation.config.simulate` casts them to the
simulation's compute format and feeds the bridged Sobol normals through here.
"""
from typing import Tuple

import jax
import jax.numpy as jnp

from engine.precision import matmul


@jax.jit
def evolve_states(x0: jax.Array, moments: Tuple[jax.Array, jax.Array, jax.Array], normals: jax.Array) -> jax.Array:
    """States `[S, T, d]` at `times[1:]` from independent standard normals `[T, S, d]`:
    `x_{i+1} = M_i x_i + b_i + L_i z_i` (ORE's `StochasticProcess::evolve` with the exact
    discretization). `moments` is `(transition, drift, cholesky)` in the normals' dtype."""
    transition, drift, cholesky = moments

    def step(x, inputs):
        M, b, L, z = inputs
        x_next = matmul(x, M.T) + b + matmul(z, L.T)
        return x_next, x_next

    initial = jnp.broadcast_to(x0, (normals.shape[1], x0.shape[0]))
    _, states = jax.lax.scan(step, initial, (transition, drift, cholesky, normals))
    return jnp.transpose(states, (1, 0, 2))
