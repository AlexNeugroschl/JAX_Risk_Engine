"""
Quasi-Monte Carlo normals: scrambled Sobol points mapped to standard normals, and a Brownian
bridge over the simulation grid (QuantLib's `BrownianBridge` construction order).

Model-independent; shared by the cross-asset simulation (`engine.simulation.cam`), the
Hull-White simulation (`engine.simulation.market_model`) and the market-risk scenario
generator (`engine.market_risk.scenarios`).
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm
from scipy.stats.qmc import Sobol


def generate_sobol_normals(num_scenarios: int, num_steps: int, num_assets: int, dtype, seed: int = 42) -> jax.Array:
    """
    Scrambled Sobol uniforms (CPU) mapped to standard normals `[TimeSteps, Scenarios,
    Assets]`.

    `seed` selects the Owen scrambling: the same seed reproduces the draw; different seeds
    are independent randomized-QMC replicates. The output is cast to `dtype`, because
    `norm.ppf` computes in float64 whenever x64 is on.
    """
    total_dimensions = num_steps * num_assets

    # Scrambling randomizes the otherwise deterministic Sobol points.
    sobol_engine = Sobol(d=total_dimensions, scramble=True, seed=seed)
    uniform_draws = sobol_engine.random(n=num_scenarios)

    # The JAX half is jitted: eagerly, each op compiled separately and dominated the
    # function's cost.
    return _sobol_uniforms_to_normals(uniform_draws, num_scenarios, num_steps, num_assets, dtype)


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def _sobol_uniforms_to_normals(uniform_draws, num_scenarios: int, num_steps: int, num_assets: int, dtype) -> jax.Array:
    """Clip the uniforms off {0, 1}, map through the inverse normal CDF, and reshape to
    `[TimeSteps, Scenarios, Assets]`. Everything but `uniform_draws` is static."""
    uniform_jax = jnp.asarray(uniform_draws, dtype=dtype)
    epsilon = jnp.finfo(dtype).eps
    uniform_clipped = jnp.clip(uniform_jax, epsilon, 1.0 - epsilon)

    normal_shocks = norm.ppf(uniform_clipped).astype(dtype)

    # [Scenarios, Steps, Assets] -> [Steps, Scenarios, Assets]
    Z = normal_shocks.reshape((num_scenarios, num_steps, num_assets))
    return jnp.transpose(Z, (1, 0, 2))

def _build_bridge_matrix(time_grid: np.ndarray) -> np.ndarray:
    """
    Brownian-bridge construction matrix B with W(t_1..t_n) = B @ Z, where Z are independent
    normals ordered by Sobol dimension (dimension 0 is the endpoint, 1 the midpoint, ...),
    by the recursive bisection of QuantLib's `BrownianBridge`.
    """
    times = np.asarray(time_grid, dtype=np.float64)[1:]  # drop t=0
    n = times.shape[0]
    B = np.zeros((n, n), dtype=np.float64)

    left_index = np.zeros(n, dtype=np.int64)
    right_index = np.zeros(n, dtype=np.int64)
    bridge_index = np.zeros(n, dtype=np.int64)
    left_weight = np.zeros(n, dtype=np.float64)
    right_weight = np.zeros(n, dtype=np.float64)
    std_dev = np.zeros(n, dtype=np.float64)

    # map_[k] != 0 marks slot k as already constructed; each iteration bisects the widest
    # remaining gap (as QuantLib's BrownianBridge).
    map_ = np.zeros(n, dtype=np.int64)
    map_[n - 1] = 1
    bridge_index[0] = n - 1
    std_dev[0] = np.sqrt(times[-1])

    for i in range(1, n):
        j = 0
        while map_[j] != 0:
            j += 1
        k = j
        while map_[k] == 0:
            k += 1
        # Midpoint between the known bounds j-1 and k.
        l = j + ((k - 1 - j) // 2)
        map_[l] = i

        left_index[i] = j
        right_index[i] = k
        bridge_index[i] = l

        left_t = times[j - 1] if j != 0 else 0.0
        right_t = times[k]
        mid_t = times[l]

        left_weight[i] = (right_t - mid_t) / (right_t - left_t)
        right_weight[i] = (mid_t - left_t) / (right_t - left_t)
        std_dev[i] = np.sqrt(
            (mid_t - left_t) * (right_t - mid_t) / (right_t - left_t)
        )

    # Turn the recursion into an explicit linear map from Z to W.
    B[n - 1, 0] = std_dev[0]
    for i in range(1, n):
        row = bridge_index[i]
        B[row, i] += std_dev[i]
        if left_index[i] != 0:
            B[row, :] += left_weight[i] * B[left_index[i] - 1, :]
        if right_index[i] != n:
            B[row, :] += right_weight[i] * B[right_index[i], :]

    return B


@jax.jit
def _apply_bridge_matrix(B_matrix: jax.Array, Z: jax.Array) -> jax.Array:
    return jnp.tensordot(B_matrix, Z, axes=([1], [0]))


def apply_brownian_bridge(Z: jax.Array, time_grid: jax.Array) -> jax.Array:
    """Independent normals `[TimeSteps, Scenarios, Assets]` -> bridged increments,
    standardized by sqrt(dt)."""
    num_steps, num_scenarios, num_assets = Z.shape

    B_matrix_np = _build_bridge_matrix(np.asarray(time_grid))
    B_matrix = jnp.asarray(B_matrix_np, dtype=Z.dtype)

    # Path values W(t_i), variance ordered by the bridge.
    W_paths = _apply_bridge_matrix(B_matrix, Z)
    
    # Path values -> increments dW.
    W_paths_with_zero = jnp.concatenate(
        [jnp.zeros((1, num_scenarios, num_assets), dtype=Z.dtype), W_paths], 
        axis=0
    )
    dW = jnp.diff(W_paths_with_zero, axis=0)
    
    # Standardize the increments to N(0, 1) per step.
    dt = jnp.diff(time_grid)
    Z_sequential = dW / jnp.sqrt(dt)[:, None, None]
    
    return Z_sequential
