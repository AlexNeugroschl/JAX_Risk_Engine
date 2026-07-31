import jax
# Enforce 64-bit precision for financial stability
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.stats import norm
from scipy.stats.qmc import Sobol
from typing import Dict, Any

# =============================================================================
# PHASE 1: QUASI-MONTE CARLO (CPU -> GPU)
# =============================================================================
def generate_sobol_normals(num_scenarios: int, num_steps: int, num_assets: int, dtype) -> jax.Array:
    """
    CPU: Generates Sobol sequences.
    GPU: Converts to Normal shocks.
    Returns: [TimeSteps, Scenarios, Assets]
    """
    total_dimensions = num_steps * num_assets
    
    # Scramble adds necessary randomness to the deterministic Sobol points
    sobol_engine = Sobol(d=total_dimensions, scramble=True, seed=42)
    uniform_draws = sobol_engine.random(n=num_scenarios)
    
    # Transfer to JAX and convert to standard normals
    uniform_jax = jnp.array(uniform_draws, dtype=dtype)
    epsilon = jnp.finfo(dtype).eps
    uniform_clipped = jnp.clip(uniform_jax, epsilon, 1.0 - epsilon)
    
    normal_shocks = norm.ppf(uniform_clipped)
    
    # Reshape to match JAX loop expectations
    Z = normal_shocks.reshape((num_scenarios, num_steps, num_assets))
    return jnp.transpose(Z, (1, 0, 2))

@jax.jit
def apply_brownian_bridge(Z: jax.Array, time_grid: jax.Array) -> jax.Array:
    """
    Transforms independent Normal shocks into Bridged Chronological Shocks.
    """
    num_steps, num_scenarios, num_assets = Z.shape
    
    # Placeholder: In production, replace jnp.eye with ORE's extracted Bridge Matrix
    B_matrix = jnp.eye(num_steps, dtype=Z.dtype)
    
    # Apply Bridge Matrix to reorder variance
    W_paths = jnp.tensordot(B_matrix, Z, axes=([1], [0]))
    
    # Convert absolute paths back to sequential steps (dW)
    W_paths_with_zero = jnp.concatenate(
        [jnp.zeros((1, num_scenarios, num_assets), dtype=Z.dtype), W_paths], 
        axis=0
    )
    dW = jnp.diff(W_paths_with_zero, axis=0)
    
    # Standardize increments for the JAX step function
    dt = jnp.diff(time_grid)
    Z_sequential = dW / jnp.sqrt(dt)[:, None, None]
    
    return Z_sequential


# =============================================================================
# PHASE 2: CROSS-ASSET MODEL ENGINE (GPU)
# =============================================================================
@jax.jit
def _simulate_cross_asset_paths_jit(
    eq_S0: jax.Array, eq_div_t: jax.Array, rate_mapping: jax.Array, eq_sigma_t: jax.Array,
    hw_r0: jax.Array, hw_theta_t: jax.Array, hw_sigma_t: jax.Array, hw_a: jax.Array,
    L_t: jax.Array, dt_t: jax.Array, Z_bridged: jax.Array
):
    num_scenarios = Z_bridged.shape[1]
    num_eq = eq_S0.shape[0]
    num_hw = hw_r0.shape[0]
    compute_dtype = eq_S0.dtype

    def step_fn(state, step_inputs):
        eq_t, r_t, N_t = state
        Z_i, L_i, dt_i, div_eq, sig_eq, theta_hw, sig_hw = step_inputs
        
        # Correlate all assets
        Z_corr = jnp.dot(Z_i, L_i.T)
        Z_eq = Z_corr[:, :num_eq]
        Z_hw = Z_corr[:, num_eq:]
        
        # 1. HW1F (Interest Rates)
        decay = jnp.exp(-hw_a * dt_i)
        variance_hw = (1.0 - jnp.exp(-2.0 * hw_a * dt_i)) / (2.0 * hw_a)
        shock_hw = sig_hw * jnp.sqrt(variance_hw) * Z_hw
        r_next = r_t * decay + theta_hw + shock_hw
        
        # 2. GBM (Equities / FX) with Dynamic Drift (Uncovered Interest Rate Parity)
        dynamic_mu = jnp.dot(r_t, rate_mapping.T) - div_eq
        drift_eq = (dynamic_mu - 0.5 * sig_eq**2) * dt_i
        shock_eq = sig_eq * jnp.sqrt(dt_i) * Z_eq
        eq_next = eq_t * jnp.exp(drift_eq + shock_eq)
        
        # 3. Update Numéraire (Using Index 0 as base discount curve)
        N_next = N_t * jnp.exp(r_t[:, 0] * dt_i)
        
        new_state = (eq_next, r_next, N_next)
        return new_state, new_state

    # Initialize Day-0 States
    eq_initial = jnp.broadcast_to(eq_S0, (num_scenarios, num_eq))
    hw_initial = jnp.broadcast_to(hw_r0, (num_scenarios, num_hw))
    N_initial = jnp.ones((num_scenarios,), dtype=compute_dtype) 

    # Execute Time Machine
    _, (eq_paths, hw_paths, N_paths) = jax.lax.scan(
        step_fn, 
        (eq_initial, hw_initial, N_initial), 
        (Z_bridged, L_t, dt_t, eq_div_t, eq_sigma_t, hw_theta_t, hw_sigma_t)
    )

    return (
        jnp.transpose(eq_paths, (1, 0, 2)), 
        jnp.transpose(hw_paths, (1, 0, 2)), 
        jnp.transpose(N_paths, (1, 0))
    )


# =============================================================================
# PHASE 3: YIELD CURVE RECONSTRUCTION
# =============================================================================
@jax.jit
def reconstruct_yield_curves(hw_paths: jax.Array, A: jax.Array, B: jax.Array) -> jax.Array:
    """
    Expands simulated 1D short rates into 2D discount curves for future pricing.
    Outputs: [Scenarios, TimeSteps, Maturities, NumRates]
    """
    r_t = hw_paths[:, :, None, :]  
    A_bcast = A[None, :, :, :]     
    B_bcast = B[None, :, :, :]     
    
    # Vectorized Hull-White Affine Formula
    discount_curves = A_bcast * jnp.exp(-B_bcast * r_t)
    return discount_curves


# =============================================================================
# PHASE 4: PUBLIC API WRAPPER
# =============================================================================
def generate_paths(config: Dict[str, Any], precision: int = 64) -> Dict[str, jax.Array]:
    jax.config.update("jax_enable_x64", precision == 64)
    dtype = jnp.float64 if precision == 64 else jnp.float32

    # 1. Base Setup
    time_grid = jnp.array(config["time_grid"], dtype=dtype)
    dt_t = jnp.diff(time_grid)
    num_steps = dt_t.shape[0]
    num_scenarios = int(config.get("scenarios", 10000))

    # 2. Equities
    eq_cfg = config["equities"]
    eq_S0 = jnp.array(eq_cfg["initial_prices"], dtype=dtype)
    eq_div_t = jnp.tile(jnp.array(eq_cfg["dividend_yields"], dtype=dtype), (num_steps, 1))
    rate_mapping = jnp.array(eq_cfg["rate_mapping"], dtype=dtype)
    num_eq = eq_S0.shape[0]
    
    # 3. Rates
    hw_cfg = config["rates"]
    hw_r0 = jnp.array(hw_cfg["initial_rates"], dtype=dtype)
    hw_theta_t = jnp.tile(jnp.array(hw_cfg["theta"], dtype=dtype), (num_steps, 1))
    hw_a = jnp.array(hw_cfg["mean_reversion"], dtype=dtype)
    num_hw = hw_r0.shape[0]

    # 4. Joint Matrix
    cov_raw = jnp.array(config["joint_covariance"], dtype=dtype)
    cov_t = jnp.tile(cov_raw[None, :, :], (num_steps, 1, 1))
    L_t = jnp.linalg.cholesky(cov_t)
    joint_sigma_t = jnp.sqrt(jnp.diagonal(cov_t, axis1=1, axis2=2))
    eq_sigma_t = joint_sigma_t[:, :num_eq]
    hw_sigma_t = joint_sigma_t[:, num_eq:]

    # 5. Core Simulation Pipeline
    Z_sobol = generate_sobol_normals(num_scenarios, num_steps, num_eq + num_hw, dtype)
    Z_bridged = apply_brownian_bridge(Z_sobol, time_grid)
    
    eq_paths, hw_paths, numeraire_paths = _simulate_cross_asset_paths_jit(
        eq_S0, eq_div_t, rate_mapping, eq_sigma_t, 
        hw_r0, hw_theta_t, hw_sigma_t, hw_a, 
        L_t, dt_t, Z_bridged
    )

    results = {
        "equities": eq_paths,
        "rates": hw_paths,
        "numeraire": numeraire_paths
    }

    # 6. Yield Curve Reconstruction (If requested in config)
    if "maturities" in hw_cfg:
        maturities = jnp.array(hw_cfg["maturities"], dtype=dtype)
        num_maturities = maturities.shape[0]
        
        # Generate Mock A and B Matrices for standalone execution
        A_matrix = jnp.ones((num_steps, num_maturities, num_hw), dtype=dtype)
        
        step_times = time_grid[1:] # We evaluate AT the step ends
        T_minus_t = jnp.maximum(maturities[None, :] - step_times[:, None], 0.0)
        
        # B(t,T) formula. Shape mapping: T_minus_t is [TimeSteps, Maturities]
        # hw_a is [NumRates]. We broadcast appropriately.
        hw_a_bcast = hw_a[None, None, :] 
        B_matrix = (1.0 - jnp.exp(-hw_a_bcast * T_minus_t[:, :, None])) / hw_a_bcast
        
        yield_cube = reconstruct_yield_curves(hw_paths, A_matrix, B_matrix)
        results["yield_curves"] = yield_cube

    return results


# =============================================================================
# EXECUTION DEMONSTRATION
# =============================================================================
if __name__ == "__main__":
    payload = {
        "time_grid": [0.0, 0.25, 0.50, 0.75, 1.0], 
        "scenarios": 4096,                         
        "equities": {
            "initial_prices": [150.0, 1.10],       # AAPL, EUR/USD
            "dividend_yields": [0.01, 0.00],       
            "rate_mapping": [
                [1.0, 0.0],                        # AAPL relies purely on USD rate (index 0)
                [1.0, -1.0]                        # EUR/USD relies on USD - EUR
            ]
        },
        "rates": {
            "initial_rates": [0.03, 0.02],         # USD SOFR, EURIBOR
            "theta": [0.03, 0.02],
            "mean_reversion": [0.1, 0.15],
            "maturities": [1.0, 2.0, 5.0, 10.0]    # Output curves out to 10Y
        },
        "joint_covariance": [
            [0.0400, 0.0000, 0.0010, 0.0005],  # AAPL
            [0.0000, 0.0100, 0.0002, -0.0001], # EUR/USD
            [0.0010, 0.0002, 0.0001, 0.00008], # USD SOFR
            [0.0005, -0.0001, 0.00008, 0.0002] # EURIBOR
        ]
    }

    print("Initializing QMC Pipeline & JIT Compilation...")
    market_cubes = generate_paths(payload)

    print("\n--- Base Tensors ---")
    print(f"Equities/FX:  {market_cubes['equities'].shape}")
    print(f"Rates:        {market_cubes['rates'].shape}")
    print(f"Numéraire:    {market_cubes['numeraire'].shape}")
    
    print("\n--- 4D Yield Curve Matrix ---")
    print(f"Yield Curves: {market_cubes['yield_curves'].shape}")
    
    print("\n[Sample] Scenario 0, Step 1 (t=0.25), USD Discount Factors:")
    print(f"To Year 1:  {market_cubes['yield_curves'][0, 0, 0, 0]:.4f}")
    print(f"To Year 2:  {market_cubes['yield_curves'][0, 0, 1, 0]:.4f}")
    print(f"To Year 5:  {market_cubes['yield_curves'][0, 0, 2, 0]:.4f}")
    print(f"To Year 10: {market_cubes['yield_curves'][0, 0, 3, 0]:.4f}")