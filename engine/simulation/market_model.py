"""
Legacy Hull-White Monte Carlo simulation: a joint Hull-White / lognormal cross-asset path
simulation, and a yield-curve cube rebuilt from the short rates.

Not the default. The portfolio path simulates ORE's cross-asset model instead
(`engine.simulation.cam`); this module is kept for the legacy `PortfolioRequest` shape and
for callers that want the Hull-White dynamics. The Sobol/Brownian-bridge generators now live
in `engine.simulation.random` and are re-exported here.

Differs from ORE's cross-asset model (CAM), which simulates LGM states fitted to each
currency's curve:
  * Each rate factor is a Hull-White short rate (not LGM; I-44) reverting to a constant,
    user-supplied `theta` from `initial_rates`, while the cube's A(t,T) is fitted to the
    input curve. The two are consistent only for a flat curve with theta and r(0) at its
    level, so the simulated discount factors do not reprice the input curve on a sloped
    one (I-42, audit M-1, 4-9% errors).
  * The numeraire is a money-market account accrued discretely, exp(r(t_i) dt), off rate
    factor 0 only (I-45).
  * Equities/FX are lognormal with drift `rate_mapping . r(t) - dividend_yield`.
  * Volatilities and correlations are constant, taken from `joint_covariance`.
"""
import jax
# jax_enable_x64 is process-global and must be on before any float64 array exists.
# `generate_paths` sets it per call and restores it afterwards (I-14); functions taking a
# `dtype` honour it regardless of the flag.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional

from engine.models.hull_white import (
    A as _hw_A, B as _hw_B, ZeroCurve as _HwZeroCurve, log_discount as _hw_log_discount,
)

# Phase 1: quasi-Monte Carlo normals (shared; re-exported for existing callers)
from engine.simulation.random import (  # noqa: E402,F401
    _apply_bridge_matrix,
    _build_bridge_matrix,
    _sobol_uniforms_to_normals,
    apply_brownian_bridge,
    generate_sobol_normals,
)


# Phase 2: cross-asset path simulation
@jax.jit
def _simulate_cross_asset_paths_jit(
    eq_S0: jax.Array, eq_div_t: jax.Array, rate_mapping: jax.Array, eq_sigma_t: jax.Array,
    hw_r0: jax.Array, hw_theta_t: jax.Array, hw_sigma_t: jax.Array, hw_a: jax.Array,
    L_t: jax.Array, dt_t: jax.Array, Z_bridged: jax.Array
):
    """
    Joint Hull-White (rates) and lognormal (equities/FX) paths, stepped with
    `jax.lax.scan`.

    Per step: correlate the bridged shocks with the Cholesky factor `L_t`; step each short
    rate with the exact Ornstein-Uhlenbeck transition toward the constant `theta` (see the
    module docstring); step each equity with drift `rate_mapping . r(t) - dividend`; accrue
    the numeraire off rate factor 0.

    Shapes: eq_S0 [NumEq], hw_r0 [NumHW]; eq_div_t/eq_sigma_t/hw_theta_t/hw_sigma_t
    [TimeSteps, NumEq or NumHW]; rate_mapping [NumEq, NumHW]; L_t [TimeSteps, NumEq+NumHW,
    NumEq+NumHW]; dt_t [TimeSteps]; Z_bridged [TimeSteps, Scenarios, NumEq+NumHW].
    Returns (eq_paths, hw_paths, numeraire_paths), each [Scenarios, TimeSteps, ...].
    """
    num_scenarios = Z_bridged.shape[1]
    num_eq = eq_S0.shape[0]
    num_hw = hw_r0.shape[0]
    compute_dtype = eq_S0.dtype

    def step_fn(state, step_inputs):
        eq_t, r_t, N_t = state
        Z_i, L_i, dt_i, div_eq, sig_eq, theta_hw, sig_hw = step_inputs
        
        # Correlate the shocks.
        Z_corr = jnp.dot(Z_i, L_i.T)
        Z_eq = Z_corr[:, :num_eq]
        Z_hw = Z_corr[:, num_eq:]
        
        # 1. Short rates: exact OU transition, mean r*decay + theta*(1-decay), variance
        #    sigma^2*(1-exp(-2a dt))/(2a). The a -> 0 limit (variance dt) is selected with
        #    jnp.where on a placeholder `a`, so a == 0 never divides by zero.
        hw_a_safe = jnp.where(hw_a == 0.0, 1.0, hw_a)
        decay = jnp.exp(-hw_a * dt_i)
        variance_hw = jnp.where(
            hw_a == 0.0,
            dt_i,
            (1.0 - jnp.exp(-2.0 * hw_a_safe * dt_i)) / (2.0 * hw_a_safe),
        )
        shock_hw = sig_hw * jnp.sqrt(variance_hw) * Z_hw
        r_next = r_t * decay + theta_hw * (1.0 - decay) + shock_hw
        
        # 2. Equities/FX: lognormal with drift rate_mapping . r(t) - dividend yield.
        dynamic_mu = jnp.dot(r_t, rate_mapping.T) - div_eq
        drift_eq = (dynamic_mu - 0.5 * sig_eq**2) * dt_i
        shock_eq = sig_eq * jnp.sqrt(dt_i) * Z_eq
        eq_next = eq_t * jnp.exp(drift_eq + shock_eq)
        
        # 3. Numeraire: money-market account on rate factor 0, left-point rule.
        N_next = N_t * jnp.exp(r_t[:, 0] * dt_i)
        
        new_state = (eq_next, r_next, N_next)
        return new_state, new_state

    # Day-0 state.
    eq_initial = jnp.broadcast_to(eq_S0, (num_scenarios, num_eq))
    hw_initial = jnp.broadcast_to(hw_r0, (num_scenarios, num_hw))
    N_initial = jnp.ones((num_scenarios,), dtype=compute_dtype) 

    # Step through time.
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


# Phase 3: yield-curve reconstruction
def _initial_log_discount(zero_times: np.ndarray, zero_rates: np.ndarray, t: np.ndarray) -> np.ndarray:
    """ln P(0,t): linear zero-rate interpolation, flat extrapolation
    (`engine.models.hull_white.log_discount` on NumPy arrays)."""
    curve = _HwZeroCurve(pillar_times=jnp.asarray(zero_times), pillar_rates=jnp.asarray(zero_rates))
    return np.asarray(_hw_log_discount(curve, jnp.asarray(t)))


def compute_hw_A_matrix(
    zero_curves: List["ZeroCurveConfig"],
    hw_a: np.ndarray,
    hw_sigma: np.ndarray,
    step_times: np.ndarray,
    maturities: np.ndarray,
    B_matrix: np.ndarray,
) -> np.ndarray:
    """
    Hull-White A(t,T) per rate factor, each fitted to that factor's own zero curve (as
    each currency in ORE's CAM has its own curve), via `engine.models.hull_white.A`.

    zero_curves: one `ZeroCurveConfig` per rate factor, in NumRates order.
    Shapes: step_times [TimeSteps], maturities [Maturities],
    B_matrix [TimeSteps, Maturities, NumRates] -> A [TimeSteps, Maturities, NumRates].
    Returns a NumPy array.
    """
    num_hw = hw_a.shape[0]
    step_times_j = jnp.asarray(step_times, dtype=jnp.float64)
    maturities_j = jnp.asarray(maturities, dtype=jnp.float64)
    t_grid = step_times_j[:, None]        # [TimeSteps, 1]
    T_grid = maturities_j[None, :]        # [1, Maturities]
    B_matrix_j = jnp.asarray(B_matrix, dtype=jnp.float64)

    A_per_factor = []
    for k in range(num_hw):
        curve = _HwZeroCurve(
            pillar_times=jnp.asarray(zero_curves[k].times, dtype=jnp.float64),
            pillar_rates=jnp.asarray(zero_curves[k].rates, dtype=jnp.float64),
        )
        # Pass the caller's B (clamped at 0 for past pillars) so A and exp(-B*r) agree
        # (see hull_white.A).
        A_k = _hw_A(
            curve, t_grid, T_grid, float(hw_a[k]), float(hw_sigma[k]),
            B_override=B_matrix_j[:, :, k],
        )  # [TimeSteps, Maturities]
        A_per_factor.append(A_k)
    return np.asarray(jnp.stack(A_per_factor, axis=-1))


def validate_joint_covariance(matrix) -> None:
    """
    Reject a `joint_covariance` that is not square, symmetric (within 1e-8) or positive
    semi-definite (eigenvalues >= -1e-8). Otherwise Cholesky returns NaN and every path
    becomes NaN without an error. The message names the offending eigenvalues.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"joint_covariance must be a square 2D matrix; got shape {m.shape}")

    tol = 1e-8
    asymmetry = np.abs(m - m.T)
    if np.any(asymmetry > tol):
        i, j = np.unravel_index(np.argmax(asymmetry), asymmetry.shape)
        raise ValueError(
            f"joint_covariance must be symmetric (within {tol}); largest "
            f"asymmetry {asymmetry[i, j]:.3e} at ({i}, {j}): "
            f"matrix[{i}][{j}]={m[i, j]} vs matrix[{j}][{i}]={m[j, i]}"
        )

    eigenvalues = np.linalg.eigvalsh(m)
    negative = np.nonzero(eigenvalues < -tol)[0]
    if negative.size > 0:
        offenders = ", ".join(f"eigenvalue[{i}]={eigenvalues[i]:.3e}" for i in negative)
        raise ValueError(
            f"joint_covariance is not positive semi-definite: {offenders} "
            f"(full spectrum: {np.round(eigenvalues, 8).tolist()}). A "
            f"correlation/covariance matrix assembled from independently "
            f"estimated pairwise entries is a common way to end up here -- "
            f"see nearest_psd() for an opt-in repair utility."
        )


def nearest_psd(matrix, epsilon: float = 1e-10) -> np.ndarray:
    """
    Opt-in repair of a non-PSD `joint_covariance`: clip eigenvalues up to `epsilon`,
    reconstruct and re-symmetrize.

    Clips to a small positive `epsilon`, not 0: an exactly rank-deficient matrix still
    gives NaN from `jnp.linalg.cholesky`. Never called automatically; a bad correlation
    input is rejected unless the caller repairs it knowingly.
    """
    m = np.asarray(matrix, dtype=np.float64)
    symmetric = 0.5 * (m + m.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, epsilon)
    repaired = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    return 0.5 * (repaired + repaired.T)


@jax.jit
def reconstruct_yield_curves(hw_paths: jax.Array, A: jax.Array, B: jax.Array) -> jax.Array:
    """
    Short-rate paths -> discount curves P(t,T) = A(t,T) * exp(-B(t,T) * r(t)), shape
    [Scenarios, TimeSteps, Maturities, NumRates].
    """
    r_t = hw_paths[:, :, None, :]  
    A_bcast = A[None, :, :, :]     
    B_bcast = B[None, :, :, :]     
    
    # Hull-White affine bond price, vectorized.
    discount_curves = A_bcast * jnp.exp(-B_bcast * r_t)
    return discount_curves


# Phase 4: public API
@dataclass
class ZeroCurveConfig:
    """Today's zero curve for one rate factor (pillar times in years, continuously
    compounded zero rates).

    `provenance` is optional metadata (observed, assumed or synthetic); the simulation
    never reads it. `None` means unstated, which `engine.integration` treats as not
    observed when it reports a result's market provenance (I-11). See
    `engine.integration.market_inputs.CurveProvenance`.
    """
    times: List[float]
    rates: List[float]
    # Optional[object] rather than CurveProvenance, so this module does not import
    # engine.integration (the dependency runs the other way).
    provenance: Optional[object] = None


@dataclass
class EquityConfig:
    """Equity/FX leg. `rate_mapping[i]` holds equity i's drift coefficient on each rate
    factor (row length NumHW)."""
    initial_prices: List[float]
    dividend_yields: List[float]
    rate_mapping: List[List[float]]


@dataclass
class RatesConfig:
    """Hull-White rates leg, one entry per rate factor in `initial_rates`, `theta`,
    `mean_reversion`. Setting `maturities` adds a `yield_curves` output and requires
    `initial_zero_curves`, one curve per factor in the same order."""
    initial_rates: List[float]
    theta: List[float]
    mean_reversion: List[float]
    maturities: Optional[List[float]] = None
    initial_zero_curves: Optional[List[ZeroCurveConfig]] = None


@dataclass
class SimulationConfig:
    """
    Configuration for `generate_paths` (see each nested dataclass).

    time_grid: absolute times, ascending, starting at 0.0.
    joint_covariance: [NumEq+NumHW] square, equities first then rates, in the order of
        `equities.initial_prices` and `rates.initial_rates`. The diagonal gives each
        factor's (constant) volatility squared; off-diagonals give correlations.
    seed: Sobol scrambling seed (see `generate_sobol_normals`).
    """
    time_grid: List[float]
    equities: EquityConfig
    rates: RatesConfig
    joint_covariance: List[List[float]]
    scenarios: int = 10000
    seed: int = 42


def generate_paths(config: SimulationConfig, precision: int = 64) -> Dict[str, jax.Array]:
    """
    Sobol -> Brownian bridge -> cross-asset paths -> (optional) yield-curve cube.

    precision: 64 (float64) or 32 (float32). `jax_enable_x64` is process-global, so it is
        set for this call and restored on exit (I-14).

    Returns "equities" [S,T,NumEq], "rates" [S,T,NumHW], "numeraire" [S,T], and, if
    `config.rates.maturities` is set, "yield_curves" [S,T,Maturities,NumHW]. The time axis
    is `time_grid[1:]`.
    """
    validate_joint_covariance(config.joint_covariance)

    with _x64_enabled(precision == 64):
        return _generate_paths_inner(config, precision)


@contextmanager
def _x64_enabled(enabled: bool):
    """Set `jax_enable_x64` for the body and restore the previous value (not
    unconditionally True), so an all-float32 process stays float32."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", enabled)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _generate_paths_inner(config: SimulationConfig, precision: int) -> Dict[str, jax.Array]:
    """Body of `generate_paths`; call that instead."""
    dtype = jnp.float64 if precision == 64 else jnp.float32

    # 1. Setup
    time_grid = jnp.array(config.time_grid, dtype=dtype)
    dt_t = jnp.diff(time_grid)
    num_steps = dt_t.shape[0]
    num_scenarios = int(config.scenarios)

    # 2. Equities
    eq_cfg = config.equities
    eq_S0 = jnp.array(eq_cfg.initial_prices, dtype=dtype)
    num_eq = eq_S0.shape[0]
    if len(eq_cfg.dividend_yields) != num_eq:
        raise ValueError(
            f"equities.dividend_yields must have exactly one entry per "
            f"equity: got {len(eq_cfg.dividend_yields)} entries for "
            f"{num_eq} equities."
        )
    if len(eq_cfg.rate_mapping) != num_eq:
        raise ValueError(
            f"equities.rate_mapping must have exactly one row per equity: "
            f"got {len(eq_cfg.rate_mapping)} rows for {num_eq} equities."
        )
    eq_div_t = jnp.tile(jnp.array(eq_cfg.dividend_yields, dtype=dtype), (num_steps, 1))
    rate_mapping = jnp.array(eq_cfg.rate_mapping, dtype=dtype)

    # 3. Rates
    hw_cfg = config.rates
    hw_r0 = jnp.array(hw_cfg.initial_rates, dtype=dtype)
    num_hw = hw_r0.shape[0]
    if len(hw_cfg.theta) != num_hw:
        raise ValueError(
            f"rates.theta must have exactly one entry per rate factor: "
            f"got {len(hw_cfg.theta)} entries for {num_hw} rate factors."
        )
    if len(hw_cfg.mean_reversion) != num_hw:
        raise ValueError(
            f"rates.mean_reversion must have exactly one entry per rate "
            f"factor: got {len(hw_cfg.mean_reversion)} entries for "
            f"{num_hw} rate factors."
        )
    if rate_mapping.shape[1] != num_hw:
        raise ValueError(
            f"equities.rate_mapping rows must have one column per rate "
            f"factor: got {rate_mapping.shape[1]} columns for {num_hw} "
            f"rate factors."
        )
    hw_theta_t = jnp.tile(jnp.array(hw_cfg.theta, dtype=dtype), (num_steps, 1))
    hw_a = jnp.array(hw_cfg.mean_reversion, dtype=dtype)

    num_joint = num_eq + num_hw
    if (
        len(config.joint_covariance) != num_joint
        or any(len(row) != num_joint for row in config.joint_covariance)
    ):
        raise ValueError(
            f"joint_covariance must be a square "
            f"({num_joint}x{num_joint}) matrix (equities first, then "
            f"rate factors, {num_eq} equities + {num_hw} rate factors): "
            f"got a matrix with {len(config.joint_covariance)} rows."
        )

    # 4. Correlation and volatilities. L_t is the Cholesky factor of the correlation
    #    (unit diagonal); each shock is scaled by its own sigma in the step function, so a
    #    Cholesky of the raw covariance would apply every volatility twice.
    cov_raw = jnp.array(config.joint_covariance, dtype=dtype)
    cov_t = jnp.tile(cov_raw[None, :, :], (num_steps, 1, 1))
    joint_sigma_t = jnp.sqrt(jnp.diagonal(cov_t, axis1=1, axis2=2))
    # A factor with zero variance would make its correlation row 0/0 = NaN, which Cholesky
    # spreads to every factor. Its shock is multiplied by sigma = 0 anyway, so its row and
    # column are replaced by the identity.
    sigma_is_zero = joint_sigma_t == 0.0
    pair_is_zero = sigma_is_zero[:, :, None] | sigma_is_zero[:, None, :]
    joint_sigma_t_safe = jnp.where(sigma_is_zero, 1.0, joint_sigma_t)
    corr_t_raw = cov_t / (joint_sigma_t_safe[:, :, None] * joint_sigma_t_safe[:, None, :])
    identity = jnp.eye(num_eq + num_hw, dtype=dtype)[None, :, :]
    corr_t = jnp.where(pair_is_zero, identity, corr_t_raw)
    L_t = jnp.linalg.cholesky(corr_t)
    eq_sigma_t = joint_sigma_t[:, :num_eq]
    hw_sigma_t = joint_sigma_t[:, num_eq:]

    # 5. Simulation
    Z_sobol = generate_sobol_normals(num_scenarios, num_steps, num_eq + num_hw, dtype, seed=config.seed)
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

    # 6. Yield-curve cube (if maturities are configured)
    if hw_cfg.maturities is not None:
        maturities = jnp.array(hw_cfg.maturities, dtype=dtype)

        step_times = time_grid[1:]  # the cube is evaluated at the step ends
        T_minus_t = jnp.maximum(maturities[None, :] - step_times[:, None], 0.0)

        # B(t,T) with T - t clamped at 0, so a pillar already in the past gets B = 0 (see
        # I-04: its value is then not a discount factor).
        B_matrix = jax.vmap(lambda a: _hw_B(0.0, T_minus_t, a), out_axes=-1)(hw_a)

        # A(t,T), fitted per rate factor to that factor's own curve.
        if len(hw_cfg.initial_zero_curves) != num_hw:
            raise ValueError(
                f"rates.initial_zero_curves must have exactly one curve per "
                f"rate factor: got {len(hw_cfg.initial_zero_curves)} curves "
                f"for {num_hw} rate factors."
            )
        A_matrix_np = compute_hw_A_matrix(
            zero_curves=hw_cfg.initial_zero_curves,
            hw_a=np.asarray(hw_cfg.mean_reversion, dtype=np.float64),
            hw_sigma=np.asarray(hw_sigma_t[0], dtype=np.float64),
            step_times=np.asarray(step_times, dtype=np.float64),
            maturities=np.asarray(maturities, dtype=np.float64),
            B_matrix=np.asarray(B_matrix, dtype=np.float64),
        )
        A_matrix = jnp.asarray(A_matrix_np, dtype=dtype)

        yield_cube = reconstruct_yield_curves(hw_paths, A_matrix, B_matrix)
        results["yield_curves"] = yield_cube

    return results


# Demo
if __name__ == "__main__":
    from engine.simulation.demo_scenarios import cross_asset_demo_config

    print("Initializing QMC Pipeline & JIT Compilation...")
    market_cubes = generate_paths(cross_asset_demo_config())

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