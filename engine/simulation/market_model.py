import jax
# JAX requires jax_enable_x64 to be set once, globally, before any float64
# array can be created at all -- it is not a per-array/per-call setting (a
# JAX/XLA constraint, not a design choice in this codebase). Default to
# 64-bit precision at import time; generate_paths() re-toggles this per call
# based on its `precision` argument, which is the correct, idiomatic JAX
# pattern for supporting both precisions in one process (verified: sequential
# calls with different `precision` values produce correctly-typed output
# each time). Every function in this module that accepts an explicit `dtype`
# honors it regardless of the ambient global flag (see generate_sobol_normals).
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm
from scipy.stats.qmc import Sobol
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional

from engine.models.hull_white import A as _hw_A, B as _hw_B, ZeroCurve as _HwZeroCurve

# =============================================================================
# PHASE 1: QUASI-MONTE CARLO (CPU -> GPU)
# =============================================================================
def generate_sobol_normals(num_scenarios: int, num_steps: int, num_assets: int, dtype) -> jax.Array:
    """
    CPU: Generates Sobol sequences.
    GPU: Converts to Normal shocks.
    Returns: [TimeSteps, Scenarios, Assets]

    `dtype` is always honored on output, regardless of the ambient global
    jax_enable_x64 state: jax.scipy.stats.norm.ppf computes internally in
    float64 whenever x64 mode is on (even for a float32 input), so the
    result is explicitly cast back to `dtype` before returning rather than
    trusting norm.ppf's output dtype directly.
    """
    total_dimensions = num_steps * num_assets

    # Scramble adds necessary randomness to the deterministic Sobol points
    sobol_engine = Sobol(d=total_dimensions, scramble=True, seed=42)
    uniform_draws = sobol_engine.random(n=num_scenarios)

    # Transfer to JAX and convert to standard normals. The JAX half is
    # factored into a jitted helper below: without a jit boundary each of
    # clip/ppf/reshape/transpose dispatches eagerly (separately traced,
    # lowered, compiled, executed and discarded), which measurably dominated
    # this function's cost -- it was the single largest source of XLA
    # compilations in a whole pricing job.
    return _sobol_uniforms_to_normals(uniform_draws, num_scenarios, num_steps, num_assets, dtype)


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def _sobol_uniforms_to_normals(uniform_draws, num_scenarios: int, num_steps: int, num_assets: int, dtype) -> jax.Array:
    """GPU half of `generate_sobol_normals`: clip the CPU-generated Sobol
    uniforms off the open-interval endpoints, map them through the inverse
    normal CDF, and reshape to [TimeSteps, Scenarios, Assets].

    Everything except `uniform_draws` is static (plain ints and a dtype), so
    one compiled kernel is reused for every call with the same simulation
    shape. See `generate_sobol_normals`'s docstring on why the explicit
    `.astype(dtype)` is required regardless of the ambient x64 state."""
    uniform_jax = jnp.asarray(uniform_draws, dtype=dtype)
    epsilon = jnp.finfo(dtype).eps
    uniform_clipped = jnp.clip(uniform_jax, epsilon, 1.0 - epsilon)

    normal_shocks = norm.ppf(uniform_clipped).astype(dtype)

    # Reshape to match JAX loop expectations
    Z = normal_shocks.reshape((num_scenarios, num_steps, num_assets))
    return jnp.transpose(Z, (1, 0, 2))

def _build_bridge_matrix(time_grid: np.ndarray) -> np.ndarray:
    """
    CPU: Constructs the Brownian Bridge path-construction matrix B such that
    W(t_1..t_n) = B @ Z, where Z are independent N(0,1) draws ordered by
    Sobol-dimension significance (dimension 0 = path endpoint, dimension 1 =
    midpoint, etc.), following the standard recursive bisection algorithm
    used by QuantLib/ORE's BrownianBridge.
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

    # map_[k] != 0 marks slot k as already assigned a construction order;
    # each iteration bisects the widest unfilled gap (QuantLib BrownianBridge).
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
        # Choose the midpoint index between the two known bounds j-1..k
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

    # Translate the (left/right/bridge) recursion into an explicit linear map
    # from Z (Sobol-ordered independent normals) to W (path values at each
    # grid time), so it can be applied as a single matrix multiply.
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
    """
    Transforms independent Normal shocks into Bridged Chronological Shocks.
    """
    num_steps, num_scenarios, num_assets = Z.shape

    B_matrix_np = _build_bridge_matrix(np.asarray(time_grid))
    B_matrix = jnp.asarray(B_matrix_np, dtype=Z.dtype)

    # Apply Bridge Matrix to reorder variance
    W_paths = _apply_bridge_matrix(B_matrix, Z)
    
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
    """
    GPU: joint Hull-White 1-Factor (rates) + correlated GBM (equities/FX)
    Monte Carlo path simulation, stepped chronologically via jax.lax.scan.

    Per step: correlates the bridged shocks via the (per-step) Cholesky
    factor L_t, evolves each rate factor with the exact HW1F transition
    (mean-reverting Ornstein-Uhlenbeck), evolves each equity/FX path via GBM
    with a dynamic drift derived from the simulated short rates (Uncovered
    Interest Rate Parity: rate_mapping maps each equity/FX to the rate
    factor(s) it depends on), and accrues a money-market numeraire off rate
    factor index 0 only (single base-currency discounting account -- see
    "Using Index 0 as base discount curve" below).

    Shapes: eq_S0/hw_r0 are [NumEq]/[NumHW]; eq_div_t/eq_sigma_t/hw_theta_t/
    hw_sigma_t are [TimeSteps, NumEq or NumHW]; rate_mapping is
    [NumEq, NumHW]; L_t is [TimeSteps, NumEq+NumHW, NumEq+NumHW]; dt_t is
    [TimeSteps]; Z_bridged is [TimeSteps, Scenarios, NumEq+NumHW].
    Returns (eq_paths, hw_paths, numeraire_paths), each
    [Scenarios, TimeSteps, ...].
    """
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
        # Exact Ornstein-Uhlenbeck transition: E[r(t+dt)|r(t)] = r(t)*decay +
        # theta*(1-decay), NOT r(t)*decay + theta -- the latter treats theta
        # as a flat per-step drift increment rather than the long-run mean
        # target, causing r(t) to diverge upward (or downward) without
        # bound every step instead of reverting toward theta. This was
        # invisible in every prior demo/test because they all set
        # theta == initial_rates (a fixed point ONLY under the correct
        # formula); confirmed against the closed-form OU transition mean
        # directly (e.g. a=0.03, dt=0.5, theta != r0 diverges by several
        # points after just a few steps under the old formula).
        # hw_a==0.0 (arithmetic Brownian motion, the mathematically valid
        # a->0 limit of OU mean reversion) is a removable 0/0 singularity
        # in variance_hw as literally written -- guarded by evaluating the
        # formula on a safe placeholder a (never actually 0) and selecting
        # the analytic limit (variance_hw->dt) via jnp.where instead,
        # rather than letting hw_a==0.0 divide by zero into NaN. Both
        # branches are evaluated unconditionally (branch-free/jit-
        # friendly), the placeholder is just discarded when hw_a != 0.
        hw_a_safe = jnp.where(hw_a == 0.0, 1.0, hw_a)
        decay = jnp.exp(-hw_a * dt_i)
        variance_hw = jnp.where(
            hw_a == 0.0,
            dt_i,
            (1.0 - jnp.exp(-2.0 * hw_a_safe * dt_i)) / (2.0 * hw_a_safe),
        )
        shock_hw = sig_hw * jnp.sqrt(variance_hw) * Z_hw
        r_next = r_t * decay + theta_hw * (1.0 - decay) + shock_hw
        
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
def _initial_log_discount(zero_times: np.ndarray, zero_rates: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Continuously-compounded log discount factor ln P(0,t) via linear
    interpolation on zero rates, flat-extrapolated at the curve ends. Thin
    NumPy-facing wrapper around `engine.models.hull_white.log_discount` --
    kept for existing NumPy call sites/tests; the underlying formula lives
    in exactly one place (`engine/models/hull_white.py`), not re-derived
    here."""
    curve = _HwZeroCurve(pillar_times=jnp.asarray(zero_times), pillar_rates=jnp.asarray(zero_rates))
    from engine.models.hull_white import log_discount
    return np.asarray(log_discount(curve, jnp.asarray(t)))


def compute_hw_A_matrix(
    zero_curves: List["ZeroCurveConfig"],
    hw_a: np.ndarray,
    hw_sigma: np.ndarray,
    step_times: np.ndarray,
    maturities: np.ndarray,
    B_matrix: np.ndarray,
) -> np.ndarray:
    """
    Closed-form Hull-White 1-Factor A(t,T), calibrated independently per
    rate factor against that factor's OWN initial zero curve -- a thin
    per-rate-factor `jax.vmap` wrapper around
    `engine.models.hull_white.A` (the single shared implementation of this
    formula; see that module's docstring for the math and the ORE
    correspondence). Matches ORE's Cross-Asset Model: every
    IrLgm1fParametrization is constructed with its own (Currency,
    YieldTermStructureHandle) pair -- live-verified against the installed
    ORE package that no shared-curve constructor path exists (a
    2-currency CrossAssetModel with distinct USD 3%/EUR 2% flat curves
    retains each currency's own discount factors throughout, never
    cross-contaminating). One shared curve across factors would be a bug,
    not a legitimate simplification of ORE's design.

    zero_curves: one ZeroCurveConfig per rate factor (len == hw_a.shape[0]),
    in the same order as hw_a/hw_sigma/maturities' NumRates axis.
    Shapes: step_times [TimeSteps], maturities [Maturities], B_matrix
    [TimeSteps, Maturities, NumRates] -> returns A [TimeSteps, Maturities, NumRates].

    Returns a plain `np.ndarray` (this function's existing, NumPy-facing
    contract used by `generate_paths` below and by this module's own
    tests) even though the computation itself now runs through JAX.
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
        # B_override=the caller's own B_matrix slice (see A()'s docstring
        # on B_override for why this must be threaded through rather than
        # recomputed from t_grid/T_grid here).
        A_k = _hw_A(
            curve, t_grid, T_grid, float(hw_a[k]), float(hw_sigma[k]),
            B_override=B_matrix_j[:, :, k],
        )  # [TimeSteps, Maturities]
        A_per_factor.append(A_k)
    return np.asarray(jnp.stack(A_per_factor, axis=-1))


def validate_joint_covariance(matrix) -> None:
    """
    Guards `generate_paths` against a non-PSD `joint_covariance` -- without
    this check, `jnp.linalg.cholesky` on an invalid correlation matrix
    silently produces an all-NaN factor, which then silently NaNs every
    simulated path with no error raised anywhere (see
    docs/planning/traderx-integration.md's gap item 1). Cheap: this runs
    once per `generate_paths` call, on the CPU, not once per scenario.

    Checks (in order): square, symmetric (within float tolerance), and
    positive semi-definite (every eigenvalue >= -tol, via
    `np.linalg.eigvalsh` -- the standard, numerically stable eigenvalue
    routine for symmetric matrices). Raises `ValueError` naming the
    specific problem; for a non-PSD matrix, names every offending
    (negative) eigenvalue and its index in the ascending-sorted spectrum
    `eigvalsh` returns.
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
    Opt-in "best effort" repair for a `joint_covariance` that fails
    `validate_joint_covariance`: standard eigenvalue-clipping projection
    onto the nearest positive semi-definite matrix (clip every eigenvalue
    below `epsilon` up to `epsilon`, reconstruct from the clipped spectrum,
    re-symmetrize to cancel floating-point asymmetry introduced by the
    reconstruction).

    Clips to `epsilon` (a small positive number), not literally 0 --
    clipping all the way to exactly 0 produces a mathematically-PSD-but-
    numerically-rank-deficient matrix, which is a real floating-point
    Cholesky failure mode in its own right (the same singular-boundary
    phenomenon as an exact rho=+-1 correlation: `jnp.linalg.cholesky`
    silently returns NaN at exact rank deficiency, confirmed directly in
    `tests/test_market_model.py::TestCholeskyOnDegenerateCorrelation`) --
    so a repair that clips to exactly 0 would frequently hand
    `generate_paths` a matrix that still NaNs downstream, defeating the
    entire point of "opt-in best-effort repair." `epsilon` trades a
    negligible amount of variance for a matrix that is genuinely usable.

    Deliberately never called automatically by `generate_paths` or
    anything in `engine/portfolio/request.py` -- a genuinely bad correlation input
    must be rejected loudly, not silently altered and priced anyway. A
    caller who explicitly wants "close enough" behavior calls this
    directly and is then responsible for having done so knowingly.
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
@dataclass
class ZeroCurveConfig:
    """Today's market zero curve pillars for one rate factor, used to
    calibrate the Hull-White A(t,T) term (see compute_hw_A_matrix).

    `provenance` records where these numbers came from -- an observed
    bootstrap, a named assumed profile, or synthetic fixture data. It is
    **optional and defaults to None**, so every existing caller is
    unaffected and unchanged: this field is metadata only, read by nothing
    in the simulation math below, and `generate_paths` never branches on
    it.

    **Why it exists at all** (plan §W0.6, part of I-11): before it, this
    dataclass carried times and rates and nothing else, so a flat 3%
    assumption and a bootstrapped market curve were *the same object* and
    no downstream code could tell them apart. A result computed against an
    assumption looked exactly like one computed against the market. The
    engine now carries the distinction internally rather than only at its
    edges -- see `engine.integration.market_inputs.CurveProvenance`.

    `None` means "unstated", which is honestly different from
    `inputOrigin: "observed"`. Nothing infers observedness from a missing
    provenance; `engine.integration` treats unstated as not-observed when
    it computes a result's top-level `marketProvenance`.
    """
    times: List[float]
    rates: List[float]
    # Typed as Optional[object] rather than Optional[CurveProvenance] to
    # keep this module free of any engine.integration import: the
    # dependency runs integration -> simulation, and reversing it here
    # would make the simulation layer depend on the TraderX boundary.
    provenance: Optional[object] = None


@dataclass
class EquityConfig:
    """Equity/FX leg of the cross-asset model. rate_mapping[i] gives the
    Uncovered-Interest-Rate-Parity drift coefficients for equity/FX i
    against every Hull-White rate factor (row length == RatesConfig's
    NumHW)."""
    initial_prices: List[float]
    dividend_yields: List[float]
    rate_mapping: List[List[float]]


@dataclass
class RatesConfig:
    """Hull-White 1-Factor rates leg. initial_rates/theta/mean_reversion
    are one entry per rate factor. maturities is optional -- its presence
    triggers yield_curves output in generate_paths' return dict; when set,
    initial_zero_curves is required: one ZeroCurveConfig PER rate factor, in
    the same order as initial_rates/theta/mean_reversion (len must match).
    This mirrors ORE's Cross-Asset Model exactly -- every
    IrLgm1fParametrization is constructed with its own (Currency,
    YieldTermStructureHandle) pair, never a curve shared across factors
    (live-verified against the installed ORE package; see
    compute_hw_A_matrix's docstring for the evidence)."""
    initial_rates: List[float]
    theta: List[float]
    mean_reversion: List[float]
    maturities: Optional[List[float]] = None
    initial_zero_curves: Optional[List[ZeroCurveConfig]] = None


@dataclass
class SimulationConfig:
    """
    Typed configuration for generate_paths, mirroring the engine's actual
    parameter structure 1:1 (see each nested dataclass's docstring). This
    is the canonical, IDE- and API-friendly entry point -- catches
    misspelled/missing fields at construction time via Python's own
    dataclass machinery, rather than a KeyError deep inside generate_paths.
    Also the natural shape for the Pydantic schema in `engine/api/schemas.py`
    to mirror -- see the roadmap's "TraderX API integration" phase
    (docs/planning/roadmap-and-history.md) for status.

    time_grid: absolute times, ascending, starting at 0.0.
    joint_covariance: [NumEq+NumHW, NumEq+NumHW], equities first then rates,
        in the same order as equities.initial_prices / rates.initial_rates.
    """
    time_grid: List[float]
    equities: EquityConfig
    rates: RatesConfig
    joint_covariance: List[List[float]]
    scenarios: int = 10000


def generate_paths(config: SimulationConfig, precision: int = 64) -> Dict[str, jax.Array]:
    """
    Runs the full Sobol -> Brownian bridge -> cross-asset Monte Carlo ->
    yield curve reconstruction pipeline from a SimulationConfig.

    precision: 64 for float64 (default; required for the FP64 "ground truth"
        runs), 32 for float32 (Phase 9 lower-precision comparison runs).
        jax_enable_x64 must be a process-global JAX/XLA setting (not a
        per-array choice -- see the module-level comment above its default),
        so this toggles it for the duration of this call; sequential calls
        with different `precision` values each produce correctly-typed
        output. generate_sobol_normals honors `dtype` directly regardless of
        this global state, so calling it standalone is also safe.

    Returns a dict with "equities" [S,T,NumEq], "rates" [S,T,NumHW],
    "numeraire" [S,T], and (if config.rates.maturities is set)
    "yield_curves" [S,T,Maturities,NumHW].
    """
    validate_joint_covariance(config.joint_covariance)

    jax.config.update("jax_enable_x64", precision == 64)
    dtype = jnp.float64 if precision == 64 else jnp.float32

    # 1. Base Setup
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

    # 4. Joint Matrix
    # L_t must carry CORRELATION only (unit diagonal), not the raw
    # covariance magnitude -- step_fn separately multiplies each factor's
    # correlated shock by its own sig_eq/hw_sigma_t (below), so a Cholesky
    # factor built from the raw covariance matrix would double-apply every
    # factor's volatility (once via L_t's own diagonal scale, once via the
    # explicit sig_eq/sig_hw multiplication downstream). Confirmed as a
    # real, previously-undetected bug: with the raw-covariance Cholesky, a
    # configured 20% equity vol produced an actual simulated log-return std
    # of ~4% (0.2^2), and a configured 1.5% rate vol produced an actual
    # simulated short-rate std smaller by the same squared factor -- caught
    # by checking simulated variance against the closed-form HW1F/GBM
    # transition variance directly, not just cross-checking formulas at a
    # single point.
    cov_raw = jnp.array(config.joint_covariance, dtype=dtype)
    cov_t = jnp.tile(cov_raw[None, :, :], (num_steps, 1, 1))
    joint_sigma_t = jnp.sqrt(jnp.diagonal(cov_t, axis1=1, axis2=2))
    # A factor with exactly zero variance makes its row/column of
    # sigma_i*sigma_j a literal 0/0 -- NaN, which jnp.linalg.cholesky then
    # propagates through the ENTIRE factor matrix (not just that factor's
    # own row/column), silently poisoning every other, perfectly
    # well-defined factor too. That factor's shock is multiplied by its
    # own sigma=0 downstream (shock_hw/shock_eq) regardless of what
    # correlation value it carries here, so its off-diagonal correlation
    # entries are numerically irrelevant -- guarded by substituting the
    # identity row/column (0 off-diagonal, 1 on-diagonal) for any
    # zero-variance factor before Cholesky, which keeps every other
    # factor's real correlation structure and Cholesky factor intact.
    sigma_is_zero = joint_sigma_t == 0.0
    pair_is_zero = sigma_is_zero[:, :, None] | sigma_is_zero[:, None, :]
    joint_sigma_t_safe = jnp.where(sigma_is_zero, 1.0, joint_sigma_t)
    corr_t_raw = cov_t / (joint_sigma_t_safe[:, :, None] * joint_sigma_t_safe[:, None, :])
    identity = jnp.eye(num_eq + num_hw, dtype=dtype)[None, :, :]
    corr_t = jnp.where(pair_is_zero, identity, corr_t_raw)
    L_t = jnp.linalg.cholesky(corr_t)
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
    if hw_cfg.maturities is not None:
        maturities = jnp.array(hw_cfg.maturities, dtype=dtype)

        step_times = time_grid[1:] # We evaluate AT the step ends
        T_minus_t = jnp.maximum(maturities[None, :] - step_times[:, None], 0.0)

        # B(t,T) formula via the single shared implementation
        # (engine.models.hull_white.B, vmapped across rate factors). Uses
        # T_minus_t (clamped at 0 for a cashflow past its own maturity,
        # e.g. a pillar earlier than the current step) rather than raw
        # T-t directly -- B's own a==0 removable-singularity guard still
        # applies unchanged since it operates on this already-clamped gap.
        B_matrix = jax.vmap(lambda a: _hw_B(0.0, T_minus_t, a), out_axes=-1)(hw_a)

        # A(t,T): calibrated to today's market zero curve, independently per
        # rate factor, so each factor's simulated discount factors reprice
        # its OWN initial term structure (matches ORE's Cross-Asset Model --
        # see compute_hw_A_matrix's docstring).
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


# =============================================================================
# EXECUTION DEMONSTRATION
# =============================================================================
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