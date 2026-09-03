"""
Bermudan swaption pricing via a numeric LGM short-rate grid, using Hagan's
Gaussian-quadrature convolution scheme. This is also the engine behind
American swaption pricing -- see engine.instruments.american_swaption, a
thin wrapper around this module (American exercise is priced here as a
finely-discretized Bermudan, exactly matching ORE's own design).

**Fully JAX-native.** An earlier version of this module ran its backward
induction in plain NumPy on the CPU (a dynamic Python loop over a
`set`/`round()`-deduplicated list of grid times, with dict-based snapshot
bookkeeping) -- correct, and extensively verified against ORE, but with no
JAX computational graph for `jax.grad` to differentiate through (blocking
Greeks) and no ability to run on GPU. This version reproduces the exact
same algorithm as a `jax.lax.scan` over a PRECOMPUTED, fixed-length grid-
time schedule (built once, in plain Python, from the trade's own
`exercise_times`/requested `condition_times` -- a genuinely static
quantity, known before any array math starts, so padding it to a fixed
length is not an approximation, just a JAX tracing requirement). See
"THE GRID-TIME SCHEDULE" section below for the exact construction. Cross-
validated exhaustively against the original NumPy engine (kept as
`engine/instruments/bermudan_swaption_numpy_reference.py` during this
validation period) across a randomized sweep of configs before being
adopted as this module's implementation -- see
`tests/test_bermudan_jax_vs_numpy.py`.

Why not Jamshidian's decomposition (see engine.instruments.european_swaption):
Jamshidian's trick relies on the option having a SINGLE exercise date, so the
whole coupon bond's exercise boundary collapses to one critical rate r* at
that one date. Early exercise (a discrete list of exercise dates, or --
via engine.instruments.american_swaption -- any date in a window) breaks
that -- the holder's optimal decision at an early date depends on the
(model-implied) continuation value of holding the option further, which has
no closed form. ORE itself only uses Jamshidian's engine for European
swaptions and switches to a genuinely numeric engine,
`QuantExt::NumericLgmMultiLegOptionEngine`, for both Bermudan and American --
confirmed by reading ORE's own trade-builder source
(`OREData/ored/portfolio/builders/swaption.hpp`/`.cpp`): `EuropeanSwaption`
routes to `BlackBachelierSwaptionEngine`; `BermudanSwaption` and
`AmericanSwaption` both route to `LGMSwaptionEngineBuilder`, which
constructs `NumericLgmMultiLegOptionEngine` backed by either a
`LgmConvolutionSolver2` ("Grid", the default for Bermudan in ORE's own
example configs) or an `LgmFdSolver` ("FD", the default for American there).
This module implements the "Grid" convolution solver -- it is the simpler of
ORE's two numerically-equivalent backward-induction schemes to reproduce
exactly (a closed-form Gaussian quadrature, not a PDE discretization scheme
with its own scheme-dependent truncation error), and both solvers plug into
the *identical* `max(intrinsic, continuation)` backward-induction loop in
`NumericLgmMultiLegOptionEngineBase::calculate()` -- the choice of solver is
a numerical-implementation detail, not a modeling difference; see
docs/instruments/american-bermudan-swaptions.md for the full writeup.

Algorithm (see docs/instruments/american-bermudan-swaptions.md for full derivation and every
cited ORE source line):

1. Build the underlying swap's fixed and floating cashflow schedule with
   ORE's own MakeVanillaSwap machinery (exactly as
   engine.instruments.swap does) -- date generation and accrual math match
   ORE exactly.
2. Build a 1-D short-rate state grid at each backward-induction time step:
   `x_k = k * dx(t)`, `dx(t) = sigma*sqrt(t) / n_per_std`, spanning
   `+/- std_devs` standard deviations of the model's own `t`-conditional
   distribution -- the same grid shape as
   `QuantExt::LgmConvolutionSolver2::stateGrid` (`zeta(t) = sigma^2 * t` for
   a constant-parameter HW1F/LGM model, so `dx` here is exactly that
   function's `sqrt(zeta(t))/nx`).
3. At the final grid time, the underlying swap's remaining value is the
   closed-form LGM swap NPV (fixed leg + floating leg, each cashflow priced
   off `P(t,T,x) = [P(0,T)/P(0,t)]*exp(-0.5*(H(T)^2-H(t)^2)*zeta(t))*
   exp(-(H(T)-H(t))*x)` at every state-grid node -- `engine.models.lgm.
   bond_price`, live-verified to machine precision against
   `ORE.LinearGaussMarkovModel.discountBond`). **Deliberately NOT** this
   codebase's other HW1F closed form (`engine.models.hull_white`, used
   throughout engine.instruments.european_swaption and swap) -- see
   `engine.models.lgm`'s own docstring for why: `ORE.HullWhite` and
   `ORE.LinearGaussMarkovModel`, despite sharing (a, sigma) and today's
   curve, are NOT the same numerical model realization for t>0, a finding
   made and verified live while originally building this module. Since
   ORE's actual Bermudan/American engine is built on
   `LinearGaussMarkovModel`, this module matches THAT model exclusively,
   not the plain-HullWhite closed forms used elsewhere in this codebase.
4. Roll backward one grid step at a time via Hagan's quadrature convolution
   (`_hagan_quadrature_weights`, `_rollback_one_step`) -- precomputed
   trapezoid-of-normal-density weights applied to a linearly-interpolated
   value function, exactly `QuantExt::LgmConvolutionSolver2`'s scheme
   (Hagan, "Methodology for callable swaps and Bermudan exercise into
   swaptions").
5. At every exercise date, apply `optionValue = max(continuationValue,
   intrinsicValue)` elementwise across the state grid, where
   `intrinsicValue` is the underlying swap's own remaining NPV at that node
   (the value of exercising into the swap right there) -- the exact rule in
   `NumericLgmMultiLegOptionEngineBase::calculate()`.
6. The t=0 grid collapses to the single node `x=0` (matching
   `LgmConvolutionSolver2::stateGrid(0)`); reading off that node gives the
   base-case NPV. Conditional (per-scenario, per-step) NPV is produced by
   linearly interpolating the SAME rolled-back value function onto each
   scenario's simulated short rate at that step, exactly the same
   conditioning approach engine.instruments.european_swaption uses (the
   model's Markov property makes any such conditional evaluation exact, not
   an approximation) -- except here a distinct backward induction must be
   re-run rolled back TO each requested step time (a Bermudan option's
   value depends on the ENTIRE remaining exercise schedule, so it cannot be
   evaluated at an arbitrary future time from a single t=0 rollback the way
   Jamshidian's closed form can).

**Validation approach** (see docs/instruments/american-bermudan-swaptions.md and
tests/test_bermudan_swaption.py): ORE's Python (SWIG) bindings expose
`QuantExt.LinearGaussMarkovModel`'s closed-form pieces (`discountBond`,
`numeraire`, `parametrization().H/zeta/Hprime`) but do NOT expose a
constructible `NumericLgmMultiLegOptionEngine` (its constructor is not
bound) or `LgmConvolutionSolver2` directly -- confirmed by inspection
(`ORE.NumericLgmMultiLegOptionEngine()` raises "No constructor defined").
So, unlike the swap/VaR modules, this module's full backward-induction
engine cannot be cross-checked by calling an equivalent live ORE engine
object directly end-to-end. It IS, however, validated at the formula level
against live ORE objects, plus several model-independent structural checks:
  (a) Every closed-form building block this module's backward induction is
      built from -- H(t), zeta(t), H'(t), and the full bond price
      P(t,T,x) itself -- is live-verified to machine precision against
      `ORE.IrLgm1fConstantParametrization`/`ORE.LinearGaussMarkovModel`
      directly (see engine/models/lgm.py).
  (b) A single-exercise-date Bermudan (mathematically == a European
      swaption) must reproduce an INDEPENDENT Jamshidian-style closed-form
      decomposition built on the SAME LGM bond formula (not this
      codebase's HullWhite-parametrized Jamshidian pricer in
      european_swaption.py, which was confirmed -- while building this
      module -- to be a genuinely different model realization for t>0; see
      engine/models/lgm.py's docstring) -- verified to match to ~1e-7
      relative or better, confirming the backward-induction/convolution
      machinery, run with only one exercise opportunity, collapses to the
      correct closed form.
  (c) Monotonicity: a Bermudan/American swaption must be worth at least as
      much as the otherwise-identical European swaption exercisable only at
      the LAST of its dates (more exercise opportunities cannot decrease an
      option's value) -- a model-independent no-arbitrage bound, checked
      numerically.
  (d) Grid convergence: refining the state-grid resolution
      (n_per_std, std_devs) must change the price by a shrinking amount
      (standard numerical-scheme convergence check).
  (e) Cross-validation against the original NumPy engine (see module-level
      docstring above) across a randomized config sweep, the direct check
      that this rewrite is a faithful port and not just independently
      plausible.

Known limitation: no mid-coupon proration -- see BermudanSwaptionConfig's
own docstring and tests/test_bermudan_swaption.py::TestMidCouponKnownLimitation.
"""
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
import ORE

from engine.simulation.market_model import ZeroCurveConfig
from engine.models.static_key import StaticKeyMixin
from engine.models.ore_builders import build_vanilla_swap
from engine.portfolio.validation import _validate_common_fields, _validate_tenor
from engine.models.hull_white import ZeroCurve as _HwZeroCurve
from engine.models.lgm import (
    H as _H,
    H_prime as _H_prime,
    Sigma,
    bond_price as _lgm_bond_price,
    numeraire as _lgm_numeraire,
    r_from_x as _lgm_r_from_x,
    x_from_r as _lgm_x_from_r,
    zeta as _lgm_zeta,
)

DAY_COUNTER = ORE.Actual365Fixed()


@dataclass
class BermudanSwaptionConfig:
    """
    One Bermudan swaption: the option to enter a vanilla fixed-vs-floating
    swap on any one of a discrete list of exercise dates.

    exercise_times: sorted list of year-fractions from evaluation_date, each
    a date on which the holder may exercise into the (then-remaining)
    underlying swap. Must all be strictly less than the underlying swap's
    final maturity and (for the mid-coupon-safe subset implemented here --
    see docs/instruments/american-bermudan-swaptions.md's "Scope" section) coincide with one
    of the underlying's own accrual dates, matching ORE's own standard
    "coterminal" Bermudan structure (each exercise date is a fixed-leg reset
    date) -- ORE's own calibration machinery
    (SwaptionEngineBuilder::model()) is built around exactly this
    coterminal-date assumption for Bermudan/American calibration baskets.

    rate_factor_index/hw_a/hw_sigma/initial_zero_curve: same meaning and
    same single-model-pricing rationale as
    engine.instruments.european_swaption.SwaptionConfig -- see that
    module's docstring.

    hw_sigma accepts either a plain float (a flat volatility -- backward
    compatible with every existing caller) or an
    `engine.models.lgm.Sigma` (a genuine ORE-style piecewise-constant
    term structure, e.g. the output of `engine.calibration`'s LGM
    calibration routine) -- every formula this module calls into
    (`engine.models.lgm`) accepts both transparently via `as_sigma`, so no
    other code here needs to branch on which case a given trade is using.

    n_per_std/std_devs: state-grid resolution (points per standard
    deviation of the model's conditional distribution / how many standard
    deviations the grid spans) -- the numeric-scheme convergence parameters
    corresponding to ORE's own `nx`/`sx` Grid-engine parameters
    (`LGMGridSwaptionEngineBuilder`, `OREData/ored/portfolio/builders/
    swaption.cpp`).
    """
    notional: float
    fixed_rate: float
    payer: bool
    rate_factor_index: int
    hw_a: float
    hw_sigma: Optional[Union[float, Sigma]]
    initial_zero_curve: ZeroCurveConfig
    exercise_times: Sequence[float]
    swap_tenor: str = "5Y"
    index_tenor_months: int = 6
    floating_spread: float = 0.0
    n_per_std: int = 48
    std_devs: float = 6.0
    evaluation_date: ORE.Date = field(default_factory=lambda: ORE.Settings.instance().evaluationDate)

    def __post_init__(self) -> None:
        _validate_common_fields(self.notional, self.fixed_rate, self.evaluation_date)
        _validate_tenor(self.swap_tenor, "swap_tenor")
        # None is a valid sentinel meaning "uncalibrated" -- engine.portfolio.
        # price_portfolio fills it in via engine.calibration.lgm.
        # calibrate_lgm_sigma before this config ever reaches a pricer; a
        # bare BermudanSwaptionConfig(hw_sigma=None) constructed outside
        # that flow is likewise valid to build (just not directly priceable
        # until hw_sigma is filled in).
        if self.hw_sigma is not None:
            sigma_values = self.hw_sigma.values if isinstance(self.hw_sigma, Sigma) else [self.hw_sigma]
            if any(v != v or v in (float("inf"), float("-inf")) for v in np.asarray(sigma_values, dtype=np.float64).tolist()):
                raise ValueError(f"hw_sigma must be finite; got {self.hw_sigma}")
        if len(self.exercise_times) == 0:
            raise ValueError("exercise_times must be non-empty")
        times_list = [float(t) for t in self.exercise_times]
        if times_list != sorted(times_list):
            raise ValueError(f"exercise_times must be sorted ascending; got {times_list}")


def _build_ore_swap(cfg) -> ORE.VanillaSwap:
    """CPU: builds the real ORE underlying swap -- see
    `engine.models.ore_builders.build_vanilla_swap`, the single shared
    implementation of this construction."""
    return build_vanilla_swap(
        notional=cfg.notional, fixed_rate=cfg.fixed_rate, payer=cfg.payer,
        swap_tenor=cfg.swap_tenor, index_tenor_months=cfg.index_tenor_months,
        floating_spread=cfg.floating_spread, evaluation_date=cfg.evaluation_date,
    )


@dataclass(frozen=True, eq=False)
class _PreparedBermudan(StaticKeyMixin):
    payer: bool
    notional: float
    exercise_times: np.ndarray          # [E] sorted ascending
    fixed_times: np.ndarray             # [Nf] fixed payment times
    fixed_start_times: np.ndarray       # [Nf] fixed accrual-start times
    fixed_amounts: np.ndarray           # [Nf] notional*rate*accrual (ORE's own coupon.amount())
    float_pay_times: np.ndarray         # [Ncf]
    float_start_times: np.ndarray       # [Ncf]
    float_end_times: np.ndarray         # [Ncf]
    float_accrual: np.ndarray           # [Ncf]
    float_spread: float
    rate_factor_index: int
    hw_a: float
    hw_sigma: Union[float, Sigma]
    zero_times: np.ndarray
    zero_rates: np.ndarray
    n_per_std: int
    std_devs: float
    final_maturity: float


def prepare_bermudan(cfg: BermudanSwaptionConfig) -> _PreparedBermudan:
    """CPU: build the ORE underlying swap and extract both legs' full
    cashflow schedules (unlike Jamshidian, early exercise means the
    floating leg cannot be collapsed to a telescoping notional identity --
    the continuation value at each node needs the ACTUAL remaining swap
    value, so every floating coupon's forward-rate ingredients are kept
    explicitly). Static per swaption -- run once, not per grid node/step.

    fixed_amounts stores each coupon's already-computed cash amount
    (notional*rate*accrual, read directly from ORE's own
    FixedRateCoupon.amount(), the same source
    engine.instruments.european_swaption.prepare_swaption uses) rather than
    a separate accrual-fraction array -- _hw_swap_value_at_nodes discounts
    these amounts directly, with no need to re-multiply by rate/accrual."""
    swap = _build_ore_swap(cfg)
    today = cfg.evaluation_date

    fixed_times, fixed_start_times, fixed_amounts = [], [], []
    for cf in swap.fixedLeg():
        c = ORE.as_fixed_rate_coupon(cf)
        fixed_times.append(DAY_COUNTER.yearFraction(today, c.date()))
        fixed_start_times.append(DAY_COUNTER.yearFraction(today, c.accrualStartDate()))
        fixed_amounts.append(c.amount())

    float_pay, float_start, float_end, float_accrual = [], [], [], []
    for cf in swap.floatingLeg():
        c = ORE.as_floating_rate_coupon(cf)
        float_pay.append(DAY_COUNTER.yearFraction(today, c.date()))
        float_start.append(DAY_COUNTER.yearFraction(today, c.accrualStartDate()))
        float_end.append(DAY_COUNTER.yearFraction(today, c.accrualEndDate()))
        float_accrual.append(c.accrualPeriod())

    exercise_times = np.asarray(sorted(cfg.exercise_times), dtype=np.float64)
    final_maturity = max(fixed_times[-1], float_pay[-1])
    if np.any(exercise_times >= final_maturity):
        raise ValueError(
            f"exercise_times must all be strictly before the underlying "
            f"swap's final maturity ({final_maturity}); got {exercise_times.tolist()}"
        )

    return _PreparedBermudan(
        payer=cfg.payer,
        notional=swap.fixedNominals()[0] if swap.fixedNominals() else swap.nominal(),
        exercise_times=exercise_times,
        fixed_times=np.asarray(fixed_times), fixed_start_times=np.asarray(fixed_start_times),
        fixed_amounts=np.asarray(fixed_amounts),
        float_pay_times=np.asarray(float_pay), float_start_times=np.asarray(float_start),
        float_end_times=np.asarray(float_end), float_accrual=np.asarray(float_accrual),
        float_spread=cfg.floating_spread,
        rate_factor_index=cfg.rate_factor_index, hw_a=cfg.hw_a, hw_sigma=cfg.hw_sigma,
        zero_times=np.asarray(cfg.initial_zero_curve.times, dtype=np.float64),
        zero_rates=np.asarray(cfg.initial_zero_curve.rates, dtype=np.float64),
        n_per_std=cfg.n_per_std, std_devs=cfg.std_devs,
        final_maturity=final_maturity,
    )


def _zero_curve_of(swap: _PreparedBermudan) -> _HwZeroCurve:
    """Builds the `ZeroCurve` `_run_backward_induction` prices against.

    Deliberately does NOT hardcode a dtype the way this used to (a bare
    `jnp.asarray(..., dtype=jnp.float64)`): `swap.zero_rates` is a plain
    `np.ndarray` for ordinary (non-Greeks) pricing (`prepare_bermudan`
    always builds it that way), but `engine.risk.greeks._bermudan_price_fn`
    substitutes a differentiable JAX array there (via `dataclasses.replace`)
    to carry the `risk`-precision `pillar_rates` a caller's `PrecisionConfig`
    requested -- a hardcoded cast here would silently upcast that array
    back to float64 regardless, breaking the `risk` knob for Bermudan/
    American Greeks specifically. `jnp.asarray` on an already-JAX array with
    no explicit `dtype` is a no-op (preserves whatever dtype it already
    has); on a plain `np.ndarray` it defaults to that array's own NumPy
    dtype (float64, from `prepare_bermudan`'s own construction) -- so both
    callers get exactly the dtype they need with no explicit branching."""
    return _HwZeroCurve(
        pillar_times=jnp.asarray(swap.zero_times),
        pillar_rates=jnp.asarray(swap.zero_rates),
    )


# =============================================================================
# STATE GRID / HAGAN QUADRATURE CONVOLUTION
#
# Mirrors QuantExt::LgmConvolutionSolver2 (QuantExt/qle/models/
# lgmconvolutionsolver2.hpp/.cpp), using LGM's OWN state variable x(t) --
# NOT this codebase's direct short-rate parametrization r(t) used elsewhere
# (simulation, swap, european_swaption). This is a deliberate, verified
# departure from the "reuse the direct-r parametrization everywhere"
# pattern the rest of this codebase follows -- see engine/models/lgm.py's
# docstring for the full reasoning (x(t) is driftless, which is what makes
# Hagan's quadrature convolution valid at all).
# =============================================================================
def _state_grid(sigma: float, t: jax.Array, n_per_std: int, std_devs: float, dtype=jnp.float64) -> jax.Array:
    """
    The centered LGM state grid at time t: `x_k = k*dx`, `dx =
    sqrt(zeta(t)) / n_per_std`, spanning `+/- std_devs` standard deviations
    -- exactly `LgmConvolutionSolver2::stateGrid`'s `dx = sqrt(zeta(t))/nx_`
    construction, with `mx_ = round(std_devs * n_per_std)` points on each
    side of zero, always including x=0 itself (an odd-length grid, matching
    ORE's `2*mx_+1` point count). `mx` is a Python int (a config-time
    constant, not data-dependent), so this function's OUTPUT SHAPE is
    always `2*mx+1` regardless of `t` -- required for `jax.lax.scan`, whose
    carry must have a fixed shape at every step.

    At t=0, zeta(0)=0 and the grid collapses to the single value x=0
    EVERYWHERE (not just index mx) -- matching
    `LgmConvolutionSolver2::stateGrid`'s explicit `t=0` special case
    (`if (close_enough(t,0.0)) return RandomVariable(2*mx_+1, 0.0);`).

    **Gradient-safe at zeta==0** (not just forward-value-safe): `d(sqrt(z))
    /dz` is itself a 0/0 indeterminate form at `z=0` (`sqrt`'s own
    derivative, `0.5/sqrt(z)`, diverges as `z->0+`, and JAX's `sqrt` JVP
    rule evaluates that formula literally), so `jax.grad`/`jax.hessian`
    with respect to `sigma` through a naive `jnp.sqrt(zeta)` produces NaN
    at t=0 even though the FORWARD value (0.0) is perfectly well-defined
    -- confirmed directly (`engine.risk.greeks.bermudan_vega`'s own
    development surfaced this: `d(NPV)/d(sigma_values[0])` came back NaN
    until this guard was added, the first caller in this codebase to
    differentiate through `_state_grid` at all). Guarded via the standard
    branch-free `jnp.where` pattern used throughout `engine.models.
    hull_white`/`engine.models.lgm` (evaluate `sqrt` on a safe placeholder
    that is never actually 0, then select the correct branch) -- both
    branches are always computed (required for `jax.jit`/`jax.grad`
    tracing), the placeholder result is simply discarded when `zeta > 0`.

    `dtype`: an EXPLICIT parameter, not derived from `sigma`/`t` -- this
    function (via `_run_backward_induction`) is shared between plain
    Bermudan/American pricing (which must stay float64-internal regardless
    of `PrecisionConfig.risk`, governed only by `PrecisionConfig.pricing`
    -- and, per `price_bermudan_swaptions`' own final `hw_paths.dtype`
    cast, is actually pricing-precision-agnostic internally either way) and
    `engine.risk.greeks.bermudan_vega`/`bermudan_delta_gamma` (which must
    honor `PrecisionConfig.risk`). `_run_backward_induction` passes
    `curve.pillar_rates.dtype` here -- already the correct dtype for both
    callers, since `_zero_curve_of` derives it from `swap.zero_rates`
    (plain `np.ndarray`, hence float64, for ordinary pricing; whatever
    `risk`-dtype JAX array Greeks substituted in otherwise) -- rather than
    blindly deriving from `sigma`, which would be wrong: `t`/`sigma` are
    the SAME hardcoded-float64 `grid_times`/`sigma` pairing whether or not
    Greeks are in play, so deriving a "requested" dtype from them here
    would either always read float64 (no risk=32 effect at all) or require
    yet another parameter threaded from further up -- an explicit
    caller-supplied dtype is the simplest correct fix.
    """
    mx = int(round(std_devs * n_per_std))
    z = jnp.maximum(_lgm_zeta(sigma, t), 0.0)
    z_safe = jnp.where(z > 0.0, z, 1.0)
    dx = jnp.where(z > 0.0, jnp.sqrt(z_safe), 0.0) / n_per_std
    return dx * jnp.arange(-mx, mx + 1, dtype=dtype)


def _hagan_quadrature_weights(n_per_std: int, std_devs: float) -> np.ndarray:
    """
    Hagan's closed-form trapezoid-of-normal-density quadrature weights on a
    fixed standardized y-grid (`y_i = i/n_per_std`, spanning `+/-std_devs`
    standard deviations) -- exactly `LgmConvolutionSolver2`'s constructor
    (lgmconvolutionsolver2.cpp lines 25-58), which piecewise-linearly
    interpolates the value function between grid nodes and integrates that
    against the exact Gaussian transition density in closed form:

        w_i = (1 + y_i/h)*N(y_i+h) - 2*(y_i/h)*N(y_i) - (1 - y_i/h)*N(y_i-h)
              + (G(y_i+h) - 2*G(y_i) + G(y_i-h)) / h

    where N is the standard normal CDF, G is the standard normal PDF, and
    h = 1/n_per_std is the standardized grid spacing -- boundary-adjusted at
    the first/last node (ORE's own i=0/i=2*my_ special cases), where the
    outer half-interval has no neighbor to interpolate against and the
    weight reduces to the plain CDF-difference (flat extrapolation beyond
    the grid).

    A one-time, config-only (not per-scenario/per-step) precomputation --
    plain NumPy/SciPy is appropriate here, not a JAX tracing concern.
    """
    from scipy.stats import norm as scipy_norm

    h = 1.0 / n_per_std
    my = int(round(std_devs * n_per_std))
    y = h * np.arange(-my, my + 1, dtype=np.float64)
    n = y.shape[0]
    w = np.zeros(n, dtype=np.float64)

    Ncdf = scipy_norm.cdf
    Npdf = scipy_norm.pdf

    # Interior nodes: full Hagan closed-form weight (a value function that
    # is piecewise-linear between y_{i-1}, y_i, y_{i+1}, integrated exactly
    # against the standard normal density).
    for i in range(1, n - 1):
        yi = y[i]
        term_cdf = ((1.0 + yi / h) * Ncdf(yi + h)
                    - 2.0 * (yi / h) * Ncdf(yi)
                    - (1.0 - yi / h) * Ncdf(yi - h))
        term_pdf = (Npdf(yi + h) - 2.0 * Npdf(yi) + Npdf(yi - h)) / h
        w[i] = term_cdf + term_pdf

    # Boundary nodes: only one neighbor exists, so the value function is
    # taken as flat beyond the grid edge (the same flat-extrapolation
    # convention _rollback_one_step's jnp.interp applies) -- the weight is
    # the mass of the standard normal density assigned to that flat outer
    # region plus the linear-interpolation contribution from the one
    # existing inner neighbor, i.e. the i==0 / i==n-1 special-case formulas
    # in LgmConvolutionSolver2's constructor.
    y0 = y[0]
    w[0] = Ncdf(y0 + h) - (y0 / h) * (Ncdf(y0 + h) - Ncdf(y0)) - (Npdf(y0 + h) - Npdf(y0)) / h
    yN = y[-1]
    w[-1] = (1.0 - Ncdf(yN - h)) + (yN / h) * (Ncdf(yN) - Ncdf(yN - h)) - (Npdf(yN) - Npdf(yN - h)) / h
    return w


def _rollback_one_step(
    values: jax.Array, x_from: jax.Array, x_to: jax.Array,
    quad_y: jax.Array, quad_w: jax.Array, std_from_to: jax.Array,
) -> jax.Array:
    """
    E[values(x_from) | x_to] under the model's exact Gaussian transition
    law, evaluated at every point of `x_to` simultaneously, reproducing
    `LgmConvolutionSolver2::rollback`.

    For each target node `x_to[k]`, the conditional distribution of
    `x_from` is Gaussian with mean `x_to[k]` (LGM/HW1F's own state variable
    is driftless in this deviation parametrization -- the SAME identity
    docs/reference/ore-parity.md section 3a already establishes from
    `IrLgm1fStateProcess::expectation()`) and standard deviation
    `std_from_to = sqrt(zeta(t_from) - zeta(t_to))`. Hagan's quadrature
    re-expresses `E[f(x_from)] = sum_i w_i * f(x_to[k] + y_i*std_from_to)`
    for the precomputed standardized nodes/weights `quad_y`/`quad_w`; each
    query point is linearly interpolated into the `x_from` grid (flat
    outside its range) exactly as `LgmConvolutionSolver2::rollback` does.
    """
    query = x_to[..., None] + quad_y[None, :] * std_from_to  # [..., n_to, n_quad]
    interpolated = jnp.interp(
        query.reshape(-1), x_from, values,
        left=values[0], right=values[-1],
    ).reshape(query.shape)
    return jnp.sum(interpolated * quad_w[None, :], axis=-1)


def _discount_at_nodes(curve: _HwZeroCurve, x_nodes: jax.Array, t: jax.Array, times: jax.Array, alive_mask: jax.Array, a: float, sigma: float) -> jax.Array:
    """P(t, times; x_nodes) at every LGM state-grid node, for every cashflow
    time, via `engine.models.lgm.bond_price` (this module's own live-
    verified LGM bond formula -- see `engine.models.lgm`'s docstring for
    why this module does NOT reuse `engine.models.hull_white`, a different
    model realization for t>0) -- zeroed out for any cashflow already
    paid/expired, matching NumericLgmMultiLegOptionEngineBase's cashflow
    bookkeeping (a coupon is folded into the running NPV once and never
    revisited once its own time passes).

    `alive_mask` is precomputed by the caller (see `_hw_swap_value_at_nodes`)
    using the SAME `>= t - 1e-9` tolerance the original NumPy engine used --
    a coupon whose start/end/pay time lands EXACTLY on the exercise/
    conditioning time `t` is the common case for a reset-aligned exercise
    date (not an edge case), and a stricter zero-tolerance comparison here
    previously zeroed out P(t, t; x) for that coupon's own accrual-start
    date even though the caller's mask had already classified it as alive,
    corrupting that coupon's forward-rate ratio (found and fixed during
    this module's own test-suite development, see
    tests/test_bermudan_swaption.py::TestMidCouponKnownLimitation).

    times/alive_mask may be empty (shape `[0]`) for a swap leg with zero
    cashflows still outstanding -- handled by the caller via `jnp.where`
    rather than a Python-level shape branch, so this function itself
    assumes non-empty inputs and is `jax.jit`-safe."""
    P = jax.vmap(lambda T: _lgm_bond_price(curve, a, sigma, t, T, x_nodes))(times)  # [Ntimes, Nnodes]
    return (P * alive_mask[:, None]).T  # [Nnodes, Ntimes]


def _hw_swap_value_at_nodes(swap: _PreparedBermudan, curve: _HwZeroCurve, x_nodes: jax.Array, t: jax.Array) -> jax.Array:
    """
    The underlying swap's own remaining NPV (float leg - fixed leg,
    payer-signed) at time t, evaluated at every LGM state-grid node in
    `x_nodes` -- the "intrinsic" (exercise) value used at each exercise
    date, following the same fixed-leg/floating-leg NPV structure as
    engine.instruments.swap._price_one_swap, just evaluated at model-grid
    nodes instead of simulated scenario paths.

    fixed_amounts is used directly (notional*rate*accrual, read straight
    from ORE's own FixedRateCoupon.amount() in prepare_bermudan) rather
    than re-deriving rate*accrual -- avoids re-deriving a day-count detail
    ORE's own schedule generation has already resolved exactly.
    """
    a, sigma = swap.hw_a, swap.hw_sigma
    # Derived from x_nodes' own dtype (not hardcoded) -- x_nodes is
    # `_state_grid`'s output, already correctly float64 for ordinary
    # pricing or risk-dtype for Greeks (see _state_grid's docstring); these
    # trade-schedule arrays (fixed_times/fixed_amounts/etc., read straight
    # off `_PreparedBermudan`, always plain np.ndarray/float64 regardless of
    # caller) would otherwise silently upcast x_nodes/curve back to float64
    # under jax_enable_x64=True the moment they're combined via
    # _discount_at_nodes below.
    dtype = x_nodes.dtype

    fixed_times = jnp.asarray(swap.fixed_times, dtype=dtype)
    fixed_start_times = jnp.asarray(swap.fixed_start_times, dtype=dtype)
    fixed_amounts = jnp.asarray(swap.fixed_amounts, dtype=dtype)
    # Same accrual-start-based liveness rule as the floating leg below: a
    # fixed coupon is only a genuine remaining cashflow once its OWN
    # accrual period has not yet begun relative to t, which is exact (not
    # an approximation) at any reset-aligned t within this module's
    # documented scope.
    fixed_alive = (fixed_start_times >= t - 1e-9).astype(dtype)
    fixed_disc = _discount_at_nodes(curve, x_nodes, t, fixed_times, fixed_alive, a, sigma)  # [Nnodes, Nf]
    fixed_leg_pv = fixed_disc @ fixed_amounts  # [Nnodes]

    float_pay_times = jnp.asarray(swap.float_pay_times, dtype=dtype)
    float_start_times = jnp.asarray(swap.float_start_times, dtype=dtype)
    float_end_times = jnp.asarray(swap.float_end_times, dtype=dtype)
    float_accrual = jnp.asarray(swap.float_accrual, dtype=dtype)

    # Only coupons whose accrual has NOT YET BEGUN (start >= t) are
    # included -- P(t, accrual_start) via the closed-form LGM bond formula
    # is only a genuine discount factor for accrual_start >= t (the same
    # T<t clamping issue documented as a known limitation in
    # engine.instruments.swap's module docstring). This is not an
    # approximation HERE specifically because every exercise/condition
    # time this module ever evaluates this function at is, by this
    # module's own documented scope (BermudanSwaptionConfig's docstring),
    # a reset/accrual-start date of the underlying swap -- so at any such
    # t, every floating coupon either has start >= t (not yet begun,
    # correctly priced) or start < t only for a coupon whose OWN start was
    # a prior, already-passed reset date entirely (which, at a
    # reset-aligned t, coincides with end <= t too, i.e. it is a fully
    # elapsed coupon that must be excluded from the swap's remaining value
    # in any case -- consistent with, not a workaround of, the exclusion).
    float_alive = (float_start_times >= t - 1e-9).astype(dtype)
    p_start = _discount_at_nodes(curve, x_nodes, t, float_start_times, float_alive, a, sigma)  # [Nnodes, Ncf]
    p_end = _discount_at_nodes(curve, x_nodes, t, float_end_times, float_alive, a, sigma)
    p_end_safe = jnp.where(p_end == 0.0, 1.0, p_end)
    forward_rate = jnp.where(float_alive[None, :] > 0.0, (p_start / p_end_safe - 1.0) / float_accrual[None, :], 0.0)
    float_pay_disc = _discount_at_nodes(curve, x_nodes, t, float_pay_times, float_alive, a, sigma)
    float_cashflow = swap.notional * (forward_rate + swap.float_spread) * float_accrual[None, :]
    float_leg_pv = jnp.sum(float_cashflow * float_pay_disc * float_alive[None, :], axis=1)  # [Nnodes]

    npv = float_leg_pv - fixed_leg_pv
    return npv if swap.payer else -npv


# =============================================================================
# THE GRID-TIME SCHEDULE (precomputed once per trade, plain Python)
# =============================================================================
@dataclass(frozen=True, eq=False)
class _GridSchedule(StaticKeyMixin):
    """The full, fixed-length, descending-sorted list of times the backward
    induction walks through -- `{0, final_maturity} ∪ exercise_times ∪
    condition_times`, deduplicated. Built once per `(trade, condition_times)`
    pair in plain Python/NumPy (a static, config-time computation -- this
    is exactly what the original NumPy engine's own `set()`/`round()`-based
    construction did, just done ONCE outside the JAX trace here rather than
    per-call inside a Python loop, so the resulting array's LENGTH is a
    genuine Python int JAX can treat as a static shape)."""
    times: np.ndarray            # [G] descending
    is_exercise: np.ndarray      # [G] bool
    is_condition: np.ndarray     # [G] bool
    condition_index: np.ndarray  # [G] int, index into the original condition_times list (-1 if not a condition time)


def _build_grid_schedule(swap: _PreparedBermudan, condition_times: Sequence[float]) -> _GridSchedule:
    round_dp = 12
    exercise_set = set(round(float(t), round_dp) for t in swap.exercise_times)
    condition_list = [round(float(t), round_dp) for t in condition_times]
    condition_set = set(condition_list)
    all_times = sorted(set([0.0, swap.final_maturity]) | exercise_set | condition_set, reverse=True)

    condition_first_index = {}
    for i, t in enumerate(condition_list):
        condition_first_index.setdefault(t, i)

    times = np.asarray(all_times, dtype=np.float64)
    is_exercise = np.asarray([t in exercise_set for t in all_times], dtype=bool)
    is_condition = np.asarray([t in condition_set for t in all_times], dtype=bool)
    condition_index = np.asarray(
        [condition_first_index.get(t, -1) for t in all_times], dtype=np.int64
    )
    return _GridSchedule(times=times, is_exercise=is_exercise, is_condition=is_condition, condition_index=condition_index)


# =============================================================================
# BACKWARD INDUCTION (jax.lax.scan)
# =============================================================================
@dataclass
class _RolledBackValue:
    """The result of running backward induction on a prepared Bermudan/
    American swaption: the state grid and option-value function at time 0
    (a single node, x=0), plus everything needed to evaluate the SAME
    rolled-back value function conditional on an arbitrary simulated short
    rate at any of the requested `condition_times` (see
    price_bermudan_swaptions)."""
    value_at_t0: jax.Array  # 0-d JAX scalar, NOT a plain float -- kept as a
    # traced array so engine.risk.greeks can differentiate straight through
    # it (jax.grad/jax.hessian w.r.t. the curve/sigma this was computed
    # from); price_bermudan_swaption_base (the plain-float production
    # entry point) does its OWN float() cast on this value, one layer
    # further out, so ordinary (non-Greeks) callers see no behavior change.
    condition_times: np.ndarray
    condition_state_grids: List[np.ndarray]
    condition_values: List[np.ndarray]


def _run_backward_induction(swap: _PreparedBermudan, condition_times: Sequence[float]) -> _RolledBackValue:
    """
    The core Hagan-convolution backward induction (module docstring steps
    2-6), run once per prepared swaption, as a single `jax.lax.scan` over
    a precomputed, fixed-length grid-time schedule (`_build_grid_schedule`).

    **Runs entirely in numeraire-deflated units** (`reduced = value / N(t,x)`
    at every grid time) -- required for Hagan's quadrature convolution to be
    valid at all: LGM's state x(t) is driftless, so `E[reduced(x_from) |
    x_to]` under x's own Gaussian transition law is exactly the deflated
    value at `x_to` (a true martingale identity, matching
    `QuantExt::LinearGaussMarkovModel::reducedDiscountBond`'s whole reason
    for existing -- see docs/reference/ore-parity.md's "parametrization note").
    Rolling back RAW (non-deflated) values under x's transition law would
    silently give the wrong answer, since a raw bond/swap price is NOT a
    martingale under x's driftless transition on its own (only the
    numeraire-deflated price is). The early-exercise
    `max(intrinsic, continuation)` comparison is applied in deflated units
    too (dividing the intrinsic swap value by the SAME N(t,x) the
    continuation value is already deflated by) -- valid because N(t,x) > 0
    always, so `max` commutes with the deflation.

    Every grid step's (raw, re-inflated) value function is stacked into the
    scan's output (fixed shape: `[NumGridTimes, NumStateGridPoints]`), then
    the caller selects out condition-time rows via `schedule.is_condition`/
    `condition_index` -- this replaces the original NumPy engine's dict-
    based snapshot bookkeeping with a plain array select, since
    `jax.lax.scan` can only produce fixed-shape stacked outputs, not a
    variable-size dict.
    """
    schedule = _build_grid_schedule(swap, condition_times)
    num_grid = len(schedule.times)
    # The array-producing core (state grids + the lax.scan rollback) is
    # factored out below; the NumPy snapshot bookkeeping stays here, since it
    # concretizes (np.asarray) and indexes with Python ints.
    #
    # Deliberately NOT jax.jit'd with `swap`/`schedule` as static arguments,
    # unlike the swap/European-swaption pricers' own `_Prepared*` structures:
    # `engine.risk.greeks` differentiates Bermudan/American Delta/Gamma/Vega
    # straight THROUGH this function, calling it with a `_PreparedBermudan`
    # whose `zero_rates` (and, for Vega, `hw_sigma`) are live `jax.grad`
    # tracers rather than concrete arrays. A jit static argument must be
    # hashable and concrete, so a tracer-carrying `_PreparedBermudan` can
    # never be one -- and `hw_sigma` may be a `Sigma`, a JAX pytree of
    # arrays, which is unhashable for the same reason. The `lax.scan` below
    # is itself a single fused primitive, so the eager-dispatch cost here is
    # far lower than it would be for an unfused elementwise pricer.
    x_all, values_all = _backward_induction_arrays(swap, schedule)
    grid_times = jnp.asarray(schedule.times, dtype=_zero_curve_of(swap).pillar_rates.dtype)
    curve = _zero_curve_of(swap)
    a, sigma = swap.hw_a, swap.hw_sigma

    condition_state_grids: List[np.ndarray] = []
    condition_values: List[np.ndarray] = []
    if len(condition_times) > 0:
        order = np.argsort([schedule.condition_index[i] for i in range(num_grid) if schedule.is_condition[i]])
        cond_rows = np.nonzero(schedule.is_condition)[0][order]
        for row in cond_rows:
            # Convert the LGM state grid x to the corresponding short rate
            # r BEFORE storing the snapshot -- price_bermudan_swaptions
            # below interpolates each scenario's SIMULATED short rate
            # r_t (engine.simulation's own r(t) parametrization) directly
            # against this stored grid, so the grid must already be in
            # r-space, not raw LGM state x-space (r(t,x) is affine in x --
            # see engine.models.lgm.r_from_x -- so this conversion is
            # exact, not an approximation).
            r_grid = _lgm_r_from_x(curve, a, sigma, grid_times[row], x_all[row])
            condition_state_grids.append(np.asarray(r_grid))
            condition_values.append(np.asarray(values_all[row]))

    t0_row = int(np.nonzero(np.isclose(schedule.times, 0.0))[0][0])
    value_at_t0 = values_all[t0_row, x_all.shape[1] // 2]

    return _RolledBackValue(
        value_at_t0=value_at_t0,
        condition_times=np.asarray(condition_times, dtype=np.float64),
        condition_state_grids=condition_state_grids,
        condition_values=condition_values,
    )


def _backward_induction_arrays(swap: _PreparedBermudan, schedule: "_GridSchedule"):
    """The array core of `_run_backward_induction`: builds the LGM state grid
    at every scheduled grid time and rolls the value function backward
    through them with one `jax.lax.scan`, returning the stacked
    `(x_all, values_all)` of shape `[NumGridTimes, NumStateGridPoints]`.

    Kept as a separate function purely for readability -- see the caller on
    why this is deliberately not `jax.jit`-wrapped (it must stay traceable by
    `jax.grad` with tracer-carrying arguments). The `np.asarray`-based
    snapshot selection stays in the caller for the complementary reason: it
    concretizes values and indexes with Python ints.
    """
    a, sigma, n_per_std, std_devs = swap.hw_a, swap.hw_sigma, swap.n_per_std, swap.std_devs
    curve = _zero_curve_of(swap)
    # Derived from curve's own dtype (not hardcoded) -- see _state_grid's
    # docstring for why this is the correct signal to key off of: float64
    # for ordinary pricing (curve.pillar_rates is always a plain np.ndarray
    # there), the requested risk-precision dtype for engine.risk.greeks'
    # Bermudan/American Delta/Gamma/Vega/Theta. Without this, quad_w/quad_y/
    # grid_times being hardcoded float64 would silently upcast a risk=32
    # Greeks computation back to float64 under jax_enable_x64=True the
    # moment they combine with sigma/x -- confirmed directly (a float32
    # array times a hardcoded-float64 array promotes to float64 whenever
    # x64 is enabled, regardless of which operand is which dtype).
    dtype = curve.pillar_rates.dtype
    quad_w = jnp.asarray(_hagan_quadrature_weights(n_per_std, std_devs), dtype=dtype)
    my = int(round(std_devs * n_per_std))
    quad_y = jnp.asarray((1.0 / n_per_std) * np.arange(-my, my + 1, dtype=np.float64), dtype=dtype)

    grid_times = jnp.asarray(schedule.times, dtype=dtype)
    is_exercise = jnp.asarray(schedule.is_exercise)

    x0 = _state_grid(sigma, grid_times[0], n_per_std, std_devs, dtype=dtype)
    values0 = _hw_swap_value_at_nodes(swap, curve, x0, grid_times[0])
    values0 = jnp.where(is_exercise[0], jnp.maximum(values0, values0), values0)  # exercise at the first (latest) grid time is a no-op maximum with itself; kept for structural symmetry with the loop body
    numeraire0 = _lgm_numeraire(curve, a, sigma, grid_times[0], x0)
    reduced0 = values0 / numeraire0

    def step(carry, step_inputs):
        x_prev, t_prev, reduced_prev = carry
        t, is_ex = step_inputs

        x_t = _state_grid(sigma, t, n_per_std, std_devs, dtype=dtype)
        # Same gradient-safe sqrt guard as _state_grid (see that function's
        # docstring): sqrt's own derivative is a 0/0 indeterminate form at
        # 0, and even though the jnp.where below already selects the
        # OTHER branch's forward value whenever this variance is 0,
        # jax.grad still backpropagates through BOTH branches (0 * NaN =
        # NaN) unless std_step itself is computed off a safe placeholder.
        var_step = jnp.maximum(_lgm_zeta(sigma, t_prev) - _lgm_zeta(sigma, t), 0.0)
        var_step_safe = jnp.where(var_step > 0.0, var_step, 1.0)
        std_step = jnp.where(var_step > 0.0, jnp.sqrt(var_step_safe), 0.0)

        reduced_continuation = jnp.where(
            std_step > 0.0,
            _rollback_one_step(reduced_prev, x_prev, x_t, quad_y, quad_w, std_step),
            jnp.interp(x_t, x_prev, reduced_prev),
        )

        numeraire_t = _lgm_numeraire(curve, a, sigma, t, x_t)
        intrinsic = _hw_swap_value_at_nodes(swap, curve, x_t, t)
        reduced_intrinsic = intrinsic / numeraire_t
        reduced_t = jnp.where(is_ex, jnp.maximum(reduced_continuation, reduced_intrinsic), reduced_continuation)

        raw_values_t = reduced_t * numeraire_t
        new_carry = (x_t, t, reduced_t)
        return new_carry, (x_t, raw_values_t)

    _, (x_scan, values_scan) = jax.lax.scan(
        step, (x0, grid_times[0], reduced0), (grid_times[1:], is_exercise[1:]),
    )
    # Prepend the first (latest) grid time's own values so the stacked
    # output covers every grid time, matching schedule.times' own indexing.
    x_all = jnp.concatenate([x0[None, :], x_scan], axis=0)          # [G, Nnodes]
    values_all = jnp.concatenate([values0[None, :], values_scan], axis=0)  # [G, Nnodes]
    return x_all, values_all


def price_bermudan_swaption_base(cfg: BermudanSwaptionConfig) -> float:
    """t=0 NPV of a single Bermudan swaption (no simulated conditioning) --
    the value read off the backward induction's own x=0 node, exactly
    LgmConvolutionSolver2::stateGrid(0)'s single-point convention. Casts
    `_RolledBackValue.value_at_t0` (kept as a JAX scalar internally, so
    `engine.risk.greeks` can differentiate through `_run_backward_induction`
    directly) to a plain Python float here, at this plain-pricing entry
    point only."""
    swap = prepare_bermudan(cfg)
    result = _run_backward_induction(swap, condition_times=[])
    return float(result.value_at_t0)


def price_bermudan_swaptions(
    bermudan_configs: List[BermudanSwaptionConfig],
    hw_paths: jax.Array,
    step_times: jax.Array,
) -> jax.Array:
    """
    hw_paths: [Scenarios, TimeSteps, NumHW], typically
        engine.simulation.generate_paths(...)["rates"].
    step_times: [TimeSteps] absolute simulation times (year-fractions from
        evaluation_date).
    Returns: [Scenarios, TimeSteps, Trades] NPV cube, conditioning each
        trade's own rolled-back value function on the simulated short rate
        at every (scenario, step) pair -- the same Markov-conditioning
        approach engine.instruments.european_swaption uses (see that
        module's docstring), except here the value function conditioned on
        must itself come from a full backward induction run out to each
        requested step_time (an option with remaining early-exercise
        opportunities cannot be evaluated at an arbitrary future time from
        a single t=0 rollback the way Jamshidian's closed form can -- its
        value depends on the entire remaining exercise schedule).

    Steps at or after a trade's LAST exercise time are priced as exactly 0
    (matching this codebase's European swaption convention of reporting 0
    NPV after an option's own final exercise opportunity -- ORE's own
    Instrument.NPV() convention).
    """
    step_times_np = np.asarray(step_times, dtype=np.float64)
    per_trade = []
    for cfg in bermudan_configs:
        swap = prepare_bermudan(cfg)
        r_t = np.asarray(hw_paths[:, :, cfg.rate_factor_index])  # [S, T]

        last_exercise = float(swap.exercise_times[-1])
        condition_steps = [t for t in step_times_np if t < last_exercise]
        result = _run_backward_induction(swap, condition_times=condition_steps)

        npv = np.zeros_like(r_t)
        for i, t in enumerate(step_times_np):
            if t >= last_exercise:
                continue
            j = condition_steps.index(t)
            x_grid, v_grid = result.condition_state_grids[j], result.condition_values[j]
            npv[:, i] = np.interp(r_t[:, i], x_grid, v_grid, left=v_grid[0], right=v_grid[-1])

        per_trade.append(jnp.asarray(npv, dtype=hw_paths.dtype))
    return jnp.stack(per_trade, axis=-1)


# =============================================================================
# EXECUTION DEMONSTRATION
# =============================================================================
if __name__ == "__main__":
    from engine.simulation.market_model import generate_paths
    from engine.simulation.demo_scenarios import EVAL_DATE, swaption_demo_config

    config = swaption_demo_config()
    market_cubes = generate_paths(config)
    step_times = jnp.array(config.time_grid[1:], dtype=jnp.float64)

    zero_curve = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[0.03] * 6)

    bermudan_cfg = BermudanSwaptionConfig(
        notional=1_000_000.0,
        fixed_rate=0.030,
        payer=True,
        rate_factor_index=0,
        hw_a=config.rates.mean_reversion[0],
        hw_sigma=float(np.sqrt(config.joint_covariance[1][1])),
        initial_zero_curve=zero_curve,
        exercise_times=[1.0, 2.0, 3.0, 4.0],
        swap_tenor="5Y",
    )
    print("Bermudan t=0 NPV:", price_bermudan_swaption_base(bermudan_cfg))

    npv_cube = price_bermudan_swaptions([bermudan_cfg], market_cubes["rates"], step_times)
    print("Bermudan NPV cube shape:", npv_cube.shape)
    for i, t in enumerate(config.time_grid[1:]):
        print(f"  t={t:.2f}: mean NPV across scenarios = {float(jnp.mean(npv_cube[:, i, 0])):.2f}")
