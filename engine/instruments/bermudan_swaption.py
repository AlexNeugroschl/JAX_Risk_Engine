"""
Bermudan swaption pricing via a numeric LGM short-rate grid, using Hagan's
Gaussian-quadrature convolution scheme. This is also the engine behind
American swaption pricing -- see engine.instruments.american_swaption, a
thin wrapper around this module (American exercise is priced here as a
finely-discretized Bermudan, exactly matching ORE's own design).

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
docs/10-american-swaptions.md for the full writeup.

Algorithm (see docs/10-american-swaptions.md for full derivation and every
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
   exp(-(H(T)-H(t))*x)` at every state-grid node -- `_lgm_bond`, live-
   verified to machine precision against `ORE.LinearGaussMarkovModel.
   discountBond`). **Deliberately NOT** this codebase's other HW1F closed
   form (`compute_hw_A`/`_hw_B`, used throughout
   engine.instruments.european_swaption and swap) -- see `_lgm_bond`'s own
   docstring for why: `ORE.HullWhite` and `ORE.LinearGaussMarkovModel`,
   despite sharing (a, sigma) and today's curve, are NOT the same numerical
   model realization for t>0, a finding made and verified live while
   building this module. Since ORE's actual Bermudan/American engine is
   built on `LinearGaussMarkovModel`, this module matches THAT model
   exclusively, not the plain-HullWhite closed forms used elsewhere in this
   codebase.
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

**Validation approach** (see docs/10-american-swaptions.md and
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
      P(t,T,x) itself (`_lgm_bond`) -- is live-verified to machine
      precision against `ORE.IrLgm1fConstantParametrization`/
      `ORE.LinearGaussMarkovModel` directly.
  (b) A single-exercise-date Bermudan (mathematically == a European
      swaption) must reproduce an INDEPENDENT Jamshidian-style closed-form
      decomposition built on the SAME `_lgm_bond` formula (not this
      codebase's HullWhite-parametrized Jamshidian pricer in
      european_swaption.py, which was confirmed -- while building this
      module -- to be a genuinely different model realization for t>0; see
      `_lgm_bond`'s docstring) -- verified to match to ~1e-7 relative or
      better, confirming the backward-induction/convolution machinery,
      run with only one exercise opportunity, collapses to the correct
      closed form.
  (c) Monotonicity: a Bermudan/American swaption must be worth at least as
      much as the otherwise-identical European swaption exercisable only at
      the LAST of its dates (more exercise opportunities cannot decrease an
      option's value) -- a model-independent no-arbitrage bound, checked
      numerically.
  (d) Grid convergence: refining the state-grid resolution
      (n_per_std, std_devs) must change the price by a shrinking amount
      (standard numerical-scheme convergence check).
"""
from dataclasses import dataclass, field
from typing import List, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import ORE

from engine.simulation import ZeroCurveConfig

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
    see docs/10-american-swaptions.md's "Scope" section) coincide with one
    of the underlying's own accrual dates, matching ORE's own standard
    "coterminal" Bermudan structure (each exercise date is a fixed-leg reset
    date) -- ORE's own calibration machinery
    (SwaptionEngineBuilder::model()) is built around exactly this
    coterminal-date assumption for Bermudan/American calibration baskets.

    rate_factor_index/hw_a/hw_sigma/initial_zero_curve: same meaning and
    same single-model-pricing rationale as
    engine.instruments.european_swaption.SwaptionConfig -- see that
    module's docstring.

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
    hw_sigma: float
    initial_zero_curve: ZeroCurveConfig
    exercise_times: Sequence[float]
    swap_tenor: str = "5Y"
    index_tenor_months: int = 6
    floating_spread: float = 0.0
    n_per_std: int = 48
    std_devs: float = 6.0
    evaluation_date: ORE.Date = field(default_factory=lambda: ORE.Settings.instance().evaluationDate)


def _build_ore_swap(cfg) -> ORE.VanillaSwap:
    """CPU: builds the real ORE underlying swap -- identical pattern to
    swap._build_ore_swap / european_swaption._build_ore_swap."""
    ORE.Settings.instance().evaluationDate = cfg.evaluation_date
    dummy_forward_curve = ORE.YieldTermStructureHandle(
        ORE.FlatForward(cfg.evaluation_date, 0.0, DAY_COUNTER)
    )
    index = ORE.IborIndex(
        "SimIndex", ORE.Period(cfg.index_tenor_months, ORE.Months), 2,
        ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False,
        DAY_COUNTER, dummy_forward_curve,
    )
    swap_type = ORE.VanillaSwap.Payer if cfg.payer else ORE.VanillaSwap.Receiver
    swap = ORE.MakeVanillaSwap(
        ORE.Period(cfg.swap_tenor), index, cfg.fixed_rate,
        nominal=cfg.notional,
        swapType=swap_type,
        floatingLegSpread=cfg.floating_spread,
        fixedLegDayCount=DAY_COUNTER,
        floatingLegDayCount=DAY_COUNTER,
    )
    return swap


@dataclass
class _PreparedBermudan:
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
    hw_sigma: float
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


# =============================================================================
# STATE GRID / HAGAN QUADRATURE CONVOLUTION
#
# Mirrors QuantExt::LgmConvolutionSolver2 (QuantExt/qle/models/
# lgmconvolutionsolver2.hpp/.cpp), using LGM's OWN state variable x(t) --
# NOT this codebase's direct short-rate parametrization r(t) used elsewhere
# (simulation, swap, european_swaption). This is a deliberate, verified
# departure from the "reuse the direct-r parametrization everywhere"
# pattern the rest of this codebase follows, for a precise reason: LGM's
# x(t) is DRIFTLESS (confirmed directly from
# `IrLgm1fStateProcess::expectation()` in docs/09-ore-parity.md section 3a),
# so its transition law needs no mean-reversion decay term and its
# NUMERAIRE-DEFLATED bond price is an exact martingale under x's own
# Gaussian transition -- both properties Hagan's convolution scheme relies
# on. The direct short rate r(t) is NOT driftless (it mean-reverts to
# theta), so naively convolving raw (non-deflated) payoffs under r(t)'s own
# transition law does NOT reproduce the correct conditional expectation --
# this was verified by direct construction while building this module (a
# rollback of a known zero-coupon bond price under the raw-r/non-deflated
# approach mismatched the closed-form target by ~3%, while the
# x-state/numeraire-deflated approach matches to <1e-4 relative, shrinking
# with grid resolution -- see docs/10-american-swaptions.md's "why x, not
# r" section for the full derivation and this discrepancy).
#
# x(t) relates to this codebase's other closed forms via the identities
# already established in docs/09-ore-parity.md section 3b ("A parametrization
# note: LGM vs. plain Hull-White"): H(t) = B(0,t) (this codebase's own
# _hw_B with t=0), zeta(t) = hw_sigma^2 * t, and the short rate itself is
# r(t) = f(0,t) + x(t)*H'(t) + zeta(t)*H'(t)*H(t) -- used only to convert an
# x-grid into the r-values compute_hw_A/_hw_B's payoff formulas need (see
# _r_from_x); the rollback/convolution machinery itself never touches r
# directly.
# =============================================================================
def _zeta(sigma: float, t: float) -> float:
    """Accumulated variance of the driftless LGM state x(t):
    zeta(t) = sigma^2 * t -- QuantExt::IrLgm1fConstantParametrization::zeta
    (live-verified identity, see docs/09-ore-parity.md section 3b)."""
    return (sigma ** 2) * max(t, 0.0)


def _H(a: float, t: float) -> float:
    """H(t) = (1-exp(-a*t))/a -- QuantExt::Lgm1fParametrization's H(t), the
    same shape as this codebase's own B(t,T) with t=0 (docs/09-ore-parity.md
    section 3b's live-verified identity)."""
    return (1.0 - np.exp(-a * t)) / a if t > 0.0 else 0.0


def _Hprime(a: float, t: float) -> float:
    """H'(t) = exp(-a*t)."""
    return np.exp(-a * t)


def _lgm_bond(zero_times: np.ndarray, zero_rates: np.ndarray, a: float, sigma: float, t: float, T: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    P(t,T,x) directly in LGM's own state variable, with NO detour through
    this codebase's r(t)-parametrized compute_hw_A/_hw_B:

        P(t,T,x) = [P(0,T)/P(0,t)] * exp(-0.5*(H(T)^2-H(t)^2)*zeta(t))
                                    * exp(-(H(T)-H(t))*x)

    -- QuantExt::LinearGaussMarkovModel::discountBond (lgm.hpp lines
    252-280, docs/09-ore-parity.md's "parametrization note"), live-verified
    directly against `ORE.LinearGaussMarkovModel.discountBond(t,T,x)` to
    machine precision (~1e-16 relative) while building this module.

    **This function exists, deliberately NOT reusing compute_hw_A/_hw_B
    (this codebase's OTHER, independently-verified HW1F closed form, used
    everywhere else in this codebase), because of a finding made while
    building this module: `ORE.HullWhite` (`QuantLib::HullWhite`, driven by
    compute_hw_A/_hw_B's own A(t,T)/B(t,T)) and
    `ORE.LinearGaussMarkovModel`/`QuantExt::CrossAssetModel` (driven by
    H(t)/zeta(t)) are NOT the same numerical model realization for t>0,
    despite being constructed with identical (a, sigma) and sharing the
    same t=0 curve -- live-verified directly: `HW.discountBond(t,T,r=f(0,t))`
    and `LGM.discountBond(t,T,x=0)` (the "no shock, on today's curve" point
    for each model's own natural reference state) differ by ~0.6% at t=3y
    for a=0.03,sigma=0.02, even though EVERY individual building block
    checked along the way (H(t), zeta(t), H'(t), f(0,t), the r(t,x)
    identity confirmed via finite difference against the LGM formula
    itself, and A(t,T)/B(t,T) confirmed exactly against
    ORE.HullWhite.discountBond for arbitrary r) is independently exact. The
    two ORE classes are both self-consistent one-factor affine short-rate
    models (each satisfies its own `-d/dT log P(t,T)|T=t == r` identity
    exactly, live-verified) but are evidently calibrated/parametrized
    differently for t>0 -- see docs/10-american-swaptions.md's "why x, not
    r" section for the full investigation trail. Since ORE's actual
    Bermudan/American engine (`NumericLgmMultiLegOptionEngine`) is built on
    `LinearGaussMarkovModel`, THIS is the correct model to match here, and
    this module uses ONLY this formula for every discount factor --
    compute_hw_A/_hw_B are not used anywhere in this module.
    """
    P0T = np.exp(-np.interp(T, zero_times, zero_rates) * T)
    P0t = np.exp(-np.interp(t, zero_times, zero_rates) * t) if t > 0.0 else 1.0
    Ht, HT = _H(a, t), _H(a, T)
    return (P0T / P0t) * np.exp(-0.5 * (HT ** 2 - Ht ** 2) * _zeta(sigma, t)) * np.exp(-(HT - Ht) * x)


def _r_from_x(zero_times: np.ndarray, zero_rates: np.ndarray, a: float, sigma: float, t: float, x: np.ndarray) -> np.ndarray:
    """
    The short rate implied by LGM's own state x(t) and bond formula (NOT
    ORE.HullWhite's short rate -- see _lgm_bond's docstring; the two are
    different models for t>0):

        r(t,x) = f(0,t) + x*H'(t) + zeta(t)*H'(t)*H(t)

    live-verified via central finite difference directly on _lgm_bond
    (`-d/dT log P(t,T,x) |_{T=t}`, matching to ~1e-12 relative) -- this IS
    LGM's own genuine short rate at state x, used only to convert this
    module's simulated hw_paths (engine.simulation's own literal
    short-rate simulation, calibrated with the SAME (a, sigma) this module
    receives) into the corresponding LGM state x for conditioning (see
    _x_from_r, its exact inverse).
    """
    eps = 1e-6
    log_P0_t = -np.interp(t, zero_times, zero_rates) * t
    log_P0_t_eps = -np.interp(t + eps, zero_times, zero_rates) * (t + eps)
    f0t = -(log_P0_t_eps - log_P0_t) / eps
    Hp = _Hprime(a, t)
    return f0t + x * Hp + _zeta(sigma, t) * Hp * _H(a, t)


def _x_from_r(zero_times: np.ndarray, zero_rates: np.ndarray, a: float, sigma: float, t: float, r: np.ndarray) -> np.ndarray:
    """Exact inverse of _r_from_x (r(t,x) is affine/linear in x, so this is
    a closed-form solve, not a numerical root-find): converts a short rate
    (e.g. a simulated hw_paths value) into the corresponding LGM state x at
    the same time t, for interpolating into an x-indexed value function."""
    if t <= 0.0:
        return np.zeros_like(r)
    eps = 1e-6
    log_P0_t = -np.interp(t, zero_times, zero_rates) * t
    log_P0_t_eps = -np.interp(t + eps, zero_times, zero_rates) * (t + eps)
    f0t = -(log_P0_t_eps - log_P0_t) / eps
    Hp = _Hprime(a, t)
    return (r - f0t - _zeta(sigma, t) * Hp * _H(a, t)) / Hp


def _state_grid(sigma: float, t: float, n_per_std: int, std_devs: float) -> np.ndarray:
    """
    The centered LGM state grid at time t: `x_k = k*dx`, `dx =
    sqrt(zeta(t)) / n_per_std`, spanning `+/- std_devs` standard deviations
    -- exactly `LgmConvolutionSolver2::stateGrid`'s `dx = sqrt(zeta(t))/nx_`
    construction (lgmconvolutionsolver2.cpp), with `mx_ = round(std_devs *
    n_per_std)` points on each side of zero, always including x=0 itself
    (an odd-length grid, matching ORE's `2*mx_+1` point count).

    At t=0, zeta(0)=0 and the grid collapses to the single point x=0 --
    matching `LgmConvolutionSolver2::stateGrid`'s explicit `t=0` special
    case (`if (close_enough(t,0.0)) return RandomVariable(2*mx_+1, 0.0);`).
    """
    if t <= 0.0:
        mx = int(round(std_devs * n_per_std))
        return np.zeros(2 * mx + 1, dtype=np.float64)
    dx = np.sqrt(_zeta(sigma, t)) / n_per_std
    mx = int(round(std_devs * n_per_std))
    return dx * np.arange(-mx, mx + 1, dtype=np.float64)


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
    GPU: E[values(x_from) | x_to] under the model's exact Gaussian
    transition law, evaluated at every point of `x_to` simultaneously
    (vectorized across the leading batch axes of `values`/`x_to`, e.g.
    [Scenarios] or [Trades] -- see _run_backward_induction), reproducing
    `LgmConvolutionSolver2::rollback`.

    For each target node `x_to[k]`, the conditional distribution of
    `x_from` is Gaussian with mean `x_to[k]` (LGM/HW1F's own state variable
    is driftless in this deviation parametrization -- the SAME identity
    docs/09-ore-parity.md section 3a already establishes from
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


def _numeraire(zero_times: np.ndarray, zero_rates: np.ndarray, a: float, sigma: float, t: float, x: np.ndarray) -> np.ndarray:
    """
    N(t,x) = exp(0.5*H(t)^2*zeta(t) + H(t)*x) / P(0,t) --
    QuantExt::LinearGaussMarkovModel::numeraire (lgm.hpp lines 227-250,
    docs/09-ore-parity.md section "A parametrization note"). Every payoff is
    divided by this before rolling back (see _hw_swap_reduced_value_at_nodes)
    so that the rolled-back quantity is a true Q-martingale under x(t)'s own
    driftless transition law -- see module docstring's "STATE GRID" section
    header comment for why this deflation is mathematically required (a
    plain, non-deflated rollback of P(t,T) under x's own transition does NOT
    reproduce P(t_to,T) -- verified directly while building this module).
    """
    if t <= 0.0:
        return np.ones_like(x)
    P0t = np.exp(-np.interp(t, zero_times, zero_rates) * t)
    Ht = _H(a, t)
    return np.exp(0.5 * Ht ** 2 * _zeta(sigma, t) + Ht * x) / P0t


def _discount_at_nodes(x_nodes: np.ndarray, t: float, times: np.ndarray, swap: _PreparedBermudan) -> np.ndarray:
    """P(t, times; x_nodes) at every LGM state-grid node, for every cashflow
    time, via _lgm_bond (the directly x-parametrized, live-verified LGM
    bond formula -- see its own docstring for why this module uses it
    instead of this codebase's r(t)-parametrized compute_hw_A/_hw_B) --
    zeroed out for any cashflow already paid/expired (times <= t), matching
    NumericLgmMultiLegOptionEngineBase's cashflow bookkeeping (a coupon is
    folded into the running NPV once and never revisited once its own time
    passes).

    Uses the SAME `>= t - 1e-9` tolerance as _hw_swap_value_at_nodes's own
    fixed/float alive masks (not a strict `times > t`) -- a coupon whose
    start/end/pay time lands EXACTLY on the exercise/conditioning time `t`
    is the common case for a reset-aligned exercise date (not an edge
    case), and a stricter zero-tolerance comparison here previously zeroed
    out P(t, t; x) for that coupon's own accrual-start date even though the
    caller's mask had already classified it as alive, corrupting that
    coupon's forward-rate ratio (found and fixed during this module's own
    test-suite development, see
    tests/test_bermudan_swaption.py::TestMidCouponKnownLimitation)."""
    if times.size == 0:
        return np.zeros((x_nodes.shape[0], 0))
    alive = (times > t - 1e-9).astype(np.float64)
    P = np.stack([
        _lgm_bond(swap.zero_times, swap.zero_rates, swap.hw_a, swap.hw_sigma, t, float(T), x_nodes)
        for T in times
    ], axis=1)
    return P * alive[None, :]


def _hw_swap_value_at_nodes(x_nodes: np.ndarray, t: float, swap: _PreparedBermudan) -> np.ndarray:
    """
    The underlying swap's own remaining NPV (float leg - fixed leg,
    payer-signed) at time t, evaluated at every LGM state-grid node in
    `x_nodes` -- the "intrinsic" (exercise) value used at each exercise
    date, via _discount_at_nodes/_lgm_bond (this module's own live-verified
    LGM bond formula -- see _lgm_bond's docstring for why this module does
    NOT reuse engine.instruments.european_swaption's compute_hw_A/_hw_B, a
    different model realization for t>0), following the same fixed-leg /
    floating-leg NPV structure as engine.instruments.swap._price_one_swap,
    just evaluated at model-grid nodes instead of simulated scenario paths.

    fixed_amounts is used directly (notional*rate*accrual, read straight
    from ORE's own FixedRateCoupon.amount() in prepare_bermudan) rather
    than re-deriving rate*accrual -- avoids re-deriving a day-count detail
    ORE's own schedule generation has already resolved exactly.
    """
    # Same accrual-start-based liveness rule as the floating leg below (see
    # its comment): a fixed coupon is only a genuine remaining cashflow
    # once its OWN accrual period has not yet begun relative to t, which is
    # exact (not an approximation) at any reset-aligned t within this
    # module's documented scope.
    fixed_alive = (swap.fixed_start_times >= t - 1e-9)
    fixed_disc = _discount_at_nodes(x_nodes, t, swap.fixed_times, swap)
    fixed_leg_pv = fixed_disc @ (swap.fixed_amounts * fixed_alive)

    if swap.float_pay_times.size == 0:
        float_leg_pv = np.zeros_like(fixed_leg_pv)
    else:
        # Only coupons whose accrual has NOT YET BEGUN (start >= t) are
        # included -- P(t, accrual_start) via the closed-form A/B formula is
        # only a genuine discount factor for accrual_start >= t (the same
        # T<t clamping issue documented as a known limitation in
        # engine.instruments.swap's module docstring). This is not an
        # approximation HERE specifically because every exercise/condition
        # time this module ever evaluates _hw_swap_value_at_nodes at is, by
        # this module's own documented scope (BermudanSwaptionConfig's
        # docstring), a reset/accrual-start date of the underlying swap --
        # so at any such t, every floating coupon either has start >= t
        # (not yet begun, correctly priced) or start < t only for a coupon
        # whose OWN start was a prior, already-passed reset date entirely
        # (which, at a reset-aligned t, coincides with end <= t too, i.e.
        # it is a fully elapsed coupon that must be excluded from the
        # swap's remaining value in any case -- consistent with, not a
        # workaround of, the exclusion).
        alive = (swap.float_start_times >= t - 1e-9)
        p_start = _discount_at_nodes(x_nodes, t, swap.float_start_times, swap)
        p_end = _discount_at_nodes(x_nodes, t, swap.float_end_times, swap)
        p_end_safe = np.where(p_end == 0.0, 1.0, p_end)
        forward_rate = np.where(alive[None, :], (p_start / p_end_safe - 1.0) / swap.float_accrual[None, :], 0.0)
        float_pay_disc = _discount_at_nodes(x_nodes, t, swap.float_pay_times, swap)
        float_cashflow = swap.notional * (forward_rate + swap.float_spread) * swap.float_accrual[None, :]
        float_leg_pv = np.sum(float_cashflow * float_pay_disc * alive[None, :], axis=1)

    npv = float_leg_pv - fixed_leg_pv
    return npv if swap.payer else -npv


@dataclass
class _RolledBackValue:
    """The result of running backward induction on a prepared Bermudan/
    American swaption: the state grid and option-value function at time 0
    (a single node, x=0), plus everything needed to evaluate the SAME
    rolled-back value function conditional on an arbitrary simulated short
    rate at any of the requested `condition_times` (see
    price_bermudan_swaptions)."""
    value_at_t0: float
    condition_times: np.ndarray
    condition_state_grids: List[np.ndarray]
    condition_values: List[np.ndarray]


def _run_backward_induction(swap: _PreparedBermudan, condition_times: Sequence[float]) -> _RolledBackValue:
    """
    CPU/NumPy: the core Hagan-convolution backward induction (module
    docstring steps 2-6), run once per prepared swaption.

    **Runs entirely in numeraire-deflated units** (`reduced = value / N(t,x)`
    at every grid time) -- required for Hagan's quadrature convolution to be
    valid at all: LGM's state x(t) is driftless, so `E[reduced(x_from) |
    x_to]` under x's own Gaussian transition law is exactly the deflated
    value at `x_to` (a true martingale identity, matching
    `QuantExt::LinearGaussMarkovModel::reducedDiscountBond`'s whole reason
    for existing -- see docs/09-ore-parity.md's "parametrization note").
    Rolling back RAW (non-deflated) values under x's transition law would
    silently give the wrong answer, since a raw bond/swap price is NOT a
    martingale under x's driftless transition on its own (only the
    numeraire-deflated price is) -- this was the actual bug caught and
    fixed while building this module (see the "STATE GRID" section's header
    comment above for the numeric evidence). The early-exercise
    `max(intrinsic, continuation)` comparison is applied in deflated units
    too (dividing the intrinsic swap value by the SAME N(t,x) the
    continuation value is already deflated by) -- valid because N(t,x) > 0
    always, so `max` commutes with the deflation.

    Grid times = every exercise date, every condition_time the caller wants
    an intermediate value at, the final maturity, and 0.0 -- sorted
    descending for the backward walk. At each grid time (other than the
    first/latest), the deflated value function is rolled back one step from
    the previous (later) grid time via Hagan's quadrature convolution, then,
    if that time is an exercise date, floored at the deflated intrinsic
    (exercise) value -- exactly
    `NumericLgmMultiLegOptionEngineBase::calculate()`'s rule, expressed in
    reduced units.

    The (raw, re-inflated) value function at each `condition_time` is
    snapshotted (grid + values) as the walk passes through it, so callers
    can later condition on an arbitrary simulated short rate at that
    specific time without re-running the whole induction per scenario.
    """
    a, sigma, n_per_std, std_devs = swap.hw_a, swap.hw_sigma, swap.n_per_std, swap.std_devs
    quad_w = _hagan_quadrature_weights(n_per_std, std_devs)
    my = int(round(std_devs * n_per_std))
    quad_y = (1.0 / n_per_std) * np.arange(-my, my + 1, dtype=np.float64)

    exercise_set = set(round(float(t), 12) for t in swap.exercise_times)
    condition_set = set(round(float(t), 12) for t in condition_times)
    grid_times = sorted(set([0.0, swap.final_maturity]) | exercise_set | condition_set, reverse=True)

    x_prev = _state_grid(sigma, grid_times[0], n_per_std, std_devs)
    t_prev = grid_times[0]
    numeraire_prev = _numeraire(swap.zero_times, swap.zero_rates, a, sigma, t_prev, x_prev)
    raw_values = _hw_swap_value_at_nodes(x_prev, t_prev, swap)
    if round(t_prev, 12) in exercise_set:
        raw_values = np.maximum(raw_values, _hw_swap_value_at_nodes(x_prev, t_prev, swap))
    reduced = raw_values / numeraire_prev

    def r_grid_at(t_val: float, x_grid: np.ndarray) -> np.ndarray:
        return _r_from_x(swap.zero_times, swap.zero_rates, a, sigma, t_val, x_grid)

    snapshots = {}
    if round(t_prev, 12) in condition_set:
        snapshots[round(t_prev, 12)] = (r_grid_at(t_prev, x_prev), raw_values.copy())

    for t in grid_times[1:]:
        x_t = _state_grid(sigma, t, n_per_std, std_devs)
        std_step = np.sqrt(max(_zeta(sigma, t_prev) - _zeta(sigma, t), 0.0))
        if std_step > 0.0:
            reduced_continuation = np.asarray(_rollback_one_step(
                jnp.asarray(reduced), jnp.asarray(x_prev), jnp.asarray(x_t),
                jnp.asarray(quad_y), jnp.asarray(quad_w), jnp.asarray(std_step),
            ))
        else:
            # t == t_prev in model time (can happen if two grid times
            # coincide after float rounding) -- no actual time has passed.
            reduced_continuation = np.interp(x_t, x_prev, reduced)

        numeraire_t = _numeraire(swap.zero_times, swap.zero_rates, a, sigma, t, x_t)
        if round(t, 12) in exercise_set:
            reduced_intrinsic = _hw_swap_value_at_nodes(x_t, t, swap) / numeraire_t
            reduced = np.maximum(reduced_continuation, reduced_intrinsic)
        else:
            reduced = reduced_continuation

        raw_values = reduced * numeraire_t
        x_prev, t_prev, numeraire_prev = x_t, t, numeraire_t
        if round(t, 12) in condition_set:
            snapshots[round(t, 12)] = (r_grid_at(t, x_t), raw_values.copy())

    condition_state_grids = [snapshots[round(float(t), 12)][0] for t in condition_times]
    condition_values = [snapshots[round(float(t), 12)][1] for t in condition_times]
    value_at_t0 = float(snapshots[round(0.0, 12)][1][0]) if round(0.0, 12) in snapshots else float(raw_values[0])

    return _RolledBackValue(
        value_at_t0=value_at_t0,
        condition_times=np.asarray(condition_times, dtype=np.float64),
        condition_state_grids=condition_state_grids,
        condition_values=condition_values,
    )


def price_bermudan_swaption_base(cfg: BermudanSwaptionConfig) -> float:
    """t=0 NPV of a single Bermudan swaption (no simulated conditioning) --
    the value read off the backward induction's own x=0 node, exactly
    LgmConvolutionSolver2::stateGrid(0)'s single-point convention."""
    swap = prepare_bermudan(cfg)
    result = _run_backward_induction(swap, condition_times=[])
    return result.value_at_t0


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
    from engine.simulation import generate_paths
    from engine.scenarios import EVAL_DATE, swaption_demo_config

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
