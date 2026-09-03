"""
European swaption pricing via Jamshidian's decomposition.

Trade structure (the underlying swap's schedule, day-count accrual, coupon
amounts) is built with ORE's own `MakeVanillaSwap` machinery, exactly as in
`engine.instruments.swap` -- date generation and accrual math match ORE
exactly rather than being reimplemented.

Pricing model: Jamshidian's trick, the same closed-form decomposition
`ORE.JamshidianSwaptionEngine` uses for a European swaption under a
Hull-White 1-Factor short rate model. A European swaption on a fixed-vs-
floating swap is mathematically equivalent to a European option on a
coupon-bearing bond (the swap's fixed leg, plus a final notional exchange,
against a floating leg that -- at par, under single-curve discounting --
always redeems at exactly the notional). Jamshidian's trick expresses that
coupon-bond option as a portfolio of zero-coupon bond options, each of which
has a closed form under HW1F (the model is affine, so every zero-coupon
bond price is a monotonic function of the short rate, letting the coupon
bond's exercise boundary be expressed as a single critical short rate `r*`
shared by every leg of the decomposition).

**Live-verified against ORE**, not assumed from a textbook: an independent
Python reimplementation of this exact formula (using the SAME zero-curve
interpolation `compute_hw_A_matrix` uses, not any ORE-internal shortcut) was
checked against `ORE.JamshidianSwaptionEngine.NPV()` across payer/receiver,
ITM/ATM/OTM, and multiple tenors, matching to a relative precision of 1e-6
or better -- see this module's test suite for the same methodology encoded
as regression tests.

Single-model pricing (unlike the linear swap pricer): Jamshidian's trick
prices the underlying swap and its option under ONE Hull-White model/curve,
not a separate discount/forward curve pair -- there is no multi-curve
Jamshidian formula in ORE either, since the bond-option decomposition is
intrinsically tied to one model's affine bond-price dynamics
(`ORE.JamshidianSwaptionEngine` itself takes a single `ShortRateModel`). A
`rate_factor_index` selects which of the simulation's Hull-White factors
underlies both legs of the swap and the option itself.

Conditional (future-time) pricing: to produce a
`[Scenarios, TimeSteps, Trades]` NPV cube -- the same contract every other
pricer in this codebase returns, required for `engine.risk`
-- this module evaluates Jamshidian's formula not just at t=0 but at every
simulated (scenario, time step) pair, conditional on that scenario's
simulated short rate at that step. This is a direct consequence of HW1F's
Markov property: the model's own closed-form conditional bond price,
`P(t,T) = A(t,T) * exp(-B(t,T) * r(t))`, is exactly the formula
`engine.simulation.compute_hw_A_matrix`/`reconstruct_yield_curves`
already use to build the yield curve cube -- Jamshidian's decomposition
still applies verbatim with `t` (the simulated time) in place of "today".
This conditional generalization was itself live-verified against ORE (by
rebuilding ORE's own evaluation date and implied curve at a later time and
re-pricing with a fresh `JamshidianSwaptionEngine`), matching to the same
~1e-6 relative precision as the t=0 case. Once `t` passes the option's own
exercise time `T0`, NPV is reported as exactly 0 (a European option carries
no value after its own expiry).
"""
from dataclasses import dataclass, field
from functools import partial
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import ORE

from engine.simulation.market_model import ZeroCurveConfig
from engine.models.static_key import StaticKeyMixin
from engine.models.ore_builders import DAY_COUNTER, build_vanilla_swap
from engine.portfolio.validation import _validate_common_fields, _validate_tenor
from engine.models.hull_white import (
    A as _hw_A,
    B as _hw_B,
    ZeroCurve as _HwZeroCurve,
    bond_call as _bond_call,
    bond_option_sigma as _bond_option_sigma,
    bond_put as _bond_put,
)


def compute_hw_A(zero_times: np.ndarray, zero_rates: np.ndarray, t: np.ndarray, T: np.ndarray, a: float, sigma: float) -> np.ndarray:
    """Thin NumPy-facing wrapper around `engine.models.hull_white.A` --
    kept under this module's original name/signature (`zero_times`/
    `zero_rates` arrays rather than a `ZeroCurve`) since it's part of this
    module's own public surface (imported directly by
    `tests/test_ore_parity.py` and others) and since `_PreparedSwaption`
    stores the curve as plain NumPy arrays. The formula itself lives in
    exactly one place -- `engine/models/hull_white.py` -- not re-derived
    here."""
    curve = _HwZeroCurve(pillar_times=jnp.asarray(zero_times), pillar_rates=jnp.asarray(zero_rates))
    return np.asarray(_hw_A(curve, jnp.asarray(t), jnp.asarray(T), a, sigma))


@dataclass
class SwaptionConfig:
    """
    One European swaption: the option to enter a vanilla fixed-vs-floating
    swap at the exercise date.

    rate_factor_index selects which simulation Hull-White factor prices
    BOTH the underlying swap and the option (Jamshidian's trick is a
    single-model formula -- see module docstring). hw_a/hw_sigma/
    initial_zero_curve MUST match that factor's own calibration in the
    simulation's RatesConfig (mean_reversion[rate_factor_index], the
    per-step volatility implied by joint_covariance for that factor, and
    initial_zero_curves[rate_factor_index]) -- they parametrize the
    closed-form bond-price/bond-option formulas directly, and there is no
    way to recover them from the simulated paths alone.

    swap_tenor/index_tenor_months/floating_spread: same meaning as
    engine.instruments.swap.SwapConfig, describing the underlying swap ORE
    builds via MakeVanillaSwap.

    forward_start: an ORE.Period the underlying swap's first accrual is
    delayed by beyond the standard spot lag (e.g. ORE.Period(5, ORE.Years)
    for a swaption exercisable in 5Y -- the common case, since a spot-lag-
    only exercise date is only ~2 days away and expires almost immediately
    in any simulation with a coarser time grid). Defaults to no delay (just
    the standard 2-day spot lag, like swap.SwapConfig).

    exercise_lag_days: business days from evaluation_date + forward_start to
    the option's exercise date (2 = standard spot lag, matching
    MakeVanillaSwap's own default settlement-day convention -- so with no
    forward_start, the exercise date coincides with the swap's own spot-lag
    accrual start, and with a forward_start it precedes that start by the
    same 2-day convention).
    """
    notional: float
    fixed_rate: float
    payer: bool
    rate_factor_index: int
    hw_a: float
    hw_sigma: float
    initial_zero_curve: ZeroCurveConfig
    swap_tenor: str = "5Y"
    index_tenor_months: int = 6
    floating_spread: float = 0.0
    forward_start: ORE.Period = field(default_factory=lambda: ORE.Period(0, ORE.Days))
    exercise_lag_days: int = 2
    evaluation_date: ORE.Date = field(default_factory=lambda: ORE.Settings.instance().evaluationDate)

    def __post_init__(self) -> None:
        _validate_common_fields(self.notional, self.fixed_rate, self.evaluation_date)
        _validate_tenor(self.swap_tenor, "swap_tenor")
        if self.hw_sigma != self.hw_sigma or self.hw_sigma in (float("inf"), float("-inf")):
            raise ValueError(f"hw_sigma must be finite; got {self.hw_sigma}")


def _build_ore_swap(cfg: SwaptionConfig) -> ORE.VanillaSwap:
    """CPU: builds the real ORE underlying swap (schedules, day counts,
    conventions) -- see `engine.models.ore_builders.build_vanilla_swap`,
    the single shared implementation of this construction."""
    return build_vanilla_swap(
        notional=cfg.notional, fixed_rate=cfg.fixed_rate, payer=cfg.payer,
        swap_tenor=cfg.swap_tenor, index_tenor_months=cfg.index_tenor_months,
        floating_spread=cfg.floating_spread, evaluation_date=cfg.evaluation_date,
        forward_start=cfg.forward_start,
    )


@dataclass(frozen=True, eq=False)
class _PreparedSwaption(StaticKeyMixin):
    """Every field is compile-time-constant trade/model structure, resolved
    once by `prepare_swaption` -- nothing here varies per scenario/step.

    Hashable by value via `StaticKeyMixin`, so it can be passed as a
    `jax.jit` STATIC argument to `_price_one_swaption` -- which is what lets
    that whole kernel, including `_solve_rstar`'s 100 fixed bisection
    iterations, compile once and then be reused. See
    `engine.models.static_key` for the full rationale.
    """
    payer: bool
    notional: float
    exercise_time: float               # T0, year-fraction from evaluation_date
    accrual_start_time: float          # T_start, the underlying swap's own first accrual date
    fixed_cashflow_times: np.ndarray   # [N] year-fractions from evaluation_date
    fixed_cashflow_amounts: np.ndarray  # [N]
    rate_factor_index: int
    hw_a: float
    hw_sigma: float
    zero_times: np.ndarray
    zero_rates: np.ndarray


def prepare_swaption(cfg: SwaptionConfig) -> _PreparedSwaption:
    """
    CPU: build the ORE underlying swap and extract the fixed leg's
    cashflow times/amounts, the swap's own accrual start time, and the
    option's exercise time, all as year-fractions from `cfg.evaluation_date`.
    Static per swaption -- run once, not per scenario/step.

    The floating leg's own reset dates are NOT extracted: Jamshidian's
    decomposition relies on the standard identity that an at-par floating
    leg (under single-curve discounting, which this module assumes since it
    prices off one HW factor) collapses to a single telescoping value,
    `notional * (P(T0,T_start) - P(T0,T_last))` -- receiving the notional
    at the swap's own first accrual start `T_start` and paying it back at
    the final date `T_last`, with every intermediate reset cancelling. This
    is the same identity `float_leg_pv` in
    swap._price_one_swap computes explicitly per-period; here
    it is used in its closed (telescoping) form, which is exact for a
    genuinely at-par floater (spread=0, forwarding curve == discounting
    curve -- both true here since Jamshidian's trick is single-curve).

    T_start is NOT assumed to equal the exercise time T0: for a
    forward-starting swap (SwaptionConfig.forward_start != 0), the
    underlying's first accrual begins `exercise_lag_days` AFTER the
    exercise date (the same spot-lag convention MakeVanillaSwap itself
    applies), so `P(T0,T_start)` in the identity above is a genuine
    (near-1, but not exactly 1) discount factor, not an identity -- an
    earlier version of this module assumed T_start == T0 unconditionally,
    which is only exactly true for a NON-forward-starting swaption (spot
    lag and exercise lag coincide there) and was caught by cross-checking a
    forward-starting swaption directly against
    ORE.JamshidianSwaptionEngine.NPV(), which diverged by ~1% until this
    term was added -- see this module's test suite for the regression test.
    """
    swap = _build_ore_swap(cfg)
    today = cfg.evaluation_date
    accrual_start_date = ORE.as_fixed_rate_coupon(swap.fixedLeg()[0]).accrualStartDate()
    # Exercise date: exercise_lag_days before the FORWARD-START point (today,
    # for a non-forward-starting swaption), NOT before accrual_start_date --
    # accrual_start_date is already the forward-start point pushed out by the
    # index's own spot lag, so subtracting exercise_lag_days from it a
    # SECOND time would double-count that lag (live-verified: with no
    # forward_start, ORE.JamshidianSwaptionEngine.NPV() at that
    # double-lagged date collapses to exactly 0, since it lands back on
    # `today` itself -- a zero-maturity option is worthless by
    # construction, not a pricing bug).
    forward_start_date = ORE.TARGET().advance(today, cfg.forward_start)
    exercise_date = ORE.TARGET().advance(forward_start_date, cfg.exercise_lag_days, ORE.Days)

    fixed_times, fixed_amounts = [], []
    for cf in swap.fixedLeg():
        c = ORE.as_fixed_rate_coupon(cf)
        fixed_times.append(DAY_COUNTER.yearFraction(today, c.date()))
        fixed_amounts.append(c.amount())

    return _PreparedSwaption(
        payer=cfg.payer,
        notional=swap.fixedNominals()[0] if swap.fixedNominals() else swap.nominal(),
        exercise_time=DAY_COUNTER.yearFraction(today, exercise_date),
        accrual_start_time=DAY_COUNTER.yearFraction(today, accrual_start_date),
        fixed_cashflow_times=np.array(fixed_times),
        fixed_cashflow_amounts=np.array(fixed_amounts),
        rate_factor_index=cfg.rate_factor_index,
        hw_a=cfg.hw_a,
        hw_sigma=cfg.hw_sigma,
        zero_times=np.asarray(cfg.initial_zero_curve.times, dtype=np.float64),
        zero_rates=np.asarray(cfg.initial_zero_curve.rates, dtype=np.float64),
    )


# =============================================================================
# HULL-WHITE 1-FACTOR CLOSED-FORM BUILDING BLOCKS
#
# `compute_hw_A` (this module's own NumPy-facing wrapper, defined above)
# and `_hw_B`/`_bond_option_sigma`/`_bond_call`/`_bond_put` (imported
# directly from `engine.models.hull_white`) are the single shared
# implementation of every HW1F closed form this module needs -- see that
# module's docstring for the math and the ORE correspondence. Nothing here
# re-derives them.
# =============================================================================
def _bisect_rstar(coupon_bond_value_fn, t_shape, iterations: int, dtype=jnp.float64) -> jax.Array:
    """The bisection itself (forward value only, no gradient guarantees --
    see `_solve_rstar`, which wraps this with a differentiable correction).
    Kept as a standalone function so `_solve_rstar`'s `jax.custom_jvp`
    primal can call it directly under `jax.lax.stop_gradient`.

    `dtype`: an explicit parameter, not `jnp.ones(t_shape)`'s own implicit
    default -- `jnp.ones`/`jnp.zeros` with no `dtype` argument silently pick
    up JAX's ambient default float dtype (float64 whenever `jax_enable_x64`
    is on, REGARDLESS of what dtype the surrounding pricing computation
    actually wants), which used to upcast `rstar` back to float64 even when
    every other array in `_price_one_swaption` (A_T0_Ti/B_T0_Ti/etc.) was
    correctly float32 under `PrecisionConfig.pricing=32` -- confirmed
    directly (see PrecisionConfig's docstring for why any one hardcoded/
    implicitly-defaulted array anywhere in this chain silently promotes the
    whole downstream computation back to float64 whenever `jax_enable_x64`
    is process-globally on)."""
    lo = -jnp.ones(t_shape, dtype=dtype) * 2.0
    hi = jnp.ones(t_shape, dtype=dtype) * 2.0

    def expand_body(_, carry):
        lo, hi = carry
        val_lo = coupon_bond_value_fn(lo)
        val_hi = coupon_bond_value_fn(hi)
        # Monotone decreasing: a real bracket has val_lo > 0 > val_hi. If
        # val_hi is still positive, the root is above hi -- shift the
        # whole window up by its own width. If val_lo is already
        # negative, the root is below lo -- shift down. Width stays fixed
        # (a shift, not a widen), which keeps the bisection step size
        # well-behaved regardless of how many expansions are needed.
        width = hi - lo
        shift_up = val_hi > 0.0
        shift_down = val_lo < 0.0
        new_lo = jnp.where(shift_up, hi, jnp.where(shift_down, lo - width, lo))
        new_hi = jnp.where(shift_up, hi + width, jnp.where(shift_down, lo, hi))
        return (new_lo, new_hi)

    lo, hi = jax.lax.fori_loop(0, 20, expand_body, (lo, hi))

    def body(_, carry):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        val_mid = coupon_bond_value_fn(mid)
        lo = jnp.where(val_mid > 0.0, mid, lo)
        hi = jnp.where(val_mid > 0.0, hi, mid)
        return (lo, hi)

    lo, hi = jax.lax.fori_loop(0, iterations, body, (lo, hi))
    return 0.5 * (lo + hi)


def _solve_rstar(coupon_bond_value_fn, params, t_shape, iterations: int = 100) -> jax.Array:
    """
    Vectorized bisection for Jamshidian's critical short rate r*(scenario,
    step): the short rate at the exercise date at which the signed coupon
    bond (every fixed cashflow, plus final notional, minus the notional
    received back at the swap's own accrual start -- see
    prepare_swaption's docstring) is worth exactly 0 -- the exercise
    boundary shared by every zero-coupon leg of the decomposition (see
    module docstring).

    `coupon_bond_value_fn(r, params) -> value`: unlike a plain closure
    over whatever the caller needs, `params` is an EXPLICIT pytree
    argument holding every value the caller wants `_solve_rstar`'s output
    to be differentiable with respect to (e.g. `A_T0_Ti`, `B_T0_Ti`,
    `all_amounts` in `_price_one_swaption` below) -- required by
    `jax.custom_jvp` (see "Gradient correctness" below), which needs an
    explicit primal argument to define a JVP rule against; a plain Python
    closure's captured tracers cannot be attached to a `custom_jvp` rule
    directly. Any value `coupon_bond_value_fn` needs that the caller does
    NOT want a gradient with respect to (e.g. `T0`, day-count-derived
    times) can still be closed over normally -- only pass through `params`
    what should flow into Delta/Gamma.

    Monotonic and well-posed in practice: every POSITIVE-amount leg's bond
    price P(T0,Ti;r) is strictly decreasing in r (B(T0,Ti) > 0 for every
    Ti > T0), while the single NEGATIVE-amount leg (the T_start notional
    receipt) is strictly increasing in r -- but B(T0,T_start) is tiny
    (T_start is only `exercise_lag_days` after T0, a couple of days,
    versus years for every other leg), so its contribution is dominated by
    every other leg's for any realistic swaption and the sum remains
    monotonically decreasing across the whole practical rate range
    (verified numerically in this module's tests, not just assumed).
    Bisection (rather than Newton's method) is used because it vectorizes
    across every (scenario, time step) pair with a fixed iteration count
    under jax.jit, with no data-dependent stopping condition needed.

    Bracket width: a fixed [-2, 2] (a +-200% short rate) safely brackets
    r* for any realistic trade, but a sufficiently deep-ITM payer (an
    extreme negative fixed_rate) pushes the true root outside it -- the
    coupon bond value is then the SAME sign at both lo and hi (monotone
    decreasing, never crossing zero inside the bracket), so plain
    bisection's `val_mid > 0.0` update collapses `hi` onto `lo` every
    iteration and silently returns the bracket's own edge as a fake root,
    rather than raising or converging to the true (out-of-bracket) value.
    Guarded by doubling the bracket outward (still branch-free/jit-
    friendly, a fixed iteration count) whenever the initial bracket
    doesn't actually contain a sign change, before bisecting -- this keeps
    the common in-bracket case at its original cost while making the rare
    out-of-bracket case converge to the true root instead of a silently
    wrong value.

    **Gradient correctness (implicit function theorem), not just the
    forward value.** Naively differentiating through 100 iterations of a
    comparison-based bisection loop (`jnp.where(val_mid > 0.0, ...)`) does
    NOT produce a correct gradient -- the comparison itself has zero
    gradient everywhere, so a plain `jax.grad` through the unrolled loop
    silently returns 0 regardless of how the true root actually moves with
    the function's own parameters (verified directly: a toy
    `f(r,c) = c - r` bisected this way gives `d(rstar)/dc = 0.0` under
    naive autodiff, vs. the true value `1.0` -- this is what
    `engine/risk/greeks.py`'s Delta/Gamma computation surfaced, since it's
    the first caller in this codebase to differentiate through
    `_solve_rstar`; `price_swaptions` itself never needed a gradient of
    its own root-find).

    Fixed via `jax.custom_jvp` implementing the implicit function theorem
    directly: at a root of `f(r*, params) = 0`,
    `dr*/dparams . v = -(df/dparams . v) / (df/dr)` for any tangent
    direction `v` -- computed here via one `jax.grad` (for `df/dr`, at the
    stop-gradient'd converged root) and one `jax.jvp` (for the directional
    derivative `df/dparams . v`), both cheap relative to the 100-iteration
    bisection itself. This is DELIBERATELY a `custom_jvp`, not the simpler
    "differentiable Newton correction after stop_gradient" trick (tried
    first, and it works correctly for `jax.grad` alone) -- that simpler
    trick's own correction expression is not itself well-defined for a
    SECOND differentiation pass (`jax.hessian`, i.e. Gamma), since the
    `stop_gradient` sits directly in the expression `jax.hessian`
    differentiates, rather than inside a `custom_jvp` rule (which JAX
    knows how to re-differentiate through correctly, because a
    `custom_jvp` rule is itself an ordinary, twice-differentiable Python
    function of the SAME `params`, evaluated fresh on each differentiation
    pass -- confirmed directly: a toy cubic root-find
    (`f(r,c) = c - r**3`, closed-form `d^2(rstar)/dc^2` known exactly)
    matches this implementation's `jax.hessian` output to machine
    precision, whereas the simpler stop_gradient-only trick gives exactly
    `0.0` for the second derivative).

    `df/dr` is computed via `jax.grad` on the SUM of `coupon_bond_value_fn`
    across the batch -- valid here (not just convenient) because this
    function is applied strictly elementwise across `t_shape`: each batch
    entry's output depends only on that same entry's own `params` (the
    per-(scenario,step) A/B/cashflow terms in `_price_one_swaption`), so
    summing before differentiating recovers each element's own partial
    derivative exactly, with no cross-element contamination -- confirmed
    directly against literal finite-difference bump-and-revalue in
    `tests/test_greeks.py`.
    """
    # Derived from params' own leaves (not hardcoded/left to jnp.ones'
    # implicit default) -- see _bisect_rstar's docstring for why. Every
    # caller of _solve_rstar passes at least one JAX array carrying the
    # dtype rstar itself should have (A_T0_Ti in _price_one_swaption/
    # engine.risk.greeks._swaption_price_fn); a plain Python float leaf
    # (e.g. all_amounts before being made a JAX array) would fall back to
    # float64 here via jnp.result_type's own weak-type promotion, matching
    # this function's prior always-float64 behavior for such a case.
    dtype = jnp.result_type(*[leaf for leaf in jax.tree_util.tree_leaves(params)])

    @jax.custom_jvp
    def solve(p):
        f = lambda r: coupon_bond_value_fn(r, p)
        rstar = _bisect_rstar(f, t_shape, iterations, dtype=dtype)
        return jax.lax.stop_gradient(rstar)

    @solve.defjvp
    def solve_jvp(primals, tangents):
        p, = primals
        p_dot, = tangents
        rstar_val = solve(p)
        df_dr = jax.grad(lambda r: jnp.sum(coupon_bond_value_fn(r, p)))(rstar_val)
        _, df_dparams_dot = jax.jvp(lambda pp: coupon_bond_value_fn(rstar_val, pp), (p,), (p_dot,))
        rstar_dot = -df_dparams_dot / df_dr
        return rstar_val, rstar_dot

    return solve(params)


@partial(jax.jit, static_argnums=2)
def _price_one_swaption(
    hw_paths: jax.Array, step_times: jax.Array, swaption: _PreparedSwaption,
) -> jax.Array:
    """
    GPU: [Scenarios, TimeSteps] NPV for a single prepared swaption via
    Jamshidian's decomposition, vectorized across every simulated scenario
    and time step.

    For each (scenario, step) with conditioning time t = step_times[step]
    and simulated short rate r(t) = hw_paths[..., rate_factor_index]:
      1. Solve r* at the exercise time T0 (see _solve_rstar) from the
         identity that the fixed coupon bond (every fixed cashflow, plus
         final notional, MINUS the notional received back at the swap's own
         accrual start T_start -- see prepare_swaption's docstring on why
         T_start is a genuine bond option leg, not an identity) is worth
         exactly 0 at the exercise boundary. A(T0,Ti)/A(T0,T_start) are
         computed once on CPU (depend only on T0/T_start and today's
         curve).
      2. Each fixed cashflow, the final notional, and the T_start notional
         receipt each become a zero-coupon bond option struck at
         K_i = P(T0, Ti; r*), priced as of t via the Black-on-bond formula,
         conditioned on r(t).
      3. Payer swaption = sum of bond PUTS (a payer benefits when rates
         rise, i.e. bond prices fall below their strikes); receiver
         swaption = sum of bond CALLS. Sign convention live-verified
         against ORE.JamshidianSwaptionEngine across payer/receiver in this
         module's tests. The T_start leg carries a NEGATIVE amount (it is
         received, not paid, by the fixed-payer's coupon bond), which
         flows through the same put/call sum via its signed cashflow.

    Once t >= T0 the option has already expired -- NPV is reported as
    exactly 0 for those (scenario, step) entries (a European option carries
    no value after its own exercise date; this mirrors ORE's own
    Instrument.NPV() convention of 0 after expiry rather than raising).

    Returns [Scenarios, TimeSteps].
    """
    a = swaption.hw_a
    sigma = swaption.hw_sigma
    T0 = swaption.exercise_time
    T_start = swaption.accrual_start_time
    cf_times = swaption.fixed_cashflow_times
    cf_amounts = jnp.asarray(swaption.fixed_cashflow_amounts, dtype=hw_paths.dtype)
    notional = swaption.notional
    # coupons + final notional (paid) + notional received back at T_start
    # (negative amount -- see docstring above).
    all_times = np.concatenate([cf_times, cf_times[-1:], [T_start]])
    all_amounts = jnp.concatenate([
        cf_amounts,
        jnp.asarray([notional, -notional], dtype=hw_paths.dtype),
    ])

    # A(T0, Ti) for every coupon/notional date -- depends only on T0 and
    # today's curve (NOT per scenario/step). Calls the JAX-native
    # `hull_white.A` directly rather than this module's `compute_hw_A`
    # NumPy-facing wrapper: that wrapper's `np.asarray` return would
    # concretize a traced value and break this function's `jax.jit`
    # (TracerArrayConversionError). Same formula either way -- the wrapper
    # is a thin NumPy adapter over exactly this call, kept for its external
    # callers (tests/test_ore_parity.py, tests/test_greeks.py).
    _curve = _HwZeroCurve(
        pillar_times=jnp.asarray(swaption.zero_times),
        pillar_rates=jnp.asarray(swaption.zero_rates),
    )
    A_T0_Ti = jnp.asarray(
        _hw_A(_curve, jnp.full_like(jnp.asarray(all_times), T0), jnp.asarray(all_times), a, sigma),
        dtype=hw_paths.dtype,
    )
    B_T0_Ti = _hw_B(T0, jnp.asarray(all_times, dtype=hw_paths.dtype), a)  # [N+1]

    def coupon_bond_value(rstar, params):
        # rstar: [S, T] -> prices: [S, T, N+1]. params carries every value
        # _solve_rstar's caller might want Delta/Gamma with respect to
        # (see that function's docstring on why this must be an explicit
        # argument, not a closure) -- A_T0_Ti (curve-dependent) here;
        # B_T0_Ti (mean-reversion-only) stays closed over since this
        # module's Greeks are curve sensitivities, not model-parameter
        # ones (see engine/risk/greeks.py's module docstring on Vega scope).
        A_T0_Ti_p, all_amounts_p = params
        prices = A_T0_Ti_p[None, None, :] * jnp.exp(-B_T0_Ti[None, None, :] * rstar[..., None])
        return jnp.sum(prices * all_amounts_p[None, None, :], axis=-1)

    r_t = hw_paths[:, :, swaption.rate_factor_index]  # [S, T]
    num_scenarios, num_steps = r_t.shape

    rstar = _solve_rstar(coupon_bond_value, (A_T0_Ti, all_amounts), (num_scenarios, num_steps))
    K = A_T0_Ti[None, None, :] * jnp.exp(-B_T0_Ti[None, None, :] * rstar[..., None])  # [S,T,N+1] strikes

    # A(t, Ti) and A(t, T0) -- conditioning point t varies per step, so
    # these ARE computed per (scenario-independent) step, once per swaption
    # (not per scenario -- only depends on the step's t, not r(t)).
    # Same JAX-native `hull_white.A` call as A_T0_Ti above (see the comment
    # there on why not this module's NumPy-facing `compute_hw_A` wrapper) --
    # `step_times` is a traced argument here, so it must stay a JAX array.
    _step_times_j = jnp.asarray(step_times)
    _all_times_j = jnp.asarray(all_times)
    A_t_Ti = jnp.asarray(
        _hw_A(_curve, _step_times_j[:, None], _all_times_j[None, :], a, sigma),
        dtype=hw_paths.dtype,
    )  # [TimeSteps, N+1]
    A_t_T0 = jnp.asarray(
        _hw_A(_curve, _step_times_j, jnp.full_like(_step_times_j, T0), a, sigma),
        dtype=hw_paths.dtype,
    )  # [TimeSteps]

    B_t_Ti = _hw_B(step_times[:, None], jnp.asarray(all_times, dtype=hw_paths.dtype)[None, :], a)  # [T, N+1]
    B_t_T0 = _hw_B(step_times, T0, a)  # [T]

    P_t_Ti = A_t_Ti[None, :, :] * jnp.exp(-B_t_Ti[None, :, :] * r_t[:, :, None])  # [S,T,N+1]
    P_t_T0 = A_t_T0[None, :] * jnp.exp(-B_t_T0[None, :] * r_t)  # [S,T]

    sigma_p = _bond_option_sigma(
        T0, jnp.asarray(all_times, dtype=hw_paths.dtype)[None, :], step_times[:, None], a, sigma,
    )  # [T, N+1] -- may be exactly 0 (t==T0, or a leg maturing at T0); _bond_call/_bond_put
    # handle that zero-vol limit internally (see their docstrings).

    bond_fn = _bond_put if swaption.payer else _bond_call
    per_leg = bond_fn(
        P_t_T0[:, :, None], P_t_Ti, K, sigma_p[None, :, :],
    )  # [S, T, N+1]
    npv_unexpired = jnp.sum(per_leg * all_amounts[None, None, :], axis=-1)  # [S, T]

    not_yet_expired = step_times[None, :] < T0
    return jnp.where(not_yet_expired, npv_unexpired, 0.0)


def price_swaptions(hw_paths: jax.Array, step_times: jax.Array, swaption_configs: List[SwaptionConfig]) -> jax.Array:
    """
    hw_paths: [Scenarios, TimeSteps, NumHW], typically
        engine.simulation.generate_paths(...)["rates"].
    step_times: [TimeSteps] absolute simulation times (year-fractions from
        evaluation_date) -- the same `time_grid[1:]` values
        generate_paths uses internally.
    swaption_configs: a list of one or more SwaptionConfig objects.
    Returns: [Scenarios, TimeSteps, Trades] NPV cube (0 after each
        swaption's own exercise date -- see _price_one_swaption).
    """
    step_times_jax = jnp.asarray(step_times, dtype=hw_paths.dtype)
    prepared = [prepare_swaption(cfg) for cfg in swaption_configs]
    per_trade = [_price_one_swaption(hw_paths, step_times_jax, swaption) for swaption in prepared]
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

    # Exercisable in 3Y (well within this scenario's 5Y time grid), into a
    # 2Y underlying swap -- illustrates both a still-alive option (steps at
    # 0.5Y-2Y) and a post-exercise, zeroed NPV (steps at 4Y-5Y).
    swaption_cfg = SwaptionConfig(
        notional=1_000_000.0,
        fixed_rate=0.030,
        payer=True,
        rate_factor_index=0,
        hw_a=config.rates.mean_reversion[0],
        hw_sigma=float(np.sqrt(config.joint_covariance[1][1])),
        initial_zero_curve=ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[0.03] * 6),
        swap_tenor="2Y",
        forward_start=ORE.Period(3, ORE.Years),
        evaluation_date=EVAL_DATE,
    )

    npv_cube = price_swaptions(market_cubes["rates"], step_times, [swaption_cfg])
    print("Swaption NPV cube shape:", npv_cube.shape)
    for i, t in enumerate(config.time_grid[1:]):
        print(f"  t={t:.2f}: mean NPV across scenarios = {float(jnp.mean(npv_cube[:, i, 0])):.2f}")
