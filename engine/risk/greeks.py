"""
Delta, Gamma, Vega, and Theta for interest rate swaps, European swaptions,
and Bermudan/American swaptions.

**Method: JAX automatic differentiation, scaled to ORE's bump convention --
not literal bump-and-revalue.** ORE itself computes Delta/Gamma via
`OREAnalytics::SensitivityAnalysis`/`SensitivityScenarioGenerator`
(`OREAnalytics/orea/engine/sensitivityanalysis.cpp`,
`OREAnalytics/orea/scenario/sensitivityscenariogenerator.cpp`): bump one
yield-curve pillar at a time by a configured size (ORE's own example
config, `Examples/MarketRisk/Input/sensitivity.xml`, uses `ShiftType =
Absolute`, `ShiftSize = 0.0001` -- 1bp -- for discount/index curves),
reprice the whole portfolio under each bumped scenario, and finite-difference
the resulting NPVs (`OREAnalytics/orea/cube/sensitivitycube.cpp`:
`delta = NPV_up - NPV_base`, `gamma = NPV_up - 2*NPV_base + NPV_down`, for
a `Forward`-scheme, single-bump-size configuration -- ORE's own default).
This module computes the mathematically identical quantity a different
way: the gradient and Hessian diagonal of NPV with respect to each curve
pillar's zero rate (via `jax.grad` and Hessian-vector products -- see
`_grad_and_hessian_diagonal`), scaled by the same 1bp bump size, giving
ORE's exact "dollar Delta/Gamma for a 1bp move" with no finite-difference
truncation error and no arbitrary step-size choice. This mirrors ORE's own design decision to
maintain closed-form `DiscountingSwapEngineDeltaGamma`/
`BlackSwaptionEngineDeltaGamma` engines as an independent check on its
bump-and-revalue numbers (`QuantExt/qle/pricingengines/
discountingswapenginedeltagamma.hpp`,
`blackswaptionenginedeltagamma.hpp`, exercised by
`OREAnalytics/test/sensitivityvsanalytic.cpp`) -- this module goes one
step further and uses the closed-form (autodiff) route as the primary
implementation, not just a validation side-channel, since it is exact
rather than approximate.

**Bucketed (per-pillar), triangular-interpolated, matching ORE's default --
not a single parallel shift.** ORE's `ShiftScenarioGenerator::applyShift`
(`OREAnalytics/orea/scenario/shiftscenariogenerator.cpp`) bumps one pillar
at a time with a triangular ("tent") weight that ramps from 0 at the
neighboring pillars to 1 at the bumped pillar itself (flat-extrapolated
beyond the first/last pillar). The curve interpolation this module uses
(`_zero_rate_at`, linear on zero rates via `jnp.interp`, flat at the
ends -- identical convention to `compute_hw_A`/`_initial_log_discount` in
`european_swaption.py`) has exactly that same piecewise-linear support, so
differentiating NPV with respect to a single pillar's zero rate via
`jax.grad` automatically produces the same triangular sensitivity ORE's
explicit bump shape encodes -- no separate bump-shape code is needed here.

**Rho:** ORE has no separate "Rho" concept for rate-sensitive instruments
-- `QuantExt::RiskFactorKey::KeyType` has no rho-specific entry, and
`ReportWriter::writeSensitivityReport` emits only "Delta"/"Gamma" columns
for whatever risk factor was bumped (`OREAnalytics/orea/report/
reportwriter.cpp`) -- so an interest-rate-curve Delta (as computed here) IS
ORE's own equivalent of a textbook "Rho." No separate Rho function exists
in this module.

**Vega (Bermudan/American only -- see `bermudan_vega` below).** ORE's own
swaption Vega bumps the market-quoted implied-volatility surface used to
CALIBRATE the model
(`SensitivityScenarioGenerator::generateSwaptionVolScenarios`), not a raw
model parameter -- `hw_sigma` on its own, with no calibration step behind
it, is not an ORE risk factor. `engine.calibration` (added after this
module was first written) now provides exactly that missing calibration
step (`engine.calibration.lgm.calibrate_lgm_sigma`, a bootstrap fit of a
piecewise `Sigma` to a co-terminal basket of market vols -- see that
module's own docstring), which makes a genuine, ORE-equivalent Vega
well-defined for the FIRST time in this codebase: `d(NPV)/d(market_vol) =
d(NPV)/d(sigma) * d(sigma)/d(market_vol)`, computed via the implicit
function theorem through the calibration bootstrap's own root-find (see
`bermudan_vega`'s docstring for the full derivation) rather than literally
re-running calibration once per bumped market vol (which is what ORE
itself does, and which this module deliberately avoids as unnecessary
extra cost given the closed-form alternative). `swap`/`european_swaption`
still have no Vega here (no genuine per-instrument sigma sensitivity is
meaningful for a linear swap; a European swaption's `SwaptionConfig` was
never migrated to accept a calibrated `Sigma` the way `bermudan_swaption`
was -- see `docs/planning/roadmap-and-history.md`).

**Theta.** ORE's Theta (`SensitivityAnalysis::generateSensitivities`,
`OREAnalytics/orea/engine/sensitivityanalysis.cpp`, lines ~253-307)
advances the evaluation date by a configured period (default 1 day), holds
every market quote's own shape/level fixed (no re-simulation, no
re-fitting), rebuilds the term structures at the new "today" (so the SAME
curve object yields different discount factors when read from a later
reference date -- a pure roll/carry effect), reprices, and adds back any
cashflow paid in the interim so a coupon payment isn't misread as a value
loss: `Theta = NPV(t+dt, same curve) - NPV(t) + CF(t, t+dt)`. This module
reproduces that definition exactly (see `swap_theta`/`swaption_theta`/
`bermudan_theta` below) -- a real forward-difference along the time axis
(there is no meaningful "derivative" of an evaluation date to autodiff),
not a `jax.grad` computation.

**Bermudan/American Delta/Gamma/Vega/Theta (`bermudan_delta_gamma`,
`bermudan_vega`, `bermudan_theta` below).** Feasible since Phase 2 ported
`bermudan_swaption.py`'s backward induction to `jax.lax.scan` (previously a
plain-NumPy grid method with no JAX computational graph at all -- the
reason this module originally scoped Bermudan/American Greeks out
entirely, see `docs/planning/roadmap-and-history.md`). An
`AmericanSwaptionConfig` is priced by the same prepared backward induction
as a `BermudanSwaptionConfig` (it differs only in its option times and
exercise style), so one set of functions here takes either -- there is no
separate `american_*` Greeks function.
"""
import dataclasses
from typing import Dict, Union

import jax
import jax.numpy as jnp
import numpy as np
import ORE

from engine.instruments.swap import SwapConfig, _price_one_swap, prepare_swap
from engine.instruments.european_swaption import (
    SwaptionConfig,
    _bond_call,
    _bond_option_sigma,
    _bond_put,
    _hw_B,
    _solve_rstar,
    prepare_swaption,
)
from engine.instruments.bermudan_swaption import (
    BermudanSwaptionConfig,
    _PreparedBermudan,
    _run_backward_induction,
    prepare_bermudan,
)
from engine.models.ore_builders import (  # noqa: F401  (DAY_COUNTER is a re-export)
    DAY_COUNTER,
    TIME_AXIS_DAY_COUNTER,
    build_vanilla_swap,
    fixed_leg_cashflows,
    floating_leg_cashflows,
)
from engine.models.hull_white import A as _hw_A, ZeroCurve, discount as _discount_at, zero_rate as _zero_rate_at
from engine.models.lgm import Sigma, as_sigma

# ORE's own example sensitivity config default (Examples/MarketRisk/Input/
# sensitivity.xml, <DiscountCurve>/<IndexCurve> <ShiftType>Absolute</ShiftType>
# <ShiftSize>0.0001</ShiftSize>) -- a 1 basis point absolute zero-rate bump.
DEFAULT_RATE_BUMP = 0.0001

# ORE's own default Theta horizon (SensitivityAnalysis's thetaPeriod_
# constructor default, sensitivityanalysis.hpp) -- advance the evaluation
# date by one calendar day, holding the market curve's shape/quotes fixed.
DEFAULT_THETA_DAYS = 1

#: The simulation TIME AXIS day count and its deprecated `DAY_COUNTER`
#: alias are imported from `engine.models.ore_builders` (see the import
#: above), not re-constructed here -- see that module's TWO ROLES block.
#: Permanently ACT/365; used only to turn dates into year-fractions on the
#: simulation axis, never as an accrual basis.

# `ZeroCurve`, curve interpolation (`_zero_rate_at`/`_discount_at`), and the
# JAX-differentiable A(t,T) formula all now come directly from
# `engine.models.hull_white` -- the single shared implementation this
# module used to duplicate (see that module's docstring). The main pricers
# (`price_swaps`/`price_swaptions`) also use it now, so `swap_delta_gamma`/
# `swaption_delta_gamma` below differentiate the exact same formula the
# forward pricers evaluate, not a separately-maintained JAX twin of it.


# =============================================================================
# GRADIENT + HESSIAN-DIAGONAL (shared by every Delta/Gamma function below)
# =============================================================================
def _grad_and_hessian_diagonal(price_fn, x, *rest):
    """`(d f/d x_i, d^2 f/d x_i^2)` for every `i` -- the gradient and the
    DIAGONAL of the Hessian, without ever materializing the Hessian.

    **Why not `jnp.diagonal(jax.hessian(f)(x))`.** Every Delta/Gamma
    function in this module reports only the same-pillar second partial
    (ORE's own `SensitivityCube::gamma` is a cross-SCENARIO second
    difference, so it has no cross-pillar term to match -- see
    `swap_delta_gamma`'s docstring). Building the full `[n, n]` Hessian to
    keep `n` of its entries means `jax.hessian`'s forward-over-reverse
    (`jacfwd(jacrev(f))`) traces the whole pricer `n` times over and
    discards `n^2 - n` of the results. This computes each diagonal entry as
    one Hessian-vector product against a basis vector instead:
    `hvp(f, x, e_i)[i] == d^2 f / d x_i^2`, at one gradient's cost each.

    The two are mathematically identical for any twice-differentiable `f`
    (the HVP *is* a Hessian row; taking its `i`-th entry picks the diagonal
    element), which
    `tests/test_profiling_and_jit.py::TestHessianDiagonalEquivalence` pins
    directly against `jnp.diagonal(jax.hessian(...))` for all three
    instrument types, plus an analytic case with a known closed form.

    `rest` carries any further positional arguments `price_fn` takes and
    which are held FIXED here (e.g. `_bermudan_price_fn`'s `sigma_values`,
    or the opposite curve in the two-curve swap case) -- differentiation is
    always with respect to the FIRST argument, `x`.

    **One jitted program for both outputs.** Gradient and Hessian diagonal
    are computed inside a single `jax.jit`, so the whole derivative -- not
    just the pricer it differentiates -- compiles to one XLA program instead
    of dispatching its elementwise work op-by-op. On the reference Bermudan
    trade this is the difference between 470 compilations and 5.

    **Known residue: one compile per CALL, not per curve shape.** `price_fn`
    is a fresh closure every time (each `_*_price_fn` rebuilds it, capturing
    that trade's own prepared structure), and `jax.jit` treats a new Python
    function object as a new function -- so `combined` below recompiles on
    each call even for an identical trade. Measured: a repeated
    `bermudan_delta_gamma` costs 1 compilation rather than 0. That is a
    ~40-70x improvement on where this started, so it was not addressed in the
    same change. It is now filed as **`docs/known-issues.md` I-21**, with the
    full callsite enumeration and a prototyped fix (memoize the jitted wrapper
    on `static_key(prepared)`, verified to reach zero steady-state recompiles
    with bit-identical output and no key collisions). Read that entry before
    attempting it -- a memo keyed wrongly returns a program compiled for a
    DIFFERENT trade, which is silently wrong numbers rather than slowness.
    """
    def combined(xi, *fixed):
        def f(inner):
            return price_fn(inner, *fixed)

        grad = jax.grad(f)(xi)

        # vmap over the basis vectors runs all n HVPs as one batched program
        # rather than n separate dispatches -- the same reason jax.hessian
        # itself is written as a vmap'd jacfwd internally.
        basis = jnp.eye(xi.shape[0], dtype=xi.dtype)

        def hvp(v):
            return jax.jvp(jax.grad(f), (xi,), (v,))[1]

        rows = jax.vmap(hvp)(basis)   # [n, n]; only its diagonal escapes
        return grad, jnp.diagonal(rows)

    return jax.jit(combined)(x, *rest)


# =============================================================================
# SWAP: DELTA / GAMMA
# =============================================================================
def _yield_curves_from_zero_curves(
    disc_curve: ZeroCurve, fwd_curve: ZeroCurve, maturities: jax.Array,
) -> jax.Array:
    """Builds the `[1, 1, Maturities, 2]` yield_curves tensor
    `_price_one_swap` expects, directly from two `ZeroCurve`s -- a
    deterministic (no simulated noise), differentiable stand-in for
    `generate_paths(...)["yield_curves"]`'s zero-shock t=0 slice. Curve
    index 0 is always the discount curve, index 1 the forward curve, in
    this tensor -- a fixed local convention for this Greeks module only
    (independent of whatever discount_curve_index/forward_curve_index a
    caller's SwapConfig was built with against some larger simulation
    cube), which is why `swap_delta_gamma`/`swap_theta` below construct
    their own single-swap `SwapConfig` copy with indices forced to
    (0, 1)."""
    disc = _discount_at(disc_curve, maturities)
    fwd = _discount_at(fwd_curve, maturities)
    cube = jnp.stack([disc, fwd], axis=-1)  # [Maturities, 2]
    return cube[None, None, :, :]


def _swap_price_fn(cfg: SwapConfig, disc_curve: ZeroCurve, fwd_curve: ZeroCurve):
    """Builds a `(disc_rates, fwd_rates) -> t=0 NPV` closure, differentiable
    end-to-end via `jax.grad`/`jax.hessian`. `cfg`'s own trade structure
    (schedule, day counts) is resolved via ORE ONCE, outside the returned
    closure -- only the curve-to-NPV tensor math is retraced per gradient
    evaluation, matching every other pricer's own CPU-setup/GPU-math split
    (see swap.py's module docstring).

    Unlike the main pricer (which requires every cashflow date to land
    EXACTLY on a pre-tabulated `maturities` pillar -- see
    swap.py's maturity-pillar-alignment limitation), this function uses
    the swap's own real cashflow dates AS the "maturities" array and
    evaluates the curve continuously at exactly those dates via
    `_discount_at`'s interpolation -- there is no pillar-mismatch
    constraint to satisfy here, since the yield-curve tensor is built
    fresh (via interpolation, not lookup) for exactly the dates this one
    trade needs."""
    local_cfg = SwapConfig(
        notional=cfg.notional, fixed_rate=cfg.fixed_rate, payer=cfg.payer,
        discount_curve_index=0, forward_curve_index=1,
        swap_tenor=cfg.swap_tenor, index_tenor_months=cfg.index_tenor_months,
        floating_spread=cfg.floating_spread, evaluation_date=cfg.evaluation_date,
    )
    swap = build_vanilla_swap(
        notional=local_cfg.notional, fixed_rate=local_cfg.fixed_rate, payer=local_cfg.payer,
        swap_tenor=local_cfg.swap_tenor, index_tenor_months=local_cfg.index_tenor_months,
        floating_spread=local_cfg.floating_spread, evaluation_date=local_cfg.evaluation_date,
    )
    today = local_cfg.evaluation_date
    fixed = fixed_leg_cashflows(swap, today)
    floating = floating_leg_cashflows(swap, today)

    # The swap's own cashflow dates, deduplicated -- exactly the set of
    # times the curve needs to be evaluated at (via interpolation, not a
    # fixed pillar lookup), used both as the "maturities" pillar array
    # AND as the query points for the differentiable curve.
    maturities = sorted(set(
        fixed.payment_times.tolist() + floating.payment_times.tolist()
        + floating.accrual_start_times.tolist() + floating.accrual_end_times.tolist()
    ))
    prepared_swap = prepare_swap(local_cfg, np.asarray(maturities))
    # Derived from disc_curve's own dtype (not hardcoded) -- disc_curve/
    # fwd_curve carry whatever dtype the caller built them at (governed by
    # PrecisionConfig.risk when reached via engine.portfolio.request), and
    # mixing a hardcoded-float64 array with a float32 curve inside price_fn
    # below would silently upcast the curve back to float64 under
    # jax_enable_x64=True (confirmed: jnp.interp promotes a float32/float64
    # mix to float64 whenever x64 is enabled, regardless of which operand is
    # which dtype) -- see this module's docstring and engine.portfolio.
    # request's "Concurrency" section on why jax_enable_x64 being process-
    # global makes this a genuine, not theoretical, correctness gap.
    maturities_jax = jnp.asarray(maturities, dtype=disc_curve.pillar_rates.dtype)

    def price_fn(disc_rates: jax.Array, fwd_rates: jax.Array) -> jax.Array:
        dc = ZeroCurve(pillar_times=disc_curve.pillar_times, pillar_rates=disc_rates)
        fc = ZeroCurve(pillar_times=fwd_curve.pillar_times, pillar_rates=fwd_rates)
        yield_curves = _yield_curves_from_zero_curves(dc, fc, maturities_jax)
        npv = _price_one_swap(yield_curves, prepared_swap)
        return npv[0, 0]

    return price_fn


def swap_delta_gamma(
    cfg: SwapConfig,
    disc_curve: ZeroCurve,
    fwd_curve: ZeroCurve,
    bump_size: float = DEFAULT_RATE_BUMP,
) -> Dict[str, jax.Array]:
    """
    Per-pillar Delta and Gamma of one swap's t=0 NPV with respect to its
    own discount and forward zero curves, in ORE's own units (dollar NPV
    change for a `bump_size`, 1bp by default, absolute zero-rate move at
    that pillar -- see module docstring's "Method" section).

    Returns a dict with `"discount_delta"`/`"discount_gamma"` (w.r.t.
    `disc_curve.pillar_rates`) and `"forward_delta"`/`"forward_gamma"`
    (w.r.t. `fwd_curve.pillar_rates`), each shaped `[len(pillar_rates)]`.
    Gamma is the pure second partial `d^2 NPV / d rate_j^2` at each
    pillar -- ORE's own `SensitivityCube::gamma` is a cross-SCENARIO
    second difference and so only ever reports this same-pillar diagonal
    term, never a cross-pillar Hessian; this function matches that scope
    (the diagonal of `jax.hessian`, not the full pillar-by-pillar matrix).
    If `disc_curve is fwd_curve` (single-curve discounting), the two
    curves still get independent gradients from this function's
    perspective (each treated as its own free variable) -- summing
    `discount_delta + forward_delta` pillar-by-pillar recovers the total
    sensitivity to that one shared curve in that case.
    """
    price_fn = _swap_price_fn(cfg, disc_curve, fwd_curve)

    # `_grad_and_hessian_diagonal` jits internally (see its docstring) and
    # always differentiates its FIRST array argument, so the forward-curve
    # call passes the two curves in swapped order behind a small adapter.
    disc_delta, disc_gamma = _grad_and_hessian_diagonal(
        price_fn, disc_curve.pillar_rates, fwd_curve.pillar_rates
    )
    fwd_delta, fwd_gamma = _grad_and_hessian_diagonal(
        lambda fwd_rates, disc_rates: price_fn(disc_rates, fwd_rates),
        fwd_curve.pillar_rates, disc_curve.pillar_rates,
    )

    return {
        "discount_delta": disc_delta * bump_size,
        "discount_gamma": disc_gamma * bump_size ** 2,
        "forward_delta": fwd_delta * bump_size,
        "forward_gamma": fwd_gamma * bump_size ** 2,
    }


# =============================================================================
# SWAP: THETA
# =============================================================================
def swap_theta(
    cfg: SwapConfig,
    disc_curve: ZeroCurve,
    fwd_curve: ZeroCurve,
    theta_days: int = DEFAULT_THETA_DAYS,
) -> float:
    """
    Theta = NPV(today + theta_days, SAME curve) - NPV(today) +
    cashflow paid in (today, today+theta_days] -- ORE's own definition
    (see module docstring's "Theta" section), reproduced literally:
    `disc_curve`/`fwd_curve` are the SAME curve objects at both valuation
    dates (no re-simulation, no re-fitting -- only the evaluation date and
    each cashflow's own year-fraction-from-today shift, since the curves
    are read from the new reference date), and any coupon whose payment
    date falls in the interim is added back so it isn't misread as a
    value loss.

    Unlike swap_delta_gamma, this is NOT an autodiff computation -- an
    evaluation date has no continuous derivative to take; it is a literal
    forward difference along the time axis, exactly mirroring ORE's own
    finite-difference-in-time Theta.
    """
    # Both valuations jitted: each is one compiled program instead of an
    # eager op-by-op walk through the whole pricer (see
    # `docs/concepts/profiling.md`). The two dates produce two DIFFERENT
    # trade structures (different cashflow year-fractions), so they are
    # genuinely two programs, not a cache hit -- jitting still collapses
    # each one's own internal dispatch.
    base_price_fn = _swap_price_fn(cfg, disc_curve, fwd_curve)
    base_npv = float(jax.jit(base_price_fn)(disc_curve.pillar_rates, fwd_curve.pillar_rates))

    theta_date = ORE.TARGET().advance(cfg.evaluation_date, theta_days, ORE.Days)
    theta_cfg = SwapConfig(
        notional=cfg.notional, fixed_rate=cfg.fixed_rate, payer=cfg.payer,
        discount_curve_index=cfg.discount_curve_index, forward_curve_index=cfg.forward_curve_index,
        swap_tenor=cfg.swap_tenor, index_tenor_months=cfg.index_tenor_months,
        floating_spread=cfg.floating_spread, evaluation_date=theta_date,
    )
    theta_price_fn = _swap_price_fn(theta_cfg, disc_curve, fwd_curve)
    theta_npv = float(jax.jit(theta_price_fn)(disc_curve.pillar_rates, fwd_curve.pillar_rates))

    period_flow = _swap_cashflows_in_period(cfg, cfg.evaluation_date, theta_date)

    return theta_npv - base_npv + period_flow


def _swap_cashflows_in_period(cfg: SwapConfig, start: ORE.Date, end: ORE.Date) -> float:
    """Sums every fixed/floating cashflow (signed per cfg.payer, matching
    _price_one_swap's own payer-negation convention) whose payment date
    falls in `(start, end]` -- ORE's own `aggregateTradeFlow` step in
    `SensitivityAnalysis::generateSensitivities`
    (`OREAnalytics/orea/engine/sensitivityanalysis.cpp`), which prevents a
    coupon paid during the Theta horizon from being misattributed as a
    pure valuation loss. This module doesn't yet simulate what the
    floating leg's rate WOULD be over (start, end] (a fixing that hasn't
    happened yet as of `start`), so this only sums the FIXED leg's
    cashflows in the window -- documented, not silent: a floating payment
    landing inside a Theta window this short (the default is 1 day) is
    the overwhelmingly common case where this simplification is exact
    anyway, since a swap's floating leg pays only on its own (typically
    monthly-or-longer) reset dates, essentially never within a single
    day of `start`."""
    ORE.Settings.instance().evaluationDate = cfg.evaluation_date
    swap = build_vanilla_swap(
        notional=cfg.notional, fixed_rate=cfg.fixed_rate, payer=cfg.payer,
        swap_tenor=cfg.swap_tenor, index_tenor_months=cfg.index_tenor_months,
        floating_spread=cfg.floating_spread, evaluation_date=cfg.evaluation_date,
    )
    fixed = fixed_leg_cashflows(swap, cfg.evaluation_date)

    start_frac = TIME_AXIS_DAY_COUNTER.yearFraction(cfg.evaluation_date, start)
    end_frac = TIME_AXIS_DAY_COUNTER.yearFraction(cfg.evaluation_date, end)
    in_window = (fixed.payment_times > start_frac) & (fixed.payment_times <= end_frac)
    fixed_flow = float(np.sum(
        in_window * fixed.notional * cfg.fixed_rate * fixed.accrual_fractions
    ))
    # Fixed leg is PAID by a payer, so it's a cash outflow (negative to the
    # payer's own NPV convention) -- matches _price_one_swap's
    # `npv = float_leg_pv - fixed_leg_pv` sign, negated again for payer=False.
    signed_flow = -fixed_flow if cfg.payer else fixed_flow
    return signed_flow


# =============================================================================
# EUROPEAN SWAPTION: DELTA / GAMMA
# =============================================================================
def _swaption_price_fn(cfg: SwaptionConfig, curve: ZeroCurve):
    """Builds a `curve_rates -> t=0 NPV` closure for a single European
    swaption, differentiable via `jax.grad`/`jax.hessian` with respect to
    `curve.pillar_rates` -- reimplements `_price_one_swaption`'s t=0 path
    using `engine.models.hull_white.A` (the SAME shared, JAX-native
    formula `price_swaptions`' own NumPy-facing `compute_hw_A` wraps) in
    place of a separately-maintained JAX twin, and
    otherwise reusing the SAME JAX building blocks the main pricer already
    uses (`_hw_B`, `_bond_option_sigma`, `_bond_call`/`_bond_put`,
    `_solve_rstar`) -- only the today's-curve lookup changes, not the
    option-pricing formulas themselves. Trade structure (cashflow times/
    amounts, exercise time) is resolved via `prepare_swaption` ONCE,
    outside the returned closure, exactly like `_swap_price_fn` above."""
    swaption = prepare_swaption(cfg)
    a = swaption.hw_a
    sigma = swaption.hw_sigma
    T0 = swaption.exercise_time
    T_start = swaption.accrual_start_time
    cf_times = swaption.fixed_cashflow_times
    notional = swaption.notional
    # Derived from curve's own dtype (not hardcoded) -- see
    # _swap_price_fn's maturities_jax comment above for why a hardcoded
    # dtype here would silently upcast curve.pillar_rates back to float64
    # under jax_enable_x64=True whenever a caller requests risk=32.
    _dtype = curve.pillar_rates.dtype
    all_times = jnp.asarray(
        np.concatenate([cf_times, cf_times[-1:], [T_start]]), dtype=_dtype
    )
    all_amounts = jnp.concatenate([
        jnp.asarray(swaption.fixed_cashflow_amounts, dtype=_dtype),
        jnp.asarray([notional, -notional], dtype=_dtype),
    ])
    B_T0_Ti = _hw_B(T0, all_times, a)  # [N+1]

    def coupon_bond_value(rstar, params):
        # rstar: scalar. params = A_T0_Ti -- the explicit pytree
        # _solve_rstar's jax.custom_jvp differentiates with respect to
        # (see european_swaption._solve_rstar's docstring on why this
        # must be explicit, not a closure).
        A_T0_Ti = params
        prices = A_T0_Ti * jnp.exp(-B_T0_Ti * rstar)
        return jnp.sum(prices * all_amounts)

    def price_fn(pillar_rates: jax.Array) -> jax.Array:
        curve_local = ZeroCurve(pillar_times=curve.pillar_times, pillar_rates=pillar_rates)
        A_T0_Ti = _hw_A(curve_local, jnp.full_like(all_times, T0), all_times, a, sigma)

        rstar = _solve_rstar(coupon_bond_value, A_T0_Ti, ())
        K = A_T0_Ti * jnp.exp(-B_T0_Ti * rstar)

        # t=0 conditioning: P(t,S) = A(t,S)*exp(-B(t,S)*r(t)) for ANY t,
        # including t=0 -- B(0,S) is NOT zero (only t==S makes B(t,S)==0),
        # so r(0) genuinely matters here, exactly as it does at every other
        # step in _price_one_swaption. r(0), the model's own "no shock"
        # short rate, is the curve's own initial instantaneous forward
        # rate f(0,0) -- the same quantity _price_one_swaption reads off
        # hw_paths[:, 0, :] at the simulation's own first step (by
        # construction, RatesConfig.initial_rates is chosen to equal the
        # calibration curve's own short end in every scenario in this
        # codebase). Computed here as the curve's own zero rate at the
        # (numerically tiny but nonzero) limit t->0, matching
        # compute_hw_A's own f(0,t) finite-difference convention.
        r0 = _zero_rate_at(curve_local, jnp.asarray(1e-6))
        B_0_Ti = _hw_B(0.0, all_times, a)
        B_0_T0 = _hw_B(0.0, T0, a)
        A_0_Ti = _hw_A(curve_local, jnp.zeros_like(all_times), all_times, a, sigma)
        A_0_T0 = _hw_A(curve_local, jnp.zeros(()), T0, a, sigma)
        P_0_Ti = A_0_Ti * jnp.exp(-B_0_Ti * r0)
        P_0_T0 = A_0_T0 * jnp.exp(-B_0_T0 * r0)

        sigma_p = _bond_option_sigma(T0, all_times, 0.0, a, sigma)
        bond_fn = _bond_put if swaption.payer else _bond_call
        per_leg = bond_fn(P_0_T0, P_0_Ti, K, sigma_p)
        return jnp.sum(per_leg * all_amounts)

    return price_fn


def swaption_delta_gamma(
    cfg: SwaptionConfig,
    curve: ZeroCurve,
    bump_size: float = DEFAULT_RATE_BUMP,
) -> Dict[str, jax.Array]:
    """
    Per-pillar Delta and Gamma of one European swaption's t=0 NPV with
    respect to its own Hull-White calibration curve, in ORE's own units
    (dollar NPV change for a `bump_size`, 1bp by default, absolute
    zero-rate move at that pillar -- see module docstring's "Method"
    section).

    `curve` should be the SAME zero curve as `cfg.initial_zero_curve`
    (as `ZeroCurve.pillar_rates` rather than a plain list) -- this
    function does not read `cfg.initial_zero_curve` itself, since that
    field is a plain-Python `ZeroCurveConfig`, not a JAX array; callers
    build the matching differentiable `ZeroCurve` explicitly (see
    `tests/test_greeks.py` for the pattern).

    Returns `{"delta": [...], "gamma": [...]}`, each shaped
    `[len(curve.pillar_rates)]`. Gamma is the pure second partial at each
    pillar (the Hessian's diagonal, computed via Hessian-vector products
    rather than by building the matrix -- see `_grad_and_hessian_diagonal`),
    matching ORE's own `SensitivityCube::gamma` scope -- see `swap_delta_gamma`'s docstring
    for why only the diagonal is reported.
    """
    price_fn = _swaption_price_fn(cfg, curve)
    delta, gamma = _grad_and_hessian_diagonal(price_fn, curve.pillar_rates)
    return {
        "delta": delta * bump_size,
        "gamma": gamma * bump_size ** 2,
    }


# =============================================================================
# EUROPEAN SWAPTION: THETA
# =============================================================================
def swaption_theta(
    cfg: SwaptionConfig,
    curve: ZeroCurve,
    theta_days: int = DEFAULT_THETA_DAYS,
) -> float:
    """
    Theta = NPV(today + theta_days, SAME curve) - NPV(today) -- ORE's own
    definition (see module docstring's "Theta" section). A European
    swaption pays no interim cashflow before its own exercise date (see
    `european_swaption.py`'s module docstring: the underlying swap's
    cashflows only matter AT exercise, decomposed into the option's
    payoff, not paid independently beforehand), so there is no
    `+ period_flow` term to add back here, unlike `swap_theta` -- Theta is
    a pure repricing difference. Uses the same t=0 (unconditional)
    valuation as `swaption_delta_gamma`, at two different evaluation
    dates -- not the conditional (t>0, simulated-path) pricing
    `price_swaptions` itself supports, since Theta by definition compares
    two DETERMINISTIC valuations (today vs. today+1) under the SAME
    (unshocked) curve, not a simulated scenario.
    """
    # Jitted for the same reason as `swap_theta`'s own pair -- see the
    # comment there and `docs/concepts/profiling.md`. This one mattered
    # most: measured at 56 separate XLA compilations before jitting (the
    # Jamshidian r* solve dispatches a long elementwise chain eagerly),
    # against 11 for the whole grad+Hessian-diagonal Delta/Gamma pair.
    base_price_fn = _swaption_price_fn(cfg, curve)
    base_npv = float(jax.jit(base_price_fn)(curve.pillar_rates))

    theta_date = ORE.TARGET().advance(cfg.evaluation_date, theta_days, ORE.Days)
    theta_cfg = SwaptionConfig(
        notional=cfg.notional, fixed_rate=cfg.fixed_rate, payer=cfg.payer,
        rate_factor_index=cfg.rate_factor_index, hw_a=cfg.hw_a, hw_sigma=cfg.hw_sigma,
        initial_zero_curve=cfg.initial_zero_curve, swap_tenor=cfg.swap_tenor,
        index_tenor_months=cfg.index_tenor_months, floating_spread=cfg.floating_spread,
        forward_start=cfg.forward_start, exercise_lag_days=cfg.exercise_lag_days,
        evaluation_date=theta_date,
    )
    theta_price_fn = _swaption_price_fn(theta_cfg, curve)
    theta_npv = float(jax.jit(theta_price_fn)(curve.pillar_rates))

    return theta_npv - base_npv


# =============================================================================
# BERMUDAN / AMERICAN SWAPTION: DELTA / GAMMA / VEGA / THETA
# =============================================================================
def _bermudan_price_fn(cfg: BermudanSwaptionConfig, curve: ZeroCurve):
    """
    Builds a `(pillar_rates, sigma_values) -> t=0 NPV` closure for a single
    Bermudan/American swaption, differentiable via `jax.grad`/`jax.hessian`
    with respect to BOTH the curve's own pillar rates (Delta/Gamma) and the
    calibrated Sigma's own bucket values (Vega) -- `_run_backward_
    induction` is already fully JAX-native end-to-end as of Phase 2 (no
    NumPy/Python control flow standing between these inputs and the
    output), so this closure needs no separate JAX reimplementation of the
    backward induction the way `_swaption_price_fn` needed for Jamshidian's
    (originally NumPy) formula -- it reuses `prepare_bermudan`/
    `_run_backward_induction` directly, only substituting differentiable
    values for `_PreparedBermudan.zero_rates`/`hw_sigma` via
    `dataclasses.replace`.

    `sigma_values` is always a flat JAX array -- `len(cfg.hw_sigma.values)`
    if `cfg.hw_sigma` is already a `Sigma` (the calibrated-Vega case), or a
    single-element array wrapping a flat scalar otherwise (Delta/Gamma-only
    callers, or a flat-sigma trade with no calibration behind it). Bucket
    BREAKPOINTS (`Sigma.times`) are never a differentiation target (mirrors
    `ZeroCurve.pillar_times` never being one for Delta/Gamma -- ORE's own
    sensitivity framework bumps a bucket's VALUE, never its own time grid).

    `cfg.hw_sigma` (NOT this closure's own `curve` argument) determines the
    initial zero curve `_PreparedBermudan.zero_times` uses -- unlike
    `_swaption_price_fn`, whose caller passes the curve as a wholly
    separate argument, `cfg.initial_zero_curve` is what `prepare_bermudan`
    reads to set `zero_times`; the `curve` parameter here supplies the
    PILLAR TIMES (`curve.pillar_times` must equal `cfg.initial_zero_curve.
    times` -- the same "same curve, JAX-native copy" convention
    `swaption_delta_gamma`'s own docstring documents) while `pillar_rates`
    becomes the differentiable rate values substituted in.
    """
    from dataclasses import replace

    swap = prepare_bermudan(cfg)
    base_sigma = as_sigma(cfg.hw_sigma)
    sigma_times = base_sigma.times

    def price_fn(pillar_rates: jax.Array, sigma_values: jax.Array) -> jax.Array:
        local_swap = replace(
            swap,
            zero_rates=pillar_rates,
            hw_sigma=Sigma(times=sigma_times, values=sigma_values),
        )
        result = _run_backward_induction(local_swap, condition_times=[])
        return result.value_at_t0

    return price_fn, base_sigma.values


def bermudan_delta_gamma(
    cfg: BermudanSwaptionConfig,
    curve: ZeroCurve,
    bump_size: float = DEFAULT_RATE_BUMP,
) -> Dict[str, jax.Array]:
    """
    Per-pillar Delta and Gamma of one Bermudan/American swaption's t=0 NPV
    with respect to its own LGM calibration curve, in ORE's own units
    (dollar NPV change for a `bump_size`, 1bp by default, absolute
    zero-rate move at that pillar -- see module docstring's "Method"
    section). Same convention as `swaption_delta_gamma` -- `curve` should
    share `cfg.initial_zero_curve`'s own pillar times (see
    `_bermudan_price_fn`'s docstring).

    Returns `{"delta": [...], "gamma": [...]}`, each shaped
    `[len(curve.pillar_rates)]`. Gamma is the pure second partial at each
    pillar (the Hessian's diagonal, computed via Hessian-vector products
    rather than by building the matrix -- see `_grad_and_hessian_diagonal`),
    matching ORE's own `SensitivityCube::gamma` scope.
    """
    price_fn, sigma_values = _bermudan_price_fn(cfg, curve)
    delta, gamma = _grad_and_hessian_diagonal(price_fn, curve.pillar_rates, sigma_values)
    return {
        "delta": delta * bump_size,
        "gamma": gamma * bump_size ** 2,
    }


def bermudan_theta(
    cfg: BermudanSwaptionConfig,
    curve: ZeroCurve,
    theta_days: int = DEFAULT_THETA_DAYS,
) -> float:
    """
    Theta = NPV(today + theta_days, SAME curve) - NPV(today) -- ORE's own
    definition (see module docstring's "Theta" section), same
    t=0-only/no-interim-cashflow reasoning as `swaption_theta` (a Bermudan/
    American's exercise value already prices in every remaining cashflow
    via the backward induction itself; there is no separately-paid coupon
    between "today" and "today+1" to add back).
    """
    # Jitted for the same reason as `swap_theta`/`swaption_theta` above.
    # `_run_backward_induction` is itself jitted one layer down now (see
    # `bermudan_swaption._backward_induction_arrays`), so this outer jit
    # only folds in the small amount of surrounding work -- but it keeps
    # the Theta path consistent with the others and costs nothing.
    price_fn, sigma_values = _bermudan_price_fn(cfg, curve)
    base_npv = float(jax.jit(price_fn)(curve.pillar_rates, sigma_values))

    # The same trade one day on: its exercise DATES stay put and every time
    # is re-derived from the new evaluation date, exactly as ORE re-derives
    # optionTimes. (When exercise was given as year fractions this silently
    # moved every exercise opportunity a day later as well.)
    theta_date = ORE.TARGET().advance(cfg.evaluation_date, theta_days, ORE.Days)
    theta_cfg = dataclasses.replace(cfg, evaluation_date=theta_date)
    theta_price_fn, theta_sigma_values = _bermudan_price_fn(theta_cfg, curve)
    theta_npv = float(jax.jit(theta_price_fn)(curve.pillar_rates, theta_sigma_values))

    return theta_npv - base_npv


def bermudan_vega(
    cfg: BermudanSwaptionConfig,
    curve: ZeroCurve,
    calibration_targets,
    market_vol_bump: float = 0.0001,
) -> jax.Array:
    """
    Per-basket-instrument Vega: dollar NPV change of the Bermudan/American
    swaption for a `market_vol_bump` (1bp of normal vol by default) move in
    ONE market swaption's own quoted volatility, holding every other market
    quote fixed -- ORE's own `generateSwaptionVolScenarios` bump
    definition, but computed via the implicit function theorem through
    `engine.calibration.lgm.calibrate_lgm_sigma`'s bootstrap root-find
    rather than literally re-running calibration once per bumped market vol
    (what ORE itself does -- this closed-form route is exact, not merely
    faster, avoiding both the bootstrap's own root-find tolerance AND
    finite-difference truncation error).

    `cfg.hw_sigma` MUST be the `Sigma` `calibrate_lgm_sigma` produced from
    `calibration_targets` (in the SAME order) -- this function does not
    re-run calibration itself, only differentiates through the relationship
    between `calibration_targets[i].market_vol` and `cfg.hw_sigma`.

    **Derivation.** The bootstrap calibrates bucket `j`'s sigma value
    `s_j` as the root of `g_j(s_0,...,s_j; v_j) := price_lgm_swaption(...,
    Sigma(times, [s_0,...,s_j]), targets[j]) - bachelier_swaption_price(
    targets[j] with market_vol=v_j, curve) == 0` (see
    `engine.calibration.lgm`'s own docstring: bucket `j` depends on EVERY
    earlier bucket, since `zeta` accumulates, but not on any LATER
    bucket, and only on ITS OWN target's market vol `v_j`, never a later
    target's).

    A bump to `v_i` moves `s_i` (directly, via `g_i`'s own root-find), and
    THROUGH `s_i`, moves every LATER bucket `s_j` (`j>i`) too, since
    `g_j` depends on `s_i` whenever `i<j` -- an earlier version of this
    function assumed `d(s_j)/d(v_i) = 0` for `j != i`, which is WRONG for
    `j>i` (only correct for `j<i`, since the bootstrap is triangular
    forward in time, not backward): confirmed directly via finite
    difference (`tests/test_greeks_bermudan.py`'s Vega cross-check caught
    a systematic 45-78% error in early buckets under that wrong
    assumption, while the LAST bucket -- which genuinely has no later
    bucket to affect -- matched almost exactly, the tell that pinpointed
    the missing cross-bucket term).

    The correct recursion (forward substitution, `j` in increasing order,
    for the FULL lower-triangular Jacobian `ds_j/dv_i`, `i <= j`), by
    total-derivative differentiation of `g_j(s_0,...,s_j; v_j)=0` with
    respect to `v_i`:

        dg_j/ds_j * ds_j/dv_i + sum_{k<j} dg_j/ds_k * ds_k/dv_i
            + dg_j/dv_j * [i==j] = 0

        ds_j/dv_i = -( sum_{k<j} dg_j/ds_k * ds_k/dv_i + dg_j/dv_j*[i==j] )
                    / (dg_j/ds_j)

    (`dg_j/dv_j` only appears in `g_j`'s own equation, i.e. only for
    `i==j`; for `i<j`, `v_i` influences `g_j` PURELY through `s_i`'s own
    already-computed `ds_i/dv_i` -- and, transitively, through every
    `s_k` for `i<=k<j`, captured by the `sum_{k<j}` term using rows
    already computed earlier in the forward substitution). Each `dg_j/
    ds_k` (for `k<=j`) and `dg_j/dv_j` is one `jax.grad` of `price_lgm_
    swaption(..., Sigma(times, [s_0..s_j]), targets[j])` with respect to
    bucket `k`'s own value (or, for the market term, one `jax.grad` of
    `bachelier_swaption_price` w.r.t. `v_j`) -- `O(n_buckets^2)` total
    `jax.grad` calls for the full Jacobian, negligible next to the
    Bermudan pricer's own cost per call.

    Total Vega w.r.t. `v_i` is then the chain rule summed over every
    bucket: `d(NPV)/d(v_i) = sum_j d(NPV)/d(s_j) * d(s_j)/d(v_i)` (NOT
    just the `j==i` term, per the correction above) -- computed via one
    `jax.grad` of the Bermudan price w.r.t. the full sigma_values vector,
    dotted against the Jacobian's own `i`-th column.

    Returns an array of shape `[len(calibration_targets)]`: the Vega to
    each basket instrument's own market vol, in dollars per
    `market_vol_bump` (1bp of normal vol by default).
    """
    from dataclasses import replace as _replace
    from engine.calibration.basket import bachelier_swaption_price, price_lgm_swaption

    sigma = as_sigma(cfg.hw_sigma)
    n = sigma.values.shape[0]
    assert n == len(calibration_targets), (
        "cfg.hw_sigma must be the Sigma calibrate_lgm_sigma produced from "
        "calibration_targets, with one bucket per target"
    )

    price_fn, sigma_values = _bermudan_price_fn(cfg, curve)
    d_npv_d_s = jax.jit(jax.grad(price_fn, argnums=1))(curve.pillar_rates, sigma_values)  # [n]

    # Derived from curve's own dtype (not hardcoded) -- same reasoning as
    # _swaption_price_fn's all_times/all_amounts above: J is combined with
    # d_npv_d_s (which follows curve/sigma_values' own dtype) via the final
    # d_npv_d_s @ J matmul below, so a hardcoded float64 here would silently
    # upcast a risk=32 request's float32 Vega computation back to float64.
    _dtype = curve.pillar_rates.dtype
    # Full lower-triangular Jacobian J[j, i] = d(s_j)/d(v_i), built by
    # forward substitution over j (increasing bucket index).
    #
    # The forward substitution is a genuine sequential Python loop -- row j
    # reads rows <j, so it cannot be vectorized away -- but each row's own
    # two gradients ARE jitted below. Left un-jitted, each bucket dispatched
    # its whole `price_lgm_swaption` gradient op-by-op, and the running
    # `J.at[j, :].set(row)` writes showed up in the profiler trace as
    # ~1000 eager `dynamic_update_index_in_dim` dispatches. Accumulating the
    # rows in a plain Python list and stacking ONCE at the end removes those
    # entirely (one `jnp.stack` instead of n scatter-writes).
    rows = []

    for j, target_j in enumerate(calibration_targets):
        bucket_times_j = sigma.times[:j]

        def model_price_wrt_prefix(values_prefix, _j=j, _bucket_times=bucket_times_j, _target=target_j):
            return price_lgm_swaption(curve, cfg.hw_a, Sigma(times=_bucket_times, values=values_prefix), _target)

        # dg_j/ds_k for every k <= j, via one jax.grad w.r.t. the whole
        # [s_0,...,s_j] prefix (cheaper than j+1 separate scalar grads).
        dg_j_ds = jax.jit(jax.grad(model_price_wrt_prefix))(sigma.values[: j + 1])  # [j+1]

        def market_price_wrt_v_j(v_j, _target=target_j):
            bumped = _replace(_target, market_vol=v_j)
            return bachelier_swaption_price(bumped, curve)

        # g_j := model_price - market_price, so dg_j/dv_j = -d(market_price)/dv_j.
        dg_j_dv_j = -jax.jit(jax.grad(market_price_wrt_v_j))(jnp.asarray(target_j.market_vol))

        if j > 0:
            prev = jnp.stack(rows)                       # [j, n], rows already computed
            cross_term = jnp.sum(dg_j_ds[:j, None] * prev, axis=0)
        else:
            cross_term = jnp.zeros((n,), dtype=_dtype)
        rows.append(-(cross_term.at[j].add(dg_j_dv_j)) / dg_j_ds[j])

    J = jnp.stack(rows)  # [n, n], lower-triangular by construction
    vega_per_unit_vol = d_npv_d_s @ J  # [n]
    return vega_per_unit_vol * market_vol_bump


# =============================================================================
# EXECUTION DEMONSTRATION
# =============================================================================
if __name__ == "__main__":
    from engine.simulation.demo_scenarios import EVAL_DATE

    pillar_times = [1.0, 2.0, 5.0, 10.0, 30.0]

    # --- Swap Delta/Gamma/Theta ---
    disc_curve = ZeroCurve.flat(0.030, pillar_times)
    fwd_curve = ZeroCurve.flat(0.035, pillar_times)
    swap_cfg = SwapConfig(
        notional=1_000_000.0, fixed_rate=0.032, payer=True,
        discount_curve_index=0, forward_curve_index=1,
        swap_tenor="5Y", evaluation_date=EVAL_DATE,
    )
    swap_greeks = swap_delta_gamma(swap_cfg, disc_curve, fwd_curve)
    print("--- Swap Greeks (5Y payer, notional $1MM) ---")
    print("Discount curve Delta ($/1bp):", [round(float(v), 2) for v in swap_greeks["discount_delta"]])
    print("Forward curve Delta ($/1bp): ", [round(float(v), 2) for v in swap_greeks["forward_delta"]])
    print("Discount curve Gamma:        ", [round(float(v), 6) for v in swap_greeks["discount_gamma"]])
    print("Theta (1 day):", round(swap_theta(swap_cfg, disc_curve, fwd_curve), 4))

    # --- European swaption Delta/Gamma/Theta ---
    from engine.simulation.market_model import ZeroCurveConfig

    swaption_curve = ZeroCurve.flat(0.03, pillar_times)
    swaption_cfg = SwaptionConfig(
        notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
        hw_a=0.03, hw_sigma=0.01,
        initial_zero_curve=ZeroCurveConfig(times=pillar_times, rates=[0.03] * len(pillar_times)),
        swap_tenor="5Y", forward_start=ORE.Period(3, ORE.Years), evaluation_date=EVAL_DATE,
    )
    swaption_greeks = swaption_delta_gamma(swaption_cfg, swaption_curve)
    print("\n--- European Swaption Greeks (3Y into 5Y payer, notional $1MM) ---")
    print("Delta ($/1bp):", [round(float(v), 2) for v in swaption_greeks["delta"]])
    print("Gamma:        ", [round(float(v), 6) for v in swaption_greeks["gamma"]])
    print("Theta (1 day):", round(swaption_theta(swaption_cfg, swaption_curve), 4))
