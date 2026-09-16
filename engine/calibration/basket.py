"""
Co-terminal European swaption calibration basket, and the LGM closed-form
swaption pricer each basket instrument is priced with during calibration.

**Co-terminal basket construction**, mirroring
`ore::data::IrModelBuilder::buildSwaptionBasket()`
(`OREData/ored/model/irmodelbuilder.cpp`): one European swaption per
Bermudan/American exercise date, each into a swap that matures on the
SAME final date as the trade being calibrated for (the underlying's own
final maturity) -- so an exercise schedule `[1Y, 2Y, 3Y, 4Y]` against a
trade maturing in `5Y` produces basket instruments `1Yx4Y, 2Yx3Y, 3Yx2Y,
4Yx1Y` (expiry x tenor-to-the-shared-final-maturity). This is ORE's own
"diagonal"/co-terminal convention (as opposed to a "co-initial" basket,
which ORE also supports but does not use by default) -- chosen because a
Bermudan's exercise-into-the-remaining-swap structure is naturally
co-terminal: exercising at `t_i` always means entering the SAME underlying
swap that runs to the trade's final maturity, just with a shorter
remaining tenor.

**Pricing each basket instrument** (`price_lgm_swaption` below): the LGM
analogue of `QuantExt::AnalyticLgmSwaptionEngine` -- Jamshidian's bond-
option decomposition expressed in LGM's own `H`/`zeta`/`x`-state variables
(`engine.models.lgm`), rather than Hull-White's `A`/`B`/`r`-state variables
(`engine.models.hull_white`, which `engine.instruments.european_swaption`
uses). A SEPARATE closed-form pricer from `european_swaption.py`'s
Jamshidian decomposition -- not a duplicate of it -- because the two
modules price under genuinely different model realizations (see
`engine.models.lgm`'s own module docstring: HW1F and LGM are NOT
interchangeable parametrizations of the same model for t>0), and because
this pricer's whole purpose is to be evaluated repeatedly with a TRIAL
`Sigma` during calibration (see `engine/calibration/lgm.py`), which
`european_swaption.py`'s HW1F-only formulas cannot express (LGM's
`Sigma` is genuinely piecewise; HW1F stays constant-only in this codebase
-- see `engine.models.hull_white`'s module docstring on why).

**Live-verified against ORE**, via two independent routes since
`QuantExt::AnalyticLgmSwaptionEngine`'s constructor is not exposed through
this codebase's installed ORE Python bindings (confirmed directly: SWIG
only exposes `enableCache`/`clearCache`/`setZetaShift`/`resetZetaShift` on
`ORE.AnalyticLgmSwaptionEngine`, not a usable constructor -- see
`ORE-SWIG/QuantExt-SWIG/SWIG/qle_pricingengines.i`, which declares no
`%extend` constructor for this class): (1) every individual formula
(`bond_price`, `bond_option_sigma`, `numeraire`) is independently
live-verified to machine precision against `ORE.LinearGaussMarkovModel`'s
own exposed methods (see `engine/models/lgm.py` and
`tests/test_models_piecewise_sigma.py`); (2) the FULL swaption price this
module computes is cross-checked against a numeraire-deflated Monte Carlo
simulation of `x(T0) ~ N(0, zeta(T0))` (the model's own exact terminal
distribution, per `QuantExt::IrLgm1fStateProcess::variance`), discounted
through `engine.models.lgm.numeraire` rather than naively through
`P(0,T0)` -- LGM's own measure is NOT the T0-forward measure, so naive
discounting was tried first and shown to disagree with the closed form by
~11%, a real modeling bug in that MC methodology (not the closed form,
which was independently confirmed correct once the MC was fixed to
properly deflate by the model's own numeraire) -- see this module's test
suite for the corrected version, matching to within Monte Carlo standard
error (~0.05% relative, N=3,000,000 paths).
"""
from dataclasses import dataclass
from typing import List, Union

import jax
import jax.numpy as jnp
import numpy as np
import ORE

from engine.models.hull_white import ZeroCurve, bond_call, bond_put, discount
from engine.models.lgm import Sigma, bond_option_sigma, bond_price
from engine.models.ore_builders import TIME_AXIS_DAY_COUNTER, build_vanilla_swap, fixed_leg_cashflows


@dataclass
class CalibrationTarget:
    """One co-terminal European swaption calibration target: its own
    expiry/underlying cashflow structure (built via a real ORE swap, exact
    date/accrual conventions) plus the market volatility it must be priced
    to match.

    market_vol: a normal (basis-point) volatility -- this codebase prices
    every swaption via the Bachelier/normal-vol convention throughout
    (matching `engine.risk.greeks`'s own Vega convention), NOT
    shifted-lognormal, so `black_price` below uses the Bachelier formula
    directly rather than converting through a shift.
    """
    expiry_time: float                 # T0, year-fraction from evaluation_date
    accrual_start_time: float          # T_start, the underlying's own first accrual date
    fixed_cashflow_times: np.ndarray   # [N] year-fractions from evaluation_date
    fixed_cashflow_amounts: np.ndarray  # [N], at the ATM fixed rate (see build_coterminal_basket)
    fixed_accrual_fractions: np.ndarray  # [N], ORE's own accrualPeriod() per coupon
    notional: float
    payer: bool
    forward_rate: float                # the underlying's own par/ATM rate (== strike, by construction)
    market_vol: float


def build_coterminal_basket(
    exercise_times: List[float],
    final_maturity_time: float,
    notional: float,
    payer: bool,
    market_vols: List[float],
    zero_curve: ZeroCurve,
    evaluation_date: ORE.Date,
    index_tenor_months: int = 6,
) -> List[CalibrationTarget]:
    """
    Builds one co-terminal `CalibrationTarget` per exercise date: a
    European swaption expiring at that date, into a swap running from the
    exercise date to `final_maturity_time` -- ORE's own diagonal/
    co-terminal basket convention (see module docstring).

    Each underlying swap is struck AT-THE-MONEY (its own par rate under
    `zero_curve`, computed directly from the swap's own annuity/discount
    factors -- the standard par-swap-rate identity, not re-derived from
    ORE internals) -- calibration baskets are conventionally ATM (matching
    `ore::data::IrModelBuilder::getStrike` returning `Null<Real>()`, ORE's
    own sentinel for "use the ATM strike", whenever no explicit strike is
    configured, which is the common case this module targets).

    `market_vols[i]` is the market NORMAL volatility for the `i`-th
    exercise date's swaption (same length/order as `exercise_times`) --
    supplied by the caller (this module does not fetch a live market vol
    surface; `engine/calibration/lgm.py`'s own callers are expected to
    supply real market quotes).
    """
    assert len(exercise_times) == len(market_vols)
    targets = []
    for T0, vol in zip(exercise_times, market_vols):
        tenor_years = final_maturity_time - T0
        assert tenor_years > 0.0, "co-terminal basket requires every exercise time to precede the final maturity"
        forward_start_years = T0
        # Whole MONTHS, never a fractional-year string: `ORE.Period(str)`
        # only parses an integer count with a unit -- `ORE.Period("0.75Y")`
        # silently parses to `0Y` (confirmed directly: no exception, no
        # fractional-year support at all), which would silently build a
        # zero-length (or wrong-length) underlying swap for any exercise
        # date that doesn't split the trade's own maturity into whole
        # years -- a real bug caught by this module's own edge-case tests
        # (a sub-year gap to final maturity, e.g. a 3M-tenor final bucket).
        # Rounding to the nearest whole month (not year) keeps the
        # resulting swap's own tenor accurate to within half a month for
        # any exercise schedule, matching the day-count precision every
        # other period in this codebase is built to.
        tenor_months = int(round(tenor_years * 12))
        assert tenor_months > 0, (
            f"co-terminal basket requires a strictly positive whole-month tenor to final "
            f"maturity; got {tenor_years} years ({tenor_months} months) for exercise time {T0}"
        )
        swap_tenor = f"{tenor_months}M"

        # Build once at a placeholder rate to get the schedule/discount
        # factors, then re-strike at the par rate implied by zero_curve
        # (NOT ORE's own discount curve, since calibration is meant to
        # price consistently against the SAME curve engine.models.lgm
        # itself discounts with -- see price_lgm_swaption below).
        placeholder = build_vanilla_swap(
            notional=notional, fixed_rate=0.03, payer=payer,
            swap_tenor=swap_tenor, index_tenor_months=index_tenor_months,
            floating_spread=0.0, evaluation_date=evaluation_date,
            forward_start=ORE.Period(int(round(forward_start_years * 12)), ORE.Months),
        )
        today = evaluation_date
        accrual_start_date = ORE.as_fixed_rate_coupon(placeholder.fixedLeg()[0]).accrualStartDate()
        exercise_date = ORE.TARGET().advance(accrual_start_date, -2, ORE.Days)

        fixed = fixed_leg_cashflows(placeholder, today)
        T_start = TIME_AXIS_DAY_COUNTER.yearFraction(today, accrual_start_date)
        annuity = float(jnp.sum(jnp.asarray(fixed.accrual_fractions) * discount(zero_curve, jnp.asarray(fixed.payment_times))))
        P_start = float(discount(zero_curve, T_start))
        P_end = float(discount(zero_curve, fixed.payment_times[-1]))
        par_rate = (P_start - P_end) / annuity

        fixed_amounts = np.asarray(fixed.accrual_fractions) * par_rate * notional

        targets.append(CalibrationTarget(
            expiry_time=TIME_AXIS_DAY_COUNTER.yearFraction(today, exercise_date),
            accrual_start_time=T_start,
            fixed_cashflow_times=fixed.payment_times,
            fixed_cashflow_amounts=fixed_amounts,
            fixed_accrual_fractions=np.asarray(fixed.accrual_fractions),
            notional=notional,
            payer=payer,
            forward_rate=par_rate,
            market_vol=vol,
        ))
    return targets


def _bisect_xstar_raw(coupon_bond_value_fn, iterations: int = 100) -> jax.Array:
    """The bisection itself (forward value only -- see `_bisect_xstar`,
    which wraps this with a differentiable implicit-function-theorem
    correction). NOT differentiable correctly on its own: `val > 0.0`'s
    comparison has zero gradient everywhere, so a naive `jax.grad` through
    this loop silently ignores how x* itself shifts with `coupon_bond_
    value_fn`'s own parameters (sigma, a, the curve) -- captured only
    `price_lgm_swaption`'s DIRECT dependence on sigma (through `K`/
    `sigma_p`/`P0_Ti` evaluated AT a fixed x*), not the INDIRECT
    dependence through x* moving -- silently wrong by construction, not
    merely imprecise. Confirmed directly: an earlier version of this
    function was used without the correction below, and
    `engine.risk.greeks.bermudan_vega`'s own cross-check against a full
    finite-difference recalibration caught a systematic ~6% error in
    `price_lgm_swaption`'s own `jax.grad` w.r.t. sigma, traced to exactly
    this missing term (see `tests/test_calibration_basket.py`'s gradient
    correctness tests, and `tests/test_greeks_bermudan.py`'s Vega
    finite-difference check, both of which fail without `_bisect_xstar`'s
    `custom_jvp` wrapper)."""
    lo, hi = jnp.array(-2.0), jnp.array(2.0)

    def body(carry, _):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        val = coupon_bond_value_fn(mid)
        lo = jnp.where(val > 0.0, mid, lo)
        hi = jnp.where(val > 0.0, hi, mid)
        return (lo, hi), None

    (lo, hi), _ = jax.lax.scan(body, (lo, hi), None, length=iterations)
    return 0.5 * (lo + hi)


def _bisect_xstar(coupon_bond_value_fn, params, iterations: int = 100) -> jax.Array:
    """
    Differentiable wrapper around `_bisect_xstar_raw`, via the implicit
    function theorem -- the SAME correction `european_swaption._solve_
    rstar` applies (see that function's own extensive docstring for the
    full derivation; this is its single-scalar specialization, not a
    separate derivation): at a root of `f(x*, params) = 0`,
    `dx*/dparams . v = -(df/dparams . v) / (df/dx)` for any tangent
    direction `v`, computed via one `jax.grad` (for `df/dx`, at the
    stop-gradient'd converged root) and one `jax.jvp` (for the directional
    derivative `df/dparams . v`).

    `coupon_bond_value_fn(x, params) -> value`: `params` is an explicit
    pytree (here, `(a, sigma)`, whatever `price_lgm_swaption`'s caller
    wants `jax.grad` with respect to) -- required by `jax.custom_jvp`,
    which needs an explicit primal argument to attach a JVP rule to; a
    plain closure's captured tracers cannot be used directly (same
    constraint `_solve_rstar`'s own docstring explains).
    """
    @jax.custom_jvp
    def solve(p):
        f = lambda x: coupon_bond_value_fn(x, p)
        xstar = _bisect_xstar_raw(f)
        return jax.lax.stop_gradient(xstar)

    @solve.defjvp
    def solve_jvp(primals, tangents):
        p, = primals
        p_dot, = tangents
        xstar_val = solve(p)
        df_dx = jax.grad(lambda x: coupon_bond_value_fn(x, p))(xstar_val)
        _, df_dparams_dot = jax.jvp(lambda pp: coupon_bond_value_fn(xstar_val, pp), (p,), (p_dot,))
        xstar_dot = -df_dparams_dot / df_dx
        return xstar_val, xstar_dot

    return solve(params)


def price_lgm_swaption(
    curve: ZeroCurve, a: float, sigma: Union[float, Sigma], target: CalibrationTarget,
) -> jax.Array:
    """
    t=0 NPV of one co-terminal European swaption under LGM, for a trial
    `(a, sigma)` -- the model-price half of calibration's error function
    `model_price - market_price` (see `engine/calibration/lgm.py`).

    Jamshidian's decomposition in LGM's own state variable (see module
    docstring): find x* such that the signed coupon bond (every fixed
    cashflow, the final notional, minus the notional received back at the
    swap's own accrual start) is worth exactly 0 at the exercise date, then
    price each leg as a zero-coupon bond option struck at that leg's
    x*-implied forward price. Payer = sum of bond puts, receiver = sum of
    bond calls -- identical sign convention to
    `engine.instruments.european_swaption._price_one_swaption`, live-
    verified there against `ORE.JamshidianSwaptionEngine` (a DIFFERENT
    model, HW1F, but the SAME put/call/payer/receiver sign convention,
    since both are Jamshidian-style decompositions of the same trade
    economics).

    Differentiable end-to-end in `sigma` (and `a`) via `_bisect_xstar`'s
    implicit-function-theorem correction -- REQUIRED, not merely more
    precise than a naive bisection gradient: see `_bisect_xstar_raw`'s own
    docstring for the concrete ~6% error this correction fixes (caught via
    `engine.risk.greeks.bermudan_vega`'s finite-difference cross-check).
    """
    T0 = target.expiry_time
    T_start = target.accrual_start_time
    cf_times = jnp.asarray(target.fixed_cashflow_times)
    cf_amounts = jnp.asarray(target.fixed_cashflow_amounts)
    notional = target.notional

    all_times = jnp.concatenate([cf_times, cf_times[-1:], jnp.asarray([T_start])])
    all_amounts = jnp.concatenate([cf_amounts, jnp.asarray([notional, -notional])])

    P0_Ti = bond_price(curve, a, sigma, 0.0, all_times, 0.0)
    P0_T0 = bond_price(curve, a, sigma, 0.0, T0, 0.0)

    def coupon_bond_value(x, params):
        a_p, sigma_p = params
        P_T0_Ti = bond_price(curve, a_p, sigma_p, T0, all_times, x)
        return jnp.sum(P_T0_Ti * all_amounts)

    xstar = _bisect_xstar(coupon_bond_value, (a, sigma))
    K = bond_price(curve, a, sigma, T0, all_times, xstar)
    sigma_p = bond_option_sigma(a, sigma, T0, all_times, 0.0)

    bond_fn = bond_put if target.payer else bond_call
    per_leg = bond_fn(P0_T0, P0_Ti, K, sigma_p)
    return jnp.sum(per_leg * all_amounts)


def bachelier_swaption_price(target: CalibrationTarget, curve: ZeroCurve) -> jax.Array:
    """
    Market price implied by `target.market_vol` via the Bachelier (normal)
    formula on the underlying's own par annuity -- the calibration target
    `price_lgm_swaption` above is fit against. Same normal-vol convention
    `engine.risk.greeks` uses for Vega (see that module's docstring): price
    = notional * annuity * [ (F-K)*N(d) + vol*sqrt(T0)*phi(d) ], with
    `d = (F-K)/(vol*sqrt(T0))` and F == K exactly for an ATM basket (see
    `build_coterminal_basket`, which always strikes at par) -- so this
    collapses to `notional * annuity * vol * sqrt(T0/(2*pi))`, the standard
    ATM Bachelier straddle-half formula, but the general (non-ATM-safe)
    form is used here so this function stays correct if a future caller
    supplies a non-ATM basket. `annuity` here is computed from the SAME
    `zero_curve`/cashflow schedule `build_coterminal_basket` used to strike
    the basket at par, so `forward == target.forward_rate` exactly (no
    curve mismatch between the two price functions being compared during
    calibration).
    """
    from jax.scipy.stats import norm

    T0 = target.expiry_time
    cf_times = jnp.asarray(target.fixed_cashflow_times)
    cf_fractions = jnp.asarray(target.fixed_accrual_fractions)

    annuity = jnp.sum(cf_fractions * discount(curve, cf_times))
    forward = target.forward_rate
    strike = target.forward_rate  # ATM by construction (see build_coterminal_basket)

    vol = target.market_vol
    sqrt_T0 = jnp.sqrt(T0)
    d = jnp.where(vol > 0.0, (forward - strike) / (vol * sqrt_T0), 0.0)
    price = target.notional * annuity * ((forward - strike) * norm.cdf(d) + vol * sqrt_T0 * norm.pdf(d))
    return price
