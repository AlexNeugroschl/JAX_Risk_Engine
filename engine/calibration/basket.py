"""
Co-terminal European swaption basket, and the LGM closed-form swaption pricer used to fit it.

Basket: one European per exercise time, each into a swap ending on the shared final
maturity (e.g. exercises 1Y..4Y on a 5Y trade give 1Yx4Y, 2Yx3Y, 3Yx2Y, 4Yx1Y), as ORE's
`IrModelBuilder::buildSwaptionBasket` builds by default, struck ATM.

Differs from ORE: ORE derives the basket from the trade's own exercise dates. Here the
caller supplies exercise and maturity times as year fractions; each is rounded to whole
months and turned into a tenor-quoted swap on the evaluation date, and the expiry is set
two TARGET business days before that swap's first accrual date (I-47).

Pricer: `price_lgm_swaption` is Jamshidian's decomposition in the LGM state variable, the
equivalent of `QuantExt::AnalyticLgmSwaptionEngine` (whose constructor ORE's Python
bindings do not expose). Its building blocks match `ORE.LinearGaussMarkovModel` to machine
precision, and the full price is checked against a numeraire-deflated Monte Carlo of
x(T0) ~ N(0, zeta(T0)) in tests/test_calibration_basket.py. It is separate from
`engine.instruments.european_swaption`, which prices under Hull-White (see
`engine.models.lgm` for why the two differ).
"""
from dataclasses import dataclass, field
from typing import List, Union

import jax
import jax.numpy as jnp
import numpy as np
import ORE

from engine.instruments.schedules import build_vanilla_swap, fixed_leg_cashflows, resolve_swap_dates
from engine.market_data.curves import ZeroCurve, discount
from engine.market_data.day_counts import TIME_AXIS_DAY_COUNTER
from engine.models.hull_white import bond_call, bond_put
from engine.models.lgm import Sigma, bond_option_sigma, bond_price
from engine.solvers.roots import DEFAULT_SOLVER, Steps, implicit_root


@jax.tree_util.register_dataclass
@dataclass
class CalibrationTarget:
    """One co-terminal European swaption to fit: the underlying's fixed-leg cashflows
    (from a real ORE swap) and its market volatility.

    `market_vol` is a normal (Bachelier) volatility in absolute rate units (0.01 = 100bp).
    A pytree (`payer` static), so the pricers are jitted with it as an argument.
    """
    expiry_time: float                 # T0, year-fraction from evaluation_date
    accrual_start_time: float          # T_start, the underlying's own first accrual date
    fixed_cashflow_times: np.ndarray   # [N] year-fractions from evaluation_date
    fixed_cashflow_amounts: np.ndarray  # [N], at the ATM fixed rate (see build_coterminal_basket)
    fixed_accrual_fractions: np.ndarray  # [N], ORE's own accrualPeriod() per coupon
    notional: float
    payer: bool = field(metadata=dict(static=True))
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
    One ATM co-terminal `CalibrationTarget` per exercise time, each into a swap running to
    `final_maturity_time`. `market_vols[i]` is the normal vol for `exercise_times[i]`.

    The strike is the underlying's par rate on `zero_curve`, (P(T_start) - P(T_end)) /
    annuity, which is what ORE's ATM default (`IrModelBuilder::getStrike` returning
    `Null<Real>()`) gives on a single curve. Forward start and tenor are rounded to whole
    months (see module docstring).
    """
    if len(exercise_times) != len(market_vols):
        raise ValueError(
            f"exercise_times and market_vols must have the same length; got "
            f"{len(exercise_times)} and {len(market_vols)}"
        )
    targets = []
    for T0, vol in zip(exercise_times, market_vols):
        tenor_years = final_maturity_time - T0
        if tenor_years <= 0.0:
            raise ValueError("co-terminal basket requires every exercise time to precede the final maturity")
        forward_start_years = T0
        # Whole months: `ORE.Period("0.75Y")` silently parses as 0Y.
        tenor_months = int(round(tenor_years * 12))
        if tenor_months <= 0:
            raise ValueError(
                f"co-terminal basket requires a strictly positive whole-month tenor to final "
                f"maturity; got {tenor_years} years ({tenor_months} months) for exercise time {T0}"
            )
        swap_tenor = f"{tenor_months}M"

        # Build at a placeholder rate for the schedule, then strike at par on `zero_curve`,
        # the curve the LGM pricer discounts with. Market quotes are measured from today,
        # so the tenor is resolved on the evaluation date.
        effective_date, maturity_date = resolve_swap_dates(
            evaluation_date, swap_tenor, ORE.Period(int(round(forward_start_years * 12)), ORE.Months))
        placeholder = build_vanilla_swap(
            notional=notional, fixed_rate=0.03, payer=payer,
            effective_date=effective_date, maturity_date=maturity_date,
            index_tenor_months=index_tenor_months, floating_spread=0.0,
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


#: Steps of x* (`engine.solvers.roots`): Bisection's are the count before Newton (A-21),
#: Newton's measured (2026-10-07: x* from 0 reaches its rounding in 3 steps; two of margin,
#: tests/test_root_solvers.py).
X_STAR_STEPS = Steps(bisection=100, newton=5)

#: x*'s starting window, widened when it does not hold the root.
X_STAR_WINDOW = 2.0


def price_lgm_swaption(
    curve: ZeroCurve, a: float, sigma: Union[float, Sigma], target: CalibrationTarget,
    solver: str = DEFAULT_SOLVER,
) -> jax.Array:
    """
    t=0 price of one co-terminal European swaption under LGM for trial `(a, sigma)`.

    Jamshidian: find x* where the signed coupon bond (fixed coupons, final notional, minus
    the notional at accrual start) is worth 0 at expiry, then price each cashflow as a zero
    bond option struck at its price at x*. Payer = sum of puts, receiver = sum of calls.

    Differentiable in `a` and `sigma`: x* by `solver`, with its derivative by the implicit
    function theorem (`implicit_root`). The comparisons of a bisection, or Newton's
    iterations, carry no derivative of how x* moves with the parameters, a ~6% Vega error
    when the bisection was used uncorrected.
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

    xstar = implicit_root(coupon_bond_value, (a, sigma), jnp.zeros((), dtype=P0_T0.dtype), solver=solver,
                          steps=X_STAR_STEPS, increasing=False, window=X_STAR_WINDOW)
    K = bond_price(curve, a, sigma, T0, all_times, xstar)
    sigma_p = bond_option_sigma(a, sigma, T0, all_times, 0.0)

    bond_fn = bond_put if target.payer else bond_call
    per_leg = bond_fn(P0_T0, P0_Ti, K, sigma_p)
    return jnp.sum(per_leg * all_amounts)


def bachelier_swaption_price(target: CalibrationTarget, curve: ZeroCurve) -> jax.Array:
    """
    Market price of `target` from its normal vol (Bachelier on the underlying's annuity):

        notional * annuity * [(F-K)*N(d) + vol*sqrt(T0)*phi(d)],  d = (F-K)/(vol*sqrt(T0))

    The target carries no separate strike, so K = F and this is
    notional * annuity * vol * sqrt(T0 / (2*pi)).
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
