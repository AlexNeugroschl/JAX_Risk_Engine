"""
ORE's `QuantExt::NumericLgmMultiLegOptionEngine` (Grid solver) reproduced in JAX: the pricing
engine of Bermudan and American swaptions (`engine.instruments.bermudan_swaption`,
`engine.instruments.american_swaption`), as ORE's `LGMGridSwaptionEngineBuilder` builds it.

The engine's model is an LGM whose reversion, volatility and grid settings come from the
pricing configuration (`engine.pricing.config.LgmSwaptionEngineConfig`, ORE's
`LGMGridSwaptionEngineBuilder`), calibrated per trade (`engine.pricing.bermudan`); here they
are explicit arguments of `prepare_bermudan`.

At the same grid settings this returns ORE's numbers, not just their converged limit:
tests/test_ore_lgm_parity.py prices through ORE's own engine
(tests/support/ore_lgm_oracle.py) and agrees to 1e-11 relative, at any `ShiftHorizon`. Parity
is with the Grid solver; ORE's FD solver is not reproduced (F-01). It rests on matching each of:

1. The model: LGM bond price and numeraire (`engine.models.lgm`), with H shifted to vanish
   at the trade's shift horizon (`LgmBuilder`; the horizon is the builder's, see
   `engine.pricing.bermudan.shift_horizon_time`), so the state grid below is centred on the
   shifted state.
2. The solver: `LgmConvolutionSolver2`'s state grid `x_k = k*sqrt(zeta(t))/nx` with
   `floor(sx*nx)` points either side of zero, Hagan's quadrature weights (including the
   boundary formula and clamping of rounding-negative weights), and linear interpolation
   in the rollback.
3. Exercise in dates, converted with the curve's day counter as ORE derives
   `optionTimes`. Bermudan: dates after the evaluation date. American: ORE's truncated
   uniform grid (see `engine.instruments.american_swaption`).
4. Coupon membership (`buildCashflowInfo`): Bermudan exercise enters whole periods (a
   coupon belongs while `t <= accrualStart`); American enters broken periods (while
   `t <= accrualEnd`, credited `couponRatio(t)`).
5. Cashflow values: fixed amounts, and Ibor rates projected over the index fixing period
   with `LgmVectorised::fixing`'s clamps.
6. ORE's backward-loop bookkeeping (`_GridSchedule`): when each cashflow is added to the
   rolled-back underlying, cached, or credited provisionally. It depends only on times, so
   it is precomputed as masks. A closed-form exercise value converges to the same limit but
   differs by up to ~1e-4 at a 48-point grid.

**The induction as matrix products** (F-07, I-83; docs/planning/details/precision.md §8.2). In
the numeraire's units every quantity the induction rolls back is a function on the state grid,
and neither the grid nor the rollback depends on the curves: the grid `x_k = k sqrt(zeta(t))/nx`
and the Gaussian kernel between two times are the model's alone. So each step's rollback is one
`[nodes, nodes]` operator (`_rollback_operator`), shared by every path, applied to every column
(the option, the underlying, the cached coupons) of every path at once by one matrix product.
A cashflow's value on the grid is a path's coefficient times a function of the state that is
the model's alone (`ORE`'s `reducedDiscountBond`: `P(0, T) exp(-H_T^2 zeta_t / 2 - H_T x)`), an
Ibor coupon projected over its index period two such terms, so every cashflow the bookkeeping
adds on a row is one product of the paths' coefficients `[S, J]` with those functions, weighted
by the row's actions (`_Induction.targets`). The rolled-back values are then never formed per
path and step from the curves, and the memory is `[S, columns, nodes]` (the cached coupons in as
few columns as are ever cached at once, `_cache_slots`), against `[S, nodes, nodes]` per step
when the rollback interpolated each path's columns (I-83).

**The difference form.** The induction runs on the level curves (no path axis, in their own
format: float64 on the paths) and, given each path's `Deviations` from them, on each path's
deviation, everything linear (the rollback, the bookkeeping) applying to the deviations alone in
the stage's compute format, the products sharing their right operand (the operator, the
functions on the grid; A-23). At an exercise, with the level's exercise value less its
continuation value `c`, a path's deviation is `max(d_option - max(c, 0), d_exercise + min(c,
0))`, exactly the path's `max` less the level's: the level enters a path only through `c`, in
two terms of the accumulate format (`engine.precision.add_level`). So a format below float32
rounds deviations, never the option's value; at float64 the paths' values are those of the
induction run on each path's curves, to rounding. `grid_value` is the level alone: today's
price, the AD Greeks' price function and every t=0 value.

The induction is one `jax.lax.scan`, so `jax.grad`/`jax.hessian` differentiate through it. A row
with nothing to roll back (the first, ORE's no-op rollbacks, padding) skips its products
(`lax.cond`), so trades padded to one shape (`one_shape`: a path cube's dates) share one program
at no cost. Many paths are priced in blocks (`path_blocks`, `blocked`) so that the induction's
memory stays within `induction_memory_budget()` per device at any path count.

Seasoned trades are priced as ORE prices them: exercise dates on or before the evaluation
date are dropped, coupons that can no longer enter any exercise are not valued, coupons
fixed before it use `fixings`, and after the last exercise date the option is worth 0.

ORE prices Bermudans and Americans with this engine (`LGMGridSwaptionEngineBuilder`,
OREData/ored/portfolio/builders/swaption.hpp). See
docs/instruments/american-bermudan-swaptions.md for the derivation.
"""
import dataclasses
import math
from dataclasses import dataclass, fields
from typing import List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import ORE
from jax import lax
from jax.tree_util import register_pytree_node_class

from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig, ExerciseStyle, underlying_swap
from engine.instruments.schedules import known_fixing
from engine.market_data.curves import curve_dtype, discount, log_discount_deviation
from engine.market_data.day_counts import time_from_reference
from engine.models.lgm import H as _lgm_H, Sigma, horizon_shift as _lgm_horizon_shift, zeta as _lgm_zeta
from engine.precision import Arithmetic, add_level, matmul
from engine.pricing.legs import Deviations

# Every time here is on the simulation's time axis (`engine.market_data.day_counts`), never an
# instrument's accrual basis.


@register_pytree_node_class
@dataclass(frozen=True, eq=False)
class _PreparedBermudan:
    """A Bermudan/American swaption's prepared structure.

    A pytree whose static aux data is only what fixes the program's shape or control flow:
    `payer` and the grid settings `n_per_std`/`std_devs` (`_STATIC`). Every other field is a
    traced child: the model's curves (a `ZeroCurve` today, a path's `DiscountCurve` in the
    simulation; `index_curve=None` means the discount curve) and volatility, which are the
    differentiation targets (Delta/Gamma, Vega; `engine.risk.greeks.price_functions` substitutes
    tracers into them), and every time, amount and fixing of the schedule.

    So the induction (`_induction`) compiles once per trade shape (coupon and exercise
    counts, grid size) for pricing, `jax.grad` and `jax.hessian`, whatever the trade's
    dates, the valuation date or its size. The host-side rows (`induction_rows`) read the
    concrete fields before the jit boundary.
    """
    payer: bool
    notional: float
    exercise_times: np.ndarray          # [E] ORE's optionTimes, ascending
    # Per coupon, ORE's `CashflowInfo`: pay time, accrual start/end (the
    # couponRatio's couponStartTime_/couponEndTime_), and the time up to
    # which the coupon still belongs to the exercised-into swap
    # (belongsToUnderlyingMaxTime_, set by the exercise style).
    fixed_times: np.ndarray             # [Nf] fixed payment times
    fixed_start_times: np.ndarray       # [Nf]
    fixed_end_times: np.ndarray         # [Nf]
    fixed_belongs_until: np.ndarray     # [Nf]
    fixed_amounts: np.ndarray           # [Nf] notional*rate*accrual (ORE's own coupon.amount())
    float_pay_times: np.ndarray         # [Ncf]
    float_start_times: np.ndarray       # [Ncf]
    float_end_times: np.ndarray         # [Ncf]
    float_belongs_until: np.ndarray     # [Ncf]
    float_accrual: np.ndarray           # [Ncf] coupon accrualPeriod()
    # The index's own fixing period, which ORE's LgmVectorised::fixing
    # projects over -- NOT always the accrual period (I-31).
    float_index_start_times: np.ndarray  # [Ncf] valueDate(fixingDate)
    float_index_end_times: np.ndarray    # [Ncf] maturityDate(valueDate)
    float_index_dcf: np.ndarray          # [Ncf] index day count over that period
    float_fixing_times: np.ndarray       # [Ncf] max(0, fixing time): ORE's maxEstimationTime_
    float_fixed_today: np.ndarray        # [Ncf] bool: fixes today, forecast off today's curve
    float_is_known: np.ndarray           # [Ncf] bool: the fixing is known (historical)
    float_known_rates: np.ndarray        # [Ncf] that fixing where float_is_known, else 0 (traced:
                                         # a simulation supplies one per path)
    float_fixing_serials: np.ndarray     # [Ncf] the fixing dates (serial numbers)
    float_spread: float
    reversion: float
    sigma: Union[float, Sigma]    # the LGM's (Hagan) volatility
    shift_horizon: float          # years; H is shifted to vanish there (0: unshifted)
    curve: object                 # ZeroCurve or DiscountCurve (None until a caller sets it)
    index_curve: object           # None, ZeroCurve or DiscountCurve
    n_per_std: int
    std_devs: float
    final_maturity: float

    # Static aux data, in tree_flatten order (see the class docstring).
    _STATIC = ("payer", "n_per_std", "std_devs")

    def tree_flatten(self):
        children = tuple(getattr(self, f.name) for f in fields(self) if f.name not in self._STATIC)
        return children, tuple(getattr(self, name) for name in self._STATIC)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        traced = [f.name for f in fields(cls) if f.name not in cls._STATIC]
        return cls(**dict(zip(cls._STATIC, aux_data)), **dict(zip(traced, children)))


def _belongs_until(style: ExerciseStyle, accrual_start: float, accrual_end: float) -> float:
    """ORE's `belongsToUnderlyingMaxTime_` for a coupon with no notice
    period (`buildCashflowInfo`, lines 107-115)."""
    return accrual_end if style is ExerciseStyle.AMERICAN else accrual_start


def prepare_bermudan(cfg: Union[BermudanSwaptionConfig, AmericanSwaptionConfig], *, reversion: float,
                     sigma: Union[float, Sigma], shift_horizon: float, n_per_std: int, std_devs: float,
                     exercise_time_steps_per_year: int, built_on: Optional[ORE.Date] = None, curve=None,
                     index_curve=None) -> _PreparedBermudan:
    """Build the ORE underlying and resolve what `NumericLgmMultiLegOptionEngineBase`
    resolves before its backward run, for a Bermudan or American config:

      * the option times (`cfg.option_times`, an American's on `exercise_time_steps_per_year`,
        its window as built on `built_on`, by default the evaluation date);
      * per coupon, ORE's `CashflowInfo`: pay time, accrual start/end, the time it stops
        belonging to the exercised-into swap (by `cfg.exercise_style`), and for a floating
        coupon the index fixing period and its day count fraction.

    The model is the engine's LGM: constant `reversion` and Hagan volatility `sigma`, H shifted
    to vanish at `shift_horizon` years (ORE's `LgmData::shiftHorizon`; 0 for none), on the
    grid `n_per_std`/`std_devs` (ORE's `nx`/`sx`). `curve` discounts and is the LGM's term
    structure; `index_curve` (default: `curve`) projects the coupons. Either may be left
    unset and replaced later (`dataclasses.replace`), as a simulation does per path.

    A coupon whose belongs-until time is before the evaluation date is dropped (ORE's
    `isPartOfUnderlying(t)` is false for every `t >= 0`). A kept floating coupon whose
    fixing date has passed takes its fixing from `cfg.fixings`; a missing one raises.
    Fixed amounts are ORE's `FixedRateCoupon.amount()`."""
    if not math.isfinite(shift_horizon) or shift_horizon < 0.0:
        raise ValueError(f"shift_horizon must be finite and >= 0 years (ORE's LgmBuilder); got {shift_horizon!r}")
    swap = underlying_swap(cfg)
    today = cfg.evaluation_date
    style = cfg.exercise_style
    t_of = lambda d: time_from_reference(today, d)  # noqa: E731

    fixed = {key: [] for key in ("pay", "start", "end", "belongs", "amount")}
    for cf in swap.fixedLeg():
        c = ORE.as_fixed_rate_coupon(cf)
        start, end = t_of(c.accrualStartDate()), t_of(c.accrualEndDate())
        belongs = _belongs_until(style, start, end)
        if belongs < 0.0:
            continue
        fixed["pay"].append(t_of(c.date()))
        fixed["start"].append(start)
        fixed["end"].append(end)
        fixed["belongs"].append(belongs)
        fixed["amount"].append(c.amount())

    floating = {key: [] for key in (
        "pay", "start", "end", "belongs", "accrual", "idx_start", "idx_end", "idx_dcf", "fixing",
        "fixed_today", "known", "known_rate", "fixing_serial")}
    for cf in swap.floatingLeg():
        c = ORE.as_floating_rate_coupon(cf)
        start, end = t_of(c.accrualStartDate()), t_of(c.accrualEndDate())
        belongs = _belongs_until(style, start, end)
        if belongs < 0.0:
            continue
        index = c.index()
        fixing_date = c.fixingDate()
        known = known_fixing(fixing_date, today, cfg.fixings)
        index_start = index.valueDate(fixing_date)
        index_end = index.maturityDate(index_start)
        floating["pay"].append(t_of(c.date()))
        floating["start"].append(start)
        floating["end"].append(end)
        floating["belongs"].append(belongs)
        floating["accrual"].append(c.accrualPeriod())
        floating["idx_start"].append(t_of(index_start))
        floating["idx_end"].append(t_of(index_end))
        floating["idx_dcf"].append(index.dayCounter().yearFraction(index_start, index_end))
        floating["fixing"].append(max(0.0, t_of(fixing_date)))
        floating["fixed_today"].append(fixing_date == today and known is None)
        floating["known"].append(known is not None)
        floating["known_rate"].append(0.0 if known is None else known)
        floating["fixing_serial"].append(fixing_date.serialNumber())

    exercise_times = np.asarray(cfg.option_times(exercise_time_steps_per_year, built_on), dtype=np.float64)
    if exercise_times.size == 0:
        raise ValueError(f"no exercise opportunity falls after the evaluation date {today}")
    final_maturity = max(fixed["pay"][-1], floating["pay"][-1])
    if np.any(exercise_times >= final_maturity):
        raise ValueError(
            f"exercise must fall strictly before the underlying swap's final maturity "
            f"(t={final_maturity}); got option times up to t={float(exercise_times[-1])}"
        )

    as_array = lambda values, dtype=np.float64: np.asarray(values, dtype=dtype)  # noqa: E731
    return _PreparedBermudan(
        payer=cfg.payer,
        notional=swap.fixedNominals()[0] if swap.fixedNominals() else swap.nominal(),
        exercise_times=exercise_times,
        fixed_times=as_array(fixed["pay"]),
        fixed_start_times=as_array(fixed["start"]),
        fixed_end_times=as_array(fixed["end"]),
        fixed_belongs_until=as_array(fixed["belongs"]),
        fixed_amounts=as_array(fixed["amount"]),
        float_pay_times=as_array(floating["pay"]),
        float_start_times=as_array(floating["start"]),
        float_end_times=as_array(floating["end"]),
        float_belongs_until=as_array(floating["belongs"]),
        float_accrual=as_array(floating["accrual"]),
        float_index_start_times=as_array(floating["idx_start"]),
        float_index_end_times=as_array(floating["idx_end"]),
        float_index_dcf=as_array(floating["idx_dcf"]),
        float_fixing_times=as_array(floating["fixing"]),
        float_fixed_today=as_array(floating["fixed_today"], dtype=bool),
        float_is_known=as_array(floating["known"], dtype=bool),
        float_known_rates=as_array(floating["known_rate"]),
        float_fixing_serials=as_array(floating["fixing_serial"], dtype=np.int64),
        float_spread=cfg.floating_spread,
        reversion=reversion, sigma=sigma, shift_horizon=float(shift_horizon), curve=curve, index_curve=index_curve,
        n_per_std=n_per_std, std_devs=std_devs,
        final_maturity=final_maturity,
    )


def _zero_curve_of(swap: _PreparedBermudan):
    """The curve the induction discounts on (and the LGM's term structure). No dtype is
    forced: it is float64 for pricing, and a risk-precision JAX array when
    `engine.risk.greeks.ad` substitutes one."""
    return swap.curve


def _index_curve_of(swap: _PreparedBermudan):
    """The Ibor index's forwarding curve; the discount curve unless one was given."""
    return swap.curve if swap.index_curve is None else swap.index_curve


# State grid and Hagan quadrature convolution, as QuantExt::LgmConvolutionSolver2
# (QuantExt/qle/models/lgmconvolutionsolver2.cpp), in the LGM state variable x(t).
def _safe_sqrt(z: jax.Array) -> jax.Array:
    """sqrt(max(z, 0)) with a finite gradient at 0 (a plain `jnp.sqrt` gives NaN Vega at t=0:
    `jax.grad` differentiates both branches of a `where`, so the root is taken of a safe
    placeholder)."""
    z = jnp.maximum(z, 0.0)
    return jnp.where(z > 0.0, jnp.sqrt(jnp.where(z > 0.0, z, 1.0)), 0.0)


def _grid_spacing(sigma, t: jax.Array, n_per_std: int) -> jax.Array:
    """The state grid's spacing at time t, `sqrt(zeta(t))/n_per_std` (0 at t=0)."""
    return _safe_sqrt(_lgm_zeta(sigma, t)) / n_per_std


def _state_grid(sigma: float, t: jax.Array, n_per_std: int, std_devs: float, dtype=jnp.float64) -> jax.Array:
    """
    LGM state grid at time t: `x_k = k*sqrt(zeta(t))/n_per_std` for
    `k = -mx..mx`, `mx = grid_half_width(std_devs, n_per_std)`, as
    `LgmConvolutionSolver2::stateGrid`. The shape is fixed (`2*mx+1`) for `lax.scan`; at
    t=0 every point is 0, as in ORE.

    `dtype` is passed explicitly: the caller uses `curve.pillar_rates.dtype`, which is the
    pricing or risk precision as appropriate, whereas `sigma` and `t` are float64 either
    way.
    """
    mx = grid_half_width(std_devs, n_per_std)
    return _grid_spacing(sigma, t, n_per_std).astype(dtype) * jnp.arange(-mx, mx + 1, dtype=dtype)


def grid_half_width(std_devs: float, n_per_std: int) -> int:
    """Points on each side of zero: ORE's `mx_`/`my_ = static_cast<int>(
    floor(s * n) + 0.5)` (LgmConvolutionSolver2's constructor) -- a floor,
    not a rounding, of `std_devs * n_per_std`."""
    return int(math.floor(std_devs * n_per_std) + 0.5)


def _hagan_quadrature_weights(n_per_std: int, std_devs: float) -> np.ndarray:
    """
    Hagan's quadrature weights on the standardized nodes `y_i = h*(i - my)`,
    `h = 1/n_per_std`, as `LgmConvolutionSolver2`'s constructor:

        interior:  w_i = (1 + y_i/h)*N(y_i+h) - 2*(y_i/h)*N(y_i) - (1 - y_i/h)*N(y_i-h)
                         + (G(y_i+h) - 2*G(y_i) + G(y_i-h)) / h
        i = 0 and i = 2*my (both, with y_0):
                   w_i = (1 + y_0/h)*N(y_0+h) - (y_0/h)*N(y_0) + (G(y_0+h) - G(y_0)) / h

    N/G are the standard normal CDF/PDF. A weight negative through rounding is set to 0;
    below -1e-10 raises, as ORE does.
    """
    from scipy.stats import norm as scipy_norm

    h = 1.0 / n_per_std
    my = grid_half_width(std_devs, n_per_std)
    N, G = scipy_norm.cdf, scipy_norm.pdf
    y = np.asarray([h * (i - my) for i in range(2 * my + 1)], dtype=np.float64)
    w = np.empty_like(y)
    y0 = y[0]
    boundary = (1.0 + y0 / h) * N(y0 + h) - y0 / h * N(y0) + (G(y0 + h) - G(y0)) / h
    for i, yi in enumerate(y):
        if i == 0 or i == 2 * my:
            w[i] = boundary
        else:
            w[i] = ((1.0 + yi / h) * N(yi + h) - 2.0 * yi / h * N(yi) - (1.0 - yi / h) * N(yi - h)
                    + (G(yi + h) - 2.0 * G(yi) + G(yi - h)) / h)
        if w[i] < 0.0:
            if w[i] <= -1.0e-10:
                raise ValueError(f"negative convolution weight {w[i]} at i={i}")
            w[i] = 0.0
    return w


def _quadrature_nodes(n_per_std: int, std_devs: float) -> np.ndarray:
    """The standardized nodes `y_i = h*(i - my)` the weights above belong to."""
    my = grid_half_width(std_devs, n_per_std)
    return np.asarray([(1.0 / n_per_std) * (i - my) for i in range(2 * my + 1)], dtype=np.float64)


def _rollback_operator(dx_from: jax.Array, x_to: jax.Array, std: jax.Array, quad_y: jax.Array,
                       quad_w: jax.Array) -> jax.Array:
    """
    `R` `[n_to, n_from]` with `R @ f` = E[f(x_from) | x_to] at every point of `x_to`, as
    `LgmConvolutionSolver2::rollback`, for any column of values `f` on the grid
    `x_from = dx_from * (k - mx)`, `k = 0..2 mx`.

    x is driftless, so x_from given x_to is Gaussian with mean x_to and standard deviation
    `std = sqrt(zeta(t_from) - zeta(t_to))`. The expectation is `sum_i w_i f(x_to + y_i std)`,
    f linearly interpolated on `x_from` and flat outside it, which is linear in f: each query
    point puts `w_i (1 - u)` on the node below it and `w_i u` on the node above, `u` its fraction
    of the spacing. Where `std` is 0 (no variance between the two times) it is ORE's
    interpolation alone: each point of `x_to` with weight 1 (the quadrature's middle node, y=0).

    The operator depends on the model and the two times only, never on a path's curves: one
    per step serves every path and column (see the module docstring). Built by a scatter of its
    `2 n_to n_quad` weights (each row's query points are spread over the whole grid, so it has
    no band structure to exploit).
    """
    n_from = quad_y.shape[0]
    mx = (n_from - 1) // 2
    middle = jnp.arange(n_from) == mx
    weights = jnp.where(std > 0.0, quad_w, middle.astype(quad_w.dtype))                       # [n_quad]
    query = x_to[:, None] + quad_y[None, :] * std                                           # [n_to, n_quad]
    spacing = jnp.where(dx_from > 0.0, dx_from, 1.0)
    position = jnp.clip(query / spacing + mx, 0.0, n_from - 1.0)
    floor = jnp.clip(jnp.floor(position), 0.0, n_from - 2.0)
    above_share = position - floor
    below = floor.astype(jnp.int32)
    rows = jnp.broadcast_to(jnp.arange(x_to.shape[0])[:, None], query.shape)
    operator = jnp.zeros((x_to.shape[0], n_from), dtype=quad_w.dtype)
    operator = operator.at[rows, below].add(weights * (1.0 - above_share))
    return operator.at[rows, below + 1].add(weights * above_share)


def _close_enough(x: float, y: float) -> bool:
    """QuantLib's `close_enough(x, y)` (ql/math/comparison.hpp, n = 42) --
    the comparison every time test in ORE's backward loop uses."""
    if x == y:
        return True
    diff, tolerance = abs(x - y), 42.0 * np.finfo(np.float64).eps
    if x == 0.0 or y == 0.0:
        return diff < tolerance * tolerance
    return diff <= tolerance * abs(x) or diff <= tolerance * abs(y)


# Grid-time schedule and ORE's cashflow bookkeeping (precomputed per trade)
@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class _GridSchedule:
    """The descending grid times of the backward induction and, per grid time and
    cashflow, what ORE's backward loop does with that cashflow there.

    Times are `{0} u optionTimes`, deduplicated exactly (ORE's `std::set<Real> timeGrid`).
    Never rounded, so an option time stays identical to the coupon date it names.

    Actions replay `NumericLgmMultiLegOptionEngineBase::calculate()`. For cashflow `i` at
    row `g`, in numeraire-deflated units as in ORE:

      add_pv          underlyingNpv += pv                       (-> Done)
      cache_to_under  underlyingNpv += cache; cache cleared      (-> Done)
      start_cache     cache = pv                                 (-> Cached)
      from_cache      provisionalNpv += cache * couponRatio
      non_cached      provisionalNpvNonCached += pv * couponRatio

    The exercise value at an option time is
    `underlyingNpv + provisionalNpv + provisionalNpvNonCached`. ORE's `mustBeEstimated`
    branch applies only to capped/floored coupons, which a vanilla swap does not have.

    A pytree of arrays: the induction reads it traced, so schedules of one shape share a
    program.
    """
    times: np.ndarray            # [G] descending
    is_exercise: np.ndarray      # [G] bool
    rollback_is_identity: np.ndarray  # [G] bool: close_enough(t_prev, t), ORE's no-op rollback
    add_pv: np.ndarray           # [G, C]
    cache_to_under: np.ndarray   # [G, C]
    start_cache: np.ndarray      # [G, C]
    from_cache: np.ndarray       # [G, C] (includes the row a cache starts on)
    non_cached: np.ndarray       # [G, C]
    coupon_ratio: np.ndarray     # [G, C]


def _cashflow_timing(swap: _PreparedBermudan):
    """Per cashflow, fixed leg first and then floating, the four times ORE's
    `CashflowInfo` tests against: belongs-until, accrual start and end (for
    `couponRatio`), and the latest time the amount can be estimated
    (`maxEstimationTime_`: the pay time for a fixed coupon, the fixing time
    for an Ibor coupon)."""
    belongs = np.concatenate([swap.fixed_belongs_until, swap.float_belongs_until])
    start = np.concatenate([swap.fixed_start_times, swap.float_start_times])
    end = np.concatenate([swap.fixed_end_times, swap.float_end_times])
    max_estimation = np.concatenate([swap.fixed_times, swap.float_fixing_times])
    return belongs, start, end, max_estimation


def _build_grid_schedule(swap: _PreparedBermudan) -> _GridSchedule:
    exercise_set = set(float(t) for t in swap.exercise_times)
    times = sorted({0.0} | exercise_set, reverse=True)

    belongs, start, end, max_estimation = _cashflow_timing(swap)
    num_rows, num_cashflows = len(times), len(belongs)
    masks = {name: np.zeros((num_rows, num_cashflows), dtype=bool)
             for name in ("add_pv", "cache_to_under", "start_cache", "from_cache", "non_cached")}
    coupon_ratio = np.zeros((num_rows, num_cashflows), dtype=np.float64)
    OPEN, CACHED, DONE = 0, 1, 2
    status = [OPEN] * num_cashflows

    for g, t in enumerate(times):
        for i in range(num_cashflows):
            if status[i] == DONE:
                continue
            # CashflowInfo::isPartOfUnderlying
            if not (t < belongs[i] or _close_enough(t, belongs[i])):
                continue
            # CashflowInfo::couponRatio (no notice period, so no lag)
            ratio = max(0.0, min(1.0, (end[i] - t) / (end[i] - start[i])))
            coupon_ratio[g, i] = ratio
            broken = not _close_enough(ratio, 1.0)
            if status[i] == CACHED:
                if broken:
                    masks["from_cache"][g, i] = True
                else:
                    masks["cache_to_under"][g, i] = True
                    status[i] = DONE
            elif t < max_estimation[i] or _close_enough(t, max_estimation[i]):  # canBeEstimated
                if broken:
                    masks["start_cache"][g, i] = True
                    masks["from_cache"][g, i] = True
                    status[i] = CACHED
                else:
                    masks["add_pv"][g, i] = True
                    status[i] = DONE
            else:
                masks["non_cached"][g, i] = True

    rollback_is_identity = np.asarray(
        [False] + [_close_enough(times[g - 1], times[g]) for g in range(1, num_rows)], dtype=bool)
    return _GridSchedule(
        times=np.asarray(times, dtype=np.float64),
        is_exercise=np.asarray([t in exercise_set for t in times], dtype=bool),
        rollback_is_identity=rollback_is_identity,
        coupon_ratio=coupon_ratio,
        **masks,
    )


# =============================================================================
# THE INDUCTION'S ROWS: ORE's bookkeeping in the kernel's form (host)
# =============================================================================
def _cache_slots(schedule: _GridSchedule) -> Tuple[np.ndarray, int]:
    """The column of the induction that holds each cashflow's cache (-1 for one never cached),
    and how many there are: a coupon is cached from the row it starts on (`start_cache`) up to
    and including the row it moves into the underlying (`cache_to_under`), or to the end, and a
    column is reused once its coupon has moved, so there are as many columns as coupons are ever
    cached at once (a Bermudan none, since it enters whole periods; an American one or two per
    leg), not one per cashflow."""
    num_rows, num_cashflows = schedule.start_cache.shape
    slot_of = np.full(num_cashflows, -1, dtype=np.int64)
    busy_until: List[int] = []   # per column, the last row its coupon holds it
    starts = {c: int(np.argmax(column)) for c, column in enumerate(schedule.start_cache.T) if column.any()}
    for c, start in sorted(starts.items(), key=lambda item: item[1]):
        moved = np.nonzero(schedule.cache_to_under[:, c])[0]
        end = int(moved[0]) if moved.size else num_rows - 1
        free = [k for k, until in enumerate(busy_until) if until < start]
        if free:
            busy_until[free[0]] = end
        else:
            busy_until.append(end)
        slot_of[c] = free[0] if free else len(busy_until) - 1
    return slot_of, len(busy_until)


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class _Induction:
    """The backward loop's rows in the kernel's form (`induction_rows`), latest first: what each
    row does with the induction's columns `[option, underlying, cache_0, ..., cache_{K-1}]` and
    with the cashflows' values, built from `_GridSchedule` on the host. A pytree of arrays: the
    kernel reads it traced, so inductions of one shape share a program.

    Per row, the cashflows' values enter through `targets` `[G, 2 + K, J]`, the weight of each
    basis function of the cashflows (`J`: a function per fixed coupon, two per floating one,
    `_induction`) in each of the row's targets: 0, the provisional value of the coupons not
    cached (`non_cached * couponRatio`); 1, what is added to the underlying (`add_pv`); 2 + k, the
    value cache column k starts with (`start_cache`). Cache column k then moves into the
    underlying where `moves` (`cache_to_under`) and is credited to the exercise value with
    weight `shares` (`from_cache * couponRatio`). `rollback`: whether the row rolls the columns
    back from the one before (not the first, nor ORE's no-op rollbacks, nor padding)."""
    times: np.ndarray        # [G] descending; padding repeats the first time
    rollback: np.ndarray     # [G] bool
    is_exercise: np.ndarray  # [G] bool
    targets: np.ndarray      # [G, 2 + K, J]
    starts: np.ndarray       # [G, K] bool
    moves: np.ndarray        # [G, K] bool
    shares: np.ndarray       # [G, K]

    @property
    def shape(self) -> Tuple[int, int]:
        """(rows, cache columns)."""
        return self.starts.shape


def induction_rows(swap: _PreparedBermudan) -> _Induction:
    """The induction's rows for the prepared trade (see `_Induction`), from ORE's bookkeeping
    (`_build_grid_schedule`)."""
    schedule = _build_grid_schedule(swap)
    slot_of, num_slots = _cache_slots(schedule)
    num_fixed, num_float = len(swap.fixed_times), len(swap.float_pay_times)
    # Each basis function's cashflow: the fixed coupons, then each floating coupon twice.
    cashflow = np.concatenate([np.arange(num_fixed + num_float), num_fixed + np.arange(num_float)])
    in_slot = slot_of[None, :] == np.arange(num_slots)[:, None]                                # [K, C]
    starts = schedule.start_cache[:, None, :] & in_slot[None]                                  # [G, K, C]
    per_cashflow = np.concatenate([(schedule.non_cached * schedule.coupon_ratio)[:, None, :],
                                   schedule.add_pv[:, None, :].astype(np.float64), starts.astype(np.float64)],
                                  axis=1)                                                       # [G, 2 + K, C]
    rollback = ~schedule.rollback_is_identity
    rollback[0] = False
    return _Induction(
        times=schedule.times, rollback=rollback, is_exercise=schedule.is_exercise,
        targets=per_cashflow[:, :, cashflow], starts=starts.any(axis=-1),
        moves=(schedule.cache_to_under[:, None, :] & in_slot[None]).any(axis=-1),
        shares=((schedule.from_cache * schedule.coupon_ratio)[:, None, :] * in_slot[None]).sum(axis=-1))


def _padded_rows(rows: _Induction, num_rows: int, num_slots: int) -> _Induction:
    """`rows` with rows that do nothing put before its first (its time, no rollback, no
    exercise, no cashflow) and cache columns that are never used, to `num_rows` and
    `num_slots`."""
    extra_rows, extra_slots = num_rows - rows.shape[0], num_slots - rows.shape[1]
    if extra_rows < 0 or extra_slots < 0:
        raise ValueError(f"cannot pad an induction of shape {rows.shape} to ({num_rows}, {num_slots})")

    def pad(a, slot_axis=None):
        widths = [(extra_rows, 0)] + [(0, 0)] * (a.ndim - 1)
        if slot_axis is not None:
            widths[slot_axis] = (0, extra_slots)
        return np.pad(a, widths)

    return _Induction(times=np.concatenate([np.full(extra_rows, rows.times[0]), rows.times]),
                      rollback=pad(rows.rollback), is_exercise=pad(rows.is_exercise),
                      targets=pad(rows.targets, slot_axis=1), starts=pad(rows.starts, 1), moves=pad(rows.moves, 1),
                      shares=pad(rows.shares, 1))


#: `_PreparedBermudan`'s per-coupon fields, and what a coupon added to pad a trade holds: it
#: never belongs to the underlying (no row acts on it) and is worth 0 (no amount or accrual).
_FIXED_PADDING = {"fixed_times": 0.0, "fixed_start_times": 0.0, "fixed_end_times": 0.0,
                  "fixed_belongs_until": -1.0, "fixed_amounts": 0.0}
_FLOAT_PADDING = {"float_pay_times": 0.0, "float_start_times": 0.0, "float_end_times": 0.0,
                  "float_belongs_until": -1.0, "float_accrual": 0.0, "float_index_start_times": 0.0,
                  "float_index_end_times": 0.0, "float_index_dcf": 1.0, "float_fixing_times": 0.0,
                  "float_fixed_today": False, "float_is_known": False, "float_known_rates": 0.0,
                  "float_fixing_serials": 0}


def one_shape(prepared: Sequence[_PreparedBermudan]) -> List[Tuple[_PreparedBermudan, _Induction]]:
    """Each prepared trade and its induction's rows, padded to the largest counts among them of
    exercise times, fixed and floating coupons, rows and cache columns (with coupons, rows and
    columns that do nothing), so that one compiled program prices them all: a path cube's
    dates, whose exercises and coupons run out one by one. The values are those of each trade
    unpadded."""
    num_fixed = max(len(p.fixed_times) for p in prepared)
    num_float = max(len(p.float_pay_times) for p in prepared)
    num_exercises = max(len(p.exercise_times) for p in prepared)

    def pad(swap, width, padding):
        return {name: np.concatenate([np.asarray(getattr(swap, name)),
                                      np.full(width - len(getattr(swap, name)), value,
                                              dtype=np.asarray(getattr(swap, name)).dtype)])
                for name, value in padding.items()}

    # An exercise time repeated is the same opportunity (ORE's grid is a set of times).
    padded = [dataclasses.replace(p, **pad(p, num_fixed, _FIXED_PADDING), **pad(p, num_float, _FLOAT_PADDING),
                                  **pad(p, num_exercises, {"exercise_times": p.exercise_times[-1]}))
              for p in prepared]
    rows = [induction_rows(p) for p in padded]
    num_rows, num_slots = (max(r.shape[i] for r in rows) for i in (0, 1))
    return [(p, _padded_rows(r, num_rows, num_slots)) for p, r in zip(padded, rows)]


#: How many arrays the size of a path's columns on the grid (`[columns, nodes]`) the induction
#: holds at once per path at its peak: the columns, their product with the operator (twice as
#: wide with two terms of a shared operand below float32, A-23), the cashflows' values on the
#: row's targets and the rolled and exercised copies. XLA's memory analysis of the compiled
#: induction measured 3.1 (a Bermudan) to 5.3 (an American in FP8) on a CPU (2026-10-13); a
#: test holds the bound.
_WORKING_COPIES = 6


#: CPU fallback for the working memory one induction over many paths or scenarios is given: a path cube's
#: date is priced in blocks of paths (`path_blocks`), market risk in batches of scenarios
#: (`engine.risk.market.revaluation.scenario_batch_size`), each within it. An American on the
#: demo's grid holds about 27 KB per path at float64 (36 KB as the GPU compiles it), 9.5 GB at
#: 262,144 paths, more than an 8 GB card holds (2026-10-13).
INDUCTION_MEMORY_BUDGET = 2 ** 30


def induction_memory_budget() -> int:
    """A stable per-device induction budget: an eighth of the smallest accelerator limit.

    Leave room for the scenario market, stored columns and XLA workspaces. An eighth
    keeps the RTX 5060's measured throughput without the extra 1.56 GB a half used
    (precision design §17); the hardware campaign measures the tradeoff on other devices.
    CPU devices and backends without a memory limit use `INDUCTION_MEMORY_BUDGET`.
    Use capacity, never current free memory: identical jobs keep the same block shapes
    and reuse their programs even when other allocations change (I-105).
    """
    budgets = []
    for device in jax.local_devices():
        stats = device.memory_stats() if device.platform != "cpu" else None
        limit = (stats or {}).get("bytes_limit", 0)
        budgets.append(max(1, int(limit) // 8) if limit and limit > 0 else INDUCTION_MEMORY_BUDGET)
    return min(budgets, default=INDUCTION_MEMORY_BUDGET)


def working_bytes_per_path(swap: _PreparedBermudan, itemsize: int) -> int:
    """About how many bytes the induction holds per path, its values `itemsize` bytes each:
    `_WORKING_COPIES` arrays of the path's columns on the grid, and the cashflows'
    coefficients. What a block of paths or a batch of scenarios is sized by
    (`induction_memory_budget`)."""
    rows = induction_rows(swap)
    nodes = 2 * grid_half_width(swap.std_devs, swap.n_per_std) + 1
    _, columns, basis = rows.targets.shape
    return itemsize * (_WORKING_COPIES * columns * nodes + 2 * basis)


def working_bytes_per_model(swap: _PreparedBermudan, itemsize: int) -> int:
    """About how many bytes one induction's level holds, its values `itemsize` bytes each: the
    row's `[nodes, nodes]` rollback operator, its cashflows' functions on the grid and
    `_WORKING_COPIES` arrays of its columns. What each scenario of a batch adds when it has a
    model of its own, an option recalibrated under each market-risk scenario
    (`engine.risk.market.revaluation.scenario_batch_size`), on top of its path's
    `working_bytes_per_path`."""
    rows = induction_rows(swap)
    nodes = 2 * grid_half_width(swap.std_devs, swap.n_per_std) + 1
    _, columns, basis = rows.targets.shape
    return itemsize * nodes * (nodes + basis + _WORKING_COPIES * columns)


def path_blocks(paths: int, bytes_per_path: int, devices: int = 1) -> int:
    """The fewest blocks of equal size that `paths` paths (split evenly across `devices`) are
    priced in so that each device's induction stays within `induction_memory_budget()`: a divisor of
    each device's share of the paths, so every block holds as many paths on every device (1 when
    they fit at once)."""
    share = paths // devices
    budget = induction_memory_budget()
    for blocks in range(1, share + 1):
        if share % blocks == 0 and (share // blocks) * bytes_per_path <= budget:
            return blocks
    return max(share, 1)


def blocked(price, deviations: Deviations, blocks: int):
    """`price(deviations)` -> `(level, deviation [S])` over `blocks` blocks of paths, one after
    another inside the program (`lax.map`), each block's paths strided through all of them
    (block b holds paths b, b + blocks, ...): under the scenario axis's split across devices each
    block takes its share from every device, so every device works on every block. The values are
    those of one call on every path. `blocks` is static (a program per count)."""
    if blocks == 1:
        return price(deviations)
    by_block = jax.tree_util.tree_map(lambda a: jnp.moveaxis(a.reshape((-1, blocks) + a.shape[1:]), 1, 0),
                                      deviations)
    levels, moved = lax.map(price, by_block)                                          # [blocks], [blocks, S/blocks]
    return levels[0], jnp.moveaxis(moved, 0, 1).reshape(-1)


# =============================================================================
# BACKWARD INDUCTION (jax.lax.scan), level and deviations
# =============================================================================
def grid_value(swap: _PreparedBermudan, rows: Optional[_Induction] = None) -> jax.Array:
    """ORE's NPV on the prepared trade's curves (`NumericLgmMultiLegOptionEngineBase::
    calculate()`): the induction's value at t=0 on the state x=0, as
    `LgmConvolutionSolver2::stateGrid(0)`. A JAX scalar, so `jax.grad` differentiates it.

    `rows` is `induction_rows(swap)`, built here when omitted; a caller inside a jit builds it
    beforehand, since it reads the trade's times on the host."""
    level, _ = option_values(swap, rows)
    return level


def option_values(swap: _PreparedBermudan, rows: Optional[_Induction] = None,
                  deviations: Optional[Deviations] = None,
                  arithmetic: Optional[Arithmetic] = None) -> Tuple[jax.Array, Optional[jax.Array]]:
    """`(level, deviation)`: the option's t=0 value on the prepared trade's curves (the level,
    `grid_value`) and, given each path's `Deviations` from them (the discount and index curves'
    node values `[S, K]`, each floating coupon's known rate `[S, Ncf]`), each path's deviation
    from it `[S]` (None without them), whose products run in `arithmetic` (the deviations' own
    format, float64 or float32, by default) and whose value is in its accumulate format; a
    path's value is `engine.precision.add_level(level, deviation)` (see the module docstring).
    `rows` as `grid_value`'s."""
    rows = induction_rows(swap) if rows is None else rows
    if deviations is not None:
        arithmetic = arithmetic or Arithmetic.of(deviations.discount.dtype)
    return _induction(swap, rows, deviations, arithmetic)


@jax.jit
def _induction(swap: _PreparedBermudan, rows: _Induction, deviations: Optional[Deviations],
               arithmetic: Optional[Arithmetic]):
    """ORE's backward loop on the level and, given `deviations`, on each path's deviation
    (`option_values`). Per row, latest first, as `calculate()` orders it:

      1. roll the columns (the option, the underlying, the cache columns) back from the previous
         row's grid by the row's operator (`_rollback_operator`), one product;
      2. add the cashflows' values to the row's targets (`_Induction`), one product of their
         coefficients with their functions on the grid;
      3. at an option time, `option = max(option, underlying + provisional)`, on the paths in
         difference form.

    Every value is deflated by the numeraire, as in ORE (`LgmVectorised::reducedDiscountBond`),
    so a cashflow's value at a node is its coefficient (the curve's discount factor at its
    payment, for an Ibor coupon times its forecast's ratio, `_Coefficients`) times
    `exp(-q zeta(t)/2 - h x)` (`_basis`); x is driftless, so the rollback of a deflated value is
    its conditional expectation. At t=0 and x=0 the numeraire is 1: the value is the option's
    column at the middle node of the last row. Jitted with `swap`, `rows` and `deviations` as
    pytrees and arithmetic formats static (keys traced): one program per trade shape and arithmetic."""
    a, sigma, n_per_std = swap.reversion, swap.sigma, swap.n_per_std
    curve, index = _zero_curve_of(swap), _index_curve_of(swap)
    # The level works in the curve's dtype (pricing or risk precision): float64 constants
    # would promote a float32 Greeks trace.
    dtype = curve_dtype(curve)
    level = _Coefficients.of(swap, curve, index, dtype)
    shift = _lgm_horizon_shift(a, swap.shift_horizon).astype(dtype)
    mx = grid_half_width(swap.std_devs, n_per_std)
    middle = mx
    nodes = jnp.arange(-mx, mx + 1, dtype=dtype)
    quad_w = jnp.asarray(_hagan_quadrature_weights(n_per_std, swap.std_devs), dtype=dtype)
    quad_y = jnp.asarray(_quadrature_nodes(n_per_std, swap.std_devs), dtype=dtype)
    num_columns = rows.targets.shape[1]
    as_level = lambda x: jnp.asarray(x, dtype=dtype)  # noqa: E731
    on_paths = deviations is not None
    moved = _Moved.of(swap, curve, index, level, deviations, arithmetic.dtype) if on_paths else None

    def step(carry, row):
        dx_prev, t_prev, values, paths = carry
        t, roll, is_exercise, targets, starts, moves, shares, row_id = row
        row_arithmetic = arithmetic.fold_in(row_id) if on_paths else None
        zeta_t = _lgm_zeta(sigma, t)
        dx = _grid_spacing(sigma, t, n_per_std).astype(dtype)
        x = dx * nodes
        std = _safe_sqrt(_lgm_zeta(sigma, t_prev) - zeta_t).astype(dtype)

        def rolled(columns):
            operator = _rollback_operator(dx_prev, x, std, quad_y, quad_w).T                     # [from, to]
            level_columns, path_columns = columns
            if path_columns is not None:
                path_columns = row_arithmetic.matmul(path_columns, operator, shared="right", stream=0)
            return matmul(level_columns, operator), path_columns

        values, paths = lax.cond(roll, rolled, lambda columns: columns, (values, paths))
        start, end, basis = _basis(swap, t, x, zeta_t.astype(dtype), shift)
        weighted = jnp.swapaxes(targets[:, :, None] * basis[None], 0, 1).reshape(basis.shape[0], -1)  # [J, T*N]
        ratio = discount(index, start) / discount(index, end)
        added = matmul(level.at_row(ratio), weighted).reshape(num_columns, -1)                   # [T, N]
        values, exercise_value = _bookkeeping(values, added, starts, moves, shares)
        continuation = values[0]
        values = values.at[0].set(jnp.where(is_exercise, jnp.maximum(continuation, exercise_value), continuation))
        if on_paths:
            operand = moved.at_row(level, ratio, start, end)                                     # [S, J]
            added = row_arithmetic.matmul(operand, weighted, shared="right", stream=1).reshape(operand.shape[0], num_columns, -1)
            paths, path_exercise = _bookkeeping(paths, added, starts, moves, shares)
            # Each path's max less the level's (see the module docstring).
            gain = exercise_value - continuation
            exercised = jnp.maximum(add_level(-jnp.maximum(gain, 0.0), paths[:, 0]),
                                    add_level(jnp.minimum(gain, 0.0), path_exercise))
            paths = paths.at[:, 0].set(jnp.where(is_exercise, exercised, paths[:, 0]))
        return (dx, t, values, paths), None

    zeros = jnp.zeros((num_columns, nodes.shape[0]), dtype=dtype)
    paths = None
    if on_paths:
        paths = jnp.zeros((deviations.discount.shape[0], num_columns, nodes.shape[0]), dtype=arithmetic.dtype)
    xs = (as_level(rows.times), jnp.asarray(rows.rollback), jnp.asarray(rows.is_exercise), as_level(rows.targets),
          jnp.asarray(rows.starts), jnp.asarray(rows.moves), as_level(rows.shares), jnp.arange(len(rows.times)))
    (_, _, values, paths), _ = lax.scan(step, (jnp.zeros((), dtype), as_level(rows.times[0]), zeros, paths), xs)
    return values[0, middle], (None if paths is None else paths[:, 0, middle])


def _basis(swap: _PreparedBermudan, t, x, zeta_t, shift):
    """`(T1, T2, basis)`: the floating coupons' forecast periods on a row at time `t`
    (`LgmVectorised::fixing`'s clamps, `T1 = max(t, start)`, `T2 = max(T1, end)`) and the
    cashflows' basis functions on its grid `x`, `[J, nodes]`: `exp(-q zeta(t)/2 - h x)` with,
    `H` shifted by `shift`, `h = H_T` and `q = H_T^2` for each fixed coupon's payment and each
    floating coupon's (ORE's `reducedDiscountBond`), and `h = H_T1 - H_T2 + H_pay`, `q = H_T1^2 -
    H_T2^2 + H_pay^2` for each floating coupon's forecast times its payment (`_Coefficients`)."""
    as_t = lambda times: jnp.asarray(times, dtype=x.dtype)  # noqa: E731
    h_of = lambda T: _lgm_H(swap.reversion, T, shift)  # noqa: E731
    start = jnp.maximum(t, as_t(swap.float_index_start_times))
    end = jnp.maximum(start, as_t(swap.float_index_end_times))
    h_fixed, h_pay = h_of(as_t(swap.fixed_times)), h_of(as_t(swap.float_pay_times))
    h_start, h_end = h_of(start), h_of(end)
    h = jnp.concatenate([h_fixed, h_pay, h_start - h_end + h_pay])
    q = jnp.concatenate([h_fixed ** 2, h_pay ** 2, h_start ** 2 - h_end ** 2 + h_pay ** 2])
    return start, end, jnp.exp(-(0.5 * q[:, None] * zeta_t + h[:, None] * x[None, :]))


def _bookkeeping(columns, added, starts, moves, shares):
    """A row's cashflow actions on the induction's columns `[..., option, underlying, cache...,
    nodes]` (the level's, or each path's), given the cashflows' values on the row's targets
    `added` (`_Induction`): the underlying gains its coupons and the cache columns that move into
    it, cache columns start and are cleared, and the exercise value is the underlying plus the
    provisional values. Returns the columns and the exercise value. Linear, so it applies to the
    level and to a path's deviation alike."""
    cache = columns[..., 2:, :]
    moves, starts = moves[:, None], starts[:, None]
    underlying = columns[..., 1, :] + added[..., 1, :] + jnp.sum(jnp.where(moves, cache, 0.0), axis=-2)
    cache = jnp.where(moves, 0.0, jnp.where(starts, added[..., 2:, :], cache))
    provisional = added[..., 0, :] + jnp.sum(shares[:, None].astype(cache.dtype) * cache, axis=-2)
    columns = jnp.concatenate([columns[..., :1, :], underlying[..., None, :], cache], axis=-2)
    return columns, underlying + provisional


class _Coefficients:
    """Each cashflow's value at a node, in the numeraire's units, is its coefficient times a
    basis function of the state (`_induction`): a fixed coupon's `amount P(0, T)` (signed for
    the holder: a payer swaption pays the fixed leg) with `exp(-H_T^2 zeta/2 - H_T x)`, and an
    Ibor coupon `N tau (F + m) P(0, pay)` with `F = (P_idx(t, T1; x) / P_idx(t, T2; x) - 1) /
    dcf` as two: `N tau (m - 1/dcf) P(0, pay)` with the payment's function, and `N tau / dcf
    P(0, pay) P_idx(0, T1) / P_idx(0, T2)` with the payment's times the forecast's (`T1 = max(t,
    start)`, `T2 = max(T1, end)`: `LgmVectorised::fixing`'s clamps, so the ratio depends on the
    row past the index start). A coupon whose rate is known (a fixing, or today's forecast off
    today's curve) has the first alone, at `N tau (F + m) P(0, pay)`."""

    def __init__(self, fixed, float_discount, forward, bps, today):
        self.fixed = fixed                     # [Nf]
        self.float_discount = float_discount   # [Ncf]
        self.forward = forward                 # [Ncf]: times the forecast's ratio on a row
        self.bps = bps                         # [Ncf]: N tau P(0, pay), signed
        self.today = today                     # [Ncf]: today's forecast's ratio / dcf (a fixing today)

    @classmethod
    def of(cls, swap: _PreparedBermudan, curve, index, dtype) -> "_Coefficients":
        """The coefficients on the curves `curve` and `index`, in `dtype`."""
        cast = lambda x: jnp.asarray(x, dtype=dtype)  # noqa: E731
        sign = 1.0 if swap.payer else -1.0
        dcf = cast(swap.float_index_dcf)
        bps = sign * cast(swap.notional) * cast(swap.float_accrual) * discount(curve, cast(swap.float_pay_times))
        today = (discount(index, cast(swap.float_index_start_times))
                 / discount(index, cast(swap.float_index_end_times))) / dcf
        known = jnp.where(jnp.asarray(swap.float_is_known), cast(swap.float_known_rates), today - 1.0 / dcf)
        projected = _projected(swap)
        return cls(fixed=-sign * cast(swap.fixed_amounts) * discount(curve, cast(swap.fixed_times)),
                   float_discount=bps * (jnp.where(projected, -1.0 / dcf, known) + cast(swap.float_spread)),
                   forward=jnp.where(projected, bps / dcf, 0.0), bps=bps, today=today)

    def at_row(self, ratio) -> jax.Array:
        """`[J]`: every basis function's coefficient on a row whose forecasts' ratios are
        `ratio` `[Ncf]`."""
        return jnp.concatenate([self.fixed, self.float_discount, self.forward * ratio])


def _projected(swap: _PreparedBermudan) -> jax.Array:
    """Each floating coupon whose rate is forecast on the grid (neither known nor fixed today)."""
    return ~jnp.asarray(swap.float_is_known) & ~jnp.asarray(swap.float_fixed_today)


class _Moved:
    """Each path's deviation of the cashflows' coefficients from the level's (`_Coefficients`),
    in the deviations' format: with `e = expm1(dL)` the deviation of a discount factor at a
    payment (`dL` its log's, the curve's node deviations carried to the time by its own
    interpolation, `log_discount_deviation`), a fixed coupon's is `a e`, a floating coupon's
    first `a e + N tau P0 dF (1 + e)` (`dF` its known rate's deviation: a path fixing's, or today's
    forecast's off the path's curve) and its forecast's `a expm1(dL + dL_idx(T1) - dL_idx(T2))`,
    as the legs' (`engine.pricing.legs`): each a sensitivity times a deviation."""

    def __init__(self, fixed, float_discount, pay, index, index_deviation, dtype):
        self.fixed, self.float_discount, self.pay = fixed, float_discount, pay
        self.index, self.index_deviation, self.dtype = index, index_deviation, dtype

    @classmethod
    def of(cls, swap: _PreparedBermudan, curve, index, level: _Coefficients, deviations: Deviations,
           dtype) -> "_Moved":
        cast = lambda x: jnp.asarray(x, dtype=dtype)  # noqa: E731
        at = lambda c, deviation, times: log_discount_deviation(c, deviation, cast(times))  # noqa: E731
        e_fixed = jnp.expm1(at(curve, deviations.discount, swap.fixed_times))
        pay = at(curve, deviations.discount, swap.float_pay_times)                            # [S, Ncf]
        e_pay = jnp.expm1(pay)
        start, end = swap.float_index_start_times, swap.float_index_end_times
        today = cast(level.today) * jnp.expm1(at(index, deviations.index, start) - at(index, deviations.index, end))
        rate = jnp.where(jnp.asarray(swap.float_is_known), deviations.fixings,
                         jnp.where(jnp.asarray(swap.float_fixed_today), today, 0.0))
        return cls(fixed=cast(level.fixed) * e_fixed,
                   float_discount=cast(level.float_discount) * e_pay + cast(level.bps) * rate * (1.0 + e_pay),
                   pay=pay, index=index, index_deviation=deviations.index, dtype=dtype)

    def at_row(self, level: _Coefficients, ratio, start, end) -> jax.Array:
        """`[S, J]`: every basis function's coefficient's deviation on a row whose forecasts
        run from `start` to `end` `[Ncf]` at the level's ratios `ratio`."""
        forecast = (log_discount_deviation(self.index, self.index_deviation, start.astype(self.dtype))
                    - log_discount_deviation(self.index, self.index_deviation, end.astype(self.dtype)))
        forward = jnp.asarray(level.forward * ratio, dtype=self.dtype) * jnp.expm1(self.pay + forecast)
        return jnp.concatenate([self.fixed, self.float_discount, forward], axis=-1)
