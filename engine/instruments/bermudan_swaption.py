"""
Bermudan and American swaptions: the trades, and ORE's `QuantExt::NumericLgmMultiLegOptionEngine`
(Grid solver) reproduced in JAX.

The trade names its currency, index, exercise and settlement; it carries no curve or model. The
engine's model is an LGM whose reversion, volatility and grid settings come from the pricing
configuration (`engine.valuation.config.LgmSwaptionEngineConfig`, ORE's
`LGMGridSwaptionEngineBuilder`), calibrated per trade (`engine.valuation.bermudan`); here they
are explicit arguments of `prepare_bermudan`.

At the same grid settings this returns ORE's numbers, not just their converged limit:
tests/test_ore_lgm_parity.py prices through ORE's own engine
(tests/support/ore_lgm_oracle.py) and agrees to 1e-11 relative. Parity is with the
Grid solver at `ShiftHorizon=0`; ORE's FD solver and default `ShiftHorizon=0.5` are not
reproduced (I-32). It rests on matching each of:

1. The model: LGM bond price and numeraire (`engine.models.lgm`).
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

The induction is one `jax.lax.scan`, so `jax.grad`/`jax.hessian` differentiate through it,
and `jax.vmap` runs it on every simulated path at once (`engine.valuation.bermudan`).

Seasoned trades are priced as ORE prices them: exercise dates on or before the evaluation
date are dropped, coupons that can no longer enter any exercise are not valued, coupons
fixed before it use `fixings`, and after the last exercise date the option is worth 0.

ORE prices Bermudans and Americans with this engine (`LGMGridSwaptionEngineBuilder`,
OREData/ored/portfolio/builders/swaption.hpp). See
docs/instruments/american-bermudan-swaptions.md for the derivation.
"""
import math
from dataclasses import InitVar, dataclass, field, fields
from enum import Enum
from functools import partial
from typing import Dict, List, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
import ORE
from jax.tree_util import register_pytree_node_class

from engine.models.curves import curve_dtype
from engine.models.static_key import StaticKeyMixin
from engine.models.ore_builders import (  # noqa: F401  (DAY_COUNTER is a re-export)
    DAY_COUNTER,
    TIME_AXIS_DAY_COUNTER,
    book_swap_dates,
    build_vanilla_swap,
    is_live,
    known_fixing,
    time_from_reference,
    validate_fixings,
)
from engine.instruments._validation import _validate_common_fields, _validate_identity, _validate_settlement
from engine.models.lgm import (
    Sigma,
    bond_price as _lgm_bond_price,
    numeraire as _lgm_numeraire,
    zeta as _lgm_zeta,
)

# `TIME_AXIS_DAY_COUNTER` (ACT/365) comes from `engine.models.ore_builders`: every time here
# is on the simulation's time axis, never an instrument's accrual basis.


class ExerciseStyle(Enum):
    """Which of ORE's two coupon-membership rules an option's exercise uses
    -- `NumericLgmMultiLegOptionEngineBase::buildCashflowInfo`
    (QuantExt/qle/pricingengines/numericlgmmultilegoptionengine.cpp):

      * BERMUDAN: a coupon belongs to the exercised-into swap while
        `t <= accrualStart`. An exercise inside a period enters the next
        WHOLE period ("bermudan exercise implies that we always exercise
        into whole periods").
      * AMERICAN: a coupon belongs while `t <= accrualEnd`, credited
        `couponRatio(t)` of its value ("american exercise implies that we
        can exercise into broken periods").
    """
    BERMUDAN = "bermudan"
    AMERICAN = "american"


@dataclass
class BermudanSwaptionConfig:
    """
    One Bermudan swaption: the option to enter a vanilla swap on any of a list of dates.

    trade_id / evaluation_date: as in `SwapConfig` (required, keyword only).
    currency / settlement: as in `SwaptionConfig`.
    exercise_dates: ascending `ORE.Date`s. Converted to times with the curve's day counter,
        as ORE derives `optionTimes`, so an exercise date and the accrual date it names map
        to the same float. Dates on or before `evaluation_date` are not exercise
        opportunities. A date inside an accrual period enters the next whole period, as in
        ORE. `exercisable_dates(cfg)` lists the underlying's accrual starts.
    effective_date / maturity_date / swap_tenor / fixings: as in `SwapConfig`. A fixing is
        needed only for a coupon fixed before `evaluation_date` that can still be entered.
    """
    notional: float
    fixed_rate: float
    payer: bool
    exercise_dates: Sequence[ORE.Date] = ()
    effective_date: Optional[ORE.Date] = None
    maturity_date: Optional[ORE.Date] = None
    swap_tenor: InitVar[Optional[str]] = None
    index_tenor_months: int = 6
    floating_spread: float = 0.0
    fixings: Dict[ORE.Date, float] = field(default_factory=dict)
    currency: str = "USD"
    settlement: str = "Physical"
    trade_id: str = field(kw_only=True)
    evaluation_date: ORE.Date = field(kw_only=True)

    exercise_style = ExerciseStyle.BERMUDAN

    def option_times(self, exercise_time_steps_per_year: Optional[int] = None) -> List[float]:
        """ORE's Bermudan `optionTimes`: the time of every exercise date strictly after the
        evaluation date (`calculate()`, lines 487-493). The argument is an American's; a
        Bermudan ignores it."""
        return [time_from_reference(self.evaluation_date, d)
                for d in self.exercise_dates if d > self.evaluation_date]

    def is_expired(self) -> bool:
        """ORE's `Instrument::isExpired`: the last exercise date is on or
        before the evaluation date. An expired option is worth 0."""
        return not is_live(self.exercise_dates[-1], self.evaluation_date)

    def __post_init__(self, swap_tenor: Optional[str]) -> None:
        _validate_identity(self.trade_id, self.evaluation_date)
        _validate_common_fields(self.notional, self.fixed_rate, self.evaluation_date)
        book_swap_dates(self, swap_tenor)
        validate_fixings(self.fixings)
        _validate_settlement(self.settlement)
        if len(self.exercise_dates) == 0:
            raise ValueError("exercise_dates must be non-empty")
        if any(not isinstance(d, ORE.Date) for d in self.exercise_dates):
            raise TypeError(f"exercise_dates must be ORE.Date objects; got {list(self.exercise_dates)}")
        if any(b < a for a, b in zip(self.exercise_dates, self.exercise_dates[1:])):
            raise ValueError(f"exercise_dates must be sorted ascending; got {list(self.exercise_dates)}")


def _build_ore_swap(cfg) -> ORE.VanillaSwap:
    """The ORE underlying swap (see `engine.models.ore_builders.build_vanilla_swap`)."""
    return build_vanilla_swap(
        notional=cfg.notional, fixed_rate=cfg.fixed_rate, payer=cfg.payer,
        effective_date=cfg.effective_date, maturity_date=cfg.maturity_date,
        index_tenor_months=cfg.index_tenor_months, floating_spread=cfg.floating_spread,
    )


@register_pytree_node_class
@dataclass(frozen=True, eq=False)
class _PreparedBermudan(StaticKeyMixin):
    """A Bermudan/American swaption's prepared structure.

    A pytree split between traced children (`_TRACED`) and static aux data:
      - `curve`, `index_curve`, `sigma`: the model's curves (a `ZeroCurve` today, a
        path's `DiscountCurve` in the simulation; `index_curve=None` means the discount
        curve) and volatility, the differentiation targets (Delta/Gamma, Vega);
        `engine.risk.price_functions` substitutes tracers into them.
      - `notional`, `fixed_amounts`: scale only, traced so trades differing only in size
        share one compiled kernel.
      - everything else (schedule, exercise times, grid settings): static structure that
        keys the jit cache.

    This lets `_backward_induction_arrays` be jitted even when called with tracers, so it
    compiles once per trade shape for pricing, `jax.grad` and `jax.hessian`.
    `StaticKeyMixin` makes the aux data hashable by value.
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
    curve: object                 # ZeroCurve or DiscountCurve (None until a caller sets it)
    index_curve: object           # None, ZeroCurve or DiscountCurve
    n_per_std: int
    std_devs: float
    final_maturity: float

    # Pytree children, in tree_flatten order (see the class docstring).
    _TRACED = ("curve", "index_curve", "sigma", "notional", "fixed_amounts", "float_known_rates")

    def tree_flatten(self):
        """Children: the `_TRACED` fields (`sigma` may itself be a `Sigma` pytree).
        Aux data: every other field as a hashable by-value tuple."""
        children = tuple(getattr(self, name) for name in self._TRACED)
        static_fields = tuple(
            (f.name, _norm_static(getattr(self, f.name)))
            for f in fields(self) if f.name not in self._TRACED
        )
        return children, static_fields

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        kwargs = {name: _denorm_static(value) for name, value in aux_data}
        kwargs.update(dict(zip(cls._TRACED, children)))
        return cls(**kwargs)


def _norm_static(value):
    """NumPy array -> hashable `(tag, bytes, shape, dtype)`. Like `static_key._norm`, but
    reversible, because `tree_unflatten` must rebuild the array."""
    if isinstance(value, np.ndarray):
        return ("__ndarray__", value.tobytes(), value.shape, str(value.dtype))
    return value


def _denorm_static(value):
    """Inverse of `_norm_static`. Returns a writable copy (`np.frombuffer` is read-only)."""
    if isinstance(value, tuple) and len(value) == 4 and value[0] == "__ndarray__":
        _, raw, shape, dtype = value
        return np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape).copy()
    return value


def exercisable_dates(cfg) -> List[ORE.Date]:
    """The underlying's own fixed accrual start dates, ascending -- the
    exercise dates of a standard coterminal Bermudan, where each exercise
    enters a whole remaining swap.

    They are a property of the ORE-generated schedule, so a caller cannot
    know them before the swap is built; this builds it and reads them off.
    Every element is a valid exercise date, including the last (it starts
    the final accrual period, strictly before final maturity)."""
    swap = _build_ore_swap(cfg)
    return [ORE.as_fixed_rate_coupon(cf).accrualStartDate() for cf in swap.fixedLeg()]


def _belongs_until(style: ExerciseStyle, accrual_start: float, accrual_end: float) -> float:
    """ORE's `belongsToUnderlyingMaxTime_` for a coupon with no notice
    period (`buildCashflowInfo`, lines 107-115)."""
    return accrual_end if style is ExerciseStyle.AMERICAN else accrual_start


def prepare_bermudan(cfg: "BermudanSwaptionConfig | AmericanSwaptionConfig", *, reversion: float,
                     sigma: Union[float, Sigma], n_per_std: int, std_devs: float,
                     exercise_time_steps_per_year: int, curve=None, index_curve=None) -> _PreparedBermudan:
    """Build the ORE underlying and resolve what `NumericLgmMultiLegOptionEngineBase`
    resolves before its backward run, for a Bermudan or American config:

      * the option times (`cfg.option_times`, an American's on `exercise_time_steps_per_year`);
      * per coupon, ORE's `CashflowInfo`: pay time, accrual start/end, the time it stops
        belonging to the exercised-into swap (by `cfg.exercise_style`), and for a floating
        coupon the index fixing period and its day count fraction.

    The model is the engine's LGM: constant `reversion` and Hagan volatility `sigma`, on the
    grid `n_per_std`/`std_devs` (ORE's `nx`/`sx`). `curve` discounts and is the LGM's term
    structure; `index_curve` (default: `curve`) projects the coupons. Either may be left
    unset and replaced later (`dataclasses.replace`), as a simulation does per path.

    A coupon whose belongs-until time is before the evaluation date is dropped (ORE's
    `isPartOfUnderlying(t)` is false for every `t >= 0`). A kept floating coupon whose
    fixing date has passed takes its fixing from `cfg.fixings`; a missing one raises.
    Fixed amounts are ORE's `FixedRateCoupon.amount()`."""
    swap = _build_ore_swap(cfg)
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

    exercise_times = np.asarray(cfg.option_times(exercise_time_steps_per_year), dtype=np.float64)
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
        reversion=reversion, sigma=sigma, curve=curve, index_curve=index_curve,
        n_per_std=n_per_std, std_devs=std_devs,
        final_maturity=final_maturity,
    )


def _zero_curve_of(swap: _PreparedBermudan):
    """The curve the induction discounts on (and the LGM's term structure). No dtype is
    forced: it is float64 for pricing, and a risk-precision JAX array when
    `engine.risk.greeks` substitutes one."""
    return swap.curve


def _index_curve_of(swap: _PreparedBermudan):
    """The Ibor index's forwarding curve; the discount curve unless one was given."""
    return swap.curve if swap.index_curve is None else swap.index_curve


# State grid and Hagan quadrature convolution, as QuantExt::LgmConvolutionSolver2
# (QuantExt/qle/models/lgmconvolutionsolver2.cpp), in the LGM state variable x(t).
def _state_grid(sigma: float, t: jax.Array, n_per_std: int, std_devs: float, dtype=jnp.float64) -> jax.Array:
    """
    LGM state grid at time t: `x_k = k*sqrt(zeta(t))/n_per_std` for
    `k = -mx..mx`, `mx = _grid_half_width(std_devs, n_per_std)`, as
    `LgmConvolutionSolver2::stateGrid`. The shape is fixed (`2*mx+1`) for `lax.scan`; at
    t=0 every point is 0, as in ORE.

    The sqrt is guarded so its gradient is finite at zeta == 0 (a plain `jnp.sqrt` gives
    NaN Vega at t=0).

    `dtype` is passed explicitly: the caller uses `curve.pillar_rates.dtype`, which is the
    pricing or risk precision as appropriate, whereas `sigma` and `t` are float64 either
    way.
    """
    mx = _grid_half_width(std_devs, n_per_std)
    z = jnp.maximum(_lgm_zeta(sigma, t), 0.0)
    z_safe = jnp.where(z > 0.0, z, 1.0)
    dx = jnp.where(z > 0.0, jnp.sqrt(z_safe), 0.0) / n_per_std
    return dx * jnp.arange(-mx, mx + 1, dtype=dtype)


def _grid_half_width(std_devs: float, n_per_std: int) -> int:
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
    my = _grid_half_width(std_devs, n_per_std)
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
    my = _grid_half_width(std_devs, n_per_std)
    return np.asarray([(1.0 / n_per_std) * (i - my) for i in range(2 * my + 1)], dtype=np.float64)


def _rollback_one_step(
    values: jax.Array, x_from: jax.Array, x_to: jax.Array,
    quad_y: jax.Array, quad_w: jax.Array, std_from_to: jax.Array,
) -> jax.Array:
    """
    E[values(x_from) | x_to] at every point of `x_to`, as `LgmConvolutionSolver2::rollback`.

    x is driftless, so x_from given x_to is Gaussian with mean x_to and standard deviation
    `std_from_to = sqrt(zeta(t_from) - zeta(t_to))`. The expectation is
    `sum_i w_i * f(x_to + y_i * std_from_to)`, with f linearly interpolated on the
    `x_from` grid and flat outside it.
    """
    query = x_to[..., None] + quad_y[None, :] * std_from_to  # [..., n_to, n_quad]
    interpolated = jnp.interp(
        query.reshape(-1), x_from, values,
        left=values[0], right=values[-1],
    ).reshape(query.shape)
    return jnp.sum(interpolated * quad_w[None, :], axis=-1)


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
@dataclass(frozen=True, eq=False)
class _GridSchedule(StaticKeyMixin):
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


def _bond_prices_at_nodes(curve, a: float, sigma, t: jax.Array,
                          maturities: jax.Array, x_nodes: jax.Array) -> jax.Array:
    """P(t, T; x) for every state node and maturity (`engine.models.lgm.bond_price`).
    Shape [Nnodes, Nmaturities]."""
    return jax.vmap(lambda T: _lgm_bond_price(curve, a, sigma, t, T, x_nodes))(maturities).T


def _cashflow_values_at_nodes(swap: _PreparedBermudan, curve, x_nodes: jax.Array, t: jax.Array) -> jax.Array:
    """
    Every cashflow's value at time `t` on every state node, signed for the option holder:
    ORE's `CashflowInfo::pv`. Shape [Nnodes, C], fixed leg first, then floating.

      * fixed coupon: `amount * P(t, pay; x)`;
      * Ibor coupon: `(fixing(t, x) + spread) * accrual * notional * P(t, pay; x)`, with
        `LgmVectorised::fixing` projecting over the index period [d1, d2] on the index's
        forwarding curve: `(P_idx(t,T1)/P_idx(t,T2) - 1) / dcf(d1, d2)`,
        `T1 = max(t, d1)`, `T2 = max(T1, d2)`.
        Past d1 this projects only the remaining stub, which `couponRatio` scales again;
        that is ORE's behaviour. A fixing dated on or before the evaluation date is its
        historical value if known, else (today's) the forecast off today's curve.
    """
    a, sigma = swap.reversion, swap.sigma
    # Work in x_nodes' dtype; the float64 schedule arrays would otherwise upcast float32.
    dtype = x_nodes.dtype
    as_dtype = lambda values: jnp.asarray(values, dtype=dtype)  # noqa: E731

    fixed = (_bond_prices_at_nodes(curve, a, sigma, t, as_dtype(swap.fixed_times), x_nodes)
             * as_dtype(swap.fixed_amounts)[None, :])

    index_curve = _index_curve_of(swap)
    index_start = as_dtype(swap.float_index_start_times)
    index_end = as_dtype(swap.float_index_end_times)
    index_dcf = as_dtype(swap.float_index_dcf)
    T1 = jnp.maximum(t, index_start)
    T2 = jnp.maximum(T1, index_end)
    projected = (_bond_prices_at_nodes(index_curve, a, sigma, t, T1, x_nodes)
                 / _bond_prices_at_nodes(index_curve, a, sigma, t, T2, x_nodes) - 1.0) / index_dcf
    zero = jnp.zeros((), dtype=dtype)
    fixed_today = (_lgm_bond_price(index_curve, a, sigma, zero, index_start, zero)
                   / _lgm_bond_price(index_curve, a, sigma, zero, index_end, zero) - 1.0) / index_dcf
    fixing = jnp.where(jnp.asarray(swap.float_fixed_today)[None, :], fixed_today[None, :], projected)
    fixing = jnp.where(jnp.asarray(swap.float_is_known)[None, :], as_dtype(swap.float_known_rates)[None, :], fixing)
    floating = (as_dtype(swap.notional) * (fixing + swap.float_spread) * as_dtype(swap.float_accrual)[None, :]
                * _bond_prices_at_nodes(curve, a, sigma, t, as_dtype(swap.float_pay_times), x_nodes))

    # The holder of a payer swaption pays the fixed leg and receives the
    # floating one; a receiver the reverse (ORE's per-leg `payrec`).
    sign = 1.0 if swap.payer else -1.0
    return jnp.concatenate([-sign * fixed, sign * floating], axis=1)


# =============================================================================
# BACKWARD INDUCTION (jax.lax.scan)
# =============================================================================
def grid_value(swap: _PreparedBermudan) -> jax.Array:
    """ORE's NPV on the prepared trade's curves (`NumericLgmMultiLegOptionEngineBase::
    calculate()`): the induction's value at t=0 on the state x=0, as
    `LgmConvolutionSolver2::stateGrid(0)`. A JAX scalar, so `jax.grad` differentiates it.

    Values are numeraire-deflated, as in ORE (`LgmVectorised::reducedDiscountBond`); x is
    driftless, so the rollback of a deflated value is its conditional expectation. The
    exercise max is taken in the same units."""
    _, values = _backward_induction_arrays(swap, _build_grid_schedule(swap))
    return values[-1, values.shape[1] // 2]


@partial(jax.jit, static_argnums=1)
def _backward_induction_arrays(swap: _PreparedBermudan, schedule: "_GridSchedule"):
    """ORE's backward loop, returning `(x_all, option_values_all)`, each
    `[NumGridTimes, NumStateGridPoints]` (option values re-inflated to raw units).

    Per grid row, latest first, as `calculate()` orders it:
      1. roll `option`, `underlying` and each cashflow cache back from the previous grid
         time (`LgmConvolutionSolver2::rollback`; a no-op where ORE's is, i.e.
         `close_enough(t0, t1)`);
      2. apply the row's cashflow actions (see `_GridSchedule`);
      3. at an option time, `option = max(option, underlying + provisional + non_cached)`.

    Jitted with `swap` as a pytree and `schedule` static: one program per trade shape.
    """
    a, sigma, n_per_std, std_devs = swap.reversion, swap.sigma, swap.n_per_std, swap.std_devs
    curve = _zero_curve_of(swap)
    # Work in the curve's dtype (pricing or risk precision); hardcoded float64 constants
    # would upcast a float32 Greeks trace.
    dtype = curve_dtype(curve)
    quad_w = jnp.asarray(_hagan_quadrature_weights(n_per_std, std_devs), dtype=dtype)
    quad_y = jnp.asarray(_quadrature_nodes(n_per_std, std_devs), dtype=dtype)
    as_mask = lambda mask: jnp.asarray(mask, dtype=dtype)  # noqa: E731

    num_nodes = 2 * _grid_half_width(std_devs, n_per_std) + 1
    num_cashflows = schedule.add_pv.shape[1]
    grid_times = jnp.asarray(schedule.times, dtype=dtype)
    t_first = grid_times[0]

    def roll_back(values, x_prev, x_t, std_step, identity):
        rolled = jnp.where(
            std_step > 0.0,
            _rollback_one_step(values, x_prev, x_t, quad_y, quad_w, std_step),
            jnp.interp(x_t, x_prev, values),
        )
        return jnp.where(identity, values, rolled)

    def step(carry, row):
        x_prev, t_prev, option, underlying, cache = carry
        (t, is_exercise, identity, add_pv, cache_to_under, start_cache,
         from_cache, non_cached, coupon_ratio) = row

        x_t = _state_grid(sigma, t, n_per_std, std_devs, dtype=dtype)
        # Same gradient-safe sqrt guard as _state_grid: sqrt's derivative is
        # 0/0 at 0, and jax.grad backpropagates through both jnp.where
        # branches, so the variance must be taken off a safe placeholder.
        var_step = jnp.maximum(_lgm_zeta(sigma, t_prev) - _lgm_zeta(sigma, t), 0.0)
        var_step_safe = jnp.where(var_step > 0.0, var_step, 1.0)
        std_step = jnp.where(var_step > 0.0, jnp.sqrt(var_step_safe), 0.0)
        is_first = t == t_first
        identity = identity | is_first
        option = roll_back(option, x_prev, x_t, std_step, identity)
        underlying = roll_back(underlying, x_prev, x_t, std_step, identity)
        cache = jax.vmap(lambda column: roll_back(column, x_prev, x_t, std_step, identity),
                         in_axes=1, out_axes=1)(cache)

        numeraire = _lgm_numeraire(curve, a, sigma, t, x_t)
        pv = _cashflow_values_at_nodes(swap, curve, x_t, t) / numeraire[:, None]  # reduced, [Nn, C]

        underlying = underlying + pv @ add_pv + cache @ cache_to_under
        cache = jnp.where(start_cache[None, :] > 0.0, pv, cache)
        cache = jnp.where(cache_to_under[None, :] > 0.0, 0.0, cache)
        provisional = (cache * coupon_ratio[None, :]) @ from_cache
        provisional_non_cached = (pv * coupon_ratio[None, :]) @ non_cached

        exercise_value = underlying + provisional + provisional_non_cached
        option = jnp.where(is_exercise, jnp.maximum(option, exercise_value), option)
        # The carry is held in the curve's dtype, the precision the caller
        # asked for: a float64 `Sigma` or notional would otherwise promote a
        # float32 Greeks run, and lax.scan requires a fixed carry type.
        x_t, option, underlying, cache = (v.astype(dtype) for v in (x_t, option, underlying, cache))
        return (x_t, t, option, underlying, cache), (x_t, (option * numeraire).astype(dtype))

    zeros = jnp.zeros((num_nodes,), dtype=dtype)
    init = (zeros, t_first, zeros, zeros, jnp.zeros((num_nodes, num_cashflows), dtype=dtype))
    rows = (
        grid_times,
        jnp.asarray(schedule.is_exercise),
        jnp.asarray(schedule.rollback_is_identity),
        as_mask(schedule.add_pv), as_mask(schedule.cache_to_under), as_mask(schedule.start_cache),
        as_mask(schedule.from_cache), as_mask(schedule.non_cached),
        jnp.asarray(schedule.coupon_ratio, dtype=dtype),
    )
    _, (x_all, values_all) = jax.lax.scan(step, init, rows)
    return x_all, values_all
