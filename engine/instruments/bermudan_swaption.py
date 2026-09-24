"""
Bermudan and American swaption pricing: ORE's
`QuantExt::NumericLgmMultiLegOptionEngine` (Grid solver), reproduced in JAX.

**What "reproduced" means here, and how it is checked.** At the same grid
settings this module returns ORE's numbers, not merely their converged
limit: tests/test_ore_lgm_parity.py prices Bermudans (aligned and
mid-period), Americans, piecewise volatility and the zero-vol limit through
ORE's own engine, in-process via engine/validation/ore_lgm_oracle.py, and
agrees to within 1e-11 relative. That parity is with ORE's "Grid" solver at
`ShiftHorizon=0`; ORE's FD solver and its default `ShiftHorizon=0.5` are
not reproduced (I-32 in docs/known-issues.md). That depends on reproducing each of the following
exactly, not just to the same limit:

1. **The model.** The LGM bond price `P(t,T,x)` and numeraire `N(t,x)`
   (`engine.models.lgm`, live-verified against
   `ORE.LinearGaussMarkovModel`). **Deliberately NOT** this codebase's other
   HW1F closed form (`engine.models.hull_white`): `ORE.HullWhite` and
   `ORE.LinearGaussMarkovModel` share `(a, sigma)` and today's curve but are
   different numerical realizations for t>0.
2. **The solver.** `LgmConvolutionSolver2`: the state grid
   `x_k = k*sqrt(zeta(t))/nx` with `mx = floor(sx*nx)` points either side of
   zero, Hagan's closed-form quadrature weights (including ORE's boundary
   formula and its clamping of rounding-negative weights), and the
   linear-interpolation rollback between consecutive grid times.
3. **The exercise contract.** Exercise is given in DATES and converted to
   times with the curve's day counter (`time_from_reference`), as ORE
   derives `optionTimes`; Bermudan option times are the dates after the
   evaluation date, American ones ORE's truncated uniform grid (see
   engine.instruments.american_swaption).
4. **Which coupons an exercise enters** (`ExerciseStyle`,
   `buildCashflowInfo`): a Bermudan exercises into whole periods -- a coupon
   belongs while `t <= accrualStart`; an American into broken ones -- while
   `t <= accrualEnd`, credited `couponRatio(t)`.
5. **How each cashflow is valued** (`_cashflow_values_at_nodes`): fixed
   amounts, and Ibor rates projected over the INDEX fixing period with ORE's
   `LgmVectorised::fixing` clamps.
6. **ORE's backward loop itself** (`_GridSchedule`, `_backward_induction_arrays`):
   cashflows are added into a rolled-back `underlyingNpv` at the latest grid
   time they can be, broken coupons are cached and rolled back, and the
   exercise value is `underlyingNpv + provisionalNpv +
   provisionalNpvNonCached` -- ORE's bookkeeping, replayed from masks
   precomputed in Python, since every branch of it depends only on times.
   Evaluating the same exercise value in closed form instead converges to
   the same limit but differs by up to ~1e-4 at a 48-point grid.

**Fully JAX-native.** The backward induction is one `jax.lax.scan` over a
precomputed, fixed-length schedule, so `jax.grad`/`jax.hessian` differentiate
through it (engine.risk.greeks) and it runs on any JAX backend.

**Conditioning** (per-scenario, per-step NPV): extra grid rows at the
simulation's step times; the rolled-back option value at such a row is
interpolated onto each scenario's simulated short rate. The model's Markov
property makes that conditional value exact; the extra rows are the one
place this engine's grid differs from ORE's, which has no such concept.

Why not Jamshidian's decomposition (see engine.instruments.european_swaption):
it needs a single exercise date. ORE itself uses Jamshidian only for
European swaptions and this numeric engine for Bermudan and American ones
(`OREData/ored/portfolio/builders/swaption.cpp`).

See docs/instruments/american-bermudan-swaptions.md for the derivation and
the ORE source for each step.
"""
import math
from dataclasses import dataclass, field, fields
from enum import Enum
from functools import partial
from typing import List, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
import ORE
from jax.tree_util import register_pytree_node_class

from engine.simulation.market_model import ZeroCurveConfig
from engine.models.static_key import StaticKeyMixin
from engine.models.ore_builders import (  # noqa: F401  (DAY_COUNTER is a re-export)
    DAY_COUNTER,
    TIME_AXIS_DAY_COUNTER,
    build_vanilla_swap,
    time_from_reference,
)
from engine.portfolio.validation import _validate_common_fields, _validate_hw_sigma, _validate_tenor
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

#: The simulation TIME AXIS day count and its deprecated `DAY_COUNTER`
#: alias are imported from `engine.models.ore_builders` (see the import
#: above), not re-constructed here -- see that module's TWO ROLES block.
#: Permanently ACT/365: every time here indexes the simulated curve cube's
#: own axis. Never an instrument's accrual basis.


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
    One Bermudan swaption: the option to enter a vanilla fixed-vs-floating
    swap on any one of a discrete list of exercise dates.

    exercise_dates: ascending `ORE.Date`s on which the holder may exercise
    into the (then-remaining) underlying swap -- ORE's own exercise contract.
    Times are derived from them with the curve's day counter, exactly as ORE
    derives `optionTimes`, so an exercise date and the accrual date it names
    map to the identical time and no tolerance is ever needed to match them.
    Dates on or before `evaluation_date` are not exercise opportunities
    (ORE: `if (d > refDate)`). A date inside an accrual period is legitimate
    and, as in ORE, exercises into the next whole period (see
    `ExerciseStyle.BERMUDAN`). `exercisable_dates(cfg)` lists the
    underlying's own accrual starts.

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
    exercise_dates: Sequence[ORE.Date]
    swap_tenor: str = "5Y"
    index_tenor_months: int = 6
    floating_spread: float = 0.0
    n_per_std: int = 48
    std_devs: float = 6.0
    evaluation_date: ORE.Date = field(default_factory=lambda: ORE.Settings.instance().evaluationDate)

    exercise_style = ExerciseStyle.BERMUDAN

    def option_times(self) -> List[float]:
        """ORE's Bermudan `optionTimes`: the time of every exercise date
        strictly after the evaluation date (`calculate()`, lines 487-493)."""
        return [time_from_reference(self.evaluation_date, d)
                for d in self.exercise_dates if d > self.evaluation_date]

    def __post_init__(self) -> None:
        _validate_common_fields(self.notional, self.fixed_rate, self.evaluation_date)
        _validate_tenor(self.swap_tenor, "swap_tenor")
        # None is a valid sentinel meaning "uncalibrated" -- engine.portfolio.
        # price_portfolio fills it in via engine.calibration.lgm.
        # calibrate_lgm_sigma before this config ever reaches a pricer; a
        # bare BermudanSwaptionConfig(hw_sigma=None) constructed outside
        # that flow is likewise valid to build (just not directly priceable
        # until hw_sigma is filled in).
        _validate_hw_sigma(self.hw_sigma)
        if len(self.exercise_dates) == 0:
            raise ValueError("exercise_dates must be non-empty")
        if any(not isinstance(d, ORE.Date) for d in self.exercise_dates):
            raise TypeError(f"exercise_dates must be ORE.Date objects; got {list(self.exercise_dates)}")
        if any(b < a for a, b in zip(self.exercise_dates, self.exercise_dates[1:])):
            raise ValueError(f"exercise_dates must be sorted ascending; got {list(self.exercise_dates)}")


def _build_ore_swap(cfg) -> ORE.VanillaSwap:
    """CPU: builds the real ORE underlying swap -- see
    `engine.models.ore_builders.build_vanilla_swap`, the single shared
    implementation of this construction."""
    return build_vanilla_swap(
        notional=cfg.notional, fixed_rate=cfg.fixed_rate, payer=cfg.payer,
        swap_tenor=cfg.swap_tenor, index_tenor_months=cfg.index_tenor_months,
        floating_spread=cfg.floating_spread, evaluation_date=cfg.evaluation_date,
    )


@register_pytree_node_class
@dataclass(frozen=True, eq=False)
class _PreparedBermudan(StaticKeyMixin):
    """One Bermudan/American swaption's prepared (CPU-resolved) structure.

    **Registered as a JAX pytree, split between traced children and static
    aux data** (`_TRACED` below is the authoritative list):

      - `zero_rates`, `hw_sigma` -- the genuine DIFFERENTIATION targets
        (Delta/Gamma w.r.t. the curve's pillar rates, Vega w.r.t. the
        calibrated Sigma's bucket values -- see
        `engine.risk.greeks._bermudan_price_fn`, which substitutes live
        `jax.grad` tracers into exactly these two via `dataclasses.replace`).
      - `notional`, `fixed_amounts` -- pure numeric SCALE. Not
        differentiated, but traced anyway so trades differing only in size
        share one compiled kernel instead of recompiling per notional.
      - everything else -- the ORE-resolved cashflow schedule, exercise
        times, grid resolution -- is compile-time-constant trade STRUCTURE
        and goes into the aux-data (static) slot, where it keys the cache.

    **Why the split matters.** Before it existed, this whole object had to
    be a `jax.jit` STATIC argument (hashable and concrete, via
    `StaticKeyMixin`), which a tracer-carrying copy can never be -- so
    `_backward_induction_arrays` could not be jitted at all whenever it was
    reached from `engine.risk.greeks`, and every elementwise op around its
    `lax.scan` dispatched as its own tiny XLA program. Measured on the
    4-trade demo portfolio that cost ~22MB of profiler trace and hundreds
    of separate compilations PER tree-priced trade (see
    `docs/concepts/profiling.md`). Splitting differentiable children from
    static aux data is what lets tracers flow through as pytree LEAVES --
    exactly what pytrees are for -- while `jax.jit` still keys its cache on
    the static structure, so the induction compiles ONCE per trade shape and
    is reused by the forward pricer, `jax.grad` and `jax.hessian` alike.

    `StaticKeyMixin` is retained (not redundant): the aux-data tuple this
    pytree hands to `jax.jit` must itself be hashable/comparable by value,
    and `static_key` is what normalizes the NumPy schedule arrays in it.
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
    float_fixed_today: np.ndarray        # [Ncf] bool: fixing date == evaluation date
    float_spread: float
    rate_factor_index: int
    hw_a: float
    hw_sigma: Union[float, Sigma]
    zero_times: np.ndarray
    zero_rates: np.ndarray
    n_per_std: int
    std_devs: float
    final_maturity: float

    # Fields carried as pytree CHILDREN (traced), in tree_flatten's own
    # child order. `zero_rates`/`hw_sigma` are the genuine differentiation
    # targets; `notional`/`fixed_amounts` are here for a different reason --
    # they are pure numeric SCALE, not structure, so keeping them traced
    # means two trades differing only in size (the common case across a real
    # portfolio) share ONE compiled kernel instead of forcing a recompile
    # per notional. Confirmed directly: before this, a second trade
    # identical but for its notional compiled a fresh
    # `_backward_induction_arrays`; after, it compiles none.
    _TRACED = ("zero_rates", "hw_sigma", "notional", "fixed_amounts")

    def tree_flatten(self):
        """Children: the `_TRACED` fields above -- differentiation targets
        plus the pure-scale numerics. `hw_sigma` may itself be a `Sigma` (a
        registered pytree of arrays) or a plain float; either way JAX
        recurses into it correctly as a child.
        Aux data: every other field, as a hashable by-value tuple, so two
        preparations of the same trade collapse onto one compiled kernel
        (the same by-value-not-by-identity reasoning `static_key`'s own
        docstring gives)."""
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
    """NumPy arrays -> a hashable `(bytes, shape, dtype)` triple, so the
    aux-data tuple `tree_flatten` produces can be used as a `jax.jit` cache
    key. Mirrors `engine.models.static_key._norm`'s own normalization (the
    same by-value, content-based scheme), but must be REVERSIBLE here --
    `tree_unflatten` has to rebuild the real array -- which is why this is a
    separate function rather than a reuse of `_norm`."""
    if isinstance(value, np.ndarray):
        return ("__ndarray__", value.tobytes(), value.shape, str(value.dtype))
    return value


def _denorm_static(value):
    """Inverse of `_norm_static` -- rebuilds the NumPy array from its
    content triple. `tree_unflatten` must reconstruct a genuinely equivalent
    object, since JAX round-trips a pytree through flatten/unflatten on
    every `jit`/`grad` boundary crossing.

    The `.copy()` is deliberate: `np.frombuffer` returns a READ-ONLY view
    onto the bytes object, and nothing in this module mutates a schedule
    array today -- but handing back a silently-immutable array where the
    original was writable is the kind of difference that surfaces much
    later, far from here, as a confusing `ValueError: assignment destination
    is read-only`. A schedule array is a handful of floats; the copy is
    free next to the compilation this whole path exists to avoid."""
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


def prepare_bermudan(cfg: "BermudanSwaptionConfig | AmericanSwaptionConfig") -> _PreparedBermudan:
    """CPU: build the ORE underlying swap and resolve everything ORE's
    `NumericLgmMultiLegOptionEngineBase` resolves before its backward run,
    for a `BermudanSwaptionConfig` or an `AmericanSwaptionConfig` alike:

      * the option times (`cfg.option_times()` -- each config type knows
        its own ORE construction);
      * per coupon, ORE's `CashflowInfo`: pay time, accrual start/end, the
        time it stops belonging to the exercised-into swap (by
        `cfg.exercise_style`), and for a floating coupon the index's own
        fixing period and day count fraction, which is what ORE projects
        the rate over.

    fixed_amounts are ORE's own `FixedRateCoupon.amount()`, the same source
    `engine.instruments.european_swaption.prepare_swaption` uses.

    Static per swaption -- run once, not per grid node/step."""
    swap = _build_ore_swap(cfg)
    today = cfg.evaluation_date
    style = cfg.exercise_style
    t_of = lambda d: time_from_reference(today, d)  # noqa: E731

    fixed = {key: [] for key in ("pay", "start", "end", "belongs", "amount")}
    for cf in swap.fixedLeg():
        c = ORE.as_fixed_rate_coupon(cf)
        start, end = t_of(c.accrualStartDate()), t_of(c.accrualEndDate())
        fixed["pay"].append(t_of(c.date()))
        fixed["start"].append(start)
        fixed["end"].append(end)
        fixed["belongs"].append(_belongs_until(style, start, end))
        fixed["amount"].append(c.amount())

    floating = {key: [] for key in (
        "pay", "start", "end", "belongs", "accrual", "idx_start", "idx_end", "idx_dcf", "fixing", "fixed_today")}
    for cf in swap.floatingLeg():
        c = ORE.as_floating_rate_coupon(cf)
        index = c.index()
        fixing_date = c.fixingDate()
        if fixing_date < today:
            raise ValueError(
                f"floating coupon fixing on {fixing_date} is before the evaluation date {today}; "
                f"pricing it needs a historical fixing, which this engine does not hold (I-04)"
            )
        index_start = index.valueDate(fixing_date)
        index_end = index.maturityDate(index_start)
        start, end = t_of(c.accrualStartDate()), t_of(c.accrualEndDate())
        floating["pay"].append(t_of(c.date()))
        floating["start"].append(start)
        floating["end"].append(end)
        floating["belongs"].append(_belongs_until(style, start, end))
        floating["accrual"].append(c.accrualPeriod())
        floating["idx_start"].append(t_of(index_start))
        floating["idx_end"].append(t_of(index_end))
        floating["idx_dcf"].append(index.dayCounter().yearFraction(index_start, index_end))
        floating["fixing"].append(max(0.0, t_of(fixing_date)))
        floating["fixed_today"].append(fixing_date == today)

    exercise_times = np.asarray(cfg.option_times(), dtype=np.float64)
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
    construction, with `mx_ = _grid_half_width(std_devs, n_per_std)` points
    on each side of zero, always including x=0 itself (an odd-length grid, matching
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
    Hagan's closed-form quadrature weights on the standardized grid
    `y_i = h*(i - my)`, `h = 1/n_per_std` -- `LgmConvolutionSolver2`'s
    constructor (lgmconvolutionsolver2.cpp), formula for formula:

        interior:  w_i = (1 + y_i/h)*N(y_i+h) - 2*(y_i/h)*N(y_i) - (1 - y_i/h)*N(y_i-h)
                         + (G(y_i+h) - 2*G(y_i) + G(y_i-h)) / h
        i = 0 and i = 2*my (both, with y_0):
                   w_i = (1 + y_0/h)*N(y_0+h) - (y_0/h)*N(y_0) + (G(y_0+h) - G(y_0)) / h

    with N/G the standard normal CDF/PDF. A weight that comes out negative
    through rounding is set to 0, as ORE does (it refuses one below -1e-10).

    A one-time, config-only precomputation -- plain NumPy/SciPy is
    appropriate here, not a JAX tracing concern.
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


def _close_enough(x: float, y: float) -> bool:
    """QuantLib's `close_enough(x, y)` (ql/math/comparison.hpp, n = 42) --
    the comparison every time test in ORE's backward loop uses."""
    if x == y:
        return True
    diff, tolerance = abs(x - y), 42.0 * np.finfo(np.float64).eps
    if x == 0.0 or y == 0.0:
        return diff < tolerance * tolerance
    return diff <= tolerance * abs(x) or diff <= tolerance * abs(y)


# =============================================================================
# THE GRID-TIME SCHEDULE AND ORE'S CASHFLOW BOOKKEEPING
# (precomputed once per trade, plain Python)
# =============================================================================
@dataclass(frozen=True, eq=False)
class _GridSchedule(StaticKeyMixin):
    """The descending grid times the backward induction walks, and, per grid
    time and per cashflow, what ORE's backward loop does with that cashflow
    there.

    **Times** are `{0} ∪ optionTimes ∪ condition_times`, deduplicated
    EXACTLY -- ORE's `std::set<Real> timeGrid` (`calculate()`, lines
    520-524), plus this engine's conditioning times. Never rounded: an option
    time must stay bit-identical to the belongs-until time of the coupon
    whose accrual date it names.

    **Cashflow actions** replay `NumericLgmMultiLegOptionEngineBase::
    calculate()` (lines 555-585). Every branch there depends only on the
    grid time and the cashflow's own times, never on the model state, so
    the whole Open -> Cached -> Done status history is known before any
    array math runs; `_build_grid_schedule` records it as 0/1 masks, and the
    scan in `_backward_induction_arrays` applies them. For cashflow `i` at
    grid row `g` (in reduced, numeraire-deflated units, like ORE):

      add_pv          underlyingNpv += pv                       (-> Done)
      cache_to_under  underlyingNpv += cache; cache cleared      (-> Done)
      start_cache     cache = pv                                 (-> Cached)
      from_cache      provisionalNpv += cache * couponRatio
      non_cached      provisionalNpvNonCached += pv * couponRatio

    and the exercise value at an option time is `underlyingNpv +
    provisionalNpv + provisionalNpvNonCached`. `underlyingNpv` and every
    cache are rolled back numerically between grid times, exactly as ORE
    rolls them; the exercise value is therefore ORE's numerical
    approximation, not only its mathematical limit.

    ORE's `mustBeEstimated` branch applies only to cashflows with an exact
    estimation time (capped/floored coupons); a vanilla swap has none.
    """
    times: np.ndarray            # [G] descending
    is_exercise: np.ndarray      # [G] bool
    is_condition: np.ndarray     # [G] bool
    condition_index: np.ndarray  # [G] int, index into the original condition_times list (-1 if not a condition time)
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


def _build_grid_schedule(swap: _PreparedBermudan, condition_times: Sequence[float]) -> _GridSchedule:
    exercise_set = set(float(t) for t in swap.exercise_times)
    condition_list = [float(t) for t in condition_times]
    condition_set = set(condition_list)
    times = sorted({0.0} | exercise_set | condition_set, reverse=True)

    condition_first_index = {}
    for i, t in enumerate(condition_list):
        condition_first_index.setdefault(t, i)

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
        is_condition=np.asarray([t in condition_set for t in times], dtype=bool),
        condition_index=np.asarray([condition_first_index.get(t, -1) for t in times], dtype=np.int64),
        rollback_is_identity=rollback_is_identity,
        coupon_ratio=coupon_ratio,
        **masks,
    )


def _bond_prices_at_nodes(curve: _HwZeroCurve, a: float, sigma, t: jax.Array,
                          maturities: jax.Array, x_nodes: jax.Array) -> jax.Array:
    """P(t, T; x) for every state-grid node and every maturity, via
    `engine.models.lgm.bond_price` (live-verified against
    `ORE.LinearGaussMarkovModel.discountBond`). Shape [Nnodes, Nmaturities]."""
    return jax.vmap(lambda T: _lgm_bond_price(curve, a, sigma, t, T, x_nodes))(maturities).T


def _cashflow_values_at_nodes(swap: _PreparedBermudan, curve: _HwZeroCurve, x_nodes: jax.Array, t: jax.Array) -> jax.Array:
    """
    Every cashflow's value at time `t` and every state node -- ORE's
    `CashflowInfo::pv` calculators (`buildCashflowInfo`), signed by the
    option holder's side of each leg. Shape [Nnodes, C], fixed leg first,
    then floating (the `_cashflow_timing` order).

      * fixed coupon: `amount * P(t, pay; x)`;
      * Ibor coupon: `(fixing(t, x) + spread) * accrual * notional *
        P(t, pay; x)`, with `LgmVectorised::fixing` projecting over the
        INDEX period `[d1, d2]` (not the accrual period, I-31):
        `(P(t,T1)/P(t,T2) - 1) / dcf(d1, d2)`, `T1 = max(t, d1)`,
        `T2 = max(T1, d2)`. Once `t` passes `d1` the clamp projects only
        the remaining stub, which `couponRatio` then scales again -- ORE's
        own behaviour, kept. A fixing dated on the evaluation date is
        deterministic in ORE (`index->fixing(today)`, forecast off today's
        curve) and is so here.
    """
    a, sigma = swap.hw_a, swap.hw_sigma
    # Derived from x_nodes' own dtype (not hardcoded): float64 for ordinary
    # pricing, the requested risk dtype for Greeks. The schedule arrays are
    # plain float64 NumPy and would otherwise upcast a float32 trace.
    dtype = x_nodes.dtype
    as_dtype = lambda values: jnp.asarray(values, dtype=dtype)  # noqa: E731

    fixed = (_bond_prices_at_nodes(curve, a, sigma, t, as_dtype(swap.fixed_times), x_nodes)
             * as_dtype(swap.fixed_amounts)[None, :])

    index_start = as_dtype(swap.float_index_start_times)
    index_end = as_dtype(swap.float_index_end_times)
    index_dcf = as_dtype(swap.float_index_dcf)
    T1 = jnp.maximum(t, index_start)
    T2 = jnp.maximum(T1, index_end)
    projected = (_bond_prices_at_nodes(curve, a, sigma, t, T1, x_nodes)
                 / _bond_prices_at_nodes(curve, a, sigma, t, T2, x_nodes) - 1.0) / index_dcf
    zero = jnp.zeros((), dtype=dtype)
    fixed_today = (_lgm_bond_price(curve, a, sigma, zero, index_start, zero)
                   / _lgm_bond_price(curve, a, sigma, zero, index_end, zero) - 1.0) / index_dcf
    fixing = jnp.where(jnp.asarray(swap.float_fixed_today)[None, :], fixed_today[None, :], projected)
    floating = (as_dtype(swap.notional) * (fixing + swap.float_spread) * as_dtype(swap.float_accrual)[None, :]
                * _bond_prices_at_nodes(curve, a, sigma, t, as_dtype(swap.float_pay_times), x_nodes))

    # The holder of a payer swaption pays the fixed leg and receives the
    # floating one; a receiver the reverse (ORE's per-leg `payrec`).
    sign = 1.0 if swap.payer else -1.0
    return jnp.concatenate([-sign * fixed, sign * floating], axis=1)


# =============================================================================
# BACKWARD INDUCTION (jax.lax.scan)
# =============================================================================
@dataclass
class _RolledBackValue:
    """The result of running backward induction on a prepared Bermudan/
    American swaption: the option value at time 0 (a single node, x=0),
    plus everything needed to evaluate the SAME rolled-back value function
    conditional on an arbitrary simulated short rate at any of the
    requested `condition_times` (see price_bermudan_swaptions)."""
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
    ORE's backward run (`NumericLgmMultiLegOptionEngineBase::calculate()`)
    as a single `jax.lax.scan` over the precomputed `_GridSchedule`, plus
    the snapshot bookkeeping for conditioning.

    **Everything is in numeraire-deflated ("reduced") units**, as in ORE
    (`LgmVectorised::reducedDiscountBond`): LGM's state x(t) is driftless,
    so `E[reduced(x_from) | x_to]` under x's own Gaussian transition law is
    exactly the deflated value at `x_to` -- the martingale identity that
    makes Hagan's convolution rollback valid. The early-exercise
    `max(option, exercise)` is taken in the same units.

    Every grid step's (raw, re-inflated) option value function is stacked
    into the scan's output, and the caller selects the condition-time rows
    via `schedule.is_condition`/`condition_index`.
    """
    schedule = _build_grid_schedule(swap, condition_times)
    num_grid = len(schedule.times)
    # The array-producing core IS `jax.jit`-wrapped; the NumPy snapshot
    # bookkeeping stays here, since it concretizes (np.asarray) and indexes
    # with Python ints.
    #
    # `swap` crosses that jit boundary as a PYTREE, not as a static argument
    # (see `_PreparedBermudan`'s own docstring): its differentiable fields
    # (`zero_rates`, `hw_sigma`) are children, so `engine.risk.greeks` can
    # keep differentiating straight through this function with live
    # `jax.grad` tracers in exactly those two slots, while the trade's
    # static structure rides along as hashable aux data that keys the jit
    # cache. `schedule` stays a genuine static argument -- it is pure
    # config-time NumPy, built before any array math starts.
    x_all, values_all = _backward_induction_arrays(swap, schedule)
    curve = _zero_curve_of(swap)
    grid_times = jnp.asarray(schedule.times, dtype=curve.pillar_rates.dtype)
    a, sigma = swap.hw_a, swap.hw_sigma

    condition_state_grids: List[np.ndarray] = []
    condition_values: List[np.ndarray] = []
    if len(condition_times) > 0:
        order = np.argsort([schedule.condition_index[i] for i in range(num_grid) if schedule.is_condition[i]])
        cond_rows = np.nonzero(schedule.is_condition)[0][order]
        for row in cond_rows:
            # Convert the LGM state grid x to the corresponding short rate
            # r BEFORE storing the snapshot -- price_bermudan_swaptions
            # interpolates each scenario's SIMULATED short rate r_t
            # (engine.simulation's own r(t) parametrization) directly
            # against this stored grid, so the grid must already be in
            # r-space (r(t,x) is affine in x -- see engine.models.lgm.r_from_x
            # -- so this conversion is exact).
            r_grid = _lgm_r_from_x(curve, a, sigma, grid_times[row], x_all[row])
            condition_state_grids.append(np.asarray(r_grid))
            condition_values.append(np.asarray(values_all[row]))

    t0_row = num_grid - 1  # the grid is descending and always ends at 0
    value_at_t0 = values_all[t0_row, x_all.shape[1] // 2]

    return _RolledBackValue(
        value_at_t0=value_at_t0,
        condition_times=np.asarray(condition_times, dtype=np.float64),
        condition_state_grids=condition_state_grids,
        condition_values=condition_values,
    )


@partial(jax.jit, static_argnums=1)
def _backward_induction_arrays(swap: _PreparedBermudan, schedule: "_GridSchedule"):
    """The array core of `_run_backward_induction`: ORE's backward loop,
    returning the stacked `(x_all, option_values_all)` of shape
    `[NumGridTimes, NumStateGridPoints]` (option values re-inflated to raw
    units).

    Per grid row, latest first, exactly as `calculate()` orders it:
      1. roll the carried `option`, `underlying` and per-cashflow `cache`
         values back from the previous (later) grid time to this one --
         `LgmConvolutionSolver2::rollback`, a no-op where ORE's is
         (`close_enough(t0, t1)`);
      2. apply this row's cashflow actions (see `_GridSchedule`);
      3. at an option time, `option = max(option, underlying + provisional
         + non_cached)`.

    **`jax.jit`-wrapped, with `swap` as a PYTREE argument and `schedule` as
    a static one.** `swap`'s differentiable fields (`zero_rates`,
    `hw_sigma`) are pytree children, so a `jax.grad`/`jax.hessian` tracer
    substituted into either one flows across this boundary as an ordinary
    traced leaf -- while the trade's static structure rides in the aux-data
    slot and keys the compilation cache. Everything inside compiles into ONE
    program per distinct trade shape; see `docs/concepts/profiling.md` for
    the measured before/after of making this boundary jittable.
    """
    a, sigma, n_per_std, std_devs = swap.hw_a, swap.hw_sigma, swap.n_per_std, swap.std_devs
    curve = _zero_curve_of(swap)
    # Derived from curve's own dtype (not hardcoded) -- see _state_grid's
    # docstring: float64 for ordinary pricing, the requested risk dtype for
    # Greeks. Hardcoded float64 constants would silently upcast a float32
    # Greeks trace under jax_enable_x64=True.
    dtype = curve.pillar_rates.dtype
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


def price_bermudan_swaption_base(cfg: "BermudanSwaptionConfig | AmericanSwaptionConfig") -> float:
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
    bermudan_configs: List["BermudanSwaptionConfig | AmericanSwaptionConfig"],
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
        exercise_dates=[EVAL_DATE + ORE.Period(years, ORE.Years) for years in (1, 2, 3, 4)],
        swap_tenor="5Y",
        evaluation_date=EVAL_DATE,
    )
    print("Bermudan t=0 NPV:", price_bermudan_swaption_base(bermudan_cfg))

    npv_cube = price_bermudan_swaptions([bermudan_cfg], market_cubes["rates"], step_times)
    print("Bermudan NPV cube shape:", npv_cube.shape)
    for i, t in enumerate(config.time_grid[1:]):
        print(f"  t={t:.2f}: mean NPV across scenarios = {float(jnp.mean(npv_cube[:, i, 0])):.2f}")
