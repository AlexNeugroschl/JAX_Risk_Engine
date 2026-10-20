"""
ORE's LGM calibration: `LgmBuilder` with `Calibration = Bootstrap` (OREData/ored/model/
lgmbuilder.cpp, irmodelbuilder.cpp), reproduced in JAX for today's market and for every
simulated path.

  * Basket (`build_basket`): each instrument is QuantLib's own `SwaptionHelper`, built on the
    currency's swap index conventions from an (expiry, term) pair given as tenors (the CAM's
    `CalibrationSwaptions`, plan V-6) or dates (a trade's co-terminal basket, plan V-5), and
    read into arrays. Strikes are ATM, or a deal strike moved to ATM +- 3 ATM standard
    deviations when further out (`createSwaptionHelper`'s fallback rule 1).
  * Market value (`market_price`): the helper's `BachelierSwaptionEngine` value on its own
    ATM forward and annuity (nominal 1), out of the money (`SwaptionHelper`: receiver when
    the strike is at or below the forward); at a zero volatility its intrinsic value, as
    QuantLib's formula gives it (`engine.models.bachelier`).
  * Model value (`price_pair`, with the market value): `AnalyticLgmSwaptionEngine` with ORE's default
    `FloatSpreadMapping::proRata`: the swaption is priced on the discount curve alone, and the
    difference between each floating coupon and its discount-curve "flat" amount is moved
    onto the fixed coupons (`S_j`) and the start (`S_m1`).
  * Bootstrap (`bootstrap_sigma`; `bootstrap` inside a traced function):
    `calibrateVolatilitiesIterative` -- one piecewise-constant bucket per helper, ending at the
    helper's expiry (the last open), each solved so the model
    matches the market with the earlier buckets fixed. ORE minimises the relative price error
    with Levenberg-Marquardt to `EndCriteria(1000, 500, 1e-8, 1e-8, 1e-8)`, and finds y* with
    Brent at accuracy 1e-6; here each bucket's volatility is the root on [1e-6, 0.2], and y*
    the root of `yStarHelper`, both to float64 rounding with the configured solver
    (`engine.solvers.roots`, decision A-21): the same roots where they exist, more exactly.
  * Acceptance (`BootstrapResult.misses`, `require_accepted`): `LgmBuilder::calibrate`'s. The
    calibration error is the root mean square over the basket of each helper's relative price
    error |market - model| / market (`getCalibrationError`, `RelativePriceError`), and a
    calibration whose error is not below the tolerance (ORE's `Tolerance`, or the cross-asset
    model's `BootstrapTolerance`) misses: a helper no non-negative variance reaches (its bucket
    at the bracket's floor), a non-finite helper value (a zero ATM volatility's market value is
    0), and a bucket at the bracket's ceiling, where ORE's unbounded optimizer would go on
    (I-73). ORE then fails the build unless `ContinueOnCalibrationError`; the engine refuses a
    miss the same way (`CalibrationError`; `ContinueOnCalibrationError` is not offered, A-24).
  * Derivatives (`calibrated_values`): the calibrated buckets as a function of the curves and the
    helpers' volatilities, differentiated by the implicit function theorem on each helper's
    equation in its own zeta (`helper_residuals`), to any order, the bootstrap's own root-finding
    never differentiated: what an option recalibrated under a bump or a scenario responds with
    (`engine.risk.greeks.price_functions`; I-78).

Every pricing function takes discount and index curves of either type (`ZeroCurve` today,
`DiscountCurve` on a path) and broadcasts over their leading batch axes.
"""
import dataclasses
import math
from dataclasses import dataclass, fields
from functools import partial
from typing import List, NamedTuple, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
import ORE
from jax.scipy.stats import norm

from engine.instruments.schedules import evaluation_date, ibor_index, par_coupon_forecast_period
from engine.market_data.curves import curve_dtype, discount
from engine.market_data.day_counts import TIME_AXIS_DAY_COUNTER, resolve_accrual_day_count
from engine.market_data.market import (
    VOL_BUSINESS_DAY_CONVENTION, VOL_CALENDAR, SwaptionVolSurface, swap_length, swap_length_between,
)
from engine.models.bachelier import bachelier
from engine.models.lgm import H as lgm_H
from engine.precision import matmul
from engine.solvers.roots import DEFAULT_SOLVER, Steps, elementwise_root_tangent, implicit_root, solve

#: `IrModelBuilder::maxAtmStdDev`: a helper strike further from ATM is moved to this many ATM
#: standard deviations (fallback rule 1).
MAX_ATM_STD_DEV = 3.0

#: The bracket of one bucket's volatility: [0.01bp, 2000bp] of normal volatility.
SIGMA_BRACKET = (1e-6, 0.20)

#: The default acceptance tolerance of a calibration's error (`BootstrapResult.misses`). ORE has
#: no default: its `Tolerance` (a trade's LGM) and `BootstrapTolerance` (the cross-asset model)
#: must be configured. 1e-4 is the value of ORE's shared example configuration
#: (Examples/Input/pricingengine.xml), of its products example's American swaption and of every
#: example simulation's `BootstrapTolerance`; the products example's Bermudan takes 0.20.
BOOTSTRAP_TOLERANCE = 1e-4


def check_tolerance(tolerance: float, name: str) -> None:
    """Refuse an acceptance tolerance that is not finite and positive, naming the setting."""
    if not (isinstance(tolerance, (int, float)) and math.isfinite(tolerance) and tolerance > 0.0):
        raise ValueError(f"{name} must be finite and > 0 (ORE compares the calibration error with it); "
                         f"got {tolerance!r}")


class CalibrationError(ValueError):
    """A calibration that misses ORE's acceptance (`require_accepted`), as `LgmBuilder` fails
    without `ContinueOnCalibrationError`."""

#: Steps of a bucket's volatility and of a helper's y* (`engine.solvers.roots`). Bisection's
#: are the counts before Newton (A-21). Newton's are measured (2026-10-07, the shared sloped market
#: today and on 256 LGM and Hull-White paths): a bucket reaches its rounding in 5 steps, a y*
#: from 0 in 3; each count has two steps of margin
#: (tests/test_root_solvers.py).
SIGMA_STEPS = Steps(bisection=60, newton=7)
Y_STAR_STEPS = Steps(bisection=100, newton=5)

#: y*'s starting window, widened when it does not hold the root.
Y_STAR_WINDOW = 1.0


@dataclass(frozen=True)
class SwapIndexConventions:
    """The swap index a currency's helpers are built on (ORE's `swapIndexBase`): its fixed
    leg tenor and day counter, and the tenor of its Ibor index (`ibor_index`). ORE switches to
    a short swap index below its tenor; one index serves every term here."""
    fixed_tenor: str = "1Y"
    fixed_day_counter: str = "ACT/365"
    index_tenor_months: int = 6


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class BasketInstrument:
    """One `SwaptionHelper` on a reference date: times are ACT/365 from it (the model's and
    the helper's pricing time axis).

    `vol_option_time`/`vol_swap_length` are where ORE reads its volatility (for a tenor
    expiry the vol surface's own option date, which can differ from the helper's exercise
    date). Floating coupons carry the owner fixed coupon and weight of ORE's proRata mapping
    (`owner = -1`: beyond the fixed schedule, ignored as ORE's loop ignores it).

    A pytree, so a basket is an argument of the jitted bootstrap rather than baked into it:
    one program serves every basket of the same shape (I-22). The proRata mapping (`owner`,
    `lambda2`) is static, as `_corrections` builds a matrix from it on the host."""
    expiry_time: float
    vol_option_time: float
    vol_swap_length: float
    fixed_pay: np.ndarray
    fixed_accrual: np.ndarray
    float_pay: np.ndarray
    float_start: np.ndarray
    float_end: np.ndarray
    float_accrual: np.ndarray
    forecast_start: np.ndarray
    forecast_end: np.ndarray
    spanning: np.ndarray
    owner: np.ndarray
    lambda2: np.ndarray
    deal_strike: Optional[float] = None

    @property
    def last(self) -> int:
        """The index of the last fixed coupon, where the swap's notional is repaid."""
        return self.fixed_pay.size - 1

    def astype(self, dtype) -> "BasketInstrument":
        """The same helper with every real-valued array in `dtype`, so it prices in the
        dtype of the curves it is given."""
        cast = {f.name: np.asarray(getattr(self, f.name), dtype=dtype) for f in fields(self)
                if isinstance(getattr(self, f.name), np.ndarray) and getattr(self, f.name).dtype.kind == "f"}
        return dataclasses.replace(self, **cast)

    def tree_flatten(self):
        children = tuple(getattr(self, f.name) for f in fields(self) if f.name not in _STATIC_FIELDS)
        static = tuple((tuple(getattr(self, name).tolist()), getattr(self, name).dtype.str) for name in _STATIC_FIELDS)
        return children, static

    @classmethod
    def tree_unflatten(cls, static, children):
        names = [f.name for f in fields(cls) if f.name not in _STATIC_FIELDS]
        arrays = {name: np.asarray(values, dtype=dtype) for name, (values, dtype) in zip(_STATIC_FIELDS, static)}
        return cls(**dict(zip(names, children)), **arrays)


#: `BasketInstrument` fields kept on the host when it crosses a `jax.jit` boundary.
_STATIC_FIELDS = ("owner", "lambda2")


class HelperStack(NamedTuple):
    """A basket's helpers as one helper of arrays with a leading helper axis, each padded to the
    basket's most fixed and floating coupons, so its pricing is traced once and mapped over the
    helpers (`helper_residuals`) rather than traced per helper. A padded coupon is inert: no
    accrual, paid when the helper's last coupon is (a fixed one) or starting and ending there (a
    floating one, owned by no fixed coupon), so it adds exactly 0 to every sum; `last` is each
    helper's last real fixed coupon, where its notional is repaid; `deal_strike` NaN where the
    helper is at the money (`stack_helpers`)."""
    expiry_time: jax.Array
    vol_option_time: jax.Array
    vol_swap_length: jax.Array
    fixed_pay: jax.Array
    fixed_accrual: jax.Array
    float_pay: jax.Array
    float_start: jax.Array
    float_end: jax.Array
    float_accrual: jax.Array
    forecast_start: jax.Array
    forecast_end: jax.Array
    spanning: jax.Array
    owner: jax.Array
    lambda2: jax.Array
    last: jax.Array
    deal_strike: Optional[jax.Array]


def stack_helpers(basket: Sequence[BasketInstrument]) -> HelperStack:
    """The basket as a `HelperStack`: traceable, so a price function stacks the helpers it is
    given inside its own program."""
    n_fixed = max(b.fixed_pay.size for b in basket)
    n_float = max(b.float_pay.size for b in basket)

    def padded(values, size, fill):
        values = jnp.asarray(values)
        return jnp.concatenate([values, jnp.broadcast_to(fill, (size - values.shape[0],)).astype(values.dtype)])

    def stacked(name, size, fill):
        """Field `name` of every helper, padded to `size` with `fill(helper)`."""
        return jnp.stack([padded(getattr(b, name), size, fill(b)) for b in basket])

    last_of = lambda name: lambda b: getattr(b, name)[-1]  # noqa: E731
    nothing = lambda b: 0.0  # noqa: E731
    strikes = [b.deal_strike for b in basket]
    return HelperStack(
        expiry_time=jnp.stack([jnp.asarray(b.expiry_time) for b in basket]),
        vol_option_time=jnp.stack([jnp.asarray(b.vol_option_time) for b in basket]),
        vol_swap_length=jnp.stack([jnp.asarray(b.vol_swap_length) for b in basket]),
        fixed_pay=stacked("fixed_pay", n_fixed, last_of("fixed_pay")),
        fixed_accrual=stacked("fixed_accrual", n_fixed, nothing),
        float_pay=stacked("float_pay", n_float, last_of("float_pay")),
        float_start=stacked("float_start", n_float, last_of("float_end")),
        float_end=stacked("float_end", n_float, last_of("float_end")),
        float_accrual=stacked("float_accrual", n_float, nothing),
        forecast_start=stacked("forecast_start", n_float, last_of("forecast_end")),
        forecast_end=stacked("forecast_end", n_float, last_of("forecast_end")),
        spanning=stacked("spanning", n_float, lambda b: 1.0),
        owner=np.stack([np.pad(b.owner, (0, n_float - b.owner.size), constant_values=-1) for b in basket]),
        lambda2=np.stack([np.pad(b.lambda2, (0, n_float - b.lambda2.size)) for b in basket]),
        last=np.array([b.last for b in basket]),
        deal_strike=None if all(k is None for k in strikes) else jnp.stack(
            [jnp.asarray(np.nan if k is None else k, dtype=jnp.result_type(basket[0].fixed_accrual)) for k in strikes]))


def _helper(reference: ORE.Date, expiry, term, conventions: SwapIndexConventions):
    """QuantLib's `SwaptionHelper` for (expiry, term), as `createSwaptionHelper` builds it,
    on a placeholder curve: its schedule depends only on the reference date."""
    curve = ORE.YieldTermStructureHandle(ORE.FlatForward(reference, 0.03, TIME_AXIS_DAY_COUNTER))
    index = ibor_index(conventions.index_tenor_months, curve)
    day_counter = resolve_accrual_day_count(conventions.fixed_day_counter)
    return ORE.SwaptionHelper(
        expiry, term, ORE.QuoteHandle(ORE.SimpleQuote(0.01)), index, ORE.Period(conventions.fixed_tenor),
        day_counter, index.dayCounter(), curve, ORE.BlackCalibrationHelper.RelativePriceError,
        ORE.nullDouble(), 1.0, ORE.Normal)


def build_basket(
    reference: ORE.Date,
    expiries: Sequence[Union[str, ORE.Date]],
    terms: Sequence[Union[str, ORE.Date]],
    conventions: SwapIndexConventions = SwapIndexConventions(),
    deal_strikes: Optional[Sequence[Optional[float]]] = None,
) -> List[BasketInstrument]:
    """The helpers `IrModelBuilder::buildSwaptionBasket` builds on `reference`.

    Expiries and terms are both tenors (`"1Y"`, the CAM's basket) or both dates (a trade's
    exercise dates and maturity). A date term is extended to at least one month past the
    helper's start (`getExpiryAndTerm`). `deal_strikes[i] = None` means ATM. The vols are
    read separately (`basket_vols`, or the pricing context's lookup on a bumped or rolled market)."""
    if len(expiries) != len(terms):
        raise ValueError("expiries and terms must have the same length")
    deal_strikes = list(deal_strikes) if deal_strikes is not None else [None] * len(expiries)
    t_of = lambda d: TIME_AXIS_DAY_COUNTER.yearFraction(reference, d)  # noqa: E731
    basket = []
    with evaluation_date(reference):  # the helper projects its underlying from it
        for expiry, term, strike in zip(expiries, terms, deal_strikes):
            if isinstance(expiry, ORE.Date) != isinstance(term, ORE.Date):
                raise TypeError("give expiry and term both as tenors or both as dates")
            if isinstance(expiry, ORE.Date):
                index = ibor_index(conventions.index_tenor_months)
                start = index.valueDate(index.fixingCalendar().adjust(expiry))
                term = max(term, start + ORE.Period(1, ORE.Months))
                helper = _helper(reference, expiry, term, conventions)
                vol_option_time, vol_length = t_of(expiry), max(swap_length_between(start, term), 1.0 / 12.0)
            else:
                helper = _helper(reference, ORE.Period(expiry), ORE.Period(term), conventions)
                option_date = VOL_CALENDAR.advance(reference, ORE.Period(expiry), VOL_BUSINESS_DAY_CONVENTION)
                vol_option_time, vol_length = t_of(option_date), max(swap_length(term), 1.0 / 12.0)
            swap = helper.underlying()
            fixed = [ORE.as_fixed_rate_coupon(c) for c in swap.fixedLeg()]
            floating = [ORE.as_floating_rate_coupon(c) for c in swap.floatingLeg()]
            ratio = max(1, int(len(floating) / len(fixed) + 0.5))
            owner = np.full(len(floating), -1, dtype=np.int64)
            lambda2 = np.zeros(len(floating))
            for k in range(min(len(floating), ratio * len(fixed))):
                owner[k], lambda2[k] = k // ratio, (k % ratio + 1) / ratio
            periods = [par_coupon_forecast_period(c) for c in floating]
            basket.append(BasketInstrument(
                expiry_time=t_of(helper.swaptionExpiryDate()),
                vol_option_time=vol_option_time,
                vol_swap_length=vol_length,
                fixed_pay=np.array([t_of(c.date()) for c in fixed]),
                fixed_accrual=np.array([c.accrualPeriod() for c in fixed]),
                float_pay=np.array([t_of(c.date()) for c in floating]),
                float_start=np.array([t_of(c.accrualStartDate()) for c in floating]),
                float_end=np.array([t_of(c.accrualEndDate()) for c in floating]),
                float_accrual=np.array([c.accrualPeriod() for c in floating]),
                forecast_start=np.array([t_of(p[0]) for p in periods]),
                forecast_end=np.array([t_of(p[1]) for p in periods]),
                spanning=np.array([p[2] for p in periods]),
                owner=owner, lambda2=lambda2, deal_strike=strike,
            ))
    return basket


def basket_vols(basket: Sequence[BasketInstrument], surface: SwaptionVolSurface, reference: ORE.Date) -> np.ndarray:
    """Each helper's volatility on today's surface, where ORE reads it."""
    return np.array([float(surface.volatility(reference, b.vol_option_time, b.vol_swap_length)) for b in basket])


# ---------------------------------------------------------------------------
# Pricing (JAX; broadcasts over the curves' leading batch axes)
# ---------------------------------------------------------------------------
@dataclass
class _Legs:
    """The curve-dependent pieces of one helper: discount factors, the ATM forward, the
    annuity, and the floating coupons' amounts on the index curve and "flat" amounts on the
    discount curve."""
    fixed_df: jax.Array     # [..., Nf]
    d0: jax.Array           # [...]  discount at the first floating accrual start
    float_df: jax.Array     # [..., Nc]
    float_amount: jax.Array  # [..., Nc] at-par forecast off the index curve, nominal 1
    flat_amount: jax.Array  # [..., Nc] P_d(start)/P_d(end) - 1
    annuity: jax.Array      # [...]
    forward: jax.Array      # [...]


def _legs(instrument: BasketInstrument, disc, index) -> _Legs:
    fixed_df = discount(disc, jnp.asarray(instrument.fixed_pay))
    float_df = discount(disc, jnp.asarray(instrument.float_pay))
    forecast = (discount(index, jnp.asarray(instrument.forecast_start))
                / discount(index, jnp.asarray(instrument.forecast_end)) - 1.0) / instrument.spanning
    float_amount = forecast * instrument.float_accrual
    flat_amount = (discount(disc, jnp.asarray(instrument.float_start))
                   / discount(disc, jnp.asarray(instrument.float_end)) - 1.0)
    annuity = jnp.sum(fixed_df * instrument.fixed_accrual, axis=-1)
    forward = jnp.sum(float_amount * float_df, axis=-1) / annuity
    d0 = discount(disc, jnp.asarray(instrument.float_start[0]))
    return _Legs(fixed_df, d0, float_df, float_amount, flat_amount, annuity, forward)


def _strike_and_receiver(instrument: BasketInstrument, legs: _Legs, atm_std_dev):
    """The helper's strike (ATM, or the deal strike within MAX_ATM_STD_DEV ATM standard
    deviations of the forward) and whether it is a receiver (`strike <= forward`)."""
    if instrument.deal_strike is None:
        return legs.forward, jnp.ones_like(legs.forward, dtype=bool)
    band = MAX_ATM_STD_DEV * atm_std_dev
    clipped = jnp.clip(instrument.deal_strike, legs.forward - band, legs.forward + band)
    # A stack's helper at the money has a NaN deal strike (`stack_helpers`).
    strike = jnp.where(jnp.isnan(instrument.deal_strike), legs.forward, clipped)
    return strike, strike <= legs.forward


def _market(instrument: BasketInstrument, legs: _Legs, vol):
    """(market value, strike, is_receiver): Bachelier on the ATM forward and annuity, with
    `vol` the normal volatility ORE reads for the helper (its intrinsic value at a zero
    volatility, 0 at the money)."""
    std_dev = vol * jnp.sqrt(instrument.expiry_time)
    strike, receiver = _strike_and_receiver(instrument, legs, std_dev)
    return legs.annuity * bachelier(legs.forward, strike, std_dev, call=~receiver), strike, receiver


def market_price(instrument: BasketInstrument, disc, index, vol) -> jax.Array:
    """The helper's market value (nominal 1)."""
    return _market(instrument, _legs(instrument, disc, index), vol)[0]


def price_pair(instrument: BasketInstrument, disc, index, vol, reversion: float, zeta_expiry,
               solver: str = DEFAULT_SOLVER):
    """(market value, model value) of one helper, nominal 1, on the same strike and type.
    The model is an LGM with constant `reversion` and zeta(expiry) = `zeta_expiry`; `solver`
    finds its y*."""
    legs = _legs(instrument, disc, index)
    market, strike, receiver = _market(instrument, legs, vol)
    return market, _analytic_lgm(instrument, legs, reversion, zeta_expiry, strike, receiver, solver)


def _corrections(instrument: BasketInstrument, legs: _Legs):
    """proRata `S_j` per fixed coupon and `S_m1` at the start (`AnalyticLgmSwaptionEngine::
    calculate`, nominal 1)."""
    correction = (legs.float_amount - legs.flat_amount) * legs.float_df
    dtype = correction.dtype
    owned = jnp.asarray(instrument.owner) >= 0
    lambda2 = jnp.where(owned, jnp.asarray(instrument.lambda2, dtype=dtype), 0.0)
    lambda1 = jnp.where(owned, 1.0 - jnp.asarray(instrument.lambda2, dtype=dtype), 0.0)
    n_fixed = instrument.fixed_pay.shape[-1]
    to_fixed = (jnp.asarray(instrument.owner)[:, None] == jnp.arange(n_fixed)).astype(dtype)   # [Nc, Nf]
    sum1 = matmul(correction * lambda1, to_fixed)               # [..., Nf]
    sum2 = matmul(correction * lambda2, to_fixed)
    # S_j = sum2_j / D_j + sum1_{j+1} / D_j; S_m1 = sum1_0 / D0.
    shifted = jnp.concatenate([sum1[..., 1:], jnp.zeros_like(sum1[..., :1])], axis=-1)
    S = (sum2 + shifted) / legs.fixed_df
    S_m1 = sum1[..., 0] / legs.d0
    return S, S_m1


def _analytic_lgm(instrument, legs: _Legs, reversion, zeta_expiry, strike, receiver, solver: str):
    """`AnalyticLgmSwaptionEngine::calculate` (j1 = k1 = 0: a helper starts after expiry), in
    closed form around the root y* of the exercise boundary (`yStarHelper`), found by `solver`
    from 0. A helper is struck within `MAX_ATM_STD_DEV` of the money, so y* is near 0, and from
    there Newton converges in 3 steps; from the y* of another volatility (a warm start) it can
    start far up the boundary's steep side and need more (measured in float32 on a path:
    5 steps from -0.5 left a helper 2% off). The price is stationary in y* at the root, so its
    first derivatives with y* held fixed are exact; y* still carries its implicit derivative
    (`implicit_root`), which the second derivatives need: a calibration's second-order response
    to the curves (`calibrated_values`), an option's Gamma through its recalibration."""
    S, S_m1 = _corrections(instrument, legs)
    amounts = jnp.asarray(instrument.fixed_accrual) * strike[..., None] - S      # [..., Nf]
    H0 = lgm_H(reversion, jnp.asarray(instrument.float_start[0]))
    dH = lgm_H(reversion, jnp.asarray(instrument.fixed_pay)) - H0                # [Nf]
    zeta = jnp.asarray(zeta_expiry)[..., None]
    D = legs.fixed_df
    # The notional is repaid with the last fixed coupon (`last`: a stack's helper may have
    # padded coupons after it).
    D_last, dH_last = jnp.take(D, instrument.last, axis=-1), jnp.take(dH, instrument.last, axis=-1)
    shape = jnp.broadcast_shapes(legs.d0.shape, zeta.shape[:-1])
    y_star = implicit_root(_y_star_helper, (amounts, D, dH, zeta, S_m1, legs.d0, D_last, dH_last),
                           jnp.zeros(shape, dtype=legs.d0.dtype), solver=solver, steps=Y_STAR_STEPS,
                           increasing=False, window=Y_STAR_WINDOW)
    sqrt_zeta = jnp.sqrt(zeta[..., 0])
    w = jnp.where(receiver, 1.0, -1.0)   # ORE: Call (payer) -> -1, Put (receiver) -> +1
    Phi = lambda arg: norm.cdf(w * arg / sqrt_zeta)  # noqa: E731
    total = jnp.sum(amounts * D * norm.cdf(w[..., None] * (y_star[..., None] + dH * zeta) / sqrt_zeta[..., None]),
                    axis=-1)
    total = total - S_m1 * legs.d0 * Phi(y_star)
    total = total + D_last * Phi(y_star + dH_last * zeta[..., 0]) - legs.d0 * Phi(y_star)
    return w * total


def _y_star_helper(y, params):
    """`yStarHelper`: the underlying at expiry as a function of the state y, decreasing in it."""
    amounts, D, dH, zeta, S_m1, d0, D_last, dH_last = params
    bond = lambda h, y: jnp.exp(-h * y - 0.5 * h ** 2 * zeta[..., 0])  # noqa: E731
    bonds = jnp.exp(-dH * y[..., None] - 0.5 * dH ** 2 * zeta)
    return jnp.sum(amounts * D * bonds, axis=-1) - S_m1 * d0 + D_last * bond(dH_last, y) - d0


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
@dataclass
class BootstrapResult:
    """Bucket times (the helpers' expiries but the last) and values `[..., n]`, with each
    helper's market and model value at the result. `hit_ceiling` flags a bucket that could
    not reach its market price inside the bracket; `rmse` `[...]` is ORE's calibration error,
    the root mean square of the helpers' relative price errors (`getCalibrationError`)."""
    times: np.ndarray
    values: jax.Array
    market: jax.Array
    model: jax.Array
    hit_ceiling: jax.Array
    rmse: jax.Array

    def misses(self, tolerance: float) -> jax.Array:
        """`[...]`: True where the calibration misses ORE's acceptance (see the module
        docstring): its error not below `tolerance` (a non-finite one included), or a bucket at
        the bracket's ceiling."""
        return _misses(self.rmse, self.hit_ceiling, tolerance)


@jax.jit
def _misses(rmse: jax.Array, hit_ceiling: jax.Array, tolerance) -> jax.Array:
    """`BootstrapResult.misses`, one program per shape whatever the tolerance."""
    return ~(rmse < tolerance) | jnp.any(hit_ceiling, axis=-1)


def require_accepted(result: BootstrapResult, tolerance: float, what: str) -> None:
    """Refuse an unbatched calibration that misses ORE's acceptance (`BootstrapResult.misses`),
    with a `CalibrationError` naming `what` (the trade or currency, and the date), its error and
    the tolerance, as `LgmBuilder` fails without `ContinueOnCalibrationError`."""
    if not bool(result.misses(tolerance)):
        return
    rmse = float(result.rmse)
    reasons = []
    if not np.isfinite(rmse):
        reasons.append("a helper's market or model value is not finite (a zero volatility at the money has no "
                       "relative price error)")
    elif not rmse < tolerance:
        reasons.append(f"the root mean square of its helpers' relative price errors is {rmse:.3g}, not below the "
                       f"tolerance {tolerance:g}")
    ceiling = np.flatnonzero(np.asarray(result.hit_ceiling))
    if ceiling.size:
        reasons.append(f"helper(s) {ceiling.tolist()} need a volatility above the bracket's {SIGMA_BRACKET[1]}")
    raise CalibrationError(f"{what}: the LGM calibration misses its basket (ORE's LgmBuilder acceptance): "
                           + "; ".join(reasons))


def ceiling_tolerance(dtype) -> float:
    """How close to the bracket's top a solved volatility counts as the top: 1e-9 relative, or
    four units in the last place where the dtype cannot resolve 1e-9 (in float32
    `hi * (1 - 1e-9)` rounds to `hi`, so a bucket stuck below it went unflagged). Newton
    returns the top itself for a bucket out of reach."""
    return max(1e-9, 4.0 * float(jnp.finfo(dtype).eps))


def bootstrap_sigma(basket: Sequence[BasketInstrument], disc, index, vols, reversion: float,
                    solver: str = DEFAULT_SOLVER) -> BootstrapResult:
    """`calibrateVolatilitiesIterative` over `basket` (ascending expiries): bucket i covers
    [expiry_{i-1}, expiry_i) (the last is open), and zeta(expiry_i) = zeta(expiry_{i-1}) +
    sigma_i^2 (expiry_i - expiry_{i-1}). `vols` is `[..., n]` (or `[n]`), matched to the
    curves' batch axes. Computed in the curves' dtype (float64: every calibration is on a
    market of the as-of date, A-10), each root by `solver` (`bootstrap`)."""
    expiries = basket_expiries(basket)
    dtype = curve_dtype(disc)
    values, market, model, hit = bootstrap(tuple(b.astype(dtype) for b in basket), disc, index, vols,
                                           np.diff(expiries, prepend=0.0), reversion, solver)
    return BootstrapResult(times=expiries[:-1], values=values, market=market, model=model, hit_ceiling=hit,
                           rmse=_calibration_error(market, model))


def basket_expiries(basket: Sequence[BasketInstrument]) -> np.ndarray:
    """The helpers' expiry times, on the host: the bootstrap's bucket ends. A basket must have at
    least one helper and strictly increasing expiries."""
    if not basket:
        raise ValueError("a bootstrap needs at least one helper")
    expiries = np.array([float(b.expiry_time) for b in basket])
    if np.any(np.diff(expiries) <= 0.0):
        raise ValueError("basket expiries must increase strictly")
    return expiries


def bootstrap(basket: Sequence[BasketInstrument], disc, index, vols, dt, reversion: float,
              solver: str = DEFAULT_SOLVER):
    """`bootstrap_sigma`'s buckets, traceable: the helpers in the curves' dtype and `dt`, each
    bucket's length (the first from 0), as data, so a price function calibrates inside its own
    program (`engine.risk.greeks.price_functions`). Returns `(values, market, model,
    hit_ceiling)`, each `[..., n]`.

    Newton starts each bucket at its helper's market volatility, close to the bucket's in level,
    and never at an edge of the bracket (where a bucket after a floored one would start, the
    slope vanishes). The values carry no derivative; `calibrated_values` gives them the
    calibration's.

    Each bucket is one jitted program per helper shape and curve shape (`_bootstrap_bucket`),
    whatever the dates, rates and volatilities (I-22); a basket's helpers recur in the baskets
    of other trades and bumped markets, which reuse them."""
    dtype = curve_dtype(disc)
    vols = jnp.asarray(vols, dtype=dtype)
    values, markets, models, ceiling = [], [], [], []
    zeta_before = jnp.zeros(vols.shape[:-1], dtype=dtype)
    for i, instrument in enumerate(basket):
        sigma, market, model, hit, zeta_before = _bootstrap_bucket(
            instrument, disc, index, vols[..., i], zeta_before, jnp.asarray(dt[i], dtype=dtype), reversion, solver)
        values.append(sigma)
        markets.append(market)
        models.append(model)
        ceiling.append(hit)
    stack = lambda xs: jnp.stack(xs, axis=-1)  # noqa: E731
    return stack(values), stack(markets), stack(models), stack(ceiling)


@jax.jit
def _calibration_error(market: jax.Array, model: jax.Array) -> jax.Array:
    """`getCalibrationError` with `RelativePriceError` over the last axis, the basket."""
    return jnp.sqrt(jnp.mean(jnp.square((market - model) / market), axis=-1))


def calibration_misses(market: jax.Array, model: jax.Array, hit_ceiling: jax.Array, tolerance) -> jax.Array:
    """`[...]`: True where a calibration's helpers' `market` and `model` values `[..., n]` miss
    ORE's acceptance (`BootstrapResult.misses`), traceable."""
    return _misses(_calibration_error(market, model), hit_ceiling, tolerance)


@partial(jax.jit, static_argnums=(7,))
def _bootstrap_bucket(instrument: BasketInstrument, disc, index, vol, zeta_before, dt, reversion, solver: str):
    """One bucket of the bootstrap (`_bucket`): one program per helper and curve shape."""
    return _bucket(instrument, disc, index, vol, zeta_before, dt, reversion=reversion, solver=solver)


def _bucket(instrument: BasketInstrument, disc, index, vol, zeta_before, dt, *, reversion, solver: str):
    """One bucket, the earlier ones fixed (they reach it as `zeta_before`, zeta at the previous
    helper's expiry): its volatility, the root on `SIGMA_BRACKET` of the helper's model value
    less its market value, found from the helper's market volatility. Returns `(sigma, market,
    model, hit_ceiling, zeta at this expiry)`, each over the batch: the curves' batch axes
    broadcast with the volatility's (either may be unbatched). The root carries no derivative:
    a calibration's is the whole basket's (`calibrated_values`)."""
    dtype = curve_dtype(disc)
    # `price_pair` with what does not depend on the bucket's volatility taken out of the
    # solve: the helper's legs, its market value, strike and type.
    legs = _legs(instrument, disc, index)
    batch = jnp.broadcast_shapes(jnp.shape(vol), jnp.shape(zeta_before), legs.annuity.shape)
    vol, base = jnp.broadcast_to(vol, batch), jnp.broadcast_to(zeta_before, batch)
    market, strike, receiver = _market(instrument, legs, vol)

    def model_value(zeta):
        return _analytic_lgm(instrument, legs, reversion, zeta, strike, receiver, solver)

    def residual(sigma, params):
        base, market = params
        return model_value(base + sigma ** 2 * dt) - market

    sigma = solve(residual, (base, market), vol, solver=solver, steps=SIGMA_STEPS, increasing=True,
                  bracket=SIGMA_BRACKET)
    zeta = base + sigma ** 2 * dt
    hit = sigma >= SIGMA_BRACKET[1] * (1.0 - ceiling_tolerance(dtype))
    return sigma, market, model_value(zeta), hit, zeta


def helper_residuals(basket: Sequence[BasketInstrument], disc, index, vols, zeta, reversion: float,
                     solver: str = DEFAULT_SOLVER) -> jax.Array:
    """`[n]`: each helper's model value less its market value on the LGM whose zeta at helper j's
    expiry is `zeta[j]`. Elementwise in zeta: helper j reads its own expiry's alone, so the
    bootstrap's buckets, `zeta(expiry_j) = sum_{i<=j} values_i^2 dt_i`, make each helper's
    equation a root in its own zeta. The helpers are one `HelperStack`, priced by one trace mapped
    over them: a derivative of the residuals traces a helper's price (its y* root included) once,
    not once per helper (an American's 16 helpers' second derivatives compiled for 33 s traced one
    by one; 2026-10-19)."""
    def residual(helper, vol, z):
        market, model = price_pair(helper, disc, index, vol, reversion, z, solver)
        return model - market

    return jax.vmap(residual, in_axes=(0, -1, -1), out_axes=-1)(stack_helpers(basket), vols, zeta)


def calibrated_values(basket: Sequence[BasketInstrument], disc, index, vols, dt, reversion: float,
                      solver: str = DEFAULT_SOLVER) -> jax.Array:
    """`[n]`: the bootstrap's bucket values on one market (`bootstrap`), with the implicit
    function theorem's derivative in the curves and the volatilities: the calibration's
    response to them, of any order. What a price function differentiated through its
    calibration reads (`engine.risk.greeks.price_functions`): exact for an accepted calibration,
    every bucket a root inside the bracket (one at an edge misses ORE's acceptance and is
    refused).

    Each helper's equation is a root in its own zeta (`helper_residuals`), so zeta's tangent is
    elementwise (`engine.solvers.roots.elementwise_root_tangent`: one forward pass along ones in
    zeta, one in the market), and the buckets' follows from `zeta_j = sum_{i<=j} values_i^2 dt_i`:
    `d values_i = (d zeta_i - d zeta_{i-1}) / (2 values_i dt_i)`. The bootstrap's own steps are never
    differentiated. Against the triangular system in the buckets themselves (a Jacobian of a
    forward pass per bucket and a triangular solve, 2026-10-17), an American's second derivatives
    traced each helper's price in fewer derivative contexts (2026-10-19)."""
    dt = jnp.asarray(dt, dtype=jnp.result_type(vols))

    @jax.custom_jvp
    def buckets(market):
        return jax.lax.stop_gradient(bootstrap(basket, *market, dt, reversion, solver)[0])

    @buckets.defjvp
    def buckets_jvp(primals, tangents):
        (market,), (market_dot,) = primals, tangents
        values = buckets(market)
        residual = lambda zeta, m: helper_residuals(basket, *m, zeta, reversion, solver)  # noqa: E731
        zeta_dot = elementwise_root_tangent(residual, jnp.cumsum(jnp.square(values) * dt, axis=-1), market,
                                            market_dot)
        moved = zeta_dot - jnp.concatenate([jnp.zeros_like(zeta_dot[..., :1]), zeta_dot[..., :-1]], axis=-1)
        return values, moved / (2.0 * values * dt)

    return buckets((disc, index, vols))
