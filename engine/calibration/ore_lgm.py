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
    the strike is at or below the forward).
  * Model value (`price_pair`, with the market value): `AnalyticLgmSwaptionEngine` with ORE's default
    `FloatSpreadMapping::proRata`: the swaption is priced on the discount curve alone, and the
    difference between each floating coupon and its discount-curve "flat" amount is moved
    onto the fixed coupons (`S_j`) and the start (`S_m1`).
  * Bootstrap (`bootstrap_sigma`): `calibrateVolatilitiesIterative` -- one piecewise-constant
    bucket per helper, ending at the helper's expiry (the last open), each solved so the model
    matches the market with the earlier buckets fixed. ORE minimises the relative price error
    with Levenberg-Marquardt; here each bucket is bisected on [1e-6, 0.2] (plan X-7), the same
    root where one exists.

Every pricing function takes discount and index curves of either type (`ZeroCurve` today,
`DiscountCurve` on a path) and broadcasts over their leading batch axes, so one call
calibrates every path at once.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
import ORE
from jax.scipy.stats import norm

from engine.market import (
    VOL_BUSINESS_DAY_CONVENTION, VOL_CALENDAR, SwaptionVolSurface, swap_length, swap_length_between,
)
from engine.models.curves import discount
from engine.models.lgm import H as lgm_H
from engine.models.ore_builders import (
    TIME_AXIS_DAY_COUNTER, ibor_index, par_coupon_forecast_period, resolve_accrual_day_count,
)
from engine.models.static_key import StaticKeyMixin

#: `IrModelBuilder::maxAtmStdDev`: a helper strike further from ATM is moved to this many ATM
#: standard deviations (fallback rule 1).
MAX_ATM_STD_DEV = 3.0

#: Bisection bracket for one bucket's volatility, as `engine.calibration.lgm`.
SIGMA_BRACKET = (1e-6, 0.20)


@dataclass(frozen=True)
class SwapIndexConventions:
    """The swap index a currency's helpers are built on (ORE's `swapIndexBase`): its fixed
    leg tenor and day counter, and the tenor of its Ibor index (`ibor_index`). ORE switches to
    a short swap index below its tenor; one index serves every term here."""
    fixed_tenor: str = "1Y"
    fixed_day_counter: str = "ACT/365"
    index_tenor_months: int = 6


@dataclass(frozen=True, eq=False)
class BasketInstrument(StaticKeyMixin):
    """One `SwaptionHelper` on a reference date: times are ACT/365 from it (the model's and
    the helper's pricing time axis).

    `vol_option_time`/`vol_swap_length` are where ORE reads its volatility (for a tenor
    expiry the vol surface's own option date, which can differ from the helper's exercise
    date). Floating coupons carry the owner fixed coupon and weight of ORE's proRata mapping
    (`owner = -1`: beyond the fixed schedule, ignored as ORE's loop ignores it)."""
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
    read separately (`basket_vols`), since on a path they depend on the date."""
    if len(expiries) != len(terms):
        raise ValueError("expiries and terms must have the same length")
    deal_strikes = list(deal_strikes) if deal_strikes is not None else [None] * len(expiries)
    t_of = lambda d: TIME_AXIS_DAY_COUNTER.yearFraction(reference, d)  # noqa: E731
    basket = []
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
    strike = jnp.clip(instrument.deal_strike, legs.forward - band, legs.forward + band)
    return strike, strike <= legs.forward


def _market(instrument: BasketInstrument, legs: _Legs, vol):
    """(market value, strike, is_receiver): Bachelier on the ATM forward and annuity, with
    `vol` the normal volatility ORE reads for the helper."""
    std_dev = vol * jnp.sqrt(instrument.expiry_time)
    strike, receiver = _strike_and_receiver(instrument, legs, std_dev)
    w = jnp.where(receiver, -1.0, 1.0)
    d = (legs.forward - strike) / std_dev
    value = legs.annuity * (w * (legs.forward - strike) * norm.cdf(w * d) + std_dev * norm.pdf(d))
    return value, strike, receiver


def market_price(instrument: BasketInstrument, disc, index, vol) -> jax.Array:
    """The helper's market value (nominal 1)."""
    return _market(instrument, _legs(instrument, disc, index), vol)[0]


def price_pair(instrument: BasketInstrument, disc, index, vol, reversion: float, zeta_expiry):
    """(market value, model value) of one helper, nominal 1, on the same strike and type.
    The model is an LGM with constant `reversion` and zeta(expiry) = `zeta_expiry`."""
    legs = _legs(instrument, disc, index)
    market, strike, receiver = _market(instrument, legs, vol)
    return market, _analytic_lgm(instrument, legs, reversion, zeta_expiry, strike, receiver)


def _corrections(instrument: BasketInstrument, legs: _Legs):
    """proRata `S_j` per fixed coupon and `S_m1` at the start (`AnalyticLgmSwaptionEngine::
    calculate`, nominal 1)."""
    correction = (legs.float_amount - legs.flat_amount) * legs.float_df
    owned = instrument.owner >= 0
    owner = np.where(owned, instrument.owner, 0)
    lambda2 = np.where(owned, instrument.lambda2, 0.0)
    lambda1 = np.where(owned, 1.0 - instrument.lambda2, 0.0)
    n_fixed = instrument.fixed_pay.size
    to_fixed = np.eye(n_fixed)[owner] * owned[:, None]          # [Nc, Nf]
    sum1 = (correction * lambda1) @ to_fixed                    # [..., Nf]
    sum2 = (correction * lambda2) @ to_fixed
    # S_j = sum2_j / D_j + sum1_{j+1} / D_j; S_m1 = sum1_0 / D0.
    shifted = jnp.concatenate([sum1[..., 1:], jnp.zeros_like(sum1[..., :1])], axis=-1)
    S = (sum2 + shifted) / legs.fixed_df
    S_m1 = sum1[..., 0] / legs.d0
    return S, S_m1


def _analytic_lgm(instrument, legs: _Legs, reversion, zeta_expiry, strike, receiver):
    """`AnalyticLgmSwaptionEngine::calculate` (j1 = k1 = 0: a helper starts after expiry),
    in closed form around the root y* of the exercise boundary (`yStarHelper`)."""
    S, S_m1 = _corrections(instrument, legs)
    amounts = jnp.asarray(instrument.fixed_accrual) * strike[..., None] - S      # [..., Nf]
    H0 = lgm_H(reversion, jnp.asarray(instrument.float_start[0]))
    dH = lgm_H(reversion, jnp.asarray(instrument.fixed_pay)) - H0                # [Nf]
    zeta = jnp.asarray(zeta_expiry)[..., None]
    D = legs.fixed_df

    def boundary(y):  # yStarHelper: the underlying at expiry as a function of y
        y = y[..., None]
        bonds = jnp.exp(-dH * y - 0.5 * dH ** 2 * zeta)
        return (jnp.sum(amounts * D * bonds, axis=-1) - S_m1 * legs.d0
                + D[..., -1] * bonds[..., -1] - legs.d0)

    y_star = _solve_monotone_root(boundary, legs.d0.shape)
    sqrt_zeta = jnp.sqrt(zeta[..., 0])
    w = jnp.where(receiver, 1.0, -1.0)   # ORE: Call (payer) -> -1, Put (receiver) -> +1
    Phi = lambda arg: norm.cdf(w * arg / sqrt_zeta)  # noqa: E731
    total = jnp.sum(amounts * D * norm.cdf(w[..., None] * (y_star[..., None] + dH * zeta) / sqrt_zeta[..., None]),
                    axis=-1)
    total = total - S_m1 * legs.d0 * Phi(y_star)
    total = total + D[..., -1] * Phi(y_star + dH[-1] * zeta[..., 0]) - legs.d0 * Phi(y_star)
    return w * total


def _solve_monotone_root(f, shape, iterations: int = 100) -> jax.Array:
    """Root of a function decreasing in y, elementwise over `shape`: the bracket starts at
    [-1, 1] and doubles until it holds the root, then bisects. No derivative is propagated
    (the price is stationary in y* at the root, so its derivatives with y* held fixed are
    exact)."""
    lo, hi = -jnp.ones(shape), jnp.ones(shape)

    def widen(_, bounds):
        lo, hi = bounds
        return jnp.where(f(lo) < 0.0, 2.0 * lo, lo), jnp.where(f(hi) > 0.0, 2.0 * hi, hi)

    lo, hi = jax.lax.fori_loop(0, 60, widen, (lo, hi))

    def halve(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        above = f(mid) > 0.0
        return jnp.where(above, mid, lo), jnp.where(above, hi, mid)

    lo, hi = jax.lax.fori_loop(0, iterations, halve, (lo, hi))
    return jax.lax.stop_gradient(0.5 * (lo + hi))


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
@dataclass
class BootstrapResult:
    """Bucket times (the helpers' expiries but the last) and values `[..., n]`, with each
    helper's market and model value at the result. `hit_ceiling` flags a bucket that could
    not reach its market price inside the bracket."""
    times: np.ndarray
    values: jax.Array
    market: jax.Array
    model: jax.Array
    hit_ceiling: jax.Array


def bootstrap_sigma(basket: Sequence[BasketInstrument], disc, index, vols, reversion: float,
                    iterations: int = 60) -> BootstrapResult:
    """`calibrateVolatilitiesIterative` over `basket` (ascending expiries): bucket i covers
    [expiry_{i-1}, expiry_i) (the last is open), and zeta(expiry_i) = zeta(expiry_{i-1}) +
    sigma_i^2 (expiry_i - expiry_{i-1}). `vols` is `[..., n]` (or `[n]`), matched to the
    curves' batch axes."""
    expiries = np.array([b.expiry_time for b in basket])
    if np.any(np.diff(expiries) <= 0.0):
        raise ValueError("basket expiries must increase strictly")
    # One calibration per path: the batch shape is the curves' batch axes broadcast with the
    # vols' (either may be unbatched).
    batch = jnp.broadcast_shapes(jnp.shape(vols)[:-1], _legs(basket[0], disc, index).annuity.shape)
    vols = jnp.broadcast_to(jnp.asarray(vols), batch + (len(basket),))
    lo, hi = SIGMA_BRACKET
    values, markets, models, ceiling = [], [], [], []
    zeta_before = None
    for i, instrument in enumerate(basket):
        dt = expiries[i] - (expiries[i - 1] if i > 0 else 0.0)
        vol_i = vols[..., i]
        base = jnp.zeros_like(vol_i) if zeta_before is None else zeta_before

        def residual(sigma, _instrument=instrument, _vol=vol_i, _base=base, _dt=dt):
            market, model = price_pair(_instrument, disc, index, _vol, reversion, _base + sigma ** 2 * _dt)
            return model - market

        def halve(bounds, _):
            a, b = bounds
            mid = 0.5 * (a + b)
            below = residual(mid) < 0.0
            return (jnp.where(below, mid, a), jnp.where(below, b, mid)), None

        start = (jnp.full(vol_i.shape, lo), jnp.full(vol_i.shape, hi))
        (a, b), _ = jax.lax.scan(halve, start, None, length=iterations)
        sigma = 0.5 * (a + b)
        zeta_before = base + sigma ** 2 * dt
        market, model = price_pair(instrument, disc, index, vol_i, reversion, zeta_before)
        values.append(sigma)
        markets.append(market)
        models.append(model)
        ceiling.append(sigma >= hi * (1.0 - 1e-9))
    stack = lambda xs: jnp.stack(xs, axis=-1)  # noqa: E731
    return BootstrapResult(times=expiries[:-1], values=stack(values), market=stack(markets),
                           model=stack(models), hit_ceiling=stack(ceiling))
