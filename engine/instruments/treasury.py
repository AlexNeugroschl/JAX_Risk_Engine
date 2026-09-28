"""
Treasury bills and notes as an instrument type for `engine.portfolio`.

A `BondConfig` carries its own `ZeroCurveConfig` (like `SwaptionConfig`, unlike
`SwapConfig`'s curve indexes), so there is no curve index to resolve.

Pricing: every remaining cashflow discounted off that curve, linear zero-rate
interpolation, flat extrapolation, continuous compounding over ACT/365 from the evaluation
date. NPV is dirty. This restates the convention of `engine.integration.bill`/`note`
rather than importing it (integration sits above instruments); the two are pinned
together by `TestAgreesWithTheIntegrationPricers`.

Differs from ORE: this is not ORE's bond engine (`DiscountingRiskyBondEngine`). There is no
settlement lag, security spread or credit curve, and the coupon schedule is supplied, not
generated.

No scenario NPV (I-24): a bond has no stochastic driver here, so its `npv_cube` column
would be one t=0 number broadcast across scenarios, giving VaR = 0 and ES = NaN
(`tests/test_treasury_instrument.py::TestScenarioPricingIsRefused`). `price_bond_scenarios`
raises instead. Bonds reach `base_npv`, `base_npv_per_trade` and Greeks; they do not reach
`npv_cube`.
"""
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import ORE

from engine.day_count import UnsupportedDayCountError, resolve_accrual_day_count
from engine.simulation.market_model import ZeroCurveConfig

#: Discounting day count (ACT/365, the simulation time axis). Distinct from the bond's
#: accrual day count.
DISCOUNT_DAY_COUNT = ORE.Actual365Fixed()

#: Per-pillar bump for `rate_sensitivity`, in absolute rate terms (1bp).
RATE_BUMP = 1e-4


class BondPricingError(Exception):
    """A bond could not be priced. Separate from `ValueError` so pricing refusals are
    distinguishable from malformed requests."""


class ScenarioPricingNotSupported(BondPricingError):
    """A `BondConfig` reached the scenario (`npv_cube`) path, which bonds do not support
    (I-24; see the module docstring)."""


@dataclass(frozen=True)
class CouponPeriod:
    """One coupon period, supplied rather than generated (a regenerated schedule that
    disagreed with the booked one would reprice every coupon)."""
    start_date: ORE.Date
    end_date: ORE.Date
    payment_date: Optional[ORE.Date] = None

    def payment(self) -> ORE.Date:
        return self.payment_date if self.payment_date is not None else self.end_date


@dataclass(frozen=True)
class BondConfig:
    """A Treasury bill or note, priced off its own curve.

    A bill is `coupon_schedule=()` with `coupon_rate=0.0`.

    `face_amount` is signed: a short position has a negative face and a negative NPV. Do
    not apply a separate sign on top.

    `initial_zero_curve` is this bond's own curve, not an index into the simulation's
    curves.
    """
    face_amount: float
    maturity_date: ORE.Date
    evaluation_date: ORE.Date
    initial_zero_curve: ZeroCurveConfig
    #: Annual coupon rate as a decimal (0.04 == 4%), not a percent.
    coupon_rate: float = 0.0
    coupon_schedule: Tuple[CouponPeriod, ...] = ()
    redemption_fraction: float = 1.0
    #: Coupon accrual day count; refused if unsupported. Ignored without coupons.
    accrual_day_count: str = "ACT/ACT (ICMA)"
    #: Index of this bond's curve in a market-risk run's curve list
    #: (`engine.market_risk.RateRiskFactors.curves`); `initial_zero_curve` must then equal
    #: that curve. `None` elsewhere.
    curve_index: Optional[int] = None

    def __post_init__(self):
        if self.maturity_date <= self.evaluation_date:
            raise BondPricingError(
                f"maturity_date {_iso(self.maturity_date)} is not after "
                f"evaluation_date {_iso(self.evaluation_date)}. A matured bond has "
                f"no remaining cashflow to discount; its value is a settlement "
                f"question, not a pricing one."
            )
        if self.redemption_fraction < 0:
            raise BondPricingError(
                f"redemption_fraction={self.redemption_fraction} is negative"
            )
        if self.coupon_schedule and self.coupon_rate == 0.0:
            raise BondPricingError(
                "coupon_schedule is non-empty but coupon_rate is 0.0. A schedule of "
                "zero coupons is a contradiction: either the instrument is a bill "
                "(empty schedule) or it pays a coupon. Refused rather than priced "
                "as a bill, because which of the two was meant changes the price."
            )
        if self.coupon_rate != 0.0 and not self.coupon_schedule:
            raise BondPricingError(
                f"coupon_rate={self.coupon_rate} is non-zero but no coupon_schedule "
                f"was supplied. A schedule is not derived from a rate here -- a "
                f"generated schedule that disagreed with the booked one would "
                f"silently reprice every coupon."
            )
        if self.coupon_schedule:
            # Resolve now so an unsupported convention fails at construction.
            try:
                resolve_accrual_day_count(self.accrual_day_count)
            except UnsupportedDayCountError as exc:
                raise BondPricingError(str(exc)) from None
            _validate_schedule(self.coupon_schedule, self.maturity_date)

    @property
    def is_bill(self) -> bool:
        """Whether this is the zero-coupon single-cashflow case."""
        return not self.coupon_schedule

    @property
    def notional(self) -> float:
        """Alias for `face_amount`, for the shared trade-labelling helpers in
        `engine.portfolio.request`."""
        return self.face_amount


def _validate_schedule(schedule: Sequence[CouponPeriod], maturity: ORE.Date) -> None:
    """Structure and contiguity of an explicit coupon schedule, as
    `engine.integration.note._parse_schedule` checks it. Gaps and overlaps are refused."""
    for i, period in enumerate(schedule):
        if period.end_date <= period.start_date:
            raise BondPricingError(
                f"coupon_schedule[{i}] ends {_iso(period.end_date)} on or before it "
                f"starts {_iso(period.start_date)}; a period with no length cannot accrue."
            )
        if period.payment() < period.end_date:
            raise BondPricingError(
                f"coupon_schedule[{i}] pays {_iso(period.payment())} before its "
                f"accrual ends {_iso(period.end_date)}."
            )
    for i in range(1, len(schedule)):
        if schedule[i].start_date != schedule[i - 1].end_date:
            raise BondPricingError(
                f"coupon_schedule[{i}] starts {_iso(schedule[i].start_date)} but "
                f"coupon_schedule[{i-1}] ended {_iso(schedule[i-1].end_date)}. A gap "
                f"or overlap between coupon periods means the schedule does not "
                f"describe one continuous instrument, and is refused rather than bridged."
            )
    if schedule and schedule[-1].end_date != maturity:
        raise BondPricingError(
            f"the coupon schedule ends {_iso(schedule[-1].end_date)} but maturity_date "
            f"is {_iso(maturity)}. The redemption and the final coupon must fall on the "
            f"same date; a disagreement means the two describe different instruments."
        )


def _iso(date: ORE.Date) -> str:
    return f"{date.year():04d}-{date.month():02d}-{date.dayOfMonth():02d}"


def _zero_rate_at(curve: ZeroCurveConfig, t: float) -> float:
    """Zero rate at `t`: linear between pillars, flat beyond the ends."""
    times = list(curve.times)
    rates = list(curve.rates)
    if not times:
        raise BondPricingError("initial_zero_curve has no pillars")
    if t <= times[0]:
        return rates[0]
    if t >= times[-1]:
        return rates[-1]
    for i in range(1, len(times)):
        if t <= times[i]:
            span = times[i] - times[i - 1]
            if span == 0:
                return rates[i]
            w = (t - times[i - 1]) / span
            return rates[i - 1] + w * (rates[i] - rates[i - 1])
    return rates[-1]


def _discount_factor(curve: ZeroCurveConfig, valuation: ORE.Date, date: ORE.Date,
                     rate_shift: float = 0.0) -> Tuple[float, float]:
    """`(discount_factor, year_fraction)` for one date: continuously compounded over an
    ACT/365 year fraction."""
    year_fraction = DISCOUNT_DAY_COUNT.yearFraction(valuation, date)
    zero_rate = _zero_rate_at(curve, year_fraction) + rate_shift
    return math.exp(-zero_rate * year_fraction), year_fraction


def accrued_interest(cfg: BondConfig) -> float:
    """Accrued interest in currency (position-signed) from the coupon schedule; 0 for a
    bill or before the first period.

    This is the recomputed-schedule figure only. The integration boundary also reconciles
    against the exporter's published accrued fraction (`engine.integration.note`); a
    direct caller has no exporter to reconcile against.
    """
    if not cfg.coupon_schedule:
        return 0.0
    day_count = resolve_accrual_day_count(cfg.accrual_day_count)
    for period in cfg.coupon_schedule:
        if period.start_date <= cfg.evaluation_date < period.end_date:
            fraction = cfg.coupon_rate * day_count.yearFraction(
                period.start_date, cfg.evaluation_date,
                period.start_date, period.end_date,
            )
            return fraction * cfg.face_amount
    return 0.0


def _remaining_cashflows(cfg: BondConfig) -> Tuple[Tuple[ORE.Date, float], ...]:
    """Remaining cashflows as `(payment_date, amount per unit face)`: coupons in schedule
    order, then the redemption. A coupon paid on or before the evaluation date is
    excluded. Shared by `price_bond_base` and `bond_price_function`."""
    flows = []
    if cfg.coupon_schedule:
        day_count = resolve_accrual_day_count(cfg.accrual_day_count)
        for period in cfg.coupon_schedule:
            payment = period.payment()
            if payment <= cfg.evaluation_date:
                continue
            accrual_fraction = day_count.yearFraction(
                period.start_date, period.end_date,
                period.start_date, period.end_date,
            )
            flows.append((payment, cfg.coupon_rate * accrual_fraction))
    flows.append((cfg.maturity_date, cfg.redemption_fraction))
    return tuple(flows)


def price_bond_base(cfg: BondConfig, rate_shift: float = 0.0) -> float:
    """t=0 dirty NPV: every remaining cashflow, discounted. `clean_npv_of` subtracts
    accrued interest. `rate_shift` shifts the whole curve in parallel."""
    total_per_unit_face = 0.0
    for payment, amount in _remaining_cashflows(cfg):
        df, _ = _discount_factor(cfg.initial_zero_curve, cfg.evaluation_date, payment, rate_shift)
        total_per_unit_face += amount * df
    return cfg.face_amount * total_per_unit_face


def bond_price_function(cfg: BondConfig):
    """`f(pillar_rates) -> t=0 dirty NPV` in JAX, for shocked-curve revaluation
    (`engine.market_risk`) and per-pillar differentiation.

    Same cashflows and convention as `price_bond_base`, with the pillar rates traced:
    `f(initial_zero_curve.rates) == price_bond_base(cfg)`. Works in `pillar_rates`' dtype.
    """
    import jax.numpy as jnp

    flows = _remaining_cashflows(cfg)
    times = np.asarray(
        [DISCOUNT_DAY_COUNT.yearFraction(cfg.evaluation_date, payment) for payment, _ in flows],
        dtype=np.float64,
    )
    amounts = np.asarray([amount for _, amount in flows], dtype=np.float64) * cfg.face_amount
    pillar_times = np.asarray(cfg.initial_zero_curve.times, dtype=np.float64)
    if pillar_times.size == 0:
        raise BondPricingError("initial_zero_curve has no pillars")

    def price(pillar_rates):
        dtype = jnp.asarray(pillar_rates).dtype
        t = jnp.asarray(times, dtype=dtype)
        zero = jnp.interp(t, jnp.asarray(pillar_times, dtype=dtype), pillar_rates)
        return jnp.sum(jnp.asarray(amounts, dtype=dtype) * jnp.exp(-zero * t))

    return price


def clean_npv_of(cfg: BondConfig) -> float:
    """`price_bond_base(cfg) - accrued_interest(cfg)`, for comparison with a quoted
    (clean) price."""
    return price_bond_base(cfg) - accrued_interest(cfg)


def rate_sensitivity(cfg: BondConfig, bump: float = RATE_BUMP) -> float:
    """Change in dirty NPV for a parallel `bump` of the zero curve, by bumped
    revaluation. Parallel only (I-16)."""
    return price_bond_base(cfg, rate_shift=bump) - price_bond_base(cfg)


def price_bond_scenarios(*_args, **_kwargs):
    """Always raises: a bond has no scenario NPV (I-24; see the module docstring)."""
    raise ScenarioPricingNotSupported(
        "a BondConfig has no scenario NPV: it is closed-form arithmetic against a "
        "single deterministic curve, with no stochastic driver and no time "
        "evolution. Filling an [Scenarios, TimeSteps] column would mean "
        "broadcasting one t=0 number across every entry, producing a zero-variance "
        "column whose VaR is 0.00 and whose ES is NaN -- a position reported as "
        "risk-measured when its risk was never modelled. Bonds reach base_npv and "
        "the Greeks path, which are real; npv_cube/VaR needs a bond scenario model "
        "(I-24)."
    )
