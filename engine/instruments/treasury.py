"""
W1.5 -- Treasury bills and notes as a first-class instrument type for
`engine.portfolio`.

Both Treasury pricers were built at the integration boundary
(`engine.integration.bill`/`note`, W1.2/W1.3) because that is where the
TraderX bundle arrives. That left a real gap, recorded as the wire-through
half of **I-07**: a direct Python caller of `price_portfolio` had no bond
pricer at all, because `engine/instruments/` was still exactly four
rate-derivative modules. This module closes that gap.

`BondConfig` carries **its own `ZeroCurveConfig`**, in the same shape as
`SwaptionConfig` and unlike `SwapConfig`'s curve *indexes*. That is the
whole reason a bond cannot reproduce **I-01**: there is no index to resolve,
so there is no resolution step to forget, and `_compute_all_greeks` cannot
silently skip one for want of a `SimulationConfig` it was never passed.

---

**The scenario dimension is refused, not broadcast. Read this before
adding one.**

`price_portfolio`'s `npv_cube` is `[Scenarios, TimeSteps, Trades]`: each
column is a trade's *conditional* NPV at each simulated future step, and
`engine.risk.var_es` turns those columns into VaR and ES. Every existing
instrument type fills its column from simulated Hull-White paths.

A bond priced here has **no such column to fill**. It is closed-form
arithmetic against one curve -- there is no stochastic driver and no time
evolution, so the only thing that *could* be written into a
`[Scenarios, TimeSteps]` slab is one t=0 number broadcast across every
entry. That is precisely the silent approximation this codebase refuses,
and it fails in a way that looks measured:

    a zero-variance column produces VaR = 0.00 and ES = NaN

-- confirmed directly, not reasoned about (see
`tests/test_treasury_instrument.py::TestScenarioPricingIsRefused`). A
consumer reading `VaR_95 = 0.00` for a $100k bill would conclude the
position carries no risk, when in fact **its risk was never modelled**. The
number is not conservative, not approximate, and not labelled -- it is
absent, wearing the shape of a measurement.

So `price_bond_scenarios` does not exist, and `_price_by_type` **raises**
rather than inventing a column. `price_bond_base` (t=0, deterministic) is
the real, honest thing this module offers, and it is what
`_base_npv_per_trade` and the Greeks path consume. A bond reaches a
portfolio's `base_npv` and `base_npv_per_trade`; it does not reach
`npv_cube`, and `price_portfolio` says so by name instead of quietly
returning a zero.

**Closing this needs a bond scenario model** -- rate paths repriced through
the bond's own schedule -- which is genuine modelling work, not plumbing.
Registered as **I-24**.

---

**Pricing math is not reimplemented here.** The discounting convention
(ACT/365 Fixed, continuously compounded) and the accrual treatment are the
same ones `engine.integration.bill`/`note` already state and test. This
module deliberately imports **neither**: `engine.integration` sits *above*
`engine.instruments` in the dependency order (integration imports
instruments, never the reverse), and reversing that to share ~20 lines of
`exp(-r*t)` would couple the instrument layer to the TraderX bundle format.
The duplication is small, fully tested, and pinned against the integration
pricers by `TestAgreesWithTheIntegrationPricers`, which prices the *same*
instrument through both paths and asserts they agree to the cent -- so a
drift between them fails a test rather than going unnoticed.
"""
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import ORE

from engine.day_count import UnsupportedDayCountError, resolve_accrual_day_count
from engine.simulation.market_model import ZeroCurveConfig

#: Discounting day count for this instrument -- ACT/365 Fixed, matching the
#: engine's simulation time axis and `engine.integration.bill`'s stated
#: boundary policy. This is the *discounting* convention and is distinct
#: from the instrument's own *accrual* convention (the W1.1 split).
DISCOUNT_DAY_COUNT = ORE.Actual365Fixed()

#: Per-pillar bump for `rate_sensitivity`, in absolute rate terms (1bp).
RATE_BUMP = 1e-4


class BondPricingError(Exception):
    """A bond could not be priced. Distinct from `ValueError` so a caller
    can tell a *pricing* refusal from a malformed-request error."""


class ScenarioPricingNotSupported(BondPricingError):
    """Raised when a `BondConfig` reaches the scenario/`npv_cube` path.

    Deliberately its own type, and deliberately raised rather than handled
    by returning a broadcast constant -- see the module docstring. A caller
    that wants a bond's t=0 value should use `compute_greeks`/`base_npv`,
    both of which are real; a caller that wants its VaR needs a bond
    scenario model, which does not exist yet (**I-24**).
    """


@dataclass(frozen=True)
class CouponPeriod:
    """One explicit coupon period.

    Supplied rather than generated, for the same reason
    `engine.integration.note` refuses to regenerate a schedule: a schedule
    derived by stepping back from maturity that disagreed with the booked
    one would silently reprice every coupon.
    """
    start_date: ORE.Date
    end_date: ORE.Date
    payment_date: Optional[ORE.Date] = None

    def payment(self) -> ORE.Date:
        return self.payment_date if self.payment_date is not None else self.end_date


@dataclass(frozen=True)
class BondConfig:
    """A Treasury bill or note, priced off its own curve.

    **A bill is `coupon_schedule=()` with `coupon_rate=0.0`** -- the same
    single-cashflow instrument `engine.integration.bill` prices, expressed
    as the degenerate case of this type rather than as a separate class.
    One type keeps `_price_by_type`/`_base_npv_per_trade`/
    `_compute_all_greeks` to one branch each; two would double every
    routing site for an instrument that differs only by having no coupons.

    `face_amount` is **signed**: a short position is a negative face and
    returns a negative NPV in one step. There is no separate sign factor to
    apply, and applying one on top would flip a short position positive --
    the double-sign bug TraderX flagged in their v3 §2.

    `initial_zero_curve` is this bond's own curve, not an index into
    `SimulationConfig.rates.initial_zero_curves`. See the module docstring:
    that choice is what makes **I-01** structurally unreachable here.
    """
    face_amount: float
    maturity_date: ORE.Date
    evaluation_date: ORE.Date
    initial_zero_curve: ZeroCurveConfig
    #: Annual coupon rate as a DECIMAL (0.04 == 4%), not a percent. The
    #: integration boundary converts `couponRatePercent` before
    #: constructing this; stating the unit here keeps the two from drifting.
    coupon_rate: float = 0.0
    coupon_schedule: Tuple[CouponPeriod, ...] = ()
    redemption_fraction: float = 1.0
    #: The instrument's own ACCRUAL day count (W1.1), resolved and refused
    #: if unsupported -- never defaulted. Ignored when there are no coupons.
    accrual_day_count: str = "ACT/ACT (ICMA)"
    #: Index of the market curve this bond discounts on, in the curve list a
    #: market-risk run shocks (`engine.market_risk.RateRiskFactors.curves`).
    #: Needed only there, where every trade must say which shocked curve to
    #: revalue against; `initial_zero_curve` must then equal that curve.
    #: `None` everywhere else.
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
            # Resolve eagerly so an unsupported convention fails at
            # CONSTRUCTION, naming the trade, rather than mid-pricing.
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
        """Alias for `face_amount`.

        Exists so the shared validation/labelling helpers in
        `engine.portfolio.request` -- which format every trade as
        `trade[i] (Type, notional=...)` -- work for a bond without a
        special case.
        """
        return self.face_amount


def _validate_schedule(schedule: Sequence[CouponPeriod], maturity: ORE.Date) -> None:
    """Structure and contiguity of an explicit coupon schedule.

    Mirrors `engine.integration.note._parse_schedule`'s checks. A gap or
    overlap means the periods do not describe one continuous instrument and
    is refused rather than bridged.
    """
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
    """The curve's zero rate at time `t`, linearly interpolated between
    pillars and held flat beyond the ends.

    Flat extrapolation is stated rather than assumed: extrapolating a slope
    past the last pillar produces a confident number from no data, and for
    a long bond that error compounds through every discount factor.
    """
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
    """`(discount_factor, year_fraction)` for one date off `curve`.

    Continuously compounded over an ACT/365 year fraction, matching
    `engine.integration.bill`/`note` exactly -- see the module docstring on
    why that convention is restated here rather than imported.
    """
    year_fraction = DISCOUNT_DAY_COUNT.yearFraction(valuation, date)
    zero_rate = _zero_rate_at(curve, year_fraction) + rate_shift
    return math.exp(-zero_rate * year_fraction), year_fraction


def accrued_interest(cfg: BondConfig) -> float:
    """Accrued interest in currency, recomputed from the coupon schedule.

    Position-signed, via `face_amount`. Returns 0.0 for a bill (no coupons
    to accrue) and 0.0 before the first period starts -- both are structural
    facts, not missing values.

    **This is the `recomputed-schedule` path only.** The integration
    boundary additionally reconciles against the exporter's published
    `accruedInterestFraction` and reports *that* one
    (`engine.integration.note`'s two-path rule). A direct Python caller has
    no exporter and therefore no second path, so there is nothing to
    reconcile against and no label to disambiguate.
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
    """Every cashflow still to be paid, as `(payment_date, amount per unit
    face)`, coupons in schedule order and then the redemption.

    The single definition of the bond's cashflows: `price_bond_base` and
    `bond_price_function` both discount exactly this list, so the float and
    the JAX pricer cannot disagree about what is being paid. A coupon paid
    on or before the evaluation date is excluded -- it is not this
    position's cashflow any more, and including it would double-count what
    the accrued figure excludes.
    """
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
    """t=0 **dirty** NPV of one bond: every remaining cashflow, discounted.

    Dirty (full) rather than clean, matching `engine.integration.note.
    NotePrice.npv` and for the same reason stated there: a "bond NPV" that
    silently meant the clean value would be off by the accrued interest --
    about $1,857 on a $100k note -- which is large enough to matter and
    small enough to look like a curve difference. `clean_npv_of` is
    available for a caller reconciling against a quoted clean price.

    `rate_shift` parallel-shifts the curve, which is what `rate_sensitivity`
    uses. A coupon already paid on or before the evaluation date is
    excluded, not discounted from the past (see `_remaining_cashflows`).
    """
    total_per_unit_face = 0.0
    for payment, amount in _remaining_cashflows(cfg):
        df, _ = _discount_factor(cfg.initial_zero_curve, cfg.evaluation_date, payment, rate_shift)
        total_per_unit_face += amount * df
    return cfg.face_amount * total_per_unit_face


def bond_price_function(cfg: BondConfig):
    """`f(pillar_rates) -> t=0 dirty NPV` as a JAX function, for revaluing
    the bond under shocked curves (`engine.market_risk`) or differentiating
    it per pillar.

    Discounts `_remaining_cashflows` exactly as `price_bond_base` does --
    linear interpolation of zero rates on `cfg.initial_zero_curve`'s pillar
    times, flat beyond the ends, continuous compounding over ACT/365 -- but
    with the pillar RATES as a traced argument. `price_bond_base(cfg)`
    equals `f(initial_zero_curve.rates)`, and a parallel `rate_shift` equals
    `f(rates + shift)`. The working dtype is that of `pillar_rates`.
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
    """`price_bond_base(cfg) - accrued_interest(cfg)`.

    Carried because a quoted bond price is conventionally clean, so a
    consumer reconciling against a market quote compares like with like.
    """
    return price_bond_base(cfg) - accrued_interest(cfg)


def rate_sensitivity(cfg: BondConfig, bump: float = RATE_BUMP) -> float:
    """Change in dirty NPV for a `bump` parallel shift of the zero curve.

    A **bumped revaluation** through the same code path, not a
    differentiated formula -- a sensitivity derived from an expression that
    has drifted from the pricer measures the expression, not the price.
    Parallel-only, matching `engine.integration.note.rate_sensitivity` and
    subject to the same limitation recorded as **I-16**.
    """
    return price_bond_base(cfg, rate_shift=bump) - price_bond_base(cfg)


def price_bond_scenarios(*_args, **_kwargs):
    """**Always raises.** A bond has no scenario NPV at this boundary.

    This function exists so the refusal has a *name and a docstring* at the
    place a contributor would look for the missing capability, rather than
    being an absence they fill in with a broadcast. See the module
    docstring for the measured consequence of filling it in naively:
    VaR = 0.00 and ES = NaN, a number that looks measured and is not.
    """
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
