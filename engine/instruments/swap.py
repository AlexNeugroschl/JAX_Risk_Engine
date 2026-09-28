"""
Vanilla interest rate swap pricing.

Trade structure (schedules, day-count accrual, coupon amounts) is built with
ORE's own `VanillaSwap`/`MakeVanillaSwap` machinery, so date generation and
accrual math match ORE exactly rather than being reimplemented. The resulting
static (date -> year-fraction, accrual, notional) arrays are then evaluated
against the JAX-simulated yield curve cube across every scenario and time
step -- that tensor contraction is the only part that needs to be fast.

Multi-curve: each swap names a `discount_curve_index` and a
`forward_curve_index` into the simulation's `yield_curves` cube (its
`NumRates` axis). Both legs discount off `discount_curve_index`; the floating
leg's forward rates are read off `forward_curve_index`. This mirrors ORE's
`DiscountingSwapEngine` (single discount curve) + `IborIndex` (its own,
possibly different, `forwardingTermStructure`) split.

**Trade dates (audit M-4).** A `SwapConfig` holds its booked
`effective_date`/`maturity_date`; the schedule never depends on the
evaluation date, so the same config is the same trade on every date. On an
evaluation date after the effective date the swap is SEASONED and is priced
as ORE prices it: cashflows paid on or before that date are gone
(`ore_builders.is_live`), and a floating coupon whose fixing date has
passed pays its historical fixing from `SwapConfig.fixings` -- a missing one
raises, as it does in ORE (`ore_builders.known_fixing`). Checked against
`ORE.DiscountingSwapEngine` in tests/test_trade_dates.py.

**Known limitation: no representation of a coupon that fixes DURING the
simulation.** Historical fixings cover coupons fixed before the evaluation
date. At a simulated step past a later fixing, `_price_one_swap` computes
every cashflow's forward rate and discount factor using `yield_curves[scenario, step, ...]`, which represents the model's
conditional discount factor P(step_time, maturity) -- a well-defined
quantity only for maturity >= step_time (see
simulation.reconstruct_yield_curves' B(t,T) clamp at T<t). For a
swap whose accrual has ALREADY STARTED by a given simulated step_time (true
of every step after the swap's own first accrual date -- i.e. every step
after t=0 for a spot-starting swap, which is every existing demo/test
scenario's swap), the floating leg's first coupon's accrual-start date is
in the past relative to that step, and P(step_time, accrual_start) is not
meaningful (it silently returns a clamped, non-discount-factor value rather
than raising). This produces a small but real, previously-undetected NPV
error at every step beyond t=0 for the AGED portion of a swap's floating
leg -- caught via a direct cross-check against ORE at a future evaluation
date (an implied curve rebuilt from the same conditional Hull-White
discount factors) in tests/test_swap.py's
TestAgedSwapKnownLimitation, which pins down the current (imperfect)
behavior as a documented gap rather than a silent one. This does NOT affect
t=0 pricing (every remaining cashflow is in the future there, and every
coupon fixed before it is priced off its historical fixing) or forward-starting
trades priced before their own accrual begins -- both remain exact, as
every other test in this suite demonstrates. Fixing this properly (tracking
already-fixed rates per scenario/step, or excluding elapsed cashflows from
the sum) is intentionally out of scope here and left for a follow-up.

Registered as **I-04** in docs/known-issues.md, which records its full blast
radius (every npv_cube value past first accrual, and therefore every VaR/ES
number derived from it) and the fact that closing it needs historical
published fixings that no current input source supplies -- engine work alone
cannot close it. `engine.portfolio.request._warn_if_aged_swap_exposure` now
warns per affected swap into `PortfolioResult.warnings`, so the gap is
advertised rather than silent; the pricing itself is unchanged.
"""
from dataclasses import InitVar, dataclass, field
from functools import partial
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import ORE

from engine.models.static_key import StaticKeyMixin
from engine.models.ore_builders import (
    DEFAULT_ACCRUAL_DAY_COUNT,
    book_swap_dates,
    build_vanilla_swap,
    fixed_leg_cashflows as _fixed_leg_cashflows,
    floating_leg_cashflows as _floating_leg_cashflows,
    resolve_accrual_day_count,
    validate_fixings,
)
from engine.portfolio.validation import _validate_common_fields


@dataclass
class SwapConfig:
    """
    One vanilla fixed-vs-floating interest rate swap.

    discount_curve_index / forward_curve_index index into the NumRates axis
    of the simulation's yield_curves cube (engine.simulation
    generate_paths' "rates" config -- each index is one Hull-White factor,
    calibrated against its own entry in initial_zero_curves). Both legs
    discount off discount_curve_index; the
    floating leg's forward rates are read off forward_curve_index. Equal
    indices reduce to single-curve discounting; distinct indices reproduce
    ORE's multi-curve DiscountingSwapEngine (discount curve) + IborIndex
    (its own, separate forwardingTermStructure) split.

    effective_date / maturity_date: the booked schedule's start and
    (unadjusted) end, as in an ORE trade's `ScheduleData`. They define the
    trade; `evaluation_date` only says when it is priced (audit M-4).
    swap_tenor: booking convenience, an ORE Period string ("5Y", "18M")
    resolved ONCE, at construction, to the dates a spot-starting swap traded
    on `evaluation_date` has (`ore_builders.resolve_swap_dates`). Give either
    it or both dates. It is not stored: `dataclasses.replace(cfg,
    evaluation_date=...)` keeps the dates, and so the trade.
    index_tenor_months: floating leg reset frequency in months (6 = semi-annual).
    fixings: historical fixings of the floating index, `{ORE.Date: rate}`.
    Needed only for a coupon that fixed before `evaluation_date` and has not
    yet paid; ORE would equally refuse to price it without one.
    """
    notional: float
    fixed_rate: float
    payer: bool
    discount_curve_index: int
    forward_curve_index: int
    effective_date: Optional[ORE.Date] = None
    maturity_date: Optional[ORE.Date] = None
    swap_tenor: InitVar[Optional[str]] = None
    index_tenor_months: int = 6
    floating_spread: float = 0.0
    evaluation_date: ORE.Date = field(default_factory=lambda: ORE.Settings.instance().evaluationDate)
    #: The day count this swap's COUPONS accrue on -- a property of the
    #: booking, not of the engine (W1.1). Defaults to ACT/365, which is what
    #: every swap got before this field existed, so existing behavior is
    #: unchanged. A name outside `SUPPORTED_ACCRUAL_DAY_COUNTS` is REFUSED at
    #: construction, never defaulted.
    #:
    #: Distinct from the simulation time axis, which is permanently ACT/365 --
    #: see `engine.models.ore_builders`' TWO ROLES block.
    accrual_day_count: str = DEFAULT_ACCRUAL_DAY_COUNT
    fixings: Dict[ORE.Date, float] = field(default_factory=dict)

    def __post_init__(self, swap_tenor: Optional[str]) -> None:
        _validate_common_fields(self.notional, self.fixed_rate, self.evaluation_date)
        book_swap_dates(self, swap_tenor)
        validate_fixings(self.fixings)
        # Fail at construction, where the offending trade is identifiable,
        # rather than deep inside ORE at pricing time.
        resolve_accrual_day_count(self.accrual_day_count)


def _build_ore_swap(cfg: SwapConfig) -> ORE.VanillaSwap:
    """CPU: builds the real ORE trade (schedules, day counts, conventions)
    -- see `engine.models.ore_builders.build_vanilla_swap`, the single
    shared implementation of this construction (used identically by
    `european_swaption.py`/`bermudan_swaption.py`)."""
    return build_vanilla_swap(
        notional=cfg.notional, fixed_rate=cfg.fixed_rate, payer=cfg.payer,
        effective_date=cfg.effective_date, maturity_date=cfg.maturity_date,
        index_tenor_months=cfg.index_tenor_months, floating_spread=cfg.floating_spread,
        accrual_day_count=cfg.accrual_day_count,
    )


@dataclass
class SwapSchedule:
    """A swap's remaining cashflows on its evaluation date: the fixed leg,
    and the floating leg with each coupon marked fixed (a known fixing) or
    still to be projected off the forwarding curve."""
    fixed: object     # ore_builders.LegCashflows
    floating: object  # ore_builders.LegCashflows, read with the trade's fixings

    def pillar_times(self) -> List[float]:
        """Every time the pricer reads a discount factor at: the payment
        times, and the accrual start/end of each coupon still projected (a
        fixed coupon needs no forward, and a seasoned one's start is in the
        past)."""
        projected = ~self.floating.is_fixed
        return sorted(set(
            self.fixed.payment_times.tolist() + self.floating.payment_times.tolist()
            + self.floating.accrual_start_times[projected].tolist()
            + self.floating.accrual_end_times[projected].tolist()
        ))


def swap_schedule(cfg: SwapConfig) -> SwapSchedule:
    """CPU: the ORE trade's remaining cashflows on `cfg.evaluation_date`."""
    swap = _build_ore_swap(cfg)
    return SwapSchedule(
        fixed=_fixed_leg_cashflows(swap, cfg.evaluation_date),
        floating=_floating_leg_cashflows(swap, cfg.evaluation_date, cfg.fixings),
    )


def _maturity_indices(times: np.ndarray, maturities: np.ndarray) -> np.ndarray:
    """Static index lookup: each cashflow time must land exactly on a
    simulation maturity pillar (no curve interpolation -- see plan).

    np.searchsorted returns len(maturities) for any time past the last
    pillar -- an out-of-bounds index. That must be rejected outright (not
    clipped before the closeness check), since JAX silently clips
    out-of-bounds array indices rather than raising: an unclipped,
    out-of-bounds index reaching yield_curves[..., idx] downstream would
    silently price the cashflow off the wrong (last) pillar instead of
    failing loudly here.

    searchsorted's default side='left' picks the insertion point BEFORE
    any equal element, so a time that's within atol but a hair ABOVE its
    pillar (float roundoff from day-count/schedule arithmetic, not a real
    difference) gets an index pointing at the *next* pillar instead of the
    intended one -- rejecting a cashflow that's genuinely within
    tolerance, while the same-magnitude roundoff BELOW a pillar matches
    fine. Fixed by also considering the LEFT neighbor of searchsorted's
    raw index and picking whichever of the two candidate pillars is
    actually closer to `times`, before the tolerance check -- this makes
    the tolerance symmetric regardless of which side of the pillar the
    roundoff lands on, while still requiring genuine closeness (a time
    truly between two pillars, more than atol from both, is still
    rejected).
    """
    atol = 1e-6
    raw = np.searchsorted(maturities, times)
    left = np.clip(raw - 1, 0, len(maturities) - 1)
    right = np.clip(raw, 0, len(maturities) - 1)
    use_left = np.abs(maturities[left] - times) <= np.abs(maturities[right] - times)
    indices = np.where(use_left, left, right)
    matches = np.isclose(maturities[indices], times, atol=atol)
    if not np.all(matches):
        raise ValueError(
            "Swap cashflow times must be a subset of the simulation's "
            "rates.maturities pillars; got cashflow times "
            f"{times.tolist()} against maturities {maturities.tolist()}"
        )
    return indices


@dataclass(frozen=True, eq=False)
class _PreparedSwap(StaticKeyMixin):
    """Every field is compile-time-constant trade structure, resolved once by
    `prepare_swap` -- nothing here varies per scenario/step.

    `frozen=True` plus `StaticKeyMixin`'s by-value `__hash__`/`__eq__` make
    this usable as a `jax.jit` STATIC argument (see `_price_one_swap`), which
    is what lets the whole pricing kernel compile once and then be reused --
    see `engine.models.static_key` for why the generated dataclass
    `__hash__`/`__eq__` cannot do this and why by-value (not by-identity)
    matters here.

    The floating leg is split: `float_*` are the coupons still projected off
    the forwarding curve, `known_float_*` those whose fixing is already
    known (a seasoned trade's current coupon), which pay a fixed amount.
    """
    payer: bool
    fixed_notional: float
    fixed_rate: float
    fixed_accrual: np.ndarray
    fixed_pay_idx: np.ndarray
    float_notional: float
    float_spread: float
    float_accrual: np.ndarray
    float_pay_idx: np.ndarray
    float_start_idx: np.ndarray
    float_end_idx: np.ndarray
    known_float_amounts: np.ndarray
    known_float_pay_idx: np.ndarray
    discount_curve_index: int
    forward_curve_index: int


def prepare_swap(cfg: SwapConfig, maturities: np.ndarray) -> _PreparedSwap:
    """CPU: build the ORE trade and resolve every remaining cashflow onto the
    simulation's maturity pillars. Static per swap -- run once, not per
    scenario/step."""
    schedule = swap_schedule(cfg)
    fixed, floating = schedule.fixed, schedule.floating
    projected, known = ~floating.is_fixed, floating.is_fixed
    # ORE's IborCoupon amount: nominal * (fixing + spread) * accrualPeriod.
    known_amounts = (floating.notional * (floating.fixed_rates[known] + cfg.floating_spread)
                     * floating.accrual_fractions[known])

    return _PreparedSwap(
        payer=cfg.payer,
        fixed_notional=fixed.notional,
        fixed_rate=cfg.fixed_rate,
        fixed_accrual=fixed.accrual_fractions,
        fixed_pay_idx=_maturity_indices(fixed.payment_times, maturities),
        float_notional=floating.notional,
        float_spread=cfg.floating_spread,
        float_accrual=floating.accrual_fractions[projected],
        float_pay_idx=_maturity_indices(floating.payment_times[projected], maturities),
        float_start_idx=_maturity_indices(floating.accrual_start_times[projected], maturities),
        float_end_idx=_maturity_indices(floating.accrual_end_times[projected], maturities),
        known_float_amounts=known_amounts,
        known_float_pay_idx=_maturity_indices(floating.payment_times[known], maturities),
        discount_curve_index=cfg.discount_curve_index,
        forward_curve_index=cfg.forward_curve_index,
    )


@partial(jax.jit, static_argnums=1)
def _price_one_swap(yield_curves: jax.Array, swap: _PreparedSwap) -> jax.Array:
    """
    GPU: [Scenarios, TimeSteps] NPV for a single prepared swap, vectorized
    across every simulated scenario and step at once via the precomputed
    maturity-pillar indices (no interpolation, no per-cashflow Python loop).

    Fixed leg PV(t)  = notional * fixed_rate * sum_i[ accrual_i * P_disc(t, T_i) ]
    Float leg PV(t)  = notional * sum_i[ (F_i(t) + spread) * accrual_i * P_disc(t, T_i) ]
                       + sum_k[ known_amount_k * P_disc(t, T_k) ]
        where F_i(t) = (P_fwd(t, T_{i-1}) / P_fwd(t, T_i) - 1) / accrual_i
        is the simulated forward rate implied by the forwarding curve
        (single-period, at-par coupon convention -- matches ORE's
        IborCoupon.usingAtParCoupons() default, live-verified against
        ORE.VanillaSwap.floatingLegNPV() in tests/test_swap.py), and
        known_amount_k is a coupon whose fixing is already known (see
        `prepare_swap`).
    NPV(t) = floatLegPV(t) - fixedLegPV(t), negated for payer=False --
        matches ORE.VanillaSwap.Payer/.Receiver sign convention.

    No optionality is priced here (this is a linear instrument), so this is
    a direct expectation under each simulated scenario/step -- no nested
    Monte Carlo or numeraire-based discounting is needed.
    """
    disc = yield_curves[:, :, :, swap.discount_curve_index]  # [S, T, Maturities]
    fwd = yield_curves[:, :, :, swap.forward_curve_index]

    fixed_accrual = jnp.asarray(swap.fixed_accrual, dtype=yield_curves.dtype)
    fixed_disc = disc[:, :, swap.fixed_pay_idx]  # [S, T, NumFixedCF]
    fixed_leg_pv = swap.fixed_notional * swap.fixed_rate * jnp.tensordot(
        fixed_disc, fixed_accrual, axes=([2], [0])
    )

    float_accrual = jnp.asarray(swap.float_accrual, dtype=yield_curves.dtype)
    p_start = fwd[:, :, swap.float_start_idx]
    p_end = fwd[:, :, swap.float_end_idx]
    forward_rate = (p_start / p_end - 1.0) / float_accrual[None, None, :]

    float_disc = disc[:, :, swap.float_pay_idx]
    float_cashflow = swap.float_notional * (forward_rate + swap.float_spread) * float_accrual[None, None, :]
    known_amounts = jnp.asarray(swap.known_float_amounts, dtype=yield_curves.dtype)
    float_leg_pv = jnp.sum(float_cashflow * float_disc, axis=2) + jnp.tensordot(
        disc[:, :, swap.known_float_pay_idx], known_amounts, axes=([2], [0])
    )

    npv = float_leg_pv - fixed_leg_pv
    return npv if swap.payer else -npv


def price_swaps(yield_curves: jax.Array, maturities: np.ndarray, swap_configs: List[SwapConfig]) -> jax.Array:
    """
    yield_curves: [Scenarios, TimeSteps, Maturities, NumRates], from
        engine.simulation.generate_paths(...)["yield_curves"].
    maturities: the same absolute-time pillar array passed as
        config["rates"]["maturities"] to generate_paths.
    Returns: [Scenarios, TimeSteps, Trades] NPV cube.
    """
    maturities_np = np.asarray(maturities)
    prepared = [prepare_swap(cfg, maturities_np) for cfg in swap_configs]
    per_trade = [_price_one_swap(yield_curves, swap) for swap in prepared]
    return jnp.stack(per_trade, axis=-1)


# =============================================================================
# EXECUTION DEMONSTRATION
# =============================================================================
if __name__ == "__main__":
    from engine.simulation.market_model import generate_paths
    from engine.simulation.demo_scenarios import EVAL_DATE, SWAP_DEMO_MATURITIES, single_currency_swap_demo_config

    market_cubes = generate_paths(single_currency_swap_demo_config())

    swap_cfg = SwapConfig(
        notional=1_000_000.0,
        fixed_rate=0.03,
        payer=True,
        discount_curve_index=0,
        forward_curve_index=1,
        swap_tenor="2Y",
        evaluation_date=EVAL_DATE,
    )

    npv_cube = price_swaps(market_cubes["yield_curves"], SWAP_DEMO_MATURITIES, [swap_cfg])
    print("NPV cube shape:", npv_cube.shape)
    # The cube's time axis is time_grid[1:] -- index 0 is the first simulated
    # step, not t=0.
    first_step = single_currency_swap_demo_config().time_grid[1]
    print(f"Mean NPV across scenarios at t={first_step}:", float(jnp.mean(npv_cube[:, 0, 0])))
