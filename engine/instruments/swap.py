"""
Vanilla fixed-vs-floating swap pricing on the simulated yield-curve cube.

Schedules and accrual fractions come from ORE (`MakeVanillaSwap`); the pricing is a tensor
contraction over the cube `[scenario, step, maturity, curve]`.

Multi-curve: both legs discount off `discount_curve_index`; floating forwards come from
`forward_curve_index`, as ORE's `DiscountingSwapEngine` plus an `IborIndex` with its own
forwarding curve. Floating coupons are projected over their accrual period (ORE's default
at-par coupons).

Seasoned trades are priced as ORE prices them at t=0: cashflows paid on or before the
evaluation date are dropped (`ore_builders.is_live`), and a coupon fixed before it pays its
historical fixing from `SwapConfig.fixings`, or raises if missing
(`ore_builders.known_fixing`). Checked against `ORE.DiscountingSwapEngine` in
tests/test_trade_dates.py.

Differs from ORE:
  * Discount factors are read directly at cube pillars, with no interpolation; every
    cashflow time must be a pillar of the simulation (`_maturity_indices`).
  * Known issue (I-36): the floating forward is annualized by the leg's accrual fraction,
    where ORE uses the index day count's spanning time. They agree only when the leg
    accrues on the index's ACT/365; any other `accrual_day_count` misprices the leg.
  * Known limitation (I-04, FLAGGED): at a simulated step past a coupon's accrual start or
    payment date, the coupon is still priced off the cube, whose P(t,T) for T < t is a
    clamped value rather than a discount factor. There is no per-path fixing of coupons
    that fix during the simulation, and no removal of cashflows paid during it. t=0 prices
    are exact. `engine.portfolio.request` warns per affected swap.
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
    One vanilla fixed-vs-floating swap.

    discount_curve_index / forward_curve_index: curves on the cube's rate axis. Equal
        indices give single-curve pricing.
    effective_date / maturity_date: the booked schedule's start and unadjusted end. They
        define the trade; `evaluation_date` only sets when it is priced.
    swap_tenor: booking shortcut ("5Y", "18M"), resolved once at construction to the
        dates of a spot-starting swap traded on `evaluation_date`. Give either it or both
        dates. Not stored, so `dataclasses.replace(cfg, evaluation_date=...)` keeps the
        trade.
    index_tenor_months: floating reset frequency in months.
    fixings: historical index fixings, `{ORE.Date: rate}`; needed only for a coupon that
        fixed before `evaluation_date` and has not yet paid.
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
    #: Coupon accrual day count, a property of the booking. Names outside
    #: `SUPPORTED_ACCRUAL_DAY_COUNTS` are refused. Not the simulation time axis (always
    #: ACT/365; see `engine.models.ore_builders`).
    accrual_day_count: str = DEFAULT_ACCRUAL_DAY_COUNT
    fixings: Dict[ORE.Date, float] = field(default_factory=dict)

    def __post_init__(self, swap_tenor: Optional[str]) -> None:
        _validate_common_fields(self.notional, self.fixed_rate, self.evaluation_date)
        book_swap_dates(self, swap_tenor)
        validate_fixings(self.fixings)
        # Validate here, where the trade is identifiable, not later inside ORE.
        resolve_accrual_day_count(self.accrual_day_count)


def _build_ore_swap(cfg: SwapConfig) -> ORE.VanillaSwap:
    """The ORE trade (see `engine.models.ore_builders.build_vanilla_swap`)."""
    return build_vanilla_swap(
        notional=cfg.notional, fixed_rate=cfg.fixed_rate, payer=cfg.payer,
        effective_date=cfg.effective_date, maturity_date=cfg.maturity_date,
        index_tenor_months=cfg.index_tenor_months, floating_spread=cfg.floating_spread,
        accrual_day_count=cfg.accrual_day_count,
    )


@dataclass
class SwapSchedule:
    """A swap's remaining cashflows on its evaluation date. Floating coupons are marked
    fixed (known fixing) or projected."""
    fixed: object     # ore_builders.LegCashflows
    floating: object  # ore_builders.LegCashflows, read with the trade's fixings

    def pillar_times(self) -> List[float]:
        """Every time a discount factor is read at: payment times, plus accrual start and
        end of each projected coupon."""
        projected = ~self.floating.is_fixed
        return sorted(set(
            self.fixed.payment_times.tolist() + self.floating.payment_times.tolist()
            + self.floating.accrual_start_times[projected].tolist()
            + self.floating.accrual_end_times[projected].tolist()
        ))


def swap_schedule(cfg: SwapConfig) -> SwapSchedule:
    """The ORE trade's remaining cashflows on `cfg.evaluation_date`."""
    swap = _build_ore_swap(cfg)
    return SwapSchedule(
        fixed=_fixed_leg_cashflows(swap, cfg.evaluation_date),
        floating=_floating_leg_cashflows(swap, cfg.evaluation_date, cfg.fixings),
    )


def _maturity_indices(times: np.ndarray, maturities: np.ndarray) -> np.ndarray:
    """Index of the cube pillar equal to each cashflow time (within 1e-6), or raise.

    Takes the nearer of the two neighbouring pillars so that round-off on either side
    matches. Out-of-range times must raise here: JAX clips out-of-bounds indices, which
    would silently price off the last pillar.
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
    """Per-trade constant structure, resolved once by `prepare_swap`; a `jax.jit` static
    argument (see `engine.models.static_key`).

    Floating coupons are split into `float_*` (projected off the forwarding curve) and
    `known_float_*` (fixing already known, a fixed amount).
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
    """Build the ORE trade and map its remaining cashflows onto the cube's pillars."""
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
    [Scenarios, TimeSteps] NPV of one prepared swap.

        Fixed PV(t) = N * K * sum_i accrual_i * P_disc(t, T_i)
        Float PV(t) = N * sum_i (F_i(t) + spread) * accrual_i * P_disc(t, T_i)
                      + sum_k known_amount_k * P_disc(t, T_k)
        F_i(t)      = (P_fwd(t, start_i) / P_fwd(t, end_i) - 1) / accrual_i
        NPV(t)      = Float PV - Fixed PV for a payer; negated for a receiver.

    Floating accrual and forward period coincide (ORE's at-par coupon default); matches
    `ORE.VanillaSwap.floatingLegNPV()` in tests/test_swap.py for an ACT/365 leg (see I-36 in
    the module docstring for other day counts).
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
    NPV cube `[Scenarios, TimeSteps, Trades]` for `swap_configs`.

    yield_curves: `[Scenarios, TimeSteps, Maturities, NumRates]` from
        `engine.simulation.generate_paths(...)["yield_curves"]`.
    maturities: the pillar times given to `generate_paths` as `rates.maturities`.
    """
    maturities_np = np.asarray(maturities)
    prepared = [prepare_swap(cfg, maturities_np) for cfg in swap_configs]
    per_trade = [_price_one_swap(yield_curves, swap) for swap in prepared]
    return jnp.stack(per_trade, axis=-1)


# Demo
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
    # The cube's time axis is time_grid[1:]: index 0 is the first simulated step, not t=0.
    first_step = single_currency_swap_demo_config().time_grid[1]
    print(f"Mean NPV across scenarios at t={first_step}:", float(jnp.mean(npv_cube[:, 0, 0])))
