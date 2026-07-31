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
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import ORE

DAY_COUNTER = ORE.Actual365Fixed()


@dataclass
class SwapConfig:
    """
    One vanilla fixed-vs-floating interest rate swap.

    discount_curve_index / forward_curve_index index into the NumRates axis
    of the simulation's yield_curves cube (engine.market_simulations
    generate_paths' "rates" config -- each index is one Hull-White factor /
    initial_zero_curve). Both legs discount off discount_curve_index; the
    floating leg's forward rates are read off forward_curve_index. Equal
    indices reduce to single-curve discounting; distinct indices reproduce
    ORE's multi-curve DiscountingSwapEngine (discount curve) + IborIndex
    (its own, separate forwardingTermStructure) split.

    swap_tenor: ORE Period string, e.g. "5Y", "18M".
    index_tenor_months: floating leg reset frequency in months (6 = semi-annual).
    """
    notional: float
    fixed_rate: float
    payer: bool
    discount_curve_index: int
    forward_curve_index: int
    swap_tenor: str = "5Y"
    index_tenor_months: int = 6
    floating_spread: float = 0.0
    evaluation_date: ORE.Date = field(default_factory=lambda: ORE.Settings.instance().evaluationDate)


@dataclass
class _LegCashflows:
    payment_times: np.ndarray       # [N] year-fractions from evaluation_date
    accrual_start_times: np.ndarray  # [N]
    accrual_end_times: np.ndarray    # [N]
    accrual_fractions: np.ndarray    # [N]
    notional: float


def _build_ore_swap(cfg: SwapConfig) -> ORE.VanillaSwap:
    """CPU: builds the real ORE trade (schedules, day counts, conventions).

    Both legs explicitly use Actual/365Fixed (DAY_COUNTER) rather than
    relying on MakeVanillaSwap's implicit per-index defaults (which differ
    unpredictably by index/currency, e.g. Euribor6M defaults to 30/360 fixed
    vs Act/360 float) -- this keeps the convention a documented, deliberate
    choice consistent with the simulation's own year-fraction time axis.
    """
    ORE.Settings.instance().evaluationDate = cfg.evaluation_date
    dummy_forward_curve = ORE.YieldTermStructureHandle(
        ORE.FlatForward(cfg.evaluation_date, 0.0, DAY_COUNTER)
    )
    index = ORE.IborIndex(
        "SimIndex", ORE.Period(cfg.index_tenor_months, ORE.Months), 2,
        ORE.USDCurrency(), ORE.TARGET(), ORE.ModifiedFollowing, False,
        DAY_COUNTER, dummy_forward_curve,
    )
    swap_type = ORE.VanillaSwap.Payer if cfg.payer else ORE.VanillaSwap.Receiver
    swap = ORE.MakeVanillaSwap(
        ORE.Period(cfg.swap_tenor), index, cfg.fixed_rate,
        nominal=cfg.notional,
        swapType=swap_type,
        floatingLegSpread=cfg.floating_spread,
        fixedLegDayCount=DAY_COUNTER,
        floatingLegDayCount=DAY_COUNTER,
    )
    return swap


def _fixed_leg_cashflows(swap: ORE.VanillaSwap, today: ORE.Date) -> _LegCashflows:
    """CPU: extracts each fixed coupon's payment/accrual dates (as
    year-fractions from `today`) and ORE's own accrualPeriod() for each,
    from the real ORE-generated schedule -- no date/day-count math
    reimplemented here."""
    payment_times, accrual_starts, accrual_ends, fractions = [], [], [], []
    for cf in swap.fixedLeg():
        c = ORE.as_fixed_rate_coupon(cf)
        payment_times.append(DAY_COUNTER.yearFraction(today, c.date()))
        accrual_starts.append(DAY_COUNTER.yearFraction(today, c.accrualStartDate()))
        accrual_ends.append(DAY_COUNTER.yearFraction(today, c.accrualEndDate()))
        fractions.append(c.accrualPeriod())
    return _LegCashflows(
        payment_times=np.array(payment_times),
        accrual_start_times=np.array(accrual_starts),
        accrual_end_times=np.array(accrual_ends),
        accrual_fractions=np.array(fractions),
        notional=swap.fixedNominals()[0] if swap.fixedNominals() else swap.nominal(),
    )


def _floating_leg_cashflows(swap: ORE.VanillaSwap, today: ORE.Date) -> _LegCashflows:
    """CPU: same extraction as _fixed_leg_cashflows, for the floating leg's
    coupons. accrual_start/end_times are what forward rates get computed
    from in _price_one_swap -- ORE's own fixing is not used, since the
    whole point is repricing under the JAX-simulated scenarios."""
    payment_times, accrual_starts, accrual_ends, fractions = [], [], [], []
    for cf in swap.floatingLeg():
        c = ORE.as_floating_rate_coupon(cf)
        payment_times.append(DAY_COUNTER.yearFraction(today, c.date()))
        accrual_starts.append(DAY_COUNTER.yearFraction(today, c.accrualStartDate()))
        accrual_ends.append(DAY_COUNTER.yearFraction(today, c.accrualEndDate()))
        fractions.append(c.accrualPeriod())
    return _LegCashflows(
        payment_times=np.array(payment_times),
        accrual_start_times=np.array(accrual_starts),
        accrual_end_times=np.array(accrual_ends),
        accrual_fractions=np.array(fractions),
        notional=swap.floatingNominals()[0] if swap.floatingNominals() else swap.nominal(),
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
    """
    indices = np.searchsorted(maturities, times)
    in_bounds = indices < len(maturities)
    matches = np.zeros_like(in_bounds)
    matches[in_bounds] = np.isclose(maturities[indices[in_bounds]], times[in_bounds], atol=1e-6)
    if not np.all(matches):
        raise ValueError(
            "Swap cashflow times must be a subset of the simulation's "
            "rates.maturities pillars; got cashflow times "
            f"{times.tolist()} against maturities {maturities.tolist()}"
        )
    return indices


@dataclass
class _PreparedSwap:
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
    discount_curve_index: int
    forward_curve_index: int


def prepare_swap(cfg: SwapConfig, maturities: np.ndarray) -> _PreparedSwap:
    """CPU: build the ORE trade and resolve every cashflow onto the
    simulation's maturity pillars. Static per swap -- run once, not per
    scenario/step."""
    swap = _build_ore_swap(cfg)
    today = cfg.evaluation_date

    fixed = _fixed_leg_cashflows(swap, today)
    floating = _floating_leg_cashflows(swap, today)

    return _PreparedSwap(
        payer=cfg.payer,
        fixed_notional=fixed.notional,
        fixed_rate=cfg.fixed_rate,
        fixed_accrual=fixed.accrual_fractions,
        fixed_pay_idx=_maturity_indices(fixed.payment_times, maturities),
        float_notional=floating.notional,
        float_spread=cfg.floating_spread,
        float_accrual=floating.accrual_fractions,
        float_pay_idx=_maturity_indices(floating.payment_times, maturities),
        float_start_idx=_maturity_indices(floating.accrual_start_times, maturities),
        float_end_idx=_maturity_indices(floating.accrual_end_times, maturities),
        discount_curve_index=cfg.discount_curve_index,
        forward_curve_index=cfg.forward_curve_index,
    )


def _price_one_swap(yield_curves: jax.Array, swap: _PreparedSwap) -> jax.Array:
    """
    GPU: [Scenarios, TimeSteps] NPV for a single prepared swap, vectorized
    across every simulated scenario and step at once via the precomputed
    maturity-pillar indices (no interpolation, no per-cashflow Python loop).

    Fixed leg PV(t)  = notional * fixed_rate * sum_i[ accrual_i * P_disc(t, T_i) ]
    Float leg PV(t)  = notional * sum_i[ (F_i(t) + spread) * accrual_i * P_disc(t, T_i) ]
        where F_i(t) = (P_fwd(t, T_{i-1}) / P_fwd(t, T_i) - 1) / accrual_i
        is the simulated forward rate implied by the forwarding curve
        (single-period, at-par coupon convention -- matches ORE's
        IborCoupon.usingAtParCoupons() default, live-verified against
        ORE.VanillaSwap.floatingLegNPV() in tests/test_interest_rate_swap.py).
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
    float_leg_pv = jnp.sum(float_cashflow * float_disc, axis=2)

    npv = float_leg_pv - fixed_leg_pv
    return npv if swap.payer else -npv


def price_swaps(yield_curves: jax.Array, maturities: np.ndarray, swap_configs: List[SwapConfig]) -> jax.Array:
    """
    yield_curves: [Scenarios, TimeSteps, Maturities, NumRates], from
        engine.market_simulations.generate_paths(...)["yield_curves"].
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
    from engine.market_simulations import generate_paths

    eval_date = ORE.Date(30, 7, 2026)

    payload = {
        "time_grid": [0.0, 0.5, 1.0, 1.5, 2.0],
        "scenarios": 4096,
        "equities": {
            "initial_prices": [150.0],
            "dividend_yields": [0.0],
            "rate_mapping": [[1.0, 0.0]],
        },
        "rates": {
            # Two correlated USD factors: 0 = OIS/discounting, 1 = Euribor-style forwarding
            "initial_rates": [0.030, 0.035],
            "theta": [0.030, 0.035],
            "mean_reversion": [0.03, 0.03],
            # Union of both legs' accrual/payment dates (annual fixed, semi-annual float,
            # including the swap's spot-lag effective date as the first float accrual start)
            "maturities": [
                0.010958904109589041, 0.5150684931506849, 1.010958904109589,
                1.515068493150685, 2.0136986301369864,
            ],
            "initial_zero_curve": {
                "times": [0.0, 1.0, 2.0, 5.0, 10.0, 30.0],
                "rates": [0.030, 0.030, 0.030, 0.030, 0.030, 0.030],
            },
        },
        "joint_covariance": [
            [0.0400, 0.0000, 0.0000],
            [0.0000, 0.0001, 0.00005],
            [0.0000, 0.00005, 0.0001],
        ],
    }
    market_cubes = generate_paths(payload)

    swap_cfg = SwapConfig(
        notional=1_000_000.0,
        fixed_rate=0.03,
        payer=True,
        discount_curve_index=0,
        forward_curve_index=1,
        swap_tenor="2Y",
        evaluation_date=eval_date,
    )

    npv_cube = price_swaps(market_cubes["yield_curves"], payload["rates"]["maturities"], [swap_cfg])
    print("NPV cube shape:", npv_cube.shape)
    print("Mean t=0 NPV across scenarios:", float(jnp.mean(npv_cube[:, 0, 0])))
