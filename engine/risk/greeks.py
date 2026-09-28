"""
Delta, Gamma, Vega and Theta for swaps, European swaptions and Bermudan/American swaptions.

Delta/Gamma: per pillar of each curve, in ORE's units (NPV change for a 1bp absolute
zero-rate bump). ORE bumps one pillar at a time with a triangular shape
(`ShiftScenarioGenerator::applyShift`) and differences NPVs (`SensitivityCube`:
`delta = NPV_up - NPV_base`, `gamma = NPV_up - 2*NPV_base + NPV_down`). Linear zero-rate
interpolation has the same triangular dependence on each pillar, so here Delta is
`dNPV/dz_i * bump` and Gamma is `d^2NPV/dz_i^2 * bump^2`, by autodiff. Only the diagonal
Gamma is computed; ORE's optional cross-gammas are not.

Differs from ORE: these are derivatives, the bump -> 0 limit of ORE's finite differences.
ORE's forward-difference Delta also contains half the Gamma (`0.5 * gamma`) and higher
terms, so the two differ by O(bump^2); they agree only in that limit. Rate Delta is ORE's
equivalent of Rho; there is no separate Rho.

Vega (Bermudan/American only): ORE bumps the market swaption vols used to calibrate the
model and recalibrates. Here the calibration (`engine.calibration.lgm`) is differentiated
through by the implicit function theorem instead of rerun, giving
dNPV/d(market vol) per basket instrument (see `bermudan_vega`). Swaps and Europeans have no
Vega here.

Theta: ORE's definition (`SensitivityAnalysis`, OREAnalytics/orea/engine/
sensitivityanalysis.cpp): value the same trade on a later evaluation date against the same
market, and add back cashflows paid in between:
`Theta = NPV(t+dt) - NPV(t) + CF(t, t+dt]`. The trade keeps its booked dates, so it is one
day older; a fixing that prints in [t, t+dt) is taken at the base valuation's forecast
(`_theta_fixings`).

Known issue: the later date is `TARGET().advance(t, theta_days, Days)`, i.e. business
days, whereas ORE uses `asof + thetaPeriod` (calendar days). From a Friday the engine
measures three days of Theta where ORE measures one; see I-38 in docs/known-issues.md.

An `AmericanSwaptionConfig` goes through the same functions as a Bermudan.
"""
import dataclasses
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import ORE

from engine.instruments.swap import SwapConfig, _build_ore_swap, swap_schedule
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.bermudan_swaption import _build_ore_swap as _build_ore_underlying
from engine.models.ore_builders import (  # noqa: F401  (DAY_COUNTER is a re-export)
    DAY_COUNTER,
    TIME_AXIS_DAY_COUNTER,
    time_from_reference,
)
# `_hw_A` and `_zero_rate_at` are imported by tests from this module.
from engine.models.hull_white import (  # noqa: F401
    A as _hw_A, ZeroCurve, discount as _discount_at, zero_rate as _zero_rate_at,
)
from engine.models.lgm import Sigma, as_sigma
# The t=0 price functions live in engine.risk.price_functions (shared with
# engine.market_risk); these private names are kept for existing callers.
from engine.risk.price_functions import (
    bermudan_price_function as _bermudan_price_fn,
    swap_price_function as _swap_price_fn,
    swaption_price_function as _swaption_price_fn,
)

# 1bp absolute zero-rate bump, as ORE's example sensitivity config
# (Examples/MarketRisk/Input/sensitivity.xml).
DEFAULT_RATE_BUMP = 0.0001

# Theta horizon in days; ORE's default `thetaPeriod` is 1 day. See the module docstring on
# business vs calendar days.
DEFAULT_THETA_DAYS = 1

# `TIME_AXIS_DAY_COUNTER` (ACT/365) converts dates to years on the simulation axis.


# Gradient and Hessian diagonal (shared by every Delta/Gamma function)
def _grad_and_hessian_diagonal(price_fn, x, *rest):
    """`(df/dx_i, d^2f/dx_i^2)` for every `i`, without building the Hessian.

    Each diagonal entry is one Hessian-vector product against a basis vector, batched
    with `vmap`; checked against `jnp.diagonal(jax.hessian(...))` in
    `tests/test_profiling_and_jit.py::TestHessianDiagonalEquivalence`.
    Differentiates with respect to the first argument; `rest` is held fixed.

    Both outputs come from one jitted program. `price_fn` is a fresh closure on every
    call, so it recompiles once per call (I-21; read that entry before memoizing it, since
    a wrongly keyed memo returns a program compiled for a different trade).
    """
    def combined(xi, *fixed):
        def f(inner):
            return price_fn(inner, *fixed)

        grad = jax.grad(f)(xi)

        # vmap over the basis vectors: all n HVPs as one batched program.
        basis = jnp.eye(xi.shape[0], dtype=xi.dtype)

        def hvp(v):
            return jax.jvp(jax.grad(f), (xi,), (v,))[1]

        rows = jax.vmap(hvp)(basis)   # [n, n]; only its diagonal escapes
        return grad, jnp.diagonal(rows)

    return jax.jit(combined)(x, *rest)


# Swap: Delta / Gamma
def swap_delta_gamma(
    cfg: SwapConfig,
    disc_curve: ZeroCurve,
    fwd_curve: ZeroCurve,
    bump_size: float = DEFAULT_RATE_BUMP,
) -> Dict[str, jax.Array]:
    """
    Per-pillar Delta and Gamma of one swap's t=0 NPV with respect to its discount and
    forward curves, per `bump_size` (see module docstring).

    Returns `discount_delta`/`discount_gamma` and `forward_delta`/`forward_gamma`, each
    `[len(pillar_rates)]`. The two curves are independent variables even if they are the
    same curve; then the total sensitivity is `discount_delta + forward_delta`.
    """
    price_fn = _swap_price_fn(cfg, disc_curve, fwd_curve)

    # `_grad_and_hessian_diagonal` differentiates its first argument, so the forward-curve
    # call swaps the argument order.
    disc_delta, disc_gamma = _grad_and_hessian_diagonal(
        price_fn, disc_curve.pillar_rates, fwd_curve.pillar_rates
    )
    fwd_delta, fwd_gamma = _grad_and_hessian_diagonal(
        lambda fwd_rates, disc_rates: price_fn(disc_rates, fwd_rates),
        fwd_curve.pillar_rates, disc_curve.pillar_rates,
    )

    return {
        "discount_delta": disc_delta * bump_size,
        "discount_gamma": disc_gamma * bump_size ** 2,
        "forward_delta": fwd_delta * bump_size,
        "forward_gamma": fwd_gamma * bump_size ** 2,
    }


# Swap: Theta
def swap_theta(
    cfg: SwapConfig,
    disc_curve: ZeroCurve,
    fwd_curve: ZeroCurve,
    theta_days: int = DEFAULT_THETA_DAYS,
) -> float:
    """
    Theta = NPV(t + theta_days, same curves) - NPV(t) + cashflows paid in (t, t + theta_days]
    (see module docstring). A finite difference in time, not autodiff.
    """
    # Each valuation is jitted into one program (docs/concepts/profiling.md).
    base_price_fn = _swap_price_fn(cfg, disc_curve, fwd_curve)
    base_npv = float(jax.jit(base_price_fn)(disc_curve.pillar_rates, fwd_curve.pillar_rates))

    theta_date = ORE.TARGET().advance(cfg.evaluation_date, theta_days, ORE.Days)

    def at_par_forecast(coupon):
        # The rate the base valuation projects for this coupon: at par, over its accrual
        # period (as `_price_one_swap`).
        start, end = (time_from_reference(cfg.evaluation_date, d)
                      for d in (coupon.accrualStartDate(), coupon.accrualEndDate()))
        p_start, p_end = (float(_discount_at(fwd_curve, jnp.asarray(t))) for t in (start, end))
        return (p_start / p_end - 1.0) / coupon.accrualPeriod()

    fixings = _theta_fixings(cfg, _build_ore_swap(cfg), theta_date, at_par_forecast)
    theta_cfg = dataclasses.replace(cfg, evaluation_date=theta_date, fixings=fixings)
    theta_price_fn = _swap_price_fn(theta_cfg, disc_curve, fwd_curve)
    theta_npv = float(jax.jit(theta_price_fn)(disc_curve.pillar_rates, fwd_curve.pillar_rates))

    period_flow = _swap_cashflows_in_period(cfg, cfg.evaluation_date, theta_date, fwd_curve)

    return theta_npv - base_npv + period_flow


def _theta_fixings(cfg, swap: ORE.VanillaSwap, theta_date: ORE.Date, forecast) -> Dict[ORE.Date, float]:
    """`cfg.fixings` plus, for each floating coupon fixing in
    `[cfg.evaluation_date, theta_date)` without a supplied fixing, `forecast(coupon)`. On
    the Theta date those fixings are history; Theta holds the curve fixed, so they print
    at the base forecast."""
    fixings = dict(cfg.fixings)
    for cf in swap.floatingLeg():
        coupon = ORE.as_floating_rate_coupon(cf)
        fixing_date = coupon.fixingDate()
        if cfg.evaluation_date <= fixing_date < theta_date and fixing_date not in fixings:
            fixings[fixing_date] = forecast(coupon)
    return fixings


def _swap_cashflows_in_period(cfg: SwapConfig, start: ORE.Date, end: ORE.Date, fwd_curve: ZeroCurve) -> float:
    """Net cashflow (signed as in `_price_one_swap`) paid in `(start, end]`, ORE's
    `aggregateTradeFlow` step. Both legs are included; floating coupons use their known
    fixing or are projected off `fwd_curve` as the pricer projects them."""
    schedule = swap_schedule(cfg)
    fixed, floating = schedule.fixed, schedule.floating

    start_frac = TIME_AXIS_DAY_COUNTER.yearFraction(cfg.evaluation_date, start)
    end_frac = TIME_AXIS_DAY_COUNTER.yearFraction(cfg.evaluation_date, end)

    def in_window(times):
        return (times > start_frac) & (times <= end_frac)

    fixed_flow = float(np.sum(
        in_window(fixed.payment_times) * fixed.notional * cfg.fixed_rate * fixed.accrual_fractions
    ))

    float_mask = in_window(floating.payment_times)
    float_flow = 0.0
    if np.any(float_mask):
        p_start = np.asarray(_discount_at(fwd_curve, jnp.asarray(floating.accrual_start_times[float_mask])), dtype=np.float64)
        p_end = np.asarray(_discount_at(fwd_curve, jnp.asarray(floating.accrual_end_times[float_mask])), dtype=np.float64)
        accrual = floating.accrual_fractions[float_mask]
        projected = (p_start / p_end - 1.0) / accrual
        rate = np.where(floating.is_fixed[float_mask], floating.fixed_rates[float_mask], projected)
        float_flow = float(np.sum(floating.notional * (rate + cfg.floating_spread) * accrual))

    # npv = float leg - fixed leg, negated for a receiver (as `_price_one_swap`).
    net = float_flow - fixed_flow
    return net if cfg.payer else -net


# European swaption: Delta / Gamma
def swaption_delta_gamma(
    cfg: SwaptionConfig,
    curve: ZeroCurve,
    bump_size: float = DEFAULT_RATE_BUMP,
) -> Dict[str, jax.Array]:
    """
    Per-pillar Delta and Gamma of one European swaption's t=0 NPV with respect to its
    Hull-White curve, per `bump_size`. `curve` must be `cfg.initial_zero_curve` as a
    `ZeroCurve` (this function does not read the config's curve).

    Returns `{"delta", "gamma"}`, each `[len(curve.pillar_rates)]`; Gamma is diagonal.
    """
    price_fn = _swaption_price_fn(cfg, curve)
    delta, gamma = _grad_and_hessian_diagonal(price_fn, curve.pillar_rates)
    return {
        "delta": delta * bump_size,
        "gamma": gamma * bump_size ** 2,
    }


# European swaption: Theta
def swaption_theta(
    cfg: SwaptionConfig,
    curve: ZeroCurve,
    theta_days: int = DEFAULT_THETA_DAYS,
) -> float:
    """
    Theta = NPV(t + theta_days, same curve) - NPV(t), with the exercise date fixed. No
    cashflow is paid before exercise, so there is no flow term.
    """
    # Jitted: the Jamshidian r* solve otherwise dispatches op by op.
    base_price_fn = _swaption_price_fn(cfg, curve)
    base_npv = float(jax.jit(base_price_fn)(curve.pillar_rates))

    # The same option one day older: its exercise date is unchanged.
    theta_date = ORE.TARGET().advance(cfg.evaluation_date, theta_days, ORE.Days)
    theta_cfg = dataclasses.replace(cfg, evaluation_date=theta_date)
    theta_price_fn = _swaption_price_fn(theta_cfg, curve)
    theta_npv = float(jax.jit(theta_price_fn)(curve.pillar_rates))

    return theta_npv - base_npv


# Bermudan / American swaption: Delta / Gamma / Vega / Theta
def bermudan_delta_gamma(
    cfg: BermudanSwaptionConfig,
    curve: ZeroCurve,
    bump_size: float = DEFAULT_RATE_BUMP,
) -> Dict[str, jax.Array]:
    """
    Per-pillar Delta and Gamma of one Bermudan/American swaption's t=0 NPV with respect to
    its curve, per `bump_size`. `curve` must have `cfg.initial_zero_curve`'s pillar times.

    Returns `{"delta", "gamma"}`, each `[len(curve.pillar_rates)]`; Gamma is diagonal.
    """
    price_fn, sigma_values = _bermudan_price_fn(cfg, curve)
    delta, gamma = _grad_and_hessian_diagonal(price_fn, curve.pillar_rates, sigma_values)
    return {
        "delta": delta * bump_size,
        "gamma": gamma * bump_size ** 2,
    }


def bermudan_theta(
    cfg: BermudanSwaptionConfig,
    curve: ZeroCurve,
    theta_days: int = DEFAULT_THETA_DAYS,
) -> float:
    """
    Theta = NPV(t + theta_days, same curve) - NPV(t). An option pays nothing before
    exercise, so there is no flow term.
    """
    # Jitted for consistency with the other Theta functions.
    price_fn, sigma_values = _bermudan_price_fn(cfg, curve)
    base_npv = float(jax.jit(price_fn)(curve.pillar_rates, sigma_values))

    # Same trade one day on: dates fixed, times re-derived from the new evaluation date,
    # as ORE re-derives optionTimes.
    theta_date = ORE.TARGET().advance(cfg.evaluation_date, theta_days, ORE.Days)

    def index_forecast(coupon):
        # Today's forecast over the index period, as the LGM engine forecasts a fixing
        # dated today (`LgmVectorised::fixing`).
        index = coupon.index()
        value_date = index.valueDate(coupon.fixingDate())
        maturity = index.maturityDate(value_date)
        p1, p2 = (float(_discount_at(curve, jnp.asarray(time_from_reference(cfg.evaluation_date, d))))
                  for d in (value_date, maturity))
        return (p1 / p2 - 1.0) / index.dayCounter().yearFraction(value_date, maturity)

    fixings = _theta_fixings(cfg, _build_ore_underlying(cfg), theta_date, index_forecast)
    theta_cfg = dataclasses.replace(cfg, evaluation_date=theta_date, fixings=fixings)
    theta_price_fn, theta_sigma_values = _bermudan_price_fn(theta_cfg, curve)
    theta_npv = float(jax.jit(theta_price_fn)(curve.pillar_rates, theta_sigma_values))

    return theta_npv - base_npv


def bermudan_vega(
    cfg: BermudanSwaptionConfig,
    curve: ZeroCurve,
    calibration_targets,
    market_vol_bump: float = 0.0001,
) -> jax.Array:
    """
    Vega per basket instrument: NPV change for a `market_vol_bump` (1bp normal) move in one
    market swaption vol, others fixed. ORE bumps and recalibrates
    (`generateSwaptionVolScenarios`); here the bootstrap is differentiated instead.

    `cfg.hw_sigma` must be the `Sigma` that `calibrate_lgm_sigma` produced from
    `calibration_targets`, in the same order. Calibration is not rerun.

    Bucket `j`'s value s_j is the root of
    `g_j(s_0..s_j; v_j) = price_lgm_swaption(Sigma([s_0..s_j]), target_j) - bachelier(v_j)`.
    It depends on every earlier bucket, so a bump to v_i moves s_i and, through it, every
    later s_j. Differentiating g_j = 0 with respect to v_i:

        ds_j/dv_i = -( sum_{k<j} dg_j/ds_k * ds_k/dv_i + dg_j/dv_j * [i==j] ) / (dg_j/ds_j)

    solved by forward substitution over j (a lower-triangular Jacobian, O(n^2) gradients).
    Then `dNPV/dv_i = sum_j dNPV/ds_j * ds_j/dv_i`.

    Returns `[len(calibration_targets)]` Vegas, per `market_vol_bump`.
    """
    from dataclasses import replace as _replace
    from engine.calibration.basket import bachelier_swaption_price, price_lgm_swaption

    sigma = as_sigma(cfg.hw_sigma)
    n = sigma.values.shape[0]
    if n != len(calibration_targets):
        raise ValueError(
            "cfg.hw_sigma must be the Sigma calibrate_lgm_sigma produced from "
            f"calibration_targets, with one bucket per target; got {n} buckets "
            f"for {len(calibration_targets)} targets"
        )

    price_fn, sigma_values = _bermudan_price_fn(cfg, curve)
    d_npv_d_s = jax.jit(jax.grad(price_fn, argnums=1))(curve.pillar_rates, sigma_values)  # [n]

    # Work in the curve's dtype so a float32 risk run is not upcast.
    _dtype = curve.pillar_rates.dtype
    # Lower-triangular Jacobian J[j, i] = d(s_j)/d(v_i), by forward substitution over j.
    # The loop is sequential (row j reads rows < j); each row's gradients are jitted, and
    # rows are stacked once at the end rather than written one by one.
    rows = []

    for j, target_j in enumerate(calibration_targets):
        bucket_times_j = sigma.times[:j]

        def model_price_wrt_prefix(values_prefix, _j=j, _bucket_times=bucket_times_j, _target=target_j):
            return price_lgm_swaption(curve, cfg.hw_a, Sigma(times=_bucket_times, values=values_prefix), _target)

        # dg_j/ds_k for every k <= j in one gradient over the prefix [s_0..s_j].
        dg_j_ds = jax.jit(jax.grad(model_price_wrt_prefix))(sigma.values[: j + 1])  # [j+1]

        def market_price_wrt_v_j(v_j, _target=target_j):
            bumped = _replace(_target, market_vol=v_j)
            return bachelier_swaption_price(bumped, curve)

        # g_j := model_price - market_price, so dg_j/dv_j = -d(market_price)/dv_j.
        dg_j_dv_j = -jax.jit(jax.grad(market_price_wrt_v_j))(jnp.asarray(target_j.market_vol))

        if j > 0:
            prev = jnp.stack(rows)                       # [j, n], rows already computed
            cross_term = jnp.sum(dg_j_ds[:j, None] * prev, axis=0)
        else:
            cross_term = jnp.zeros((n,), dtype=_dtype)
        rows.append(-(cross_term.at[j].add(dg_j_dv_j)) / dg_j_ds[j])

    J = jnp.stack(rows)  # [n, n], lower-triangular by construction
    vega_per_unit_vol = d_npv_d_s @ J  # [n]
    return vega_per_unit_vol * market_vol_bump


# Demo
if __name__ == "__main__":
    from engine.simulation.demo_scenarios import EVAL_DATE

    pillar_times = [1.0, 2.0, 5.0, 10.0, 30.0]

    # --- Swap Delta/Gamma/Theta ---
    disc_curve = ZeroCurve.flat(0.030, pillar_times)
    fwd_curve = ZeroCurve.flat(0.035, pillar_times)
    swap_cfg = SwapConfig(
        notional=1_000_000.0, fixed_rate=0.032, payer=True,
        discount_curve_index=0, forward_curve_index=1,
        swap_tenor="5Y", evaluation_date=EVAL_DATE,
    )
    swap_greeks = swap_delta_gamma(swap_cfg, disc_curve, fwd_curve)
    print("--- Swap Greeks (5Y payer, notional $1MM) ---")
    print("Discount curve Delta ($/1bp):", [round(float(v), 2) for v in swap_greeks["discount_delta"]])
    print("Forward curve Delta ($/1bp): ", [round(float(v), 2) for v in swap_greeks["forward_delta"]])
    print("Discount curve Gamma:        ", [round(float(v), 6) for v in swap_greeks["discount_gamma"]])
    print("Theta (1 day):", round(swap_theta(swap_cfg, disc_curve, fwd_curve), 4))

    # --- European swaption Delta/Gamma/Theta ---
    from engine.simulation.market_model import ZeroCurveConfig

    swaption_curve = ZeroCurve.flat(0.03, pillar_times)
    swaption_cfg = SwaptionConfig(
        notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
        hw_a=0.03, hw_sigma=0.01,
        initial_zero_curve=ZeroCurveConfig(times=pillar_times, rates=[0.03] * len(pillar_times)),
        swap_tenor="5Y", forward_start=ORE.Period(3, ORE.Years), evaluation_date=EVAL_DATE,
    )
    swaption_greeks = swaption_delta_gamma(swaption_cfg, swaption_curve)
    print("\n--- European Swaption Greeks (3Y into 5Y payer, notional $1MM) ---")
    print("Delta ($/1bp):", [round(float(v), 2) for v in swaption_greeks["delta"]])
    print("Gamma:        ", [round(float(v), 6) for v in swaption_greeks["gamma"]])
    print("Theta (1 day):", round(swaption_theta(swaption_cfg, swaption_curve), 4))
