"""
European swaption pricing by Jamshidian's decomposition under Hull-White, as
`ORE.JamshidianSwaptionEngine`.

The underlying swap is built with ORE (`MakeVanillaSwap`). The swaption is an option on a
coupon bond: the fixed coupons plus the final notional, minus the notional at accrual start
(the floating leg at par). Hull-White bond prices are monotone in the short rate, so a
single critical rate r* splits it into zero-bond options with closed forms. One model and
one curve price both the swap and the option (`rate_factor_index`); like ORE, there is no
multi-curve version.

Differs from ORE:
  * ORE's default European swaption engine is Black/Bachelier on a volatility surface
    (`EuropeanSwaptionEngineBuilder`); this engine uses Hull-White Jamshidian instead, and
    matches `ORE.JamshidianSwaptionEngine` (tests/test_ore_parity.py,
    tests/test_trade_dates.py). The price is a model value for `hw_sigma`, not the market
    price ORE's default gives (I-46).
  * The formula is evaluated at every simulated (scenario, step), conditional on the
    simulated short rate r(t), to produce an NPV cube. It is zero from the exercise time
    on, on every path, even where the option was exercised (audit M-3).

A config holds its booked `exercise_date` and underlying dates; on or after the exercise
date the option is expired and worth 0 (ORE's `Instrument::isExpired`).

Known issue: `floating_spread` is accepted but ignored (the floating leg is taken at par).
ORE's `JamshidianSwaptionEngine` refuses a non-zero spread; see I-37 in docs/known-issues.md.
"""
from dataclasses import InitVar, dataclass, field
from functools import partial
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import ORE

from engine.simulation.market_model import ZeroCurveConfig
from engine.models.static_key import StaticKeyMixin
from engine.models.ore_builders import (
    SWAP_CALENDAR,
    TIME_AXIS_DAY_COUNTER,
    book_swap_dates,
    build_vanilla_swap,
    is_live,
)
from engine.portfolio.validation import _validate_common_fields, _validate_hw_sigma
from engine.models.hull_white import (
    A as _hw_A,
    B as _hw_B,
    ZeroCurve as _HwZeroCurve,
    bond_call as _bond_call,
    bond_option_sigma as _bond_option_sigma,
    bond_put as _bond_put,
)


def compute_hw_A(zero_times: np.ndarray, zero_rates: np.ndarray, t: np.ndarray, T: np.ndarray, a: float, sigma: float) -> np.ndarray:
    """`engine.models.hull_white.A` on NumPy arrays, for callers holding the curve as
    `zero_times`/`zero_rates` (tests and `_PreparedSwaption`)."""
    curve = _HwZeroCurve(pillar_times=jnp.asarray(zero_times), pillar_rates=jnp.asarray(zero_rates))
    return np.asarray(_hw_A(curve, jnp.asarray(t), jnp.asarray(T), a, sigma))


@dataclass
class SwaptionConfig:
    """
    One European swaption: the option to enter a vanilla swap on `exercise_date`.

    rate_factor_index: the simulated Hull-White factor that prices both the swap and the
        option. `hw_a`, `hw_sigma` and `initial_zero_curve` must match that factor's
        simulation parameters; they cannot be recovered from the paths.
    exercise_date / effective_date / maturity_date: the booked expiry and the underlying's
        schedule. They define the trade; `evaluation_date` only sets when it is priced.
    index_tenor_months / floating_spread: as in `SwapConfig` (but see the module
        docstring: the spread is currently ignored).

    Booking by tenor instead (resolved once, on `evaluation_date`, not stored):
      * swap_tenor: the underlying's length, e.g. "5Y";
      * forward_start: ORE.Period by which the underlying starts after spot;
      * exercise_lag_days: business days from `evaluation_date + forward_start` to the
        exercise date (default 2, so without a forward start the exercise date is the
        underlying's spot start).
    """
    notional: float
    fixed_rate: float
    payer: bool
    rate_factor_index: int
    hw_a: float
    hw_sigma: float
    initial_zero_curve: ZeroCurveConfig
    exercise_date: Optional[ORE.Date] = None
    effective_date: Optional[ORE.Date] = None
    maturity_date: Optional[ORE.Date] = None
    swap_tenor: InitVar[Optional[str]] = None
    index_tenor_months: int = 6
    floating_spread: float = 0.0
    forward_start: InitVar[Optional[ORE.Period]] = None
    exercise_lag_days: InitVar[Optional[int]] = None
    evaluation_date: ORE.Date = field(default_factory=lambda: ORE.Settings.instance().evaluationDate)

    def __post_init__(self, swap_tenor, forward_start, exercise_lag_days) -> None:
        _validate_common_fields(self.notional, self.fixed_rate, self.evaluation_date)
        if swap_tenor is not None:
            if self.exercise_date is not None:
                raise ValueError("give either swap_tenor or exercise_date/effective_date/maturity_date, not both")
            self.exercise_date = resolve_exercise_date(
                self.evaluation_date, forward_start, 2 if exercise_lag_days is None else exercise_lag_days)
        elif exercise_lag_days is not None:
            raise ValueError("exercise_lag_days is only meaningful with swap_tenor")
        book_swap_dates(self, swap_tenor, forward_start)
        if not isinstance(self.exercise_date, ORE.Date):
            raise TypeError(f"exercise_date must be an ORE.Date; got {self.exercise_date!r}")
        if not self.exercise_date < self.maturity_date:
            raise ValueError(
                f"exercise_date ({self.exercise_date}) must be before maturity_date ({self.maturity_date})")
        if self.hw_sigma is None:
            raise ValueError("hw_sigma is required for a European swaption")
        _validate_hw_sigma(self.hw_sigma)

    def is_expired(self) -> bool:
        """Exercise date on or before the evaluation date (ORE's `isExpired`)."""
        return not is_live(self.exercise_date, self.evaluation_date)


def resolve_exercise_date(trade_date: ORE.Date, forward_start, exercise_lag_days: int) -> ORE.Date:
    """Exercise date of a swaption booked by tenor: `exercise_lag_days` business days
    after `trade_date + forward_start`. Not measured back from the accrual start, which
    already includes the spot lag."""
    forward_start = forward_start if forward_start is not None else ORE.Period(0, ORE.Days)
    forward_start_date = SWAP_CALENDAR.advance(trade_date, forward_start)
    return SWAP_CALENDAR.advance(forward_start_date, exercise_lag_days, ORE.Days)


def _build_ore_swap(cfg: SwaptionConfig) -> ORE.VanillaSwap:
    """The ORE underlying swap (see `engine.models.ore_builders.build_vanilla_swap`)."""
    return build_vanilla_swap(
        notional=cfg.notional, fixed_rate=cfg.fixed_rate, payer=cfg.payer,
        effective_date=cfg.effective_date, maturity_date=cfg.maturity_date,
        index_tenor_months=cfg.index_tenor_months, floating_spread=cfg.floating_spread,
    )


@dataclass(frozen=True, eq=False)
class _PreparedSwaption(StaticKeyMixin):
    """Per-trade constant structure, resolved once by `prepare_swaption`; a `jax.jit`
    static argument (see `engine.models.static_key`)."""
    payer: bool
    notional: float
    exercise_time: float               # T0, year-fraction from evaluation_date
    accrual_start_time: float          # T_start, the underlying swap's own first accrual date
    fixed_cashflow_times: np.ndarray   # [N] year-fractions from evaluation_date
    fixed_cashflow_amounts: np.ndarray  # [N]
    rate_factor_index: int
    hw_a: float
    hw_sigma: float
    zero_times: np.ndarray
    zero_rates: np.ndarray


def prepare_swaption(cfg: SwaptionConfig) -> _PreparedSwaption:
    """
    Build the ORE underlying and extract the fixed cashflow times and amounts, the accrual
    start time and the exercise time, in years from `cfg.evaluation_date`.

    The floating leg is not read. On one curve with no spread it is worth
    N * (P(T0, T_start) - P(T0, T_end)). T_start is the first fixed accrual start (ORE's
    `valueTime`, `fixedResetDates[0]`), which can be after the exercise time.
    """
    swap = _build_ore_swap(cfg)
    today = cfg.evaluation_date
    accrual_start_date = ORE.as_fixed_rate_coupon(swap.fixedLeg()[0]).accrualStartDate()

    fixed_times, fixed_amounts = [], []
    for cf in swap.fixedLeg():
        c = ORE.as_fixed_rate_coupon(cf)
        fixed_times.append(TIME_AXIS_DAY_COUNTER.yearFraction(today, c.date()))
        fixed_amounts.append(c.amount())

    return _PreparedSwaption(
        payer=cfg.payer,
        notional=swap.fixedNominals()[0] if swap.fixedNominals() else swap.nominal(),
        exercise_time=TIME_AXIS_DAY_COUNTER.yearFraction(today, cfg.exercise_date),
        accrual_start_time=TIME_AXIS_DAY_COUNTER.yearFraction(today, accrual_start_date),
        fixed_cashflow_times=np.array(fixed_times),
        fixed_cashflow_amounts=np.array(fixed_amounts),
        rate_factor_index=cfg.rate_factor_index,
        hw_a=cfg.hw_a,
        hw_sigma=cfg.hw_sigma,
        zero_times=np.asarray(cfg.initial_zero_curve.times, dtype=np.float64),
        zero_rates=np.asarray(cfg.initial_zero_curve.rates, dtype=np.float64),
    )


# Hull-White closed forms come from `engine.models.hull_white`.
def _bisect_rstar(coupon_bond_value_fn, t_shape, iterations: int, dtype=jnp.float64) -> jax.Array:
    """Bisection for r* (forward value only; see `_solve_rstar` for the gradient).

    Starts on [-2, 2], shifts the window by its width up to 20 times until it brackets a
    sign change, then bisects. `dtype` is explicit because `jnp.ones` without it follows
    JAX's global default and would upcast a float32 computation."""
    lo = -jnp.ones(t_shape, dtype=dtype) * 2.0
    hi = jnp.ones(t_shape, dtype=dtype) * 2.0

    def expand_body(_, carry):
        lo, hi = carry
        val_lo = coupon_bond_value_fn(lo)
        val_hi = coupon_bond_value_fn(hi)
        # The value decreases in r: shift the window up if val_hi > 0, down if val_lo < 0.
        width = hi - lo
        shift_up = val_hi > 0.0
        shift_down = val_lo < 0.0
        new_lo = jnp.where(shift_up, hi, jnp.where(shift_down, lo - width, lo))
        new_hi = jnp.where(shift_up, hi + width, jnp.where(shift_down, lo, hi))
        return (new_lo, new_hi)

    lo, hi = jax.lax.fori_loop(0, 20, expand_body, (lo, hi))

    def body(_, carry):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        val_mid = coupon_bond_value_fn(mid)
        lo = jnp.where(val_mid > 0.0, mid, lo)
        hi = jnp.where(val_mid > 0.0, hi, mid)
        return (lo, hi)

    lo, hi = jax.lax.fori_loop(0, iterations, body, (lo, hi))
    return 0.5 * (lo + hi)


def _solve_rstar(coupon_bond_value_fn, params, t_shape, iterations: int = 100) -> jax.Array:
    """
    Critical short rate r*(scenario, step) at the exercise time: where the signed coupon
    bond (fixed coupons, plus final notional, minus the notional at T_start) is worth 0.

    `coupon_bond_value_fn(r, params)`: `params` holds everything the result must be
    differentiable in (e.g. `A_T0_Ti`, `all_amounts`). It is an explicit argument because
    `jax.custom_jvp` attaches tangents only to explicit primals. Values that need no
    gradient can be closed over.

    Bisection with a fixed iteration count, so it vectorizes under jit.

    Gradient: differentiating through bisection gives 0 (the comparison has no
    derivative). A `custom_jvp` applies the implicit function theorem instead,
    dr* = -(df/dparams . v) / (df/dr). Because the rule is itself differentiable, second
    derivatives (Gamma via `jax.hessian`) are also correct, which a stop-gradient Newton
    correction is not. `df/dr` is taken as the gradient of the batch sum, which is exact
    because each batch element depends only on its own inputs.
    """
    # Take rstar's dtype from params (see `_bisect_rstar`).
    dtype = jnp.result_type(*[leaf for leaf in jax.tree_util.tree_leaves(params)])

    @jax.custom_jvp
    def solve(p):
        f = lambda r: coupon_bond_value_fn(r, p)
        rstar = _bisect_rstar(f, t_shape, iterations, dtype=dtype)
        return jax.lax.stop_gradient(rstar)

    @solve.defjvp
    def solve_jvp(primals, tangents):
        p, = primals
        p_dot, = tangents
        rstar_val = solve(p)
        df_dr = jax.grad(lambda r: jnp.sum(coupon_bond_value_fn(r, p)))(rstar_val)
        _, df_dparams_dot = jax.jvp(lambda pp: coupon_bond_value_fn(rstar_val, pp), (p,), (p_dot,))
        rstar_dot = -df_dparams_dot / df_dr
        return rstar_val, rstar_dot

    return solve(params)


@partial(jax.jit, static_argnums=2)
def _price_one_swaption(
    hw_paths: jax.Array, step_times: jax.Array, swaption: _PreparedSwaption,
) -> jax.Array:
    """
    [Scenarios, TimeSteps] NPV of one prepared swaption, conditional on the simulated short
    rate r(t) at each step t.

      1. Solve r* at the exercise time T0. A(T0, Ti) depends only on T0 and today's curve.
      2. Each cashflow of the coupon bond (coupons, final notional, and the negative
         notional at T_start) becomes a zero-bond option struck at K_i = P(T0, Ti; r*),
         valued at t given r(t) by the Black-on-bond formula.
      3. Payer = sum of puts, receiver = sum of calls, weighted by the signed amounts.

    Zero for t >= T0.
    """
    a = swaption.hw_a
    sigma = swaption.hw_sigma
    T0 = swaption.exercise_time
    T_start = swaption.accrual_start_time
    cf_times = swaption.fixed_cashflow_times
    cf_amounts = jnp.asarray(swaption.fixed_cashflow_amounts, dtype=hw_paths.dtype)
    notional = swaption.notional
    # Coupons, final notional (paid), notional at T_start (received).
    all_times = np.concatenate([cf_times, cf_times[-1:], [T_start]])
    all_amounts = jnp.concatenate([
        cf_amounts,
        jnp.asarray([notional, -notional], dtype=hw_paths.dtype),
    ])

    # A(T0, Ti) depends only on T0 and today's curve. Use `hull_white.A` directly;
    # `compute_hw_A` returns NumPy and would break tracing.
    _curve = _HwZeroCurve(
        pillar_times=jnp.asarray(swaption.zero_times),
        pillar_rates=jnp.asarray(swaption.zero_rates),
    )
    A_T0_Ti = jnp.asarray(
        _hw_A(_curve, jnp.full_like(jnp.asarray(all_times), T0), jnp.asarray(all_times), a, sigma),
        dtype=hw_paths.dtype,
    )
    B_T0_Ti = _hw_B(T0, jnp.asarray(all_times, dtype=hw_paths.dtype), a)  # [N+1]

    def coupon_bond_value(rstar, params):
        # rstar: [S, T] -> [S, T]. Only A_T0_Ti (curve-dependent) and the amounts are
        # differentiated through; B depends only on mean reversion.
        A_T0_Ti_p, all_amounts_p = params
        prices = A_T0_Ti_p[None, None, :] * jnp.exp(-B_T0_Ti[None, None, :] * rstar[..., None])
        return jnp.sum(prices * all_amounts_p[None, None, :], axis=-1)

    r_t = hw_paths[:, :, swaption.rate_factor_index]  # [S, T]
    num_scenarios, num_steps = r_t.shape

    rstar = _solve_rstar(coupon_bond_value, (A_T0_Ti, all_amounts), (num_scenarios, num_steps))
    K = A_T0_Ti[None, None, :] * jnp.exp(-B_T0_Ti[None, None, :] * rstar[..., None])  # [S,T,N+1] strikes

    # A(t, Ti) and A(t, T0) per step (independent of the scenario).
    _step_times_j = jnp.asarray(step_times)
    _all_times_j = jnp.asarray(all_times)
    A_t_Ti = jnp.asarray(
        _hw_A(_curve, _step_times_j[:, None], _all_times_j[None, :], a, sigma),
        dtype=hw_paths.dtype,
    )  # [TimeSteps, N+1]
    A_t_T0 = jnp.asarray(
        _hw_A(_curve, _step_times_j, jnp.full_like(_step_times_j, T0), a, sigma),
        dtype=hw_paths.dtype,
    )  # [TimeSteps]

    B_t_Ti = _hw_B(step_times[:, None], jnp.asarray(all_times, dtype=hw_paths.dtype)[None, :], a)  # [T, N+1]
    B_t_T0 = _hw_B(step_times, T0, a)  # [T]

    P_t_Ti = A_t_Ti[None, :, :] * jnp.exp(-B_t_Ti[None, :, :] * r_t[:, :, None])  # [S,T,N+1]
    P_t_T0 = A_t_T0[None, :] * jnp.exp(-B_t_T0[None, :] * r_t)  # [S,T]

    sigma_p = _bond_option_sigma(
        T0, jnp.asarray(all_times, dtype=hw_paths.dtype)[None, :], step_times[:, None], a, sigma,
    )  # [T, N+1]; zero at t == T0 or for a leg maturing at T0 (handled by bond_call/put)

    bond_fn = _bond_put if swaption.payer else _bond_call
    per_leg = bond_fn(
        P_t_T0[:, :, None], P_t_Ti, K, sigma_p[None, :, :],
    )  # [S, T, N+1]
    npv_unexpired = jnp.sum(per_leg * all_amounts[None, None, :], axis=-1)  # [S, T]

    not_yet_expired = step_times[None, :] < T0
    return jnp.where(not_yet_expired, npv_unexpired, 0.0)


def price_swaptions(hw_paths: jax.Array, step_times: jax.Array, swaption_configs: List[SwaptionConfig]) -> jax.Array:
    """
    NPV cube `[Scenarios, TimeSteps, Trades]` for `swaption_configs`; zero from each
    exercise date on.

    hw_paths: `[Scenarios, TimeSteps, NumHW]` short rates, from
        `engine.simulation.generate_paths(...)["rates"]`.
    step_times: `[TimeSteps]` times of those steps (`time_grid[1:]`).
    """
    step_times_jax = jnp.asarray(step_times, dtype=hw_paths.dtype)
    per_trade = [
        # An expired option is worth 0 at every step (ORE's isExpired).
        jnp.zeros(hw_paths.shape[:2], dtype=hw_paths.dtype) if cfg.is_expired()
        else _price_one_swaption(hw_paths, step_times_jax, prepare_swaption(cfg))
        for cfg in swaption_configs
    ]
    return jnp.stack(per_trade, axis=-1)


# Demo
if __name__ == "__main__":
    from engine.simulation.market_model import generate_paths
    from engine.simulation.demo_scenarios import EVAL_DATE, swaption_demo_config

    config = swaption_demo_config()
    market_cubes = generate_paths(config)
    step_times = jnp.array(config.time_grid[1:], dtype=jnp.float64)

    # Exercise in 3Y into a 2Y swap: live at the early steps, zero after exercise.
    swaption_cfg = SwaptionConfig(
        notional=1_000_000.0,
        fixed_rate=0.030,
        payer=True,
        rate_factor_index=0,
        hw_a=config.rates.mean_reversion[0],
        hw_sigma=float(np.sqrt(config.joint_covariance[1][1])),
        initial_zero_curve=ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[0.03] * 6),
        swap_tenor="2Y",
        forward_start=ORE.Period(3, ORE.Years),
        evaluation_date=EVAL_DATE,
    )

    npv_cube = price_swaptions(market_cubes["rates"], step_times, [swaption_cfg])
    print("Swaption NPV cube shape:", npv_cube.shape)
    for i, t in enumerate(config.time_grid[1:]):
        print(f"  t={t:.2f}: mean NPV across scenarios = {float(jnp.mean(npv_cube[:, i, 0])):.2f}")
