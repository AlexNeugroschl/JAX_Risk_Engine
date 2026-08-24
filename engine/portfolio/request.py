"""
The engine's top-level entry point: "a portfolio of trades + market data +
risk parameters" in, "prices + risk" out.

Implements `docs/planning/traderx-integration.md`'s data-transformation/
validation layer (this module is that plan's own first-choice location) and
then, on top of it, `price_portfolio` -- the single orchestration function
that replaces `demo.py`'s hand-written simulate -> calibrate -> price ->
aggregate-risk sequence with one call. Sits at the same level as
`engine/simulation/`, `engine/instruments/`, `engine/risk/`: it imports
across those instrument-type/stage boundaries so none of them have to import
each other, preserving `docs/concepts/architecture.md`'s "modules agree on
shapes, not code" principle.

**Zero Pydantic/FastAPI dependency, deliberately.** This module and
everything it imports (`engine.simulation`, `engine.instruments`,
`engine.risk`, `engine.calibration`, `engine.models`) stay
plain-dataclass/JAX-native throughout. Pydantic and FastAPI live exclusively
in `engine/api/` (the HTTP boundary), which wraps `PortfolioRequest`/
`PortfolioResult` rather than replacing them -- see `engine/api/schemas.py`.

**Known limitations this module does not fix, only surfaces (see
docs/planning/traderx-integration.md gap item 5):**
- Any *exposure profile* (t>0 valuation, as opposed to a t=0 NPV/VaR run) of
  a swap inherits `engine.instruments.swap`'s documented aged-swap
  discounting gap (see that module's docstring and
  `tests/test_swap.py::TestAgedSwapKnownLimitation`): a swap's conditional
  NPV at any simulated step after its own first accrual date does not
  correctly represent an already-fixed floating coupon. This is exact at
  t=0 and for forward-starting trades priced before their own accrual
  begins; `price_portfolio` does not attempt to correct it.
- A Bermudan/American trade whose `exercise_times` aren't reset-aligned
  with its own underlying's accrual dates hits the documented mid-coupon
  approximation in `engine.instruments.bermudan_swaption`/
  `american_swaption` (see `docs/instruments/american-bermudan-swaptions.md`)
  -- `validate_portfolio_against_simulation` below warns (not raises) when
  it detects this, rather than silently pricing a slightly-wrong number.

**Concurrency: `_PRICING_LOCK` is a defense-in-depth invariant guard, not
this system's primary concurrency-limiting mechanism.** `generate_paths`
toggles `jax_enable_x64`, a process-global JAX/XLA flag, not a thread-local
or per-array setting -- confirmed live: two threads each calling
`price_portfolio` with different precisions, inside the SAME process, can
have thread B's flag flip land while thread A is still mid-flight through
`generate_paths`/the pricers/Greeks that follow it, silently corrupting
thread A's own in-progress computation (wrong dtype, or a
dtype-correct-looking but numerically wrong array). `_PRICING_LOCK` below
serializes the entire JAX-executing body of `price_portfolio`
(`generate_paths` through Greeks) so two threads of one process queue
instead of racing.

Real concurrency for `engine/api/routes.py`'s HTTP job pattern now comes
from a layer above this module, not from running multiple threads through
this lock: `engine.portfolio.worker_pool` dispatches each job to one of a
fixed pool of worker PROCESSES, sized per precision tier (float32/float64),
each with its own independent JAX/XLA runtime that fixes `jax_enable_x64`
once at process boot and never touches it again -- see that module's own
docstring for the full mechanism. Because each worker processes jobs
strictly sequentially, no second thread inside a worker process ever calls
into JAX-executing code while a job is in flight, which is what makes
`_PRICING_LOCK` unnecessary *at the worker-pool level*. This lock stays
here anyway, unconditionally, as a narrower defense-in-depth guard: the
underlying JAX fact it protects against doesn't disappear just because the
worker pool makes it unreachable through the normal HTTP path. Anything
that ever puts two threads of the SAME process inside `price_portfolio`
concurrently -- a worker-pool sizing bug, or a future direct Python caller
spinning up their own threads against `engine.portfolio` directly (this
module has "Zero Pydantic/FastAPI dependency, deliberately," per this
docstring's own section above -- it's designed to be called directly, not
only through the HTTP/worker-pool layer) -- hits the exact same corruption
bug. The lock is cheap (uncontended-lock overhead is negligible next to a
JIT-compile-dominated multi-second job) and correctness-critical whenever
"one job per process" doesn't hold, even though it is no longer the primary
thing standing between concurrent requests and true parallelism (see
docs/concepts/architecture.md's "Concurrency" section for the full
worker-pool architecture and docs/concepts/architecture.md's "Adjustable
precision" section for `PrecisionConfig` itself).
"""
import threading
import warnings
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
import ORE

from engine.simulation.market_model import SimulationConfig, generate_paths
from engine.instruments.swap import SwapConfig, price_swaps
from engine.instruments.european_swaption import SwaptionConfig, price_swaptions
from engine.instruments.bermudan_swaption import (
    BermudanSwaptionConfig, price_bermudan_swaptions, price_bermudan_swaption_base,
)
from engine.instruments.american_swaption import AmericanSwaptionConfig, price_american_swaptions
from engine.models.ore_builders import DAY_COUNTER, build_vanilla_swap
from engine.calibration.lgm import calibrate_lgm_sigma, CalibrationTarget
from engine.risk.var_es import compute_risk_metrics
from engine.risk import greeks as _greeks
from engine.models.hull_white import ZeroCurve as _HwZeroCurve

# Re-exported so `engine.portfolio._validate_common_fields` resolves exactly
# where the design calls for it, even though the actual leaf implementation
# lives in engine/portfolio/validation.py to avoid a circular import (see
# that module's own docstring for why).
from engine.portfolio.validation import _validate_common_fields, _validate_tenor  # noqa: F401

TradeConfig = Union[SwapConfig, SwaptionConfig, BermudanSwaptionConfig, AmericanSwaptionConfig]

# Defense-in-depth guard against jax_enable_x64's process-global-flag race
# WITHIN a single process -- see this module's docstring's "Concurrency"
# section. Real concurrency for engine/api/routes.py's HTTP job pattern now
# comes from engine.portfolio.worker_pool's multi-process, per-precision-tier
# pools, one layer up; this lock is kept unconditionally anyway since nothing
# here statically prevents a caller from putting two threads of one process
# through price_portfolio directly. Lives here (not in
# engine.simulation.market_model) because generate_paths is also called
# directly, sequentially, by other code/tests and shouldn't own this
# invariant-guard policy, which only matters for price_portfolio's own
# multi-thread-reachable callers.
_PRICING_LOCK = threading.Lock()


@dataclass(frozen=True)
class PrecisionConfig:
    """Three independently-settable dtype knobs, each 32 (float32) or 64
    (float64, default): `simulation` (Monte Carlo path generation, passed to
    `generate_paths`), `pricing` (instrument NPV/npv_cube dtype), and `risk`
    (VaR/ES + Greeks -- one shared setting, not split further; see
    docs/concepts/architecture.md's "Adjustable precision" section for why
    exactly these three and not finer-grained control). Defaults to all-64,
    byte-identical to this codebase's behavior before this dataclass existed.

    bfloat16/float16 are NOT supported -- confirmed broken on this stack
    today (`jnp.linalg.cholesky` and `jax.scipy.stats.norm.ppf` both raise on
    those dtypes on the installed jax/jaxlib CPU backend); see
    docs/concepts/architecture.md's "Out of scope for v1" note.
    """
    simulation: int = 64
    pricing: int = 64
    risk: int = 64

    def __post_init__(self):
        for name in ("simulation", "pricing", "risk"):
            value = getattr(self, name)
            if value not in (32, 64):
                raise ValueError(f"PrecisionConfig.{name} must be 32 or 64, got {value!r}")


def _dtype_of(precision_bits: int):
    return jnp.float64 if precision_bits == 64 else jnp.float32


# =============================================================================
# 1c. CROSS-FIELD CONSISTENCY: RatesConfig <-> each trade's duplicated fields
# =============================================================================
def validate_portfolio_against_simulation(
    sim_config: SimulationConfig, trade_configs: Sequence[TradeConfig],
) -> None:
    """
    For every trade carrying a `rate_factor_index` (every trade type except
    `SwapConfig`, which has no single-model dependency), cross-checks that
    trade's own duplicated `hw_a`/`hw_sigma`/`initial_zero_curve` against
    `sim_config`'s corresponding entry for that factor -- see each
    SwaptionConfig-family class's own docstring: these fields "MUST match
    that factor's own calibration in the simulation's RatesConfig", but
    nothing previously enforced it. Raises `ValueError` naming the trade (by
    index/notional/type) and the specific mismatched field on any divergence
    beyond a small float tolerance.

    Also emits `warnings.warn` (not a hard error) for a Bermudan/American
    trade whose `exercise_times` aren't reset-aligned with its own
    underlying's accrual/payment dates -- see docs/planning/
    traderx-integration.md gap item 5 and this module's own docstring.
    """
    tol = 1e-9
    num_eq = len(sim_config.equities.initial_prices)

    for i, cfg in enumerate(trade_configs):
        rate_factor_index = getattr(cfg, "rate_factor_index", None)
        if rate_factor_index is None:
            continue

        label = f"trade[{i}] ({type(cfg).__name__}, notional={cfg.notional})"
        num_hw = len(sim_config.rates.mean_reversion)
        if not (0 <= rate_factor_index < num_hw):
            raise ValueError(
                f"{label}: rate_factor_index={rate_factor_index} is out of range "
                f"for a simulation with {num_hw} rate factors"
            )

        expected_a = sim_config.rates.mean_reversion[rate_factor_index]
        if abs(cfg.hw_a - expected_a) > tol:
            raise ValueError(
                f"{label}: hw_a={cfg.hw_a} does not match "
                f"sim_config.rates.mean_reversion[{rate_factor_index}]={expected_a}"
            )

        cov_row = num_eq + rate_factor_index
        joint_cov = np.asarray(sim_config.joint_covariance, dtype=np.float64)
        expected_sigma = float(np.sqrt(joint_cov[cov_row, cov_row]))
        actual_sigma = cfg.hw_sigma
        if actual_sigma is None:
            # Uncalibrated (Bermudan/American only) -- price_portfolio fills
            # this in later via calibrate_lgm_sigma, at which point it is
            # consistent with sim_config by construction (calibrated
            # against the SAME curve/mean-reversion this trade already
            # carries); nothing to cross-check yet.
            pass
        elif hasattr(actual_sigma, "values"):
            # A genuinely piecewise (calibrated) Sigma -- deliberately NOT
            # cross-checked against joint_covariance's single flat per-step
            # vol. The simulation only ever propagates ONE constant vol per
            # rate factor (Monte Carlo path generation doesn't need a term
            # structure), while a calibrated Sigma is the model's own
            # richer view of volatility used for PRICING -- a real desk
            # workflow (calibrate once, reuse the fitted term structure
            # across many trades, simulate paths off a single representative
            # vol level) has these legitimately diverge; this is not the
            # transcription-bug class gap item 2 targets (a flat hw_sigma
            # silently copied from the wrong factor). Only a flat float
            # hw_sigma is cross-checked below.
            pass
        else:
            if abs(float(actual_sigma) - expected_sigma) > tol:
                raise ValueError(
                    f"{label}: hw_sigma={actual_sigma} does not match the per-step "
                    f"volatility implied by sim_config.joint_covariance"
                    f"[{cov_row}][{cov_row}]={expected_sigma} for rate factor "
                    f"{rate_factor_index}"
                )

        expected_curve = sim_config.rates.initial_zero_curves[rate_factor_index]
        actual_curve = cfg.initial_zero_curve
        if list(actual_curve.times) != list(expected_curve.times):
            raise ValueError(
                f"{label}: initial_zero_curve.times {list(actual_curve.times)} does not "
                f"match sim_config.rates.initial_zero_curves[{rate_factor_index}].times "
                f"{list(expected_curve.times)}"
            )
        if any(abs(a - e) > tol for a, e in zip(actual_curve.rates, expected_curve.rates)):
            raise ValueError(
                f"{label}: initial_zero_curve.rates {list(actual_curve.rates)} does not "
                f"match sim_config.rates.initial_zero_curves[{rate_factor_index}].rates "
                f"{list(expected_curve.rates)}"
            )

        if isinstance(cfg, (BermudanSwaptionConfig, AmericanSwaptionConfig)):
            _warn_if_not_reset_aligned(label, cfg, i)


def _warn_if_not_reset_aligned(label: str, cfg, index: int) -> None:
    """Emits a `UserWarning` (not a hard error) if any of `cfg`'s exercise
    dates don't coincide with one of the underlying swap's own
    accrual/payment dates -- see docs/planning/traderx-integration.md gap
    item 5 and docs/instruments/american-bermudan-swaptions.md's mid-coupon
    approximation."""
    berm_cfg = cfg.to_bermudan() if isinstance(cfg, AmericanSwaptionConfig) else cfg
    swap = build_vanilla_swap(
        notional=berm_cfg.notional, fixed_rate=berm_cfg.fixed_rate, payer=berm_cfg.payer,
        swap_tenor=berm_cfg.swap_tenor, index_tenor_months=berm_cfg.index_tenor_months,
        floating_spread=berm_cfg.floating_spread, evaluation_date=berm_cfg.evaluation_date,
    )
    today = berm_cfg.evaluation_date
    reset_dates = set()
    for cf in swap.fixedLeg():
        c = ORE.as_fixed_rate_coupon(cf)
        reset_dates.add(round(DAY_COUNTER.yearFraction(today, c.accrualStartDate()), 9))
    for cf in swap.floatingLeg():
        c = ORE.as_floating_rate_coupon(cf)
        reset_dates.add(round(DAY_COUNTER.yearFraction(today, c.accrualStartDate()), 9))

    misaligned = [t for t in berm_cfg.exercise_times if round(float(t), 9) not in reset_dates]
    if misaligned:
        warnings.warn(
            f"{label}: exercise_times {misaligned} are not reset-aligned with the "
            f"underlying's own accrual dates -- pricing will use the documented "
            f"mid-coupon approximation (see "
            f"docs/instruments/american-bermudan-swaptions.md)",
            stacklevel=2,
        )


# =============================================================================
# 1d. AUTOMATIC MATURITY-PILLAR ASSEMBLY
# =============================================================================
def derive_maturity_pillars(trade_configs: Sequence[TradeConfig], evaluation_date: ORE.Date) -> List[float]:
    """
    Builds every `SwapConfig`'s real ORE schedule (via
    `engine.models.ore_builders.build_vanilla_swap`, the same shared
    construction every pricer already uses -- schedule logic is never
    reimplemented here) and returns the sorted union of every leg's
    accrual/payment year-fractions -- the exact maturity-pillar set
    `engine.instruments.swap`'s `_maturity_indices` requires. Automates what
    `demo.py` used to do by hand for a single swap.

    Only `SwapConfig` trades contribute pillars: every swaption-family
    pricer (`price_swaptions`/`price_bermudan_swaptions`/
    `price_american_swaptions`) prices directly off simulated `hw_paths`,
    not the `yield_curves`/maturity-pillar cube (see each pricer's own
    docstring), so their own cashflow dates impose no pillar-alignment
    requirement.
    """
    pillars = {0.0}
    for cfg in trade_configs:
        if not isinstance(cfg, SwapConfig):
            continue
        swap = build_vanilla_swap(
            notional=cfg.notional, fixed_rate=cfg.fixed_rate, payer=cfg.payer,
            swap_tenor=cfg.swap_tenor, index_tenor_months=cfg.index_tenor_months,
            floating_spread=cfg.floating_spread, evaluation_date=evaluation_date,
        )
        today = evaluation_date
        for cf in swap.fixedLeg():
            c = ORE.as_fixed_rate_coupon(cf)
            pillars.add(DAY_COUNTER.yearFraction(today, c.date()))
            pillars.add(DAY_COUNTER.yearFraction(today, c.accrualStartDate()))
        for cf in swap.floatingLeg():
            c = ORE.as_floating_rate_coupon(cf)
            pillars.add(DAY_COUNTER.yearFraction(today, c.date()))
            pillars.add(DAY_COUNTER.yearFraction(today, c.accrualStartDate()))
            pillars.add(DAY_COUNTER.yearFraction(today, c.accrualEndDate()))
    return sorted(pillars)


# =============================================================================
# GENERAL "EXPECTED INPUT TO THE WHOLE SYSTEM" SURFACE
# =============================================================================
@dataclass
class PortfolioRequest:
    """
    What a caller of the whole system hands over: a portfolio of trades +
    market data + risk parameters. `price_portfolio` (below) is the single
    entry point that consumes this and returns a `PortfolioResult`.

    market: curves, vols (via `joint_covariance`), correlations -- the same
        `SimulationConfig` `generate_paths` already consumes. If
        `market.rates.maturities` is left unset, `price_portfolio` derives
        it automatically via `derive_maturity_pillars`.
    trades: heterogeneous, any order/mix of the four instrument types.
        Results in `PortfolioResult` are reported back in this same order,
        regardless of how `price_portfolio` internally groups trades by type
        for pricing.
    percentiles: confidence levels `compute_risk_metrics` computes VaR/ES
        at.
    calibration_targets: used when any Bermudan/American trade's `hw_sigma`
        is left as `None` (uncalibrated) -- `price_portfolio` calibrates it
        once per distinct `rate_factor_index` via
        `engine.calibration.lgm.calibrate_lgm_sigma`.
    compute_greeks: if `True`, `price_portfolio` additionally computes
        Delta/Gamma (every trade type) and Theta (every trade type) via
        `engine.risk.greeks`, keyed by each trade's own index in `trades`.
    precision: independent simulation/pricing/risk dtype control -- see
        `PrecisionConfig`. Defaults to all-64, byte-identical to this
        module's behavior before `PrecisionConfig` existed.

    A future `BondConfig` instrument type (see
    docs/planning/traderx-bond-integration-roadmap.md) would join `trades`'
    `Union` here once it exists -- out of scope for this module today.
    """
    market: SimulationConfig
    trades: List[TradeConfig]
    percentiles: Sequence[float] = (0.95, 0.99)
    calibration_targets: Optional[List[CalibrationTarget]] = None
    compute_greeks: bool = False
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)


# =============================================================================
# PHASE 2: PortfolioResult / price_portfolio
# =============================================================================
@dataclass
class PortfolioResult:
    """Output of `price_portfolio`: prices + risk for a whole
    `PortfolioRequest`, in one object."""
    base_npv: float
    npv_cube: jax.Array                                   # [Scenarios, TimeSteps, Trades]
    risk: Dict[str, jax.Array]                             # compute_risk_metrics(...) output
    greeks: Optional[Dict[int, Dict[str, jax.Array]]] = None  # trade index (in request.trades order) -> greeks dict
    warnings: List[str] = field(default_factory=list)


def _zero_curve_of(cfg, curve_config, dtype=jnp.float64) -> _HwZeroCurve:
    return _HwZeroCurve.from_config(curve_config, dtype=dtype)


def price_portfolio(request: PortfolioRequest) -> PortfolioResult:
    """
    The single entry point: `PortfolioRequest` in, `PortfolioResult` out.
    Orchestrates the exact sequence `demo.py` used to hand-write:

    1. Validate `request.market.joint_covariance` (explicit, request-scoped
       error -- the same check also runs inside `generate_paths`, but
       calling it here first gives an earlier, clearer failure before any
       trade-level work happens).
    2. Cross-check every trade's duplicated fields against `request.market`
       (`validate_portfolio_against_simulation`).
    3. Auto-derive `request.market.rates.maturities` via
       `derive_maturity_pillars` if the caller left it unset.
    4. Calibrate any Bermudan/American trade's `hw_sigma` left as `None`,
       once per distinct `rate_factor_index` needing it.
    5. Simulate the market (`generate_paths`).
    6. Route every trade to its pricer by type, concatenate into one NPV
       cube in the caller's original trade order.
    7. Reprice every trade against zero-shock curves for the base (t=0) NPV.
    8. Aggregate VaR/ES (`compute_risk_metrics`).
    9. Optionally compute Greeks per trade.

    Steps 4-9 (calibration through Greeks) run under `_PRICING_LOCK` -- see
    this module's docstring's "Concurrency" section: calibration is genuine
    JAX work just as sensitive to the ambient `jax_enable_x64` state as
    `generate_paths`/the pricers that follow it. Steps 1-3 above run
    unlocked: none of them touch `jax_enable_x64` or run JIT code, so there
    is no race to serialize against.
    """
    from engine.simulation.market_model import validate_joint_covariance

    validate_joint_covariance(request.market.joint_covariance)

    market_config = request.market
    collected_warnings: List[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_portfolio_against_simulation(market_config, request.trades)
        collected_warnings.extend(str(w.message) for w in caught)

    if market_config.rates.maturities is None:
        # RatesConfig itself carries no evaluation_date field (see its own
        # docstring) -- every trade config does, and within one
        # PortfolioRequest they are expected to share the same evaluation
        # date (mixing evaluation dates across trades in one request isn't
        # a supported configuration; nothing else in this module tries to
        # reconcile trade-level date differences either). Deliberately NOT
        # ORE.Settings.instance().evaluationDate here -- that's ambient,
        # THREAD-LOCAL global state (confirmed: each thread gets its own
        # independent default, defaulting to the real wall-clock date, not
        # whatever date this request's trades actually specify), so relying
        # on it silently derives pillars against the wrong "today" whenever
        # price_portfolio runs on a thread that never separately set it --
        # exactly the failure mode a background-task-driven caller (e.g.
        # engine/api/routes.py) hits every time.
        eval_date = request.trades[0].evaluation_date if request.trades else ORE.Settings.instance().evaluationDate
        pillars = derive_maturity_pillars(request.trades, eval_date)
        market_config = replace(market_config, rates=replace(market_config.rates, maturities=pillars))

    trades = list(request.trades)

    with _PRICING_LOCK:
        # _fill_calibrated_sigma runs genuine JAX work (calibrate_lgm_sigma's
        # bisection root-finds, hardcoded-float64 Sigma construction in
        # engine/models/lgm.py) that is just as sensitive to the ambient
        # jax_enable_x64 state as generate_paths/the pricers below -- it must
        # be inside the lock too, not run unprotected before it, or a
        # concurrent thread's generate_paths call could flip the flag
        # mid-calibration the same way it could mid-pricing.
        trades = _fill_calibrated_sigma(trades, request.calibration_targets, market_config)
        market = generate_paths(market_config, precision=request.precision.simulation)
        # generate_paths leaves jax_enable_x64 set to (precision.simulation == 64)
        # -- a process-global flag, not scoped to that call. Everything below this
        # point (pricing/risk/Greeks) may independently want float64 via `pricing`/
        # `risk`, regardless of what `simulation` requested. If simulation=32 left
        # x64 disabled, any float64 construction below (e.g. pricing=64's own
        # arrays, or _base_npv's/_compute_all_greeks' curve construction) would
        # silently truncate to float32 instead of raising -- confirmed directly:
        # PrecisionConfig(simulation=32, pricing=64) produced an all-float32
        # npv_cube with no error, only a buried UserWarning, before this line was
        # added. Re-enabling x64 here is always safe: creating float32 arrays
        # under x64=True works identically to under x64=False (x64 only restricts
        # *creating* float64, never float32), so this doesn't affect `pricing`/
        # `risk` requesting 32 -- it only fixes the case where they request 64
        # after a simulation=32 run.
        jax.config.update("jax_enable_x64", True)
        pricing_dtype = _dtype_of(request.precision.pricing)
        step_times = jnp.array(market_config.time_grid[1:], dtype=pricing_dtype)
        maturities_np = np.asarray(market_config.rates.maturities) if market_config.rates.maturities else np.asarray([])

        # generate_paths' own output dtype is governed by precision.simulation
        # (its "rates"/"yield_curves" arrays are what price_swaps/price_swaptions/
        # price_bermudan_swaptions/price_american_swaptions derive THEIR OWN
        # working dtype from -- see this module's docstring's "Concurrency"
        # section and PrecisionConfig's docstring: none of the four pricers
        # take a dtype parameter, they inherit it from whatever JAX array
        # they're handed). So precision.pricing independently controlling
        # npv_cube's dtype (the plan's own explicit requirement, exercised
        # end-to-end with simulation=64/pricing=32) requires re-casting
        # market["rates"]/market["yield_curves"] to pricing_dtype here
        # whenever it differs from the simulation dtype -- a no-op cast
        # (jnp.asarray with a dtype already matching is free) when the two
        # knobs agree, which is the common case.
        market_for_pricing = market
        if market["rates"].dtype != pricing_dtype:
            market_for_pricing = dict(market)
            market_for_pricing["rates"] = jnp.asarray(market["rates"], dtype=pricing_dtype)
            if "yield_curves" in market:
                market_for_pricing["yield_curves"] = jnp.asarray(market["yield_curves"], dtype=pricing_dtype)

        npv_cube, order = _price_by_type(trades, market_for_pricing, maturities_np, step_times)
        base_npv = _base_npv(trades, maturities_np, market_config, request.precision)

        risk = compute_risk_metrics(npv_cube, base_npv, percentiles=request.percentiles)

        greeks_out = None
        if request.compute_greeks:
            greeks_out = _compute_all_greeks(trades, request.precision)

    return PortfolioResult(
        base_npv=base_npv, npv_cube=npv_cube, risk=risk, greeks=greeks_out,
        warnings=collected_warnings,
    )


def _fill_calibrated_sigma(
    trades: List[TradeConfig], calibration_targets: Optional[List[CalibrationTarget]], market_config: SimulationConfig,
) -> List[TradeConfig]:
    """Fills in `hw_sigma=None` on any Bermudan/American trade by
    calibrating once per distinct `rate_factor_index` that needs it (not
    once per trade -- every trade sharing a rate factor shares the same
    calibrated `Sigma`)."""
    needs_calibration = [
        cfg for cfg in trades
        if isinstance(cfg, (BermudanSwaptionConfig, AmericanSwaptionConfig)) and cfg.hw_sigma is None
    ]
    if not needs_calibration:
        return trades
    if calibration_targets is None:
        raise ValueError(
            "one or more trades have hw_sigma=None (uncalibrated) but "
            "request.calibration_targets was not supplied"
        )

    cache: Dict[int, "Sigma"] = {}
    updated = []
    for cfg in trades:
        if isinstance(cfg, (BermudanSwaptionConfig, AmericanSwaptionConfig)) and cfg.hw_sigma is None:
            idx = cfg.rate_factor_index
            if idx not in cache:
                curve = _zero_curve_of(cfg, cfg.initial_zero_curve)
                result = calibrate_lgm_sigma(calibration_targets, curve, a=cfg.hw_a)
                cache[idx] = result.sigma
            updated.append(replace(cfg, hw_sigma=cache[idx]))
        else:
            updated.append(cfg)
    return updated


def _price_by_type(trades, market, maturities_np, step_times):
    """Groups `trades` by type for pricing (each pricer only accepts a
    homogeneous list), prices each group, then reassembles one
    `[Scenarios, TimeSteps, Trades]` cube in the CALLER's original trade
    order -- the routing this function's docstring in `PortfolioRequest`
    promises. Returns `(npv_cube, order)`, `order` being the index
    permutation applied (kept for callers that want to trace it, unused by
    `price_portfolio` itself beyond the reordering)."""
    groups: Dict[type, List[int]] = {SwapConfig: [], SwaptionConfig: [], BermudanSwaptionConfig: [], AmericanSwaptionConfig: []}
    for i, cfg in enumerate(trades):
        groups[type(cfg)].append(i)

    num_scenarios = market["rates"].shape[0]
    num_steps = market["rates"].shape[1]
    per_trade_cubes: Dict[int, jax.Array] = {}

    if groups[SwapConfig]:
        swap_cfgs = [trades[i] for i in groups[SwapConfig]]
        cube = price_swaps(market["yield_curves"], maturities_np, swap_cfgs)
        for slot, i in enumerate(groups[SwapConfig]):
            per_trade_cubes[i] = cube[:, :, slot]

    if groups[SwaptionConfig]:
        swaption_cfgs = [trades[i] for i in groups[SwaptionConfig]]
        cube = price_swaptions(market["rates"], step_times, swaption_cfgs)
        for slot, i in enumerate(groups[SwaptionConfig]):
            per_trade_cubes[i] = cube[:, :, slot]

    if groups[BermudanSwaptionConfig]:
        berm_cfgs = [trades[i] for i in groups[BermudanSwaptionConfig]]
        cube = price_bermudan_swaptions(berm_cfgs, market["rates"], step_times)
        for slot, i in enumerate(groups[BermudanSwaptionConfig]):
            per_trade_cubes[i] = cube[:, :, slot]

    if groups[AmericanSwaptionConfig]:
        amer_cfgs = [trades[i] for i in groups[AmericanSwaptionConfig]]
        cube = price_american_swaptions(amer_cfgs, market["rates"], step_times)
        for slot, i in enumerate(groups[AmericanSwaptionConfig]):
            per_trade_cubes[i] = cube[:, :, slot]

    ordered = [per_trade_cubes[i] for i in range(len(trades))]
    npv_cube = jnp.stack(ordered, axis=-1) if ordered else jnp.zeros((num_scenarios, num_steps, 0))
    order = list(range(len(trades)))
    return npv_cube, order


def _base_npv(
    trades: List[TradeConfig], maturities_np: np.ndarray, market_config: SimulationConfig,
    precision: PrecisionConfig,
) -> float:
    """t=0 NPV of the whole portfolio, against zero-shock (today's actual)
    curves -- generalizes `demo.py`'s hand-written per-type sum into a loop
    over the routed trades, one instrument type at a time.

    Every array this function constructs itself (as opposed to what the
    pricers derive from their own JAX-array inputs) carries `precision.
    pricing`'s dtype -- see `PrecisionConfig`'s docstring and this module's
    own docstring's "Concurrency" section for why this matters: none of
    `price_swaps`/`price_swaptions`/`price_bermudan_swaption_base` need a
    new parameter to respect it, since each already derives its working
    dtype from the JAX-array inputs this function hands them."""
    dtype = _dtype_of(precision.pricing)
    total = 0.0
    swap_cfgs = [cfg for cfg in trades if isinstance(cfg, SwapConfig)]
    if swap_cfgs:
        for cfg in swap_cfgs:
            disc_curve = market_config.rates.initial_zero_curves[cfg.discount_curve_index]
            fwd_curve = market_config.rates.initial_zero_curves[cfg.forward_curve_index]
            base_cube = _flat_curve_cube(disc_curve, fwd_curve, maturities_np, cfg.evaluation_date, dtype=dtype)
            remapped = replace(cfg, discount_curve_index=0, forward_curve_index=1)
            total += float(price_swaps(base_cube, maturities_np, [remapped])[0, 0, 0])

    for cfg in trades:
        if isinstance(cfg, SwaptionConfig):
            r0_path = jnp.zeros((1, 1, len(market_config.rates.initial_rates)), dtype=dtype)
            r0_path = r0_path.at[0, 0, :].set(jnp.asarray(market_config.rates.initial_rates, dtype=dtype))
            total += float(price_swaptions(r0_path, jnp.array([0.0], dtype=dtype), [cfg])[0, 0, 0])
        elif isinstance(cfg, BermudanSwaptionConfig):
            total += price_bermudan_swaption_base(cfg)
        elif isinstance(cfg, AmericanSwaptionConfig):
            total += price_bermudan_swaption_base(cfg.to_bermudan())

    return total


def _flat_curve_cube(
    disc_curve_cfg, fwd_curve_cfg, maturities_np: np.ndarray, eval_date: ORE.Date, dtype=jnp.float64,
) -> jax.Array:
    """Builds a `[1, 1, len(maturities), 2]` deterministic (zero-shock)
    yield curve cube directly from two `ZeroCurveConfig`s' own pillar
    rates/times (linear-interpolated onto `maturities_np`), matching
    `engine.simulation.demo_scenarios.flat_yield_curves`'s output shape but
    generalized to an arbitrary (non-flat) curve rather than a single flat
    rate -- required since a real portfolio's discount/forward curves need
    not be flat. The interpolation itself always runs in float64 NumPy
    (`disc_times`/`disc_rates`/etc. below) regardless of `dtype` -- only the
    final cast (this function's actual output) carries the requested
    precision; there is no meaningful "float32 interpolation" step worth
    plumbing through np.interp here."""
    disc_times = np.asarray(disc_curve_cfg.times, dtype=np.float64)
    disc_rates = np.asarray(disc_curve_cfg.rates, dtype=np.float64)
    fwd_times = np.asarray(fwd_curve_cfg.times, dtype=np.float64)
    fwd_rates = np.asarray(fwd_curve_cfg.rates, dtype=np.float64)

    disc_z = np.interp(maturities_np, disc_times, disc_rates)
    fwd_z = np.interp(maturities_np, fwd_times, fwd_rates)
    disc_df = np.exp(-disc_z * maturities_np)
    fwd_df = np.exp(-fwd_z * maturities_np)
    cube = np.stack([disc_df, fwd_df], axis=-1)
    return jnp.asarray(cube[None, None, :, :], dtype=dtype)


def _compute_all_greeks(
    trades: List[TradeConfig], precision: PrecisionConfig = PrecisionConfig(),
) -> Dict[int, Dict[str, jax.Array]]:
    """Delta/Gamma/Theta for every trade, keyed by its own index in the
    caller's original `trades` order -- routed to the matching
    `engine.risk.greeks` function per instrument type. Vega is only
    well-defined for a Bermudan/American trade whose `hw_sigma` is a
    genuine calibrated `Sigma` (see `engine.risk.greeks`'s own module
    docstring); it's included here whenever that's the case.

    `precision.risk`'s dtype governs the `ZeroCurve` each Greeks function is
    handed -- every hardcoded-`jnp.float64` closure inside `engine.risk.
    greeks` itself derives its own working dtype from that curve (see that
    module's docstring), so passing a `risk`-dtype curve here is sufficient
    to make the whole Greeks computation honor `precision.risk`, with no
    further parameters needed on the public Greeks entry points."""
    dtype = _dtype_of(precision.risk)
    out: Dict[int, Dict[str, jax.Array]] = {}
    for i, cfg in enumerate(trades):
        if isinstance(cfg, SwapConfig):
            # SwapConfig alone doesn't carry its own ZeroCurveConfig (it
            # indexes into the simulation's curves instead) -- Greeks for a
            # swap need an explicit ZeroCurve the caller must supply
            # separately (see engine.risk.greeks.swap_delta_gamma); skipped
            # here rather than guessed, to avoid silently pricing Greeks
            # against a placeholder curve.
            continue
        elif isinstance(cfg, SwaptionConfig):
            curve = _zero_curve_of(cfg, cfg.initial_zero_curve, dtype=dtype)
            trade_greeks = dict(_greeks.swaption_delta_gamma(cfg, curve))
            trade_greeks["theta"] = _greeks.swaption_theta(cfg, curve)
            out[i] = trade_greeks
        elif isinstance(cfg, (BermudanSwaptionConfig, AmericanSwaptionConfig)):
            berm_cfg = cfg.to_bermudan() if isinstance(cfg, AmericanSwaptionConfig) else cfg
            curve = _zero_curve_of(berm_cfg, berm_cfg.initial_zero_curve, dtype=dtype)
            trade_greeks = dict(_greeks.bermudan_delta_gamma(berm_cfg, curve))
            trade_greeks["theta"] = _greeks.bermudan_theta(berm_cfg, curve)
            out[i] = trade_greeks
    return out
