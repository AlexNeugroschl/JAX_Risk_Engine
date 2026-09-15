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
from engine.models.lgm import Sigma

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
class PricingPrecisionOverride:
    """Optional per-instrument-type drill-down for `PrecisionConfig.pricing`.
    Any field left `None` falls back to `default`. Purely additive: a caller
    who only sets `default` gets today's flat-`pricing=N` behavior exactly --
    see `_resolve_pricing_dtype` below, the single place this is resolved.
    """
    default: int = 64
    swap: Optional[int] = None
    european_swaption: Optional[int] = None
    bermudan_swaption: Optional[int] = None
    american_swaption: Optional[int] = None

    def __post_init__(self):
        for name in ("default", "swap", "european_swaption", "bermudan_swaption", "american_swaption"):
            value = getattr(self, name)
            if value is not None and value not in (32, 64):
                raise ValueError(f"PricingPrecisionOverride.{name} must be 32 or 64, got {value!r}")


@dataclass(frozen=True)
class RiskPrecisionOverride:
    """Optional per-Greek/per-metric drill-down for `PrecisionConfig.risk`.
    `delta_gamma` is ONE knob for both Delta and Gamma -- they're derived
    from a single jax.grad+jax.hessian pair against one curve inside one
    `engine.risk.greeks` call, so they can't be split further without
    invasive surgery there (see docs/concepts/architecture.md's "Adjustable
    precision" section). `var_es` is NOT curve-driven like the other three
    -- `price_portfolio` re-casts `npv_cube`/`base_npv` immediately before
    calling `compute_risk_metrics`, since VaR/ES has no curve of its own.
    """
    default: int = 64
    delta_gamma: Optional[int] = None
    theta: Optional[int] = None
    vega: Optional[int] = None
    var_es: Optional[int] = None

    def __post_init__(self):
        for name in ("default", "delta_gamma", "theta", "vega", "var_es"):
            value = getattr(self, name)
            if value is not None and value not in (32, 64):
                raise ValueError(f"RiskPrecisionOverride.{name} must be 32 or 64, got {value!r}")


@dataclass(frozen=True)
class PrecisionConfig:
    """Four independently-settable dtype knobs, each 32 (float32) or 64
    (float64, default): `simulation` (Monte Carlo path generation, passed to
    `generate_paths`), `pricing` (instrument NPV/npv_cube dtype), `risk`
    (VaR/ES + Greeks), and `calibration` (LGM sigma bootstrap dtype).
    Defaults to all-64, byte-identical to this codebase's behavior before
    this dataclass existed.

    `pricing` and `risk` each additionally accept a structured override
    (`PricingPrecisionOverride`/`RiskPrecisionOverride`) instead of a flat
    `int`, for optional per-instrument-type / per-Greek drill-down -- a flat
    `int` is sugar for "every sub-field at this precision," resolved through
    the exact same `_resolve_pricing_dtype`/`_resolve_risk_dtype` helpers a
    structured override uses, never a separate code path. `simulation` and
    `calibration` stay flat `int`-only: each has exactly one call site, so no
    drill-down axis applies.

    bfloat16/float16/FP8/FP4 are NOT supported for any of these four knobs --
    confirmed broken on this stack today (`jnp.linalg.cholesky` and
    `jax.scipy.stats.norm.ppf` both raise on every sub-float32 dtype tested,
    not just bfloat16, on the installed jax/jaxlib CPU backend); see
    docs/concepts/architecture.md's "Option B: why uniform sub-float32
    precision is not achievable today" section. A separate, narrower FP8/FP4
    mechanism scoped ONLY to the two matmul-shaped sub-steps inside
    `generate_paths` (`MatmulPrecisionConfig`) has been designed but not yet
    implemented in this codebase -- see the same architecture doc section.
    """
    simulation: int = 64
    pricing: Union[int, PricingPrecisionOverride] = 64
    risk: Union[int, RiskPrecisionOverride] = 64
    calibration: int = 64

    def __post_init__(self):
        for name in ("simulation", "calibration"):
            value = getattr(self, name)
            if value not in (32, 64):
                raise ValueError(f"PrecisionConfig.{name} must be 32 or 64, got {value!r}")
        if isinstance(self.pricing, int) and self.pricing not in (32, 64):
            raise ValueError(f"PrecisionConfig.pricing must be 32, 64, or a PricingPrecisionOverride, got {self.pricing!r}")
        if isinstance(self.risk, int) and self.risk not in (32, 64):
            raise ValueError(f"PrecisionConfig.risk must be 32, 64, or a RiskPrecisionOverride, got {self.risk!r}")


def _dtype_of(precision_bits: int):
    return jnp.float64 if precision_bits == 64 else jnp.float32


_PRICING_TYPE_FIELD = {
    SwapConfig: "swap",
    SwaptionConfig: "european_swaption",
    BermudanSwaptionConfig: "bermudan_swaption",
    AmericanSwaptionConfig: "american_swaption",
}


def _resolve_pricing_dtype(pricing: Union[int, PricingPrecisionOverride], trade_type: type):
    """Single place PrecisionConfig.pricing's flat-int-or-override shape is
    resolved to a concrete dtype for one trade type -- every pricing dispatch
    point calls this instead of re-deriving the branch itself."""
    if isinstance(pricing, int):
        return _dtype_of(pricing)
    bits = getattr(pricing, _PRICING_TYPE_FIELD[trade_type]) or pricing.default
    return _dtype_of(bits)


def _resolve_risk_dtype(risk: Union[int, RiskPrecisionOverride], metric: str):
    """Single place PrecisionConfig.risk's flat-int-or-override shape is
    resolved to a concrete dtype for one metric ('delta_gamma'/'theta'/
    'vega'/'var_es') -- every risk dispatch point calls this instead of
    re-deriving the branch itself."""
    if isinstance(risk, int):
        return _dtype_of(risk)
    bits = getattr(risk, metric) or risk.default
    return _dtype_of(bits)


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

    Also emits `warnings.warn` (not a hard error) for two known-limitation
    cases, both collected into `PortfolioResult.warnings` by
    `price_portfolio`:

    - a Bermudan/American trade whose `exercise_times` aren't reset-aligned
      with its own underlying's accrual/payment dates -- see docs/planning/
      traderx-integration.md gap item 5 and this module's own docstring;
    - a `SwapConfig` that will be AGED (its floating leg already accruing)
      at one or more simulated steps beyond t=0 -- see
      `_warn_if_aged_swap_exposure`. This is the one check here that applies
      to `SwapConfig`, which otherwise carries no `rate_factor_index` to
      cross-check.
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

    _warn_if_aged_swap_exposure(sim_config, trade_configs)


def _warn_if_aged_swap_exposure(sim_config: SimulationConfig, trade_configs) -> None:
    """Emits a `UserWarning` for every `SwapConfig` whose floating leg will
    have ALREADY STARTED accruing at one or more of the simulation's own
    `time_grid` steps beyond t=0.

    This surfaces the documented aged-swap limitation (see
    `engine.instruments.swap`'s module docstring and
    `tests/test_swap.py::TestAgedSwapKnownLimitation`): `price_swaps` has no
    representation of an already-fixed floating coupon, so at any simulated
    step past a swap's first accrual start the elapsed period is discounted
    with a clamped, non-meaningful P(t,T) for T<t instead of being excluded
    or fixed. t=0 valuation is unaffected and exact.

    **Why a warning and not a fix here:** fixing it requires per-scenario
    already-fixed rates (or exclusion of elapsed cashflows) inside the
    pricing kernel, AND the historical fixings to populate them -- neither
    of which exists in this engine or in the current TraderX export. What
    this function removes is the SILENCE: before it, a caller requesting a
    multi-step `npv_cube` (and therefore every VaR/ES number derived from
    it) inherited a known inaccuracy with nothing in the result saying so.
    `price_portfolio` collects these into `PortfolioResult.warnings`.

    Only steps STRICTLY after t=0 are considered, and a forward-starting
    swap is only flagged once the grid actually reaches its accrual start --
    both remain exact otherwise, so neither should warn."""
    steps_after_zero = [t for t in sim_config.time_grid if float(t) > 0.0]
    if not steps_after_zero:
        return
    last_step = max(float(t) for t in steps_after_zero)

    for i, cfg in enumerate(trade_configs):
        if not isinstance(cfg, SwapConfig):
            continue
        swap = build_vanilla_swap(
            notional=cfg.notional, fixed_rate=cfg.fixed_rate, payer=cfg.payer,
            swap_tenor=cfg.swap_tenor, index_tenor_months=cfg.index_tenor_months,
            floating_spread=cfg.floating_spread, evaluation_date=cfg.evaluation_date,
        )
        today = cfg.evaluation_date
        starts = [
            DAY_COUNTER.yearFraction(today, ORE.as_floating_rate_coupon(cf).accrualStartDate())
            for cf in swap.floatingLeg()
        ]
        if not starts:
            continue
        first_accrual_start = min(starts)
        # Aged only if the grid actually advances past the first accrual
        # start. A forward-starting swap whose accrual begins after the last
        # simulated step is never aged within this simulation.
        if last_step > first_accrual_start:
            aged_steps = [t for t in steps_after_zero if float(t) > first_accrual_start]
            warnings.warn(
                f"trade[{i}] (SwapConfig, notional={cfg.notional}): floating leg has "
                f"already started accruing (first accrual start t="
                f"{first_accrual_start:.6f}) at {len(aged_steps)} simulated time step(s) "
                f"beyond t=0 (up to t={last_step:.6f}). Conditional NPV at those steps "
                f"uses the documented aged-swap approximation -- an already-fixed "
                f"floating coupon is not represented, so npv_cube values at those "
                f"steps, and any VaR/ES/exposure derived from them, carry a known "
                f"inaccuracy. t=0 base NPV is unaffected. See "
                f"engine/instruments/swap.py's module docstring.",
                stacklevel=2,
            )


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
    precision: independent simulation/pricing/risk/calibration dtype
        control, with optional per-instrument-type/per-Greek drill-down --
        see `PrecisionConfig`. Defaults to all-64, byte-identical to this
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
    # t=0 NPV of each trade individually, in request.trades order.
    # `base_npv` is by construction `sum(base_npv_per_trade)` -- the total and
    # the breakdown are computed once and cannot disagree. Required to
    # reconcile a portfolio total against identified positions/contracts
    # rather than reporting only an unattributable aggregate.
    base_npv_per_trade: List[float] = field(default_factory=list)


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
        trades = _fill_calibrated_sigma(trades, request.calibration_targets, market_config, request.precision)
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
        maturities_np = np.asarray(market_config.rates.maturities) if market_config.rates.maturities else np.asarray([])

        # _price_by_type now resolves and casts to each instrument type's own
        # dtype internally (via _resolve_pricing_dtype), since
        # precision.pricing may be a per-instrument-type PricingPrecisionOverride
        # rather than one flat int shared by every trade -- see that
        # function's docstring for the jnp.stack widest-dtype-wins consequence
        # this produces on npv_cube itself when buckets disagree.
        step_times = jnp.array(market_config.time_grid[1:], dtype=jnp.float64)
        npv_cube, order = _price_by_type(trades, market, maturities_np, step_times, request.precision.pricing)
        base_npv_per_trade = _base_npv_per_trade(trades, maturities_np, market_config, request.precision)
        base_npv = float(sum(base_npv_per_trade))

        # risk.var_es is NOT curve-driven like delta_gamma/theta/vega -- it has
        # no curve of its own, so honoring an override that differs from
        # `pricing` means an explicit re-cast of npv_cube immediately before
        # compute_risk_metrics, not a substituted input array upstream.
        # base_npv is a plain Python float -- JAX's scalar-promotion rules
        # already resolve `portfolio_npv - base_npv` to the ARRAY operand's
        # dtype (confirmed in engine.risk.var_es.portfolio_pnl), so it needs
        # no separate cast. This can only ever narrow precision relative to
        # what `pricing` already produced -- it can't recover precision
        # `pricing` already lost.
        var_es_dtype = _resolve_risk_dtype(request.precision.risk, "var_es")
        npv_cube_for_risk = npv_cube if npv_cube.dtype == var_es_dtype else jnp.asarray(npv_cube, dtype=var_es_dtype)
        risk = compute_risk_metrics(npv_cube_for_risk, base_npv, percentiles=request.percentiles)

        greeks_out = None
        if request.compute_greeks:
            greeks_out = _compute_all_greeks(
                trades, market_config, request.precision,
                calibration_targets=request.calibration_targets,
            )

    return PortfolioResult(
        base_npv=base_npv, npv_cube=npv_cube, risk=risk, greeks=greeks_out,
        warnings=collected_warnings, base_npv_per_trade=base_npv_per_trade,
    )


def _fill_calibrated_sigma(
    trades: List[TradeConfig], calibration_targets: Optional[List[CalibrationTarget]], market_config: SimulationConfig,
    precision: PrecisionConfig = PrecisionConfig(),
) -> List[TradeConfig]:
    """Fills in `hw_sigma=None` on any Bermudan/American trade by
    calibrating once per distinct `rate_factor_index` that needs it (not
    once per trade -- every trade sharing a rate factor shares the same
    calibrated `Sigma`). `precision.calibration`'s dtype governs the curve
    handed to `calibrate_lgm_sigma`, which now derives its own working dtype
    from `curve.pillar_rates.dtype` (see that function's docstring) rather
    than hardcoding float64, mirroring the same curve-driven pattern
    `_compute_all_greeks` uses for `precision.risk`."""
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

    calib_dtype = _dtype_of(precision.calibration)
    cache: Dict[int, "Sigma"] = {}
    updated = []
    for cfg in trades:
        if isinstance(cfg, (BermudanSwaptionConfig, AmericanSwaptionConfig)) and cfg.hw_sigma is None:
            idx = cfg.rate_factor_index
            if idx not in cache:
                curve = _zero_curve_of(cfg, cfg.initial_zero_curve, dtype=calib_dtype)
                result = calibrate_lgm_sigma(calibration_targets, curve, a=cfg.hw_a)
                cache[idx] = result.sigma
            updated.append(replace(cfg, hw_sigma=cache[idx]))
        else:
            updated.append(cfg)
    return updated


def _price_by_type(trades, market, maturities_np, step_times, pricing: Union[int, PricingPrecisionOverride] = 64):
    """Groups `trades` by type for pricing (each pricer only accepts a
    homogeneous list), prices each group, then reassembles one
    `[Scenarios, TimeSteps, Trades]` cube in the CALLER's original trade
    order -- the routing this function's docstring in `PortfolioRequest`
    promises. Returns `(npv_cube, order)`, `order` being the index
    permutation applied (kept for callers that want to trace it, unused by
    `price_portfolio` itself beyond the reordering).

    `pricing` may be a flat int or a `PricingPrecisionOverride` -- each
    instrument-type bucket resolves and casts to ITS OWN dtype via
    `_resolve_pricing_dtype` before pricing, independent of the others.

    Consequence to know, not a bug: when buckets resolve to different
    dtypes, the final `jnp.stack` below promotes `npv_cube` to the WIDEST
    dtype present (confirmed: `jnp.stack([float64, float32], axis=-1).dtype
    == float64`) -- a mixed-precision request still controls the *cost* of
    computing each bucket's own cube (an expensive Bermudan tree running
    cheaper while a trivial swap stays exact), but `npv_cube.dtype` itself
    reflects the widest bucket present, not necessarily the one a caller
    drilled down on."""
    groups: Dict[type, List[int]] = {SwapConfig: [], SwaptionConfig: [], BermudanSwaptionConfig: [], AmericanSwaptionConfig: []}
    for i, cfg in enumerate(trades):
        groups[type(cfg)].append(i)

    num_scenarios = market["rates"].shape[0]
    num_steps = market["rates"].shape[1]
    per_trade_cubes: Dict[int, jax.Array] = {}

    def _cast(arr, dtype):
        return arr if arr.dtype == dtype else jnp.asarray(arr, dtype=dtype)

    if groups[SwapConfig]:
        dtype = _resolve_pricing_dtype(pricing, SwapConfig)
        swap_cfgs = [trades[i] for i in groups[SwapConfig]]
        cube = price_swaps(_cast(market["yield_curves"], dtype), maturities_np, swap_cfgs)
        for slot, i in enumerate(groups[SwapConfig]):
            per_trade_cubes[i] = cube[:, :, slot]

    if groups[SwaptionConfig]:
        dtype = _resolve_pricing_dtype(pricing, SwaptionConfig)
        swaption_cfgs = [trades[i] for i in groups[SwaptionConfig]]
        cube = price_swaptions(_cast(market["rates"], dtype), _cast(step_times, dtype), swaption_cfgs)
        for slot, i in enumerate(groups[SwaptionConfig]):
            per_trade_cubes[i] = cube[:, :, slot]

    if groups[BermudanSwaptionConfig]:
        dtype = _resolve_pricing_dtype(pricing, BermudanSwaptionConfig)
        berm_cfgs = [trades[i] for i in groups[BermudanSwaptionConfig]]
        cube = price_bermudan_swaptions(berm_cfgs, _cast(market["rates"], dtype), _cast(step_times, dtype))
        for slot, i in enumerate(groups[BermudanSwaptionConfig]):
            per_trade_cubes[i] = cube[:, :, slot]

    if groups[AmericanSwaptionConfig]:
        dtype = _resolve_pricing_dtype(pricing, AmericanSwaptionConfig)
        amer_cfgs = [trades[i] for i in groups[AmericanSwaptionConfig]]
        cube = price_american_swaptions(amer_cfgs, _cast(market["rates"], dtype), _cast(step_times, dtype))
        for slot, i in enumerate(groups[AmericanSwaptionConfig]):
            per_trade_cubes[i] = cube[:, :, slot]

    ordered = [per_trade_cubes[i] for i in range(len(trades))]
    npv_cube = jnp.stack(ordered, axis=-1) if ordered else jnp.zeros((num_scenarios, num_steps, 0))
    order = list(range(len(trades)))
    return npv_cube, order


def _base_npv_per_trade(
    trades: List[TradeConfig], maturities_np: np.ndarray, market_config: SimulationConfig,
    precision: PrecisionConfig,
) -> List[float]:
    """t=0 NPV of EACH trade, against zero-shock (today's actual) curves,
    returned in the caller's own `trades` order -- generalizes `demo.py`'s
    hand-written per-type sum into a loop over the routed trades, one
    instrument type at a time.

    **Returns per-trade values, not just their sum.** `PortfolioResult.
    base_npv` is then defined as `sum(...)` of this list, so the reported
    total and the reported per-trade breakdown cannot disagree -- they are
    the same numbers. Before this function returned a list, only the
    aggregate float existed, and a portfolio total could not be reconciled
    against identified rows at all (the integration requirement behind
    `PortfolioResult.base_npv_per_trade`).

    Every array this function constructs itself (as opposed to what the
    pricers derive from their own JAX-array inputs) carries `precision.
    pricing`'s resolved per-instrument-type dtype (via
    `_resolve_pricing_dtype`, mirroring `_price_by_type`) -- see
    `PrecisionConfig`'s docstring and this module's own docstring's
    "Concurrency" section for why this matters: none of `price_swaps`/
    `price_swaptions`/`price_bermudan_swaption_base` need a new parameter to
    respect it, since each already derives its working dtype from the
    JAX-array inputs this function hands them. `price_bermudan_swaption_base`
    itself takes no dtype input at all -- a pre-existing scope boundary this
    function doesn't attempt to fix."""
    per_trade: List[float] = [0.0] * len(trades)
    for i, cfg in enumerate(trades):
        if isinstance(cfg, SwapConfig):
            swap_dtype = _resolve_pricing_dtype(precision.pricing, SwapConfig)
            disc_curve = market_config.rates.initial_zero_curves[cfg.discount_curve_index]
            fwd_curve = market_config.rates.initial_zero_curves[cfg.forward_curve_index]
            base_cube = _flat_curve_cube(disc_curve, fwd_curve, maturities_np, cfg.evaluation_date, dtype=swap_dtype)
            remapped = replace(cfg, discount_curve_index=0, forward_curve_index=1)
            per_trade[i] = float(price_swaps(base_cube, maturities_np, [remapped])[0, 0, 0])
        elif isinstance(cfg, SwaptionConfig):
            dtype = _resolve_pricing_dtype(precision.pricing, SwaptionConfig)
            r0_path = jnp.zeros((1, 1, len(market_config.rates.initial_rates)), dtype=dtype)
            r0_path = r0_path.at[0, 0, :].set(jnp.asarray(market_config.rates.initial_rates, dtype=dtype))
            per_trade[i] = float(price_swaptions(r0_path, jnp.array([0.0], dtype=dtype), [cfg])[0, 0, 0])
        elif isinstance(cfg, BermudanSwaptionConfig):
            per_trade[i] = price_bermudan_swaption_base(cfg)
        elif isinstance(cfg, AmericanSwaptionConfig):
            per_trade[i] = price_bermudan_swaption_base(cfg.to_bermudan())

    return per_trade


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


def _swap_curve_configs(cfg: SwapConfig, market_config: SimulationConfig, trade_index: int):
    """Resolves one `SwapConfig`'s `discount_curve_index`/
    `forward_curve_index` into the two `ZeroCurveConfig`s they name in
    `market_config.rates.initial_zero_curves`.

    Raises `ValueError` naming the trade and the offending index if either
    is out of range. This is deliberately a hard failure rather than a
    clamp or a fallback to curve 0: an out-of-range index means the request
    is internally inconsistent, and silently substituting SOME curve would
    produce a plausible-looking sensitivity computed against a curve the
    trade was never booked against -- precisely the class of silent
    mispricing this module's other validators exist to prevent."""
    curves = market_config.rates.initial_zero_curves
    for name, idx in (("discount_curve_index", cfg.discount_curve_index),
                      ("forward_curve_index", cfg.forward_curve_index)):
        if not 0 <= idx < len(curves):
            raise ValueError(
                f"trade[{trade_index}] (SwapConfig, notional={cfg.notional}): "
                f"{name}={idx} is out of range for "
                f"sim_config.rates.initial_zero_curves (length {len(curves)})"
            )
    return curves[cfg.discount_curve_index], curves[cfg.forward_curve_index]


def _compute_all_greeks(
    trades: List[TradeConfig],
    market_config: SimulationConfig,
    precision: PrecisionConfig = PrecisionConfig(),
    calibration_targets: Optional[List[CalibrationTarget]] = None,
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
    further parameters needed on the public Greeks entry points.

    `precision.risk` may be a flat int or a `RiskPrecisionOverride` -- when
    it's an override, `delta_gamma` and `theta` are each resolved (via
    `_resolve_risk_dtype`) and built against their OWN separately-constructed
    `ZeroCurve`, so the two can differ. `delta_gamma` stays one shared knob
    for both Delta and Gamma (see `RiskPrecisionOverride`'s docstring for
    why). When both resolve to the same dtype (the common flat-`risk=N`
    case), this pays one small, redundant extra curve-construction call --
    a deliberate simplicity-over-micro-optimization choice, since building a
    `ZeroCurve` is a cheap pillar-count array build, not a JIT-compiled
    trace.

    `market_config` supplies the simulation's own
    `rates.initial_zero_curves`, which is what makes SWAP Greeks reachable
    here: a `SwapConfig` carries curve INDEXES
    (`discount_curve_index`/`forward_curve_index`) rather than its own
    `ZeroCurveConfig`, so resolving them needs the very
    `SimulationConfig` those indexes are defined against. Before this
    parameter existed, this function had no access to it and skipped every
    swap outright -- `compute_greeks=True` silently returned a result with
    no entry for any swap, even though `engine.risk.greeks.swap_delta_gamma`
    /`swap_theta` were fully implemented and tested. The curves are resolved
    explicitly from the request's own market, never guessed or defaulted; an
    out-of-range index raises (see `_swap_curve_configs`) rather than
    silently pricing Greeks off the wrong pillar."""
    out: Dict[int, Dict[str, jax.Array]] = {}
    for i, cfg in enumerate(trades):
        if isinstance(cfg, SwapConfig):
            disc_cfg, fwd_cfg = _swap_curve_configs(cfg, market_config, i)
            dg_dtype = _resolve_risk_dtype(precision.risk, "delta_gamma")
            theta_dtype = _resolve_risk_dtype(precision.risk, "theta")
            trade_greeks = dict(_greeks.swap_delta_gamma(
                cfg,
                _zero_curve_of(cfg, disc_cfg, dtype=dg_dtype),
                _zero_curve_of(cfg, fwd_cfg, dtype=dg_dtype),
            ))
            trade_greeks["theta"] = _greeks.swap_theta(
                cfg,
                _zero_curve_of(cfg, disc_cfg, dtype=theta_dtype),
                _zero_curve_of(cfg, fwd_cfg, dtype=theta_dtype),
            )
            out[i] = trade_greeks
        elif isinstance(cfg, SwaptionConfig):
            dg_curve = _zero_curve_of(cfg, cfg.initial_zero_curve, dtype=_resolve_risk_dtype(precision.risk, "delta_gamma"))
            theta_curve = _zero_curve_of(cfg, cfg.initial_zero_curve, dtype=_resolve_risk_dtype(precision.risk, "theta"))
            trade_greeks = dict(_greeks.swaption_delta_gamma(cfg, dg_curve))
            trade_greeks["theta"] = _greeks.swaption_theta(cfg, theta_curve)
            out[i] = trade_greeks
        elif isinstance(cfg, (BermudanSwaptionConfig, AmericanSwaptionConfig)):
            berm_cfg = cfg.to_bermudan() if isinstance(cfg, AmericanSwaptionConfig) else cfg
            dg_curve = _zero_curve_of(berm_cfg, berm_cfg.initial_zero_curve, dtype=_resolve_risk_dtype(precision.risk, "delta_gamma"))
            theta_curve = _zero_curve_of(berm_cfg, berm_cfg.initial_zero_curve, dtype=_resolve_risk_dtype(precision.risk, "theta"))
            trade_greeks = dict(_greeks.bermudan_delta_gamma(berm_cfg, dg_curve))
            trade_greeks["theta"] = _greeks.bermudan_theta(berm_cfg, theta_curve)
            # Vega is only well-defined when hw_sigma is a genuine CALIBRATED
            # Sigma term structure produced from `calibration_targets` (in
            # that same order) -- `bermudan_vega` differentiates through the
            # bootstrap relating each target's market_vol to that Sigma, so a
            # flat/hand-set hw_sigma has no market quote to be sensitive TO.
            # Skipped (not raised) in that case: a flat-sigma Bermudan is a
            # legitimate request, it simply has no Vega to report.
            if calibration_targets and isinstance(berm_cfg.hw_sigma, Sigma):
                vega_curve = _zero_curve_of(
                    berm_cfg, berm_cfg.initial_zero_curve,
                    dtype=_resolve_risk_dtype(precision.risk, "vega"),
                )
                trade_greeks["vega"] = _greeks.bermudan_vega(
                    berm_cfg, vega_curve, calibration_targets,
                )
            out[i] = trade_greeks
    return out
