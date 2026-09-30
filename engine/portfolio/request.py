"""
The engine's portfolio entry point: `PortfolioRequest` (trades, market, risk settings) in,
`PortfolioResult` (base NPVs, scenario cube, exposure, Greeks) out, via `price_portfolio`.

Pure dataclasses and JAX; the HTTP layer (`engine/api/`) wraps these types.

Two paths, chosen by the request's `market`:

  * an `engine.market.Market` (the default, `engine.portfolio.market_path`): ORE's
    semantics end to end -- the cross-asset LGM simulation, each trade priced on every path
    with its t=0 engine, exercise and fixings as ORE handles them, ORE's sensitivities;
  * a `SimulationConfig` (this module): the Hull-White model, a supported non-default
    option, with trade-level model parameters. Its cube carries known limitations, warned about per trade rather than
    corrected: aged swaps (I-04, audit M-2), options vanishing at expiry (audit M-3), and short
    rates inconsistent with a sloped curve (audit M-1). t=0 base NPVs are unaffected.

Concurrency: `jax_enable_x64` is process-global, so two threads pricing at different
precisions in one process can corrupt each other. `_PRICING_LOCK` serializes the JAX work of
`price_portfolio`. The HTTP path runs each job in a single-threaded worker process
(`engine.portfolio.worker_pool`), so the lock only matters for direct multi-threaded
callers, and is kept for them.
"""
import threading
import warnings
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
import ORE

from engine.market import Market
from engine.simulation.config import CamConfig
from engine.simulation.market_model import SimulationConfig, generate_paths
from engine.valuation.config import PricingConfig
from engine.instruments.swap import SwapConfig, price_swaps, swap_schedule
from engine.instruments.european_swaption import SwaptionConfig, prepare_swaption, price_swaptions
from engine.instruments.bermudan_swaption import (
    BermudanSwaptionConfig, price_bermudan_swaptions, price_bermudan_swaption_base,
)
from engine.instruments.american_swaption import AmericanSwaptionConfig, price_american_swaptions
from engine.instruments.treasury import (
    RATE_BUMP, BondConfig, ScenarioPricingNotSupported, _remaining_cashflows, price_bond_base,
)
from engine.calibration.lgm import calibrate_lgm_sigma, CalibrationTarget
from engine.risk.var_es import ENGINE_RISK_MEASURE
from engine.risk import greeks as _greeks
from engine.risk.exposure import ExposureProfile, exposure_profile, netting_set_profile
from engine.models.hull_white import discount as _hw_discount
from engine.models.hull_white import ZeroCurve as _HwZeroCurve
from engine.models.lgm import Sigma

# Re-exported so `engine.portfolio._validate_common_fields` resolves (implemented in
# validation.py to avoid an import cycle).
from engine.portfolio.validation import _validate_common_fields, _validate_tenor  # noqa: F401
from engine.portfolio.validation import validate_single_evaluation_date
from engine.portfolio.profiling import phase as _phase

TradeConfig = Union[
    SwapConfig, SwaptionConfig, BermudanSwaptionConfig, AmericanSwaptionConfig, BondConfig,
]

#: Trade types with no scenario (`npv_cube`) representation (I-24). They get base NPV and
#: Greeks; `_price_by_type` refuses them rather than broadcasting a constant column.
#: `engine.market_risk` covers them by t=0 revaluation.
DETERMINISTIC_ONLY_TYPES = (BondConfig,)

# Serializes price_portfolio's JAX work within a process (see the module docstring).
_PRICING_LOCK = threading.Lock()


@dataclass(frozen=True)
class PricingPrecisionOverride:
    """Per-instrument-type overrides for `PrecisionConfig.pricing`; `None` fields fall back
    to `default`. Resolved only in `_resolve_pricing_dtype`."""
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
    """Per-metric overrides for `PrecisionConfig.risk`; `None` fields fall back to
    `default`. Delta and Gamma share `delta_gamma` (one gradient/Hessian computation).
    `exposure` has no curve, so `price_portfolio` casts `npv_cube` to it before the
    statistics."""
    default: int = 64
    delta_gamma: Optional[int] = None
    theta: Optional[int] = None
    vega: Optional[int] = None
    exposure: Optional[int] = None

    def __post_init__(self):
        for name in ("default", "delta_gamma", "theta", "vega", "exposure"):
            value = getattr(self, name)
            if value is not None and value not in (32, 64):
                raise ValueError(f"RiskPrecisionOverride.{name} must be 32 or 64, got {value!r}")


@dataclass(frozen=True)
class PrecisionConfig:
    """
    Dtype knobs, each 32 or 64 (default 64): `simulation` (`generate_paths`), `pricing`
    (NPVs and `npv_cube`), `risk` (exposure and Greeks), `calibration` (the LGM sigma
    bootstrap). `pricing` and `risk` also accept a per-type/per-metric override object; a
    plain int means every sub-field.

    Sub-float32 dtypes are not supported: `jnp.linalg.cholesky` and
    `jax.scipy.stats.norm.ppf` raise on them on the installed CPU backend (see
    docs/concepts/architecture.md, "Adjustable precision").
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
    """`PrecisionConfig.pricing` (int or override) -> dtype for one trade type."""
    if isinstance(pricing, int):
        return _dtype_of(pricing)
    bits = getattr(pricing, _PRICING_TYPE_FIELD[trade_type]) or pricing.default
    return _dtype_of(bits)


def _resolve_risk_dtype(risk: Union[int, RiskPrecisionOverride], metric: str):
    """`PrecisionConfig.risk` (int or override) -> dtype for one metric ('delta_gamma',
    'theta', 'vega', 'exposure')."""
    if isinstance(risk, int):
        return _dtype_of(risk)
    bits = getattr(risk, metric) or risk.default
    return _dtype_of(bits)


# Cross-checks between the simulation config and each trade's duplicated fields
def validate_portfolio_against_simulation(
    sim_config: SimulationConfig, trade_configs: Sequence[TradeConfig],
) -> None:
    """
    Check each trade against `sim_config` before pricing; raise `ValueError` naming the
    trade and field on a mismatch.

    For trades with a `rate_factor_index` (all but swaps): `hw_a`, a flat `hw_sigma` and
    `initial_zero_curve` must match that factor's mean reversion, volatility
    (sqrt of the `joint_covariance` diagonal) and curve. Also: one evaluation date for all
    trades, and swap curve indices in range.

    Emits `warnings.warn` (collected into `PortfolioResult.warnings`) for the known cube
    limitations: aged swaps, options expiring inside the horizon, and short rates
    inconsistent with their curve.
    """
    tol = 1e-9
    num_eq = len(sim_config.equities.initial_prices)

    for i, cfg in enumerate(trade_configs):
        if not isinstance(cfg, (SwaptionConfig, BermudanSwaptionConfig, AmericanSwaptionConfig)):
            continue
        rate_factor_index = cfg.rate_factor_index
        if rate_factor_index is None:
            raise ValueError(f"trade[{i}] ({type(cfg).__name__}, notional={cfg.notional}): the Hull-White "
                             f"path needs rate_factor_index; price it on a Market instead")

        label = f"trade[{i}] ({type(cfg).__name__}, notional={cfg.notional})"
        missing = [name for name in ("hw_a", "initial_zero_curve") if getattr(cfg, name) is None]
        if missing:
            raise ValueError(f"{label}: the Hull-White path needs {', '.join(missing)} on the trade; "
                             f"price it on a Market instead for ORE's engines")
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
            # Uncalibrated: price_portfolio calibrates it later; nothing to check yet.
            pass
        elif hasattr(actual_sigma, "values"):
            # A piecewise (calibrated) Sigma is not checked: the simulation uses one flat
            # volatility per factor, and a calibrated pricing term structure may
            # legitimately differ from it.
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

    validate_single_evaluation_date(trade_configs)
    _validate_swap_curve_indices(sim_config, trade_configs)
    _warn_if_aged_swap_exposure(sim_config, trade_configs)
    _warn_if_option_expires_within_simulation(sim_config, trade_configs)
    _warn_if_rates_inconsistent_with_curve(sim_config)


def _validate_swap_curve_indices(
    sim_config: SimulationConfig, trade_configs: Sequence[TradeConfig],
) -> None:
    """Range-check every swap's `discount_curve_index`/`forward_curve_index` before any
    pricing. A negative index would otherwise wrap in Python (-1 -> last curve) and price
    silently against the wrong curve (I-13)."""
    curves = sim_config.rates.initial_zero_curves
    for i, cfg in enumerate(trade_configs):
        if not isinstance(cfg, SwapConfig):
            continue
        label = f"trade[{i}] ({type(cfg).__name__}, notional={cfg.notional})"
        for name, idx in (("discount_curve_index", cfg.discount_curve_index),
                          ("forward_curve_index", cfg.forward_curve_index)):
            if idx is None:
                raise ValueError(f"{label}: the Hull-White path needs {name}; price it on a Market instead")
            if not 0 <= idx < len(curves):
                raise ValueError(
                    f"{label}: {name}={idx} is out of range for "
                    f"sim_config.rates.initial_zero_curves (length {len(curves)}). "
                    f"A negative index would otherwise select a curve by wrapping "
                    f"(-1 -> the last curve), pricing the trade against a curve it "
                    f"was never booked against."
                )


def _warn_if_aged_swap_exposure(sim_config: SimulationConfig, trade_configs) -> None:
    """Warn for every swap whose floating leg has started accruing at a simulated step
    after t=0: from then on the cube misprices it (I-04, audit M-2; see
    `engine.instruments.swap`). t=0 and forward-starting swaps before their start are
    exact and are not flagged."""
    steps_after_zero = [t for t in sim_config.time_grid if float(t) > 0.0]
    if not steps_after_zero:
        return
    last_step = max(float(t) for t in steps_after_zero)

    for i, cfg in enumerate(trade_configs):
        if not isinstance(cfg, SwapConfig):
            continue
        starts = swap_schedule(cfg).floating.accrual_start_times
        if starts.size == 0:
            continue
        first_accrual_start = float(starts.min())
        # Aged only if the grid passes the first accrual start.
        if last_step > first_accrual_start:
            aged_steps = [t for t in steps_after_zero if float(t) > first_accrual_start]
            warnings.warn(
                f"trade[{i}] (SwapConfig, notional={cfg.notional}): floating leg has "
                f"already started accruing (first accrual start t="
                f"{first_accrual_start:.6f}) at {len(aged_steps)} simulated time step(s) "
                f"beyond t=0 (up to t={last_step:.6f}). Conditional NPV at those steps "
                f"uses the documented aged-swap approximation -- an already-fixed "
                f"floating coupon is not represented, and cashflows already paid by a "
                f"step are still counted in its NPV, so npv_cube values at those "
                f"steps, and any exposure derived from them, carry a known "
                f"inaccuracy. t=0 base NPV is unaffected. See "
                f"docs/planning/engine-audit.md (M-2).",
                stacklevel=2,
            )


def _warn_if_option_expires_within_simulation(sim_config: SimulationConfig, trade_configs) -> None:
    """Warn for every swaption whose last exercise falls inside the simulated horizon: its
    cube value is 0 from then on, even on paths where it was exercised (audit M-3)."""
    steps_after_zero = [float(t) for t in sim_config.time_grid if float(t) > 0.0]
    if not steps_after_zero:
        return
    last_step = max(steps_after_zero)
    for i, cfg in enumerate(trade_configs):
        if isinstance(cfg, SwaptionConfig):
            last_exercise = prepare_swaption(cfg).exercise_time
        elif isinstance(cfg, (BermudanSwaptionConfig, AmericanSwaptionConfig)):
            option_times = cfg.option_times()
            if not option_times:
                continue  # refused by the pricer itself
            last_exercise = max(option_times)
        else:
            continue
        if last_exercise < last_step:
            after = sum(1 for t in steps_after_zero if t >= last_exercise)
            warnings.warn(
                f"trade[{i}] ({type(cfg).__name__}, notional={cfg.notional}): last exercise "
                f"at t={last_exercise:.6f} falls inside the simulated horizon (up to "
                f"t={last_step:.6f}). Its npv_cube value is 0 at the {after} step(s) from "
                f"then on; exercise into the underlying swap is not tracked, so exposure "
                f"after expiry is misstated. See docs/planning/engine-audit.md (M-3).",
                stacklevel=2,
            )


def _warn_if_rates_inconsistent_with_curve(sim_config: SimulationConfig) -> None:
    """Warn when a rate factor's simulated short rate is inconsistent with its curve
    (audit M-1): unless the curve is flat and `initial_rates == theta ==` its level, the
    simulated discount factors do not reprice the curve (4-9% off at t=2y on 3%->5%)."""
    rates = sim_config.rates
    tol = 1e-12
    for k, curve in enumerate(rates.initial_zero_curves or []):
        level = float(curve.rates[0])
        flat = all(abs(float(r) - level) <= tol for r in curve.rates)
        consistent = (
            flat
            and abs(float(rates.initial_rates[k]) - level) <= tol
            and abs(float(rates.theta[k]) - level) <= tol
        )
        if not consistent:
            warnings.warn(
                f"rate factor {k}: the simulated short rate (initial_rates="
                f"{rates.initial_rates[k]}, theta={rates.theta[k]}) is not consistent with "
                f"its initial zero curve (rates {list(curve.rates)}), so simulated discount "
                f"factors are not arbitrage-free against that curve and every npv_cube value "
                f"past t=0, and the exposure derived from it, is biased. Only a flat curve "
                f"with initial_rates == theta == its level is consistent. See "
                f"docs/planning/engine-audit.md (M-1).",
                stacklevel=2,
            )


# Maturity pillars for the swap cube
def derive_maturity_pillars(trade_configs: Sequence[TradeConfig], evaluation_date: ORE.Date) -> List[float]:
    """
    Sorted union of the times each swap's pricer reads a discount factor at on
    `evaluation_date` (`SwapSchedule.pillar_times`), plus 0: the pillars
    `engine.instruments.swap._maturity_indices` requires. Swaptions read the short-rate
    paths, not the cube, so they add none.
    """
    pillars = {0.0}
    for cfg in trade_configs:
        if not isinstance(cfg, SwapConfig):
            continue
        pillars.update(swap_schedule(replace(cfg, evaluation_date=evaluation_date)).pillar_times())
    return sorted(pillars)


# Request and result
@dataclass
class PortfolioRequest:
    """
    Input to `price_portfolio`.

    market: today's `Market` (the market path; then `simulation`, `pricing` and
        `base_currency` apply), or a Hull-White `SimulationConfig` for `generate_paths` (if
        `market.rates.maturities` is unset, it is derived with `derive_maturity_pillars`).
    trades: any mix of trade types; results come back in this order.
    pfe_quantiles: PFE quantiles for the exposure profiles.
    calibration_targets: used to calibrate any Bermudan/American with `hw_sigma=None`,
        once per `rate_factor_index`. One basket serves every rate factor (ORE instead
        calibrates each trade to a basket built from its own exercise dates; I-47).
    compute_greeks: also compute Delta/Gamma/Theta (and Vega where defined) per trade.
    precision: see `PrecisionConfig`.
    trade_ids: optional, one unique id per trade, echoed on the result so a caller need not
        rely on positions (I-10).

    Bonds (`BondConfig`) are priced at t=0 only: no cube column, no exposure (I-24), and
    Greeks by bumped revaluation (`_bond_greeks`).

    The cube is a multi-step risk-neutral simulation, used for exposure profiles
    (`engine.risk.exposure`). Short-horizon VaR/ES is `engine.market_risk.run_market_risk`.
    """
    market: Union[SimulationConfig, Market]
    trades: List[TradeConfig]
    pfe_quantiles: Sequence[float] = (0.95, 0.99)
    calibration_targets: Optional[List[CalibrationTarget]] = None
    compute_greeks: bool = False
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    #: Whether to build `npv_cube` and the exposure profiles. On the Hull-White path, set False
    #: for a portfolio with deterministic-only trades (bonds): the result then has an empty
    #: `npv_cube` and no exposure (absent, not zero), and `scenario_risk_available` says so.
    scenario_risk: bool = True
    #: Market path only: ORE's simulation configuration (required with `scenario_risk`),
    #: pricing engines, and the base currency when there is no simulation.
    simulation: Optional[CamConfig] = None
    pricing: PricingConfig = field(default_factory=PricingConfig)
    base_currency: str = "USD"
    trade_ids: Optional[Sequence[str]] = None

    def __post_init__(self):
        if self.trade_ids is None:
            return
        ids = list(self.trade_ids)
        if len(ids) != len(self.trades):
            raise ValueError(f"trade_ids has {len(ids)} entries for {len(self.trades)} trades")
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"trade_ids must be unique non-empty strings; got {ids}")


# price_portfolio
@dataclass
class PortfolioResult:
    """Output of `price_portfolio`."""
    base_npv: float
    npv_cube: jax.Array                                   # [Scenarios, TimeSteps, Trades]
    #: The whole portfolio as one netting set, no collateral. `None` without scenario risk.
    exposure: Optional[ExposureProfile] = None
    #: Standalone exposure per trade, in request order. Empty without scenario risk.
    trade_exposures: List[ExposureProfile] = field(default_factory=list)
    greeks: Optional[Dict[int, Dict[str, jax.Array]]] = None  # trade index (in request.trades order) -> greeks dict
    warnings: List[str] = field(default_factory=list)
    # t=0 NPV per trade, in request order; `base_npv` is their sum.
    base_npv_per_trade: List[float] = field(default_factory=list)
    # Whether `npv_cube`/`exposure` were computed (`scenario_risk`). When False, `exposure`
    # is None and `npv_cube` has no time steps. Carried on the result for consumers that
    # never see the request.
    scenario_risk_available: bool = True
    # The measure of the exposure figures: `ENGINE_RISK_MEASURE` (risk-neutral-pricing), or
    # None without scenario risk (I-11).
    measure: Optional[str] = None
    # The request's `trade_ids`, in request order, or None if it had none (I-10).
    trade_ids: Optional[List[str]] = None


def price_portfolio(request: PortfolioRequest) -> PortfolioResult:
    """
    Price a `PortfolioRequest`:

    1. Validate `joint_covariance` (also done in `generate_paths`; here it fails earlier).
    2. Cross-check trades against the market (`validate_portfolio_against_simulation`),
       collecting warnings.
    3. Derive the cube's maturity pillars if unset.
    4. Calibrate any `hw_sigma=None` Bermudan/American, once per rate factor.
    5. Simulate (`generate_paths`).
    6. Price every trade into one `[Scenarios, TimeSteps, Trades]` cube, in request order.
    7. Price every trade at t=0 for the base NPVs.
    8. Exposure profiles, netting set and per trade.
    9. Optionally Greeks.

    Steps 4-9 run under `_PRICING_LOCK`; 1-3 run no JAX code.

    A request whose `market` is a `Market` goes to the market path instead
    (`engine.portfolio.market_path.price_on_market`). Either way the result carries the
    request's `trade_ids`.
    """
    if isinstance(request.market, Market):
        from engine.portfolio.market_path import price_on_market
        with _PRICING_LOCK:
            result = price_on_market(request)
    else:
        result = _price_on_simulation(request)
    result.trade_ids = None if request.trade_ids is None else list(request.trade_ids)
    return result


def _price_on_simulation(request: PortfolioRequest) -> PortfolioResult:
    """`price_portfolio` on the Hull-White path (steps 1-9 above)."""
    from engine.simulation.market_model import validate_joint_covariance

    validate_joint_covariance(request.market.joint_covariance)

    market_config = request.market
    collected_warnings: List[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_portfolio_against_simulation(market_config, request.trades)
        collected_warnings.extend(str(w.message) for w in caught)

    if market_config.rates.maturities is None:
        # Use the trades' evaluation date (all trades share one). Not
        # ORE.Settings.evaluationDate: that is thread-local and defaults to today on a
        # thread that never set it, such as an API background task.
        eval_date = request.trades[0].evaluation_date if request.trades else ORE.Settings.instance().evaluationDate
        pillars = derive_maturity_pillars(request.trades, eval_date)
        market_config = replace(market_config, rates=replace(market_config.rates, maturities=pillars))

    trades = list(request.trades)

    with _PRICING_LOCK:
        # Calibration runs JAX code too, so it is inside the lock. Each `_phase` labels a
        # region in profiler traces (see engine.portfolio.profiling).
        with _phase("calibration"):
            trades = _fill_calibrated_sigma(trades, request.calibration_targets, market_config, request.precision)
        with _phase("simulation"):
            market = generate_paths(market_config, precision=request.precision.simulation)
        # generate_paths restores x64 to its prior value, which is off in a 32-bit worker.
        # Later steps may still want float64, and float32 arrays are unaffected by the
        # flag, so enable it.
        jax.config.update("jax_enable_x64", True)
        maturities_np = np.asarray(market_config.rates.maturities) if market_config.rates.maturities else np.asarray([])

        # Each instrument type is priced in its own resolved dtype; jnp.stack then
        # promotes npv_cube to the widest one present.
        step_times = jnp.array(market_config.time_grid[1:], dtype=jnp.float64)
        if request.scenario_risk:
            with _phase("pricing"):
                npv_cube = _price_by_type(trades, market, maturities_np, step_times, request.precision.pricing)
        else:
            # Empty, not zero-filled: zeros would read as real NPVs.
            num_scenarios = market["rates"].shape[0]
            npv_cube = jnp.zeros((num_scenarios, 0, 0))
        with _phase("base_npv"):
            base_npv_per_trade = _base_npv_per_trade(trades, maturities_np, market_config, request.precision)
            base_npv = float(sum(base_npv_per_trade))

        # Exposure has no curve, so its dtype is applied by casting the cube.
        exposure = None
        trade_exposures: List[ExposureProfile] = []
        if request.scenario_risk:
            exposure_dtype = _resolve_risk_dtype(request.precision.risk, "exposure")
            with _phase("exposure"):
                exposure, trade_exposures = _exposure_profiles(
                    npv_cube.astype(exposure_dtype), base_npv_per_trade, market, market_config,
                    request.pfe_quantiles,
                )

        greeks_out = None
        if request.compute_greeks:
            with _phase("greeks"):
                greeks_out = _compute_all_greeks(
                    trades, market_config, request.precision,
                    calibration_targets=request.calibration_targets,
                )

    return PortfolioResult(
        base_npv=base_npv, npv_cube=npv_cube, exposure=exposure, trade_exposures=trade_exposures,
        greeks=greeks_out,
        warnings=collected_warnings, base_npv_per_trade=base_npv_per_trade,
        scenario_risk_available=request.scenario_risk,
        measure=ENGINE_RISK_MEASURE if request.scenario_risk else None,
    )


def _exposure_profiles(npv_cube, base_npv_per_trade, market, market_config, quantiles):
    """Netting-set and per-trade exposure profiles from the cube. The numeraire accrues on
    rate factor 0, so P(0,t) for EE_B comes from that factor's curve."""
    step_times = np.asarray(market_config.time_grid[1:], dtype=np.float64)
    dtype = npv_cube.dtype
    numeraire = jnp.asarray(market["numeraire"], dtype=dtype)
    discount = _hw_discount(
        _HwZeroCurve.from_config(market_config.rates.initial_zero_curves[0], dtype=dtype),
        jnp.asarray(step_times, dtype=dtype),
    )
    netting_set = netting_set_profile(npv_cube, base_npv_per_trade, numeraire, discount, step_times, quantiles)
    per_trade = [
        exposure_profile(npv_cube[:, :, i], base_npv_per_trade[i], numeraire, discount, step_times, quantiles)
        for i in range(npv_cube.shape[-1])
    ]
    return netting_set, per_trade


def _fill_calibrated_sigma(
    trades: List[TradeConfig], calibration_targets: Optional[List[CalibrationTarget]], market_config: SimulationConfig,
    precision: PrecisionConfig = PrecisionConfig(),
) -> List[TradeConfig]:
    """Replace `hw_sigma=None` on Bermudans/Americans with a `Sigma` calibrated to
    `calibration_targets`, once per `rate_factor_index` (trades on the same factor share
    it). The same targets are used for every factor. Works in `precision.calibration`."""
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
                curve = _HwZeroCurve.from_config(cfg.initial_zero_curve, dtype=calib_dtype)
                result = calibrate_lgm_sigma(calibration_targets, curve, a=cfg.hw_a)
                cache[idx] = result.sigma
            updated.append(replace(cfg, hw_sigma=cache[idx]))
        else:
            updated.append(cfg)
    return updated


def _price_by_type(trades, market, maturities_np, step_times, pricing: Union[int, PricingPrecisionOverride] = 64):
    """Price each trade type as a group and reassemble one `[Scenarios, TimeSteps, Trades]`
    cube in request order.

    Each group is priced in its own resolved dtype; `jnp.stack` promotes the cube to the
    widest dtype present."""
    # Refuse deterministic-only trades by name before pricing (I-24).
    deterministic = [
        (i, cfg) for i, cfg in enumerate(trades)
        if isinstance(cfg, DETERMINISTIC_ONLY_TYPES)
    ]
    if deterministic:
        names = ", ".join(
            f"trade[{i}] ({type(cfg).__name__}, notional={cfg.notional})"
            for i, cfg in deterministic
        )
        raise ScenarioPricingNotSupported(
            f"{names}: no scenario NPV, so cannot appear in npv_cube. "
            f"Such a trade is closed-form arithmetic against a single deterministic "
            f"curve -- there is no stochastic driver to vary across scenarios and no "
            f"time evolution to step. Its t=0 value IS available: call "
            f"`_base_npv_per_trade`/`price_bond_base` directly, or use "
            f"`price_portfolio(..., scenario_risk=False)` to get base NPV and Greeks "
            f"without an exposure profile, and `engine.market_risk` for short-horizon "
            f"VaR/ES. Refused rather than broadcast: a constant column reports a "
            f"position as risk-measured when its risk was never modelled (I-24)."
        )

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
        cfgs = [trades[i] for i in groups[BermudanSwaptionConfig]]
        cube = price_bermudan_swaptions(cfgs, _cast(market["rates"], dtype), _cast(step_times, dtype))
        for slot, i in enumerate(groups[BermudanSwaptionConfig]):
            per_trade_cubes[i] = cube[:, :, slot]

    if groups[AmericanSwaptionConfig]:
        dtype = _resolve_pricing_dtype(pricing, AmericanSwaptionConfig)
        amer_cfgs = [trades[i] for i in groups[AmericanSwaptionConfig]]
        cube = price_american_swaptions(amer_cfgs, _cast(market["rates"], dtype), _cast(step_times, dtype))
        for slot, i in enumerate(groups[AmericanSwaptionConfig]):
            per_trade_cubes[i] = cube[:, :, slot]

    ordered = [per_trade_cubes[i] for i in range(len(trades))]
    return jnp.stack(ordered, axis=-1) if ordered else jnp.zeros((num_scenarios, num_steps, 0))


def _base_npv_per_trade(
    trades: List[TradeConfig], maturities_np: np.ndarray, market_config: SimulationConfig,
    precision: PrecisionConfig,
) -> List[float]:
    """t=0 NPV of each trade on today's curves, in request order. `PortfolioResult.base_npv`
    is their sum.

    Swaps use a one-step cube of today's curves; Europeans condition on
    `market.rates.initial_rates` as r(0) (equal to the curve-consistent price only when
    initial_rates is the curve's short end, see audit M-1); Bermudans/Americans use the
    backward induction (float64, no dtype input); bonds use `price_bond_base`."""
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
        elif isinstance(cfg, (BermudanSwaptionConfig, AmericanSwaptionConfig)):
            per_trade[i] = price_bermudan_swaption_base(cfg)
        elif isinstance(cfg, BondConfig):
            # Dirty NPV (as engine.integration.note). Plain float arithmetic, so no dtype.
            per_trade[i] = price_bond_base(cfg)

    return per_trade


def _flat_curve_cube(
    disc_curve_cfg, fwd_curve_cfg, maturities_np: np.ndarray, eval_date: ORE.Date, dtype=jnp.float64,
) -> jax.Array:
    """`[1, 1, len(maturities), 2]` cube of today's discount factors from two
    `ZeroCurveConfig`s (`engine.models.curves.discount`). Interpolates in float64 and casts
    to `dtype`."""
    maturities = jnp.asarray(maturities_np, dtype=jnp.float64)
    cube = jnp.stack([
        _hw_discount(_HwZeroCurve.from_config(disc_curve_cfg), maturities),
        _hw_discount(_HwZeroCurve.from_config(fwd_curve_cfg), maturities),
    ], axis=-1)
    return jnp.asarray(cube[None, None, :, :], dtype=dtype)


def _swap_curve_configs(cfg: SwapConfig, market_config: SimulationConfig, trade_index: int):
    """The two `ZeroCurveConfig`s a swap's curve indices name; raises on an out-of-range
    index rather than falling back to another curve."""
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
    """Greeks for every trade, keyed by request index.

    Each Greek gets a `ZeroCurve` in its resolved `precision.risk` dtype, and the Greeks
    functions work in their curve's dtype. Swaps take their curves from
    `market_config.rates.initial_zero_curves` by index; an out-of-range index raises."""
    out: Dict[int, Dict[str, jax.Array]] = {}
    for i, cfg in enumerate(trades):
        # Label each trade's Greeks in profiler traces.
        with _phase(f"greeks/trade{i}/{type(cfg).__name__}"):
            trade_greeks = _greeks_for_one_trade(
                cfg, i, market_config, precision, calibration_targets,
            )
        if trade_greeks is not None:
            out[i] = trade_greeks
    return out


def _greeks_for_one_trade(
    cfg: TradeConfig,
    index: int,
    market_config: SimulationConfig,
    precision: PrecisionConfig,
    calibration_targets: Optional[List[CalibrationTarget]],
) -> Optional[Dict[str, jax.Array]]:
    """Greeks for one trade by type; `None` for a type with no Greeks."""
    if isinstance(cfg, SwapConfig):
        disc_cfg, fwd_cfg = _swap_curve_configs(cfg, market_config, index)
        dg_dtype = _resolve_risk_dtype(precision.risk, "delta_gamma")
        theta_dtype = _resolve_risk_dtype(precision.risk, "theta")
        trade_greeks = dict(_greeks.swap_delta_gamma(
            cfg,
            _HwZeroCurve.from_config(disc_cfg, dtype=dg_dtype),
            _HwZeroCurve.from_config(fwd_cfg, dtype=dg_dtype),
        ))
        trade_greeks["theta"] = _greeks.swap_theta(
            cfg,
            _HwZeroCurve.from_config(disc_cfg, dtype=theta_dtype),
            _HwZeroCurve.from_config(fwd_cfg, dtype=theta_dtype),
        )
        return trade_greeks

    if isinstance(cfg, SwaptionConfig):
        dg_curve = _HwZeroCurve.from_config(cfg.initial_zero_curve, dtype=_resolve_risk_dtype(precision.risk, "delta_gamma"))
        theta_curve = _HwZeroCurve.from_config(cfg.initial_zero_curve, dtype=_resolve_risk_dtype(precision.risk, "theta"))
        trade_greeks = dict(_greeks.swaption_delta_gamma(cfg, dg_curve))
        trade_greeks["theta"] = _greeks.swaption_theta(cfg, theta_curve)
        return trade_greeks

    if isinstance(cfg, (BermudanSwaptionConfig, AmericanSwaptionConfig)):
        dg_curve = _HwZeroCurve.from_config(cfg.initial_zero_curve, dtype=_resolve_risk_dtype(precision.risk, "delta_gamma"))
        theta_curve = _HwZeroCurve.from_config(cfg.initial_zero_curve, dtype=_resolve_risk_dtype(precision.risk, "theta"))
        trade_greeks = dict(_greeks.bermudan_delta_gamma(cfg, dg_curve))
        trade_greeks["theta"] = _greeks.bermudan_theta(cfg, theta_curve)
        # Vega only for a Sigma calibrated from `calibration_targets` (it differentiates
        # through that bootstrap). A flat hw_sigma has no market vol to be sensitive to, so
        # Vega is omitted, not reported as 0.
        if calibration_targets and isinstance(cfg.hw_sigma, Sigma):
            vega_curve = _HwZeroCurve.from_config(
                cfg.initial_zero_curve,
                dtype=_resolve_risk_dtype(precision.risk, "vega"),
            )
            trade_greeks["vega"] = _greeks.bermudan_vega(
                cfg, vega_curve, calibration_targets,
            )
        return trade_greeks

    if isinstance(cfg, BondConfig):
        # Every trade type needs a branch here; a missing one returns None and silently
        # drops that type's Greeks (I-01). Bonds are plain Python floats, not JAX, so
        # their Greeks are bumped revaluations and `precision.risk` does not apply.
        return _bond_greeks(cfg)

    return None


def _bond_greeks(cfg: BondConfig) -> Dict[str, jax.Array]:
    """Delta/Gamma/Theta for one `BondConfig`, by bumped revaluation.

    Delta: central difference `(P(+1bp) - P(-1bp)) / 2` of the dirty NPV. The integration
    boundary's `rateSensitivity` is the one-sided `P(+1bp) - P(0)` agreed with TraderX; the
    two differ by the curvature term (~1.4e-4 on a 6M bill at 100k face).
    Gamma: `P(+1bp) - 2P(0) + P(-1bp)`.
    Theta: dirty NPV with the evaluation date one calendar day later, same curve, minus
    today's, plus the flows paid in (t, t + 1] (ORE's Theta adds the period's cash flows
    back, as `swap_theta` does; I-39).
    Vega: omitted (no volatility input), not 0.
    """
    base = price_bond_base(cfg)
    up = price_bond_base(cfg, rate_shift=RATE_BUMP)
    down = price_bond_base(cfg, rate_shift=-RATE_BUMP)

    # Central difference.
    delta = (up - down) / 2.0
    gamma = up - 2.0 * base + down

    out = {"delta": jnp.asarray(delta), "gamma": jnp.asarray(gamma)}

    # For a bond maturing tomorrow there is no next-day instrument to reprice, so Theta is
    # omitted rather than failing the whole call.
    if cfg.maturity_date > cfg.evaluation_date + 1:
        one_day_on = replace(cfg, evaluation_date=cfg.evaluation_date + 1)
        paid = sum(amount for date, amount in _remaining_cashflows(cfg) if date <= one_day_on.evaluation_date)
        out["theta"] = jnp.asarray(price_bond_base(one_day_on) - base + paid * cfg.face_amount)

    return out
