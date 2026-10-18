"""
The ORE parity matrix (I-116): every product by every analytic by every public setting, each cell
either compared with ORE by a parity test or given a stated reason why not, generated from the
tests and published in docs/reference/ore-parity.md.

**How a test declares its cells.** A parity test carries the marker
`@pytest.mark.ore_parity(cells(...), tolerance="...")`: `cells` derives the cells from the very
objects the test runs (its trades, `PricingConfig`, `GreeksConfig`, `CamConfig`, market-risk
request), so a cell is what was compared, not what a docstring says. A parametrized test may
declare per parameter (`pytest.param(..., marks=...)`): the seeded sweeps declare each case's own
cells. A cell is `(analytic, product, setting, value)`; `setting` "" is the product compared at
all, under whatever settings its test ran.

**What must have a cell.** `SETTINGS` lists every public setting: a path from the class that
holds it (`LgmSwaptionEngineConfig.strategy`, `LgmConfig.swap_index.fixed_tenor`), the analytics
and products it applies to, and the values the matrix needs a cell for: each choice of an
enumerated setting, else its default and any other value ("= 0.5", "≠ 0.5"). Every value of
every setting, for every analytic and product it applies to, and every product of every analytic,
has a cell or a reason in `REASONS`; tests/test_ore_parity_matrix.py fails otherwise. A reason is
one of three kinds: ORE cannot express it (the oracles' refusals, I-116's item 4), it is not
ORE's (an engine setting without an ORE counterpart), or it is not compared yet (with the item
that will compare it).

**Every field is classified.** The same test walks every dataclass reachable from the public
requests and trades (`ROOTS`) and fails on a field that is neither the leaf of a setting's path,
nor an input (`INPUTS`: trade terms and market data, which the sweeps draw, and identities), nor
in a class with one reason for all its fields (`CLASS_REASONS`, the precision policy). So a new
setting fails the suite until it has a cell or a reason. The HTTP routes are classified the same
way (`ROUTES`: the analytics each route runs).

**Generated.** `python -m tests.support.ore_parity_matrix --write` collects the tests (pytest's
collection with the `Collector` plugin, about two seconds) and writes the matrix between the
markers in docs/reference/ore-parity.md; the test fails if the published matrix is not what the
tests declare.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from engine.instruments._validation import LONG_SHORT, SETTLEMENT_TYPES
from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig
from engine.instruments.treasury import BondConfig, CouponPeriod
from engine.market_data.day_counts import SUPPORTED_ACCRUAL_DAY_COUNTS
from engine.market_data.market import CurrencyMarket, EquityMarket, Market, SwaptionVolSurface, ZeroCurveConfig
from engine.market_simulation.config import SWAPTION_VOL_DECAY_MODES, CamConfig, HullWhiteConfig, LgmConfig
from engine.models.lgm import Sigma
from engine.precision import Precision, StagePrecision
from engine.pricing.config import (
    CALIBRATION_METHODS, CALIBRATION_STRATEGIES, EUROPEAN_ENGINES, JamshidianEngineConfig, LgmSwaptionEngineConfig,
    PricingConfig,
)
from engine.risk.greeks.bump import SensitivityConfig
from engine.risk.market import MarketRiskRequest, RateRiskFactors, ShockScenarios
from engine.risk.market.scenarios import SOURCES
from engine.run import GreeksConfig, PortfolioRequest, RunConfig
from engine.run.config import GREEKS_METHODS
from engine.solvers.roots import SOLVERS

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "reference" / "ore-parity.md"
BEGIN, END = "<!-- ore-parity-matrix:begin -->", "<!-- ore-parity-matrix:end -->"

#: The analytics, as the run request will name them (I-87), and what each compares.
ANALYTICS = {
    "npv": "Today's value",
    "exposure": "Exposure: the cube on ORE's paths, the profiles, the simulation in distribution",
    "sensitivities": "Sensitivities: Delta, Gamma, Vega, Theta",
    "market_risk": "Market risk: the P&L of every scenario, VaR and ES",
    "calibration": "Calibration of the simulation's model",
}
#: The trade products (each trade class's `product`), the columns of every analytic but calibration.
PRODUCTS = {"swap": "Swap", "european_swaption": "European", "bermudan_swaption": "Bermudan",
            "american_swaption": "American", "bond": "Bond"}
#: The calibration analytic's columns: what is calibrated.
MODELS = {"cross_asset_model": "Cross-asset model (POST /calibration/cam)",
          "standalone_lgm": "Standalone LGM (POST /calibration/lgm)"}
TRADE_ANALYTICS = ("npv", "exposure", "sensitivities", "market_risk")
TRADE_CLASSES = (SwapConfig, SwaptionConfig, BermudanSwaptionConfig, AmericanSwaptionConfig, BondConfig)
LGM_OPTIONS = ("bermudan_swaption", "american_swaption")
VOL_READERS = ("european_swaption",) + LGM_OPTIONS
ALL = tuple(PRODUCTS)


def products_of(analytic: str) -> Tuple[str, ...]:
    return tuple(MODELS) if analytic == "calibration" else ALL


@dataclass(frozen=True, order=True)
class Cell:
    """One cell: `product` (or a model, for calibration) compared with ORE in `analytic`,
    with `setting` at `value`; `setting` "" is the product compared at all."""
    analytic: str
    product: str
    setting: str = ""
    value: str = ""


# ---------------------------------------------------------------------------------------------
# The settings
# ---------------------------------------------------------------------------------------------

#: The classes a setting's path may start from, by name.
_CLASSES = {cls.__name__: cls for cls in (SwapConfig, SwaptionConfig, BermudanSwaptionConfig, AmericanSwaptionConfig,
                                          BondConfig, CouponPeriod, LgmSwaptionEngineConfig, JamshidianEngineConfig,
                                          PricingConfig, SensitivityConfig, GreeksConfig, CamConfig, LgmConfig,
                                          HullWhiteConfig, MarketRiskRequest, ShockScenarios, PortfolioRequest,
                                          RunConfig)}


def _dataclasses_in(annotation) -> List[type]:
    if typing.get_origin(annotation) is None:
        return [annotation] if isinstance(annotation, type) and dataclasses.is_dataclass(annotation) else []
    return [t for arg in typing.get_args(annotation) for t in _dataclasses_in(arg)]


def _leaf(key: str) -> Tuple[type, str]:
    """The class and field a setting's path ends on."""
    cls, *path = key.split(".")
    owner = _CLASSES[cls]
    for name in path[:-1]:
        owner = _dataclasses_in(typing.get_type_hints(owner)[name])[0]
    return owner, path[-1]



@dataclass(frozen=True)
class Setting:
    """A public setting: `key` is its path from the class holding it (the group), `labels` the
    values the matrix needs a cell for, `label(value, group)` the label of a value (None where
    the setting does not apply, as a strategy without a calibration)."""
    key: str
    analytics: Tuple[str, ...]
    products: Tuple[str, ...]
    labels: Tuple[str, ...]
    label: Callable[[Any, Any], Optional[str]]

    @property
    def group(self) -> str:
        return self.key.split(".", 1)[0]

    def read(self, group: Any) -> Any:
        value = group
        for name in self.key.split(".")[1:]:
            value = getattr(value, name)
        return value


def _default_of(key: str) -> Any:
    """The default of the field `key` ends on, or `dataclasses.MISSING` for a required one."""
    owner, name = _leaf(key)
    field = {f.name: f for f in dataclasses.fields(owner)}[name]
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return dataclasses.MISSING


def _choice(key, analytics, products, choices, when: Callable[[Any], bool] = lambda group: True,
            name: Callable[[Any], str] = str) -> Setting:
    """An enumerated setting: a cell for each choice."""
    return Setting(key, tuple(analytics), tuple(products), tuple(name(c) for c in choices),
                   lambda value, group: name(value) if when(group) else None)


def _against_default(key, analytics, products, when: Callable[[Any], bool] = lambda group: True) -> Setting:
    """A setting compared at its default and at another value; a required one at any value."""
    default = _default_of(key)
    if default is dataclasses.MISSING:
        return Setting(key, tuple(analytics), tuple(products), ("any",),
                       lambda value, group: "any" if when(group) else None)
    same, other = f"= {_shown(default)}", f"≠ {_shown(default)}"
    return Setting(key, tuple(analytics), tuple(products), (same, other),
                   lambda value, group: (same if value == default else other) if when(group) else None)


def _shown(value) -> str:
    if isinstance(value, (tuple, list)):
        return "(" + ", ".join(map(str, value)) + ")" if len(value) <= 4 else f"the {len(value)} defaults"
    if dataclasses.is_dataclass(value):
        return type(value).__name__ + "()"
    return str(value)


def _labelled(key, analytics, products, labels: Dict[str, Callable[[Any], bool]]) -> Setting:
    """A setting whose values fall into named kinds (`labels`: name -> predicate), in order."""
    def label(value, group):
        return next(name for name, holds in labels.items() if holds(value))
    return Setting(key, tuple(analytics), tuple(products), tuple(labels), label)


def _calibrated(engine: LgmSwaptionEngineConfig) -> bool:
    return engine.calibration == "Bootstrap"


_TRADE_SETTINGS: List[Setting] = []
for _cls in TRADE_CLASSES:
    _name, _product = _cls.__name__, (_cls.product,)
    _fields = {f.name for f in dataclasses.fields(_cls)}
    if "payer" in _fields:
        _TRADE_SETTINGS.append(_labelled(f"{_name}.payer", TRADE_ANALYTICS, _product,
                                         {"payer": bool, "receiver": lambda v: not v}))
    for _field in ("index_tenor_months", "floating_spread", "currency"):
        if _field in _fields:
            _TRADE_SETTINGS.append(_against_default(f"{_name}.{_field}", TRADE_ANALYTICS, _product))
    if "fixings" in _fields:
        _TRADE_SETTINGS.append(_labelled(f"{_name}.fixings", TRADE_ANALYTICS, _product,
                                         {"none": lambda v: not v, "past fixings": bool}))
    if "settlement" in _fields:
        _TRADE_SETTINGS.append(_choice(f"{_name}.settlement", TRADE_ANALYTICS, _product, SETTLEMENT_TYPES))
    if "long_short" in _fields:
        _TRADE_SETTINGS.append(_choice(f"{_name}.long_short", TRADE_ANALYTICS, _product, LONG_SHORT))
    if "accrual_day_count" in _fields:
        _TRADE_SETTINGS.append(_choice(f"{_name}.accrual_day_count", TRADE_ANALYTICS, _product,
                                       sorted(SUPPORTED_ACCRUAL_DAY_COUNTS)))
_TRADE_SETTINGS += [
    _labelled("BondConfig.coupon_schedule", TRADE_ANALYTICS, ("bond",),
              {"coupons": bool, "none (a bill)": lambda v: not v}),
    _against_default("BondConfig.redemption_fraction", TRADE_ANALYTICS, ("bond",)),
    _labelled("BondConfig.coupon_schedule.payment_date", TRADE_ANALYTICS, ("bond",),
              {"at each period's end": lambda v: v is None, "delayed": lambda v: v is not None}),
]

_ENGINE_SETTINGS = [
    _choice("LgmSwaptionEngineConfig.calibration", TRADE_ANALYTICS, LGM_OPTIONS, CALIBRATION_METHODS),
    _choice("LgmSwaptionEngineConfig.strategy", TRADE_ANALYTICS, LGM_OPTIONS, CALIBRATION_STRATEGIES, _calibrated),
    _against_default("LgmSwaptionEngineConfig.bootstrap_tolerance", TRADE_ANALYTICS, LGM_OPTIONS, _calibrated),
    _against_default("LgmSwaptionEngineConfig.reversion", TRADE_ANALYTICS, LGM_OPTIONS),
    _against_default("LgmSwaptionEngineConfig.volatility", TRADE_ANALYTICS, LGM_OPTIONS),
    _against_default("LgmSwaptionEngineConfig.reference_calibration_grid", TRADE_ANALYTICS, LGM_OPTIONS),
    _against_default("LgmSwaptionEngineConfig.shift_horizon", TRADE_ANALYTICS, LGM_OPTIONS),
    _against_default("LgmSwaptionEngineConfig.n_per_std", TRADE_ANALYTICS, LGM_OPTIONS),
    _against_default("LgmSwaptionEngineConfig.std_devs", TRADE_ANALYTICS, LGM_OPTIONS),
    _against_default("LgmSwaptionEngineConfig.exercise_time_steps_per_year", TRADE_ANALYTICS, ("american_swaption",)),
    _against_default("LgmSwaptionEngineConfig.swap_index.fixed_tenor", TRADE_ANALYTICS, LGM_OPTIONS, _calibrated),
    _against_default("LgmSwaptionEngineConfig.swap_index.fixed_day_counter", TRADE_ANALYTICS, LGM_OPTIONS,
                     _calibrated),
    _against_default("LgmSwaptionEngineConfig.swap_index.index_tenor_months", TRADE_ANALYTICS, LGM_OPTIONS,
                     _calibrated),
    _choice("LgmSwaptionEngineConfig.solver", TRADE_ANALYTICS, LGM_OPTIONS, SOLVERS),
    _choice("PricingConfig.european", TRADE_ANALYTICS, ("european_swaption",), EUROPEAN_ENGINES),
    _against_default("JamshidianEngineConfig.reversion", TRADE_ANALYTICS, ("european_swaption",)),
    _against_default("JamshidianEngineConfig.volatility", TRADE_ANALYTICS, ("european_swaption",)),
    _choice("JamshidianEngineConfig.solver", TRADE_ANALYTICS, ("european_swaption",), SOLVERS),
]

_SENSITIVITY_SETTINGS = [
    _choice("GreeksConfig.method", ("sensitivities",), ALL, GREEKS_METHODS),
    _against_default("SensitivityConfig.curve_tenors", ("sensitivities",), ALL),
    _against_default("SensitivityConfig.curve_shift", ("sensitivities",), ALL),
    _against_default("SensitivityConfig.vol_shift", ("sensitivities",), VOL_READERS),
    _against_default("SensitivityConfig.theta_days", ("sensitivities",), ALL),
]

_SIMULATION_SETTINGS = [
    _labelled("CamConfig.ir", ("exposure", "calibration"), ALL + ("cross_asset_model",),
              {"LgmConfig": lambda v: all(type(m) is LgmConfig for m in v.values()),
               "HullWhiteConfig": lambda v: all(type(m) is HullWhiteConfig for m in v.values())}),
    _against_default("CamConfig.curve_tenors", ("exposure",), ALL),
    _choice("CamConfig.swaption_vol_decay", ("exposure",), ALL, SWAPTION_VOL_DECAY_MODES),
    _against_default("CamConfig.bootstrap_tolerance", ("exposure", "calibration"), ALL + ("cross_asset_model",)),
    _against_default("CamConfig.replicates", ("exposure",), ALL),
    _labelled("CamConfig.base_currency", ("exposure",), ALL,
              {"USD": lambda v: v == "USD", "another currency": lambda v: v != "USD"}),
    _labelled("CamConfig.fx_volatilities", ("exposure",), ALL, {"none": lambda v: not v, "FX simulated": bool}),
    _labelled("CamConfig.equity_volatilities", ("exposure",), ALL,
              {"none": lambda v: not v, "equities simulated": bool}),
    _labelled("CamConfig.correlations", ("exposure",), ALL, {"none": lambda v: not v, "correlated": bool}),
    _labelled("LgmConfig.reversion", ("exposure", "calibration"), ALL + ("cross_asset_model",),
              {"zero": lambda v: v == 0.0, "positive": lambda v: v > 0.0, "negative": lambda v: v < 0.0}),
    _labelled("LgmConfig.volatility", ("exposure", "calibration"), ALL + ("cross_asset_model",),
              {"constant": lambda v: not isinstance(v, Sigma), "piecewise": lambda v: isinstance(v, Sigma)}),
    _labelled("LgmConfig.calibration_expiries", ("exposure", "calibration"), ALL + ("cross_asset_model",),
              {"calibrated": bool, "not calibrated": lambda v: not v}),
    _against_default("LgmConfig.swap_index.fixed_tenor", ("exposure", "calibration"), ALL + ("cross_asset_model",)),
    _against_default("LgmConfig.swap_index.fixed_day_counter", ("exposure", "calibration"),
                     ALL + ("cross_asset_model",)),
    _against_default("LgmConfig.swap_index.index_tenor_months", ("exposure", "calibration"),
                     ALL + ("cross_asset_model",)),
    _choice("LgmConfig.solver", ("exposure", "calibration"), ALL + ("cross_asset_model",), SOLVERS),
    _against_default("PortfolioRequest.pfe_quantiles", ("exposure",), ALL),
]

_MARKET_RISK_SETTINGS = [
    _choice("ShockScenarios.source", ("market_risk",), ALL, SOURCES),
    _labelled("ShockScenarios.replicates", ("market_risk",), ALL, {"one": lambda v: v == 1, "several": lambda v: v > 1}),
    _against_default("ShockScenarios.bootstrap_replicates", ("market_risk",), ALL),
    _against_default("ShockScenarios.block_length", ("market_risk",), ALL),
    _against_default("ShockScenarios.bootstrap_seed", ("market_risk",), ALL),
    _against_default("MarketRiskRequest.quantiles", ("market_risk",), ALL),
    _against_default("MarketRiskRequest.batch_size", ("market_risk",), ALL),
    _against_default("MarketRiskRequest.base_currency", ("market_risk",), ALL),
    _labelled("RunConfig.base_currency", ("npv", "exposure", "sensitivities"), ALL,
              {"USD": lambda v: v in (None, "USD"), "another currency": lambda v: v not in (None, "USD")}),
]

SETTINGS: Tuple[Setting, ...] = tuple(_TRADE_SETTINGS + _ENGINE_SETTINGS + _SENSITIVITY_SETTINGS
                                      + _SIMULATION_SETTINGS + _MARKET_RISK_SETTINGS)


# ---------------------------------------------------------------------------------------------
# A test's cells
# ---------------------------------------------------------------------------------------------

def _engine_of(cfg, pricing: PricingConfig):
    """The group objects a trade's pricing settings are read from."""
    if isinstance(cfg, AmericanSwaptionConfig):
        return {"LgmSwaptionEngineConfig": pricing.american}
    if isinstance(cfg, BermudanSwaptionConfig):
        return {"LgmSwaptionEngineConfig": pricing.bermudan}
    if isinstance(cfg, SwaptionConfig):
        groups = {"PricingConfig": pricing}
        if pricing.jamshidian is not None:
            groups["JamshidianEngineConfig"] = pricing.jamshidian
        return groups
    return {}


def cells(analytic: str, trades: Iterable = (), pricing: PricingConfig = PricingConfig(), *,
          greeks: Optional[GreeksConfig] = None, simulation: Optional[CamConfig] = None,
          pfe_quantiles: Optional[Sequence[float]] = None, market_risk: Optional[MarketRiskRequest] = None,
          run: Optional[RunConfig] = None, models: Sequence[str] = ()) -> Tuple[Cell, ...]:
    """The cells a test of `analytic` compares: each of `trades` (or each of `models`, for
    calibration) under `pricing`, `greeks`, `simulation`, `pfe_quantiles`, `market_risk` (its
    pricing, scenarios and figures) and `run` (its reporting currency, USD unless given), as given."""
    if analytic not in ANALYTICS:
        raise ValueError(f"unknown analytic {analytic!r}; the analytics are {list(ANALYTICS)}")
    if market_risk is not None:
        trades, pricing = list(trades) or market_risk.trades, market_risk.pricing
    out = set()
    shared = {}
    if greeks is not None:
        shared.update(GreeksConfig=greeks, SensitivityConfig=greeks.sensitivity)
    if simulation is not None:
        shared["CamConfig"] = simulation
        models_ = list(simulation.ir.values())
        if len(models_) == 1:
            shared["LgmConfig"] = models_[0]
    if pfe_quantiles is not None:
        shared["PortfolioRequest"] = _PfeQuantiles(tuple(pfe_quantiles))
    if market_risk is not None:
        shared.update(MarketRiskRequest=market_risk, ShockScenarios=market_risk.scenarios)
    shared["RunConfig"] = run if run is not None else RunConfig()  # each trade reported in its own (USD) currency
    subjects = [(model, {}) for model in models] if analytic == "calibration" else \
        [(cfg.product, {type(cfg).__name__: cfg, **_engine_of(cfg, pricing)}) for cfg in trades]
    for product, groups in subjects:
        if product not in products_of(analytic):
            raise ValueError(f"{product!r} is not a column of {analytic!r}")
        out.add(Cell(analytic, product))
        groups = {**shared, **groups}
        for setting in SETTINGS:
            if analytic not in setting.analytics or product not in setting.products or setting.group not in groups:
                continue
            for label in _labels(setting, groups[setting.group]):
                out.add(Cell(analytic, product, setting.key, label))
    return tuple(sorted(out))


@dataclass(frozen=True)
class _PfeQuantiles:
    """The `PortfolioRequest` setting an exposure test runs with, for a test that compares the
    profiles without building the request."""
    pfe_quantiles: Tuple[float, ...]


def _labels(setting: Setting, group) -> List[str]:
    if setting.key == "BondConfig.coupon_schedule.payment_date":  # one value per period
        return sorted({setting.label(p.payment_date, group) for p in group.coupon_schedule})
    label = setting.label(setting.read(group), group)
    return [] if label is None else [label]


# ---------------------------------------------------------------------------------------------
# Reasons
# ---------------------------------------------------------------------------------------------

#: The kinds of reason a cell may have instead of a test.
CANNOT = "ORE cannot express it"
NOT_ORES = "not ORE's"
NOT_YET = "not compared yet"


@dataclass(frozen=True)
class Reason:
    kind: str
    text: str


_PRECISION = Reason(NOT_ORES, "the precision research axis (decision X-2): ORE parity is held at the float64 "
                              "default, and each format is measured against float64 by the evidence table "
                              "(compliance/precision-evidence.md)")
_SECOND_CURRENCY = Reason(NOT_YET, "a second currency: the oracles are one-currency until F-04 extends them")
_UNCERTAINTY = Reason(NOT_ORES, "a sampling-uncertainty setting (I-111, F-10): ORE has no error estimate of a "
                                "VaR; every scenario's P&L is compared, whatever the setting")
_JAMSHIDIAN = Reason(CANNOT, "ORE has no builder for QuantLib's Jamshidian engine, so no ORE analytic prices with "
                             "it; its value and every market-risk scenario are QuantLib's engine's, in "
                             "tests/test_jamshidian.py and tests/test_market_risk_ore_parity.py")
_GRID_AD = Reason(NOT_ORES, "a grid-priced option's AD Greeks are its price's derivatives, not ORE's 1bp secants "
                            "across the grid's kinks (I-78); tests/test_greeks_bermudan.py holds them to "
                            "central differences of the engine's own recalibrating price, and the result says so")

#: `(analytic, product, setting, value)` -> why the cell has no test; "*" matches anything.
REASONS: Dict[Tuple[str, str, str, str], Reason] = {
    ("*", "*", "SwapConfig.currency", "≠ USD"): _SECOND_CURRENCY,
    ("*", "*", "SwaptionConfig.currency", "≠ USD"): _SECOND_CURRENCY,
    ("*", "*", "BermudanSwaptionConfig.currency", "≠ USD"): _SECOND_CURRENCY,
    ("*", "*", "AmericanSwaptionConfig.currency", "≠ USD"): _SECOND_CURRENCY,
    ("*", "*", "BondConfig.currency", "≠ USD"): _SECOND_CURRENCY,
    ("*", "*", "RunConfig.base_currency", "another currency"): _SECOND_CURRENCY,
    ("*", "*", "MarketRiskRequest.base_currency", "≠ USD"): _SECOND_CURRENCY,
    ("exposure", "*", "CamConfig.base_currency", "another currency"): _SECOND_CURRENCY,
    ("exposure", "*", "CamConfig.fx_volatilities", "FX simulated"): _SECOND_CURRENCY,
    ("exposure", "*", "CamConfig.equity_volatilities", "equities simulated"): Reason(
        NOT_YET, "an equity in the simulation: no equity trade yet (I-18, F-04)"),
    ("exposure", "*", "CamConfig.correlations", "correlated"): _SECOND_CURRENCY,
    ("*", "*", "BondConfig.coupon_schedule.payment_date", "delayed"): Reason(
        CANNOT, "the oracle's bond pays each coupon at its period's end (ORE's leg data pays at a lag in days, not "
                "on a date per period), and a delayed last payment is valued wrong today (I-104)"),
    ("*", "*", "BondConfig.redemption_fraction", "≠ 1.0"): Reason(
        CANNOT, "the oracle books a coupon bond redeemed at par; a bill redeemed below par is a TraderX "
                "instrument priced by its own pricer (I-93)"),
    ("*", "european_swaption", "PricingConfig.european", "Jamshidian"): _JAMSHIDIAN,
    ("*", "european_swaption", "JamshidianEngineConfig.reversion", "*"): _JAMSHIDIAN,
    ("*", "european_swaption", "JamshidianEngineConfig.volatility", "*"): _JAMSHIDIAN,
    ("*", "european_swaption", "JamshidianEngineConfig.solver", "*"): _JAMSHIDIAN,
    ("sensitivities", "bermudan_swaption", "GreeksConfig.method", "AD"): _GRID_AD,
    ("sensitivities", "american_swaption", "GreeksConfig.method", "AD"): _GRID_AD,
    ("market_risk", "*", "ShockScenarios.replicates", "several"): _UNCERTAINTY,
    ("market_risk", "*", "ShockScenarios.bootstrap_replicates", "*"): _UNCERTAINTY,
    ("market_risk", "*", "ShockScenarios.block_length", "*"): _UNCERTAINTY,
    ("market_risk", "*", "ShockScenarios.bootstrap_seed", "*"): _UNCERTAINTY,
    ("market_risk", "*", "MarketRiskRequest.batch_size", "≠ 256"): Reason(
        NOT_ORES, "how many scenarios are revalued at once, a memory setting; the same P&L within the margin "
                  "(tests/test_market_risk.py)"),
    ("exposure", "*", "PortfolioRequest.pfe_quantiles", "= (0.95, 0.99)"): Reason(
        CANNOT, "ORE's exposure analytic reports one PFE quantile (`PfeQuantile`); each quantile is its own figure of "
                "one definition, compared at 0.95"),
    ("exposure", "*", "CamConfig.replicates", "≠ 1"): Reason(
        NOT_ORES, "the engine's own scrambles of its paths (X-1); on ORE's paths (L3) a cell is compared whatever "
                  "drew them"),
    ("calibration", "standalone_lgm", "", ""): Reason(
        NOT_YET, "the standalone bootstrap (`calibrate_lgm_sigma`) is the second LGM calibration, which retires or "
                 "moves to tests/support once the run request's calibration analytic replaces it (I-94); the "
                 "trades' and the cross-asset model's calibrations are ORE's"),
}


def reason_key(cell: Cell) -> Optional[Tuple[str, str, str, str]]:
    """The key of `REASONS` the cell's reason is under, the most specific first; None without one."""
    for key in ((cell.analytic, cell.product, cell.setting, cell.value), ("*", cell.product, cell.setting, cell.value),
                (cell.analytic, "*", cell.setting, cell.value), ("*", "*", cell.setting, cell.value),
                (cell.analytic, cell.product, cell.setting, "*"), ("*", cell.product, cell.setting, "*"),
                (cell.analytic, "*", cell.setting, "*"), ("*", "*", cell.setting, "*")):
        if key in REASONS:
            return key
    return None


def reason_for(cell: Cell) -> Optional[Reason]:
    key = reason_key(cell)
    return None if key is None else REASONS[key]


def required_cells() -> List[Cell]:
    """Every cell the matrix needs a test or a reason for (see the module docstring)."""
    out = []
    for analytic in ANALYTICS:
        for product in products_of(analytic):
            out.append(Cell(analytic, product))
            for setting in SETTINGS:
                if analytic in setting.analytics and product in setting.products:
                    out += [Cell(analytic, product, setting.key, label) for label in setting.labels]
    return out


# ---------------------------------------------------------------------------------------------
# Every field classified
# ---------------------------------------------------------------------------------------------

#: The public requests, trades and configurations whose dataclasses are walked.
ROOTS = (PortfolioRequest, MarketRiskRequest) + TRADE_CLASSES

_TERM = "a trade term, drawn by the sweeps (tests/support/parity_sweep.py)"
_IDENTITY = "identity: names the trade, or the date it is valued on (the market's as-of date)"
_MARKET = "market data, drawn by the sweeps: sloped, inverted, negative, flat and steep first segments"
_RUN = "what the request runs on, or which analytics and outputs it asks for: no number depends on it"

#: `Class.field` -> why it is an input, not a setting.
INPUTS: Dict[str, str] = {
    **{f"{cls.__name__}.{name}": _TERM for cls in TRADE_CLASSES
       for name in ("notional", "fixed_rate", "face_amount", "coupon_rate", "effective_date", "maturity_date",
                    "exercise_date", "exercise_dates", "first_exercise_date", "last_exercise_date")
       if name in {f.name for f in dataclasses.fields(cls)}},
    **{f"{cls.__name__}.{name}": _IDENTITY for cls in TRADE_CLASSES for name in ("trade_id", "evaluation_date")},
    "CouponPeriod.start_date": _TERM, "CouponPeriod.end_date": _TERM,
    "PortfolioRequest.market": _MARKET, "PortfolioRequest.trades": _RUN, "PortfolioRequest.config": _RUN,
    "PortfolioRequest.compute_greeks": _RUN, "PortfolioRequest.scenario_risk": _RUN,
    "PortfolioRequest.include_npv_cube": _RUN,
    "MarketRiskRequest.trades": _RUN, "MarketRiskRequest.market": _MARKET, "MarketRiskRequest.scenarios": _RUN,
    "MarketRiskRequest.pricing": _RUN, "MarketRiskRequest.precision": _RUN,
    "RunConfig.simulation": _RUN, "RunConfig.pricing": _RUN, "RunConfig.greeks": _RUN, "RunConfig.precision": _RUN,
    "PricingConfig.bermudan": _RUN, "PricingConfig.american": _RUN, "PricingConfig.jamshidian": _RUN,
    "GreeksConfig.sensitivity": _RUN,
    "ShockScenarios.factors": "the market's curves (every pillar a factor)",
    "ShockScenarios.shifts": "the scenarios themselves, drawn by the sweeps (Monte Carlo and historical)",
    "ShockScenarios.horizon_days": "the historical windows' length, drawn by the sweeps",
    "ShockScenarios.measure": "a label of the figures' measure; no number depends on it",
    "ShockScenarios.windows": "labels of the historical windows; no number depends on them",
    "CamConfig.dates": "the simulation grid, a term of the run (the parity tests' dates bracket every event)",
    "CamConfig.samples": "the path count: statistical (L4); on ORE's paths (L3) the cells are compared one by one",
    "CamConfig.seed": "the engine's random numbers (X-1); on ORE's paths (L3) none are drawn",
    "CamConfig.ir": _RUN,
    "LgmConfig.calibration_terms": "the basket's terms, with its expiries (`LgmConfig.calibration_expiries`)",
}
#: Classes whose every field has one reason.
CLASS_REASONS: Dict[type, Reason] = {Precision: _PRECISION, StagePrecision: _PRECISION}
#: Market data and model data, inputs as a whole.
INPUT_CLASSES = {Market: _MARKET, CurrencyMarket: _MARKET, ZeroCurveConfig: _MARKET, SwaptionVolSurface: _MARKET,
                 EquityMarket: "an equity's market data: no equity trade yet (I-18)", RateRiskFactors: _MARKET,
                 Sigma: "a piecewise volatility's times and values (`LgmConfig.volatility`)"}


def reachable_fields() -> List[Tuple[type, str, Any]]:
    """`(class, field, annotation)` of every dataclass reachable from `ROOTS`."""
    seen, out, todo = set(), [], list(ROOTS)
    while todo:
        cls = todo.pop()
        if cls in seen:
            continue
        seen.add(cls)
        hints = typing.get_type_hints(cls)
        for f in dataclasses.fields(cls):
            out.append((cls, f.name, hints.get(f.name)))
            todo += _dataclasses_in(hints.get(f.name))
    return out


def unclassified_fields() -> List[str]:
    """Every reachable field that is neither a setting's leaf, nor an input, nor in a class with
    one reason (see the module docstring)."""
    leaves = {_leaf(s.key) for s in SETTINGS} | {_leaf(key) for key in INPUTS}
    missing = []
    for cls, name, annotation in reachable_fields():
        if cls in CLASS_REASONS or cls in INPUT_CLASSES:
            continue
        if any(issubclass(cls, owner) and name == field for owner, field in leaves):
            continue
        if _dataclasses_in(annotation) and not any(c in INPUT_CLASSES or c in CLASS_REASONS
                                                   for c in _dataclasses_in(annotation)):
            continue  # a group of settings, whose own fields are walked
        missing.append(f"{cls.__name__}.{name}")
    return sorted(missing)


#: Each HTTP route of `engine.api.app.app` (its OpenAPI paths): the analytics it runs, space-separated
#: ("" for a route that computes no figure: status, jobs, artifacts, schemas). TraderX's `/eod/price` prices
#: Treasury bills and notes by its own pricers (I-93), the `npv` analytic's bond column once it moves onto the
#: engine.
ROUTES = {
    "/health": "", "/version": "", "/jobs/{job_id}": "", "/jobs/{job_id}/artifacts/{name}/{chunk}": "",
    "/portfolio/price": "npv exposure sensitivities",
    "/portfolio/market-risk": "market_risk",
    "/calibration/cam": "calibration",
    "/calibration/lgm": "calibration",
    "/eod/capabilities": "", "/eod/schemas/result": "", "/eod/schemas/capabilities": "",
    "/eod/price": "npv",
    "/eod/results/by-workload/{workload_key_value}": "", "/eod/attempts/{attempt_id}": "",
}


# ---------------------------------------------------------------------------------------------
# Collection and the published matrix
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Declared:
    """One test function's declaration: its cells, tolerance and tier."""
    test: str
    tolerance: str
    full_tier_only: bool
    cells: Tuple[Cell, ...]


class Collector:
    """A pytest plugin collecting every `ore_parity` marker of the collected items."""

    def __init__(self):
        self.declared: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def pytest_collection_modifyitems(self, items):
        for item in items:
            marks = list(item.iter_markers("ore_parity"))
            if not marks:
                continue
            test = item.nodeid.split("[", 1)[0]
            slow = item.get_closest_marker("slow") is not None
            for mark in marks:
                tolerance = mark.kwargs.get("tolerance", "")
                entry = self.declared.setdefault((test, tolerance), {"cells": set(), "fast": False})
                entry["fast"] |= not slow
                for group in mark.args:
                    entry["cells"].update(group)

    def result(self) -> List[Declared]:
        return [Declared(test, tolerance, not entry["fast"], tuple(sorted(entry["cells"])))
                for (test, tolerance), entry in sorted(self.declared.items())]


def collect_in_process(args: Sequence[str] = ("tests",)) -> List[Declared]:
    import pytest
    collector = Collector()
    code = pytest.main(["--collect-only", "-qq", "-p", "no:cacheprovider", *args], plugins=[collector])
    if code != 0:
        raise RuntimeError(f"collecting the tests failed (pytest exit code {code})")
    return collector.result()


def collect() -> List[Declared]:
    """Every declaration of the suite, collected in a fresh process (pytest's collection imports
    every test module)."""
    process = subprocess.run([sys.executable, "-m", "tests.support.ore_parity_matrix", "--json"], cwd=REPO,
                             capture_output=True, text=True, stdin=subprocess.DEVNULL,
                             env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    if process.returncode != 0:
        raise RuntimeError(f"collecting the parity tests failed:\n{process.stdout[-4000:]}\n{process.stderr[-4000:]}")
    payload = process.stdout[process.stdout.index("{\"declared\""):]
    return [Declared(d["test"], d["tolerance"], d["full_tier_only"], tuple(Cell(*c) for c in d["cells"]))
            for d in json.loads(payload)["declared"]]


def uncovered(declared: Sequence[Declared]) -> List[Cell]:
    """Required cells with neither a test nor a reason."""
    covered = {cell for d in declared for cell in d.cells}
    return [cell for cell in required_cells() if cell not in covered and reason_for(cell) is None]


def undeclarable(declared: Sequence[Declared]) -> List[str]:
    """Declared cells the matrix has no row for (a setting or value it does not list)."""
    required = set(required_cells())
    return sorted({f"{d.test}: {cell}" for d in declared for cell in d.cells if cell not in required})


def render(declared: Sequence[Declared]) -> str:
    """The published matrix (see the module docstring), between `BEGIN` and `END`."""
    tests = {d.test: d for d in declared}
    numbers = {test: i + 1 for i, test in enumerate(sorted(tests))}
    by_cell: Dict[Cell, List[Declared]] = {}
    for d in declared:
        for cell in d.cells:
            by_cell.setdefault(cell, []).append(d)
    reasons: Dict[Reason, int] = {}
    lines = [BEGIN, "", "*Generated by `python -m tests.support.ore_parity_matrix --write` from the parity tests' "
             "`ore_parity` markers; do not edit by hand. tests/test_ore_parity_matrix.py fails if it is stale, or "
             "if a public setting or analytic has neither a test nor a reason.*", "",
             "Each cell names the tests that compare it with ORE (**T**n, listed below with their tolerances; "
             "*full* where only the full tier's sweep reaches it), or the reason it is not compared (**R**n); "
             "· where the setting does not apply to the product. The first row of each table is the product "
             "compared at all."]
    for analytic, title in ANALYTICS.items():
        columns = products_of(analytic)
        names = MODELS if analytic == "calibration" else PRODUCTS
        lines += ["", f"#### {title} (`{analytic}`)", "",
                  "| Setting | Value | " + " | ".join(names[c] for c in columns) + " |",
                  "|---|---|" + "---|" * len(columns)]
        rows = [("", "")] + [(s.key, label) for s in SETTINGS if analytic in s.analytics for label in s.labels]
        for key, label in rows:
            entries = []
            setting = next((s for s in SETTINGS if s.key == key), None)
            for product in columns:
                if setting is not None and product not in setting.products:
                    entries.append("·")
                    continue
                cell = Cell(analytic, product, key, label)
                found = by_cell.get(cell, [])
                if found:
                    refs = sorted({numbers[d.test] for d in found})
                    full = all(d.full_tier_only for d in found)
                    entries.append(", ".join(f"T{n}" for n in refs) + (" *full*" if full else ""))
                    continue
                reason = reason_for(cell)
                if reason is None:
                    entries.append("**none**")
                else:
                    entries.append(f"R{reasons.setdefault(reason, len(reasons) + 1)}")
            setting_text = f"`{key}`" if key else "*compared*"
            lines.append(f"| {setting_text} | {label} | " + " | ".join(entries) + " |")
    lines += ["", "**Tests.**", ""]
    for test, n in sorted(numbers.items(), key=lambda kv: kv[1]):
        d = tests[test]
        tier = " (full tier only)" if d.full_tier_only else ""
        lines.append(f"- **T{n}** `{test}`{tier}: {d.tolerance}")
    lines += ["", "**Reasons.**", ""]
    for reason, n in sorted(reasons.items(), key=lambda kv: kv[1]):
        lines.append(f"- **R{n}** ({reason.kind}) {reason.text}.")
    lines += ["", END]
    return "\n".join(lines)


def published() -> str:
    text = DOC.read_text(encoding="utf-8")
    start, end = text.index(BEGIN), text.index(END) + len(END)
    return text[start:end]


def write(declared: Sequence[Declared]) -> None:
    text = DOC.read_text(encoding="utf-8")
    start, end = text.index(BEGIN), text.index(END) + len(END)
    DOC.write_text(text[:start] + render(declared) + text[end:], encoding="utf-8")


def main(argv: Sequence[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if "--json" in argv:
        # Through the package's module, whose `Cell` the test modules' markers hold, not `__main__`'s.
        from tests.support.ore_parity_matrix import collect_in_process as collect_here
        declared = collect_here()
        print(json.dumps({"declared": [{"test": d.test, "tolerance": d.tolerance, "full_tier_only": d.full_tier_only,
                                        "cells": [dataclasses.astuple(c) for c in d.cells]} for d in declared]}))
        return 0
    if "--write" in argv:
        declared = collect()
        write(declared)
        missing = uncovered(declared)
        print(f"wrote the matrix to {DOC.relative_to(REPO)}; {len(missing)} cells have neither a test nor a reason")
        for cell in missing:
            print("   ", cell)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
