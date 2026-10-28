"""
Every setting the engine has reaches the HTTP API, and its value arrives where it belongs
(decision A-2, I-56, I-88).

**Names.** From the run request (`engine.run.RunRequest`, the one way in, I-87) the walk follows
every field's type into every configuration dataclass it can hold, and the parameters of the
functions a request's fields are given to (`historical_scenarios`, `calibrate_cam`). Each
dataclass met must have the request schema that mirrors it (`MIRRORS`), and each of its fields
must be a field of that schema by the same name, renamed, or exempt with the reason it has no
field; a function's parameters likewise. A new setting without an API field fails here, and so
does a new configuration type without a schema, or an exemption or rename that names a field no
longer there.

**Values** (I-88). For every field of every schema, the test builds the schema with that field
set to a value other than its default (`NON_DEFAULT`, else one derived from the field's type),
converts it as the API does (`.to_dataclass()`), and compares the engine object with the
conversion of the schema at its defaults: the value must change the mirrored setting and nothing
but it (and the settings `NON_DEFAULT` names as companions, or a field's `EFFECTS`). A field the
schema accepts and then drops, or sends to the wrong setting, fails here; so does a field with no
non-default value to try. The run request's own mapping onto the engine's entry points
(`portfolio_request`, `market_risk_request`, `RunRequestConfig.cam`) is checked field by field
on a request that sets every setting.
"""
import dataclasses
import inspect
import typing
from typing import Any, Dict, Tuple

import numpy as np
import ORE
import pytest
from pydantic import BaseModel

from engine.api import requests as wire
from engine.calibration.cam import calibrate_cam
from engine.calibration.lgm import SwapIndexConventions
from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig
from engine.instruments.treasury import BondConfig, CouponPeriod
from engine.market_data.market import CurrencyMarket, EquityMarket, Market, SwaptionVolSurface, ZeroCurveConfig
from engine.models.lgm import Sigma
from engine.precision import ExecutionPolicy, InternalPrecision, Precision, StagePrecision, ProductOptions
from engine.risk.market import MonteCarloScenarios, RateRiskFactors, ShockScenarios, historical_scenarios
from engine.run import (
    CamConfig, ExposureConfig, GreeksConfig, HullWhiteConfig, JamshidianEngineConfig, LgmConfig,
    LgmSwaptionEngineConfig, MarketRiskConfig, ModelsConfig, PortfolioRequest, PricingConfig, RunConfig, RunRequest,
    RunRequestConfig, SensitivityConfig, SimulationParameters,
)
from engine.run.run_request import market_risk_request, portfolio_request
from engine.risk.market import MarketRiskRequest

#: Every trade is valued on the market's date: the request has one date, not one per trade.
ON_THE_MARKET_DATE = {"evaluation_date": "every trade is valued on market.asof"}


class Rename(str):
    """A field the schema carries under another name."""


#: engine type -> (the schema mirroring it, {field: reason it has no field, or Rename}).
MIRRORS = {
    RunRequest: (wire.RunRequestSchema, {"include_npv_cube": Rename("outputs")}),
    RunRequestConfig: (wire.RunRequestConfigSchema, {}),
    ModelsConfig: (wire.ModelsSchema, {}),
    SimulationParameters: (wire.SimulationSchema, {}),
    LgmConfig: (wire.LgmConfigSchema, {}),
    HullWhiteConfig: (wire.HullWhiteConfigSchema, {}),
    Sigma: (wire.PiecewiseVolatilitySchema, {}),
    SwapIndexConventions: (wire.SwapIndexConventionsSchema, {}),
    PricingConfig: (wire.PricingConfigSchema, {}),
    LgmSwaptionEngineConfig: (wire.LgmEngineSchema, {}),
    JamshidianEngineConfig: (wire.JamshidianEngineSchema, {}),
    GreeksConfig: (wire.SensitivitySchema, {"sensitivity": "flattened: SensitivityConfig's fields are the section's"}),
    SensitivityConfig: (wire.SensitivitySchema, {}),
    ExposureConfig: (wire.ExposureSchema, {}),
    MarketRiskConfig: (wire.MarketRiskSchema, {}),
    MonteCarloScenarios: (wire.MonteCarloScenariosSchema, {}),
    ShockScenarios: (wire.HistoricalScenariosSchema, {
        "shifts": "observed in the history (historical_scenarios, whose every parameter is a field)",
        "measure": "historical-forecast, the measure of every shock scenario",
        "windows": Rename("dates"),
        "replicates": "independent scrambles are Monte Carlo scenarios' (MonteCarloScenarios); a history has none"}),
    Precision: (wire.PrecisionSchema, {}),
    ExecutionPolicy: (wire.ExecutionPolicySchema, {}),
    InternalPrecision: (wire.InternalPrecisionSchema, {}),
    StagePrecision: (wire.StagePrecisionSchema, {}),
    ProductOptions: (wire.ProductOptionsSchema, {}),
    Market: (wire.MarketSchema, {}),
    CurrencyMarket: (wire.CurrencyMarketSchema, {}),
    EquityMarket: (wire.EquityMarketSchema, {}),
    SwaptionVolSurface: (wire.SwaptionVolSurfaceSchema, {}),
    ZeroCurveConfig: (wire.ZeroCurveConfigSchema,
                      {"provenance": "the TraderX path's metadata, which pricing never reads (engine.market_data.market)"}),
    SwapConfig: (wire.SwapTradeSchema, ON_THE_MARKET_DATE),
    SwaptionConfig: (wire.EuropeanTradeSchema, ON_THE_MARKET_DATE),
    BermudanSwaptionConfig: (wire.BermudanTradeSchema, ON_THE_MARKET_DATE),
    AmericanSwaptionConfig: (wire.AmericanTradeSchema, ON_THE_MARKET_DATE),
    BondConfig: (wire.BondTradeSchema, ON_THE_MARKET_DATE),
    CouponPeriod: (wire.CouponPeriodSchema, {}),
}

#: Types a request never gives: the engine builds them from fields it does give.
BUILT = {RateRiskFactors: "the market's curves `scenarios.factors` names (`_ScenariosSchema.risk_factors`)"}

#: Functions a request's fields are given to: function -> (schema, exemptions as MIRRORS').
FUNCTIONS = {
    historical_scenarios: (wire.HistoricalScenariosSchema, {}),
    calibrate_cam: (wire.ModelsSchema, {"market": "the run request's market", "lgm_configs": Rename("ir"),
                                        "tolerance": Rename("bootstrap_tolerance"),
                                        "precision": "the run request's config.precision"}),
}

ROOTS = (RunRequest,)


def _dataclasses_in(annotation):
    """The dataclass types an annotation can hold (through Optional, Union, containers, aliases)."""
    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        return {annotation}
    return set().union(*(_dataclasses_in(a) for a in typing.get_args(annotation)))


def _reachable():
    """Every dataclass the roots' fields (and the functions' parameters) can hold, transitively."""
    seen, todo = set(), list(ROOTS)
    for function in FUNCTIONS:
        hints = typing.get_type_hints(function)
        todo += [t for name in inspect.signature(function).parameters for t in _dataclasses_in(hints.get(name))]
    while todo:
        cls = todo.pop()
        if cls in seen or cls in BUILT:
            continue
        seen.add(cls)
        todo += [t for hint in typing.get_type_hints(cls).values() for t in _dataclasses_in(hint)]
    return seen


def _missing(names, mirror):
    """The names with no field in the mirroring schema nor an exemption; and stale exemptions."""
    schema, exempt = mirror
    fields, missing = set(schema.model_fields), []
    for name in names:
        target = exempt.get(name)
        if isinstance(target, Rename):
            if target not in fields:
                missing.append(f"{name} (renamed {target!r}, which the schema lacks)")
        elif target is None and name not in fields:
            missing.append(name)
    stale = sorted(set(exempt) - set(names))
    return missing, stale


def test_every_configuration_type_has_a_schema():
    unmirrored = sorted(cls.__qualname__ for cls in _reachable() if cls not in MIRRORS)
    assert not unmirrored, (f"configuration types with no request schema: {unmirrored}; add one to "
                            f"engine/api/requests.py and to MIRRORS")


@pytest.mark.parametrize("cls", sorted(MIRRORS, key=lambda c: c.__qualname__), ids=lambda c: c.__qualname__)
def test_every_field_of_a_configuration_type_has_an_api_field(cls):
    missing, stale = _missing([f.name for f in dataclasses.fields(cls)], MIRRORS[cls])
    assert not missing, f"{cls.__qualname__}: settings without an API field in {MIRRORS[cls][0]}: {missing}"
    assert not stale, f"{cls.__qualname__}: exemptions naming no field: {stale}"


@pytest.mark.parametrize("function", list(FUNCTIONS), ids=lambda f: f.__name__)
def test_every_parameter_of_a_function_has_an_api_field(function):
    missing, stale = _missing(list(inspect.signature(function).parameters), FUNCTIONS[function])
    assert not missing, f"{function.__name__}: parameters without an API field: {missing}"
    assert not stale, f"{function.__name__}: exemptions naming no parameter: {stale}"


def test_every_mirror_is_reached_from_the_run_request():
    """MIRRORS lists nothing the run request cannot reach (a type left behind by a removed setting)."""
    assert set(MIRRORS) <= _reachable()


def test_the_walk_finds_a_setting_without_a_field():
    """The test can fail: a field added to an engine type is reported missing."""
    @dataclasses.dataclass(frozen=True)
    class Extended(SensitivityConfig):
        shift_scheme: str = "Forward"

    missing, _ = _missing([f.name for f in dataclasses.fields(Extended)], MIRRORS[SensitivityConfig])
    assert missing == ["shift_scheme"]


# --- values: every field's value arrives in the engine (I-88) ------------------------------------

ASOF = "2026-07-30"
CURVE = {"times": [0.0, 1.0, 5.0], "rates": [0.01, 0.02, 0.03]}
OTHER_CURVE = {"times": [0.0, 2.0, 5.0], "rates": [0.01, 0.02, 0.04]}
VOLS = {"option_tenors": ["1Y", "2Y"], "swap_tenors": ["1Y", "5Y"], "vols": [[0.01, 0.01], [0.01, 0.01]]}
MARKET = {"asof": ASOF, "currencies": {"USD": {"discount_curve": CURVE, "index_curves": {"USD-SIMINDEX-6M": CURVE}},
                                       "EUR": {"discount_curve": CURVE}}}
_MARKET = wire.MarketSchema.model_validate(MARKET).to_dataclass()
LABELS = RateRiskFactors.from_market(_MARKET).labels()
COVARIANCE = (np.eye(len(LABELS)) * 1e-6).tolist()
HISTORY = (0.02 + np.cumsum(np.full((40, len(LABELS)), 1e-4), axis=0)).tolist()
MONTE_CARLO = {"source": "monte-carlo", "factors": LABELS, "covariance": COVARIANCE, "horizon_days": 10,
               "num_scenarios": 8}
SWAP = {"notional": 1e6, "fixed_rate": 0.03, "payer": True, "effective_date": "2026-08-03",
        "maturity_date": "2031-08-03"}


class With:
    """A non-default value set together with companion fields it needs (each also allowed to
    change its own setting)."""

    def __init__(self, value, **companions):
        self.value, self.companions = value, companions


def _trade(schema):
    return lambda s: wire._trades([s], _MARKET)[0]


#: schema -> (the fields every instance needs, how the API converts one).
SCHEMAS: Dict[type, Tuple[dict, Any]] = {
    wire.ZeroCurveConfigSchema: (CURVE, lambda s: s.to_dataclass()),
    wire.SwaptionVolSurfaceSchema: (VOLS, lambda s: s.to_dataclass()),
    wire.CurrencyMarketSchema: ({"discount_curve": CURVE}, lambda s: s.to_dataclass()),
    wire.EquityMarketSchema: ({"currency": "USD", "spot": 100.0}, lambda s: s.to_dataclass()),
    wire.MarketSchema: (MARKET, lambda s: s.to_dataclass()),
    wire.SwapIndexConventionsSchema: ({}, lambda s: s.to_dataclass()),
    wire.PiecewiseVolatilitySchema: ({"times": [1.0], "values": [0.01, 0.02]}, lambda s: s.to_dataclass()),
    wire.LgmConfigSchema: ({"reversion": 0.03}, lambda s: s.to_dataclass()),
    wire.HullWhiteConfigSchema: ({"model": "HullWhite", "reversion": 0.03}, lambda s: s.to_dataclass()),
    wire.ModelsSchema: ({"ir": {"USD": {"reversion": 0.03}}}, lambda s: s.to_dataclass()),
    wire.SimulationSchema: ({"dates": ["2027-01-30"]}, lambda s: s.to_dataclass()),
    wire.LgmEngineSchema: ({}, lambda s: s.to_dataclass()),
    wire.JamshidianEngineSchema: ({"reversion": 0.03, "volatility": 0.01}, lambda s: s.to_dataclass()),
    wire.PricingConfigSchema: ({}, lambda s: s.to_dataclass()),
    wire.SensitivitySchema: ({}, lambda s: s.to_dataclass()),
    wire.ExposureSchema: ({}, lambda s: s.to_dataclass()),
    wire.StagePrecisionSchema: ({}, lambda s: s.to_dataclass()),
    wire.ProductOptionsSchema: ({}, lambda s: s.to_dataclass()),
    wire.PrecisionSchema: ({}, lambda s: s.to_dataclass()),
    wire.ExecutionPolicySchema: ({}, lambda s: s.to_dataclass()),
    wire.InternalPrecisionSchema: ({}, lambda s: s.to_dataclass()),
    wire.SwapTradeSchema: (SWAP, _trade(wire.SwapTradeSchema)),
    wire.EuropeanTradeSchema: ({"notional": 1e6, "fixed_rate": 0.03, "payer": True, "swap_tenor": "2Y",
                                "forward_start": "1Y"}, _trade(wire.EuropeanTradeSchema)),
    wire.BermudanTradeSchema: ({**SWAP, "effective_date": "2027-02-03", "maturity_date": "2033-02-03",
                                "exercise_dates": ["2028-02-03"]}, _trade(wire.BermudanTradeSchema)),
    wire.AmericanTradeSchema: ({**SWAP, "effective_date": "2027-02-03", "maturity_date": "2033-02-03",
                                "first_exercise_date": "2027-02-03", "last_exercise_date": "2031-02-03"},
                               _trade(wire.AmericanTradeSchema)),
    wire.BondTradeSchema: ({"face_amount": 1000.0, "maturity_date": "2028-07-30"}, _trade(wire.BondTradeSchema)),
    wire.CouponPeriodSchema: ({"start_date": "2026-07-30", "end_date": "2027-01-30"}, lambda s: s.to_dataclass()),
    wire.MonteCarloScenariosSchema: (MONTE_CARLO, lambda s: s.to_dataclass(_MARKET)),
    wire.HistoricalScenariosSchema: ({"source": "historical", "factors": LABELS, "history": HISTORY,
                                      "horizon_days": 10}, lambda s: s.to_dataclass(_MARKET)),
    wire.MarketRiskSchema: ({"scenarios": MONTE_CARLO}, lambda s: s.to_dataclass(_MARKET)),
    wire.RunRequestConfigSchema: ({}, lambda s: s.to_dataclass(_MARKET)),
    wire.RunRequestSchema: ({"market": MARKET, "analytics": ["npv"]}, lambda s: s.to_dataclass()),
}

#: (schema, field) -> a value other than the default, where the field's type does not give one.
NON_DEFAULT = {
    (wire.RunRequestConfigSchema, "execution"): {"backend": "cpu"},
    (wire.PrecisionSchema, "internal"): {"reductions": "float32"},
    (wire.StagePrecisionSchema, "product"): With({"shared_terms": 2},
        compute="float16", accumulate="float32", storage="float32"),
    (wire.PrecisionSchema, "execution"): {"mode": "allow_emulation"},
    (wire.ZeroCurveConfigSchema, "times"): [0.0, 2.0, 5.0],
    (wire.ZeroCurveConfigSchema, "rates"): [0.01, 0.02, 0.04],
    (wire.SwaptionVolSurfaceSchema, "option_tenors"): ["1Y", "3Y"],
    (wire.SwaptionVolSurfaceSchema, "swap_tenors"): ["1Y", "4Y"],
    (wire.SwaptionVolSurfaceSchema, "vols"): [[0.01, 0.02], [0.01, 0.01]],
    (wire.CurrencyMarketSchema, "discount_curve"): OTHER_CURVE,
    (wire.CurrencyMarketSchema, "index_curves"): {"USD-SIMINDEX-6M": CURVE},
    (wire.CurrencyMarketSchema, "swaption_vols"): VOLS,
    (wire.EquityMarketSchema, "currency"): "EUR",
    (wire.EquityMarketSchema, "dividend_curve"): CURVE,
    (wire.MarketSchema, "asof"): "2026-07-31",
    (wire.MarketSchema, "currencies"): {"USD": {"discount_curve": OTHER_CURVE}},
    (wire.MarketSchema, "fx_spots"): {"EURUSD": 1.1},
    (wire.MarketSchema, "equities"): {"SP5": {"currency": "USD", "spot": 100.0}},
    (wire.SwapIndexConventionsSchema, "fixed_tenor"): "6M",
    (wire.SwapIndexConventionsSchema, "fixed_day_counter"): "30/360",
    (wire.SwapIndexConventionsSchema, "index_tenor_months"): 3,
    (wire.PiecewiseVolatilitySchema, "times"): [2.0],
    (wire.PiecewiseVolatilitySchema, "values"): [0.01, 0.03],
    (wire.LgmConfigSchema, "volatility"): {"times": [1.0], "values": [0.01, 0.02]},
    (wire.LgmConfigSchema, "calibration_expiries"): With(["1Y"], calibration_terms=["5Y"]),
    (wire.LgmConfigSchema, "calibration_terms"): With(["5Y"], calibration_expiries=["1Y"]),
    (wire.LgmConfigSchema, "swap_index"): {"fixed_tenor": "6M"},
    (wire.HullWhiteConfigSchema, "volatility"): 0.02,
    (wire.HullWhiteConfigSchema, "calibration_expiries"): With(["1Y"], calibration_terms=["5Y"]),
    (wire.HullWhiteConfigSchema, "calibration_terms"): With(["5Y"], calibration_expiries=["1Y"]),
    (wire.HullWhiteConfigSchema, "swap_index"): {"fixed_tenor": "6M"},
    (wire.ModelsSchema, "ir"): {"USD": {"reversion": 0.04}},
    (wire.ModelsSchema, "fx_volatilities"): {"EUR": 0.1},
    (wire.ModelsSchema, "equity_volatilities"): {"SP5": 0.2},
    (wire.ModelsSchema, "correlations"): [{"factor1": "IR:USD", "factor2": "IR:EUR", "value": 0.5}],
    (wire.SimulationSchema, "dates"): ["2027-02-28"],
    (wire.SimulationSchema, "replicates"): 2,
    (wire.SimulationSchema, "curve_tenors"): ["1Y", "5Y"],
    (wire.LgmEngineSchema, "reference_calibration_grid"): "200,3M",
    (wire.LgmEngineSchema, "swap_index"): {"fixed_tenor": "6M"},
    (wire.PricingConfigSchema, "european"): With("Jamshidian", jamshidian={"reversion": 0.03, "volatility": 0.01}),
    (wire.PricingConfigSchema, "jamshidian"): With({"reversion": 0.03, "volatility": 0.01}, european="Jamshidian"),
    (wire.PricingConfigSchema, "bermudan"): {"n_per_std": 12},
    (wire.PricingConfigSchema, "american"): {"n_per_std": 12},
    (wire.SensitivitySchema, "curve_tenors"): ["1Y", "2Y"],
    (wire.ExposureSchema, "pfe_quantiles"): [0.9],
    (wire.StagePrecisionSchema, "storage"): "float32",
    (wire.StagePrecisionSchema, "compute"): With("float32", accumulate="float32", storage="float32"),
    (wire.StagePrecisionSchema, "accumulate"): With("float32", compute="float32", storage="float32"),
    (wire.StagePrecisionSchema, "compute_rounding"): With("stochastic", compute="bfloat16", accumulate="float32",
                                                          storage="float32"),
    (wire.PrecisionSchema, "simulation"): {"storage": "float32"},
    (wire.PrecisionSchema, "market"): {"storage": "float32"},
    (wire.PrecisionSchema, "pricing"): {"storage": "float32"},
    (wire.PrecisionSchema, "by_product"): {"swap": {"storage": "float32"}},
    (wire.PrecisionSchema, "by_trade"): {"t": {"storage": "float32"}},
    (wire.PrecisionSchema, "rounding"): With("stochastic", pricing={"storage": "bfloat16"}),
    (wire.PrecisionSchema, "offset"): With("none", pricing={"storage": "bfloat16"}),
    (wire.SwapTradeSchema, "trade_id"): "named",
    (wire.SwapTradeSchema, "currency"): "EUR",
    (wire.SwapTradeSchema, "effective_date"): "2026-09-03",
    (wire.SwapTradeSchema, "maturity_date"): "2032-08-03",
    (wire.SwapTradeSchema, "swap_tenor"): With("6Y", effective_date=None, maturity_date=None),
    (wire.SwapTradeSchema, "index_tenor_months"): 3,
    (wire.SwapTradeSchema, "accrual_day_count"): "ACT/ACT (ICMA)",
    (wire.SwapTradeSchema, "fixings"): {"2026-07-28": 0.03},
    (wire.EuropeanTradeSchema, "trade_id"): "named",
    (wire.EuropeanTradeSchema, "currency"): "EUR",
    (wire.EuropeanTradeSchema, "effective_date"): With("2027-09-03", maturity_date="2029-09-03",
                                                       exercise_date="2027-08-30", swap_tenor=None, forward_start=None),
    (wire.EuropeanTradeSchema, "maturity_date"): With("2029-09-03", effective_date="2027-09-03",
                                                      exercise_date="2027-08-30", swap_tenor=None, forward_start=None),
    (wire.EuropeanTradeSchema, "exercise_date"): With("2027-08-30", effective_date="2027-09-03",
                                                      maturity_date="2029-09-03", swap_tenor=None, forward_start=None),
    (wire.EuropeanTradeSchema, "exercise_lag_days"): 4,
    (wire.EuropeanTradeSchema, "swap_tenor"): "3Y",
    (wire.EuropeanTradeSchema, "forward_start"): "2Y",
    (wire.EuropeanTradeSchema, "index_tenor_months"): 3,
    (wire.BermudanTradeSchema, "trade_id"): "named",
    (wire.BermudanTradeSchema, "currency"): "EUR",
    (wire.BermudanTradeSchema, "effective_date"): "2027-03-03",
    (wire.BermudanTradeSchema, "maturity_date"): "2034-02-03",
    (wire.BermudanTradeSchema, "swap_tenor"): With("6Y", effective_date=None, maturity_date=None),
    (wire.BermudanTradeSchema, "index_tenor_months"): 3,
    (wire.BermudanTradeSchema, "exercise_dates"): ["2029-02-05"],
    (wire.BermudanTradeSchema, "fixings"): {"2026-07-28": 0.03},
    (wire.AmericanTradeSchema, "trade_id"): "named",
    (wire.AmericanTradeSchema, "currency"): "EUR",
    (wire.AmericanTradeSchema, "effective_date"): "2027-03-03",
    (wire.AmericanTradeSchema, "maturity_date"): "2034-02-03",
    (wire.AmericanTradeSchema, "swap_tenor"): With("6Y", effective_date=None, maturity_date=None),
    (wire.AmericanTradeSchema, "index_tenor_months"): 3,
    (wire.AmericanTradeSchema, "first_exercise_date"): "2028-02-03",
    (wire.AmericanTradeSchema, "last_exercise_date"): "2032-02-03",
    (wire.AmericanTradeSchema, "fixings"): {"2026-07-28": 0.03},
    (wire.BondTradeSchema, "trade_id"): "named",
    (wire.BondTradeSchema, "currency"): "EUR",
    (wire.BondTradeSchema, "maturity_date"): "2029-07-30",
    (wire.BondTradeSchema, "coupon_rate"): With(0.05, coupon_schedule=[{"start_date": "2026-07-30",
                                                                        "end_date": "2028-07-30"}]),
    (wire.BondTradeSchema, "coupon_schedule"): With([{"start_date": "2026-07-30", "end_date": "2028-07-30"}],
                                                    coupon_rate=0.05),
    (wire.BondTradeSchema, "redemption_fraction"): 0.5,
    (wire.BondTradeSchema, "accrual_day_count"): "ACT/365",
    (wire.CouponPeriodSchema, "start_date"): "2026-08-30",
    (wire.CouponPeriodSchema, "end_date"): "2027-02-28",
    (wire.CouponPeriodSchema, "payment_date"): "2027-02-01",
    (wire.MonteCarloScenariosSchema, "factors"): With(LABELS[:3], covariance=(np.eye(3) * 1e-6).tolist()),
    (wire.MonteCarloScenariosSchema, "covariance"): (np.eye(len(LABELS)) * 2e-6).tolist(),
    (wire.MonteCarloScenariosSchema, "replicates"): 2,
    (wire.HistoricalScenariosSchema, "factors"): With(LABELS[:3], history=np.asarray(HISTORY)[:, :3].tolist()),
    (wire.HistoricalScenariosSchema, "history"): (np.asarray(HISTORY) * 2).tolist(),
    (wire.HistoricalScenariosSchema, "dates"): [f"d{i}" for i in range(40)],
    (wire.HistoricalScenariosSchema, "block_length"): 11,
    (wire.MarketRiskSchema, "scenarios"): {**MONTE_CARLO, "seed": 43},
    (wire.MarketRiskSchema, "quantiles"): [0.95],
    (wire.RunRequestConfigSchema, "base_currency"): "EUR",
    (wire.RunRequestConfigSchema, "models"): {"ir": {"USD": {"reversion": 0.03}}},
    (wire.RunRequestConfigSchema, "simulation"): {"dates": ["2027-01-30"]},
    (wire.RunRequestConfigSchema, "pricing"): {"bermudan": {"n_per_std": 12}},
    (wire.RunRequestConfigSchema, "sensitivity"): {"method": "AD"},
    (wire.RunRequestConfigSchema, "exposure"): {"pfe_quantiles": [0.9]},
    (wire.RunRequestConfigSchema, "market_risk"): {"scenarios": MONTE_CARLO},
    (wire.RunRequestConfigSchema, "precision"): {"pricing": {"storage": "float32"}},
    (wire.RunRequestSchema, "market"): {**MARKET, "asof": "2026-07-31"},
    (wire.RunRequestSchema, "portfolio"): [{"trade_type": "swap", **SWAP}],
    (wire.RunRequestSchema, "analytics"): ["exposure"],
    (wire.RunRequestSchema, "config"): {"base_currency": "EUR"},
    (wire.RunRequestSchema, "outputs"): {"npv_cube": "none"},
}

#: (schema, field) -> the engine setting it arrives in, where that is not the field of its name.
ARRIVES_IN = {
    (wire.SensitivitySchema, "curve_tenors"): "sensitivity.curve_tenors",
    (wire.SensitivitySchema, "curve_shift"): "sensitivity.curve_shift",
    (wire.SensitivitySchema, "vol_shift"): "sensitivity.vol_shift",
    (wire.SensitivitySchema, "theta_days"): "sensitivity.theta_days",
    (wire.HistoricalScenariosSchema, "history"): "shifts",
    (wire.HistoricalScenariosSchema, "dates"): "windows",
    (wire.RunRequestSchema, "outputs"): "include_npv_cube",
    (wire.SwapTradeSchema, "swap_tenor"): "maturity_date",
    (wire.EuropeanTradeSchema, "swap_tenor"): "maturity_date",
    (wire.EuropeanTradeSchema, "forward_start"): "exercise_date",
    (wire.EuropeanTradeSchema, "exercise_lag_days"): "exercise_date",
    (wire.BermudanTradeSchema, "swap_tenor"): "maturity_date",
    (wire.AmericanTradeSchema, "swap_tenor"): "maturity_date",
}
#: (schema, field) -> the other settings its value moves, as the engine derives them.
EFFECTS = {
    (wire.PrecisionSchema, "profile"): ("simulation", "market", "pricing", "internal"),
    (wire.RunRequestConfigSchema, "execution"): ("precision.execution",),
    (wire.EuropeanTradeSchema, "swap_tenor"): ("effective_date",),
    (wire.EuropeanTradeSchema, "forward_start"): ("effective_date", "maturity_date"),
    (wire.EuropeanTradeSchema, "exercise_lag_days"): ("effective_date", "maturity_date"),
    (wire.HistoricalScenariosSchema, "factors"): ("shifts",),
    (wire.HistoricalScenariosSchema, "horizon_days"): ("shifts",),
}
#: Fields with no engine setting of their own, with the reason.
NOT_SETTINGS = {
    (wire.LgmConfigSchema, "model"): "selects the schema: LgmConfig, or HullWhiteConfig for `HullWhite`",
    (wire.HullWhiteConfigSchema, "model"): "selects the schema",
    (wire.MonteCarloScenariosSchema, "source"): "selects the schema",
    (wire.HistoricalScenariosSchema, "source"): "selects the schema",
    (wire.RunRequestSchema, "idempotency_key"): "the job queue's (I-89, tests/test_engine_worker.py)",
    (wire.RunRequestSchema, "keep"): "the job queue's retention (I-76, tests/test_engine_worker.py)",
    **{(schema, "trade_type"): "selects the schema"
       for schema in (wire.SwapTradeSchema, wire.EuropeanTradeSchema, wire.BermudanTradeSchema,
                      wire.AmericanTradeSchema, wire.BondTradeSchema)},
}


def _derived(schema, name, base):
    """A value of the field other than its default (or, for a required field, than `base`'s),
    from its type; None when the type gives none (then `NON_DEFAULT` must)."""
    info = schema.model_fields[name]
    default = base[name] if info.is_required() else info.get_default(call_default_factory=True)
    annotation = info.annotation
    choices = typing.get_args(annotation) if typing.get_origin(annotation) is typing.Literal else ()
    if choices:
        return next(c for c in choices if c != default)
    if annotation is bool:
        return not default
    if annotation is int and isinstance(default, int) and not isinstance(default, bool):
        return default + 1
    if annotation is float and isinstance(default, (int, float)) and not isinstance(default, bool):
        return default * 2.0 + 0.001
    return None


def _leaves(value, path=""):
    """`{path: comparable value}` of every leaf of an engine object."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out = {}
        for f in dataclasses.fields(value):
            out.update(_leaves(getattr(value, f.name), f"{path}.{f.name}" if path else f.name))
        return out
    if isinstance(value, typing.Mapping):
        out = {path: ("mapping", len(value))}
        for key, item in value.items():
            out.update(_leaves(item, f"{path}[{key!r}]"))
        return out
    if isinstance(value, (list, tuple)):
        out = {path: ("sequence", len(value))}
        for i, item in enumerate(value):
            out.update(_leaves(item, f"{path}[{i}]"))
        return out
    if isinstance(value, ORE.Date):
        return {path: value.serialNumber()}
    if isinstance(value, ORE.Period):
        return {path: str(value)}
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        array = np.asarray(value)
        return {path: (array.shape, tuple(array.ravel().tolist()))}
    return {path: value}


def _under(path, prefix):
    return path == prefix or path.startswith(prefix + ".") or path.startswith(prefix + "[")


def _cases():
    for schema in SCHEMAS:
        for name in schema.model_fields:
            yield pytest.param(schema, name, id=f"{schema.__name__}.{name}")


@pytest.mark.parametrize("schema, name", list(_cases()))
def test_every_fields_value_arrives_in_its_setting_and_only_there(schema, name):
    """I-88 (see the module docstring)."""
    if (schema, name) in NOT_SETTINGS:
        return
    base, convert = SCHEMAS[schema]
    value = NON_DEFAULT.get((schema, name))
    if value is None:
        value = _derived(schema, name, base)
    assert value is not None, f"{schema.__name__}.{name}: give NON_DEFAULT a value other than its default"
    companions = {}
    if isinstance(value, With):
        value, companions = value.value, value.companions
    fields = {**base, **{k: v for k, v in companions.items()}, name: value}
    fields = {k: v for k, v in fields.items() if v is not None}
    before = _leaves(convert(schema.model_validate(base)))
    after = _leaves(convert(schema.model_validate(fields)))
    changed = {p for p in set(before) | set(after) if before.get(p, "absent") != after.get(p, "absent")}
    target = ARRIVES_IN.get((schema, name), name)
    assert any(_under(p, target) for p in changed), f"{schema.__name__}.{name}={value!r} did not reach {target}"
    allowed = (target, *(ARRIVES_IN.get((schema, c), c) for c in companions), *EFFECTS.get((schema, name), ()))
    stray = sorted(p for p in changed if not any(_under(p, a) for a in allowed))
    assert not stray, f"{schema.__name__}.{name} also moved {stray}"


def test_a_schema_that_drops_a_field_fails_the_value_check(monkeypatch):
    """The test can fail (red first, I-88): a conversion that drops `vol_shift` is caught."""
    original = wire.SensitivitySchema.to_dataclass

    def dropping(self):
        return original(self.model_copy(update={"vol_shift": 1e-4}))

    monkeypatch.setattr(wire.SensitivitySchema, "to_dataclass", dropping)
    with pytest.raises(AssertionError, match="did not reach sensitivity.vol_shift"):
        test_every_fields_value_arrives_in_its_setting_and_only_there(wire.SensitivitySchema, "vol_shift")


def test_every_schema_of_the_wire_is_checked():
    """`SCHEMAS` covers every model of the request but those that only group others' fields."""
    models = {cls for cls in vars(wire).values()
              if isinstance(cls, type) and issubclass(cls, BaseModel) and cls.__module__ == wire.__name__
              and not cls.__name__.startswith("_")}
    grouping = {wire.CorrelationSchema: "one entry of ModelsSchema.correlations",
                wire.OutputsSchema: "RunRequestSchema.outputs: the job's, but `npv_cube` (include_npv_cube)"}
    assert models - set(SCHEMAS) == set(grouping)


# --- the run request onto the engine's entry points ---------------------------------------------

def _rich_request() -> RunRequest:
    """A run request that sets every setting the entry points read to a value other than its
    default."""
    body = {"market": MARKET, "portfolio": [{"trade_type": "swap", "trade_id": "s", **SWAP}],
            "analytics": ["npv", "exposure", "sensitivities", "market_risk"],
            "config": {
                "base_currency": "USD",
                "models": {"ir": {"USD": {"reversion": 0.04, "volatility": 0.02}, "EUR": {"reversion": 0.02}},
                           "fx_volatilities": {"EUR": 0.11}, "equity_volatilities": {},
                           "correlations": [{"factor1": "IR:USD", "factor2": "IR:EUR", "value": 0.4}],
                           "bootstrap_tolerance": 2e-4},
                "simulation": {"dates": ["2027-01-30", "2027-07-30"], "samples": 32, "seed": 9, "replicates": 2,
                               "curve_tenors": ["1Y", "5Y"], "swaption_vol_decay": "ConstantVariance"},
                "pricing": {"bermudan": {"n_per_std": 12}},
                "sensitivity": {"method": "AD", "curve_tenors": ["1Y", "3Y"], "curve_shift": 2e-4},
                "exposure": {"pfe_quantiles": [0.9]},
                "market_risk": {"scenarios": MONTE_CARLO, "quantiles": [0.95], "batch_size": 64},
                "precision": {"pricing": {"storage": "float32"}, "paired_fraction": 0.5},
            },
            "outputs": {"npv_cube": "none"}}
    return wire.RunRequestSchema.model_validate(body).to_dataclass()


def test_the_cross_asset_model_is_the_models_on_the_simulations_grid():
    request = _rich_request()
    models, sim = request.config.models, request.config.simulation
    assert request.config.cam() == CamConfig(
        dates=sim.dates, base_currency="USD", ir=dict(models.ir), fx_volatilities=dict(models.fx_volatilities),
        equity_volatilities=dict(models.equity_volatilities), correlations=dict(models.correlations),
        curve_tenors=sim.curve_tenors, samples=sim.samples, seed=sim.seed, swaption_vol_decay=sim.swaption_vol_decay,
        bootstrap_tolerance=models.bootstrap_tolerance, replicates=sim.replicates)
    assert {f.name for f in dataclasses.fields(CamConfig)} == (
        {f.name for f in dataclasses.fields(ModelsConfig)} | {f.name for f in dataclasses.fields(SimulationParameters)}
        | {"base_currency"})


def test_the_portfolio_analytics_are_one_price_portfolio_of_every_setting():
    request = _rich_request()
    config = request.config
    assert portfolio_request(request) == PortfolioRequest(
        market=request.market, trades=list(request.portfolio),
        config=RunConfig(simulation=config.cam(), pricing=config.pricing, greeks=config.sensitivity,
                         precision=config.precision, base_currency="USD"),
        pfe_quantiles=config.exposure.pfe_quantiles, compute_greeks=True, scenario_risk=True, include_npv_cube=False)
    assert {f.name for f in dataclasses.fields(PortfolioRequest)} == {
        "market", "trades", "config", "pfe_quantiles", "compute_greeks", "scenario_risk", "include_npv_cube"}
    assert {f.name for f in dataclasses.fields(RunConfig)} == {
        "simulation", "pricing", "greeks", "precision", "base_currency", "execution"}


def test_the_market_risk_analytic_is_one_run_market_risk_of_every_setting():
    request = _rich_request()
    config, risk = request.config, request.config.market_risk
    scenarios = risk.scenarios.draw()
    built = market_risk_request(request, scenarios)
    assert built == MarketRiskRequest(trades=list(request.portfolio), market=request.market, scenarios=scenarios,
                                      pricing=config.pricing, quantiles=risk.quantiles, precision=config.precision,
                                      batch_size=risk.batch_size, base_currency="USD")
    assert {f.name for f in dataclasses.fields(MarketRiskRequest)} == {
        "trades", "market", "scenarios", "pricing", "quantiles", "precision", "batch_size", "base_currency"}
