"""
Every setting the engine has reaches the HTTP API (decision A-2, I-56).

From each entry point an HTTP route serves (`price_portfolio`'s `PortfolioRequest`,
`run_market_risk`'s `MarketRiskRequest`, the scenario generators and `calibrate_cam`), the walk
follows every field's type into every configuration dataclass it can hold. Each dataclass met must
have the request schema that mirrors it (`MIRRORS`), and each of its fields must be a field of
that schema by the same name, renamed, or exempt with the reason it has no field (`exempt`); a
function's parameters likewise. A new setting without an API field fails here, and so does a new
configuration type without a schema, or an exemption or rename that names a field no longer there.

Names only: a field's type on the wire is the schema's (an ISO string for an `ORE.Date`, a list
for a tuple), and each schema converts with `.to_dataclass()`, which the route and worker tests
exercise.
"""
import dataclasses
import inspect
import typing

import pytest

from engine.api import requests as wire
from engine.calibration.cam import calibrate_cam
from engine.calibration.ore_lgm import SwapIndexConventions
from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig
from engine.instruments.treasury import BondConfig, CouponPeriod
from engine.market_data.market import CurrencyMarket, EquityMarket, Market, SwaptionVolSurface, ZeroCurveConfig
from engine.models.lgm import Sigma
from engine.precision import Precision, StagePrecision
from engine.risk.market import MarketRiskRequest, RateRiskFactors, ShockScenarios, historical_scenarios, monte_carlo_scenarios
from engine.run import (
    CamConfig, GreeksConfig, HullWhiteConfig, JamshidianEngineConfig, LgmConfig, LgmSwaptionEngineConfig,
    PortfolioRequest, PricingConfig, RunConfig, SensitivityConfig,
)

#: Every trade is valued on the market's date: the request has one date, not one per trade.
ON_THE_MARKET_DATE = {"evaluation_date": "every trade is valued on market.asof"}


class Rename(str):
    """A field the schema carries under another name."""


#: engine type -> (the schema or schemas mirroring it, {field: reason it has no field, or Rename}).
MIRRORS = {
    PortfolioRequest: (wire.MarketPortfolioRequestSchema, {"config": "flattened: RunConfig's fields are the request's"}),
    RunConfig: (wire.MarketPortfolioRequestSchema, {}),
    CamConfig: (wire.CamConfigSchema, {}),
    LgmConfig: (wire.LgmConfigSchema, {}),
    HullWhiteConfig: (wire.HullWhiteConfigSchema, {}),
    Sigma: (wire.PiecewiseVolatilitySchema, {}),
    SwapIndexConventions: (wire.SwapIndexConventionsSchema, {}),
    PricingConfig: (wire.PricingConfigSchema, {}),
    LgmSwaptionEngineConfig: (wire.LgmEngineSchema, {}),
    JamshidianEngineConfig: (wire.JamshidianEngineSchema, {}),
    GreeksConfig: (wire.GreeksConfigSchema, {}),
    SensitivityConfig: (wire.SensitivityConfigSchema, {}),
    Precision: (wire.PrecisionSchema, {}),
    StagePrecision: (wire.StagePrecisionSchema, {}),
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
    MarketRiskRequest: (wire.MarketRiskRequestSchema, {}),
    ShockScenarios: ((wire.MonteCarloScenariosSchema, wire.HistoricalScenariosSchema), {
        "shifts": "drawn from the covariance (monte_carlo_scenarios) or observed in the history "
                  "(historical_scenarios), whose every parameter is a field (FUNCTIONS)",
        "measure": "historical-forecast, the measure of every shock scenario",
        "windows": Rename("dates")}),
}

#: Types a request never gives: the engine builds them from fields it does give.
BUILT = {RateRiskFactors: "the market's curves `scenarios.factors` names (MarketRiskRequestSchema)"}

#: Entry points whose parameters are the schema's fields: function -> (schema, exemptions as MIRRORS').
FUNCTIONS = {
    monte_carlo_scenarios: (wire.MonteCarloScenariosSchema, {}),
    historical_scenarios: (wire.HistoricalScenariosSchema, {}),
    calibrate_cam: (wire.CamCalibrationRequestSchema, {"lgm_configs": Rename("ir")}),
}

ROOTS = (PortfolioRequest, MarketRiskRequest)


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


def _fields(schema_or_schemas):
    """Every field name of the schema, or of any schema of a group (alternatives of one union,
    such as the scenario sources: a field is reachable if one of them has it)."""
    group = schema_or_schemas if isinstance(schema_or_schemas, tuple) else (schema_or_schemas,)
    return set().union(*(schema.model_fields for schema in group))


def _missing(names, mirror):
    """The names with no field in the mirroring schema(s) nor an exemption; and stale exemptions."""
    schema, exempt = mirror
    fields, missing = _fields(schema), []
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
def test_every_parameter_of_an_entry_point_has_an_api_field(function):
    missing, stale = _missing(list(inspect.signature(function).parameters), FUNCTIONS[function])
    assert not missing, f"{function.__name__}: parameters without an API field: {missing}"
    assert not stale, f"{function.__name__}: exemptions naming no parameter: {stale}"


def test_every_mirror_is_reached_from_a_route():
    """MIRRORS lists nothing the routes cannot reach (a type left behind by a removed setting)."""
    assert set(MIRRORS) <= _reachable()


def test_the_walk_finds_a_setting_without_a_field():
    """The test can fail: a field added to an engine type is reported missing."""
    @dataclasses.dataclass(frozen=True)
    class Extended(SensitivityConfig):
        shift_scheme: str = "Forward"

    missing, _ = _missing([f.name for f in dataclasses.fields(Extended)], MIRRORS[SensitivityConfig])
    assert missing == ["shift_scheme"]
