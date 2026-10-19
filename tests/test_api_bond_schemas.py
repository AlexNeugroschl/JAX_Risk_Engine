"""
The bond's HTTP surface (`engine.api.requests.BondTradeSchema`): discriminated-union
routing, the request/response round trip, serialization of Greeks (I-25), and the exposure
analytic.
"""
import ORE
import numpy as np
import pytest
from pydantic import ValidationError

from engine.api.requests import BondTradeSchema, RunRequestSchema
from engine.api.results import GreeksSchema, run_result_document
from engine.instruments.treasury import BondConfig, BondPricingError
from engine.pricing.cube import value_today
from engine.run import run

CURVE = {"times": [0.0, 1.0, 2.0, 5.0, 10.0, 30.0], "rates": [0.03, 0.03, 0.032, 0.035, 0.038, 0.04]}
MARKET = {"asof": "2025-06-02", "currencies": {"USD": {"discount_curve": CURVE}}}
EXPOSURE = {"models": {"ir": {"USD": {"model": "HullWhite", "reversion": 0.03, "volatility": 0.01}}},
            "simulation": {"dates": ["2025-12-02", "2026-06-02"], "samples": 32}}

BILL_JSON = {"trade_type": "bond", "trade_id": "bill", "face_amount": 100000.0, "maturity_date": "2025-12-15"}

NOTE_JSON = {
    "trade_type": "bond", "trade_id": "note", "face_amount": 100000.0, "maturity_date": "2026-06-15",
    "coupon_rate": 0.04, "accrual_day_count": "ACT/ACT (ICMA)",
    "coupon_schedule": [
        {"start_date": "2024-12-15", "end_date": "2025-06-15", "payment_date": "2025-06-15"},
        {"start_date": "2025-06-15", "end_date": "2025-12-15", "payment_date": "2025-12-15"},
        {"start_date": "2025-12-15", "end_date": "2026-06-15", "payment_date": "2026-06-15"},
    ],
}


def request_json(trades, analytics=("npv",), **config) -> dict:
    return {"market": MARKET, "portfolio": trades, "analytics": list(analytics), "config": config}


def parse(trades, analytics=("npv",), **config):
    return RunRequestSchema.model_validate(request_json(trades, analytics, **config)).to_dataclass()


def document(trades, analytics=("npv",), **config):
    """The wire result of a run, as the engine worker serializes it."""
    schema = RunRequestSchema.model_validate(request_json(trades, analytics, **config))
    return run_result_document(run(schema.to_dataclass()), schema, "run", "/runs/run/artifacts")[0]


class TestBondSchemaRouting:
    """The union routes `trade_type: "bond"`."""

    def test_a_bond_parses_from_json(self):
        parsed = RunRequestSchema.model_validate(request_json([BILL_JSON]))
        assert isinstance(parsed.portfolio[0], BondTradeSchema)

    def test_a_bond_converts_to_its_dataclass(self):
        bond = parse([BILL_JSON]).portfolio[0]
        assert isinstance(bond, BondConfig) and bond.trade_id == "bill"

    def test_an_unknown_trade_type_is_still_rejected(self):
        with pytest.raises(ValidationError):
            RunRequestSchema.model_validate(request_json([dict(BILL_JSON, trade_type="cdo")]))

    def test_a_bond_carrying_a_curve_is_refused(self):
        """A trade names its currency; a curve of its own is refused, not silently dropped."""
        with pytest.raises(ValidationError, match="initial_zero_curve"):
            RunRequestSchema.model_validate(request_json([dict(BILL_JSON, initial_zero_curve=CURVE)]))

    def test_the_bond_is_valued_on_the_markets_date(self):
        assert parse([BILL_JSON]).portfolio[0].evaluation_date == ORE.Date(2, 6, 2025)


class TestBondSchemaConversionIsFaithful:
    """The JSON gives the same price as the dataclass directly."""

    def test_a_bill_prices_the_same_over_the_schema(self):
        request = parse([BILL_JSON])
        direct = BondConfig(face_amount=100000.0, maturity_date=ORE.Date(15, 12, 2025),
                            evaluation_date=ORE.Date(2, 6, 2025), trade_id="bill")
        assert value_today(request.portfolio, request.market, "USD") == value_today([direct], request.market, "USD")

    def test_a_note_schedule_survives_the_round_trip(self):
        bond = parse([NOTE_JSON]).portfolio[0]
        assert len(bond.coupon_schedule) == 3
        assert bond.coupon_schedule[0].start_date == ORE.Date(15, 12, 2024)
        assert bond.coupon_schedule[-1].end_date == ORE.Date(15, 6, 2026)

    def test_an_absent_payment_date_falls_back_to_the_end_date(self):
        no_payment = dict(NOTE_JSON, coupon_schedule=[{"start_date": p["start_date"], "end_date": p["end_date"]}
                                                      for p in NOTE_JSON["coupon_schedule"]])
        assert parse([no_payment]).portfolio[0].coupon_schedule[0].payment() == ORE.Date(15, 6, 2025)

    def test_a_malformed_bond_is_refused_at_conversion(self):
        """A matured bond is refused."""
        with pytest.raises(BondPricingError):
            parse([dict(BILL_JSON, maturity_date="2020-01-01")])


class TestBondGreeksSerializeOverHttp:
    """I-25: scalar Greeks serialize. A 0-d array's `.tolist()` is a bare float, so a
    `from_dataclass` that iterated it unconditionally raised `TypeError: 'float' object is not
    iterable`."""

    def test_a_scalar_greek_serializes(self):
        schema = GreeksSchema.from_dataclass({"delta": np.asarray(-5.284)})
        assert schema.values["delta"] == pytest.approx([-5.284])

    def test_a_scalar_greek_becomes_a_one_element_list(self):
        schema = GreeksSchema.from_dataclass({"delta": np.asarray(-5.284)})
        assert isinstance(schema.values["delta"], list) and len(schema.values["delta"]) == 1

    def test_a_vector_greek_is_unchanged(self):
        schema = GreeksSchema.from_dataclass({"delta": np.asarray([1.0, 2.0, 3.0])})
        assert schema.values["delta"] == pytest.approx([1.0, 2.0, 3.0])

    def test_theta_is_still_a_scalar_field(self):
        assert GreeksSchema.from_dataclass({"theta": np.asarray(8.088)}).theta == pytest.approx(8.088)

    def test_a_full_bond_result_serializes_end_to_end(self):
        """Price a bond with Greeks, then serialize the whole result and round-trip it."""
        serialized = document([BILL_JSON], analytics=("npv", "sensitivities"))
        assert serialized.trades[0].greeks is not None
        assert sum(serialized.trades[0].greeks.values["delta:discount:USD"]) < 0
        assert "delta:discount:USD" in serialized.model_dump_json()


class TestTheExposureAnalyticOverHttp:
    """Today's value alone, or with the bond on every path."""

    def test_without_exposure_the_result_has_no_exposure(self):
        serialized = document([BILL_JSON])
        assert serialized.exposure is None and [t.exposure for t in serialized.trades] == [None]

    def test_the_value_today_is_reported_without_exposure(self):
        serialized = document([BILL_JSON])
        assert serialized.npv > 90_000.0 and [t.npv for t in serialized.trades] == [serialized.npv]

    def test_a_bond_is_priced_on_every_path_over_http(self):
        """I-24: before 2026-10-01 this was refused (`ScenarioPricingNotSupported`)."""
        serialized = document([NOTE_JSON], analytics=("npv", "exposure"), **EXPOSURE)
        assert serialized.exposure is not None and serialized.trades[0].exposure is not None


class TestOpenApiSchemaIncludesTheBond:
    """The capability contract advertises the bond type."""

    def test_the_bond_appears_in_the_request_schema(self):
        assert "BondTradeSchema" in RunRequestSchema.model_json_schema().get("$defs", {})

    def test_the_analytics_are_documented(self):
        assert "analytics" in RunRequestSchema.model_json_schema()["properties"]
